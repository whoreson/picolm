// backend_gpu_host_core.cu - GPU initialization, memory, tensor, matmul, expert MLP
#include "backend_gpu_kernels.cuh"

static void gpu_dispatch_print(const char *name) {
    static char seen[64][64] = {0};
    for (int i = 0; i < 64; i++) {
        if (seen[i][0] == '\0') { snprintf(seen[i], 64, "%s", name); fprintf(stderr, "[GPU] kernel: %s\n", name); return; }
        if (strcmp(seen[i], name) == 0) return;
    }
}

gpu_device_ctx_t g_gpu_ctx[PICOLM_GPU_MAX_DEVICES];
int g_nctx;

/* Mutex protecting g_gpu_ctx scratch buffer resize (reserve/reserve_pinned). */
#ifdef _WIN32
SRWLOCK g_resize_mutex = SRWLOCK_INIT;
#else
pthread_mutex_t g_resize_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

/* Grow-only device buffer helper, mirroring model.c ssm_mm_alloc /
 * the existing g_ssm_mm_*_dev pattern used for the decode-path GPU
 * matmul scratch buffers. The batched SSM prefill wrappers below
 * (conv1d_batch, l2norm_batch, vecdot_batch, prefill_gated_norm,
 * chunked_recurrence) originally did a fresh gpuMalloc/gpuFree on
 * EVERY call -- since ssm_prefill_layer runs once per layer, that
 * is 30-40 cudaMalloc/cudaFree pairs per layer, ~1500-2000 per
 * prompt for a typical model. Profiling showed cudaMalloc alone
 * costing multiple seconds (individual calls up to several hundred
 * ms, consistent with allocator fragmentation from repeated alloc/
 * free of varying sizes) and cudaMemcpy costing tens of seconds
 * (individual calls blocking for seconds, consistent with waiting
 * behind queued work rather than pure PCIe transfer time) --
 * exactly what shows up in perf as unresolved 0x... addresses
 * inside the driver. Buffers allocated via this helper are grown
 * as needed and never freed until the device changes, eliminating
 * that overhead from the hot path entirely. */

int ssm_batch_scratch_ensure(void **buf, size_t *cap, size_t need) {
    if (*buf && *cap >= need) return 1;
    if (*buf) { gpuDeviceSynchronize(); gpuFree(*buf); *buf = NULL; *cap = 0; }
    if (!gpu_ok(gpuMalloc(buf, need), "ssm batch scratch grow")) {
        { size_t fb=0,tb=0; gpuMemGetInfo(&fb,&tb);
          fprintf(stderr,"[GPU] ssm_batch_scratch_ensure OOM: need=%zu KB, gpu_free=%.1f MB, gpu_total=%.1f MB\n",
              need/1024, fb/(1024.0*1024), tb/(1024.0*1024)); }
        *cap = 0; return 0;
    }
    *cap = need;
    return 1;
}

int reserve(float **ptr, size_t *cap, size_t bytes) {
    gpu_mutex_lock();
    if (*cap >= bytes) { gpu_mutex_unlock(); return 1; }
    if (*ptr) { gpuDeviceSynchronize(); gpuFree(*ptr); }
    *ptr = NULL; *cap = 0;
    /* Use device memory for scratch buffers.
     * cpu writes via explicit cudaMemcpyAsync H2D, gpu reads from device mem,
     * gpu writes to device mem, then cpu reads via explicit cudaMemcpyAsync D2H.
     * This avoids cudaMallocManaged page-fault overhead and err=1 issues
     * seen on GB10 (sm_121) with CUDA 13.0 where managed memory mixed with
     * device memory in kernel arguments causes cudaErrorInvalidValue on first
     * kernel launches. */
    gpuError_t err = gpuMalloc(ptr, bytes);
    if (!gpu_ok(err, "scratch allocation")) {
        { size_t fb=0,tb=0; gpuMemGetInfo(&fb,&tb);
          fprintf(stderr,"[GPU] reserve OOM: need=%zu KB, gpu_free=%.1f MB, gpu_total=%.1f MB\n",
              bytes/1024, fb/(1024.0*1024), tb/(1024.0*1024)); }
        gpu_mutex_unlock();
        return 0;
    }
    /* Zero-init: GB10 cudaMalloc returns stale VRAM contents causing
     * non-determinism in kernels that read before writing all elements. */
    if (*ptr) gpuMemset(*ptr, 0, bytes);
    *cap = bytes;
    gpu_mutex_unlock();
    return 1;
}

PICOLM_UNUSED int reserve_pinned(float **ptr, size_t *cap, size_t bytes) {
    gpu_mutex_lock();
    if (*cap >= bytes) { gpu_mutex_unlock(); return 1; }
    if (*ptr) gpuFreeHost(*ptr);
    *ptr = NULL; *cap = 0;
    if (!gpu_ok(gpuMallocHost(ptr, bytes), "pinned staging allocation")) {
        gpu_mutex_unlock();
        return 0;
    }
    *cap = bytes;
    gpu_mutex_unlock();
    return 1;
}

/* Reserve int8_t device buffer (for Q8_0 quantized activations) */
int reserve_i8(int8_t **ptr, size_t *cap, size_t bytes) {
    gpu_mutex_lock();
    if (*cap >= bytes) { gpu_mutex_unlock(); return 1; }
    /* MUST synchronize with GPU before freeing, otherwise a kernel
     * launched on ctx->stream may still be reading from the old buffer
     * when gpuFree releases it. This causes non-deterministic memory
     * corruption that varies across runs. */
    if (*ptr) { gpuDeviceSynchronize(); gpuFree(*ptr); }
    *ptr = NULL; *cap = 0;
    gpuError_t err = gpuMalloc(ptr, bytes);
    if (!gpu_ok(err, "int8 scratch allocation")) {
        { size_t fb=0,tb=0; gpuMemGetInfo(&fb,&tb);
          fprintf(stderr,"[GPU] reserve_i8 OOM: need=%zu KB, gpu_free=%.1f MB, gpu_total=%.1f MB\n",
              bytes/1024, fb/(1024.0*1024), tb/(1024.0*1024)); }
        gpu_mutex_unlock();
        return 0;
    }
    *cap = bytes;
    gpu_mutex_unlock();
    return 1;
}

/* ---- Public API ---- */

extern "C"
int picolm_gpu_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > PICOLM_GPU_MAX_DEVICES) return 0;
#ifndef __HIP__
    /* CUDA 13 / GB10 (sm_121): multi-engine scheduling can reorder/overlap
     * H2D, kernel, and D2H work even though our code issues everything
     * synchronously per-call, causing run-to-run nondeterminism at temp=0
     * (see gpu_nondeterminism.md -- isolated empirically, fixes it 8/8 with
     * no measured perf cost since we already sync per call). Must be set
     * before the first CUDA driver call, so this has to be the very first
     * thing in init. overwrite=0: respect an explicit user-set value. */
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1", 0);
#endif
    if (!gpu_ok(gpuGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            fprintf(stderr, "[GPU] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0; return 0;
        }
        if (find_ctx(device)) {
            fprintf(stderr, "[GPU] duplicate device %d\n", device);
            g_nctx = 0; return 0;
        }
        gpu_device_ctx_t *ctx = &g_gpu_ctx[g_nctx];
        memset(ctx, 0, sizeof(*ctx));
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        gpuDeviceProp prop;
        if (!gpu_ok(gpuGetDeviceProperties(&prop, device), "device properties")) {
            g_nctx = 0; return 0;
        }
        ctx->compute_major = prop.major;
        ctx->compute_minor = prop.minor;
#ifdef __HIP__
        ctx->has_imma = 0;  /* HIP: no NVIDIA PTX IMMA support */
#else
        ctx->has_imma = (prop.major >= 8);  /* NVIDIA: Turing+ sm_80 */
#endif
        if (!gpu_ok(gpuStreamCreateWithFlags(&ctx->stream,
#ifdef __HIP__
                                              hipStreamNonBlocking
#else
                                              cudaStreamNonBlocking
#endif
                                              ), "stream creation")) {
            g_nctx = 0; return 0;
        }
        g_nctx++;
        fprintf(stderr, "[GPU] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                device, prop.name, prop.totalGlobalMem / 1e9,
                prop.major, prop.minor);
    }
    return 1;
}

extern "C"
void picolm_gpu_shutdown(void) {
    /* Free KV cache allocations */
    picolm_gpu_kv_free();
    /* Free pipeline buffers */
    picolm_gpu_pipeline_free();
    for (int i = 0; i < g_nctx; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!select_ctx(ctx)) continue;
        /* Free cached RMSNorm weights */
        for (int j = 0; j < ctx->rmsnorm_w_n; j++) {
            if (ctx->rmsnorm_w_dev[j]) gpuFree(ctx->rmsnorm_w_dev[j]);
        }
        ctx->rmsnorm_w_n = 0;
        if (ctx->x) gpuFree(ctx->x);
        if (ctx->y) gpuFree(ctx->y);
        if (ctx->gate) gpuFree(ctx->gate);
        if (ctx->up) gpuFree(ctx->up);
        if (ctx->q8_xq) gpuFree(ctx->q8_xq);
        if (ctx->q8_xd) gpuFree(ctx->q8_xd);
        if (ctx->host_x) gpuFreeHost(ctx->host_x);
        if (ctx->host_y) gpuFreeHost(ctx->host_y);
        if (ctx->stream) gpuStreamDestroy(ctx->stream);
        ctx->x = ctx->y = ctx->gate = ctx->up = NULL;
        ctx->q8_xq = NULL;
        ctx->q8_xd = NULL;
        ctx->host_x = ctx->host_y = NULL;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->q8_xq_cap = ctx->q8_xd_cap = 0;
        ctx->host_x_cap = ctx->host_y_cap = 0;
        ctx->stream = NULL;
    }
    g_nctx = 0;
}

/* Detect if we're on a unified memory SoC (Grace-Blackwell, Apple Silicon, etc.) */
static PICOLM_UNUSED int is_unified_memory(void) {
#ifdef __HIP__
    return 0; /* HIP: treat as discrete GPU (conservative) */
#else
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    /* On Grace-Blackwell, unifiedAddressing is true and the device has no
     * separate PCIe link width (it's chip-to-chip). Check for integrated
     * memory by looking at the bus type. */
    return prop.unifiedAddressing && prop.l2CacheSize > 0;
#endif
}

extern "C"
int picolm_gpu_device_count(void) { return g_nctx; }

extern "C"
int picolm_gpu_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_gpu_ctx[index].device : -1;
}

extern "C"
int picolm_gpu_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    size_t fb = 0, tb = 0;
    return gpu_ok(gpuMemGetInfo(&fb, &tb), "memory info") && (*free_bytes = fb, *total_bytes = tb, 1);
}

/* Map GGUF_TYPE to block size and values per block.
 * F32 (0), F16 (1), BF16 (30): no blocks, each is an individual element. */
static int gguf_block_size(gguf_type_t qtype) {
    switch (qtype) {
    case 0:  return 0;    /* F32: no blocks */
    case 1:  return 0;    /* F16: no blocks */
    case 30: return 0;    /* BF16: no blocks */
    case 2:  return 18;   /* Q4_0: 18 bytes per 32 values */
    case 8:  return GPU_BLOCK_Q8_0_SIZE; /* Q8_0: 34 bytes per 32 values */
    case 10: return 84;    /* Q2_K: 84 bytes per 256 values */
    case 11: return 110;   /* Q3_K: 110 bytes per 256 values */
    case 12: return GPU_BLOCK_Q4_K_SIZE;  /* Q4_K: 144 bytes per 256 values */
    case 13: return GPU_BLOCK_Q5_K_SIZE;  /* Q5_K: 176 bytes per 256 values */
    case 14: return GPU_BLOCK_Q6_K_SIZE;  /* Q6_K: 210 bytes per 256 values */
    case 41: return 18;   /* Q1_0: 18 bytes per 128 values */
    case 42: return 34;   /* Q2_0: 34 bytes per 128 values */
    default: return 0;
    }
}

extern "C"
int picolm_gpu_tensor_upload(void **tensor,
                              const void *weights,
                              gguf_type_t qtype, int I, int O, int device) {
    if (!tensor || !weights || I < 1 || O < 1) return 0;
    picolm_gpu_tensor_t **tp = (picolm_gpu_tensor_t **)tensor;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;
    int bs = gguf_block_size(qtype);
    if (!bs && qtype != 0 && qtype != 1 && qtype != 30) return 0;
    if (*tp) return 1; /* idempotent */

    /* Compute row bytes */
    size_t row_bytes;
    int vals_per_block;
    if (qtype == 0) {
        row_bytes = (size_t)I * sizeof(float);
        vals_per_block = 1;
    } else if (qtype == 1 || qtype == 30) {
        row_bytes = (size_t)I * sizeof(uint16_t);
        vals_per_block = 1;
    } else if (qtype == 10 || qtype == 11 || qtype == 12 || qtype == 13 || qtype == 14) {
        vals_per_block = 256;
        row_bytes = (size_t)((I + vals_per_block - 1) / vals_per_block) * bs;
    } else if (qtype == 41 || qtype == 42) {
        vals_per_block = 128;
        row_bytes = (size_t)((I + vals_per_block - 1) / vals_per_block) * bs;
    } else {
        vals_per_block = 32;
        row_bytes = (size_t)((I + vals_per_block - 1) / vals_per_block) * bs;
    }
    size_t total = row_bytes * (size_t)O;

    picolm_gpu_tensor_t *t = (picolm_gpu_tensor_t *)calloc(1, sizeof(*t));
    if (!t) return 0;
    t->qtype = qtype; t->I = I; t->O = O; t->device = device;
    t->row_bytes = row_bytes; t->block_size = bs;

    /* Q2_K/Q3_K/Q4_K/Q5_K/Q6_K fast path: convert to Q8_0 at upload time for the int8-MAC kernel.
     * These have per-sub-block scales that the Q8_0 kernel can't handle,
     * but requantizing to Q8_0 (1 scale per 32 elements) is accurate enough
     * and enables the highly-optimized picolm_q8_q8_matmul path. */
    if (qtype == 10 || qtype == 11 || qtype == 12 || qtype == 13 || qtype == 14) {
        /* Dequant Q2_K/Q3_K/Q4_K/Q5_K/Q6_K to F32, then requant to Q8_0 */
        size_t f32_bytes = (size_t)I * (size_t)O * sizeof(float);
        float *f32_buf = (float *)calloc(I * O, sizeof(float));
        if (!f32_buf) { gpuFree(t->weights); free(t); return 0; }

        /* Dequant each row */
        int nb = I / 256;
        int blk_bytes;
        if (qtype == 10) blk_bytes = 84;
        else if (qtype == 11) blk_bytes = 110;
        else if (qtype == 12) blk_bytes = 144;
        else if (qtype == 13) blk_bytes = 176;
        else blk_bytes = 210; /* qtype == 14 */
        for (int row = 0; row < O; row++) {
            float *dst = f32_buf + row * I;
            const uint8_t *row_start = (const uint8_t *)weights + row * nb * blk_bytes;
            if (qtype == 10)
                dequantize_row_q2_K(row_start, dst, I);
            else if (qtype == 11)
                dequantize_row_q3_K(row_start, dst, I);
            else if (qtype == 12)
                dequantize_row_q4_K(row_start, dst, I);
            else if (qtype == 13)
                dequantize_row_q5_K(row_start, dst, I);
            else
                dequantize_row_q6_K(row_start, dst, I);
        }

        /* Requant to Q8_0 */
        size_t q8_total = (size_t)O * ((I + 31) / 32) * 34;  /* 34 bytes per Q8_0 block */
        uint8_t *q8_buf = (uint8_t *)calloc(q8_total, 1);
        if (!q8_buf) { free(f32_buf); gpuFree(t->weights); free(t); return 0; }

        for (int row = 0; row < O; row++) {
            const float *src = f32_buf + row * I;
            block_q8_0 *q8_blocks = (block_q8_0 *)(q8_buf + row * ((I + 31) / 32) * 34);
            quantize_row_q8_0(src, q8_blocks, I);
        }

        free(f32_buf);

        /* Upload Q8_0 bytes, record as Q8_0 type */
        if (!gpu_ok(gpuMalloc(&t->weights, q8_total), "tensor allocation (q8)") ||
            !gpu_ok(gpuMemcpy(t->weights, q8_buf, q8_total, gpuMemcpyHostToDevice),
                    "tensor upload (q8)")) {
            free(q8_buf); gpuFree(t->weights); free(t); return 0;
        }
        free(q8_buf);

        t->qtype = (gguf_type_t)8;  /* Now Q8_0 for the fast path */
        t->block_size = 34;
        t->row_bytes = ((I + 31) / 32) * 34;
        t->zero_copy = 0;
        t->tracked = 1;
        ctx->tensor_count++;
        ctx->tensor_bytes += q8_total;
        {
            static int first_print = 1;
            if (first_print) {
                fprintf(stderr, "[GPU] upload mode: q2/q3/q4/q5/q6->q8 (int8-MAC fast path)\n");
                first_print = 0;
            }
        }
        *tp = t;
        return 1;
    }

    /* Try zero-copy first: register CPU memory with GPU (unified memory SoC) */
    /* NOTE: mmap'd file-backed memory can cause zero-copy to silently produce
     * wrong results on some platforms. cudaHostRegister may succeed but the GPU
     * sees stale or unmapped pages. We use cudaHostRegister with
     * cudaHostRegisterPortable and verify the upload works.
     *
     * For now, skip zero-copy and always copy. The copy happens once at model
     * load time, and on SoC systems with unified memory the effective cost is
     * just a page-table walk, not a real data copy. */
    if (!gpu_ok(gpuMalloc(&t->weights, total), "tensor allocation")) {
        fprintf(stderr, "[GPU] OOM: I=%d O=%d qtype=%d total=%zu MB (gpu_used=%.1f MB)\n",
                I, O, qtype, total/(1024*1024), ctx->tensor_bytes/(1024.0*1024));
        free(t); return 0;
    }
    if (!gpu_ok(gpuMemcpy(t->weights, weights, total, gpuMemcpyHostToDevice),
                "tensor upload")) {
        gpuFree(t->weights); free(t); return 0;
    }
    t->zero_copy = 0;
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += total;
    
    /* Print upload summary for first tensor */
    {
        static int first_print = 1;
        if (first_print) {
            fprintf(stderr, "[GPU] upload mode: %s\n", t->zero_copy ? "zero-copy (unified)" : "copied");
            first_print = 0;
        }
    }
    *tp = t;
    return 1;
}

#ifdef PICOLM_SSM_VERIFY
extern "C" void picolm_gpu_debug_tensor(const char *name, void *tp, int device, int layer, int dump_weights) {
    picolm_gpu_tensor_t *t = (picolm_gpu_tensor_t *)tp;
    if (!t) return;
    fprintf(stderr, "[DBG l%d] %s tensor: qtype=%d I=%d O=%d row_bytes=%zu\n",
        layer, name, (int)t->qtype, t->I, t->O, t->row_bytes);
    if (dump_weights && t->weights) {
        /* Dequantize first 64 floats */
        float wbuf[64] = {0};
        if (t->qtype == 0) {
            /* F32: D2H direct */
            cudaSetDevice(device);
            cudaMemcpy(wbuf, t->weights, 64 * sizeof(float), cudaMemcpyDeviceToHost);
        } else if (t->qtype == 2) {
            /* Q8_0: dump raw blocks then dequantize */
            unsigned char raw[66]; /* 2 blocks: 32+1 + 32+1 = 66 */
            cudaSetDevice(device);
            cudaMemcpy(raw, t->weights, 66, cudaMemcpyDeviceToHost);
            float sc0 = *(const float *)(raw + 32);
            for (int i = 0; i < 32; i++) wbuf[i] = (raw[i] - 128) * sc0;
            float sc1 = *(const float *)(raw + 64);
            for (int i = 0; i < 32; i++) wbuf[32 + i] = (raw[33 + i] - 128) * sc1;
        } else if (t->qtype == 8) {
            /* BF16: D2H and convert */
            unsigned short b16[64];
            cudaSetDevice(device);
            cudaMemcpy(b16, t->weights, 64 * 2, cudaMemcpyDeviceToHost);
            for (int i = 0; i < 64; i++) {
                unsigned int bits = (unsigned int)b16[i] << 16;
                float f; memcpy(&f, &bits, 4);
                wbuf[i] = f;
            }
        }
        fprintf(stderr, "[DBG l%d] %s_w[0][:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f}\n",
            layer, name, wbuf[0],wbuf[1],wbuf[2],wbuf[3],wbuf[4],wbuf[5],wbuf[6],wbuf[7]);
        double wr = 0; for (int i = 0; i < 64; i++) wr += wbuf[i] * wbuf[i];
        fprintf(stderr, "[DBG l%d] %s_w rms64=%.6f\n", layer, name, sqrt(wr / 64));
    }
}
#endif /* PICOLM_SSM_VERIFY */

extern "C"
void picolm_gpu_tensor_free(picolm_gpu_tensor_t *t) {
    if (!t) return;
    gpu_device_ctx_t *ctx = find_ctx(t->device);
    if (ctx && t->tracked) {
        if (select_ctx(ctx)) {
            ctx->tensor_count--;
            ctx->tensor_bytes -= t->row_bytes * (size_t)t->O;
        }
    }
    if (t->weights) {
        if (t->zero_copy) {
            gpuHostUnregister(t->weights);
        } else {
            gpuFree(t->weights);
        }
        t->weights = NULL;
    }
    free(t);
}

extern "C"
size_t picolm_gpu_tensor_bytes(const picolm_gpu_tensor_t *t) {
    return t ? t->row_bytes * (size_t)t->O : 0;
}

extern "C"
int picolm_gpu_tensor_device(const picolm_gpu_tensor_t *t) {
    return t ? t->device : -1;
}
const void *picolm_gpu_tensor_weights(const picolm_gpu_tensor_t *t) {
    return t ? t->weights : NULL;
}

/* Upload a plain host F32 vector (norm weights, RoPE cos/sin tables --
 * anything that isn't a quantized matmul weight matrix, so
 * picolm_gpu_tensor_upload doesn't apply) to a freshly allocated device
 * buffer. Caller owns the returned pointer and must gpuFree() it (or
 * just leak it for the lifetime of the process, same as other one-time
 * per-model uploads here). Returns NULL on failure. */
extern "C" float *
picolm_gpu_upload_f32(const float *host, size_t n, int device) {
    if (!host || n < 1) return NULL;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return NULL;

    float *dev = NULL;
    size_t bytes = n * sizeof(float);
    if (!gpu_ok(gpuMalloc(&dev, bytes), "f32 vector allocation")) return NULL;
    if (!gpu_ok(gpuMemcpy(dev, host, bytes, gpuMemcpyHostToDevice), "f32 vector upload")) {
        gpuFree(dev);
        return NULL;
    }
    return dev;
}

/* Generic device memory allocation. Returns NULL on failure. */
void *picolm_gpu_alloc_device(size_t bytes, int device) {
    if (bytes < 1) return NULL;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return NULL;
    void *ptr = NULL;
    if (!gpu_ok(gpuMalloc(&ptr, bytes), "device alloc")) return NULL;
    return ptr;
}

/* Device memory set to value. Uses gpuMemset (zero-fill). */
extern "C"
int picolm_gpu_device_memset(void *dev_ptr, int value, size_t bytes, int device) {
    if (!dev_ptr || bytes < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    return gpu_ok(gpuMemset(dev_ptr, value, bytes), "device memset");
}

/* Upload a host int32 array to device. Returns device pointer or NULL. */
void *picolm_gpu_upload_int(const int *host, size_t n, int device) {
    if (!host || n < 1) return NULL;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return NULL;
    void *dev = NULL;
    size_t bytes = n * sizeof(int);
    if (!gpu_ok(gpuMalloc(&dev, bytes), "int vector allocation")) return NULL;
    if (!gpu_ok(gpuMemcpy(dev, host, bytes, gpuMemcpyHostToDevice), "int vector upload")) {
        gpuFree(dev);
        return NULL;
    }
    return dev;
}

extern "C"
int picolm_gpu_matmul(picolm_gpu_tensor_t *t, float *y, const float *x, int S, int device) {
    if (!t || !y || !x || S < 1) return 0;
    /* Minimum I for GPU to be worthwhile vs CPU kernel launch overhead.
     * Smaller tensors: kernel launch (~200us) dominates compute. */
    if (t->I < 512 || t->O < 256) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    int I = t->I, O = t->O;
    size_t xb = (size_t)S * I * sizeof(float);
    size_t yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) ||
        !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;

    /* ---- Q8_0 special path: quantize x to Q8_0 on GPU, then int8 MAC ----
     * This matches the CPU path in tensor.c exactly:
     *   1. quantize_row_q8_0(x) -> int8 qs + float d per 32-block
     *   2. vec_dot_q8_0_q8_0_deltas(int8_x, float_xd, int8_w, fp16_wd)
     * GPU quantize kernel outputs d as FP32 directly (matching CPU's
     * pre-converted qx_d array), avoiding an extra FP16->FP32 conversion
     * step. The matmul kernel then does pure int8 MAC with fp32 scales. */
    if (t->qtype == GGUF_TYPE_Q8_0) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0; /* must be aligned */

        /* Upload fp32 input */
        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;

        /* Allocate quantized input buffers: qs (int8_t[S*I]) + d (float[S*n_blocks])
         * Round S up to multiple of 16 for IMMA kernel (reads up to S+15 in tail tile).
         * The quantize kernel writes exactly S rows; the IMMA kernel reads up to S+15
         * but only writes output for rows < S. Padding prevents OOB reads. */
        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        /* Step 1: Quantize x to Q8_0 on GPU */
        dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        if (ctx->has_imma && S >= 16 && O >= 8) {
            gpu_dispatch_print("matmul_imma_host");
            dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
            picolm_q8_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
                ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        } else if (S > 1 && t->row_bytes + 2048 <= 49152) {
            gpu_dispatch_print("matmul_tiled_host");
            dim3 grid((unsigned)O, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
            picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)t->row_bytes, ctx->stream>>>(
                ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        } else {
            gpu_dispatch_print("matmul_scalar_host");
            dim3 grid((unsigned)O, (unsigned)S);
            picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
                ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        }
        if (!gpu_ok(gpuGetLastError(), "q8 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 matmul sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* FP16/BF16 tiled path: load weight row into shared memory once per
     * tile of F16_TILE_S positions, reducing global weight reads by
     * S/TILE_S. Falls back to picolm_quant_matmul for small S or
     * when shared memory is insufficient. */
    if ((t->qtype == GGUF_TYPE_F16 || t->qtype == GGUF_TYPE_BF16) &&
        S >= F16_TILE_S && (int)t->row_bytes + 2048 <= 49152) {
        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        dim3 grid((unsigned)O, (unsigned)((S + F16_TILE_S - 1) / F16_TILE_S));
        unsigned smem = (unsigned)(t->row_bytes + 256 * sizeof(double));
        if (t->qtype == GGUF_TYPE_F16) {
            gpu_dispatch_print("matmul_f16_tiled_host");
            picolm_f16_f16_matmul_tiled<<<grid, 256, smem, ctx->stream>>>(
                ctx->y, ctx->x, (const uint16_t *)t->weights, S, I, O,
                (int)t->row_bytes, O);
        } else {
            gpu_dispatch_print("matmul_bf16_tiled_host");
            picolm_bf16_f32_matmul_tiled<<<grid, 256, smem, ctx->stream>>>(
                ctx->y, ctx->x, (const uint16_t *)t->weights, S, I, O,
                (int)t->row_bytes, O);
        }
        if (!gpu_ok(gpuGetLastError(), "f16 tiled matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "f16 tiled matmul sync")) return 0;
        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* Generic path for all other quant types */
    /* Scratch buffers are in device memory. Explicit H2D copy needed. */
    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;

    gpu_dispatch_print("matmul_quant_host");
    dim3 grid((unsigned)O, (unsigned)S);
    picolm_quant_matmul<<<grid, 256, 0, ctx->stream>>>(ctx->y, ctx->x, t->weights,
                                        t->qtype, S, I, O,
                                        (int)t->row_bytes, 0, 0);
    if (!gpu_ok(gpuGetLastError(), "matmul launch") ||
        !gpu_ok(gpuDeviceSynchronize(), "matmul sync")) return 0;

    /* Scratch buffers are in device memory. Explicit D2H copy needed. */
    if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

/* Device-native matmul: x_dev and y_dev are already device-resident
 * pointers owned by the caller (Phase 2 pipeline buffers) -- no H2D, no
 * D2H, no gpuDeviceSynchronize(). Mirrors picolm_gpu_matmul() exactly
 * (same eligibility checks, same kernels, same Q8_0 special-case path)
 * except for the copies and the per-step sync.
 *
 * Safe to omit the sync here: all kernels below are launched on
 * ctx->stream, and CUDA guarantees in-order execution of work queued to
 * the same stream, so the q8 quantize kernel is guaranteed complete
 * before the q8_q8 matmul kernel reads its output, with no explicit sync
 * needed in between. The caller (model_forward_gpu) must call
 * gpuDeviceSynchronize() exactly once, after the very last op of the
 * whole forward pass and before reading any result back via D2H -- not
 * after each layer or each matmul. That single sync is what actually
 * eliminates the ~14-syncs/layer overhead Phase 2 exists to remove.
 *
 * x_dev/y_dev must NOT alias ctx->x/ctx->y/ctx->q8_xq/ctx->q8_xd (those
 * are still used internally here as Q8_0 quantize scratch) -- pass
 * dedicated pipeline buffers. */
extern "C" int
picolm_gpu_matmul_dev(picolm_gpu_tensor_t *t, float *y_dev, const float *x_dev,
                       int S, int device, int y_stride, int x_stride) {
    if (!t || !y_dev || !x_dev || S < 1) return 0;
    if (t->I < 512 || t->O < 256) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    int I = t->I, O = t->O;
    int ys = y_stride > 0 ? y_stride : O;

    if (t->qtype == GGUF_TYPE_Q8_0 && !getenv("PICOLM_FORCE_F32_MATMUL")) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0;
        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;
        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);
        dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        if (x_stride > 0) {
            picolm_quantize_q8_0_strided<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
                ctx->q8_xq, ctx->q8_xd, x_dev, I, S, x_stride);
        } else {
            picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
                ctx->q8_xq, ctx->q8_xd, x_dev, I, S);
        }
        if (!gpu_ok(gpuGetLastError(), "q8 quantize (dev)")) return 0;
        if (ctx->has_imma && S >= 16 && O >= 8) {
            gpu_dispatch_print("matmul_imma_dev");
            dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
            picolm_q8_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
                y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        } else if (S >= Q8_TILE_S && t->row_bytes + 2048 <= 49152) {
            gpu_dispatch_print("matmul_tiled_dev");
            dim3 grid((unsigned)O, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
            picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)t->row_bytes, ctx->stream>>>(
                y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        } else {
            gpu_dispatch_print("matmul_scalar_dev");
            dim3 grid((unsigned)O, (unsigned)S);
            picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
                y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        }
        if (!gpu_ok(gpuGetLastError(), "q8 matmul (dev)")) return 0;
        return 1;
    }
    /* FP16/BF16 tiled path for device-resident tensors */
    if ((t->qtype == GGUF_TYPE_F16 || t->qtype == GGUF_TYPE_BF16) &&
        S >= F16_TILE_S && (int)t->row_bytes + 2048 <= 49152) {
        dim3 grid((unsigned)O, (unsigned)((S + F16_TILE_S - 1) / F16_TILE_S));
        unsigned smem = (unsigned)(t->row_bytes + 256 * sizeof(double));
        if (t->qtype == GGUF_TYPE_F16) {
            gpu_dispatch_print("matmul_f16_tiled_dev");
            picolm_f16_f16_matmul_tiled<<<grid, 256, smem, ctx->stream>>>(
                y_dev, x_dev, (const uint16_t *)t->weights, S, I, O,
                (int)t->row_bytes, ys);
        } else {
            gpu_dispatch_print("matmul_bf16_tiled_dev");
            picolm_bf16_f32_matmul_tiled<<<grid, 256, smem, ctx->stream>>>(
                y_dev, x_dev, (const uint16_t *)t->weights, S, I, O,
                (int)t->row_bytes, ys);
        }
        if (!gpu_ok(gpuGetLastError(), "f16 tiled matmul (dev)")) return 0;
        return 1;
    }

    if (t->qtype == GGUF_TYPE_Q8_0) {
        /* Fallback: use per-thread F32 dequant for Q8_0 */
        gpu_dispatch_print("matmul_quant_dev_fallback");
        dim3 grid((unsigned)O, (unsigned)S);
        picolm_quant_matmul<<<grid, 256, 0, ctx->stream>>>(y_dev, x_dev, t->weights,
                                            t->qtype, S, I, O,
                                            (int)t->row_bytes, x_stride, ys);
        if (!gpu_ok(gpuGetLastError(), "q8 matmul f32 fallback (dev)")) return 0;
        return 1;
    }
    /* Non-Q8_0 types: use per-thread dequant + F32 accumulation */
    gpu_dispatch_print("matmul_quant_dev");
    dim3 grid((unsigned)O, (unsigned)S);
    picolm_quant_matmul<<<grid, 256, 0, ctx->stream>>>(y_dev, x_dev, t->weights,
                                        t->qtype, S, I, O,
                                        (int)t->row_bytes, x_stride, ys);
    if (!gpu_ok(gpuGetLastError(), "matmul launch (dev)")) return 0;
    return 1;
}

/* Strided variant: x_stride > 0 overrides default I stride.
 * Only called for SSM output projection where pipe buffer stride != value_dim. */
extern "C" int
picolm_gpu_matmul_dev_strided(picolm_gpu_tensor_t *t, float *y_dev,
                               const float *x_dev, int S, int device, int x_stride, int y_stride) {
    if (!t || !y_dev || !x_dev || S < 1 || x_stride <= 0) return 0;
    if (t->I < 512 || t->O < 256) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;
    int I = t->I, O = t->O;
    int ys = y_stride > 0 ? y_stride : O;
    if (t->qtype == GGUF_TYPE_Q8_0) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0;
        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;
        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);
        dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0_strided<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, x_dev, I, S, x_stride);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize strided (dev)")) return 0;
        /* No gpuDeviceSynchronize() needed: stream ordering guarantees correctness. */
        gpu_dispatch_print("matmul_scalar_strided_dev");
        dim3 grid((unsigned)O, (unsigned)S);
        picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q8 matmul strided (dev)")) return 0;
        return 1;
    }
    /* FP16/BF16 tiled path for strided tensors (only when x is contiguous) */
    if ((t->qtype == GGUF_TYPE_F16 || t->qtype == GGUF_TYPE_BF16) &&
        x_stride == I && S >= F16_TILE_S && (int)t->row_bytes + 2048 <= 49152) {
        dim3 grid((unsigned)O, (unsigned)((S + F16_TILE_S - 1) / F16_TILE_S));
        unsigned smem = (unsigned)(t->row_bytes + 256 * sizeof(double));
        if (t->qtype == GGUF_TYPE_F16) {
            gpu_dispatch_print("matmul_f16_tiled_strided_dev");
            picolm_f16_f16_matmul_tiled<<<grid, 256, smem, ctx->stream>>>(
                y_dev, x_dev, (const uint16_t *)t->weights, S, I, O,
                (int)t->row_bytes, ys);
        } else {
            gpu_dispatch_print("matmul_bf16_tiled_strided_dev");
            picolm_bf16_f32_matmul_tiled<<<grid, 256, smem, ctx->stream>>>(
                y_dev, x_dev, (const uint16_t *)t->weights, S, I, O,
                (int)t->row_bytes, ys);
        }
        if (!gpu_ok(gpuGetLastError(), "f16 tiled matmul strided (dev)")) return 0;
        return 1;
    }
    gpu_dispatch_print("matmul_quant_strided_dev");
    dim3 grid((unsigned)O, (unsigned)S);
    picolm_quant_matmul<<<grid, 256, 0, ctx->stream>>>(y_dev, x_dev, t->weights,
                                        t->qtype, S, I, O, (int)t->row_bytes, x_stride, ys);
    if (!gpu_ok(gpuGetLastError(), "matmul strided (dev)")) return 0;
    return 1;
}

extern "C"
int picolm_gpu_expert_mlp(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                           picolm_gpu_tensor_t *down, float *y, const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;

    /* Only Q8_0 int8-MAC and FP16/BF16 tiled paths are implemented here.
     * For all other types (F32, Q4_K etc.), return 0
     * so the caller falls back to the per-matmul CPU path. */
    int is_f16 = (gate->qtype == GGUF_TYPE_F16 || gate->qtype == GGUF_TYPE_BF16);
    int is_q80 = (gate->qtype == GGUF_TYPE_Q8_0);
    if (!(is_f16 || is_q80)) return 0;
    if (gate->qtype != up->qtype || gate->qtype != down->qtype) return 0;

    gpu_device_ctx_t *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;

    int D = gate->I, I = gate->O;
    size_t xb = (size_t)S * D * sizeof(float);
    size_t ib = (size_t)S * I * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) ||
        !reserve(&ctx->y, &ctx->y_cap, xb) ||
        !reserve(&ctx->gate, &ctx->gate_cap, ib) ||
        !reserve(&ctx->up, &ctx->up_cap, ib)) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "expert input")) return 0;

    /* ---- FP16/BF16 tiled path ----
     * Direct F32 x FP16/BF16 matmul with tiled weight loading.
     * No quantization needed. */
    if (is_f16 && S >= F16_TILE_S &&
        (int)gate->row_bytes + 2048 <= 49152 &&
        (int)up->row_bytes + 2048 <= 49152 &&
        (int)down->row_bytes + 2048 <= 49152) {
        unsigned smem = (unsigned)(gate->row_bytes + 256 * sizeof(double));

        /* Step 1: gate = f16 tiled(input, gate_weights) -> F32[I*S] */
        dim3 grid_g((unsigned)I, (unsigned)((S + F16_TILE_S - 1) / F16_TILE_S));
        if (gate->qtype == GGUF_TYPE_F16) {
            gpu_dispatch_print("expert_mlp_gate_f16_tiled_host");
            picolm_f16_f16_matmul_tiled<<<grid_g, 256, smem, ctx->stream>>>(
                ctx->gate, ctx->x, (const uint16_t *)gate->weights, S, D, I,
                (int)gate->row_bytes, I);
        } else {
            gpu_dispatch_print("expert_mlp_gate_bf16_tiled_host");
            picolm_bf16_f32_matmul_tiled<<<grid_g, 256, smem, ctx->stream>>>(
                ctx->gate, ctx->x, (const uint16_t *)gate->weights, S, D, I,
                (int)gate->row_bytes, I);
        }
        if (!gpu_ok(gpuGetLastError(), "expert f16 gate")) return 0;

        /* Step 2: up = f16 tiled(input, up_weights) -> F32[I*S] */
        unsigned smem_up = (unsigned)(up->row_bytes + 256 * sizeof(double));
        if (up->qtype == GGUF_TYPE_F16) {
            gpu_dispatch_print("expert_mlp_up_f16_tiled_host");
            picolm_f16_f16_matmul_tiled<<<grid_g, 256, smem_up, ctx->stream>>>(
                ctx->up, ctx->x, (const uint16_t *)up->weights, S, D, I,
                (int)up->row_bytes, I);
        } else {
            gpu_dispatch_print("expert_mlp_up_bf16_tiled_host");
            picolm_bf16_f32_matmul_tiled<<<grid_g, 256, smem_up, ctx->stream>>>(
                ctx->up, ctx->x, (const uint16_t *)up->weights, S, D, I,
                (int)up->row_bytes, I);
        }
        if (!gpu_ok(gpuGetLastError(), "expert f16 up")) return 0;

        /* Step 3: silu(gate) * up -> ctx->gate */
        size_t n = (size_t)S * I;
        picolm_silu_mul<<<(unsigned)((n + 255) / 256), 256, 0, ctx->stream>>>(ctx->gate, ctx->up, n);
        if (!gpu_ok(gpuGetLastError(), "expert silu")) return 0;

        /* Step 4: y = f16 tiled(silu*up, down_weights) -> F32[D*S] */
        unsigned smem_dn = (unsigned)(down->row_bytes + 256 * sizeof(double));
        dim3 grid_d((unsigned)D, (unsigned)((S + F16_TILE_S - 1) / F16_TILE_S));
        if (down->qtype == GGUF_TYPE_F16) {
            gpu_dispatch_print("expert_mlp_down_f16_tiled_host");
            picolm_f16_f16_matmul_tiled<<<grid_d, 256, smem_dn, ctx->stream>>>(
                ctx->y, ctx->gate, (const uint16_t *)down->weights, S, I, D,
                (int)down->row_bytes, D);
        } else {
            gpu_dispatch_print("expert_mlp_down_bf16_tiled_host");
            picolm_bf16_f32_matmul_tiled<<<grid_d, 256, smem_dn, ctx->stream>>>(
                ctx->y, ctx->gate, (const uint16_t *)down->weights, S, I, D,
                (int)down->row_bytes, D);
        }
        if (!gpu_ok(gpuGetLastError(), "expert f16 down")) return 0;

        if (!gpu_ok(gpuDeviceSynchronize(), "expert f16 MLP sync")) return 0;
        return gpu_ok(gpuMemcpy(y, ctx->y, xb, gpuMemcpyDeviceToHost), "expert output");
    }

    /* ---- Q8_0 fast path: quantize F32 input to Q8_0, then int8 MAC ----
     * All 3 projections (gate, up, down) use Q8_0 weights. The input to
     * gate+up is the same F32 vector, so we quantize it once and reuse
     * the Q8_0 copy for both. The input to down is silu(gate)*up (F32),
     * which we quantize again before the down projection.
     * This replaces the slow per-block picolm_quant_matmul kernel with
     * the fast picolm_q8_q8_matmul_tiled kernel that tiles the Q8_0
     * weight row in shared memory across 32 query positions. */
    int d_blocks = D / 32;
    int i_blocks = I / 32;
    if (d_blocks < 1 || i_blocks < 1) return 0; /* must be aligned */

    /* Round S up for IMMA kernel tail-tile reads */
    int S_padded = (S + 15) & ~15;
    size_t xq_d_bytes = (size_t)S_padded * D;
    size_t xd_d_bytes = (size_t)S_padded * d_blocks * sizeof(float);
    size_t xq_i_bytes = (size_t)S_padded * I;
    size_t xd_i_bytes = (size_t)S_padded * i_blocks * sizeof(float);

    if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_d_bytes > xq_i_bytes ? xq_d_bytes : xq_i_bytes) ||
        !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_d_bytes > xd_i_bytes ? xd_d_bytes : xd_i_bytes)) return 0;

    /* Step 1: Quantize F32 input [D*S] to Q8_0 */
    dim3 q_grid_d((unsigned)d_blocks, (unsigned)S);
    picolm_quantize_q8_0<<<q_grid_d, 32, 32 * sizeof(float), ctx->stream>>>(
        ctx->q8_xq, ctx->q8_xd, ctx->x, D, S);
    if (!gpu_ok(gpuGetLastError(), "expert q8 quantize input")) return 0;

    /* Step 2: gate = q8_q8 matmul(input_q8, gate_weights) -> F32[I*S] */
    if (ctx->has_imma && S >= 16 && I >= 8) {
        gpu_dispatch_print("expert_mlp_gate_imma_host");
        dim3 grid((unsigned)((I + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, gate->weights, S, D, I, (int)gate->row_bytes, I);
    } else if (S > 1 && gate->row_bytes + 2048 <= 49152) {
        gpu_dispatch_print("expert_mlp_gate_tiled_host");
        dim3 grid((unsigned)I, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)gate->row_bytes, ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, gate->weights, S, D, I, (int)gate->row_bytes, I);
    } else {
        gpu_dispatch_print("expert_mlp_gate_scalar_host");
        dim3 grid((unsigned)I, (unsigned)S);
        picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, gate->weights, S, D, I, (int)gate->row_bytes, I);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 gate")) return 0;

    /* Step 3: up = q8_q8 matmul(input_q8, up_weights) -> F32[I*S] */
    if (ctx->has_imma && S >= 16 && I >= 8) {
        gpu_dispatch_print("expert_mlp_up_imma_host");
        dim3 grid((unsigned)((I + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, up->weights, S, D, I, (int)up->row_bytes, I);
    } else if (S > 1 && up->row_bytes + 2048 <= 49152) {
        gpu_dispatch_print("expert_mlp_up_tiled_host");
        dim3 grid((unsigned)I, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)up->row_bytes, ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, up->weights, S, D, I, (int)up->row_bytes, I);
    } else {
        gpu_dispatch_print("expert_mlp_up_scalar_host");
        dim3 grid((unsigned)I, (unsigned)S);
        picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, up->weights, S, D, I, (int)up->row_bytes, I);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 up")) return 0;

    /* Step 4: silu(gate) * up -> ctx->gate (in-place, F32[I*S]) */
    size_t n = (size_t)S * I;
    picolm_silu_mul<<<(unsigned)((n + 255) / 256), 256, 0, ctx->stream>>>(ctx->gate, ctx->up, n);
    if (!gpu_ok(gpuGetLastError(), "expert silu")) return 0;

    /* Step 5: Quantize F32 silu*up [I*S] to Q8_0 for down projection */
    dim3 q_grid_i((unsigned)i_blocks, (unsigned)S);
    picolm_quantize_q8_0<<<q_grid_i, 32, 32 * sizeof(float), ctx->stream>>>(
        ctx->q8_xq, ctx->q8_xd, ctx->gate, I, S);
    if (!gpu_ok(gpuGetLastError(), "expert q8 quantize hidden")) return 0;

    /* Step 6: y = q8_q8 matmul(hidden_q8, down_weights) -> F32[D*S] */
    if (ctx->has_imma && S >= 16 && D >= 8) {
        gpu_dispatch_print("expert_mlp_down_imma_host");
        dim3 grid((unsigned)((D + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, down->weights, S, I, D, (int)down->row_bytes, D);
    } else if (S > 1 && down->row_bytes + 2048 <= 49152) {
        gpu_dispatch_print("expert_mlp_down_tiled_host");
        dim3 grid((unsigned)D, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)down->row_bytes, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, down->weights, S, I, D, (int)down->row_bytes, D);
    } else {
        gpu_dispatch_print("expert_mlp_down_scalar_host");
        dim3 grid((unsigned)D, (unsigned)S);
        picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, down->weights, S, I, D, (int)down->row_bytes, D);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 down")) return 0;

    if (!gpu_ok(gpuDeviceSynchronize(), "expert MLP sync")) return 0;
    return gpu_ok(gpuMemcpy(y, ctx->y, xb, gpuMemcpyDeviceToHost), "expert output");
}

extern "C"
int picolm_gpu_w4a16_mlp(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                          picolm_gpu_tensor_t *down, float *y, const float *x, int S) {
#ifdef PICOLM_GPU_WMMA_AVAILABLE
    if (!gate || !up || !down || !x || !y || S < 1) return 0;
    if (gate->qtype != 2 || up->qtype != 2 || down->qtype != 2) return 0; /* Q4_0 only */
    if (gate->device != up->device || gate->device != down->device) return 0;
    if (gate->I != up->I || gate->O != up->O || down->I != gate->O || down->O != gate->I)
        return 0;

    gpu_device_ctx_t *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;

    int D = gate->I, I = gate->O;
    size_t xb = (size_t)S * D * sizeof(float);
    size_t ib = (size_t)S * I * sizeof(float);

    if (!reserve(&ctx->x, &ctx->x_cap, xb) ||
        !reserve(&ctx->gate, &ctx->gate_cap, ib) ||
        !reserve(&ctx->up, &ctx->up_cap, ib) ||
        !reserve(&ctx->y, &ctx->y_cap, xb)) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "w4a16 input")) return 0;

    /* fused gate+up via WMMA */
    dim3 hidden((unsigned)((I + 63) / 64), (unsigned)((S + 15) / 16));
    picolm_w4a16_gate_up<<<hidden, 256>>>(ctx->gate, ctx->up, ctx->x,
        gate->weights, up->weights, S, D, I, gate->block_size);
    /* silu(gate) * up */
    size_t n = (size_t)S * I;
    picolm_silu_mul<<<(unsigned)((n + 255) / 256), 256>>>(ctx->gate, ctx->up, n);
    /* down via WMMA */
    dim3 output((unsigned)((D + 63) / 64), (unsigned)((S + 15) / 16));
    picolm_w4a16_matmul<<<output, 256>>>(ctx->y, ctx->gate, down->weights, S, I, D, down->block_size);

    if (!gpu_ok(gpuGetLastError(), "w4a16 launch")) return 0;
#ifdef __HIP__
    if (!gpu_ok(gpuMemcpy(y, ctx->y, xb, gpuMemcpyDeviceToHost), "w4a16 output")) return 0;
#else
    memcpy(y, ctx->y, xb);
#endif
    return 1;
#else
    (void)gate; (void)up; (void)down; (void)y; (void)x; (void)S;
    return 0; /* WMMA not available on this arch */
#endif
}

/* General-purpose WMMA matmul for Q4_0 weights.
 * Handles any tensor, not just expert MLP.
 * M = rows (S in our convention), K = columns (I), N = output (O)
 * Returns 1 on success, 0 to fall back to quant_matmul.
 * Constraints: qtype must be Q4_0, N%64==0, M%16==0, K%32==0. */
extern "C"
int picolm_gpu_w4a16_matmul(picolm_gpu_tensor_t *t, float *y, const float *x, int S, int device) {
#ifdef PICOLM_GPU_WMMA_AVAILABLE
    if (!t || !y || !x || S < 1 || t->qtype != 2) return 0; /* Q4_0 only */
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    int M = S, K = t->I, N = t->O;
    /* WMMA is only worthwhile for large batch sizes.
     * Small M wastes SM resources (16x64 tile for a 1-row output).
     * N must be at least 64 (one grid-x tile). */
    if (M < 16 || N < 64 || K < 64) return 0;

    size_t xb = (size_t)M * K * sizeof(float);
    size_t yb = (size_t)M * N * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) ||
        !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "w4a16 input")) return 0;

    dim3 grid((unsigned)((N + 63) / 64), (unsigned)((M + 15) / 16));
    picolm_w4a16_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, M, K, N, t->block_size);
    if (!gpu_ok(gpuGetLastError(), "w4a16 matmul") ||
        !gpu_ok(gpuDeviceSynchronize(), "w4a16 sync")) return 0;

    if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "w4a16 output")) return 0;
    return 1;
#else
    (void)t; (void)y; (void)x; (void)S; (void)device;
    return 0;
#endif
}

/* ---- SSM recurrence API ---- */
