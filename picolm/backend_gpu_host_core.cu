// backend_gpu_host_core.cu - GPU initialization, memory, tensor, matmul, expert MLP
#include "backend_gpu_kernels.cuh"

/* PICOLM_GPU_PROF hooks (implemented in backend_gpu_host_misc.cu). */
extern "C" int  picolm_gpu_prof_slot(const char *name);
extern "C" void picolm_gpu_prof_begin(int device);
extern "C" void picolm_gpu_prof_end(int slot);
extern "C" void picolm_gpu_launch_tmr_report(void);


/* Win32 API forward declarations for CUDA_DEVICE_MAX_CONNECTIONS env var
 * management on Windows. Avoids pulling in <windows.h> (min/max/byte macro
 * pollution) into a .cu file compiled by nvcc with CUDA/C++ headers.
 * kernel32.lib is part of MSVC's default link libraries. */
#ifdef _WIN32
extern "C" {
    __declspec(dllimport) unsigned long __stdcall
        GetEnvironmentVariableA(const char *, char *, unsigned long);
    __declspec(dllimport) int __stdcall
        SetEnvironmentVariableA(const char *, const char *);
}
#endif

/* Phase 8: forward declare for nvcc/MSVC cross-TU kernel launch.
 * The .cuh declaration isn't picked up by MSVC's C++ front-end. */
__global__ void picolm_q8_q8_matmul_imma_smw16(float *, const int8_t *, const float *,
                                                const void *, int, int, int, int, int);

/* PICOLM_GEMV_DBG: the layer whose decode/prefill pass is currently running.
 * Set by the layer loops in model_ssm.c purely so the GEMV debug line can be
 * attributed to a layer. Zero when nothing has set it. */
int picolm_gpu_dbg_layer = -1;
extern "C" void picolm_gpu_set_dbg_layer(int l) { picolm_gpu_dbg_layer = l; }

static void gpu_dispatch_print(const char *name) {
    /* Called on EVERY kernel launch (~350 per decode token). Once all 64 slots
     * are used there is provably nothing left to print, so bail immediately
     * instead of running 64 strcmp()s per launch. This is not a micro
     * optimisation: at 350 launches/token the scan alone was measurable next
     * to the launch cost. */
    static char seen[64][64] = {0};
    static int  filled = 0;
    static int  full = 0;
    if (full) return;
    for (int i = 0; i < filled; i++) {
        if (strcmp(seen[i], name) == 0) return;
    }
    snprintf(seen[filled], 64, "%s", name);
    fprintf(stderr, "[GPU] kernel: %s\n", name);
    if (++filled == 64) full = 1;
}

/* Cached PICOLM_NO_IMMA check. getenv() is O(n) in env list and not
 * thread-safe on all platforms; cache the result on first call. */
static int picolm_gpu_no_imma(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("PICOLM_NO_IMMA") ? 1 : 0;
    return cached;
}

/* Same argument for the other per-launch switches. These are read on EVERY
 * matmul dispatch -- ~350 launches per decode token -- and MSVC's getenv()
 * walks the whole environment block with strcmp each time. Measured on the
 * 4090: ~71 us of host CPU per launch, of which these getenv() calls were the
 * dominant non-CUDA cost (a launch should be ~3-5 us). Cached once, exactly
 * like PICOLM_NO_IMMA above.
 *
 * None of these are expected to change at runtime; an env var that could be
 * toggled mid-process would be a different design. */
static int picolm_gpu_env_flag(const char *name) {
    return getenv(name) ? 1 : 0;
}
static int picolm_gpu_no_gemv(void) {
    static int cached = -1;
    if (cached < 0) cached = picolm_gpu_env_flag("PICOLM_NO_GEMV");
    return cached;
}

/* K-quant (and IQ6_K) decode GEMV for this type, 0 if none. These kernels
 * are plain CUDA (DP4A), no IMMA needed. */
static int kquant_which(gguf_type_t qt) {
    switch ((int)qt) {
    case GGUF_TYPE_Q2_K: return 2;
    case GGUF_TYPE_Q3_K: return 3;
    case GGUF_TYPE_Q4_K: return 4;
    case GGUF_TYPE_Q5_K: return 5;
    case GGUF_TYPE_Q6_K: return 6;
    case GGUF_TYPE_IQ6_K: return 141;
    default: return 0;
    }
}
static int kquant_gemv_dev(gpu_device_ctx_t *ctx, picolm_gpu_tensor_t *t, float *y_dev,
                           const float *x_dev, int I, int O, int ys, int x_stride, int which);
static int picolm_gpu_no_sm_imma(void) {
    static int cached = -1;
    if (cached < 0) cached = picolm_gpu_env_flag("PICOLM_NO_SM_IMMA");
    return cached;
}
/* Q8_0 prefill without IMMA: the DP4A GEMM, unless PICOLM_NO_Q8_DP4A asks
 * for the old tiled kernel (same output, bit for bit). CUDA only: the
 * kernel is written for 32-lane warps. */
/* PICOLM_NO_Q8_HMMA: on Volta, use the DP4A kernel instead of the tensor
 * core one for Q8_0 prefill. */
static int picolm_gpu_q8_hmma(void) {
    static int cached = -1;
    if (cached < 0) cached = !picolm_gpu_env_flag("PICOLM_NO_Q8_HMMA");
    return cached;
}
/* Q8_0 prefill via the DP4A GEMM. CUDA only (32-lane warps).
 *
 * NOTE this predicate does NOT look at the architecture, only the env flag.
 * Its callers all sit AFTER the has_imma branch, so on Turing/Ampere/Ada the
 * IMMA kernel returns first and this never fires -- correct, but only
 * positionally. Making it depend on the device keeps that true by
 * construction rather than by branch order. */
static int picolm_gpu_q8_dp4a(gpu_device_ctx_t *ctx) {
#ifdef __HIP__
    (void)ctx;
    return 0;
#else
    static int cached = -1;
    if (cached < 0) cached = !picolm_gpu_env_flag("PICOLM_NO_Q8_DP4A");
    return cached && ctx && !ctx->has_imma;
#endif
}

/* ================================================================
 * Q8_0 matmul kernel selection -- ONE place, used by all three callers
 * ================================================================
 *
 * The Q8_0 dispatch used to be written out three times:
 *     picolm_gpu_matmul()  (host pointers, does its own H2D/D2H)
 *     matmul_dev_impl()    (device pointers, no sync)
 *     imma_dev_q8()        (device pointers already quantised, used by the
 *                           fused QKV / gate-up paths)
 * with subtly different thresholds, and they had already drifted:
 *
 *     path            smw16          tiled fallback gate
 *     host            MISSING        S > 1
 *     dev             S>=32&&O>=16   S >= Q8_TILE_S   <- different!
 *     imma_dev_q8     S>=32&&O>=16   S >= Q8_TILE_S   <- different!
 *
 * The dev chains therefore sent every 2..31-row matmul to the SCALAR kernel
 * that the host chain happily ran tiled. That is a silent slowdown on the
 * hottest path in the backend, and exactly the class of bug the duplication
 * invites.
 *
 * Now: q8mm_pick() decides, q8mm_launch() launches. The callers pass the same
 * (xq, xd, weights, S, I, O, row_bytes, ys) they always had; only their
 * own scratch-buffer names differ, which is why the launcher takes those
 * directly rather than a ctx.
 *
 * Review concern #2 is also resolved here: picolm_gpu_q8_dp4a()'s
 * "!ctx->has_imma" precondition is tested INSIDE q8mm_pick(), which runs
 * before every branch, so reordering the chain can no longer silently
 * dead-code the DP4A path.
 */
typedef enum {
    Q8MM_GEMV = 0,      /* S==1 decode, warps cooperate on one row   */
    Q8MM_SMW16,         /* shared-memory staged IMMA, 32x16 tiles    */
    Q8MM_IMMA_W16,
    Q8MM_IMMA,
    Q8MM_HMMA,          /* Volta fp16 tensor cores                   */
    Q8MM_DP4A,          /* Volta/Pascal int8 dot                     */
    Q8MM_TILED,
    Q8MM_SCALAR
} q8mm_kind_t;

/* Pick the kernel. `allow_gemv` is false for the host path, which uploads its
 * activation panel once and always wants a tile kernel. */
static q8mm_kind_t q8mm_pick(gpu_device_ctx_t *ctx, int S, int O,
                             size_t row_bytes, int allow_gemv) {
    if (allow_gemv && S == 1 && O >= 1 && !picolm_gpu_no_gemv())
        return Q8MM_GEMV;

#ifndef __HIP__
    /* Shared-memory staged IMMA. Needs 544 B of dynamic shared memory and a
     * full 32-row tile to be worth the setup. */
    if (!picolm_gpu_no_sm_imma() && ctx->has_imma && !picolm_gpu_no_imma()
        && S >= 32 && O >= 16)
        return Q8MM_SMW16;
#endif
#if defined(PICOLM_IMMA_W16)
    if (ctx->has_imma && !picolm_gpu_no_imma() && S >= 16 && O >= 16)
        return Q8MM_IMMA_W16;
#endif
    if (ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8)
        return Q8MM_IMMA;
    if (S > 1 && ctx->has_hmma && picolm_gpu_q8_hmma())
        return Q8MM_HMMA;
    if (S > 1 && picolm_gpu_q8_dp4a(ctx))
        return Q8MM_DP4A;
    /* S > 1, not S >= Q8_TILE_S: the tiled kernel handles short batches fine
     * and the scalar kernel is much slower. This is the threshold the host
     * chain always used; the dev chains were the ones that disagreed. */
    if (S > 1 && row_bytes + 2048 <= 49152)
        return Q8MM_TILED;
    return Q8MM_SCALAR;
}

/* Launch the chosen kernel. Returns 0 if the launch itself failed, so the
 * callers can fall through to what they did before. */
static int q8mm_launch(q8mm_kind_t kind, gpu_device_ctx_t *ctx,
                       float *y, const int8_t *xq, const float *xd,
                       const picolm_gpu_tensor_t *t, int S, int I, int O,
                       int row_bytes, int ys) {
    switch (kind) {
    case Q8MM_SMW16: {
        gpu_dispatch_print("matmul_imma_smw16");
        dim3 grid((unsigned)((O + 15) / 16), (unsigned)((S + 31) / 32));
        int smem = 16 * GPU_BLOCK_Q8_0_SIZE;
        picolm_q8_q8_matmul_imma_smw16<<<grid, 64, smem, ctx->stream>>>(
            y, xq, xd, t->weights, S, I, O, row_bytes, ys);
        break;
    }
#if defined(PICOLM_IMMA_W16)
    case Q8MM_IMMA_W16: {
        gpu_dispatch_print("matmul_imma_w16");
        dim3 grid((unsigned)((O + 15) / 16), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma_w16<<<grid, 32, 0, ctx->stream>>>(
            y, xq, xd, t->weights, S, I, O, row_bytes, ys);
        break;
    }
#endif
    case Q8MM_IMMA: {
        gpu_dispatch_print("matmul_imma");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            y, xq, xd, t->weights, S, I, O, row_bytes, ys);
        break;
    }
    case Q8MM_HMMA: {
        gpu_dispatch_print("matmul_hmma");
        dim3 grid((unsigned)((O + 63) / 64), (unsigned)((S + 127) / 128));
        picolm_q8_q8_matmul_hmma<<<grid, 128, 0, ctx->stream>>>(
            y, xq, xd, t->weights, S, I, O, row_bytes, ys);
        break;
    }
    case Q8MM_DP4A: {
        gpu_dispatch_print("matmul_dp4a");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 7) / 8));
        picolm_q8_q8_matmul_dp4a<<<grid, 256, 0, ctx->stream>>>(
            y, xq, xd, t->weights, S, I, O, row_bytes, ys);
        break;
    }
    case Q8MM_TILED: {
        gpu_dispatch_print("matmul_tiled");
        dim3 grid((unsigned)O, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)row_bytes, ctx->stream>>>(
            y, xq, xd, t->weights, S, I, O, row_bytes, ys);
        break;
    }
    case Q8MM_SCALAR: {
        gpu_dispatch_print("matmul_scalar");
        dim3 grid((unsigned)O, (unsigned)S);
        picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
            y, xq, xd, t->weights, S, I, O, row_bytes, ys);
        break;
    }
    case Q8MM_GEMV:
    default:
        return 0;   /* callers handle GEMV themselves (needs its own args) */
    }
    return gpu_ok(gpuGetLastError(), "q8 matmul launch");
}

static int picolm_gpu_force_f32(void) {
    static int cached = -1;
    if (cached < 0) cached = picolm_gpu_env_flag("PICOLM_FORCE_F32_MATMUL");
    return cached;
}
static int picolm_gpu_no_fused_qkv(void) {
    static int cached = -1;
    if (cached < 0) cached = picolm_gpu_env_flag("PICOLM_NO_FUSED_QKV");
    return cached;
}

gpu_device_ctx_t g_gpu_ctx[PICOLM_GPU_MAX_DEVICES];
int g_nctx;

/* Last device passed to gpuSetDevice() through select_ctx(). See the comment
 * on select_ctx() in backend_gpu_common.cuh for why this is cached. Any code
 * that calls gpuSetDevice() directly must reset this to -1. */
int g_gpu_selected_device = -1;

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

/* Forward declarations for CPU dequant/quant functions (from quant.c).
 * These are extern "C" in quant.c, linked from the .cu object file. */
extern void dequantize_row_q2_K(const void *, float *, int);
extern void dequantize_row_q3_K(const void *, float *, int);
extern void dequantize_row_q4_K(const void *, float *, int);
extern void dequantize_row_q5_K(const void *, float *, int);
extern void dequantize_row_q6_K(const void *, float *, int);
extern void dequantize_row_iq4_nl(const void *, float *, int);
extern void quantize_row_q8_0(const float *, void *, int);

/* ---- Public API ---- */

extern "C"
int picolm_gpu_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > PICOLM_GPU_MAX_DEVICES) return 0;
#ifndef __HIP__
    /* Windows WDDM: On Windows, the default-stream gpuMemcpy() may return
     * before the data is actually readable by device kernels on ctx->stream.
     * The host-path matmul (picolm_gpu_matmul) works around this with explicit
     * gpuDeviceSynchronize() after each H2D. For safety, also reduce the
     * number of hardware work queues to 1 for more deterministic scheduling.
     *
     * On Windows: _putenv_s() crashes because this TU (compiled by nvcc/MSVC)
     * may use a different CRT instance than the one the CUDA driver queries
     * getenv() through. Mixing CRT-owned environment blocks corrupts the
     * process env block. SetEnvironmentVariableA writes the OS-level process
     * environment block directly, visible to all CRTs and the CUDA driver.
     * We forward-declare the Win32 APIs to avoid pulling in <windows.h>
     * (min/max/byte macro pollution) into a .cu file with CUDA/C++ headers.
     * See also: picolm.c uses the same SetEnvironmentVariableA pattern. */
#ifdef _WIN32
    {
        char existing[8];
        if (GetEnvironmentVariableA("CUDA_DEVICE_MAX_CONNECTIONS", existing, sizeof(existing)) == 0) {
            SetEnvironmentVariableA("CUDA_DEVICE_MAX_CONNECTIONS", "1");
        }
    }
#else
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1", 0);
#endif
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
        ctx->has_hmma = (prop.major == 7 && prop.minor == 0);  /* Volta */
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
        /* On Windows WDDM, force CUDA context creation early to avoid TDR timeout */
        { void *tmp; if (gpuMalloc(&tmp, 256) == gpuSuccess) gpuFree(tmp); }

        g_nctx++;
        fprintf(stderr, "[GPU] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                device, prop.name, prop.totalGlobalMem / 1e9,
                prop.major, prop.minor);
    }
    return 1;
}

extern "C"
void picolm_gpu_shutdown(void) {
    /* Resolve and print PICOLM_GPU_PROF totals before tearing anything down. */
    { extern void picolm_gpu_prof_report(void); picolm_gpu_prof_report(); }
    { picolm_gpu_launch_tmr_report(); }
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
 * F32 (0), F16 (1), BF16 (30): no blocks, each is an individual element.
 *
 * IMPORTANT: When adding a new quantization format, this function MUST be
 * updated, or GPU tensor uploads will silently fail (return 0 from
 * gguf_block_size -> upload function returns 0 -> NULL weight pointers ->
 * garbled GPU output). See also picolm_quant_matmul kernel switch and
 * picolm_gpu_upload_f32 for F32 tensors. */
static int gguf_block_size(gguf_type_t qtype) {
    switch (qtype) {
    case 0:  return 0;    /* F32: no blocks */
    case 1:  return 0;    /* F16: no blocks */
    case 30: return 0;    /* BF16: no blocks */
    case 2:  return 18;   /* Q4_0: 18 bytes per 32 values */
    case 3:  return 20;   /* Q4_1: 20 bytes per 32 values */
    case 8:  return GPU_BLOCK_Q8_0_SIZE; /* Q8_0: 34 bytes per 32 values */
    case 10: return 84;    /* Q2_K: 84 bytes per 256 values */
    case 11: return GPU_BLOCK_Q3_K_SIZE;  /* Q3_K: 110 bytes per 256 values */
    case 12: return GPU_BLOCK_Q4_K_SIZE;  /* Q4_K: 144 bytes per 256 values */
    case 13: return GPU_BLOCK_Q5_K_SIZE;  /* Q5_K: 176 bytes per 256 values */
    case 14: return GPU_BLOCK_Q6_K_SIZE;  /* Q6_K: 210 bytes per 256 values */
    case 20: return 18;   /* IQ4_NL: 18 bytes per 32 values (LUT-based, same layout as Q4_0) */
    case 41: return 18;   /* Q1_0: 18 bytes per 128 values */
    case 42: return 34;   /* Q2_0: 34 bytes per 128 values */
    case 133: return 26;  /* Q6_0: 26 bytes per 32 values (F16 d + 8 qh + 16 qs) */
    /* Q8_0_R8 is 8 rows interleaved into 272 bytes, i.e. 34 bytes per row per
     * 32 values -- the same per-row density as Q8_0. It has no native GPU
     * kernel; picolm_gpu_tensor_upload repacks it to Q8_0 on the way in (see
     * the Q8_0_R8 branch there), so it runs the ordinary Q8_0 kernels. */
    case 203: return 34;
    case 141: return GPU_BLOCK_IQ6_K_SIZE; /* IQ6_K: 212 bytes per 256 values */
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
    if (!bs && qtype != 0 && qtype != 1 && qtype != 30) {
        fprintf(stderr, "[GPU] upload FAIL: unsupported quantization type %d (I=%d O=%d)\n"
                "  FIX: add case %d: return <block_size>; to gguf_block_size() in backend_gpu_host_core.cu\n"
                "  Also: add dequant case to picolm_quant_matmul in backend_gpu_kernels.cu\n",
                (int)qtype, I, O, (int)qtype);
        return 0;
    }
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
    } else if (qtype == 10 || qtype == 11 || qtype == 12 || qtype == 13 || qtype == 14 || qtype == 141) {
        vals_per_block = 256;
        row_bytes = (size_t)((I + vals_per_block - 1) / vals_per_block) * bs;
    } else if (qtype == 41 || qtype == 42) {
        vals_per_block = 128;
        row_bytes = (size_t)((I + vals_per_block - 1) / vals_per_block) * bs;
    } else {
        vals_per_block = 32;
        row_bytes = (size_t)((I + vals_per_block - 1) / vals_per_block) * bs;
    }
    /* Q8_0_R8 -> Q8_0 repack on upload.
     *
     * block_q8_0_r8 is 272 bytes: uint16 d[8] then int8 qs[256] holding 8 rows
     * of 32 values interleaved. The GPU has no _R8 kernel and does not need
     * one: the CPU keeps using the interleaved form (its AVX2/VNNI GEMV wants
     * it), while the GPU gets plain Q8_0 and runs the existing kernels, so a
     * mixed model like IQ6_K + Q8_0_R8 gets the CPU fast path AND full GPU
     * coverage instead of the whole pipeline being vetoed as "unsupported".
     *
     * The conversion is a pure permutation -- no arithmetic, and d is already
     * FP16 exactly as block_q8_0 stores it. Per row r, the 32 int8 values are
     *   qs[m]      = b->qs[32*l + 4*r + i]        m = 4*l + i,      l<4, i<4
     *   qs[m + 16] = b->qs[32*l + 4*r + i + 128]  (same l, i)
     * which is what dequantize_row_q8_0_r8_single() reads. */
    if (qtype == 203) {
        if ((I % 32) != 0) return 0;
        const int nb = I / 32;                    /* _R8 blocks per row */
        size_t out_row_bytes = (size_t)nb * 34;   /* block_q8_0 row */
        size_t out_total = out_row_bytes * (size_t)O;
        uint8_t *packed = (uint8_t *)malloc(out_total);
        if (!packed) return 0;
        const uint8_t *src = (const uint8_t *)weights;
        /* one physical _R8 group covers 8 logical rows */
        for (int og = 0; og < (O + 7) / 8; og++) {
            for (int r = 0; r < 8; r++) {
                const int row = og * 8 + r;
                if (row >= O) break;
                for (int ib = 0; ib < nb; ib++) {
                    const uint8_t *b = src + ((size_t)og * nb + ib) * 272;
                    uint8_t *o = packed + (size_t)row * out_row_bytes + (size_t)ib * 34;
                    uint16_t d = (uint16_t)(b[2*r] | ((uint16_t)b[2*r + 1] << 8));
                    o[0] = (uint8_t)(d & 0xFF);
                    o[1] = (uint8_t)(d >> 8);
                    const uint8_t *q = b + 16;
                    for (int l = 0; l < 4; l++) {
                        for (int i = 0; i < 4; i++) {
                            o[2 + 4*l + i]      = (uint8_t)q[32*l + 4*r + i];
                            o[2 + 4*l + i + 16] = (uint8_t)q[32*l + 4*r + i + 128];
                        }
                    }
                }
            }
        }
        picolm_gpu_tensor_t *tr = (picolm_gpu_tensor_t *)calloc(1, sizeof(*tr));
        if (!tr) { free(packed); return 0; }
        tr->qtype = (gguf_type_t)8;              /* now ordinary Q8_0 */
        tr->I = I; tr->O = O; tr->device = device;
        tr->row_bytes = out_row_bytes;
        tr->block_size = 34;
        if (!gpu_ok(gpuMalloc(&tr->weights, out_total), "tensor allocation (q8_0_r8 repack)") ||
            !gpu_ok(gpuMemcpy(tr->weights, packed, out_total, gpuMemcpyHostToDevice),
                    "tensor upload (q8_0_r8 repack)")) {
            free(packed); gpuFree(tr->weights); free(tr); return 0;
        }
        free(packed);
        tr->zero_copy = 0;
        tr->tracked = 1;
        ctx->tensor_count++;
        ctx->tensor_bytes += out_total;
        {
            static int r8_print = 1;
            if (r8_print) {
                r8_print = 0;
                fprintf(stderr, "[GPU] upload: Q8_0_R8 (type 203) repacked to Q8_0 "
                                "(%d rows x %d blocks, %zu MB) -- CPU keeps the _R8 form\n",
                        O, nb, out_total / (1024*1024));
            }
        }
        *tp = tr;
        return 1;
    }

    size_t total = row_bytes * (size_t)O;

    picolm_gpu_tensor_t *t = (picolm_gpu_tensor_t *)calloc(1, sizeof(*t));
    if (!t) return 0;
    t->qtype = qtype; t->I = I; t->O = O; t->device = device;
    t->row_bytes = row_bytes; t->block_size = bs;

    /* Native K-quant upload path. All K-quant types (Q2_K through Q6_K)
     * use native IMMA fast path when eligible. Fall back to scalar
     * dequant when IMMA cannot fire (small S/O, non-256-aligned I). */
    if ((qtype == 10 || qtype == 11 || qtype == 12 || qtype == 13 || qtype == 14 || qtype == 141) && (I % 256) == 0 && O >= 8) {
        size_t blk_size = (qtype == 141) ? GPU_BLOCK_IQ6_K_SIZE :
                          (qtype == 14) ? GPU_BLOCK_Q6_K_SIZE :
                          (qtype == 13) ? GPU_BLOCK_Q5_K_SIZE :
                          (qtype == 12) ? GPU_BLOCK_Q4_K_SIZE :
                          (qtype == 11) ? GPU_BLOCK_Q3_K_SIZE :
                                          GPU_BLOCK_Q2_K_SIZE;
        size_t qk_total = (size_t)I / 256 * blk_size * (size_t)O;
        if (!gpu_ok(gpuMalloc(&t->weights, qk_total), "tensor allocation (qk)") ||
            !gpu_ok(gpuMemcpy(t->weights, weights, qk_total, gpuMemcpyHostToDevice),
                    "tensor upload (qk)")) {
            gpuFree(t->weights); free(t); return 0;
        }
        t->qtype = (gguf_type_t)qtype;
        t->block_size = blk_size;
        t->row_bytes = (size_t)I / 256 * blk_size;
        t->zero_copy = 0;
        t->tracked = 1;
        ctx->tensor_count++;
        ctx->tensor_bytes += qk_total;
        {
            static int first_print = 1;
            if (first_print) {
                const char *qname = (qtype == 10) ? "Q2_K" :
                                    (qtype == 11) ? "Q3_K" :
                                    (qtype == 12) ? "Q4_K" :
                                    (qtype == 13) ? "Q5_K" : "Q6_K";
                /* Check if IMMA is available on this GPU */
                const char *path = "GPU dequant + IMMA";
                if (!ctx->has_imma) path = "GPU dequant (no IMMA, scalar fallback)";
                fprintf(stderr,
                    "[GPU] upload mode: %s native (%s, %zu MB)\n",
                    qname, path, qk_total / (1024*1024));
                first_print = 0;
            }
        }
        *tp = t;
        return 1;
    }
    if (qtype == 10 || qtype == 11 || qtype == 12 || qtype == 13 || qtype == 14) {
        /* Fallback for K-quantized types when native path unavailable:
         * dequant to F32, requant to Q8_0 */
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
                const char *qname = (qtype == 10) ? "Q2_K" :
                                    (qtype == 11) ? "Q3_K" :
                                    (qtype == 12) ? "Q4_K" :
                                    (qtype == 13) ? "Q5_K" : "Q6_K";
                size_t native_mb = (size_t)O * row_bytes / (1024*1024);
                size_t q8_mb = q8_total / (1024*1024);
                fprintf(stderr,
                    "[GPU] K-quant fallback: %s -> Q8_0 "
                    "(I%%256!=0 or O<8: I=%d O=%d) "
                    "VRAM %zu->%zu MB (%.0f%% of native)\n",
                    qname, I, O, native_mb, q8_mb,
                    (double)q8_total / (row_bytes * (size_t)O) * 100.0);
                first_print = 0;
            }
        }
        *tp = t;
        return 1;
    }

    /* IQ4_NL (type 20): convert to Q8_0 at upload time.
     * On GPUs without IMMA/MFMA (e.g. MI50), the scalar picolm_quant_matmul
     * dequant path with LUT is extremely slow. Converting to Q8_0 on CPU at
     * upload time lets us use the much faster Q8_0 scalar int8 MAC path.
     * On GPUs with IMMA (NVIDIA), a native IQ4_NL IMMA kernel would be better,
     * but Q8_0 fallback is still fast enough. */
    if (qtype == 20) {
        float *f32_buf = (float *)calloc(I * O, sizeof(float));
        if (!f32_buf) { gpuFree(t->weights); free(t); return 0; }

        /* Dequant IQ4_NL to F32 on CPU */
        for (int row = 0; row < O; row++) {
            float *dst = f32_buf + row * I;
            const void *row_start = (const uint8_t *)weights + row * ((I + 31) / 32) * 18;
            dequantize_row_iq4_nl(row_start, dst, I);
        }

        /* Requant to Q8_0 */
        size_t q8_total = (size_t)O * ((I + 31) / 32) * 34;
        uint8_t *q8_buf = (uint8_t *)calloc(q8_total, 1);
        if (!q8_buf) { free(f32_buf); gpuFree(t->weights); free(t); return 0; }

        for (int row = 0; row < O; row++) {
            const float *src = f32_buf + row * I;
            block_q8_0 *q8_blocks = (block_q8_0 *)(q8_buf + row * ((I + 31) / 32) * 34);
            quantize_row_q8_0(src, q8_blocks, I);
        }
        free(f32_buf);

        if (!gpu_ok(gpuMalloc(&t->weights, q8_total), "tensor allocation (iq4_nl->q8)") ||
            !gpu_ok(gpuMemcpy(t->weights, q8_buf, q8_total, gpuMemcpyHostToDevice),
                    "tensor upload (iq4_nl->q8)")) {
            free(q8_buf); gpuFree(t->weights); free(t); return 0; }
        free(q8_buf);

        t->qtype = (gguf_type_t)8;
        t->block_size = 34;
        t->row_bytes = ((I + 31) / 32) * 34;
        t->zero_copy = 0;
        t->tracked = 1;
        ctx->tensor_count++;
        ctx->tensor_bytes += q8_total;
        static int iq4nl_print = 1;
        if (iq4nl_print) {
            size_t native_mb = ((I + 31) / 32) * 18 * (size_t)O / (1024*1024);
            size_t q8_mb = q8_total / (1024*1024);
            fprintf(stderr, "[GPU] IQ4_NL -> Q8_0 conversion: %zu MB native -> %zu MB Q8_0 (%.0f%% overhead, scalar LUT dequant too slow on GPU)\n",
                    native_mb, q8_mb, (double)q8_total / (((I + 31) / 32) * 18 * (size_t)O) * 100.0);
            iq4nl_print = 0;
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

    /* Per-quant-type upload summary (printed once per type) */
    {
        static int seen_types[32] = {0};
        static int seen_count = 0;
        int idx = -1;
        if (qtype < 32) {
            if (!seen_types[qtype]) {
                seen_types[qtype] = 1;
                idx = qtype;
            }
        }
        if (idx >= 0 && seen_count < 4) {
            const char *qname = "???";
            switch (qtype) {
                case 0: qname = "F32"; break;
                case 1: qname = "F16"; break;
                case 2: qname = "Q4_0"; break;
                case 8: qname = "Q8_0"; break;
                case 30: qname = "BF16"; break;
                case 41: qname = "Q1_0"; break;
                case 42: qname = "Q2_0"; break;
            }
            /* Print the real size. Integer MB truncation made a 256-byte F32
             * vector print as "0 MB", which reads like a bug in the upload. */
            if (total >= 1024 * 1024 || getenv("PICOLM_UPLOAD_DBG"))
                fprintf(stderr, "[GPU] upload: %s (type %d) native, copied to GPU (%.2f MB, %zu bytes)\n",
                        qname, qtype, (double)total / (1024.0 * 1024.0), total);
            else
                fprintf(stderr, "[GPU] upload: %s (type %d) native, copied to GPU (%zu bytes)\n",
                        qname, qtype, total);
            seen_count++;
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
static int matmul_host_impl(picolm_gpu_tensor_t *t, float *y, const float *x, int S, int device) {
    if (!t || !y || !x || S < 1) return 0;
    /* Minimum I for GPU to be worthwhile vs CPU kernel launch overhead.
     * Smaller tensors: kernel launch (~200us) dominates compute. */
    if (t->I < 512) return 0;  /* O gate removed: blocked K/V projections with O<256 */
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
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();
        /* TODO(optimization): This per-matmul H2D + sync + D2H cycle is ~50ms
         * overhead per layer on the host path. With NGL=1, prefill drops to
         * ~20 tok/s vs ~287 tok/s CPU-only. The real fix is to route NGL
         * through the device path (picolm_gpu_matmul_dev) with device-resident
         * buffers, eliminating per-matmul H2D/D2H entirely. See also:
         * picolm_gpu_matmul_dev() below and model_forward_prefill_gpu(). */

        /* Allocate quantized input buffers: qs (int8_t[S*I]) + d (float[S*n_blocks])
         * Round S up to multiple of 16 for IMMA kernel (reads up to S+15 in tail tile).
         * The quantize kernel writes exactly S rows; the IMMA kernel reads up to S+15
         * but only writes output for rows < S. Padding prevents OOB reads. */
        int S_padded = (S + 15) & ~15;
        /* TODO(rc1): The quantize kernel (picolm_quantize_q8_0) computes
         * n_blocks = (I+31)/32 and writes S * n_blocks * 32 bytes to qs_out.
         * xq_bytes = S_padded * I is correct when I%32==0 (always guaranteed
         * by caller guards: I%32!=0 or I%256==0). If any caller ever passes
         * non-32-aligned I, this becomes an OOB write. Consider using
         * xq_bytes = S_padded * n_blocks * 32 for defensive correctness.
         * Applies to all xq_bytes allocations in this file (24 sites). */
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        /* Zero-fill before quantize (matches device path: gpuMemsetAsync).
         * reserve() only zeros on initial allocation. On reuse, stale data
         * from a larger previous S can leak into IMMA tail tiles. */
        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);

        /* Step 1: Quantize x to Q8_0 on GPU */
    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        /* One selector for all three chains -- see q8mm_pick(). The host path
         * never uses the GEMV (it uploads a full activation panel), and it now
         * gets the smw16 kernel it was previously missing. */
        if (!q8mm_launch(q8mm_pick(ctx, S, O, t->row_bytes, 0), ctx,
                         ctx->y, ctx->q8_xq, ctx->q8_xd, t, S, I, O,
                         (int)t->row_bytes, O)) return 0;
        if (!gpu_ok(gpuGetLastError(), "q8 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 matmul sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* Q4_0 x Q8_0 IMMA path (host) */
    if (t->qtype == GGUF_TYPE_Q4_0 && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0;
        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();

        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);
    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        gpu_dispatch_print("matmul_q4_0_imma_host");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q4_0_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        if (!gpu_ok(gpuGetLastError(), "q4_0 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q4_0 matmul sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* Q4_1 x Q8_0 IMMA path (host) */
    if (t->qtype == GGUF_TYPE_Q4_1 && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0;
        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();

        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);

    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        gpu_dispatch_print("matmul_q4_1_imma_host");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q4_1_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        if (!gpu_ok(gpuGetLastError(), "q4_1 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q4_1 matmul sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* TODO: Q6_K decode warp-shuffle path (S=1) -- DISABLED: out-of-bounds ql read
     *
     * Root cause (2025-08-31): The kernel's nested loop structure (half/sub_chunk/sc_half/sub_l)
     * caused ql_c[sub_l+32] reads to exceed the 64-byte per-half ql window when sc_half=1
     * (sub_l=16..47, so sub_l+32=48..79). For half=1, ql_c=ql+64, meaning ql[143] was read
     * past the 128-byte ql array into qh memory, producing garbage dequant values.
     *
     * A fix was attempted by restructuring the loop to match the CPU reference exactly,
     * but the kernel remained broken. The scalar fallback path (picolm_quant_matmul)
     * produces correct output for Q6_K decode, so the decode warp is left disabled.
     * The warp approach is only useful as a speedup for S=1 decode; correctness comes
     * from the scalar path. See /data4/work/notes/picolm/q6k-decode-warp-bug.md */
    if (0 && t->qtype == GGUF_TYPE_Q6_K && S == 1 && I % 256 == 0) {
        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();

        gpu_dispatch_print("matmul_q6_decode_host");
        dim3 grid((unsigned)O);
        picolm_q6_k_decode_warp<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->x, t->weights, I, O, (int)t->row_bytes);
        if (!gpu_ok(gpuGetLastError(), "q6 decode warp") ||
            !gpu_ok(gpuDeviceSynchronize(), "q6 decode warp sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* Q2_0 x Q8_0 IMMA path (host) */
    if (t->qtype == GGUF_TYPE_Q2_0 && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0;
        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();

        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);

    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        gpu_dispatch_print("matmul_q2_0_imma_host");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q2_0_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        if (!gpu_ok(gpuGetLastError(), "q2_0 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q2_0 matmul sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* Q1_0 x Q8_0 IMMA path (host) */
    if (t->qtype == GGUF_TYPE_Q1_0 && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0;
        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();

        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);

    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        gpu_dispatch_print("matmul_q1_0_imma_host");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q1_0_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        if (!gpu_ok(gpuGetLastError(), "q1_0 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q1_0 matmul sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* K-quant decode (S == 1) on the GEMVs. They used to be reachable only
     * through the has_imma branches, so a GPU without IMMA ran K-quant decode
     * on the per-element dequant matmul. IMMA GPUs keep their IMMA path here. */
    if (!ctx->has_imma && S == 1 && O >= 8 && I % 256 == 0 && kquant_which(t->qtype) && !picolm_gpu_no_gemv()) {
        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        if (kquant_gemv_dev(ctx, t, ctx->y, ctx->x, I, O, O, 0, kquant_which(t->qtype))) {
            if (!gpu_ok(gpuStreamSynchronize(ctx->stream), "kquant gemv sync") ||
                !gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
            return 1;
        }
    }

    /* Q6_K x Q8_0 IMMA path: quantize activations to Q8_0, native Q6_K weights. */
    if (t->qtype == GGUF_TYPE_Q6_K && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8 && I % 256 == 0) {
        int n_blocks = I / 32;
        if (n_blocks < 1) return 0;

        /* Upload fp32 input */
        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();

        /* Quantize x to Q8_0 on GPU (same as Q8_0 path) */
        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);

    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        gpu_dispatch_print("matmul_q6_imma_host");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q6_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        if (!gpu_ok(gpuGetLastError(), "q6 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q6 matmul sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* Q2_K x Q8_0 IMMA path: quantize activations to Q8_0, native Q2_K weights. */
    if (t->qtype == GGUF_TYPE_Q2_K && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8 && I % 256 == 0) {
        int n_blocks = I / 32;
        if (n_blocks < 1) return 0;

        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();

        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);

    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        gpu_dispatch_print("matmul_q2_imma_host");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q2_k_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        if (!gpu_ok(gpuGetLastError(), "q2 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q2 matmul sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* Q3_K x Q8_0 IMMA path: quantize activations to Q8_0, native Q3_K weights. */
    if (t->qtype == GGUF_TYPE_Q3_K && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8 && I % 256 == 0) {
        int n_blocks = I / 32;
        if (n_blocks < 1) return 0;

        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();

        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);

    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        gpu_dispatch_print("matmul_q3_imma_host");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q3_k_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        if (!gpu_ok(gpuGetLastError(), "q3 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q3 matmul sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* Q5_K x Q8_0 IMMA path: quantize activations to Q8_0, native Q5_K weights. */
    if (t->qtype == GGUF_TYPE_Q5_K && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8 && I % 256 == 0) {
        int n_blocks = I / 32;
        if (n_blocks < 1) return 0;

        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();

        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);

    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        gpu_dispatch_print("matmul_q5_imma_host");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q5_k_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        if (!gpu_ok(gpuGetLastError(), "q5 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q5 matmul sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* Q4_K x Q8_0 IMMA host path: ENABLED (device-side disabled). */
    if (t->qtype == GGUF_TYPE_Q4_K && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8 && I % 256 == 0) {
        int n_blocks = I / 32;
        if (n_blocks < 1) return 0;

        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();

        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);

    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        gpu_dispatch_print("matmul_q4_imma_host");

        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q4_k_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        if (!gpu_ok(gpuGetLastError(), "q4 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q4 matmul sync")) return 0;

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
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();
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
        /* WDDM: default-stream H2D may return before device reads */
        gpuDeviceSynchronize();

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

/* Profiled wrapper for the HOST matmul path.
 *
 * The dev wrapper (matmul_dev_impl) has per-quant slots; this one had NONE, and
 * prefill uses both. That is how q6_K's prefill cost became invisible while
 * matmul_q6_imma_mt_dev was still running: the FFN down projection dispatches
 * here, not through the dev path. Same slot names, so a quant type reports one
 * line whichever path served it -- add "_host" only if the split is needed.
 *
 * NOTE the slot is taken with S as passed, so this uses the same decode/prefill
 * split as matmul_dev_prof_slot(): S<=1 is decode and is already covered by the
 * GEMV slots inside the branches, so only S>1 is timed here. */
static int matmul_dev_prof_slot(const picolm_gpu_tensor_t *t, int S);

extern "C" int
picolm_gpu_matmul(picolm_gpu_tensor_t *t, float *y, const float *x, int S, int device) {
    int ps = (t && S > 1) ? matmul_dev_prof_slot(t, S) : -1;
    if (ps >= 0) picolm_gpu_prof_begin(device);
    int rc = matmul_host_impl(t, y, x, S, device);
    if (ps >= 0) picolm_gpu_prof_end(ps);
    return rc;
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
/* Q8_K activation scratch + one GEMV launch, shared by the Q5_K and Q6_K
 * decode paths. Returns 1 on success, 0 to let the caller fall back to the
 * IMMA tile paths (which still use Q8_0 activations). */
static int
kquant_gemv_dev(gpu_device_ctx_t *ctx, picolm_gpu_tensor_t *t, float *y_dev,
                const float *x_dev, int I, int O, int ys, int x_stride,
                int which /* 5 or 6 */) {
    const int n_super = I / 256;
    if (n_super < 1 || I % 256 != 0) return 0;
    if (x_stride > 0 && x_stride != I) return 0;   /* GEMV assumes packed x */

    dim3 q_grid((unsigned)n_super, 1);
    size_t qs_bytes    = (size_t)n_super * 256;
    size_t d_bytes     = (size_t)n_super * sizeof(float);
    size_t bsums_bytes = (size_t)n_super * 16 * sizeof(int16_t);
    if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, qs_bytes) ||
        !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, d_bytes)) return 0;
    if (ctx->q8_k_bsums_cap < bsums_bytes) {
        if (ctx->q8_k_bsums) gpuFree(ctx->q8_k_bsums);
        if (!gpu_ok(gpuMalloc(&ctx->q8_k_bsums, bsums_bytes), "q8_k bsums alloc")) {
            ctx->q8_k_bsums = NULL; ctx->q8_k_bsums_cap = 0; return 0;
        }
        ctx->q8_k_bsums_cap = bsums_bytes;
    }

    dim3 grid((unsigned)O);
    /* PICOLM_GEMV_DBG=1: one line per K-quant GEMV launch, with the layer it
     * belongs to, so per-token weight traffic can be attributed to individual
     * shapes. Use this rather than dividing a whole-run call count by the
     * token count: that mixes in prefill and overstates the per-token work. */
    if (getenv("PICOLM_GEMV_DBG"))
        fprintf(stderr, "[GEMV] L=%d which=%d I=%d O=%d row_bytes=%d\n",
                picolm_gpu_dbg_layer, which, I, O, (int)t->row_bytes);
    int _ps = picolm_gpu_prof_slot(which == 141 ? "gemv_iq6_K" : which == 2 ? "gemv_q2_K" : which == 3 ? "gemv_q3_K" : which == 4 ? "gemv_q4_K" : which == 5 ? "gemv_q5_K" : "gemv_q6_K");
    /* The begin must cover the activation quantize too: it is a real kernel on
     * this stream and, at one launch per matmul call, it is part of the cost
     * being attributed. This used to pass device 0 unconditionally as well,
     * which times the wrong context on a multi-device setup. */
    { int _psq = picolm_gpu_prof_slot("quantize_q8");
      picolm_gpu_prof_begin(ctx->device);
      picolm_quantize_q8_K<<<q_grid, 256, 0, ctx->stream>>>(
          ctx->q8_xd, ctx->q8_xq, ctx->q8_k_bsums, x_dev, I, 1);
      picolm_gpu_prof_end(_psq);
    }
    if (!gpu_ok(gpuGetLastError(), "q8_K quantize (dev)")) return 0;
    picolm_gpu_prof_begin(ctx->device);
    if (which == 141) {
        gpu_dispatch_print("matmul_iq6_gemv_dev");
        picolm_iq6_K_gemv<<<grid, 32 * GEMV_WARPS, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, ctx->q8_k_bsums, t->weights,
            I, O, (int)t->row_bytes, ys);
    } else if (which == 2) {
        gpu_dispatch_print("matmul_q2_gemv_dev");
        picolm_q2_K_gemv<<<grid, 32 * GEMV_WARPS, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, ctx->q8_k_bsums, t->weights,
            I, O, (int)t->row_bytes, ys);
    } else if (which == 3) {
        gpu_dispatch_print("matmul_q3_gemv_dev");
        picolm_q3_K_gemv<<<grid, 32 * GEMV_WARPS, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, ctx->q8_k_bsums, t->weights,
            I, O, (int)t->row_bytes, ys);
    } else {
        gpu_dispatch_print(which == 4 ? "matmul_q4_gemv_dev" : which == 5 ? "matmul_q5_gemv_dev" : "matmul_q6_gemv_dev");
        picolm_kquant_gemv_w_launch(which, ctx->stream, y_dev, ctx->q8_xq, ctx->q8_xd,
                                    t->weights, I, O, (int)t->row_bytes);
    }
    picolm_gpu_prof_end(_ps);
    return gpu_ok(gpuGetLastError(), "kquant gemv (dev)");
}

/* PICOLM_LAUNCH_TMR=1: accumulate the CPU wall time spent inside
 * picolm_gpu_matmul_dev itself (i.e. enqueue cost, NOT GPU execution), and
 * report it at exit. This is the number that decides whether a decode token is
 * launch-bound; PICOLM_GPU_PROF cannot see it because its events measure the
 * stream, not the host. */
static double g_launch_ms = 0.0;
static long   g_launch_n  = 0;
static int    g_launch_on = -1;
static double lt_now(void)
{
#if defined(_WIN32)
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
#endif
}
extern "C" void picolm_gpu_launch_tmr_report(void) {
    if (g_launch_on != 1 || g_launch_n == 0) return;
    fprintf(stderr, "[LNTMR] %ld matmul_dev calls, %.1f ms total, %.1f us/call (HOST side only)\n",
            g_launch_n, g_launch_ms, 1000.0 * g_launch_ms / (double)g_launch_n);
}

static int
matmul_dev_impl(picolm_gpu_tensor_t *t, float *y_dev, const float *x_dev,
                int S, int device, int y_stride, int x_stride) {
    if (!t || !y_dev || !x_dev || S < 1) return 0;
    if (t->I < 64) return 0;  /* O gate removed: blocked K/V projections with O<256 */
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    int I = t->I, O = t->O;
    int ys = y_stride > 0 ? y_stride : O;

    if (t->qtype == GGUF_TYPE_Q8_0 && !picolm_gpu_force_f32()) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0;
        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;
        /* The zero-fill only exists for the IMMA tile tail: those kernels read
         * S_padded rows and would otherwise pick up stale data from a previous,
         * larger S. The GEMV (S==1) reads exactly I values and touches no
         * padding, so it does not need it -- and at ~350 launches per decode
         * token, two extra driver calls per launch is real CPU time on the
         * critical path. Decided once, used by both this and the dispatch. */
        const int gemv_path = (S == 1 && O >= 1 && !picolm_gpu_no_gemv());
        if (!gemv_path) {
            gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
            gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);
        }
    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        if (x_stride > 0) {
            picolm_quantize_q8_0_strided<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
                ctx->q8_xq, ctx->q8_xd, x_dev, I, S, x_stride);
        } else {
            picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
                ctx->q8_xq, ctx->q8_xd, x_dev, I, S);
        }
        if (!gpu_ok(gpuGetLastError(), "q8 quantize (dev)")) return 0;

        /* Decode fast path: a real GEMV, one warp per weight row.
         *
         * Taken only at S==1. Everything below shapes a tensor-core tile, and
         * at S=1 that means 16 rows of mma for 1 useful row (94% waste) with
         * one 32-thread block per 8 output columns. GEMV streams each weight
         * row exactly once and does no padding work.
         *
         * PICOLM_NO_GEMV=1 forces the IMMA path back on, for A/B measurement
         * and for bisecting any numerical difference to a single kernel.
         * (PICOLM_NO_DP4A=1 separately forces the scalar int8 dot inside the
         * GEMV, so the two switches isolate the kernel shape from the ISA.) */
        if (gemv_path) {
            gpu_dispatch_print("matmul_q8_gemv_dev");
            /* One output row per block; all GEMV_WARPS warps of the block
             * cooperate on it (see the kernel comment for why cooperating on
             * one row beats assigning one row per warp). */
            dim3 grid((unsigned)O);
            if (getenv("PICOLM_GEMV_DBG"))
                fprintf(stderr, "[GEMV] L=%d which=8 I=%d O=%d row_bytes=%d\n",
                        picolm_gpu_dbg_layer, I, O, (int)t->row_bytes);
            int _ps = picolm_gpu_prof_slot("gemv_q8_0");
            picolm_gpu_prof_begin(device);
            picolm_q8_0_gemv<<<grid, 32 * GEMV_WARPS, 0, ctx->stream>>>(
                y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, I, O, (int)t->row_bytes, ys);
            picolm_gpu_prof_end(_ps);
            if (gpu_ok(gpuGetLastError(), "q8 gemv (dev)")) return 1;
            /* fall through to the tile paths if the GEMV launch failed */
        }

        /* Same selector as the host path. GEMV was already tried above, so
         * allow_gemv=0 here; everything else is shared. */
        if (!q8mm_launch(q8mm_pick(ctx, S, O, t->row_bytes, 0), ctx,
                         y_dev, ctx->q8_xq, ctx->q8_xd, t, S, I, O,
                         (int)t->row_bytes, ys)) return 0;
        if (!gpu_ok(gpuGetLastError(), "q8 matmul (dev)")) return 0;
        return 1;
    }

    /* Q4_0 x Q8_0 IMMA path (device) */
    if (t->qtype == GGUF_TYPE_Q4_0 && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8) {
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
        gpu_dispatch_print("matmul_q4_0_imma_dev");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q4_0_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q4_0 matmul (dev)")) return 0;
        return 1;
    }

    /* Q4_1 x Q8_0 IMMA path (device) */
    if (t->qtype == GGUF_TYPE_Q4_1 && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8) {
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
        gpu_dispatch_print("matmul_q4_1_imma_dev");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q4_1_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q4_1 matmul (dev)")) return 0;
        return 1;
    }

    /* TODO: Q6_K decode warp-shuffle path (device, S=1) -- DISABLED: out-of-bounds ql read
     * Same root cause as host path above. See comment at line ~701 and
     * /data4/work/notes/picolm/q6k-decode-warp-bug.md for details. */
    if (0 && t->qtype == GGUF_TYPE_Q6_K && S == 1 && I % 256 == 0) {
        gpu_dispatch_print("matmul_q6_decode_dev");
        dim3 grid((unsigned)O);
        picolm_q6_k_decode_warp<<<grid, 32, 0, ctx->stream>>>(
            y_dev, x_dev, t->weights, I, O, (int)t->row_bytes);
        if (!gpu_ok(gpuGetLastError(), "q6 decode warp (dev)")) return 0;
        return 1;
    }

    /* Q2_0 x Q8_0 IMMA path (device) */
    if (t->qtype == GGUF_TYPE_Q2_0 && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8) {
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
        gpu_dispatch_print("matmul_q2_0_imma_dev");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q2_0_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q2_0 matmul (dev)")) return 0;
        return 1;
    }

    /* Q1_0 x Q8_0 IMMA path (device) */
    if (t->qtype == GGUF_TYPE_Q1_0 && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8) {
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
        gpu_dispatch_print("matmul_q1_0_imma_dev");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q1_0_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q1_0 matmul (dev)")) return 0;
        return 1;
    }

    /* Q2_K x Q8_0 IMMA path (device) */
    /* IQ6_K: decode GEMV against Q8_K (mirrors the CPU, whose IQ6_K vec_dot
     * also consumes Q8_K). Guarded on (I % 256) == 0 like the other K-quants.
     * The generic picolm_quant_matmul fallback below has no IQ6_K dequant case,
     * so without this an IQ6_K tensor simply declines (returns 0) and the
     * caller falls back to the CPU. */
    if (t->qtype == GGUF_TYPE_IQ6_K && S >= 1 && O >= 8 && I % 256 == 0) {
        /* S==1 -> the purpose-built GEMV. Anything else falls through to the
         * generic picolm_quant_matmul below, which now has a case 141
         * per-element dequant. (An earlier version returned 0 here, which
         * looked safe but silently disabled IQ6_K for ALL of prefill: the
         * dequant case could never be reached.) */
        if (S == 1 && !picolm_gpu_no_gemv() && kquant_gemv_dev(ctx, t, y_dev, x_dev, I, O, ys, x_stride, 141))
            return 1;
    }

    /* K-quant decode on the GEMVs on any GPU; the has_imma branches below
     * used to be the only way in (see the host path). */
    if (S == 1 && O >= 8 && I % 256 == 0 && kquant_which(t->qtype) && !picolm_gpu_no_gemv() &&
        kquant_gemv_dev(ctx, t, y_dev, x_dev, I, O, ys, x_stride, kquant_which(t->qtype)))
        return 1;

    if (t->qtype == GGUF_TYPE_Q2_K && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8 && I % 256 == 0) {
        if (S == 1 && !picolm_gpu_no_gemv() && kquant_gemv_dev(ctx, t, y_dev, x_dev, I, O, ys, x_stride, 2))
            return 1;
        int n_blocks = I / 32;
        if (n_blocks < 1) return 0;
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
        gpu_dispatch_print("matmul_q2_imma_dev");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q2_k_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q2 matmul (dev)")) return 0;
        return 1;
    }

    /* Q3_K x Q8_0 IMMA path (device) */
    if (t->qtype == GGUF_TYPE_Q3_K && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8 && I % 256 == 0) {
        if (S == 1 && !picolm_gpu_no_gemv() && kquant_gemv_dev(ctx, t, y_dev, x_dev, I, O, ys, x_stride, 3))
            return 1;
        int n_blocks = I / 32;
        if (n_blocks < 1) return 0;
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
        gpu_dispatch_print("matmul_q3_imma_dev");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q3_k_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q3 matmul (dev)")) return 0;
        return 1;
    }

    /* Q6_K x Q8_0 IMMA path (device): quantize activations to Q8_0, native Q6_K weights.
     * Uses picolm_q6_q8_matmul_imma (m16n8k32 tensor cores, 2 IMMA per K-step). */
    if (t->qtype == GGUF_TYPE_Q6_K && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8 && I % 256 == 0) {
        /* Decode fast path: Q6_K GEMV against Q8_K activations. */
        if (S == 1 && !picolm_gpu_no_gemv() && kquant_gemv_dev(ctx, t, y_dev, x_dev, I, O, ys, x_stride, 6))
            return 1;
        int n_blocks = I / 32;
        if (n_blocks < 1) return 0;
        int S_padded = (S + 15) & ~15;
        size_t xq_bytes = (size_t)S_padded * I;
        size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;
        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);
    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        if (x_stride > 0) {
            picolm_quantize_q8_0_strided<<<q_grid, 32, 32*sizeof(float), ctx->stream>>>(
                ctx->q8_xq, ctx->q8_xd, x_dev, I, S, x_stride);
        } else {
            picolm_quantize_q8_0<<<q_grid, 32, 32*sizeof(float), ctx->stream>>>(
                ctx->q8_xq, ctx->q8_xd, x_dev, I, S);
        }
        if (!gpu_ok(gpuGetLastError(), "q8 quantize (dev)")) return 0;
        /* Multi-tile: see picolm_q6_k_q8_matmul_imma_mt. Weights were being
         * re-read once per 16-token block, exactly as Q5_K was.
         * PICOLM_NO_MT_Q6K=1 restores the single-tile kernel for A/B. */
        static int no_mt_q6k = -1;
        if (no_mt_q6k < 0) no_mt_q6k = getenv("PICOLM_NO_MT_Q6K") ? 1 : 0;
        if (!no_mt_q6k && S > 16) {
            gpu_dispatch_print("matmul_q6_imma_mt_dev");
            /* Same stride rule as the Q5_K launch above: blockIdx.y maps to
             * 16*Q6K_TILES_S rows, so the grid must divide by that. */
            dim3 grid((unsigned)((O + 7) / 8),
                      (unsigned)((S + 16 * Q6K_TILES_S - 1) / (16 * Q6K_TILES_S)));
            int smt = Q6K_TILES_S * 32 * 4 * (int)sizeof(float);
            picolm_q6_k_q8_matmul_imma_mt<<<grid, 32, smt, ctx->stream>>>(
                y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
            if (gpu_ok(gpuGetLastError(), "q6 matmul mt (dev)")) return 1;
        }
        gpu_dispatch_print("matmul_q6_imma_dev");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q6_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q6 matmul (dev)")) return 0;

        return 1;
    }

    /* Q5_K x Q8_0 IMMA path (device) -- ENABLED */
    if (t->qtype == GGUF_TYPE_Q5_K && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8 && I % 256 == 0) {
        /* Decode fast path: Q5_K GEMV against Q8_K activations, matching the
         * CPU's activation format for K-quant weights. */
        if (S == 1 && !picolm_gpu_no_gemv() && kquant_gemv_dev(ctx, t, y_dev, x_dev, I, O, ys, x_stride, 5))
            return 1;
        int n_blocks = I / 32;
        if (n_blocks < 1) return 0;
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
        /* Multi-tile kernel: decode the weight fragment once per k-block and
         * reuse it across Q5K_TILES_S 16-row tiles. The single-tile kernel
         * re-streams every weight byte once per 16-token block, which at a
         * 512-token ubatch is a 32x read amplification -- measured as kernel
         * time scaling exactly linearly with token count (1.03/2.10/4.27 ms
         * per call at 128/256/512 tokens).
         * PICOLM_NO_MT_Q5K=1 restores the single-tile kernel for A/B. */
        static int no_mt_q5k = -1;
        if (no_mt_q5k < 0) no_mt_q5k = getenv("PICOLM_NO_MT_Q5K") ? 1 : 0;
        /* Multi-warp kernel first: 8 warps per block instead of 1. The
         * single-warp multi-tile kernel measured 4% of bandwidth and 3.4% of
         * int8 peak simultaneously, i.e. occupancy-bound, which a 32-thread
         * block cannot fix. PICOLM_NO_W8_Q5K=1 falls back to it. */
        static int no_w8_q5k = -1;
        if (no_w8_q5k < 0) no_w8_q5k = getenv("PICOLM_NO_W8_Q5K") ? 1 : 0;
        if (!no_w8_q5k && !no_mt_q5k && S > 16) {
            unsigned gy = (unsigned)((S + 16 * Q5K_TILES_S - 1) / (16 * Q5K_TILES_S));
            unsigned gx = (unsigned)((O + 8 * Q5KW_WARPS - 1) / (8 * Q5KW_WARPS));
            int smt = 0;   /* register accumulators now; no dynamic shared memory */

            /* Superblock-unrolled inner loop: 8 back-to-back mma into
             * independent accumulators, then 8 epilogues, instead of
             * one mma alternating with scalar work.
             * PICOLM_NO_SB_Q5K=1 selects the plain w8 kernel for A/B. */
            static int no_sb_q5k = -1;
            if (no_sb_q5k < 0) no_sb_q5k = getenv("PICOLM_NO_SB_Q5K") ? 1 : 0;
            if (!no_sb_q5k && (I % 256) == 0) {
                gpu_dispatch_print("matmul_q5_imma_sb_dev");
                picolm_q5_k_q8_matmul_imma_sb<<<dim3(gx, gy), 32 * Q5KW_WARPS, 0, ctx->stream>>>(
                    y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
                gpuError_t sbe = gpuGetLastError();
                if (sbe == gpuSuccess) return 1;
                fprintf(stderr, "ERROR: q5 sb kernel LAUNCH FAILED (%d)\n", (int)sbe);
                gpuGetLastError();
            }
            gpu_dispatch_print("matmul_q5_imma_w8_dev");
            picolm_q5_k_q8_matmul_imma_w8<<<dim3(gx, gy), 32 * Q5KW_WARPS, smt, ctx->stream>>>(
                y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
            if (gpu_ok(gpuGetLastError(), "q5 matmul w8 (dev)")) return 1;
        }
        if (!no_mt_q5k && S > 16) {
            gpu_dispatch_print("matmul_q5_imma_mt_dev");
            /* Cap the tile count so short chunks stay wide: with a fixed 32
             * tiles a 64-token ubatch would launch O/8 blocks of which each
             * uses 4 tiles, wasting 87% of the accumulator work. Using
             * ceil(S/16) when that is smaller keeps the grid shape matched
             * to the work actually present. */
            /* The kernel derives tile_s0 = blockIdx.y * 16 * Q5K_TILES_S from
             * the COMPILE-TIME constant, so the grid must use that same
             * stride. Capping the stride here (while the kernel kept using
             * Q5K_TILES_S) made the blocks overlap and skip rows, which is
             * what produced a different argmax at 423 tokens. Cap the ROW
             * COUNT per block of the grid instead: with tiles_s < Q5K_TILES_S
             * every block still owns Q5K_TILES_S tiles-worth of stride and
             * simply processes fewer of them, which the kernel already
             * handles through n_tiles. */
            dim3 grid((unsigned)((O + 7) / 8),
                      (unsigned)((S + 16 * Q5K_TILES_S - 1) / (16 * Q5K_TILES_S)));
            int smt = Q5K_TILES_S * 32 * 4 * (int)sizeof(float);
            picolm_q5_k_q8_matmul_imma_mt<<<grid, 32, smt, ctx->stream>>>(
                y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
            if (gpu_ok(gpuGetLastError(), "q5 matmul mt (dev)")) return 1;
        }
        gpu_dispatch_print("matmul_q5_imma_dev");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q5_k_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q5 matmul (dev)")) return 0;
        return 1;
    }

    /* Q4_K x Q8_0 IMMA path (device) -- ENABLED */
    if (t->qtype == GGUF_TYPE_Q4_K && ctx->has_imma && !picolm_gpu_no_imma() && S >= 1 && O >= 8 && I % 256 == 0) {
        if (S == 1 && !picolm_gpu_no_gemv() && kquant_gemv_dev(ctx, t, y_dev, x_dev, I, O, ys, x_stride, 4))
            return 1;
        int n_blocks = I / 32;
        if (n_blocks < 1) return 0;
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

        gpu_dispatch_print("matmul_q4_imma_dev");
        dim3 grid((unsigned)((O + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q4_k_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q4 matmul (dev)")) return 0;
        return 1;
    }

    /* FP16/BF16 tiled path for device-resident tensors.
     * Tiled kernel assumes contiguous x (stride == I); fall through to
     * generic path when x_stride is set and differs from I. */
    if ((t->qtype == GGUF_TYPE_F16 || t->qtype == GGUF_TYPE_BF16) &&
        (x_stride <= 0 || x_stride == I) &&
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
    /* Debug removed: was reading y_dev on default stream while kernel ran on ctx->stream */
    return 1;
}

/* Slot name for one matmul_dev call, split by S.
 *
 * Every quant type has its own 30-line copy-pasted activation-quantize block
 * inside matmul_dev_impl (the survey counted ten), so instrumenting each call
 * site would mean touching ten places and would still miss the next one added.
 * Doing it at the wrapper means every path is covered by construction.
 *
 * The S split matters: S==1 takes the GEMV kernels (which have their own,
 * more detailed slots inside kquant_gemv_dev / the Q8_0 GEMV branch), while
 * S>=2 is prefill and a completely different set of tile kernels
 * (matmul_q5_imma_dev, matmul_q6_imma_dev, matmul_imma_smw16_dev...) that had
 * no slot at all. Prefill is the slower of the two paths and was therefore
 * the one with no visibility.
 */
static int matmul_dev_prof_slot(const picolm_gpu_tensor_t *t, int S) {
    if (S <= 1) return -1;   /* decode: already instrumented at the kernels */
    switch (t->qtype) {
    case GGUF_TYPE_Q8_0: return picolm_gpu_prof_slot("pf_matmul_q8_0");
    case GGUF_TYPE_Q6_K: return picolm_gpu_prof_slot("pf_matmul_q6_K");
    case GGUF_TYPE_Q5_K: return picolm_gpu_prof_slot("pf_matmul_q5_K");
    case GGUF_TYPE_Q4_K: return picolm_gpu_prof_slot("pf_matmul_q4_K");
    case GGUF_TYPE_Q4_0: return picolm_gpu_prof_slot("pf_matmul_q4_0");
    case GGUF_TYPE_Q8_K: return picolm_gpu_prof_slot("pf_matmul_q8_K");
    case GGUF_TYPE_F16:  return picolm_gpu_prof_slot("pf_matmul_f16");
    case GGUF_TYPE_BF16: return picolm_gpu_prof_slot("pf_matmul_bf16");
    case GGUF_TYPE_F32:  return picolm_gpu_prof_slot("pf_matmul_f32");
    default:             return picolm_gpu_prof_slot("pf_matmul_other");
    }
}

extern "C" int
picolm_gpu_matmul_dev(picolm_gpu_tensor_t *t, float *y_dev, const float *x_dev,
                       int S, int device, int y_stride, int x_stride) {
    if (g_launch_on < 0) g_launch_on = getenv("PICOLM_LAUNCH_TMR") ? 1 : 0;

    int ps = t ? matmul_dev_prof_slot(t, S) : -1;
    if (ps >= 0) picolm_gpu_prof_begin(device);

    int rc;
    if (!g_launch_on) {
        rc = matmul_dev_impl(t, y_dev, x_dev, S, device, y_stride, x_stride);
    } else {
        double t0 = lt_now();
        rc = matmul_dev_impl(t, y_dev, x_dev, S, device, y_stride, x_stride);
        g_launch_ms += lt_now() - t0;
        g_launch_n++;
    }

    if (ps >= 0) picolm_gpu_prof_end(ps);
    return rc;
}

/* --gpu-q8diff: compare the GPU activation quantiser against the CPU's.
 *
 * Decides whether the next large piece of work is a correctness fix or only a
 * performance change. See the comment on the CLI side for the reasoning.
 * Returns 0 when the int8 payloads are identical.
 */
extern "C" int picolm_gpu_q8_quant_diff(int I);
extern "C" int picolm_gpu_q8_quant_diff(int I) {
    if (I <= 0 || (I % 32) != 0) { fprintf(stderr, "I must be a positive multiple of 32\n"); return 1; }
    int nb = I / 32;

    float *x = (float *)malloc((size_t)I * sizeof(float));
    int8_t *cpu_row = (int8_t *)malloc((size_t)nb * 34);
    if (!x || !cpu_row) { fprintf(stderr, "OOM\n"); return 1; }

    /* Deterministic pseudo-random activations with a realistic dynamic range;
     * a block whose values are all equal cannot distinguish rounding. */
    unsigned st = 1234u;
    for (int i = 0; i < I; i++) {
        st = st * 1664525u + 1013904223u;
        x[i] = ((float)((st >> 8) & 0xFFFF) / 32768.0f - 1.0f) * 8.0f;
    }

    /* CPU reference: same function the CPU matmul path uses. */
    extern void quantize_row_q8_0(const float *x, void *y, int k);
    quantize_row_q8_0(x, cpu_row, I);

    { int dev0 = 0;
      if (!picolm_gpu_init(&dev0, 1)) { fprintf(stderr, "GPU init failed\n"); free(x); free(cpu_row); return 1; } }
    gpu_device_ctx_t *ctx = find_ctx(0);
    if (!ctx || !select_ctx(ctx)) { fprintf(stderr, "no ctx\n"); return 1; }

    int8_t *d_qs = NULL; float *d_d = NULL, *d_x = NULL;
    size_t qs_bytes = (size_t)nb * 32, d_bytes = (size_t)nb * sizeof(float);
    if (!gpu_ok(gpuMalloc((void **)&d_qs, qs_bytes), "q8diff qs") ||
        !gpu_ok(gpuMalloc((void **)&d_d, d_bytes), "q8diff d") ||
        !gpu_ok(gpuMalloc((void **)&d_x, (size_t)I * sizeof(float)), "q8diff x")) {
        fprintf(stderr, "gpuMalloc failed\n"); return 1;
    }
    gpuMemcpy(d_x, x, (size_t)I * sizeof(float), gpuMemcpyHostToDevice);
    /* One block of 32 threads per 32-value block, exactly as the real dispatch
     * does (q_grid = (n_blocks, S)). */
    dim3 grid((unsigned)nb, 1);
    picolm_quantize_q8_0<<<grid, 32, 32 * sizeof(float), ctx->stream>>>(d_qs, d_d, d_x, I, 1);
    if (!gpu_ok(gpuGetLastError(), "q8diff kernel")) return 1;
    gpuDeviceSynchronize();

    int8_t *gpu_qs = (int8_t *)malloc(qs_bytes);
    float *gpu_d = (float *)malloc(d_bytes);
    if (!gpu_qs || !gpu_d) { fprintf(stderr, "OOM\n"); return 1; }
    gpuMemcpy(gpu_qs, d_qs, qs_bytes, gpuMemcpyDeviceToHost);
    gpuMemcpy(gpu_d, d_d, d_bytes, gpuMemcpyDeviceToHost);

    int qs_bad = 0, d_bad = 0, max_qd = 0;
    double max_dd = 0.0, sum_abs_dd = 0.0;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = (const uint8_t *)cpu_row + (size_t)b * 34;
        uint16_t d16 = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        float cd = gpu_fp16_to_fp32((int)d16);
        double dd = (double)cd - (double)gpu_d[b];
        if (dd < 0) dd = -dd;
        sum_abs_dd += dd;
        if (cd != gpu_d[b]) { d_bad++; if (dd > max_dd) max_dd = dd; }
        for (int j = 0; j < 32; j++) {
            int cq = (int8_t)blk[2 + j];
            int gq = (int)gpu_qs[b * 32 + j];
            if (cq != gq) {
                qs_bad++;
                int q = cq - gq; if (q < 0) q = -q;
                if (q > max_qd) max_qd = q;
            }
        }
    }

    fprintf(stderr, "\n[Q8DIFF] I=%d  blocks=%d\n", I, nb);
    fprintf(stderr, "  int8 payload : %d / %d differ   (max |dq| = %d)\n", qs_bad, nb * 32, max_qd);
    fprintf(stderr, "  d scale      : %d / %d blocks differ (max |dd| = %.9g, mean %.3g)\n",
            d_bad, nb, max_dd, nb ? sum_abs_dd / (double)nb : 0.0);
    fprintf(stderr, "  VERDICT: %s\n",
            (qs_bad == 0 && d_bad == 0)
              ? "GPU and CPU Q8_0 agree EXACTLY -- the Q8_0 quantiser is not the long-context divergence"
              : "they DIFFER -- the quantiser is a genuine divergence source");

    free(x); free(cpu_row); free(gpu_qs); free(gpu_d);
    gpuFree(d_qs); gpuFree(d_d); gpuFree(d_x);
    return (qs_bad == 0 && d_bad == 0) ? 0 : 2;
}

/* picolm_gpu_matmul_logits: output projection (lm_head) on GPU.
 * Specialized for S=1 (single token), Q8_0 weights.
 * Takes input from device, writes logits to device.
 * Avoids the D2H+H2D round-trip of picolm_gpu_matmul. */
extern "C"
int picolm_gpu_matmul_logits(picolm_gpu_tensor_t *t, float *logits_dev,
                              const float *x_dev, int device) {
    if (!t || !logits_dev || !x_dev) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    int I = t->I, O = t->O;

    if (t->qtype == GGUF_TYPE_Q8_0) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0;

        /* Quantize F32 input [I] to Q8_0 on GPU */
        size_t xq_bytes = (size_t)I;
        size_t xd_bytes = (size_t)n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);

        dim3 q_grid((unsigned)n_blocks, (unsigned)1);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, x_dev, I, 1);
        if (!gpu_ok(gpuGetLastError(), "logits q8 quantize")) return 0;

        /* S=1: can't use IMMA (needs S>=16) or tiled (needs S>=Q8_TILE_S).
         * Use scalar kernel with O blocks, each computing one logit. */
        gpu_dispatch_print("matmul_logits_dev");
        dim3 grid((unsigned)O, (unsigned)1);
        picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
            logits_dev, ctx->q8_xq, ctx->q8_xd, t->weights,
            1, I, O, (int)t->row_bytes, O);
        if (!gpu_ok(gpuGetLastError(), "logits matmul")) return 0;
        return 1;
    }

    /* IQ6_K lm_head. Needed because the caller's fallback path installs this
     * tensor as a GPU handle and then calls the CPU matmul(), which routes
     * straight back to picolm_quant_matmul -- whose switch has no IQ6_K case
     * and therefore ZEROES the logits, poisoning every later token. Reusing
     * kquant_gemv_dev (Q8_K quantise + IQ6_K GEMV) avoids that entirely. */
    if (kquant_which(t->qtype) && (I % 256) == 0 && O >= 8) {   /* and the other K-quants */
        if (kquant_gemv_dev(ctx, t, logits_dev, x_dev, I, O, O, 0, kquant_which(t->qtype))) return 1;
        return 0;
    }
    return 0; /* unsupported type */
}

/* Strided variant: x_stride > 0 overrides default I stride.
 * Only called for SSM output projection where pipe buffer stride != value_dim. */
extern "C" int
picolm_gpu_matmul_dev_strided(picolm_gpu_tensor_t *t, float *y_dev,
                               const float *x_dev, int S, int device, int x_stride, int y_stride) {
    if (!t || !y_dev || !x_dev || S < 1 || x_stride <= 0) return 0;
    if (t->I < 64) return 0;  /* O gate removed: blocked K/V projections with O<256 */
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

/* ================================================================
 * Phase 1: Fused QKV activation quantization + IMMA matmuls
 * ================================================================
 * Quantize x_dev once to Q8_0, then run Q, K, V IMMA matmuls
 * sharing the same quantized activation buffer. This eliminates 2
 * redundant quantize kernel launches and associated memset/reads
 * per layer (96 fewer kernel launches across 48 layers).
 * ================================================================ */

static int imma_dev_q8(picolm_gpu_tensor_t *t, float *y_dev,
    const int8_t *xq, const float *xd, int S, int I, int O,
    int ys, gpu_device_ctx_t *ctx) {
    return q8mm_launch(q8mm_pick(ctx, S, O, t->row_bytes, 0), ctx,
                       y_dev, xq, xd, t, S, I, O, (int)t->row_bytes, ys);
}

extern "C"
int picolm_gpu_matmul_dev_qkv(picolm_gpu_tensor_t *tq, picolm_gpu_tensor_t *tk, picolm_gpu_tensor_t *tv,
                               float *bq, float *bk, float *bv,
                               const float *x_dev, int S, int device,
                               int y_stride_q, int y_stride_kv, int x_stride) {
    if (!tq || !tk || !tv || !bq || !bk || !bv || !x_dev || S < 1) return 0;
    if (tq->I < 512 || tk->I < 512 || tv->I < 512) return 0;
    if (tq->I != tk->I || tq->I != tv->I) return 0;
    if (tq->qtype != GGUF_TYPE_Q8_0 || tk->qtype != GGUF_TYPE_Q8_0 || tv->qtype != GGUF_TYPE_Q8_0)
        return 0;
    if (picolm_gpu_force_f32()) return 0;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    int I = tq->I;
    int n_blocks = I / 32;
    if (n_blocks < 1 || I % 32 != 0) return 0;

    int S_padded = (S + 15) & ~15;
    size_t xq_bytes = (size_t)S_padded * I;
    size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
    if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
        !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

    /* Quantize F32 input [I*S] to Q8_0 ONCE */
    gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
    gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);
    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
    if (x_stride > 0 && x_stride != I) {
        picolm_quantize_q8_0_strided<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, x_dev, I, S, x_stride);
    } else {
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, x_dev, I, S);
    }
    if (!gpu_ok(gpuGetLastError(), "q8 quantize QKV (dev)")) return 0;

    static int qkv_info = 1;
    int ys_q = y_stride_q > 0 ? y_stride_q : tq->O;
    int ys_kv = y_stride_kv > 0 ? y_stride_kv : tk->O;

    /* Phase 4: try fused QKV IMMA W16 kernel (single launch for all 3, 16x16 tiles).
     * Only on CUDA sm_80+ where IMMA is available. Falls back to 3x separate IMMA. */
#ifdef PICOLM_CUDA
    if (ctx->has_imma && !picolm_gpu_no_imma() && S >= 16 && tq->O >= 8 && tk->O >= 8 && tv->O >= 8 &&
        !picolm_gpu_no_fused_qkv()) {
        int max_o = tq->O;
        if (tk->O > max_o) max_o = tk->O;
        if (tv->O > max_o) max_o = tv->O;
        dim3 grid((unsigned)((max_o + 15) / 16), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma_qkv<<<grid, 32, 0, ctx->stream>>>(
            bq, bk, bv,
            ctx->q8_xq, ctx->q8_xd,
            tq->weights, tk->weights, tv->weights,
            S, I, tq->O, tk->O, tv->O,
            (int)tq->row_bytes, ys_q, ys_kv);
        if (gpu_ok(gpuGetLastError(), "q8 fused qkv imma w16 (dev)")) {
            gpu_dispatch_print("matmul_imma_qkv_w16_dev");
            static int fused_info = 1;
            if (fused_info) { fused_info = 0; fprintf(stderr, "INFO: QKV fused IMMA W16 (1 launch, 1 quantize, S=%d)\n", S); }
            return 1;
        }
    }
#endif

    /* Fallback: 3 separate IMMA matmuls. */
    if (!imma_dev_q8(tq, bq, ctx->q8_xq, ctx->q8_xd, S, I, tq->O, ys_q, ctx)) return 0;
    if (!imma_dev_q8(tk, bk, ctx->q8_xq, ctx->q8_xd, S, I, tk->O, ys_kv, ctx)) return 0;
    if (!imma_dev_q8(tv, bv, ctx->q8_xq, ctx->q8_xd, S, I, tv->O, ys_kv, ctx)) return 0;

    if (qkv_info) { qkv_info = 0; fprintf(stderr, "INFO: QKV matmul fused (1 quantize, 3x IMMA, S=%d)\n", S); }
    return 1;
}

/* ================================================================
 * Phase 7: Fused RMSNorm + Quantize + QKV IMMA
 * ================================================================
 * Replaces: rmsnorm_batched_dev() + quantize() + fused QKV IMMA
 * Reads bx[S][xb_stride] directly, does per-row RMSNorm with the
 * attention norm weight, quantizes to Q8_0, then runs fused QKV
 * IMMA. Eliminates the F32 bxb intermediate buffer entirely.
 * ================================================================ */

extern "C"
int picolm_gpu_rmsnorm_matmul_dev_qkv(picolm_gpu_tensor_t *tq, picolm_gpu_tensor_t *tk, picolm_gpu_tensor_t *tv,
                                       float *bq, float *bk, float *bv,
                                       const float *bx, const float *rmsnorm_w,
                                       int dim, float eps, int S, int xb_stride,
                                       int device, int y_stride_q, int y_stride_kv) {
    if (!tq || !tk || !tv || !bq || !bk || !bv || !bx || S < 1) return 0;
    if (tq->I < 512 || tk->I < 512 || tv->I < 512) return 0;
    if (tq->I != tk->I || tq->I != tv->I) return 0;
    if (tq->qtype != GGUF_TYPE_Q8_0 || tk->qtype != GGUF_TYPE_Q8_0 || tv->qtype != GGUF_TYPE_Q8_0)
        return 0;
    if (picolm_gpu_force_f32()) return 0;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;
    if (!ctx->has_imma) return 0;
    int I = tq->I;
    int n_blocks = I / 32;
    if (n_blocks < 1 || I % 32 != 0) return 0;

    int S_padded = (S + 15) & ~15;
    size_t xq_bytes = (size_t)S_padded * I;
    size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
    if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
        !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

    gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
    gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);

    /* Phase 7: fused RMSNorm + quantize (one kernel, no F32 bxb). */
    {
        int n_threads = min(dim, 256);
        dim3 grid((unsigned)S, 1, 1);
        picolm_rmsnorm_quantize_q8_0_kernel<<<grid, n_threads, 0, ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, bx, rmsnorm_w,
            dim, eps, S, xb_stride);
        if (!gpu_ok(gpuGetLastError(), "rmsnorm+quantize fused (dev)")) return 0;
    }

    int ys_q = y_stride_q > 0 ? y_stride_q : tq->O;
    int ys_kv = y_stride_kv > 0 ? y_stride_kv : tk->O;

#ifdef PICOLM_CUDA
    if (ctx->has_imma && !picolm_gpu_no_imma() && S >= 16 && tq->O >= 8 && tk->O >= 8 && tv->O >= 8 &&
        !picolm_gpu_no_fused_qkv()) {
        int max_o = tq->O;
        if (tk->O > max_o) max_o = tk->O;
        if (tv->O > max_o) max_o = tv->O;
        dim3 grid((unsigned)((max_o + 15) / 16), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma_qkv<<<grid, 32, 0, ctx->stream>>>(
            bq, bk, bv,
            ctx->q8_xq, ctx->q8_xd,
            tq->weights, tk->weights, tv->weights,
            S, I, tq->O, tk->O, tv->O,
            (int)tq->row_bytes, ys_q, ys_kv);
        if (gpu_ok(gpuGetLastError(), "q8 fused qkv imma w16 (rmsnorm)")) {
            static int ph7_info = 1;
            if (ph7_info) { ph7_info = 0; fprintf(stderr, "INFO: Phase 7 active: RMSNorm+quantize+QKV fused (S=%d)\n", S); }
            return 1;
        }
    }
#endif

    /* Fallback: 3 separate IMMA. */
    if (!imma_dev_q8(tq, bq, ctx->q8_xq, ctx->q8_xd, S, I, tq->O, ys_q, ctx)) return 0;
    if (!imma_dev_q8(tk, bk, ctx->q8_xq, ctx->q8_xd, S, I, tk->O, ys_kv, ctx)) return 0;
    if (!imma_dev_q8(tv, bv, ctx->q8_xq, ctx->q8_xd, S, I, tv->O, ys_kv, ctx)) return 0;
    return 1;
}

/* ================================================================
 * Phase 2: Fused gate+up activation quantization + IMMA matmuls
 * ================================================================
 * Quantize x_dev once to Q8_0, then run gate+up IMMA matmuls
 * sharing the same quantized activation buffer.
 * ================================================================ */

extern "C"
int picolm_gpu_matmul_dev_gu(picolm_gpu_tensor_t *tg, picolm_gpu_tensor_t *tu,
                              float *bgate, float *bup,
                              const float *x_dev, int S, int device,
                              int y_stride, int x_stride) {
    if (!tg || !tu || !bgate || !bup || !x_dev || S < 1) return 0;
    if (tg->I < 512 || tu->I < 512) return 0;
    if (tg->I != tu->I) return 0;
    if (tg->qtype != GGUF_TYPE_Q8_0 || tu->qtype != GGUF_TYPE_Q8_0) return 0;
    if (picolm_gpu_force_f32()) return 0;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    int I = tg->I;
    int n_blocks = I / 32;
    if (n_blocks < 1 || I % 32 != 0) return 0;

    int S_padded = (S + 15) & ~15;
    size_t xq_bytes = (size_t)S_padded * I;
    size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
    if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
        !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

    /* Quantize F32 input [I*S] to Q8_0 ONCE */
    gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
    gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);
    dim3 q_grid((unsigned)n_blocks, (unsigned)S);
    if (x_stride > 0 && x_stride != I) {
        picolm_quantize_q8_0_strided<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, x_dev, I, S, x_stride);
    } else {
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, x_dev, I, S);
    }
    if (!gpu_ok(gpuGetLastError(), "q8 quantize GU (dev)")) return 0;

    static int gu_info = 1;
    /* bgate = tg @ xq */
    int ys = y_stride > 0 ? y_stride : tg->O;
    if (!imma_dev_q8(tg, bgate, ctx->q8_xq, ctx->q8_xd, S, I, tg->O, ys, ctx)) return 0;
    /* bup = tu @ xq (same quantized activation) */
    if (!imma_dev_q8(tu, bup, ctx->q8_xq, ctx->q8_xd, S, I, tu->O, ys, ctx)) return 0;

    if (gu_info) { gu_info = 0; fprintf(stderr, "INFO: Gate+Up matmul fused (1 quantize, 2x IMMA, S=%d)\n", S); }
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
    gpuDeviceSynchronize(); /* WDDM: H2D may return early */

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
#if defined(PICOLM_IMMA_W16)
    if (ctx->has_imma && S >= 16 && I >= 16) {
        gpu_dispatch_print("expert_mlp_gate_imma_w16_host");
        dim3 grid((unsigned)((I + 15) / 16), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma_w16<<<grid, 32, 0, ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, gate->weights, S, D, I, (int)gate->row_bytes, I);
    } else
#endif
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
#if defined(PICOLM_IMMA_W16)
    if (ctx->has_imma && S >= 16 && I >= 16) {
        gpu_dispatch_print("expert_mlp_up_imma_w16_host");
        dim3 grid((unsigned)((I + 15) / 16), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma_w16<<<grid, 32, 0, ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, up->weights, S, D, I, (int)up->row_bytes, I);
    } else
#endif
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
#if defined(PICOLM_IMMA_W16)
    if (ctx->has_imma && S >= 16 && D >= 16) {
        gpu_dispatch_print("expert_mlp_down_imma_w16_host");
        dim3 grid((unsigned)((D + 15) / 16), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma_w16<<<grid, 32, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, down->weights, S, I, D, (int)down->row_bytes, D);
    } else
#endif
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
    gpuDeviceSynchronize(); /* WDDM: H2D may return early */

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
    gpuDeviceSynchronize(); /* WDDM: H2D may return early */

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
