#ifndef BACKEND_GPU_KERNELS_CUH
#define BACKEND_GPU_KERNELS_CUH

#include "backend_gpu_common.cuh"
#include "backend_gpu_dequant.cuh"

__global__ void
picolm_quant_matmul(float *y, const float *x, const void *weights,
                    gguf_type_t qtype, int S, int I, int O, int row_bytes, int x_stride, int y_stride) {
    /* bytes_per_block: stride between consecutive blocks in memory */
    int bytes_per_block;
    switch (qtype) {
        case GGUF_TYPE_F32:  bytes_per_block = 4; break;      /* 1 float */
        case GGUF_TYPE_F16:  bytes_per_block = 2; break;      /* 1 uint16_t */
        case 30:             bytes_per_block = 2; break;      /* BF16 */
        case GGUF_TYPE_Q4_0: bytes_per_block = GPU_BLOCK_Q4_0_SIZE; break;  /* 18 */
        case GGUF_TYPE_Q8_0: bytes_per_block = GPU_BLOCK_Q8_0_SIZE; break;  /* 34 */
        case 10:             bytes_per_block = GPU_BLOCK_Q2_K_SIZE; break;  /* Q2_K: 84 */
        case 11:             bytes_per_block = 110; break;                   /* Q3_K: 110 */
        case GGUF_TYPE_Q4_K: bytes_per_block = GPU_BLOCK_Q4_K_SIZE; break;  /* 144 */
        case 13:             bytes_per_block = GPU_BLOCK_Q5_K_SIZE; break;  /* Q5_K: 176 */
        case 14:             bytes_per_block = GPU_BLOCK_Q6_K_SIZE; break;  /* Q6_K: 210 */
        case 41:             bytes_per_block = 18; break;      /* Q1_0 */
        case 42:             bytes_per_block = 34; break;      /* Q2_0 */
        default: bytes_per_block = 18; break;
    }
    int o = gpuBlockIdx_x;
    int s = gpuBlockIdx_y;
    if (o >= O || s >= S) return;
    size_t rs = x_stride > 0 ? (size_t)x_stride : (size_t)I;
    int ys = y_stride > 0 ? y_stride : O;

    double sum = 0.0;
    const char *wrow = (const char *)weights + (size_t)o * row_bytes;

    switch (qtype) {
    case 0: /* GGUF_TYPE_F32 */
        for (int i = gpuThreadIdx_x; i < I; i += gpuBlockDim_x) {
            sum += x[rs*s+i] * ((const float *)(wrow))[i];
        }
        break;

    case 1: /* GGUF_TYPE_F16 */
        /* Raw FP16 array, no block structure. Dequant on-the-fly. */
        for (int i = gpuThreadIdx_x; i < I; i += gpuBlockDim_x) {
            sum += x[rs*s+i] * dequant_f16(wrow, i);
        }
        break;

    case 30: /* GGUF_TYPE_BF16 */
        /* Raw BF16 array, no block structure. Dequant on-the-fly. */
        for (int i = gpuThreadIdx_x; i < I; i += gpuBlockDim_x) {
            sum += x[rs*s+i] * dequant_bf16(wrow, i);
        }
        break;

    case 2: /* GGUF_TYPE_Q4_0 */
        /* I values in 32-value blocks (18 bytes each) */
        {
            int n_blocks = I / 32;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 32; j++) {
                    int i = bi * 32 + j;
                    sum += x[rs*s+i] * dequant_q4_0(blk, j);
                }
            }
        }
        break;

    case 8: /* GGUF_TYPE_Q8_0 */
        /* I values in 32-value blocks (34 bytes each) */
        {
            int n_blocks = I / 32;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 32; j++) {
                    int i = bi * 32 + j;
                    sum += x[rs*s+i] * dequant_q8_0(blk, j);
                }
            }
        }
        break;

    case 10: /* GGUF_TYPE_Q2_K */
        /* 256 values per block (84 bytes). Per-element dequant via helper. */
        {
            int n_blocks = I / 256;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 256; j++) {
                    int i = bi * 256 + j;
                    sum += x[rs*s+i] * dequant_q2_K(blk, j);
                }
            }
        }
        break;

    case 12: /* GGUF_TYPE_Q4_K */
        /* 256 values per block (144 bytes). Per-element dequant via helper. */
        {
            int n_blocks = I / 256;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 256; j++) {
                    int i = bi * 256 + j;
                    sum += x[rs*s+i] * dequant_q4_K(blk, j);
                }
            }
        }
        break;

    case 13: /* GGUF_TYPE_Q5_K */
        /* 256 values per block (176 bytes). Per-element dequant via helper. */
        {
            int n_blocks = I / 256;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 256; j++) {
                    int i = bi * 256 + j;
                    sum += x[rs*s+i] * dequant_q5_K(blk, j);
                }
            }
        }
        break;

    case 14: /* GGUF_TYPE_Q6_K */
        /* 256 values per block (210 bytes). Per-element dequant via helper. */
        {
            int n_blocks = I / 256;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 256; j++) {
                    int i = bi * 256 + j;
                    sum += x[rs*s+i] * dequant_q6_K(blk, j);
                }
            }
        }
        break;

    case 41: /* GGUF_TYPE_Q1_0 */
        /* 128 values per block, 18 bytes */
        {
            int n_blocks = I / 128;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 128; j++) {
                    int i = bi * 128 + j;
                    sum += x[rs*s+i] * dequant_q1_0(blk, j);
                }
            }
        }
        break;

    case 42: /* GGUF_TYPE_Q2_0 */
        /* 128 values per block, 34 bytes */
        {
            int n_blocks = I / 128;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 128; j++) {
                    int i = bi * 128 + j;
                    sum += x[rs*s+i] * dequant_q2_0(blk, j);
                }
            }
        }
        break;

    default:
        break;
    }

    /* Shared-memory tree reduce with double precision */
    __shared__ double partial[256];
    partial[gpuThreadIdx_x] = sum;
    gpuSyncthreads();
    for (int n = gpuBlockDim_x >> 1; n; n >>= 1) {
        if (gpuThreadIdx_x < n)
            partial[gpuThreadIdx_x] += partial[gpuThreadIdx_x + n];
        gpuSyncthreads();
    }
    if (!gpuThreadIdx_x)
        y[(size_t)s * ys + o] = (float)partial[0];
}

/* ---- GPU-side Q8_0 quantization kernel ----
 *
 * Mirrors CPU quantize_row_q8_0 in quant.c exactly.
 * Per 32-element block: d = max(|x|) / 127, qs[i] = round(x[i] / d).
 * Output: int8_t qs[32] + uint16_t d (FP16) per block = 34 bytes.
 *
 * Grid: [S, n_blocks_per_row], Block: 32 threads.
 * Each block quantizes one 32-element chunk of one input row. */
__global__ void
picolm_quantize_q8_0(int8_t *qs_out, float *d_out,
                      const float *x, int I, int S) {
    int s = gpuBlockIdx_y;           /* row index */
    int block = gpuBlockIdx_x;       /* 32-value block within row */
    int tid = gpuThreadIdx_x;        /* 0..31 */
    int n_blocks = (I + 31) / 32;
    if (block >= n_blocks || s >= S) return;

    const float *xb = x + (size_t)s * I + (size_t)block * 32;

    __shared__ float amax_shared[32];
    float v = (tid < I - block * 32) ? xb[tid] : 0.0f;
    amax_shared[tid] = fabsf(v);
    gpuSyncthreads();

    /* Reduce max across 32 threads */
    for (int stride = 16; stride; stride >>= 1) {
        if (tid < stride)
            amax_shared[tid] = fmaxf(amax_shared[tid], amax_shared[tid + stride]);
        gpuSyncthreads();
    }

    float amax = amax_shared[0];
    float d = amax / 127.0f;
    float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    int q = (int)lrintf(v * id);
    /* Clamp to [-128, 127] */
    if (q > 127) q = 127;
    if (q < -128) q = -128;

    qs_out[(size_t)s * (size_t)n_blocks * 32 + (size_t)block * 32 + tid] = (int8_t)q;
    if (tid == 0)
        d_out[(size_t)s * (size_t)n_blocks + block] = d;
}

/* Strided variant: x_stride > 0 overrides the default s*I stride.
 * Output is always contiguous [S][I]. Only called when x_stride != I. */
__global__ void
picolm_quantize_q8_0_strided(int8_t *qs_out, float *d_out,
                              const float *x, int I, int S, int x_stride) {
    int s = gpuBlockIdx_y;
    int block = gpuBlockIdx_x;
    int tid = gpuThreadIdx_x;
    int n_blocks = (I + 31) / 32;
    if (block >= n_blocks || s >= S) return;
    const float *xb = x + (size_t)s * x_stride + (size_t)block * 32;
    __shared__ float amax_shared[32];
    float v = (tid < I - block * 32) ? xb[tid] : 0.0f;
    amax_shared[tid] = fabsf(v);
    gpuSyncthreads();
    for (int stride = 16; stride; stride >>= 1) {
        if (tid < stride)
            amax_shared[tid] = fmaxf(amax_shared[tid], amax_shared[tid + stride]);
        gpuSyncthreads();
    }
    float amax = amax_shared[0];
    float d = amax / 127.0f;
    float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    int q = (int)lrintf(v * id);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    qs_out[(size_t)s * (size_t)n_blocks * 32 + (size_t)block * 32 + tid] = (int8_t)q;
    if (tid == 0)
        d_out[(size_t)s * (size_t)n_blocks + block] = d;
}

/* ---- Q8_0 x Q8_0 integer-MAC matmul kernel ----
 *
 * Matches CPU vec_dot_q8_0_q8_0_deltas exactly:
 *   For each 32-element block: dot(int8_x, int8_w) * d_x * d_w
 * where d_x and d_w are pre-converted to FP32.
 *
 * Grid: [O, S], Block: 256 threads.
 * Each thread block computes one output element y[s*O + o].
 *
 * xq: quantized input, shape [S, n_blocks, 32] of int8_t
 * xd: x deltas as FP32, shape [S, n_blocks] (pre-converted from FP16)
 * weights: Q8_0 weight blocks, shape [O, n_blocks] * 34 bytes each */
__global__ void
picolm_q8_q8_matmul(float *y,
                     const int8_t *xq, const float *xd,
                     const void *weights,
                     int S, int I, int O, int row_bytes, int y_stride) {
    int o = gpuBlockIdx_x;
    int s = gpuBlockIdx_y;
    if (o >= O || s >= S) return;
    int ys = y_stride > 0 ? y_stride : O;

    int n_blocks = I / 32;
    const char *wrow = (const char *)weights + (size_t)o * row_bytes;
    const int8_t *xrow = xq + (size_t)s * I;
    const float *xdrow = xd + (size_t)s * n_blocks;

    double sum = 0.0;
    for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
        const uint8_t *b = (const uint8_t *)wrow + (size_t)bi * GPU_BLOCK_Q8_0_SIZE;
        uint16_t wd_raw = b[0] | ((uint16_t)b[1] << 8);
        float wd = gpu_fp16_to_fp32(wd_raw);
        float xdv = xdrow[bi];

        const int8_t *wq = (const int8_t *)(b + 2);
        const int8_t *xqp = xrow + bi * 32;

        int32_t acc = 0;
#if 0 /* __dp4a: 4-way int8 MAC (sm_61+). Verified bit-exact but on GB10 (sm_121)
       * ptxas lowers __dp4a to scalar IMAD -- no hardware DP4A on Blackwell.
       * Kept in #if 0 for architectures that have it (e.g. Turing/Ampere
       * where __dp4a maps to real DP4A instructions). Guarded to avoid
       * the memcpy overhead on platforms where it doesn't help. */
#ifndef __HIP__
        {
            int32_t wq4[8], xq4[8];
            memcpy(wq4, wq, 32);
            memcpy(xq4, xqp, 32);
#pragma unroll
            for (int j = 0; j < 8; j++)
                acc = __dp4a(wq4[j], xq4[j], acc);
        }
#else
        for (int j = 0; j < 32; j++)
            acc += (int32_t)wq[j] * (int32_t)xqp[j];
#endif
#else
        for (int j = 0; j < 32; j++)
            acc += (int32_t)wq[j] * (int32_t)xqp[j];
#endif

        sum += (double)acc * (double)wd * (double)xdv;
    }

    /* Shared-memory tree reduce with double precision */
    __shared__ double partial[256];
    partial[gpuThreadIdx_x] = sum;
    gpuSyncthreads();
    for (int n = gpuBlockDim_x >> 1; n; n >>= 1) {
        if (gpuThreadIdx_x < n)
            partial[gpuThreadIdx_x] += partial[gpuThreadIdx_x + n];
        gpuSyncthreads();
    }
    if (!gpuThreadIdx_x)
        y[(size_t)s * ys + o] = (float)partial[0];
}

/* Number of sequence positions per tile in the tiled Q8_0 matmul.
 * Shared memory usage (row_bytes, up to ~15KB for a 14336-wide FFN row)
 * doesn't depend on this constant -- positions are handled in an inner
 * loop, not concurrently. Stays well under the default 48KB/block. */
#define Q8_TILE_S 32

/* Tiled Q8_0 matmul: loads each weight row into shared memory once per
 * tile of Q8_TILE_S query positions, then reuses it in the inner loop.
 * Original reads the same weight row from global memory independently
 * for every (output_row, s) pair -- for S=291 prefill, that's 291
 * redundant global reads of the same row.
 * Same arithmetic, same accumulation order per (o,s) as original --
 * bit-exact. For S=1 (decode) reduces to one tile of size 1. */
__global__ void
picolm_q8_q8_matmul_tiled(float *y,
                           const int8_t *xq, const float *xd,
                           const void *weights,
                           int S, int I, int O, int row_bytes, int y_stride) {
    int o = gpuBlockIdx_x;
    int tile = gpuBlockIdx_y;
    if (o >= O) return;
    int s0 = tile * Q8_TILE_S;
    if (s0 >= S) return;
    int s_count = min(Q8_TILE_S, S - s0);
    int ys = y_stride > 0 ? y_stride : O;

    int n_blocks = I / 32;
    const uint8_t *wrow = (const uint8_t *)weights + (size_t)o * row_bytes;

    /* Load weight row into shared memory (dynamic portion) */
    extern __shared__ uint8_t wrow_sh[];
    for (int b = gpuThreadIdx_x; b < row_bytes; b += gpuBlockDim_x) {
        wrow_sh[b] = wrow[b];
    }
    gpuSyncthreads();

    __shared__ double partial[256];

    for (int ls = 0; ls < s_count; ls++) {
        int s = s0 + ls;
        const int8_t *xrow = xq + (size_t)s * I;
        const float *xdrow = xd + (size_t)s * n_blocks;

        double sum = 0.0;
        for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
            const uint8_t *b = wrow_sh + (size_t)bi * GPU_BLOCK_Q8_0_SIZE;
            uint16_t wd_raw = b[0] | ((uint16_t)b[1] << 8);
            float wd = gpu_fp16_to_fp32(wd_raw);
            float xdv = xdrow[bi];

            const int8_t *wq = (const int8_t *)(b + 2);
            const int8_t *xqp = xrow + bi * 32;

            int32_t acc = 0;
#if 0 /* __dp4a: see picolm_q8_q8_matmul comment above */
#ifndef __HIP__
            {
                int32_t wq4[8], xq4[8];
                memcpy(wq4, wq, 32);
                memcpy(xq4, xqp, 32);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    acc = __dp4a(wq4[j], xq4[j], acc);
            }
#else
            for (int j = 0; j < 32; j++)
                acc += (int32_t)wq[j] * (int32_t)xqp[j];
#endif
#else
            for (int j = 0; j < 32; j++)
                acc += (int32_t)wq[j] * (int32_t)xqp[j];
#endif

            sum += (double)acc * (double)wd * (double)xdv;
        }

        /* Shared-memory tree reduce with double precision */
        partial[gpuThreadIdx_x] = sum;
        gpuSyncthreads();
        for (int n = gpuBlockDim_x >> 1; n; n >>= 1) {
            if (gpuThreadIdx_x < n)
                partial[gpuThreadIdx_x] += partial[gpuThreadIdx_x + n];
            gpuSyncthreads();
        }
        if (!gpuThreadIdx_x)
            y[(size_t)s * ys + o] = (float)partial[0];
        gpuSyncthreads();
    }
}

/* ---- silu_mul kernel ----
 * Element-wise: gate[i] = gate[i] / (1 + exp(-gate[i])) * up[i] */
__global__ void
picolm_silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

/* ================================================================
 * Phase 1: GPU-resident KV cache + attention kernels
 * ================================================================ */

/* Phase 1.5: no store kernel needed. The CPU-side KV cache row layout
 * ([kv_head][head_dim] contiguous FP16) is byte-identical to the device
 * layout, so storing is a single gpuMemcpyAsync -- see
 * picolm_gpu_kv_store_rows() in the host API section below. This replaces
 * the old picolm_gpu_kv_store_kernel / picolm_gpu_kv_store_batch_kernel,
 * which cost one launch per (position, kv_head) and dominated prefill
 * time (O(n_tokens * n_kv_heads) launches for a handful of FP16 elements
 * each).
 *
 * ---- Decode attention kernel ----
 * One thread block per KV head. Within each block, kv_mul warps (or thread
 * groups) process the kv_mul query heads that share this KV head.
 *
 * Shared memory: holds one KV position's K and V (head_dim FP16 each).
 * Each query-head group maintains its own online-softmax state in registers.
 *
 /* ---- Decode attention kernel (one thread block per KV head,
 * processes all kv_mul Q heads, loop over positions, shared mem K/V) ----
 *
 * Architecture: each block handles one KV head and its kv_mul grouped Q heads.
 * Shared memory holds one KV position's K and V vectors.
 * A tree reduce accumulates partial dot products across threads.
 * Thread 0 performs the online-softmax update and V accumulation.
 *
 * Shared memory: [K:head_dim u16][V:head_dim u16][reduce:256 float]
 * The reduce area is oversized (256 floats = 1KB) to handle the tree reduce.
 * For head_dim=128, kv_mul=8: 256+256+1024 = 1536 bytes. */
/* Rewritten: the previous version had every thread declare and
 * zero-init a private acc[8][256] float array (8KB/thread) -- far
 * beyond the register file, so it spills to local memory, and it's
 * done redundantly by all n_threads threads even though only thread 0
 * ever used its own copy for the actual accumulation (which then ran
 * serially on thread 0 alone, 255 threads idling at the barrier every
 * position). Measured at 894.6us avg vs a ~3us KV-bytes-read bandwidth
 * floor at pos=200 -- ~300x off, consistent with this being the actual
 * cost driver rather than genuine attention compute.
 *
 * Fix: one shared-memory accumulator [kv_mul][head_dim], written once
 * (cooperative zero-init), and the per-position update parallelized
 * across all threads (grid-stride over head_dim) instead of a thread-0
 * serial loop. The online-softmax branch decision is still a scalar
 * computed by thread 0 (cheap), broadcast via two shared scalars
 * (rescale, weight) so the unified update
 *   acc[d] = acc[d]*rescale + weight*v[d]
 * covers both softmax-update branches without duplicating the loop. */
__global__ void
picolm_gpu_attention_decode_kernel(
        float *xb_out,
        const float *q_dev,
        const uint16_t *kv_k,
        const uint16_t *kv_v,
        int layer_ordinal,
        int pos,
        int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
        size_t kv_pos_stride_bytes,
        size_t kv_head_stride_bytes)
{
    int kv_h = (int)gpuBlockIdx_x;
    if (kv_h >= n_kv_heads) return;

    int kv_mul = n_heads / n_kv_heads;
    int first_qh = kv_h * kv_mul;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    size_t layer_base = (size_t)layer_ordinal * max_seq_len * n_kv_heads * head_dim;

    /* Shared memory: K + V (u16), reduce area (256 float), then the
     * accumulator [kv_mul][head_dim] float, then max_score[kv_mul] and
     * sum_exp[kv_mul] float. Sized by the host wrapper. */
    extern __shared__ uint8_t smem[];
    uint16_t *k_sh = (uint16_t *)smem;
    uint16_t *v_sh = k_sh + head_dim;
    float *reduce_sh = (float *)((uint8_t *)v_sh + head_dim * sizeof(uint16_t));
    float *acc_sh = reduce_sh + 256;
    float *max_score_sh = acc_sh + (size_t)kv_mul * head_dim;
    float *sum_exp_sh = max_score_sh + kv_mul;

    __shared__ float rescale_sh, weight_sh;

    for (int i = tid; i < kv_mul * head_dim; i += n_threads) acc_sh[i] = 0.0f;
    for (int g = tid; g < kv_mul; g += n_threads) {
        max_score_sh[g] = -1e30f;
        sum_exp_sh[g] = 0.0f;
    }
    gpuSyncthreads();

    float attn_scale = 1.0f / sqrtf((float)head_dim);

    for (int t = 0; t <= pos; t++) {
        size_t k_off = layer_base + (size_t)t * kv_pos_stride_bytes / 2 + kv_h * kv_head_stride_bytes / 2;
        size_t v_off = k_off;
        for (int d = tid; d < head_dim; d += n_threads) {
            k_sh[d] = kv_k[k_off + d];
            v_sh[d] = kv_v[v_off + d];
        }
        gpuSyncthreads();

        for (int g = 0; g < kv_mul; g++) {
            const float *qg = q_dev + (size_t)(first_qh + g) * head_dim;
            float score;
            if (tid == 0) {
                /* Match CPU AVX-512 accumulation: n_chunks of 16, tree reduce */
                int n_chunks = head_dim / 16;
                float chunk[16] = {0};
                for (int c = 0; c < n_chunks; c++) {
                    float s = 0;
                    for (int d = c * 16; d < (c + 1) * 16; d++) {
                        s = fmaf(qg[d], gpu_fp16_to_fp32(k_sh[d]), s);
                    }
                    chunk[c] = s;
                }
                for (int stride = n_chunks / 2; stride > 0; stride >>= 1) {
                    for (int c = 0; c < stride; c++) chunk[c] += chunk[c + stride];
                }
                score = chunk[0] * attn_scale;
            }

            if (tid == 0) {
                if (score > max_score_sh[g]) {
                    rescale_sh = expf(max_score_sh[g] - score);
                    weight_sh = 1.0f;
                    sum_exp_sh[g] = sum_exp_sh[g] * rescale_sh + 1.0f;
                    max_score_sh[g] = score;
                } else {
                    rescale_sh = 1.0f;
                    weight_sh = expf(score - max_score_sh[g]);
                    sum_exp_sh[g] += weight_sh;
                }
            }
            gpuSyncthreads();

            float *accg = acc_sh + (size_t)g * head_dim;
            for (int d = tid; d < head_dim; d += n_threads) {
                float v_f = gpu_fp16_to_fp32(v_sh[d]);
                if (weight_sh == 1.0f) {
                    /* score > max: CPU fmadd(acc, correction, v) */
                    accg[d] = fmaf(accg[d], rescale_sh, v_f);
                } else {
                    /* score <= max: CPU AVX-512 fmadd(v, weight, acc) */
                    accg[d] = fmaf(weight_sh, v_f, accg[d]);
                }
            }
            gpuSyncthreads(); /* reduce_sh/rescale_sh/weight_sh reused next g/t iter */
        }
    }

    /* Normalize and write output, parallelized across threads */
    for (int g = 0; g < kv_mul; g++) {
        float inv_sum = (sum_exp_sh[g] > 0.0f) ? (1.0f / sum_exp_sh[g]) : 0.0f;
        float *xbhg = xb_out + (size_t)(first_qh + g) * head_dim;
        float *accg = acc_sh + (size_t)g * head_dim;
        for (int d = tid; d < head_dim; d += n_threads) {
            xbhg[d] = accg[d] * inv_sum;
        }
    }
}

/* ---- Split-K decode attention ----
 * The shared-memory rewrite above fixed the accumulator bug, but grid =
 * n_kv_heads (likely ~8 blocks total) still massively underutilizes the
 * GPU's SM count, especially as context grows and each block serially
 * walks more positions. This splits the KV range across multiple blocks
 * per KV head (grid.y = n_splits), each producing a partial online-
 * softmax state, merged by a second small kernel -- standard
 * "flash-decoding" pattern for GQA decode with few KV heads.
 *
 * NOTE: unlike every other attention kernel change this session, this
 * is NOT expected to be bit-exact with the single-pass kernel. Merging
 * partials involves re-scaling by exp(local_max - global_max) in a
 * different summation order than the single serial pass -- floating
 * point addition isn't associative, so tiny (sub-ULP-accumulation-scale)
 * differences are expected and fine. Validate against the existing
 * PICOLM_DBG_ATTN/PICOLM_DBG_PIPELINE tolerance (1e-2/1e-3), not for
 * exact equality -- a nonzero-but-below-threshold diff here is correct
 * behavior, not a regression. */
#define ATTN_DECODE_MAX_SPLITS 32
#define ATTN_DECODE_MIN_CHUNK  64

/* Partial state layout (flat float buffer, sized by the host wrapper):
 *   partial_max: [n_heads][n_splits]
 *   partial_sum: [n_heads][n_splits]
 *   partial_acc: [n_heads][n_splits][head_dim]
 * index(kv_h, split, g) = (kv_h * n_splits + split) * kv_mul + g */
__global__ void
picolm_gpu_attention_decode_split_kernel(
        float *partial_max, float *partial_sum, float *partial_acc,
        const float *q_dev,
        const uint16_t *kv_k, const uint16_t *kv_v,
        int layer_ordinal, int pos,
        int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
        size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes,
        int n_splits, int chunk_size)
{
    int kv_h = (int)gpuBlockIdx_x;
    int split = (int)gpuBlockIdx_y;
    if (kv_h >= n_kv_heads || split >= n_splits) return;

    int kv_mul = n_heads / n_kv_heads;
    int first_qh = kv_h * kv_mul;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;
    int idx_base = (kv_h * n_splits + split) * kv_mul;

    int t0 = split * chunk_size;
    int t1 = min(t0 + chunk_size, pos + 1);

    extern __shared__ uint8_t smem[];
    uint16_t *k_sh = (uint16_t *)smem;
    uint16_t *v_sh = k_sh + head_dim;
    float *reduce_sh = (float *)((uint8_t *)v_sh + head_dim * sizeof(uint16_t));
    float *acc_sh = reduce_sh + 256;
    float *max_score_sh = acc_sh + (size_t)kv_mul * head_dim;
    float *sum_exp_sh = max_score_sh + kv_mul;
    __shared__ float rescale_sh, weight_sh;

    for (int i = tid; i < kv_mul * head_dim; i += n_threads) acc_sh[i] = 0.0f;
    for (int g = tid; g < kv_mul; g += n_threads) {
        max_score_sh[g] = -1e30f;
        sum_exp_sh[g] = 0.0f;
    }
    gpuSyncthreads();

    if (t0 < t1) {
        size_t layer_base = (size_t)layer_ordinal * max_seq_len * n_kv_heads * head_dim;
        float attn_scale = 1.0f / sqrtf((float)head_dim);

        for (int t = t0; t < t1; t++) {
            size_t k_off = layer_base + (size_t)t * kv_pos_stride_bytes / 2 + kv_h * kv_head_stride_bytes / 2;
            for (int d = tid; d < head_dim; d += n_threads) {
                k_sh[d] = kv_k[k_off + d];
                v_sh[d] = kv_v[k_off + d];
            }
            gpuSyncthreads();

            for (int g = 0; g < kv_mul; g++) {
                const float *qg = q_dev + (size_t)(first_qh + g) * head_dim;
                float score;
                if (tid == 0) {
                    int n_chunks = head_dim / 16;
                    float chunk[16] = {0};
                    for (int c = 0; c < n_chunks; c++) {
                        float s = 0;
                        for (int d = c * 16; d < (c + 1) * 16; d++) {
                            s = fmaf(qg[d], gpu_fp16_to_fp32(k_sh[d]), s);
                        }
                        chunk[c] = s;
                    }
                    for (int stride = n_chunks / 2; stride > 0; stride >>= 1) {
                        for (int c = 0; c < stride; c++) chunk[c] += chunk[c + stride];
                    }
                    score = chunk[0] * attn_scale;
                }

                if (tid == 0) {
                    if (score > max_score_sh[g]) {
                        rescale_sh = expf(max_score_sh[g] - score);
                        weight_sh = 1.0f;
                        sum_exp_sh[g] = sum_exp_sh[g] * rescale_sh + 1.0f;
                        max_score_sh[g] = score;
                    } else {
                        rescale_sh = 1.0f;
                        weight_sh = expf(score - max_score_sh[g]);
                        sum_exp_sh[g] += weight_sh;
                    }
                }
                gpuSyncthreads();

                float *accg = acc_sh + (size_t)g * head_dim;
                for (int d = tid; d < head_dim; d += n_threads) {
                    float v_f2 = gpu_fp16_to_fp32(v_sh[d]);
                    if (weight_sh == 1.0f) {
                        /* score > max: CPU fmadd(acc, correction, v) */
                        accg[d] = fmaf(accg[d], rescale_sh, v_f2);
                    } else {
                        /* score <= max: CPU AVX-512 fmadd(v, weight, acc) */
                        accg[d] = fmaf(weight_sh, v_f2, accg[d]);
                    }
                }
                gpuSyncthreads();
            }
        }
    }

    for (int g = tid; g < kv_mul; g += n_threads) {
        partial_max[idx_base + g] = max_score_sh[g];
        partial_sum[idx_base + g] = sum_exp_sh[g];
    }
    for (int i = tid; i < kv_mul * head_dim; i += n_threads)
        partial_acc[(size_t)idx_base * head_dim + i] = acc_sh[i];
}

/* Merges n_splits partial states per KV head via the standard online-
 * softmax merge rule, writes final normalized output. Grid = n_kv_heads. */
__global__ void
picolm_gpu_attention_decode_merge_kernel(
        float *xb_out,
        const float *partial_max, const float *partial_sum, const float *partial_acc,
        int n_heads, int n_kv_heads, int head_dim, int n_splits)
{
    int kv_h = (int)gpuBlockIdx_x;
    int kv_mul = n_heads / n_kv_heads;
    int first_qh = kv_h * kv_mul;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    extern __shared__ float smem_f[];
    float *global_max_sh = smem_f;         /* [kv_mul] */
    float *global_sum_sh = global_max_sh + kv_mul; /* [kv_mul] */

    for (int g = tid; g < kv_mul; g += n_threads) {
        float m = -1e30f;
        for (int sp = 0; sp < n_splits; sp++) {
            int idx = (kv_h * n_splits + sp) * kv_mul + g;
            float v = partial_max[idx];
            if (v > m) m = v;
        }
        global_max_sh[g] = m;
    }
    gpuSyncthreads();

    for (int g = tid; g < kv_mul; g += n_threads) {
        float m = global_max_sh[g];
        float s = 0.0f;
        for (int sp = 0; sp < n_splits; sp++) {
            int idx = (kv_h * n_splits + sp) * kv_mul + g;
            s += partial_sum[idx] * expf(partial_max[idx] - m);
        }
        global_sum_sh[g] = s;
    }
    gpuSyncthreads();

    for (int g = 0; g < kv_mul; g++) {
        float m = global_max_sh[g];
        float inv_sum = 1.0f / global_sum_sh[g];
        float *xbhg = xb_out + (size_t)(first_qh + g) * head_dim;
        for (int d = tid; d < head_dim; d += n_threads) {
            float acc_d = 0.0f;
            for (int sp = 0; sp < n_splits; sp++) {
                int idx = (kv_h * n_splits + sp) * kv_mul + g;
                float w = expf(partial_max[idx] - m);
                acc_d += w * partial_acc[(size_t)idx * head_dim + d];
            }
            xbhg[d] = acc_d * inv_sum;
        }
    }
}

/* ---- Prefill tiled attention kernel ----
 * Processes one (query_head, token_tile) pair.
 * Tiled over KV positions with online-softmax merge.
 *
 * Grid: [n_heads, ceil(n_tokens / TOKEN_TILE)]
 * Each block handles one head and one tile of query tokens.
 * Inner loop tiles over KV positions. */
#define ATTN_TILE_K 32
#define ATTN_TILE_Q 32

/* FP32 K/V variant: reads K/V as FP32 instead of FP16 from KV cache.
 * Identical algorithm to picolm_gpu_attention_prefill_kernel but takes
 * FP32 K/V buffers with layout [pos][kv_head][head_dim]. */
__global__ void
picolm_gpu_attention_prefill_f32kv_kernel(
        float *xb_out,       /* [n_tokens][n_heads][head_dim] */
        const float *q_dev,  /* [n_tokens][n_heads][head_dim] */
        const float *kv_k,   /* [pos][kv_head][head_dim] FP32 */
        const float *kv_v,   /* [pos][kv_head][head_dim] FP32 */
        int start_pos, int n_tokens,
        int n_heads, int n_kv_heads, int head_dim)
{
    int h = (int)gpuBlockIdx_x;
    int tile_q_idx = (int)gpuBlockIdx_y;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    int kv_h = h / (n_heads / n_kv_heads);
    float attn_scale = 1.0f / sqrtf((float)head_dim);

    int q_start = tile_q_idx * ATTN_TILE_Q;
    int q_end = min(q_start + ATTN_TILE_Q, n_tokens);
    int n_q = q_end - q_start;

    /* Shared memory: K tile + V tile (FP32) + reduce + acc + max/sum */
    extern __shared__ uint8_t smem[];
    float *k_tile_f = (float *)smem;
    float *v_tile_f = k_tile_f + ATTN_TILE_K * head_dim;
    float *reduce_sh = v_tile_f + ATTN_TILE_K * head_dim;
    float *acc_sh = reduce_sh + 256;
    float *max_score_sh = acc_sh + (size_t)ATTN_TILE_Q * head_dim;
    float *sum_exp_sh = max_score_sh + ATTN_TILE_Q;
    __shared__ float rescale_sh, weight_sh;

    for (int i = tid; i < ATTN_TILE_Q * head_dim; i += n_threads) acc_sh[i] = 0.0f;
    for (int qi = tid; qi < ATTN_TILE_Q; qi += n_threads) {
        max_score_sh[qi] = -1e30f;
        sum_exp_sh[qi] = 0.0f;
    }
    for (int i = tid; i < 32; i += n_threads) reduce_sh[i] = 0.0f;
    gpuSyncthreads();

    int total_kv = start_pos + n_tokens;
    size_t kv_pos_stride = (size_t)n_kv_heads * head_dim;
    size_t kv_head_stride = head_dim;

    for (int t0 = 0; t0 < total_kv; t0 += ATTN_TILE_K) {
        int t_end = min(t0 + ATTN_TILE_K, total_kv);
        int tile_k_size = t_end - t0;

        for (int d = tid; d < head_dim; d += n_threads) {
            for (int ti = 0; ti < tile_k_size; ti++) {
                size_t off = (size_t)(t0 + ti) * kv_pos_stride + kv_h * kv_head_stride + d;
                k_tile_f[ti * head_dim + d] = kv_k[off];
                v_tile_f[ti * head_dim + d] = kv_v[off];
            }
        }
        gpuSyncthreads();

        for (int qi = 0; qi < n_q; qi++) {
            int global_q = q_start + qi;
            int global_pos = start_pos + global_q;
            const float *qg = q_dev + (size_t)(global_q * n_heads + h) * head_dim;
            float *accqi = acc_sh + (size_t)qi * head_dim;

            if (tid == 0) { rescale_sh = 1.0f; weight_sh = 1.0f; }
            gpuSyncthreads();

            for (int ti = 0; ti < tile_k_size; ti++) {
                int global_kv = t0 + ti;
                if (global_kv > global_pos) continue;

                float score;
                if (tid == 0) {
                    /* Match CPU AVX-512 accumulation order: 16-wide chunks + tree reduce.
                     * This produces bit-identical scores to the CPU vec_dot_f16_f32. */
                    int n_chunks = head_dim / 16;
                    float chunk[16] = {0};
                    for (int c = 0; c < n_chunks; c++) {
                        float s = 0;
                        for (int d = c * 16; d < (c + 1) * 16; d++) {
                            s = fmaf(qg[d], k_tile_f[ti * head_dim + d], s);
                        }
                        chunk[c] = s;
                    }
                    /* Generalized tree reduce: fold pairs until one accumulator remains. */
                    for (int stride = n_chunks / 2; stride > 0; stride >>= 1) {
                        for (int c = 0; c < stride; c++) {
                            chunk[c] += chunk[c + stride];
                        }
                    }
                    score = chunk[0] * attn_scale;
                    /* Debug: first Q token, first KV pos, head 0 */
#ifdef PICOLM_SSM_VERIFY
                    if (h == 0 && qi == 0 && ti == 0) {
                        printf("ATNDBG: f32kv q0[:4]={%.6f,%.6f,%.6f,%.6f} k0[:4]={%.6f,%.6f,%.6f,%.6f} score=%.6f\n",
                            qg[0], qg[1], qg[2], qg[3],
                            k_tile_f[0], k_tile_f[1], k_tile_f[2], k_tile_f[3], score);
                    }
#endif
                }

                if (tid == 0) {
                    if (score > max_score_sh[qi]) {
                        rescale_sh = expf(max_score_sh[qi] - score);
                        weight_sh = 1.0f;
                        sum_exp_sh[qi] = sum_exp_sh[qi] * rescale_sh + 1.0f;
                        max_score_sh[qi] = score;
                    } else {
                        rescale_sh = 1.0f;
                        weight_sh = expf(score - max_score_sh[qi]);
                        sum_exp_sh[qi] += weight_sh;
                    }
                }
                gpuSyncthreads();

                for (int d = tid; d < head_dim; d += n_threads) {
                    if (weight_sh == 1.0f) {
                        /* score > max: CPU AVX-512 fmadd(acc, correction, v) */
                        accqi[d] = fmaf(accqi[d], rescale_sh, v_tile_f[ti * head_dim + d]);
                    } else {
                        /* score <= max: CPU AVX-512 fmadd(v, weight, acc) */
                        accqi[d] = fmaf(weight_sh, v_tile_f[ti * head_dim + d], accqi[d]);
                    }
                }
                gpuSyncthreads();
            }
        }
    }

    for (int qi = 0; qi < n_q; qi++) {
        int global_q = q_start + qi;
        float inv_sum = (sum_exp_sh[qi] > 0.0f) ? (1.0f / sum_exp_sh[qi]) : 0.0f;
        float *xbhg = xb_out + (size_t)(global_q * n_heads + h) * head_dim;
        float *accqi = acc_sh + (size_t)qi * head_dim;
        for (int d = tid; d < head_dim; d += n_threads) {
            xbhg[d] = accqi[d] * inv_sum;
        }
    }
}

__global__ void
picolm_gpu_attention_prefill_kernel(
        float *xb_out,        /* [n_tokens][n_heads][head_dim] */
        const float *q_dev,   /* [n_tokens][n_heads][head_dim] */
        const uint16_t *kv_k, /* [layer][pos][kv_head][head_dim] FP16 */
        const uint16_t *kv_v, /* [layer][pos][kv_head][head_dim] FP16 */
        int layer_ordinal,
        int start_pos, int n_tokens,
        int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
        size_t kv_pos_stride_bytes,
        size_t kv_head_stride_bytes)
{
    int h = (int)gpuBlockIdx_x;       /* query head */
    int tile_q_idx = (int)gpuBlockIdx_y; /* tile of query tokens */
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    int kv_h = h / (n_heads / n_kv_heads);
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    /* Layer base offset in uint16_t */
    size_t layer_base = (size_t)layer_ordinal * max_seq_len * n_kv_heads * head_dim;

    /* Which query tokens does this block handle? */
    int q_start = tile_q_idx * ATTN_TILE_Q;
    int q_end = min(q_start + ATTN_TILE_Q, n_tokens);
    int n_q = q_end - q_start;

    /* Shared memory: K tile + V tile (u16) + reduce (256 float) +
     * acc[ATTN_TILE_Q][head_dim] (float) + max_score[ATTN_TILE_Q] +
     * sum_exp[ATTN_TILE_Q] (float). Sized by the host wrapper, which
     * also opts into extended dynamic shared memory since this total
     * exceeds the default 48KB for typical head_dim/tile sizes.
     *
     * Previously acc/max_score/sum_exp were per-thread local arrays
     * (acc[32][256] = 32KB/thread) -- spilled to local memory, declared
     * and zero-initialized redundantly by every one of n_threads
     * threads, and only ever read/written by thread 0 in the update
     * loop below while every other thread idled at the barrier. Same
     * bug class as the decode attention kernel, fixed the same way:
     * one shared copy, update parallelized across threads. */
    extern __shared__ uint8_t smem[];
    uint16_t *k_tile = (uint16_t *)smem;
    uint16_t *v_tile = k_tile + ATTN_TILE_K * head_dim;
    float *reduce_sh = (float *)((uint8_t *)v_tile + ATTN_TILE_K * head_dim * sizeof(uint16_t));
    float *acc_sh = reduce_sh + 256;               /* [ATTN_TILE_Q][head_dim] */
    float *max_score_sh = acc_sh + (size_t)ATTN_TILE_Q * head_dim;  /* [ATTN_TILE_Q] */
    float *sum_exp_sh = max_score_sh + ATTN_TILE_Q; /* [ATTN_TILE_Q] */
    __shared__ float rescale_sh, weight_sh;

    for (int i = tid; i < ATTN_TILE_Q * head_dim; i += n_threads) acc_sh[i] = 0.0f;
    for (int qi = tid; qi < ATTN_TILE_Q; qi += n_threads) {
        max_score_sh[qi] = -1e30f;
        sum_exp_sh[qi] = 0.0f;
    }
    /* Zero reduce_sh entries used for inter-warp partial sums (max 32 warps). */
    for (int i = tid; i < 32; i += n_threads) reduce_sh[i] = 0.0f;
    gpuSyncthreads();

    /* Total KV positions visible: start_pos + n_tokens */
    int total_kv = start_pos + n_tokens;

    /* Tile over KV positions */
    for (int t0 = 0; t0 < total_kv; t0 += ATTN_TILE_K) {
        int t_end = min(t0 + ATTN_TILE_K, total_kv);
        int tile_k_size = t_end - t0;

        /* Load K and V tile into shared memory */
        for (int d = tid; d < head_dim; d += n_threads) {
            for (int ti = 0; ti < tile_k_size; ti++) {
                size_t k_off = layer_base + kv_h * kv_head_stride_bytes / 2 + (size_t)(t0 + ti) * kv_pos_stride_bytes / 2 + d;
                k_tile[ti * head_dim + d] = kv_k[k_off];
                v_tile[ti * head_dim + d] = kv_v[k_off];
            }
        }
        gpuSyncthreads();

        /* For each query token in this block, compute attention over this KV tile */
        for (int qi = 0; qi < n_q; qi++) {
            int global_q = q_start + qi;
            int global_pos = start_pos + global_q;

            const float *qg = q_dev + (size_t)(global_q * n_heads + h) * head_dim;
            float *accqi = acc_sh + (size_t)qi * head_dim;

            /* Initialize shared scalars for this token to prevent stale
             * values from the previous qi iteration affecting softmax. */
            if (tid == 0) { rescale_sh = 1.0f; weight_sh = 1.0f; }
            gpuSyncthreads();

            for (int ti = 0; ti < tile_k_size; ti++) {
                int global_kv = t0 + ti;
                if (global_kv > global_pos) continue;

                /* Compute score: thread-0 only, AVX-512 matching accumulation.
                 * n_chunks of 16 with fmaf, then tree reduce. */
                float score;
                if (tid == 0) {
                    int n_chunks = head_dim / 16;
                    float chunk[16] = {0};
                    for (int c = 0; c < n_chunks; c++) {
                        float s = 0;
                        for (int d = c * 16; d < (c + 1) * 16; d++) {
                            s = fmaf(qg[d], gpu_fp16_to_fp32(k_tile[ti * head_dim + d]), s);
                        }
                        chunk[c] = s;
                    }
                    for (int stride = n_chunks / 2; stride > 0; stride >>= 1) {
                        for (int c = 0; c < stride; c++) chunk[c] += chunk[c + stride];
                    }
                    score = chunk[0] * attn_scale;
                    reduce_sh[0] = score;
                }
                if (tid == 0) score = reduce_sh[0];
                /* Debug: dump Q and K first 4 elements for first Q token, first KV pos, head 0 */
#ifdef PICOLM_SSM_VERIFY
                if (h == 0 && qi == 0 && ti == 0 && tid == 0) {
                    printf("ATNDBG: prefill l=%d q0[:4]={%.6f,%.6f,%.6f,%.6f} k0[:4]={%.6f,%.6f,%.6f,%.6f} score=%.6f\n",
                        layer_ordinal,
                        qg[0], qg[1], qg[2], qg[3],
                        gpu_fp16_to_fp32(k_tile[0]), gpu_fp16_to_fp32(k_tile[1]),
                        gpu_fp16_to_fp32(k_tile[2]), gpu_fp16_to_fp32(k_tile[3]),
                        score);
                }
#endif

                /* Online softmax branch decision on thread 0, broadcast
                 * via shared scalars; accumulator update parallelized
                 * across all threads below. */
                if (tid == 0) {
                    if (score > max_score_sh[qi]) {
                        rescale_sh = expf(max_score_sh[qi] - score);
                        weight_sh = 1.0f;
                        sum_exp_sh[qi] = sum_exp_sh[qi] * rescale_sh + 1.0f;
                        max_score_sh[qi] = score;
                    } else {
                        rescale_sh = 1.0f;
                        weight_sh = expf(score - max_score_sh[qi]);
                        sum_exp_sh[qi] += weight_sh;
                    }
                }
                gpuSyncthreads();

                if (weight_sh == 1.0f) {
                    /* score > max: CPU does fmadd(acc, correction, v) -- single rounding */
                    for (int d = tid; d < head_dim; d += n_threads) {
                        accqi[d] = fmaf(accqi[d], rescale_sh, gpu_fp16_to_fp32(v_tile[ti * head_dim + d]));
                    }
                } else {
                    /* score <= max: CPU AVX-512 does fmadd(v, weight, acc) -- single rounding */
                    for (int d = tid; d < head_dim; d += n_threads) {
                        accqi[d] = fmaf(weight_sh, gpu_fp16_to_fp32(v_tile[ti * head_dim + d]), accqi[d]);
                    }
                }
                gpuSyncthreads(); /* reduce_sh/rescale_sh/weight_sh reused next ti/qi */
            }
        }
    }

    /* Normalize and write output, parallelized across threads */
    for (int qi = 0; qi < n_q; qi++) {
        int global_q = q_start + qi;
        float inv_sum = (sum_exp_sh[qi] > 0.0f) ? (1.0f / sum_exp_sh[qi]) : 0.0f;
        float *xbhg = xb_out + (size_t)(global_q * n_heads + h) * head_dim;
        float *accqi = acc_sh + (size_t)qi * head_dim;
        for (int d = tid; d < head_dim; d += n_threads) {
            xbhg[d] = accqi[d] * inv_sum;
        }
    }
}

/* ---- SSM alpha/beta batched vec_dot kernel ----
 * Each thread block handles one head. Weights: column-major
 * [dim, n_heads], each column is a quantized row.
 *
 * Thread-0 sequential accumulation to bit-match the CPU scalar
 * reference (model.c ssm_forward, steps 9-10). IMPORTANT: for
 * Q8_0/Q4_0-weight heads the CPU reference does NOT multiply the raw
 * float activation against the dequantized weight -- it quantizes xb
 * to Q8_0 ONCE per token (quantize_row_q8_0) and reuses that quantized
 * copy for every head's dot product via vec_dot_q8_0_q8_0_deltas
 * (Q8_0 weights) or vec_dot_q4_0_q8_0 (Q4_0 weights), both of which are
 * genuine int8 x int8 MACs. Multiplying the un-quantized float x
 * against the dequantized weight (the previous version of this kernel,
 * and thread-0 across accumulation order alone) is a DIFFERENT,
 * strictly more precise computation than what the CPU actually does,
 * and will not reproduce the CPU's output bit-for-bit -- the int8
 * round-trip on x is lossy and that lossiness is part of the reference.
 * We replicate that quantization here, redundantly once per head-block
 * (dim is at most a few thousand elements, this is cheap relative to
 * the model as a whole), so the GPU path takes the same lossy path the
 * CPU does. Only the F32 weight case multiplies against x directly,
 * matching vec_dot_f32_f32's scalar fallback. */
#define PICOLM_SSM_VECDOT_MAX_DIM 8192

__global__ void
picolm_ssm_vecdot_kernel(float *out,
                          const float *x,
                          const void *weights,
                          gguf_type_t qtype,
                          int dim, int n_v_heads,
                          int row_bytes,
                          const int *head_map) {
    int h = gpuBlockIdx_x;
    if (h >= n_v_heads) return;
    int tid = gpuThreadIdx_x;
    int gh = head_map ? head_map[h] : h;
    const char *wrow = (const char *)weights + (size_t)gh * row_bytes;

    /* Q8_0-quantized copy of x, shared within the block (only thread 0
     * writes it, but declaring it __shared__ keeps it off each thread's
     * local/register budget). Matches quantize_row_q8_0's scalar path:
     * per 32-element block, scale = max(|x|)/127, round-to-nearest
     * (ties away from zero, via lroundf) clamped to [-127,127]/[-128
     * on the low side], delta round-tripped through fp16 exactly as
     * the CPU's block_q8_0.d field is. */
    __shared__ int8_t xq[PICOLM_SSM_VECDOT_MAX_DIM];
    __shared__ float xq_d[PICOLM_SSM_VECDOT_MAX_DIM / 32];

    if ((qtype == 2 || qtype == 8) && tid == 0) {
        int nb = dim / 32;
        for (int bi = 0; bi < nb; bi++) {
            float asmax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float v = x[bi * 32 + j];
                if (v < 0.0f) v = -v;
                if (v > asmax) asmax = v;
            }
            float d = asmax / 127.0f;
            xq_d[bi] = gpu_fp16_to_fp32(gpu_fp32_to_fp16(d));
            float id = (asmax != 0.0f) ? 127.0f / asmax : 0.0f;
            for (int j = 0; j < 32; j++) {
                int v = (int)lroundf(x[bi * 32 + j] * id);
                if (v > 127) v = 127;
                if (v < -127) v = -128;
                xq[bi * 32 + j] = (int8_t)v;
            }
        }
    }
    gpuSyncthreads();

    if (tid != 0) return;

    float sum = 0.0f;

    switch (qtype) {
    case 0: { /* F32 -- ssm_alpha/ssm_beta stay unquantized on some models
               * (Qwen3.6): these are small, precision-sensitive gating
               * projections. Matches vec_dot_f32_f32's scalar
               * fallback: plain left-to-right float accumulation. */
        const float *wrow_f = (const float *)wrow;
        for (int i = 0; i < dim; i++) sum += wrow_f[i] * x[i];
        break;
    }
    case 2: { /* Q4_0 weight x Q8_0(x) -- matches vec_dot_q4_0_q8_0 scalar tail */
        int n_blocks = dim / 32;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 18;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);
            const uint8_t *qs = blk + 2;
            const int8_t *yq = xq + bi * 32;
            int sumi0 = 0, sumi1 = 0;
            for (int j = 0; j < 16; j++) {
                int v0 = (qs[j] & 0x0F) - 8;
                int v1 = (qs[j] >> 4) - 8;
                sumi0 += v0 * yq[j];
                sumi1 += v1 * yq[j + 16];
            }
            sum += (float)(sumi0 + sumi1) * wd * xq_d[bi];
        }
        break;
    }
    case 8: { /* Q8_0 weight x Q8_0(x) -- matches vec_dot_q8_0_q8_0_deltas scalar tail */
        int n_blocks = dim / 32;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 34;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);
            const int8_t *qs = (const int8_t *)(blk + 2);
            const int8_t *yq = xq + bi * 32;
            int sumi = 0;
            for (int j = 0; j < 32; j++) sumi += (int)yq[j] * (int)qs[j];
            sum += (float)sumi * xq_d[bi] * wd;
        }
        break;
    }
    case 30: { /* BF16 -- dequant each weight to F32, accumulate */
        const uint16_t *wrow_bf = (const uint16_t *)wrow;
        for (int i = 0; i < dim; i++) sum += dequant_bf16(wrow, i) * x[i];
        break;
    }
    default:
        break;
    }
    out[h] = sum;
}

/* Batched-over-tokens version of the above: identical per-(token,head)
 * computation (including the same redundant per-block re-quantization
 * of x into Q8_0 -- kept for simplicity/correctness parity with the
 * per-token kernel; x is small enough that this isn't the bottleneck).
 * x is [n_tokens][dim], out is [n_tokens][n_v_heads]. */
__global__ void
picolm_ssm_vecdot_batch_kernel(float *out,
                                const float *x,
                                const void *weights,
                                gguf_type_t qtype,
                                int dim, int n_v_heads, int n_tokens,
                                int row_bytes,
                                const int *head_map, int in_stride, int out_stride) {
    int h = gpuBlockIdx_x;
    int t = gpuBlockIdx_y;
    if (h >= n_v_heads || t >= n_tokens) return;
    int tid = gpuThreadIdx_x;
    int gh = head_map ? head_map[h] : h;
    const char *wrow = (const char *)weights + (size_t)gh * row_bytes;
    int ts = in_stride > 0 ? in_stride : dim;
    const float *xt = x + (size_t)t * ts;

    __shared__ int8_t xq[PICOLM_SSM_VECDOT_MAX_DIM];
    __shared__ float xq_d[PICOLM_SSM_VECDOT_MAX_DIM / 32];

    if ((qtype == 2 || qtype == 8) && tid == 0) {
        int nb = dim / 32;
        for (int bi = 0; bi < nb; bi++) {
            float asmax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float v = xt[bi * 32 + j];
                if (v < 0.0f) v = -v;
                if (v > asmax) asmax = v;
            }
            float d = asmax / 127.0f;
            xq_d[bi] = gpu_fp16_to_fp32(gpu_fp32_to_fp16(d));
            float id = (asmax != 0.0f) ? 127.0f / asmax : 0.0f;
            for (int j = 0; j < 32; j++) {
                int v = (int)lroundf(xt[bi * 32 + j] * id);
                if (v > 127) v = 127;
                if (v < -127) v = -128;
                xq[bi * 32 + j] = (int8_t)v;
            }
        }
    }
    gpuSyncthreads();

    if (tid != 0) return;

    float sum = 0.0f;

    switch (qtype) {
    case 0: {
        const float *wrow_f = (const float *)wrow;
        for (int i = 0; i < dim; i++) sum += wrow_f[i] * xt[i];
        break;
    }
    case 2: {
        int n_blocks = dim / 32;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 18;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);
            const uint8_t *qs = blk + 2;
            const int8_t *yq = xq + bi * 32;
            int sumi0 = 0, sumi1 = 0;
            for (int j = 0; j < 16; j++) {
                int v0 = (qs[j] & 0x0F) - 8;
                int v1 = (qs[j] >> 4) - 8;
                sumi0 += v0 * yq[j];
                sumi1 += v1 * yq[j + 16];
            }
            sum += (float)(sumi0 + sumi1) * wd * xq_d[bi];
        }
        break;
    }
    case 8: {
        int n_blocks = dim / 32;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 34;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);
            const int8_t *qs = (const int8_t *)(blk + 2);
            const int8_t *yq = xq + bi * 32;
            int sumi = 0;
            for (int j = 0; j < 32; j++) sumi += (int)yq[j] * (int)qs[j];
            sum += (float)sumi * xq_d[bi] * wd;
        }
        break;
    }
    case 30: {
        /* BF16: raw bf16 array, no block structure.
         * Use dequant_bf16 (portable zero-extend), not __bfloat162float (broken). */
        const uint16_t *wrow_bf = (const uint16_t *)wrow;
        for (int i = 0; i < dim; i++) sum += dequant_bf16(wrow, i) * xt[i];
        break;
    }
    default:
        break;
    }
    int _os = out_stride > 0 ? out_stride : n_v_heads;
    out[(size_t)t * _os + h] = sum;
}

/* ---- SSM causal conv1d + state shift (fused) ----
 * Direct port of the CPU reference (ssm_forward, conv1d section): for
 * each channel co, sum d_conv taps (d_conv-1 from history in
 * conv_state, the newest tap from new_input), SiLU-activate, then
 * shift conv_state left and append new_input as the newest row.
 *
 * Embarrassingly parallel across conv_dim channels -- each thread
 * owns one channel end-to-end (read old state, compute, write output,
 * shift its own channel's history), no shared memory, no
 * synchronization needed at all. Genuinely new: no GPU conv1d existed
 * before this (CPU always did this step). */
__global__ void
picolm_gpu_ssm_conv1d_kernel(float *conv_output, float *conv_state,
                              const float *new_input, const float *conv1d_w,
                              int conv_dim, int d_conv) {
    int co = (int)gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (co >= conv_dim) return;

    int n_state_rows = d_conv - 1;
    float sum = 0.0f;
    for (int d = 0; d < n_state_rows; d++)
        sum += conv1d_w[d + co * d_conv] * conv_state[d * conv_dim + co];
    sum += conv1d_w[(d_conv - 1) + co * d_conv] * new_input[co];
    conv_output[co] = sum / (1.0f + expf(-sum)); /* silu */

    /* Shift history left, append newest -- safe to do after computing
     * the output above since each thread only ever touches its own
     * channel's column, no cross-thread dependency. */
    for (int r = 0; r < n_state_rows - 1; r++)
        conv_state[r * conv_dim + co] = conv_state[(r + 1) * conv_dim + co];
    if (n_state_rows > 0)
        conv_state[(n_state_rows - 1) * conv_dim + co] = new_input[co];
}

/* Host wrapper is below, after helper functions (find_ctx, gpu_ok, select_ctx).
 * See picolm_gpu_ssm_conv1d_dev definition further down. */

/* ---- SSM causal conv1d + state shift, BATCHED over tokens ----
 * Same computation as picolm_gpu_ssm_conv1d_kernel above, called
 * conceptually n_tokens times in sequence -- but each channel-thread
 * keeps its own small history window in registers across all n_tokens
 * instead of round-tripping through conv_state in global memory once
 * per token. conv_state is read once at the start and written once at
 * the end. d_conv is always small in practice (4 is typical for this
 * family of models); PICOLM_SSM_CONV_MAX_D_CONV is a generous cap --
 * the host wrapper below refuses to dispatch (returns 0, caller falls
 * back to the CPU/per-token path) if d_conv exceeds it, rather than
 * silently truncating history.
 *
 * Bit-exact with the per-token kernel run n_tokens times: hist[] is
 * updated with the identical shift-then-append order conv_state itself
 * would be, and the tap sum is accumulated in the identical order
 * (d=0..n_state_rows-1 then the new-sample tap), so this produces the
 * same float sequence, not just a mathematically equivalent one. */
#define PICOLM_SSM_CONV_MAX_D_CONV 16

__global__ void
picolm_gpu_ssm_conv1d_batch_kernel(float *conv_output,     /* out [n_tokens][stride] */
                                    float *conv_state,      /* in/out [d_conv-1][conv_dim], persistent */
                                    const float *new_input, /* in [n_tokens][stride] */
                                    const float *conv1d_w,  /* in [conv_dim][d_conv] */
                                    int conv_dim, int d_conv, int n_tokens, int stride) {
    int co = (int)gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (co >= conv_dim) return;

    int n_state_rows = d_conv - 1;

    float hist[PICOLM_SSM_CONV_MAX_D_CONV];
    for (int r = 0; r < n_state_rows; r++)
        hist[r] = conv_state[r * conv_dim + co];

    float w[PICOLM_SSM_CONV_MAX_D_CONV];
    for (int d = 0; d < d_conv; d++)
        w[d] = conv1d_w[d + co * d_conv];

    for (int t = 0; t < n_tokens; t++) {
        float new_sample = new_input[(size_t)t * stride + co];
        float sum = 0.0f;
        for (int d = 0; d < n_state_rows; d++)
            sum += w[d] * hist[d];
        sum += w[n_state_rows] * new_sample;
        conv_output[(size_t)t * stride + co] = sum / (1.0f + expf(-sum)); /* silu */

        for (int r = 0; r < n_state_rows - 1; r++)
            hist[r] = hist[r + 1];
        if (n_state_rows > 0)
            hist[n_state_rows - 1] = new_sample;
    }

    for (int r = 0; r < n_state_rows; r++)
        conv_state[r * conv_dim + co] = hist[r];
}

/* Host wrapper for the batched conv1d kernel is below, next to
 * picolm_gpu_ssm_conv1d_dev, for the same reason (needs find_ctx/gpu_ok). */

/* ---- SSM recurrence kernel ----
 *
 * One thread block per head, 256 threads per block.
 * Each head independently processes its d_state x d_state state block.
 * d_state can be up to 256 -> d_state^2 = 65536 elements.
 *
 * Steps per head:
 * 1. Decay: state *= gate_exp
 * 2. sk = state @ k (matrix-vector)
 * 3. d = (v - sk) * beta
 * 4. state += k * d^T (outer product)
 * 5. output = state @ q (matrix-vector), written dim-major
 */

/* ---- Warp-shuffle SSM recurrence kernel ----
 *
 * Row-parallel: each warp processes one row of the d_state x d_state
 * state matrix. Lanes within the warp distribute columns, and warp
 * shuffles perform the matvec reduction.
 *
 * Grid: (n_v_heads, 1, d_state/num_warps)
 * Block: (GPU_WARP_SIZE, num_warps, 1)
 * On CUDA (warp=32): block=(32,4,1)=128 threads, grid.z=32 for d_state=128
 * On HIP/gfx906 (wave=64): block=(64,4,1)=256 threads, grid.z=32 for d_state=128
 *
 * No shared memory, no syncthreads. All state in registers + warp shuffles.
 *
 * XOR shuffle: handled by gpu_shfl_xor_sync() wrapper, not via macro,
 * because HIP's __shfl_xor has different signature than CUDA's
 * __shfl_xor_sync. ROCm 6.x hipShflXorSync is only available when
 * HIP_ENABLE_WARP_SYNC_BUILTINS is defined. */

#if !defined(__HIP__)
__device__ inline float
gpu_shfl_xor_sync(float var, int laneMask, int width) {
    return __shfl_xor_sync(0xffffffff, var, laneMask, width);
}
#else
__device__ inline float
gpu_shfl_xor_sync(float var, int laneMask, int width) {
    return __shfl_xor(var, laneMask, width);
}
#endif

template <int warp_size>
__device__ inline float
picolm_warp_reduce_sum(float val) {
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        val += gpu_shfl_xor_sync(val, offset, warp_size);
    }
    return val;
}

/* NOTE (2026-08-11): This kernel was previously (32 lanes = 32 disjoint
 * column-blocks, one row per warp, XOR-shuffle tree reduction over all
 * 32 partials) row-parallel but used a completely different summation
 * tree from the CPU NEON reference (picolm_ssm_recurrence_kernel below):
 * NEON accumulates 4 *strided* partial sums (columns cb+0, cb+4, cb+8...
 * go into s0; cb+1,cb+5,... into s1; etc.) via serial fmaf, then combines
 * with exactly two pairwise adds: (s0+s1)+(s2+s3). A 32-wide tree over
 * differently-grouped partials is bit-different from that, and the
 * ~1e-7 per-row error it introduces compounds over 48 SSM layers into
 * wrong hidden states (see ssm_recurrence_verify.c / ssmreport.txt in
 * project notes -- gpu_fp64 vs cpu_neon was ALSO ~1.7e-7 off, proving
 * this is an order-of-operations mismatch, not a precision one).
 *
 * Fixed design: instead of 32 lanes cooperating on one row, split the
 * warp into GROUPS_PER_WARP groups of exactly 4 lanes each and give
 * every group its own row. Within a group, sub-lane `sub` (0..3) plays
 * the role of the CPU's s{sub} accumulator: it strides over columns
 * `vv*4 + sub` for vv in [0, S_v/4), doing the *same sequence* of fmaf
 * calls in the *same order* as the CPU loop. The two-step XOR-shuffle
 * reduction below (mask 1, then mask 2, both width-4) computes exactly
 * (s0+s1)+(s2+s3): float addition is commutative bit-for-bit (only
 * associativity/grouping affects rounding), so pairing lane 0 with lane
 * 1 first (mask=1) and then combining with the {2,3} pair (mask=2)
 * reproduces the CPU's exact grouping in every one of the 4 lanes.
 * This is bit-exact with the thread-0 kernel/CPU NEON path while still
 * processing GROUPS_PER_WARP rows concurrently per warp (8x on a
 * 32-wide warp) instead of thread-0's single-row-at-a-time serial loop
 * over all S_v rows in one thread.
 *
 * MUST be re-validated against ssm_recurrence_verify.c (gpu_fixed_step)
 * on real hardware before this dispatches in production -- this rewrite
 * has not been compiled/run on a GPU in this session. */
template <int S_v>
__global__ void
picolm_ssm_recurrence_warp_kernel(float *state,
                                   const float *q_conv,
                                   const float *k_conv,
                                   const float *v_conv,
                                   const float *gate_exp,
                                   const float *beta,
                                   float *ssm_output,
                                   int n_v_heads, int repeat) {
    constexpr int GROUP = 4;                         /* matches CPU's 4 strided accumulators */
    constexpr int GROUPS_PER_WARP = GPU_WARP_SIZE / GROUP;
    constexpr int d4 = S_v / GROUP;                  /* S_v always 16/32/64/128, multiple of 4 */

    const int head = gpuBlockIdx_x;
    const int lane = gpuThreadIdx_x;                 /* 0..GPU_WARP_SIZE-1 */
    const int group_in_warp = lane / GROUP;          /* which row within this warp */
    const int sub = lane % GROUP;                    /* 0..3, role of CPU's s0..s3 */
    const int row = (gpuBlockIdx_z * gpuBlockDim_y + gpuThreadIdx_y) * GROUPS_PER_WARP
                     + group_in_warp;
    if (head >= n_v_heads || row >= S_v) return;

    const int kh = head / repeat;
    const float ge = gate_exp[head];
    const float bh = beta[head];

    float *st = state + (size_t)head * S_v * S_v + (size_t)row * S_v;
    const float *khv = k_conv + (size_t)kh * S_v;
    const float *qh  = q_conv + (size_t)kh * S_v;
    const float v0 = v_conv[(size_t)head * S_v + row];

    /* Pass 1 (matches CPU's decay+sk loop): decay each of this sub-lane's
     * strided columns and fmaf-accumulate the dot product with k, in
     * ascending vv order, exactly like the CPU's s{sub} accumulator. */
    float s_acc = 0.0f;
    for (int vv = 0; vv < d4; vv++) {
        int c = vv * GROUP + sub;
        float r = st[c] * ge;
        st[c] = r;
        s_acc = fmaf(r, khv[c], s_acc);
    }
    /* (s0+s1)+(s2+s3) via two width-4 XOR shuffles -- see note above. */
    float t1 = s_acc + gpu_shfl_xor_sync(s_acc, 1, GROUP);
    float sk_row = t1 + gpu_shfl_xor_sync(t1, 2, GROUP);

    const float dv = (v0 - sk_row) * bh;

    /* Pass 2 (matches CPU's rank-1 update + output loop): update this
     * sub-lane's strided columns with the rank-1 term and fmaf-accumulate
     * the output dot product with q, same order as the CPU loop. */
    float o_acc = 0.0f;
    for (int vv = 0; vv < d4; vv++) {
        int c = vv * GROUP + sub;
        float r = fmaf(khv[c], dv, st[c]);
        st[c] = r;
        o_acc = fmaf(r, qh[c], o_acc);
    }
    float u1 = o_acc + gpu_shfl_xor_sync(o_acc, 1, GROUP);
    float out_row = u1 + gpu_shfl_xor_sync(u1, 2, GROUP);

    if (sub == 0) {
        ssm_output[(size_t)row * n_v_heads + head] = out_row;
    }
}

/* Old thread-0 kernel (fallback for unsupported d_state values):
 * One thread block per head, 256 threads per block.
 */
__global__ void
picolm_ssm_recurrence_kernel(float *state,
                              const float *q_conv,
                              const float *k_conv,
                              const float *v_conv,
                              const float *gate_exp,
                              const float *beta,
                              float *ssm_output,
                              int n_v_heads, int d_state,
                              int repeat) {
    int h = gpuBlockIdx_x;
    if (h >= n_v_heads) return;
    if (gpuThreadIdx_x != 0) return;

    /* Thread-0 sequential, matching CPU NEON accumulation order.
     * The DGX Spark runs ARM64 with NEON. The CPU ssm_head_task uses
     * 4-lane strided fmaf accumulators combined by pairwise reduction
     * (vaddvq_f32: (a0+a1)+(a2+a3)). This is a different summation
     * tree from strict left-to-right addition, producing ~1e-7 relative
     * difference that compounds over 48+ SSM layers.
     *
     * Replicate NEON's exact order: 4 independent fmaf() strided
     * accumulators, combined with pairwise (lo+hi) reduction.
     * Decay is applied per-element BEFORE the sk dot product,
     * matching NEON which decays each loaded element before fma. */
    int kh = h / repeat;
    const float *qh = q_conv + (size_t)kh * d_state;
    const float *khv = k_conv + (size_t)kh * d_state;
    const float *vh = v_conv + (size_t)h * d_state;
    float ge = gate_exp[h];
    float bh = beta[h];
    float *st = state + (size_t)h * d_state * d_state;

    int d4 = d_state / 4; /* d_state always 16/32/64/128 */
    float sk_local[256];
    float d_local[256];

    /* Decay + sk: 4-lane strided fmaf per row, matching NEON */
    for (int row = 0; row < d_state; row++) {
        float *sr = st + (size_t)row * d_state;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int vv = 0; vv < d4; vv++) {
            int cb = vv * 4;
            float r0 = sr[cb+0] * ge; sr[cb+0] = r0;
            float r1 = sr[cb+1] * ge; sr[cb+1] = r1;
            float r2 = sr[cb+2] * ge; sr[cb+2] = r2;
            float r3 = sr[cb+3] * ge; sr[cb+3] = r3;
            s0 = fmaf(r0, khv[cb+0], s0);
            s1 = fmaf(r1, khv[cb+1], s1);
            s2 = fmaf(r2, khv[cb+2], s2);
            s3 = fmaf(r3, khv[cb+3], s3);
        }
        sk_local[row] = (s0 + s1) + (s2 + s3);
        d_local[row] = (vh[row] - sk_local[row]) * bh;
    }

    /* Rank-1 update + output: 4-lane strided fmaf per row */
    for (int row = 0; row < d_state; row++) {
        float dv = d_local[row];
        float *sr = st + (size_t)row * d_state;
        float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
        for (int vv = 0; vv < d4; vv++) {
            int cb = vv * 4;
            float r0 = fmaf(khv[cb+0], dv, sr[cb+0]); sr[cb+0] = r0;
            float r1 = fmaf(khv[cb+1], dv, sr[cb+1]); sr[cb+1] = r1;
            float r2 = fmaf(khv[cb+2], dv, sr[cb+2]); sr[cb+2] = r2;
            float r3 = fmaf(khv[cb+3], dv, sr[cb+3]); sr[cb+3] = r3;
            o0 = fmaf(r0, qh[cb+0], o0);
            o1 = fmaf(r1, qh[cb+1], o1);
            o2 = fmaf(r2, qh[cb+2], o2);
            o3 = fmaf(r3, qh[cb+3], o3);
        }
        ssm_output[(size_t)row * n_v_heads + h] = (o0 + o1) + (o2 + o3);
    }
}

/* ---- w4a16_matmul kernel (Tensor Core path) ----
 *
 * NVIDIA: WMMA via nvcuda::wmma namespace (sm_70+, CUDA 10+)
 * AMD: hipWMMA via wmma namespace (gfx940+, ROCm 6.x)
 *
 * On chips without Tensor Cores (e.g. gfx906, sm_60), these kernels
 * are simply not compiled. picolm_gpu_w4a16_mlp() returns 0 and the
 * caller falls back to quant_matmul.
 *
 * Four warps share one A tile and compute 16x64 outputs.
 * Grid: [ceil(N/64), ceil(M/16)], Block: 256 threads (4 warps)
 *
 * Weights are in GGUF block_q4_0 format (18 bytes per 32 values).
 * Dequant on-the-fly to FP16 for the WMMA B tile.
 */

/* HIP WMMA: only on CDNA2+ (gfx940, gfx941, gfx942). gfx906/908 have no WMMA.
 * The hip/wmma/wmma.h header is only available in ROCm >= 6.4. On older
 * ROCm releases WMMA is silently disabled so the non-WMA path is used. */
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#ifdef __has_include
#if __has_include(<hip/wmma/wmma.h>)
#define PICOLM_GPU_WMMA_AVAILABLE 1
#include <hip/wmma/wmma.h>
#endif
#endif
#endif
#endif

/* CUDA WMMA: sm_70+ (Volta and newer)
 * Guard must exclude host compilation phase to avoid CUDAFE stub generation
 * failures (wmma types are device-only). Host gets a forward declaration. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define PICOLM_GPU_WMMA_AVAILABLE 1
#endif

/* Forward declarations for host compilation (stubs only, no body) */
#ifndef PICOLM_GPU_WMMA_AVAILABLE
__global__ void picolm_w4a16_matmul(float *y, const float *x, const void *weights,
                                     int M, int K, int N, int block_size);
__global__ void picolm_w4a16_gate_up(float *gate, float *up, const float *x,
                                      const void *gate_weights, const void *up_weights,
                                      int M, int K, int N, int block_size);
#endif

#ifdef PICOLM_GPU_WMMA_AVAILABLE

/* Dequant one nibble from block_q4_0 to FP16.
 * GGUF Q4_0: values 0..15 = low nibble of qs[0..15], values 16..31 = high nibble.
 * NOT interleaved (that was the Q4_0 precision bug from earlier). */
__device__ static inline half dequant_q4_0_elem_fp16(const void *blk, int j) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    int v;
    if (j < 16) v = b[2 + j] & 0xF;
    else v = b[2 + j - 16] >> 4;
#ifdef __HIP__
    half d; d.__x = d_raw;
    half val; val.__x = gpu_fp32_to_fp16((float)(v - 8));
    return d * val;
#else
    half d = __ushort_as_half(d_raw);
    half val = __float2half((float)(v - 8));
    return d * val;
#endif
}

__global__ void
picolm_w4a16_matmul(float *y, const float *x, const void *weights,
                     int M, int K, int N, int block_size) {
    int warp = gpuThreadIdx_x >> 5;
    int lane = gpuThreadIdx_x & 31;
    int m0 = gpuBlockIdx_y * 16;
    int n0 = gpuBlockIdx_x * 64 + warp * 16;

#ifdef __HIP__
    using namespace wmma;
    __shared__ half ah[256];
    __shared__ half bh[4][256];
    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.f);
#else
    using namespace nvcuda;
    __shared__ __half ah[256];
    __shared__ __half bh[4][256];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
#endif

    int nblocks = ((K + 31) / 32);  /* blocks per row */
    for (int k0 = 0; k0 < K; k0 += 16) {
        /* Load A tile (FP32 -> FP16) */
        for (int z = gpuThreadIdx_x; z < 256; z += gpuBlockDim_x) {
            int m = z / 16, k = z % 16;
            int gm = m0 + m, gk = k0 + k;
            ah[z] = (gm < M && gk < K) ? __float2half(x[(size_t)gm * K + gk]) : __float2half(0.f);
        }
        /* Dequant B tile from block_q4_0 */
        for (int z = lane; z < 256; z += 32) {
            int n = z / 16, gk = k0 + (z % 16), gn = n0 + n;
            if (gn >= N || gk >= K) {
                bh[warp][z] = __float2half(0.f);
                continue;
            }
            int bi = gk / 32, ji = gk % 32;
            const void *blk_p = (const char *)weights + (size_t)gn * nblocks * block_size + (size_t)bi * block_size;
            bh[warp][z] = dequant_q4_0_elem_fp16(blk_p, ji);
        }
        gpuSyncthreads();

#ifdef __HIP__
        fragment<matrix_a, 16, 16, 16, half, row_major> af;
        fragment<matrix_b, 16, 16, 16, half, col_major> bf;
        load_matrix_sync(af, ah, 16);
        load_matrix_sync(bf, bh[warp], 16);
        mma_sync(acc, af, bf, acc);
#else
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
        wmma::load_matrix_sync(af, ah, 16);
        wmma::load_matrix_sync(bf, bh[warp], 16);
        wmma::mma_sync(acc, af, bf, acc);
#endif
        gpuSyncthreads();
    }

#ifdef __HIP__
    __shared__ float out[4][256];
    store_matrix_sync(out[warp], acc, 16, mem_row_major);
#else
    __shared__ float out[4][256];
    wmma::store_matrix_sync(out[warp], acc, 16, wmma::mem_row_major);
#endif
    for (int z = lane; z < 256; z += 32) {
        int m = z / 16, n = z % 16;
        if (m0 + m < M && n0 + n < N)
            y[(size_t)(m0 + m) * N + n0 + n] = out[warp][z];
    }
}

/* ---- w4a16_gate_up kernel (fused gate+up) ---- */

__global__ void
picolm_w4a16_gate_up(float *gate, float *up, const float *x,
                      const void *gw, const void *uw,
                      int M, int K, int N, int block_size) {
    int warp = gpuThreadIdx_x >> 5;
    int lane = gpuThreadIdx_x & 31;
    int which = warp & 1, tile = warp >> 1;
    int m0 = gpuBlockIdx_y * 16;
    int n0 = gpuBlockIdx_x * 64 + tile * 16;
    const void *w = which ? uw : gw;
    float *y = which ? up : gate;

#ifdef __HIP__
    using namespace wmma;
    __shared__ half ah[256];
    __shared__ half bh[8][256];
    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.f);
#else
    using namespace nvcuda;
    __shared__ __half ah[256];
    __shared__ __half bh[8][256];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
#endif

    int nblocks = ((K + 31) / 32);
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int z = gpuThreadIdx_x; z < 256; z += gpuBlockDim_x) {
            int m = z / 16, k = z % 16;
            int gm = m0 + m, gk = k0 + k;
            ah[z] = (gm < M && gk < K) ? __float2half(x[(size_t)gm * K + gk]) : __float2half(0.f);
        }
        for (int z = lane; z < 256; z += 32) {
            int n = z / 16, gk = k0 + (z % 16), gn = n0 + n;
            if (gn >= N || gk >= K) {
                bh[warp][z] = __float2half(0.f);
                continue;
            }
            int bi = gk / 32, ji = gk % 32;
            const void *blk_p = (const char *)w + (size_t)gn * nblocks * block_size + (size_t)bi * block_size;
            bh[warp][z] = dequant_q4_0_elem_fp16(blk_p, ji);
        }
        gpuSyncthreads();

#ifdef __HIP__
        fragment<matrix_a, 16, 16, 16, half, row_major> af;
        fragment<matrix_b, 16, 16, 16, half, col_major> bf;
        load_matrix_sync(af, ah, 16);
        load_matrix_sync(bf, bh[warp], 16);
        mma_sync(acc, af, bf, acc);
#else
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
        wmma::load_matrix_sync(af, ah, 16);
        wmma::load_matrix_sync(bf, bh[warp], 16);
        wmma::mma_sync(acc, af, bf, acc);
#endif
        gpuSyncthreads();
    }

#ifdef __HIP__
    __shared__ float out[8][256];
    store_matrix_sync(out[warp], acc, 16, mem_row_major);
#else
    __shared__ float out[8][256];
    wmma::store_matrix_sync(out[warp], acc, 16, wmma::mem_row_major);
#endif
    for (int z = lane; z < 256; z += 32) {
        int m = z / 16, n = z % 16;
        if (m0 + m < M && n0 + n < N)
            y[(size_t)(m0 + m) * N + n0 + n] = out[warp][z];
    }
}

#endif /* PICOLM_GPU_WMMA_AVAILABLE */

/* ---- Host-side data structures ---- */

struct picolm_gpu_tensor {
    void *weights;
    gguf_type_t qtype;
    int I, O, device;
    size_t row_bytes;
    int block_size;  /* bytes per quant block (18 for q4_0, 34 for q8_0, etc.) */
    int tracked;
    int zero_copy;   /* 1 if using cudaHostRegister (unified memory), 0 if copied */
};

typedef struct {
    int device;
    int compute_major, compute_minor;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    float *host_x, *host_y;
    size_t host_x_cap, host_y_cap;
    /* Q8_0 GPU matmul scratch buffers */
    int8_t *q8_xq;      /* quantized input activations (int8) */
    float *q8_xd;       /* pre-converted x deltas (fp32) */
    size_t q8_xq_cap;   /* capacity in bytes for q8_xq */
    size_t q8_xd_cap;   /* capacity in bytes for q8_xd */
    gpuStream_t stream;
    size_t tensor_count, tensor_bytes;
    /* Phase 2: fixed-size device-resident pipeline buffers for
     * model_forward_gpu() decode (S=1 only). Allocated once at model
     * load via picolm_gpu_pipeline_alloc(), sized from model config, no
     * grow-on-demand -- deliberately NOT reserve()-based, so the
     * per-token layer chain never risks a mid-pass cudaFree/cudaMalloc
     * racing a still-queued kernel that reads the old pointer. Must not
     * alias x/y/q8_xq/q8_xd above (those stay used internally by
     * picolm_gpu_matmul_dev's Q8_0 path). */
    float *pipe_x, *pipe_xb, *pipe_q, *pipe_k, *pipe_v,
          *pipe_attn_out, *pipe_ffn_norm, *pipe_gate, *pipe_up;
    int pipe_ready; /* 1 once the above are allocated for this device */
    /* Split-K decode attention scratch: [n_heads*n_splits] max + sum
     * floats, then [n_heads*n_splits*head_dim] acc floats, grown via
     * reserve() on first use (size depends on n_splits, which varies
     * with context length, capped at ATTN_DECODE_MAX_SPLITS). */
    float *attn_partial;
    size_t attn_partial_cap;
    /* Prefill batch buffers: [max_seq_len][dim] for S>1 prefill pipeline.
     * Same set of buffers as decode but sized for n_tokens rows. */
    float *pipe_x_b, *pipe_xb_b, *pipe_q_b, *pipe_k_b, *pipe_v_b,
          *pipe_attn_out_b, *pipe_ffn_norm_b, *pipe_gate_b, *pipe_up_b;
    int pipe_b_ready;
    /* SSM pipeline buffers (decode, S=1). Allocated once per model if
     * model has SSM layers. Sized from model's SSM config (conv_dim,
     * ssm_d_inner, dt_rank). These are the working buffers used by the
     * hybrid SSM+attention branch in model_forward_gpu(). */
    float *ssm_qkv_raw;      /* [conv_dim] attn_qkv output, conv1d input */
    float *ssm_conv_out;     /* [conv_dim] conv1d output, split into q/k/v */
    float *ssm_xb2;          /* [ssm_d_inner] attn_gate_ssm output (z-gate) */
    float *ssm_xb2_remap;    /* [ssm_d_inner] remapped xb2 (if do_remap) */
    float *ssm_v_remap;      /* [ssm_d_inner] remapped v_conv (if do_remap) */
    float *ssm_alpha_raw;    /* [n_v_heads] alpha vecdot output */
    float *ssm_beta_raw;     /* [n_v_heads] beta vecdot output */
    float *ssm_gate_exp;     /* [n_v_heads] gate exp output */
    float *ssm_beta;         /* [n_v_heads] beta sigmoid output */
    float *ssm_output;       /* [ssm_d_inner] recurrence output */
    float *ssm_final_output; /* [ssm_d_inner] gated_norm output */
    int ssm_ready;           /* 1 once SSM buffers are allocated */
    /* Prefill SSM scratch buffer: grow-only, used by ssm_chunked_recurrence_dev */
    float *ssm_prefill_scratch;
    size_t ssm_prefill_scratch_cap;
    /* RMSNorm weight cache: maps host weight pointers to device copies */
    void *rmsnorm_w_dev[64];
    const void *rmsnorm_w_keys[64];
    int rmsnorm_w_n;
    } gpu_device_ctx_t;

#endif /* BACKEND_GPU_KERNELS_CUH */
