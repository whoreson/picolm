// backend_gpu_kernels.cu - GPU kernel definitions (moved from kernels.cuh)
// This file is compiled once to avoid ODR violations when kernels.cuh
// is included by multiple host .cu files.
#include "backend_gpu_kernels.cuh"

/* lines 7-184 from backend_gpu_kernels.cuh */
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

/* lines 194-228 from backend_gpu_kernels.cuh */
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

/* Quantize F32/BF16 weight rows to Q8_0 block format.
 * Each block handles one weight row (one head).
 * Output: contiguous block_q8_0 layout [n_heads][n_blocks*GPU_BLOCK_Q8_0_SIZE].
 * Grid: [n_heads], Block: 32 threads.
 * Each block quantizes all 32-element sub-blocks of its row. */
__global__ void
picolm_quantize_weights_to_q8_0(void *dst, const float *src, gguf_type_t qtype,
                                 int I, int n_heads, int src_stride, int dst_block_stride) {
    int h = gpuBlockIdx_x;
    if (h >= n_heads) return;
    int tid = gpuThreadIdx_x;

    int n_blocks = I / 32;
    int rs = src_stride > 0 ? src_stride : I;
    int ds = dst_block_stride > 0 ? dst_block_stride : n_blocks * GPU_BLOCK_Q8_0_SIZE;
    uint8_t *dst_row = (uint8_t *)dst + (size_t)h * ds;

    const float *row_f = src + (size_t)h * rs;

    /* For BF16 input, dequant in the load */
    if (qtype == 30) {
        const uint16_t *row_bf = (const uint16_t *)src + (size_t)h * (rs / sizeof(uint16_t));
        for (int bi = tid; bi < n_blocks; bi += 32) {
            uint8_t *blk = dst_row + (size_t)bi * GPU_BLOCK_Q8_0_SIZE;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float v = dequant_bf16((const void *)(row_bf + bi * 32 + j), 0);
                if (v < 0.0f) v = -v;
                if (v > amax) amax = v;
            }
            float d = amax / 127.0f;
            uint16_t d_raw = gpu_fp32_to_fp16(d);
            blk[0] = d_raw & 0xFF; blk[1] = (d_raw >> 8) & 0xFF;
            float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
            int8_t *qs = (int8_t *)(blk + 2);
            for (int j = 0; j < 32; j++) {
                float v = dequant_bf16((const void *)(row_bf + bi * 32 + j), 0);
                int q = (int)lroundf(v * id);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[j] = (int8_t)q;
            }
        }
    } else {
        for (int bi = tid; bi < n_blocks; bi += 32) {
            uint8_t *blk = dst_row + (size_t)bi * GPU_BLOCK_Q8_0_SIZE;
            const float *xb = row_f + bi * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float v = xb[j];
                if (v < 0.0f) v = -v;
                if (v > amax) amax = v;
            }
            float d = amax / 127.0f;
            uint16_t d_raw = gpu_fp32_to_fp16(d);
            blk[0] = d_raw & 0xFF; blk[1] = (d_raw >> 8) & 0xFF;
            float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
            int8_t *qs = (int8_t *)(blk + 2);
            for (int j = 0; j < 32; j++) {
                int q = (int)lroundf(xb[j] * id);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[j] = (int8_t)q;
            }
        }
    }
}

/* lines 232-259 from backend_gpu_kernels.cuh */
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

/* lines 273-336 from backend_gpu_kernels.cuh */
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

/* lines 351-426 from backend_gpu_kernels.cuh */
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

/* ---- FP16 tiled matmul kernel ----
 *
 * Loads each FP16 weight row into shared memory once per tile of F16_TILE_S
 * query positions, then reuses it for all positions in the tile. This
 * eliminates redundant global memory reads of the weight row (S/TILE_S
 * reduction). FP32 activations are read from global memory per position.
 *
 * When FAST_FP16_AVAILABLE is defined (HIP all archs, CUDA sm_60+), the
 * inner loop processes two FP16 values at once via half2, halving the
 * number of FP16->FP32 conversions. */
__global__ void
picolm_f16_f16_matmul_tiled(
    float *y,           /* [S][O] output, FP32 */
    const float *x,     /* [S][I] activations, FP32 */
    const uint16_t *w,  /* [O][I] weights, FP16 */
    int S, int I, int O, int row_bytes, int y_stride) {
    int o = gpuBlockIdx_x;
    int tile = gpuBlockIdx_y;
    if (o >= O) return;
    int s0 = tile * F16_TILE_S;
    if (s0 >= S) return;
    int s_count = min(F16_TILE_S, S - s0);
    int ys = y_stride > 0 ? y_stride : O;

    const uint8_t *wrow = (const uint8_t *)w + (size_t)o * row_bytes;

    /* Load weight row into shared memory (raw bytes) */
    extern __shared__ uint8_t wrow_sh[];
    for (int b = gpuThreadIdx_x; b < row_bytes; b += gpuBlockDim_x)
        wrow_sh[b] = wrow[b];
    gpuSyncthreads();

    const uint16_t *wrow_h = (const uint16_t *)wrow_sh;
    __shared__ double partial[256];

    for (int ls = 0; ls < s_count; ls++) {
        int s = s0 + ls;
        const float *xrow = x + (size_t)s * I;

        double sum = 0.0;
#ifdef FAST_FP16_AVAILABLE
        /* Process two FP16 values per iteration using half2 */
        for (int i = gpuThreadIdx_x * 2; i + 1 < I; i += gpuBlockDim_x * 2) {
            half2 wh = make_half2(__ushort_as_half(wrow_h[i]),
                                  __ushort_as_half(wrow_h[i + 1]));
            float2 xf;
            xf.x = xrow[i];
            xf.y = xrow[i + 1];
            float2 pf = __half22float2(wh);
            sum += (double)xf.x * (double)pf.x + (double)xf.y * (double)pf.y;
        }
        /* Handle odd remainder: thread 0 does the last element */
        if (threadIdx.x == 0 && (I & 1))
            sum += (double)xrow[I - 1] * (double)gpu_fp16_to_fp32(wrow_h[I - 1]);
#else
        for (int i = gpuThreadIdx_x; i < I; i += gpuBlockDim_x)
            sum += (double)xrow[i] * (double)gpu_fp16_to_fp32(wrow_h[i]);
#endif

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

/* ---- BF16 tiled matmul kernel ----
 *
 * Same tiling as picolm_f16_f16_matmul_tiled, but BF16->FP32 conversion
 * is a simple zero-extend of the upper 16 bits: ((uint32_t)w[i]) << 16. */
__global__ void
picolm_bf16_f32_matmul_tiled(
    float *y,           /* [S][O] output, FP32 */
    const float *x,     /* [S][I] activations, FP32 */
    const uint16_t *w,  /* [O][I] weights, BF16 raw */
    int S, int I, int O, int row_bytes, int y_stride) {
    int o = gpuBlockIdx_x;
    int tile = gpuBlockIdx_y;
    if (o >= O) return;
    int s0 = tile * F16_TILE_S;
    if (s0 >= S) return;
    int s_count = min(F16_TILE_S, S - s0);
    int ys = y_stride > 0 ? y_stride : O;

    const uint8_t *wrow = (const uint8_t *)w + (size_t)o * row_bytes;

    /* Load weight row into shared memory (raw bytes) */
    extern __shared__ uint8_t wrow_sh[];
    for (int b = gpuThreadIdx_x; b < row_bytes; b += gpuBlockDim_x)
        wrow_sh[b] = wrow[b];
    gpuSyncthreads();

    const uint16_t *wrow_h = (const uint16_t *)wrow_sh;
    __shared__ double partial[256];

    for (int ls = 0; ls < s_count; ls++) {
        int s = s0 + ls;
        const float *xrow = x + (size_t)s * I;

        double sum = 0.0;
        for (int i = gpuThreadIdx_x; i < I; i += gpuBlockDim_x) {
            /* BF16 -> FP32: zero-extend upper 16 bits, reinterpret as float */
            uint32_t bits = ((uint32_t)wrow_h[i]) << 16;
            float wf;
            memcpy(&wf, &bits, sizeof(float));
            sum += (double)xrow[i] * (double)wf;
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

__global__ void
picolm_q8_q8_matmul_imma(float *y, const int8_t *xq, const float *xd,
                          const void *weights, int S, int I, int O,
                          int row_bytes, int y_stride) {
    /* PTX ISA m16n8k32: groupID=t/4, tid_in_group=t%4 (ALL operands) */
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s = gpuBlockIdx_y * 16;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < n_blocks; kb++) {
        /* A: f=0->row=gid,col=tid*4; f=1->row=gid+8,col=tid*4
           f=2->row=gid,col=tid*4+16; f=3->row=gid+8,col=tid*4+16 */
        int a0, a1, a2, a3;
        {
            int c0 = tid * 4;
            int c2 = c0 + 16;
            int src0 = (int)(tile_s + gid) * I + kb * 32 + c0;
            int src1 = (int)(tile_s + gid + 8) * I + kb * 32 + c0;
            int src2 = (int)(tile_s + gid) * I + kb * 32 + c2;
            int src3 = (int)(tile_s + gid + 8) * I + kb * 32 + c2;
            int32_t *p = (int32_t *)(const int8_t *)(&xq[src0]); a0 = p[0];
            p = (int32_t *)(const int8_t *)(&xq[src1]); a1 = p[0];
            p = (int32_t *)(const int8_t *)(&xq[src2]); a2 = p[0];
            p = (int32_t *)(const int8_t *)(&xq[src3]); a3 = p[0];
        }
        /* B: f=0->row=tid*4,col=gid; f=1->row=tid*4+16,col=gid.
           Vectorized: 4 bytes per register are contiguous in Q8_0 block. */
        int b0, b1;
        {
            int wc = tile_o + gid;
            int r0 = tid * 4;
            int r1 = r0 + 16;
            const uint8_t *wblk = w + (size_t)wc * row_bytes +
                                  (size_t)kb * GPU_BLOCK_Q8_0_SIZE + 2;
            memcpy(&b0, wblk + r0, 4);
            memcpy(&b1, wblk + r1, 4);
        }
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
        /* D: Epilogue dedup. dx depends only on row (2 vals), wd on col (2 vals). */
        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];
        int wc0 = tile_o + tid * 2;
        int wc1 = tile_o + tid * 2 + 1;
        const uint8_t *wb0 = w + (size_t)wc0 * row_bytes + (size_t)kb * GPU_BLOCK_Q8_0_SIZE;
        const uint8_t *wb1 = w + (size_t)wc1 * row_bytes + (size_t)kb * GPU_BLOCK_Q8_0_SIZE;
        float wd0 = gpu_fp16_to_fp32(wb0[0] | ((uint16_t)wb0[1] << 8));
        float wd1 = gpu_fp16_to_fp32(wb1[0] | ((uint16_t)wb1[1] << 8));

        sum[0] += (float)d0 * dx0 * wd0;
        sum[1] += (float)d1 * dx0 * wd1;
        sum[2] += (float)d2 * dx1 * wd0;
        sum[3] += (float)d3 * dx1 * wd1;
    }
    for (int f = 0; f < 4; f++) {
        int row = (f < 2) ? gid : gid + 8;
        int col = tid * 2 + (f % 2);
        int gr = tile_s + row;
        int gc = tile_o + col;
        if (gr < S && gc < O)
            y[(size_t)gr * ys + (size_t)gc] = sum[f];
    }
}

/* lines 430-437 from backend_gpu_kernels.cuh */
__global__ void
picolm_silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

/* lines 488-599 from backend_gpu_kernels.cuh */
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

/* lines 627-733 from backend_gpu_kernels.cuh */
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

/* lines 737-789 from backend_gpu_kernels.cuh */
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

/* lines 804-940 from backend_gpu_kernels.cuh */
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

    /* Causal early-exit: this block processes Q tokens [q_start .. q_end).
     * The last Q token can attend to KV positions up to start_pos + q_end - 1.
     * Cap the KV loop to this block's own horizon, not the global prompt length. */
    int block_kv_limit = min(start_pos + n_tokens, start_pos + q_end);
    size_t kv_pos_stride = (size_t)n_kv_heads * head_dim;
    size_t kv_head_stride = head_dim;

    for (int t0 = 0; t0 < block_kv_limit; t0 += ATTN_TILE_K) {
        int t_end = min(t0 + ATTN_TILE_K, block_kv_limit);
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

/* lines 942-1110 from backend_gpu_kernels.cuh */
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

    /* Causal early-exit: this block processes Q tokens [q_start .. q_end).
     * The last Q token can attend to KV positions up to start_pos + q_end - 1.
     * Cap the KV loop to this block's own horizon, not the global prompt length. */
    int block_kv_limit = min(start_pos + n_tokens, start_pos + q_end);

    /* Tile over KV positions (bounded by this block's causal horizon) */
    for (int t0 = 0; t0 < block_kv_limit; t0 += ATTN_TILE_K) {
        int t_end = min(t0 + ATTN_TILE_K, block_kv_limit);
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

                /* Compute score: parallel across threads, same 16-element chunk
                 * accumulation order as the CPU AVX-512 reference. Each thread
                 * handles one chunk (head_dim/16 chunks, <= 16 for head_dim<=256).
                 * Tree reduce across threads. Bit-exact with original because
                 * chunk boundaries and fma order are preserved. */
                float score;
                {
                    int n_chunks = head_dim / 16;
                    float local_chunk = 0;
                    if (tid < n_chunks) {
                        for (int d = tid * 16; d < (tid + 1) * 16; d++) {
                            local_chunk = fmaf(qg[d], gpu_fp16_to_fp32(k_tile[ti * head_dim + d]), local_chunk);
                        }
                    }
                    reduce_sh[tid] = local_chunk;
                    gpuSyncthreads();
                    for (int stride = n_threads / 2; stride > 0; stride >>= 1) {
                        if (tid < stride) reduce_sh[tid] += reduce_sh[tid + stride];
                        gpuSyncthreads();
                    }
                    score = reduce_sh[0] * attn_scale;
                }
                
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

/* FA2 kernel - FP16 Tensor Core Flash Attention 2 Prefill
 *
 * mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 for Q@K scoring.
 * One block per (KV head, Q-tile of 64 rows). 4 warps per block, each
 * warp handles 16 Q rows for 1 query head within the GQA group.
 *
 * Shared memory: K tile (2KB) + V tile (2KB) + Q tile FP16 (8KB)
 *   + score buffer (4KB) + acc (32KB) + max/sum (512B) = 48.5KB
 *
 * Grid: [n_kv_heads, ceil(n_tokens/64)], Block: 128 threads.
 *
 * NOTE: NOT bit-exact with scalar kernel due to different accumulation
 * order in IMMA. Validate using tolerance (1e-2).
 */

#define FA2_TILE_Q 16
#define FA2_TILE_K 16

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

/* lines 1244-1341 from backend_gpu_kernels.cuh */
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

/* lines 1354-1375 from backend_gpu_kernels.cuh */
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

/* lines 1399-1434 from backend_gpu_kernels.cuh */
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


__global__ void
picolm_gpu_attention_prefill_fa2_kernel(
    float *xb_out, const float *q_dev,
    const uint16_t *kv_k, const uint16_t *kv_v,
    int layer_ordinal, int start_pos, int n_tokens,
    int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
    size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes)
{
    int q_h = gpuBlockIdx_x;        /* query head 0..n_heads-1 */
    int q_tile_idx = gpuBlockIdx_y;
    int t = gpuThreadIdx_x;
    int lt = t % 32;
    int warp = t / 32;
    int kv_mul = n_heads / n_kv_heads;
    int kv_h = q_h / kv_mul;

    float attn_scale = 1.0f / sqrtf((float)head_dim);
    size_t layer_base = (size_t)layer_ordinal * max_seq_len * n_kv_heads * head_dim;
    int n_total_kv = start_pos + n_tokens;
    int n_k16 = head_dim / 16;
    int row_stride32 = head_dim / 2;

    int warp_q_base = q_tile_idx * 64 + warp * FA2_TILE_Q;
    int groupID = lt >> 2;
    int tid_in_grp = lt & 3;

    extern __shared__ uint8_t smem[];
    uint16_t *k_tile_sh = (uint16_t *)smem;
    uint16_t *v_tile_sh = k_tile_sh + FA2_TILE_K * head_dim;
    uint16_t *q_tile_sh = v_tile_sh + FA2_TILE_K * head_dim;
    float *score_sh = (float *)((uint8_t *)q_tile_sh + 64 * head_dim * sizeof(uint16_t));
    float *acc_sh = score_sh + 64 * FA2_TILE_K;
    float *max_sh = acc_sh + 64 * head_dim;
    float *sum_sh = max_sh + 64;

    /* Phase 1: Load Q */
    {
        for (int d = lt; d < FA2_TILE_Q * head_dim; d += 32) {
            int row = d / head_dim;
            int dim = d % head_dim;
            int global_q = warp_q_base + row;
            if (global_q < n_tokens) {
                const float *q_row = q_dev + (size_t)(global_q * n_heads + q_h) * head_dim;
                q_tile_sh[(warp * FA2_TILE_Q + row) * head_dim + dim] = gpu_fp32_to_fp16(q_row[dim]);
            }
        }
    }
    gpuSyncthreads();

    /* Phase 2: Init all 64 Q rows */
    for (int i = t; i < 64 * head_dim; i += 128) acc_sh[i] = 0.0f;
    for (int i = t; i < 64; i += 128) { max_sh[i] = -1e30f; sum_sh[i] = 0.0f; }
    gpuSyncthreads();

    /* Phase 3: KV tiling with causal early-exit.
     * This block processes Q rows [warp_q_base .. warp_q_base + 64).
     * The last Q row can attend to KV positions up to start_pos + warp_q_base + 63.
     * Cap the loop to this horizon instead of the global n_total_kv. */
    int block_kv_limit_fa2 = min(n_total_kv, start_pos + warp_q_base + 64);
    for (int t0 = 0; t0 < block_kv_limit_fa2; t0 += FA2_TILE_K) {
        int t_end = min(t0 + FA2_TILE_K, block_kv_limit_fa2);
        int tk_size = t_end - t0;

        {
            for (int d = t; d < tk_size * head_dim; d += 128) {
                int ti = d / head_dim;
                int dim = d % head_dim;
                size_t kv_off = layer_base + kv_h * kv_head_stride_bytes / 2
                    + (size_t)(t0 + ti) * kv_pos_stride_bytes / 2 + dim;
                k_tile_sh[ti * head_dim + dim] = kv_k[kv_off];
                v_tile_sh[ti * head_dim + dim] = kv_v[kv_off];
            }
        }
        gpuSyncthreads();

        { for (int i = lt; i < 16 * FA2_TILE_K; i += 32) score_sh[(warp*FA2_TILE_Q) * FA2_TILE_K + i] = 0.0f; }
        /* Warp-level sync: each warp only reads its own score_sh region */
#ifndef __HIP__
        __syncwarp();
#else
        gpuSyncthreads();
#endif

        for (int k16 = 0; k16 < n_k16; k16++) {
#ifndef __HIP__
            for (int ki8 = 0; ki8 < tk_size; ki8 += 8) {
                int mc = tid_in_grp * 2;
                int col32 = (k16 * 16 + mc) / 2;
                int col32_2 = (k16 * 16 + mc + 8) / 2;
                int a0 = ((int *)q_tile_sh)[(warp*FA2_TILE_Q + groupID) * row_stride32 + col32];
                int a1 = ((int *)q_tile_sh)[(warp*FA2_TILE_Q + groupID + 8) * row_stride32 + col32];
                int a2 = ((int *)q_tile_sh)[(warp*FA2_TILE_Q + groupID) * row_stride32 + col32_2];
                int a3 = ((int *)q_tile_sh)[(warp*FA2_TILE_Q + groupID + 8) * row_stride32 + col32_2];

                int ki = ki8 + groupID;
                int dim32 = (k16 * 16 + tid_in_grp * 2) / 2;
                int dim32_2 = (k16 * 16 + tid_in_grp * 2 + 8) / 2;
                int b0 = ki < tk_size ? ((int *)k_tile_sh)[ki * row_stride32 + dim32] : 0;
                int b1 = ki < tk_size ? ((int *)k_tile_sh)[ki * row_stride32 + dim32_2] : 0;

                float d0 = 0.0f, d1 = 0.0f, d2_ = 0.0f, d3 = 0.0f;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(d0), "+f"(d1), "+f"(d2_), "+f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                );

                {
                    int k = ki8 + tid_in_grp * 2;
                    if (k < tk_size) score_sh[(warp*FA2_TILE_Q + groupID) * FA2_TILE_K + k] += d0;
                    k = ki8 + tid_in_grp * 2 + 1;
                    if (k < tk_size) score_sh[(warp*FA2_TILE_Q + groupID) * FA2_TILE_K + k] += d1;
                    k = ki8 + tid_in_grp * 2;
                    if (k < tk_size) score_sh[(warp*FA2_TILE_Q + groupID + 8) * FA2_TILE_K + k] += d2_;
                    k = ki8 + tid_in_grp * 2 + 1;
                    if (k < tk_size) score_sh[(warp*FA2_TILE_Q + groupID + 8) * FA2_TILE_K + k] += d3;
                }
            }
#endif /* !__HIP__ */
        }
        /* Warp-level sync: each warp only reads its own score_sh region */
#ifndef __HIP__
        __syncwarp();
#else
        gpuSyncthreads();
#endif

        /* Softmax + V accumulation per K-tile */
        {
            /* Each IMMA group (4 threads) computed scores for 2 Q rows.
             * Threads 0,1 in each group do softmax for those 2 rows.
             * Threads 2,3 are idle in softmax. */
            int srow = (tid_in_grp < 2) ? (groupID + tid_in_grp * 8) : -1;
            if (srow >= 0) {
                int qi = warp * FA2_TILE_Q + srow;
                int gq = warp_q_base + srow;
                if (gq < n_tokens) {
                    float *aq = acc_sh + qi * head_dim;
                    float rm = max_sh[qi], rs = sum_sh[qi];
                    for (int ti = 0; ti < tk_size; ti++) {
                        int gk = t0 + ti;
                        if (gk > start_pos + gq) continue;
                        float sc = score_sh[(warp*FA2_TILE_Q + srow) * FA2_TILE_K + ti] * attn_scale;
                        if (sc > rm) {
                            float re = expf(rm - sc);
                            for (int d = 0; d < head_dim; d++)
                                aq[d] = fmaf(aq[d], re, gpu_fp16_to_fp32(v_tile_sh[ti * head_dim + d]));
                            rs = rs * re + 1.0f; rm = sc;
                        } else {
                            float w = expf(sc - rm);
                            for (int d = 0; d < head_dim; d++)
                                aq[d] = fmaf(w, gpu_fp16_to_fp32(v_tile_sh[ti * head_dim + d]), aq[d]);
                            rs += w;
                        }
                    }
                    max_sh[qi] = rm; sum_sh[qi] = rs;
                }
            }
        }
        gpuSyncthreads();
    }

    /* Phase 4: Normalize and write output */
    gpuSyncthreads();
    for (int wr = 0; wr < FA2_TILE_Q; wr++) {
        int gq = warp_q_base + wr;
        if (gq >= n_tokens) continue;
        int qi = warp * FA2_TILE_Q + wr;
        float inv = (sum_sh[qi] > 0.0f) ? (1.0f / sum_sh[qi]) : 0.0f;
        float *xb = xb_out + (size_t)(gq * n_heads + q_h) * head_dim;
        float *aq = acc_sh + qi * head_dim;
        for (int d = lt; d < head_dim; d += 32)
            xb[d] = aq[d] * inv;
    }
}
