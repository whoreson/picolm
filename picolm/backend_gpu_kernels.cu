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
        case GGUF_TYPE_Q4_1: bytes_per_block = 20; break;                   /* Q4_1: 20 */
        case GGUF_TYPE_Q8_0: bytes_per_block = GPU_BLOCK_Q8_0_SIZE; break;  /* 34 */
        case 10:             bytes_per_block = GPU_BLOCK_Q2_K_SIZE; break;  /* Q2_K: 84 */
        case 11:             bytes_per_block = GPU_BLOCK_Q3_K_SIZE; break;   /* Q3_K: 110 */
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

    case 3: /* GGUF_TYPE_Q4_1 */
        /* I values in 32-value blocks (20 bytes each) */
        {
            int n_blocks = I / 32;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 32; j++) {
                    int i = bi * 32 + j;
                    sum += x[rs*s+i] * dequant_q4_1(blk, j);
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

    case 11: /* GGUF_TYPE_Q3_K */
        /* 256 values per block (110 bytes). Per-element dequant via helper. */
        {
            int n_blocks = I / 256;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 256; j++) {
                    int i = bi * 256 + j;
                    sum += x[rs*s+i] * dequant_q3_K(blk, j);
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
                    float wval = dequant_q6_K(blk, j);
                    sum += x[rs*s+i] * wval;
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
        /* Unsupported quantization type - output will be zero */
        if (gpuThreadIdx_x == 0 && s == 0) {
            printf("[GPU WARN] picolm_quant_matmul: unsupported qtype=%d (I=%d O=%d) - output zeroed\n", (int)qtype, I, O);
        }
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

/* Q6_K x Q8_0 IMMA kernel.
 *
 * Q6_K: 210 bytes per 256 weights. 6-bit values (stored 0..63, actual -32..31).
 * Layout: ql[128] + qh[64] + scales[16] + d[2].
 * 256 elements = 8 groups of 32. Each group has 2 int8 sub-scales (per 16 elems).
 *
 * IMMA m16n8k32: b0 covers K[0..15], b1 covers K[16..31].
 * Two IMMA calls per K-step: (b0=real,b1=0) then (b0=0,b1=real).
 *
 * Grid: [(O+7)/8, (S+15)/16], Block: 32 threads. Same as Q8_0 IMMA.
 */
__global__ void
picolm_q6_q8_matmul_imma(float *y, const int8_t *xq, const float *xd,
                          const void *weights, int S, int I, int O,
                          int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s = gpuBlockIdx_y * 16;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < n_blocks; kb++) {
        /* Q6_K block + group within block.
         * Each Q6_K block has 8 groups of 32 elements (8 IMMA K-steps). */
        int q6_block = kb / 8;
        int g = kb % 8;

        /* Weight column offset (O-dimension) */
        int wc = tile_o + gid;
        size_t wrow_base = (size_t)wc * row_bytes + q6_block * GPU_BLOCK_Q6_K_SIZE;

        /* Scales are read in epilogue from correct output columns (wc0,wc1),
         * not from gid column (which was only for B-fragment feeding). */

        /* Determine ql and qh byte offsets within this Q6_K block for group g.
         *
         * Chunk (0 or 1): chunk 0 = elems 0..127, chunk 1 = elems 128..255
         * Half (0 or 1): 0 = first 32 elems of chunk (ql[0..31] or [64..95]),
         *                1 = second 32 (ql[32..63] or [96..127])
         * Nibble (0 or 1): 0 = low nibble of ql, 1 = high nibble
         * QH pair (0..3): which pair of 2 bits from qh (0=bits 0-1, 1=bits 2-3,
         *                 2=bits 4-5, 3=bits 6-7)
         *
         * Group -> (chunk, half, nibble, qh_pair):
         *   0: (0, 0, 0, 0)  1: (0, 1, 0, 1)  2: (0, 0, 1, 2)  3: (0, 1, 1, 3)
         *   4: (1, 0, 0, 0)  5: (1, 1, 0, 1)  6: (1, 0, 1, 2)  7: (1, 1, 1, 3) */
        int chunk = g / 4;
        int half = (g % 4) % 2;      // 0,1,0,1 for g=0,1,2,3
        int nibble = (g % 4) / 2;    // 0,0,1,1 for g=0,1,2,3
        int qh_pair = g % 4;

        int ql_base = chunk * 64 + half * 32;   /* ql byte offset within Q6_K block */
        int qh_base = 128 + chunk * 32;          /* qh byte offset (128 = ql[128] end) */
        int qh_shift = qh_pair * 2;              /* bit shift for qh extraction */

        /* A: activation fragments (Q8_0, identical to Q8_0 IMMA kernel) */
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

        /* B: unpack Q6_K 6-bit values to int8, packed 4-per-int32.
         * b0 = elements tid*4..tid*4+3 (K[0..15]), b1 = tid*4+16..tid*4+19 (K[16..31]).
         *
         * Each element e uses:
         *   ql byte: ql_base + e (low or high nibble)
         *   qh byte: qh_base + e (extract qh_shift..qh_shift+1 bits)
         *   6-bit value: (nibble ? ql>>4 : ql&0xF) | (qh_bits << 4)
         *   int8 value: 6bit - 32 (per-byte subtraction) */

        /* Load ql bytes for b0 (4 bytes) and b1 (4 bytes) */
        uint32_t ql0_raw, ql1_raw;
        {
            const uint8_t *qlp = w + wrow_base + ql_base;
            memcpy(&ql0_raw, qlp + tid * 4, 4);
            memcpy(&ql1_raw, qlp + 16 + tid * 4, 4);
        }
        /* Extract low/high nibble */
        uint32_t ql0, ql1;
        if (nibble) {
            ql0 = (ql0_raw >> 4) & 0x0F0F0F0F;
            ql1 = (ql1_raw >> 4) & 0x0F0F0F0F;
        } else {
            ql0 = ql0_raw & 0x0F0F0F0F;
            ql1 = ql1_raw & 0x0F0F0F0F;
        }

        /* Load qh bytes and extract 2-bit high values */
        uint32_t qh0, qh1;
        {
            const uint8_t *qhp = w + wrow_base + qh_base;
            uint32_t qh0_raw, qh1_raw;
            memcpy(&qh0_raw, qhp + tid * 4, 4);
            memcpy(&qh1_raw, qhp + 16 + tid * 4, 4);
            /* Extract 2 bits from each byte, place at bits 4-5.
             * Must mask BEFORE shifting to avoid cross-byte contamination. */
            if (qh_shift == 0) {
                qh0 = (qh0_raw & 0x03030303) << 4;
                qh1 = (qh1_raw & 0x03030303) << 4;
            } else if (qh_shift == 2) {
                qh0 = ((qh0_raw >> 2) & 0x03030303) << 4;
                qh1 = ((qh1_raw >> 2) & 0x03030303) << 4;
            } else if (qh_shift == 4) {
                qh0 = qh0_raw & 0x30303030;
                qh1 = qh1_raw & 0x30303030;
            } else {
                qh0 = ((qh0_raw >> 6) & 0x03030303) << 4;
                qh1 = ((qh1_raw >> 6) & 0x03030303) << 4;
            }
        }

        /* Combine: 6-bit value = ql_nibble | qh_bits, then subtract 32 per byte.
         * Each byte holds a 6-bit value (0..63) -> subtract 32 -> (-32..31) as signed int8.
         * Must mask each byte to 8 bits after subtraction to avoid sign-extension leakage. */
        uint32_t b0, b1;
        { uint32_t v = ql0 | qh0; b0 = (((v & 0xFF) - 32) & 0xFF) | ((((v >> 8) & 0xFF) - 32) & 0xFF) << 8 | ((((v >> 16) & 0xFF) - 32) & 0xFF) << 16 | (((v >> 24) - 32) & 0xFF) << 24; }
        { uint32_t v = ql1 | qh1; b1 = (((v & 0xFF) - 32) & 0xFF) | ((((v >> 8) & 0xFF) - 32) & 0xFF) << 8 | ((((v >> 16) & 0xFF) - 32) & 0xFF) << 16 | (((v >> 24) - 32) & 0xFF) << 24; }

        /* Two IMMA calls: first K[0..15] with b0, then K[16..31] with b1 */
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        int d0b = 0, d1b = 0, d2b = 0, d3b = 0;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        /* First IMMA: b0 real, b1 zero */
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(0));
        /* Second IMMA: b0 zero, b1 real */
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0b), "+r"(d1b), "+r"(d2b), "+r"(d3b)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(0), "r"(b1));
#endif

        /* Epilogue: apply scales from the CORRECT output columns (tid*2, tid*2+1),
         * NOT from gid (which was only used for B-fragment feeding).
         * Matches picolm_q8_q8_matmul_imma epilogue pattern.
         * sum[0] -> col tid*2, row gid; sum[1] -> col tid*2+1, row gid
         * sum[2] -> col tid*2, row gid+8; sum[3] -> col tid*2+1, row gid+8 */
        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];

        /* Read scales for output column wc0 = tile_o + tid*2 */
        { size_t wb0_off = (size_t)(tile_o + tid * 2) * row_bytes + q6_block * GPU_BLOCK_Q6_K_SIZE;
          const uint8_t *wb0 = w + wb0_off;
          float wd0 = gpu_fp16_to_fp32(wb0[208] | ((uint16_t)wb0[209] << 8));
          float sc00 = wd0 * (float)(int8_t)wb0[192 + g * 2];
          float sc01 = wd0 * (float)(int8_t)wb0[192 + g * 2 + 1];
          sum[0] += (float)d0  * dx0 * sc00 + (float)d0b  * dx0 * sc01;
          sum[2] += (float)d2  * dx1 * sc00 + (float)d2b  * dx1 * sc01; }

        /* Read scales for output column wc1 = tile_o + tid*2 + 1 */
        { size_t wb1_off = (size_t)(tile_o + tid * 2 + 1) * row_bytes + q6_block * GPU_BLOCK_Q6_K_SIZE;
          const uint8_t *wb1 = w + wb1_off;
          float wd1 = gpu_fp16_to_fp32(wb1[208] | ((uint16_t)wb1[209] << 8));
          float sc10 = wd1 * (float)(int8_t)wb1[192 + g * 2];
          float sc11 = wd1 * (float)(int8_t)wb1[192 + g * 2 + 1];
          sum[1] += (float)d1  * dx0 * sc10 + (float)d1b  * dx0 * sc11;
          sum[3] += (float)d3  * dx1 * sc10 + (float)d3b  * dx1 * sc11; }
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

/* Q6_K decode warp-shuffle kernel (S=1, one token).
 *
 * Each thread computes one output column. All 32 threads cooperatively
 * dequantize one Q6_K block (256 values, 210 bytes) from global memory,
 * then perform the dot product against the single-token activation.
 *
 * Q6_K layout per block (210 bytes):
 *   ql[128] + qh[64] + scales[16] + d[2]
 * Each block has 16 sub-blocks of 16 values, each with its own int8 scale.
 * 6-bit values: 4 bits from ql, 2 bits from qh, stored unsigned [0..63],
 * actual value = stored - 32 (range -32..+31).
 *
 * Grid: [O], Block: 32 threads. */
__global__ void
picolm_q6_k_decode_warp(float *y, const float *x,
                         const void *weights, int I, int O,
                         int row_bytes) {
    int o = gpuBlockIdx_x;
    int tid = gpuThreadIdx_x;
    if (o >= O) return;

    const uint8_t *w = (const uint8_t *)weights;
    const uint8_t *wrow = w + (size_t)o * row_bytes;
    const float *xp = x;

    int n_blocks = I / 256;
    double sum = 0.0;

    for (int bi = 0; bi < n_blocks; bi++) {
        const uint8_t *blk = wrow + (size_t)bi * GPU_BLOCK_Q6_K_SIZE;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t *scales = (const int8_t *)(qh + 64);
        uint16_t d_raw = blk[208] | ((uint16_t)blk[209] << 8);
        float d = gpu_fp16_to_fp32(d_raw);

        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_c = ql + half * 64;
            const uint8_t *qh_c = qh + half * 32;
            for (int sub_chunk = 0; sub_chunk < 4; sub_chunk++) {
                /* Two scale groups per sub_chunk: low (sub_l 0..15) and high (16..31) */
                for (int sc_half = 0; sc_half < 2; sc_half++) {
                    int sc_idx = half * 8 + sub_chunk * 2 + sc_half;
                    float sc_v = (float)scales[sc_idx];
                    int sub_l_base = sc_half * 16;

                    for (int sub_l = tid + sub_l_base; sub_l < sub_l_base + 32; sub_l += 32) {
                        uint8_t ql_val, qh_val;
                        if (sub_chunk == 0) {
                            ql_val = ql_c[sub_l] & 0xF;
                            qh_val = (qh_c[sub_l] >> 0) & 3;
                        } else if (sub_chunk == 1) {
                            ql_val = ql_c[sub_l + 32] & 0xF;
                            qh_val = (qh_c[sub_l] >> 2) & 3;
                        } else if (sub_chunk == 2) {
                            ql_val = ql_c[sub_l] >> 4;
                            qh_val = (qh_c[sub_l] >> 4) & 3;
                        } else {
                            ql_val = ql_c[sub_l + 32] >> 4;
                            qh_val = (qh_c[sub_l] >> 6) & 3;
                        }

                        int q = (ql_val | (qh_val << 4)) - 32;
                        int elem_idx = bi * 256 + half * 128 + sub_chunk * 32 + sub_l;
                        sum += (double)xp[elem_idx] * (double)d * (double)sc_v * (double)q;
                    }
                }
            }
        }
    }
    y[o] = (float)sum;
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
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

/* Q8_0 x Q8_0 IMMA kernel with 16-wide output tiles.
 *
 * Same as picolm_q8_q8_matmul_imma but processes 16 output columns per block
 * instead of 8. Two m16n8k32 MMA invocations share the same A fragment
 * (activations), doubling arithmetic intensity per byte of activation read.
 * Grid: [(O+15)/16, (S+15)/16], Block: 32 threads (1 warp).
 *
 * Register pressure: 8 accumulators (vs 4), 4 B-fragment regs (vs 2),
 * 4 weight-scale reads (vs 2). Activation reads unchanged.
 */
__global__ void
picolm_q8_q8_matmul_imma_w16(float *y, const int8_t *xq, const float *xd,
                              const void *weights, int S, int I, int O,
                              int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 16;
    int tile_s = gpuBlockIdx_y * 16;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;
    float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < n_blocks; kb++) {
        /* A fragment: 16 rows x 32 cols of int8 (shared by both MMA invocations) */
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

        /* B fragment for first 8 output cols (tile_o + 0..7) */
        int b0a, b1a;
        {
            int wc = tile_o + gid;
            int r0 = tid * 4;
            int r1 = r0 + 16;
            const uint8_t *wblk = w + (size_t)wc * row_bytes +
                                  (size_t)kb * GPU_BLOCK_Q8_0_SIZE + 2;
            memcpy(&b0a, wblk + r0, 4);
            memcpy(&b1a, wblk + r1, 4);
        }
        /* B fragment for second 8 output cols (tile_o + 8..15) */
        int b0b, b1b;
        {
            int wc = tile_o + gid + 8;
            int r0 = tid * 4;
            int r1 = r0 + 16;
            const uint8_t *wblk = w + (size_t)wc * row_bytes +
                                  (size_t)kb * GPU_BLOCK_Q8_0_SIZE + 2;
            memcpy(&b0b, wblk + r0, 4);
            memcpy(&b1b, wblk + r1, 4);
        }

        int d0a = 0, d1a = 0, d2a = 0, d3a = 0;
        int d0b = 0, d1b = 0, d2b = 0, d3b = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0a), "+r"(d1a), "+r"(d2a), "+r"(d3a)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0a), "r"(b1a));
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0b), "+r"(d1b), "+r"(d2b), "+r"(d3b)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0b), "r"(b1b));
#endif
        /* Activation scales (shared by both 8-col halves) */
        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];

        /* Weight scales for first 8 cols */
        int wc0 = tile_o + tid * 2;
        int wc1 = tile_o + tid * 2 + 1;
        const uint8_t *wb0 = w + (size_t)wc0 * row_bytes + (size_t)kb * GPU_BLOCK_Q8_0_SIZE;
        const uint8_t *wb1 = w + (size_t)wc1 * row_bytes + (size_t)kb * GPU_BLOCK_Q8_0_SIZE;
        float wd0 = gpu_fp16_to_fp32(wb0[0] | ((uint16_t)wb0[1] << 8));
        float wd1 = gpu_fp16_to_fp32(wb1[0] | ((uint16_t)wb1[1] << 8));

        /* Weight scales for second 8 cols */
        int wc2 = tile_o + tid * 2 + 8;
        int wc3 = tile_o + tid * 2 + 9;
        const uint8_t *wb2 = w + (size_t)wc2 * row_bytes + (size_t)kb * GPU_BLOCK_Q8_0_SIZE;
        const uint8_t *wb3 = w + (size_t)wc3 * row_bytes + (size_t)kb * GPU_BLOCK_Q8_0_SIZE;
        float wd2 = gpu_fp16_to_fp32(wb2[0] | ((uint16_t)wb2[1] << 8));
        float wd3 = gpu_fp16_to_fp32(wb3[0] | ((uint16_t)wb3[1] << 8));

        /* Accumulate first 8 cols: sum[0..3] */
        sum[0] += (float)d0a * dx0 * wd0;
        sum[1] += (float)d1a * dx0 * wd1;
        sum[2] += (float)d2a * dx1 * wd0;
        sum[3] += (float)d3a * dx1 * wd1;

        /* Accumulate second 8 cols: sum[4..7] */
        sum[4] += (float)d0b * dx0 * wd2;
        sum[5] += (float)d1b * dx0 * wd3;
        sum[6] += (float)d2b * dx1 * wd2;
        sum[7] += (float)d3b * dx1 * wd3;
    }

    /* Write 16x16 output tile: 8 values per thread.
     * sum[0..3] -> first 8 cols (same layout as 16x8 kernel)
     * sum[4..7] -> second 8 cols (offset by 8) */
    for (int f = 0; f < 8; f++) {
        int row = (f % 4 < 2) ? gid : gid + 8;
        int half = f / 4;  /* 0 = first 8 cols, 1 = second 8 cols */
        int col = tid * 2 + (f % 2) + half * 8;
        int gr = tile_s + row;
        int gc = tile_o + col;
        if (gr < S && gc < O)
            y[(size_t)gr * ys + (size_t)gc] = sum[f];
    }

    }

/* Phase 6 (abandoned): cp.async pipelined IMMA was attempted but
 * Q8_0's 34-byte block format produces misaligned global memory
 * addresses that trigger cudaErrorMisalignedAddress (716) on sm_89.
 * cp.async requires strict 16B alignment. Shared memory staging
 * without cp.async adds a double-load penalty with no latency hiding.
 * See /data4/work/notes/picolm/phase6_cpasync_alignment_issue.md */

/* ================================================================
 * Phase 8: Shared-memory staged IMMA kernel (w16)
 * ================================================================
 * New approach: stage weight rows in shared memory to reduce L2
 * traffic. Each block processes a 32x16 output tile (2 warps,
 * 64 threads). Weight data for the 16 output columns is loaded
 * once per KB-tile into shared memory, then reused by all 32
 * rows. This doubles the arithmetic intensity per weight byte
 * fetched (32 rows vs 16 rows per weight fetch).
 *
 * Shared memory per KB-tile: 16 cols * GPU_BLOCK_Q8_0_SIZE bytes
 * = 16 * 34 = 544 bytes. Fits easily in 48KB.
 *
 * Grid: [(O+15)/16, (S+31)/32], Block: 64 threads (2 warps).
 * Dynamic shared memory: (O/16 block width doesn't matter,
 * one KB-tile staged at a time = ~544B).
 *
 * Key insight: the "double-load penalty" of staging weights in
 * shared memory is acceptable because each staged weight block
 * is reused across 32 output rows (2x the w16 kernel), netting
 * a 2x improvement in arithmetic intensity for weight reads.
 * ================================================================ */
#ifndef __HIP__
__global__ void
picolm_q8_q8_matmul_imma_smw16(float *y, const int8_t *xq, const float *xd,
                                const void *weights, int S, int I, int O,
                                int row_bytes, int y_stride) {
    extern __shared__ uint8_t smem[];

    int warp_id = gpuThreadIdx_x / 32;   /* 0 or 1 */
    int wid   = gpuThreadIdx_x % 32;     /* 0..31 within warp */
    int gid   = wid / 4;                 /* 0..7 within warp */
    int tid   = wid % 4;                 /* 0..3 within subgroup */

    int tile_o = gpuBlockIdx_x * 16;
    int tile_s = gpuBlockIdx_y * 32;     /* 32 rows per block (2 warps of 16) */
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;

    /* Each warp handles 16 rows: warp 0 = tile_s..tile_s+15,
     * warp 1 = tile_s+16..tile_s+31. */
    int warp_s_base = tile_s + warp_id * 16;

    float sum[8] = {0};

    /* Shared memory layout per KB-tile:
     * [wc*34 bytes for each of the 16 output columns]
     * Total: 16 * GPU_BLOCK_Q8_0_SIZE = 544 bytes */
    int smem_stride = GPU_BLOCK_Q8_0_SIZE; /* bytes per weight col in smem */

    for (int kb = 0; kb < n_blocks; kb++) {
        /* --- Stage 16 weight columns into shared memory ---
         * 16 columns x 34 bytes = 544B. 64 threads x 8.5 bytes/thread.
         * Each thread loads 8 or 9 bytes (byte-by-byte to avoid
         * crossing 34-byte block boundaries). */
        {
            for (int b = gpuThreadIdx_x; b < 16 * smem_stride; b += 64) {
                int src_col = b / smem_stride;
                int src_off = b % smem_stride;
                int wc = tile_o + src_col;
                smem[b] = w[(size_t)wc * row_bytes +
                            (size_t)kb * GPU_BLOCK_Q8_0_SIZE + src_off];
            }
        }
        gpuSyncthreads();

        /* --- Compute for this warp's 16 rows --- */
        {
            /* Activation scales for this warp's rows. */
            float dx0 = xd[(size_t)(warp_s_base + gid) * n_blocks + kb];
            float dx1 = xd[(size_t)(warp_s_base + gid + 8) * n_blocks + kb];

            /* A fragments: 4 x 4B reads from activation. */
            int a0, a1, a2, a3;
            {
                int c0 = tid * 4;
                int c2 = c0 + 16;
                int src0 = (int)(warp_s_base + gid) * I + kb * 32 + c0;
                int src1 = (int)(warp_s_base + gid + 8) * I + kb * 32 + c0;
                int src2 = (int)(warp_s_base + gid) * I + kb * 32 + c2;
                int src3 = (int)(warp_s_base + gid + 8) * I + kb * 32 + c2;
                int32_t *p = (int32_t *)(const int8_t *)(&xq[src0]); a0 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src1]); a1 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src2]); a2 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src3]); a3 = p[0];
            }

            /* B fragments from shared memory: two halves (cols 0-7, 8-15). */
            int b0a, b1a, b0b, b1b;
            {
                /* First half: cols tile_o + 0..7 */
                int r0 = tid * 4;
                int r1 = r0 + 16;
                memcpy(&b0a, smem + gid * smem_stride + 2 + r0, 4);
                memcpy(&b1a, smem + gid * smem_stride + 2 + r1, 4);
                /* Second half: cols tile_o + 8..15 */
                memcpy(&b0b, smem + (gid + 8) * smem_stride + 2 + r0, 4);
                memcpy(&b1b, smem + (gid + 8) * smem_stride + 2 + r1, 4);
            }

            int d0a = 0, d1a = 0, d2a = 0, d3a = 0;
            int d0b = 0, d1b = 0, d2b = 0, d3b = 0;
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(d0a), "+r"(d1a), "+r"(d2a), "+r"(d3a)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0a), "r"(b1a));
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(d0b), "+r"(d1b), "+r"(d2b), "+r"(d3b)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0b), "r"(b1b));

            /* Weight scales from shared memory. */
            {
                int wc0 = tid * 2, wc1 = wc0 + 1;
                uint16_t wd0_raw, wd1_raw;
                memcpy(&wd0_raw, smem + wc0 * smem_stride, 2);
                memcpy(&wd1_raw, smem + wc1 * smem_stride, 2);
                float wd0 = gpu_fp16_to_fp32(wd0_raw);
                float wd1 = gpu_fp16_to_fp32(wd1_raw);
                sum[0] += (float)d0a * dx0 * wd0;
                sum[1] += (float)d1a * dx0 * wd1;
                sum[2] += (float)d2a * dx1 * wd0;
                sum[3] += (float)d3a * dx1 * wd1;
            }
            {
                int wc2 = tid * 2 + 8, wc3 = wc2 + 1;
                uint16_t wd2_raw, wd3_raw;
                memcpy(&wd2_raw, smem + wc2 * smem_stride, 2);
                memcpy(&wd3_raw, smem + wc3 * smem_stride, 2);
                float wd2 = gpu_fp16_to_fp32(wd2_raw);
                float wd3 = gpu_fp16_to_fp32(wd3_raw);
                sum[4] += (float)d0b * dx0 * wd2;
                sum[5] += (float)d1b * dx0 * wd3;
                sum[6] += (float)d2b * dx1 * wd2;
                sum[7] += (float)d3b * dx1 * wd3;
            }
        }
        gpuSyncthreads(); /* needed before next iteration's smem write */
    }

    /* Write 16 output values per thread (8 accumulators). */
    for (int f = 0; f < 8; f++) {
        int row = (f % 4 < 2) ? gid : gid + 8;
        int half = f / 4;
        int col = tid * 2 + (f % 2) + half * 8;
        int gr = warp_s_base + row;
        int gc = tile_o + col;
        if (gr < S && gc < O)
            y[(size_t)gr * ys + (size_t)gc] = sum[f];
    }

    }
#endif /* __HIP__ */

/* ================================================================
 * Phase 7: Fused RMSNorm + Quantize kernel
 * ================================================================
 * Replaces: picolm_gpu_rmsnorm_batched_dev() + quantize kernel
 * Input: bx[S][dim] (strided) + rmsnorm weight[dim]
 * Output: q8_xq[S][I] + q8_xd[S][n_blocks]  (same buffers as regular quantize)
 *
 * One block per row, 256 threads. Two phases:
 *  Phase 1: RMSNorm reduction (sum of squares -> inv_rms)
 *  Phase 2: Normalize * weight, quantize to Q8_0 per 32-element block
 *
 * This eliminates the F32 bxb intermediate buffer entirely.
 * ================================================================ */
__global__ void
picolm_rmsnorm_quantize_q8_0_kernel(int8_t *qs_out, float *d_out,
                                     const float *x, const float *weight,
                                     int dim, float eps, int S, int x_stride) {
    int row = gpuBlockIdx_x;
    if (row >= S) return;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    const float *xr = x + (size_t)row * x_stride;
    int n_blocks = (dim + 31) / 32;

    /* Phase 1: RMSNorm reduction. */
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += n_threads) {
        sum_sq += xr[i] * xr[i];
    }
    __shared__ float ssum[256];
    ssum[tid] = sum_sq;
    gpuSyncthreads();
    for (int s = n_threads / 2; s > 0; s >>= 1) {
        if (tid < s) ssum[tid] += ssum[tid + s];
        gpuSyncthreads();
    }
    float inv_rms = rsqrtf(ssum[0] / dim + eps);

    /* Phase 2: Normalize * weight, quantize to Q8_0.
     * Each thread handles one 32-element block of this row. */
    for (int block = tid; block < n_blocks; block += n_threads) {
        const float *xb = xr + block * 32;
        int n = dim - block * 32;
        if (n > 32) n = 32;

        /* Find max abs value in this block (normalized). */
        float amax = 0.0f;
        for (int j = 0; j < n; j++) {
            float v = fabsf(xb[j] * inv_rms * weight[block * 32 + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;

        /* Quantize. */
        int8_t *qs_row = qs_out + (size_t)row * (size_t)n_blocks * 32 + (size_t)block * 32;
        for (int j = 0; j < n; j++) {
            float v = xb[j] * inv_rms * weight[block * 32 + j];
            int q = (int)lrintf(v * id);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            qs_row[j] = (int8_t)q;
        }
        /* Zero out any padding elements for the last (partial) block. */
        for (int j = n; j < 32; j++) {
            qs_row[j] = 0;
        }

        /* Write scale for this block. */
        d_out[(size_t)row * n_blocks + block] = d;
    }
}

/* ================================================================
 * Phase 4: Fused QKV IMMA kernel (W16 variant)
 * ================================================================
 * Single kernel launch for Q+K+V projections.
 * 16x16 output tiles (each thread: 8 accumulators).
 * Reads activation once per KB-tile, reuses for three weight matrices.
 * Eliminates 2 kernel launch overheads and keeps L2 cache warm
 * across all three matmuls.
 *
 * Each block: compute Q 16x16 tile, then K 16x16 tile, then V 16x16 tile.
 * Per KB-tile: 1 A-fragment read + 6 IMMA invocations (2 per projection).
 * Grid: [(max(Oq,Ok,Ov)+15)/16, (S+15)/16], Block: 32 threads.
 * ================================================================ */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)

/* Helper: compute one 16x16 IMMA tile for a single weight matrix.
 * Reads A fragments per KB-tile. Accumulates into sum[8]. */
#ifndef __HIP__
__device__ void
imma_qkv_one_w16(const uint8_t *w, const int8_t *xq, const float *xd,
                 int tile_o, int tile_s, int gid, int tid,
                 int I, int row_bytes, float sum[8]) {
    int n_blocks = I / 32;
    for (int kb = 0; kb < n_blocks; kb++) {
        /* A fragments: 16 rows x 32 cols of int8 (shared by both MMA halves). */
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

        /* B fragment first half: cols 0..7. */
        int b0a, b1a;
        {
            int wc = tile_o + gid;
            int r0 = tid * 4;
            int r1 = r0 + 16;
            const uint8_t *wblk = w + (size_t)wc * row_bytes +
                                  (size_t)kb * GPU_BLOCK_Q8_0_SIZE + 2;
            memcpy(&b0a, wblk + r0, 4);
            memcpy(&b1a, wblk + r1, 4);
        }
        /* B fragment second half: cols 8..15. */
        int b0b, b1b;
        {
            int wc = tile_o + gid + 8;
            int r0 = tid * 4;
            int r1 = r0 + 16;
            const uint8_t *wblk = w + (size_t)wc * row_bytes +
                                  (size_t)kb * GPU_BLOCK_Q8_0_SIZE + 2;
            memcpy(&b0b, wblk + r0, 4);
            memcpy(&b1b, wblk + r1, 4);
        }

        int d0a = 0, d1a = 0, d2a = 0, d3a = 0;
        int d0b = 0, d1b = 0, d2b = 0, d3b = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0a), "+r"(d1a), "+r"(d2a), "+r"(d3a)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0a), "r"(b1a));
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0b), "+r"(d1b), "+r"(d2b), "+r"(d3b)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0b), "r"(b1b));

        /* Activation scales (shared by both 8-col halves). */
        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];

        /* Weight scales first half. */
        {
            int wc0 = tile_o + tid * 2, wc1 = tile_o + tid * 2 + 1;
            const uint8_t *wb0 = w + (size_t)wc0 * row_bytes + (size_t)kb * GPU_BLOCK_Q8_0_SIZE;
            const uint8_t *wb1 = w + (size_t)wc1 * row_bytes + (size_t)kb * GPU_BLOCK_Q8_0_SIZE;
            uint16_t wd0_raw, wd1_raw;
            memcpy(&wd0_raw, wb0, 2);
            memcpy(&wd1_raw, wb1, 2);
            float wd0 = gpu_fp16_to_fp32(wd0_raw);
            float wd1 = gpu_fp16_to_fp32(wd1_raw);
            sum[0] += (float)d0a * dx0 * wd0;
            sum[1] += (float)d1a * dx0 * wd1;
            sum[2] += (float)d2a * dx1 * wd0;
            sum[3] += (float)d3a * dx1 * wd1;
        }
        /* Weight scales second half. */
        {
            int wc2 = tile_o + tid * 2 + 8;
            int wc3 = tile_o + tid * 2 + 9;
            const uint8_t *wb2 = w + (size_t)wc2 * row_bytes + (size_t)kb * GPU_BLOCK_Q8_0_SIZE;
            const uint8_t *wb3 = w + (size_t)wc3 * row_bytes + (size_t)kb * GPU_BLOCK_Q8_0_SIZE;
            uint16_t wd2_raw, wd3_raw;
            memcpy(&wd2_raw, wb2, 2);
            memcpy(&wd3_raw, wb3, 2);
            float wd2 = gpu_fp16_to_fp32(wd2_raw);
            float wd3 = gpu_fp16_to_fp32(wd3_raw);
            sum[4] += (float)d0b * dx0 * wd2;
            sum[5] += (float)d1b * dx0 * wd3;
            sum[6] += (float)d2b * dx1 * wd2;
            sum[7] += (float)d3b * dx1 * wd3;
        }
    }
}
#endif /* __HIP__ */

/* Write 8 FP32 output values from sum[] to the output buffer. */
#ifndef __HIP__
__device__ void
imma_qkv_write_tile(float *out, int tile_o, int tile_s, int gid, int tid,
                    int S, int O, int ys, const float sum[8]) {
    for (int f = 0; f < 8; f++) {
        int row = (f % 4 < 2) ? gid : gid + 8;
        int half = f / 4;
        int col = tid * 2 + (f % 2) + half * 8;
        int gr = tile_s + row;
        int gc = tile_o + col;
        if (gr < S && gc < O)
            out[(size_t)gr * ys + (size_t)gc] = sum[f];
    }
}

__global__ void
picolm_q8_q8_matmul_imma_qkv(float *bq, float *bk, float *bv,
                              const int8_t *xq, const float *xd,
                              const void *weights_q, const void *weights_k,
                              const void *weights_v,
                              int S, int I,
                              int Oq, int Ok, int Ov,
                              int row_bytes,
                              int ys_q, int ys_kv) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_s = gpuBlockIdx_y * 16;
    int tile_o = gpuBlockIdx_x * 16;

    const uint8_t *wq = (const uint8_t *)weights_q;
    const uint8_t *wk = (const uint8_t *)weights_k;
    const uint8_t *wv = (const uint8_t *)weights_v;

    /* ---- Q projection ---- */
    if (tile_o + 16 <= Oq || tile_o < Oq) {
        float sum[8] = {0};
        imma_qkv_one_w16(wq, xq, xd, tile_o, tile_s, gid, tid, I, row_bytes, sum);
        imma_qkv_write_tile(bq, tile_o, tile_s, gid, tid, S, Oq, ys_q, sum);
    }

    /* ---- K projection ---- */
    if (tile_o + 16 <= Ok || tile_o < Ok) {
        float sum[8] = {0};
        imma_qkv_one_w16(wk, xq, xd, tile_o, tile_s, gid, tid, I, row_bytes, sum);
        imma_qkv_write_tile(bk, tile_o, tile_s, gid, tid, S, Ok, ys_kv, sum);
    }

    /* ---- V projection ---- */
    if (tile_o + 16 <= Ov || tile_o < Ov) {
        float sum[8] = {0};
        imma_qkv_one_w16(wv, xq, xd, tile_o, tile_s, gid, tid, I, row_bytes, sum);
        imma_qkv_write_tile(bv, tile_o, tile_s, gid, tid, S, Ov, ys_kv, sum);
    }
}
#endif /* __HIP__ */

#endif /* __CUDA_ARCH__ >= 800 */


/* Q4_0 x Q8_0 IMMA kernel.
 *
 * Q4_0: 18 bytes per 32 weights. 4-bit values (stored 0..15, actual -8..7).
 * Layout: d[2] (fp16 scale) + qs[16] (16 bytes of packed nibbles).
 * Nibble layout per GGUF: byte j = {elem j low nibble, elem j+16 high nibble}.
 * Dequant: val = d * (qs[j] & 0xF - 8) for low, d * (qs[j] >> 4 - 8) for high.
 *
 * IMMA m16n8k32: 1 call per K-step (uniform scale per 32).
 * Grid: [(O+7)/8, (S+15)/16], Block: 32 threads. Same as Q8_0 IMMA.
 */
__global__ void
picolm_q4_0_q8_matmul_imma(float *y, const int8_t *xq, const float *xd,
                            const void *weights, int S, int I, int O,
                            int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s = gpuBlockIdx_y * 16;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < n_blocks; kb++) {
        /* A: activation fragments (same as Q8_0) */
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

        /* B: unpack Q4_0 nibbles to int32 (4 per register).
         * Block: d[0-1] fp16, qs[2..17] (16 bytes).
         * Each qs byte j holds elem j (low nibble) and elem j+16 (high nibble).
         * b0 = elements tid*4..tid*4+3 (K[0..15]): from qs bytes tid*2 and tid*2+1.
         * b1 = elements tid*4+16..tid*4+19 (K[16..31]): from qs bytes 16+tid*2 and 16+tid*2+1.
         *
         * For IMMA s8, each byte must be sign-extended to int8.
         * Q4_0 stores values 0..15, actual -8..7 after subtracting 8.
         * We subtract 0x08 from each byte before IMMA (per-byte). */
        int b0, b1;
        {
            int wc = tile_o + gid;
            const uint8_t *qs = w + (size_t)wc * row_bytes +
                                (size_t)kb * GPU_BLOCK_Q4_0_SIZE + 2;
            uint32_t raw0, raw1;
            memcpy(&raw0, qs + tid * 2, 2);
            memcpy(&raw1, qs + 16 + tid * 2, 2);

            /* b0: low nibbles of 2 bytes -> 4 elements 0..3 */
            uint32_t lo = (raw0 & 0x0F) | (((raw0 >> 4) & 0x0F) << 8)
                        | ((raw0 >> 12) & 0x0F) << 16 | ((raw0 >> 20) & 0x0F) << 24;
            /* b1: high nibbles of 2 bytes -> 4 elements 16..19 */
            uint32_t hi = (raw1 >> 4) & 0x0F;
            hi |= (((raw1 >> 8) & 0x0F) << 8);
            hi |= (((raw1 >> 12) & 0x0F) << 16);
            hi |= ((raw1 >> 20) & 0x0F) << 24;

            /* Subtract 8 from each byte for signed interpretation */
            { uint32_t v = lo; b0 = ((v & 0xFF) - 8) | (((v >> 8) & 0xFF) - 8) << 8 | (((v >> 16) & 0xFF) - 8) << 16 | ((v >> 24) - 8) << 24; }
            { uint32_t v = hi; b1 = ((v & 0xFF) - 8) | (((v >> 8) & 0xFF) - 8) << 8 | (((v >> 16) & 0xFF) - 8) << 16 | ((v >> 24) - 8) << 24; }
        }

        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif

        /* Epilogue: single scale per block (shared across all output columns) */
        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];
        int wc = tile_o + gid;
        const uint8_t *wb = w + (size_t)wc * row_bytes + (size_t)kb * GPU_BLOCK_Q4_0_SIZE;
        float wd = gpu_fp16_to_fp32(wb[0] | ((uint16_t)wb[1] << 8));

        sum[0] += wd * dx0 * (float)d0;
        sum[1] += wd * dx0 * (float)d1;
        sum[2] += wd * dx1 * (float)d2;
        sum[3] += wd * dx1 * (float)d3;
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

/* Q4_1 x Q8_0 IMMA kernel.
 *
 * Q4_1: 20 bytes per 32 weights. Unsigned nibbles + per-block min.
 * Layout: d[2] (fp16 scale) + m[2] (fp16 min) + qs[16] (packed nibbles).
 * Nibble layout per GGUF: byte j = {elem j low, elem j+16 high}.
 * Dequant: val = d * qs[j] + m, where qs[j] is unsigned 0..15.
 *
 * IMMA m16n8k32: b fragments hold unsigned 0..15 in low byte.
 * Epilogue: sum += d * IMMA_result + m * sum(q8) * dx.
 * The m * sum(q8) term: sum of unsigned nibbles per K-block.
 *
 * Grid: [(O+7)/8, (S+15)/16], Block: 32 threads. Same as Q8_0 IMMA.
 */
__global__ void
picolm_q4_1_q8_matmul_imma(float *y, const int8_t *xq, const float *xd,
                            const void *weights, int S, int I, int O,
                            int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s = gpuBlockIdx_y * 16;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < n_blocks; kb++) {
        /* A: activation fragments (same as Q8_0) */
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

        /* B: unpack Q4_1 nibbles to int32 (unsigned 0..15 per byte).
         * Block: d[0-1] fp16, m[2-3] fp16, qs[4..19] (16 bytes).
         * Same nibble layout as Q4_0. IMMA will sign-extend from low
         * byte, but nibbles 0..15 have high bit clear, so sign-extension
         * yields the correct unsigned value 0..15. */
        int b0, b1;
        {
            int wc = tile_o + gid;
            const uint8_t *qs = w + (size_t)wc * row_bytes +
                                (size_t)kb * GPU_BLOCK_Q4_1_SIZE + 4;
            uint32_t raw0, raw1;
            memcpy(&raw0, qs + tid * 2, 2);
            memcpy(&raw1, qs + 16 + tid * 2, 2);

            uint32_t lo = (raw0 & 0x0F) | (((raw0 >> 4) & 0x0F) << 8)
                        | ((raw0 >> 12) & 0x0F) << 16 | ((raw0 >> 20) & 0x0F) << 24;
            uint32_t hi = (raw1 >> 4) & 0x0F;
            hi |= (((raw1 >> 8) & 0x0F) << 8);
            hi |= (((raw1 >> 12) & 0x0F) << 16);
            hi |= ((raw1 >> 20) & 0x0F) << 24;

            b0 = lo;
            b1 = hi;
            /* No subtract: nibbles 0..15 already have correct sign bit */
        }

        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif

        /* Epilogue: sum += d * IMMA + m * sum_q8 * dx.
         * sum_q8 = sum of Q8_0 activations over the 32-element block.
         * Computed via warp shfl (same pattern as Q5_K/Q4_K). */
        int sq0 = 0, sq1 = 0;
        {
            int8_t *av;
            av = (int8_t *)&a0; sq0 += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a2; sq0 += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a1; sq1 += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a3; sq1 += av[0] + av[1] + av[2] + av[3];
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
        {
            unsigned mask = 0xF << (gid * 4);
            sq0 += gpuShflDownSync(mask, sq0, 2);
            sq1 += gpuShflDownSync(mask, sq1, 2);
            sq0 += gpuShflDownSync(mask, sq0, 1);
            sq1 += gpuShflDownSync(mask, sq1, 1);
            /* Broadcast reduced sums to all threads in the group */
            sq0 = gpuShflSync(mask, sq0, 0);
            sq1 = gpuShflSync(mask, sq1, 0);
        }
#endif

        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];
        int wc = tile_o + gid;
        const uint8_t *wb = w + (size_t)wc * row_bytes + (size_t)kb * GPU_BLOCK_Q4_1_SIZE;
        float wd = gpu_fp16_to_fp32(wb[0] | ((uint16_t)wb[1] << 8));
        float wm = gpu_fp16_to_fp32(wb[2] | ((uint16_t)wb[3] << 8));

        if (tid == 0) {
            sum[0] += wm * dx0 * (float)sq0;
            sum[2] += wm * dx1 * (float)sq1;
        }
        sum[0] += wd * dx0 * (float)d0;
        sum[1] += wd * dx0 * (float)d1;
        sum[2] += wd * dx1 * (float)d2;
        sum[3] += wd * dx1 * (float)d3;
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

/* Q2_0 x Q8_0 IMMA kernel.
 *
 * Q2_0: 34 bytes per 128 weights. 2-bit values (stored 0..3).
 * Layout: d[2] (fp16 scale) + qs[32] (packed 2-bit values).
 * Dequant: val = (qs_val - 1) * d, mapping {0,1,2,3} -> {-d, 0, +d, +2d}.
 * Each qs byte holds 4 values at bit positions [0..1], [2..3], [4..5], [6..7].
 * 128 values = 4 IMMA steps of 32 K-values each, all sharing one scale d.
 *
 * Per IMMA step (32 K-values = 8 qs bytes): thread tid reads one qs byte
 * for K[tid*4..tid*4+3] and one qs byte for K[tid*4+16..tid*4+19].
 *
 * Grid: [(O+7)/8, (S+15)/16], Block: 32 threads. Same as Q8_0 IMMA.
 */
__global__ void
picolm_q2_0_q8_matmul_imma(float *y, const int8_t *xq, const float *xd,
                            const void *weights, int S, int I, int O,
                            int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s = gpuBlockIdx_y * 16;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < n_blocks; kb++) {
        /* A: activation fragments (same as Q8_0) */
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

        /* B: unpack one Q2_0 qs byte per fragment.
         * q2_block = kb / 4. qs_off = (kb % 4) * 8 within qs[32].
         * tid reads byte qs_off+tid (K[0..3]) and qs_off+4+tid (K[16..19]). */
        int b0, b1;
        {
            int wc = tile_o + gid;
            int q2_block = kb / 4;
            int qs_off = (kb & 3) * 8;
            const uint8_t *qs = w + (size_t)wc * row_bytes +
                                (size_t)q2_block * GPU_BLOCK_Q2_0_SIZE + 2;
            uint32_t v0 = qs[qs_off + tid];
            uint32_t v1 = qs[qs_off + 4 + tid];

            uint32_t raw0 = (v0 & 3) | (((v0 >> 2) & 3) << 8)
                          | (((v0 >> 4) & 3) << 16) | ((v0 >> 6) << 24);
            uint32_t raw1 = (v1 & 3) | (((v1 >> 2) & 3) << 8)
                          | (((v1 >> 4) & 3) << 16) | ((v1 >> 6) << 24);
            { uint32_t v = raw0; b0 = ((v & 0xFF) - 1) | (((v >> 8) & 0xFF) - 1) << 8 | (((v >> 16) & 0xFF) - 1) << 16 | ((v >> 24) - 1) << 24; }
            { uint32_t v = raw1; b1 = ((v & 0xFF) - 1) | (((v >> 8) & 0xFF) - 1) << 8 | (((v >> 16) & 0xFF) - 1) << 16 | ((v >> 24) - 1) << 24; }
        }

        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif

        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];
        int wc = tile_o + gid;
        const uint8_t *wb = w + (size_t)wc * row_bytes +
                            (size_t)(kb / 4) * GPU_BLOCK_Q2_0_SIZE;
        float wd = gpu_fp16_to_fp32(wb[0] | ((uint16_t)wb[1] << 8));

        sum[0] += wd * dx0 * (float)d0;
        sum[1] += wd * dx0 * (float)d1;
        sum[2] += wd * dx1 * (float)d2;
        sum[3] += wd * dx1 * (float)d3;
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

/* Q1_0 x Q8_0 IMMA kernel.
 *
 * Q1_0: 18 bytes per 128 weights. 1-bit sign + scale.
 * Layout: d[2] (fp16 scale) + qs[16] (128 sign bits).
 * Dequant: val = (bit ? +d : -d).
 * Each qs byte holds 8 sign bits. 128 values = 4 IMMA steps of 32.
 *
 * Per IMMA step: 32 K-values = 4 qs bytes. Thread tid reads one qs byte
 * for K[tid*4..tid*4+3] (bits (tid%2)*4..(tid%2)*4+3 of byte qs_off+tid/2)
 * and one qs byte for K[tid*4+16..tid*4+19] (same bits of byte qs_off+2+tid/2).
 * Each sign bit is converted to {-1, +1}: bit=0 -> 0xFF, bit=1 -> 0x01.
 *
 * Grid: [(O+7)/8, (S+15)/16], Block: 32 threads. Same as Q8_0 IMMA.
 */
__global__ void
picolm_q1_0_q8_matmul_imma(float *y, const int8_t *xq, const float *xd,
                            const void *weights, int S, int I, int O,
                            int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s = gpuBlockIdx_y * 16;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < n_blocks; kb++) {
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

        int b0, b1;
        {
            int wc = tile_o + gid;
            int q1_block = kb / 4;
            int qs_off = (kb & 3) * 4;
            const uint8_t *qs = w + (size_t)wc * row_bytes +
                                (size_t)q1_block * GPU_BLOCK_Q1_0_SIZE + 2;
            uint32_t shift = (tid & 1) << 2;
            uint32_t raw0 = (qs[qs_off + (tid >> 1)] >> shift) & 0x0F;
            uint32_t raw1 = (qs[qs_off + 2 + (tid >> 1)] >> shift) & 0x0F;
            /* Convert 4 sign bits to {-1,+1} packed int32. */
            b0 = 0; b1 = 0;
            for (int i = 0; i < 4; i++) {
                b0 |= (int)(((raw0 >> i) & 1) ? 0xFF : 0x01) << (i * 8);
                b1 |= (int)(((raw1 >> i) & 1) ? 0xFF : 0x01) << (i * 8);
            }
        }

        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif

        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];
        int wc = tile_o + gid;
        const uint8_t *wb = w + (size_t)wc * row_bytes +
                            (size_t)(kb / 4) * GPU_BLOCK_Q1_0_SIZE;
        float wd = gpu_fp16_to_fp32(wb[0] | ((uint16_t)wb[1] << 8));

        sum[0] += wd * dx0 * (float)d0;
        sum[1] += wd * dx0 * (float)d1;
        sum[2] += wd * dx1 * (float)d2;
        sum[3] += wd * dx1 * (float)d3;
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

/* Q5_K x Q8_0 IMMA kernel.
 *
 * Q5_K: 176 bytes per 256 weights. 5-bit values (stored 0..31).
 * Layout: d[2] + dm[2] + scales[12] + qh[32] + qs[128].
 * 256 elements = 8 sub-blocks of 32. Each sub-block has one 6-bit scale
 * and one 6-bit min (packed in scales[12] via get_scale_min_k4).
 * Dequant: val = d * sc * qs5 - dm * mn, where qs5 = low4 | (high1 << 4).
 *
 * GPU activations are Q8_0 (32-block, no bsums). The -dm*mn term is
 * handled per-K-block: compute sum(q8[32]) per row via warp shfl,
 * then subtract dm*mn*sum_q8*dx from the partial sum.
 *
 * IMMA m16n8k32: 1 call per K-step (uniform scale per 32).
 * Grid: [(O+7)/8, (S+15)/16], Block: 32 threads. Same as Q8_0 IMMA.
 */
__global__ void
picolm_q5_k_q8_matmul_imma(float *y, const int8_t *xq, const float *xd,
                            const void *weights, int S, int I, int O,
                            int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s = gpuBlockIdx_y * 16;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < n_blocks; kb++) {
        int q5_block = kb / 8;
        int g = kb % 8;

        int wc = tile_o + gid;
        size_t wrow_base = (size_t)wc * row_bytes + q5_block * GPU_BLOCK_Q5_K_SIZE;
        const uint8_t *wb = w + wrow_base;

        /* Scales are read in epilogue from correct output columns (wc0,wc1),
         * not from gid column (which was only for B-fragment feeding). */

        /* A: activation fragments */
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

        /* B: unpack Q5_K 5-bit values to int32 (4 per register).
         * Block layout: d(0-1), dm(2-3), scales(4-15), qh(16-47), qs(48-175).
         * 4 groups of 64 values, each group has 2 sub-blocks of 32 (g=0..7).
         * group = g / 2 (0..3), sub = g % 2 (0 or 1).
         *
         * qs: 128 bytes at offset 48. Per group: 32 bytes.
         *   sub=0: low nibble of qs[group*32 + l], sub=1: high nibble.
         * qh: 32 bytes at offset 16. One byte per element position l (0..31).
         *   high bit extracted via mask = (sub ? (2<<(2*group)) : (1<<(2*group))).
         *   This means bit position shifts per group (bits 0-1 for g0, 2-3 for g1, etc.)
         *
         * Thread tid processes 4 elements: tid*4..tid*4+3 for b0, tid*4+16..tid*4+19 for b1. */
        int b0, b1;
        {
            int group = g >> 1;       /* 0..3 */
            int sub = g & 1;          /* 0 or 1 */

            /* qs base for this group: 32 bytes cover 64 values (2 sub-blocks) */
            const uint8_t *qs_base = wb + 48 + group * 32;
            /* qh is always at offset 16, 32 bytes total */
            const uint8_t *qh_base = wb + 16;

            /* b0: elements 0..3 */
            uint32_t b0_val = 0;
            int bit_shift = 2 * group + sub;  /* sub=0: bits 0,2,4,6; sub=1: bits 1,3,5,7 */
            for (int e = 0; e < 4; e++) {
                int l = tid * 4 + e;
                uint8_t qs_byte = qs_base[l];
                uint8_t nibble = sub ? (qs_byte >> 4) : (qs_byte & 0xF);
                uint8_t high = ((qh_base[l] >> bit_shift) & 1) ? 16 : 0;
                b0_val |= (uint32_t)(nibble + high) << (e * 8);
            }
            b0 = (int)b0_val;

            /* b1: elements 16..19 */
            uint32_t b1_val = 0;
            for (int e = 0; e < 4; e++) {
                int l = 16 + tid * 4 + e;
                uint8_t qs_byte = qs_base[l];
                uint8_t nibble = sub ? (qs_byte >> 4) : (qs_byte & 0xF);
                uint8_t high = ((qh_base[l] >> bit_shift) & 1) ? 16 : 0;
                b1_val |= (uint32_t)(nibble + high) << (e * 8);
            }
            b1 = (int)b1_val;
        }

        /* IMMA m16n8k32 */
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif

        /* Min-correction: compute sum(q8[32]) per row via shfl. */
        int sq0 = 0, sq1 = 0;
        {
            int8_t *av;
            av = (int8_t *)&a0; sq0 += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a2; sq0 += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a1; sq1 += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a3; sq1 += av[0] + av[1] + av[2] + av[3];
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
        {
            unsigned mask = 0xF << (gid * 4);
            sq0 += gpuShflDownSync(mask, sq0, 2);
            sq1 += gpuShflDownSync(mask, sq1, 2);
            sq0 += gpuShflDownSync(mask, sq0, 1);
            sq1 += gpuShflDownSync(mask, sq1, 1);
            /* Broadcast reduced sums to all threads in the group */
            sq0 = gpuShflSync(mask, sq0, 0);
            sq1 = gpuShflSync(mask, sq1, 0);
        }
#endif

        /* Epilogue: apply scales from CORRECT output columns (wc0,wc1),
         * not from gid column. Matches picolm_q8_q8_matmul_imma. */
        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];

        /* Read scales for output column wc0 = tile_o + tid*2 */
        {
            size_t wb0_off = (size_t)(tile_o + tid * 2) * row_bytes + q5_block * GPU_BLOCK_Q5_K_SIZE;
            const uint8_t *wb0 = w + wb0_off;
            float wd0 = gpu_fp16_to_fp32(wb0[0] | ((uint16_t)wb0[1] << 8));
            float wdm0 = gpu_fp16_to_fp32(wb0[2] | ((uint16_t)wb0[3] << 8));
            uint8_t sc0, mn0;
            { const uint8_t *s0 = wb0 + 4;
              if (g < 4) { sc0 = s0[g] & 63; mn0 = s0[g + 4] & 63; }
              else { sc0 = (uint8_t)((s0[g + 4] & 0xF) | ((s0[g - 4] >> 6) << 4));
                      mn0 = (uint8_t)((s0[g + 4] >> 4) | ((s0[g] >> 6) << 4)); } }
            float wds0 = wd0 * (float)sc0;
            float wdmn0 = wdm0 * (float)mn0;
            float corr0 = wdmn0 * dx0 * (float)sq0;
            sum[0] += wds0 * dx0 * (float)d0 - corr0;
            sum[2] += wds0 * dx1 * (float)d2 - wdmn0 * dx1 * (float)sq1;
        }
        /* Read scales for output column wc1 = tile_o + tid*2+1 */
        {
            size_t wb1_off = (size_t)(tile_o + tid * 2 + 1) * row_bytes + q5_block * GPU_BLOCK_Q5_K_SIZE;
            const uint8_t *wb1 = w + wb1_off;
            float wd1 = gpu_fp16_to_fp32(wb1[0] | ((uint16_t)wb1[1] << 8));
            float wdm1 = gpu_fp16_to_fp32(wb1[2] | ((uint16_t)wb1[3] << 8));
            uint8_t sc1, mn1;
            { const uint8_t *s1 = wb1 + 4;
              if (g < 4) { sc1 = s1[g] & 63; mn1 = s1[g + 4] & 63; }
              else { sc1 = (uint8_t)((s1[g + 4] & 0xF) | ((s1[g - 4] >> 6) << 4));
                      mn1 = (uint8_t)((s1[g + 4] >> 4) | ((s1[g] >> 6) << 4)); } }
            float wds1 = wd1 * (float)sc1;
            float wdmn1 = wdm1 * (float)mn1;
            float corr1 = wdmn1 * dx0 * (float)sq0;
            sum[1] += wds1 * dx0 * (float)d1 - corr1;
            sum[3] += wds1 * dx1 * (float)d3 - wdmn1 * dx1 * (float)sq1;
        }
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

/* Q4_K x Q8_0 IMMA kernel.
 * Q4_K: 256 values in 144 bytes. block_q4_K layout:
 *   d(2), dmin(2), scales[12], qs[128].
 *   8 sub-blocks of 32 values each. Each sub-block has its own 6-bit
 *   scale (sc) and 6-bit min (mn), packed via get_scale_min_k4.
 *   qs[128]: 4-bit nibbles. Sub-blocks 0,2,4,6 use low nibbles of
 *   qs[0..31], qs[32..63], qs[64..95], qs[96..127]. Sub-blocks
 *   1,3,5,7 use high nibbles of the same bytes.
 *   Dequant: val = d * sc * qs4 - dmin * mn.
 *
 * Same IMMA m16n8k32 pattern as Q5_K. 32 threads (1 warp), 16x8 tile.
 * Activation is pre-quantized to Q8_0 (int8 + per-32-element scale). */
__global__ void
picolm_q4_k_q8_matmul_imma(float *y, const int8_t *xq, const float *xd,
                            const void *weights, int S, int I, int O,
                            int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s = gpuBlockIdx_y * 16;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < n_blocks; kb++) {
        int q4_block = kb / 8;
        int g = kb % 8;  /* sub-block index 0..7 within the 256-value block */

        int wc = tile_o + gid;
        size_t wrow_base = (size_t)wc * row_bytes + q4_block * GPU_BLOCK_Q4_K_SIZE;
        const uint8_t *wb = w + wrow_base;

        /* Scales are read in epilogue from correct output columns (wc0,wc1),
         * not from gid column (which was only for B-fragment feeding). */

        /* A: activation fragments (same as Q5_K) */
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

        /* B: unpack Q4_K 4-bit nibbles to int32 (4 per register).
         * Block: d(0-1), dm(2-3), scales(4-15), qs(16-143).
         * qs starts at offset 16. 4 groups of 32 bytes, each group holds
         * 2 sub-blocks (low nibbles = sub-block 2*j, high nibbles = sub-block 2*j+1).
         *
         * g >> 1 selects the qs group (0..3). g & 1 selects low vs high nibble.
         * Within the group, each thread reads 4 bytes (one element per byte,
         * like picolm_q6_q8_matmul_imma). tid*4 selects the starting byte.
         *
         * For IMMA s8 format: nibbles are unsigned 0..15. The min-correction
         * subtracts dmin*mn*sum(q8) to handle the unsigned offset. */
        int b0, b1;
        {
            int qs_group = g >> 1;  /* 0..3 */
            const uint8_t *qs_p = wb + 16 + qs_group * 32;
            uint32_t raw0, raw1;
            /* b0: 4 elements at bytes tid*4 .. tid*4+3 */
            memcpy(&raw0, qs_p + tid * 4, 4);
            /* b1: 4 elements at bytes tid*4+16 .. tid*4+19 */
            memcpy(&raw1, qs_p + 16 + tid * 4, 4);

            if (g & 1) {
                /* High nibbles: extract independently per byte */
                raw0 = ((raw0 >> 4) & 0x0F) | (((raw0 >> 12) & 0x0F) << 8)
                     | (((raw0 >> 20) & 0x0F) << 16) | (((raw0 >> 28) & 0x0F) << 24);
                raw1 = ((raw1 >> 4) & 0x0F) | (((raw1 >> 12) & 0x0F) << 8)
                     | (((raw1 >> 20) & 0x0F) << 16) | (((raw1 >> 28) & 0x0F) << 24);
            } else {
                /* Low nibbles: mask with 0x0F */
                raw0 = raw0 & 0x0F0F0F0F;
                raw1 = raw1 & 0x0F0F0F0F;
            }
            b0 = raw0;
            b1 = raw1;
        }

        /* IMMA m16n8k32 */
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif

        /* Min-correction: same pattern as Q5_K */
        int sq0 = 0, sq1 = 0;
        {
            int8_t *av;
            av = (int8_t *)&a0; sq0 += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a2; sq0 += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a1; sq1 += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a3; sq1 += av[0] + av[1] + av[2] + av[3];
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
        {
            unsigned mask = 0xF << (gid * 4);
            sq0 += gpuShflDownSync(mask, sq0, 2);
            sq1 += gpuShflDownSync(mask, sq1, 2);
            sq0 += gpuShflDownSync(mask, sq0, 1);
            sq1 += gpuShflDownSync(mask, sq1, 1);
            /* Broadcast reduced sums to all threads in the group */
            sq0 = gpuShflSync(mask, sq0, 0);
            sq1 = gpuShflSync(mask, sq1, 0);
        }
#endif

        /* Epilogue: apply scales from CORRECT output columns (wc0,wc1),
         * not from gid column (which was only for B-fragment feeding).
         * Matches picolm_q8_q8_matmul_imma epilogue pattern. */
        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];

        /* Read scales for output column wc0 = tile_o + tid*2 */
        {
            size_t wb0_off = (size_t)(tile_o + tid * 2) * row_bytes + q4_block * GPU_BLOCK_Q4_K_SIZE;
            const uint8_t *wb0 = w + wb0_off;
            float wd0 = gpu_fp16_to_fp32(wb0[0] | ((uint16_t)wb0[1] << 8));
            float wdm0 = gpu_fp16_to_fp32(wb0[2] | ((uint16_t)wb0[3] << 8));
            uint8_t sc0, mn0;
            { const uint8_t *s0 = wb0 + 4;
              if (g < 4) { sc0 = s0[g] & 63; mn0 = s0[g + 4] & 63; }
              else { sc0 = (uint8_t)((s0[g + 4] & 0xF) | ((s0[g - 4] >> 6) << 4));
                      mn0 = (uint8_t)((s0[g + 4] >> 4) | ((s0[g] >> 6) << 4)); } }
            float wds0 = wd0 * (float)sc0;
            float wdmn0 = wdm0 * (float)mn0;
            float corr0 = wdmn0 * dx0 * (float)sq0;
            sum[0] += wds0 * dx0 * (float)d0 - corr0;
            sum[2] += wds0 * dx1 * (float)d2 - wdmn0 * dx1 * (float)sq1;
        }
        /* Read scales for output column wc1 = tile_o + tid*2+1 */
        {
            size_t wb1_off = (size_t)(tile_o + tid * 2 + 1) * row_bytes + q4_block * GPU_BLOCK_Q4_K_SIZE;
            const uint8_t *wb1 = w + wb1_off;
            float wd1 = gpu_fp16_to_fp32(wb1[0] | ((uint16_t)wb1[1] << 8));
            float wdm1 = gpu_fp16_to_fp32(wb1[2] | ((uint16_t)wb1[3] << 8));
            uint8_t sc1, mn1;
            { const uint8_t *s1 = wb1 + 4;
              if (g < 4) { sc1 = s1[g] & 63; mn1 = s1[g + 4] & 63; }
              else { sc1 = (uint8_t)((s1[g + 4] & 0xF) | ((s1[g - 4] >> 6) << 4));
                      mn1 = (uint8_t)((s1[g + 4] >> 4) | ((s1[g] >> 6) << 4)); } }
            float wds1 = wd1 * (float)sc1;
            float wdmn1 = wdm1 * (float)mn1;
            float corr1 = wdmn1 * dx0 * (float)sq0;
            sum[1] += wds1 * dx0 * (float)d1 - corr1;
            sum[3] += wds1 * dx1 * (float)d3 - wdmn1 * dx1 * (float)sq1;
        }
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

/* Q3_K x Q8_0 IMMA kernel.
 * Q3_K: 256 values in 110 bytes. block_q3_K layout:
 *   hmask[32] (0..31) + qs[64] (32..95) + scales[12] (96..107) + d[2] (108..109).
 *   16 sub-blocks of 16 values each. Each sub-block has a 6-bit scale.
 *   Value: 2-bit qs + 1-bit hmask -> 3-bit signed [-4..3].
 *   Dequant: val = d * (scale_i - 32) * q3_val.
 *
 * IMMA m16n8k32: two calls per K-step (like Q6_K).
 *   Call 1: b0=real(K[0..15]), b1=0
 *   Call 2: b0=0, b1=real(K[16..31])
 * Grid: [(O+7)/8, (S+15)/16], Block: 32 threads.
 * Activation is pre-quantized to Q8_0 (int8 + per-32-element scale). */
__global__ void
picolm_q3_k_q8_matmul_imma(float *y, const int8_t *xq, const float *xd,
                            const void *weights, int S, int I, int O,
                            int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s = gpuBlockIdx_y * 16;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < n_blocks; kb++) {
        /* Q3_K: 256 values = 16 sub-blocks of 16. Each IMMA step covers
         * 32 K-values = 2 sub-blocks of 16. Two IMMA calls per step:
         * call 1: b0=real(K[0..15]), b1=0; call 2: b0=0, b1=real(K[16..31]). */
        int q3_block = kb / 8;           /* which 256-value block */
        int g = (kb % 8) * 2;           /* even sub-block: 0,2,4,...,14 */

        int wc = tile_o + gid;
        size_t wrow_base = (size_t)wc * row_bytes + q3_block * GPU_BLOCK_Q3_K_SIZE;
        const uint8_t *wb = w + wrow_base;

        /* Scales are read in epilogue from correct output columns (wc0,wc1),
         * not from gid column (which was only for B-fragment feeding). */

        /* Q3_K layout: 2 chunks of 128 values. Each chunk: 8 sub-blocks of 16.
         * g -> chunk (0..1), group (0..3), sub (0..1). */
        int chunk = g / 8;
        int group = (g % 8) / 2;  /* 0..3 */
        int sub = g % 2;          /* 0 or 1 */

        int bit_shift = group * 2;    /* qs bit shift: 0, 2, 4, 6 (resets per chunk) */
        int hbit = 1 << (chunk * 4 + group);  /* hmask bit cycles 1,2,4,8,16,32,64,128 */

        const uint8_t *qs_base = wb + 32 + chunk * 32 + sub * 16;
        const uint8_t *hm_base = wb + sub * 16;  /* hmask is shared across both chunks */

        /* A: activation fragments */
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

        /* B: unpack Q3_K 3-bit values to int8, 4 per int32.
         * qs: 1 byte per value, 4 values packed as 2-bit each. Thread tid processes
         * elements tid*4..tid*4+3 within the sub-block.
         * hmask: 1 byte per value, extract bit corresponding to `group`. */

        int8_t qs4[4];
        for (int e = 0; e < 4; e++) {
            qs4[e] = (int8_t)((qs_base[tid * 4 + e] >> bit_shift) & 3);
        }

        uint8_t hm_bit[4];
        for (int e = 0; e < 4; e++) {
            hm_bit[e] = (hm_base[tid * 4 + e] & hbit) ? 1 : 0;
        }

        /* Combine: val = qs - (hmask_bit ? 0 : 4) -> [-4..3] */
        int8_t q3_4[4];
        for (int e = 0; e < 4; e++) {
            q3_4[e] = qs4[e] - (hm_bit[e] ? 0 : 4);
        }
        int b0;
        memcpy(&b0, q3_4, 4);

        /* IMMA call 1: b0 real, b1=0 */
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        int d0b = 0, d1b = 0, d2b = 0, d3b = 0;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(0));
#endif

        /* Paired sub-block g+1: same chunk, same group, toggled sub. */
        int sub2 = 1 - sub;
        const uint8_t *qs_base2 = wb + 32 + chunk * 32 + sub2 * 16;
        const uint8_t *hm_base2 = wb + sub2 * 16;  /* hmask shared across chunks */

        int b1;
        {
            int8_t q3_4_2[4];
            for (int e = 0; e < 4; e++) {
                int8_t qs_val = (int8_t)((qs_base2[tid * 4 + e] >> bit_shift) & 3);
                uint8_t hm_val = (hm_base2[tid * 4 + e] & hbit) ? 1 : 0;
                q3_4_2[e] = qs_val - (hm_val ? 0 : 4);
            }
            memcpy(&b1, q3_4_2, 4);
        }

        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0b), "+r"(d1b), "+r"(d2b), "+r"(d3b)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(0), "r"(b1));
#endif

        /* Epilogue: apply scales from CORRECT output columns (wc0,wc1),
         * not from gid column. Q3_K: 2 IMMA calls, 2 sub-blocks per step.
         * Unpack scales from 12-byte bit-interleaved format. */
        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];

        /* Read scales for output column wc0 = tile_o + tid*2 */
        {
            size_t wb0_off = (size_t)(tile_o + tid * 2) * row_bytes + q3_block * GPU_BLOCK_Q3_K_SIZE;
            const uint8_t *wb0 = w + wb0_off;
            float wd0 = gpu_fp16_to_fp32(wb0[108] | ((uint16_t)wb0[109] << 8));
            uint32_t aux0[4];
            memcpy(aux0, wb0 + 96, 12);
            { uint32_t tmp = aux0[2];
              aux0[2] = ((aux0[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
              aux0[3] = ((aux0[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
              aux0[0] = (aux0[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4);
              aux0[1] = (aux0[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4); }
            const int8_t *scales0 = (const int8_t *)aux0;
            float wds0 = wd0 * (float)(scales0[g] - 32);
            float wds1 = wd0 * (float)(scales0[g + 1] - 32);
            sum[0] += wds0 * dx0 * (float)d0 + wds1 * dx0 * (float)d0b;
            sum[2] += wds0 * dx1 * (float)d2 + wds1 * dx1 * (float)d2b;
        }
        /* Read scales for output column wc1 = tile_o + tid*2+1 */
        {
            size_t wb1_off = (size_t)(tile_o + tid * 2 + 1) * row_bytes + q3_block * GPU_BLOCK_Q3_K_SIZE;
            const uint8_t *wb1 = w + wb1_off;
            float wd1 = gpu_fp16_to_fp32(wb1[108] | ((uint16_t)wb1[109] << 8));
            uint32_t aux1[4];
            memcpy(aux1, wb1 + 96, 12);
            { uint32_t tmp = aux1[2];
              aux1[2] = ((aux1[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
              aux1[3] = ((aux1[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
              aux1[0] = (aux1[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4);
              aux1[1] = (aux1[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4); }
            const int8_t *scales1 = (const int8_t *)aux1;
            float wds0 = wd1 * (float)(scales1[g] - 32);
            float wds1 = wd1 * (float)(scales1[g + 1] - 32);
            sum[1] += wds0 * dx0 * (float)d1 + wds1 * dx0 * (float)d1b;
            sum[3] += wds0 * dx1 * (float)d3 + wds1 * dx1 * (float)d3b;
        }
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

/* Q2_K x Q8_0 IMMA kernel.
 *
 * Q2_K: 256 values in 84 bytes. block_q2_K layout:
 *   scales[16] (0..15) + qs[64] (16..79) + d(80-81) + dmin(82-83).
 *   16 sub-blocks of 16 values each. Each sub-block has a 4-bit scale
 *   and a 4-bit min.
 *   Value: 2-bit qs (unsigned 0..3).
 *   Dequant: val = d * scale * q - dmin * min.
 *
 * Layout per 256-value block:
 *   2 chunks of 128 values. Each chunk: 4 groups (bit-shift 0,2,4,6)
 *   x 2 sub-blocks of 16 values = 8 sub-blocks per chunk.
 *   Sub-block g (0..15): chunk = (g/4)/2, group = (g/2)%4, sub = g%2.
 *   qs offset: 16 + chunk*32 + sub*16, bit shift = group*2.
 *   scale = scales[g] & 0xF, min = scales[g] >> 4.
 *
 * IMMA m16n8k32: two calls per K-step (like Q3_K).
 *   Call 1: b0=real(K[0..15]), b1=0  (sub-block g)
 *   Call 2: b0=0, b1=real(K[16..31]) (sub-block g+1)
 *   Each call processes 16 weight values against 32 activation values.
 *
 * Min-correction: per-sub-block mins. Each sub-block's min is applied
 * only to its respective 16 activation values. We need half-sums:
 *   sq0_first = sum of a0's 4 int8 values x 4 threads (first 16 of row)
 *   sq0_second = sum of a2's 4 int8 values x 4 threads (second 16 of row)
 *
 * Grid: [(O+7)/8, (S+15)/16], Block: 32 threads.
 * Activation is pre-quantized to Q8_0 (int8 + per-32-element scale). */
__global__ void
picolm_q2_k_q8_matmul_imma(float *y, const int8_t *xq, const float *xd,
                            const void *weights, int S, int I, int O,
                            int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s = gpuBlockIdx_y * 16;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < n_blocks; kb++) {
        /* Q2_K: 256 values = 16 sub-blocks of 16. Each IMMA step covers
         * 32 K-values = 2 sub-blocks of 16. Two IMMA calls per step. */
        int q2_block = kb / 8;            /* which 256-value block */
        int g = (kb % 8) * 2;            /* even sub-block: 0,2,4,...,14 */

        int wc = tile_o + gid;
        size_t wrow_base = (size_t)wc * row_bytes + q2_block * GPU_BLOCK_Q2_K_SIZE;
        const uint8_t *wb = w + wrow_base;

        /* Scales are read in epilogue from correct output columns (wc0,wc1),
         * not from gid column (which was only for B-fragment feeding). */

        /* A: activation fragments */
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

        /* B: unpack Q2_K 2-bit values to int32 (4 per register).
         * qs: 64 bytes starting at offset 16. Per chunk: 32 bytes.
         * Per sub: 16 bytes. Each byte holds 4 values (2 bits each).
         * Thread tid processes elements tid*4..tid*4+3 within the sub-block.
         * g is the sub-block index (0,2,4,...,14). group=g/2 within chunk maps
         * to CPU's inner loop iteration (j=group/2, sub=group%2). */
        int chunk = g / 8;
        int group = (g % 8) / 2;
        int sub = g % 2;
        int bit_shift = group * 2;

        const uint8_t *qs_base = wb + 16 + chunk * 32 + sub * 16;
        int8_t q2_4[4];
        for (int e = 0; e < 4; e++) {
            q2_4[e] = (int8_t)((qs_base[tid * 4 + e] >> bit_shift) & 3);
        }
        int b0;
        memcpy(&b0, q2_4, 4);

        /* Paired sub-block g+1 */
        int sub2 = 1 - sub;
        const uint8_t *qs_base2 = wb + 16 + chunk * 32 + sub2 * 16;
        int8_t q2_4_2[4];
        for (int e = 0; e < 4; e++) {
            q2_4_2[e] = (int8_t)((qs_base2[tid * 4 + e] >> bit_shift) & 3);
        }
        int b1;
        memcpy(&b1, q2_4_2, 4);

        /* IMMA call 1: b0 real, b1=0 (sub-block g) */
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        int d0b = 0, d1b = 0, d2b = 0, d3b = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(0));
#endif
        /* IMMA call 2: b0=0, b1 real (sub-block g+1) */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0b), "+r"(d1b), "+r"(d2b), "+r"(d3b)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(0), "r"(b1));
#endif

        /* Min-correction: Q2_K formula is val = d*scale*q - dmin*min.
         *
         * IMMA fragment layout (m16n8k32, s8):
         *   a0: row=gid,     K cols 0..15 (first 16 activations)
         *   a1: row=gid+8,   K cols 0..15 (first 16 activations)
         *   a2: row=gid,     K cols 16..31 (second 16 activations)
         *   a3: row=gid+8,   K cols 16..31 (second 16 activations)
         *   b0: K rows 0..15 (sub-block g weights), col=groupID
         *   b1: K rows 16..31 (sub-block g+1 weights), col=groupID
         *
         * Call 1 (b0=real,b1=0): d0 = sum(first_16_act * q2_g)
         * Call 2 (b0=0,b1=real): d0b = sum(second_16_act * q2_g+1)
         *
         * Min-correction per sub-block:
         *   sub-block g:   dmin * mn_g   * sum(first_16_act)
         *   sub-block g+1: dmin * mn_g+1 * sum(second_16_act)
         *
         * We need SEPARATE activation sums for the first 16 and second 16.
         * a0/a1 contribute to the first-half sum, a2/a3 to the second-half sum. */
        int sq0a = 0, sq0b = 0;  /* row gid: first-16 / second-16 sums */
        int sq1a = 0, sq1b = 0;  /* row gid+8: first-16 / second-16 sums */
        {
            int8_t *av;
            av = (int8_t *)&a0; sq0a += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a2; sq0b += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a1; sq1a += av[0] + av[1] + av[2] + av[3];
            av = (int8_t *)&a3; sq1b += av[0] + av[1] + av[2] + av[3];
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
        {
            unsigned mask = 0xF << (gid * 4);
            sq0a += gpuShflDownSync(mask, sq0a, 2);
            sq0b += gpuShflDownSync(mask, sq0b, 2);
            sq1a += gpuShflDownSync(mask, sq1a, 2);
            sq1b += gpuShflDownSync(mask, sq1b, 2);
            sq0a += gpuShflDownSync(mask, sq0a, 1);
            sq0b += gpuShflDownSync(mask, sq0b, 1);
            sq1a += gpuShflDownSync(mask, sq1a, 1);
            sq1b += gpuShflDownSync(mask, sq1b, 1);
            /* Broadcast reduced sums to all threads in the group */
            sq0a = gpuShflSync(mask, sq0a, 0);
            sq0b = gpuShflSync(mask, sq0b, 0);
            sq1a = gpuShflSync(mask, sq1a, 0);
            sq1b = gpuShflSync(mask, sq1b, 0);
        }
#endif

        /* Epilogue: apply scales from CORRECT output columns (wc0,wc1),
         * not from gid column. Q2_K: 2 IMMA calls, 2 sub-blocks per step. */
        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];

        /* Read scales for output column wc0 = tile_o + tid*2 */
        {
            size_t wb0_off = (size_t)(tile_o + tid * 2) * row_bytes + q2_block * GPU_BLOCK_Q2_K_SIZE;
            const uint8_t *wb0 = w + wb0_off;
            float wd0 = gpu_fp16_to_fp32(wb0[80] | ((uint16_t)wb0[81] << 8));
            float wdm0 = gpu_fp16_to_fp32(wb0[82] | ((uint16_t)wb0[83] << 8));
            uint8_t sc0 = wb0[g] & 0xF;
            uint8_t mn0 = wb0[g] >> 4;
            uint8_t sc1 = wb0[g + 1] & 0xF;
            uint8_t mn1 = wb0[g + 1] >> 4;
            float wds0 = wd0 * (float)sc0;
            float wdmn0 = wdm0 * (float)mn0;
            float wds1 = wd0 * (float)sc1;
            float wdmn1 = wdm0 * (float)mn1;
            /* min-correction: mn_g * first_half_sum + mn_g+1 * second_half_sum */
            float corr0 = wdmn0 * dx0 * (float)sq0a + wdmn1 * dx0 * (float)sq0b;
            sum[0] += wds0 * dx0 * (float)d0 + wds1 * dx0 * (float)d0b - corr0;
            float corr1 = wdmn0 * dx1 * (float)sq1a + wdmn1 * dx1 * (float)sq1b;
            sum[2] += wds0 * dx1 * (float)d2 + wds1 * dx1 * (float)d2b - corr1;
        }
        /* Read scales for output column wc1 = tile_o + tid*2+1 */
        {
            size_t wb1_off = (size_t)(tile_o + tid * 2 + 1) * row_bytes + q2_block * GPU_BLOCK_Q2_K_SIZE;
            const uint8_t *wb1 = w + wb1_off;
            float wd1 = gpu_fp16_to_fp32(wb1[80] | ((uint16_t)wb1[81] << 8));
            float wdm1 = gpu_fp16_to_fp32(wb1[82] | ((uint16_t)wb1[83] << 8));
            uint8_t sc0 = wb1[g] & 0xF;
            uint8_t mn0 = wb1[g] >> 4;
            uint8_t sc1 = wb1[g + 1] & 0xF;
            uint8_t mn1 = wb1[g + 1] >> 4;
            float wds0 = wd1 * (float)sc0;
            float wdmn0 = wdm1 * (float)mn0;
            float wds1 = wd1 * (float)sc1;
            float wdmn1 = wdm1 * (float)mn1;
            float corr0 = wdmn0 * dx0 * (float)sq0a + wdmn1 * dx0 * (float)sq0b;
            sum[1] += wds0 * dx0 * (float)d1 + wds1 * dx0 * (float)d1b - corr0;
            float corr1 = wdmn0 * dx1 * (float)sq1a + wdmn1 * dx1 * (float)sq1b;
            sum[3] += wds0 * dx1 * (float)d3 + wds1 * dx1 * (float)d3b - corr1;
        }
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

/* TODO: decode kernel is thread-0-serial for the Q@K dot product.
 * Each of pos+1 KV positions: tid==0 computes head_dim FMAs alone,
 * syncthreads, then all threads distribute V-accumulation. At pos=2000,
 * head_dim=128 that's ~256k serial FMAs while 127 threads idle per block.
 * The split-kernel path (decode_split_kernel) partially mitigates via
 * cross-block parallelism, but within a single block the problem is the
 * same. A warp-shuffle reduction (same pattern as
 * picolm_gpu_attention_prefill_warpgrp_kernel) would be the direct fix.
 * Lower priority than prefill because decode slope is gentler (~40%
 * slowdown over 2900 tokens vs prefill's old 8x), but worth addressing. */
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
        int n_heads, int n_kv_heads, int head_dim,
        int tile_q)
{
    int h = (int)gpuBlockIdx_x;
    int tile_q_idx = (int)gpuBlockIdx_y;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    int kv_h = h / (n_heads / n_kv_heads);
    float attn_scale = 1.0f / sqrtf((float)head_dim);

    int q_start = tile_q_idx * tile_q;
    int q_end = min(q_start + tile_q, n_tokens);
    int n_q = q_end - q_start;

    /* Shared memory: K tile + V tile (FP32) + reduce + acc + max/sum */
    extern __shared__ uint8_t smem[];
    float *k_tile_f = (float *)smem;
    float *v_tile_f = k_tile_f + ATTN_TILE_K * head_dim;
    float *reduce_sh = v_tile_f + ATTN_TILE_K * head_dim;
    float *acc_sh = reduce_sh + 256;
    float *max_score_sh = acc_sh + (size_t)tile_q * head_dim;
    float *sum_exp_sh = max_score_sh + tile_q;
    __shared__ float rescale_sh, weight_sh;

    for (int i = tid; i < tile_q * head_dim; i += n_threads) acc_sh[i] = 0.0f;
    for (int qi = tid; qi < tile_q; qi += n_threads) {
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
        size_t kv_head_stride_bytes,
        int tile_q)
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
    int q_start = tile_q_idx * tile_q;
    int q_end = min(q_start + tile_q, n_tokens);
    int n_q = q_end - q_start;

    /* Shared memory: K tile + V tile (u16) + reduce (256 float) +
     * acc[tile_q][head_dim] (float) + max_score[tile_q] +
     * sum_exp[tile_q] (float). Sized by the host wrapper, which
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
    float *acc_sh = reduce_sh + 256;               /* [tile_q][head_dim] */
    float *max_score_sh = acc_sh + (size_t)tile_q * head_dim;  /* [tile_q] */
    float *sum_exp_sh = max_score_sh + tile_q; /* [tile_q] */
    __shared__ float rescale_sh, weight_sh;

    for (int i = tid; i < tile_q * head_dim; i += n_threads) acc_sh[i] = 0.0f;
    for (int qi = tid; qi < tile_q; qi += n_threads) {
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

/* ---- Broadcast lane 0's value to every lane in a `width`-wide subgroup.
 * Used below to replace the old kernel's "thread 0 decides -> shared
 * scalar -> syncthreads -> everyone reads" pattern with a register-only
 * equivalent. Uses raw __shfl_sync/__shfl (not the gpuShflSync macro from
 * backend_gpu_common.cuh) because we need the source-lane argument, not
 * the XOR/UP/DOWN variants. */
#if !defined(__HIP__)
__device__ inline float
gpu_shfl_bcast0(float var, int width) {
    return __shfl_sync(0xffffffff, var, 0, width);
}
#else
__device__ inline float
gpu_shfl_bcast0(float var, int width) {
    return __shfl(var, 0, width);
}
#endif

/* ---- Warp/wavefront-group scalar attention prefill ----
 *
 * Same algorithm, same tiling, same online-softmax as
 * picolm_gpu_attention_prefill_kernel above. The only change: instead of
 * one block of 128 threads cooperating on ONE query row at a time via a
 * full tree-reduce + syncthreads for every (query row, KV position) pair,
 * the block is split into ATTN_WARPGRP_SIZE-wide subgroups (4 groups of
 * 32 for a 128-thread block). Each subgroup owns a disjoint, fixed subset
 * of this tile's query rows for the kernel's entire duration and runs
 * fully independently: warp-shuffle reduction, no shared-memory tree, no
 * syncthreads except around the collective K/V tile load that all
 * subgroups read from.
 *
 * On a GPU with no tensor cores (gfx906/MI50 and similar), this is what
 * FA2 already gets NVIDIA for free via 4-warps-per-block parallelism --
 * here it's the same structural idea applied to the scalar dot product.
 *
 * Bit-exactness argument (why this matches the CPU AVX-512 reference
 * exactly, same as the block-wide kernel above):
 * The old kernel reduces head_dim/16 nonzero partial sums (n_chunks,
 * always <=16 since head_dim<=256) across a 128-wide shared-memory tree
 * with descending strides 64,32,16,8,4,2,1. Strides 64 and 32 are
 * UNCONDITIONALLY no-ops: at stride=64, every lane tid<64 adds
 * reduce_sh[tid+64], and tid+64>=64>16>=n_chunks is always zero (adding
 * exact 0.0 changes no bits); at stride=32 the same holds since
 * tid+32>=32>16>=n_chunks. So the real computation only ever starts at
 * stride=16, and strides 16,8,4,2,1 over a 128-wide zero-padded buffer
 * are data-flow-identical to the same five strides over a 32-wide
 * zero-padded buffer, which is exactly what a width-32 descending
 * shfl_down_sync reduction computes into lane 0. Every other lane's
 * value at the end is unused garbage, same as the old kernel only ever
 * reading reduce_sh[0]. Lane 0's value is then broadcast (not reduced --
 * a bit-copy, not a floating op) to the rest of its subgroup so all 32
 * lanes take the same online-softmax branch; the V-accumulation step is
 * a plain per-dimension fmaf update with no cross-lane summation, so
 * distributing it 32-wide instead of 128-wide changes nothing (each
 * output dimension is still touched by exactly one lane, in the same
 * across-KV-tile order, for the whole kernel).
 *
 * Default scalar path on both HIP and CUDA. The legacy block-wide-reduce
 * kernel is still available behind PICOLM_ATTN_SLOW_SCALAR=1. */
__global__ void
picolm_gpu_attention_prefill_warpgrp_kernel(
        float *xb_out,        /* [n_tokens][n_heads][head_dim] */
        const float *q_dev,   /* [n_tokens][n_heads][head_dim] */
        const uint16_t *kv_k, /* [layer][pos][kv_head][head_dim] FP16 */
        const uint16_t *kv_v, /* [layer][pos][kv_head][head_dim] FP16 */
        int layer_ordinal,
        int start_pos, int n_tokens,
        int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
        size_t kv_pos_stride_bytes,
        size_t kv_head_stride_bytes,
        int tile_q)
{
    const int GRP = ATTN_WARPGRP_SIZE;
    int h = (int)gpuBlockIdx_x;
    int tile_q_idx = (int)gpuBlockIdx_y;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;
    int n_groups = n_threads / GRP;
    int gid = tid / GRP;   /* which query-row subgroup this thread belongs to */
    int lane = tid % GRP;  /* lane within the subgroup, 0..GRP-1 */

    int kv_h = h / (n_heads / n_kv_heads);
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    size_t layer_base = (size_t)layer_ordinal * max_seq_len * n_kv_heads * head_dim;

    int q_start = tile_q_idx * tile_q;
    int q_end = min(q_start + tile_q, n_tokens);
    int n_q = q_end - q_start;

    /* Shared memory: K tile + V tile (u16) + acc[tile_q][head_dim] +
     * max_score[tile_q] + sum_exp[tile_q] (float). No reduce_sh buffer --
     * that 256-float block-wide scratch is gone, since reduction is now
     * entirely warp-local. Sized by attn_prefill_warpgrp_shared_bytes()
     * on the host side. */
    extern __shared__ uint8_t smem[];
    uint16_t *k_tile = (uint16_t *)smem;
    uint16_t *v_tile = k_tile + ATTN_TILE_K * head_dim;
    float *acc_sh = (float *)((uint8_t *)v_tile + ATTN_TILE_K * head_dim * sizeof(uint16_t));
    float *max_score_sh = acc_sh + (size_t)tile_q * head_dim;
    float *sum_exp_sh = max_score_sh + tile_q;

    for (int i = tid; i < tile_q * head_dim; i += n_threads) acc_sh[i] = 0.0f;
    for (int qi = tid; qi < tile_q; qi += n_threads) {
        max_score_sh[qi] = -1e30f;
        sum_exp_sh[qi] = 0.0f;
    }
    gpuSyncthreads();

    int block_kv_limit = min(start_pos + n_tokens, start_pos + q_end);
    int n_chunks = head_dim / 16; /* always <= 16, see bit-exactness note above */

    for (int t0 = 0; t0 < block_kv_limit; t0 += ATTN_TILE_K) {
        int t_end = min(t0 + ATTN_TILE_K, block_kv_limit);
        int tile_k_size = t_end - t0;

        /* Collective K/V tile load -- every thread in the block
         * participates regardless of subgroup. Vectorized to uint4
         * (8 half-precision elements, 16B) per transaction when the
         * layout allows it: this only changes how the bytes are
         * fetched, not their values, so it stays bit-exact with the
         * scalar path. Falls back to the original 2-byte-at-a-time
         * loop for any head_dim not a multiple of 8 (kept for safety,
         * not expected to trigger on head_dim in {64,96,128,256}).
         *
         * Alignment note: base_off (in uint16 units) is
         * layer_base + kv_h*(kv_head_stride_bytes/2) + pos*(kv_pos_stride_bytes/2).
         * Given the documented [layer][pos][kv_head][head_dim] layout,
         * kv_head_stride_bytes/2 == head_dim and kv_pos_stride_bytes/2
         * == n_kv_heads*head_dim, so every term is a multiple of
         * head_dim -- base_off is a multiple of 8 whenever head_dim is,
         * which is what uint4 (8-element) alignment needs. Device
         * allocations are at minimum 256B-aligned, so the base pointer
         * itself is never the constraint. */
        if ((head_dim & 7) == 0) {
            int vec_elems = head_dim >> 3;
            for (int ti = 0; ti < tile_k_size; ti++) {
                size_t base_off = layer_base + kv_h * kv_head_stride_bytes / 2
                    + (size_t)(t0 + ti) * kv_pos_stride_bytes / 2;
                const uint4 *k_src = (const uint4 *)(kv_k + base_off);
                const uint4 *v_src = (const uint4 *)(kv_v + base_off);
                uint4 *k_dst = (uint4 *)(k_tile + ti * head_dim);
                uint4 *v_dst = (uint4 *)(v_tile + ti * head_dim);
                for (int vd = tid; vd < vec_elems; vd += n_threads) {
                    k_dst[vd] = k_src[vd];
                    v_dst[vd] = v_src[vd];
                }
            }
        } else {
            for (int d = tid; d < head_dim; d += n_threads) {
                for (int ti = 0; ti < tile_k_size; ti++) {
                    size_t k_off = layer_base + kv_h * kv_head_stride_bytes / 2
                        + (size_t)(t0 + ti) * kv_pos_stride_bytes / 2 + d;
                    k_tile[ti * head_dim + d] = kv_k[k_off];
                    v_tile[ti * head_dim + d] = kv_v[k_off];
                }
            }
        }
        gpuSyncthreads(); /* only barrier in the whole KV-tile body */

        /* Each subgroup independently owns a fixed, disjoint rotation of
         * this tile's query rows. From here to the next tile's load,
         * subgroups never touch each other's acc_sh/max_score_sh/
         * sum_exp_sh slots, so no further syncthreads is needed. */
        for (int qi = gid; qi < n_q; qi += n_groups) {
            int global_q = q_start + qi;
            int global_pos = start_pos + global_q;
            const float *qg = q_dev + (size_t)(global_q * n_heads + h) * head_dim;
            float *accqi = acc_sh + (size_t)qi * head_dim;

            for (int ti = 0; ti < tile_k_size; ti++) {
                int global_kv = t0 + ti;
                if (global_kv > global_pos) continue;

                /* TODO: replace scalar fmaf+gpu_fp16_to_fp32 per-element with
                 * v_dot2_f32_f16 (1 instruction = 2 FP16 FMAs, FP32 accumulate).
                 * Available on gfx906, CDNA, RDNA2+. Would halve instruction count
                 * here and is the single biggest remaining win vs llama.cpp's
                 * fattn-tile kernel on MI50. Requires keeping K tile as half2
                 * in shared memory (not uint16_t) and Q as float2 pairs.
                 * Bit-exactness needs careful verification since the FMAs are
                 * paired differently (d[2k]*q[2k] + d[2k+1]*q[2k+1] in one
                 * instruction vs sequential fmaf). */
                float local_chunk = 0.0f;
                if (lane < n_chunks) {
                    for (int d = lane * 16; d < (lane + 1) * 16; d++) {
                        local_chunk = fmaf(qg[d], gpu_fp16_to_fp32(k_tile[ti * head_dim + d]), local_chunk);
                    }
                }
                float val = local_chunk;
                val += gpuShflDownSync(0xffffffff, val, 16, GRP);
                val += gpuShflDownSync(0xffffffff, val, 8, GRP);
                val += gpuShflDownSync(0xffffffff, val, 4, GRP);
                val += gpuShflDownSync(0xffffffff, val, 2, GRP);
                val += gpuShflDownSync(0xffffffff, val, 1, GRP);
                /* Only lane 0 now holds the true sum (see bit-exactness
                 * note); broadcast it so every lane in the subgroup takes
                 * the identical online-softmax branch below. */
                float score = gpu_shfl_bcast0(val * attn_scale, GRP);

                float old_max = max_score_sh[qi];
                float rescale, weight;
                if (score > old_max) {
                    rescale = expf(old_max - score);
                    weight = 1.0f;
                    if (lane == 0) {
                        sum_exp_sh[qi] = sum_exp_sh[qi] * rescale + 1.0f;
                        max_score_sh[qi] = score;
                    }
                } else {
                    rescale = 1.0f;
                    weight = expf(score - old_max);
                    if (lane == 0) sum_exp_sh[qi] += weight;
                }

                for (int d = lane; d < head_dim; d += GRP) {
                    if (weight == 1.0f) {
                        accqi[d] = fmaf(accqi[d], rescale, gpu_fp16_to_fp32(v_tile[ti * head_dim + d]));
                    } else {
                        accqi[d] = fmaf(weight, gpu_fp16_to_fp32(v_tile[ti * head_dim + d]), accqi[d]);
                    }
                }
            }
        }
        gpuSyncthreads(); /* before next tile overwrites k_tile/v_tile */
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

/* ---- Warp/wavefront-group scalar attention prefill, dot2 variant ----
 *
 * Same structure as picolm_gpu_attention_prefill_warpgrp_kernel above
 * (same subgroup-per-query-row scheme, same tiling, same online
 * softmax), but the per-element fmaf+gpu_fp16_to_fp32 score loop is
 * replaced with gpu_fp16_mad() -- 1 packed FP16x2 multiply + FP32
 * horizontal-add per two elements, matching llama.cpp's
 * v_dot2_f32_f16 usage on gfx906/CDNA/RDNA2+.
 *
 * NOT bit-exact with the CPU AVX-512 reference or with the fmaf
 * warpgrp kernel above: the two products in each pair are computed at
 * native FP16 multiply precision before being widened and summed in
 * FP32, and Q is downcast fp32->fp16 to form the packed operand (K
 * needs no conversion -- it's already stored as raw FP16 bits, so
 * reinterpreting adjacent uint16 pairs as half2 is free). This is a
 * real precision change, not just a reordering, so max_diff against
 * the CPU reference will be nonzero. Validate with PICOLM_LOGITS_DUMP
 * across a full 48-layer forward pass at realistic context lengths
 * before trusting it for anything beyond a benchmark: llama.cpp has
 * no CPU-bit-exact requirement to begin with, picolm does, and this
 * kernel deliberately trades that away for throughput on non-IMMA
 * GPUs. Gated behind PICOLM_ATTN_WARPGRP=2, off by default. Falls
 * back to the fmaf warpgrp kernel at compile time if FAST_FP16_AVAILABLE
 * isn't defined for the target.
 *
 * CAUTION: written and reasoned through without a GPU in this session,
 * same disclaimer as the kernels above it. */
#ifdef GPU_FP16_DOT2_AVAILABLE
__global__ void
picolm_gpu_attention_prefill_warpgrp_dot2_kernel(
        float *xb_out, const float *q_dev,
        const uint16_t *kv_k, const uint16_t *kv_v,
        int layer_ordinal,
        int start_pos, int n_tokens,
        int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
        size_t kv_pos_stride_bytes,
        size_t kv_head_stride_bytes,
        int tile_q)
{
    const int GRP = ATTN_WARPGRP_SIZE;
    int h = (int)gpuBlockIdx_x;
    int tile_q_idx = (int)gpuBlockIdx_y;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;
    int n_groups = n_threads / GRP;
    int gid = tid / GRP;
    int lane = tid % GRP;

    int kv_h = h / (n_heads / n_kv_heads);
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    size_t layer_base = (size_t)layer_ordinal * max_seq_len * n_kv_heads * head_dim;

    int q_start = tile_q_idx * tile_q;
    int q_end = min(q_start + tile_q, n_tokens);
    int n_q = q_end - q_start;

    extern __shared__ uint8_t smem[];
    /* Shared memory layout:
     *   Q tile: tile_q * head_dim * 2 bytes (FP16)
     *   K tile: ATTN_TILE_K * head_dim * 2 bytes
     *   V tile: ATTN_TILE_K * head_dim * 2 bytes
     *   acc:    tile_q * head_dim * 4 bytes
     *   max:    tile_q * 4 bytes
     *   sum:    tile_q * 4 bytes
     */
    uint16_t *q_tile = (uint16_t *)smem;
    uint16_t *k_tile = q_tile + (size_t)tile_q * head_dim;
    uint16_t *v_tile = k_tile + ATTN_TILE_K * head_dim;
    float *acc_sh = (float *)((uint8_t *)v_tile + ATTN_TILE_K * head_dim * sizeof(uint16_t));
    float *max_score_sh = acc_sh + (size_t)tile_q * head_dim;
    float *sum_exp_sh = max_score_sh + tile_q;

    /* Load Q tile from global FP32 into shared FP16. */
    for (int i = tid; i < tile_q * head_dim; i += n_threads) {
        int qi = i / head_dim;
        int d  = i % head_dim;
        int global_q = q_start + qi;
        const float *qg = q_dev + (size_t)(global_q * n_heads + h) * head_dim;
        q_tile[i] = gpu_fp32_to_fp16(qg[d]);
        acc_sh[i] = 0.0f;
    }
    for (int qi = tid; qi < tile_q; qi += n_threads) {
        max_score_sh[qi] = -1e30f;
        sum_exp_sh[qi] = 0.0f;
    }
    gpuSyncthreads();

    int block_kv_limit = min(start_pos + n_tokens, start_pos + q_end);
    int n_chunks = head_dim / 16; /* each chunk = 8 half2 pairs */

    for (int t0 = 0; t0 < block_kv_limit; t0 += ATTN_TILE_K) {
        int t_end = min(t0 + ATTN_TILE_K, block_kv_limit);
        int tile_k_size = t_end - t0;

        if ((head_dim & 7) == 0) {
            int vec_elems = head_dim >> 3;
            for (int ti = 0; ti < tile_k_size; ti++) {
                size_t base_off = layer_base + kv_h * kv_head_stride_bytes / 2
                    + (size_t)(t0 + ti) * kv_pos_stride_bytes / 2;
                const uint4 *k_src = (const uint4 *)(kv_k + base_off);
                const uint4 *v_src = (const uint4 *)(kv_v + base_off);
                uint4 *k_dst = (uint4 *)(k_tile + ti * head_dim);
                uint4 *v_dst = (uint4 *)(v_tile + ti * head_dim);
                for (int vd = tid; vd < vec_elems; vd += n_threads) {
                    k_dst[vd] = k_src[vd];
                    v_dst[vd] = v_src[vd];
                }
            }
        } else {
            for (int d = tid; d < head_dim; d += n_threads) {
                for (int ti = 0; ti < tile_k_size; ti++) {
                    size_t k_off = layer_base + kv_h * kv_head_stride_bytes / 2
                        + (size_t)(t0 + ti) * kv_pos_stride_bytes / 2 + d;
                    k_tile[ti * head_dim + d] = kv_k[k_off];
                    v_tile[ti * head_dim + d] = kv_v[k_off];
                }
            }
        }
        gpuSyncthreads();

        for (int qi = gid; qi < n_q; qi += n_groups) {
            int global_q = q_start + qi;
            int global_pos = start_pos + global_q;
            /* Read Q from shared memory (FP16) instead of global (FP32).
             * Eliminates __floats2half2_rn and global memory traffic per
             * KV position. */
            const half2 *q2_base = (const half2 *)&q_tile[(size_t)qi * head_dim];
            float *accqi = acc_sh + (size_t)qi * head_dim;

            for (int ti = 0; ti < tile_k_size; ti++) {
                int global_kv = t0 + ti;
                if (global_kv > global_pos) continue;

                /* 8 packed FMAs instead of 16 scalar ones. Both Q and K
                 * are read as half2 directly from shared memory FP16.
                 * Uses gpu_fp16_dot2 (v_dot2_f32_f16, 1 ISA instruction
                 * per pair) when available, falls back to gpu_fp16_mad. */
                float local_chunk = 0.0f;
                if (lane < n_chunks) {
                    const half2 *q2 = &q2_base[lane * 8];
                    const half2 *k2 = (const half2 *)&k_tile[ti * head_dim + lane * 16];
                    for (int p = 0; p < 8; p++) {
#ifdef GPU_FP16_DOT2_AVAILABLE
                        gpu_fp16_dot2(local_chunk, q2[p], k2[p]);
#else
                        gpu_fp16_mad(local_chunk, q2[p], k2[p]);
#endif
                    }
                }
                float val = local_chunk;
                val += gpuShflDownSync(0xffffffff, val, 16, GRP);
                val += gpuShflDownSync(0xffffffff, val, 8, GRP);
                val += gpuShflDownSync(0xffffffff, val, 4, GRP);
                val += gpuShflDownSync(0xffffffff, val, 2, GRP);
                val += gpuShflDownSync(0xffffffff, val, 1, GRP);
                float score = gpu_shfl_bcast0(val * attn_scale, GRP);

                float old_max = max_score_sh[qi];
                float rescale, weight;
                if (score > old_max) {
                    rescale = expf(old_max - score);
                    weight = 1.0f;
                    if (lane == 0) {
                        sum_exp_sh[qi] = sum_exp_sh[qi] * rescale + 1.0f;
                        max_score_sh[qi] = score;
                    }
                } else {
                    rescale = 1.0f;
                    weight = expf(score - old_max);
                    if (lane == 0) sum_exp_sh[qi] += weight;
                }

                for (int d = lane; d < head_dim; d += GRP) {
                    if (weight == 1.0f) {
                        accqi[d] = fmaf(accqi[d], rescale, gpu_fp16_to_fp32(v_tile[ti * head_dim + d]));
                    } else {
                        accqi[d] = fmaf(weight, gpu_fp16_to_fp32(v_tile[ti * head_dim + d]), accqi[d]);
                    }
                }
            }
        }
        gpuSyncthreads();
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
#endif /* GPU_FP16_DOT2_AVAILABLE */

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

/* FA2_TILE_Q and FA2_TILE_K are defined in backend_gpu_kernels.cuh. */

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
        for (int i = 0; i < dim; i++) sum += dequant_bf16(wrow, i) * x[i];
        break;
    }
    case 41: { /* Q1_0 -- 128 values per block, sign bits.
                * val[j] = (bit[j] ? +d : -d) = d * (2*bit[j] - 1)
                * sum += d * (signed_sum_of_bits) * x[j] */
        int n_blocks = dim / 128;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 18;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);
            const uint8_t *qs = blk + 2; /* 16 bytes = 128 sign bits */
            for (int j = 0; j < 128; j++) {
                int bit = (qs[j >> 3] >> (j & 7)) & 1;
                int sign = (bit ? 1 : -1);
                sum += wd * sign * x[bi * 128 + j];
            }
        }
        break;
    }
    case 14: { /* Q6_K -- 256 values per block, 210 bytes */
        int n_blocks = dim / 256;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * GPU_BLOCK_Q6_K_SIZE;
            const uint8_t *ql = blk;
            const uint8_t *qh = blk + 128;
            const int8_t *scales = (const int8_t *)(qh + 64);
            uint16_t d_raw = blk[208] | ((uint16_t)blk[209] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);

            for (int half = 0; half < 2; half++) {
                const uint8_t *ql_c = ql + half * 64;
                const uint8_t *qh_c = qh + half * 32;
                for (int sub_chunk = 0; sub_chunk < 4; sub_chunk++) {
                    for (int sc_half = 0; sc_half < 2; sc_half++) {
                        int sc_idx = half * 8 + sub_chunk * 2 + sc_half;
                        float sc_v = (float)scales[sc_idx];
                        int sub_l_base = sc_half * 16;
                        for (int sub_l = 0; sub_l < 32; sub_l++) {
                            uint8_t ql_val, qh_val;
                            if (sub_chunk == 0) {
                                ql_val = ql_c[sub_l] & 0xF;
                                qh_val = (qh_c[sub_l] >> 0) & 3;
                            } else if (sub_chunk == 1) {
                                ql_val = ql_c[sub_l + 32] & 0xF;
                                qh_val = (qh_c[sub_l] >> 2) & 3;
                            } else if (sub_chunk == 2) {
                                ql_val = ql_c[sub_l] >> 4;
                                qh_val = (qh_c[sub_l] >> 4) & 3;
                            } else {
                                ql_val = ql_c[sub_l + 32] >> 4;
                                qh_val = (qh_c[sub_l] >> 6) & 3;
                            }
                            int q = (ql_val | (qh_val << 4)) - 32;
                            sum += wd * sc_v * (float)q * x[bi * 256 + half * 128 + sub_chunk * 32 + sub_l_base + sub_l];
                        }
                    }
                }
            }
        }
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
        for (int i = 0; i < dim; i++) sum += dequant_bf16(wrow, i) * xt[i];
        break;
    }
    case 41: { /* Q1_0 -- 128 values per block, sign bits */
        int n_blocks = dim / 128;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 18;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);
            const uint8_t *qs = blk + 2; /* 16 bytes = 128 sign bits */
            for (int j = 0; j < 128; j++) {
                int bit = (qs[j >> 3] >> (j & 7)) & 1;
                int sign = (bit ? 1 : -1);
                sum += wd * sign * xt[bi * 128 + j];
            }
        }
        break;
    }
    case 14: { /* Q6_K -- 256 values per block, 210 bytes */
        int n_blocks = dim / 256;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * GPU_BLOCK_Q6_K_SIZE;
            const uint8_t *ql = blk;
            const uint8_t *qh = blk + 128;
            const int8_t *scales = (const int8_t *)(qh + 64);
            uint16_t d_raw = blk[208] | ((uint16_t)blk[209] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);

            for (int half = 0; half < 2; half++) {
                const uint8_t *ql_c = ql + half * 64;
                const uint8_t *qh_c = qh + half * 32;
                for (int sub_chunk = 0; sub_chunk < 4; sub_chunk++) {
                    for (int sc_half = 0; sc_half < 2; sc_half++) {
                        int sc_idx = half * 8 + sub_chunk * 2 + sc_half;
                        float sc_v = (float)scales[sc_idx];
                        int sub_l_base = sc_half * 16;
                        for (int sub_l = 0; sub_l < 32; sub_l++) {
                            uint8_t ql_val, qh_val;
                            if (sub_chunk == 0) {
                                ql_val = ql_c[sub_l] & 0xF;
                                qh_val = (qh_c[sub_l] >> 0) & 3;
                            } else if (sub_chunk == 1) {
                                ql_val = ql_c[sub_l + 32] & 0xF;
                                qh_val = (qh_c[sub_l] >> 2) & 3;
                            } else if (sub_chunk == 2) {
                                ql_val = ql_c[sub_l] >> 4;
                                qh_val = (qh_c[sub_l] >> 4) & 3;
                            } else {
                                ql_val = ql_c[sub_l + 32] >> 4;
                                qh_val = (qh_c[sub_l] >> 6) & 3;
                            }
                            int q = (ql_val | (qh_val << 4)) - 32;
                            sum += wd * sc_v * (float)q * xt[bi * 256 + half * 128 + sub_chunk * 32 + sub_l_base + sub_l];
                        }
                    }
                }
            }
        }
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
