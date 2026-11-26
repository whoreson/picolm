// backend_gpu_kernels.cu - GPU kernel definitions (moved from kernels.cuh)
// This file is compiled once to avoid ODR violations when kernels.cuh
// is included by multiple host .cu files.
#include "backend_gpu_kernels.cuh"

/* Model-specified attention-logit scale (e.g. granite.attention.scale = 1/128,
 * mistral3.attention.temperature_scale). Set once at model load via
 * picolm_gpu_set_attn_scale(). 0.0f = use the default 1/sqrt(head_dim).
 *
 * Device code is linked per-translation-unit (no relocatable device code),
 * so this variable MUST be defined in the same TU as the kernels that read
 * it, and picolm_gpu_set_attn_scale() (below) writes to this very copy. */
__device__ float picolm_gpu_d_attn_scale = 0.0f;

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
        case 141:            bytes_per_block = GPU_BLOCK_IQ6_K_SIZE; break; /* IQ6_K: 212 */
        case 41:             bytes_per_block = 18; break;      /* Q1_0 */
        case 42:             bytes_per_block = 34; break;      /* Q2_0 */
        case GGUF_TYPE_IQ4_NL: bytes_per_block = GPU_BLOCK_Q4_0_SIZE; break;  /* IQ4_NL: 18 */
        case 133:              bytes_per_block = 26; break;                      /* Q6_0: 26 */
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

    case 20: /* GGUF_TYPE_IQ4_NL */
        /* I values in 32-value blocks (18 bytes each, same layout as Q4_0) */
        {
            int n_blocks = I / 32;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 32; j++) {
                    int i = bi * 32 + j;
                    sum += x[rs*s+i] * dequant_iq4_nl(blk, j);
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

    case 141: /* GGUF_TYPE_IQ6_K -- non-linear 6-bit, 212 bytes per 256 values.
               * This case is what makes IQ6_K work at S>1: the decode GEMV only
               * covers S==1, and without a per-element dequant here prefill and
               * every S>1 matmul fell into the `default:` branch below, which
               * ZEROES the output and poisons the whole run. */
        {
            int n_blocks = I / 256;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 256; j++) {
                    int i = bi * 256 + j;
                    sum += x[rs*s+i] * dequant_iq6_K(blk, j);
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

    case 133: /* GGUF_TYPE_Q6_0 */
        /* 32 values per block, 26 bytes */
        {
            int n_blocks = I / 32;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 32; j++) {
                    int i = bi * 32 + j;
                    sum += x[rs*s+i] * dequant_q6_0(blk, j);
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
    if (!gpuThreadIdx_x) {
        float val = (float)partial[0];
        y[(size_t)s * ys + o] = val;
    }
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
    float id = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    int q = (int)lroundf(v * id);
    /* Clamp to [-127, 127] to match CPU quantize_row_q8_0 */
    if (q > 127) q = 127;
    if (q < -127) q = -127;

    qs_out[(size_t)s * (size_t)n_blocks * 32 + (size_t)block * 32 + tid] = (int8_t)q;
    if (tid == 0)
        /* ROUND d TO FP16, because that is what it is.
         *
         * `d` is the block_q8_0 scale, and in the GGUF format that field is
         * an fp16. The CPU's quantize_row_q8_0 therefore stores
         * fp32_to_fp16(maxS/127.0f), while this kernel was storing the full
         * fp32 value into its separate float buffer. Measured with
         * --gpu-q8diff: the int8 payloads agree EXACTLY (0/5120 differ) but
         * 159/160 blocks had a different d, mean |dd| 9.5e-6.
         *
         * That is not extra precision, it is a DIFFERENT FUNCTION: every one
         * of the 32 values in the block is scaled by a slightly different
         * number, in every matmul of every layer. Tiny per token, but it
         * accumulates over the residual stream, and at 9848 tokens of context
         * GPU and CPU prefill disagree by max|logit diff| 11.4 with different
         * argmaxes. Applying the same fp16 rounding makes the two paths
         * compute the same thing at the source instead of drifting apart.
         */
        d_out[(size_t)s * (size_t)n_blocks + block] =
            gpu_fp16_to_fp32((int)gpu_fp32_to_fp16(d));
}

/* UNUSED AT PRESENT (see the note at the end of this comment).
 *
 * picolm_quantize_q8_0_fast -- same result as picolm_quantize_q8_0, done by a
 * WARP per 32-value block instead of a 32-thread BLOCK per 32-value block.
 *
 * Why this exists, and it is the whole prefill story:
 *   The scalar version allocates one CUDA block per 32 values and reduces the
 *   abs-max through __shared__ with FIVE gpuSyncthreads() per block. For the
 *   27B at S=423 that is a grid of (160, 423) = 67680 blocks, each 32 threads,
 *   each with 5 block-wide barriers -- 338400 barriers to quantise 2.17 MB.
 *   Measured: 688 ms for 248 calls, i.e. ~1.6 GB/s, 0.2% of the card.
 *
 *   That cost was previously attributed to the q5_K matmul because the
 *   profiler's slot wraps picolm_gpu_matmul_dev(), which includes the
 *   quantize. Proof: building the matmul with an EMPTY inner loop left the
 *   q5_K slot at 688 ms while prefill speed did not move.
 *
 * This version: Q8Q_WARPS warps per block, one warp per 32 values, warp
 * shuffle reduction (no shared memory, no barriers), and grid 8x smaller.
 * Same arithmetic and the same fp16 rounding of d, so the output is
 * bit-identical -- verified with --gpu-q8diff and PICOLM_DUMP_LOGITS.
 *
 * MEASURED, AND IT DID NOT HELP: once the matmul was properly isolated it
 * turned out the quantise was never the bottleneck. Fast vs old on the 27B at
 * 423 tokens: 228/238/231 vs 228/236/237 tok/s, i.e. nothing. The 688 ms that
 * had been attributed to "the q5_K slot including the quantise" is the q5_K
 * matmul itself. Kept rather than deleted because it is strictly better code
 * (no barriers, 1/8 the blocks) and will matter once the real fix lands, but
 * it is currently unreferenced -- do not assume wiring it up gains anything.
 */
#define Q8Q_WARPS 8

__global__ void
picolm_quantize_q8_0_fast(int8_t *qs_out, float *d_out,
                           const float *x, int I, int S) {
    int n_blocks = (I + 31) >> 5;
    int wid  = (int)(gpuThreadIdx_x >> 5);          /* warp within block  */
    int lane = (int)(gpuThreadIdx_x & 31);

    /* Flatten (row, block) across the whole grid so there is no 2-D launch
     * limit and every warp gets exactly one 32-value block. */
    long long gid = (long long)gpuBlockIdx_x * Q8Q_WARPS + wid;
    long long total = (long long)S * n_blocks;
    if (gid >= total) return;

    int s = (int)(gid / n_blocks);
    int blk = (int)(gid % n_blocks);
    const float *xb = x + (size_t)s * (size_t)I + (size_t)blk * 32;

    float v = xb[lane];
    float amax = fabsf(v);
    /* butterfly max across the warp -- no shared memory, no barriers */
    for (int off = 16; off > 0; off >>= 1) {
        float o = gpuShflXorSync(0xffffffffu, amax, off);
        amax = fmaxf(amax, o);
    }

    float d  = amax / 127.0f;
    float id = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    int q = (int)lroundf(v * id);
    if (q > 127) q = 127;
    if (q < -127) q = -127;

    qs_out[(size_t)s * (size_t)n_blocks * 32 + (size_t)blk * 32 + lane] = (int8_t)q;
    if (lane == 0)
        d_out[(size_t)s * (size_t)n_blocks + blk] =
            gpu_fp16_to_fp32((int)gpu_fp32_to_fp16(d));
}

/* Strided variant: the activation row is x[s*x_stride ...], same otherwise. */
__global__ void
picolm_quantize_q8_0_fast_strided(int8_t *qs_out, float *d_out,
                                   const float *x, int I, int S, int x_stride) {
    int n_blocks = (I + 31) >> 5;
    int wid  = (int)(gpuThreadIdx_x >> 5);
    int lane = (int)(gpuThreadIdx_x & 31);

    long long gid = (long long)gpuBlockIdx_x * Q8Q_WARPS + wid;
    long long total = (long long)S * n_blocks;
    if (gid >= total) return;

    int s = (int)(gid / n_blocks);
    int blk = (int)(gid % n_blocks);
    const float *xb = x + (size_t)s * (size_t)x_stride + (size_t)blk * 32;

    float v = xb[lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) {
        float o = gpuShflXorSync(0xffffffffu, amax, off);
        amax = fmaxf(amax, o);
    }

    float d  = amax / 127.0f;
    float id = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    int q = (int)lroundf(v * id);
    if (q > 127) q = 127;
    if (q < -127) q = -127;

    qs_out[(size_t)s * (size_t)n_blocks * 32 + (size_t)blk * 32 + lane] = (int8_t)q;
    if (lane == 0)
        d_out[(size_t)s * (size_t)n_blocks + blk] =
            gpu_fp16_to_fp32((int)gpu_fp32_to_fp16(d));
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
    float id = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    int q = (int)lroundf(v * id);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
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



/* ================================================================
 * Q8_K activation quantiser (device)
 * ================================================================
 *
 * The CPU uses Q8_K (256-value super-blocks) as the activation format for
 * every K-quant weight (gguf_type_act_fmt() in quant.h), because the K-quant
 * vec_dot_*_q8_K kernels are written for per-sub-block scales. The GPU had
 * only a Q8_0 quantiser, so K-quant weights were multiplied by 32-value-block
 * activations instead -- the documented CPU/GPU drift measured in
 * cuda-first-survey.txt (fp32 `d` vs fp16 `d`, not a corruption). This
 * kernel removes that difference at the source.
 *
 * Byte-for-byte identical to quantize_row_q8_K() in quant.c:
 *   d      = amax / 127                     (stored as float, NOT fp16)
 *   qs[j]  = round_to_nearest(x[j] * 127/amax)
 *   bsums[k] = sum(qs[k*16 .. k*16+15])
 * and the all-zero super-block (amax == 0) gets d=0, qs=0, bsums=0 so no
 * inf/NaN can appear downstream.
 *
 * Layout per block: 292 bytes = float d + int8 qs[256] + int16 bsums[16].
 * One CUDA block per super-block; 256 threads, one value each, absmax via a
 * shared-memory tree reduce. Output buffers therefore need
 * (I/256)*292 bytes per row, not (I/32)*34.
 */
/* GPU_BLOCK_Q8_K_SIZE is defined in backend_gpu_kernels.cuh. */
__global__ void
picolm_quantize_q8_K(float *d_out, int8_t *qs_out, int16_t *bsums_out,
                      const float *x, int I, int S) {
    int s = (int)gpuBlockIdx_y;          /* row */
    int sb = (int)gpuBlockIdx_x;         /* super-block within row */
    int tid = (int)gpuThreadIdx_x;       /* 0..255 */
    int n_super = I / 256;
    if (sb >= n_super || s >= S) return;

    const float *xb = x + (size_t)s * I + (size_t)sb * 256;
    float v = xb[tid];

    __shared__ float amax_sh[256];
    amax_sh[tid] = fabsf(v);
    gpuSyncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = amax_sh[tid], b = amax_sh[tid + stride];
            if (b > a) amax_sh[tid] = b;
        }
        gpuSyncthreads();
    }
    float amax = amax_sh[0];

    size_t base = (size_t)s * n_super + sb;
    float  *d_out_b    = d_out    + base;
    int8_t *qs_out_b   = qs_out   + base * 256;
    int16_t *bsums_b   = bsums_out + base * 16;

    if (amax == 0.0f) {
        /* Match the CPU's explicit zero-fill: avoids 0/0 -> NaN below. */
        qs_out_b[tid] = 0;
        if (tid < 16) bsums_b[tid] = 0;
        if (tid == 0) *d_out_b = 0.0f;
        return;
    }

    /* CPU: id = 127/amax; d = 1/id. Kept in that order so the float `d` is
     * bit-identical, not merely close. */
    float id = 127.0f / amax;
    if (tid == 0) *d_out_b = 1.0f / id;

    /* CPU scalar path: (int)(x*id + (x>=0 ? 0.5f : -0.5f)) -- a truncating
     * bias, which is round-half-away-from-zero. rintf() is round-half-even,
     * so it would differ on exact .5 boundaries; use the same trick. */
    int q = (int)(v * id + (v >= 0.0f ? 0.5f : -0.5f));
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    qs_out_b[tid] = (int8_t)q;

    /* bsums: one per group of 16 values. Threads 0..15 each own a group and
     * can read the qs values they just wrote (same block, after a barrier). */
    gpuSyncthreads();
    if (tid < 16) {
        int acc = 0;
        for (int j = 0; j < 16; j++) acc += (int)qs_out_b[tid * 16 + j];
        bsums_b[tid] = (int16_t)acc;
    }
}

/* ================================================================
 * Quantised GEMV -- the decode (S=1) fast path
 * ================================================================
 *
 * Why this exists: at S=1 the IMMA path still launches a 16-row tensor-core
 * tile, so 15 of 16 rows do useless work on zero padding and one 32-thread
 * block serves just 8 output columns. Tensor cores are the wrong tool for
 * GEMV by construction (a 16x8x32 mma for 1 useful row IS the bug). The right
 * shape is: stream each weight row exactly once, coalesced, and reduce.
 *
 * SHAPE (taken from llama.cpp's k_mul_mat_vec_q, which saturates the memory
 * bus where our first attempt did not -- 761 GB/s vs 319 GB/s measured on a
 * 4090 with Fimbulvetr-11B-Q8_0):
 *
 *   - ALL GEMV_WARPS warps of the block cooperate on ONE output row.
 *   - Each thread handles GEMV_VDR ints (8 bytes) of a block per step, so
 *     GEMV_QI/GEMV_VDR threads cover one block together.
 *   - Consecutive blocks are handed to consecutive thread groups, so one
 *     iteration issues GEMV_BLOCKS_PER_ITER independent block loads: that is
 *     the memory-level parallelism the first version lacked (it walked one
 *     block at a time per lane with a serial 8-step inner loop).
 *   - Per-block the raw int32 dot is accumulated and only scaled at the END
 *     by d8_0*d8_1, instead of a double multiply per block.
 *
 * Block layout is GGUF Q8_0: uint16 d + int8 qs[32] = 34 bytes, 4-byte
 * aligned in the qs array only (hence the b2-style 2x16-bit reads below --
 * reading an int directly from &qs[k] would be a misaligned 32-bit load).
 *
 * Weights are read through gpu_dot4_i8(): DP4A on NVIDIA sm_61+, SDOT on
 * gfx906+, scalar fallback that is bit-identical. Note the earlier measurement
 * that DP4A vs scalar changed nothing (28.9 vs 28.5 tok/s) -- this kernel is
 * bandwidth-bound, so the load pattern is what matters, not the ISA.
 */
/* 32-bit read of two 16-bit halves. The qs arrays are 2-byte aligned but not
 * necessarily 4-byte aligned, so this is the portable form of an int load. */
__device__ static inline int gemv_get_int_b2(const int8_t *qs, int i32) {
    const uint16_t *x16 = (const uint16_t *)qs;
    int x32  = (int)(x16[2 * i32 + 0] << 0);
    x32     |= (int)(x16[2 * i32 + 1] << 16);
    return x32;
}

__global__ void
picolm_q8_0_gemv(float *y,
                 const int8_t *xq, const float *xd,
                 const void *weights,
                 int I, int O, int row_bytes, int y_stride) {
    const int lane = (int)(gpuThreadIdx_x & 31);
    const int warp = (int)(gpuThreadIdx_x >> 5);
    const int tid  = warp * 32 + lane;          /* 0 .. 32*GEMV_WARPS-1 */
    const int o    = (int)gpuBlockIdx_x;        /* one row per block */
    if (o >= O) return;
    int ys = y_stride > 0 ? y_stride : O;

    const int n_blocks = I / 32;
    const uint8_t *wrow = (const uint8_t *)weights + (size_t)o * row_bytes;

    /* Which block this thread starts on, and which int pair within it. */
    const int kbx0 = tid / GEMV_THREADS_PER_BLK;
    const int kqs  = GEMV_VDR * (tid % GEMV_THREADS_PER_BLK);

    /* Raw int8 dot products, scaled at the very end. float is enough here:
     * llama.cpp uses float for the same accumulator and this matches its
     * numerics, which is the reference we are trying to close the gap to. */
    float sumf = 0.0f;

    for (int kbx = kbx0; kbx < n_blocks; kbx += GEMV_BLOCKS_PER_ITER) {
        const uint8_t *b = wrow + (size_t)kbx * GPU_BLOCK_Q8_0_SIZE;
        const uint16_t wd_raw = (uint16_t)(b[0] | ((uint16_t)b[1] << 8));
        const float wd = gpu_fp16_to_fp32(wd_raw);
        const float xdv = xd[kbx];

        const int8_t *wq = (const int8_t *)(b + 2);
        const int8_t *xqp = xq + (size_t)kbx * 32;

        /* GEMV_VDR ints of this block, i.e. 8 bytes, contiguous and aligned
         * to the thread's group offset. */
        int acc = 0;
        for (int i = 0; i < GEMV_VDR; i++)
            acc = gpu_dot4_i8_packed(gemv_get_int_b2(wq, kqs + i),
                                     gemv_get_int_b2(xqp, kqs + i), acc);

        sumf += wd * xdv * (float)acc;
    }

    /* Cross-warp reduction. Every warp holds a partial over its own blocks;
     * warps 1..N-1 stash theirs in shared memory and warp 0 finishes. */
    __shared__ float red[GEMV_WARPS > 1 ? GEMV_WARPS - 1 : 1][32];
    if (warp > 0) red[warp - 1][lane] = sumf;
    gpuSyncthreads();
    if (warp > 0) return;

    for (int w = 0; w < GEMV_WARPS - 1; w++)
        sumf += red[w][lane];

    /* Warp butterfly: every lane ends with the block total. */
    for (int off = 16; off > 0; off >>= 1)
        sumf += gpuShflXorSync(0xFFFFFFFFu, sumf, off);

    if (lane == 0) y[o] = sumf;
}

/* ================================================================
 * K-quant GEMV (decode, S=1) against Q8_K activations
 * ================================================================
 *
 * Same warp-per-row shape as picolm_q8_0_gemv, but for weights whose natural
 * activation format is Q8_K (gguf_type_act_fmt(): every K-quant weight). Using
 * Q8_0 activations against K-quant weights was the documented CPU/GPU drift;
 * these kernels make the GPU consume the same activation format as the CPU.
 *
 * Both bodies are direct transcriptions of the SCALAR reference paths in
 * quant.c (vec_dot_q5_K_q8_K / vec_dot_q6_K_q8_K, "Scalar fallback" sections),
 * because those are the unambiguous specification -- the AVX2 paths are
 * vectorised restatements of the same arithmetic. Keeping the structure
 * identical means a future SIMD version can be diffed against these.
 *
 * Work split: one 256-value super-block is covered by
 * GEMVK_THREADS_PER_SB (8) threads, each taking 32 values, so a warp covers
 * 4 super-blocks per step and consecutive lanes read consecutive bytes.
 */
#define GEMVK_THREADS_PER_SB  8
#define GEMVK_BLOCKS_PER_ITER (GEMV_WARPS * 32 / GEMVK_THREADS_PER_SB)  /* 16 */

/* Unpack the 6-bit scale/min pairs of a Q5_K super-block into 8 scales
 * followed by 8 mins, exactly like the utmp shuffling in quant.c's
 * vec_dot_q5_K_q8_K. */
/* Materialise only the ONE (scale, min) pair this thread needs. The previous
 * version built all 16 bytes on every thread and then used two of them, so
 * with 8 threads per super-block the shuffle ran 8x redundantly. On Q3_K,
 * where 16 threads each used 1 of 16, removing that redundancy took the
 * kernel from 77 to ~220 tok/s -- it was the dominant cost, not the dot
 * product. out[0] = scale[sub], out[1] = min[sub]. */
__device__ static inline void gemv_q5k_scale_min(const uint8_t *packed, int sub,
                                                  uint8_t *out /* 2 */) {
    uint32_t utmp[4];
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;
    memcpy(utmp, packed, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    { const uint32_t uaux = utmp[1] & kmask1;
      utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
      utmp[2] = uaux; }
    utmp[0] &= kmask1;
    /* bytes 0..7 = 8 scales, bytes 8..15 = 8 mins */
    const uint8_t *sc = (const uint8_t *)(const void *)&utmp[0];
    out[0] = sc[sub];
    out[1] = sc[8 + sub];
}

__global__ void
picolm_q5_K_gemv(float *y,
                 const int8_t *xq, const float *xd, const int16_t *xbsums,
                 const void *weights,
                 int I, int O, int row_bytes, int y_stride) {
    const int lane = (int)(gpuThreadIdx_x & 31);
    const int warp = (int)(gpuThreadIdx_x >> 5);
    const int tid  = warp * 32 + lane;
    const int o    = (int)gpuBlockIdx_x;
    if (o >= O) return;
    int ys = y_stride > 0 ? y_stride : O;

    const int n_super = I / 256;
    const uint8_t *wrow = (const uint8_t *)weights + (size_t)o * row_bytes;

    const int sb0 = tid / GEMVK_THREADS_PER_SB;   /* first super-block */
    const int sub = tid % GEMVK_THREADS_PER_SB;   /* which 32 of the 256 */

    float sumf = 0.0f;

    for (int sb = sb0; sb < n_super; sb += GEMVK_BLOCKS_PER_ITER) {
        const uint8_t *wb = wrow + (size_t)sb * GPU_BLOCK_Q5_K_SIZE;
        const float wd   = gpu_fp16_to_fp32((uint16_t)(wb[0] | ((uint16_t)wb[1] << 8)));
        const float wdm  = gpu_fp16_to_fp32((uint16_t)(wb[2] | ((uint16_t)wb[3] << 8)));
        const float d    = xd[sb] * wd;
        const float dmin = xd[sb] * wdm;

        uint8_t sc2[2];
        gemv_q5k_scale_min(wb + 4, sub, sc2);

        const uint8_t *qh = wb + 16;
        const uint8_t *qs = wb + 48;
        const int8_t  *q8 = xq + (size_t)sb * 256;

        const int group = sub >> 1;     /* 0..3: selects a 32-byte qs window */
        const int half  = sub & 1;      /* 0/1: low or high nibble           */
        /* The reference walks `m` as 1,2,4,...,128 -- one bit per 32-value
         * group, in order. So the 5th bit for sub-block `sub` is simply
         * bit `sub`, NOT (half ? 2 : 1) << (2*group) (which collides across
         * groups and left group 3 reading bit 8..15 of "qh[l]" -- qh is only
         * 8 bytes logically but the array is 32, so it silently read garbage).
         * This was the Q5_K bug. */
        const uint8_t bitmask = (uint8_t)(1u << sub);

        /* Unpack 32 five-bit values, then dot them 4 at a time through
         * gpu_dot4_i8_packed. The naive version multiplied per element with a
         * branch on `half` inside the loop and measured 398 GB/s -- less than
         * half what the Q8_0 GEMV reaches with the same shape (878 GB/s),
         * because the unpack dominated. The nibble selection is now hoisted
         * out of the loop (it depends only on `sub`) and the multiply is a
         * single DP4A/SDOT per 4 values.
         *
         * Values are 0..31, which does not fit signed int8 semantics for the
         * high bit: the 5th bit sets 0x10, so 16..31 would read as negative if
         * interpreted as int8. The reference treats them as UNSIGNED and
         * multiplies by a signed q8, so subtract 16 from every weight value
         * and add back 16*sum(q8) afterwards -- that keeps the arithmetic in
         * int8 range and is exactly what the +16*bias term below cancels. */
        /* Activation (q8) is read 8 bytes at a time via gemv_get_int_b2, the
         * SAME load the Q8_0 GEMV uses. Reading it as four separate byte
         * accesses measured 30% of this kernel's runtime: an ablation that
         * dropped the q8 read entirely took 0.0907 -> 0.0636 ms/call, landing
         * on Q8_0's 0.0564, while dropping the qh unpack saved only 7%. The
         * traffic is identical between the two kernels (same I, S=1), so it is
         * the 4-byte strided access pattern, not the volume. */
        int acc = 0;
        for (int l = 0; l < 32; l += 8) {
            /* 8 weight values: bytes l..l+7 of this group's window. */
            const uint8_t p0 = qs[group * 32 + l + 0], p1 = qs[group * 32 + l + 1];
            const uint8_t p2 = qs[group * 32 + l + 2], p3 = qs[group * 32 + l + 3];
            const uint8_t p4 = qs[group * 32 + l + 4], p5 = qs[group * 32 + l + 5];
            const uint8_t p6 = qs[group * 32 + l + 6], p7 = qs[group * 32 + l + 7];
            const uint8_t h0 = qh[l+0], h1 = qh[l+1], h2 = qh[l+2], h3 = qh[l+3];
            const uint8_t h4 = qh[l+4], h5 = qh[l+5], h6 = qh[l+6], h7 = qh[l+7];

            /* Two 4-value groups, packed into ints for DP4A/SDOT. */
            int8_t wA[4], wB[4];
            wA[0] = (int8_t)(((half ? (p0 >> 4) : (p0 & 0xF)) | ((h0 & bitmask) ? 0x10 : 0)) - 16);
            wA[1] = (int8_t)(((half ? (p1 >> 4) : (p1 & 0xF)) | ((h1 & bitmask) ? 0x10 : 0)) - 16);
            wA[2] = (int8_t)(((half ? (p2 >> 4) : (p2 & 0xF)) | ((h2 & bitmask) ? 0x10 : 0)) - 16);
            wA[3] = (int8_t)(((half ? (p3 >> 4) : (p3 & 0xF)) | ((h3 & bitmask) ? 0x10 : 0)) - 16);
            wB[0] = (int8_t)(((half ? (p4 >> 4) : (p4 & 0xF)) | ((h4 & bitmask) ? 0x10 : 0)) - 16);
            wB[1] = (int8_t)(((half ? (p5 >> 4) : (p5 & 0xF)) | ((h5 & bitmask) ? 0x10 : 0)) - 16);
            wB[2] = (int8_t)(((half ? (p6 >> 4) : (p6 & 0xF)) | ((h6 & bitmask) ? 0x10 : 0)) - 16);
            wB[3] = (int8_t)(((half ? (p7 >> 4) : (p7 & 0xF)) | ((h7 & bitmask) ? 0x10 : 0)) - 16);

            const int8_t *qa = q8 + (size_t)sub * 32 + l;
            /* Two 32-bit activation loads (bytes l..l+3 and l+4..l+7). */
            const int aA = gemv_get_int_b2(qa, 0);
            const int aB = gemv_get_int_b2(qa, 1);

            const int8_t *aAp = (const int8_t *)&aA;
            const int8_t *aBp = (const int8_t *)&aB;
            int sumA = (int)aAp[0]+(int)aAp[1]+(int)aAp[2]+(int)aAp[3];
            int sumB = (int)aBp[0]+(int)aBp[1]+(int)aBp[2]+(int)aBp[3];

            int wAi, wBi;
            memcpy(&wAi, wA, 4); memcpy(&wBi, wB, 4);
            acc = gpu_dot4_i8_packed(wAi, aA, acc);
            acc = gpu_dot4_i8_packed(wBi, aB, acc);
            acc += 16 * (sumA + sumB);          /* undo the -16 shift */
        }

        /* scales[is++] runs sequentially with j (0..7) == our `sub`.
         * The reference bias loop is  sumi += bsums[j] * mins[j/2]  for
         * j in 0..15, i.e. mins[m] pairs with bsums[2m] and bsums[2m+1].
         * Sub-block `sub` owns exactly those two bsums, so its min index is
         * `sub`, not sub/2. */
        const int scale = (int)sc2[0];
        const int minv  = (int)sc2[1];
        sumf += scale * d * (float)acc;
        /* Bias correction over this 32-value sub-block = two 16-value bsums. */
        const int bsum = (int)xbsums[sb * 16 + sub * 2]
                       + (int)xbsums[sb * 16 + sub * 2 + 1];
        sumf -= dmin * (float)minv * (float)bsum;
    }

    __shared__ float red[GEMV_WARPS > 1 ? GEMV_WARPS - 1 : 1][32];
    if (warp > 0) red[warp - 1][lane] = sumf;
    gpuSyncthreads();
    if (warp > 0) return;
    for (int w = 0; w < GEMV_WARPS - 1; w++) sumf += red[w][lane];
    for (int off = 16; off > 0; off >>= 1)
        sumf += gpuShflXorSync(0xFFFFFFFFu, sumf, off);
    if (lane == 0) y[o] = sumf;
}

/* Q6_K: same shape. Transcribed from the "Scalar fallback" in
 * vec_dot_q6_K_q8_K. Layout: ql[128], qh[64], scales[16] (int8), d (fp16). */
__global__ void
picolm_q6_K_gemv(float *y,
                 const int8_t *xq, const float *xd, const int16_t *xbsums,
                 const void *weights,
                 int I, int O, int row_bytes, int y_stride) {
    const int lane = (int)(gpuThreadIdx_x & 31);
    const int warp = (int)(gpuThreadIdx_x >> 5);
    const int tid  = warp * 32 + lane;
    const int o    = (int)gpuBlockIdx_x;
    if (o >= O) return;
    int ys = y_stride > 0 ? y_stride : O;

    const int n_super = I / 256;
    const uint8_t *wrow = (const uint8_t *)weights + (size_t)o * row_bytes;

    const int sb0 = tid / GEMVK_THREADS_PER_SB;
    const int sub = tid % GEMVK_THREADS_PER_SB;

    float sumf = 0.0f;

    for (int sb = sb0; sb < n_super; sb += GEMVK_BLOCKS_PER_ITER) {
        const uint8_t *wb = wrow + (size_t)sb * GPU_BLOCK_Q6_K_SIZE;
        const uint8_t *ql = wb;
        const uint8_t *qh = wb + 128;
        const int8_t  *sc = (const int8_t *)(wb + 192);
        const uint16_t d_raw = (uint16_t)(wb[208] | ((uint16_t)wb[209] << 8));
        const float d = xd[sb] * gpu_fp16_to_fp32(d_raw);

        const int8_t *q8 = xq + (size_t)sb * 256;

        /* A 128-value chunk holds 4 interleaved sub-blocks. Per the reference
         * (quant.c vec_dot_q6_K_q8_K scalar path), sub-block `which` sums
         * q8_c[q8off + 0..15] for l=0..15 and q8_c[q8off + 16..31] for
         * l=16..31 -- i.e. the two halves of the 32 q8 values are taken from
         * the SAME ql/qh byte range but different l, so l must run the full
         * 0..31 here while the byte index wraps at 16. Getting this wrong
         * silently mis-associates half of every sub-block. */
        const int chunk = sub >> 2;          /* 0 or 1 */
        const int which = sub & 3;           /* 0..3 -> q1..q4 */
        const uint8_t *ql_c = ql + chunk * 64;
        const uint8_t *qh_c = qh + chunk * 32;
        const int8_t  *q8_c = q8 + chunk * 128;

        /* IMPORTANT: a 32-value sub-block spans TWO reference entries.
         * sums[is+2*which] accumulates l=0..15 and sums[is+2*which+1]
         * accumulates l=16..31, each with its OWN scale sc[is+2*which] and
         * sc[is+2*which+1] (and its own bsums entry). Treating the 32 values
         * as one block with a single scale was the bug: it produced plausible
         * magnitudes but a wrong answer for 8 of every 16 scale slots. */
        const int j0 = chunk * 8 + 2 * which;      /* first of the pair */
        const int j1 = j0 + 1;
        int acc0 = 0, acc1 = 0;
        /* Everything that depends only on `which` is hoisted out, so each of
         * the four cases becomes a straight-line 16-iteration loop with no
         * branch and no switch -- same shape as the reference, which has two
         * separate 16-iteration loops rather than one 32-iteration loop with a
         * predicate. The previous version switched on `which` and tested
         * `l < 16` on EVERY iteration; measured 448 GB/s against Q8_0's 839.
         *
         * NOTE: the byte index is `l`, not `l & 15`, and the q8 index runs the
         * full l=0..31 for both halves. Folding the halves with a mask
         * re-reads the first 16 bytes of every sub-block. */
        {
            const int  ql_off  = (which & 1) ? 32 : 0;   /* q1/q3 vs q2/q4 */
            const int  ql_sh   = (which & 2) ? 4 : 0;    /* high vs low nibble */
            const int  qh_sh   = which * 2;              /* 0,2,4,6 */
            const int  q8_off  = which * 32;             /* 0,32,64,96 */
            const uint8_t *qlp = ql_c + ql_off;
            const int8_t  *q8p = q8_c + q8_off;

            /* l=0..15 -> acc0, l=16..31 -> acc1, both indexing qlp[l] and
             * qh_c[l] with the FULL l (see the note above). */
            for (int l = 0; l < 16; l++) {
                const int q = ((qlp[l] >> ql_sh) & 0xF)
                            | (((qh_c[l] >> qh_sh) & 3) << 4);
                acc0 += q * q8p[l];
            }
            for (int l = 16; l < 32; l++) {
                const int q = ((qlp[l] >> ql_sh) & 0xF)
                            | (((qh_c[l] >> qh_sh) & 3) << 4);
                acc1 += q * q8p[l];
            }
        }

        /* Reference: block_sum += sc[j]*sums[j] - 32*bsums[j]*sc[j] */
        const int s0 = (int)sc[j0], s1 = (int)sc[j1];
        const int b0 = (int)xbsums[sb * 16 + j0];
        const int b1 = (int)xbsums[sb * 16 + j1];
        acc0 = acc0 * s0 - 32 * b0 * s0;
        acc1 = acc1 * s1 - 32 * b1 * s1;
        sumf += d * (float)(acc0 + acc1);
    }

    __shared__ float red[GEMV_WARPS > 1 ? GEMV_WARPS - 1 : 1][32];
    if (warp > 0) red[warp - 1][lane] = sumf;
    gpuSyncthreads();
    if (warp > 0) return;
    for (int w = 0; w < GEMV_WARPS - 1; w++) sumf += red[w][lane];
    for (int off = 16; off > 0; off >>= 1)
        sumf += gpuShflXorSync(0xFFFFFFFFu, sumf, off);
    if (lane == 0) y[o] = sumf;
}

/* Q4_K: 256 weights in 144 bytes -- d(2) dmin(2) scales[12] qs[128].
 * Same shape as Q5_K minus the qh array: for j=0..3 the bytes qs[j*32..j*32+31]
 * hold sub-block j*2 in their low nibbles and j*2+1 in their high nibbles.
 * Scales/mins use the same 6-bit utmp packing as Q5_K. */
__global__ void
picolm_q4_K_gemv(float *y,
                 const int8_t *xq, const float *xd, const int16_t *xbsums,
                 const void *weights,
                 int I, int O, int row_bytes, int y_stride) {
    const int lane = (int)(gpuThreadIdx_x & 31);
    const int warp = (int)(gpuThreadIdx_x >> 5);
    const int tid  = warp * 32 + lane;
    const int o    = (int)gpuBlockIdx_x;
    if (o >= O) return;
    int ys = y_stride > 0 ? y_stride : O;

    const int n_super = I / 256;
    const uint8_t *wrow = (const uint8_t *)weights + (size_t)o * row_bytes;

    const int sb0 = tid / GEMVK_THREADS_PER_SB;
    const int sub = tid % GEMVK_THREADS_PER_SB;

    float sumf = 0.0f;

    for (int sb = sb0; sb < n_super; sb += GEMVK_BLOCKS_PER_ITER) {
        const uint8_t *wb = wrow + (size_t)sb * GPU_BLOCK_Q4_K_SIZE;
        const float wd   = gpu_fp16_to_fp32((uint16_t)(wb[0] | ((uint16_t)wb[1] << 8)));
        const float wdm  = gpu_fp16_to_fp32((uint16_t)(wb[2] | ((uint16_t)wb[3] << 8)));
        const float d    = xd[sb] * wd;
        const float dmin = xd[sb] * wdm;

        uint8_t sc2[2];
        gemv_q5k_scale_min(wb + 4, sub, sc2);   /* same 6-bit packing */

        const uint8_t *qs = wb + 16;
        const int8_t  *q8 = xq + (size_t)sb * 256;

        const int group = sub >> 1;
        const int half  = sub & 1;
        const uint8_t *qsg = qs + group * 32;

        /* 4-bit values 0..15 fit signed int8 directly (no bias needed, unlike
         * Q5_K's 5-bit). Activations read 4 bytes at a time via
         * gemv_get_int_b2-style packing -- reading them as separate bytes was
         * what cost Q5_K 30% before that fix. */
        int acc = 0;
        for (int l = 0; l < 32; l += 8) {
            int8_t wA[4], wB[4];
            wA[0] = (int8_t)(half ? (qsg[l+0] >> 4) : (qsg[l+0] & 0xF));
            wA[1] = (int8_t)(half ? (qsg[l+1] >> 4) : (qsg[l+1] & 0xF));
            wA[2] = (int8_t)(half ? (qsg[l+2] >> 4) : (qsg[l+2] & 0xF));
            wA[3] = (int8_t)(half ? (qsg[l+3] >> 4) : (qsg[l+3] & 0xF));
            wB[0] = (int8_t)(half ? (qsg[l+4] >> 4) : (qsg[l+4] & 0xF));
            wB[1] = (int8_t)(half ? (qsg[l+5] >> 4) : (qsg[l+5] & 0xF));
            wB[2] = (int8_t)(half ? (qsg[l+6] >> 4) : (qsg[l+6] & 0xF));
            wB[3] = (int8_t)(half ? (qsg[l+7] >> 4) : (qsg[l+7] & 0xF));

            const int8_t *qa = q8 + (size_t)sub * 32 + l;
            const int aA = gemv_get_int_b2(qa, 0);
            const int aB = gemv_get_int_b2(qa, 1);
            int wAi, wBi;
            memcpy(&wAi, wA, 4); memcpy(&wBi, wB, 4);
            acc = gpu_dot4_i8_packed(wAi, aA, acc);
            acc = gpu_dot4_i8_packed(wBi, aB, acc);
        }

        const int scale = (int)sc2[0];
        const int minv  = (int)sc2[1];
        const int bsum  = (int)xbsums[sb * 16 + sub * 2]
                        + (int)xbsums[sb * 16 + sub * 2 + 1];
        sumf += scale * d * (float)acc;
        sumf -= dmin * (float)minv * (float)bsum;
    }

    __shared__ float red[GEMV_WARPS > 1 ? GEMV_WARPS - 1 : 1][32];
    if (warp > 0) red[warp - 1][lane] = sumf;
    gpuSyncthreads();
    if (warp > 0) return;
    for (int w = 0; w < GEMV_WARPS - 1; w++) sumf += red[w][lane];
    for (int off = 16; off > 0; off >>= 1)
        sumf += gpuShflXorSync(0xFFFFFFFFu, sumf, off);
    if (lane == 0) y[o] = sumf;
}

/* Q3_K: 256 weights in 110 bytes -- hmask[32] qs[64] scales[12] d(2).
 *
 * Transcribed from dequantize_row_q3_K() in quant.c, which is the readable
 * authority (Q3_K has NO scalar fallback in vec_dot_q3_K_q8_K):
 *
 *   - 16 six-bit scales from the 12 bytes via a 3-byte shuffle, each biased
 *     by -32 (signed, and there are no mins -- so no bias-correction term).
 *   - TWO chunks of 128 values. Per chunk, qs advances 32 bytes and the bit
 *     shift walks 0,2,4,6, giving 8 groups of 16 values each.
 *   - A group's value is (2-bit field) - (hmask bit clear ? 4 : 0), i.e.
 *     signed 3-bit -4..3. The hmask bit doubles every 16-value group.
 *
 * So a sub-block is 16 values (not 32 as in Q4_K/Q5_K): 16 sub-blocks per
 * super-block, 16 scales. GEMVK_THREADS_PER_SB is therefore 16 here, one
 * thread per sub-block, each reading 16 values and 16 q8 entries.
 */
#define Q3K_SUBS_PER_SB 16

/* Only the ONE scale this thread uses is materialised: the previous version
 * built all 16 into a local array on every thread, so each of the 16 threads
 * per super-block did the full shuffle and then used one entry. Four ALU ops
 * for the shuffle are cheaper than the array stores + loop that followed. */
__device__ static inline int gemv_q3k_scale(const uint8_t *b, int s) {
    uint32_t aux[4];
    memcpy(aux, b, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x03030303u) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 6) & 0x03030303u) << 4);
    aux[0] = (aux[0] & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x03030303u) << 4);
    aux[1] = (aux[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
    const int8_t *sc = (const int8_t *)(const void *)aux;
    return (int)sc[s] - 32;
}

__global__ void
picolm_q3_K_gemv(float *y,
                 const int8_t *xq, const float *xd, const int16_t *xbsums,
                 const void *weights,
                 int I, int O, int row_bytes, int y_stride) {
    (void)xbsums;   /* no mins -> no bias term */
    const int lane = (int)(gpuThreadIdx_x & 31);
    const int warp = (int)(gpuThreadIdx_x >> 5);
    const int tid  = warp * 32 + lane;
    const int o    = (int)gpuBlockIdx_x;
    if (o >= O) return;

    const int n_super = I / 256;
    const uint8_t *wrow = (const uint8_t *)weights + (size_t)o * row_bytes;

    /* 16 threads per super-block (one per 16-value sub-block); a 128-thread
     * block therefore covers 8 super-blocks per step. */
    const int sb0 = tid / Q3K_SUBS_PER_SB;              /* 0..7  */
    const int sub = tid % Q3K_SUBS_PER_SB;              /* 0..15 */
    const int stride = (GEMV_WARPS * 32) / Q3K_SUBS_PER_SB;  /* 8 */

    float sumf = 0.0f;

    for (int sb = sb0; sb < n_super; sb += stride) {
        const uint8_t *wb = wrow + (size_t)sb * GPU_BLOCK_Q3_K_SIZE;
        const uint8_t *hm = wb;            /* hmask[32]  */
        const uint8_t *qs = wb + 32;       /* qs[64]     */
        const uint16_t d_raw = (uint16_t)(wb[108] | ((uint16_t)wb[109] << 8));
        const float d = xd[sb] * gpu_fp16_to_fp32(d_raw);

        const int scale = gemv_q3k_scale(wb + 96, sub);

        /* Sub-block (== one 16-value scale group) `sub` = 0..15, mapping taken
         * directly from dequantize_row_q3_K's loop order:
         *   chunk  = sub / 8          (qs advances 32 bytes per chunk)
         *   w      = sub % 8          (8 groups per chunk)
         *   j      = w / 2            (bit shift 2*j)
         *   half   = w % 2            (bytes 0..15 or 16..31)
         *   hmask byte offset = half ? 16 : 0
         *   hmask bit = chunk*4 + j   (m carries across chunks, it is not reset)
         *   q8 offset = sub*16
         */
        const int chunk = sub >> 3;
        const int w     = sub & 7;
        const int j     = w >> 1;
        const int half  = w & 1;
        const int shift = 2 * j;
        const int boff  = chunk * 32 + (half ? 16 : 0);
        const uint8_t mbit = (uint8_t)(1u << (chunk * 4 + j));
        const uint8_t *qrun = qs + boff;
        const uint8_t *hrun = hm + (half ? 16 : 0);
        const int8_t  *q8  = xq + (size_t)sb * 256 + (size_t)sub * 16;

        int acc = 0;
        for (int l = 0; l < 16; l += 4) {
            int8_t w4[4];
            for (int t = 0; t < 4; t++) {
                int v = (qrun[l + t] >> shift) & 3;
                if (!(hrun[l + t] & mbit)) v -= 4;
                w4[t] = (int8_t)v;
            }
            int aA = gemv_get_int_b2(q8 + l, 0);
            int wAi;
            memcpy(&wAi, w4, 4);
            acc = gpu_dot4_i8_packed(wAi, aA, acc);
        }

        sumf += d * (float)scale * (float)acc;
    }

    __shared__ float red[GEMV_WARPS > 1 ? GEMV_WARPS - 1 : 1][32];
    if (warp > 0) red[warp - 1][lane] = sumf;
    gpuSyncthreads();
    if (warp > 0) return;
    for (int w = 0; w < GEMV_WARPS - 1; w++) sumf += red[w][lane];
    for (int off = 16; off > 0; off >>= 1)
        sumf += gpuShflXorSync(0xFFFFFFFFu, sumf, off);
    if (lane == 0) y[o] = sumf;
}


/* Q2_K: 256 weights in 84 bytes -- scales[16] qs[64] d(2) dmin(2).
 *
 * Layout is Q3_K's minus the hmask: 16 sub-blocks of 16 values, two chunks of
 * 128, qs advancing 32 bytes per chunk with the bit shift walking 0,2,4,6.
 * Values are UNSIGNED 2-bit (0..3) and scales[16] holds one 4-bit scale and
 * one 4-bit min per sub-block (low nibble = scale, high nibble = min), so
 * unlike Q4_K/Q5_K there is no 6-bit unpacking, and since a sub-block here is
 * exactly 16 values it maps 1:1 onto one Q8_K bsums entry -- no pairing.
 *
 * Transcribed from dequantize_row_q2_K() in quant.c. */
__global__ void
picolm_q2_K_gemv(float *y,
                 const int8_t *xq, const float *xd, const int16_t *xbsums,
                 const void *weights,
                 int I, int O, int row_bytes, int y_stride) {
    const int lane = (int)(gpuThreadIdx_x & 31);
    const int warp = (int)(gpuThreadIdx_x >> 5);
    const int tid  = warp * 32 + lane;
    const int o    = (int)gpuBlockIdx_x;
    if (o >= O) return;

    const int n_super = I / 256;
    const uint8_t *wrow = (const uint8_t *)weights + (size_t)o * row_bytes;

    const int sb0 = tid / Q3K_SUBS_PER_SB;                  /* 16 subs/sb */
    const int sub = tid % Q3K_SUBS_PER_SB;
    const int stride = (GEMV_WARPS * 32) / Q3K_SUBS_PER_SB; /* 8 */

    float sumf = 0.0f;

    for (int sb = sb0; sb < n_super; sb += stride) {
        const uint8_t *wb = wrow + (size_t)sb * GPU_BLOCK_Q2_K_SIZE;
        const uint16_t d_raw    = (uint16_t)(wb[80] | ((uint16_t)wb[81] << 8));
        const uint16_t dmin_raw = (uint16_t)(wb[82] | ((uint16_t)wb[83] << 8));
        const float d    = xd[sb] * gpu_fp16_to_fp32(d_raw);
        const float dmin = xd[sb] * gpu_fp16_to_fp32(dmin_raw);

        /* Only the one 4-bit scale/min pair this thread uses. */
        const uint8_t sc = wb[sub];
        const int scale = (int)(sc & 0xF);
        const int minv  = (int)(sc >> 4);

        const int chunk = sub >> 3;
        const int w     = sub & 7;
        const int j     = w >> 1;
        const int half  = w & 1;
        const int shift = 2 * j;
        /* block_q2_K: scales[16] qs[64] d(2) dmin(2) -- qs starts at byte 16.
         * (Q3_K's is at 32 because hmask[32] comes first; copying that offset
         * here read scale/d bytes as quants and produced garbage.) */
        const uint8_t *qrun = wb + 16 + chunk * 32 + (half ? 16 : 0);
        const int8_t  *q8   = xq + (size_t)sb * 256 + (size_t)sub * 16;

        int acc = 0;
        for (int l = 0; l < 16; l += 4) {
            int8_t w4[4];
            for (int t = 0; t < 4; t++)
                w4[t] = (int8_t)((qrun[l + t] >> shift) & 3);   /* 0..3 */
            int aA = gemv_get_int_b2(q8 + l, 0);
            int wAi;
            memcpy(&wAi, w4, 4);
            acc = gpu_dot4_i8_packed(wAi, aA, acc);
        }

        sumf += scale * d * (float)acc;
        sumf -= dmin * (float)minv * (float)xbsums[sb * 16 + sub];
    }

    __shared__ float red[GEMV_WARPS > 1 ? GEMV_WARPS - 1 : 1][32];
    if (warp > 0) red[warp - 1][lane] = sumf;
    gpuSyncthreads();
    if (warp > 0) return;
    for (int w2 = 0; w2 < GEMV_WARPS - 1; w2++) sumf += red[w2][lane];
    for (int off = 16; off > 0; off >>= 1)
        sumf += gpuShflXorSync(0xFFFFFFFFu, sumf, off);
    if (lane == 0) y[o] = sumf;
}


/* ================================================================
 * IQ6_K GEMV (decode, S=1) against Q8_K activations
 * ================================================================
 *
 * 212-byte block: d(2) extra(2) scales[16] qs[128] qh[64] = 256 values,
 * organized as FOUR 64-value sub-blocks. Within a sub-block, four groups of
 * 16 values each carry their own signed scale (scales[4*ib64 + g]) and their
 * own LUT-select bit ((extra >> g) & 1, with extra >>= 4 per sub-block).
 *
 * The 6-bit code is NOT a linear quantisation: it indexes a 64-entry
 * non-linear table, value = iq6nl_lut[q] - 128 + mn. Transcribed from
 * dequantize_row_iq6_k() and the scalar branch of vec_dot_iq6_k_q8_k() in
 * quant.c, which agree.
 *
 * LUT handling: the table is 64 bytes and lives in __constant__ memory (the
 * natural place on CUDA; HIP maps it to the same read-only path). The values
 * are -127..121, which fits int8, so the dot still runs through
 * gpu_dot4_i8_packed once the LUT result has been materialised.
 */
/* __device__, not __constant__: the tree uses no __constant__ at all today and
 * adding one is a new CUDA/HIP surface for a 64-byte table. A __device__ array
 * is handled identically by nvcc and hipcc, and the kernel is bandwidth-bound
 * on the weights anyway, so the caching difference does not matter. */
/* LUT comes from backend_gpu_dequant.cuh (iq6nl_lut_dev). */
#define IQ6K_VALS_PER_SB  64
/* one thread per 16-value group: 4 groups per 64-value sub-block, so 16
 * threads cover a whole super-block (4 sub-blocks x 4 groups) */
#define IQ6K_GROUPS_PER_SUPER 16

__global__ void
picolm_iq6_K_gemv(float *y,
                  const int8_t *xq, const float *xd, const int16_t *xbsums,
                  const void *weights,
                  int I, int O, int row_bytes, int y_stride) {
    (void)xbsums;   /* no mins term: the bias lives in the LUT value */
    const int lane = (int)(gpuThreadIdx_x & 31);
    const int warp = (int)(gpuThreadIdx_x >> 5);
    const int tid  = warp * 32 + lane;
    const int o    = (int)gpuBlockIdx_x;
    if (o >= O) return;

    const int n_super = I / 256;
    const uint8_t *wrow = (const uint8_t *)weights + (size_t)o * row_bytes;

    const int sb0    = tid / IQ6K_GROUPS_PER_SUPER;                  /* 0..7 */
    const int grp_id = tid % IQ6K_GROUPS_PER_SUPER;                  /* 0..15 */
    const int stride = (GEMV_WARPS * 32) / IQ6K_GROUPS_PER_SUPER;    /* 8 */

    float sumf = 0.0f;

    for (int sb = sb0; sb < n_super; sb += stride) {
        const uint8_t *wb = wrow + (size_t)sb * GPU_BLOCK_IQ6_K_SIZE;
        const uint16_t d_raw = (uint16_t)(wb[0] | ((uint16_t)wb[1] << 8));
        const uint16_t extra = (uint16_t)(wb[2] | ((uint16_t)wb[3] << 8));
        const int8_t  *sl = (const int8_t *)(wb + 4);
        const uint8_t *qs = wb + 20;
        const uint8_t *qh = wb + 148;
        const float d = xd[sb] * gpu_fp16_to_fp32(d_raw);

        /* grp_id -> sub-block ib64 (0..3) and group g (0..3) */
        const int ib64 = grp_id >> 2;
        const int g    = grp_id & 3;

        /* Pointer walk, matching the reference exactly:
         *   qs += 32 per sub-block, qh += 32 every second sub-block,
         *   shift += 4 per sub-block and wraps 8 -> 0,
         *   extra >>= 4 per sub-block. */
        const uint8_t *qs_c = qs + ib64 * 32;
        const uint8_t *qh_c = qh + (ib64 >> 1) * 32;
        const int shift = (ib64 & 1) * 4;
        const int mn = (extra >> (4 * ib64 + g)) & 1;
        const int scale = (int)sl[4 * ib64 + g];

        /* q8 for this 16-value group sits at grp_id*16 within the super-block */
        const int8_t *q8 = xq + (size_t)sb * 256 + (size_t)grp_id * 16;

        /* Hoist everything that depends only on g out of the loop: the previous
         * version ran a 4-way `if (g == ...)` chain on every one of the 16
         * elements, which measured 376 GB/s against the ~800 the other K-quant
         * GEMVs reach. Only two things vary with g -- the qs byte offset
         * (0 or 16), the nibble selector, and the qh bit field -- so the loop
         * below is straight-line with no branch. */
        const int   qs_off = (g & 1) ? 16 : 0;      /* q2/q4 read +16      */
        const int   nib_sh = (g & 2) ? 4 : 0;       /* q3/q4 high nibble   */
        const int   hi_msk = (g & 2) ? 0x0C : 0x03; /* qh bits 2-3 or 0-1  */
        const int   hi_sh  = (g & 2) ? 2 : 4;       /* shift into bits 4-5 */
        const uint8_t *qs_g = qs_c + qs_off;
        const uint8_t *qh_g = qh_c + qs_off;

        int acc = 0;
        for (int l = 0; l < 16; l += 4) {
            int8_t w4[4];
            for (int t = 0; t < 4; t++) {
                const int j = l + t;
                const int q = ((qs_g[j] >> nib_sh) & 0x0F)
                            | (((qh_g[j] >> shift) & hi_msk) << hi_sh);
                w4[t] = (int8_t)((int)iq6nl_lut_dev[q] - 128 + mn);
            }
            int aA = gemv_get_int_b2(q8 + l, 0);
            int wAi;
            memcpy(&wAi, w4, 4);
            acc = gpu_dot4_i8_packed(wAi, aA, acc);
        }

        sumf += d * (float)scale * (float)acc;
    }

    __shared__ float red[GEMV_WARPS > 1 ? GEMV_WARPS - 1 : 1][32];
    if (warp > 0) red[warp - 1][lane] = sumf;
    gpuSyncthreads();
    if (warp > 0) return;
    for (int w2 = 0; w2 < GEMV_WARPS - 1; w2++) sumf += red[w2][lane];
    for (int off = 16; off > 0; off >>= 1)
        sumf += gpuShflXorSync(0xFFFFFFFFu, sumf, off);
    if (lane == 0) y[o] = sumf;
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

/* ---- Q8_0 x Q8_0 GEMM for GPUs without IMMA (Pascal/Volta/Turing) ----
 *
 * Same output as picolm_q8_q8_matmul_tiled, bit for bit, about 6x faster
 * on a V100.
 *
 * The tiled kernel gives each output a 256-thread block: thread t sums the
 * terms of blocks t, t+256, ... in double, then a shared-memory tree adds
 * slot t+n into slot t for n = 128, 64, ..., 1. The result depends only on
 * the 256 slot values and the pairs that tree adds, so forming the same
 * slots and adding the same pairs gives the same double, hence the same
 * float.
 *
 * Here a warp owns Q8G_WO x Q8G_WS outputs. Lane r forms slots r, r+32, ...,
 * r+224 and adds them like tree levels 128/64/32; taking the slots in the
 * order 0,4,2,6,1,5,3,7 (times 32, plus r) needs only three doubles of stack
 * per output. Levels 16..1 are the shuffle-down at the end. Blocks of 32
 * Q8_0 blocks are staged in shared memory, one per lane, and every staged
 * weight block is used for Q8G_TS tokens and every activation block for
 * Q8G_TO rows; the old kernel used each weight byte once per token.
 *
 * 32-lane warps only (CUDA); the host keeps the tiled kernel on HIP. */
#define Q8G_TO 8    /* weight rows per thread block */
#define Q8G_TS 8    /* tokens per thread block */
#define Q8G_WO 2    /* rows per warp */
#define Q8G_WS 4    /* tokens per warp: 8 warps = (8/2) x (8/4) */
#define Q8G_THREADS 256
#define Q8G_PAD 12  /* ints per staged block: 8 + 4 keeps int4 reads conflict-free */

__global__ void __launch_bounds__(Q8G_THREADS)
picolm_q8_q8_matmul_dp4a(float *y, const int8_t *xq, const float *xd,
                         const void *weights,
                         int S, int I, int O, int row_bytes, int y_stride) {
    __shared__ __align__(16) int wq_sh[Q8G_TO][32 * Q8G_PAD];
    __shared__ float wd_sh[Q8G_TO][32];
    __shared__ __align__(16) int xq_sh[Q8G_TS][32 * Q8G_PAD];
    __shared__ float xd_sh[Q8G_TS][32];

    int tid = gpuThreadIdx_x, lane = tid & 31, warp = tid >> 5;
    int o0 = gpuBlockIdx_x * Q8G_TO, s0 = gpuBlockIdx_y * Q8G_TS;
    int wo = (warp % (Q8G_TO / Q8G_WO)) * Q8G_WO;
    int ws = (warp / (Q8G_TO / Q8G_WO)) * Q8G_WS;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    int n_chunks = (n_blocks + 31) / 32;

    double a[Q8G_WO][Q8G_WS], b[Q8G_WO][Q8G_WS], c[Q8G_WO][Q8G_WS];

    for (int j = 0; j < 8; j++) {
        int k = ((j & 1) << 2) | (j & 2) | ((j & 4) >> 2);  /* bit-reverse j */
        double leaf[Q8G_WO][Q8G_WS];
        for (int i = 0; i < Q8G_WO; i++)
            for (int l = 0; l < Q8G_WS; l++) leaf[i][l] = 0.0;

        /* Slot k*32 + lane holds blocks (k + 8m)*32 + lane, m = 0, 1, ... */
        for (int ch = k; ch < n_chunks; ch += 8) {
            int bi0 = ch * 32;
            for (int it = tid; it < (Q8G_TO + Q8G_TS) * 32; it += Q8G_THREADS) {
                int r = it >> 5, bj = it & 31, bi = bi0 + bj;
                if (r < Q8G_TO) {
                    int o = o0 + r, *wp = &wq_sh[r][bj * Q8G_PAD];
                    if (o < O && bi < n_blocks) {
                        const uint16_t *p = (const uint16_t *)((const uint8_t *)weights
                            + (size_t)o * row_bytes + (size_t)bi * GPU_BLOCK_Q8_0_SIZE);
                        wd_sh[r][bj] = gpu_fp16_to_fp32(p[0]);
                        for (int i = 0; i < 8; i++)
                            wp[i] = (int)((uint32_t)p[1 + 2 * i] | ((uint32_t)p[2 + 2 * i] << 16));
                    } else {
                        wd_sh[r][bj] = 0.0f;
                        for (int i = 0; i < 8; i++) wp[i] = 0;
                    }
                } else {
                    int ls = r - Q8G_TO, s = s0 + ls, *xp = &xq_sh[ls][bj * Q8G_PAD];
                    if (s < S && bi < n_blocks) {
                        const int4 *q = (const int4 *)(xq + (size_t)s * I + (size_t)bi * 32);
                        xd_sh[ls][bj] = xd[(size_t)s * n_blocks + bi];
                        ((int4 *)xp)[0] = q[0]; ((int4 *)xp)[1] = q[1];
                    } else {
                        xd_sh[ls][bj] = 0.0f;
                        for (int i = 0; i < 8; i++) xp[i] = 0;
                    }
                }
            }
            gpuSyncthreads();

            if (bi0 + lane < n_blocks) {
                int4 w[Q8G_WO][2];
                double wdv[Q8G_WO];
                for (int i = 0; i < Q8G_WO; i++) {
                    const int4 *p = (const int4 *)&wq_sh[wo + i][lane * Q8G_PAD];
                    w[i][0] = p[0]; w[i][1] = p[1];
                    wdv[i] = (double)wd_sh[wo + i][lane];
                }
                for (int l = 0; l < Q8G_WS; l++) {
                    const int4 *p = (const int4 *)&xq_sh[ws + l][lane * Q8G_PAD];
                    int4 x0 = p[0], x1 = p[1];
                    double xdv = (double)xd_sh[ws + l][lane];
                    for (int i = 0; i < Q8G_WO; i++) {
                        int32_t acc = 0;
                        acc = gpu_dot4_i8_packed(w[i][0].x, x0.x, acc);
                        acc = gpu_dot4_i8_packed(w[i][0].y, x0.y, acc);
                        acc = gpu_dot4_i8_packed(w[i][0].z, x0.z, acc);
                        acc = gpu_dot4_i8_packed(w[i][0].w, x0.w, acc);
                        acc = gpu_dot4_i8_packed(w[i][1].x, x1.x, acc);
                        acc = gpu_dot4_i8_packed(w[i][1].y, x1.y, acc);
                        acc = gpu_dot4_i8_packed(w[i][1].z, x1.z, acc);
                        acc = gpu_dot4_i8_packed(w[i][1].w, x1.w, acc);
                        /* Same expression and order as the tiled kernel. */
                        leaf[i][l] += (double)acc * wdv[i] * xdv;
                    }
                }
            }
            gpuSyncthreads();
        }

        /* Tree levels 128, 64, 32 over this lane's slots k = 0,4,2,6,1,5,3,7. */
        for (int i = 0; i < Q8G_WO; i++)
            for (int l = 0; l < Q8G_WS; l++) {
                double v = leaf[i][l];
                switch (j) {
                case 0: a[i][l] = v; break;
                case 1: a[i][l] += v; break;
                case 2: b[i][l] = v; break;
                case 3: b[i][l] += v; a[i][l] += b[i][l]; break;
                case 4: b[i][l] = v; break;
                case 5: b[i][l] += v; break;
                case 6: c[i][l] = v; break;
                case 7: c[i][l] += v; b[i][l] += c[i][l]; a[i][l] += b[i][l]; break;
                }
            }
    }

    /* Tree levels 16..1: lane t adds lane t+n, exactly like slot t and t+n. */
    for (int i = 0; i < Q8G_WO; i++)
        for (int l = 0; l < Q8G_WS; l++) {
            double v = a[i][l];
            for (int n = 16; n > 0; n >>= 1)
                v += gpuShflDownSync(0xFFFFFFFFu, v, n);
            int o = o0 + wo + i, s = s0 + ws + l;
            if (lane == 0 && o < O && s < S)
                y[(size_t)s * ys + o] = (float)v;
        }
}

/* ---- Q8_0 x Q8_0 GEMM on Volta tensor cores (mma.m8n8k4, f16 in, f32 acc) ----
 *
 * Volta has no int8 MMA, but int8 values are exact in fp16, their products
 * are exact, and the 32-term dot product of one Q8_0 block (|sum| < 2^19)
 * is exact in the f32 accumulator. So each block's integer dot product is
 * computed on the tensor cores exactly, and only then scaled:
 * sum += dot * wd * xd in f32, block by block, like the CPU.
 *
 * mma.m8n8k4 runs four independent 8x8x4 products, one per quad-pair
 * qp = (lane >> 2) & 3. With r = (lane & 3) + 4 * (lane >> 4), a lane feeds
 * row r of A and column r of B (k = 0..3), and gets D[j] at
 *   row (lane & 1) + 2 * ((j >> 1) & 1) + 4 * (lane >> 4),
 *   col (j & 1) + 2 * ((lane >> 1) & 1) + 4 * (j >> 2)
 * (measured on a V100). Quad-pairs are laid out 2 x 2, so one instruction
 * covers 16 rows x 16 tokens; a warp does Q8H_WG x Q8H_WH of those.
 *
 * Not bit-identical to the DP4A/tiled kernels (f32, not double, across
 * blocks); max error vs a double CPU reference ~3e-7 of sum |terms|. */
#define Q8H_TO 64   /* weight rows per thread block */
#define Q8H_TS 128  /* tokens per thread block */
#define Q8H_KB 2    /* Q8_0 blocks staged at a time */
#define Q8H_WG 4    /* 16-row groups per warp: a warp does 64 rows x 32 tokens */
#define Q8H_WH 2    /* 16-token groups per warp */
#define Q8H_WR (Q8H_TO / (16 * Q8H_WG))          /* warps along rows */
#define Q8H_THREADS (32 * Q8H_WR * (Q8H_TS / (16 * Q8H_WH)))  /* 128 */
#define Q8H_LD (Q8H_KB * 32 + 4)  /* halves per staged row: 2 mod 32 words, no bank conflicts */

/* Four int8 (one 32-bit word) to two exact half2: bias to 0..255, put each
 * byte under the exponent of 1024.0 (0x64xx = 1024 + byte), subtract 1152. */
__device__ static inline void q8h_cvt4(__half2 *dst, uint32_t v) {
    v ^= 0x80808080u;
    uint32_t lo = __byte_perm(v, 0x64646464u, 0x4140);
    uint32_t hi = __byte_perm(v, 0x64646464u, 0x4342);
    __half2 bias = __float2half2_rn(1152.0f);
    dst[0] = __hsub2(*(__half2 *)&lo, bias);
    dst[1] = __hsub2(*(__half2 *)&hi, bias);
}

__global__ void __launch_bounds__(Q8H_THREADS)
picolm_q8_q8_matmul_hmma(float *y, const int8_t *xq, const float *xd,
                         const void *weights,
                         int S, int I, int O, int row_bytes, int y_stride) {
    __shared__ __align__(16) __half w_sh[Q8H_TO][Q8H_LD];
    __shared__ __align__(16) __half x_sh[Q8H_TS][Q8H_LD];
    __shared__ float wd_sh[Q8H_KB][Q8H_TO];
    __shared__ float xd_sh[Q8H_KB][Q8H_TS];

    int tid = gpuThreadIdx_x, lane = tid & 31, warp = tid >> 5;
    int o0 = gpuBlockIdx_x * Q8H_TO, s0 = gpuBlockIdx_y * Q8H_TS;
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;

    int qp = (lane >> 2) & 3;
    int r = (lane & 3) + 4 * (lane >> 4);                 /* A row / B col */
    int wr = (warp % Q8H_WR) * 16 * Q8H_WG, wc = (warp / Q8H_WR) * 16 * Q8H_WH;  /* warp tile */
    int arow[Q8H_WG], bcol[Q8H_WH];
    for (int g = 0; g < Q8H_WG; g++) arow[g] = wr + 16 * g + 8 * (qp & 1) + r;
    for (int h = 0; h < Q8H_WH; h++) bcol[h] = wc + 16 * h + 8 * (qp >> 1) + r;
    int drow[Q8H_WG][2], dcol[Q8H_WH][4];   /* rows/tokens of this lane's D values */
    for (int g = 0; g < Q8H_WG; g++)
        for (int u = 0; u < 2; u++)
            drow[g][u] = wr + 16 * g + 8 * (qp & 1) + (lane & 1) + 2 * u + 4 * (lane >> 4);
    for (int h = 0; h < Q8H_WH; h++)
        for (int u = 0; u < 4; u++)
            dcol[h][u] = wc + 16 * h + 8 * (qp >> 1) + (u & 1) + 2 * ((lane >> 1) & 1) + 4 * (u >> 1);

    float sum[Q8H_WG][Q8H_WH][8];
    for (int g = 0; g < Q8H_WG; g++)
        for (int h = 0; h < Q8H_WH; h++)
            for (int j = 0; j < 8; j++) sum[g][h][j] = 0.0f;

    for (int b0 = 0; b0 < n_blocks; b0 += Q8H_KB) {
        /* Stage: one thread per (row or token, block); 2 x 256 units. */
        for (int it = tid; it < (Q8H_TO + Q8H_TS) * Q8H_KB; it += Q8H_THREADS) {
            int kb = it % Q8H_KB, rr = it / Q8H_KB, bi = b0 + kb;
            uint32_t q[8];
            float d = 0.0f;
            if (rr < Q8H_TO) {
                int o = o0 + rr;
                if (o < O && bi < n_blocks) {
                    const uint16_t *p = (const uint16_t *)((const uint8_t *)weights
                        + (size_t)o * row_bytes + (size_t)bi * GPU_BLOCK_Q8_0_SIZE);
                    d = gpu_fp16_to_fp32(p[0]);
                    for (int i = 0; i < 8; i++) q[i] = (uint32_t)p[1 + 2 * i] | ((uint32_t)p[2 + 2 * i] << 16);
                } else for (int i = 0; i < 8; i++) q[i] = 0;
                wd_sh[kb][rr] = d;
                for (int i = 0; i < 8; i++) q8h_cvt4((__half2 *)&w_sh[rr][kb * 32 + 4 * i], q[i]);
            } else {
                int ls = rr - Q8H_TO, s = s0 + ls;
                if (s < S && bi < n_blocks) {
                    const int4 *p = (const int4 *)(xq + (size_t)s * I + (size_t)bi * 32);
                    ((int4 *)q)[0] = p[0]; ((int4 *)q)[1] = p[1];
                    d = xd[(size_t)s * n_blocks + bi];
                } else for (int i = 0; i < 8; i++) q[i] = 0;
                xd_sh[kb][ls] = d;
                for (int i = 0; i < 8; i++) q8h_cvt4((__half2 *)&x_sh[ls][kb * 32 + 4 * i], q[i]);
            }
        }
        gpuSyncthreads();

        for (int kb = 0; kb < Q8H_KB; kb++) {
            float acc[Q8H_WG][Q8H_WH][8];
            for (int g = 0; g < Q8H_WG; g++)
                for (int h = 0; h < Q8H_WH; h++)
                    for (int j = 0; j < 8; j++) acc[g][h][j] = 0.0f;
#pragma unroll
            for (int k = kb * 32; k < kb * 32 + 32; k += 4) {
                uint2 a[Q8H_WG], b[Q8H_WH];
                for (int g = 0; g < Q8H_WG; g++) a[g] = *(const uint2 *)&w_sh[arow[g]][k];
                for (int h = 0; h < Q8H_WH; h++) b[h] = *(const uint2 *)&x_sh[bcol[h]][k];
                for (int g = 0; g < Q8H_WG; g++)
                    for (int h = 0; h < Q8H_WH; h++) {
                        float *c = acc[g][h];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIP__)
                        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                            "{%0,%1,%2,%3,%4,%5,%6,%7};"
                            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
                              "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                            : "r"(a[g].x), "r"(a[g].y), "r"(b[h].x), "r"(b[h].y));
#endif
                    }
            }
            /* acc holds exact integer dot products; scale them in. D[j] sits
             * at row drow[g][(j >> 1) & 1], token dcol[h][(j & 1) | (j >> 2) << 1]. */
            float wd[Q8H_WG][2], xs[Q8H_WH][4];
            for (int g = 0; g < Q8H_WG; g++)
                for (int u = 0; u < 2; u++) wd[g][u] = wd_sh[kb][drow[g][u]];
            for (int h = 0; h < Q8H_WH; h++)
                for (int u = 0; u < 4; u++) xs[h][u] = xd_sh[kb][dcol[h][u]];
            for (int g = 0; g < Q8H_WG; g++)
                for (int h = 0; h < Q8H_WH; h++)
                    for (int j = 0; j < 8; j++)
                        sum[g][h][j] += acc[g][h][j] * (wd[g][(j >> 1) & 1] * xs[h][(j & 1) | (j >> 2) << 1]);
        }
        gpuSyncthreads();
    }

    for (int g = 0; g < Q8H_WG; g++)
        for (int h = 0; h < Q8H_WH; h++)
            for (int j = 0; j < 8; j++) {
                int o = o0 + drow[g][(j >> 1) & 1], s = s0 + dcol[h][(j & 1) | (j >> 2) << 1];
                if (o < O && s < S) y[(size_t)s * ys + o] = sum[g][h][j];
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
 * TODO: This kernel is currently DISABLED due to an out-of-bounds ql read bug
 * that was not fully resolved. The original nested loop structure (half/sub_chunk/
 * sc_half/sub_l) caused ql_c[sub_l+32] to read past the 64-byte per-half ql window
 * into qh memory. A restructured version matching the CPU reference was attempted
 * but produced incorrect results. The scalar fallback path works correctly.
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

        /* Match CPU dequantize_row_q6_K loop structure exactly.
         * Two halves of 128 elements each. Each half has 4 sub_chunks
         * of 32 elements. Per sub_chunk, two scale groups (sub_l 0..15,
         * 16..31). The ql array is 128 bytes: first 64 bytes for half 0,
         * second 64 bytes for half 1. The qh array is 64 bytes: first 32
         * for half 0, second 32 for half 1.
         *
         * Within each half: ql_c points to the half's 64-byte ql window.
         * sub_chunk 0 (q1): reads ql_c[0..31] low nibbles
         * sub_chunk 1 (q2): reads ql_c[32..63] low nibbles
         * sub_chunk 2 (q3): reads ql_c[0..31] high nibbles
         * sub_chunk 3 (q4): reads ql_c[32..63] high nibbles
         * All sub_chunks read qh_c[0..31]. */
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_c = ql + half * 64;
            const uint8_t *qh_c = qh + half * 32;
            int is_base = half * 8;

            for (int sc_half = 0; sc_half < 2; sc_half++) {
                int sub_l_base = sc_half * 16;  /* 0 or 16 */
                for (int sub_l = tid + sub_l_base; sub_l < sub_l_base + 32; sub_l += 32) {
                    int is_l = is_base + (sub_l / 16);  /* scale group base for this sub_l */

                    /* q1: sub_chunk=0, sc=sc_half, sc_idx=is_l+0 */
                    {
                        uint8_t ql_val = ql_c[sub_l] & 0xF;
                        uint8_t qh_val = (qh_c[sub_l] >> 0) & 3;
                        int q = (ql_val | (qh_val << 4)) - 32;
                        int elem_idx = bi * 256 + half * 128 + 0 * 32 + sub_l;
                        sum += (double)xp[elem_idx] * (double)d * (double)scales[is_l + 0] * (double)q;
                    }
                    /* q2: sub_chunk=1, sc=sc_half, sc_idx=is_l+2 */
                    {
                        uint8_t ql_val = ql_c[sub_l + 32] & 0xF;
                        uint8_t qh_val = (qh_c[sub_l] >> 2) & 3;
                        int q = (ql_val | (qh_val << 4)) - 32;
                        int elem_idx = bi * 256 + half * 128 + 1 * 32 + sub_l;
                        sum += (double)xp[elem_idx] * (double)d * (double)scales[is_l + 2] * (double)q;
                    }
                    /* q3: sub_chunk=2, sc=sc_half, sc_idx=is_l+4 */
                    {
                        uint8_t ql_val = ql_c[sub_l] >> 4;
                        uint8_t qh_val = (qh_c[sub_l] >> 4) & 3;
                        int q = (ql_val | (qh_val << 4)) - 32;
                        int elem_idx = bi * 256 + half * 128 + 2 * 32 + sub_l;
                        sum += (double)xp[elem_idx] * (double)d * (double)scales[is_l + 4] * (double)q;
                    }
                    /* q4: sub_chunk=3, sc=sc_half, sc_idx=is_l+6 */
                    {
                        uint8_t ql_val = ql_c[sub_l + 32] >> 4;
                        uint8_t qh_val = (qh_c[sub_l] >> 6) & 3;
                        int q = (ql_val | (qh_val << 4)) - 32;
                        int elem_idx = bi * 256 + half * 128 + 3 * 32 + sub_l;
                        sum += (double)xp[elem_idx] * (double)d * (double)scales[is_l + 6] * (double)q;
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
                                (size_t)kb * GPU_BLOCK_Q4_0_SIZE + 2;   /* 16 bytes */
            /* m16n8k32 B fragment: this lane's b0 holds K[tid*4 .. tid*4+3]
             * and b1 holds K[16+tid*4 .. 16+tid*4+3].
             *
             * GGUF Q4_0 packs those 32 values into only SIXTEEN bytes, with
             * BOTH nibbles of a byte belonging to the SAME 16-byte window
             * (dequantize_row_q4_0): qs[j] & 0xF -> element j, and
             * qs[j] >> 4 -> element j+16. K[0..15] are the low nibbles of
             * qs[0..15] and K[16..31] the high nibbles of the SAME bytes.
             *
             * So b0 and b1 are both built from the four bytes
             * qs[tid*4 .. tid*4+3]: low nibbles for b0, high nibbles for b1.
             * (The previous code took them from 2 different byte ranges and
             * read the high-nibble register starting at qs+16, i.e. past the
             * end of the 16-byte array, into the next block's scale.) */
            uint32_t raw;
            memcpy(&raw, qs + tid * 4, 4);

            uint32_t lo = raw & 0x0F0F0F0F;                    /* -> K[tid*4 ..] */
            uint32_t hi = (raw >> 4) & 0x0F0F0F0F;              /* -> K[16+tid*4 ..] */

            /* Q4_0 stores 0..15 meaning -8..7: subtract 8 from every byte.
             * Mask to 0xFF after the subtraction, otherwise a zero nibble
             * (0 - 8) borrows into the neighbouring byte. */
            { uint32_t v = lo;
              b0 = (int)((((v      ) & 0xFF) - 8) & 0xFF)
                 | (int)(((((v >>  8) & 0xFF) - 8) & 0xFF) <<  8)
                 | (int)(((((v >> 16) & 0xFF) - 8) & 0xFF) << 16)
                 | (int)(((((v >> 24) & 0xFF) - 8) & 0xFF) << 24); }
            { uint32_t v = hi;
              b1 = (int)((((v      ) & 0xFF) - 8) & 0xFF)
                 | (int)(((((v >>  8) & 0xFF) - 8) & 0xFF) <<  8)
                 | (int)(((((v >> 16) & 0xFF) - 8) & 0xFF) << 16)
                 | (int)(((((v >> 24) & 0xFF) - 8) & 0xFF) << 24); }
        }

        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif

        /* Epilogue. Q4_0 has one scale per (output row, block) -- there are no
         * per-sub-block scales -- but the row that matters is the OUTPUT COLUMN
         * this lane accumulates, not gid.
         *
         * The m16n8k32 accumulator layout (see the store loop below and the
         * working picolm_q8_q8_matmul_imma) puts sum[0]/sum[2] in output column
         * tile_o + tid*2 and sum[1]/sum[3] in tile_o + tid*2 + 1. Looking the
         * scale up at tile_o + gid means every thread applies a different
         * column's weight scale. */
        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];
        const uint8_t *wb0 = w + (size_t)(tile_o + tid * 2) * row_bytes
                             + (size_t)kb * GPU_BLOCK_Q4_0_SIZE;
        const uint8_t *wb1 = w + (size_t)(tile_o + tid * 2 + 1) * row_bytes
                             + (size_t)kb * GPU_BLOCK_Q4_0_SIZE;
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
                                (size_t)kb * GPU_BLOCK_Q4_1_SIZE + 4;  /* 16 bytes */
            /* Same layout and same fix as Q4_0: 32 values in SIXTEEN bytes,
             * qs[j] & 0xF -> element j, qs[j] >> 4 -> element j+16
             * (dequantize_row_q4_1). This lane's b0 holds K[tid*4 .. tid*4+3]
             * and b1 holds K[16+tid*4 ..], so both come from the same four
             * bytes qs[tid*4 .. tid*4+3]. */
            uint32_t raw;
            memcpy(&raw, qs + tid * 4, 4);

            b0 = (int)(raw & 0x0F0F0F0F);            /* K[tid*4 ..]    */
            b1 = (int)((raw >> 4) & 0x0F0F0F0F);     /* K[16+tid*4 ..] */
            /* No bias needed: Q4_1 nibbles are unsigned 0..15 (top bit clear,
             * so the s8 IMMA sign-extension keeps them positive) and the min
             * term `m` is applied in the epilogue. */
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
            /* Reduce over the 4 lanes of this group (lanes gid*4..gid*4+3)
             * and broadcast the total to all of them.
             *
             * The mask must name the whole warp. All 32 lanes execute these
             * calls, and __shfl_*_sync requires that every lane in the mask
             * takes part; a per-group mask (0xF << gid*4) makes lanes holding
             * DIFFERENT masks join the same collective, which is undefined
             * behaviour. Grouping is expressed by WIDTH=4 instead: that splits
             * the warp into 8 independent 4-lane sub-groups and makes both the
             * down-shuffle and the broadcast group-relative, so srcLane 0 means
             * "lane 0 of this group" rather than lane 0 of the warp. */
            const unsigned mask = 0xFFFFFFFFu;
            sq0 += gpuShflDownSync(mask, sq0, 2, 4);
            sq1 += gpuShflDownSync(mask, sq1, 2, 4);
            sq0 += gpuShflDownSync(mask, sq0, 1, 4);
            sq1 += gpuShflDownSync(mask, sq1, 1, 4);
            sq0 = gpuShflSync(mask, sq0, 0, 4);
            sq1 = gpuShflSync(mask, sq1, 0, 4);
        }
#endif

        /* Epilogue: y = d*IMMA + m*sum(q8), applied per OUTPUT COLUMN.
         *
         * sum[0]/sum[2] belong to output column tile_o + tid*2 and
         * sum[1]/sum[3] to tile_o + tid*2 + 1 (the accumulator layout; see the
         * store loop). The old code looked the scales up at tile_o + gid, and
         * applied the m*sum(q8) min term to only two of the four accumulators,
         * and only on tid==0. */
        float dx0 = xd[(size_t)(tile_s + gid) * n_blocks + kb];
        float dx1 = xd[(size_t)(tile_s + gid + 8) * n_blocks + kb];
        const uint8_t *wb0 = w + (size_t)(tile_o + tid * 2) * row_bytes
                             + (size_t)kb * GPU_BLOCK_Q4_1_SIZE;
        const uint8_t *wb1 = w + (size_t)(tile_o + tid * 2 + 1) * row_bytes
                             + (size_t)kb * GPU_BLOCK_Q4_1_SIZE;
        float wd0 = gpu_fp16_to_fp32(wb0[0] | ((uint16_t)wb0[1] << 8));
        float wm0 = gpu_fp16_to_fp32(wb0[2] | ((uint16_t)wb0[3] << 8));
        float wd1 = gpu_fp16_to_fp32(wb1[0] | ((uint16_t)wb1[1] << 8));
        float wm1 = gpu_fp16_to_fp32(wb1[2] | ((uint16_t)wb1[3] << 8));

        sum[0] += wd0 * dx0 * (float)d0 + wm0 * dx0 * (float)sq0;
        sum[1] += wd1 * dx0 * (float)d1 + wm1 * dx0 * (float)sq0;
        sum[2] += wd0 * dx1 * (float)d2 + wm0 * dx1 * (float)sq1;
        sum[3] += wd1 * dx1 * (float)d3 + wm1 * dx1 * (float)sq1;
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

            /* FIXED: the qh high-bit position must match the CPU reference's
             * u1/u2 masks exactly: u1 = 1 << (2*group), u2 = 2 << (2*group).
             * The previous code used bit_shift = 2 + 2*group for sub=1,
             * which is off by one (tests bit 2*group+2, not 2*group+1).
             * At group=3 this even shifted by 8, always reading 0.
             * Verified against quant.c dequantize_row_q5_K: 4/8
             * (group,sub) combinations were wrong before this fix. */
            int bit_shift = 2 * group + sub;  /* sub=0: bits 0,2,4,6; sub=1: bits 1,3,5,7 */

            /* b0: elements 0..3 */
            uint32_t b0_val = 0;
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
            /* Reduce over the 4 lanes of this group (lanes gid*4..gid*4+3)
             * and broadcast the total to all of them.
             *
             * The mask must name the whole warp. All 32 lanes execute these
             * calls, and __shfl_*_sync requires that every lane in the mask
             * takes part; a per-group mask (0xF << gid*4) makes lanes holding
             * DIFFERENT masks join the same collective, which is undefined
             * behaviour. Grouping is expressed by WIDTH=4 instead: that splits
             * the warp into 8 independent 4-lane sub-groups and makes both the
             * down-shuffle and the broadcast group-relative, so srcLane 0 means
             * "lane 0 of this group" rather than lane 0 of the warp. */
            const unsigned mask = 0xFFFFFFFFu;
            sq0 += gpuShflDownSync(mask, sq0, 2, 4);
            sq1 += gpuShflDownSync(mask, sq1, 2, 4);
            sq0 += gpuShflDownSync(mask, sq0, 1, 4);
            sq1 += gpuShflDownSync(mask, sq1, 1, 4);
            sq0 = gpuShflSync(mask, sq0, 0, 4);
            sq1 = gpuShflSync(mask, sq1, 0, 4);
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

/* Q6_K x Q8_0 IMMA kernel, MULTI-TILE.
 *
 * Same reasoning as picolm_q5_k_q8_matmul_imma_mt: the single-tile Q6_K kernel
 * streams the whole K extent of 8 output rows to produce one 16x8 tile, so
 * every weight byte is re-read once per 16-token block. After the Q5_K
 * multi-tile change made Q5_K 7x cheaper, Q6_K became the largest prefill
 * matmul (53% of the matmul time), so it gets the same treatment: decode the
 * weight fragment and the per-column scales ONCE per k-block, then reuse them
 * across Q6K_TILES_S independent 16-row accumulators.
 *
 * Arithmetic per output element is unchanged, so results are bit-identical to
 * the single-tile kernel (verified with PICOLM_DUMP_LOGITS, max|diff| 0.0).
 */
__global__ void
picolm_q6_k_q8_matmul_imma_mt(float *y, const int8_t *xq, const float *xd,
                               const void *weights, int S, int I, int O,
                               int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s0 = gpuBlockIdx_y * (16 * Q6K_TILES_S);
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;

    int n_tiles = (S - tile_s0 + 15) >> 4;
    if (n_tiles > Q6K_TILES_S) n_tiles = Q6K_TILES_S;
    if (n_tiles <= 0) return;

    /* Shared-memory accumulators -- see the long comment in
     * picolm_q5_k_q8_matmul_imma_mt for why these cannot be a register array. */
    extern __shared__ float sum_sh[];
    for (int i = gpuThreadIdx_x; i < Q6K_TILES_S * 32 * 4; i += gpuBlockDim_x)
        sum_sh[i] = 0.0f;
    gpuSyncthreads();
    float *mysum = sum_sh + (size_t)gpuThreadIdx_x * 4;

    for (int kb = 0; kb < n_blocks; kb++) {
        int q6_block = kb / 8;
        int g = kb % 8;

        int wc = tile_o + gid;
        size_t wrow_base = (size_t)wc * row_bytes + q6_block * GPU_BLOCK_Q6_K_SIZE;

        int chunk = g / 4;
        int half = (g % 4) % 2;
        int nibble = (g % 4) / 2;
        int qh_pair = g % 4;

        int ql_base = chunk * 64 + half * 32;
        int qh_base = 128 + chunk * 32;
        int qh_shift = qh_pair * 2;

        /* ---- WEIGHT fragment: decoded once, reused by every tile ---- */
        uint32_t ql0_raw, ql1_raw;
        {
            const uint8_t *qlp = w + wrow_base + ql_base;
            memcpy(&ql0_raw, qlp + tid * 4, 4);
            memcpy(&ql1_raw, qlp + 16 + tid * 4, 4);
        }
        uint32_t ql0, ql1;
        if (nibble) {
            ql0 = (ql0_raw >> 4) & 0x0F0F0F0F;
            ql1 = (ql1_raw >> 4) & 0x0F0F0F0F;
        } else {
            ql0 = ql0_raw & 0x0F0F0F0F;
            ql1 = ql1_raw & 0x0F0F0F0F;
        }
        uint32_t qh0, qh1;
        {
            const uint8_t *qhp = w + wrow_base + qh_base;
            uint32_t qh0_raw, qh1_raw;
            memcpy(&qh0_raw, qhp + tid * 4, 4);
            memcpy(&qh1_raw, qhp + 16 + tid * 4, 4);
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
        uint32_t b0, b1;
        { uint32_t v = ql0 | qh0; b0 = (((v & 0xFF) - 32) & 0xFF) | ((((v >> 8) & 0xFF) - 32) & 0xFF) << 8 | ((((v >> 16) & 0xFF) - 32) & 0xFF) << 16 | (((v >> 24) - 32) & 0xFF) << 24; }
        { uint32_t v = ql1 | qh1; b1 = (((v & 0xFF) - 32) & 0xFF) | ((((v >> 8) & 0xFF) - 32) & 0xFF) << 8 | ((((v >> 16) & 0xFF) - 32) & 0xFF) << 16 | (((v >> 24) - 32) & 0xFF) << 24; }

        /* ---- per-output-column scales, independent of the tile ---- */
        float sc00, sc01, sc10, sc11;
        { size_t wb0_off = (size_t)(tile_o + tid * 2) * row_bytes + q6_block * GPU_BLOCK_Q6_K_SIZE;
          const uint8_t *wb0 = w + wb0_off;
          float wd0 = gpu_fp16_to_fp32((uint16_t)(wb0[208] | (wb0[209] << 8)));
          sc00 = wd0 * (float)(int8_t)wb0[192 + g * 2];
          sc01 = wd0 * (float)(int8_t)wb0[192 + g * 2 + 1]; }
        { size_t wb1_off = (size_t)(tile_o + tid * 2 + 1) * row_bytes + q6_block * GPU_BLOCK_Q6_K_SIZE;
          const uint8_t *wb1 = w + wb1_off;
          float wd1 = gpu_fp16_to_fp32((uint16_t)(wb1[208] | (wb1[209] << 8)));
          sc10 = wd1 * (float)(int8_t)wb1[192 + g * 2];
          sc11 = wd1 * (float)(int8_t)wb1[192 + g * 2 + 1]; }

        for (int t = 0; t < n_tiles; t++) {
            int tile_s = tile_s0 + t * 16;

            int a0, a1, a2, a3;
            {
                int c0 = tid * 4;
                int c2 = c0 + 16;
                int row0 = tile_s + gid;
                int row1 = tile_s + gid + 8;
                if (row0 >= S) row0 = S - 1;
                if (row1 >= S) row1 = S - 1;
                int src0 = row0 * I + kb * 32 + c0;
                int src1 = row1 * I + kb * 32 + c0;
                int src2 = row0 * I + kb * 32 + c2;
                int src3 = row1 * I + kb * 32 + c2;
                int32_t *p;
                p = (int32_t *)(const int8_t *)(&xq[src0]); a0 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src1]); a1 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src2]); a2 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src3]); a3 = p[0];
            }

            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            int d0b = 0, d1b = 0, d2b = 0, d3b = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(0));
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(d0b), "+r"(d1b), "+r"(d2b), "+r"(d3b)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(0), "r"(b1));
#endif
            const int row_a = tile_s + gid;
            const int row_b = tile_s + gid + 8;
            float dx0 = (row_a < S) ? xd[(size_t)row_a * n_blocks + kb] : 0.0f;
            float dx1 = (row_b < S) ? xd[(size_t)row_b * n_blocks + kb] : 0.0f;

            { float *S4 = mysum + (size_t)t * 32 * 4;
              S4[0] += (float)d0  * dx0 * sc00 + (float)d0b  * dx0 * sc01;
              S4[2] += (float)d2  * dx1 * sc00 + (float)d2b  * dx1 * sc01;
              S4[1] += (float)d1  * dx0 * sc10 + (float)d1b  * dx0 * sc11;
              S4[3] += (float)d3  * dx1 * sc10 + (float)d3b  * dx1 * sc11; }
        }
    }

    for (int t = 0; t < n_tiles; t++) {
        int tile_s = tile_s0 + t * 16;
        for (int f = 0; f < 4; f++) {
            int row = tile_s + ((f < 2) ? gid : gid + 8);
            int col = tile_o + tid * 2 + (f % 2);
            if (row < S && col < O)
                y[(size_t)row * ys + (size_t)col] = mysum[(size_t)t * 32 * 4 + f];
        }
    }
}

/* Q5_K x Q8_0 IMMA kernel, COALESCED shared-memory weight staging.
 *
 * WHY (this is the prefill root cause, measured):
 *   The per-warp weight read in every other Q5_K kernel is 8-row strided. With
 *   gid = lane/4 selecting the output row, the 32 lanes of a warp read 8
 *   segments of 16 contiguous bytes, each row_bytes (3520 B for ffn_gate)
 *   apart. That touches 8 different pages per k-block step and uses 16 of
 *   every 128-byte cache line -- 12.5%.
 *
 *     measured      23.5 GB in 695 ms =  34 GB/s  (3% of 1008 GB/s)
 *     decode GEMV   on the SAME weights          = 805 GB/s
 *
 *   The GEMV wins because all 4 warps of a block cooperate on ONE row, so its
 *   reads are contiguous. This kernel applies the same idea to the GEMM: the
 *   whole block loads the 176-byte Q5_K block for each of its 8 output rows
 *   with a LINEAR, coalesced read into shared memory, then each warp unpacks
 *   its own row from shared memory into the mma fragments.
 *
 *   8 rows x 176 B = 1408 B staged per k-block by 256 threads = 5.5 B/thread,
 *   i.e. one coalesced 128-byte transaction per row per 32 threads.
 *
 * Arithmetic is unchanged -- same nibble/high-bit decode, same min-correction,
 * same fp32 epilogue -- so the output must stay bit-identical. Validate with
 * PICOLM_DUMP_LOGITS and --gpu-q8diff before believing any speed number.
 *
 * Shared memory: weight stage 8*176 = 1408 B (double-buffered: 2816 B), plus
 * the accumulator block Q5KW_WARPS*32*Q5K_TILES_S*4 floats = 64 KB at TILES=16,
 * which needs the cudaFuncSetAttribute opt-in (see the dispatch).
 */
__global__ void
picolm_q5_k_q8_matmul_imma_cs(float *y, const int8_t *xq, const float *xd,
                               const void *weights, int S, int I, int O,
                               int row_bytes, int y_stride) {
    const int warp = (int)(gpuThreadIdx_x >> 5);
    const int lane = (int)(gpuThreadIdx_x & 31);
    const int tIdx = (int)gpuThreadIdx_x;
    const int nThreads = (int)gpuBlockDim_x;
    int gid = lane / 4;
    int tid = lane % 4;

    int tile_o = (int)gpuBlockIdx_x * (8 * Q5KW_WARPS) + warp * 8;
    int tile_s0 = (int)gpuBlockIdx_y * (16 * Q5K_TILES_S);
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;

    int n_tiles = (S - tile_s0 + 15) >> 4;
    if (n_tiles > Q5K_TILES_S) n_tiles = Q5K_TILES_S;
    if (n_tiles <= 0) return;

    extern __shared__ float cs_smem[];
    /* [accumulators][staged weight block: 8 rows x 176 B] */
    float *sum_sh = cs_smem;
    const int acc_floats = Q5KW_WARPS * 32 * Q5K_TILES_S * 4;
    uint8_t *wstage = (uint8_t *)(sum_sh + acc_floats);   /* Q5KW_WARPS * 176 */

    {
        int total = Q5KW_WARPS * 32 * Q5K_TILES_S * 4;
        for (int i = tIdx; i < total; i += nThreads) sum_sh[i] = 0.0f;
        /* zero the stage so the tail rows (col >= O) are deterministic */
        int wtotal = Q5KW_WARPS * GPU_BLOCK_Q5_K_SIZE;
        for (int i = tIdx; i < wtotal; i += nThreads) wstage[i] = 0;
    }
    gpuSyncthreads();

    float *mysum = sum_sh + ((size_t)warp * 32 + lane) * 4;

    for (int kb = 0; kb < n_blocks; kb++) {
        int q5_block = kb / 8;
        int g = kb % 8;

        /* ---- COALESCED staged load of this k-block for all 8 rows ----
         * Each of the block's 256 threads copies a 176/warps-worth slice of one
         * row: total 8*176 = 1408 bytes, read as consecutive bytes. */
        {
            const int per = GPU_BLOCK_Q5_K_SIZE;              /* 176 */
            int total = Q5KW_WARPS * per;
            for (int i = tIdx; i < total; i += nThreads) {
                int r = i / per;             /* which of the 8 output rows */
                int off = i - r * per;       /* byte within the Q5_K block  */
                int col = tile_o + r;
                if (col < O)
                    wstage[(size_t)r * per + off] =
                        ((const uint8_t *)weights)[(size_t)col * row_bytes
                                                   + (size_t)q5_block * GPU_BLOCK_Q5_K_SIZE + off];
            }
        }
        gpuSyncthreads();

        /* ---- from here on this is the multi-warp kernel, reading shared ---- */
        const uint8_t *wb = wstage + (size_t)warp * GPU_BLOCK_Q5_K_SIZE;

        int b0, b1;
        {
            int group = g >> 1;
            int sub = g & 1;
            const uint8_t *qs_base = wb + 48 + group * 32;
            const uint8_t *qh_base = wb + 16;
            int bit_shift = 2 * group + sub;
            uint32_t b0_val = 0, b1_val = 0;
            for (int e = 0; e < 4; e++) {
                int l = tid * 4 + e;
                uint8_t qs_byte = qs_base[l];
                uint8_t nibble = sub ? (qs_byte >> 4) : (qs_byte & 0xF);
                uint8_t high = ((qh_base[l] >> bit_shift) & 1) ? 16 : 0;
                b0_val |= (uint32_t)(nibble + high) << (e * 8);

                int l1 = 16 + tid * 4 + e;
                uint8_t qs1 = qs_base[l1];
                uint8_t nib1 = sub ? (qs1 >> 4) : (qs1 & 0xF);
                uint8_t hi1 = ((qh_base[l1] >> bit_shift) & 1) ? 16 : 0;
                b1_val |= (uint32_t)(nib1 + hi1) << (e * 8);
            }
            b0 = (int)b0_val;
            b1 = (int)b1_val;
        }

        float wds0, wdmn0, wds1, wdmn1;
        {
            /* per-OUTPUT-COLUMN scales: each has its own row, so these still
             * come from global memory; they are 4 bytes per row per k-block,
             * a 16th of the weight traffic, so they are not the bottleneck. */
            size_t wb0_off = (size_t)(tile_o + tid * 2) * row_bytes + q5_block * GPU_BLOCK_Q5_K_SIZE;
            const uint8_t *wb0 = w + wb0_off;
            float wd0 = gpu_fp16_to_fp32((uint16_t)(wb0[0] | (wb0[1] << 8)));
            float wdm0 = gpu_fp16_to_fp32((uint16_t)(wb0[2] | (wb0[3] << 8)));
            uint8_t sc0, mn0;
            { const uint8_t *s0 = wb0 + 4;
              if (g < 4) { sc0 = s0[g] & 63; mn0 = s0[g + 4] & 63; }
              else { sc0 = (uint8_t)((s0[g + 4] & 0xF) | ((s0[g - 4] >> 6) << 4));
                      mn0 = (uint8_t)((s0[g + 4] >> 4) | ((s0[g] >> 6) << 4)); } }
            wds0 = wd0 * (float)sc0;
            wdmn0 = wdm0 * (float)mn0;

            size_t wb1_off = (size_t)(tile_o + tid * 2 + 1) * row_bytes + q5_block * GPU_BLOCK_Q5_K_SIZE;
            const uint8_t *wb1 = w + wb1_off;
            float wd1 = gpu_fp16_to_fp32((uint16_t)(wb1[0] | (wb1[1] << 8)));
            float wdm1 = gpu_fp16_to_fp32((uint16_t)(wb1[2] | (wb1[3] << 8)));
            uint8_t sc1, mn1;
            { const uint8_t *s1 = wb1 + 4;
              if (g < 4) { sc1 = s1[g] & 63; mn1 = s1[g + 4] & 63; }
              else { sc1 = (uint8_t)((s1[g + 4] & 0xF) | ((s1[g - 4] >> 6) << 4));
                      mn1 = (uint8_t)((s1[g + 4] >> 4) | ((s1[g] >> 6) << 4)); } }
            wds1 = wd1 * (float)sc1;
            wdmn1 = wdm1 * (float)mn1;
        }

        for (int t = 0; t < n_tiles; t++) {
            int tile_s = tile_s0 + t * 16;

            int a0, a1, a2, a3;
            {
                int c0 = tid * 4;
                int c2 = c0 + 16;
                int row0 = tile_s + gid;
                int row1 = tile_s + gid + 8;
                if (row0 >= S) row0 = S - 1;
                if (row1 >= S) row1 = S - 1;
                int src0 = row0 * I + kb * 32 + c0;
                int src1 = row1 * I + kb * 32 + c0;
                int src2 = row0 * I + kb * 32 + c2;
                int src3 = row1 * I + kb * 32 + c2;
                int32_t *p;
                p = (int32_t *)(const int8_t *)(&xq[src0]); a0 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src1]); a1 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src2]); a2 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src3]); a3 = p[0];
            }

            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif

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
                const unsigned mask = 0xFFFFFFFFu;
                sq0 += gpuShflDownSync(mask, sq0, 2, 4);
                sq1 += gpuShflDownSync(mask, sq1, 2, 4);
                sq0 += gpuShflDownSync(mask, sq0, 1, 4);
                sq1 += gpuShflDownSync(mask, sq1, 1, 4);
                sq0 = gpuShflSync(mask, sq0, 0, 4);
                sq1 = gpuShflSync(mask, sq1, 0, 4);
            }
#endif

            const int row_a = tile_s + gid;
            const int row_b = tile_s + gid + 8;
            float dx0 = (row_a < S) ? xd[(size_t)row_a * n_blocks + kb] : 0.0f;
            float dx1 = (row_b < S) ? xd[(size_t)row_b * n_blocks + kb] : 0.0f;

            { float *S4 = mysum + (size_t)t * Q5KW_WARPS * 32 * 4;
              S4[0] += wds0 * dx0 * (float)d0 - wdmn0 * dx0 * (float)sq0;
              S4[2] += wds0 * dx1 * (float)d2 - wdmn0 * dx1 * (float)sq1;
              S4[1] += wds1 * dx0 * (float)d1 - wdmn1 * dx0 * (float)sq0;
              S4[3] += wds1 * dx1 * (float)d3 - wdmn1 * dx1 * (float)sq1; }
        }
        gpuSyncthreads();   /* stage is reused next k-block */
    }

    for (int t = 0; t < n_tiles; t++) {
        int tile_s = tile_s0 + t * 16;
        for (int f = 0; f < 4; f++) {
            int row = tile_s + ((f < 2) ? gid : gid + 8);
            int col = tile_o + tid * 2 + (f % 2);
            if (row < S && col < O)
                y[(size_t)row * ys + (size_t)col] = mysum[(size_t)t * Q5KW_WARPS * 32 * 4 + f];
        }
    }
}

#ifndef PICOLM_ABL_MMA_TILES
#define PICOLM_ABL_MMA_TILES 1
#endif

/* Q5_K x Q8_0 IMMA kernel, MULTI-WARP x MULTI-TILE.
 *
 * WHY: measured on the 27B at S=423, the multi-tile kernel above runs at
 * 4% of memory bandwidth AND 3.4% of int8 tensor-core peak. When neither
 * roofline is close, the kernel is occupancy/latency bound, and the arithmetic
 * says exactly why: the block is 32 threads, so an SM can never host more than
 * 32 useful threads per block x (blocks/SM). A first-generation warp-per-tile
 * GEMM needs many warps per SM to hide the load latency of the weight stream,
 * and one warp per block cannot provide them.
 *
 * WHAT THIS DOES: keeps every piece of per-tile arithmetic identical to the
 * multi-tile kernel (same m16n8k32 mma, same Q5_K nibble/high-bit decode, same
 * min-correction, same fp32 epilogue) and only changes how work is grouped:
 *
 *   block = Q5KW_WARPS warps, 32 threads each
 *   warp  w handles output columns tile_o + w*8 .. +7   (its own 8-col tile)
 *   all warps share the same Q5K_TILES_S row-tiles and walk K together
 *
 * So each block now produces Q5KW_WARPS*8 output columns x Q5K_TILES_S*16 rows
 * instead of 8 x that. Occupancy per SM rises by Q5KW_WARPS, and the row-tile
 * reuse across K (the multi-tile win) is unchanged because each warp still
 * reuses its own weight fragment across its Q5K_TILES_S tiles.
 *
 * The weight fragment cannot be shared ACROSS warps (each warp owns different
 * output columns, so it needs different weight bytes), which is why this is a
 * grouping change and not a shared-memory B-fragment change.
 */
#define Q5KW_WARPS 8

__global__ void
picolm_q5_k_q8_matmul_imma_w8(float *y, const int8_t *xq, const float *xd,
                               const void *weights, int S, int I, int O,
                               int row_bytes, int y_stride) {
    const int warp = (int)(gpuThreadIdx_x >> 5);        /* which 8-col tile */
    const int lane = (int)(gpuThreadIdx_x & 31);
    int gid = lane / 4;
    int tid = lane % 4;

    int tile_o = (int)gpuBlockIdx_x * (8 * Q5KW_WARPS) + warp * 8;
    int tile_s0 = (int)gpuBlockIdx_y * (16 * Q5K_TILES_S);
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;

#ifdef PICOLM_ABL_DEAD_KERNEL
    /* ABLATION ONLY: the whole kernel body is gone -- no loop, no loads, no
     * shared memory init. What is left is launch + scheduling + the empty
     * grid. Compare against a build where the launch is skipped entirely
     * (PICOLM_ABL_NO_Q5_LAUNCH) to separate launch cost from work. */
    if (y) y[0] = y[0];
    return;
#endif
    int n_tiles = (S - tile_s0 + 15) >> 4;
    if (n_tiles > Q5K_TILES_S) n_tiles = Q5K_TILES_S;
    if (n_tiles <= 0) return;

    /* Accumulators: one 4-float group per (thread, tile). Indexed by the
     * block-wide thread id so warps never collide. Shared memory rather than a
     * register array for the reason recorded on the multi-tile kernel: as
     * registers the array spills from 8 tiles up and the spill changes the
     * RESULT, not just the speed. */
    /* Accumulators in REGISTERS, one 4-float group per tile.
     *
     * They used to live in shared memory, updated on every k-block:
     * 16 tiles x 160 k-blocks x 4 floats = 10240 read-modify-writes per warp
     * per tensor, on the same addresses, with ~30-cycle dependent latency.
     * Measured, that was ~600 ms of the 695 ms q5_K kernel.
     *
     * Q5K_TILES_S is 4 here so float acc[4][4] = 16 registers, which fits --
     * the spill that forced shared memory only happens from 8 tiles up. */
    float acc[Q5K_TILES_S][4];
    for (int t = 0; t < Q5K_TILES_S; t++)
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;

    for (int kb = 0; kb < n_blocks; kb++) {
        int q5_block = kb / 8;
        int g = kb % 8;

        int wc = tile_o + gid;
        size_t wrow_base = (size_t)wc * row_bytes + q5_block * GPU_BLOCK_Q5_K_SIZE;
        const uint8_t *wb = w + wrow_base;

        int b0, b1;
        {
            int group = g >> 1;
            int sub = g & 1;
            const uint8_t *qs_base = wb + 48 + group * 32;
            const uint8_t *qh_base = wb + 16;
            int bit_shift = 2 * group + sub;
            uint32_t b0_val = 0, b1_val = 0;
            for (int e = 0; e < 4; e++) {
                int l = tid * 4 + e;
                uint8_t qs_byte = qs_base[l];
                uint8_t nibble = sub ? (qs_byte >> 4) : (qs_byte & 0xF);
                uint8_t high = ((qh_base[l] >> bit_shift) & 1) ? 16 : 0;
                b0_val |= (uint32_t)(nibble + high) << (e * 8);

                int l1 = 16 + tid * 4 + e;
                uint8_t qs1 = qs_base[l1];
                uint8_t nib1 = sub ? (qs1 >> 4) : (qs1 & 0xF);
                uint8_t hi1 = ((qh_base[l1] >> bit_shift) & 1) ? 16 : 0;
                b1_val |= (uint32_t)(nib1 + hi1) << (e * 8);
            }
            b0 = (int)b0_val;
            b1 = (int)b1_val;
        }

#if defined(PICOLM_ABL_STREAM_ONLY) || defined(PICOLM_ABL_W_PLUS_MMA)
        /* ABLATION: weights read and decoded exactly as the real kernel does. */
        acc[0][0] += (float)(b0 + b1);
#ifndef PICOLM_ABL_W_PLUS_MMA
        continue;
#endif
#ifdef PICOLM_ABL_W_PLUS_MMA
        /* add back ONLY the mma, with fixed A fragments and unit scales.
         * PICOLM_ABL_MMA_TILES=<n> sets how many of the n_tiles are done, so
         * the per-tile cost can be measured instead of guessed: one tile is
         * 1/16th of the real inner loop. */
        for (int tt = 0; tt < PICOLM_ABL_MMA_TILES; tt++) {
            int a0x, a1x, a2x, a3x;
#ifdef PICOLM_ABL_PLUS_AGATHER
            /* ALSO do the real A gather (the 4 scattered int32 loads) */
            {
                int tile_s = tile_s0 + tt * 16;
                int c0 = tid * 4;
                int c2 = c0 + 16;
                int row0 = tile_s + gid;
                int row1 = tile_s + gid + 8;
                if (row0 >= S) row0 = S - 1;
                if (row1 >= S) row1 = S - 1;
                int32_t *p;
                p = (int32_t *)(const int8_t *)(&xq[row0 * I + kb * 32 + c0]); a0x = p[0];
                p = (int32_t *)(const int8_t *)(&xq[row1 * I + kb * 32 + c0]); a1x = p[0];
                p = (int32_t *)(const int8_t *)(&xq[row0 * I + kb * 32 + c2]); a2x = p[0];
                p = (int32_t *)(const int8_t *)(&xq[row1 * I + kb * 32 + c2]); a3x = p[0];
            }
#else
            a0x = 0x04030201; a1x = 0x04030201; a2x = 0x04030201; a3x = 0x04030201;
#endif
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                : "r"(a0x), "r"(a1x), "r"(a2x), "r"(a3x), "r"(b0), "r"(b1));
#endif
            acc[tt][0] += (float)(d0 + d1 + d2 + d3);
        }
        continue;
#endif
#endif

        float wds0, wdmn0, wds1, wdmn1;
        {
            size_t wb0_off = (size_t)(tile_o + tid * 2) * row_bytes + q5_block * GPU_BLOCK_Q5_K_SIZE;
            const uint8_t *wb0 = w + wb0_off;
            float wd0 = gpu_fp16_to_fp32((uint16_t)(wb0[0] | (wb0[1] << 8)));
            float wdm0 = gpu_fp16_to_fp32((uint16_t)(wb0[2] | (wb0[3] << 8)));
            uint8_t sc0, mn0;
            { const uint8_t *s0 = wb0 + 4;
              if (g < 4) { sc0 = s0[g] & 63; mn0 = s0[g + 4] & 63; }
              else { sc0 = (uint8_t)((s0[g + 4] & 0xF) | ((s0[g - 4] >> 6) << 4));
                      mn0 = (uint8_t)((s0[g + 4] >> 4) | ((s0[g] >> 6) << 4)); } }
            wds0 = wd0 * (float)sc0;
            wdmn0 = wdm0 * (float)mn0;

            size_t wb1_off = (size_t)(tile_o + tid * 2 + 1) * row_bytes + q5_block * GPU_BLOCK_Q5_K_SIZE;
            const uint8_t *wb1 = w + wb1_off;
            float wd1 = gpu_fp16_to_fp32((uint16_t)(wb1[0] | (wb1[1] << 8)));
            float wdm1 = gpu_fp16_to_fp32((uint16_t)(wb1[2] | (wb1[3] << 8)));
            uint8_t sc1, mn1;
            { const uint8_t *s1 = wb1 + 4;
              if (g < 4) { sc1 = s1[g] & 63; mn1 = s1[g + 4] & 63; }
              else { sc1 = (uint8_t)((s1[g + 4] & 0xF) | ((s1[g - 4] >> 6) << 4));
                      mn1 = (uint8_t)((s1[g + 4] >> 4) | ((s1[g] >> 6) << 4)); } }
            wds1 = wd1 * (float)sc1;
            wdmn1 = wdm1 * (float)mn1;
        }

        for (int t = 0; t < n_tiles; t++) {
            int tile_s = tile_s0 + t * 16;

            int a0, a1, a2, a3;
            {
                int c0 = tid * 4;
                int c2 = c0 + 16;
                int row0 = tile_s + gid;
                int row1 = tile_s + gid + 8;
                if (row0 >= S) row0 = S - 1;
                if (row1 >= S) row1 = S - 1;
                int src0 = row0 * I + kb * 32 + c0;
                int src1 = row1 * I + kb * 32 + c0;
                int src2 = row0 * I + kb * 32 + c2;
                int src3 = row1 * I + kb * 32 + c2;
                int32_t *p;
                p = (int32_t *)(const int8_t *)(&xq[src0]); a0 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src1]); a1 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src2]); a2 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src3]); a3 = p[0];
            }

            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif

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
                const unsigned mask = 0xFFFFFFFFu;
                sq0 += gpuShflDownSync(mask, sq0, 2, 4);
                sq1 += gpuShflDownSync(mask, sq1, 2, 4);
                sq0 += gpuShflDownSync(mask, sq0, 1, 4);
                sq1 += gpuShflDownSync(mask, sq1, 1, 4);
                sq0 = gpuShflSync(mask, sq0, 0, 4);
                sq1 = gpuShflSync(mask, sq1, 0, 4);
            }
#endif

            const int row_a = tile_s + gid;
            const int row_b = tile_s + gid + 8;
            float dx0 = (row_a < S) ? xd[(size_t)row_a * n_blocks + kb] : 0.0f;
            float dx1 = (row_b < S) ? xd[(size_t)row_b * n_blocks + kb] : 0.0f;

            acc[t][0] += wds0 * dx0 * (float)d0 - wdmn0 * dx0 * (float)sq0;
            acc[t][2] += wds0 * dx1 * (float)d2 - wdmn0 * dx1 * (float)sq1;
            acc[t][1] += wds1 * dx0 * (float)d1 - wdmn1 * dx0 * (float)sq0;
            acc[t][3] += wds1 * dx1 * (float)d3 - wdmn1 * dx1 * (float)sq1;
        }
    }

    for (int t = 0; t < n_tiles; t++) {
        int tile_s = tile_s0 + t * 16;
        for (int f = 0; f < 4; f++) {
            int row = tile_s + ((f < 2) ? gid : gid + 8);
            int col = tile_o + tid * 2 + (f % 2);
            if (row < S && col < O)
                y[(size_t)row * ys + (size_t)col] = acc[t][f];
        }
    }
}

/* NEGATIVE RESULT, recorded so it is not retried: shared-memory staging for
 * Q5_K LOSES, and clearly.
 *
 * A staged kernel was written to the same shape as the one that works for
 * q8_0 (picolm_q8_q8_matmul_imma_smw16: 16 columns x 16 rows per block,
 * coalesced per-column staging, 64 threads). q8_0 moves 32 MB/ms and unstaged
 * q5_K only 24, which looked like a 1.33x to be had.
 *
 * It is not. Measured on the 27B at 423 tokens:
 *     unstaged (imma_sb)                     360-394 tok/s
 *     staged, 1 k-step per barrier           216 tok/s
 *     staged, 2 k-steps per barrier          191 tok/s
 *     staged, 4 k-steps per barrier          181 tok/s
 *     staged, 8 k-steps per barrier          150 tok/s
 * Correct at every setting (max|logit diff| 2.72, argmax same), just slow.
 *
 * WHY: a block_q5_K is 176 bytes against block_q8_0's 34. The same staging
 * geometry therefore needs 5x the shared memory, and it is paid twice --
 *   * occupancy: 16 cols x 8 x 176 = 22.5 KB per block caps an SM at 2 blocks,
 *     i.e. 128 of 2048 thread slots = 6%;
 *   * barriers: staging one k-step at a time means 160 __syncthreads() per
 *     block, and batching k-steps to amortise them only makes the shared
 *     memory worse.
 * The unstaged imma_sb kernel uses no shared memory, runs 256 threads, and has
 * no barrier inside the k loop at all.
 *
 * So q8_0's advantage is NOT "it stages". It is that a 34-byte block is cheap
 * to stage. The conclusion for q5_K is the opposite of the hypothesis: keep it
 * unstructured, and look elsewhere. */
/* Q5_K x Q8_0 IMMA kernel, SUPERBLOCK-UNROLLED inner loop.
 *
 * WHY: after moving the accumulators to registers the q5_K kernel ran at
 * 28 TFLOP/s = 4% of the card's int8 peak, with the weight stream, the A
 * gather, the min-correction and the scale decode all measuring as nearly free
 * alongside it. What is left is the shape of the inner loop: per k-block it
 * issues ONE mma and then ~15 dependent scalar operations around it, and it
 * does that 8 times per Q5_K super-block, so the tensor core alternates with
 * scalar work instead of running in bursts.
 *
 * This kernel keeps the per-output arithmetic identical but restructures the
 * k loop: for one super-block (the 8 sub-blocks g = 0..7) it decodes all eight
 * weight fragments, issues the eight mma back to back into eight INDEPENDENT
 * accumulator sets, and only then runs the eight epilogues. Independent
 * accumulators mean no dependency chain between the mma, so the scheduler can
 * put real work behind each one.
 *
 * Registers: acc[8][4] = 32 floats + d[8][4] = 32 ints + fragments ~ 90-110,
 * comfortably inside the 255 budget at 256 threads/block.
 *
 * The Q5_K block (256 values, 176 bytes) holds exactly 8 sub-blocks of 32
 * values, which is why g unrolls to 8 with no remainder.
 */
__global__ void
picolm_q5_k_q8_matmul_imma_sb(float *y, const int8_t *xq, const float *xd,
                               const void *weights, int S, int I, int O,
                               int row_bytes, int y_stride) {
    const int warp = (int)(gpuThreadIdx_x >> 5);
    const int lane = (int)(gpuThreadIdx_x & 31);
    int gid = lane / 4;
    int tid = lane % 4;

    int tile_o = (int)gpuBlockIdx_x * (8 * Q5KW_WARPS) + warp * 8;
    int tile_s0 = (int)gpuBlockIdx_y * (16 * Q5K_TILES_S);
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    int n_super  = n_blocks / 8;               /* Q5_K super-blocks */
    const uint8_t *w = (const uint8_t *)weights;

    int n_tiles = (S - tile_s0 + 15) >> 4;
    if (n_tiles > Q5K_TILES_S) n_tiles = Q5K_TILES_S;
    if (n_tiles <= 0) return;

    float acc[Q5K_TILES_S][4];
    for (int t = 0; t < Q5K_TILES_S; t++)
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;

    int wc = tile_o + gid;

    for (int sb = 0; sb < n_super; sb++) {
        size_t base = (size_t)wc * row_bytes + (size_t)sb * GPU_BLOCK_Q5_K_SIZE;
        const uint8_t *wb = w + base;

        /* ---- decode all 8 sub-block weight fragments first ---- */
        int bb0[8], bb1[8];
#pragma unroll
        for (int g = 0; g < 8; g++) {
            int group = g >> 1;
            int sub = g & 1;
            const uint8_t *qs_base = wb + 48 + group * 32;
            const uint8_t *qh_base = wb + 16;
            int bit_shift = 2 * group + sub;
            uint32_t b0_val = 0, b1_val = 0;
            for (int e = 0; e < 4; e++) {
                int l = tid * 4 + e;
                uint8_t qs_byte = qs_base[l];
                uint8_t nibble = sub ? (qs_byte >> 4) : (qs_byte & 0xF);
                uint8_t high = ((qh_base[l] >> bit_shift) & 1) ? 16 : 0;
                b0_val |= (uint32_t)(nibble + high) << (e * 8);

                int l1 = 16 + tid * 4 + e;
                uint8_t qs1 = qs_base[l1];
                uint8_t nib1 = sub ? (qs1 >> 4) : (qs1 & 0xF);
                uint8_t hi1 = ((qh_base[l1] >> bit_shift) & 1) ? 16 : 0;
                b1_val |= (uint32_t)(nib1 + hi1) << (e * 8);
            }
            bb0[g] = (int)b0_val;
            bb1[g] = (int)b1_val;
        }

        /* ---- per-output-column scales for all 8 sub-blocks ---- */
        float s0a[8], s0m[8], s1a[8], s1m[8];
        {
            size_t o0 = (size_t)(tile_o + tid * 2) * row_bytes + (size_t)sb * GPU_BLOCK_Q5_K_SIZE;
            const uint8_t *wb0 = w + o0;
            float wd0  = gpu_fp16_to_fp32((uint16_t)(wb0[0] | (wb0[1] << 8)));
            float wdm0 = gpu_fp16_to_fp32((uint16_t)(wb0[2] | (wb0[3] << 8)));
            size_t o1 = (size_t)(tile_o + tid * 2 + 1) * row_bytes + (size_t)sb * GPU_BLOCK_Q5_K_SIZE;
            const uint8_t *wb1 = w + o1;
            float wd1  = gpu_fp16_to_fp32((uint16_t)(wb1[0] | (wb1[1] << 8)));
            float wdm1 = gpu_fp16_to_fp32((uint16_t)(wb1[2] | (wb1[3] << 8)));
            const uint8_t *p0 = wb0 + 4;
            const uint8_t *p1 = wb1 + 4;
#pragma unroll
            for (int g = 0; g < 8; g++) {
                uint8_t sc0, mn0, sc1, mn1;
                if (g < 4) {
                    sc0 = p0[g] & 63; mn0 = p0[g + 4] & 63;
                    sc1 = p1[g] & 63; mn1 = p1[g + 4] & 63;
                } else {
                    sc0 = (uint8_t)((p0[g + 4] & 0xF) | ((p0[g - 4] >> 6) << 4));
                    mn0 = (uint8_t)((p0[g + 4] >> 4) | ((p0[g] >> 6) << 4));
                    sc1 = (uint8_t)((p1[g + 4] & 0xF) | ((p1[g - 4] >> 6) << 4));
                    mn1 = (uint8_t)((p1[g + 4] >> 4) | ((p1[g] >> 6) << 4));
                }
                s0a[g] = wd0 * (float)sc0;  s0m[g] = wdm0 * (float)mn0;
                s1a[g] = wd1 * (float)sc1;  s1m[g] = wdm1 * (float)mn1;
            }
        }

        for (int t = 0; t < n_tiles; t++) {
            int tile_s = tile_s0 + t * 16;
            int kb0 = sb * 8;

            /* ---- broadcast this tile's xd values across the warp ----
             * Every lane needs xd[tile_s + gid + {0,8}][kb] for each of the 8
             * sub-blocks, and gid is just lane/4, so the whole 16-row tile only
             * needs 16 distinct floats per kb -- but the loads as written were
             * one per lane, strided nb*4 = 640 bytes apart, i.e. 8 cache lines
             * per 32 useful bytes, repeated for every tile and k-block.
             * Dropping them entirely measured 491 -> 344 ms, so they were 30%
             * of the kernel.
             *
             * Now lanes 0..15 each load ONE value (contiguous within a kb
             * column: they are tile_s+0..15, stride nb, so still strided, but
             * 16 loads per kb instead of 8 per kb per lane) and the result is
             * broadcast with shuffles. That is 16 global loads per (tile, kb)
             * for the whole warp instead of 64. */
            /* Load this lane's xd values for all 8 sub-blocks of the
             * super-block, straight into registers.
             *
             * The original code loaded xd[row][kb] inside the tile loop and
             * the epilogue, so every lane issued 2 loads per (tile, k-block)
             * at addresses nb*4 = 640 bytes apart -- 8 cache lines per 32
             * useful bytes. Removing them entirely measured 491 -> 344 ms.
             *
             * Here each lane loads the 8 k-blocks of its own two rows
             * (gid and gid+8) as one run of 8 CONSECUTIVE floats from each
             * row, i.e. xd[row*nb + kb0 .. +7]: 2 contiguous 32-byte reads per
             * lane instead of 16 scattered 4-byte ones, and no shuffles. */
            float xdb[8][2];
            {
                int ra = tile_s + gid;
                int rb = tile_s + gid + 8;
                if (ra < S) {
                    const float *pa = xd + (size_t)ra * n_blocks + kb0;
                    for (int q = 0; q < 8; q++) xdb[q][0] = pa[q];
                } else {
                    for (int q = 0; q < 8; q++) xdb[q][0] = 0.0f;
                }
                if (rb < S) {
                    const float *pb = xd + (size_t)rb * n_blocks + kb0;
                    for (int q = 0; q < 8; q++) xdb[q][1] = pb[q];
                } else {
                    for (int q = 0; q < 8; q++) xdb[q][1] = 0.0f;
                }
            }

            /* ---- 8 back-to-back mma with independent accumulators ---- */
            int d0[8], d1[8], d2[8], d3[8];
            int q0[8], q1[8];
#pragma unroll
            for (int g = 0; g < 8; g++) { d0[g] = d1[g] = d2[g] = d3[g] = 0; q0[g] = q1[g] = 0; }

#pragma unroll
            for (int g = 0; g < 8; g++) {
                int kb = kb0 + g;
                int a0, a1, a2, a3;
                {
                    int c0 = tid * 4;
                    int c2 = c0 + 16;
                    int row0 = tile_s + gid;
                    int row1 = tile_s + gid + 8;
                    if (row0 >= S) row0 = S - 1;
                    if (row1 >= S) row1 = S - 1;
                    int32_t *p;
                    p = (int32_t *)(const int8_t *)(&xq[row0 * I + kb * 32 + c0]); a0 = p[0];
                    p = (int32_t *)(const int8_t *)(&xq[row1 * I + kb * 32 + c0]); a1 = p[0];
                    p = (int32_t *)(const int8_t *)(&xq[row0 * I + kb * 32 + c2]); a2 = p[0];
                    p = (int32_t *)(const int8_t *)(&xq[row1 * I + kb * 32 + c2]); a3 = p[0];
                }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(d0[g]), "+r"(d1[g]), "+r"(d2[g]), "+r"(d3[g])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0[g]), "r"(bb1[g]));
#endif
#ifdef PICOLM_ABL_NO_SHUF
                { q0[g] = 0; q1[g] = 0; }
#else
                {
                    int s0v = 0, s1v = 0;
                    int8_t *av;
                    av = (int8_t *)&a0; s0v += av[0] + av[1] + av[2] + av[3];
                    av = (int8_t *)&a2; s0v += av[0] + av[1] + av[2] + av[3];
                    av = (int8_t *)&a1; s1v += av[0] + av[1] + av[2] + av[3];
                    av = (int8_t *)&a3; s1v += av[0] + av[1] + av[2] + av[3];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                    const unsigned mask = 0xFFFFFFFFu;
                    s0v += gpuShflDownSync(mask, s0v, 2, 4);
                    s1v += gpuShflDownSync(mask, s1v, 2, 4);
                    s0v += gpuShflDownSync(mask, s0v, 1, 4);
                    s1v += gpuShflDownSync(mask, s1v, 1, 4);
                    s0v = gpuShflSync(mask, s0v, 0, 4);
                    s1v = gpuShflSync(mask, s1v, 0, 4);
#endif
                    q0[g] = s0v; q1[g] = s1v;
                }
#endif
            }

            /* ---- epilogue: same arithmetic and order as the non-unrolled
             * kernel, per sub-block, accumulating into this tile's registers ---- */
            const int row_a = tile_s + gid;
            const int row_b = tile_s + gid + 8;
#pragma unroll
            for (int g = 0; g < 8; g++) {
                int kb = kb0 + g;
#ifdef PICOLM_ABL_NO_XD
                float dx0 = 1.0f, dx1 = 1.0f;   /* ABLATION: drop the epilogue loads */
#else
                /* gid-indexed values come from the per-tile broadcast above.
                 * NOTE this is only valid because kb == kb0 in this loop: the
                 * sb kernel groups the 8 sub-blocks of one Q5_K super-block,
                 * and xd is indexed by kb, so values loaded for kb0 do NOT
                 * serve kb0+1..7. Keep the real load for those. */
                float dx0 = xdb[g][0];
                float dx1 = xdb[g][1];
#endif
                acc[t][0] += s0a[g] * dx0 * (float)d0[g] - s0m[g] * dx0 * (float)q0[g];
                acc[t][2] += s0a[g] * dx1 * (float)d2[g] - s0m[g] * dx1 * (float)q1[g];
                acc[t][1] += s1a[g] * dx0 * (float)d1[g] - s1m[g] * dx0 * (float)q0[g];
                acc[t][3] += s1a[g] * dx1 * (float)d3[g] - s1m[g] * dx1 * (float)q1[g];
            }
        }
    }

    for (int t = 0; t < n_tiles; t++) {
        int tile_s = tile_s0 + t * 16;
        for (int f = 0; f < 4; f++) {
            int row = tile_s + ((f < 2) ? gid : gid + 8);
            int col = tile_o + tid * 2 + (f % 2);
            if (row < S && col < O)
                y[(size_t)row * ys + (size_t)col] = acc[t][f];
        }
    }
}

/* Q5_K x Q8_0 IMMA kernel, MULTI-TILE (TILES_S row-tiles of 16 per block).
 *
 * Why this exists: the single-tile kernel above (picolm_q5_k_q8_matmul_imma)
 * computes ONE 16x8 output tile per warp and streams the entire K extent of
 * those 8 output rows to do it. The weight bytes are therefore re-read once
 * per 16-token row-block: for a 512-token prefill chunk that is a 32x read
 * amplification, and it is measurable as "kernel time scales exactly linearly
 * with token count while the per-call time doubles as the ubatch doubles"
 * (measured: tokens 128/256/512 -> 1.03/2.10/4.27 ms per call, total time flat
 * at ~1040 ms). Prefill was moving 317 GB per chunk instead of 11.7 GB, at a
 * healthy 315 GB/s -- i.e. bandwidth-bound on 27x too much traffic, 2.3% of
 * int8 tensor-core peak.
 *
 * This kernel keeps the SAME per-tile arithmetic (same m16n8k32 mma, same
 * nibble/high-bit decode as the reference, same min-correction), but decodes
 * the weight fragment ONCE per k-block and reuses it for TILES_S independent
 * 16-row accumulators. That cuts the weight traffic by TILES_S with no change
 * to what any individual output element computes.
 *
 * Registers: sum[] is TILES_S*4 floats. TILES_S=4 -> 16 accumulators plus the
 * 4 activation fragments and 2 weight fragments per tile set, which stays well
 * inside the 255-register budget at 32 threads/block.
 */
__global__ void
picolm_q5_k_q8_matmul_imma_mt(float *y, const int8_t *xq, const float *xd,
                               const void *weights, int S, int I, int O,
                               int row_bytes, int y_stride) {
    int gid = gpuThreadIdx_x / 4;
    int tid = gpuThreadIdx_x % 4;
    int tile_o = gpuBlockIdx_x * 8;
    int tile_s0 = gpuBlockIdx_y * (16 * Q5K_TILES_S);
    int ys = y_stride > 0 ? y_stride : O;
    int n_blocks = I / 32;
    const uint8_t *w = (const uint8_t *)weights;

    /* How many of the Q5K_TILES_S tiles this block actually owns. The last
     * block in y may own fewer, and for a short ubatch (a 2-token chunk tail,
     * say) that is most of them; skipping them here avoids both the wasted
     * accumulator work and the clamped-row loads below. */
    int n_tiles = (S - tile_s0 + 15) >> 4;
    if (n_tiles > Q5K_TILES_S) n_tiles = Q5K_TILES_S;
    if (n_tiles <= 0) return;

    /* Accumulators live in SHARED memory, not registers.
     *
     * As a register array this kernel is correct up to Q5K_TILES_S=4 and
     * silently WRONG at 8: sum[8][4] is 32 live floats per thread before the
     * 4 activation fragments, 2 weight fragments and address arithmetic, so
     * ptxas spills and the result changes (measured: 2 and 4 tiles match the
     * CPU bit-for-bit, 8 tiles diverges with a different argmax). Staging them
     * in shared memory removes the ceiling: 32 threads x 4 floats per tile is
     * only 512 bytes for 8 tiles.
     *
     * Layout: sum_sh[(t * 32 + tid) * 4 + f] -- per-thread, so the hot path
     * still reads/writes its own slice and needs no synchronisation. */
    extern __shared__ float sum_sh[];
    for (int i = gpuThreadIdx_x; i < Q5K_TILES_S * 32 * 4; i += gpuBlockDim_x)
        sum_sh[i] = 0.0f;
    gpuSyncthreads();
    float *mysum = sum_sh + (size_t)gpuThreadIdx_x * 4;   /* stride 32*4 per tile */

    for (int kb = 0; kb < n_blocks; kb++) {
        int q5_block = kb / 8;
        int g = kb % 8;

        /* ---- WEIGHT fragment: decoded once, reused by every tile ---- */
        int wc = tile_o + gid;
        size_t wrow_base = (size_t)wc * row_bytes + q5_block * GPU_BLOCK_Q5_K_SIZE;
        const uint8_t *wb = w + wrow_base;

        int b0, b1;
        {
            int group = g >> 1;
            int sub = g & 1;
            const uint8_t *qs_base = wb + 48 + group * 32;
            const uint8_t *qh_base = wb + 16;
            int bit_shift = 2 * group + sub;

            uint32_t b0_val = 0, b1_val = 0;
            for (int e = 0; e < 4; e++) {
                int l = tid * 4 + e;
                uint8_t qs_byte = qs_base[l];
                uint8_t nibble = sub ? (qs_byte >> 4) : (qs_byte & 0xF);
                uint8_t high = ((qh_base[l] >> bit_shift) & 1) ? 16 : 0;
                b0_val |= (uint32_t)(nibble + high) << (e * 8);

                int l1 = 16 + tid * 4 + e;
                uint8_t qs1 = qs_base[l1];
                uint8_t nib1 = sub ? (qs1 >> 4) : (qs1 & 0xF);
                uint8_t hi1 = ((qh_base[l1] >> bit_shift) & 1) ? 16 : 0;
                b1_val |= (uint32_t)(nib1 + hi1) << (e * 8);
            }
            b0 = (int)b0_val;
            b1 = (int)b1_val;
        }

        /* ---- per-output-column scales, also independent of the tile ---- */
        float wds0, wdmn0, wds1, wdmn1;
        {
            size_t wb0_off = (size_t)(tile_o + tid * 2) * row_bytes + q5_block * GPU_BLOCK_Q5_K_SIZE;
            const uint8_t *wb0 = w + wb0_off;
            float wd0 = gpu_fp16_to_fp32((uint16_t)(wb0[0] | (wb0[1] << 8)));
            float wdm0 = gpu_fp16_to_fp32((uint16_t)(wb0[2] | (wb0[3] << 8)));
            uint8_t sc0, mn0;
            { const uint8_t *s0 = wb0 + 4;
              if (g < 4) { sc0 = s0[g] & 63; mn0 = s0[g + 4] & 63; }
              else { sc0 = (uint8_t)((s0[g + 4] & 0xF) | ((s0[g - 4] >> 6) << 4));
                      mn0 = (uint8_t)((s0[g + 4] >> 4) | ((s0[g] >> 6) << 4)); } }
            wds0 = wd0 * (float)sc0;
            wdmn0 = wdm0 * (float)mn0;

            size_t wb1_off = (size_t)(tile_o + tid * 2 + 1) * row_bytes + q5_block * GPU_BLOCK_Q5_K_SIZE;
            const uint8_t *wb1 = w + wb1_off;
            float wd1 = gpu_fp16_to_fp32((uint16_t)(wb1[0] | (wb1[1] << 8)));
            float wdm1 = gpu_fp16_to_fp32((uint16_t)(wb1[2] | (wb1[3] << 8)));
            uint8_t sc1, mn1;
            { const uint8_t *s1 = wb1 + 4;
              if (g < 4) { sc1 = s1[g] & 63; mn1 = s1[g + 4] & 63; }
              else { sc1 = (uint8_t)((s1[g + 4] & 0xF) | ((s1[g - 4] >> 6) << 4));
                      mn1 = (uint8_t)((s1[g + 4] >> 4) | ((s1[g] >> 6) << 4)); } }
            wds1 = wd1 * (float)sc1;
            wdmn1 = wdm1 * (float)mn1;
        }

        /* ---- one pass per 16-row tile, sharing b0/b1/scales above ---- */
        for (int t = 0; t < n_tiles; t++) {
            int tile_s = tile_s0 + t * 16;

            int a0, a1, a2, a3;
            {
                int c0 = tid * 4;
                int c2 = c0 + 16;
                int row0 = tile_s + gid;
                int row1 = tile_s + gid + 8;
                if (row0 >= S) row0 = S - 1;   /* clamp: reads stay in the S rows */
                if (row1 >= S) row1 = S - 1;
                int src0 = row0 * I + kb * 32 + c0;
                int src1 = row1 * I + kb * 32 + c0;
                int src2 = row0 * I + kb * 32 + c2;
                int src3 = row1 * I + kb * 32 + c2;
                int32_t *p;
                p = (int32_t *)(const int8_t *)(&xq[src0]); a0 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src1]); a1 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src2]); a2 = p[0];
                p = (int32_t *)(const int8_t *)(&xq[src3]); a3 = p[0];
            }

            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif

            int sq0 = 0, sq1 = 0;
            {
                int8_t *av;
                av = (int8_t *)&a0; sq0 += av[0] + av[1] + av[2] + av[3];
                av = (int8_t *)&a2; sq0 += av[0] + av[1] + av[2] + av[3];
                av = (int8_t *)&a1; sq1 += av[0] + av[1] + av[2] + av[3];
                av = (int8_t *)&a3; sq1 += av[0] + av[1] + av[2] + av[3];
            }
#ifdef PICOLM_ABL_NO_MINCORR2
            /* ABLATION: drop 16 byte adds + 6 warp shuffles per tile per k-block. */
            (void)sq0; (void)sq1;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
            {
                const unsigned mask = 0xFFFFFFFFu;
                sq0 += gpuShflDownSync(mask, sq0, 2, 4);
                sq1 += gpuShflDownSync(mask, sq1, 2, 4);
                sq0 += gpuShflDownSync(mask, sq0, 1, 4);
                sq1 += gpuShflDownSync(mask, sq1, 1, 4);
                sq0 = gpuShflSync(mask, sq0, 0, 4);
                sq1 = gpuShflSync(mask, sq1, 0, 4);
            }
#endif

            const int row_a = tile_s + gid;
            const int row_b = tile_s + gid + 8;
            float dx0 = (row_a < S) ? xd[(size_t)row_a * n_blocks + kb] : 0.0f;
            float dx1 = (row_b < S) ? xd[(size_t)row_b * n_blocks + kb] : 0.0f;

            { float *S4 = mysum + (size_t)t * 32 * 4;
              S4[0] += wds0 * dx0 * (float)d0 - wdmn0 * dx0 * (float)sq0;
              S4[2] += wds0 * dx1 * (float)d2 - wdmn0 * dx1 * (float)sq1;
              S4[1] += wds1 * dx0 * (float)d1 - wdmn1 * dx0 * (float)sq0;
              S4[3] += wds1 * dx1 * (float)d3 - wdmn1 * dx1 * (float)sq1; }
        }
    }

    for (int t = 0; t < n_tiles; t++) {
        int tile_s = tile_s0 + t * 16;
        for (int f = 0; f < 4; f++) {
            int row = tile_s + ((f < 2) ? gid : gid + 8);
            int col = tile_o + tid * 2 + (f % 2);
            if (row < S && col < O)
                y[(size_t)row * ys + (size_t)col] = mysum[(size_t)t * 32 * 4 + f];
        }
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
            /* Reduce over the 4 lanes of this group (lanes gid*4..gid*4+3)
             * and broadcast the total to all of them.
             *
             * The mask must name the whole warp. All 32 lanes execute these
             * calls, and __shfl_*_sync requires that every lane in the mask
             * takes part; a per-group mask (0xF << gid*4) makes lanes holding
             * DIFFERENT masks join the same collective, which is undefined
             * behaviour. Grouping is expressed by WIDTH=4 instead: that splits
             * the warp into 8 independent 4-lane sub-groups and makes both the
             * down-shuffle and the broadcast group-relative, so srcLane 0 means
             * "lane 0 of this group" rather than lane 0 of the warp. */
            const unsigned mask = 0xFFFFFFFFu;
            sq0 += gpuShflDownSync(mask, sq0, 2, 4);
            sq1 += gpuShflDownSync(mask, sq1, 2, 4);
            sq0 += gpuShflDownSync(mask, sq0, 1, 4);
            sq1 += gpuShflDownSync(mask, sq1, 1, 4);
            sq0 = gpuShflSync(mask, sq0, 0, 4);
            sq1 = gpuShflSync(mask, sq1, 0, 4);
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
            /* Same fix as the other K-quant kernels: the mask must name the
             * whole warp (all 32 lanes execute this collective), and the 4-lane
             * grouping is expressed by WIDTH=4 so srcLane 0 means "lane 0 of
             * this group" rather than warp lane 0. */
            const unsigned mask = 0xFFFFFFFFu;
            sq0a += gpuShflDownSync(mask, sq0a, 2, 4);
            sq0b += gpuShflDownSync(mask, sq0b, 2, 4);
            sq1a += gpuShflDownSync(mask, sq1a, 2, 4);
            sq1b += gpuShflDownSync(mask, sq1b, 2, 4);
            sq0a += gpuShflDownSync(mask, sq0a, 1, 4);
            sq0b += gpuShflDownSync(mask, sq0b, 1, 4);
            sq1a += gpuShflDownSync(mask, sq1a, 1, 4);
            sq1b += gpuShflDownSync(mask, sq1b, 1, 4);
            sq0a = gpuShflSync(mask, sq0a, 0, 4);
            sq0b = gpuShflSync(mask, sq0b, 0, 4);
            sq1a = gpuShflSync(mask, sq1a, 0, 4);
            sq1b = gpuShflSync(mask, sq1b, 0, 4);
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
        size_t kv_head_stride_bytes, size_t kv_head_stride_v_bytes,
        int n_swa, int kv_type)
{
    int kv_h = (int)gpuBlockIdx_x;
    if (kv_h >= n_kv_heads) return;

    int kv_mul = n_heads / n_kv_heads;
    int first_qh = kv_h * kv_mul;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    size_t layer_base = (size_t)layer_ordinal * max_seq_len * kv_pos_stride_bytes;

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

    float attn_scale = (picolm_gpu_d_attn_scale > 0.0f) ? picolm_gpu_d_attn_scale : 1.0f / sqrtf((float)head_dim);

    /* Sliding window: restrict the scan to [max(0,pos-n_swa+1), pos] when
     * this layer is SWA (n_swa > 0). n_swa == 0 is the pre-SWA, full
     * causal attention behavior (t_start stays 0). Mirrors attn_core()
     * in model_attention.c exactly, so CPU and GPU decode agree. */
    int t_start = (n_swa > 0) ? (pos - n_swa + 1) : 0;
    if (t_start < 0) t_start = 0;

    for (int t = t_start; t <= pos; t++) {
        /* Byte offsets, so the same arithmetic works for F16 and for the
         * block-quantized formats (whose per-head stride is not 2*head_dim). */
        size_t row_off = layer_base + (size_t)t * kv_pos_stride_bytes + (size_t)kv_h * kv_head_stride_bytes;
        size_t v_off = row_off - (size_t)kv_h * kv_head_stride_bytes + (size_t)kv_h * kv_head_stride_v_bytes;
        kv_row_load_f16((const uint8_t *)kv_k, row_off, kv_type, head_dim, tid, n_threads, k_sh);
        kv_row_load_f16((const uint8_t *)kv_v, v_off, kv_type, head_dim, tid, n_threads, v_sh);
        gpuSyncthreads();

        for (int g = 0; g < kv_mul; g++) {
            const float *qg = q_dev + (size_t)(first_qh + g) * head_dim;
            float score;
            if (tid == 0) {
                /* Q@K dot product: 16-element chunks + tail (handles head_dim < 16) */
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
                /* Handle tail: elements not covered by complete 16-element chunks */
                float tail = 0;
                for (int d = n_chunks * 16; d < head_dim; d++) {
                    tail = fmaf(qg[d], gpu_fp16_to_fp32(k_sh[d]), tail);
                }
                score = (chunk[0] + tail) * attn_scale;
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
        size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes, size_t kv_head_stride_v_bytes,
        int n_splits, int chunk_size, int n_swa, int kv_type)
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

    /* Sliding window: raise this split's floor to the window start. A
     * split entirely below the window ends up with t0 >= t1 and simply
     * contributes an empty (max=-1e30, sum=0) partial result, which
     * picolm_gpu_attention_decode_merge_kernel already treats as
     * zero-weight -- exp(partial_max - global_max) underflows to 0.0f
     * when partial_max is left at its -1e30 sentinel. */
    if (n_swa > 0) {
        int win_start = pos - n_swa + 1;
        if (win_start < 0) win_start = 0;
        if (t0 < win_start) t0 = win_start;
    }

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
        size_t layer_base = (size_t)layer_ordinal * max_seq_len * kv_pos_stride_bytes;
        float attn_scale = (picolm_gpu_d_attn_scale > 0.0f) ? picolm_gpu_d_attn_scale : 1.0f / sqrtf((float)head_dim);

        for (int t = t0; t < t1; t++) {
            size_t row_off = layer_base + (size_t)t * kv_pos_stride_bytes + (size_t)kv_h * kv_head_stride_bytes;
            size_t v_off = row_off - (size_t)kv_h * kv_head_stride_bytes + (size_t)kv_h * kv_head_stride_v_bytes;
            kv_row_load_f16((const uint8_t *)kv_k, row_off, kv_type, head_dim, tid, n_threads, k_sh);
            kv_row_load_f16((const uint8_t *)kv_v, v_off, kv_type, head_dim, tid, n_threads, v_sh);
            gpuSyncthreads();

            for (int g = 0; g < kv_mul; g++) {
                const float *qg = q_dev + (size_t)(first_qh + g) * head_dim;
                /* QK dot, WARP-PARALLEL.
                 *
                 * This used to run entirely on `tid == 0`: for the 27B that is
                 * head_dim(256) * kv_mul(6) = 1536 serial FMAs per key while
                 * the other 255 threads of the block waited on the barrier
                 * after it. Measured 14.4 us per key at 9848 context, i.e.
                 * ~48000x off memory bandwidth, and decode attention was 68%
                 * of the whole decode token.
                 *
                 * Now each of the 32 lanes of the first warp handles one
                 * 16-element chunk (chunk c -> lane c) and the per-lane
                 * partial is tree-reduced across the warp. With head_dim 256
                 * that is n_chunks = 16 chunks over 32 lanes, so lanes 0..15
                 * work and 16..31 contribute 0 -- still a 16x cut in the
                 * serial chain, and the reduction order inside a chunk is
                 * unchanged (same 16 sequential FMAs), so results differ only
                 * by the final tree order.
                 *
                 * This is a decode kernel, so it is validated against the
                 * CPU with PICOLM_DBG_PIPELINE / PICOLM_DUMP_LOGITS
                 * tolerance (1e-2), not for bit equality -- see the block
                 * comment above ATTN_DECODE_MAX_SPLITS. */
                float score;
                {
                    const int lane = tid & 31;
                    int n_chunks = head_dim / 16;
                    float mine = 0.0f;
                    if (tid < 32 && lane < n_chunks) {
                        int d0 = lane * 16;
                        for (int d = d0; d < d0 + 16; d++)
                            mine = fmaf(qg[d], gpu_fp16_to_fp32(k_sh[d]), mine);
                    }
                    /* butterfly reduce over the warp */
                    for (int off = 16; off > 0; off >>= 1)
                        mine += gpuShflXorSync(0xffffffffu, mine, off);
                    if (tid == 0) {
                        /* tail elements not covered by whole 16-chunks */
                        float tail = 0.0f;
                        for (int d = n_chunks * 16; d < head_dim; d++)
                            tail = fmaf(qg[d], gpu_fp16_to_fp32(k_sh[d]), tail);
                        score = (mine + tail) * attn_scale;
                    }
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

/* ---- Decode attention, warp per key (split-K partials for the merge kernel) ----
 *
 * The split kernel above walks its keys one at a time with the whole block:
 * load the K/V row, sync, score on one warp, sync, update the softmax on
 * thread 0, sync, accumulate V -- three barriers per key and query head.
 * Here each of the 4 warps of a block takes every 4th key of the split on
 * its own, ADW_KU keys per step so their loads overlap. Lane l holds DPL =
 * head_dim/32 consecutive dimensions of Q (all query heads of this KV head,
 * pre-scaled), K and V; a score is a warp-wide sum, and each warp keeps its
 * own online softmax per query head. At the end the 4 warps are combined and
 * written in the split kernel's partial format, so
 * picolm_gpu_attention_decode_merge_kernel finishes the job unchanged.
 *
 * G is the largest kv_mul the instance handles, DPL the dimensions per lane;
 * the host picks G in {1,2,4,8} and DPL in {2,4,8} (head_dim 64/128/256).
 * F16, Q8_0 or Q4_0 KV cache (kv_type), 32-lane warps (CUDA). Not bit-identical to the split kernel
 * (different summation order); decode attention is validated to a
 * tolerance, see above. */
#define ADW_KU 4    /* keys per warp per step */

template <int DPL>
__device__ static inline void adw_load(float *dst, const __half *src) {
    __half h[DPL];
    if (DPL == 8) *(uint4 *)h = *(const uint4 *)src;
    else if (DPL == 4) *(uint2 *)h = *(const uint2 *)src;
    else *(uint32_t *)h = *(const uint32_t *)src;
#pragma unroll
    for (int i = 0; i < DPL; i++) dst[i] = __half2float(h[i]);
}

template <int G, int DPL>
__global__ void __launch_bounds__(128)
picolm_gpu_attention_decode_warp_kernel(
        float *partial_max, float *partial_sum, float *partial_acc,
        const float *q_dev,
        const uint16_t *kv_k, const uint16_t *kv_v,
        int layer_ordinal, int pos,
        int n_heads, int n_kv_heads, int max_seq_len,
        size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes, size_t kv_head_stride_v_bytes,
        int n_splits, int chunk_size, int n_swa, int kv_type)
{
    const int head_dim = 32 * DPL;
    __shared__ float wm[4][G], wl[4][G];
    __shared__ float wacc[4][G * 32 * DPL];

    int kv_h = gpuBlockIdx_x, split = gpuBlockIdx_y;
    int tid = gpuThreadIdx_x, lane = tid & 31, warp = tid >> 5;
    int kv_mul = n_heads / n_kv_heads, first_qh = kv_h * kv_mul;
    int d0 = lane * DPL;

    int t0 = split * chunk_size, t1 = min(t0 + chunk_size, pos + 1);
    if (n_swa > 0) t0 = max(t0, pos - n_swa + 1);

    size_t pos_stride = kv_pos_stride_bytes / 2;
    size_t layer_base = (size_t)layer_ordinal * max_seq_len * pos_stride;
    const __half *kb = (const __half *)kv_k + layer_base + (size_t)kv_h * (kv_head_stride_bytes / 2) + d0;
    const __half *vb = (const __half *)kv_v + layer_base + (size_t)kv_h * (kv_head_stride_v_bytes / 2) + d0;
    size_t kbo = (size_t)layer_ordinal * max_seq_len * kv_pos_stride_bytes + (size_t)kv_h * kv_head_stride_bytes;
    size_t vbo = (size_t)layer_ordinal * max_seq_len * kv_pos_stride_bytes + (size_t)kv_h * kv_head_stride_v_bytes;

    float scale = 1.0f / sqrtf((float)head_dim);
    float q[G][DPL], acc[G][DPL], m[G], l[G];
#pragma unroll
    for (int g = 0; g < G; g++) {
        m[g] = -1e30f; l[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) {
            acc[g][i] = 0.0f;
            q[g][i] = g < kv_mul ? q_dev[(size_t)(first_qh + g) * head_dim + d0 + i] * scale : 0.0f;
        }
    }

    for (int tb = t0 + warp; tb < t1; tb += 4 * ADW_KU) {
        float k[ADW_KU][DPL], v[ADW_KU][DPL];
#pragma unroll
        for (int u = 0; u < ADW_KU; u++) {
            int t = min(tb + 4 * u, t1 - 1);   /* a clamped duplicate is skipped below */
            if (kv_type == 0) {
                adw_load<DPL>(k[u], kb + (size_t)t * pos_stride);
                adw_load<DPL>(v[u], vb + (size_t)t * pos_stride);
            } else {   /* Q8_0 / Q4_0: a lane's DPL values sit in one 32-value block */
                size_t ko = kbo + (size_t)t * kv_pos_stride_bytes, vo = vbo + (size_t)t * kv_pos_stride_bytes;
#pragma unroll
                for (int i = 0; i < DPL; i++) {
                    k[u][i] = kv_row_dequant((const uint8_t *)kv_k, ko, kv_type, d0 + i);
                    v[u][i] = kv_row_dequant((const uint8_t *)kv_v, vo, kv_type, d0 + i);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < ADW_KU; u++) {
            if (tb + 4 * u >= t1) break;
#pragma unroll
            for (int g = 0; g < G; g++) {
                if (g >= kv_mul) break;
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < DPL; i++) s = fmaf(q[g][i], k[u][i], s);
                for (int off = 16; off > 0; off >>= 1) s += gpuShflXorSync(0xFFFFFFFFu, s, off);
                if (s > m[g]) {
                    float r = expf(m[g] - s);
                    l[g] = l[g] * r + 1.0f;
#pragma unroll
                    for (int i = 0; i < DPL; i++) acc[g][i] = fmaf(acc[g][i], r, v[u][i]);
                    m[g] = s;
                } else {
                    float w = expf(s - m[g]);
                    l[g] += w;
#pragma unroll
                    for (int i = 0; i < DPL; i++) acc[g][i] = fmaf(w, v[u][i], acc[g][i]);
                }
            }
        }
    }

    /* Combine the 4 warps into this split's partial. */
#pragma unroll
    for (int g = 0; g < G; g++) {
        if (g >= kv_mul) break;
        if (lane == 0) { wm[warp][g] = m[g]; wl[warp][g] = l[g]; }
#pragma unroll
        for (int i = 0; i < DPL; i++) wacc[warp][g * head_dim + d0 + i] = acc[g][i];
    }
    gpuSyncthreads();
    int idx_base = (kv_h * n_splits + split) * kv_mul;
    for (int e = tid; e < kv_mul * head_dim; e += 128) {
        int g = e / head_dim;
        float mx = fmaxf(fmaxf(wm[0][g], wm[1][g]), fmaxf(wm[2][g], wm[3][g]));
        float a = 0.0f, s = 0.0f;
        for (int w = 0; w < 4; w++) {
            float f = expf(wm[w][g] - mx);
            a += f * wacc[w][e];
            s += f * wl[w][g];
        }
        partial_acc[(size_t)idx_base * head_dim + e] = a;
        if (e % head_dim == 0) { partial_max[idx_base + g] = mx; partial_sum[idx_base + g] = s; }
    }
}

/* Launch the instance for (kv_mul, head_dim); 0 if there is none. */
int picolm_gpu_attention_decode_warp_launch(dim3 grid, gpuStream_t stream,
        float *partial_max, float *partial_sum, float *partial_acc, const float *q_dev,
        const uint16_t *kv_k, const uint16_t *kv_v, int layer_ordinal, int pos,
        int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
        size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes, size_t kv_head_stride_v_bytes,
        int n_splits, int chunk_size, int n_swa, int kv_type) {
    int kv_mul = n_heads / n_kv_heads;
    int g = kv_mul <= 1 ? 1 : kv_mul <= 2 ? 2 : kv_mul <= 4 ? 4 : kv_mul <= 8 ? 8 : 0;
#define ADW_CASE(G, D) \
    if (g == G && head_dim == 32 * D) { \
        picolm_gpu_attention_decode_warp_kernel<G, D><<<grid, 128, 0, stream>>>( \
            partial_max, partial_sum, partial_acc, q_dev, kv_k, kv_v, layer_ordinal, pos, \
            n_heads, n_kv_heads, max_seq_len, kv_pos_stride_bytes, kv_head_stride_bytes, \
            kv_head_stride_v_bytes, n_splits, chunk_size, n_swa, kv_type); \
        return 1; \
    }
    ADW_CASE(1, 2) ADW_CASE(1, 4) ADW_CASE(1, 8)
    ADW_CASE(2, 2) ADW_CASE(2, 4) ADW_CASE(2, 8)
    ADW_CASE(4, 2) ADW_CASE(4, 4) ADW_CASE(4, 8)
    ADW_CASE(8, 2) ADW_CASE(8, 4) ADW_CASE(8, 8)
#undef ADW_CASE
    return 0;
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

    /* Merge: acc = sum_sp exp(partial_max[sp] - m) * partial_acc[sp].
     *
     * The weight exp(partial_max[sp] - m) does not depend on d, but it used to
     * be recomputed inside the per-d loop -- head_dim * n_splits expf() calls
     * per query head instead of n_splits. At n_splits = 154 (9848 context) and
     * head_dim 256 that is ~39400 transcendental calls where 154 suffice.
     * Hoisted into shared memory, computed once per (g, sp). */
    {
        static const int MERGE_W_MAX = 512;   /* == ATTN_DECODE_MAX_SPLITS */
        float *w_sh = smem_f + kv_mul * 2;      /* [kv_mul][min(n_splits,MAX)] */
        int nw = n_splits < MERGE_W_MAX ? n_splits : MERGE_W_MAX;
        for (int g2 = 0; g2 < kv_mul; g2++) {
            float m2 = global_max_sh[g2];
            for (int sp = tid; sp < nw; sp += n_threads) {
                int idx = (kv_h * n_splits + sp) * kv_mul + g2;
                w_sh[g2 * MERGE_W_MAX + sp] = expf(partial_max[idx] - m2);
            }
        }
        gpuSyncthreads();

        for (int g = 0; g < kv_mul; g++) {
            float inv_sum = 1.0f / global_sum_sh[g];
            float *xbhg = xb_out + (size_t)(first_qh + g) * head_dim;
            for (int d = tid; d < head_dim; d += n_threads) {
                float acc_d = 0.0f;
                for (int sp = 0; sp < nw; sp++) {
                    int idx = (kv_h * n_splits + sp) * kv_mul + g;
                    acc_d += w_sh[g * MERGE_W_MAX + sp] * partial_acc[(size_t)idx * head_dim + d];
                }
                xbhg[d] = acc_d * inv_sum;
            }
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
        int tile_q, int n_swa)
{
    int h = (int)gpuBlockIdx_x;
    int tile_q_idx = (int)gpuBlockIdx_y;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    int kv_h = h / (n_heads / n_kv_heads);
    float attn_scale = (picolm_gpu_d_attn_scale > 0.0f) ? picolm_gpu_d_attn_scale : 1.0f / sqrtf((float)head_dim);

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
                /* Sliding window: KV positions older than the window are
                 * skipped exactly like future positions above -- same
                 * per-row check, other side of the causal boundary. */
                if (n_swa > 0 && global_kv < global_pos - n_swa + 1) continue;

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
        size_t kv_head_stride_bytes, size_t kv_head_stride_v_bytes,
        int tile_q, int n_swa, int kv_type)
{
    int h = (int)gpuBlockIdx_x;       /* query head */
    int tile_q_idx = (int)gpuBlockIdx_y; /* tile of query tokens */
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    int kv_h = h / (n_heads / n_kv_heads);
    float attn_scale = (picolm_gpu_d_attn_scale > 0.0f) ? picolm_gpu_d_attn_scale : 1.0f / sqrtf((float)head_dim);
    /* Layer base offset in uint16_t */
    size_t layer_base = (size_t)layer_ordinal * max_seq_len * kv_pos_stride_bytes;

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
                size_t row_off = layer_base + (size_t)kv_h * kv_head_stride_bytes + (size_t)(t0 + ti) * kv_pos_stride_bytes + d;
                k_tile[ti * head_dim + d] = gpu_fp32_to_fp16(kv_row_dequant((const uint8_t *)kv_k, row_off - d, kv_type, d));
                v_tile[ti * head_dim + d] = gpu_fp32_to_fp16(kv_row_dequant((const uint8_t *)kv_v, row_off - (size_t)kv_h * kv_head_stride_bytes + (size_t)kv_h * kv_head_stride_v_bytes, kv_type, d));
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
                /* Sliding window: see picolm_gpu_attention_prefill_f32kv_kernel. */
                if (n_swa > 0 && global_kv < global_pos - n_swa + 1) continue;

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

/* ---- Volta tensor-core attention prefill (FlashAttention-2 style) ----
 *
 * For has_hmma (sm_70) GPUs, head_dim 128, F16/Q8_0/Q4_0 KV cache. One thread block
 * is one query head x 64 queries; each of the 4 warps owns 16 queries and
 * walks the KV positions in tiles of 32:
 *   S = Q K^T on the tensor cores (fp16 in, f32 out), scaled and masked
 *   (causal + sliding window); online softmax per row over the whole tile
 *   (one rescale per tile, not per key); P = exp(S - max) to fp16;
 *   O += P V on the tensor cores.
 * mma.m8n8k4 fragment layout: see picolm_q8_q8_matmul_hmma.
 *
 * Not bit-identical to the scalar kernels (fp16 Q and P, tile-wise softmax),
 * the same class of rounding as picolm_gpu_attention_prefill_fa2_kernel. */
#define FAV_D   128             /* head_dim */
#define FAV_TQ  64              /* queries per thread block (4 warps x 16) */
#define FAV_TK  32              /* KV positions per tile */
#define FAV_LDQ (FAV_D + 4)     /* halves per row: 66 words, no bank conflicts */
#define FAV_LDV (FAV_TK + 4)    /* halves per transposed V row: 18 words */

__device__ static inline void fav_mma(float *c, uint2 a, uint2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIP__)
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
#endif
}

__global__ void __launch_bounds__(128)
picolm_gpu_attention_prefill_hmma_kernel(
        float *xb_out, const float *q_dev,
        const uint16_t *kv_k, const uint16_t *kv_v,
        int layer_ordinal, int start_pos, int n_tokens,
        int n_heads, int n_kv_heads, int max_seq_len,
        size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes, size_t kv_head_stride_v_bytes,
        int n_swa, int kv_type)
{
    __shared__ __align__(16) __half q_sh[FAV_TQ][FAV_LDQ];
    __shared__ __align__(16) __half k_sh[FAV_TK][FAV_LDQ];
    __shared__ __align__(16) __half vt_sh[FAV_D][FAV_LDV];
    __shared__ __align__(16) __half p_sh[4][16][FAV_LDV];

    int h = gpuBlockIdx_x, q0 = gpuBlockIdx_y * FAV_TQ;
    int tid = gpuThreadIdx_x, lane = tid & 31, warp = tid >> 5;
    int kv_h = h / (n_heads / n_kv_heads);
    float scale = 1.0f / sqrtf((float)FAV_D);
    size_t layer_base = (size_t)layer_ordinal * max_seq_len * (kv_pos_stride_bytes / 2);
    const uint16_t *kbase = kv_k + layer_base + (size_t)kv_h * (kv_head_stride_bytes / 2);
    const uint16_t *vbase = kv_v + layer_base + (size_t)kv_h * (kv_head_stride_v_bytes / 2);
    size_t pos_stride = kv_pos_stride_bytes / 2;
    size_t kbo = (size_t)layer_ordinal * max_seq_len * kv_pos_stride_bytes + (size_t)kv_h * kv_head_stride_bytes;
    size_t vbo = (size_t)layer_ordinal * max_seq_len * kv_pos_stride_bytes + (size_t)kv_h * kv_head_stride_v_bytes;

    int qp = (lane >> 2) & 3;
    int r = (lane & 3) + 4 * (lane >> 4);                  /* A row / B col */
    int arow = 8 * (qp & 1) + r;                           /* within the warp's 16 rows */
    int drow[2];                                           /* rows of this lane's D */
    for (int u = 0; u < 2; u++) drow[u] = 8 * (qp & 1) + (lane & 1) + 2 * u + 4 * (lane >> 4);
    int dcol[4];                                           /* D columns within a 16-wide group */
    for (int u = 0; u < 4; u++) dcol[u] = 8 * (qp >> 1) + (u & 1) + 2 * ((lane >> 1) & 1) + 4 * (u >> 1);
#define FAV_COL(j) dcol[((j) & 1) | ((j) >> 2) << 1]

    for (int i = tid; i < FAV_TQ * FAV_D; i += 128) {
        int qi = i / FAV_D, d = i % FAV_D, q = q0 + qi;
        float v = q < n_tokens ? q_dev[((size_t)q * n_heads + h) * FAV_D + d] : 0.0f;
        q_sh[qi][d] = __float2half(v);
    }

    float o[FAV_D / 16][8], m[2], l[2];
    for (int n = 0; n < FAV_D / 16; n++)
        for (int j = 0; j < 8; j++) o[n][j] = 0.0f;
    for (int u = 0; u < 2; u++) { m[u] = -INFINITY; l[u] = 0.0f; }
    int qpos[2];
    for (int u = 0; u < 2; u++) qpos[u] = start_pos + q0 + warp * 16 + drow[u];

    int kv_end = min(start_pos + n_tokens, start_pos + q0 + FAV_TQ);
    int kv_begin = n_swa > 0 ? max(0, start_pos + q0 - n_swa + 1) / FAV_TK * FAV_TK : 0;
    for (int t0 = kv_begin; t0 < kv_end; t0 += FAV_TK) {
        gpuSyncthreads();   /* previous tile's k_sh/vt_sh reads are done */
        for (int i = tid; i < FAV_TK * FAV_D / 8; i += 128) {
            int t = i / (FAV_D / 8), d = (i % (FAV_D / 8)) * 8;
            uint4 kq = make_uint4(0, 0, 0, 0), vq = kq;
            if (t0 + t < kv_end && kv_type == 0) {
                kq = *(const uint4 *)(kbase + (size_t)(t0 + t) * pos_stride + d);
                vq = *(const uint4 *)(vbase + (size_t)(t0 + t) * pos_stride + d);
            } else if (t0 + t < kv_end) {   /* Q8_0 / Q4_0 */
                size_t ko = kbo + (size_t)(t0 + t) * kv_pos_stride_bytes, vo = vbo + (size_t)(t0 + t) * kv_pos_stride_bytes;
                __half *kh = (__half *)&kq, *vh = (__half *)&vq;
                for (int e = 0; e < 8; e++) {
                    kh[e] = __float2half(kv_row_dequant((const uint8_t *)kv_k, ko, kv_type, d + e));
                    vh[e] = __float2half(kv_row_dequant((const uint8_t *)kv_v, vo, kv_type, d + e));
                }
            }
            *(uint2 *)&k_sh[t][d] = make_uint2(kq.x, kq.y);
            *(uint2 *)&k_sh[t][d + 4] = make_uint2(kq.z, kq.w);
            const __half *vh = (const __half *)&vq;
            for (int e = 0; e < 8; e++) vt_sh[d + e][t] = vh[e];
        }
        gpuSyncthreads();

        /* S = Q K^T for this warp's 16 queries x 32 keys. */
        float s[FAV_TK / 16][8];
        for (int n = 0; n < FAV_TK / 16; n++)
            for (int j = 0; j < 8; j++) s[n][j] = 0.0f;
#pragma unroll 8
        for (int k = 0; k < FAV_D; k += 4) {
            uint2 a = *(const uint2 *)&q_sh[warp * 16 + arow][k];
            for (int n = 0; n < FAV_TK / 16; n++)
                fav_mma(s[n], a, *(const uint2 *)&k_sh[16 * n + 8 * (qp >> 1) + r][k]);
        }

        /* Scale, mask, row max. A row is spread over the 4 lanes that
         * differ in lane bits 1 and 3. */
        float mt[2] = {-INFINITY, -INFINITY};
        for (int n = 0; n < FAV_TK / 16; n++)
            for (int j = 0; j < 8; j++) {
                int u = (j >> 1) & 1, key = t0 + 16 * n + FAV_COL(j);
                float v = s[n][j] * scale;
                if (key > qpos[u] || key >= kv_end || (n_swa > 0 && key < qpos[u] - n_swa + 1)) v = -INFINITY;
                s[n][j] = v;
                mt[u] = fmaxf(mt[u], v);
            }
        float alpha[2], mn[2], ls[2] = {0.0f, 0.0f};
        for (int u = 0; u < 2; u++) {
            mt[u] = fmaxf(mt[u], gpuShflXorSync(0xFFFFFFFFu, mt[u], 2));
            mt[u] = fmaxf(mt[u], gpuShflXorSync(0xFFFFFFFFu, mt[u], 8));
            mn[u] = fmaxf(m[u], mt[u]);
            alpha[u] = mn[u] == -INFINITY ? 0.0f : expf(m[u] - mn[u]);
            m[u] = mn[u];
        }
        /* P = exp(S - max) as fp16 into this warp's p_sh, for the A operand. */
        for (int n = 0; n < FAV_TK / 16; n++)
            for (int j = 0; j < 8; j++) {
                int u = (j >> 1) & 1;
                float p = mn[u] == -INFINITY ? 0.0f : expf(s[n][j] - mn[u]);
                ls[u] += p;
                p_sh[warp][drow[u]][16 * n + FAV_COL(j)] = __float2half(p);
            }
        for (int u = 0; u < 2; u++) {
            ls[u] += gpuShflXorSync(0xFFFFFFFFu, ls[u], 2);
            ls[u] += gpuShflXorSync(0xFFFFFFFFu, ls[u], 8);
            l[u] = l[u] * alpha[u] + ls[u];
        }
        for (int n = 0; n < FAV_D / 16; n++)
            for (int j = 0; j < 8; j++) o[n][j] *= alpha[(j >> 1) & 1];
        __syncwarp();

        /* O += P V. */
#pragma unroll
        for (int k = 0; k < FAV_TK; k += 4) {
            uint2 a = *(const uint2 *)&p_sh[warp][arow][k];
            for (int n = 0; n < FAV_D / 16; n++)
                fav_mma(o[n], a, *(const uint2 *)&vt_sh[16 * n + 8 * (qp >> 1) + r][k]);
        }
        __syncwarp();   /* p_sh is rewritten next tile */
    }

    for (int u = 0; u < 2; u++) {
        int q = q0 + warp * 16 + drow[u];
        if (q >= n_tokens) continue;
        float inv = l[u] > 0.0f ? 1.0f / l[u] : 0.0f;
        float *out = xb_out + ((size_t)q * n_heads + h) * FAV_D;
        for (int n = 0; n < FAV_D / 16; n++)
            for (int j = 0; j < 8; j++)
                if (((j >> 1) & 1) == u) out[16 * n + FAV_COL(j)] = o[n][j] * inv;
    }
#undef FAV_COL
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
        const uint16_t *kv_k, /* [layer][pos][kv_head] rows; the per-head
                               * stride is kv_head_stride_bytes and the
                               * storage format is kv_type (0 = F16,
                               * 1 = Q8_0, 2 = Q4_0). Read through
                               * kv_row_load_f16()/kv_row_dequant() so all
                               * three formats work. */
        const uint16_t *kv_v, /* same layout/format as kv_k, but with its
                               * own per-head stride (kv_head_stride_v_bytes) */
        int layer_ordinal,
        int start_pos, int n_tokens,
        int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
        size_t kv_pos_stride_bytes,
        size_t kv_head_stride_bytes, size_t kv_head_stride_v_bytes,
        int tile_q, int n_swa, int kv_type)
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
    float attn_scale = (picolm_gpu_d_attn_scale > 0.0f) ? picolm_gpu_d_attn_scale : 1.0f / sqrtf((float)head_dim);
    size_t layer_base = (size_t)layer_ordinal * max_seq_len * kv_pos_stride_bytes;

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
        if (kv_type == 0 && (head_dim & 7) == 0) {   /* F16-only fast path */
            int vec_elems = head_dim >> 3;
            for (int ti = 0; ti < tile_k_size; ti++) {
                /* layer_base is a BYTE offset; halve it back to uint16
                 * elements for this pointer-arithmetic fast path (the old
                 * code had layer_base already in elements, so this is the
                 * same address it computed before). */
                size_t base_off = layer_base / 2 + (size_t)kv_h * (kv_head_stride_bytes / 2)
                    + (size_t)(t0 + ti) * (kv_pos_stride_bytes / 2);
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
                    size_t head_off = layer_base + (size_t)kv_h * kv_head_stride_bytes
                        + (size_t)(t0 + ti) * kv_pos_stride_bytes;
                    k_tile[ti * head_dim + d] = gpu_fp32_to_fp16(kv_row_dequant((const uint8_t *)kv_k, head_off, kv_type, d));
                    size_t v_head_off = head_off - (size_t)kv_h * kv_head_stride_bytes + (size_t)kv_h * kv_head_stride_v_bytes;
                    v_tile[ti * head_dim + d] = gpu_fp32_to_fp16(kv_row_dequant((const uint8_t *)kv_v, v_head_off, kv_type, d));
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
                /* Sliding window: see picolm_gpu_attention_prefill_f32kv_kernel. */
                if (n_swa > 0 && global_kv < global_pos - n_swa + 1) continue;

                /* TODO(perf): replace scalar fmaf+gpu_fp16_to_fp32 per-element with
                 * v_dot2_f32_f16 (1 instruction = 2 FP16 FMAs, FP32 accumulate).
                 * Available on gfx906, CDNA, RDNA2+ (HIP), and NVIDIA sm_80+ (__hfma2).
                 * Would halve instruction count here and is the single biggest remaining
                 * win vs llama.cpp's fattn-tile kernel on MI50. Requires keeping K tile
                 * as half2 in shared memory (not uint16_t) and Q as float2 pairs.
                 * Bit-exactness needs careful verification since the FMAs are
                 * paired differently (d[2k]*q[2k] + d[2k+1]*q[2k+1] in one
                 * instruction vs sequential fmaf).
                 *
                 * TODO(perf, head_dim=256): On NVIDIA GPUs with IMMA, this warpgrp
                 * kernel is used as fallback when FA2 kernel exceeds 96 KB shared memory
                 * (head_dim=256 needs 136.5 KB). The warpgrp path does NOT use tensor
                 * cores, leaving them idle. For Qwen3.5/3.6 (head_dim=256), prefer a
                 * FA2 variant with 32 Q rows per block instead of 64.
                 * See /data4/work/notes/picolm/fa2_headdim256.md */
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
        size_t kv_head_stride_bytes, size_t kv_head_stride_v_bytes,
        int tile_q, int n_swa, int kv_type)
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
    float attn_scale = (picolm_gpu_d_attn_scale > 0.0f) ? picolm_gpu_d_attn_scale : 1.0f / sqrtf((float)head_dim);
    size_t layer_base = (size_t)layer_ordinal * max_seq_len * kv_pos_stride_bytes;

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

        if (kv_type == 0 && (head_dim & 7) == 0) {   /* F16-only fast path */
            int vec_elems = head_dim >> 3;
            for (int ti = 0; ti < tile_k_size; ti++) {
                /* layer_base is a BYTE offset; halve it back to uint16
                 * elements for this pointer-arithmetic fast path (the old
                 * code had layer_base already in elements, so this is the
                 * same address it computed before). */
                size_t base_off = layer_base / 2 + (size_t)kv_h * (kv_head_stride_bytes / 2)
                    + (size_t)(t0 + ti) * (kv_pos_stride_bytes / 2);
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
                    size_t head_off = layer_base + (size_t)kv_h * kv_head_stride_bytes
                        + (size_t)(t0 + ti) * kv_pos_stride_bytes;
                    k_tile[ti * head_dim + d] = gpu_fp32_to_fp16(kv_row_dequant((const uint8_t *)kv_k, head_off, kv_type, d));
                    size_t v_head_off = head_off - (size_t)kv_h * kv_head_stride_bytes + (size_t)kv_h * kv_head_stride_v_bytes;
                    v_tile[ti * head_dim + d] = gpu_fp32_to_fp16(kv_row_dequant((const uint8_t *)kv_v, v_head_off, kv_type, d));
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
                /* Sliding window: see picolm_gpu_attention_prefill_f32kv_kernel. */
                if (n_swa > 0 && global_kv < global_pos - n_swa + 1) continue;

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

/* picolm_ssm_vecdot_batch_warp_kernel -- batched sibling of
 * picolm_ssm_vecdot_warp_kernel.
 *
 * The scalar batch kernel does `if (tid != 0) return;` and computes each
 * (head, token) dot product on ONE thread, launched as
 * <<<dim3(n_v_heads, n_tokens), 256>>> with 255 of every 256 threads idle.
 * For the 27B that is 48 x 423 = 20304 blocks each running a serial
 * dim-length FMA chain, and it measured 2.05 ms per call / 196 ms of a
 * prefill, 20% of all profiled GPU time.
 *
 * This version runs one WARP per (head, token): the 32 lanes split dim between
 * them and a butterfly reduce finishes it, so the serial chain is dim/32
 * instead of dim and every launched thread does work.
 *
 * Numerics: same summation-structure caveat as the single-token warp kernel --
 * the serial version is not bit-identical to the CPU either, and the oracle is
 * PICOLM_DUMP_LOGITS, not eyeballed text.
 */
#define VECB_WARPS 8

__global__ void
picolm_ssm_vecdot_batch_warp_kernel(float *out, const float *x, const void *weights,
                                     gguf_type_t qtype, int dim, int n_v_heads, int n_tokens,
                                     int row_bytes, const int *head_map,
                                     int in_stride, int out_stride) {
    int wid  = (int)(gpuThreadIdx_x >> 5);
    int lane = (int)(gpuThreadIdx_x & 31);

    long long gid = (long long)gpuBlockIdx_x * VECB_WARPS + wid;
    long long total = (long long)n_v_heads * n_tokens;
    if (gid >= total) return;

    int h = (int)(gid % n_v_heads);
    int t = (int)(gid / n_v_heads);

    int gh = head_map ? head_map[h] : h;
    const char *wrow = (const char *)weights + (size_t)gh * row_bytes;
    int ts = in_stride > 0 ? in_stride : dim;
    const float *xt = x + (size_t)t * ts;

    float sum = 0.0f;
    switch (qtype) {
    case 0: {   /* F32 */
        const float *wf = (const float *)wrow;
        float a0 = 0.0f, a1 = 0.0f;
        int i = lane;
        for (; i + 32 < dim; i += 64) {
            a0 = fmaf(wf[i],      xt[i],      a0);
            a1 = fmaf(wf[i + 32], xt[i + 32], a1);
        }
        for (; i < dim; i += 32) a0 = fmaf(wf[i], xt[i], a0);
        sum = a0 + a1;
        break;
    }
    case 2: {   /* Q4_0 x f32 */
        const int nb = dim / 32;
        float acc = 0.0f;
        for (int bi = lane; bi < nb; bi += 32) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 18;
            float wd = gpu_fp16_to_fp32((uint16_t)(blk[0] | ((uint16_t)blk[1] << 8)));
            const uint8_t *qs = blk + 2;
            const float *xb = xt + bi * 32;
            float s2 = 0.0f;
            for (int j = 0; j < 16; j++) {
                s2 = fmaf((float)((qs[j] & 0x0F) - 8), xb[j],      s2);
                s2 = fmaf((float)((qs[j] >> 4) - 8), xb[j + 16], s2);
            }
            acc = fmaf(s2, wd, acc);
        }
        sum = acc;
        break;
    }
    case 8: {   /* Q8_0 x f32 */
        const int nb = dim / 32;
        float acc = 0.0f;
        for (int bi = lane; bi < nb; bi += 32) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 34;
            float wd = gpu_fp16_to_fp32((uint16_t)(blk[0] | ((uint16_t)blk[1] << 8)));
            const int8_t *qs = (const int8_t *)(blk + 2);
            const float *xb = xt + bi * 32;
            float s2 = 0.0f;
            for (int j = 0; j < 32; j++) s2 = fmaf((float)qs[j], xb[j], s2);
            acc = fmaf(s2, wd, acc);
        }
        sum = acc;
        break;
    }
    case 30: {  /* BF16 */
        float a0 = 0.0f, a1 = 0.0f;
        int i = lane;
        for (; i + 32 < dim; i += 64) {
            a0 = fmaf(dequant_bf16(wrow, i),      xt[i],      a0);
            a1 = fmaf(dequant_bf16(wrow, i + 32), xt[i + 32], a1);
        }
        for (; i < dim; i += 32) a0 = fmaf(dequant_bf16(wrow, i), xt[i], a0);
        sum = a0 + a1;
        break;
    }
    default:
        return;
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += gpuShflDownSync(0xffffffffu, sum, off);

    if (lane == 0) {
        int os = out_stride > 0 ? out_stride : n_v_heads;
        out[(size_t)t * os + h] = sum;
    }
}

/* ================================================================
 * picolm_ssm_vecdot_warp_kernel -- same result as
 * picolm_ssm_vecdot_kernel, done by a WARP per head instead of one
 * thread per head.
 *
 * The single-thread version wastes 255 of every 256 launched threads
 * (`if (tid != 0) return;`) and then runs a 5120-iteration serial FMA
 * chain with no ILP. Measured on the 27B it moved 960 KB in 0.21 ms =
 * 4.6 GB/s, i.e. 0.5% of the card, twice per SSM layer.
 *
 * Layout: VEC_WARP_HEADS heads per block, one warp each, so a block
 * covers several heads with every lane busy. Lane `l` owns elements
 * l, l+32, l+64, ... of its head's row and the shared x vector; the
 * warp tree-reduces at the end.
 *
 * NUMERICS: this is NOT bit-identical to the serial version, and it
 * does not try to be. The serial version is not bit-identical to the
 * CPU either -- quant.c's vec_dot_f32_f32 on this box is AVX-512 and
 * uses 16-lane _mm512_fmadd_ps accumulators plus
 * _mm512_reduce_add_ps. Summation order therefore already differs
 * between the two shipped paths. Validate with PICOLM_DUMP_LOGITS
 * (max|diff| + argmax), never by eyeballing text.
 * ================================================================ */
template <int TH>
__device__ __forceinline__ void
picolm_ssm_vecdot_warp_body(int h, const float *x, const void *weights,
                            gguf_type_t qtype, int dim, int row_bytes,
                            const int *head_map, float *out, int out_index, int n_heads_here) {
    const int lane = (int)(gpuThreadIdx_x & 31u);
    const int warp = (int)(gpuThreadIdx_x >> 5);
    if (warp >= TH) return;
    const int head = h + warp;
    if (warp >= n_heads_here) return;

    const int gh = head_map ? head_map[head] : head;
    const char *wrow = (const char *)weights + (size_t)gh * row_bytes;

    float sum = 0.0f;

    switch (qtype) {
    case 0: { /* F32: two independent accumulators for ILP */
        const float *wrow_f = (const float *)wrow;
        float a0 = 0.0f, a1 = 0.0f;
        int i = lane;
        for (; i + 32 < dim; i += 64) {
            a0 = fmaf(wrow_f[i],      x[i],      a0);
            a1 = fmaf(wrow_f[i + 32], x[i + 32], a1);
        }
        for (; i < dim; i += 32) a0 = fmaf(wrow_f[i], x[i], a0);
        sum = a0 + a1;
        break;
    }
    case 2: { /* Q4_0 x f32 x -- integer MAC per 32-value block, then scale */
        const int nb = dim / 32;
        float acc = 0.0f;
        for (int bi = lane; bi < nb; bi += 32) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 18;
            float wd = gpu_fp16_to_fp32((uint16_t)(blk[0] | ((uint16_t)blk[1] << 8)));
            const uint8_t *qs = blk + 2;
            const float *xb = x + bi * 32;
            float s = 0.0f;
            for (int j = 0; j < 16; j++) {
                s = fmaf((float)((qs[j] & 0x0F) - 8), xb[j],      s);
                s = fmaf((float)((qs[j] >> 4) - 8), xb[j + 16], s);
            }
            acc = fmaf(s, wd, acc);
        }
        sum = acc;
        break;
    }
    case 8: { /* Q8_0 x f32 */
        const int nb = dim / 32;
        float acc = 0.0f;
        for (int bi = lane; bi < nb; bi += 32) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 34;
            float wd = gpu_fp16_to_fp32((uint16_t)(blk[0] | ((uint16_t)blk[1] << 8)));
            const int8_t *qs = (const int8_t *)(blk + 2);
            const float *xb = x + bi * 32;
            float s = 0.0f;
            for (int j = 0; j < 32; j++) s = fmaf((float)qs[j], xb[j], s);
            acc = fmaf(s, wd, acc);
        }
        sum = acc;
        break;
    }
    case 30: { /* BF16 */
        float a0 = 0.0f, a1 = 0.0f;
        int i = lane;
        for (; i + 32 < dim; i += 64) {
            a0 = fmaf(dequant_bf16(wrow, i),      x[i],      a0);
            a1 = fmaf(dequant_bf16(wrow, i + 32), x[i + 32], a1);
        }
        for (; i < dim; i += 32) a0 = fmaf(dequant_bf16(wrow, i), x[i], a0);
        sum = a0 + a1;
        break;
    }
    case 41: { /* Q1_0, 128 values per block */
        const int nb = dim / 128;
        float acc = 0.0f;
        for (int bi = lane; bi < nb; bi += 32) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 18;
            float wd = gpu_fp16_to_fp32((uint16_t)(blk[0] | ((uint16_t)blk[1] << 8)));
            const uint8_t *qs = blk + 2;
            const float *xb = x + bi * 128;
            float s = 0.0f;
            for (int j = 0; j < 128; j++)
                s = fmaf(((qs[j >> 3] >> (j & 7)) & 1) ? wd : -wd, xb[j], s);
            acc += s;
        }
        sum = acc;
        break;
    }
    case 14: { /* Q6_K, 256 values per block (210 bytes) */
        const int nb = dim / 256;
        float acc = 0.0f;
        for (int bi = lane; bi < nb; bi += 32) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * GPU_BLOCK_Q6_K_SIZE;
            const uint8_t *ql = blk;
            const uint8_t *qh = blk + 128;
            const int8_t  *scales = (const int8_t *)(qh + 64);
            float wd = gpu_fp16_to_fp32((uint16_t)(blk[208] | ((uint16_t)blk[209] << 8)));
            const float *xb = x + bi * 256;
            float s = 0.0f;
            for (int half = 0; half < 2; half++) {
                const uint8_t *ql_c = ql + half * 64;
                const uint8_t *qh_c = qh + half * 32;
                for (int sc = 0; sc < 8; sc++) {
                    const int sc_half = sc & 1, sub_chunk = sc >> 1;
                    float sc_v = (float)scales[half * 8 + sc];
                    const int base = half * 128 + sub_chunk * 32 + sc_half * 16;
                    for (int l = 0; l < 32; l++) {
                        uint8_t qv, hv;
                        if (sub_chunk == 0)      { qv = ql_c[l] & 0xF;      hv = (qh_c[l] >> 0) & 3; }
                        else if (sub_chunk == 1) { qv = ql_c[l + 32] & 0xF; hv = (qh_c[l] >> 2) & 3; }
                        else if (sub_chunk == 2) { qv = ql_c[l] >> 4;      hv = (qh_c[l] >> 4) & 3; }
                        else                     { qv = ql_c[l + 32] >> 4; hv = (qh_c[l] >> 6) & 3; }
                        int q = (qv | (hv << 4)) - 32;
                        s = fmaf(wd * sc_v * (float)q, xb[base + l], s);
                    }
                }
            }
            acc += s;
        }
        sum = acc;
        break;
    }
    default:
        return;
    }

    /* warp tree reduce */
    for (int off = 16; off > 0; off >>= 1)
        sum += gpuShflDownSync(0xffffffffu, sum, off);

    if (lane == 0 && out)
        out[out_index + warp] = sum;
}

__global__ void
picolm_ssm_vecdot_warp_kernel(float *out,
                               const float *x,
                               const void *weights,
                               gguf_type_t qtype,
                               int dim, int n_v_heads,
                               int row_bytes,
                               const int *head_map) {
    const int base = (int)gpuBlockIdx_x * VEC_WARP_HEADS;
    if (base >= n_v_heads) return;
    const int n = n_v_heads - base;
    if (n >= VEC_WARP_HEADS)      picolm_ssm_vecdot_warp_body<VEC_WARP_HEADS>(base, x, weights, qtype, dim, row_bytes, head_map, out, base, n);
    else if (n >= 3)              picolm_ssm_vecdot_warp_body<3>(base, x, weights, qtype, dim, row_bytes, head_map, out, base, n);
    else if (n >= 2)              picolm_ssm_vecdot_warp_body<2>(base, x, weights, qtype, dim, row_bytes, head_map, out, base, n);
    else                          picolm_ssm_vecdot_warp_body<1>(base, x, weights, qtype, dim, row_bytes, head_map, out, base, n);
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
    /* WARNING: the token loop below MUST stay serial, in order.
     *
     * The caller aliases input and output (conv1d_batch_dev(bxb, ..., bxb)).
     * Each token's input is read into a register before its own output is
     * written, so a single thread walking t = 0..n_tokens-1 is safe, and the
     * hist[] window holds the ORIGINAL inputs because they were captured
     * before being overwritten. Any reordering across tokens -- splitting the
     * token range over blockIdx.y, or computing x[t-d_conv+1..t] directly from
     * the array -- makes one block overwrite x[t] before another reads it.
     *
     * Both were tried. Splitting tokens over blockIdx.y measured max|logit
     * diff| 15.4 against the previous build with a different argmax, i.e.
     * garbage, and prefill barely moved (0.443 -> 0.367 ms/call) because the
     * cost is the dependent chain, not the shift.
     *
     * If this ever needs to be faster: the honest route is to copy the input
     * out first (one D2D of n_tokens*stride*4 bytes, ~17 MB for the 27B) and
     * then let the tokens run independently against the copy. The measured
     * shape says so -- 423 dependent iterations at ~1.1 us each is memory
     * latency on the chain, not ALU, and expf is only ~1% of the 0.466 ms. */
    int n_state_rows = d_conv - 1;
    int co = (int)gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (co >= conv_dim) return;

    float w[PICOLM_SSM_CONV_MAX_D_CONV];
    for (int d = 0; d < d_conv; d++)
        w[d] = conv1d_w[d + co * d_conv];

    float hist[PICOLM_SSM_CONV_MAX_D_CONV];
    for (int r = 0; r < n_state_rows; r++)
        hist[r] = conv_state[(size_t)r * conv_dim + co];

    for (int t = 0; t < n_tokens; t++) {
        float new_sample = new_input[(size_t)t * stride + co];
        float sum = 0.0f;
        for (int d = 0; d < n_state_rows; d++)
            sum += w[d] * hist[d];
        sum += w[n_state_rows] * new_sample;
        conv_output[(size_t)t * stride + co] = sum / (1.0f + expf(-sum));

        for (int r = 0; r < n_state_rows - 1; r++)
            hist[r] = hist[r + 1];
        if (n_state_rows > 0)
            hist[n_state_rows - 1] = new_sample;
    }

    for (int r = 0; r < n_state_rows; r++)
        conv_state[(size_t)r * conv_dim + co] = hist[r];
}
__global__ void
picolm_ssm_recurrence_kernel(float *state,
                              const float *q_conv,
                              const float *k_conv,
                              const float *v_conv,
                              const float *gate_exp,
                              const float *beta,
                              float *ssm_output,
                              int n_v_heads, int d_state, int head_v_dim,
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
    /* The recurrence runs one independent slot per *value* dimension: the
     * state is [head_v_dim][d_state], with one rank-1 update per slot. So the
     * row count and the per-head stride of state and v_conv are head_v_dim,
     * NOT d_state. They are equal for every shipped Qwen (both 128), which is
     * why using d_state here went unnoticed, but it would read past v_conv and
     * mis-address the state block as soon as the two differ. This mirrors the
     * CPU fix in ssm_head_task(). */
    int kh = h / repeat;
    const float *qh = q_conv + (size_t)kh * d_state;
    const float *khv = k_conv + (size_t)kh * d_state;
    const float *vh = v_conv + (size_t)h * head_v_dim;
    float ge = gate_exp[h];
    float bh = beta[h];
    float *st = state + (size_t)h * head_v_dim * d_state;

    int d4 = d_state / 4; /* d_state always 16/32/64/128 */
    float sk_local[256];
    float d_local[256];

    /* Decay + sk: 4-lane strided fmaf per row, matching NEON */
    for (int row = 0; row < head_v_dim; row++) {
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
    for (int row = 0; row < head_v_dim; row++) {
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


/* ================================================================
 * FA3 prefill attention: tensor-core QK^T AND P@V, O in registers.
 * The kernel lives in backend_gpu_attn_fa3.cuh (so tests/fa3_emu.cpp can
 * execute the identical source on a CPU); this is its launcher.
 *
 * Why it exists: the FA2 kernel above keeps a 64 x head_dim fp32 O accumulator
 * in shared memory and needs 136.5 KB at head_dim 256 against a 99 KB cap, so
 * Qwen3.5/3.6 prefill attention fell through to the scalar warpgrp kernel
 * (~0.6 TFLOP/s, O(n^2), the reason prefill rate collapses with context).
 * ================================================================ */
#include "backend_gpu_attn_fa3.cuh"

#ifndef __HIP__
template <int HD, int BN>
static int fa3_launch_t(gpuStream_t stream, float *out, const float *q,
                        const uint8_t *k, const uint8_t *v,
                        int layer, int start_pos, int n_tokens, int n_heads, int n_kv,
                        int max_seq, size_t kps, size_t khs, size_t vps, size_t vhs,
                        int n_swa, int kt, int vt) {
    const fa3_layout L = fa3_make_layout(HD, BN, kt, vt);
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if ((size_t)optin < L.total) {                    /* does not fit on this GPU: caller falls back */
        if (getenv("PICOLM_FA3_DEBUG"))
            fprintf(stderr, "[FA3] reject: smem hd=%d bn=%d needs %zu B, device opt-in max %d B\n",
                    HD, BN, L.total, optin);
        return 0;
    }
    static size_t configured[16];                     /* per instantiation, per device */
    if (dev >= 0 && dev < 16 && L.total > configured[dev]) {
        if (cudaFuncSetAttribute((const void *)picolm_gpu_attention_prefill_fa3_kernel<HD, BN>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.total) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        configured[dev] = L.total;
    }
    dim3 grid((unsigned)n_heads, (unsigned)((n_tokens + FA3_BM - 1) / FA3_BM), 1);
    picolm_gpu_attention_prefill_fa3_kernel<HD, BN><<<grid, FA3_THREADS, L.total, stream>>>(
        out, q, k, v, layer, start_pos, n_tokens, n_heads, n_kv, max_seq,
        kps, khs, vps, vhs, n_swa, kt, vt);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess && getenv("PICOLM_FA3_DEBUG"))
        fprintf(stderr, "[FA3] reject: launch failed: %s (hd=%d bn=%d grid=%ux%u smem=%zu)\n",
                cudaGetErrorString(e), HD, BN, grid.x, grid.y, L.total);
    return e == cudaSuccess;
}
#endif

/* ================================================================
 * fp16 tensor-core GEMM for Q5_K / Q6_K / Q8_0 weights (prefill).
 * The kernel and its design notes live in backend_gpu_gemm_f16.cuh; this is its
 * launcher. Tile sizes are picked per call from the grid shape (see below).
 * ================================================================ */
#include "backend_gpu_gemm_f16.cuh"

#ifndef __HIP__
template <int FMT, int BM, int BN>
static int gemmf_launch_t(gpuStream_t stream, float *y, const float *x, const uint8_t *w,
                          int S, int I, int O, size_t row_bytes, int xs, int ys) {
    const size_t smem = gemmf_smem_bytes(FMT, BM, BN);
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if ((size_t)optin < smem) return 0;               /* does not fit on this GPU: caller falls back */
    static size_t configured[16];                     /* per instantiation, per device */
    if (dev >= 0 && dev < 16 && smem > configured[dev]) {
        if (cudaFuncSetAttribute((const void *)picolm_gpu_gemm_f16_kernel<FMT, BM, BN>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        configured[dev] = smem;
    }
    /* blockIdx.x walks token tiles, blockIdx.y weight row-blocks: the token tiles of one
     * row-block are scheduled together, so the weights are read from DRAM once and the
     * other tiles hit them in L2. */
    dim3 grid((unsigned)((S + BN - 1) / BN), (unsigned)((O + BM - 1) / BM), 1);
    picolm_gpu_gemm_f16_kernel<FMT, BM, BN><<<grid, GEMMF_THREADS, smem, stream>>>(
        y, x, w, S, I, O, row_bytes, xs, ys);
    return cudaGetLastError() == cudaSuccess;
}

template <int FMT>
static int gemmf_launch_fmt(int bm, int bn, gpuStream_t stream, float *y, const float *x, const uint8_t *w,
                            int S, int I, int O, size_t rb, int xs, int ys) {
    if (bm == 128 && bn == 128) return gemmf_launch_t<FMT, 128, 128>(stream, y, x, w, S, I, O, rb, xs, ys);
    if (bm == 128 && bn == 64)  return gemmf_launch_t<FMT, 128, 64>(stream, y, x, w, S, I, O, rb, xs, ys);
    if (bm == 64  && bn == 128) return gemmf_launch_t<FMT, 64, 128>(stream, y, x, w, S, I, O, rb, xs, ys);
    if (bm == 64  && bn == 64)  return gemmf_launch_t<FMT, 64, 64>(stream, y, x, w, S, I, O, rb, xs, ys);
    return gemmf_launch_t<FMT, 64, 32>(stream, y, x, w, S, I, O, rb, xs, ys);
}

/* Pick BM x BN. A block always does BM*BN*K work, so the time is about
 * (number of waves) * BM * BN, and a partial last wave wastes the rest of the GPU:
 * O=5120 with 128x128 tiles and S=512 is 40*4 = 160 blocks on 128 SMs = 2 waves for
 * 1.25 waves of work. Smaller tiles issue more ldmatrix per mma, hence the weights.
 * PICOLM_GEMM_F16_TILE=BMxBN (e.g. 128x64) overrides this for tuning on real hardware. */
static void gemmf_pick_tile(int S, int O, int nsm, int *bm_out, int *bn_out) {
    static int forced_bm = -1, forced_bn = -1;
    if (forced_bm < 0) {
        forced_bm = forced_bn = 0;
        const char *e = getenv("PICOLM_GEMM_F16_TILE");
        int a = 0, b = 0;
        if (e && sscanf(e, "%dx%d", &a, &b) == 2 && (a == 64 || a == 128) && (b == 32 || b == 64 || b == 128)) {
            forced_bm = a; forced_bn = b;
        }
    }
    if (forced_bm) { *bm_out = forced_bm; *bn_out = forced_bn; return; }
    static const int cand[5][2] = {{128, 128}, {128, 64}, {64, 128}, {64, 64}, {64, 32}};
    static const double eff[5] = {1.00, 1.12, 1.12, 1.30, 1.55};   /* smaller tiles = more ldmatrix per mma */
    double best = 1e300; int bi = 0;
    for (int i = 0; i < 5; i++) {
        const long blocks = (long)((S + cand[i][1] - 1) / cand[i][1]) * ((O + cand[i][0] - 1) / cand[i][0]);
        const long waves = (blocks + nsm - 1) / nsm;
        const double cost = (double)waves * cand[i][0] * cand[i][1] * eff[i];
        if (cost < best) { best = cost; bi = i; }
    }
    *bm_out = cand[bi][0]; *bn_out = cand[bi][1];
}
#endif

/* Returns 1 if launched, 0 if the shape / format / alignment is unsupported (caller falls back).
 * fmt: 5 = Q5_K, 6 = Q6_K, 8 = Q8_0. */
int picolm_gpu_gemm_f16_launch(gpuStream_t stream, int fmt, float *y, const float *x, const void *w,
                               int S, int I, int O, size_t row_bytes, int x_stride, int y_stride) {
#ifdef __HIP__
    (void)stream; (void)fmt; (void)y; (void)x; (void)w; (void)S; (void)I; (void)O;
    (void)row_bytes; (void)x_stride; (void)y_stride;
    return 0;
#else
    if (S < 1 || O < 8 || (O & 1)) return 0;
    if (fmt == GEMMF_Q5_K || fmt == GEMMF_Q6_K) { if (I % 256) return 0; }
    else if (fmt == GEMMF_Q8_0) { if (I % 64) return 0; }
    else return 0;
    const int xs = x_stride > 0 ? x_stride : I, ys = y_stride > 0 ? y_stride : O;
    if ((xs & 3) || (ys & 1)) return 0;               /* float4 activation loads, float2 stores */
    if (((uintptr_t)x & 15) || ((uintptr_t)y & 7)) return 0;
    if (fmt == GEMMF_Q5_K && (((uintptr_t)w & 15) || (row_bytes & 15))) return 0;   /* 16-byte cp.async */
    if (fmt == GEMMF_Q8_0 && (((uintptr_t)w & 3) || (row_bytes & 3))) return 0;     /* 4-byte cp.async */
    if (fmt == GEMMF_Q6_K && ((uintptr_t)w & 1)) return 0;
    int dev = 0, nsm = 0, bm = 0, bn = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    gemmf_pick_tile(S, O, nsm > 0 ? nsm : 1, &bm, &bn);
    const uint8_t *wp = (const uint8_t *)w;
    if (fmt == GEMMF_Q5_K) return gemmf_launch_fmt<GEMMF_Q5_K>(bm, bn, stream, y, x, wp, S, I, O, row_bytes, xs, ys);
    if (fmt == GEMMF_Q6_K) return gemmf_launch_fmt<GEMMF_Q6_K>(bm, bn, stream, y, x, wp, S, I, O, row_bytes, xs, ys);
    return gemmf_launch_fmt<GEMMF_Q8_0>(bm, bn, stream, y, x, wp, S, I, O, row_bytes, xs, ys);
#endif
}

/* Returns 1 if launched, 0 if this shape/format is unsupported (caller falls back). */
int picolm_gpu_attention_prefill_fa3_launch(
        gpuStream_t stream, float *out, const float *q, const void *kv_k, const void *kv_v,
        int layer_ordinal, int start_pos, int n_tokens, int n_heads, int n_kv_heads,
        int head_dim, int max_seq_len,
        size_t k_pos_stride, size_t k_head_stride, size_t v_pos_stride, size_t v_head_stride,
        int n_swa, int kt, int vt) {
#ifdef __HIP__
    (void)stream; (void)out; (void)q; (void)kv_k; (void)kv_v; (void)layer_ordinal; (void)start_pos;
    (void)n_tokens; (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)max_seq_len;
    (void)k_pos_stride; (void)k_head_stride; (void)v_pos_stride; (void)v_head_stride;
    (void)n_swa; (void)kt; (void)vt;
    return 0;
#else
    /* PICOLM_FA3_DEBUG prints which precondition rejected the launch; every one
     * of these returns 0 silently by design (the caller falls back), which
     * makes "why is FA3 not running" unanswerable without it. */
    static const int fa3dbg = getenv("PICOLM_FA3_DEBUG") ? 1 : 0;
#define FA3_REJECT(cond, ...) do { if (cond) { if (fa3dbg) { fprintf(stderr, "[FA3] reject: " __VA_ARGS__); fprintf(stderr, "\n"); } return 0; } } while (0)
    FA3_REJECT(n_tokens < 1 || n_kv_heads < 1 || n_heads % n_kv_heads != 0,
               "geometry n_tokens=%d n_kv=%d n_heads=%d", n_tokens, n_kv_heads, n_heads);
    FA3_REJECT(kt < 0 || kt > 4 || vt < 0 || vt > 4, "kv types kt=%d vt=%d", kt, vt);
    FA3_REJECT(head_dim != 64 && head_dim != 128 && head_dim != 256,
               "head_dim=%d (need 64/128/256)", head_dim);
    /* the kernel addresses rows as contiguous [head_dim] fp16 / [head_dim/32] blocks */
    const size_t krow = kt == 0 ? (size_t)head_dim * 2 : (size_t)fa3_raw_row_bytes(head_dim, kt);
    const size_t vrow = vt == 0 ? (size_t)head_dim * 2 : (size_t)fa3_raw_row_bytes(head_dim, vt);
    FA3_REJECT(k_head_stride != krow || v_head_stride != vrow,
               "head_stride k=%zu (need %zu) v=%zu (need %zu)",
               k_head_stride, krow, v_head_stride, vrow);
    FA3_REJECT((k_pos_stride & (kt == 0 ? 15 : 3)) || (v_pos_stride & (vt == 0 ? 15 : 3)),
               "pos_stride not aligned k=%zu (mask %d) v=%zu (mask %d)",
               k_pos_stride, kt == 0 ? 15 : 3, v_pos_stride, vt == 0 ? 15 : 3);
    FA3_REJECT(((uintptr_t)kv_k & 15) || ((uintptr_t)kv_v & 15) || ((uintptr_t)q & 15) || ((uintptr_t)out & 7),
               "pointer alignment k=%p v=%p q=%p out=%p",
               (const void *)kv_k, (const void *)kv_v, (const void *)q, (const void *)out);
    const uint8_t *k = (const uint8_t *)kv_k, *v = (const uint8_t *)kv_v;
    if (head_dim == 256)
        return fa3_launch_t<256, 32>(stream, out, q, k, v, layer_ordinal, start_pos, n_tokens, n_heads,
                                     n_kv_heads, max_seq_len, k_pos_stride, k_head_stride, v_pos_stride,
                                     v_head_stride, n_swa, kt, vt);
    if (head_dim == 128)
        return fa3_launch_t<128, 64>(stream, out, q, k, v, layer_ordinal, start_pos, n_tokens, n_heads,
                                     n_kv_heads, max_seq_len, k_pos_stride, k_head_stride, v_pos_stride,
                                     v_head_stride, n_swa, kt, vt);
    return fa3_launch_t<64, 64>(stream, out, q, k, v, layer_ordinal, start_pos, n_tokens, n_heads,
                                n_kv_heads, max_seq_len, k_pos_stride, k_head_stride, v_pos_stride,
                                v_head_stride, n_swa, kt, vt);
#endif
}

__global__ void
picolm_gpu_attention_prefill_fa2_kernel(
    float *xb_out, const float *q_dev,
    const uint16_t *kv_k, const uint16_t *kv_v,
    int layer_ordinal, int start_pos, int n_tokens,
    int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
    size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes, size_t kv_head_stride_v_bytes, int n_swa, int kv_type)
{
    int q_h = gpuBlockIdx_x;        /* query head 0..n_heads-1 */
    int q_tile_idx = gpuBlockIdx_y;
    int t = gpuThreadIdx_x;
    int lt = t % 32;
    int warp = t / 32;
    int kv_mul = n_heads / n_kv_heads;
    int kv_h = q_h / kv_mul;

    float attn_scale = (picolm_gpu_d_attn_scale > 0.0f) ? picolm_gpu_d_attn_scale : 1.0f / sqrtf((float)head_dim);
    size_t layer_base = (size_t)layer_ordinal * max_seq_len * kv_pos_stride_bytes;
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
                size_t head_off = layer_base + (size_t)kv_h * kv_head_stride_bytes
                    + (size_t)(t0 + ti) * kv_pos_stride_bytes;
                k_tile_sh[ti * head_dim + dim] = gpu_fp32_to_fp16(kv_row_dequant((const uint8_t *)kv_k, head_off, kv_type, dim));
                v_tile_sh[ti * head_dim + dim] = gpu_fp32_to_fp16(kv_row_dequant((const uint8_t *)kv_v, head_off - (size_t)kv_h * kv_head_stride_bytes + (size_t)kv_h * kv_head_stride_v_bytes, kv_type, dim));
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(d0), "+f"(d1), "+f"(d2_), "+f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                );
#endif

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
                        /* Sliding window: see picolm_gpu_attention_prefill_f32kv_kernel. */
                        if (n_swa > 0 && gk < start_pos + gq - n_swa + 1) continue;
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

/* Model-specified attention scale (granite.attention.scale etc.).
 * 0.0f => kernels fall back to 1/sqrt(head_dim). Defined in this TU so the
 * device link at compile time resolves picolm_gpu_d_attn_scale for the
 * attention kernels above (see the note at the top of this file). */
extern "C" void
picolm_gpu_set_attn_scale(float scale) {
    gpuMemcpyToSymbol(picolm_gpu_d_attn_scale, &scale, sizeof(float), 0,
                      gpuMemcpyHostToDevice);
}
