#ifndef BACKEND_GPU_GEMM_F16_CUH
#define BACKEND_GPU_GEMM_F16_CUH
/* ---------------------------------------------------------------------------
 * Prefill GEMM for quantised weights:  Y[s][o] = sum_k X[s][k] * W[o][k]
 *   X  fp32 [S][I]   activations (row stride x_stride floats)
 *   W  Q5_K / Q6_K / Q8_0, native GGUF layout, row o at w + o*row_bytes
 *   Y  fp32 [S][O]   (row stride y_stride floats)
 *
 * Why this and not another int8 kernel.  The int8 kernels it replaces issue one
 * m16n8k32 mma per 32 K-values and then run a float epilogue (scale, min
 * correction, I2F, accumulate) around every one of them, plus a decode that
 * reads weight bytes one at a time. Counting instructions, that is 70+ non-tensor
 * instructions per mma, so they are issue-bound at a few percent of int8 peak.
 * Here the quantised weight tile is dequantised ONCE per block tile into fp16 in
 * shared memory and the inner loop is nothing but ldmatrix + mma.m16n8k16 with
 * fp32 accumulation across the whole of K. There is no per-sub-block epilogue,
 * activations stay fp16 (no Q8_0 activation quantisation, so it is more accurate
 * than the path it replaces), and one kernel serves every quant format: only the
 * decode routine differs.
 *
 * Tiling.  A block computes BM outputs x BN tokens, 8 warps arranged 2 (tokens)
 * x 4 (outputs); with BM=BN=128 each warp owns 64 tokens x 32 outputs = 4x4
 * m16n8 tiles = 64 fp32 accumulators per lane. K is walked in slabs of 64.
 * Both operands are "row-major with k contiguous", so this is the same
 * ldmatrix/mma mapping as Q.K^T in the FA3 kernel (X plays Q, W plays K).
 *
 * Weights are fetched raw with cp.async, double-buffered one STAGE ahead:
 *   Q5_K / Q6_K: a stage is one 256-value superblock (4 slabs), 176 / 210 bytes
 *                per row. Q6_K superblocks are only 2-byte aligned, so they are
 *                copied through a 4-byte-aligned window (see issue_raw).
 *   Q8_0:        a stage is one slab (2 blocks of 34 bytes = 68 bytes per row).
 * Each slab is then decoded smem->smem into an XOR-swizzled fp16 tile; the
 * activation tile is converted fp32->fp16 on the way in (register-prefetched one
 * slab ahead, saturated to +-65504 so an outlier cannot become inf).
 *
 * Numerics: the weight decode uses the same fp32 formulas as the CPU
 * dequantize_row_* reference (then rounds once to fp16: relative error 2^-11,
 * far below the 2^-8 of the Q8_0 activation quantisation it removes). mma
 * accumulates in fp32. NOT bit-identical to the int8 kernels or the CPU.
 *
 * The same source is executed on the CPU by tests/gemm_emu.cpp
 * (-DPICOLM_FA3_EMU) so the real kernel logic is verified without a GPU.
 * ------------------------------------------------------------------------- */
#include "backend_gpu_mma.cuh"

#define GEMMF_THREADS 256
#define GEMMF_BK      64

enum { GEMMF_Q5_K = 5, GEMMF_IQ6_K = 7, GEMMF_Q6_K = 6, GEMMF_Q8_0 = 8,
       GEMMF_IQ4_K = 9 };

/* smem bytes per row per raw stage (includes Q6_K's alignment window; IQ6_K
 * blocks are 212 bytes too, but 4-byte aligned so no window is needed) */
static inline FA3_HD int gemmf_stage_bytes(int fmt) {
    return fmt == GEMMF_Q5_K ? 176 : (fmt == GEMMF_Q6_K || fmt == GEMMF_IQ6_K) ? 212
         : fmt == GEMMF_IQ4_K ? 144 : 68;
}
/* 64-wide K slabs per raw stage */
static inline FA3_HD int gemmf_slabs_per_stage(int fmt) { return fmt == GEMMF_Q8_0 ? 1 : 4; }
/* smem: X16 tile | W16 tile | 2 raw stages, tiles are rows of 64 fp16 = 128 bytes */
static inline FA3_HD size_t gemmf_smem_bytes(int fmt, int bm, int bn) {
    return (size_t)(bn + bm) * 128 + (size_t)2 * bm * gemmf_stage_bytes(fmt);
}

#if FA3_ENABLED

__device__ static inline uint32_t gemmf_ld16(const uint8_t *p) { return *(const uint16_t *)p; }
__device__ static inline float    gemmf_u8f(uint32_t w, int e) { return (float)((w >> (8 * e)) & 0xFFu); }

/* IQ6_K non-linear 6-bit LUT: the same 64 bytes as iq6nl_lut[] in quant.c (the
 * kernel devalues them by 128 at use, exactly like the CPU dequantiser). The
 * nvcc build does NOT define the table here -- a file-scope __device__ array
 * referenced from a __global__ *template* makes nvcc's cudafe MSVC host pass
 * fail with "C2039: is not a member of `global namespace'" -- instead it uses
 * iq6nl_lut_dev from backend_gpu_dequant.cuh, which backend_gpu_kernels.cu
 * includes (via backend_gpu_kernels.cuh) before this header, and which is
 * already proven to compile in that position (picolm_iq6_K_gemv uses it).
 * Only the emulator (tests/gemm_emu.cpp, standalone, PICOLM_FA3_EMU) needs its
 * own copy; it verifies the table against quant.c bit for bit. */
#ifdef PICOLM_FA3_EMU
static const uint8_t gemmf_iq6k_lut[64] = {
       1,   7,  13,  19,  24,  30,  35,  40,  44,  49,  54,  58,  62,  66,  70,  74,
      77,  81,  84,  88,  91,  94,  97, 100, 103, 106, 109, 112, 115, 117, 120, 123,
     126, 128, 131, 134, 137, 140, 142, 145, 148, 151, 155, 158, 161, 164, 168, 172,
     175, 179, 183, 187, 191, 196, 200, 205, 210, 215, 220, 226, 231, 237, 243, 249,
};
#else
#define gemmf_iq6k_lut iq6nl_lut_dev
#endif

/* IQ4_K 32-entry LUT (iq4k_values in quant.c): same proven pattern as the
 * IQ6_K table above -- alias the device copy in backend_gpu_dequant.cuh for
 * the nvcc build, local static copy for the standalone emulator. */
#ifdef PICOLM_FA3_EMU
static const int8_t gemmf_iq4k_lut[32] = {
   -127, -104,  -83,  -65,  -49,  -35,  -22,  -10,    1,   13,   25,   38,   53,   69,   89,  113,
   -123, -100,  -79,  -61,  -45,  -31,  -18,   -6,    5,   17,   29,   42,   57,   73,   93,  117,
};
#else
#define gemmf_iq4k_lut iq4k_values_dev
#endif

template <int FMT, int BM, int BN>
__global__ void __launch_bounds__(GEMMF_THREADS, 1)
picolm_gpu_gemm_f16_kernel(float *__restrict__ y, const float *__restrict__ x,
                           const uint8_t *__restrict__ w,
                           int S, int I, int O, size_t row_bytes, int x_stride, int y_stride)
{
    static_assert(BM == 64 || BM == 128, "BM");
    static_assert(BN == 32 || BN == 64 || BN == 128, "BN");
    constexpr int MT = BN / 32;                       /* m16 token tiles per warp */
    constexpr int NT = BM / 32;                       /* n8 output tiles per warp */
    constexpr int XU = BN * 8 / GEMMF_THREADS;        /* X chunks (8 values) per thread per slab */
    constexpr int WU = BM * 8 / GEMMF_THREADS;        /* W chunks per thread per slab */
    constexpr int CB = FMT == GEMMF_Q5_K ? 176 : (FMT == GEMMF_Q6_K || FMT == GEMMF_IQ6_K) ? 212
                 : FMT == GEMMF_IQ4_K ? 144 : 68;
    constexpr int SPS = FMT == GEMMF_Q8_0 ? 1 : 4;    /* slabs per raw stage */

    FA3_DYN_SMEM_PTR(sm);
    uint4  *X4  = (uint4 *)sm;
    uint4  *W4  = X4 + BN * 8;
    uint8_t *raw = sm + (size_t)(BN + BM) * 128;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tg = lane & 3;
    const int wsx = warp & 1, wox = warp >> 1;        /* token half, output quarter */
    const int tile_s0 = blockIdx.x * BN, tile_o0 = blockIdx.y * BM;
    const int xs = x_stride > 0 ? x_stride : I;
    const int ys = y_stride > 0 ? y_stride : O;
    const int n_slabs = I / GEMMF_BK;
    const int n_stages = n_slabs / SPS;

    /* zero the raw staging area once: rows past O are never fetched, and a stale NaN
     * pattern there would otherwise be decoded (harmless to valid outputs, since each
     * output depends only on its own weight row, but it keeps the data defined) */
    for (int i = tid; i < (int)((size_t)2 * BM * CB / 16); i += GEMMF_THREADS)
        ((uint4 *)raw)[i] = make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();

    /* ---- raw weight stage -> smem (async) ---- */
    auto issue_raw = [&](int c, int buf) {
        uint8_t *dst0 = raw + (size_t)buf * BM * CB;
        if (FMT == GEMMF_Q5_K) {
            for (int u = tid; u < BM * 11; u += GEMMF_THREADS) {
                const int r = u / 11, k = u - r * 11, o = tile_o0 + r;
                if (o < O) fa3_cp16(dst0 + r * CB + k * 16, w + (size_t)o * row_bytes + (size_t)c * 176 + k * 16);
            }
        } else if (FMT == GEMMF_Q8_0) {
            for (int u = tid; u < BM * 17; u += GEMMF_THREADS) {
                const int r = u / 17, k = u - r * 17, o = tile_o0 + r;
                if (o < O) fa3_cp4(dst0 + r * CB + k * 4, w + (size_t)o * row_bytes + (size_t)c * 68 + k * 4);
            }
        } else if (FMT == GEMMF_IQ6_K) {
            /* IQ6_K block = 212 bytes = 53 four-byte chunks. 212 and row_bytes
             * (= n_blocks * 212) are multiples of 4, so every chunk is naturally
             * 4-byte aligned (the launcher guards the base pointer) and the plain
             * Q8_0-style cp4 loop covers the whole stage -- no Q6_K-style window. */
            for (int u = tid; u < BM * 53; u += GEMMF_THREADS) {
                const int r = u / 53, k = u - r * 53, o = tile_o0 + r;
                if (o < O) fa3_cp4(dst0 + r * CB + k * 4, w + (size_t)o * row_bytes + (size_t)c * 212 + k * 4);
            }
        } else if (FMT == GEMMF_IQ4_K) {
            /* IQ4_K block = 144 bytes = 36 four-byte chunks, every chunk
             * 4-byte aligned (144 = 16*9, so row_bytes is 16-aligned when the
             * base pointer is). */
            for (int u = tid; u < BM * 36; u += GEMMF_THREADS) {
                const int r = u / 36, k = u - r * 36, o = tile_o0 + r;
                if (o < O) fa3_cp4(dst0 + r * CB + k * 4, w + (size_t)o * row_bytes + (size_t)c * 144 + k * 4);
            }
        } else {
            /* Q6_K superblock = 210 bytes at a 2-byte-aligned address: copy through a
             * 4-byte-aligned window starting at A & ~3. The payload then sits at
             * offset (A & 3) inside the row's smem stage. The last 2 bytes of a
             * window that starts aligned are copied with a plain load/store so the
             * copy never reads past the end of the tensor. */
            for (int u = tid; u < BM * 53; u += GEMMF_THREADS) {
                const int r = u / 53, k = u - r * 53, o = tile_o0 + r;
                if (o < O) {
                    const size_t A = (size_t)o * row_bytes + (size_t)c * 210;
                    const size_t wa = A & ~(size_t)3;
                    const int nb = (int)(A & 3) + 210, fw = nb >> 2;
                    if (k < fw) fa3_cp4(dst0 + r * CB + k * 4, w + wa + k * 4);
                    else if ((nb & 3) && k == fw)
                        *(uint16_t *)(dst0 + r * CB + k * 4) = *(const uint16_t *)(w + wa + k * 4);
                }
            }
        }
    };

    /* ---- activation slab: fp32 global -> registers (prefetch), then -> fp16 smem ---- */
    float4 xr[XU][2];
    auto load_x = [&](int slab) {
        const int k0 = slab * GEMMF_BK;
        for (int i = 0; i < XU; i++) {
            const int u = tid + i * GEMMF_THREADS, r = u >> 3, c = u & 7, s = tile_s0 + r;
            if (s < S) {
                const float4 *p = (const float4 *)(x + (size_t)s * xs + k0 + c * 8);
                xr[i][0] = p[0]; xr[i][1] = p[1];
            } else {
                xr[i][0] = make_float4(0.f, 0.f, 0.f, 0.f); xr[i][1] = xr[i][0];
            }
        }
    };
    auto sat = [](float v) { return fminf(fmaxf(v, -65504.f), 65504.f); };   /* NaN propagates */
    auto store_x = [&]() {
        for (int i = 0; i < XU; i++) {
            const int u = tid + i * GEMMF_THREADS, r = u >> 3, c = u & 7;
            X4[FA3_SW(r, c, 8)] = make_uint4(
                fa3_pack2(sat(xr[i][0].x), sat(xr[i][0].y)), fa3_pack2(sat(xr[i][0].z), sat(xr[i][0].w)),
                fa3_pack2(sat(xr[i][1].x), sat(xr[i][1].y)), fa3_pack2(sat(xr[i][1].z), sat(xr[i][1].w)));
        }
    };

    /* ---- weight slab: raw smem -> fp16 swizzled tile. `sub` = slab within the stage ---- */
    auto decode = [&](int c, int sub, int buf) {
        const uint8_t *rawb = raw + (size_t)buf * BM * CB;
        for (int i = 0; i < WU; i++) {
            const int u = tid + i * GEMMF_THREADS, r = u >> 3, c8 = u & 7;
            const uint8_t *rr = rawb + (size_t)r * CB;
            uint32_t o4[4];
            if (FMT == GEMMF_Q5_K) {
                /* values 0..31 of the slab = low nibbles of qs[32*sub + l] (sub-block 2*sub),
                 * values 32..63 = high nibbles (sub-block 2*sub+1); the 5th bit of value l in
                 * sub-block jj is bit jj of qh[l]. */
                const int half = c8 >> 2, l0 = (c8 & 3) * 8, jj = 2 * sub + half;
                const uint2 qs = *(const uint2 *)(rr + 48 + 32 * sub + l0);
                const uint2 qh = *(const uint2 *)(rr + 16 + l0);
                const uint32_t sh = 4u * (uint32_t)half, bp = (uint32_t)jj;
                const uint32_t qa = ((qs.x >> sh) & 0x0F0F0F0Fu) | (((qh.x >> bp) & 0x01010101u) << 4);
                const uint32_t qb = ((qs.y >> sh) & 0x0F0F0F0Fu) | (((qh.y >> bp) & 0x01010101u) << 4);
                const uint8_t *p = rr + 4;                         /* 12 packed 6-bit scales/mins */
                uint32_t sc, mn;
                if (jj < 4) { sc = p[jj] & 63u; mn = p[jj + 4] & 63u; }
                else {
                    sc = (uint32_t)((p[jj + 4] & 0xF) | ((p[jj - 4] >> 6) << 4));
                    mn = (uint32_t)((p[jj + 4] >> 4) | ((p[jj] >> 6) << 4));
                }
                const float d  = fa3_f16_to_f32(gemmf_ld16(rr)), dm = fa3_f16_to_f32(gemmf_ld16(rr + 2));
                const float sa = d * (float)sc, sb = dm * (float)mn;
                o4[0] = fa3_pack2(sa * gemmf_u8f(qa, 0) - sb, sa * gemmf_u8f(qa, 1) - sb);
                o4[1] = fa3_pack2(sa * gemmf_u8f(qa, 2) - sb, sa * gemmf_u8f(qa, 3) - sb);
                o4[2] = fa3_pack2(sa * gemmf_u8f(qb, 0) - sb, sa * gemmf_u8f(qb, 1) - sb);
                o4[3] = fa3_pack2(sa * gemmf_u8f(qb, 2) - sb, sa * gemmf_u8f(qb, 3) - sb);
            } else if (FMT == GEMMF_Q8_0) {
                const uint8_t *bp = rr + (c8 >> 2) * 34;            /* 2 blocks per slab */
                const int l0 = (c8 & 3) * 8;
                const float d = fa3_f16_to_f32(gemmf_ld16(bp));
                float v[8];
                for (int e = 0; e < 4; e++) {
                    const uint32_t q = gemmf_ld16(bp + 2 + l0 + 2 * e);
                    v[2 * e]     = d * (float)(int)(int8_t)(uint8_t)(q & 0xFFu);
                    v[2 * e + 1] = d * (float)(int)(int8_t)(uint8_t)(q >> 8);
                }
                for (int e = 0; e < 4; e++) o4[e] = fa3_pack2(v[2 * e], v[2 * e + 1]);
            } else if (FMT == GEMMF_IQ6_K) {
                /* IQ6_K (GGUF 141): 212-byte block per 256 values = 4 slabs of 64.
                 * block_iq6_k layout: d(2) extra(2) scales[16] qs[128] qh[64].
                 * dequantize_row_iq6_k() fills dst[j], dst[j+16], dst[j+32],
                 * dst[j+48] of sub-block `sub` from groups g=0..3:
                 *   g=0: qs[j]    low nibble, qh[j]    bits shift..shift+1
                 *   g=1: qs[j+16] low nibble, qh[j+16] bits shift..shift+1
                 *   g=2: qs[j]    high nibble, qh[j]    bits shift+2..shift+3
                 *   g=3: qs[j+16] high nibble, qh[j+16] bits shift+2..shift+3
                 * with shift = 4*(sub&1) and the qh base advanced by 32 for sub>=2;
                 * q = nibble | hi2<<4, value = d*scales[4*sub+g] * (lut[q]-128+m),
                 * m = bit (4*sub+g) of the uint16 `extra` at byte 2.
                 * The thread's 8 values (c8*8..c8*8+7) share one group g = c8>>1
                 * and j0 = (c8&1)*8, and the qs/qh BYTE index is the same
                 * (j0 + t + 16*(g&1)) for both, so two uint32 loads cover them.
                 * Same fp32 op order as the reference (d*sc first, then one
                 * multiply by lut-128+m), so the emulator's identity test
                 * matches dequantize_row_iq6_k bit for bit. */
                const int g = c8 >> 1, j0 = (c8 & 1) * 8;
                const float d = fa3_f16_to_f32(gemmf_ld16(rr));
                const float dl = d * (float)(int8_t)rr[4 + 4 * sub + g];
                const int eb = 4 * sub + g;
                const int m = (int)((rr[2 + (eb >> 3)] >> (eb & 7)) & 1u);
                const int shn = (g & 2) ? 4 : 0;          /* low vs high nibble */
                const int shh = 4 * (sub & 1) + (g & 2);  /* 2-bit field in qh   */
                const int qo = 20 + 32 * sub + 16 * (g & 1) + j0;
                const int ho = 148 + 32 * (sub >> 1) + 16 * (g & 1) + j0;
                const uint32_t qs0 = *(const uint32_t *)(rr + qo);
                const uint32_t qs1 = *(const uint32_t *)(rr + qo + 4);
                const uint32_t qh0 = *(const uint32_t *)(rr + ho);
                const uint32_t qh1 = *(const uint32_t *)(rr + ho + 4);
                const uint32_t qa = ((qs0 >> shn) & 0x0F0F0F0Fu) | (((qh0 >> shh) & 0x03030303u) << 4);
                const uint32_t qb = ((qs1 >> shn) & 0x0F0F0F0Fu) | (((qh1 >> shh) & 0x03030303u) << 4);
                const uint32_t qs4[2] = { qa, qb };
                float v[8];
                for (int t = 0; t < 8; t++) {
                    const uint32_t q = (qs4[t >> 2] >> (8 * (t & 3))) & 0xFFu;
                    v[t] = dl * (float)((int)gemmf_iq6k_lut[q] - 128 + m);
                }
                for (int e = 0; e < 4; e++) o4[e] = fa3_pack2(v[2 * e], v[2 * e + 1]);
            } else if (FMT == GEMMF_IQ4_K) {
                /* IQ4_K (GGUF 139): 144-byte block per 256 values = 4 slabs of
                 * 64. block_iq4_k layout: d(2) extra(2) scales_h[4] scales_l[8]
                 * qs[128]. dequantize_row_iq4_k() per 32-value sub-block jj:
                 *   sh  = scales_h[jj/2] >> (4*(jj%2))         (low 2 bits used)
                 *   dl1 = d * (((scales_l[jj] & 0xf) | ((sh << 4) & 0x30)) - 32)
                 *   dl2 = d * (((scales_l[jj] >> 4)   | ((sh << 2) & 0x30)) - 32)
                 *   values  0..15 = dl1 * lut[(extra>>(2*jj)   & 1)*16 + qs[j]&0xf]
                 *   values 16..31 = dl2 * lut[(extra>>(2*jj+1) & 1)*16 + qs[j]>>4]
                 * lut = iq4k_values (32 entries), qs advances 16 bytes per
                 * sub-block. Thread mapping: within the sub-block, elements
                 * 0..15 are the LOW nibbles of qs[0..15] and 16..31 the HIGH
                 * nibbles of the SAME 16 bytes (unlike Q5_K, whose halves use
                 * different bytes). Thread c8 covers slab values 8*c8..8*c8+7:
                 *   jj = 2*sub + (c8>>2)   sub-block
                 *   lo = (c8>>1)&1         low/high nibble half
                 *   bo = (c8&1)*8          byte offset 0 or 8 in qs[16*jj..]
                 * so the 8 values are one nibble of each of 8 consecutive
                 * bytes = two uint32 loads. All 16 bytes are covered because
                 * bo spans 0,8 within each half. */
                const int jj = 2 * sub + (c8 >> 2);
                const int lo = (c8 >> 1) & 1, bo = (c8 & 1) * 8;
                const float d = fa3_f16_to_f32(gemmf_ld16(rr));
                const uint8_t shb = rr[4 + (jj >> 1)], slb = rr[8 + jj];
                /* 4-bit window for sub-block jj; scale1 (lo=0) takes its bits
                 * 0-1, scale2 (lo=1) its bits 2-3 (matches (sh<<4)&0x30 and
                 * (sh<<2)&0x30 in dequantize_row_iq4_k) */
                const uint32_t shw = (uint32_t)(shb >> (4 * (jj & 1))) & 0xFu;
                const uint32_t sc = lo ? ((slb >> 4) | ((shw & 0xCu) << 2))
                                       : ((slb & 0xfu) | ((shw & 3u) << 4));
                const float dl = d * (float)((int)sc - 32);
                const int tb = (int)((gemmf_ld16(rr + 2) >> (2 * jj + lo)) & 1u) << 4;
                const uint32_t qa = *(const uint32_t *)(rr + 16 + 16 * jj + bo);
                const uint32_t qb = *(const uint32_t *)(rr + 16 + 16 * jj + bo + 4);
                const uint32_t qs4[2] = { qa, qb };
                float v[8];
                for (int t = 0; t < 8; t++) {
                    const uint32_t byte = (qs4[t >> 2] >> (8 * (t & 3))) & 0xFFu;
                    v[t] = dl * (float)(int)gemmf_iq4k_lut[tb + ((byte >> (4 * lo)) & 0xFu)];
                }
                for (int e = 0; e < 4; e++) o4[e] = fa3_pack2(v[2 * e], v[2 * e + 1]);
            } else {
                /* Q6_K: slab = 64 values = two 32-groups of one 128-half. gg = group + 2*pr
                 * selects y[l + 32*gg] of the reference: ql nibble (lo for gg<2, hi for gg>=2),
                 * ql byte offset 32*(gg&1), qh bits 2*gg..2*gg+1, scale index 8h + 2*gg + l/16. */
                const int h = sub >> 1, pr = sub & 1, group = c8 >> 2, l0 = (c8 & 3) * 8, gg = group + 2 * pr;
                const size_t A = (size_t)(tile_o0 + r) * row_bytes + (size_t)c * 210;
                const uint8_t *r6 = rr + (A & 3);                   /* payload start inside the window */
                const uint8_t *qlp = r6 + 64 * h + 32 * group + l0;
                const uint8_t *qhp = r6 + 128 + 32 * h + l0;
                const uint32_t ql0 = gemmf_ld16(qlp)     | (gemmf_ld16(qlp + 2) << 16);
                const uint32_t ql1 = gemmf_ld16(qlp + 4) | (gemmf_ld16(qlp + 6) << 16);
                const uint32_t qh0 = gemmf_ld16(qhp)     | (gemmf_ld16(qhp + 2) << 16);
                const uint32_t qh1 = gemmf_ld16(qhp + 4) | (gemmf_ld16(qhp + 6) << 16);
                const uint32_t ns = pr ? 4u : 0u, hs = 2u * (uint32_t)gg;
                const uint32_t qa = ((ql0 >> ns) & 0x0F0F0F0Fu) | (((qh0 >> hs) & 0x03030303u) << 4);
                const uint32_t qb = ((ql1 >> ns) & 0x0F0F0F0Fu) | (((qh1 >> hs) & 0x03030303u) << 4);
                const int sci = (int)(int8_t)r6[192 + 8 * h + 2 * gg + (l0 >> 4)];
                const float d = fa3_f16_to_f32(gemmf_ld16(r6 + 208));
                const float ds = d * (float)sci;                    /* d * sc, then * q (reference order) */
                o4[0] = fa3_pack2(ds * (gemmf_u8f(qa, 0) - 32.f), ds * (gemmf_u8f(qa, 1) - 32.f));
                o4[1] = fa3_pack2(ds * (gemmf_u8f(qa, 2) - 32.f), ds * (gemmf_u8f(qa, 3) - 32.f));
                o4[2] = fa3_pack2(ds * (gemmf_u8f(qb, 0) - 32.f), ds * (gemmf_u8f(qb, 1) - 32.f));
                o4[3] = fa3_pack2(ds * (gemmf_u8f(qb, 2) - 32.f), ds * (gemmf_u8f(qb, 3) - 32.f));
            }
            W4[FA3_SW(r, c8, 8)] = make_uint4(o4[0], o4[1], o4[2], o4[3]);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int a = 0; a < MT; a++)
#pragma unroll
        for (int b = 0; b < NT; b++) { acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f; }

    /* ldmatrix lane addressing: A = X rows (like Q in FA3), B = W rows (like K in FA3) */
    const int lj = lane >> 3, lr = lane & 7;
    const int a_row0 = wsx * (BN / 2) + (lj & 1) * 8 + lr;       /* + mt*16 */
    const int b_row0 = wox * (BM / 4) + (lj >> 1) * 8 + lr;      /* + np*16 */

    issue_raw(0, 0);
    fa3_cp_commit();
    load_x(0);

    for (int c = 0; c < n_stages; c++) {
        if (c + 1 < n_stages) issue_raw(c + 1, (c + 1) & 1);
        fa3_cp_commit();                 /* always commit so wait_group counting stays uniform */
        fa3_cp_wait1();                  /* this thread's copies for stage c have landed */
        __syncthreads();                 /* ...and everyone's are visible */

        for (int sub = 0; sub < SPS; sub++) {
            const int slab = c * SPS + sub;
            decode(c, sub, c & 1);
            store_x();
            if (slab + 1 < n_slabs) load_x(slab + 1);   /* prefetch next slab; consumed after the barriers */
            __syncthreads();

#pragma unroll
            for (int ks = 0; ks < GEMMF_BK / 16; ks++) {
                uint32_t a[MT][4];
#pragma unroll
                for (int mt = 0; mt < MT; mt++)
                    fa3_ldsm_x4(a[mt], &X4[(a_row0 + mt * 16) * 8 + ((2 * ks + (lj >> 1)) ^ lr)]);
#pragma unroll
                for (int np = 0; np < NT / 2; np++) {
                    uint32_t b[4];
                    fa3_ldsm_x4(b, &W4[(b_row0 + np * 16) * 8 + ((2 * ks + (lj & 1)) ^ lr)]);
#pragma unroll
                    for (int mt = 0; mt < MT; mt++) {
                        fa3_mma(acc[mt][2 * np],     a[mt], b[0], b[1]);
                        fa3_mma(acc[mt][2 * np + 1], a[mt], b[2], b[3]);
                    }
                }
            }
            __syncthreads();             /* all warps done with this slab before it is overwritten */
        }
    }

    /* ---- epilogue: C fragment -> y (rows g / g+8, columns 2*tg, 2*tg+1) ---- */
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const int s0 = tile_s0 + wsx * (BN / 2) + mt * 16 + g;
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const int o = tile_o0 + wox * (BM / 4) + nt * 8 + tg * 2;
            if (o < O) {
                if (s0 < S)     *(float2 *)(y + (size_t)s0 * ys + o)       = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
                if (s0 + 8 < S) *(float2 *)(y + (size_t)(s0 + 8) * ys + o) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
            }
        }
    }
}

#else  /* !FA3_ENABLED: host pass / pre-sm_80 device pass: empty stub so the launcher links */
template <int FMT, int BM, int BN>
__global__ void picolm_gpu_gemm_f16_kernel(float *__restrict__, const float *__restrict__,
                                           const uint8_t *__restrict__,
                                           int, int, int, size_t, int, int) {}
#endif /* FA3_ENABLED */

#endif /* BACKEND_GPU_GEMM_F16_CUH */
