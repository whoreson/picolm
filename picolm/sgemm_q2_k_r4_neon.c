/* ============================================================
 * Q2_K_R4 x Q8_K NEON GEMM for ARM (GGUF type 210)
 *
 * Why this file exists
 * --------------------
 * q2_k_r4 is the 4-row-interleaved Q2_K used by ik_llama.cpp. Until now PicoLM
 * had NO GEMM for it: type_dispatch.c left fn_gemm = NULL on ARM, so prefill
 * fell all the way back to gemv_q2_k_r4(), which dequantises a row to FLOAT and
 * then does a scalar dot. That is 17.8 million calls per Fimbulvetr prefill
 * pass, each with its own malloc, and measured 0.2 tok/s (186-250 s per pass).
 *
 * What the interleave buys
 * ------------------------
 * This is a DECODE-format win, not a prefill one -- worth stating because it is
 * easy to get backwards. Measured on the plain Q2_K kernels (m=4096, k=4096):
 *
 *     n      per-column vec_dot        tiled GEMM
 *     1           9.0 GFLOPS             n/a
 *     4          21.9                   28.7
 *     44         21.6                   36.1
 *
 * The per-column path is flat from n=4 up (it re-decodes the weight block once
 * per column), and the tiled GEMM is flat from n~16 up (decode amortises over
 * the tile's columns). At n=1 there are no columns to amortise over, so the
 * only way to reuse a decoded weight block is to have 4 weight rows share it.
 * That is exactly what R4 provides.
 *
 * Layout (must match dequantize_row_q2_k_r4_single() in quant.c, which is the
 * verified reference -- it was cross-checked against ik_llama.cpp):
 *
 *   d[0..3]      fp16 super-block scale, one per row
 *   d[4..7]      fp16 dmin,                one per row
 *   scales[8*ib + row + 0]  (min<<4 | scale) for values  0..15 of sub-block ib
 *   scales[8*ib + row + 4]  (min<<4 | scale) for values 16..31
 *   qs[32*ib + 4*row + i]        2-bit, shifts 0/2/4/6 -> values i, i+4, i+8, i+12
 *   qs[32*ib + 4*row + i + 16]   same for values 16+i, 20+i, 24+i, 28+i
 *
 * So one 16-byte load at qs + 32*ib yields, per row, 4 bytes = 16 values, and
 * the four rows occupy 4-byte groups. Value order within a row is strided by 4,
 * which is why the scale for the whole 16-value group is uniform (scales[..+0]),
 * and the second half uses scales[..+4].
 *
 * Structure
 * ---------
 *  - 4 weight rows x N activation columns. Decode happens ONCE per (block, row)
 *    and feeds all N columns, which is the amortisation the format exists for.
 *  - DOTPROD (`vdotq_s32`) preferred, I8MM and vmull/vpaddl as fallbacks, same
 *    as sgemm_k_q8k_neon_fast.c.
 *  - dmin folded via the activation's bsums, exactly as the Q8_K reference does.
 * ============================================================ */
#include "quant.h"
#include "sgemm.h"

#if defined(__ARM_NEON)

#include <string.h>

/* 16-value int8 dot -> 4 lane partials. */
static inline int32x4_t q2kr4_dot(int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(vdupq_n_s32(0), a, b);
#elif defined(__ARM_FEATURE_MATMUL_INT8)
    return vmmlaq_s32(vdupq_n_s32(0), a, b);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(a),  vget_low_s8(b));
    int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
#endif
}

static inline int32_t q2kr4_fold(int32x4_t v) {
#if defined(__ARM_FEATURE_MATMUL_INT8) && !defined(__ARM_FEATURE_DOTPROD)
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 3);
#else
    return vaddvq_s32(v);
#endif
}

/* Scaled-and-accumulated dot, keeping lanes intact (no horizontal reduction). */
static inline int32x4_t q2kr4_acc(int32x4_t acc, int8x16_t a, int8x16_t b, int32_t sc) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), a, b), sc);
#elif defined(__ARM_FEATURE_MATMUL_INT8)
    (void)sc;
    return vmmlaq_s32(acc, a, b);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(a),  vget_low_s8(b));
    int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vmulq_n_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)), sc));
#endif
}

#define Q2KR4_TILE_N 8

/* 4 weight rows x N activation columns, whole k. */
/* Build the four rows' 16-value int8 vectors for ONE half of one sub-block.
 *
 * The 16-byte group at qs + 32*sb + (half ? 16 : 0) is laid out
 *     [ r0b0 r0b1 r0b2 r0b3 | r1b0..r1b3 | r2b0..r2b3 | r3b0..r3b3 ]
 * so one vand/vshr pass yields the whole group's values for ALL FOUR rows at
 * once -- this is the saving the R4 interleave buys. What each register then
 * holds is, per lane group, one row's four bytes for a single shift; a vzip
 * turns that into a contiguous 16-value vector per row, ready for one vdotq.
 *
 * Value order within a row is: shift 0 gives values 0..3, shift 1 gives 4..7,
 * shift 2 gives 8..11, shift 3 gives 12..15. So out[r] = [v0..v15]. */
static inline void q2kr4_unpack_half(const uint8_t *g, int8x16_t out[4]) {
    const uint8x16_t m3 = vdupq_n_u8(3);
    const uint8x16_t x  = vld1q_u8(g);
    const int8x16_t s0 = vreinterpretq_s8_u8(vandq_u8(x, m3));
    const int8x16_t s1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x, 2), m3));
    const int8x16_t s2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x, 4), m3));
    const int8x16_t s3 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x, 6), m3));

    /* Each sX = [r0_i0 r0_i1 r0_i2 r0_i3 | r1... | r2... | r3...], i.e. the
     * 32-bit lanes are rows. Two zip stages turn that into one contiguous
     * 16-value vector per row:
     *   vzip1q_s32(s0,s1) -> [r0_s0 | r0_s1 | r2_s0 | r2_s1]
     *   vzip1q_s64(that, vzip1q_s32(s2,s3)) -> [r0_s0 | r0_s1 | r0_s2 | r0_s3]
     * which is row 0's values 0..15 in q8 order. vzip2q_* selects rows 2 and 3,
     * and the same pair built from rows 1 and 3 in the second zip gives row 1.
     * (Verified by permutation test; the naive "zip4" ordering is wrong here.) */
    const int32x4_t a02 = vzip1q_s32(vreinterpretq_s32_s8(s0), vreinterpretq_s32_s8(s1));
    const int32x4_t b02 = vzip1q_s32(vreinterpretq_s32_s8(s2), vreinterpretq_s32_s8(s3));
    const int32x4_t a13 = vzip2q_s32(vreinterpretq_s32_s8(s0), vreinterpretq_s32_s8(s1));
    const int32x4_t b13 = vzip2q_s32(vreinterpretq_s32_s8(s2), vreinterpretq_s32_s8(s3));

    /* a02 carries rows {0,2} and a13 carries rows {1,3}. vzip1q_s64 over
     * (a02,b02) therefore yields row 0 in the low half and row 2 in the high
     * half; vzip2q_s64 the same for the second 8-byte group, i.e. rows 2/3's
     * remaining values. The row indices come out as {0,2} then {1,3}, NOT
     * {0,1} then {2,3} -- writing them in ascending order silently swaps rows
     * 1 and 2. */
    out[0] = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s32(a02),
                                             vreinterpretq_s64_s32(b02)));
    out[2] = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s32(a13),
                                             vreinterpretq_s64_s32(b13)));
    out[1] = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s32(a02),
                                             vreinterpretq_s64_s32(b02)));
    out[3] = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s32(a13),
                                             vreinterpretq_s64_s32(b13)));
}

static void q2kr4_tile(const block_q2_k_r4 *A,
                       const block_q8_K *B, int ldb,
                       float *C, int ldc, int k_blocks,
                       int i0, int i1, int jc0, int jc1) {
    for (int jc = jc0; jc < jc1; jc += Q2KR4_TILE_N) {
        const int nn = (jc + Q2KR4_TILE_N <= jc1) ? Q2KR4_TILE_N : (jc1 - jc);

        for (int ib = i0; ib < i1; ib += 4) {
            if (ib + 4 > i1) break;         /* R4 needs all 4 rows */

            float acc[4][Q2KR4_TILE_N];
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < nn; ++c) acc[r][c] = 0.0f;

            for (int l = 0; l < k_blocks; ++l) {
                const block_q2_k_r4 *aq = &A[(size_t)(ib / 4) * k_blocks + l];

                float dr[4], dmr[4];
                for (int r = 0; r < 4; ++r) {
                    dr[r]  = fp16_to_fp32_lookup(aq->d[r]);
                    dmr[r] = fp16_to_fp32_lookup(aq->d[r + 4]);
                }

                const block_q8_K *bq[Q2KR4_TILE_N];
                float bd[Q2KR4_TILE_N];
                for (int c = 0; c < nn; ++c) {
                    bq[c] = &B[(size_t)ldb * (jc + c) + l];
                    bd[c] = bq[c]->d;
                }

                /* Integer accumulators live ACROSS the whole QK_K block so the
                 * per-sub-block scale folds into the lanes and there is no
                 * horizontal reduction inside the k loop. One fold per
                 * (row, col) at the very end, plus one for the dmin term.
                 * Integer, because both the q2 values and the q8 values are
                 * int8 and the scales are small. */
                /* ONE int32x4 per (row,col): both halves of every sub-block
                 * accumulate into the same lanes, since both are scaled by a
                 * per-sub-block constant and the lanes are independent
                 * partials. Halving the accumulator count (32 instead of 64 for
                 * an 8-wide column tile) is what keeps them out of the stack. */
                int32x4_t isum[4][Q2KR4_TILE_N];
                int32_t  bacc[4][Q2KR4_TILE_N];
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < nn; ++c) {
                        isum[r][c] = vdupq_n_s32(0);
                        bacc[r][c] = 0;
                    }

                for (int sb = 0; sb < 8; ++sb) {
                    const uint8_t *qsw = aq->qs + 32 * sb;

                    /* Decode both halves for all four rows at once: this is the
                     * 4x the R4 interleave buys. */
                    int8x16_t v0[4], v1[4];
                    q2kr4_unpack_half(qsw,      v0);
                    q2kr4_unpack_half(qsw + 16, v1);

                    for (int r = 0; r < 4; ++r) {
                        const int16_t s0 = (int16_t)(aq->scales[8 * sb + r + 0] & 0x0F);
                        const int16_t s1 = (int16_t)(aq->scales[8 * sb + r + 4] & 0x0F);
                        const int16_t m0 = (int16_t)(aq->scales[8 * sb + r + 0] >> 4);
                        const int16_t m1 = (int16_t)(aq->scales[8 * sb + r + 4] >> 4);
                        for (int c = 0; c < nn; ++c) {
                            const int8_t *q8c = bq[c]->qs + 32 * sb;
                            isum[r][c] = q2kr4_acc(isum[r][c], v0[r],
                                                   vld1q_s8(q8c),      s0);
                            isum[r][c] = q2kr4_acc(isum[r][c], v1[r],
                                                   vld1q_s8(q8c + 16), s1);
                            bacc[r][c] += m0 * (int32_t)bq[c]->bsums[2 * sb]
                                        + m1 * (int32_t)bq[c]->bsums[2 * sb + 1];
                        }
                    }
                }
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < nn; ++c) {
                        const int32_t si  = q2kr4_fold(isum[r][c]);
                        acc[r][c] += dr[r] * bd[c] * (float)si
                                   - dmr[r] * bd[c] * (float)bacc[r][c];
                    }
            }
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < nn; ++c)
                    C[(size_t)ldc * (jc + c) + (ib + r)] = acc[r][c];
        }
    }
}

int sgemm_q2_k_r4_q8_k_neon_fast(int nrows, int ncols, int k,
                                 const void *vx, const void *vy, float *out,
                                 size_t bs, int ith, int nth) {
    if (nrows < 4 || ncols < 1 || k <= 0 || (k % QK_K) != 0 || (nrows % 4) != 0)
        return 0;
    const int nb = k / QK_K;
    const int64_t tiles = nrows / 4;
    const int64_t per = (tiles + nth - 1) / nth;
    const int64_t t0 = (int64_t)ith * per;
    int64_t t1 = t0 + per; if (t1 > tiles) t1 = tiles;
    if (t0 >= t1) return 0;
    q2kr4_tile((const block_q2_k_r4 *)vx, (const block_q8_K *)vy, nb,
               out, (int)bs, nb, (int)(t0 * 4), (int)(t1 * 4), 0, ncols);
    return (int)((t1 - t0) * 4);
}

#endif /* __ARM_NEON */
