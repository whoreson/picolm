/* ============================================================
 * Q2_K x Q8_K NEON prefill GEMM  (ARM: plain NEON / DOTPROD / I8MM)
 *
 * Why this file exists: on ARM the Q2_K type had no batched GEMM at all --
 * type_dispatch.c set fn_gemm = NULL and every prefill token scored through
 * a per-column vec_dot_q2_K_q8_K() call, so each weight block's 2-bit unpack
 * and scale decode was redone once per token. Q2_K is the dominant type in
 * low-bit models (Fimbulvetr-Q2_K: 193 of 338 weight tensors), so that showed
 * up as ~2 tok/s prefill. This kernel decodes each weight sub-block ONCE and
 * applies it to R_T rows of activations.
 *
 * Tile: 4 weight rows x 4 activation rows.
 *   - the 4 activation rows' q8 vectors are loaded once per sub-block and
 *     reused by all 4 weight rows;
 *   - the 4 weight rows' 2-bit groups are loaded/decoded once per sub-block
 *     and reused by all 4 activation rows.
 *
 * Per Q2_K block (256 values) the layout is 16 sub-blocks of 16 values:
 *   scales[16] holds (min<<4 | scale) for each sub-block;
 *   qs[64] holds 2-bit quants, 4 shifts x 2 chunks x 2 halves.
 * Dequant: val = d*scale*q - dmin*min, and the dmin term is folded into a
 * per-sub-block bsums dot product (exactly as vec_dot_q2_K_q8_K does).
 *
 * Integer cores:
 *   __ARM_FEATURE_MATMUL_INT8 -> vmmlaq_s32 (lanes 0+3 = the 16-value dot)
 *   __ARM_FEATURE_DOTPROD     -> vdotq_s32  (two half-vectors summed)
 *   otherwise                 -> vmull_s8 + vpaddlq_s16 (portable NEON)
 * All three produce the same integer result; only the accumulation order of
 * the float scale application differs, so results match vec_dot to ~1 ulp.
 * ============================================================ */
#include "quant.h"
#include "sgemm.h"

#if defined(__ARM_NEON)

/* 16-value signed dot products for the NEON sub-block width used below.
 * Returns the dot of two 16-wide int8 vectors as (lo4 lane sums + hi4). */
static inline int32x4_t q2k_dot16(int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    /* vmmlaq_s32 treats the 16 bytes as a 2x8 matrix. For a pure dot we only
     * need lanes 0 and 3 (diagonal); the caller sums them. Here we return a
     * 4-lane vector where lane0+lane3 is the dot, matching the I8MM contract. */
    return vmmlaq_s32(vdupq_n_s32(0), a, b);
#elif defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(vdupq_n_s32(0), a, b);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(a),  vget_low_s8(b));
    int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
#endif
}

/* Horizontal sum of a q2k_dot16() result, honouring the I8MM lane rule. */
static inline int32_t q2k_hsum16(int32x4_t v) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 3);
#else
    return vaddvq_s32(v);
#endif
}

/* Decode the 16 2-bit sub-block groups of one Q2_K block once, so they can be
 * dotted against several activation rows. g[2*k] is the low half of sub-block
 * k, g[2*k+1] the high half (the pairing vec_dot_q2_K_q8_K uses). */
static inline void q2k_decode(const block_q2_K *x, int8x16_t g[16]) {
    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const uint8_t *q2 = x->qs;
    int gi = 0;
    for (int chunk = 0; chunk < 2; ++chunk) {
        uint8x16_t qa = vld1q_u8(q2);
        uint8x16_t qb = vld1q_u8(q2 + 16);
        q2 += 32;
        for (int sh = 0; sh < 4; ++sh) {
            g[gi++] = vreinterpretq_s8_u8(vandq_u8(qa, m3));
            g[gi++] = vreinterpretq_s8_u8(vandq_u8(qb, m3));
            qa = vshrq_n_u8(qa, 2);
            qb = vshrq_n_u8(qb, 2);
        }
    }
}

/* Dot the decoded groups against one Q8_K row, applying the 16 sub-block scales.
 * The Q8_K row is walked in the same 32-byte steps as q2k_decode(). */
static inline int32_t q2k_dot_decoded(const int8x16_t g[16], const int8_t sc[16],
                                      const block_q8_K *y) {
    const int8_t *q8 = y->qs;
    int32_t isum = 0;
    for (int k = 0; k < 8; ++k) {
        const int8x16_t q8a = vld1q_s8(q8);
        const int8x16_t q8b = vld1q_s8(q8 + 16);
        q8 += 32;
        isum += q2k_hsum16(q2k_dot16(g[2 * k],     q8a)) * (int32_t)sc[2 * k]
              + q2k_hsum16(q2k_dot16(g[2 * k + 1], q8b)) * (int32_t)sc[2 * k + 1];
    }
    return isum;
}

/* Decode one Q2_K block against one Q8_K row. Kept for reference/self-test. */
static inline int32_t q2k_block_dot(const block_q2_K *x, const block_q8_K *y) {
    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);

    const uint8x16_t ms = vld1q_u8(x->scales);
    const uint8x16_t scales = vandq_u8(ms, m4);
    int8_t sc[16];
    vst1q_s8(sc, vreinterpretq_s8_u8(scales));

    const uint8_t *q2 = x->qs;
    const int8_t  *q8 = y->qs;

    /* Scalar accumulation: q2k_dot16() already reduces each 16-value sub-block
     * to a single number, so there is nothing to gain from keeping a vector
     * accumulator (and vaddvq on a broadcast vector would count it 4x). */
    int32_t isum = 0;
    int is = 0;
    for (int chunk = 0; chunk < 2; ++chunk) {
        uint8x16_t qa = vld1q_u8(q2);
        uint8x16_t qb = vld1q_u8(q2 + 16);
        q2 += 32;
        for (int sh = 0; sh < 4; ++sh) {
            const int8x16_t ga = vreinterpretq_s8_u8(vandq_u8(qa, m3));
            const int8x16_t gb = vreinterpretq_s8_u8(vandq_u8(qb, m3));
            qa = vshrq_n_u8(qa, 2);
            qb = vshrq_n_u8(qb, 2);

            const int8x16_t q8a = vld1q_s8(q8);
            const int8x16_t q8b = vld1q_s8(q8 + 16);
            q8 += 32;

            /* sub-block 2*sh uses ga vs q8a; 2*sh+1 uses gb vs q8b */
            const int32_t dot_a = q2k_hsum16(q2k_dot16(ga, q8a));
            const int32_t dot_b = q2k_hsum16(q2k_dot16(gb, q8b));
            isum += dot_a * (int32_t)sc[is] + dot_b * (int32_t)sc[is + 1];
            is += 2;
        }
    }
    return isum;
}

/* dmin term: sum over the 16 sub-blocks of min_s * bsums_s. */
static inline int32_t q2k_bias(const block_q2_K *x, const block_q8_K *y) {
    const uint8x16_t ms = vld1q_u8(x->scales);
    const uint8x16_t mins = vshrq_n_u8(ms, 4);
    const int16x4_t b0 = vld1_s16(y->bsums);
    const int16x4_t b1 = vld1_s16(y->bsums + 4);
    const int16x4_t b2 = vld1_s16(y->bsums + 8);
    const int16x4_t b3 = vld1_s16(y->bsums + 12);
    const int16x8_t ml = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
    const int16x8_t mh = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));
    const int32x4_t mm0 = vmull_s16(vget_low_s16(ml),  b0);
    const int32x4_t mm1 = vmull_s16(vget_high_s16(ml), b1);
    const int32x4_t mm2 = vmull_s16(vget_low_s16(mh),  b2);
    const int32x4_t mm3 = vmull_s16(vget_high_s16(mh), b3);
    return vaddvq_s32(vaddq_s32(vaddq_s32(mm0, mm1), vaddq_s32(mm2, mm3)));
}

/* 4 weight rows x 4 activation rows. A[lda*i + l] / B[ldb*j + l]. */
static void sgemm_q2k_q8k_neon_4x4(const block_q2_K *A, int lda,
                                   const block_q8_K *B, int ldb,
                                   float *C, int ldc,
                                   int m, int n, int k_blocks,
                                   int64_t i0, int64_t i1,
                                   int64_t j0, int64_t j1) {
    for (int64_t ii = i0; ii < i1; ii += 4) {
        int64_t nrows = i1 - ii; if (nrows > 4) nrows = 4;
        for (int64_t jj = j0; jj < j1; jj += 4) {
            int64_t ncols = j1 - jj; if (ncols > 4) ncols = 4;

            float acc[4][4];
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) acc[r][c] = 0.0f;

            for (int64_t l = 0; l < k_blocks; ++l) {
                const block_q8_K *bq[4];
                float bd[4];
                for (int jr = 0; jr < ncols; ++jr) {
                    bq[jr] = &B[ldb * (jj + jr) + l];
                    bd[jr] = bq[jr]->d;
                }
                for (int ir = 0; ir < nrows; ++ir) {
                    const block_q2_K *aq = &A[lda * (ii + ir) + l];
                    const float d    = fp16_to_fp32_lookup(aq->d);
                    const float dmin = fp16_to_fp32_lookup(aq->dmin);

                    /* WEIGHT SIDE: decode this block's 2-bit groups once... */
                    int8x16_t g[16];
                    q2k_decode(aq, g);
                    int8_t sc[16];
                    vst1q_s8(sc, vreinterpretq_s8_u8(
                        vandq_u8(vld1q_u8(aq->scales), vdupq_n_u8(0x0F))));

                    /* ...then reuse them across every activation row. */
                    for (int jr = 0; jr < ncols; ++jr) {
                        const int32_t isum = q2k_dot_decoded(g, sc, bq[jr]);
                        const int32_t bsum = q2k_bias(aq, bq[jr]);
                        acc[jr][ir] += d * bd[jr] * (float)isum
                                     - dmin * bd[jr] * (float)bsum;
                    }
                }
            }
            for (int jr = 0; jr < ncols; ++jr)
                for (int ir = 0; ir < nrows; ++ir)
                    C[ldc * (jj + jr) + (ii + ir)] = acc[jr][ir];
        }
    }
}

int sgemm_q2_k_q8_k_neon(int nrows, int ncols, int k, const void *vx,
                         const void *vy, float *out, size_t bs,
                         int ith, int nth) {
    const block_q2_K *A = (const block_q2_K *)vx;
    const block_q8_K *B = (const block_q8_K *)vy;
    if (nrows < 1 || ncols < 1 || k <= 0 || (k % QK_K) != 0) return 0;
    const int nb = k / QK_K;

    /* Row-block partition across threads. */
    int64_t rows_per = (nrows + nth - 1) / nth;
    int64_t i0 = (int64_t)ith * rows_per;
    int64_t i1 = i0 + rows_per; if (i1 > nrows) i1 = nrows;
    if (i0 >= i1) return 0;

    sgemm_q2k_q8k_neon_4x4(A, nb, B, nb, out, (int)bs,
                           nrows, ncols, nb, i0, i1, 0, ncols);
    return (int)(i1 - i0);
}

#endif /* __ARM_NEON */
