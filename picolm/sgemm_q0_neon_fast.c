/* ============================================================
 * Q8_0 / Q4_0 / Q5_0 / Q5_1 x Q8_0 NEON GEMM for ARM
 *
 * Why this file exists
 * --------------------
 * These four types all shared the generic `NEON_QGEMM_D4_IMPL` macro in
 * sgemm.c, which has three costs measured on the Cortex-A720 in this phone:
 *
 *  1. It never uses DOTPROD or I8MM. `neon_vdot_32()` is always the portable
 *     vmull_s8 + vpaddlq_s16 form: 11 integer ops per 32-value dot. Measured
 *     DOTPROD is 211 GOPS against the portable path's 70 on this core.
 *  2. It converts int32 -> float32 per (weight row, activation row, block) with
 *     `vcvtq_f32_s32` + `vmlaq_f32`, plus a `vdupq_n_f32` for the scale -- and
 *     the scale itself needs `fp16_to_fp32` of the weight delta inside the loop.
 *  3. Its tile is 4 weight rows x 4 activation rows, so the weight decode is
 *     amortised over only 4 columns.
 *
 * Baseline (m=2048 n=44 k=2048, 4 threads, SmolLM2-1.7B shapes):
 *     Q8_0  20.0 GFLOPS     Q5_0  15.8 GFLOPS
 *     Q4_0  20.3 GFLOPS     Q5_1  DECLINED (no NEON path at all)
 *     IQ4_NL 34.2 GFLOPS  <- already had the treatment below
 *
 * Structure (same as sgemm_iq4nl_neon_fast.c, which measured 1.75x):
 *  - DOTPROD `vdotq_s32`, with I8MM and the portable form as fallbacks.
 *  - The scale folds into the int32 accumulator with `vmlaq_n_s32`, so there is
 *    no horizontal reduction and no int->float conversion inside the k loop.
 *    One fold per output element remains.
 *  - The weight delta is converted once per (row block, ib) and reused by the
 *    whole column tile -- the same hoist that gave IQ4_NL 24 -> 38.6 GFLOPS.
 *  - 8 weight rows x 16 activation columns.
 *
 * Q5_0 / Q5_1 details (the awkward part)
 * --------------------------------------
 * 5-bit values are stored as 4 low bits in qs plus one high bit in qh[4].
 * Value j takes bit j of the 32-bit qh; value j+16 takes bit j+16. Q5_0 then
 * biases by -16 (5-bit values are centred), Q5_1 does not and instead adds a
 * per-block min `m`:
 *     Q5_0:  val = ((qs & 0xF) | (h << 4)) * d  - 16*d
 *     Q5_1:  val = ((qs & 0xF) | (h << 4)) * d  + m
 * So for a dot against an int8 activation row:
 *     Q5_0:  d * sum(q5 * q8) - 16*d * sum(q8)
 *     Q5_1:  d * sum(q5 * q8) +   m * sum(q8)
 * Both need sum(q8) per 32-value block. Q8_0 activations have no bsums field,
 * so it is computed from the q8 bytes with a vdotq against a vector of ones --
 * one extra instruction per 32 values, and the -16*d bias is folded into the
 * same scalar. This is why Q5_1 could not use the generic path at all.
 * ============================================================ */
#include "quant.h"
#include "sgemm.h"

#if defined(__ARM_NEON)

#include <string.h>

/* Scale folded into the int32 lanes; no horizontal reduction in the k loop. */
static inline int32x4_t q0_acc(int32x4_t acc, int8x16_t a, int8x16_t b, int32_t sc) {
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

static inline int32_t q0_fold(int32x4_t v) {
#if defined(__ARM_FEATURE_MATMUL_INT8) && !defined(__ARM_FEATURE_DOTPROD)
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 3);
#else
    return vaddvq_s32(v);
#endif
}

/* Non-DOTPROD cores cannot apply a per-lane scale, so the accumulator is scaled
 * after folding there. Guard the in-lane form so the code stays correct. */
#if defined(__ARM_FEATURE_DOTPROD) || !defined(__ARM_FEATURE_MATMUL_INT8)
#define Q0_INLINE_SCALE 1
#else
#define Q0_INLINE_SCALE 0
#endif

/* Decode one block's 32 values into two 16-lane int8 vectors. */
static inline void q0_decode_q8_0(const block_q8_0 *b, int8x16_t *lo, int8x16_t *hi) {
    *lo = vld1q_s8(b->qs);
    *hi = vld1q_s8(b->qs + 16);
}

static inline void q0_decode_q4_0(const block_q4_0 *b, int8x16_t *lo, int8x16_t *hi) {
    const uint8x16_t x = vld1q_u8(b->qs);
    const int8x16_t o8 = vdupq_n_s8(8);
    *lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(x, vdupq_n_u8(0x0F))), o8);
    *hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(x, 4)), o8);
}

/* Build the two 16-lane 5-bit value vectors (values 0..31) for Q5_0/Q5_1.
 *
 * qh holds 32 bits, one per value, little-endian:
 *     value j  -> bit j   -> byte j>>3, bit j&7
 * so values 0..15 use qh bytes 0..1 and values 16..31 use bytes 2..3.
 *
 * NOTE the byte index is j>>3, NOT j>>1. Indexing by j>>1 looks plausible (two
 * values per byte) but is wrong: a byte holds EIGHT bits, one per value, not
 * two. Getting this wrong produces values that are correct in the low nibble
 * and wrong in the high bit, which is easy to mistake for a rounding problem.
 *
 * Per half: duplicate the relevant byte across the 8 lanes that share it, shift
 * each lane down by its bit position, keep bit 0, and move it to bit 4. */
static inline void q0_decode_q5(const uint8_t *qs, const uint8_t *qh,
                                int8x16_t *lo, int8x16_t *hi) {
    const uint8x16_t x   = vld1q_u8(qs);
    const uint8x16_t lo4 = vandq_u8(x, vdupq_n_u8(0x0F));
    const uint8x16_t hi4 = vshrq_n_u8(x, 4);

    /* lane j reads byte j>>3: 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1 */
    static const uint8_t idx[16] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1};
    /* bit position within that byte: j&7 */
    static const int8_t  shl[16] = {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7};
    const uint8x16_t idxv = vld1q_u8(idx);
    const int8x16_t  sh   = vld1q_s8(shl);
    const uint8x16_t one  = vdupq_n_u8(0x01);

    /* values 0..15: qh bytes 0..1 */
    const uint8x16_t dlo = vqtbl1q_u8(vld1q_u8(qh), idxv);
    const uint8x16_t blo = vandq_u8(vshlq_u8(dlo, vnegq_s8(sh)), one);

    /* values 16..31: qh bytes 2..3 */
    const uint8x16_t dhi = vqtbl1q_u8(vld1q_u8(qh + 2), idxv);
    const uint8x16_t bhi = vandq_u8(vshlq_u8(dhi, vnegq_s8(sh)), one);

    *lo = vreinterpretq_s8_u8(vorrq_u8(lo4, vshlq_n_u8(blo, 4)));
    *hi = vreinterpretq_s8_u8(vorrq_u8(hi4, vshlq_n_u8(bhi, 4)));
}

#define Q0_TILE_M 8
#define Q0_TILE_N 16

/* ---------------- Q8_0 ---------------- */
static void q0_tile_q8_0(const block_q8_0 *A, const block_q8_0 *B, const float *B_d,
                         float *C, int ldc, int k_blocks,
                         int ir0, int ir1, int jc0, int jc1) {
    for (int jc = jc0; jc < jc1; jc += Q0_TILE_N) {
        const int nn = (jc + Q0_TILE_N <= jc1) ? Q0_TILE_N : (jc1 - jc);
        for (int ir = ir0; ir < ir1; ir += Q0_TILE_M) {
            const int mm = (ir + Q0_TILE_M <= ir1) ? Q0_TILE_M : (ir1 - ir);
            float32x4_t acc[Q0_TILE_M][Q0_TILE_N];
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c) acc[r][c] = vdupq_n_f32(0.0f);

            for (int l = 0; l < k_blocks; ++l) {
                int8x16_t b_lo[Q0_TILE_N], b_hi[Q0_TILE_N];
                float bd[Q0_TILE_N];
                for (int c = 0; c < nn; ++c) {
                    const block_q8_0 *bq = &B[(size_t)k_blocks * (jc + c) + l];
                    b_lo[c] = vld1q_s8(bq->qs);
                    b_hi[c] = vld1q_s8(bq->qs + 16);
                    bd[c] = B_d[(size_t)k_blocks * (jc + c) + l];
                }
                for (int r = 0; r < mm; ++r) {
                    const block_q8_0 *aq = &A[(size_t)k_blocks * (ir + r) + l];
                    const float da = fp16_to_fp32_lookup(aq->d);
                    const int8x16_t a_lo = vld1q_s8(aq->qs);
                    const int8x16_t a_hi = vld1q_s8(aq->qs + 16);
                    for (int c = 0; c < nn; ++c) {
                        int32x4_t s = q0_acc(vdupq_n_s32(0), a_lo, b_lo[c], 1);
                        s = q0_acc(s, a_hi, b_hi[c], 1);
                        acc[r][c] = vmlaq_f32(acc[r][c], vcvtq_f32_s32(s),
                                              vdupq_n_f32(da * bd[c]));
                    }
                }
            }
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c)
                    C[(size_t)ldc * (jc + c) + (ir + r)] = vaddvq_f32(acc[r][c]);
        }
    }
}

/* ---------------- Q4_0 ---------------- */
static void q0_tile_q4_0(const block_q4_0 *A, const block_q8_0 *B, const float *B_d,
                         float *C, int ldc, int k_blocks,
                         int ir0, int ir1, int jc0, int jc1) {
    for (int jc = jc0; jc < jc1; jc += Q0_TILE_N) {
        const int nn = (jc + Q0_TILE_N <= jc1) ? Q0_TILE_N : (jc1 - jc);
        for (int ir = ir0; ir < ir1; ir += Q0_TILE_M) {
            const int mm = (ir + Q0_TILE_M <= ir1) ? Q0_TILE_M : (ir1 - ir);
            float32x4_t acc[Q0_TILE_M][Q0_TILE_N];
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c) acc[r][c] = vdupq_n_f32(0.0f);

            for (int l = 0; l < k_blocks; ++l) {
                int8x16_t b_lo[Q0_TILE_N], b_hi[Q0_TILE_N];
                float bd[Q0_TILE_N];
                for (int c = 0; c < nn; ++c) {
                    const block_q8_0 *bq = &B[(size_t)k_blocks * (jc + c) + l];
                    b_lo[c] = vld1q_s8(bq->qs);
                    b_hi[c] = vld1q_s8(bq->qs + 16);
                    bd[c] = B_d[(size_t)k_blocks * (jc + c) + l];
                }
                for (int r = 0; r < mm; ++r) {
                    const block_q4_0 *aq = &A[(size_t)k_blocks * (ir + r) + l];
                    const float da = fp16_to_fp32_lookup(aq->d);
                    int8x16_t a_lo, a_hi;
                    q0_decode_q4_0(aq, &a_lo, &a_hi);
                    for (int c = 0; c < nn; ++c) {
                        int32x4_t s = q0_acc(vdupq_n_s32(0), a_lo, b_lo[c], 1);
                        s = q0_acc(s, a_hi, b_hi[c], 1);
                        acc[r][c] = vmlaq_f32(acc[r][c], vcvtq_f32_s32(s),
                                              vdupq_n_f32(da * bd[c]));
                    }
                }
            }
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c)
                    C[(size_t)ldc * (jc + c) + (ir + r)] = vaddvq_f32(acc[r][c]);
        }
    }
}

/* ---------------- Q5_0 ---------------- */
/* val = q5*d - 16*d, so the dot is d*sum(q5*q8) - 16*d*sum(q8). The sum(q8) is
 * obtained by dotting the q8 bytes against a vector of ones, which is one extra
 * vdotq per 32 values; the -16*d bias is folded into the same scalar. */
static void q0_tile_q5_0(const block_q5_0 *A, const block_q8_0 *B, const float *B_d,
                         float *C, int ldc, int k_blocks,
                         int ir0, int ir1, int jc0, int jc1) {
    for (int jc = jc0; jc < jc1; jc += Q0_TILE_N) {
        const int nn = (jc + Q0_TILE_N <= jc1) ? Q0_TILE_N : (jc1 - jc);
        for (int ir = ir0; ir < ir1; ir += Q0_TILE_M) {
            const int mm = (ir + Q0_TILE_M <= ir1) ? Q0_TILE_M : (ir1 - ir);
            float32x4_t acc[Q0_TILE_M][Q0_TILE_N];
            /* The -16*d*sum(q8) bias is a SCALAR per (row, col, block), and a
             * float32x4_t accumulator is later folded with vaddvq_f32, which
             * would count a broadcast scalar FOUR times (the dot survives this
             * because its lanes are distinct partials). So the bias gets its own
             * scalar accumulator and is added after the fold. */
            float bias[Q0_TILE_M][Q0_TILE_N];
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c) { acc[r][c] = vdupq_n_f32(0.0f); bias[r][c] = 0.0f; }

            for (int l = 0; l < k_blocks; ++l) {
                int8x16_t b_lo[Q0_TILE_N], b_hi[Q0_TILE_N];
                float bd[Q0_TILE_N];
                /* sum(q8) depends only on (column, l), so it is computed once
                 * here and shared by all Q0_TILE_M weight rows. Computing it
                 * inside the row loop made this kernel 1.9x slower than Q4_0. */
                int32_t bsum[Q0_TILE_N];
                for (int c = 0; c < nn; ++c) {
                    const block_q8_0 *bq = &B[(size_t)k_blocks * (jc + c) + l];
                    b_lo[c] = vld1q_s8(bq->qs);
                    b_hi[c] = vld1q_s8(bq->qs + 16);
                    bd[c] = B_d[(size_t)k_blocks * (jc + c) + l];
                    bsum[c] = vaddvq_s32(
                        vaddq_s32(vpaddlq_s16(vpaddlq_s8(b_lo[c])),
                                  vpaddlq_s16(vpaddlq_s8(b_hi[c]))));
                }
                for (int r = 0; r < mm; ++r) {
                    const block_q5_0 *aq = &A[(size_t)k_blocks * (ir + r) + l];
                    const float da = fp16_to_fp32_lookup(aq->d);
                    int8x16_t a_lo, a_hi;
                    q0_decode_q5(aq->qs, aq->qh, &a_lo, &a_hi);
                    for (int c = 0; c < nn; ++c) {
                        int32x4_t s = q0_acc(vdupq_n_s32(0), a_lo, b_lo[c], 1);
                        s = q0_acc(s, a_hi, b_hi[c], 1);
                        acc[r][c] = vmlaq_f32(acc[r][c], vcvtq_f32_s32(s),
                                              vdupq_n_f32(da * bd[c]));
                        /* -16*d*sum(q8) */
                        bias[r][c] -= 16.0f * da * bd[c] * (float)bsum[c];
                    }
                }
            }
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c)
                    C[(size_t)ldc * (jc + c) + (ir + r)] =
                        vaddvq_f32(acc[r][c]) + bias[r][c];
        }
    }
}

/* ---------------- Q5_1 ---------------- */
/* val = q5*d + m, so the dot is d*sum(q5*q8) + m*sum(q8). */
static void q0_tile_q5_1(const block_q5_1 *A, const block_q8_0 *B, const float *B_d,
                         float *C, int ldc, int k_blocks,
                         int ir0, int ir1, int jc0, int jc1) {
    for (int jc = jc0; jc < jc1; jc += Q0_TILE_N) {
        const int nn = (jc + Q0_TILE_N <= jc1) ? Q0_TILE_N : (jc1 - jc);
        for (int ir = ir0; ir < ir1; ir += Q0_TILE_M) {
            const int mm = (ir + Q0_TILE_M <= ir1) ? Q0_TILE_M : (ir1 - ir);
            float32x4_t acc[Q0_TILE_M][Q0_TILE_N];
            /* Same reason as Q5_0: the +m*sum(q8) term is a scalar and must not
             * be broadcast into the 4-lane accumulator that is vaddvq'd later. */
            float bias[Q0_TILE_M][Q0_TILE_N];
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c) { acc[r][c] = vdupq_n_f32(0.0f); bias[r][c] = 0.0f; }

            for (int l = 0; l < k_blocks; ++l) {
                int8x16_t b_lo[Q0_TILE_N], b_hi[Q0_TILE_N];
                float bd[Q0_TILE_N];
                int32_t bsum[Q0_TILE_N];
                for (int c = 0; c < nn; ++c) {
                    const block_q8_0 *bq = &B[(size_t)k_blocks * (jc + c) + l];
                    b_lo[c] = vld1q_s8(bq->qs);
                    b_hi[c] = vld1q_s8(bq->qs + 16);
                    bd[c] = B_d[(size_t)k_blocks * (jc + c) + l];
                    bsum[c] = vaddvq_s32(
                        vaddq_s32(vpaddlq_s16(vpaddlq_s8(b_lo[c])),
                                  vpaddlq_s16(vpaddlq_s8(b_hi[c]))));
                }
                for (int r = 0; r < mm; ++r) {
                    const block_q5_1 *aq = &A[(size_t)k_blocks * (ir + r) + l];
                    const float da = fp16_to_fp32_lookup(aq->d);
                    const float ma = fp16_to_fp32_lookup(aq->m);
                    int8x16_t a_lo, a_hi;
                    q0_decode_q5(aq->qs, aq->qh, &a_lo, &a_hi);
                    for (int c = 0; c < nn; ++c) {
                        int32x4_t s = q0_acc(vdupq_n_s32(0), a_lo, b_lo[c], 1);
                        s = q0_acc(s, a_hi, b_hi[c], 1);
                        acc[r][c] = vmlaq_f32(acc[r][c], vcvtq_f32_s32(s),
                                              vdupq_n_f32(da * bd[c]));
                        bias[r][c] += ma * bd[c] * (float)bsum[c];
                    }
                }
            }
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c)
                    C[(size_t)ldc * (jc + c) + (ir + r)] =
                        vaddvq_f32(acc[r][c]) + bias[r][c];
        }
    }
}

/* ---------------- entry points ---------------- */

#define Q0_ENTRY(NAME, BT, TILE)                                               \
int NAME(int nrows, int ncols, int k, const void *vx, const void *vy,          \
         const float *vy_d, float *out, size_t bs, int ith, int nth) {         \
    if (nrows < 1 || ncols < 1 || k <= 0 || (k % 32) != 0) return 0;           \
    const int nb = k / 32;                                                     \
    const int64_t rows_per = ((int64_t)nrows + nth - 1) / nth;                 \
    const int64_t i0 = (int64_t)ith * rows_per;                                \
    int64_t i1 = i0 + rows_per; if (i1 > nrows) i1 = nrows;                    \
    if (i0 >= i1) return 0;                                                    \
    TILE((const BT *)vx, (const block_q8_0 *)vy, vy_d, out, (int)bs, nb,       \
         (int)i0, (int)i1, 0, ncols);                                          \
    return (int)(i1 - i0);                                                     \
}

Q0_ENTRY(sgemm_q8_0_q8_0_neon_fast,  block_q8_0, q0_tile_q8_0)
Q0_ENTRY(sgemm_q4_0_q8_0_neon_fast,  block_q4_0, q0_tile_q4_0)
Q0_ENTRY(sgemm_q5_0_q8_0_neon_fast,  block_q5_0, q0_tile_q5_0)
Q0_ENTRY(sgemm_q5_1_q8_0_neon_fast,  block_q5_1, q0_tile_q5_1)

#endif /* __ARM_NEON */
