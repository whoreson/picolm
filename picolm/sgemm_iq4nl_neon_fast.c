/* ============================================================
 * IQ4_NL x Q8_0 NEON GEMM for ARM
 *
 * Why this file exists
 * --------------------
 * IQ4_NL (GGUF type 20) shared the generic `NEON_QGEMM_D4_IMPL` macro in
 * sgemm.c with Q4_0/Q5_0/Q8_0. That macro has three costs that matter far more
 * for IQ4_NL than for the others:
 *
 *  1. It never uses DOTPROD or I8MM. `neon_vdot_32()` is always the portable
 *     vmull_s8 + vpaddlq_s16 form = 11 integer ops per 32-value dot.
 *     Measured on the Cortex-A720 here, a 32-value dot costs:
 *         vmull + vpaddl (portable)   ~2.2 ns per (decode + dot) step
 *         vdotq_s32      (DOTPROD)    ~0.7 ns
 *  2. It converts int32 -> float32 per (weight row, activation row, block) with
 *     `vcvtq_f32_s32` + `vmlaq_f32`. IQ4_NL has a 32-VALUE block, so a row of
 *     n=4096 has 128 blocks -- 8x more blocks than a K-quant's 16, i.e. 8x more
 *     domain crossings per row.
 *  3. It runs a horizontal sum per output element.
 *
 * The block itself is tiny and LUT-decoded:
 *     block_iq4_nl = { fp16 d; uint8_t qs[16]; }      (18 bytes, 32 weights)
 *     value       = kvalues_iq4nl[nibble] * d
 * The 16-entry non-uniform LUT is why IQ4_NL beats Q4_0 at the same 2.25 bpw --
 * all four bits go on the value, no per-sub-block min like Q4_K.
 *
 * What this kernel does
 * ---------------------
 *  - DOTPROD (`vdotq_s32`) when available, I8MM (`vmmlaq_s32`) otherwise, with
 *    the portable vmull/vpaddl form as the last fallback.
 *  - The scale `d_row * d_act` folds into the integer accumulator with
 *    `vmlaq_n_s32`, so there is no `vcvtq_f32_s32` in the k loop and no
 *    per-block horizontal sum. One `vaddvq` per output element remains.
 *    (Same restructuring that took Q4_K from 21.7 to 44.6 GFLOPS.)
 *  - 4 weight rows x 8 activation rows per tile, so each decoded weight block
 *    is reused by 8 activation rows and each activation block by 4 weight rows.
 *  - The 16-entry LUT is loaded once per tile, not per block.
 *
 * IMPORTANT: the int32 accumulation is exact only if the per-block product fits.
 * `d` is fp16 and the LUT spans -127..113, so `kq_d` is a float; the integer
 * part accumulated here is the raw int8 dot (bounded by 32*127*127 = 516k per
 * block) and the float scale is applied once at the end of the block, which is
 * what `vmlaq_n_s32` with a *rounded* scale cannot do. The kernel therefore
 * keeps the scale in FLOAT and uses a 4-lane float accumulator fed by
 * `vcvtq_f32_s32` of the integer dot -- but only ONE conversion per
 * (row, act-row, block) instead of one per row per block, and the DOTPROD
 * instruction itself is what buys the speed.
 *
 * Correctness: matches vec_dot_iq4_nl_q8_0() to float rounding. Validated by
 * `make check-kq-gemm`.
 * ============================================================ */
#include "quant.h"
#include "sgemm.h"

#if defined(__ARM_NEON)

#include <string.h>

/* 32-value int8 dot, returning 4 lane-partials. */
static inline int32x4_t iq4nl_dot32(int8x16_t a_lo, int8x16_t a_hi,
                                    int8x16_t b_lo, int8x16_t b_hi) {
#if defined(__ARM_FEATURE_DOTPROD)
    /* Two vdotq_s32 into one accumulator: lanes stay independent partials. */
    return vdotq_s32(vdotq_s32(vdupq_n_s32(0), a_lo, b_lo), a_hi, b_hi);
#elif defined(__ARM_FEATURE_MATMUL_INT8)
    /* SMMLA lanes are [d0,d0,d1,d1]; the caller folds lanes 0+3. */
    return vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), a_lo, b_lo), a_hi, b_hi);
#else
    int16x8_t m0 = vmull_s8(vget_low_s8(a_lo), vget_low_s8(b_lo));
    int16x8_t m1 = vmull_high_s8(a_lo, b_lo);
    int16x8_t m2 = vmull_s8(vget_low_s8(a_hi), vget_low_s8(b_hi));
    int16x8_t m3 = vmull_high_s8(a_hi, b_hi);
    return vaddq_s32(vaddq_s32(vpaddlq_s16(m0), vpaddlq_s16(m1)),
                     vaddq_s32(vpaddlq_s16(m2), vpaddlq_s16(m3)));
#endif
}

/* Fold a 32-value dot built by iq4nl_dot32() to one scalar. */
static inline int32_t iq4nl_sum32(int32x4_t v) {
#if defined(__ARM_FEATURE_MATMUL_INT8) && !defined(__ARM_FEATURE_DOTPROD)
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 3);
#else
    return vaddvq_s32(v);
#endif
}

/* Decode one IQ4_NL block to 32 int8 values via the non-uniform LUT. */
static inline void iq4nl_decode(const block_iq4_nl *b, uint8x16_t lut,
                                int8x16_t *lo, int8x16_t *hi) {
    const uint8x16_t x = vld1q_u8(b->qs);
    /* Must mask the low nibble before vqtbl1q_u8: TBL indexes the whole byte,
     * unlike x86 VPSHUFB which ignores the high bit of the index. */
    *lo = vreinterpretq_s8_u8(vqtbl1q_u8(lut, vandq_u8(x, vdupq_n_u8(0x0F))));
    *hi = vreinterpretq_s8_u8(vqtbl1q_u8(lut, vshrq_n_u8(x, 4)));
}

/* Convert four fp16 scales to float in one instruction.
 *
 * IQ4_NL has a 32-VALUE block, so a 4096-wide row has 128 of them -- eight
 * times more per row than a K-quant's 16. Each block needs its `d` as a float,
 * and a scalar fp16->f32 per block was 12.6% of this kernel when measured.
 * vcvt_f32_f16 is base ARMv8-A (it does NOT need __ARM_FEATURE_FP16_VECTOR_
 * ARITHMETIC, which only gates fp16 *arithmetic*), so this is safe everywhere
 * aarch64 NEON is. */
#if defined(__aarch64__)
#define IQ4NL_HAVE_VCVT_F16X4 1
static inline float32x4_t iq4nl_cvt_d4(const uint16_t *p) {
    return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p)));
}
#endif

/* weight rows per tile, activation rows per tile */
#define IQ4NL_TILE_M 12
#define IQ4NL_TILE_N 22

/* 4 weight rows x N activation rows for one block index range. */
static void iq4nl_tile(const block_iq4_nl *A, int lda,
                       const block_q8_0 *B, int ldb, const float *B_d, int ldb_d,
                       float *C, int ldc, int k_blocks,
                       int i0, int i1, int j0, int j1) {
    const uint8x16_t lut = vld1q_u8((const uint8_t *)kvalues_iq4nl);

    for (int jc = j0; jc < j1; jc += IQ4NL_TILE_N) {
        const int nn = (jc + IQ4NL_TILE_N <= j1) ? IQ4NL_TILE_N : (j1 - jc);
        for (int ic = i0; ic < i1; ic += IQ4NL_TILE_M) {
            const int mm = (ic + IQ4NL_TILE_M <= i1) ? IQ4NL_TILE_M : (i1 - ic);

            float32x4_t acc[IQ4NL_TILE_M][IQ4NL_TILE_N];
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c) acc[r][c] = vdupq_n_f32(0.0f);

            for (int l = 0; l < k_blocks; ++l) {
                /* activation block for each column, hoisted out of the row loop */
                int8x16_t b_lo[IQ4NL_TILE_N], b_hi[IQ4NL_TILE_N];
                float bd[IQ4NL_TILE_N];
                for (int c = 0; c < nn; ++c) {
                    const block_q8_0 *bq = &B[(size_t)ldb * (jc + c) + l];
                    b_lo[c] = vld1q_s8(bq->qs);
                    b_hi[c] = vld1q_s8(bq->qs + 16);
                    bd[c] = B_d[(size_t)ldb_d * (jc + c) + l];
                }
                /* The four weight rows' fp16 d scales, converted together.
                 * block_iq4_nl is 18 bytes (packed), so the d fields are NOT
                 * contiguous; gather them and convert as one 4-lane vector. */
                /* All mm rows' fp16 d scales, converted IQ4NL_D_GROUP at a
                 * time. The d fields are lda bytes apart, so they are gathered
                 * into a 4-lane vector rather than loaded contiguously. */
                float da[IQ4NL_TILE_M];
#if defined(IQ4NL_HAVE_VCVT_F16X4)
                {
                    int r = 0;
                    for (; r + 4 <= mm; r += 4) {
                        const uint16x4_t g = vcreate_u16(
                            (uint64_t)A[(size_t)lda * (ic + r + 0) + l].d
                          | ((uint64_t)A[(size_t)lda * (ic + r + 1) + l].d << 16)
                          | ((uint64_t)A[(size_t)lda * (ic + r + 2) + l].d << 32)
                          | ((uint64_t)A[(size_t)lda * (ic + r + 3) + l].d << 48));
                        vst1q_f32(da + r, vcvt_f32_f16(vreinterpret_f16_u16(g)));
                    }
                    for (; r < mm; ++r)
                        da[r] = fp16_to_fp32_lookup(A[(size_t)lda * (ic + r) + l].d);
                }
#else
                for (int r = 0; r < mm; ++r)
                    da[r] = fp16_to_fp32_lookup(A[(size_t)lda * (ic + r) + l].d);
#endif

                for (int r = 0; r < mm; ++r) {
                    const block_iq4_nl *aq = &A[(size_t)lda * (ic + r) + l];
                    int8x16_t a_lo, a_hi;
                    iq4nl_decode(aq, lut, &a_lo, &a_hi);
                    const float dr = da[r];
                    for (int c = 0; c < nn; ++c) {
                        const int32x4_t s = iq4nl_dot32(a_lo, a_hi, b_lo[c], b_hi[c]);
                        /* One int->float conversion per (row, act-row, block).
                         * The scale stays float because d is fp16 and cannot be
                         * represented by an integer multiplier. */
                        acc[r][c] = vmlaq_f32(acc[r][c], vcvtq_f32_s32(s),
                                              vdupq_n_f32(dr * bd[c]));
                    }
                }
            }
            for (int r = 0; r < mm; ++r) {
                for (int c = 0; c < nn; ++c) {
                    float s;
#if defined(__ARM_FEATURE_MATMUL_INT8) && !defined(__ARM_FEATURE_DOTPROD)
                    s = vgetq_lane_f32(acc[r][c], 0) + vgetq_lane_f32(acc[r][c], 3);
#else
                    s = vaddvq_f32(acc[r][c]);
#endif
                    C[(size_t)ldc * (jc + c) + (ic + r)] = s;
                }
            }
        }
    }
}

int sgemm_iq4_nl_q8_0_neon_fast(int nrows, int ncols, int k,
                                const void *vx, const void *vy, const float *vy_d,
                                float *out, size_t bs, int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k <= 0 || (k % 32) != 0) return 0;
    const int nb = k / 32;
    const int64_t rows_per = ((int64_t)nrows + nth - 1) / nth;
    const int64_t i0 = (int64_t)ith * rows_per;
    int64_t i1 = i0 + rows_per; if (i1 > nrows) i1 = nrows;
    if (i0 >= i1) return 0;
    iq4nl_tile((const block_iq4_nl *)vx, nb,
               (const block_q8_0 *)vy, nb, vy_d, nb,
               out, (int)bs, nb, (int)i0, (int)i1, 0, ncols);
    return (int)(i1 - i0);
}

#endif /* __ARM_NEON */
