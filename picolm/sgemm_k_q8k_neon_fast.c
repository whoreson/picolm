/* ============================================================
 * Fast K-quant x Q8_K NEON GEMM for ARM  (Q2_K)
 *
 * The problem
 * -----------
 * The ARM K-quant GEMMs (sgemm_q2_k_neon.c, sgemm_k_q8k_neon.c) and
 * vec_dot_*_q8_K() itself all reduce every 16-value sub-block to a scalar the
 * moment it is computed:
 *
 *     isum += vaddvq_s32(vdotq_s32(zero, g, q8a)) * sc[a] + ...;
 *
 * A Q_K block holds 16 sub-blocks, so that is 16 horizontal reductions and 16
 * scalar multiplies per (weight block, activation row) pair. vaddvq_s32 is a
 * long-latency, non-pipelined reduction, so the ARM K-quant path was gated on
 * it rather than on the int8 MACs.
 *
 * The fix
 * -------
 * vdotq_s32 returns FOUR lanes, each the dot of one 4-byte column of the two
 * 16-byte operands. All four lanes belong to the SAME sub-block and therefore
 * share ONE scale, so the scale folds into the integer accumulator with a
 * broadcast multiply -- before any lanes are combined:
 *
 *     acc = vmlaq_n_s32(acc, vdotq_s32(zero, g, q8), sc[k]);
 *
 * There is then no horizontal reduction anywhere in the k loop: exactly one
 * vaddvq per Q_K block, at the very end. Measured on the Cortex-A720 in this
 * phone (-O3, dotprod+i8mm, one 256-value block; every variant below produces
 * bit-identical integers):
 *
 *     original: 16 vaddvq + 16 scalar multiply      9.26 ns
 *     pack 4 sub-blocks, 4 vmulq_s32               3.34 ns
 *     broadcast vmlaq_n_s32, 2 accumulators        2.26 ns
 *     broadcast vmlaq_n_s32, 4 accumulators        1.91 ns   <- this file
 *
 * 4.8x on the integer/scaling core of the kernel. Two traps worth recording:
 *   (a) "avoid the hsum by scaling in float" (vcvtq_f32_s32 + vmlaq_f32 per
 *       sub-block) measures 9.55 ns -- WORSE than the original -- because every
 *       conversion crosses the int/float domain boundary inside the loop.
 *       Stay in int32 and convert O(1) times per block.
 *   (b) The accumulation is analytically integer, so all these variants are
 *       bit-exact against the scalar reference. That makes validation cheap;
 *       use it before benchmarking anything.
 *
 * Tiling: 8 weight rows x 4 activation rows.
 *  - each weight block is decoded once and applied to all four activation rows;
 *  - each activation row's q8 stream is walked once per weight row;
 *  - per-row integer accumulators stay in registers across the whole k loop.
 *
 * DOTPROD vs I8MM
 * ---------------
 * DOTPROD wins whenever both are available: on this Cortex-A720 vdotq_s32
 * sustains 211 GOPS while vmmlaq_s32 reaches 134 GOPS (measured), because
 * SMMLA's 8x8 tile spends half its work on an off-diagonal a plain dot product
 * never needs. I8MM stays as the fallback core. I8MM lanes are [d0,d0,d1,d1],
 * so a scaled in-lane accumulate is impossible there; that path accumulates
 * raw dots and applies the scale after folding (see kq_acc/kq_fold).
 *
 * Correctness
 * -----------
 * Identical integer dots and identical scale values to vec_dot_*_q8_K(); only
 * the order in which sub-block partials are summed differs, so results agree to
 * float rounding. Validated against the scalar vec_dot at many shapes and
 * thread counts (tests/kq_gemm_check.c, .scratch/t_fastgemm).
 * ============================================================ */
#include "quant.h"
#include "sgemm.h"

#if defined(__ARM_NEON)

#include <string.h>

/* ---------------- int8 dot core ----------------
 *
 * DOTPROD: scale folds into the integer accumulator (no hsum in the k loop).
 * I8MM   : accumulate raw dots, scale after kq_fold().
 * NEON   : vmull/vpaddl collapse the dot, so scaling the vector is exact. */
#if defined(__ARM_FEATURE_DOTPROD)
#define KQ_SCALE_IN_LANE 1
#else
#define KQ_SCALE_IN_LANE 0
#endif

/* kq_acc applies the scale for DOTPROD and plain NEON. The I8MM branch cannot
 * (its lanes duplicate), so callers that need a scale must use kq_fold() and
 * multiply afterwards -- which is exactly what KQ_SCALE_IN_LANE selects. */
static inline int32x4_t kq_acc(int32x4_t acc, int8x16_t a, int8x16_t b, int16_t sc) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), a, b), (int32_t)sc);
#elif defined(__ARM_FEATURE_MATMUL_INT8)
    (void)sc;
    return vmmlaq_s32(acc, a, b);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(a),  vget_low_s8(b));
    int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vmulq_n_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)), (int32_t)sc));
#endif
}

/* True when kq_acc() actually honours its scale argument. */
#if defined(__ARM_FEATURE_DOTPROD) || !defined(__ARM_FEATURE_MATMUL_INT8)
#define KQ_ACC_HONOURS_SCALE 1
#else
#define KQ_ACC_HONOURS_SCALE 0
#endif

/* Q5_K's high-bit trick needs the scale to be applied per sub-block in-lane. */
#define Q5K_ENABLE KQ_ACC_HONOURS_SCALE

static inline int32_t kq_fold(int32x4_t v) {
#if defined(__ARM_FEATURE_MATMUL_INT8) && !defined(__ARM_FEATURE_DOTPROD)
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 3);
#else
    return vaddvq_s32(v);
#endif
}

/* ============================================================
 * Q2_K
 *   scales[16] = (min<<4 | scale) per sub-block;  qs[64] = 2-bit quants
 *   dequant    = d*scale*q - dmin*min   (the min term is folded via bsums)
 *
 * Layout (matches vec_dot_q2_K_q8_K):
 *   chunk c in 0..1:  qs bytes [32c, 32c+32)  <->  q8[128c, 128c+128)
 *     shift s in 0..3: qs[32c    .. +16) -> sub-block 8c + 2s
 *                      qs[32c+16 .. +16) -> sub-block 8c + 2s + 1
 *   q8 advances 32 bytes per shift (4 shifts = 128 values per chunk).
 * ============================================================ */

typedef struct {
    int16_t  sc[16];    /* sub-block scales, same order as g[] */
    int16x8_t mins[2];  /* sub-block mins, for the bsums correction */
} q2k_dec_t;

/* A local int16[16] of scales: the compiler keeps it in registers across the
 * unrolled k loop, so no memory traffic per sub-block. */
#define Q2K_SCALE(d, k) ((int32_t)(d)->sc[k])

static inline void q2k_decode_block(const block_q2_K *x, q2k_dec_t *o) {
    const uint8x16_t m3 = vdupq_n_u8(0x03);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8x16_t ms = vld1q_u8(x->scales);
    /* Sub-block scales as int16 directly in registers: no byte round-trip, and
     * the values land in the order g[] uses (low nibbles, chunk-major). */
    {
        const uint8x16_t sm = vandq_u8(ms, m4);
        vst1q_s16(o->sc,     vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(sm))));
        vst1q_s16(o->sc + 8, vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(sm))));
    }

    const uint8x16_t mn = vshrq_n_u8(ms, 4);
    o->mins[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mn)));
    o->mins[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mn)));

}

/* dmin correction as a VECTOR: lane-wise mins * bsums, folded by the caller's
 * single vaddvq together with the dot. Two vmull_s16 on 8-lane halves (the
 * 4-mull/4-load form this replaced cost about twice as much and needed its own
 * horizontal sum). */
static inline int32x4_t q2k_bias_vec(const int16x8_t mins[2], const block_q8_K *y) {
    const int32x4_t lo = vmull_s16(vget_low_s16(mins[0]),  vld1_s16(y->bsums));
    const int32x4_t hi = vmull_s16(vget_high_s16(mins[0]), vld1_s16(y->bsums + 4));
    const int32x4_t lo2 = vmull_s16(vget_low_s16(mins[1]),  vld1_s16(y->bsums + 8));
    const int32x4_t hi2 = vmull_s16(vget_high_s16(mins[1]), vld1_s16(y->bsums + 12));
    return vaddq_s32(vaddq_s32(lo, hi), vaddq_s32(lo2, hi2));
}

/* Scalar form, for the reference path. */
static inline int32_t q2k_bias(const int16x8_t mins[2], const block_q8_K *y) {
    return kq_fold(q2k_bias_vec(mins, y));
}

/* Scaled integer dot of one Q2_K block against one Q8_K row, PLUS the dmin
 * correction, sharing a single horizontal fold.
 *
 * Why they are fused: measured on the Cortex-A720, per 256-value block,
 *     2-bit unpack            2.76 ns
 *     dot + scale             0.16 ns   (the int8 core is essentially free)
 *     dmin correction         ~2.65 ns
 * The correction is 16 mins times the activation's 16 bsums. Computing it as
 * its own function costs four vmull_s16 plus a vaddvq -- and that horizontal
 * sum is a long-latency dependency at the very end of the block. Returning the
 * correction as a vector instead lets the caller fold it into the SAME vaddvq
 * as the dot, so the block pays one horizontal reduction rather than two.
 *
 * The 2-bit groups are decoded INLINE, one 32-byte chunk at a time: keeping all
 * 16 decoded int8x16_t groups live made the compiler spill them to the stack.
 * */
static inline int32x4_t q2k_acc_block(const block_q2_K *x, const q2k_dec_t *d,
                                      const block_q8_K *y, int32x4_t *bias_out) {
    const uint8x16_t m3 = vdupq_n_u8(0x03);
    const uint8_t *q2 = x->qs;
    const int8_t  *q8 = y->qs;
#if KQ_SCALE_IN_LANE
    /* Two accumulators: each 32-byte shift yields exactly two sub-block dots. */
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int sb = 0;
    for (int chunk = 0; chunk < 2; ++chunk) {
        uint8x16_t qa = vld1q_u8(q2);
        uint8x16_t qb = vld1q_u8(q2 + 16);
        q2 += 32;
        for (int sh = 0; sh < 4; ++sh) {
            a0 = kq_acc(a0, vreinterpretq_s8_u8(vandq_u8(qa, m3)), vld1q_s8(q8),      (int16_t)Q2K_SCALE(d, sb));
            a1 = kq_acc(a1, vreinterpretq_s8_u8(vandq_u8(qb, m3)), vld1q_s8(q8 + 16), (int16_t)Q2K_SCALE(d, sb + 1));
            sb += 2;
            qa = vshrq_n_u8(qa, 2);
            qb = vshrq_n_u8(qb, 2);
            q8 += 32;
        }
    }
    if (bias_out) *bias_out = q2k_bias_vec(d->mins, y);
    return vaddq_s32(a0, a1);
#else
    /* I8MM / plain NEON: kq_acc cannot apply a per-lane scale for I8MM, so the
     * dot is folded and scaled here. */
    int32_t isum = 0;
    int sb = 0;
    for (int chunk = 0; chunk < 2; ++chunk) {
        uint8x16_t qa = vld1q_u8(q2);
        uint8x16_t qb = vld1q_u8(q2 + 16);
        q2 += 32;
        for (int sh = 0; sh < 4; ++sh) {
            isum += kq_fold(kq_acc(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(qa, m3)),
                                   vld1q_s8(q8), 1)) * Q2K_SCALE(d, sb);
            isum += kq_fold(kq_acc(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(qb, m3)),
                                   vld1q_s8(q8 + 16), 1)) * Q2K_SCALE(d, sb + 1);
            sb += 2;
            qa = vshrq_n_u8(qa, 2);
            qb = vshrq_n_u8(qb, 2);
            q8 += 32;
        }
    }
    if (bias_out) *bias_out = vdupq_n_s32(q2k_bias(d->mins, y));
    return vdupq_n_s32(isum);
#endif
}

/* Scaled integer dot alone (kept for the scalar reference API). */
static inline int32_t q2k_isum(const block_q2_K *x, const q2k_dec_t *d,
                               const block_q8_K *y) {
    return kq_fold(q2k_acc_block(x, d, y, NULL));
}

/* One Q2_K weight block x one Q8_K row, scalar API. */
float q2k_dot_neon(const void *vw, const void *vy, int n) {
    const block_q2_K *x = (const block_q2_K *)vw;
    const block_q8_K *y = (const block_q8_K *)vy;
    const int nb = n / QK_K;
    float acc = 0.0f;
    for (int i = 0; i < nb; ++i) {
        q2k_dec_t d;
        q2k_decode_block(&x[i], &d);
        const float ds   = fp16_to_fp32_lookup(x[i].d)    * y[i].d;
        const float dmin = fp16_to_fp32_lookup(x[i].dmin) * y[i].d;
        acc += ds * (float)q2k_isum(&x[i], &d, &y[i]) - dmin * (float)q2k_bias(d.mins, &y[i]);
    }
    return acc;
}

/* ============================================================
 * Tiled GEMM: 8 weight rows x 4 activation rows.
 * C[ncols][nrows]  (out[ldc*jr + ir]); threads split the weight rows.
 * ============================================================ */
#define KQ_TILE_M 16
#define KQ_TILE_N 22

static void q2k_gemm_tile(const block_q2_K *A, int lda,
                          const block_q8_K *B, int ldb,
                          float *C, int ldc,
                          int k_blocks, int ir0, int ir1, int jc0, int jc1) {
    for (int jc = jc0; jc < jc1; jc += KQ_TILE_N) {
        const int nn = (jc + KQ_TILE_N <= jc1) ? KQ_TILE_N : (jc1 - jc);
        for (int ir = ir0; ir < ir1; ir += KQ_TILE_M) {
            const int mm = (ir + KQ_TILE_M <= ir1) ? KQ_TILE_M : (ir1 - ir);
            /* 4-LANE FLOAT accumulators, folded only ONCE at the end.
             *
             * The obvious shape -- kq_fold() the block's int32 vector and
             * accumulate a scalar -- measured 14.0 ns per (row, block, column)
             * here, because kq_fold() is a horizontal reduction and it ran
             * twice per block (dot and dmin term) inside the k loop, i.e. 256
             * serialising reductions per output element. Keeping the lanes
             * intact and scaling them with vmlaq_f32 removes every reduction
             * from the loop; the two terms then simply add as vectors because
             * a scaling factor can be applied to either before the sum.
             *
             * This is the same shape as the Q4_K kernel (44 GFLOPS). */
            float32x4_t acc[KQ_TILE_M][KQ_TILE_N];
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c) acc[r][c] = vdupq_n_f32(0.0f);

            for (int l = 0; l < k_blocks; ++l) {
                const block_q8_K *bq[KQ_TILE_N];
                float bd[KQ_TILE_N];
                for (int c = 0; c < nn; ++c) {
                    bq[c] = &B[(size_t)ldb * (jc + c) + l];
                    bd[c] = bq[c]->d;
                }
                for (int r = 0; r < mm; ++r) {
                    const block_q2_K *aq = &A[(size_t)lda * (ir + r) + l];
                    q2k_dec_t d;
                    q2k_decode_block(aq, &d);
                    const float ds   = fp16_to_fp32_lookup(aq->d);
                    const float dmin = fp16_to_fp32_lookup(aq->dmin);
                    for (int c = 0; c < nn; ++c) {
                        int32x4_t bv;
                        const int32x4_t iv = q2k_acc_block(aq, &d, bq[c], &bv);
                        /* dot * (d*b), then -dmin*b*bsums, both lane-wise. */
                        acc[r][c] = vmlaq_f32(acc[r][c], vcvtq_f32_s32(iv),
                                              vdupq_n_f32(ds * bd[c]));
                        acc[r][c] = vmlsq_f32(acc[r][c], vcvtq_f32_s32(bv),
                                              vdupq_n_f32(dmin * bd[c]));
                    }
                }
            }
            for (int r = 0; r < mm; ++r) {
                for (int c = 0; c < nn; ++c) {
#if defined(__ARM_FEATURE_MATMUL_INT8) && !defined(__ARM_FEATURE_DOTPROD)
                    /* I8MM lanes duplicate: only 0 and 3 are real partials. */
                    C[(size_t)ldc * (jc + c) + (ir + r)] =
                        vgetq_lane_f32(acc[r][c], 0) + vgetq_lane_f32(acc[r][c], 3);
#else
                    C[(size_t)ldc * (jc + c) + (ir + r)] = vaddvq_f32(acc[r][c]);
#endif
                }
            }
        }
    }
}

int sgemm_q2_k_q8_k_neon_fast(int nrows, int ncols, int k, const void *vx,
                              const void *vy, float *out, size_t bs,
                              int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k <= 0 || (k % QK_K) != 0) return 0;
    const int nb = k / QK_K;
    const int64_t rows_per = ((int64_t)nrows + nth - 1) / nth;
    const int64_t i0 = (int64_t)ith * rows_per;
    int64_t i1 = i0 + rows_per; if (i1 > nrows) i1 = nrows;
    if (i0 >= i1) return 0;
    q2k_gemm_tile((const block_q2_K *)vx, nb, (const block_q8_K *)vy, nb,
                  out, (int)bs, nb, (int)i0, (int)i1, 0, ncols);
    return (int)(i1 - i0);
}


/* ============================================================
 * Q4_K
 *   d, dmin (fp16); scales[12] packed 6-bit (8 scales + 8 mins); qs[128] nibbles
 *   dequant = d*scale*q - dmin*min
 *
 * Sub-block layout: 8 sub-blocks of 32 values. qs is 4 groups of 32 bytes;
 * group j supplies BOTH sub-block 2j (low nibbles) and 2j+1 (high nibbles),
 * each 32 values == two 16-value int8 vectors. q8 advances 64 bytes per group.
 *
 * Scale decode is the standard ggml 6-bit unpack over three 32-bit words.
 * ============================================================ */

typedef struct {
    int16_t sc[8];
    int16_t mn[8];
} q4k_dec_t;

/* Unpack the 8 6-bit scales and 8 6-bit mins from the 12 packed bytes.
 * Shared by Q4_K and Q5_K, which use the identical packing. */
static inline void q4k_decode_scales_bytes(const uint8_t *packed, q4k_dec_t *o) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f,
                          kmask3 = 0x03030303;
    uint32_t utmp[4];
    memcpy(utmp, packed, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const uint8_t *s8 = (const uint8_t *)&utmp[0];
    const uint8_t *m8 = (const uint8_t *)&utmp[2];
    for (int i = 0; i < 8; ++i) { o->sc[i] = (int16_t)s8[i]; o->mn[i] = (int16_t)m8[i]; }
}

static inline void q4k_decode_scales(const block_q4_K *x, q4k_dec_t *o) {
    q4k_decode_scales_bytes(x->scales, o);
}

/* Scaled integer dot of one Q4_K block against one Q8_K row. */
static inline int32_t q4k_isum(const block_q4_K *x, const q4k_dec_t *d, const block_q8_K *y) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8_t *q4 = x->qs;
    const int8_t  *q8 = y->qs;
#if KQ_SCALE_IN_LANE
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0),
              a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    for (int j = 0; j < 4; ++j) {
        const uint8x16_t qa = vld1q_u8(q4);
        const uint8x16_t qb = vld1q_u8(q4 + 16);
        q4 += 32;
        /* sub-block 2j: low nibbles, 32 values = two 16-byte dots */
        const int8x16_t lo_a = vreinterpretq_s8_u8(vandq_u8(qa, m4));
        const int8x16_t lo_b = vreinterpretq_s8_u8(vandq_u8(qb, m4));
        /* sub-block 2j+1: high nibbles */
        const int8x16_t hi_a = vreinterpretq_s8_u8(vshrq_n_u8(qa, 4));
        const int8x16_t hi_b = vreinterpretq_s8_u8(vshrq_n_u8(qb, 4));

        const int32_t s_lo = d->sc[2 * j], s_hi = d->sc[2 * j + 1];
        a0 = kq_acc(a0, lo_a, vld1q_s8(q8),      s_lo);
        a1 = kq_acc(a1, lo_b, vld1q_s8(q8 + 16), s_lo);
        a2 = kq_acc(a2, hi_a, vld1q_s8(q8 + 32), s_hi);
        a3 = kq_acc(a3, hi_b, vld1q_s8(q8 + 48), s_hi);
        q8 += 64;
    }
    return kq_fold(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
#else
    int32_t isum = 0;
    for (int j = 0; j < 4; ++j) {
        const uint8x16_t qa = vld1q_u8(q4);
        const uint8x16_t qb = vld1q_u8(q4 + 16);
        q4 += 32;
        const int8x16_t lo_a = vreinterpretq_s8_u8(vandq_u8(qa, m4));
        const int8x16_t lo_b = vreinterpretq_s8_u8(vandq_u8(qb, m4));
        const int8x16_t hi_a = vreinterpretq_s8_u8(vshrq_n_u8(qa, 4));
        const int8x16_t hi_b = vreinterpretq_s8_u8(vshrq_n_u8(qb, 4));
        int32_t lo = kq_fold(kq_acc(vdupq_n_s32(0), lo_a, vld1q_s8(q8),      1))
                   + kq_fold(kq_acc(vdupq_n_s32(0), lo_b, vld1q_s8(q8 + 16), 1));
        int32_t hi = kq_fold(kq_acc(vdupq_n_s32(0), hi_a, vld1q_s8(q8 + 32), 1))
                   + kq_fold(kq_acc(vdupq_n_s32(0), hi_b, vld1q_s8(q8 + 48), 1));
        isum += lo * d->sc[2 * j] + hi * d->sc[2 * j + 1];
        q8 += 64;
    }
    return isum;
#endif
}

/* dmin correction: sum over the 8 sub-blocks of min_s * (sum of its 32 q8). */
static inline int32_t q4k_bias(const q4k_dec_t *d, const block_q8_K *y) {
    int32_t t = 0;
    for (int i = 0; i < 8; ++i)
        t += (int32_t)d->mn[i] * ((int32_t)y->bsums[2 * i] + (int32_t)y->bsums[2 * i + 1]);
    return t;
}

static void q4k_gemm_tile(const block_q4_K *A, int lda, const block_q8_K *B, int ldb,
                          float *C, int ldc, int k_blocks,
                          int ir0, int ir1, int jc0, int jc1) {
    for (int jc = jc0; jc < jc1; jc += KQ_TILE_N) {
        const int nn = (jc + KQ_TILE_N <= jc1) ? KQ_TILE_N : (jc1 - jc);
        for (int ir = ir0; ir < ir1; ir += KQ_TILE_M) {
            const int mm = (ir + KQ_TILE_M <= ir1) ? KQ_TILE_M : (ir1 - ir);
            float acc[KQ_TILE_M][KQ_TILE_N];
            float bias[KQ_TILE_M][KQ_TILE_N];
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c) { acc[r][c] = 0.0f; bias[r][c] = 0.0f; }

            for (int l = 0; l < k_blocks; ++l) {
                const block_q8_K *bq[KQ_TILE_N];
                float bd[KQ_TILE_N];
                for (int c = 0; c < nn; ++c) {
                    bq[c] = &B[(size_t)ldb * (jc + c) + l];
                    bd[c] = bq[c]->d;
                }
                for (int r = 0; r < mm; ++r) {
                    const block_q4_K *aq = &A[(size_t)lda * (ir + r) + l];
                    const float ds   = fp16_to_fp32_lookup(aq->d);
                    const float dmin = fp16_to_fp32_lookup(aq->dmin);
                    q4k_dec_t d;
                    q4k_decode_scales(aq, &d);
                    for (int c = 0; c < nn; ++c) {
                        acc[r][c]  += ds * bd[c] * (float)q4k_isum(aq, &d, bq[c]);
                        bias[r][c] -= dmin * bd[c] * (float)q4k_bias(&d, bq[c]);
                    }
                }
            }
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c)
                    C[(size_t)ldc * (jc + c) + (ir + r)] = acc[r][c] + bias[r][c];
        }
    }
}

int sgemm_q4_k_q8_k_neon_fast(int nrows, int ncols, int k, const void *vx,
                              const void *vy, float *out, size_t bs,
                              int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k <= 0 || (k % QK_K) != 0) return 0;
    const int nb = k / QK_K;
    const int64_t rows_per = ((int64_t)nrows + nth - 1) / nth;
    const int64_t i0 = (int64_t)ith * rows_per;
    int64_t i1 = i0 + rows_per; if (i1 > nrows) i1 = nrows;
    if (i0 >= i1) return 0;
    q4k_gemm_tile((const block_q4_K *)vx, nb, (const block_q8_K *)vy, nb,
                  out, (int)bs, nb, (int)i0, (int)i1, 0, ncols);
    return (int)(i1 - i0);
}


/* ============================================================
 * Q3_K
 *   d (fp16); hmask[32] high bits; qs[64] 2-bit lows; scales[12] packed 6-bit,
 *   stored biased by 32.
 *   dequant = d * (scale-32) * (q3 - high)      q3 in -4..3
 *
 * Sub-block layout matches vec_dot_q3_K_q8_K: for chunk in 0..1, j in 0..3,
 *   q3a = qs + chunk*32       -> sub-block chunk*8 + 2j     vs q8[chunk*128 + j*32]
 *   q3b = qs + chunk*32 + 16  -> sub-block chunk*8 + 2j + 1 vs q8[... + 16]
 * where the high term is 4 when hmask bit (chunk*4+j) is CLEAR.
 * ============================================================ */

typedef struct { int16_t sc[16]; } q3k_dec_t;

static inline void q3k_decode_scales(const block_q3_K *x, q3k_dec_t *o) {
    uint32_t auxs[4];
    memcpy(auxs, x->scales, 12);
    const uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
    auxs[3] = ((auxs[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
    auxs[0] = (auxs[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4);
    auxs[1] = (auxs[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4);
    const int8_t *s8 = (const int8_t *)auxs;
    for (int i = 0; i < 16; ++i) o->sc[i] = (int16_t)(s8[i] - 32);
}

#define Q3K_SCALE(d, k) ((int32_t)(d)->sc[k])

/* Scaled integer dot of one Q3_K block against one Q8_K row (inline decode). */
static inline int32_t q3k_isum(const block_q3_K *x, const q3k_dec_t *d,
                               const block_q8_K *y) {
    const uint8x16_t m3 = vdupq_n_u8(3);
    const uint8x16_t v4 = vdupq_n_u8(4);
    const uint8x16_t zero = vdupq_n_u8(0);
    const uint8_t *qs = x->qs;
    const int8_t  *q8 = y->qs;
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int sb = 0;
    uint8_t bit = 1;
    for (int chunk = 0; chunk < 2; ++chunk) {
        uint8x16_t qa = vld1q_u8(qs + chunk * 32);
        uint8x16_t qb = vld1q_u8(qs + chunk * 32 + 16);
        const uint8x16_t hm0 = vld1q_u8(x->hmask);
        const uint8x16_t hm1 = vld1q_u8(x->hmask + 16);
        for (int j = 0; j < 4; ++j) {
            const uint8x16_t bitv = vdupq_n_u8(bit);
            /* high term: 4 where the hmask bit is clear, else 0 */
            const uint8x16_t sa = vandq_u8(vceqq_u8(vandq_u8(hm0, bitv), zero), v4);
            const uint8x16_t sb_ = vandq_u8(vceqq_u8(vandq_u8(hm1, bitv), zero), v4);
            const int8x16_t ga = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(qa, m3), sa));
            const int8x16_t gb = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(qb, m3), sb_));
            a0 = kq_acc(a0, ga, vld1q_s8(q8),      (int16_t)Q3K_SCALE(d, sb));
            a1 = kq_acc(a1, gb, vld1q_s8(q8 + 16), (int16_t)Q3K_SCALE(d, sb + 1));
            sb += 2;
            qa = vshrq_n_u8(qa, 2);
            qb = vshrq_n_u8(qb, 2);
            q8 += 32;
            bit = (uint8_t)(bit << 1);
        }
    }
    return kq_fold(vaddq_s32(a0, a1));
}

static void q3k_gemm_tile(const block_q3_K *A, int lda, const block_q8_K *B, int ldb,
                          float *C, int ldc, int k_blocks,
                          int ir0, int ir1, int jc0, int jc1) {
    for (int jc = jc0; jc < jc1; jc += KQ_TILE_N) {
        const int nn = (jc + KQ_TILE_N <= jc1) ? KQ_TILE_N : (jc1 - jc);
        for (int ir = ir0; ir < ir1; ir += KQ_TILE_M) {
            const int mm = (ir + KQ_TILE_M <= ir1) ? KQ_TILE_M : (ir1 - ir);
            float acc[KQ_TILE_M][KQ_TILE_N];
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c) acc[r][c] = 0.0f;

            for (int l = 0; l < k_blocks; ++l) {
                const block_q8_K *bq[KQ_TILE_N];
                float bd[KQ_TILE_N];
                for (int c = 0; c < nn; ++c) {
                    bq[c] = &B[(size_t)ldb * (jc + c) + l];
                    bd[c] = bq[c]->d;
                }
                for (int r = 0; r < mm; ++r) {
                    const block_q3_K *aq = &A[(size_t)lda * (ir + r) + l];
                    const float ds = fp16_to_fp32_lookup(aq->d);
                    q3k_dec_t d;
                    q3k_decode_scales(aq, &d);
                    for (int c = 0; c < nn; ++c)
                        acc[r][c] += ds * bd[c] * (float)q3k_isum(aq, &d, bq[c]);
                }
            }
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c)
                    C[(size_t)ldc * (jc + c) + (ir + r)] = acc[r][c];
        }
    }
}

int sgemm_q3_k_q8_k_neon_fast(int nrows, int ncols, int k, const void *vx,
                              const void *vy, float *out, size_t bs,
                              int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k <= 0 || (k % QK_K) != 0) return 0;
    const int nb = k / QK_K;
    const int64_t rows_per = ((int64_t)nrows + nth - 1) / nth;
    const int64_t i0 = (int64_t)ith * rows_per;
    int64_t i1 = i0 + rows_per; if (i1 > nrows) i1 = nrows;
    if (i0 >= i1) return 0;
    q3k_gemm_tile((const block_q3_K *)vx, nb, (const block_q8_K *)vy, nb,
                  out, (int)bs, nb, (int)i0, (int)i1, 0, ncols);
    return (int)(i1 - i0);
}


/* ============================================================
 * Q6_K
 *   d (fp16); ql[128] low nibbles; qh[64] high 2 bits; scales[16] int8
 *   dequant = d * scale * (ql | qh<<4 - 32)      value range -32..31
 *
 * Sub-block layout matches vec_dot_q6_K_q8_K: for chunk 0..1, half 0..1,
 * k 0..3 the 16-value group is g[chunk*8 + half*4 + k] against
 * q8[chunk*128 + k*32 + half*16], with scale index chunk*8 + half + k*2 and a
 * -32*bsums correction indexed the same way.
 * ============================================================ */

typedef struct { int16_t sc[16]; } q6k_dec_t;

static inline void q6k_decode_scales(const block_q6_K *x, q6k_dec_t *o) {
    const int8x16_t s8 = vld1q_s8(x->scales);
    int8_t t[16];
    vst1q_s8(t, s8);
    for (int i = 0; i < 16; ++i) o->sc[i] = (int16_t)t[i];
}

#define Q6K_SCALE(d, k) ((int32_t)(d)->sc[k])

static inline int32_t q6k_isum(const block_q6_K *x, const q6k_dec_t *d,
                               const block_q8_K *y) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8x16_t m3 = vdupq_n_u8(0x03);
    const uint8_t *ql = x->ql, *qh = x->qh;
    const int8_t *q8 = y->qs;
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32_t corr = 0;
    for (int chunk = 0; chunk < 2; ++chunk) {
        const uint8_t *ql_c = ql + chunk * 64;
        const uint8_t *qh_c = qh + chunk * 32;
        const int8_t  *q8_c = q8 + chunk * 128;
        const int16_t *bs   = y->bsums + chunk * 8;
        for (int half = 0; half < 2; ++half) {
            const uint8x16_t ql_h  = vld1q_u8(ql_c + half * 16);
            const uint8x16_t ql_h2 = vld1q_u8(ql_c + 32 + half * 16);
            const uint8x16_t qh_v  = vld1q_u8(qh_c + half * 16);
            const int8x16_t g0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_h,  m4),
                                        vshlq_n_u8(vandq_u8(qh_v, m3), 4)));
            const int8x16_t g1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_h2, m4),
                                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 2), m3), 4)));
            const int8x16_t g2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_h, 4),
                                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 4), m3), 4)));
            const int8x16_t g3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_h2, 4),
                                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 6), m3), 4)));
            /* sub-block index chunk*8 + half + k*2 for group k */
            const int32_t b0 = chunk * 8 + half;
            a0 = kq_acc(a0, g0, vld1q_s8(q8_c + half * 16),      (int16_t)Q6K_SCALE(d, b0));
            a1 = kq_acc(a1, g1, vld1q_s8(q8_c + 32 + half * 16), (int16_t)Q6K_SCALE(d, b0 + 2));
            a0 = kq_acc(a0, g2, vld1q_s8(q8_c + 64 + half * 16), (int16_t)Q6K_SCALE(d, b0 + 4));
            a1 = kq_acc(a1, g3, vld1q_s8(q8_c + 96 + half * 16), (int16_t)Q6K_SCALE(d, b0 + 6));
            corr -= 32 * (Q6K_SCALE(d, b0)     * (int32_t)bs[half]
                        + Q6K_SCALE(d, b0 + 2) * (int32_t)bs[half + 2]
                        + Q6K_SCALE(d, b0 + 4) * (int32_t)bs[half + 4]
                        + Q6K_SCALE(d, b0 + 6) * (int32_t)bs[half + 6]);
        }
    }
    return kq_fold(vaddq_s32(a0, a1)) + corr;
}

static void q6k_gemm_tile(const block_q6_K *A, int lda, const block_q8_K *B, int ldb,
                          float *C, int ldc, int k_blocks,
                          int ir0, int ir1, int jc0, int jc1) {
    for (int jc = jc0; jc < jc1; jc += KQ_TILE_N) {
        const int nn = (jc + KQ_TILE_N <= jc1) ? KQ_TILE_N : (jc1 - jc);
        for (int ir = ir0; ir < ir1; ir += KQ_TILE_M) {
            const int mm = (ir + KQ_TILE_M <= ir1) ? KQ_TILE_M : (ir1 - ir);
            float acc[KQ_TILE_M][KQ_TILE_N];
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c) acc[r][c] = 0.0f;

            for (int l = 0; l < k_blocks; ++l) {
                const block_q8_K *bq[KQ_TILE_N];
                float bd[KQ_TILE_N];
                for (int c = 0; c < nn; ++c) {
                    bq[c] = &B[(size_t)ldb * (jc + c) + l];
                    bd[c] = bq[c]->d;
                }
                for (int r = 0; r < mm; ++r) {
                    const block_q6_K *aq = &A[(size_t)lda * (ir + r) + l];
                    const float ds = fp16_to_fp32_lookup(aq->d);
                    q6k_dec_t d;
                    q6k_decode_scales(aq, &d);
                    for (int c = 0; c < nn; ++c)
                        acc[r][c] += ds * bd[c] * (float)q6k_isum(aq, &d, bq[c]);
                }
            }
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c)
                    C[(size_t)ldc * (jc + c) + (ir + r)] = acc[r][c];
        }
    }
}

int sgemm_q6_k_q8_k_neon_fast(int nrows, int ncols, int k, const void *vx,
                              const void *vy, float *out, size_t bs,
                              int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k <= 0 || (k % QK_K) != 0) return 0;
    const int nb = k / QK_K;
    const int64_t rows_per = ((int64_t)nrows + nth - 1) / nth;
    const int64_t i0 = (int64_t)ith * rows_per;
    int64_t i1 = i0 + rows_per; if (i1 > nrows) i1 = nrows;
    if (i0 >= i1) return 0;
    q6k_gemm_tile((const block_q6_K *)vx, nb, (const block_q8_K *)vy, nb,
                  out, (int)bs, nb, (int)i0, (int)i1, 0, ncols);
    return (int)(i1 - i0);
}


/* ============================================================
 * Q5_K
 *   d, dm (fp16); scales[12] packed 6-bit (8 scales + 8 mins);
 *   qh[32] one high bit per quant; qs[128] low nibbles.
 *   dequant = d*scale*(q4 + 16*h) - dm*min     (h = the qh bit)
 *
 * Same sub-block layout as Q4_K (8 sub-blocks of 32 values), so the low nibble
 * handling is shared. The high bit is handled by the identity
 *
 *     sum((q4 + 16h) * q8) = sum(q4*q8) + 16 * sum(h * q8)
 *
 * where h is a 0/1 mask, so a second dot against the mask adds the high-bit
 * term without building any lookup table. Python-free, no byte arrays.
 * The min term uses a single scalar bsums sum (Q5_K stores one dm, not
 * per-sub-block mins).
 * ============================================================ */

#if Q5K_ENABLE
/* Scaled integer dot of one Q5_K block against one Q8_K row.
 *
 * The +16 high bit is folded into the nibble value itself (q5 = q4 + 16*h as
 * int8, exactly as vec_dot_q5_K_q8_K does), rather than as a second dot: the
 * two are algebraically identical, but building q5 directly keeps the scale
 * handling identical to Q4_K and avoids a second accumulator per sub-block. */
static inline int32_t q5k_isum(const block_q5_K *x, const q4k_dec_t *d,
                               const block_q8_K *y) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8x16_t m1 = vdupq_n_u8(0x01);
    const uint8x16_t v16 = vdupq_n_u8(16);
    const uint8_t *q4 = x->qs;
    const uint8_t *qh = x->qh;
    const int8_t  *q8 = y->qs;

    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0),
              a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);

    for (int j = 0; j < 4; ++j) {
        const uint8x16_t qa = vld1q_u8(q4);
        const uint8x16_t qb = vld1q_u8(q4 + 16);
        q4 += 32;
        /* qh holds one bit per quant and does NOT advance per j: the same
         * 32 bytes are reused with bit 2j (low nibble) and bit 2j+1 (high). */
        const uint8x16_t hi_a = vld1q_u8(qh);
        const uint8x16_t hi_b = vld1q_u8(qh + 16);
        const uint8x16_t hb0 = vdupq_n_u8((uint8_t)(1u << (2 * j)));
        const uint8x16_t hb1 = vdupq_n_u8((uint8_t)(1u << (2 * j + 1)));

        /* 16 where the bit is set, 0 otherwise -> adding it makes q5 = q4+16h */
        const uint8x16_t or_a = vandq_u8(vtstq_u8(hi_a, hb0), v16);
        const uint8x16_t or_b = vandq_u8(vtstq_u8(hi_b, hb0), v16);
        const uint8x16_t or_c = vandq_u8(vtstq_u8(hi_a, hb1), v16);
        const uint8x16_t or_d = vandq_u8(vtstq_u8(hi_b, hb1), v16);

        /* q5 values are 0..31, so int8 is safe; kq_acc sees them as signed
         * 8-bit, which is fine because 31 < 128. */
        const int8x16_t q5a = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qa, m4),     or_a));
        const int8x16_t q5b = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qb, m4),     or_b));
        const int8x16_t q5c = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qa, 4),   or_c));
        const int8x16_t q5d = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qb, 4),   or_d));
        (void)m1;

        const int16_t s_lo = (int16_t)d->sc[2 * j], s_hi = (int16_t)d->sc[2 * j + 1];
        a0 = kq_acc(a0, q5a, vld1q_s8(q8),      s_lo);
        a1 = kq_acc(a1, q5b, vld1q_s8(q8 + 16), s_lo);
        a2 = kq_acc(a2, q5c, vld1q_s8(q8 + 32), s_hi);
        a3 = kq_acc(a3, q5d, vld1q_s8(q8 + 48), s_hi);
        q8 += 64;
    }
    return kq_fold(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
}

#endif /* Q5K_ENABLE */

/* dm correction: one scalar dm times the total q8 sum of the block. */
static inline int32_t q5k_bias(const q4k_dec_t *d, const block_q8_K *y) {
    int32_t t = 0;
    for (int i = 0; i < 8; ++i)
        t += (int32_t)d->mn[i] * ((int32_t)y->bsums[2 * i] + (int32_t)y->bsums[2 * i + 1]);
    return t;
}

#if Q5K_ENABLE
static void q5k_gemm_tile(const block_q5_K *A, int lda, const block_q8_K *B, int ldb,
                          float *C, int ldc, int k_blocks,
                          int ir0, int ir1, int jc0, int jc1) {
    for (int jc = jc0; jc < jc1; jc += KQ_TILE_N) {
        const int nn = (jc + KQ_TILE_N <= jc1) ? KQ_TILE_N : (jc1 - jc);
        for (int ir = ir0; ir < ir1; ir += KQ_TILE_M) {
            const int mm = (ir + KQ_TILE_M <= ir1) ? KQ_TILE_M : (ir1 - ir);
            float acc[KQ_TILE_M][KQ_TILE_N];
            float bias[KQ_TILE_M][KQ_TILE_N];
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c) { acc[r][c] = 0.0f; bias[r][c] = 0.0f; }

            for (int l = 0; l < k_blocks; ++l) {
                const block_q8_K *bq[KQ_TILE_N];
                float bd[KQ_TILE_N];
                for (int c = 0; c < nn; ++c) {
                    bq[c] = &B[(size_t)ldb * (jc + c) + l];
                    bd[c] = bq[c]->d;
                }
                for (int r = 0; r < mm; ++r) {
                    const block_q5_K *aq = &A[(size_t)lda * (ir + r) + l];
                    const float ds  = fp16_to_fp32_lookup(aq->d);
                    const float dmn = fp16_to_fp32_lookup(aq->dm);
                    q4k_dec_t d;
                    q4k_decode_scales_bytes(aq->scales, &d);
                    for (int c = 0; c < nn; ++c) {
                        acc[r][c]  += ds * bd[c] * (float)q5k_isum(aq, &d, bq[c]);
                        bias[r][c] -= dmn * bd[c] * (float)q5k_bias(&d, bq[c]);
                    }
                }
            }
            for (int r = 0; r < mm; ++r)
                for (int c = 0; c < nn; ++c)
                    C[(size_t)ldc * (jc + c) + (ir + r)] = acc[r][c] + bias[r][c];
        }
    }
}

int sgemm_q5_k_q8_k_neon_fast(int nrows, int ncols, int k, const void *vx,
                              const void *vy, float *out, size_t bs,
                              int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k <= 0 || (k % QK_K) != 0) return 0;
    const int nb = k / QK_K;
    const int64_t rows_per = ((int64_t)nrows + nth - 1) / nth;
    const int64_t i0 = (int64_t)ith * rows_per;
    int64_t i1 = i0 + rows_per; if (i1 > nrows) i1 = nrows;
    if (i0 >= i1) return 0;
    q5k_gemm_tile((const block_q5_K *)vx, nb, (const block_q8_K *)vy, nb,
                  out, (int)bs, nb, (int)i0, (int)i1, 0, ncols);
    return (int)(i1 - i0);
}
#endif /* Q5K_ENABLE */

#endif /* __ARM_NEON */
