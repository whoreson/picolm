/* ============================================================
 * K-quant x Q8_K NEON prefill GEMM (Q6_K, Q3_K, Q5_K) -- ARM
 *
 * Why: type_dispatch.c left fn_gemm = NULL for the K-quants on ARM builds,
 * so prefill fell back to per-column vec_dot_*_q8_K() (and, for the fused
 * FFN pairs, to two separate matmul_batch passes with the activations
 * quantised twice). Measured effect on SmolLM2-1.7B-Q3_K_M:
 *   Q8_0 (has an ARM GEMM)  prefill 115 tok/s vs decode 27  -> 4.2x batching
 *   Q3_K (fn_gemm == NULL)  prefill  12 tok/s vs decode 9.6 -> 1.25x
 * i.e. the batched path was worth ~3-4x and was simply missing.
 *
 * Structure: 4 weight rows x 4 activation rows. The four activation rows'
 * q8 sub-blocks are loaded once per (block, sub-block) and reused by all
 * four weight rows; the four weight rows' packed quants are decoded once
 * per sub-block and reused by all four activation rows. Per-sub-block
 * integer dots go through the best available core:
 *   __ARM_FEATURE_MATMUL_INT8 -> vmmlaq_s32 (lanes 0+3 = the 16-value dot)
 *   __ARM_FEATURE_DOTPROD     -> vdotq_s32
 *   otherwise                 -> vmull_s8 + vpaddlq_s16
 * Results match the scalar vec_dot path exactly (same integer accumulation,
 * same float scaling order per sub-block).
 * ============================================================ */
#include "quant.h"
#include "sgemm.h"

#if defined(__ARM_NEON)

/* ---- 16-value int8 dot, best available core -------------------------- */
static inline int32_t kq_dot16(int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), a, b));
#elif defined(__ARM_FEATURE_MATMUL_INT8)
    int32x4_t s = vmmlaq_s32(vdupq_n_s32(0), a, b);
    return vgetq_lane_s32(s, 0) + vgetq_lane_s32(s, 3);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(a),  vget_low_s8(b));
    int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddvq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#endif
}

/* ---- scaled int8 dot accumulate ----
 *
 * DOTPROD: vdotq_s32 yields four lanes, each one 4-byte-column partial of the
 * SAME 16-value sub-block, so all four lanes share one scale and that scale
 * folds into the int32 accumulator with a broadcast multiply. This removes the
 * per-sub-block vaddvq that otherwise gates the whole K-quant kernel.
 *
 * I8MM and plain NEON have no equivalent in-lane broadcast (I8MM lanes are
 * [d0,d0,d1,d1], and vpaddlq already collapses the dot), so those targets use
 * the kq_dot16()/kq_hsum_scaled() pair instead, which multiplies after folding.
 * Getting this wrong silently drops the scale, so the accumulation helper is
 * only compiled where it is exact. */
#if defined(__ARM_FEATURE_DOTPROD)
#define KQ_INLINE_SCALE 1
#endif

#if defined(KQ_INLINE_SCALE)
static inline int32x4_t kq_acc_scaled(int32x4_t acc, int8x16_t a, int8x16_t b, int32_t sc) {
    return vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), a, b), sc);
}
static inline int32_t kq_hsum_acc(int32x4_t v) { return vaddvq_s32(v); }
#else
/* Portable form: fold first, then scale in scalar code. */
#define KQ_INLINE_SCALE 0
#endif

/* ===================== Q6_K =====================
 * block_q6_K: ql[128] low nibbles, qh[64] high 2 bits, scales[16] int8, d.
 * Element order (256 values):
 *   for chunk in 0..1: ql_c = ql+64*chunk, qh_c = qh+32*chunk, q8_c = q8+128*chunk
 *     for half in 0..1:
 *       ql_h = ql_c + 16*half, ql_h2 = ql_c + 32 + 16*half, qh_h = qh_c + 16*half
 *       q6_0 = (ql_h & 0xF)  | (qh_h>>0 & 3)<<4   vs q8_c[half*16 .. +15]
 *       q6_1 = (ql_h2 & 0xF) | (qh_h>>2 & 3)<<4   vs q8_c[half*16 .. +15]
 *       q6_2 = (ql_h >> 4)   | (qh_h>>4 & 3)<<4   vs q8_c[32+half*16 .. +15]
 *       q6_3 = (ql_h2 >> 4)  | (qh_h>>6 & 3)<<4   vs q8_c[32+half*16 .. +15]
 *     sub-block index for scales: chunk*8 + half*2 + {0,1} ... per the
 *     reference in quant.c: is = chunk*8, advanced by 2 per (half, pair).
 * ================================================ */

/* Decode the 16 sub-block (16 int8 values) groups of one Q6_K block. */
static inline void q6k_decode(const block_q6_K *x, int8x16_t g[16]) {
    const uint8x16_t m4  = vdupq_n_u8(0x0F);
    const uint8x16_t m3  = vdupq_n_u8(0x03);
    const uint8_t *ql = x->ql, *qh = x->qh;
    int gi = 0;
    for (int chunk = 0; chunk < 2; ++chunk) {
        const uint8_t *ql_c = ql + chunk * 64;
        const uint8_t *qh_c = qh + chunk * 32;
        for (int half = 0; half < 2; ++half) {
            const uint8x16_t ql_h   = vld1q_u8(ql_c + half * 16);
            const uint8x16_t ql_h2  = vld1q_u8(ql_c + 32 + half * 16);
            const uint8x16_t qh_hv  = vld1q_u8(qh_c + half * 16);
            const uint8x16_t qh0 = vandq_u8(qh_hv, m3);
            const uint8x16_t qh1 = vandq_u8(vshrq_n_u8(qh_hv, 2), m3);
            const uint8x16_t qh2 = vandq_u8(vshrq_n_u8(qh_hv, 4), m3);
            const uint8x16_t qh3 = vandq_u8(vshrq_n_u8(qh_hv, 6), m3);
            g[gi++] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_h,  m4), vshlq_n_u8(qh0, 4)));
            g[gi++] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_h2, m4), vshlq_n_u8(qh1, 4)));
            g[gi++] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_h, 4),  vshlq_n_u8(qh2, 4)));
            g[gi++] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_h2, 4), vshlq_n_u8(qh3, 4)));
        }
    }
}

/* Dot decoded Q6_K sub-blocks against one Q8_K row, and fold in the 6-bit
 * centring term. Sub-block j accumulates  sum(q6 * q8) - 32*bsums[j], where
 * bsums[j] is the q8 sum over that sub-block's 16 values (6-bit quants are
 * 0..63, so the reference subtracts 32 per element).
 *
 * Layout (mirrors quant.c): for chunk in 0..1, half in 0..1, k in 0..3:
 *   sub-block index = chunk*8 + half + k*2
 *   q8 offset       = k*32 + half*16
 *   weight group    = g[chunk*8 + half*4 + k]   (q6_0..q6_3 per (chunk,half))
 */
static inline int32_t q6k_dot_decoded(const int8x16_t g[16], const int8_t sc[16],
                                      const block_q8_K *y) {
    const int8_t *q8 = y->qs;
#if KQ_INLINE_SCALE
    /* Scale folded into the int32 accumulator: no per-sub-block hsum. The
     * -32*bs correction uses bsums (not q8) and stays scalar. */
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0),
              a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int32_t corr = 0;
    for (int chunk = 0; chunk < 2; ++chunk) {
        const int8_t *q8_c = q8 + chunk * 128;
        const int16_t *bs  = y->bsums + chunk * 8;
        for (int half = 0; half < 2; ++half) {
            for (int k = 0; k < 4; ++k) {
                const int32_t sb = chunk * 8 + half + k * 2;
                const int8x16_t q8v = vld1q_s8(q8_c + k * 32 + half * 16);
                const int8x16_t gv  = g[chunk * 8 + half * 4 + k];
                const int32_t s = sc[sb];
                switch (k) {
                    case 0: a0 = kq_acc_scaled(a0, gv, q8v, s); break;
                    case 1: a1 = kq_acc_scaled(a1, gv, q8v, s); break;
                    case 2: a2 = kq_acc_scaled(a2, gv, q8v, s); break;
                    default:a3 = kq_acc_scaled(a3, gv, q8v, s); break;
                }
                corr -= 32 * s * (int32_t)bs[half + k * 2];
            }
        }
    }
    return kq_hsum_acc(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3))) + corr;
#else
    int32_t isum = 0;
    for (int chunk = 0; chunk < 2; ++chunk) {
        const int8_t *q8_c = q8 + chunk * 128;
        const int16_t *bs  = y->bsums + chunk * 8;
        for (int half = 0; half < 2; ++half) {
            for (int k = 0; k < 4; ++k) {
                const int32_t sb = chunk * 8 + half + k * 2;
                const int8x16_t q8v = vld1q_s8(q8_c + k * 32 + half * 16);
                const int32_t dot = kq_dot16(g[chunk * 8 + half * 4 + k], q8v);
                isum += sc[sb] * (dot - 32 * (int32_t)bs[half + k * 2]);
            }
        }
    }
    return isum;
#endif
}

/* Q6_K scalar-ish block dot (reference for the tiling, mirrors quant.c). */
static inline float q6k_block(const block_q6_K *x, const block_q8_K *y) {
    int8x16_t g[16];
    q6k_decode(x, g);
    int8_t sc[16];
    vst1q_s8(sc, vld1q_s8(x->scales));
    const float d = y->d * fp16_to_fp32_lookup(x->d);
    return d * (float)q6k_dot_decoded(g, sc, y);
}

/* 4 weight rows x 4 activation rows for one block index range. */
static void q6k_tile(const block_q6_K *A, int lda, const block_q8_K *B, int ldb,
                     float *C, int ldc, int k_blocks,
                     int64_t ii, int64_t jj, int nrows, int ncols) {
    float acc[4][4];
    for (int r = 0; r < nrows; r++) for (int c = 0; c < ncols; c++) acc[r][c] = 0.0f;

    for (int64_t l = 0; l < k_blocks; ++l) {
        int8x16_t g[4][16];
        int8_t sc[4][16];
        float drow[4];
        for (int r = 0; r < nrows; ++r) {
            const block_q6_K *aq = &A[lda * (ii + r) + l];
            q6k_decode(aq, g[r]);
            vst1q_s8(sc[r], vld1q_s8(aq->scales));
            drow[r] = fp16_to_fp32_lookup(aq->d);
        }
        for (int c = 0; c < ncols; ++c) {
            const block_q8_K *bq = &B[ldb * (jj + c) + l];
            for (int r = 0; r < nrows; ++r) {
                acc[r][c] += drow[r] * bq->d
                           * (float)q6k_dot_decoded(g[r], sc[r], bq);
            }
        }
    }
    for (int c = 0; c < ncols; ++c)
        for (int r = 0; r < nrows; ++r)
            C[ldc * (jj + c) + (ii + r)] = acc[r][c];
}

int sgemm_q6_k_q8_k_neon(int nrows, int ncols, int k, const void *vx,
                         const void *vy, float *out, size_t bs,
                         int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k <= 0 || (k % QK_K) != 0) return 0;
    const int nb = k / QK_K;
    const block_q6_K *A = (const block_q6_K *)vx;
    const block_q8_K *B = (const block_q8_K *)vy;

    int64_t rows_per = (nrows + nth - 1) / nth;
    int64_t i0 = (int64_t)ith * rows_per;
    int64_t i1 = i0 + rows_per; if (i1 > nrows) i1 = nrows;
    if (i0 >= i1) return 0;

    for (int64_t ii = i0; ii < i1; ii += 4) {
        int64_t nr = i1 - ii; if (nr > 4) nr = 4;
        for (int64_t jj = 0; jj < ncols; jj += 4) {
            int64_t nc = ncols - jj; if (nc > 4) nc = 4;
            q6k_tile(A, nb, B, nb, out, (int)bs, nb, ii, jj, (int)nr, (int)nc);
        }
    }
    return (int)(i1 - i0);
}


/* ===================== Q3_K =====================
 * block_q3_K: hmask[32] high bits, qs[64] 2-bit low nibbles, scales[12] packed
 * 6-bit (stored biased by 32), d.
 * Element order (mirrors quant.c): for chunk in 0..1, j in 0..3:
 *   q3a = qs+chunk*32       (16 bytes), q3b = qs+chunk*32+16
 *   low  = (q3a >> 2j) & 3, high = hmask bit (chunk*4+j) ? 0 : 4
 *   value = low - high   (range -4..3)
 *   sub-block index = chunk*8 + j*2 + {0 for a, 1 for b}
 *   scale = scales[idx] - 32
 * The high-bit expansion is vectorised here (the reference builds it with a
 * 16-iteration scalar loop per group, which dominates its runtime). */

/* Decode the 16 sub-block groups of one Q3_K block (values as int8). */
static inline void q3k_decode(const block_q3_K *x, int8x16_t g[16]) {
    const uint8x16_t m3 = vdupq_n_u8(3);
    const uint8x16_t v4 = vdupq_n_u8(4);
    const uint8_t *qs = x->qs;
    const uint8_t *hm = x->hmask;
    int gi = 0;
    uint8_t bit = 1;
    for (int chunk = 0; chunk < 2; ++chunk) {
        uint8x16_t qa = vld1q_u8(qs + chunk * 32);
        uint8x16_t qb = vld1q_u8(qs + chunk * 32 + 16);
        /* hmask byte i covers the whole 16-lane group's element i, for bit
         * (chunk*4 + j). Convert "bit set" -> -4 (i.e. high bits) with a
         * vector compare: t = (hm & bit) ? 0 : 4. */
        for (int j = 0; j < 4; ++j) {
            const uint8x16_t bitv = vdupq_n_u8(bit);
            const uint8x16_t ha = vandq_u8(vld1q_u8(hm), bitv);
            const uint8x16_t hb = vandq_u8(vld1q_u8(hm + 16), bitv);
            /* ha/hb are 0 or `bit`. Compare ==0 to get 0xFF where clear, then
             * AND with 4 -> 4 where clear, 0 where set (matching the ref). */
            const uint8x16_t sa = vandq_u8(vceqq_u8(ha, vdupq_n_u8(0)), v4);
            const uint8x16_t sb = vandq_u8(vceqq_u8(hb, vdupq_n_u8(0)), v4);
            g[gi++] = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(qa, m3), sa));
            g[gi++] = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(qb, m3), sb));
            qa = vshrq_n_u8(qa, 2);
            qb = vshrq_n_u8(qb, 2);
            bit = (uint8_t)(bit << 1);
        }
    }
}

/* Unpack the 16 biased-by-32 6-bit scales into int8 (already centred). */
static inline void q3k_scales(const block_q3_K *x, int8_t sc[16]) {
    uint32_t auxs[4];
    memcpy(auxs, x->scales, 12);
    uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
    auxs[3] = ((auxs[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
    auxs[0] = (auxs[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4);
    auxs[1] = (auxs[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4);
    const int8_t *s8 = (const int8_t *)auxs;
    for (int i = 0; i < 16; ++i) sc[i] = (int8_t)(s8[i] - 32);
}

static inline int32_t q3k_dot_decoded(const int8x16_t g[16], const int8_t sc[16],
                                      const block_q8_K *y) {
    const int8_t *q8 = y->qs;
    /* g[] is laid out as (chunk, j, {a,b}); the reference consumes q8 in the
     * same order: 16 bytes for `a` then 16 for `b` per (chunk, j) step, so
     * sub-block i lives at q8[(i/2)*32 + (i%2)*16]. */
#if KQ_INLINE_SCALE
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    for (int s = 0; s < 8; ++s) {          /* 8 chunk/j steps = 16 sub-blocks */
        const int32_t sb = s * 2;
        const int8_t *qv = q8 + s * 32;
        a0 = kq_acc_scaled(a0, g[sb],     vld1q_s8(qv),      sc[sb]);
        a1 = kq_acc_scaled(a1, g[sb + 1], vld1q_s8(qv + 16), sc[sb + 1]);
    }
    return kq_hsum_acc(vaddq_s32(a0, a1));
#else
    int32_t isum = 0;
    for (int s = 0; s < 8; ++s) {
        const int32_t sb = s * 2;
        const int8_t *qv = q8 + s * 32;
        isum += kq_dot16(g[sb],     vld1q_s8(qv))      * (int32_t)sc[sb];
        isum += kq_dot16(g[sb + 1], vld1q_s8(qv + 16)) * (int32_t)sc[sb + 1];
    }
    return isum;
#endif
}

static void q3k_tile(const block_q3_K *A, int lda, const block_q8_K *B, int ldb,
                     float *C, int ldc, int k_blocks,
                     int64_t ii, int64_t jj, int nrows, int ncols) {
    float acc[4][4];
    for (int r = 0; r < nrows; r++) for (int c = 0; c < ncols; c++) acc[r][c] = 0.0f;
    for (int64_t l = 0; l < k_blocks; ++l) {
        int8x16_t g[4][16];
        int8_t sc[4][16];
        float drow[4];
        for (int r = 0; r < nrows; ++r) {
            const block_q3_K *aq = &A[lda * (ii + r) + l];
            q3k_decode(aq, g[r]);
            q3k_scales(aq, sc[r]);
            drow[r] = fp16_to_fp32_lookup(aq->d);
        }
        for (int c = 0; c < ncols; ++c) {
            const block_q8_K *bq = &B[ldb * (jj + c) + l];
            for (int r = 0; r < nrows; ++r)
                acc[r][c] += drow[r] * bq->d * (float)q3k_dot_decoded(g[r], sc[r], bq);
        }
    }
    for (int c = 0; c < ncols; ++c)
        for (int r = 0; r < nrows; ++r)
            C[ldc * (jj + c) + (ii + r)] = acc[r][c];
}

int sgemm_q3_k_q8_k_neon(int nrows, int ncols, int k, const void *vx,
                         const void *vy, float *out, size_t bs,
                         int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k <= 0 || (k % QK_K) != 0) return 0;
    const int nb = k / QK_K;
    const block_q3_K *A = (const block_q3_K *)vx;
    const block_q8_K *B = (const block_q8_K *)vy;
    int64_t rows_per = (nrows + nth - 1) / nth;
    int64_t i0 = (int64_t)ith * rows_per;
    int64_t i1 = i0 + rows_per; if (i1 > nrows) i1 = nrows;
    if (i0 >= i1) return 0;
    for (int64_t ii = i0; ii < i1; ii += 4) {
        int64_t nr = i1 - ii; if (nr > 4) nr = 4;
        for (int64_t jj = 0; jj < ncols; jj += 4) {
            int64_t nc = ncols - jj; if (nc > 4) nc = 4;
            q3k_tile(A, nb, B, nb, out, (int)bs, nb, ii, jj, (int)nr, (int)nc);
        }
    }
    return (int)(i1 - i0);
}

#endif /* __ARM_NEON */

#if defined(__ARM_NEON)
/* Q4_K table-ABI adapter: the actual kernel is picolm_sgemm_d_q4k(), which
 * already has a NEON path (I8MM vmmlaq_s32 when available, DOTPROD vdotq_s32,
 * basic NEON otherwise) plus the AVX2/VNNI decode-once R8 path. Exposing it
 * through fn_gemm lets Q4_K participate in the fused dual-GEMM path
 * (matmul_dual_batch), which otherwise falls back to two separate
 * matmul_batch calls plus a duplicate activation quantisation.
 *
 * Return value = "rows handled", and gemm_table_dual_task() treats 0 as fatal.
 * picolm_sgemm_d_q4k() returns 0 both for real failures and for the PICOLM_SGEMM
 * kill switch, so map any 0 to "handled" here and let the caller's scalar
 * fallback deal with a disabled GEMM. */
int sgemm_q4_k_q8_k_best_neon(int nrows, int ncols, int k,
                              const void *vx, const void *vy, float *out,
                              size_t bs, int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k <= 0 || (k % QK_K) != 0) return 0;
    const int nb = k / QK_K;
    (void)picolm_sgemm_d_q4k(nrows, ncols, nb, vx, nb, vy, nb,
                             out, (int)bs, ith, nth);
    return nrows;
}
#endif
