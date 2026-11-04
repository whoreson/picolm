/* ================================================================
 * Q4_0_R8 x Q8_0 AVX2 GEMV kernel (optimized v3)
 * ================================================================
 * Port of llama.cpp ik branch iqk_gemm_legacy_quants.cpp:
 *   mul_mat_q4_0_r8_q8_2_avx2
 *
 * Weights: block_q4_0x8 (= block_iq4_nl_r8 layout)
 *   d[8] FP16 scales + qs[128] interleaved nibble-bytes
 *   Each block covers 8 rows x 32 values, unsigned nibbles [0..15].
 *
 * Activations: block_q8_0 (plain, one per row)
 *   d FP16 scale + qs[32] signed int8
 *
 * AVX2 + FMA + F16C required.
 *
 * Bias correction: sum((n-8)*q) = sum(n*q) - 8*sum(q).
 *
 * STATUS: WORKING, OPTIMIZED v3
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "quant.h"
#if defined(__AVX2__)

/* External: fp16 lookup table for fast conversion */
/* fp16_to_fp32_lookup declared in quant.h */

static inline void prepare_q4_0_quants_avx2(const uint8_t *qs,
                                              __m256i *v, const __m256i m4) {
    __m256i bits1 = _mm256_loadu_si256((const __m256i *)qs + 0);
    __m256i bits2 = _mm256_loadu_si256((const __m256i *)qs + 1);
    __m256i bits3 = _mm256_loadu_si256((const __m256i *)qs + 2);
    __m256i bits4 = _mm256_loadu_si256((const __m256i *)qs + 3);
    v[0] = _mm256_and_si256(bits1, m4);
    v[1] = _mm256_and_si256(bits2, m4);
    v[2] = _mm256_and_si256(bits3, m4);
    v[3] = _mm256_and_si256(bits4, m4);
    v[4] = _mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4);
    v[5] = _mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4);
    v[6] = _mm256_and_si256(_mm256_srli_epi16(bits3, 4), m4);
    v[7] = _mm256_and_si256(_mm256_srli_epi16(bits4, 4), m4);
}

static inline __m256i accum_q4_0_quants_avx2(const __m256i *qx,
                                               const int8_t *qs) {
    __m128i y4l = _mm_loadu_si128((const __m128i *)qs + 0);
    __m128i y4h = _mm_loadu_si128((const __m128i *)qs + 1);
    __m256i yl = _mm256_broadcastsi128_si256(y4l);
    __m256i yh = _mm256_broadcastsi128_si256(y4h);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    __m256i sumi = _mm256_setzero_si256();
    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(yl, 0x00));
    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(yl, 0x55));
    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(yl, 0xaa));
    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(yl, 0xff));
    sumi = _mm256_dpbusd_epi32(sumi, qx[4], _mm256_shuffle_epi32(yh, 0x00));
    sumi = _mm256_dpbusd_epi32(sumi, qx[5], _mm256_shuffle_epi32(yh, 0x55));
    sumi = _mm256_dpbusd_epi32(sumi, qx[6], _mm256_shuffle_epi32(yh, 0xaa));
    sumi = _mm256_dpbusd_epi32(sumi, qx[7], _mm256_shuffle_epi32(yh, 0xff));
    return sumi;
#else
    __m256i m1 = _mm256_set1_epi16(1);
    __m256i s1 = _mm256_add_epi16(
        _mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(yl, 0x00)),
        _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(yl, 0x55)));
    __m256i s2 = _mm256_add_epi16(
        _mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(yl, 0xaa)),
        _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(yl, 0xff)));
    __m256i s3 = _mm256_add_epi16(
        _mm256_maddubs_epi16(qx[4], _mm256_shuffle_epi32(yh, 0x00)),
        _mm256_maddubs_epi16(qx[5], _mm256_shuffle_epi32(yh, 0x55)));
    __m256i s4 = _mm256_add_epi16(
        _mm256_maddubs_epi16(qx[6], _mm256_shuffle_epi32(yh, 0xaa)),
        _mm256_maddubs_epi16(qx[7], _mm256_shuffle_epi32(yh, 0xff)));
    return _mm256_madd_epi16(m1, _mm256_add_epi16(
        _mm256_add_epi16(s1, s2),
        _mm256_add_epi16(s3, s4)));
#endif
}

void vec_dot_q4_0_r8_q8_0_avx2(const void *vx, const void *wy, int n,
                                 float *out, int nrows) {
#if defined(__AVX2__) && defined(__F16C__)
    assert(nrows == 8);
    assert(n % 32 == 0);

    const block_q4_0x8 *iq4 = (const block_q4_0x8 *)vx;
    const block_q8_0 *qy = (const block_q8_0 *)wy;
    const __m256i m4 = _mm256_set1_epi8(0xf);
    const int nb = n / 32;
    __m256i v[8];
    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ib++) {
        /* Load 8 weight scales (FP16 -> FP32 via hardware) */
        __m256 w_scales = _mm256_cvtph_ps(
            _mm_loadu_si128((const __m128i *)iq4[ib].d));

        /* Activation scale via lookup table (fast) */
        float as = fp16_to_fp32_lookup(qy[ib].d);

        /* Sum of activation qs for bias correction */
        /* Use SSE to sum 32 int8 values faster */
        __m128i t0 = _mm_loadu_si128((const __m128i *)qy[ib].qs + 0);
        __m128i t1 = _mm_loadu_si128((const __m128i *)qy[ib].qs + 1);
        __m128i s0 = _mm_sad_epu8(_mm_setzero_si128(), t0);
        __m128i s1 = _mm_sad_epu8(_mm_setzero_si128(), t1);
        __m128i s_all = _mm_add_epi16(s0, s1);
        /* This gives sum of unsigned bytes, but we need signed.
         * For signed: sum = sum_unsigned - 256 * count_negative
         * But that's complex. Just use scalar for correctness. */
        int asum = 0;
        for (int j = 0; j < 32; j++) asum += (int)(int8_t)qy[ib].qs[j];

        prepare_q4_0_quants_avx2(iq4[ib].qs, v, m4);
        __m256i sumi = accum_q4_0_quants_avx2(v, qy[ib].qs);
        __m256 sumf = _mm256_cvtepi32_ps(sumi);

        /* acc += w_scales * as * sumf - w_scales * as * 8 * asum
         * = w_scales * as * (sumf - 8*asum) */
        __m256 scaled = _mm256_mul_ps(w_scales, _mm256_set1_ps(as));
        __m256 corr = _mm256_fmsub_ps(sumf, _mm256_set1_ps(1.f),
                                       _mm256_set1_ps(8.f * asum));
        /* Actually: sumf - 8*asum per lane, then multiply by scaled */
        __m256 corrected = _mm256_sub_ps(sumf, _mm256_set1_ps(8.f * asum));
        acc = _mm256_fmadd_ps(scaled, corrected, acc);
    }

    _mm256_storeu_ps(out, acc);
#else
    (void)vx; (void)wy; (void)n; (void)out; (void)nrows;
    /* Returning zeros here silently produced a model that decodes to
     * garbage with no error: on a build without the required SIMD the
     * caller has no way to tell "unsupported" from "all weights zero".
     * Refuse instead. */
    fprintf(stderr,
            "FATAL: vec_dot_q4_0_r8_q8_2_avx2 called in a build without the required SIMD.\n"
            "       This type has no scalar fallback; refusing to return zeros.\n");
    abort();
#endif
}

int sgemm_q4_0_r8_q8_0_avx2(int nrows, int ncols, int k,
                             const void *vx, const void *vy,
                             float *out, size_t bs,
                             int ith, int nth) {
#if defined(__AVX2__) && defined(__F16C__)
    if (nrows < 8 || ncols < 1 || k % 32 != 0 || nrows % 8 != 0)
        return 0;

    size_t a_row_bytes = (size_t)(k / 32) * sizeof(block_q8_0);
    int start_col = (ncols * ith) / nth;
    int end_col = (ncols * (ith + 1)) / nth;
    if (end_col <= start_col) return 0;

    size_t w_stride = (size_t)(k / 32) * sizeof(block_q4_0x8);
    for (int w8 = 0; w8 < nrows / 8; w8++) {
        const block_q4_0x8 *iq4_base = (const block_q4_0x8 *)
            ((const char *)vx + w8 * w_stride);
        for (int c = start_col; c < end_col; c++) {
            const block_q8_0 *qy = (const block_q8_0 *)
                ((const char *)vy + c * a_row_bytes);
            /* Output: row-major with stride bs per batch */
            float *out_base = out + w8 * 8 + c * bs;
            vec_dot_q4_0_r8_q8_0_avx2(iq4_base, qy, k, out_base, 8);
        }
    }
    return (nrows / 8) * 8;
#else
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy;
    (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
#endif
}

/* ================================================================
 * Q4_0_R8 x Q8_2 AVX2 kernel
 * Same as Q8_0 version but reads the precomputed activation sum
 * (block_q8_2.s) instead of computing it in a scalar loop.
 * This saves a 32-iteration scalar loop per block.
 * ================================================================ */
void vec_dot_q4_0_r8_q8_2_avx2(const void *vx, const void *wy, int n,
                                 float *out, int nrows) {
#if defined(__AVX2__) && defined(__F16C__)
    assert(nrows == 8);
    assert(n % 32 == 0);

    const block_q4_0x8 *iq4 = (const block_q4_0x8 *)vx;
    const block_q8_2 *qy = (const block_q8_2 *)wy;
    const __m256i m4 = _mm256_set1_epi8(0xf);
    const int nb = n / 32;
    __m256i v[8];
    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ib++) {
        __m256 w_scales = _mm256_cvtph_ps(
            _mm_loadu_si128((const __m128i *)iq4[ib].d));

        float as = fp16_to_fp32_lookup(qy[ib].d);
        int asum = (int)qy[ib].s;  /* precomputed sum of qs */

        prepare_q4_0_quants_avx2(iq4[ib].qs, v, m4);
        __m256i sumi = accum_q4_0_quants_avx2(v, qy[ib].qs);
        __m256 sumf = _mm256_cvtepi32_ps(sumi);

        __m256 scaled = _mm256_mul_ps(w_scales, _mm256_set1_ps(as));
        __m256 corrected = _mm256_sub_ps(sumf, _mm256_set1_ps(8.f * asum));
        acc = _mm256_fmadd_ps(scaled, corrected, acc);
    }

    _mm256_storeu_ps(out, acc);
#else
    (void)vx; (void)wy; (void)n; (void)out; (void)nrows;
    /* Returning zeros here silently produced a model that decodes to
     * garbage with no error: on a build without the required SIMD the
     * caller has no way to tell "unsupported" from "all weights zero".
     * Refuse instead. */
    fprintf(stderr,
            "FATAL: vec_dot_q4_0_r8_q8_2_avx2 called in a build without the required SIMD.\n"
            "       This type has no scalar fallback; refusing to return zeros.\n");
    abort();
#endif
}

int sgemm_q4_0_r8_q8_2_avx2(int nrows, int ncols, int k,
                             const void *vx, const void *vy,
                             float *out, size_t bs,
                             int ith, int nth) {
#if defined(__AVX2__) && defined(__F16C__)
    if (nrows < 8 || ncols < 1 || k % 32 != 0 || nrows % 8 != 0)
        return 0;

    size_t a_row_bytes = (size_t)(k / 32) * sizeof(block_q8_2);
    int start_col = (ncols * ith) / nth;
    int end_col = (ncols * (ith + 1)) / nth;
    if (end_col <= start_col) return 0;

    size_t w_stride = (size_t)(k / 32) * sizeof(block_q4_0x8);
    for (int w8 = 0; w8 < nrows / 8; w8++) {
        const block_q4_0x8 *iq4_base = (const block_q4_0x8 *)
            ((const char *)vx + w8 * w_stride);
        for (int c = start_col; c < end_col; c++) {
            const block_q8_2 *qy = (const block_q8_2 *)
                ((const char *)vy + c * a_row_bytes);
            float *out_base = out + w8 * 8 + c * bs;
            vec_dot_q4_0_r8_q8_2_avx2(iq4_base, qy, k, out_base, 8);
        }
    }
    return (nrows / 8) * 8;
#else
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy;
    (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
#endif
}
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
/* ================================================================
 * ARM NEON (DOTPROD) implementations for Q4_0_R8.
 *
 * Q4_0_R8 layout per block (block_q4_0x8, 144 bytes):
 *   d[8]     FP16 scales (one per row)
 *   qs[128]  interleaved nibbles: qs[32*l + 4*r + j] holds row r,
 *            value 4*l+j in the LOW nibble and value 16+4*l+j in the
 *            HIGH nibble (both biased by +8).
 *
 * NEON DOTPROD GEMV (8 rows at a time):
 *   vld1q_u8(qs + 32*l) loads rows 0-3's 4 bytes for chunk l.
 *   After nibble expansion, byte position 4*r+j = row r's value 4*l+j.
 *   vdotq_s32 lane r = sum(j) w[4r+j]*a[j], so broadcasting the
 *   4 activation bytes a[4l..4l+3] into all 4 lane groups gives
 *   lane r = row r's partial dot for this chunk.
 *   4 chunks x (lo+hi) nibbles x 2 row groups = 8 vdotq_s32 per block.
 *
 * Bias correction: stored nibble = value + 8, so the DOTPROD computes
 *   sum((value+8)*a) = sum(value*a) + 8*sum(a).
 *   8*sum(a) is row-independent; per-row scale is d[row]*act_scale.
 *   out[row] = (biased_dot - 8*asum) * d[row] * act_scale.
 * ================================================================ */
#include <arm_neon.h>

/* Broadcast 4 consecutive int8 values to all 4 DOTPROD lane groups. */
static inline int8x16_t q4r8_bcast4_neon(const int8_t *a) {
    uint32_t u;
    memcpy(&u, a, 4);
    return vreinterpretq_s8_u32(vdupq_n_u32(u));
}

/* GEMV: 8 weight rows x 1 activation row (block_q8_0). */
static void q4r8_gemv_q8_0_neon(const block_q4_0x8 *xb, const block_q8_0 *y,
                                int nb, float *out) {
    float acc[8];
    for (int r = 0; r < 8; r++) acc[r] = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        const uint8_t *qs = xb[ib].qs;
        const int8_t *ya = y[ib].qs;

        int32x4_t blk_lo03 = vdupq_n_s32(0);
        int32x4_t blk_hi03 = vdupq_n_s32(0);
        int32x4_t blk_lo47 = vdupq_n_s32(0);
        int32x4_t blk_hi47 = vdupq_n_s32(0);

        for (int l = 0; l < 4; l++) {
            uint8x16_t v03 = vld1q_u8(qs + 32 * l);        /* rows 0-3 */
            uint8x16_t v47 = vld1q_u8(qs + 32 * l + 16);   /* rows 4-7 */

            int8x16_t lo03 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v03, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
            int8x16_t hi03 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v03, 4)), vdupq_n_s8(8));
            int8x16_t lo47 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v47, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
            int8x16_t hi47 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v47, 4)), vdupq_n_s8(8));

            int8x16_t abl = q4r8_bcast4_neon(ya + 4 * l);
            int8x16_t abh = q4r8_bcast4_neon(ya + 16 + 4 * l);

            blk_lo03 = vdotq_s32(blk_lo03, lo03, abl);
            blk_hi03 = vdotq_s32(blk_hi03, hi03, abh);
            blk_lo47 = vdotq_s32(blk_lo47, lo47, abl);
            blk_hi47 = vdotq_s32(blk_hi47, hi47, abh);
        }

        /* Nibbles already bias-corrected (-8) during expansion: true dot. */
        int32x4_t dot03 = vaddq_s32(blk_lo03, blk_hi03);
        int32x4_t dot47 = vaddq_s32(blk_lo47, blk_hi47);

        float as = fp16_to_fp32_lookup(y[ib].d);
        float16x8_t d8 = vld1q_f16((const float16_t*)xb[ib].d);
        float32x4_t sc03 = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(d8)), as);
        float32x4_t sc47 = vmulq_n_f32(vcvt_f32_f16(vget_high_f16(d8)), as);

        float32x4_t f03 = vmulq_f32(vcvtq_f32_s32(dot03), sc03);
        float32x4_t f47 = vmulq_f32(vcvtq_f32_s32(dot47), sc47);
        vst1q_f32(acc + 0, vaddq_f32(vld1q_f32(acc + 0), f03));
        vst1q_f32(acc + 4, vaddq_f32(vld1q_f32(acc + 4), f47));
    }
    memcpy(out, acc, 8 * sizeof(float));
}

/* GEMV: 8 weight rows x 1 activation row (block_q8_2, int16 precomputed sum). */
static void q4r8_gemv_q8_2_neon(const block_q4_0x8 *xb, const block_q8_2 *y,
                                int nb, float *out) {
    float acc[8];
    for (int r = 0; r < 8; r++) acc[r] = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        const uint8_t *qs = xb[ib].qs;
        const int8_t *ya = y[ib].qs;

        int32x4_t blk_lo03 = vdupq_n_s32(0);
        int32x4_t blk_hi03 = vdupq_n_s32(0);
        int32x4_t blk_lo47 = vdupq_n_s32(0);
        int32x4_t blk_hi47 = vdupq_n_s32(0);

        for (int l = 0; l < 4; l++) {
            uint8x16_t v03 = vld1q_u8(qs + 32 * l);
            uint8x16_t v47 = vld1q_u8(qs + 32 * l + 16);

            int8x16_t lo03 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v03, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
            int8x16_t hi03 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v03, 4)), vdupq_n_s8(8));
            int8x16_t lo47 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v47, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
            int8x16_t hi47 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v47, 4)), vdupq_n_s8(8));

            int8x16_t abl = q4r8_bcast4_neon(ya + 4 * l);
            int8x16_t abh = q4r8_bcast4_neon(ya + 16 + 4 * l);

            blk_lo03 = vdotq_s32(blk_lo03, lo03, abl);
            blk_hi03 = vdotq_s32(blk_hi03, hi03, abh);
            blk_lo47 = vdotq_s32(blk_lo47, lo47, abl);
            blk_hi47 = vdotq_s32(blk_hi47, hi47, abh);
        }

        /* Nibbles already bias-corrected (-8) during expansion: true dot. */
        int32x4_t dot03 = vaddq_s32(blk_lo03, blk_hi03);
        int32x4_t dot47 = vaddq_s32(blk_lo47, blk_hi47);
        (void)y[ib].s; /* precomputed sum not needed (bias pre-corrected) */

        float as = fp16_to_fp32_lookup(y[ib].d);
        float16x8_t d8 = vld1q_f16((const float16_t*)xb[ib].d);
        float32x4_t sc03 = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(d8)), as);
        float32x4_t sc47 = vmulq_n_f32(vcvt_f32_f16(vget_high_f16(d8)), as);

        float32x4_t f03 = vmulq_f32(vcvtq_f32_s32(dot03), sc03);
        float32x4_t f47 = vmulq_f32(vcvtq_f32_s32(dot47), sc47);
        vst1q_f32(acc + 0, vaddq_f32(vld1q_f32(acc + 0), f03));
        vst1q_f32(acc + 4, vaddq_f32(vld1q_f32(acc + 4), f47));
    }
    memcpy(out, acc, 8 * sizeof(float));
}

void vec_dot_q4_0_r8_q8_0_avx2(const void *vx, const void *wy, int n,
                                float *out, int nrows) {
    const block_q4_0x8 *xb = (const block_q4_0x8 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)wy;
    int nb = n / 32;
    int groups = nrows / 8;
    for (int g = 0; g < groups; g++)
        q4r8_gemv_q8_0_neon(xb + (size_t)g * nb, y, nb, out + g * 8);
    for (int r = groups * 8; r < nrows; r++) {
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++) {
            const uint8_t *qs = xb[ib].qs;
            const float d = fp16_to_fp32_lookup(xb[ib].d[r]) * fp16_to_fp32_lookup(y[ib].d);
            const int8_t *ya = y[ib].qs;
            int sumi = 0;
            for (int l = 0; l < 4; l++) {
                const uint8_t *q = qs + 32 * l + 4 * r;
                for (int j = 0; j < 4; j++) {
                    sumi += ((int)(q[j] & 0xF) - 8) * (int)ya[4 * l + j];
                    sumi += ((int)(q[j] >> 4) - 8) * (int)ya[4 * l + j + 16];
                }
            }
            sumf += (float)sumi * d;
        }
        out[r] = sumf;
    }
}

void vec_dot_q4_0_r8_q8_2_avx2(const void *vx, const void *wy, int n,
                                float *out, int nrows) {
    const block_q4_0x8 *xb = (const block_q4_0x8 *)vx;
    const block_q8_2 *y = (const block_q8_2 *)wy;
    int nb = n / 32;
    int groups = nrows / 8;
    for (int g = 0; g < groups; g++)
        q4r8_gemv_q8_2_neon(xb + (size_t)g * nb, y, nb, out + g * 8);
    for (int r = groups * 8; r < nrows; r++) {
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++) {
            const uint8_t *qs = xb[ib].qs;
            const float d = fp16_to_fp32_lookup(xb[ib].d[r]) * fp16_to_fp32_lookup(y[ib].d);
            const int8_t *ya = y[ib].qs;
            int sumi = 0;
            for (int l = 0; l < 4; l++) {
                const uint8_t *q = qs + 32 * l + 4 * r;
                for (int j = 0; j < 4; j++) {
                    sumi += ((int)(q[j] & 0xF) - 8) * (int)ya[4 * l + j];
                    sumi += ((int)(q[j] >> 4) - 8) * (int)ya[4 * l + j + 16];
                }
            }
            sumf += (float)sumi * d;
        }
        out[r] = sumf;
    }
}

int sgemm_q4_0_r8_q8_0_avx2(int nrows, int ncols, int k, const void *vx, const void *vy,
                             float *out, size_t bs, int ith, int nth) {
    if (nrows < 8 || ncols < 1 || k % 32 != 0 || nrows % 8 != 0) return 0;
    size_t a_row_bytes = (size_t)(k / 32) * sizeof(block_q8_0);
    size_t w_stride = (size_t)(k / 32) * sizeof(block_q4_0x8);
    int start_col = (ncols * ith) / nth;
    int end_col = (ncols * (ith + 1)) / nth;
    if (end_col <= start_col) return 0;
    for (int c = start_col; c < end_col; c++) {
        const block_q8_0 *qy = (const block_q8_0 *)((const char *)vy + (size_t)c * a_row_bytes);
        for (int w8 = 0; w8 < nrows / 8; w8++) {
            const block_q4_0x8 *iq4 = (const block_q4_0x8 *)((const char *)vx + (size_t)w8 * w_stride);
            vec_dot_q4_0_r8_q8_0_avx2(iq4, qy, k, out + w8 * 8 + (size_t)c * bs, 8);
        }
    }
    return (nrows / 8) * 8;
}

int sgemm_q4_0_r8_q8_2_avx2(int nrows, int ncols, int k, const void *vx, const void *vy,
                             float *out, size_t bs, int ith, int nth) {
    if (nrows < 8 || ncols < 1 || k % 32 != 0 || nrows % 8 != 0) return 0;
    size_t a_row_bytes = (size_t)(k / 32) * sizeof(block_q8_2);
    size_t w_stride = (size_t)(k / 32) * sizeof(block_q4_0x8);
    int start_col = (ncols * ith) / nth;
    int end_col = (ncols * (ith + 1)) / nth;
    if (end_col <= start_col) return 0;
    for (int c = start_col; c < end_col; c++) {
        const block_q8_2 *qy = (const block_q8_2 *)((const char *)vy + (size_t)c * a_row_bytes);
        for (int w8 = 0; w8 < nrows / 8; w8++) {
            const block_q4_0x8 *iq4 = (const block_q4_0x8 *)((const char *)vx + (size_t)w8 * w_stride);
            vec_dot_q4_0_r8_q8_2_avx2(iq4, qy, k, out + w8 * 8 + (size_t)c * bs, 8);
        }
    }
    return (nrows / 8) * 8;
}
#else
/* No AVX2, no NEON DOTPROD: portable scalar GEMV + stub GEMMs.
 *
 * These used to be empty stubs that returned nothing, leaving `out` untouched.
 * A model containing a Q4_0_R8 tensor therefore decoded to garbage on any
 * build without AVX2 -- with no error, no warning, and output that still looks
 * like text. Found by diffing single-type models between the SIMD and scalar
 * builds: exactly the two R8 types differed.
 *
 * block_q4_0x8 is 8 rows x 32 values: {fp16 d[8]; u8 qs[128]}. Row r's nibble
 * bytes are qs[32*l + 4*r + j] for l = 0..3, j = 0..3; the low nibble is value
 * 4*l+j and the high nibble value 4*l+j+16, both biased by 8. Mirrors
 * dequantize_row_q4_0_r8_single.
 *
 * Integer MAC then one float scale, in the style of vec_dot_q5_0_q8_0: the
 * point is that it is correct everywhere, not fast anywhere.
 */
#define Q4R8_PORTABLE(NAME, TYPE_Y)                                              \
void NAME(const void *vx, const void *wy, int n, float *out, int nrows) {        \
    const block_q4_0x8 *xb = (const block_q4_0x8 *)vx;                           \
    const TYPE_Y *y = (const TYPE_Y *)wy;                                        \
    const int nb = n / 32;                                                       \
    for (int row = 0; row < nrows; row++) {                                      \
        float sumf = 0.0f;                                                       \
        for (int ib = 0; ib < nb; ib++) {                                        \
            const float d = fp16_to_fp32(xb[ib].d[row]) * fp16_to_fp32(y[ib].d); \
            const uint8_t *qs = xb[ib].qs;                                       \
            const int8_t *ya = y[ib].qs;                                         \
            int sumi = 0;                                                        \
            for (int l = 0; l < 4; l++) {                                        \
                const uint8_t *q = qs + 32 * l + 4 * row;                        \
                for (int j = 0; j < 4; j++) {                                    \
                    sumi += ((int)(q[j] & 0xF) - 8) * (int)ya[4 * l + j];        \
                    sumi += ((int)(q[j] >> 4)  - 8) * (int)ya[4 * l + j + 16];   \
                }                                                                \
            }                                                                    \
            sumf += (float)sumi * d;                                             \
        }                                                                        \
        out[row] = sumf;                                                         \
    }                                                                            \
}

Q4R8_PORTABLE(vec_dot_q4_0_r8_q8_0_avx2, block_q8_0)
Q4R8_PORTABLE(vec_dot_q4_0_r8_q8_2_avx2, block_q8_2)
#undef Q4R8_PORTABLE

int sgemm_q4_0_r8_q8_0_avx2(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)vx; (void)vy; (void)nrows; (void)ncols; (void)k; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q4_0_r8_q8_2_avx2(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)vx; (void)vy; (void)nrows; (void)ncols; (void)k; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
#endif
