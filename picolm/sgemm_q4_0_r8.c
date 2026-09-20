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
    memset(out, 0, nrows * sizeof(float));
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
    memset(out, 0, nrows * sizeof(float));
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
