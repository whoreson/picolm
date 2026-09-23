/* ================================================================
 * IQ2_K_R4 x Q8_K AVX2/AVX-512 GEMV and GEMM kernels
 * ================================================================
 * Port of llama.cpp ik branch iqk_gemm_iqk_quants.cpp:
 *   mul_mat_iq2_k_r4_q8_k
 *
 * Weights: block_iq2_k_r4 (GGUF type 337)
 * Activations: block_q8_K (one per row)
 *
 * Algorithm: LUT-based dequantization via iq2nl_values table,
 * then SIGNED*SIGNED dot product with Q8_K activations.
 *
 * Sign trick: dpbusd/maddubs do UNSIGNED*SIGNED. For signed
 * multiplication, we use |qx| for the first operand and
 * sign_extend(qx, y) for the second:
 *   s = sign_epi8(qx, qx)     -> |qx|
 *   sy = sign_epi8(y_shuf, qx) -> y with sign of qx
 *   dpbusd(s, sy) = |qx| * (y * sign(qx)) = qx * y
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

#define QK_K 256

static inline __m256i load_scales_avx2(const uint64_t *stored_scales, int ib) {
    /* Load 8 bytes from stored_scales[ib]. Sign-extend to 8 int16,
     * then extract the low 4 int16 (rows 0..3) and extend to 4 int32. */
    __m128i s128 = _mm_loadl_epi64((const __m128i *)(stored_scales + ib));
    __m128i s16 = _mm_cvtepi8_epi16(s128);  /* 8 int8 -> 8 int16 */
    __m256i i32 = _mm256_cvtepi16_epi32(s16);  /* low 4 int16 -> 4 int32, upper 4 zero */
    return i32;
}

void vec_dot_iq2_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
#if defined(__AVX2__) && defined(__F16C__)
    assert(nrows == 4);
    assert(n % QK_K == 0);

    const block_iq2_k_r4 *iq2 = (const block_iq2_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    static const int8_t kvalues_iq2nl[32] = {
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
    };
    const __m256i values = _mm256_loadu_si256((const __m256i *)kvalues_iq2nl);
    const __m256i shift_shuffle = _mm256_set_epi64x(
        0x0707070706060606ULL, 0x0505050504040404ULL,
        0x0303030302020202ULL, 0x0101010100000000ULL);
    const __m256i m4 = _mm256_set1_epi8(0xf);
    const __m256i ms = _mm256_set1_epi8(4);
    const __m256i m03 = _mm256_set1_epi8(0x03);
    const __m256i m16 = _mm256_set1_epi16(1);

    __m256 acc = _mm256_setzero_ps();
    uint64_t stored_scales[8];

    for (int ibl = 0; ibl < nb; ibl++) {
        __m128 dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq2[ibl].d));
        __m256 d4 = _mm256_set_m128(dl, dl);
        float q8_scale = qk[ibl].d;
        __m256 d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8_scale));

        __m256i extra = _mm256_set1_epi64x(*(const uint64_t *)iq2[ibl].extra);
        __m256i slbits = _mm256_loadu_si256((const __m256i *)iq2[ibl].scales);
        __m256i i8scales1 = _mm256_add_epi8(_mm256_and_si256(slbits, m4), _mm256_set1_epi8(-8));
        __m256i i8scales2 = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4), _mm256_set1_epi8(-8));
        _mm256_storeu_si256((__m256i *)stored_scales + 0, i8scales1);
        _mm256_storeu_si256((__m256i *)stored_scales + 1, i8scales2);

        /* Bias correction for sign trick: scale * bsum * min_value */
        __m256i isum = _mm256_setzero_si256(); /* DISABLED: iq234_k_accum_mins_vecdot(i8scales1, i8scales2, qk + ibl, &isum, -32); */

        for (int ib = 0; ib < QK_K / 32; ib++) {
            __m256i scales = load_scales_avx2(stored_scales, ib);

            __m256i lb = _mm256_loadu_si256((const __m256i *)iq2[ibl].qs + ib);
            __m256i shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra, 2));
            extra = _mm256_srli_epi16(extra, 1);
            shift = _mm256_shuffle_epi8(shift, shift_shuffle);

            __m256i qx[4];
            qx[0] = _mm256_and_si256(lb, m03);
            qx[1] = _mm256_and_si256(_mm256_srli_epi16(lb, 2), m03);
            qx[2] = _mm256_and_si256(_mm256_srli_epi16(lb, 4), m03);
            qx[3] = _mm256_and_si256(_mm256_srli_epi16(lb, 6), m03);
            qx[0] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[0], shift));
            qx[1] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[1], shift));
            qx[2] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[2], shift));
            qx[3] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[3], shift));

            __m256i y_reg = _mm256_loadu_si256((const __m256i *)qk[ibl].qs + ib);

            /* Sign trick: |qx| for unsigned multiply, sign(qx) applied to y */
            __m256i s0 = _mm256_sign_epi8(qx[0], qx[0]);
            __m256i s1 = _mm256_sign_epi8(qx[1], qx[1]);
            __m256i s2 = _mm256_sign_epi8(qx[2], qx[2]);
            __m256i s3 = _mm256_sign_epi8(qx[3], qx[3]);

#ifdef __AVX512VNNI__
            __m256i sumi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x00), qx[0]));
            sumi = _mm256_dpbusd_epi32(sumi, s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x55), qx[1]));
            sumi = _mm256_dpbusd_epi32(sumi, s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xaa), qx[2]));
            sumi = _mm256_dpbusd_epi32(sumi, s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xff), qx[3]));
#else
            __m256i t1 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x00), qx[0])));
            __m256i t2 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x55), qx[1])));
            __m256i t3 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xaa), qx[2])));
            __m256i t4 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xff), qx[3])));
            __m256i sumi = _mm256_add_epi32(_mm256_add_epi32(t1, t2), _mm256_add_epi32(t3, t4));
#endif

            isum = _mm256_add_epi32(isum, _mm256_mullo_epi32(scales, sumi));
        }

        acc = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum), acc);
    }

    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    _mm_storeu_ps(out, sum);
#else
    (void)vx; (void)wy; (void)n; (void)out; (void)nrows;
    memset(out, 0, nrows * sizeof(float));
#endif
}

int sgemm_iq2_k_r4_q8_k_avx2(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth) {
#if defined(__AVX2__) && defined(__F16C__)
    if (nrows < 4 || ncols < 1 || k % QK_K != 0 || nrows % 4 != 0)
        return 0;

    const int nb = k / QK_K;

    static const int8_t kvalues_iq2nl[32] = {
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
    };
    const __m256i values = _mm256_loadu_si256((const __m256i *)kvalues_iq2nl);
    const __m256i shift_shuffle = _mm256_set_epi64x(
        0x0707070706060606ULL, 0x0505050504040404ULL,
        0x0303030302020202ULL, 0x0101010100000000ULL);
    const __m256i m4 = _mm256_set1_epi8(0xf);
    const __m256i ms = _mm256_set1_epi8(4);
    const __m256i m03 = _mm256_set1_epi8(0x03);
    const __m256i m16 = _mm256_set1_epi16(1);

    {
        int64_t ytiles = nrows / 4;
        int64_t xtiles = ncols / 2;
        int64_t n_tail = ncols - xtiles * 2;
        int64_t xtiles_ext = xtiles + (n_tail > 0 ? 1 : 0);
        int64_t tiles = ytiles * xtiles_ext;
        if (tiles <= 0) return 0;

        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles) end = tiles;

        size_t w_block_bytes = nb * sizeof(block_iq2_k_r4);
        size_t a_row_bytes = nb * sizeof(block_q8_K);

        for (int64_t job = start; job < end; job++) {
            int64_t ii = (job / xtiles_ext) * 4;
            int64_t xt = job % xtiles_ext;
            int64_t jj = xt * 2;
            int64_t ncols_tile = (xt < xtiles) ? 2 : n_tail;
            if (ncols_tile < 1) ncols_tile = 1;

            const block_iq2_k_r4 *iq2 = (const block_iq2_k_r4 *)
                ((const char *)vx + (ii / 4) * w_block_bytes);

            __m256 acc[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };

            const block_q8_K *qk_ptr[2];
            for (int c = 0; c < ncols_tile; c++) {
                qk_ptr[c] = (const block_q8_K *)
                    ((const char *)vy + (jj + c) * a_row_bytes);
            }

            for (int ibl = 0; ibl < nb; ibl++) {
                __m128 dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq2[ibl].d));
                __m256 d4 = _mm256_set_m128(dl, dl);
                __m256i extra = _mm256_set1_epi64x(*(const uint64_t *)iq2[ibl].extra);

                __m256i slbits = _mm256_loadu_si256((const __m256i *)iq2[ibl].scales);
                __m256i i8scales1 = _mm256_add_epi8(_mm256_and_si256(slbits, m4), _mm256_set1_epi8(-8));
                __m256i i8scales2 = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4), _mm256_set1_epi8(-8));
                uint64_t stored_scales[8];
                _mm256_storeu_si256((__m256i *)stored_scales + 0, i8scales1);
                _mm256_storeu_si256((__m256i *)stored_scales + 1, i8scales2);

                for (int c = 0; c < ncols_tile; c++) {
                    float q8_scale = qk_ptr[c][ibl].d;
                    __m256 d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8_scale));
                    /* Bias correction for sign trick: DISABLED */
                    __m256i isum = _mm256_setzero_si256();

                    /* Reload extra for each column (weights are shared, extra is per-block) */
                    __m256i extra_c = extra;

                    for (int ib = 0; ib < QK_K / 32; ib++) {
                        __m256i scales = load_scales_avx2(stored_scales, ib);

                        __m256i lb = _mm256_loadu_si256((const __m256i *)iq2[ibl].qs + ib);
                        __m256i shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra_c, 2));
                        extra_c = _mm256_srli_epi16(extra_c, 1);
                        shift = _mm256_shuffle_epi8(shift, shift_shuffle);

                        __m256i qx[4];
                        qx[0] = _mm256_and_si256(lb, m03);
                        qx[1] = _mm256_and_si256(_mm256_srli_epi16(lb, 2), m03);
                        qx[2] = _mm256_and_si256(_mm256_srli_epi16(lb, 4), m03);
                        qx[3] = _mm256_and_si256(_mm256_srli_epi16(lb, 6), m03);
                        qx[0] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[0], shift));
                        qx[1] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[1], shift));
                        qx[2] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[2], shift));
                        qx[3] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[3], shift));

                        __m256i y_reg = _mm256_loadu_si256((const __m256i *)qk_ptr[c][ibl].qs + ib);

                        __m256i s0 = _mm256_sign_epi8(qx[0], qx[0]);
                        __m256i s1 = _mm256_sign_epi8(qx[1], qx[1]);
                        __m256i s2 = _mm256_sign_epi8(qx[2], qx[2]);
                        __m256i s3 = _mm256_sign_epi8(qx[3], qx[3]);

#ifdef __AVX512VNNI__
                        __m256i sumi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x00), qx[0]));
                        sumi = _mm256_dpbusd_epi32(sumi, s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x55), qx[1]));
                        sumi = _mm256_dpbusd_epi32(sumi, s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xaa), qx[2]));
                        sumi = _mm256_dpbusd_epi32(sumi, s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xff), qx[3]));
#else
                        __m256i t1 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x00), qx[0])));
                        __m256i t2 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x55), qx[1])));
                        __m256i t3 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xaa), qx[2])));
                        __m256i t4 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xff), qx[3])));
                        __m256i sumi = _mm256_add_epi32(_mm256_add_epi32(t1, t2), _mm256_add_epi32(t3, t4));
#endif

                        isum = _mm256_add_epi32(isum, _mm256_mullo_epi32(scales, sumi));
                    }

                    acc[c] = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum), acc[c]);
                }
            }

            for (int c = 0; c < ncols_tile; c++) {
                __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc[c]),
                                        _mm256_extractf128_ps(acc[c], 1));
                float *out_c = out + ii + (jj + c) * bs;
                _mm_storeu_ps(out_c, sum);
            }
        }
    }
    return nrows;
#else
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy;
    (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
#endif
}
