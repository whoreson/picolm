/* ================================================================
 * IQ3_K (plain, GGUF type 138) x Q8_K AVX2 GEMV and GEMM kernels
 * ================================================================
 * Port of llama.cpp iqk_gemm_iqk_quants.cpp dequantization logic
 * adapted for single-row (non-interleaved) IQ3_K format.
 *
 * Weights: block_iq3_k (GGUF type 138), 110 bytes/block
 * Activations: block_q8_K
 *
 * Layout per block (256 values):
 *   d:        FP16 global scale
 *   extra:    16 bits, 2 bits per 32-value subblock (LUT table select)
 *   scales_h: 16 bits, sign bits for 8 scale pairs
 *   scales_l: 8 bytes, 4-bit magnitude per scale
 *   qs:       64 bytes, 2-bit low values (4 per byte)
 *   qh:       32 bytes, 1 high bit per value
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

#define QK_K 256

/* Helper: reduce 8 int32 in a __m256i to a single int32 sum */
static inline int hsum_i32x8(__m256i v) {
    __m256i s1 = _mm256_shuffle_epi32(v, 0x1B);
    __m256i a1 = _mm256_add_epi32(v, s1);
    s1 = _mm256_shuffle_epi32(a1, 0x31);
    a1 = _mm256_add_epi32(a1, s1);
    __m128i a_lo = _mm256_castsi256_si128(a1);
    __m128i a_hi = _mm256_extracti128_si256(a1, 1);
    return _mm_cvtsi128_si32(_mm_add_epi32(a_lo, a_hi));
}

void vec_dot_iq3_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
#if defined(__AVX2__) && defined(__F16C__)
    assert(n % QK_K == 0);

    const block_iq3_k *iq3 = (const block_iq3_k *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    /* iq3nl_values LUT: 16 entries (8 normal + 8 shifted), broadcast to both lanes */
    static const int8_t kvalues_iq3nl[32] = {
        -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
        -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
    };
    const __m256i values = _mm256_loadu_si256((const __m256i *)kvalues_iq3nl);
    const __m256i m03 = _mm256_set1_epi8(0x03);

    float result = 0.0f;

    for (int ibl = 0; ibl < nb; ibl++) {
        float d = fp16_to_fp32_lookup(iq3[ibl].d);
        float q8_scale = qk[ibl].d;
        float dy = d * q8_scale;

        uint16_t sh = iq3[ibl].scales_h;
        uint16_t extra = iq3[ibl].extra;
        const uint8_t *qs = iq3[ibl].qs;
        const uint8_t *qh = iq3[ibl].qh;
        const int8_t *q8 = qk[ibl].qs;

        __m256i isum = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            /* Scale extraction: magnitude from scales_l, sign from scales_h */
            uint8_t mag_lo = iq3[ibl].scales_l[ib32] & 0xf;
            uint8_t mag_hi = iq3[ibl].scales_l[ib32] >> 4;
            int scale_lo = (int)(mag_lo * 2 + 1) * ((sh & 1) ? 1 : -1);
            int scale_hi = (int)(mag_hi * 2 + 1) * ((sh & 2) ? 1 : -1);
            sh >>= 2;

            __m256i scales = _mm256_set_epi32(
                scale_hi, scale_hi, scale_hi, scale_hi,
                scale_lo, scale_lo, scale_lo, scale_lo);

            /* LUT table selection: 8 normal + 8 shifted */
            int shift_idx  = ((extra >> 0) & 1) * 8;
            int shift_idx2 = ((extra >> 1) & 1) * 8;
            extra >>= 2;

            int shift_l = 2 * (ib32 % 4);   /* 0, 2, 4, 6, 0, 2, 4, 6 */
            int shift_h = ib32 % 8;         /* 0, 1, 2, 3, 4, 5, 6, 7 */
            int qs_off  = (ib32 / 4) * 32;   /* 0 or 32 */

            /* Load qs low bytes (j=0..15) and high bytes (j=16..31) */
            __m128i lb_lo = _mm_loadu_si128((const __m128i *)(qs + qs_off + 0));
            __m128i lb_hi = _mm_loadu_si128((const __m128i *)(qs + qs_off + 16));

            /* Load qh bytes for high bit extraction.
             * qh has 32 bytes per block. For each subblock, qh[j] and qh[j+16]
             * with shift_h = ib32%8. Each qh byte provides 8 high bits
             * (one per subblock) for the same j position. */
            __m128i hb = _mm_loadu_si128((const __m128i *)(qh));

            /* Broadcast to 256-bit */
            __m256i lb_lo256 = _mm256_broadcastsi128_si256(lb_lo);
            __m256i lb_hi256 = _mm256_broadcastsi128_si256(lb_hi);
            __m256i hb256 = _mm256_broadcastsi128_si256(hb);

            __m256i y_reg = _mm256_loadu_si256((const __m256i *)(q8 + ib32 * 32));

            /* Extract low 2 bits */
            __m256i ql_lo, ql_hi;
            if (shift_l == 0) {
                ql_lo = _mm256_and_si256(lb_lo256, m03);
                ql_hi = _mm256_and_si256(lb_hi256, m03);
            } else if (shift_l == 2) {
                ql_lo = _mm256_and_si256(_mm256_srli_epi16(lb_lo256, 2), m03);
                ql_hi = _mm256_and_si256(_mm256_srli_epi16(lb_hi256, 2), m03);
            } else if (shift_l == 4) {
                ql_lo = _mm256_and_si256(_mm256_srli_epi16(lb_lo256, 4), m03);
                ql_hi = _mm256_and_si256(_mm256_srli_epi16(lb_hi256, 4), m03);
            } else {
                ql_lo = _mm256_and_si256(_mm256_srli_epi16(lb_lo256, 6), m03);
                ql_hi = _mm256_and_si256(_mm256_srli_epi16(lb_hi256, 6), m03);
            }

            /* Extract high bit from qh: (qh[j] >> shift_h) & 1, then << 2 */
            __m256i qh_shifted = _mm256_srli_epi16(hb256, shift_h);
            qh_shifted = _mm256_and_si256(qh_shifted, m03);  /* mask to 0 or 1 */
            qh_shifted = _mm256_slli_epi16(qh_shifted, 2);   /* shift to 0 or 4 */

            /* Combine low 2 bits + high 1 bit to get 3-bit index (0-7) */
            __m256i qx_lo = _mm256_or_si256(ql_lo, qh_shifted);
            __m256i qx_hi = _mm256_or_si256(ql_hi, qh_shifted);

            /* LUT lookup: 3-bit index -> int8 dequantized value */
            qx_lo = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx_lo, _mm256_set1_epi8(shift_idx)));
            qx_hi = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx_hi, _mm256_set1_epi8(shift_idx2)));

            /* Sign trick for signed*signed via maddubs */
            __m256i s_lo = _mm256_sign_epi8(qx_lo, qx_lo);
            __m256i s_hi = _mm256_sign_epi8(qx_hi, qx_hi);

            /* Split y_reg: low 128 bits for lo, high 128 bits for hi */
            __m256i y_lo = y_reg;
            __m256i y_hi = _mm256_castsi128_si256(_mm256_extracti128_si256(y_reg, 1));

            __m256i sumi;
#ifdef __AVX512VNNI__
            __m256i acc_lo = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                s_lo, _mm256_sign_epi8(y_lo, qx_lo));
            __m256i acc_hi = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                s_hi, _mm256_sign_epi8(y_hi, qx_hi));
            sumi = _mm256_add_epi32(acc_lo, acc_hi);
#else
            __m128i mad_lo = _mm256_castsi256_si128(
                _mm256_maddubs_epi16(s_lo, _mm256_sign_epi8(y_lo, qx_lo)));
            __m128i mad_hi = _mm256_castsi256_si128(
                _mm256_maddubs_epi16(s_hi, _mm256_sign_epi8(y_hi, qx_hi)));
            __m128i t1 = _mm_madd_epi16(_mm_set1_epi16(1), mad_lo);
            __m128i t2 = _mm_madd_epi16(_mm_set1_epi16(1), mad_hi);
            sumi = _mm256_insertf128_si256(_mm256_castsi128_si256(t1), t2, 1);
#endif

            isum = _mm256_add_epi32(isum, _mm256_mullo_epi32(scales, sumi));
            q8 += 32;
        }

        int total = hsum_i32x8(isum);
        result += (float)total * dy;
    }

    *out = result;
#else
    (void)vx; (void)wy; (void)n; (void)out;
    *out = 0.0f;
#endif
}

int sgemm_iq3_k_q8_k_avx2(int nrows, int ncols, int k,
                           const void *vx, const void *vy,
                           float *out, size_t bs,
                           int ith, int nth) {
#if defined(__AVX2__) && defined(__F16C__)
    if (nrows < 1 || ncols < 1 || k % QK_K != 0)
        return 0;

    const int nb = k / QK_K;

    static const int8_t kvalues_iq3nl[32] = {
        -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
        -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
    };
    const __m256i values = _mm256_loadu_si256((const __m256i *)kvalues_iq3nl);
    const __m256i m03 = _mm256_set1_epi8(0x03);

    int64_t ytiles = nrows;
    int64_t xtiles = ncols / 2;
    int64_t n_tail = ncols - xtiles * 2;
    int64_t xtiles_ext = xtiles + (n_tail > 0 ? 1 : 0);
    int64_t tiles = ytiles * xtiles_ext;
    if (tiles <= 0) return 0;

    int64_t duty = (tiles + nth - 1) / nth;
    int64_t start = duty * ith;
    int64_t end = start + duty;
    if (end > tiles) end = tiles;

    size_t w_row_bytes = nb * sizeof(block_iq3_k);
    size_t a_row_bytes = nb * sizeof(block_q8_K);

    for (int64_t job = start; job < end; job++) {
        int64_t ii = job / xtiles_ext;
        int64_t xt = job % xtiles_ext;
        int64_t jj = xt * 2;
        int64_t ncols_tile = (xt < xtiles) ? 2 : n_tail;
        if (ncols_tile < 1) ncols_tile = 1;

        const block_iq3_k *iq3 = (const block_iq3_k *)
            ((const char *)vx + ii * w_row_bytes);

        float acc[2] = { 0.0f, 0.0f };

        const block_q8_K *qk_ptr[2];
        for (int c = 0; c < ncols_tile; c++) {
            qk_ptr[c] = (const block_q8_K *)
                ((const char *)vy + (jj + c) * a_row_bytes);
        }

        for (int ibl = 0; ibl < nb; ibl++) {
            float d = fp16_to_fp32_lookup(iq3[ibl].d);
            uint16_t sh = iq3[ibl].scales_h;
            uint16_t extra = iq3[ibl].extra;
            const uint8_t *qs = iq3[ibl].qs;
            const uint8_t *qh = iq3[ibl].qh;

            for (int c = 0; c < ncols_tile; c++) {
                float q8_scale = qk_ptr[c][ibl].d;
                float dy = d * q8_scale;
                const int8_t *q8 = qk_ptr[c][ibl].qs;
                __m256i isum = _mm256_setzero_si256();

                uint16_t sh_c = sh;
                uint16_t extra_c = extra;

                for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    uint8_t mag_lo = iq3[ibl].scales_l[ib32] & 0xf;
                    uint8_t mag_hi = iq3[ibl].scales_l[ib32] >> 4;
                    int scale_lo = (int)(mag_lo * 2 + 1) * ((sh_c & 1) ? 1 : -1);
                    int scale_hi = (int)(mag_hi * 2 + 1) * ((sh_c & 2) ? 1 : -1);
                    sh_c >>= 2;

                    __m256i scales = _mm256_set_epi32(
                        scale_hi, scale_hi, scale_hi, scale_hi,
                        scale_lo, scale_lo, scale_lo, scale_lo);

                    int shift_idx  = ((extra_c >> 0) & 1) * 8;
                    int shift_idx2 = ((extra_c >> 1) & 1) * 8;
                    extra_c >>= 2;

                    int shift_l = 2 * (ib32 % 4);
                    int shift_h = ib32 % 8;
                    int qs_off = (ib32 / 4) * 32;

                    __m128i lb_lo = _mm_loadu_si128((const __m128i *)(qs + qs_off + 0));
                    __m128i lb_hi = _mm_loadu_si128((const __m128i *)(qs + qs_off + 16));
                    __m128i hb = _mm_loadu_si128((const __m128i *)(qh));

                    __m256i lb_lo256 = _mm256_broadcastsi128_si256(lb_lo);
                    __m256i lb_hi256 = _mm256_broadcastsi128_si256(lb_hi);
                    __m256i hb256 = _mm256_broadcastsi128_si256(hb);

                    __m256i y_reg = _mm256_loadu_si256((const __m256i *)(q8 + ib32 * 32));

                    __m256i ql_lo, ql_hi;
                    if (shift_l == 0) {
                        ql_lo = _mm256_and_si256(lb_lo256, m03);
                        ql_hi = _mm256_and_si256(lb_hi256, m03);
                    } else if (shift_l == 2) {
                        ql_lo = _mm256_and_si256(_mm256_srli_epi16(lb_lo256, 2), m03);
                        ql_hi = _mm256_and_si256(_mm256_srli_epi16(lb_hi256, 2), m03);
                    } else if (shift_l == 4) {
                        ql_lo = _mm256_and_si256(_mm256_srli_epi16(lb_lo256, 4), m03);
                        ql_hi = _mm256_and_si256(_mm256_srli_epi16(lb_hi256, 4), m03);
                    } else {
                        ql_lo = _mm256_and_si256(_mm256_srli_epi16(lb_lo256, 6), m03);
                        ql_hi = _mm256_and_si256(_mm256_srli_epi16(lb_hi256, 6), m03);
                    }

                    __m256i qh_shifted = _mm256_srli_epi16(hb256, shift_h);
                    qh_shifted = _mm256_and_si256(qh_shifted, m03);
                    qh_shifted = _mm256_slli_epi16(qh_shifted, 2);

                    __m256i qx_lo = _mm256_or_si256(ql_lo, qh_shifted);
                    __m256i qx_hi = _mm256_or_si256(ql_hi, qh_shifted);

                    qx_lo = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx_lo, _mm256_set1_epi8(shift_idx)));
                    qx_hi = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx_hi, _mm256_set1_epi8(shift_idx2)));

                    __m256i s_lo = _mm256_sign_epi8(qx_lo, qx_lo);
                    __m256i s_hi = _mm256_sign_epi8(qx_hi, qx_hi);

                    __m256i y_lo = y_reg;
                    __m256i y_hi = _mm256_castsi128_si256(_mm256_extracti128_si256(y_reg, 1));

                    __m256i sumi;
#ifdef __AVX512VNNI__
                    __m256i acc_lo = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                        s_lo, _mm256_sign_epi8(y_lo, qx_lo));
                    __m256i acc_hi = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                        s_hi, _mm256_sign_epi8(y_hi, qx_hi));
                    sumi = _mm256_add_epi32(acc_lo, acc_hi);
#else
                    __m128i mad_lo = _mm256_castsi256_si128(
                        _mm256_maddubs_epi16(s_lo, _mm256_sign_epi8(y_lo, qx_lo)));
                    __m128i mad_hi = _mm256_castsi256_si128(
                        _mm256_maddubs_epi16(s_hi, _mm256_sign_epi8(y_hi, qx_hi)));
                    __m128i t1 = _mm_madd_epi16(_mm_set1_epi16(1), mad_lo);
                    __m128i t2 = _mm_madd_epi16(_mm_set1_epi16(1), mad_hi);
                    sumi = _mm256_insertf128_si256(_mm256_castsi128_si256(t1), t2, 1);
#endif
                    isum = _mm256_add_epi32(isum, _mm256_mullo_epi32(scales, sumi));
                    q8 += 32;
                }

                int total = hsum_i32x8(isum);
                acc[c] += (float)total * dy;
            }
        }

        for (int c = 0; c < ncols_tile; c++) {
            out[ii + (jj + c) * bs] = acc[c];
        }
    }
    return nrows;
#else
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy;
    (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
#endif
}
#else
/* Non-AVX2 stubs for cross-compilation compatibility */
void vec_dot_iq3_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) { (void)vx; (void)wy; (void)n; (void)out; }
int sgemm_iq3_k_q8_k_avx2(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)vx; (void)vy; (void)nrows; (void)ncols; (void)k; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
#endif
