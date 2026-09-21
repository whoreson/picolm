/* ================================================================
 * IQ2_K (plain, GGUF type 137) x Q8_K AVX2 GEMV and GEMM kernels
 * ================================================================
 * Weights: block_iq2_k (GGUF type 137), 76 bytes/block
 * Activations: block_q8_K
 *
 * Per subblock: 32 values from qs[j]+qs[j+16] (j=0..15), shift 0,2,4,6
 * ib32 0-3 use qs[0..31], ib32 4-7 use qs[32..63]
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

void vec_dot_iq2_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
#if defined(__AVX2__) && defined(__F16C__)
    assert(n % QK_K == 0);

    const block_iq2_k *iq2 = (const block_iq2_k *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    /* LUT: 8 entries (4 normal + 4 shifted), broadcast to both 128-bit lanes */
    static const int8_t kvalues_iq2nl[32] = {
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
    };
    const __m256i values = _mm256_loadu_si256((const __m256i *)kvalues_iq2nl);

    float result = 0.0f;

    for (int ibl = 0; ibl < nb; ibl++) {
        float d = fp16_to_fp32_lookup(iq2[ibl].d);
        float q8_scale = qk[ibl].d;
        float dy = d * q8_scale;

        uint16_t extra = iq2[ibl].extra;
        const uint8_t *qs = iq2[ibl].qs;
        const int8_t *q8 = qk[ibl].qs;

        __m256i isum = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            int scale_lo = (iq2[ibl].scales[ib32] & 0xf) - 8;
            int scale_hi = (iq2[ibl].scales[ib32] >> 4) - 8;
            __m256i scales = _mm256_set_epi32(
                scale_hi, scale_hi, scale_hi, scale_hi,
                scale_lo, scale_lo, scale_lo, scale_lo);

            /* LUT table selection: each extra bit selects normal (0) or shifted (+4) */
            int shift_idx  = ((extra >> 0) & 1) * 4;
            int shift_idx2 = ((extra >> 1) & 1) * 4;
            extra >>= 2;

            int shift = 2 * (ib32 % 4);
            int qs_off = (ib32 / 4) * 32;

            __m128i lb_lo = _mm_loadu_si128((const __m128i *)(qs + qs_off + 0));
            __m128i lb_hi = _mm_loadu_si128((const __m128i *)(qs + qs_off + 16));

            /* Load 32 activations: q8[ib32*32 .. ib32*32+31] */
            __m256i y_reg = _mm256_loadu_si256((const __m256i *)(q8 + ib32 * 32));

            /* Extract low 2 bits from qs */
            __m128i ql_lo, ql_hi;
            if (shift == 0) {
                ql_lo = _mm_and_si128(lb_lo, _mm_set1_epi8(0x03));
                ql_hi = _mm_and_si128(lb_hi, _mm_set1_epi8(0x03));
            } else if (shift == 2) {
                ql_lo = _mm_and_si128(_mm_srli_epi16(lb_lo, 2), _mm_set1_epi8(0x03));
                ql_hi = _mm_and_si128(_mm_srli_epi16(lb_hi, 2), _mm_set1_epi8(0x03));
            } else if (shift == 4) {
                ql_lo = _mm_and_si128(_mm_srli_epi16(lb_lo, 4), _mm_set1_epi8(0x03));
                ql_hi = _mm_and_si128(_mm_srli_epi16(lb_hi, 4), _mm_set1_epi8(0x03));
            } else {
                ql_lo = _mm_and_si128(_mm_srli_epi16(lb_lo, 6), _mm_set1_epi8(0x03));
                ql_hi = _mm_and_si128(_mm_srli_epi16(lb_hi, 6), _mm_set1_epi8(0x03));
            }

            /* LUT lookup: 2-bit index -> int8 dequantized value */
            ql_lo = _mm_shuffle_epi8(_mm256_castsi256_si128(values),
                                     _mm_add_epi8(ql_lo, _mm_set1_epi8(shift_idx)));
            ql_hi = _mm_shuffle_epi8(_mm256_castsi256_si128(values),
                                     _mm_add_epi8(ql_hi, _mm_set1_epi8(shift_idx2)));

            /* Broadcast 16 dequantized values to 32 lanes (for 256-bit ops) */
            __m256i qx_lo = _mm256_broadcastsi128_si256(ql_lo);
            __m256i qx_hi = _mm256_broadcastsi128_si256(ql_hi);

            /* Sign trick for signed*signed via maddubs (unsigned*signed):
             * |a| * sign(a)*b = a*b */
            __m256i s_lo = _mm256_sign_epi8(qx_lo, qx_lo);
            __m256i s_hi = _mm256_sign_epi8(qx_hi, qx_hi);

            /* maddubs uses only the low 128 bits of each 256-bit operand.
             * For lo half: s_lo[0..15] = |qx_lo[0..15]|, y_reg[0..15] = q8[0..31] low 16 bytes
             * For hi half: s_hi[0..15] = |qx_hi[0..15]|, need q8[16..31] in low 128 bits
             *
             * sign(y, qx) applies qx sign to y. For lo: sign(y_reg[0..15], qx_lo[0..15]).
             * For hi: need sign(q8[16..31], qx_hi[0..15]).
             * Extract high 128 bits of y_reg and cast to low 128 of a 256-bit reg. */
            __m256i y_lo = y_reg;  /* low 128 bits already have q8[0..15] */
            __m256i y_hi = _mm256_castsi128_si256(_mm256_extracti128_si256(y_reg, 1));

            __m256i sumi;
#ifdef __AVX512VNNI__
            /* VNNI path: dpbusd does unsigned*signed with accumulation */
            __m256i acc_lo = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                s_lo, _mm256_sign_epi8(y_lo, qx_lo));
            __m256i acc_hi = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                s_hi, _mm256_sign_epi8(y_hi, qx_hi));
            sumi = _mm256_add_epi32(acc_lo, acc_hi);
#else
            /* AVX2 path: maddubs + madd_epi16 */
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

int sgemm_iq2_k_q8_k_avx2(int nrows, int ncols, int k,
                           const void *vx, const void *vy,
                           float *out, size_t bs,
                           int ith, int nth) {
#if defined(__AVX2__) && defined(__F16C__)
    if (nrows < 1 || ncols < 1 || k % QK_K != 0)
        return 0;

    const int nb = k / QK_K;

    static const int8_t kvalues_iq2nl[32] = {
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
        -31, -13, 1, 17, -26, -8, 6, 22,
    };
    const __m256i values = _mm256_loadu_si256((const __m256i *)kvalues_iq2nl);

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

    size_t w_row_bytes = nb * sizeof(block_iq2_k);
    size_t a_row_bytes = nb * sizeof(block_q8_K);

    for (int64_t job = start; job < end; job++) {
        int64_t ii = job / xtiles_ext;
        int64_t xt = job % xtiles_ext;
        int64_t jj = xt * 2;
        int64_t ncols_tile = (xt < xtiles) ? 2 : n_tail;
        if (ncols_tile < 1) ncols_tile = 1;

        const block_iq2_k *iq2 = (const block_iq2_k *)
            ((const char *)vx + ii * w_row_bytes);

        float acc[2] = { 0.0f, 0.0f };

        const block_q8_K *qk_ptr[2];
        for (int c = 0; c < ncols_tile; c++) {
            qk_ptr[c] = (const block_q8_K *)
                ((const char *)vy + (jj + c) * a_row_bytes);
        }

        for (int ibl = 0; ibl < nb; ibl++) {
            float d = fp16_to_fp32_lookup(iq2[ibl].d);
            uint16_t extra = iq2[ibl].extra;
            const uint8_t *qs = iq2[ibl].qs;

            for (int c = 0; c < ncols_tile; c++) {
                float q8_scale = qk_ptr[c][ibl].d;
                float dy = d * q8_scale;
                const int8_t *q8 = qk_ptr[c][ibl].qs;
                __m256i isum = _mm256_setzero_si256();

                uint16_t extra_c = extra;

                for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    int scale_lo = (iq2[ibl].scales[ib32] & 0xf) - 8;
                    int scale_hi = (iq2[ibl].scales[ib32] >> 4) - 8;
                    __m256i scales = _mm256_set_epi32(
                        scale_hi, scale_hi, scale_hi, scale_hi,
                        scale_lo, scale_lo, scale_lo, scale_lo);

                    int shift_idx  = ((extra_c >> 0) & 1) * 4;
                    int shift_idx2 = ((extra_c >> 1) & 1) * 4;
                    extra_c >>= 2;

                    int shift = 2 * (ib32 % 4);
                    int qs_off = (ib32 / 4) * 32;

                    __m128i lb_lo = _mm_loadu_si128((const __m128i *)(qs + qs_off + 0));
                    __m128i lb_hi = _mm_loadu_si128((const __m128i *)(qs + qs_off + 16));
                    __m256i y_reg = _mm256_loadu_si256((const __m256i *)(q8 + ib32 * 32));

                    __m128i ql_lo, ql_hi;
                    if (shift == 0) {
                        ql_lo = _mm_and_si128(lb_lo, _mm_set1_epi8(0x03));
                        ql_hi = _mm_and_si128(lb_hi, _mm_set1_epi8(0x03));
                    } else if (shift == 2) {
                        ql_lo = _mm_and_si128(_mm_srli_epi16(lb_lo, 2), _mm_set1_epi8(0x03));
                        ql_hi = _mm_and_si128(_mm_srli_epi16(lb_hi, 2), _mm_set1_epi8(0x03));
                    } else if (shift == 4) {
                        ql_lo = _mm_and_si128(_mm_srli_epi16(lb_lo, 4), _mm_set1_epi8(0x03));
                        ql_hi = _mm_and_si128(_mm_srli_epi16(lb_hi, 4), _mm_set1_epi8(0x03));
                    } else {
                        ql_lo = _mm_and_si128(_mm_srli_epi16(lb_lo, 6), _mm_set1_epi8(0x03));
                        ql_hi = _mm_and_si128(_mm_srli_epi16(lb_hi, 6), _mm_set1_epi8(0x03));
                    }

                    ql_lo = _mm_shuffle_epi8(_mm256_castsi256_si128(values),
                                             _mm_add_epi8(ql_lo, _mm_set1_epi8(shift_idx)));
                    ql_hi = _mm_shuffle_epi8(_mm256_castsi256_si128(values),
                                             _mm_add_epi8(ql_hi, _mm_set1_epi8(shift_idx2)));

                    __m256i qx_lo = _mm256_broadcastsi128_si256(ql_lo);
                    __m256i qx_hi = _mm256_broadcastsi128_si256(ql_hi);

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
