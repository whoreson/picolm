/* ================================================================
 * IQ6_K (GGUF type 141) x Q8_K AVX2 GEMV kernel
 * ================================================================
 * Direct AVX2 translation of the scalar reference in quant.c.
 * For each 64-value sub-block, 4 parallel streams of 16 values each.
 * ================================================================ */

#include <stdint.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "quant.h"

#define QK_K 256

static const uint8_t iq6nl_lut[64] = {
       1,    7,   13,   19,   24,   30,   35,   40,   44,   49,   54,   58,   62,   66,   70,   74,
      77,   81,   84,   88,   91,   94,   97,  100,  103,  106,  109,  112,  115,  117,  120,  123,
     126,  128,  131,  134,  137,  140,  142,  145,  148,  151,  155,  158,  161,  164,  168,  172,
     175,  179,  183,  187,  191,  196,  200,  205,  210,  215,  220,  226,  231,  237,  243,  249,
};

#if defined(__AVX2__) && defined(__F16C__)

static inline __m128i iq6_lut_select(__m128i q6, __m128i lut0, __m128i lut1, __m128i lut2, __m128i lut3) {
    const __m128i m0f = _mm_set1_epi8(0x0f);
    const __m128i m03 = _mm_set1_epi8(0x03);
    __m128i low4 = _mm_and_si128(q6, m0f);
    __m128i high2 = _mm_and_si128(_mm_srli_epi16(q6, 4), m03);

    __m128i v0 = _mm_shuffle_epi8(lut0, low4);
    __m128i v1 = _mm_shuffle_epi8(lut1, low4);
    __m128i v2 = _mm_shuffle_epi8(lut2, low4);
    __m128i v3 = _mm_shuffle_epi8(lut3, low4);

    __m128i m0 = _mm_cmpeq_epi8(high2, _mm_setzero_si128());
    __m128i m1 = _mm_cmpeq_epi8(high2, _mm_set1_epi8(1));
    __m128i m2 = _mm_cmpeq_epi8(high2, _mm_set1_epi8(2));

    return _mm_blendv_epi8(_mm_blendv_epi8(v3, v2, m2), _mm_blendv_epi8(v1, v0, m0), m1);
}

static inline int iq6_signed_mac(__m128i val, __m128i y_vec) {
    __m128i abs_val = _mm_sign_epi8(val, val);
    __m128i sy = _mm_sign_epi8(y_vec, val);
    __m128i p16 = _mm_maddubs_epi16(abs_val, sy);

    __m128i hi16 = _mm_unpackhi_epi64(p16, p16);
    __m128i lo16 = _mm_unpacklo_epi64(p16, p16);
    __m128i one16 = _mm_set1_epi16(1);
    __m128i s0 = _mm_madd_epi16(lo16, one16);
    __m128i s1 = _mm_madd_epi16(hi16, one16);
    __m128i s = _mm_add_epi32(s0, s1);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x1b));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x33));
    return _mm_cvtsi128_si32(s);
}

void vec_dot_iq6_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
    const block_iq6_k *x = (const block_iq6_k *)vx;
    const block_q8_K *y = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    const __m128i m128 = _mm_set1_epi8(128);
    const __m128i m0f  = _mm_set1_epi8(0x0f);
    const __m128i m03  = _mm_set1_epi8(0x03);
    const __m128i m0c  = _mm_set1_epi8(0x0c);

    const __m128i lut0 = _mm_loadu_si128((const __m128i *)(iq6nl_lut + 0));
    const __m128i lut1 = _mm_loadu_si128((const __m128i *)(iq6nl_lut + 16));
    const __m128i lut2 = _mm_loadu_si128((const __m128i *)(iq6nl_lut + 32));
    const __m128i lut3 = _mm_loadu_si128((const __m128i *)(iq6nl_lut + 48));

    float sumf = 0.0f;

    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d = fp16_to_fp32(x[ibl].d);
        const uint8_t *qs = x[ibl].qs;
        const uint8_t *qh = x[ibl].qh;
        const int8_t *sl = x[ibl].scales;
        uint16_t extra = x[ibl].extra;
        const int8_t *q8 = y[ibl].qs;
        const float q8_scale = y[ibl].d;

        float block_sum = 0.0f;
        int shift = 0;

        for (int ib64 = 0; ib64 < QK_K / 64; ++ib64) {
            float sc[4];
            int8_t mn[4];
            for (int j = 0; j < 4; j++) {
                sc[j] = d * sl[4 * ib64 + j];
                mn[j] = (extra >> j) & 1;
            }

            /* Load qs[0..15] and qs[16..31] */
            __m128i qs0 = _mm_loadu_si128((const __m128i *)(qs + 0));
            __m128i qs1 = _mm_loadu_si128((const __m128i *)(qs + 16));

            /* Load qh[0..15] and qh[16..31] */
            __m128i qh0 = _mm_loadu_si128((const __m128i *)(qh + 0));
            __m128i qh1 = _mm_loadu_si128((const __m128i *)(qh + 16));

            /* Shift qh right by 'shift' bits */
            __m128i qh0s = _mm_srli_epi16(qh0, shift);
            __m128i qh1s = _mm_srli_epi16(qh1, shift);

            /* Extract low 4 bits and high 4 bits from qs */
            __m128i qs0_lo = _mm_and_si128(qs0, m0f);    /* qs[0..15] & 0xf */
            __m128i qs1_lo = _mm_and_si128(qs1, m0f);    /* qs[16..31] & 0xf */
            __m128i qs0_hi = _mm_and_si128(_mm_srli_epi16(qs0, 4), m0f);  /* qs[0..15] >> 4 */
            __m128i qs1_hi = _mm_and_si128(_mm_srli_epi16(qs1, 4), m0f);  /* qs[16..31] >> 4 */

            /* Extract qh bits: (qh >> shift) & 0x03 for q1,q2; & 0x0c for q3,q4 */
            __m128i qh0_lo2 = _mm_and_si128(qh0s, m03);  /* bits 0-1 */
            __m128i qh1_lo2 = _mm_and_si128(qh1s, m03);
            __m128i qh0_hi2 = _mm_and_si128(qh0s, m0c);  /* bits 2-3 */
            __m128i qh1_hi2 = _mm_and_si128(qh1s, m0c);

            /* q1 = qs0_lo | (qh0_lo2 << 4) */
            __m128i q1 = _mm_or_si128(qs0_lo, _mm_slli_epi16(qh0_lo2, 4));
            /* q2 = qs1_lo | (qh1_lo2 << 4) */
            __m128i q2 = _mm_or_si128(qs1_lo, _mm_slli_epi16(qh1_lo2, 4));
            /* q3 = qs0_hi | (qh0_hi2 << 2) -- NOTE: << 2, not << 4! */
            __m128i q3 = _mm_or_si128(qs0_hi, _mm_slli_epi16(qh0_hi2, 2));
            /* q4 = qs1_hi | (qh1_hi2 << 2) */
            __m128i q4 = _mm_or_si128(qs1_hi, _mm_slli_epi16(qh1_hi2, 2));

            /* LUT lookup for each stream */
            __m128i val1 = iq6_lut_select(q1, lut0, lut1, lut2, lut3);
            __m128i val2 = iq6_lut_select(q2, lut0, lut1, lut2, lut3);
            __m128i val3 = iq6_lut_select(q3, lut0, lut1, lut2, lut3);
            __m128i val4 = iq6_lut_select(q4, lut0, lut1, lut2, lut3);

            /* Subtract 128 and add min bit */
            val1 = _mm_add_epi8(_mm_sub_epi8(val1, m128), _mm_set1_epi8(mn[0]));
            val2 = _mm_add_epi8(_mm_sub_epi8(val2, m128), _mm_set1_epi8(mn[1]));
            val3 = _mm_add_epi8(_mm_sub_epi8(val3, m128), _mm_set1_epi8(mn[2]));
            val4 = _mm_add_epi8(_mm_sub_epi8(val4, m128), _mm_set1_epi8(mn[3]));

            /* Load Q8_K activations */
            __m128i y0 = _mm_loadu_si128((const __m128i *)(q8 + 0));
            __m128i y1 = _mm_loadu_si128((const __m128i *)(q8 + 16));
            __m128i y2 = _mm_loadu_si128((const __m128i *)(q8 + 32));
            __m128i y3 = _mm_loadu_si128((const __m128i *)(q8 + 48));

            /* Sign trick: signed MAC via maddubs */
            block_sum += (float)iq6_signed_mac(val1, y0) * sc[0];
            block_sum += (float)iq6_signed_mac(val2, y1) * sc[1];
            block_sum += (float)iq6_signed_mac(val3, y2) * sc[2];
            block_sum += (float)iq6_signed_mac(val4, y3) * sc[3];

            qs += 32;
            q8 += 64;
            extra >>= 4;
            shift += 4;
            if (shift == 8) { qh += 32; shift = 0; }
        }
        sumf += block_sum * q8_scale;
    }
    *out = sumf;
}

#else

void vec_dot_iq6_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
    (void)vx; (void)wy; (void)n; (void)out;
}

#endif

