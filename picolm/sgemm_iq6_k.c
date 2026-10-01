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

/* iq6nl_lut defined in quant.c, declared extern in quant.h */

#if defined(__AVX2__)

static inline __m128i iq6_lut_select(__m128i q6, __m128i lut0, __m128i lut1, __m128i lut2, __m128i lut3) {
    /* Select 16-byte LUT quarter based on bits 5-4 of q6 (values 0..3).
     * hi2==0 -> lut0, hi2==1 -> lut1, hi2==2 -> lut2, hi2==3 -> lut3.
     * blendv_epi8 uses bit 7 of each byte as selector.
     * Strategy: two-level mux. First select within pairs using bit 0 of hi2,
     * then select between pairs using bit 1 of hi2. */
    const __m128i m0f = _mm_set1_epi8(0x0f);
    const __m128i m80 = _mm_set1_epi8((char)0x80);
    __m128i low4 = _mm_and_si128(q6, m0f);

    __m128i v0 = _mm_shuffle_epi8(lut0, low4);
    __m128i v1 = _mm_shuffle_epi8(lut1, low4);
    __m128i v2 = _mm_shuffle_epi8(lut2, low4);
    __m128i v3 = _mm_shuffle_epi8(lut3, low4);

    /* Extract hi2 = (q6 >> 4) & 3, as bytes.
     * bit 0 of hi2 = bit 4 of q6 -> move to bit 7: srl by 4, srl by 3...
     * Actually: (q6 >> 4) & 0x03 gives 0..3 in each byte.
     * bit 0 of that = ((q6 >> 4) & 1) -> need to move to bit 7: << 7
     * bit 1 of that = ((q6 >> 4) & 2) -> need to move to bit 7: << 6
     * But << 7 on a value that's 0 or 1 gives 0 or 0x80. Correct.
     * << 6 on a value that's 0 or 2 gives 0 or 0xC0 -> 0x80 in byte. Correct.
     * However, slli_epi16 shifts 16-bit ints, so 1<<7=0x80 fits, 2<<6=0xC0 fits. */
    __m128i hi2 = _mm_and_si128(_mm_srli_epi16(q6, 4), _mm_set1_epi8(0x03));

    /* mask0: bit 0 of hi2 moved to bit 7. And with 0x80 to ensure byte clean. */
    __m128i mask0 = _mm_and_si128(_mm_slli_epi16(hi2, 7), m80);
    /* mask1: bit 1 of hi2 moved to bit 7. And with 0x80. */
    __m128i mask1 = _mm_and_si128(_mm_slli_epi16(hi2, 6), m80);

    /* Within-pair select: hi2 bit 0 */
    __m128i v_lo = _mm_blendv_epi8(v0, v1, mask0);  /* bit0=0: v0, bit0=1: v1 */
    __m128i v_hi = _mm_blendv_epi8(v2, v3, mask0);  /* bit0=0: v2, bit0=1: v3 */
    /* Between-pair select: hi2 bit 1 */
    return _mm_blendv_epi8(v_lo, v_hi, mask1);      /* bit1=0: v_lo, bit1=1: v_hi */
}

static inline int iq6_signed_mac(__m128i val, __m128i y_vec) {
    __m128i abs_val = _mm_sign_epi8(val, val);
    __m128i sy = _mm_sign_epi8(y_vec, val);
    __m128i p16 = _mm_maddubs_epi16(abs_val, sy);

    /* Horizontal sum of 8 int16 -> single int32.
     * madd_epi16 with ones: 8 int16 -> 4 int32 (pairwise sums).
     * Then reduce 4 int32 to 1 via two shuffle-adds. */
    const __m128i one16 = _mm_set1_epi16(1);
    __m128i s = _mm_madd_epi16(p16, one16);  /* {p0+p1, p2+p3, p4+p5, p6+p7} */
    s = _mm_hadd_epi32(s, s);                 /* {p0+p1+p2+p3, p2+p3+p4+p5, p4+p5+p6+p7, junk} */
    s = _mm_shuffle_epi32(s, 0x33);           /* broadcast lane 2 = p4+p5+p6+p7? NO */
    /* Actually hadd_epi32: {a,b,c,d} -> {a+b, b+c, c+d, d+c} -- wait no */
    /* hadd_epi32(s, s): horizontal add of adjacent pairs within and across 128-bit lanes */
    /* For 128-bit register {a,b,c,d}: hadd gives {a+b, c+d, c+d, c+d} -- NO */
    /* Let me just use the straightforward approach */
    (void)s;
    int32_t tmp[4];
    _mm_storeu_si128((__m128i*)tmp, _mm_madd_epi16(p16, one16));
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
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
    (void)vx; (void)wy; (void)n;
    *out = 0.0f;
}

#endif

