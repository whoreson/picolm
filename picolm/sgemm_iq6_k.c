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
     * Move bit 5 to bit 7: slli by 3. Move bit 4 to bit 7: slli by 3.
     * blendv uses bit 7 of each byte as mask (1=take second arg). */
    const __m128i m0f = _mm_set1_epi8(0x0f);
    __m128i low4 = _mm_and_si128(q6, m0f);

    __m128i v0 = _mm_shuffle_epi8(lut0, low4);
    __m128i v1 = _mm_shuffle_epi8(lut1, low4);
    __m128i v2 = _mm_shuffle_epi8(lut2, low4);
    __m128i v3 = _mm_shuffle_epi8(lut3, low4);

    /* Extract bit 5 and bit 4 of q6, move to bit 7 position */
    __m128i bit5 = _mm_slli_epi16(q6, 3);   /* bit 5 -> bit 7 */
    __m128i bit4 = _mm_slli_epi16(q6, 2);   /* bit 4 -> bit 6, need >>1... */
    /* Actually: bit 4 is at position 4. To get to bit 7: shift left by 3.
     * But bit 5 also shifts into bit 8 (out of range for byte). So:
     * bit5 = (q6 >> 5) & 1, then << 7 = (q6 << 3) & 0x80
     * bit4 = (q6 >> 4) & 1, then << 7 = (q6 << 2) & 0x80 */
    bit5 = _mm_slli_epi16(q6, 3);   /* bit 5 -> bit 7, bit 4 -> bit 6 */
    bit4 = _mm_slli_epi16(q6, 2);   /* bit 4 -> bit 6, bit 5 -> bit 7 */

    /* mask for high2 bits: bit5 has bit 7 set when bit 5 of q6 is 1
     * bit4 has bit 7 set when bit 4 of q6 is 1 (after shifting)
     * Actually bit4 = q6 << 2: bit 4 goes to bit 6, not 7.
     * Let me redo: bit 4 -> bit 7 needs shift by 3, same as bit 5.
     * But we want them separately. Use srli first. */
    __m128i hi2 = _mm_and_si128(_mm_srli_epi16(q6, 4), _mm_set1_epi8(0x03));
    /* hi2 is 0..3. bit 0 of hi2 = old bit 4, bit 1 of hi2 = old bit 5.
     * To select: if hi2==0: v0, hi2==1: v1, hi2==2: v2, hi2==3: v3.
     * Use two blendv: first on bit 1 (select between {v0,v1} and {v2,v3}),
     * then on bit 0 (select within pair). */
    __m128i sel_lo = _mm_slli_epi16(hi2, 6);  /* bit 0 -> bit 6... need bit 7 */
    /* blendv_epi8 uses bit 7. So shift bit 0 to bit 7: << 7.
     * But hi2 is 0..3, so bit 0 is at position 0, bit 1 at position 1.
     * bit 0 -> bit 7: slli by 7. bit 1 -> bit 7: slli by 6. */
    __m128i mask0 = _mm_slli_epi16(hi2, 7);  /* bit 0 -> bit 7 */
    __m128i mask1 = _mm_slli_epi16(hi2, 6);  /* bit 1 -> bit 7 */

    /* v_lo = blendv(v0, v1, mask0) -- if bit 0 set, take v1 */
    __m128i v_lo = _mm_blendv_epi8(v0, v1, mask0);
    /* v_hi = blendv(v2, v3, mask0) -- if bit 0 set, take v3 */
    __m128i v_hi = _mm_blendv_epi8(v2, v3, mask0);
    /* result = blendv(v_lo, v_hi, mask1) -- if bit 1 set, take v_hi */
    return _mm_blendv_epi8(v_lo, v_hi, mask1);
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

/* ================================================================
 * AVX-512 VNNI path: vpdpbusd (unsigned-signed, 32-bit accum)
 * ================================================================
 * dpbusd is unsigned-signed without saturation. The LUT values
 * (0..249) are naturally unsigned. We multiply |y| with lut values,
 * then correct the sign via bsums:
 *
 *   result = sum_i(lut[q_i] * q8_i) - 128 * sum(q8_i) + mn * sum(q8_i)
 *          = dpbusd(lut, |y|) + (mn - 128) * bsum
 *
 * This eliminates the sign trick entirely.
 * ================================================================ */
#if defined(__AVX512F__) && defined(__AVX512VNNI__)

void vec_dot_iq6_k_q8_k_vnni(const void *vx, const void *wy, int n, float *out) {
    const block_iq6_k *x = (const block_iq6_k *)vx;
    const block_q8_K *y = (const block_q8_K *)wy;
    const int nb = n / QK_K;

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
        const int16_t *bsums = y[ibl].bsums;

        float block_sum = 0.0f;
        int shift = 0;

        for (int ib64 = 0; ib64 < QK_K / 64; ++ib64) {
            float sc[4];
            int mn[4];
            for (int j = 0; j < 4; j++) {
                sc[j] = d * sl[4 * ib64 + j];
                mn[j] = (extra >> j) & 1;
            }

            /* Load qs and qh */
            __m128i qs0 = _mm_loadu_si128((const __m128i *)(qs + 0));
            __m128i qs1 = _mm_loadu_si128((const __m128i *)(qs + 16));
            __m128i qh0 = _mm_loadu_si128((const __m128i *)(qh + 0));
            __m128i qh1 = _mm_loadu_si128((const __m128i *)(qh + 16));

            __m128i qh0s = _mm_srli_epi16(qh0, shift);
            __m128i qh1s = _mm_srli_epi16(qh1, shift);

            __m128i qs0_lo = _mm_and_si128(qs0, m0f);
            __m128i qs1_lo = _mm_and_si128(qs1, m0f);
            __m128i qs0_hi = _mm_and_si128(_mm_srli_epi16(qs0, 4), m0f);
            __m128i qs1_hi = _mm_and_si128(_mm_srli_epi16(qs1, 4), m0f);

            __m128i qh0_lo2 = _mm_and_si128(qh0s, m03);
            __m128i qh1_lo2 = _mm_and_si128(qh1s, m03);
            __m128i qh0_hi2 = _mm_and_si128(qh0s, m0c);
            __m128i qh1_hi2 = _mm_and_si128(qh1s, m0c);

            __m128i q1 = _mm_or_si128(qs0_lo, _mm_slli_epi16(qh0_lo2, 4));
            __m128i q2 = _mm_or_si128(qs1_lo, _mm_slli_epi16(qh1_lo2, 4));
            __m128i q3 = _mm_or_si128(qs0_hi, _mm_slli_epi16(qh0_hi2, 2));
            __m128i q4 = _mm_or_si128(qs1_hi, _mm_slli_epi16(qh1_hi2, 2));

            /* LUT lookup -> unsigned values 0..249 */
            __m128i val1 = iq6_lut_select(q1, lut0, lut1, lut2, lut3);
            __m128i val2 = iq6_lut_select(q2, lut0, lut1, lut2, lut3);
            __m128i val3 = iq6_lut_select(q3, lut0, lut1, lut2, lut3);
            __m128i val4 = iq6_lut_select(q4, lut0, lut1, lut2, lut3);

            /* Load |q8| for dpbusd (unsigned-signed MAC) */
            __m128i y0 = _mm_loadu_si128((const __m128i *)(q8 + 0));
            __m128i y1 = _mm_loadu_si128((const __m128i *)(q8 + 16));
            __m128i y2 = _mm_loadu_si128((const __m128i *)(q8 + 32));
            __m128i y3 = _mm_loadu_si128((const __m128i *)(q8 + 48));
            __m128i y0_abs = _mm_sign_epi8(y0, y0);
            __m128i y1_abs = _mm_sign_epi8(y1, y1);
            __m128i y2_abs = _mm_sign_epi8(y2, y2);
            __m128i y3_abs = _mm_sign_epi8(y3, y3);

            /* dpbusd: unsigned lUT * signed |y| -> 4x int32 per stream.
             * _mm_dpbusd_epi32 accumulates into int32 without saturation.
             * We need to zero-extend lut values to uint8 (they're already 0..249).
             * But dpbusd works on 128-bit: 16 uint8 * 16 int8 -> 4 int32.
             * Accumulate into 4 int32 per stream. */
            __m128i acc = _mm_setzero_si128();
            acc = _mm_dpbusd_epi32(acc, val1, y0_abs);
            acc = _mm_dpbusd_epi32(acc, val2, y1_abs);
            acc = _mm_dpbusd_epi32(acc, val3, y2_abs);
            acc = _mm_dpbusd_epi32(acc, val4, y3_abs);

            /* acc now has 4 int32 partial sums from all 4 streams combined.
             * But wait - dpbusd accumulates within each 128-bit lane.
             * Each call adds 4 new int32 to the 4 existing ones.
             * After 4 calls: acc[0] = sum of all stream0 dpbusd + stream1 + stream2 + stream3.
             * Actually no - dpbusd processes 16 pairs per call, producing 4 int32.
             * Each int32 covers 4 consecutive byte pairs.
             * After 4 calls on different data, acc[0] = dpbusd(val1[0..3],y0[0..3])
             *   + dpbusd(val2[0..3],y1[0..3]) + dpbusd(val3[0..3],y2[0..3])
             *   + dpbusd(val4[0..3],y3[0..3]).
             * And acc[1]..acc[3] cover the next groups of 4 bytes each.
             * Total: 16 values per stream * 4 streams = 64 values.
             * 4 int32 * 16 values each = 64. Correct!
             *
             * Now we need to multiply each int32 by its scale and sum.
             * But the 4 int32 in acc correspond to different byte ranges,
             * not different streams. We can't easily separate them by stream.
             *
             * The scalar approach: compute dpbusd per stream, then scale per stream. */
            (void)acc;

            __m128i a0 = _mm_dpbusd_epi32(_mm_setzero_si128(), val1, y0_abs);
            __m128i a1 = _mm_dpbusd_epi32(_mm_setzero_si128(), val2, y1_abs);
            __m128i a2 = _mm_dpbusd_epi32(_mm_setzero_si128(), val3, y2_abs);
            __m128i a3 = _mm_dpbusd_epi32(_mm_setzero_si128(), val4, y3_abs);

            /* Scale each stream's 4 int32 by sl[g], then hsum.
             * dpbusd gives 4 int32, each covering 4 byte pairs.
             * All 4 int32 in a stream share the same scale. */
            __m128i s0 = _mm_madd_epi16(
                _mm_cvtepi32_epi16(a0), _mm_set1_epi16(sl[4*ib64]));
            /* cvtepi32_epi16 truncates to 16-bit, which will overflow.
             * Use mulpd or just do scalar multiply on each int32. */
            (void)s0;

            /* Scalar approach: extract 4 int32, multiply by scale, sum */
            int32_t tmp0[4], tmp1[4], tmp2[4], tmp3[4];
            _mm_storeu_si128((__m128i*)tmp0, a0);
            _mm_storeu_si128((__m128i*)tmp1, a1);
            _mm_storeu_si128((__m128i*)tmp2, a2);
            _mm_storeu_si128((__m128i*)tmp3, a3);

            int dp[4] = {0, 0, 0, 0};
            for (int k = 0; k < 4; k++) {
                dp[0] += tmp0[k]; dp[1] += tmp1[k];
                dp[2] += tmp2[k]; dp[3] += tmp3[k];
            }

            /* Bias correction: subtract 128*bsum, add mn*bsum.
             * dpbusd(lut, |y|) = sum(lut[q_i] * |y_i|)
             * We want: sum((lut[q_i] - 128 + mn) * y_i)
             *         = sum(lut[q_i] * y_i) - 128*sum(y_i) + mn*sum(y_i)
             * dpbusd used |y_i|, so we got sum(lut[q_i] * |y_i|).
             * We need sum(lut[q_i] * y_i) = sum(lut[q_i] * sign(y_i) * |y_i|).
             * dpbusd is unsigned-signed, so it computes sum(lut * sign(y)*|y|) = sum(lut*y).
             * Wait - dpbusd takes (uint8, int8). val is uint8 (0..249), y_abs is |y|.
             * dpbusd computes sum(val_i * y_abs_i) = sum(lut * |y|).
             * But we need sum(lut * y) = sum(lut * sign(y) * |y|).
             * The sign is lost! We need to use the original y (with sign), not |y|.
             *
             * dpbusd is unsigned-signed: it treats the second operand as SIGNED.
             * So _mm_dpbusd_epi32(zero, val, y) where y has sign will give
             * sum(val_i * y_i) correctly. val is treated as unsigned (0..255).
             * Since lut values are 0..249, this is correct. */

            /* Redo with signed y */
            a0 = _mm_dpbusd_epi32(_mm_setzero_si128(), val1, y0);
            a1 = _mm_dpbusd_epi32(_mm_setzero_si128(), val2, y1);
            a2 = _mm_dpbusd_epi32(_mm_setzero_si128(), val3, y2);
            a3 = _mm_dpbusd_epi32(_mm_setzero_si128(), val4, y3);

            _mm_storeu_si128((__m128i*)tmp0, a0);
            _mm_storeu_si128((__m128i*)tmp1, a1);
            _mm_storeu_si128((__m128i*)tmp2, a2);
            _mm_storeu_si128((__m128i*)tmp3, a3);

            dp[0] = dp[1] = dp[2] = dp[3] = 0;
            for (int k = 0; k < 4; k++) {
                dp[0] += tmp0[k]; dp[1] += tmp1[k];
                dp[2] += tmp2[k]; dp[3] += tmp3[k];
            }

            /* dp[g] = sum(lut[q_i] * y_i) for stream g.
             * We want: sum((lut[q_i] - 128 + mn[g]) * y_i)
             *         = dp[g] + (mn[g] - 128) * bsum[g]
             * bsum[g] = sum(y_i) for stream g. */
            int bs_off = ib64 * 4;
            int b0 = bsums[bs_off+0];
            int b1 = bsums[bs_off+1];
            int b2 = bsums[bs_off+2];
            int b3 = bsums[bs_off+3];

            block_sum += sc[0] * (dp[0] + (mn[0] - 128) * b0);
            block_sum += sc[1] * (dp[1] + (mn[1] - 128) * b1);
            block_sum += sc[2] * (dp[2] + (mn[2] - 128) * b2);
            block_sum += sc[3] * (dp[3] + (mn[3] - 128) * b3);

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
#endif

