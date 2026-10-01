/* ================================================================
 * IQ6_K (GGUF type 141) x Q8_K AVX2 + VNNI kernels
 * ================================================================
 * Layout per 256-value block (see quant.h block_iq6_k):
 *   value k of group g = 16-value group, g = k / 16
 *   w = iq6nl_lut[q6] - 128 + m_g,  q6 = lo4 | hi2 << 4,  m_g = bit g of extra
 *   result = d * sum_g scale_g * sum_{k in g} w_k * y_k
 *
 * Method (per block):
 *   1. Decode 256 weights to signed int8 (w[8] x 32 bytes), WITHOUT m_g.
 *      LUT is pre-biased by -128, so pshufb output is already signed.
 *      LUT quarter is chosen with blendv; the blend masks come straight
 *      from qh by a left shift (bit -> bit 7). No q6 reassembly.
 *   2. Dot with Q8_K: abs/sign trick + maddubs (int16), then madd_epi16
 *      against a per-16-group int16 scale vector. This applies the scale
 *      and reduces to int32 in one step. No per-group horizontal sum.
 *   3. The m_g term is  sum_g scale_g * m_g * bsums[g]  (one madd).
 *   4. One int32 -> float conversion per block, one horizontal sum per row.
 *
 * Int32 range: |acc| <= 256 * 127 * 127 * 127 + small < 2^31.
 * The batch kernel decodes each weight block once and reuses it for up
 * to IQ6K_BATCH_TILE activation rows.
 * ================================================================ */

#include <stdint.h>
#include <stddef.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "quant.h"

#define QK_K 256

/* iq6nl_lut defined in quant.c, declared extern in quant.h */

#if defined(__AVX2__)

typedef struct { __m256i l0, l1, l2, l3; } iq6_luts_t;

/* Signed LUT quarters: (iq6nl_lut[i] - 128) as int8, duplicated in both
 * 128-bit lanes (pshufb is per lane). x - 128 == x + 128 (mod 256). */
static inline iq6_luts_t iq6_luts_init(void) {
    const __m256i m128 = _mm256_set1_epi8((char)0x80);
    iq6_luts_t L;
    L.l0 = _mm256_sub_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq6nl_lut +  0))), m128);
    L.l1 = _mm256_sub_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq6nl_lut + 16))), m128);
    L.l2 = _mm256_sub_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq6nl_lut + 32))), m128);
    L.l3 = _mm256_sub_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq6nl_lut + 48))), m128);
    return L;
}

static inline float iq6_f16(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    return fp16_to_fp32(h);
#endif
}

static inline float iq6_hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

/* low4: 32 bytes, values 0..15 (low 4 bits of q6).
 * qh:   32 raw qh bytes.
 * c0/c1: shift counts that move the two high bits of q6 (bit 4, bit 5)
 *        to bit 7 of each byte. blendv reads bit 7 only, so bits that
 *        spill in from the neighbour byte do not matter (shift <= 7). */
static inline __m256i iq6_lookup(__m256i low4, __m256i qh, __m128i c0, __m128i c1,
                                 const iq6_luts_t *L) {
    const __m256i v0 = _mm256_shuffle_epi8(L->l0, low4);
    const __m256i v1 = _mm256_shuffle_epi8(L->l1, low4);
    const __m256i v2 = _mm256_shuffle_epi8(L->l2, low4);
    const __m256i v3 = _mm256_shuffle_epi8(L->l3, low4);
    const __m256i m0 = _mm256_sll_epi16(qh, c0);   /* q6 bit 4 -> bit 7 */
    const __m256i m1 = _mm256_sll_epi16(qh, c1);   /* q6 bit 5 -> bit 7 */
    const __m256i lo = _mm256_blendv_epi8(v0, v1, m0);
    const __m256i hi = _mm256_blendv_epi8(v2, v3, m0);
    return _mm256_blendv_epi8(lo, hi, m1);
}

/* int16 vector: lanes 0..7 = a, lanes 8..15 = b. */
static inline __m256i iq6_scale_pair(int a, int b) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16((short)a)),
                                   _mm_set1_epi16((short)b), 1);
}

/* Decode one block.
 *   w[k]  : signed int8 weights for values 32k .. 32k+31 (no m_g added)
 *   sp[k] : int16 scale vector for w[k]: lanes 0-7 = scale of group 2k,
 *           lanes 8-15 = scale of group 2k+1
 *   smin  : int16[16], scale_g if m_g == 1 else 0 (for the bsums term) */
static inline void iq6_decode_block(const block_iq6_k *x, const iq6_luts_t *L,
                                    __m256i w[8], __m256i sp[8], __m256i *smin) {
    const __m256i m0f = _mm256_set1_epi8(0x0f);
    const int8_t *sl = x->scales;

    for (int ih = 0; ih < 2; ++ih) {
        const __m256i qh = _mm256_loadu_si256((const __m256i *)(x->qh + 32 * ih));
        for (int sh = 0; sh < 2; ++sh) {
            const int ib = 2 * ih + sh;          /* 64-value sub-block */
            const int s0 = 4 * sh;               /* qh bit offset */
            const __m256i qs = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * ib));
            const __m256i lo = _mm256_and_si256(qs, m0f);                       /* values 64ib+0..31  */
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qs, 4), m0f); /* values 64ib+32..63 */

            /* low nibbles use qh bits (s0, s0+1); high nibbles use (s0+2, s0+3) */
            w[2 * ib + 0] = iq6_lookup(lo, qh, _mm_cvtsi32_si128(7 - s0), _mm_cvtsi32_si128(6 - s0), L);
            w[2 * ib + 1] = iq6_lookup(hi, qh, _mm_cvtsi32_si128(5 - s0), _mm_cvtsi32_si128(4 - s0), L);

            sp[2 * ib + 0] = iq6_scale_pair(sl[4 * ib + 0], sl[4 * ib + 1]);
            sp[2 * ib + 1] = iq6_scale_pair(sl[4 * ib + 2], sl[4 * ib + 3]);
        }
    }

    {
        const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)sl));
        const __m256i bit  = _mm256_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128,
                                               256, 512, 1024, 2048, 4096, 8192, 16384, (short)0x8000);
        const __m256i ex   = _mm256_set1_epi16((short)x->extra);
        const __m256i sel  = _mm256_cmpeq_epi16(_mm256_and_si256(ex, bit), bit);
        *smin = _mm256_and_si256(sc16, sel);
    }
}

/* int8 x int8 -> int32x8 with per-16-group scale applied. */
static inline __m256i iq6_mac(__m256i val, __m256i y, __m256i scale16) {
    const __m256i a = _mm256_sign_epi8(val, val);   /* |val| */
    const __m256i s = _mm256_sign_epi8(y, val);     /* y * sign(val) */
    const __m256i p = _mm256_maddubs_epi16(a, s);   /* 16x int16 pair sums */
    return _mm256_madd_epi16(p, scale16);           /* 8x int32, scaled */
}

/* Integer dot of one decoded block against one Q8_K block (scale not applied). */
static inline __m256i iq6_block_dot(const __m256i w[8], const __m256i sp[8], __m256i smin,
                                    const block_q8_K *yb) {
    __m256i acc = _mm256_madd_epi16(smin, _mm256_loadu_si256((const __m256i *)yb->bsums));
    for (int k = 0; k < 8; ++k) {
        acc = _mm256_add_epi32(acc,
              iq6_mac(w[k], _mm256_loadu_si256((const __m256i *)(yb->qs + 32 * k)), sp[k]));
    }
    return acc;
}

void vec_dot_iq6_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
    const block_iq6_k *x = (const block_iq6_k *)vx;
    const block_q8_K *y = (const block_q8_K *)wy;
    const int nb = n / QK_K;
    const iq6_luts_t L = iq6_luts_init();

    __m256 fsum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        __m256i w[8], sp[8], smin;
        iq6_decode_block(&x[ibl], &L, w, sp, &smin);
        const __m256i acc = iq6_block_dot(w, sp, smin, &y[ibl]);
        const float dd = iq6_f16(x[ibl].d) * y[ibl].d;
        fsum = _mm256_add_ps(fsum, _mm256_mul_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(dd)));
    }
    *out = iq6_hsum_ps(fsum);
}

/* One weight row x ncols activation rows. Activation row c starts at
 * (const char *)wy + c * y_stride (bytes). out[c] receives the dot product. */
void vec_dot_iq6_k_q8_k_avx2_batch(const void *vx, const void *wy, size_t y_stride,
                                   int n, int ncols, float *out) {
    while (ncols > IQ6K_BATCH_TILE) {
        vec_dot_iq6_k_q8_k_avx2_batch(vx, wy, y_stride, n, IQ6K_BATCH_TILE, out);
        wy = (const char *)wy + (size_t)IQ6K_BATCH_TILE * y_stride;
        out += IQ6K_BATCH_TILE;
        ncols -= IQ6K_BATCH_TILE;
    }
    if (ncols <= 0) return;

    const block_iq6_k *x = (const block_iq6_k *)vx;
    const int nb = n / QK_K;
    const iq6_luts_t L = iq6_luts_init();

    __m256 fs[IQ6K_BATCH_TILE];
    for (int c = 0; c < ncols; ++c) fs[c] = _mm256_setzero_ps();

    for (int ibl = 0; ibl < nb; ++ibl) {
        __m256i w[8], sp[8], smin;
        iq6_decode_block(&x[ibl], &L, w, sp, &smin);
        const float d = iq6_f16(x[ibl].d);
        for (int c = 0; c < ncols; ++c) {
            const block_q8_K *yb = (const block_q8_K *)((const char *)wy + (size_t)c * y_stride) + ibl;
            const __m256i acc = iq6_block_dot(w, sp, smin, yb);
            fs[c] = _mm256_add_ps(fs[c],
                    _mm256_mul_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(d * yb->d)));
        }
    }
    for (int c = 0; c < ncols; ++c) out[c] = iq6_hsum_ps(fs[c]);
}

#else  /* !__AVX2__ : never called, callers guard on __AVX2__ */

void vec_dot_iq6_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
    (void)vx; (void)wy; (void)n;
    *out = 0.0f;
}

void vec_dot_iq6_k_q8_k_avx2_batch(const void *vx, const void *wy, size_t y_stride,
                                   int n, int ncols, float *out) {
    (void)vx; (void)wy; (void)y_stride; (void)n;
    for (int c = 0; c < ncols; ++c) out[c] = 0.0f;
}

#endif

/* ================================================================
 * AVX-512 VNNI path: vpdpbusd (unsigned-signed, 32-bit accum)
 * ================================================================
 * dpbusd is unsigned-signed without saturation. The LUT values
 * (0..249) are naturally unsigned. We multiply with signed y directly.
 *
 *   result = sum_i(lut[q_i] * y_i) + (mn - 128) * bsums[g]
 *
 * This eliminates the sign trick entirely.
 * ================================================================ */
#if defined(__AVX512F__) && defined(__AVX512VNNI__)

/* iq6_lut_select reused from AVX2 path (128-bit only) */
static inline __m128i iq6_lut_select(__m128i q6, __m128i lut0, __m128i lut1, __m128i lut2, __m128i lut3) {
    const __m128i m0f = _mm_set1_epi8(0x0f);
    const __m128i m80 = _mm_set1_epi8((char)0x80);
    __m128i low4 = _mm_and_si128(q6, m0f);

    __m128i v0 = _mm_shuffle_epi8(lut0, low4);
    __m128i v1 = _mm_shuffle_epi8(lut1, low4);
    __m128i v2 = _mm_shuffle_epi8(lut2, low4);
    __m128i v3 = _mm_shuffle_epi8(lut3, low4);

    __m128i hi2 = _mm_and_si128(_mm_srli_epi16(q6, 4), _mm_set1_epi8(0x03));
    __m128i mask0 = _mm_and_si128(_mm_slli_epi16(hi2, 7), m80);
    __m128i mask1 = _mm_and_si128(_mm_slli_epi16(hi2, 6), m80);

    __m128i v_lo = _mm_blendv_epi8(v0, v1, mask0);
    __m128i v_hi = _mm_blendv_epi8(v2, v3, mask0);
    return _mm_blendv_epi8(v_lo, v_hi, mask1);
}

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

            __m128i val1 = iq6_lut_select(q1, lut0, lut1, lut2, lut3);
            __m128i val2 = iq6_lut_select(q2, lut0, lut1, lut2, lut3);
            __m128i val3 = iq6_lut_select(q3, lut0, lut1, lut2, lut3);
            __m128i val4 = iq6_lut_select(q4, lut0, lut1, lut2, lut3);

            __m128i y0 = _mm_loadu_si128((const __m128i *)(q8 + 0));
            __m128i y1 = _mm_loadu_si128((const __m128i *)(q8 + 16));
            __m128i y2 = _mm_loadu_si128((const __m128i *)(q8 + 32));
            __m128i y3 = _mm_loadu_si128((const __m128i *)(q8 + 48));

            __m128i a0 = _mm_dpbusd_epi32(_mm_setzero_si128(), val1, y0);
            __m128i a1 = _mm_dpbusd_epi32(_mm_setzero_si128(), val2, y1);
            __m128i a2 = _mm_dpbusd_epi32(_mm_setzero_si128(), val3, y2);
            __m128i a3 = _mm_dpbusd_epi32(_mm_setzero_si128(), val4, y3);

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

