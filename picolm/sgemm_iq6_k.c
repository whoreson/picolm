/* ================================================================
 * IQ6_K (GGUF type 141) x Q8_K AVX2 + VNNI kernels
 * ================================================================
 * Layout per 256-value block (see quant.h block_iq6_k):
 *   value k of group g = 16-value group, g = k / 16
 *   w = iq6nl_lut[q6] - 128 + m_g,  q6 = lo4 | hi2 << 4,  m_g = bit g of extra
 *   result = d * sum_g scale_g * sum_{k in g} w_k * y_k
 *
 * AVX2: decode-once with pre-biased LUT (lut - 128 as int8).
 *       sign trick + maddubs + madd_epi16(scale16) for int32 accum.
 *       smin * bsums for the m_g correction.
 *
 * VNNI: decode-once with unsigned LUT (lut as uint8, 1..249).
 *       dpbusd (uint8 * int8 -> int32, no saturation).
 *       Per-group: hsum 4 dpbusd int32 -> 1 int32 per group.
 *       Correction: (mn[g] - 128) * bsums[g].
 *       Scale: float multiply per group (16 groups, scalar loop).
 * ================================================================ */

#include <stdint.h>
#include <stddef.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(PICOLM_NEON)
#include <arm_neon.h>
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

static inline __m256i iq6_lookup(__m256i low4, __m256i qh, __m128i c0, __m128i c1,
                                 const iq6_luts_t *L) {
    const __m256i v0 = _mm256_shuffle_epi8(L->l0, low4);
    const __m256i v1 = _mm256_shuffle_epi8(L->l1, low4);
    const __m256i v2 = _mm256_shuffle_epi8(L->l2, low4);
    const __m256i v3 = _mm256_shuffle_epi8(L->l3, low4);
    const __m256i m0 = _mm256_sll_epi16(qh, c0);
    const __m256i m1 = _mm256_sll_epi16(qh, c1);
    const __m256i lo = _mm256_blendv_epi8(v0, v1, m0);
    const __m256i hi = _mm256_blendv_epi8(v2, v3, m0);
    return _mm256_blendv_epi8(lo, hi, m1);
}

static inline __m256i iq6_scale_pair(int a, int b) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16((short)a)),
                                   _mm_set1_epi16((short)b), 1);
}

static inline void iq6_decode_block(const block_iq6_k *x, const iq6_luts_t *L,
                                    __m256i w[8], __m256i sp[8], __m256i *smin) {
    const __m256i m0f = _mm256_set1_epi8(0x0f);
    const int8_t *sl = x->scales;

    for (int ih = 0; ih < 2; ++ih) {
        const __m256i qh = _mm256_loadu_si256((const __m256i *)(x->qh + 32 * ih));
        for (int sh = 0; sh < 2; ++sh) {
            const int ib = 2 * ih + sh;
            const int s0 = 4 * sh;
            const __m256i qs = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * ib));
            const __m256i lo = _mm256_and_si256(qs, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qs, 4), m0f);

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

static inline __m256i iq6_mac(__m256i val, __m256i y, __m256i scale16) {
    const __m256i a = _mm256_sign_epi8(val, val);
    const __m256i s = _mm256_sign_epi8(y, val);
    const __m256i p = _mm256_maddubs_epi16(a, s);
    return _mm256_madd_epi16(p, scale16);
}

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

/* ================================================================
 * IQ6_K x Q8_K tiled GEMM (AVX2)
 * ================================================================
 * C[nrows x ncols] = W[nrows x k] * A[ncols x k]^T
 * W = IQ6_K weights, A = Q8_K activations (pre-quantized).
 *
 * Tiling: 1 weight row x 2 activation columns per tile.
 * Thread distribution: tiles = nrows * (ncols/2), work = tiles/nth.
 *
 * Each tile: decode weight row once per block, dot against 2 activations.
 * ================================================================ */
int sgemm_iq6_k_q8_k_avx2(int nrows, int ncols, int k,
                           const void *vx, const void *vy,
                           float *out, size_t bs,
                           int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k % QK_K != 0)
        return 0;

    const int nb = k / QK_K;
    const iq6_luts_t L = iq6_luts_init();

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

    size_t w_row_bytes = nb * sizeof(block_iq6_k);
    size_t a_row_bytes = nb * sizeof(block_q8_K);

    for (int64_t job = start; job < end; job++) {
        int64_t ii = job / xtiles_ext;
        int64_t xt = job % xtiles_ext;
        int64_t jj = xt * 2;
        int64_t ncols_tile = (xt < xtiles) ? 2 : n_tail;
        if (ncols_tile < 1) ncols_tile = 1;

        const block_iq6_k *w = (const block_iq6_k *)
            ((const char *)vx + ii * w_row_bytes);

        float acc[2] = { 0.0f, 0.0f };

        const block_q8_K *qk_ptr[2];
        for (int c = 0; c < ncols_tile; c++) {
            qk_ptr[c] = (const block_q8_K *)
                ((const char *)vy + (jj + c) * a_row_bytes);
        }

        __m256 fsum[2];
        fsum[0] = _mm256_setzero_ps();
        fsum[1] = _mm256_setzero_ps();

        for (int ibl = 0; ibl < nb; ++ibl) {
            __m256i wv[8], sp[8], smin;
            iq6_decode_block(&w[ibl], &L, wv, sp, &smin);
            const float d = iq6_f16(w[ibl].d);

            for (int c = 0; c < ncols_tile; c++) {
                const __m256i acc_vec = iq6_block_dot(wv, sp, smin, &qk_ptr[c][ibl]);
                const float dd = d * qk_ptr[c][ibl].d;
                fsum[c] = _mm256_add_ps(fsum[c],
                        _mm256_mul_ps(_mm256_cvtepi32_ps(acc_vec), _mm256_set1_ps(dd)));
            }
        }

        for (int c = 0; c < ncols_tile; c++) {
            out[ii + (jj + c) * bs] = iq6_hsum_ps(fsum[c]);
        }
    }
    return nrows;
}

#else

void vec_dot_iq6_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
    (void)vx; (void)wy; (void)n; *out = 0.0f;
}
void vec_dot_iq6_k_q8_k_avx2_batch(const void *vx, const void *wy, size_t y_stride,
                                   int n, int ncols, float *out) {
    (void)vx; (void)wy; (void)y_stride; (void)n;
    for (int c = 0; c < ncols; ++c) out[c] = 0.0f;
}

#endif

/* ================================================================
 * AVX-512 VNNI path: decode-once + dpbusd, per-group scalar scale
 * ================================================================
 * Structure matches AVX2 decode-once, but uses unsigned LUT + dpbusd.
 *
 * Per group g (16 values):
 *   dp[g] = sum of 4 dpbusd calls (each covering 4 byte-pairs)
 *   result[g] = dp[g] + (mn[g] - 128) * bsums[g]
 *   final += scale[g] * result[g] * d * yb->d
 *
 * This is correct because:
 *   scalar: sum((lut[q] - 128 + mn) * y) * scale
 *          = (sum(lut[q]*y) - 128*sum(y) + mn*sum(y)) * scale
 *          = (dpbusd(lut,y) + (mn-128)*bsums) * scale
 *
 * dpbusd is unsigned-signed: lut (uint8, 1..249) * y (int8, -128..127).
 * This is exactly what we need since lut values are positive.
 * ================================================================ */
#if defined(__AVX512F__) && defined(__AVX512VNNI__)

typedef struct { __m256i l0, l1, l2, l3; } iq6_luts_vnni_t;

static inline iq6_luts_vnni_t iq6_luts_vnni_init(void) {
    iq6_luts_vnni_t L;
    L.l0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq6nl_lut +  0)));
    L.l1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq6nl_lut + 16)));
    L.l2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq6nl_lut + 32)));
    L.l3 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(iq6nl_lut + 48)));
    return L;
}

static inline __m256i iq6_lookup_vnni(__m256i low4, __m256i qh, __m128i c0, __m128i c1,
                                      const iq6_luts_vnni_t *L) {
    const __m256i v0 = _mm256_shuffle_epi8(L->l0, low4);
    const __m256i v1 = _mm256_shuffle_epi8(L->l1, low4);
    const __m256i v2 = _mm256_shuffle_epi8(L->l2, low4);
    const __m256i v3 = _mm256_shuffle_epi8(L->l3, low4);
    const __m256i m0 = _mm256_sll_epi16(qh, c0);
    const __m256i m1 = _mm256_sll_epi16(qh, c1);
    const __m256i lo = _mm256_blendv_epi8(v0, v1, m0);
    const __m256i hi = _mm256_blendv_epi8(v2, v3, m0);
    return _mm256_blendv_epi8(lo, hi, m1);
}

/* Decode one block: 8 x __m256i unsigned weights. */
static inline void iq6_decode_block_vnni(const block_iq6_k *x, const iq6_luts_vnni_t *L,
                                          __m256i w[8]) {
    const __m256i m0f = _mm256_set1_epi8(0x0f);

    for (int ih = 0; ih < 2; ++ih) {
        const __m256i qh = _mm256_loadu_si256((const __m256i *)(x->qh + 32 * ih));
        for (int sh = 0; sh < 2; ++sh) {
            const int ib = 2 * ih + sh;
            const int s0 = 4 * sh;
            const __m256i qs = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * ib));
            const __m256i lo = _mm256_and_si256(qs, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qs, 4), m0f);

            w[2 * ib + 0] = iq6_lookup_vnni(lo, qh, _mm_cvtsi32_si128(7 - s0), _mm_cvtsi32_si128(6 - s0), L);
            w[2 * ib + 1] = iq6_lookup_vnni(hi, qh, _mm_cvtsi32_si128(5 - s0), _mm_cvtsi32_si128(4 - s0), L);
        }
    }
}

/* VNNI dot: one decoded block against one Q8_K block.
 * Writes 16 int32 to grp (one per 16-value group), including correction. */
static inline void iq6_block_dot_vnni(const __m256i w[8], uint16_t extra,
                                       const block_q8_K *yb, int32_t grp[16]) {
    const int16_t *bsums16 = yb->bsums;
    const int8_t *q8 = yb->qs;

    /* Each w[k] covers 32 values = 2 groups of 16.
     * w[k] is 32 bytes: low 16 bytes = group 2k, high 16 bytes = group 2k+1.
     * dpbusd on 128-bit: 16 uint8 * 16 int8 -> 4 int32.
     * hsum of 4 int32 -> 1 int32 per group. */
    for (int k = 0; k < 8; ++k) {
        const __m128i w_lo = _mm256_castsi256_si128(w[k]);
        const __m128i w_hi = _mm256_extracti128_si256(w[k], 1);
        const __m128i y_lo = _mm_loadu_si128((const __m128i *)(q8 + 32 * k));
        const __m128i y_hi = _mm_loadu_si128((const __m128i *)(q8 + 32 * k + 16));

        __m128i dp_lo = _mm_dpbusd_epi32(_mm_setzero_si128(), w_lo, y_lo);
        __m128i dp_hi = _mm_dpbusd_epi32(_mm_setzero_si128(), w_hi, y_hi);

        /* hsum of 4 int32 -> 1 int32 */
        __m128i h_lo = _mm_add_epi32(dp_lo, _mm_shuffle_epi32(dp_lo, 0x1B));
        h_lo = _mm_add_epi32(h_lo, _mm_shuffle_epi32(h_lo, 0x4E));
        __m128i h_hi = _mm_add_epi32(dp_hi, _mm_shuffle_epi32(dp_hi, 0x1B));
        h_hi = _mm_add_epi32(h_hi, _mm_shuffle_epi32(h_hi, 0x4E));

        grp[2 * k]     = _mm_cvtsi128_si32(h_lo);
        grp[2 * k + 1] = _mm_cvtsi128_si32(h_hi);
    }

    /* Apply correction: (mn[g] - 128) * bsums[g] */
    for (int g = 0; g < 16; g++) {
        int mn = (extra >> g) & 1;
        grp[g] += (mn - 128) * bsums16[g];
    }
}

void vec_dot_iq6_k_q8_k_vnni(const void *vx, const void *wy, int n, float *out) {
    const block_iq6_k *x = (const block_iq6_k *)vx;
    const block_q8_K *y = (const block_q8_K *)wy;
    const int nb = n / QK_K;
    const iq6_luts_vnni_t L = iq6_luts_vnni_init();

    float sumf = 0.0f;
    for (int ibl = 0; ibl < nb; ++ibl) {
        __m256i w[8];
        iq6_decode_block_vnni(&x[ibl], &L, w);
        int32_t grp[16];
        iq6_block_dot_vnni(w, x[ibl].extra, &y[ibl], grp);

        const float dd = fp16_to_fp32(x[ibl].d) * y[ibl].d;
        const int8_t *sl = x[ibl].scales;
        for (int g = 0; g < 16; g++) {
            sumf += dd * sl[g] * grp[g];
        }
    }
    *out = sumf;
}

void vec_dot_iq6_k_q8_k_vnni_batch(const void *vx, const void *wy, size_t y_stride,
                                    int n, int ncols, float *out) {
    while (ncols > IQ6K_BATCH_TILE) {
        vec_dot_iq6_k_q8_k_vnni_batch(vx, wy, y_stride, n, IQ6K_BATCH_TILE, out);
        wy = (const char *)wy + (size_t)IQ6K_BATCH_TILE * y_stride;
        out += IQ6K_BATCH_TILE;
        ncols -= IQ6K_BATCH_TILE;
    }
    if (ncols <= 0) return;

    const block_iq6_k *x = (const block_iq6_k *)vx;
    const int nb = n / QK_K;
    const iq6_luts_vnni_t L = iq6_luts_vnni_init();

    float fs[IQ6K_BATCH_TILE] = {0};

    for (int ibl = 0; ibl < nb; ++ibl) {
        __m256i w[8];
        iq6_decode_block_vnni(&x[ibl], &L, w);
        const float d = fp16_to_fp32(x[ibl].d);
        const int8_t *sl = x[ibl].scales;
        for (int c = 0; c < ncols; ++c) {
            const block_q8_K *yb = (const block_q8_K *)((const char *)wy + (size_t)c * y_stride) + ibl;
            int32_t grp[16];
            iq6_block_dot_vnni(w, x[ibl].extra, yb, grp);
            const float dd = d * yb->d;
            for (int g = 0; g < 16; g++) {
                fs[c] += dd * sl[g] * grp[g];
            }
        }
    }
    for (int c = 0; c < ncols; ++c) out[c] = fs[c];
}
#endif

/* ================================================================
 * ARM NEON path: decode-once + per-group int8 MAC
 * ================================================================ */
#if defined(PICOLM_NEON)

static inline void iq6_decode_block_neon(const block_iq6_k *x, int8_t w[256]) {
    const uint8_t *qs = x->qs;
    const uint8_t *qh = x->qh;
    uint16_t extra = x->extra;
    for (int v = 0; v < 256; v++) {
        int chunk64 = v / 64;
        int chunk2  = v / 128;
        int shift   = ((v / 64) % 2 == 0) ? 0 : 4;

        int byte_qs = chunk64 * 32 + (v % 32);
        int nibble  = (v / 32) & 1;
        int low4    = (qs[byte_qs] >> (4 * nibble)) & 0xf;

        int byte_qh  = chunk2 * 32 + (v % 32);
        int bit_pair = shift + ((v / 32) & 1) * 2;
        int high2    = (qh[byte_qh] >> bit_pair) & 0x3;

        int q6 = low4 | (high2 << 4);
        int g  = v / 16;
        int mn = (extra >> g) & 1;
        w[v] = (int8_t)((int)iq6nl_lut[q6] - 128 + mn);
    }
}

static inline int32_t iq6_dot_group_neon(const int8_t w[16], const int8_t a[16]) {
#if defined(__ARM_FEATURE_MATMUL_INT8) || defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    const int8x16_t wv = vld1q_s8(w);
    const int8x16_t av = vld1q_s8(a);
    int32x4_t s = vmmlaq_s32(vdupq_n_s32(0), wv, av);
    return vaddvq_s32(s);
#else
    const int8x16_t wv = vld1q_s8(w);
    const int8x16_t av = vld1q_s8(a);
    int16x8_t p0 = vmull_s8(vget_low_s8(wv), vget_low_s8(av));
    int16x8_t p1 = vmull_s8(vget_high_s8(wv), vget_high_s8(av));
    int32x4_t s = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
    return vaddvq_s32(s);
#endif
}

void vec_dot_iq6_k_q8_k_neon(const void *vx, const void *wy, int n, float *out) {
    const block_iq6_k *x = (const block_iq6_k *)vx;
    const block_q8_K *y = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int ibl = 0; ibl < nb; ++ibl) {
        int8_t w[256];
        iq6_decode_block_neon(&x[ibl], w);
        const float dd = fp16_to_fp32_lookup(x[ibl].d) * y[ibl].d;
        const int8_t *q8 = y[ibl].qs;
        const int8_t *sl = x[ibl].scales;

        int32_t block_sum = 0;
        for (int g = 0; g < 16; g++) {
            block_sum += sl[g] * iq6_dot_group_neon(w + g * 16, q8 + g * 16);
        }
        sumf += dd * (float)block_sum;
    }
    *out = sumf;
}

void vec_dot_iq6_k_q8_k_neon_batch(const void *vx, const void *wy, size_t y_stride,
                                   int n, int ncols, float *out) {
    const block_iq6_k *x = (const block_iq6_k *)vx;
    const int nb = n / QK_K;

    for (int c = 0; c < ncols; c++) out[c] = 0.0f;

    for (int ibl = 0; ibl < nb; ++ibl) {
        int8_t w[256];
        iq6_decode_block_neon(&x[ibl], w);
        const float d = fp16_to_fp32_lookup(x[ibl].d);
        const int8_t *sl = x[ibl].scales;
        for (int c = 0; c < ncols; c++) {
            const block_q8_K *yb = (const block_q8_K *)((const char *)wy + (size_t)c * y_stride) + ibl;
            const float dd = d * yb->d;
            const int8_t *q8 = yb->qs;
            int32_t block_sum = 0;
            for (int g = 0; g < 16; g++) {
                block_sum += sl[g] * iq6_dot_group_neon(w + g * 16, q8 + g * 16);
            }
            out[c] += dd * (float)block_sum;
        }
    }
}

int sgemm_iq6_k_q8_k_neon(int nrows, int ncols, int k,
                          const void *vx, const void *vy,
                          float *out, size_t bs,
                          int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k % QK_K != 0)
        return 0;

    const int nb = k / QK_K;
    const size_t w_row_bytes = (size_t)nb * sizeof(block_iq6_k);
    const size_t a_row_bytes = (size_t)nb * sizeof(block_q8_K);

    int64_t tiles = (int64_t)nrows * ncols;
    int64_t duty = (tiles + nth - 1) / nth;
    int64_t start = duty * ith;
    int64_t end = start + duty;
    if (end > tiles) end = tiles;

    for (int64_t t = start; t < end; t++) {
        int i = (int)(t / ncols);
        int j = (int)(t % ncols);
        const void *wrow = (const char *)vx + (size_t)i * w_row_bytes;
        const void *acol = (const char *)vy + (size_t)j * a_row_bytes;
        float result;
        vec_dot_iq6_k_q8_k_neon(wrow, acol, k, &result);
        out[i + (size_t)j * bs] = result;
    }
    return nrows;
}

#endif /* PICOLM_NEON */

