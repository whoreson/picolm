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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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


/* ================================================================
 * Decode-once R8 GEMM (prefill): sgemm_iq6_k_q8_k_r8t
 *
 * The tile kernel above handles 1 weight row x 2 activation columns and re-decodes every
 * IQ6_K block (the 64-entry LUT lookups) for each tile: with 500+ prompt columns each block
 * is decoded ~250 times, and the decode costs more than the dot products. Here a group of 8
 * weight rows is decoded ONCE per column panel into an int8 R8 scratch:
 *   - exact signed weights w = iq6nl_lut[q6] - 128 + m_g (always fits int8, |w| <= 127),
 *     laid out like Q8_K_R8: vector v (0..63 per 256-value block) = dword v of each of the
 *     8 rows, so one 32-byte vector carries 4 consecutive values of 8 rows;
 *   - the per-16-value-group int8 scales as int16 pairs [s_r, s_r] per row, so ONE
 *     madd_epi16 applies them while summing the maddubs pairs (no correction terms, no
 *     bsums: the +m_g offset is folded into the weights);
 *   - 8 fp32 row scales per block.
 * A tile of NC columns then streams the whole scratch: per 32 MACs sign + maddubs + madd +
 * add, abs(w) shared by the NC columns, one cvt/mul/fma fixup per 256-value block and
 * column. Q8_K activations are read in place (no staging); column panels of IQ6T_PC columns
 * keep them in L2 while a thread sweeps its weight groups. Results match the tile kernel
 * (identical integer sums; only fp32 accumulation order differs).
 * ================================================================ */
#if defined(__AVX512VL__)
#define IQ6T_NCMAX 8
#else
#define IQ6T_NCMAX 4
#endif
#define IQ6T_PC 96
#if defined(__FMA__)
#define IQ6T_FMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#define IQ6T_FNMA(a, b, c) _mm256_fnmadd_ps((a), (b), (c))
#else
#define IQ6T_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#define IQ6T_FNMA(a, b, c) _mm256_sub_ps((c), _mm256_mul_ps((a), (b)))
#endif

typedef struct { int8_t *w; int16_t *sc; float *d; size_t w_cap, sc_cap, d_cap; } iq6t_scratch_t;
#if !defined(_WIN32)
static __thread iq6t_scratch_t iq6t_tl;
#else
static iq6t_scratch_t iq6t_tl;
#endif

static int iq6t_reserve(int nb) {
    const size_t w = (size_t)nb * 2048, sc = (size_t)nb * 512, d = (size_t)nb * 32;
    if (iq6t_tl.w_cap < w)   { _mm_free(iq6t_tl.w);  iq6t_tl.w  = (int8_t *)_mm_malloc(w, 64);  iq6t_tl.w_cap  = iq6t_tl.w  ? w  : 0; }
    if (iq6t_tl.sc_cap < sc) { _mm_free(iq6t_tl.sc); iq6t_tl.sc = (int16_t *)_mm_malloc(sc, 64); iq6t_tl.sc_cap = iq6t_tl.sc ? sc : 0; }
    if (iq6t_tl.d_cap < d)   { _mm_free(iq6t_tl.d);  iq6t_tl.d  = (float *)_mm_malloc(d, 64);   iq6t_tl.d_cap  = iq6t_tl.d  ? d  : 0; }
    return iq6t_tl.w && iq6t_tl.sc && iq6t_tl.d;
}

/* 8x8 transpose of 32-bit words: o[j] = dword j of each of r[0..7] */
static inline void iq6t_transpose8(const __m256i r[8], __m256i o[8]) {
    const __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]), t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    const __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]), t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    const __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]), t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    const __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]), t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    const __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
    const __m256i u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
    const __m256i u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6);
    const __m256i u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
    o[0] = _mm256_permute2x128_si256(u0, u4, 0x20); o[1] = _mm256_permute2x128_si256(u1, u5, 0x20);
    o[2] = _mm256_permute2x128_si256(u2, u6, 0x20); o[3] = _mm256_permute2x128_si256(u3, u7, 0x20);
    o[4] = _mm256_permute2x128_si256(u0, u4, 0x31); o[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    o[6] = _mm256_permute2x128_si256(u2, u6, 0x31); o[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}

/* One block's 256 weights as 8 vectors of 32 values in natural order, m_g folded in. */
static inline void iq6t_decode_w(const block_iq6_k *x, const iq6_luts_t *L, __m256i w[8]) {
    const __m256i m0f = _mm256_set1_epi8(0x0f);
    for (int ih = 0; ih < 2; ++ih) {
        const __m256i qh = _mm256_loadu_si256((const __m256i *)(x->qh + 32 * ih));
        for (int sh = 0; sh < 2; ++sh) {
            const int ib = 2 * ih + sh, s0 = 4 * sh;
            const __m256i qs = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * ib));
            const __m256i lo = _mm256_and_si256(qs, m0f);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qs, 4), m0f);
            w[2 * ib + 0] = iq6_lookup(lo, qh, _mm_cvtsi32_si128(7 - s0), _mm_cvtsi32_si128(6 - s0), L);
            w[2 * ib + 1] = iq6_lookup(hi, qh, _mm_cvtsi32_si128(5 - s0), _mm_cvtsi32_si128(4 - s0), L);
        }
    }
    const unsigned ex = x->extra;
    for (int k = 0; k < 8; ++k) {   /* vector k = groups 2k (low lane) and 2k+1 (high lane) */
        const __m256i mn = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi8((char)((ex >> (2 * k)) & 1))),
                                                   _mm_set1_epi8((char)((ex >> (2 * k + 1)) & 1)), 1);
        w[k] = _mm256_add_epi8(w[k], mn);
    }
}

static void iq6t_decode_group(const char *wbase, size_t w_row_bytes, int nb, const iq6_luts_t *L) {
    for (int ib = 0; ib < nb; ++ib) {
        __m256i wv[8][8];
        float dd[8];
        int16_t *scb = iq6t_tl.sc + (size_t)ib * 256;
        for (int r = 0; r < 8; ++r) {
            const block_iq6_k *x = (const block_iq6_k *)(wbase + (size_t)r * w_row_bytes) + ib;
            iq6t_decode_w(x, L, wv[r]);
            dd[r] = iq6_f16(x->d);
            for (int g = 0; g < 16; ++g) {
                const int16_t sl = (int16_t)x->scales[g];
                scb[g * 16 + 2 * r] = sl;
                scb[g * 16 + 2 * r + 1] = sl;
            }
        }
        int8_t *wb = iq6t_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i rr[8], o[8];
            for (int r = 0; r < 8; ++r) rr[r] = wv[r][k];
            iq6t_transpose8(rr, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), o[j]);
        }
        _mm256_storeu_ps(iq6t_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}

/* ---- IQ4_XS source for the same scratch ----
 * An IQ4_XS super-block is 8 sub-blocks of 32 values: sub-block k's qs[16k..16k+15] hold values
 * 32k+0..15 (low nibbles) and 32k+16..31 (high nibbles), mapped through kvalues_iq4nl (int8, exact),
 * with ONE 6-bit scale (ls-32) per sub-block. That is two identical 16-value groups in the IQ6_K
 * scratch (int8 weights as R8 vectors, per-16-group scales, 8 fp32 row scales), so only the
 * decode differs and both compute kernels below are reused unchanged. */
static inline always_inline void iq4xs_load_wv(const block_iq4_xs *x, __m256i wv[8]) {
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)kvalues_iq4nl));
    const __m128i m15 = _mm_set1_epi8(15);
    for (int k = 0; k < 8; ++k) {
        const __m128i v = _mm_loadu_si128((const __m128i *)(x->qs + 16 * k));
        const __m128i lo = _mm_and_si128(v, m15), hi = _mm_and_si128(_mm_srli_epi16(v, 4), m15);
        wv[k] = _mm256_shuffle_epi8(lut, _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1));
    }
}
static inline int iq4xs_subscale(const block_iq4_xs *x, int k) {
    return (int)(((x->scales_l[k >> 1] >> (4 * (k & 1))) & 0xf) | (((x->scales_h >> (2 * k)) & 3) << 4)) - 32;
}

static void iq4xs_t_decode_group(const char *wbase, size_t w_row_bytes, int nb) {
    for (int ib = 0; ib < nb; ++ib) {
        __m256i wv[8][8];
        float dd[8];
        int16_t *scb = iq6t_tl.sc + (size_t)ib * 256;
        for (int r = 0; r < 8; ++r) {
            const block_iq4_xs *x = (const block_iq4_xs *)(wbase + (size_t)r * w_row_bytes) + ib;
            iq4xs_load_wv(x, wv[r]);
            dd[r] = iq6_f16(x->d);
            for (int k = 0; k < 8; ++k) {
                const int16_t sl = (int16_t)iq4xs_subscale(x, k);
                for (int gg = 0; gg < 2; ++gg) {
                    scb[(2 * k + gg) * 16 + 2 * r] = sl;
                    scb[(2 * k + gg) * 16 + 2 * r + 1] = sl;
                }
            }
        }
        int8_t *wb = iq6t_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i rr[8], o[8];
            for (int r = 0; r < 8; ++r) rr[r] = wv[r][k];
            iq6t_transpose8(rr, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), o[j]);
        }
        _mm256_storeu_ps(iq6t_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}

/* ---- Q6_K / Q3_K sources for the same scratch ----
 * Both are symmetric (no mins): a signed int8 value per weight and one integer scale per 16 values,
 * i.e. exactly the scratch the IQ6_K kernels consume, so only the decode differs (kinds 2 and 3).
 *   Q6_K : q = ((ql nibble) | (2 high bits << 4)) - 32, scale = int8 scales[g], natural 16-groups.
 *   Q3_K : q = (2 low bits | (hmask bit << 2)) - 4,     scale = 6-bit packed scale - 32. */
enum { KQ_Q6K = 2, KQ_Q3K = 3, KQ_Q4K = 4, KQ_Q5K = 5, KQ_Q2K = 6, KQ_Q4KR4 = 7, KQ_Q6KR4 = 8, KQ_IQ4K = 9, KQ_IQ4KR4 = 10, KQ_IQ2K = 11, KQ_IQ3K = 12, KQ_IQ2KR4 = 13, KQ_IQ3KR4 = 14 };
static inline always_inline void kq_load_q6(const block_q6_K *x, __m256i wv[8], int8_t sc[16]) {
    const __m256i m0f = _mm256_set1_epi8(0x0F), m03 = _mm256_set1_epi8(0x03), m32 = _mm256_set1_epi8(32);
    for (int n = 0; n < 2; ++n) {                       /* two 128-value halves */
        const __m256i l0 = _mm256_loadu_si256((const __m256i *)(x->ql + 64 * n));
        const __m256i l1 = _mm256_loadu_si256((const __m256i *)(x->ql + 64 * n + 32));
        const __m256i h  = _mm256_loadu_si256((const __m256i *)(x->qh + 32 * n));
        const __m256i q1 = _mm256_or_si256(_mm256_and_si256(l0, m0f), _mm256_slli_epi16(_mm256_and_si256(h, m03), 4));
        const __m256i q2 = _mm256_or_si256(_mm256_and_si256(l1, m0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), m03), 4));
        const __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), m03), 4));
        const __m256i q4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), m03), 4));
        wv[4 * n + 0] = _mm256_sub_epi8(q1, m32); wv[4 * n + 1] = _mm256_sub_epi8(q2, m32);
        wv[4 * n + 2] = _mm256_sub_epi8(q3, m32); wv[4 * n + 3] = _mm256_sub_epi8(q4, m32);
    }
    for (int g = 0; g < 16; ++g) sc[g] = x->scales[g];
}
static inline always_inline void kq_load_q3(const block_q3_K *x, __m256i wv[8], int8_t sc[16]) {
    const __m256i m03 = _mm256_set1_epi8(0x03), one = _mm256_set1_epi8(1), four = _mm256_set1_epi8(4);
    const __m256i hm = _mm256_loadu_si256((const __m256i *)x->hmask);
    for (int n = 0; n < 2; ++n) {
        const __m256i q = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * n));
        for (int j = 0; j < 4; ++j) {
            const int k = 4 * n + j;
            __m256i lo = _mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m03);
            __m256i hb = _mm256_and_si256(_mm256_srli_epi16(hm, k), one);
            wv[k] = _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hb, 2)), four);
        }
    }
    uint32_t aux[4]; memcpy(aux, x->scales, 12);
    const uint32_t k1 = 0x03030303u, k2 = 0x0f0f0f0fu, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4);
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    const int8_t *a = (const int8_t *)aux;
    for (int g = 0; g < 16; ++g) sc[g] = (int8_t)((uint8_t)a[g] - 32);
}
/* ---- IQ4_K source (plain) ----
 * Per 32-value sub-block k: qs[16k..16k+15] low nibbles = values 0..15, high nibbles = 16..31, mapped
 * through a 16-entry int8 table chosen per half by one `extra` bit (iq4k_values: two tables of 16),
 * with an independent signed scale per half ((6 bits) - 32), i.e. natural 16-groups 2k and 2k+1.
 * VPSHUFB is lane-local, so ONE shuffle with [tableA | tableB] decodes both halves. No mins. */
static inline always_inline void kq_load_iq4k(const block_iq4_k *x, __m256i wv[8], int8_t sc[16]) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const unsigned extra = x->extra;
    for (int k = 0; k < 8; ++k) {
        const __m128i v = _mm_loadu_si128((const __m128i *)(x->qs + 16 * k));
        const __m128i lo = _mm_and_si128(v, _mm256_castsi256_si128(m0f)), hi = _mm_and_si128(_mm_srli_epi16(v, 4), _mm256_castsi256_si128(m0f));
        const __m256i idx = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        const __m128i ta = _mm_loadu_si128((const __m128i *)(iq4k_values + (((extra >> (2 * k)) & 1) ? 16 : 0)));
        const __m128i tb = _mm_loadu_si128((const __m128i *)(iq4k_values + (((extra >> (2 * k + 1)) & 1) ? 16 : 0)));
        wv[k] = _mm256_shuffle_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(ta), tb, 1), idx);
        const unsigned sh = (unsigned)x->scales_h[k >> 1] >> (4 * (k & 1));
        sc[2 * k]     = (int8_t)(((x->scales_l[k] & 0xf) | ((sh << 4) & 0x30)) - 32);
        sc[2 * k + 1] = (int8_t)(((x->scales_l[k] >> 4)  | ((sh << 2) & 0x30)) - 32);
    }
}

/* ---- IQ2_K / IQ3_K sources (plain) ----
 * Sub-block k (32 values): 2-bit fields qs[32*(k/4) + p] >> 2*(k%4) (p = 0..31; p<16 first half, p>=16
 * second half), plus for IQ3_K the third bit (qh[p] >> k) & 1; each half indexes its own table picked
 * by an `extra` bit (iq2nl_values: 2 x 4 entries, iq3nl_values: 2 x 8) and has its own signed scale:
 *   IQ2_K: (nibble) - 8        IQ3_K: (2*nibble + 1) * (sign bit ? -1 : 1).   No mins.
 * One lane-local VPSHUFB with [tableA | tableB] decodes both halves. */
static inline always_inline void kq_load_iq2k(const block_iq2_k *x, __m256i wv[8], int8_t sc[16]) {
    const __m256i m03 = _mm256_set1_epi8(0x03);
    const unsigned extra = x->extra;
    for (int k = 0; k < 8; ++k) {
        const __m256i q = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * (k >> 2)));
        const __m256i idx = _mm256_and_si256(_mm256_srl_epi16(q, _mm_cvtsi32_si128(2 * (k & 3))), m03);
        int ia, ib2; memcpy(&ia, iq2nl_values + (((extra >> (2 * k)) & 1) ? 4 : 0), 4); memcpy(&ib2, iq2nl_values + (((extra >> (2 * k + 1)) & 1) ? 4 : 0), 4);
        wv[k] = _mm256_shuffle_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_cvtsi32_si128(ia)), _mm_cvtsi32_si128(ib2), 1), idx);
        sc[2 * k] = (int8_t)((x->scales[k] & 0xf) - 8); sc[2 * k + 1] = (int8_t)((x->scales[k] >> 4) - 8);
    }
}
static inline always_inline void kq_load_iq3k(const block_iq3_k *x, __m256i wv[8], int8_t sc[16]) {
    const __m256i m03 = _mm256_set1_epi8(0x03), one = _mm256_set1_epi8(1);
    const __m256i qh = _mm256_loadu_si256((const __m256i *)x->qh);
    const unsigned extra = x->extra, sh = x->scales_h;
    for (int k = 0; k < 8; ++k) {
        const __m256i q = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * (k >> 2)));
        const __m256i lo = _mm256_and_si256(_mm256_srl_epi16(q, _mm_cvtsi32_si128(2 * (k & 3))), m03);
        const __m256i hb = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(k)), one);
        const __m256i idx = _mm256_or_si256(lo, _mm256_slli_epi16(hb, 2));
        const __m128i ta = _mm_loadl_epi64((const __m128i *)(iq3nl_values + (((extra >> (2 * k)) & 1) ? 8 : 0)));
        const __m128i tb = _mm_loadl_epi64((const __m128i *)(iq3nl_values + (((extra >> (2 * k + 1)) & 1) ? 8 : 0)));
        wv[k] = _mm256_shuffle_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(ta), tb, 1), idx);
        const int m1 = 2 * (x->scales_l[k] & 0xf) + 1, m2 = 2 * (x->scales_l[k] >> 4) + 1;
        sc[2 * k] = (int8_t)(((sh >> (2 * k)) & 1) ? -m1 : m1); sc[2 * k + 1] = (int8_t)(((sh >> (2 * k + 1)) & 1) ? -m2 : m2);
    }
}

static inline size_t kq_block_bytes(int kind) {
    switch (kind) {
    case 0:       return sizeof(block_iq6_k);
    case 1:       return sizeof(block_iq4_xs);
    case KQ_Q6K:  return sizeof(block_q6_K);
    case KQ_Q3K:  return sizeof(block_q3_K);
    case KQ_Q4K:  return sizeof(block_q4_K);
    case KQ_Q5K:  return sizeof(block_q5_K);
    case KQ_Q4KR4: return sizeof(block_q4_k_r4) / 4;   /* per-row equivalent: 8 rows = two consecutive R4 groups */
    case KQ_Q6KR4: return sizeof(block_q6_K_R4) / 4;
    case KQ_IQ4K:  return sizeof(block_iq4_k);
    case KQ_IQ4KR4: return sizeof(block_iq4_k_r4) / 4;
    case KQ_IQ2K:  return sizeof(block_iq2_k);
    case KQ_IQ3K:  return sizeof(block_iq3_k);
    case KQ_IQ2KR4: return sizeof(block_iq2_k_r4) / 4;
    case KQ_IQ3KR4: return sizeof(block_iq3_k_r4) / 4;
    default:      return sizeof(block_q2_K);
    }
}


/* ---- Q4_K / Q5_K / Q2_K sources (asymmetric: w = d*sc*q - dmin*m, q unsigned) ----
 * The unsigned quants (0..15, 0..31, 0..3) are valid non-negative int8 inputs for both dot kernels
 * (the AVX2 sign trick and the VNNI +128 trick are exact for them), so the dot part reuses the
 * tiles unchanged. The min term is   dmin * dy * sum_g m_g * bsum_g   with bsum_g the Q8_K per-16
 * sums: the tiles add it with one madd_epi16 per pair of 16-groups. 32-wide sub-blocks (Q4_K/Q5_K)
 * simply give both of their 16-groups the same scale and min. */
static inline void kq_k4_scale_min(const uint8_t *q, int j, int *sc, int *m) {
    if (j < 4) { *sc = q[j] & 63; *m = q[j + 4] & 63; }
    else { *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
static inline void kq_k4_meta(const uint8_t *scales, int8_t sc[16], uint8_t mn[16]) {
    for (int s = 0; s < 8; ++s) {
        int a, b; kq_k4_scale_min(scales, s, &a, &b);
        sc[2 * s] = sc[2 * s + 1] = (int8_t)a; mn[2 * s] = mn[2 * s + 1] = (uint8_t)b;
    }
}
static inline always_inline void kq_load_q4(const block_q4_K *x, __m256i wv[8], int8_t sc[16], uint8_t mn[16]) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    for (int i = 0; i < 4; ++i) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * i));
        wv[2 * i]     = _mm256_and_si256(v, m0f);
        wv[2 * i + 1] = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
    }
    kq_k4_meta(x->scales, sc, mn);
}
static inline always_inline void kq_load_q5(const block_q5_K *x, __m256i wv[8], int8_t sc[16], uint8_t mn[16]) {
    const __m256i m0f = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
    const __m256i qh = _mm256_loadu_si256((const __m256i *)x->qh);
    for (int i = 0; i < 4; ++i) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * i));
        const __m256i lo = _mm256_and_si256(v, m0f), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
        const __m256i h0 = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * i), one);
        const __m256i h1 = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * i + 1), one);
        wv[2 * i]     = _mm256_or_si256(lo, _mm256_slli_epi16(h0, 4));
        wv[2 * i + 1] = _mm256_or_si256(hi, _mm256_slli_epi16(h1, 4));
    }
    kq_k4_meta(x->scales, sc, mn);
}
static inline always_inline void kq_load_q2(const block_q2_K *x, __m256i wv[8], int8_t sc[16], uint8_t mn[16]) {
    const __m256i m03 = _mm256_set1_epi8(0x03);
    for (int n = 0; n < 2; ++n) {
        const __m256i q = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * n));
        for (int j = 0; j < 4; ++j) wv[4 * n + j] = _mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m03);
    }
    for (int g = 0; g < 16; ++g) { sc[g] = (int8_t)(x->scales[g] & 0xF); mn[g] = (uint8_t)(x->scales[g] >> 4); }
}

/* Per-thread scratch for the min term: mn = for each super-block 8 pairs (16-group 2p, 2p+1) of int16
 * mins per row laid out [p][2r + {0,1}] (one 256-bit madd operand per pair); dm = dmin per row. */
typedef struct { int16_t *mn; float *dm; size_t mn_cap, dm_cap; } kqm_scratch_t;
#if !defined(_WIN32)
static __thread kqm_scratch_t kqm_tl;
#else
static kqm_scratch_t kqm_tl;
#endif
static int kqm_reserve(int nb) {
    const size_t mn = (size_t)nb * 256, dm = (size_t)nb * 32;
    if (kqm_tl.mn_cap < mn) { _mm_free(kqm_tl.mn); kqm_tl.mn = (int16_t *)_mm_malloc(mn, 64); kqm_tl.mn_cap = kqm_tl.mn ? mn : 0; }
    if (kqm_tl.dm_cap < dm) { _mm_free(kqm_tl.dm); kqm_tl.dm = (float *)_mm_malloc(dm, 64); kqm_tl.dm_cap = kqm_tl.dm ? dm : 0; }
    return kqm_tl.mn && kqm_tl.dm;
}
static inline void kqm_store(int ib, int r, const uint8_t mn[16], float dmin) {
    int16_t *dst = kqm_tl.mn + (size_t)ib * 128;
    for (int p = 0; p < 8; ++p) { dst[p * 16 + 2 * r] = mn[2 * p]; dst[p * 16 + 2 * r + 1] = mn[2 * p + 1]; }
    kqm_tl.dm[(size_t)ib * 8 + r] = dmin;
}
/* One row of a min-type block: fills wv, sc, and the min scratch; returns d. */
static inline always_inline float kq_load_min_row(int kind, const char *xr, int ib, int r, __m256i wv[8], int8_t sc[16]) {
    uint8_t mn[16]; float d, dmin;
    if (kind == KQ_Q4K)      { const block_q4_K *x = (const block_q4_K *)xr + ib; kq_load_q4(x, wv, sc, mn); d = iq6_f16(x->d); dmin = iq6_f16(x->dmin); }
    else if (kind == KQ_Q5K) { const block_q5_K *x = (const block_q5_K *)xr + ib; kq_load_q5(x, wv, sc, mn); d = iq6_f16(x->d); dmin = iq6_f16(x->dm); }
    else                     { const block_q2_K *x = (const block_q2_K *)xr + ib; kq_load_q2(x, wv, sc, mn); d = iq6_f16(x->d); dmin = iq6_f16(x->dmin); }
    kqm_store(ib, r, mn, dmin);
    return d;
}

static void kq_t_decode_group(int kind, const char *wbase, size_t w_row_bytes, int nb) {
    for (int ib = 0; ib < nb; ++ib) {
        __m256i wv[8][8];
        float dd[8];
        int16_t *scb = iq6t_tl.sc + (size_t)ib * 256;
        for (int r = 0; r < 8; ++r) {
            const char *xr = wbase + (size_t)r * w_row_bytes;
            int8_t sc[16];
            if (kind == KQ_Q6K) { const block_q6_K *x = (const block_q6_K *)xr + ib; kq_load_q6(x, wv[r], sc); dd[r] = iq6_f16(x->d); }
            else if (kind == KQ_Q3K) { const block_q3_K *x = (const block_q3_K *)xr + ib; kq_load_q3(x, wv[r], sc); dd[r] = iq6_f16(x->d); }
            else if (kind == KQ_IQ4K) { const block_iq4_k *x = (const block_iq4_k *)xr + ib; kq_load_iq4k(x, wv[r], sc); dd[r] = iq6_f16(x->d); }
            else if (kind == KQ_IQ2K) { const block_iq2_k *x = (const block_iq2_k *)xr + ib; kq_load_iq2k(x, wv[r], sc); dd[r] = iq6_f16(x->d); }
            else if (kind == KQ_IQ3K) { const block_iq3_k *x = (const block_iq3_k *)xr + ib; kq_load_iq3k(x, wv[r], sc); dd[r] = iq6_f16(x->d); }
            else dd[r] = kq_load_min_row(kind, xr, ib, r, wv[r], sc);
            for (int g = 0; g < 16; ++g) { scb[g * 16 + 2 * r] = sc[g]; scb[g * 16 + 2 * r + 1] = sc[g]; }
        }
        int8_t *wb = iq6t_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i rr[8], o[8];
            for (int r = 0; r < 8; ++r) rr[r] = wv[r][k];
            iq6t_transpose8(rr, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), o[j]);
        }
        _mm256_storeu_ps(iq6t_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}


/* ---- Q4_K_R4 source (stored 4-row interleaved Q4_K, 576B per 4 rows x 256 values) ----
 * Per 32-value sub-block ib, qs[64*ib + 16*k + 4*row + i] holds the nibbles of row `row` with the
 * same fixed pattern as IQ4_NL_R4 (lo/hi of slice 0,2 -> values 0..15, slices 1,3 -> 16..31), so no
 * transpose is needed: an R8 vector j is the nibble-extract of the matching 16-byte slice of two R4
 * groups (rows 0-3 | rows 4-7).  Scale/min: is = 4*ib + row, 6 bits each (see the reference
 * dequantize_row_q4_k_r4_single).  d[row] = scale, d[row+4] = min. */
static inline void kqr4_scale_min(const block_q4_k_r4 *x, int ib, int row, int *sc, int *mn) {
    const int is = 4 * ib + row;
    const unsigned l = x->scales_l[is], h = (unsigned)x->scales_h[is & 15] >> (4 * (is >> 4));
    *sc = (int)((l & 0xf) | ((h & 0x03) << 4));
    *mn = (int)((l >> 4) | ((h & 0x0c) << 2));
}
/* Writes the 8 R8 vectors (o[j]) of sub-block k for rows a (0-3) | b (4-7) of super-block ib. */
static inline always_inline void kqr4_vecs(const block_q4_k_r4 *a, const block_q4_k_r4 *b, int k, __m256i o[8]) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    __m256i w[4];
    for (int kk = 0; kk < 4; ++kk)
        w[kk] = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a->qs + 64 * k + 16 * kk))),
                                         _mm_loadu_si128((const __m128i *)(b->qs + 64 * k + 16 * kk)), 1);
#define KR4_LO(x) _mm256_and_si256((x), m0f)
#define KR4_HI(x) _mm256_and_si256(_mm256_srli_epi16((x), 4), m0f)
    o[0] = KR4_LO(w[0]); o[1] = KR4_LO(w[2]); o[2] = KR4_HI(w[0]); o[3] = KR4_HI(w[2]);
    o[4] = KR4_LO(w[1]); o[5] = KR4_LO(w[3]); o[6] = KR4_HI(w[1]); o[7] = KR4_HI(w[3]);
#undef KR4_LO
#undef KR4_HI
}
/* per-row metadata for super-block ib: scales per 16-group (duplicated per 32-block), mins -> kqm scratch */
static inline void kqr4_meta(const block_q4_k_r4 *a, const block_q4_k_r4 *b, int ib, int8_t sc[8][16], float dd[8]) {
    for (int r = 0; r < 8; ++r) {
        const block_q4_k_r4 *x = r < 4 ? a : b; const int row = r & 3;
        uint8_t mn[16];
        for (int k = 0; k < 8; ++k) {
            int s, m; kqr4_scale_min(x, k, row, &s, &m);
            sc[r][2 * k] = sc[r][2 * k + 1] = (int8_t)s; mn[2 * k] = mn[2 * k + 1] = (uint8_t)m;
        }
        dd[r] = iq6_f16(x->d[row]);
        kqm_store(ib, r, mn, iq6_f16(x->d[row + 4]));
    }
}
static void kqr4_t_decode_group(const char *wbase, size_t w_row_bytes, int nb) {
    const block_q4_k_r4 *A = (const block_q4_k_r4 *)wbase, *Bg = (const block_q4_k_r4 *)(wbase + 4 * w_row_bytes);
    for (int ib = 0; ib < nb; ++ib) {
        int8_t sc[8][16]; float dd[8];
        kqr4_meta(A + ib, Bg + ib, ib, sc, dd);
        int16_t *scb = iq6t_tl.sc + (size_t)ib * 256;
        for (int r = 0; r < 8; ++r) for (int g = 0; g < 16; ++g) { scb[g * 16 + 2 * r] = sc[r][g]; scb[g * 16 + 2 * r + 1] = sc[r][g]; }
        int8_t *wb = iq6t_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i o[8]; kqr4_vecs(A + ib, Bg + ib, k, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), o[j]);
        }
        _mm256_storeu_ps(iq6t_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}


/* ---- Q6_K_R4 source (stored 4-row interleaved Q6_K, 840B per 4 rows x 256 values) ----
 * Per 32-value sub-block ib: ql slices ql[64*ib + 16*kk + 4*row + i] (kk = 0..3) and qh slices
 * qh[32*ib + 16*hh + 4*row + i] (hh = 0,1).  Reference: dequantize_row_q6_K_R4.  Value = (ql nibble |
 * 2 qh bits at 0x30) - 32 with the dword/slice mapping
 *   j0: lo(ql0) | qh0<<4    j1: lo(ql2) | qh0       j2: hi(ql0) | qh0<<2    j3: hi(ql2) | qh0>>2
 *   j4: lo(ql1) | qh1<<4    j5: lo(ql3) | qh1       j6: hi(ql1) | qh1<<2    j7: hi(ql3) | qh1>>2
 * (value index = 4*j + i); scales: s[8*ib + row] covers positions 0..15, s[8*ib + row + 4] positions
 * 16..31, i.e. natural 16-groups 2*ib and 2*ib+1.  No mins. */
static inline always_inline void kqr6_vecs(const block_q6_K_R4 *a, const block_q6_K_R4 *b, int ib, __m256i o[8]) {
    const __m256i m0f = _mm256_set1_epi8(0x0F), m30 = _mm256_set1_epi8(0x30), m32 = _mm256_set1_epi8(32);
    __m256i ql[4], qh[2];
    for (int kk = 0; kk < 4; ++kk)
        ql[kk] = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a->ql + 64 * ib + 16 * kk))),
                                          _mm_loadu_si128((const __m128i *)(b->ql + 64 * ib + 16 * kk)), 1);
    for (int hh = 0; hh < 2; ++hh)
        qh[hh] = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a->qh + 32 * ib + 16 * hh))),
                                          _mm_loadu_si128((const __m128i *)(b->qh + 32 * ib + 16 * hh)), 1);
#define R6_LO(x) _mm256_and_si256((x), m0f)
#define R6_HI(x) _mm256_and_si256(_mm256_srli_epi16((x), 4), m0f)
#define R6_H(x, sl, sr) _mm256_and_si256((sl) ? _mm256_slli_epi16((x), (sl)) : _mm256_srli_epi16((x), (sr)), m30)
    o[0] = _mm256_sub_epi8(_mm256_or_si256(R6_LO(ql[0]), R6_H(qh[0], 4, 0)), m32);
    o[1] = _mm256_sub_epi8(_mm256_or_si256(R6_LO(ql[2]), _mm256_and_si256(qh[0], m30)), m32);
    o[2] = _mm256_sub_epi8(_mm256_or_si256(R6_HI(ql[0]), R6_H(qh[0], 2, 0)), m32);
    o[3] = _mm256_sub_epi8(_mm256_or_si256(R6_HI(ql[2]), R6_H(qh[0], 0, 2)), m32);
    o[4] = _mm256_sub_epi8(_mm256_or_si256(R6_LO(ql[1]), R6_H(qh[1], 4, 0)), m32);
    o[5] = _mm256_sub_epi8(_mm256_or_si256(R6_LO(ql[3]), _mm256_and_si256(qh[1], m30)), m32);
    o[6] = _mm256_sub_epi8(_mm256_or_si256(R6_HI(ql[1]), R6_H(qh[1], 2, 0)), m32);
    o[7] = _mm256_sub_epi8(_mm256_or_si256(R6_HI(ql[3]), R6_H(qh[1], 0, 2)), m32);
#undef R6_LO
#undef R6_HI
#undef R6_H
}
static inline void kqr6_meta(const block_q6_K_R4 *a, const block_q6_K_R4 *b, int8_t sc[8][16], float dd[8]) {
    for (int r = 0; r < 8; ++r) {
        const block_q6_K_R4 *x = r < 4 ? a : b; const int row = r & 3;
        for (int ib = 0; ib < 8; ++ib) { sc[r][2 * ib] = x->scales[8 * ib + row]; sc[r][2 * ib + 1] = x->scales[8 * ib + row + 4]; }
        dd[r] = iq6_f16(x->d[row]);
    }
}
static void kqr6_t_decode_group(const char *wbase, size_t w_row_bytes, int nb) {
    const block_q6_K_R4 *A = (const block_q6_K_R4 *)wbase, *Bg = (const block_q6_K_R4 *)(wbase + 4 * w_row_bytes);
    for (int ib = 0; ib < nb; ++ib) {
        int8_t sc[8][16]; float dd[8];
        kqr6_meta(A + ib, Bg + ib, sc, dd);
        int16_t *scb = iq6t_tl.sc + (size_t)ib * 256;
        for (int r = 0; r < 8; ++r) for (int g = 0; g < 16; ++g) { scb[g * 16 + 2 * r] = sc[r][g]; scb[g * 16 + 2 * r + 1] = sc[r][g]; }
        int8_t *wb = iq6t_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i o[8]; kqr6_vecs(A + ib, Bg + ib, k, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), o[j]);
        }
        _mm256_storeu_ps(iq6t_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}


/* ---- IQ4_K_R4 source (stored 4-row interleaved IQ4_K, 576B per 4 rows x 256 values) ----
 * qs[64*ib + 16*kk + 4*row + i]: same slice/nibble pattern as IQ4_NL_R4 (dword mapping j0..j7 =
 * lo(w0) lo(w2) hi(w0) hi(w2) lo(w1) lo(w3) hi(w1) hi(w3)); slices 0,2 (j0..j3) are the first 16-group,
 * slices 1,3 (j4..j7) the second. Each ROW has its own table: values1 = extra[row] bit ib (first
 * group), values2 = extra[row+4] bit ib (second group), so a lane (4 rows) can mix tables: shuffle
 * with both tables and blend per row.  Scales: is = 8*ib + row (first group), +4 (second group),
 * see dequantize_row_iq4_k_r4_single. d[row] per row, no mins. */
static inline always_inline void kqi4r_vecs(const block_iq4_k_r4 *a, const block_iq4_k_r4 *b, int ib, __m256i o[8]) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m128i t0 = _mm_loadu_si128((const __m128i *)iq4k_values), t1 = _mm_loadu_si128((const __m128i *)(iq4k_values + 16));
    const __m256i T0 = _mm256_broadcastsi128_si256(t0), T1 = _mm256_broadcastsi128_si256(t1);
    /* per-row table masks (0xFF bytes for rows whose table is #1), group 1 uses extra[row], group 2 extra[row+4] */
    uint8_t mk1[32], mk2[32];
    for (int r = 0; r < 4; ++r) {
        const unsigned e1a = (a->extra[r] >> ib) & 1, e2a = (a->extra[r + 4] >> ib) & 1;
        const unsigned e1b = (b->extra[r] >> ib) & 1, e2b = (b->extra[r + 4] >> ib) & 1;
        for (int i = 0; i < 4; ++i) {
            mk1[4 * r + i] = e1a ? 0xFF : 0; mk1[16 + 4 * r + i] = e1b ? 0xFF : 0;
            mk2[4 * r + i] = e2a ? 0xFF : 0; mk2[16 + 4 * r + i] = e2b ? 0xFF : 0;
        }
    }
    const __m256i M1 = _mm256_loadu_si256((const __m256i *)mk1), M2 = _mm256_loadu_si256((const __m256i *)mk2);
    __m256i w[4];
    for (int kk = 0; kk < 4; ++kk)
        w[kk] = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a->qs + 64 * ib + 16 * kk))),
                                         _mm_loadu_si128((const __m128i *)(b->qs + 64 * ib + 16 * kk)), 1);
#define I4_LO(x) _mm256_and_si256((x), m0f)
#define I4_HI(x) _mm256_and_si256(_mm256_srli_epi16((x), 4), m0f)
#define I4_LUT(idx, M) _mm256_blendv_epi8(_mm256_shuffle_epi8(T0, (idx)), _mm256_shuffle_epi8(T1, (idx)), (M))
    o[0] = I4_LUT(I4_LO(w[0]), M1); o[1] = I4_LUT(I4_LO(w[2]), M1); o[2] = I4_LUT(I4_HI(w[0]), M1); o[3] = I4_LUT(I4_HI(w[2]), M1);
    o[4] = I4_LUT(I4_LO(w[1]), M2); o[5] = I4_LUT(I4_LO(w[3]), M2); o[6] = I4_LUT(I4_HI(w[1]), M2); o[7] = I4_LUT(I4_HI(w[3]), M2);
#undef I4_LO
#undef I4_HI
#undef I4_LUT
}
static inline int kqi4r_scale(const block_iq4_k_r4 *x, int is) {
    return (int)((((x->scales_l[is % 32] >> (4 * (is / 32))) & 0xf) | (((x->scales_h[is % 16] >> (2 * (is / 16))) & 3) << 4))) - 32;
}
static inline void kqi4r_meta(const block_iq4_k_r4 *a, const block_iq4_k_r4 *b, int8_t sc[8][16], float dd[8]) {
    for (int r = 0; r < 8; ++r) {
        const block_iq4_k_r4 *x = r < 4 ? a : b; const int row = r & 3;
        for (int ib = 0; ib < 8; ++ib) { sc[r][2 * ib] = (int8_t)kqi4r_scale(x, 8 * ib + row); sc[r][2 * ib + 1] = (int8_t)kqi4r_scale(x, 8 * ib + row + 4); }
        dd[r] = iq6_f16(x->d[row]);
    }
}
static void kqi4r_t_decode_group(const char *wbase, size_t w_row_bytes, int nb) {
    const block_iq4_k_r4 *A = (const block_iq4_k_r4 *)wbase, *Bg = (const block_iq4_k_r4 *)(wbase + 4 * w_row_bytes);
    for (int ib = 0; ib < nb; ++ib) {
        int8_t sc[8][16]; float dd[8];
        kqi4r_meta(A + ib, Bg + ib, sc, dd);
        int16_t *scb = iq6t_tl.sc + (size_t)ib * 256;
        for (int r = 0; r < 8; ++r) for (int g = 0; g < 16; ++g) { scb[g * 16 + 2 * r] = sc[r][g]; scb[g * 16 + 2 * r + 1] = sc[r][g]; }
        int8_t *wb = iq6t_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i o[8]; kqi4r_vecs(A + ib, Bg + ib, k, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), o[j]);
        }
        _mm256_storeu_ps(iq6t_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}


/* ---- IQ2_K_R4 / IQ3_K_R4 sources (stored 4-row interleaved) ----
 * Per sub-block ib: qs slices at qs + 32*ib (+0: first half, +16: second half), each slice = 16 bytes with
 * dword r = row r; a byte holds four 2-bit fields f -> value position i + 4f, so R8 vector j (positions
 * 4j..4j+3) is field j (j<4, first slice) or j-4 (second slice).  IQ3_K_R4 adds the third bit from
 * qh + 16*ib (same slice shape, bit j of each byte).  Tables are per ROW (extra[row] bit ib first half,
 * extra[row+4] second half): shuffle with both tables and blend per row.  Scales: is = 8*ib + row (+4 for
 * the second half) as in the references.  d[row] per row, no mins. */
static inline always_inline void kqik4_vecs(int kind, const void *va, const void *vb, int ib, __m256i o[8]) {
    const int is3 = (kind == KQ_IQ3KR4);
    const uint8_t *qa = is3 ? ((const block_iq3_k_r4 *)va)->qs : ((const block_iq2_k_r4 *)va)->qs;
    const uint8_t *qb = is3 ? ((const block_iq3_k_r4 *)vb)->qs : ((const block_iq2_k_r4 *)vb)->qs;
    const uint8_t *ea = is3 ? ((const block_iq3_k_r4 *)va)->extra : ((const block_iq2_k_r4 *)va)->extra;
    const uint8_t *eb = is3 ? ((const block_iq3_k_r4 *)vb)->extra : ((const block_iq2_k_r4 *)vb)->extra;
    const __m256i m03 = _mm256_set1_epi8(0x03), one = _mm256_set1_epi8(1);
    __m256i T0, T1;
    if (is3) { T0 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i *)iq3nl_values)); T1 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i *)(iq3nl_values + 8))); }
    else { int t0, t1; memcpy(&t0, iq2nl_values, 4); memcpy(&t1, iq2nl_values + 4, 4);
           T0 = _mm256_broadcastsi128_si256(_mm_cvtsi32_si128(t0)); T1 = _mm256_broadcastsi128_si256(_mm_cvtsi32_si128(t1)); }
    uint8_t mk1[32], mk2[32];
    for (int r = 0; r < 4; ++r) for (int i = 0; i < 4; ++i) {
        mk1[4 * r + i] = ((ea[r] >> ib) & 1) ? 0xFF : 0;     mk1[16 + 4 * r + i] = ((eb[r] >> ib) & 1) ? 0xFF : 0;
        mk2[4 * r + i] = ((ea[r + 4] >> ib) & 1) ? 0xFF : 0; mk2[16 + 4 * r + i] = ((eb[r + 4] >> ib) & 1) ? 0xFF : 0;
    }
    const __m256i M1 = _mm256_loadu_si256((const __m256i *)mk1), M2 = _mm256_loadu_si256((const __m256i *)mk2);
#define IK_SL(a, b) _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a))), _mm_loadu_si128((const __m128i *)(b)), 1)
    const __m256i SA = IK_SL(qa + 32 * ib, qb + 32 * ib), SB = IK_SL(qa + 32 * ib + 16, qb + 32 * ib + 16);
    __m256i QH = _mm256_setzero_si256();
    if (is3) QH = IK_SL(((const block_iq3_k_r4 *)va)->qh + 16 * ib, ((const block_iq3_k_r4 *)vb)->qh + 16 * ib);
#undef IK_SL
#define IK_IDX(S, f, j) ({ __m256i _l = _mm256_and_si256(_mm256_srli_epi16((S), 2 * (f)), m03); \
        if (is3) _l = _mm256_or_si256(_l, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(QH, (j)), one), 2)); _l; })
#define IK_LUT(idx, M) _mm256_blendv_epi8(_mm256_shuffle_epi8(T0, (idx)), _mm256_shuffle_epi8(T1, (idx)), (M))
    o[0] = IK_LUT(IK_IDX(SA, 0, 0), M1); o[1] = IK_LUT(IK_IDX(SA, 1, 1), M1); o[2] = IK_LUT(IK_IDX(SA, 2, 2), M1); o[3] = IK_LUT(IK_IDX(SA, 3, 3), M1);
    o[4] = IK_LUT(IK_IDX(SB, 0, 4), M2); o[5] = IK_LUT(IK_IDX(SB, 1, 5), M2); o[6] = IK_LUT(IK_IDX(SB, 2, 6), M2); o[7] = IK_LUT(IK_IDX(SB, 3, 7), M2);
#undef IK_IDX
#undef IK_LUT
}
static inline int kqik4_scale2(const block_iq2_k_r4 *x, int is) { return (int)((x->scales[is % 32] >> (4 * (is / 32))) & 0xf) - 8; }
static inline int kqik4_scale3(const block_iq3_k_r4 *x, int is) {
    const int m = 2 * (int)((x->scales_l[is % 32] >> (4 * (is / 32))) & 0xf) + 1;
    return ((x->scales_h[is % 8] >> (is / 8)) & 1) ? -m : m;
}
static inline void kqik4_meta(int kind, const void *va, const void *vb, int8_t sc[8][16], float dd[8]) {
    for (int r = 0; r < 8; ++r) {
        const void *x = r < 4 ? va : vb; const int row = r & 3;
        for (int ib = 0; ib < 8; ++ib) {
            if (kind == KQ_IQ3KR4) { sc[r][2 * ib] = (int8_t)kqik4_scale3((const block_iq3_k_r4 *)x, 8 * ib + row); sc[r][2 * ib + 1] = (int8_t)kqik4_scale3((const block_iq3_k_r4 *)x, 8 * ib + row + 4); }
            else { sc[r][2 * ib] = (int8_t)kqik4_scale2((const block_iq2_k_r4 *)x, 8 * ib + row); sc[r][2 * ib + 1] = (int8_t)kqik4_scale2((const block_iq2_k_r4 *)x, 8 * ib + row + 4); }
        }
        dd[r] = iq6_f16(kind == KQ_IQ3KR4 ? ((const block_iq3_k_r4 *)x)->d[row] : ((const block_iq2_k_r4 *)x)->d[row]);
    }
}
static inline size_t kqik4_gbytes(int kind) { return kind == KQ_IQ3KR4 ? sizeof(block_iq3_k_r4) : sizeof(block_iq2_k_r4); }
static void kqik4_t_decode_group(int kind, const char *wbase, size_t w_row_bytes, int nb) {
    const size_t bb = kqik4_gbytes(kind);
    for (int ib = 0; ib < nb; ++ib) {
        const void *a = wbase + (size_t)ib * bb, *b = wbase + 4 * w_row_bytes + (size_t)ib * bb;
        int8_t sc[8][16]; float dd[8];
        kqik4_meta(kind, a, b, sc, dd);
        int16_t *scb = iq6t_tl.sc + (size_t)ib * 256;
        for (int r = 0; r < 8; ++r) for (int g = 0; g < 16; ++g) { scb[g * 16 + 2 * r] = sc[r][g]; scb[g * 16 + 2 * r + 1] = sc[r][g]; }
        int8_t *wb = iq6t_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i o[8]; kqik4_vecs(kind, a, b, k, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), o[j]);
        }
        _mm256_storeu_ps(iq6t_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}

static inline always_inline
void iq6t_tile(int nb, const char *y0, size_t y_row_bytes, float *out, size_t bs, const int NC, const int HASMIN) {
    __m256 facc[IQ6T_NCMAX];
#pragma GCC unroll 8
    for (int c = 0; c < NC; c++) facc[c] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const int8_t *wb = iq6t_tl.w + (size_t)ib * 2048;
        const int16_t *sb = iq6t_tl.sc + (size_t)ib * 256;
        const block_q8_K *yb[IQ6T_NCMAX];
#pragma GCC unroll 8
        for (int c = 0; c < NC; c++) yb[c] = (const block_q8_K *)(y0 + (size_t)c * y_row_bytes) + ib;
        __m256i acc[IQ6T_NCMAX];
#pragma GCC unroll 8
        for (int c = 0; c < NC; c++) acc[c] = _mm256_setzero_si256();
        for (int g = 0; g < 16; ++g) {
            const __m256i s16 = _mm256_loadu_si256((const __m256i *)(sb + g * 16));
            for (int jj = 0; jj < 4; ++jj) {
                const int v = 4 * g + jj;
                const __m256i w = _mm256_loadu_si256((const __m256i *)(wb + (size_t)v * 32));
                const __m256i aw = _mm256_sign_epi8(w, w);
#pragma GCC unroll 8
                for (int c = 0; c < NC; c++) {
                    int yd; memcpy(&yd, yb[c]->qs + 4 * v, 4);
                    const __m256i sy = _mm256_sign_epi8(_mm256_set1_epi32(yd), w);
                    acc[c] = _mm256_add_epi32(acc[c], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sy), s16));
                }
            }
        }
        const __m256 wsc = _mm256_loadu_ps(iq6t_tl.d + (size_t)ib * 8);
#pragma GCC unroll 8
        for (int c = 0; c < NC; c++)
            facc[c] = IQ6T_FMA(_mm256_cvtepi32_ps(acc[c]), _mm256_mul_ps(wsc, _mm256_set1_ps(yb[c]->d)), facc[c]);
        if (HASMIN) {   /* - dmin * dy * sum_g m_g * bsum_g */
            const __m256 dmv = _mm256_loadu_ps(kqm_tl.dm + (size_t)ib * 8);
            const int16_t *mnb = kqm_tl.mn + (size_t)ib * 128;
#pragma GCC unroll 8
            for (int c = 0; c < NC; c++) {
                __m256i ms = _mm256_setzero_si256();
                for (int pp = 0; pp < 8; ++pp) {
                    int bw; memcpy(&bw, yb[c]->bsums + 2 * pp, 4);
                    ms = _mm256_add_epi32(ms, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(mnb + pp * 16)), _mm256_set1_epi32(bw)));
                }
                facc[c] = IQ6T_FNMA(_mm256_cvtepi32_ps(ms), _mm256_mul_ps(dmv, _mm256_set1_ps(yb[c]->d)), facc[c]);
            }
        }
    }
#pragma GCC unroll 8
    for (int c = 0; c < NC; c++) _mm256_storeu_ps(out + (size_t)c * bs, facc[c]);
}

static int iq6t_drive(int kind, int nrows, int ncols, int k, const void *vx, const void *vy,
                      float *out, size_t bs, int ith, int nth) {
    if (nrows < 8 || (nrows % 8) != 0 || ncols < 1 || k < QK_K || (k % QK_K) != 0 || nth < 1) return 0;
    const int nb = k / QK_K, groups = nrows / 8;
    const int gpt = (groups + nth - 1) / nth;
    const int g_start = ith * gpt, g_end = (ith + 1) * gpt < groups ? (ith + 1) * gpt : groups;
    if (g_start >= g_end) return nrows;
    if (!iq6t_reserve(nb)) return 0;
    const int hasmin = (kind >= KQ_Q4K && kind <= KQ_Q4KR4);
    if (hasmin && !kqm_reserve(nb)) return 0;
    iq6_luts_t L; if (!kind) L = iq6_luts_init();
    const size_t w_row_bytes = (size_t)nb * kq_block_bytes(kind), y_row_bytes = (size_t)nb * sizeof(block_q8_K);
    for (int c0 = 0; c0 < ncols; c0 += IQ6T_PC) {
        const int pc = (ncols - c0 < IQ6T_PC) ? (ncols - c0) : IQ6T_PC;
        for (int g = g_start; g < g_end; ++g) {
            if (kind == KQ_Q4KR4) kqr4_t_decode_group((const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else if (kind == KQ_IQ4KR4) kqi4r_t_decode_group((const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else if (kind == KQ_IQ2KR4 || kind == KQ_IQ3KR4) kqik4_t_decode_group(kind, (const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else if (kind == KQ_Q6KR4) kqr6_t_decode_group((const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else if (kind >= KQ_Q6K) kq_t_decode_group(kind, (const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else if (kind) iq4xs_t_decode_group((const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else      iq6t_decode_group((const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb, &L);
            float *og = out + (size_t)c0 * bs + (size_t)g * 8;
            int cc = 0;
#define IQ6T_RUN(NCV) do { if (hasmin) iq6t_tile(nb, (const char *)vy + (size_t)(c0 + cc) * y_row_bytes, y_row_bytes, og + (size_t)cc * bs, bs, NCV, 1); else iq6t_tile(nb, (const char *)vy + (size_t)(c0 + cc) * y_row_bytes, y_row_bytes, og + (size_t)cc * bs, bs, NCV, 0); } while (0)
#if IQ6T_NCMAX >= 8
            for (; cc + 8 <= pc; cc += 8) IQ6T_RUN(8);
#endif
            for (; cc + 4 <= pc; cc += 4) IQ6T_RUN(4);
            if (cc + 2 <= pc) { IQ6T_RUN(2); cc += 2; }
            if (cc + 1 <= pc) { IQ6T_RUN(1); cc += 1; }
#undef IQ6T_RUN
        }
    }
    return nrows;
}

int sgemm_iq6_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy,
                         float *out, size_t bs, int ith, int nth) {
    return iq6t_drive(0, nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
/* IQ4_XS x Q8_K decode-once GEMM (AVX2): k = number of elements per row, rows contiguous. */
int sgemm_iq4_xs_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy,
                          float *out, size_t bs, int ith, int nth) {
    return iq6t_drive(1, nrows, ncols, k, vx, vy, out, bs, ith, nth);
}

/* Q6_K / Q3_K x Q8_K decode-once GEMMs (k = elements per row, rows contiguous). */
int sgemm_q6_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_Q6K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q3_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_Q3K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q4_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_Q4K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q5_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_Q5K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q2_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_Q2K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q4_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_Q4KR4, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q6_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_Q6KR4, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq4_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_IQ4K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq4_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_IQ4KR4, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq2_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_IQ2K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq3_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_IQ3K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq2_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_IQ2KR4, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq3_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6t_drive(KQ_IQ3KR4, nrows, ncols, k, vx, vy, out, bs, ith, nth); }


/* ================================================================
 * VNNI variant of the decode-once GEMM: sgemm_iq6_k_q8_k_r8v  (AVX-512 VNNI + VL)
 *
 * Same scratch idea, but the inner product uses dpbusd (1 uop per 32 MACs instead of
 * sign + maddubs + madd) with the shift on the WEIGHTS, so Q8_K activations are read in
 * place and no activation copy is staged:
 *   wu = w ^ 0x80 (= w + 128, u8) is the unsigned operand, y (int8) the signed one, so
 *   sum(wu*y) = sum(w*y) + 128*sum(y). Each 16-value group's accumulator is SEEDED with
 *   -128*bsum_g (bsums are already in every Q8_K block), which cancels the bias exactly.
 *   Per group: t = seed; 4x dpbusd; isum += mullo(t, sl32_g)  (per-row int32 group scales).
 * 8 rows x IQ6V_NC columns per tile; the int32 seeds (-128*bsum) are staged once per column
 * panel (64B per block and column).
 * ================================================================ */
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define IQ6V_NC 8

typedef struct { uint8_t *w; int32_t *sl; float *d; int32_t *cr; size_t w_cap, sl_cap, d_cap, cr_cap; } iq6v_scratch_t;
#if !defined(_WIN32)
static __thread iq6v_scratch_t iq6v_tl;
#else
static iq6v_scratch_t iq6v_tl;
#endif

static int iq6v_reserve(int nb, int pc) {
    const size_t w = (size_t)nb * 2048, sl = (size_t)nb * 512, d = (size_t)nb * 32, cr = (size_t)pc * nb * 64;
    if (iq6v_tl.w_cap < w)   { _mm_free(iq6v_tl.w);  iq6v_tl.w  = (uint8_t *)_mm_malloc(w, 64);  iq6v_tl.w_cap  = iq6v_tl.w  ? w  : 0; }
    if (iq6v_tl.sl_cap < sl) { _mm_free(iq6v_tl.sl); iq6v_tl.sl = (int32_t *)_mm_malloc(sl, 64); iq6v_tl.sl_cap = iq6v_tl.sl ? sl : 0; }
    if (iq6v_tl.d_cap < d)   { _mm_free(iq6v_tl.d);  iq6v_tl.d  = (float *)_mm_malloc(d, 64);   iq6v_tl.d_cap  = iq6v_tl.d  ? d  : 0; }
    if (iq6v_tl.cr_cap < cr) { _mm_free(iq6v_tl.cr); iq6v_tl.cr = (int32_t *)_mm_malloc(cr, 64); iq6v_tl.cr_cap = iq6v_tl.cr ? cr : 0; }
    return iq6v_tl.w && iq6v_tl.sl && iq6v_tl.d && iq6v_tl.cr;
}

static void iq6v_decode_group(const char *wbase, size_t w_row_bytes, int nb, const iq6_luts_t *L) {
    const __m256i x80 = _mm256_set1_epi8((char)0x80);
    for (int ib = 0; ib < nb; ++ib) {
        __m256i wv[8][8];
        float dd[8];
        int32_t *slb = iq6v_tl.sl + (size_t)ib * 128;
        for (int r = 0; r < 8; ++r) {
            const block_iq6_k *x = (const block_iq6_k *)(wbase + (size_t)r * w_row_bytes) + ib;
            iq6t_decode_w(x, L, wv[r]);
            for (int k = 0; k < 8; ++k) wv[r][k] = _mm256_xor_si256(wv[r][k], x80);   /* w + 128 as u8 */
            dd[r] = iq6_f16(x->d);
            for (int g = 0; g < 16; ++g) slb[g * 8 + r] = (int32_t)x->scales[g];
        }
        uint8_t *wb = iq6v_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i rr[8], o[8];
            for (int r = 0; r < 8; ++r) rr[r] = wv[r][k];
            iq6t_transpose8(rr, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), o[j]);
        }
        _mm256_storeu_ps(iq6v_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}

static void iq4xs_v_decode_group(const char *wbase, size_t w_row_bytes, int nb) {
    const __m256i x80 = _mm256_set1_epi8((char)0x80);
    for (int ib = 0; ib < nb; ++ib) {
        __m256i wv[8][8];
        float dd[8];
        int32_t *slb = iq6v_tl.sl + (size_t)ib * 128;
        for (int r = 0; r < 8; ++r) {
            const block_iq4_xs *x = (const block_iq4_xs *)(wbase + (size_t)r * w_row_bytes) + ib;
            iq4xs_load_wv(x, wv[r]);
            for (int k = 0; k < 8; ++k) {
                wv[r][k] = _mm256_xor_si256(wv[r][k], x80);   /* w + 128 as u8 */
                const int32_t sl = iq4xs_subscale(x, k);
                slb[(2 * k) * 8 + r] = sl; slb[(2 * k + 1) * 8 + r] = sl;
            }
            dd[r] = iq6_f16(x->d);
        }
        uint8_t *wb = iq6v_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i rr[8], o[8];
            for (int r = 0; r < 8; ++r) rr[r] = wv[r][k];
            iq6t_transpose8(rr, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), o[j]);
        }
        _mm256_storeu_ps(iq6v_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}

static void kq_v_decode_group(int kind, const char *wbase, size_t w_row_bytes, int nb) {
    const __m256i x80 = _mm256_set1_epi8((char)0x80);
    for (int ib = 0; ib < nb; ++ib) {
        __m256i wv[8][8];
        float dd[8];
        int32_t *slb = iq6v_tl.sl + (size_t)ib * 128;
        for (int r = 0; r < 8; ++r) {
            const char *xr = wbase + (size_t)r * w_row_bytes;
            int8_t sc[16];
            if (kind == KQ_Q6K) { const block_q6_K *x = (const block_q6_K *)xr + ib; kq_load_q6(x, wv[r], sc); dd[r] = iq6_f16(x->d); }
            else if (kind == KQ_Q3K) { const block_q3_K *x = (const block_q3_K *)xr + ib; kq_load_q3(x, wv[r], sc); dd[r] = iq6_f16(x->d); }
            else if (kind == KQ_IQ4K) { const block_iq4_k *x = (const block_iq4_k *)xr + ib; kq_load_iq4k(x, wv[r], sc); dd[r] = iq6_f16(x->d); }
            else if (kind == KQ_IQ2K) { const block_iq2_k *x = (const block_iq2_k *)xr + ib; kq_load_iq2k(x, wv[r], sc); dd[r] = iq6_f16(x->d); }
            else if (kind == KQ_IQ3K) { const block_iq3_k *x = (const block_iq3_k *)xr + ib; kq_load_iq3k(x, wv[r], sc); dd[r] = iq6_f16(x->d); }
            else dd[r] = kq_load_min_row(kind, xr, ib, r, wv[r], sc);
            for (int k = 0; k < 8; ++k) wv[r][k] = _mm256_xor_si256(wv[r][k], x80);   /* w + 128 as u8 */
            for (int g = 0; g < 16; ++g) slb[g * 8 + r] = sc[g];
        }
        uint8_t *wb = iq6v_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i rr[8], o[8];
            for (int r = 0; r < 8; ++r) rr[r] = wv[r][k];
            iq6t_transpose8(rr, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), o[j]);
        }
        _mm256_storeu_ps(iq6v_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}

static void kqr4_v_decode_group(const char *wbase, size_t w_row_bytes, int nb) {
    const __m256i x80 = _mm256_set1_epi8((char)0x80);
    const block_q4_k_r4 *A = (const block_q4_k_r4 *)wbase, *Bg = (const block_q4_k_r4 *)(wbase + 4 * w_row_bytes);
    for (int ib = 0; ib < nb; ++ib) {
        int8_t sc[8][16]; float dd[8];
        kqr4_meta(A + ib, Bg + ib, ib, sc, dd);
        int32_t *slb = iq6v_tl.sl + (size_t)ib * 128;
        for (int r = 0; r < 8; ++r) for (int g = 0; g < 16; ++g) slb[g * 8 + r] = sc[r][g];
        uint8_t *wb = iq6v_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i o[8]; kqr4_vecs(A + ib, Bg + ib, k, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), _mm256_xor_si256(o[j], x80));
        }
        _mm256_storeu_ps(iq6v_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}

static void kqr6_v_decode_group(const char *wbase, size_t w_row_bytes, int nb) {
    const __m256i x80 = _mm256_set1_epi8((char)0x80);
    const block_q6_K_R4 *A = (const block_q6_K_R4 *)wbase, *Bg = (const block_q6_K_R4 *)(wbase + 4 * w_row_bytes);
    for (int ib = 0; ib < nb; ++ib) {
        int8_t sc[8][16]; float dd[8];
        kqr6_meta(A + ib, Bg + ib, sc, dd);
        int32_t *slb = iq6v_tl.sl + (size_t)ib * 128;
        for (int r = 0; r < 8; ++r) for (int g = 0; g < 16; ++g) slb[g * 8 + r] = sc[r][g];
        uint8_t *wb = iq6v_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i o[8]; kqr6_vecs(A + ib, Bg + ib, k, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), _mm256_xor_si256(o[j], x80));
        }
        _mm256_storeu_ps(iq6v_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}

static void kqi4r_v_decode_group(const char *wbase, size_t w_row_bytes, int nb) {
    const __m256i x80 = _mm256_set1_epi8((char)0x80);
    const block_iq4_k_r4 *A = (const block_iq4_k_r4 *)wbase, *Bg = (const block_iq4_k_r4 *)(wbase + 4 * w_row_bytes);
    for (int ib = 0; ib < nb; ++ib) {
        int8_t sc[8][16]; float dd[8];
        kqi4r_meta(A + ib, Bg + ib, sc, dd);
        int32_t *slb = iq6v_tl.sl + (size_t)ib * 128;
        for (int r = 0; r < 8; ++r) for (int g = 0; g < 16; ++g) slb[g * 8 + r] = sc[r][g];
        uint8_t *wb = iq6v_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i o[8]; kqi4r_vecs(A + ib, Bg + ib, k, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), _mm256_xor_si256(o[j], x80));
        }
        _mm256_storeu_ps(iq6v_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}

static void kqik4_v_decode_group(int kind, const char *wbase, size_t w_row_bytes, int nb) {
    const __m256i x80 = _mm256_set1_epi8((char)0x80);
    const size_t bb = kqik4_gbytes(kind);
    for (int ib = 0; ib < nb; ++ib) {
        const void *a = wbase + (size_t)ib * bb, *b = wbase + 4 * w_row_bytes + (size_t)ib * bb;
        int8_t sc[8][16]; float dd[8];
        kqik4_meta(kind, a, b, sc, dd);
        int32_t *slb = iq6v_tl.sl + (size_t)ib * 128;
        for (int r = 0; r < 8; ++r) for (int g = 0; g < 16; ++g) slb[g * 8 + r] = sc[r][g];
        uint8_t *wb = iq6v_tl.w + (size_t)ib * 2048;
        for (int k = 0; k < 8; ++k) {
            __m256i o[8]; kqik4_vecs(kind, a, b, k, o);
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256((__m256i *)(wb + (size_t)(8 * k + j) * 32), _mm256_xor_si256(o[j], x80));
        }
        _mm256_storeu_ps(iq6v_tl.d + (size_t)ib * 8, _mm256_loadu_ps(dd));
    }
}

/* seeds[c][ib][g] = -128 * bsum_g of column c, block ib */
static void iq6v_stage_seeds(const char *vy, size_t y_row_bytes, int c0, int pc, int nb) {
    for (int c = 0; c < pc; ++c) {
        const block_q8_K *yb = (const block_q8_K *)(vy + (size_t)(c0 + c) * y_row_bytes);
        int32_t *dst = iq6v_tl.cr + (size_t)c * nb * 16;
        for (int ib = 0; ib < nb; ++ib) {
            const __m256i b16 = _mm256_loadu_si256((const __m256i *)yb[ib].bsums);
            const __m256i lo = _mm256_slli_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b16)), 7);
            const __m256i hi = _mm256_slli_epi32(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b16, 1)), 7);
            _mm256_storeu_si256((__m256i *)(dst + (size_t)ib * 16),     _mm256_sub_epi32(_mm256_setzero_si256(), lo));
            _mm256_storeu_si256((__m256i *)(dst + (size_t)ib * 16 + 8), _mm256_sub_epi32(_mm256_setzero_si256(), hi));
        }
    }
}

static inline always_inline
void iq6v_tile(int nb, const char *y0, size_t y_row_bytes, const int32_t *cr0, float *out, size_t bs, const int NC, const int HASMIN) {
    __m256 facc[IQ6V_NC];
#pragma GCC unroll 8
    for (int c = 0; c < NC; c++) facc[c] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t *wb = iq6v_tl.w + (size_t)ib * 2048;
        const int32_t *slb = iq6v_tl.sl + (size_t)ib * 128;
        const block_q8_K *yb[IQ6V_NC];
        const int32_t *crb[IQ6V_NC];
#pragma GCC unroll 8
        for (int c = 0; c < NC; c++) {
            yb[c] = (const block_q8_K *)(y0 + (size_t)c * y_row_bytes) + ib;
            crb[c] = cr0 + ((size_t)c * nb + ib) * 16;
        }
        __m256i isum[IQ6V_NC];
#pragma GCC unroll 8
        for (int c = 0; c < NC; c++) isum[c] = _mm256_setzero_si256();
        for (int g = 0; g < 16; ++g) {
            const __m256i w0 = _mm256_loadu_si256((const __m256i *)(wb + (size_t)(4 * g + 0) * 32));
            const __m256i w1 = _mm256_loadu_si256((const __m256i *)(wb + (size_t)(4 * g + 1) * 32));
            const __m256i w2 = _mm256_loadu_si256((const __m256i *)(wb + (size_t)(4 * g + 2) * 32));
            const __m256i w3 = _mm256_loadu_si256((const __m256i *)(wb + (size_t)(4 * g + 3) * 32));
            const __m256i s32 = _mm256_loadu_si256((const __m256i *)(slb + g * 8));
#pragma GCC unroll 8
            for (int c = 0; c < NC; c++) {
                const int8_t *q = yb[c]->qs + 16 * g;
                int d0, d1, d2, d3;
                memcpy(&d0, q, 4); memcpy(&d1, q + 4, 4); memcpy(&d2, q + 8, 4); memcpy(&d3, q + 12, 4);
                __m256i t = _mm256_set1_epi32(crb[c][g]);
                t = _mm256_dpbusd_epi32(t, w0, _mm256_set1_epi32(d0));
                t = _mm256_dpbusd_epi32(t, w1, _mm256_set1_epi32(d1));
                t = _mm256_dpbusd_epi32(t, w2, _mm256_set1_epi32(d2));
                t = _mm256_dpbusd_epi32(t, w3, _mm256_set1_epi32(d3));
                isum[c] = _mm256_add_epi32(isum[c], _mm256_mullo_epi32(t, s32));
            }
        }
        const __m256 wsc = _mm256_loadu_ps(iq6v_tl.d + (size_t)ib * 8);
#pragma GCC unroll 8
        for (int c = 0; c < NC; c++)
            facc[c] = IQ6T_FMA(_mm256_cvtepi32_ps(isum[c]), _mm256_mul_ps(wsc, _mm256_set1_ps(yb[c]->d)), facc[c]);
        if (HASMIN) {   /* - dmin * dy * sum_g m_g * bsum_g */
            const __m256 dmv = _mm256_loadu_ps(kqm_tl.dm + (size_t)ib * 8);
            const int16_t *mnb = kqm_tl.mn + (size_t)ib * 128;
#pragma GCC unroll 8
            for (int c = 0; c < NC; c++) {
                __m256i ms = _mm256_setzero_si256();
                for (int pp = 0; pp < 8; ++pp) {
                    int bw; memcpy(&bw, yb[c]->bsums + 2 * pp, 4);
                    ms = _mm256_add_epi32(ms, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(mnb + pp * 16)), _mm256_set1_epi32(bw)));
                }
                facc[c] = IQ6T_FNMA(_mm256_cvtepi32_ps(ms), _mm256_mul_ps(dmv, _mm256_set1_ps(yb[c]->d)), facc[c]);
            }
        }
    }
#pragma GCC unroll 8
    for (int c = 0; c < NC; c++) _mm256_storeu_ps(out + (size_t)c * bs, facc[c]);
}

static int iq6v_drive(int kind, int nrows, int ncols, int k, const void *vx, const void *vy,
                      float *out, size_t bs, int ith, int nth) {
    if (nrows < 8 || (nrows % 8) != 0 || ncols < 1 || k < QK_K || (k % QK_K) != 0 || nth < 1) return 0;
    const int nb = k / QK_K, groups = nrows / 8;
    const int gpt = (groups + nth - 1) / nth;
    const int g_start = ith * gpt, g_end = (ith + 1) * gpt < groups ? (ith + 1) * gpt : groups;
    if (g_start >= g_end) return nrows;
    if (!iq6v_reserve(nb, ncols < IQ6T_PC ? ncols : IQ6T_PC)) return 0;
    const int hasmin = (kind >= KQ_Q4K && kind <= KQ_Q4KR4);
    if (hasmin && !kqm_reserve(nb)) return 0;
    iq6_luts_t L; if (!kind) L = iq6_luts_init();
    const size_t w_row_bytes = (size_t)nb * kq_block_bytes(kind), y_row_bytes = (size_t)nb * sizeof(block_q8_K);
    for (int c0 = 0; c0 < ncols; c0 += IQ6T_PC) {
        const int pc = (ncols - c0 < IQ6T_PC) ? (ncols - c0) : IQ6T_PC;
        iq6v_stage_seeds((const char *)vy, y_row_bytes, c0, pc, nb);
        for (int g = g_start; g < g_end; ++g) {
            if (kind == KQ_Q4KR4) kqr4_v_decode_group((const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else if (kind == KQ_IQ4KR4) kqi4r_v_decode_group((const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else if (kind == KQ_IQ2KR4 || kind == KQ_IQ3KR4) kqik4_v_decode_group(kind, (const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else if (kind == KQ_Q6KR4) kqr6_v_decode_group((const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else if (kind >= KQ_Q6K) kq_v_decode_group(kind, (const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else if (kind) iq4xs_v_decode_group((const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb);
            else      iq6v_decode_group((const char *)vx + (size_t)g * 8 * w_row_bytes, w_row_bytes, nb, &L);
            float *og = out + (size_t)c0 * bs + (size_t)g * 8;
            int cc = 0;
#define IQ6V_ARGS(NCV) nb, (const char *)vy + (size_t)(c0 + cc) * y_row_bytes, y_row_bytes, \
                                iq6v_tl.cr + (size_t)cc * nb * 16, og + (size_t)cc * bs, bs, NCV
#define IQ6V_RUN(NCV) do { if (hasmin) iq6v_tile(IQ6V_ARGS(NCV), 1); else iq6v_tile(IQ6V_ARGS(NCV), 0); } while (0)
            for (; cc + 8 <= pc; cc += 8) IQ6V_RUN(8);
            if (cc + 4 <= pc) { IQ6V_RUN(4); cc += 4; }
            if (cc + 2 <= pc) { IQ6V_RUN(2); cc += 2; }
            if (cc + 1 <= pc) { IQ6V_RUN(1); cc += 1; }
#undef IQ6V_RUN
        }
    }
    return nrows;
}

int sgemm_iq6_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy,
                         float *out, size_t bs, int ith, int nth) {
    return iq6v_drive(0, nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
int sgemm_iq4_xs_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy,
                          float *out, size_t bs, int ith, int nth) {
    return iq6v_drive(1, nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
int sgemm_q6_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_Q6K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q3_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_Q3K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q4_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_Q4K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q5_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_Q5K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q2_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_Q2K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q4_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_Q4KR4, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_q6_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_Q6KR4, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq4_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_IQ4K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq4_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_IQ4KR4, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq2_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_IQ2K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq3_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_IQ3K, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq2_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_IQ2KR4, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
int sgemm_iq3_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { return iq6v_drive(KQ_IQ3KR4, nrows, ncols, k, vx, vy, out, bs, ith, nth); }
#else
int sgemm_iq6_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy,
                         float *out, size_t bs, int ith, int nth) {
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth;
    return 0;   /* no AVX-512 VNNI+VL: caller falls back to r8t */
}
int sgemm_iq4_xs_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy,
                          float *out, size_t bs, int ith, int nth) {
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
}
int sgemm_q6_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q3_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q4_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q5_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q2_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q4_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q6_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq4_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq4_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq2_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq3_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq2_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq3_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
#endif /* VNNI variant */

#else

void vec_dot_iq6_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
    (void)vx; (void)wy; (void)n; *out = 0.0f;
}
void vec_dot_iq6_k_q8_k_avx2_batch(const void *vx, const void *wy, size_t y_stride,
                                   int n, int ncols, float *out) {
    (void)vx; (void)wy; (void)y_stride; (void)n;
    for (int c = 0; c < ncols; ++c) out[c] = 0.0f;
}
int sgemm_iq6_k_q8_k_avx2(int nrows, int ncols, int k, const void *vx, const void *vy,
                          float *out, size_t bs, int ith, int nth) {
    (void)vx; (void)vy; (void)nrows; (void)ncols; (void)k; (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
}

int sgemm_iq6_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy,
                         float *out, size_t bs, int ith, int nth) {
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
}

int sgemm_iq6_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy,
                         float *out, size_t bs, int ith, int nth) {
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
}
int sgemm_q6_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q6_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q3_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q3_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q4_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q4_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q5_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q5_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q2_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q2_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q4_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q4_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q6_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_q6_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq4_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq4_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq4_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq4_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq2_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq2_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq3_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq3_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq2_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq2_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq3_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq3_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq4_xs_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
int sgemm_iq4_xs_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0; }

#endif

/* fn_gemm for IQ6_K: the decode-once R8 kernel for real prefill batches, the tile kernel otherwise.
 * Always computes the full result (gemm_table_task ignores the return value).
 * PICOLM_IQ6K_R8=0 forces the tile kernel; PICOLM_IQ6K_R8_MIN_N sets the smallest batch (default 16). */
int sgemm_iq6_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy,
                          float *out, size_t bs, int ith, int nth) {
    static int en = -1, min_n = 16;
    if (en < 0) {
        const char *e = getenv("PICOLM_IQ6K_R8"), *mn = getenv("PICOLM_IQ6K_R8_MIN_N");
        if (mn && atoi(mn) > 0) min_n = atoi(mn);
        en = !(e && e[0] == '0');
    }
    static int vnni_en = -1;
    if (vnni_en < 0) { const char *e = getenv("PICOLM_IQ6K_VNNI"); vnni_en = !(e && e[0] == '0'); }
    if (en && ncols >= min_n) {
        int r = vnni_en ? sgemm_iq6_k_q8_k_r8v(nrows, ncols, k, vx, vy, out, bs, ith, nth) : 0;
        const char *which = "sgemm_iq6_k_q8_k_r8v (decode-once R8, VNNI)";
        if (!r) { r = sgemm_iq6_k_q8_k_r8t(nrows, ncols, k, vx, vy, out, bs, ith, nth); which = "sgemm_iq6_k_q8_k_r8t (decode-once R8, AVX2)"; }
        if (r) {
            static int traced; if (!traced && getenv("PICOLM_DISPATCH")) { traced = 1; fprintf(stderr, "TRACE kernel: %s\n", which); }
            return r;
        }
    }
    return sgemm_iq6_k_q8_k_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}

/* ================================================================
 * Table fn_gemm for Q6_K / Q3_K. The table GEMM wrapper ignores return values, so this must always
 * produce the thread's full share: decode-once R8 (VNNI, then AVX2) when the batch and matrix are big
 * enough, otherwise a per-column dot over this thread's rows -- exactly what these types ran before
 * (bit-identical). PICOLM_KQ_R8=0 disables the fast path, PICOLM_KQ_VNNI=0 forces AVX2,
 * PICOLM_KQ_R8_MIN_N (default 16) sets the smallest batch; the size gate is picolm_r8_big_enough.
 * ================================================================ */
typedef int (*kq_r8_fn)(int, int, int, const void *, const void *, float *, size_t, int, int);
/* Fast path only: returns the kernel's result, or 0 if the batch/shape/size gate says no. */
static int kq_fast(const char *name, kq_r8_fn r8v, kq_r8_fn r8t,
                   int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    static int en = -1, min_n = 16, vnni_en = 1;
    if (en < 0) {
        const char *e = getenv("PICOLM_KQ_R8"), *mn = getenv("PICOLM_KQ_R8_MIN_N"), *v = getenv("PICOLM_KQ_VNNI");
        if (mn && atoi(mn) > 0) min_n = atoi(mn);
        if (v && v[0] == '0') vnni_en = 0;
        en = !(e && e[0] == '0');
    }
    if (!(en && ncols >= min_n && nrows % 8 == 0 && k % QK_K == 0 && picolm_r8_big_enough(nrows, k / 32))) return 0;
    int r = vnni_en ? r8v(nrows, ncols, k, vx, vy, out, bs, ith, nth) : 0;
    const char *which = "decode-once R8, VNNI";
    if (!r) { r = r8t(nrows, ncols, k, vx, vy, out, bs, ith, nth); which = "decode-once R8, AVX2"; }
    if (r) {
        static int traced; if (!traced && getenv("PICOLM_DISPATCH")) { traced = 1; fprintf(stderr, "TRACE kernel: %s (%s)\n", name, which); }
    }
    return r;
}
/* Table fn_gemm flavour: always produces this thread's full share (the table wrapper ignores return
 * values): the fast path, else a per-column dot over this thread's rows, which is exactly what these
 * types ran before (bit-identical). PICOLM_KQ_R8=0 disables the fast path, PICOLM_KQ_VNNI=0 forces
 * AVX2, PICOLM_KQ_R8_MIN_N (default 16) sets the smallest batch; size gate: picolm_r8_big_enough. */
static int kq_best(const char *name, kq_r8_fn r8v, kq_r8_fn r8t, float (*dot)(const void *, const void *, int), size_t wblock,
                   int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    int r = kq_fast(name, r8v, r8t, nrows, ncols, k, vx, vy, out, bs, ith, nth);
    if (r) return r;
    const int nb = k / QK_K, gpt = (nrows + nth - 1) / nth;
    const int r0 = ith * gpt, r1 = (ith + 1) * gpt < nrows ? (ith + 1) * gpt : nrows;
    for (int i = r0; i < r1; ++i)
        for (int c = 0; c < ncols; ++c)
            out[(size_t)c * bs + i] = dot((const char *)vx + (size_t)i * nb * wblock, (const char *)vy + (size_t)c * nb * sizeof(block_q8_K), k);
    return 1;
}
int sgemm_q6_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    return kq_best("sgemm_q6_k_q8_k", sgemm_q6_k_q8_k_r8v, sgemm_q6_k_q8_k_r8t, vec_dot_q6_K_q8_K, sizeof(block_q6_K), nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
int sgemm_q3_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    return kq_best("sgemm_q3_k_q8_k", sgemm_q3_k_q8_k_r8v, sgemm_q3_k_q8_k_r8t, vec_dot_q3_K_q8_K, sizeof(block_q3_K), nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
int sgemm_q5_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    return kq_best("sgemm_q5_k_q8_k", sgemm_q5_k_q8_k_r8v, sgemm_q5_k_q8_k_r8t, vec_dot_q5_K_q8_K, sizeof(block_q5_K), nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
int sgemm_q2_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    return kq_best("sgemm_q2_k_q8_k", sgemm_q2_k_q8_k_r8v, sgemm_q2_k_q8_k_r8t, vec_dot_q2_K_q8_K, sizeof(block_q2_K), nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
/* Q4_K: no table entry (picolm_sgemm_d_q4k has its own signature); returns 1 if handled, 0 to fall through. */
int sgemm_q4_k_q8_k_fast(int nrows, int ncols, int k_blocks, const void *vx, int lda, const void *vy, int ldb, float *out, size_t bs, int ith, int nth) {
    if (lda != k_blocks || ldb != k_blocks) return 0;
    return kq_fast("sgemm_q4_k_q8_k", sgemm_q4_k_q8_k_r8v, sgemm_q4_k_q8_k_r8t, nrows, ncols, k_blocks * QK_K, vx, vy, out, bs, ith, nth) != 0;
}
/* Q4_K_R4 table fn_gemm: decode-once R8 (VNNI / AVX2) for big shapes, else the original 1x4 kernel. */
int sgemm_q4_k_r4_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    int r = kq_fast("sgemm_q4_k_r4_q8_k", sgemm_q4_k_r4_q8_k_r8v, sgemm_q4_k_r4_q8_k_r8t, nrows, ncols, k, vx, vy, out, bs, ith, nth);
    if (r) return r;
    return sgemm_q4_k_r4_q8_k_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
/* Q6_K_R4 table fn_gemm: decode-once R8 for big shapes, else the original kernel. */
int sgemm_q6_k_r4_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    int r = kq_fast("sgemm_q6_k_r4_q8_k", sgemm_q6_k_r4_q8_k_r8v, sgemm_q6_k_r4_q8_k_r8t, nrows, ncols, k, vx, vy, out, bs, ith, nth);
    if (r) return r;
    return sgemm_q6_k_r4_q8_k(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
/* IQ4_K / IQ4_K_R4 table fn_gemm: decode-once R8 for big shapes, else the original kernels. */
int sgemm_iq4_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    int r = kq_fast("sgemm_iq4_k_q8_k", sgemm_iq4_k_q8_k_r8v, sgemm_iq4_k_q8_k_r8t, nrows, ncols, k, vx, vy, out, bs, ith, nth);
    if (r) return r;
    return sgemm_iq4_k_q8_k_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
int sgemm_iq4_k_r4_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    int r = kq_fast("sgemm_iq4_k_r4_q8_k", sgemm_iq4_k_r4_q8_k_r8v, sgemm_iq4_k_r4_q8_k_r8t, nrows, ncols, k, vx, vy, out, bs, ith, nth);
    if (r) return r;
    return sgemm_iq4_k_r4_q8_k_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
/* IQ2_K / IQ3_K (+R4) table fn_gemm: decode-once R8 for big shapes, else the original kernels. */
int sgemm_iq2_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    int r = kq_fast("sgemm_iq2_k_q8_k", sgemm_iq2_k_q8_k_r8v, sgemm_iq2_k_q8_k_r8t, nrows, ncols, k, vx, vy, out, bs, ith, nth);
    if (r) return r;
    return sgemm_iq2_k_q8_k_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
int sgemm_iq3_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    int r = kq_fast("sgemm_iq3_k_q8_k", sgemm_iq3_k_q8_k_r8v, sgemm_iq3_k_q8_k_r8t, nrows, ncols, k, vx, vy, out, bs, ith, nth);
    if (r) return r;
    return sgemm_iq3_k_q8_k_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
int sgemm_iq2_k_r4_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    int r = kq_fast("sgemm_iq2_k_r4_q8_k", sgemm_iq2_k_r4_q8_k_r8v, sgemm_iq2_k_r4_q8_k_r8t, nrows, ncols, k, vx, vy, out, bs, ith, nth);
    if (r) return r;
    return sgemm_iq2_k_r4_q8_k_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
int sgemm_iq3_k_r4_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) {
    int r = kq_fast("sgemm_iq3_k_r4_q8_k", sgemm_iq3_k_r4_q8_k_r8v, sgemm_iq3_k_r4_q8_k_r8t, nrows, ncols, k, vx, vy, out, bs, ith, nth);
    if (r) return r;
    return sgemm_iq3_k_r4_q8_k_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}

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

