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
#if defined(PICOLM_IQ6K_VNNI)

typedef struct {
    __m512i l0, l1, l2, l3;                 /* unsigned LUT quarters, in all 4 lanes */
    __m512i m0f;
    __m512i cnt0e, cnt1e, cnt0o, cnt1o;     /* sllv counts: words 0-15 (lo nibbles) | 16-31 (hi nibbles) */
    __m512i sidx[4];                        /* permutexvar indices: lane l <- group 4k + l/4 */
    __m256i bit16;                          /* 1 << g, g = 0..15 */
} iq6v_ctx_t;

typedef struct {
    __m512i uw[4];      /* unsigned weights, values 64k .. 64k+63 */
    __m512  scf[4];     /* float scale per dpbusd lane */
    __m256i scc;        /* int16[16]: scale_g * (128 - m_g) */
} iq6v_blk_t;

static inline __m512i iq6v_pair(int a, int b) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16((short)a)),
                              _mm256_set1_epi16((short)b), 1);
}

static inline iq6v_ctx_t iq6v_ctx_init(void) {
    iq6v_ctx_t C;
    C.l0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(iq6nl_lut +  0)));
    C.l1 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(iq6nl_lut + 16)));
    C.l2 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(iq6nl_lut + 32)));
    C.l3 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(iq6nl_lut + 48)));
    C.m0f = _mm512_set1_epi8(0x0f);
    /* even 64-block (qh bit offset 0): lo uses bits 0,1 ; hi uses bits 2,3
     * odd  64-block (qh bit offset 4): lo uses bits 4,5 ; hi uses bits 6,7
     * count = 7 - bit  moves that bit to bit 7 of its byte */
    C.cnt0e = iq6v_pair(7, 5);
    C.cnt1e = iq6v_pair(6, 4);
    C.cnt0o = iq6v_pair(3, 1);
    C.cnt1o = iq6v_pair(2, 0);
    const __m512i base = _mm512_setr_epi32(0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3);
    for (int k = 0; k < 4; ++k) C.sidx[k] = _mm512_add_epi32(base, _mm512_set1_epi32(4 * k));
    C.bit16 = _mm256_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128,
                                256, 512, 1024, 2048, 4096, 8192, 16384, (short)0x8000);
    return C;
}

/* low4: nibbles (0..15) in value order; qh: qh bytes duplicated in both
 * 256-bit halves; cnt0/cnt1: shift counts moving q6 bit 4 / bit 5 to bit 7. */
static inline __m512i iq6v_lookup(__m512i low4, __m512i qh, __m512i cnt0, __m512i cnt1,
                                  const iq6v_ctx_t *C) {
    const __m512i v0 = _mm512_shuffle_epi8(C->l0, low4);
    const __m512i v1 = _mm512_shuffle_epi8(C->l1, low4);
    const __m512i v2 = _mm512_shuffle_epi8(C->l2, low4);
    const __m512i v3 = _mm512_shuffle_epi8(C->l3, low4);
    const __mmask64 k0 = _mm512_movepi8_mask(_mm512_sllv_epi16(qh, cnt0));
    const __mmask64 k1 = _mm512_movepi8_mask(_mm512_sllv_epi16(qh, cnt1));
    const __m512i lo = _mm512_mask_blend_epi8(k0, v0, v1);
    const __m512i hi = _mm512_mask_blend_epi8(k0, v2, v3);
    return _mm512_mask_blend_epi8(k1, lo, hi);
}

static inline void iq6v_decode(const block_iq6_k *x, const iq6v_ctx_t *C, iq6v_blk_t *B) {
    for (int ih = 0; ih < 2; ++ih) {
        const __m256i Q  = _mm256_loadu_si256((const __m256i *)(x->qh + 32 * ih));
        const __m512i Qz = _mm512_inserti64x4(_mm512_castsi256_si512(Q), Q, 1);
        /* X = [qs of 64-block 2ih (32 B) | qs of 64-block 2ih+1 (32 B)] */
        const __m512i X  = _mm512_loadu_si512((const void *)(x->qs + 64 * ih));
        const __m512i lo = _mm512_and_si512(X, C->m0f);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(X, 4), C->m0f);
        /* natural value order = [lo nibbles (32) | hi nibbles (32)] per 64-block */
        const __m512i i0 = _mm512_shuffle_i64x2(lo, hi, 0x44);   /* lanes lo0 lo1 hi0 hi1 */
        const __m512i i1 = _mm512_shuffle_i64x2(lo, hi, 0xEE);   /* lanes lo2 lo3 hi2 hi3 */
        B->uw[2 * ih + 0] = iq6v_lookup(i0, Qz, C->cnt0e, C->cnt1e, C);
        B->uw[2 * ih + 1] = iq6v_lookup(i1, Qz, C->cnt0o, C->cnt1o, C);
    }

    const __m128i sc8 = _mm_loadu_si128((const __m128i *)x->scales);
    const __m512 sf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(sc8));
    for (int k = 0; k < 4; ++k) B->scf[k] = _mm512_permutexvar_ps(C->sidx[k], sf);

    const __m256i sc16 = _mm256_cvtepi8_epi16(sc8);
    const __m256i ex   = _mm256_set1_epi16((short)x->extra);
    const __m256i sel  = _mm256_cmpeq_epi16(_mm256_and_si256(ex, C->bit16), C->bit16);  /* -1 if m_g */
    B->scc = _mm256_mullo_epi16(sc16, _mm256_add_epi16(_mm256_set1_epi16(128), sel));   /* s*(128-m) */
}

/* One decoded weight block against one Q8_K block. d = weight block scale. */
static inline void iq6v_col(const iq6v_blk_t *B, const block_q8_K *yb, float d,
                            __m512 *facc, __m256 *fcorr) {
    const __m512i z = _mm512_setzero_si512();
    const __m512 f0 = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(z, B->uw[0], _mm512_loadu_si512((const void *)(yb->qs +   0))));
    const __m512 f1 = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(z, B->uw[1], _mm512_loadu_si512((const void *)(yb->qs +  64))));
    const __m512 f2 = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(z, B->uw[2], _mm512_loadu_si512((const void *)(yb->qs + 128))));
    const __m512 f3 = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(z, B->uw[3], _mm512_loadu_si512((const void *)(yb->qs + 192))));
    const __m512 t01 = _mm512_fmadd_ps(f1, B->scf[1], _mm512_mul_ps(f0, B->scf[0]));
    const __m512 t23 = _mm512_fmadd_ps(f3, B->scf[3], _mm512_mul_ps(f2, B->scf[2]));
    const float dd = d * yb->d;
    *facc = _mm512_fmadd_ps(_mm512_add_ps(t01, t23), _mm512_set1_ps(dd), *facc);

    const __m256i cb = _mm256_madd_epi16(B->scc, _mm256_loadu_si256((const __m256i *)yb->bsums));
    *fcorr = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(cb), _mm256_set1_ps(dd), *fcorr);
}

static inline float iq6v_finish(__m512 facc, __m256 fcorr) {
    return _mm512_reduce_add_ps(facc) + iq6_hsum_ps(fcorr);
}

void vec_dot_iq6_k_q8_k_vnni(const void *vx, const void *wy, int n, float *out) {
    const block_iq6_k *x = (const block_iq6_k *)vx;
    const block_q8_K *y = (const block_q8_K *)wy;
    const int nb = n / QK_K;
    const iq6v_ctx_t C = iq6v_ctx_init();

    __m512 facc = _mm512_setzero_ps();
    __m256 fcorr = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        iq6v_blk_t B;
        iq6v_decode(&x[ibl], &C, &B);
        iq6v_col(&B, &y[ibl], iq6_f16(x[ibl].d), &facc, &fcorr);
    }
    *out = iq6v_finish(facc, fcorr);
}

/* NC is a compile-time constant at every call site (switch below), so the
 * column loops unroll and facc/fcorr stay in registers. */
static inline __attribute__((always_inline))
void iq6v_batch_n(const void *vx, const void *wy, size_t y_stride, int n, float *out, const int NC) {
    const block_iq6_k *x = (const block_iq6_k *)vx;
    const int nb = n / QK_K;
    const iq6v_ctx_t C = iq6v_ctx_init();

    __m512 fa[IQ6K_BATCH_TILE];
    __m256 fc[IQ6K_BATCH_TILE];
    for (int c = 0; c < NC; ++c) { fa[c] = _mm512_setzero_ps(); fc[c] = _mm256_setzero_ps(); }

    for (int ibl = 0; ibl < nb; ++ibl) {
        iq6v_blk_t B;
        iq6v_decode(&x[ibl], &C, &B);
        const float d = iq6_f16(x[ibl].d);
        for (int c = 0; c < NC; ++c) {
            const block_q8_K *yb = (const block_q8_K *)((const char *)wy + (size_t)c * y_stride) + ibl;
            iq6v_col(&B, yb, d, &fa[c], &fc[c]);
        }
    }
    for (int c = 0; c < NC; ++c) out[c] = iq6v_finish(fa[c], fc[c]);
}

void vec_dot_iq6_k_q8_k_vnni_batch(const void *vx, const void *wy, size_t y_stride,
                                   int n, int ncols, float *out) {
    while (ncols > IQ6K_BATCH_TILE) {
        vec_dot_iq6_k_q8_k_vnni_batch(vx, wy, y_stride, n, IQ6K_BATCH_TILE, out);
        wy = (const char *)wy + (size_t)IQ6K_BATCH_TILE * y_stride;
        out += IQ6K_BATCH_TILE;
        ncols -= IQ6K_BATCH_TILE;
    }
    switch (ncols) {
        case 1: iq6v_batch_n(vx, wy, y_stride, n, out, 1); break;
        case 2: iq6v_batch_n(vx, wy, y_stride, n, out, 2); break;
        case 3: iq6v_batch_n(vx, wy, y_stride, n, out, 3); break;
#if IQ6K_BATCH_TILE >= 4
        case 4: iq6v_batch_n(vx, wy, y_stride, n, out, 4); break;
#endif
#if IQ6K_BATCH_TILE >= 5
        case 5: iq6v_batch_n(vx, wy, y_stride, n, out, 5); break;
        case 6: iq6v_batch_n(vx, wy, y_stride, n, out, 6); break;
        case 7: iq6v_batch_n(vx, wy, y_stride, n, out, 7); break;
#endif
#if IQ6K_BATCH_TILE >= 8
        case 8: iq6v_batch_n(vx, wy, y_stride, n, out, 8); break;
#endif
        default: break;
    }
}
#endif

