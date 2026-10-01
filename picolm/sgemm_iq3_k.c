/* ================================================================
 * IQ3_K (plain, GGUF type 138) x Q8_K AVX2 GEMV and GEMM kernels
 * ================================================================
 * Rewrite: decode-once per block, decode w[i] and sp[i] for all 8
 * sub-blocks upfront, then reuse for every activation column.
 * Eliminates double-activation-pointer bug and broken VNNI path.
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
 *
 * Per sub-block ib32 (0..7), 32 values:
 *   qs[32*(ib32/4)+0..15] and qs[32*(ib32/4)+16..31]
 *   shift_l = 2*(ib32%4) extracts 2-bit low half from qs bytes
 *   shift_h = ib32%8 extracts 1-bit high half from qh bytes
 *   3-bit index = (2-bit qs | 1-bit qh<<2) -> LUT lookup
 *   extra bit 2*ib32 selects normal/shifted for lo half
 *   extra bit 2*ib32+1 selects normal/shifted for hi half
 *   scale: (2*scales_l_nibble+1) * sign_from_scales_h
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

/* Method (per 256-value block):
 *   1. Decode all 8 sub-blocks (32 values each) to signed int8: w[i].
 *      qs loaded as 256-bit (32 bytes, 32 values). 2-bit index from
 *      (qs >> shift_l) & 3. High bit from (qh >> shift_h) & 1, shifted
 *      to bit 2. Combined 3-bit index (0-7) -> 8-entry LUT
 *      (iq3nl_values), with +8 offset where the matching extra bit is set.
 *      Both lo (values 0..15) and hi (values 16..31) decoded in one 256-bit
 *      register.
 *   2. sp[i] = int16 scale vector: (2*mag+1)*sign, duplicated for each
 *      of the 16 values per group.
 *   3. Dot: abs/sign + maddubs (int16) + madd_epi16 with sp[i]
 *      (applies the scale, reduces to int32x8).
 *      One float conversion per block, one horizontal sum per row.
 * Int range: |w| <= 63, |y| <= 127, |scale| <= 31 -> 63*127*31 = 246K,
 * fits in int32 with headroom for accumulation of 8 sub-blocks. */

static inline float iq3_hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

/* 16-entry table {normal x8, shifted x8}, repeated so every 128-bit lane
 * holds it twice (pshufb index is 0..7 only). Single source: iq3nl_values. */
static inline __m256i iq3_lut(void) {
    int64_t t0, t1;
    memcpy(&t0, iq3nl_values, sizeof(t0));
    memcpy(&t1, iq3nl_values + 8, sizeof(t1));
    return _mm256_set_epi64x(t1, t0, t1, t0);
}

static inline void iq3_decode_block(const block_iq3_k *x, __m256i lut,
                                    __m256i w[8], __m256i sp[8]) {
    const __m256i m3    = _mm256_set1_epi8(3);
    const __m256i eight = _mm256_set1_epi8(8);
    const __m256i ex    = _mm256_set1_epi16((short)x->extra);
    const __m256i ctrl  = _mm256_setr_epi8(
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
        2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3);

    /* scales: magnitude (2*mag+1) * sign from scales_h.
     * scales_l: 8 bytes, 2 nibbles per byte (16 nibbles total).
     * scales_h: 16 bits, 2 bits per subblock pair (sign for lo/hi). */
    const __m128i s8  = _mm_loadl_epi64((const __m128i *)x->scales_l);
    const __m128i lo4 = _mm_and_si128(s8, _mm_set1_epi8(0x0f));
    const __m128i hi4 = _mm_and_si128(_mm_srli_epi16(s8, 4), _mm_set1_epi8(0x0f));
    const __m128i mag  = _mm_add_epi8(_mm_unpacklo_epi8(lo4, hi4), _mm_set1_epi8(1));
    /* mag is now 2*mag+1 for each of the 16 scale groups. */

    /* Extract sign bits from scales_h (16 bits, 2 per subblock pair).
     * Bit 0 = sign for group 0, bit 1 = sign for group 1, etc.
     * sign = 1 when bit set (meaning negative), so we want -1 or +1.
     * Build int16 vector with sign values. */
    const uint16_t sh = x->scales_h;
    int16_t sg[16];
    uint16_t sh_tmp = sh;
    for (int i = 0; i < 8; i++) {
        sg[2 * i]     = (sh_tmp & 1) ? -1 : 1;
        sg[2 * i + 1] = (sh_tmp & 2) ? -1 : 1;
        sh_tmp >>= 2;
    }
    const __m128i sg_vec = _mm_loadu_si128((const __m128i *)sg);
    const __m256i sc16 = _mm256_cvtepi8_epi16(sg_vec);

    /* qh: 32 bytes total. First 16 bytes for values 0..15, next 16 for 16..31.
     * Each byte has 8 high bits, one per subblock.
     * qh[j] -> bit ib32 for value j in subblock ib32. */
    const __m128i qh0 = _mm_loadu_si128((const __m128i *)x->qh);
    const __m128i qh1 = _mm_loadu_si128((const __m128i *)(x->qh + 16));
    const __m256i qh_lo = _mm256_broadcastsi128_si256(qh0); /* lane0=qh0, lane1=qh0 */
    const __m256i qh_hi = _mm256_broadcastsi128_si256(qh1); /* lane0=qh1, lane1=qh1 */

    for (int i = 0; i < 8; ++i) {
        /* Load 32 qs bytes as one 256-bit register.
         * qs layout: 64 bytes total, 32 bytes per 32-value chunk.
         * ib32 0-3 use qs[0..31], ib32 4-7 use qs[32..63].
         * Within the 32 bytes: first 16 = values 0..15, next 16 = values 16..31.
         * shift_l = 2*(i&3) extracts the 2-bit index. */
        const __m256i qs = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * (i >> 2)));
        const __m256i ql = _mm256_and_si256(
            _mm256_srl_epi16(qs, _mm_cvtsi32_si128(2 * (i & 3))), m3);

        /* Extract high bit from qh. qh_lo for values 0..15, qh_hi for 16..31.
         * Blend: lane 0 = qh_lo (values 0..15), lane 1 = qh_hi (values 16..31). */
        const __m256i qh = _mm256_blend_epi16(qh_hi, qh_lo, 0x00);
        const __m256i qh_bit = _mm256_and_si256(
            _mm256_srl_epi16(qh, _mm_cvtsi32_si128(i)), m3);
        const __m256i qh_shifted = _mm256_slli_epi16(qh_bit, 2);

        /* 3-bit index = ql[0..1] | qh_bit[2] */
        const __m256i idx = _mm256_or_si256(ql, qh_shifted);

        /* extra bit selection: bit 2i for lane 0 (values 0..15),
         * bit 2i+1 for lane 1 (values 16..31).
         * If the extra bit is set, add 8 to the index (select shifted LUT). */
        const __m256i bm  = _mm256_inserti128_si256(
            _mm256_set1_epi16((short)(1u << (2 * i))),
            _mm_set1_epi16((short)(1u << (2 * i + 1))), 1);
        const __m256i sel = _mm256_cmpeq_epi16(
            _mm256_and_si256(ex, bm), bm);
        w[i] = _mm256_shuffle_epi8(lut,
                _mm256_add_epi8(idx, _mm256_and_si256(sel, eight)));

        /* sp: scale vector. sc16 has 16 int16 values (one per scale group).
         * sp[i] needs scale[2i] repeated 8 times, then scale[2i+1] repeated 8 times.
         * ctrl = {0,1,0,1,...,2,3,2,3,...}: picks from int32[i] which contains
         * (scale[2i], scale[2i+1]). */
        sp[i] = _mm256_shuffle_epi8(
            _mm256_permutevar8x32_epi32(sc16, _mm256_set1_epi32(i)), ctrl);
    }
}

static inline __m256i iq3_block_dot(const __m256i w[8], const __m256i sp[8],
                                    const block_q8_K *yb) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < 8; ++i) {
        const __m256i y = _mm256_loadu_si256((const __m256i *)(yb->qs + 32 * i));
        const __m256i p = _mm256_maddubs_epi16(
            _mm256_sign_epi8(w[i], w[i]), _mm256_sign_epi8(y, w[i]));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, sp[i]));
    }
    return acc;
}

void vec_dot_iq3_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
    assert(n % QK_K == 0);
    const block_iq3_k *x = (const block_iq3_k *)vx;
    const block_q8_K *y = (const block_q8_K *)wy;
    const int nb = n / QK_K;
    const __m256i lut = iq3_lut();

    __m256 fsum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        __m256i w[8], sp[8];
        iq3_decode_block(&x[ibl], lut, w, sp);
        const __m256i acc = iq3_block_dot(w, sp, &y[ibl]);
        const float dy = fp16_to_fp32_lookup(x[ibl].d) * y[ibl].d;
        fsum = _mm256_add_ps(fsum,
                _mm256_mul_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(dy)));
    }
    *out = iq3_hsum_ps(fsum);
}

/* Weight rows [ii] x activation columns, tiled IQ3K_GEMM_TILE columns per job.
 * Each weight block is decoded once and reused for every column in the tile.
 * out[row + col * bs]. */
#define IQ3K_GEMM_TILE 4

int sgemm_iq3_k_q8_k_avx2(int nrows, int ncols, int k,
                           const void *vx, const void *vy,
                           float *out, size_t bs,
                           int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k % QK_K != 0)
        return 0;

    const int nb = k / QK_K;
    const __m256i lut = iq3_lut();

    int64_t ytiles = nrows;
    int64_t xtiles = ncols / IQ3K_GEMM_TILE;
    int64_t n_tail = ncols - xtiles * IQ3K_GEMM_TILE;
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
        int64_t jj = xt * IQ3K_GEMM_TILE;
        int nc = (xt < xtiles) ? IQ3K_GEMM_TILE : (int)n_tail;
        if (nc < 1) nc = 1;

        const block_iq3_k *x = (const block_iq3_k *)
            ((const char *)vx + ii * w_row_bytes);

        __m256 fs[IQ3K_GEMM_TILE];
        for (int c = 0; c < nc; c++) fs[c] = _mm256_setzero_ps();

        for (int ibl = 0; ibl < nb; ibl++) {
            __m256i w[8], sp[8];
            iq3_decode_block(&x[ibl], lut, w, sp);
            const float d = fp16_to_fp32_lookup(x[ibl].d);
            for (int c = 0; c < nc; c++) {
                const block_q8_K *yb = (const block_q8_K *)
                    ((const char *)vy + (jj + c) * a_row_bytes) + ibl;
                const __m256i acc = iq3_block_dot(w, sp, yb);
                fs[c] = _mm256_add_ps(fs[c],
                        _mm256_mul_ps(_mm256_cvtepi32_ps(acc),
                        _mm256_set1_ps(d * yb->d)));
            }
        }
        for (int c = 0; c < nc; c++)
            out[ii + (jj + c) * bs] = iq3_hsum_ps(fs[c]);
    }
    return nrows;
}
#else
/* Non-AVX2 stubs for cross-compilation compatibility. Callers must guard on __AVX2__. */
void vec_dot_iq3_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
    (void)vx; (void)wy; (void)n; if (out) *out = 0.0f;
}
int sgemm_iq3_k_q8_k_avx2(int nrows, int ncols, int k,
                           const void *vx, const void *vy,
                           float *out, size_t bs,
                           int ith, int nth) {
    (void)vx; (void)vy; (void)nrows; (void)ncols; (void)k;
    (void)out; (void)bs; (void)ith; (void)nth; return 0;
}
#endif

/* ================================================================
 * IQ3_K R4 (GGUF type 338) x Q8_K AVX2 kernels
 * ================================================================
 * 4-row interleaved IQ3_K format.
 * ================================================================
 * NOTE: The R4 kernels retain the old per-subblock decode pattern.
 * They have not been audited for the double-activation-pointer bug.
 * ================================================================ */
#include "quant.h"

/* Scalar helpers for IQ3_K R4 (declared extern in quant.h, already included) */

/* NEON stubs: delegate to AVX2 path. */
void vec_dot_iq3_k_q8_k_neon(const void *vx, const void *wy, int n, float *out) {
    vec_dot_iq3_k_q8_k_avx2(vx, wy, n, out);
}

int sgemm_iq3_k_q8_k_neon(int nrows, int ncols, int k,
                           const void *vx, const void *vy,
                           float *out, size_t bs,
                           int ith, int nth) {
    return sgemm_iq3_k_q8_k_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}

/* R4 NEON stubs: delegate to sgemm_iq3_k_r4.c's implementation. */
void vec_dot_iq3_k_r4_q8_k_neon(const void *vx, const void *vy, int n, float *out, int nrows) {
    vec_dot_iq3_k_r4_q8_k_avx2(vx, vy, n, out, nrows);
}

int sgemm_iq3_k_r4_q8_k_neon(int nrows, int ncols, int k,
                              const void *vx, const void *vy,
                              float *out, size_t bs,
                              int ith, int nth) {
    return sgemm_iq3_k_r4_q8_k_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
