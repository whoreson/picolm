/* ================================================================
 * IQ4_NL_R4 x Q8_0 AVX2 / NEON GEMV and GEMM kernels
 * ================================================================
 * Weights:     block_iq4_nl_r4 (GGUF type 220): d[4] fp16, qs[64]
 * Activations: block_q8_0 (32 int8 values + fp16 scale)
 *
 * Layout. qs[16*k + 4*r + i] (k = 0..3, row r = 0..3, i = 0..3) holds two
 * nibbles of row r. Output positions inside the 32-value block:
 *       k   low nibble    high nibble
 *       0       i+0           i+8
 *       1       i+16          i+24
 *       2       i+4           i+12
 *       3       i+20          i+28
 *
 * Method. One 16-byte load qs[16k .. 16k+15] is [row0 x4 | row1 x4 | row2 x4 |
 * row3 x4]: after the nibble LUT every 32-bit lane is the 4 weights of ONE row
 * that meet the SAME 4 activations. So the activation quad is simply the
 * 32-bit word ya[4j..4j+3] broadcast to all lanes, and a 4-byte dot product
 * (dpbusd / maddubs+madd / sdot) yields one int32 per row, in row order. No
 * byte gathers and no horizontal sums:
 *       k   low quad  high quad          (quad j = ya[4j .. 4j+3])
 *       0      j=0       j=2
 *       1      j=4       j=6
 *       2      j=1       j=3
 *       3      j=5       j=7
 * AVX2: low nibbles in the low 128-bit lane, high nibbles in the high lane, so
 * one ymm op covers both. Accumulation stays int32 per block; the float side
 * is  acc4 += float(sum) * d_w[row] * d_y.
 * Weights are decoded once per block and reused for up to 4 columns (GEMM).
 *
 * Int range: |w| <= 127, |y| <= 127; 4 products per lane, 8 per block-lane
 * pair -> |sum| <= 8 * 127 * 127 * 4 = 516K. No int16 accumulation anywhere
 * (maddubs pairs are <= 32258 and are widened immediately).
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "quant.h"

#define IQ4R4_GEMM_TILE 4

/* ================================================================
 * AVX2
 * ================================================================ */
#if defined(__AVX2__)

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#  define IQ4R4_VNNI 1
#endif

static inline float iq4r4_f16(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    return fp16_to_fp32_lookup(h);
#endif
}

/* the 4 per-row fp16 scales of one block -> 4 floats */
static inline __m128 iq4r4_d4(const block_iq4_nl_r4 *b) {
    uint64_t t;
    memcpy(&t, b->d, sizeof(t));
#if defined(__F16C__)
    return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)&t));
#else
    return _mm_setr_ps(fp16_to_fp32_lookup((uint16_t)(t)),
                       fp16_to_fp32_lookup((uint16_t)(t >> 16)),
                       fp16_to_fp32_lookup((uint16_t)(t >> 32)),
                       fp16_to_fp32_lookup((uint16_t)(t >> 48)));
#endif
}

static inline __m256i iq4r4_lut(void) {
    return _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)kvalues_iq4nl));
}

/* w[k]: signed int8 weights. low 128-bit lane = low nibbles of qs[16k..16k+15],
 * high lane = high nibbles. aw[k] = |w[k]| (for the sign trick). */
static inline void iq4r4_decode(const block_iq4_nl_r4 *b, __m256i lut,
                                __m256i w[4], __m256i aw[4]) {
    const __m256i m0f = _mm256_set1_epi8(15);
    for (int k = 0; k < 4; ++k) {
        const __m128i v = _mm_loadu_si128((const __m128i *)(b->qs + 16 * k));
        const __m256i x = _mm256_and_si256(
            _mm256_inserti128_si256(_mm256_castsi128_si256(v), _mm_srli_epi16(v, 4), 1), m0f);
        w[k]  = _mm256_shuffle_epi8(lut, x);
        aw[k] = _mm256_sign_epi8(w[k], w[k]);
    }
}

/* int32x8: low lane = rows 0..3 (low-nibble part), high lane = rows 0..3 (high-nibble part) */
static inline __m256i iq4r4_dot(const __m256i w[4], const __m256i aw[4], const int8_t *ya) {
    const __m256i Y = _mm256_loadu_si256((const __m256i *)ya);          /* dwords d0..d7 */
    const __m256i i0 = _mm256_setr_epi32(0,0,0,0, 2,2,2,2);
    const __m256i i1 = _mm256_setr_epi32(4,4,4,4, 6,6,6,6);
    const __m256i i2 = _mm256_setr_epi32(1,1,1,1, 3,3,3,3);
    const __m256i i3 = _mm256_setr_epi32(5,5,5,5, 7,7,7,7);
    const __m256i y0 = _mm256_permutevar8x32_epi32(Y, i0);
    const __m256i y1 = _mm256_permutevar8x32_epi32(Y, i1);
    const __m256i y2 = _mm256_permutevar8x32_epi32(Y, i2);
    const __m256i y3 = _mm256_permutevar8x32_epi32(Y, i3);
    const __m256i s0 = _mm256_sign_epi8(y0, w[0]);
    const __m256i s1 = _mm256_sign_epi8(y1, w[1]);
    const __m256i s2 = _mm256_sign_epi8(y2, w[2]);
    const __m256i s3 = _mm256_sign_epi8(y3, w[3]);
#if defined(IQ4R4_VNNI)
    __m256i acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), aw[0], s0);
    acc = _mm256_dpbusd_epi32(acc, aw[1], s1);
    acc = _mm256_dpbusd_epi32(acc, aw[2], s2);
    acc = _mm256_dpbusd_epi32(acc, aw[3], s3);
    return acc;
#else
    const __m256i one = _mm256_set1_epi16(1);
    /* widen each maddubs result to int32 before summing (int16 would overflow) */
    __m256i acc = _mm256_madd_epi16(_mm256_maddubs_epi16(aw[0], s0), one);
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(aw[1], s1), one));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(aw[2], s2), one));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(aw[3], s3), one));
    return acc;
#endif
}

/* acc4 += float(rows) * d_w[row] * d_y */
static inline __m128 iq4r4_accum(__m128 acc4, __m256i acc, const block_iq4_nl_r4 *b, uint16_t yd) {
    const __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    const __m128 sc = _mm_mul_ps(iq4r4_d4(b), _mm_set1_ps(iq4r4_f16(yd)));
    return _mm_add_ps(acc4, _mm_mul_ps(_mm_cvtepi32_ps(s4), sc));
}

void vec_dot_iq4_nl_r4_q8_0_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
    assert(nrows == 4);
    assert(n % 32 == 0);
    (void)nrows;

    const block_iq4_nl_r4 *xb = (const block_iq4_nl_r4 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)wy;
    const int nb = n / 32;
    const __m256i lut = iq4r4_lut();

    __m128 acc4 = _mm_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        __m256i w[4], aw[4];
        iq4r4_decode(&xb[ib], lut, w, aw);
        acc4 = iq4r4_accum(acc4, iq4r4_dot(w, aw, y[ib].qs), &xb[ib], y[ib].d);
    }
    _mm_storeu_ps(out, acc4);
}

/* NC is a compile-time constant at every call site -> accumulators stay in registers. */
static inline __attribute__((always_inline))
void iq4r4_gemm_tile(const block_iq4_nl_r4 *wb, const block_q8_0 *a, size_t a_stride_blocks,
                     int nb, const int NC, __m128 res[IQ4R4_GEMM_TILE]) {
    const __m256i lut = iq4r4_lut();
    __m128 acc4[IQ4R4_GEMM_TILE];
    for (int c = 0; c < NC; ++c) acc4[c] = _mm_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
        __m256i w[4], aw[4];
        iq4r4_decode(&wb[ib], lut, w, aw);
        for (int c = 0; c < NC; ++c) {
            const block_q8_0 *ac = a + (size_t)c * a_stride_blocks + ib;
            acc4[c] = iq4r4_accum(acc4[c], iq4r4_dot(w, aw, ac->qs), &wb[ib], ac->d);
        }
    }
    for (int c = 0; c < NC; ++c) res[c] = acc4[c];
}

/* out[nc * bs + row] */
int sgemm_iq4_nl_r4_q8_0_avx2(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth) {
    if (k % 32 != 0) return 0;
    const int nb = k / 32;

    const int row_groups = (nrows + 3) / 4;
    const int rg_per_thread = (row_groups + nth - 1) / nth;
    const int rg_start = ith * rg_per_thread;
    int rg_end = rg_start + rg_per_thread;
    if (rg_end > row_groups) rg_end = row_groups;

    const block_iq4_nl_r4 *w = (const block_iq4_nl_r4 *)vx;
    const block_q8_0 *a = (const block_q8_0 *)vy;

    for (int rg = rg_start; rg < rg_end; ++rg) {
        const int base_row = rg * 4;
        const int rows_here = (nrows - base_row) < 4 ? (nrows - base_row) : 4;
        const block_iq4_nl_r4 *wb = w + (size_t)rg * (size_t)nb;

        for (int c0 = 0; c0 < ncols; c0 += IQ4R4_GEMM_TILE) {
            const int nc = (ncols - c0) < IQ4R4_GEMM_TILE ? (ncols - c0) : IQ4R4_GEMM_TILE;
            const block_q8_0 *ac = a + (size_t)c0 * (size_t)nb;
            __m128 res[IQ4R4_GEMM_TILE];
            switch (nc) {
                case 1: iq4r4_gemm_tile(wb, ac, (size_t)nb, nb, 1, res); break;
                case 2: iq4r4_gemm_tile(wb, ac, (size_t)nb, nb, 2, res); break;
                case 3: iq4r4_gemm_tile(wb, ac, (size_t)nb, nb, 3, res); break;
                default: iq4r4_gemm_tile(wb, ac, (size_t)nb, nb, 4, res); break;
            }
            for (int c = 0; c < nc; ++c) {
                float tmp[4];
                _mm_storeu_ps(tmp, res[c]);
                for (int r = 0; r < rows_here; ++r)
                    out[(size_t)(c0 + c) * bs + base_row + r] = tmp[r];
            }
        }
    }
    return 1;
}

#endif /* __AVX2__ */

/* ================================================================
 * NEON (same lane-per-row scheme; sdot when available)
 * ================================================================ */
#if defined(__ARM_NEON)

/* acc lane r += sum_{i<4} w[4r+i] * y[4r+i]   (y = one quad broadcast to all lanes) */
static inline int32x4_t iq4r4_neon_dot(int32x4_t acc, int8x16_t w, int8x16_t y) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, w, y);
#else
    const int32x4_t lo = vpaddlq_s16(vmull_s8(vget_low_s8(w),  vget_low_s8(y)));   /* r0a r0b r1a r1b */
    const int32x4_t hi = vpaddlq_s16(vmull_s8(vget_high_s8(w), vget_high_s8(y)));  /* r2a r2b r3a r3b */
    return vaddq_s32(acc, vpaddq_s32(lo, hi));                                      /* r0 r1 r2 r3 */
#endif
}

static inline int8x16_t iq4r4_quad(const uint32_t *yq, int j) {
    return vreinterpretq_s8_u32(vdupq_n_u32(yq[j]));
}

/* wlo[k] / whi[k]: weights of low / high nibbles of qs[16k..16k+15] */
static inline void iq4r4_neon_decode(const block_iq4_nl_r4 *b, int8x16_t lut,
                                     int8x16_t wlo[4], int8x16_t whi[4]) {
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    for (int k = 0; k < 4; ++k) {
        const uint8x16_t v = vld1q_u8(b->qs + 16 * k);
        wlo[k] = vqtbl1q_s8(lut, vandq_u8(v, m0f));
        whi[k] = vqtbl1q_s8(lut, vshrq_n_u8(v, 4));
    }
}

static inline int32x4_t iq4r4_neon_block(const int8x16_t wlo[4], const int8x16_t whi[4], const int8_t *ya) {
    uint32_t yq[8];
    memcpy(yq, ya, sizeof(yq));
    int32x4_t acc = vdupq_n_s32(0);
    acc = iq4r4_neon_dot(acc, wlo[0], iq4r4_quad(yq, 0));
    acc = iq4r4_neon_dot(acc, whi[0], iq4r4_quad(yq, 2));
    acc = iq4r4_neon_dot(acc, wlo[1], iq4r4_quad(yq, 4));
    acc = iq4r4_neon_dot(acc, whi[1], iq4r4_quad(yq, 6));
    acc = iq4r4_neon_dot(acc, wlo[2], iq4r4_quad(yq, 1));
    acc = iq4r4_neon_dot(acc, whi[2], iq4r4_quad(yq, 3));
    acc = iq4r4_neon_dot(acc, wlo[3], iq4r4_quad(yq, 5));
    acc = iq4r4_neon_dot(acc, whi[3], iq4r4_quad(yq, 7));
    return acc;
}

static inline float32x4_t iq4r4_neon_accum(float32x4_t acc4, int32x4_t s, const block_iq4_nl_r4 *b, uint16_t yd) {
    float d[4];
    const float yf = fp16_to_fp32_lookup(yd);
    for (int r = 0; r < 4; ++r) d[r] = fp16_to_fp32_lookup(b->d[r]) * yf;
    return vfmaq_f32(acc4, vcvtq_f32_s32(s), vld1q_f32(d));
}

void vec_dot_iq4_nl_r4_q8_0_neon(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
    assert(nrows == 4);
    assert(n % 32 == 0);
    (void)nrows;

    const block_iq4_nl_r4 *xb = (const block_iq4_nl_r4 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)wy;
    const int nb = n / 32;
    const int8x16_t lut = vld1q_s8(kvalues_iq4nl);

    float32x4_t acc4 = vdupq_n_f32(0.0f);
    for (int ib = 0; ib < nb; ++ib) {
        int8x16_t wlo[4], whi[4];
        iq4r4_neon_decode(&xb[ib], lut, wlo, whi);
        acc4 = iq4r4_neon_accum(acc4, iq4r4_neon_block(wlo, whi, y[ib].qs), &xb[ib], y[ib].d);
    }
    vst1q_f32(out, acc4);
}

int sgemm_iq4_nl_r4_q8_0_neon(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth) {
    if (k % 32 != 0) return 0;
    const int nb = k / 32;

    const int row_groups = (nrows + 3) / 4;
    const int rg_per_thread = (row_groups + nth - 1) / nth;
    const int rg_start = ith * rg_per_thread;
    int rg_end = rg_start + rg_per_thread;
    if (rg_end > row_groups) rg_end = row_groups;

    const block_iq4_nl_r4 *w = (const block_iq4_nl_r4 *)vx;
    const block_q8_0 *a = (const block_q8_0 *)vy;
    const int8x16_t lut = vld1q_s8(kvalues_iq4nl);

    for (int rg = rg_start; rg < rg_end; ++rg) {
        const int base_row = rg * 4;
        const int rows_here = (nrows - base_row) < 4 ? (nrows - base_row) : 4;
        const block_iq4_nl_r4 *wb = w + (size_t)rg * (size_t)nb;

        for (int c0 = 0; c0 < ncols; c0 += IQ4R4_GEMM_TILE) {
            const int nc = (ncols - c0) < IQ4R4_GEMM_TILE ? (ncols - c0) : IQ4R4_GEMM_TILE;
            float32x4_t acc4[IQ4R4_GEMM_TILE];
            for (int c = 0; c < nc; ++c) acc4[c] = vdupq_n_f32(0.0f);

            for (int ib = 0; ib < nb; ++ib) {
                int8x16_t wlo[4], whi[4];
                iq4r4_neon_decode(&wb[ib], lut, wlo, whi);      /* decoded once per tile */
                for (int c = 0; c < nc; ++c) {
                    const block_q8_0 *ac = a + (size_t)(c0 + c) * (size_t)nb + ib;
                    acc4[c] = iq4r4_neon_accum(acc4[c], iq4r4_neon_block(wlo, whi, ac->qs), &wb[ib], ac->d);
                }
            }
            for (int c = 0; c < nc; ++c) {
                float tmp[4];
                vst1q_f32(tmp, acc4[c]);
                for (int r = 0; r < rows_here; ++r)
                    out[(size_t)(c0 + c) * bs + base_row + r] = tmp[r];
            }
        }
    }
    return 1;
}

#endif /* __ARM_NEON */

/* ================================================================
 * Non-AVX2 builds (ARM, plain x86-64): tensor.c's r4_dual_lookup() table refers to the
 * _avx2-named symbols unconditionally, so they must exist at link time. They are only
 * ever called from PICOLM_AVX2 code paths. Abort loudly if that ever stops being true.
 * ================================================================ */
#if !defined(__AVX2__)
void vec_dot_iq4_nl_r4_q8_0_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
    (void)vx; (void)wy; (void)n; (void)out; (void)nrows;
    fprintf(stderr, "FATAL: vec_dot_iq4_nl_r4_q8_0_avx2 called in a build without AVX2\n");
    abort();
}
int sgemm_iq4_nl_r4_q8_0_avx2(int nrows, int ncols, int k, const void *vx, const void *vy,
                               float *out, size_t bs, int ith, int nth) {
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth;
    return 0;   /* "not handled" */
}
#endif
