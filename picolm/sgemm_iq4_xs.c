/* ================================================================
 * IQ4_XS x Q8_K kernels: AVX2 GEMV, AVX2 tiled GEMM, NEON GEMV
 * ================================================================
 * block_iq4_xs: d (fp16), scales_h (16 bit), scales_l[4], qs[128].
 *   sub-block j (0..7) = values 32j..32j+31 = qs[16j..16j+15]:
 *     low nibbles -> values 32j+0..15, high nibbles -> values 32j+16..31
 *   scale_j = ((scales_l[j/2] >> 4*(j%2)) & 15 | ((scales_h >> 2j) & 3) << 4) - 32
 *   value = d * scale_j * kvalues_iq4nl[nibble]
 *
 * AVX2 method (per 256-value block):
 *   - one 32-byte load = two sub-blocks. The LUT is applied to the low-nibble
 *     and the high-nibble vectors (both lanes at once). GEMM: one vperm2i128
 *     each puts [low16 | high16] of the SAME sub-block back together (done once
 *     per weight block, shared by all activation rows). GEMV: no permutes; the
 *     activation vectors are built to match instead (vinserti128 from memory
 *     does not compete with pshufb for port 5).
 *   - the 8 scales are expanded with vector ops (srlv/or/sub) into one
 *     register of int16 pairs; sp_j = permutevar8x32(S2, j) is the broadcast
 *     scale for sub-block j. No scalar scale arithmetic in the loop.
 *   - d is converted inline (F16C) instead of calling fp16_to_fp32_lookup():
 *     a real call per block forces every live ymm register to be spilled.
 *   - dot: sign trick + maddubs (int16, |pair| <= 32258) + madd_epi16 with
 *     sp_j (applies the scale and widens in one op). int32 acc per block,
 *     one float conversion per block, one horizontal sum per row.
 *   - GEMM: the weight block (w, |w|, sp) is decoded ONCE and reused for
 *     IQ4XS_GEMM_TILE activation rows (compile-time constant -> accumulators
 *     stay in registers). Work item = (weight row, tile of activation rows).
 * Int range: |sum| <= 8 * 2 * 32258 * 32 = 16.5M.
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "quant.h"

/* activation rows per GEMM work item (measured: 2 -> 9.2, 4 -> 6.8, 8 -> 5.9 ns per block-column) */
#ifndef IQ4XS_GEMM_TILE
#define IQ4XS_GEMM_TILE 8
#endif

#if defined(__AVX2__)

static inline float xs_f16(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    return fp16_to_fp32_lookup(h);
#endif
}

static inline __m256i xs_lut(void) {
    return _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)kvalues_iq4nl));
}

/* int32 lane pair (scale_j, scale_j) per sub-block j, as int16 x2 per 32-bit lane */
static inline __m256i xs_scales(const block_iq4_xs *x) {
    uint32_t sl32;
    memcpy(&sl32, x->scales_l, sizeof(sl32));
    const __m128i v   = _mm_cvtsi32_si128((int)sl32);
    const __m128i m4  = _mm_set1_epi8(0x0f);
    const __m128i lo  = _mm_and_si128(v, m4);
    const __m128i hi  = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
    const __m256i low4 = _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi));          /* s0..s7 low 4 bits */
    const __m256i high2 = _mm256_and_si256(
        _mm256_srlv_epi32(_mm256_set1_epi32((int)x->scales_h),
                          _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14)),
        _mm256_set1_epi32(3));
    const __m256i S = _mm256_sub_epi32(_mm256_or_si256(low4, _mm256_slli_epi32(high2, 4)),
                                       _mm256_set1_epi32(32));                      /* scale_j, int32 */
    return _mm256_or_si256(_mm256_and_si256(S, _mm256_set1_epi32(0xffff)),
                           _mm256_slli_epi32(S, 16));                               /* (s,s) as int16 pair */
}

static inline __m256i xs_sp(__m256i S2, int j) {
    return _mm256_permutevar8x32_epi32(S2, _mm256_set1_epi32(j));
}

/* sub-blocks 2i and 2i+1 -> w0, w1 (signed int8, values in natural order) */
static inline void xs_decode_pair(const block_iq4_xs *x, __m256i lut, int i, __m256i *w0, __m256i *w1) {
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m256i q  = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * i));      /* [sb 2i | sb 2i+1] */
    const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(q, m4));           /* [lo(2i)  | lo(2i+1)] */
    const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
    *w0 = _mm256_permute2x128_si256(lo, hi, 0x20);                                  /* [lo(2i)   | hi(2i)]   */
    *w1 = _mm256_permute2x128_si256(lo, hi, 0x31);                                  /* [lo(2i+1) | hi(2i+1)] */
}

static inline __m256i xs_mac(__m256i aw, __m256i w, __m256i y, __m256i sp) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(y, w)), sp);
}

static inline float xs_hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

static inline __m256 xs_fma(__m256 a, __m256 b, __m256 c) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

/* ---------------------------------------------------------------- GEMV
 * Single activation row: instead of re-joining [low16 | high16] of each sub-block with
 * vperm2i128 (port 5, same as pshufb), keep the LUT output as [lo(2i) | lo(2i+1)] and
 * [hi(2i) | hi(2i+1)] and build the matching activation vectors with
 * vinserti128-from-memory (runs on any vector ALU port). One scale vector per PAIR:
 * lane 0 = scale(2i), lane 1 = scale(2i+1), valid for both the lo and the hi vector. */
static inline __m256i xs_load2(const int8_t *p0, const int8_t *p1) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)p0)),
                                   _mm_loadu_si128((const __m128i *)p1), 1);
}

void vec_dot_iq4_xs_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
    const block_iq4_xs *x = (const block_iq4_xs *)vx;
    const block_q8_K *y = (const block_q8_K *)wy;
    const int nb = n / QK_K;
    const __m256i lut = xs_lut();
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m256i pidx0 = _mm256_setr_epi32(0,0,0,0, 1,1,1,1);
    const __m256i pidx1 = _mm256_setr_epi32(2,2,2,2, 3,3,3,3);
    const __m256i pidx2 = _mm256_setr_epi32(4,4,4,4, 5,5,5,5);
    const __m256i pidx3 = _mm256_setr_epi32(6,6,6,6, 7,7,7,7);

    __m256 facc = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        const block_iq4_xs *xb = &x[ibl];
        const __m256i S2 = xs_scales(xb);
        const int8_t *q8 = y[ibl].qs;
        const __m256i spp[4] = { _mm256_permutevar8x32_epi32(S2, pidx0), _mm256_permutevar8x32_epi32(S2, pidx1),
                                 _mm256_permutevar8x32_epi32(S2, pidx2), _mm256_permutevar8x32_epi32(S2, pidx3) };
        __m256i acc = _mm256_setzero_si256();
        for (int i = 0; i < 4; ++i) {
            const __m256i q  = _mm256_loadu_si256((const __m256i *)(xb->qs + 32 * i));
            const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(q, m4));
            const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
            const __m256i yl = xs_load2(q8 + 64 * i,      q8 + 64 * i + 32);   /* [y(2i)[0..15]  | y(2i+1)[0..15]]  */
            const __m256i yh = xs_load2(q8 + 64 * i + 16, q8 + 64 * i + 48);   /* [y(2i)[16..31] | y(2i+1)[16..31]] */
            acc = _mm256_add_epi32(acc, xs_mac(_mm256_sign_epi8(lo, lo), lo, yl, spp[i]));
            acc = _mm256_add_epi32(acc, xs_mac(_mm256_sign_epi8(hi, hi), hi, yh, spp[i]));
        }
        const float dd = xs_f16(xb->d) * y[ibl].d;
        facc = xs_fma(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(dd), facc);
    }
    *out = xs_hsum(facc);
}

/* ---------------------------------------------------------------- GEMM
 * C[ldc * col + row]; A: m rows of k_blocks blocks (row stride lda blocks);
 * B: n rows of k_blocks Q8_K blocks (row stride ldb blocks). */
static inline __attribute__((always_inline))
void xs_tile(const block_iq4_xs *arow, const block_q8_K *b0, int ldb, int k_blocks,
             const __m256i lut, const int NC, float *res) {
    __m256 facc[IQ4XS_GEMM_TILE];
    for (int c = 0; c < NC; ++c) facc[c] = _mm256_setzero_ps();

    for (int blk = 0; blk < k_blocks; ++blk) {
        const block_iq4_xs *x = arow + blk;
        __m256i w[8], aw[8], sp[8];
        const __m256i S2 = xs_scales(x);
        for (int i = 0; i < 4; ++i) {
            xs_decode_pair(x, lut, i, &w[2 * i], &w[2 * i + 1]);
        }
        for (int j = 0; j < 8; ++j) { aw[j] = _mm256_sign_epi8(w[j], w[j]); sp[j] = xs_sp(S2, j); }
        const float d = xs_f16(x->d);

        for (int c = 0; c < NC; ++c) {
            const block_q8_K *yb = b0 + (size_t)c * ldb + blk;
            __m256i acc = _mm256_setzero_si256();
            for (int j = 0; j < 8; ++j) {
                acc = _mm256_add_epi32(acc,
                      xs_mac(aw[j], w[j], _mm256_loadu_si256((const __m256i *)(yb->qs + 32 * j)), sp[j]));
            }
            facc[c] = xs_fma(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(d * yb->d), facc[c]);
        }
    }
    for (int c = 0; c < NC; ++c) res[c] = xs_hsum(facc[c]);
}

static void sgemm_iq4xs_q8k(int m, int n, int k_blocks,
                            const block_iq4_xs *A, int lda,
                            const block_q8_K *B, int ldb,
                            float *C, int ldc, int ith, int nth) {
    const __m256i lut = xs_lut();
    const int64_t n_tiles = (n + IQ4XS_GEMM_TILE - 1) / IQ4XS_GEMM_TILE;
    const int64_t total = (int64_t)m * n_tiles;
    if (total <= 0) return;

    const int64_t duty = (total + nth - 1) / nth;
    const int64_t start = duty * ith;
    int64_t end = start + duty;
    if (end > total) end = total;

    for (int64_t job = start; job < end; ++job) {
        const int64_t ii = job / n_tiles;                /* weight row */
        const int64_t jj = (job % n_tiles) * IQ4XS_GEMM_TILE;
        const int nc = (n - jj) < IQ4XS_GEMM_TILE ? (int)(n - jj) : IQ4XS_GEMM_TILE;
        const block_iq4_xs *arow = A + (size_t)lda * ii;
        const block_q8_K *b0 = B + (size_t)ldb * jj;
        float res[IQ4XS_GEMM_TILE];
        switch (nc) {
            case 1:  xs_tile(arow, b0, ldb, k_blocks, lut, 1, res); break;
            case 2:  xs_tile(arow, b0, ldb, k_blocks, lut, 2, res); break;
            case 3:  xs_tile(arow, b0, ldb, k_blocks, lut, 3, res); break;
#if IQ4XS_GEMM_TILE >= 8
            case 4:  xs_tile(arow, b0, ldb, k_blocks, lut, 4, res); break;
            case 5:  xs_tile(arow, b0, ldb, k_blocks, lut, 5, res); break;
            case 6:  xs_tile(arow, b0, ldb, k_blocks, lut, 6, res); break;
            case 7:  xs_tile(arow, b0, ldb, k_blocks, lut, 7, res); break;
            default: xs_tile(arow, b0, ldb, k_blocks, lut, 8, res); break;
#else
            default: xs_tile(arow, b0, ldb, k_blocks, lut, 4, res); break;
#endif
        }
        for (int c = 0; c < nc; ++c) C[(size_t)ldc * (jj + c) + ii] = res[c];
    }
}

#endif /* __AVX2__ */

/* IQ4_XS x Q8_K tiled GEMM wrapper (called from tensor.c). Returns 0 = not handled. */
int picolm_sgemm_d_iq4xs(int m, int n, int k_blocks_q4xs,
                         const void *A, int lda_q4xs,
                         const void *B, int ldb_q8k,
                         float *C, int ldc, int ith, int nth) {
#if defined(__AVX2__)
    sgemm_iq4xs_q8k(m, n, k_blocks_q4xs, (const block_iq4_xs *)A, lda_q4xs,
                    (const block_q8_K *)B, ldb_q8k, C, ldc, ith, nth);
    return 1;
#else
    (void)m; (void)n; (void)k_blocks_q4xs; (void)A; (void)lda_q4xs;
    (void)B; (void)ldb_q8k; (void)C; (void)ldc; (void)ith; (void)nth;
    return 0;
#endif
}

#if !defined(__AVX2__)
/* tensor.c / quant.c only call this under PICOLM_AVX2; keep the symbol for linking. */
void vec_dot_iq4_xs_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
    (void)vx; (void)wy; (void)n; (void)out;
    fprintf(stderr, "FATAL: vec_dot_iq4_xs_q8_k_avx2 called in a build without AVX2\n");
    abort();
}
#endif

/* ---------------------------------------------------------------- NEON GEMV
 * Fixed: vmlaq_n_s32 requires int32x4_t, not int32x2_t. */
#if defined(__ARM_NEON)
void vec_dot_iq4_xs_q8_k_neon(const void *vx, const void *wy, int n, float *out) {
    const block_iq4_xs *x = (const block_iq4_xs *)vx;
    const block_q8_K *y = (const block_q8_K *)wy;
    const int nb = n / QK_K;
    const int8x16_t lut = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    float sumf = 0.0f;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t *qs = x[ibl].qs;
        const int8_t *q8 = y[ibl].qs;
        int32x4_t sumi = vdupq_n_s32(0);
        for (int ib = 0; ib < 8; ++ib) {
            const uint8x16_t q4 = vld1q_u8(qs + 16 * ib);
            const int8x16_t wlo = vqtbl1q_s8(lut, vandq_u8(q4, m4));     /* values 32ib + 0..15  */
            const int8x16_t whi = vqtbl1q_s8(lut, vshrq_n_u8(q4, 4));    /* values 32ib + 16..31 */
            const int8x16_t ylo = vld1q_s8(q8 + 32 * ib);
            const int8x16_t yhi = vld1q_s8(q8 + 32 * ib + 16);
#if defined(__ARM_FEATURE_DOTPROD)
            int32x4_t p = vdotq_s32(vdupq_n_s32(0), wlo, ylo);
            p = vdotq_s32(p, whi, yhi);
#else
            int32x4_t p = vaddq_s32(
                vpaddlq_s16(vmull_s8(vget_low_s8(wlo),  vget_low_s8(ylo))),
                vpaddlq_s16(vmull_s8(vget_high_s8(wlo), vget_high_s8(ylo))));
            p = vaddq_s32(p, vaddq_s32(
                vpaddlq_s16(vmull_s8(vget_low_s8(whi),  vget_low_s8(yhi))),
                vpaddlq_s16(vmull_s8(vget_high_s8(whi), vget_high_s8(yhi)))));
#endif
            const int ls = (int)(((x[ibl].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                                 (((x[ibl].scales_h >> (2 * ib)) & 3) << 4)) - 32;
            sumi = vmlaq_n_s32(sumi, p, ls);
        }
        sumf += fp16_to_fp32_lookup(x[ibl].d) * y[ibl].d * (float)vaddvq_s32(sumi);
    }
    *out = sumf;
}
#endif /* __ARM_NEON */
