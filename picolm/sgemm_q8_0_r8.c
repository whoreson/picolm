/* ================================================================
 * Q8_0_R8 x Q8_0 AVX2 GEMV and GEMM kernels
 * ================================================================
 * Weights: block_q8_0_r8 (GGUF type 203/208, ik_llama transposed layout)
 *   d[8] FP16 scales (one per row) + qs[256] interleaved int8
 *   Layout per 32-value block (ik_llama repack_q8_0):
 *     qs[32*l + 4*k + i     ] = row k, value 4*l + i      (values  0-15)
 *     qs[32*l + 4*k + i +128] = row k, value 16 + 4*l + i  (values 16-31)
 *   for l=0..3, k=0..7, i=0..3.
 *   I.e. within each 32-byte chunk l (at qs[32*l] and qs[128+32*l]),
 *   byte offset 4*k+i holds row k's value (4*l+i) or (16+4*l+i).
 *
 * Activations: block_q8_0 (plain, one per row)
 *   d FP16 scale + qs[32] signed int8
 *
 * AVX2 GEMV: for each block, 8 __m256i dpbusd-style operations produce
 *   8 int32 lane results (one per row). The activation dword containing
 *   values [4*l .. 4*l+3] is broadcast to all 8 lanes via shuffle_epi32,
 *   then sign-trick maddubs+madd gives per-row int32 partial dots.
 *
 * AVX2 + FMA + F16C required.
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "quant.h"

/* Size gate for the decode-once R8 prefill paths (Q8_0 / Q4_0 / IQ4_NL / IQ4_NL_R4 / IQ4_XS).
 * Those kernels are numerically correct (~1e-7 against an independent reference) but accumulate
 * floats in a different order than the reference tile kernels (integer sum over the whole block,
 * one float accumulator per row, vs eight per-lane float accumulators), so their results are not
 * bit-identical to the reference path or to the NEON kernels. For toy-sized matrices that is all
 * downside: they gain nothing, and bit-exact output is what tiny test models / CI rely on
 * (PICOLM_GEMM_TEST, the smoke tests' expected text). So they only engage when
 * rows * 32-value-blocks-per-row >= PICOLM_R8_MIN_BLOCKS (default 16384, roughly a 512x1024
 * matrix). Set PICOLM_R8_MIN_BLOCKS=0 to always allow them (the benches do). */
int picolm_r8_big_enough(int m, int k_blocks) {
    static long mn = -1;
    if (mn < 0) { const char *e = getenv("PICOLM_R8_MIN_BLOCKS"); mn = (e && *e) ? atol(e) : 16384; if (mn < 0) mn = 0; }
    return (long)m * (long)k_blocks >= mn;
}


#if defined(__AVX2__) && defined(__F16C__)

/* ================================================================
 * gemv_r8_block: 8-row x 1-block dot product (inner kernel).
 * Computes 8 int32 dot products for one block_q8_0_r8 group
 * against one block_q8_0 activation, using the transposed layout.
 * Returns __m256i with lane r = dot(row r, activation) for this block.
 * ================================================================ */
static inline __m256i gemv_r8_block(const block_q8_0_r8 *w, const block_q8_0 *q) {
    const __m256i m1 = _mm256_set1_epi16(1);
    const __m128i y4l = _mm_loadu_si128((const __m128i *)(q->qs));
    const __m128i y4h = _mm_loadu_si128((const __m128i *)(q->qs + 16));
    const __m256i yl = _mm256_set_m128i(y4l, y4l);
    const __m256i yh = _mm256_set_m128i(y4h, y4h);
    __m256i sumi = _mm256_setzero_si256();
    const __m256i *wq = (const __m256i *)w->qs;
    /* First half: chunks 0-3, activation dwords 0-3 from yl */
    {
        const __m256i wv = _mm256_loadu_si256(wq + 0);
        const __m256i yb = _mm256_shuffle_epi32(yl, _MM_SHUFFLE(0,0,0,0));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(m1,
            _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(yb, wv))));
    }
    {
        const __m256i wv = _mm256_loadu_si256(wq + 1);
        const __m256i yb = _mm256_shuffle_epi32(yl, _MM_SHUFFLE(1,1,1,1));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(m1,
            _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(yb, wv))));
    }
    {
        const __m256i wv = _mm256_loadu_si256(wq + 2);
        const __m256i yb = _mm256_shuffle_epi32(yl, _MM_SHUFFLE(2,2,2,2));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(m1,
            _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(yb, wv))));
    }
    {
        const __m256i wv = _mm256_loadu_si256(wq + 3);
        const __m256i yb = _mm256_shuffle_epi32(yl, _MM_SHUFFLE(3,3,3,3));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(m1,
            _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(yb, wv))));
    }
    /* Second half: chunks 4-7, activation dwords 0-3 from yh */
    {
        const __m256i wv = _mm256_loadu_si256(wq + 4);
        const __m256i yb = _mm256_shuffle_epi32(yh, _MM_SHUFFLE(0,0,0,0));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(m1,
            _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(yb, wv))));
    }
    {
        const __m256i wv = _mm256_loadu_si256(wq + 5);
        const __m256i yb = _mm256_shuffle_epi32(yh, _MM_SHUFFLE(1,1,1,1));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(m1,
            _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(yb, wv))));
    }
    {
        const __m256i wv = _mm256_loadu_si256(wq + 6);
        const __m256i yb = _mm256_shuffle_epi32(yh, _MM_SHUFFLE(2,2,2,2));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(m1,
            _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(yb, wv))));
    }
    {
        const __m256i wv = _mm256_loadu_si256(wq + 7);
        const __m256i yb = _mm256_shuffle_epi32(yh, _MM_SHUFFLE(3,3,3,3));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(m1,
            _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(yb, wv))));
    }
    return sumi;
}

/* ================================================================
 * vec_dot_q8_0_r8_q8_0_avx2: 8 weight rows x 1 activation row
 * out[r] = sum_ib dot(w8[ib], row r) * fp16(w8[ib].d[r]) * fp16(qy[ib].d)
 * ================================================================ */
void vec_dot_q8_0_r8_q8_0_avx2(const void *vx, const void *wy, int n,
                               float *out, int nrows) {
    assert(nrows == 8);
    assert(n % 32 == 0);
    if (!vx || !wy || !out) { for (int r = 0; r < nrows; r++) out[r] = 0; return; }

    const block_q8_0_r8 *w8 = (const block_q8_0_r8 *)vx;
    const block_q8_0 *qy = (const block_q8_0 *)wy;
    const int nb = n / 32;
    const __m256i m1 = _mm256_set1_epi16(1);

    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ib++) {
        __m256 w_scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)w8[ib].d));
        const float y_scale = fp16_to_fp32_lookup(qy[ib].d);
        __m256i sumi = gemv_r8_block(w8 + ib, qy + ib);
        __m256 dot_f = _mm256_cvtepi32_ps(sumi);
        __m256 scaled = _mm256_mul_ps(dot_f, w_scales);
        acc = _mm256_fmadd_ps(scaled, _mm256_set1_ps(y_scale), acc);
    }

    _mm256_storeu_ps(out, acc);
}

/* ================================================================
 * repack_q8_0_to_q8_0_r8: plain Q8_0 rows -> block_q8_0_r8 groups.
 *
 * Converts `nrows` independent plain Q8_0 rows (row stride = n/32 blocks)
 * into nrows/8 interleaved Q8_0_R8 groups, using the same layout as the
 * ik_llama repack (see the layout diagram at the top of this file):
 *   qs[32*l + 4*k + i     ] = row k, value 4*l + i      (values  0-15)
 *   qs[32*l + 4*k + i +128] = row k, value 16 + 4*l + i  (values 16-31)
 * Deltas are copied verbatim (already fp16 in both formats), so the GEMM
 * sees exactly the same quantization as plain Q8_0.
 *
 * Used by the tiled prefill attention path to turn the dequantized-from-
 * cache K tile (relaid out as contiguous plain Q8_0 rows) into GEMM-ready
 * weights. Identical on every platform: it has no intrinsics.
 * ================================================================ */

void repack_q8_0_to_q8_0_r8(const void *src, void *dst, int nrows, int n) {
    if (!src || !dst || nrows <= 0 || n <= 0 || n % 32 != 0 || nrows % 8 != 0) return;
    const int nb = n / 32;
    const block_q8_0 *s = (const block_q8_0 *)src;
    block_q8_0_r8 *d = (block_q8_0_r8 *)dst;
    for (int g = 0; g < nrows / 8; g++) {
        const block_q8_0 *sg = s + (size_t)g * 8 * nb;
        block_q8_0_r8 *dg = d + (size_t)g * nb;
        for (int ib = 0; ib < nb; ib++) {
            for (int r = 0; r < 8; r++) dg[ib].d[r] = sg[(size_t)r * nb + ib].d;
            for (int l = 0; l < 4; l++) {
                for (int k = 0; k < 8; k++) {
                    const int8_t *qs = sg[(size_t)k * nb + ib].qs;
                    for (int i = 0; i < 4; i++) {
                        dg[ib].qs[32 * l + 4 * k + i]       = qs[4 * l + i];
                        dg[ib].qs[32 * l + 4 * k + i + 128] = qs[4 * l + i + 16];
                    }
                }
            }
        }
    }
}

/* Same as vec_dot_q8_0_r8_q8_0_avx2 but with caller-supplied activation
 * deltas (q_d may be NULL -> use the fp16 delta stored in the blocks).
 * Mirrors picolm_sgemm_d's B_d convention: the batched matmul paths scale
 * each Q8_0 block by delta(q) * delta(w) instead of the fp16 block scale,
 * so the tiled attention path can reproduce matmul_batch numerics. */
void vec_dot_q8_0_r8_q8_0_d_avx2(const void *vx, const void *wy, const float *q_d,
                                 int n, float *out, int nrows) {
    assert(nrows == 8);
    assert(n % 32 == 0);
    if (!vx || !wy || !out) { for (int r = 0; r < nrows; r++) out[r] = 0; return; }

    const block_q8_0_r8 *w8 = (const block_q8_0_r8 *)vx;
    const block_q8_0 *qy = (const block_q8_0 *)wy;
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ib++) {
        __m256 w_scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)w8[ib].d));
        const float y_scale = q_d ? q_d[ib] : fp16_to_fp32_lookup(qy[ib].d);
        __m256i sumi = gemv_r8_block(w8 + ib, qy + ib);
        __m256 dot_f = _mm256_cvtepi32_ps(sumi);
        __m256 scaled = _mm256_mul_ps(dot_f, w_scales);
        acc = _mm256_fmadd_ps(scaled, _mm256_set1_ps(y_scale), acc);
    }

    _mm256_storeu_ps(out, acc);
}

/* ================================================================
 * sgemm_q8_0_r8_q8_0_d_avx2: GEMM, 8 weight rows at a time
 * nrows: total weight rows (d), multiple of 8
 * ncols: activation columns (batch size)
 * k: shared dimension in values (multiple of 32)
 * vx: block_q8_0_r8 weights, group stride = (k/32) * sizeof(block_q8_0_r8)
 * vy: block_q8_0 activations, column stride = (k/32) * sizeof(block_q8_0)
 * out: float output [ncols][nrows] (out[c * bs + r])
 * bs: output row stride (typically d)
 * ith/nth: thread splitting over ncols
 * ================================================================ */
int sgemm_q8_0_r8_q8_0_d_avx2(int nrows, int ncols, int k,
                              const void *vx, const void *vy,
                              const float *vy_d, int vy_d_stride,
                              float *out, size_t bs, int ith, int nth) {
    if (nrows < 8 || ncols < 1 || k % 32 != 0 || nrows % 8 != 0)
        return 0;

    const int nb = k / 32;
    const block_q8_0_r8 *w8 = (const block_q8_0_r8 *)vx;
    const size_t a_stride = (size_t)nb * sizeof(block_q8_0);
    const size_t w_stride = (size_t)nb * sizeof(block_q8_0_r8);

    const int ncols_per_thread = (ncols + nth - 1) / nth;
    const int col_start = ith * ncols_per_thread;
    const int col_end = (ith + 1) * ncols_per_thread < ncols ? (ith + 1) * ncols_per_thread : ncols;

    /* 8 rows x 4 columns tile: load weight group once into L1, process 4
     * activation columns per pass over the K dimension. This amortizes
     * the weight load over 4 columns instead of 1. */
    {
        const int total_groups = nrows / 8;
        const int groups_per_thread = (total_groups + nth - 1) / nth;
        const int g_start = ith * groups_per_thread;
        const int g_end = (ith + 1) * groups_per_thread < total_groups ?
                          (ith + 1) * groups_per_thread : total_groups;
        const int nb4 = nb / 4; /* groups of 4 K-blocks for unrolling */

        for (int w8_idx = g_start; w8_idx < g_end; w8_idx++) {
            const block_q8_0_r8 *w_base = (const block_q8_0_r8 *)
                ((const char *)vx + (size_t)w8_idx * w_stride);
            int c = 0;
            for (; c + 3 < ncols; c += 4) {
                const block_q8_0 *qy0 = (const block_q8_0 *)((const char *)vy + (size_t)(c+0) * a_stride);
                const block_q8_0 *qy1 = (const block_q8_0 *)((const char *)vy + (size_t)(c+1) * a_stride);
                const block_q8_0 *qy2 = (const block_q8_0 *)((const char *)vy + (size_t)(c+2) * a_stride);
                const block_q8_0 *qy3 = (const block_q8_0 *)((const char *)vy + (size_t)(c+3) * a_stride);
                __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
                for (int ib = 0; ib < nb; ib++) {
                    __m256 ws = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)w_base[ib].d));
                    float ys[4];
                    if (vy_d) {
                        ys[0] = vy_d[(size_t)(c+0) * vy_d_stride + ib];
                        ys[1] = vy_d[(size_t)(c+1) * vy_d_stride + ib];
                        ys[2] = vy_d[(size_t)(c+2) * vy_d_stride + ib];
                        ys[3] = vy_d[(size_t)(c+3) * vy_d_stride + ib];
                    } else {
                        ys[0] = fp16_to_fp32_lookup(qy0[ib].d);
                        ys[1] = fp16_to_fp32_lookup(qy1[ib].d);
                        ys[2] = fp16_to_fp32_lookup(qy2[ib].d);
                        ys[3] = fp16_to_fp32_lookup(qy3[ib].d);
                    }
                    /* Compute dot products for 4 columns */
                    __m256i sumi0 = gemv_r8_block(w_base + ib, qy0 + ib);
                    __m256i sumi1 = gemv_r8_block(w_base + ib, qy1 + ib);
                    __m256i sumi2 = gemv_r8_block(w_base + ib, qy2 + ib);
                    __m256i sumi3 = gemv_r8_block(w_base + ib, qy3 + ib);
                    __m256 d0 = _mm256_mul_ps(_mm256_cvtepi32_ps(sumi0), _mm256_mul_ps(ws, _mm256_set1_ps(ys[0])));
                    __m256 d1 = _mm256_mul_ps(_mm256_cvtepi32_ps(sumi1), _mm256_mul_ps(ws, _mm256_set1_ps(ys[1])));
                    __m256 d2 = _mm256_mul_ps(_mm256_cvtepi32_ps(sumi2), _mm256_mul_ps(ws, _mm256_set1_ps(ys[2])));
                    __m256 d3 = _mm256_mul_ps(_mm256_cvtepi32_ps(sumi3), _mm256_mul_ps(ws, _mm256_set1_ps(ys[3])));
                    acc0 = _mm256_add_ps(acc0, d0);
                    acc1 = _mm256_add_ps(acc1, d1);
                    acc2 = _mm256_add_ps(acc2, d2);
                    acc3 = _mm256_add_ps(acc3, d3);
                }
                _mm256_storeu_ps(out + (size_t)(c+0) * bs + w8_idx * 8, acc0);
                _mm256_storeu_ps(out + (size_t)(c+1) * bs + w8_idx * 8, acc1);
                _mm256_storeu_ps(out + (size_t)(c+2) * bs + w8_idx * 8, acc2);
                _mm256_storeu_ps(out + (size_t)(c+3) * bs + w8_idx * 8, acc3);
            }
            /* Tail columns */
            for (; c < ncols; c++) {
                const block_q8_0 *qy = (const block_q8_0 *)
                    ((const char *)vy + (size_t)c * a_stride);
                float *out_base = out + (size_t)c * bs + w8_idx * 8;
                vec_dot_q8_0_r8_q8_0_d_avx2(w_base, qy,
                    vy_d ? vy_d + (size_t)c * vy_d_stride : NULL,
                    k, out_base, 8);
            }
        }
    }
    return (nrows / 8) * 8;
}

/* Thin wrapper: deltas derived from the fp16 block scales in the blocks. */
int sgemm_q8_0_r8_q8_0_avx2(int nrows, int ncols, int k,
                            const void *vx, const void *vy,
                            float *out, size_t bs, int ith, int nth) {
    return sgemm_q8_0_r8_q8_0_d_avx2(nrows, ncols, k, vx, vy, NULL, 0,
                                     out, bs, ith, nth);
}

#else /* !__AVX2__ */

/* Portable fallback: Q8_0_R8 weights x Q8_0 activations.
 *
 * block_q8_0_r8 is {fp16 d[8]; int8 qs[256]} = 8 rows x 32 values, stored as
 * an 8x8 transpose of 32-bit words: row r's 32 values are
 *   qs[32*l + 4*r + i] for l = 0..3, i = 0..3  ->  value 4*l + i, and
 *   qs[32*l + 4*r + i + 128]                   ->  value 4*l + i + 16.
 * (See dequantize_row_q8_0_r8_single, which this mirrors.)
 *
 * Dequantize-then-integer-dot, exactly like vec_dot_q5_0_q8_0: the point is
 * that it works on every target, not that it is fast. Without it a build
 * lacking AVX2 silently returned zeros for this type, which produced coherent-
 * looking-but-wrong text (see the fix that introduced this fallback). */
void vec_dot_q8_0_r8_q8_0_avx2(const void *vx, const void *wy, int n,
                               float *out, int nrows) {
    const block_q8_0_r8 *xb = (const block_q8_0_r8 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)wy;
    const int nb = n / 32;
    for (int row = 0; row < nrows; row++) {
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++) {
            const float d = fp16_to_fp32(xb[ib].d[row]) * fp16_to_fp32(y[ib].d);
            const int8_t *qs = xb[ib].qs;
            const int8_t *ya = y[ib].qs;
            int sumi = 0;
            for (int l = 0; l < 4; l++) {
                const int8_t *q = qs + 32 * l + 4 * row;
                for (int i = 0; i < 4; i++) {
                    sumi += (int)q[i]       * (int)ya[4 * l + i];
                    sumi += (int)q[i + 128] * (int)ya[4 * l + i + 16];
                }
            }
            sumf += (float)sumi * d;
        }
        out[row] = sumf;
    }
}

int sgemm_q8_0_r8_q8_0_avx2(int nrows, int ncols, int k,
                            const void *vx, const void *vy,
                            float *out, size_t bs, int ith, int nth) {
    (void)vx; (void)vy; (void)k; (void)out; (void)bs; (void)ith; (void)nth;
    (void)nrows; (void)ncols;
    return 0;
}

/* Portable fallback: as above but with caller-supplied fp32 activation deltas. */
void vec_dot_q8_0_r8_q8_0_d_avx2(const void *vx, const void *wy, const float *q_d,
                                 int n, float *out, int nrows) {
    const block_q8_0_r8 *xb = (const block_q8_0_r8 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)wy;
    const int nb = n / 32;
    for (int row = 0; row < nrows; row++) {
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++) {
            const float dy = q_d ? q_d[ib] : fp16_to_fp32(y[ib].d);
            const float d = fp16_to_fp32(xb[ib].d[row]) * dy;
            const int8_t *qs = xb[ib].qs;
            const int8_t *ya = y[ib].qs;
            int sumi = 0;
            for (int l = 0; l < 4; l++) {
                const int8_t *q = qs + 32 * l + 4 * row;
                for (int i = 0; i < 4; i++) {
                    sumi += (int)q[i]       * (int)ya[4 * l + i];
                    sumi += (int)q[i + 128] * (int)ya[4 * l + i + 16];
                }
            }
            sumf += (float)sumi * d;
        }
        out[row] = sumf;
    }
}

int sgemm_q8_0_r8_q8_0_d_avx2(int nrows, int ncols, int k,
                              const void *vx, const void *vy,
                              const float *vy_d, int vy_d_stride,
                              float *out, size_t bs, int ith, int nth) {
    (void)vx; (void)vy; (void)vy_d; (void)vy_d_stride; (void)k; (void)out;
    (void)bs; (void)ith; (void)nth; (void)nrows; (void)ncols;
    return 0;
}

void repack_q8_0_to_q8_0_r8(const void *src, void *dst, int nrows, int n) {
    if (!src || !dst || nrows <= 0 || n <= 0 || n % 32 != 0 || nrows % 8 != 0) return;
    const int nb = n / 32;
    const block_q8_0 *s = (const block_q8_0 *)src;
    block_q8_0_r8 *d = (block_q8_0_r8 *)dst;
    for (int g = 0; g < nrows / 8; g++) {
        const block_q8_0 *sg = s + (size_t)g * 8 * nb;
        block_q8_0_r8 *dg = d + (size_t)g * nb;
        for (int ib = 0; ib < nb; ib++) {
            for (int r = 0; r < 8; r++) dg[ib].d[r] = sg[(size_t)r * nb + ib].d;
            for (int l = 0; l < 4; l++) {
                for (int k = 0; k < 8; k++) {
                    const int8_t *qs = sg[(size_t)k * nb + ib].qs;
                    for (int i = 0; i < 4; i++) {
                        dg[ib].qs[32 * l + 4 * k + i]       = qs[4 * l + i];
                        dg[ib].qs[32 * l + 4 * k + i + 128] = qs[4 * l + i + 16];
                    }
                }
            }
        }
    }
}

#endif

#if defined(__AVX2__) && defined(__F16C__)
/* ================================================================
 * sgemm_q8_0_r8_avx2t_q8_0: panel-blocked AVX2 GEMM for Q8_0_R8 weights (no VNNI needed).
 *
 * The older AVX2 kernel above sweeps ALL activation columns for every 8-row weight group, so on
 * a CPU with a 256 KB L2 the whole activation matrix (n*k bytes, MBs) streams from L3 once per
 * group. Here a thread stages a panel of PC columns (int8 + fp32 scales, sized to fit in L2) and
 * sweeps all its weight groups against it, so activations are read from L3 once per panel.
 * Tile: 8 rows x 4 columns. abs(w) is computed once per weight vector and shared by the 4
 * columns; per (vector, column): broadcast + sign + maddubs + madd + add. One cvt/mul/fma
 * fixup per block and column. Integer sums are identical to the older kernel.
 * ================================================================ */
#if defined(__FMA__)
#define R8A_FMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define R8A_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif

static inline __attribute__((always_inline))
void r8a_tile(const block_q8_0_r8 *w, const int8_t *yq, const float *ys, int nb,
              float *out, size_t bs, const int NC) {
    __m256 facc[4];
#pragma GCC unroll 4
    for (int c = 0; c < NC; c++) facc[c] = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    for (int ib = 0; ib < nb; ib++) {
        const __m256i *wq = (const __m256i *)w[ib].qs;
        const __m256 wsc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)w[ib].d));
        __m256i isum[4];
#pragma GCC unroll 4
        for (int c = 0; c < NC; c++) isum[c] = _mm256_setzero_si256();
#pragma GCC unroll 8
        for (int j = 0; j < 8; j++) {
            const __m256i wj = _mm256_loadu_si256(wq + j);
            const __m256i aw = _mm256_sign_epi8(wj, wj);          /* abs(w), unsigned operand */
#pragma GCC unroll 4
            for (int c = 0; c < NC; c++) {
                int d; memcpy(&d, yq + ((size_t)c * nb + ib) * 32 + 4 * j, 4);
                const __m256i sy = _mm256_sign_epi8(_mm256_set1_epi32(d), wj);
                isum[c] = _mm256_add_epi32(isum[c], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sy), ones));
            }
        }
#pragma GCC unroll 4
        for (int c = 0; c < NC; c++) {
            const __m256 sc = _mm256_mul_ps(wsc, _mm256_set1_ps(ys[(size_t)c * nb + ib]));
            facc[c] = R8A_FMA(_mm256_cvtepi32_ps(isum[c]), sc, facc[c]);
        }
    }
#pragma GCC unroll 4
    for (int c = 0; c < NC; c++) _mm256_storeu_ps(out + (size_t)c * bs, facc[c]);
}

typedef struct { int8_t *yq; float *ys; block_q8_0_r8 *wr; size_t yq_cap, ys_cap, wr_cap; } r8a_scratch_t;
static __thread r8a_scratch_t r8a_tl;

static int r8a_reserve(int pc, int nb, int need_wr) {
    const size_t yq = (size_t)pc * nb * 32, ys = (size_t)pc * nb * sizeof(float), wr = need_wr ? (size_t)nb * sizeof(block_q8_0_r8) : 0;
    if (r8a_tl.yq_cap < yq) { _mm_free(r8a_tl.yq); r8a_tl.yq = (int8_t *)_mm_malloc((yq + 63) & ~(size_t)63, 64); r8a_tl.yq_cap = r8a_tl.yq ? yq : 0; }
    if (r8a_tl.ys_cap < ys) { _mm_free(r8a_tl.ys); r8a_tl.ys = (float *)_mm_malloc((ys + 63) & ~(size_t)63, 64); r8a_tl.ys_cap = r8a_tl.ys ? ys : 0; }
    if (r8a_tl.wr_cap < wr) { _mm_free(r8a_tl.wr); r8a_tl.wr = (block_q8_0_r8 *)_mm_malloc((wr + 63) & ~(size_t)63, 64); r8a_tl.wr_cap = r8a_tl.wr ? wr : 0; }
    return r8a_tl.yq && r8a_tl.ys && (!need_wr || r8a_tl.wr);
}

/* Panel width: ~R8A_PANEL_BYTES of staged activations (int8 + fp32 scale per block), multiple of 4.
 * Default 160 KB suits a 256 KB L2; PICOLM_Q8_AVX2_PANEL_KB overrides. */
static int r8a_panel_cols(int nb) {
    static int kb = 0;
    if (!kb) { const char *e = getenv("PICOLM_Q8_AVX2_PANEL_KB"); kb = (e && atoi(e) > 0) ? atoi(e) : 160; }
    int pc = (int)(((size_t)kb * 1024) / ((size_t)nb * 36)) & ~3;
    return pc < 4 ? 4 : pc;
}

static void r8a_stage_panel(const void *vy, size_t ldb_blocks, int c0, int pc, int nb) {
    for (int c = 0; c < pc; c++) {
        const block_q8_0 *qy = (const block_q8_0 *)vy + (size_t)(c0 + c) * ldb_blocks;
        int8_t *dst = r8a_tl.yq + (size_t)c * nb * 32;
        float *sd = r8a_tl.ys + (size_t)c * nb;
        for (int ib = 0; ib < nb; ib++) {
            _mm256_storeu_si256((__m256i *)(dst + (size_t)ib * 32), _mm256_loadu_si256((const __m256i *)qy[ib].qs));
            sd[ib] = _cvtsh_ss(qy[ib].d);
        }
    }
}

static inline void r8a_tiles(const block_q8_0_r8 *wg, int pc, int nb, float *og, size_t bs) {
    int cc = 0;
#define R8A_RUN(NCV) r8a_tile(wg, r8a_tl.yq + (size_t)cc * nb * 32, r8a_tl.ys + (size_t)cc * nb, nb, og + (size_t)cc * bs, bs, NCV)
    for (; cc + 4 <= pc; cc += 4) R8A_RUN(4);
    if (cc + 2 <= pc) { R8A_RUN(2); cc += 2; }
    if (cc + 1 <= pc) { R8A_RUN(1); cc += 1; }
#undef R8A_RUN
}

int sgemm_q8_0_r8_avx2t_q8_0(int nrows, int ncols, int k, const void *vx, const void *vy,
                             float *out, size_t bs, int ith, int nth) {
    if (nrows < 8 || ncols < 1 || k % 32 != 0 || nrows % 8 != 0) return 0;
    const int nb = k / 32;
    const size_t w_stride = (size_t)nb * sizeof(block_q8_0_r8);
    const int total_groups = nrows / 8, gpt = (total_groups + nth - 1) / nth;
    const int g_start = ith * gpt, g_end = (ith + 1) * gpt < total_groups ? (ith + 1) * gpt : total_groups;
    if (g_start >= g_end) return (nrows / 8) * 8;
    const int PC = r8a_panel_cols(nb);
    if (!r8a_reserve(ncols < PC ? ncols : PC, nb, 0)) return 0;
    for (int c0 = 0; c0 < ncols; c0 += PC) {
        const int pc = (ncols - c0 < PC) ? (ncols - c0) : PC;
        r8a_stage_panel(vy, (size_t)nb, c0, pc, nb);
        for (int g = g_start; g < g_end; g++)
            r8a_tiles((const block_q8_0_r8 *)((const char *)vx + (size_t)g * w_stride), pc, nb,
                      out + (size_t)c0 * bs + (size_t)g * 8, bs);
    }
    return (nrows / 8) * 8;
}

/* Row loaders for the repack: one weight block -> its 32 int8 values in natural order.
 *   Q8_0 : the stored int8s.            Q4_0  : nibble - 8.
 *   IQ4_NL: kvalues_iq4nl[nibble] (VPSHUFB LUT; the table is int8 so the result is exact).
 * Q4_0 / IQ4_NL blocks are 18 bytes (fp16 d + 16 packed bytes: values 0..15 = low nibbles, 16..31 =
 * high nibbles), so after decoding the int8 data is identical to a Q8_0 block's and everything
 * downstream (R8 transpose, tiled VNNI/AVX2 kernels, activation staging) is shared. */
enum { R8A_T_Q8_0 = 0, R8A_T_Q4_0 = 1, R8A_T_IQ4_NL = 2 };
static inline __attribute__((always_inline)) __m256i r8a_load_row(const void *blk, const int T) {
    if (T == R8A_T_Q8_0) return _mm256_loadu_si256((const __m256i *)((const char *)blk + 2));
    const __m128i v = _mm_loadu_si128((const __m128i *)((const char *)blk + 2));
    const __m128i m = _mm_set1_epi8(15);
    const __m128i lo = _mm_and_si128(v, m), hi = _mm_and_si128(_mm_srli_epi16(v, 4), m);
    const __m256i r = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    if (T == R8A_T_Q4_0) return _mm256_sub_epi8(r, _mm256_set1_epi8(8));
    return _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)kvalues_iq4nl)), r);
}

/* Repack one group of 8 weight rows (Q8_0, Q4_0 or IQ4_NL) into R8 blocks. An R8 block is
 * byte-for-byte the size of 8 Q8_0 blocks (272B), so this needs only an L2-sized scratch (nb*272B)
 * and no model-format change. The qs part is an 8x8 transpose of 32-bit words: R8 vector j = dword
 * j of each row. */
static inline __attribute__((always_inline))
void r8a_repack_impl(const char *A, size_t lda, int g, int nb, block_q8_0_r8 *dst, const int T) {
    const size_t bsz = (T == R8A_T_Q8_0) ? sizeof(block_q8_0) : sizeof(block_q4_0);
    const char *row[8];
    for (int r = 0; r < 8; r++) row[r] = A + (size_t)(g * 8 + r) * lda * bsz;
    for (int ib = 0; ib < nb; ib++) {
#define RB(r) (row[r] + (size_t)ib * bsz)
        __m256i r0 = r8a_load_row(RB(0), T), r1 = r8a_load_row(RB(1), T);
        __m256i r2 = r8a_load_row(RB(2), T), r3 = r8a_load_row(RB(3), T);
        __m256i r4 = r8a_load_row(RB(4), T), r5 = r8a_load_row(RB(5), T);
        __m256i r6 = r8a_load_row(RB(6), T), r7 = r8a_load_row(RB(7), T);
        __m256i t0 = _mm256_unpacklo_epi32(r0, r1), t1 = _mm256_unpackhi_epi32(r0, r1);
        __m256i t2 = _mm256_unpacklo_epi32(r2, r3), t3 = _mm256_unpackhi_epi32(r2, r3);
        __m256i t4 = _mm256_unpacklo_epi32(r4, r5), t5 = _mm256_unpackhi_epi32(r4, r5);
        __m256i t6 = _mm256_unpacklo_epi32(r6, r7), t7 = _mm256_unpackhi_epi32(r6, r7);
        __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
        __m256i u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
        __m256i u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6);
        __m256i u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
        __m256i *q = (__m256i *)dst[ib].qs;
        _mm256_storeu_si256(q + 0, _mm256_permute2x128_si256(u0, u4, 0x20));
        _mm256_storeu_si256(q + 1, _mm256_permute2x128_si256(u1, u5, 0x20));
        _mm256_storeu_si256(q + 2, _mm256_permute2x128_si256(u2, u6, 0x20));
        _mm256_storeu_si256(q + 3, _mm256_permute2x128_si256(u3, u7, 0x20));
        _mm256_storeu_si256(q + 4, _mm256_permute2x128_si256(u0, u4, 0x31));
        _mm256_storeu_si256(q + 5, _mm256_permute2x128_si256(u1, u5, 0x31));
        _mm256_storeu_si256(q + 6, _mm256_permute2x128_si256(u2, u6, 0x31));
        _mm256_storeu_si256(q + 7, _mm256_permute2x128_si256(u3, u7, 0x31));
        for (int r = 0; r < 8; r++) memcpy(&dst[ib].d[r], RB(r), sizeof(uint16_t));
#undef RB
    }
}

/* IQ4_NL_R4 source: one 72-byte block = 4 interleaved rows of {fp16 d, qs[64]}.
 *
 * Row r's 32 nibble values come from FOUR 4-byte windows spread over the whole
 * 64-byte qs[]. That is the Q4_0_4_4 "blocklen=4" interleave (see
 * dequantize_row_iq4_nl_r4_single in quant.c), with the non-linear IQ4_NL LUT
 * instead of the signed-nibble subtraction:
 *
 *     qs[4*r + i +  0] -> low  nibble = value[i +  0], high nibble = value[i +  8]
 *     qs[4*r + i + 16] -> low  nibble = value[i + 16], high nibble = value[i + 24]
 *     qs[4*r + i + 32] -> low  nibble = value[i +  4], high nibble = value[i + 12]
 *     qs[4*r + i + 48] -> low  nibble = value[i + 20], high nibble = value[i + 28]
 *
 * (An earlier version read qs[16*r .. 16*r+15] as if each row were 16 contiguous
 * bytes, which is the Q4_0_4_4 *byte* layout rather than this one: it silently
 * produced 32 values built from only a quarter of the block, so every prefill
 * GEMM that went through the R8 path was wrong.)
 *
 * Decode each row to its 32 int8 values in NATURAL order, then R8-transpose the
 * 8 rows (two R4 groups of 4) exactly like r8a_repack_impl(). */
static void r8a_repack_r4_iq4nl(const block_iq4_nl_r4 *A, int g, int nb, block_q8_0_r8 *dst) {
    const block_iq4_nl_r4 *pa = A + (size_t)(2 * g) * nb, *pb = A + (size_t)(2 * g + 1) * nb;
    for (int ib = 0; ib < nb; ib++) {
        /* Decode 8 rows to 8 x 32-byte int8 vectors, then R8-transpose. */
        __m256i r[8];
        for (int rr = 0; rr < 8; ++rr) {
            const block_iq4_nl_r4 *b = (rr < 4) ? &pa[ib] : &pb[ib];
            const int row = rr & 3;
            const uint8_t *qs = b->qs;
            int8_t v[32];
            for (int i = 0; i < 4; ++i) {
                const uint8_t b0 = qs[4 * row + i +  0];
                const uint8_t b1 = qs[4 * row + i + 16];
                const uint8_t b2 = qs[4 * row + i + 32];
                const uint8_t b3 = qs[4 * row + i + 48];
                v[i +  0] = kvalues_iq4nl[b0 & 0xf];
                v[i +  8] = kvalues_iq4nl[b0 >> 4];
                v[i + 16] = kvalues_iq4nl[b1 & 0xf];
                v[i + 24] = kvalues_iq4nl[b1 >> 4];
                v[i +  4] = kvalues_iq4nl[b2 & 0xf];
                v[i + 12] = kvalues_iq4nl[b2 >> 4];
                v[i + 20] = kvalues_iq4nl[b3 & 0xf];
                v[i + 28] = kvalues_iq4nl[b3 >> 4];
            }
            r[rr] = _mm256_loadu_si256((const __m256i *)v);
        }
        __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]), t1 = _mm256_unpackhi_epi32(r[0], r[1]);
        __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]), t3 = _mm256_unpackhi_epi32(r[2], r[3]);
        __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]), t5 = _mm256_unpackhi_epi32(r[4], r[5]);
        __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]), t7 = _mm256_unpackhi_epi32(r[6], r[7]);
        __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
        __m256i u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
        __m256i u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6);
        __m256i u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
        __m256i *q = (__m256i *)dst[ib].qs;
        _mm256_storeu_si256(q + 0, _mm256_permute2x128_si256(u0, u4, 0x20));
        _mm256_storeu_si256(q + 1, _mm256_permute2x128_si256(u1, u5, 0x20));
        _mm256_storeu_si256(q + 2, _mm256_permute2x128_si256(u2, u6, 0x20));
        _mm256_storeu_si256(q + 3, _mm256_permute2x128_si256(u3, u7, 0x20));
        _mm256_storeu_si256(q + 4, _mm256_permute2x128_si256(u0, u4, 0x31));
        _mm256_storeu_si256(q + 5, _mm256_permute2x128_si256(u1, u5, 0x31));
        _mm256_storeu_si256(q + 6, _mm256_permute2x128_si256(u2, u6, 0x31));
        _mm256_storeu_si256(q + 7, _mm256_permute2x128_si256(u3, u7, 0x31));
        memcpy(&dst[ib].d[0], pa[ib].d, 4 * sizeof(uint16_t));
        memcpy(&dst[ib].d[4], pb[ib].d, 4 * sizeof(uint16_t));
    }
}

static void r8a_repack_group(int atype, const void *A, size_t lda, int g, int nb, block_q8_0_r8 *dst) {
    switch (atype) {
    case GGUF_TYPE_Q4_0:   r8a_repack_impl((const char *)A, lda, g, nb, dst, R8A_T_Q4_0); break;
    case GGUF_TYPE_IQ4_NL: r8a_repack_impl((const char *)A, lda, g, nb, dst, R8A_T_IQ4_NL); break;
    case GGUF_TYPE_IQ4_NL_R4: (void)lda; r8a_repack_r4_iq4nl((const block_iq4_nl_r4 *)A, g, nb, dst); break;
    default:               r8a_repack_impl((const char *)A, lda, g, nb, dst, R8A_T_Q8_0); break;
    }
}

/* Plain Q8_0 weights, AVX2 (no VNNI): repack each 8-row group once per panel, then the same tile. */
int picolm_sgemm_r8dec_avx2t(int atype, int m, int n, int k_blocks, const void *vA, int lda,
                             const void *vB, int ldb, float *C, int ldc, int ith, int nth) {
    if (atype != GGUF_TYPE_Q8_0 && atype != GGUF_TYPE_Q4_0 && atype != GGUF_TYPE_IQ4_NL && atype != GGUF_TYPE_IQ4_NL_R4) return 0;
    if (m < 8 || (m % 8) != 0 || n < 1 || k_blocks < 1 || nth < 1) return 0;
    const int nb = k_blocks, total_groups = m / 8, gpt = (total_groups + nth - 1) / nth;
    const int g_start = ith * gpt, g_end = (ith + 1) * gpt < total_groups ? (ith + 1) * gpt : total_groups;
    if (g_start >= g_end) return 1;
    const int PC = r8a_panel_cols(nb);
    if (!r8a_reserve(n < PC ? n : PC, nb, 1)) return 0;
    for (int c0 = 0; c0 < n; c0 += PC) {
        const int pc = (n - c0 < PC) ? (n - c0) : PC;
        r8a_stage_panel(vB, (size_t)ldb, c0, pc, nb);
        for (int g = g_start; g < g_end; g++) {
            r8a_repack_group(atype, vA, (size_t)lda, g, nb, r8a_tl.wr);
            r8a_tiles(r8a_tl.wr, pc, nb, C + (size_t)c0 * ldc + (size_t)g * 8, (size_t)ldc);
        }
    }
    return 1;
}

int picolm_sgemm_q8_0_avx2t(int m, int n, int k_blocks, const void *vA, int lda,
                            const void *vB, int ldb, float *C, int ldc, int ith, int nth) {
    return picolm_sgemm_r8dec_avx2t(GGUF_TYPE_Q8_0, m, n, k_blocks, vA, lda, vB, ldb, C, ldc, ith, nth);
}
#else
int sgemm_q8_0_r8_avx2t_q8_0(int nrows, int ncols, int k, const void *vx, const void *vy,
                             float *out, size_t bs, int ith, int nth) {
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth; return 0;
}
int picolm_sgemm_r8dec_avx2t(int atype, int m, int n, int k_blocks, const void *vA, int lda,
                             const void *vB, int ldb, float *C, int ldc, int ith, int nth) {
    (void)atype; (void)m; (void)n; (void)k_blocks; (void)vA; (void)lda; (void)vB; (void)ldb; (void)C; (void)ldc; (void)ith; (void)nth; return 0;
}
int picolm_sgemm_q8_0_avx2t(int m, int n, int k_blocks, const void *vA, int lda,
                            const void *vB, int ldb, float *C, int ldc, int ith, int nth) {
    (void)m; (void)n; (void)k_blocks; (void)vA; (void)lda; (void)vB; (void)ldb; (void)C; (void)ldc; (void)ith; (void)nth; return 0;
}
#endif

/* ================================================================
 * sgemm_q8_0_r8_vnni_q8_0: AVX-512VL VNNI (256-bit dpbusd) GEMM for Q8_0_R8 weights.
 *
 * Why it is faster than the AVX2 kernel above (and than the 4x2-row tile in
 * picolm_sgemm_d, which is bound by per-block scale fixups, not by the dot):
 *  - R8 layout: one dpbusd per (weight vector, column) yields 8 row results, so no
 *    horizontal reductions; the fp16->fp32 scale conversion and the int->float fixup
 *    are paid once per 8 dpbusd instead of once per 1.
 *  - dpbusd wants (unsigned, signed): activations are shifted to u8 (q ^ 0x80 = q+128)
 *    once per column panel, and the +128*sum(w) bias is removed with a per-block row
 *    sum of the weights (8 extra dpbusd per block, shared by all NC columns of the tile).
 *    No sign_epi8 per (vector, column).
 *  - 8 rows x NC=12 columns register tile (32 ymm registers via AVX-512VL): the 8 weight
 *    vectors of a block stay in registers while 12 columns stream through.
 *  - Column panels (R8V_PC columns of u8 activations) are staged in per-thread scratch
 *    so they stay in L2 while a thread sweeps its weight groups.
 * Contract is identical to sgemm_q8_0_r8_q8_0_avx2: nrows%8==0, k%32==0, activations are
 * plain block_q8_0 columns, out[c*bs + r], threads split the weight groups (ith/nth).
 * ================================================================ */
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX2__) && defined(__F16C__)

#define R8V_PC 96      /* columns per panel: 96*k bytes of u8 activations (384KB at k=4096) */

static inline __attribute__((always_inline))
void r8v_tile(const block_q8_0_r8 *w, const uint8_t *yu, const float *ys, int nb,
              float *out, size_t bs, const int NC) {
    __m256 facc[12];
#pragma GCC unroll 12
    for (int c = 0; c < NC; c++) facc[c] = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi8(1);
    for (int ib = 0; ib < nb; ib++) {
        const __m256i *wq = (const __m256i *)w[ib].qs;
        const __m256i w0 = _mm256_loadu_si256(wq + 0), w1 = _mm256_loadu_si256(wq + 1);
        const __m256i w2 = _mm256_loadu_si256(wq + 2), w3 = _mm256_loadu_si256(wq + 3);
        const __m256i w4 = _mm256_loadu_si256(wq + 4), w5 = _mm256_loadu_si256(wq + 5);
        const __m256i w6 = _mm256_loadu_si256(wq + 6), w7 = _mm256_loadu_si256(wq + 7);
        const __m256 wsc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)w[ib].d));
        /* bias removal: 128 * (sum of the block's 32 weights) per row, two chains */
        __m256i sa = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones, w0);
        __m256i sb = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones, w1);
        sa = _mm256_dpbusd_epi32(sa, ones, w2); sb = _mm256_dpbusd_epi32(sb, ones, w3);
        sa = _mm256_dpbusd_epi32(sa, ones, w4); sb = _mm256_dpbusd_epi32(sb, ones, w5);
        sa = _mm256_dpbusd_epi32(sa, ones, w6); sb = _mm256_dpbusd_epi32(sb, ones, w7);
        const __m256i corr = _mm256_slli_epi32(_mm256_add_epi32(sa, sb), 7);
#pragma GCC unroll 12
        for (int c = 0; c < NC; c++) {
            const uint8_t *yb = yu + ((size_t)c * nb + ib) * 32;
            int d0, d1, d2, d3, d4, d5, d6, d7;
            memcpy(&d0, yb, 4);      memcpy(&d1, yb + 4, 4);  memcpy(&d2, yb + 8, 4);  memcpy(&d3, yb + 12, 4);
            memcpy(&d4, yb + 16, 4); memcpy(&d5, yb + 20, 4); memcpy(&d6, yb + 24, 4); memcpy(&d7, yb + 28, 4);
            __m256i a = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_set1_epi32(d0), w0);
            __m256i b = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_set1_epi32(d1), w1);
            a = _mm256_dpbusd_epi32(a, _mm256_set1_epi32(d2), w2); b = _mm256_dpbusd_epi32(b, _mm256_set1_epi32(d3), w3);
            a = _mm256_dpbusd_epi32(a, _mm256_set1_epi32(d4), w4); b = _mm256_dpbusd_epi32(b, _mm256_set1_epi32(d5), w5);
            a = _mm256_dpbusd_epi32(a, _mm256_set1_epi32(d6), w6); b = _mm256_dpbusd_epi32(b, _mm256_set1_epi32(d7), w7);
            a = _mm256_sub_epi32(_mm256_add_epi32(a, b), corr);
            const __m256 sc = _mm256_mul_ps(wsc, _mm256_set1_ps(ys[(size_t)c * nb + ib]));
            facc[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a), sc, facc[c]);
        }
    }
#pragma GCC unroll 12
    for (int c = 0; c < NC; c++) _mm256_storeu_ps(out + (size_t)c * bs, facc[c]);
}

/* per-thread scratch: u8 activation panel, fp32 scales, repacked weight group */
typedef struct { uint8_t *yu; float *ys; block_q8_0_r8 *wr; size_t yu_cap, ys_cap, wr_cap; } r8v_scratch_t;
static __thread r8v_scratch_t r8v_tl;

static int r8v_reserve(int pc, int nb, int need_wr) {
    const size_t yu = (size_t)pc * nb * 32, ys = (size_t)pc * nb * sizeof(float), wr = need_wr ? (size_t)nb * sizeof(block_q8_0_r8) : 0;
    if (r8v_tl.yu_cap < yu) { _mm_free(r8v_tl.yu); r8v_tl.yu = (uint8_t *)_mm_malloc((yu + 63) & ~(size_t)63, 64); r8v_tl.yu_cap = r8v_tl.yu ? yu : 0; }
    if (r8v_tl.ys_cap < ys) { _mm_free(r8v_tl.ys); r8v_tl.ys = (float *)_mm_malloc((ys + 63) & ~(size_t)63, 64); r8v_tl.ys_cap = r8v_tl.ys ? ys : 0; }
    if (r8v_tl.wr_cap < wr) { _mm_free(r8v_tl.wr); r8v_tl.wr = (block_q8_0_r8 *)_mm_malloc((wr + 63) & ~(size_t)63, 64); r8v_tl.wr_cap = r8v_tl.wr ? wr : 0; }
    return r8v_tl.yu && r8v_tl.ys && (!need_wr || r8v_tl.wr);
}

/* Stage columns [c0, c0+pc) of the activations as u8 (q ^ 0x80) + fp32 scales. */
static void r8v_stage_panel(const void *vy, size_t ldb_blocks, int c0, int pc, int nb) {
    const __m256i x80 = _mm256_set1_epi8((char)0x80);
    for (int c = 0; c < pc; c++) {
        const block_q8_0 *qy = (const block_q8_0 *)vy + (size_t)(c0 + c) * ldb_blocks;
        uint8_t *dst = r8v_tl.yu + (size_t)c * nb * 32;
        float *sd = r8v_tl.ys + (size_t)c * nb;
        for (int ib = 0; ib < nb; ib++) {
            _mm256_storeu_si256((__m256i *)(dst + (size_t)ib * 32),
                _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)qy[ib].qs), x80));
            sd[ib] = _cvtsh_ss(qy[ib].d);
        }
    }
}

/* All column tiles of the staged panel against one 8-row group. */
static inline void r8v_tiles(const block_q8_0_r8 *wg, int pc, int nb, float *og, size_t bs) {
    int cc = 0;
#define R8V_RUN(NCV) r8v_tile(wg, r8v_tl.yu + (size_t)cc * nb * 32, r8v_tl.ys + (size_t)cc * nb, nb, og + (size_t)cc * bs, bs, NCV)
    for (; cc + 12 <= pc; cc += 12) R8V_RUN(12);
    if (cc + 8 <= pc) { R8V_RUN(8); cc += 8; }
    if (cc + 4 <= pc) { R8V_RUN(4); cc += 4; }
    if (cc + 2 <= pc) { R8V_RUN(2); cc += 2; }
    if (cc + 1 <= pc) { R8V_RUN(1); cc += 1; }
#undef R8V_RUN
}

int sgemm_q8_0_r8_vnni_q8_0(int nrows, int ncols, int k,
                            const void *vx, const void *vy,
                            float *out, size_t bs, int ith, int nth) {
    if (nrows < 8 || ncols < 1 || k % 32 != 0 || nrows % 8 != 0) return 0;
    const int nb = k / 32;
    const size_t w_stride = (size_t)nb * sizeof(block_q8_0_r8);
    const int total_groups = nrows / 8;
    const int gpt = (total_groups + nth - 1) / nth;
    const int g_start = ith * gpt;
    const int g_end = (ith + 1) * gpt < total_groups ? (ith + 1) * gpt : total_groups;
    if (g_start >= g_end) return (nrows / 8) * 8;
    if (!r8v_reserve(ncols < R8V_PC ? ncols : R8V_PC, nb, 0)) return 0;
    for (int c0 = 0; c0 < ncols; c0 += R8V_PC) {
        const int pc = (ncols - c0 < R8V_PC) ? (ncols - c0) : R8V_PC;
        r8v_stage_panel(vy, (size_t)nb, c0, pc, nb);
        for (int g = g_start; g < g_end; g++)
            r8v_tiles((const block_q8_0_r8 *)((const char *)vx + (size_t)g * w_stride), pc, nb,
                      out + (size_t)c0 * bs + (size_t)g * 8, bs);
    }
    return (nrows / 8) * 8;
}

/* Plain Q8_0 weights (rows of lda blocks) x Q8_0 activations (columns of ldb blocks),
 * same output convention as picolm_sgemm_d: C[c*ldc + r]. Returns 0 if unsupported. */
int picolm_sgemm_r8dec_vnni(int atype, int m, int n, int k_blocks, const void *vA, int lda,
                            const void *vB, int ldb, float *C, int ldc, int ith, int nth) {
    if (atype != GGUF_TYPE_Q8_0 && atype != GGUF_TYPE_Q4_0 && atype != GGUF_TYPE_IQ4_NL && atype != GGUF_TYPE_IQ4_NL_R4) return 0;
    if (m < 8 || (m % 8) != 0 || n < 1 || k_blocks < 1 || nth < 1) return 0;
    const int nb = k_blocks, total_groups = m / 8;
    const int gpt = (total_groups + nth - 1) / nth;
    const int g_start = ith * gpt;
    const int g_end = (ith + 1) * gpt < total_groups ? (ith + 1) * gpt : total_groups;
    if (g_start >= g_end) return 1;
    if (!r8v_reserve(n < R8V_PC ? n : R8V_PC, nb, 1)) return 0;
    for (int c0 = 0; c0 < n; c0 += R8V_PC) {
        const int pc = (n - c0 < R8V_PC) ? (n - c0) : R8V_PC;
        r8v_stage_panel(vB, (size_t)ldb, c0, pc, nb);
        for (int g = g_start; g < g_end; g++) {
            r8a_repack_group(atype, vA, (size_t)lda, g, nb, r8v_tl.wr);
            r8v_tiles(r8v_tl.wr, pc, nb, C + (size_t)c0 * ldc + (size_t)g * 8, (size_t)ldc);
        }
    }
    return 1;
}

int picolm_sgemm_q8_0_vnni(int m, int n, int k_blocks, const void *vA, int lda,
                           const void *vB, int ldb, float *C, int ldc, int ith, int nth) {
    return picolm_sgemm_r8dec_vnni(GGUF_TYPE_Q8_0, m, n, k_blocks, vA, lda, vB, ldb, C, ldc, ith, nth);
}

#else /* no AVX-512 VNNI+VL */

int sgemm_q8_0_r8_vnni_q8_0(int nrows, int ncols, int k, const void *vx, const void *vy,
                            float *out, size_t bs, int ith, int nth) {
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy; (void)out; (void)bs; (void)ith; (void)nth;
    return 0;   /* caller falls back to sgemm_q8_0_r8_q8_0_avx2 */
}

int picolm_sgemm_r8dec_vnni(int atype, int m, int n, int k_blocks, const void *vA, int lda,
                            const void *vB, int ldb, float *C, int ldc, int ith, int nth) {
    (void)atype; (void)m; (void)n; (void)k_blocks; (void)vA; (void)lda; (void)vB; (void)ldb; (void)C; (void)ldc; (void)ith; (void)nth; return 0;
}
int picolm_sgemm_q8_0_vnni(int m, int n, int k_blocks, const void *vA, int lda,
                           const void *vB, int ldb, float *C, int ldc, int ith, int nth) {
    (void)m; (void)n; (void)k_blocks; (void)vA; (void)lda; (void)vB; (void)ldb; (void)C; (void)ldc; (void)ith; (void)nth;
    return 0;
}

#endif

/* fn_gemm for Q8_0_R8 in the type table: the VNNI kernel when it is compiled in (no repack
 * needed, the weights are already R8 on disk) and the batch is large enough to pay for the
 * activation panel staging, otherwise the AVX2 kernel. Always produces the full result:
 * gemm_table_task() ignores the return value. PICOLM_Q8_VNNI=0 forces the AVX2 kernel;
 * PICOLM_Q8_VNNI_MIN_N sets the smallest batch (default 8). */
int sgemm_q8_0_r8_q8_0_best(int nrows, int ncols, int k, const void *vx, const void *vy,
                            float *out, size_t bs, int ith, int nth) {
    static int en = -1, min_n = 8;
    if (en < 0) {
        const char *e = getenv("PICOLM_Q8_VNNI"), *mn = getenv("PICOLM_Q8_VNNI_MIN_N");
        if (mn && atoi(mn) > 0) min_n = atoi(mn);
        en = !(e && e[0] == '0');
    }
    if (en && ncols >= min_n) {
        int r = sgemm_q8_0_r8_vnni_q8_0(nrows, ncols, k, vx, vy, out, bs, ith, nth);
        if (r) {
            static int traced; if (!traced && getenv("PICOLM_DISPATCH")) { traced = 1; fprintf(stderr, "TRACE kernel: sgemm_q8_0_r8_vnni_q8_0 (Q8_0_R8)\n"); }
            return r;
        }
    }
    {
        static int en2 = -1;
        if (en2 < 0) { const char *e = getenv("PICOLM_Q8_AVX2T"); en2 = !(e && e[0] == '0'); }
        if (en2 && ncols >= 8) {
            int r = sgemm_q8_0_r8_avx2t_q8_0(nrows, ncols, k, vx, vy, out, bs, ith, nth);
            if (r) {
                static int traced; if (!traced && getenv("PICOLM_DISPATCH")) { traced = 1; fprintf(stderr, "TRACE kernel: sgemm_q8_0_r8_avx2t_q8_0 (Q8_0_R8, panel-blocked AVX2)\n"); }
                return r;
            }
        }
    }
    return sgemm_q8_0_r8_q8_0_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}

/* ================================================================
 * sgemm_iq4_nl_r4_q8_0_best: table fn_gemm for IQ4_NL_R4. Same decode-once structure as plain
 * IQ4_NL: each 8-row group (two R4 groups) is expanded once per column panel to an int8 R8 scratch
 * and fed to the tiled VNNI (AVX-512 VNNI+VL builds) or AVX2 kernel. Falls back to the original
 * 1x4-tile kernel for small batches, row counts not divisible by 8, or builds without AVX2.
 * PICOLM_Q4_R8=0 disables; PICOLM_Q4_R8_MIN_N (default 16) sets the smallest batch.
 * ================================================================ */
int sgemm_iq4_nl_r4_q8_0_best(int nrows, int ncols, int k, const void *vx, const void *vy,
                              float *out, size_t bs, int ith, int nth) {
    static int en = -1, min_n = 16, vnni_en = 1, avx2_en = 1;
    if (en < 0) {
        const char *e = getenv("PICOLM_Q4_R8"), *mn = getenv("PICOLM_Q4_R8_MIN_N");
        const char *ve = getenv("PICOLM_Q8_VNNI"), *ae = getenv("PICOLM_Q8_AVX2T");
        if (mn && atoi(mn) > 0) min_n = atoi(mn);
        if (ve && ve[0] == '0') vnni_en = 0;
        if (ae && ae[0] == '0') avx2_en = 0;
        en = !(e && e[0] == '0');
    }
    if (en && ncols >= min_n && k % 32 == 0 && nrows % 8 == 0 && picolm_r8_big_enough(nrows, k / 32)) {
        const int nb = k / 32;
        if (vnni_en && picolm_sgemm_r8dec_vnni(GGUF_TYPE_IQ4_NL_R4, nrows, ncols, nb, vx, nb, vy, nb, out, (int)bs, ith, nth)) {
            static int tr; if (!tr && getenv("PICOLM_DISPATCH")) { tr = 1; fprintf(stderr, "TRACE kernel: picolm_sgemm_r8dec_vnni (IQ4_NL_R4)\n"); }
            return 1;
        }
        if (avx2_en && picolm_sgemm_r8dec_avx2t(GGUF_TYPE_IQ4_NL_R4, nrows, ncols, nb, vx, nb, vy, nb, out, (int)bs, ith, nth)) {
            static int tr; if (!tr && getenv("PICOLM_DISPATCH")) { tr = 1; fprintf(stderr, "TRACE kernel: picolm_sgemm_r8dec_avx2t (IQ4_NL_R4)\n"); }
            return 1;
        }
    }
    return sgemm_iq4_nl_r4_q8_0_avx2(nrows, ncols, k, vx, vy, out, bs, ith, nth);
}
