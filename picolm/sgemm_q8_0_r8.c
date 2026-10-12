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
    for (int l = 0; l < 4; l++) {
        const __m256i wv = _mm256_loadu_si256(wq + l);
        const __m256i yb = _mm256_shuffle_epi32(yl, _MM_SHUFFLE(l,l,l,l));
        const __m256i aw = _mm256_sign_epi8(wv, wv);
        const __m256i sy = _mm256_sign_epi8(yb, wv);
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(m1, _mm256_maddubs_epi16(aw, sy)));
    }
    for (int l = 0; l < 4; l++) {
        const __m256i wv = _mm256_loadu_si256(wq + 4 + l);
        const __m256i yb = _mm256_shuffle_epi32(yh, _MM_SHUFFLE(l,l,l,l));
        const __m256i aw = _mm256_sign_epi8(wv, wv);
        const __m256i sy = _mm256_sign_epi8(yb, wv);
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(m1, _mm256_maddubs_epi16(aw, sy)));
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
 * sgemm_q8_0_r8_q8_0_avx2: GEMM, 8 weight rows at a time
 * nrows: total weight rows (d), multiple of 8
 * ncols: activation columns (batch size)
 * k: shared dimension in values (multiple of 32)
 * vx: block_q8_0_r8 weights, group stride = (k/32) * sizeof(block_q8_0_r8)
 * vy: block_q8_0 activations, column stride = (k/32) * sizeof(block_q8_0)
 * out: float output [ncols][nrows] (out[c * bs + r])
 * bs: output row stride (typically d)
 * ith/nth: thread splitting over ncols
 * ================================================================ */
int sgemm_q8_0_r8_q8_0_avx2(int nrows, int ncols, int k,
                            const void *vx, const void *vy,
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
                    ys[0] = fp16_to_fp32_lookup(qy0[ib].d);
                    ys[1] = fp16_to_fp32_lookup(qy1[ib].d);
                    ys[2] = fp16_to_fp32_lookup(qy2[ib].d);
                    ys[3] = fp16_to_fp32_lookup(qy3[ib].d);
                    __m256 ysc = _mm256_set_ps(ys[3], ys[2], ys[1], ys[0], ys[3], ys[2], ys[1], ys[0]);
                    (void)ysc;
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
                vec_dot_q8_0_r8_q8_0_avx2(w_base, qy, k, out_base, 8);
            }
        }
    }
    return (nrows / 8) * 8;
}

#else /* !__AVX2__ */

void vec_dot_q8_0_r8_q8_0_avx2(const void *vx, const void *wy, int n,
                               float *out, int nrows) {
    (void)vx; (void)wy; (void)n; (void)out; (void)nrows;
    memset(out, 0, nrows * sizeof(float));
}

int sgemm_q8_0_r8_q8_0_avx2(int nrows, int ncols, int k,
                            const void *vx, const void *vy,
                            float *out, size_t bs, int ith, int nth) {
    (void)vx; (void)vy; (void)k; (void)out; (void)bs; (void)ith; (void)nth;
    (void)nrows; (void)ncols;
    return 0;
}

#endif
