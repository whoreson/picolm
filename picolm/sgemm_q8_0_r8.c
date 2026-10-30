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

void vec_dot_q8_0_r8_q8_0_d_avx2(const void *vx, const void *wy, const float *q_d,
                                 int n, float *out, int nrows) {
    (void)vx; (void)wy; (void)q_d; (void)n;
    if (out) memset(out, 0, (size_t)nrows * sizeof(float));
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
