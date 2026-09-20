/* ================================================================
 * Q8_K_R8 x Q8_K AVX2 GEMV and GEMM kernels
 * ================================================================
 * Port of llama.cpp ik branch iqk_gemm_kquants.cpp:
 *   mul_mat_q8_k_r8_q8_k
 *
 * Weights: block_q8_k_r8 (GGUF type 399)
 *   d[8] FP16 scales + qs[2048] interleaved int8
 *   Each block covers 8 rows x 256 values (QK_K=256).
 *   Interleaving: qs[32*ib + 4*k + i] for ib=0..63, k=0..7, i=0..3
 *
 * Activations: block_q8_K (one per row)
 *   d (float scale) + qs[256] signed int8
 *
 * Algorithm: sign trick for signed multiplication via maddubs/dpbusd.
 *   s = _mm256_sign_epi8(qx, qx)    // |qx| with sign
 *   sy = _mm256_sign_epi8(shuf_y, qx)  // copy qx sign to activation
 *   maddubs(|qx|, sy) = qx * y     // correct signed dot product
 *
 * 8 rows share the same 256-bit accumulator, each row's contribution
 * goes to a different int32 lane (4 bytes per row per 32-byte chunk).
 *
 * AVX2 + FMA + F16C required. AVX-512 VNNI optional (faster dpbusd).
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

#define QK_K 256  /* Q8_K block length */

/* ================================================================
 * vec_dot_q8_k_r8_q8_k_avx2: 8 weight rows x 1 activation row
 * ================================================================
 * vx: block_q8_k_r8 pointer (n/256 blocks)
 * wy: block_q8_K pointer (n/256 blocks)
 * n: inner dimension (multiple of 256)
 * out: 8 output floats
 * nrows: must be 8
 */
void vec_dot_q8_k_r8_q8_k_avx2(const void *vx, const void *wy, int n,
                                 float *out, int nrows) {
#if defined(__AVX2__) && defined(__F16C__)
    assert(nrows == 8);
    assert(n % QK_K == 0);

    const block_q8_k_r8 *iq8 = (const block_q8_k_r8 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    __m256 acc[8] = {0};
    __m256i isum[8] = {0};

    for (int ib = 0; ib < nb; ib++) {
        /* Load 8 FP16 weight scales -> 8 FP32 */
        __m256 d4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[ib].d));
        float scale_y = qk[ib].d;
        __m256 d4y = _mm256_mul_ps(d4, _mm256_set1_ps(scale_y));

        for (int ib2 = 0; ib2 < QK_K / 16; ib2++) {
            /* Load 4 x 32-byte interleaved weight chunks.
             * Each chunk has 8 rows x 4 bytes interleaved.
             * Offset in bytes: (32*ib2 + 4*k + i) where k=row, i=0..3 */
            __m256i qx[4];
            qx[0] = _mm256_loadu_si256((const __m256i *)iq8[ib].qs + 4 * ib2 + 0);
            qx[1] = _mm256_loadu_si256((const __m256i *)iq8[ib].qs + 4 * ib2 + 1);
            qx[2] = _mm256_loadu_si256((const __m256i *)iq8[ib].qs + 4 * ib2 + 2);
            qx[3] = _mm256_loadu_si256((const __m256i *)iq8[ib].qs + 4 * ib2 + 3);

            /* Sign trick: |qx| for unsigned multiply */
            __m256i s0 = _mm256_sign_epi8(qx[0], qx[0]);
            __m256i s1 = _mm256_sign_epi8(qx[1], qx[1]);
            __m256i s2 = _mm256_sign_epi8(qx[2], qx[2]);
            __m256i s3 = _mm256_sign_epi8(qx[3], qx[3]);

            /* Load 16 activation bytes and broadcast to 256-bit */
            __m128i y128 = _mm_loadu_si128((const __m128i *)qk[ib].qs + ib2);
            __m256i y = _mm256_broadcastsi128_si256(y128);

#ifdef __AVX512VNNI__
#ifdef __AVX512VL__
            isum[0] = _mm256_dpbusd_epi32(isum[0], s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]));
            isum[0] = _mm256_dpbusd_epi32(isum[0], s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1]));
            isum[0] = _mm256_dpbusd_epi32(isum[0], s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]));
            isum[0] = _mm256_dpbusd_epi32(isum[0], s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3]));
#else
            __m256i t1 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0])));
            __m256i t2 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1])));
            __m256i t3 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2])));
            __m256i t4 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3])));
            isum[0] = _mm256_add_epi32(isum[0], _mm256_add_epi32(t1, t2));
            isum[0] = _mm256_add_epi32(isum[0], _mm256_add_epi32(t3, t4));
#endif
#else
            __m256i t1 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0])));
            __m256i t2 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1])));
            __m256i t3 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2])));
            __m256i t4 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3])));
            isum[0] = _mm256_add_epi32(isum[0], _mm256_add_epi32(t1, t2));
            isum[0] = _mm256_add_epi32(isum[0], _mm256_add_epi32(t3, t4));
#endif
        }

        /* Apply scales and accumulate */
        acc[0] = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum[0]), acc[0]);
        isum[0] = _mm256_setzero_si256();
    }

    /* Store 8 output floats */
    _mm256_storeu_ps(out, acc[0]);
#else
    (void)vx; (void)wy; (void)n; (void)out; (void)nrows;
    memset(out, 0, nrows * sizeof(float));
#endif
}

/* ================================================================
 * sgemm_q8_k_r8_q8_k_avx2: batched GEMM
 * ================================================================
 * nrows: number of weight rows (multiple of 8)
 * ncols: number of activation rows (batch size, >= 1)
 * k: inner dimension (multiple of 256)
 * vx: block_q8_k_r8 weights (nrows * row_bytes bytes)
 * vy: block_q8_K activations (ncols * a_row_bytes bytes)
 * out: float[nrows * ncols] (row-major)
 * bs: output row stride (in floats, = ncols)
 * ith, nth: thread partitioning of columns
 */
int sgemm_q8_k_r8_q8_k_avx2(int nrows, int ncols, int k,
                             const void *vx, const void *vy,
                             float *out, size_t bs,
                             int ith, int nth) {
#if defined(__AVX2__) && defined(__F16C__)
    if (nrows < 8 || ncols < 1 || k % QK_K != 0 || nrows % 8 != 0)
        return 0;

    size_t a_row_bytes = (size_t)(k / QK_K) * sizeof(block_q8_K);
    int start_col = (ncols * ith) / nth;
    int end_col = (ncols * (ith + 1)) / nth;
    if (end_col <= start_col) return 0;

    size_t w_stride = (size_t)(k / QK_K) * sizeof(block_q8_k_r8);
    for (int w8 = 0; w8 < nrows / 8; w8++) {
        const block_q8_k_r8 *iq8_base = (const block_q8_k_r8 *)
            ((const char *)vx + w8 * w_stride);
        for (int c = start_col; c < end_col; c++) {
            const block_q8_K *qk = (const block_q8_K *)
                ((const char *)vy + c * a_row_bytes);
            float *out_base = out + w8 * 8 + c * bs;
            vec_dot_q8_k_r8_q8_k_avx2(iq8_base, qk, k, out_base, 8);
        }
    }
    return (nrows / 8) * 8;
#else
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy;
    (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
#endif
}

