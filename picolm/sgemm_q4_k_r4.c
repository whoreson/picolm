/* sgemm_q4_k_r4.c -- AVX2 kernels for Q4_K_R4 (GGUF type 212)
 *
 * Q4_K_R4: 4-row interleaved Q4_K, 576 bytes per block.
 *   d[8]:       FP16: d[0..3]=scales per row, d[4..7]=mins per row
 *   scales_h[16]: 2-bit scale extensions
 *   scales_l[32]: 4-bit magnitude scales
 *   qs[512]:     4-bit values interleaved across 4 rows
 *
 * Scale encoding: 6-bit unsigned
 *   scale = (scales_l[is] & 0xf) | ((scales_h[is%16] >> 4*(is/16)) & 0x03) << 4
 *   min   = (scales_l[is] >> 4)  | ((scales_h[is%16] >> 4*(is/16)) & 0x0c) << 2
 *   where is = 4*ib + row (ib=0..7 subblocks, row=0..3)
 *   Each subblock (32 values) has ONE scale and ONE min per row.
 *
 * Dequant: val = d[k] * scale * q - m[k] * min
 * No LUT (raw 4-bit values 0..15).
 */

#include "quant.h"
#include <stdlib.h>
#include <assert.h>

/* Forward declarations for extern symbols from quant.c */
extern float fp16_to_fp32_lookup(uint16_t h);
extern float vec_dot_f32_f32(const void *a, const float *b, int n);
extern void dequantize_row_q4_k_r4_single(const block_q4_k_r4 *x, float *dst, int n, int row);
extern void vec_dot_q4_k_r4_q8_k_batch4(const void *vx, const void *vy, int n, float *out);

#ifdef PICOLM_AVX2
#include <immintrin.h>

/* vec_dot_q4_k_r4_q8_k_avx2: AVX2 GEMV for Q4_K_R4 x Q8_K.
 *
 * Processes 4 weight rows (R4 interleaved) x 1 activation row (Q8_K).
 * Uses maddubs_epi16 for unsigned x signed int8 MAC.
 *
 * Ported from ik_llama.cpp iqk_gemm_kquants.cpp:1537 mul_mat_q4_k_r4_q8_k.
 * Bias correction: scalar loop over subblocks (PicoLM's block_q8_K stores
 * bsums as int16[16], requiring scalar min extraction).
 */
void vec_dot_q4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
#if defined(__AVX2__) && defined(__F16C__)
    assert(nrows == 4);
    assert(n % QK_K == 0);

    const block_q4_k_r4 *x = (const block_q4_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    const __m256i mf  = _mm256_set1_epi8(0xf);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    const __m256i m16 = _mm256_set1_epi16(1);

    __m256 acc = _mm256_setzero_ps();

    for (int ibl = 0; ibl < nb; ibl++) {
        const block_q4_k_r4 *b = &x[ibl];
        const block_q8_K *q = &qk[ibl];

        /* Load FP16 scales: d[0..3]=scale, d[4..7]=min.
         * dl = cvtph_ps(d[0..7]) -> [s0,s1,s2,s3, m0,m1,m2,m3]
         * d4 = [s0,s1,s2,s3, s0,s1,s2,s3] (duplicated) */
        __m256 dl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)b->d));
        __m256 d4 = _mm256_set_m128(_mm256_castps256_ps128(dl), _mm256_castps256_ps128(dl));

        /* Scale extraction:
         * lbits = scales_l (32 bytes): low nibble = scale, high nibble = min
         * hbits = scales_h (16 bytes) loaded as [hbits, hbits << 4]
         *
         * scale = (lbits & 0xf) | (hbits & 0x30)
         * min   = ((lbits >> 4) & 0xf) | ((hbits >> 2) & 0x30)
         */
        __m256i lbits = _mm256_loadu_si256((const __m256i *)b->scales_l);
        __m128i hbits128 = _mm_loadu_si128((const __m128i *)b->scales_h);
        __m256i hbits = _mm256_set_m128i(hbits128, _mm_slli_epi16(hbits128, 4));

        /* 6-bit scales: (lbits & 0xf) | (hbits & 0x30) */
        __m256i scales_6bit = _mm256_or_si256(_mm256_and_si256(lbits, mf),
                                               _mm256_and_si256(hbits, m30));

        /* Store scales for per-subblock access.
         * scales_6bit has 32 bytes: byte[is] = 6-bit scale for is=0..31.
         * is = 4*ib + k, so bytes 4*ib..4*ib+3 are the 4 rows of subblock ib. */
        uint32_t sc_val[8];
        _mm256_storeu_si256((__m256i *)sc_val, scales_6bit);

        __m256i isum = _mm256_setzero_si256();

        for (int ib = 0; ib < QK_K / 32; ib++) {
            /* Load scales for this subblock: 4 rows (is = 4*ib + k).
             * Broadcast 4 bytes to all 4 dwords, then cvtepi8_epi32 gives
             * [s0,s1,s2,s3, s0,s1,s2,s3] which matches isum's 8 lanes. */
            uint32_t s32 = sc_val[ib];
            __m256i scales = _mm256_cvtepi8_epi32(_mm_set1_epi32(s32));

            /* Dequantize: two 32-byte loads (bits1 + bits2) */
            __m256i bits1 = _mm256_loadu_si256((const __m256i *)b->qs + 2 * ib + 0);
            __m256i bits2 = _mm256_loadu_si256((const __m256i *)b->qs + 2 * ib + 1);

            /* No LUT -- raw nibble values are already 0..15 */
            __m256i qx[4];
            qx[0] = _mm256_and_si256(bits1, mf);
            qx[1] = _mm256_and_si256(bits2, mf);
            qx[2] = _mm256_and_si256(_mm256_srli_epi16(bits1, 4), mf);
            qx[3] = _mm256_and_si256(_mm256_srli_epi16(bits2, 4), mf);

            /* maddubs: unsigned x signed -> signed 16-bit */
            __m256i y_reg = _mm256_loadu_si256((const __m256i *)q->qs + ib);

            __m256i t1 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y_reg, 0x00)));
            __m256i t2 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y_reg, 0x55)));
            __m256i t3 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y_reg, 0xaa)));
            __m256i t4 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y_reg, 0xff)));
            __m256i sumi = _mm256_add_epi32(_mm256_add_epi32(t1, t2), _mm256_add_epi32(t3, t4));

            /* Scale x int32 dot */
            isum = _mm256_add_epi32(isum, _mm256_mullo_epi32(scales, sumi));
        }

        /* Bias correction: scalar loop over subblocks.
         * Each subblock has ONE min per row (6-bit), and 2 bsums entries
         * (bsums[ib*2+0] and bsums[ib*2+1]).
         * bias[k] = sum over ib of: ml * (bsums[ib*2+0] + bsums[ib*2+1])
         * where ml = m[k] * min_6bit
         */
        float bias[4] = {0, 0, 0, 0};
        for (int ib = 0; ib < QK_K / 32; ib++) {
            for (int k = 0; k < 4; k++) {
                int is = 4 * ib + k;
                float ml = fp16_to_fp32_lookup(b->d[k + 4]) *
                    ((b->scales_l[is] >> 4) |
                     ((b->scales_h[is % 16] >> (4 * (is / 16))) & 0x0c) << 2);
                bias[k] += ml * (q->bsums[ib * 2 + 0] + q->bsums[ib * 2 + 1]);
            }
        }
        __m128 bias128 = _mm_loadu_ps(bias);
        __m256 bias256 = _mm256_set_m128(bias128, bias128);

        /* acc += d4 * q8_scale * isum - bias * q8_scale */
        float q8_scale = q->d;
        __m256 dq = _mm256_set1_ps(q8_scale);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(d4, dq), _mm256_cvtepi32_ps(isum), acc);
        acc = _mm256_sub_ps(acc, _mm256_mul_ps(bias256, dq));
    }

    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    _mm_storeu_ps(out, sum);
#else
    /* Scalar fallback: dequantize each weight row, dequantize Q8_K activations,
     * then F32 dot product. */
    {
        float *w_tmp = (float *)malloc((size_t)n * sizeof(float));
        float *a_tmp = (float *)malloc((size_t)n * sizeof(float));
        for (int r = 0; r < nrows; r++) {
            if (w_tmp && a_tmp) {
                dequantize_row_q4_k_r4_single((const block_q4_k_r4 *)vx, w_tmp, n, r);
                /* wy is block_q8_K -- dequantize to F32 */
                for (int i = 0; i < n; i++) {
                    int ib = i / 256;
                    int io = i % 256;
                    a_tmp[i] = (float)((const block_q8_K *)wy)[ib].qs[io] *
                               ((const block_q8_K *)wy)[ib].d / 127.0f;
                }
                out[r] = vec_dot_f32_f32(w_tmp, a_tmp, n);
            }
        }
        free(w_tmp);
        free(a_tmp);
    }
#endif
}

#else
/* Non-AVX2 stub */
void vec_dot_q4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
    (void)vx; (void)wy; (void)n; (void)out; (void)nrows;
}
#endif

/* ================================================================
 * Q4_K_R4 x Q8_K GEMM wrapper
 * ================================================================
 * Port of llama.cpp ik branch iqk_gemm_kquants.cpp:
 *   mul_mat_q4_k_r4_q8_k
 *
 * Weights: block_q4_k_r4 (4-row interleaved Q4_K, 576 bytes/group)
 * Activations: block_q8_K (plain, one per activation row)
 *
 * Calls vec_dot_q4_k_r4_q8_k_batch4 from quant.c (scalar, correct
 * stride-4 interleaved qs layout). The AVX2 GEMV in this file has
 * a known bug with the qs layout and is not used here.
 * ================================================================ */
int sgemm_q4_k_r4_q8_k_avx2(int nrows, int ncols, int k,
                              const void *vx, const void *vy,
                              float *out, size_t bs,
                              int ith, int nth) {
    if (nrows < 4 || ncols < 1 || k <= 0 || k % QK_K != 0 || nrows % 4 != 0) {
        return 0;
    }

    /* Per-row-equivalent byte stride: gguf_type_row_size(GGUF_TYPE_Q4_K_R4, k)
     * == sizeof(block_q4_k_r4) * (k/QK_K) / 4. Multiplying by 4 gives the
     * real byte size of one interleaved block_q4_k_r4 group. */
    const size_t row_bytes = gguf_type_row_size(GGUF_TYPE_Q4_K_R4, k);
    const size_t group_bytes = row_bytes * 4;
    const size_t q8k_row_bytes = gguf_type_row_size(GGUF_TYPE_Q8_K, k);

    int start_col = (ncols * ith) / nth;
    int end_col = (ncols * (ith + 1)) / nth;
    if (end_col <= start_col) return 0;

    for (int w4 = 0; w4 < nrows / 4; w4++) {
        const char *group_base = (const char *)vx + (size_t)w4 * group_bytes;
        for (int c = start_col; c < end_col; c++) {
            const char *qy = (const char *)vy + (size_t)c * q8k_row_bytes;
            float out4[4];
            vec_dot_q4_k_r4_q8_k_batch4(group_base, qy, k, out4);
            for (int r = 0; r < 4; r++)
                out[w4 * 4 + r + c * bs] = out4[r];
        }
    }
    return (nrows / 4) * 4;
}
