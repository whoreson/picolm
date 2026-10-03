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
 *
 * Dequant: val = d[k] * scale * q - m[k] * min
 * No LUT (raw 4-bit values 0..15).
 *
 * AVX2 GEMV: vec_dot_q4_k_r4_q8_k_avx2
 *   Uses per-chunk processing with shuffle-based row extraction.
 *   For each of 4 chunks (16 bytes), extracts 4 bytes per row using
 *   _mm_shuffle_epi8, then does 8 nibble-activation MACs per chunk
 *   using broadcasted maddubs. 4 chunks summed = 32 MACs per row.
 *
 * GEMM: sgemm_q4_k_r4_q8_k_avx2
 *   Wrapper that tiles the GEMV over row groups and columns.
 *   Uses vec_dot_q4_k_r4_q8_k_batch4 from quant.c for correctness.
 */

#include "quant.h"
#include <stdlib.h>
#include <assert.h>


/* Forward declarations for extern symbols from quant.c */
extern float fp16_to_fp32_lookup(uint16_t h);
extern float vec_dot_f32_f32(const void *a, const float *b, int n);
extern void dequantize_row_q4_k_r4_single(const block_q4_k_r4 *x, float *dst, int n, int row);
extern void vec_dot_q4_k_r4_q8_k_batch4(const void *vx, const void *vy, int n, float *out);

#if defined(PICOLM_AVX2)
#include <immintrin.h>

void vec_dot_q4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
    assert(nrows == 4);
    assert(n % QK_K == 0);

    const block_q4_k_r4 *x = (const block_q4_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    const __m256i mf  = _mm256_set1_epi8(0xf);
    const __m256i m30 = _mm256_set1_epi8(0x30);

    /* Shuffle masks: extract row k from a 16-byte chunk.
     * Within each chunk, row k's bytes are at offsets k, k+4, k+8, k+12.
     * Each byte is broadcast 4x for maddubs -> madd_epi16 reduction. */
    const __m128i pick_row[4] = {
        _mm_set_epi8(12,12,12,12, 8,8,8,8, 4,4,4,4, 0,0,0,0),
        _mm_set_epi8(13,13,13,13, 9,9,9,9, 5,5,5,5, 1,1,1,1),
        _mm_set_epi8(14,14,14,14, 10,10,10,10, 6,6,6,6, 2,2,2,2),
        _mm_set_epi8(15,15,15,15, 11,11,11,11, 7,7,7,7, 3,3,3,3),
    };

    __m256 acc = _mm256_setzero_ps();

    for (int ibl = 0; ibl < nb; ibl++) {
        const block_q4_k_r4 *b = &x[ibl];
        const block_q8_K *q = &qk[ibl];

        __m256 dl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)b->d));
        __m256 d4 = _mm256_set_m128(_mm256_castps256_ps128(dl), _mm256_castps256_ps128(dl));

        __m256i lbits = _mm256_loadu_si256((const __m256i *)b->scales_l);
        __m128i hbits128 = _mm_loadu_si128((const __m128i *)b->scales_h);
        __m256i hbits = _mm256_set_m128i(hbits128, _mm_slli_epi16(hbits128, 4));
        __m256i scales_6bit = _mm256_or_si256(_mm256_and_si256(lbits, mf),
                                               _mm256_and_si256(hbits, m30));
        uint32_t sc_val[8];
        _mm256_storeu_si256((__m256i *)sc_val, scales_6bit);

        __m256i isum = _mm256_setzero_si256();

        for (int ib = 0; ib < QK_K / 32; ib++) {
            uint32_t s32 = sc_val[ib];
            __m256i scales = _mm256_cvtepi8_epi32(_mm_set1_epi32(s32));

            const uint8_t *qs = b->qs + 64 * ib;
            __m128i c[4];
            c[0] = _mm_loadu_si128((const __m128i *)(qs + 0));
            c[1] = _mm_loadu_si128((const __m128i *)(qs + 16));
            c[2] = _mm_loadu_si128((const __m128i *)(qs + 32));
            c[3] = _mm_loadu_si128((const __m128i *)(qs + 48));

            __m256i y_reg = _mm256_loadu_si256((const __m256i *)q->qs + ib);
            __m128i y_lo = _mm256_castsi256_si128(y_reg);
            __m128i y_hi = _mm256_extracti128_si256(y_reg, 1);

            __m256i raw_dot = _mm256_setzero_si256();

            for (int k = 0; k < 4; k++) {
                /* Extract row k from each chunk: 4 unique bytes each.
                 * After pick_row: bytes at positions 0,4,8,12 (4x repeat). */
                __m128i r[4];
                for (int cc = 0; cc < 4; cc++)
                    r[cc] = _mm_shuffle_epi8(c[cc], pick_row[k]);

                /* Process each chunk: 4 bytes x 2 nibbles = 8 MACs.
                 * For each byte: lo_nibble * act_lo + hi_nibble * act_hi.
                 * Use _mm256_maddubs_epi16 with broadcasted values.
                 *
                 * Chunk 0: q8[i+0] for lo, q8[i+8] for hi, i=0..3
                 * Chunk 1: q8[i+16] for lo, q8[i+24] for hi
                 * Chunk 2: q8[i+4] for lo, q8[i+12] for hi
                 * Chunk 3: q8[i+20] for lo, q8[i+28] for hi
                 *
                 * For each chunk, build a 32-byte weight vector:
                 *   low 16 = 4 lo nibbles each repeated 4x (bytes 0-3,4-7,8-11,12-15)
                 *   high 16 = 4 hi nibbles each repeated 4x
                 * And a 32-byte activation vector:
                 *   low 16 = 4 act_lo bytes each repeated 4x
                 *   high 16 = 4 act_hi bytes each repeated 4x
                 * maddubs: 32 pairs -> 16 int16 -> madd_epi16 -> 8 int32 -> sum -> 1 int32 */

                int32_t row_sum = 0;

                for (int cc = 0; cc < 4; cc++) {
                    /* r[cc] has 4 unique bytes at positions 0,4,8,12 (4x repeat each)
                     * Extract bytes 0-3 (one unique byte per value).
                     * pick_first4: take byte 0 four times, byte 1 four times, etc. */
                    __m128i b4 = _mm_shuffle_epi8(r[cc],
                        _mm_set_epi8(3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0));
                    /* b4 = [v0,v0,v0,v0, v1,v1,v1,v1, v2,v2,v2,v2, v3,v3,v3,v3] */

                    __m128i lo4 = _mm_and_si128(b4, _mm256_castsi256_si128(mf));
                    __m128i hi4 = _mm_and_si128(_mm_srli_epi16(b4, 4), _mm256_castsi256_si128(mf));

                    /* Build activation for this chunk */
                    int act_lo_base, act_hi_base;
                    if (cc == 0) { act_lo_base = 0; act_hi_base = 8; }
                    else if (cc == 1) { act_lo_base = 16; act_hi_base = 24; }
                    else if (cc == 2) { act_lo_base = 4; act_hi_base = 12; }
                    else { act_lo_base = 20; act_hi_base = 28; }

                    __m128i act_lo = _mm_shuffle_epi8(
                        act_lo_base < 16 ? y_lo : y_hi,
                        _mm_set_epi8(
                            act_lo_base+3,act_lo_base+3,act_lo_base+3,act_lo_base+3,
                            act_lo_base+2,act_lo_base+2,act_lo_base+2,act_lo_base+2,
                            act_lo_base+1,act_lo_base+1,act_lo_base+1,act_lo_base+1,
                            act_lo_base,act_lo_base,act_lo_base,act_lo_base));
                    __m128i act_hi = _mm_shuffle_epi8(
                        act_hi_base < 16 ? y_lo : y_hi,
                        _mm_set_epi8(
                            act_hi_base+3,act_hi_base+3,act_hi_base+3,act_hi_base+3,
                            act_hi_base+2,act_hi_base+2,act_hi_base+2,act_hi_base+2,
                            act_hi_base+1,act_hi_base+1,act_hi_base+1,act_hi_base+1,
                            act_hi_base,act_hi_base,act_hi_base,act_hi_base));

                    /* Build 32-byte vectors */
                    __m256i w32 = _mm256_set_m128i(hi4, lo4);
                    __m256i a32 = _mm256_set_m128i(act_hi, act_lo);

                    /* maddubs + madd_epi16 -> 8 int32 (4 from lo, 4 from hi) */
                    __m256i t = _mm256_madd_epi16(_mm256_set1_epi16(1),
                                                   _mm256_maddubs_epi16(w32, a32));
                    /* Sum 8 int32 -> 1 int32 */
                    __m128i t128_lo = _mm256_castsi256_si128(t);
                    __m128i t128_hi = _mm256_extracti128_si256(t, 1);
                    __m128i t128 = _mm_add_epi32(t128_lo, t128_hi);
                    t128 = _mm_add_epi32(t128, _mm_shuffle_epi32(t128, 0x5));
                    t128 = _mm_add_epi32(t128, _mm_shuffle_epi32(t128, 0x3));
                    row_sum += (int32_t)_mm_cvtsi128_si32(t128);
                }

                int32_t *raw_arr = (int32_t *)&raw_dot;
                raw_arr[k] = row_sum;
            }

            isum = _mm256_add_epi32(isum, _mm256_mullo_epi32(scales, raw_dot));
        }

        /* Bias correction */
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

        float q8_scale = q->d;
        __m256 dq = _mm256_set1_ps(q8_scale);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(d4, dq), _mm256_cvtepi32_ps(isum), acc);
        acc = _mm256_sub_ps(acc, _mm256_mul_ps(bias256, dq));
    }

    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    _mm_storeu_ps(out, sum);
}
#elif defined(PICOLM_NEON)
#include <arm_neon.h>

/* NEON GEMV for Q4_K_R4 x Q8_K.
 * Processes 4 interleaved rows against one Q8_K activation row.
 * Uses scalar extraction (R4 layout is not SIMD-friendly on basic NEON)
 * but vectorizes the int8 MAC across 16 values per row with vpadalq_s16. */
void vec_dot_q4_k_r4_q8_k_neon(const void *vx, const void *wy, int n,
                                float *out, int nrows) {
    if (nrows != 4 || n % QK_K != 0) {
        vec_dot_q4_k_r4_q8_k_batch4(vx, wy, n, out);
        return;
    }

    const block_q4_k_r4 *x = (const block_q4_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    float acc[4] = {0, 0, 0, 0};

    for (int ibl = 0; ibl < nb; ibl++) {
        const block_q4_k_r4 *b = &x[ibl];
        const block_q8_K *q = &qk[ibl];
        float q8_scale = q->d;

        /* Bias correction: m * min * sum(q8) per row */
        float bias[4] = {0, 0, 0, 0};
        for (int ib = 0; ib < QK_K / 32; ib++) {
            for (int k = 0; k < 4; k++) {
                int is = 4 * ib + k;
                float ml = fp16_to_fp32_lookup(b->d[k + 4]) *
                    ((b->scales_l[is] >> 4) |
                     (((b->scales_h[is % 16] >> (4 * (is / 16))) & 0x0c) << 2));
                bias[k] += ml * (q->bsums[ib * 2 + 0] + q->bsums[ib * 2 + 1]);
            }
        }

        for (int ib = 0; ib < QK_K / 32; ib++) {
            float scales[4];
            for (int k = 0; k < 4; k++) {
                int is = 4 * ib + k;
                scales[k] = fp16_to_fp32_lookup(b->d[k]) *
                    ((b->scales_l[is] & 0xf) |
                     (((b->scales_h[is % 16] >> (4 * (is / 16))) & 0x03) << 4));
            }

            /* Reorder Q8_K activations to match the Q4_K_R4 weight order. */
            int8_t a_reordered[32];
            const int8_t *q8 = q->qs + 32 * ib;
            for (int i = 0; i < 4; i++) {
                a_reordered[8 * i + 0] = q8[i + 0];
                a_reordered[8 * i + 1] = q8[i + 8];
                a_reordered[8 * i + 2] = q8[i + 16];
                a_reordered[8 * i + 3] = q8[i + 24];
                a_reordered[8 * i + 4] = q8[i + 4];
                a_reordered[8 * i + 5] = q8[i + 12];
                a_reordered[8 * i + 6] = q8[i + 20];
                a_reordered[8 * i + 7] = q8[i + 28];
            }

            /* Extract weight nibbles for each row in the same order. */
            int8_t w_reordered[4][32];
            for (int k = 0; k < 4; k++) {
                const uint8_t *qs = b->qs + 64 * ib + 4 * k;
                for (int i = 0; i < 4; i++) {
                    w_reordered[k][8 * i + 0] = (int8_t)(qs[i + 0] & 0xf);
                    w_reordered[k][8 * i + 1] = (int8_t)(qs[i + 0] >> 4);
                    w_reordered[k][8 * i + 2] = (int8_t)(qs[i + 16] & 0xf);
                    w_reordered[k][8 * i + 3] = (int8_t)(qs[i + 16] >> 4);
                    w_reordered[k][8 * i + 4] = (int8_t)(qs[i + 32] & 0xf);
                    w_reordered[k][8 * i + 5] = (int8_t)(qs[i + 32] >> 4);
                    w_reordered[k][8 * i + 6] = (int8_t)(qs[i + 48] & 0xf);
                    w_reordered[k][8 * i + 7] = (int8_t)(qs[i + 48] >> 4);
                }
            }

            int32x4_t sums[4] = { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0) };
            for (int j = 0; j < 32; j += 8) {
                int8x8_t a8 = vld1_s8(a_reordered + j);
                for (int k = 0; k < 4; k++) {
                    int8x8_t w8 = vld1_s8(w_reordered[k] + j);
                    int16x8_t p = vmull_s8(w8, a8);
                    sums[k] = vpadalq_s16(sums[k], p);
                }
            }
            for (int k = 0; k < 4; k++)
                acc[k] += (float)vaddvq_s32(sums[k]) * scales[k] * q8_scale;
        }

        for (int k = 0; k < 4; k++)
            acc[k] -= bias[k] * q8_scale;
    }

    for (int k = 0; k < 4; k++)
        out[k] = acc[k];
}
#else
void vec_dot_q4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
    float *w_tmp = (float *)malloc((size_t)n * sizeof(float));
    float *a_tmp = (float *)malloc((size_t)n * sizeof(float));
    for (int r = 0; r < nrows; r++) {
        if (w_tmp && a_tmp) {
            dequantize_row_q4_k_r4_single((const block_q4_k_r4 *)vx, w_tmp, n, r);
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

/* ================================================================
 * Q4_K_R4 x Q8_K GEMM wrapper
 * ================================================================
 * Uses vec_dot_q4_k_r4_q8_k_batch4 from quant.c for correctness.
 * ================================================================ */
int sgemm_q4_k_r4_q8_k_avx2(int nrows, int ncols, int k,
                              const void *vx, const void *vy,
                              float *out, size_t bs,
                              int ith, int nth) {
    if (nrows < 4 || ncols < 1 || k <= 0 || k % QK_K != 0 || nrows % 4 != 0) {
        return 0;
    }

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
