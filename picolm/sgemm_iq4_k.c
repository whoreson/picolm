/* ================================================================
 * IQ4_K (plain, GGUF type 139) x Q8_K AVX2 GEMV kernel
 * ================================================================
 * Port of llama.cpp iqk_quantize.cpp dequantization logic
 * adapted for single-row (non-interleaved) IQ4_K format.
 *
 * Weights: block_iq4_k (GGUF type 139), 144 bytes/block
 * Activations: block_q8_K
 *
 * Layout per block (256 values):
 *   d:        FP16 global scale
 *   extra:    16 bits, 2 bits per 32-value subblock (LUT table select)
 *   scales_h: 4 bytes, 2 bits per scale (16 scales, 2 per subblock)
 *   scales_l: 8 bytes, 4-bit magnitude per scale (16 total)
 *   qs:       128 bytes, 4-bit values (2 per byte)
 *
 * Scale: 6-bit signed = (scales_l 4-bit | scales_h 2-bit) - 32
 * LUT: 32 entries (2 tables of 16), extra selects per subblock-half
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

#define QK_K 256

/* Helper: reduce 8 int32 in a __m256i to a single int32 sum */
static inline int hsum_i32x8(__m256i v) {
    __m256i s1 = _mm256_shuffle_epi32(v, 0x1B);
    __m256i a1 = _mm256_add_epi32(v, s1);
    s1 = _mm256_shuffle_epi32(a1, 0x31);
    a1 = _mm256_add_epi32(a1, s1);
    __m128i a_lo = _mm256_castsi256_si128(a1);
    __m128i a_hi = _mm256_extracti128_si256(a1, 1);
    return _mm_cvtsi128_si32(_mm_add_epi32(a_lo, a_hi));
}

void vec_dot_iq4_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
#if defined(__AVX2__) && defined(__F16C__)
    assert(n % QK_K == 0);

    const block_iq4_k *iq4 = (const block_iq4_k *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    /* iq4k_values LUT: 32 entries (16 normal + 16 shifted).
     * Accessed via _mm_shuffle_epi8 with 4-bit indices (0..15). */

    float result = 0.0f;

    for (int ibl = 0; ibl < nb; ibl++) {
        float d = fp16_to_fp32_lookup(iq4[ibl].d);
        float q8_scale = qk[ibl].d;
        float dy = d * q8_scale;

        uint16_t extra = iq4[ibl].extra;
        const uint8_t *qs = iq4[ibl].qs;
        const int8_t *q8 = qk[ibl].qs;

        __m256i isum = _mm256_setzero_si256();

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            /* Scale extraction: 6-bit signed = (scales_l 4-bit | scales_h 2-bit) - 32 */
            const uint8_t sh = iq4[ibl].scales_h[ib / 2] >> (4 * (ib % 2));
            int scale_lo = (int)(((iq4[ibl].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
            int scale_hi = (int)(((iq4[ibl].scales_l[ib] >> 4) | ((sh << 2) & 0x30)) - 32);

            /* LUT table selection per subblock-half */
            int use_table1_lo = extra & 1;
            int use_table1_hi = extra & 2;
            extra >>= 2;

            /* Dequantize 16 lo-nibbles and 16 hi-nibbles to int8, multiply by Q8, accumulate */
            __m128i qs_vec = _mm_loadu_si128((const __m128i *)qs);

            /* Low nibbles: values 0..15 */
            __m128i qx_lo = _mm_and_si128(qs_vec, _mm_set1_epi8(0x0f));
            __m128i y_lo = _mm_shuffle_epi8(
                _mm_loadu_si128((const __m128i *)(use_table1_lo ? iq4k_values + 16 : iq4k_values)),
                qx_lo);

            /* High nibbles: values 16..31 */
            __m128i qx_hi = _mm_and_si128(_mm_srli_epi16(qs_vec, 4), _mm_set1_epi8(0x0f));
            __m128i y_hi = _mm_shuffle_epi8(
                _mm_loadu_si128((const __m128i *)(use_table1_hi ? iq4k_values + 16 : iq4k_values)),
                qx_hi);

            /* Sign trick for signed multiplication via dpbusd/maddubs */
            /* |qx| * sign_extend(qx, y) = qx * y */
            __m128i abs_qx_lo = _mm_sign_epi8(qx_lo, qx_lo);  /* |qx_lo| */
            __m128i abs_qx_hi = _mm_sign_epi8(qx_hi, qx_hi);  /* |qx_hi| */
            __m128i sy_lo = _mm_sign_epi8(y_lo, qx_lo);       /* y_lo * sign(qx_lo) */
            __m128i sy_hi = _mm_sign_epi8(y_hi, qx_hi);       /* y_hi * sign(qx_hi) */

            /* Unsigned * signed dot product -> int16, then widen to int32 */
            __m128i sum16_lo = _mm_maddubs_epi16(abs_qx_lo, sy_lo);
            __m128i sum16_hi = _mm_maddubs_epi16(abs_qx_hi, sy_hi);

            /* Apply per-subblock scales */
            __m128i scale_lo_v = _mm_set1_epi16(scale_lo);
            __m128i scale_hi_v = _mm_set1_epi16(scale_hi);
            __m128i prod_lo = _mm_madd_epi16(sum16_lo, scale_lo_v);
            __m128i prod_hi = _mm_madd_epi16(sum16_hi, scale_hi_v);

            /* Accumulate */
            isum = _mm256_add_epi32(isum, _mm256_set_m128i(prod_hi, prod_lo));

            qs += 16;
            q8 += 32;
        }

        int isum_scalar = hsum_i32x8(isum);
        result += (float)isum_scalar * dy;
    }

    *out = result;
#else
    /* Scalar fallback */
    *out = vec_dot_iq4_k_q8_k(vx, wy, n);
#endif
}

/* ================================================================
 * IQ4_K_R4 x Q8_K AVX2 GEMV kernel (4-row interleaved)
 * GGUF type 339
 * ================================================================
 * Weights: block_iq4_k_r4 (576 bytes, 4 rows x 256 values)
 * Activations: block_q8_K (one row of activations)
 *
 * Processes 4 rows simultaneously, producing 4 dot products.
 * Scale layout: scales_l[32], scales_h[16] (interleaved across rows).
 * qs layout: qs[512], row-interleaved.
 * ================================================================ */

void vec_dot_iq4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
#if defined(__AVX2__) && defined(__F16C__)
    assert(nrows == 4);
    assert(n % QK_K == 0);

    const block_iq4_k_r4 *iq4 = (const block_iq4_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int ibl = 0; ibl < nb; ibl++) {
        const block_iq4_k_r4 *b = &iq4[ibl];
        const block_q8_K *q = &qk[ibl];

        float q8_scale = q->d;

        /* Process each row */
        for (int row = 0; row < 4; row++) {
            float d = fp16_to_fp32_lookup(b->d[row]);
            float dy = d * q8_scale;

            const int8_t *q8 = q->qs;  /* Reset q8 pointer for each row */

            for (int ib = 0; ib < QK_K / 32; ++ib) {
                /* Scale extraction for this row/subblock */
                int is = 8 * ib + row;
                int scale_lo = (int)((((b->scales_l[is % 32] >> (4 * (is / 32))) & 0xf) |
                                      (((b->scales_h[is % 16] >> (2 * (is / 16))) & 3) << 4)) - 32);
                is += 4;
                int scale_hi = (int)((((b->scales_l[is % 32] >> (4 * (is / 32))) & 0xf) |
                                      (((b->scales_h[is % 16] >> (2 * (is / 16))) & 3) << 4)) - 32);

                /* LUT table selection */
                int use_table1_lo = b->extra[row] & (1 << ib);
                int use_table1_hi = b->extra[row + 4] & (1 << ib);

                /* Load qs for this row/subblock: 16 bytes per row per subblock
                 * Layout: qs[64*ib + 4*row + i + offset] for i=0..3 */
                /* We need 16 bytes (32 nibbles) for this row's subblock.
                 * The bytes are interleaved: 4 rows, each contributing 4 bytes per i-group.
                 * For row k, subblock ib: bytes at qs[64*ib + 4*k + 0..3] (4 bytes = 8 nibbles)
                 * plus qs[64*ib + 4*k + 16..19] (for hi-nibbles of second 16)
                 * plus qs[64*ib + 4*k + 32..35] (for lo-nibbles of second group)
                 * plus qs[64*ib + 4*k + 48..51] (for hi-nibbles of second group)
                 *
                 * This is non-contiguous in memory. Scalar dequant is simpler here. */

                /* Fall back to scalar for this subblock due to interleaved layout */
                float sumi = 0.0f;
                const int8_t *vals1 = use_table1_lo ? iq4k_values + 16 : iq4k_values;
                const int8_t *vals2 = use_table1_hi ? iq4k_values + 16 : iq4k_values;

                for (int i = 0; i < 4; ++i) {
                    /* dl1 values: qs offsets 0 and 0 (lo/hi nibble) */
                    uint8_t qbyte0 = b->qs[64 * ib + 4 * row + i + 0];
                    sumi += scale_lo * vals1[qbyte0 & 0xf] * q8[i + 0];
                    sumi += scale_lo * vals1[qbyte0 >> 4] * q8[i + 8];

                    /* dl2 values: qs offsets 16 and 16 */
                    uint8_t qbyte1 = b->qs[64 * ib + 4 * row + i + 16];
                    sumi += scale_hi * vals2[qbyte1 & 0xf] * q8[i + 16];
                    sumi += scale_hi * vals2[qbyte1 >> 4] * q8[i + 24];

                    /* dl1 second group: qs offsets 32 */
                    uint8_t qbyte2 = b->qs[64 * ib + 4 * row + i + 32];
                    sumi += scale_lo * vals1[qbyte2 & 0xf] * q8[i + 4];
                    sumi += scale_lo * vals1[qbyte2 >> 4] * q8[i + 12];

                    /* dl2 second group: qs offsets 48 */
                    uint8_t qbyte3 = b->qs[64 * ib + 4 * row + i + 48];
                    sumi += scale_hi * vals2[qbyte3 & 0xf] * q8[i + 20];
                    sumi += scale_hi * vals2[qbyte3 >> 4] * q8[i + 28];
                }
                q8 += 32;

                /* Accumulate into FP32 result directly */
                result[row] += sumi * dy;
            }
        }
    }

    for (int r = 0; r < nrows; r++) out[r] = result[r];
#else
    /* Scalar fallback: dequantize each weight row, dequantize Q8_K activations,
     * then F32 dot product. */
    {
        float *w_tmp = (float *)malloc((size_t)n * sizeof(float));
        float *a_tmp = (float *)malloc((size_t)n * sizeof(float));
        for (int r = 0; r < nrows; r++) {
            if (w_tmp && a_tmp) {
                dequantize_row_iq4_k_r4_single((const block_iq4_k_r4 *)vx, w_tmp, n, r);
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
