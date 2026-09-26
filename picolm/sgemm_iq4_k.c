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
#if defined(__AVX2__)

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

    /* iq4k_values LUT: 32 entries (2 tables of 16), broadcast to both lanes */
    static const int8_t kvalues_iq4k[32] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
    };
    const __m256i values = _mm256_loadu_si256((const __m256i *)kvalues_iq4k);

    const __m256i m4  = _mm256_set1_epi8(0xf);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    const __m256i m32 = _mm256_set1_epi8(32);
    const __m256i ms  = _mm256_set1_epi8(4);
    const __m256i shift_shuffle = _mm256_set_epi64x(
        0x0707070706060606ULL, 0x0505050504040404ULL,
        0x0303030302020202ULL, 0x0101010100000000ULL);
    const __m256i m16 = _mm256_set1_epi16(1);

    __m256 acc = _mm256_setzero_ps();
    uint64_t stored_scales[8];

    for (int ibl = 0; ibl < nb; ibl++) {
        const block_iq4_k_r4 *b = &iq4[ibl];
        const block_q8_K *q = &qk[ibl];

        __m128 dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)b->d));
        __m256 d4 = _mm256_set_m128(dl, dl);
        float q8_scale = q->d;
        __m256 d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8_scale));

        /* Scale extraction: same as ik_llama's mul_mat_iq4_k_r4_q8_k */
        __m256i slbits = _mm256_loadu_si256((const __m256i *)b->scales_l);
        __m256i sl1 = _mm256_and_si256(slbits, m4);
        __m256i sl2 = _mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4);
        __m128i shbits = _mm_loadu_si128((const __m128i*)b->scales_h);
        __m256i sh = _mm256_set_m128i(_mm_srli_epi16(shbits, 2), shbits);
        __m256i i8scales1 = _mm256_sub_epi8(
            _mm256_or_si256(sl1, _mm256_and_si256(m30, _mm256_slli_epi16(sh, 4))), m32);
        __m256i i8scales2 = _mm256_sub_epi8(
            _mm256_or_si256(sl2, _mm256_and_si256(m30, sh)), m32);
        _mm256_storeu_si256((__m256i *)stored_scales + 0, i8scales1);
        _mm256_storeu_si256((__m256i *)stored_scales + 1, i8scales2);

        __m256i extra = _mm256_set1_epi64x(*(const uint64_t *)b->extra);
        __m256i isum = _mm256_setzero_si256();

        for (int ib = 0; ib < QK_K / 32; ib++) {
            /* Load scales for this subblock: same pattern as IQ2_K_R4/IQ3_K_R4 */
            __m128i s128 = _mm_loadl_epi64((const __m128i *)(stored_scales + ib));
            __m128i s16 = _mm_cvtepi8_epi16(s128);
            __m256i scales = _mm256_cvtepi16_epi32(s16);

            /* Dequantize via LUT: two 32-byte loads (bits1 + bits2) */
            __m256i bits1 = _mm256_loadu_si256((const __m256i *)b->qs + 2 * ib + 0);
            __m256i bits2 = _mm256_loadu_si256((const __m256i *)b->qs + 2 * ib + 1);
            __m256i shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra, 2));
            extra = _mm256_srli_epi16(extra, 1);
            shift = _mm256_shuffle_epi8(shift, shift_shuffle);

            __m256i qx[4];
            qx[0] = _mm256_add_epi8(shift, _mm256_shuffle_epi8(values, _mm256_and_si256(bits1, m4)));
            qx[1] = _mm256_add_epi8(shift, _mm256_shuffle_epi8(values, _mm256_and_si256(bits2, m4)));
            qx[2] = _mm256_add_epi8(shift, _mm256_shuffle_epi8(values, _mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4)));
            qx[3] = _mm256_add_epi8(shift, _mm256_shuffle_epi8(values, _mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4)));

            /* Sign trick: |qx| for unsigned multiply, sign applied to y */
            __m256i s0 = _mm256_sign_epi8(qx[0], qx[0]);
            __m256i s1 = _mm256_sign_epi8(qx[1], qx[1]);
            __m256i s2 = _mm256_sign_epi8(qx[2], qx[2]);
            __m256i s3 = _mm256_sign_epi8(qx[3], qx[3]);

            __m256i y_reg = _mm256_loadu_si256((const __m256i *)q->qs + ib);

            __m256i t1 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x00), qx[0])));
            __m256i t2 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x55), qx[1])));
            __m256i t3 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xaa), qx[2])));
            __m256i t4 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xff), qx[3])));
            __m256i sumi = _mm256_add_epi32(_mm256_add_epi32(t1, t2), _mm256_add_epi32(t3, t4));

            isum = _mm256_add_epi32(isum, _mm256_mullo_epi32(scales, sumi));
        }

        acc = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum), acc);
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
#else
/* Non-AVX2 stubs for cross-compilation compatibility */
void vec_dot_iq4_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) { (void)vx; (void)wy; (void)n; (void)out; }
void vec_dot_iq4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n, float *out, int nrows) { (void)vx; (void)wy; (void)n; (void)out; (void)nrows; }
#endif
