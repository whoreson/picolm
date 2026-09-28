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

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

/* IQ4_K NEON LUT: 32 entries (16 normal + 16 shifted), padded to 32 bytes.
 * Table 0 (indices 0-15):  {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113}
 * Table 1 (indices 16-31): {-123,-100,-79,-61,-45,-31,-18, -6, 5,17,29,42,57,73,93,117} */
static const int8_t iq4k_values_neon[32] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
};

/* ================================================================
 * IQ4_K plain x Q8_K ARM NEON GEMV kernel
 *
 * 4-bit non-linear quantization:
 *   qs[128] = 4-bit values (2 per byte, 256 values total)
 *   scales_h[4] = 2 bits per scale (16 scales total)
 *   scales_l[8] = 4-bit magnitude per scale (16 total)
 *   Scale = (scales_l_4bit | (scales_h_2bit << 4)) - 32  (6-bit signed, offset 32)
 *   LUT: 32 entries, 4-bit index (0-15), extra bits select table (+16)
 * ================================================================ */
void vec_dot_iq4_k_q8_k_neon(const void *vx, const void *wy, int n, float *out) {
    assert(n % QK_K == 0);

    const block_iq4_k *iq4 = (const block_iq4_k *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    /* Two separate LUTs: normal (0-15) and shifted (16-31).
     * vqtbl1q_u8 only uses low 4 bits of index (0-15), so we can't
     * use a single 32-entry LUT with offset. Instead, load both tables
     * and select per-element. */
    const uint8x16_t lut0 = vld1q_u8((const uint8_t *)iq4k_values_neon);       /* indices 0-15 */
    const uint8x16_t lut1 = vld1q_u8((const uint8_t *)iq4k_values_neon + 16); /* indices 16-31 */
    const uint8x16_t m0f = vdupq_n_u8(0x0f);

    float result = 0.0f;

    for (int ibl = 0; ibl < nb; ibl++) {
        float d = fp16_to_fp32_lookup(iq4[ibl].d);
        float q8_scale = qk[ibl].d;
        float dy = d * q8_scale;

        uint16_t extra = iq4[ibl].extra;
        const uint8_t *qs = iq4[ibl].qs;
        const int8_t *q8_base = qk[ibl].qs;

        int32x4_t isum = vdupq_n_s32(0);

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            /* Scale extraction: 6-bit signed = (scales_l 4-bit | scales_h 2-bit) - 32 */
            const uint8_t sh = iq4[ibl].scales_h[ib / 2] >> (4 * (ib % 2));
            int scale_lo = (int)(((iq4[ibl].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
            int scale_hi = (int)(((iq4[ibl].scales_l[ib] >> 4) | ((sh << 2) & 0x30)) - 32);

            /* LUT table selection per subblock-half: 0=normal, 1=shifted */
            int use_shifted_lo = extra & 1;
            int use_shifted_hi = (extra >> 1) & 1;
            extra >>= 2;

            /* Load 32 activations */
            int8x16_t y_lo = vld1q_s8(q8_base + ib * 32);
            int8x16_t y_hi = vld1q_s8(q8_base + ib * 32 + 16);

            /* Load 16 qs bytes (32 4-bit values) */
            uint8x16_t qs_vec = vld1q_u8(qs);

            /* Low nibbles: values 0..15 */
            uint8x16_t ql = vandq_u8(qs_vec, m0f);
            /* High nibbles: values 0..15 */
            uint8x16_t qh = vandq_u8(vshrq_n_u8(qs_vec, 4), m0f);

            /* LUT lookup: select between lut0 and lut1 based on extra bit */
            uint8x16_t qx_lo_u8, qx_hi_u8;
            if (use_shifted_lo) qx_lo_u8 = vqtbl1q_u8(lut1, ql);
            else qx_lo_u8 = vqtbl1q_u8(lut0, ql);
            if (use_shifted_hi) qx_hi_u8 = vqtbl1q_u8(lut1, qh);
            else qx_hi_u8 = vqtbl1q_u8(lut0, qh);
            int8x16_t qx_lo = vreinterpretq_s8_u8(qx_lo_u8);
            int8x16_t qx_hi = vreinterpretq_s8_u8(qx_hi_u8);

            /* int8 MAC via widening multiply */
            int32x4_t s_lo = vaddq_s32(
                vpaddlq_s16(vmull_s8(vget_low_s8(qx_lo), vget_low_s8(y_lo))),
                vpaddlq_s16(vmull_high_s8(qx_lo, y_lo)));
            int32x4_t s_hi = vaddq_s32(
                vpaddlq_s16(vmull_s8(vget_low_s8(qx_hi), vget_low_s8(y_hi))),
                vpaddlq_s16(vmull_high_s8(qx_hi, y_hi)));

            /* Apply per-subblock scales */
            isum = vaddq_s32(isum, vaddq_s32(
                vmulq_n_s32(s_lo, scale_lo),
                vmulq_n_s32(s_hi, scale_hi)));

            qs += 16;
        }

        int total = vaddlvq_s32(isum);
        result += (float)total * dy;
    }

    *out = result;
}

/* IQ4_K plain x Q8_K ARM NEON GEMM kernel.
 * Tiled: 4 weight rows x 2 activation rows per tile. */
int sgemm_iq4_k_q8_k_neon(int nrows, int ncols, int k,
                           const void *vx, const void *vy,
                           float *out, size_t bs,
                           int ith, int nth) {
    int row_stride = 4;
    int col_stride = 2;
    int nrows_tile = (nrows + row_stride - 1) / row_stride;
    int ncols_tile = (ncols + col_stride - 1) / col_stride;

    int rows_per_thread = (nrows_tile + nth - 1) / nth;
    int row_start = ith * rows_per_thread;
    int row_end = row_start + rows_per_thread;
    if (row_end > nrows_tile) row_end = nrows_tile;
    if (row_start >= row_end) return row_start;

    const uint8x16_t lut0 = vld1q_u8((const uint8_t *)iq4k_values_neon);
    const uint8x16_t lut1 = vld1q_u8((const uint8_t *)iq4k_values_neon + 16);
    const uint8x16_t m0f = vdupq_n_u8(0x0f);

    for (int ii = row_start; ii < row_end; ii++) {
        int actual_nrows = ii * row_stride;
        if (actual_nrows + row_stride > nrows) {
            row_stride = nrows - actual_nrows;
            if (row_stride <= 0) break;
        }

        for (int jj = 0; jj < ncols_tile; jj++) {
            int actual_ncols = jj * col_stride;
            if (actual_ncols + col_stride > ncols) {
                col_stride = ncols - actual_ncols;
                if (col_stride <= 0) break;
            }

            float acc[4] = {0};

            for (int col = 0; col < col_stride; col++) {
                const int8_t *q8_row = ((const int8_t *)vy) + (actual_ncols + col) * k;

                for (int ibl = 0; ibl < k / QK_K; ibl++) {
                    const block_iq4_k *blk = (const block_iq4_k *)vx + ibl * row_stride;
                    float q8_scale = ((const block_q8_K *)q8_row)[ibl].d;
                    const int8_t *q8_base = ((const block_q8_K *)q8_row)[ibl].qs;

                    for (int r = 0; r < row_stride; r++) {
                        float d = fp16_to_fp32_lookup(blk[r].d);
                        float dy = d * q8_scale;
                        uint16_t extra = blk[r].extra;
                        const uint8_t *qs = blk[r].qs;

                        int32x4_t isum = vdupq_n_s32(0);

                        for (int ib = 0; ib < QK_K / 32; ++ib) {
                            const uint8_t sh = blk[r].scales_h[ib / 2] >> (4 * (ib % 2));
                            int scale_lo = (int)(((blk[r].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
                            int scale_hi = (int)(((blk[r].scales_l[ib] >> 4) | ((sh << 2) & 0x30)) - 32);

                            int use_shifted_lo = extra & 1;
                            int use_shifted_hi = (extra >> 1) & 1;
                            extra >>= 2;

                            int8x16_t y_lo = vld1q_s8(q8_base + ib * 32);
                            int8x16_t y_hi = vld1q_s8(q8_base + ib * 32 + 16);

                            uint8x16_t qs_vec = vld1q_u8(qs);
                            uint8x16_t ql = vandq_u8(qs_vec, m0f);
                            uint8x16_t qh = vandq_u8(vshrq_n_u8(qs_vec, 4), m0f);

                            uint8x16_t qx_lo_u8, qx_hi_u8;
                            if (use_shifted_lo) qx_lo_u8 = vqtbl1q_u8(lut1, ql);
                            else qx_lo_u8 = vqtbl1q_u8(lut0, ql);
                            if (use_shifted_hi) qx_hi_u8 = vqtbl1q_u8(lut1, qh);
                            else qx_hi_u8 = vqtbl1q_u8(lut0, qh);
                            int8x16_t qx_lo = vreinterpretq_s8_u8(qx_lo_u8);
                            int8x16_t qx_hi = vreinterpretq_s8_u8(qx_hi_u8);

                            int32x4_t s_lo = vaddq_s32(
                                vpaddlq_s16(vmull_s8(vget_low_s8(qx_lo), vget_low_s8(y_lo))),
                                vpaddlq_s16(vmull_high_s8(qx_lo, y_lo)));
                            int32x4_t s_hi = vaddq_s32(
                                vpaddlq_s16(vmull_s8(vget_low_s8(qx_hi), vget_low_s8(y_hi))),
                                vpaddlq_s16(vmull_high_s8(qx_hi, y_hi)));

                            isum = vaddq_s32(isum, vaddq_s32(
                                vmulq_n_s32(s_lo, scale_lo),
                                vmulq_n_s32(s_hi, scale_hi)));

                            qs += 16;
                        }

                        acc[r] += (float)vaddlvq_s32(isum) * dy;
                    }
                }
            }

            for (int r = 0; r < row_stride; r++) {
                out[actual_nrows + r + (jj * col_stride) * bs] = acc[r];
            }
        }
    }

    return row_end;
}

/* NEON function declarations for quant.c dispatcher */
extern void vec_dot_iq4_k_q8_k_neon(const void *vx, const void *wy, int n, float *out);
extern int sgemm_iq4_k_q8_k_neon(int nrows, int ncols, int k,
                                  const void *vx, const void *vy,
                                  float *out, size_t bs, int ith, int nth);
#endif /* ARM_NEON */

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

/* ================================================================
 * IQ4_K_R4 x Q8_K ARM NEON GEMV kernel
 *
 * 4-row interleaved 4-bit non-linear quantization.
 * Scalar inner loop with scalar LUT lookup (R4 layout makes SIMD
 * difficult due to complex interleaving of qs/scales/extra).
 * ================================================================ */
static inline int iq4_k_r4_scale(const uint8_t *scales_l, const uint8_t *scales_h, int idx) {
    int sl = (scales_l[idx % 32] >> (4 * (idx / 32))) & 0xf;
    int sh = (scales_h[idx % 16] >> (2 * (idx / 16))) & 3;
    return (sl | (sh << 4)) - 32;
}

void vec_dot_iq4_k_r4_q8_k_neon(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
    assert(nrows == 4);
    assert(n % QK_K == 0);

    const block_iq4_k_r4 *iq4 = (const block_iq4_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    /* Two LUT tables for scalar lookup */
    const int8_t lut0[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };
    const int8_t lut1[16] = {
        -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
    };

    float accf[4] = {0, 0, 0, 0};
    float d_arr[4];

    for (int ibl = 0; ibl < nb; ibl++) {
        for (int r = 0; r < 4; r++)
            d_arr[r] = fp16_to_fp32_lookup(iq4[ibl].d[r]);
        float q8_scale = qk[ibl].d;
        const int8_t *q8_base = qk[ibl].qs;
        const uint8_t *scales_l = iq4[ibl].scales_l;
        const uint8_t *scales_h = iq4[ibl].scales_h;
        const uint8_t *extra = iq4[ibl].extra;
        const uint8_t *qs_base = iq4[ibl].qs;

        for (int ib = 0; ib < QK_K / 32; ib++) {
            const uint8_t *qs = qs_base + ib * 64;  /* 64 bytes per subblock for 4 rows */

            for (int iy = 0; iy < 4; iy++) {
                int s1 = iq4_k_r4_scale(scales_l, scales_h, 8 * ib + iy);
                int s2 = iq4_k_r4_scale(scales_l, scales_h, 8 * ib + iy + 4);

                const int8_t *values1 = (extra[iy] & (1 << ib)) ? lut1 : lut0;
                const int8_t *values2 = (extra[iy + 4] & (1 << ib)) ? lut1 : lut0;

                int32_t row_sum = 0;
                int base_qs = 64 * ib + 4 * iy;
                int base_q8 = 32 * ib;

                /* Process 32 values: 4 groups of 8 (i=0..3, each producing 2 values) */
                for (int i = 0; i < 4; i++) {
                    /* Low nibble of qs[base_qs + i] -> q8[base_q8 + i] */
                    row_sum += s1 * values1[qs[base_qs + i] & 0xf] * q8_base[base_q8 + i + 0];
                    /* High nibble of qs[base_qs + i] -> q8[base_q8 + i + 8] */
                    row_sum += s1 * values1[qs[base_qs + i] >> 4] * q8_base[base_q8 + i + 8];
                    /* Low nibble of qs[base_qs + i + 16] -> q8[base_q8 + i + 16] */
                    row_sum += s2 * values2[qs[base_qs + i + 16] & 0xf] * q8_base[base_q8 + i + 16];
                    /* High nibble of qs[base_qs + i + 16] -> q8[base_q8 + i + 24] */
                    row_sum += s2 * values2[qs[base_qs + i + 16] >> 4] * q8_base[base_q8 + i + 24];
                    /* Low nibble of qs[base_qs + i + 32] -> q8[base_q8 + i + 4] */
                    row_sum += s1 * values1[qs[base_qs + i + 32] & 0xf] * q8_base[base_q8 + i + 4];
                    /* High nibble of qs[base_qs + i + 32] -> q8[base_q8 + i + 12] */
                    row_sum += s1 * values1[qs[base_qs + i + 32] >> 4] * q8_base[base_q8 + i + 12];
                    /* Low nibble of qs[base_qs + i + 48] -> q8[base_q8 + i + 20] */
                    row_sum += s2 * values2[qs[base_qs + i + 48] & 0xf] * q8_base[base_q8 + i + 20];
                    /* High nibble of qs[base_qs + i + 48] -> q8[base_q8 + i + 28] */
                    row_sum += s2 * values2[qs[base_qs + i + 48] >> 4] * q8_base[base_q8 + i + 28];
                }

                accf[iy] += (float)row_sum * d_arr[iy] * q8_scale;
            }
        }
    }

    for (int r = 0; r < 4; r++) out[r] = accf[r];
}

/* IQ4_K_R4 x Q8_K ARM NEON GEMM kernel.
 * Tiled: 4 weight rows x 2 activation rows per tile.
 * Uses scalar inner loop (R4 layout makes SIMD difficult). */
int sgemm_iq4_k_r4_q8_k_neon(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth) {
    if (nrows < 4 || ncols < 1 || k % QK_K != 0 || nrows % 4 != 0)
        return 0;

    const int nb = k / QK_K;

    int64_t ytiles = nrows / 4;
    int64_t xtiles = ncols / 2;
    int64_t n_tail = ncols - xtiles * 2;
    int64_t xtiles_ext = xtiles + (n_tail > 0 ? 1 : 0);
    int64_t tiles = ytiles * xtiles_ext;
    if (tiles <= 0) return 0;

    int64_t duty = (tiles + nth - 1) / nth;
    int64_t start = duty * ith;
    int64_t end = start + duty;
    if (end > tiles) end = tiles;

    size_t w_block_bytes = nb * sizeof(block_iq4_k_r4);
    size_t a_row_bytes = nb * sizeof(block_q8_K);

    /* LUT tables */
    const int8_t lut0[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };
    const int8_t lut1[16] = {
        -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
    };

    for (int64_t job = start; job < end; job++) {
        int64_t ii = (job / xtiles_ext) * 4;
        int64_t xt = job % xtiles_ext;
        int64_t jj = xt * 2;
        int64_t ncols_tile = (xt < xtiles) ? 2 : n_tail;
        if (ncols_tile < 1) ncols_tile = 1;

        const block_iq4_k_r4 *iq4 = (const block_iq4_k_r4 *)
            ((const char *)vx + (ii / 4) * w_block_bytes);

        float acc[4][2] = { {{0}} };

        const block_q8_K *qk_ptr[2];
        for (int c = 0; c < ncols_tile; c++) {
            qk_ptr[c] = (const block_q8_K *)
                ((const char *)vy + (jj + c) * a_row_bytes);
        }

        for (int ibl = 0; ibl < nb; ibl++) {
            float d_arr[4];
            for (int r = 0; r < 4; r++)
                d_arr[r] = fp16_to_fp32_lookup(iq4[ibl].d[r]);
            const uint8_t *scales_l = iq4[ibl].scales_l;
            const uint8_t *scales_h = iq4[ibl].scales_h;
            const uint8_t *extra = iq4[ibl].extra;
            const uint8_t *qs_base = iq4[ibl].qs;

            for (int c = 0; c < ncols_tile; c++) {
                float q8_scale = qk_ptr[c][ibl].d;
                const int8_t *q8_base = qk_ptr[c][ibl].qs;

                for (int iy = 0; iy < 4; iy++) {
                    int32_t sumi = 0;
                    for (int ib = 0; ib < QK_K / 32; ib++) {
                        int s1 = iq4_k_r4_scale(scales_l, scales_h, 8 * ib + iy);
                        int s2 = iq4_k_r4_scale(scales_l, scales_h, 8 * ib + iy + 4);

                        const int8_t *values1 = (extra[iy] & (1 << ib)) ? lut1 : lut0;
                        const int8_t *values2 = (extra[iy + 4] & (1 << ib)) ? lut1 : lut0;

                        int base_qs = 64 * ib + 4 * iy;
                        int base_q8 = 32 * ib;
                        for (int i = 0; i < 4; i++) {
                            sumi += s1 * values1[qs_base[base_qs + i] & 0xf] * q8_base[base_q8 + i + 0];
                            sumi += s1 * values1[qs_base[base_qs + i] >> 4] * q8_base[base_q8 + i + 8];
                            sumi += s2 * values2[qs_base[base_qs + i + 16] & 0xf] * q8_base[base_q8 + i + 16];
                            sumi += s2 * values2[qs_base[base_qs + i + 16] >> 4] * q8_base[base_q8 + i + 24];
                            sumi += s1 * values1[qs_base[base_qs + i + 32] & 0xf] * q8_base[base_q8 + i + 4];
                            sumi += s1 * values1[qs_base[base_qs + i + 32] >> 4] * q8_base[base_q8 + i + 12];
                            sumi += s2 * values2[qs_base[base_qs + i + 48] & 0xf] * q8_base[base_q8 + i + 20];
                            sumi += s2 * values2[qs_base[base_qs + i + 48] >> 4] * q8_base[base_q8 + i + 28];
                        }
                    }
                    acc[iy][c] += (float)sumi * d_arr[iy] * q8_scale;
                }
            }
        }

        /* Store results */
        for (int c = 0; c < ncols_tile; c++) {
            for (int r = 0; r < 4; r++) {
                out[ii + r + (jj + c) * bs] = acc[r][c];
            }
        }
    }

    return nrows;
}

/* NEON R4 function declarations for quant.c dispatcher */
extern void vec_dot_iq4_k_r4_q8_k_neon(const void *vx, const void *wy, int n, float *out, int nrows);
extern int sgemm_iq4_k_r4_q8_k_neon(int nrows, int ncols, int k,
                                     const void *vx, const void *vy,
                                     float *out, size_t bs, int ith, int nth);
#endif /* ARM_NEON */
