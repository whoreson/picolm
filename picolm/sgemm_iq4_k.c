#include <math.h>
#include <assert.h>
/* ================================================================
 * IQ4_K (plain, GGUF type 139) x Q8_K AVX2 GEMV kernel
 * IQ4_K_R4 (GGUF type 339) x Q8_K AVX2 GEMV + GEMM kernels
 * ================================================================
 * Port of llama.cpp iqk_quantize.cpp dequantization logic
 * adapted for single-row (non-interleaved IQ4_K) and
 * 4-row interleaved (IQ4_K_R4) formats.
 *
 * Weights: block_iq4_k (144 bytes/block) or block_iq4_k_r4 (576 bytes/block)
 * Activations: block_q8_K
 *
 * IQ4_K_R4 layout per block (576 bytes = 4 rows x 144):
 *   d[4]:        FP16 global scales, one per row (8 bytes)
 *   extra[8]:    LUT table select bits (dl1: bits 0-3 = rows 0-3,
 *                dl2: bits 4-7 = rows 0-3); bit ib for subblock ib
 *   scales_h[16]: 2-bit extensions for 64 scales
 *   scales_l[32]: 4-bit magnitudes for 64 scales
 *   qs[512]:     4-bit values, 4-row interleaved
 *
 * Scale index: is = 8*ib + row + 4*half (half=0 -> dl1, half=1 -> dl2)
 *   byte = scales_l[is % 32], nibble = (is / 32) ? high : low
 *   hbyte = scales_h[is % 16], shift = 2 * (is / 16)
 *   scale = ((nibble | ((hbyte >> shift) & 3) << 4)) - 32
 *
 * qs layout per subblock (64 bytes):
 *   bytes  0-15: [r0(4B), r1(4B), r2(4B), r3(4B)] = dl1 lo nibbles
 *   bytes 16-31: [r0(4B), r1(4B), r2(4B), r3(4B)] = dl2 lo nibbles
 *   bytes 32-47: [r0(4B), r1(4B), r2(4B), r3(4B)] = dl1 hi nibbles
 *   bytes 48-63: [r0(4B), r1(4B), r2(4B), r3(4B)] = dl2 hi nibbles
 *
 * Per row r, subblock ib (output positions within the 32-value group):
 *   dst[0..3]   = LUT(lo(qs[64*ib + 4r + 0..3]))    * dl1
 *   dst[4..7]   = LUT(lo(qs[64*ib + 4r + 32..35]))  * dl1
 *   dst[8..11]  = LUT(hi(qs[64*ib + 4r + 0..3]))    * dl1
 *   dst[12..15] = LUT(hi(qs[64*ib + 4r + 32..35]))  * dl1
 *   dst[16..19] = LUT(lo(qs[64*ib + 4r + 16..19]))  * dl2
 *   dst[20..23] = LUT(lo(qs[64*ib + 4r + 48..51]))  * dl2
 *   dst[24..27] = LUT(hi(qs[64*ib + 4r + 16..19]))  * dl2
 *   dst[28..31] = LUT(hi(qs[64*ib + 4r + 48..51]))  * dl2
 * ================================================================ */

#include <stdint.h>
#include <string.h>
#include <math.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "quant.h"

#define QK_K 256

/* ================================================================
 * IQ4_K (plain, single-row) x Q8_K AVX2 GEMV
 * ================================================================ */
void vec_dot_iq4_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
#if defined(__AVX2__) && defined(__F16C__)
    const block_iq4_k *iq4 = (const block_iq4_k *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    static const int8_t iq4k_values_l[32] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
    };

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
            const uint8_t sh = iq4[ibl].scales_h[ib / 2] >> (4 * (ib % 2));
            int scale_lo = (int)(((iq4[ibl].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
            int scale_hi = (int)(((iq4[ibl].scales_l[ib] >> 4) | ((sh << 2) & 0x30)) - 32);

            int use_table1_lo = extra & 1;
            int use_table1_hi = extra & 2;
            extra >>= 2;

            __m128i qs_vec = _mm_loadu_si128((const __m128i *)qs);

            __m128i qx_lo = _mm_and_si128(qs_vec, _mm_set1_epi8(0x0f));
            __m128i y_lo = _mm_shuffle_epi8(
                _mm_loadu_si128((const __m128i *)(use_table1_lo ? iq4k_values_l + 16 : iq4k_values_l)),
                qx_lo);
            __m128i qx_hi = _mm_and_si128(_mm_srli_epi16(qs_vec, 4), _mm_set1_epi8(0x0f));
            __m128i y_hi = _mm_shuffle_epi8(
                _mm_loadu_si128((const __m128i *)(use_table1_hi ? iq4k_values_l + 16 : iq4k_values_l)),
                qx_hi);

            __m128i abs_qx_lo = _mm_sign_epi8(qx_lo, qx_lo);
            __m128i abs_qx_hi = _mm_sign_epi8(qx_hi, qx_hi);
            __m128i sy_lo = _mm_sign_epi8(y_lo, qx_lo);
            __m128i sy_hi = _mm_sign_epi8(y_hi, qx_hi);

            __m128i sum16_lo = _mm_maddubs_epi16(abs_qx_lo, sy_lo);
            __m128i sum16_hi = _mm_maddubs_epi16(abs_qx_hi, sy_hi);

            __m128i scale_lo_v = _mm_set1_epi16(scale_lo);
            __m128i scale_hi_v = _mm_set1_epi16(scale_hi);
            __m128i prod_lo = _mm_madd_epi16(sum16_lo, scale_lo_v);
            __m128i prod_hi = _mm_madd_epi16(sum16_hi, scale_hi_v);

            isum = _mm256_add_epi32(isum, _mm256_set_m128i(prod_hi, prod_lo));

            qs += 16;
            q8 += 32;
        }

        int isum_scalar = _mm256_extract_epi32(isum, 0) + _mm256_extract_epi32(isum, 1)
                        + _mm256_extract_epi32(isum, 2) + _mm256_extract_epi32(isum, 3)
                        + _mm256_extract_epi32(isum, 4) + _mm256_extract_epi32(isum, 5)
                        + _mm256_extract_epi32(isum, 6) + _mm256_extract_epi32(isum, 7);
        result += (float)isum_scalar * dy;
    }

    *out = result;
#else
    (void)vx; (void)wy; (void)n;
    *out = 0.0f;
#endif
}
/* ================================================================

 * IQ4_K_R4 x Q8_K AVX2 GEMV kernel (4-row interleaved)
 * GGUF type 339
 * ================================================================
 * Processes 4 rows simultaneously, producing 4 dot products.
 * Uses blendv for LUT table selection (correct but not optimal).
 * ================================================================ */
void vec_dot_iq4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
#if defined(__AVX2__) && defined(__F16C__)
    if (nrows != 4 || n % QK_K != 0) {
        /* Fallback: dequantize + scalar dot */
        const block_iq4_k_r4 *b = (const block_iq4_k_r4 *)vx;
        const block_q8_K *qk = (const block_q8_K *)wy;
        const int nb = n / QK_K;
        for (int r = 0; r < nrows && r < 4; r++) {
            float w_tmp[QK_K];
            float sum = 0.0f;
            for (int ibl = 0; ibl < nb; ibl++) {
                dequantize_row_iq4_k_r4_single(&b[ibl], w_tmp, QK_K, r);
                float q8_scale = qk[ibl].d;
                for (int i = 0; i < QK_K; i++) {
                    sum += w_tmp[i] * (float)qk[ibl].qs[i] * q8_scale;
                }
            }
            out[r] = sum;
        }
        return;
    }

    const block_iq4_k_r4 *blocks = (const block_iq4_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    static const int8_t kvalues_iq4k[32] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
    };
    const __m256i lut_t0 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)kvalues_iq4k));
    const __m256i lut_t1 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)(kvalues_iq4k + 16)));

    const __m256i m4  = _mm256_set1_epi8(0xf);
    const __m256i m16 = _mm256_set1_epi16(1);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    const __m256i m32 = _mm256_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int ibl = 0; ibl < nb; ibl++) {
        const block_iq4_k_r4 *b = &blocks[ibl];

        __m128 dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)b->d));
        __m256 d4 = _mm256_set_m128(dl, dl);

        float q8_scale = qk[ibl].d;
        __m256 d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8_scale));

        /* Scale extraction: 64 int8 scales total, 8 per subblock
         * stored[ib*8 .. ib*8+7] = [dl1_r0..r3, dl2_r0..r3] for subblock ib */
        __m256i slbits = _mm256_loadu_si256((const __m256i *)b->scales_l);
        __m256i sl1 = _mm256_and_si256(slbits, m4);
        __m256i sl2 = _mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4);
        __m128i shbits = _mm_loadu_si128((const __m128i*)b->scales_h);
        __m256i sh = _mm256_set_m128i(_mm_srli_epi16(shbits, 2), shbits);
        __m256i i8s1 = _mm256_sub_epi8(
            _mm256_or_si256(sl1, _mm256_and_si256(m30, _mm256_slli_epi16(sh, 4))), m32);
        __m256i i8s2 = _mm256_sub_epi8(
            _mm256_or_si256(sl2, _mm256_and_si256(m30, sh)), m32);

        int8_t stored[64] __attribute__((aligned(32)));
        _mm256_storeu_si256((__m256i *)stored, i8s1);
        _mm256_storeu_si256((__m256i *)(stored + 32), i8s2);

        __m256i isum = _mm256_setzero_si256();

        for (int ib = 0; ib < QK_K / 32; ib++) {
            /* Load 8 int8 scales for this subblock, sign-extend to int32 */
            __m128i s8 = _mm_loadl_epi64((const __m128i *)(stored + ib*8));
            __m256i scales_i32 = _mm256_cvtepi8_epi32(s8);
            /* scales_i32 = [dl1_r0, dl1_r1, dl1_r2, dl1_r3, dl2_r0, dl2_r1, dl2_r2, dl2_r3] */

            /* Blend mask: low lane bytes 0-3=row0 dl1, 4-7=row1 dl1, etc.
             *              high lane bytes 0-3=row0 dl2, 4-7=row1 dl2, etc. */
            uint8_t e0 = (b->extra[0] & (1 << ib)) ? 0xFF : 0x00;
            uint8_t e1 = (b->extra[1] & (1 << ib)) ? 0xFF : 0x00;
            uint8_t e2 = (b->extra[2] & (1 << ib)) ? 0xFF : 0x00;
            uint8_t e3 = (b->extra[3] & (1 << ib)) ? 0xFF : 0x00;
            uint8_t e4 = (b->extra[4] & (1 << ib)) ? 0xFF : 0x00;
            uint8_t e5 = (b->extra[5] & (1 << ib)) ? 0xFF : 0x00;
            uint8_t e6 = (b->extra[6] & (1 << ib)) ? 0xFF : 0x00;
            uint8_t e7 = (b->extra[7] & (1 << ib)) ? 0xFF : 0x00;
            __m256i blend_mask = _mm256_set_epi8(
                e7, e7, e7, e7, e6, e6, e6, e6, e5, e5, e5, e5, e4, e4, e4, e4,
                e3, e3, e3, e3, e2, e2, e2, e2, e1, e1, e1, e1, e0, e0, e0, e0);

            const uint8_t *qs_base = b->qs + 64*ib;
            const int8_t *q8s = qk[ibl].qs + 32*ib;

            __m256i v0 = _mm256_loadu_si256((const __m256i *)qs_base);
            __m256i v1 = _mm256_loadu_si256((const __m256i *)(qs_base + 32));
            __m256i y_reg = _mm256_loadu_si256((const __m256i *)q8s);

            __m256i v0_lo = _mm256_and_si256(v0, m4);
            __m256i v0_hi = _mm256_and_si256(_mm256_srli_epi16(v0, 4), m4);
            __m256i v1_lo = _mm256_and_si256(v1, m4);
            __m256i v1_hi = _mm256_and_si256(_mm256_srli_epi16(v1, 4), m4);

            /* LUT with per-row table selection via blendv */
            __m256i q0 = _mm256_blendv_epi8(
                _mm256_shuffle_epi8(lut_t0, v0_lo),
                _mm256_shuffle_epi8(lut_t1, v0_lo), blend_mask);
            __m256i q1 = _mm256_blendv_epi8(
                _mm256_shuffle_epi8(lut_t0, v1_lo),
                _mm256_shuffle_epi8(lut_t1, v1_lo), blend_mask);
            __m256i q2 = _mm256_blendv_epi8(
                _mm256_shuffle_epi8(lut_t0, v0_hi),
                _mm256_shuffle_epi8(lut_t1, v0_hi), blend_mask);
            __m256i q3 = _mm256_blendv_epi8(
                _mm256_shuffle_epi8(lut_t0, v1_hi),
                _mm256_shuffle_epi8(lut_t1, v1_hi), blend_mask);

            /* Activation broadcasts (per 128-bit lane):
             * y00: low = q[0..3] bcast,  high = q[16..19] bcast
             * y55: low = q[4..7] bcast,  high = q[20..23] bcast
             * yaa: low = q[8..11] bcast, high = q[24..27] bcast
             * yff: low = q[12..15] bcast, high = q[28..31] bcast */
            __m256i y00 = _mm256_shuffle_epi32(y_reg, 0x00);
            __m256i y55 = _mm256_shuffle_epi32(y_reg, 0x55);
            __m256i yaa = _mm256_shuffle_epi32(y_reg, 0xaa);
            __m256i yff = _mm256_shuffle_epi32(y_reg, 0xff);

            /* Sign trick: |qx| * sign(qx, y) = qx * y */
            __m256i s0 = _mm256_sign_epi8(q0, q0);
            __m256i s1 = _mm256_sign_epi8(q1, q1);
            __m256i s2 = _mm256_sign_epi8(q2, q2);
            __m256i s3 = _mm256_sign_epi8(q3, q3);

            __m256i t0 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(y00, q0)));
            __m256i t1 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(y55, q1)));
            __m256i t2 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(yaa, q2)));
            __m256i t3 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(yff, q3)));

            __m256i sumi = _mm256_add_epi32(_mm256_add_epi32(t0, t1), _mm256_add_epi32(t2, t3));
            isum = _mm256_add_epi32(isum, _mm256_mullo_epi32(scales_i32, sumi));
        }

        /* isum = [dl1_r0..r3 (low lane), dl2_r0..r3 (high lane)] (int32)
         * Combine: total[r] = dl1_total[r] + dl2_total[r] */
        __m128i lo32 = _mm256_castsi256_si128(isum);
        __m128i hi32 = _mm256_extracti128_si256(isum, 1);
        __m128i total = _mm_add_epi32(lo32, hi32);
        __m128 total_f = _mm_cvtepi32_ps(total);
        /* Multiply by d[row] * q8_scale */
        __m128 d4_128 = _mm256_castps256_ps128(d4y);
        __m128 result_128 = _mm_mul_ps(total_f, d4_128);
        acc = _mm256_add_ps(acc, _mm256_set_m128(result_128, result_128));
    }

    /* Extract 4 row results (low 128 bits has 4 floats, same as high) */
    __m128 result = _mm256_castps256_ps128(acc);
    _mm_storeu_ps(out, result);
#else
    (void)vx; (void)wy; (void)n; (void)out; (void)nrows;
#endif
}

/* ================================================================
 * IQ4_K_R4 x Q8_K AVX2 tiled GEMM kernel (4 rows x 2 cols per tile)
 * GGUF type 339
 * ================================================================ */

/* ================================================================
 * IQ4_K_R4 x Q8_K AVX2 tiled GEMM kernel (4 rows x N cols per tile)
 * GGUF type 339
 *
 * Optimized inner loop: weight-dependent LUT lookups hoisted outside
 * the column loop (computed once per subblock, reused for all cols).
 * ================================================================ */

/* ================================================================
 * IQ4_K_R4 x Q8_K AVX2 tiled GEMM kernel (4 rows x N cols per tile)
 * GGUF type 339
 *
 * Optimized inner loop: weight-dependent LUT lookups hoisted outside
 * the column loop (computed once per subblock, reused for all cols).
 * ================================================================ */
int sgemm_iq4_k_r4_q8_k_avx2(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth) {
#if defined(__AVX2__) && defined(__F16C__)
    if (nrows < 4 || ncols < 1 || k % QK_K != 0 || nrows % 4 != 0)
        return 0;

    const int nb = k / QK_K;

    static const int8_t kvalues_iq4k[32] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
    };
    const __m256i lut_t0 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)kvalues_iq4k));
    const __m256i lut_t1 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)(kvalues_iq4k + 16)));

    const __m256i m4  = _mm256_set1_epi8(0xf);
    const __m256i m16 = _mm256_set1_epi16(1);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    const __m256i m32 = _mm256_set1_epi8(32);

    int64_t ytiles = nrows / 4;
    int64_t xtiles = ncols / 4;
    int64_t n_tail = ncols - xtiles * 4;
    int64_t xtiles_ext = xtiles + (n_tail > 0 ? 1 : 0);
    int64_t tiles = ytiles * xtiles_ext;
    if (tiles <= 0) return 0;

    int64_t duty = (tiles + nth - 1) / nth;
    int64_t start = duty * ith;
    int64_t end = start + duty;
    if (end > tiles) end = tiles;

    size_t w_block_bytes = nb * sizeof(block_iq4_k_r4);
    size_t a_row_bytes = nb * sizeof(block_q8_K);

    for (int64_t job = start; job < end; job++) {
        int64_t ii = (job / xtiles_ext) * 4;
        int64_t xt = job % xtiles_ext;
        int64_t jj = xt * 4;
        int64_t ncols_tile = (xt < xtiles) ? 4 : n_tail;
        if (ncols_tile < 1) ncols_tile = 1;

        const block_iq4_k_r4 *iq4 = (const block_iq4_k_r4 *)
            ((const char *)vx + (ii / 4) * w_block_bytes);

        __m128 acc[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };

        const block_q8_K *qk_ptr[4];
        for (int c = 0; c < ncols_tile; c++) {
            qk_ptr[c] = (const block_q8_K *)
                ((const char *)vy + (jj + c) * a_row_bytes);
        }

        for (int ibl = 0; ibl < nb; ibl++) {
            const block_iq4_k_r4 *b = &iq4[ibl];

            __m128 dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)b->d));
            __m256 d4 = _mm256_set_m128(dl, dl);

            /* Scale extraction (per block, hoisted outside column loop) */
            __m256i slbits = _mm256_loadu_si256((const __m256i *)b->scales_l);
            __m256i sl1 = _mm256_and_si256(slbits, m4);
            __m256i sl2 = _mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4);
            __m128i shbits = _mm_loadu_si128((const __m128i*)b->scales_h);
            __m256i sh = _mm256_set_m128i(_mm_srli_epi16(shbits, 2), shbits);
            __m256i i8s1 = _mm256_sub_epi8(
                _mm256_or_si256(sl1, _mm256_and_si256(m30, _mm256_slli_epi16(sh, 4))), m32);
            __m256i i8s2 = _mm256_sub_epi8(
                _mm256_or_si256(sl2, _mm256_and_si256(m30, sh)), m32);

            int8_t stored[64] __attribute__((aligned(32)));
            _mm256_storeu_si256((__m256i *)stored, i8s1);
            _mm256_storeu_si256((__m256i *)(stored + 32), i8s2);

            /* LUT blend masks for all 8 subblocks (hoisted outside column loop) */
            __m256i bmask[8];
            for (int ib = 0; ib < 8; ib++) {
                uint8_t e0 = (b->extra[0] & (1 << ib)) ? 0xFF : 0x00;
                uint8_t e1 = (b->extra[1] & (1 << ib)) ? 0xFF : 0x00;
                uint8_t e2 = (b->extra[2] & (1 << ib)) ? 0xFF : 0x00;
                uint8_t e3 = (b->extra[3] & (1 << ib)) ? 0xFF : 0x00;
                uint8_t e4 = (b->extra[4] & (1 << ib)) ? 0xFF : 0x00;
                uint8_t e5 = (b->extra[5] & (1 << ib)) ? 0xFF : 0x00;
                uint8_t e6 = (b->extra[6] & (1 << ib)) ? 0xFF : 0x00;
                uint8_t e7 = (b->extra[7] & (1 << ib)) ? 0xFF : 0x00;
                bmask[ib] = _mm256_set_epi8(
                    e7, e7, e7, e7, e6, e6, e6, e6,
                    e5, e5, e5, e5, e4, e4, e4, e4,
                    e3, e3, e3, e3, e2, e2, e2, e2,
                    e1, e1, e1, e1, e0, e0, e0, e0);
            }

            __m256i isum_c0 = _mm256_setzero_si256();
            __m256i isum_c1 = _mm256_setzero_si256();
            __m256i isum_c2 = _mm256_setzero_si256();
            __m256i isum_c3 = _mm256_setzero_si256();
            const int8_t *q8_c0 = qk_ptr[0][ibl].qs;
            const int8_t *q8_c1 = (ncols_tile > 1) ? qk_ptr[1][ibl].qs : NULL;
            const int8_t *q8_c2 = (ncols_tile > 2) ? qk_ptr[2][ibl].qs : NULL;
            const int8_t *q8_c3 = (ncols_tile > 3) ? qk_ptr[3][ibl].qs : NULL;

            /* Process each subblock: LUT lookup once, then all columns */
            for (int ib = 0; ib < QK_K / 32; ib++) {
                const uint8_t *qs_base = b->qs + 64*ib;
                __m256i v0 = _mm256_loadu_si256((const __m256i *)qs_base);
                __m256i v1 = _mm256_loadu_si256((const __m256i *)(qs_base + 32));
                __m256i v0_lo = _mm256_and_si256(v0, m4);
                __m256i v0_hi = _mm256_and_si256(_mm256_srli_epi16(v0, 4), m4);
                __m256i v1_lo = _mm256_and_si256(v1, m4);
                __m256i v1_hi = _mm256_and_si256(_mm256_srli_epi16(v1, 4), m4);
                __m256i bm = bmask[ib];
                __m256i q0 = _mm256_blendv_epi8(
                    _mm256_shuffle_epi8(lut_t0, v0_lo),
                    _mm256_shuffle_epi8(lut_t1, v0_lo), bm);
                __m256i q1 = _mm256_blendv_epi8(
                    _mm256_shuffle_epi8(lut_t0, v1_lo),
                    _mm256_shuffle_epi8(lut_t1, v1_lo), bm);
                __m256i q2 = _mm256_blendv_epi8(
                    _mm256_shuffle_epi8(lut_t0, v0_hi),
                    _mm256_shuffle_epi8(lut_t1, v0_hi), bm);
                __m256i q3 = _mm256_blendv_epi8(
                    _mm256_shuffle_epi8(lut_t0, v1_hi),
                    _mm256_shuffle_epi8(lut_t1, v1_hi), bm);

                __m128i s8 = _mm_loadl_epi64((const __m128i *)(stored + ib*8));
                __m256i scales_i32 = _mm256_cvtepi8_epi32(s8);

                __m256i s0 = _mm256_sign_epi8(q0, q0);
                __m256i s1 = _mm256_sign_epi8(q1, q1);
                __m256i s2 = _mm256_sign_epi8(q2, q2);
                __m256i s3 = _mm256_sign_epi8(q3, q3);

                /* Column 0 */
                __m256i y_reg = _mm256_loadu_si256((const __m256i *)(q8_c0 + 32*ib));
                __m256i y00 = _mm256_shuffle_epi32(y_reg, 0x00);
                __m256i y55 = _mm256_shuffle_epi32(y_reg, 0x55);
                __m256i yaa = _mm256_shuffle_epi32(y_reg, 0xaa);
                __m256i yff = _mm256_shuffle_epi32(y_reg, 0xff);
                __m256i t0 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(y00, q0)));
                __m256i t1 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(y55, q1)));
                __m256i t2 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(yaa, q2)));
                __m256i t3 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(yff, q3)));
                __m256i sumi = _mm256_add_epi32(_mm256_add_epi32(t0, t1), _mm256_add_epi32(t2, t3));
                isum_c0 = _mm256_add_epi32(isum_c0, _mm256_mullo_epi32(scales_i32, sumi));

                if (ncols_tile > 1) {
                    /* Column 1 */
                    __m256i y_reg1 = _mm256_loadu_si256((const __m256i *)(q8_c1 + 32*ib));
                    __m256i y01 = _mm256_shuffle_epi32(y_reg1, 0x00);
                    __m256i y51 = _mm256_shuffle_epi32(y_reg1, 0x55);
                    __m256i ya1 = _mm256_shuffle_epi32(y_reg1, 0xaa);
                    __m256i yf1 = _mm256_shuffle_epi32(y_reg1, 0xff);
                    __m256i u0 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(y01, q0)));
                    __m256i u1 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(y51, q1)));
                    __m256i u2 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(ya1, q2)));
                    __m256i u3 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(yf1, q3)));
                    __m256i sumi1 = _mm256_add_epi32(_mm256_add_epi32(u0, u1), _mm256_add_epi32(u2, u3));
                    isum_c1 = _mm256_add_epi32(isum_c1, _mm256_mullo_epi32(scales_i32, sumi1));
                }
                if (ncols_tile > 2) {
                    /* Column 2 */
                    __m256i y_reg2 = _mm256_loadu_si256((const __m256i *)(q8_c2 + 32*ib));
                    __m256i y02 = _mm256_shuffle_epi32(y_reg2, 0x00);
                    __m256i y52 = _mm256_shuffle_epi32(y_reg2, 0x55);
                    __m256i ya2 = _mm256_shuffle_epi32(y_reg2, 0xaa);
                    __m256i yf2 = _mm256_shuffle_epi32(y_reg2, 0xff);
                    __m256i w0 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(y02, q0)));
                    __m256i w1 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(y52, q1)));
                    __m256i w2 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(ya2, q2)));
                    __m256i w3 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(yf2, q3)));
                    __m256i sumi2 = _mm256_add_epi32(_mm256_add_epi32(w0, w1), _mm256_add_epi32(w2, w3));
                    isum_c2 = _mm256_add_epi32(isum_c2, _mm256_mullo_epi32(scales_i32, sumi2));
                }
                if (ncols_tile > 3) {
                    /* Column 3 */
                    __m256i y_reg3 = _mm256_loadu_si256((const __m256i *)(q8_c3 + 32*ib));
                    __m256i y03 = _mm256_shuffle_epi32(y_reg3, 0x00);
                    __m256i y53 = _mm256_shuffle_epi32(y_reg3, 0x55);
                    __m256i ya3 = _mm256_shuffle_epi32(y_reg3, 0xaa);
                    __m256i yf3 = _mm256_shuffle_epi32(y_reg3, 0xff);
                    __m256i x0 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(y03, q0)));
                    __m256i x1 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(y53, q1)));
                    __m256i x2 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(ya3, q2)));
                    __m256i x3 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(yf3, q3)));
                    __m256i sumi3 = _mm256_add_epi32(_mm256_add_epi32(x0, x1), _mm256_add_epi32(x2, x3));
                    isum_c3 = _mm256_add_epi32(isum_c3, _mm256_mullo_epi32(scales_i32, sumi3));
                }
            }

            /* Scale and accumulate per column */
            {
                float q8_scale = qk_ptr[0][ibl].d;
                __m128 d4q8 = _mm_mul_ps(_mm256_castps256_ps128(d4),
                                          _mm_set1_ps(q8_scale));
                __m128i lo32 = _mm256_castsi256_si128(isum_c0);
                __m128i hi32 = _mm256_extracti128_si256(isum_c0, 1);
                __m128i total = _mm_add_epi32(lo32, hi32);
                __m128 result_128 = _mm_mul_ps(_mm_cvtepi32_ps(total), d4q8);
                acc[0] = _mm_add_ps(acc[0], result_128);
            }
            if (ncols_tile > 1) {
                float q8_scale = qk_ptr[1][ibl].d;
                __m128 d4q8 = _mm_mul_ps(_mm256_castps256_ps128(d4),
                                          _mm_set1_ps(q8_scale));
                __m128i lo32 = _mm256_castsi256_si128(isum_c1);
                __m128i hi32 = _mm256_extracti128_si256(isum_c1, 1);
                __m128i total = _mm_add_epi32(lo32, hi32);
                __m128 result_128 = _mm_mul_ps(_mm_cvtepi32_ps(total), d4q8);
                acc[1] = _mm_add_ps(acc[1], result_128);
            }
            if (ncols_tile > 2) {
                float q8_scale = qk_ptr[2][ibl].d;
                __m128 d4q8 = _mm_mul_ps(_mm256_castps256_ps128(d4),
                                          _mm_set1_ps(q8_scale));
                __m128i lo32 = _mm256_castsi256_si128(isum_c2);
                __m128i hi32 = _mm256_extracti128_si256(isum_c2, 1);
                __m128i total = _mm_add_epi32(lo32, hi32);
                __m128 result_128 = _mm_mul_ps(_mm_cvtepi32_ps(total), d4q8);
                acc[2] = _mm_add_ps(acc[2], result_128);
            }
            if (ncols_tile > 3) {
                float q8_scale = qk_ptr[3][ibl].d;
                __m128 d4q8 = _mm_mul_ps(_mm256_castps256_ps128(d4),
                                          _mm_set1_ps(q8_scale));
                __m128i lo32 = _mm256_castsi256_si128(isum_c3);
                __m128i hi32 = _mm256_extracti128_si256(isum_c3, 1);
                __m128i total = _mm_add_epi32(lo32, hi32);
                __m128 result_128 = _mm_mul_ps(_mm_cvtepi32_ps(total), d4q8);
                acc[3] = _mm_add_ps(acc[3], result_128);
            }
        }

        for (int c = 0; c < ncols_tile; c++) {
            float *out_c = out + ii + (jj + c) * bs;
            _mm_storeu_ps(out_c, acc[c]);
        }
    }
    return nrows;
#else
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy;
    (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
#endif
}

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
                    row_sum += s1 * values1[qs_base[base_qs + i] & 0xf] * q8_base[base_q8 + i + 0];
                    /* High nibble of qs[base_qs + i] -> q8[base_q8 + i + 8] */
                    row_sum += s1 * values1[qs_base[base_qs + i] >> 4] * q8_base[base_q8 + i + 8];
                    /* Low nibble of qs[base_qs + i + 16] -> q8[base_q8 + i + 16] */
                    row_sum += s2 * values2[qs_base[base_qs + i + 16] & 0xf] * q8_base[base_q8 + i + 16];
                    /* High nibble of qs[base_qs + i + 16] -> q8[base_q8 + i + 24] */
                    row_sum += s2 * values2[qs_base[base_qs + i + 16] >> 4] * q8_base[base_q8 + i + 24];
                    /* Low nibble of qs[base_qs + i + 32] -> q8[base_q8 + i + 4] */
                    row_sum += s1 * values1[qs_base[base_qs + i + 32] & 0xf] * q8_base[base_q8 + i + 4];
                    /* High nibble of qs[base_qs + i + 32] -> q8[base_q8 + i + 12] */
                    row_sum += s1 * values1[qs_base[base_qs + i + 32] >> 4] * q8_base[base_q8 + i + 12];
                    /* Low nibble of qs[base_qs + i + 48] -> q8[base_q8 + i + 20] */
                    row_sum += s2 * values2[qs_base[base_qs + i + 48] & 0xf] * q8_base[base_q8 + i + 20];
                    /* High nibble of qs[base_qs + i + 48] -> q8[base_q8 + i + 28] */
                    row_sum += s2 * values2[qs_base[base_qs + i + 48] >> 4] * q8_base[base_q8 + i + 28];
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
