/* ================================================================
 * IQ3_K_R4 x Q8_K AVX2/AVX-512 GEMV and GEMM kernels
 * ================================================================
 * Port of llama.cpp ik branch iqk_gemm_iqk_quants.cpp:
 *   mul_mat_iq3_k_r4_q8_k
 *
 * Weights: block_iq3_k_r4 (GGUF type 338)
 * Activations: block_q8_K (one per row)
 *
 * Algorithm: LUT-based dequantization via iq3nl_values table (16 entries),
 * 3-bit + high-bit extraction, split magnitude+sign scale encoding,
 * then SIGNED*SIGNED dot product with Q8_K activations.
 *
 * Sign trick: dpbusd/maddubs do UNSIGNED*SIGNED. For signed
 * multiplication, we use |qx| for the first operand and
 * sign_extend(qx, y) for the second:
 *   s = sign_epi8(qx, qx)     -> |qx|
 *   sy = sign_epi8(y_shuf, qx) -> y with sign of qx
 *   dpbusd(s, sy) = |qx| * (y * sign(qx)) = qx * y
 *
 * Scale encoding (CRITICAL): IQ3_K uses split magnitude + sign:
 *   scales_l: 4-bit magnitude, processed as (mag & 0xf) * 2 + 1
 *   scales_h: 1 bit per scale (packed), 1 = positive, 0 = negative
 *   Final scale = sign(scales_l_processed, scales_h_sign)
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

static inline __m256i load_scales_avx2(const uint64_t *stored_scales, int ib) {
    /* Load 8 bytes from stored_scales[ib]. Sign-extend to 8 int16,
     * then extract the low 4 int16 (rows 0..3) and extend to 4 int32. */
    __m128i s128 = _mm_loadl_epi64((const __m128i *)(stored_scales + ib));
    __m128i s16 = _mm_cvtepi8_epi16(s128);  /* 8 int8 -> 8 int16 */
    __m256i i32 = _mm256_cvtepi16_epi32(s16);  /* low 4 int16 -> 4 int32, upper 4 zero */
    return i32;
}

/* Bias correction for sign trick in IQ2/3/4_K_R4 kernels.
 * Same as in sgemm_iq2_k_r4.c, duplicated here to avoid cross-file dependencies. */
void vec_dot_iq3_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
#if defined(__AVX2__) && defined(__F16C__)
    assert(nrows == 4);
    assert(n % QK_K == 0);

    const block_iq3_k_r4 *iq3 = (const block_iq3_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    /* iq3nl_values LUT: 16 entries (2 tables of 8).
     * We need a 32-byte LUT (4 copies of 8 bytes) for shuffle addressing.
     * Layout: [table0_0..7, table0_0..7, table1_0..7, table1_0..7]
     * Actually we load all 16, then broadcast to 32 bytes.
     * The shift is 0..7 (3-bit index) + 0 or 8 (from extra bits).
     * So we need indices 0..15 accessible. We store as 32 bytes with
     * each table repeated twice for alignment. */
    /* Full 16-entry combined table (table0 || table1), BROADCAST to both
     * 128-bit lanes -- matches ik_llama.cpp's
     *   values = MM256_SET1_M128I(values128)  // broadcastsi128_si256
     * NOT [table0,table0,table1,table1] (grouping same table per lane),
     * which is wrong: _mm256_shuffle_epi8 never crosses the 128-bit lane
     * boundary, so an index of 8..15 must resolve to table1 *within the
     * same lane* as whichever lane holds that qx byte. */
    static const int8_t kvalues_iq3nl[32] = {
        -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
        -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
    };
    const __m256i values = _mm256_loadu_si256((const __m256i *)kvalues_iq3nl);
    const __m256i shift_shuffle = _mm256_set_epi64x(
        0x0707070706060606ULL, 0x0505050504040404ULL,
        0x0303030302020202ULL, 0x0101010100000000ULL);
    const __m256i m4 = _mm256_set1_epi8(0xf);
    const __m256i ms = _mm256_set1_epi8(8);
    const __m256i m03 = _mm256_set1_epi8(0x03);
    const __m256i m04 = _mm256_set1_epi8(0x04);
#ifndef __AVX512VNNI__
    const __m256i m16 = _mm256_set1_epi16(1);
#endif

    /* smask for sign extraction from scales_h */
    const __m256i smask = _mm256_set_epi64x(
        0x0808080808080808ULL, 0x0404040404040404ULL,
        0x0202020202020202ULL, 0x0101010101010101ULL);

    __m256 acc = _mm256_setzero_ps();
    uint64_t stored_scales[8];

    for (int ibl = 0; ibl < nb; ibl++) {
        __m128 dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq3[ibl].d));
        __m256 d4 = _mm256_set_m128(dl, dl);
        float q8_scale = qk[ibl].d;
        __m256 d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8_scale));

        /* Extra bits for LUT table selection */
        __m256i extra = _mm256_set1_epi64x(*(const uint64_t *)iq3[ibl].extra);

        /* Scale extraction: split magnitude + sign
         * scales_l: 4-bit magnitude -> (mag & 0xf) * 2 + 1
         * scales_h: 1 bit per scale -> sign (1=pos, 0=neg) */
        __m256i slbits = _mm256_loadu_si256((const __m256i *)iq3[ibl].scales_l);
        __m256i sl1 = _mm256_add_epi8(_mm256_slli_epi16(_mm256_and_si256(slbits, m4), 1), _mm256_set1_epi8(1));
        __m256i sl2 = _mm256_add_epi8(_mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4), 1), _mm256_set1_epi8(1));

        __m256i sh = _mm256_set1_epi64x(((const uint64_t *)iq3[ibl].scales_h)[0]);
        __m256i sh1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sh, smask), smask), _mm256_set1_epi8(1));
        __m256i sh2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_srli_epi16(sh, 4), smask), smask), _mm256_set1_epi8(1));

        __m256i i8scales1 = _mm256_sign_epi8(sl1, sh1);
        __m256i i8scales2 = _mm256_sign_epi8(sl2, sh2);
        _mm256_storeu_si256((__m256i *)stored_scales + 0, i8scales1);
        _mm256_storeu_si256((__m256i *)stored_scales + 1, i8scales2);

        /* Bias correction for sign trick: DISABLED - implementation incorrect */
        __m256i isum = _mm256_setzero_si256();

        for (int ib = 0; ib < QK_K / 32; ib++) {
            __m256i scales = load_scales_avx2(stored_scales, ib);

            /* 3-bit + high-bit extraction:
             * qs: 3-bit values (4 per byte)
             * qh: 1 high bit per value (8 per byte)
             * Combined: (qs_bits | (qh_bit << 2)) -> 4-bit LUT index */
            __m256i lb = _mm256_loadu_si256((const __m256i *)iq3[ibl].qs + ib);
            __m128i hbits = _mm_loadu_si128((const __m128i *)iq3[ibl].qh + ib);

            /* Replicate hbits to 256 bits: [hbits, hbits<<4]
             * This gives us the high bits for all 4 qx registers */
            __m256i hb = _mm256_set_m128i(hbits, _mm_slli_epi16(hbits, 4));

            __m256i shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra, 3));
            extra = _mm256_srli_epi16(extra, 1);
            shift = _mm256_shuffle_epi8(shift, shift_shuffle);

            __m256i qx[4];
            qx[0] = _mm256_or_si256(_mm256_and_si256(lb, m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 2)));
            qx[1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 2), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 3)));
            qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 4), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 4)));
            qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 6), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 5)));

            qx[0] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[0], shift));
            qx[1] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[1], shift));
            qx[2] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[2], shift));
            qx[3] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[3], shift));

            __m256i y_reg = _mm256_loadu_si256((const __m256i *)qk[ibl].qs + ib);

            /* Sign trick: |qx| for unsigned multiply, sign(qx) applied to y */
            __m256i s0 = _mm256_sign_epi8(qx[0], qx[0]);
            __m256i s1 = _mm256_sign_epi8(qx[1], qx[1]);
            __m256i s2 = _mm256_sign_epi8(qx[2], qx[2]);
            __m256i s3 = _mm256_sign_epi8(qx[3], qx[3]);

#ifdef __AVX512VNNI__
            __m256i sumi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x00), qx[0]));
            sumi = _mm256_dpbusd_epi32(sumi, s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x55), qx[1]));
            sumi = _mm256_dpbusd_epi32(sumi, s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xaa), qx[2]));
            sumi = _mm256_dpbusd_epi32(sumi, s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xff), qx[3]));
#else
            __m256i t1 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x00), qx[0])));
            __m256i t2 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x55), qx[1])));
            __m256i t3 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xaa), qx[2])));
            __m256i t4 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xff), qx[3])));
            __m256i sumi = _mm256_add_epi32(_mm256_add_epi32(t1, t2), _mm256_add_epi32(t3, t4));
#endif

            isum = _mm256_add_epi32(isum, _mm256_mullo_epi32(scales, sumi));
        }

        acc = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum), acc);
    }

    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    _mm_storeu_ps(out, sum);
#else
    (void)vx; (void)wy; (void)n; (void)out; (void)nrows;
    memset(out, 0, nrows * sizeof(float));
#endif
}

int sgemm_iq3_k_r4_q8_k_avx2(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth) {
#if defined(__AVX2__) && defined(__F16C__)
    if (nrows < 4 || ncols < 1 || k % QK_K != 0 || nrows % 4 != 0)
        return 0;

    const int nb = k / QK_K;

    /* Full 16-entry combined table (table0 || table1), BROADCAST to both
     * 128-bit lanes -- same rationale as vec_dot above. */
    static const int8_t kvalues_iq3nl[32] = {
        -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
        -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
    };
    const __m256i values = _mm256_loadu_si256((const __m256i *)kvalues_iq3nl);
    const __m256i shift_shuffle = _mm256_set_epi64x(
        0x0707070706060606ULL, 0x0505050504040404ULL,
        0x0303030302020202ULL, 0x0101010100000000ULL);
    const __m256i m4 = _mm256_set1_epi8(0xf);
    const __m256i ms = _mm256_set1_epi8(8);
    const __m256i m03 = _mm256_set1_epi8(0x03);
    const __m256i m04 = _mm256_set1_epi8(0x04);
#ifndef __AVX512VNNI__
    const __m256i m16 = _mm256_set1_epi16(1);
#endif

    const __m256i smask = _mm256_set_epi64x(
        0x0808080808080808ULL, 0x0404040404040404ULL,
        0x0202020202020202ULL, 0x0101010101010101ULL);

    {
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

        size_t w_block_bytes = nb * sizeof(block_iq3_k_r4);
        size_t a_row_bytes = nb * sizeof(block_q8_K);

        for (int64_t job = start; job < end; job++) {
            int64_t ii = (job / xtiles_ext) * 4;
            int64_t xt = job % xtiles_ext;
            int64_t jj = xt * 2;
            int64_t ncols_tile = (xt < xtiles) ? 2 : n_tail;
            if (ncols_tile < 1) ncols_tile = 1;

            const block_iq3_k_r4 *iq3 = (const block_iq3_k_r4 *)
                ((const char *)vx + (ii / 4) * w_block_bytes);

            __m256 acc[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };

            const block_q8_K *qk_ptr[2];
            for (int c = 0; c < ncols_tile; c++) {
                qk_ptr[c] = (const block_q8_K *)
                    ((const char *)vy + (jj + c) * a_row_bytes);
            }

            for (int ibl = 0; ibl < nb; ibl++) {
                __m128 dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq3[ibl].d));
                __m256 d4 = _mm256_set_m128(dl, dl);
                __m256i extra = _mm256_set1_epi64x(*(const uint64_t *)iq3[ibl].extra);

                /* Scale extraction: split magnitude + sign */
                __m256i slbits = _mm256_loadu_si256((const __m256i *)iq3[ibl].scales_l);
                __m256i sl1 = _mm256_add_epi8(_mm256_slli_epi16(_mm256_and_si256(slbits, m4), 1), _mm256_set1_epi8(1));
                __m256i sl2 = _mm256_add_epi8(_mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4), 1), _mm256_set1_epi8(1));

                __m256i sh = _mm256_set1_epi64x(((const uint64_t *)iq3[ibl].scales_h)[0]);
                __m256i sh1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sh, smask), smask), _mm256_set1_epi8(1));
                __m256i sh2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_srli_epi16(sh, 4), smask), smask), _mm256_set1_epi8(1));

                __m256i i8scales1 = _mm256_sign_epi8(sl1, sh1);
                __m256i i8scales2 = _mm256_sign_epi8(sl2, sh2);
                uint64_t stored_scales[8];
                _mm256_storeu_si256((__m256i *)stored_scales + 0, i8scales1);
                _mm256_storeu_si256((__m256i *)stored_scales + 1, i8scales2);

                for (int c = 0; c < ncols_tile; c++) {
                    float q8_scale = qk_ptr[c][ibl].d;
                    __m256 d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8_scale));
                    /* Bias correction for sign trick: DISABLED */
                    __m256i isum = _mm256_setzero_si256();

                    __m256i extra_c = extra;

                    for (int ib = 0; ib < QK_K / 32; ib++) {
                        __m256i scales = load_scales_avx2(stored_scales, ib);

                        __m256i lb = _mm256_loadu_si256((const __m256i *)iq3[ibl].qs + ib);
                        __m128i hbits = _mm_loadu_si128((const __m128i *)iq3[ibl].qh + ib);
                        __m256i hb = _mm256_set_m128i(hbits, _mm_slli_epi16(hbits, 4));

                        __m256i shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra_c, 3));
                        extra_c = _mm256_srli_epi16(extra_c, 1);
                        shift = _mm256_shuffle_epi8(shift, shift_shuffle);

                        __m256i qx[4];
                        qx[0] = _mm256_or_si256(_mm256_and_si256(lb, m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 2)));
                        qx[1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 2), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 3)));
                        qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 4), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 4)));
                        qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 6), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 5)));

                        qx[0] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[0], shift));
                        qx[1] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[1], shift));
                        qx[2] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[2], shift));
                        qx[3] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[3], shift));

                        __m256i y_reg = _mm256_loadu_si256((const __m256i *)qk_ptr[c][ibl].qs + ib);

                        __m256i s0 = _mm256_sign_epi8(qx[0], qx[0]);
                        __m256i s1 = _mm256_sign_epi8(qx[1], qx[1]);
                        __m256i s2 = _mm256_sign_epi8(qx[2], qx[2]);
                        __m256i s3 = _mm256_sign_epi8(qx[3], qx[3]);

#ifdef __AVX512VNNI__
                        __m256i sumi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x00), qx[0]));
                        sumi = _mm256_dpbusd_epi32(sumi, s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x55), qx[1]));
                        sumi = _mm256_dpbusd_epi32(sumi, s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xaa), qx[2]));
                        sumi = _mm256_dpbusd_epi32(sumi, s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xff), qx[3]));
#else
                        __m256i t1 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x00), qx[0])));
                        __m256i t2 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0x55), qx[1])));
                        __m256i t3 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xaa), qx[2])));
                        __m256i t4 = _mm256_madd_epi16(m16, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y_reg, 0xff), qx[3])));
                        __m256i sumi = _mm256_add_epi32(_mm256_add_epi32(t1, t2), _mm256_add_epi32(t3, t4));
#endif

                        isum = _mm256_add_epi32(isum, _mm256_mullo_epi32(scales, sumi));
                    }

                    acc[c] = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum), acc[c]);
                }
            }

            for (int c = 0; c < ncols_tile; c++) {
                __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc[c]),
                                        _mm256_extractf128_ps(acc[c], 1));
                float *out_c = out + ii + (jj + c) * bs;
                _mm_storeu_ps(out_c, sum);
            }
        }
    }
    return nrows;
#else
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy;
    (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
#endif
}

#else
/* Non-AVX2 stubs for cross-compilation compatibility */
void vec_dot_iq3_k_r4_q8_k_avx2(const void *vx, const void *wy, int n, float *out, int nrows) { (void)vx; (void)wy; (void)n; (void)out; (void)nrows; }
int sgemm_iq3_k_r4_q8_k_avx2(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)vx; (void)vy; (void)nrows; (void)ncols; (void)k; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
#endif

#if defined(PICOLM_NEON)
#include <arm_neon.h>

static inline int8_t iq3_k_r4_scale_l(const uint8_t *scales_l, int idx) {
    return (int8_t)((scales_l[idx % 32] >> (4 * (idx / 32))) & 0xf);
}

static inline int iq3_k_r4_scale_sign(const uint8_t *scales_h, int idx) {
    return ((scales_h[idx % 8] >> (idx / 8)) & 1) ? 1 : -1;
}

void vec_dot_iq3_k_r4_q8_k_neon(const void *vx, const void *vy, int n,
                                  float *out, int nrows) {
    if (nrows != 4 || n % QK_K != 0) {
        const block_iq3_k_r4 *b = (const block_iq3_k_r4 *)vx;
        const block_q8_K *qk = (const block_q8_K *)vy;
        const int nb = n / QK_K;
        for (int r = 0; r < nrows && r < 4; r++) {
            float w_tmp[QK_K];
            float sum = 0.0f;
            for (int ibl = 0; ibl < nb; ibl++) {
                dequantize_row_iq3_k_r4_single(&b[ibl], w_tmp, QK_K, r);
                float q8_scale = qk[ibl].d;
                for (int i = 0; i < QK_K; i++)
                    sum += w_tmp[i] * (float)qk[ibl].qs[i] * q8_scale;
            }
            out[r] = sum;
        }
        return;
    }

    const block_iq3_k_r4 *iq3 = (const block_iq3_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)vy;
    const int nb = n / QK_K;

    float accf[4] = {0, 0, 0, 0};
    float d_arr[4];

    for (int ibl = 0; ibl < nb; ibl++) {
        for (int r = 0; r < 4; r++)
            d_arr[r] = fp16_to_fp32_lookup(iq3[ibl].d[r]);
        float q8_scale = qk[ibl].d;
        const int8_t *q8_base = qk[ibl].qs;
        const uint8_t *scales_l = iq3[ibl].scales_l;
        const uint8_t *scales_h = iq3[ibl].scales_h;
        const uint8_t *extra = iq3[ibl].extra;
        const uint8_t *ql_base = iq3[ibl].qs;
        const uint8_t *qh_base = iq3[ibl].qh;

        for (int ib = 0; ib < QK_K / 32; ib++) {
            const uint8_t *ql = ql_base + ib * 32;
            const uint8_t *qh = qh_base + ib * 16;

            for (int iy = 0; iy < 4; iy++) {
                int is1 = 8 * ib + iy;
                int is2 = is1 + 4;
                int8_t mag1 = iq3_k_r4_scale_l(scales_l, is1);
                int8_t mag2 = iq3_k_r4_scale_l(scales_l, is2);
                int sign1 = iq3_k_r4_scale_sign(scales_h, is1);
                int sign2 = iq3_k_r4_scale_sign(scales_h, is2);
                int s1 = (2 * mag1 + 1) * sign1;
                int s2 = (2 * mag2 + 1) * sign2;

                const int8_t *values1 = iq3nl_values + (extra[iy + 0] & (1 << ib) ? 8 : 0);
                const int8_t *values2 = iq3nl_values + (extra[iy + 4] & (1 << ib) ? 8 : 0);

                int32_t row_sum = 0;

                for (int i = 0; i < 4; i++) {
                    uint8_t b0 = ql[4 * iy + i];
                    uint8_t h = qh[4 * iy + i];
                    int base = 32 * ib + i;

                    row_sum += s1 * values1[((b0 >> 0) & 0x03) | ((h << 2) & 4)] * q8_base[base +  0];
                    row_sum += s1 * values1[((b0 >> 2) & 0x03) | ((h << 1) & 4)] * q8_base[base +  4];
                    row_sum += s1 * values1[((b0 >> 4) & 0x03) | ((h << 0) & 4)] * q8_base[base +  8];
                    row_sum += s1 * values1[((b0 >> 6) & 0x03) | ((h >> 1) & 4)] * q8_base[base + 12];

                    uint8_t b1 = ql[4 * iy + i + 16];
                    row_sum += s2 * values2[((b1 >> 0) & 0x03) | ((h >> 2) & 4)] * q8_base[base + 16];
                    row_sum += s2 * values2[((b1 >> 2) & 0x03) | ((h >> 3) & 4)] * q8_base[base + 20];
                    row_sum += s2 * values2[((b1 >> 4) & 0x03) | ((h >> 4) & 4)] * q8_base[base + 24];
                    row_sum += s2 * values2[((b1 >> 6) & 0x03) | ((h >> 5) & 4)] * q8_base[base + 28];
                }

                accf[iy] += (float)row_sum * d_arr[iy] * q8_scale;
            }
        }
    }

    for (int r = 0; r < 4; r++) out[r] = accf[r];
}
#endif /* PICOLM_NEON */
