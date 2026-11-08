/* sgemm_q4_k_r4.c -- AVX2 kernels for Q4_K_R4 (GGUF type 212)
 *
 * Q4_K_R4: 4-row interleaved Q4_K, 576 bytes per block.
 * Scale layout: 32 scales = 8 subblocks * 4 rows.
 *   scale index: is = 4*ib + row
 *   low nibble of scales_l[is] = d scale, high nibble = m scale.
 */

#include "quant.h"
#include <stdlib.h>
#include <assert.h>

#if defined(PICOLM_AVX2)
#include <immintrin.h>

static inline float get_d_scale(const block_q4_k_r4 *b, int is) {
    uint8_t sl = b->scales_l[is] & 0xf;
    uint8_t sh = (b->scales_h[is % 16] >> (4 * (is / 16))) & 0x03;
    return (float)(sl | (sh << 4));
}

static inline float get_m_scale(const block_q4_k_r4 *b, int is) {
    uint8_t sl = (b->scales_l[is] >> 4) & 0xf;
    uint8_t sh = (b->scales_h[is % 16] >> (4 * (is / 16))) & 0x0c;
    return (float)(sl | (sh << 2));
}

/* Q4_K_R4 x Q8_K, four rows per call (the decode GEMV).
 *
 * Layout (see dequantize_row_q4_k_r4_single): per 32-value sub-block ib, qs[64*ib + 16*kk + 4*row + i]
 * -- a 16-byte slice kk is 4 rows x one dword.  The low nibble of slice kk holds values
 *   {0, 16, 4, 20}[kk] + i  and the high nibble {8, 24, 12, 28}[kk] + i, so each (slice, nibble) pair
 * multiplies ONE contiguous group of 4 activations.  Broadcasting that 4-byte activation group across the
 * four row dwords and running maddubs + madd against the unsigned nibbles gives all four rows' partial
 * sums at once (lane r = row r); per sub-block the eight partials are added, multiplied by the vector of
 * the four rows' integer scales, and accumulated over the super-block.  The min term needs only
 * m * (bsums of the sub-block) per row.  Scales/mins are unpacked as vectors (4 consecutive bytes of
 * scales_l, 4 of scales_h).  One float update per super-block instead of one per (sub-block, row).
 * (Accumulation order differs from the previous kernel, ~1e-7 relative.) */
void vec_dot_q4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
    assert(nrows == 4);
    assert(n % QK_K == 0);

    const block_q4_k_r4 *x = (const block_q4_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    const __m128i m0f = _mm_set1_epi8(0x0f), ones16 = _mm_set1_epi16(1);
    const __m128i m03 = _mm_set1_epi32(0x03), m0c = _mm_set1_epi32(0x0c), mlo4 = _mm_set1_epi32(0x0f);
    /* activation dword offsets (within a 32-value sub-block) for (lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3) */
    static const int yoff[8] = { 0, 8, 16, 24, 4, 12, 20, 28 };

    __m128 outv = _mm_setzero_ps();

    for (int ibl = 0; ibl < nb; ibl++) {
        const block_q4_k_r4 *b = &x[ibl];
        const block_q8_K *q = &qk[ibl];

        __m128i acc = _mm_setzero_si128();    /* sum_ib scale[ib][row] * isum[ib][row]   */
        __m128i mns = _mm_setzero_si128();    /* sum_ib min[ib][row]   * bsum32[ib]      */

        for (int ib = 0; ib < QK_K / 32; ib++) {
            const uint8_t *qs = b->qs + 64 * ib;
            const int8_t *q8 = q->qs + 32 * ib;

            int32_t lw, hw;
            memcpy(&lw, b->scales_l + 4 * ib, 4);
            memcpy(&hw, b->scales_h + 4 * (ib & 3), 4);
            const __m128i l4 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(lw));
            const __m128i h4 = _mm_srli_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(hw)), 4 * (ib >> 2));
            const __m128i sc4 = _mm_or_si128(_mm_and_si128(l4, mlo4), _mm_slli_epi32(_mm_and_si128(h4, m03), 4));
            const __m128i mn4 = _mm_or_si128(_mm_srli_epi32(l4, 4), _mm_slli_epi32(_mm_and_si128(h4, m0c), 2));

            __m128i isum = _mm_setzero_si128();
            for (int kk = 0; kk < 4; kk++) {
                const __m128i c = _mm_loadu_si128((const __m128i *)(qs + 16 * kk));
                int32_t ylo, yhi;
                memcpy(&ylo, q8 + yoff[2 * kk], 4);
                memcpy(&yhi, q8 + yoff[2 * kk + 1], 4);
                const __m128i plo = _mm_maddubs_epi16(_mm_and_si128(c, m0f), _mm_set1_epi32(ylo));
                const __m128i phi = _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(c, 4), m0f), _mm_set1_epi32(yhi));
                isum = _mm_add_epi32(isum, _mm_madd_epi16(_mm_add_epi16(plo, phi), ones16));
            }
            acc = _mm_add_epi32(acc, _mm_mullo_epi32(isum, sc4));
            mns = _mm_add_epi32(mns, _mm_mullo_epi32(mn4, _mm_set1_epi32((int)q->bsums[2 * ib] + (int)q->bsums[2 * ib + 1])));
        }

        const __m128 d4 = _mm_set_ps(fp16_to_fp32_lookup(b->d[3]), fp16_to_fp32_lookup(b->d[2]),
                                     fp16_to_fp32_lookup(b->d[1]), fp16_to_fp32_lookup(b->d[0]));
        const __m128 m4 = _mm_set_ps(fp16_to_fp32_lookup(b->d[7]), fp16_to_fp32_lookup(b->d[6]),
                                     fp16_to_fp32_lookup(b->d[5]), fp16_to_fp32_lookup(b->d[4]));
        const __m128 vdy = _mm_set1_ps(q->d);
        outv = _mm_add_ps(outv, _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(d4, _mm_cvtepi32_ps(acc)), _mm_mul_ps(m4, _mm_cvtepi32_ps(mns))), vdy));
    }

    _mm_storeu_ps(out, outv);
}

int sgemm_q4_k_r4_q8_k_avx2(int nrows, int ncols, int k,
                             const void *vx, const void *vy,
                             float *out, size_t bs,
                             int ith, int nth) {
    if (nrows < 4 || ncols < 1 || k % QK_K != 0 || nrows % 4 != 0)
        return 0;

    size_t a_row_bytes = (size_t)(k / QK_K) * sizeof(block_q8_K);
    int start_col = (ncols * ith) / nth;
    int end_col = (ncols * (ith + 1)) / nth;
    if (end_col <= start_col) return 0;

    size_t w_stride = (size_t)(k / QK_K) * sizeof(block_q4_k_r4);
    for (int w4 = 0; w4 < nrows / 4; w4++) {
        const block_q4_k_r4 *iq4_base = (const block_q4_k_r4 *)
            ((const char *)vx + w4 * w_stride);
        for (int c = start_col; c < end_col; c++) {
            const block_q8_K *qy = (const block_q8_K *)
                ((const char *)vy + c * a_row_bytes);
            float *out_base = out + w4 * 4 + c * bs;
            vec_dot_q4_k_r4_q8_k_avx2(iq4_base, qy, k, out_base, 4);
        }
    }
    return (nrows / 4) * 4;
}

#else
void vec_dot_q4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n, float *out, int nrows) { (void)vx; (void)wy; (void)n; (void)out; (void)nrows; }
int sgemm_q4_k_r4_q8_k_avx2(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)vx; (void)vy; (void)nrows; (void)ncols; (void)k; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
#endif

/* ================================================================
 * Q4_K_R4 x Q8_K ARM NEON GEMV kernel (4-row interleaved)
 * GGUF type 212
 * ================================================================ */
#if defined(PICOLM_NEON)

static inline float q4_k_r4_get_d_scale(const block_q4_k_r4 *b, int is) {
    uint8_t sl = b->scales_l[is] & 0xf;
    uint8_t sh = (b->scales_h[is % 16] >> (4 * (is / 16))) & 0x03;
    return (float)(sl | (sh << 4));
}

static inline float q4_k_r4_get_m_scale(const block_q4_k_r4 *b, int is) {
    uint8_t sl = (b->scales_l[is] >> 4) & 0xf;
    uint8_t sh = (b->scales_h[is % 16] >> (4 * (is / 16))) & 0x0c;
    return (float)(sl | (sh << 2));
}

void vec_dot_q4_k_r4_q8_k_neon(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
    if (nrows != 4 || n % QK_K != 0) {
        const block_q4_k_r4 *b = (const block_q4_k_r4 *)vx;
        const block_q8_K *qk = (const block_q8_K *)wy;
        const int nb = n / QK_K;
        for (int r = 0; r < nrows && r < 4; r++) {
            float w_tmp[QK_K];
            float sum = 0.0f;
            for (int ibl = 0; ibl < nb; ibl++) {
                dequantize_row_q4_k_r4_single(&b[ibl], w_tmp, QK_K, r);
                float q8_scale = qk[ibl].d;
                for (int i = 0; i < QK_K; i++) {
                    sum += w_tmp[i] * (float)qk[ibl].qs[i] * q8_scale;
                }
            }
            out[r] = sum;
        }
        return;
    }

    const block_q4_k_r4 *x = (const block_q4_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    float outv[4] = {0, 0, 0, 0};

    for (int ibl = 0; ibl < nb; ibl++) {
        const block_q4_k_r4 *b = &x[ibl];
        const block_q8_K *q = &qk[ibl];

        float q8_scale = q->d;

        /* Per-row bias correction: -m[k] * min * sum(q8) */
        float bias[4] = {0, 0, 0, 0};
        for (int ib = 0; ib < QK_K / 32; ib++) {
            for (int k = 0; k < 4; k++) {
                int is = 4 * ib + k;
                float ml = fp16_to_fp32_lookup(b->d[k + 4]) * q4_k_r4_get_m_scale(b, is);
                bias[k] += ml * (q->bsums[ib * 2 + 0] + q->bsums[ib * 2 + 1]);
            }
        }

        for (int ib = 0; ib < QK_K / 32; ib++) {
            const uint8_t *qs = b->qs + 64 * ib;
            const int8_t *q8 = q->qs + 32 * ib;

            for (int k = 0; k < 4; k++) {
                int is = 4 * ib + k;
                float d = fp16_to_fp32_lookup(b->d[k]);
                float s = q4_k_r4_get_d_scale(b, is);

                float sum = 0.0f;
                /* 4 chunks of 4 bytes each, interleaved across rows */
                for (int cc = 0; cc < 4; cc++) {
                    int qs_off = 4 * k + cc;
                    const uint8_t *qs_chunk = qs + qs_off;
                    int q8_lo_base, q8_hi_base;
                    if (cc == 0) { q8_lo_base = 0; q8_hi_base = 8; }
                    else if (cc == 1) { q8_lo_base = 16; q8_hi_base = 24; }
                    else if (cc == 2) { q8_lo_base = 4; q8_hi_base = 12; }
                    else { q8_lo_base = 20; q8_hi_base = 28; }

                    for (int i = 0; i < 4; i++) {
                        uint8_t v = qs_chunk[i];
                        sum += (float)(v & 0xf) * q8[q8_lo_base + i * 2 + 0];
                        sum += (float)(v >> 4) * q8[q8_hi_base + i * 2 + 0];
                        sum += (float)(v & 0xf) * q8[q8_lo_base + i * 2 + 1];
                        sum += (float)(v >> 4) * q8[q8_hi_base + i * 2 + 1];
                    }
                }
                outv[k] += d * s * sum * q8_scale;
            }
        }

        for (int k = 0; k < 4; k++) {
            outv[k] -= bias[k] * q8_scale;
        }
    }

    for (int k = 0; k < 4; k++) out[k] = outv[k];
}

int sgemm_q4_k_r4_q8_k_neon(int nrows, int ncols, int k,
                             const void *vx, const void *vy,
                             float *out, size_t bs,
                             int ith, int nth) {
    if (nrows < 4 || ncols < 1 || k % QK_K != 0 || nrows % 4 != 0)
        return 0;

    size_t a_row_bytes = (size_t)(k / QK_K) * sizeof(block_q8_K);
    int start_col = (ncols * ith) / nth;
    int end_col = (ncols * (ith + 1)) / nth;
    if (end_col <= start_col) return 0;

    size_t w_stride = (size_t)(k / QK_K) * sizeof(block_q4_k_r4);
    for (int w4 = 0; w4 < nrows / 4; w4++) {
        const block_q4_k_r4 *q4_base = (const block_q4_k_r4 *)
            ((const char *)vx + w4 * w_stride);
        for (int c = start_col; c < end_col; c++) {
            const block_q8_K *qy = (const block_q8_K *)
                ((const char *)vy + c * a_row_bytes);
            float *out_base = out + w4 * 4 + c * bs;
            vec_dot_q4_k_r4_q8_k_neon(q4_base, qy, k, out_base, 4);
        }
    }
    return (nrows / 4) * 4;
}

/* NEON R4 function declarations for quant.c dispatcher */
extern void vec_dot_q4_k_r4_q8_k_neon(const void *vx, const void *wy, int n, float *out, int nrows);
extern int sgemm_q4_k_r4_q8_k_neon(int nrows, int ncols, int k,
                                    const void *vx, const void *vy,
                                    float *out, size_t bs, int ith, int nth);
#endif /* PICOLM_NEON */
