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

void vec_dot_q4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
    assert(nrows == 4);
    assert(n % QK_K == 0);

    const block_q4_k_r4 *x = (const block_q4_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    const __m128i mf  = _mm_set1_epi8(0xf);

    /* Extract row k's 4 unique bytes from a 16-byte chunk into low positions.
     * Layout: chunk bytes 0..3 = row0, 4..7 = row1, 8..11 = row2, 12..15 = row3.
     * Use -128 (0x80 as signed char) to zero the upper bytes. */
    const __m128i pick_src[4] = {
        _mm_set_epi8(-128,-128,-128,-128, -128,-128,-128,-128,
                     -128,-128,-128,-128, 3,2,1,0),
        _mm_set_epi8(-128,-128,-128,-128, -128,-128,-128,-128,
                     -128,-128,-128,-128, 7,6,5,4),
        _mm_set_epi8(-128,-128,-128,-128, -128,-128,-128,-128,
                     -128,-128,-128,-128, 11,10,9,8),
        _mm_set_epi8(-128,-128,-128,-128, -128,-128,-128,-128,
                     -128,-128,-128,-128, 15,14,13,12),
    };

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
                float ml = fp16_to_fp32_lookup(b->d[k + 4]) * get_m_scale(b, is);
                bias[k] += ml * (q->bsums[ib * 2 + 0] + q->bsums[ib * 2 + 1]);
            }
        }

        for (int ib = 0; ib < QK_K / 32; ib++) {
            const uint8_t *qs = b->qs + 64 * ib;
            const int8_t *q8 = q->qs + 32 * ib;

            __m128i c[4];
            c[0] = _mm_loadu_si128((const __m128i *)(qs + 0));
            c[1] = _mm_loadu_si128((const __m128i *)(qs + 16));
            c[2] = _mm_loadu_si128((const __m128i *)(qs + 32));
            c[3] = _mm_loadu_si128((const __m128i *)(qs + 48));

            for (int k = 0; k < 4; k++) {
                __m128i dots1 = _mm_setzero_si128();  /* chunks 0,1 */
                __m128i dots2 = _mm_setzero_si128();  /* chunks 2,3 */

                for (int cc = 0; cc < 4; cc++) {
                    /* Extract row k's 4 unique bytes from chunk cc into low positions */
                    __m128i r = _mm_shuffle_epi8(c[cc], pick_src[k]);
                    /* r = [b0,b1,b2,b3, 0,...,0] */

                    /* Split into lo/hi nibbles and interleave */
                    __m128i lo = _mm_and_si128(r, mf);
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(r, 4), mf);
                    __m128i w8 = _mm_unpacklo_epi8(lo, hi);
                    /* w8 = [lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3, 0,...,0] */

                    /* Activation indices for this chunk */
                    int act_lo_base, act_hi_base;
                    if (cc == 0) { act_lo_base = 0; act_hi_base = 8; }
                    else if (cc == 1) { act_lo_base = 16; act_hi_base = 24; }
                    else if (cc == 2) { act_lo_base = 4; act_hi_base = 12; }
                    else { act_lo_base = 20; act_hi_base = 28; }

                    /* Load 4 int8 activations for lo and hi, sign-extend to 4 int16s each */
                    int32_t act_lo32, act_hi32;
                    memcpy(&act_lo32, q8 + act_lo_base, sizeof(act_lo32));
                    memcpy(&act_hi32, q8 + act_hi_base, sizeof(act_hi32));
                    __m128i act_lo = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(act_lo32));
                    __m128i act_hi = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(act_hi32));
                    __m128i a16 = _mm_unpacklo_epi16(act_lo, act_hi);
                    /* a16 = [a_lo0,a_hi0,a_lo1,a_hi1,a_lo2,a_hi2,a_lo3,a_hi3] as 8 int16s */

                    /* Convert weights to int16s and multiply-accumulate */
                    __m128i w16 = _mm_cvtepu8_epi16(w8);
                    __m128i prod = _mm_madd_epi16(w16, a16);
                    /* prod = 4 int32s */

                    if (cc < 2) dots1 = _mm_add_epi32(dots1, prod);
                    else        dots2 = _mm_add_epi32(dots2, prod);
                }

                /* Reduce 4 int32s to 1 */
                __m128i rd1 = _mm_add_epi32(dots1, _mm_shuffle_epi32(dots1, 0x4e));
                rd1 = _mm_add_epi32(rd1, _mm_shuffle_epi32(rd1, 0xb1));
                __m128i rd2 = _mm_add_epi32(dots2, _mm_shuffle_epi32(dots2, 0x4e));
                rd2 = _mm_add_epi32(rd2, _mm_shuffle_epi32(rd2, 0xb1));

                /* Apply scale */
                int is = 4 * ib + k;
                float d = fp16_to_fp32_lookup(b->d[k]);
                float s = get_d_scale(b, is);
                outv[k] += (d * s * (float)(_mm_cvtsi128_si32(rd1) + _mm_cvtsi128_si32(rd2))) * q8_scale;
            }
        }

        for (int k = 0; k < 4; k++) {
            outv[k] -= bias[k] * q8_scale;
        }
    }

    for (int k = 0; k < 4; k++) out[k] = outv[k];
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
