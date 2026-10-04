/* sgemm_q4_k_r4.c -- AVX2 kernels for Q4_K_R4 (GGUF type 212) */

#include "quant.h"
#include <stdlib.h>
#include <assert.h>

#if defined(PICOLM_AVX2)
#include <immintrin.h>

void vec_dot_q4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows) {
    assert(nrows == 4);
    assert(n % QK_K == 0);

    const block_q4_k_r4 *x = (const block_q4_k_r4 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    const __m128i mf  = _mm_set1_epi8(0xf);

    /* Extract row k's 4 unique bytes from a 16-byte chunk.
     * Layout: chunk bytes 0..3 = row0, 4..7 = row1, 8..11 = row2, 12..15 = row3. */
    const __m128i pick_row[4] = {
        _mm_set_epi8(0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
                     0x80,0x80,0x80,0x80, 3,2,1,0),
        _mm_set_epi8(0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
                     0x80,0x80,0x80,0x80, 7,6,5,4),
        _mm_set_epi8(0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
                     0x80,0x80,0x80,0x80, 11,10,9,8),
        _mm_set_epi8(0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
                     0x80,0x80,0x80,0x80, 15,14,13,12),
    };

    float outv[4] = {0, 0, 0, 0};

    for (int ibl = 0; ibl < nb; ibl++) {
        const block_q4_k_r4 *b = &x[ibl];
        const block_q8_K *q = &qk[ibl];

        float q8_scale = q->d;

        /* Per-row bias correction: -m * min * sum(q8) */
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

        /* Extract 6-bit scales as 4 bytes per subblock */
        uint32_t sc_val[8];
        for (int ib = 0; ib < QK_K / 32; ib++) {
            uint32_t tmp = 0;
            for (int k = 0; k < 4; k++) {
                int is = 4 * ib + k;
                tmp |= (((b->scales_l[is] & 0xf) |
                         (((b->scales_h[is % 16] >> (4 * (is / 16))) & 0x03) << 4)) << (8 * k));
            }
            sc_val[ib] = tmp;
        }

        int32_t row_sum[4] = {0, 0, 0, 0};

        for (int ib = 0; ib < QK_K / 32; ib++) {
            const uint8_t *qs = b->qs + 64 * ib;
            const int8_t *q8 = q->qs + 32 * ib;

            __m128i c[4];
            c[0] = _mm_loadu_si128((const __m128i *)(qs + 0));
            c[1] = _mm_loadu_si128((const __m128i *)(qs + 16));
            c[2] = _mm_loadu_si128((const __m128i *)(qs + 32));
            c[3] = _mm_loadu_si128((const __m128i *)(qs + 48));

            for (int k = 0; k < 4; k++) {
                __m128i row_dots = _mm_setzero_si128();

                for (int cc = 0; cc < 4; cc++) {
                    /* Extract row k's 4 unique bytes from chunk cc into low positions */
                    __m128i r = _mm_shuffle_epi8(c[cc], pick_row[k]);
                    /* r = [b0,b1,b2,b3, 0,...,0] */

                    /* Split into 8 unique nibbles */
                    __m128i lo = _mm_and_si128(r, mf);
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(r, 4), mf);
                    __m128i w = _mm_unpacklo_epi64(lo, hi);
                    /* w = [lo0,lo1,lo2,lo3, hi0,hi1,hi2,hi3, 0,...,0] */

                    /* Activation indices for this chunk */
                    int act_lo_base, act_hi_base;
                    if (cc == 0) { act_lo_base = 0; act_hi_base = 8; }
                    else if (cc == 1) { act_lo_base = 16; act_hi_base = 24; }
                    else if (cc == 2) { act_lo_base = 4; act_hi_base = 12; }
                    else { act_lo_base = 20; act_hi_base = 28; }

                    /* Load 4 int8 activations for lo and hi, sign-extend to 4 int16s each */
                    __m128i act_lo = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(*(const int32_t *)(q8 + act_lo_base)));
                    __m128i act_hi = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(*(const int32_t *)(q8 + act_hi_base)));
                    __m128i a = _mm_unpacklo_epi64(act_lo, act_hi);
                    /* a = [a_lo0,a_lo1,a_lo2,a_lo3, a_hi0,a_hi1,a_hi2,a_hi3] as 8 int16s */

                    /* Convert weights to int16s and multiply-accumulate */
                    __m128i w16 = _mm_cvtepu8_epi16(w);
                    __m128i prod = _mm_madd_epi16(w16, a);
                    /* prod = 4 int32s */
                    row_dots = _mm_add_epi32(row_dots, prod);
                }

                /* Reduce 4 int32s to 1 */
                __m128i rd = _mm_add_epi32(row_dots, _mm_shuffle_epi32(row_dots, 0x4e));
                rd = _mm_add_epi32(rd, _mm_shuffle_epi32(rd, 0xb1));

                /* Scale: sc_val[ib] byte k */
                uint8_t sc = (sc_val[ib] >> (8 * k)) & 0xff;
                row_sum[k] += (int32_t)_mm_cvtsi128_si32(rd) * (int32_t)(int8_t)sc;
            }
        }

        for (int k = 0; k < 4; k++) {
            float d = fp16_to_fp32_lookup(b->d[k]);
            outv[k] += (d * (float)row_sum[k] - bias[k]) * q8_scale;
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
