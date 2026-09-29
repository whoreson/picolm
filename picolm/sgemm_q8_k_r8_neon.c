/* ================================================================
 * NEON Q8_K_R8 x Q8_K GEMV and GEMM kernels
 * ================================================================
 * Scalar-style kernels that work on all ARM NEON platforms.
 * Uses scalar int8 multiply-accumulate with NEON for FP32 scaling.
 *
 * Weight layout: block_q8_k_r8 (GGUF type 399)
 *   d[8] FP16 scales + qs[2048] interleaved int8
 *   qs[32*ib + 4*k + i] for ib=0..63, k=row(0..7), i=0..3
 *   Each block: 8 rows x 256 values
 *
 * Activation: block_q8_K (one per row)
 *   d (float scale) + qs[256] signed int8
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include "quant.h"

#if defined(PICOLM_NEON)
#include <arm_neon.h>

#define QK_K 256

void vec_dot_q8_k_r8_q8_k_neon(const void *vx, const void *wy, int n,
                                float *out, int nrows) {
    assert(nrows == 8);
    assert(n % QK_K == 0);

    const block_q8_k_r8 *iq8 = (const block_q8_k_r8 *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    for (int k = 0; k < 8; k++) out[k] = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        float d_vals[8];
        for (int k = 0; k < 8; k++) d_vals[k] = fp16_to_fp32_lookup(iq8[ib].d[k]);
        float scale_y = qk[ib].d;

        const int8_t *qs = (const int8_t *)iq8[ib].qs;
        const int8_t *a = qk[ib].qs;

        for (int rk = 0; rk < 8; rk++) {
            float sum = 0.0f;
            for (int ib2 = 0; ib2 < QK_K / 16; ib2++) {
                for (int chunk = 0; chunk < 4; chunk++) {
                    const int8_t *w = qs + 32 * (4 * ib2 + chunk);
                    const int8_t *act = a + 16 * ib2 + 4 * chunk;
                    for (int j = 0; j < 4; j++) {
                        sum += (float)w[4 * rk + j] * act[j];
                    }
                }
            }
            out[rk] += sum * d_vals[rk] * scale_y;
        }
    }
}

int sgemm_q8_k_r8_q8_k_neon(int nrows, int ncols, int k,
                             const void *vx, const void *vy,
                             float *out, size_t bs,
                             int ith, int nth) {
    if (nrows < 8 || ncols < 1 || k % QK_K != 0 || nrows % 8 != 0)
        return 0;

    const int nb = k / QK_K;

    int64_t ytiles = nrows / 8;
    int64_t xtiles = ncols / 2;
    int64_t n_tail = ncols - xtiles * 2;
    int64_t xtiles_ext = xtiles + (n_tail > 0 ? 1 : 0);
    int64_t tiles = ytiles * xtiles_ext;
    if (tiles <= 0) return 0;

    int64_t duty = (tiles + nth - 1) / nth;
    int64_t start = duty * ith;
    int64_t end = start + duty;
    if (end > tiles) end = tiles;

    size_t w_group_bytes = nb * sizeof(block_q8_k_r8);
    size_t a_row_bytes = nb * sizeof(block_q8_K);

    for (int64_t job = start; job < end; job++) {
        int64_t ii = (job / xtiles_ext) * 8;
        int64_t xt = job % xtiles_ext;
        int64_t jj = xt * 2;
        int64_t ncols_tile = (xt < xtiles) ? 2 : n_tail;
        if (ncols_tile < 1) ncols_tile = 1;

        const block_q8_k_r8 *iq8 = (const block_q8_k_r8 *)
            ((const char *)vx + (ii / 8) * w_group_bytes);

        float *out_c[2];
        for (int c = 0; c < ncols_tile; c++)
            out_c[c] = out + ii + (jj + c) * bs;

        /* Local accumulators (zeroed, like AVX2 path).
         * 8 rows x 2 activation columns. */
        float acc[8][2] = {{0}};

        const block_q8_K *qk_ptr[2];

        for (int ibl = 0; ibl < nb; ibl++) {
            float d_vals[8];
            for (int k = 0; k < 8; k++) d_vals[k] = fp16_to_fp32_lookup(iq8[ibl].d[k]);

            for (int c = 0; c < ncols_tile; c++) {
                qk_ptr[c] = (const block_q8_K *)
                    ((const char *)vy + (jj + c) * a_row_bytes);
            }

            const int8_t *qs = (const int8_t *)iq8[ibl].qs;
            float row_sums[8][2] = {{0}};

            for (int c = 0; c < ncols_tile; c++) {
                const int8_t *a = qk_ptr[c][ibl].qs;

                for (int rk = 0; rk < 8; rk++) {
                    float sum = 0.0f;
                    /* Q8_K_R8 layout: qs[32*ib + 4*rk + i] where ib=0..63, rk=0..7, i=0..3.
                     * Each 32-byte chunk: 8 rows x 4 values, each row's 4 bytes contiguous. */
                    for (int ib2 = 0; ib2 < QK_K / 16; ib2++) {
                        for (int chunk = 0; chunk < 4; chunk++) {
                            const int8_t *w = qs + 32 * (4 * ib2 + chunk);
                            const int8_t *act = a + 16 * ib2 + 4 * chunk;
                            for (int j = 0; j < 4; j++) {
                                sum += (float)w[4 * rk + j] * act[j];
                            }
                        }
                    }
                    row_sums[rk][c] = sum;
                }
            }

            for (int c = 0; c < ncols_tile; c++) {
                float scale_y = qk_ptr[c][ibl].d;
                for (int rk = 0; rk < 8; rk++) {
                    acc[rk][c] += row_sums[rk][c] * d_vals[rk] * scale_y;
                }
            }
        }

        /* Store results to output buffer (not +=, to avoid stale data) */
        for (int c = 0; c < ncols_tile; c++) {
            for (int rk = 0; rk < 8; rk++) {
                out_c[c][rk] = acc[rk][c];
            }
        }
    }

    return nrows;
}

#else
/* Non-NEON stubs */
void vec_dot_q8_k_r8_q8_k_neon(const void *vx, const void *wy, int n, float *out, int nrows) {
    (void)vx; (void)wy; (void)n; (void)out; (void)nrows;
}
int sgemm_q8_k_r8_q8_k_neon(int nrows, int ncols, int k, const void *vx, const void *vy,
                             float *out, size_t bs, int ith, int nth) {
    (void)nrows; (void)ncols; (void)k; (void)vx; (void)vy;
    (void)out; (void)bs; (void)ith; (void)nth;
    return 0;
}
#endif

