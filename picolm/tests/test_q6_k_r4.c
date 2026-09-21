/* Standalone correctness test for Q6_K_R4 dequantize + vec_dot + sgemm.
 *
 * Ground truth: dequantize_row_q6_K_R4() (manually verified byte-exact
 * against ik_llama.cpp's repack_q6_k()/convert_q6_k() source) turns a
 * randomized block_q6_K_R4 group into 4 plain float rows; a randomized
 * block_q8_K is turned into 256 plain floats the same way (qs[i]*d).
 * A naive float dot product of the two is compared against:
 *   - vec_dot_q6_K_R4_q8_K (AVX2 path, since this build has __AVX2__)
 *   - sgemm_q6_k_r4_q8_k (tiles the same kernel over rows/cols)
 *
 * Run twice: once compiled with -mavx2 (tests the AVX2 kernel) and once
 * with plain -O2 -std=c11 (no AVX2 defines -> tests the scalar fallback).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../quant.h"

static uint32_t rng_state = 0xC0FFEEu;
static uint32_t xrand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static int randi(int lo, int hi) { return lo + (int)(xrand() % (uint32_t)(hi - lo + 1)); }

int main(void) {
    fp16_table_init();

    const int n = 256 * 3; /* 3 blocks per row, exercise the ibl loop */
    const int nb = n / 256;
    int fails = 0;
    const int NCASES = 200;

    for (int c = 0; c < NCASES; c++) {
        block_q6_K_R4 *wx = (block_q6_K_R4 *)malloc((size_t)nb * sizeof(block_q6_K_R4));
        block_q8_K *qy = (block_q8_K *)malloc((size_t)nb * sizeof(block_q8_K));

        for (int b = 0; b < nb; b++) {
            for (int k = 0; k < 4; k++) {
                /* Random fp16 delta, modest magnitude */
                uint16_t sign = (uint16_t)(randi(0, 1) << 15);
                uint16_t exp = (uint16_t)(randi(8, 18) << 10); /* keep away from inf/nan/denorm */
                uint16_t mant = (uint16_t)randi(0, 1023);
                wx[b].d[k] = (uint16_t)(sign | exp | mant);
            }
            for (int i = 0; i < 64; i++) wx[b].scales[i] = (int8_t)randi(-40, 40);
            for (int i = 0; i < 256; i++) wx[b].qh[i] = (uint8_t)randi(0, 255);
            for (int i = 0; i < 512; i++) wx[b].ql[i] = (uint8_t)randi(0, 255);

            qy[b].d = (float)randi(1, 20) * 0.0625f;
            for (int i = 0; i < 256; i++) qy[b].qs[i] = (int8_t)randi(-127, 127);
            for (int g = 0; g < 16; g++) {
                int s = 0;
                for (int i = 0; i < 16; i++) s += qy[b].qs[g * 16 + i];
                qy[b].bsums[g] = (int16_t)s;
            }
        }

        /* Ground truth via dequantize (manually verified against repack_q6_k) */
        float *wf = (float *)malloc((size_t)4 * n * sizeof(float));
        dequantize_row_q6_K_R4(wx, wf, n);
        float *yf = (float *)malloc((size_t)n * sizeof(float));
        for (int b = 0; b < nb; b++)
            for (int i = 0; i < 256; i++)
                yf[b * 256 + i] = qy[b].d * (float)qy[b].qs[i];

        double expected[4];
        for (int k = 0; k < 4; k++) {
            double s = 0.0;
            for (int i = 0; i < n; i++) s += (double)wf[k * n + i] * (double)yf[i];
            expected[k] = s;
        }

        float got[4];
        vec_dot_q6_K_R4_q8_K(wx, qy, n, got);

        for (int k = 0; k < 4; k++) {
            double rel = fabs((double)got[k] - expected[k]) / (fabs(expected[k]) + 1e-6);
            if (rel > 1e-3) {
                printf("MISMATCH case %d row %d: expected %.6f got %.6f (rel %.6g)\n",
                       c, k, expected[k], got[k], rel);
                fails++;
            }
        }

        /* Also exercise sgemm_q6_k_r4_q8_k with 8 rows (2 groups) x 3 cols */
        {
            const int nrows = 8, ncols = 3;
            block_q6_K_R4 *wx8 = (block_q6_K_R4 *)malloc((size_t)2 * nb * sizeof(block_q6_K_R4));
            memcpy(wx8, wx, (size_t)nb * sizeof(block_q6_K_R4));
            memcpy(wx8 + nb, wx, (size_t)nb * sizeof(block_q6_K_R4)); /* same data, 2nd group */
            block_q8_K *qy3 = (block_q8_K *)malloc((size_t)ncols * nb * sizeof(block_q8_K));
            for (int col = 0; col < ncols; col++)
                memcpy(qy3 + col * nb, qy, (size_t)nb * sizeof(block_q8_K));

            float out[8 * 3];
            memset(out, 0, sizeof(out));
            int done = sgemm_q6_k_r4_q8_k(nrows, ncols, n, wx8, qy3, out, nrows, 0, 1);
            if (done != nrows) {
                printf("sgemm returned %d, expected %d\n", done, nrows);
                fails++;
            } else {
                for (int col = 0; col < ncols; col++) {
                    for (int k = 0; k < 4; k++) {
                        double rel = fabs((double)out[col * nrows + k] - expected[k]) / (fabs(expected[k]) + 1e-6);
                        if (rel > 1e-3) {
                            printf("SGEMM MISMATCH col %d row %d: expected %.6f got %.6f\n",
                                   col, k, expected[k], out[col * nrows + k]);
                            fails++;
                        }
                        /* second row-group (rows 4-7) is a copy of the first */
                        double rel2 = fabs((double)out[col * nrows + 4 + k] - expected[k]) / (fabs(expected[k]) + 1e-6);
                        if (rel2 > 1e-3) {
                            printf("SGEMM MISMATCH(g2) col %d row %d: expected %.6f got %.6f\n",
                                   col, k, expected[k], out[col * nrows + 4 + k]);
                            fails++;
                        }
                    }
                }
            }
            free(wx8);
            free(qy3);
        }

        free(wx); free(qy); free(wf); free(yf);
    }

    if (fails) {
        printf("FAILED: %d mismatches\n", fails);
        return 1;
    }
    printf("OK: all %d cases passed (vec_dot + sgemm)\n", NCASES);
    return 0;
}
