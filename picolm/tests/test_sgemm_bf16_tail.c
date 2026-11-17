/* test_sgemm_bf16_tail.c
 *
 * sgemm_bf16_f32 / sgemm_bf16_bf16 vs a double-precision reference, at k values
 * that exercise the unrolled main loop, the 16-wide remainder loop, and the
 * tail rows/columns of the tile loop.
 *
 * Why this exists: the BF16 AVX-512 kernels unroll their k loop SGEMM_KU (16)
 * sixteen-wide steps at a time and finish k % (16*SGEMM_KU) in a remainder
 * loop. The loop advance and the number of emitted steps must stay in sync --
 * an advance larger than the steps emitted silently skips a slice of the k
 * range and still produces plausible-looking numbers. That is exactly the bug
 * this test caught during development (SGEMM_KU>4 with only 4 steps emitted:
 * correct on k=2048 by luck of alignment, garbage on the model output).
 *
 * The reference is computed in double: BF16 weights are exact in fp32, so the
 * only error source is fp32 accumulation order, and a generous 1e-3 relative
 * tolerance is plenty to catch a skipped k slice (which shows up as O(1)
 * relative error while still "looking like" a dot product).
 *
 * Build/run: make check-sgemm-bf16
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "quant.h"
#include "sgemm.h"

float bf16_to_fp32(uint16_t x);

static unsigned rs = 3;
static unsigned rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

/* k values: multiples of 16 (the SGEMM gate), covering
 *   - smaller than one unrolled iteration,
 *   - exactly one, and one plus/minus a step,
 *   - the real model widths (2048/4096/6656) and non-multiples of 16*KU.
 * k values that are not multiples of 16 are rejected by picolm_sgemm() by
 * design and are not tested here. */
static const int ks[] = { 16, 32, 48, 240, 256, 272, 512, 2048, 2064, 2080,
                          4096, 4112, 6656, 6672 };

static int check_f32(int k, int m, int n, int verbose) {
    uint16_t *W = malloc((size_t)m * k * sizeof(uint16_t));
    float *X = malloc((size_t)n * k * sizeof(float));
    float *C = malloc((size_t)n * m * sizeof(float));
    if (!W || !X || !C) { printf("OOM\n"); return 1; }

    /* BF16 values of sane magnitude (exponent 125..128 => 0.03..8). */
    for (int i = 0; i < m * k; i++) {
        uint32_t f = (rnd() & 0x80000000u) | ((uint32_t)(rnd() % 4 + 125) << 23) | (rnd() & 0x7FFFFF);
        W[i] = (uint16_t)(f >> 16);
    }
    for (int i = 0; i < n * k; i++)
        X[i] = ((float)(int)(rnd() % 2000) - 1000.0f) / 1000.0f;

    memset(C, 0, (size_t)n * m * sizeof(float));
    if (!picolm_sgemm(m, n, k, W, k, X, k, C, m, GGUF_TYPE_BF16, GGUF_TYPE_F32, 0, 1)) {
        printf("FAIL bf16_f32 k=%-5d: picolm_sgemm declined an aligned shape\n", k);
        free(W); free(X); free(C);
        return 1;
    }

    int bad = 0;
    for (int jj = 0; jj < n; jj++)
        for (int ii = 0; ii < m; ii++) {
            double s = 0.0;
            for (int l = 0; l < k; l++)
                s += (double)bf16_to_fp32(W[(size_t)ii * k + l]) * (double)X[(size_t)jj * k + l];
            double got = C[(size_t)jj * m + ii];
            double rel = fabs(got - s) / (fabs(s) + 1e-6);
            if (rel > 1e-3) {
                if (bad < 4)
                    printf("FAIL bf16_f32 k=%-5d [%d,%d] got=%.6g ref=%.6g rel=%.2e\n",
                           k, jj, ii, got, s, rel);
                bad++;
            }
        }
    if (verbose)
        printf("  %s bf16_f32 k=%-5d m=%d n=%d (%d bad)\n", bad ? "FAIL" : "ok  ", k, m, n, bad);
    free(W); free(X); free(C);
    return bad ? 1 : 0;
}

static int check_bf16(int k, int m, int n, int verbose) {
    uint16_t *W = malloc((size_t)m * k * sizeof(uint16_t));
    uint16_t *X = malloc((size_t)n * k * sizeof(uint16_t));
    float *C = malloc((size_t)n * m * sizeof(float));
    if (!W || !X || !C) { printf("OOM\n"); return 1; }
    for (int i = 0; i < m * k; i++) {
        uint32_t f = (rnd() & 0x80000000u) | ((uint32_t)(rnd() % 4 + 125) << 23) | (rnd() & 0x7FFFFF);
        W[i] = (uint16_t)(f >> 16);
    }
    for (int i = 0; i < n * k; i++) {
        uint32_t f = (rnd() & 0x80000000u) | ((uint32_t)(rnd() % 4 + 118) << 23) | (rnd() & 0x7FFFFF);
        X[i] = (uint16_t)(f >> 16);
    }
    memset(C, 0, (size_t)n * m * sizeof(float));
    /* SGEMM_KN for the BF16 kernel is 16 on AVX-512 and 8 on AVX2; the gate
     * below only needs to hold for AVX-512, where both types are supported. */
    if (!picolm_sgemm(m, n, k, W, k, X, k, C, m, GGUF_TYPE_BF16, GGUF_TYPE_BF16, 0, 1)) {
        if (verbose) printf("  skip bf16_bf16 (no kernel for this build)\n");
        free(W); free(X); free(C);
        return 0;
    }
    int bad = 0;
    for (int jj = 0; jj < n; jj++)
        for (int ii = 0; ii < m; ii++) {
            double s = 0.0;
            for (int l = 0; l < k; l++)
                s += (double)bf16_to_fp32(W[(size_t)ii * k + l]) * (double)bf16_to_fp32(X[(size_t)jj * k + l]);
            double got = C[(size_t)jj * m + ii];
            double rel = fabs(got - s) / (fabs(s) + 1e-6);
            if (rel > 1e-3) {
                if (bad < 4)
                    printf("FAIL bf16_bf16 k=%-5d [%d,%d] got=%.6g ref=%.6g rel=%.2e\n",
                           k, jj, ii, got, s, rel);
                bad++;
            }
        }
    if (verbose)
        printf("  %s bf16_bf16 k=%-5d m=%d n=%d (%d bad)\n", bad ? "FAIL" : "ok  ", k, m, n, bad);
    free(W); free(X); free(C);
    return bad ? 1 : 0;
}

int main(void) {
    fp16_table_init();
    int fails = 0, runs = 0;
    printf("sgemm BF16 tail/unroll test\n");
    for (unsigned ki = 0; ki < sizeof(ks) / sizeof(ks[0]); ki++) {
        int k = ks[ki];
        /* m must satisfy m % 4 == 0 (the kernel tiles 4 rows); n >= 2.
         * 8 x 13 exercises the row block, the RN column tile and its tail. */
        fails += check_f32(k, 8, 13, 1);
        fails += check_f32(k, 16, 6, 1);
        fails += check_bf16(k, 8, 13, 1);
        runs += 3;
    }
    printf("\n%d/%d cases passed\n", runs - fails, runs);
    if (fails) printf("FAILED\n");
    return fails ? 1 : 0;
}
