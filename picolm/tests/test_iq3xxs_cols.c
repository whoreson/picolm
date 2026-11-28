/* test_iq3xxs_cols: vec_dot_iq3_xxs_q8_k_cols() must be BIT-IDENTICAL to calling
 * vec_dot_iq3_xxs_q8_k() once per column, for every column count (the kernel
 * dispatches 8/7/6/5/4/3/2/1-wide groups) and every K the MoE shapes use, and
 * through the table fn_gemm for both sides of the R8 gate.
 *
 * Exhaustive over rows and columns; weights are random bits with sane fp16
 * scales (every IQ3_XXS index encoding is legal). fp16_table_init() first:
 * without it every scale is 0 and the test passes vacuously (0 == 0), so the
 * test also asserts the outputs are non-trivial.
 *
 * Build/run:  make check-iq3xxs-cols */
#include "quant.h"
#include "tensor.h"
#include "type_dispatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t rs = 0xC0FFEE11u;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

static int run(int n, int d, int ncols)
{
    const gguf_type_t T = GGUF_TYPE_IQ3_XXS;
    const type_info_t *ti = typeinfo(T);
    const size_t rb = gguf_type_row_size(T, n), bb = rb / (size_t)(n / 256);
    const size_t qs = (size_t)(n / 256) * sizeof(block_q8_K);
    uint8_t *W = (uint8_t *)malloc(rb * (size_t)d);
    float *x = (float *)malloc((size_t)ncols * n * sizeof(float));
    char *q = (char *)malloc(qs * (size_t)ncols);
    float *res = (float *)malloc((size_t)ncols * sizeof(float));
    float *out = (float *)malloc((size_t)ncols * d * sizeof(float));
    if (!W || !x || !q || !res || !out) return 1;
    for (size_t i = 0; i < rb * (size_t)d; i++) W[i] = (uint8_t)rnd();
    const uint16_t sc = 0x2C00;
    for (int r = 0; r < d; r++)
        for (size_t o = 0; o + bb <= rb; o += bb) memcpy(W + (size_t)r * rb + o, &sc, 2);
    for (size_t i = 0; i < (size_t)ncols * n; i++) x[i] = ((int)(rnd() % 20001) - 10000) / 3333.0f;
    for (int c = 0; c < ncols; c++) quantize_row_q8_K(x + (size_t)c * n, q + c * qs, n);

    int bad = 0; double mag = 0;
    for (int i = 0; i < d; i++) {
        vec_dot_iq3_xxs_q8_k_cols(W + (size_t)i * rb, q, qs, ncols, n, res);
        for (int c = 0; c < ncols; c++) {
            float ref = vec_dot_iq3_xxs_q8_k(W + (size_t)i * rb, q + c * qs, n);
            if (memcmp(&ref, &res[c], 4)) bad++;
            mag += fabs(ref);
        }
    }
    /* Through the table GEMM (below the R8 gate it must equal the per-column dot too). */
    memset(out, 0, (size_t)ncols * d * sizeof(float));
    ti->fn_gemm(d, ncols, n, W, q, out, (size_t)d, 0, 1);
    if (ncols < picolm_iq3xxs_cols_gate())   /* below the R8 gate it must equal the per-column dot */
        for (int i = 0; i < d; i++)
            for (int c = 0; c < ncols; c++) {
                float ref = vec_dot_iq3_xxs_q8_k(W + (size_t)i * rb, q + c * qs, n);
                if (memcmp(&ref, &out[(size_t)c * d + i], 4)) bad++;
            }
    if (mag <= 0.0) { printf("  n=%d ncols=%d: VACUOUS (all zero)\n", n, ncols); bad++; }
    if (bad) printf("  n=%-5d d=%-4d ncols=%-3d FAIL: %d mismatching values\n", n, d, ncols, bad);
    free(W); free(x); free(q); free(res); free(out);
    return bad != 0;
}

int main(void)
{
    fp16_table_init();
    int fails = 0, cases = 0;
    const int ns[] = { 256, 768, 2048, 7168 };
    for (unsigned a = 0; a < sizeof(ns) / sizeof(*ns); a++)
        for (int nc = 1; nc <= 70; nc++) { fails += run(ns[a], 24, nc); cases++; }
    printf("test_iq3xxs_cols: %d cases, %d failed\n", cases, fails);
    return fails != 0;
}
