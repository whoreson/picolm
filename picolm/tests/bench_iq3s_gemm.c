/* IQ3_S prefill GEMM: decode-once R8 kernel vs the scalar-integer GEMV.
 *
 *   ./bench_iq3s_gemm check                    edge-shape correctness vs scalar
 *   ./bench_iq3s_gemm [m n k reps threads]     throughput (GMAC/s)
 *
 * Weights are random IQ3_S blocks (any bit pattern is a valid block: the 9-bit
 * grid index accepts any qs/qh bits, the sign bytes and 4-bit scale nibbles
 * are always in range; d is a sane fp16). Activations are real Q8_K blocks
 * from quantize_row_q8_K. Reference = vec_dot_iq3_s_q8_k_scalar (the
 * integer-exact dequant-in-loop kernel also used as the scalar fallback).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quant.h"
#include "tensor.h"
void fp16_table_init(void);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

static unsigned rs = 7;
static unsigned rnd(void){ rs = rs*1664525u + 1013904223u; return rs >> 8; }

static block_iq3_s *mkW(int m, int nb) {
    block_iq3_s *W = malloc((size_t)m * nb * sizeof(block_iq3_s));
    for (size_t i = 0; i < (size_t)m * nb; i++) {
        uint8_t *b = (uint8_t *)&W[i];
        for (size_t j = 2; j < sizeof(block_iq3_s); j++) b[j] = (uint8_t)rnd();
        W[i].d = fp32_to_fp16(0.001f + (rnd() & 255) * 1e-5f);
    }
    return W;
}
static block_q8_K *mkY(int n, int k) {
    size_t nb = k / 256, rb = nb * sizeof(block_q8_K);
    block_q8_K *Y = malloc((size_t)n * rb);
    float *t = malloc(4 * (size_t)k);
    for (int c = 0; c < n; c++) {
        for (int i = 0; i < k; i++) t[i] = ((rnd() & 0xFFFF) / 32768.0f - 1.0f) * ((c % 5) + 1);
        quantize_row_q8_K(t, (char *)Y + (size_t)c * rb, k);
    }
    free(t);
    return Y;
}

typedef struct { int which, m, n, k, nth; const void *W, *Y; float *C; } ctx_t;
static void task(int idx, void *p) {
    ctx_t *c = (ctx_t *)p;
    if (c->which == 0) {
        int gpt = (c->m + c->nth - 1) / c->nth;
        int r0 = idx * gpt, r1 = (idx + 1) * gpt < c->m ? (idx + 1) * gpt : c->m;
        size_t nb = (size_t)c->k / 256, wrow = nb * sizeof(block_iq3_s);
        for (int i = r0; i < r1; i++)
            for (int col = 0; col < c->n; col++)
                c->C[(size_t)col * c->m + i] =
                    vec_dot_iq3_s_q8_k((const char *)c->W + (size_t)i * wrow,
                                       (const char *)c->Y + (size_t)col * nb * sizeof(block_q8_K), c->k);
    } else if (c->which == 1) {
        sgemm_iq3_s_q8_k_r8t(c->m, c->n, c->k, c->W, c->Y, c->C, (size_t)c->m, idx, c->nth);
    } else {
        sgemm_iq3_s_q8_k_r8v(c->m, c->n, c->k, c->W, c->Y, c->C, (size_t)c->m, idx, c->nth);
    }
}

static void run(int which, int m, int n, int k, int nth,
                const block_iq3_s *W, const block_q8_K *Y, float *C) {
    ctx_t c = { which, m, n, k, nth, W, Y, C };
    if (nth == 1) { task(0, &c); return; }
    /* emulate the pool: partition ith serially for the check */
    for (int th = 0; th < nth; th++) task(th, &c);
}

static int havev_g(void);
static int check(void) {
    int ms[]  = {8, 16, 24, 40, 136, 200};
    int ns[]  = {1, 2, 3, 5, 7, 8, 9, 11, 12, 13, 25, 95, 96, 97, 200};
    int ks[]  = {256, 512, 1280, 4096};
    int ths[] = {1, 2, 3, 7};
    int bad = 0, total = 0;
    for (int a = 0; a < 6; a++) for (int b = 0; b < 15; b++) for (int c = 0; c < 4; c++) for (int t = 0; t < 4; t++) {
        int m = ms[a], n = ns[b], k = ks[c], nth = ths[t], nb = k / 256;
        block_iq3_s *W = mkW(m, nb);
        block_q8_K *Y = mkY(n, k);
        float *C1 = calloc((size_t)n*m, 4);
        run(1, m, n, k, nth, W, Y, C1);   /* r8t */
        float *C3 = calloc((size_t)n*m, 4);
        int havev = sgemm_iq3_s_q8_k_r8v(m, n, k, W, Y, C3, (size_t)m, 0, 1) != 0;
        int ok = 1;
        if (havev) {
            memset(C3, 0, (size_t)n*m*4);
            run(2, m, n, k, nth, W, Y, C3);
            double m3 = 0, r3 = 0;
            for (size_t i = 0; i < (size_t)n*m; i++) { double d = fabs(C1[i]-C3[i]); if (d > m3) m3 = d; r3 += (double)C1[i]*C1[i]; }
            r3 = sqrt(r3/((size_t)n*m) + 1e-30);
            if (m3 > 2e-5*r3 + 1e-3) { ok = 0; printf("FAIL r8v m=%d n=%d k=%d nth=%d maxdiff=%.3g rms=%.3g\n", m, n, k, nth, m3, r3); }
        }
        /* reference: scalar integer-exact GEMV */
        double md = 0, rm = 0;
        size_t wrow = (size_t)nb * sizeof(block_iq3_s);
        for (int col = 0; col < n; col++)
            for (int i = 0; i < m; i++) {
                double ref = vec_dot_iq3_s_q8_k_scalar((const char *)W + (size_t)i*wrow,
                                                       (const char *)Y + (size_t)col*nb*sizeof(block_q8_K), k);
                double d = fabs(ref - (double)C1[(size_t)col*m + i]);
                if (d > md) md = d;
                rm += ref*ref;
            }
        rm = sqrt(rm/((size_t)n*m) + 1e-30);
        total++;
        if (!ok || md > 1e-3*rm + 1e-3) { bad++; printf("FAIL m=%d n=%d k=%d nth=%d maxdiff=%.3g rms=%.3g\n", m, n, k, nth, md, rm); }
        free(W); free(Y); free(C1); free(C3);
    }
    printf("check: %d/%d shapes OK (r8v %s)\n", total-bad, total, havev_g() ? "tested" : "unavailable");
    return bad;
}
static int havev_g(void) {
    /* probe once with a minimal shape */
    int k = 256, nb = 1;
    block_iq3_s *W = mkW(8, nb);
    block_q8_K *Y = mkY(1, k);
    float *C = calloc(8 * 1, 4);
    int r = sgemm_iq3_s_q8_k_r8v(8, 1, k, W, Y, C, 8, 0, 1) != 0;
    free(W); free(Y); free(C);
    return r;
}

int main(int argc, char **argv) {
    fp16_table_init();
    if (argc > 1 && !strcmp(argv[1], "check")) return check();
    int m = argc > 1 ? atoi(argv[1]) : 16384;
    int n = argc > 2 ? atoi(argv[2]) : 126;
    int k = argc > 3 ? atoi(argv[3]) : 7168;
    int reps = argc > 4 ? atoi(argv[4]) : 3;
    int nth = argc > 5 ? atoi(argv[5]) : 24;
    int nb = k / 256;
    block_iq3_s *W = mkW(m, nb);
    block_q8_K *Y = mkY(n, k);
    float *C = calloc((size_t)n*m, 4);
    run(1, m, n, k, nth, W, Y, C);   /* warm */
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        run(1, m, n, k, nth, W, Y, C);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    printf("r8t m=%d n=%d k=%d: %.2f ms  %.2f GMAC/s\n", m, n, k, best*1e3,
           (double)m*n*k / (best * 1e9));
    if (havev_g()) {
        memset(C, 0, (size_t)n*m*4);
        run(2, m, n, k, nth, W, Y, C);
        best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = now();
            run(2, m, n, k, nth, W, Y, C);
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("r8v m=%d n=%d k=%d: %.2f ms  %.2f GMAC/s\n", m, n, k, best*1e3,
               (double)m*n*k / (best * 1e9));
    }
    free(W); free(Y); free(C);
    return 0;
}

double get_time_ms(void) { return now() * 1e3; }
