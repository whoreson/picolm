/* IQ3_XXS prefill GEMM: decode-once R8 kernel vs the per-column GEMV.
 *
 *   ./bench_iq3xxs_gemm check                    edge-shape correctness vs scalar
 *   ./bench_iq3xxs_gemm [m n k reps threads]     throughput (GOPS)
 *                                                e.g. 17408 512 5120 3 24
 *
 * Weights are random IQ3_XXS blocks (any bit pattern is a valid block: the grid
 * index is any byte and the 4-bit scale nibble is always in range; d is set to a
 * sane fp16). Activations are real Q8_K blocks from quantize_row_q8_K.
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

static unsigned rs = 5;
static unsigned rnd(void){ rs = rs*1664525u + 1013904223u; return rs >> 8; }

static block_iq3_xxs *mkW(int m, int nb) {
    block_iq3_xxs *W = malloc((size_t)m * nb * sizeof(block_iq3_xxs));
    for (size_t i = 0; i < (size_t)m * nb; i++) {
        uint8_t *b = (uint8_t *)&W[i];
        for (size_t j = 2; j < sizeof(block_iq3_xxs); j++) b[j] = (uint8_t)rnd();
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

/* "Old" reference path: one GEMV call per activation column, as the pre-fix
 * dispatch did before any repacking existed. */
typedef struct { int which, m, n, k, nth; const void *W, *Y; float *C; } ctx_t;
static void task(int idx, void *p) {
    ctx_t *c = (ctx_t *)p;
    if (c->which == 0) {
        /* old: threaded by row, plain vec_dot per (row, column) */
        int gpt = (c->m + c->nth - 1) / c->nth;
        int r0 = idx * gpt, r1 = (idx + 1) * gpt < c->m ? (idx + 1) * gpt : c->m;
        size_t nb = (size_t)c->k / 256, wrow = nb * sizeof(block_iq3_xxs);
        for (int i = r0; i < r1; i++)
            for (int col = 0; col < c->n; col++)
                c->C[(size_t)col * c->m + i] =
                    vec_dot_iq3_xxs_q8_k((const char *)c->W + (size_t)i * wrow,
                                         (const char *)c->Y + (size_t)col * nb * sizeof(block_q8_K), c->k);
    } else if (c->which == 1) {
        sgemm_iq3_xxs_q8_k_r8t(c->m, c->n, c->k, c->W, c->Y, c->C, (size_t)c->m, idx, c->nth);
    } else if (c->which == 3) {
        sgemm_iq3_xxs_q8_k_neon(c->m, c->n, c->k, c->W, c->Y, c->C, (size_t)c->m, idx, c->nth);
    } else {
        sgemm_iq3_xxs_q8_k_r8v(c->m, c->n, c->k, c->W, c->Y, c->C, (size_t)c->m, idx, c->nth);
    }
}

static int check(void) {
    int ms[]  = {8, 16, 24, 40, 136, 200};
    int ns[]  = {1, 2, 3, 5, 7, 8, 9, 11, 12, 13, 25, 95, 96, 97, 200};
    int ks[]  = {256, 512, 1280, 4096};
    int ths[] = {1, 2, 3, 7};
    int bad = 0, total = 0;
    for (int a = 0; a < 6; a++) for (int b = 0; b < 15; b++) for (int c = 0; c < 4; c++) for (int t = 0; t < 4; t++) {
        int m = ms[a], n = ns[b], k = ks[c], nth = ths[t], nb = k / 256;
        block_iq3_xxs *W = mkW(m, nb);
        block_q8_K *Y = mkY(n, k);
        float *C1 = calloc((size_t)n*m, 4), *C2 = calloc((size_t)n*m, 4);
        int ok = 1;
        for (int th = 0; th < nth; th++) ok &= (sgemm_iq3_xxs_q8_k_r8t(m, n, k, W, Y, C1, m, th, nth) != 0);
        float *C3 = calloc((size_t)n*m, 4);
        int okv = 1, havev = sgemm_iq3_xxs_q8_k_r8v(m, n, k, W, Y, C3, m, 0, 1) != 0;
        if (havev) {
            memset(C3, 0, (size_t)n*m*4);
            for (int th = 0; th < nth; th++) okv &= (sgemm_iq3_xxs_q8_k_r8v(m, n, k, W, Y, C3, m, th, nth) != 0);
            double m3 = 0, r3 = 0;
            for (size_t i = 0; i < (size_t)n*m; i++) { double d = fabs(C1[i]-C3[i]); if (d > m3) m3 = d; r3 += (double)C1[i]*C1[i]; }
            r3 = sqrt(r3/((size_t)n*m) + 1e-30);
            if (!okv || m3 > 2e-5*r3 + 1e-3) { ok = 0; printf("FAIL r8v m=%d n=%d k=%d nth=%d maxdiff=%.3g rms=%.3g\n", m, n, k, nth, m3, r3); }
        }
        free(C3);
        /* Reference: the scalar integer-exact GEMV, row by row. */
        double md = 0, rm = 0;
        size_t wrow = (size_t)nb * sizeof(block_iq3_xxs);
        for (int col = 0; col < n; col++)
            for (int i = 0; i < m; i++) {
                double ref = vec_dot_iq3_xxs_q8_k_ref((const char *)W + (size_t)i*wrow,
                                                      (const char *)Y + (size_t)col*nb*sizeof(block_q8_K), k);
                double d = fabs(ref - (double)C1[(size_t)col*m + i]);
                if (d > md) md = d;
                rm += ref*ref;
            }
        rm = sqrt(rm/((size_t)n*m) + 1e-30);
        total++;
        if (!ok || md > 1e-3*rm + 1e-3) { bad++; printf("FAIL m=%d n=%d k=%d nth=%d ok=%d maxdiff=%.3g rms=%.3g\n", m, n, k, nth, ok, md, rm); }
        free(W); free(Y); free(C1); free(C2);
    }
    printf("check: %d/%d shapes OK\n", total-bad, total);
    return bad;
}

int main(int argc, char **argv) {
    fp16_table_init();
    if (argc > 1 && !strcmp(argv[1], "check")) return check() ? 1 : 0;

    int m = argc > 1 ? atoi(argv[1]) : 17408;
    int n = argc > 2 ? atoi(argv[2]) : 512;
    int k = argc > 3 ? atoi(argv[3]) : 5120;
    int reps = argc > 4 ? atoi(argv[4]) : 2;
    int nth = argc > 5 ? atoi(argv[5]) : 1;
    if (nth > 1) { tensor_set_n_threads(nth); tensor_threadpool_init(nth); }

    int nb = k / 256;
    block_iq3_xxs *W = mkW(m, nb);
    block_q8_K *Y = mkY(n, k);
    float *C0 = calloc((size_t)n*m, 4), *C1 = calloc((size_t)n*m, 4), *C2 = calloc((size_t)n*m, 4);
    double ops = 2.0*m*n*k, gt[2] = {0,0};

    for (int which = 0; which < 2; which++) {
        float *C = which ? C1 : C0;
        ctx_t cx = {which, m, n, k, nth, W, Y, C};
        double best = 1e9;
        for (int r = -1; r < reps; r++) {
            double t = now();
            if (nth > 1) tensor_parallel_for(nth, task, &cx); else task(0, &cx);
            t = now() - t;
            if (r >= 0 && t < best) best = t;
        }
        gt[which] = ops / best / 1e9;
        printf("  %-34s: %8.2f GOPS total = %6.2f per thread  (%d thread%s)\n",
               which ? "decode-once R8 AVX2 (new)" : "per-column GEMV (old)",
               gt[which], gt[which]/nth, nth, nth > 1 ? "s" : "");
    }
    int havev = sgemm_iq3_xxs_q8_k_r8v(m, n, k, W, Y, C2, m, 0, 1) != 0;
    double g2 = 0;
    if (havev) {
        ctx_t cx = {2, m, n, k, nth, W, Y, C2};
        double best = 1e9;
        for (int r = -1; r < reps; r++) {
            double t = now();
            if (nth > 1) tensor_parallel_for(nth, task, &cx); else task(0, &cx);
            t = now() - t;
            if (r >= 0 && t < best) best = t;
        }
        g2 = ops / best / 1e9;
        printf("  %-34s: %8.2f GOPS total = %6.2f per thread  (%d thread%s)\n",
               "decode-once R8 VNNI (new)", g2, g2/nth, nth, nth > 1 ? "s" : "");
    }
    /* ARM: the NEON tiled GEMM (UNTESTED kernel, see its TRACE line). The
     * x86-only kernels above return 0 / write nothing useful here, so on ARM
     * this is the number that matters. */
    double gn = 0;
    {
        float *Cn = calloc((size_t)n*m, 4);
        int haven = sgemm_iq3_xxs_q8_k_neon(m, n, k, W, Y, Cn, m, 0, 1) != 0;
        if (haven) {
            ctx_t cx = {3, m, n, k, nth, W, Y, Cn};
            double best = 1e9;
            for (int r = -1; r < reps; r++) {
                double t = now();
                if (nth > 1) tensor_parallel_for(nth, task, &cx); else task(0, &cx);
                t = now() - t;
                if (r >= 0 && t < best) best = t;
            }
            gn = ops / best / 1e9;
            printf("  %-34s: %8.2f GOPS total = %6.2f per thread  (%d thread%s)\n",
                   "NEON tiled (new, UNTESTED)", gn, gn/nth, nth, nth > 1 ? "s" : "");
            /* quality of the NEON kernel against the scalar reference */
            double mn = 0, rn = 0;
            size_t wrow = (size_t)(k/256) * sizeof(block_iq3_xxs);
            for (int col = 0; col < n; col++)
                for (int i = 0; i < m; i++) {
                    double ref = vec_dot_iq3_xxs_q8_k_ref((const char *)W + (size_t)i*wrow,
                                                          (const char *)Y + (size_t)col*(k/256)*sizeof(block_q8_K), k);
                    double d = fabs(ref - (double)Cn[(size_t)col*m + i]);
                    if (d > mn) mn = d;
                    rn += ref*ref;
                }
            rn = sqrt(rn/((size_t)n*m) + 1e-30);
            printf("  NEON vs scalar reference: max|diff|=%.3g rms=%.3g (relative %.2e)\n",
                   mn, rn, mn/(rn+1e-30));
        } else {
            printf("  %-34s: not built for this host\n", "NEON tiled (new, UNTESTED)");
        }
        free(Cn);
    }
    /* Quality: new kernels vs the scalar integer-exact reference, and each other. */
    double md = 0, rm = 0;
    for (size_t i = 0; i < (size_t)n*m; i++) { double d = fabs(C0[i]-C1[i]); if (d > md) md = d; rm += (double)C0[i]*C0[i]; }
    rm = sqrt(rm/((size_t)n*m) + 1e-30);
    printf("  speedup vs old: AVX2 decode-once %.2fx", gt[1]/gt[0]);
    if (havev) printf("   VNNI decode-once %.2fx", g2/gt[0]);
    printf("\n  max|old-new|=%.3g  rms=%.3g (relative %.2e)\n", md, rm, md/(rm+1e-30));
    return 0;
}
