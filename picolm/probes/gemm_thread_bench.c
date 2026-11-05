/* ============================================================================
 * probes/gemm_thread_bench.c - GEMM threading-scaling microbenchmark
 * ============================================================================
 *
 * Measures a single prefill GEMM the way PicoLM runs it (quantize activations
 * once, then picolm_sgemm_d() via tensor_parallel_for over the thread pool) and
 * splits the wall time into:
 *
 *     total = activation quantization  +  pool dispatch (barriers)  +  GEMM work
 *
 * Runs the same shape at -j 1,2,4,8,...,J and prints tok-equivalent scaling
 * plus parallel efficiency. A GEMM that behaves like ik_llama.cpp scales ~linearly
 * and keeps its per-thread GOPS constant; a GEMM whose pool barrier dominates
 * (small row counts per thread) shows per-thread GOPS falling with -j.
 *
 * Why a probe instead of only end-to-end: the process-level benchmark
 * (./picolm --benchmark / the "Prefill: N tokens in ..." line) mixes this with
 * attention, RoPE, KV store and sampling; the probe isolates the matmul, and it
 * works without a model file so it can be run on every host / CI.
 *
 * usage: ./probes/gemm_thread_bench [options]
 *   -m ROWS   output rows of the GEMM (weight rows)          default 6656
 *   -n TOKENS activation columns (= prefill tokens)          default 128
 *   -k COLS   input columns (= model dim)                    default 2048
 *   -t TYPE   weight quant: q8_0 | q4_0 | iq4_nl | q4_k      default q8_0
 *   -j LIST   comma separated thread counts, or MAX (1..max)  default 1,2,4,8,16,24
 *   -r REPS   repetitions per thread count                    default 5
 *   -c        check the threaded result against 1 thread and exit
 *
 * Nothing here needs a GGUF; weights and activations are random. pic
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quant.h"
#include "sgemm.h"
#include "tensor.h"

void fp16_table_init(void);

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static unsigned rs = 1;
static float fr(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

/* One GEMM dispatch through the pool, exactly like matmul_batch does it:
 * each thread runs picolm_sgemm_d with its own (ith, nth). */
typedef struct {
    int m, n, k_blocks, nth;
    const void *W; int lda;
    const block_q8_0 *B; int ldb;
    const float *B_d; int ldb_d;
    float *C; int ldc;
    gguf_type_t qtype;
    volatile int failed;
} gctx_t;

static void gemm_task(int idx, void *p) {
    gctx_t *g = (gctx_t *)p;
    int ok;
    if (g->qtype == GGUF_TYPE_Q4_K) {
        ok = picolm_sgemm_d_q4k(g->m, g->n, g->k_blocks, g->W, g->lda, g->B, g->ldb, g->C, g->ldc, idx, g->nth);
    } else {
        ok = picolm_sgemm_d(g->m, g->n, g->k_blocks, g->W, g->lda, g->B, g->ldb, g->B_d, g->ldb_d,
                            g->C, g->ldc, g->qtype, idx, g->nth);
    }
    if (!ok) g->failed = 1;
}

/* Barrier cost probe: an empty parallel region of the same width. */
static void nop_task(int idx, void *p) { (void)idx; (void)p; }

int main(int argc, char **argv) {
    int m = 6656, n = 128, k = 2048, reps = 5, check = 0;
    const char *type = "q8_0";
    int jlist[64], nj = 0;
    int jmax = 24;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) m = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) type = argv[++i];
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-j") && i + 1 < argc) {
            char *s = argv[++i], *p = s;
            while (*p) {
                char *e; long v = strtol(p, &e, 10);
                if (e == p) break;
                if (nj < 64) jlist[nj++] = (int)v;
                p = e; while (*p == ',' || *p == ' ') p++;
            }
        } else if (!strcmp(argv[i], "-c")) check = 1;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }

    gguf_type_t qt;
    if (!strcmp(type, "q8_0")) qt = GGUF_TYPE_Q8_0;
    else if (!strcmp(type, "q4_0")) qt = GGUF_TYPE_Q4_0;
    else if (!strcmp(type, "iq4_nl")) qt = GGUF_TYPE_IQ4_NL;
    else if (!strcmp(type, "q4_k")) qt = GGUF_TYPE_Q4_K;
    else { fprintf(stderr, "unknown type %s\n", type); return 2; }

    /* Q4_K needs k % 256 == 0 */
    if (qt == GGUF_TYPE_Q4_K && (k % 256) != 0) { fprintf(stderr, "-t q4_k needs -k a multiple of 256\n"); return 2; }
    if ((k % 32) != 0) { fprintf(stderr, "-k must be a multiple of 32\n"); return 2; }

    fp16_table_init();

    const int blocks_32 = k / 32;                 /* Q8_0/Q4_0/IQ4_NL blocks per row */
    const int nb = (qt == GGUF_TYPE_Q4_K) ? k / 256 : blocks_32;
    const size_t rb = gguf_type_row_size(qt, k);  /* weight row bytes */
    const size_t ab = gguf_type_row_size(GGUF_TYPE_Q8_0, k);  /* Q8_0 activation row bytes */

    /* random float activations, quantized once (as matmul_batch does) */
    float *xf = malloc((size_t)n * k * sizeof(float));
    for (size_t i = 0; i < (size_t)n * k; i++) xf[i] = fr();
    block_q8_0 *A = malloc((size_t)n * ab);
    float *Ad = malloc((size_t)n * blocks_32 * sizeof(float));
    for (int c = 0; c < n; c++) {
        quantize_row_q8_0(xf + (size_t)c * k, (char *)A + (size_t)c * ab, k);
        const block_q8_0 *bl = (const block_q8_0 *)((char *)A + (size_t)c * ab);
        for (int b = 0; b < blocks_32; b++) Ad[(size_t)c * blocks_32 + b] = fp16_to_fp32(bl[b].d);
    }
    /* Q4_K consumes Q8_K activations instead */
    block_q8_K *AK = NULL;
    if (qt == GGUF_TYPE_Q4_K) {
        AK = malloc((size_t)n * nb * sizeof(block_q8_K));
        for (int c = 0; c < n; c++) quantize_row_q8_K(xf + (size_t)c * k, AK + (size_t)c * nb, k);
    }

    /* random weights in the requested format */
    void *W = malloc((size_t)m * rb);
    float *wrow = malloc((size_t)k * sizeof(float));
    for (int r = 0; r < m; r++) {
        for (int i = 0; i < k; i++) wrow[i] = fr();
        switch (qt) {
            case GGUF_TYPE_Q8_0:   quantize_row_q8_0(wrow, (char *)W + (size_t)r * rb, k); break;
            case GGUF_TYPE_Q4_0:   quantize_row_q4_0(wrow, (char *)W + (size_t)r * rb, k); break;
            case GGUF_TYPE_IQ4_NL: quantize_row_iq4_nl(wrow, (char *)W + (size_t)r * rb, k); break;
            /* no Q4_K quantizer in this tree (production .ggufs arrive pre-quantized):
             * random nibbles with sane fp16 scales, as bench_kq.c does. */
            case GGUF_TYPE_Q4_K: {
                char *blk = (char *)W + (size_t)r * rb;
                for (int b = 0; b < nb; b++, blk += sizeof(block_q4_K)) {
                    block_q4_K *q4 = (block_q4_K *)blk;
                    memset(q4->qs, 0, sizeof q4->qs);
                    for (size_t i = 0; i < sizeof q4->qs; i++) q4->qs[i] = (uint8_t)(rs = rs * 1664525u + 1013904223u);
                    uint16_t d = fp32_to_fp16(0.001f + (rs >> 8 & 255) * 1e-5f);
                    memcpy(&q4->d, &d, 2);
                    memcpy(&q4->dmin, &d, 2);
                }
                break;
            }
            default: return 2;
        }
    }
    free(wrow);

    float *C = calloc((size_t)n * m, sizeof(float));
    double ops = 2.0 * (double)m * n * k;

    /* reference (1 thread) for the check mode */
    const block_q8_0 *Ab = (qt == GGUF_TYPE_Q4_K) ? (const block_q8_0 *)AK : A;
    gctx_t g = { m, n, nb, 1, W, nb, Ab, blocks_32, Ad, blocks_32, C, m, qt, 0 };
    gemm_task(0, &g);
    if (g.failed) { fprintf(stderr, "FATAL: picolm_sgemm_d unsupported for type %s in this build\n", type); return 1; }

    if (check) {
        float *C1 = malloc((size_t)n * m * sizeof(float));
        memcpy(C1, C, (size_t)n * m * sizeof(float));
        int bad = 0;
        static const int jchk[] = { 2, 3, 5, 8, 16, 24 };
        for (int ji = 0; ji < 6; ji++) {
            int j = jchk[ji];
            if (j <= 0) continue;
            tensor_set_n_threads(j);
            tensor_threadpool_init(j);
            memset(C, 0, (size_t)n * m * sizeof(float));
            gctx_t gj = { m, n, nb, j, W, nb, Ab, blocks_32, Ad, blocks_32, C, m, qt, 0 };
            tensor_parallel_for(j, gemm_task, &gj);
            double md = 0, rms = 0;
            for (size_t i = 0; i < (size_t)n * m; i++) { double d = fabs(C[i] - C1[i]); if (d > md) md = d; rms += C1[i] * (double)C1[i]; }
            rms = sqrt(rms / ((size_t)n * m) + 1e-30);
            if (md > 1e-3 * rms + 1e-4) { bad++; printf("FAIL j=%d maxdiff=%.3g (rms %.3g)\n", j, md, rms); }
            else if (md == 0.0) printf("j=%2d exact\n", j);
            else printf("j=%2d maxdiff=%.3g (rms %.3g) OK\n", j, md, rms);
        }
        free(C1);
        printf("check: %s\n", bad ? "FAILED" : "all thread counts agree");
        return bad ? 1 : 0;
    }

    if (nj == 0) { int v[] = { 1, 2, 4, 8, 16, 24 }; for (int i = 0; i < 6; i++) if (v[i] <= jmax) jlist[nj++] = v[i]; }
    if (nj == 1 && jlist[0] > 1) { int jm = jlist[0]; nj = 0; for (int j = 1; j <= jm; j *= 2) jlist[nj++] = j; if (jlist[nj - 1] != jm) jlist[nj++] = jm; }

    printf("GEMM thread scaling: m=%d n=%d k=%d type=%s  (%.2f GMAC, %.1f MB weights)\n",
           m, n, k, type, ops / 2e9, (double)m * rb / 1e6);
    printf("  j  per-thread rows   work ms   total GOPS   per-thread GOPS   scaling   empty-barrier ms\n");

    double base_gops = 0;
    for (int ji = 0; ji < nj; ji++) {
        int j = jlist[ji];
        if (j < 1) continue;
        tensor_set_n_threads(j);
        tensor_threadpool_init(j);

        double best = 1e9, best_empty = 1e9;
        for (int r = 0; r < reps; r++) {
            /* flush C so the kernel really writes it */
            memset(C, 0, (size_t)n * m * sizeof(float));
            gctx_t gj = { m, n, nb, j, W, nb, Ab, blocks_32, Ad, blocks_32, C, m, qt, 0 };
            double t0 = now_s();
            tensor_parallel_for(j, gemm_task, &gj);
            double dt = now_s() - t0;
            if (dt < best) best = dt;
            if (gj.failed) { printf("  %2d  GEMM returned 0 (unsupported)\n", j); break; }

            double t1 = now_s();
            tensor_parallel_for(j, nop_task, NULL);
            double de = now_s() - t1;
            if (de < best_empty) best_empty = de;
        }
        double gops = ops / best / 1e9;
        if (j == jlist[0]) base_gops = gops * j;   /* per-thread-normalised baseline */
        double ideal = base_gops * j;
        printf("  %2d   %8.1f       %7.2f     %8.1f        %8.1f       %5.1f%%   %8.3f\n",
               j, (double)m / j, best * 1e3, gops, gops / j,
               ideal > 0 ? 100.0 * gops / ideal : 0.0, best_empty * 1e3);
    }

    /* Reference: what end-to-end tok/s a prefill of `n` tokens through a
     * transformer would get if every layer were 3 such GEMMs. */
    printf("\nnote: 'scaling' is relative to the %d-thread run; 100%% = the GEMM\n"
           "      keeps its per-thread throughput as threads are added.\n", jlist[0]);
    free(W); free(A); free(Ad); free(C); free(xf);
    return 0;
}
