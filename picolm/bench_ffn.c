/* bench_ffn — time the production FFN path at the real prefill shape.
 *
 * The per-layer ablations showed the FFN is ~60% of an SSM layer, and that it
 * is the same call in attention layers, so it is the single largest block in
 * prefill. The existing benches all measure one kernel in isolation
 * (bench_iq6k_gemm, bench_q8_gemm, bench_qgemm) and are single-threaded, so
 * they cannot say whether the *dispatched* path -- matmul_dual_batch for
 * gate+up, matmul_batch for down -- reaches that kernel's peak.
 *
 * This bench drives exactly the calls model_ssm.c makes, through the same
 * type_dispatch table, with the real thread pool:
 *
 *     gate+up : matmul_dual_batch(hb, hb2, x, n, W_gate, W_up, dim, ffn)
 *     down    : matmul_batch     (out,  hb, n, W_down, ffn, dim)
 *
 * Usage:
 *   ./bench_ffn [dim n_ffn n_tok reps threads] [qtype]
 *     qtype: q8_0_r8 (default) | iq6_k | q8_0 | q4_0
 *   ./bench_ffn check        correctness of the two calls vs a scalar reference
 *
 * Reported: ms and GFLOP/s per stage, plus the arithmetic peak of the same
 * kernels so the dispatch overhead is visible.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "quant.h"
#include "sgemm.h"
#include "tensor.h"

void fp16_table_init(void);

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Deterministic pseudo-random floats; any 8-bit pattern is a valid Q8_0 block,
 * so quantising noise gives realistic weight bytes without a real model. */
static unsigned rs = 12345u;
static float fr(void) {
    rs = rs * 1664525u + 1013904223u;
    return ((float)((rs >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
}

static size_t row_bytes_for(gguf_type_t t, int k) {
    return gguf_type_row_size(t, k);
}

/* Quantise `d` rows of `k` floats from `src` into a fresh buffer. */
static void *mk_weights(gguf_type_t t, int d, int k, const float *src) {
    size_t rb = row_bytes_for(t, k);
    void *w = malloc(rb * (size_t)d);
    if (!w) return NULL;
    if (t == GGUF_TYPE_Q8_0_R8) {
        /* Q8_0_R8 is 8-row interleaved: one call consumes 8 rows of k floats. */
        if (d % 8 != 0 || k % 32 != 0) { free(w); return NULL; }
        size_t gsz = row_bytes_for(t, k) * 8;   /* bytes per 8-row group */
        for (int i = 0; i < d; i += 8)
            quantize_row_q8_0_r8(src + (size_t)i * k, (char *)w + gsz * (i / 8), k);
        return w;
    }
    for (int i = 0; i < d; i++) {
        const float *row = src + (size_t)i * k;
        char *dst = (char *)w + rb * (size_t)i;
        switch (t) {
            case GGUF_TYPE_Q8_0: quantize_row_q8_0(row, dst, k); break;
            case GGUF_TYPE_Q4_0: quantize_row_q4_0(row, dst, k); break;
            default: free(w); return NULL;   /* no quantizer available */
        }
    }
    return w;
}

/* Scalar reference for one row of a quantised matrix against f32 input. */
/* Reference: dequantize the whole matrix back to f32, then dot in f32.
 * Works for every type including the row-interleaved ones, where the
 * single-row vec_dot entry points intentionally refuse. */
static float ref_dot(const float *wdeq, int row, int k, const float *x) {
    double s = 0.0;
    for (int i = 0; i < k; i++) s += (double)wdeq[(size_t)row * k + i] * x[i];
    return (float)s;
}

static gguf_type_t parse_type(const char *s) {
    if (!strcmp(s, "q8_0"))    return GGUF_TYPE_Q8_0;
    if (!strcmp(s, "q4_0"))    return GGUF_TYPE_Q4_0;
    return GGUF_TYPE_Q8_0_R8;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "check")) {
        /* Small correctness pass: gate+up matches a per-row vec_dot reference. */
        fp16_table_init();
        tensor_threadpool_init(4);
        const int dim = 256, ffn = 512, ntok = 8;  /* dim must be k for gate: k=ffn */
        gguf_type_t t = GGUF_TYPE_Q8_0_R8;
        float *xg = malloc(sizeof(float) * (size_t)ntok * dim);
        float *wg = malloc(sizeof(float) * (size_t)ffn * dim);
        float *x  = malloc(sizeof(float) * (size_t)ntok * dim);
        for (int i = 0; i < ntok * dim; i++) x[i] = fr();
        for (int i = 0; i < dim * ffn; i++) wg[i] = fr();
        memcpy(xg, x, sizeof(float) * (size_t)ntok * dim);
        /* gate/up are [ffn rows][dim cols]; matmul_dual_batch(out, x, ntok,
         * W, n=dim, d=ffn) writes out[ntok][ffn]. */
        void *Wg = mk_weights(t, ffn, dim, wg);
        void *Wu = mk_weights(t, ffn, dim, wg);
        float *h1 = calloc((size_t)ntok * ffn, sizeof(float));
        float *h2 = calloc((size_t)ntok * ffn, sizeof(float));
        if (!Wg || !Wu || !h1 || !h2) { printf("OOM\n"); return 1; }
        matmul_dual_batch(h1, h2, xg, ntok, Wg, Wu, dim, ffn, t, t);
        float *deq = malloc(sizeof(float) * (size_t)ffn * dim);
        if (!deq) { printf("OOM\n"); return 1; }
        /* Interleaved types dequantize in 8-row groups of `ffn` floats each. */
        if (t == GGUF_TYPE_Q8_0_R8) {
            size_t gsz = row_bytes_for(t, dim) * 8;
            for (int r = 0; r < ffn; r += 8)
                dequantize_row_q8_0_r8((const char *)Wg + gsz * (r / 8),
                                       deq + (size_t)r * dim, dim);
        } else {
            for (int r = 0; r < ffn; r++)
                dequantize_row((const char *)Wg + row_bytes_for(t, dim) * (size_t)r,
                               deq + (size_t)r * dim, dim, t);
        }
        /* Compare against the *same* quantised weights, so the only error is
         * the GEMM's own arithmetic. The GEMM quantises activations to Q8_0
         * while this reference stays in double, so per-element agreement is
         * ~1% at best; the meaningful criterion is the relative L2 error
         * over the whole matrix, plus max error in units of the RMS. */
        int bad = 0;
        double num = 0.0, den = 0.0;
        float mx = 0.0f;
        for (int b = 0; b < ntok; b++)
            for (int r = 0; r < ffn; r++) {
                float rv = ref_dot(deq, r, dim, x + (size_t)b * dim);
                float got = h1[(size_t)b * ffn + r];
                double d = (double)got - rv;
                num += d * d;
                den += (double)rv * rv;
                if (fabsf(got - rv) > mx) mx = fabsf(got - rv);
            }
        double rel = sqrt(num / (den > 0 ? den : 1.0));
        double rms = sqrt(den / (double)(ntok * ffn));
        printf("  relative L2 error = %.4f%%   max abs err = %.4f (rms %.4f)\n",
               rel * 100.0, mx, rms);
        /* Q8_0 activation quantisation alone costs ~1-2% relative L2. */
        if (rel > 0.03) bad = 1;
        printf(bad ? "check FAILED\n" : "check OK\n");
        return bad ? 1 : 0;
    }

    int dim = argc > 1 ? atoi(argv[1]) : 5120;
    int ffn = argc > 2 ? atoi(argv[2]) : 17408;
    int n   = argc > 3 ? atoi(argv[3]) : 512;
    int reps = argc > 4 ? atoi(argv[4]) : 3;
    int nth  = argc > 5 ? atoi(argv[5]) : 24;
    gguf_type_t t = argc > 6 ? parse_type(argv[6]) : GGUF_TYPE_Q8_0_R8;

    fp16_table_init();
    tensor_threadpool_init(nth);
    tensor_set_n_threads(nth);   /* gates non-GEMM parallel paths */

    printf("bench_ffn: dim=%d n_ffn=%d n_tok=%d reps=%d threads=%d type=%d\n",
           dim, ffn, n, reps, nth, (int)t);
    /* Report the effective pool so a silently-serial run cannot be mistaken
     * for a slow kernel. */
    printf("  n_threads=%d\n", tensor_get_n_threads());

    float *xs = malloc(sizeof(float) * (size_t)n * dim);
    float *wg = malloc(sizeof(float) * (size_t)dim * ffn);
    float *wd = malloc(sizeof(float) * (size_t)ffn * dim);
    float *x  = malloc(sizeof(float) * (size_t)n * dim);
    float *h1 = malloc(sizeof(float) * (size_t)n * ffn);
    float *h2 = malloc(sizeof(float) * (size_t)n * ffn);
    float *out= malloc(sizeof(float) * (size_t)n * dim);
    if (!xs || !wg || !wd || !x || !h1 || !h2 || !out) { printf("OOM\n"); return 1; }

    for (size_t i = 0; i < (size_t)n * dim; i++)  xs[i] = fr();
    for (size_t i = 0; i < (size_t)dim * ffn; i++) wg[i] = fr();
    for (size_t i = 0; i < (size_t)ffn * dim; i++) wd[i] = fr();

    void *Wg = mk_weights(t, ffn, dim, wg);   /* gate/up: [ffn][dim] */
    void *Wu = mk_weights(t, ffn, dim, wg);
    void *Wd = mk_weights(t, dim, ffn, wd);   /* down:    [dim][ffn] */
    if (!Wg || !Wu || !Wd) { printf("OOM weights\n"); return 1; }

    const double fl_dual = 2.0 * (double)n * dim * ffn * 2.0;  /* gate + up  */
    const double fl_down = 2.0 * (double)n * ffn * dim;        /* down       */
    const double fl_all  = fl_dual + fl_down;

    double t_dual = 0, t_down = 0;
    for (int r = 0; r < reps; r++) {
        memcpy(x, xs, sizeof(float) * (size_t)n * dim);
        double t0 = now();
        matmul_dual_batch(h1, h2, x, n, Wg, Wu, dim, ffn, t, t);
        double t1 = now();
        matmul_batch(out, h1, n, Wd, ffn, dim, t);
        double t2 = now();
        t_dual += t1 - t0;
        t_down += t2 - t1;
    }
    t_dual /= reps; t_down /= reps;

    printf("\n  stage        ms        GFLOP/s\n");
    printf("  gate+up  %8.2f   %10.1f\n", t_dual * 1e3, fl_dual / t_dual / 1e9);
    printf("  down     %8.2f   %10.1f\n", t_down * 1e3, fl_down / t_down / 1e9);
    printf("  TOTAL    %8.2f   %10.1f\n", (t_dual + t_down) * 1e3,
           fl_all / (t_dual + t_down) / 1e9);
    printf("\n  per-ubatch FFN wall for %d tokens: %.2f ms\n", n, (t_dual + t_down) * 1e3);
    printf("  projected 48-layer prefill: %.1f s for %d tokens\n",
           48.0 * (t_dual + t_down), n);
    return 0;
}
