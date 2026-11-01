/* Thread-pool stress test: random dispatch sizes, per-dispatch verification, idle gaps that
 * force the sleep path, and real matmul dispatches. Detects lost wakeups (hang), stale or
 * early completion (wrong result) and data races (run under -fsanitize=thread).
 * usage: test_pool_stress [threads] [dispatches] [seed]
 * env:   PICOLM_POOL_SPIN_US selects the spin budget (0 = always sleep). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "tensor.h"
#include "type_dispatch.h"
#include "quant.h"

static int *g_out; static int g_iter;
static void fill_task(int idx, void *ctx) { (void)ctx; g_out[idx] = idx * 7 + g_iter; }
static unsigned rs = 12345;
static unsigned rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }


/* ---- nested dispatch: a task that itself calls tensor_parallel_for must run serially (no corruption/deadlock) ---- */
static int g_nest[64 * 16];
typedef struct { int outer; } nest_ctx_t;
static void nest_inner(int idx, void *p) { int outer = ((nest_ctx_t *)p)->outer; g_nest[outer * 16 + idx] = outer * 1000 + idx + 1; }
static void nest_outer(int idx, void *p) { (void)p; nest_ctx_t c = { idx }; tensor_parallel_for(12, nest_inner, &c); }
static int test_nested(int rounds) {
    for (int r = 0; r < rounds; r++) {
        memset(g_nest, 0, sizeof g_nest);
        int outer_n = 1 + (r % 64);
        tensor_parallel_for(outer_n, nest_outer, NULL);
        for (int o = 0; o < outer_n; o++) for (int i = 0; i < 12; i++)
            if (g_nest[o * 16 + i] != o * 1000 + i + 1) { printf("FAIL nested round=%d outer=%d inner=%d got %d\n", r, o, i, g_nest[o * 16 + i]); return 1; }
    }
    return 0;
}

/* ---- row-parallel batched activation quantization == per-row serial reference ---- */
static int test_quant_rows(void) {
    gguf_type_t types[2] = { GGUF_TYPE_Q8_0, GGUF_TYPE_IQ6_K };   /* Q8_0 acts (+deltas), Q8_K acts */
    for (int t = 0; t < 2; t++) {
        const int n = 1024, B = 61;
        float *x = malloc((size_t)B * n * sizeof(float));
        for (size_t i = 0; i < (size_t)B * n; i++) x[i] = ((int)(rnd() % 4001) - 2000) / 500.0f;
        quant_buf_t all = quant_activations_ex(types[t], x, B, n, NULL, 0, 0);   /* parallel path (B >= 16, pool > 1) */
        if (!all.qbuf) { printf("FAIL quant alloc\n"); return 1; }
        for (int b = 0; b < B; b++) {
            quant_buf_t one = quant_activations_ex(types[t], x + (size_t)b * n, 1, n, NULL, 0, 0);   /* serial path */
            int bad = memcmp((char *)all.qbuf + (size_t)b * all.qstride, one.qbuf, all.qstride) != 0;
            if (!bad && all.dbuf) bad = memcmp(all.dbuf + (size_t)b * (n / 32), one.dbuf, (n / 32) * sizeof(float)) != 0;
            free_quant_buf(&one, NULL);
            if (bad) { printf("FAIL quant rows type=%d row=%d differs from serial\n", (int)types[t], b); return 1; }
        }
        free_quant_buf(&all, NULL); free(x);
    }
    return 0;
}

int main(int argc, char **argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 4, N = argc > 2 ? atoi(argv[2]) : 20000;
    rs ^= (unsigned)(argc > 3 ? atoi(argv[3]) : 1) * 2654435761u;
    tensor_set_n_threads(nt);
    tensor_threadpool_init(nt);
    g_out = calloc(256, sizeof(int));
    const int n = 96, d = 70;                       /* matmul check: F32 weights */
    float *W = malloc(sizeof(float) * n * d), *x = malloc(sizeof(float) * n);
    float *o = malloc(sizeof(float) * d), *ref = malloc(sizeof(float) * d);
    for (int i = 0; i < n * d; i++) W[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;
    for (int i = 0; i < n; i++) x[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;
    for (int r = 0; r < d; r++) { double a = 0; for (int c = 0; c < n; c++) a += (double)W[(size_t)r * n + c] * x[c]; ref[r] = (float)a; }
    int bad = 0;
    for (int it = 0; it < N && !bad; it++) {
        g_iter = it;
        int count = 1 + rnd() % 100;
        memset(g_out, 0xFF, sizeof(int) * 256);
        tensor_parallel_for(count, fill_task, NULL);
        for (int i = 0; i < count; i++) if (g_out[i] != i * 7 + it) { printf("FAIL it=%d idx=%d got %d want %d\n", it, i, g_out[i], i * 7 + it); bad = 1; break; }
        for (int i = count; i < 256 && !bad; i++) if (g_out[i] != -1) { printf("FAIL it=%d wrote outside count (idx %d)\n", it, i); bad = 1; }
        if (it % 7 == 0) {
            memset(o, 0, sizeof(float) * d);
            matmul(o, x, W, n, d, GGUF_TYPE_F32);
            for (int r = 0; r < d; r++) if (fabsf(o[r] - ref[r]) > 1e-3f * (1 + fabsf(ref[r]))) { printf("FAIL it=%d matmul row %d %g vs %g\n", it, r, o[r], ref[r]); bad = 1; break; }
        }
        unsigned g = rnd() % 200;
        if (g == 0) usleep(3000);        /* long idle: workers fall into the condvar sleep */
        else if (g < 20) usleep(rnd() % 100);
    }
    if (!bad) bad = test_nested(300);
    if (!bad) bad = test_quant_rows();
    tensor_threadpool_free();
    printf("%s nt=%d dispatches=%d\n", bad ? "FAILED" : "OK", nt, N);
    return bad;
}
