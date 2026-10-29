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

static int *g_out; static int g_iter;
static void fill_task(int idx, void *ctx) { (void)ctx; g_out[idx] = idx * 7 + g_iter; }
static unsigned rs = 12345;
static unsigned rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

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
    tensor_threadpool_free();
    printf("%s nt=%d dispatches=%d\n", bad ? "FAILED" : "OK", nt, N);
    return bad;
}
