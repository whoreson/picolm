/* PICOLM_POOL_GAPS accounting: known serial gaps between dispatches must be
 * recovered, and gap + main-slot + wait must account for the wall time. */
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }
static void task(int i, void *ctx) { (void)i; (void)ctx; usleep(100); }

int main(void) {
    setenv("PICOLM_POOL_GAPS", "1", 1);
    tensor_set_threads(4); tensor_threadpool_init(4);
    tensor_parallel_for(4, task, NULL);          /* warm the pool */
    tensor_pool_gap_reset();
    const int N = 200;
    double t0 = now_ms();
    for (int i = 0; i < N; i++) {
        usleep(300);                              /* the "serial main-thread work" */
        tensor_parallel_for(4, task, NULL);
    }
    double wall = now_ms() - t0;
    double gap, mn, wt, wk; long n;
    if (!tensor_pool_gap_take(&gap, &mn, &wt, &wk, &n)) { puts("no data"); return 2; }
    double sum = gap + mn + wt;
    printf("dispatches=%ld  wall=%.1f ms   gap=%.1f  main=%.1f  wait=%.1f  (sum %.1f)  wake=%.1f us\n",
           n, wall, gap, mn, wt, sum, wk);
    int ok = (n == N) && gap >= N * 0.3 * 0.95 && gap < N * 0.3 * 4 && sum > 0.85 * wall && sum < 1.02 * wall;
    puts(ok ? "PASS" : "FAIL");
    return !ok;
}
