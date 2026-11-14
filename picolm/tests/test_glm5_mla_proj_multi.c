/* tensor_matmul_multi over the 64 per-head MLA projections (k_b: n=256,d=512;
 * v_b: n=512,d=256) must equal 64 sequential matmul() calls bit-for-bit. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "tensor.h"
#include "quant.h"
#include <time.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }
static unsigned long long rs = 88172645463325252ULL;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (float)((double)(rs >> 11) / 9007199254740992.0) * 2.f - 1.f; }

static int run(const char *name, int n, int d, int nth) {
    const int NH = 64;
    const size_t row = gguf_type_row_size(GGUF_TYPE_Q8_0, n), head = (size_t)d * row;
    uint8_t *W = malloc(head * NH);
    float *tmp = malloc((size_t)n * 4);
    for (int h = 0; h < NH; h++) for (int r = 0; r < d; r++) {
        for (int i = 0; i < n; i++) tmp[i] = rnd();
        quantize_row_q8_0(tmp, W + h*head + r*row, n);
    }
    float *x = malloc((size_t)NH*n*4), *oA = calloc((size_t)NH*d, 4), *oB = calloc((size_t)NH*d, 4);
    for (int i = 0; i < NH*n; i++) x[i] = rnd();
    double t0 = get_time_ms();
    for (int rep = 0; rep < 20; rep++)
        for (int h = 0; h < NH; h++) matmul(oA + h*d, x + h*n, W + h*head, n, d, GGUF_TYPE_Q8_0);
    double t1 = get_time_ms();
    mm_job_t jobs[64]; int ran = 1;
    for (int h = 0; h < NH; h++) { jobs[h].out = oB + h*d; jobs[h].x = x + h*n;
        jobs[h].W = W + h*head; jobs[h].n = n; jobs[h].d = d; jobs[h].qtype = GGUF_TYPE_Q8_0; }
    for (int rep = 0; rep < 20; rep++) ran &= tensor_matmul_multi(jobs, NH);
    double t2 = get_time_ms();
    int bad = memcmp(oA, oB, (size_t)NH*d*4) != 0;
    printf("%-4s -j%-2d  batch_ran=%d  %s   64x matmul %.3f ms   multi %.3f ms  (%.2fx)\n",
           name, nth, ran, bad ? "MISMATCH" : "bit-identical", (t1-t0)/20, (t2-t1)/20, (t1-t0)/(t2-t1));
    free(W); free(tmp); free(x); free(oA); free(oB);
    return bad || !ran;
}
int main(void) {
    int fails = 0;
    const int nths[] = {2, 8, 24};
    for (int i = 0; i < 3; i++) {
        tensor_set_threads(nths[i]); tensor_threadpool_init(nths[i]);
        fails += run("k_b", 256, 512, nths[i]);
        fails += run("v_b", 512, 256, nths[i]);
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
