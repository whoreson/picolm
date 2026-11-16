/* tensor_matmul_batch_multi must equal matmul_batch() bit-for-bit, for every job, at any thread
 * count, token count and chunking. Jobs mimic the GLM MoE expert shapes (gate/up: n=4096,d=2048;
 * down: n=2048,d=4096) with random IQ6_K weights. */
#include "model.h"
#include "tensor.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }
static unsigned long long rs = 88172645463325252ULL;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 16); }
static float frnd(void) { return (float)((int)(rnd() % 20001) - 10000) / 10000.f; }

static void *make_w(int d, int n) {
    const int nb = n / 256;
    block_iq6_k *W = malloc((size_t)d * nb * sizeof(block_iq6_k));
    for (size_t i = 0; i < (size_t)d * nb; i++) { block_iq6_k *b = &W[i];
        for (size_t j = 0; j < sizeof *b; j++) ((uint8_t *)b)[j] = (uint8_t)rnd();
        b->d = 0x2c00 + (rnd() & 0xff); for (int g = 0; g < 16; g++) b->scales[g] = (int8_t)((int)(rnd() % 64) - 32); }
    return W;
}

int main(int argc, char **argv) {
    fp16_table_init();
    int fails = 0;
    const int nths[] = { 1, 4, 8, 24 };
    const int ncols[] = { 2, 3, 4, 5, 8, 13, 16, 17, 31, 40, 100 };
    const int NJ = 11;
    void *Wg[2]; Wg[0] = make_w(2048, 4096); Wg[1] = make_w(4096, 2048);
    for (int ti = 0; ti < 4; ti++) {
        const int nth = nths[ti];
        tensor_set_threads(nth); tensor_threadpool_init(nth);
        mm_bjob_t jobs[NJ]; float *refs[NJ], *outs[NJ], *xs[NJ];
        for (int j = 0; j < NJ; j++) {
            const int shape = j & 1, n = shape ? 2048 : 4096, d = shape ? 4096 : 2048, nbat = ncols[j];
            xs[j] = malloc((size_t)nbat * n * 4); for (int i = 0; i < nbat * n; i++) xs[j][i] = frnd() * 2.f;
            refs[j] = malloc((size_t)nbat * d * 4); outs[j] = malloc((size_t)nbat * d * 4);
            memset(outs[j], 0xAB, (size_t)nbat * d * 4);
            jobs[j] = (mm_bjob_t){ outs[j], xs[j], Wg[shape], nbat, n, d, GGUF_TYPE_IQ6_K };
        }
        /* reference at the same thread count: this is exactly what the MoE code runs today */
        double t0 = get_time_ms();
        for (int j = 0; j < NJ; j++) matmul_batch(refs[j], xs[j], jobs[j].n_batch, jobs[j].W, jobs[j].n, jobs[j].d, GGUF_TYPE_IQ6_K);
        double t1 = get_time_ms();
        int ran = tensor_matmul_batch_multi(jobs, NJ);
        double t2 = get_time_ms();
        int bad = !ran;
        for (int j = 0; j < NJ && !bad; j++)
            if (memcmp(refs[j], outs[j], (size_t)jobs[j].n_batch * jobs[j].d * 4)) { bad = 1; printf("  job %d (cols=%d) MISMATCH\n", j, jobs[j].n_batch); }
        printf("threads=%-2d  %d jobs  ran=%d  %s   matmul_batch loop %.1f ms   batch_multi %.1f ms\n",
               nth, NJ, ran, bad ? "MISMATCH" : "bit-identical", t1 - t0, t2 - t1);
        fails += bad;
        for (int j = 0; j < NJ; j++) { free(xs[j]); free(refs[j]); free(outs[j]); }
    }
    /* declines cleanly */
    { mm_bjob_t bad1 = { NULL, NULL, Wg[0], 1, 4096, 2048, GGUF_TYPE_IQ6_K };
      float o[4096], x[4096]; bad1.out = o; bad1.x = x;
      if (tensor_matmul_batch_multi(&bad1, 1) != 0) { puts("n_batch=1 should decline"); fails++; } }
    puts(fails ? "FAIL" : "PASS");
    (void)argc; (void)argv;
    return fails != 0;
}
