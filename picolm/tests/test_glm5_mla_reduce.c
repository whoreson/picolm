/* Bit-exactness + timing: single-pass head-block MLA reduce vs the per-head
 * glm5_mla_reduce_head() reference, dense and DSA-selected positions. */
#include "model_glm5.c"
#include <time.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }
static unsigned long long rs = 88172645463325252ULL;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (float)((double)(rs >> 11) / 9007199254740992.0) * 2.f - 1.f; }

int main(void) {
    const int KVL = 512, NKV = 7111, HB[] = {1, 2, 3, 4, 8, 16};
    float *cache = malloc((size_t)NKV * KVL * 4);
    for (size_t i = 0; i < (size_t)NKV * KVL; i++) cache[i] = rnd();
    int *sel = malloc(NKV * sizeof(int)); int nsel = 0;
    for (int i = 0; i < NKV; i += 3) sel[nsel++] = i;   /* DSA-style sparse list */
    int fails = 0;
    for (int dense = 1; dense >= 0; dense--) {
        const int n = dense ? NKV : nsel;
        for (unsigned b = 0; b < sizeof HB / sizeof *HB; b++) {
            const int hn = HB[b];
            float *s0 = malloc((size_t)hn * n * 4), *s1 = malloc((size_t)hn * n * 4);
            float *cA = malloc((size_t)hn * KVL * 4), *cB = malloc((size_t)hn * KVL * 4);
            for (size_t i = 0; i < (size_t)hn * n; i++) s0[i] = s1[i] = rnd() * 12.f; /* peaky softmax: exercises p<1e-10 skip */
            memset(cA, 0xAB, (size_t)hn*KVL*4); memset(cB, 0xCD, (size_t)hn*KVL*4);
            double t0 = get_time_ms();
            for (int k = 0; k < hn; k++)
                glm5_mla_reduce_head(cA + (size_t)k*KVL, s0 + (size_t)k*n, cache, dense ? NULL : sel, n, dense, KVL);
            double t1 = get_time_ms();
            glm5_mla_reduce_block(cB, s1, n, hn, cache, dense ? NULL : sel, dense, KVL);
            double t2 = get_time_ms();
            int bad = memcmp(cA, cB, (size_t)hn*KVL*4) || memcmp(s0, s1, (size_t)hn*n*4);
            printf("%-6s hb=%-2d %s   per-head %6.2f ms  block %6.2f ms  (%.2fx)\n",
                   dense ? "dense" : "sparse", hn, bad ? "MISMATCH" : "bit-identical",
                   t1 - t0, t2 - t1, (t1 - t0) / (t2 - t1));
            fails += bad;
            free(s0); free(s1); free(cA); free(cB);
        }
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
