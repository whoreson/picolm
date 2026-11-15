/* Split-KV decode attention (glm5_mla_splitkv) vs the F16 head-block path and a
 * double-precision reference. The scores are bit-identical to the head-block
 * path by construction (same FMA order); the softmax/PV are summed per key chunk,
 * so the outputs agree to float rounding, not bitwise. */
#include "model_glm5.c"
#include <time.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }
static unsigned long long rs = 88172645463325252ULL;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (float)((double)(rs >> 11) / 9007199254740992.0) * 2.f - 1.f; }

static void ref_attn(const float *q, const uint16_t *c16, double *out, int NH, int KVL, int n, float scale) {
    double *sc = malloc((size_t)n * sizeof(double));
    for (int h = 0; h < NH; h++) {
        double mx = -1e300;
        for (int p = 0; p < n; p++) { double d = 0;
            for (int i = 0; i < KVL; i++) d += (double)q[(size_t)h*KVL+i] * fp16_to_fp32_lookup(c16[(size_t)p*KVL+i]);
            sc[p] = d * scale; if (sc[p] > mx) mx = sc[p]; }
        double sum = 0; for (int p = 0; p < n; p++) { sc[p] = exp(sc[p] - mx); sum += sc[p]; }
        double *o = out + (size_t)h * KVL; for (int i = 0; i < KVL; i++) o[i] = 0;
        for (int p = 0; p < n; p++) { double w = sc[p] / sum;
            for (int i = 0; i < KVL; i++) o[i] += w * fp16_to_fp32_lookup(c16[(size_t)p*KVL+i]); }
    }
    free(sc);
}

static int run_case(int NH, int KVL, int n, int nchunks, int do_ref, int nth, int timing) {
    uint16_t *c16 = malloc((size_t)n * KVL * 2);
    float *c32 = malloc((size_t)n * KVL * 4);
    for (size_t i = 0; i < (size_t)n * KVL; i++) { c16[i] = fp32_to_fp16(rnd()); c32[i] = fp16_to_fp32_lookup(c16[i]); }
    const size_t QN = (size_t)NH * KVL;
    float *q = malloc(QN * 4), *oA = malloc(QN * 4), *oB = malloc(QN * 4);
    for (size_t i = 0; i < QN; i++) q[i] = rnd() * 3.f;
    memset(oA, 0xAB, QN * 4); memset(oB, 0xCD, QN * 4);
    const float scale = 0.25f;
    tensor_set_threads(nth); tensor_threadpool_init(nth);

    /* existing path: head blocks of 2 on the F16 shadow */
    const int hb = 2, nb = (NH + hb - 1) / hb;
    float *scr = malloc((size_t)nb * hb * n * 4);
    glm5_mla_thr_t tc = { c32, c16, NULL, q, oA, scr, NH, KVL, n, 0, 1, hb, scale };
    double t0 = get_time_ms();
    tensor_parallel_for(nb, glm5_mla_head_block_task, &tc);
    double t1 = get_time_ms();
    int ran = glm5_mla_splitkv(c16, q, oB, NH, KVL, n, scale, nchunks);
    double t2 = get_time_ms();

    double dmax = 0, amax = 0, eA = 0, eB = 0;
    for (size_t i = 0; i < QN; i++) { double d = fabs((double)oA[i] - oB[i]); if (d > dmax) dmax = d;
        if (fabs(oA[i]) > amax) amax = fabs(oA[i]); }
    int bad = !ran || !isfinite(dmax) || !(dmax < 2e-4 * (amax + 1e-6));
    if (do_ref) {
        double *r = malloc(QN * sizeof(double));
        ref_attn(q, c16, r, NH, KVL, n, scale);
        for (size_t i = 0; i < QN; i++) { double a = fabs(oA[i] - r[i]), b = fabs(oB[i] - r[i]);
            if (a > eA) eA = a;
            if (b > eB) eB = b; }
        if (eB > 4 * eA + 1e-5) bad = 1;
        free(r);
    }
    printf("NH=%-2d n_kv=%-5d chunks=%-2d  old-vs-new max|d|=%.2e", NH, n, nchunks, dmax);
    if (do_ref) printf("  err vs double: old %.2e new %.2e", eA, eB);
    if (timing) printf("  | head-block %.2f ms  split-kv %.2f ms (%.1fx)", t1 - t0, t2 - t1, (t1 - t0) / (t2 - t1));
    printf("  %s\n", bad ? "FAIL" : "ok");
    free(c16); free(c32); free(q); free(oA); free(oB); free(scr);
    return bad;
}

int main(void) {
    fp16_table_init();   /* fp16_to_fp32_lookup() is all zeros until this runs */
    int f = 0;
    f += run_case(64, 512,    1,  1, 1, 4, 0);   /* single key */
    f += run_case(64, 512,    5,  8, 1, 4, 0);   /* more chunks than keys -> clamped */
    f += run_case(64, 512,   64,  3, 1, 4, 0);
    f += run_case(12, 512,  513,  5, 1, 4, 0);   /* NH not a multiple of 4 (tail heads) */
    f += run_case(13, 512, 1000, 24, 1, 4, 0);   /* odd NH: single-head PV tail */
    f += run_case(64, 512, 1000, 64, 1, 8, 0);
    f += run_case(64, 512, 2047,  1, 1, 2, 0);   /* one chunk == plain two-pass softmax */
    f += run_case(64, 512, 7111, 24, 0, 1, 1);   /* real shape, single-thread timing */
    f += run_case(64, 512, 7111, 24, 0, 1, 1);
    puts(f ? "FAIL" : "PASS");
    return f != 0;
}
