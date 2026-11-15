/* Blocked flash-style prefill attention (glm5_mla_attn_pfa) vs the existing
 * per-(head,token) core (glm5_mla_attn_batch with PICOLM_GLM5_PFA unset) and a
 * double-precision reference. Not bit-identical by design (online softmax); the
 * new kernel must be at least as accurate as the old one. */
#include "model_glm5.c"
#include <time.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }
static unsigned long long rs = 88172645463325252ULL;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (float)((double)(rs >> 11) / 9007199254740992.0) * 2.f - 1.f; }

static void ref_attn(const float *q, const float *cache, double *out, int NH, int KVL,
                     int n_tokens, int start_pos, float scale) {
    double *sc = malloc((size_t)(start_pos + n_tokens) * sizeof(double));
    for (int t = 0; t < n_tokens; t++) for (int h = 0; h < NH; h++) {
        const float *qh = q + ((size_t)t * NH + h) * KVL; const int n = start_pos + t + 1;
        double mx = -1e300;
        for (int p = 0; p < n; p++) { double d = 0; for (int i = 0; i < KVL; i++) d += (double)qh[i] * cache[(size_t)p*KVL+i];
            sc[p] = d * scale; if (sc[p] > mx) mx = sc[p]; }
        double sum = 0; for (int p = 0; p < n; p++) { sc[p] = exp(sc[p] - mx); sum += sc[p]; }
        double *o = out + ((size_t)t * NH + h) * KVL;
        for (int i = 0; i < KVL; i++) o[i] = 0;
        for (int p = 0; p < n; p++) { double w = sc[p] / sum; for (int i = 0; i < KVL; i++) o[i] += w * cache[(size_t)p*KVL+i]; }
    }
    free(sc);
}

static int run_case(int NH, int KVL, int n_tokens, int start_pos, int do_ref, int nth, int timing) {
    const int MAXSEQ = start_pos + n_tokens + 8;
    model_t *m = calloc(1, sizeof *m); model_config_t *c = &m->config;
    c->n_heads = NH; c->glm5_kv_lora = KVL; c->glm5_mla_head_dim = 16; c->max_seq_len = MAXSEQ;
    run_state_t *s = &m->state;
    s->glm5.mla_kv = calloc((size_t)MAXSEQ * KVL, sizeof(float));
    for (size_t i = 0; i < (size_t)MAXSEQ * KVL; i++) s->glm5.mla_kv[i] = rnd();
    const size_t QN = (size_t)n_tokens * NH * KVL;
    float *q = malloc(QN * 4), *oA = malloc(QN * 4), *oB = malloc(QN * 4);
    for (size_t i = 0; i < QN; i++) q[i] = rnd() * 3.f;
    memset(oA, 0xAB, QN*4); memset(oB, 0xCD, QN*4);
    tensor_set_threads(nth); tensor_threadpool_init(nth);
    double t0 = get_time_ms();
    glm5_mla_attn_batch(c, s, 0, q, oA, n_tokens, start_pos);            /* existing path */
    double t1 = get_time_ms();
    int ran = glm5_mla_attn_pfa(c, s, 0, q, oB, n_tokens, start_pos);    /* new path */
    double t2 = get_time_ms();
    double dmax = 0, amax = 0, eA = 0, eB = 0;
    for (size_t i = 0; i < QN; i++) { double d = fabs((double)oA[i] - oB[i]); if (d > dmax) dmax = d;
        if (fabs(oA[i]) > amax) amax = fabs(oA[i]); }
    int bad = !ran || !(dmax < 2e-4 * (amax + 1e-6)) || !isfinite(dmax);
    if (do_ref) {
        double *r = malloc(QN * sizeof(double));
        ref_attn(q, s->glm5.mla_kv, r, NH, KVL, n_tokens, start_pos, 1.0f / sqrtf(16.f));
        for (size_t i = 0; i < QN; i++) { double a = fabs(oA[i] - r[i]), b = fabs(oB[i] - r[i]);
            if (a > eA) eA = a;
            if (b > eB) eB = b; }
        if (eB > 4 * eA + 1e-5) bad = 1;      /* new must be about as accurate as old */
        free(r);
    }
    printf("NH=%-2d KVL=%d tok=%-3d start=%-5d  old-vs-new max|d|=%.2e (|o|max %.2f)", NH, KVL, n_tokens, start_pos, dmax, amax);
    if (do_ref) printf("  err vs double: old %.2e new %.2e", eA, eB);
    if (timing) printf("  | old %.0f ms  new %.0f ms (%.1fx)", t1-t0, t2-t1, (t1-t0)/(t2-t1));
    printf("  %s\n", bad ? "FAIL" : "ok");
    free(q); free(oA); free(oB); free(s->glm5.mla_kv); free(m);
    return bad;
}

int main(void) {
    int f = 0;
    f += run_case(16, 512,  1,    0, 1, 4, 0);   /* single token, single key */
    f += run_case(16, 512,  8,    0, 1, 4, 0);   /* pure diagonal block */
    f += run_case(12, 512, 13,   63, 1, 4, 0);   /* NH not a multiple of 8; start at a block edge */
    f += run_case(16, 512, 37, 1000, 1, 4, 0);   /* ragged token tail, many key blocks */
    f += run_case( 8, 512, 64,  129, 1, 8, 0);
    f += run_case(64, 512, 16, 3000, 0, 1, 1);   /* real head count, single thread timing */
    puts(f ? "FAIL" : "PASS");
    return f != 0;
}
