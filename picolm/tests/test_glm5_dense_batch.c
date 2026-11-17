/* Batched dense FFN (glm5_dense_ffn_batch) vs the per-token GEMV loop it replaces.
 *
 * A GEMM sums in a different order than a GEMV, so each stage agrees only to float rounding
 * (~1e-7 relative) -- but the down projection re-quantizes its input to Q8_K, and a 1e-7 difference
 * in a value sitting on a rounding boundary flips it by one quantization step. That moves every
 * output of that token by up to ~1e-3 of the largest activation: ordinary Q8 noise, not an error.
 * So the checks are:  gate/up and SwiGLU agree to 1e-5;  the down GEMM agrees to 1e-5 when fed
 * identical input;  end to end, RMS error <= 2e-3 and max error <= 1e-2 of the largest output.
 * NT=1 must decline. Random IQ6_K weights at a reduced size. */
#include "model_glm5.c"
#include <time.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }
static unsigned long long rs = 88172645463325252ULL;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 16); }
static float frnd(void) { return (float)((int)(rnd() % 20001) - 10000) / 10000.f; }
static void *make_w(size_t rows, int n) {
    const int nb = n / 256; block_iq6_k *W = malloc(rows * nb * sizeof(block_iq6_k));
    for (size_t i = 0; i < rows * nb; i++) { block_iq6_k *b = &W[i];
        for (size_t j = 0; j < sizeof *b; j++) ((uint8_t *)b)[j] = (uint8_t)rnd();
        b->d = 0x1800 + (rnd() & 0x3f); for (int g = 0; g < 16; g++) b->scales[g] = (int8_t)((int)(rnd() % 64) - 32); }
    return W;
}
static double maxrel(const float *a, const float *b, size_t n) {
    double d = 0, m = 0; for (size_t i = 0; i < n; i++) { d = fmax(d, fabs((double)a[i] - b[i])); m = fmax(m, fabs(a[i])); }
    return d / (m + 1e-30);
}
static int run(int NT, int nth) {
    const int D = 512, FF = 1536;
    tensor_set_threads(nth); tensor_threadpool_init(nth);
    model_t *m = calloc(1, sizeof *m); m->config.n_embd = D; m->config.n_ffn = FF;
    run_state_t *s = &m->state; s->hb = calloc(FF, 4); s->hb2 = calloc(FF, 4);
    layer_weights_t lw; memset(&lw, 0, sizeof lw);
    lw.ffn_gate = make_w(FF, D); lw.type_ffn_gate = GGUF_TYPE_IQ6_K;
    lw.ffn_up   = make_w(FF, D); lw.type_ffn_up   = GGUF_TYPE_IQ6_K;
    lw.ffn_down = make_w(D, FF); lw.type_ffn_down = GGUF_TYPE_IQ6_K;
    float *cur = malloc((size_t)NT * D * 4); for (int i = 0; i < NT * D; i++) cur[i] = frnd() * 0.3f;
    float *oA = malloc((size_t)NT * D * 4), *oB = malloc((size_t)NT * D * 4);
    double t0 = get_time_ms();
    for (int t = 0; t < NT; t++) glm5_dense_ffn(m, &lw, s, cur + (size_t)t * D, oA + (size_t)t * D);
    double t1 = get_time_ms();
    int ran = glm5_dense_ffn_batch(m, &lw, cur, oB, NT);
    double t2 = get_time_ms();
    if (NT < 2) { printf("NT=%-3d -j%-2d  %s (declines)\n", NT, nth, ran ? "FAIL" : "ok"); return ran != 0; }

    int bad = !ran;
    /* stage checks with identical inputs */
    float *g = malloc((size_t)NT*FF*4), *u = malloc((size_t)NT*FF*4), *gs = malloc((size_t)NT*FF*4), *us = malloc((size_t)NT*FF*4);
    matmul_batch(g, cur, NT, lw.ffn_gate, D, FF, GGUF_TYPE_IQ6_K); matmul_batch(u, cur, NT, lw.ffn_up, D, FF, GGUF_TYPE_IQ6_K);
    for (int t = 0; t < NT; t++) { matmul(gs + (size_t)t*FF, cur + (size_t)t*D, lw.ffn_gate, D, FF, GGUF_TYPE_IQ6_K);
                                   matmul(us + (size_t)t*FF, cur + (size_t)t*D, lw.ffn_up,   D, FF, GGUF_TYPE_IQ6_K); }
    const double r_gate = maxrel(gs, g, (size_t)NT*FF), r_up = maxrel(us, u, (size_t)NT*FF);
    float *h = malloc((size_t)NT*FF*4), *hb = malloc((size_t)NT*FF*4);
    for (int i = 0; i < NT*FF; i++) h[i] = gs[i] / (1.0f + expf(-gs[i])) * us[i];
    memcpy(hb, g, (size_t)NT*FF*4); { ep_swiglu_t sw = { hb, u, FF, NT, 0.0f }; for (int b = 0; b < (NT + 7) / 8; b++) ep_swiglu_task(b, &sw); }
    const double r_sw = maxrel(h, hb, (size_t)NT*FF);
    float *d1 = malloc((size_t)NT*D*4), *d2 = malloc((size_t)NT*D*4);
    matmul_batch(d1, h, NT, lw.ffn_down, FF, D, GGUF_TYPE_IQ6_K);
    for (int t = 0; t < NT; t++) matmul(d2 + (size_t)t*D, h + (size_t)t*FF, lw.ffn_down, FF, D, GGUF_TYPE_IQ6_K);
    const double r_dn = maxrel(d2, d1, (size_t)NT*D);
    if (r_gate > 1e-5 || r_up > 1e-5 || r_sw > 1e-5 || r_dn > 1e-5) bad = 1;
    /* end to end */
    double se = 0, sa = 0, mx = 0, am = 0; int nf = 0;
    for (size_t i = 0; i < (size_t)NT * D; i++) { if (!isfinite(oA[i]) || !isfinite(oB[i])) nf++;
        double d = (double)oA[i] - oB[i]; se += d*d; sa += (double)oA[i]*oA[i]; mx = fmax(mx, fabs(d)); am = fmax(am, fabs(oA[i])); }
    const double rms = sqrt(se / (sa + 1e-30)), rmax = mx / (am + 1e-30);
    if (nf || am == 0 || rms > 2e-3 || rmax > 1e-2) bad = 1;
    printf("NT=%-3d -j%-2d  %s  stages: gate %.0e up %.0e swiglu %.0e down %.0e | end-to-end rms %.1e max %.1e | per-token %.1f ms batched %.1f ms\n",
           NT, nth, bad ? "FAIL" : "ok", r_gate, r_up, r_sw, r_dn, rms, rmax, t1 - t0, t2 - t1);
    return bad;
}
int main(void) {
    fp16_table_init(); int f = 0;
    const int nts[] = { 1, 2, 5, 37, 100 };
    for (int ti = 0; ti < 2; ti++) for (unsigned i = 0; i < 5; i++) f += run(nts[i], ti ? 8 : 1);
    puts(f ? "FAIL" : "PASS"); return f != 0;
}
