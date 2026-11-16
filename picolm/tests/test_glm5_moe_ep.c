/* Expert-parallel MoE prefill (glm5_moe_ffn_batch_ep) vs the reference (glm5_moe_ffn_batch_ref) on a
 * synthetic layer: random IQ6_K experts, F32 router with selection bias, optional shared expert.
 *   arg "perttoken": shared expert stays per token -> the WHOLE output must be bit-identical.
 *   default        : shared expert batched (GEMM instead of GEMV) -> routed part is checked bit-for-bit
 *                    with the shared expert disabled, the full output to float rounding.
 * Token counts are chosen to produce zero-, one- and many-token experts. */
#include "model_glm5.c"
#include <time.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }
static unsigned long long rs = 88172645463325252ULL;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 16); }
static float frnd(void) { return (float)((int)(rnd() % 20001) - 10000) / 10000.f; }

static void *make_w(size_t rows, int n) {
    const int nb = n / 256;
    block_iq6_k *W = malloc(rows * nb * sizeof(block_iq6_k));
    for (size_t i = 0; i < rows * nb; i++) { block_iq6_k *b = &W[i];
        for (size_t j = 0; j < sizeof *b; j++) ((uint8_t *)b)[j] = (uint8_t)rnd();
        b->d = 0x1800 + (rnd() & 0x3f); for (int g = 0; g < 16; g++) b->scales[g] = (int8_t)((int)(rnd() % 64) - 32); }
    return W;
}

static int run(int NT, int AK, int nth, int with_shexp, int exact) {
    const int D = 512, FF = 256, SFF = 256, NE = 24;
    tensor_set_threads(nth); tensor_threadpool_init(nth);
    model_t *m = calloc(1, sizeof *m); model_config_t *c = &m->config;
    c->n_embd = D; c->n_expert = NE; c->n_expert_used = AK; c->n_ff_exp = FF; c->n_ff_shexp = SFF;
    c->glm5_moe_scale = 1.5f;
    run_state_t *s = &m->state;
    s->expert_logits = calloc(NE, 4); s->hb_exp = calloc(SFF > FF ? SFF : FF, 4); s->hb2_exp = calloc(SFF > FF ? SFF : FF, 4);
    s->glm5.moe_down = calloc(D, 4);

    glm5_layer_weights_t gw; memset(&gw, 0, sizeof gw);
    layer_weights_t lw; memset(&lw, 0, sizeof lw);
    float *router = malloc((size_t)NE * D * 4); for (int i = 0; i < NE * D; i++) router[i] = frnd() * 0.5f;
    float *bias = malloc(NE * 4); for (int i = 0; i < NE; i++) bias[i] = frnd() * 0.2f;
    gw.exp_probs_b = bias;
    lw.ffn_gate_inp = router; lw.type_ffn_gate_inp = GGUF_TYPE_F32;
    lw.ffn_gate_exps = make_w((size_t)NE * FF, D); lw.type_ffn_gate_exps = GGUF_TYPE_IQ6_K;
    lw.ffn_up_exps   = make_w((size_t)NE * FF, D); lw.type_ffn_up_exps   = GGUF_TYPE_IQ6_K;
    lw.ffn_down_exps = make_w((size_t)NE * D, FF); lw.type_ffn_down_exps = GGUF_TYPE_IQ6_K;
    if (with_shexp) {
        lw.ffn_gate_shexp = make_w(SFF, D); lw.type_ffn_gate_shexp = GGUF_TYPE_IQ6_K;
        lw.ffn_up_shexp   = make_w(SFF, D); lw.type_ffn_up_shexp   = GGUF_TYPE_IQ6_K;
        lw.ffn_down_shexp = make_w(D, SFF); lw.type_ffn_down_shexp = GGUF_TYPE_IQ6_K;
    }
    float *cur = malloc((size_t)NT * D * 4); for (int i = 0; i < NT * D; i++) cur[i] = frnd() * 0.3f;
    float *oA = malloc((size_t)NT * D * 4), *oB = malloc((size_t)NT * D * 4);
    memset(oA, 0xAB, (size_t)NT * D * 4); memset(oB, 0xCD, (size_t)NT * D * 4);

    double t0 = get_time_ms();
    glm5_moe_ffn_batch_ref(m, &gw, &lw, s, cur, oA, NT);
    double t1 = get_time_ms();
    int ran = glm5_moe_ffn_batch_ep(m, &gw, &lw, s, cur, oB, NT);
    double t2 = get_time_ms();

    int bad = !ran, nonfinite = 0; double dmax = 0, amax = 0;
    for (size_t i = 0; i < (size_t)NT * D; i++) {
        if (!isfinite(oA[i]) || !isfinite(oB[i])) nonfinite++;
        double d = fabs((double)oA[i] - oB[i]); if (d > dmax) dmax = d; if (fabs(oA[i]) > amax) amax = fabs(oA[i]);
    }
    if (exact) { if (memcmp(oA, oB, (size_t)NT * D * 4)) bad = 1; }
    else if (!(dmax <= 2e-5 * (amax + 1e-6))) bad = 1;
    if (nonfinite || amax == 0.0) bad = 1;
    /* routing histogram, so the log shows the cases really occurred */
    int cnt[NE]; memset(cnt, 0, sizeof cnt);
    { const int D2 = D; (void)D2;
      for (int t = 0; t < NT; t++) {
          float lg[NE]; matmul(lg, cur + (size_t)t * D, router, D, NE, GGUF_TYPE_F32);
          float pr[NE]; for (int i = 0; i < NE; i++) { pr[i] = 1.f / (1.f + expf(-lg[i])); lg[i] = pr[i] + bias[i]; }
          for (int k = 0; k < AK; k++) { int b = 0; for (int i = 1; i < NE; i++) if (lg[i] > lg[b]) b = i; cnt[b]++; lg[b] = -1e30f; } } }
    int n0 = 0, n1 = 0, nm = 0, mx = 0; for (int e = 0; e < NE; e++) { if (cnt[e] == 0) n0++; else if (cnt[e] == 1) n1++; else nm++; if (cnt[e] > mx) mx = cnt[e]; }
    printf("NT=%-3d AK=%d -j%-2d shexp=%d  experts: %d empty %d single %d multi (max %d tok)  %s %s  ref %.1f ms  ep %.1f ms  max|d|=%.2e rel=%.1e\n",
           NT, AK, nth, with_shexp, n0, n1, nm, mx, exact ? "exact" : "approx", bad ? "FAIL" : "ok", t1 - t0, t2 - t1, dmax, dmax / (amax + 1e-30));
    return bad;
}

int main(int argc, char **argv) {
    fp16_table_init();
    const int per_token = argc > 1 && !strcmp(argv[1], "perttoken");
    if (per_token) setenv("PICOLM_GLM5_SHEXP_BATCH", "0", 1);
    int f = 0;
    const int nts[] = { 2, 3, 6, 11, 40, 97 };
    for (int ti = 0; ti < 2; ti++) {
        const int nth = ti ? 8 : 1;
        for (unsigned i = 0; i < sizeof nts / sizeof *nts; i++) {
            f += run(nts[i], 4, nth, 0, 1);                       /* routed only: always bit-identical */
            f += run(nts[i], 4, nth, 1, per_token);               /* with shared expert */
        }
        f += run(64, 6, nth, 1, per_token);
    }
    puts(f ? "FAIL" : "PASS");
    return f != 0;
}
