/* Parallel prefill loops must be bit-identical to the serial ones they replace:
 *  (a) glm5_kda_layer_par (one task per head, tokens inner) vs n_tokens sequential
 *      glm5_kda_tail_par calls -- outputs and the carried conv/recurrent state;
 *  (b) pf_run_hcpre / pf_run_hcpost (one task per token) vs the original serial loops. */
#include "model_glm5.c"
#include <time.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }
static unsigned long long rs = 88172645463325252ULL;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 16); }
static float frnd(void) { return (float)((int)(rnd() % 20001) - 10000) / 10000.f; }
static void fill(float *p, size_t n, float sc) { for (size_t i = 0; i < n; i++) p[i] = frnd() * sc; }

static int test_kda(int NT, int nth) {
    const int H = 64, KD = 128, CK = 4, DI = H * KD, CD = 3 * DI;
    tensor_set_threads(nth); tensor_threadpool_init(nth);
    model_t *m = calloc(1, sizeof *m);
    m->config.glm5_n_kda_heads = H; m->config.glm5_kda_dim = KD; m->config.ssm_d_conv = CK;
    m->config.n_embd = 64; m->config.glm5_kda_lower_bound = -5.0f; m->config.rms_norm_eps = 1e-5f;
    glm5_layer_weights_t gw; memset(&gw, 0, sizeof gw);
    float *a = malloc(H * 4), *dtb = malloc(DI * 4), *nw = malloc(KD * 4);
    float *cq = malloc((size_t)CK * DI * 4), *ck = malloc((size_t)CK * DI * 4), *cv = malloc((size_t)CK * DI * 4);
    for (int i = 0; i < H; i++) a[i] = -fabsf(frnd()) - 0.05f;
    fill(dtb, DI, 1.f); for (int i = 0; i < KD; i++) nw[i] = 1.f + frnd() * 0.2f;
    fill(cq, (size_t)CK * DI, .6f); fill(ck, (size_t)CK * DI, .6f); fill(cv, (size_t)CK * DI, .6f);
    gw.kda_a = a; gw.kda_dt_bias = dtb; gw.kda_norm = nw; gw.kda_conv_q = cq; gw.kda_conv_k = ck; gw.kda_conv_v = cv;
    layer_weights_t lw; memset(&lw, 0, sizeof lw);
    run_state_t *sA = &m->state, *sB = calloc(1, sizeof *sB);
    const size_t SS = (size_t)H * KD * KD, CS = (size_t)(CK - 1) * CD;
    float *stA = malloc(SS * 4), *stB = malloc(SS * 4), *csA = malloc(CS * 4), *csB = malloc(CS * 4);
    fill(stA, SS, .3f); memcpy(stB, stA, SS * 4); fill(csA, CS, .5f); memcpy(csB, csA, CS * 4);
    sA->glm5.kda_state[0] = stA; sA->glm5.kda_conv_state[0] = csA;
    sB->glm5.kda_state[0] = stB; sB->glm5.kda_conv_state[0] = csB;

    float *q = malloc((size_t)NT * DI * 4), *k = malloc((size_t)NT * DI * 4), *v = malloc((size_t)NT * DI * 4);
    float *z = malloc((size_t)NT * DI * 4), *raw = malloc((size_t)NT * DI * 4), *br = malloc((size_t)NT * H * 4);
    fill(q, (size_t)NT * DI, 2.f); fill(k, (size_t)NT * DI, 2.f); fill(v, (size_t)NT * DI, 2.f);
    fill(z, (size_t)NT * DI, 3.f); fill(raw, (size_t)NT * DI, 2.f); fill(br, (size_t)NT * H, 2.f);
    float *coA = malloc((size_t)NT * CD * 4), *coB = malloc((size_t)NT * CD * 4);
    float *hoA = malloc((size_t)NT * DI * 4), *hoB = malloc((size_t)NT * DI * 4);
    memset(hoA, 0xAB, (size_t)NT * DI * 4); memset(hoB, 0xCD, (size_t)NT * DI * 4);
    float *gA = malloc(DI * 4), *gB = malloc(DI * 4), *bA = malloc(H * 4), *bB = malloc(H * 4), *vn = malloc(KD * 4);

    double t0 = get_time_ms();
    for (int t = 0; t < NT; t++)
        glm5_kda_tail_par(m, &gw, &lw, sA, 0, t, q + (size_t)t*DI, k + (size_t)t*DI, v + (size_t)t*DI,
                          z + (size_t)t*DI, raw + (size_t)t*DI, br + (size_t)t*H,
                          coA + (size_t)t*CD, gA, hoA + (size_t)t*DI, vn, bA, NULL, 0);
    double t1 = get_time_ms();
    int ran = glm5_kda_layer_par(m, &gw, sB, 0, NT, q, k, v, z, raw, br, coB, gB, hoB, bB);
    double t2 = get_time_ms();
    int bad = !ran || memcmp(hoA, hoB, (size_t)NT * DI * 4) || memcmp(coA, coB, (size_t)NT * CD * 4) ||
              memcmp(stA, stB, SS * 4) || memcmp(csA, csB, CS * 4);
    printf("kda   NT=%-3d -j%-2d  %s   per-token dispatch %.1f ms   head-major layer %.1f ms\n",
           NT, nth, bad ? "MISMATCH" : "bit-identical", t1 - t0, t2 - t1);
    return bad;
}

static int test_hc(int NT, int nth, int is_attn) {
    const int S = 4, D = 512, M = (2 + S) * S;
    tensor_set_threads(nth); tensor_threadpool_init(nth);
    model_t *m = calloc(1, sizeof *m); model_config_t *c = &m->config;
    c->glm5_hc = S; c->n_embd = D; c->glm5_hc_eps = 1e-6f; c->glm5_sinkhorn_iters = 3; c->rms_norm_eps = 1e-5f;
    run_state_t *s = &m->state; s->glm5.hc_scratch = calloc((size_t)S * D + M, 4);
    glm5_layer_weights_t gw; memset(&gw, 0, sizeof gw);
    const int nb = (S * D) / 256;
    block_iq6_k *W = malloc((size_t)M * nb * sizeof(block_iq6_k));
    for (size_t i = 0; i < (size_t)M * nb; i++) { block_iq6_k *b = &W[i];
        for (size_t j = 0; j < sizeof *b; j++) ((uint8_t *)b)[j] = (uint8_t)rnd();
        b->d = 0x1800 + (rnd() & 0x3f); for (int g = 0; g < 16; g++) b->scales[g] = (int8_t)((int)(rnd() % 64) - 32); }
    float base[M], scale[3]; fill(base, M, 0.5f); scale[0] = 0.3f; scale[1] = 0.4f; scale[2] = 0.5f;
    if (is_attn) { gw.hc_attn_fn = W; gw.type_hc_attn_fn = GGUF_TYPE_IQ6_K; gw.hc_attn_base = base; gw.hc_attn_scale = scale; }
    else         { gw.hc_ffn_fn  = W; gw.type_hc_ffn_fn  = GGUF_TYPE_IQ6_K; gw.hc_ffn_base  = base; gw.hc_ffn_scale  = scale; }
    float *nw = malloc(D * 4); for (int i = 0; i < D; i++) nw[i] = 1.f + frnd() * 0.1f;

    float *hx0 = malloc((size_t)NT * S * D * 4); fill(hx0, (size_t)NT * S * D, 1.f);
    float *hxA = malloc((size_t)NT * S * D * 4), *hxB = malloc((size_t)NT * S * D * 4);
    memcpy(hxA, hx0, (size_t)NT * S * D * 4); memcpy(hxB, hx0, (size_t)NT * S * D * 4);
    float *curA = malloc((size_t)NT * D * 4), *curB = malloc((size_t)NT * D * 4);
    float *resA = malloc((size_t)NT * S * D * 4), *resB = malloc((size_t)NT * S * D * 4);
    float *poA = malloc((size_t)NT * S * 4), *poB = malloc((size_t)NT * S * 4);
    float *coA = malloc((size_t)NT * S * S * 4), *coB = malloc((size_t)NT * S * S * 4);
    /* serial reference: the original loop */
    for (int t = 0; t < NT; t++) {
        float *hc_x_t = hxA + (size_t)t*S*D, *cur_t = curA + (size_t)t*D, *res_t = resA + (size_t)t*S*D;
        float *post_t = poA + (size_t)t*S, *comb_t = coA + (size_t)t*S*S;
        memcpy(res_t, hc_x_t, (size_t)S*D*sizeof(float));
        glm5_hc_pre(m, &gw, is_attn, hc_x_t, cur_t, post_t, comb_t, s->glm5.hc_scratch);
        rmsnorm(cur_t, cur_t, nw, D, c->rms_norm_eps);
    }
    int ran = pf_run_hcpre(m, &gw, is_attn, NT, hxB, curB, resB, poB, coB, nw);
    int bad = !ran || memcmp(curA, curB, (size_t)NT*D*4) || memcmp(resA, resB, (size_t)NT*S*D*4) ||
              memcmp(poA, poB, (size_t)NT*S*4) || memcmp(coA, coB, (size_t)NT*S*S*4);
    /* hc_post */
    float *attn = malloc((size_t)NT * D * 4); fill(attn, (size_t)NT * D, 1.f);
    for (int t = 0; t < NT; t++)
        glm5_hc_post(m, attn + (size_t)t*D, resA + (size_t)t*S*D, poA + (size_t)t*S, coA + (size_t)t*S*S, hxA + (size_t)t*S*D);
    pf_run_hcpost(m, NT, hxB, resB, poB, coB, attn);
    bad |= memcmp(hxA, hxB, (size_t)NT*S*D*4) != 0;
    printf("hc_%s NT=%-3d -j%-2d  pre+post %s\n", is_attn ? "attn" : "ffn ", NT, nth, bad ? "MISMATCH" : "bit-identical");
    return bad;
}

int main(void) {
    fp16_table_init();
    int f = 0;
    const int nts[] = { 1, 2, 37, 130 };
    for (int ti = 0; ti < 3; ti++) {
        const int nth = ti == 0 ? 1 : (ti == 1 ? 4 : 24);
        for (unsigned i = 0; i < 4; i++) { f += test_kda(nts[i], nth); f += test_hc(nts[i], nth, i & 1); }
    }
    puts(f ? "FAIL" : "PASS");
    return f != 0;
}
