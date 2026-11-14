/* Bit-exactness + timing test: head-parallel KDA core vs the serial reference.
 *
 * glm5_kda_tail() (serial, the existing prefill/reference implementation) and
 * glm5_kda_tail_par() (one task per head) must produce byte-identical
 * head_out / conv_out / g / beta AND leave byte-identical kda_state and
 * conv_state, over several consecutive steps (state carries across steps, so a
 * race or a mis-sliced channel shows up quickly). Included directly so the
 * static kernels are reachable. */
#include "model_glm5.c"
#include <time.h>

double get_time_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }

static unsigned long long rs = 88172645463325252ULL;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (float)((double)(rs >> 11) / 9007199254740992.0) * 2.f - 1.f; }
static void fill(float *p, size_t n, float sc) { for (size_t i = 0; i < n; i++) p[i] = rnd() * sc; }

int main(int argc, char **argv) {
    const int H = 64, KD = 128, CK = 4, STEPS = 6;
    const int nth = argc > 1 ? atoi(argv[1]) : 8;
    const int DI = H * KD, CD = 3 * DI;
    tensor_set_threads(nth); tensor_threadpool_init(nth);

    model_t *m = calloc(1, sizeof *m);
    m->config.glm5_n_kda_heads = H; m->config.glm5_kda_dim = KD;
    m->config.ssm_d_conv = CK; m->config.n_embd = 64;
    m->config.glm5_kda_lower_bound = -5.0f; m->config.rms_norm_eps = 1e-5f;

    glm5_layer_weights_t gw; memset(&gw, 0, sizeof gw);
    float *a = malloc(H*4), *dtb = malloc(DI*4), *nw = malloc(KD*4);
    float *cq = malloc((size_t)CK*DI*4), *ck = malloc((size_t)CK*DI*4), *cv = malloc((size_t)CK*DI*4);
    for (int i = 0; i < H; i++) a[i] = -fabsf(rnd()) - 0.05f;
    fill(dtb, DI, 1.f); for (int i = 0; i < KD; i++) nw[i] = 1.f + rnd()*0.2f;
    fill(cq, (size_t)CK*DI, .6f); fill(ck, (size_t)CK*DI, .6f); fill(cv, (size_t)CK*DI, .6f);
    gw.kda_a = a; gw.kda_dt_bias = dtb; gw.kda_norm = nw;
    gw.kda_conv_q = cq; gw.kda_conv_k = ck; gw.kda_conv_v = cv;
    layer_weights_t lw; memset(&lw, 0, sizeof lw);

    /* two independent run states with identical initial KDA state */
    run_state_t *sA = &m->state;
    run_state_t *sB = calloc(1, sizeof *sB);
    size_t SS = (size_t)H*KD*KD, CS = (size_t)(CK-1)*CD;
    float *stA = malloc(SS*4), *stB = malloc(SS*4), *csA = malloc(CS*4), *csB = malloc(CS*4);
    fill(stA, SS, .3f); memcpy(stB, stA, SS*4);
    fill(csA, CS, .5f); memcpy(csB, csA, CS*4);
    sA->glm5.kda_state[0] = stA; sA->glm5.kda_conv_state[0] = csA;
    sB->glm5.kda_state[0] = stB; sB->glm5.kda_conv_state[0] = csB;

    float *q = malloc(DI*4), *k = malloc(DI*4), *v = malloc(DI*4), *z = malloc(DI*4), *raw = malloc(DI*4);
    float *br = malloc(H*4);
    float *coA = malloc(CD*4), *coB = malloc(CD*4), *gA = malloc(DI*4), *gB = malloc(DI*4);
    float *hoA = malloc(DI*4), *hoB = malloc(DI*4), *bA = malloc(H*4), *bB = malloc(H*4), *vn = malloc(KD*4);

    int fails = 0; double tS = 0, tP = 0;
    for (int step = 0; step < STEPS; step++) {
        fill(q, DI, 2.f); fill(k, DI, 2.f); fill(v, DI, 2.f); fill(z, DI, 3.f);
        fill(raw, DI, 2.f); fill(br, H, 2.f);
        memset(hoA, 0xAB, DI*4); memset(hoB, 0xCD, DI*4);
        double t0 = get_time_ms();
        glm5_kda_tail(m, &gw, &lw, sA, 0, step, q, k, v, z, raw, br, coA, gA, hoA, vn, bA, NULL, 0);
        double t1 = get_time_ms();
        glm5_kda_tail_par(m, &gw, &lw, sB, 0, step, q, k, v, z, raw, br, coB, gB, hoB, NULL, bB, NULL, 0);
        double t2 = get_time_ms();
        tS += t1 - t0; tP += t2 - t1;
        int bad = memcmp(hoA, hoB, DI*4) || memcmp(coA, coB, CD*4) || memcmp(gA, gB, DI*4) ||
                  memcmp(bA, bB, H*4)    || memcmp(stA, stB, SS*4) || memcmp(csA, csB, CS*4);
        printf("step %d: %s\n", step, bad ? "MISMATCH" : "bit-identical");
        fails += bad;
    }
    printf("threads=%d  serial %.2f ms/step  parallel %.2f ms/step  (%.2fx)\n",
           nth, tS/STEPS, tP/STEPS, tS/tP);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
