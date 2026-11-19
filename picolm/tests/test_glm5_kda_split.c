/* glm5_kda_layer_split (state-independent math in a token-parallel pre-pass, then a per-head
 * recurrence) must be bit-identical to glm5_kda_layer_par: outputs, conv planes, and the carried
 * recurrent + conv state -- including chunks shorter than the conv history (NT < CONV_K-1) and
 * several consecutive chunks (history and state cross chunk boundaries). */
#include "model_glm5.c"
#include <time.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }
static unsigned long long rs = 88172645463325252ULL;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 16); }
static float frnd(void) { return (float)((int)(rnd() % 20001) - 10000) / 10000.f; }
static void fill(float *p, size_t n, float sc) { for (size_t i = 0; i < n; i++) p[i] = frnd() * sc; }

static int run(const int *chunks, int nch, int nth, int H) {
    const int KD = 128, CK = 4, DI = H * KD, CD = 3 * DI;
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
    run_state_t *sA = &m->state, *sB = calloc(1, sizeof *sB);
    const size_t SS = (size_t)H * KD * KD, CS = (size_t)(CK - 1) * CD;
    float *stA = malloc(SS * 4), *stB = malloc(SS * 4), *csA = malloc(CS * 4), *csB = malloc(CS * 4);
    fill(stA, SS, .3f); memcpy(stB, stA, SS * 4); fill(csA, CS, .5f); memcpy(csB, csA, CS * 4);
    sA->glm5.kda_state[0] = stA; sA->glm5.kda_conv_state[0] = csA;
    sB->glm5.kda_state[0] = stB; sB->glm5.kda_conv_state[0] = csB;

    int bad = 0; double tA = 0, tB = 0, tok = 0;
    for (int c = 0; c < nch; c++) {
        const int NT = chunks[c];
        float *q = malloc((size_t)NT * DI * 4), *k = malloc((size_t)NT * DI * 4), *v = malloc((size_t)NT * DI * 4);
        float *z = malloc((size_t)NT * DI * 4), *raw = malloc((size_t)NT * DI * 4), *br = malloc((size_t)NT * H * 4);
        fill(q, (size_t)NT * DI, 2.f); fill(k, (size_t)NT * DI, 2.f); fill(v, (size_t)NT * DI, 2.f);
        fill(z, (size_t)NT * DI, 3.f); fill(raw, (size_t)NT * DI, 2.f); fill(br, (size_t)NT * H, 2.f);
        float *coA = malloc((size_t)NT * CD * 4), *coB = malloc((size_t)NT * CD * 4);
        float *hoA = malloc((size_t)NT * DI * 4), *hoB = malloc((size_t)NT * DI * 4);
        memset(hoA, 0xAB, (size_t)NT * DI * 4); memset(hoB, 0xCD, (size_t)NT * DI * 4);
        memset(coA, 0xAB, (size_t)NT * CD * 4); memset(coB, 0xCD, (size_t)NT * CD * 4);
        float *gA = malloc(DI * 4), *gB = malloc(DI * 4), *bA = malloc(H * 4), *bB = malloc(H * 4);
        double t0 = get_time_ms();
        int r1 = glm5_kda_layer_par(m, &gw, sA, 0, NT, q, k, v, z, raw, br, coA, gA, hoA, bA);
        double t1 = get_time_ms();
        int r2 = glm5_kda_layer_split(m, &gw, sB, 0, NT, q, k, v, z, raw, br, coB, gB, hoB, bB);
        double t2 = get_time_ms();
        tA += t1 - t0; tB += t2 - t1; tok += NT;
        int b = !r1 || !r2 || memcmp(hoA, hoB, (size_t)NT * DI * 4) || memcmp(coA, coB, (size_t)NT * CD * 4) ||
                memcmp(stA, stB, SS * 4) || memcmp(csA, csB, CS * 4);
        if (b) printf("   chunk %d (NT=%d) MISMATCH\n", c, NT);
        bad |= b;
        free(q); free(k); free(v); free(z); free(raw); free(br); free(coA); free(coB); free(hoA); free(hoB);
        free(gA); free(gB); free(bA); free(bB);
    }
    printf("H=%-2d -j%-2d chunks:", H, nth); for (int c = 0; c < nch; c++) printf(" %d", chunks[c]);
    printf("  %s   par %.1f ms  split %.1f ms (%.1f us/token vs %.1f)\n", bad ? "MISMATCH" : "bit-identical",
           tA, tB, tB * 1e3 / tok, tA * 1e3 / tok);
    return bad;
}

#if defined(PICOLM_AVX512)
/* vector expf vs libm expf: max ULP error and the overflow/underflow limits */
static int test_expf(void) {
    float xs[16], ys[16]; double maxulp = 0; int bad = 0; long n = 0;
    for (int it = 0; it < 400000; it++) {
        for (int i = 0; i < 16; i++) xs[i] = ((float)(rnd() % 2000001) / 10000.0f) - 100.0f;     /* [-100, 100] */
        _mm512_storeu_ps(ys, kda_expf16(_mm512_loadu_ps(xs)));
        for (int i = 0; i < 16; i++) {
            const float ref = expf(xs[i]);
            if (!isfinite(ref)) { if (!isinf(ys[i])) bad++; continue; }
            if (ref < 1.2e-38f) continue;                                   /* denormal range: ULP is meaningless */
            int32_t a, b; memcpy(&a, &ref, 4); memcpy(&b, &ys[i], 4);
            const double u = fabs((double)a - (double)b); if (u > maxulp) maxulp = u; n++;
        }
    }
    float lim[16] = { 89.5f, 200.f, 1e30f, -104.5f, -200.f, -1e30f, 0.f, 1.f, -1.f, 88.7f, 88.8f, -87.f, -90.f, 50.f, -50.f, 10.f };
    _mm512_storeu_ps(ys, kda_expf16(_mm512_loadu_ps(lim)));
    if (!(isinf(ys[0]) && isinf(ys[1]) && isinf(ys[2]))) { bad++; puts("   overflow limit wrong"); }
    if (!(ys[3] == 0.f && ys[4] == 0.f && ys[5] == 0.f)) { bad++; puts("   underflow limit wrong"); }
    if (ys[6] != 1.0f) { bad++; puts("   exp(0) != 1"); }
    printf("vector expf: max error %.0f ULP over %ld samples in [-100,100]  %s\n", maxulp, n, (maxulp <= 3 && !bad) ? "ok" : "FAIL");
    return !(maxulp <= 3 && !bad);
}

/* whole layer with the vector pre-pass vs the scalar one (== glm5_kda_layer_par): close, not bitwise */
static int run_vexp(const int *chunks, int nch, int nth, int H) {
    const int KD = 128, CK = 4, DI = H * KD, CD = 3 * DI;
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
    run_state_t *sA = &m->state, *sB = calloc(1, sizeof *sB);
    const size_t SS = (size_t)H * KD * KD, CS = (size_t)(CK - 1) * CD;
    float *stA = malloc(SS * 4), *stB = malloc(SS * 4), *csA = malloc(CS * 4), *csB = malloc(CS * 4);
    fill(stA, SS, .3f); memcpy(stB, stA, SS * 4); fill(csA, CS, .5f); memcpy(csB, csA, CS * 4);
    sA->glm5.kda_state[0] = stA; sA->glm5.kda_conv_state[0] = csA;
    sB->glm5.kda_state[0] = stB; sB->glm5.kda_conv_state[0] = csB;
    double worst_o = 0, worst_s = 0, tS = 0, tV = 0, tok = 0; int bad = 0;
    for (int c = 0; c < nch; c++) {
        const int NT = chunks[c];
        float *q = malloc((size_t)NT*DI*4), *k = malloc((size_t)NT*DI*4), *v = malloc((size_t)NT*DI*4);
        float *z = malloc((size_t)NT*DI*4), *raw = malloc((size_t)NT*DI*4), *br = malloc((size_t)NT*H*4);
        fill(q,(size_t)NT*DI,2.f); fill(k,(size_t)NT*DI,2.f); fill(v,(size_t)NT*DI,2.f); fill(z,(size_t)NT*DI,3.f); fill(raw,(size_t)NT*DI,2.f); fill(br,(size_t)NT*H,2.f);
        float *coA = malloc((size_t)NT*CD*4), *coB = malloc((size_t)NT*CD*4), *hoA = malloc((size_t)NT*DI*4), *hoB = malloc((size_t)NT*DI*4);
        float *gA = malloc(DI*4), *gB = malloc(DI*4), *bA = malloc(H*4), *bB = malloc(H*4);
        double t0 = get_time_ms();
        glm5_kda_layer_split_ex(m, &gw, sA, 0, NT, q,k,v,z,raw,br, coA,gA,hoA,bA, 0);
        double t1 = get_time_ms();
        int r = glm5_kda_layer_split_ex(m, &gw, sB, 0, NT, q,k,v,z,raw,br, coB,gB,hoB,bB, 1);
        double t2 = get_time_ms(); tS += t1 - t0; tV += t2 - t1; tok += NT;
        double mo = 0, am = 0, ms = 0, as = 0;
        for (size_t i = 0; i < (size_t)NT*DI; i++) { mo = fmax(mo, fabs((double)hoA[i]-hoB[i])); am = fmax(am, fabs(hoA[i])); }
        for (size_t i = 0; i < SS; i++) { ms = fmax(ms, fabs((double)stA[i]-stB[i])); as = fmax(as, fabs(stA[i])); }
        worst_o = fmax(worst_o, mo/(am+1e-30)); worst_s = fmax(worst_s, ms/(as+1e-30));
        if (!r) bad = 1;
        free(q);free(k);free(v);free(z);free(raw);free(br);free(coA);free(coB);free(hoA);free(hoB);free(gA);free(gB);free(bA);free(bB);
    }
    if (!(worst_o < 2e-5 && worst_s < 2e-5)) bad = 1;
    printf("vexp H=%-2d -j%-2d chunks:", H, nth); for (int c = 0; c < nch; c++) printf(" %d", chunks[c]);
    printf("  max rel diff: outputs %.1e state %.1e  %s   scalar %.1f ms  vector %.1f ms (%.1f us/token vs %.1f)\n",
           worst_o, worst_s, bad ? "FAIL" : "ok", tS, tV, tV*1e3/tok, tS*1e3/tok);
    return bad;
}
#endif

int main(void) {
    int f = 0;
    const int c1[] = { 1 }, c2[] = { 2, 1, 3 }, c3[] = { 3, 4, 5 }, c4[] = { 37, 1, 130 }, c5[] = { 64, 64, 65 };
    for (int ti = 0; ti < 3; ti++) {
        const int nth = ti == 0 ? 1 : (ti == 1 ? 8 : 24);
        f += run(c1, 1, nth, 64); f += run(c2, 3, nth, 64); f += run(c3, 3, nth, 64);
        f += run(c4, 3, nth, 64); f += run(c5, 3, nth, 64);
    }
    f += run(c4, 3, 8, 6);   /* H not a multiple of 4: single-head tasks */
#if defined(PICOLM_AVX512)
    f += test_expf();
    f += run_vexp(c2, 3, 1, 64); f += run_vexp(c4, 3, 1, 64); f += run_vexp(c5, 3, 8, 64); f += run_vexp(c4, 3, 8, 6);
#endif
    puts(f ? "FAIL" : "PASS");
    return f != 0;
}
