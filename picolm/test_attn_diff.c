/* Differential test: tiled prefill attention vs grouped (reference) path.
 *
 * The tiled path was disabled in c7f7cdb due to a regression from 540d68a.
 * This harness exercises batch_attention_layer() with identical inputs and
 * compares outputs between the two internal implementations.
 *
 * The gate inside batch_attention_layer() is patched for this test build
 * (PICOLM_TEST_FORCE_TILED env var selects which path runs).
 *
 * Build: see build command in run script.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "model.h"
#include "quant.h"

void tensor_set_n_threads(int n);
void fp16_table_init(void);

#define ATTN_TILE 64

extern void batch_attention_layer(
        float *xb_batch, const float *q_batch,
        const uint8_t *kcache, const uint8_t *vcache,
        int n_tokens, int start_pos,
        int n_heads, int n_kv_heads, int head_dim,
        int xb_stride,
        int kv_type_k, int kv_type_v,
        size_t kv_row_size_k, size_t kv_row_size_v,
        size_t kv_head_stride_k, size_t kv_head_stride_v,
        float attn_scale, int n_swa);

/* Stub: picolm.c defines it, but we don't link picolm.o */
double get_time_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static uint64_t rng_state = 0x123456789abcdefULL;
static float frand(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((rng_state >> 40) & 0xFFFF) / 65535.0f - 0.5f;
}

typedef struct {
    int n_tokens, start_pos, n_heads, n_kv_heads, head_dim, xb_stride;
    float *q_batch, *xb_ref, *xb_tiled;
    uint8_t *kcache, *vcache;
    size_t kv_row_size_k, kv_row_size_v, kv_head_stride_k, kv_head_stride_v;
    int kv_type_k, kv_type_v;   /* kv_cache_type_t values */
} test_ctx_t;

/* KV row/head stride for a cache type, mirroring model_core.c's
 * kv_row_size_gqa()/kv_head_stride() so the harness exercises the real
 * GQA-interleaved layout (blocks are contiguous across heads). */
static size_t t_kv_head_stride(int kv_type, int head_dim) {
    switch (kv_type) {
        case KV_CACHE_F16:  return (size_t)head_dim * sizeof(uint16_t);
        case KV_CACHE_Q8_0: return (size_t)(head_dim / 32) * sizeof(block_q8_0);
        case KV_CACHE_Q4_0: return (size_t)(head_dim / 32) * sizeof(block_q4_0);
        case KV_CACHE_TQ3:  return (size_t)(head_dim / TQ3_BLOCK_SIZE) * sizeof(block_tq3);
        case KV_CACHE_TQ4:  return (size_t)(head_dim / TQ4_BLOCK_SIZE) * sizeof(block_tq4);
    }
    return 0;
}
static int t_kv_type_from_name(const char *n) {
    if (!n) return KV_CACHE_F16;
    if (!strcmp(n, "f16")) return KV_CACHE_F16;
    if (!strcmp(n, "q8_0")) return KV_CACHE_Q8_0;
    if (!strcmp(n, "q4_0")) return KV_CACHE_Q4_0;
    if (!strcmp(n, "tq3")) return KV_CACHE_TQ3;
    if (!strcmp(n, "tq4")) return KV_CACHE_TQ4;
    return KV_CACHE_F16;
}

static const char *g_dump_file = NULL;
static float *g_topic = NULL;   /* per-kv-head topic direction (PICOLM_TEST_STRUCT) */

static void fill_cache(test_ctx_t *tc) {
    int n_kv = tc->n_kv_heads, hd = tc->head_dim;
    tc->kv_head_stride_k = t_kv_head_stride(tc->kv_type_k, hd);
    tc->kv_head_stride_v = t_kv_head_stride(tc->kv_type_v, hd);
    tc->kv_row_size_k = tc->kv_head_stride_k * (size_t)n_kv;
    tc->kv_row_size_v = tc->kv_head_stride_v * (size_t)n_kv;
    int total = tc->start_pos + tc->n_tokens;
    tc->kcache = malloc((size_t)total * tc->kv_row_size_k);
    tc->vcache = malloc((size_t)total * tc->kv_row_size_v);
    if (g_dump_file) {
        /* Load real model data: q_batch, kcache, vcache concatenated */
        FILE *f = fopen(g_dump_file, "rb");
        if (!f) { perror("dump file"); exit(1); }
        size_t qn = (size_t)tc->n_tokens * tc->n_heads * hd;
        tc->q_batch = malloc(qn * sizeof(float));
        if (fread(tc->q_batch, sizeof(float), qn, f) != qn) { fprintf(stderr, "q read fail\n"); exit(1); }
        if (fread(tc->kcache, 1, (size_t)total * tc->kv_row_size_k, f) != (size_t)total * tc->kv_row_size_k) { fprintf(stderr, "k read fail\n"); exit(1); }
        if (fread(tc->vcache, 1, (size_t)total * tc->kv_row_size_v, f) != (size_t)total * tc->kv_row_size_v) { fprintf(stderr, "v read fail\n"); exit(1); }
        fclose(f);
        fprintf(stderr, "[harness] loaded real model data from %s\n", g_dump_file);
        return;
    }
    float *kf = malloc((size_t)n_kv * hd * sizeof(float));
    float *vf = malloc((size_t)n_kv * hd * sizeof(float));
    /* PICOLM_TEST_STRUCT=1: attention-like sparsity. Each KV head has a shared
     * "topic" direction u_h. K_p = noise + beta_p*u_h where beta_p is heavy
     * tailed: huge for the sink (p==0), large for rare "needle" positions
     * (PICOLM_TEST_NEEDLE per mille, default 5), ~0 elsewhere. Queries carry
     * +gamma*u_h (see fill_queries). Uniform-random K (the default) yields flat
     * softmax and nothing is ever skippable, which says nothing about real
     * models; this gives a controllable, conservative proxy. */
    const int structd = getenv("PICOLM_TEST_STRUCT") ? atoi(getenv("PICOLM_TEST_STRUCT")) : 0;
    const float needle = getenv("PICOLM_TEST_NEEDLE") ? (float)atof(getenv("PICOLM_TEST_NEEDLE")) : 5.0f;
    float *u = NULL;
    if (structd) {
        u = malloc((size_t)n_kv * hd * sizeof(float));
        for (int i = 0; i < n_kv * hd; i++) u[i] = frand() * 2.0f;
        g_topic = u;
    }
    for (int p = 0; p < total; p++) {
        const char *kr = getenv("PICOLM_TEST_KRAMP");   /* K scale grows with position */
        float kramp = kr ? (float)atof(kr) : 0.0f;
        float kscale = 1.0f + kramp * (float)p / (float)total;
        for (int i = 0; i < n_kv * hd; i++) {
            kf[i] = frand() * kscale;
            vf[i] = frand();
        }
        if (structd) {
            float beta;
            float r = frand() + 0.5f;                  /* uniform [0,1) */
            if (p == 0)                 beta = 3.0f;   /* attention sink */
            else if (r * 1000.0f < needle) beta = 1.0f + 2.0f * (frand() + 0.5f); /* needle */
            else                        beta = 0.05f * frand();
            for (int i = 0; i < n_kv * hd; i++) kf[i] += beta * u[i];
        }
        /* Quantize head by head: the head slice is block aligned in every
         * supported type, exactly like the real KV store loop. */
        for (int h = 0; h < n_kv; h++) {
            uint8_t *kd = tc->kcache + (size_t)p * tc->kv_row_size_k
                        + (size_t)h * tc->kv_head_stride_k;
            uint8_t *vd = tc->vcache + (size_t)p * tc->kv_row_size_v
                        + (size_t)h * tc->kv_head_stride_v;
            const float *ks = kf + (size_t)h * hd;
            const float *vs = vf + (size_t)h * hd;
            switch (tc->kv_type_k) {
                case KV_CACHE_F16:
                    for (int d = 0; d < hd; d++) ((uint16_t *)kd)[d] = fp32_to_fp16(ks[d]);
                    break;
                case KV_CACHE_Q8_0: quantize_row_q8_0(ks, kd, hd); break;
                case KV_CACHE_Q4_0: quantize_row_q4_0(ks, kd, hd); break;
                case KV_CACHE_TQ3:  quantize_row_tq3(ks, kd, hd); break;
                default:            quantize_row_tq4(ks, kd, hd); break;
            }
            switch (tc->kv_type_v) {
                case KV_CACHE_F16:
                    for (int d = 0; d < hd; d++) ((uint16_t *)vd)[d] = fp32_to_fp16(vs[d]);
                    break;
                case KV_CACHE_Q8_0: quantize_row_q8_0(vs, vd, hd); break;
                case KV_CACHE_Q4_0: quantize_row_q4_0(vs, vd, hd); break;
                case KV_CACHE_TQ3:  quantize_row_tq3(vs, vd, hd); break;
                default:            quantize_row_tq4(vs, vd, hd); break;
            }
        }
    }
    free(kf); free(vf);
}

static void fill_queries(test_ctx_t *tc) {
    int nh = tc->n_heads, hd = tc->head_dim;
    if (g_dump_file) return; /* loaded from dump */
    tc->q_batch = malloc((size_t)tc->n_tokens * nh * hd * sizeof(float));
    const char *qs = getenv("PICOLM_TEST_QSCALE");  /* >1 => peaky softmax, large scores */
    float qscale = qs ? (float)atof(qs) : 1.0f;
    for (int i = 0; i < tc->n_tokens * nh * hd; i++)
        tc->q_batch[i] = frand() * 2.0f * qscale;
    if (g_topic) {   /* add the topic direction so q . (beta*u) is large for sink/needles */
        const int kvm = nh / tc->n_kv_heads;
        const float gq = getenv("PICOLM_TEST_QTOPIC") ? (float)atof(getenv("PICOLM_TEST_QTOPIC")) : 1.0f;
        for (int t = 0; t < tc->n_tokens; t++)
            for (int h = 0; h < nh; h++)
                for (int d = 0; d < hd; d++)
                    tc->q_batch[((size_t)t * nh + h) * hd + d] += gq * g_topic[(size_t)(h / kvm) * hd + d];
    }
}

static void run_path(test_ctx_t *tc, float *out, int use_tiled) {
    memset(out, 0, (size_t)tc->n_tokens * tc->xb_stride * sizeof(float));
    /* Save user env vars for these knobs so we can restore them after the call.
     * The harness controls them internally: grouped ref always runs with skip=0
     * and qk=f32 (exact), tiled runs with whatever the user sets. */
    char *oskip = getenv("PICOLM_ATTN_SKIP_T") ? strdup(getenv("PICOLM_ATTN_SKIP_T")) : NULL;
    char *oqk = getenv("PICOLM_ATTN_QK") ? strdup(getenv("PICOLM_ATTN_QK")) : NULL;
    int had_skip = oskip != NULL;
    int had_qk = oqk != NULL;
    setenv("PICOLM_TEST_FORCE_TILED", use_tiled ? "1" : "0", 1);
    /* Both runs: exact F32, no skip. The harness compares grouped vs tiled
     * at the same precision level. Users can override with PICOLM_ATTN_SKIP_T
     * and PICOLM_ATTN_QK but those affect both paths equally. */
    setenv("PICOLM_ATTN_SKIP_T", "0", 1);
    setenv("PICOLM_ATTN_QK", "f32", 1);
    batch_attention_layer(out, tc->q_batch, tc->kcache, tc->vcache,
                          tc->n_tokens, tc->start_pos,
                          tc->n_heads, tc->n_kv_heads, tc->head_dim,
                          tc->xb_stride,
                          tc->kv_type_k, tc->kv_type_v,
                          tc->kv_row_size_k, tc->kv_row_size_v,
                          tc->kv_head_stride_k, tc->kv_head_stride_v,
                          1.0f / sqrtf((float)tc->head_dim), 0 /* n_swa */);
    unsetenv("PICOLM_TEST_FORCE_TILED");
    /* Restore user settings. */
    if (had_skip) { setenv("PICOLM_ATTN_SKIP_T", oskip, 1); free(oskip); }
    else unsetenv("PICOLM_ATTN_SKIP_T");
    if (had_qk) { setenv("PICOLM_ATTN_QK", oqk, 1); free(oqk); }
    else unsetenv("PICOLM_ATTN_QK");
}

static double max_abs_diff(float *a, float *b, size_t n, size_t *at) {
    double md = 0; *at = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        /* NaN/Inf must fail: 'd > md' is false for NaN and would hide it. */
        if (!isfinite(d)) d = INFINITY;
        if (d > md) { md = d; *at = i; }
    }
    return md;
}

int main(int argc, char **argv) {
    int n_heads    = argc > 1 ? atoi(argv[1]) : 48;
    int n_kv_heads = argc > 2 ? atoi(argv[2]) : 48;
    int head_dim   = argc > 3 ? atoi(argv[3]) : 128;
    int n_tokens   = argc > 4 ? atoi(argv[4]) : 129;
    int start_pos  = argc > 5 ? atoi(argv[5]) : 0;
    int n_swa      = argc > 6 ? atoi(argv[6]) : 0;
    int seed       = argc > 7 ? atoi(argv[7]) : 1;
    int n_threads  = argc > 8 ? atoi(argv[8]) : 1;
    g_dump_file = (argc > 9 && argv[9][0] && strcmp(argv[9], "-") != 0) ? argv[9] : NULL;
    /* KV cache type: argv[10], or PICOLM_TEST_KV (argv[9] "-" = default f16). */
    const char *kv_tk = argc > 10 ? argv[10]
                      : (argc > 9 && !g_dump_file ? argv[9] : getenv("PICOLM_TEST_KV"));
    int kv_type_k = t_kv_type_from_name(kv_tk);
    int kv_type_v = kv_type_k;

    tensor_set_n_threads(n_threads);
    fp16_table_init();
    rng_state ^= (uint64_t)seed * 0x9e3779b97f4a7c15ULL;

    test_ctx_t tc;
    tc.n_tokens = n_tokens; tc.start_pos = start_pos;
    tc.n_heads = n_heads; tc.n_kv_heads = n_kv_heads;
    tc.head_dim = head_dim; tc.xb_stride = n_heads * head_dim;
    tc.kv_type_k = kv_type_k; tc.kv_type_v = kv_type_v;
    fill_cache(&tc);
    fill_queries(&tc);

    tc.xb_ref   = calloc((size_t)n_tokens * tc.xb_stride, sizeof(float));
    tc.xb_tiled = calloc((size_t)n_tokens * tc.xb_stride, sizeof(float));

    printf("CONFIG: heads=%d kv=%d hd=%d tokens=%d start=%d swa=%d threads=%d kvtype=%s\n",
           n_heads, n_kv_heads, head_dim, n_tokens, start_pos, n_swa, n_threads, kv_tk ? kv_tk : "f16");
    printf("  tiled path: n_token_groups=%d, tasks=%d\n",
           (n_tokens + ATTN_TILE - 1) / ATTN_TILE,
           n_kv_heads * ((n_tokens + ATTN_TILE - 1) / ATTN_TILE));

    size_t n = (size_t)n_tokens * tc.xb_stride, at = 0;
    /* PICOLM_TEST_NODIFF: skip the reference (grouped) run and the output
     * comparison entirely, for benchmarking. The grouped call costs roughly
     * as much as a tiled call and its output is not needed when only timing
     * is wanted. Uses only getenv/atoi, so it builds unchanged on Windows. */
    const char *nd = getenv("PICOLM_TEST_NODIFF");
    int no_diff = nd ? atoi(nd) != 0 : 0;
    double s1 = 0;
    if (!no_diff) {
        run_path(&tc, tc.xb_ref, 0);
        for (size_t i = 0; i < n; i++) s1 += tc.xb_ref[i];
    }
    run_path(&tc, tc.xb_tiled, 1);
    double s2 = 0; for (size_t i = 0; i < n; i++) s2 += tc.xb_tiled[i];
    {   /* dump tiled output for cross-binary comparison */
        const char *dump = getenv("PICOLM_ATTN_DUMP");
        if (dump) {
            FILE *f = fopen(dump, "wb");
            fwrite(tc.xb_tiled, sizeof(float), n, f);
            fclose(f);
        }
    }
    if (no_diff)
        fprintf(stderr, "[dbg] call1 sum=(skipped)  call2 sum=%.6f\n", s2);
    else
        fprintf(stderr, "[dbg] call1 sum=%.6f  call2 sum=%.6f\n", s1, s2);
    {   /* Optional timing: PICOLM_TEST_REPS=N -> median ms/call of the tiled path */
        const char *rp = getenv("PICOLM_TEST_REPS");
        int reps = rp ? atoi(rp) : 0;
        if (reps > 0) {
            double *tm = malloc((size_t)reps * sizeof(double));
            for (int r = 0; r < reps; r++) {
                double t0 = get_time_ms();
                run_path(&tc, tc.xb_tiled, 1);
                tm[r] = get_time_ms() - t0;
            }
            for (int i = 1; i < reps; i++) { double v = tm[i]; int j = i - 1;
                while (j >= 0 && tm[j] > v) { tm[j+1] = tm[j]; j--; } tm[j+1] = v; }
            printf("TIMING: tiled median=%.3f ms  min=%.3f ms  (reps=%d)\n", tm[reps/2], tm[0], reps);
            free(tm);
        }
    }
    if (no_diff) {
        printf("RESULT: diff SKIPPED (PICOLM_TEST_NODIFF)\n");
        return 0;
    }
    double md = max_abs_diff(tc.xb_ref, tc.xb_tiled, n, &at);
    const char *tl = getenv("PICOLM_TEST_TOL");
    double tol = tl ? atof(tl) : 1e-4;   /* baseline f32 noise is ~1e-7 */
    double rms_ref = 0, rms_tiled = 0;
    for (size_t i = 0; i < n; i++) {
        rms_ref += (double)tc.xb_ref[i] * tc.xb_ref[i];
        rms_tiled += (double)tc.xb_tiled[i] * tc.xb_tiled[i];
    }
    rms_ref = sqrt(rms_ref / n); rms_tiled = sqrt(rms_tiled / n);
    {   /* rms of the difference, relative to rms_ref: the number that matters for skip/i8 tuning */
        double sd = 0;
        for (size_t i = 0; i < n; i++) { double d = (double)tc.xb_ref[i] - tc.xb_tiled[i]; sd += d * d; }
        printf("        rms_diff=%.3g  rel_rms_diff=%.3g\n", sqrt(sd / n), rms_ref > 0 ? sqrt(sd / n) / rms_ref : 0.0);
    }
    printf("RESULT: max_abs_diff=%.6g at [%zu] (tok %zu head %zu dim %zu)\n"
           "        rms_ref=%.6g rms_tiled=%.6g  %s\n",
           md, at, at / tc.xb_stride, (at % tc.xb_stride) / head_dim, at % head_dim,
           rms_ref, rms_tiled, md < tol ? "MATCH" : "*** MISMATCH ***");
    return md < tol ? 0 : 1;
}
