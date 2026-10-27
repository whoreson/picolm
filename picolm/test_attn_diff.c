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
} test_ctx_t;

static const char *g_dump_file = NULL;

static void fill_cache(test_ctx_t *tc) {
    int n_kv = tc->n_kv_heads, hd = tc->head_dim;
    tc->kv_head_stride_k = (size_t)hd * 2;
    tc->kv_head_stride_v = (size_t)hd * 2;
    tc->kv_row_size_k = (size_t)n_kv * hd * 2;
    tc->kv_row_size_v = (size_t)n_kv * hd * 2;
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
    for (int p = 0; p < total; p++) {
        uint16_t *krow = (uint16_t *)(tc->kcache + (size_t)p * tc->kv_row_size_k);
        uint16_t *vrow = (uint16_t *)(tc->vcache + (size_t)p * tc->kv_row_size_v);
        const char *kr = getenv("PICOLM_TEST_KRAMP");   /* K scale grows with position */
        float kramp = kr ? (float)atof(kr) : 0.0f;
        float kscale = 1.0f + kramp * (float)p / (float)total;
        for (int i = 0; i < n_kv * hd; i++) {
            krow[i] = fp32_to_fp16(frand() * kscale);
            vrow[i] = fp32_to_fp16(frand());
        }
    }
}

static void fill_queries(test_ctx_t *tc) {
    int nh = tc->n_heads, hd = tc->head_dim;
    if (g_dump_file) return; /* loaded from dump */
    tc->q_batch = malloc((size_t)tc->n_tokens * nh * hd * sizeof(float));
    const char *qs = getenv("PICOLM_TEST_QSCALE");  /* >1 => peaky softmax, large scores */
    float qscale = qs ? (float)atof(qs) : 1.0f;
    for (int i = 0; i < tc->n_tokens * nh * hd; i++)
        tc->q_batch[i] = frand() * 2.0f * qscale;
}

static void run_path(test_ctx_t *tc, float *out, int use_tiled) {
    memset(out, 0, (size_t)tc->n_tokens * tc->xb_stride * sizeof(float));
    setenv("PICOLM_TEST_FORCE_TILED", use_tiled ? "1" : "0", 1);
    batch_attention_layer(out, tc->q_batch, tc->kcache, tc->vcache,
                          tc->n_tokens, tc->start_pos,
                          tc->n_heads, tc->n_kv_heads, tc->head_dim,
                          tc->xb_stride,
                          KV_CACHE_F16, KV_CACHE_F16,
                          tc->kv_row_size_k, tc->kv_row_size_v,
                          tc->kv_head_stride_k, tc->kv_head_stride_v,
                          1.0f / sqrtf((float)tc->head_dim), 0 /* n_swa */);
    unsetenv("PICOLM_TEST_FORCE_TILED");
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
    g_dump_file = argc > 9 ? argv[9] : NULL;

    tensor_set_n_threads(n_threads);
    fp16_table_init();
    rng_state ^= (uint64_t)seed * 0x9e3779b97f4a7c15ULL;

    test_ctx_t tc;
    tc.n_tokens = n_tokens; tc.start_pos = start_pos;
    tc.n_heads = n_heads; tc.n_kv_heads = n_kv_heads;
    tc.head_dim = head_dim; tc.xb_stride = n_heads * head_dim;
    fill_cache(&tc);
    fill_queries(&tc);

    tc.xb_ref   = calloc((size_t)n_tokens * tc.xb_stride, sizeof(float));
    tc.xb_tiled = calloc((size_t)n_tokens * tc.xb_stride, sizeof(float));

    printf("CONFIG: heads=%d kv=%d hd=%d tokens=%d start=%d swa=%d threads=%d\n",
           n_heads, n_kv_heads, head_dim, n_tokens, start_pos, n_swa, n_threads);
    printf("  tiled path: n_token_groups=%d, tasks=%d\n",
           (n_tokens + ATTN_TILE - 1) / ATTN_TILE,
           n_kv_heads * ((n_tokens + ATTN_TILE - 1) / ATTN_TILE));

    size_t n = (size_t)n_tokens * tc.xb_stride, at = 0;
    run_path(&tc, tc.xb_ref, 0);
    double s1 = 0; for (size_t i = 0; i < n; i++) s1 += tc.xb_ref[i];
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
    double md = max_abs_diff(tc.xb_ref, tc.xb_tiled, n, &at);
    const char *tl = getenv("PICOLM_TEST_TOL");
    double tol = tl ? atof(tl) : 1e-4;   /* baseline f32 noise is ~1e-7 */
    double rms_ref = 0, rms_tiled = 0;
    for (size_t i = 0; i < n; i++) {
        rms_ref += (double)tc.xb_ref[i] * tc.xb_ref[i];
        rms_tiled += (double)tc.xb_tiled[i] * tc.xb_tiled[i];
    }
    rms_ref = sqrt(rms_ref / n); rms_tiled = sqrt(rms_tiled / n);
    printf("RESULT: max_abs_diff=%.6g at [%zu] (tok %zu head %zu dim %zu)\n"
           "        rms_ref=%.6g rms_tiled=%.6g  %s\n",
           md, at, at / tc.xb_stride, (at % tc.xb_stride) / head_dim, at % head_dim,
           rms_ref, rms_tiled, md < tol ? "MATCH" : "*** MISMATCH ***");
    return md < tol ? 0 : 1;
}
