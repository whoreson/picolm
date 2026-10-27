/* test_attn_tiled.c: differential correctness test for tiled prefill attention.
 *
 * Strategy: for each configuration (heads, kv_heads, head_dim, n_tokens,
 * start_pos, n_swa, threads), run batch_attention_layer() twice with
 * identical random inputs -- once through the grouped (reference) path
 * and once through the tiled path -- and compare outputs element-wise.
 *
 * Covers aligned and unaligned start_pos, tile-boundary token counts,
 * SWA, and multi-threaded execution.
 *
 * Build (from picolm/ project root, after `make native`):
 *   cc -O2 -std=c11 -I. -fopenmp -o test_attn_tiled tests/test_attn_tiled.c \
 *       model_attention.o tensor.o quant.o sgemm.o sgemm_q4_0x8.o \
 *       sgemm_q4i_0x8.o -lm -lpthread -fopenmp -lstdc++
 *
 * Or simply: make test_attn_tiled
 */
#define _POSIX_C_SOURCE 199309L
#include "model.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

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

/* Portable setenv/unsetenv -- matches picolm.c Windows shim */
#ifdef _WIN32
    #include <windows.h>
    #define setenv(k,v,o)     do { SetEnvironmentVariableA(k, v); } while(0)
    #define unsetenv(k)       do { SetEnvironmentVariableA(k, NULL); } while(0)
#endif

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_fail++; } \
    else { g_pass++; } \
} while (0)

/* Simple LCG RNG -- deterministic across platforms */
static uint64_t rng_state = 0x123456789abcdefULL;
static void rng_seed(uint64_t s) {
    rng_state = s * 0x9e3779b97f4a7c15ULL;
}
static float frand(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((rng_state >> 40) & 0xFFFF) / 65535.0f - 0.5f;
}

typedef struct {
    int n_tokens, start_pos, n_heads, n_kv_heads, head_dim, xb_stride;
    float *q_batch;
    uint8_t *kcache, *vcache;
    size_t kv_row_size_k, kv_row_size_v, kv_head_stride_k, kv_head_stride_v;
} ctx_t;

static void fill_cache(ctx_t *tc) {
    int n_kv = tc->n_kv_heads, hd = tc->head_dim;
    tc->kv_head_stride_k = (size_t)hd * 2;
    tc->kv_head_stride_v = (size_t)hd * 2;
    tc->kv_row_size_k = (size_t)n_kv * hd * 2;
    tc->kv_row_size_v = (size_t)n_kv * hd * 2;
    int total = tc->start_pos + tc->n_tokens;
    tc->kcache = malloc((size_t)total * tc->kv_row_size_k);
    tc->vcache = malloc((size_t)total * tc->kv_row_size_v);
    for (int p = 0; p < total; p++) {
        uint16_t *krow = (uint16_t *)(tc->kcache + (size_t)p * tc->kv_row_size_k);
        uint16_t *vrow = (uint16_t *)(tc->vcache + (size_t)p * tc->kv_row_size_v);
        for (int i = 0; i < n_kv * hd; i++) {
            krow[i] = fp32_to_fp16(frand());
            vrow[i] = fp32_to_fp16(frand());
        }
    }
}

static void fill_queries(ctx_t *tc) {
    int nh = tc->n_heads, hd = tc->head_dim;
    tc->q_batch = malloc((size_t)tc->n_tokens * nh * hd * sizeof(float));
    for (int i = 0; i < tc->n_tokens * nh * hd; i++)
        tc->q_batch[i] = frand() * 2.0f;
}

static void run_path(ctx_t *tc, float *out, int use_tiled, int n_swa) {
    memset(out, 0, (size_t)tc->n_tokens * tc->xb_stride * sizeof(float));
    setenv("PICOLM_TEST_FORCE_TILED", use_tiled ? "1" : "0", 1);
    batch_attention_layer(out, tc->q_batch, tc->kcache, tc->vcache,
                          tc->n_tokens, tc->start_pos,
                          tc->n_heads, tc->n_kv_heads, tc->head_dim,
                          tc->xb_stride,
                          KV_CACHE_F16, KV_CACHE_F16,
                          tc->kv_row_size_k, tc->kv_row_size_v,
                          tc->kv_head_stride_k, tc->kv_head_stride_v,
                          1.0f / sqrtf((float)tc->head_dim), n_swa);
    unsetenv("PICOLM_TEST_FORCE_TILED");
}

static double max_abs_diff(float *a, float *b, size_t n) {
    double md = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > md) md = d;
    }
    return md;
}

static void run_one(int heads, int kv, int hd, int tokens, int start, int swa, int thr) {
    ctx_t tc;
    memset(&tc, 0, sizeof(tc));
    tc.n_tokens = tokens; tc.start_pos = start;
    tc.n_heads = heads; tc.n_kv_heads = kv;
    tc.head_dim = hd; tc.xb_stride = heads * hd;

    rng_seed((uint64_t)heads * 1000000ULL + (uint64_t)kv * 10000ULL
            + (uint64_t)hd * 1000ULL + (uint64_t)tokens * 100ULL
            + (uint64_t)start * 10ULL + (uint64_t)thr);

    fill_cache(&tc);
    fill_queries(&tc);

    size_t n = (size_t)tokens * tc.xb_stride;
    float *xb_ref = calloc(n, sizeof(float));
    float *xb_tiled = calloc(n, sizeof(float));

    tensor_set_n_threads(thr);
    run_path(&tc, xb_ref, 0, swa);
    run_path(&tc, xb_tiled, 1, swa);

    double md = max_abs_diff(xb_ref, xb_tiled, n);

    char msg[256];
    snprintf(msg, sizeof(msg), "h=%d kv=%d hd=%d tok=%d start=%d swa=%d thr=%d",
             heads, kv, hd, tokens, start, swa, thr);
    CHECK(md < 1e-3, msg);
    if (md >= 1e-3) {
        printf("       max_diff=%.6g at start_pos=%d\n", md, start);
    }

    free(xb_ref);
    free(xb_tiled);
    free(tc.q_batch);
    free(tc.kcache);
    free(tc.vcache);
}

int main(void) {
    fp16_table_init();
    tensor_set_n_threads(1);

    printf("=== Tiled prefill attention differential test ===\n\n");

    /* Sweep: aligned and unaligned start_pos, various token counts */
    int sp[] = {0, 1, 30, 63, 64, 65, 100, 127, 128, 200};
    int nt[] = {128, 129, 191, 192, 200, 333};
    for (int si = 0; si < (int)(sizeof(sp)/sizeof(sp[0])); si++) {
        for (int ti = 0; ti < (int)(sizeof(nt)/sizeof(nt[0])); ti++) {
            run_one(8, 4, 64, nt[ti], sp[si], 0, 1);
        }
    }

    /* SWA tests (n_swa > 0) */
    for (int si = 0; si < 3; si++) {
        run_one(8, 4, 64, 300, sp[si], 96, 1);
        run_one(8, 4, 64, 300, sp[si], 96, 4);
    }

    /* Multi-head, multi-thread, various configs */
    run_one(8,  4, 128, 200, 30, 0, 4);
    run_one(8,  4, 128, 200, 100, 0, 4);
    run_one(16, 2, 128, 200, 30, 0, 1);
    run_one(16, 2, 128, 200, 100, 0, 1);
    run_one(32, 8, 128, 200, 64, 0, 4);
    run_one(32, 8, 128, 200, 65, 0, 4);

    printf("\n%d/%d passed\n", g_pass, g_pass + g_fail);
    if (g_fail == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("SOME TESTS FAILED\n");
    }
    return g_fail;
}
