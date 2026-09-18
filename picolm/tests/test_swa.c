/* test_swa.c: differential correctness test for sliding-window attention.
 *
 * Strategy: rather than hand-computing softmax reference values, we exploit
 * the defining property of SWA -- for query position `pos` with window
 * `n_swa`, only KV positions in [max(0,pos-n_swa+1), pos] may influence the
 * output. So:
 *   - Mutating K/V at a position BEFORE the window must leave the output
 *     bit-identical (that position is masked out).
 *   - Mutating K/V at a position INSIDE the window must change the output
 *     (sanity check that we haven't masked *everything*).
 *   - With n_swa=0 (the default/no-op case for every non-SWA model),
 *     mutating any past position must change the output (full attention).
 *
 * This directly exercises attn_core() (decode path) and
 * batch_attention_layer() -> batch_attention_tiled() (tiled prefill path,
 * forced by using >= 2*ATTN_TILE(=64) tokens), i.e. the two most important
 * call sites patched for SWA support.
 *
 * Build (from picolm/ project root, after `make native`):
 *   cc -O2 -std=c11 -I. -fopenmp -o tests/test_swa tests/test_swa.c \
 *       model_attention.o tensor.o quant.o sgemm.o sgemm_q4_0x8.o \
 *       sgemm_q4i_0x8.o -lm -lpthread -fopenmp -lstdc++
 */
#include "model.h"
#include "model_internal.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_fail++; } \
    else { printf("ok:   %s\n", msg); } \
} while (0)

static void fill_kv(uint16_t *k, uint16_t *v, int seq_len, int n_kv_heads, int head_dim,
                     size_t row_elems, unsigned seed) {
    srand(seed);
    for (int p = 0; p < seq_len; p++) {
        for (size_t e = 0; e < row_elems; e++) {
            float kf = 0.02f * (float)((int)(e + (size_t)p * 7) % 13 - 6);
            float vf = 0.02f * (float)((int)(e + (size_t)p * 5) % 11 - 5);
            k[(size_t)p * row_elems + e] = fp32_to_fp16(kf);
            v[(size_t)p * row_elems + e] = fp32_to_fp16(vf);
        }
    }
    (void)n_kv_heads; (void)head_dim;
}

/* ---- Decode path (attn_core) ---- */
static void test_decode(void) {
    printf("\n=== Decode path (attn_core) ===\n");
    const int head_dim = 8;
    const int pos = 20;
    const int seq_len = 32;
    const size_t row_elems = (size_t)head_dim; /* n_kv_heads=1 */

    uint16_t *kcache = malloc((size_t)seq_len * row_elems * sizeof(uint16_t));
    uint16_t *vcache = malloc((size_t)seq_len * row_elems * sizeof(uint16_t));
    fill_kv(kcache, vcache, seq_len, 1, head_dim, row_elems, 1);

    float q[8];
    for (int d = 0; d < head_dim; d++) q[d] = 0.1f * (float)(d + 1);

    float attn_scale = 1.0f / sqrtf((float)head_dim);
    float out_base[8], out_mut[8];

    size_t kv_row_size = row_elems * sizeof(uint16_t);
    size_t kv_head_stride = (size_t)head_dim * sizeof(uint16_t);

    /* --- n_swa = 5: window is [16, 20] --- */
    int n_swa = 5;

    attn_core(out_base, q, 0, pos, (uint8_t *)kcache, (uint8_t *)vcache,
              KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
              kv_head_stride, kv_head_stride, head_dim, attn_scale, n_swa);

    /* Mutate a position BEFORE the window (pos=10 < 16): output must be
     * bit-identical, since attn_core must never even look at it. */
    uint16_t *kcache2 = malloc((size_t)seq_len * row_elems * sizeof(uint16_t));
    uint16_t *vcache2 = malloc((size_t)seq_len * row_elems * sizeof(uint16_t));
    memcpy(kcache2, kcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) {
        kcache2[10 * row_elems + e] = fp32_to_fp16(999.0f);
        vcache2[10 * row_elems + e] = fp32_to_fp16(999.0f);
    }
    attn_core(out_mut, q, 0, pos, (uint8_t *)kcache2, (uint8_t *)vcache2,
              KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
              kv_head_stride, kv_head_stride, head_dim, attn_scale, n_swa);
    int same = memcmp(out_base, out_mut, sizeof(out_base)) == 0;
    CHECK(same, "decode: mutating a pre-window position does not change output");

    /* Mutate a position INSIDE the window (pos=18 in [16,20]): output must
     * change -- otherwise we've masked everything by mistake. */
    memcpy(kcache2, kcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) {
        kcache2[18 * row_elems + e] = fp32_to_fp16(3.0f);
        vcache2[18 * row_elems + e] = fp32_to_fp16(3.0f);
    }
    attn_core(out_mut, q, 0, pos, (uint8_t *)kcache2, (uint8_t *)vcache2,
              KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
              kv_head_stride, kv_head_stride, head_dim, attn_scale, n_swa);
    int changed = memcmp(out_base, out_mut, sizeof(out_base)) != 0;
    CHECK(changed, "decode: mutating an in-window position changes output");

    /* --- n_swa = 0 (full attention, the default/no-op case): mutating any
     * past position must change the output. --- */
    attn_core(out_base, q, 0, pos, (uint8_t *)kcache, (uint8_t *)vcache,
              KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
              kv_head_stride, kv_head_stride, head_dim, attn_scale, 0);
    memcpy(kcache2, kcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) {
        kcache2[10 * row_elems + e] = fp32_to_fp16(999.0f);
        vcache2[10 * row_elems + e] = fp32_to_fp16(999.0f);
    }
    attn_core(out_mut, q, 0, pos, (uint8_t *)kcache2, (uint8_t *)vcache2,
              KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
              kv_head_stride, kv_head_stride, head_dim, attn_scale, 0);
    changed = memcmp(out_base, out_mut, sizeof(out_base)) != 0;
    CHECK(changed, "decode: n_swa=0 is full attention (past position affects output)");

    free(kcache); free(vcache); free(kcache2); free(vcache2);
}

/* ---- Tiled prefill path (batch_attention_layer -> batch_attention_tiled) --- */
static void test_prefill_tiled(void) {
    printf("\n=== Tiled prefill path (batch_attention_layer) ===\n");
    const int head_dim = 8;
    const int n_heads = 1, n_kv_heads = 1;
    const int n_tokens = 200;   /* >= 2*ATTN_TILE(64) to force the tiled path */
    const int start_pos = 0;
    const int qi = 150;         /* the query row we inspect */
    const int n_swa = 10;       /* window for token 150 is [141, 150] */
    const size_t row_elems = (size_t)n_kv_heads * head_dim;

    uint16_t *kcache = malloc((size_t)n_tokens * row_elems * sizeof(uint16_t));
    uint16_t *vcache = malloc((size_t)n_tokens * row_elems * sizeof(uint16_t));
    fill_kv(kcache, vcache, n_tokens, n_kv_heads, head_dim, row_elems, 7);

    float *q_batch = malloc((size_t)n_tokens * n_heads * head_dim * sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        for (int d = 0; d < head_dim; d++)
            q_batch[t * head_dim + d] = 0.05f * (float)((t * 3 + d) % 9 - 4);

    float *xb_batch = malloc((size_t)n_tokens * n_heads * head_dim * sizeof(float));
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    size_t kv_row_size = row_elems * sizeof(uint16_t);
    size_t kv_head_stride = (size_t)head_dim * sizeof(uint16_t);

    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache, (uint8_t *)vcache,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim /* xb_stride */,
                           KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);

    float row_base[8];
    memcpy(row_base, xb_batch + (size_t)qi * head_dim, sizeof(row_base));

    /* Mutate a KV position well BEFORE token qi's window (100 << 141):
     * row `qi`'s output must not change. */
    uint16_t *kcache2 = malloc((size_t)n_tokens * row_elems * sizeof(uint16_t));
    uint16_t *vcache2 = malloc((size_t)n_tokens * row_elems * sizeof(uint16_t));
    memcpy(kcache2, kcache, (size_t)n_tokens * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)n_tokens * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) {
        kcache2[100 * row_elems + e] = fp32_to_fp16(999.0f);
        vcache2[100 * row_elems + e] = fp32_to_fp16(999.0f);
    }
    memset(xb_batch, 0, (size_t)n_tokens * n_heads * head_dim * sizeof(float));
    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache2, (uint8_t *)vcache2,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim, KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);
    float row_mut[8];
    memcpy(row_mut, xb_batch + (size_t)qi * head_dim, sizeof(row_mut));
    int same = memcmp(row_base, row_mut, sizeof(row_base)) == 0;
    CHECK(same, "prefill(tiled): mutating a pre-window KV position leaves token 150's row unchanged");

    /* Mutate a KV position INSIDE token qi's window (145 in [141,150]):
     * row `qi`'s output must change. */
    memcpy(kcache2, kcache, (size_t)n_tokens * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)n_tokens * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) {
        kcache2[145 * row_elems + e] = fp32_to_fp16(3.0f);
        vcache2[145 * row_elems + e] = fp32_to_fp16(3.0f);
    }
    memset(xb_batch, 0, (size_t)n_tokens * n_heads * head_dim * sizeof(float));
    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache2, (uint8_t *)vcache2,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim, KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);
    memcpy(row_mut, xb_batch + (size_t)qi * head_dim, sizeof(row_mut));
    int changed = memcmp(row_base, row_mut, sizeof(row_base)) != 0;
    CHECK(changed, "prefill(tiled): mutating an in-window KV position changes token 150's row");

    /* Also sanity-check an EARLY token (qi=20, window=[11,20], well inside
     * the very first tile) behaves the same way, exercising the
     * tile-skip / diagonal-tile logic near the start of the sequence. */
    const int qi2 = 20;
    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache, (uint8_t *)vcache,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim, KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);
    memcpy(row_base, xb_batch + (size_t)qi2 * head_dim, sizeof(row_base));
    memcpy(kcache2, kcache, (size_t)n_tokens * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)n_tokens * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) { /* position 5: before qi2's window [11,20] */
        kcache2[5 * row_elems + e] = fp32_to_fp16(999.0f);
        vcache2[5 * row_elems + e] = fp32_to_fp16(999.0f);
    }
    memset(xb_batch, 0, (size_t)n_tokens * n_heads * head_dim * sizeof(float));
    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache2, (uint8_t *)vcache2,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim, KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);
    memcpy(row_mut, xb_batch + (size_t)qi2 * head_dim, sizeof(row_mut));
    same = memcmp(row_base, row_mut, sizeof(row_base)) == 0;
    CHECK(same, "prefill(tiled): early-token pre-window mutation (token 20) leaves its row unchanged");

    free(kcache); free(vcache); free(kcache2); free(vcache2); free(q_batch); free(xb_batch);
}

/* ---- Decode path (attention_group): the ACTUAL production decode
 * dispatch, not just its attn_core() fallback ----
 *
 * model_core.c / model_gemma3n.c never call attn_core() directly for a
 * normal GQA decode step; they build an attn_group_ctx_t and call
 * attention_group() once per KV head via tensor_parallel_for(). That
 * function re-implements the same sliding-window loop-bound logic
 * independently (it has its own copy of "t_start = pos - n_swa + 1..."),
 * including a SEPARATE #ifdef PICOLM_AVX512 fast path with its own copy
 * of the K/V dot-product math. test_decode() above never exercises any
 * of this, so a bug introduced only in attention_group() (or only in one
 * of its two dot-product code paths) would go completely undetected. */
static void test_decode_group(void) {
    printf("\n=== Decode path (attention_group, real GQA dispatch) ===\n");
    const int head_dim = 8;
    const int n_kv_heads = 2;
    const int kv_mul = 2;              /* n_heads = 4 */
    const int n_heads = n_kv_heads * kv_mul;
    const int pos = 20;
    const int seq_len = 32;
    const size_t row_elems = (size_t)n_kv_heads * head_dim;

    uint16_t *kcache = malloc((size_t)seq_len * row_elems * sizeof(uint16_t));
    uint16_t *vcache = malloc((size_t)seq_len * row_elems * sizeof(uint16_t));
    fill_kv(kcache, vcache, seq_len, n_kv_heads, head_dim, row_elems, 3);

    float *q = malloc((size_t)n_heads * head_dim * sizeof(float));
    for (int h = 0; h < n_heads; h++)
        for (int d = 0; d < head_dim; d++)
            q[h * head_dim + d] = 0.07f * (float)((h * 5 + d) % 7 - 3);

    float *xb = malloc((size_t)n_heads * head_dim * sizeof(float));
    float *xb_base = malloc((size_t)n_heads * head_dim * sizeof(float));
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    size_t kv_row_size = row_elems * sizeof(uint16_t);
    size_t kv_head_stride = (size_t)head_dim * sizeof(uint16_t);
    const int n_swa = 5; /* window is [16, 20] */

    attn_group_ctx_t gctx;
    memset(&gctx, 0, sizeof(gctx));
    gctx.kv_mul = kv_mul; gctx.n_kv_heads = n_kv_heads; gctx.head_dim = head_dim;
    gctx.pos = pos;
    gctx.kv_type_k = KV_CACHE_F16; gctx.kv_type_v = KV_CACHE_F16;
    gctx.kv_row_size_k = kv_row_size; gctx.kv_row_size_v = kv_row_size;
    gctx.kv_head_stride_k = kv_head_stride; gctx.kv_head_stride_v = kv_head_stride;
    gctx.kcache = (uint8_t *)kcache; gctx.vcache = (uint8_t *)vcache;
    gctx.q = q; gctx.xb = xb_base;
    gctx.attn_scale = attn_scale;
    gctx.n_swa = n_swa;
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) attention_group(kv_h, &gctx);

    /* Mutate a position BEFORE the window (pos=10 < 16), on BOTH kv heads:
     * every query head's output must be bit-identical. */
    uint16_t *kcache2 = malloc((size_t)seq_len * row_elems * sizeof(uint16_t));
    uint16_t *vcache2 = malloc((size_t)seq_len * row_elems * sizeof(uint16_t));
    memcpy(kcache2, kcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) {
        kcache2[10 * row_elems + e] = fp32_to_fp16(999.0f);
        vcache2[10 * row_elems + e] = fp32_to_fp16(999.0f);
    }
    gctx.kcache = (uint8_t *)kcache2; gctx.vcache = (uint8_t *)vcache2; gctx.xb = xb;
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) attention_group(kv_h, &gctx);
    int same = memcmp(xb_base, xb, (size_t)n_heads * head_dim * sizeof(float)) == 0;
    CHECK(same, "decode(attention_group): mutating a pre-window position does not change any head's output");

    /* Mutate a position INSIDE the window (pos=18 in [16,20]): every
     * query head's output must change. */
    memcpy(kcache2, kcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) {
        kcache2[18 * row_elems + e] = fp32_to_fp16(3.0f);
        vcache2[18 * row_elems + e] = fp32_to_fp16(3.0f);
    }
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) attention_group(kv_h, &gctx);
    int all_changed = 1;
    for (int h = 0; h < n_heads; h++) {
        if (memcmp(xb_base + h * head_dim, xb + h * head_dim, head_dim * sizeof(float)) == 0)
            all_changed = 0;
    }
    CHECK(all_changed, "decode(attention_group): mutating an in-window position changes every head's output");

    /* n_swa=0 (full attention): a pre-window mutation must now matter. */
    gctx.kcache = (uint8_t *)kcache; gctx.vcache = (uint8_t *)vcache; gctx.xb = xb_base;
    gctx.n_swa = 0;
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) attention_group(kv_h, &gctx);
    memcpy(kcache2, kcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)seq_len * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) {
        kcache2[10 * row_elems + e] = fp32_to_fp16(999.0f);
        vcache2[10 * row_elems + e] = fp32_to_fp16(999.0f);
    }
    gctx.kcache = (uint8_t *)kcache2; gctx.vcache = (uint8_t *)vcache2; gctx.xb = xb;
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) attention_group(kv_h, &gctx);
    int changed = memcmp(xb_base, xb, (size_t)n_heads * head_dim * sizeof(float)) != 0;
    CHECK(changed, "decode(attention_group): n_swa=0 is full attention (past position affects output)");

    free(kcache); free(vcache); free(kcache2); free(vcache2); free(q); free(xb); free(xb_base);
}

/* ---- Cross-check: attn_core() and attention_group() must agree ----
 *
 * attn_core() (the kv_mul>8 / head_dim>256 fallback) and attention_group()
 * (the normal GQA path) each carry their OWN, independently-written copy
 * of the "restrict the scan to the sliding window" loop bound. There is
 * nothing in the type system stopping those two copies from drifting
 * apart. For kv_mul=1 they compute the exact same mathematical quantity,
 * so their numeric outputs (allowing for floating-point reassociation
 * between attn_core's scalar loop and attention_group's AVX-512/scalar
 * loop) must match closely -- for every n_swa tried, including 0. */
static void test_core_vs_group_consistency(void) {
    printf("\n=== Cross-check: attn_core() vs attention_group() agree ===\n");
    const int head_dim = 16;
    const int pos = 25;
    const int seq_len = 32;
    const size_t row_elems = (size_t)head_dim; /* n_kv_heads=1, kv_mul=1 */

    uint16_t *kcache = malloc((size_t)seq_len * row_elems * sizeof(uint16_t));
    uint16_t *vcache = malloc((size_t)seq_len * row_elems * sizeof(uint16_t));
    fill_kv(kcache, vcache, seq_len, 1, head_dim, row_elems, 42);

    float q[16];
    for (int d = 0; d < head_dim; d++) q[d] = 0.09f * (float)(d - 7);

    float attn_scale = 1.0f / sqrtf((float)head_dim);
    size_t kv_row_size = row_elems * sizeof(uint16_t);
    size_t kv_head_stride = (size_t)head_dim * sizeof(uint16_t);

    int swa_values[] = {0, 1, 5, 12, 26 /* >= pos+1: same as unbounded */};
    for (size_t i = 0; i < sizeof(swa_values) / sizeof(swa_values[0]); i++) {
        int n_swa = swa_values[i];

        float out_core[16];
        attn_core(out_core, q, 0, pos, (uint8_t *)kcache, (uint8_t *)vcache,
                  KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                  kv_head_stride, kv_head_stride, head_dim, attn_scale, n_swa);

        float out_group[16];
        attn_group_ctx_t gctx;
        memset(&gctx, 0, sizeof(gctx));
        gctx.kv_mul = 1; gctx.n_kv_heads = 1; gctx.head_dim = head_dim; gctx.pos = pos;
        gctx.kv_type_k = KV_CACHE_F16; gctx.kv_type_v = KV_CACHE_F16;
        gctx.kv_row_size_k = kv_row_size; gctx.kv_row_size_v = kv_row_size;
        gctx.kv_head_stride_k = kv_head_stride; gctx.kv_head_stride_v = kv_head_stride;
        gctx.kcache = (uint8_t *)kcache; gctx.vcache = (uint8_t *)vcache;
        gctx.q = q; gctx.xb = out_group;
        gctx.attn_scale = attn_scale;
        gctx.n_swa = n_swa;
        attention_group(0, &gctx);

        float max_abs_diff = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float diff = fabsf(out_core[d] - out_group[d]);
            if (diff > max_abs_diff) max_abs_diff = diff;
        }
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "attn_core vs attention_group agree for n_swa=%d (max abs diff=%.3g)",
                 n_swa, (double)max_abs_diff);
        CHECK(max_abs_diff < 1e-3f, msg);
    }

    free(kcache); free(vcache);
}

/* ---- Small-batch (non-tiled) prefill path ----
 *
 * batch_attention_layer() has TWO internal implementations: the tiled
 * path (batch_attention_tiled(), forced above with n_tokens=200) and a
 * plain per-(token,head) loop over attn_core() used whenever n_tokens is
 * too small to justify tiling (< 2*ATTN_TILE = 128, see model_attention.c).
 * Both must apply n_swa correctly; this exercises the small path end to
 * end through the real batch_attention_layer() dispatcher (not attn_core()
 * directly), which is what actually runs for short prompts/generations. */
static void test_prefill_small_batch(void) {
    printf("\n=== Small-batch (non-tiled) prefill path (batch_attention_layer) ===\n");
    const int head_dim = 8;
    const int n_heads = 1, n_kv_heads = 1;
    const int n_tokens = 40;    /* < 2*ATTN_TILE(64): forces the non-tiled path */
    const int start_pos = 0;
    const int qi = 30;
    const int n_swa = 6;        /* window for token 30 is [25, 30] */
    const size_t row_elems = (size_t)n_kv_heads * head_dim;

    uint16_t *kcache = malloc((size_t)n_tokens * row_elems * sizeof(uint16_t));
    uint16_t *vcache = malloc((size_t)n_tokens * row_elems * sizeof(uint16_t));
    fill_kv(kcache, vcache, n_tokens, n_kv_heads, head_dim, row_elems, 11);

    float *q_batch = malloc((size_t)n_tokens * n_heads * head_dim * sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        for (int d = 0; d < head_dim; d++)
            q_batch[t * head_dim + d] = 0.04f * (float)((t * 2 + d) % 7 - 3);

    float *xb_batch = malloc((size_t)n_tokens * n_heads * head_dim * sizeof(float));
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    size_t kv_row_size = row_elems * sizeof(uint16_t);
    size_t kv_head_stride = (size_t)head_dim * sizeof(uint16_t);

    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache, (uint8_t *)vcache,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim, KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);
    float row_base[8];
    memcpy(row_base, xb_batch + (size_t)qi * head_dim, sizeof(row_base));

    uint16_t *kcache2 = malloc((size_t)n_tokens * row_elems * sizeof(uint16_t));
    uint16_t *vcache2 = malloc((size_t)n_tokens * row_elems * sizeof(uint16_t));
    memcpy(kcache2, kcache, (size_t)n_tokens * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)n_tokens * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) { /* position 15: before qi's window [25,30] */
        kcache2[15 * row_elems + e] = fp32_to_fp16(999.0f);
        vcache2[15 * row_elems + e] = fp32_to_fp16(999.0f);
    }
    memset(xb_batch, 0, (size_t)n_tokens * n_heads * head_dim * sizeof(float));
    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache2, (uint8_t *)vcache2,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim, KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);
    float row_mut[8];
    memcpy(row_mut, xb_batch + (size_t)qi * head_dim, sizeof(row_mut));
    int same = memcmp(row_base, row_mut, sizeof(row_base)) == 0;
    CHECK(same, "prefill(small-batch): mutating a pre-window KV position leaves the row unchanged");

    memcpy(kcache2, kcache, (size_t)n_tokens * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)n_tokens * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) { /* position 27: inside qi's window [25,30] */
        kcache2[27 * row_elems + e] = fp32_to_fp16(3.0f);
        vcache2[27 * row_elems + e] = fp32_to_fp16(3.0f);
    }
    memset(xb_batch, 0, (size_t)n_tokens * n_heads * head_dim * sizeof(float));
    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache2, (uint8_t *)vcache2,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim, KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);
    memcpy(row_mut, xb_batch + (size_t)qi * head_dim, sizeof(row_mut));
    int changed = memcmp(row_base, row_mut, sizeof(row_base)) != 0;
    CHECK(changed, "prefill(small-batch): mutating an in-window KV position changes the row");

    free(kcache); free(vcache); free(kcache2); free(vcache2); free(q_batch); free(xb_batch);
}

/* ---- Tiled prefill with start_pos > 0 (continuation prefill) ----
 *
 * Regression test for a bug found and fixed while adding SWA support:
 * batch_attention_tiled()'s diagonal-tile causal mask used to compute the
 * query's absolute position as (group_token_start + token_idx), silently
 * dropping start_pos, so it was only correct when start_pos==0 (a fresh
 * prompt prefilled from empty cache). Any prefill that continues an
 * existing conversation (start_pos > 0) with a large enough batch to hit
 * the tiled path (>= 128 tokens) would have both its causal mask AND its
 * SWA window computed against the wrong (too-small) absolute position.
 * This test forces exactly that: start_pos=64, n_tokens=140 (still over
 * the 128-token tiled-path threshold), with a window comfortably inside
 * the continued range. */
static void test_prefill_continuation(void) {
    printf("\n=== Tiled prefill with start_pos > 0 (continuation) ===\n");
    const int head_dim = 8;
    const int n_heads = 1, n_kv_heads = 1;
    const int start_pos = 64;
    const int n_tokens = 140;    /* >= 128: still forces the tiled path */
    const int cache_len = start_pos + n_tokens;
    const int qi_local = 100;                    /* local index into this batch */
    const int qi_abs = start_pos + qi_local;      /* = 164 */
    const int n_swa = 15;                         /* window: [150, 164] */
    const size_t row_elems = (size_t)n_kv_heads * head_dim;

    uint16_t *kcache = malloc((size_t)cache_len * row_elems * sizeof(uint16_t));
    uint16_t *vcache = malloc((size_t)cache_len * row_elems * sizeof(uint16_t));
    fill_kv(kcache, vcache, cache_len, n_kv_heads, head_dim, row_elems, 21);

    float *q_batch = malloc((size_t)n_tokens * n_heads * head_dim * sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        for (int d = 0; d < head_dim; d++)
            q_batch[t * head_dim + d] = 0.03f * (float)((t * 3 + d) % 9 - 4);

    float *xb_batch = malloc((size_t)n_tokens * n_heads * head_dim * sizeof(float));
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    size_t kv_row_size = row_elems * sizeof(uint16_t);
    size_t kv_head_stride = (size_t)head_dim * sizeof(uint16_t);

    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache, (uint8_t *)vcache,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim, KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);
    float row_base[8];
    memcpy(row_base, xb_batch + (size_t)qi_local * head_dim, sizeof(row_base));

    /* Sanity: qi_abs (164) must be treated as CAUSALLY VALID for itself and
     * everything in [150,164]; a naive (buggy) implementation that treats
     * qi_local (100) as the absolute position would compute a window of
     * roughly [86,100] instead -- entirely the wrong part of the cache.
     * Mutate absolute position 155 (inside the CORRECT window [150,164],
     * but outside the WRONG window [86,100] a start_pos-dropping bug would
     * use): a correct implementation must change the output; the old
     * buggy one would not have. */
    uint16_t *kcache2 = malloc((size_t)cache_len * row_elems * sizeof(uint16_t));
    uint16_t *vcache2 = malloc((size_t)cache_len * row_elems * sizeof(uint16_t));
    memcpy(kcache2, kcache, (size_t)cache_len * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)cache_len * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) {
        kcache2[155 * row_elems + e] = fp32_to_fp16(3.0f);
        vcache2[155 * row_elems + e] = fp32_to_fp16(3.0f);
    }
    memset(xb_batch, 0, (size_t)n_tokens * n_heads * head_dim * sizeof(float));
    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache2, (uint8_t *)vcache2,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim, KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);
    float row_mut[8];
    memcpy(row_mut, xb_batch + (size_t)qi_local * head_dim, sizeof(row_mut));
    int changed = memcmp(row_base, row_mut, sizeof(row_base)) != 0;
    CHECK(changed, "prefill(continuation, start_pos=64): in-window (absolute) mutation changes the row");

    /* Mutate absolute position 100 -- inside the WRONG (start_pos-dropping
     * bug's) window [86,100], but outside the CORRECT window [150,164].
     * A correct implementation must NOT be affected by this; the old buggy
     * one would have been. This is the sharpest possible regression check
     * for the exact bug that was fixed. */
    memcpy(kcache2, kcache, (size_t)cache_len * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)cache_len * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) {
        kcache2[100 * row_elems + e] = fp32_to_fp16(999.0f);
        vcache2[100 * row_elems + e] = fp32_to_fp16(999.0f);
    }
    memset(xb_batch, 0, (size_t)n_tokens * n_heads * head_dim * sizeof(float));
    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache2, (uint8_t *)vcache2,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim, KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);
    memcpy(row_mut, xb_batch + (size_t)qi_local * head_dim, sizeof(row_mut));
    int same = memcmp(row_base, row_mut, sizeof(row_base)) == 0;
    CHECK(same, "prefill(continuation, start_pos=64): pre-window (absolute) mutation leaves the row unchanged"
                " [regression test for the start_pos-dropping diagonal-mask bug]");

    /* Also confirm causal masking itself (not just SWA) is respected with
     * start_pos>0: mutating a KV position AFTER qi_abs (a future token,
     * e.g. 170) must never affect qi's output, window or no window. */
    memcpy(kcache2, kcache, (size_t)cache_len * row_elems * sizeof(uint16_t));
    memcpy(vcache2, vcache, (size_t)cache_len * row_elems * sizeof(uint16_t));
    for (size_t e = 0; e < row_elems; e++) {
        kcache2[170 * row_elems + e] = fp32_to_fp16(-777.0f);
        vcache2[170 * row_elems + e] = fp32_to_fp16(-777.0f);
    }
    memset(xb_batch, 0, (size_t)n_tokens * n_heads * head_dim * sizeof(float));
    batch_attention_layer(xb_batch, q_batch, (uint8_t *)kcache2, (uint8_t *)vcache2,
                           n_tokens, start_pos, n_heads, n_kv_heads, head_dim,
                           head_dim, KV_CACHE_F16, KV_CACHE_F16, kv_row_size, kv_row_size,
                           kv_head_stride, kv_head_stride, attn_scale, n_swa);
    memcpy(row_mut, xb_batch + (size_t)qi_local * head_dim, sizeof(row_mut));
    same = memcmp(row_base, row_mut, sizeof(row_base)) == 0;
    CHECK(same, "prefill(continuation, start_pos=64): future-position mutation leaves the row unchanged (causal mask)");

    free(kcache); free(vcache); free(kcache2); free(vcache2); free(q_batch); free(xb_batch);
}

int main(void) {
    fp16_table_init();
    test_decode();
    test_decode_group();
    test_core_vs_group_consistency();
    test_prefill_tiled();
    test_prefill_small_batch();
    test_prefill_continuation();
    printf("\n%s (%d failure%s)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED",
           g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}