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
 * Build: make test_swa
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

int main(void) {
    fp16_table_init();
    test_decode();
    test_prefill_tiled();
    printf("\n%s (%d failure%s)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED",
           g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
