/* Round-trip test for model_glm5_state_save/restore.
 * Builds a synthetic GLM-5-shaped run state (no weights needed), scribbles
 * known patterns into every snapshot-covered buffer, saves, scrambles,
 * restores, and verifies byte-exact recovery. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model.h"

#define CHECK(c,msg) do{ if(!(c)){ printf("FAIL %s\n",msg); fails++; } \
                          else { printf("ok   %s\n",msg); } }while(0)

static int fails = 0;

/* ------------------------------------------------------------------ */
/* Phase 2: end-to-end v5 KV-cache file round-trip (kvcache_save/load) */
/* Also asserts v4 (non-GLM5) back-compat, since the header gained a    */
/* 14th word in v5 and must stay 13 words for v4.                       */
/* ------------------------------------------------------------------ */

static void setup_model(model_t *m, model_config_t *c, run_state_t *s, int glm5) {
    memset(m, 0, sizeof(*m));
    c->is_glm5 = glm5; c->n_layers = 8;
    c->n_layer_kv_from_start = glm5 ? 7 : 8;
    c->n_kv_heads = 2; c->head_dim = 8; c->max_seq_len = 64;
    c->glm5_kv_lora = 16; c->glm5_kda_dim = 8; c->glm5_n_kda_heads = 2;
    c->ssm_d_conv = 4; c->glm5_n_mla = 0; c->glm5_n_kda = 0;
    for (int l = 0; l < 8; l++) {
        int mla = glm5 && (l == 0 || l == 2 || l == 4 || l == 6);
        if (glm5 && l >= c->n_layer_kv_from_start) {
            c->glm5_layer_is_mla[l] = 1; c->glm5_layer_mla_ord[l] = -1; continue;
        }
        c->glm5_layer_is_mla[l] = (uint8_t)mla;
        c->glm5_layer_mla_ord[l] = mla ? (int16_t)c->glm5_n_mla++ : -1;
        if (glm5 && !mla) c->glm5_n_kda++;
    }
    int kv_elems = c->n_kv_heads * c->head_dim;
    s->kv_type_k = KV_CACHE_F16; s->kv_type_v = KV_CACHE_F16;
    s->kv_row_size_k = (size_t)kv_elems * 2; s->kv_row_size_v = s->kv_row_size_k;
    s->kv_layer_count = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (glm5) { s->kv_layer_map[l] = 0; s->kv_layer_ordinal[l] = -1; }
        else { s->kv_layer_map[l] = 1; s->kv_layer_ordinal[l] = s->kv_layer_count++; }
    }
    s->key_cache = calloc((size_t)s->kv_layer_count * c->max_seq_len * s->kv_row_size_k, 1);
    s->val_cache = calloc((size_t)s->kv_layer_count * c->max_seq_len * s->kv_row_size_v, 1);
    if (glm5) {
        int kda_dim = c->glm5_kda_dim, d_inner = c->glm5_n_kda_heads * kda_dim;
        int conv_dim = 3 * d_inner;
        s->glm5.mla_kv = calloc((size_t)c->glm5_n_mla * c->max_seq_len * c->glm5_kv_lora, sizeof(float));
        for (int l = 0; l < c->n_layers; l++) {
            if (l >= c->n_layer_kv_from_start || c->glm5_layer_is_mla[l]) continue;
            s->glm5.kda_conv_state[l] = calloc((size_t)(c->ssm_d_conv - 1) * conv_dim, sizeof(float));
            s->glm5.kda_state[l] = calloc((size_t)c->glm5_n_kda_heads * kda_dim * kda_dim, sizeof(float));
        }
    }
}

static void fill_model(model_t *m, model_config_t *c, run_state_t *s, float seed) {
    int kv_elems = c->n_kv_heads * c->head_dim;
    for (size_t i = 0; i < (size_t)s->kv_layer_count * c->max_seq_len * (size_t)kv_elems; i++) {
        ((uint16_t *)s->key_cache)[i] = fp32_to_fp16(seed + i * 0.001f);
        ((uint16_t *)s->val_cache)[i] = fp32_to_fp16(seed - i * 0.001f);
    }
    if (!c->is_glm5) return;
    int kda_dim = c->glm5_kda_dim, d_inner = c->glm5_n_kda_heads * kda_dim;
    int conv_dim = 3 * d_inner;
    size_t mt = (size_t)c->glm5_n_mla * c->max_seq_len * c->glm5_kv_lora;
    for (size_t i = 0; i < mt; i++) s->glm5.mla_kv[i] = seed + (float)i * 0.01f;
    for (int l = 0; l < c->n_layers; l++) {
        if (s->glm5.kda_conv_state[l]) {
            size_t n = (size_t)(c->ssm_d_conv - 1) * conv_dim;
            for (size_t i = 0; i < n; i++) s->glm5.kda_conv_state[l][i] = seed + l * 100 + i * 0.1f;
        }
        if (s->glm5.kda_state[l]) {
            size_t n = (size_t)c->glm5_n_kda_heads * kda_dim * kda_dim;
            for (size_t i = 0; i < n; i++) s->glm5.kda_state[l][i] = seed + l * 500 + i * 0.2f;
        }
    }
}

/* Compare one model's K cache against another's, per layer at the real
 * ao*max_seq_len*row stride (NOT packed -- positions beyond n_pos are gaps). */
static int kv_layer_equal(const model_t *a, const model_t *b, int n_pos) {
    for (int l = 0; l < a->config.n_layers; l++) {
        if (!a->state.kv_layer_map[l]) continue;
        int ao = a->state.kv_layer_ordinal[l];
        const uint8_t *pa = a->state.key_cache +
            (size_t)ao * a->config.max_seq_len * a->state.kv_row_size_k;
        const uint8_t *pb = b->state.key_cache +
            (size_t)ao * b->config.max_seq_len * b->state.kv_row_size_k;
        if (memcmp(pa, pb, (size_t)n_pos * a->state.kv_row_size_k) != 0) return 0;
    }
    return 1;
}

static int test_kv_file_roundtrip(void) {
    int fails0 = fails;
    int toks[40];
    for (int i = 0; i < 40; i++) toks[i] = 100 + i;
    const char *path = "test_glm5_snapshot.kv";

    /* GLM-5: v5 file must carry and restore the full hybrid state. */
    {
        model_t m; setup_model(&m, &m.config, &m.state, 1);
        fill_model(&m, &m.config, &m.state, 1.0f);
        CHECK(kvcache_save(&m, path, 40, toks) == 0, "glm5 file: save ok");

        int kda_dim = m.config.glm5_kda_dim;
        int conv_dim = 3 * m.config.glm5_n_kda_heads * kda_dim;
        int n_m = m.config.glm5_n_mla;
        float *rm = malloc((size_t)n_m * 40 * m.config.glm5_kv_lora * sizeof(float));
        for (int o = 0; o < n_m; o++)
            memcpy(rm + (size_t)o * 40 * m.config.glm5_kv_lora,
                   m.state.glm5.mla_kv + (size_t)o * m.config.max_seq_len * m.config.glm5_kv_lora,
                   40 * (size_t)m.config.glm5_kv_lora * sizeof(float));
        float *rc[64] = {0}, *rs[64] = {0};
        for (int l = 0; l < 8; l++) {
            if (m.state.glm5.kda_conv_state[l]) {
                size_t n = (size_t)(m.config.ssm_d_conv - 1) * conv_dim;
                rc[l] = malloc(n * sizeof(float));
                memcpy(rc[l], m.state.glm5.kda_conv_state[l], n * sizeof(float));
            }
            if (m.state.glm5.kda_state[l]) {
                size_t n = (size_t)m.config.glm5_n_kda_heads * kda_dim * kda_dim;
                rs[l] = malloc(n * sizeof(float));
                memcpy(rs[l], m.state.glm5.kda_state[l], n * sizeof(float));
            }
        }

        model_t m2; setup_model(&m2, &m2.config, &m2.state, 1);
        int *lt = NULL;
        CHECK(kvcache_load(&m2, path, &lt) == 40, "glm5 file: load reports 40 positions");
        CHECK(kv_layer_equal(&m, &m2, 40), "glm5 file: KV cache bytes round-trip");

        int mok = 1;
        for (int o = 0; o < n_m; o++)
            if (memcmp(m2.state.glm5.mla_kv + (size_t)o * m2.config.max_seq_len * m2.config.glm5_kv_lora,
                       rm + (size_t)o * 40 * m.config.glm5_kv_lora,
                       40 * (size_t)m.config.glm5_kv_lora * sizeof(float)) != 0) mok = 0;
        CHECK(mok, "glm5 file: MLA latents round-trip");

        int kok = 1;
        for (int l = 0; l < 8; l++) {
            if (rc[l] && memcmp(m2.state.glm5.kda_conv_state[l], rc[l],
                    (size_t)(m.config.ssm_d_conv - 1) * conv_dim * sizeof(float)) != 0) kok = 0;
            if (rs[l] && memcmp(m2.state.glm5.kda_state[l], rs[l],
                    (size_t)m.config.glm5_n_kda_heads * kda_dim * kda_dim * sizeof(float)) != 0) kok = 0;
        }
        CHECK(kok, "glm5 file: KDA states round-trip");
        free(lt);
    }

    /* Non-GLM5: must still write a 13-word v4 header and round-trip. */
    {
        model_t m; setup_model(&m, &m.config, &m.state, 0);
        fill_model(&m, &m.config, &m.state, 2.0f);
        CHECK(kvcache_save(&m, path, 32, toks) == 0, "v4 file: save ok");
        model_t m2; setup_model(&m2, &m2.config, &m2.state, 0);
        int *lt = NULL;
        CHECK(kvcache_load(&m2, path, &lt) == 32, "v4 file: load reports 32 positions");
        CHECK(kv_layer_equal(&m, &m2, 32), "v4 file: KV bytes round-trip (back-compat)");
        free(lt);
    }
    remove(path);
    return fails - fails0;
}

int main(void) {
    static model_t m;
    static run_state_t *s = &m.state;
    model_config_t *c = &m.config;
    memset(&m, 0, sizeof(m));

    /* Synthetic but structurally faithful GLM-5 config. */
    c->is_glm5 = 1;
    c->n_layers = 8;
    c->n_layer_kv_from_start = 7;      /* layer 7 = NextN tail (no state) */
    c->glm5_kv_lora = 16;
    c->glm5_kda_dim = 8;
    c->glm5_n_kda_heads = 2;
    c->ssm_d_conv = 4;
    c->max_seq_len = 64;
    c->glm5_n_mla = 0;
    c->glm5_n_kda = 0;
    /* layers 0,2,4 = MLA; 1,3,5 = KDA; 6 = MLA; 7 = NextN */
    for (int l = 0; l < 8; l++) {
        int mla = (l == 0 || l == 2 || l == 4 || l == 6);
        if (l >= c->n_layer_kv_from_start) {
            c->glm5_layer_is_mla[l] = 1;
            c->glm5_layer_mla_ord[l] = -1;
            continue;
        }
        c->glm5_layer_is_mla[l] = (uint8_t)mla;
        c->glm5_layer_mla_ord[l] = mla ? (int16_t)c->glm5_n_mla++ : -1;
        if (!mla) c->glm5_n_kda++;
    }
    printf("config: n_layers=%d n_mla=%d n_kda=%d kv_lora=%d kda_dim=%d\n",
           c->n_layers, c->glm5_n_mla, c->glm5_n_kda,
           c->glm5_kv_lora, c->glm5_kda_dim);

    int kda_dim = c->glm5_kda_dim;
    int d_inner = c->glm5_n_kda_heads * kda_dim;
    int conv_dim = 3 * d_inner;

    /* Allocate exactly the buffers the snapshot touches. */
    size_t mla_total = (size_t)c->glm5_n_mla * c->max_seq_len * c->glm5_kv_lora;
    s->glm5.mla_kv = calloc(mla_total, sizeof(float));
    for (int l = 0; l < c->n_layers; l++) {
        if (l >= c->n_layer_kv_from_start || c->glm5_layer_is_mla[l]) continue;
        s->glm5.kda_conv_state[l] = calloc((size_t)(c->ssm_d_conv - 1) * conv_dim, sizeof(float));
        s->glm5.kda_state[l] = calloc((size_t)c->glm5_n_kda_heads * kda_dim * kda_dim, sizeof(float));
    }

    const int n_pos = 40;

    /* Fill with distinguishable, position/layer-dependent values. */
    for (size_t i = 0; i < mla_total; i++) s->glm5.mla_kv[i] = (float)(i % 977) * 0.5f - 100.0f;
    for (int l = 0; l < c->n_layers; l++) {
        if (s->glm5.kda_conv_state[l]) {
            size_t n = (size_t)(c->ssm_d_conv - 1) * conv_dim;
            for (size_t i = 0; i < n; i++) s->glm5.kda_conv_state[l][i] = (float)(l * 1000 + i) * 0.25f;
        }
        if (s->glm5.kda_state[l]) {
            size_t n = (size_t)c->glm5_n_kda_heads * kda_dim * kda_dim;
            for (size_t i = 0; i < n; i++) s->glm5.kda_state[l][i] = (float)(l * 5000 + i) * 0.125f;
        }
    }

    size_t sz = model_glm5_snapshot_size(&m, n_pos);
    CHECK(sz > 0, "snapshot size > 0");
    size_t expect = (size_t)c->glm5_n_mla * n_pos * c->glm5_kv_lora * sizeof(float)
                  + (size_t)c->glm5_n_kda * (c->ssm_d_conv - 1) * conv_dim * sizeof(float)
                  + (size_t)c->glm5_n_kda * c->glm5_n_kda_heads * kda_dim * kda_dim * sizeof(float);
    CHECK(sz == expect, "snapshot size matches independent computation");
    printf("snapshot_size(%d pos) = %zu bytes (expected %zu)\n", n_pos, sz, expect);

    uint8_t *buf = malloc(sz);
    size_t wr = model_glm5_state_save(&m, buf, sz, n_pos);
    CHECK(wr == sz, "save wrote exactly snapshot_size bytes");
    printf("save wrote %zu / %zu\n", wr, sz);

    /* Reference copies for comparison. */
    size_t per_ord = (size_t)n_pos * c->glm5_kv_lora;
    float *ref_mla = malloc(per_ord * sizeof(float) * c->glm5_n_mla);
    for (int ord = 0; ord < c->glm5_n_mla; ord++)
        memcpy(ref_mla + (size_t)ord * per_ord,
               s->glm5.mla_kv + (size_t)ord * c->max_seq_len * c->glm5_kv_lora,
               per_ord * sizeof(float));
    float *ref_conv[64] = {0}, *ref_state[64] = {0};
    for (int l = 0; l < c->n_layers; l++) {
        if (s->glm5.kda_conv_state[l]) {
            size_t n = (size_t)(c->ssm_d_conv - 1) * conv_dim;
            ref_conv[l] = malloc(n * sizeof(float));
            memcpy(ref_conv[l], s->glm5.kda_conv_state[l], n * sizeof(float));
        }
        if (s->glm5.kda_state[l]) {
            size_t n = (size_t)c->glm5_n_kda_heads * kda_dim * kda_dim;
            ref_state[l] = malloc(n * sizeof(float));
            memcpy(ref_state[l], s->glm5.kda_state[l], n * sizeof(float));
        }
    }

    /* Scramble everything, including positions >= n_pos (which are NOT saved
     * and must survive the restore untouched -- that is the rewind contract). */
    for (size_t i = 0; i < mla_total; i++) s->glm5.mla_kv[i] = -7777.0f;
    for (int l = 0; l < c->n_layers; l++) {
        if (s->glm5.kda_conv_state[l]) {
            size_t n = (size_t)(c->ssm_d_conv - 1) * conv_dim;
            for (size_t i = 0; i < n; i++) s->glm5.kda_conv_state[l][i] = -8888.0f;
        }
        if (s->glm5.kda_state[l]) {
            size_t n = (size_t)c->glm5_n_kda_heads * kda_dim * kda_dim;
            for (size_t i = 0; i < n; i++) s->glm5.kda_state[l][i] = -9999.0f;
        }
    }

    size_t rd = model_glm5_state_restore(&m, buf, sz, n_pos);
    CHECK(rd == sz, "restore consumed exactly snapshot_size bytes");
    printf("restore consumed %zu / %zu\n", rd, sz);

    /* Verify MLA rows [0, n_pos) recovered exactly, with correct per-layer stride. */
    for (int ord = 0; ord < c->glm5_n_mla; ord++) {
        const float *got = s->glm5.mla_kv + (size_t)ord * c->max_seq_len * c->glm5_kv_lora;
        const float *exp = ref_mla + (size_t)ord * n_pos * c->glm5_kv_lora;
        int ok = memcmp(got, exp, (size_t)n_pos * c->glm5_kv_lora * sizeof(float)) == 0;
        if (!ok) printf("  MLA ordinal %d mismatch\n", ord);
        CHECK(ok, "MLA latent rows [0,n_pos) restored exactly (per-ordinal stride)");
    }
    for (int l = 0; l < c->n_layers; l++) {
        if (ref_conv[l]) {
            size_t n = (size_t)(c->ssm_d_conv - 1) * conv_dim;
            CHECK(memcmp(s->glm5.kda_conv_state[l], ref_conv[l], n * sizeof(float)) == 0,
                  "KDA conv state restored exactly");
        }
        if (ref_state[l]) {
            size_t n = (size_t)c->glm5_n_kda_heads * kda_dim * kda_dim;
            CHECK(memcmp(s->glm5.kda_state[l], ref_state[l], n * sizeof(float)) == 0,
                  "KDA recurrent state restored exactly");
        }
    }
    /* Positions >= n_pos must remain scrambled, checked per ordinal so the
     * scan cannot stray into a neighbouring ordinal's restored rows. */
    {
        int intact = 1;
        for (int ord = 0; ord < c->glm5_n_mla; ord++) {
            const float *base = s->glm5.mla_kv + (size_t)ord * c->max_seq_len * c->glm5_kv_lora;
            for (int p = n_pos; p < c->max_seq_len; p++)
                for (int i = 0; i < c->glm5_kv_lora; i++)
                    if (base[(size_t)p * c->glm5_kv_lora + i] != -7777.0f) { intact = 0; }
        }
        CHECK(intact, "positions >= n_pos left untouched (rewind contract)");
    }

    /* Truncated buffer must be rejected, not silently half-applied. */
    size_t bad = model_glm5_state_save(&m, buf, sz - 1, n_pos);
    CHECK(bad == 0, "save rejects undersized buffer");
    bad = model_glm5_state_restore(&m, buf, sz - 1, n_pos);
    CHECK(bad == 0, "restore rejects undersized buffer");

    /* n_pos=1 must produce a proportionally smaller blob. */
    size_t s1 = model_glm5_snapshot_size(&m, 1);
    CHECK(s1 < sz, "snapshot size scales with n_pos");


    printf("\n--- file round-trip ---\n");
    test_kv_file_roundtrip();

    printf(fails ? "\n*** %d FAILURES ***\n" : "\nALL PASS\n", fails);
    return fails != 0;
}
