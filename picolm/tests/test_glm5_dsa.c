/* Structural tests for the DSA k-pool selector (glm5_dsa_select).
 *
 * The selector is exercised directly against synthetic indexer state so the
 * invariants can be checked without a model file. What matters here is not
 * numerical agreement with the reference (the shipped ik build never enables
 * cparams.dsa) but that the selection is well-formed and causal: a sparse
 * attention mask that includes a future position, a duplicate, or an index
 * past n_kv-1 would silently corrupt every MLA layer that uses it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model.h"

static int fails = 0;
#define CHECK(c, m) do{ if(!(c)){ printf("FAIL %s\n", m); fails++; } \
                        else { printf("ok   %s\n", m); } }while(0)

/* Provided by model_glm5.c. */
int glm5_dsa_select(const model_config_t *c, const glm5_layer_weights_t *gw,
                    run_state_t *s, const float *cur, const float *qr,
                    int pos, int mla_ord, int *sel_out);
void glm5_dsa_store_kg(const model_config_t *c, const glm5_layer_weights_t *gw,
                       run_state_t *s, const float *cur, int pos, int mla_ord);

#define D   64    /* n_embd */
#define ID  32    /* idx_dim */
#define NH  4     /* idx_nh */
#define QL  16    /* q_lora */
#define R   4     /* kpool */
#define MAXSEQ 128

static void fill_kg(model_config_t *c, run_state_t *s, int pos, int mla_ord)
{
    /* Directly seed the kr_l cache: k = pattern, g = pattern. Values are
     * deterministic per position so pool keys are distinguishable. */
    for (int p = 0; p <= pos; p++) {
        float *row = s->glm5.idx_kg + (size_t)mla_ord * c->max_seq_len * 2 * ID
                   + (size_t)p * 2 * ID;
        for (int i = 0; i < ID; i++) {
            row[i]      = sinf((float)(p * ID + i) * 0.017f);          /* key  */
            row[ID + i] = cosf((float)(p * 7 + i) * 0.011f) * 0.5f;    /* gate */
        }
    }
}

int main(void)
{
    static model_config_t cfg;
    static run_state_t st;
    static glm5_layer_weights_t gw;
    memset(&cfg, 0, sizeof(cfg));
    memset(&st, 0, sizeof(st));
    memset(&gw, 0, sizeof(gw));

    cfg.n_embd = D; cfg.glm5_idx_dim = ID; cfg.glm5_idx_nh = NH;
    cfg.glm5_idx_kpool = R; cfg.glm5_idx_topk = 8;   /* 2 pools */
    cfg.glm5_q_lora = QL; cfg.max_seq_len = MAXSEQ; cfg.glm5_ln_eps = 1e-6f;

    static float kg[MAXSEQ * 2 * ID];
    st.glm5.idx_kg = kg;

    /* Minimal indexer weights: APE, q_b and proj are the three the selector
     * dereferences. The projections are tiny (1x1-style) so every pool key
     * maps to a real value; values are chosen only to be deterministic. */
    static float ape[R * ID], q_b_w[QL * NH * ID], proj_w[D * NH];
    for (int j = 0; j < R; j++)
        for (int i = 0; i < ID; i++) ape[j * ID + i] = 0.001f * (float)(j * ID + i);
    for (int i = 0; i < QL * NH * ID; i++) q_b_w[i] = 0.002f * (float)((i % 13) - 6);
    for (int i = 0; i < D * NH; i++) proj_w[i] = 0.003f * (float)((i % 7) - 3);
    gw.idx_ape = ape;
    gw.idx_q_b = q_b_w; gw.type_idx_q_b = GGUF_TYPE_F32;
    gw.idx_proj = proj_w; gw.type_idx_proj = GGUF_TYPE_F32;

    /* --- too few positions to pool -> dense fallback --- */
    {
        fill_kg(&cfg, &st, 2, 0);
        int sel[64];
        int n = glm5_dsa_select(&cfg, &gw, &st, NULL, NULL, 2, 0, sel);
        CHECK(n == 0, "pos < kpool requests the dense fallback");
    }

    /* --- selection engages, and is well-formed ---------------------------------
     * With R=4 the reference reserves r-1=3 cells for the trailing incomplete
     * pool, so (n_kv - 3) / 4 must reach 1: n_kv >= 7 (pos >= 6). Below that
     * both the reference and PicoLM fall back to dense attention. */
    {
        const int pos = 12;
        fill_kg(&cfg, &st, pos, 0);
        static float cur[D], qr[QL];
        for (int i = 0; i < D; i++) cur[i] = 0.01f * (float)(i % 7) - 0.03f;
        for (int i = 0; i < QL; i++) qr[i] = 0.02f * (float)(i % 5) - 0.04f;
        int sel[64];
        int n = glm5_dsa_select(&cfg, &gw, &st, cur, qr, pos, 0, sel);
        CHECK(n > 0, "enough complete pools produces a non-empty selection");
        int ok_bounds = 1, ok_sorted = 1, ok_dedup = 1;
        for (int i = 0; i < n; i++) {
            if (sel[i] < 0 || sel[i] > pos) ok_bounds = 0;
            if (i && sel[i] < sel[i-1]) ok_sorted = 0;
            if (i && sel[i] == sel[i-1]) ok_dedup = 0;
        }
        CHECK(ok_bounds, "selection never references a future/out-of-range cell");
        CHECK(ok_sorted, "selection is sorted ascending");
        CHECK(ok_dedup,  "selection contains no duplicates");
    }

    /* --- longer context: bounded by top_k and still causal --- */
    {
        const int pos = 100;
        fill_kg(&cfg, &st, pos, 0);
        static float cur[D], qr[QL];
        for (int i = 0; i < D; i++) cur[i] = 0.01f * (float)(i % 7) - 0.03f;
        for (int i = 0; i < QL; i++) qr[i] = 0.02f * (float)(i % 5) - 0.04f;
        int sel[512];
        int n = glm5_dsa_select(&cfg, &gw, &st, cur, qr, pos, 0, sel);
        CHECK(n > 0 && n <= cfg.glm5_idx_topk + (R - 1),
              "selection bounded by top_k plus the tail");
        int ok_bounds = 1;
        for (int i = 0; i < n; i++) if (sel[i] < 0 || sel[i] > pos) ok_bounds = 0;
        CHECK(ok_bounds, "long-context selection stays causal");

        /* The tail (most recent positions) must always be present, otherwise
         * the model would lose the tokens it is most likely to attend to. */
        int have_tail = 1;
        for (int t = (pos + 1) / R * R; t <= pos; t++) {
            int found = 0;
            for (int i = 0; i < n; i++) if (sel[i] == t) found = 1;
            if (!found) have_tail = 0;
        }
        CHECK(have_tail, "incomplete trailing pool is always appended");
    }

    /* --- determinism: same state, same selection --- */
    {
        const int pos = 40;
        fill_kg(&cfg, &st, pos, 0);
        static float cur[D], qr[QL];
        for (int i = 0; i < D; i++) cur[i] = 0.013f * (float)(i % 11) - 0.07f;
        for (int i = 0; i < QL; i++) qr[i] = 0.017f * (float)(i % 3) - 0.02f;
        int a[512], b[512];
        int na = glm5_dsa_select(&cfg, &gw, &st, cur, qr, pos, 0, a);
        int nb = glm5_dsa_select(&cfg, &gw, &st, cur, qr, pos, 0, b);
        CHECK(na == nb && memcmp(a, b, (size_t)na * sizeof(int)) == 0,
              "selection is deterministic");
    }

    /* --- geometry guard: kpool larger than the static bound falls back --- */
    {
        model_config_t big = cfg;
        big.glm5_idx_kpool = 64;    /* > GLM5_MAX_KPOOL */
        const int pos = 40;
        fill_kg(&cfg, &st, pos, 0);
        static float cur[D], qr[QL];
        for (int i = 0; i < D; i++) cur[i] = 0.01f;
        for (int i = 0; i < QL; i++) qr[i] = 0.01f;
        int sel[512];
        int n = glm5_dsa_select(&big, &gw, &st, cur, qr, pos, 0, sel);
        CHECK(n == 0, "oversized kpool degrades to dense instead of overflowing");
    }

    printf(fails ? "\n*** %d FAILURES ***\n" : "\nALL PASS\n", fails);
    return fails != 0;
}
