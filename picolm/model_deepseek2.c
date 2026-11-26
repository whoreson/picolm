/* model_deepseek2.c -- DeepSeek V3 (GGUF arch "deepseek2") forward pass.
 *
 * Architecture (DeepSeek-V3-0324):
 *   - MLA (Multi-head Latent Attention) with ABSORPTION, partial RoPE
 *   - low-rank Q: wq_a -> RMSNorm -> wq_b
 *   - compressed KV: wkv_a_mqa -> [k_pe(64) ; kv_latent(512)] -> RMSNorm on the latent
 *   - K/V decompression matrices wk_b / wv_b, derived from the combined
 *     attn_kv_b at load time (ds2_prepare_mla), Q8_0
 *   - dense SwiGLU FFN on the first `leading_dense_block_count` layers
 *   - MoE FFN elsewhere: 256 experts, top-8, sigmoid routing with a
 *     selection-only bias, weight renormalisation, scale 2.5, 1 shared expert
 *
 * Reference: ik_llama.cpp src/graphs/build_deepseek2.cpp, absorb path
 * (mla_attn >= 1), and src/llama.cpp llm_prepare_mla() for wk_b/wv_b.
 *
 * CPU-only; GPU dispatch is not wired for deepseek2 yet.
 *
 * --- MLA absorb path, exact shapes (V3-0324) ---
 *   D=7168  NH=128  QL=1536  KVL=512  ROPE=64  NOPE=128  HV=128
 *
 *   qr    = rmsnorm(Wq_a @ cur, q_a_norm)                 [QL]
 *   q     = Wq_b @ qr                                     [NH * (NOPE+ROPE)]
 *   kvr   = Wkv_a_mqa @ cur                               [KVL + ROPE]
 *   k_pe  = kvr[KVL .. KVL+ROPE)      (NOT normalised)     [ROPE]
 *   kv_c  = rmsnorm(kvr[0 .. KVL), kv_a_norm)             [KVL]
 *   q_rope[h] = rope(q[h*(NOPE+ROPE) + NOPE .. +ROPE])     [ROPE]
 *   k_pe      = rope(k_pe)
 *   cache[pos] = [k_pe ; kv_c]                             [ROPE + KVL]
 *   q_abs[h]  = Wk_b[h] @ q_nope[h]                        [KVL]
 *   q_comb[h] = [q_rope[h] ; q_abs[h]]                     [ROPE + KVL]
 *   for each cached t: s_t = dot(q_comb[h], cache[t]) * kq_scale
 *   ctx[h]    = softmax(s) weighted sum of cache[t][ROPE .. ROPE+KVL)   [KVL]
 *   o[h]      = Wv_b[h] @ ctx[h]                           [HV]
 *   out       = Wo @ concat(o)                             [D]
 *
 * kq_scale = mscale^2 / sqrt(NOPE+ROPE), with
 *   mscale = (1 + rope_yarn_log_mul * log(1/freq_scale)).
 * The RoPE table already carries the per-dim mscale (resolve_rope_mscale), so
 * the attention scale must NOT apply it again -- see ds2_kq_scale().
 */
#include "model.h"
#include "model_internal.h"
#include "tensor.h"
#include "quant.h"
#include "steer.h"
#ifdef PICOLM_VIZ
#include "viz.h"
#endif
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* Scores row stride for the PFA attention tail (ds2_pfa_scores/pv).
 * One block of keys per pass; see ds2_scores_ctx. */
#define DS2_PFA_BK 64

#ifndef INFINITY
#define INFINITY (1.0f/0.0f)
#endif

/* ------------------------------------------------------------------ */
/* Load-time MLA preparation: wk_b / wv_b from the combined attn_kv_b  */
/*                                                                     */
/* ik's llm_prepare_mla() does exactly this. With                     */
/*   wkv_b  = [KVL rows][NH * (NOPE + HV) cols]  (ne0 = KVL contiguous) */
/*   head h occupies cols [h*(NOPE+HV), (h+1)*(NOPE+HV))               */
/*                                                                     */
/*   wk_b[h][d][r]  = wkv_b[r][h*(NOPE+HV) + d]         (TRANSPOSED)   */
/*       => stored as ne = {NOPE, KVL, NH}, i.e. the GLM-5 attn_k_b     */
/*          layout: per head, NOPE rows of KVL contiguous values.       */
/*   wv_b[h][r][dv] = wkv_b[r][h*(NOPE+HV) + NOPE + dv]  (NOT transposed) */
/*       => ne = {KVL, HV, NH}: per head, HV rows of KVL contiguous.    */
/*                                                                     */
/* Both are written out as Q8_0 so that the per-head matmuls in the     */
/* forward pass stay on the fast int8 kernel.                          */
/* ------------------------------------------------------------------ */

/* Heap-resident, per-layer derived MLA matrices. NULL when the GGUF ships
 * attn_k_b/attn_v_b split (then the mmap'd tensors are used directly). */
static void **g_ds2_wk = NULL;
static void **g_ds2_wv = NULL;
static int    g_ds2_nlayers = 0;

/* Derived matrices are kept in F32.
 *
 * Requantising them (Q8_0 was the first choice, to halve the memory) adds a
 * per-layer error of ~1.7% on the attention output; validated by
 * PICOLM_DS2_XCHECK, which compares the absorbed path against the
 * non-absorbed wkv_b reference. That is small per layer but it compounds over
 * all 61 layers and flattens the final logits, so the derivation is done
 * losslessly: 61 * (128*512 + 512*128) * 4 B = 4.1 GB. */
static size_t ds2_wk_bytes(int NH, int NOPE, int KVL) {
    return (size_t)NH * KVL * (size_t)NOPE * sizeof(float);
}
static size_t ds2_wv_bytes(int NH, int HV, int KVL) {
    return (size_t)NH * HV * (size_t)KVL * sizeof(float);
}

/* Free any derived buffers (called from the model teardown if present). */
void ds2_free_mla(void)
{
    if (g_ds2_wk) {
        for (int i = 0; i < g_ds2_nlayers; i++) free(g_ds2_wk[i]);
        free(g_ds2_wk); g_ds2_wk = NULL;
    }
    if (g_ds2_wv) {
        for (int i = 0; i < g_ds2_nlayers; i++) free(g_ds2_wv[i]);
        free(g_ds2_wv); g_ds2_wv = NULL;
    }
    g_ds2_nlayers = 0;
}

/* Build one layer's wk_b/wv_b from its combined attn_kv_b.
 * wkv_b is [KVL][NH*(NOPE+HV)], type qt. Allocates *wk_out / *wv_out as Q8_0.
 * Returns 0 on success. */
static int ds2_build_kv_b(int NH, int NOPE, int HV, int KVL,
                          const void *wkv_b, gguf_type_t qt,
                          void **wk_out, void **wv_out)
{
    const int per_head = NOPE + HV;
    const int ncol = NH * per_head;

    void *wk = malloc(ds2_wk_bytes(NH, NOPE, KVL));
    void *wv = malloc(ds2_wv_bytes(NH, HV, KVL));
    if (!wk || !wv) { free(wk); free(wv); return -1; }

    memset(wk, 0, ds2_wk_bytes(NH, NOPE, KVL));
    memset(wv, 0, ds2_wv_bytes(NH, HV, KVL));

    /* Dequantize ONE GGUF ROW AT A TIME. Each quantized row has KVL elements
     * (ne[0] contiguous); there are ncol = NH*per_head rows (ne[1]). The old
     * code iterated r=0..KVL reading ncol elements per "row", which ran off
     * the end of every row and pulled garbage from adjacent rows. */
    float *wrow = (float *)malloc((size_t)KVL * sizeof(float));
    const size_t wkv_row_sz = gguf_type_row_size(qt, KVL);
    if (!wrow) { free(wk); free(wv); return -1; }

    for (int c = 0; c < ncol; c++) {
        dequantize_row((const uint8_t *)wkv_b + (size_t)c * wkv_row_sz,
                       wrow, KVL, qt);
        /* wrow[r] = wkv_b[r, c]. Distribute column c:
         *   wk[h*KVL*NOPE + r*NOPE + off] = wkv_b[r, h*per_head + off]
         *   wv[h*HV*KVL + dv*KVL + r]     = wkv_b[r, h*per_head + NOPE + dv]
         */
        int h = c / per_head;
        int off = c % per_head;
        if (off < NOPE) {
            float *base = (float *)wk + (size_t)h * (size_t)KVL * (size_t)NOPE;
            for (int r = 0; r < KVL; r++)
                base[(size_t)r * (size_t)NOPE + off] = wrow[r];
        } else {
            int dv = off - NOPE;
            memcpy((float *)wv + ((size_t)h * (size_t)HV + (size_t)dv) * (size_t)KVL,
                   wrow, (size_t)KVL * sizeof(float));
        }
    }
    free(wrow);
    *wk_out = wk; *wv_out = wv;
    return 0;
}

/* Prepare wk_b/wv_b for every layer that needs them. Called once at the end of
 * model loading. Safe no-op when nothing needs deriving. */
int ds2_prepare_mla(model_t *m)
{
    model_config_t *c = &m->config;
    if (!c->is_deepseek2) return 0;
    if (c->ds2_wkv_b_split) return 0;      /* already split in the GGUF */

    const int NH = c->n_heads;
    const int NOPE = c->ds2_nope;
    const int HV = c->ds2_head_v;
    const int KVL = c->glm5_kv_lora;

    if (!m->weights.glm5) {
        fprintf(stderr, "ERROR: deepseek2: no glm5 weight array for MLA derivation\n");
        return -1;
    }

    ds2_free_mla();
    g_ds2_nlayers = c->n_layers;
    g_ds2_wk = (void **)calloc((size_t)c->n_layers, sizeof(void *));
    g_ds2_wv = (void **)calloc((size_t)c->n_layers, sizeof(void *));
    if (!g_ds2_wk || !g_ds2_wv) { ds2_free_mla(); return -1; }

    int n_done = 0;
    for (int l = 0; l < c->n_layers; l++) {
        glm5_layer_weights_t *gw = &m->weights.glm5[l];
        if (!gw->mla_kv_b) continue;
        void *wk = NULL, *wv = NULL;
        if (ds2_build_kv_b(NH, NOPE, HV, KVL, gw->mla_kv_b, gw->type_mla_kv_b, &wk, &wv) != 0) {
            fprintf(stderr, "ERROR: deepseek2: OOM deriving wk_b/wv_b for layer %d\n", l);
            ds2_free_mla();
            return -1;
        }
        g_ds2_wk[l] = wk;
        g_ds2_wv[l] = wv;
        n_done++;
    }
    if (n_done != c->n_layers) {
        fprintf(stderr, "WARN: deepseek2: derived wk_b/wv_b for %d/%d layers\n",
                n_done, c->n_layers);
    }
    fprintf(stderr, "deepseek2: derived wk_b/wv_b for %d layers "
                    "(%zu MB wk + %zu MB wv, F32)\n",
            n_done,
            (size_t)n_done * ds2_wk_bytes(NH, NOPE, KVL) >> 20,
            (size_t)n_done * ds2_wv_bytes(NH, HV, KVL) >> 20);
    return 0;
}

static void ds2_yarn_factors(const model_config_t *c,
                             float *out_mscale, float *out_rope_mag);

/* Local copies of model_core.c's static helpers (they are static there and
 * used only by the debug path here). */
static float ds2_yarn_ramp(float low, float high, float i0) {
    float y = (i0 / 2.0f - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}
static void ds2_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
                               float beta_fast, float beta_slow, float dims[2]) {
    const float pi = 3.14159265358979f;
    float start = floorf((float)n_dims * logf((float)n_ctx_orig / (beta_fast * 2.0f * pi)) / (2.0f * logf(freq_base)));
    float end   = ceilf((float)n_dims * logf((float)n_ctx_orig / (beta_slow * 2.0f * pi)) / (2.0f * logf(freq_base)));
    dims[0] = fmaxf(0.0f, start);
    dims[1] = fminf((float)(n_dims - 1), end);
}
static float ds2_kq_scale_of(const model_config_t *c, float mscale);

/* Build the DeepSeek V3 YaRN tables and the attention score scale. Called
 * once from model_load(), after the run state arena exists. */
int ds2_init_rope(model_t *m)
{
    model_config_t *c = &m->config;
    run_state_t *s = &m->state;
    if (!c->is_deepseek2) return 0;
    if (!s->ds2.rope_cos || !s->ds2.rope_sin) {
        fprintf(stderr, "ERROR: deepseek2: no YaRN table storage\n");
        return -1;
    }
    float mscale, rope_mag;
    ds2_yarn_factors(c, &mscale, &rope_mag);
    const float ext_factor = (c->rope_scaling_type == 0) ? 0.0f : 1.0f;
    /* ggml's rope_yarn() multiplies its mscale PARAMETER by the inline factor
     * (1 + 0.1*log(1/freq_scale)) whenever ext_factor != 0, so the magnitude
     * ik_llama.cpp's deepseek2 graph ends up with in the table is
     *   attn_factor_scaled * (1 + 0.1*log(1/fs))  ==  1.0 (for attn_factor=1)
     * while the YaRN magnitude lives entirely in kq_scale = mscale^2/sqrt(dk).
     * build_rope_table() applies its mscale argument as the FINAL magnitude
     * (no inline multiply), so we must do the inline multiply here. Missing
     * this made every rope dim 0.7305x too small (verified vs ggml). */
    float table_mag = rope_mag;
    if (ext_factor != 0.0f && c->rope_freq_scale > 0.0f)
        table_mag *= 1.0f + 0.1f * logf(1.0f / c->rope_freq_scale);
    rope_build_table_ext(c, s->ds2.rope_cos, s->ds2.rope_sin, table_mag, ext_factor);
    s->ds2.kq_scale = ds2_kq_scale_of(c, mscale);
    fprintf(stderr, "deepseek2: yarn mscale=%.6f rope_mag=%.6f kq_scale=%.6f "
                    "(freq_scale=%.6f log_mul=%.4f)\n",
            (double)mscale, (double)rope_mag, (double)s->ds2.kq_scale,
            (double)c->rope_freq_scale, (double)c->rope_yarn_log_mul);
    if (getenv("PICOLM_DS2_ROPEDBG")) {
        /* ggml rope_yarn() reference: theta mixing + the INLINE magnitude
         * multiply mscale *= 1+0.1*log(1/freq_scale) (applied only when
         * ext_factor != 0). Our table must match cos/sin of that formula
         * with mscale_param = attn_factor_scaled. */
        const int half = c->ds2_rope / 2;
        const float fs = c->rope_freq_scale;
        const float base = c->rope_freq_base;
        const uint32_t nctx = (c->rope_ctx_orig > 0) ? c->rope_ctx_orig : (uint32_t)c->max_seq_len;
        const float bf = c->rope_beta_fast, bs = c->rope_beta_slow;
        float cd[2];
        ds2_yarn_corr_dims(c->ds2_rope, (int)nctx, base, bf, bs, cd);
        fprintf(stderr, "[ds2 rope] corr_dims=%.3f,%.3f base=%.1f fs=%.5f nctx=%u\n",
                (double)cd[0], (double)cd[1], (double)base, (double)fs, (unsigned)nctx);
        for (int p = 0; p < 2 && p < c->max_seq_len; p++) {
            const float tscale = powf(base, -2.0f / (float)c->ds2_rope);
            float th = (float)p;
            float ref_cos[32], ref_sin[32];
            for (int i = 0; i < half; i++) {
                /* ggml rope_cache_init: theta starts at pos and is multiplied
                 * by theta_scale = base^(-2/n_dims) once per PAIR, i.e.
                 * theta_extrap(i) = pos * base^(-2i/n_dims) exactly. */
                float theta_ex = th;
                float th_in = fs * theta_ex;
                float ramp = ds2_yarn_ramp(cd[0], cd[1], (float)(i * 2)) * ext_factor;
                float th_f = th_in * (1.0f - ramp) + theta_ex * ramp;
                float mag = rope_mag * (1.0f + 0.1f * logf(1.0f / fs));
                ref_cos[i] = cosf(th_f) * mag;
                ref_sin[i] = sinf(th_f) * mag;
                th *= tscale;
            }
            fprintf(stderr, "[ds2 rope] pos=%d ours vs ggml-ref:\n", p);
            for (int i = 0; i < half && i < 6; i++)
                fprintf(stderr, "   i=%2d cos %+.7f / %+.7f  sin %+.7f / %+.7f\n",
                        i, (double)s->ds2.rope_cos[p * half + i], (double)ref_cos[i],
                        (double)s->ds2.rope_sin[p * half + i], (double)ref_sin[i]);
        }
    }
    return 0;
}

/* Per-layer wk_b/wv_b + row sizes, abstracting over "mmap'd split tensors"
 * and "heap-derived from attn_kv_b". */
static void ds2_scores_ctx(const model_t *m, int layer, run_state_t *s,
                           const float *q_in, const float *qabs,
                           float *q_comb, float *scores,
                           int pos, float *ctx,
                           int NH, int KVL, int ROPE, int NOPE, int KHK, int LROW);
static void ds2_kv_b_ptrs(const model_t *m, int layer,
                          const void **wk, gguf_type_t *wk_t, size_t *wk_head_sz,
                          const void **wv, gguf_type_t *wv_t, size_t *wv_head_sz)
{
    const model_config_t *c = &m->config;
    const glm5_layer_weights_t *gw = &m->weights.glm5[layer];
    const int NH = c->n_heads;
    const int NOPE = c->ds2_nope;
    const int HV = c->ds2_head_v;
    const int KVL = c->glm5_kv_lora;

    if (g_ds2_wk && g_ds2_wk[layer]) {
        *wk = g_ds2_wk[layer];
        *wk_t = GGUF_TYPE_F32;
        *wk_head_sz = (size_t)KVL * NOPE * sizeof(float);
        *wv = g_ds2_wv[layer];
        *wv_t = GGUF_TYPE_F32;
        *wv_head_sz = (size_t)HV * KVL * sizeof(float);
        return;
    }
    /* Split GGUF tensors: attn_k_b ne={NOPE,KVL,NH}, attn_v_b ne={KVL,HV,NH}. */
    *wk = gw->mla_k_b;
    *wk_t = gw->type_mla_k_b;
    *wk_head_sz = (size_t)KVL * gguf_type_row_size(gw->type_mla_k_b, NOPE);
    *wv = gw->mla_v_b;
    *wv_t = gw->type_mla_v_b;
    *wv_head_sz = (size_t)HV * gguf_type_row_size(gw->type_mla_v_b, KVL);
    (void)NH;
}

/* ------------------------------------------------------------------ */
/* YaRN magnitude and attention scale, DeepSeek V3 convention           */
/*                                                                     */
/* Reference: build_deepseek2() in ik_llama.cpp. With                 */
/*   lmul  = rope.yarn_log_multiplier / 0.1   (GGUF stores pre-divided)  */
/*   logf_ = log(1 / rope_freq_scale)                                   */
/*                                                                     */
/*   mscale             = 1 + lmul * logf_                              */
/*   attn_factor_scaled = 1 / (1 + 0.1 * logf_)                          */
/*   kq_scale           = mscale^2 / sqrt(key_length)                    */
/*                                                                     */
/* The RoPE table is built with `attn_factor_scaled` (NOT mscale), and   */
/* the score scale carries the whole mscale^2. That is why this cannot   */
/* use the shared resolve_rope_mscale(): that helper implements the       */
/* "ratio of two get_mscale()" variant and assumes the rope math applies  */
/* the inline magnitude itself.                                          */
/* ------------------------------------------------------------------ */
static void ds2_yarn_factors(const model_config_t *c,
                             float *out_mscale, float *out_rope_mag)
{
    float mscale = 1.0f;
    float attn_factor_scaled = 1.0f;
    if (c->rope_scaling_type != 0 && c->rope_freq_scale > 0.0f) {
        const float factor = 1.0f / c->rope_freq_scale;      /* 40 for V3 */
        if (factor > 1.0f) {
            const float logf_ = logf(factor);
            /* ik_llama.cpp reads rope.yarn_log_multiplier RAW for deepseek2
             * (hparams.cpp: LLM_KV_ROPE_SCALING_YARN_LOG_MUL, no /0.1), so the
             * score magnitude is
             *     mscale = attn_factor * (1 + log_mul * log(1/freq_scale))
             * with attn_factor = 1.0 (no yarn_attn_factor key in these GGUFs).
             * Only the RoPE TABLE uses attn_factor_scaled below. Dividing
             * log_mul by 0.1 here (the Mistral-3/V2-mainline rule) made
             * kq_scale ~12x too large, which inflated every attention output
             * and flattened the final logits. */
            const float lmul = c->rope_yarn_log_mul;
            mscale = 1.0f + lmul * logf_;
            attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf_);
        }
    }
    if (c->rope_yarn_attn_factor != 0.0f) mscale *= c->rope_yarn_attn_factor;
    if (c->rope_attn_factor     != 0.0f) mscale *= c->rope_attn_factor;
    *out_mscale = mscale;
    *out_rope_mag = attn_factor_scaled;
}

static float ds2_kq_scale_of(const model_config_t *c, float mscale)
{
    return mscale * mscale / sqrtf((float)c->glm5_mla_head_dim);
}

/* ------------------------------------------------------------------ */
/* MLA attention (decode, single token)                                */
/* ------------------------------------------------------------------ */

/* Scratch layout inside s->ds2.scratch (floats), matching the arena sizing
 * in allocate_run_state():
 *   qr[QL] | q[NH*KHK] | qabs[NH*KVL] | kvr[LROW] | ctx[NH*KVL]
 *   | scores[max_seq] | o_concat[NH*HV]
 */
static void ds2_mla_forward(const model_t *m, int layer,
                            run_state_t *s, const float *cur, int pos,
                            float *out)
{
    const model_config_t *c = &m->config;
    const glm5_layer_weights_t *gw = &m->weights.glm5[layer];
    const layer_weights_t *lw = &m->weights.layers[layer];
    const int D    = c->n_embd;
    const int NH   = c->n_heads;
    const int QL   = c->glm5_q_lora;
    const int KVL  = c->glm5_kv_lora;
    const int ROPE = c->ds2_rope;
    const int NOPE = c->ds2_nope;
    const int HV   = c->ds2_head_v;
    const int KHK  = c->glm5_mla_head_dim;   /* NOPE + ROPE */
    const int LROW = KVL + ROPE;             /* cache row width */

    float *sc = s->ds2.scratch;
    float *qr    = sc;  sc += QL;
    float *q     = sc;  sc += (size_t)NH * KHK;
    float *qabs  = sc;  sc += (size_t)NH * KVL;
    /* Canonical order (matches model_core.c allocation):
     * qr, q, qabs, q_comb(NH*LROW), ctx(NH*KVL), scores(NH*DS2_PFA_BK),
     * kvr(LROW), o_concat(NH*HV). */
    float *q_comb = sc; sc += (size_t)NH * LROW;
    float *ctx   = sc;  sc += (size_t)NH * KVL;
    float *scores = sc; sc += (size_t)NH * DS2_PFA_BK;
    float *kvr   = sc;  sc += LROW;
    float *o_concat = sc;
    const int prof = (getenv("PICOLM_DS2_PROF") != NULL);
    double p0 = prof ? get_time_ms() : 0, p1, p2, p3, p4, p5, p6;
    if (getenv("PICOLM_DS2_DBG2") && layer == 0) {
        size_t need = (size_t)QL + (size_t)NH*KHK + 2*(size_t)NH*KVL + LROW
                    + c->max_seq_len + (size_t)NH*HV;
        fprintf(stderr, "[ds2 scr] scratch=%p need=%zu floats ; mla_kv=%p stride=%zu ; ropecos=%p ropesin=%p\n",
                (void*)s->ds2.scratch, need, (void*)s->ds2.mla_kv, s->ds2.mla_kv_layer_stride,
                (void*)s->ds2.rope_cos, (void*)s->ds2.rope_sin);
        fprintf(stderr, "[ds2 scr] gap scratch_end->ropecos = %ld floats\n",
                (long)((float*)s->ds2.rope_cos - (s->ds2.scratch + need)));
    }

    /* ---- Q: wq_a -> RMSNorm -> wq_b (or direct attn_q for "lite") ---- */
    if (c->ds2_has_q_lora) {
        float *tmp = s->hb;      /* >= QL floats (n_ffn >= 1536) */
        matmul(tmp, cur, gw->mla_q_a, D, QL, gw->type_mla_q_a);
        {
            float ss = 0.f;
            for (int i = 0; i < QL; i++) ss += tmp[i] * tmp[i];
            const float inv = 1.0f / sqrtf(ss / QL + c->rms_norm_eps);
            const float *nw = (const float *)gw->mla_q_a_norm;   /* F32 in GGUF */
            for (int i = 0; i < QL; i++) qr[i] = tmp[i] * inv * nw[i];
        }
        matmul(q, qr, gw->mla_q_b, QL, NH * KHK, gw->type_mla_q_b);
    } else {
        matmul(q, cur, gw->mla_q, D, NH * KHK, gw->type_mla_q);
    }

    /* ---- KV: wkv_a_mqa -> [k_pe ; kv_latent], norm the latent only ---- */
    p1 = prof ? get_time_ms() : 0;
    matmul(kvr, cur, gw->mla_kv_a_mqa, D, LROW, gw->type_mla_kv_a_mqa);
    {
        /* kvr holds [KVL latent][ROPE pe]; norm the first KVL. */
        float ss = 0.f;
        for (int i = 0; i < KVL; i++) ss += kvr[i] * kvr[i];
        const float inv = 1.0f / sqrtf(ss / KVL + c->rms_norm_eps);
        const float *nw = (const float *)gw->mla_kv_a_norm;      /* F32 */
        for (int i = 0; i < KVL; i++) kvr[i] *= inv * nw[i];
    }

    /* ---- RoPE on q_rope (per head, last ROPE dims) and on k_pe ----
     * tensor.c's rope() (AVX-512/AVX/NEON, pairwise == rope_type 0). */
    {
        const int half = ROPE / 2;
        const float *cos_pos = s->ds2.rope_cos + (size_t)pos * half;
        const float *sin_pos = s->ds2.rope_sin + (size_t)pos * half;
        for (int h = 0; h < NH; h++)
            rope(q + (size_t)h * KHK + NOPE, NULL, ROPE, 1, 0,
                 cos_pos, sin_pos, 0, half);
        rope(kvr + KVL, NULL, ROPE, 1, 0, cos_pos, sin_pos, 0, half);
    }

    /* ---- Store the cache row: [k_pe ; kv_latent] (rope FIRST) ---- */
    float *cache = s->ds2.mla_kv + (size_t)layer * s->ds2.mla_kv_layer_stride;
    {
        float *row = cache + (size_t)pos * LROW;
        memcpy(row, kvr + KVL, (size_t)ROPE * sizeof(float));   /* k_pe */
        memcpy(row + ROPE, kvr, (size_t)KVL * sizeof(float));   /* latent */
    }

    /* ---- Absorb q_nope into the latent space: q_abs[h] = Wk_b[h] @ q_nope ---- */
    p2 = prof ? get_time_ms() : 0;
    int multi_ret = -1;
    {
        const void *wk; gguf_type_t wk_t; size_t wk_head_sz;
        const void *wv; gguf_type_t wv_t; size_t wv_head_sz;
        ds2_kv_b_ptrs(m, layer, &wk, &wk_t, &wk_head_sz, &wv, &wv_t, &wv_head_sz);

        {
            mm_job_t jobs[128];
            const int nj = NH <= 128 ? NH : 0;
            for (int h = 0; h < nj; h++) {
                jobs[h].out = qabs + (size_t)h * KVL;
                jobs[h].x   = q + (size_t)h * KHK;      /* q_nope is the FIRST NOPE dims */
                jobs[h].W   = (const uint8_t *)wk + (size_t)h * wk_head_sz;
                jobs[h].n   = NOPE;
                jobs[h].d   = KVL;
                jobs[h].qtype = wk_t;
            }
            if (!nj || (multi_ret = tensor_matmul_multi(jobs, nj)) == 0) {
                if (prof && !nj) {
                    static int once = 0;
                    if (!once) { once = 1;
                        fprintf(stderr, "[ds2] absorb: nj=0 NH=%d\n", NH); }
                }
                for (int h = 0; h < NH; h++)
                    matmul(qabs + (size_t)h * KVL, q + (size_t)h * KHK,
                           (const uint8_t *)wk + (size_t)h * wk_head_sz,
                           NOPE, KVL, wk_t);
            }
        }

        /* ---- Attention over the compressed cache (MQA, 1 KV head) ----
         * PFA-style: all 128 heads scored against the shared cache in one
         * vectorized pass (port of glm5_pfa_scores/pfa_pv). */
        p3 = prof ? get_time_ms() : 0;
        if (prof) {
            static int once = 0;
            if (!once) { once = 1;
                fprintf(stderr, "[ds2] absorb multi_ret=%d\n", multi_ret); }
        }
        if (prof && layer == 0) {
            /* raw read probe over the same wk buffer: isolates memory-side
             * (NUMA/page/frequency) from dispatch/kernel cost */
            double t0 = get_time_ms();
            const float *wp = (const float *)wk;
            size_t nfloats = ds2_wk_bytes(NH, NOPE, KVL) / sizeof(float);
            float acc = 0.f;
            for (size_t i = 0; i < nfloats; i += 16) acc += wp[i];
            double t1 = get_time_ms();
            fprintf(stderr, "[ds2 memprobe] read %.1f MB serial-strided: %.2f ms (%.2f GB/s) acc=%g\n",
                    nfloats * 4.0 / 1e6, t1 - t0,
                    nfloats * 4.0 / 1e6 / ((t1 - t0) / 1e3), acc);
        }
        {
            ds2_scores_ctx(m, layer, s, q, qabs, q_comb, scores, pos, ctx,
                           NH, KVL, ROPE, NOPE, KHK, LROW);
        }

        /* ---- V projection: o[h] = Wv_b[h] @ ctx[h] ---- */
        p4 = prof ? get_time_ms() : 0;
        {
            mm_job_t jobs[128];
            const int nj = NH <= 128 ? NH : 0;
            for (int h = 0; h < nj; h++) {
                jobs[h].out = o_concat + (size_t)h * HV;
                jobs[h].x   = ctx + (size_t)h * KVL;
                jobs[h].W   = (const uint8_t *)wv + (size_t)h * wv_head_sz;
                jobs[h].n   = KVL;
                jobs[h].d   = HV;
                jobs[h].qtype = wv_t;
            }
            if (!nj || !tensor_matmul_multi(jobs, nj)) {
                for (int h = 0; h < NH; h++)
                    matmul(o_concat + (size_t)h * HV, ctx + (size_t)h * KVL,
                           (const uint8_t *)wv + (size_t)h * wv_head_sz,
                           KVL, HV, wv_t);
            }
        }
    }

    /* ---- Output projection ---- */
    p5 = prof ? get_time_ms() : 0;
    matmul(out, o_concat, lw->attn_output, NH * HV, D, lw->type_attn_output);
    if (prof) {
        static double s_q = 0, s_kv = 0, s_abs = 0, s_sc = 0, s_v = 0, s_o = 0;
        p6 = get_time_ms();
        s_q += p1 - p0; s_kv += p2 - p1; s_abs += p3 - p2;
        s_sc += p4 - p3; s_v += p5 - p4; s_o += p6 - p5;
        if (layer == 0) {
            fprintf(stderr, "[ds2 aprof] q=%.1f kv=%.1f absorb=%.1f scores=%.1f v=%.1f outproj=%.1f ms\n",
                    s_q, s_kv, s_abs, s_sc, s_v, s_o);
            s_q = s_kv = s_abs = s_sc = s_v = s_o = 0;
        }
    }
}

/* ------------------------------------------------------------------ */
/* DEBUG: full non-absorbed reference for ONE token's attention output. */
/*                                                                     */
/* The reference graph without absorption decompresses EVERY cached     */
/* latent with wkv_b and runs plain MHA over the per-head K/V. That is  */
/* the same attention the absorbed path computes, but through a         */
/* completely different tensor slice, so a mismatch localises the bug   */
/* to wk_b/wv_b/absorption rather than to Q/KV projection or RoPE.      */
/* Enabled with PICOLM_DS2_XCHECK=1.                                    */
/* ------------------------------------------------------------------ */
static void ds2_xcheck_attn(const model_t *m, int layer, run_state_t *s,
                            const float *cur, int pos, const float *got)
{
    const model_config_t *c = &m->config;
    const glm5_layer_weights_t *gw = &m->weights.glm5[layer];
    const layer_weights_t *lw = &m->weights.layers[layer];
    const int D = c->n_embd, NH = c->n_heads;
    const int QL = c->glm5_q_lora, KVL = c->glm5_kv_lora;
    const int ROPE = c->ds2_rope, NOPE = c->ds2_nope, HV = c->ds2_head_v;
    const int KHK = c->glm5_mla_head_dim, LROW = KVL + ROPE;
    const int per_head = NOPE + HV;
    const int n_kv = pos + 1;

    float *qr    = (float *)malloc((size_t)QL * sizeof(float));
    float *q     = (float *)malloc((size_t)(NH * KHK) * sizeof(float));
    float *kv    = (float *)malloc((size_t)LROW * sizeof(float));
    float *kall  = (float *)malloc((size_t)n_kv * (size_t)NH * (size_t)per_head * sizeof(float));
    float *outv  = (float *)malloc((size_t)(NH * HV) * sizeof(float));
    float *ref   = (float *)malloc((size_t)D * sizeof(float));
    float *scores = (float *)malloc((size_t)n_kv * sizeof(float));
    if (!qr || !q || !kv || !kall || !outv || !ref || !scores) goto done;

    /* q -> RMSNorm -> Wq_b, then RoPE on the last ROPE dims of each head */
    if (c->ds2_has_q_lora) {
        matmul(qr, cur, gw->mla_q_a, D, QL, gw->type_mla_q_a);
        {
            float ss = 0.f;
            for (int i = 0; i < QL; i++) ss += qr[i] * qr[i];
            const float inv = 1.0f / sqrtf(ss / QL + c->rms_norm_eps);
            const float *nw = (const float *)gw->mla_q_a_norm;
            for (int i = 0; i < QL; i++) qr[i] *= inv * nw[i];
        }
        matmul(q, qr, gw->mla_q_b, QL, NH * KHK, gw->type_mla_q_b);
    } else {
        matmul(q, cur, gw->mla_q, D, NH * KHK, gw->type_mla_q);
    }
    {
        const int half = ROPE / 2;
        const float *cp = s->ds2.rope_cos + (size_t)pos * half;
        const float *sp = s->ds2.rope_sin + (size_t)pos * half;
        for (int h = 0; h < NH; h++) {
            float *qp = q + (size_t)h * KHK + NOPE;
            for (int i = 0; i < half; i++) {
                float v0 = qp[2*i], v1 = qp[2*i+1];
                qp[2*i]   = v0*cp[i] - v1*sp[i];
                qp[2*i+1] = v0*sp[i] + v1*cp[i];
            }
        }
    }

    /* Decompress every cached latent: [t][h] = [k_nope(NOPE) ; v(HV)].
     *
     * wkv_b is [KVL rows][NH*per_head cols] (ne0 = KVL), i.e. the DOT runs
     * over the CONTIGUOUS axis and the output is indexed by the column. That
     * is ggml_mul_mat's transposed convention, which matmul() cannot express
     * (it needs row_i contiguous over n). Dequantise the whole layer slice
     * once and dot explicitly. */
    {
        const size_t wkv_row_sz = gguf_type_row_size(gw->type_mla_kv_b, KVL);
        const int ncol = NH * per_head;
        float *w32 = (float *)malloc((size_t)KVL * (size_t)ncol * sizeof(float));
        if (!w32) goto done;
        for (int r = 0; r < KVL; r++)
            dequantize_row((const uint8_t *)gw->mla_kv_b + (size_t)r * wkv_row_sz,
                           w32 + (size_t)r * ncol, ncol, gw->type_mla_kv_b);
        const float *cache = s->ds2.mla_kv + (size_t)layer * s->ds2.mla_kv_layer_stride;
        for (int t = 0; t < n_kv; t++) {
            const float *lat = cache + (size_t)t * LROW + ROPE;
            float *dst = kall + (size_t)t * ncol;
            for (int col = 0; col < ncol; col++) {
                float acc = 0.f;
                for (int r = 0; r < KVL; r++) acc += w32[(size_t)r * ncol + col] * lat[r];
                dst[col] = acc;
            }
        }
        free(w32);
    }

    const float scale = s->ds2.kq_scale;
    {
        const float *cache = s->ds2.mla_kv + (size_t)layer * s->ds2.mla_kv_layer_stride;
        for (int h = 0; h < NH; h++) {
            const float *q_nope = q + (size_t)h * KHK;
            const float *q_rope = q_nope + NOPE;
            for (int t = 0; t < n_kv; t++) {
                const float *kn = kall + ((size_t)t * NH + h) * per_head;
                const float *row = cache + (size_t)t * LROW;
                float acc = 0.f;
                for (int d = 0; d < NOPE; d++) acc += kn[d] * q_nope[d];
                for (int d = 0; d < ROPE; d++) acc += row[d] * q_rope[d];
                scores[t] = acc * scale;
            }
            float mx = -INFINITY;
            for (int t = 0; t < n_kv; t++) if (scores[t] > mx) mx = scores[t];
            float sum = 0.f;
            for (int t = 0; t < n_kv; t++) { float e = expf(scores[t]-mx); scores[t]=e; sum+=e; }
            float *oh = outv + (size_t)h * HV;
            memset(oh, 0, (size_t)HV * sizeof(float));
            for (int t = 0; t < n_kv; t++) {
                const float pr = scores[t] / sum;
                if (pr < 1e-10f) continue;
                const float *v = kall + ((size_t)t * NH + h) * per_head + NOPE;
                for (int d = 0; d < HV; d++) oh[d] += pr * v[d];
            }
        }
    }
    matmul(ref, outv, lw->attn_output, NH * HV, D, lw->type_attn_output);

    /* At pos 0 the softmax is a delta, so o[h] must equal Wv_b[h] @ v[h].
     * Compare that directly: it isolates the wv_b mapping from everything
     * else (no q, no scores, no softmax). */
    if (pos == 0) {
        const void *wkp; gguf_type_t wkt; size_t wkhs;
        const void *wvp; gguf_type_t wvt; size_t wvhs;
        ds2_kv_b_ptrs(m, layer, &wkp, &wkt, &wkhs, &wvp, &wvt, &wvhs);
        const float *cache = s->ds2.mla_kv + (size_t)layer * s->ds2.mla_kv_layer_stride;
        double worst = 0; int wr = -1;
        for (int h = 0; h < NH; h++) {
            float abs_o[128];
            matmul(abs_o, cache + ROPE, (const uint8_t *)wvp + (size_t)h * wvhs,
                   KVL, HV, wvt);
            const float *v = kall + (size_t)h * per_head + NOPE;
            double sa = 0, sd = 0;
            for (int d = 0; d < HV; d++) {
                sa += (double)v[d]*v[d];
                sd += (double)(v[d]-abs_o[d])*(v[d]-abs_o[d]);
            }
            double rel = sqrt(sd/(sa>0?sa:1));
            if (rel > worst) { worst = rel; wr = h; }
        }
        fprintf(stderr, "[ds2 xwv] pos0 worst head=%d rel=%.3e\n", wr, worst);
        {
            const uint8_t *wvh0 = (const uint8_t *)wvp;
            size_t wv_row = gguf_type_row_size(wvt, KVL);
            float row0[KVL], row1[KVL];
            dequantize_row(wvh0, row0, KVL, wvt);
            dequantize_row(wvh0 + wv_row, row1, KVL, wvt);
            /* expected: row dv=0 -> wkv_b[r][h*per_head+NOPE+0] over r */
            fprintf(stderr, "[ds2 xwv0] wv_b rows0/1 [0..3]: %.6f %.6f %.6f %.6f | %.6f %.6f %.6f %.6f\n",
                    (double)row0[0], (double)row0[1], (double)row0[2], (double)row0[3],
                    (double)row1[0], (double)row1[1], (double)row1[2], (double)row1[3]);
            /* expected from wkv_b, dequantised on the fly */
            {
                const size_t wkv_row_sz = gguf_type_row_size(gw->type_mla_kv_b, KVL);
                float d0[4], d1[4];
                for (int r = 0; r < 4; r++) {
                    float *tmp = (float *)malloc((size_t)(NH*per_head)*sizeof(float));
                    dequantize_row((const uint8_t *)gw->mla_kv_b + (size_t)r*wkv_row_sz,
                                   tmp, NH*per_head, gw->type_mla_kv_b);
                    d0[r] = tmp[0*per_head + NOPE + 0];
                    d1[r] = tmp[0*per_head + NOPE + 1];
                    free(tmp);
                }
                fprintf(stderr, "[ds2 xwv0] expect row0[0..3]=%.6f %.6f %.6f %.6f  row1[0..3]=%.6f %.6f %.6f %.6f\n",
                        (double)d0[0],(double)d0[1],(double)d0[2],(double)d0[3],
                        (double)d1[0],(double)d1[1],(double)d1[2],(double)d1[3]);
            }
        }
    }
    {
        double sa = 0, sd = 0, sm = 0;
        for (int i = 0; i < D; i++) {
            double a = ref[i], bb = got[i];
            sa += a*a; sd += (a-bb)*(a-bb);
            if (fabs(a-bb) > sm) sm = fabs(a-bb);
        }
        fprintf(stderr, "[ds2 xcheck] l%d pos=%d ref_rms=%.6f rel=%.3e max=%.3e\n",
                layer, pos, sqrt(sa/D), sqrt(sd/(sa>0?sa:1)), sm);
    }
done:
    free(qr); free(q); free(kv); free(kall); free(outv); free(ref); free(scores);
}

/* Per-layer hidden-state dump for oracle comparison against ik_llama.cpp's
 * eval-callback dumps (QORACLE_STATS=..., QORACLE_DUMP=<dir>).
 *   PICOLM_DS2_LSTATS="attn_norm,q,kv_compressed,..."  tensor names to dump
 *   PICOLM_DS2_LPOS=<pos>   dump only at this position
 *   PICOLM_DS2_LFILE=<dir>  output directory (files <name>-<layer>.f32)
 * Name map (PicoLM -> ik llama.cpp cb name):
 *   attn_norm  -> attn_norm    (post attn_norm residual, the block input)
 *   q          -> q            (after wq_b, pre-rope, [NH*KHK])
 *   q_rope     -> q_rope       (after rope, we dump the rope half per head)
 *   kv_compressed -> kv_compressed (post kv_a_norm latent, [KVL])
 *   k_rope     -> k_rope       (rope'd k_pe, [ROPE])
 *   attn_out   -> kqv_out      (after wo, pre-residual)
 *   l_out      -> inpL of next layer / final norm input (residual stream)
 *   moe_cur    -> ffn_norm output (MoE/dense FFN input)
 *   ffn_out    -> ffn_out_* (FFN contribution, pre-residual)
 */
static int  ds2_ldump_pos   = -1;
static int  ds2_ldump_init  = 0;
static char ds2_ldump_names[512];
static char ds2_ldump_dir[400];
static FILE *ds2_ldump_fp[8];
static char ds2_ldump_fn[8][64];
static int  ds2_ldump_n = 0;

static void ds2_dump_vec(const char *dir, const char *name, int layer,
                         const float *v, int n) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s-%d.f32", dir, name, layer);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(v, sizeof(float), (size_t)n, f);
    fclose(f);
}

static void ds2_layer_dump(const model_t *m, int il, int pos,
                           const float *cur, const float *attn_out,
                           const run_state_t *s) {
    const model_config_t *c = &m->config;
    if (!ds2_ldump_init) {
        ds2_ldump_init = 1;
        const char *names = getenv("PICOLM_DS2_LSTATS");
        const char *dir = getenv("PICOLM_DS2_LFILE");
        const char *p = getenv("PICOLM_DS2_LPOS");
        if (names && dir && p) {
            ds2_ldump_pos = atoi(p);
            snprintf(ds2_ldump_names, sizeof(ds2_ldump_names), "%s", names);
            snprintf(ds2_ldump_dir, sizeof(ds2_ldump_dir), "%s", dir);
            /* parse the comma list once */
            char *tok = ds2_ldump_names;
            while (*tok && ds2_ldump_n < 8) {
                ds2_ldump_fn[ds2_ldump_n][0] = '\0';
                char *e = tok;
                while (*e && *e != ',') e++;
                size_t len = (size_t)(e - tok);
                if (len >= sizeof(ds2_ldump_fn[0])) len = sizeof(ds2_ldump_fn[0]) - 1;
                memcpy(ds2_ldump_fn[ds2_ldump_n], tok, len);
                ds2_ldump_fn[ds2_ldump_n][len] = '\0';
                ds2_ldump_n++;
                if (!*e) break;
                tok = e + 1;
            }
        }
    }
    if (ds2_ldump_n == 0 || pos != ds2_ldump_pos) return;
    const int D = c->n_embd;
    const int NH = c->n_heads, KHK = c->glm5_mla_head_dim;
    for (int i = 0; i < ds2_ldump_n; i++) {
        const char *nm = ds2_ldump_fn[i];
        if (!strcmp(nm, "attn_norm")) {
            ds2_dump_vec(ds2_ldump_dir, "attn_norm", il, cur, D);
        } else if (!strcmp(nm, "l_out")) {
            ds2_dump_vec(ds2_ldump_dir, "l_out", il, s->x, D);
        } else if (!strcmp(nm, "attn_out")) {
            ds2_dump_vec(ds2_ldump_dir, "attn_out", il, attn_out, D);
        } else if (!strcmp(nm, "q")) {
            /* [NH*KHK] after wq_b, pre-rope -- matches ik's cb(q,"q") */
            ds2_dump_vec(ds2_ldump_dir, "q", il, s->ds2.scratch + c->glm5_q_lora,
                         (size_t)NH * KHK);
        } else if (!strcmp(nm, "kv_compressed")) {
            ds2_dump_vec(ds2_ldump_dir, "kv_compressed", il,
                         s->ds2.scratch + c->glm5_q_lora + (size_t)NH * KHK
                         + (size_t)NH * c->glm5_kv_lora, c->glm5_kv_lora);
        } else if (!strcmp(nm, "attn_out_prewo")) {
            ds2_dump_vec(ds2_ldump_dir, "attn_out_prewo", il, attn_out, D);
        }
    }
}

/* Dense SwiGLU FFN for the leading dense layers (standard ffn_gate/up/down). */
static void ds2_dense_ffn(const model_t *m, const layer_weights_t *lw,
                          run_state_t *s, const float *cur, float *out)
{
    const model_config_t *c = &m->config;
    const int D = c->n_embd;
    const int FF = c->n_ffn;

    matmul(s->hb,  cur, lw->ffn_gate, D, FF, lw->type_ffn_gate);
    matmul(s->hb2, cur, lw->ffn_up,   D, FF, lw->type_ffn_up);
    silu_mul(s->hb, s->hb, s->hb2, FF);
    matmul(out, s->hb, lw->ffn_down, FF, D, lw->type_ffn_down);
}

/* ------------------------------------------------------------------ */
/* Batched prefill for DeepSeek V3                                     */
/* Batches Q/KV projections and output head via matmul_batch.          */
/* Attention scoring and MoE FFN run per-token (inherently sequential  */
/* for MLA since each token attends to all previous positions).        */
/* ------------------------------------------------------------------ */
float *model_forward_deepseek2(model_t *m, int token, int pos);

/* Batched RMSNorm helper (mirrors model_core.c's batch_rmsnorm_ctx_t) */
typedef struct {
    float *out, *x;
    const float *w;
    int dim;
    float eps;
} ds2_batch_rmsnorm_ctx_t;

static void ds2_batch_rmsnorm_task(int idx, void *ctxp) {
    ds2_batch_rmsnorm_ctx_t *ctx = (ds2_batch_rmsnorm_ctx_t *)ctxp;
    rmsnorm(ctx->out + (size_t)idx * ctx->dim, ctx->x + (size_t)idx * ctx->dim,
            ctx->w, ctx->dim, ctx->eps);
}

/* Batched silu_mul helper (mirrors model_core.c's silu_mul_ctx_t) */
typedef struct {
    float *a, *b;
    int dim;
} ds2_silu_mul_ctx_t;

static void ds2_silu_mul_task(int idx, void *ctxp) {
    ds2_silu_mul_ctx_t *ctx = (ds2_silu_mul_ctx_t *)ctxp;
    silu_mul(ctx->a + (size_t)idx * ctx->dim,
             ctx->a + (size_t)idx * ctx->dim,
             ctx->b + (size_t)idx * ctx->dim,
             ctx->dim);
}

/* Per-token MLA attention (absorbed) for the batched prefill. Takes the
 * per-token Q row [NH*KHK, already RoPE'd] and the cache for this layer
 * (row pos already stored), writes o_concat [NH*HV] and the output-projected
 * block result [D]. Same math as ds2_mla_forward's attention half. */
static void ds2_mla_attention_forward(const model_t *m, int layer,
                                      run_state_t *s, const float *q_in,
                                      int pos, float *o_concat, float *out,
                                      int D, int NH, int KVL, int ROPE,
                                      int NOPE, int HV, int KHK, int LROW,
                                      int HVNH)
{
    float *sc = s->ds2.scratch;
    /* Canonical order subset (decode's qr/q/kvr/o_concat slots are free
     * during prefill): qabs(NH*KVL) + q_comb(NH*LROW) + ctx(NH*KVL)
     * + scores(NH*DS2_PFA_BK). */
    float *qabs   = sc; sc += (size_t)NH * KVL;
    float *q_comb = sc; sc += (size_t)NH * LROW;
    float *ctx    = sc; sc += (size_t)NH * KVL;
    float *scores = sc; sc += (size_t)NH * DS2_PFA_BK;
    (void)sc;

    /* Absorb q_nope into latent: q_abs[h] = Wk_b[h] @ q_nope[h] */
    {
        const void *wk; gguf_type_t wk_t; size_t wk_head_sz;
        const void *wv; gguf_type_t wv_t; size_t wv_head_sz;
        ds2_kv_b_ptrs(m, layer, &wk, &wk_t, &wk_head_sz, &wv, &wv_t, &wv_head_sz);
        mm_job_t jobs[128];
        const int nj = NH <= 128 ? NH : 0;
        for (int h = 0; h < nj; h++) {
            jobs[h].out = qabs + (size_t)h * KVL;
            jobs[h].x   = q_in + (size_t)h * KHK;
            jobs[h].W   = (const uint8_t *)wk + (size_t)h * wk_head_sz;
            jobs[h].n   = NOPE;
            jobs[h].d   = KVL;
            jobs[h].qtype = wk_t;
        }
        if (!nj || !tensor_matmul_multi(jobs, nj)) {
            for (int h = 0; h < NH; h++)
                matmul(qabs + (size_t)h * KVL, q_in + (size_t)h * KHK,
                       (const uint8_t *)wk + (size_t)h * wk_head_sz,
                       NOPE, KVL, wk_t);
        }
    }

    /* Attention scoring + softmax + weighted sum over the shared cache:
     * PFA-style, all heads in one pass (see ds2_scores_ctx). */
    ds2_scores_ctx(m, layer, s, q_in, qabs, q_comb, scores, pos, ctx,
                   NH, KVL, ROPE, NOPE, KHK, LROW);

    /* V projection: o[h] = Wv_b[h] @ ctx[h] */
    {
        const void *wv; gguf_type_t wv_t; size_t wv_head_sz;
        const void *wk_dummy; gguf_type_t wk_dummy_t; size_t wk_dummy_sz;
        ds2_kv_b_ptrs(m, layer, &wk_dummy, &wk_dummy_t, &wk_dummy_sz,
                      &wv, &wv_t, &wv_head_sz);
        mm_job_t jobs[128];
        const int nj = NH <= 128 ? NH : 0;
        for (int h = 0; h < nj; h++) {
            jobs[h].out = o_concat + (size_t)h * HV;
            jobs[h].x   = ctx + (size_t)h * KVL;
            jobs[h].W   = (const uint8_t *)wv + (size_t)h * wv_head_sz;
            jobs[h].n   = KVL;
            jobs[h].d   = HV;
            jobs[h].qtype = wv_t;
        }
        if (!nj || !tensor_matmul_multi(jobs, nj)) {
            for (int h = 0; h < NH; h++)
                matmul(o_concat + (size_t)h * HV, ctx + (size_t)h * KVL,
                       (const uint8_t *)wv + (size_t)h * wv_head_sz,
                       KVL, HV, wv_t);
        }
    }

    /* Output projection: out = attn_output @ o_concat */
    matmul(out, o_concat, m->weights.layers[layer].attn_output, HVNH, D,
           m->weights.layers[layer].type_attn_output);
}

float *model_forward_prefill_deepseek2(model_t *m, const int *tokens, int n_tokens,
                                       int start_pos, volatile int *interrupt)
{
    /* PICOLM_DS2_NO_BATCH=1: fall back to per-token forward for logit comparison. */
    if (getenv("PICOLM_DS2_NO_BATCH")) {
        for (int i = 0; i < n_tokens; i++) {
            if (interrupt && *interrupt) break;
            (void)model_forward_deepseek2(m, tokens[i], start_pos + i);
        }
        return m->state.logits;
    }

    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;
    const int D = c->n_embd;
    const int NH = c->n_heads;
    const int QL = c->glm5_q_lora;
    const int KVL = c->glm5_kv_lora;
    const int ROPE = c->ds2_rope;
    const int NOPE = c->ds2_nope;
    const int HV = c->ds2_head_v;
    const int KHK = c->glm5_mla_head_dim;   /* NOPE + ROPE */
    const int LROW = KVL + ROPE;            /* cache row width */
    const int HVNH = NH * HV;               /* output concat width */
    const size_t NT = (size_t)n_tokens;

    /* Per-token scratch arena. Buffers:
     *   x_batch    residual stream [D]
     *   xb_batch   RMSNorm'd input [D]
     *   q_a_batch  wq_a output [QL]   (separate: q_batch is overwritten by
     *              wq_b with a larger per-token stride, so a fused layout
     *              would alias later tokens' wq_a outputs -- the bug that
     *              broke the first batched-prefill attempt)
     *   q_batch    Q projection output [NH*KHK]
     *   kvr_batch  KV projection output [LROW]
     *   o_batch    attention output concat [max(HVNH, D)]
     *   attn_batch after output projection / FFN output [D]
     *   hb_batch   dense FFN hidden [n_ffn]
     *   hb2_batch  dense FFN hidden 2 [n_ffn]
     */
    const int ffn_buf = c->n_ffn;
    const int max_buf = (HVNH > D) ? HVNH : D;
    size_t total_bufs = NT * ((size_t)D * 3 + (size_t)QL + (size_t)NH * (size_t)KHK
                            + (size_t)LROW + (size_t)max_buf
                            + (size_t)ffn_buf * 2);
    float *pf = (float *)malloc(total_bufs * sizeof(float));
    if (!pf) {
        /* OOM: fall back to per-token */
        for (int i = 0; i < n_tokens; i++) {
            if (interrupt && *interrupt) break;
            (void)model_forward_deepseek2(m, tokens[i], start_pos + i);
        }
        return s->logits;
    }
    float *p = pf;
    float *x_batch    = p; p += NT * D;
    float *xb_batch   = p; p += NT * D;
    float *q_a_batch  = p; p += NT * (size_t)QL;
    float *q_batch    = p; p += NT * (size_t)NH * KHK;
    float *kvr_batch  = p; p += NT * LROW;
    float *o_batch    = p; p += NT * max_buf;
    float *attn_batch = p; p += NT * D;
    float *hb_batch   = p; p += NT * ffn_buf;
    float *hb2_batch  = p; p += NT * ffn_buf;

    /* ---- Embed all tokens ---- */
    for (int t = 0; t < n_tokens; t++) {
        const size_t row_bytes = gguf_type_row_size(w->type_token_embd, D);
        const void *embd_row = (const uint8_t *)w->token_embd
                             + (size_t)tokens[t] * row_bytes;
        dequantize_row(embd_row, x_batch + (size_t)t * D, D, w->type_token_embd);
    }

    /* ---- Layer loop ---- */
    for (int il = 0; il < c->n_layers; il++) {
        const layer_weights_t *lw = &w->layers[il];
        const glm5_layer_weights_t *gw = &w->glm5[il];
        BENCH_LAYER_START();

        if (interrupt && *interrupt) { free(pf); return NULL; }

        /* ---- Attention block ---- */
        { ds2_batch_rmsnorm_ctx_t rctx =
              { xb_batch, x_batch, s->attn_norm_w[il], D, c->rms_norm_eps };
          tensor_parallel_for(n_tokens, ds2_batch_rmsnorm_task, &rctx); }

        /* Q projection: wq_a (D -> QL), batched; then per-token RMSNorm and
         * wq_b (QL -> NH*KHK). q_a_batch keeps the wq_a outputs alive while
         * q_batch is filled at its own (larger) stride. */
        if (c->ds2_has_q_lora) {
            tensor_set_repacked(NULL);
            matmul_batch(q_a_batch, xb_batch, n_tokens, gw->mla_q_a, D, QL,
                         gw->type_mla_q_a);
            for (int t = 0; t < n_tokens; t++) {
                float *qt = q_batch + (size_t)t * (size_t)NH * KHK;
                const float *qa = q_a_batch + (size_t)t * QL;
                float *tmp = s->hb;   /* >= QL floats (n_ffn >= 1536) */
                rmsnorm(tmp, qa, (const float *)gw->mla_q_a_norm, QL, c->rms_norm_eps);
                matmul(qt, tmp, gw->mla_q_b, QL, NH * KHK, gw->type_mla_q_b);
            }
        } else {
            tensor_set_repacked(NULL);
            matmul_batch(q_batch, xb_batch, n_tokens, gw->mla_q, D, NH * KHK,
                         gw->type_mla_q);
        }

        /* KV projection: wkv_a_mqa (D -> LROW), batched; then per-token
         * RMSNorm on the latent (first KVL dims). */
        tensor_set_repacked(NULL);
        matmul_batch(kvr_batch, xb_batch, n_tokens, gw->mla_kv_a_mqa, D, LROW,
                     gw->type_mla_kv_a_mqa);
        {
            const float *kv_norm_w = (const float *)gw->mla_kv_a_norm;
            for (int t = 0; t < n_tokens; t++) {
                float *kvr = kvr_batch + (size_t)t * LROW;
                float ss = 0.f;
                for (int i = 0; i < KVL; i++) ss += kvr[i] * kvr[i];
                const float inv = 1.0f / sqrtf(ss / KVL + c->rms_norm_eps);
                for (int i = 0; i < KVL; i++) kvr[i] *= inv * kv_norm_w[i];
            }
        }

        /* RoPE on q_rope (per head, last ROPE dims, pairwise) and on k_pe
         * via tensor.c's rope() (AVX-512/AVX/NEON). */
        {
            const int half = ROPE / 2;
            for (int t = 0; t < n_tokens; t++) {
                const int pos = start_pos + t;
                const float *cos_pos = s->ds2.rope_cos + (size_t)pos * half;
                const float *sin_pos = s->ds2.rope_sin + (size_t)pos * half;
                float *qb = q_batch + (size_t)t * (size_t)NH * KHK;
                for (int h = 0; h < NH; h++)
                    rope(qb + (size_t)h * KHK + NOPE, NULL, ROPE, 1, 0,
                         cos_pos, sin_pos, 0, half);
                rope(kvr_batch + (size_t)t * LROW + KVL, NULL, ROPE, 1, 0,
                     cos_pos, sin_pos, 0, half);
            }
        }

        /* Store cache rows for all tokens: [k_pe(ROPE); kv_latent(KVL)] */
        {
            float *cache = s->ds2.mla_kv + (size_t)il * s->ds2.mla_kv_layer_stride;
            for (int t = 0; t < n_tokens; t++) {
                float *row = cache + (size_t)(start_pos + t) * LROW;
                const float *kvr = kvr_batch + (size_t)t * LROW;
                memcpy(row, kvr + KVL, (size_t)ROPE * sizeof(float));    /* k_pe */
                memcpy(row + ROPE, kvr, (size_t)KVL * sizeof(float));    /* latent */
            }
        }

        /* Per-token MLA attention (absorb + score + softmax + V + output proj) */
        for (int t = 0; t < n_tokens; t++) {
            const int pos = start_pos + t;
            ds2_mla_attention_forward(m, il, s,
                                      q_batch + (size_t)t * (size_t)NH * KHK,
                                      pos,
                                      o_batch + (size_t)t * max_buf,
                                      attn_batch + (size_t)t * D,
                                      D, NH, KVL, ROPE, NOPE, HV, KHK, LROW,
                                      HVNH);
        }

        /* Residual add: x += attn_out */
        for (int t = 0; t < n_tokens; t++) {
            vec_add(x_batch + (size_t)t * D, attn_batch + (size_t)t * D, D);
        }

        /* ---- FFN block ---- */
        { ds2_batch_rmsnorm_ctx_t rctx =
              { xb_batch, x_batch, s->post_attn_norm_w[il], D, c->rms_norm_eps };
          tensor_parallel_for(n_tokens, ds2_batch_rmsnorm_task, &rctx); }

        if (il < c->n_layer_dense_lead) {
            /* Dense SwiGLU, batched */
            matmul_dual_batch(hb_batch, hb2_batch, xb_batch, n_tokens,
                              lw->ffn_gate, lw->ffn_up, D, ffn_buf,
                              lw->type_ffn_gate, lw->type_ffn_up);
            { ds2_silu_mul_ctx_t sctx = { hb_batch, hb2_batch, ffn_buf };
              tensor_parallel_for(n_tokens, ds2_silu_mul_task, &sctx); }
            matmul_batch(attn_batch, hb_batch, n_tokens, lw->ffn_down, ffn_buf, D,
                         lw->type_ffn_down);
        } else {
            /* MoE FFN: per-token (variable expert selection) */
            for (int t = 0; t < n_tokens; t++) {
                memcpy(s->ds2.cur, xb_batch + (size_t)t * D, D * sizeof(float));
                glm5_moe_ffn(m, gw, lw, s, s->ds2.cur, s->ds2.attn_out, il);
                memcpy(attn_batch + (size_t)t * D, s->ds2.attn_out, D * sizeof(float));
            }
        }

        /* Residual add: x += ffn_out */
        for (int t = 0; t < n_tokens; t++) {
            vec_add(x_batch + (size_t)t * D, attn_batch + (size_t)t * D, D);
        }

#ifdef PICOLM_VIZ
        for (int t = 0; t < n_tokens; t++)
            viz_push_row(il, start_pos + t, x_batch + (size_t)t * D, D);
#endif
        steer_hook(il, x_batch, n_tokens, D, start_pos);

        BENCH_LAYER_END(il, 1);
    }

    /* Final: RMSNorm(output_norm) then the output head, last token only.
     * Uses s->ds2.cur; x_batch lives inside pf, freed below. */
    {
        float *cur = s->ds2.cur;
        memcpy(cur, x_batch + (size_t)(n_tokens - 1) * D, D * sizeof(float));
        rmsnorm(cur, cur, s->output_norm_w, D, c->rms_norm_eps);
        matmul(s->logits, cur, w->output, D, c->vocab_size, w->type_output);
    }
    free(pf);
    return s->logits;
}

/* ------------------------------------------------------------------ */
/* PFA-style attention tail (port of model_glm5.c's glm5_pfa_scores /  */
/* glm5_pfa_pv, adapted for the ds2 cache: rows are LROW=576 wide,      */
/* [k_pe(ROPE) ; kv(KVL)], so both kernels take an explicit row         */
/* stride). AVX-512 with AVX2 and scalar fallbacks, same tiling as      */
/* GLM5: 4 query rows x 4 key rows per iteration.                      */
/* ------------------------------------------------------------------ */

#if defined(PICOLM_AVX512)
static inline __m512 ds2_pfa_hsum16(const __m512 a[16])
{
    __m512 t[8], u[4];
    for (int i = 0; i < 8; i++)
        t[i] = _mm512_add_ps(_mm512_unpacklo_ps(a[2*i], a[2*i+1]), _mm512_unpackhi_ps(a[2*i], a[2*i+1]));
    for (int i = 0; i < 4; i++)
        u[i] = _mm512_add_ps(_mm512_shuffle_ps(t[2*i], t[2*i+1], 0x44), _mm512_shuffle_ps(t[2*i], t[2*i+1], 0xEE));
    __m128 v[4];
    for (int i = 0; i < 4; i++) {
        const __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(u[i]), 1));
        const __m256 h  = _mm256_add_ps(_mm512_castps512_ps256(u[i]), hi);
        v[i] = _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1));
    }
    __m512 r = _mm512_castps128_ps512(v[0]);
    r = _mm512_insertf32x4(r, v[1], 1);
    r = _mm512_insertf32x4(r, v[2], 2);
    return _mm512_insertf32x4(r, v[3], 3);
}
#endif

#if defined(PICOLM_AVX2)
static inline float ds2_hsum256(const __m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}
#endif

/* S[r*DS2_PFA_BK + j] = dot(qp[r], kb[j*kstride]) * scale, over nk keys,
 * KVD elements per dot. kb rows are kstride apart (ds2 cache: LROW). */
static void ds2_pfa_scores(const float *const *qp, int R, const float *kb,
                           int nk, int KVD, int kstride, float *S, float scale)
{
    int r = 0;
#if defined(PICOLM_AVX512)
    if ((KVD & 15) == 0) {
        for (; r + 3 < R; r += 4) {
            int j = 0;
            for (; j + 3 < nk; j += 4) {
                __m512 a00 = _mm512_setzero_ps(), a01 = a00, a02 = a00, a03 = a00,
                       a10 = a00, a11 = a00, a12 = a00, a13 = a00,
                       a20 = a00, a21 = a00, a22 = a00, a23 = a00,
                       a30 = a00, a31 = a00, a32 = a00, a33 = a00;
                const float *k0 = kb + (size_t)(j + 0) * kstride,
                            *k1 = kb + (size_t)(j + 1) * kstride,
                            *k2 = kb + (size_t)(j + 2) * kstride,
                            *k3 = kb + (size_t)(j + 3) * kstride;
                const float *q0 = qp[r], *q1 = qp[r + 1], *q2 = qp[r + 2], *q3 = qp[r + 3];
                for (int d = 0; d < KVD; d += 16) {
                    const __m512 v0 = _mm512_loadu_ps(k0 + d), v1 = _mm512_loadu_ps(k1 + d),
                                 v2 = _mm512_loadu_ps(k2 + d), v3 = _mm512_loadu_ps(k3 + d);
                    const __m512 w0 = _mm512_loadu_ps(q0 + d), w1 = _mm512_loadu_ps(q1 + d),
                                 w2 = _mm512_loadu_ps(q2 + d), w3 = _mm512_loadu_ps(q3 + d);
                    a00 = _mm512_fmadd_ps(v0, w0, a00); a01 = _mm512_fmadd_ps(v1, w0, a01);
                    a02 = _mm512_fmadd_ps(v2, w0, a02); a03 = _mm512_fmadd_ps(v3, w0, a03);
                    a10 = _mm512_fmadd_ps(v0, w1, a10); a11 = _mm512_fmadd_ps(v1, w1, a11);
                    a12 = _mm512_fmadd_ps(v2, w1, a12); a13 = _mm512_fmadd_ps(v3, w1, a13);
                    a20 = _mm512_fmadd_ps(v0, w2, a20); a21 = _mm512_fmadd_ps(v1, w2, a21);
                    a22 = _mm512_fmadd_ps(v2, w2, a22); a23 = _mm512_fmadd_ps(v3, w2, a23);
                    a30 = _mm512_fmadd_ps(v0, w3, a30); a31 = _mm512_fmadd_ps(v1, w3, a31);
                    a32 = _mm512_fmadd_ps(v2, w3, a32); a33 = _mm512_fmadd_ps(v3, w3, a33);
                }
                const __m512 acc[16] = { a00, a01, a02, a03, a10, a11, a12, a13,
                                         a20, a21, a22, a23, a30, a31, a32, a33 };
                const __m512 sum = _mm512_mul_ps(ds2_pfa_hsum16(acc), _mm512_set1_ps(scale));
                _mm_storeu_ps(S + (size_t)(r + 0) * DS2_PFA_BK + j, _mm512_extractf32x4_ps(sum, 0));
                _mm_storeu_ps(S + (size_t)(r + 1) * DS2_PFA_BK + j, _mm512_extractf32x4_ps(sum, 1));
                _mm_storeu_ps(S + (size_t)(r + 2) * DS2_PFA_BK + j, _mm512_extractf32x4_ps(sum, 2));
                _mm_storeu_ps(S + (size_t)(r + 3) * DS2_PFA_BK + j, _mm512_extractf32x4_ps(sum, 3));
            }
            for (; j < nk; j++)
                for (int i = 0; i < 4; i++)
                    S[(size_t)(r + i) * DS2_PFA_BK + j] =
                        vec_dot_f32_f32(kb + (size_t)j * kstride, qp[r + i], KVD) * scale;
        }
    }
#elif defined(PICOLM_AVX2)
    if ((KVD & 7) == 0) {
        for (; r + 3 < R; r += 4) {
            int j = 0;
            for (; j + 3 < nk; j += 4) {
                __m256 a00 = _mm256_setzero_ps(), a01 = a00, a02 = a00, a03 = a00,
                       a10 = a00, a11 = a00, a12 = a00, a13 = a00,
                       a20 = a00, a21 = a00, a22 = a00, a23 = a00,
                       a30 = a00, a31 = a00, a32 = a00, a33 = a00;
                const float *k0 = kb + (size_t)(j + 0) * kstride,
                            *k1 = kb + (size_t)(j + 1) * kstride,
                            *k2 = kb + (size_t)(j + 2) * kstride,
                            *k3 = kb + (size_t)(j + 3) * kstride;
                const float *q0 = qp[r], *q1 = qp[r + 1], *q2 = qp[r + 2], *q3 = qp[r + 3];
                for (int d = 0; d < KVD; d += 8) {
                    const __m256 v0 = _mm256_loadu_ps(k0 + d), v1 = _mm256_loadu_ps(k1 + d),
                                 v2 = _mm256_loadu_ps(k2 + d), v3 = _mm256_loadu_ps(k3 + d);
                    const __m256 w0 = _mm256_loadu_ps(q0 + d), w1 = _mm256_loadu_ps(q1 + d),
                                 w2 = _mm256_loadu_ps(q2 + d), w3 = _mm256_loadu_ps(q3 + d);
                    a00 = _mm256_fmadd_ps(v0, w0, a00); a01 = _mm256_fmadd_ps(v1, w0, a01);
                    a02 = _mm256_fmadd_ps(v2, w0, a02); a03 = _mm256_fmadd_ps(v3, w0, a03);
                    a10 = _mm256_fmadd_ps(v0, w1, a10); a11 = _mm256_fmadd_ps(v1, w1, a11);
                    a12 = _mm256_fmadd_ps(v2, w1, a12); a13 = _mm256_fmadd_ps(v3, w1, a13);
                    a20 = _mm256_fmadd_ps(v0, w2, a20); a21 = _mm256_fmadd_ps(v1, w2, a21);
                    a22 = _mm256_fmadd_ps(v2, w2, a22); a23 = _mm256_fmadd_ps(v3, w2, a23);
                    a30 = _mm256_fmadd_ps(v0, w3, a30); a31 = _mm256_fmadd_ps(v1, w3, a31);
                    a32 = _mm256_fmadd_ps(v2, w3, a32); a33 = _mm256_fmadd_ps(v3, w3, a33);
                }
                float *s0 = S + (size_t)(r + 0) * DS2_PFA_BK + j, *s1 = S + (size_t)(r + 1) * DS2_PFA_BK + j;
                float *s2 = S + (size_t)(r + 2) * DS2_PFA_BK + j, *s3 = S + (size_t)(r + 3) * DS2_PFA_BK + j;
                s0[0] = ds2_hsum256(a00) * scale; s0[1] = ds2_hsum256(a01) * scale;
                s0[2] = ds2_hsum256(a02) * scale; s0[3] = ds2_hsum256(a03) * scale;
                s1[0] = ds2_hsum256(a10) * scale; s1[1] = ds2_hsum256(a11) * scale;
                s1[2] = ds2_hsum256(a12) * scale; s1[3] = ds2_hsum256(a13) * scale;
                s2[0] = ds2_hsum256(a20) * scale; s2[1] = ds2_hsum256(a21) * scale;
                s2[2] = ds2_hsum256(a22) * scale; s2[3] = ds2_hsum256(a23) * scale;
                s3[0] = ds2_hsum256(a30) * scale; s3[1] = ds2_hsum256(a31) * scale;
                s3[2] = ds2_hsum256(a32) * scale; s3[3] = ds2_hsum256(a33) * scale;
            }
            for (; j < nk; j++)
                for (int i = 0; i < 4; i++)
                    S[(size_t)(r + i) * DS2_PFA_BK + j] =
                        vec_dot_f32_f32(kb + (size_t)j * kstride, qp[r + i], KVD) * scale;
        }
    }
#endif
    for (; r < R; r++)
        for (int j = 0; j < nk; j++)
            S[(size_t)r * DS2_PFA_BK + j] =
                vec_dot_f32_f32(kb + (size_t)j * kstride, qp[r], KVD) * scale;
}

/* O[r*KVL + d] = sum_j P[r*DS2_PFA_BK + j] * kb[j*kstride + d]  (O is zeroed) */
static void ds2_pfa_pv(float *O, int R, const float *P, int nk,
                       const float *kb, int kstride, int KVL)
{
    int r = 0;
#if defined(PICOLM_AVX512)
    if ((KVL & 63) == 0) {
        for (; r + 3 < R; r += 4) {
            for (int c0 = 0; c0 < KVL; c0 += 64) {
                __m512 a[4][4];
                for (int i = 0; i < 4; i++) {
                    for (int u = 0; u < 4; u++)
                        a[i][u] = _mm512_loadu_ps(O + (size_t)(r + i) * KVL + c0 + 16 * u);
                }
                for (int j = 0; j < nk; j++) {
                    const float *kr = kb + (size_t)j * kstride + c0;
                    const __m512 v0 = _mm512_loadu_ps(kr), v1 = _mm512_loadu_ps(kr + 16),
                                 v2 = _mm512_loadu_ps(kr + 32), v3 = _mm512_loadu_ps(kr + 48);
                    for (int i = 0; i < 4; i++) {
                        const __m512 pb = _mm512_set1_ps(P[(size_t)(r + i) * DS2_PFA_BK + j]);
                        a[i][0] = _mm512_fmadd_ps(pb, v0, a[i][0]);
                        a[i][1] = _mm512_fmadd_ps(pb, v1, a[i][1]);
                        a[i][2] = _mm512_fmadd_ps(pb, v2, a[i][2]);
                        a[i][3] = _mm512_fmadd_ps(pb, v3, a[i][3]);
                    }
                }
                for (int i = 0; i < 4; i++) {
                    for (int u = 0; u < 4; u++)
                        _mm512_storeu_ps(O + (size_t)(r + i) * KVL + c0 + 16 * u, a[i][u]);
                }
            }
        }
    }
#elif defined(PICOLM_AVX2)
    if ((KVL & 31) == 0) {
        for (; r + 3 < R; r += 4) {
            for (int c0 = 0; c0 < KVL; c0 += 32) {
                __m256 a[4][4];
                for (int i = 0; i < 4; i++) {
                    for (int u = 0; u < 4; u++)
                        a[i][u] = _mm256_loadu_ps(O + (size_t)(r + i) * KVL + c0 + 8 * u);
                }
                for (int j = 0; j < nk; j++) {
                    const float *kr = kb + (size_t)j * kstride + c0;
                    const __m256 v0 = _mm256_loadu_ps(kr), v1 = _mm256_loadu_ps(kr + 8),
                                 v2 = _mm256_loadu_ps(kr + 16), v3 = _mm256_loadu_ps(kr + 24);
                    for (int i = 0; i < 4; i++) {
                        const __m256 pb = _mm256_set1_ps(P[(size_t)(r + i) * DS2_PFA_BK + j]);
                        a[i][0] = _mm256_fmadd_ps(pb, v0, a[i][0]);
                        a[i][1] = _mm256_fmadd_ps(pb, v1, a[i][1]);
                        a[i][2] = _mm256_fmadd_ps(pb, v2, a[i][2]);
                        a[i][3] = _mm256_fmadd_ps(pb, v3, a[i][3]);
                    }
                }
                for (int i = 0; i < 4; i++) {
                    for (int u = 0; u < 4; u++)
                        _mm256_storeu_ps(O + (size_t)(r + i) * KVL + c0 + 8 * u, a[i][u]);
                }
            }
        }
    }
#endif
    for (; r < R; r++) {
        float *o = O + (size_t)r * KVL;
        for (int j = 0; j < nk; j++) {
            const float w = P[(size_t)r * DS2_PFA_BK + j];
            const float *kr = kb + (size_t)j * kstride;
            for (int d = 0; d < KVL; d++) o[d] += w * kr[d];
        }
    }
}

/* Combined scores + softmax + PV over ALL heads at once (MQA: the cache is
 * shared). q_in: [NH * KHK] (RoPE'd), qabs: [NH * KVL] (absorbed q).
 * q_comb [NH*LROW] and scores [NH*DS2_PFA_BK] are per-caller scratch.
 * Writes ctx [NH * KVL].
 *
 * Keys are processed in DS2_PFA_BK-sized blocks with an ONLINE softmax
 * (running max m and sum l per head, GLM-5 PFA style): the scores buffer
 * only ever holds one block (nk <= DS2_PFA_BK per row), so context length
 * is unbounded by the scratch size. Final ctx = (sum_b sum_t e^{s_t-m} v_t)
 * / l. */
static void ds2_scores_ctx(const model_t *m, int layer, run_state_t *s,
                           const float *q_in, const float *qabs,
                           float *q_comb, float *scores,
                           int pos, float *ctx,
                           int NH, int KVL, int ROPE, int NOPE, int KHK, int LROW)
{
    const int n_kv = pos + 1;

    /* q_comb[h] = [q_rope(ROPE) ; q_abs(KVL)] -- rope first, matching cache. */
    for (int h = 0; h < NH; h++) {
        memcpy(q_comb + (size_t)h * LROW, q_in + (size_t)h * KHK + NOPE,
               (size_t)ROPE * sizeof(float));
        memcpy(q_comb + (size_t)h * LROW + ROPE, qabs + (size_t)h * KVL,
               (size_t)KVL * sizeof(float));
    }

    const float *qp[128];
    for (int h = 0; h < NH && h < 128; h++) qp[h] = q_comb + (size_t)h * LROW;

    float *cache = s->ds2.mla_kv + (size_t)layer * s->ds2.mla_kv_layer_stride;

    float run_max[128], run_sum[128];
    for (int h = 0; h < NH && h < 128; h++) { run_max[h] = -INFINITY; run_sum[h] = 0.f; }
    memset(ctx, 0, (size_t)NH * KVL * sizeof(float));

    for (int kb0 = 0; kb0 < n_kv; kb0 += DS2_PFA_BK) {
        const int nk = (n_kv - kb0 < DS2_PFA_BK) ? n_kv - kb0 : DS2_PFA_BK;
        const float *kb = cache + (size_t)kb0 * LROW;
        ds2_pfa_scores(qp, NH, kb, nk, LROW, LROW, scores, s->ds2.kq_scale);

        /* online softmax update per head */
        for (int h = 0; h < NH && h < 128; h++) {
            float *sr = scores + (size_t)h * DS2_PFA_BK;
            float mx = -INFINITY;
            for (int t = 0; t < nk; t++) if (sr[t] > mx) mx = sr[t];
            const float mnew = (mx > run_max[h]) ? mx : run_max[h];
            const float alpha = expf(run_max[h] - mnew);   /* expf(-inf - finite)=0 on 1st block */
            float *ch = ctx + (size_t)h * KVL;
            for (int d = 0; d < KVL; d++) ch[d] *= alpha;
            run_sum[h] *= alpha;
            for (int t = 0; t < nk; t++) { float e = expf(sr[t] - mnew); sr[t] = e; run_sum[h] += e; }
            run_max[h] = mnew;
        }

        /* ctx += P @ V  (V = latent part of this block's cache rows) */
        ds2_pfa_pv(ctx, NH, scores, nk, kb + ROPE, LROW, KVL);
    }

    for (int h = 0; h < NH && h < 128; h++) {
        if (!(run_sum[h] > 0.f)) run_sum[h] = 1.f;   /* guard */
        const float inv = 1.0f / run_sum[h];
        float *ch = ctx + (size_t)h * KVL;
        for (int d = 0; d < KVL; d++) ch[d] *= inv;
    }
}

/* ------------------------------------------------------------------ */
/* Single-token decode forward for DeepSeek V3                        */
/* ------------------------------------------------------------------ */
float *model_forward_deepseek2(model_t *m, int token, int pos)
{
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;
    const int D = c->n_embd;

    if (pos >= c->max_seq_len) {
        fprintf(stderr, "WARN: model_forward_deepseek2 pos=%d >= max_seq_len=%d\n",
                pos, c->max_seq_len);
        return s->logits;
    }

    /* Embedding lookup into s->x (the residual stream). */
    {
        const size_t row_bytes = gguf_type_row_size(w->type_token_embd, D);
        const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
        dequantize_row(embd_row, s->x, D, w->type_token_embd);
    }

    if (getenv("PICOLM_DS2_NORM1")) {
        static int done = 0;
        if (!done) { done = 1;
            for (int i = 0; i < D; i++) {
                s->output_norm_w[i] = 1.0f;
                for (int l = 0; l < c->n_layers; l++) {
                    s->attn_norm_w[l][i] = 1.0f;
                    s->post_attn_norm_w[l][i] = 1.0f;
                }
            }
            fprintf(stderr, "[ds2] forced all norm weights to 1.0\n");
        }
    }
    float *cur      = s->ds2.cur;       /* [D] normed block input */
    float *attn_out = s->ds2.attn_out;  /* [D] attention / FFN block output */
    const int prof_on = (getenv("PICOLM_DS2_PROF") != NULL);

    const int dbg = (getenv("PICOLM_DS2_DBG") != NULL);
    int stop_layer = -1;
    { const char *e = getenv("PICOLM_STOP_LAYER"); if (e) stop_layer = atoi(e); }
    if (dbg) {
        float ss = 0.f; for (int i = 0; i < D; i++) ss += s->x[i] * s->x[i];
        fprintf(stderr, "[ds2] pos=%d embd rms=%.6f x[0..3]=%.6f %.6f %.6f %.6f\n",
                pos, sqrtf(ss / D), s->x[0], s->x[1], s->x[2], s->x[3]);
        fprintf(stderr, "[ds2] norm0 w[0..3]=%.6f %.6f %.6f %.6f  postnorm[0..3]=%.6f %.6f %.6f %.6f\n",
                s->attn_norm_w[0][0], s->attn_norm_w[0][1], s->attn_norm_w[0][2], s->attn_norm_w[0][3],
                s->post_attn_norm_w[0][0], s->post_attn_norm_w[0][1], s->post_attn_norm_w[0][2], s->post_attn_norm_w[0][3]);
        {
            const float *an = (const float *)m->weights.layers[0].attn_norm;
            const float *emb = (const float *)m->weights.token_embd;
            fprintf(stderr, "[ds2] l0 attn_norm ptr=%p type=%d raw[0..3]=%.6f %.6f %.6f %.6f\n",
                    (const void*)an, (int)m->weights.layers[0].type_attn_norm,
                    (double)an[0], (double)an[1], (double)an[2], (double)an[3]);
            {
                const uint8_t *q = (const uint8_t *)m->weights.output_norm;
                int si = -1;
                for (int k = 0; k < m->n_splits; k++) {
                    const uint8_t *a = (const uint8_t *)m->splits[k].mmap_addr;
                    if (q >= a && q < a + m->splits[k].mmap_size) { si = k; break; }
                }
                fprintf(stderr, "[ds2] output_norm split=%d off=%zu base=%zu\n", si,
                        si >= 0 ? (size_t)(q - (const uint8_t *)m->splits[si].mmap_addr) : 0,
                        si >= 0 ? m->tensor_data_base[si] : 0);
                const uint8_t *e = (const uint8_t *)m->weights.token_embd;
                for (int k = 0; k < m->n_splits; k++) {
                    const uint8_t *a = (const uint8_t *)m->splits[k].mmap_addr;
                    if (e >= a && e < a + m->splits[k].mmap_size) {
                        fprintf(stderr, "[ds2] token_embd split=%d off=%zu base=%zu\n", k,
                                (size_t)(e - a), m->tensor_data_base[k]);
                        break;
                    }
                }
            }
            fprintf(stderr, "[ds2] output_norm type=%d rms=", (int)m->weights.type_output_norm);
            {
                double ss=0; const float *o=(const float*)m->weights.output_norm;
                for (int i=0;i<c->n_embd;i++) ss+=(double)o[i]*o[i];
                fprintf(stderr, "%.6f\n", sqrt(ss/c->n_embd));
            }
            {
                const float *pn = (const float *)m->weights.layers[0].post_attn_norm;
                const float *on = (const float *)m->weights.output_norm;
                fprintf(stderr, "[ds2] pn=%p on=%p an=%p  pn[0..3]=%.6f %.6f %.6f %.6f on[0..3]=%.6f %.6f %.6f %.6f\n",
                        (const void*)pn, (const void*)on, (const void*)an,
                        (double)pn[0], (double)pn[1], (double)pn[2], (double)pn[3],
                        (double)on[0], (double)on[1], (double)on[2], (double)on[3]);
            }
            fprintf(stderr, "[ds2] embd ptr=%p type=%d bytes[0..15]=%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
                    (const void*)emb, (int)m->weights.type_token_embd,
                    ((const unsigned char*)emb)[0], ((const unsigned char*)emb)[1],
                    ((const unsigned char*)emb)[2], ((const unsigned char*)emb)[3],
                    ((const unsigned char*)emb)[4], ((const unsigned char*)emb)[5],
                    ((const unsigned char*)emb)[6], ((const unsigned char*)emb)[7],
                    ((const unsigned char*)emb)[8], ((const unsigned char*)emb)[9],
                    ((const unsigned char*)emb)[10], ((const unsigned char*)emb)[11],
                    ((const unsigned char*)emb)[12], ((const unsigned char*)emb)[13],
                    ((const unsigned char*)emb)[14], ((const unsigned char*)emb)[15]);
        }
        fprintf(stderr, "[ds2] outnorm[0..3]=%.6f %.6f %.6f %.6f\n",
                s->output_norm_w[0], s->output_norm_w[1], s->output_norm_w[2], s->output_norm_w[3]);
    }

    for (int il = 0; il < c->n_layers; il++) {
        const layer_weights_t *lw = &w->layers[il];
        const glm5_layer_weights_t *gw = &w->glm5[il];
        BENCH_LAYER_START();

        /* ---- attention block ---- */
        const int prof = prof_on;
        double _ta = prof ? get_time_ms() : 0;
        rmsnorm(cur, s->x, s->attn_norm_w[il], D, c->rms_norm_eps);
        if (getenv("PICOLM_DS2_NOATTN")) memset(attn_out, 0, (size_t)D * sizeof(float));
        else ds2_mla_forward(m, il, s, cur, pos, attn_out);
        double _tat = prof ? get_time_ms() : 0;
        if (getenv("PICOLM_DS2_XCHECK"))
            ds2_xcheck_attn(m, il, s, cur, pos, attn_out);
        ds2_layer_dump(m, il, pos, cur, attn_out, s);
        vec_add(s->x, attn_out, D);

        /* ---- FFN block ---- */
        double _tf = prof ? get_time_ms() : 0;
        rmsnorm(cur, s->x, s->post_attn_norm_w[il], D, c->rms_norm_eps);
        if (getenv("PICOLM_DS2_NOFFN")) {
            memset(attn_out, 0, (size_t)D * sizeof(float));
        } else if (il < c->n_layer_dense_lead) {
            ds2_dense_ffn(m, lw, s, cur, attn_out);
        } else {
            glm5_moe_ffn(m, gw, lw, s, cur, attn_out, il);
        }
        vec_add(s->x, attn_out, D);

        if (prof) {
            static double s_attn = 0, s_ffn = 0;
            double now = get_time_ms();
            s_attn += _tat - _ta; s_ffn += now - _tf;
            if (il == c->n_layers - 1) {
                fprintf(stderr, "[ds2 prof] token attn=%.0fms ffn=%.0fms\n", s_attn, s_ffn);
                s_attn = s_ffn = 0;
            }
        }

        if (dbg) {
            float sa = 0.f, sf = 0.f, sx = 0.f;
            for (int i = 0; i < D; i++) {
                sa += attn_out[i] * attn_out[i];
                sf += cur[i] * cur[i];
                sx += s->x[i] * s->x[i];
            }
            fprintf(stderr, "[ds2] l%02d %s attn=%.5f ffn_cur=%.5f x=%.5f\n",
                    il, il < c->n_layer_dense_lead ? "dense" : "moe",
                    sqrtf(sa / D), sqrtf(sf / D), sqrtf(sx / D));
        }

#ifdef PICOLM_VIZ
        viz_push_row(il, pos, s->x, D);
#endif

        BENCH_LAYER_END(il, 0);
        if (stop_layer >= 0 && il >= stop_layer) break;
    }

    /* Final: RMSNorm(output_norm) then the output head. */
    if (getenv("PICOLM_DS2_DBG")) {
        double xr = 0;
        for (int i = 0; i < D; i++) xr += (double)s->x[i]*s->x[i];
        fprintf(stderr, "[ds2 head] residual rms=%.6f\n", sqrt(xr/D));
    }
    rmsnorm(cur, s->x, s->output_norm_w, D, c->rms_norm_eps);
    if (getenv("PICOLM_DS2_DBG")) {
        double hr = 0, lr = 0;
        for (int i = 0; i < D; i++) { hr += (double)cur[i]*cur[i]; }
        fprintf(stderr, "[ds2 head] normed rms=%.6f  outnorm_rms=0.3849 expected\n", sqrt(hr/D));
        (void)lr;
    }
    if (getenv("PICOLM_DS2_HSCALE")) {
        float k = (float)atof(getenv("PICOLM_DS2_HSCALE"));
        for (int i = 0; i < D; i++) cur[i] *= k;
    }
    matmul(s->logits, cur, w->output, D, c->vocab_size, w->type_output);

    /* PICOLM_DS2_DUMP_POS: dump raw logits as text, in the same format as
     * PICOLM_DUMP_LOGITS, so it can be diffed against a llama-perplexity
     * reference. Accepts either one position ("62") or a comma-separated
     * list ("32,40,62"), or an open-ended range ("32-" = every position
     * from 32 on). A full-model load is far too expensive to pay again
     * per position. */
    {
        const char *dp = getenv("PICOLM_DS2_DUMP_POS");
        if (dp && *dp) {
            /* "62" one position | "32,40,62" a list | "32-" every pos >= 32 */
            int dump = 0;
            size_t dplen = strlen(dp);
            if (dplen > 0 && dp[dplen - 1] == '-') {
                dump = (pos >= atoi(dp));
            } else {
                const char *p = dp;
                while (*p) {
                    if (*p == ',') { p++; continue; }
                    { char *endp = NULL; long v = strtol(p, &endp, 10);
                      if (endp == p) break;
                      if ((long)pos == v) { dump = 1; break; }
                      p = endp; }
                }
            }
            if (dump) {
                static int first_dump = 1;
                const int V = c->vocab_size;
                FILE *out = stderr;
                const char *path = getenv("PICOLM_DS2_DUMP_FILE");
                if (path) {
                    /* Truncate on the first dump of this process, append after
                     * (one run dumps many positions into one file). */
                    FILE *f = fopen(path, first_dump ? "w" : "a");
                    if (f) out = f;
                }
                first_dump = 0;
                fprintf(out, "pos=%d vocab=%d\n", pos, V);
                for (int i = 0; i < V; i++) fprintf(out, "%d %.6f\n", i, s->logits[i]);
                if (out != stderr) fclose(out);
            }
        }
    }


    return s->logits;
}
