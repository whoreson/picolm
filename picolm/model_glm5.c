/* model_glm5.c -- GLM-5.3-Flash (glm5next) forward pass.
 *
 * Architecture: trunk layers + optional NextN (MTP) layer.
 *   - mHC hyper-connections: S residual streams, Sinkhorn-mixed
 *   - KDA linear attention (channel-gated delta rule) on il%4 != 3 layers
 *   - Absorbed NoPE-MLA (latent KV) on il%4 == 3 layers
 *   - Dense FFN on first n_layer_dense_lead layers, MoE on the rest
 *
 * Reference: ik_old/src/graphs/build_glm5next.cpp,
 *            ik_old/ggml/src/ggml.c hc_pre_f32 / hc_post_f32 / kda_f32.
 *
 * CPU-only forward. GPU dispatch is NOT wired for glm5.
 */
#include "model.h"
#include "model_internal.h"
#include "tensor.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define GLM5_KD_MAX 128   /* max KDA head dim for stack buffers */

/* Per-layer debugging: PICOLM_DBG_GLM5=1 prints one line per block with the
 * residual-stream RMS and the first few values of cur/post/comb, which is what
 * localises numerical drift against the ik oracle. */
static int glm5_dbg_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("PICOLM_DBG_GLM5") ? 1 : 0;
    return cached;
}

/* Verbose per-kernel dumps (KDA projections, decay/beta stats). Separate from
 * PICOLM_DBG_GLM5 so the per-layer summary stays readable. */
static int glm5_dbg2_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("PICOLM_DBG_GLM5_KDA") ? 1 : 0;
    return cached;
}

/* PICOLM_GLM5_NOATTN=1: zero out the attention/KDA block output. Diagnostic
 * bisection aid (isolates hc + FFN numerics); not a supported mode. */
static int glm5_noattn_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("PICOLM_GLM5_NOATTN") ? 1 : 0;
    return cached;
}

/* SwiGLU clamp value actually applied.
 *
 * The GGUF carries swiglu_clamp_exp/shexp = 10.0 for every layer, but the ik
 * reference implementation for glm5next never reads those keys, so the oracle
 * runs UNCLAMPED. Match the oracle by default and expose the clamp behind an
 * env var so both behaviours can be A/B'd (see glm53flash.md Q1):
 *   PICOLM_GLM5_CLAMP=1     -> apply the GGUF clamp value
 *   PICOLM_GLM5_NOCLAMP=1   -> force it off (same as default)
 */
static float glm5_clamp_value(const model_config_t *c) {
    static int mode = -1;   /* -1 unknown, 0 off, 1 on */
    if (mode < 0) {
        mode = getenv("PICOLM_GLM5_CLAMP") ? 1 : 0;
    }
    if (!mode) return 0.0f;   /* 0 == disabled */
    return c->glm5_clamp_exp;
}

static float glm5_rms(const float *v, int n) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += v[i] * v[i];
    return sqrtf(ss / (float)n);
}

/* ==================================================================== */
/* Numerical instrumentation (PICOLM_GLM5_STATS / PICOLM_GLM5_STRICT).  */
/*                                                                      */
/* Goal: localise divergence from the ik oracle. Every tag prints one   */
/* stable line so that PicoLM's output can be `diff`ed against ik's     */
/* named-tensor dumps (same format). Range checks encode the magnitudes */
/* that are provable from the algorithm, so a violation is a definite    */
/* bug rather than a judgement call.                                     */
/*                                                                      */
/*   PICOLM_GLM5_STATS=<substr>  print stats for tags containing <substr> */
/*                               ("all" = everything, per-layer points)  */
/*   PICOLM_GLM5_STRICT=1        abort() on the first range violation    */
/*   PICOLM_STOP_LAYER=<n>       stop the trunk loop after layer n       */
/* ==================================================================== */

static int glm5_stats_mode(const char **substr) {
    static const char *pat = NULL;
    static int init = 0;
    if (!init) { pat = getenv("PICOLM_GLM5_STATS"); init = 1; }
    *substr = pat;
    return pat != NULL;
}

static int glm5_strict_mode(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("PICOLM_GLM5_STRICT") ? 1 : 0;
    return cached;
}

static int glm5_stop_layer(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("PICOLM_STOP_LAYER");
        cached = e ? atoi(e) : -1;
    }
    return cached;
}

/* Stable checksum: order-sensitive, sensitive to sign and magnitude, prints as
 * %.9e so two implementations can be compared textually. */
static double glm5_crc(const float *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i] * (double)(i + 1);
    return s;
}

static int glm5_tag_selected(const char *tag, const char *pat) {
    if (!pat) return 0;
    if (strcmp(pat, "all") == 0) return 1;
    return strstr(tag, pat) != NULL;
}

/* Print rms/min/max/absmax/crc + first 4 values for a vector. */
static void glm5_stats(const char *tag, const float *v, int n) {
    const char *pat;
    if (n <= 0 || !v) return;
    if (!glm5_stats_mode(&pat) || !glm5_tag_selected(tag, pat)) return;
    float mn = v[0], mx = v[0], amax = fabsf(v[0]);
    int n_nan = 0, n_inf = 0;
    for (int i = 0; i < n; i++) {
        float x = v[i];
        if (x != x) { n_nan++; continue; }
        if (isinf(x)) { n_inf++; continue; }
        if (x < mn) mn = x;
        if (x > mx) mx = x;
        if (fabsf(x) > amax) amax = fabsf(x);
    }
    char first[96];
    int off = 0;
    for (int i = 0; i < 4 && i < n && off < (int)sizeof(first) - 16; i++) {
        int w = snprintf(first + off, sizeof(first) - (size_t)off, " %.6f", v[i]);
        if (w <= 0) break;
        off += w;
    }
    if (off == 0) { first[0] = '\0'; }
    fprintf(stderr, "GLM5 STAT %s n=%d rms=%.6f min=%.6f max=%.6f absmax=%.6f "
                    "crc=%.9e nan=%d inf=%d v:%s\n",
            tag, n, glm5_rms(v, n), mn, mx, amax, glm5_crc(v, n), n_nan, n_inf, first);
}

/* Range assertion. `lo`/`hi` are inclusive. Reports every violation of a tag
 * only once so that a per-token loop does not flood the log. */
static int glm5_chk(const char *tag, const float *v, int n, float lo, float hi) {
    static char seen[64][96];
    static int n_seen = 0;
    int bad = -1;
    float badv = 0.f;
    if (n <= 0 || !v) return 1;
    for (int i = 0; i < n; i++) {
        float x = v[i];
        if (x != x || isinf(x) || x < lo || x > hi) { bad = i; badv = x; break; }
    }
    if (bad < 0) return 1;
    int known = 0;
    for (int i = 0; i < n_seen; i++)
        if (strcmp(seen[i], tag) == 0) { known = 1; break; }
    if (!known) {
        if (n_seen < (int)(sizeof(seen) / sizeof(seen[0])))
            snprintf(seen[n_seen++], sizeof(seen[0]), "%s", tag);
        fprintf(stderr, "GLM5 RANGE VIOLATION %s: v[%d]=%.6g outside [%.6g, %.6g]\n",
                tag, bad, badv, lo, hi);
        if (glm5_strict_mode()) {
            fflush(stderr);
            abort();
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* mHC hyper-connections                                              */
/* ------------------------------------------------------------------ */

/* hc_pre: mix S residual streams into one input for the current block.
 *
 * Graph: flat[S*D] -> rms_norm(eps=f_norm_rms_eps) -> matmul(hc_fn) -> hc_pre
 * hc_pre kernel (ggml.c:23042): input m[(2+S)*S], scale[3], base[(2+S)*S].
 *   pre[i]  = sigmoid(m[i]     * scale[0] + base[i])       + eps   (i < S)
 *   post[i] = 2*sigmoid(m[S+i] * scale[1] + base[S+i])             (i < S)
 *   comb[i] = m[2S+i] * scale[2] + base[2S+i]  -> [S][S] row-major
 *   Sinkhorn: row-softmax(+eps/elt), col-norm(sum+eps), then
 *             (iters-1) x [row-L1(sum+eps), col-L1(sum+eps)]
 *
 * Outputs: cur[D], post[S], comb[S*S].
 *
 * scratch layout (caller-provided, >= S*D + M floats, M=(2+S)*S):
 *   scratch[0 .. S*D-1] = flat normed input
 *   scratch[S*D .. S*D+M-1] = m vector (matmul output, then pre/comb)
 */
/* Tag of the hc_pre call in flight, so the intermediates can be reported with
 * ik-compatible names ("hc_pre-<il>", "hc_pre_mixes-<il>"). Set by the caller. */
static char g_hc_tag[48];

static void glm5_hc_pre(const model_t *m, const glm5_layer_weights_t *lw,
                        int is_attn, const float *x /* [S][D] */,
                        float *cur, float *post, float *comb,
                        float *scratch)
{
    const model_config_t *c = &m->config;
    const int S = c->glm5_hc;
    const int D = c->n_embd;
    const int M = (2 + S) * S;

    /* hc_*_fn is a quantized weight (IQ6_K in the reference GGUF), so it must go
     * through matmul(). base/scale are plain F32 vectors in the GGUF. */
    const void *fn      = is_attn ? lw->hc_attn_fn  : lw->hc_ffn_fn;
    const float *base   = (const float *)(is_attn ? lw->hc_attn_base : lw->hc_ffn_base);
    const float *scale  = (const float *)(is_attn ? lw->hc_attn_scale : lw->hc_ffn_scale);

    float *flat = scratch;          /* [S*D] */
    float *mm   = scratch + S*D;    /* [M] */
    gguf_type_t fn_type = is_attn ? lw->type_hc_attn_fn : lw->type_hc_ffn_fn;

    /* Fail loudly rather than silently zeroing: a missing hc_* tensor means the
     * layer was not mapped (e.g. the NextN block, which has no hc tensors and
     * must never reach this function). */
    if (!fn || !base || !scale) {
        fprintf(stderr, "FATAL glm5 hc_pre: %s block missing hc tensors "
                        "(fn=%p base=%p scale=%p)\n",
                is_attn ? "attn" : "ffn", fn, (const void*)base, (const void*)scale);
        abort();
    }

    /* 1. Flatten + joint RMS norm (no weight, eps = rms_norm_eps = 1e-5).
     * Matches ggml_compute_forward_rms_norm_f32, which accumulates in
     * ggml_float (double) and computes mean = sum/ne00. The double accumulator
     * is not strictly required but keeps this bit-stable across compiler flags.
     * NOTE: c->rms_norm_eps must be attention.layer_norm_rms_epsilon (1e-5), NOT
     * attention.layer_norm_epsilon (1e-6, which is the DSA indexer's LayerNorm
     * eps). Mixing them up scales this by a constant 1.004845 and silently
     * degrades the entire model. */
    memcpy(flat, x, (size_t)S * D * sizeof(float));
    if (g_hc_tag[0]) {
        char t[96]; snprintf(t, sizeof(t), "hc_flat_in-%s", g_hc_tag);
        glm5_stats(t, flat, S*D);
    }
    {
        double ss = 0.0;
        for (int i = 0; i < S*D; i++) ss += (double)flat[i] * (double)flat[i];
        float inv = 1.0f / (float)sqrt((double)ss / (double)(S*D) + (double)c->rms_norm_eps);
        if (g_hc_tag[0] && glm5_dbg_enabled())
            fprintf(stderr, "GLM5 hc_pre dbg: S=%d D=%d S*D=%d ss=%.9e mean=%.9e eps=%.3e inv=%.6f\n",
                    S, D, S*D, ss, ss/(double)(S*D), (double)c->rms_norm_eps, (double)inv);
        for (int i = 0; i < S*D; i++) flat[i] *= inv;
    }

    if (g_hc_tag[0]) {
        char t[96]; snprintf(t, sizeof(t), "hc_pre-%s", g_hc_tag);
        glm5_stats(t, flat, S*D);
    }

    /* 2. m = fn @ flat
     * hc_attn_fn has ne = {S*D, M} (i.e. --list-tensors prints "16384,24" and
     * its matmul column says n=16384,d=24 for the real model). matmul reads
     * d rows of n elements, so n must be the row length (ne[0] = S*D) and d the
     * row count (ne[1] = M). Getting these the wrong way round produces a
     * silently mis-shaped result that poisons every layer via the mHC gate.
     * NOTE: do NOT add base here — base/scale are applied per-element below
     * (exactly as ggml_hc_pre does with bias[]/scale[]). */
    matmul(mm, flat, fn, D*S, M, fn_type);
    if (g_hc_tag[0]) {
        char t[96]; snprintf(t, sizeof(t), "hc_pre_mixes-%s", g_hc_tag);
        glm5_stats(t, mm, M);
    }

    /* 3. pre / post / comb-raw */
    float pre[8];  /* S <= 8 per ggml assert */
    for (int i = 0; i < S; i++) {
        float v = mm[i] * scale[0] + base[i];
        pre[i] = 1.0f / (1.0f + expf(-v)) + c->glm5_hc_eps;
    }
    for (int i = 0; i < S; i++) {
        float v = mm[S+i] * scale[1] + base[S+i];
        post[i] = 2.0f * (1.0f / (1.0f + expf(-v)));
    }
    for (int i = 0; i < S*S; i++)
        comb[i] = mm[2*S + i] * scale[2] + base[2*S + i];

    /* 4. Sinkhorn (ggml.c:23086-23110) */
    for (int r = 0; r < S; r++) {
        float *row = comb + r*S;
        float mx = row[0];
        for (int cc = 1; cc < S; cc++) if (row[cc] > mx) mx = row[cc];
        float sum = 0.f;
        for (int cc = 0; cc < S; cc++) { row[cc] = expf(row[cc] - mx); sum += row[cc]; }
        for (int cc = 0; cc < S; cc++) row[cc] = row[cc]/sum + c->glm5_hc_eps;
    }
    for (int cc = 0; cc < S; cc++) {
        float sum = c->glm5_hc_eps;
        for (int r = 0; r < S; r++) sum += comb[r*S + cc];
        for (int r = 0; r < S; r++) comb[r*S + cc] /= sum;
    }
    for (int it = 1; it < c->glm5_sinkhorn_iters; it++) {
        for (int r = 0; r < S; r++) {
            float sum = c->glm5_hc_eps;
            for (int cc = 0; cc < S; cc++) sum += comb[r*S + cc];
            for (int cc = 0; cc < S; cc++) comb[r*S + cc] /= sum;
        }
        for (int cc = 0; cc < S; cc++) {
            float sum = c->glm5_hc_eps;
            for (int r = 0; r < S; r++) sum += comb[r*S + cc];
            for (int r = 0; r < S; r++) comb[r*S + cc] /= sum;
        }
    }

    /* 5. cur[d] = sum_s pre[s] * x[s][d] */
    memset(cur, 0, D * sizeof(float));
    for (int s = 0; s < S; s++) {
        const float *xs = x + s * D;
        float w = pre[s];
        for (int d = 0; d < D; d++) cur[d] += w * xs[d];
    }
}

/* glm5_hc_post: scatter the block output back into the hc streams.
 *   out[i][d] = a[d]*post[i] + sum_j comb[j*S+i] * res[j][d]
 * (ggml.c:23128, comb indexed [j*S+i]: j=source stream, i=dest stream)
 */
static void glm5_hc_post(const model_t *m, const float *a, const float *res,
                         const float *post, const float *comb, float *out)
{
    const model_config_t *c = &m->config;
    const int S = c->glm5_hc;
    const int D = c->n_embd;

    for (int i = 0; i < S; i++) {
        float *oi = out + i * D;
        float pi = post[i];
        for (int d = 0; d < D; d++) oi[d] = a[d] * pi;
        for (int j = 0; j < S; j++) {
            float w = comb[j*S + i];
            const float *rj = res + j * D;
            for (int d = 0; d < D; d++) oi[d] += w * rj[d];
        }
    }
}

/* ------------------------------------------------------------------ */
/* KDA linear attention (single token, sequential decode)              */
/* ------------------------------------------------------------------ */

/* GLM-5 KDA layer forward for one token.
 *
 * Weight layout (ne0=input/row_len, ne1=output/rows):
 *   attn_q    [DI, D]   (attn_q.weight: ne0=D, ne1=DI -> matmul(D, DI))
 *   attn_k    [DI, D]
 *   attn_v    [DI, D]
 *   ssm_f_a   [KD, D]   (down-proj: D -> KD)
 *   ssm_f_b   [DI, KD]  (up-proj: KD -> DI)
 *   ssm_g_a   [KD, D]
 *   ssm_g_b   [DI, KD]
 *   ssm_beta  [H, D]    (per-head beta projection)
 *   ssm_a     [H]       (-exp(A_log), per-head decay rate)
 *   ssm_dt.bias [DI]    (per-channel dt bias)
 *   ssm_norm  [KD]      (per-head RMSNorm weight, KD=head_dim)
 *   attn_output [D, DI] (output proj: DI -> D)
 *
 * Scratch layout (s->glm5.kda_scratch):
 *   q[DI], k[DI], v[DI], z[DI], raw[DI], conv_out[3*DI], beta[H], g[DI],
 *   fa_out[KD], head_out[DI]
 */
static void glm5_kda_forward(
    const model_t *m, const glm5_layer_weights_t *gw, const layer_weights_t *lw,
    run_state_t *s, const float *cur, int pos, int layer_idx, float *out)
{
    const model_config_t *c = &m->config;
    const int D = c->n_embd;
    const int H = c->glm5_n_kda_heads;
    const int KD = c->glm5_kda_dim;
    const int DI = H * KD;
    const int CONV_K = c->ssm_d_conv;
    const int CONV_DIM = 3 * DI;

    float *sc = s->glm5.kda_scratch;
    float *q  = sc; sc += DI;
    float *k  = sc; sc += DI;
    float *v  = sc; sc += DI;
    float *z  = sc; sc += DI;
    float *raw = sc; sc += DI;
    float *conv_out = sc; sc += CONV_DIM;
    float *beta = sc; sc += H;
    float *g = sc; sc += DI;
    float *fa_out = sc; sc += KD;
    float *head_out = sc;   /* [DI] */

    /* Tag prefix shared by every stat/check emitted from this layer. */
    char tg[48];
    snprintf(tg, sizeof(tg), "l%02d.kda", layer_idx);
    char t[96];

    float *conv_state = s->glm5.kda_conv_state[layer_idx];
    float *kda_state = s->glm5.kda_state[layer_idx];

    glm5_stats("x.kda.in", cur, D);

    /* q, k, v projections */
    matmul(q, cur, lw->attn_q, D, DI, lw->type_attn_q);
    matmul(k, cur, lw->attn_k, D, DI, lw->type_attn_k);
    matmul(v, cur, lw->attn_v, D, DI, lw->type_attn_v);
    snprintf(t, sizeof(t), "%s.q", tg);   glm5_stats(t, q, DI);   glm5_chk(t, q, DI, -1e4f, 1e4f);
    snprintf(t, sizeof(t), "%s.k", tg);   glm5_stats(t, k, DI);   glm5_chk(t, k, DI, -1e4f, 1e4f);
    snprintf(t, sizeof(t), "%s.v", tg);   glm5_stats(t, v, DI);   glm5_chk(t, v, DI, -1e4f, 1e4f);
    snprintf(t, sizeof(t), "%s.dt_bias", tg);
    glm5_stats(t, (const float *)gw->kda_dt_bias, DI);
    snprintf(t, sizeof(t), "%s.ssm_a", tg);
    glm5_stats(t, (const float *)gw->kda_a, H);

    /* z = g_b @ (g_a @ cur): gate, [KD] -> [DI] */
    matmul(fa_out, cur, gw->kda_g_a, D, KD, gw->type_kda_g_a);
    matmul(z, fa_out, gw->kda_g_b, KD, DI, gw->type_kda_g_b);

    /* raw = f_b @ (f_a @ cur): decay raw input, [KD] -> [DI] */
    matmul(fa_out, cur, gw->kda_f_a, D, KD, gw->type_kda_f_a);
    matmul(raw, fa_out, gw->kda_f_b, KD, DI, gw->type_kda_f_b);
    snprintf(t, sizeof(t), "%s.z", tg);   glm5_stats(t, z, DI);   glm5_chk(t, z, DI, -1e4f, 1e4f);
    snprintf(t, sizeof(t), "%s.raw", tg); glm5_stats(t, raw, DI); glm5_chk(t, raw, DI, -1e4f, 1e4f);

    /* beta_raw = ssm_beta @ cur  [H] */
    {
        float *beta_raw = fa_out;  /* reuse: [H] output */
        matmul(beta_raw, cur, gw->kda_beta, D, H, gw->type_kda_beta);
        for (int h = 0; h < H; h++)
            beta[h] = 1.0f / (1.0f + expf(-beta_raw[h]));
    }
    /* beta = sigmoid(.) is provably in (0,1). */
    snprintf(t, sizeof(t), "%s.beta", tg);
    glm5_stats(t, beta, H);  glm5_chk(t, beta, H, 0.0f, 1.0f);

    /* g[h*KD+ch] = lower_bound * sigmoid(-a[h] * (raw[idx] + dt_bias[idx]))
     * ssm_a holds -exp(A_log) (already negated at load time).
     * raw and dt_bias are [H*KD] head-major: idx = h*KD + ch. */
    {
        const float *ssm_a_w = (const float *)gw->kda_a;
        const float *dt_bias = (const float *)gw->kda_dt_bias;
        for (int h = 0; h < H; h++) {
            float a_val = ssm_a_w[h];
            for (int ch = 0; ch < KD; ch++) {
                int idx = h*KD + ch;
                g[idx] = c->glm5_kda_lower_bound *
                         (1.0f / (1.0f + expf(a_val * (raw[idx] + dt_bias[idx]))));
            }
        }
    }
    /* g = lower_bound * sigmoid(.) is provably in (lower_bound, 0) since
     * lower_bound < 0 and sigmoid in (0,1). */
    snprintf(t, sizeof(t), "%s.g", tg);
    glm5_stats(t, g, DI);
    glm5_chk(t, g, DI, c->glm5_kda_lower_bound, 0.0f);

    /* Causal depthwise conv1d over concatenated [q;k;v] then SiLU.
     * Each conv tensor is ne = {CONV_K, 1, DI}: ggml stores it tap-fastest, so
     * the weight for (tap t, channel ch) is at  w[t + ch*CONV_K]  (stride 1 over
     * taps, CONV_K over channels). Verified against ggml_compute_forward_ssm_conv_f32,
     * which reads c[i0 + i1*nc] with nb[1] == nc.
     * conv_state: [(CONV_K-1) * CONV_DIM], state[t*CONV_DIM + ch] = input
     * from step (pos - CONV_K + 1 + t), oldest first. */
    {
        const float *wq = (const float *)gw->kda_conv_q;
        const float *wk = (const float *)gw->kda_conv_k;
        const float *wv = (const float *)gw->kda_conv_v;
        if (glm5_dbg2_enabled())
            fprintf(stderr, "GLM5 KDA l%02d conv weights: q=%p rms=%.6f | k=%p rms=%.6f | v=%p rms=%.6f"
                            "  (wq[0..3]=%.4f %.4f %.4f %.4f / wv[0..3]=%.4f %.4f %.4f %.4f)\n",
                    layer_idx, (const void*)wq, glm5_rms(wq, CONV_K*DI),
                    (const void*)wk, glm5_rms(wk, CONV_K*DI),
                    (const void*)wv, glm5_rms(wv, CONV_K*DI),
                    wq[0], wq[1], wq[2], wq[3], wv[0], wv[1], wv[2], wv[3]);
        for (int ch = 0; ch < CONV_DIM; ch++) {
            float xs[GLM5_KD_MAX];  /* CONV_K <= 4 */
            for (int t = 0; t < CONV_K - 1; t++)
                xs[t] = conv_state[t * CONV_DIM + ch];
            float x_new;
            if (ch < DI) x_new = q[ch];
            else if (ch < 2*DI) x_new = k[ch - DI];
            else x_new = v[ch - 2*DI];
            xs[CONV_K - 1] = x_new;
            const float *w = (ch < DI) ? wq : (ch < 2*DI) ? wk : wv;
            int ch_off = (ch < DI) ? ch : (ch < 2*DI) ? ch - DI : ch - 2*DI;
            float acc = 0.f;
            for (int t = 0; t < CONV_K; t++)
                acc += xs[t] * w[t + ch_off*CONV_K];
            /* SiLU */
            conv_out[ch] = acc / (1.0f + expf(-acc));
            /* shift state: drop oldest, append current */
            for (int t = 0; t < CONV_K - 2; t++)
                conv_state[t * CONV_DIM + ch] = conv_state[(t+1) * CONV_DIM + ch];
            conv_state[(CONV_K - 2) * CONV_DIM + ch] = x_new;
        }
    }

    /* Per-head L2 norm of q and k (after conv+SiLU) */
    for (int h = 0; h < H; h++) {
        float *qh = conv_out + h*KD;
        float *kh = conv_out + DI + h*KD;
        float sq = 0.f, sk = 0.f;
        for (int ch = 0; ch < KD; ch++) { sq += qh[ch]*qh[ch]; sk += kh[ch]*kh[ch]; }
        float inv_q = 1.0f / fmaxf(sqrtf(sq), 1e-5f);
        float inv_k = 1.0f / fmaxf(sqrtf(sk), 1e-5f);
        for (int ch = 0; ch < KD; ch++) { qh[ch] *= inv_q; kh[ch] *= inv_k; }
    }

    /* conv_out is computed above; per-head L2 norms must be ~1 after the
     * normalisation loop, which is a strong correctness signal. */
    snprintf(t, sizeof(t), "%s.conv_out", tg);
    glm5_stats(t, conv_out, CONV_DIM);
    glm5_chk(t, conv_out, CONV_DIM, -1e4f, 1e4f);
    if (glm5_dbg2_enabled()) {
        /* Section RMS of conv_out: q, k, v thirds. v is NOT L2-normalised so
         * its magnitude must stay comparable to the raw projection. */
        {
            float sq = 0.f, sk = 0.f, sv = 0.f;
            for (int i = 0; i < DI; i++) {
                sq += conv_out[i]*conv_out[i];
                sk += conv_out[DI+i]*conv_out[DI+i];
                sv += conv_out[2*DI+i]*conv_out[2*DI+i];
            }
            fprintf(stderr, "GLM5 KDA l%02d sections(after conv+SiLU+L2): q=%.6f k=%.6f v=%.6f "
                            "(pre-conv rms was q=%.4f k=%.4f v=%.4f)\n",
                    layer_idx, sqrtf(sq/DI), sqrtf(sk/DI), sqrtf(sv/DI),
                    glm5_rms(q, DI), glm5_rms(k, DI), glm5_rms(v, DI));
        }
        for (int h = 0; h < H && h < 4; h++) {
            float sq = 0.f, sk = 0.f;
            for (int ch = 0; ch < KD; ch++) {
                float qv = conv_out[h*KD + ch];
                float kv = conv_out[DI + h*KD + ch];
                sq += qv*qv; sk += kv*kv;
            }
            fprintf(stderr, "GLM5 KDA l%02d h%d: |q|=%.6f |k|=%.6f (expect ~1)\n",
                    layer_idx, h, sqrtf(sq), sqrtf(sk));
        }
    }

    /* Recurrent state update + output (ggml.c:23430 kda_f32).
     * q = conv_out[0..DI), k = conv_out[DI..2DI), v = conv_out[2DI..3DI)
     * state layout: state[row + col*KD + h*KD*KD] (column-major within head).
     * out[row] = scale * sum_col(S[row,col] * q[col]*decay[col])
     *          + v_new[row] * attn_score
     * attn_score = scale * dot(k, q)
     * v_new[row] = beta * (v[row] - sum_col(S[row,col] * k[col]*decay[col]))
     * decay[col] = exp(min(g[col], 50))
     * state[row,col] = clamp(decay[col]*state[row,col] + v_new[row]*k[col])
     */
    {
        const float *qq = conv_out;
        const float *kk = conv_out + DI;
        const float *vv = conv_out + 2*DI;
        const float scale = 1.0f / sqrtf((float)KD);

        for (int h = 0; h < H; h++) {
            const float *qh = qq + h*KD;
            const float *kh = kk + h*KD;
            const float *vh = vv + h*KD;
            float *Sh = kda_state + (size_t)h * KD * KD;
            float decay[GLM5_KD_MAX], v_new[GLM5_KD_MAX];

            float attn_score = 0.f;
            for (int i = 0; i < KD; i++)
                attn_score += kh[i] * qh[i];
            attn_score *= scale;

            for (int col = 0; col < KD; col++)
                decay[col] = expf(fminf(g[h*KD + col], 50.0f));

            for (int row = 0; row < KD; row++) {
                const float *Srow = Sh + row;   /* column-major: stride KD */
                float v_prime = 0.f;
                float out_val = 0.f;
                for (int col = 0; col < KD; col++) {
                    float st = Srow[col*KD];
                    v_prime += st * (kh[col] * decay[col]);
                    out_val += st * (qh[col] * decay[col]);
                }
                float vn = beta[h] * (vh[row] - v_prime);
                v_new[row] = vn;
                head_out[h*KD + row] = out_val * scale + vn * attn_score;
            }
            /* Diagnose the three contributions to head_out separately: at pos 0
             * the state is zero so out_val == 0 and the whole head output is
             * v_new * attn_score. If head_out is tiny, either v_new or
             * attn_score collapsed. */
            if (glm5_dbg2_enabled() && h < 4) {
                float rsq = 0.f, rvn = 0.f, ro = 0.f, rdec = 0.f;
                for (int i = 0; i < KD; i++) {
                    rsq += qh[i]*qh[i];
                    rvn += v_new[i]*v_new[i];
                    ro  += head_out[h*KD + i]*head_out[h*KD + i];
                    rdec += decay[i];
                }
                fprintf(stderr, "GLM5 KDA l%02d h%d pos=%d: attn_score=%.6f "
                                "|q_conv|=%.4f |v_new|=%.6f |head_out|=%.6f "
                                "decay_mean=%.4f beta=%.4f\n",
                        layer_idx, h, pos, attn_score, sqrtf(rsq), sqrtf(rvn),
                        sqrtf(ro), rdec / KD, beta[h]);
            }
            /* State update (separate loop, needs v_new complete) */
            for (int col = 0; col < KD; col++) {
                float k_col = kh[col];
                float dec = decay[col];
                for (int row = 0; row < KD; row++) {
                    float st = Sh[row + col*KD];
                    st = dec * st + v_new[row] * k_col;
                    Sh[row + col*KD] = fminf(fmaxf(st, -1e6f), 1e6f);
                }
            }
        }
    }

    /* Per-head RMSNorm with ssm_norm weight, then sigmoid gate (z) */
    {
        snprintf(t, sizeof(t), "%s.head_out", tg);
        glm5_stats(t, head_out, DI);
        const float *norm_w = (const float *)gw->kda_norm;
        for (int h = 0; h < H; h++) {
            float *oh = head_out + h*KD;
            float ss = 0.f;
            for (int ch = 0; ch < KD; ch++) ss += oh[ch]*oh[ch];
            float inv = 1.0f / sqrtf(ss / KD + c->rms_norm_eps);
            for (int ch = 0; ch < KD; ch++) oh[ch] *= inv * norm_w[ch];
        }
        snprintf(t, sizeof(t), "%s.normed", tg);
        glm5_stats(t, head_out, DI);
        for (int i = 0; i < DI; i++)
            head_out[i] *= 1.0f / (1.0f + expf(-z[i]));
        snprintf(t, sizeof(t), "%s.gated", tg);
        glm5_stats(t, head_out, DI);
    }

    /* Output projection: attn_output [D, DI] @ o[DI] -> out[D] */
    matmul(out, head_out, lw->attn_output, DI, D, lw->type_attn_output);
    snprintf(t, sizeof(t), "%s.out", tg);
    glm5_stats(t, out, D);
    glm5_chk(t, out, D, -1e4f, 1e4f);
}

/* ------------------------------------------------------------------ */
/* Absorbed NoPE-MLA (single token, dense attention over latent cache) */
/* ------------------------------------------------------------------ */

/* GLM-5 MLA layer forward for one token.
 *
 * Weights:
 *   attn_q_a      [Q_LORA, D]        (D -> q_lora)
 *   attn_q_a_norm [Q_LORA]           (RMSNorm weight)
 *   attn_q_b      [n_heads*MLA_HD, Q_LORA]   (q_lora -> n_heads * mla_head_dim)
 *   attn_kv_a_mqa [KV_LORA, D]       (D -> kv_lora + n_rot; n_rot=0 here)
 *   attn_kv_a_norm [KV_LORA]         (RMSNorm weight on the latent)
 *   attn_k_b      [MLA_HD, KV_LORA, n_heads]  3-D: per-head Wk_b (kv_lora -> mla_hd)
 *   attn_v_b      [KV_LORA, MLA_HD, n_heads]  3-D: per-head Wv_b (mla_hd -> kv_lora)
 *   attn_output   [D, n_heads*MLA_HD]
 *
 * Cache: latent [512] per token per MLA layer (F16), K==V==latent.
 *
 * Absorbed form (no RoPE):
 *   c = RMSNorm(cur, attn_norm)
 *   qr = RMSNorm(Wq_a c, q_a_norm)          [q_lora]
 *   q  = Wq_b qr                            [n_heads * mla_hd]
 *   Qabs[h] = Wk_b[h] (mla_hd x kv_lora) @ q[h]  -> [kv_lora] per head
 *   kv = RMSNorm(Wkv_a c, kv_a_norm)        [kv_lora] -> cache (K=V)
 *   per head: score[t] = dot(Qabs[h], kv[t]) / sqrt(mla_hd)
 *             softmax -> ctx[h] = sum p[t] * kv[t]     [kv_lora]
 *             o[h] = Wv_b[h] (mla_hd x kv_lora) @ ctx[h]  -> [mla_hd]
 *   out = attn_output @ concat(o[h])         [D]
 */
static void glm5_mla_forward(
    const model_t *m, const glm5_layer_weights_t *gw, const layer_weights_t *lw,
    run_state_t *s, const float *cur, int pos, int n_ctx, int layer_idx,
    int mla_ord, float *out)
{
    const model_config_t *c = &m->config;
    const int D = c->n_embd;
    const int NH = c->n_heads;
    const int QL = c->glm5_q_lora;
    const int KVL = c->glm5_kv_lora;
    const int MHD = c->glm5_mla_head_dim;

    float *sc = s->glm5.mla_scratch;
    float *qr = sc; sc += QL;
    float *qabs = sc; sc += KVL * NH;   /* absorbed Q per head */
    float *kv = sc; sc += KVL;          /* current token's latent */
    float *ctx = sc; sc += KVL * NH;    /* attention output per head */
    float *scores = sc;                 /* [max_seq] */
    (void)gw;
    (void)n_ctx; (void)layer_idx;
    /* NOTE: `cur` is ALREADY RMSNorm(cur, attn_norm) — the caller applies it
     * once for both KDA and MLA. Do not norm again here. */

    /* qr = RMSNorm(Wq_a @ c, q_a_norm) */
    {
        float *tmp = s->glm5.mla_q;   /* reuse: [NH*MHD] >= [QL] */
        matmul(tmp, cur, gw->mla_q_a, D, QL, gw->type_mla_q_a);
        float ss = 0.f;
        for (int i = 0; i < QL; i++) ss += tmp[i] * tmp[i];
        float inv = 1.0f / sqrtf(ss / QL + c->rms_norm_eps);
        const float *w = (const float *)gw->mla_q_a_norm;   /* F32 in GGUF */
        for (int i = 0; i < QL; i++) qr[i] = tmp[i] * inv * w[i];
    }

    /* q = Wq_b @ qr   [NH*MHD] */
    {
        float *q = s->glm5.mla_q;   /* [NH * MHD] */
        matmul(q, qr, gw->mla_q_b, QL, NH*MHD, gw->type_mla_q_b);

        /* Qabs[h] = Wk_b[h] @ q[h*MHD .. (h+1)*MHD)   [KVL] per head
         * attn_k_b is 3-D [MLA_HD, KVL, NH] in ggml order:
         *   ne[0]=MLA_HD (rows of the per-head matmul), ne[1]=KVL, ne[2]=NH.
         * In memory (row-major ne0 fastest): head h's matrix is at offset
         *   h * MLA_HD * KVL, with row r at offset h*MHD*KVL + r*KVL.
         * So Wk_b[h] is [KVL rows x MHD cols], and Qabs[h] = Wk_b[h] @ q_h
         * means: for each of KVL rows, dot with q_h[MHD].
         * Wait -- ggml ne[0] is the fastest-varying (contiguous) dim.
         * attn_k_b ne = {MHD, KVL, NH}: row r (of KVL), col d (of MHD)
         *   element = data[h*KVL*MHD + r*MHD + d].
         * So per head: matrix KVL x MHD, Wk_b[h][r][d] = data[h*KVL*MHD + r*MHD + d].
         * Qabs[h][r] = sum_d Wk_b[h][r][d] * q_h[d]  -> [KVL]. */
        const uint8_t *kb = (const uint8_t *)gw->mla_k_b;
        size_t k_row_sz = gguf_type_row_size(gw->type_mla_k_b, MHD);
        size_t k_head_sz = (size_t)KVL * k_row_sz;
        for (int h = 0; h < NH; h++) {
            const void *wh = kb + h * k_head_sz;
            matmul(qabs + h*KVL, q + h*MHD, wh, MHD, KVL, gw->type_mla_k_b);
        }
    }

    /* kv = RMSNorm(Wkv_a @ c, kv_a_norm) -> latent, store in cache */
    {
        float *tmp = s->glm5.mla_q;   /* reuse */
        matmul(tmp, cur, gw->mla_kv_a_mqa, D, KVL, gw->type_mla_kv_a_mqa);
        float ss = 0.f;
        for (int i = 0; i < KVL; i++) ss += tmp[i] * tmp[i];
        float inv = 1.0f / sqrtf(ss / KVL + c->rms_norm_eps);
        const float *w = (const float *)gw->mla_kv_a_norm;   /* F32 in GGUF */
        for (int i = 0; i < KVL; i++) kv[i] = tmp[i] * inv * w[i];

        /* Store in the latent cache (K and V alias the same buffer) */
        float *cache = s->glm5.mla_kv + (size_t)mla_ord * c->max_seq_len * KVL;
        memcpy(cache + (size_t)pos * KVL, kv, KVL * sizeof(float));
    }

    /* Attention: per head, scores over all cached latents */
    {
        const float scale = 1.0f / sqrtf((float)MHD);
        float *cache = s->glm5.mla_kv + (size_t)mla_ord * c->max_seq_len * KVL;
        int n_kv = pos + 1;

        for (int h = 0; h < NH; h++) {
            const float *qh = qabs + h * KVL;
            /* scores */
            for (int t = 0; t < n_kv; t++) {
                const float *kt = cache + (size_t)t * KVL;
                float dot = 0.f;
                for (int i = 0; i < KVL; i++) dot += qh[i] * kt[i];
                scores[t] = dot * scale;
            }
            /* softmax */
            {
                float mx = scores[0];
                for (int t = 1; t < n_kv; t++) if (scores[t] > mx) mx = scores[t];
                float sum = 0.f;
                for (int t = 0; t < n_kv; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
                float inv = 1.0f / sum;
                for (int t = 0; t < n_kv; t++) scores[t] *= inv;
            }
            /* ctx[h] = sum_t p[t] * cache[t] */
            float *ch = ctx + h * KVL;
            memset(ch, 0, KVL * sizeof(float));
            for (int t = 0; t < n_kv; t++) {
                float p = scores[t];
                if (p < 1e-10f) continue;
                const float *kt = cache + (size_t)t * KVL;
                for (int i = 0; i < KVL; i++) ch[i] += p * kt[i];
            }
        }
    }

    /* o[h] = Wv_b[h] @ ctx[h]  -> [MHD] per head; concat -> [NH*MHD] */
    {
        float *o_concat = s->glm5.mla_q;   /* [NH*MHD] reuse */
        const uint8_t *vb = (const uint8_t *)gw->mla_v_b;
        size_t v_row_sz = gguf_type_row_size(gw->type_mla_v_b, KVL);
        size_t v_head_sz = (size_t)MHD * v_row_sz;
        /* attn_v_b ne = {KVL, MHD, NH}: per head, matrix MHD x KVL.
         * Wv_b[h][r][d] = data[h*MHD*KVL + r*KVL + d], r in [0,MHD), d in [0,KVL).
         * o[h][r] = sum_d Wv_b[h][r][d] * ctx[h][d]. */
        for (int h = 0; h < NH; h++) {
            const void *wh = vb + h * v_head_sz;
            matmul(o_concat + h*MHD, ctx + h*KVL, wh, KVL, MHD, gw->type_mla_v_b);
        }
        matmul(out, o_concat, lw->attn_output, NH*MHD, D, lw->type_attn_output);
    }
}

/* ------------------------------------------------------------------ */
/* Dense FFN and MoE FFN                                               */
/* ------------------------------------------------------------------ */

static void glm5_dense_ffn(const model_t *m, const layer_weights_t *lw,
                           run_state_t *s, const float *cur, float *out)
{
    const model_config_t *c = &m->config;
    const int D = c->n_embd;
    const int FF = c->n_ffn;
    const float clamp_v = glm5_clamp_value(c);
    matmul(s->hb, cur, lw->ffn_gate, D, FF, lw->type_ffn_gate);
    matmul(s->hb2, cur, lw->ffn_up, D, FF, lw->type_ffn_up);
    for (int i = 0; i < FF; i++) {
        float g = s->hb[i];
        float u = s->hb2[i];
        /* SwiGLU with optional clamp */
        if (clamp_v > 0.0f) {
            g = fminf(g, clamp_v);
            u = fminf(fmaxf(u, -clamp_v), clamp_v);
        }
        s->hb[i] = g / (1.0f + expf(-g)) * u;
    }
    matmul(out, s->hb, lw->ffn_down, FF, D, lw->type_ffn_down);
    { char t[96]; snprintf(t, sizeof(t), "x.ffn.dense.out"); glm5_stats(t, out, D); }
}

static void glm5_moe_ffn(const model_t *m, const glm5_layer_weights_t *gw,
                         const layer_weights_t *lw, run_state_t *s,
                         const float *cur, float *out)
{
    const model_config_t *c = &m->config;
    const int D = c->n_embd;
    const int NE = c->n_expert;
    const int AK = c->n_expert_used;
    const int FF = c->n_ff_exp;         /* expert FFN width (2048) */
    const int SFF = c->n_ff_shexp;      /* shared expert FFN width */
    const float clamp_v = glm5_clamp_value(c);

    char t[96];
    glm5_stats("x.moe.in", cur, D);

    /* Local probs[]: router probabilities before the selection bias. */
    float probs[512];
    if (NE > (int)(sizeof(probs)/sizeof(probs[0]))) {
        fprintf(stderr, "ERROR: glm5 MoE: n_expert=%d exceeds stack probs[]\n", NE);
        memset(out, 0, D * sizeof(float));
        return;
    }

    float *logits = s->expert_logits;
    matmul(logits, cur, lw->ffn_gate_inp, D, NE, lw->type_ffn_gate_inp);

    /* sigmoid router (p), then + bias for SELECTION only (GLM-5 semantics).
     * Keep the pre-bias sigmoid values around: they are the expert weights. */
    for (int i = 0; i < NE; i++) logits[i] = 1.0f / (1.0f + expf(-logits[i]));
    for (int i = 0; i < NE; i++) probs[i] = logits[i];
    /* Router probabilities are sigmoids, so provably in (0,1). */
    glm5_stats("x.moe.probs", probs, NE);
    glm5_chk("x.moe.probs", probs, NE, 0.0f, 1.0f);
    if (gw->exp_probs_b) {
        const float *bias = (const float *)gw->exp_probs_b;
        for (int i = 0; i < NE; i++) logits[i] += bias[i];
    }

    /* top-k selection over (p + bias); weights come from the raw p values. */
    int sel[16];  /* AK <= 16 */
    float wts[16];
    for (int k = 0; k < AK; k++) {
        int best = -1; float bv = -1e30f;
        for (int i = 0; i < NE; i++) {
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        }
        sel[k] = best;
        logits[best] = -1e30f;  /* remove from further selection */
    }
    {
        float sum = 0.f;
        for (int k = 0; k < AK; k++) wts[k] = probs[sel[k]];
        for (int k = 0; k < AK; k++) sum += wts[k];
        float inv = 1.0f / sum;  /* no epsilon for GLM */
        float sc2 = inv * c->glm5_moe_scale;
        for (int k = 0; k < AK; k++) wts[k] *= sc2;
    }

    /* routed experts */
    memset(out, 0, D * sizeof(float));
    for (int k = 0; k < AK; k++) {
        int e = sel[k];
        float w = wts[k];
        if (w == 0.f) continue;
        /* expert e: gate/up/down from the 3-D expert tensors.
         * ffn_gate_exps ne = {D, FF, NE}: per-expert stride = FF * row_size(type, D).
         * ffn_down_exps ne = {FF, D, NE}: per-expert stride = D * row_size(type, FF). */
        size_t gate_stride = (size_t)FF * gguf_type_row_size(lw->type_ffn_gate_exps, D);
        size_t down_stride = (size_t)D * gguf_type_row_size(lw->type_ffn_down_exps, FF);
        const void *gate_w = (const uint8_t *)lw->ffn_gate_exps + (size_t)e * gate_stride;
        const void *up_w   = (const uint8_t *)lw->ffn_up_exps   + (size_t)e * gate_stride;
        const void *down_w = (const uint8_t *)lw->ffn_down_exps + (size_t)e * down_stride;
        matmul(s->hb_exp, cur, gate_w, D, FF, lw->type_ffn_gate_exps);
        matmul(s->hb2_exp, cur, up_w, D, FF, lw->type_ffn_up_exps);
        for (int i = 0; i < FF; i++) {
            float g = s->hb_exp[i];
            float u = s->hb2_exp[i];
            if (clamp_v > 0.0f) {
                g = fminf(g, clamp_v);
                u = fminf(fmaxf(u, -clamp_v), clamp_v);
            }
            s->hb_exp[i] = g / (1.0f + expf(-g)) * u;
        }
        matmul(s->glm5.moe_down, s->hb_exp, down_w, FF, D, lw->type_ffn_down_exps);
        for (int i = 0; i < D; i++) out[i] += w * s->glm5.moe_down[i];
    }

    glm5_stats("x.moe.routed", out, D);

    /* shared expert (no sigmoid gate for GLM-5) */
    if (lw->ffn_gate_shexp) {
        matmul(s->hb_exp, cur, lw->ffn_gate_shexp, D, SFF, lw->type_ffn_gate_shexp);
        matmul(s->hb2_exp, cur, lw->ffn_up_shexp, D, SFF, lw->type_ffn_up_shexp);
        for (int i = 0; i < SFF; i++) {
            float g = s->hb_exp[i];
            float u = s->hb2_exp[i];
            if (clamp_v > 0.0f) {
                g = fminf(g, clamp_v);
                u = fminf(fmaxf(u, -clamp_v), clamp_v);
            }
            s->hb_exp[i] = g / (1.0f + expf(-g)) * u;
        }
        matmul(s->glm5.moe_down, s->hb_exp, lw->ffn_down_shexp, SFF, D, lw->type_ffn_down_shexp);
        snprintf(t, sizeof(t), "x.moe.shexp");
        glm5_stats(t, s->glm5.moe_down, D);
        for (int i = 0; i < D; i++) out[i] += s->glm5.moe_down[i];
    }
    glm5_stats("x.moe.out", out, D);
    glm5_chk("x.moe.out", out, D, -1e4f, 1e4f);
}

/* ------------------------------------------------------------------ */
/* Main forward pass                                                   */
/* ------------------------------------------------------------------ */

/* Single-token decode forward for GLM-5.3-Flash.
 * Returns logits [n_vocab] in s->logits. */
float *model_forward_glm5(model_t *m, int token, int pos)
{
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;
    const int D = c->n_embd;
    const int S = c->glm5_hc;
    const int M = (2 + S) * S;

    /* Token embedding lookup + hc_init: broadcast to all S streams */
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, D);
        const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
        dequantize_row(embd_row, s->x, D, w->type_token_embd);
        for (int i = 0; i < S; i++)
            memcpy(s->glm5.hc_x + i*D, s->x, D * sizeof(float));
        glm5_stats("x.embd", s->x, D);
        glm5_chk("x.embd", s->x, D, -1e4f, 1e4f);
    }

    /* scratch for hc_pre: flat normed [S*D] + m vector [M] */
    float *hc_scratch = s->glm5.hc_scratch;
    float *pre_out = s->glm5.hc_m;   /* [M] reused per block */
    float *post = pre_out + M;       /* need S floats -- hc_m is (2+S)*S >= S+2S+... */
    float *comb = post + S;          /* need S*S floats */

    /* Block loop: attention (KDA or MLA) then FFN, each wrapped in hc_pre/post.
     * Only the trunk is executed; the trailing NextN/MTP block (layer
     * n_layers-1 when nextn_predict_layers > 0) has no hc_* tensors and is
     * reserved for speculative decoding (M10). */
    const int n_trunk = c->n_layer_kv_from_start > 0 && c->n_layer_kv_from_start <= c->n_layers
                      ? c->n_layer_kv_from_start : c->n_layers;
    const int stop_layer = glm5_stop_layer();
    char lt[64], t[96];
    for (int il = 0; il < n_trunk; il++) {
        const layer_weights_t *lw = &w->layers[il];
        const glm5_layer_weights_t *gw = &w->glm5[il];
        int mla_ord = c->glm5_layer_mla_ord[il];  /* -1 for KDA layers */
        snprintf(lt, sizeof(lt), "l%02d", il);

        /* ---- attention block ---- */
        /* hc_res holds the current streams (input to hc_post) */
        memcpy(s->glm5.hc_res, s->glm5.hc_x, (size_t)S * D * sizeof(float));

        snprintf(t, sizeof(t), "%s.x_in", lt);
        glm5_stats(t, s->glm5.hc_x, S*D);

        snprintf(g_hc_tag, sizeof(g_hc_tag), "%d", il);
        glm5_hc_pre(m, gw, 1, s->glm5.hc_x, s->glm5.cur, post, comb, hc_scratch);
        snprintf(t, sizeof(t), "%s.hc_attn_pre", lt);
        glm5_stats(t, s->glm5.cur, D);
        snprintf(t, sizeof(t), "%s.hc_attn_post", lt);
        glm5_stats(t, post, S);
        glm5_chk(t, post, S, 0.0f, 2.0f + 1e-3f);   /* 2*sigmoid, provably (0,2) */
        snprintf(t, sizeof(t), "%s.hc_attn_comb", lt);
        glm5_stats(t, comb, S*S);

        /* RMSNorm with the predequantized attn_norm weight */
        rmsnorm(s->glm5.cur, s->glm5.cur, s->attn_norm_w[il], D, c->rms_norm_eps);
        snprintf(t, sizeof(t), "%s.cur_normed", lt);
        glm5_stats(t, s->glm5.cur, D);

        /* attention output into s->glm5.attn_out */
        float *attn_out = s->glm5.attn_out;
        /* PICOLM_GLM5_NOATTN=1 skips the attention/KDA block entirely (its output
         * is zeroed). Diagnostic bisection tool: isolates hc + FFN/MoE numerics
         * from the attention kernels. NOT a supported inference mode. */
        if (glm5_noattn_enabled()) {
            memset(attn_out, 0, (size_t)D * sizeof(float));
        } else if (mla_ord >= 0)
            glm5_mla_forward(m, gw, lw, s, s->glm5.cur, pos, c->max_seq_len, il, mla_ord, attn_out);
        else
            glm5_kda_forward(m, gw, lw, s, s->glm5.cur, pos, il, attn_out);

        snprintf(t, sizeof(t), "%s.attn_out", lt);
        glm5_stats(t, attn_out, D);
        glm5_chk(t, attn_out, D, -1e4f, 1e4f);
        if (glm5_dbg_enabled())
            fprintf(stderr, "GLM5 l%02d attn %s: a_rms=%.4f n_attn=%d\n", il,
                    mla_ord >= 0 ? "MLA" : "KDA", glm5_rms(attn_out, D), D);

        /* hc_post for attention */
        glm5_hc_post(m, attn_out, s->glm5.hc_res, post, comb, s->glm5.hc_x);
        snprintf(t, sizeof(t), "%s.x_mid", lt);
        glm5_stats(t, s->glm5.hc_x, S*D);

        /* ---- FFN block ---- */
        memcpy(s->glm5.hc_res, s->glm5.hc_x, (size_t)S * D * sizeof(float));
        glm5_hc_pre(m, gw, 0, s->glm5.hc_x, s->glm5.cur, post, comb, hc_scratch);
        snprintf(t, sizeof(t), "%s.hc_ffn_pre", lt);
        glm5_stats(t, s->glm5.cur, D);

        /* RMSNorm with the predequantized ffn_norm (post_attn_norm) weight */
        rmsnorm(s->glm5.cur, s->glm5.cur, s->post_attn_norm_w[il], D, c->rms_norm_eps);
        snprintf(t, sizeof(t), "%s.ffn_cur_normed", lt);
        glm5_stats(t, s->glm5.cur, D);

        /* FFN or MoE */
        if (il < c->n_layer_dense_lead) {
            /* dense FFN (layers 0..n_dense-1 are dense, rest are MoE) */
            glm5_dense_ffn(m, lw, s, s->glm5.cur, s->glm5.attn_out);
        } else {
            glm5_moe_ffn(m, gw, lw, s, s->glm5.cur, s->glm5.attn_out);
        }

        snprintf(t, sizeof(t), "%s.ffn_out", lt);
        glm5_stats(t, s->glm5.attn_out, D);
        glm5_chk(t, s->glm5.attn_out, D, -1e4f, 1e4f);

        glm5_hc_post(m, s->glm5.attn_out, s->glm5.hc_res, post, comb, s->glm5.hc_x);
        snprintf(t, sizeof(t), "%s.x_out", lt);
        glm5_stats(t, s->glm5.hc_x, S*D);
        if (glm5_dbg_enabled())
            fprintf(stderr, "GLM5 l%02d ffn  %s: f_rms=%.4f x_rms=%.4f\n",
                    il, il < c->n_layer_dense_lead ? "dense" : "moe",
                    glm5_rms(s->glm5.attn_out, D), glm5_rms(s->glm5.hc_x, S*D));

        if (stop_layer >= 0 && il >= stop_layer) {
            if (glm5_dbg_enabled())
                fprintf(stderr, "GLM5: stopping after layer %d (PICOLM_STOP_LAYER)\n", il);
            break;
        }
    }

    /* Final: mean over S streams -> RMSNorm(output_norm) -> output head */
    {
        float inv_s = 1.0f / (float)S;
        for (int d = 0; d < D; d++) {
            float acc = 0.f;
            for (int i = 0; i < S; i++) acc += s->glm5.hc_x[i*D + d];
            s->glm5.cur[d] = acc * inv_s;
        }
        float ss = 0.f;
        for (int i = 0; i < D; i++) ss += s->glm5.cur[i] * s->glm5.cur[i];
        float inv = 1.0f / sqrtf(ss / D + c->rms_norm_eps);
        const float *nw = s->output_norm_w;   /* predequantized output_norm */
        for (int i = 0; i < D; i++) s->glm5.cur[i] *= inv * nw[i];
        matmul(s->logits, s->glm5.cur, w->output, D, c->vocab_size, w->type_output);
    }

    /* Diagnostics: top-N logits + a few sanity stats. A healthy distribution
     * after a chat-formatted prompt should have a clear peak and a sensible
     * spread; a near-uniform or NaN distribution means the numerics are off.
     * Works on a copy so s->logits is left untouched for the sampler. */
    if (glm5_dbg_enabled()) {
        enum { TOPN = 5 };
        float lmax = s->logits[0], lmin = s->logits[0];
        double lsum = 0.0;
        int n_nan = 0;
        for (int i = 0; i < c->vocab_size; i++) {
            float v = s->logits[i];
            if (v != v) { n_nan++; continue; }
            if (v > lmax) lmax = v;
            if (v < lmin) lmin = v;
            lsum += v;
        }
        float *copy = (float *)malloc((size_t)c->vocab_size * sizeof(float));
        if (copy) {
            memcpy(copy, s->logits, (size_t)c->vocab_size * sizeof(float));
            fprintf(stderr, "GLM5 logits pos=%d: max=%.4f min=%.4f mean=%.6f nan=%d\n",
                    pos, lmax, lmin, lsum / c->vocab_size, n_nan);
            for (int k = 0; k < TOPN; k++) {
                int bi = -1; float bv = -1e30f;
                for (int i = 0; i < c->vocab_size; i++)
                    if (copy[i] > bv) { bv = copy[i]; bi = i; }
                if (bi < 0) break;
                fprintf(stderr, "  top%d: id=%d logit=%.4f\n", k, bi, bv);
                copy[bi] = -1e30f;
            }
            free(copy);
        }
    }
    return s->logits;
}

