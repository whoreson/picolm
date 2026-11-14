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
#include "quant.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#ifdef PICOLM_VIZ
#include "viz.h"
#endif

#define GLM5_KD_MAX 128   /* max KDA head dim for stack buffers */

/* Static bound for the indexer's stack/static scratch. Sized for the shipped
 * GLM-5.3 geometry (idx_nh=32, idx_dim=128, kpool=4, top_k=2048) with margin;
 * glm5_dsa_select() bails out to dense attention if a model exceeds them. */
#define GLM5_MAX_IDX_DIM  256
#define GLM5_MAX_IDX_NH   64
#define GLM5_MAX_KPOOL    16
#define GLM5_DSA_MAX_POOL 1024     /* max_seq / kpool */
#define GLM5_DSA_MAX_SEL  4096     /* <= indexer.top_k */

#ifndef INFINITY
#define INFINITY (1.0f/0.0f)
#endif


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

/* ---- Decode phase profiler: PICOLM_GLM5_PROF=1 -------------------------
 *
 * Per-token wall time of the GLM-5 decode path, split into the regions that
 * actually matter for context scaling. Prints a table on exit (atexit) plus a
 * per-token line when PICOLM_GLM5_PROF=2. Motivated by the deep-context decode
 * slowdown (see notes/picolm/glm5_decode_deep_ctx.md): the MLA attention loop
 * reads the whole latent cache once per head per token, single-threaded, so it
 * is the only term in the decode path that grows with `pos`.
 *
 * Regions:
 *   kda     - KDA linear-attention layers (O(1) in pos)
 *   mla     - MLA projections (q_a/q_b/kv_a/o + Wk_b/Wv_b per head)
 *   attn    - MLA scaled-dot-product attention loop over the latent cache
 *   idx     - DSA indexer store+select
 *   moe     - MoE / dense FFN
 *   head    - final norm + output projection
 */
static int glm5_prof_mode(void) {
    static int cached = -2;
    if (cached == -2) { const char *e = getenv("PICOLM_GLM5_PROF"); cached = e ? atoi(e) : 0; }
    return cached;
}

/* Local monotonic clock. Deliberately NOT model_core.c's get_time_ms(): the
 * GLM-5 unit tests (tests/test_glm5_dsa.c, tests/test_glm5_snapshot.c) link
 * model_glm5.c without model_core.c, so this file must not depend on symbols
 * defined there. */
static double glm5_prof_now(void) {
#if defined(_WIN32) || defined(PICOLM_DOS)
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
#endif
}

typedef enum { GP_KDA = 0, GP_MLA, GP_ATTN, GP_IDX, GP_MOE, GP_HEAD, GP_N } glm5_prof_region_t;
static const char *const glm5_prof_names[GP_N] = {
    "kda", "mla (qkv/o proj)", "mla attn (ctx loop)",
    "dsa indexer", "moe/ffn", "head",
};
static struct { double ms[GP_N]; int tokens; int printed; } g_gp;

static void glm5_prof_add(glm5_prof_region_t r, double t0) {
    if (glm5_prof_mode()) g_gp.ms[r] += glm5_prof_now() - t0;
}

static void glm5_prof_print(void) {
    if (!glm5_prof_mode() || g_gp.tokens == 0 || g_gp.printed) return;
    g_gp.printed = 1;
    double total = 0.0;
    for (int i = 0; i < GP_N; i++) total += g_gp.ms[i];
    fprintf(stderr, "\n=== GLM-5 decode phase profile (PICOLM_GLM5_PROF) ===\n");
    fprintf(stderr, "  %d decode token(s), %.0f ms total, %.0f ms/token\n",
            g_gp.tokens, total, total / g_gp.tokens);
    fprintf(stderr, "  %-22s %12s %8s\n", "region", "ms/token", "share");
    for (int i = 0; i < GP_N; i++) {
        if (g_gp.ms[i] == 0.0) continue;
        fprintf(stderr, "  %-22s %12.2f %7.1f%%\n", glm5_prof_names[i],
                g_gp.ms[i] / g_gp.tokens, 100.0 * g_gp.ms[i] / total);
    }
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

            /* The state is column-major (state[row + col*KD]). The previous
             * loop had row outermost and read Srow[col*KD], i.e. stride KD
             * floats = 512 bytes -- one cache line per load, no reuse. Walk
             * column-major instead (col outer, row inner) so each 128-float
             * column block is read contiguously and both dot products share
             * the loaded value. */
            {
                float *vp = v_new;          /* reuse v_new as the v' accumulator */
                float *ov = head_out + h*KD;
                memset(vp, 0, (size_t)KD * sizeof(float));
                memset(ov, 0, (size_t)KD * sizeof(float));
                for (int col = 0; col < KD; col++) {
                    const float *Scol = Sh + (size_t)col*KD;
                    const float kd_ = kh[col] * decay[col];
                    const float qd_ = qh[col] * decay[col];
                    for (int row = 0; row < KD; row++) {
                        float st = Scol[row];
                        vp[row] += st * kd_;
                        ov[row] += st * qd_;
                    }
                }
                for (int row = 0; row < KD; row++) {
                    float vn = beta[h] * (vh[row] - vp[row]);
                    v_new[row] = vn;
                    ov[row] = ov[row] * scale + vn * attn_score;
                }
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
                float *Sc = Sh + (size_t)col*KD;   /* contiguous column */
                int row = 0;
#if defined(PICOLM_AVX512)
                {
                    const __m512 vdec = _mm512_set1_ps(dec);
                    const __m512 vk   = _mm512_set1_ps(k_col);
                    const __m512 vlo  = _mm512_set1_ps(-1e6f);
                    const __m512 vhi  = _mm512_set1_ps(1e6f);
                    for (; row + 15 < KD; row += 16) {
                        __m512 st = _mm512_loadu_ps(Sc + row);
                        __m512 vn = _mm512_loadu_ps(v_new + row);
                        st = _mm512_add_ps(_mm512_mul_ps(vdec, st), _mm512_mul_ps(vn, vk));
                        st = _mm512_min_ps(_mm512_max_ps(st, vlo), vhi);
                        _mm512_storeu_ps(Sc + row, st);
                    }
                }
#elif defined(PICOLM_AVX2) || defined(PICOLM_AVX)
                {
                    const __m256 vk = _mm256_set1_ps(k_col);
                    const __m256 vdec = _mm256_set1_ps(dec);
                    const __m256 vlo = _mm256_set1_ps(-1e6f);
                    const __m256 vhi = _mm256_set1_ps(1e6f);
                    for (; row + 7 < KD; row += 8) {
                        __m256 st = _mm256_loadu_ps(Sc + row);
                        __m256 vn = _mm256_loadu_ps(v_new + row);
                        st = _mm256_add_ps(_mm256_mul_ps(vdec, st), _mm256_mul_ps(vn, vk));
                        st = _mm256_min_ps(_mm256_max_ps(st, vlo), vhi);
                        _mm256_storeu_ps(Sc + row, st);
                    }
                }
#endif
                for (; row < KD; row++) {
                    float st = Sc[row];
                    st = dec * st + v_new[row] * k_col;
                    Sc[row] = fminf(fmaxf(st, -1e6f), 1e6f);
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
/* DSA k-pool indexer helpers (defined below). */
static void glm5_dsa_store_kg(const model_config_t *c, const glm5_layer_weights_t *gw,
                              run_state_t *s, const float *cur, int pos, int mla_ord);
int glm5_dsa_select(const model_config_t *c, const glm5_layer_weights_t *gw,
                    run_state_t *s, const float *cur, const float *qr,
                    int pos, int mla_ord, int *sel_out);

/* Batched MLA projections for prefill.
 *
 * Runs the three weight-bound stages of an MLA layer for n_tokens at once and
 * writes per-token planes. The attention itself is NOT here: it walks the
 * growing latent cache and stays per-token in glm5_mla_forward(), which also
 * does the cache store and the DSA indexer.
 *
 * Why this is worth a separate function: an MLA layer streams 99 MB of weights,
 * of which attn_q_b (1536x16384) and attn_output (16384x4096) are 76 MB. Doing
 * those per token re-reads them n_tokens times, and measurement shows MLA
 * layers cost 1177 ms against 817 ms for a KDA layer that streams MORE weight
 * bytes -- the per-token projections are the whole difference.
 *
 * Layouts written (all [n_tokens][...], row t at offset t*stride):
 *   qr_b   [QL]      post-norm q latent
 *   q_b    [NH*MHD]  Wq_b @ qr
 *   qabs_b [NH*KVL]  absorbed q per head   (stage 2, needs q_b complete)
 *   kv_b   [KVL]     post-norm latent, NOT yet in the cache
 *
 * attn_output is deliberately not here: it consumes the attention result, so
 * the caller batches it after the per-token attention loop.
 */
static void glm5_mla_project_batch(
    const model_t *m, const glm5_layer_weights_t *gw,
    const float *cur_b, int n_tokens,
    float *qr_b, float *q_b, float *kv_b)
{
    const model_config_t *c = &m->config;
    const int D = c->n_embd;
    const int NH = c->n_heads;
    const int QL = c->glm5_q_lora;
    const int KVL = c->glm5_kv_lora;
    const int MHD = c->glm5_mla_head_dim;
    const int QD = NH * MHD;

    /* qr = RMSNorm(Wq_a @ c, q_a_norm) */
    matmul_batch(qr_b, cur_b, n_tokens, gw->mla_q_a, D, QL, gw->type_mla_q_a);
    {
        const float *w = (const float *)gw->mla_q_a_norm;
        for (int t = 0; t < n_tokens; t++) {
            float *qr = qr_b + (size_t)t*QL;
            float ss = 0.f;
            for (int i = 0; i < QL; i++) ss += qr[i] * qr[i];
            float inv = 1.0f / sqrtf(ss / QL + c->rms_norm_eps);
            for (int i = 0; i < QL; i++) qr[i] *= inv * w[i];
        }
    }

    /* q = Wq_b @ qr  -- 1536x16384 IQ6_K, the single largest MLA weight */
    matmul_batch(q_b, qr_b, n_tokens, gw->mla_q_b, QL, QD, gw->type_mla_q_b);

    /* kv = RMSNorm(Wkv_a @ c, kv_a_norm). Cached per token by the caller.
     *
     * Deliberately per-token despite being batchable: attn_kv_a_mqa is only
     * 4096x512 (2 MB), so batching it saves nothing measurable, and it is Q8_0
     * -- where matmul_batch() selects a different kernel than matmul() and the
     * results differ by ~2e-7. On this model that is enough to flip a near-tied
     * token (see the workplan's sensitivity table), and measured against the
     * per-token reference it did exactly that. Not worth the risk for 2 MB. */
    {
        const float *w = (const float *)gw->mla_kv_a_norm;
        for (int t = 0; t < n_tokens; t++) {
            float *kv = kv_b + (size_t)t*KVL;
            matmul(kv, cur_b + (size_t)t*D, gw->mla_kv_a_mqa, D, KVL,
                   gw->type_mla_kv_a_mqa);
            float ss = 0.f;
            for (int i = 0; i < KVL; i++) ss += kv[i] * kv[i];
            float inv = 1.0f / sqrtf(ss / KVL + c->rms_norm_eps);
            for (int i = 0; i < KVL; i++) kv[i] *= inv * w[i];
        }
    }

    /* Qabs is NOT computed here: it is consumed immediately by that token's
     * attention, so keeping [n_tokens][NH*KVL] (32768 floats/token = 64 MB at
     * 512 tokens) alive through the batch would cost more arena than it saves.
     * The per-token loop computes it into one shared [NH*KVL] scratch. */
}

/* dst[i] += s * src[i], vectorised.
 *
 * The MLA ctx accumulation is a weighted sum over cache positions; written as a
 * scalar loop it was half of the 4.6 s the MLA attention spent per 120-token
 * prefill. Kept local because nothing else in the tree needs exactly this
 * signature (tensor.c has vec_add / elemwise_mul, neither with a scale).
 *
 * SIMD tiers match the rest of the tree: AVX-512 (16-wide), AVX2 (8-wide), NEON
 * (4-wide), then a scalar tail. Note the AVX2 tier uses _mm256_fmadd_ps, which
 * needs FMA, so it is gated on PICOLM_AVX2 and not on PICOLM_AVX -- an AVX-only
 * build has no FMA and the Makefile does not define PICOLM_FMA for it. AVX-only
 * therefore falls through to the scalar loop, which is correct; GLM-5 is not a
 * sensible target below AVX2 anyway. */
static void vec_scale_acc_f32(float *dst, const float *src, float s, int n) {
    int i = 0;
#if defined(PICOLM_AVX512)
    {
        const __m512 vs = _mm512_set1_ps(s);
        for (; i + 15 < n; i += 16)
            _mm512_storeu_ps(dst + i,
                _mm512_fmadd_ps(vs, _mm512_loadu_ps(src + i),
                                _mm512_loadu_ps(dst + i)));
    }
#elif defined(PICOLM_AVX2)
    {
        const __m256 vs = _mm256_set1_ps(s);
        for (; i + 7 < n; i += 8)
            _mm256_storeu_ps(dst + i,
                _mm256_fmadd_ps(vs, _mm256_loadu_ps(src + i),
                                _mm256_loadu_ps(dst + i)));
    }
#elif defined(PICOLM_NEON)
    {
        const float32x4_t vs = vdupq_n_f32(s);
        for (; i + 3 < n; i += 4)
            vst1q_f32(dst + i, vmlaq_f32(vld1q_f32(dst + i), vs, vld1q_f32(src + i)));
    }
#endif
    for (; i < n; i++) dst[i] += s * src[i];
}

/* Batched MLA attention over the latent cache (prefill, dense/DSA-free).
 *
 * The per-token core does, for one token: for each of NH heads, dot its
 * absorbed q against every cached position, softmax, then accumulate the
 * cache into ctx. Written per token that is NH * (pos+1) separate KVL-wide
 * dots and re-reads the whole cache once per head -- measured at 2.2 GFLOP/s,
 * 84% of all MLA time.
 *
 * Here the token loop is inverted: each (head, token-chunk) pair becomes one
 * task, so
 *   - the cache row is read once and used for every token in the chunk,
 *   - the work is spread over the thread pool instead of one core,
 *   - the causal mask is just a per-token bound (t <= pos), so no mask tensor.
 *
 * Only valid when there is no DSA selection: with `sel` non-empty the sparse
 * index list is per token and the per-token core is used instead (DSA measured
 * 0.0% of MLA time at 120 tokens, so this is not a hot path today).
 *
 * qabs_b is [n_tokens][NH*KVL]; ctx_b is [n_tokens][NH*KVL].
 */
typedef struct {
    const float *qabs;    /* [n_tokens][NH*KVL] */
    const float *cache;   /* [max_seq][KVL] for this MLA ordinal */
    float       *ctx;     /* [n_tokens][NH*KVL] */
    float       *scores;  /* task scratch [n_tasks][scores_stride] */
    int n_tokens, start_pos, NH, KVL, n_chunk, n_chunks, scores_stride;
    float scale;
} mla_attn_ctx_t;

static void mla_attn_task(int idx, void *vp) {
    mla_attn_ctx_t *c = (mla_attn_ctx_t *)vp;
    const int NH = c->NH, KVL = c->KVL;
    const int h  = idx / c->n_chunks;              /* head */
    const int t0 = (idx % c->n_chunks) * c->n_chunk;  /* token chunk base */
    int t1 = t0 + c->n_chunk;
    if (t1 > c->n_tokens) t1 = c->n_tokens;

    for (int t = t0; t < t1; t++) {
        const float *qh = c->qabs + (size_t)t*NH*KVL + (size_t)h*KVL;
        float *ch = c->ctx + (size_t)t*NH*KVL + (size_t)h*KVL;
        float *sc = c->scores + (size_t)idx * c->scores_stride;
        const int n_kv = c->start_pos + t + 1;   /* causal bound */

        float mx = -INFINITY;
        for (int p = 0; p < n_kv; p++) {
            float d = vec_dot_f32_f32(c->cache + (size_t)p*KVL, qh, KVL) * c->scale;
            sc[p] = d;
            if (d > mx) mx = d;
        }
        float sum = 0.f;
        for (int p = 0; p < n_kv; p++) { sc[p] = expf(sc[p] - mx); sum += sc[p]; }
        const float inv = 1.0f / sum;

        memset(ch, 0, (size_t)KVL * sizeof(float));
        for (int p = 0; p < n_kv; p++) {
            const float w = sc[p] * inv;
            if (w < 1e-10f) continue;
            vec_scale_acc_f32(ch, c->cache + (size_t)p*KVL, w, KVL);
        }
    }
}

static void glm5_mla_attn_batch(
    const model_config_t *c, run_state_t *s, int mla_ord,
    const float *qabs_b, float *ctx_b, int n_tokens, int start_pos)
{
    const int NH = c->n_heads;
    const int KVL = c->glm5_kv_lora;
    const float scale = 1.0f / sqrtf((float)c->glm5_mla_head_dim);
    const float *cache = s->glm5.mla_kv + (size_t)mla_ord * c->max_seq_len * KVL;

    /* One scores row per task: the pool runs tasks in parallel and each needs
     * its own [max_seq] scratch. Aim for ~4x as many tasks as threads so a slow
     * one does not stall the barrier. */
    int nth = 24;
    int n_chunk = (n_tokens > 0) ? (n_tokens * 4 + nth - 1) / nth : 1;
    if (n_chunk < 1) n_chunk = 1;
    const int n_chunks = (n_tokens + n_chunk - 1) / n_chunk;
    const int n_tasks = NH * n_chunks;

    static float *scores_pool = NULL;
    static size_t scores_pool_n = 0;
    size_t need = (size_t)n_tasks * (size_t)c->max_seq_len;
    if (scores_pool_n < need) {
        free(scores_pool);
        scores_pool = malloc(need * sizeof(float));
        scores_pool_n = scores_pool ? need : 0;
    }
    if (!scores_pool) return;   /* caller falls back to the per-token core */

    mla_attn_ctx_t ctx = { qabs_b, cache, ctx_b, scores_pool,
                           n_tokens, start_pos, NH, KVL, n_chunk, n_chunks,
                           c->max_seq_len, scale };
    /* tensor_parallel_for passes the task index, so the task itself picks its
     * scores slice; it must be told where the pool starts. */
    ctx.scores = scores_pool;
    tensor_parallel_for(n_tasks, mla_attn_task, &ctx);
}

/* Per-token MLA core.
 *
 * `pre` is optional. When non-NULL the caller has already run
 * glm5_mla_project_batch() for the whole token batch and `pre` points at this
 * token's slice; the projection stages below are then skipped. When NULL the
 * function does everything itself, which is the decode path.
 *
 * The attention cannot be batched away: it walks the latent cache up to `pos`,
 * and in a batched prefill every token's latents must already be stored before
 * any attention runs, so the caller stores the whole batch first (see
 * glm5_mla_project_batch + the prefill loop). */
typedef struct {
    const float *qr;    /* [QL]      post-norm q latent */
    const float *q;     /* [NH*MHD]  Wq_b @ qr           */
    const float *qabs;  /* [NH*KVL]  absorbed q per head  */
    const float *kv;    /* [KVL]     post-norm latent, already stored in cache */
} glm5_mla_pre_t;

/* Decode-time MLA attention over the latent cache                     */
/*                                                                     */
/* This is the ONLY part of GLM-5 decode whose cost grows with `pos`,   */
/* and until now it dominated deep-context decoding completely:        */
/*   1.4 ms/token at pos 0  ->  1444 ms/token at pos 7100 (76% of the  */
/*   whole decode step, 0.5 tok/s → ~0 tok/s).                          */
/*                                                                     */
/* Three things were wrong with the original loop:                     */
/*                                                                     */
/*  1. It was single-threaded. The rest of the decode path uses        */
/*     tensor_parallel_for(); this loop did not, so one core scanned   */
/*     the cache while 23 sat in the pool barrier.                     */
/*                                                                     */
/*  2. Every head re-read the entire cache. The cache is a SHARED      */
/*     latent ([KVL=512] per position, one row for all NH=64 heads),   */
/*     so the naive loop read each row 64 times:                       */
/*       932 MB per layer x 11 layers = 10.2 GB per token.            */
/*     Measured 7.2 GB/s of single-core bandwidth -- i.e. it was       */
/*     purely a memory-bandwidth crawl, not a FLOP problem.            */
/*     Fix: block over heads, so one cache row load feeds HB heads    */
/*     (HB accumulators). Row traffic drops by HB.                     */
/*                                                                     */
/*  3. It ran a separate `scores` pass and a separate `ctx` pass, both */
/*     re-streaming the cache. Now the score pass is shared across the */
/*     whole head block, so the cache is streamed twice per token      */
/*     instead of 2 * NH times.                                        */
/*                                                                     */
/* Correctness: per head, positions are still accumulated in ascending  */
/* order by glm5_mla_reduce_head(), so the result is BIT-IDENTICAL to   */
/* the old loop (verified: all 154880 logits, max|diff| = 0, both at    */
/* pos 0 and at pos 7109). Only the scheduling changed.                 */
/* PICOLM_GLM5_MLA_SCALAR=1 restores the original single-threaded       */
/* per-head loop, for A/B and as the DSA path.                          */
/* ------------------------------------------------------------------ */

/* PICOLM_GLM5_MLA_SCALAR=1 forces the original per-head single-threaded
 * attention loop. Kept as an escape hatch for bit-exact A/B and as the
 * reference the threaded path is validated against. */
static int glm5_mla_scalar_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("PICOLM_GLM5_MLA_SCALAR") ? atoi(getenv("PICOLM_GLM5_MLA_SCALAR")) : 0;
    return cached;
}

/* Heads per task in the decode attention. Each head costs KVL floats of
 * accumulator, and the score row is n_kv floats, so 4..8 heads keeps the
 * working set in L2 while still amortising the cache-row load 4..8x.
 * PICOLM_GLM5_MLA_HB overrides (1 reproduces per-head traffic). */
static int glm5_mla_headblock(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("PICOLM_GLM5_MLA_HB");
        cached = e ? atoi(e) : 4;
    }
    return cached;
}

/* Per-task score scratch, grown on demand and reused across layers/tokens.
 * Size is n_blocks*hb*n_kv floats; freed at exit by the OS. */
static float *glm5_mla_scores_scratch(size_t need) {
    static float *buf = NULL;
    static size_t cap = 0;
    if (need > cap) {
        float *nb = (float *)realloc(buf, need * sizeof(float));
        if (!nb) return NULL;
        buf = nb; cap = need;
    }
    return buf;
}

typedef struct {
    const float *cache;   /* [max_seq][KVL] rows for this MLA ordinal */
    const int   *sel;     /* DSA selection, or NULL for dense */
    const float *qabs;    /* [NH][KVL] absorbed queries */
    float       *ctx;     /* [NH][KVL] output */
    float       *scores;  /* [n_blocks][hb][n_kv] scratch */
    int NH, KVL, n_kv, n_sel, dense, hb;
    float scale;
} glm5_mla_thr_t;

/* Softmax + weighted accumulation for one head over the position list already
 * scored into sc[]. sc[] is overwritten with the normalised weights. */
static inline void glm5_mla_reduce_head(
        float *ch, float *sc, const float *cache, const int *sel,
        int n, int dense, int KVL)
{
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) { float v = sc[i]; if (v > mx) mx = v; }
    float sum = 0.f;
    for (int i = 0; i < n; i++) { float e = expf(sc[i] - mx); sc[i] = e; sum += e; }
    const float inv = 1.0f / sum;
    memset(ch, 0, (size_t)KVL * sizeof(float));
    for (int i = 0; i < n; i++) {
        const float p = sc[i] * inv;
        if (p < 1e-10f) continue;
        const int t = dense ? i : sel[i];
        vec_scale_acc_f32(ch, cache + (size_t)t * KVL, p, KVL);
    }
}

/* One head block: score its heads row-by-row (so the row is loaded once and
 * used hb times instead of once per head), then softmax + PV accumulate each
 * head of the block. */
static void glm5_mla_head_block_task(int blk, void *vp) {
    glm5_mla_thr_t *t = (glm5_mla_thr_t *)vp;
    const int h0 = blk * t->hb;
    int hn = t->NH - h0;
    if (hn > t->hb) hn = t->hb;
    if (hn <= 0) return;
    const int KVL = t->KVL;
    const int n   = t->dense ? t->n_kv : t->n_sel;
    float *sb = t->scores + (size_t)blk * t->hb * (size_t)t->n_kv;

    for (int p = 0; p < n; p++) {
        const float *row = t->cache + (size_t)(t->dense ? p : t->sel[p]) * KVL;
        for (int k = 0; k < hn; k++)
            sb[(size_t)k * n + p] =
                vec_dot_f32_f32(row, t->qabs + (size_t)(h0 + k) * KVL, KVL) * t->scale;
    }
    for (int k = 0; k < hn; k++)
        glm5_mla_reduce_head(t->ctx + (size_t)(h0 + k) * KVL, sb + (size_t)k * n,
                             t->cache, t->sel, n, t->dense, KVL);
}

static void glm5_mla_forward(
    const model_t *m, const glm5_layer_weights_t *gw, const layer_weights_t *lw,
    run_state_t *s, const float *cur, int pos, int n_ctx, int layer_idx,
    int mla_ord, float *out, const glm5_mla_pre_t *pre)
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

    /* Precomputed projections (batched prefill): read straight from the
     * caller's planes. Nothing is written to qr/qabs/kv in that case. */
    const float *qr_p   = pre ? pre->qr   : qr;
    const float *qabs_p = pre ? pre->qabs : qabs;

    /* DSA k-pool selection for this token (empty -> dense attention). */
    int *sel = NULL;
    int n_sel = 0;
    static int dsa_sel_buf[GLM5_DSA_MAX_SEL];
    (void)n_ctx; (void)layer_idx;
    /* NOTE: `cur` is ALREADY RMSNorm(cur, attn_norm) — the caller applies it
     * once for both KDA and MLA. Do not norm again here. */

    /* qr = RMSNorm(Wq_a @ c, q_a_norm) */
    if (!pre) {
        float *tmp = s->glm5.mla_q;   /* reuse: [NH*MHD] >= [QL] */
        matmul(tmp, cur, gw->mla_q_a, D, QL, gw->type_mla_q_a);
        float ss = 0.f;
        for (int i = 0; i < QL; i++) ss += tmp[i] * tmp[i];
        float inv = 1.0f / sqrtf(ss / QL + c->rms_norm_eps);
        const float *w = (const float *)gw->mla_q_a_norm;   /* F32 in GGUF */
        for (int i = 0; i < QL; i++) qr[i] = tmp[i] * inv * w[i];
    }

    /* q = Wq_b @ qr   [NH*MHD] */
    if (!pre) {
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

    /* kv = RMSNorm(Wkv_a @ c, kv_a_norm) -> latent, store in cache.
     * With `pre` the latent is precomputed AND already stored by the caller
     * (the whole batch has to be in the cache before any attention runs). */
    if (!pre) {
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

    /* DSA indexer: cache this token's (key ; gate) pair and pick the pools
     * attention may look at. Runs after the latent store so that the current
     * position is visible to the selector exactly as in the reference (which
     * writes kv_cmpr and kr_l in the same graph step).
     *
     * Note this must happen even for a single-token step: the cache has to be
     * filled at every position or later positions would pool over stale rows. */
    if (c->glm5_use_dsa && s->glm5.idx_kg && gw->idx_attn_k) {
        double _t_idx = glm5_prof_mode() ? glm5_prof_now() : 0.0;
        glm5_dsa_store_kg(c, gw, s, cur, pos, mla_ord);
        /* Selection is only meaningful once there is at least one complete
         * pool plus the tail; below that the dense path is equivalent. */
        if (pos >= c->glm5_idx_kpool) {
            n_sel = glm5_dsa_select(c, gw, s, cur, qr_p, pos, mla_ord, dsa_sel_buf);
            if (n_sel > 0) sel = dsa_sel_buf;
        }
        if (_t_idx) glm5_prof_add(GP_IDX, _t_idx);
    }

    /* Attention over the cached latents.
     *
     * With DSA active, `sel` holds the cell indices chosen by the k-pool
     * indexer (ascending, deduplicated) and attention is restricted to them.
     * `n_sel == 0` means dense attention over all n_kv positions, which is
     * also what the reference does whenever the cache is too small to pool.
     * The selection is causal by construction: the indexer only ever scores
     * pools whose last member has already been seen. */
    {
        const float scale = 1.0f / sqrtf((float)MHD);
        float *cache = s->glm5.mla_kv + (size_t)mla_ord * c->max_seq_len * KVL;
        const int n_kv = pos + 1;
        const int dense = (n_sel == 0);
        double _t_attn = glm5_prof_mode() ? glm5_prof_now() : 0.0;

        if (glm5_mla_scalar_enabled() || !dense) {
            /* ---- Original single-threaded per-head loop (reference). ---- */
            for (int h = 0; h < NH; h++) {
                const float *qh = qabs_p + h * KVL;
                float *ch = ctx + h * KVL;
                if (dense) {
                    for (int t = 0; t < n_kv; t++)
                        scores[t] = vec_dot_f32_f32(cache + (size_t)t * KVL, qh, KVL) * scale;
                    glm5_mla_reduce_head(ch, scores, cache, NULL, n_kv, 1, KVL);
                } else {
                    for (int k = 0; k < n_sel; k++)
                        scores[k] = vec_dot_f32_f32(cache + (size_t)sel[k] * KVL, qh, KVL) * scale;
                    glm5_mla_reduce_head(ch, scores, cache, sel, n_sel, 0, KVL);
                }
            }
        } else {
            /* ---- Threaded, head-blocked dense attention. ----
             * One task per head block. A task scans the cache once and scores
             * all `hb` heads of its block against each row as that row is
             * loaded, so cache-row traffic drops by a factor of hb compared to
             * the per-head loop (which re-read the whole cache 64 times).
             * Tasks run on the shared pool, so the scan is also spread over
             * all cores instead of one.
             *
             * Score scratch is per task ([hb][n_kv]), so memory is
             * n_blocks * hb * n_kv floats -- the same total as the old single
             * [max_seq] row times hb, and independent of thread count. */
            int hb = glm5_mla_headblock();
            if (hb < 1) hb = 1;
            if (hb > NH) hb = NH;
            const int n_blocks = (NH + hb - 1) / hb;
            float *scr = glm5_mla_scores_scratch((size_t)n_blocks * hb * (size_t)n_kv);
            if (!scr) {
                /* Scratch allocation failed: use the scalar loop (identical math). */
                for (int h = 0; h < NH; h++) {
                    const float *qh = qabs_p + h * KVL;
                    for (int t = 0; t < n_kv; t++)
                        scores[t] = vec_dot_f32_f32(cache + (size_t)t * KVL, qh, KVL) * scale;
                    glm5_mla_reduce_head(ctx + h * KVL, scores, cache, NULL, n_kv, 1, KVL);
                }
            } else {
                glm5_mla_thr_t tc;
                tc.cache = cache; tc.sel = NULL; tc.qabs = qabs_p; tc.ctx = ctx;
                tc.scores = scr; tc.NH = NH; tc.KVL = KVL; tc.n_kv = n_kv;
                tc.dense = 1; tc.hb = hb; tc.scale = scale;
                if (glm5_prof_mode() >= 4) {
                    static int shown = 0;
                    if (!shown++) fprintf(stderr, "[glm5 mla] n_blocks=%d hb=%d n_kv=%d n_threads=%d\n",
                                           n_blocks, hb, n_kv, tensor_get_n_threads());
                }
                tensor_parallel_for(n_blocks, glm5_mla_head_block_task, &tc);
            }
        }
        if (_t_attn) glm5_prof_add(GP_ATTN, _t_attn);
        /* Report the work actually done so a run can be checked against the
         * theoretical O(pos * NH * KVL) model. */
        if (glm5_prof_mode() >= 3) {
            fprintf(stderr, "[glm5 prof] l%d mla pos=%d dense=%d n_sel=%d n_kv=%d attn=%.2fms\n",
                    layer_idx, pos, dense, n_sel, n_kv, glm5_prof_now() - _t_attn);
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
        /* o_concat is in caller-provided scratch when batching; the output
         * projection is then batched too (attn_output is 16384x4096 IQ6_K,
         * 55 MB, and would otherwise be re-streamed per token). */
        if (pre) memcpy(out, o_concat, (size_t)NH*MHD * sizeof(float));
        else    matmul(out, o_concat, lw->attn_output, NH*MHD, D, lw->type_attn_output);
    }
}

/* ------------------------------------------------------------------ */
/* DSA k-pool indexer (sparse attention on MLA layers)                 */
/*                                                                     */
/* Compresses the latent KV cache into pools of `kpool` positions. Each */
/* pool key is a learned convex mix of its member keys (softmax over a  */
/* per-member gate + intra-pool position bias APE). Pools are scored    */
/* against the query, the top-k POOLS are selected, and those pools are */
/* expanded back into the individual cache positions that attention is  */
/* then restricted to. Reference: build_glm5next_dsa_top_k() in         */
/* ik_llama.cpp, and notes/picolm/glm53_architecture_reference.md §5.   */
/*                                                                     */
/* Without this, attention is dense over the whole cache. GLM-5.3 was   */
/* trained with the indexer active, so dense attention matches the      */
/* reference only while the whole context fits in top_k cells            */
/* (indexer.top_k = 2048): beyond that the reference silently drops the */
/* least-relevant pools and PicoLM's output diverges.                   */
/*                                                                     */
/* Returns the number of selected cell indices written to `sel_out`, or */
/* 0 to request the dense fallback (cache too small to pool).           */
/* ------------------------------------------------------------------ */


/* Per-token key+gate pair into the kr_l cache, row layout [k ; g].
 * k = LayerNorm(idx_attn_k @ cur) with weight+bias, g = idx_gate @ cur. */
static void glm5_dsa_store_kg(const model_config_t *c, const glm5_layer_weights_t *gw,
                              run_state_t *s, const float *cur, int pos, int mla_ord)
{
    const int D = c->n_embd;
    const int ID = c->glm5_idx_dim;
    float k[GLM5_MAX_IDX_DIM], g[GLM5_MAX_IDX_DIM];
    /* The indexer norm of the keys is a LayerNorm (weight AND bias) with its
     * own epsilon (attention.layer_norm_epsilon = 1e-6), deliberately not the
     * RMSNorm one used everywhere else. */
    matmul(k, cur, gw->idx_attn_k, D, ID, gw->type_idx_attn_k);
    layernorm(k, k, (const float *)gw->idx_k_norm_w, (const float *)gw->idx_k_norm_b,
              ID, c->glm5_ln_eps);
    matmul(g, cur, gw->idx_gate, D, ID, gw->type_idx_gate);
    float *row = s->glm5.idx_kg + (size_t)mla_ord * c->max_seq_len * 2 * ID
               + (size_t)pos * 2 * ID;
    memcpy(row, k, (size_t)ID * sizeof(float));
    memcpy(row + ID, g, (size_t)ID * sizeof(float));
}

/* Score pools and select the top-k. `qr` is the q_lora latent of this token.
 * Writes cell indices (ascending, deduplicated) into sel_out. */
int glm5_dsa_select(const model_config_t *c, const glm5_layer_weights_t *gw,
                    run_state_t *s, const float *cur, const float *qr,
                    int pos, int mla_ord, int *sel_out)
{
    const int D  = c->n_embd;
    const int ID = c->glm5_idx_dim;
    const int NH = c->glm5_idx_nh;
    const int R  = c->glm5_idx_kpool;
    /* Defensive: a model without the full indexer tensor set must fall back
     * to dense attention rather than dereference a NULL weight. */
    if (!gw->idx_ape || !gw->idx_q_b || !gw->idx_proj) return 0;
    const int n_kv = pos + 1;
    const int n_pool = n_kv / R;
    if (n_pool == 0) return 0;

    /* Only complete pools are selectable, and the trailing partial pool is
     * always appended so the most recent tokens are never lost. */
    const int tail_cnt = R - 1;
    const int n_sel = (int)((c->glm5_idx_topk / R));
    int n_sel_ok = n_sel < n_pool ? n_sel : n_pool;
    if (((n_kv - tail_cnt) / R) < n_sel_ok) n_sel_ok = (n_kv - tail_cnt) / R;
    if (n_sel_ok < 1) return 0;

    const float *base = s->glm5.idx_kg + (size_t)mla_ord * c->max_seq_len * 2 * ID;

    /* ---- pool keys: pooled[p][i] = sum_j w[p][j] * k[member j][i] ---- */
    if (n_pool > GLM5_DSA_MAX_POOL) return 0;
    if (ID > GLM5_MAX_IDX_DIM || NH > GLM5_MAX_IDX_NH || R > GLM5_MAX_KPOOL) return 0;
    static float pooled[GLM5_DSA_MAX_POOL * GLM5_MAX_IDX_DIM];
    static float score[GLM5_DSA_MAX_POOL];

    for (int p = 0; p < n_pool; p++) {
        /* Pool key for pool p (members are positions p*R .. p*R+R-1): a
         * per-CHANNEL softmax over members of (gate + APE), then a convex
         * combination of the member keys. The reference permutes the member
         * axis to ne[0] before ggml_soft_max precisely so that each channel
         * normalises independently -- it is not a single weight per member. */
        float w[GLM5_MAX_KPOOL];
        float *pk = pooled + (size_t)p * ID;
        memset(pk, 0, (size_t)ID * sizeof(float));
        for (int i = 0; i < ID; i++) {
            float m = -INFINITY;
            for (int j = 0; j < R; j++) {
                const float *gr = base + (size_t)(p * R + j) * 2 * ID + ID;
                float v = gr[i] + ((const float *)gw->idx_ape)[(size_t)j * ID + i];
                w[j] = v;
                if (v > m) m = v;
            }
            float sum = 0.f;
            for (int j = 0; j < R; j++) { w[j] = expf(w[j] - m); sum += w[j]; }
            float inv = 1.f / sum;
            float acc = 0.f;
            for (int j = 0; j < R; j++) {
                const float *kr = base + (size_t)(p * R + j) * 2 * ID;
                acc += w[j] * inv * kr[i];
            }
            pk[i] = acc;
        }
    }

    /* ---- indexer query and weights ---- */
    static float qidx[GLM5_MAX_IDX_NH * GLM5_MAX_IDX_DIM];
    float wts[GLM5_MAX_IDX_NH];
    if (R > GLM5_MAX_KPOOL) return 0;
    if (NH * ID > GLM5_MAX_IDX_NH * GLM5_MAX_IDX_DIM) return 0;
    matmul(qidx, qr, gw->idx_q_b, c->glm5_q_lora, NH * ID, gw->type_idx_q_b);
    /* proj is F32 in the reference and is multiplied by 1/sqrt(d*nh); a
     * quantized proj would perturb the discrete top-k and cannot be
     * corrected for downstream. */
    matmul(wts, cur, gw->idx_proj, D, NH, gw->type_idx_proj);
    {
        const float scl = 1.0f / sqrtf((float)(ID * NH));
        for (int h = 0; h < NH; h++) wts[h] *= scl;
    }

    /* ---- score[p] = sum_h wts[h] * relu(pooled[p] . q_idx[h]) ---- */
    for (int p = 0; p < n_pool; p++) {
        const float *pk = pooled + (size_t)p * ID;
        float acc = 0.f;
        for (int h = 0; h < NH; h++) {
            const float *qh = qidx + (size_t)h * ID;
            float dot = 0.f;
            for (int i = 0; i < ID; i++) dot += pk[i] * qh[i];
            if (dot < 0.f) dot = 0.f;          /* relu */
            acc += wts[h] * dot;
        }
        score[p] = acc;
    }

    /* Mask pools that are not fully visible to this query: a pool is visible
     * once its last member position (p*R + R - 1) has been seen. */
    const int n_pool_visible = (n_kv / R);
    for (int p = 0; p < n_pool; p++)
        if (p >= n_pool_visible || (p * R + R - 1) > pos) score[p] = -INFINITY;

    /* ---- top-k pools (selection sort; n_pool <= 2048 here) ---- */
    int picked[GLM5_DSA_MAX_SEL];
    int n_picked = 0;
    {
        /* Simple partial selection: repeatedly take the max remaining. */
        static unsigned char used[GLM5_DSA_MAX_POOL];
        memset(used, 0, (size_t)n_pool);
        for (int k = 0; k < n_sel_ok; k++) {
            int best = -1; float bv = -INFINITY;
            for (int p = 0; p < n_pool; p++) {
                if (used[p]) continue;
                if (score[p] > bv) { bv = score[p]; best = p; }
            }
            if (best < 0 || bv == -INFINITY) break;
            used[best] = 1;
            picked[n_picked++] = best;
        }
    }

    /* ---- expand pools to cell indices, then append the tail ---- */
    int n_out = 0;
    for (int k = 0; k < n_picked; k++)
        for (int j = 0; j < R; j++)
            if (n_out < GLM5_DSA_MAX_SEL) sel_out[n_out++] = picked[k] * R + j;
    {
        /* Trailing incomplete pool: positions (n_pool*R) .. pos inclusive. */
        for (int t = n_pool * R; t <= pos && n_out < GLM5_DSA_MAX_SEL; t++)
            sel_out[n_out++] = t;
    }
    /* Sort ascending and dedup so the caller can walk the cache linearly
     * (keeps the attention loop's memory access pattern sequential). */
    for (int i = 1; i < n_out; i++) {
        int v = sel_out[i], j = i - 1;
        while (j >= 0 && sel_out[j] > v) { sel_out[j + 1] = sel_out[j]; j--; }
        sel_out[j + 1] = v;
    }
    {
        int w = 0;
        for (int i = 0; i < n_out; i++)
            if (i == 0 || sel_out[i] != sel_out[w - 1]) sel_out[w++] = sel_out[i];
        n_out = w;
    }
    return n_out;
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

/* GLM-5 shared expert: ungated SwiGLU, added to `out` (no routing weight).
 * Split out so the batched MoE can reuse it verbatim. */
static void glm5_moe_shared_expert(const model_t *m, const layer_weights_t *lw,
                                   run_state_t *s, const float *cur, float *out,
                                   int SFF)
{
    const model_config_t *c = &m->config;
    const int D = c->n_embd;
    const float clamp_v = glm5_clamp_value(c);

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
    for (int i = 0; i < D; i++) out[i] += s->glm5.moe_down[i];
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
        glm5_moe_shared_expert(m, lw, s, cur, out, SFF);
        snprintf(t, sizeof(t), "x.moe.shexp");
        glm5_stats(t, s->glm5.moe_down, D);
    }
    glm5_stats("x.moe.out", out, D);
    glm5_chk("x.moe.out", out, D, -1e4f, 1e4f);
}

/* Batched MoE for prefill.
 *
 * The per-token glm5_moe_ffn() reads 8 experts (165 MB) per token. With 288
 * experts and top-8, a 110-token prefill selects essentially ALL of them, so
 * the same expert weights get streamed ~3x. Sorting tokens by expert lets each
 * expert's weights be read once and GEMM'd against all its tokens.
 *
 * Routing semantics are IDENTICAL to glm5_moe_ffn() and stay per token:
 * sigmoid for the weight, +bias for selection only, renormalise, * moe_scale,
 * ungated shared expert. Only the expert loop is restructured.
 *
 * Note the sorted group is small -- n_tokens*topk/n_expert is ~3 at 110 tokens
 * -- so this does NOT become a big GEMM. The win is purely reading each
 * expert's weights once instead of once per selecting token; matmul_batch at
 * n_batch=3 still uses the same vec_dot kernels as the per-token path.
 *
 * TODO(unexplained long-generation divergence):
 *   With PICOLM_GLM5_MOE_PERTOKEN as the A/B switch, generated output is
 *   identical up to ~20 tokens but diverges at >=40 on some prompts, e.g.
 *
 *     -pf "Write a C function that reverses a string in place, with a brief
 *          explanation." -n 120 -c 512 -t 0 -j 24 --prefault
 *
 *   The two outputs differ only by one blank line after
 *   `if (str == NULL) return;` -- i.e. a single token -- and then re-converge.
 *
 *   THIS IS NOT (directly) THIS FUNCTION. Measured with the same
 *   worst_rel-vs-glm5_moe_ffn() instrumentation that proved the batching
 *   exact: the batched MoE output is BIT-IDENTICAL (worst_rel == 0, every
 *   layer, prefill AND generation steps). So the divergence is upstream or
 *   downstream of the MoE, and only becomes visible once generation is long
 *   enough to reach a near-tied token (see the workplan's sensitivity table:
 *   a 1e-6 perturbation moves logits ~8% on this model).
 *
 *   Prime suspect: the batched MLA attention in glm5_mla_attn_batch(), which
 *   was vectorised and re-ordered in the same session. It has NOT been checked
 *   against the per-token core the same way the MoE was.
 *
 *   What to do: add the worst_rel instrumentation to glm5_mla_attn_batch() vs
 *   the per-token attention path (compare ctx_b per token, and qabs_b), at
 *   prefill positions past ~20 tokens. Then either clear the MLA path or fix
 *   it. Do NOT "fix" this by loosening the comparison -- first establish
 *   whether it is a real defect or float reassociation, because the two need
 *   opposite responses.
 */
static void glm5_moe_ffn_batch(const model_t *m, const glm5_layer_weights_t *gw,
                               const layer_weights_t *lw, run_state_t *s,
                               const float *cur_b, float *out_b, int n_tokens)
{
    const model_config_t *c = &m->config;
    const int D = c->n_embd;
    const int NE = c->n_expert;
    const int AK = c->n_expert_used;
    const int FF = c->n_ff_exp;
    const float clamp_v = glm5_clamp_value(c);
    const int NT = n_tokens;

    /* Per-token routing, exactly as glm5_moe_ffn(). */
    int   *sel_all = malloc((size_t)NT * AK * sizeof(int));
    float *wts_all = malloc((size_t)NT * AK * sizeof(float));
    float *probs   = malloc((size_t)NE * sizeof(float));
    int   *count   = calloc((size_t)NE + 1, sizeof(int));
    int   *cursor  = malloc(((size_t)NE + 1) * sizeof(int));
    int   *bucket  = malloc((size_t)NT * AK * sizeof(int));   /* (t,slot) pairs */
    float *gu      = malloc((size_t)NT * AK * FF * sizeof(float));  /* gate/up staging */
    float *gu2     = malloc((size_t)NT * AK * FF * sizeof(float));
    float *dwn     = malloc((size_t)NT * AK * D * sizeof(float));   /* per-slot down out */
    float *slot_out= calloc((size_t)NT * AK * D, sizeof(float));    /* [t][slot][D] */
    float *xbuf    = malloc((size_t)NT * D * sizeof(float));        /* per-expert gather buf */
    /* All-or-nothing: every buffer is in this one check. xbuf used to be
     * allocated later and tested with `if (xbuf)`, so an OOM there silently
     * skipped every routed expert and produced wrong output with no error. */
    if (!sel_all || !wts_all || !probs || !count || !cursor || !bucket
        || !gu || !gu2 || !dwn || !slot_out || !xbuf) {
        free(sel_all); free(wts_all); free(probs); free(count); free(cursor);
        free(bucket); free(gu); free(gu2); free(dwn); free(slot_out); free(xbuf);
        /* Fall back to the per-token path rather than producing garbage.
         * glm5_moe_ffn() zeroes `out` itself, so no pre-clearing is needed. */
        for (int t = 0; t < NT; t++)
            glm5_moe_ffn(m, gw, lw, s, cur_b + (size_t)t*D, out_b + (size_t)t*D);
        return;
    }

    for (int t = 0; t < NT; t++) {
        const float *cur = cur_b + (size_t)t*D;
        float *logits = s->expert_logits;
        matmul(logits, cur, lw->ffn_gate_inp, D, NE, lw->type_ffn_gate_inp);
        for (int i = 0; i < NE; i++) logits[i] = 1.0f / (1.0f + expf(-logits[i]));
        memcpy(probs, logits, (size_t)NE * sizeof(float));
        if (gw->exp_probs_b) {
            const float *bias = (const float *)gw->exp_probs_b;
            for (int i = 0; i < NE; i++) logits[i] += bias[i];
        }
        for (int k = 0; k < AK; k++) {
            int best = -1; float bv = -1e30f;
            for (int i = 0; i < NE; i++)
                if (logits[i] > bv) { bv = logits[i]; best = i; }
            sel_all[t*AK + k] = best;
            logits[best] = -1e30f;
        }
        float sum = 0.f;
        for (int k = 0; k < AK; k++) { wts_all[t*AK+k] = probs[sel_all[t*AK+k]]; sum += wts_all[t*AK+k]; }
        float sc2 = (1.0f / sum) * c->glm5_moe_scale;
        for (int k = 0; k < AK; k++) wts_all[t*AK+k] *= sc2;
        for (int k = 0; k < AK; k++) count[sel_all[t*AK+k]]++;
    }

    /* Bucket (t, slot) by expert: counting sort, stable enough (order within a
     * bucket does not matter -- each entry writes its own output row). */
    {
        int acc = 0;
        for (int e = 0; e < NE; e++) { cursor[e] = acc; acc += count[e]; }
        memset(count, 0, (size_t)NE * sizeof(int));
        for (int t = 0; t < NT; t++)
            for (int k = 0; k < AK; k++) {
                int e = sel_all[t*AK+k];
                int pos = cursor[e] + count[e]++;
                bucket[pos] = t*AK + k;
            }
    }

    /* Accumulators start at zero: routed experts and the shared expert both
     * do out[j] += ... */
    memset(out_b, 0, (size_t)NT * D * sizeof(float));

    size_t gate_stride = (size_t)FF * gguf_type_row_size(lw->type_ffn_gate_exps, D);
    size_t down_stride = (size_t)D * gguf_type_row_size(lw->type_ffn_down_exps, FF);

    /* Gather activations for one expert into a contiguous [count][D] block so
     * the GEMM reads them contiguously too. */
    {
        for (int e = 0; e < NE; e++) {
            int n_e = count[e];
            if (n_e <= 0) continue;
            const int *bk = bucket + cursor[e];
            for (int i = 0; i < n_e; i++)
                memcpy(xbuf + (size_t)i*D, cur_b + (size_t)(bk[i]/AK)*D,
                       (size_t)D * sizeof(float));

            const void *gate_w = (const uint8_t *)lw->ffn_gate_exps + (size_t)e * gate_stride;
            const void *up_w   = (const uint8_t *)lw->ffn_up_exps   + (size_t)e * gate_stride;
            const void *down_w = (const uint8_t *)lw->ffn_down_exps + (size_t)e * down_stride;

            /* One read of this expert's gate+up, GEMM'd against all its tokens.
             *
             * n_e == 1 goes through matmul() rather than matmul_batch(): the
             * two pick different kernels for a single column (fn_gemv vs
             * threaded_worker), which was enough to shift the MoE output by
             * 7e-8 and, after 45 layers of amplification, flip a near-tied
             * token away from the verified per-token output. Single-token
             * buckets are also the common case for a small prompt, so this
             * costs nothing -- the win comes from the n_e >= 2 buckets. */
            if (n_e == 1) {
                matmul(gu,  xbuf, gate_w, D, FF, lw->type_ffn_gate_exps);
                matmul(gu2, xbuf, up_w,   D, FF, lw->type_ffn_up_exps);
            } else {
                matmul_batch(gu,  xbuf, n_e, gate_w, D, FF, lw->type_ffn_gate_exps);
                matmul_batch(gu2, xbuf, n_e, up_w,   D, FF, lw->type_ffn_up_exps);
            }

            for (int i = 0; i < n_e; i++) {
                float *g = gu  + (size_t)i*FF;
                float *u = gu2 + (size_t)i*FF;
                for (int j = 0; j < FF; j++) {
                    float gv = g[j], uv = u[j];
                    if (clamp_v > 0.0f) {
                        gv = fminf(gv, clamp_v);
                        uv = fminf(fmaxf(uv, -clamp_v), clamp_v);
                    }
                    g[j] = gv / (1.0f + expf(-gv)) * uv;
                }
            }
            if (n_e == 1) matmul(dwn, gu, down_w, FF, D, lw->type_ffn_down_exps);
            else          matmul_batch(dwn, gu, n_e, down_w, FF, D, lw->type_ffn_down_exps);
            /* Stash per (token, slot) instead of accumulating now. Expert-sorted
             * order differs from the reference's slot order, and adding eight
             * 4096-long vectors in a different order changes the float result
             * by ~1 ulp -- enough, after 45 layers, to flip near-tied tokens and
             * (measured) turn "2+2?" into a repetition loop. Summing in slot
             * order afterwards reproduces the per-token path exactly. */
            for (int i = 0; i < n_e; i++) {
                int ts = bk[i];
                memcpy(slot_out + (size_t)ts*D, dwn + (size_t)i*D,
                       (size_t)D * sizeof(float));
            }
        }
    }

    /* Reduce slots in order 0..AK-1, exactly like the per-token path. */
    for (int t = 0; t < NT; t++)
        for (int k = 0; k < AK; k++) {
            float w = wts_all[t*AK+k];
            const float *dd = slot_out + ((size_t)t*AK + k)*D;
            float *o = out_b + (size_t)t*D;
            for (int j = 0; j < D; j++) o[j] += w * dd[j];
        }

    /* Shared expert, still per token (1/8 the flops, 36 ms/layer total). */
    if (lw->ffn_gate_shexp) {
        const int SFF = c->n_ff_shexp;
        for (int t = 0; t < NT; t++)
            glm5_moe_shared_expert(m, lw, s, cur_b + (size_t)t*D, out_b + (size_t)t*D, SFF);
    }

    free(sel_all); free(wts_all); free(probs); free(count); free(cursor);
    free(bucket); free(gu); free(gu2); free(dwn); free(slot_out); free(xbuf);
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
        BENCH_LAYER_START();
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
        double _t_attnblk = glm5_prof_mode() ? glm5_prof_now() : 0.0;
        if (glm5_noattn_enabled()) {
            memset(attn_out, 0, (size_t)D * sizeof(float));
        } else if (mla_ord >= 0) {
            /* GP_MLA/GP_ATTN/GP_IDX are accounted for inside glm5_mla_forward;
             * subtract them back out so the remainder is pure projection cost. */
            double _i0 = g_gp.ms[GP_ATTN] + g_gp.ms[GP_MLA] + g_gp.ms[GP_IDX];
            glm5_mla_forward(m, gw, lw, s, s->glm5.cur, pos, c->max_seq_len, il, mla_ord, attn_out, NULL);
            if (glm5_prof_mode()) {
                double _inner = (g_gp.ms[GP_ATTN] + g_gp.ms[GP_MLA] + g_gp.ms[GP_IDX]) - _i0;
                double _wall = glm5_prof_now() - _t_attnblk;
                if (_wall > _inner) g_gp.ms[GP_MLA] += _wall - _inner;
            }
            _t_attnblk = 0.0;
        } else {
            glm5_kda_forward(m, gw, lw, s, s->glm5.cur, pos, il, attn_out);
        }
        if (_t_attnblk) glm5_prof_add(GP_KDA, _t_attnblk);

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
        double _t_ffn = glm5_prof_mode() ? glm5_prof_now() : 0.0;
        if (il < c->n_layer_dense_lead) {
            /* dense FFN (layers 0..n_dense-1 are dense, rest are MoE) */
            glm5_dense_ffn(m, lw, s, s->glm5.cur, s->glm5.attn_out);
        } else {
            glm5_moe_ffn(m, gw, lw, s, s->glm5.cur, s->glm5.attn_out);
        }
        if (_t_ffn) glm5_prof_add(GP_MOE, _t_ffn);

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

        BENCH_LAYER_END(il, 0);

        if (stop_layer >= 0 && il >= stop_layer) {
            if (glm5_dbg_enabled())
                fprintf(stderr, "GLM5: stopping after layer %d (PICOLM_STOP_LAYER)\n", il);
            break;
        }

#ifdef PICOLM_VIZ
        /* Push mean-reduced activation for visualization.
         * GLM has S residual streams; viz expects a single dim-sized vector. */
        {
            float inv_s = 1.0f / (float)S;
            for (int d = 0; d < D; d++) {
                float acc = 0.f;
                for (int i = 0; i < S; i++) acc += s->glm5.hc_x[i*D + d];
                s->glm5.cur[d] = acc * inv_s;
            }
            viz_push_row(il, pos, s->glm5.cur, D);
        }
#endif
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
        double _t_head = glm5_prof_mode() ? glm5_prof_now() : 0.0;
        matmul(s->logits, s->glm5.cur, w->output, D, c->vocab_size, w->type_output);
        if (_t_head) glm5_prof_add(GP_HEAD, _t_head);
    }
    if (glm5_prof_mode()) {
        if (g_gp.tokens == 0) atexit(glm5_prof_print);
        g_gp.tokens++;
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


/* ------------------------------------------------------------------ */
/* Batched prefill                                                     */
/* ------------------------------------------------------------------ */

/* KDA tail: conv1d + SiLU + L2 norm + recurrence + head norm + gate + out
 * projection. Extracted verbatim from glm5_kda_forward() so that the batched
 * prefill can call it per-token after batching the projections.
 *
 * Inputs (all pre-computed by the caller, per token):
 *   q, k, v  [DI]  raw projections (NOT convolved)
 *   z        [DI]  gate pre-activation
 *   raw      [DI]  decay raw input
 *   beta_raw [H]   beta pre-activation
 * Scratch (caller-provided, per-token or shared -- tail is sequential):
 *   conv_out [3*DI], g [DI], v_new [KD], beta [H]
 * Output:
 *   out      [D]   attention output projection
 *
 * Bit-identical to glm5_kda_forward() for the same inputs: the conv, state
 * recurrence, norms and gate are the same code, just moved. The state
 * (kda_state, conv_state) is read and written exactly once per call, so
 * calling this per token in sequence order reproduces the per-token path.
 */
static void glm5_kda_tail(
    const model_t *m, const glm5_layer_weights_t *gw, const layer_weights_t *lw,
    run_state_t *s, int layer_idx, int pos,
    const float *q, const float *k, const float *v,
    const float *z, const float *raw, const float *beta_raw,
    float *conv_out, float *g, float *head_out,
    float *v_new, float *beta,
    float *out, int do_out_proj)
{
    const model_config_t *c = &m->config;
    const int H = c->glm5_n_kda_heads;
    const int KD = c->glm5_kda_dim;
    const int DI = H * KD;
    const int CONV_K = c->ssm_d_conv;
    const int CONV_DIM = 3 * DI;
    const int D = c->n_embd;
    (void)pos; (void)layer_idx;

    float *kda_state = s->glm5.kda_state[layer_idx];
    float *conv_state = s->glm5.kda_conv_state[layer_idx];

    /* beta = sigmoid(beta_raw) */
    for (int h = 0; h < H; h++)
        beta[h] = 1.0f / (1.0f + expf(-beta_raw[h]));

    /* g[h*KD+ch] = lower_bound * sigmoid(-a[h] * (raw[idx] + dt_bias[idx])) */
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

    /* Causal depthwise conv1d over concatenated [q;k;v] then SiLU. */
    {
        const float *wq = (const float *)gw->kda_conv_q;
        const float *wk = (const float *)gw->kda_conv_k;
        const float *wv = (const float *)gw->kda_conv_v;
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
            conv_out[ch] = acc / (1.0f + expf(-acc));
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

    /* Recurrent state update + output (column-major state walk). */
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
            float decay[GLM5_KD_MAX];

            float attn_score = 0.f;
            for (int i = 0; i < KD; i++)
                attn_score += kh[i] * qh[i];
            attn_score *= scale;

            for (int col = 0; col < KD; col++)
                decay[col] = expf(fminf(g[h*KD + col], 50.0f));

            {
                float *vp = v_new;
                float *ov = head_out + h*KD;
                memset(vp, 0, (size_t)KD * sizeof(float));
                memset(ov, 0, (size_t)KD * sizeof(float));
                for (int col = 0; col < KD; col++) {
                    const float *Scol = Sh + (size_t)col*KD;
                    const float kd_ = kh[col] * decay[col];
                    const float qd_ = qh[col] * decay[col];
                    for (int row = 0; row < KD; row++) {
                        float st = Scol[row];
                        vp[row] += st * kd_;
                        ov[row] += st * qd_;
                    }
                }
                for (int row = 0; row < KD; row++) {
                    float vn = beta[h] * (vh[row] - vp[row]);
                    v_new[row] = vn;
                    ov[row] = ov[row] * scale + vn * attn_score;
                }
            }
            /* State update (same SIMD structure as glm5_kda_forward) */
            for (int col = 0; col < KD; col++) {
                float k_col = kh[col];
                float dec = decay[col];
                float *Sc = Sh + (size_t)col*KD;
                int row = 0;
#if defined(PICOLM_AVX512)
                {
                    const __m512 vdec = _mm512_set1_ps(dec);
                    const __m512 vk   = _mm512_set1_ps(k_col);
                    const __m512 vlo  = _mm512_set1_ps(-1e6f);
                    const __m512 vhi  = _mm512_set1_ps(1e6f);
                    for (; row + 15 < KD; row += 16) {
                        __m512 st = _mm512_loadu_ps(Sc + row);
                        __m512 vn = _mm512_loadu_ps(v_new + row);
                        st = _mm512_add_ps(_mm512_mul_ps(vdec, st), _mm512_mul_ps(vn, vk));
                        st = _mm512_min_ps(_mm512_max_ps(st, vlo), vhi);
                        _mm512_storeu_ps(Sc + row, st);
                    }
                }
#elif defined(PICOLM_AVX2) || defined(PICOLM_AVX)
                {
                    const __m256 vk = _mm256_set1_ps(k_col);
                    const __m256 vdec = _mm256_set1_ps(dec);
                    const __m256 vlo  = _mm256_set1_ps(-1e6f);
                    const __m256 vhi  = _mm256_set1_ps(1e6f);
                    for (; row + 7 < KD; row += 8) {
                        __m256 st = _mm256_loadu_ps(Sc + row);
                        __m256 vn = _mm256_loadu_ps(v_new + row);
                        st = _mm256_add_ps(_mm256_mul_ps(vdec, st), _mm256_mul_ps(vn, vk));
                        st = _mm256_min_ps(_mm256_max_ps(st, vlo), vhi);
                        _mm256_storeu_ps(Sc + row, st);
                    }
                }
#endif
                for (; row < KD; row++) {
                    float st = Sc[row];
                    st = dec * st + v_new[row] * k_col;
                    Sc[row] = fminf(fmaxf(st, -1e6f), 1e6f);
                }
            }
        }
    }

    /* Per-head RMSNorm with ssm_norm weight, then sigmoid gate (z) */
    {
        const float *norm_w = (const float *)gw->kda_norm;
        for (int h = 0; h < H; h++) {
            float *oh = head_out + h*KD;
            float ss = 0.f;
            for (int ch = 0; ch < KD; ch++) ss += oh[ch]*oh[ch];
            float inv = 1.0f / sqrtf(ss / KD + c->rms_norm_eps);
            for (int ch = 0; ch < KD; ch++) oh[ch] *= inv * norm_w[ch];
        }
        for (int i = 0; i < DI; i++)
            head_out[i] *= 1.0f / (1.0f + expf(-z[i]));
    }

    /* Output projection.
     *
     * do_out_proj == 0 leaves the gated head_out in place for the caller to
     * project for the whole token batch at once (attn_output is DI->D, 33.5M
     * IQ6_K weights, so it is worth streaming once per layer instead of once
     * per token). The value written to `out` is then the *pre-projection*
     * gated output, not the block output -- callers relying on `out` must pass
     * do_out_proj = 1. */
    if (do_out_proj)
        matmul(out, head_out, lw->attn_output, DI, D, lw->type_attn_output);
}

/* Batched GLM-5 prefill. Processes n_tokens (<= GLM5_PREFILL_MAX_TOKENS)
 * starting at start_pos. Batches the weight-bound KDA projections and the
 * attention output projection across tokens; everything else stays per-token.
 * Returns the logits of the LAST token (caller convention).
 *
 * The caller (model_forward_prefill in model_core.c) chunks prompts longer
 * than GLM5_PREFILL_MAX_TOKENS into multiple calls; the KDA recurrent state
 * and MLA latent caches live in the run state and carry across chunks. */
float *model_forward_prefill_glm5(model_t *m, const int *tokens, int n_tokens,
                                  int start_pos, volatile int *interrupt)
{
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;
    const int D = c->n_embd;
    const int S = c->glm5_hc;
    const int M = (2 + S) * S;
    const int H = c->glm5_n_kda_heads;
    const int KD = c->glm5_kda_dim;
    const int DI = H * KD;
    const int n_trunk = c->n_layer_kv_from_start > 0 && c->n_layer_kv_from_start <= c->n_layers
                      ? c->n_layer_kv_from_start : c->n_layers;
    /* PICOLM_STOP_LAYER is documented as a bisection aid for the GLM-5 forward
     * and is honoured by the per-token path. Honour it here too: without this,
     * a batched-vs-per-token bisection runs all 45 layers on one side and N on
     * the other, and every comparison looks like a massive divergence. */
    const int stop_layer = glm5_stop_layer();
    (void)interrupt;

    /* Chunk long prompts: the scratch arena is sized for GLM5_PREFILL_MAX_TOKENS
     * and the KDA recurrent state / MLA latent caches live in the run state,
     * carrying across chunk boundaries automatically. */
    while (n_tokens > GLM5_PREFILL_MAX_TOKENS) {
        model_forward_prefill_glm5(m, tokens, GLM5_PREFILL_MAX_TOKENS, start_pos, interrupt);
        tokens += GLM5_PREFILL_MAX_TOKENS;
        start_pos += GLM5_PREFILL_MAX_TOKENS;
        n_tokens -= GLM5_PREFILL_MAX_TOKENS;
    }
    if (n_tokens <= 0) return s->logits;

    /* Per-token planes into the prefill arena.
     *
     * PICOLM_GLM5_NO_BATCH forces the per-token path unconditionally. That is
     * the A/B switch for validating the batched prefill: run the same prompt
     * with and without it and compare the generated tokens (and, if needed,
     * PICOLM_DUMP_LOGITS output). Keep it: every batching change has to be
     * re-validated against this path, and the reference implementation is the
     * only thing that can tell you a new kernel went wrong.
     *
     * Note that the two paths are NOT expected to agree bit-for-bit for
     * n_tokens > 1. matmul_batch() selects different kernels (GEMM_table,
     * qgemm_d) than matmul() (fn_gemv) and sums in a different order. On this
     * model a 1e-6 relative perturbation of one projection already moves the
     * logits by ~8%, because the output distribution is nearly degenerate, so
     * compare argmax sequences, not logits. */
    static int no_batch = -1;
    if (no_batch < 0) no_batch = getenv("PICOLM_GLM5_NO_BATCH") ? 1 : 0;
    float *pf = s->glm5.prefill_buf;
    if (no_batch || !pf || n_tokens > s->glm5.prefill_n_tokens) {
        /* Per-token fallback via the GLM-5 per-token forward
         * (model_forward_glm5 is in this file, so the standalone
         * test_glm5_dsa probe can link). */
        for (int i = 0; i < n_tokens; i++) {
            if (interrupt && *interrupt) break;
            (void)model_forward_glm5(m, tokens[i], start_pos + i);
        }
        return s->logits;
    }
    {
        const size_t NT = (size_t)n_tokens;
        float *hc_x_b   = pf + s->glm5.pf_hc_x_off;    /* [N][S*D] */
        float *hc_res_b = pf + s->glm5.pf_hc_res_off;  /* [N][S*D] */
        float *cur_b    = pf + s->glm5.pf_cur_off;     /* [N][D] */
        float *attn_b   = pf + s->glm5.pf_attn_off;    /* [N][D] */
        float *post_b   = pf + s->glm5.pf_post_off;    /* [N][S] */
        float *comb_b   = pf + s->glm5.pf_comb_off;    /* [N][S*S] */
        float *q_b      = pf + s->glm5.pf_qkv_off;     /* [N][DI] */
        float *k_b      = q_b  + NT*DI;                /* [N][DI] */
        float *v_b      = q_b  + NT*2*DI;              /* [N][DI] */
        float *z_b      = pf + s->glm5.pf_z_off;       /* [N][DI] */
        float *raw_b    = pf + s->glm5.pf_raw_off;     /* [N][DI] */
        float *conv_b   = pf + s->glm5.pf_conv_off;    /* [N][3*DI] */
        float *head_b   = pf + s->glm5.pf_head_off;    /* [N][DI] */
        float *beta_b   = pf + s->glm5.pf_beta_off;    /* [N][H] */
        float *fa_b     = pf + s->glm5.pf_fa_off;      /* [N][KD] staging */
        float *gb_b     = pf + s->glm5.pf_gb_off;      /* [N][DI] = z output */
        float *fb_b     = pf + s->glm5.pf_fb_off;      /* [N][DI] = raw output */
        /* Shared per-token scratch for the sequential KDA tail (one token at a
         * time, so these can overlap the per-token kda_scratch layout):
         *   g     [DI] at kda_scratch + 8*DI + H   (the `g` slot)
         *   beta  [H]  at kda_scratch + 8*DI       (the `beta` slot)
         *   v_new [KD] at kda_scratch + 9*DI + H   (the `fa_out` slot) */
        float *g_scr   = s->glm5.kda_scratch + (size_t)8*DI + H;
        float *beta_sh = s->glm5.kda_scratch + (size_t)8*DI;
        float *v_new   = s->glm5.kda_scratch + (size_t)9*DI + H;

        /* ---- Embed all tokens, broadcast to S streams ---- */
        for (int t = 0; t < n_tokens; t++) {
            size_t row_bytes = gguf_type_row_size(w->type_token_embd, D);
            const void *embd_row = (const uint8_t *)w->token_embd
                                 + (size_t)tokens[t] * row_bytes;
            dequantize_row(embd_row, s->x, D, w->type_token_embd);
            for (int si = 0; si < S; si++)
                memcpy(hc_x_b + (size_t)t*S*D + si*D, s->x, D * sizeof(float));
        }

        for (int il = 0; il < n_trunk; il++) {
            const layer_weights_t *lw = &w->layers[il];
            const glm5_layer_weights_t *gw = &w->glm5[il];
            int mla_ord = c->glm5_layer_mla_ord[il];
            BENCH_LAYER_START();
            /* ============ Attention block ============ */

            /* Phase A: hc_pre(attn) + RMSNorm, per token. Saves hc_res,
             * post, comb per token for the hc_post in phase D. */
            for (int t = 0; t < n_tokens; t++) {
                float *hc_x_t = hc_x_b + (size_t)t*S*D;
                float *cur_t  = cur_b + (size_t)t*D;
                float *res_t  = hc_res_b + (size_t)t*S*D;
                float *post_t = post_b + (size_t)t*S;
                float *comb_t = comb_b + (size_t)t*S*S;
                memcpy(res_t, hc_x_t, (size_t)S*D*sizeof(float));
                glm5_hc_pre(m, gw, 1, hc_x_t, cur_t, post_t, comb_t, s->glm5.hc_scratch);
                rmsnorm(cur_t, cur_t, s->attn_norm_w[il], D, c->rms_norm_eps);
            }

            if (mla_ord >= 0) {
                /* MLA: batch the weight-bound projections across the token
                 * batch, then run the per-token attention core.
                 *
                 * Plane reuse: the qkv plane (3*DI = 24576 floats) already
                 * exists and is unused on MLA layers, so carve q (NH*MHD=16384),
                 * qr (QL=1536) and kv (KVL=512) out of it -- 18432 floats, fits.
                 * qabs (NH*KVL=32768) is per-token-only (attention consumes it
                 * immediately) so it lives in one shared scratch instead of an
                 * [n_tokens] plane: 64 MB at 512 tokens would dwarf the arena.
                 *
                 * The whole batch's latents must be in the cache BEFORE any
                 * attention runs, which is exactly what this ordering gives:
                 * glm5_mla_project_batch() fills kv_b, we store all of it, then
                 * the attention loop reads the cache. */
                {
                    const int NH = c->n_heads;
                    const int QL = c->glm5_q_lora;
                    const int KVL = c->glm5_kv_lora;
                    const int MHD = c->glm5_mla_head_dim;
                    const int QD = NH * MHD;
                    float *qr_plane = pf + s->glm5.pf_mla_qr_off;   /* [N][QL] */
                    float *q_plane  = pf + s->glm5.pf_mla_q_off;    /* [N][QD] */
                    float *o_plane  = pf + s->glm5.pf_mla_o_off;    /* [N][QD] */
                    float *kv_plane = pf + s->glm5.pf_mla_kv_off;   /* [N][KVL] */
                    /* qabs is per-token-only, so one shared [NH*KVL] buffer. It
                     * must be its own allocation: putting it in kda_scratch ran
                     * past the end of that buffer (8*DI + NH*KVL = 98304 vs the
                     * 82112 it owns) and into the KDA state. */
                    glm5_mla_project_batch(m, gw, cur_b, n_tokens,
                                           qr_plane, q_plane, kv_plane);

                    /* Store the whole batch's latents first (causality: token t
                     * must not see t+1, but must see 0..t). */
                    {
                        float *cache = s->glm5.mla_kv
                                     + (size_t)mla_ord * c->max_seq_len * KVL;
                        for (int t = 0; t < n_tokens; t++)
                            memcpy(cache + (size_t)(start_pos + t) * KVL,
                                   kv_plane + (size_t)t*KVL, KVL * sizeof(float));
                    }

                    /* qabs for the whole batch: qabs[t][h] = Wk_b[h] @ q_t[h].
                     * Not a persistent arena plane -- see the qabs note above;
                     * it is consumed immediately by the attention below. */
                    {
                        static float *qabs_b = NULL;
                        static size_t qabs_b_n = 0;
                        size_t need = (size_t)NT * NH * KVL;
                        if (qabs_b_n < need) {
                            free(qabs_b);
                            qabs_b = malloc(need * sizeof(float));
                            qabs_b_n = qabs_b ? need : 0;
                        }
                        if (qabs_b) {
                            /* Per-head [N][KVL] out of a batched [N][MHD] in:
                             * matmul_batch writes out[t*d + row], so its output
                             * stride would interleave heads. Run the 64 per-head
                             * GEMMs over the batch instead -- Wk_b is 16 MB in
                             * total and each head's slice is already resident. */
                            const uint8_t *kb = (const uint8_t *)gw->mla_k_b;
                            size_t k_row_sz = gguf_type_row_size(gw->type_mla_k_b, MHD);
                            size_t k_head_sz = (size_t)KVL * k_row_sz;
                            /* Per head, over ALL tokens: the head's weights
                             * (104 KB) are read once instead of once per token
                             * (that re-read was 1.6 GB/layer). matmul_batch
                             * can't be used directly -- it writes out[t*d+row]
                             * and we need [t][h] interleaving -- so the token
                             * loop stays outside and the activation rows for one
                             * head are gathered contiguously first. */
                            for (int h = 0; h < NH; h++) {
                                static float *qh_buf = NULL, *qh_dst = NULL;
                                static size_t qh_n = 0;
                                size_t qh_need = (size_t)NT * (MHD > KVL ? MHD : KVL);
                                if (qh_n < qh_need) {
                                    free(qh_buf); free(qh_dst);
                                    qh_buf = malloc(qh_need * sizeof(float));
                                    qh_dst = malloc(qh_need * sizeof(float));
                                    qh_n = (qh_buf && qh_dst) ? qh_need : 0;
                                }
                                for (int t = 0; t < n_tokens; t++)
                                    memcpy(qh_buf + (size_t)t*MHD,
                                           q_plane + (size_t)t*QD + (size_t)h*MHD,
                                           (size_t)MHD * sizeof(float));
                                /* One GEMM: [N][MHD] x Wk_b[h] -> [N][KVL],
                                 * then scatter into the [t][h] layout the
                                 * attention reads. */
                                matmul_batch(qh_dst, qh_buf, n_tokens,
                                             kb + h*k_head_sz, MHD, KVL,
                                             gw->type_mla_k_b);
                                for (int t = 0; t < n_tokens; t++)
                                    memcpy(qabs_b + ((size_t)t*NH + h)*KVL,
                                           qh_dst + (size_t)t*KVL,
                                           (size_t)KVL * sizeof(float));
                            }

                            /* Attention over the latent cache, all tokens at
                             * once. Only for the dense case: a DSA selection is
                             * per token, so it keeps the per-token core. */
                            int dense = 1;
                            if (c->glm5_use_dsa && c->glm5_idx_kpool > 0 &&
                                start_pos + n_tokens - 1 >= c->glm5_idx_kpool)
                                dense = 0;

                            static float *ctx_b = NULL;
                            static size_t ctx_b_n = 0;
                            if (!dense) ctx_b_n = 0;   /* forces per-token path */
                            if (dense) {
                                if (ctx_b_n < need) {
                                    free(ctx_b);
                                    ctx_b = malloc(need * sizeof(float));
                                    ctx_b_n = ctx_b ? need : 0;
                                }
                                if (ctx_b)
                                    glm5_mla_attn_batch(c, s, mla_ord, qabs_b, ctx_b,
                                                        n_tokens, start_pos);
                            }
                            if (dense && ctx_b) {
                                /* o[h] = Wv_b[h] @ ctx[h] per token */
                                /* Same shape as qabs, same reasoning: per
                                 * head over all tokens, so Wv_b's 104 KB per
                                 * head is read once per layer not once per
                                 * token. Gather ctx rows for the head, GEMM,
                                 * scatter back to [t][h]. */
                                const uint8_t *vb = (const uint8_t *)gw->mla_v_b;
                                size_t v_row_sz = gguf_type_row_size(gw->type_mla_v_b, KVL);
                                size_t v_head_sz = (size_t)MHD * v_row_sz;
                                for (int h = 0; h < NH; h++) {
                                    static float *vh_buf = NULL, *vh_dst = NULL;
                                    static size_t vh_n = 0;
                                    size_t vh_need = (size_t)NT * (MHD > KVL ? MHD : KVL);
                                    if (vh_n < vh_need) {
                                        free(vh_buf); free(vh_dst);
                                        vh_buf = malloc(vh_need * sizeof(float));
                                        vh_dst = malloc(vh_need * sizeof(float));
                                        vh_n = (vh_buf && vh_dst) ? vh_need : 0;
                                    }
                                    if (!vh_buf || !vh_dst) break;
                                    for (int t = 0; t < n_tokens; t++)
                                        memcpy(vh_buf + (size_t)t*KVL,
                                               ctx_b + ((size_t)t*NH + h)*KVL,
                                               (size_t)KVL * sizeof(float));
                                    matmul_batch(vh_dst, vh_buf, n_tokens,
                                                 vb + h*v_head_sz, KVL, MHD,
                                                 gw->type_mla_v_b);
                                    for (int t = 0; t < n_tokens; t++)
                                        memcpy(o_plane + (size_t)t*QD + (size_t)h*MHD,
                                               vh_dst + (size_t)t*MHD,
                                               (size_t)MHD * sizeof(float));
                                }
                            } else {
                                /* DSA / allocation failure: per-token core. */
                                for (int t = 0; t < n_tokens && (!interrupt || !*interrupt); t++) {
                                    glm5_mla_pre_t pre = {
                                        qr_plane + (size_t)t*QL,
                                        q_plane + (size_t)t*QD,
                                        qabs_b + (size_t)t*NH*KVL,
                                        kv_plane + (size_t)t*KVL,
                                    };
                                    glm5_mla_forward(m, gw, lw, s, cur_b + (size_t)t*D,
                                                     start_pos + t, c->max_seq_len,
                                                     il, mla_ord,
                                                     o_plane + (size_t)t*QD, &pre);
                                }
                            }
                        } else {
                            /* Could not allocate qabs: fall back to the
                             * per-token core, which uses its own scratch. */
                            for (int t = 0; t < n_tokens && (!interrupt || !*interrupt); t++)
                                glm5_mla_forward(m, gw, lw, s, cur_b + (size_t)t*D,
                                                 start_pos + t, c->max_seq_len,
                                                 il, mla_ord,
                                                 o_plane + (size_t)t*QD, NULL);
                        }
                    }
                    /* Batched attention output projection: [N][NH*MHD] -> [N][D]. */
                    matmul_batch(attn_b, o_plane, n_tokens, lw->attn_output,
                                 QD, D, lw->type_attn_output);
                }
            } else {
                /* KDA: batch the weight-bound projections, then run the
                 * sequential tail per token. The projections read cur_b and
                 * write into per-token planes; the tail reads those planes
                 * and updates the recurrent state in token order. */
                matmul_batch(q_b, cur_b, n_tokens, lw->attn_q, D, DI, lw->type_attn_q);
                matmul_batch(k_b, cur_b, n_tokens, lw->attn_k, D, DI, lw->type_attn_k);
                matmul_batch(v_b, cur_b, n_tokens, lw->attn_v, D, DI, lw->type_attn_v);
                /* gate: z = g_b @ (g_a @ cur) */
                matmul_batch(fa_b, cur_b, n_tokens, gw->kda_g_a, D, KD, gw->type_kda_g_a);
                matmul_batch(gb_b, fa_b, n_tokens, gw->kda_g_b, KD, DI, gw->type_kda_g_b);
                /* decay: raw = f_b @ (f_a @ cur) -- fa_b reused after g_b */
                matmul_batch(fa_b, cur_b, n_tokens, gw->kda_f_a, D, KD, gw->type_kda_f_a);
                matmul_batch(fb_b, fa_b, n_tokens, gw->kda_f_b, KD, DI, gw->type_kda_f_b);
                /* beta: [N][H] */
                matmul_batch(beta_b, cur_b, n_tokens, gw->kda_beta, D, H, gw->type_kda_beta);

                /* Sequential tail: conv -> recurrence -> norm -> gate. The
                 * tail's output projection is deferred so attn_output can be
                 * GEMM'd for all tokens at once below. */
                for (int t = 0; t < n_tokens; t++) {
                    glm5_kda_tail(m, gw, lw, s, il, start_pos + t,
                                  q_b + (size_t)t*DI,
                                  k_b + (size_t)t*DI,
                                  v_b + (size_t)t*DI,
                                  gb_b + (size_t)t*DI,   /* z */
                                  fb_b + (size_t)t*DI,   /* raw */
                                  beta_b + (size_t)t*H,
                                  conv_b + (size_t)t*3*DI,  /* conv scratch */
                                  g_scr,                    /* g [DI] shared */
                                  head_b + (size_t)t*DI,
                                  v_new, beta_sh,
                                  attn_b + (size_t)t*D,
                                  0);                       /* defer out proj */
                }
                /* Batched attention output projection: [N][DI] -> [N][D]. */
                matmul_batch(attn_b, head_b, n_tokens, lw->attn_output, DI, D,
                             lw->type_attn_output);
            }

            /* Phase D: hc_post(attn) per token */
            for (int t = 0; t < n_tokens; t++) {
                float *hc_x_t = hc_x_b + (size_t)t*S*D;
                float *res_t  = hc_res_b + (size_t)t*S*D;
                float *post_t = post_b + (size_t)t*S;
                float *comb_t = comb_b + (size_t)t*S*S;
                float *attn_t = attn_b + (size_t)t*D;
                glm5_hc_post(m, attn_t, res_t, post_t, comb_t, hc_x_t);
            }

            /* ============ FFN block ============ */

            /* Phase A': hc_pre(ffn) + RMSNorm per token (reuses post/comb planes) */
            for (int t = 0; t < n_tokens; t++) {
                float *hc_x_t = hc_x_b + (size_t)t*S*D;
                float *cur_t  = cur_b + (size_t)t*D;
                float *res_t  = hc_res_b + (size_t)t*S*D;
                float *post_t = post_b + (size_t)t*S;
                float *comb_t = comb_b + (size_t)t*S*S;
                memcpy(res_t, hc_x_t, (size_t)S*D*sizeof(float));
                glm5_hc_pre(m, gw, 0, hc_x_t, cur_t, post_t, comb_t, s->glm5.hc_scratch);
                rmsnorm(cur_t, cur_t, s->post_attn_norm_w[il], D, c->rms_norm_eps);
            }

            /* Phase B': FFN / MoE.
             *
             * Dense layers stay per token (only 3 of them, and n_ffn=12288 is
             * a single big GEMM per token anyway). MoE prefill goes through
             * the expert-sorted batched path: at 110 tokens essentially all
             * 288 experts are selected, so the per-token loop streamed each
             * expert's weights ~3x. Routing stays per token; only the expert
             * loop is restructured. */
            /* NOTE: there is an unexplained long-generation divergence that is
             * NOT caused by this batching -- see the TODO above
             * glm5_moe_ffn_batch(). */
            if (il < c->n_layer_dense_lead) {
                for (int t = 0; t < n_tokens; t++)
                    glm5_dense_ffn(m, lw, s, cur_b + (size_t)t*D, attn_b + (size_t)t*D);
            } else if (n_tokens > 1 && !getenv("PICOLM_GLM5_MOE_PERTOKEN")) {
                glm5_moe_ffn_batch(m, gw, lw, s, cur_b, attn_b, n_tokens);
            } else {
                for (int t = 0; t < n_tokens; t++)
                    glm5_moe_ffn(m, gw, lw, s, cur_b + (size_t)t*D, attn_b + (size_t)t*D);
            }

            /* Phase D': hc_post(ffn) per token */
            for (int t = 0; t < n_tokens; t++) {
                float *hc_x_t = hc_x_b + (size_t)t*S*D;
                float *res_t  = hc_res_b + (size_t)t*S*D;
                float *post_t = post_b + (size_t)t*S;
                float *comb_t = comb_b + (size_t)t*S*S;
                float *attn_t = attn_b + (size_t)t*D;
                glm5_hc_post(m, attn_t, res_t, post_t, comb_t, hc_x_t);
            }
#ifdef PICOLM_VIZ
            {
                /* Mean-reduce S streams -> single vector per token, then push batch. */
                float *reduced = s->glm5.cur;  /* scratch [D] */
                /* attn_b is [N*D] and no longer needed after hc_post for this layer */
                float *reduced_batch = attn_b;
                float inv_s = 1.0f / (float)S;
                for (int t = 0; t < n_tokens; t++) {
                    float *hc_x_t = hc_x_b + (size_t)t*S*D;
                    float *rb = reduced_batch + (size_t)t*D;
                    for (int d = 0; d < D; d++) {
                        float acc = 0.f;
                        for (int si = 0; si < S; si++) acc += hc_x_t[si*D + d];
                        rb[d] = acc * inv_s;
                    }
                }
                viz_push_batch(il, start_pos, reduced_batch, n_tokens, D);
            }
#endif
            if (stop_layer >= 0 && il >= stop_layer) {
                BENCH_LAYER_END(il, 1);
                if (glm5_dbg_enabled())
                    fprintf(stderr, "GLM5 prefill: stopping after layer %d (PICOLM_STOP_LAYER)\n", il);
                break;
            }
            BENCH_LAYER_END(il, 1);
        }

        /* ---- Final: mean over S streams -> RMSNorm -> output head ---- */
        {
            float *hc_x_last = hc_x_b + (size_t)(n_tokens-1)*S*D;
            float inv_s = 1.0f / (float)S;
            for (int d = 0; d < D; d++) {
                float acc = 0.f;
                for (int si = 0; si < S; si++) acc += hc_x_last[si*D + d];
                s->glm5.cur[d] = acc * inv_s;
            }
            float ss = 0.f;
            for (int i = 0; i < D; i++) ss += s->glm5.cur[i] * s->glm5.cur[i];
            float inv = 1.0f / sqrtf(ss / D + c->rms_norm_eps);
            const float *nw = s->output_norm_w;
            for (int i = 0; i < D; i++) s->glm5.cur[i] *= inv * nw[i];
            matmul(s->logits, s->glm5.cur, w->output, D, c->vocab_size, w->type_output);
        }
        return s->logits;
    }

}

/* ------------------------------------------------------------------ */
/* KV-cache state snapshot (v5): MLA latent caches + KDA states        */
/* ------------------------------------------------------------------ */

/* Blob layout (all little-endian float32):
 *   [MLA latent caches, one per MLA trunk layer, in MLA-ordinal order]
 *       each: n_pos rows of kv_lora floats (stride in memory: max_seq_len)
 *   [KDA conv states, ascending layer index]
 *       each: (d_conv-1) * conv_dim floats
 *   [KDA recurrent states, ascending layer index]
 *       each: n_kda_heads * kda_dim * kda_dim floats
 *
 * Note the two different orderings: MLA caches are addressed by their
 * compact ordinal (glm5_layer_mla_ord), while KDA state pointers are indexed
 * by raw layer number, matching how each is allocated in model_core.c.
 * Layers with no MLA ordinal (-1: the NextN/MTP tail) and layers with no KDA
 * state are skipped, which is what keeps the blob size in sync with
 * model_glm5_snapshot_size().
 *
 * Only n_pos rows of each MLA latent are saved; positions >= n_pos still hold
 * whatever was there before the restore and must not be read, so the caller
 * must resume at n_pos. That is exactly the rewind contract used by the SSM
 * snapshot path. */
/* KDA-only recurrent-state snapshot (used by the generic rewind/checkpoint
 * layer). Deliberately excludes the MLA latent caches: those are positional
 * and rewinding leaves rows 0..N-1 valid, so snapshotting them would make
 * every checkpoint grow with context for no benefit. See model.h.
 *
 * Layout: [conv state for each KDA layer][recurrent state for each KDA layer],
 * both in ascending trunk-layer order, skipping MLA/NextN layers (which own no
 * KDA state). Size is position-independent. */
static int glm5_kda_conv_dim(const model_config_t *c) {
    return 3 * c->glm5_n_kda_heads * c->glm5_kda_dim;
}

size_t model_glm5_kda_state_size(const model_t *m) {
    const model_config_t *c = &m->config;
    if (!c->is_glm5 || c->glm5_n_kda <= 0) return 0;
    const int conv_dim = glm5_kda_conv_dim(c);
    const int kda_dim = c->glm5_kda_dim;
    return (size_t)c->glm5_n_kda * (c->ssm_d_conv - 1) * conv_dim * sizeof(float)
         + (size_t)c->glm5_n_kda * c->glm5_n_kda_heads * kda_dim * kda_dim * sizeof(float);
}

size_t model_glm5_kda_state_save(const model_t *m, uint8_t *buf, size_t buf_size) {
    const model_config_t *c = &m->config;
    const run_state_t *s = &m->state;
    if (!c->is_glm5) return 0;
    size_t needed = model_glm5_kda_state_size(m);
    if (needed == 0 || buf_size < needed) return 0;
    const int conv_dim = glm5_kda_conv_dim(c);
    const int kda_dim = c->glm5_kda_dim;
    const size_t csz = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
    const size_t ssz = (size_t)c->glm5_n_kda_heads * kda_dim * kda_dim * sizeof(float);
    size_t off = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (!s->glm5.kda_conv_state[l]) continue;
        memcpy(buf + off, s->glm5.kda_conv_state[l], csz);
        off += csz;
    }
    for (int l = 0; l < c->n_layers; l++) {
        if (!s->glm5.kda_state[l] || c->glm5_layer_is_mla[l]) continue;
        memcpy(buf + off, s->glm5.kda_state[l], ssz);
        off += ssz;
    }
    return off;
}

size_t model_glm5_kda_state_restore(model_t *m, const uint8_t *buf, size_t buf_size) {
    const model_config_t *c = &m->config;
    run_state_t *s = &m->state;
    if (!c->is_glm5) return 0;
    size_t needed = model_glm5_kda_state_size(m);
    if (needed == 0 || buf_size < needed) return 0;
    const int conv_dim = glm5_kda_conv_dim(c);
    const int kda_dim = c->glm5_kda_dim;
    const size_t csz = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
    const size_t ssz = (size_t)c->glm5_n_kda_heads * kda_dim * kda_dim * sizeof(float);
    size_t off = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (!s->glm5.kda_conv_state[l]) continue;
        memcpy(s->glm5.kda_conv_state[l], buf + off, csz);
        off += csz;
    }
    for (int l = 0; l < c->n_layers; l++) {
        if (!s->glm5.kda_state[l] || c->glm5_layer_is_mla[l]) continue;
        memcpy(s->glm5.kda_state[l], buf + off, ssz);
        off += ssz;
    }
    return off;
}

size_t model_glm5_snapshot_size(const model_t *m, int n_pos) {
    const model_config_t *c = &m->config;
    if (!c->is_glm5) return 0;
    int kda_dim = c->glm5_kda_dim;
    int d_inner = c->glm5_n_kda_heads * kda_dim;
    int conv_dim = 3 * d_inner;
    size_t sz = 0;
    sz += (size_t)c->glm5_n_mla * n_pos * c->glm5_kv_lora * sizeof(float);
    sz += (size_t)c->glm5_n_kda * (c->ssm_d_conv - 1) * conv_dim * sizeof(float);
    sz += (size_t)c->glm5_n_kda * c->glm5_n_kda_heads * kda_dim * kda_dim * sizeof(float);
    return sz;
}

size_t model_glm5_state_save(const model_t *m, uint8_t *buf, size_t buf_size, int n_pos) {
    const model_config_t *c = &m->config;
    const run_state_t *s = &m->state;
    if (!c->is_glm5) return 0;
    size_t needed = model_glm5_snapshot_size(m, n_pos);
    if (buf_size < needed) {
        fprintf(stderr, "kvcache_save: glm5 state buffer too small (%zu < %zu)\n", buf_size, needed);
        return 0;
    }
    size_t off = 0;
    int kda_dim = c->glm5_kda_dim;
    int d_inner = c->glm5_n_kda_heads * kda_dim;
    int conv_dim = 3 * d_inner;
    /* MLA latent caches (n_pos rows of each). Gate on the ordinal, not just
     * is_mla: the NextN/MTP tail layer is MLA-shaped but deliberately has
     * ord -1 and no cache slot (see the ordinal assignment in model_gguf.c),
     * so including it would read out of bounds and desync the blob layout
     * from model_glm5_snapshot_size(), which counts only n_mla. */
    for (int l = 0; l < c->n_layers; l++) {
        int ord = c->glm5_layer_mla_ord[l];
        if (ord < 0) continue;
        const float *cache = s->glm5.mla_kv + (size_t)ord * c->max_seq_len * c->glm5_kv_lora;
        size_t sz = (size_t)n_pos * c->glm5_kv_lora * sizeof(float);
        memcpy(buf + off, cache, sz);
        off += sz;
    }
    /* KDA conv + recurrent states. The NULL-state checks cover both the MLA
     * trunk layers and the NextN tail, which own no KDA state. */
    for (int l = 0; l < c->n_layers; l++) {
        if (!s->glm5.kda_conv_state[l] || !s->glm5.kda_state[l]) continue;
        size_t csz = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
        memcpy(buf + off, s->glm5.kda_conv_state[l], csz);
        off += csz;
    }
    for (int l = 0; l < c->n_layers; l++) {
        if (c->glm5_layer_is_mla[l]) continue;
        if (!s->glm5.kda_state[l]) continue;
        size_t ssz = (size_t)c->glm5_n_kda_heads * kda_dim * kda_dim * sizeof(float);
        memcpy(buf + off, s->glm5.kda_state[l], ssz);
        off += ssz;
    }
    return off;
}

size_t model_glm5_state_restore(model_t *m, const uint8_t *buf, size_t buf_size, int n_pos) {
    const model_config_t *c = &m->config;
    run_state_t *s = &m->state;
    if (!c->is_glm5) return 0;
    size_t needed = model_glm5_snapshot_size(m, n_pos);
    if (buf_size < needed) {
        fprintf(stderr, "kvcache_load: glm5 state buffer too small (%zu < %zu)\n", buf_size, needed);
        return 0;
    }
    size_t off = 0;
    int kda_dim = c->glm5_kda_dim;
    int d_inner = c->glm5_n_kda_heads * kda_dim;
    int conv_dim = 3 * d_inner;
    for (int l = 0; l < c->n_layers; l++) {
        int ord = c->glm5_layer_mla_ord[l];
        if (ord < 0) continue;
        float *cache = s->glm5.mla_kv + (size_t)ord * c->max_seq_len * c->glm5_kv_lora;
        size_t sz = (size_t)n_pos * c->glm5_kv_lora * sizeof(float);
        memcpy(cache, buf + off, sz);
        off += sz;
    }
    for (int l = 0; l < c->n_layers; l++) {
        if (c->glm5_layer_is_mla[l]) continue;
        if (!s->glm5.kda_conv_state[l] || !s->glm5.kda_state[l]) continue;
        size_t csz = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
        memcpy(s->glm5.kda_conv_state[l], buf + off, csz);
        off += csz;
    }
    for (int l = 0; l < c->n_layers; l++) {
        if (c->glm5_layer_is_mla[l]) continue;
        if (!s->glm5.kda_state[l]) continue;
        size_t ssz = (size_t)c->glm5_n_kda_heads * kda_dim * kda_dim * sizeof(float);
        memcpy(s->glm5.kda_state[l], buf + off, ssz);
        off += ssz;
    }
    return off;
}
