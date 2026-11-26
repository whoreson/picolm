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

/* PICOLM_DBG_MOE_OUT=1: per-expert SwiGLU diagnostics when any gate/up
 * value exceeds ±5, and routed-output outlier reports (see glm5_moe_ffn). */
static int moe_dbg_on = -1;

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
#ifdef CLOCK_MONOTONIC
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
#else
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#endif
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
    {   /* PICOLM_POOL_GAPS=1: where the main thread's time goes around the pool */
        double gap, mn, wt, wk; long n;
        const int T = g_gp.tokens - 1;   /* first token was dropped by the reset */
        if (T > 0 && tensor_pool_gap_take(&gap, &mn, &wt, &wk, &n)) {
            const double sum = gap + mn + wt;
            fprintf(stderr, "  --- pool timing (PICOLM_POOL_GAPS), per token over %d tokens ---\n", T);
            fprintf(stderr, "  dispatches            %10.0f\n", (double)n / T);
            fprintf(stderr, "  serial gap (workers idle, main busy)  %8.2f ms  %5.1f%%\n", gap / T, 100.0 * gap / sum);
            fprintf(stderr, "  main's own slot                       %8.2f ms  %5.1f%%\n", mn / T, 100.0 * mn / sum);
            fprintf(stderr, "  waiting for stragglers (barrier skew) %8.2f ms  %5.1f%%\n", wt / T, 100.0 * wt / sum);
            fprintf(stderr, "  mean per dispatch: gap %.1f us, skew %.1f us, wake latency %.2f us\n",
                    gap / n * 1e3, wt / n * 1e3, wk);
        }
    }
}

/* SwiGLU clamp value actually applied.
 *
 * The GGUF carries swiglu_clamp_exp/shexp = 10.0 for every layer.  The ik
 * glm5next graph builder parses those keys but never applies them, so the
 * oracle runs UNCLAMPED -- which is why this used to default to off.
 *
 * Measurement says the clamp is the correct behaviour, though:
 *   - layer 10 shared expert rms    unclamped 0.211491 / clamped 0.341236
 *     (ik 0.278092; clamped overshoots but the TOTAL MoE output lands on
 *      ik: 0.343310 vs ik 0.346686, where unclamped gives 0.214758)
 *   - 21-token "capital of France" logits vs the ik oracle:
 *     cosine 0.9437 unclamped -> 0.9571 clamped, rms diff 0.645 -> 0.573.
 * So the clamp is ON by default now.
 *
 *   PICOLM_GLM5_NOCLAMP=1   -> force it off (reproduces the old behaviour)
 *   PICOLM_GLM5_CLAMP=1     -> force it on (now redundant, kept for scripts)
 */
static float glm5_clamp_value(const model_config_t *c) {
    static int mode = -1;   /* -1 unknown, 0 off, 1 on */
    if (mode < 0) {
        mode = getenv("PICOLM_GLM5_NOCLAMP") ? 0 : 1;
        if (getenv("PICOLM_GLM5_EXPCLAMP")) mode = atoi(getenv("PICOLM_GLM5_EXPCLAMP"));
    }
    if (!mode) return 0.0f;   /* 0 == disabled */
    return c->glm5_clamp_exp;
}

/* Same, for the shared expert.  The GGUF carries a SEPARATE
 * swiglu_clamp_shexp array (10.0 for every layer, same value as _exp in
 * practice, but it is a distinct key and gating the shared expert on the
 * routed-expert limit would be wrong the moment they differ). */
static float glm5_clamp_value_shexp(const model_config_t *c) {
    static int mode = -1;
    if (mode < 0) {
        mode = getenv("PICOLM_GLM5_NOCLAMP") ? 0 : 1;
    }
    if (!mode) return 0.0f;
    return c->glm5_clamp_shexp;
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

static int glm5_stats_mode(const char **substr);
static int glm5_stats_wanted(void) { const char *p; return glm5_stats_mode(&p); }

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
static float hc_pre_dbg_pre[8];

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
    for (int i = 0; i < S && i < 8; i++) hc_pre_dbg_pre[i] = pre[i];

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

/* Head-parallel KDA core (defined after glm5_kda_tail). Same contract and
 * bit-identical results; see the comment there. */
static void glm5_kda_tail_par(
    const model_t *m, const glm5_layer_weights_t *gw, const layer_weights_t *lw,
    run_state_t *s, int layer_idx, int pos,
    const float *q, const float *k, const float *v,
    const float *z, const float *raw, const float *beta_raw,
    float *conv_out, float *g, float *head_out,
    float *v_new, float *beta,
    float *out, int do_out_proj);

/* The head-parallel path skips the per-stage stats/range diagnostics, so it is
 * used only when none of them is requested. PICOLM_GLM5_KDA_SERIAL=1 forces the
 * original serial code (A/B timing, bisection). */
static int glm5_kda_par_ok(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *pat;
        cached = !(getenv("PICOLM_GLM5_KDA_SERIAL") || glm5_dbg2_enabled() ||
                   glm5_stats_mode(&pat) || glm5_strict_mode());
    }
    return cached;
}

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
    float *beta_raw_buf = sc; sc += H;  /* ssm_beta @ cur, batched with g_a/f_a */
    float *fa_out2 = sc; sc += KD;   /* second low-rank scratch: g_a and f_a are
                                      * issued in one batched dispatch, so each
                                      * needs its own destination */
    float *head_out = sc;   /* [DI] */

    /* Tag prefix shared by every stat/check emitted from this layer. */
    char tg[48];
    snprintf(tg, sizeof(tg), "l%02d.kda", layer_idx);
    char t[96];

    float *conv_state = s->glm5.kda_conv_state[layer_idx];
    float *kda_state = s->glm5.kda_state[layer_idx];

    glm5_stats("x.kda.in", cur, D);

    /* q, k, v projections — independent, same input, so one dispatch. */
    {
        mm_job_t j[3] = {
            { q, cur, lw->attn_q, D, DI, lw->type_attn_q },
            { k, cur, lw->attn_k, D, DI, lw->type_attn_k },
            { v, cur, lw->attn_v, D, DI, lw->type_attn_v },
        };
        if (!tensor_matmul_multi(j, 3)) {
            for (int i = 0; i < 3; i++)
                matmul(j[i].out, j[i].x, j[i].W, j[i].n, j[i].d, j[i].qtype);
        }
    }
    snprintf(t, sizeof(t), "%s.q", tg);   glm5_stats(t, q, DI);   glm5_chk(t, q, DI, -1e4f, 1e4f);
    snprintf(t, sizeof(t), "%s.k", tg);   glm5_stats(t, k, DI);   glm5_chk(t, k, DI, -1e4f, 1e4f);
    snprintf(t, sizeof(t), "%s.v", tg);   glm5_stats(t, v, DI);   glm5_chk(t, v, DI, -1e4f, 1e4f);
    snprintf(t, sizeof(t), "%s.dt_bias", tg);
    glm5_stats(t, (const float *)gw->kda_dt_bias, DI);
    snprintf(t, sizeof(t), "%s.ssm_a", tg);
    glm5_stats(t, (const float *)gw->kda_a, H);

    /* Low-rank inputs all read `cur`, so g_a, f_a and beta share one
     * dispatch; each writes its own destination. */
    {
        float *beta_raw_p = beta_raw_buf;
        mm_job_t j[3] = {
            { fa_out,  cur, gw->kda_g_a, D, KD, gw->type_kda_g_a },
            { fa_out2, cur, gw->kda_f_a, D, KD, gw->type_kda_f_a },
            { beta_raw_p, cur, gw->kda_beta, D, H, gw->type_kda_beta },
        };
        if (!tensor_matmul_multi(j, 3)) {
            for (int i = 0; i < 3; i++)
                matmul(j[i].out, j[i].x, j[i].W, j[i].n, j[i].d, j[i].qtype);
        }
    }

    /* z = g_b @ (g_a @ cur): gate, [KD] -> [DI] */
    matmul(z, fa_out, gw->kda_g_b, KD, DI, gw->type_kda_g_b);

    /* raw = f_b @ (f_a @ cur): decay raw input, [KD] -> [DI] */
    matmul(raw, fa_out2, gw->kda_f_b, KD, DI, gw->type_kda_f_b);
    snprintf(t, sizeof(t), "%s.z", tg);   glm5_stats(t, z, DI);   glm5_chk(t, z, DI, -1e4f, 1e4f);
    snprintf(t, sizeof(t), "%s.raw", tg); glm5_stats(t, raw, DI); glm5_chk(t, raw, DI, -1e4f, 1e4f);

    /* Everything from here to the output projection is head-local (g, conv,
     * L2 norm, state update, per-head RMSNorm, gate), so it runs one head per
     * task across the pool instead of serially on the main thread while the
     * other workers spin. */
    if (glm5_kda_par_ok()) {
        glm5_kda_tail_par(m, gw, lw, s, layer_idx, pos, q, k, v, z, raw,
                          beta_raw_buf, conv_out, g, head_out, NULL, beta,
                          out, 1);
        snprintf(t, sizeof(t), "%s.out", tg);
        glm5_chk(t, out, D, -1e4f, 1e4f);
        return;
    }

    /* beta = sigmoid(beta_raw); beta_raw was computed in the batched dispatch
     * above (it only depends on `cur`). */
    {
        for (int h = 0; h < H; h++)
            beta[h] = 1.0f / (1.0f + expf(-beta_raw_buf[h]));
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
    if (getenv("PICOLM_DBG_CONVXY") && atoi(getenv("PICOLM_DBG_CONVXY")) == layer_idx) {
        const char *dd = getenv("PICOLM_DUMP_DIR");
        char fn[512];
        snprintf(fn, sizeof(fn), "%s/conv_output_silu-%d.f32", dd ? dd : "/tmp/picd", layer_idx);
        FILE *fp = fopen(fn, "wb");
        if (fp) { fwrite(conv_out, sizeof(float), (size_t)CONV_DIM, fp); fclose(fp); }
    }
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

    /* Per-head RMSNorm with ssm_norm weight, then sigmoid gate (z).
     *
     * The gate is PLAIN SIGMOID, measured rather than assumed:
     * llama-delta-net.h declares the KDA output gate_op as SILU and
     * build_glm5next.cpp does not override it, but fitting ik's own dumped
     * attn_rms_norm-0 / z-0 / attn_out_norm-0 gives
     *     attn_out_norm == attn_rms_norm * sigmoid(z)   relrms 0.000000
     *     attn_out_norm == attn_rms_norm * silu(z)      relrms 1.2138
     * so sigmoid is what actually runs.  Switching this to SiLU drops the
     * 21-token "capital of France" prompt from cosine 0.94 to 0.20 against
     * the ik oracle.  ik's llama-kda.cpp:91 spells it ggml_sigmoid too. */
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

#if defined(PICOLM_AVX512)
/* expf for 16 lanes: Cephes range reduction + degree-6 polynomial, scaled with vscalefps so that
 * overflow -> +inf and underflow -> 0 like libm. Max error ~2 ULP against glibc's expf, i.e. NOT
 * bit-identical; that is why the vector pre-pass is opt-in (PICOLM_GLM5_KDA_VEXP=1). */
static inline __m512 kda_expf16(__m512 x)
{
    /* expf(89) overflows to +inf and expf(-104) underflows to 0 in float, so clamping first keeps the
     * range reduction sane for huge |x| without changing those limits. */
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-104.0f)), _mm512_set1_ps(89.0f));
    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 c1 = _mm512_set1_ps(0.693359375f), c2 = _mm512_set1_ps(-2.12194440e-4f);
    const __m512 fx = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(fx, c1, x);
    r = _mm512_fnmadd_ps(fx, c2, r);
    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.0000001201E-1f));
    const __m512 r2 = _mm512_mul_ps(r, r);
    y = _mm512_fmadd_ps(y, r2, _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
    return _mm512_scalef_ps(y, fx);
}
#endif

/* ---- Blocked flash-style prefill attention (absorbed MLA, dense) ------------
 *
 * mla_attn_task() above is GEMV-shaped: for every (head, token) it streams the
 * whole F32 latent cache twice with 512-wide dots, so at 10k context a layer
 * moves ~650 GB of cache through L3 and runs at ~2.5 GFLOP/s per thread (the
 * "2.2 GFLOP/s" in its header comment). The comment claims the loop inversion
 * lets a cache row serve every token of a chunk, but the code is
 * `for t { for p { dot } }`, so the cache is re-read per token.
 *
 * This is the same math as a GEMM + online softmax (the structure ik_llama.cpp
 * uses): a task owns PFA_BQ tokens x PFA_HG heads = up to 64 query rows and walks
 * the keys in blocks of PFA_BK. Per block
 *     S[r][j] = scale * dot(q_r, k_j)        4 rows x 4 keys FMA register tile
 *     online softmax per row (running max m, sum l, rescale of O)
 *     O[r][:] += P[r][j] * k_j[:]             4 rows x 64 cols kept in zmm
 * so each key block (64 rows x 2 KB = 128 KB, L2-resident) is loaded once per 64
 * query rows instead of once per query: ~64x less cache traffic and the inner
 * loops are FMA-bound instead of load-bound.
 *
 * Causal mask: row r (token t) sees keys <= start_pos + t; the diagonal block is
 * masked per row. Not bit-identical to mla_attn_task (online softmax, no
 * p < 1e-10 skip); accurate to float rounding. Opt-in: PICOLM_GLM5_PFA=1.
 * Scratch is on the stack (no TLS: the MSVC build has none). */
#define PFA_HG    8
#define PFA_BQ    8
#define PFA_BK    64
#define PFA_MAXR  (PFA_HG * PFA_BQ)
#define PFA_MAXKVL 512

/* Default ON. PICOLM_GLM5_PFA=0 disables it and gets the older GEMV-shaped
 * mla_attn_task path back (useful as an A/B reference: PFA is not bit-identical
 * to it, see the note above). */
static int glm5_pfa_enabled(void) {
    static int cached = -1;
    if (cached < 0) { const char *e = getenv("PICOLM_GLM5_PFA"); cached = !(e && atoi(e) == 0); }
    return cached;
}

/* PICOLM_GLM5_PFA_VEXP=0: disable vectorized PFA softmax (shuffle-tree reduce + vec expf).
 * ON by default on AVX-512; not bit-identical to the scalar path (summation order, expf rounding). */
static int glm5_pfa_vexp_enabled(void) {
    static int cached = -1;
#if defined(PICOLM_AVX512)
    if (cached < 0) { const char *e = getenv("PICOLM_GLM5_PFA_VEXP"); cached = !(e && atoi(e) == 0); }
#else
    cached = 0;
#endif
    return cached;
}

typedef struct {
    const float *qabs, *cache; float *ctx;
    int n_tokens, start_pos, NH, KVL, nTB, perm, vx;   /* vx: AVX-512 vector softmax + tree-reduced scores */
    float scale;
} glm5_pfa_t;

#if defined(PICOLM_AVX512)
/* Lane k of the result = horizontal sum of a[k]. 24 unpack/add + 12 shuffle/add + 16 cross-lane
 * instructions instead of 16 x ~9 for _mm512_reduce_add_ps. */
static inline __m512 glm5_pfa_hsum16(const __m512 a[16])
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

/* S[r][j] = scale * dot(q[r], kb[j]) for r<R, j<nk. kb rows are KVL floats apart. */
static void glm5_pfa_scores(const float *const *qp, int R, const float *kb, int nk,
                            int KVL, float scale, float *S, int vx)
{
    int r = 0;
#if defined(PICOLM_AVX512)
    if ((KVL & 15) == 0) {
        for (; r + 3 < R; r += 4) {
            int j = 0;
            for (; j + 3 < nk; j += 4) {
                __m512 a00 = _mm512_setzero_ps(), a01 = a00, a02 = a00, a03 = a00,
                       a10 = a00, a11 = a00, a12 = a00, a13 = a00,
                       a20 = a00, a21 = a00, a22 = a00, a23 = a00,
                       a30 = a00, a31 = a00, a32 = a00, a33 = a00;
                const float *k0 = kb + (size_t)(j + 0) * KVL, *k1 = kb + (size_t)(j + 1) * KVL,
                            *k2 = kb + (size_t)(j + 2) * KVL, *k3 = kb + (size_t)(j + 3) * KVL;
                const float *q0 = qp[r], *q1 = qp[r + 1], *q2 = qp[r + 2], *q3 = qp[r + 3];
                for (int d = 0; d < KVL; d += 16) {
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
                if (vx) {
                    const __m512 acc[16] = { a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33 };
                    const __m512 sum = _mm512_mul_ps(glm5_pfa_hsum16(acc), _mm512_set1_ps(scale));
                    _mm_storeu_ps(S + (size_t)(r + 0) * PFA_BK + j, _mm512_extractf32x4_ps(sum, 0));
                    _mm_storeu_ps(S + (size_t)(r + 1) * PFA_BK + j, _mm512_extractf32x4_ps(sum, 1));
                    _mm_storeu_ps(S + (size_t)(r + 2) * PFA_BK + j, _mm512_extractf32x4_ps(sum, 2));
                    _mm_storeu_ps(S + (size_t)(r + 3) * PFA_BK + j, _mm512_extractf32x4_ps(sum, 3));
                    continue;
                }
                float *s0 = S + (size_t)(r + 0) * PFA_BK + j, *s1 = S + (size_t)(r + 1) * PFA_BK + j;
                float *s2 = S + (size_t)(r + 2) * PFA_BK + j, *s3 = S + (size_t)(r + 3) * PFA_BK + j;
                s0[0] = _mm512_reduce_add_ps(a00) * scale; s0[1] = _mm512_reduce_add_ps(a01) * scale;
                s0[2] = _mm512_reduce_add_ps(a02) * scale; s0[3] = _mm512_reduce_add_ps(a03) * scale;
                s1[0] = _mm512_reduce_add_ps(a10) * scale; s1[1] = _mm512_reduce_add_ps(a11) * scale;
                s1[2] = _mm512_reduce_add_ps(a12) * scale; s1[3] = _mm512_reduce_add_ps(a13) * scale;
                s2[0] = _mm512_reduce_add_ps(a20) * scale; s2[1] = _mm512_reduce_add_ps(a21) * scale;
                s2[2] = _mm512_reduce_add_ps(a22) * scale; s2[3] = _mm512_reduce_add_ps(a23) * scale;
                s3[0] = _mm512_reduce_add_ps(a30) * scale; s3[1] = _mm512_reduce_add_ps(a31) * scale;
                s3[2] = _mm512_reduce_add_ps(a32) * scale; s3[3] = _mm512_reduce_add_ps(a33) * scale;
            }
            for (; j < nk; j++)
                for (int i = 0; i < 4; i++)
                    S[(size_t)(r + i) * PFA_BK + j] =
                        vec_dot_f32_f32(kb + (size_t)j * KVL, qp[r + i], KVL) * scale;
        }
    }
#elif defined(PICOLM_AVX2)
    if ((KVL & 7) == 0) {
        for (; r + 3 < R; r += 4) {
            int j = 0;
            for (; j + 3 < nk; j += 4) {
                __m256 a00 = _mm256_setzero_ps(), a01 = a00, a02 = a00, a03 = a00,
                       a10 = a00, a11 = a00, a12 = a00, a13 = a00,
                       a20 = a00, a21 = a00, a22 = a00, a23 = a00,
                       a30 = a00, a31 = a00, a32 = a00, a33 = a00;
                const float *k0 = kb + (size_t)(j + 0) * KVL, *k1 = kb + (size_t)(j + 1) * KVL,
                            *k2 = kb + (size_t)(j + 2) * KVL, *k3 = kb + (size_t)(j + 3) * KVL;
                const float *q0 = qp[r], *q1 = qp[r + 1], *q2 = qp[r + 2], *q3 = qp[r + 3];
                for (int d = 0; d < KVL; d += 8) {
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
                float *s0 = S + (size_t)(r + 0) * PFA_BK + j, *s1 = S + (size_t)(r + 1) * PFA_BK + j;
                float *s2 = S + (size_t)(r + 2) * PFA_BK + j, *s3 = S + (size_t)(r + 3) * PFA_BK + j;
                s0[0] = hsum_avx(a00) * scale; s0[1] = hsum_avx(a01) * scale;
                s0[2] = hsum_avx(a02) * scale; s0[3] = hsum_avx(a03) * scale;
                s1[0] = hsum_avx(a10) * scale; s1[1] = hsum_avx(a11) * scale;
                s1[2] = hsum_avx(a12) * scale; s1[3] = hsum_avx(a13) * scale;
                s2[0] = hsum_avx(a20) * scale; s2[1] = hsum_avx(a21) * scale;
                s2[2] = hsum_avx(a22) * scale; s2[3] = hsum_avx(a23) * scale;
                s3[0] = hsum_avx(a30) * scale; s3[1] = hsum_avx(a31) * scale;
                s3[2] = hsum_avx(a32) * scale; s3[3] = hsum_avx(a33) * scale;
            }
            for (; j < nk; j++)
                for (int i = 0; i < 4; i++)
                    S[(size_t)(r + i) * PFA_BK + j] =
                        vec_dot_f32_f32(kb + (size_t)j * KVL, qp[r + i], KVL) * scale;
        }
    }
#endif
    for (; r < R; r++)
        for (int j = 0; j < nk; j++)
            S[(size_t)r * PFA_BK + j] = vec_dot_f32_f32(kb + (size_t)j * KVL, qp[r], KVL) * scale;
}

/* O[r][:] += sum_j P[r][j] * kb[j][:] */
static void glm5_pfa_pv(float *O, int R, const float *P, int nk, const float *kb, int KVL)
{
    int r = 0;
#if defined(PICOLM_AVX512)
    if ((KVL & 63) == 0) {
        for (; r + 3 < R; r += 4) {
            for (int c0 = 0; c0 < KVL; c0 += 64) {
                __m512 a[4][4];
                #pragma GCC unroll 4
                for (int i = 0; i < 4; i++) {
                    #pragma GCC unroll 4
                    for (int u = 0; u < 4; u++) a[i][u] = _mm512_loadu_ps(O + (size_t)(r + i) * KVL + c0 + 16 * u);
                }
                for (int j = 0; j < nk; j++) {
                    const float *kr = kb + (size_t)j * KVL + c0;
                    const __m512 v0 = _mm512_loadu_ps(kr), v1 = _mm512_loadu_ps(kr + 16),
                                 v2 = _mm512_loadu_ps(kr + 32), v3 = _mm512_loadu_ps(kr + 48);
                    #pragma GCC unroll 4
                    for (int i = 0; i < 4; i++) {
                        const __m512 pb = _mm512_set1_ps(P[(size_t)(r + i) * PFA_BK + j]);
                        a[i][0] = _mm512_fmadd_ps(pb, v0, a[i][0]);
                        a[i][1] = _mm512_fmadd_ps(pb, v1, a[i][1]);
                        a[i][2] = _mm512_fmadd_ps(pb, v2, a[i][2]);
                        a[i][3] = _mm512_fmadd_ps(pb, v3, a[i][3]);
                    }
                }
                #pragma GCC unroll 4
                for (int i = 0; i < 4; i++) {
                    #pragma GCC unroll 4
                    for (int u = 0; u < 4; u++) _mm512_storeu_ps(O + (size_t)(r + i) * KVL + c0 + 16 * u, a[i][u]);
                }
            }
        }
    }
#elif defined(PICOLM_AVX2)
    if ((KVL & 31) == 0) {
        for (; r + 3 < R; r += 4) {
            for (int c0 = 0; c0 < KVL; c0 += 32) {
                __m256 a[4][4];
                #pragma GCC unroll 4
                for (int i = 0; i < 4; i++) {
                    #pragma GCC unroll 4
                    for (int u = 0; u < 4; u++) a[i][u] = _mm256_loadu_ps(O + (size_t)(r + i) * KVL + c0 + 8 * u);
                }
                for (int j = 0; j < nk; j++) {
                    const float *kr = kb + (size_t)j * KVL + c0;
                    const __m256 v0 = _mm256_loadu_ps(kr), v1 = _mm256_loadu_ps(kr + 8),
                                 v2 = _mm256_loadu_ps(kr + 16), v3 = _mm256_loadu_ps(kr + 24);
                    #pragma GCC unroll 4
                    for (int i = 0; i < 4; i++) {
                        const __m256 pb = _mm256_set1_ps(P[(size_t)(r + i) * PFA_BK + j]);
                        a[i][0] = _mm256_fmadd_ps(pb, v0, a[i][0]);
                        a[i][1] = _mm256_fmadd_ps(pb, v1, a[i][1]);
                        a[i][2] = _mm256_fmadd_ps(pb, v2, a[i][2]);
                        a[i][3] = _mm256_fmadd_ps(pb, v3, a[i][3]);
                    }
                }
                #pragma GCC unroll 4
                for (int i = 0; i < 4; i++) {
                    #pragma GCC unroll 4
                    for (int u = 0; u < 4; u++) _mm256_storeu_ps(O + (size_t)(r + i) * KVL + c0 + 8 * u, a[i][u]);
                }
            }
        }
    }
#endif
    for (; r < R; r++) {
        float *o = O + (size_t)r * KVL;
        for (int j = 0; j < nk; j++) {
            const float w = P[(size_t)r * PFA_BK + j];
            const float *kr = kb + (size_t)j * KVL;
            for (int d = 0; d < KVL; d++) o[d] += w * kr[d];
        }
    }
}

static void glm5_pfa_task(int idx, void *vp)
{
    const glm5_pfa_t *x = (const glm5_pfa_t *)vp;
    const int NH = x->NH, KVL = x->KVL;
    const int hg = idx / x->nTB;
    const int tb = (int)(((long long)(idx % x->nTB) * x->perm) % x->nTB);  /* spread cost across threads */
    const int h0 = hg * PFA_HG, hn = (NH - h0 < PFA_HG) ? NH - h0 : PFA_HG;
    const int t0 = tb * PFA_BQ,  tn = (x->n_tokens - t0 < PFA_BQ) ? x->n_tokens - t0 : PFA_BQ;
    const int R = tn * hn;
    if (R <= 0) return;

    float O[PFA_MAXR * PFA_MAXKVL];
    float S[PFA_MAXR * PFA_BK];
    float m[PFA_MAXR], l[PFA_MAXR];
    const float *qp[PFA_MAXR];
    int lim[PFA_MAXR];
    for (int tt = 0; tt < tn; tt++)
        for (int hh = 0; hh < hn; hh++) {
            const int r = tt * hn + hh;
            qp[r]  = x->qabs + ((size_t)(t0 + tt) * NH + h0 + hh) * KVL;
            lim[r] = x->start_pos + t0 + tt;      /* last visible key, inclusive */
            m[r] = -INFINITY; l[r] = 0.f;
        }
    memset(O, 0, (size_t)R * KVL * sizeof(float));
    const int nkeys = x->start_pos + t0 + tn;      /* keys any row in this task can see */

    for (int kb = 0; kb < nkeys; kb += PFA_BK) {
        const int nk = (nkeys - kb < PFA_BK) ? nkeys - kb : PFA_BK;
        const float *kblk = x->cache + (size_t)kb * KVL;
        glm5_pfa_scores(qp, R, kblk, nk, KVL, x->scale, S, x->vx);

        for (int r = 0; r < R; r++) {
            float *sr = S + (size_t)r * PFA_BK;
            const int vis = lim[r] - kb + 1;       /* visible keys in this block */
            int nv = vis < 0 ? 0 : (vis > nk ? nk : vis);
#if defined(PICOLM_AVX512)
            if (x->vx) {
                if (nv == 0) { for (int j = 0; j < nk; j++) sr[j] = 0.f; continue; }
                __m512 vmx = _mm512_set1_ps(-INFINITY);
                for (int j = 0; j < nv; j += 16) {
                    const int rem = nv - j;
                    const __mmask16 mk = rem >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << rem) - 1u);
                    vmx = _mm512_mask_max_ps(vmx, mk, vmx, _mm512_loadu_ps(sr + j));
                }
                const float bm = _mm512_reduce_max_ps(vmx);
                const float mn = bm > m[r] ? bm : m[r];
                const float alpha = expf(m[r] - mn);
                const __m512 vmn = _mm512_set1_ps(mn);
                __m512 vsum = _mm512_setzero_ps();
                for (int j = 0; j < nv; j += 16) {
                    const int rem = nv - j;
                    const __mmask16 mk = rem >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << rem) - 1u);
                    const __m512 e = _mm512_maskz_mov_ps(mk, kda_expf16(_mm512_sub_ps(_mm512_loadu_ps(sr + j), vmn)));
                    _mm512_storeu_ps(sr + j, e);              /* lanes >= nv are stored as 0 */
                    vsum = _mm512_add_ps(vsum, e);
                }
                const float sum = _mm512_reduce_add_ps(vsum);
                for (int j = (nv + 15) & ~15; j < nk; j++) sr[j] = 0.f;
                l[r] = l[r] * alpha + sum;
                if (alpha != 1.0f) {
                    float *o = O + (size_t)r * KVL;
                    for (int d = 0; d < KVL; d++) o[d] *= alpha;
                }
                m[r] = mn;
                continue;
            }
#endif
            for (int j = nv; j < nk; j++) sr[j] = -INFINITY;
            float bm = -INFINITY;
            for (int j = 0; j < nv; j++) if (sr[j] > bm) bm = sr[j];
            if (nv == 0) { for (int j = 0; j < nk; j++) sr[j] = 0.f; continue; }
            const float mn = bm > m[r] ? bm : m[r];
            const float alpha = expf(m[r] - mn);   /* m[r] = -inf on the first block -> 0 */
            float sum = 0.f;
            for (int j = 0; j < nv; j++) { float e = expf(sr[j] - mn); sr[j] = e; sum += e; }
            for (int j = nv; j < nk; j++) sr[j] = 0.f;
            l[r] = l[r] * alpha + sum;
            if (alpha != 1.0f) {
                float *o = O + (size_t)r * KVL;
                for (int d = 0; d < KVL; d++) o[d] *= alpha;
            }
            m[r] = mn;
        }
        glm5_pfa_pv(O, R, S, nk, kblk, KVL);
    }

    for (int tt = 0; tt < tn; tt++)
        for (int hh = 0; hh < hn; hh++) {
            const int r = tt * hn + hh;
            const float inv = 1.0f / l[r];
            const float *o = O + (size_t)r * KVL;
            float *dst = x->ctx + ((size_t)(t0 + tt) * NH + h0 + hh) * KVL;
            for (int d = 0; d < KVL; d++) dst[d] = o[d] * inv;
        }
}

/* Returns 1 if it ran (caller falls back to the per-(head,token) core otherwise). */
static int glm5_mla_attn_pfa_ex(const model_config_t *c, run_state_t *s, int mla_ord,
                                const float *qabs_b, float *ctx_b, int n_tokens, int start_pos, int vexp_req)
{
    const int NH = c->n_heads, KVL = c->glm5_kv_lora;
    if (KVL > PFA_MAXKVL || KVL < 16 || n_tokens < 1) return 0;
    glm5_pfa_t x;
    x.qabs = qabs_b; x.ctx = ctx_b; x.n_tokens = n_tokens; x.start_pos = start_pos; x.vx = vexp_req;
    x.NH = NH; x.KVL = KVL; x.scale = 1.0f / sqrtf((float)c->glm5_mla_head_dim);
    x.cache = s->glm5.mla_kv + (size_t)mla_ord * c->max_seq_len * KVL;
    x.nTB = (n_tokens + PFA_BQ - 1) / PFA_BQ;
    /* Stride permutation, coprime with nTB: causal cost grows with the token block,
     * and the pool hands each thread a contiguous index range, so spread the blocks. */
    x.perm = 1;
    for (int cand = (int)(x.nTB * 0.618) | 1; cand < x.nTB * 4 + 8; cand += 2) {
        int a = cand, b = x.nTB;
        while (b) { int t = a % b; a = b; b = t; }
        if (a == 1) { x.perm = cand; break; }
    }
    const int nHG = (NH + PFA_HG - 1) / PFA_HG;
    tensor_parallel_for(nHG * x.nTB, glm5_pfa_task, &x);
    return 1;
}

static int glm5_mla_attn_pfa(const model_config_t *c, run_state_t *s, int mla_ord,
                             const float *qabs_b, float *ctx_b, int n_tokens, int start_pos)
{
    return glm5_mla_attn_pfa_ex(c, s, mla_ord, qabs_b, ctx_b, n_tokens, start_pos, glm5_pfa_vexp_enabled());
}

static void glm5_mla_attn_batch(
    const model_config_t *c, run_state_t *s, int mla_ord,
    const float *qabs_b, float *ctx_b, int n_tokens, int start_pos)
{
    if (glm5_pfa_enabled() && glm5_mla_attn_pfa(c, s, mla_ord, qabs_b, ctx_b, n_tokens, start_pos))
        return;
    const int NH = c->n_heads;
    const int KVL = c->glm5_kv_lora;
    const float scale = 1.0f / sqrtf((float)c->glm5_mla_head_dim);
    const float *cache = s->glm5.mla_kv + (size_t)mla_ord * c->max_seq_len * KVL;

    /* One scores row per task: the pool runs tasks in parallel and each needs
     * its own [max_seq] scratch, so n_tasks is ALSO the scratch size
     * (n_tasks * max_seq_len floats). It therefore has to be derived from the
     * machine, not guessed: a hardcoded 24 gave 384 tasks on a 1-core build
     * (run inline, one after another) and too few tasks on a 48-core one, and
     * it allocated ~150 MB of scratch at -c 100000 where ~26 MB is enough.
     *
     * n_threads is already clamped to MAX_THREADS by tensor_set_threads() and
     * the pool is created once from it (pool_init), so tensor_get_n_threads()
     * is the value to size against.
     *
     * Task count: there are NH heads and the batch is split into n_chunks token
     * ranges, giving NH*n_chunks tasks. NH alone already fills a many-core box,
     * so only add a chunk when the thread count exceeds NH. n_tasks therefore
     * stays at ~max(NH, n_threads), which both keeps the scratch bounded and
     * gives every worker something to do. */
    int nth = tensor_get_n_threads();
    if (nth < 1) nth = 1;
    const int target = (nth > NH) ? nth : NH;   /* tasks we would like */
    int n_chunks = (target + NH - 1) / NH;
    if (n_chunks < 1) n_chunks = 1;
    if (n_tokens > 0 && n_chunks > n_tokens) n_chunks = n_tokens;
    const int n_chunk = (n_tokens > 0) ? (n_tokens + n_chunks - 1) / n_chunks : 1;
    int n_tasks = NH * n_chunks;
    if (n_tasks < 1) n_tasks = 1;

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
        /* 2, not 4. The number of head blocks has to fit the pool: NH=64 with
         * hb=4 gives 16 blocks over 24 threads, so 8 threads idle and the
         * barrier waits on the 16 that ran. hb=2 gives 32 blocks, which
         * balances across 24 threads. Measured at pos 7111, MLA attention:
         *   hb=2  82-86 ms    hb=3  86-89 ms    hb=4  92-94 ms
         * The block-reduce path needs a low hb to show its benefit at all --
         * at hb=4 the imbalance hid it completely. */
        cached = e ? atoi(e) : 2;
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

/* ---- F16 latent cache readers ---------------------------------------- *
 *
 * The decode attention pass is pure bandwidth over the latent cache: measured
 * time tracks cache bytes ~1:1 with depth (14.1 ms at pos 1024, 87.9 ms at pos
 * 7109). The cache is stored F32 as the master (snapshots and prefill use it),
 * and mirrored into an F16 shadow (s->glm5.mla_kv16) that these read. Halving
 * the bytes halves the pass.
 *
 * Not bit-identical: each cached value is rounded to F16 first. Validate with
 * top-1 agreement / logit KL, not max|diff|. PICOLM_GLM5_MLA_F16=0 disables.
 */
/* Refresh the F16 shadow of the MLA latent cache from the F32 master for rows
 * [pos0, pos0+n) of MLA ordinal `ord`.
 *
 * Decode attention reads ONLY the shadow, so every writer of the F32 master
 * has to mirror into it: the decode store does that inline, and the batched
 * prefill store and the snapshot/checkpoint restore must call this. A row that
 * is never mirrored stays all-zero in the shadow, i.e. a prompt (or restored
 * context) that attention silently cannot see. */
static void glm5_mla_shadow_sync(const run_state_t *s, int max_seq, int KVL,
                                 int ord, int pos0, int n)
{
    if (!s->glm5.mla_kv16 || n <= 0) return;
    const size_t base = ((size_t)ord * max_seq + (size_t)pos0) * KVL;
    const float *src = s->glm5.mla_kv + base;
    uint16_t *dst = s->glm5.mla_kv16 + base;
    const size_t cnt = (size_t)n * KVL;
    for (size_t i = 0; i < cnt; i++) dst[i] = fp32_to_fp16(src[i]);
}

#if defined(PICOLM_AVX512)
/* F16 -> F32 for 16 lanes via _mm512_cvtph_ps (F16C + AVX-512F).
 * (Not _cvtsh_ss: that one is AVX512-FP16, a different ISA extension.) */
static inline __m512 glm5_f16x16_to_ps(const uint16_t *p) {
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));
}
#endif

/* dot(u16 row, f32 q) for KVL values */
static inline float glm5_mla_dot_f16(const uint16_t *row16, const float *q, int KVL) {
#if defined(PICOLM_AVX512)
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < KVL; i += 16)
        acc = _mm512_fmadd_ps(glm5_f16x16_to_ps(row16 + i), _mm512_loadu_ps(q + i), acc);
    float s = _mm512_reduce_add_ps(acc);
    for (; i < KVL; i++) s += fp16_to_fp32_lookup(row16[i]) * q[i];
    return s;
#elif defined(GLM5_FAST_F16)   /* AVX2 + F16C */
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < KVL; i += 8)
        acc = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row16 + i))),
                              _mm256_loadu_ps(q + i), acc);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    float s = _mm_cvtss_f32(lo);
    for (; i < KVL; i++) s += fp16_to_fp32_lookup(row16[i]) * q[i];
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < KVL; i++) s += fp16_to_fp32_lookup(row16[i]) * q[i];
    return s;
#endif
}

/* dst[i] += p * u16row[i] */
static inline void glm5_mla_accum_f16(float *dst, const uint16_t *row16, float p, int KVL) {
#if defined(PICOLM_AVX512)
    const __m512 vp = _mm512_set1_ps(p);
    int i = 0;
    for (; i + 15 < KVL; i += 16)
        _mm512_storeu_ps(dst + i,
            _mm512_fmadd_ps(vp, glm5_f16x16_to_ps(row16 + i), _mm512_loadu_ps(dst + i)));
    for (; i < KVL; i++) dst[i] += p * fp16_to_fp32_lookup(row16[i]);
#elif defined(GLM5_FAST_F16)   /* AVX2 + F16C */
    const __m256 vp = _mm256_set1_ps(p);
    int i = 0;
    for (; i + 7 < KVL; i += 8)
        _mm256_storeu_ps(dst + i,
            _mm256_fmadd_ps(vp, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row16 + i))),
                            _mm256_loadu_ps(dst + i)));
    for (; i < KVL; i++) dst[i] += p * fp16_to_fp32_lookup(row16[i]);
#else
    for (int i = 0; i < KVL; i++) dst[i] += p * fp16_to_fp32_lookup(row16[i]);
#endif
}

typedef struct {
    const float *cache;   /* [max_seq][KVL] rows for this MLA ordinal */
    const uint16_t *cache16; /* F16 shadow of `cache`, or NULL */
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

/* Softmax + PV for a whole head block with ONE pass over the cache.
 *
 * glm5_mla_reduce_head() re-streams the entire latent cache (n_kv x KVL floats,
 * 14.6 MB at 7k context) for every head, so a block of hb heads read it hb
 * times and a layer read it 64 times: ~10 GB/token of L3 traffic at 7k, which
 * is what the MLA attention region was bound by. The score pass already shares
 * each row across the block; this does the same for the weighted sum: each
 * cache row is loaded once and accumulated into all hn head accumulators
 * (hn*KVL floats, L1-resident).
 *
 * Bit-identical to calling glm5_mla_reduce_head() per head: for every head the
 * softmax is computed by the same code and the positions are accumulated in the
 * same ascending order with the same skip rule; only the interleaving ACROSS
 * heads changes, and heads share no state.
 * PICOLM_GLM5_MLA_REDUCE_PERHEAD=1 restores the per-head loop (A/B). */
#define GLM5_MLA_RB_MAX 16
static int glm5_mla_reduce_perhead(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("PICOLM_GLM5_MLA_REDUCE_PERHEAD") ? 1 : 0;
    return cached;
}

static void glm5_mla_reduce_block(
        float *ctx0, float *sc0, int n, int hn,
        const float *cache, const uint16_t *cache16, const int *sel, int dense, int KVL)
{
    float inv[GLM5_MLA_RB_MAX];
    for (int k = 0; k < hn; k++) {
        float *sc = sc0 + (size_t)k * n;
        float mx = -INFINITY;
        for (int i = 0; i < n; i++) { float v = sc[i]; if (v > mx) mx = v; }
        float sum = 0.f;
        for (int i = 0; i < n; i++) { float e = expf(sc[i] - mx); sc[i] = e; sum += e; }
        inv[k] = 1.0f / sum;
        memset(ctx0 + (size_t)k * KVL, 0, (size_t)KVL * sizeof(float));
    }
    if (cache16) {
        for (int i = 0; i < n; i++) {
            const int t = dense ? i : sel[i];
            const uint16_t *row = cache16 + (size_t)t * KVL;
            for (int k = 0; k < hn; k++) {
                const float p = sc0[(size_t)k * n + i] * inv[k];
                if (p < 1e-10f) continue;
                glm5_mla_accum_f16(ctx0 + (size_t)k * KVL, row, p, KVL);
            }
        }
        return;
    }
    for (int i = 0; i < n; i++) {
        const int t = dense ? i : sel[i];
        const float *row = cache + (size_t)t * KVL;
        for (int k = 0; k < hn; k++) {
            const float p = sc0[(size_t)k * n + i] * inv[k];
            if (p < 1e-10f) continue;
            vec_scale_acc_f32(ctx0 + (size_t)k * KVL, row, p, KVL);
        }
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

    if (t->cache16) {
        for (int p = 0; p < n; p++) {
            const uint16_t *row = t->cache16 + (size_t)(t->dense ? p : t->sel[p]) * KVL;
            for (int k = 0; k < hn; k++)
                sb[(size_t)k * n + p] =
                    glm5_mla_dot_f16(row, t->qabs + (size_t)(h0 + k) * KVL, KVL) * t->scale;
        }
    } else {
        for (int p = 0; p < n; p++) {
            const float *row = t->cache + (size_t)(t->dense ? p : t->sel[p]) * KVL;
            for (int k = 0; k < hn; k++)
                sb[(size_t)k * n + p] =
                    vec_dot_f32_f32(row, t->qabs + (size_t)(h0 + k) * KVL, KVL) * t->scale;
        }
    }
    if (hn <= GLM5_MLA_RB_MAX && !glm5_mla_reduce_perhead()) {
        glm5_mla_reduce_block(t->ctx + (size_t)h0 * KVL, sb, n, hn,
                              t->cache, t->cache16, t->sel, t->dense, KVL);
    } else {
        for (int k = 0; k < hn; k++)
            glm5_mla_reduce_head(t->ctx + (size_t)(h0 + k) * KVL, sb + (size_t)k * n,
                                 t->cache, t->sel, n, t->dense, KVL);
    }
}

/* ---- Split-KV ("flash-decoding") dense MLA attention over the F16 shadow ----
 *
 * The head-block path gives each task a few heads and the WHOLE cache, so every
 * cache row is pulled from L3 once per block per pass: 32 blocks x 2 passes at
 * hb=2, i.e. ~64x the cache bytes per layer. That is why attention time tracked
 * cache bytes and why no kernel-side trick (register tiling, F16) could do more
 * than halve it. ik_llama.cpp's CPU flash-attn does the opposite for decode:
 * split the KEYS across threads, give each thread ALL heads for its key range
 * (the latent is a single shared K/V head), then merge the per-thread partial
 * softmaxes. Each row is then read from L3 once (1x, not 64x) and, because a
 * thread's chunk (~300 rows x 1 KB at 7k/24 threads) stays in L2, the PV pass
 * re-reads it from L2 rather than L3.
 *
 *   chunk task c (key range [p0,p1)), for all NH heads:
 *     S[h][p]   = scale * dot(q_h, k_p)    4 heads x 4 rows register tile; the FMA
 *                                           order matches glm5_mla_dot_f16, so the
 *                                           scores are bit-identical to the
 *                                           head-block path
 *     m_c, l_c  = chunk max / sum of exp(S - m_c)
 *     part[h]   = sum_p exp(S[h][p]-m_c) * k_p   2 heads x 128 columns kept in zmm
 *                                           across the whole key loop
 *   merge task h: M = max_c m_c; out = sum_c exp(m_c-M) part_c / sum_c exp(m_c-M) l_c
 *
 * NOT bit-identical to the head-block path (softmax is summed per chunk, and the
 * p < 1e-10 skip is gone); accurate to float rounding. ON by default for
 * n_kv >= 512; PICOLM_GLM5_MLA_SPLITKV=0 disables it, =N changes the threshold. */
#define GLM5_SKV_MAX_CHUNKS 64

static int glm5_mla_splitkv_min(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("PICOLM_GLM5_MLA_SPLITKV");
        int v = e ? atoi(e) : 1;          /* unset => enabled, threshold 512 */
        cached = v <= 0 ? 0 : (v == 1 ? 512 : v);
    }
    return cached;
}

typedef struct {
    const uint16_t *cache16;
    const float *qabs;
    float *ctx;
    float *part;      /* [nchunks][NH][KVL] */
    float *pm, *pl;   /* [nchunks][NH]      */
    float *sc;        /* [nchunks][NH][cmax] */
    int NH, KVL, n_kv, nchunks, cmax;
    int sb;           /* key sub-block per pass (0 = whole chunk per pass, the original kernel) */
    float scale;
} glm5_skv_t;

/* ---- L2-blocked chunk task ------------------------------------------------------------------
 *
 * The original chunk task sweeps the WHOLE chunk (ctx/nthreads rows of 1 KB) once per 4-head quad
 * in the score pass (16 sweeps) and once per head pair x column tile in the PV pass (128 sweeps),
 * and its score buffer is NH x len floats. That is fine while a thread's chunk fits its L2
 * (1 MB on Cascade Lake: ~24k context with 24 threads) and ruinous after: every sweep re-reads the
 * chunk from L3/DRAM, ~200 MB per thread per layer at 100k. Measured on the Xeon, the marginal decode
 * cost per context token tripled (2 -> 6 us) exactly there, while ik stays at ~1.3 us.
 *
 * Here a thread processes its key range in sub-blocks of `sb` keys (default 256 = 256 KB of F16 rows
 * + 64 KB of scores + 128 KB of q + 128 KB of accumulators): scores -> online softmax (running max m,
 * sum l, accumulators rescaled) -> PV, all on an L2-resident sub-block, so each key row is fetched
 * from memory once. The merge across chunks is unchanged (part, m, l keep their meaning).
 * Tiles: scores 8 heads x 2 keys, PV 8 heads x 32 columns -- 16 accumulators each, but only 2
 * F16->F32 conversions per 16 FMAs (the 4x4 / 2x8 tiles needed 4 and 8).
 * PICOLM_GLM5_SKV_SB=<n> sets the sub-block (16..512); 0 selects the original kernel. */
#define GLM5_SKV_SB_MAX 512
static int glm5_skv_sb(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("PICOLM_GLM5_SKV_SB");
        int v = e ? atoi(e) : 256;
        if (v <= 0) v = 0; else { if (v < 16) v = 16; if (v > GLM5_SKV_SB_MAX) v = GLM5_SKV_SB_MAX; v &= ~15; }
        cached = v;
    }
    return cached;
}

/* S[h*SB + j] = scale * dot(q_h, row_j) for h < NH, j < nk */
static void glm5_skv_scores_blk(const float *q, int NH, const uint16_t *rows, int nk, int KVL,
                                float scale, float *S, int SB)
{
    int h = 0;
#if defined(PICOLM_AVX512)
    if ((KVL & 15) == 0) {
        for (; h + 7 < NH; h += 8) {
            const float *qh = q + (size_t)h * KVL;
            int p = 0;
            for (; p + 1 < nk; p += 2) {
                __m512 a0[8], a1[8];
                #pragma GCC unroll 8
                for (int hh = 0; hh < 8; hh++) { a0[hh] = _mm512_setzero_ps(); a1[hh] = a0[hh]; }
                const uint16_t *r0 = rows + (size_t)p * KVL, *r1 = r0 + KVL;
                for (int d = 0; d < KVL; d += 16) {
                    const __m512 k0 = glm5_f16x16_to_ps(r0 + d), k1 = glm5_f16x16_to_ps(r1 + d);
                    #pragma GCC unroll 8
                    for (int hh = 0; hh < 8; hh++) {
                        const __m512 w = _mm512_loadu_ps(qh + (size_t)hh * KVL + d);
                        a0[hh] = _mm512_fmadd_ps(k0, w, a0[hh]);
                        a1[hh] = _mm512_fmadd_ps(k1, w, a1[hh]);
                    }
                }
                __m512 acc[16];
                #pragma GCC unroll 8
                for (int hh = 0; hh < 8; hh++) { acc[2 * hh] = a0[hh]; acc[2 * hh + 1] = a1[hh]; }
                float tmp[16];
                _mm512_storeu_ps(tmp, _mm512_mul_ps(glm5_pfa_hsum16(acc), _mm512_set1_ps(scale)));
                for (int hh = 0; hh < 8; hh++) {
                    S[(size_t)(h + hh) * SB + p]     = tmp[2 * hh];
                    S[(size_t)(h + hh) * SB + p + 1] = tmp[2 * hh + 1];
                }
            }
            for (; p < nk; p++)
                for (int hh = 0; hh < 8; hh++)
                    S[(size_t)(h + hh) * SB + p] =
                        glm5_mla_dot_f16(rows + (size_t)p * KVL, qh + (size_t)hh * KVL, KVL) * scale;
        }
    }
#endif
    for (; h < NH; h++)
        for (int p = 0; p < nk; p++)
            S[(size_t)h * SB + p] = glm5_mla_dot_f16(rows + (size_t)p * KVL, q + (size_t)h * KVL, KVL) * scale;
}

/* part[h][:] += sum_j S[h*SB + j] * row_j[:] */
static void glm5_skv_pv_blk(float *part, int NH, const float *S, int nk, const uint16_t *rows,
                            int KVL, int SB)
{
    int h = 0;
#if defined(PICOLM_AVX512)
    if ((KVL & 31) == 0) {
        for (; h + 7 < NH; h += 8) {
            for (int c0 = 0; c0 < KVL; c0 += 32) {
                __m512 a0[8], a1[8];
                #pragma GCC unroll 8
                for (int hh = 0; hh < 8; hh++) {
                    a0[hh] = _mm512_loadu_ps(part + (size_t)(h + hh) * KVL + c0);
                    a1[hh] = _mm512_loadu_ps(part + (size_t)(h + hh) * KVL + c0 + 16);
                }
                for (int p = 0; p < nk; p++) {
                    const uint16_t *row = rows + (size_t)p * KVL + c0;
                    const __m512 r0 = glm5_f16x16_to_ps(row), r1 = glm5_f16x16_to_ps(row + 16);
                    #pragma GCC unroll 8
                    for (int hh = 0; hh < 8; hh++) {
                        const __m512 pw = _mm512_set1_ps(S[(size_t)(h + hh) * SB + p]);
                        a0[hh] = _mm512_fmadd_ps(pw, r0, a0[hh]);
                        a1[hh] = _mm512_fmadd_ps(pw, r1, a1[hh]);
                    }
                }
                #pragma GCC unroll 8
                for (int hh = 0; hh < 8; hh++) {
                    _mm512_storeu_ps(part + (size_t)(h + hh) * KVL + c0, a0[hh]);
                    _mm512_storeu_ps(part + (size_t)(h + hh) * KVL + c0 + 16, a1[hh]);
                }
            }
        }
    }
#endif
    for (; h < NH; h++)
        for (int p = 0; p < nk; p++)
            glm5_mla_accum_f16(part + (size_t)h * KVL, rows + (size_t)p * KVL, S[(size_t)h * SB + p], KVL);
}

static void glm5_skv_chunk_task_blk(int c, const glm5_skv_t *x)
{
    const int NH = x->NH, KVL = x->KVL, SB = x->sb;
    const int p0 = (int)((long long)x->n_kv * c / x->nchunks);
    const int p1 = (int)((long long)x->n_kv * (c + 1) / x->nchunks);
    const int len = p1 - p0;
    float *part = x->part + (size_t)c * NH * KVL;
    float *pm = x->pm + (size_t)c * NH, *pl = x->pl + (size_t)c * NH;
    const uint16_t *base = x->cache16 + (size_t)p0 * KVL;

    memset(part, 0, (size_t)NH * KVL * sizeof(float));
    if (len <= 0) { for (int h = 0; h < NH; h++) pm[h] = -INFINITY, pl[h] = 0.f; return; }

    float S[64 * GLM5_SKV_SB_MAX];
    float m[64], l[64];
    for (int h = 0; h < NH; h++) { m[h] = -INFINITY; l[h] = 0.f; }

    for (int kb = 0; kb < len; kb += SB) {
        const int nk = (len - kb < SB) ? len - kb : SB;
        const uint16_t *rows = base + (size_t)kb * KVL;
        glm5_skv_scores_blk(x->qabs, NH, rows, nk, KVL, x->scale, S, SB);
        for (int h = 0; h < NH; h++) {
            float *sr = S + (size_t)h * SB;
            float sum;
            float mn;
#if defined(PICOLM_AVX512)
            __m512 vmx = _mm512_set1_ps(-INFINITY);
            for (int j = 0; j < nk; j += 16) {
                const int rem = nk - j;
                const __mmask16 mk = rem >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << rem) - 1u);
                vmx = _mm512_mask_max_ps(vmx, mk, vmx, _mm512_loadu_ps(sr + j));
            }
            { const float bm = _mm512_reduce_max_ps(vmx); mn = bm > m[h] ? bm : m[h]; }
            {
                const __m512 vmn = _mm512_set1_ps(mn);
                __m512 vsum = _mm512_setzero_ps();
                for (int j = 0; j < nk; j += 16) {
                    const int rem = nk - j;
                    const __mmask16 mk = rem >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << rem) - 1u);
                    const __m512 e = _mm512_maskz_mov_ps(mk, kda_expf16(_mm512_sub_ps(_mm512_loadu_ps(sr + j), vmn)));
                    _mm512_storeu_ps(sr + j, e);
                    vsum = _mm512_add_ps(vsum, e);
                }
                sum = _mm512_reduce_add_ps(vsum);
            }
#else
            float bm = -INFINITY;
            for (int j = 0; j < nk; j++) if (sr[j] > bm) bm = sr[j];
            mn = bm > m[h] ? bm : m[h];
            sum = 0.f;
            for (int j = 0; j < nk; j++) { const float e = expf(sr[j] - mn); sr[j] = e; sum += e; }
#endif
            const float alpha = (m[h] == -INFINITY) ? 0.0f : expf(m[h] - mn);
            l[h] = l[h] * alpha + sum;
            if (alpha != 1.0f) {
                float *o = part + (size_t)h * KVL;
                for (int d = 0; d < KVL; d++) o[d] *= alpha;
            }
            m[h] = mn;
        }
        glm5_skv_pv_blk(part, NH, S, nk, rows, KVL, SB);
    }
    for (int h = 0; h < NH; h++) { pm[h] = m[h]; pl[h] = l[h]; }
}

static void glm5_skv_chunk_task(int c, void *vp)
{
    const glm5_skv_t *x = (const glm5_skv_t *)vp;
    if (x->sb > 0) { glm5_skv_chunk_task_blk(c, x); return; }
    const int NH = x->NH, KVL = x->KVL, cmax = x->cmax;
    const int p0 = (int)((long long)x->n_kv * c / x->nchunks);
    const int p1 = (int)((long long)x->n_kv * (c + 1) / x->nchunks);
    const int len = p1 - p0;
    float *S    = x->sc   + (size_t)c * NH * cmax;
    float *part = x->part + (size_t)c * NH * KVL;
    float *pm   = x->pm   + (size_t)c * NH;
    float *pl   = x->pl   + (size_t)c * NH;
    const uint16_t *base = x->cache16 + (size_t)p0 * KVL;
    const float *q = x->qabs;
    const float scale = x->scale;

    if (len <= 0) {       /* empty chunk: weight exp(-inf) = 0 in the merge */
        for (int h = 0; h < NH; h++) pm[h] = -INFINITY, pl[h] = 0.f;
        memset(part, 0, (size_t)NH * KVL * sizeof(float));
        return;
    }

    /* ---- scores ---- */
#if defined(PICOLM_AVX512)
    int h = 0;
    if ((KVL & 15) == 0) {
        for (; h + 3 < NH; h += 4) {
            int p = 0;
            for (; p + 3 < len; p += 4) {
                __m512 a00 = _mm512_setzero_ps(), a01 = a00, a02 = a00, a03 = a00,
                       a10 = a00, a11 = a00, a12 = a00, a13 = a00,
                       a20 = a00, a21 = a00, a22 = a00, a23 = a00,
                       a30 = a00, a31 = a00, a32 = a00, a33 = a00;
                const uint16_t *r0 = base + (size_t)(p + 0) * KVL;
                const uint16_t *r1 = base + (size_t)(p + 1) * KVL;
                const uint16_t *r2 = base + (size_t)(p + 2) * KVL;
                const uint16_t *r3 = base + (size_t)(p + 3) * KVL;
                const float *q0 = q + (size_t)(h + 0) * KVL;
                const float *q1 = q + (size_t)(h + 1) * KVL;
                const float *q2 = q + (size_t)(h + 2) * KVL;
                const float *q3 = q + (size_t)(h + 3) * KVL;
                for (int d = 0; d < KVL; d += 16) {
                    const __m512 v0 = glm5_f16x16_to_ps(r0 + d), v1 = glm5_f16x16_to_ps(r1 + d),
                                 v2 = glm5_f16x16_to_ps(r2 + d), v3 = glm5_f16x16_to_ps(r3 + d);
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
                float *s0 = S + (size_t)(h + 0) * cmax + p, *s1 = S + (size_t)(h + 1) * cmax + p;
                float *s2 = S + (size_t)(h + 2) * cmax + p, *s3 = S + (size_t)(h + 3) * cmax + p;
                s0[0] = _mm512_reduce_add_ps(a00) * scale; s0[1] = _mm512_reduce_add_ps(a01) * scale;
                s0[2] = _mm512_reduce_add_ps(a02) * scale; s0[3] = _mm512_reduce_add_ps(a03) * scale;
                s1[0] = _mm512_reduce_add_ps(a10) * scale; s1[1] = _mm512_reduce_add_ps(a11) * scale;
                s1[2] = _mm512_reduce_add_ps(a12) * scale; s1[3] = _mm512_reduce_add_ps(a13) * scale;
                s2[0] = _mm512_reduce_add_ps(a20) * scale; s2[1] = _mm512_reduce_add_ps(a21) * scale;
                s2[2] = _mm512_reduce_add_ps(a22) * scale; s2[3] = _mm512_reduce_add_ps(a23) * scale;
                s3[0] = _mm512_reduce_add_ps(a30) * scale; s3[1] = _mm512_reduce_add_ps(a31) * scale;
                s3[2] = _mm512_reduce_add_ps(a32) * scale; s3[3] = _mm512_reduce_add_ps(a33) * scale;
            }
            for (; p < len; p++)
                for (int j = 0; j < 4; j++)
                    S[(size_t)(h + j) * cmax + p] =
                        glm5_mla_dot_f16(base + (size_t)p * KVL, q + (size_t)(h + j) * KVL, KVL) * scale;
        }
    }
    for (; h < NH; h++)
        for (int p = 0; p < len; p++)
            S[(size_t)h * cmax + p] =
                glm5_mla_dot_f16(base + (size_t)p * KVL, q + (size_t)h * KVL, KVL) * scale;
#else
    for (int h = 0; h < NH; h++)
        for (int p = 0; p < len; p++)
            S[(size_t)h * cmax + p] =
                glm5_mla_dot_f16(base + (size_t)p * KVL, q + (size_t)h * KVL, KVL) * scale;
#endif

    /* ---- chunk-local softmax ---- */
    for (int h2 = 0; h2 < NH; h2++) {
        float *sr = S + (size_t)h2 * cmax;
        float mx = -INFINITY;
        for (int i = 0; i < len; i++) if (sr[i] > mx) mx = sr[i];
        float sum = 0.f;
        for (int i = 0; i < len; i++) { float e = expf(sr[i] - mx); sr[i] = e; sum += e; }
        pm[h2] = mx; pl[h2] = sum;
    }

    /* ---- partial PV: accumulators for a 128-column tile of two heads stay in
     * zmm registers across the whole key loop ---- */
#if defined(PICOLM_AVX512)
    if ((KVL & 127) == 0) {
        int h2 = 0;
        for (; h2 + 1 < NH; h2 += 2) {
            const float *s0 = S + (size_t)h2 * cmax, *s1 = s0 + cmax;
            for (int c0 = 0; c0 < KVL; c0 += 128) {
                __m512 a0[8], a1[8];
                #pragma GCC unroll 8
                for (int j = 0; j < 8; j++) { a0[j] = _mm512_setzero_ps(); a1[j] = a0[j]; }
                for (int p = 0; p < len; p++) {
                    const uint16_t *row = base + (size_t)p * KVL + c0;
                    const __m512 v0 = _mm512_set1_ps(s0[p]), v1 = _mm512_set1_ps(s1[p]);
                    #pragma GCC unroll 8
                    for (int j = 0; j < 8; j++) {
                        const __m512 r = glm5_f16x16_to_ps(row + 16 * j);
                        a0[j] = _mm512_fmadd_ps(v0, r, a0[j]);
                        a1[j] = _mm512_fmadd_ps(v1, r, a1[j]);
                    }
                }
                float *o0 = part + (size_t)h2 * KVL + c0, *o1 = o0 + KVL;
                #pragma GCC unroll 8
                for (int j = 0; j < 8; j++) {
                    _mm512_storeu_ps(o0 + 16 * j, a0[j]);
                    _mm512_storeu_ps(o1 + 16 * j, a1[j]);
                }
            }
        }
        for (; h2 < NH; h2++) {
            float *o = part + (size_t)h2 * KVL;
            memset(o, 0, (size_t)KVL * sizeof(float));
            for (int p = 0; p < len; p++)
                glm5_mla_accum_f16(o, base + (size_t)p * KVL, S[(size_t)h2 * cmax + p], KVL);
        }
        return;
    }
#endif
    for (int h2 = 0; h2 < NH; h2++) {
        float *o = part + (size_t)h2 * KVL;
        memset(o, 0, (size_t)KVL * sizeof(float));
        for (int p = 0; p < len; p++)
            glm5_mla_accum_f16(o, base + (size_t)p * KVL, S[(size_t)h2 * cmax + p], KVL);
    }
}

static void glm5_skv_merge_task(int h, void *vp)
{
    const glm5_skv_t *x = (const glm5_skv_t *)vp;
    const int NH = x->NH, KVL = x->KVL, nc = x->nchunks;
    float M = -INFINITY;
    for (int c = 0; c < nc; c++) { float m = x->pm[(size_t)c * NH + h]; if (m > M) M = m; }
    float w[GLM5_SKV_MAX_CHUNKS], L = 0.f;
    for (int c = 0; c < nc; c++) {
        w[c] = expf(x->pm[(size_t)c * NH + h] - M);
        L += w[c] * x->pl[(size_t)c * NH + h];
    }
    const float invL = 1.0f / L;
    float *out = x->ctx + (size_t)h * KVL;
    memset(out, 0, (size_t)KVL * sizeof(float));
    for (int c = 0; c < nc; c++) {
        const float f = w[c] * invL;
        const float *pt = x->part + ((size_t)c * NH + h) * KVL;
        for (int d = 0; d < KVL; d++) out[d] += f * pt[d];
    }
}

/* Returns 1 if it ran. nchunks is clamped to [1, min(64, n_kv)]. */
static int glm5_mla_splitkv_ex(const uint16_t *cache16, const float *qabs, float *ctx,
                               int NH, int KVL, int n_kv, float scale, int nchunks, int sb)
{
    if (!cache16 || n_kv < 1 || NH < 1) return 0;
    if (NH > 64) sb = 0;                    /* the blocked task keeps its per-head state on the stack */
    if (nchunks > GLM5_SKV_MAX_CHUNKS) nchunks = GLM5_SKV_MAX_CHUNKS;
    if (nchunks > n_kv) nchunks = n_kv;
    if (nchunks < 1) nchunks = 1;
    const int cmax = (n_kv + nchunks - 1) / nchunks;
    const size_t n_part = (size_t)nchunks * NH * KVL, n_ml = (size_t)nchunks * NH,
                 n_sc = sb > 0 ? 0 : (size_t)nchunks * NH * cmax;     /* blocked task keeps scores on its stack */
    static float *buf = NULL; static size_t cap = 0;
    const size_t need = n_part + 2 * n_ml + n_sc;
    if (need > cap) {
        float *nb = (float *)realloc(buf, need * sizeof(float));
        if (!nb) return 0;
        buf = nb; cap = need;
    }
    glm5_skv_t x;
    x.cache16 = cache16; x.qabs = qabs; x.ctx = ctx;
    x.part = buf; x.pm = buf + n_part; x.pl = x.pm + n_ml; x.sc = x.pl + n_ml;
    x.NH = NH; x.KVL = KVL; x.n_kv = n_kv; x.nchunks = nchunks; x.cmax = cmax; x.scale = scale; x.sb = sb;
    tensor_parallel_for(nchunks, glm5_skv_chunk_task, &x);
    tensor_parallel_for(NH, glm5_skv_merge_task, &x);
    return 1;
}

static int glm5_mla_splitkv(const uint16_t *cache16, const float *qabs, float *ctx,
                            int NH, int KVL, int n_kv, float scale, int nchunks)
{
    return glm5_mla_splitkv_ex(cache16, qabs, ctx, NH, KVL, n_kv, scale, nchunks, glm5_skv_sb());
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
        /* 64 independent per-head matmuls: one pool wake instead of 64
         * (tensor_matmul_multi is bit-identical to per-job matmul()). */
        {
            mm_job_t jobs[64];
            const int nj = NH <= 64 ? NH : 0;
            for (int h = 0; h < nj; h++) {
                jobs[h].out = qabs + h*KVL; jobs[h].x = q + h*MHD;
                jobs[h].W = kb + h * k_head_sz; jobs[h].n = MHD; jobs[h].d = KVL;
                jobs[h].qtype = gw->type_mla_k_b;
            }
            if (!nj || !tensor_matmul_multi(jobs, nj)) {
                for (int h = 0; h < NH; h++) {
                    const void *wh = kb + h * k_head_sz;
                    matmul(qabs + h*KVL, q + h*MHD, wh, MHD, KVL, gw->type_mla_k_b);
                }
            }
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
        /* Mirror into the F16 shadow, which is what the attention pass reads. */
        if (s->glm5.mla_kv16) {
            uint16_t *c16 = s->glm5.mla_kv16 + (size_t)mla_ord * c->max_seq_len * KVL
                          + (size_t)pos * KVL;
            for (int d = 0; d < KVL; d++) c16[d] = fp32_to_fp16(kv[d]);
        }
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
        /* F16 shadow for the dense path only. The DSA branch and the scalar
         * reference loop keep reading the F32 master: they are not the
         * bandwidth problem, and staying on F32 keeps them exact. */
        const uint16_t *cache16 = s->glm5.mla_kv16
            ? s->glm5.mla_kv16 + (size_t)mla_ord * c->max_seq_len * KVL : NULL;
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
            int skv_done = 0;
            if (cache16 && glm5_mla_splitkv_min() > 0 && n_kv >= glm5_mla_splitkv_min()) {
                int nc = tensor_get_n_threads();
                if (nc > n_kv / 64) nc = n_kv / 64;
                skv_done = glm5_mla_splitkv(cache16, qabs_p, ctx, NH, KVL, n_kv, scale, nc);
            }
            if (!skv_done) {
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
                tc.cache = cache; tc.cache16 = cache16; tc.sel = NULL;
                tc.qabs = qabs_p; tc.ctx = ctx;
                tc.scores = scr; tc.NH = NH; tc.KVL = KVL; tc.n_kv = n_kv;
                tc.dense = 1; tc.hb = hb; tc.scale = scale;
                if (glm5_prof_mode() >= 4) {
                    static int shown = 0;
                    if (!shown++) fprintf(stderr, "[glm5 mla] n_blocks=%d hb=%d n_kv=%d n_threads=%d\n",
                                           n_blocks, hb, n_kv, tensor_get_n_threads());
                }
                tensor_parallel_for(n_blocks, glm5_mla_head_block_task, &tc);
            }
            }   /* !skv_done */
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
        {
            mm_job_t jobs[64];
            const int nj = NH <= 64 ? NH : 0;
            for (int h = 0; h < nj; h++) {
                jobs[h].out = o_concat + h*MHD; jobs[h].x = ctx + h*KVL;
                jobs[h].W = vb + h * v_head_sz; jobs[h].n = KVL; jobs[h].d = MHD;
                jobs[h].qtype = gw->type_mla_v_b;
            }
            if (!nj || !tensor_matmul_multi(jobs, nj)) {
                for (int h = 0; h < NH; h++) {
                    const void *wh = vb + h * v_head_sz;
                    matmul(o_concat + h*MHD, ctx + h*KVL, wh, KVL, MHD, gw->type_mla_v_b);
                }
            }
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
void glm5_moe_shared_expert(const model_t *m, const layer_weights_t *lw,
                            run_state_t *s, const float *cur, float *out,
                            int SFF)
{
    const model_config_t *c = &m->config;
    const int D = c->n_embd;
    const float clamp_v = glm5_clamp_value_shexp(c);

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

/* Non-static: reused verbatim by the DeepSeek V3 forward pass
 * (model_deepseek2.c), whose routing semantics are identical. */
void glm5_moe_ffn(const model_t *m, const glm5_layer_weights_t *gw,
                  const layer_weights_t *lw, run_state_t *s,
                  const float *cur, float *out, int layer_idx)
{
    if (moe_dbg_on < 0) {
        const char *e = getenv("PICOLM_DBG_MOE_OUT");
        moe_dbg_on = (e && *e && *e != '0');
    }
    const model_config_t *c = &m->config;
    const int D = c->n_embd;
    const int NE = c->n_expert;
    const int AK = c->n_expert_used;
    const int FF = c->n_ff_exp;         /* expert FFN width (2048) */
    const int SFF = c->n_ff_shexp;      /* shared expert FFN width */
    const float clamp_v = glm5_clamp_value(c);

    char t[96];
    glm5_stats("x.moe.in", cur, D);
    if (getenv("PICOLM_DBG_MOEIN")) {
        double ss=0; for (int q=0;q<D;q++) ss+=(double)cur[q]*cur[q];
        static int once=0;
        if (once++ < 80) fprintf(stderr, "[moein L%d] cur rms=%.6f first3 %.6f %.6f %.6f\n",
                layer_idx, (float)sqrt(ss/D), cur[0], cur[1], cur[2]);
    }

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
    if (glm5_dbg_enabled()) {
        float mn=logits[0], mx=logits[0]; double ss=0;
        for (int i = 0; i < NE; i++) { if (logits[i]<mn) mn=logits[i]; if (logits[i]>mx) mx=logits[i]; ss+=(double)logits[i]*logits[i]; }
        fprintf(stderr, "[moe logits] rms=%.6f min=%.6f max=%.6f\n", (float)sqrt(ss/NE), mn, mx);
    }
    /* Dump the raw router logits so they can be diffed element-wise against
     * ik's ffn_moe_logits-<il>.f32 -- expert selection on this model is a
     * near-tie (see notes), so this is the only way to tell "different
     * summation order" from "wrong tensor". */
    {
        static int dump_layer = -2;
        if (dump_layer == -2) {
            const char *e = getenv("PICOLM_DBG_MOE_LOGITS");
            dump_layer = e ? atoi(e) : -1;
        }
        if (dump_layer >= 0 && dump_layer == layer_idx) {
            const char *dir = getenv("PICOLM_DUMP_DIR");
            char fn[512];
            static int nw = 0;
            snprintf(fn, sizeof(fn), "%s/ffn_moe_logits-%d.f32", dir ? dir : "/tmp/picd", dump_layer);
            FILE *fp = fopen(fn, "wb");
            if (fp) { fwrite(logits, sizeof(float), (size_t)NE, fp); fclose(fp); }
            /* Also the router input, so the two can be checked together. */
            snprintf(fn, sizeof(fn), "%s/ffn_norm-%d.f32", dir ? dir : "/tmp/picd", dump_layer);
            fp = fopen(fn, "wb");
            if (fp) { fwrite(cur, sizeof(float), (size_t)D, fp); fclose(fp); }
            if (nw++ < 2) fprintf(stderr, "[moe dump] wrote layer %d (pass %d)\n", dump_layer, nw);
        }
    }
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
        if (glm5_dbg_enabled()) {
            float sn = 0.f; for (int k = 0; k < AK; k++) sn += wts[k]*wts[k];
            fprintf(stderr, "[moe sel] sel=");
            for (int k = 0; k < AK; k++) fprintf(stderr, " %d", sel[k]);
            fprintf(stderr, "  sum=%.6f norm_rms=%.6f scale=%.3f w:", sum, sqrtf(sn/AK), c->glm5_moe_scale);
            for (int k = 0; k < AK; k++) fprintf(stderr, " %.4f", wts[k]);
            fprintf(stderr, "\n");
        }
        for (int k = 0; k < AK; k++) wts[k] *= sc2;
    }

    /* routed experts, batched.
     *
     * The 8 experts are independent: same shapes, different weight slices, and
     * all read the same `cur`. Issuing them as individual matmul() calls costs
     * 24 pool wakes per layer (~1000 per token over 42 layers). Instead:
     *
     *   phase 1: all 8 gate and all 8 up GEMMs, in one dispatch
     *            (16 independent jobs, gate into gate_batch[k], up into
     *             up_batch[k])
     *   phase 2: SwiGLU + clamp for each expert, into gate_batch[k]
     *            (moved out of the per-expert loop so it runs once per k)
     *   phase 3: all 8 down GEMMs, in one dispatch, into down_batch[k]
     *   phase 4: accumulate w[k] * down_batch[k] in fixed k order
     *
     * ffn_gate_exps / ffn_up_exps ne = {D, FF, NE}: per-expert stride is
     * FF * row_size(type, D). ffn_down_exps ne = {FF, D, NE}: per-expert
     * stride is D * row_size(type, FF).
     *
     * The accumulation order (k ascending) is unchanged, so this is
     * bit-identical to the per-expert loop. */
    memset(out, 0, D * sizeof(float));
    {
        const size_t gate_stride = (size_t)FF * gguf_type_row_size(lw->type_ffn_gate_exps, D);
        const size_t down_stride = (size_t)D * gguf_type_row_size(lw->type_ffn_down_exps, FF);
        float *gate_b = s->gate_batch;                  /* [FF][AK] */
        float *up_b   = s->up_batch;                    /* [FF][AK] */
        float *down_b = s->down_batch;                  /* [D][AK]  */

        mm_job_t jobs[2 * 16];
        int nj = 0;
        for (int k = 0; k < AK; k++) {
            if (wts[k] == 0.f) continue;
            const int e = sel[k];
            jobs[nj].out   = gate_b + (size_t)k * FF;
            jobs[nj].x     = cur;
            jobs[nj].W     = (const uint8_t *)lw->ffn_gate_exps + (size_t)e * gate_stride;
            jobs[nj].n     = D;
            jobs[nj].d     = FF;
            jobs[nj].qtype = lw->type_ffn_gate_exps;
            nj++;
            jobs[nj].out   = up_b + (size_t)k * FF;
            jobs[nj].x     = cur;
            jobs[nj].W     = (const uint8_t *)lw->ffn_up_exps + (size_t)e * gate_stride;
            jobs[nj].n     = D;
            jobs[nj].d     = FF;
            jobs[nj].qtype = lw->type_ffn_up_exps;
            nj++;
        }
        if (!tensor_matmul_multi(jobs, nj)) {
            for (int i = 0; i < nj; i++)
                matmul(jobs[i].out, jobs[i].x, jobs[i].W, jobs[i].n, jobs[i].d, jobs[i].qtype);
        }

        /* SwiGLU (+ optional clamp) per expert, in place into gate_b[k]. */
        const int moe_dbg = moe_dbg_on;
        for (int k = 0; k < AK; k++) {
            if (wts[k] == 0.f) continue;
            float *g_buf = gate_b + (size_t)k * FF;
            const float *u_buf = up_b + (size_t)k * FF;
            float gmax = -INFINITY, gmin = INFINITY, umax = -INFINITY, umin = INFINITY;
            for (int i = 0; i < FF; i++) {
                float g = g_buf[i];
                float u = u_buf[i];
                if (moe_dbg) {
                    if (g > gmax) gmax = g;
                    if (g < gmin) gmin = g;
                    if (u > umax) umax = u;
                    if (u < umin) umin = u;
                }
                if (clamp_v > 0.0f) {
                    g = fminf(g, clamp_v);
                    u = fminf(fmaxf(u, -clamp_v), clamp_v);
                }
                g_buf[i] = g / (1.0f + expf(-g)) * u;
            }
            if (moe_dbg && (gmax > 5.f || gmin < -5.f || umax > 5.f || umin < -5.f)) {
                double ss = 0.f;
                for (int i = 0; i < FF; i++) ss += (double)g_buf[i] * g_buf[i];
                fprintf(stderr, "[moe glue] l%d k=%d e=%d w=%.4f glue_rms=%.4f g_max=%.4f g_min=%.4f u_max=%.4f u_min=%.4f\n",
                        layer_idx, k, sel[k], wts[k], sqrt(ss / FF), gmax, gmin, umax, umin);
            }
        }

        nj = 0;
        for (int k = 0; k < AK; k++) {
            if (wts[k] == 0.f) continue;
            const int e = sel[k];
            jobs[nj].out   = down_b + (size_t)k * D;
            jobs[nj].x     = gate_b + (size_t)k * FF;
            jobs[nj].W     = (const uint8_t *)lw->ffn_down_exps + (size_t)e * down_stride;
            jobs[nj].n     = FF;
            jobs[nj].d     = D;
            jobs[nj].qtype = lw->type_ffn_down_exps;
            nj++;
        }
        if (!tensor_matmul_multi(jobs, nj)) {
            for (int i = 0; i < nj; i++)
                matmul(jobs[i].out, jobs[i].x, jobs[i].W, jobs[i].n, jobs[i].d, jobs[i].qtype);
        }

        for (int k = 0; k < AK; k++) {
            if (wts[k] == 0.f) continue;
            const float w = wts[k];
            const float *d_buf = down_b + (size_t)k * D;
            for (int i = 0; i < D; i++) out[i] += w * d_buf[i];
        }
    }

    glm5_stats("x.moe.routed", out, D);

    /* PICOLM_DBG_MOE_OUT=1: report large routed-expert contributions so the
     * "expert explosion" (one SwiGLU outlier dominating the sum) can be
     * localised per layer/position without drowning in per-layer noise. */
    if (moe_dbg_on) {
        float omax = 0.f;
        int omi = -1;
        for (int i = 0; i < D; i++) {
            float a = fabsf(out[i]);
            if (a > omax) { omax = a; omi = i; }
        }
        if (omax > 5.f) {
            float orms = 0.f;
            double ss = 0;
            for (int i = 0; i < D; i++) ss += (double)out[i] * out[i];
            orms = (float)sqrt(ss / D);
            fprintf(stderr, "[moe out] l%d pos? out_max=%.2f at[%d] rms=%.4f\n",
                    layer_idx, omax, omi, orms);
        }
    }

    /* shared expert (no sigmoid gate for GLM-5) */
    if (lw->ffn_gate_shexp) {
        glm5_moe_shared_expert(m, lw, s, cur, out, SFF);
        snprintf(t, sizeof(t), "x.moe.shexp");
        glm5_stats(t, s->glm5.moe_down, D);
    }
    glm5_stats("x.moe.out", out, D);
    {
        char tc[96];
        snprintf(tc, sizeof tc, "x.moe.out-l%d", layer_idx);
        glm5_chk(tc, out, D, -1e4f, 1e4f);
    }
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
static void glm5_moe_ffn_batch_ref(const model_t *m, const glm5_layer_weights_t *gw,
                               const layer_weights_t *lw, run_state_t *s,
                               const float *cur_b, float *out_b, int n_tokens, int layer_idx)
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
            glm5_moe_ffn(m, gw, lw, s, cur_b + (size_t)t*D, out_b + (size_t)t*D, layer_idx);
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
        if (getenv("PICOLM_DBG_HCVEC") && atoi(getenv("PICOLM_DBG_HCVEC")) == layer_idx) {
            for (int q=0;q<D;q++) s->glm5.moe_down[q]=out_b[q];   /* save routed */
        }
        for (int t = 0; t < NT; t++)
            glm5_moe_shared_expert(m, lw, s, cur_b + (size_t)t*D, out_b + (size_t)t*D, SFF);
        if (getenv("PICOLM_DBG_HCVEC") && atoi(getenv("PICOLM_DBG_HCVEC")) == layer_idx) {
            double sr=0, ss2=0;
            for (int q=0;q<D;q++) { sr+=(double)s->glm5.moe_down[q]*s->glm5.moe_down[q];
                                    double sh=out_b[q]-s->glm5.moe_down[q]; ss2+=sh*sh; }
            fprintf(stderr, "[moechk L%d ref] routed=%.6f shared=%.6f total=%.6f\n", layer_idx,
                    (float)sqrt(sr/D), (float)sqrt(ss2/D),
                    (float)sqrt((double)((sr+ss2)/D)));
        }
    }

    free(sel_all); free(wts_all); free(probs); free(count); free(cursor);
    free(bucket); free(gu); free(gu2); free(dwn); free(slot_out); free(xbuf);
}

/* ---- Expert-parallel MoE prefill (PICOLM_GLM5_MOE_EP=1) -------------------------------------
 *
 * glm5_moe_ffn_batch_ref() runs the experts one after another: per expert it gathers the token
 * rows, quantizes them (twice: gate and up), and issues three matmul_batch() calls, each of which
 * is its own pool dispatch whose thread count is clamped to n_batch (so an expert with 4 tokens
 * runs its GEMM on 4 of 24 threads). Between the dispatches the main thread does the SwiGLU, the
 * copies and the quantization while the other workers spin. That is ~768 dispatches per layer,
 * each with only ~0.1 ms of parallel work, plus ~300 MB of malloc/free churn (page faults) per
 * layer, plus routing and the shared expert run one token at a time.
 *
 * This version is the same math in about eight dispatches:
 *   route (parallel over tokens)  ->  bucket + gather (parallel)  ->  gate/up for ALL experts
 *   as one tensor_matmul_batch_multi  ->  SwiGLU (parallel over rows)  ->  down for all experts
 *   as one call  ->  reduce in slot order (parallel over tokens)  ->  shared expert, batched.
 * Every per-row/per-column result is computed by the same kernel as before (the multi call is
 * bit-identical to matmul_batch, see tests/test_batch_multi.c), experts with a single token still
 * go through matmul() exactly like the reference, and the slot-order reduction is unchanged, so
 * the routed part is bit-identical to the reference. Only the shared expert differs when batched:
 * a GEMM instead of 512 GEMVs reads its weights once instead of once per token (~10 GB/layer at
 * 512 tokens) but sums in a different order; PICOLM_GLM5_SHEXP_BATCH=0 keeps it per token.
 * PICOLM_GLM5_MOE_PROF=1 prints per-phase times for both this path and the reference. */
/* Default ON. The expert-parallel batched MoE prefill is bit-identical to the
 * reference for the routed part (see the long note above) and falls back to the
 * reference if it cannot run, so it is safe as a default.
 * PICOLM_GLM5_MOE_EP=0 forces the reference path. */
static int glm5_moe_ep_enabled(void) {
    static int cached = -1;
    if (cached < 0) { const char *e = getenv("PICOLM_GLM5_MOE_EP"); cached = !(e && atoi(e) == 0); }
    return cached;
}
static int glm5_shexp_batch_enabled(void) {
    static int cached = -1;
    if (cached < 0) { const char *e = getenv("PICOLM_GLM5_SHEXP_BATCH"); cached = !(e && atoi(e) == 0); }
    return cached;
}

enum { EPP_ROUTE, EPP_BUCKET, EPP_GATEUP, EPP_SWIGLU, EPP_DOWN, EPP_SINGLES, EPP_REDUCE, EPP_SHARED, EPP_TOTAL, EPP_N };
static double g_epp_ms[EPP_N]; static long g_epp_calls; static int g_epp_mode = -1;  /* 0 ref, 1 ep */
static int glm5_moe_prof_on(void) {
    static int cached = -1;
    if (cached < 0) { const char *e = getenv("PICOLM_GLM5_MOE_PROF"); cached = (e && atoi(e) > 0) ? 1 : 0; }
    return cached;
}
static void glm5_moe_prof_print(void) {
    if (!g_epp_calls) return;
    static const char *nm[EPP_N] = { "route", "bucket+gather", "gate/up", "swiglu", "down", "single-token experts", "reduce", "shared expert", "TOTAL" };
    fprintf(stderr, "\nMoE prefill (%s path), %ld layer-calls, mean ms per call:\n", g_epp_mode ? "expert-parallel" : "reference", g_epp_calls);
    for (int i = 0; i < EPP_N; i++)
        if (g_epp_ms[i] > 0.0) fprintf(stderr, "  %-22s %9.2f\n", nm[i], g_epp_ms[i] / g_epp_calls);
}
#define EPP_BEGIN(t) double t = glm5_moe_prof_on() ? glm5_prof_now() : 0.0
#define EPP_END(t, ph) do { if (t) g_epp_ms[ph] += glm5_prof_now() - t; } while (0)

typedef struct {
    const float *cur_b; int D, NE, AK;
    const layer_weights_t *lw; const float *bias; float moe_scale;
    int *sel; float *wts;
} ep_route_t;
static void ep_route_task(int t, void *vp) {
    const ep_route_t *x = (const ep_route_t *)vp;
    const int NE = x->NE, AK = x->AK;
    float logits[512], probs[512];
    matmul(logits, x->cur_b + (size_t)t * x->D, x->lw->ffn_gate_inp, x->D, NE, x->lw->type_ffn_gate_inp);
    for (int i = 0; i < NE; i++) logits[i] = 1.0f / (1.0f + expf(-logits[i]));
    memcpy(probs, logits, (size_t)NE * sizeof(float));
    if (x->bias) for (int i = 0; i < NE; i++) logits[i] += x->bias[i];
    int *sel = x->sel + (size_t)t * AK; float *wts = x->wts + (size_t)t * AK;
    for (int k = 0; k < AK; k++) {
        int best = -1; float bv = -1e30f;
        for (int i = 0; i < NE; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        sel[k] = best;
        logits[best] = -1e30f;
    }
    float sum = 0.f;
    for (int k = 0; k < AK; k++) { wts[k] = probs[sel[k]]; sum += wts[k]; }
    float sc2 = (1.0f / sum) * x->moe_scale;
    for (int k = 0; k < AK; k++) wts[k] *= sc2;
}

typedef struct { const float *cur_b; const int *bucket; float *xg; int D, AK, rows; } ep_gather_t;
static void ep_gather_task(int b, void *vp) {
    const ep_gather_t *x = (const ep_gather_t *)vp;
    const int r0 = b * 32, r1 = (r0 + 32 < x->rows) ? r0 + 32 : x->rows;
    for (int r = r0; r < r1; r++)
        memcpy(x->xg + (size_t)r * x->D, x->cur_b + (size_t)(x->bucket[r] / x->AK) * x->D, (size_t)x->D * sizeof(float));
}

typedef struct { float *g, *u; int FF, rows; float clamp_v; } ep_swiglu_t;
static void ep_swiglu_task(int b, void *vp) {
    const ep_swiglu_t *x = (const ep_swiglu_t *)vp;
    const int r0 = b * 8, r1 = (r0 + 8 < x->rows) ? r0 + 8 : x->rows;
    for (int r = r0; r < r1; r++) {
        float *g = x->g + (size_t)r * x->FF, *u = x->u + (size_t)r * x->FF;
        for (int j = 0; j < x->FF; j++) {
            float gv = g[j], uv = u[j];
            if (x->clamp_v > 0.0f) { gv = fminf(gv, x->clamp_v); uv = fminf(fmaxf(uv, -x->clamp_v), x->clamp_v); }
            g[j] = gv / (1.0f + expf(-gv)) * uv;
        }
    }
}

typedef struct { const layer_weights_t *lw; const int *single; const int *cursor;
                 const float *xg; float *gu, *gu2, *dwn; size_t gate_stride, down_stride;
                 int D, FF; int phase; } ep_single_t;
static void ep_single_task(int i, void *vp) {
    const ep_single_t *x = (const ep_single_t *)vp;
    const int e = x->single[i], pos = x->cursor[e];
    if (x->phase == 0) {
        const void *gate_w = (const uint8_t *)x->lw->ffn_gate_exps + (size_t)e * x->gate_stride;
        const void *up_w   = (const uint8_t *)x->lw->ffn_up_exps   + (size_t)e * x->gate_stride;
        matmul(x->gu  + (size_t)pos * x->FF, x->xg + (size_t)pos * x->D, gate_w, x->D, x->FF, x->lw->type_ffn_gate_exps);
        matmul(x->gu2 + (size_t)pos * x->FF, x->xg + (size_t)pos * x->D, up_w,   x->D, x->FF, x->lw->type_ffn_up_exps);
    } else {
        const void *down_w = (const uint8_t *)x->lw->ffn_down_exps + (size_t)e * x->down_stride;
        matmul(x->dwn + (size_t)pos * x->D, x->gu + (size_t)pos * x->FF, down_w, x->FF, x->D, x->lw->type_ffn_down_exps);
    }
}

typedef struct { const float *wts; const int *inv; const float *dwn; float *out_b; int D, AK; } ep_reduce_t;
static void ep_reduce_task(int t, void *vp) {
    const ep_reduce_t *x = (const ep_reduce_t *)vp;
    float *o = x->out_b + (size_t)t * x->D;
    memset(o, 0, (size_t)x->D * sizeof(float));
    for (int k = 0; k < x->AK; k++) {
        const float w = x->wts[t * x->AK + k];
        const float *dd = x->dwn + (size_t)x->inv[t * x->AK + k] * x->D;
        for (int j = 0; j < x->D; j++) o[j] += w * dd[j];
    }
}

typedef struct { float *out_b; const float *add; int D; } ep_add_t;
static void ep_add_task(int t, void *vp) {
    const ep_add_t *x = (const ep_add_t *)vp;
    float *o = x->out_b + (size_t)t * x->D; const float *a = x->add + (size_t)t * x->D;
    for (int i = 0; i < x->D; i++) o[i] += a[i];
}

static int ep_grow(void **p, size_t *cap, size_t bytes) {
    if (bytes <= *cap) return 1;
    void *n = realloc(*p, bytes);
    if (!n) return 0;
    *p = n; *cap = bytes;
    return 1;
}

/* Returns 1 if it handled the layer, 0 to fall back to the reference path (nothing was written). */
static int glm5_moe_ffn_batch_ep(const model_t *m, const glm5_layer_weights_t *gw,
                                 const layer_weights_t *lw, run_state_t *s,
                                 const float *cur_b, float *out_b, int NT, int layer_idx)
{
    const model_config_t *c = &m->config;
    const int D = c->n_embd, NE = c->n_expert, AK = c->n_expert_used, FF = c->n_ff_exp;
    const float clamp_v = glm5_clamp_value(c);
    if (NT < 2 || NE > 512 || NE < 1 || AK < 1) return 0;
    if (!tensor_matmul_batch_multi_ok(lw->type_ffn_gate_exps, D, FF) ||
        !tensor_matmul_batch_multi_ok(lw->type_ffn_up_exps, D, FF) ||
        !tensor_matmul_batch_multi_ok(lw->type_ffn_down_exps, FF, D)) return 0;

    /* persistent buffers: no malloc/free (and page faults) per layer */
    static int *sel, *count, *cursor, *bucket, *inv, *single, *multi; static float *wts, *xg, *gu, *gu2, *dwn;
    static mm_bjob_t *jobs;
    static size_t c_sel, c_cnt, c_cur, c_bkt, c_inv, c_sgl, c_mul, c_wts, c_xg, c_gu, c_gu2, c_dwn, c_jobs;
    const size_t pairs = (size_t)NT * AK;
    if (!ep_grow((void **)&sel, &c_sel, pairs * sizeof(int)) || !ep_grow((void **)&wts, &c_wts, pairs * sizeof(float)) ||
        !ep_grow((void **)&count, &c_cnt, (size_t)(NE + 1) * sizeof(int)) ||
        !ep_grow((void **)&cursor, &c_cur, (size_t)(NE + 1) * sizeof(int)) ||
        !ep_grow((void **)&single, &c_sgl, (size_t)(NE + 1) * sizeof(int)) ||
        !ep_grow((void **)&multi, &c_mul, (size_t)(NE + 1) * sizeof(int)) ||
        !ep_grow((void **)&bucket, &c_bkt, pairs * sizeof(int)) || !ep_grow((void **)&inv, &c_inv, pairs * sizeof(int)) ||
        !ep_grow((void **)&xg, &c_xg, pairs * D * sizeof(float)) || !ep_grow((void **)&gu, &c_gu, pairs * FF * sizeof(float)) ||
        !ep_grow((void **)&gu2, &c_gu2, pairs * FF * sizeof(float)) || !ep_grow((void **)&dwn, &c_dwn, pairs * D * sizeof(float)) ||
        !ep_grow((void **)&jobs, &c_jobs, (size_t)2 * (NE + 1) * sizeof(mm_bjob_t))) return 0;

    EPP_BEGIN(t_all);
    /* 1. routing, one token per task (matmul() runs serially inside a task: same per-row kernel) */
    EPP_BEGIN(t0);
    { ep_route_t rc = { cur_b, D, NE, AK, lw, gw->exp_probs_b ? (const float *)gw->exp_probs_b : NULL,
                        c->glm5_moe_scale, sel, wts };
      tensor_parallel_for_dyn(NT, ep_route_task, &rc); }
    EPP_END(t0, EPP_ROUTE);

    /* 2. bucket (t, slot) by expert, remembering where each pair landed (inv) */
    EPP_BEGIN(t1);
    memset(count, 0, (size_t)(NE + 1) * sizeof(int));
    for (size_t i = 0; i < pairs; i++) count[sel[i]]++;
    { int acc = 0; for (int e = 0; e < NE; e++) { cursor[e] = acc; acc += count[e]; } cursor[NE] = acc; }
    { int *fill = (int *)calloc((size_t)NE + 1, sizeof(int)); if (!fill) return 0;
      for (int t = 0; t < NT; t++) for (int k = 0; k < AK; k++) {
          const int e = sel[t * AK + k], pos = cursor[e] + fill[e]++;
          bucket[pos] = t * AK + k; inv[t * AK + k] = pos; }
      free(fill); }
    int n_single = 0, n_multi = 0;
    for (int e = 0; e < NE; e++) { if (count[e] == 1) single[n_single++] = e; else if (count[e] >= 2) multi[n_multi++] = e; }
    { ep_gather_t gc = { cur_b, bucket, xg, D, AK, (int)pairs };
      tensor_parallel_for_dyn(((int)pairs + 31) / 32, ep_gather_task, &gc); }
    EPP_END(t1, EPP_BUCKET);

    const size_t gate_stride = (size_t)FF * gguf_type_row_size(lw->type_ffn_gate_exps, D);
    const size_t down_stride = (size_t)D  * gguf_type_row_size(lw->type_ffn_down_exps, FF);
    ep_single_t sc = { lw, single, cursor, xg, gu, gu2, dwn, gate_stride, down_stride, D, FF, 0 };

    /* 3. gate/up: single-token experts via matmul() (as the reference), the rest in ONE call */
    EPP_BEGIN(t2);
    if (n_single) { sc.phase = 0; tensor_parallel_for_dyn(n_single, ep_single_task, &sc); }
    EPP_END(t2, EPP_SINGLES);
    EPP_BEGIN(t3);
    { int nj = 0;
      for (int i = 0; i < n_multi; i++) {
          const int e = multi[i], pos = cursor[e];
          jobs[nj++] = (mm_bjob_t){ gu  + (size_t)pos * FF, xg + (size_t)pos * D,
                                    (const uint8_t *)lw->ffn_gate_exps + (size_t)e * gate_stride, count[e], D, FF, lw->type_ffn_gate_exps };
          jobs[nj++] = (mm_bjob_t){ gu2 + (size_t)pos * FF, xg + (size_t)pos * D,
                                    (const uint8_t *)lw->ffn_up_exps   + (size_t)e * gate_stride, count[e], D, FF, lw->type_ffn_up_exps };
      }
      if (nj && !tensor_matmul_batch_multi(jobs, nj))
          for (int j = 0; j < nj; j++) matmul_batch(jobs[j].out, jobs[j].x, jobs[j].n_batch, jobs[j].W, jobs[j].n, jobs[j].d, jobs[j].qtype); }
    EPP_END(t3, EPP_GATEUP);

    /* 4. SwiGLU over every sorted row */
    EPP_BEGIN(t4);
    { ep_swiglu_t sw = { gu, gu2, FF, (int)pairs, clamp_v };
      tensor_parallel_for_dyn(((int)pairs + 7) / 8, ep_swiglu_task, &sw); }
    EPP_END(t4, EPP_SWIGLU);

    /* 5. down: one call for all multi-token experts, then the single-token ones */
    EPP_BEGIN(t5);
    { int nj = 0;
      for (int i = 0; i < n_multi; i++) {
          const int e = multi[i], pos = cursor[e];
          jobs[nj++] = (mm_bjob_t){ dwn + (size_t)pos * D, gu + (size_t)pos * FF,
                                    (const uint8_t *)lw->ffn_down_exps + (size_t)e * down_stride, count[e], FF, D, lw->type_ffn_down_exps };
      }
      if (nj && !tensor_matmul_batch_multi(jobs, nj))
          for (int j = 0; j < nj; j++) matmul_batch(jobs[j].out, jobs[j].x, jobs[j].n_batch, jobs[j].W, jobs[j].n, jobs[j].d, jobs[j].qtype); }
    EPP_END(t5, EPP_DOWN);
    EPP_BEGIN(t6);
    if (n_single) { sc.phase = 1; tensor_parallel_for_dyn(n_single, ep_single_task, &sc); }
    EPP_END(t6, EPP_SINGLES);

    /* 6. reduce slots in order 0..AK-1 per token, exactly like the reference */
    EPP_BEGIN(t7);
    { ep_reduce_t rc = { wts, inv, dwn, out_b, D, AK };
      tensor_parallel_for_dyn(NT, ep_reduce_task, &rc); }
    EPP_END(t7, EPP_REDUCE);

    /* 7. shared expert */
    EPP_BEGIN(t8);
    if (getenv("PICOLM_DBG_HCVEC") && atoi(getenv("PICOLM_DBG_HCVEC")) == layer_idx) {
        double ss=0; for (int q=0;q<D;q++) ss+=(double)out_b[q]*out_b[q];
        fprintf(stderr, "[moechk L%d] routed-only rms=%.6f\n", layer_idx, (float)sqrt(ss/D));
    }
    if (lw->ffn_gate_shexp) {
        const int SFF = c->n_ff_shexp;
        if (glm5_shexp_batch_enabled()) {
            static float *shg, *shu, *shd; static size_t c_shg, c_shu, c_shd;
            if (ep_grow((void **)&shg, &c_shg, (size_t)NT * SFF * sizeof(float)) &&
                ep_grow((void **)&shu, &c_shu, (size_t)NT * SFF * sizeof(float)) &&
                ep_grow((void **)&shd, &c_shd, (size_t)NT * D * sizeof(float))) {
                matmul_batch(shg, cur_b, NT, lw->ffn_gate_shexp, D, SFF, lw->type_ffn_gate_shexp);
                matmul_batch(shu, cur_b, NT, lw->ffn_up_shexp,   D, SFF, lw->type_ffn_up_shexp);
                { ep_swiglu_t sw = { shg, shu, SFF, NT, glm5_clamp_value_shexp(c) };
                  tensor_parallel_for_dyn((NT + 7) / 8, ep_swiglu_task, &sw); }
                matmul_batch(shd, shg, NT, lw->ffn_down_shexp, SFF, D, lw->type_ffn_down_shexp);
                { ep_add_t ac = { out_b, shd, D }; tensor_parallel_for_dyn(NT, ep_add_task, &ac); }
            } else {
                for (int t = 0; t < NT; t++) glm5_moe_shared_expert(m, lw, s, cur_b + (size_t)t * D, out_b + (size_t)t * D, SFF);
            }
        } else {
            for (int t = 0; t < NT; t++) glm5_moe_shared_expert(m, lw, s, cur_b + (size_t)t * D, out_b + (size_t)t * D, SFF);
        }
    }
    EPP_END(t8, EPP_SHARED);
    if (getenv("PICOLM_DBG_HCVEC") && atoi(getenv("PICOLM_DBG_HCVEC")) == layer_idx) {
        const char *dd = getenv("PICOLM_DUMP_DIR");
        char fn[512];
        snprintf(fn, sizeof(fn), "%s/moe_out-%d.f32", dd ? dd : "/tmp/picd", layer_idx);
        FILE *fp = fopen(fn, "wb");
        if (fp) { fwrite(out_b, sizeof(float), (size_t)m->config.n_embd, fp); fclose(fp); }
        double ss=0; for (int q=0;q<m->config.n_embd;q++) ss+=(double)out_b[q]*out_b[q];
        fprintf(stderr, "[moechk L%d] out_b rms=%.6f (routed+shared)\n", layer_idx, (float)sqrt(ss/m->config.n_embd));
    }
    EPP_END(t_all, EPP_TOTAL);
    return 1;
}

/* Dispatcher: the expert-parallel path when enabled and applicable, else the reference. */
static void glm5_moe_ffn_batch(const model_t *m, const glm5_layer_weights_t *gw,
                               const layer_weights_t *lw, run_state_t *s,
                               const float *cur_b, float *out_b, int n_tokens, int layer_idx)
{
    const int prof = glm5_moe_prof_on();
    if (prof && g_epp_calls == 0) atexit(glm5_moe_prof_print);
    if (getenv("PICOLM_DBG_HCVEC") && atoi(getenv("PICOLM_DBG_HCVEC")) == layer_idx) {
        double ss=0; for (int q=0;q<m->config.n_embd;q++) ss+=(double)cur_b[q]*cur_b[q];
        fprintf(stderr, "[moechk L%d] moe_in rms=%.6f  ep_enabled=%d\n", layer_idx,
                (float)sqrt(ss/m->config.n_embd), glm5_moe_ep_enabled());
    }
    if (glm5_moe_ep_enabled() && glm5_moe_ffn_batch_ep(m, gw, lw, s, cur_b, out_b, n_tokens, layer_idx)) {
        if (prof) { g_epp_mode = 1; g_epp_calls++; }
        /* Report the batched MoE output so a prefill can be diffed against
         * ik's ffn_moe_out-<il> at all; without this the whole FFN was
         * invisible on the batched path and only the per-token decode path
         * emitted x.moe.* tags. */
        if (glm5_stats_wanted()) {
            char t2[96];
            snprintf(t2, sizeof(t2), "l%02d.moe.out", layer_idx);
            glm5_stats(t2, out_b, m->config.n_embd);
        }
        return;
    }
    const double t0 = prof ? glm5_prof_now() : 0.0;
    glm5_moe_ffn_batch_ref(m, gw, lw, s, cur_b, out_b, n_tokens, layer_idx);
    if (prof) { g_epp_mode = 0; g_epp_calls++; g_epp_ms[EPP_TOTAL] += glm5_prof_now() - t0; }
    if (glm5_stats_wanted()) {
        char t2[96];
        snprintf(t2, sizeof(t2), "l%02d.moe.out", layer_idx);
        glm5_stats(t2, out_b, m->config.n_embd);
    }
    {
        const char *e = getenv("PICOLM_DBG_HCVEC");
        if (e && atoi(e) == layer_idx) {
            const char *dd = getenv("PICOLM_DUMP_DIR");
            char fn[512];
            double ss=0; for (int q=0;q<m->config.n_embd;q++) ss+=(double)out_b[q]*out_b[q];
            fprintf(stderr, "[moechk L%d ref] out_b rms=%.6f\n", layer_idx, (float)sqrt(ss/m->config.n_embd));
            snprintf(fn, sizeof(fn), "%s/moe_out-%d.f32", dd ? dd : "/tmp/picd", layer_idx);
            FILE *fp = fopen(fn, "wb");
            if (fp) { fwrite(out_b, sizeof(float), (size_t)m->config.n_embd, fp); fclose(fp); }
        }
    }
}

/* ---- Batched dense FFN (the first n_layer_dense_lead layers) --------------------------------
 * glm5_dense_ffn() is a GEMV per token. With FF = n_ffn (12288) that re-reads ~127 MB of weights
 * for every token: ~50 GB (~0.9 s) per dense layer at 400-500 tokens, which is more than a whole
 * MoE layer now costs. Batching the tokens reads the weights once. A GEMM sums in a different
 * order than a GEMV, so the result differs at float rounding (~1e-7 relative, same as the batched
 * shared expert).
 *
 * ON by default; PICOLM_GLM5_DENSE_BATCH=0 forces the per-token loop (the GEMV-exact reference,
 * useful when diffing an argmax sequence against another engine).
 * Returns 1 if it ran; 0 means "use the per-token loop" (also what it returns for NT < 2, so
 * single-token decode is unaffected either way). */
static int glm5_dense_batch_enabled(void) {
    static int cached = -1;
    if (cached < 0) { const char *e = getenv("PICOLM_GLM5_DENSE_BATCH"); cached = !(e && atoi(e) == 0); }
    return cached;
}
static int glm5_dense_ffn_batch(const model_t *m, const layer_weights_t *lw, const float *cur_b, float *out_b, int NT)
{
    const model_config_t *c = &m->config;
    const int D = c->n_embd, FF = c->n_ffn;
    if (NT < 2 || FF < 1) return 0;
    static float *g, *u; static size_t cg, cu;
    if (!ep_grow((void **)&g, &cg, (size_t)NT * FF * sizeof(float)) ||
        !ep_grow((void **)&u, &cu, (size_t)NT * FF * sizeof(float))) return 0;
    matmul_batch(g, cur_b, NT, lw->ffn_gate, D, FF, lw->type_ffn_gate);
    matmul_batch(u, cur_b, NT, lw->ffn_up,   D, FF, lw->type_ffn_up);
    { ep_swiglu_t sw = { g, u, FF, NT, glm5_clamp_value(c) };
      tensor_parallel_for_dyn((NT + 7) / 8, ep_swiglu_task, &sw); }
    matmul_batch(out_b, g, NT, lw->ffn_down, FF, D, lw->type_ffn_down);
    return 1;
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
            glm5_moe_ffn(m, gw, lw, s, s->glm5.cur, s->glm5.attn_out, il);
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
        if (glm5_stats_wanted()) {
            glm5_stats("head_in", s->glm5.cur, D);
            double mu = 0.0;
            for (int i = 0; i < D; i++) mu += s->glm5.cur[i];
            fprintf(stderr, "GLM5 STAT head_in_mean n=1 mean=%.9f\n", mu / D);
        }
        float inv = 1.0f / sqrtf(ss / D + c->rms_norm_eps);
        const float *nw = s->output_norm_w;   /* predequantized output_norm */
        for (int i = 0; i < D; i++) s->glm5.cur[i] *= inv * nw[i];
        if (glm5_stats_wanted()) glm5_stats("head_normed", s->glm5.cur, D);
        double _t_head = glm5_prof_mode() ? glm5_prof_now() : 0.0;
        matmul(s->logits, s->glm5.cur, w->output, D, c->vocab_size, w->type_output);
        if (_t_head) glm5_prof_add(GP_HEAD, _t_head);
    }
    if (glm5_prof_mode()) {
        if (g_gp.tokens == 0) {
            atexit(glm5_prof_print);
            tensor_pool_gap_reset();   /* drop prefault/prefill dispatches and token 1 */
        }
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

/* ---- Head-parallel KDA core -------------------------------------------
 *
 * glm5_kda_tail() walks all H heads serially on the calling thread. In decode
 * that leaves the other pool workers spinning for ~2 ms per KDA layer (g/decay
 * expf, conv1d+SiLU over 3*DI channels, and a 4 MB state read+write at
 * single-thread bandwidth). Every step is local to one head: head h only
 * touches its own slices of q/k/v/raw/z, its own conv channels, its own
 * KD*KD state block and its own head_out slice. So one task per head, with the
 * per-head arithmetic copied unchanged from glm5_kda_tail(), is bit-identical
 * (no reduction crosses heads; the build is -std=c11, i.e. -ffp-contract=off).
 *
 * Same signature as glm5_kda_tail(); v_new is unused (per-head stack scratch). */
typedef struct {
    const model_config_t *c;
    const glm5_layer_weights_t *gw;
    int KD, DI, CONV_K, CONV_DIM;
    const float *q, *k, *v, *z, *raw, *beta_raw;
    float *conv_state, *kda_state;
    float *conv_out, *g, *head_out, *beta;
    int layer_idx;
} glm5_kda_head_ctx_t;

static void glm5_kda_head_task(int h, void *vp)
{
    const glm5_kda_head_ctx_t *x = (const glm5_kda_head_ctx_t *)vp;
    const model_config_t *c = x->c;
    const glm5_layer_weights_t *gw = x->gw;
    const int KD = x->KD, DI = x->DI, CONV_K = x->CONV_K, CONV_DIM = x->CONV_DIM;
    float *conv_out = x->conv_out, *g = x->g, *head_out = x->head_out;
    float *conv_state = x->conv_state;
    const float *raw = x->raw;

    /* beta = sigmoid(beta_raw) */
    x->beta[h] = 1.0f / (1.0f + expf(-x->beta_raw[h]));
    const float beta_h = x->beta[h];

    /* g[h*KD+ch] = lower_bound * sigmoid(-a[h] * (raw[idx] + dt_bias[idx])) */
    {
        const float *dt_bias = (const float *)gw->kda_dt_bias;
        const float a_val = ((const float *)gw->kda_a)[h];
        for (int ch = 0; ch < KD; ch++) {
            int idx = h*KD + ch;
            g[idx] = c->glm5_kda_lower_bound *
                     (1.0f / (1.0f + expf(a_val * (raw[idx] + dt_bias[idx]))));
        }
    }

    /* Causal depthwise conv1d + SiLU for this head's channels in each of the
     * q / k / v thirds of [q;k;v]. */
    {
        const float *wq = (const float *)gw->kda_conv_q;
        const float *wk = (const float *)gw->kda_conv_k;
        const float *wv = (const float *)gw->kda_conv_v;
        for (int part = 0; part < 3; part++) {
            const float *src = (part == 0) ? x->q : (part == 1) ? x->k : x->v;
            const float *w   = (part == 0) ? wq   : (part == 1) ? wk   : wv;
            for (int i = 0; i < KD; i++) {
                const int ch_off = h*KD + i;
                const int ch = part*DI + ch_off;
                float xs[GLM5_KD_MAX];  /* CONV_K <= 4 */
                for (int t = 0; t < CONV_K - 1; t++)
                    xs[t] = conv_state[t * CONV_DIM + ch];
                const float x_new = src[ch_off];
                xs[CONV_K - 1] = x_new;
                float acc = 0.f;
                for (int t = 0; t < CONV_K; t++)
                    acc += xs[t] * w[t + ch_off*CONV_K];
                if (h == 0 && part == 0 && i < 12 && getenv("PICOLM_DBG_CONVXY") && atoi(getenv("PICOLM_DBG_CONVXY")) == x->layer_idx) {
                    fprintf(stderr, "[convxy L%d ch=%d] acc=%.8f xs=[%.6f %.6f %.6f] w=[%.6f %.6f %.6f %.6f]\n",
                            x->layer_idx, ch_off, acc, xs[0], xs[1], xs[2], xs[3],
                            w[0+ch_off*CONV_K], w[1+ch_off*CONV_K], w[2+ch_off*CONV_K], w[3+ch_off*CONV_K]);
                }
                conv_out[ch] = acc / (1.0f + expf(-acc));
                for (int t = 0; t < CONV_K - 2; t++)
                    conv_state[t * CONV_DIM + ch] = conv_state[(t+1) * CONV_DIM + ch];
                conv_state[(CONV_K - 2) * CONV_DIM + ch] = x_new;

            }
        }
    }

    if (getenv("PICOLM_DBG_CONVXY") && atoi(getenv("PICOLM_DBG_CONVXY")) == x->layer_idx) {
        const char *dd = getenv("PICOLM_DUMP_DIR");
        char fn[512];
        /* one file per head: q[KD], k[KD], v[KD] for that head */
        snprintf(fn, sizeof(fn), "%s/conv_h%02d_%d.f32", dd ? dd : "/tmp/picd", h, x->layer_idx);
        FILE *fp = fopen(fn, "wb");
        if (fp) {
            fwrite(conv_out + h*KD,    sizeof(float), (size_t)KD, fp);
            fwrite(conv_out + DI+h*KD, sizeof(float), (size_t)KD, fp);
            fwrite(conv_out + 2*DI+h*KD, sizeof(float), (size_t)KD, fp);
            fclose(fp);
        }
    }

    /* Per-head L2 norm of q and k (after conv+SiLU) */
    {
        float *qh = conv_out + h*KD;
        float *kh = conv_out + DI + h*KD;
        float sq = 0.f, sk = 0.f;
        for (int ch = 0; ch < KD; ch++) { sq += qh[ch]*qh[ch]; sk += kh[ch]*kh[ch]; }
        float inv_q = 1.0f / fmaxf(sqrtf(sq), 1e-5f);
        float inv_k = 1.0f / fmaxf(sqrtf(sk), 1e-5f);
        for (int ch = 0; ch < KD; ch++) { qh[ch] *= inv_q; kh[ch] *= inv_k; }
    }

    /* Recurrent state update + output (verbatim from glm5_kda_tail). */
    {
        const float scale = 1.0f / sqrtf((float)KD);
        const float *qh = conv_out + h*KD;
        const float *kh = conv_out + DI + h*KD;
        const float *vh = conv_out + 2*DI + h*KD;
        float *Sh = x->kda_state + (size_t)h * KD * KD;
        float decay[GLM5_KD_MAX], v_new[GLM5_KD_MAX];

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
                float vn = beta_h * (vh[row] - vp[row]);
                v_new[row] = vn;
                ov[row] = ov[row] * scale + vn * attn_score;
            }
        }
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

    /* Per-head RMSNorm with ssm_norm weight, then sigmoid gate (z) */
    {
        const float *norm_w = (const float *)gw->kda_norm;
        float *oh = head_out + h*KD;
        float ss = 0.f;
        for (int ch = 0; ch < KD; ch++) ss += oh[ch]*oh[ch];
        float inv = 1.0f / sqrtf(ss / KD + c->rms_norm_eps);
        for (int ch = 0; ch < KD; ch++) oh[ch] *= inv * norm_w[ch];
        for (int ch = 0; ch < KD; ch++) {
            int i = h*KD + ch;
            head_out[i] *= 1.0f / (1.0f + expf(-x->z[i]));
        }
    }
}

static void glm5_kda_tail_par(
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
    (void)pos; (void)v_new;

    glm5_kda_head_ctx_t kc = {
        c, gw, KD, DI, c->ssm_d_conv, 3 * DI,
        q, k, v, z, raw, beta_raw,
        s->glm5.kda_conv_state[layer_idx], s->glm5.kda_state[layer_idx],
        conv_out, g, head_out, beta
    };
    tensor_parallel_for(H, glm5_kda_head_task, &kc);

    if (do_out_proj)
        matmul(out, head_out, lw->attn_output, DI, c->n_embd, lw->type_attn_output);
}

/* ---- Prefill: per-token serial loops, run in parallel ----------------------------------------
 *
 * The batched prefill batches the weight-bound GEMMs but left several loops per token on the main
 * thread while the other workers spin: hc_pre (rms over S*D doubles + a tiny threaded matmul)
 * and hc_post run twice per layer for each of the 512 tokens, and the KDA tail issued one
 * pool dispatch per token (512 per layer), re-streaming the 4 MB recurrent state through DRAM
 * for every token.
 *
 *  - hc_pre / hc_post: every token is independent. One task per token, with per-token scratch.
 *    matmul() inside a task runs serially with the same per-row kernel, so the results are
 *    bit-identical to the serial loop.
 *  - KDA tail: heads are independent, tokens within a head are sequential. One task per HEAD that
 *    walks all tokens in order, so a head's 64 KB state stays in L2 for the whole chunk and the
 *    layer needs one dispatch instead of 512. Per-head arithmetic is unchanged (it is the same
 *    glm5_kda_head_task), so the results are bit-identical.
 * PICOLM_GLM5_PF_SERIAL=1 restores the old loops (A/B, bisection).
 * PICOLM_GLM5_PFPROF=1 prints where a prefill layer's time goes, for KDA and MLA layers. */
static int glm5_pf_par(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("PICOLM_GLM5_PF_SERIAL") ? 0 : 1;
    return cached;
}

enum { PF_HC_ATTN_PRE, PF_ATTN, PF_KDA_PROJ, PF_KDA_TAIL, PF_KDA_OUT,
       PF_MLA_PROJ, PF_MLA_KB, PF_MLA_CORE, PF_MLA_VB, PF_MLA_OUT, PF_HC_ATTN_POST,
       PF_HC_FFN_PRE, PF_FFN, PF_HC_FFN_POST, PF_LAYER, PF_N };
static double g_pf_ms[2][PF_N];      /* [0] KDA / dense-attention layers, [1] MLA layers */
static long   g_pf_layers[2];
/* PICOLM_GLM5_PFPROF=1: summary at the end of every prefill call; =2: also one line per layer,
 * printed (and flushed) as soon as the layer finishes. Neither depends on atexit() handlers. */
static int glm5_pfprof_mode(void) {
    static int cached = -1;
    if (cached < 0) { const char *e = getenv("PICOLM_GLM5_PFPROF"); cached = e ? atoi(e) : 0; if (cached < 0) cached = 0; }
    return cached;
}
static int glm5_pfprof_on(void) { return glm5_pfprof_mode() > 0; }
static double g_pf_cur[PF_N];       /* this layer's regions */
static void glm5_pfprof_layer_line(int il, int kind) {
    if (glm5_pfprof_mode() >= 2) {
        const double *v = g_pf_cur;
        const double acc = v[PF_HC_ATTN_PRE] + v[PF_ATTN] + v[PF_HC_ATTN_POST] + v[PF_HC_FFN_PRE] + v[PF_FFN] + v[PF_HC_FFN_POST];
        fprintf(stderr, "PFPROF layer %2d %s %7.1f ms | hc_pre %6.1f  attn %7.1f", il, kind ? "MLA" : "KDA", v[PF_LAYER],
                v[PF_HC_ATTN_PRE], v[PF_ATTN]);
        if (!kind) fprintf(stderr, " (proj %6.1f tail %6.1f out %6.1f)", v[PF_KDA_PROJ], v[PF_KDA_TAIL], v[PF_KDA_OUT]);
        else fprintf(stderr, " (proj %6.1f kb %6.1f core %6.1f vb %6.1f out %6.1f)", v[PF_MLA_PROJ], v[PF_MLA_KB], v[PF_MLA_CORE], v[PF_MLA_VB], v[PF_MLA_OUT]);
        fprintf(stderr, "  hc_post %5.1f | hc_pre2 %6.1f  ffn %7.1f  hc_post2 %5.1f | unaccounted %6.1f\n",
                v[PF_HC_ATTN_POST], v[PF_HC_FFN_PRE], v[PF_FFN], v[PF_HC_FFN_POST], v[PF_LAYER] - acc);
        fflush(stderr);
    }
    memset(g_pf_cur, 0, sizeof g_pf_cur);
}
static void glm5_pfprof_print(void) {
    static const char *nm[PF_N] = { "hc_pre (attn) + norm", "attention block", "  kda projections", "  kda tail (conv+state)",
        "  kda out projection", "  mla q/kv projections", "  mla k_b absorb (64 heads)", "  mla attention core",
        "  mla v_b (64 heads)", "  mla out projection", "hc_post (attn)", "hc_pre (ffn) + norm", "ffn / moe", "hc_post (ffn)", "LAYER TOTAL" };
    static const char *kind[2] = { "KDA layers", "MLA layers" };
    for (int k = 0; k < 2; k++) {
        if (!g_pf_layers[k]) continue;
        const double tot = g_pf_ms[k][PF_LAYER] / g_pf_layers[k];
        fprintf(stderr, "\nPrefill layer breakdown, %s (%ld layers), mean ms per layer:\n", kind[k], g_pf_layers[k]);
        for (int r = 0; r < PF_N; r++) {
            if (g_pf_ms[k][r] <= 0.0) continue;
            const double v = g_pf_ms[k][r] / g_pf_layers[k];
            fprintf(stderr, "  %-26s %9.2f ms  %5.1f%%\n", nm[r], v, 100.0 * v / tot);
        }
    }
    memset(g_pf_ms, 0, sizeof g_pf_ms); memset(g_pf_layers, 0, sizeof g_pf_layers);
}
#define PFP_BEGIN(t) double t = glm5_pfprof_on() ? glm5_prof_now() : 0.0
#define PFP_END(t, kind, r) do { if (t) { const double _d = glm5_prof_now() - (t); g_pf_ms[kind][r] += _d; g_pf_cur[r] += _d; } } while (0)

typedef struct {
    const model_t *m; const glm5_layer_weights_t *gw; int is_attn;
    float *hc_x_b, *cur_b, *hc_res_b, *post_b, *comb_b, *scr_b; const float *norm_w;
    int S, D, M; float eps;
} pf_hcpre_t;
static void pf_hcpre_task(int t, void *vp) {
    const pf_hcpre_t *x = (const pf_hcpre_t *)vp;
    const int S = x->S, D = x->D;
    float *hc_x_t = x->hc_x_b + (size_t)t * S * D, *cur_t = x->cur_b + (size_t)t * D;
    float *res_t = x->hc_res_b + (size_t)t * S * D, *post_t = x->post_b + (size_t)t * S;
    float *comb_t = x->comb_b + (size_t)t * S * S;
    memcpy(res_t, hc_x_t, (size_t)S * D * sizeof(float));
    glm5_hc_pre(x->m, x->gw, x->is_attn, hc_x_t, cur_t, post_t, comb_t, x->scr_b + (size_t)t * ((size_t)S * D + x->M));
    rmsnorm(cur_t, cur_t, x->norm_w, D, x->eps);
}
/* The per-stage hc diagnostics (stats, layer dumps) are emitted per call, so they need the serial
 * loop. Decide that from the environment, NOT from g_hc_tag: decode sets the tag for every layer
 * and never clears it, so after the first generation (i.e. in any server, whose first request is
 * a decode) the tag is stale, and gating on it forced every later prefill onto the serial hc_pre
 * loop: ~20 ms per layer instead of ~2 at 128 tokens, 27% of the whole prefill. */
static int glm5_hc_diag_wanted(void) {
    static int cached = -1;
    if (cached < 0)
        cached = (glm5_stats_wanted() || getenv("PICOLM_DBG_DUMP_LAYER") || getenv("PICOLM_DBG_HCVEC")) ? 1 : 0;
    return cached;
}

/* Which token of the prefill batch the PICOLM_DBG_DUMP_LAYER / PICOLM_DBG_HCVEC dumps describe.
 * The ik oracle's l_out-<N> tensor holds the last token of its batch, so a prefill-vs-oracle
 * element-wise diff has to select the matching token. Default 0 (the historical behaviour). */
static int glm5_dump_tok(void) {
    static int cached = -2;
    if (cached == -2) { const char *e = getenv("PICOLM_DBG_DUMP_TOK"); cached = e ? atoi(e) : 0; }
    return cached;
}
/* 1 if it ran in parallel; 0 if the caller must run the serial loop (diagnostics on, no memory). */
static int pf_run_hcpre(const model_t *m, const glm5_layer_weights_t *gw, int is_attn, int n_tokens,
                        float *hc_x_b, float *cur_b, float *hc_res_b, float *post_b, float *comb_b,
                        const float *norm_w)
{
    if (glm5_hc_diag_wanted()) return 0;          /* per-stage diagnostics are per call: stay serial */
    const model_config_t *c = &m->config;
    const int S = c->glm5_hc, D = c->n_embd, M = (2 + S) * S;
    static float *scr; static size_t cap;
    const size_t need = (size_t)n_tokens * ((size_t)S * D + M) * sizeof(float);
    if (need > cap) { float *nb = (float *)realloc(scr, need); if (!nb) return 0; scr = nb; cap = need; }
    pf_hcpre_t x = { m, gw, is_attn, hc_x_b, cur_b, hc_res_b, post_b, comb_b, scr, norm_w, S, D, M, c->rms_norm_eps };
    tensor_parallel_for_dyn(n_tokens, pf_hcpre_task, &x);
    return 1;
}

typedef struct { const model_t *m; float *hc_x_b; const float *res_b, *post_b, *comb_b, *attn_b; int S, D; } pf_hcpost_t;
static void pf_hcpost_task(int t, void *vp) {
    const pf_hcpost_t *x = (const pf_hcpost_t *)vp;
    glm5_hc_post(x->m, x->attn_b + (size_t)t * x->D, x->res_b + (size_t)t * x->S * x->D,
                 x->post_b + (size_t)t * x->S, x->comb_b + (size_t)t * x->S * x->S,
                 x->hc_x_b + (size_t)t * x->S * x->D);
}
static int pf_run_hcpost(const model_t *m, int n_tokens, float *hc_x_b, const float *res_b,
                         const float *post_b, const float *comb_b, const float *attn_b)
{
    pf_hcpost_t x = { m, hc_x_b, res_b, post_b, comb_b, attn_b, m->config.glm5_hc, m->config.n_embd };
    tensor_parallel_for_dyn(n_tokens, pf_hcpost_task, &x);
    return 1;
}

typedef struct { const glm5_kda_head_ctx_t *ctxs; int n_tokens; } pf_kda_seq_t;
static void pf_kda_head_seq_task(int h, void *vp) {
    const pf_kda_seq_t *x = (const pf_kda_seq_t *)vp;
    for (int t = 0; t < x->n_tokens; t++) glm5_kda_head_task(h, (void *)&x->ctxs[t]);
}
/* Same effect as calling glm5_kda_tail_par(..., do_out_proj=0) for t = 0..n_tokens-1 in order. */
static int glm5_kda_layer_par(const model_t *m, const glm5_layer_weights_t *gw, run_state_t *s,
                              int layer_idx, int n_tokens,
                              const float *q_b, const float *k_b, const float *v_b,
                              const float *z_b, const float *raw_b, const float *beta_b,
                              float *conv_b, float *g_scr, float *head_b, float *beta_sh)
{
    const model_config_t *c = &m->config;
    const int H = c->glm5_n_kda_heads, KD = c->glm5_kda_dim, DI = H * KD;
    static glm5_kda_head_ctx_t *ctxs; static size_t cap;
    if ((size_t)n_tokens > cap) {
        glm5_kda_head_ctx_t *nb = (glm5_kda_head_ctx_t *)realloc(ctxs, (size_t)n_tokens * sizeof *ctxs);
        if (!nb) return 0;
        ctxs = nb; cap = (size_t)n_tokens;
    }
    for (int t = 0; t < n_tokens; t++) {
        glm5_kda_head_ctx_t kc = {
            c, gw, KD, DI, c->ssm_d_conv, 3 * DI,
            q_b + (size_t)t * DI, k_b + (size_t)t * DI, v_b + (size_t)t * DI,
            z_b + (size_t)t * DI, raw_b + (size_t)t * DI, beta_b + (size_t)t * H,
            s->glm5.kda_conv_state[layer_idx], s->glm5.kda_state[layer_idx],
            conv_b + (size_t)t * 3 * DI, g_scr, head_b + (size_t)t * DI, beta_sh,
            layer_idx
        };
        ctxs[t] = kc;
    }
    pf_kda_seq_t x = { ctxs, n_tokens };
    tensor_parallel_for_dyn(H, pf_kda_head_seq_task, &x);
    {
        const char *e = getenv("PICOLM_DBG_DUMPSTATE");
        if (e && atoi(e) == layer_idx) {
            const char *dd = getenv("PICOLM_DUMP_DIR");
            char fn[512];
            snprintf(fn, sizeof(fn), "%s/kda_state-%d.f32", dd ? dd : "/tmp/picd", layer_idx);
            FILE *fp = fopen(fn, "wb");
            if (fp) { fwrite(s->glm5.kda_state[layer_idx], sizeof(float),
                             (size_t)H*KD*KD, fp); fclose(fp); }
        }
    }
    {
        const char *e = getenv("PICOLM_DBG_CONVXY");
        if (e && atoi(e) == layer_idx) {
            const char *dd = getenv("PICOLM_DUMP_DIR");
            char fn[512];
            snprintf(fn, sizeof(fn), "%s/conv_output_silu-%d.f32", dd ? dd : "/tmp/picd", layer_idx);
            FILE *fp = fopen(fn, "wb");
            if (fp) { fwrite(conv_b, sizeof(float), (size_t)(3 * DI), fp); fclose(fp); }
        }
    }
    return 1;
}

/* ---- KDA tail with the state-independent work split out (prefill) ---------------------------
 *
 * glm5_kda_head_task() does, per head and token, ~770 expf calls and a pile of scalar loops
 * (g, decay, the conv1d + SiLU over 3*KD channels, the q/k L2 norms, the output gate) -- and then
 * the recurrent state update, which is the only part that depends on the previous token. Measured
 * on the Xeon: 33 ms per KDA layer at 193 tokens, ~55 us per head-token, almost all of it that
 * state-independent math running as the sequential part.
 *
 * None of it needs the state: the conv1d reads the last CONV_K-1 INPUTS (not outputs), so token t
 * only needs tokens t-3..t of the batch (or the carried conv state for t < 3). So:
 *   phase P  (parallel over token x head): conv+SiLU, q/k norms, g -> decay, attn score, beta, gate
 *   phase C  (tiny):                       roll the carried conv state forward by n_tokens
 *   phase R  (parallel over heads):        per head, tokens in order: state update, RMSNorm, gate
 * Every value is computed with the same expressions as before, only on different threads, so the
/* PICOLM_GLM5_KDA_VEXP=0: disable vectorized KDA expf.
 * ON by default on AVX-512/AVX2; not bit-identical to scalar path. */
static int glm5_kda_vexp_enabled(void) {
    static int cached = -1;
#if defined(PICOLM_AVX512) || defined(PICOLM_AVX2)
    if (cached < 0) { const char *e = getenv("PICOLM_GLM5_KDA_VEXP"); cached = !(e && atoi(e) == 0); }
#else
    cached = 0;
#endif
    return cached;
}
static int glm5_kda_split_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("PICOLM_GLM5_KDA_SPLIT");
        cached = !(e && atoi(e) == 0) && !getenv("PICOLM_DBG_CONVXY") && !getenv("PICOLM_DBG_DUMPSTATE");
    }
    return cached;
}

typedef struct {
    const model_config_t *c; const glm5_layer_weights_t *gw;
    int KD, H, DI, CK, CD, NT, HB, vexp;          /* vexp: AVX-512 pre-pass (not bit-identical) */
    const float *q_b, *k_b, *v_b, *z_b, *raw_b, *beta_b;
    const float *old_state;                       /* conv state before this chunk (read-only in phase P) */
    float *conv_b, *head_b, *decay_p, *gate_p, *score_p, *beta_p, *kda_state;
} kda_split_t;

#if defined(PICOLM_AVX512)

/* Same arithmetic order as the scalar pre-pass; only expf differs, and the q/k L2 norms and the
 * attention-score dot keep their sequential scalar summation order. */
static void kda_split_pre_head_v(const kda_split_t *x, int t, int h)
{
    const model_config_t *c = x->c;
    const glm5_layer_weights_t *gw = x->gw;
    const int KD = x->KD, DI = x->DI, CK = x->CK, CD = x->CD, H = x->H;
    float *conv_out = x->conv_b + (size_t)t * 3 * DI;
    const float *raw = x->raw_b + (size_t)t * DI, *z = x->z_b + (size_t)t * DI;
    const __m512 one = _mm512_set1_ps(1.0f);

    x->beta_p[(size_t)t * H + h] = 1.0f / (1.0f + expf(-x->beta_b[(size_t)t * H + h]));
    {
        const float *dt_bias = (const float *)gw->kda_dt_bias;
        const __m512 a_val = _mm512_set1_ps(((const float *)gw->kda_a)[h]);
        const __m512 lb = _mm512_set1_ps(c->glm5_kda_lower_bound), cap = _mm512_set1_ps(50.0f);
        const __m512 zero = _mm512_setzero_ps();
        for (int ch = 0; ch < KD; ch += 16) {
            const int idx = h * KD + ch;
            const __m512 e = kda_expf16(_mm512_mul_ps(a_val, _mm512_add_ps(_mm512_loadu_ps(raw + idx), _mm512_loadu_ps(dt_bias + idx))));
            const __m512 g = _mm512_mul_ps(lb, _mm512_div_ps(one, _mm512_add_ps(one, e)));
            _mm512_storeu_ps(x->decay_p + (size_t)t * DI + idx, kda_expf16(_mm512_min_ps(g, cap)));
            _mm512_storeu_ps(x->gate_p + (size_t)t * DI + idx,
                             _mm512_div_ps(one, _mm512_add_ps(one, kda_expf16(_mm512_sub_ps(zero, _mm512_loadu_ps(z + idx))))));
        }
    }
    {
        const float *wq = (const float *)gw->kda_conv_q, *wk = (const float *)gw->kda_conv_k, *wv = (const float *)gw->kda_conv_v;
        const __m512i lane = _mm512_mullo_epi32(_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0), _mm512_set1_epi32(CK));
        const __m512 zero = _mm512_setzero_ps();
        for (int part = 0; part < 3; part++) {
            const float *src = (part == 0) ? x->q_b : (part == 1) ? x->k_b : x->v_b;
            const float *w   = (part == 0) ? wq     : (part == 1) ? wk     : wv;
            for (int i = 0; i < KD; i += 16) {
                const int ch_off = h * KD + i, ch = part * DI + ch_off;
                __m512 acc = zero;
                for (int tap = 0; tap < CK; tap++) {
                    __m512 xs;
                    if (tap == CK - 1) xs = _mm512_loadu_ps(src + (size_t)t * DI + ch_off);
                    else {
                        const int tt = t - (CK - 1) + tap;
                        xs = tt >= 0 ? _mm512_loadu_ps(src + (size_t)tt * DI + ch_off)
                                     : _mm512_loadu_ps(x->old_state + (size_t)(tt + CK - 1) * CD + ch);
                    }
                    const __m512 wv16 = _mm512_i32gather_ps(_mm512_add_epi32(lane, _mm512_set1_epi32(tap)),
                                                            w + (size_t)ch_off * CK, 4);
                    acc = _mm512_add_ps(acc, _mm512_mul_ps(xs, wv16));
                }
                const __m512 den = _mm512_add_ps(one, kda_expf16(_mm512_sub_ps(zero, acc)));
                _mm512_storeu_ps(conv_out + ch, _mm512_div_ps(acc, den));
            }
        }
    }
    {
        float *qh = conv_out + h * KD, *kh = conv_out + DI + h * KD;
        float sq = 0.f, sk = 0.f;
        for (int ch = 0; ch < KD; ch++) { sq += qh[ch] * qh[ch]; sk += kh[ch] * kh[ch]; }
        const float inv_q = 1.0f / fmaxf(sqrtf(sq), 1e-5f), inv_k = 1.0f / fmaxf(sqrtf(sk), 1e-5f);
        for (int ch = 0; ch < KD; ch++) { qh[ch] *= inv_q; kh[ch] *= inv_k; }
        const float scale = 1.0f / sqrtf((float)KD);
        float attn_score = 0.f;
        for (int i = 0; i < KD; i++) attn_score += kh[i] * qh[i];
        x->score_p[(size_t)t * H + h] = attn_score * scale;
    }
}
#elif defined(PICOLM_AVX2)
/* 8-lane AVX2 variant of the vector pre-pass.
 * expf: same Cephes polynomial, 8-wide. scalef implemented manually (cvttps_epi32 + exponent bit
 * construction) since AVX2 lacks _mm256_scalef_ps. Gather loads are scalar (AVX2 has no gather)
 * but the expf vectorization still dominates the savings. */
static inline __m256 kda_expf8(__m256 x)
{
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-104.0f)), _mm256_set1_ps(89.0f));
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 c1 = _mm256_set1_ps(0.693359375f), c2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fx, c1, x);
    r = _mm256_fnmadd_ps(fx, c2, r);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.0000001201E-1f));
    const __m256 r2 = _mm256_mul_ps(r, r);
    y = _mm256_fmadd_ps(y, r2, _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    /* Manual scalef: y * 2^(int) via exponent bit construction.
     * Float exponent field = fx + 127 (bias). Clamp to [-149, 127] first.
     * Below -126 the result is subnormal; set to 0 to match libm expf underflow.
     * Above 127 would overflow; set to +inf to match libm expf overflow.
     * fx == 128 is valid for expf(88-89): double y first, then scale by 2^127. */
    __m256i vfx_raw = _mm256_cvttps_epi32(fx);
    /* Handle fx=128: double y and use exponent 127 instead */
    __m256i mask128 = _mm256_cmpeq_epi32(vfx_raw, _mm256_set1_epi32(128));
    y = _mm256_blendv_ps(y, _mm256_add_ps(y, y), _mm256_castsi256_ps(mask128));
    __m256i vfx = _mm256_max_epi32(_mm256_min_epi32(vfx_raw, _mm256_set1_epi32(127)), _mm256_set1_epi32(-149));
    __m256i vexp = _mm256_slli_epi32(_mm256_add_epi32(vfx, _mm256_set1_epi32(127)), 23);
    __m256 vtwo = _mm256_castsi256_ps(vexp);
    __m256 res = _mm256_mul_ps(y, vtwo);
    /* Select: +inf where fx > 128, 0 where fx < -126, computed result otherwise */
    __m256 vzero = _mm256_setzero_ps();
    __m256 vinf  = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));
    __m256i mask_under = _mm256_cmpgt_epi32(_mm256_set1_epi32(-126), vfx_raw); /* fx <= -126 */
    __m256i mask_over  = _mm256_cmpgt_epi32(vfx_raw, _mm256_set1_epi32(128));  /* fx > 128  */
    res = _mm256_blendv_ps(res, vzero, _mm256_castsi256_ps(mask_under));
    res = _mm256_blendv_ps(res, vinf,  _mm256_castsi256_ps(mask_over));
    return res;
}

static void kda_split_pre_head_v2(const kda_split_t *x, int t, int h)
{
    const model_config_t *c = x->c;
    const glm5_layer_weights_t *gw = x->gw;
    const int KD = x->KD, DI = x->DI, CK = x->CK, CD = x->CD, H = x->H;
    float *conv_out = x->conv_b + (size_t)t * 3 * DI;
    const float *raw = x->raw_b + (size_t)t * DI, *z = x->z_b + (size_t)t * DI;
    const __m256 one = _mm256_set1_ps(1.0f);

    x->beta_p[(size_t)t * H + h] = 1.0f / (1.0f + expf(-x->beta_b[(size_t)t * H + h]));
    {
        const float *dt_bias = (const float *)gw->kda_dt_bias;
        const __m256 a_val = _mm256_set1_ps(((const float *)gw->kda_a)[h]);
        const __m256 lb = _mm256_set1_ps(c->glm5_kda_lower_bound), cap = _mm256_set1_ps(50.0f);
        const __m256 zero = _mm256_setzero_ps();
        for (int ch = 0; ch < KD; ch += 8) {
            const int idx = h * KD + ch;
            const __m256 e = kda_expf8(_mm256_mul_ps(a_val, _mm256_add_ps(_mm256_loadu_ps(raw + idx), _mm256_loadu_ps(dt_bias + idx))));
            const __m256 g = _mm256_mul_ps(lb, _mm256_div_ps(one, _mm256_add_ps(one, e)));
            _mm256_storeu_ps(x->decay_p + (size_t)t * DI + idx, kda_expf8(_mm256_min_ps(g, cap)));
            _mm256_storeu_ps(x->gate_p + (size_t)t * DI + idx,
                             _mm256_div_ps(one, _mm256_add_ps(one, kda_expf8(_mm256_sub_ps(zero, _mm256_loadu_ps(z + idx))))));
        }
    }
    {
        const float *wq = (const float *)gw->kda_conv_q, *wk = (const float *)gw->kda_conv_k, *wv = (const float *)gw->kda_conv_v;
        const __m256 zero = _mm256_setzero_ps();
        for (int part = 0; part < 3; part++) {
            const float *src = (part == 0) ? x->q_b : (part == 1) ? x->k_b : x->v_b;
            const float *w   = (part == 0) ? wq     : (part == 1) ? wk     : wv;
            for (int i = 0; i < KD; i += 8) {
                const int ch_off = h * KD + i, ch = part * DI + ch_off;
                __m256 acc = zero;
                for (int tap = 0; tap < CK; tap++) {
                    __m256 xs;
                    if (tap == CK - 1) xs = _mm256_loadu_ps(src + (size_t)t * DI + ch_off);
                    else {
                        const int tt = t - (CK - 1) + tap;
                        xs = tt >= 0 ? _mm256_loadu_ps(src + (size_t)tt * DI + ch_off)
                                     : _mm256_loadu_ps(x->old_state + (size_t)(tt + CK - 1) * CD + ch);
                    }
                    /* Scalar gather for 8 strided weights (AVX2 has no gather) */
                    const __m256 wv8 = _mm256_set_ps(
                        w[(ch_off+7) * CK + tap], w[(ch_off+6) * CK + tap],
                        w[(ch_off+5) * CK + tap], w[(ch_off+4) * CK + tap],
                        w[(ch_off+3) * CK + tap], w[(ch_off+2) * CK + tap],
                        w[(ch_off+1) * CK + tap], w[ch_off * CK + tap]);
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(xs, wv8));
                }
                const __m256 den = _mm256_add_ps(one, kda_expf8(_mm256_sub_ps(zero, acc)));
                _mm256_storeu_ps(conv_out + ch, _mm256_div_ps(acc, den));
            }
        }
    }
    {
        float *qh = conv_out + h * KD, *kh = conv_out + DI + h * KD;
        float sq = 0.f, sk = 0.f;
        for (int ch = 0; ch < KD; ch++) { sq += qh[ch] * qh[ch]; sk += kh[ch] * kh[ch]; }
        const float inv_q = 1.0f / fmaxf(sqrtf(sq), 1e-5f), inv_k = 1.0f / fmaxf(sqrtf(sk), 1e-5f);
        for (int ch = 0; ch < KD; ch++) { qh[ch] *= inv_q; kh[ch] *= inv_k; }
        const float scale = 1.0f / sqrtf((float)KD);
        float attn_score = 0.f;
        for (int i = 0; i < KD; i++) attn_score += kh[i] * qh[i];
        x->score_p[(size_t)t * H + h] = attn_score * scale;
    }
}
#endif

static void kda_split_pre_head(const kda_split_t *x, int t, int h)
{
#if defined(PICOLM_AVX512)
    if (x->vexp) { kda_split_pre_head_v(x, t, h); return; }
#elif defined(PICOLM_AVX2)
    if (x->vexp) { kda_split_pre_head_v2(x, t, h); return; }
#endif
    const model_config_t *c = x->c;
    const glm5_layer_weights_t *gw = x->gw;
    const int KD = x->KD, DI = x->DI, CK = x->CK, CD = x->CD, H = x->H;
    float *conv_out = x->conv_b + (size_t)t * 3 * DI;
    const float *raw = x->raw_b + (size_t)t * DI, *z = x->z_b + (size_t)t * DI;

    x->beta_p[(size_t)t * H + h] = 1.0f / (1.0f + expf(-x->beta_b[(size_t)t * H + h]));

    {   /* g, then decay = exp(min(g, 50)) */
        const float *dt_bias = (const float *)gw->kda_dt_bias;
        const float a_val = ((const float *)gw->kda_a)[h];
        for (int ch = 0; ch < KD; ch++) {
            const int idx = h * KD + ch;
            const float gv = c->glm5_kda_lower_bound *
                             (1.0f / (1.0f + expf(a_val * (raw[idx] + dt_bias[idx]))));
            x->decay_p[(size_t)t * DI + idx] = expf(fminf(gv, 50.0f));
        }
    }
    for (int ch = 0; ch < KD; ch++) {            /* output gate sigmoid(z) */
        const int idx = h * KD + ch;
        x->gate_p[(size_t)t * DI + idx] = 1.0f / (1.0f + expf(-z[idx]));
    }
    {   /* causal depthwise conv1d + SiLU; history = the previous inputs of this batch, or the carried state */
        const float *wq = (const float *)gw->kda_conv_q, *wk = (const float *)gw->kda_conv_k, *wv = (const float *)gw->kda_conv_v;
        for (int part = 0; part < 3; part++) {
            const float *src = (part == 0) ? x->q_b : (part == 1) ? x->k_b : x->v_b;
            const float *w   = (part == 0) ? wq     : (part == 1) ? wk     : wv;
            for (int i = 0; i < KD; i++) {
                const int ch_off = h * KD + i, ch = part * DI + ch_off;
                float xs[GLM5_KD_MAX];
                for (int tap = 0; tap < CK - 1; tap++) {
                    const int tt = t - (CK - 1) + tap;
                    xs[tap] = tt >= 0 ? src[(size_t)tt * DI + ch_off]
                                      : x->old_state[(size_t)(tt + CK - 1) * CD + ch];
                }
                xs[CK - 1] = src[(size_t)t * DI + ch_off];
                float acc = 0.f;
                for (int tap = 0; tap < CK; tap++) acc += xs[tap] * w[tap + ch_off * CK];
                conv_out[ch] = acc / (1.0f + expf(-acc));
            }
        }
    }
    {   /* per-head L2 norm of q and k */
        float *qh = conv_out + h * KD, *kh = conv_out + DI + h * KD;
        float sq = 0.f, sk = 0.f;
        for (int ch = 0; ch < KD; ch++) { sq += qh[ch] * qh[ch]; sk += kh[ch] * kh[ch]; }
        const float inv_q = 1.0f / fmaxf(sqrtf(sq), 1e-5f), inv_k = 1.0f / fmaxf(sqrtf(sk), 1e-5f);
        for (int ch = 0; ch < KD; ch++) { qh[ch] *= inv_q; kh[ch] *= inv_k; }
        const float scale = 1.0f / sqrtf((float)KD);
        float attn_score = 0.f;
        for (int i = 0; i < KD; i++) attn_score += kh[i] * qh[i];
        x->score_p[(size_t)t * H + h] = attn_score * scale;
    }
}
static void kda_split_pre_task(int idx, void *vp) {
    const kda_split_t *x = (const kda_split_t *)vp;
    const int per = x->H / x->HB;
    const int t = idx / per, hb = idx % per;
    for (int k = 0; k < x->HB; k++) kda_split_pre_head(x, t, hb * x->HB + k);
}

/* The sequential part: recurrent state update (verbatim from glm5_kda_head_task), then the per-head
 * RMSNorm and the precomputed gate. */
static void kda_rec_head(const float *qh, const float *kh, const float *vh, const float *decay,
                         float attn_score, float beta_h, float scale, float *Sh, float *head_out_h,
                         const float *gate, const float *norm_w, float eps, int KD)
{
    float v_new[GLM5_KD_MAX];
        {
            float *vp = v_new;
            float *ov = head_out_h;
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
                float vn = beta_h * (vh[row] - vp[row]);
                v_new[row] = vn;
                ov[row] = ov[row] * scale + vn * attn_score;
            }
        }
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
    {
        float *oh = head_out_h;
        float ss = 0.f;
        for (int ch = 0; ch < KD; ch++) ss += oh[ch] * oh[ch];
        const float inv = 1.0f / sqrtf(ss / KD + eps);
        for (int ch = 0; ch < KD; ch++) oh[ch] *= inv * norm_w[ch];
        for (int ch = 0; ch < KD; ch++) oh[ch] *= gate[ch];
    }
}
static void kda_split_rec_task(int h, void *vp) {
    const kda_split_t *x = (const kda_split_t *)vp;
    const int KD = x->KD, DI = x->DI, H = x->H;
    const float scale = 1.0f / sqrtf((float)KD);
    const float *norm_w = (const float *)x->gw->kda_norm;
    float *Sh = x->kda_state + (size_t)h * KD * KD;
    for (int t = 0; t < x->NT; t++) {
        const float *co = x->conv_b + (size_t)t * 3 * DI;
        kda_rec_head(co + h * KD, co + DI + h * KD, co + 2 * DI + h * KD,
                     x->decay_p + (size_t)t * DI + h * KD,
                     x->score_p[(size_t)t * H + h], x->beta_p[(size_t)t * H + h], scale, Sh,
                     x->head_b + (size_t)t * DI + h * KD, x->gate_p + (size_t)t * DI + h * KD,
                     norm_w, x->c->rms_norm_eps, KD);
    }
}

/* Same contract and bit-identical results as glm5_kda_layer_par(). */
static int glm5_kda_layer_split_ex(const model_t *m, const glm5_layer_weights_t *gw, run_state_t *s,
                                   int layer_idx, int n_tokens,
                                   const float *q_b, const float *k_b, const float *v_b,
                                   const float *z_b, const float *raw_b, const float *beta_b,
                                   float *conv_b, float *g_scr, float *head_b, float *beta_sh, int vexp_req)
{
    const model_config_t *c = &m->config;
    const int H = c->glm5_n_kda_heads, KD = c->glm5_kda_dim, DI = H * KD, CK = c->ssm_d_conv;
    static float *buf; static size_t cap;                      /* cap in bytes (ep_grow) */
    const size_t need = ((size_t)n_tokens * DI * 2 + (size_t)n_tokens * H * 2);
    if (!glm5_kda_split_enabled() || KD > GLM5_KD_MAX || CK < 2 || n_tokens < 1 || H < 1 ||
        !ep_grow((void **)&buf, &cap, need * sizeof(float)))
        return glm5_kda_layer_par(m, gw, s, layer_idx, n_tokens, q_b, k_b, v_b, z_b, raw_b, beta_b,
                                  conv_b, g_scr, head_b, beta_sh);
    float *conv_state = s->glm5.kda_conv_state[layer_idx];
    kda_split_t x;
    x.c = c; x.gw = gw; x.KD = KD; x.H = H; x.DI = DI; x.CK = CK; x.CD = 3 * DI; x.NT = n_tokens;
    x.HB = (H % 4 == 0) ? 4 : 1;
#if defined(PICOLM_AVX512)
    x.vexp = vexp_req && (KD % 16 == 0);
#elif defined(PICOLM_AVX2)
    x.vexp = vexp_req && (KD % 8 == 0);
#else
    (void)vexp_req;
    x.vexp = 0;
#endif
    x.q_b = q_b; x.k_b = k_b; x.v_b = v_b; x.z_b = z_b; x.raw_b = raw_b; x.beta_b = beta_b;
    x.old_state = conv_state; x.conv_b = conv_b; x.head_b = head_b; x.kda_state = s->glm5.kda_state[layer_idx];
    x.decay_p = buf; x.gate_p = buf + (size_t)n_tokens * DI;
    x.score_p = buf + (size_t)n_tokens * DI * 2; x.beta_p = x.score_p + (size_t)n_tokens * H;

    tensor_parallel_for_dyn(n_tokens * (H / x.HB), kda_split_pre_task, &x);       /* phase P */

    {   /* phase C: carried conv state := the last CK-1 inputs of the stream (in place, ascending) */
        const int CD = x.CD, NT = n_tokens;
        for (int ch = 0; ch < CD; ch++) {
            const int part = ch / DI, ch_off = ch % DI;
            const float *src = (part == 0) ? q_b : (part == 1) ? k_b : v_b;
            for (int tap = 0; tap < CK - 1; tap++) {
                const int tt = NT - (CK - 1) + tap;
                conv_state[(size_t)tap * CD + ch] = tt >= 0 ? src[(size_t)tt * DI + ch_off]
                                                            : conv_state[(size_t)(tap + NT) * CD + ch];
            }
        }
    }
    tensor_parallel_for_dyn(H, kda_split_rec_task, &x);                           /* phase R */
    return 1;
}

static int glm5_kda_layer_split(const model_t *m, const glm5_layer_weights_t *gw, run_state_t *s,
                                int layer_idx, int n_tokens,
                                const float *q_b, const float *k_b, const float *v_b,
                                const float *z_b, const float *raw_b, const float *beta_b,
                                float *conv_b, float *g_scr, float *head_b, float *beta_sh)
{
    return glm5_kda_layer_split_ex(m, gw, s, layer_idx, n_tokens, q_b, k_b, v_b, z_b, raw_b, beta_b,
                                   conv_b, g_scr, head_b, beta_sh, glm5_kda_vexp_enabled());
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
    /* PICOLM_POOL_GAPS also covers prefill: the counters are reset here so the
     * report describes this prefill call only (the decode path resets its own).
     * Same accounting as decode: serial gap (workers idle, main busy) vs main's
     * own slot vs straggler wait. */
    const int pg_on = tensor_pool_gaps_enabled();
    if (pg_on) tensor_pool_gap_reset();
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
    /* n_tokens == 1: the batched layout buys nothing and the per-token path is
     * both faster (no S-stream arena traffic, no batched KDA scaffolding) and
     * the one the batched path is validated against. model_forward_prefill()
     * already redirects the n_tokens==1 case here; do it again so direct
     * callers of this function (and a 1-token chunk remainder left by the
     * GLM5_PREFILL_MAX_TOKENS split above) take the same route. */
    if (no_batch || n_tokens == 1 || !pf || n_tokens > s->glm5.prefill_n_tokens) {
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
            PFP_BEGIN(tLayer);
            const int pfk = mla_ord >= 0 ? 1 : 0;
            /* ============ Attention block ============ */

            /* Phase A: hc_pre(attn) + RMSNorm, per token. Saves hc_res,
             * post, comb per token for the hc_post in phase D. */
            PFP_BEGIN(tA);
            /* Name the layer for the hc_pre diagnostics BEFORE the parallel
             * fast path is considered: pf_run_hcpre() bails to the serial loop
             * whenever g_hc_tag is set, and the serial loop is what emits the
             * per-token stats. Setting it only inside the serial loop was the
             * chicken-and-egg that made prefill stats invisible. */
            if (glm5_stats_wanted() || getenv("PICOLM_DBG_DUMP_LAYER") || getenv("PICOLM_DBG_HCVEC"))
                snprintf(g_hc_tag, sizeof(g_hc_tag), "%d", il);
            else
                g_hc_tag[0] = '\0';   /* decode leaves its tag set: do not let it force the serial path */
            if (!(glm5_pf_par() && pf_run_hcpre(m, gw, 1, n_tokens, hc_x_b, cur_b, hc_res_b, post_b, comb_b, s->attn_norm_w[il])))
            for (int t = 0; t < n_tokens; t++) {
                float *hc_x_t = hc_x_b + (size_t)t*S*D;
                float *cur_t  = cur_b + (size_t)t*D;
                float *res_t  = hc_res_b + (size_t)t*S*D;
                float *post_t = post_b + (size_t)t*S;
                float *comb_t = comb_b + (size_t)t*S*S;
                memcpy(res_t, hc_x_t, (size_t)S*D*sizeof(float));
                if (t == 0 && glm5_stats_wanted()) {
                    char t2[96];
                    snprintf(t2, sizeof(t2), "l%02d.x_in", il);
                    glm5_stats(t2, hc_x_t, S*D);
                }
                /* PICOLM_DBG_DUMP_TOK selects which token of the prefill batch is dumped (default 0).
                 * The ik oracle's l_out-<N> holds the LAST token of its batch, so diffing a prefill
                 * against QORACLE_DUMP needs this to point at the same token. */
                if (t == glm5_dump_tok()) {
                    static int dl = -2;
                    if (dl == -2) { const char *e = getenv("PICOLM_DBG_DUMP_LAYER"); dl = e ? atoi(e) : -1; }
                    if (dl == il) {
                        const char *dd = getenv("PICOLM_DUMP_DIR");
                        char fn[512];
                        snprintf(fn, sizeof(fn), "%s/pic_l_in-%d.f32", dd ? dd : "/tmp/picd", il);
                        FILE *fp = fopen(fn, "wb");
                        if (fp) { fwrite(hc_x_t, sizeof(float), (size_t)S*D, fp); fclose(fp); }
                    }
                }
                glm5_hc_pre(m, gw, 1, hc_x_t, cur_t, post_t, comb_t, s->glm5.hc_scratch);
                if (t == glm5_dump_tok()) {
                    static int wl = -2;
                    if (wl == -2) { const char *e = getenv("PICOLM_DBG_HCVEC"); wl = e ? atoi(e) : -1; }
                    if (wl == il) {
                        const char *dd = getenv("PICOLM_DUMP_DIR");
                        char fn[512];
                        snprintf(fn, sizeof(fn), "%s/attn_out-%d.f32", dd ? dd : "/tmp/picd", il);
                        FILE *fp = fopen(fn, "wb");
                        if (fp) { fwrite(hc_x_t, sizeof(float), (size_t)S*D, fp); fclose(fp); }
                        snprintf(fn, sizeof(fn), "%s/attn_comb-%d.f32", dd ? dd : "/tmp/picd", il);
                        fp = fopen(fn, "wb");
                        if (fp) { fwrite(comb_t, sizeof(float), (size_t)S*S, fp); fclose(fp); }
                        snprintf(fn, sizeof(fn), "%s/attn_post-%d.f32", dd ? dd : "/tmp/picd", il);
                        fp = fopen(fn, "wb");
                        if (fp) { fwrite(post_t, sizeof(float), (size_t)S, fp); fclose(fp); }
                    }
                    if (wl == il)
                        fprintf(stderr, "[hcvec L%d attn] pre=[%.6f %.6f %.6f %.6f] post=[%.6f %.6f %.6f %.6f] comb0=[%.6f %.6f %.6f %.6f]\n",
                                il, hc_pre_dbg_pre[0], hc_pre_dbg_pre[1], hc_pre_dbg_pre[2], hc_pre_dbg_pre[3],
                                post_t[0], post_t[1], post_t[2], post_t[3],
                                comb_t[0], comb_t[1], comb_t[2], comb_t[3]);
                }
                rmsnorm(cur_t, cur_t, s->attn_norm_w[il], D, c->rms_norm_eps);
                if (t == 0) {
                    char t2[96];
                    snprintf(t2, sizeof(t2), "l%02d.hc_attn_pre", il);
                    glm5_stats(t2, cur_t, D);
                    snprintf(t2, sizeof(t2), "l%02d.cur_normed", il);
                    glm5_stats(t2, cur_t, D);
                }
            }
            PFP_END(tA, pfk, PF_HC_ATTN_PRE);
            PFP_BEGIN(tAt);

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
                    PFP_BEGIN(tMP);
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
                        glm5_mla_shadow_sync(s, c->max_seq_len, KVL, mla_ord,
                                             start_pos, n_tokens);
                    }

                    PFP_END(tMP, pfk, PF_MLA_PROJ);
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
                            PFP_BEGIN(tMK);
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

                            PFP_END(tMK, pfk, PF_MLA_KB);

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
                                if (ctx_b) {
                                    PFP_BEGIN(tMC);
                                    glm5_mla_attn_batch(c, s, mla_ord, qabs_b, ctx_b,
                                                        n_tokens, start_pos);
                                    PFP_END(tMC, pfk, PF_MLA_CORE);
                                }
                            }
                            if (dense && ctx_b) {
                                /* o[h] = Wv_b[h] @ ctx[h] per token */
                                /* Same shape as qabs, same reasoning: per
                                 * head over all tokens, so Wv_b's 104 KB per
                                 * head is read once per layer not once per
                                 * token. Gather ctx rows for the head, GEMM,
                                 * scatter back to [t][h]. */
                                PFP_BEGIN(tMV);
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
                                PFP_END(tMV, pfk, PF_MLA_VB);
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
                    PFP_BEGIN(tMO);
                    matmul_batch(attn_b, o_plane, n_tokens, lw->attn_output,
                                 QD, D, lw->type_attn_output);
                    PFP_END(tMO, pfk, PF_MLA_OUT);
                }
            } else {
                /* KDA: batch the weight-bound projections, then run the
                 * sequential tail per token. The projections read cur_b and
                 * write into per-token planes; the tail reads those planes
                 * and updates the recurrent state in token order. */
                PFP_BEGIN(tKP);
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
                PFP_END(tKP, pfk, PF_KDA_PROJ);

                /* Sequential tail: conv -> recurrence -> norm -> gate. The
                 * tail's output projection is deferred so attn_output can be
                 * GEMM'd for all tokens at once below. */
                PFP_BEGIN(tKT);
                if (!(glm5_pf_par() && glm5_kda_layer_split(m, gw, s, il, n_tokens, q_b, k_b, v_b, gb_b, fb_b,
                                                          beta_b, conv_b, g_scr, head_b, beta_sh)))
                for (int t = 0; t < n_tokens; t++) {
                    glm5_kda_tail_par(m, gw, lw, s, il, start_pos + t,
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
                PFP_END(tKT, pfk, PF_KDA_TAIL);
                /* Batched attention output projection: [N][DI] -> [N][D]. */
                PFP_BEGIN(tKO);
                {
                    const char *e = getenv("PICOLM_DBG_DUMPATTN");
                    if (e && atoi(e) == il) {
                        const char *dd = getenv("PICOLM_DUMP_DIR");
                        char fn[512];
                        snprintf(fn, sizeof(fn), "%s/attn_gated-%d.f32", dd ? dd : "/tmp/picd", il);
                        FILE *fp = fopen(fn, "wb");
                        if (fp) { fwrite(head_b, sizeof(float), (size_t)DI, fp); fclose(fp); }
                    }
                }
                matmul_batch(attn_b, head_b, n_tokens, lw->attn_output, DI, D,
                             lw->type_attn_output);
                {
                    const char *e = getenv("PICOLM_DBG_DUMPATTN");
                    if (e && atoi(e) == il) {
                        const char *dd = getenv("PICOLM_DUMP_DIR");
                        char fn[512];
                        snprintf(fn, sizeof(fn), "%s/attn_out-%d.f32", dd ? dd : "/tmp/picd", il);
                        FILE *fp = fopen(fn, "wb");
                        if (fp) { fwrite(attn_b, sizeof(float), (size_t)D, fp); fclose(fp); }
                    }
                }
                PFP_END(tKO, pfk, PF_KDA_OUT);
            }
            PFP_END(tAt, pfk, PF_ATTN);

            /* Phase D: hc_post(attn) per token */
            PFP_BEGIN(tD);
            if (!(glm5_pf_par() && !getenv("PICOLM_DBG_HCVEC") && pf_run_hcpost(m, n_tokens, hc_x_b, hc_res_b, post_b, comb_b, attn_b)))
            for (int t = 0; t < n_tokens; t++) {
                float *hc_x_t = hc_x_b + (size_t)t*S*D;
                float *res_t  = hc_res_b + (size_t)t*S*D;
                float *post_t = post_b + (size_t)t*S;
                float *comb_t = comb_b + (size_t)t*S*S;
                float *attn_t = attn_b + (size_t)t*D;
                glm5_hc_post(m, attn_t, res_t, post_t, comb_t, hc_x_t);
                if (t == glm5_dump_tok()) {
                    static int wl = -2;
                    if (wl == -2) { const char *e = getenv("PICOLM_DBG_HCVEC"); wl = e ? atoi(e) : -1; }
                    if (wl == il) {
                        const char *dd = getenv("PICOLM_DUMP_DIR");
                        char fn[512];
                        snprintf(fn, sizeof(fn), "%s/xmid-%d.f32", dd ? dd : "/tmp/picd", il);
                        FILE *fp = fopen(fn, "wb");
                        if (fp) { fwrite(hc_x_t, sizeof(float), (size_t)S*D, fp); fclose(fp); }
                    }
                }
            }
            PFP_END(tD, pfk, PF_HC_ATTN_POST);

            /* ============ FFN block ============ */

            /* Phase A': hc_pre(ffn) + RMSNorm per token (reuses post/comb planes) */
            PFP_BEGIN(tA2);
            if (!(glm5_pf_par() && pf_run_hcpre(m, gw, 0, n_tokens, hc_x_b, cur_b, hc_res_b, post_b, comb_b, s->post_attn_norm_w[il])))
            for (int t = 0; t < n_tokens; t++) {
                float *hc_x_t = hc_x_b + (size_t)t*S*D;
                float *cur_t  = cur_b + (size_t)t*D;
                float *res_t  = hc_res_b + (size_t)t*S*D;
                float *post_t = post_b + (size_t)t*S;
                float *comb_t = comb_b + (size_t)t*S*S;
                memcpy(res_t, hc_x_t, (size_t)S*D*sizeof(float));
                if (t == 0 && getenv("PICOLM_DBG_HCVEC") && atoi(getenv("PICOLM_DBG_HCVEC")) == il) {
                    const char *dd = getenv("PICOLM_DUMP_DIR");
                    char fn[512];
                    snprintf(fn, sizeof(fn), "%s/ffnpre_in-%d.f32", dd ? dd : "/tmp/picd", il);
                    FILE *fp = fopen(fn, "wb");
                    if (fp) { fwrite(hc_x_t, sizeof(float), (size_t)S*D, fp); fclose(fp); }
                }
                glm5_hc_pre(m, gw, 0, hc_x_t, cur_t, post_t, comb_t, s->glm5.hc_scratch);
                if (t == 0 && getenv("PICOLM_DBG_HCVEC") && atoi(getenv("PICOLM_DBG_HCVEC")) == il) {
                    const char *dd = getenv("PICOLM_DUMP_DIR");
                    char fn[512];
                    snprintf(fn, sizeof(fn), "%s/ffnpre_cur-%d.f32", dd ? dd : "/tmp/picd", il);
                    FILE *fp = fopen(fn, "wb");
                    if (fp) { fwrite(cur_t, sizeof(float), (size_t)D, fp); fclose(fp); }
                }
                rmsnorm(cur_t, cur_t, s->post_attn_norm_w[il], D, c->rms_norm_eps);
                if (t == 0 && getenv("PICOLM_DBG_HCVEC") && atoi(getenv("PICOLM_DBG_HCVEC")) == il) {
                    const char *dd = getenv("PICOLM_DUMP_DIR");
                    char fn[512];
                    double ss=0; for(int q=0;q<D;q++) ss+=(double)cur_t[q]*cur_t[q];
                    fprintf(stderr, "[ffnpre L%d] normed rms=%.6f\n", il, (float)sqrt(ss/D));
                    snprintf(fn, sizeof(fn), "%s/ffn_norm-%d.f32", dd ? dd : "/tmp/picd", il);
                    FILE *fp = fopen(fn, "wb");
                    if (fp) { fwrite(cur_t, sizeof(float), (size_t)D, fp); fclose(fp); }
                }
            }
            PFP_END(tA2, pfk, PF_HC_FFN_PRE);

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
            PFP_BEGIN(tF);
            if (il < c->n_layer_dense_lead) {
                if (!(glm5_dense_batch_enabled() && glm5_dense_ffn_batch(m, lw, cur_b, attn_b, n_tokens)))
                for (int t = 0; t < n_tokens; t++)
                    glm5_dense_ffn(m, lw, s, cur_b + (size_t)t*D, attn_b + (size_t)t*D);
            } else if (n_tokens > 1 && !getenv("PICOLM_GLM5_MOE_PERTOKEN")) {
                glm5_moe_ffn_batch(m, gw, lw, s, cur_b, attn_b, n_tokens, il);
            } else {
                for (int t = 0; t < n_tokens; t++)
                    glm5_moe_ffn(m, gw, lw, s, cur_b + (size_t)t*D, attn_b + (size_t)t*D, il);
            }
            if (getenv("PICOLM_DBG_HCVEC") && atoi(getenv("PICOLM_DBG_HCVEC")) == il) {
                const char *dd = getenv("PICOLM_DUMP_DIR");
                char fn[512];
                double ss=0; for (int q=0;q<D;q++) ss+=(double)attn_b[q]*attn_b[q];
                fprintf(stderr, "[ffnchk L%d] ffn_out rms=%.6f\n", il, (float)sqrt(ss/D));
                snprintf(fn, sizeof(fn), "%s/ffn_out-%d.f32", dd ? dd : "/tmp/picd", il);
                FILE *fp = fopen(fn, "wb");
                if (fp) { fwrite(attn_b, sizeof(float), (size_t)D, fp); fclose(fp); }
            }
            PFP_END(tF, pfk, PF_FFN);

            /* Phase D': hc_post(ffn) per token */
            PFP_BEGIN(tD2);
            if (!(glm5_pf_par() && !getenv("PICOLM_DBG_HCVEC") && pf_run_hcpost(m, n_tokens, hc_x_b, hc_res_b, post_b, comb_b, attn_b)))
            for (int t = 0; t < n_tokens; t++) {
                float *hc_x_t = hc_x_b + (size_t)t*S*D;
                float *res_t  = hc_res_b + (size_t)t*S*D;
                float *post_t = post_b + (size_t)t*S;
                float *comb_t = comb_b + (size_t)t*S*S;
                float *attn_t = attn_b + (size_t)t*D;
                if (t == glm5_dump_tok()) {
                    static int wl2 = -2;
                    if (wl2 == -2) { const char *e = getenv("PICOLM_DBG_HCVEC"); wl2 = e ? atoi(e) : -1; }
                    if (wl2 == il) {
                        const char *dd = getenv("PICOLM_DUMP_DIR");
                        char fn[512];
                        snprintf(fn, sizeof(fn), "%s/ffn_attn-%d.f32", dd ? dd : "/tmp/picd", il);
                        FILE *fp = fopen(fn, "wb");
                        if (fp) { fwrite(attn_t, sizeof(float), (size_t)D, fp); fclose(fp); }
                        snprintf(fn, sizeof(fn), "%s/ffn_comb-%d.f32", dd ? dd : "/tmp/picd", il);
                        fp = fopen(fn, "wb");
                        if (fp) { fwrite(comb_t, sizeof(float), (size_t)S*S, fp); fclose(fp); }
                        snprintf(fn, sizeof(fn), "%s/ffn_post-%d.f32", dd ? dd : "/tmp/picd", il);
                        fp = fopen(fn, "wb");
                        if (fp) { fwrite(post_t, sizeof(float), (size_t)S, fp); fclose(fp); }
                    }
                }
                glm5_hc_post(m, attn_t, res_t, post_t, comb_t, hc_x_t);
            }
            PFP_END(tD2, pfk, PF_HC_FFN_POST);
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
            if (tLayer) { const double dl = glm5_prof_now() - tLayer; g_pf_ms[pfk][PF_LAYER] += dl; g_pf_cur[PF_LAYER] = dl; g_pf_layers[pfk]++; glm5_pfprof_layer_line(il, pfk); }
            BENCH_LAYER_END(il, 1);
        }

        if (glm5_pfprof_on()) glm5_pfprof_print();

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
        if (pg_on) {
            double gap, mainm, wt, wk; long n;
            if (tensor_pool_gap_take(&gap, &mainm, &wt, &wk, &n)) {
                double tot = gap + mainm + wt;
                fprintf(stderr,
                        "\n--- pool timing (PICOLM_POOL_GAPS), PREFILL of %d tokens ---\n"
                        "  dispatches                               %6ld\n"
                        "  serial gap (workers idle, main busy)     %6.1f ms  %5.1f%%\n"
                        "  main's own share of dispatches           %6.1f ms  %5.1f%%\n"
                        "  waiting for stragglers (barrier skew)    %6.1f ms  %5.1f%%\n"
                        "  mean per dispatch: gap %.1f us, skew %.1f us, wake %.2f us\n",
                        n_tokens, n, gap, tot > 0 ? 100*gap/tot : 0,
                        mainm, tot > 0 ? 100*mainm/tot : 0,
                        wt, tot > 0 ? 100*wt/tot : 0,
                        n ? gap*1e3/n : 0, n ? wt*1e3/n : 0, wk);
            }
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
        glm5_mla_shadow_sync(s, c->max_seq_len, c->glm5_kv_lora, ord, 0, n_pos);
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
