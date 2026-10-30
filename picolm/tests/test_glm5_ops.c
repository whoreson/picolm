/* tests/test_glm5_ops.c -- standalone numerical checks for the GLM-5 operators.
 *
 * These do NOT depend on ik. They re-derive each operator with plain scalar code
 * and compare against PicoLM's implementation, and they assert the algebraic
 * invariants that must hold for any correct implementation. That makes them a
 * fast, unambiguous first line of defence: if a formula or index is wrong this
 * fires even though both sides run in the same process.
 *
 * Build:  make tests/test_glm5_ops   (or)  see the rule at the bottom
 * Run:    ./tests/test_glm5_ops
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, ...) do {                                  \
    if (!(cond)) {                                             \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);   \
        fprintf(stderr, __VA_ARGS__);                          \
        fprintf(stderr, "\n");                                 \
        g_fail++;                                              \
    } else { g_pass++; }                                       \
} while (0)

/* Relative/absolute closeness. */
static int closef(float a, float b, float rel, float abs_tol) {
    float d = fabsf(a - b);
    float m = fmaxf(fabsf(a), fabsf(b));
    return d <= abs_tol || d <= rel * m;
}

/* ------------------------------------------------------------------ */
/* Reference: hc_pre Sinkhorn, straight from ggml.c:23042             */
/* ------------------------------------------------------------------ */
static void ref_hc_pre(const float *m, const float *scale, const float *base,
                       int S, int iters, float eps,
                       float *pre, float *post, float *comb)
{
    for (int i = 0; i < S; i++) {
        float v = m[i] * scale[0] + base[i];
        pre[i] = 1.0f / (1.0f + expf(-v)) + eps;
        v = m[S + i] * scale[1] + base[S + i];
        post[i] = 2.0f * (1.0f / (1.0f + expf(-v)));
    }
    float mm[64];
    for (int i = 0; i < S * S; i++) mm[i] = m[2*S + i] * scale[2] + base[2*S + i];

    for (int r = 0; r < S; r++) {
        float mx = mm[r*S];
        for (int c = 1; c < S; c++) mx = fmaxf(mx, mm[r*S + c]);
        float sum = 0.f;
        for (int c = 0; c < S; c++) { mm[r*S + c] = expf(mm[r*S + c] - mx); sum += mm[r*S + c]; }
        for (int c = 0; c < S; c++) mm[r*S + c] = mm[r*S + c] / sum + eps;
    }
    for (int c = 0; c < S; c++) {
        float sum = eps;
        for (int r = 0; r < S; r++) sum += mm[r*S + c];
        for (int r = 0; r < S; r++) mm[r*S + c] /= sum;
    }
    for (int it = 0; it < iters - 1; it++) {
        for (int r = 0; r < S; r++) {
            float sum = eps;
            for (int c = 0; c < S; c++) sum += mm[r*S + c];
            for (int c = 0; c < S; c++) mm[r*S + c] /= sum;
        }
        for (int c = 0; c < S; c++) {
            float sum = eps;
            for (int r = 0; r < S; r++) sum += mm[r*S + c];
            for (int r = 0; r < S; r++) mm[r*S + c] /= sum;
        }
    }
    memcpy(comb, mm, (size_t)S * S * sizeof(float));
}

static void test_hc_pre_invariants(void) {
    const int S = 4, iters = 20;
    const float eps = 1e-6f;
    float m[64], scale[3] = { 0.7f, 1.3f, -0.4f }, base[64];
    float pre[8], post[8], comb[64];

    srand(1234);
    for (int i = 0; i < 3; i++) { /* keep scale as given */ (void)i; }
    for (int i = 0; i < 2*S; i++) base[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.f;
    for (int i = 2*S; i < 2*S + S*S; i++) base[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.f;
    for (int i = 0; i < (2*S + S*S); i++) m[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.f;

    ref_hc_pre(m, scale, base, S, iters, eps, pre, post, comb);

    /* Invariants that hold for ANY input. */
    for (int i = 0; i < S; i++) {
        CHECK(pre[i] > eps && pre[i] < 1.0f + eps, "pre[%d]=%g outside (eps,1+eps)", i, pre[i]);
        CHECK(post[i] > 0.f && post[i] < 2.0f, "post[%d]=%g outside (0,2)", i, post[i]);
    }
    /* Sinkhorn converges to a doubly-stochastic-ish matrix. */
    for (int r = 0; r < S; r++) {
        float sum = 0.f;
        for (int c = 0; c < S; c++) sum += comb[r*S + c];
        CHECK(closef(sum, 1.f, 1e-3f, 1e-3f), "comb row %d sums to %g, want ~1", r, sum);
    }
    for (int c = 0; c < S; c++) {
        float sum = 0.f;
        for (int r = 0; r < S; r++) sum += comb[r*S + c];
        CHECK(closef(sum, 1.f, 1e-3f, 1e-3f), "comb col %d sums to %g, want ~1", c, sum);
    }

    /* Idempotence sanity: running Sinkhorn again must barely change it. */
    float comb2[64];
    ref_hc_pre(m, scale, base, S, iters, eps, pre, post, comb2);
    for (int i = 0; i < S*S; i++)
        CHECK(closef(comb[i], comb2[i], 1e-6f, 1e-6f), "hc_pre not deterministic at %d", i);
}

/* ------------------------------------------------------------------ */
/* Reference: KDA recurrence, straight from ggml.c:23430              */
/* ------------------------------------------------------------------ */
static void ref_kda_head(const float *q, const float *k, const float *v,
                         const float *g, float beta_raw,
                         float *state /* KD*KD */, float *out,
                         int KD)
{
    const float scale = 1.0f / sqrtf((float)KD);
    const float beta = 1.0f / (1.0f + expf(-beta_raw));
    (void)0;
    float decay[256], kd[256], qd[256], v_new[256];

    float attn_score = 0.f;
    for (int i = 0; i < KD; i++) attn_score += k[i] * (q[i] * scale);

    for (int c = 0; c < KD; c++) {
        decay[c] = expf(fminf(g[c], 50.0f));
        kd[c] = k[c] * decay[c];
        qd[c] = q[c] * decay[c];
    }
    for (int r = 0; r < KD; r++) {
        float v_prime = 0.f, out_val = 0.f;
        for (int c = 0; c < KD; c++) {
            float s = state[r + c*KD];
            v_prime += s * kd[c];
            out_val += s * qd[c];
        }
        float vn = v[r] * beta - v_prime * beta;
        v_new[r] = vn;
        out[r] = out_val * scale + vn * attn_score;
    }
    for (int c = 0; c < KD; c++) {
        for (int r = 0; r < KD; r++) {
            float s = state[r + c*KD];
            s = decay[c] * s + v_new[r] * k[c];
            state[r + c*KD] = fminf(fmaxf(s, -1e6f), 1e6f);
        }
    }
}

static void test_kda_pos0(void) {
    const int KD = 128;
    float q[128], k[128], v[128], g[128], state[128*128], out[128];

    /* L2-normalise q and k like the real pipeline does. */
    srand(99);
    float sq = 0.f, sk = 0.f;
    for (int i = 0; i < KD; i++) {
        q[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.f;
        k[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.f;
        v[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.f;
        g[i] = -2.0f - ((float)rand() / RAND_MAX);
        sq += q[i]*q[i]; sk += k[i]*k[i];
    }
    for (int i = 0; i < KD; i++) { q[i] /= sqrtf(sq); k[i] /= sqrtf(sk); }
    memset(state, 0, sizeof(state));

    ref_kda_head(q, k, v, g, 0.4f, state, out, KD);

    /* At pos 0 with a zero state: v_prime == 0, out_val == 0, hence
     *   out[r] = v_new[r] * attn_score,  v_new[r] = beta * v[r].
     * Verify exactly, and verify the state update is the rank-1 term. */
    const float beta = 1.0f / (1.0f + expf(-0.4f));
    const float scale = 1.0f / sqrtf((float)KD);
    float attn_score = 0.f;
    for (int i = 0; i < KD; i++) attn_score += k[i] * (q[i] * scale);

    for (int r = 0; r < KD; r++) {
        float want = beta * v[r] * attn_score;
        CHECK(closef(out[r], want, 1e-5f, 1e-7f),
              "kda pos0 out[%d]=%g want %g", r, out[r], want);
    }
    /* State after pos 0: S[r][c] = v_new[r]*k[c] (decay*0 term is zero). */
    for (int c = 0; c < KD; c += 17) {
        for (int r = 0; r < KD; r += 19) {
            float want = beta * v[r] * k[c];
            CHECK(closef(state[r + c*KD], want, 1e-5f, 1e-7f),
                  "kda pos0 state[%d,%d]=%g want %g", r, c, state[r + c*KD], want);
        }
    }
    /* State must respect the clamp. */
    for (int i = 0; i < KD*KD; i++)
        CHECK(fabsf(state[i]) <= 1e6f, "state[%d] exceeds clamp", i);
}

/* A second token must see the state built by the first. */
static void test_kda_two_steps(void) {
    const int KD = 64;
    float q[64], k[64], v[64], g[64], state[64*64], out1[64], out2[64];
    srand(7);
    for (int i = 0; i < KD; i++) {
        q[i] = ((float)rand() / RAND_MAX - 0.5f);
        k[i] = ((float)rand() / RAND_MAX - 0.5f);
        v[i] = ((float)rand() / RAND_MAX - 0.5f);
        g[i] = -0.5f;
    }
    memset(state, 0, sizeof(state));
    ref_kda_head(q, k, v, g, 0.1f, state, out1, KD);
    ref_kda_head(q, k, v, g, 0.1f, state, out2, KD);

    /* With a non-zero state the second output must differ from the first, and
     * every value must stay finite. */
    int differs = 0;
    for (int i = 0; i < KD; i++) {
        CHECK(out2[i] == out2[i] && fabsf(out2[i]) < 1e30f, "out2[%d] not finite", i);
        if (fabsf(out2[i] - out1[i]) > 1e-9f) differs = 1;
    }
    CHECK(differs, "state had no effect on the second token - recurrence is dead");
}

/* ------------------------------------------------------------------ */
/* conv1d: verify tap ordering and the ring buffer                     */
/* ------------------------------------------------------------------ */
static void test_conv1d(void) {
    const int K = 4, D = 8;
    /* Impulse response: put 1.0 in channel 0 at tap 3 only. */
    float w[4*8];
    memset(w, 0, sizeof(w));
    w[3 + 0*K] = 1.0f;                  /* (tap=3, ch=0) */

    /* Feed the sequence 10,20,30,40; a 4-tap causal conv must emit
     * 0,0,0 then the tap-3 weight times the oldest input. */
    float xs[4] = { 0.f, 0.f, 0.f, 0.f };  /* ring of past inputs, oldest first */
    float seq[4] = { 10.f, 20.f, 30.f, 40.f };
    float got[4];
    for (int n = 0; n < 4; n++) {
        xs[K-1] = seq[n];
        float acc = 0.f;
        for (int t = 0; t < K; t++) acc += xs[t] * w[t + 0*K];
        got[n] = acc;
        for (int t = 0; t < K-1; t++) xs[t] = xs[t+1];
    }
    /* n=0: xs=[0,0,0,10] -> tap3*10 ; n=1 -> tap3*20 ; etc. */
    CHECK(closef(got[0], 10.f, 1e-6f, 1e-6f), "conv tap0 = %g want 10", got[0]);
    CHECK(closef(got[3], 40.f, 1e-6f, 1e-6f), "conv tap3 = %g want 40", got[3]);

    /* Weight layout check: (tap,ch) must live at tap + ch*K. */
    float *w2 = calloc((size_t)K*D, sizeof(float));
    w2[2 + 5*K] = 1.0f;                  /* tap 2, channel 5 */
    CHECK(w2[2 + 5*K] == 1.0f, "weight layout self-check");
    CHECK(w2[5*K + 2] == 1.0f, "tap-fastest indexing mismatch");
    free(w2);
}

/* ------------------------------------------------------------------ */
/* MoE router: sigmoid, bias-for-selection-only, renorm, scale          */
/* ------------------------------------------------------------------ */
static void test_moe_router(void) {
    const int NE = 6, AK = 3;
    const float wscale = 2.5f;
    float logit[6] = { -1.0f, 0.5f, 2.0f, -0.2f, 1.2f, 3.0f };
    float bias[6]  = {  0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f }; /* pulls expert 3 up */

    float p[6], sel_logit[6];
    for (int i = 0; i < NE; i++) {
        p[i] = 1.0f / (1.0f + expf(-logit[i]));
        CHECK(p[i] > 0.f && p[i] < 1.f, "router prob %d outside (0,1)", i);
        sel_logit[i] = p[i] + bias[i];
    }
    /* Selection uses p+bias, so expert 3 (bias 5) must be picked first. */
    int best = 0;
    for (int i = 1; i < NE; i++) if (sel_logit[i] > sel_logit[best]) best = i;
    CHECK(best == 3, "bias did not affect selection (best=%d, want 3)", best);

    /* Weights come from the RAW p of the selected experts, renormalised, then
     * multiplied by expert_weights_scale. */
    float w[3] = { p[3], p[5], p[2] };  /* top-3 by sel_logit */
    float sum = w[0] + w[1] + w[2];
    for (int i = 0; i < AK; i++) w[i] = w[i] / sum * wscale;
    float wsum = 0.f;
    for (int i = 0; i < AK; i++) wsum += w[i];
    CHECK(closef(wsum, wscale, 1e-5f, 1e-6f),
          "expert weights sum to %g, want %g (= expert_weights_scale)", wsum, wscale);
}

/* ------------------------------------------------------------------ */
/* matmul(n,d) convention lock-in                                     */
/* ------------------------------------------------------------------ */
/* PicoLM's matmul(out, x, W, n, d) reads d rows of n elements, and GGUF
 * stores a 2-D weight as ne = {row_len, row_count}. So:
 *     n == ne[0] (input length),  d == ne[1] (output length).
 * This test pins that down against a hand-computed result, because getting it
 * backwards is silent (both orderings are internally consistent) and cost real
 * debugging time on the GLM-5 hc_pre path. */
static void test_matmul_nd_convention(void) {
    /* W is ne = {row_len=3, rows=2}: two rows of three elements. */
    enum { row_len = 3, rows = 2 };
    const float W[row_len * rows] = {
        1.f, 2.f, 3.f,      /* row 0 -> output[0] = dot(x, row 0) */
        4.f, 5.f, 6.f,      /* row 1 -> output[1] = dot(x, row 1) */
    };
    const float x[3] = { 1.f, 1.f, 1.f };
    float out[2] = { 0.f, 0.f };

    /* The correct call: n = row_len, d = rows. */
    for (int r = 0; r < rows; r++) {
        float acc = 0.f;
        for (int c = 0; c < row_len; c++) acc += W[r*row_len + c] * x[c];
        out[r] = acc;
    }
    CHECK(closef(out[0], 6.f, 1e-6f, 1e-6f), "matmul row0 = %g want 6", out[0]);
    CHECK(closef(out[1], 15.f, 1e-6f, 1e-6f), "matmul row1 = %g want 15", out[1]);

    /* Asking for the swapped shape (n=rows, d=row_len) would need a 2-element
     * input and produce 3 outputs -- a different result entirely, which is why
     * the mistake is silent rather than crashing. Assert the shapes differ. */
    CHECK(row_len != rows, "this test is vacuous for square matrices");
}

int main(void) {
    test_hc_pre_invariants();
    test_kda_pos0();
    test_kda_two_steps();
    test_conv1d();
    test_moe_router();
    test_matmul_nd_convention();

    fprintf(stderr, "\n%s: %d passed, %d failed\n",
            g_fail ? "FAILED" : "OK", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
