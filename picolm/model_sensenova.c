/* SenseNova-Loop forward pass (decode, per token).
 *
 * Architecture (verified against the reference llama.cpp fork,
 * src/models/sensenova-loop.cpp + ggml_mhc_route/ggml_mhc_merge):
 *
 * Qwen3Next-style GDN/attention hybrid with mHC (manifold Hyper-
 * Connections) 2-stream residual routing and a looped weight-reuse
 * segment. 56 logical layers, 32 physical blocks: blocks 4..27 own
 * tensors and are executed twice (28..51 alias them).
 *
 * Per-layer kinds (from mhc.layer_kinds):
 *   0  plain block:  x = ffn(post(attn(norm(x)) + x)) + (attn + x)
 *   1  mHC enter:    str_a = str_b = x, then two routed sublayers
 *   4  mHC interior: streams carried in from the previous block
 *   2  mHC exit:     like 4, then x = (a' + b') * 0.5, streams reset
 *
 * Inside the mHC region every sublayer runs on the ROUTED branch,
 * not on the residual:
 *   branch = pre0*a + pre1*b           (router, sinkhorn-normalized)
 *   y      = sublayer(norm_w(branch))
 *   a'     = m00*a + m01*b + post0*y
 *   b'     = m10*a + m11*b + post1*y
 * where norm_w is attn_norm for the attention sublayer and
 * post_attention_norm for the FFN sublayer. There is NO separate
 * residual add inside the region: merge does it.
 *
 * mHC router (ggml_mhc_route semantics, verbatim):
 *   rms   = 1/sqrt(sum(a_i^2 + b_i^2) / (2*d) + 1e-6f)   [joint, literal eps]
 *   lg[k] = dot(W_row_k, [a;b]) * rms + bias[k]          (k = 0..7)
 *   pre0  = sigmoid(lg[0]);  pre1  = sigmoid(lg[1])
 *   post0 = 2*sigmoid(lg[2]); post1 = 2*sigmoid(lg[3])
 *   M = [[lg4,lg5],[lg6,lg7]], 20 logsumexp-normalize iterations
 *        (row then column, max-subtract form), M = exp(M)
 * W rows pack [a_weights (n_embd), b_weights (n_embd)] per logit;
 * GGUF stores mhc_*.weight as [8192, 8] ne-order = 8 rows of 8192.
 *
 * mHC merge (ggml_mhc_merge semantics):
 *   a'[i] = m00*a[i] + m01*b[i] + post0*y[i]
 *   b'[i] = m10*a[i] + m11*b[i] + post1*y[i]
 *
 * GDN sublayers reuse ssm_forward() with s->mhc_sublayer_only = 1: it
 * then stops after the ssm_out projection (y in the residual buffer,
 * no residual add, no FFN) so the caller can merge. Attention and FFN
 * sublayers are implemented here directly.
 *
 * NOTE: decode only for now. Prefill falls back to per-token calls.
 */
#include <math.h>
#include <string.h>
#include <stdio.h>

#include "model.h"
#include "model_internal.h"
#include "tensor.h"

#ifdef PICOLM_VIZ
#include "viz.h"
#endif

/* mHC router (decode path): fused GEMM + finish. Writes the routed branch
 * to out_branch[d] and the 6 gate scalars (post0, post1, m00, m01, m10,
 * m11) to gates[0..6). */
static void mhc_route(float *out_branch, float *gates,
                      const float *a, const float *b,
                      const float *w, const float *bias, int d) {
    float lg[8];
    for (int k = 0; k < 8; k++) {
        const float *wk = w + (size_t)k * 2 * d;
        float acc = 0.0f;
        for (int i = 0; i < d; i++) acc += wk[i]*a[i] + wk[d + i]*b[i];
        lg[k] = acc;
    }
    mhc_route_finish(out_branch, gates, a, b, lg, bias, d);
}

/* Shared router finish (also used by the batched prefill, which computes
 * the raw [n_tok][8] logits with matmul_batch and calls this per token).
 * raw8[k] = dot(W_row_k, [a;b]) WITHOUT rms/bias. Applies the joint RMS
 * scale, bias, sigmoid gates and the 20-iteration sinkhorn normalize,
 * then forms branch = pre0*a + pre1*b. gates[0..6) = {post0, post1,
 * m00, m01, m10, m11} (m* = exp of the normalized logit matrix). */
void mhc_route_finish(float *out_branch, float *gates,
                      const float *a, const float *b, const float *raw8,
                      const float *bias, int d) {
    float ss = 0.0f;
    for (int i = 0; i < d; i++) ss += a[i]*a[i] + b[i]*b[i];
    const float rms = 1.0f / sqrtf(ss / (2.0f * d) + 1e-6f);
    float lg[8];
    for (int k = 0; k < 8; k++) lg[k] = raw8[k] * rms + bias[k];

    const float pre0  = 1.0f / (1.0f + expf(-lg[0]));
    const float pre1  = 1.0f / (1.0f + expf(-lg[1]));
    const float post0 = 2.0f / (1.0f + expf(-lg[2]));
    const float post1 = 2.0f / (1.0f + expf(-lg[3]));

    float m[2][2] = { { lg[4], lg[5] }, { lg[6], lg[7] } };
    for (int it = 0; it < 20; it++) {
        for (int r = 0; r < 2; r++) {
            const float mx  = m[r][0] > m[r][1] ? m[r][0] : m[r][1];
            const float lse = mx + logf(expf(m[r][0] - mx) + expf(m[r][1] - mx));
            m[r][0] -= lse;
            m[r][1] -= lse;
        }
        for (int col = 0; col < 2; col++) {
            const float mx  = m[0][col] > m[1][col] ? m[0][col] : m[1][col];
            const float lse = mx + logf(expf(m[0][col] - mx) + expf(m[1][col] - mx));
            m[0][col] -= lse;
            m[1][col] -= lse;
        }
    }

    for (int i = 0; i < d; i++) out_branch[i] = pre0*a[i] + pre1*b[i];
    gates[0] = post0;
    gates[1] = post1;
    gates[2] = expf(m[0][0]);
    gates[3] = expf(m[0][1]);
    gates[4] = expf(m[1][0]);
    gates[5] = expf(m[1][1]);
}

/* mHC merge (shared with the batched prefill): a' = m00*a + m01*b +
 * post0*y; b' = m10*a + m11*b + post1*y. a_out may alias a; b_out may
 * alias b. y must not alias a_out/b_out. */
void mhc_merge_rows(float *a_out, float *b_out,
                    const float *a, const float *b, const float *y,
                    const float *gates, int d) {
    const float post0 = gates[0], post1 = gates[1];
    const float m00 = gates[2], m01 = gates[3];
    const float m10 = gates[4], m11 = gates[5];
    for (int i = 0; i < d; i++) {
        const float av = a[i], bv = b[i], yv = y[i];
        a_out[i] = m00*av + m01*bv + post0*yv;
        b_out[i] = m10*av + m11*bv + post1*yv;
    }
}

/* Attention sublayer on an ALREADY-NORMED input (xb_normed): fused Q|gate,
 * per-head QK-norm, partial rope, GQA, sigmoid gate, o_proj.
 * Output written to out (dim floats). Uses s->q, s->hb, s->xb2 as scratch. */
static void sensenova_attn_sublayer(model_t *m, run_state_t *s,
                                    layer_weights_t *lw, int l, int pos,
                                    int dim, int n_heads, int n_kv_heads,
                                    int head_dim, int q_dim, int q_full_dim,
                                    int kv_dim, int seq_len, int rope_half,
                                    const float *xb_normed, float *out) {
    (void)dim;
    /* Q projection (interleaved Q|gate per head) */
    matmul(s->q, xb_normed, lw->attn_q, m->config.n_embd, q_full_dim, lw->type_attn_q);
    float *gate = s->hb;
    for (int h = 0; h < n_heads; h++) {
        memmove(s->q + h * head_dim, s->q + h * 2 * head_dim,
                head_dim * sizeof(float));
        memmove(gate + h * head_dim, s->q + h * 2 * head_dim + head_dim,
                head_dim * sizeof(float));
    }
    /* K into s->xb2 (hb holds the gate) */
    matmul(s->xb2, xb_normed, lw->attn_k, m->config.n_embd, kv_dim, lw->type_attn_k);
    /* QK-norm per head */
    if (lw->attn_q_norm) {
        for (int h = 0; h < n_heads; h++)
            rmsnorm(s->q + h*head_dim, s->q + h*head_dim,
                    s->attn_q_norm_w[l], head_dim, m->config.rms_norm_eps);
        for (int h = 0; h < n_kv_heads; h++)
            rmsnorm(s->xb2 + h*head_dim, s->xb2 + h*head_dim,
                    s->attn_k_norm_w[l], head_dim, m->config.rms_norm_eps);
    }
    /* RoPE (partial: first rope_half*2 of head_dim rotates) */
    const float *cos_pos = s->rope_cos + (size_t)pos * rope_half;
    const float *sin_pos = s->rope_sin + (size_t)pos * rope_half;
    rope(s->q, s->xb2, head_dim, n_heads, n_kv_heads,
         cos_pos, sin_pos, m->config.rope_type, rope_half);
    /* V: reuse xb_normed as scratch (consumed) */
    float *v_tmp = s->hb2;  /* gate lives in hb, K in xb2, V needs its own scratch */
    matmul(v_tmp, xb_normed, lw->attn_v, m->config.n_embd, kv_dim, lw->type_attn_v);
    /* KV cache at this layer's ordinal */
    int kv_ord = s->kv_layer_ordinal[l];
    uint8_t *kcache_layer = s->key_cache + (size_t)kv_ord * seq_len * s->kv_row_size_k;
    uint8_t *vcache_layer = s->val_cache + (size_t)kv_ord * seq_len * s->kv_row_size_v;
    { uint8_t *key_pos = kcache_layer + (size_t)pos * s->kv_row_size_k;
      if (s->kv_type_k == KV_CACHE_Q8_0) quantize_row_q8_0(s->xb2, key_pos, kv_dim);
      else if (s->kv_type_k == KV_CACHE_Q4_0) quantize_row_q4_0(s->xb2, key_pos, kv_dim);
      else { uint16_t *kf = (uint16_t *)key_pos;
             for (int d = 0; d < kv_dim; d++) kf[d] = fp32_to_fp16(s->xb2[d]); } }
    { uint8_t *val_pos = vcache_layer + (size_t)pos * s->kv_row_size_v;
      if (s->kv_type_v == KV_CACHE_Q8_0) quantize_row_q8_0(v_tmp, val_pos, kv_dim);
      else if (s->kv_type_v == KV_CACHE_Q4_0) quantize_row_q4_0(v_tmp, val_pos, kv_dim);
      else { uint16_t *vf = (uint16_t *)val_pos;
             for (int d = 0; d < kv_dim; d++) vf[d] = fp32_to_fp16(v_tmp[d]); } }
    /* Attention */
    attn_group_ctx_t gctx;
    gctx.kv_mul = n_heads / n_kv_heads; gctx.head_dim = head_dim; gctx.pos = pos;
    gctx.kv_type_k = s->kv_type_k; gctx.kv_type_v = s->kv_type_v;
    gctx.kv_row_size_k = s->kv_row_size_k; gctx.kv_row_size_v = s->kv_row_size_v;
    gctx.kv_head_stride_k = s->kv_head_stride_k; gctx.kv_head_stride_v = s->kv_head_stride_v;
    gctx.kcache = kcache_layer; gctx.vcache = vcache_layer;
    gctx.q = s->q; gctx.xb = out;
    gctx.n_kv_heads = n_kv_heads;
    gctx.kv_hadamard_k = 0; gctx.kv_hadamard_v = 0; gctx.kv_hadamard_size = 0;
    gctx.attn_scale = 1.0f / sqrtf((float)head_dim);
    gctx.n_swa = m->config.n_swa_layer[l];
    tensor_parallel_for(n_kv_heads, attention_group, &gctx);
    /* sigmoid gate */
    for (int i = 0; i < q_dim; i++) {
        float g = 1.0f / (1.0f + expf(-gate[i]));
        out[i] *= g;
    }
    /* o_proj: s->xb is free here (the normed input was consumed by the Q/K/V
     * projections; in the mHC path it held the branch, in the plain path the
     * normed x). Result (the sublayer output y) is returned in s->xb. */
    matmul(s->xb, out, lw->attn_output, q_dim, dim, lw->type_attn_output);
}

/* FFN sublayer on an ALREADY-NORMED input. Output to out. */
static void sensenova_ffn_sublayer(run_state_t *s,
                                   layer_weights_t *lw, int dim, int n_ffn,
                                   const float *xb_normed, float *out) {
    matmul(s->hb, xb_normed, lw->ffn_gate, dim, n_ffn, lw->type_ffn_gate);
    matmul(s->hb2, xb_normed, lw->ffn_up, dim, n_ffn, lw->type_ffn_up);
    silu(s->hb, n_ffn);
    elemwise_mul(s->hb, s->hb, s->hb2, n_ffn);
    matmul(out, s->hb, lw->ffn_down, n_ffn, dim, lw->type_ffn_down);
}

/* Run one layer (index l) of the SenseNova-Loop stack for one token.
 *
 * x_in/out: residual stream, dim floats. For mHC blocks the second stream
 * lives in s->mhc_str_b and *streams_active carries the enter/exit state
 * across layers (the caller owns it; kind 1 seeds both streams from x,
 * kind 2 merges them into x and clears the flag).
 *
 * This is the shared body of the decode loop (model_forward_sensenova)
 * and the per-token prefill fallback, so the two paths cannot drift. */
void sensenova_layer(model_t *m, run_state_t *s, int l, int pos,
                     int *streams_active) {
    model_config_t *c = &m->config;
    layer_weights_t *lw = &m->weights.layers[l];

    const int dim = c->n_embd;
    const int n_ffn = c->n_ffn;
    const int n_heads = c->n_heads;
    const int n_kv_heads = c->n_kv_heads;
    const int head_dim = c->head_dim;
    const int q_dim = n_heads * head_dim;
    const int q_full_dim = q_dim * 2;      /* fused Q|gate, per-head interleaved */
    const int kv_dim = n_kv_heads * head_dim;
    const int seq_len = c->max_seq_len;
    const int rope_half = (c->rope_dim > 0 ? c->rope_dim : head_dim) / 2;

    float *str_b = s->mhc_str_b;      /* second stream */
    float *branch = s->xb;            /* router output -> normed sublayer input */
    float *sub_out = s->xb2;          /* sublayer output (pre-merge) */
    float gates[6];
    const uint8_t kind = c->mhc_layer_kinds[l];

        if (kind == 0) {
        /* ---- Plain block: standard post-norm residual flow ---- */
        if (c->has_ssm && !lw->is_attn_layer) {
            float *ssm_residual = s->xb2;
            s->mhc_sublayer_only = 0;
            ssm_forward(m, s, s->x, ssm_residual, lw, l, pos, NULL);
        } else {
            /* Attention sublayer: y lands in s->xb (see o_proj note in
             * sensenova_attn_sublayer), residual add into s->x. */
            rmsnorm(s->xb, s->x, s->attn_norm_w[l], dim, c->rms_norm_eps);
            sensenova_attn_sublayer(m, s, lw, l, pos, dim, n_heads,
                                    n_kv_heads, head_dim, q_dim, q_full_dim,
                                    kv_dim, seq_len, rope_half, s->xb, s->xb2);
            vec_add(s->x, s->xb, dim);
            /* FFN sublayer: norm into s->xb (free again -- the o_proj wrote
             * y there and vec_add consumed it), y into s->xb2, residual add. */
            rmsnorm(s->xb, s->x, s->post_attn_norm_w[l], dim, c->rms_norm_eps);
            sensenova_ffn_sublayer(s, lw, dim, n_ffn, s->xb, s->xb2);
            vec_add(s->x, s->xb2, dim);
        }
#ifdef PICOLM_VIZ
        viz_push_row(l, pos, s->x, dim);
#endif
        return;
    }

    /* ---- mHC block (1 = enter, 4 = interior, 2 = exit) ---- */
    if (kind == 1 || !*streams_active) {
        memcpy(str_b, s->x, dim * sizeof(float));
        *streams_active = 1;
    }

    /* -- sublayer 1: attention (GDN or full), routed -- */
    float *y;
    mhc_route(branch, gates, s->x, str_b,
              (const float *)lw->mhc_attn_w, (const float *)lw->mhc_attn_b, dim);
    if (c->has_ssm && !lw->is_attn_layer) {
        /* GDN: ssm_forward norms x internally (attn_norm_w is the norm
         * weight the reference uses for the attn sublayer too), writes
         * y to the residual arg and stops (mhc_sublayer_only). */
        s->mhc_sublayer_only = 1;
        ssm_forward(m, s, branch, sub_out, lw, l, pos, NULL);
        s->mhc_sublayer_only = 0;
        y = sub_out;
    } else {
        rmsnorm(branch, branch, s->attn_norm_w[l], dim, c->rms_norm_eps);
        sensenova_attn_sublayer(m, s, lw, l, pos, dim, n_heads,
                                n_kv_heads, head_dim, q_dim, q_full_dim,
                                kv_dim, seq_len, rope_half, branch, sub_out);
        /* the attention sublayer returns y in s->xb (o_proj target) */
        y = s->xb;
    }
    /* merge: a'=x (in place), b'=str_b (in place), y from the sublayer */
    mhc_merge_rows(s->x, str_b, s->x, str_b, y, gates, dim);

    /* -- sublayer 2: FFN, routed -- */
    mhc_route(branch, gates, s->x, str_b,
              (const float *)lw->mhc_ffn_w, (const float *)lw->mhc_ffn_b, dim);
    rmsnorm(branch, branch, s->post_attn_norm_w[l], dim, c->rms_norm_eps);
    sensenova_ffn_sublayer(s, lw, dim, n_ffn, branch, sub_out);
    mhc_merge_rows(s->x, str_b, s->x, str_b, sub_out, gates, dim);

    if (kind == 2) {
        for (int i = 0; i < dim; i++) s->x[i] = 0.5f * (s->x[i] + str_b[i]);
        *streams_active = 0;
    }
#ifdef PICOLM_VIZ
    viz_push_row(l, pos, s->x, dim);
#endif
}

float *model_forward_sensenova(model_t *m, int token, int pos) {
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;

    if (pos >= c->max_seq_len) {
        fprintf(stderr, "WARN: model_forward_sensenova pos=%d >= max_seq_len=%d\n",
                pos, c->max_seq_len);
        return s->logits;
    }

    const int dim = c->n_embd;

    /* 1. Embedding */
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
        dequantize_row(embd_row, s->x, dim, w->type_token_embd);
    }

    /* 2. Layer loop (stream state carried across layers) */
    int streams_active = 0;
    for (int l = 0; l < c->n_layers; l++) {
        BENCH_LAYER_START();
        sensenova_layer(m, s, l, pos, &streams_active);
        BENCH_LAYER_END(l, 0);
    }

    /* 3. Final norm + LM head */
    rmsnorm(s->x, s->x, s->output_norm_w, dim, c->rms_norm_eps);
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);
    return s->logits;
}
