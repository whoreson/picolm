#include "model.h"
#include "tensor.h"
#include "quant.h"
#include "model_internal.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef PICOLM_GPU
#include "backend_gpu.h"
#endif


/* Get pointer to expert e's sub-tensor within a 3D expert tensor.
 * For gate_exps [n_embd, n_ff_exp, n_expert]: each expert is [n_embd, n_ff_exp].
 * For down_exps [n_ff_exp, n_embd, n_expert]: each expert is [n_ff_exp, n_embd].
 * GGUF stores dims row-major: dims[0] varies fastest.
 * Expert e starts at: e * dim1 * gguf_type_row_size(type, dim0)
 * (each row has dim0 elements, and there are dim1 rows per expert) */
static const void *get_expert_slice(const void *base, int expert,
                                    int dim0, int dim1, gguf_type_t type) {
    size_t expert_stride = (size_t)dim1 * gguf_type_row_size(type, dim0);
    return (const void *)((const uint8_t *)base + expert * expert_stride);
}

/* Parallel expert worker for moe_forward: processes one expert using
 * per-thread scratch buffers. Uses matmul_q8_seq (not matmul_q8) to
 * avoid racing on the global n_threads variable and re-entering the
 * thread pool from inside a tensor_parallel_for worker. */
typedef struct {
    const block_q8_0 *qx;       /* pre-quantized input */
    const float *qx_d;          /* input Q8_0 deltas */
    int dim, n_ff;
    int *ids;                   /* expert ids [n_used] */
    float *weights;             /* expert weights [n_used] */
    gguf_type_t type_gate, type_up, type_down;
    const void *gate_exps, *up_exps, *down_exps;
    float *expert_out;          /* output [n_used * dim], each expert writes to slot i*dim */
    run_state_t *s;
} moe_expert_ctx;

static void moe_expert_worker(int i, void *vctx) {
    moe_expert_ctx *ctx = (moe_expert_ctx *)vctx;
    int dim = ctx->dim, n_ff = ctx->n_ff;

    int eid = ctx->ids[i];
    float w_i = ctx->weights[i];

    /* Get per-thread scratch buffer */
    unsigned tid = tensor_get_thread_id();
    int nt = tensor_get_n_threads();
    if (tid >= (unsigned)nt) tid = 0;
    char *scratch = (char *)ctx->s->moe_thread_scratch + tid * ctx->s->moe_thread_stride;

    float *gate_buf = (float *)scratch;
    float *up_buf = gate_buf + n_ff;
    /* xb2 + acc area follows (dim * 2 floats), then Q8 buffers */
    char *q8_ptr = (char *)(up_buf + n_ff) + (size_t)dim * 2 * sizeof(float);
    block_q8_0 *down_qx = (block_q8_0 *)q8_ptr;
    float *down_qx_d = (float *)((char *)down_qx + gguf_type_row_size(GGUF_TYPE_Q8_0, n_ff));

    const void *gate_w = get_expert_slice(ctx->gate_exps, eid, dim, n_ff, ctx->type_gate);
    const void *up_w = get_expert_slice(ctx->up_exps, eid, dim, n_ff, ctx->type_up);
    const void *down_w = get_expert_slice(ctx->down_exps, eid, n_ff, dim, ctx->type_down);

    /* Use matmul_q8_seq: sequential, no thread pool, safe inside tensor_parallel_for */
    matmul_q8_seq(gate_buf, ctx->qx, ctx->qx_d, gate_w, dim, n_ff, ctx->type_gate);
    matmul_q8_seq(up_buf, ctx->qx, ctx->qx_d, up_w, dim, n_ff, ctx->type_up);

    /* SwiGLU: silu(gate) * up */
    silu(gate_buf, n_ff);
    elemwise_mul(gate_buf, gate_buf, up_buf, n_ff);

    /* Quantize SwiGLU output for Q8xQ8 down projection */
    {
        size_t dnb = n_ff / 32;
        quantize_row_q8_0(gate_buf, down_qx, n_ff);
        for (size_t b = 0; b < dnb; b++) {
            down_qx_d[b] = fp16_to_fp32(down_qx[b].d);
        }
    }

    /* Down projection */
    float *out = ctx->expert_out + (size_t)i * dim;
    matmul_q8_seq(out, down_qx, down_qx_d, down_w, n_ff, dim, ctx->type_down);

    /* Scale by expert weight */
#ifdef PICOLM_AVX512
    {
        __m512 bw = _mm512_set1_ps(w_i);
        int di = 0;
        for (; di + 23 < dim; di += 16) {
            __m512 v0 = _mm512_loadu_ps(out + di);
            __m512 v1 = _mm512_loadu_ps(out + di + 8);
            _mm512_storeu_ps(out + di, _mm512_mul_ps(bw, v0));
            _mm512_storeu_ps(out + di + 8, _mm512_mul_ps(bw, v1));
        }
        for (; di + 15 < dim; di += 16) {
            __m512 v = _mm512_loadu_ps(out + di);
            _mm512_storeu_ps(out + di, _mm512_mul_ps(bw, v));
        }
        for (; di < dim; di++) out[di] *= w_i;
    }
#elif defined(PICOLM_AVX2)
    {
        __m256 bw = _mm256_set1_ps(w_i);
        int di = 0;
        for (; di + 19 < dim; di += 16) {
            __m256 v0 = _mm256_loadu_ps(out + di);
            __m256 v1 = _mm256_loadu_ps(out + di + 4);
            __m256 v2 = _mm256_loadu_ps(out + di + 8);
            __m256 v3 = _mm256_loadu_ps(out + di + 12);
            _mm256_storeu_ps(out + di, _mm256_mul_ps(bw, v0));
            _mm256_storeu_ps(out + di + 4, _mm256_mul_ps(bw, v1));
            _mm256_storeu_ps(out + di + 8, _mm256_mul_ps(bw, v2));
            _mm256_storeu_ps(out + di + 12, _mm256_mul_ps(bw, v3));
        }
        for (; di < dim; di++) out[di] *= w_i;
    }
#elif defined(PICOLM_AVX)
    {
        __m128 bw = _mm_set1_ps(w_i);
        int di = 0;
        for (; di + 7 < dim; di += 8) {
            __m128 v0 = _mm_loadu_ps(out + di);
            __m128 v1 = _mm_loadu_ps(out + di + 4);
            _mm_storeu_ps(out + di, _mm_mul_ps(bw, v0));
            _mm_storeu_ps(out + di + 4, _mm_mul_ps(bw, v1));
        }
        for (; di < dim; di++) out[di] *= w_i;
    }
#elif defined(PICOLM_SSE2)
    {
        __m128 bw = _mm_set1_ps(w_i);
        int di = 0;
        for (; di + 3 < dim; di += 4) {
            __m128 v = _mm_loadu_ps(out + di);
            _mm_storeu_ps(out + di, _mm_mul_ps(bw, v));
        }
        for (; di < dim; di++) out[di] *= w_i;
    }
#elif defined(PICOLM_NEON)
    {
        float32x4_t bw = vdupq_n_f32(w_i);
        int di = 0;
        for (; di + 3 < dim; di += 4) {
            float32x4_t v = vld1q_f32(out + di);
            vst1q_f32(out + di, vmulq_f32(bw, v));
        }
        for (; di < dim; di++) out[di] *= w_i;
    }
#else
    for (int d = 0; d < dim; d++) out[d] *= w_i;
#endif
}

/* MoE forward pass: router + top-K expert selection + SwiGLU per expert + shared expert.
 * Input: x[n_embd], Output: residual[n_embd] (additive to input) */
/* Optimized MoE forward: pre-quantize x, Q8xQ8 dot products for gate+up,
 * AVX-512 vectorized accumulation. Experts processed in parallel via
 * tensor_parallel_for for multi-threaded generation. */
void moe_forward(model_t *m, run_state_t *s, const float *x, float *residual,
                        const layer_weights_t *lw) {
    model_config_t *c = &m->config;
    int dim = c->n_embd;
    int n_ff = c->n_ff_exp;
    int n_expert = c->n_expert;
    int n_used = c->n_expert_used;
    float *logits = s->expert_logits;
    int *ids = s->expert_ids;
    float *weights = s->expert_weights;
    float *moe_out = s->moe_out;
    float *expert_out = NULL;

    /* 1. Router: logits = x @ ffn_gate_inp  [n_embd, n_expert] -> [n_expert] */
    matmul(logits, (float *)x, lw->ffn_gate_inp, dim, n_expert, lw->type_ffn_gate_inp);

    /* 2. Softmax over logits */
    {
        float max_l = logits[0];
        for (int i = 1; i < n_expert; i++) {
            if (logits[i] > max_l) max_l = logits[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < n_expert; i++) {
            logits[i] = expf(logits[i] - max_l);
            sum += logits[i];
        }
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < n_expert; i++) {
            logits[i] *= inv_sum;
        }
    }

    /* 3. Find top-K experts (simple selection sort for small K) */
    {
        int idx[256];
        for (int i = 0; i < n_expert; i++) idx[i] = i;
        for (int i = 0; i < n_used; i++) {
            int best = i;
            for (int j = i + 1; j < n_expert; j++) {
                if (logits[idx[j]] > logits[idx[best]]) best = j;
            }
            { int t = idx[i]; idx[i] = idx[best]; idx[best] = t; }
            ids[i] = idx[i];
            weights[i] = logits[idx[i]];
        }
    }

    /* 3b. Normalize top-K weights by their sum */
    {
        float wsum = 0.0f;
        for (int i = 0; i < n_used; i++) wsum += weights[i];
        float inv_wsum = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;
        for (int i = 0; i < n_used; i++) weights[i] *= inv_wsum;
    }

    /* 5. Pre-quantize input x to Q8_0 ONCE (Phase A + D).
     * All expert gate+up projections reuse this quantized buffer via matmul_q8,
     * saving 16 redundant quantizations per MoE layer. */
    {
        size_t nb = dim / 32;
        block_q8_0 *qx = s->shared_qx;
        float *qx_d = s->shared_qx_d;
        quantize_row_q8_0(x, qx, dim);
        for (size_t bi = 0; bi < nb; bi++) {
            qx_d[bi] = fp16_to_fp32(qx[bi].d);
        }

        gguf_type_t type_gate = lw->type_ffn_gate_exps;
        gguf_type_t type_up = lw->type_ffn_up_exps;
        gguf_type_t type_down = lw->type_ffn_down_exps;

        /* Parallel expert dispatch: each expert runs in its own thread.
         * matmul_q8_seq is used (not matmul_q8) to avoid racing on the
         * global n_threads and re-entering the thread pool.
         * Use calloc instead of alloca to avoid Windows stack overflow. */
        expert_out = (float *)calloc((size_t)n_used * dim, sizeof(float));
        if (!expert_out) { fprintf(stderr, "OOM: expert_out\n"); return; }

        moe_expert_ctx ctx = {
            .qx = s->shared_qx,
            .qx_d = s->shared_qx_d,
            .dim = dim,
            .n_ff = n_ff,
            .ids = ids,
            .weights = weights,
            .type_gate = type_gate,
            .type_up = type_up,
            .type_down = type_down,
            .gate_exps = lw->ffn_gate_exps,
            .up_exps = lw->ffn_up_exps,
            .down_exps = lw->ffn_down_exps,
            .expert_out = expert_out,
            .s = s,
        };

        tensor_parallel_for(n_used, moe_expert_worker, &ctx);

        /* Reduce per-expert outputs into moe_out */
        memset(moe_out, 0, dim * sizeof(float));
        for (int i = 0; i < n_used; i++) {
            float *eo = expert_out + (size_t)i * dim;
#ifdef PICOLM_AVX512
            {
                int di = 0;
                for (; di + 23 < dim; di += 16) {
                    __m512 v0 = _mm512_loadu_ps(moe_out + di);
                    __m512 v1 = _mm512_loadu_ps(eo + di);
                    __m512 v2 = _mm512_loadu_ps(moe_out + di + 8);
                    __m512 v3 = _mm512_loadu_ps(eo + di + 8);
                    _mm512_storeu_ps(moe_out + di, _mm512_add_ps(v0, v1));
                    _mm512_storeu_ps(moe_out + di + 8, _mm512_add_ps(v2, v3));
                }
                for (; di < dim; di++) moe_out[di] += eo[di];
            }
#elif defined(PICOLM_AVX2)
            {
                int di = 0;
                for (; di + 20 < dim; di += 16) {
                    __m256 v0 = _mm256_loadu_ps(moe_out + di);
                    __m256 v1 = _mm256_loadu_ps(eo + di);
                    __m256 v2 = _mm256_loadu_ps(moe_out + di + 4);
                    __m256 v3 = _mm256_loadu_ps(eo + di + 4);
                    __m256 v4 = _mm256_loadu_ps(moe_out + di + 8);
                    __m256 v5 = _mm256_loadu_ps(eo + di + 8);
                    __m256 v6 = _mm256_loadu_ps(moe_out + di + 12);
                    __m256 v7 = _mm256_loadu_ps(eo + di + 12);
                    _mm256_storeu_ps(moe_out + di, _mm256_add_ps(v0, v1));
                    _mm256_storeu_ps(moe_out + di + 4, _mm256_add_ps(v2, v3));
                    _mm256_storeu_ps(moe_out + di + 8, _mm256_add_ps(v4, v5));
                    _mm256_storeu_ps(moe_out + di + 12, _mm256_add_ps(v6, v7));
                }
                for (; di < dim; di++) moe_out[di] += eo[di];
            }
#elif defined(PICOLM_AVX)
            {
                int di = 0;
                for (; di + 7 < dim; di += 8) {
                    __m128 v0 = _mm_loadu_ps(moe_out + di);
                    __m128 v1 = _mm_loadu_ps(eo + di);
                    __m128 v2 = _mm_loadu_ps(moe_out + di + 4);
                    __m128 v3 = _mm_loadu_ps(eo + di + 4);
                    _mm_storeu_ps(moe_out + di, _mm_add_ps(v0, v1));
                    _mm_storeu_ps(moe_out + di + 4, _mm_add_ps(v2, v3));
                }
                for (; di < dim; di++) moe_out[di] += eo[di];
            }
#elif defined(PICOLM_SSE2)
            {
                int di = 0;
                for (; di + 3 < dim; di += 4) {
                    __m128 v0 = _mm_loadu_ps(moe_out + di);
                    __m128 v1 = _mm_loadu_ps(eo + di);
                    _mm_storeu_ps(moe_out + di, _mm_add_ps(v0, v1));
                }
                for (; di < dim; di++) moe_out[di] += eo[di];
            }
#elif defined(PICOLM_NEON)
            {
                int di = 0;
                for (; di + 3 < dim; di += 4) {
                    float32x4_t v0 = vld1q_f32(moe_out + di);
                    float32x4_t v1 = vld1q_f32(eo + di);
                    vst1q_f32(moe_out + di, vaddq_f32(v0, v1));
                }
                for (; di < dim; di++) moe_out[di] += eo[di];
            }
#else
            for (int d = 0; d < dim; d++) moe_out[d] += eo[d];
#endif
        }
    }

    /* 6. Shared expert — Q8×Q8 with pre-allocated buffers */
    {
        int saved_threads = tensor_get_n_threads();
        tensor_set_n_threads(1);
        size_t nb = dim / 32;
        quantize_row_q8_0(x, s->shared_qx, dim);
        for (size_t bi = 0; bi < nb; bi++) {
            s->shared_qx_d[bi] = fp16_to_fp32(s->shared_qx[bi].d);
        }
        matmul_q8(s->hb, s->shared_qx, s->shared_qx_d, lw->ffn_gate_shexp, dim, c->n_ff_shexp, lw->type_ffn_gate_shexp);
        matmul_q8(s->hb2, s->shared_qx, s->shared_qx_d, lw->ffn_up_shexp, dim, c->n_ff_shexp, lw->type_ffn_up_shexp);
        silu(s->hb, c->n_ff_shexp);
        elemwise_mul(s->hb, s->hb, s->hb2, c->n_ff_shexp);
        {
            size_t dnb = c->n_ff_shexp / 32;
            quantize_row_q8_0(s->hb, s->shared_down_qx, c->n_ff_shexp);
            for (size_t bi = 0; bi < dnb; bi++) {
                s->shared_down_qx_d[bi] = fp16_to_fp32(s->shared_down_qx[bi].d);
            }
            matmul_q8(s->xb2, s->shared_down_qx, s->shared_down_qx_d, lw->ffn_down_shexp, c->n_ff_shexp, dim, lw->type_ffn_down_shexp);
        }
        tensor_set_n_threads(saved_threads);
    }

    /* Shared expert sigmoid gate: sigmoid(x @ ffn_gate_inp_shexp) */
    {
        float gate_val;
        matmul(&gate_val, (float *)x, lw->ffn_gate_inp_shexp, dim, 1, lw->type_ffn_gate_inp_shexp);
        gate_val = 1.0f / (1.0f + expf(-gate_val));
        for (int d = 0; d < dim; d++) s->xb2[d] *= gate_val;
    }

    /* 7. Combine: moe_out + shared */
    for (int d = 0; d < dim; d++) {
        residual[d] = moe_out[d] + s->xb2[d];
    }
    free(expert_out);
}


/* moe_forward_batch: mm_id-style batched MoE forward pass.
 *
 * Strategy: follow llama.cpp's ggml_mul_mat_id pattern. For each
 * projection (gate, up, down), sweep all 256 experts linearly.
 * For each expert, gather the tokens that selected it and compute
 * the projection. This ensures each expert's ~1.1MB weights are
 * streamed from RAM exactly once per layer.
 *
 * Output layout: [n_tokens * n_used * n_ff] for gate/up,
 *                [n_tokens * n_used * dim] for down.
 * Final accumulation is in Top-K order per token.
 *
 * x_batch:        input [n_tokens * dim], stride = dim
 * residual_batch: output [n_tokens * dim], stride = dim
 */
void moe_forward_batch(model_t *m, run_state_t *s,
                              const float *x_batch, float *residual_batch,
                              int n_tokens, const layer_weights_t *lw) {
    model_config_t *c = &m->config;
    int dim = c->n_embd;
    int n_ff = c->n_ff_exp;
    int n_expert = c->n_expert;
    int n_used = c->n_expert_used;

    /* Single-token fast path: use moe_forward directly.
     * The batched path has overhead from routing map setup and tensor_parallel_for
     * dispatch across 256 experts that outweighs the benefit for a single token. */
    if (n_tokens == 1) {
        moe_forward(m, s, (const float *)x_batch, residual_batch, lw);
        return;
    }

    int saved_threads = tensor_get_n_threads();
    /* Phase 1-2: routing and quantization are lightweight, keep single-threaded */
    tensor_set_n_threads(1);

    /* Pre-allocated quantization buffer variables */
    int q8_buf_per_token = s->moe_q8_buf_per_token;
    int qx_d_off = s->moe_qx_d_off;
    float *qx_all = s->moe_qx_all;

    /* Down projection Q8_0 per-token buffer size (for n_ff, not n_embd) */
    size_t down_q8_rb = n_ff / 32;
    size_t down_q8_data_off = (down_q8_rb * sizeof(block_q8_0) + sizeof(float) - 1) / sizeof(float);
    int down_q8_per_token = (int)(down_q8_data_off + down_q8_rb);

    /* On-demand mm_id buffers (sized to actual batch size).
     * Reallocate if batch size has grown since last call. */
    size_t gateup_sz = (size_t)n_tokens * n_used * n_ff * sizeof(float);
    size_t down_sz = (size_t)n_tokens * n_used * dim * sizeof(float);
    float *mm_gate_out = s->mm_gate_out;
    float *mm_up_out = s->mm_up_out;
    float *mm_down_out = s->mm_down_out;

    /* Allocate or grow if needed */
    if (!mm_gate_out || s->mm_gateup_alloc < gateup_sz) {
#ifdef _WIN32
        _aligned_free(mm_gate_out);
        _aligned_free(mm_up_out);
        _aligned_free(mm_down_out);
#else
        free(mm_gate_out);
        free(mm_up_out);
        free(mm_down_out);
#endif
#ifdef _WIN32
        s->mm_gate_out = mm_gate_out = (float *)_aligned_malloc(gateup_sz + 4095, 64);
        s->mm_up_out = mm_up_out = (float *)_aligned_malloc(gateup_sz + 4095, 64);
        s->mm_down_out = mm_down_out = (float *)_aligned_malloc(down_sz + 4095, 64);
#else
        s->mm_gate_out = mm_gate_out = (float *)aligned_alloc(64, gateup_sz + 4095);
        s->mm_up_out = mm_up_out = (float *)aligned_alloc(64, gateup_sz + 4095);
        s->mm_down_out = mm_down_out = (float *)aligned_alloc(64, down_sz + 4095);
#endif
        s->mm_gateup_alloc = gateup_sz;
        s->mm_down_alloc = down_sz;
    } else if (s->mm_down_alloc < down_sz) {
#ifdef _WIN32
        _aligned_free(mm_down_out);
#else
        free(mm_down_out);
#endif
#ifdef _WIN32
        s->mm_down_out = mm_down_out = (float *)_aligned_malloc(down_sz + 4095, 64);
#else
        s->mm_down_out = mm_down_out = (float *)aligned_alloc(64, down_sz + 4095);
#endif
        s->mm_down_alloc = down_sz;
    }

    /* Per-token expert routing tables: [n_tokens][n_used] */
    int all_ids[n_tokens][8];       /* 8 = max n_expert_used */
    float all_weights[n_tokens][8];

    /* Per-token: number of experts assigned (for variable top-K) */
    int n_experts_per_token[n_tokens];

    /* ---- Phase 1: Route all tokens ---- */
    {
        int *idx = (int *)malloc(n_expert * sizeof(int));

        /* Batched router: all tokens through ffn_gate_inp in one matmul_batch */
        {
            float *logits_batch = s->expert_logits;
            matmul_batch(logits_batch, (float *)x_batch, n_tokens,
                lw->ffn_gate_inp, dim, n_expert, lw->type_ffn_gate_inp);

            for (int t = 0; t < n_tokens; t++) {
                float *logits = logits_batch + t * n_expert;
                softmax(logits, n_expert);

                /* Top-K selection */
                {
                    int *ids = all_ids[t];
                    float *weights = all_weights[t];
                for (int i = 0; i < n_expert; i++) idx[i] = i;
                for (int i = 0; i < n_used; i++) {
                    int best = i;
                    for (int j = i + 1; j < n_expert; j++) {
                        if (logits[idx[j]] > logits[idx[best]]) best = j;
                    }
                    { int tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp; }
                    ids[i] = idx[i];
                    weights[i] = logits[idx[i]];
                }

                /* Normalize weights by their sum */
                {
                    float wsum = 0.0f;
                    for (int i = 0; i < n_used; i++) wsum += weights[i];
                    float inv_wsum = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;
                    for (int i = 0; i < n_used; i++) weights[i] *= inv_wsum;
                }
                }
            n_experts_per_token[t] = n_used;
            }
        }
        free(idx);
    }

    /* ---- Phase 2: Quantize all token inputs to Q8_0 (pre-allocated) ---- */
    {
        size_t q8_row_blocks = dim / 32;
        for (int t = 0; t < n_tokens; t++) {
            float *tbuf = qx_all + t * q8_buf_per_token;
            block_q8_0 *qx = (block_q8_0 *)tbuf;
            float *qx_d = tbuf + qx_d_off;
            quantize_row_q8_0(x_batch + t * dim, qx, dim);
            for (size_t bi = 0; bi < q8_row_blocks; bi++) {
                qx_d[bi] = fp16_to_fp32(qx[bi].d);
            }
        }
    }

    /* ---- Phase 2b: Precompute routing map for mm_id dispatch ---- */
    /* expert_assignments[eid * n_tokens + a] = packed (token << 8 | slot) */
    {
        memset(s->expert_counts, 0, n_expert * sizeof(int));
        for (int t = 0; t < n_tokens; t++) {
            int n_tok_experts = n_experts_per_token[t];
            for (int sl = 0; sl < n_tok_experts; sl++) {
                int eid = all_ids[t][sl];
                int idx = s->expert_counts[eid];
                s->expert_assignments[eid * n_tokens + idx] = (t << 8) | sl;
                s->expert_counts[eid]++;
            }
        }
    }

    /* ---- Phase 3: mm_id gate+up projections ---- */
    {
        tensor_set_n_threads(saved_threads);
        gguf_type_t type = lw->type_ffn_gate_exps;
        matmul_mm_id_gate_up(mm_gate_out, mm_up_out,
            qx_all, qx_d_off, q8_buf_per_token,
            lw->ffn_gate_exps, lw->ffn_up_exps,
            s->expert_assignments, s->expert_counts,
            n_tokens, n_used, dim, n_ff, n_expert, type);
        tensor_set_n_threads(1);
    }

    /* ---- Phase 4: SwiGLU per (token, slot) ---- */
    {
        for (int t = 0; t < n_tokens; t++) {
            for (int sl = 0; sl < n_used; sl++) {
                float *g = mm_gate_out + (size_t)t * n_used * n_ff + (size_t)sl * n_ff;
                float *u = mm_up_out + (size_t)t * n_used * n_ff + (size_t)sl * n_ff;
                silu(g, n_ff);
                elemwise_mul(g, g, u, n_ff);
            }
        }
    }
    /* ---- Phase 5: mm_id down projections ---- */
    {
        tensor_set_n_threads(saved_threads);
        gguf_type_t type = lw->type_ffn_down_exps;
        matmul_mm_id_down(mm_down_out, mm_gate_out,
            lw->ffn_down_exps,
            s->expert_assignments, s->expert_counts,
            n_tokens, n_used, dim, n_ff, n_expert, type,
            s->mm_scratch_qx, s->mm_scratch_qx_d,
            s->mm_down_qx_all, s->mm_down_qx_d_all, down_q8_per_token);
        tensor_set_n_threads(1);
    }
    /* ---- Phase 6: Weighted accumulation in Top-K order ---- */
    {
        for (int t = 0; t < n_tokens; t++) {
            float *out = residual_batch + t * dim;
            memset(out, 0, dim * sizeof(float));

            for (int sl = 0; sl < n_used; sl++) {
                float w_i = all_weights[t][sl];
                float *expert_out = mm_down_out + (size_t)t * n_used * dim + (size_t)sl * dim;

#ifdef PICOLM_AVX512
                {
                    __m512 bw = _mm512_set1_ps(w_i);
                    int di = 0;
                    for (; di + 23 < dim; di += 16) {
                        __m512 v0 = _mm512_loadu_ps(out + di);
                        __m512 v1 = _mm512_loadu_ps(expert_out + di);
                        __m512 v2 = _mm512_loadu_ps(out + di + 8);
                        __m512 v3 = _mm512_loadu_ps(expert_out + di + 8);
                        _mm512_storeu_ps(out + di,
                            _mm512_add_ps(v0, _mm512_mul_ps(bw, v1)));
                        _mm512_storeu_ps(out + di + 8,
                            _mm512_add_ps(v2, _mm512_mul_ps(bw, v3)));
                    }
                    for (; di < dim; di++)
                        out[di] += w_i * expert_out[di];
                }
#elif defined(PICOLM_AVX2)
                {
                    __m256 bw = _mm256_set1_ps(w_i);
                    int di = 0;
                    for (; di + 20 < dim; di += 16) {
                        __m256 v0 = _mm256_loadu_ps(out + di);
                        __m256 v1 = _mm256_loadu_ps(expert_out + di);
                        __m256 v2 = _mm256_loadu_ps(out + di + 4);
                        __m256 v3 = _mm256_loadu_ps(expert_out + di + 4);
                        __m256 v4 = _mm256_loadu_ps(out + di + 8);
                        __m256 v5 = _mm256_loadu_ps(expert_out + di + 8);
                        __m256 v6 = _mm256_loadu_ps(out + di + 12);
                        __m256 v7 = _mm256_loadu_ps(expert_out + di + 12);
                        _mm256_storeu_ps(out + di,
                            _mm256_add_ps(v0, _mm256_mul_ps(bw, v1)));
                        _mm256_storeu_ps(out + di + 4,
                            _mm256_add_ps(v2, _mm256_mul_ps(bw, v3)));
                        _mm256_storeu_ps(out + di + 8,
                            _mm256_add_ps(v4, _mm256_mul_ps(bw, v5)));
                        _mm256_storeu_ps(out + di + 12,
                            _mm256_add_ps(v6, _mm256_mul_ps(bw, v7)));
                    }
                    for (; di < dim; di++)
                        out[di] += w_i * expert_out[di];
                }
#elif defined(PICOLM_NEON)
                {
                    float32x4_t bw = vdupq_n_f32(w_i);
                    int di = 0;
                    for (; di + 3 < dim; di += 4) {
                        float32x4_t v0 = vld1q_f32(out + di);
                        float32x4_t v1 = vld1q_f32(expert_out + di);
                        vst1q_f32(out + di, vaddq_f32(v0, vmulq_f32(bw, v1)));
                    }
                    for (; di < dim; di++)
                        out[di] += w_i * expert_out[di];
                }
#else
                for (int d = 0; d < dim; d++)
                    out[d] += w_i * expert_out[d];
#endif
            }
        }
    }

    /* ---- Phase 7: Shared expert (batched via matmul_q8_batch) ---- */
    {
        int sh_ff = c->n_ff_shexp;
        float *sh_gate = s->sh_gate;
        float *sh_up = s->sh_up;

        matmul_q8_batch(sh_gate, qx_all, qx_d_off, q8_buf_per_token,
                        lw->ffn_gate_shexp, dim, sh_ff, n_tokens, lw->type_ffn_gate_shexp);
        matmul_q8_batch(sh_up, qx_all, qx_d_off, q8_buf_per_token,
                        lw->ffn_up_shexp, dim, sh_ff, n_tokens, lw->type_ffn_up_shexp);

        for (int t = 0; t < n_tokens; t++) {
            float *g = sh_gate + t * sh_ff;
            float *u = sh_up + t * sh_ff;
            float *out = residual_batch + t * dim;

            silu(g, sh_ff);
            elemwise_mul(g, g, u, sh_ff);

            {
                size_t dnb = sh_ff / 32;
                quantize_row_q8_0(g, s->shared_down_qx, sh_ff);
                for (size_t bi = 0; bi < dnb; bi++) {
                    s->shared_down_qx_d[bi] = fp16_to_fp32(s->shared_down_qx[bi].d);
                }
                matmul_q8(s->xb2, s->shared_down_qx, s->shared_down_qx_d, lw->ffn_down_shexp, sh_ff, dim, lw->type_ffn_down_shexp);
                }

            {
                float gate_val;
                matmul(&gate_val, x_batch + t * dim, lw->ffn_gate_inp_shexp, dim, 1, lw->type_ffn_gate_inp_shexp);
                gate_val = 1.0f / (1.0f + expf(-gate_val));
                for (int d = 0; d < dim; d++) s->xb2[d] *= gate_val;
            }

            for (int d = 0; d < dim; d++) out[d] += s->xb2[d];
        }
    }

    tensor_set_n_threads(saved_threads);
}
