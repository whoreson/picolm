#include "model.h"
#include "tensor.h"
#include "quant.h"
#include "model_internal.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef PICOLM_GPU
#ifdef PICOLM_CUDA
#include <cuda_profiler_api.h>
#endif
#endif
#ifdef PICOLM_DOS
#include <alloca.h>
#endif
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef PICOLM_GPU
#include "backend_gpu.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef PICOLM_NEON
#include <arm_neon.h>
#endif

#ifdef PICOLM_VIZ
#include "viz.h"
#endif

/* SSM verification debug: guarded by compile-time PICOLM_SSM_VERIFY define.
 * When undefined, the macro expands to 0, eliminating all debug code at compile time.
 * When defined, falls back to _SSM_DBG for runtime toggle. */
#ifdef PICOLM_SSM_VERIFY
#define _SSM_DBG 1
#else
#define _SSM_DBG (0)
#endif


/* ================================================================
 * SSM forward pass helpers (Qwen3.5)
 * ================================================================ */

#ifdef DEBUG_SSM
static void dbg_vec(const char *tag, float *v, int n, int max_print) {
    int p = n < max_print ? n : max_print;
    for (int i = 0; i < p; i++) fprintf(stderr, "%.6f ", v[i]);
    fprintf(stderr, "\n");
}
#endif

/* Qwen3.5 GGUF v-head reordering (all v-head-indexed tensors).
 *
 * The GGUF converter reorders via _reorder_v_heads: a simple transpose.
 *   Sequential: [k0v0, k0v1, k0v2, ..., k1v0, ...]
 *   GGUF:       [v0*k, v1*k, ..., vn_vpk-1*k] for each k group
 *   GGUF_index = v * n_k + k  where  k = h / n_vpk,  v = h % n_vpk
 *
 * This applies uniformly to ALL v-head-indexed tensors:
 *   attn_gate_ssm, attn_qkv V portion, ssm_conv1d V channels,
 *   ssm_alpha, ssm_beta, ssm_out columns, dt_bias, ssm_a.
 */

/* ---- SSM per-head task (threaded state recurrence) ----
 * Each of the n_v_heads has its own independent [d_state x d_state]
 * state block.  tensor_parallel_for dispatches one head per task.
 *
 * Layout note: ssm_state is stored [n_v_heads][d_state][d_state]
 * (row = contracted dim, col = output dim) with the row index being
 * the dimension summed over in sk/output.  This means sk and output
 * scan column-wise against row-major storage (stride d_state per
 * element).  The expert03 insight was that flipping the loop order
 * to row-outer makes all four steps (decay/sk/update/output)
 * contiguous.  However, the state matrix is shared with the
 * batched prefill path which assumes the original layout, so we
 * keep the existing layout here and instead compute the recurrence
 * with row-major access: d1-outer loop, d2-inner, which is still
 * reasonably cache-friendly for d_state=128. */
typedef struct {
    float *state;               /* [n_v_heads][d_state][d_state] */
    const float *q_conv, *k_conv; /* [n_k_heads][d_state], head-major */
    const float *v_conv;         /* [n_v_heads][head_v_dim], head-major */
    const float *gate_exp, *beta; /* [n_v_heads] */
    float *ssm_output;           /* [d_state][n_v_heads], dim-major */
    int d_state, head_v_dim, n_v_heads, repeat;
} ssm_head_ctx_t;

static void ssm_head_task(int h, void *ctxp) {
    ssm_head_ctx_t *ctx = (ssm_head_ctx_t *)ctxp;
    int d_state = ctx->d_state;
    int n_v_heads = ctx->n_v_heads;
    assert(d_state <= 256 && "ssm_head_task: stack scratch too large");
    assert(ctx->head_v_dim == d_state &&
           "ssm_head_task assumes head_v_dim == d_state");

    float *st = ctx->state + (size_t)h * d_state * d_state;
    float ge = ctx->gate_exp[h];
    int kh = h / ctx->repeat;
    const float *qh = ctx->q_conv + (size_t)kh * d_state;
    const float *khv = ctx->k_conv + (size_t)kh * d_state;
    const float *vh = ctx->v_conv + (size_t)h * ctx->head_v_dim;
    float bh = ctx->beta[h];

#ifdef PICOLM_AVX512
    /* ---- AVX-512 vectorized recurrence ----
     * Process 4 rows of the d_state x d_state state matrix simultaneously.
     * Each __m512 holds 16 floats. For d_state=128: 8 vector lanes per row,
     * 4 rows processed at once = 32 vector registers for the matrix.
     *
     * Layout: st[h] is [d_state][d_state] row-major.
     * We process 4 consecutive rows (r, r+1, r+2, r+3) together. */
    {
        int nr = d_state / 4; /* number of 4-row groups */
        __m512 ge_v = _mm512_set1_ps(ge);
        int d16 = d_state / 16; /* number of 16-float vector lanes per row */

        for (int g = 0; g < nr; g++) {
            int base = g * 4; /* row base in this group */
            /* Load 4 rows x d16 vectors = 4*d128 floats for this group */

            /* Phase 1: Decay + sk computation */
            /* sk[4 rows][16 lanes] = sum over columns: st[4r][c] * k[c] */
            __m512 sk0 = _mm512_setzero_ps(), sk1 = _mm512_setzero_ps();
            __m512 sk2 = _mm512_setzero_ps(), sk3 = _mm512_setzero_ps();

            for (int v = 0; v < d16; v++) {
                int col_base = v * 16;
                /* Load k[col_base..col_base+15] broadcast to decay/multiply */
                __m512 kv = _mm512_loadu_ps(khv + col_base);

                /* Row 0: decay + accumulate sk */
                __m512 r0 = _mm512_loadu_ps(st + base * d_state + col_base);
                r0 = _mm512_mul_ps(r0, ge_v);
                _mm512_storeu_ps(st + base * d_state + col_base, r0);
                sk0 = _mm512_fmadd_ps(r0, kv, sk0);

                /* Row 1 */
                __m512 r1 = _mm512_loadu_ps(st + (base+1) * d_state + col_base);
                r1 = _mm512_mul_ps(r1, ge_v);
                _mm512_storeu_ps(st + (base+1) * d_state + col_base, r1);
                sk1 = _mm512_fmadd_ps(r1, kv, sk1);

                /* Row 2 */
                __m512 r2 = _mm512_loadu_ps(st + (base+2) * d_state + col_base);
                r2 = _mm512_mul_ps(r2, ge_v);
                _mm512_storeu_ps(st + (base+2) * d_state + col_base, r2);
                sk2 = _mm512_fmadd_ps(r2, kv, sk2);

                /* Row 3 */
                __m512 r3 = _mm512_loadu_ps(st + (base+3) * d_state + col_base);
                r3 = _mm512_mul_ps(r3, ge_v);
                _mm512_storeu_ps(st + (base+3) * d_state + col_base, r3);
                sk3 = _mm512_fmadd_ps(r3, kv, sk3);
            }

            /* Horizontal reduce sk (sum across 16 lanes per __m512) */
            float sk0s = _mm512_reduce_add_ps(sk0);
            float sk1s = _mm512_reduce_add_ps(sk1);
            float sk2s = _mm512_reduce_add_ps(sk2);
            float sk3s = _mm512_reduce_add_ps(sk3);

            /* Phase 2: Compute delta = (v - sk) * beta */
            float d0 = (vh[base + 0] - sk0s) * bh;
            float d1 = (vh[base + 1] - sk1s) * bh;
            float d2 = (vh[base + 2] - sk2s) * bh;
            float d3 = (vh[base + 3] - sk3s) * bh;

            __m512 d0v = _mm512_set1_ps(d0);
            __m512 d1v = _mm512_set1_ps(d1);
            __m512 d2v = _mm512_set1_ps(d2);
            __m512 d3v = _mm512_set1_ps(d3);

            /* Phase 3: State update + output computation
             * state[r][c] += k[c] * d[r]  (rank-1 update)
             * output[r] = sum_c state[r][c] * q[c] */
            __m512 out0 = _mm512_setzero_ps(), out1 = _mm512_setzero_ps();
            __m512 out2 = _mm512_setzero_ps(), out3 = _mm512_setzero_ps();

            for (int v = 0; v < d16; v++) {
                int col_base = v * 16;
                __m512 kv = _mm512_loadu_ps(khv + col_base);
                __m512 qv = _mm512_loadu_ps(qh + col_base);

                /* Row 0: update state + accumulate output */
                __m512 r0 = _mm512_loadu_ps(st + base * d_state + col_base);
                r0 = _mm512_fmadd_ps(kv, d0v, r0);
                _mm512_storeu_ps(st + base * d_state + col_base, r0);
                out0 = _mm512_fmadd_ps(r0, qv, out0);

                /* Row 1 */
                __m512 r1 = _mm512_loadu_ps(st + (base+1) * d_state + col_base);
                r1 = _mm512_fmadd_ps(kv, d1v, r1);
                _mm512_storeu_ps(st + (base+1) * d_state + col_base, r1);
                out1 = _mm512_fmadd_ps(r1, qv, out1);

                /* Row 2 */
                __m512 r2 = _mm512_loadu_ps(st + (base+2) * d_state + col_base);
                r2 = _mm512_fmadd_ps(kv, d2v, r2);
                _mm512_storeu_ps(st + (base+2) * d_state + col_base, r2);
                out2 = _mm512_fmadd_ps(r2, qv, out2);

                /* Row 3 */
                __m512 r3 = _mm512_loadu_ps(st + (base+3) * d_state + col_base);
                r3 = _mm512_fmadd_ps(kv, d3v, r3);
                _mm512_storeu_ps(st + (base+3) * d_state + col_base, r3);
                out3 = _mm512_fmadd_ps(r3, qv, out3);
            }

            /* Horizontal reduce output */
            ctx->ssm_output[(size_t)base * n_v_heads + h] = _mm512_reduce_add_ps(out0);
            ctx->ssm_output[(size_t)(base+1) * n_v_heads + h] = _mm512_reduce_add_ps(out1);
            ctx->ssm_output[(size_t)(base+2) * n_v_heads + h] = _mm512_reduce_add_ps(out2);
            ctx->ssm_output[(size_t)(base+3) * n_v_heads + h] = _mm512_reduce_add_ps(out3);
        }
    }
#elif defined(PICOLM_AVX2) || defined(PICOLM_AVX)
    /* ---- AVX2/AVX vectorized recurrence ----
     * Process 4 rows of the d_state x d_state state matrix simultaneously.
     * Each __m256 holds 8 floats. For d_state=128: 16 vector lanes per row,
     * 4 rows processed at once = 64 vector registers for the matrix.
     *
     * Layout: st[h] is [d_state][d_state] row-major.
     * We process 4 consecutive rows (r, r+1, r+2, r+3) together.
     *
     * FMA256 macro: uses _mm256_fmadd_ps when FMA3 is available,
     * falls back to mul+add for AVX-only CPUs without FMA (e.g. Sandy Bridge). */
#ifdef PICOLM_FMA
    #define FMA256(a,b,c) _mm256_fmadd_ps((a),(b),(c))
#else
    #define FMA256(a,b,c) _mm256_add_ps(_mm256_mul_ps((a),(b)),(c))
#endif

    {
        int nr = d_state / 4; /* number of 4-row groups */
        __m256 ge_v = _mm256_set1_ps(ge);
        int d8 = d_state / 8; /* number of 8-float vector lanes per row */

        for (int g = 0; g < nr; g++) {
            int base = g * 4; /* row base in this group */

            /* Phase 1: Decay + sk computation */
            __m256 sk0 = _mm256_setzero_ps(), sk1 = _mm256_setzero_ps();
            __m256 sk2 = _mm256_setzero_ps(), sk3 = _mm256_setzero_ps();

            for (int v = 0; v < d8; v++) {
                int col_base = v * 8;
                __m256 kv = _mm256_loadu_ps(khv + col_base);

                /* Row 0 */
                __m256 r0 = _mm256_loadu_ps(st + base * d_state + col_base);
                r0 = _mm256_mul_ps(r0, ge_v);
                _mm256_storeu_ps(st + base * d_state + col_base, r0);
                sk0 = FMA256(r0, kv, sk0);

                /* Row 1 */
                __m256 r1 = _mm256_loadu_ps(st + (base+1) * d_state + col_base);
                r1 = _mm256_mul_ps(r1, ge_v);
                _mm256_storeu_ps(st + (base+1) * d_state + col_base, r1);
                sk1 = FMA256(r1, kv, sk1);

                /* Row 2 */
                __m256 r2 = _mm256_loadu_ps(st + (base+2) * d_state + col_base);
                r2 = _mm256_mul_ps(r2, ge_v);
                _mm256_storeu_ps(st + (base+2) * d_state + col_base, r2);
                sk2 = FMA256(r2, kv, sk2);

                /* Row 3 */
                __m256 r3 = _mm256_loadu_ps(st + (base+3) * d_state + col_base);
                r3 = _mm256_mul_ps(r3, ge_v);
                _mm256_storeu_ps(st + (base+3) * d_state + col_base, r3);
                sk3 = FMA256(r3, kv, sk3);
            }

            float sk0s = hreduce256_ps(sk0);
            float sk1s = hreduce256_ps(sk1);
            float sk2s = hreduce256_ps(sk2);
            float sk3s = hreduce256_ps(sk3);

            /* Phase 2: Compute delta = (v - sk) * beta */
            float d0 = (vh[base + 0] - sk0s) * bh;
            float d1 = (vh[base + 1] - sk1s) * bh;
            float d2 = (vh[base + 2] - sk2s) * bh;
            float d3 = (vh[base + 3] - sk3s) * bh;

            __m256 d0v = _mm256_set1_ps(d0);
            __m256 d1v = _mm256_set1_ps(d1);
            __m256 d2v = _mm256_set1_ps(d2);
            __m256 d3v = _mm256_set1_ps(d3);

            /* Phase 3: State update + output computation */
            __m256 out0 = _mm256_setzero_ps(), out1 = _mm256_setzero_ps();
            __m256 out2 = _mm256_setzero_ps(), out3 = _mm256_setzero_ps();

            for (int v = 0; v < d8; v++) {
                int col_base = v * 8;
                __m256 kv = _mm256_loadu_ps(khv + col_base);
                __m256 qv = _mm256_loadu_ps(qh + col_base);

                /* Row 0 */
                __m256 r0 = _mm256_loadu_ps(st + base * d_state + col_base);
                r0 = FMA256(kv, d0v, r0);
                _mm256_storeu_ps(st + base * d_state + col_base, r0);
                out0 = FMA256(r0, qv, out0);

                /* Row 1 */
                __m256 r1 = _mm256_loadu_ps(st + (base+1) * d_state + col_base);
                r1 = FMA256(kv, d1v, r1);
                _mm256_storeu_ps(st + (base+1) * d_state + col_base, r1);
                out1 = FMA256(r1, qv, out1);

                /* Row 2 */
                __m256 r2 = _mm256_loadu_ps(st + (base+2) * d_state + col_base);
                r2 = FMA256(kv, d2v, r2);
                _mm256_storeu_ps(st + (base+2) * d_state + col_base, r2);
                out2 = FMA256(r2, qv, out2);

                /* Row 3 */
                __m256 r3 = _mm256_loadu_ps(st + (base+3) * d_state + col_base);
                r3 = FMA256(kv, d3v, r3);
                _mm256_storeu_ps(st + (base+3) * d_state + col_base, r3);
                out3 = FMA256(r3, qv, out3);
            }

            /* Horizontal reduce output */
            ctx->ssm_output[(size_t)base * n_v_heads + h] = hreduce256_ps(out0);
            ctx->ssm_output[(size_t)(base+1) * n_v_heads + h] = hreduce256_ps(out1);
            ctx->ssm_output[(size_t)(base+2) * n_v_heads + h] = hreduce256_ps(out2);
            ctx->ssm_output[(size_t)(base+3) * n_v_heads + h] = hreduce256_ps(out3);
        }
    }
#elif defined(PICOLM_NEON)
    /* ---- NEON vectorized recurrence ----
     * Process 4 rows of the d_state x d_state state matrix simultaneously.
     * Each float32x4_t holds 4 floats. For d_state=128: 32 vector lanes per row,
     * 4 rows processed at once = 128 vector registers for the matrix.
     *
     * Layout: st[h] is [d_state][d_state] row-major.
     * We process 4 consecutive rows (r, r+1, r+2, r+3) together.
     *
     * NEON always has FP32 FMA via vmlaq_f32. */
    {
        int nr = d_state / 4; /* number of 4-row groups */
        float32x4_t ge_v = vdupq_n_f32(ge);
        int d4 = d_state / 4; /* number of 4-float vector lanes per row */

        for (int g = 0; g < nr; g++) {
            int base = g * 4; /* row base in this group */

            /* Phase 1: Decay + sk computation */
            float32x4_t sk0 = vdupq_n_f32(0);
            float32x4_t sk1 = vdupq_n_f32(0);
            float32x4_t sk2 = vdupq_n_f32(0);
            float32x4_t sk3 = vdupq_n_f32(0);

            for (int v = 0; v < d4; v++) {
                int col_base = v * 4;
                float32x4_t kv = vld1q_f32(khv + col_base);

                /* Row 0 */
                float32x4_t r0 = vld1q_f32(st + base * d_state + col_base);
                r0 = vmulq_f32(r0, ge_v);
                vst1q_f32(st + base * d_state + col_base, r0);
                sk0 = vmlaq_f32(sk0, r0, kv);

                /* Row 1 */
                float32x4_t r1 = vld1q_f32(st + (base+1) * d_state + col_base);
                r1 = vmulq_f32(r1, ge_v);
                vst1q_f32(st + (base+1) * d_state + col_base, r1);
                sk1 = vmlaq_f32(sk1, r1, kv);

                /* Row 2 */
                float32x4_t r2 = vld1q_f32(st + (base+2) * d_state + col_base);
                r2 = vmulq_f32(r2, ge_v);
                vst1q_f32(st + (base+2) * d_state + col_base, r2);
                sk2 = vmlaq_f32(sk2, r2, kv);

                /* Row 3 */
                float32x4_t r3 = vld1q_f32(st + (base+3) * d_state + col_base);
                r3 = vmulq_f32(r3, ge_v);
                vst1q_f32(st + (base+3) * d_state + col_base, r3);
                sk3 = vmlaq_f32(sk3, r3, kv);
            }

            float sk0s = vaddvq_f32_compat(sk0);
            float sk1s = vaddvq_f32_compat(sk1);
            float sk2s = vaddvq_f32_compat(sk2);
            float sk3s = vaddvq_f32_compat(sk3);

            /* Phase 2: Compute delta = (v - sk) * beta */
            float d0 = (vh[base + 0] - sk0s) * bh;
            float d1 = (vh[base + 1] - sk1s) * bh;
            float d2 = (vh[base + 2] - sk2s) * bh;
            float d3 = (vh[base + 3] - sk3s) * bh;

            float32x4_t d0v = vdupq_n_f32(d0);
            float32x4_t d1v = vdupq_n_f32(d1);
            float32x4_t d2v = vdupq_n_f32(d2);
            float32x4_t d3v = vdupq_n_f32(d3);

            /* Phase 3: State update + output computation */
            float32x4_t out0 = vdupq_n_f32(0);
            float32x4_t out1 = vdupq_n_f32(0);
            float32x4_t out2 = vdupq_n_f32(0);
            float32x4_t out3 = vdupq_n_f32(0);

            for (int v = 0; v < d4; v++) {
                int col_base = v * 4;
                float32x4_t kv = vld1q_f32(khv + col_base);
                float32x4_t qv = vld1q_f32(qh + col_base);

                /* Row 0 */
                float32x4_t r0 = vld1q_f32(st + base * d_state + col_base);
                r0 = vmlaq_f32(r0, kv, d0v);
                vst1q_f32(st + base * d_state + col_base, r0);
                out0 = vmlaq_f32(out0, r0, qv);

                /* Row 1 */
                float32x4_t r1 = vld1q_f32(st + (base+1) * d_state + col_base);
                r1 = vmlaq_f32(r1, kv, d1v);
                vst1q_f32(st + (base+1) * d_state + col_base, r1);
                out1 = vmlaq_f32(out1, r1, qv);

                /* Row 2 */
                float32x4_t r2 = vld1q_f32(st + (base+2) * d_state + col_base);
                r2 = vmlaq_f32(r2, kv, d2v);
                vst1q_f32(st + (base+2) * d_state + col_base, r2);
                out2 = vmlaq_f32(out2, r2, qv);

                /* Row 3 */
                float32x4_t r3 = vld1q_f32(st + (base+3) * d_state + col_base);
                r3 = vmlaq_f32(r3, kv, d3v);
                vst1q_f32(st + (base+3) * d_state + col_base, r3);
                out3 = vmlaq_f32(out3, r3, qv);
            }

            /* Horizontal reduce output */
            ctx->ssm_output[(size_t)base * n_v_heads + h] = vaddvq_f32_compat(out0);
            ctx->ssm_output[(size_t)(base+1) * n_v_heads + h] = vaddvq_f32_compat(out1);
            ctx->ssm_output[(size_t)(base+2) * n_v_heads + h] = vaddvq_f32_compat(out2);
            ctx->ssm_output[(size_t)(base+3) * n_v_heads + h] = vaddvq_f32_compat(out3);
        }
    }
#else
    /* ---- Scalar fallback ---- */
    float sk_local[256];
    float d_local[256];

    /* Decay: elementwise */
    for (int i = 0; i < d_state * d_state; i++) st[i] *= ge;

    /* sk[row] = sum_col state[row][col] * k[col] -- row-major contiguous */
    for (int row = 0; row < d_state; row++) {
        const float *st_row = st + (size_t)row * d_state;
        float sum = 0.0f;
        for (int col = 0; col < d_state; col++) sum += st_row[col] * khv[col];
        sk_local[row] = sum;
    }
    for (int row = 0; row < d_state; row++)
        d_local[row] = (vh[row] - sk_local[row]) * bh;

    /* state[row][col] += k[col] * d[row] -- row-major contiguous */
    for (int row = 0; row < d_state; row++) {
        float dv = d_local[row];
        float *st_row = st + (size_t)row * d_state;
        for (int col = 0; col < d_state; col++) st_row[col] += khv[col] * dv;
    }

    /* output[row] = sum_col state[row][col] * q[col] -- row-major contiguous */
    for (int row = 0; row < d_state; row++) {
        const float *st_row = st + (size_t)row * d_state;
        float sum = 0.0f;
        for (int col = 0; col < d_state; col++) sum += st_row[col] * qh[col];
        ctx->ssm_output[(size_t)row * n_v_heads + h] = sum;
    }
#endif
}

/* ================================================================
 * AVX-512 micro-kernels for chunked DeltaNet recurrence.
 *
 * These replace the scalar C loops in ssm_chunk_head_task with
 * AVX-512 vectorized GEMM kernels. Each kernel is single-threaded,
 * fixed-shape (d=128, cs<=64), and operates entirely in registers/L1.
 *
 * d=128 = 8 x __m512 (each holds 16 floats).
 * cs=64 max = 4 x __m512.
 *
 * The parallelism comes from the outer tensor_parallel_for across
 * (head, chunk) -- each kernel is called once per v-head per chunk.
 * ================================================================ */

#ifdef PICOLM_AVX512
#include <immintrin.h>

/* Kernel 1: V_eff -- compute S_init @ K^T -> sk[cs][d]
 *
 * sk[i][r] = sum_{c=0}^{d-1} state[r][c] * k[i][c]
 *
 * This is a GEMM: sk = state @ K^T   [d x d] @ [d x cs] -> [d x cs]
 * Transposed view: sk[cs][d] where sk[i] is the i-th row.
 */
static void ssm_kernel_veff(
        float *sk,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *k,   /* [cs][d] */
        int d, int cs)
{
    int d16 = d / 16;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *ski = sk + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int v = 0; v < d16; v++) {
                int col = v * 16;
                __m512 kv = _mm512_loadu_ps(ki + col);
                __m512 s0 = _mm512_loadu_ps(state + base * d + col);
                __m512 s1 = _mm512_loadu_ps(state + (base+1) * d + col);
                __m512 s2 = _mm512_loadu_ps(state + (base+2) * d + col);
                __m512 s3 = _mm512_loadu_ps(state + (base+3) * d + col);
                acc0 = _mm512_fmadd_ps(s0, kv, acc0);
                acc1 = _mm512_fmadd_ps(s1, kv, acc1);
                acc2 = _mm512_fmadd_ps(s2, kv, acc2);
                acc3 = _mm512_fmadd_ps(s3, kv, acc3);
            }

            ski[base] = _mm512_reduce_add_ps(acc0);
            ski[base+1] = _mm512_reduce_add_ps(acc1);
            ski[base+2] = _mm512_reduce_add_ps(acc2);
            ski[base+3] = _mm512_reduce_add_ps(acc3);
        }
    }
}

/* Kernel 2: Interaction matrix -- K @ K^T -> M[cs][cs]
 *
 * M[i][j] = (k[i] . k[j]) * decay[i][j]
 *
 * Only lower triangle (j <= i) is computed; upper triangle is zeroed.
 * The decay mask is applied elementwise.
 */
static void ssm_kernel_interaction(
        float *M,          /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *decay, /* [cs][cs] decay mask */
        int d, int cs)
{
    int d16 = d / 16;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *Mi = M + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            /* Dot product ki . kj using AVX-512 */
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            /* Unroll 4 vectors at a time = 64 floats per iteration */
            int v;
            for (v = 0; v + 3 < d16; v += 4) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(ki + v * 16),
                                        _mm512_loadu_ps(kj + v * 16), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(ki + (v+1) * 16),
                                        _mm512_loadu_ps(kj + (v+1) * 16), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(ki + (v+2) * 16),
                                        _mm512_loadu_ps(kj + (v+2) * 16), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(ki + (v+3) * 16),
                                        _mm512_loadu_ps(kj + (v+3) * 16), acc3);
            }

            /* Handle remaining lanes */
            for (; v < d16; v++) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(ki + v * 16),
                                        _mm512_loadu_ps(kj + v * 16), acc0);
            }

            /* Horizontal reduce all accumulators */
            float dot = _mm512_reduce_add_ps(acc0) + _mm512_reduce_add_ps(acc1)
                      + _mm512_reduce_add_ps(acc2) + _mm512_reduce_add_ps(acc3);

            Mi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            Mi[j] = 0.0f;
        }
    }
}

/* Kernel 3: Output cross-products -- K @ Q^T -> kq[cs][cs]
 *
 * kq[i][j] = (k[j] . q[i]) * decay[i][j]
 *
 * Note: k is indexed by j, q by i. This is effectively K^T @ Q but
 * with indices swapped compared to a standard GEMM.
 */
static void ssm_kernel_output_cross(
        float *kq,         /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *q,    /* [cs][d] */
        const float *decay, /* [cs][cs] */
        int d, int cs)
{
    int d16 = d / 16;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *kqi = kq + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            /* Dot product kj . qi using AVX-512 */
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            int v;
            for (v = 0; v + 3 < d16; v += 4) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(kj + v * 16),
                                        _mm512_loadu_ps(qi + v * 16), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(kj + (v+1) * 16),
                                        _mm512_loadu_ps(qi + (v+1) * 16), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(kj + (v+2) * 16),
                                        _mm512_loadu_ps(qi + (v+2) * 16), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(kj + (v+3) * 16),
                                        _mm512_loadu_ps(qi + (v+3) * 16), acc3);
            }

            for (; v < d16; v++) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(kj + v * 16),
                                        _mm512_loadu_ps(qi + v * 16), acc0);
            }

            float dot = _mm512_reduce_add_ps(acc0) + _mm512_reduce_add_ps(acc1)
                      + _mm512_reduce_add_ps(acc2) + _mm512_reduce_add_ps(acc3);

            kqi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            kqi[j] = 0.0f;
        }
    }
}

/* Kernel 4: State update -- accumulate decay[j] * outer(V_hat[j], k[j]) -> S_update[d][d]
 *
 * S_update[r][c] = sum_{j=0}^{cs-1} decay_to_end[j] * v_hat[j][r] * k[j][c]
 *
 * This is a weighted sum of rank-1 outer products.
 * Process d x d output tiled in AVX-512 friendly blocks.
 */
static void ssm_kernel_state_update(
        float *state,           /* [d][d] in-place: state *= total_decay, then += update */
        const float *v_hat,     /* [cs][d] */
        const float *k,         /* [cs][d] */
        const float *decay_to_end, /* [cs] decay from each position to end of chunk */
        float total_decay,
        int d, int cs)
{
    int d16 = d / 16;

    /* First: decay the existing state by total_decay */
    __m512 td = _mm512_set1_ps(total_decay);
    for (int r = 0; r < d; r++) {
        float *sr = state + r * d;
        for (int v = 0; v < d16; v++) {
            __m512 sv = _mm512_loadu_ps(sr + v * 16);
            _mm512_storeu_ps(sr + v * 16, _mm512_mul_ps(sv, td));
        }
    }

    /* Second: accumulate weighted outer products.
     * For each j: decay_to_end[j] * outer(V_hat[j], k[j])
     * Process 4 rows at a time for better register utilization. */
    int nr = d / 4; /* number of 4-row groups */

    for (int j = 0; j < cs; j++) {
        const float *vj = v_hat + j * d;
        const float *kj = k + j * d;
        float dj = decay_to_end[j];

        if (dj == 0.0f) continue;

                for (int g = 0; g < nr; g++) {
            int base = g * 4;
            float vj0 = vj[base];
            float vj1 = vj[base + 1];
            float vj2 = vj[base + 2];
            float vj3 = vj[base + 3];

            __m512 s0 = _mm512_set1_ps(vj0 * dj);
            __m512 s1 = _mm512_set1_ps(vj1 * dj);
            __m512 s2 = _mm512_set1_ps(vj2 * dj);
            __m512 s3 = _mm512_set1_ps(vj3 * dj);

            for (int v = 0; v < d16; v++) {
                int col_base = v * 16;
                __m512 kv = _mm512_loadu_ps(kj + col_base);

                __m512 r0 = _mm512_loadu_ps(state + base * d + col_base);
                __m512 r1 = _mm512_loadu_ps(state + (base+1) * d + col_base);
                __m512 r2 = _mm512_loadu_ps(state + (base+2) * d + col_base);
                __m512 r3 = _mm512_loadu_ps(state + (base+3) * d + col_base);

                _mm512_storeu_ps(state + base * d + col_base, _mm512_fmadd_ps(kv, s0, r0));
                _mm512_storeu_ps(state + (base+1) * d + col_base, _mm512_fmadd_ps(kv, s1, r1));
                _mm512_storeu_ps(state + (base+2) * d + col_base, _mm512_fmadd_ps(kv, s2, r2));
                _mm512_storeu_ps(state + (base+3) * d + col_base, _mm512_fmadd_ps(kv, s3, r3));
            }
        }
    }
}

/* Kernel 1b: S_init @ q[i] for each of cs positions -> sq[cs][d]
 *
 * sq[i][r] = sum_{c=0}^{d-1} state[r][c] * q[i][c]
 *
 * Same structure as Kernel 1 but with q instead of k.
 */
static void ssm_kernel_sq(
        float *sq,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *q,   /* [cs][d] */
        int d, int cs)
{
    int d16 = d / 16;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *sqi = sq + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int v = 0; v < d16; v++) {
                int col = v * 16;
                __m512 qv = _mm512_loadu_ps(qi + col);
                __m512 s0 = _mm512_loadu_ps(state + base * d + col);
                __m512 s1 = _mm512_loadu_ps(state + (base+1) * d + col);
                __m512 s2 = _mm512_loadu_ps(state + (base+2) * d + col);
                __m512 s3 = _mm512_loadu_ps(state + (base+3) * d + col);
                acc0 = _mm512_fmadd_ps(s0, qv, acc0);
                acc1 = _mm512_fmadd_ps(s1, qv, acc1);
                acc2 = _mm512_fmadd_ps(s2, qv, acc2);
                acc3 = _mm512_fmadd_ps(s3, qv, acc3);
            }

            sqi[base] = _mm512_reduce_add_ps(acc0);
            sqi[base+1] = _mm512_reduce_add_ps(acc1);
            sqi[base+2] = _mm512_reduce_add_ps(acc2);
            sqi[base+3] = _mm512_reduce_add_ps(acc3);
        }
    }
}
#elif defined(PICOLM_AVX2) || defined(PICOLM_AVX)
#ifdef PICOLM_FMA
    #define FMA256(a,b,c) _mm256_fmadd_ps((a),(b),(c))
#else
    #define FMA256(a,b,c) _mm256_add_ps(_mm256_mul_ps((a),(b)),(c))
#endif

/* Kernel 1: V_eff -- compute S_init @ K^T -> sk[cs][d] */
static void ssm_kernel_veff(
        float *sk,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *k,   /* [cs][d] */
        int d, int cs)
{
    int d8 = d / 8;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *ski = sk + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int v = 0; v < d8; v++) {
                int col = v * 8;
                __m256 kv = _mm256_loadu_ps(ki + col);
                __m256 s0 = _mm256_loadu_ps(state + base * d + col);
                __m256 s1 = _mm256_loadu_ps(state + (base+1) * d + col);
                __m256 s2 = _mm256_loadu_ps(state + (base+2) * d + col);
                __m256 s3 = _mm256_loadu_ps(state + (base+3) * d + col);
                acc0 = FMA256(s0, kv, acc0);
                acc1 = FMA256(s1, kv, acc1);
                acc2 = FMA256(s2, kv, acc2);
                acc3 = FMA256(s3, kv, acc3);
            }

            ski[base] = hreduce256_ps(acc0);
            ski[base+1] = hreduce256_ps(acc1);
            ski[base+2] = hreduce256_ps(acc2);
            ski[base+3] = hreduce256_ps(acc3);
        }
    }
}

/* Kernel 2: Interaction matrix -- K @ K^T -> M[cs][cs] */
static void ssm_kernel_interaction(
        float *M,          /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *decay, /* [cs][cs] decay mask */
        int d, int cs)
{
    int d8 = d / 8;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *Mi = M + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            /* Dot product ki . kj using AVX2 */
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int v;
            for (v = 0; v + 3 < d8; v += 4) {
                acc0 = FMA256(_mm256_loadu_ps(ki + v * 8),
                                        _mm256_loadu_ps(kj + v * 8), acc0);
                acc1 = FMA256(_mm256_loadu_ps(ki + (v+1) * 8),
                                        _mm256_loadu_ps(kj + (v+1) * 8), acc1);
                acc2 = FMA256(_mm256_loadu_ps(ki + (v+2) * 8),
                                        _mm256_loadu_ps(kj + (v+2) * 8), acc2);
                acc3 = FMA256(_mm256_loadu_ps(ki + (v+3) * 8),
                                        _mm256_loadu_ps(kj + (v+3) * 8), acc3);
            }

            for (; v < d8; v++) {
                acc0 = FMA256(_mm256_loadu_ps(ki + v * 8),
                                        _mm256_loadu_ps(kj + v * 8), acc0);
            }

            float dot = hreduce256_ps(acc0) + hreduce256_ps(acc1)
                      + hreduce256_ps(acc2) + hreduce256_ps(acc3);

            Mi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            Mi[j] = 0.0f;
        }
    }
}

/* Kernel 3: Output cross-products -- K @ Q^T -> kq[cs][cs] */
static void ssm_kernel_output_cross(
        float *kq,         /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *q,    /* [cs][d] */
        const float *decay, /* [cs][cs] */
        int d, int cs)
{
    int d8 = d / 8;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *kqi = kq + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            /* Dot product kj . qi using AVX2 */
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int v;
            for (v = 0; v + 3 < d8; v += 4) {
                acc0 = FMA256(_mm256_loadu_ps(kj + v * 8),
                                        _mm256_loadu_ps(qi + v * 8), acc0);
                acc1 = FMA256(_mm256_loadu_ps(kj + (v+1) * 8),
                                        _mm256_loadu_ps(qi + (v+1) * 8), acc1);
                acc2 = FMA256(_mm256_loadu_ps(kj + (v+2) * 8),
                                        _mm256_loadu_ps(qi + (v+2) * 8), acc2);
                acc3 = FMA256(_mm256_loadu_ps(kj + (v+3) * 8),
                                        _mm256_loadu_ps(qi + (v+3) * 8), acc3);
            }

            for (; v < d8; v++) {
                acc0 = FMA256(_mm256_loadu_ps(kj + v * 8),
                                        _mm256_loadu_ps(qi + v * 8), acc0);
            }

            float dot = hreduce256_ps(acc0) + hreduce256_ps(acc1)
                      + hreduce256_ps(acc2) + hreduce256_ps(acc3);

            kqi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            kqi[j] = 0.0f;
        }
    }
}

/* Kernel 4: State update -- accumulate weighted outer products */
static void ssm_kernel_state_update(
        float *state,           /* [d][d] in-place */
        const float *v_hat,     /* [cs][d] */
        const float *k,         /* [cs][d] */
        const float *decay_to_end, /* [cs] */
        float total_decay,
        int d, int cs)
{
    int d8 = d / 8;

    /* First: decay the existing state by total_decay */
    __m256 td = _mm256_set1_ps(total_decay);
    for (int r = 0; r < d; r++) {
        float *sr = state + r * d;
        for (int v = 0; v < d8; v++) {
            __m256 sv = _mm256_loadu_ps(sr + v * 8);
            _mm256_storeu_ps(sr + v * 8, _mm256_mul_ps(sv, td));
        }
    }

    /* Second: accumulate weighted outer products.
     * Process 4 rows at a time for better register utilization. */
    int nr = d / 4;

    for (int j = 0; j < cs; j++) {
        const float *vj = v_hat + j * d;
        const float *kj = k + j * d;
        float dj = decay_to_end[j];

        if (dj == 0.0f) continue;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            float vj0 = vj[base];
            float vj1 = vj[base + 1];
            float vj2 = vj[base + 2];
            float vj3 = vj[base + 3];

            __m256 s0 = _mm256_set1_ps(vj0 * dj);
            __m256 s1 = _mm256_set1_ps(vj1 * dj);
            __m256 s2 = _mm256_set1_ps(vj2 * dj);
            __m256 s3 = _mm256_set1_ps(vj3 * dj);

            for (int v = 0; v < d8; v++) {
                int col_base = v * 8;
                __m256 kv = _mm256_loadu_ps(kj + col_base);

                __m256 r0 = _mm256_loadu_ps(state + base * d + col_base);
                __m256 r1 = _mm256_loadu_ps(state + (base+1) * d + col_base);
                __m256 r2 = _mm256_loadu_ps(state + (base+2) * d + col_base);
                __m256 r3 = _mm256_loadu_ps(state + (base+3) * d + col_base);

                _mm256_storeu_ps(state + base * d + col_base, FMA256(kv, s0, r0));
                _mm256_storeu_ps(state + (base+1) * d + col_base, FMA256(kv, s1, r1));
                _mm256_storeu_ps(state + (base+2) * d + col_base, FMA256(kv, s2, r2));
                _mm256_storeu_ps(state + (base+3) * d + col_base, FMA256(kv, s3, r3));
            }
        }
    }
}

/* Kernel 1b: S_init @ Q^T -> sq[cs][d] */
static void ssm_kernel_sq(
        float *sq,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *q,   /* [cs][d] */
        int d, int cs)
{
    int d8 = d / 8;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *sqi = sq + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int v = 0; v < d8; v++) {
                int col = v * 8;
                __m256 qv = _mm256_loadu_ps(qi + col);
                __m256 s0 = _mm256_loadu_ps(state + base * d + col);
                __m256 s1 = _mm256_loadu_ps(state + (base+1) * d + col);
                __m256 s2 = _mm256_loadu_ps(state + (base+2) * d + col);
                __m256 s3 = _mm256_loadu_ps(state + (base+3) * d + col);
                acc0 = FMA256(s0, qv, acc0);
                acc1 = FMA256(s1, qv, acc1);
                acc2 = FMA256(s2, qv, acc2);
                acc3 = FMA256(s3, qv, acc3);
            }

            sqi[base] = hreduce256_ps(acc0);
            sqi[base+1] = hreduce256_ps(acc1);
            sqi[base+2] = hreduce256_ps(acc2);
            sqi[base+3] = hreduce256_ps(acc3);
        }
    }
}
#undef FMA256
#endif /* PICOLM_AVX512 || PICOLM_AVX2 || PICOLM_AVX */

/* ================================================================
 * NEON micro-kernels for chunked DeltaNet recurrence.
 *
 * d=128 = 32 x float32x4_t (each holds 4 floats).
 * cs=64 max = 16 x float32x4_t.
 *
 * NEON always has FP32 FMA via vmlaq_f32.
 * ================================================================ */
#ifdef PICOLM_NEON

/* Kernel 1: V_eff -- compute S_init @ K^T -> sk[cs][d] */
static void ssm_kernel_veff(
        float *sk,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *k,   /* [cs][d] */
        int d, int cs)
{
    int d4 = d / 4;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *ski = sk + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0);
            float32x4_t acc3 = vdupq_n_f32(0);

            for (int v = 0; v < d4; v++) {
                int col = v * 4;
                float32x4_t kv = vld1q_f32(ki + col);
                float32x4_t s0 = vld1q_f32(state + base * d + col);
                float32x4_t s1 = vld1q_f32(state + (base+1) * d + col);
                float32x4_t s2 = vld1q_f32(state + (base+2) * d + col);
                float32x4_t s3 = vld1q_f32(state + (base+3) * d + col);
                acc0 = vmlaq_f32(acc0, s0, kv);
                acc1 = vmlaq_f32(acc1, s1, kv);
                acc2 = vmlaq_f32(acc2, s2, kv);
                acc3 = vmlaq_f32(acc3, s3, kv);
            }

            ski[base] = vaddvq_f32_compat(acc0);
            ski[base+1] = vaddvq_f32_compat(acc1);
            ski[base+2] = vaddvq_f32_compat(acc2);
            ski[base+3] = vaddvq_f32_compat(acc3);
        }
    }
}

/* Kernel 2: Interaction matrix -- K @ K^T -> M[cs][cs] */
static void ssm_kernel_interaction(
        float *M,          /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *decay, /* [cs][cs] decay mask */
        int d, int cs)
{
    int d4 = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *Mi = M + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            /* Dot product ki . kj using NEON - 4 accumulators */
            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0);
            float32x4_t acc3 = vdupq_n_f32(0);

            int v;
            for (v = 0; v + 3 < d4; v += 4) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(ki + v * 4), vld1q_f32(kj + v * 4));
                acc1 = vmlaq_f32(acc1, vld1q_f32(ki + (v+1) * 4), vld1q_f32(kj + (v+1) * 4));
                acc2 = vmlaq_f32(acc2, vld1q_f32(ki + (v+2) * 4), vld1q_f32(kj + (v+2) * 4));
                acc3 = vmlaq_f32(acc3, vld1q_f32(ki + (v+3) * 4), vld1q_f32(kj + (v+3) * 4));
            }

            for (; v < d4; v++) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(ki + v * 4), vld1q_f32(kj + v * 4));
            }

            /* Reduce 4 accumulators to one float */
            float32x4_t sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float dot = vaddvq_f32_compat(sum);

            Mi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            Mi[j] = 0.0f;
        }
    }
}

/* Kernel 3: Output cross-products -- K @ Q^T -> kq[cs][cs] */
static void ssm_kernel_output_cross(
        float *kq,         /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *q,    /* [cs][d] */
        const float *decay, /* [cs][cs] */
        int d, int cs)
{
    int d4 = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *kqi = kq + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0);
            float32x4_t acc3 = vdupq_n_f32(0);

            int v;
            for (v = 0; v + 3 < d4; v += 4) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(kj + v * 4), vld1q_f32(qi + v * 4));
                acc1 = vmlaq_f32(acc1, vld1q_f32(kj + (v+1) * 4), vld1q_f32(qi + (v+1) * 4));
                acc2 = vmlaq_f32(acc2, vld1q_f32(kj + (v+2) * 4), vld1q_f32(qi + (v+2) * 4));
                acc3 = vmlaq_f32(acc3, vld1q_f32(kj + (v+3) * 4), vld1q_f32(qi + (v+3) * 4));
            }

            for (; v < d4; v++) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(kj + v * 4), vld1q_f32(qi + v * 4));
            }

            float32x4_t sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float dot = vaddvq_f32_compat(sum);

            kqi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            kqi[j] = 0.0f;
        }
    }
}

/* Kernel 4: State update -- accumulate weighted outer products */
static void ssm_kernel_state_update(
        float *state,           /* [d][d] in-place */
        const float *v_hat,     /* [cs][d] */
        const float *k,         /* [cs][d] */
        const float *decay_to_end, /* [cs] */
        float total_decay,
        int d, int cs)
{
    int d4 = d / 4;

    /* First: decay the existing state by total_decay */
    float32x4_t td = vdupq_n_f32(total_decay);
    for (int r = 0; r < d; r++) {
        float *sr = state + r * d;
        for (int v = 0; v < d4; v++) {
            float32x4_t sv = vld1q_f32(sr + v * 4);
            vst1q_f32(sr + v * 4, vmulq_f32(sv, td));
        }
    }

    /* Second: accumulate weighted outer products.
     * Process 4 rows at a time for better register utilization. */
    int nr = d / 4;

    for (int j = 0; j < cs; j++) {
        const float *vj = v_hat + j * d;
        const float *kj = k + j * d;
        float dj = decay_to_end[j];

        if (dj == 0.0f) continue;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            float vj0 = vj[base];
            float vj1 = vj[base + 1];
            float vj2 = vj[base + 2];
            float vj3 = vj[base + 3];

            float32x4_t s0 = vdupq_n_f32(vj0 * dj);
            float32x4_t s1 = vdupq_n_f32(vj1 * dj);
            float32x4_t s2 = vdupq_n_f32(vj2 * dj);
            float32x4_t s3 = vdupq_n_f32(vj3 * dj);

            for (int v = 0; v < d4; v++) {
                int col_base = v * 4;
                float32x4_t kv = vld1q_f32(kj + col_base);

                float32x4_t r0 = vld1q_f32(state + base * d + col_base);
                float32x4_t r1 = vld1q_f32(state + (base+1) * d + col_base);
                float32x4_t r2 = vld1q_f32(state + (base+2) * d + col_base);
                float32x4_t r3 = vld1q_f32(state + (base+3) * d + col_base);

                vst1q_f32(state + base * d + col_base, vmlaq_f32(r0, kv, s0));
                vst1q_f32(state + (base+1) * d + col_base, vmlaq_f32(r1, kv, s1));
                vst1q_f32(state + (base+2) * d + col_base, vmlaq_f32(r2, kv, s2));
                vst1q_f32(state + (base+3) * d + col_base, vmlaq_f32(r3, kv, s3));
            }
        }
    }
}

/* Kernel 1b: S_init @ Q^T -> sq[cs][d] */
static void ssm_kernel_sq(
        float *sq,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *q,   /* [cs][d] */
        int d, int cs)
{
    int d4 = d / 4;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *sqi = sq + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0);
            float32x4_t acc3 = vdupq_n_f32(0);

            for (int v = 0; v < d4; v++) {
                int col = v * 4;
                float32x4_t qv = vld1q_f32(qi + col);
                float32x4_t s0 = vld1q_f32(state + base * d + col);
                float32x4_t s1 = vld1q_f32(state + (base+1) * d + col);
                float32x4_t s2 = vld1q_f32(state + (base+2) * d + col);
                float32x4_t s3 = vld1q_f32(state + (base+3) * d + col);
                acc0 = vmlaq_f32(acc0, s0, qv);
                acc1 = vmlaq_f32(acc1, s1, qv);
                acc2 = vmlaq_f32(acc2, s2, qv);
                acc3 = vmlaq_f32(acc3, s3, qv);
            }

            sqi[base] = vaddvq_f32_compat(acc0);
            sqi[base+1] = vaddvq_f32_compat(acc1);
            sqi[base+2] = vaddvq_f32_compat(acc2);
            sqi[base+3] = vaddvq_f32_compat(acc3);
        }
    }
}
#endif /* PICOLM_NEON */

/* ================================================================
 * Chunked DeltaNet recurrence.
 *
 * Replaces the sequential per-token recurrence with a chunked
 * algorithm that processes CS tokens at a time using triangular
 * matrix operations. This converts O(n_tokens * d_state^2)
 * sequential work into O(n_tokens * d_state^2) parallel work.
 *
 * Per v-head, the recurrence is:
 *   S *= ge[t]          (scalar decay)
 *   sk = S @ k[t]       (d_state vector)
 *   d = (v[t] - sk) * beta[t]  (d_state vector)
 *   S += outer(k[t], d) (rank-1 update)
 *   out = S @ q[t]      (d_state vector)
 *
 * Within a chunk, we unroll this into:
 *   1. Compute cumulative decay D[t] = prod(ge[0..t])
 *   2. Build CS x CS decay mask: decay[i][j] = D[i]/D[j]
 *   3. Compute interaction matrix kb[i][j] = k[i].k[j]*beta[j] * decay
 *   4. Forward-substitute to get V_hat (solved V values)
 *   5. Compute output from initial state + intra-chunk attention
 *   6. Update state for next chunk
 *
 * Each v-head is independent; parallelized via tensor_parallel_for.
 * ================================================================ */

typedef struct {
    int idx;               /* v-head index */
    int d_state;
    int cs;                /* actual chunk size (last chunk may be smaller) */
    int repeat;            /* n_v_heads / n_k_heads */

    /* Input data: Q, K are [CS][d_state] per k-head, V is [CS][d_state] per v-head */
    const float *q;        /* [cs][d_state] for this k-head */
    const float *k;        /* [cs][d_state] for this k-head */
    const float *v;        /* [cs][d_state] for this v-head */
    const float *gate_log; /* [cs] log(gate_exp) for this v-head */
    const float *beta;     /* [cs] beta for this v-head */

    /* State: [d_state][d_state] for this v-head (in/out) */
    float *state;

    /* Output: [cs][d_state] for this v-head */
    float *out;

    /* Scratch buffer pointer (allocated externally) */
    float *scratch;
} ssm_chunk_head_task_t;

/* Process one v-head's chunked recurrence.
 * Corrected formulation following the DeltaNet chunking derivation:
 *
 * Per-token recurrence:
 *   S *= ge              (scalar decay)
 *   sk[r] = S[r] . k     (d_state vector, one per row)
 *   delta[r] = beta * (v[r] - sk[r])
 *   S += outer(k, delta) (rank-1 update)
 *   out[r] = S[r] . q    (d_state vector output)
 *
 * Chunked formulation (per v-head):
 *   cum_g[t] = sum_{j=0}^{t} log(ge[j])
 *   decay[i][j] = exp(cum_g[i] - cum_g[j]) for i >= j
 *
 *   V_eff[i] = beta[i] * (v[i] - decay[i] * S_init . k[i])
 *   M[i][j] = (k[i] . k[j]) * decay[i][j]   (no beta on k[j])
 *   V_hat[i] = V_eff[i] - beta[i] * sum_{j<i} M[i][j] * V_hat[j]
 *
 *   out[i] = decay[i] * S_init . q[i]
 *          + sum_{j<=i} (k[j] . q[i]) * decay[i][j] * V_hat[j]
 *
 *   S_new = decay[last] * S_init
 *         + sum_j decay[last][j] * outer(V_hat[j], k[j])
 */
static void ssm_chunk_head_task(int h, void *ctxp) {
    ssm_chunk_head_task_t *tasks = (ssm_chunk_head_task_t *)ctxp;
    ssm_chunk_head_task_t *ctx = &tasks[h];
    int d = ctx->d_state;
    int cs = ctx->cs;
    int kh = h / ctx->repeat;

    /* Point to this head's data within the chunk */
    const float *q = ctx->q + (size_t)kh * cs * d;
    const float *k = ctx->k + (size_t)kh * cs * d;
    const float *v = ctx->v + (size_t)h * cs * d;
    const float *gate_log = ctx->gate_log + h * cs;
    const float *beta = ctx->beta + h * cs;

    float *state = ctx->state + (size_t)h * d * d;
    /* ctx->out already offset by h*cs*d_state; no additional offset needed */
    float *out = ctx->out;

    /* Allocate scratch from the pre-allocated buffer.
     * We need:
     *   cum_g[cs], q_decay[cs]
     *   decay_mask[cs*cs], M[cs*cs]  (interaction matrix, no beta)
     *   v_eff[cs*d], v_hat[cs*d]
     *   sk[cs*d] (for AVX-512 kernel 1: S_init @ K^T)
     *   sq[cs*d] (for AVX-512 kernel 1b: S_init @ Q^T)
     *   kq[cs*cs] (for AVX-512 kernel 3: K @ Q^T)
     *   decay_to_end[cs] (for kernel 4: state update)
     * Total: 2*cs + 2*cs*cs + 3*cs*d + cs*d + cs*cs + cs floats
     * For CS=64, d=128: ~38KB per head */
    float *sp = ctx->scratch;

    float *cum_g = sp; sp += cs;
    float *q_decay = sp; sp += cs;
    float *decay_mask = sp; sp += cs * cs;
    float *M_mat = sp; sp += cs * cs;
    float *v_eff = sp; sp += cs * d;
    float *v_hat = sp; sp += cs * d;

#ifdef PICOLM_AVX512
    float *sk = sp; sp += cs * d;        /* S_init @ K^T */
    float *sq = sp; sp += cs * d;        /* S_init @ Q^T */
    float *kq = sp; sp += cs * cs;       /* K @ Q^T */
    float *decay_to_end = sp; sp += cs;  /* decay from each j to end */
#elif defined(PICOLM_AVX2) || defined(PICOLM_AVX) || defined(PICOLM_NEON)
    float *sk = sp; sp += cs * d;        /* S_init @ K^T */
    float *sq = sp; sp += cs * d;        /* S_init @ Q^T */
    float *kq = sp; sp += cs * cs;       /* K @ Q^T */
    float *decay_to_end = sp; sp += cs;  /* decay from each j to end */
#else
    (void)sp; /* silence unused warning */
#endif

    /* Step 1: Compute cumulative log-decay and decay from start */
    {
        float cum = 0.0f;
        for (int t = 0; t < cs; t++) {
            cum += gate_log[t];
            cum_g[t] = cum;
            float ex = cum;
            if (ex > 50.0f) ex = 50.0f;
            if (ex < -50.0f) ex = -50.0f;
            q_decay[t] = expf(ex);
        }
    }

    /* Pre-compute decay from each position to end of chunk (for state update) */
#if defined(PICOLM_AVX512) || defined(PICOLM_AVX2) || defined(PICOLM_AVX) || defined(PICOLM_NEON)
    {
        float cum_last = cum_g[cs - 1];
        for (int j = 0; j < cs; j++) {
            float diff = cum_last - cum_g[j];
            if (diff > 50.0f) diff = 50.0f;
            if (diff < -50.0f) diff = -50.0f;
            decay_to_end[j] = expf(diff);
        }
    }
#endif

    /* Build decay mask: decay_mask[i][j] = exp(cum_g[i] - cum_g[j]) for j <= i */
    for (int i = 0; i < cs; i++) {
        for (int j = 0; j <= i; j++) {
            float dm;
            if (i == j) {
                dm = 1.0f;
            } else {
                float diff = cum_g[i] - cum_g[j];
                if (diff > 50.0f) diff = 50.0f;
                if (diff < -50.0f) diff = -50.0f;
                dm = expf(diff);
            }
            decay_mask[i * cs + j] = dm;
        }
        for (int j = i + 1; j < cs; j++) {
            decay_mask[i * cs + j] = 0.0f;
        }
    }

#if defined(PICOLM_AVX512) || defined(PICOLM_AVX2) || defined(PICOLM_AVX) || defined(PICOLM_NEON)
    /* === AVX-512 / AVX2 / NEON micro-kernel path === */

    /* Kernel 2: Interaction matrix M = K @ K^T with decay mask */
    ssm_kernel_interaction(M_mat, k, decay_mask, d, cs);

    /* Kernel 1: sk = S_init @ K^T (for V_eff computation) */
    ssm_kernel_veff(sk, state, k, d, cs);

    /* Step 3 (scalar post-processing): V_eff[i] = beta[i] * (v[i] - decay[i] * sk[i]) */
    for (int i = 0; i < cs; i++) {
        float decay_i = q_decay[i];
        float bt = beta[i];
        const float *vi = v + i * d;
        const float *ski = sk + i * d;
        float *veffi = v_eff + i * d;
        for (int r = 0; r < d; r++) {
            veffi[r] = bt * (vi[r] - decay_i * ski[r]);
        }
    }

    /* Step 4: Forward substitution (sequential, scalar, cheap) */
    for (int i = 0; i < cs; i++) {
        float bt = beta[i];
        for (int r = 0; r < d; r++) {
            float sum_mv = 0.0f;
            for (int j = 0; j < i; j++) {
                sum_mv += M_mat[i * cs + j] * v_hat[j * d + r];
            }
            v_hat[i * d + r] = v_eff[i * d + r] - bt * sum_mv;
        }
    }

    /* Kernel 1b: sq = S_init @ Q^T (for initial state contribution to output) */
    ssm_kernel_sq(sq, state, q, d, cs);

    /* Kernel 3: kq = K @ Q^T with decay mask */
    ssm_kernel_output_cross(kq, k, q, decay_mask, d, cs);

    /* Step 5 (scalar assembly): out[i][r] = sq[i][r] * decay[i] + sum_{j<=i} kq[i][j] * v_hat[j][r] */
    for (int i = 0; i < cs; i++) {
        float decay_i = q_decay[i];
        const float *sqi = sq + i * d;
        float *outi = out + i * d;
        const float *kqi = kq + i * cs;

        /* Initial state contribution (scaled by decay) */
        for (int r = 0; r < d; r++) {
            outi[r] = sqi[r] * decay_i;
        }

        /* Intra-chunk contribution */
        for (int j = 0; j <= i; j++) {
            float attn = kqi[j];
            if (attn == 0.0f) continue;
            const float *vht = v_hat + j * d;
            for (int r = 0; r < d; r++) {
                outi[r] += attn * vht[r];
            }
        }
    }

    /* Kernel 4: State update (in-place: decay + accumulate weighted outer products) */
    {
        float cum_last = cum_g[cs - 1];
        float total_decay;
        {
            float ex = cum_last;
            if (ex > 50.0f) ex = 50.0f;
            if (ex < -50.0f) ex = -50.0f;
            total_decay = expf(ex);
        }
        ssm_kernel_state_update(state, v_hat, k, decay_to_end, total_decay, d, cs);
    }
#else
    /* === Scalar path (reference) === */

    /* Step 2: Compute interaction matrix M[i][j] = k_i . k_j * decay (scalar) */
    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        for (int j = 0; j <= i; j++) {
            float dot = 0.0f;
            const float *kj = k + j * d;
            for (int di = 0; di < d; di++) dot += ki[di] * kj[di];
            M_mat[i * cs + j] = dot * decay_mask[i * cs + j];
        }
        for (int j = i + 1; j < cs; j++) {
            M_mat[i * cs + j] = 0.0f;
        }
    }

    /* Step 3: Compute V_eff[i] = beta[i] * (v[i] - decay[i] * S_init . k[i]) */
    for (int i = 0; i < cs; i++) {
        float decay_i = q_decay[i];
        float bt = beta[i];
        const float *ki = k + i * d;
        for (int r = 0; r < d; r++) {
            float s_dot_k = 0.0f;
            const float *st_row = state + r * d;
            for (int c = 0; c < d; c++) s_dot_k += st_row[c] * ki[c];
            v_eff[i * d + r] = bt * (v[i * d + r] - decay_i * s_dot_k);
        }
    }

    /* Step 4: Forward substitution. */
    for (int i = 0; i < cs; i++) {
        float bt = beta[i];
        for (int r = 0; r < d; r++) {
            float sum_mv = 0.0f;
            for (int j = 0; j < i; j++) {
                sum_mv += M_mat[i * cs + j] * v_hat[j * d + r];
            }
            v_hat[i * d + r] = v_eff[i * d + r] - bt * sum_mv;
        }
    }

    /* Step 5: Compute output. */
    for (int i = 0; i < cs; i++) {
        float decay_i = q_decay[i];
        const float *qi = q + i * d;

        /* Initial state contribution */
        for (int r = 0; r < d; r++) {
            float s_dot_q = 0.0f;
            const float *st_row = state + r * d;
            for (int c = 0; c < d; c++) s_dot_q += st_row[c] * qi[c];
            out[i * d + r] = s_dot_q * decay_i;
        }

        /* Intra-chunk contribution (includes j == i) */
        for (int j = 0; j <= i; j++) {
            float dm = decay_mask[i * cs + j];
            float k_dot_q = 0.0f;
            const float *kj = k + j * d;
            for (int di = 0; di < d; di++) k_dot_q += kj[di] * qi[di];
            float attn = k_dot_q * dm;
            const float *vht = v_hat + j * d;
            float *outi = out + i * d;
            for (int r = 0; r < d; r++) outi[r] += attn * vht[r];
        }
    }

    /* Step 6: Update state for next chunk. */
    {
        float total_decay;
        {
            float cum = cum_g[cs - 1];
            if (cum > 50.0f) cum = 50.0f;
            if (cum < -50.0f) cum = -50.0f;
            total_decay = expf(cum);
        }

        for (int r = 0; r < d; r++) {
            for (int c = 0; c < d; c++) {
                float update = 0.0f;
                for (int j = 0; j < cs; j++) {
                    float diff = cum_g[cs - 1] - cum_g[j];
                    float decay_to_end;
                    if (diff > 50.0f) diff = 50.0f;
                    if (diff < -50.0f) diff = -50.0f;
                    decay_to_end = expf(diff);
                    update += v_hat[j * d + r] * k[j * d + c] * decay_to_end;
                }
                state[r * d + c] = state[r * d + c] * total_decay + update;
            }
        }
    }
#endif
}

/* Chunked SSM recurrence: replaces the sequential per-token loop.
 * Processes all n_tokens in chunks of CS, parallelized across v-heads.
 *
 * conv_batch layout: [n_tokens][conv_dim] where conv_dim = 2*qk_dim + value_dim
 *   Within each token: [Q[n_k_heads][d_state] | K[n_k_heads][d_state] | V[n_v_heads][head_v_dim]]
 * alpha_batch: [n_tokens][n_v_heads] gate_log values (log-space: softplus(alpha+dt_w)*a_w)
 * beta_batch:  [n_tokens][n_v_heads] sigmoid(beta) values
 * state:       [n_v_heads][d_state][d_state] recurrent state (updated in-place)
 * xb2_batch:   [n_tokens][value_dim] head-major output [h*head_v_dim + d]
 */
static void ssm_chunked_recurrence(
        const float *conv_batch,
        const float *alpha_batch,
        const float *beta_batch,
        float *state,
        float *xb2_batch,
        int n_tokens, int value_dim,
        int d_state, int n_k_heads, int n_v_heads, int head_v_dim, int repeat,
        int conv_dim, int cs)
{
    /* cs: chunk size (from model->ssm_chunk_size, default 64, 0=auto) */
    if (cs <= 0) cs = 64; /* default */
    if (cs > n_tokens) cs = n_tokens;
    int n_chunks = (n_tokens + cs - 1) / cs;
    if (n_chunks < 1) n_chunks = 1;

    int qk_dim = d_state * n_k_heads;

    /* Compute scratch size per head for ssm_chunk_head_task.
     * Base: cum_g[cs] + q_decay[cs] + decay_mask[cs*cs] + M_mat[cs*cs]
     *       + v_eff[cs*d] + v_hat[cs*d]
     * AVX-512 extras: sk[cs*d] + sq[cs*d] + kq[cs*cs] + decay_to_end[cs]
     * Total: 3*cs + 3*cs*cs + 4*cs*d (for AVX-512)
     * For cs=64, d=128: ~38KB per head */
    size_t scratch_per_head = 3UL * cs + 3UL * cs * cs + 4UL * cs * (size_t)d_state;
    scratch_per_head = (scratch_per_head + 15) & ~15UL;

    /* Allocate task contexts and per-head scratch separately to avoid stride issues */
    ssm_chunk_head_task_t *tasks = (ssm_chunk_head_task_t *)calloc(n_v_heads, sizeof(ssm_chunk_head_task_t));
    float *scratch_pool = (float *)calloc(n_v_heads * scratch_per_head, sizeof(float));
    if (!tasks || !scratch_pool) { fprintf(stderr, "OOM: chunk tasks\n"); exit(1); }
    for (int h = 0; h < n_v_heads; h++) {
        tasks[h].scratch = scratch_pool + (size_t)h * scratch_per_head;
    }

    /* Allocate gather buffers for contiguous chunk data.
     * Layout: [n_heads][cs][d] for Q/K/V, [n_v_heads][cs] for scalars.
     * This matches the access pattern in ssm_chunk_head_task:
     *   q[kh*cs*d + t*d + di], k[kh*cs*d + t*d + di], v[h*cs*d + t*d + di]
     *   gate_log[h*cs + t], beta[h*cs + t] */
    size_t cs_alloc = cs; /* runtime chunk size */
    float *chunk_q = (float *)malloc(cs_alloc * (size_t)n_k_heads * (size_t)d_state * sizeof(float));
    float *chunk_k = (float *)malloc(cs_alloc * (size_t)n_k_heads * (size_t)d_state * sizeof(float));
    float *chunk_v = (float *)malloc(cs_alloc * (size_t)n_v_heads * (size_t)head_v_dim * sizeof(float));
    float *chunk_beta = (float *)malloc(cs_alloc * (size_t)n_v_heads * sizeof(float));
    float *gate_log = (float *)malloc(cs_alloc * (size_t)n_v_heads * sizeof(float));
    float *chunk_out = (float *)malloc(cs_alloc * (size_t)n_v_heads * (size_t)d_state * sizeof(float));
    if (!chunk_q || !chunk_k || !chunk_v || !chunk_beta || !gate_log || !chunk_out) {
        fprintf(stderr, "OOM: chunk gather buffers\n"); exit(1);
    }

    /* Process chunk by chunk */
    for (int ci = 0; ci < n_chunks; ci++) {
        int cs_actual = (ci == n_chunks - 1) ? (n_tokens - ci * cs) : cs;
        if (cs_actual <= 0) break;
        int chunk_start = ci * cs;

        /* Gather Q, K, V, beta, gate_log for this chunk.
         * Layout: [n_heads][cs_actual][d] for Q/K/V, [n_v_heads][cs_actual] for scalars. */
        for (int h = 0; h < n_k_heads; h++) {
            for (int t = 0; t < cs_actual; t++) {
                const float *tok = conv_batch + (chunk_start + t) * conv_dim;
                memcpy(chunk_q + (size_t)h * cs_actual * d_state + t * d_state,
                       tok + h * d_state, d_state * sizeof(float));
                memcpy(chunk_k + (size_t)h * cs_actual * d_state + t * d_state,
                       tok + qk_dim + h * d_state, d_state * sizeof(float));
            }
        }
        for (int h = 0; h < n_v_heads; h++) {
            for (int t = 0; t < cs_actual; t++) {
                const float *tok = conv_batch + (chunk_start + t) * conv_dim;
                memcpy(chunk_v + (size_t)h * cs_actual * head_v_dim + t * head_v_dim,
                       tok + 2 * qk_dim + h * head_v_dim, head_v_dim * sizeof(float));
                chunk_beta[h * cs_actual + t] = beta_batch[(chunk_start + t) * n_v_heads + h];
                gate_log[h * cs_actual + t] = alpha_batch[(chunk_start + t) * n_v_heads + h];
            }
        }

        /* Initialize each task */
        for (int h = 0; h < n_v_heads; h++) {
            tasks[h].idx = h;
            tasks[h].d_state = d_state;
            tasks[h].cs = cs_actual;
            tasks[h].repeat = repeat;
            /* Now Q/K/V are contiguous [cs_actual][n_heads][d], so we can slice by head */
            tasks[h].q = chunk_q;
            tasks[h].k = chunk_k;
            tasks[h].v = chunk_v;
            tasks[h].gate_log = gate_log;
            tasks[h].beta = chunk_beta;
            tasks[h].state = state;
            /* Output: [cs_actual][d_state] per v-head */
            tasks[h].out = chunk_out + (size_t)h * cs_actual * d_state;
        }

        /* Process all v-heads in parallel */
        tensor_parallel_for(n_v_heads, ssm_chunk_head_task, tasks);

        /* Reshape output from [n_v_heads][cs_actual][d_state] to [cs_actual][value_dim] head-major.
         * chunk_out[h * cs_actual * d_state + t * d_state + r] -> xb2_batch[(chunk_start+t)*value_dim + h*head_v_dim + r] */
        for (int t = 0; t < cs_actual; t++) {
            float *out_tok = xb2_batch + (chunk_start + t) * value_dim;
            for (int h = 0; h < n_v_heads; h++) {
                const float *oh = chunk_out + (size_t)h * cs_actual * d_state + t * d_state;
                float *od = out_tok + h * head_v_dim;
                memcpy(od, oh, d_state * sizeof(float));
            }
        }

            }

    free(tasks); free(scratch_pool);
    free(chunk_q); free(chunk_k); free(chunk_v);
    free(chunk_beta); free(gate_log); free(chunk_out);
}

#ifdef PICOLM_GPU
/* Persistent device scratch buffers for ssm_forward()'s device-native
 * vecdot calls (picolm_gpu_ssm_vecdot_dev). The _dev variant needs
 * device-resident x/out pointers -- weights and head_map are already
 * device-resident from model load, but the activation (s->xb) lives
 * on the host and the small per-head output needs a device buffer
 * before D2H back. Sized once from dim/n_v_heads and reused. */
static void *g_ssm_vecdot_x_dev = NULL;
static size_t g_ssm_vecdot_x_bytes = 0;
static void *g_ssm_vecdot_out_dev = NULL;
static size_t g_ssm_vecdot_out_bytes = 0;
static int g_ssm_vecdot_device = -1;

static int ssm_vecdot_dev_scratch_ensure(int device, size_t x_bytes, size_t out_bytes) {
    if (g_ssm_vecdot_device != device) {
        g_ssm_vecdot_x_dev = NULL; g_ssm_vecdot_x_bytes = 0;
        g_ssm_vecdot_out_dev = NULL; g_ssm_vecdot_out_bytes = 0;
        g_ssm_vecdot_device = device;
    }
    if (!g_ssm_vecdot_x_dev || g_ssm_vecdot_x_bytes < x_bytes) {
        g_ssm_vecdot_x_dev = picolm_gpu_alloc_device(x_bytes, device);
        if (!g_ssm_vecdot_x_dev) return 0;
        g_ssm_vecdot_x_bytes = x_bytes;
    }
    if (!g_ssm_vecdot_out_dev || g_ssm_vecdot_out_bytes < out_bytes) {
        g_ssm_vecdot_out_dev = picolm_gpu_alloc_device(out_bytes, device);
        if (!g_ssm_vecdot_out_dev) return 0;
        g_ssm_vecdot_out_bytes = out_bytes;
    }
    return 1;
}

/* Device scratch buffers for batched matmul via picolm_gpu_matmul_dev.
 * Unlike the host-wrapper picolm_gpu_matmul() (called via matmul()/
 * tensor_set_gpu_tensor), which does TWO gpuDeviceSynchronize() calls
 * per Q8_0 matmul (one after quantize-x, one after the matmul kernel),
 * matmul_dev has zero internal syncs. Kernels on the same stream
 * execute in submission order, so two back-to-back matmul_dev calls
 * need only one sync total -- when host code actually reads results.
 * This is ~4x fewer syncs for the attn_qkv+attn_gate_ssm pair (1 vs 4). */
static void *g_ssm_mm_xb_dev = NULL;      static size_t g_ssm_mm_xb_bytes = 0;
static void *g_ssm_mm_qkv_dev = NULL;     static size_t g_ssm_mm_qkv_bytes = 0;
static void *g_ssm_mm_gate_dev = NULL;    static size_t g_ssm_mm_gate_bytes = 0;
static void *g_ssm_mm_outin_dev = NULL;   static size_t g_ssm_mm_outin_bytes = 0;
static void *g_ssm_mm_outres_dev = NULL;  static size_t g_ssm_mm_outres_bytes = 0;
static int g_ssm_mm_device = -1;

static int ssm_mm_alloc(void **buf, size_t *cap, size_t need, int device) {
    if (*buf && *cap >= need) return 1;
    *buf = picolm_gpu_alloc_device(need, device);
    if (!*buf) { *cap = 0; return 0; }
    *cap = need;
    return 1;
}

static int ssm_qkv_gate_dev_scratch_ensure(int device, size_t xb_bytes,
                                            size_t qkv_bytes, size_t gate_bytes) {
    if (g_ssm_mm_device != device) {
        g_ssm_mm_xb_dev = NULL; g_ssm_mm_xb_bytes = 0;
        g_ssm_mm_qkv_dev = NULL; g_ssm_mm_qkv_bytes = 0;
        g_ssm_mm_gate_dev = NULL; g_ssm_mm_gate_bytes = 0;
        g_ssm_mm_outin_dev = NULL; g_ssm_mm_outin_bytes = 0;
        g_ssm_mm_outres_dev = NULL; g_ssm_mm_outres_bytes = 0;
        g_ssm_mm_device = device;
    }
    return ssm_mm_alloc(&g_ssm_mm_xb_dev, &g_ssm_mm_xb_bytes, xb_bytes, device) &&
           ssm_mm_alloc(&g_ssm_mm_qkv_dev, &g_ssm_mm_qkv_bytes, qkv_bytes, device) &&
           ssm_mm_alloc(&g_ssm_mm_gate_dev, &g_ssm_mm_gate_bytes, gate_bytes, device);
}

static int ssm_out_dev_scratch_ensure(int device, size_t in_bytes, size_t out_bytes) {
    if (g_ssm_mm_device != device) {
        g_ssm_mm_xb_dev = NULL; g_ssm_mm_xb_bytes = 0;
        g_ssm_mm_qkv_dev = NULL; g_ssm_mm_qkv_bytes = 0;
        g_ssm_mm_gate_dev = NULL; g_ssm_mm_gate_bytes = 0;
        g_ssm_mm_outin_dev = NULL; g_ssm_mm_outin_bytes = 0;
        g_ssm_mm_outres_dev = NULL; g_ssm_mm_outres_bytes = 0;
        g_ssm_mm_device = device;
    }
    return ssm_mm_alloc(&g_ssm_mm_outin_dev, &g_ssm_mm_outin_bytes, in_bytes, device) &&
           ssm_mm_alloc(&g_ssm_mm_outres_dev, &g_ssm_mm_outres_bytes, out_bytes, device);
}
#endif

void ssm_forward(model_t *m, run_state_t *s, float *x, float *residual,
                        layer_weights_t *lw, int il, int pos, void *gpu_lw) {
#ifndef PICOLM_GPU
    (void)gpu_lw;
    (void)pos;
#endif
    model_config_t *c = &m->config;
    int dim = c->n_embd;
    int d_conv = c->ssm_d_conv;
    int d_state = c->ssm_d_state;
    int n_k_heads = c->ssm_n_group;
    int n_v_heads = c->ssm_dt_rank;
    int conv_dim = 2 * d_state * n_k_heads + c->ssm_d_inner;
    int head_v_dim = c->ssm_d_inner / n_v_heads;
    float eps = c->rms_norm_eps;
    
    /* Qwen3.5 GGUF v-head reorder parameters (used throughout this function) */
    int n_k = c->ssm_n_group;
    int n_vpk = n_v_heads / n_k;
    int half_vpk = n_vpk / 2;
    int do_remap = !m->from_safetensors && n_k > 0 && n_k < n_v_heads && half_vpk > 0;

    /* Scratch space: dedicated SSM buffer */
    float *tmp = s->ssm_tmp;

    /* 1. RMSNorm (attn_norm) */
    rmsnorm(s->xb, x, s->attn_norm_w[il], dim, eps);
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("xb[:8]", s->xb, 8, 8);
#endif
    /* 2. QKV projection: qkv_mixed = matmul(attn_qkv, xb) -> [conv_dim] */
    /* 3. Z gate: z = matmul(attn_gate_ssm, xb) -> [value_dim] */
    /* Batched: both read the same xb, so one H2D, two matmul_dev,
     * one sync, two D2H. Replaces two matmul() calls each doing
     * H2D + 2x sync + D2H (4 syncs total for Q8_0 -> 1 sync). */
    int ssm_qkv_gate_gpu_done = 0;
#ifdef PICOLM_GPU
    if (gpu_lw) {
        gpu_layer_weights_t *gl = (gpu_layer_weights_t *)gpu_lw;
        size_t xb_bytes = (size_t)dim * sizeof(float);
        size_t qkv_bytes = (size_t)conv_dim * sizeof(float);
        size_t gate_bytes = (size_t)c->ssm_d_inner * sizeof(float);
        if (gl->attn_qkv && gl->attn_gate_ssm &&
            ssm_qkv_gate_dev_scratch_ensure(m->gpu.device, xb_bytes, qkv_bytes, gate_bytes) &&
            picolm_gpu_memcpy(g_ssm_mm_xb_dev, s->xb, xb_bytes, 1, m->gpu.device) &&
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_qkv,
                                   g_ssm_mm_qkv_dev, g_ssm_mm_xb_dev, 1, m->gpu.device, 0, 0) &&
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_gate_ssm,
                                   g_ssm_mm_gate_dev, g_ssm_mm_xb_dev, 1, m->gpu.device, 0, 0) &&
            picolm_gpu_sync(m->gpu.device) &&
            picolm_gpu_memcpy(s->q, g_ssm_mm_qkv_dev, qkv_bytes, -1, m->gpu.device) &&
            picolm_gpu_memcpy(s->xb2, g_ssm_mm_gate_dev, gate_bytes, -1, m->gpu.device)) {
            ssm_qkv_gate_gpu_done = 1;
        }
    }
#endif
    if (!ssm_qkv_gate_gpu_done)
    {
        matmul(s->q, s->xb, lw->attn_qkv, dim, conv_dim, lw->type_attn_qkv);
        matmul(s->xb2, s->xb, lw->attn_gate_ssm, dim, c->ssm_d_inner, lw->type_attn_gate_ssm);
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("qkv[:8]", s->q, 8, 8);
#endif
    /* If GGUF reorders v-head rows, convert xb2 from GGUF order to sequential order */
    if (do_remap) {
        float *xb2_tmp = (float *)malloc(c->ssm_d_inner * sizeof(float));
        memcpy(xb2_tmp, s->xb2, c->ssm_d_inner * sizeof(float));
        for (int h = 0; h < n_v_heads; h++) {
            int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
            memcpy(s->xb2 + h * head_v_dim, xb2_tmp + gh * head_v_dim, head_v_dim * sizeof(float));
        }
        free(xb2_tmp);
        if (il == 0 || il == 8 || il == 16 || il == 32 || il == 48 || il == 60) {
            }
    }

    /* 4. Convolution: compute BEFORE shifting conv_state */
    float *conv_state = s->ssm_conv_state[il];
    int state_stride = conv_dim;
    int n_state_rows = d_conv - 1;
    float *conv_output = tmp; /* [conv_dim] */
    float *conv1d_w = s->ssm_conv1d_w[il];
    for (int co = 0; co < conv_dim; co++) {
        float sum = 0.0f;
        for (int d = 0; d < n_state_rows; d++) {
            sum += conv1d_w[d + co * d_conv] * conv_state[d * state_stride + co];
        }
        sum += conv1d_w[(d_conv - 1) + co * d_conv] * s->q[co];
        float v = sum;
        conv_output[co] = v * (1.0f / (1.0f + expf(-v))); /* silu */
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("conv_out[:8]", conv_output, 8, 8);
#endif

    /* Shift conv_state left and append new token */
    for (int r = 0; r < n_state_rows - 1; r++) {
        memcpy(conv_state + r * state_stride, conv_state + (r + 1) * state_stride, state_stride * sizeof(float));
    }
    memcpy(conv_state + (n_state_rows - 1) * state_stride, s->q, state_stride * sizeof(float));

    /* 6. Split into Q, K, V from conv_output (contiguous layout)
     * conv_output: [conv_dim] = [q_part + k_part + v_part]
     * Q: [head_k_dim, n_k_heads] stored head-major: [h*d_state + d]
     * K: [head_k_dim, n_k_heads] stored head-major
     * V: [head_v_dim, n_v_heads] stored head-major
     */
    int qk_dim = d_state * n_k_heads;
    float *q_conv = tmp + conv_dim; /* [qk_dim] */
    float *k_conv = tmp + conv_dim + qk_dim; /* [qk_dim] */
    float *v_conv = tmp + conv_dim + 2 * qk_dim; /* [c->ssm_d_inner] */

    memcpy(q_conv, conv_output, qk_dim * sizeof(float));
    memcpy(k_conv, conv_output + qk_dim, qk_dim * sizeof(float));
    memcpy(v_conv, conv_output + 2 * qk_dim, c->ssm_d_inner * sizeof(float));

    /* If GGUF reorders V channels, convert v_conv from GGUF order to sequential order */
    if (do_remap) {
        float *v_conv_tmp = (float *)malloc(c->ssm_d_inner * sizeof(float));
        memcpy(v_conv_tmp, v_conv, c->ssm_d_inner * sizeof(float));
        for (int h = 0; h < n_v_heads; h++) {
            int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
            memcpy(v_conv + h * head_v_dim, v_conv_tmp + gh * head_v_dim, head_v_dim * sizeof(float));
        }
        free(v_conv_tmp);
    }

    /* 7. L2 normalize Q and K per k_head */
    for (int h = 0; h < n_k_heads; h++) {
        float *qh = q_conv + h * d_state;
        float nrm = 0.0f;
        for (int d = 0; d < d_state; d++) nrm += qh[d] * qh[d];
        nrm = 1.0f / sqrtf(nrm + 1e-12f);
        for (int d = 0; d < d_state; d++) qh[d] *= nrm;
    }
    for (int h = 0; h < n_k_heads; h++) {
        float *kh = k_conv + h * d_state;
        float nrm = 0.0f;
        for (int d = 0; d < d_state; d++) nrm += kh[d] * kh[d];
        nrm = 1.0f / sqrtf(nrm + 1e-12f);
        for (int d = 0; d < d_state; d++) kh[d] *= nrm;
    }

    /* 8. Scale Q by 1/sqrt(d_state) */
    float q_scale = 1.0f / sqrtf((float)d_state);
    for (int i = 0; i < qk_dim; i++) q_conv[i] *= q_scale;
#ifdef DEBUG_SSM
    if (il == 0 || il == 8 || il == 16 || il == 32 || il == 48 || il == 60) {
        dbg_vec("q_conv_scaled[:8]", q_conv, 8, 8);
        dbg_vec("k_conv_scaled[:8]", k_conv, 8, 8);
    }
#endif

    /* Alpha/beta per-head projections share the same activation vector
     * (s->xb) across every one of the (up to dozens of) v-heads -- quantize
     * it to Q8_0 once here and reuse the fast int8 x int8 kernels
     * (vec_dot_q8_0_q8_0_deltas / vec_dot_q4_0_q8_0) for whichever of
     * alpha/beta uses a Q8_0 or Q4_0 weight, instead of the mixed
     * int8-weight x float32-activation kernel vec_dot()'s generic
     * dispatch falls back to for those types -- same fix already applied
     * to attn_core's K-dot product and to the FFN/projection matmuls. */
    uint8_t xb_q8_stack[8192 / 32 * 34];
    void *xb_q8 = (size_t)(dim / 32) * 34 <= sizeof(xb_q8_stack) ? (void *)xb_q8_stack : malloc((size_t)(dim / 32) * 34);
    float xb_q8_d_stack[8192 / 32];
    float *xb_q8_d = (dim / 32) <= (int)(sizeof(xb_q8_d_stack) / sizeof(float)) ? xb_q8_d_stack : (float *)malloc(sizeof(float) * (dim / 32));
    {
        int nb_xb = dim / 32;
        quantize_row_q8_0(s->xb, xb_q8, dim);
        const block_q8_0 *xqb = (const block_q8_0 *)xb_q8;
        for (int k = 0; k < nb_xb; k++) xb_q8_d[k] = fp16_to_fp32_lookup(xqb[k].d);
    }

    /* 9. Alpha: alpha = matmul(ssm_alpha, xb) + ssm_dt.bias -> [dt_rank] */
    /* GGUF stores [dim, n_v_heads] column-major: each head has dim contiguous elements */
    /* GGUF v-heads may be in tiled/interleaved order. Map sequential h -> GGUF head index. */
    /* Mapping: sequential [k0v0, k0v1, k0v2, ..., k0v7, k1v0, ...] */
    /*           GGUF     [k0v0, k0v2, k0v4, k0v6, k1v0, ..., k0v1, k0v3, ...] */
    /* Helper: map sequential head h -> GGUF head index gh */
    /* qwen35_vhead_gguf defined at file scope */
#ifdef PICOLM_GPU
    /* Upload xb once, shared by both alpha and beta device-native vecdot.
     * Weights (ssm_alpha_dev/ssm_beta_dev) and head_map are already
     * device-resident from model load. Only the activation needs per-call H2D
     * (~20KB) instead of picolm_gpu_ssm_vecdot()'s ~255KB weight re-upload. */
    int ssm_vecdot_gpu_ready = 0;
    if (gpu_lw && m->gpu.active) {
        size_t vx_bytes = (size_t)dim * sizeof(float);
        size_t vout_bytes = (size_t)n_v_heads * sizeof(float);
        if (ssm_vecdot_dev_scratch_ensure(m->gpu.device, vx_bytes, vout_bytes) &&
            picolm_gpu_memcpy(g_ssm_vecdot_x_dev, s->xb, vx_bytes, 1, m->gpu.device)) {
            ssm_vecdot_gpu_ready = 1;
        }
    }
    const int *ssm_vecdot_hmap_dev = do_remap ? (const int *)m->gpu.ssm_head_map_dev : NULL;
#endif
    float *alpha_out = tmp + conv_dim + 2 * qk_dim + c->ssm_d_inner; /* [dt_rank] */
    {
        gguf_type_t alpha_type = lw->type_ssm_alpha;
        size_t row_bytes = gguf_type_row_size(alpha_type, dim);
        int alpha_map[256];
        for (int h = 0; h < n_v_heads; h++) alpha_map[h] = do_remap ? qwen35_vhead_gguf(h, n_vpk, n_k) : h;
#ifdef PICOLM_GPU
        if (gpu_lw && ssm_vecdot_gpu_ready && m->gpu.ssm_alpha_dev[il] &&
            (!do_remap || ssm_vecdot_hmap_dev) &&
            (alpha_type == GGUF_TYPE_F32 || alpha_type == GGUF_TYPE_Q4_0 || alpha_type == GGUF_TYPE_Q8_0) &&
            picolm_gpu_ssm_vecdot_dev(g_ssm_vecdot_out_dev, g_ssm_vecdot_x_dev,
                                       m->gpu.ssm_alpha_dev[il], alpha_type, dim,
                                       n_v_heads, (int)row_bytes, ssm_vecdot_hmap_dev,
                                       m->gpu.device) &&
            picolm_gpu_sync(m->gpu.device) &&
            picolm_gpu_memcpy(alpha_out, g_ssm_vecdot_out_dev,
                               (size_t)n_v_heads * sizeof(float), -1, m->gpu.device)) {
            for (int h = 0; h < n_v_heads; h++) alpha_out[h] += s->ssm_dt_w[il][h];
        } else
#endif
        {
            for (int h = 0; h < n_v_heads; h++) {
                int gh = alpha_map[h];
                const uint8_t *head_data = (const uint8_t *)lw->ssm_alpha + (size_t)gh * row_bytes;
                float sum;
                if (alpha_type == GGUF_TYPE_Q8_0) sum = vec_dot_q8_0_q8_0_deltas(xb_q8, xb_q8_d, head_data, dim);
                else if (alpha_type == GGUF_TYPE_Q4_0) sum = vec_dot_q4_0_q8_0(head_data, xb_q8, dim);
                else sum = vec_dot(head_data, s->xb, dim, alpha_type);
                alpha_out[h] = sum + s->ssm_dt_w[il][h];
            }
        }
        /* alpha_map is stack-allocated */
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("alpha[:8]", alpha_out, n_v_heads, 8);
#endif

    /* gate = ssm_a * softplus(alpha) -> [dt_rank] */
    float *gate = alpha_out + n_v_heads; /* [dt_rank] */
    for (int h = 0; h < n_v_heads; h++) {
        float a = alpha_out[h];
        float sp = (a > 20.0f) ? a : (a < -20.0f) ? expf(a) : logf(1.0f + expf(a));
        gate[h] = sp * s->ssm_a_w[il][h];
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("gate[:8]", gate, n_v_heads, 8);
#endif

    /* 10. Beta: sigmoid(matmul(ssm_beta, xb)) -> [dt_rank] */
    /* GGUF stores [dim, n_v_heads] column-major, v-heads may be tiled/interleaved */
    float *beta = gate + n_v_heads; /* [dt_rank] */
    {
        gguf_type_t beta_type = lw->type_ssm_beta;
        size_t row_bytes = gguf_type_row_size(beta_type, dim);
        int beta_map[256];
        for (int h = 0; h < n_v_heads; h++) beta_map[h] = do_remap ? qwen35_vhead_gguf(h, n_vpk, n_k) : h;
#ifdef PICOLM_GPU
        if (gpu_lw && ssm_vecdot_gpu_ready && m->gpu.ssm_beta_dev[il] &&
            (!do_remap || ssm_vecdot_hmap_dev) &&
            (beta_type == GGUF_TYPE_F32 || beta_type == GGUF_TYPE_Q4_0 || beta_type == GGUF_TYPE_Q8_0) &&
            picolm_gpu_ssm_vecdot_dev(g_ssm_vecdot_out_dev, g_ssm_vecdot_x_dev,
                                       m->gpu.ssm_beta_dev[il], beta_type, dim,
                                       n_v_heads, (int)row_bytes, ssm_vecdot_hmap_dev,
                                       m->gpu.device) &&
            picolm_gpu_sync(m->gpu.device) &&
            picolm_gpu_memcpy(beta, g_ssm_vecdot_out_dev,
                               (size_t)n_v_heads * sizeof(float), -1, m->gpu.device)) {
            for (int h = 0; h < n_v_heads; h++) beta[h] = 1.0f / (1.0f + expf(-beta[h]));
        } else
#endif
        {
            for (int h = 0; h < n_v_heads; h++) {
                int gh = beta_map[h];
                const uint8_t *head_data = (const uint8_t *)lw->ssm_beta + (size_t)gh * row_bytes;
                float sum;
                if (beta_type == GGUF_TYPE_Q8_0) sum = vec_dot_q8_0_q8_0_deltas(xb_q8, xb_q8_d, head_data, dim);
                else if (beta_type == GGUF_TYPE_Q4_0) sum = vec_dot_q4_0_q8_0(head_data, xb_q8, dim);
                else sum = vec_dot(head_data, s->xb, dim, beta_type);
                beta[h] = 1.0f / (1.0f + expf(-sum));
            }
        }
        /* beta_map is stack-allocated */
    }
    if (xb_q8 != (void *)xb_q8_stack) free(xb_q8);
    if (xb_q8_d != xb_q8_d_stack) free(xb_q8_d);
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("beta[:8]", beta, n_v_heads, 8);
#endif

    /* 11. Gate expansion: exp(gate) -> [dt_rank] */
    float *gate_exp = beta + n_v_heads; /* [dt_rank] */
    for (int h = 0; h < n_v_heads; h++) {
        float g = gate[h];
        float ge = (g < -50.0f) ? 0.0f : expf(g);
        gate_exp[h] = ge;
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("gate_exp[:8]", gate_exp, n_v_heads, 8);
#endif

    /* 12-17. State recurrence, threaded across n_v_heads (each head's
     * [d_state x d_state] state block is fully independent -- no
     * cross-head data dependency within a single token, only token-to-
     * token via `state` persisting across calls). Previously this was
     * four separate fully-serial for-h loops (decay/sk/update/output)
     * with zero threading on what is the dominant per-token FLOP cost
     * for SSM layers (O(n_v_heads * d_state^2) each for sk/update/output). */
    float *state = s->ssm_state[il];
    int repeat = n_v_heads / n_k_heads;
    float *ssm_output = gate_exp + n_v_heads; /* [d_state * n_v_heads], dim-major */

    ssm_head_ctx_t ssm_ctx;
    ssm_ctx.state = state; ssm_ctx.d_state = d_state; ssm_ctx.head_v_dim = head_v_dim;
    ssm_ctx.n_v_heads = n_v_heads; ssm_ctx.repeat = repeat;
    ssm_ctx.q_conv = q_conv; ssm_ctx.k_conv = k_conv; ssm_ctx.v_conv = v_conv;
    ssm_ctx.gate_exp = gate_exp; ssm_ctx.beta = beta;
    ssm_ctx.ssm_output = ssm_output; /* shared, dim-major [d*n_v_heads+h] */
#ifdef PICOLM_GPU
    if (1) {
        int rec_done = 0;
        if (m->gpu.ssm_state_dev[il]) {
            /* m->gpu.ssm_state_dev[il] is the ONLY up-to-date copy of this
             * layer's recurrence state once GPU-native decode is running:
             * picolm_gpu_ssm_recurrence_dev() deliberately never D2H's the
             * state back to s->ssm_state[il] (that round-trip is exactly
             * what the persistent-state optimization exists to avoid, see
             * "State remains on device - no D2H needed" in backend_gpu.cu).
             * s->ssm_state[il] (== `state` below) is therefore stale/
             * divergent from the moment the first token after prefill goes
             * through this path, and only gets further out of date with
             * every subsequent token.
             *
             * Previously, a failed recurrence_dev call here silently fell
             * back to picolm_gpu_ssm_recurrence(state, ...) or the CPU
             * tensor_parallel_for path -- both of which read that same
             * stale `state`. That computes ssm_output for this token from
             * the wrong starting state (corrupting this token's logits,
             * which then feeds back into generation as the next input
             * token), while leaving the real, correct device state
             * un-updated for this token -- a silent, undetectable
             * correctness bug. A transient failure here (e.g. gpuMalloc
             * contention) becomes far more likely under added system load
             * (such as running under `perf record`), which is consistent
             * with the early-stop/garbage-output reports.
             *
             * Retry once (transient GPU errors are typically one-off),
             * then fail loudly rather than silently corrupting output. */
            if (picolm_gpu_ssm_recurrence_dev(m->gpu.ssm_state_dev[il],
                                               q_conv, k_conv, v_conv,
                                               gate_exp, beta, ssm_output,
                                               n_v_heads, d_state, repeat, m->gpu.device)) {
                rec_done = 1;
            } else if (picolm_gpu_ssm_recurrence_dev(m->gpu.ssm_state_dev[il],
                                               q_conv, k_conv, v_conv,
                                               gate_exp, beta, ssm_output,
                                               n_v_heads, d_state, repeat, m->gpu.device)) {
                fprintf(stderr, "WARN: GPU SSM recurrence retry succeeded "
                        "(layer %d, pos %d) after an initial failure -- "
                        "investigate transient GPU errors.\n", il, pos);
                rec_done = 1;
            } else {
                fprintf(stderr, "FATAL: GPU SSM recurrence failed twice for "
                        "layer %d, pos %d with device-resident state present. "
                        "Falling back to CPU/host state here would silently "
                        "corrupt this and all subsequent tokens (device state "
                        "is the sole up-to-date copy and is not host-synced "
                        "per-token). Aborting instead of producing wrong "
                        "output. Check GPU memory pressure/errors.\n", il, pos);
                abort();
            }
        }
        if (!rec_done) {
            /* No device-resident state exists for this layer (e.g. GPU
             * disabled, or state never allocated) -- `state` (host) is the
             * authoritative copy in this case, so these paths are safe. */
            if (!picolm_gpu_ssm_recurrence(state, q_conv, k_conv, v_conv,
                                            gate_exp, beta, ssm_output,
                                            n_v_heads, d_state, repeat, m->gpu.device)) {
                tensor_parallel_for(n_v_heads, ssm_head_task, &ssm_ctx);
            }
        }
    } else
#endif
    {
        tensor_parallel_for(n_v_heads, ssm_head_task, &ssm_ctx);
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("ssm_out_pre[:8]", ssm_output, head_v_dim, 8);
#endif

    /* 18. Gated normalization */
    /* ssm_output: [d * n_v_heads + h] (dim-major from delta_net output) */
    float *norm_w = s->ssm_norm_w[il]; /* [head_v_dim] */
    float *final_output = ssm_output + d_state * n_v_heads; /* [head_v_dim * n_v_heads] */
    {
        int gn_done = 0;
#ifdef PICOLM_GPU
        /* Try GPU gated norm if GPU is active.
         * Note: we pass head_map=NULL so the kernel writes in sequential order.
         * The GGUF v-head remap (do_remap) is handled separately in step 19. */
        if (gpu_lw && m->gpu.active) {
            if (picolm_gpu_ssm_gated_norm(final_output, ssm_output, s->xb2,
                                           norm_w, NULL, head_v_dim, n_v_heads, eps,
                                           m->gpu.device)) {
                gn_done = 1;
            }
        }
#endif
        if (!gn_done) {
            /* CPU fallback */
            for (int h = 0; h < n_v_heads; h++) {
                float nrm = 0.0f;
                for (int d = 0; d < head_v_dim; d++) {
                    float v = ssm_output[d * n_v_heads + h];
                    nrm += v * v;
                }
                nrm = 1.0f / sqrtf(nrm / (float)head_v_dim + eps);
                for (int d = 0; d < head_v_dim; d++) {
                    float v = ssm_output[d * n_v_heads + h];
                    float zv = s->xb2[h * head_v_dim + d];
                    float silu_z = zv * (1.0f / (1.0f + expf(-zv)));
                    final_output[h * head_v_dim + d] = v * nrm * norm_w[d] * silu_z;
                }
            }
        }
    }
#ifdef DEBUG_SSM
    if (il == 0 || il == 8 || il == 16 || il == 32 || il == 48 || il == 60) {
        dbg_vec("xb2[:8]", s->xb2, 8, 8);
        dbg_vec("final_out[:8]", final_output, 8, 8);
    }
#endif

    /* 19. Reshape to [value_dim] and output projection */
    /* final_output is [head_v_dim * n_v_heads] = [value_dim] = [4096] */
    /* ssm_out: [n_embd, value_dim] - GGUF columns may be reordered */
    float *fo_gguf = NULL;
    if (do_remap) {
        fo_gguf = alloca(c->ssm_d_inner * sizeof(float));
        for (int h = 0; h < n_v_heads; h++) {
            int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
            memcpy(fo_gguf + gh * head_v_dim, final_output + h * head_v_dim, head_v_dim * sizeof(float));
        }
    }
    const float *ssm_out_src = do_remap ? fo_gguf : final_output;
    int ssm_out_gpu_done = 0;
#ifdef PICOLM_GPU
    if (gpu_lw) {
        gpu_layer_weights_t *gl = (gpu_layer_weights_t *)gpu_lw;
        size_t in_bytes = (size_t)c->ssm_d_inner * sizeof(float);
        size_t out_bytes = (size_t)dim * sizeof(float);
        if (gl->ssm_out &&
            ssm_out_dev_scratch_ensure(m->gpu.device, in_bytes, out_bytes) &&
            picolm_gpu_memcpy(g_ssm_mm_outin_dev, ssm_out_src, in_bytes, 1, m->gpu.device) &&
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ssm_out,
                                   g_ssm_mm_outres_dev, g_ssm_mm_outin_dev, 1, m->gpu.device, 0, 0) &&
            picolm_gpu_sync(m->gpu.device) &&
            picolm_gpu_memcpy(residual, g_ssm_mm_outres_dev, out_bytes, -1, m->gpu.device)) {
            ssm_out_gpu_done = 1;
        }
    }
#endif
    if (!ssm_out_gpu_done)
    {
        matmul(residual, ssm_out_src, lw->ssm_out, c->ssm_d_inner, dim, lw->type_ssm_out);
    }
    #ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("residual[:8]", residual, 8, 8);
#endif
    /* 20. Residual add */
    vec_add(x, residual, dim);

    /* 21. Post-attention norm + FFN (only if MLP weights exist for this layer) */
    if (c->has_moe) {
        /* MoE forward pass */
        rmsnorm(s->xb, x, s->post_attn_norm_w[il], dim, eps);
        moe_forward(m, s, s->xb, s->xb, lw);
        vec_add(x, s->xb, dim);
    } else if (lw->ffn_gate && lw->ffn_up && lw->ffn_down) {
        rmsnorm(s->xb, x, s->post_attn_norm_w[il], dim, eps);
#ifdef PICOLM_GPU
        /* Fused FFN on GPU: y = down(silu(gate(x)) * up(x)) in one command
         * buffer (3 dispatches -> 1); s->xb aliases in/out safely. On miss
         * fall through to the per-matmul CPU path. Matches model_forward. */
        if (gpu_lw) {
            gpu_layer_weights_t *gl = (gpu_layer_weights_t *)gpu_lw;
            int mlp_ok = 0;
            if (gl->ffn_gate && gl->ffn_up && gl->ffn_down) {
                mlp_ok = picolm_gpu_expert_mlp((picolm_gpu_tensor_t *)gl->ffn_gate,
                                      (picolm_gpu_tensor_t *)gl->ffn_up,
                                      (picolm_gpu_tensor_t *)gl->ffn_down,
                                      s->xb, s->xb, 1);
            }
            if (mlp_ok)
                goto ssm_ffn_done;
        }
        /* Defensive: the qkv/gate/ssm_out matmuls above now go through
         * matmul_dev directly and never touch tensor_set_gpu_tensor,
         * but clear it before the CPU-fallback matmul() calls below,
         * which read the global gpu_tensor with no paired set/clear. */
        if (gpu_lw) tensor_set_gpu_tensor(NULL, 0);
#endif
        matmul(s->hb, s->xb, lw->ffn_gate, dim, c->n_ffn, lw->type_ffn_gate);
        matmul(s->hb2, s->xb, lw->ffn_up, dim, c->n_ffn, lw->type_ffn_up);
        silu(s->hb, c->n_ffn);
        elemwise_mul(s->hb, s->hb, s->hb2, c->n_ffn);
        matmul(s->xb, s->hb, lw->ffn_down, c->n_ffn, dim, lw->type_ffn_down);
#ifdef PICOLM_GPU
ssm_ffn_done:
#endif
        vec_add(x, s->xb, dim);
    }
}

/* GPU-native SSM forward pass. Operates entirely on device-resident
 * pipeline buffers (pipe_x, pipe_xb, ssm_qkv_raw, ssm_conv_out, etc.).
 * Zero H2D/D2H. The residual add writes back into pipe_x, and the FFN
 * block writes into pipe_x via the attention-layer FFN pipeline buffers.
 *
 * Returns 1 on success, 0 if any prerequisite is missing (caller should
 * fall back to D2H + ssm_forward + H2D).
 */
#ifdef PICOLM_GPU
static int
ssm_forward_gpu(model_t *m, run_state_t *s, float *x, float *residual,
                layer_weights_t *lw, int il, int pos, void *gpu_lw, int device) {
    (void)x; (void)residual;
    gpu_layer_weights_t *gl = (gpu_layer_weights_t *)gpu_lw;
    model_config_t *c = &m->config;
    gpu_weights_t *gw = &m->gpu;
    int dim = c->n_embd;
    int d_conv = c->ssm_d_conv;
    int d_state = c->ssm_d_state;
    int n_k_heads = c->ssm_n_group;
    int n_v_heads = c->ssm_dt_rank;
    int conv_dim = 2 * d_state * n_k_heads + c->ssm_d_inner;
    int qk_dim = d_state * n_k_heads;
    int head_v_dim = c->ssm_d_inner / n_v_heads;
    int repeat = n_v_heads / n_k_heads;
    float eps = c->rms_norm_eps;

    /* Bug 3 fix: this GPU-native path has no MoE FFN implementation --
     * step 16 below only wires up dense ffn_gate/up/down. On a MoE layer
     * gl->ffn_gate would be NULL and the block would silently no-op,
     * dropping the FFN entirely. Bail before any state mutation. */
    if (c->has_moe) return 0;

    /* Prerequisites: all GPU tensors must be resident */
    if (!gl->attn_qkv || !gl->attn_gate_ssm || !gl->ssm_out || !gl->ssm_conv1d) return 0;
    if (!gw->ssm_alpha_dev[il] || !gw->ssm_beta_dev[il] || !gw->ssm_a_dev[il]) return 0;
    if (!gw->ssm_dt_dev[il] || !gw->ssm_norm_dev[il]) return 0;
    if (!gw->ssm_conv_state_dev[il] || !gw->ssm_state_dev[il]) return 0;

    /* Bug 3 fix (2026-08-11): this GPU-native path has no MoE FFN
     * implementation -- step 16 below only wires up dense ffn_gate/up/
     * down. On a MoE layer gl->ffn_gate would be NULL and that whole
     * block would silently no-op, dropping the FFN's contribution
     * entirely (not a numerical difference -- a missing computation).
     * Bail here, before anything below has mutated persistent state, so
     * the caller's CPU-hybrid ssm_forward() fallback (which does call
     * moe_forward()) is always safe to take. */
    if (c->has_moe) return 0;

    /* Pipeline buffers */
    float *pipe_x = picolm_gpu_pipe_x(device);
    float *pipe_xb = picolm_gpu_pipe_xb(device);
    float *pipe_ffn_norm = picolm_gpu_pipe_ffn_norm(device);
    float *pipe_gate = picolm_gpu_pipe_gate(device);
    float *pipe_up = picolm_gpu_pipe_up(device);
    float *ssm_qkv_raw = picolm_gpu_ssm_qkv_raw(device);
    float *ssm_conv_out = picolm_gpu_ssm_conv_out(device);
    float *ssm_xb2 = picolm_gpu_ssm_xb2(device);
    float *ssm_xb2_remap = picolm_gpu_ssm_xb2_remap(device);
    float *ssm_v_remap = picolm_gpu_ssm_v_remap(device);
    float *ssm_alpha_raw = picolm_gpu_ssm_alpha_raw(device);
    float *ssm_beta_raw = picolm_gpu_ssm_beta_raw(device);
    float *ssm_gate_exp = picolm_gpu_ssm_gate_exp(device);
    float *ssm_beta_d = picolm_gpu_ssm_beta(device);
    float *ssm_output = picolm_gpu_ssm_output(device);
    float *ssm_final_output = picolm_gpu_ssm_final_output(device);
    if (!pipe_x || !pipe_xb || !ssm_qkv_raw || !ssm_conv_out ||
        !ssm_xb2 || !ssm_xb2_remap || !ssm_alpha_raw || !ssm_beta_raw ||
        !ssm_gate_exp || !ssm_beta_d || !ssm_output || !ssm_final_output) return 0;

    int n_k = c->ssm_n_group;
    int n_vpk = n_v_heads / n_k;
    int half_vpk = n_vpk / 2;
    int do_remap = !m->from_safetensors && n_k > 0 && n_k < n_v_heads && half_vpk > 0;
    const int *head_map_dev = do_remap ? (const int *)gw->ssm_head_map_dev : NULL;
    if (do_remap && !head_map_dev) return 0;

    /* 1. RMSNorm: pipe_xb = rmsnorm(pipe_x, attn_norm) */
    if (!picolm_gpu_rmsnorm_dev(pipe_xb, pipe_x,
            (float *)gw->attn_norm_dev[il], dim, eps, device)) {
        fprintf(stderr,"WARN: ssm_forward_gpu l=%d p=%d failed at rmsnorm\n",il,pos); return 0; }

    /* 2. QKV projection: ssm_qkv_raw = attn_qkv @ pipe_xb */
    if (!picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_qkv,
            ssm_qkv_raw, pipe_xb, 1, device, 0, 0)) {
        fprintf(stderr,"WARN: ssm_forward_gpu l=%d p=%d failed at qkv_proj\n",il,pos); return 0; }

    /* 3. Z gate: ssm_xb2 = attn_gate_ssm @ pipe_xb */
    if (!picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_gate_ssm,
            ssm_xb2, pipe_xb, 1, device, 0, 0)) {
        fprintf(stderr,"WARN: ssm_forward_gpu l=%d p=%d failed at gate_proj\n",il,pos); return 0; }

    /* 3b. Head permute xb2 (GGUF v-head reorder) */
    float *xb2_src = ssm_xb2;
    if (do_remap) {
        if (!picolm_gpu_ssm_head_permute_dev(ssm_xb2_remap, ssm_xb2,
                head_map_dev, head_v_dim, n_v_heads, device)) {
            fprintf(stderr,"WARN: ssm_forward_gpu l=%d p=%d failed at xb2_permute\n",il,pos); return 0; }
        xb2_src = ssm_xb2_remap;
    }

    /* 4. Conv1d + silu + state shift: ssm_conv_out = conv1d(ssm_qkv_raw) */
    /* DEBUG: download pre-conv state for CPU reference comparison */
    float *dbg_pre_state = NULL;
    float *dbg_qkv_raw = NULL;
    float *dbg_conv_w = NULL;
    if (il == 0 && pos < 40 && _SSM_DBG) {
        dbg_pre_state = alloca((d_conv - 1) * conv_dim * sizeof(float));
        dbg_qkv_raw = alloca(conv_dim * sizeof(float));
        dbg_conv_w = alloca(d_conv * conv_dim * sizeof(float));
        picolm_gpu_sync(device);
        picolm_gpu_memcpy(dbg_pre_state, gw->ssm_conv_state_dev[il], (d_conv-1) * conv_dim * sizeof(float), -1, device);
        picolm_gpu_memcpy(dbg_qkv_raw, ssm_qkv_raw, conv_dim * sizeof(float), -1, device);
        picolm_gpu_memcpy(dbg_conv_w, gw->ssm_conv1d_dev[il], d_conv * conv_dim * sizeof(float), -1, device);
    }
    if (!picolm_gpu_ssm_conv1d_dev(ssm_conv_out,
            (float *)gw->ssm_conv_state_dev[il],
            ssm_qkv_raw,
            (float *)gw->ssm_conv1d_dev[il],
            conv_dim, d_conv, device)) {
        fprintf(stderr, "FATAL: GPU SSM conv1d failed for layer %d, pos %d "
                "-- aborting rather than risk a double conv-state shift via "
                "CPU fallback.\n", il, pos);
        abort();
    }

    /* DEBUG: compare GPU conv1d output with CPU reference */
    if (il == 0 && pos < 40 && _SSM_DBG) {
        picolm_gpu_sync(device);
        float *gpu_conv_out_buf = alloca(8 * sizeof(float));
        float *cpu_conv_out_buf = alloca(8 * sizeof(float));
        picolm_gpu_memcpy(gpu_conv_out_buf, ssm_conv_out, 8 * sizeof(float), -1, device);
        for (int co = 0; co < 8; co++) {
            float sum = 0.0f;
            for (int d = 0; d < d_conv - 1; d++)
                sum += dbg_conv_w[d + co * d_conv] * dbg_pre_state[d * conv_dim + co];
            sum += dbg_conv_w[(d_conv-1) + co * d_conv] * dbg_qkv_raw[co];
            cpu_conv_out_buf[co] = sum / (1.0f + expf(-sum));
        }
        fprintf(stderr, "SSMDBG l=%d p=%d conv1d GPU[:8]={", il, pos);
        for(int i=0;i<8;i++) fprintf(stderr,"%s%.6f",i?",":"",gpu_conv_out_buf[i]);
        fprintf(stderr, "} CPUref[:8]={");
        for(int i=0;i<8;i++) fprintf(stderr,"%s%.6f",i?",":"",cpu_conv_out_buf[i]);
        fprintf(stderr, "}\n");
        /* Dump state rows for channel 0 */
        fprintf(stderr, "SSMDBG state[:4][0]={");
        for(int d=0;d<d_conv-1;d++) fprintf(stderr,"%s%.6f",d?",":"",dbg_pre_state[d*conv_dim+0]);
        fprintf(stderr, "} qkv[0]=%.6f w[0][:4]={", dbg_qkv_raw[0]);
        for(int d=0;d<d_conv;d++) fprintf(stderr,"%s%.6f",d?",":"",dbg_conv_w[d]);
        fprintf(stderr, "}\n");
    }

    /* 5. Split Q/K/V by pointer offset (no copy) */
    float *q_dev = ssm_conv_out;
    float *k_dev = ssm_conv_out + qk_dim;
    float *v_dev = ssm_conv_out + 2 * qk_dim;

    /* 6. L2 normalize Q in-place, then scale by 1/sqrt(d_state).
     * 7. L2 normalize K in-place -- NOT scaled.
     * CORRECTED: the CPU reference's qk_dim is d_state * n_k_heads,
     * the size of ONE of Q/K (q_conv and k_conv are separate arrays).
     * Its step-8 loop `for (i < qk_dim) q_conv[i] *= q_scale` therefore
     * touches only q_conv -- k_conv is never scaled. The previous version
     * scaled k_dev by q_scale too, based on a misreading that qk_dim
     * covered both; that shrinks every K vector by 1/sqrt(d_state)
     * before the recurrence, on every token of every layer.
     * Fixed: K gets extra_scale=1.0 (identity, normalize only). */
    float q_scale = 1.0f / sqrtf((float)d_state);
    if (!picolm_gpu_ssm_l2norm_dev(q_dev, d_state, n_k_heads, 1e-12f, q_scale, device) ||
        !picolm_gpu_ssm_l2norm_dev(k_dev, d_state, n_k_heads, 1e-12f, 1.0f, device)) {
        fprintf(stderr, "FATAL: GPU SSM L2 norm failed for layer %d, pos %d.\n", il, pos);
        abort();
    }

    /* 8. Head permute V (if GGUF reorders) */
    if (do_remap) {
        if (!picolm_gpu_ssm_head_permute_dev(ssm_v_remap, v_dev,
                head_map_dev, head_v_dim, n_v_heads, device)) {
            fprintf(stderr, "FATAL: GPU SSM V head-permute failed for layer %d, pos %d.\n", il, pos);
            abort();
        }
        v_dev = ssm_v_remap;
    }

    /* 9. Alpha vecdot -- type already validated at top of function */
    {
        gguf_type_t alpha_type = lw->type_ssm_alpha;
        size_t row_bytes = gguf_type_row_size(alpha_type, dim);
        if (!picolm_gpu_ssm_vecdot_dev(ssm_alpha_raw, pipe_xb,
                gw->ssm_alpha_dev[il], alpha_type, dim,
                n_v_heads, (int)row_bytes, head_map_dev, device)) {
            fprintf(stderr, "FATAL: GPU SSM alpha vecdot failed for layer %d, pos %d.\n", il, pos);
            abort();
        }
    }

    /* 10. Beta vecdot -- type already validated at top of function */
    {
        gguf_type_t beta_type = lw->type_ssm_beta;
        size_t row_bytes = gguf_type_row_size(beta_type, dim);
        if (!picolm_gpu_ssm_vecdot_dev(ssm_beta_raw, pipe_xb,
                gw->ssm_beta_dev[il], beta_type, dim,
                n_v_heads, (int)row_bytes, head_map_dev, device)) {
            fprintf(stderr, "FATAL: GPU SSM beta vecdot failed for layer %d, pos %d.\n", il, pos);
            abort();
        }
    }

    /* 11. Softplus(alpha)+gate + sigmoid(beta) */
    if (!picolm_gpu_ssm_gate_beta_dev(ssm_gate_exp, ssm_beta_d,
            ssm_alpha_raw, ssm_beta_raw,
            (float *)gw->ssm_a_dev[il],
            (float *)gw->ssm_dt_dev[il],
            n_v_heads, device)) {
        fprintf(stderr, "FATAL: GPU SSM gate/beta activation failed for layer %d, pos %d.\n", il, pos);
        abort();
    }

    /* 12. SSM recurrence (thread-0 kernel by default -- bit-exact with
     * CPU NEON; warp-shuffle only if PICOLM_SSM_WARP_KERNEL_VALIDATED).
     * This mutates gw->ssm_state_dev[il], persistent per-token state. */
    if (!picolm_gpu_ssm_recurrence_pipeline_dev(gw->ssm_state_dev[il],
            q_dev, k_dev, v_dev,
            ssm_gate_exp, ssm_beta_d,
            ssm_output, n_v_heads, d_state, repeat, device)) {
        fprintf(stderr, "FATAL: GPU SSM recurrence failed for layer %d, pos %d "
                "with device-resident state already mutated -- "
                "aborting rather than risk corrupting generation.\n", il, pos);
        abort();
    }

    /* DEBUG: dump recurrence output (dim-major ssm_output) for comparison */
    if (il == 0 && pos < 40 && _SSM_DBG) {
        picolm_gpu_sync(device);
        float *dbg_ssm_out = alloca(8 * sizeof(float));
        picolm_gpu_memcpy(dbg_ssm_out, ssm_output, 8 * sizeof(float), -1, device);
        fprintf(stderr, "SSMDBG l=%d p=%d ssm_out_pre[:8]={", il, pos);
        for(int i=0;i<8;i++) fprintf(stderr,"%s%.6f",i?",":"",dbg_ssm_out[i]);
        fprintf(stderr, "}\n");
    }

    /* 13. Gated normalization (fuses RMSNorm of dim-major ssm_output
     *     with silu(xb2) gating, writes head-major with GGUF reorder) */
    if (!picolm_gpu_ssm_gated_norm_dev(ssm_final_output, ssm_output, xb2_src,
            (float *)gw->ssm_norm_dev[il],
            head_map_dev,
            head_v_dim, n_v_heads, eps, device)) {
        fprintf(stderr, "FATAL: GPU SSM gated norm failed for layer %d, pos %d.\n", il, pos);
        abort();
    }

    /* 14. SSM output projection: pipe_xb = ssm_out @ ssm_final_output */
    if (!picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ssm_out,
            pipe_xb, ssm_final_output, 1, device, 0, 0)) {
        fprintf(stderr, "FATAL: GPU SSM output projection failed for layer %d, pos %d.\n", il, pos);
        abort();
    }

    /* DEBUG: run CPU ssm_forward in parallel to compare intermediate values */
    if (il == 0 && pos < 38 && _SSM_DBG) {
        picolm_gpu_sync(device);
        /* Download GPU intermediates */
        float *gpu_final = alloca(c->ssm_d_inner * sizeof(float));
        float *gpu_residual = alloca(dim * sizeof(float));
        float *gpu_pipe_x = alloca(dim * sizeof(float));
        float *gpu_xb2 = alloca(c->ssm_d_inner * sizeof(float));
        picolm_gpu_memcpy(gpu_final, ssm_final_output, c->ssm_d_inner * sizeof(float), -1, device);
        picolm_gpu_memcpy(gpu_residual, pipe_xb, dim * sizeof(float), -1, device);
        picolm_gpu_memcpy(gpu_pipe_x, pipe_x, dim * sizeof(float), -1, device);
        picolm_gpu_memcpy(gpu_xb2, xb2_src, c->ssm_d_inner * sizeof(float), -1, device);

        fprintf(stderr, "SSMDBG l=%d p=%d GPU ssm_final[:4]={", il, pos);
        for(int i=0;i<4;i++) fprintf(stderr,"%s%.6f",i?",":"",gpu_final[i]);
        fprintf(stderr, "} GPU res[:4]={");
        for(int i=0;i<4;i++) fprintf(stderr,"%s%.6f",i?",":"",gpu_residual[i]);
        fprintf(stderr, "} GPU xb2[:4]={");
        for(int i=0;i<4;i++) fprintf(stderr,"%s%.6f",i?",":"",gpu_xb2[i]);
        fprintf(stderr, "}\n");

        /* Compare with CPU: run CPU ssm_forward on a copy of pipe_x */
        float *cpu_x = alloca(dim * sizeof(float));
        float *cpu_res = alloca(dim * sizeof(float));
        memcpy(cpu_x, gpu_pipe_x, dim * sizeof(float));
        /* We can't easily call ssm_forward here because it modifies shared state.
         * Instead, compare the ssm_out matmul input (ssm_final_output) manually. */
    }

    /* 15. Residual add: pipe_x += pipe_xb */
    if (!picolm_gpu_residual_add(pipe_x, pipe_x, pipe_xb, 1, dim, dim, device)) {
        fprintf(stderr, "FATAL: GPU SSM residual add failed for layer %d, pos %d.\n", il, pos);
        abort();
    }

    /* FFN handled by caller (model_forward_gpu) via the !did_cpu_ssm block,
     * which uses fully device-native picolm_gpu_matmul_dev calls.
     * picolm_gpu_expert_mlp expects host pointers and would corrupt data
     * if called with device-resident pipe_x. */
    return 1;
}
#endif

/* ================================================================
 * Device-native batched SSM prefill layer (zero H2D/D2H)
 * ================================================================ */
#define SSM_DBG_SYNC /* disabled */
#ifdef PICOLM_GPU
int ssm_prefill_layer_gpu(model_t *m, run_state_t *s,
    float *bx, float *bxb, float *bq, float *battn_out, float *bffn_norm,
    float *bgate, float *bup, layer_weights_t *lw, int l,
    int n_tokens, int start_pos, int dev) {
#ifdef PICOLM_GPU
    (void)start_pos;
    model_config_t *c = &m->config;
    gpu_weights_t *gw = &m->gpu;
    int dim=c->n_embd, d_state=c->ssm_d_state;
    int n_k=c->ssm_n_group, n_v=c->ssm_dt_rank;
    int conv_dim=2*d_state*n_k+c->ssm_d_inner;
    int value_dim=c->ssm_d_inner;
    int hvdim=value_dim/n_v;
    float eps=c->rms_norm_eps;
    int repeat=n_v/n_k, qk_dim=d_state*n_k;
    gpu_layer_weights_t *gl = &m->gpu.layers[l];
    float *cs_dev=(float*)gw->ssm_conv_state_dev[l];
    float *st_dev=(float*)gw->ssm_state_dev[l];
    if(!cs_dev||!st_dev||!gl->ssm_conv1d||!gl->attn_qkv||!gl->attn_gate_ssm||!gl->ssm_out) return 0;
    /* do_remap: must match CPU reference (ssm_prefill_layer, ssm_forward, ssm_forward_gpu) */
    int n_vpk = n_k > 0 ? n_v / n_k : 0, half_vpk = n_vpk / 2;
    int do_remap = !m->from_safetensors && n_k > 0 && n_k < n_v && half_vpk > 0;
    int ok=1;
    #define _SSM_OK_CHECK(step) do { if(!ok) { fprintf(stderr,"WARN: ssm_prefill_layer_gpu l=%d failed at %s\n",l,step); return 0; } } while(0)
    /* Pipe buffer stride: must match pipeline_batch_alloc's xb_stride calculation */
    int xb2_stride = c->n_heads * c->head_dim * (c->has_ssm ? 2 : 1);
    if (dim > xb2_stride) xb2_stride = dim;
    int ssm_conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    if (ssm_conv_dim > xb2_stride) xb2_stride = ssm_conv_dim;
    /* bffn_norm temporarily holds the RMSNorm'd input until vecdot consumes it.
     * bxb becomes the Q/K/V-conv working buffer immediately after. */

    /* Per-step diagnostic: dump last token RMS to compare GPU vs CPU */
    #define _DBG_RMS(name, ptr, stride) do { \
        if (l == 0 && _SSM_DBG) { \
            float _rms=0; float _v[8]; \
            picolm_gpu_memcpy(_v, ptr + (size_t)(n_tokens-1) * stride, 32, -1, dev); \
            for (int _i=0;_i<8;_i++) _rms += _v[_i]*_v[_i]; \
            _rms = sqrtf(_rms/8); \
            fprintf(stderr, "[DBG l%d] %s rms_last=%.6f {%.6f %.6f %.6f %.6f}\n", l, name, _rms, _v[0],_v[1],_v[2],_v[3]); \
        } \
    } while(0)
    /* Canonical xb2_stride-aware per-token dump: reads 4 floats per token, reports RMS */
    #define _DBG_TOK(name, ptr) do { \
        if (l == 0 && _SSM_DBG) { \
            float _vt[40]; \
            for (int _ti=0;_ti<10;_ti++) \
                picolm_gpu_memcpy(_vt+_ti*4, ptr + (size_t)_ti * xb2_stride, 16, -1, dev); \
            float _trms=0; \
            for (int _ti=0;_ti<10;_ti++) { float _r=0; for(int _j=0;_j<4;_j++) _r+=_vt[_ti*4+_j]*_vt[_ti*4+_j]; _r=sqrtf(_r/4); _trms+=_r*_r; } \
            _trms=sqrtf(_trms/10); \
            fprintf(stderr, "[DBG l%d] %s tok0={%.6f %.6f %.6f %.6f} tok9={%.6f %.6f %.6f %.6f} mean_rms=%.6f\n", \
                l, name, _vt[0],_vt[1],_vt[2],_vt[3], _vt[36],_vt[37],_vt[38],_vt[39], _trms); \
        } \
    } while(0)

    ok&=picolm_gpu_rmsnorm_batched_dev(bffn_norm,bx,(float*)gw->attn_norm_dev[l],dim,eps,n_tokens,xb2_stride,dev);
    _SSM_OK_CHECK("rmsnorm");
    SSM_DBG_SYNC;
    _DBG_RMS("rmsnorm",bffn_norm,xb2_stride);
    /* QKV + Z-gate projections (read from RMSNorm'd input at xb2_stride) */
    ok&=picolm_gpu_matmul_dev((picolm_gpu_tensor_t*)gl->attn_qkv,bxb,bffn_norm,n_tokens,dev,xb2_stride,xb2_stride);
    _SSM_OK_CHECK("qkv_matmul");
    SSM_DBG_SYNC;
    _DBG_RMS("qkv",bxb,xb2_stride);
    ok&=picolm_gpu_matmul_dev((picolm_gpu_tensor_t*)gl->attn_gate_ssm,bq,bffn_norm,n_tokens,dev,xb2_stride,xb2_stride);
    _SSM_OK_CHECK("gate_matmul");
    SSM_DBG_SYNC;
    _DBG_RMS("gate",bq,xb2_stride);
    if(l<=4&&_SSM_DBG) {
        /* Dump FFN gate matmul output for comparison */
        { float fi[4];
          picolm_gpu_sync(dev);
          picolm_gpu_memcpy(fi, bffn_norm, 16, -1, dev);
          fprintf(stderr,"[DBG] l=%d bffn_norm_tok0[:4]={%.6f,%.6f,%.6f,%.6f}\n",l,fi[0],fi[1],fi[2],fi[3]);
        }
    }
    /* Dump conv1d weight for debugging */
    if (l == 0 && _SSM_DBG) {
        float cw[16], cs[12];
        picolm_gpu_memcpy(cw, gw->ssm_conv1d_dev[l], 64, -1, dev);
        fprintf(stderr, "[DBG l%d] conv1d_w[0][:4]={%.6f,%.6f,%.6f,%.6f} w[1][:4]={%.6f,%.6f,%.6f,%.6f}\n", l, cw[0],cw[1],cw[2],cw[3],cw[4],cw[5],cw[6],cw[7]);
        picolm_gpu_memcpy(cs, cs_dev, 48, -1, dev);
        fprintf(stderr, "[DBG l%d] conv_state[0][:4]={%.6f,%.6f,%.6f,%.6f}\n", l, cs[0],cs[1],cs[2],cs[3]);
    }
    /* Conv1d + silu (in-place on bxb which has QKV output) */
        ok&=picolm_gpu_ssm_conv1d_batch_dev(bxb,cs_dev,bxb,(float*)gw->ssm_conv1d_dev[l],conv_dim,c->ssm_d_conv,n_tokens,dev,xb2_stride);
    _SSM_OK_CHECK("conv1d");
    SSM_DBG_SYNC;
    _DBG_RMS("conv1d",bxb,xb2_stride);
    /* L2 norm Q/K */
    _DBG_RMS("pre_l2_q",bxb,xb2_stride);
    if(ok){float qs=1.0f/sqrtf((float)d_state);
            ok&=picolm_gpu_ssm_l2norm_batch_dev(bxb,d_state,n_k,n_tokens,xb2_stride,1e-12f,qs,dev);
      _SSM_OK_CHECK("l2norm_q");
    SSM_DBG_SYNC;
    _DBG_RMS("l2norm_q",bxb,xb2_stride);
            ok&=picolm_gpu_ssm_l2norm_batch_dev(bxb+qk_dim,d_state,n_k,n_tokens,xb2_stride,1e-12f,1.0f,dev);
      _SSM_OK_CHECK("l2norm_k");
    _DBG_RMS("l2norm_k",bxb+qk_dim,xb2_stride);
      _DBG_TOK("Q",bxb);}
    /* Head permute V. bxb+2*qk_dim can't be permuted into itself --
     * that races whenever head_map isn't the identity (concurrent
     * threadblocks, no ordering guarantee, one can overwrite a
     * position another hasn't read yet). Snapshot it out unaliased
     * first (identity copy, head_map=NULL), then permute from that
     * snapshot directly into its real strided destination -- both
     * steps are the same primitive, so this is "permute" used twice
     * (once as a plain copy) rather than a permute paired with an
     * unrelated generic copy kernel. */
    if(ok&&do_remap&&gw->ssm_head_map_dev){
      ok&=picolm_gpu_ssm_head_permute_batch_dev(battn_out,bxb+2*qk_dim,NULL,hvdim,n_v,n_tokens,xb2_stride,value_dim,dev);
    SSM_DBG_SYNC;
      ok&=picolm_gpu_ssm_head_permute_batch_dev(bxb+2*qk_dim,battn_out,(const int*)gw->ssm_head_map_dev,hvdim,n_v,n_tokens,value_dim,xb2_stride,dev);
      /* Z-gate (bq) permute: GGUF order -> natural order, same two-step trick */
    SSM_DBG_SYNC;
      ok&=picolm_gpu_ssm_head_permute_batch_dev(battn_out,bq,NULL,hvdim,n_v,n_tokens,xb2_stride,value_dim,dev);
    SSM_DBG_SYNC;
      ok&=picolm_gpu_ssm_head_permute_batch_dev(bq,battn_out,(const int*)gw->ssm_head_map_dev,hvdim,n_v,n_tokens,value_dim,value_dim,dev);}
    SSM_DBG_SYNC;
      _DBG_TOK("V",bxb+2*qk_dim);
    /* Alpha/beta vecdot (read from RMSNorm'd bffn_norm) */
    if(ok){gguf_type_t at=lw->type_ssm_alpha,bt=lw->type_ssm_beta;
      size_t ra=gguf_type_row_size(at,dim),rb=gguf_type_row_size(bt,dim);
      if(at!=0&&at!=2&&at!=8&&at!=30){static int w1=0;if(!w1){fprintf(stderr,"WARN: ssm_prefill_layer_gpu bail: alpha type=%d (need F32/Q4_0/Q8_0/BF16)\n",(int)at);w1=1;}return 0;}
      if(bt!=0&&bt!=2&&bt!=8&&bt!=30){static int w2=0;if(!w2){fprintf(stderr,"WARN: ssm_prefill_layer_gpu bail: beta type=%d (need F32/Q4_0/Q8_0/BF16)\n",(int)bt);w2=1;}return 0;}
      const int *hm=do_remap&&gw->ssm_head_map_dev?(const int*)gw->ssm_head_map_dev:NULL;
      ok&=picolm_gpu_ssm_vecdot_batch_dev(bgate,bffn_norm,(void*)picolm_gpu_tensor_weights((picolm_gpu_tensor_t*)gl->ssm_alpha),at,dim,n_v,n_tokens,(int)ra,hm,dev,xb2_stride,xb2_stride);
      _SSM_OK_CHECK("vecdot_alpha");
    SSM_DBG_SYNC;
      ok&=picolm_gpu_ssm_vecdot_batch_dev(bup,bffn_norm,(void*)picolm_gpu_tensor_weights((picolm_gpu_tensor_t*)gl->ssm_beta),bt,dim,n_v,n_tokens,(int)rb,hm,dev,xb2_stride,xb2_stride);
      _SSM_OK_CHECK("vecdot_beta");
    }
    SSM_DBG_SYNC;
#ifdef PICOLM_SSM_VERIFY
    if(ok && l == 0) {
        picolm_gpu_sync(dev);
        float vtmp[8];
        picolm_gpu_memcpy(vtmp, bgate, sizeof(vtmp), -1, dev);
        fprintf(stderr, "SSM_VERIFY l=0 gpu alpha_raw[:4]={%.6f,%.6f,%.6f,%.6f}\n",
            vtmp[0],vtmp[1],vtmp[2],vtmp[3]);
        picolm_gpu_memcpy(vtmp, bup, sizeof(vtmp), -1, dev);
        fprintf(stderr, "SSM_VERIFY l=0 gpu beta_raw[:4]={%.6f,%.6f,%.6f,%.6f}\n",
            vtmp[0],vtmp[1],vtmp[2],vtmp[3]);
    }
#endif
    /* Gate/beta post-process */
    if(ok){float *dw=(float*)gw->ssm_dt_dev[l],*aw=(float*)gw->ssm_a_dev[l];
      if(!dw||!aw)return 0;
      ok&=picolm_gpu_ssm_gate_beta_batch_dev(bgate,bup,bgate,bup,dw,aw,n_v,n_tokens,dev,xb2_stride);
      _SSM_OK_CHECK("gate_beta");
    }
    SSM_DBG_SYNC;
    /* Chunked recurrence */
    /* Step-by-step verify: D2H intermediates after each GPU kernel, compare with CPU */
    if(ok && _SSM_DBG && l <= 4) {
        picolm_gpu_sync(dev);
        int cs = 64; if(cs > n_tokens) cs = n_tokens;
        int d = d_state;
        size_t cb = (size_t)n_tokens * conv_dim * sizeof(float);
        size_t ab = (size_t)n_tokens * n_v * sizeof(float);
        float *ch = (float*)malloc(cb), *ah = (float*)malloc(ab), *bh = (float*)malloc(ab);
        float *sh = (float*)calloc(n_v * d * d, sizeof(float));
        /* D2H conv_batch (Q+K+V), alpha, beta */
        for (int _t = 0; _t < n_tokens; _t++) {
            picolm_gpu_memcpy(ch + _t * conv_dim, bxb + (size_t)_t * xb2_stride, conv_dim * sizeof(float), -1, dev);
            picolm_gpu_memcpy(ah + _t * n_v, bgate + (size_t)_t * xb2_stride, n_v * sizeof(float), -1, dev);
            picolm_gpu_memcpy(bh + _t * n_v, bup + (size_t)_t * xb2_stride, n_v * sizeof(float), -1, dev);
        }
        /* D2H state */
        float *st_h = (float*)calloc(n_v * d * d, sizeof(float));
        picolm_gpu_memcpy(st_h, st_dev, n_v * d * d * sizeof(float), -1, dev);
        /* Run CPU recurrence to get intermediates */
        /* We'll manually replicate the chunked steps on CPU and compare with GPU D2H dumps */
        int ca = cs; int co2 = 0;
        /* === DECAY KERNEL === */
        /* D2H d_gl (gate_log), compute CPU decay_mask, D2H GPU decay_mask */
        {
            float *d_gl_dev = (float*)malloc(n_v * ca * sizeof(float));
            float *d_gl_cpu = (float*)calloc(n_v * ca, sizeof(float));
            float *d_dm_dev = (float*)malloc(n_v * ca * ca * sizeof(float));
            float *d_dm_cpu = (float*)calloc(n_v * ca * ca, sizeof(float));
            /* gate_log is ah (alpha_batch) after gate/beta post-process */
            /* Copy gate_log from ah[co2..co2+ca] for all heads */
            for (int h = 0; h < n_v; h++)
                memcpy(d_gl_cpu + h * ca, ah + (co2 * n_v + h), ca * sizeof(float));
            /* Compute CPU decay: cum_g, q_decay, decay_mask */
            for (int h = 0; h < n_v; h++) {
                float cg = 0.0f;
                for (int t = 0; t < ca; t++) {
                    cg += d_gl_cpu[h * ca + t];
                    if (cg > 50) cg = 50;
                }
                float *dm = d_dm_cpu + h * ca * ca;
                for (int t1 = 0; t1 < ca; t1++) {
                    float g = 0.0f;
                    for (int t2 = 0; t2 <= t1; t2++) {
                        g += d_gl_cpu[h * ca + t2];
                        dm[t1 * ca + t2] = expf(-g);
                    }
                }
            }
            fprintf(stderr, "[STEP l0] decay_mask CPU dm[0][0..3]={%.6f,%.6f,%.6f,%.6f} dm[ca-1][0..3]={%.6f,%.6f,%.6f,%.6f}\n",
                d_dm_cpu[0], d_dm_cpu[1], d_dm_cpu[2], d_dm_cpu[3],
                d_dm_cpu[(ca-1)*ca], d_dm_cpu[(ca-1)*ca+1], d_dm_cpu[(ca-1)*ca+2], d_dm_cpu[(ca-1)*ca+3]);
            free(d_gl_dev); free(d_gl_cpu); free(d_dm_dev); free(d_dm_cpu);
        }
        free(ch); free(ah); free(bh); free(sh); free(st_h);
    }
    if(ok){
        ok&=picolm_gpu_ssm_chunked_recurrence_dev(bxb,bgate,bup,st_dev,battn_out,n_tokens,value_dim,xb2_stride,d_state,n_k,n_v,hvdim,repeat,conv_dim,64,dev);
        _SSM_OK_CHECK("chunked_recurrence");
    SSM_DBG_SYNC;
        _DBG_TOK("xb2",battn_out);
        if(ok && _SSM_DBG && l <= 4) {
            /* D2H GPU recurrence output and compare with CPU reference */
            size_t xb2_bytes=(size_t)n_tokens*value_dim*sizeof(float);
            float *xb2_gpu=(float*)malloc(xb2_bytes);
            float *xb2_cpu=(float*)calloc(n_tokens*value_dim,sizeof(float));
            if(xb2_gpu&&xb2_cpu){
                picolm_gpu_sync(dev);
                /* Copy xb2 with proper xb2_stride (battn_out is strided) */
                for (int _t = 0; _t < n_tokens; _t++)
                    picolm_gpu_memcpy(xb2_gpu + _t * value_dim, battn_out + (size_t)_t * xb2_stride, value_dim * sizeof(float), -1, dev);
                /* Run CPU recurrence on same inputs */
                size_t cb=(size_t)n_tokens*conv_dim*sizeof(float);
                size_t ab=(size_t)n_tokens*n_v*sizeof(float);
                float *ch=(float*)malloc(cb),*ah=(float*)malloc(ab),*bh=(float*)malloc(ab);
                if(ch&&ah&&bh){
                    /* Copy conv_batch with proper xb2_stride (not contiguous) */
                    for (int _t = 0; _t < n_tokens; _t++) {
                        picolm_gpu_memcpy(ch + _t * conv_dim, bxb + (size_t)_t * xb2_stride, conv_dim * sizeof(float), -1, dev);
                        picolm_gpu_memcpy(ah + _t * n_v, bgate + (size_t)_t * xb2_stride, n_v * sizeof(float), -1, dev);
                        picolm_gpu_memcpy(bh + _t * n_v, bup + (size_t)_t * xb2_stride, n_v * sizeof(float), -1, dev);
                    }
                    fprintf(stderr, "[DBG] l0 gate_log D2H h0: t0=%.6f t1=%.6f t5=%.6f t10=%.6f t20=%.6f t35=%.6f\n",
                        ah[0], ah[n_v], ah[5*n_v], ah[10*n_v], ah[20*n_v], ah[35*n_v]);
                    float *sh=(float*)calloc(n_v*d_state*d_state,sizeof(float));
                    ssm_chunked_recurrence(ch,ah,bh,sh,xb2_cpu,n_tokens,value_dim,d_state,n_k,n_v,hvdim,repeat,conv_dim,m->ssm_chunk_size);
                    free(sh);
                    fprintf(stderr, "[DBG] l0 CPU xb2 tok0={%.6f %.6f %.6f %.6f} tok9={%.6f %.6f %.6f %.6f}\n",
                        xb2_cpu[0],xb2_cpu[1],xb2_cpu[2],xb2_cpu[3],
                        xb2_cpu[9*value_dim],xb2_cpu[9*value_dim+1],xb2_cpu[9*value_dim+2],xb2_cpu[9*value_dim+3]);
                    /* Compare */
                    float maxd=0;
                    for(int i=0;i<n_tokens*value_dim;i++){float d=xb2_gpu[i]-xb2_cpu[i];if(d<0)d=-d;if(d>maxd)maxd=d;}
                    double gr=0,cr=0;
                    for(int i=0;i<n_tokens*value_dim;i++){gr+=xb2_gpu[i]*xb2_gpu[i];cr+=xb2_cpu[i]*xb2_cpu[i];}
                    fprintf(stderr,"[DBG] l=0 GPU vs CPU rec: max_diff=%.6e gpu_rms=%.6e cpu_rms=%.6e\n",
                        maxd, sqrt(gr/(n_tokens*value_dim)), sqrt(cr/(n_tokens*value_dim)));
                }
                free(ch);free(ah);free(bh);
            }
            free(xb2_gpu);free(xb2_cpu);
        }
    }
    /* COMPARISON: D2H last token of xb2 (recurrence output) with proper xb2_stride */
    if(ok && l <= 4 && _SSM_DBG) {
        picolm_gpu_sync(dev);
        int lt = n_tokens - 1;
        float *xb2_last = (float *)malloc(value_dim * sizeof(float));
        picolm_gpu_memcpy(xb2_last, battn_out + (size_t)lt * xb2_stride, value_dim * sizeof(float), -1, dev);
        double xrms=0; for(int i=0;i<value_dim;i++) xrms+=xb2_last[i]*xb2_last[i]; xrms=sqrt(xrms/value_dim);
        fprintf(stderr,"[CMP l%d] xb2_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms=%.6f\n",
            l, xb2_last[0],xb2_last[1],xb2_last[2],xb2_last[3],xb2_last[4],xb2_last[5],xb2_last[6],xb2_last[7], (float)xrms);
        free(xb2_last);
    }
    if(l <= 4 && _SSM_DBG) {
        /* xb2 per-head dump before gated_norm */
        { size_t lt2=(size_t)(n_tokens-1)*xb2_stride;
            float *a2=malloc(value_dim*sizeof(float));
            for(int hh=0;hh<n_v;hh++){
                float *seg=a2+hh*hvdim;
                picolm_gpu_memcpy(seg,battn_out+lt2+hh*hvdim,hvdim*sizeof(float),-1,dev);
                double h_rms=0; for(int dd=0;dd<hvdim;dd++) h_rms+=seg[dd]*seg[dd];
                if(hh<4||hh>=n_v-2||sqrt(h_rms/hvdim)>0.001){
                    fprintf(stderr,"[CMP l%d] xb2_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                        l, hh,(float)sqrt(h_rms/hvdim),seg[0],seg[1],seg[2],seg[3]);
                }
            }
            free(a2);
        }
    }
    if(ok){float *nw=(float*)gw->ssm_norm_dev[l];
      if(!nw)return 0;
      if(l<=4&&_SSM_DBG){
          float *nwf=(float*)gw->ssm_norm_dev[0];
          float *nwa=malloc(hvdim*sizeof(float));picolm_gpu_memcpy(nwa,nwf,(size_t)hvdim*sizeof(float),-1,dev);
          double nr=0;for(int i=0;i<hvdim;i++)nr+=nwa[i]*nwa[i];
          fprintf(stderr,"[CMP l%d] ssm_norm_w rms=%.6f first8={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f}\n",
              l, (float)sqrt(nr/hvdim),nwa[0],nwa[1],nwa[2],nwa[3],nwa[4],nwa[5],nwa[6],nwa[7]);
          free(nwa);
      }
      ok&=picolm_gpu_ssm_prefill_gated_norm_dev(battn_out,bq,nw,hvdim,n_v,n_tokens,eps,xb2_stride,value_dim,dev);
      _SSM_OK_CHECK("gated_norm");
    }
    if(l<=4&&_SSM_DBG){
        /* Z-gate (bq) per-head dump (bq has value_dim stride from head_permute) */
        { size_t lt2=(size_t)(n_tokens-1)*value_dim;
            float *a2=malloc(value_dim*sizeof(float));
            for(int hh=0;hh<n_v;hh++){
                float *seg=a2+hh*hvdim;
                picolm_gpu_memcpy(seg,bq+lt2+hh*hvdim,hvdim*sizeof(float),-1,dev);
                double h_rms=0;float hmax=0; for(int dd=0;dd<hvdim;dd++){h_rms+=seg[dd]*seg[dd];float av=seg[dd];if(av<0)av=-av;if(av>hmax)hmax=av;}
                if(hh<4||hh>=n_v-2||sqrt(h_rms/hvdim)>0.01){
                    fprintf(stderr,"[CMP l%d] z_h%d rms=%.6f maxabs=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                        l, hh,(float)sqrt(h_rms/hvdim),hmax,seg[0],seg[1],seg[2],seg[3]);
                }
            }
            free(a2);
        }
    }
    if(ok) ok&=picolm_gpu_sync(dev);
    SSM_DBG_SYNC;
      _DBG_TOK("gn_out",battn_out);
    if(l <= 4 && _SSM_DBG) {
        picolm_gpu_sync(dev);
        float gb[8];picolm_gpu_memcpy(gb,battn_out+(size_t)(n_tokens-1)*xb2_stride,32,-1,dev);
        double gr=0;for(int _i=0;_i<8;_i++)gr+=gb[_i]*gb[_i];
        fprintf(stderr,"[CMP l%d] gn_out_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,gb[0],gb[1],gb[2],gb[3],gb[4],gb[5],gb[6],gb[7],sqrt(gr/8));
    }
    /* Output projection: reuse bffn_norm as temp, then residual add into bx */
    if(ok){
      /* Head-permute battn_out to GGUF column order before ssm_out matmul.
       * CPU does: fo_gguf[gh] = xb2_natural[h]  (scatter, forward map)
       * GPU kernel does: dst[h] = src[map[h]]  (gather)
       * To match: need dst[h] = src[inv_gguf[h]] so that dst[gh] = src[h].
       * i.e. use the INVERSE map so the gather reproduces the scatter. */
      if(do_remap && gw->ssm_head_invmap_dev)
        ok&=picolm_gpu_ssm_head_permute_batch_dev(bxb,battn_out,(const int*)gw->ssm_head_invmap_dev,hvdim,n_v,n_tokens,xb2_stride,value_dim,dev);
      else if(do_remap && gw->ssm_head_map_dev)
        ok&=picolm_gpu_ssm_head_permute_batch_dev(bxb,battn_out,(const int*)gw->ssm_head_map_dev,hvdim,n_v,n_tokens,xb2_stride,value_dim,dev);
      else
        ok&=picolm_gpu_ssm_head_permute_batch_dev(bxb,battn_out,NULL,hvdim,n_v,n_tokens,xb2_stride,value_dim,dev);
      SSM_DBG_SYNC;
      if(!ok) return 0;
      /* ssm_out matmul: read bxb packed at value_dim, write bffn_norm packed at dim */
      if(l <= 4 && _SSM_DBG) {
          picolm_gpu_sync(dev);
          float hb[8];picolm_gpu_memcpy(hb,bxb+(size_t)(n_tokens-1)*value_dim,32,-1,dev);
          double hr=0;for(int _i=0;_i<8;_i++)hr+=hb[_i]*hb[_i];
          fprintf(stderr,"[CMP l%d] hperm_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,hb[0],hb[1],hb[2],hb[3],hb[4],hb[5],hb[6],hb[7],sqrt(hr/8));
          /* Full bxb last-token dump: per-head RMS */
          { size_t lt_off=(size_t)(n_tokens-1)*value_dim;
            int n_vh=value_dim; // = n_v * hvdim = 48*128 = 6144
            float *hbuf=malloc(n_vh*sizeof(float));
            picolm_gpu_memcpy(hbuf,bxb+lt_off,(size_t)n_vh*sizeof(float),-1,dev);
            double frms=0; for(int fi=0;fi<n_vh;fi++) frms+=hbuf[fi]*hbuf[fi];
            fprintf(stderr,"[CMP l%d] bxb_full_rms=%.6f n_vh=%d\n",l,sqrt(frms/n_vh),n_vh);
            for(int hh=0;hh<n_v;hh++){
                double h_rms=0; for(int dd=0;dd<hvdim;dd++) h_rms+=hbuf[hh*hvdim+dd]*hbuf[hh*hvdim+dd];
                if(hh<4||hh>=n_v-2||sqrt(h_rms/hvdim)>0.01){
                    fprintf(stderr,"[CMP l%d] bxb_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                        l,hh,(float)sqrt(h_rms/hvdim),hbuf[hh*hvdim],hbuf[hh*hvdim+1],hbuf[hh*hvdim+2],hbuf[hh*hvdim+3]);
                }
            }
            free(hbuf);
          }
          /* battn_out per-head dump (same token, same stride) for comparison */
          { size_t lt2=(size_t)(n_tokens-1)*xb2_stride;
            float *a2=malloc(value_dim*sizeof(float));
            // Read head-by-head from battn_out at xb2_stride offsets
            for(int hh=0;hh<n_v;hh++){
                float *seg=a2+hh*hvdim;
                picolm_gpu_memcpy(seg,battn_out+lt2+hh*hvdim,hvdim*sizeof(float),-1,dev);
                double h_rms=0; for(int dd=0;dd<hvdim;dd++) h_rms+=seg[dd]*seg[dd];
                if(hh<4||hh>=n_v-2||sqrt(h_rms/hvdim)>0.01){
                    fprintf(stderr,"[CMP l%d] attn_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                        l,hh,(float)sqrt(h_rms/hvdim),seg[0],seg[1],seg[2],seg[3]);
                }
            }
            free(a2);
          }
      }
      if(l <= 4 && _SSM_DBG) {
          /* Compare full bxb last token - find outlier elements */
          { size_t lt_off=(size_t)(n_tokens-1)*value_dim;
            float *hbuf=malloc(value_dim*sizeof(float));
            picolm_gpu_sync(dev);
            picolm_gpu_memcpy(hbuf,bxb+lt_off,(size_t)value_dim*sizeof(float),-1,dev);
            int maxidx=0; float maxab=0;
            for(int vi=0;vi<value_dim;vi++){float av=hbuf[vi];if(av<0)av=-av;if(av>maxab){maxab=av;maxidx=vi;}}
            fprintf(stderr,"[CMP l%d] bxb_maxabs=%.6f at idx=%d (head=%d off=%d)\n",l,maxab,maxidx,maxidx/hvdim,maxidx%hvdim);
            int s=maxidx-4;if(s<0)s=0;
            fprintf(stderr,"[CMP l%d] bxb[%d..%d]={",l,s,s+8);
            for(int vi=s;vi<s+8&&vi<value_dim;vi++)fprintf(stderr,"%s%.6f",vi>s?",":"",hbuf[vi]);
            fprintf(stderr,"}\n");
            /* Count elements above threshold */
            int nbig=0; for(int vi=0;vi<value_dim;vi++){float av=hbuf[vi];if(av<0)av=-av;if(av>0.01)nbig++;}
            fprintf(stderr,"[CMP l%d] bxb nbig(>0.01)=%d/%d\n",l,nbig,value_dim);
            /* Now dump battn_out at the SAME bxb index to see if the source had the outlier */
            { int ohead = maxidx / hvdim;
              int ooff = maxidx % hvdim;
              /* The bxb index maxidx came from head ohead in the permuted output.
               * The head_permute uses invmap: bxb[ohead] = battn_out[invmap[ohead]]
               * So the source head is invmap[ohead] */
              int invmap[48];
              int n_vpk = n_v / n_k;
              for(int hh=0;hh<n_v;hh++) invmap[hh] = qwen35_vhead_natural(hh, n_vpk, n_k);
              int src_head = invmap[ohead];
              size_t soff = (size_t)(n_tokens-1)*xb2_stride + (size_t)src_head*hvdim + ooff;
              float srcv; picolm_gpu_memcpy(&srcv, battn_out+soff, sizeof(float), 0, dev);
              fprintf(stderr,"[CMP l%d] battn_out src_head=%d off=%d val=%.6f (dst_head=%d val=%.6f)\n",
                  l, src_head, ooff, srcv, ohead, hbuf[maxidx]);
            }
            free(hbuf);
          }
      }
      /* Dump ssm_out weight row 0 (dequant from BF16) and bxb last token dot product */
      if(l==0 && _SSM_DBG){
          const void *sw_raw = picolm_gpu_tensor_weights((picolm_gpu_tensor_t*)gl->ssm_out);
          int sw_rb = (int)gguf_type_row_size(lw->type_ssm_out, value_dim);
          uint16_t *sw_bf = (uint16_t*)malloc(sw_rb);
          picolm_gpu_memcpy(sw_bf, sw_raw, sw_rb, -1, dev);
          /* D2H the bxb last token (value_dim floats, at offset (n_tokens-1)*value_dim) */
          float *xb_last = (float*)malloc(value_dim * sizeof(float));
          picolm_gpu_memcpy(xb_last, bxb + (size_t)(n_tokens-1)*value_dim, value_dim * sizeof(float), -1, dev);
          /* Compute dot product row 0 manually */
          float dot0 = 0;
          for(int _i=0;_i<value_dim;_i++) dot0 += bf16_to_fp32(sw_bf[_i]) * xb_last[_i];
          double xb_rms=0; for(int _i=0;_i<value_dim;_i++) xb_rms+=xb_last[_i]*xb_last[_i];
          fprintf(stderr,"[DBG l0] ssm_out: dot0=%.6f bxb_rms=%.6f\n", dot0, sqrt(xb_rms/value_dim));
          free(sw_bf); free(xb_last);
      }
      ok&=picolm_gpu_matmul_dev((picolm_gpu_tensor_t*)gl->ssm_out,bffn_norm,bxb,n_tokens,dev,xb2_stride,value_dim);
      _SSM_OK_CHECK("ssm_out_matmul");
    SSM_DBG_SYNC;
      if(l <= 4 && _SSM_DBG) {
          picolm_gpu_sync(dev);
          float so[8]; picolm_gpu_memcpy(so,bffn_norm+(size_t)(n_tokens-1)*xb2_stride,32,-1,dev);
          double srms=0;for(int _i=0;_i<8;_i++)srms+=so[_i]*so[_i];
          fprintf(stderr,"[CMP l%d] ssm_out_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,so[0],so[1],so[2],so[3],so[4],so[5],so[6],so[7],sqrt(srms/8));
      }
      ok&=picolm_gpu_residual_add(bx,bx,bffn_norm,n_tokens,dim,xb2_stride,dev);
      _SSM_OK_CHECK("ssm_residual");
    }
    SSM_DBG_SYNC;
    if(l <= 4 && _SSM_DBG) {
        picolm_gpu_sync(dev);
        float rb[8];picolm_gpu_memcpy(rb,bx+(size_t)(n_tokens-1)*xb2_stride,32,-1,dev);
        double rr=0;for(int _i=0;_i<8;_i++)rr+=rb[_i]*rb[_i];
        fprintf(stderr,"[CMP l%d] resid_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,rb[0],rb[1],rb[2],rb[3],rb[4],rb[5],rb[6],rb[7],sqrt(rr/8));
    }
    /* FFN */
    if(ok&&lw->ffn_gate&&lw->ffn_up&&lw->ffn_down){
      if(l<=4&&_SSM_DBG) {
        /* Dump FFN rmsnorm output (bffn_norm) token 0 for comparison with CPU */
        { float ffn0[4];
          picolm_gpu_sync(dev);
          picolm_gpu_memcpy(ffn0, bffn_norm, 16, -1, dev);
          fprintf(stderr,"[DBG] l=%d ffn_norm_out_tok0[:4]={%.6f,%.6f,%.6f,%.6f}\n",l,ffn0[0],ffn0[1],ffn0[2],ffn0[3]);
        }
          float *pw=(float*)malloc(dim*4); picolm_gpu_sync(dev);
          picolm_gpu_memcpy(pw, gw->post_attn_norm_dev[l], dim*4, -1, dev);
          double pr=0; for(int _i=0;_i<dim;_i++) pr += (double)pw[_i]*pw[_i];
          fprintf(stderr,"[DBG] l=%d post_attn_norm_w rms=%.6f\n",l,sqrt(pr/dim));
          free(pw);
          /* Also dump bx input rms */
          float *bxi=(float*)malloc(dim*4);
          picolm_gpu_memcpy(bxi, bx + (size_t)(n_tokens-1)*xb2_stride, dim*4, -1, dev);
          double br=0; for(int _i=0;_i<dim;_i++) br += (double)bxi[_i]*bxi[_i];
          fprintf(stderr,"[DBG] l=%d bx_ffn_in rms=%.6f\n",l,sqrt(br/dim));
          free(bxi);
      }
      ok&=picolm_gpu_rmsnorm_batched_dev(bffn_norm,bx,(float*)gw->post_attn_norm_dev[l],dim,eps,n_tokens,xb2_stride,dev);
      _SSM_OK_CHECK("ffn_rmsnorm");
    SSM_DBG_SYNC;
    if(l<=4&&_SSM_DBG) {
        float *bn=(float*)malloc(dim*4); picolm_gpu_sync(dev);
        picolm_gpu_memcpy(bn, bffn_norm + (size_t)(n_tokens-1)*xb2_stride, dim*4, -1, dev);
        double ss=0; for(int _i=0;_i<dim;_i++) ss += (double)bn[_i]*bn[_i];
        double rms = sqrt(ss/dim);
        fprintf(stderr,"[DBG] l=%d bffn_norm_last rms=%.6f stride=%d (want ~1.0)\n",l,rms,xb2_stride);
        /* Also dump first 4 elements */
        fprintf(stderr,"[DBG] l=%d bffn_norm_last[:4]={%.6f,%.6f,%.6f,%.6f}\n",l,bn[0],bn[1],bn[2],bn[3]);
    }
      ok&=picolm_gpu_expert_mlp_dev((picolm_gpu_tensor_t*)gl->ffn_gate,(picolm_gpu_tensor_t*)gl->ffn_up,(picolm_gpu_tensor_t*)gl->ffn_down,battn_out,bffn_norm,n_tokens,xb2_stride,xb2_stride,dev);
      _SSM_OK_CHECK("ffn_expert_mlp");
    SSM_DBG_SYNC;
      if(l<=4&&_SSM_DBG) {
        { float ffn0o[4];
          picolm_gpu_sync(dev);
          picolm_gpu_memcpy(ffn0o, battn_out, 16, -1, dev);
          fprintf(stderr,"[DBG] l=%d ffn_out_tok0[:4]={%.6f,%.6f,%.6f,%.6f}\n",l,ffn0o[0],ffn0o[1],ffn0o[2],ffn0o[3]);
        }
      }
      ok&=picolm_gpu_residual_add(bx,bx,battn_out,n_tokens,dim,xb2_stride,dev);
      _SSM_OK_CHECK("ffn_residual");
    }
    SSM_DBG_SYNC;
    if (l == 0 && ok && _SSM_DBG) {
        float fi[8];
        picolm_gpu_sync(dev);
        picolm_gpu_memcpy(fi, bffn_norm + (size_t)(n_tokens-1)*xb2_stride, sizeof(fi), -1, dev);
        double firms=0;for(int _i=0;_i<8;_i++)firms+=fi[_i]*fi[_i];
        fprintf(stderr,"[CMP l%d] ffn_in_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",
            l, fi[0],fi[1],fi[2],fi[3],fi[4],fi[5],fi[6],fi[7],sqrt(firms/8));
        float ff[8];
        picolm_gpu_sync(dev);
        picolm_gpu_memcpy(ff, battn_out + (size_t)(n_tokens-1)*xb2_stride, sizeof(ff), -1, dev);
        double frms=0;for(int _i=0;_i<8;_i++)frms+=ff[_i]*ff[_i];
        fprintf(stderr,"[CMP l%d] ffn_out_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",
            l, ff[0],ff[1],ff[2],ff[3],ff[4],ff[5],ff[6],ff[7],sqrt(frms/8));
    }
    /* Dump bx_last for bit-exact GPU vs CPU comparison */
    if (l == 0 && ok && _SSM_DBG) {
        picolm_gpu_sync(dev);
        float bx_v[100];
        picolm_gpu_memcpy(bx_v, bx + (size_t)(n_tokens-1)*xb2_stride, 400, -1, dev);
        float bx_rms=0;
        for (int i=0;i<100;i++) bx_rms += bx_v[i]*bx_v[i];
        bx_rms = sqrtf(bx_rms/100);
        fprintf(stderr, "[GPU l%d] bx_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms100=%.6f\n",
            l, bx_v[0],bx_v[1],bx_v[2],bx_v[3],bx_v[4],bx_v[5],bx_v[6],bx_v[7], bx_rms);
    }
    return ok;
#else
    (void)m;(void)s;(void)bx;(void)bxb;(void)bq;(void)battn_out;(void)bffn_norm;
    (void)bgate;(void)bup;(void)lw;(void)l;(void)n_tokens;(void)start_pos;(void)dev;
    return 0;
#endif
}
#endif /* PICOLM_GPU */

/* ================================================================
 * Batched SSM prefill layer.

 *
 * All projection matmuls batched across tokens (weights read once).
 * Convolution, alpha/beta projections also batched.
 * State recurrence remains sequential per token but uses threaded
 * ssm_head_task across v-heads within each token.
 * ================================================================ */
void ssm_prefill_layer(model_t *m, run_state_t *s,
                              float *x_batch, float *xb_batch, float *xb2_batch,
                              float *hb_batch, float *hb2_batch,
                              layer_weights_t *lw, int l,
                              int n_tokens, int start_pos, int xb2_stride,
                              void **gpu_lw) {
    int bi;
    (void)gpu_lw;
    (void)xb_batch; /* not used: SSM layer uses local ssm_xb buffer */
    (void)start_pos;
    model_config_t *c = &m->config;
    int dim = c->n_embd;
    int d_state = c->ssm_d_state;
    int n_k_heads = c->ssm_n_group;
    int n_v_heads = c->ssm_dt_rank;
    int conv_dim = 2 * d_state * n_k_heads + c->ssm_d_inner;
    int head_v_dim = c->ssm_d_inner / n_v_heads;
    float eps = c->rms_norm_eps;

    int n_k = c->ssm_n_group;
    int n_vpk = n_v_heads / n_k;
    int half_vpk = n_vpk / 2;
    int do_remap = !m->from_safetensors && n_k > 0 && n_k < n_v_heads && half_vpk > 0;
    int value_dim = c->ssm_d_inner;
    int qk_dim = d_state * n_k_heads;
    int repeat = n_v_heads / n_k_heads;

    float *conv_state = s->ssm_conv_state[l];
    float *state = s->ssm_state[l];
    float *conv1d_w = s->ssm_conv1d_w[l];
    int n_state_rows = c->ssm_d_conv - 1;
    int state_stride = conv_dim;
    int ri = 2 + l * 9;

    /* Allocate per-token scratch: xb (RMSNorm'd input) + qkv + z + conv_out */
    size_t per_tok = (size_t)(dim + conv_dim + value_dim + conv_dim);
    float *ssm_buf = (float *)malloc((size_t)n_tokens * per_tok * sizeof(float));
    if (!ssm_buf) { fprintf(stderr, "OOM: SSM prefill scratch\n"); exit(1); }
    float *ssm_xb = ssm_buf;              /* [n_tokens][dim] - RMSNorm'd input */
    float *qkv_batch = ssm_xb + (size_t)n_tokens * dim;
    float *z_batch = qkv_batch + (size_t)n_tokens * conv_dim;
    float *conv_batch = z_batch + (size_t)n_tokens * value_dim;

    /* 1. Batched RMSNorm (write to local ssm_xb with stride dim) */
    for (bi = 0; bi < n_tokens; bi++)
        rmsnorm(ssm_xb + bi * dim, x_batch + bi * dim, s->attn_norm_w[l], dim, eps);

    /* 2. Batched QKV projection (read from ssm_xb with stride dim) */
    tensor_set_repacked(m->repack_used[ri] ? m->repack_buffers[ri] : NULL);
#ifdef PICOLM_GPU
    /* gpu_lw is gpu_layer_weights_t[] (an array of STRUCTS, ~80B each), passed
     * as void**. Index with struct stride, NOT pointer stride -- the old
     * `gpu_lw[l]` read a field value (a heap handle) as a struct pointer and
     * crashed on gl->attn_qkv. m->gpu is zeroed in model_load, so any tensor
     * not uploaded for this layer is NULL and matmul_batch falls back to CPU. */
    if (gpu_lw) {
        gpu_layer_weights_t *gl = &((gpu_layer_weights_t *)gpu_lw)[l];
        tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->attn_qkv, m->gpu.device);
    }
#endif
    if (l <= 1 && n_tokens > 0) {
        /* Check activation values */
        float amax = 0, amin = 0;
        int isnan_cnt = 0;
        for (int j = 0; j < dim; j++) {
            float ax = ssm_xb[j];
            if (isnan(ax)) { isnan_cnt++; continue; }
            if (ax > amax) amax = ax;
            if (ax < amin) amin = ax;
        }
        }
    matmul_batch(qkv_batch, ssm_xb, n_tokens, lw->attn_qkv, dim, conv_dim, lw->type_attn_qkv);
    tensor_set_repacked(NULL);
    if (l == 0 && _SSM_DBG) {
        float qkv_rms=0; float qkv_v[4];
        for (int i=0;i<4;i++) qkv_v[i]=qkv_batch[(n_tokens-1)*conv_dim+i];
        for (int i=0;i<4;i++) qkv_rms += qkv_v[i]*qkv_v[i];
        qkv_rms = sqrtf(qkv_rms/4);
        fprintf(stderr, "[CPU l%d] qkv rms_last=%.6f {%.6f %.6f %.6f %.6f}\n", l, qkv_rms, qkv_v[0],qkv_v[1],qkv_v[2],qkv_v[3]);
        float *cw = s->ssm_conv1d_w[l];
        fprintf(stderr, "[CPU l%d] conv1d_w[0][:4]={%.6f,%.6f,%.6f,%.6f}\n", l, cw[0],cw[1],cw[2],cw[3]);
    }
#ifdef PICOLM_GPU
    if (gpu_lw) {
        gpu_layer_weights_t *gl = &((gpu_layer_weights_t *)gpu_lw)[l];
        tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->attn_gate_ssm, m->gpu.device);
    }
#endif

    /* 3. Batched Z gate projection */
    tensor_set_repacked(m->repack_used[ri+1] ? m->repack_buffers[ri+1] : NULL);
    matmul_batch(z_batch, ssm_xb, n_tokens, lw->attn_gate_ssm, dim, value_dim, lw->type_attn_gate_ssm);
    tensor_set_repacked(NULL);
#ifdef PICOLM_GPU
    if (gpu_lw) tensor_set_gpu_tensor(NULL, 0);
#endif
    if (do_remap) {
        /* Single temp buffer for reordering v-heads (reused each iteration).
         * Cannot use alloca inside the loop: that grows stack without shrinking,
         * causing stack overflow for large n_tokens (e.g. 613 * 24KB > 8MB). */
        float *zt = malloc(value_dim * sizeof(float));
        if (!zt) { fprintf(stderr, "OOM: zt remap buffer\n"); exit(1); }
        for (bi = 0; bi < n_tokens; bi++) {
            float *zb = z_batch + bi * value_dim;
            memcpy(zt, zb, value_dim * sizeof(float));
            for (int h = 0; h < n_v_heads; h++) {
                int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
                memcpy(zb + h * head_v_dim, zt + gh * head_v_dim, head_v_dim * sizeof(float));
            }
        }
        free(zt);
    }

    /* 4. Convolution + silu (sequential per token: conv_state is stateful)
     * Each token sees a different conv_state because we shift after each token. */
    int conv1d_gpu_done = 0;
#ifdef PICOLM_GPU
    if (gpu_lw) {
        conv1d_gpu_done = picolm_gpu_ssm_conv1d_batch(conv_batch, conv_state, qkv_batch,
                                                        conv1d_w, conv_dim, c->ssm_d_conv,
                                                        n_tokens, m->gpu.device);
    }
#endif
    if (!conv1d_gpu_done) {
        for (bi = 0; bi < n_tokens; bi++) {
            float *qkv = qkv_batch + bi * conv_dim;
            float *conv_out = conv_batch + bi * conv_dim;
            for (int co = 0; co < conv_dim; co++) {
                float sum = 0.0f;
                for (int d = 0; d < n_state_rows; d++)
                    sum += conv1d_w[d + co * c->ssm_d_conv] * conv_state[d * state_stride + co];
                sum += conv1d_w[(c->ssm_d_conv - 1) + co * c->ssm_d_conv] * qkv[co];
                float v = sum;
                conv_out[co] = v * (1.0f / (1.0f + expf(-v)));
            }
            /* Shift conv_state and append new token */
            for (int r = 0; r < n_state_rows - 1; r++)
                memcpy(conv_state + r * state_stride, conv_state + (r + 1) * state_stride, state_stride * sizeof(float));
            memcpy(conv_state + (n_state_rows - 1) * state_stride, qkv, state_stride * sizeof(float));
        }
    }

    /* 5. Split Q/K/V + L2 norm + Q scale (batched across tokens) */
    if (do_remap) {
        /* Single temp buffer for V reordering (reused each iteration). */
        float *vt = malloc(value_dim * sizeof(float));
        if (!vt) { fprintf(stderr, "OOM: vt remap buffer\n"); exit(1); }
        for (bi = 0; bi < n_tokens; bi++) {
            float *conv = conv_batch + bi * conv_dim;
            memcpy(vt, conv + 2*qk_dim, value_dim * sizeof(float));
            for (int h = 0; h < n_v_heads; h++) {
                int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
                memcpy(conv + 2*qk_dim + h * head_v_dim, vt + gh * head_v_dim, head_v_dim * sizeof(float));
            }
        }
        free(vt);
    }
    float q_scale = 1.0f / sqrtf((float)d_state);
    int l2norm_gpu_done = 0;
#ifdef PICOLM_GPU
    if (gpu_lw) {
        l2norm_gpu_done =
            picolm_gpu_ssm_l2norm_batch(conv_batch, d_state, n_k_heads, n_tokens,
                                         conv_dim, 1e-12f, q_scale, m->gpu.device) &&
            picolm_gpu_ssm_l2norm_batch(conv_batch + qk_dim, d_state, n_k_heads, n_tokens,
                                         conv_dim, 1e-12f, 1.0f, m->gpu.device);
    }
#endif
    if (!l2norm_gpu_done) {
        for (bi = 0; bi < n_tokens; bi++) {
            float *conv = conv_batch + bi * conv_dim;
            float *q = conv;
            float *k = conv + qk_dim;
            for (int h = 0; h < n_k_heads; h++) {
                float *qh = q + h * d_state;
                float nrm = 0.0f;
                for (int d = 0; d < d_state; d++) nrm += qh[d] * qh[d];
                nrm = 1.0f / sqrtf(nrm + 1e-12f) * q_scale;
                for (int d = 0; d < d_state; d++) qh[d] *= nrm;
            }
            for (int h = 0; h < n_k_heads; h++) {
                float *kh = k + h * d_state;
                float nrm = 0.0f;
                for (int d = 0; d < d_state; d++) nrm += kh[d] * kh[d];
                nrm = 1.0f / sqrtf(nrm + 1e-12f);
                for (int d = 0; d < d_state; d++) kh[d] *= nrm;
            }
        }
    }

    /* 6. Batched alpha + gate_exp + beta projections.
     * GGUF stores [dim, n_v_heads] column-major per head, with possible
     * v-head reordering. Each head is a vec_dot of dim elements.
     * We process all tokens and all heads in batched fashion. */
    /* Phase 1.3: alpha/beta stored in pooled ssm_buf instead of separate mallocs */
    float *alpha_batch = (float *)malloc((size_t)n_tokens * n_v_heads * sizeof(float));
    float *beta_batch = (float *)malloc((size_t)n_tokens * n_v_heads * sizeof(float));
    {
        gguf_type_t alpha_type = lw->type_ssm_alpha;
        gguf_type_t beta_type = lw->type_ssm_beta;
        size_t row_bytes_alpha = gguf_type_row_size(alpha_type, dim);
        size_t row_bytes_beta = gguf_type_row_size(beta_type, dim);

        /* Precompute head maps */
        int alpha_map[256];
        int beta_map[256];
        for (int h = 0; h < n_v_heads; h++) {
            alpha_map[h] = do_remap ? qwen35_vhead_gguf(h, n_vpk, n_k) : h;
            beta_map[h] = do_remap ? qwen35_vhead_gguf(h, n_vpk, n_k) : h;
        }

        int vecdot_gpu_done = 0;
#ifdef PICOLM_GPU
        /* Type-guarded exactly like ssm_forward_gpu's alpha/beta steps --
         * picolm_gpu_ssm_vecdot_batch only implements F32/Q4_0/Q8_0.
         * Decided jointly for alpha+beta (rather than independently) to
         * avoid splitting the shared Q8_0-quantize-once-per-token setup
         * the CPU fallback below uses for both. */
        if (gpu_lw &&
            (alpha_type == GGUF_TYPE_F32 || alpha_type == GGUF_TYPE_Q4_0 || alpha_type == GGUF_TYPE_Q8_0) &&
            (beta_type == GGUF_TYPE_F32 || beta_type == GGUF_TYPE_Q4_0 || beta_type == GGUF_TYPE_Q8_0)) {
            vecdot_gpu_done =
                picolm_gpu_ssm_vecdot_batch(alpha_batch, ssm_xb, lw->ssm_alpha, alpha_type,
                                             dim, n_v_heads, n_tokens, (int)row_bytes_alpha,
                                             do_remap ? alpha_map : NULL, m->gpu.device) &&
                picolm_gpu_ssm_vecdot_batch(beta_batch, ssm_xb, lw->ssm_beta, beta_type,
                                             dim, n_v_heads, n_tokens, (int)row_bytes_beta,
                                             do_remap ? beta_map : NULL, m->gpu.device);
            if (vecdot_gpu_done) {
                /* GPU vecdot doesn't fuse the dt_w bias (matches the
                 * un-fused contract of every other GPU vecdot call in
                 * this file) -- add it here, same as the CPU path's
                 * `al[h] = sum + s->ssm_dt_w[l][h]` below. */
                for (bi = 0; bi < n_tokens; bi++) {
                    float *al = alpha_batch + bi * n_v_heads;
                    for (int h = 0; h < n_v_heads; h++) al[h] += s->ssm_dt_w[l][h];
                }
            }
        }
#endif
        if (!vecdot_gpu_done) {
        /* Quantize all xb tokens to Q8_0 once for fast vec_dot */
        int nb_xb = dim / 32;
        uint8_t *xb_q8_batch = (uint8_t *)malloc((size_t)n_tokens * nb_xb * 34);
        float *xb_q8_d_batch = (float *)malloc((size_t)n_tokens * nb_xb * sizeof(float));
        for (bi = 0; bi < n_tokens; bi++) {
            quantize_row_q8_0(ssm_xb + bi * dim, xb_q8_batch + bi * nb_xb * 34, dim);
            const block_q8_0 *xqb = (const block_q8_0 *)(xb_q8_batch + bi * nb_xb * 34);
            for (int k = 0; k < nb_xb; k++) {
                xb_q8_d_batch[bi * nb_xb + k] = fp16_to_fp32_lookup(xqb[k].d);
            }
        }

        /* Alpha: per-token, per-head vec_dot with proper GGUF head indexing */
        for (bi = 0; bi < n_tokens; bi++) {
            const void *xb_q8 = (const void *)(xb_q8_batch + bi * nb_xb * 34);
            const float *xb_q8_d = xb_q8_d_batch + bi * nb_xb;
            float *al = alpha_batch + bi * n_v_heads;
            const uint8_t *alpha_w = (const uint8_t *)lw->ssm_alpha;
            for (int h = 0; h < n_v_heads; h++) {
                int gh = alpha_map[h];
                const uint8_t *head_data = alpha_w + (size_t)gh * row_bytes_alpha;
                float sum;
                if (alpha_type == GGUF_TYPE_Q8_0) sum = vec_dot_q8_0_q8_0_deltas(xb_q8, xb_q8_d, head_data, dim);
                else if (alpha_type == GGUF_TYPE_Q4_0) sum = vec_dot_q4_0_q8_0(head_data, xb_q8, dim);
                else sum = vec_dot(head_data, ssm_xb + bi * dim, dim, alpha_type);
                al[h] = sum + s->ssm_dt_w[l][h];
            }
            /* Beta */
            const uint8_t *beta_w = (const uint8_t *)lw->ssm_beta;
            float *bt = beta_batch + bi * n_v_heads;
            for (int h = 0; h < n_v_heads; h++) {
                int gh = beta_map[h];
                const uint8_t *head_data = beta_w + (size_t)gh * row_bytes_beta;
                float sum;
                if (beta_type == GGUF_TYPE_Q8_0) sum = vec_dot_q8_0_q8_0_deltas(xb_q8, xb_q8_d, head_data, dim);
                else if (beta_type == GGUF_TYPE_Q4_0) sum = vec_dot_q4_0_q8_0(head_data, xb_q8, dim);
                else sum = vec_dot(head_data, ssm_xb + bi * dim, dim, beta_type);
                bt[h] = sum;
            }
        }
        free(xb_q8_batch);
        free(xb_q8_d_batch);
        /* alpha_map and beta_map are stack-allocated */
        }

        /* Post-process: alpha -> softplus -> gate_log; beta -> sigmoid */
        /* Store gate_log = softplus(alpha + dt_w) * a_w directly in alpha_batch.
         * The chunked recurrence uses gate_log to build cum_g and decay_mask.
         * Previously we did expf(gate) here and logf(expf(gate)) in the recurrence --
         * a useless exp/log round-trip that compounds floating-point error. */
        for (bi = 0; bi < n_tokens; bi++) {
            float *al = alpha_batch + bi * n_v_heads;
            float *bt = beta_batch + bi * n_v_heads;
            for (int h = 0; h < n_v_heads; h++) {
                float a = al[h];
                float sp = (a > 20.0f) ? a : (a < -20.0f) ? expf(a) : logf(1.0f + expf(a));
                al[h] = sp * s->ssm_a_w[l][h];
                bt[h] = 1.0f / (1.0f + expf(-bt[h]));
            }
        }
    }

    /* 7. Chunked state recurrence (Phase 2).
     * Replaces sequential per-token recurrence with chunked DeltaNet.
     * Processes tokens in chunks of CS=64, using triangular matrix
    * operations within each chunk. Each v-head is fully independent
    * and parallelized via tensor_parallel_for.
    * For single-token inputs, falls back to the standard per-token path. */
    {
        if (n_tokens > 1) {
            int chunked_gpu_done = 0;
#ifdef PICOLM_SSM_VERIFY
            /* Save state+xb2 before GPU modifies them */
            size_t xb2_sz = (size_t)n_tokens * value_dim * sizeof(float);
            size_t st_sz = (size_t)n_v_heads * d_state * d_state * sizeof(float);
            float *xb2_pre = malloc(xb2_sz);
            float *st_pre = malloc(st_sz);
            if (xb2_pre) memcpy(xb2_pre, xb2_batch, xb2_sz);
            if (st_pre) memcpy(st_pre, state, st_sz);
#endif
#ifdef PICOLM_GPU
            if (gpu_lw) {
                chunked_gpu_done = picolm_gpu_ssm_chunked_recurrence(
                    conv_batch, alpha_batch, beta_batch,
                    state, xb2_batch,
                    n_tokens, value_dim,
                    d_state, n_k_heads, n_v_heads, head_v_dim, repeat,
                    conv_dim, m->ssm_chunk_size, m->gpu.device);
            }
#endif
            if (!chunked_gpu_done) {
                ssm_chunked_recurrence(
                    conv_batch, alpha_batch, beta_batch,
                    state, xb2_batch,
                    n_tokens, value_dim,
                    d_state, n_k_heads, n_v_heads, head_v_dim, repeat,
                    conv_dim, m->ssm_chunk_size);
            }
#ifdef PICOLM_SSM_VERIFY
            if (xb2_pre && st_pre && chunked_gpu_done) {
                /* GPU ran via host-facing wrapper. Restore state, run CPU, compare.
                 * Only verify first SSM layer to keep overhead low. */
                static int verify_cnt = 0;
                if (++verify_cnt <= 3) {
                    float *xb2_cpu = malloc(xb2_sz);
                    if (xb2_cpu) {
                        memcpy(state, st_pre, st_sz);
                        ssm_chunked_recurrence(
                            conv_batch, alpha_batch, beta_batch,
                            state, xb2_cpu,
                            n_tokens, value_dim,
                            d_state, n_k_heads, n_v_heads, head_v_dim, repeat,
                            conv_dim, m->ssm_chunk_size);
                        float max_diff = 0;
                        for (int i = 0; i < n_tokens * value_dim; i++) {
                            float d = fabsf(xb2_batch[i] - xb2_cpu[i]);
                            if (d > max_diff) max_diff = d;
                        }
                        fprintf(stderr, "SSM_VERIFY l=%d xb2_max_diff=%.6e\n", l, max_diff);
                        if (l == 0) {
                            fprintf(stderr, "SSM_VERIFY l=0 cpu_xb2[:8]={");
                            for (int i = 0; i < 8; i++) fprintf(stderr, "%s%.6f", i ? ", " : "", xb2_cpu[i]);
                            fprintf(stderr, "}\n");
                            fprintf(stderr, "SSM_VERIFY l=0 gpu_xb2[:8]={");
                            for (int i = 0; i < 8; i++) fprintf(stderr, "%s%.6f", i ? ", " : "", xb2_batch[i]);
                            fprintf(stderr, "}\n");
                            /* Dump post-ssm_out bx (residual target) from CPU path */
                            fprintf(stderr, "SSM_VERIFY l=0 cpu_post_out xb2_batch_last[:8]={");
                            int lt = n_tokens - 1;
                            for (int i = 0; i < 8; i++) fprintf(stderr, "%s%.6f", i ? ", " : "", xb2_batch[lt * xb2_stride + i]);
                            fprintf(stderr, "}\n");
                        }
                    }
                    free(xb2_cpu);
                }
            }
            free(xb2_pre);
            free(st_pre);
#endif
        } else {
            /* Single token: use the standard per-token path.
             * alpha_batch now stores gate_log (log-space), so we must
             * convert to gate_exp = expf(gate) for the per-token recurrence.
             * Note: bi == n_tokens after the loop above, so use index 0. */
            float *ssm_output = (float *)malloc((size_t)d_state * n_v_heads * sizeof(float));
            float *gate_exp_buf = (float *)alloca(n_v_heads * sizeof(float));
            for (int h = 0; h < n_v_heads; h++) {
                float g = alpha_batch[h];
                gate_exp_buf[h] = (g < -50.0f) ? 0.0f : expf(g);
            }
            ssm_head_ctx_t ssm_ctx;
            ssm_ctx.state = state;
            ssm_ctx.d_state = d_state;
            ssm_ctx.head_v_dim = head_v_dim;
            ssm_ctx.n_v_heads = n_v_heads;
            ssm_ctx.repeat = repeat;
            ssm_ctx.ssm_output = ssm_output;
            ssm_ctx.q_conv = conv_batch;
            ssm_ctx.k_conv = conv_batch + qk_dim;
            ssm_ctx.v_conv = conv_batch + 2 * qk_dim;
            ssm_ctx.gate_exp = gate_exp_buf;
            ssm_ctx.beta = beta_batch; /* bi == n_tokens, use index 0 */
            tensor_parallel_for(n_v_heads, ssm_head_task, &ssm_ctx);
            for (int d = 0; d < d_state; d++)
                for (int h = 0; h < n_v_heads; h++)
                    xb2_batch[h * head_v_dim + d] = ssm_output[d * n_v_heads + h];
            free(ssm_output);
        }
    }

    free(alpha_batch);
    free(beta_batch);

    /* 8. Gated normalization (batched across tokens) */
    int gated_norm_gpu_done = 0;
#ifdef PICOLM_GPU
    if (gpu_lw) {
        gated_norm_gpu_done = picolm_gpu_ssm_prefill_gated_norm(
            xb2_batch, z_batch, s->ssm_norm_w[l], head_v_dim, n_v_heads, n_tokens, eps, m->gpu.device);
    }
#endif
    if (!gated_norm_gpu_done) {
        for (bi = 0; bi < n_tokens; bi++) {
            float *ssm_out = xb2_batch + bi * value_dim;
            float *z = z_batch + bi * value_dim;
            float *norm_w = s->ssm_norm_w[l];
            for (int h = 0; h < n_v_heads; h++) {
                float nrm = 0.0f;
                for (int d = 0; d < head_v_dim; d++) {
                    float v = ssm_out[h * head_v_dim + d];
                    nrm += v * v;
                }
                nrm = 1.0f / sqrtf(nrm / (float)head_v_dim + eps);
                for (int d = 0; d < head_v_dim; d++) {
                    float v = ssm_out[h * head_v_dim + d];
                    float zv = z[h * head_v_dim + d];
                    ssm_out[h * head_v_dim + d] = v * nrm * norm_w[d] *
                        zv * (1.0f / (1.0f + expf(-zv)));
                }
            }
        }
    }
    if (l == 0 && _SSM_DBG) {
        float gn_rms=0; float gn_v[4];
        for (int i=0;i<4;i++) { gn_v[i]=xb2_batch[(n_tokens-1)*value_dim+i]; gn_rms+=gn_v[i]*gn_v[i]; }
        gn_rms = sqrtf(gn_rms/4);
        fprintf(stderr, "[CPU l%d] gated_norm rms_last=%.6f {%.6f %.6f %.6f %.6f}\n", l, gn_rms, gn_v[0],gn_v[1],gn_v[2],gn_v[3]);
        /* Per-head dump after gated_norm */
        { int lt=n_tokens-1; const float *xb2l=xb2_batch+lt*value_dim;
          const float *nwm=s->ssm_norm_w[l];
          double nw_rms=0;for(int i=0;i<head_v_dim;i++)nw_rms+=nwm[i]*nwm[i];
          fprintf(stderr,"[CPU l0] ssm_norm_w rms=%.6f first8={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f}\n",
              sqrtf(nw_rms/head_v_dim),nwm[0],nwm[1],nwm[2],nwm[3],nwm[4],nwm[5],nwm[6],nwm[7]);
          for(int hh=0;hh<n_v_heads;hh++){
              float h_rms=0; for(int dd=0;dd<head_v_dim;dd++) h_rms+=xb2l[hh*head_v_dim+dd]*xb2l[hh*head_v_dim+dd];
              if(hh<4||hh>=n_v_heads-2||sqrtf(h_rms/head_v_dim)>0.01){
                  fprintf(stderr,"[CPU l0] gn_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                      hh,sqrtf(h_rms/head_v_dim),xb2l[hh*head_v_dim],xb2l[hh*head_v_dim+1],xb2l[hh*head_v_dim+2],xb2l[hh*head_v_dim+3]);
              }
          }
        }
    }

    /* Per-head gated_norm dump for CPU (guarded by _SSM_DBG) */
    if (l == 0 && _SSM_DBG) {
        int lt=n_tokens-1; const float *xb2l=xb2_batch+lt*value_dim;
        const float *nwm=s->ssm_norm_w[l];
        double nw_rms=0;for(int i=0;i<head_v_dim;i++)nw_rms+=nwm[i]*nwm[i];
        fprintf(stderr,"[CPU l0] ssm_norm_w rms=%.6f first8={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f}\n",
            sqrtf(nw_rms/head_v_dim),nwm[0],nwm[1],nwm[2],nwm[3],nwm[4],nwm[5],nwm[6],nwm[7]);
        for(int hh=0;hh<n_v_heads;hh++){
            float h_rms=0; for(int dd=0;dd<head_v_dim;dd++) h_rms+=xb2l[hh*head_v_dim+dd]*xb2l[hh*head_v_dim+dd];
            if(hh<4||hh>=n_v_heads-2||sqrtf(h_rms/head_v_dim)>0.01){
                fprintf(stderr,"[CPU l0] gn_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                    hh,sqrtf(h_rms/head_v_dim),xb2l[hh*head_v_dim],xb2l[hh*head_v_dim+1],xb2l[hh*head_v_dim+2],xb2l[hh*head_v_dim+3]);
            }
        }
    }

    /* CPU xb2 per-head dump for layers 0-4 */
    if (l <= 4 && _SSM_DBG) {
        { int lt2 = n_tokens - 1;
          for (int hh = 0; hh < n_v_heads; hh++) {
              float hr = 0; for (int dd = 0; dd < head_v_dim; dd++) { float v = xb2_batch[lt2 * value_dim + hh * head_v_dim + dd]; hr += v * v; }
              float hrms = sqrtf(hr / head_v_dim);
              if (hh < 4 || hh >= n_v_heads - 2 || hrms > 0.01) {
                  fprintf(stderr, "[CPU l%d] xb2_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                      l, hh, hrms, xb2_batch[lt2*value_dim+hh*head_v_dim],
                      xb2_batch[lt2*value_dim+hh*head_v_dim+1],
                      xb2_batch[lt2*value_dim+hh*head_v_dim+2],
                      xb2_batch[lt2*value_dim+hh*head_v_dim+3]);
              }
          }
          float mx = 0; for (int i = 0; i < value_dim; i++) { float a = xb2_batch[lt2 * value_dim + i]; if (a < 0) a = -a; if (a > mx) mx = a; }
          fprintf(stderr, "[CPU l%d] xb2_maxabs=%.6f\n", l, mx);
        }
    }

    /* 9. Output projection (batched) */
    if (do_remap) {
        /* Reorder ALL tokens to GGUF column order first (cheap CPU
         * pass), then ONE batched GPU matmul across all n_tokens --
         * NOT n_tokens separate single-token matmul() calls. The old
         * per-token loop here was, per profiling, the single largest
         * cost in prefill: each single-token GPU matmul call pays its
         * own full H2D+kernel+D2H (and, via matmul()'s own scratch
         * management, malloc) overhead, so a 36-token prompt paid that
         * fixed per-call overhead 36 times per layer instead of once. */
        float *fo_gguf_batch = (float *)malloc((size_t)n_tokens * value_dim * sizeof(float));
        for (bi = 0; bi < n_tokens; bi++) {
            float *fo = xb2_batch + bi * value_dim;
            float *dst = fo_gguf_batch + bi * value_dim;
            for (int h = 0; h < n_v_heads; h++) {
                int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
                memcpy(dst + gh * head_v_dim, fo + h * head_v_dim, head_v_dim * sizeof(float));
            }
        }
        tensor_set_repacked(m->repack_used[ri+5] ? m->repack_buffers[ri+5] : NULL);
#ifdef PICOLM_GPU
        if (gpu_lw) {
            gpu_layer_weights_t *gl = &((gpu_layer_weights_t *)gpu_lw)[l];
            tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->ssm_out, m->gpu.device);
        }
#endif
        if (l == 0 && _SSM_DBG) {
            int lt=n_tokens-1; const float *fb=fo_gguf_batch+lt*value_dim;
            double frms=0; for(int fi=0;fi<value_dim;fi++) frms+=fb[fi]*fb[fi];
            fprintf(stderr,"[CPU l0] fo_gguf_rms=%.6f n_vh=%d\n",sqrtf(frms/value_dim),value_dim);
            for(int hh=0;hh<n_v_heads;hh++){
                float h_rms=0; for(int dd=0;dd<head_v_dim;dd++) h_rms+=fb[hh*head_v_dim+dd]*fb[hh*head_v_dim+dd];
                if(hh<4||hh>=n_v_heads-2||sqrtf(h_rms/head_v_dim)>0.01){
                    fprintf(stderr,"[CPU l0] fo_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                        hh,sqrtf(h_rms/head_v_dim),fb[hh*head_v_dim],fb[hh*head_v_dim+1],fb[hh*head_v_dim+2],fb[hh*head_v_dim+3]);
                }
            }
        }
        if (l <= 4 && _SSM_DBG) {
            /* Dequantize first 64 floats of ssm_out weight on CPU for comparison */
            { float *cw = malloc(64 * sizeof(float));
              if (lw->type_ssm_out == 2) {
                  const unsigned char *q = (const unsigned char *)lw->ssm_out;
                  float sc = *(const float *)(q + 32);
                  for (int i = 0; i < 32; i++) cw[i] = (q[i] - 128) * sc;
                  for (int i = 32; i < 64; i++) {
                      float sc2 = *(const float *)(q + 64 + 32);
                      cw[i] = (q[64 + i - 32] - 128) * sc2;
                  }
              } else if (lw->type_ssm_out == 0) {
                  const float *f = (const float *)lw->ssm_out;
                  for (int i = 0; i < 64; i++) cw[i] = f[i];
              }
              fprintf(stderr,"[CPU l%d] ssm_out_w[0][:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} type=%d I=%d O=%d\n",
                  l, cw[0],cw[1],cw[2],cw[3],cw[4],cw[5],cw[6],cw[7], (int)lw->type_ssm_out, value_dim, dim);
              double wr=0;for(int wi=0;wi<64;wi++)wr+=cw[wi]*cw[wi];
              fprintf(stderr,"[CPU l%d] ssm_out_w rms64=%.6f\n",l,sqrt(wr/64));
              free(cw);
            }
        }
        float *ssm_out_buf = (float *)malloc((size_t)n_tokens * dim * sizeof(float));
        matmul_batch(ssm_out_buf, fo_gguf_batch, n_tokens, lw->ssm_out, value_dim, dim, lw->type_ssm_out);
        for (bi = 0; bi < n_tokens; bi++)
            memcpy(xb2_batch + bi * xb2_stride, ssm_out_buf + bi * dim, dim * sizeof(float));
        free(ssm_out_buf);
        free(fo_gguf_batch);
        if (l <= 4 && _SSM_DBG) {
            float so[8], sor=0; int lt=n_tokens-1;
            for(int i=0;i<8;i++){so[i]=xb2_batch[lt*xb2_stride+i];sor+=so[i]*so[i];}sor=sqrtf(sor/8);
            fprintf(stderr,"[CPU l%d] ssm_out_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,so[0],so[1],so[2],so[3],so[4],so[5],so[6],so[7],sor);
        }
        tensor_set_repacked(NULL);
#ifdef PICOLM_GPU
        if (gpu_lw) tensor_set_gpu_tensor(NULL, 0);
#endif
    } else {
        tensor_set_repacked(m->repack_used[ri+5] ? m->repack_buffers[ri+5] : NULL);
#ifdef PICOLM_GPU
        if (gpu_lw) {
            gpu_layer_weights_t *gl = &((gpu_layer_weights_t *)gpu_lw)[l];
            tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->ssm_out, m->gpu.device);
        }
#endif
        /* Cannot alias in/out when value_dim != dim (strides differ, cross-token corruption) */
        float *ssm_out_buf = (float *)malloc((size_t)n_tokens * dim * sizeof(float));
        matmul_batch(ssm_out_buf, xb2_batch, n_tokens, lw->ssm_out, value_dim, dim, lw->type_ssm_out);
        if (l <= 4 && _SSM_DBG) {
            float so[8], sor=0; int lt=n_tokens-1;
            for(int i=0;i<8;i++){so[i]=ssm_out_buf[lt*dim+i];sor+=so[i]*so[i];}sor=sqrtf(sor/8);
            fprintf(stderr,"[CPU l%d] ssm_out_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,so[0],so[1],so[2],so[3],so[4],so[5],so[6],so[7],sor);
        }
        for (bi = 0; bi < n_tokens; bi++)
            memcpy(xb2_batch + bi * xb2_stride, ssm_out_buf + bi * dim, dim * sizeof(float));
        free(ssm_out_buf);
        tensor_set_repacked(NULL);
#ifdef PICOLM_GPU
        if (gpu_lw) tensor_set_gpu_tensor(NULL, 0);
#endif
    }

    /* 10. Residual add (batched) */
    for (bi = 0; bi < n_tokens; bi++) {
        float *a = x_batch + bi * dim, *b = xb2_batch + bi * xb2_stride;
        for (int d = 0; d < dim; d++) a[d] += b[d];
    }
    if (l <= 4 && _SSM_DBG) {
        float rb[8], rbr=0; int lt=n_tokens-1;
        for(int i=0;i<8;i++){rb[i]=x_batch[lt*dim+i];rbr+=rb[i]*rb[i];}rbr=sqrtf(rbr/8);
        fprintf(stderr,"[CPU l%d] bx_after_resid[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,rb[0],rb[1],rb[2],rb[3],rb[4],rb[5],rb[6],rb[7],rbr);
    }

    /* 11. Batched FFN (if present) */
    if (c->has_moe) {
        for (bi = 0; bi < n_tokens; bi++) {
            rmsnorm(ssm_xb + bi * dim, x_batch + bi * dim, s->post_attn_norm_w[l], dim, eps);
        }
        moe_forward_batch(m, s, ssm_xb, xb2_batch, n_tokens, lw);
        for (bi = 0; bi < n_tokens; bi++) {
            float *a = x_batch + bi * dim, *b = xb2_batch + bi * dim;
            for (int d = 0; d < dim; d++) a[d] += b[d];
        }
    } else if (lw->ffn_gate && lw->ffn_up && lw->ffn_down) {
        for (bi = 0; bi < n_tokens; bi++)
            rmsnorm(ssm_xb + bi * dim, x_batch + bi * dim, s->post_attn_norm_w[l], dim, eps);

        /* Fused batched FFN on GPU: y = down(silu(gate(x)) * up(x)) for all
         * n_tokens in one call (one H2D, one D2H total). Previously this
         * branch always used matmul_dual_batch for gate+up, which has NO
         * GPU dispatch implemented at all (unlike matmul_batch, it never
         * checks the gpu_tensor global) -- so SSM-layer FFN during prefill
         * was 100% CPU no matter what. */
        int ffn_gpu_done = 0;
#ifdef PICOLM_GPU
        if (gpu_lw) {
            gpu_layer_weights_t *gl = &((gpu_layer_weights_t *)gpu_lw)[l];
            if (gl->ffn_gate && gl->ffn_up && gl->ffn_down) {
                ffn_gpu_done = picolm_gpu_expert_mlp(
                    (picolm_gpu_tensor_t *)gl->ffn_gate,
                    (picolm_gpu_tensor_t *)gl->ffn_up,
                    (picolm_gpu_tensor_t *)gl->ffn_down,
                    xb2_batch, ssm_xb, n_tokens);
            }
        }
#endif
        if (!ffn_gpu_done) {
            tensor_set_repacked(m->repack_used[ri+7] ? m->repack_buffers[ri+7] : NULL);
            matmul_dual_batch(hb_batch, hb2_batch, ssm_xb, n_tokens,
                              lw->ffn_gate, lw->ffn_up, dim, c->n_ffn,
                              lw->type_ffn_gate, lw->type_ffn_up);
            tensor_set_repacked(NULL);
            if(l==0 && _SSM_DBG) {
                float *g0 = hb_batch;  /* token 0 gate output */
                fprintf(stderr,"[CPU l0] ffn_gate_tok0[:4]={%.6f,%.6f,%.6f,%.6f}\n",g0[0],g0[1],g0[2],g0[3]);
                /* RMSNorm input (ssm_xb) for token 0 */
                float *s0 = ssm_xb;
                double sr=0; for(int _i=0;_i<8;_i++) sr+=s0[_i]*s0[_i];
                fprintf(stderr,"[CPU l0] ffn_norm_in_tok0[:4]={%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",s0[0],s0[1],s0[2],s0[3],sqrt(sr/8));
            }

            for (bi = 0; bi < n_tokens; bi++) {
                silu(hb_batch + bi * c->n_ffn, c->n_ffn);
                elemwise_mul(hb_batch + bi * c->n_ffn, hb_batch + bi * c->n_ffn,
                             hb2_batch + bi * c->n_ffn, c->n_ffn);
            }

            tensor_set_repacked(m->repack_used[ri+8] ? m->repack_buffers[ri+8] : NULL);
            matmul_batch(xb2_batch, hb_batch, n_tokens, lw->ffn_down, c->n_ffn, dim, lw->type_ffn_down);
            tensor_set_repacked(NULL);
        }
        if(l==0 && _SSM_DBG) {
            float *xb2l = xb2_batch + (n_tokens-1)*dim;
            float *xb20 = xb2_batch;  /* token 0 */
            double fr=0; for(int _i=0;_i<8;_i++) fr += xb2l[_i]*xb2l[_i];
            fprintf(stderr,"[CPU l0] ffn_out_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",
                xb2l[0],xb2l[1],xb2l[2],xb2l[3],xb2l[4],xb2l[5],xb2l[6],xb2l[7],sqrt(fr/8));
            double f0=0; for(int _i=0;_i<8;_i++) f0 += xb20[_i]*xb20[_i];
            fprintf(stderr,"[CPU l0] ffn_out_tok0[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",
                xb20[0],xb20[1],xb20[2],xb20[3],xb20[4],xb20[5],xb20[6],xb20[7],sqrt(f0/8));
        }

        for (bi = 0; bi < n_tokens; bi++) {
            float *a = x_batch + bi * dim, *b = xb2_batch + bi * dim;
            for (int d = 0; d < dim; d++) a[d] += b[d];
        }
    }
#ifdef PICOLM_SSM_VERIFY
    if (l <= 4 && _SSM_DBG) {
        int lt = n_tokens - 1;
        fprintf(stderr, "SSM_VERIFY l=0 cpu bx_last[:4]={%.6f,%.6f,%.6f,%.6f}\n",
            x_batch[lt*dim], x_batch[lt*dim+1], x_batch[lt*dim+2], x_batch[lt*dim+3]);
        /* Dump RMS of xb2 (last token) for input comparison */
        { double xrms = 0;
          for(int i = 0; i < value_dim; i++) xrms += xb2_batch[lt*xb2_stride+i]*xb2_batch[lt*xb2_stride+i];
          fprintf(stderr, "SSM_VERIFY l=0 cpu xb2_rms=%.6e\n", sqrt(xrms/value_dim)); }
        /* Dump ssm_out matmul output for last token */
        { float *ssm_out_buf = (float *)malloc((size_t)n_tokens * dim * sizeof(float));
          matmul_batch(ssm_out_buf, xb2_batch, n_tokens, lw->ssm_out, value_dim, dim, lw->type_ssm_out);
          double orm = 0; for(int i=0;i<dim;i++) orm+=ssm_out_buf[lt*dim+i]*ssm_out_buf[lt*dim+i];
          fprintf(stderr, "SSM_VERIFY l=0 cpu ssm_out_rms=%.6e\n", sqrt(orm/dim));
          fprintf(stderr, "SSM_VERIFY l=0 cpu ssm_out_last[:4]={%.6f,%.6f,%.6f,%.6f}\n",
              ssm_out_buf[lt*dim],ssm_out_buf[lt*dim+1],ssm_out_buf[lt*dim+2],ssm_out_buf[lt*dim+3]);
          free(ssm_out_buf); }
    }
#endif
    if (n_tokens > 0 && _SSM_DBG) {
        int lt = n_tokens - 1;
        fprintf(stderr, "[DBG CPU l=%d] bx_last[:4]={%.6f,%.6f,%.6f,%.6f}\n", l, x_batch[lt*dim], x_batch[lt*dim+1], x_batch[lt*dim+2], x_batch[lt*dim+3]);
    }
    free(ssm_buf);
}

size_t model_ssm_snapshot_size(const model_t *m) {
    const model_config_t *c = &m->config;
    if (!c->has_ssm) return 0;

    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    size_t total = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        total += (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
        total += (size_t)c->ssm_d_state * c->ssm_d_inner * sizeof(float);
    }
    return total;
}

/* Save current SSM state into pre-allocated buffer.
 * Layout: [conv_state_l0][conv_state_l1]...[state_l0][state_l1]...
 * Returns bytes written (= model_ssm_snapshot_size on success). */
size_t model_ssm_state_save(const model_t *m, uint8_t *buf, size_t buf_size) {
    const model_config_t *c = &m->config;
    if (!c->has_ssm) return 0;

    size_t needed = model_ssm_snapshot_size(m);
    if (buf_size < needed) {
        fprintf(stderr, "SSM state save: buffer too small (%zu < %zu)\n", buf_size, needed);
        return 0;
    }

#ifdef PICOLM_GPU
    /* Sync GPU SSM state to CPU before saving.
     * When GPU decode is active, the device state is the only up-to-date copy. */
    if (m->gpu.device >= 0 && m->gpu.ssm_state_dev[0]) {
        picolm_ssm_state_sync_to_host(m, m->gpu.device);
    }
#endif

    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    size_t off = 0;
    /* First pass: save all conv_states */
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (!m->state.ssm_conv_state[l]) continue;
        size_t sz = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
        memcpy(buf + off, m->state.ssm_conv_state[l], sz);
        off += sz;
    }
    /* Second pass: save all ssm_states */
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (!m->state.ssm_state[l]) continue;
        size_t sz = (size_t)c->ssm_d_state * c->ssm_d_inner * sizeof(float);
        memcpy(buf + off, m->state.ssm_state[l], sz);
        off += sz;
    }
    return off;
}

/* Restore SSM state from buffer.
 * Returns bytes read (= model_ssm_snapshot_size on success). */
size_t model_ssm_state_restore(model_t *m, const uint8_t *buf, size_t buf_size) {
    const model_config_t *c = &m->config;
    if (!c->has_ssm) return 0;

    size_t needed = model_ssm_snapshot_size(m);
    if (buf_size < needed) {
        fprintf(stderr, "SSM state restore: buffer too small (%zu < %zu)\n", buf_size, needed);
        return 0;
    }

    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    size_t off = 0;
    /* First pass: restore all conv_states */
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (!m->state.ssm_conv_state[l]) continue;
        size_t sz = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
        memcpy(m->state.ssm_conv_state[l], buf + off, sz);
        off += sz;
    }
    /* Second pass: restore all ssm_states */
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (!m->state.ssm_state[l]) continue;
        size_t sz = (size_t)c->ssm_d_state * c->ssm_d_inner * sizeof(float);
        memcpy(m->state.ssm_state[l], buf + off, sz);
        off += sz;
    }

#ifdef PICOLM_GPU
    /* Sync CPU SSM state to GPU after restoring.
     * The GPU decode path reads from device state, not CPU state. */
    if (m->gpu.device >= 0 && m->gpu.ssm_state_dev[0]) {
        picolm_ssm_state_sync_to_device(m, m->gpu.device);
    }
#endif

    return off;
}

/* Reset all SSM state to zero (fresh start). */
void model_ssm_state_reset(model_t *m) {
    const model_config_t *c = &m->config;
    if (!c->has_ssm) return;

    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (m->state.ssm_conv_state[l])
            memset(m->state.ssm_conv_state[l], 0,
                   (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float));
        if (m->state.ssm_state[l])
            memset(m->state.ssm_state[l], 0,
                   (size_t)c->ssm_d_state * c->ssm_d_inner * sizeof(float));
    }
}

#ifdef PICOLM_GPU
void model_ssm_state_reset_gpu(model_t *m) {
    const model_config_t *c = &m->config;
    if (!c->has_ssm || !m->gpu.kv_active) return;

    int device = m->gpu.device;
    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (m->gpu.ssm_state_dev[l]) {
            size_t st_bytes = (size_t)c->ssm_d_state * c->ssm_d_inner * sizeof(float);
            picolm_gpu_device_memset(m->gpu.ssm_state_dev[l], 0, st_bytes, device);
        }
        if (m->gpu.ssm_conv_state_dev[l]) {
            size_t cs_bytes = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
            picolm_gpu_device_memset(m->gpu.ssm_conv_state_dev[l], 0, cs_bytes, device);
        }
    }
}

/* Phase 2: GPU-pipelined decode (skeleton)
 * Keeps activations on-device across all layers.
 * Falls back to model_forward until fully implemented.
 * The elementwise kernels (gpu_rmsnorm, gpu_rope_apply,
 * gpu_residual_add) are implemented in backend_gpu.cu. */
float *model_forward_gpu(model_t *m, int token, int pos) {
    /* Phase 2: GPU-pipelined forward pass.
     * Keeps activations on-device across all layers.
     * Falls back to model_forward() if pipeline not ready. */
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    gpu_weights_t *gw = &m->gpu;
    run_state_t *s = &m->state;

    int gpu_dev = gw->device;
    int dim = c->n_embd;
    int n_ffn = c->n_ffn;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int seq_len = c->max_seq_len;
    int rope_dim = (c->rope_dim > 0) ? c->rope_dim : head_dim;
    int rope_half = rope_dim / 2;
    int q_pipeline_dim = c->has_ssm ? (q_dim * 2) : q_dim;

    /* Verify pipeline is ready */
    if (!gw->kv_active) return model_forward(m, token, pos);
    if (!picolm_gpu_pipe_x(gpu_dev)) { fprintf(stderr, "[GPU] fw_gpu fallback: !pipe_x\n"); return model_forward(m, token, pos); }
    if (!gw->rope_cos_dev || !gw->rope_sin_dev) { fprintf(stderr, "[GPU] fw_gpu fallback: !rope\n"); return model_forward(m, token, pos); }

    /* Pipeline buffer pointers */
    float *pipe_x = picolm_gpu_pipe_x(gpu_dev);
    float *pipe_xb = picolm_gpu_pipe_xb(gpu_dev);
    float *pipe_q = picolm_gpu_pipe_q(gpu_dev);
    float *pipe_k = picolm_gpu_pipe_k(gpu_dev);
    float *pipe_v = picolm_gpu_pipe_v(gpu_dev);
    float *pipe_attn_out = picolm_gpu_pipe_attn_out(gpu_dev);
    float *pipe_ffn_norm = picolm_gpu_pipe_ffn_norm(gpu_dev);
    float *pipe_gate = picolm_gpu_pipe_gate(gpu_dev);
    float *pipe_up = picolm_gpu_pipe_up(gpu_dev);

    /* GPU layer weight handles */
    gpu_layer_weights_t *gl;

    /* 1. Embedding lookup on CPU, then async H2D to pipe_x.
     * Dequantize into pinned staging buffer for async copy.
     * For interleaved Q4_0 formats, fall back to CPU path. */
    if (w->type_token_embd == GGUF_TYPE_Q4_0_8_8 ||
        w->type_token_embd == GGUF_TYPE_Q4_0_4_4 ||
        w->type_token_embd == GGUF_TYPE_Q4_0_4_8) {
        /* Interleaved token embedding formats need the full model_forward
         * dequant path. Fall back to CPU. */
        return model_forward(m, token, pos);
    }
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
        float *staging = picolm_gpu_staging_host(gpu_dev, dim * sizeof(float));
        if (!staging) {
            /* Fallback: sync copy via s->x heap buffer */
            dequantize_row(embd_row, s->x, dim, w->type_token_embd);
            picolm_gpu_memcpy(pipe_x, s->x, dim * sizeof(float), 1, gpu_dev);
        } else {
            dequantize_row(embd_row, staging, dim, w->type_token_embd);
            picolm_gpu_memcpy_async(pipe_x, staging, dim * sizeof(float), 1, gpu_dev);
        }
    }

    /* KV cache store is now fully device-native (picolm_gpu_kv_store_dev),
     * no host scratch buffer needed. */

    int this_attn_ordinal = 0;

    /* 2. Per-layer pipeline */
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];
        gl = &gw->layers[l];
        int did_cpu_ssm = 0;

        if (!c->has_ssm || lw->is_attn_layer) {
            /* A. RMSNorm: pipe_xb = rmsnorm(pipe_x, attn_norm_w[l]) */
            picolm_gpu_rmsnorm_dev(pipe_xb, pipe_x,
                                    (float *)gw->attn_norm_dev[l],
                                    dim, c->rms_norm_eps, gpu_dev);

            /* B. Q projection: pipe_q = attn_q @ pipe_xb
             * For SSM models this writes q_full_dim = 2*q_dim
             * (interleaved [Q0,Gate0,Q1,Gate1,...]). */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_q,
                                   pipe_q, pipe_xb, 1, gpu_dev, 0, 0);

            /* B1. For SSM models: de-interleave Q+gate from pipe_q.
             * After this, pipe_q holds compacted Q[q_dim], pipe_gate
             * holds gate[q_dim]. For non-SSM models, skip. */
            if (c->has_ssm) {
                /* De-interleave Q+gate. Cannot write Q in-place to pipe_q:
                 * thread h writes pipe_q[h*head_dim] which overlaps with
                 * thread h+1's read from pipe_q[(h+1)*2*head_dim].
                 * Use pipe_attn_out as temp scratch for raw Q+gate data.
                 * pipe_attn_out is sized for q_pipeline_dim (>= q_full_dim).
                 * pipe_ffn_norm is only dim-sized and would overflow.
                 *
                 * Use async D2D copy on ctx->stream - the Q projection matmul
                 * is already on ctx->stream, and stream ordering ensures the
                 * D2D copy and subsequent deinterleave kernel see the data. */
                picolm_gpu_memcpy_async(pipe_attn_out, pipe_q,
                                   (size_t)n_heads * 2 * head_dim * sizeof(float),
                                   0, gpu_dev);
                picolm_gpu_qg_deinterleave_dev(pipe_attn_out, pipe_q,
                                                pipe_gate, n_heads, head_dim,
                                                gpu_dev);
            }

            /* C. K projection: pipe_k = attn_k @ pipe_xb */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_k,
                                   pipe_k, pipe_xb, 1, gpu_dev, 0, 0);

            /* D. V projection: pipe_v = attn_v @ pipe_xb */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_v,
                                   pipe_v, pipe_xb, 1, gpu_dev, 0, 0);

            /* E. QK-norm (Qwen3): per-head RMSNorm on Q and K.
             * pipe_q is [n_heads][head_dim], weight is [head_dim].
             * pipe_k is [n_kv_heads][head_dim], same weight layout. */
            if (gw->attn_qk_norm_q_dev[l]) {
                picolm_gpu_rmsnorm_batched(pipe_q, pipe_q,
                                           (float *)gw->attn_qk_norm_q_dev[l],
                                           head_dim, c->rms_norm_eps, n_heads, 0, gpu_dev);
                picolm_gpu_rmsnorm_batched(pipe_k, pipe_k,
                                           (float *)gw->attn_qk_norm_k_dev[l],
                                           head_dim, c->rms_norm_eps, n_kv_heads, 0, gpu_dev);
            }

            /* F. RoPE on Q (in-place): rope(pipe_q, n_heads) */
            /* RoPE tables for this position on device */
            float *rope_cos_pos = (float *)gw->rope_cos_dev + (size_t)pos * rope_half;
            float *rope_sin_pos = (float *)gw->rope_sin_dev + (size_t)pos * rope_half;
            picolm_gpu_rope_apply(pipe_q, n_heads, head_dim,
                                   rope_cos_pos, rope_sin_pos, rope_half, c->rope_type, gpu_dev);

            /* G. RoPE on K (in-place): rope(pipe_k, n_kv_heads) */
            picolm_gpu_rope_apply(pipe_k, n_kv_heads, head_dim,
                                   rope_cos_pos, rope_sin_pos, rope_half, c->rope_type, gpu_dev);

            /* G. KV cache store: pack F32 -> F16 and write directly into the
             * device KV cache, entirely device-to-device. The previous
             * version of this step did a synchronous D2H of pipe_k/pipe_v,
             * a CPU F16 conversion, then an H2D via picolm_gpu_kv_store_rows
             * -- each layer, twice (K and V). picolm_gpu_memcpy is a
             * blocking gpuMemcpy, so that was also forcing a full device
             * sync twice per layer (64x per token for a 32-layer model),
             * which defeated most of the point of this pipeline. This
             * version never touches the host, so that costs nothing beyond
             * two tiny kernel launches on ctx->stream.
             *
             * Trade-off: s->key_cache/val_cache (the host-side KV mirror)
             * is NOT updated here anymore. That's fine for the current
             * eligibility gate (kv_active requires no SSM, decided once at
             * model load, never changes mid-generation), but if a mid-stream
             * CPU fallback is ever introduced, it needs a one-time bulk
             * device->host flush of the whole KV cache first -- not
             * per-token reconstruction. Not needed today; flagged here so
             * it isn't a silent trap later. */
            picolm_gpu_kv_store_dev(1, this_attn_ordinal, pos, pipe_k,
                                     n_kv_heads, head_dim, seq_len, gpu_dev);
            picolm_gpu_kv_store_dev(0, this_attn_ordinal, pos, pipe_v,
                                     n_kv_heads, head_dim, seq_len, gpu_dev);
            this_attn_ordinal++;

            /* H. Attention decode: pipe_attn_out = attn(pipe_q)
             * Uses this_attn_ordinal - 1 (already incremented above), the
             * same compacted index the KV cache was just written at. This
             * ordinal only advances for actual attention layers (above) --
             * SSM/hybrid layers below have no KV cache entry at all, they
             * index ssm_state_dev/ssm_conv_state_dev by the raw layer index
             * `l` instead. */
            /* H. Attention decode: pipe_attn_out = attn(pipe_q) */
            picolm_gpu_attention_decode_dev(pipe_attn_out, pipe_q,
                                             this_attn_ordinal - 1, pos,
                                             n_heads, n_kv_heads, head_dim,
                                             seq_len, gpu_dev);

            /* I1. For SSM models: apply gate sigmoid to attention output.
             * pipe_attn_out *= sigmoid(pipe_gate)
             * This must happen before the output projection. */
            if (c->has_ssm) {
                picolm_gpu_sigmoid_mul_dev(pipe_attn_out, pipe_gate,
                                            q_dim, gpu_dev);
            }

            /* I2. Output projection: pipe_xb = attn_output @ pipe_attn_out */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_output,
                                   pipe_xb, pipe_attn_out, 1, gpu_dev, q_pipeline_dim, 0);

            /* J. Residual add: pipe_x += pipe_xb */
            picolm_gpu_residual_add(pipe_x, pipe_x, pipe_xb, 1, dim, q_pipeline_dim, gpu_dev);
        } else {
            /* SSM/hybrid layer: try GPU-native path first, fallback to CPU hybrid */
            if (!ssm_forward_gpu(m, s, s->x, s->xb2, lw, l, pos,
                                  &m->gpu.layers[l], gpu_dev)) {
                static int w2 = 0;
                if (!w2) { fprintf(stderr, "WARN: SSM decode GPU->CPU fallback\n"); w2 = 1; }
                /* Fallback to CPU hybrid.
                 * When GPU prefill was used, the SSM persistent state (conv_state,
                 * recurrence state) lives only on GPU. Sync it to CPU so that
                 * ssm_forward below reads correct values. */
                /* Sync only on first fallback per layer -- state lives on GPU
                 * after GPU prefill and must be available for CPU ssm_forward.
                 * After each CPU ssm_forward call, state is synced back to GPU
                 * so subsequent tokens are correct. */
                { int cd = 2*c->ssm_d_state*c->ssm_n_group+c->ssm_d_inner;
                  int sd = c->ssm_dt_rank*c->ssm_d_state*c->ssm_d_state;
                  if (gw->ssm_state_dev[l] && s->ssm_state[l]) {
                      picolm_gpu_sync(gpu_dev);
                      picolm_gpu_memcpy(s->ssm_state[l], gw->ssm_state_dev[l], sd*sizeof(float), -1, gpu_dev);
                  }
                  if (gw->ssm_conv_state_dev[l] && s->ssm_conv_state[l]) {
                      picolm_gpu_memcpy(s->ssm_conv_state[l], gw->ssm_conv_state_dev[l], (c->ssm_d_conv-1)*cd*sizeof(float), -1, gpu_dev);
                  }
                }
                picolm_gpu_memcpy(s->x, pipe_x, (size_t)dim * sizeof(float), -1, gpu_dev);

                float *ssm_residual = s->xb2;
                ssm_forward(m, s, s->x, ssm_residual, lw, l, pos, &m->gpu.layers[l]);

                /* Sync updated SSM state back to GPU so subsequent tokens
                 * that succeed on GPU (or fall back again) have correct state. */
                { int cd = 2*c->ssm_d_state*c->ssm_n_group+c->ssm_d_inner;
                  int sd = c->ssm_dt_rank*c->ssm_d_state*c->ssm_d_state;
                  if (gw->ssm_state_dev[l] && s->ssm_state[l]) {
                      picolm_gpu_memcpy(gw->ssm_state_dev[l], s->ssm_state[l], sd*sizeof(float), 1, gpu_dev);
                  }
                  if (gw->ssm_conv_state_dev[l] && s->ssm_conv_state[l]) {
                      picolm_gpu_memcpy(gw->ssm_conv_state_dev[l], s->ssm_conv_state[l], (c->ssm_d_conv-1)*cd*sizeof(float), 1, gpu_dev);
                  }
                }
                picolm_gpu_memcpy(pipe_x, s->x, (size_t)dim * sizeof(float), 1, gpu_dev);
                did_cpu_ssm = 1;
            }
        }

        /* FFN block: only for attention layers (and any GPU-side SSM
         * layer if that path is ever re-enabled). ssm_forward() already
         * runs its own RMSNorm+FFN+residual internally when the layer
         * has one, so running this again for a CPU-hybrid SSM layer
         * would apply the FFN twice. */
        if (!did_cpu_ssm) {
            /* K. FFN: pipe_xb = rmsnorm(pipe_x, post_attn_norm_w[l]) */
            picolm_gpu_rmsnorm_dev(pipe_ffn_norm, pipe_x,
                                    (float *)gw->post_attn_norm_dev[l],
                                    dim, c->rms_norm_eps, gpu_dev);

            /* L. Gate: pipe_gate = ffn_gate @ pipe_ffn_norm */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_gate,
                                   pipe_gate, pipe_ffn_norm, 1, gpu_dev, 0, 0);

            /* M. Up: pipe_up = ffn_up @ pipe_ffn_norm */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_up,
                                   pipe_up, pipe_ffn_norm, 1, gpu_dev, 0, 0);

            /* N. SiLU-mul: pipe_gate = silu(pipe_gate) * pipe_up (in-place on gate) */
            picolm_gpu_silu_mul_dev(pipe_gate, pipe_up, n_ffn, gpu_dev);

            /* O. Down: pipe_xb = ffn_down @ pipe_gate */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_down,
                                   pipe_xb, pipe_gate, 1, gpu_dev, q_pipeline_dim, 0);

            /* P. Residual add: pipe_x += pipe_xb */
            picolm_gpu_residual_add(pipe_x, pipe_x, pipe_xb, 1, dim, q_pipeline_dim, gpu_dev);
        } /* end if (!did_cpu_ssm) */
    }

    /* 3. Final RMSNorm: pipe_x = rmsnorm(pipe_x, output_norm_w) */
    picolm_gpu_rmsnorm_dev(pipe_x, pipe_x,
                            (float *)gw->output_norm_dev,
                            dim, c->rms_norm_eps, gpu_dev);

    /* 4. Sync once, then lm_head on host */
    picolm_gpu_sync(gpu_dev);

    /* Download pipe_x to host */
    picolm_gpu_memcpy(s->x, pipe_x, dim * sizeof(float), -1, gpu_dev);

    /* 5. Output projection -> logits (host-facing, needs D2H anyway for sampling) */
    tensor_set_repacked(m->repack_used[1] ? m->repack_buffers[1] : NULL);
    tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gw->output, gpu_dev);
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);
    tensor_set_repacked(NULL);
    tensor_set_gpu_tensor(NULL, 0);

    return s->logits;
}


/* Process one ubatch through all layers.
 * The ubatch's token embeddings are already loaded into bx.
 * start_pos is the global position of the first token in this ubatch.
 * n_ubatch is the number of tokens in this ubatch.
 * Returns 0 on success, -1 on failure. */
static int _prefill_gpu_ubatch(model_t *m, run_state_t *s, gpu_weights_t *gw,
    int gpu_dev, int dim, int n_ffn, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int rope_dim, int rope_half,
    int q_dim, int kv_dim, int q_full_dim, int xb_stride,
    float *bx, float *bxb, float *bq, float *bk, float *bv,
    float *battn_out, float *bffn_norm, float *bgate, float *bup,
    int n_ubatch, int start_pos, volatile int *interrupt) {
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    gpu_layer_weights_t *gl;

    (void)rope_dim;

    /* NaN guard macro for debugging */
#if 1
    #define _PFX_NAN_CHECK(_buf, _label) \
        do { if (_SSM_DBG) { \
            float _chk[4]; picolm_gpu_sync(gpu_dev); \
            picolm_gpu_memcpy(_chk, _buf + (size_t)(n_ubatch-1) * xb_stride, 16, -1, gpu_dev); \
            if (_chk[0] != _chk[0]) fprintf(stderr, "[NAN] %s at layer %d\n", _label, l); \
        } } while(0)
#else
    #define _PFX_NAN_CHECK(_buf, _label) ((void)0)
#endif

    /* Track attention layer ordinal as we iterate */
    int attn_ord = 0;

    for (int l = 0; l < c->n_layers; l++) {
        if (interrupt && *interrupt) return -1;

        layer_weights_t *lw = &w->layers[l];
        gl = &gw->layers[l];

        if (c->has_ssm && !lw->is_attn_layer) {
            if (ssm_prefill_layer_gpu(m, s, bx, bxb, bq, battn_out, bffn_norm, bgate, bup, lw, l, n_ubatch, start_pos, gpu_dev)) {
                /* GPU-native SSM succeeded */
            } else {
                static int w2 = 0;
                if (!w2) { fprintf(stderr, "WARN: SSM prefill GPU->hybrid CPU\n"); w2 = 1; }
                size_t batch_bytes = (size_t)n_ubatch * dim * sizeof(float);
                picolm_gpu_sync(gpu_dev);
                for (int _ci = 0; _ci < n_ubatch; _ci++) {
                    picolm_gpu_memcpy(s->x + (size_t)_ci * dim,
                        bx + (size_t)_ci * xb_stride, dim * sizeof(float), -1, gpu_dev);
                }
                { int xb2s=dim;if(c->ssm_d_inner>dim)xb2s=c->ssm_d_inner;
                  int fs=n_ffn,md=c->n_heads*2*c->head_dim;if(dim>md)md=dim;
                  int kd=c->n_kv_heads*c->head_dim,qf=c->n_heads*2*c->head_dim;
                  size_t sz=(size_t)n_ubatch*(dim+md+xb2s+qf+2*kd+2*fs);
                  float *buf=(float*)malloc(sz*sizeof(float));
                  if(!buf)return -1;
                  float *p2=buf;float *xb=p2;p2+=n_ubatch*dim;float *xbb=p2;p2+=n_ubatch*md;
                  float *xb2=p2;p2+=n_ubatch*xb2s;float *qb=p2;p2+=n_ubatch*qf;(void)qb;
                  float *kb=p2;p2+=n_ubatch*kd;(void)kb;float *vb=p2;p2+=n_ubatch*kd;(void)vb;
                  float *hb=p2;p2+=n_ubatch*fs;float *hb2=p2;p2+=n_ubatch*fs;
                  memcpy(xb,s->x,batch_bytes);
                  ssm_prefill_layer(m,s,xb,xbb,xb2,hb,hb2,lw,l,n_ubatch,start_pos,xb2s,(void**)m->gpu.layers);
                  memcpy(s->x,xb,batch_bytes);
                  for(int _bi=0;_bi<n_ubatch;_bi++){
                      picolm_gpu_memcpy(bx+(size_t)_bi*xb_stride,s->x+(size_t)_bi*dim,dim*sizeof(float),1,gpu_dev);}
                  free(buf);
                }
            }
            if(_SSM_DBG){
                picolm_gpu_sync(gpu_dev);
                float lt8[8];picolm_gpu_memcpy(lt8,bx+(size_t)(n_ubatch-1)*xb_stride,32,-1,gpu_dev);
                fprintf(stderr,"[DBG GPU l=%d] bx_last[:4]={%.6f,%.6f,%.6f,%.6f}\n",l,lt8[0],lt8[1],lt8[2],lt8[3]);}
            continue;
        }

        int attn_out_stride = n_heads * head_dim;
        int attn_ffn_stride = n_ffn;

        if (_SSM_DBG) {
            float _chk[4]; picolm_gpu_sync(gpu_dev);
            picolm_gpu_memcpy(_chk, bx + (size_t)(n_ubatch-1)*xb_stride, 16, -1, gpu_dev);
            int _nan=0; for(int _i=0;_i<4;_i++) if(isnanf(_chk[_i])) _nan++;
            if (_nan > 0) fprintf(stderr, "!!! NaN in bx input to layer %d\n", l);
        }
        /* Phase 7: try fused RMSNorm + QKV (eliminates bxb buffer for QKV).
         * Only for non-SSM models with Q8_0 attention weights. */
        extern int picolm_gpu_rmsnorm_matmul_dev_qkv(picolm_gpu_tensor_t *tq, picolm_gpu_tensor_t *tk, picolm_gpu_tensor_t *tv,
                                                      float *bq, float *bk, float *bv,
                                                      const float *bx, const float *rmsnorm_w,
                                                      int dim, float eps, int S, int xb_stride,
                                                      int device, int y_stride_q, int y_stride_kv);

        if (!c->has_ssm && !getenv("PICOLM_NO_PHASE7")) {
            if (!picolm_gpu_rmsnorm_matmul_dev_qkv(
                    (picolm_gpu_tensor_t *)gl->attn_q,
                    (picolm_gpu_tensor_t *)gl->attn_k,
                    (picolm_gpu_tensor_t *)gl->attn_v,
                    bq, bk, bv,
                    bx, (float *)gw->attn_norm_dev[l],
                    dim, c->rms_norm_eps, n_ubatch, xb_stride,
                    gpu_dev, q_full_dim, n_kv_heads*head_dim)) {
                /* Phase 7 unavailable, fall through to standard path. */
            } else {
                /* Phase 7 succeeded, skip to post-QKV code. */
                goto after_qkv;
            }
        }

        /* Standard path: RMSNorm -> QKV (uses bxb intermediate buffer). */
        picolm_gpu_rmsnorm_batched_dev(bxb, bx,
                                        (float *)gw->attn_norm_dev[l],
                                        dim, c->rms_norm_eps, n_ubatch, xb_stride, gpu_dev);

        /* NaN guard after RMSNorm -- catches NaNs at the earliest point */
        _PFX_NAN_CHECK(bxb, "post_rmsnorm");

        /* QKV projections */
        if (!c->has_ssm) {
            if (!picolm_gpu_matmul_dev_qkv(
                    (picolm_gpu_tensor_t *)gl->attn_q,
                    (picolm_gpu_tensor_t *)gl->attn_k,
                    (picolm_gpu_tensor_t *)gl->attn_v,
                    bq, bk, bv,
                    bxb, n_ubatch, gpu_dev,
                    q_full_dim, n_kv_heads*head_dim, xb_stride)) {
                static int qkv_fb=1; if(qkv_fb){qkv_fb=0; fprintf(stderr,"INFO: QKV matmul fallback (S=%d)\n",n_ubatch);}
                picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_q,
                    bq, bxb, n_ubatch, gpu_dev, q_full_dim, xb_stride);
                picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_k,
                    bk, bxb, n_ubatch, gpu_dev, n_kv_heads*head_dim, xb_stride);
                picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_v,
                    bv, bxb, n_ubatch, gpu_dev, n_kv_heads*head_dim, xb_stride);
            }
        } else {
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_q,
                bq, bxb, n_ubatch, gpu_dev, q_full_dim, xb_stride);
            {
                size_t qg_bytes = (size_t)n_ubatch * n_heads * 2 * head_dim * sizeof(float);
                picolm_gpu_memcpy_async(battn_out, bq, qg_bytes, 0, gpu_dev);
                picolm_gpu_qg_deinterleave_batched_dev(battn_out, bq, bgate,
                    n_heads, head_dim, n_ubatch, gpu_dev);
            }
            if (!picolm_gpu_matmul_dev_gu(
                    (picolm_gpu_tensor_t *)gl->attn_k,
                    (picolm_gpu_tensor_t *)gl->attn_v,
                    bk, bv,
                    bxb, n_ubatch, gpu_dev,
                    n_kv_heads*head_dim, xb_stride)) {
                static int kv_fb=1; if(kv_fb){kv_fb=0; fprintf(stderr,"INFO: KV matmul fallback (S=%d)\n",n_ubatch);}
                picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_k,
                    bk, bxb, n_ubatch, gpu_dev, n_kv_heads*head_dim, xb_stride);
                picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_v,
                    bv, bxb, n_ubatch, gpu_dev, n_kv_heads*head_dim, xb_stride);
            }
        }
after_qkv:

        /* QK-norm */
        if (gw->attn_qk_norm_q_dev[l]) {
            int q_stride = q_dim;
            int kv_stride = kv_dim;
            for (int h = 0; h < n_heads; h++) {
                float *bq_h = bq + h * head_dim;
                picolm_gpu_rmsnorm_batched(bq_h, bq_h,
                                           (float *)gw->attn_qk_norm_q_dev[l],
                                           head_dim, c->rms_norm_eps, n_ubatch, q_stride, gpu_dev);
            }
            for (int h = 0; h < n_kv_heads; h++) {
                float *bk_h = bk + h * head_dim;
                picolm_gpu_rmsnorm_batched(bk_h, bk_h,
                                           (float *)gw->attn_qk_norm_k_dev[l],
                                           head_dim, c->rms_norm_eps, n_ubatch, kv_stride, gpu_dev);
            }
        }

        /* RoPE */
        {
            picolm_gpu_rope_apply_batched(bq, n_heads, head_dim,
                (float *)gw->rope_cos_dev, (float *)gw->rope_sin_dev,
                rope_half, start_pos, n_ubatch, c->rope_type, gpu_dev);
            picolm_gpu_rope_apply_batched(bk, n_kv_heads, head_dim,
                (float *)gw->rope_cos_dev, (float *)gw->rope_sin_dev,
                rope_half, start_pos, n_ubatch, c->rope_type, gpu_dev);
        }

        /* KV cache store */
        /* attn_ord already tracks the ordinal from the loop counter above */

        picolm_gpu_kv_store_dev_batched(1, attn_ord, start_pos, n_ubatch,
                                         bk, n_kv_heads, head_dim, seq_len, gpu_dev);
        picolm_gpu_kv_store_dev_batched(0, attn_ord, start_pos, n_ubatch,
                                         bv, n_kv_heads, head_dim, seq_len, gpu_dev);

        /* Attention */
        picolm_gpu_attention_prefill_dev(battn_out, bq,
                                          attn_ord, start_pos, n_ubatch,
                                          n_heads, n_kv_heads, head_dim,
                                          seq_len, gpu_dev);

        /* SSM gate sigmoid */
        if (c->has_ssm) {
            int q_dim_l = n_heads * head_dim;
            picolm_gpu_sigmoid_mul_batched_dev(battn_out, bgate,
                                                q_dim_l, n_ubatch, gpu_dev);
        }

        /* Output projection */
        picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_output,
                               bxb, battn_out, n_ubatch, gpu_dev, xb_stride, attn_out_stride);

        /* Residual add */
        picolm_gpu_residual_add(bx, bx, bxb, n_ubatch, dim, xb_stride, gpu_dev);

        /* FFN RMSNorm */
        picolm_gpu_rmsnorm_batched_dev(bffn_norm, bx,
                                        (float *)gw->post_attn_norm_dev[l],
                                        dim, c->rms_norm_eps, n_ubatch, xb_stride, gpu_dev);

        /* FFN gate+up */
        if (!picolm_gpu_matmul_dev_gu(
                (picolm_gpu_tensor_t *)gl->ffn_gate,
                (picolm_gpu_tensor_t *)gl->ffn_up,
                bgate, bup,
                bffn_norm, n_ubatch, gpu_dev,
                attn_ffn_stride, xb_stride)) {
            static int gu_fb=1; if(gu_fb){gu_fb=0; fprintf(stderr,"INFO: FFN GU matmul fallback (S=%d)\n",n_ubatch);}
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_gate,
                bgate, bffn_norm, n_ubatch, gpu_dev, attn_ffn_stride, xb_stride);
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_up,
                bup, bffn_norm, n_ubatch, gpu_dev, attn_ffn_stride, xb_stride);
        }

        /* FFN silu_mul */
        picolm_gpu_silu_mul_dev(bgate, bup, n_ubatch * n_ffn, gpu_dev);

        /* FFN down */
        picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_down,
                               bxb, bgate, n_ubatch, gpu_dev, xb_stride, attn_ffn_stride);

        /* FFN residual */
        picolm_gpu_residual_add(bx, bx, bxb, n_ubatch, dim, xb_stride, gpu_dev);

        /* Increment attention ordinal for next layer */
        attn_ord++;

        if(_SSM_DBG && (l==3||l==48)){
            float tmp[4]; picolm_gpu_sync(gpu_dev);
            picolm_gpu_memcpy(tmp,bx+(size_t)(n_ubatch-1)*xb_stride,16,0,gpu_dev);
            fprintf(stderr,"[DBG] attn_post l=%d bx_last[:4]={%.6f,%.6f,%.6f,%.6f}\n",l,tmp[0],tmp[1],tmp[2],tmp[3]);
        }
    }
    return 0;
}

/* Phase 2+ubatch: GPU-pipelined prefill forward pass with two-tier batching.
 * Large prompts are split into ubatches of ~PICOLM_UBATCH tokens (default 1024).
 * Each ubatch goes through all layers; KV cache is incrementally built.
 * Falls back to model_forward_prefill() if pipeline not ready. */
float *model_forward_prefill_gpu(model_t *m, const int *tokens, int n_tokens, int start_pos, volatile int *interrupt) {
#ifdef PICOLM_GPU
    /* Allow nsys to skip model upload: profile only this function */
#ifdef PICOLM_CUDA
    cudaProfilerStart();
#endif
#endif
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    gpu_weights_t *gw = &m->gpu;
    run_state_t *s = &m->state;

    int gpu_dev = gw->device;
    int dim = c->n_embd;
    int n_ffn = c->n_ffn;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int seq_len = c->max_seq_len;
    int rope_dim = (c->rope_dim > 0) ? c->rope_dim : head_dim;
    int rope_half = rope_dim / 2;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int q_full_dim = c->has_ssm ? (q_dim * 2) : q_dim;
    int xb_stride = c->has_ssm ? (q_dim * 2) : q_dim;
    if (dim > xb_stride) xb_stride = dim;
    int ssm_conv_dim = c->ssm_d_inner + 2 * c->ssm_d_state * c->ssm_n_group;
    if (ssm_conv_dim > xb_stride) xb_stride = ssm_conv_dim;

    /* Verify pipeline and batch buffers are ready */
    if (!gw->kv_active) return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);
    if (!picolm_gpu_pipe_x(gpu_dev)) return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);
    if (!picolm_gpu_pipe_x_b(gpu_dev)) return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);
    if (!gw->rope_cos_dev || !gw->rope_sin_dev) return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);

    /* Batch buffer pointers */
    float *bx = picolm_gpu_pipe_x_b(gpu_dev);
    float *bxb = picolm_gpu_pipe_xb_b(gpu_dev);
    float *bq = picolm_gpu_pipe_q_b(gpu_dev);
    float *bk = picolm_gpu_pipe_k_b(gpu_dev);
    float *bv = picolm_gpu_pipe_v_b(gpu_dev);
    float *battn_out = picolm_gpu_pipe_attn_out_b(gpu_dev);
    float *bffn_norm = picolm_gpu_pipe_ffn_norm_b(gpu_dev);
    float *bgate = picolm_gpu_pipe_gate_b(gpu_dev);
    float *bup = picolm_gpu_pipe_up_b(gpu_dev);

    /* 1. Embedding lookup + two-tier batching */
    int last_ubatch_size = n_tokens;
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        if (w->type_token_embd == GGUF_TYPE_Q4_0_8_8 ||
            w->type_token_embd == GGUF_TYPE_Q4_0_4_4 ||
            w->type_token_embd == GGUF_TYPE_Q4_0_4_8) {
            return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);
        }

        /* Two-tier batching: determine ubatch size.
         * llama.cpp defaults: n_batch=2048 (caller limit), n_ubatch=512 (compute chunk).
         * Our PICOLM_UBATCH maps to n_ubatch: the physical chunk size for IMMA kernels.
         * 512 is the sweet spot: good IMMA occupancy, manageable L2 cache footprint,
         * and the Q8_0 quantized activation (512*4096 = 2MB) fits well in L2.
         * Smaller prompts fit in one ubatch with zero overhead. */
        int ubatch_size = 0;
        {
            const char *ev = getenv("PICOLM_UBATCH");
            if (ev) ubatch_size = atoi(ev);
            if (ubatch_size <= 0) ubatch_size = 512;
        }

        /* Dequantize embeddings per-ubatch to avoid huge host allocations
         * for very long prompts. Each ubatch's embeddings fit in a small
         * host buffer that gets freed/recycled each iteration. */
        size_t ubatch_embd_bytes = (size_t)ubatch_size * dim * sizeof(float);
        float *host_embd = (float *)malloc(ubatch_embd_bytes);
        if (!host_embd) {
            return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);
        }

        int offset = 0;
        while (offset < n_tokens) {
            int this_ubatch = n_tokens - offset;
            if (ubatch_size > 0 && this_ubatch > ubatch_size)
                this_ubatch = ubatch_size;

            int ubatch_start_pos = start_pos + offset;

            /* Dequantize this ubatch's embeddings on CPU */
            for (int bi = 0; bi < this_ubatch; bi++) {
                const void *embd_row = (const uint8_t *)w->token_embd + (size_t)tokens[offset + bi] * row_bytes;
                float *dst = host_embd + (size_t)bi * dim;
                dequantize_row(embd_row, dst, dim, w->type_token_embd);
            }

            /* H2D this ubatch's embeddings to bx (strided).
             * No explicit sync needed -- the subsequent kernels on
             * the same stream will wait for the transfer to complete. */
            for (int bi = 0; bi < this_ubatch; bi++) {
                picolm_gpu_memcpy_async(bx + (size_t)bi * xb_stride,
                    host_embd + (size_t)bi * dim,
                    dim * sizeof(float), 1, gpu_dev);
            }

            /* Process this ubatch through all layers */
            if (_prefill_gpu_ubatch(m, s, gw, gpu_dev,
                dim, n_ffn, n_heads, n_kv_heads, head_dim,
                seq_len, rope_dim, rope_half,
                q_dim, kv_dim, q_full_dim, xb_stride,
                bx, bxb, bq, bk, bv,
                battn_out, bffn_norm, bgate, bup,
                this_ubatch, ubatch_start_pos, interrupt)) {
                free(host_embd);
                return NULL;
            }

            last_ubatch_size = this_ubatch;
            offset += this_ubatch;
        }
        free(host_embd);
    }

    /* 3. Final rmsnorm + sync + D2H + host lm_head matmul
     * After the last ubatch, bx[0..last_ubatch_size-1] holds the hidden
     * states for the last ubatch's tokens. The very last token is at
     * bx[last_ubatch_size - 1] (0-indexed within the ubatch). */
    {
        float *last_x = bx + (size_t)(last_ubatch_size - 1) * xb_stride;
        picolm_gpu_sync(gpu_dev);
        picolm_gpu_memcpy(s->x, last_x, dim * sizeof(float), -1, gpu_dev);
        if(_SSM_DBG){
            fprintf(stderr,"[DBG] prefill_last x[:4]={%.6f,%.6f,%.6f,%.6f}\n",
                s->x[0],s->x[1],s->x[2],s->x[3]);
        }
    }

    rmsnorm(s->x, s->x, s->output_norm_w, dim, c->rms_norm_eps);

    tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gw->output, gpu_dev);
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);
    tensor_set_repacked(NULL);
    tensor_set_gpu_tensor(NULL, 0);

    return s->logits;
}

void picolm_ssm_state_sync_to_device(model_t *m, int device) {
    model_config_t *c = &m->config;
    if (!c->has_ssm) return;
    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (m->gpu.ssm_state_dev[l] && m->state.ssm_state[l]) {
            size_t sz = (size_t)c->ssm_dt_rank * c->ssm_d_state * c->ssm_d_state * sizeof(float);
            picolm_gpu_memcpy(m->gpu.ssm_state_dev[l], m->state.ssm_state[l], sz, 1, device);
        }
        if (m->gpu.ssm_conv_state_dev[l] && m->state.ssm_conv_state[l]) {
            size_t sz = (size_t)(c->ssm_d_conv - 1) * (size_t)conv_dim * sizeof(float);
            picolm_gpu_memcpy(m->gpu.ssm_conv_state_dev[l], m->state.ssm_conv_state[l], sz, 1, device);
        }
    }
#ifdef PICOLM_GPU
    #ifdef PICOLM_CUDA
    cudaProfilerStop();
#endif
#endif
}

/* Sync SSM state from GPU device memory back to CPU host memory.
 * Used when ssm_forward_gpu bails (e.g. BF16 alpha/beta unsupported)
 * so that the CPU ssm_forward fallback has correct persistent state. */
void picolm_ssm_state_sync_to_host(model_t *m, int device) {
    model_config_t *c = &m->config;
    if (!c->has_ssm) return;
    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    picolm_gpu_sync(device);
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (m->gpu.ssm_state_dev[l] && m->state.ssm_state[l]) {
            size_t sz = (size_t)c->ssm_dt_rank * c->ssm_d_state * c->ssm_d_state * sizeof(float);
            picolm_gpu_memcpy(m->state.ssm_state[l], m->gpu.ssm_state_dev[l], sz, -1, device);
        }
        if (m->gpu.ssm_conv_state_dev[l] && m->state.ssm_conv_state[l]) {
            size_t sz = (size_t)(c->ssm_d_conv - 1) * (size_t)conv_dim * sizeof(float);
            picolm_gpu_memcpy(m->state.ssm_conv_state[l], m->gpu.ssm_conv_state_dev[l], sz, -1, device);
        }
    }
}

#endif /* PICOLM_GPU */
