#ifndef TENSOR_H
#define TENSOR_H

#include "quant.h"

#define MAX_THREADS 64

/* Set the scratch buffer used for row dequantization (embedding lookup, etc).
 * Must be called once at init with a buffer of at least max_row_size floats. */
void tensor_init_scratch(float *buf, int size);

/* Set repacked Q4_0->Q4_0x8 weight pointer for this matmul (AVX2 optimization).
 * Pass NULL to disable repacked path. Must be called before matmul. */
void tensor_set_repacked(const void *ptr);

/* Set GPU tensor handle for offloading matmul to GPU (PICOLM_GPU builds only).
 * Pass NULL to disable GPU offloading. Must be called before each matmul. */
#ifdef PICOLM_GPU
typedef struct picolm_gpu_tensor picolm_gpu_tensor_t;
void tensor_set_gpu_tensor(picolm_gpu_tensor_t *t, int device);
#endif

/* Set number of threads for matmul (default: 1) */
void tensor_set_threads(int t);
int  tensor_get_threads(void);

/* Return a good default thread count based on physical core enumeration.
 * Excludes HT siblings; falls back to 4 if detection fails.
 * On big.LITTLE systems, prefers big cores only. */
int tensor_default_threads(void);
int tensor_get_big_cores(void);
void tensor_threadpool_init(int n_threads);
int tensor_get_n_threads(void);
void tensor_set_n_threads(int n);
int tensor_get_thread_id(void);  /* returns 0 for main, 1..n-1 for pool workers */
void tensor_threadpool_free(void);
void matmul_batch(float *out, const float *x, int n_batch,
                   const void *W, int n, int d, gguf_type_t qtype);

/* Generic parallel-for: splits [0, count) across the existing matmul thread
 * pool, calling fn(idx, ctx) for each index. Falls back to a plain serial
 * loop when threading is disabled or count is too small to bother.
 * Used to parallelize per-head attention (independent across heads) the
 * same way matmul_batch parallelizes per-row projections. */
void tensor_parallel_for(int count, void (*fn)(int idx, void *ctx), void *ctx);
void matmul_dual_batch(float *out1, float *out2, const float *x, int n_batch,
                        const void *W1, const void *W2,
                        int n, int d, gguf_type_t qtype1, gguf_type_t qtype2);

/* Matrix-vector multiply: out[d] = W[d, n] @ x[n]
 * W is quantized in the given type, stored row-major.
 * Uses fused dequant+dot (no scratch buffer) and optional threading. */
void matmul(float *out, const float *x, const void *W, int n, int d, gguf_type_t qtype);

/* matmul_q8: matmul with pre-quantized Q8_0 activation.
 * qx = pre-quantized input (block_q8_0 array), qx_d = pre-converted deltas. */
extern void matmul_q8(float *out, const void *qx, const float *qx_d,
                      const void *W, int n, int d, gguf_type_t qtype);
extern void matmul_q8_seq(float *out, const void *qx, const float *qx_d,
                      const void *W, int n, int d, gguf_type_t qtype);

/* matmul_q8_batch: batched Q8×Q8 with pre-quantized activations.
 * qx_all: [n_batch * q8_buf_per_token] float array with Q8_0 blocks + deltas.
 * qx_d_off: offset in floats from token buffer start to delta array.
 * out: [n_batch * d] row-major output. */
extern void matmul_q8_batch(float *out, const float *qx_all, int qx_d_off,
                            int q8_buf_per_token, const void *W,
                            int n, int d, int n_batch, gguf_type_t qtype);

/* matmul_mm_id_gate_up: MoE gate+up projections via mm_id pattern.
 * Uses precomputed routing map to avoid O(experts × tokens) scanning.
 * gate_out, up_out: [n_tokens * n_used * n_ff] output buffers */
extern void matmul_mm_id_gate_up(float *gate_out, float *up_out,
    const float *qx_all, int qx_d_off, int q8_buf_per_token,
    const void *gate_w_base, const void *up_w_base,
    const int *expert_assignments, const int *expert_counts,
    int n_tokens, int n_used, int dim, int n_ff, int n_expert,
    gguf_type_t type);

/* matmul_mm_id_down: MoE down projection via mm_id pattern.
 * Uses precomputed routing map.
 * exp_down_qx_all: per-thread × per-token Q8_0 quantization buffer
 * q8_per_token: size of each Q8_0 entry in floats */
extern void matmul_mm_id_down(float *down_out,
    const float *expert_out, const void *down_w_base,
    const int *expert_assignments, const int *expert_counts,
    int n_tokens, int n_used, int dim, int n_ff, int n_expert,
    gguf_type_t type, block_q8_0 *scratch_qx, float *scratch_qx_d,
    block_q8_0 *exp_down_qx_all, float *exp_down_qx_d_all, int q8_per_token);

/* RMS normalization: out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i] */
void rmsnorm(float *out, const float *x, const float *weight, int size, float eps);

/* In-place softmax over x[0..size-1] */
void softmax(float *x, int size);

/* Rotary position encoding using pre-computed cos/sin tables.
 * cos_pos and sin_pos point to the tables for the current position:
 *   cos_pos[i] = cos(pos / freq_base^(2i/head_dim))
 *   sin_pos[i] = sin(pos / freq_base^(2i/head_dim))
 * Each has head_dim/2 entries. */
void rope(float *q, float *k, int head_dim, int n_heads, int n_kv_heads,
          const float *cos_pos, const float *sin_pos, int rope_type, int half);

/* In-place SiLU: x[i] = x[i] / (1 + exp(-x[i])) */
void silu(float *x, int size);
void gelu(float *x, int size);

/* Element-wise multiply: out[i] = a[i] * b[i] */
void elemwise_mul(float *out, const float *a, const float *b, int size);

/* Vector add in-place: a[i] += b[i] */
void vec_add(float *a, const float *b, int size);

#endif /* TENSOR_H */
