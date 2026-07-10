#ifndef TENSOR_H
#define TENSOR_H

#include "quant.h"

#define MAX_THREADS 16

/* Set the scratch buffer used for row dequantization (embedding lookup, etc).
 * Must be called once at init with a buffer of at least max_row_size floats. */
void tensor_init_scratch(float *buf, int size);

/* Set repacked Q4_0->Q4_0x8 weight pointer for this matmul (AVX2 optimization).
 * Pass NULL to disable repacked path. Must be called before matmul. */
void tensor_set_repacked(const void *ptr);

/* Set number of threads for matmul (default: 1) */
void tensor_set_threads(int t);
int  tensor_get_threads(void);
void tensor_threadpool_init(int n_threads);
void tensor_threadpool_free(void);
void matmul_batch(float *out, const float *x, int n_batch,
                   const void *W, int n, int d, gguf_type_t qtype);
void matmul_dual_batch(float *out1, float *out2, const float *x, int n_batch,
                        const void *W1, const void *W2,
                        int n, int d, gguf_type_t qtype1, gguf_type_t qtype2);

/* Matrix-vector multiply: out[d] = W[d, n] @ x[n]
 * W is quantized in the given type, stored row-major.
 * Uses fused dequant+dot (no scratch buffer) and optional threading. */
void matmul(float *out, const float *x, const void *W, int n, int d, gguf_type_t qtype);

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

/* Element-wise multiply: out[i] = a[i] * b[i] */
void elemwise_mul(float *out, const float *a, const float *b, int size);

/* Vector add in-place: a[i] += b[i] */
void vec_add(float *a, const float *b, int size);

#endif /* TENSOR_H */
