#ifndef PICOLM_BACKEND_GPU_H
#define PICOLM_BACKEND_GPU_H

#include <stddef.h>
#include <stdint.h>

#include "quant.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PICOLM_GPU_MAX_DEVICES 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct picolm_gpu_tensor picolm_gpu_tensor_t;

/* Initialize GPU backend. `devices` is an array of GPU ordinals, `count` is the
 * number of devices. Returns 1 on success, 0 on failure (no GPU available,
 * init error, etc.). */
int picolm_gpu_init(const int *devices, int count);

/* Shutdown and free all GPU resources. */
void picolm_gpu_shutdown(void);

/* Number of initialized GPU devices. */
int picolm_gpu_device_count(void);

/* Get the GPU ordinal at index. Returns -1 on invalid index. */
int picolm_gpu_device_at(int index);

/* Memory info for a device. Returns 1 on success. */
int picolm_gpu_mem_info(int device, size_t *free_bytes, size_t *total_bytes);

/* Upload a quantized weight tensor to device memory.
 * `weights` points to GGUF-format quantized data (block_q4_0, block_q8_0, etc.)
 * `fmt` is a GGUF_TYPE enum value
 * `I` is the input dimension (column count)
 * `O` is the output dimension (row count, i.e. number of rows)
 * `device` is the GPU ordinal
 * On first call for a given slot, allocates device memory and uploads.
 * Subsequent calls with same params are no-ops (idempotent).
 * Returns 1 on success. */
int picolm_gpu_tensor_upload(void **tensor,
                              const void *weights,
                              gguf_type_t qtype, int I, int O, int device);

/* Matrix multiply on GPU: y[S*O] = x[S*I] @ W[O,I]^T
 * Weights are already resident on device via tensor_upload.
 * Activations x are transferred H2D, result y D2H.
 * Returns 1 on success. */
int picolm_gpu_matmul(picolm_gpu_tensor_t *t,
                       float *y, const float *x, int S, int device);

/* Fused expert-style MLP: y = down(silu(gate(x)) * up(x))
 * All three tensors must be resident on the same device.
 * Activations cross PCIe once in each direction.
 * Returns 1 on success. */
int picolm_gpu_expert_mlp(picolm_gpu_tensor_t *gate,
                           picolm_gpu_tensor_t *up,
                           picolm_gpu_tensor_t *down,
                           float *y, const float *x, int S);

/* W4A16 Tensor Core path: int4 weights + FP16 activations, FP32 accumulator.
 * Only works with GGUF_TYPE_Q4_0 weights and sm_70+/gfx9+ hardware.
 * gate and up share input x, down takes silu(gate)*up.
 * Returns 1 on success, 0 if unsupported. */
int picolm_gpu_w4a16_mlp(picolm_gpu_tensor_t *gate,
                          picolm_gpu_tensor_t *up,
                          picolm_gpu_tensor_t *down,
                          float *y, const float *x, int S);

/* General-purpose WMMA matmul: any Q4_0 tensor, batched over S rows.
 * Requirements: qtype==Q4_0, O%64==0, S%16==0, I%32==0.
 * Returns 1 on success, 0 if unsupported (fall back to quant_matmul). */
int picolm_gpu_w4a16_matmul(picolm_gpu_tensor_t *t,
                             float *y, const float *x, int S, int device);

/* SSM recurrence kernel: processes n_v_heads independently on GPU.
 * Each head does: decay state, compute sk=state*k, d=(v-sk)*beta,
 * state += k*d (outer product), output = state*q.
 * Layouts:
 *   state: [n_v_heads][d_state][d_state], row-major (float)
 *   q_conv: [n_k_heads][d_state] (k_head = h/repeat)
 *   k_conv: [n_k_heads][d_state]
 *   v_conv: [n_v_heads][head_v_dim] where head_v_dim == d_state
 *   gate_exp: [n_v_heads] (float decay factors)
 *   beta: [n_v_heads] (float per-head beta)
 *   ssm_output: [d_state][n_v_heads] (dim-major)
 * Returns 1 on success. */
/* SSM batched vec_dot: n_v_heads independent vec_dot calls on GPU.
 * All pointers are host-side. head_map maps sequential h -> GGUF head index
 * (NULL for identity). Returns 1 on success. */
int picolm_gpu_ssm_vecdot(float *out,
                           const float *x,
                           const void *weights,
                           gguf_type_t qtype,
                           int dim, int n_v_heads,
                           int row_bytes,
                           const int *head_map,
                           int device);

int picolm_gpu_ssm_recurrence(float *state,
                               const float *q_conv,
                               const float *k_conv,
                               const float *v_conv,
                               const float *gate_exp,
                               const float *beta,
                               float *ssm_output,
                               int n_v_heads, int d_state,
                               int repeat, int device);

/* ================================================================
 * Phase 1: GPU-resident KV cache + attention kernels
 * ================================================================ */

/* Allocate device KV cache. Called once at model_load.
 * kv_k_bytes / kv_v_bytes are the total bytes needed for each cache,
 * matching the CPU allocation: kv_layers * max_seq_len * n_kv_heads * kv_head_stride.
 * Returns 1 on success, 0 -> caller falls back to CPU KV cache. */
int picolm_gpu_kv_alloc(size_t kv_k_bytes, size_t kv_v_bytes, int device);

/* Store one or more full GQA rows (F16) into the device KV cache.
 *
 * Phase 1.5: the CPU-side KV cache (s->key_cache / s->val_cache) already
 * stores each row as [kv_head][head_dim] contiguous FP16 -- byte-identical
 * to the device layout [layer][pos][kv_head][head_dim]. There is therefore
 * no per-head or per-element work to do on the device side at all; this is
 * a single async H2D copy of `n_positions` contiguous rows starting at
 * `start_pos`. This replaces both the old per-head decode kernel launch
 * and the old per-token/per-head prefill loop (which cost O(n_tokens *
 * n_kv_heads) launches and was the source of the prefill regression).
 *
 * host_rows: pointer to `n_positions` contiguous rows, i.e.
 *            s->key_cache/val_cache + this_attn_ordinal*seq_len*row_bytes
 *                                   + start_pos*row_bytes
 * row_bytes: n_kv_heads * head_dim * sizeof(uint16_t) (== kv_row_size_k/v
 *            for KV_CACHE_F16 -- caller must only call this for F16 KV).
 * Returns 1 on success, 0 -> caller must fall back (row_bytes mismatch,
 * alloc missing, etc). */
int picolm_gpu_kv_store_rows(int is_k, int layer_ordinal, int start_pos, int n_positions,
                              const void *host_rows, size_t row_bytes,
                              int n_kv_heads, int head_dim, int max_seq_len, int device);

/* Decode-path attention: S=1, one query per head, online softmax over pos+1
 * cached positions. q is host pointer [n_heads][head_dim] in F32.
 * Output xb_out [n_heads][head_dim] in F32.
 * layer_ordinal: which KV cache layer (0..kv_layers-1).
 * pos: current position (0-indexed), attention attends to positions 0..pos.
 * Returns 1 on success, 0 -> caller falls back to attention_group(). */
int picolm_gpu_attention_decode(float *xb_out, const float *q,
                                 int layer_ordinal, int pos,
                                 int n_heads, int n_kv_heads, int head_dim,
                                 int max_seq_len, int device);

/* Prefill-path attention: S queries, causal mask, tiled online-softmax merge.
 * q: host [n_tokens][n_heads][head_dim] in F32.
 * xb_out: host [n_tokens][n_heads][head_dim] in F32 (pre-zeroed by caller).
 * start_pos: KV position of q[0] (for context continuation).
 * Returns 1 on success, 0 -> caller falls back to batch_attention_layer(). */
int picolm_gpu_attention_prefill(float *xb_out, const float *q,
                                  int layer_ordinal, int start_pos, int n_tokens,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  int max_seq_len, int device);

/* Free GPU KV cache allocations. */
void picolm_gpu_kv_free(void);

/* ================================================================
 * Phase 2: Device-resident elementwise kernels (residual pipeline)
 * ================================================================ */

/* RMSNorm on device: out[d] = x[d] * rsqrt(mean(x^2) + eps) * weight[d].
 * x and out are device pointers, weight is device pointer.
 * Returns 1 on success. */
int picolm_gpu_rmsnorm(float *out, const float *x, const float *weight,
                        int dim, float eps, int device);

/* RoPE (rotary position embedding) on device.
 * Applies pairwise sin/cos rotation to x in-place.
 * cos_tbl, sin_tbl are device pointers [half_dim].
 * Returns 1 on success. */
int picolm_gpu_rope_apply(float *x, int n_heads, int head_dim,
                           const float *cos_tbl, const float *sin_tbl,
                           int half_dim, int device);

/* Residual add on device: out[i] = a[i] + b[i] for i = 0..n-1.
 * All device pointers. Returns 1 on success. */
int picolm_gpu_residual_add(float *out, const float *a, const float *b,
                             int n, int device);

/* Free a GPU tensor (device memory + host handle). */
void picolm_gpu_tensor_free(picolm_gpu_tensor_t *tensor);

/* Get tensor size in bytes (device memory). */
size_t picolm_gpu_tensor_bytes(const picolm_gpu_tensor_t *tensor);

/* Get device ordinal for a tensor. */
int picolm_gpu_tensor_device(const picolm_gpu_tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif /* PICOLM_BACKEND_GPU_H */

