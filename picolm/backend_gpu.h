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

/* Device-native variant: x/y are already device pointers (Phase 2
 * pipeline buffers), no H2D/D2H, no internal sync. Caller must call
 * gpuDeviceSynchronize() once at the end of the whole forward pass
 * before reading anything back via D2H. Must not alias the scratch
 * buffers picolm_gpu_matmul() itself uses internally. Returns 1 on
 * success. */
int picolm_gpu_matmul_dev(picolm_gpu_tensor_t *t,
                           float *y_dev, const float *x_dev, int S, int device);

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

/* Fully device-native: weights_dev/head_map_dev uploaded once at model
 * load (not re-uploaded per call like picolm_gpu_ssm_vecdot() above).
 * x_dev/out_dev are pipeline buffers. No malloc, no H2D/D2H, no sync. */
int picolm_gpu_ssm_vecdot_dev(float *out_dev,
                               const float *x_dev,
                               const void *weights_dev,
                               gguf_type_t qtype,
                               int dim, int n_v_heads,
                               int row_bytes,
                               const int *head_map_dev,
                               int device);

/* SSM gate/beta activation (Finding 6), device-native: alpha[h] =
 * alpha_raw[h] + ssm_dt_w[h] (bias), gate_exp[h] =
 * exp(softplus(alpha[h]) * ssm_a_w[h]) (0 if underflow), beta_out[h] =
 * sigmoid(beta_raw[h]) (no bias). All pointers device-resident, no
 * H2D/D2H, no sync. */
int picolm_gpu_ssm_gate_beta_dev(float *gate_exp_out_dev, float *beta_out_dev,
                                  const float *alpha_in_dev, const float *beta_raw_in_dev,
                                  const float *ssm_a_w_dev, const float *ssm_dt_w_dev,
                                  int n_v_heads, int device);

/* SSM L2 normalization (Q/K per k_head), device-native, in-place.
 * eps must match the CPU reference (1e-12). extra_scale: pass
 * 1/sqrtf(d_state) for Q, 1.0f for K -- fuses the CPU reference's
 * separate Q-scale pass into the norm itself. Returns 1 on success. */
int picolm_gpu_ssm_l2norm_dev(float *x_dev, int head_dim, int n_heads,
                               float eps, float extra_scale, int device);

/* Generic per-head gather for the GGUF v-head remap: dst[h] =
 * src[head_map[h]]. Used for both the xb2 (z-gate) remap and the
 * v_conv remap (same head_map, head_dim=head_v_dim, n_heads=n_v_heads
 * in both cases). dst and src must not alias. Returns 1 on success. */
int picolm_gpu_ssm_head_permute_dev(float *dst_dev, const float *src_dev,
                                     const int *head_map_dev,
                                     int head_dim, int n_heads, int device);

int picolm_gpu_ssm_recurrence(float *state,
                               const float *q_conv,
                               const float *k_conv,
                               const float *v_conv,
                               const float *gate_exp,
                               const float *beta,
                               float *ssm_output,
                               int n_v_heads, int d_state,
                               int repeat, int device);

/* Device-native SSM recurrence: persistent state on device, no per-call
 * state H2D/D2H. q/k/v/gate_exp/beta still host-side per call.
 * ssm_state_dev must be allocated by picolm_gpu_alloc_device. */
int picolm_gpu_ssm_recurrence_dev(void *ssm_state_dev,
                                   const float *q_conv,
                                   const float *k_conv,
                                   const float *v_conv,
                               const float *gate_exp,
                               const float *beta,
                               float *ssm_output,
                               int n_v_heads, int d_state,
                               int repeat, int device);

/* Fully device-native SSM recurrence: q_conv/k_conv/v_conv/gate_exp/beta/
 * ssm_output are ALL device-resident pipeline buffers (unlike
 * picolm_gpu_ssm_recurrence_dev above, which still uploads/downloads
 * those per call -- only state is device-resident there). No malloc, no
 * H2D/D2H, no sync at all. Use this one in model_forward_gpu's SSM
 * layer branch; use the _dev variant above only from the CPU-orchestrated
 * ssm_forward() path where q/k/v are computed on the host. */
int picolm_gpu_ssm_recurrence_pipeline_dev(void *ssm_state_dev,
                                            const float *q_conv_dev,
                                            const float *k_conv_dev,
                                            const float *v_conv_dev,
                                            const float *gate_exp_dev,
                                            const float *beta_dev,
                                            float *ssm_output_dev,
                                            int n_v_heads, int d_state,
                                            int repeat, int device);

/* SSM causal conv1d + state shift, fused, device-native (all pointers
 * device-resident, no H2D/D2H, no internal sync). conv_state is
 * persistent per-layer device state, updated in place. Genuinely new --
 * no GPU conv1d existed before this session. Returns 1 on success. */
int picolm_gpu_ssm_conv1d_dev(float *conv_output_dev, float *conv_state_dev,
                               const float *new_input_dev, const float *conv1d_w_dev,
                               int conv_dim, int d_conv, int device);

/* SSM gated normalization, device-native (Finding 5): per-head RMSNorm
 * of dim-major ssm_output, gated by silu(head-major xb2), with the
 * GGUF v-head remap optionally fused into the output write (pass
 * head_map_dev = NULL for identity / no remap). All pointers device-
 * resident, no H2D/D2H, no sync. Returns 1 on success. */
int picolm_gpu_ssm_gated_norm_dev(float *final_output_dev,
                                   const float *ssm_output_dev,
                                   const float *xb2_dev,
                                   const float *norm_w_dev,
                                   const int *head_map_dev,
                                   int head_v_dim, int n_v_heads, float eps,
                                   int device);

/* SSM gated normalization, host-side wrapper (non-pipelined).
 * Takes CPU pointers, does its own H2D/D2H/sync. For use from
 * ssm_forward() when the pipeline is not active. */
int picolm_gpu_ssm_gated_norm(float *final_output,
                               const float *ssm_output,
                               const float *xb2,
                               const float *norm_w,
                               const int *head_map,
                               int head_v_dim, int n_v_heads, float eps,
                               int device);

/* Generic device memory allocation. Returns NULL on failure. */
void *picolm_gpu_alloc_device(size_t bytes, int device);
/* Device memory set to value (zero-fill). Returns 1 on success. */
int picolm_gpu_device_memset(void *dev_ptr, int value, size_t bytes, int device);
/* Upload a host int32 array to device. Returns device pointer or NULL. */
void *picolm_gpu_upload_int(const int *host, size_t n, int device);

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

/* Device-native KV store for model_forward_gpu(): src_dev (pipe_k/pipe_v,
 * F32, kv_dim = n_kv_heads*head_dim elements) is packed to F16 and
 * written directly into the device KV cache -- no D2H, no CPU convert,
 * no H2D, no sync. This is the version to use in the pipeline; use
 * picolm_gpu_kv_store_rows() only for the host-driven decode/prefill
 * paths (model_forward/model_forward_prefill) where K/V only ever exist
 * on the host. Returns 1 on success. */
int picolm_gpu_kv_store_dev(int is_k, int layer_ordinal, int pos,
                             const float *src_dev, int n_kv_heads, int head_dim,
                             int max_seq_len, int device);

/* Batched KV store: src_dev is [S][kv_dim] contiguous, positions
 * start_pos..start_pos+S-1. One launch for the whole prefill chunk. */
int picolm_gpu_kv_store_dev_batched(int is_k, int layer_ordinal, int start_pos, int n_positions,
                                     const float *src_dev, int n_kv_heads, int head_dim,
                                     int max_seq_len, int device);

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

/* Device-native variant: q_dev/xb_out_dev are already device pointers,
 * no H2D/D2H, no internal sync. Caller must have already issued the KV
 * store for `pos` on the same stream (ctx->stream) before calling this,
 * so in-stream ordering guarantees the write lands before this kernel's
 * read. Must not alias ctx->x/ctx->y. Returns 1 on success. */
int picolm_gpu_attention_decode_dev(float *xb_out_dev, const float *q_dev,
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

/* Device-native prefill attention: q_dev and xb_out_dev are device pointers.
 * No H2D, no D2H, no sync. S=n_tokens queries, causal mask, tiled over KV
 * positions. Same tiling as picolm_gpu_attention_prefill but no transfers. */
int picolm_gpu_attention_prefill_dev(float *xb_out_dev, const float *q_dev,
                                      int layer_ordinal, int start_pos, int n_tokens,
                                      int n_heads, int n_kv_heads, int head_dim,
                                      int max_seq_len, int device);

/* Free GPU KV cache allocations. */
void picolm_gpu_kv_free(void);

/* ================================================================
 * Phase 2: Device-resident elementwise kernels (residual pipeline)
 * ================================================================ */

/* Allocate the fixed-size device-resident pipeline buffers used by
 * model_forward_gpu() (decode, S=1 only). Call once at model load, right
 * after picolm_gpu_kv_alloc() succeeds, with sizes from model config:
 *   dim        = hidden size (residual stream width)
 *   q_dim      = n_heads * head_dim
 *   kv_dim     = n_kv_heads * head_dim
 *   ffn_hidden = FFN intermediate size
 * Idempotent (safe to call again; no-op once allocated for this
 * device -- these buffers never grow). Returns 1 on success, 0 ->
 * caller must not use model_forward_gpu() for this model/device. */
int picolm_gpu_pipeline_alloc(int dim, int q_dim, int kv_dim, int ffn_hidden, int device);

/* Allocate prefill batch pipeline buffers: [max_seq_len][dim] for S>1.
 * Same shape as pipeline_alloc but sized for batched prefill. */
int picolm_gpu_pipeline_batch_alloc(int dim, int q_dim, int kv_dim, int ffn_hidden,
                                     int max_seq_len, int device);

/* Allocate SSM-specific pipeline buffers for hybrid SSM+attention layers.
 * Called after picolm_gpu_pipeline_alloc succeeds for SSM-eligible models.
 * conv_dim = 2*d_state*n_group + d_inner, ssm_d_inner = value_dim,
 * n_v_heads = dt_rank. Returns 1 on success. */
int picolm_gpu_ssm_pipeline_alloc(int conv_dim, int ssm_d_inner, int n_v_heads, int device);

/* Free all pipeline buffers on all devices. */
void picolm_gpu_pipeline_free(void);

/* Accessors for the pipeline buffers -- gpu_device_ctx_t is file-static
 * to backend_gpu.cu, so model.c reaches the buffers through these rather
 * than a struct it can't see. Each returns NULL if picolm_gpu_pipeline_alloc()
 * hasn't been called successfully for `device` yet. */
float *picolm_gpu_pipe_x(int device);
float *picolm_gpu_pipe_xb(int device);
float *picolm_gpu_pipe_q(int device);
float *picolm_gpu_pipe_k(int device);
float *picolm_gpu_pipe_v(int device);
float *picolm_gpu_pipe_attn_out(int device);
float *picolm_gpu_pipe_ffn_norm(int device);
float *picolm_gpu_pipe_gate(int device);
float *picolm_gpu_pipe_up(int device);

/* Prefill batch buffer accessors (S>1, [max_seq_len][dim] layout). */
float *picolm_gpu_pipe_x_b(int device);
float *picolm_gpu_pipe_xb_b(int device);
float *picolm_gpu_pipe_q_b(int device);
float *picolm_gpu_pipe_k_b(int device);
float *picolm_gpu_pipe_v_b(int device);
float *picolm_gpu_pipe_attn_out_b(int device);
float *picolm_gpu_pipe_ffn_norm_b(int device);
float *picolm_gpu_pipe_gate_b(int device);
float *picolm_gpu_pipe_up_b(int device);

/* SSM pipeline buffer accessors. Returns NULL if not yet allocated. */
float *picolm_gpu_ssm_qkv_raw(int device);
float *picolm_gpu_ssm_conv_out(int device);
float *picolm_gpu_ssm_xb2(int device);
float *picolm_gpu_ssm_xb2_remap(int device);
float *picolm_gpu_ssm_v_remap(int device);
float *picolm_gpu_ssm_alpha_raw(int device);
float *picolm_gpu_ssm_beta_raw(int device);
float *picolm_gpu_ssm_gate_exp(int device);
float *picolm_gpu_ssm_beta(int device);
float *picolm_gpu_ssm_output(int device);
float *picolm_gpu_ssm_final_output(int device);

/* Device-native rmsnorm: x/out/weight are device pointers, no H2D/D2H, no sync. */
int picolm_gpu_rmsnorm_dev(float *out, const float *x, const float *weight,
                            int dim, float eps, int device);
/* Host-side rmsnorm: takes host pointers, does H2D/D2H/sync. */
int picolm_gpu_rmsnorm(float *out, const float *x, const float *weight,
                        int dim, float eps, int device);

/* Device-native batched rmsnorm: all device pointers, no H2D/D2H/sync. */
int picolm_gpu_rmsnorm_batched_dev(float *out, const float *x, const float *weight,
                                    int dim, float eps, int S, int device);
/* Host-side batched rmsnorm: takes host pointers, does H2D/D2H/sync. */
int picolm_gpu_rmsnorm_batched(float *out, const float *x, const float *weight,
                                int dim, float eps, int S, int device);

/* RoPE (rotary position embedding) on device.
 * Applies pairwise sin/cos rotation to x in-place.
 * cos_tbl, sin_tbl are device pointers [half_dim].
 * Returns 1 on success. */
int picolm_gpu_rope_apply(float *x, int n_heads, int head_dim,
                           const float *cos_tbl, const float *sin_tbl,
                           int half_dim, int rope_type, int device);

/* Batched RoPE: x is [S][n_heads][head_dim] contiguous, positions
 * start_pos..start_pos+S-1. cos_tbl_base/sin_tbl_base are the UNOFFSET
 * [max_seq_len][half_dim] base pointers. One launch for the whole chunk. */
int picolm_gpu_rope_apply_batched(float *x, int n_heads, int head_dim,
                                   const float *cos_tbl_base, const float *sin_tbl_base,
                                   int half_dim, int start_pos, int S,
                                   int rope_type, int device);

/* Residual add on device: out[i] = a[i] + b[i] for i = 0..n-1.
 * All device pointers. Returns 1 on success. */
int picolm_gpu_residual_add(float *out, const float *a, const float *b,
                             int n, int device);

/* SiLU-mul on device, in place: gate[i] = silu(gate[i]) * up[i].
 * All device pointers, on ctx->stream (required for pipeline ordering --
 * unlike the internal use inside picolm_gpu_expert_mlp(), which is safe
 * without this because that function still syncs at the end). */
int picolm_gpu_silu_mul_dev(float *gate_dev, const float *up_dev,
                             size_t n, int device);

/* The single sync point for a whole model_forward_gpu() pass -- see the
 * comment on the implementation for why every _dev primitive omits its
 * own sync. */
int picolm_gpu_sync(int device);

/* Free a GPU tensor (device memory + host handle). */
void picolm_gpu_tensor_free(picolm_gpu_tensor_t *tensor);

/* Get tensor size in bytes (device memory). */
size_t picolm_gpu_tensor_bytes(const picolm_gpu_tensor_t *tensor);

/* Get device ordinal for a tensor. */
int picolm_gpu_tensor_device(const picolm_gpu_tensor_t *tensor);
const void *picolm_gpu_tensor_weights(const picolm_gpu_tensor_t *tensor); /* device weights ptr */

/* Upload a plain host F32 vector (norm weights, RoPE cos/sin tables) to
 * a new device buffer. Returns device pointer, or NULL on failure. */
float *picolm_gpu_upload_f32(const float *host, size_t n, int device);

/* Synchronous memcpy wrapper for model.c (which is C, not CUDA).
 * dir: 1 = H2D, -1 = D2H. Returns 1 on success. */
int picolm_gpu_memcpy(void *dst, const void *src, size_t bytes, int dir, int device);

#ifdef __cplusplus
}
#endif

#endif /* PICOLM_BACKEND_GPU_H */

