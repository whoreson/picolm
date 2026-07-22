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

