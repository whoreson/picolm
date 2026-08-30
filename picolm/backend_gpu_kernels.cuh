#ifndef BACKEND_GPU_KERNELS_CUH
#define BACKEND_GPU_KERNELS_CUH

#include "backend_gpu_common.cuh"
#include "backend_gpu_dequant.cuh"


__global__ void picolm_quant_matmul(float *y, const float *x, const void *weights, gguf_type_t qtype, int S, int I, int O, int row_bytes, int x_stride, int y_stride);
__global__ void picolm_q6_q8_matmul_imma(float *y, const int8_t *xq, const float *xd, const void *weights, int S, int I, int O, int row_bytes, int y_stride);
__global__ void picolm_q6_k_decode_warp(float *y, const float *x, const void *weights, int I, int O, int row_bytes);
__global__ void picolm_q5_k_q8_matmul_imma(float *y, const int8_t *xq, const float *xd, const void *weights, int S, int I, int O, int row_bytes, int y_stride);
__global__ void picolm_q4_k_q8_matmul_imma(float *y, const int8_t *xq, const float *xd, const void *weights, int S, int I, int O, int row_bytes, int y_stride);
__global__ void picolm_q3_k_q8_matmul_imma(float *y, const int8_t *xq, const float *xd, const void *weights, int S, int I, int O, int row_bytes, int y_stride);
__global__ void picolm_q2_0_q8_matmul_imma(float *y, const int8_t *xq, const float *xd, const void *weights, int S, int I, int O, int row_bytes, int y_stride);
__global__ void picolm_q1_0_q8_matmul_imma(float *y, const int8_t *xq, const float *xd, const void *weights, int S, int I, int O, int row_bytes, int y_stride);
__global__ void picolm_q4_0_q8_matmul_imma(float *y, const int8_t *xq, const float *xd, const void *weights, int S, int I, int O, int row_bytes, int y_stride);
__global__ void picolm_q4_1_q8_matmul_imma(float *y, const int8_t *xq, const float *xd, const void *weights, int S, int I, int O, int row_bytes, int y_stride);
__global__ void picolm_q2_k_q8_matmul_imma(float *y, const int8_t *xq, const float *xd, const void *weights, int S, int I, int O, int row_bytes, int y_stride);

/* ---- GPU-side Q8_0 quantization kernel ----
 *
 * Mirrors CPU quantize_row_q8_0 in quant.c exactly.
 * Per 32-element block: d = max(|x|) / 127, qs[i] = round(x[i] / d).
 * Output: int8_t qs[32] + uint16_t d (FP16) per block = 34 bytes.
 *
 * Grid: [S, n_blocks_per_row], Block: 32 threads.
 * Each block quantizes one 32-element chunk of one input row. */

__global__ void picolm_quantize_q8_0(int8_t *qs_out, float *d_out, const float *x, int I, int S);

/* Strided variant: x_stride > 0 overrides the default s*I stride.
 * Output is always contiguous [S][I]. Only called when x_stride != I. */

__global__ void picolm_quantize_q8_0_strided(int8_t *qs_out, float *d_out, const float *x, int I, int S, int x_stride);

/* Quantize F32/BF16 weight rows to Q8_0 block format (34 bytes per 32 elements).
 * Each thread block quantizes one weight row (one head).
 * Input: F32 or BF16 array [I] elements per row.
 * Output: block_q8_0 format [n_blocks * 34] bytes per row.
 * Grid: [n_heads], Block: 32 threads.
 * Each block outputs one row's worth of Q8_0 blocks to dst[row * n_blocks * 34 + ...]. */

__global__ void picolm_quantize_weights_to_q8_0(
    void *dst, const float *src, gguf_type_t qtype,
    int I, int n_heads, int src_stride, int dst_block_stride);

/* ---- Q8_0 x Q8_0 integer-MAC matmul kernel ----
 *
 * Matches CPU vec_dot_q8_0_q8_0_deltas exactly:
 *   For each 32-element block: dot(int8_x, int8_w) * d_x * d_w
 * where d_x and d_w are pre-converted to FP32.
 *
 * Grid: [O, S], Block: 256 threads.
 * Each thread block computes one output element y[s*O + o].
 *
 * xq: quantized input, shape [S, n_blocks, 32] of int8_t
 * xd: x deltas as FP32, shape [S, n_blocks] (pre-converted from FP16)
 * weights: Q8_0 weight blocks, shape [O, n_blocks] * 34 bytes each */

__global__ void picolm_q8_q8_matmul(float *y, const int8_t *xq, const float *xd, const void *weights, int S, int I, int O, int row_bytes, int y_stride);

/* Number of sequence positions per tile in the tiled Q8_0 matmul.
 * Shared memory usage (row_bytes, up to ~15KB for a 14336-wide FFN row)
 * doesn't depend on this constant -- positions are handled in an inner
 * loop, not concurrently. Stays well under the default 48KB/block. */
#define Q8_TILE_S 32
#define F16_TILE_S 32

/* Tiled Q8_0 matmul: loads each weight row into shared memory once per
 * tile of Q8_TILE_S query positions, then reuses it in the inner loop.
 * Original reads the same weight row from global memory independently
 * for every (output_row, s) pair -- for S=291 prefill, that's 291
 * redundant global reads of the same row.
 * Same arithmetic, same accumulation order per (o,s) as original --
 * bit-exact. For S=1 (decode) reduces to one tile of size 1. */

__global__ void picolm_q8_q8_matmul_tiled(float *y, const int8_t *xq, const float *xd, const void *weights, int S, int I, int O, int row_bytes, int y_stride);

__global__ void picolm_q8_q8_matmul_imma(float *y, const int8_t *xq, const float *xd,
                                          const void *weights, int S, int I, int O,
                                          int row_bytes, int y_stride);

__global__ void picolm_q8_q8_matmul_imma_w16(float *y, const int8_t *xq, const float *xd,
                                              const void *weights, int S, int I, int O,
                                              int row_bytes, int y_stride);

/* Phase 7: Fused RMSNorm + Quantize kernel.
 * Replaces separate RMSNorm + quantize kernels, eliminating F32 bxb buffer.
 * Grid: [S] (one block per row), Block: 256 threads. */
__global__ void picolm_rmsnorm_quantize_q8_0_kernel(int8_t *qs_out, float *d_out,
                                                     const float *x, const float *weight,
                                                     int dim, float eps, int S, int x_stride);

/* Phase 8: Shared-memory staged IMMA W16.
 * 64 threads/block (2 warps), 32x16 output tiles.
 * Weights staged in shared memory, reused across 32 output rows.
 * Dynamic shared memory: 16 * GPU_BLOCK_Q8_0_SIZE bytes (~544B). */
__global__ void picolm_q8_q8_matmul_imma_smw16(float *y, const int8_t *xq, const float *xd,
                                                const void *weights, int S, int I, int O,
                                                int row_bytes, int y_stride);

/* Phase 6 (abandoned): see backend_gpu_kernels.cu for details.
 * PICOLM_ASYNC_SLOT_BYTES kept for backwards compat in dispatch dead-code. */
#ifndef PICOLM_ASYNC_SLOT_BYTES
#define PICOLM_ASYNC_SLOT_BYTES 1536u
#endif

/* Phase 4: Fused QKV IMMA kernel (CUDA sm_80+).
 * Single kernel for Q+K+V projections sharing activation reads.
 * Grid: [(max(Oq,Ok,Ov)+7)/8, (S+15)/16], Block: 32 threads. */
__global__ void picolm_q8_q8_matmul_imma_qkv(float *bq, float *bk, float *bv,
                                              const int8_t *xq, const float *xd,
                                              const void *weights_q, const void *weights_k,
                                              const void *weights_v,
                                              int S, int I,
                                              int Oq, int Ok, int Ov,
                                              int row_bytes,
                                              int ys_q, int ys_kv);

/* FP16 tiled matmul: loads FP16 weight rows into shared memory once per tile.
 * FP32 activations, FP16 weights, FP32 output. */
__global__ void picolm_f16_f16_matmul_tiled(float *y, const float *x, const uint16_t *w,
                                             int S, int I, int O, int row_bytes, int y_stride);

/* BF16 tiled matmul: same tiling, BF16 weights -> FP32 output. */
__global__ void picolm_bf16_f32_matmul_tiled(float *y, const float *x, const uint16_t *w,
                                              int S, int I, int O, int row_bytes, int y_stride);

/* ---- silu_mul kernel ----
 * Element-wise: gate[i] = gate[i] / (1 + exp(-gate[i])) * up[i] */

__global__ void picolm_silu_mul(float *gate, const float *up, size_t n);

/* ================================================================
 * Phase 1: GPU-resident KV cache + attention kernels
 * ================================================================ */

/* Phase 1.5: no store kernel needed. The CPU-side KV cache row layout
 * ([kv_head][head_dim] contiguous FP16) is byte-identical to the device
 * layout, so storing is a single gpuMemcpyAsync -- see
 * picolm_gpu_kv_store_rows() in the host API section below. This replaces
 * the old picolm_gpu_kv_store_kernel / picolm_gpu_kv_store_batch_kernel,
 * which cost one launch per (position, kv_head) and dominated prefill
 * time (O(n_tokens * n_kv_heads) launches for a handful of FP16 elements
 * each).
 *
 * ---- Decode attention kernel ----
 * One thread block per KV head. Within each block, kv_mul warps (or thread
 * groups) process the kv_mul query heads that share this KV head.
 *
 * Shared memory: holds one KV position's K and V (head_dim FP16 each).
 * Each query-head group maintains its own online-softmax state in registers.
 *
 /* ---- Decode attention kernel (one thread block per KV head,
 * processes all kv_mul Q heads, loop over positions, shared mem K/V) ----
 *
 * Architecture: each block handles one KV head and its kv_mul grouped Q heads.
 * Shared memory holds one KV position's K and V vectors.
 * A tree reduce accumulates partial dot products across threads.
 * Thread 0 performs the online-softmax update and V accumulation.
 *
 * Shared memory: [K:head_dim u16][V:head_dim u16][reduce:256 float]
 * The reduce area is oversized (256 floats = 1KB) to handle the tree reduce.
 * For head_dim=128, kv_mul=8: 256+256+1024 = 1536 bytes. */
/* Rewritten: the previous version had every thread declare and
 * zero-init a private acc[8][256] float array (8KB/thread) -- far
 * beyond the register file, so it spills to local memory, and it's
 * done redundantly by all n_threads threads even though only thread 0
 * ever used its own copy for the actual accumulation (which then ran
 * serially on thread 0 alone, 255 threads idling at the barrier every
 * position). Measured at 894.6us avg vs a ~3us KV-bytes-read bandwidth
 * floor at pos=200 -- ~300x off, consistent with this being the actual
 * cost driver rather than genuine attention compute.
 *
 * Fix: one shared-memory accumulator [kv_mul][head_dim], written once
 * (cooperative zero-init), and the per-position update parallelized
 * across all threads (grid-stride over head_dim) instead of a thread-0
 * serial loop. The online-softmax branch decision is still a scalar
 * computed by thread 0 (cheap), broadcast via two shared scalars
 * (rescale, weight) so the unified update
 *   acc[d] = acc[d]*rescale + weight*v[d]
 * covers both softmax-update branches without duplicating the loop. */

__global__ void picolm_gpu_attention_decode_kernel( float *xb_out, const float *q_dev, const uint16_t *kv_k, const uint16_t *kv_v, int layer_ordinal, int pos, int n_heads, int n_kv_heads, int head_dim, int max_seq_len, size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes);

/* ---- Split-K decode attention ----
 * The shared-memory rewrite above fixed the accumulator bug, but grid =
 * n_kv_heads (likely ~8 blocks total) still massively underutilizes the
 * GPU's SM count, especially as context grows and each block serially
 * walks more positions. This splits the KV range across multiple blocks
 * per KV head (grid.y = n_splits), each producing a partial online-
 * softmax state, merged by a second small kernel -- standard
 * "flash-decoding" pattern for GQA decode with few KV heads.
 *
 * NOTE: unlike every other attention kernel change this session, this
 * is NOT expected to be bit-exact with the single-pass kernel. Merging
 * partials involves re-scaling by exp(local_max - global_max) in a
 * different summation order than the single serial pass -- floating
 * point addition isn't associative, so tiny (sub-ULP-accumulation-scale)
 * differences are expected and fine. Validate against the existing
 * PICOLM_DBG_ATTN/PICOLM_DBG_PIPELINE tolerance (1e-2/1e-3), not for
 * exact equality -- a nonzero-but-below-threshold diff here is correct
 * behavior, not a regression. */
#define ATTN_DECODE_MAX_SPLITS 32
#define ATTN_DECODE_MIN_CHUNK  64

/* Partial state layout (flat float buffer, sized by the host wrapper):
 *   partial_max: [n_heads][n_splits]
 *   partial_sum: [n_heads][n_splits]
 *   partial_acc: [n_heads][n_splits][head_dim]
 * index(kv_h, split, g) = (kv_h * n_splits + split) * kv_mul + g */

__global__ void picolm_gpu_attention_decode_split_kernel( float *partial_max, float *partial_sum, float *partial_acc, const float *q_dev, const uint16_t *kv_k, const uint16_t *kv_v, int layer_ordinal, int pos, int n_heads, int n_kv_heads, int head_dim, int max_seq_len, size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes, int n_splits, int chunk_size);

/* Merges n_splits partial states per KV head via the standard online-
 * softmax merge rule, writes final normalized output. Grid = n_kv_heads. */

__global__ void picolm_gpu_attention_decode_merge_kernel( float *xb_out, const float *partial_max, const float *partial_sum, const float *partial_acc, int n_heads, int n_kv_heads, int head_dim, int n_splits);

/* ---- Prefill tiled attention kernel ----
 * Processes one (query_head, token_tile) pair.
 * Tiled over KV positions with online-softmax merge.
 *
 * Grid: [n_heads, ceil(n_tokens / TOKEN_TILE)]
 * Each block handles one head and one tile of query tokens.
 * Inner loop tiles over KV positions. */
#define ATTN_TILE_K 32
#define ATTN_TILE_Q 32

/* FP32 K/V variant: reads K/V as FP32 instead of FP16 from KV cache.
 * Identical algorithm to picolm_gpu_attention_prefill_kernel but takes
 * FP32 K/V buffers with layout [pos][kv_head][head_dim]. */

__global__ void picolm_gpu_attention_prefill_f32kv_kernel( float *xb_out,  const float *q_dev,  const float *kv_k,  const float *kv_v,  int start_pos, int n_tokens, int n_heads, int n_kv_heads, int head_dim, int tile_q);


__global__ void picolm_gpu_attention_prefill_kernel( float *xb_out,  const float *q_dev,  const uint16_t *kv_k,  const uint16_t *kv_v,  int layer_ordinal, int start_pos, int n_tokens, int n_heads, int n_kv_heads, int head_dim, int max_seq_len, size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes, int tile_q);

/* Warp/wavefront-group scalar attention prefill: default scalar path on
 * both HIP and CUDA. Same algorithm and same bit-exact CPU-matching
 * summation order as picolm_gpu_attention_prefill_kernel, but with the
 * block-wide 128-thread tree-reduce + syncthreads-per-KV-position replaced
 * by independent 32-lane subgroup reductions -- see backend_gpu_kernels.cu
 * for the correctness argument (why the two reductions are bit-identical).
 * Legacy block-wide-reduce kernel available via PICOLM_ATTN_SLOW_SCALAR=1. */
#define ATTN_WARPGRP_SIZE 32
__global__ void picolm_gpu_attention_prefill_warpgrp_kernel( float *xb_out,  const float *q_dev,  const uint16_t *kv_k,  const uint16_t *kv_v,  int layer_ordinal, int start_pos, int n_tokens, int n_heads, int n_kv_heads, int head_dim, int max_seq_len, size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes, int tile_q);

/* dot2 variant: same signature, only defined when FAST_FP16_AVAILABLE
 * (see backend_gpu_common.cuh). Not bit-exact -- see kernel comment in
 * backend_gpu_kernels.cu. Declared unconditionally here so host dispatch
 * code compiles on all platforms; guard the *call site*, not this
 * declaration, with #ifdef GPU_FP16_DOT2_AVAILABLE. */
/* Always declare for cudafe stub generation (body is conditional). */
extern __global__ void picolm_gpu_attention_prefill_warpgrp_dot2_kernel( float *xb_out,  const float *q_dev,  const uint16_t *kv_k,  const uint16_t *kv_v,  int layer_ordinal, int start_pos, int n_tokens, int n_heads, int n_kv_heads, int head_dim, int max_seq_len, size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes, int tile_q);

/* FP16 Tensor Core Flash Attention 2 Prefill kernel.
 * Uses mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 for Q@K scoring.
 * NOT bit-exact with scalar kernel. head_dim must be multiple of 16. */
#define FA2_TILE_Q 16
#ifdef PICOLM_IMMA_W16
#define FA2_TILE_K 32
#else
#define FA2_TILE_K 16
#endif

__global__ void picolm_gpu_attention_prefill_fa2_kernel(
    float *xb_out, const float *q_dev,
    const uint16_t *kv_k, const uint16_t *kv_v,
    int layer_ordinal, int start_pos, int n_tokens,
    int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
    size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes);

/* ---- SSM alpha/beta batched vec_dot kernel ----
 * Each thread block handles one head. Weights: column-major
 * [dim, n_heads], each column is a quantized row.
 *
 * Thread-0 sequential accumulation to bit-match the CPU scalar
 * reference (model.c ssm_forward, steps 9-10). IMPORTANT: for
 * Q8_0/Q4_0-weight heads the CPU reference does NOT multiply the raw
 * float activation against the dequantized weight -- it quantizes xb
 * to Q8_0 ONCE per token (quantize_row_q8_0) and reuses that quantized
 * copy for every head's dot product via vec_dot_q8_0_q8_0_deltas
 * (Q8_0 weights) or vec_dot_q4_0_q8_0 (Q4_0 weights), both of which are
 * genuine int8 x int8 MACs. Multiplying the un-quantized float x
 * against the dequantized weight (the previous version of this kernel,
 * and thread-0 across accumulation order alone) is a DIFFERENT,
 * strictly more precise computation than what the CPU actually does,
 * and will not reproduce the CPU's output bit-for-bit -- the int8
 * round-trip on x is lossy and that lossiness is part of the reference.
 * We replicate that quantization here, redundantly once per head-block
 * (dim is at most a few thousand elements, this is cheap relative to
 * the model as a whole), so the GPU path takes the same lossy path the
 * CPU does. Only the F32 weight case multiplies against x directly,
 * matching vec_dot_f32_f32's scalar fallback. */
#define PICOLM_SSM_VECDOT_MAX_DIM 8192


__global__ void picolm_ssm_vecdot_kernel(float *out, const float *x, const void *weights, gguf_type_t qtype, int dim, int n_v_heads, int row_bytes, const int *head_map);

/* Batched-over-tokens version of the above: identical per-(token,head)
 * computation (including the same redundant per-block re-quantization
 * of x into Q8_0 -- kept for simplicity/correctness parity with the
 * per-token kernel; x is small enough that this isn't the bottleneck).
 * x is [n_tokens][dim], out is [n_tokens][n_v_heads]. */

__global__ void picolm_ssm_vecdot_batch_kernel(float *out, const float *x, const void *weights, gguf_type_t qtype, int dim, int n_v_heads, int n_tokens, int row_bytes, const int *head_map, int in_stride, int out_stride);

/* ---- SSM causal conv1d + state shift (fused) ----
 * Direct port of the CPU reference (ssm_forward, conv1d section): for
 * each channel co, sum d_conv taps (d_conv-1 from history in
 * conv_state, the newest tap from new_input), SiLU-activate, then
 * shift conv_state left and append new_input as the newest row.
 *
 * Embarrassingly parallel across conv_dim channels -- each thread
 * owns one channel end-to-end (read old state, compute, write output,
 * shift its own channel's history), no shared memory, no
 * synchronization needed at all. Genuinely new: no GPU conv1d existed
 * before this (CPU always did this step). */

__global__ void picolm_gpu_ssm_conv1d_kernel(float *conv_output, float *conv_state, const float *new_input, const float *conv1d_w, int conv_dim, int d_conv);

/* Host wrapper is below, after helper functions (find_ctx, gpu_ok, select_ctx).
 * See picolm_gpu_ssm_conv1d_dev definition further down. */

/* ---- SSM causal conv1d + state shift, BATCHED over tokens ----
 * Same computation as picolm_gpu_ssm_conv1d_kernel above, called
 * conceptually n_tokens times in sequence -- but each channel-thread
 * keeps its own small history window in registers across all n_tokens
 * instead of round-tripping through conv_state in global memory once
 * per token. conv_state is read once at the start and written once at
 * the end. d_conv is always small in practice (4 is typical for this
 * family of models); PICOLM_SSM_CONV_MAX_D_CONV is a generous cap --
 * the host wrapper below refuses to dispatch (returns 0, caller falls
 * back to the CPU/per-token path) if d_conv exceeds it, rather than
 * silently truncating history.
 *
 * Bit-exact with the per-token kernel run n_tokens times: hist[] is
 * updated with the identical shift-then-append order conv_state itself
 * would be, and the tap sum is accumulated in the identical order
 * (d=0..n_state_rows-1 then the new-sample tap), so this produces the
 * same float sequence, not just a mathematically equivalent one. */
#define PICOLM_SSM_CONV_MAX_D_CONV 16


__global__ void picolm_gpu_ssm_conv1d_batch_kernel(float *conv_output,  float *conv_state,  const float *new_input,  const float *conv1d_w,  int conv_dim, int d_conv, int n_tokens, int stride);

/* Host wrapper for the batched conv1d kernel is below, next to
 * picolm_gpu_ssm_conv1d_dev, for the same reason (needs find_ctx/gpu_ok). */

/* ---- SSM recurrence kernel ----
 *
 * One thread block per head, 256 threads per block.
 * Each head independently processes its d_state x d_state state block.
 * d_state can be up to 256 -> d_state^2 = 65536 elements.
 *
 * Steps per head:
 * 1. Decay: state *= gate_exp
 * 2. sk = state @ k (matrix-vector)
 * 3. d = (v - sk) * beta
 * 4. state += k * d^T (outer product)
 * 5. output = state @ q (matrix-vector), written dim-major
 */

/* ---- Warp-shuffle SSM recurrence kernel ----
 *
 * Row-parallel: each warp processes one row of the d_state x d_state
 * state matrix. Lanes within the warp distribute columns, and warp
 * shuffles perform the matvec reduction.
 *
 * Grid: (n_v_heads, 1, d_state/num_warps)
 * Block: (GPU_WARP_SIZE, num_warps, 1)
 * On CUDA (warp=32): block=(32,4,1)=128 threads, grid.z=32 for d_state=128
 * On HIP/gfx906 (wave=64): block=(64,4,1)=256 threads, grid.z=32 for d_state=128
 *
 * No shared memory, no syncthreads. All state in registers + warp shuffles.
 *
 * XOR shuffle: handled by gpu_shfl_xor_sync() wrapper, not via macro,
 * because HIP's __shfl_xor has different signature than CUDA's
 * __shfl_xor_sync. ROCm 6.x hipShflXorSync is only available when
 * HIP_ENABLE_WARP_SYNC_BUILTINS is defined. */

#if !defined(__HIP__)
__device__ inline float
gpu_shfl_xor_sync(float var, int laneMask, int width) {
    return __shfl_xor_sync(0xffffffff, var, laneMask, width);
}
#else
__device__ inline float
gpu_shfl_xor_sync(float var, int laneMask, int width) {
    return __shfl_xor(var, laneMask, width);
}
#endif

template <int warp_size>
__device__ inline float
picolm_warp_reduce_sum(float val) {
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        val += gpu_shfl_xor_sync(val, offset, warp_size);
    }
    return val;
}

/* NOTE (2026-08-11): This kernel was previously (32 lanes = 32 disjoint
 * column-blocks, one row per warp, XOR-shuffle tree reduction over all
 * 32 partials) row-parallel but used a completely different summation
 * tree from the CPU NEON reference (picolm_ssm_recurrence_kernel below):
 * NEON accumulates 4 *strided* partial sums (columns cb+0, cb+4, cb+8...
 * go into s0; cb+1,cb+5,... into s1; etc.) via serial fmaf, then combines
 * with exactly two pairwise adds: (s0+s1)+(s2+s3). A 32-wide tree over
 * differently-grouped partials is bit-different from that, and the
 * ~1e-7 per-row error it introduces compounds over 48 SSM layers into
 * wrong hidden states (see ssm_recurrence_verify.c / ssmreport.txt in
 * project notes -- gpu_fp64 vs cpu_neon was ALSO ~1.7e-7 off, proving
 * this is an order-of-operations mismatch, not a precision one).
 *
 * Fixed design: instead of 32 lanes cooperating on one row, split the
 * warp into GROUPS_PER_WARP groups of exactly 4 lanes each and give
 * every group its own row. Within a group, sub-lane `sub` (0..3) plays
 * the role of the CPU's s{sub} accumulator: it strides over columns
 * `vv*4 + sub` for vv in [0, S_v/4), doing the *same sequence* of fmaf
 * calls in the *same order* as the CPU loop. The two-step XOR-shuffle
 * reduction below (mask 1, then mask 2, both width-4) computes exactly
 * (s0+s1)+(s2+s3): float addition is commutative bit-for-bit (only
 * associativity/grouping affects rounding), so pairing lane 0 with lane
 * 1 first (mask=1) and then combining with the {2,3} pair (mask=2)
 * reproduces the CPU's exact grouping in every one of the 4 lanes.
 * This is bit-exact with the thread-0 kernel/CPU NEON path while still
 * processing GROUPS_PER_WARP rows concurrently per warp (8x on a
 * 32-wide warp) instead of thread-0's single-row-at-a-time serial loop
 * over all S_v rows in one thread.
 *
 * MUST be re-validated against ssm_recurrence_verify.c (gpu_fixed_step)
 * on real hardware before this dispatches in production -- this rewrite
 * has not been compiled/run on a GPU in this session. */
template <int S_v>
__global__ void
picolm_ssm_recurrence_warp_kernel(float *state,
                                   const float *q_conv,
                                   const float *k_conv,
                                   const float *v_conv,
                                   const float *gate_exp,
                                   const float *beta,
                                   float *ssm_output,
                                   int n_v_heads, int repeat) {
    constexpr int GROUP = 4;
    constexpr int GROUPS_PER_WARP = GPU_WARP_SIZE / GROUP;
    constexpr int d4 = S_v / GROUP;

    const int head = gpuBlockIdx_x;
    const int lane = gpuThreadIdx_x;
    const int group_in_warp = lane / GROUP;
    const int sub = lane % GROUP;
    const int row = (gpuBlockIdx_z * gpuBlockDim_y + gpuThreadIdx_y) * GROUPS_PER_WARP
                     + group_in_warp;
    if (head >= n_v_heads || row >= S_v) return;

    const int kh = head / repeat;
    const float ge = gate_exp[head];
    const float bh = beta[head];

    float *st = state + (size_t)head * S_v * S_v + (size_t)row * S_v;
    const float *khv = k_conv + (size_t)kh * S_v;
    const float *qh  = q_conv + (size_t)kh * S_v;
    const float v0 = v_conv[(size_t)head * S_v + row];

    float s_acc = 0.0f;
    for (int vv = 0; vv < d4; vv++) {
        int c = vv * GROUP + sub;
        float r = st[c] * ge;
        st[c] = r;
        s_acc = fmaf(r, khv[c], s_acc);
    }
    float t1 = s_acc + gpu_shfl_xor_sync(s_acc, 1, GROUP);
    float sk_row = t1 + gpu_shfl_xor_sync(t1, 2, GROUP);

    const float dv = (v0 - sk_row) * bh;

    float o_acc = 0.0f;
    for (int vv = 0; vv < d4; vv++) {
        int c = vv * GROUP + sub;
        float r = fmaf(khv[c], dv, st[c]);
        st[c] = r;
        o_acc = fmaf(r, qh[c], o_acc);
    }
    float u1 = o_acc + gpu_shfl_xor_sync(o_acc, 1, GROUP);
    float out_row = u1 + gpu_shfl_xor_sync(u1, 2, GROUP);

    if (sub == 0) {
        ssm_output[(size_t)row * n_v_heads + head] = out_row;
    }
}

/* Old thread-0 kernel (fallback for unsupported d_state values):
 * One thread block per head, 256 threads per block.
 */

__global__ void picolm_ssm_recurrence_kernel(float *state, const float *q_conv, const float *k_conv, const float *v_conv, const float *gate_exp, const float *beta, float *ssm_output, int n_v_heads, int d_state, int repeat);

/* ---- w4a16_matmul kernel (Tensor Core path) ----
 *
 * NVIDIA: WMMA via nvcuda::wmma namespace (sm_70+, CUDA 10+)
 * AMD: hipWMMA via wmma namespace (gfx940+, ROCm 6.x)
 *
 * On chips without Tensor Cores (e.g. gfx906, sm_60), these kernels
 * are simply not compiled. picolm_gpu_w4a16_mlp() returns 0 and the
 * caller falls back to quant_matmul.
 *
 * Four warps share one A tile and compute 16x64 outputs.
 * Grid: [ceil(N/64), ceil(M/16)], Block: 256 threads (4 warps)
 *
 * Weights are in GGUF block_q4_0 format (18 bytes per 32 values).
 * Dequant on-the-fly to FP16 for the WMMA B tile.
 */

/* HIP WMMA: only on CDNA2+ (gfx940, gfx941, gfx942). gfx906/908 have no WMMA.
 * The hip/wmma/wmma.h header is only available in ROCm >= 6.4. On older
 * ROCm releases WMMA is silently disabled so the non-WMA path is used. */
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#ifdef __has_include
#if __has_include(<hip/wmma/wmma.h>)
#define PICOLM_GPU_WMMA_AVAILABLE 1
#include <hip/wmma/wmma.h>
#endif
#endif
#endif
#endif

/* CUDA WMMA: sm_70+ (Volta and newer)
 * Guard must exclude host compilation phase to avoid CUDAFE stub generation
 * failures (wmma types are device-only). Host gets a forward declaration. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define PICOLM_GPU_WMMA_AVAILABLE 1
#endif

/* Forward declarations for host compilation (stubs only, no body) */
#ifndef PICOLM_GPU_WMMA_AVAILABLE
__global__ void picolm_w4a16_matmul(float *y, const float *x, const void *weights, int M, int K, int N, int block_size);
__global__ void picolm_w4a16_gate_up(float *gate, float *up, const float *x, const void *gate_weights, const void *up_weights, int M, int K, int N, int block_size);
#else /* PICOLM_GPU_WMMA_AVAILABLE */
/* WMMA kernel definitions stay in backend_gpu_kernels.cu */
__global__ void picolm_w4a16_matmul(float *y, const float *x, const void *weights, int M, int K, int N, int block_size);
__global__ void picolm_w4a16_gate_up(float *gate, float *up, const float *x, const void *gw, const void *uw, int M, int K, int N, int block_size);
#endif /* PICOLM_GPU_WMMA_AVAILABLE */

/* ---- Host-side data structures ---- */

struct picolm_gpu_tensor {
    void *weights;
    gguf_type_t qtype;
    int I, O, device;
    size_t row_bytes;
    int block_size;  /* bytes per quant block (18 for q4_0, 34 for q8_0, etc.) */
    int tracked;
    int zero_copy;   /* 1 if using cudaHostRegister (unified memory), 0 if copied */
};

 /* Forward declarations for __global__ kernels defined in host_ssm.cu
 * (needed by host_misc.cu which calls them) */
__global__ void picolm_gpu_ssm_gate_beta_kernel(float *, float *, const float *, const float *, const float *, const float *, int);
__global__ void picolm_gpu_ssm_l2norm_kernel(float *, int, int, float, float);
__global__ void picolm_gpu_ssm_l2norm_batch_kernel(float *, int, int, int, int, float, float);
__global__ void picolm_gpu_ssm_head_permute_kernel(float *, const float *, const int *, int, int);
__global__ void picolm_gpu_ssm_gated_norm_kernel(float *, const float *, const float *, const float *, const int *, int, int, float);
__global__ void picolm_gpu_ssm_gated_norm_batch_kernel(float *, const float *, const float *, const float *, const int *, int, int, int, float);
__global__ void picolm_gpu_ssm_prefill_gated_norm_kernel(float *, const float *, const float *, int, int, int, float, int, int);
__global__ void ssm_chunk_gather_qk_kernel(float *, float *, const float *, int, int, int, int, int, int);
__global__ void ssm_chunk_gather_v_kernel(float *, float *, float *, const float *, const float *, const float *, int, int, int, int, int, int, int);
__global__ void ssm_chunk_decay_kernel(float *, float *, float *, const float *, int, int, int);
__global__ void ssm_chunk_masked_gemm_kernel(float *, const float *, const float *, const float *, int, int, int, int);
__global__ void ssm_chunk_matvec_kernel(float *, const float *, const float *, int, int, int, int);
__global__ void ssm_chunk_veff_kernel(float *, const float *, const float *, const float *, const float *, int, int, int);
__global__ void ssm_chunk_trisolve_kernel(float *, const float *, const float *, const float *, int, int, int);
__global__ void ssm_chunk_output_kernel(float *, const float *, const float *, const float *, const float *, int, int, int);
__global__ void ssm_chunk_state_update_kernel(float *, const float *, const float *, const float *, int, int, int, int);
__global__ void ssm_chunk_scatter_kernel(float *, const float *, int, int, int, int, int);

#endif /* BACKEND_GPU_KERNELS_CUH */
