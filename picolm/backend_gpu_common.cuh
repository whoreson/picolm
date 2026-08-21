#ifndef BACKEND_GPU_COMMON_CUH
#define BACKEND_GPU_COMMON_CUH

/* backend_gpu.cu
 *
 * CUDA/HIP backend for PicoLM. Single source compiled with hipcc (ROCm) or
 * nvcc (CUDA). The .cu extension is standard for both - hipcc accepts .cu
 * natively, and nvcc of course requires it.
 *
 * Quantization format: uses PicoLM's GGUF block layouts (block_q4_0, block_q8_0,
 * block_q4_K) rather than Colibri's raw-packed nibbles. Each block has its own
 * per-block scale (FP16).
 *
 * The quant_matmul kernel handles all quant types via a device-side dequant
 * helper. It's not bandwidth-optimal but is correct and simple.
 *
 * The w4a16_matmul kernel uses Tensor Cores (WMMA on CUDA, hipWMMA on ROCm)
 * for int4 weights + FP16 activations. Only enabled on sm_70+/gfx9+.
 *
 * Platform detection:
 *   __HIP__             -> HIP device code (both AMD and NVIDIA HIPC)
 *   __HIP_PLATFORM_AMD__ -> ROCm/AMD specifically
 *   __CUDA_ARCH__        -> CUDA device code
 */

#include "backend_gpu.h"
#include <stdlib.h> /* setenv() */

#ifdef __HIP__
#include <hip/hip_runtime.h>
/* hipWMMA: available on CDNA2+ (gfx940+, gfx941+, gfx942+).
 * RDNA2 (gfx1030+) has limited WMMA. gfx906/908 (RDNA2) do NOT have hipWMMA.
 * For chips without WMMA, picolm_w4a16_mlp() returns 0 and the caller
 * falls back to the quant_matmul path (which is correct, just slower). */
#else
/* NVIDIA CUDA */
#include <cuda_runtime.h>
#include <mma.h>
#include <sm_61_intrinsics.h> /* __dp4a: 4-way int8 MAC, sm_61+ */
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <synchapi.h>
#else
#include <pthread.h>
#endif

/* ---- Platform abstraction ---- */

#ifdef __HIP_PLATFORM_AMD__
#define gpuSuccess hipSuccess
#define gpuError_t hipError_t
#define gpuGetErrorString hipGetErrorString
#define gpuSetDevice hipSetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceProp hipDeviceProp_t
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDefault hipMemcpyDefault
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuStream_t hipStream_t
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuMemGetInfo hipMemGetInfo
#define gpuMallocHost hipHostMalloc
#define gpuFreeHost hipHostFree
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuGetLastError hipGetLastError
#define gpuLaunchBlockPerMultiprocessor hipDeviceAttributeMaxThreadsPerMultiProcessor
#define gpuSyncthreads __syncthreads
#define gpuShflSync hipShflSync
#define gpuShflUpSync hipShflUpSync
#define gpuDeviceSynchronize hipDeviceSynchronize
typedef hipDeviceProp_t gpuDeviceProp_t;
typedef hipError_t gpuError_t;
typedef hipStream_t gpuStream_t;
typedef hipEvent_t gpuEvent_t;
#define gpuDevice hipDevice
#define gpuThreadIdx_x hipThreadIdx_x
#define gpuThreadIdx_y hipThreadIdx_y
#define gpuBlockIdx_x hipBlockIdx_x
#define gpuBlockIdx_y hipBlockIdx_y
#define gpuBlockIdx_z hipBlockIdx_z
#define gpuBlockDim_x hipBlockDim_x
#define gpuBlockDim_y hipBlockDim_y
#define gpuBlockDim_z hipBlockDim_z
#define gpuGridDim_x hipGridDim_x
#define gpuGridDim_y hipGridDim_y
#define gpuHostRegister hipHostRegister
#define gpuHostUnregister hipHostUnregister
#define gpuMallocManaged hipMallocManaged
#define gpuMemset hipMemset
#define gpuMemsetAsync hipMemsetAsync
#define gpuFuncSetAttribute hipFuncSetAttribute
#define gpuFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
/* HIP: no hipMemAdvise equivalent; unified memory is automatic on HIP */
#define GPU_WARP_SIZE __AMDGCN_WAVEFRONT_SIZE
#else
/* NVIDIA CUDA */
#define gpuSuccess cudaSuccess
#define gpuError_t cudaError_t
#define gpuGetErrorString cudaGetErrorString
#define gpuSetDevice cudaSetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceProp cudaDeviceProp
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDefault cudaMemcpyDefault
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuStream_t cudaStream_t
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuMemGetInfo cudaMemGetInfo
#define gpuFuncSetAttribute cudaFuncSetAttribute
#define gpuFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
#define gpuMallocHost cudaMallocHost
#define gpuFreeHost cudaFreeHost
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuSyncthreads __syncthreads
#define gpuThreadIdx_x threadIdx.x
#define gpuThreadIdx_y threadIdx.y
#define gpuBlockIdx_x blockIdx.x
#define gpuBlockIdx_y blockIdx.y
#define gpuBlockIdx_z blockIdx.z
#define gpuBlockDim_x blockDim.x
#define gpuBlockDim_y blockDim.y
#define gpuBlockDim_z blockDim.z
#define gpuGridDim_x gridDim.x
#define gpuGridDim_y gridDim.y
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuShflSync __shfl_sync
#define gpuShflUpSync __shfl_up_sync
#define gpuLaunchBlockPerMultiprocessor cudaDevAttrMaxThreadsPerMultiProcessor
#define gpuDevice cudaDevice
#define gpuHostRegister cudaHostRegister
#define gpuHostUnregister cudaHostUnregister
#define gpuMallocManaged cudaMallocManaged
#define gpuMemset cudaMemset
#define gpuMemsetAsync cudaMemsetAsync
#define gpuMemAdvise cudaMemAdvise
#define gpuMemLocation cudaMemLocation
#define gpuMemLocationType cudaMemLocationType
#define gpuCudaMemLocationTypeDevice cudaMemLocationTypeDevice
#define gpuCudaMemLocationTypecpu cudaMemLocationTypecpu
#define gpuCudaMemAdviseSetPreferredLocation cudaMemAdviseSetPreferredLocation
#define GPU_WARP_SIZE 32
#endif

/* Cross-platform unused attribute */
#ifdef _WIN32
#define PICOLM_UNUSED
#else
#define PICOLM_UNUSED __attribute__((unused))
#endif

/* ---- Device-side FP16 helpers ---- */

#ifdef __HIP_PLATFORM_AMD__
#ifdef __HIP__
#include <hip/hip_fp16.h>
__host__ __device__ static inline float gpu_fp16_to_fp32(uint16_t h) {
    return (float)__ushort_as_half(h);
}
__host__ __device__ static inline uint16_t gpu_fp32_to_fp16(float f) {
    return __half_as_ushort(__float2half(f));
}
#else
__host__ __device__ static inline float gpu_fp16_to_fp32(uint16_t h) {
    return (float)__half_raw(h);
}
__host__ __device__ static inline uint16_t gpu_fp32_to_fp16(float f) {
    return __half_as_ushort(__float2half(f));
}
#endif
#else
/* CUDA: __half is a native type */
__host__ __device__ static inline float gpu_fp16_to_fp32(unsigned short h) {
    return __half2float(__ushort_as_half(h));
}
__host__ __device__ PICOLM_UNUSED static inline unsigned short gpu_fp32_to_fp16(float f) {
    return __half_as_ushort(__float2half(f));
}
#endif

/* ---- Device-side quantization block dequantization ----
 *
 * These mirror PicoLM's block layouts in quant.h.
 * All blocks have FP16 scales (uint16_t d field).
 */

/* Block sizes in bytes (from quant.h structs) */
#define GPU_BLOCK_Q4_0_SIZE  18  /* uint16_t d + uint8_t qs[16] */
#define GPU_BLOCK_Q2_K_SIZE  84  /* block_q2_K: scales[16]+qs[64]+d+dm */
#define GPU_BLOCK_Q4_K_SIZE  144 /* block_q4_K from quant.h */
#define GPU_BLOCK_Q5_K_SIZE  176 /* block_q5_K: d+dm+scales[12]+qh[32]+qs[128] */
#define GPU_BLOCK_Q6_K_SIZE  210 /* block_q6_K: ql[128]+qh[64]+scales[16]+d[2] */
#define GPU_BLOCK_Q8_0_SIZE  34  /* uint16_t d + int8_t qs[32] */

/* block_q4_0: 18 bytes = 2B scale(FP16) + 16B qs (32 values)
 * GGUF nibble layout (per vec_dot_q4_0_f32 in quant.c):
 *   qs[0..15] low nibble  -> values 0..15  (qs[j] & 0xF)
 *   qs[0..15] high nibble -> values 16..31  (qs[j] >> 4)
 * This is NOT interleaved! All low nibbles come first, then high nibbles.
 * The old interleaved reading (low0,high0,low1,high1,...) was WRONG
 * and produced garbage output. */

/* ---- Forward declarations for cross-module host calls ---- */
typedef struct {
    int device; int compute_major, compute_minor;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    float *host_x, *host_y;
    size_t host_x_cap, host_y_cap;
    int8_t *q8_xq; float *q8_xd;
    size_t q8_xq_cap, q8_xd_cap;
    gpuStream_t stream;
    size_t tensor_count, tensor_bytes;
    float *pipe_x, *pipe_xb, *pipe_q, *pipe_k, *pipe_v,
          *pipe_attn_out, *pipe_ffn_norm, *pipe_gate, *pipe_up;
    int pipe_ready;
    float *attn_partial; size_t attn_partial_cap;
    float *pipe_x_b, *pipe_xb_b, *pipe_q_b, *pipe_k_b, *pipe_v_b,
          *pipe_attn_out_b, *pipe_ffn_norm_b, *pipe_gate_b, *pipe_up_b;
    int pipe_b_ready;
    float *ssm_qkv_raw, *ssm_conv_out, *ssm_xb2, *ssm_xb2_remap,
          *ssm_v_remap, *ssm_alpha_raw, *ssm_beta_raw, *ssm_gate_exp,
          *ssm_beta, *ssm_output, *ssm_final_output;
    int ssm_ready;
    float *ssm_prefill_scratch; size_t ssm_prefill_scratch_cap;
    void *rmsnorm_w_dev[64]; const void *rmsnorm_w_keys[64];
    int rmsnorm_w_n;
} gpu_device_ctx_t;
extern gpu_device_ctx_t g_gpu_ctx[PICOLM_GPU_MAX_DEVICES];
extern int g_nctx;
static inline int gpu_ok(gpuError_t err, const char *what) {
    if (err != gpuSuccess) {
        fprintf(stderr, "[GPU] %s: %d %s\n", what, (int)err, gpuGetErrorString(err));
        return 0;
    }
    return 1;
}

static inline gpu_device_ctx_t *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++)
        if (g_gpu_ctx[i].device == device) return &g_gpu_ctx[i];
    return NULL;
}

static inline int select_ctx(gpu_device_ctx_t *ctx) {
    if (!ctx) return 0;
    return gpuSetDevice(ctx->device) == gpuSuccess;
}

/* ---- Mutex for device context scratch buffer resize ---- */
#ifdef _WIN32
extern SRWLOCK g_resize_mutex;
#define gpu_mutex_lock()   AcquireSRWLockExclusive(&g_resize_mutex)
#define gpu_mutex_unlock() ReleaseSRWLockExclusive(&g_resize_mutex)
#else
extern pthread_mutex_t g_resize_mutex;
#define gpu_mutex_lock()   pthread_mutex_lock(&g_resize_mutex)
#define gpu_mutex_unlock() pthread_mutex_unlock(&g_resize_mutex)
#endif

/* ---- Forward declarations for cross-module host functions ---- */
extern int reserve(float **ptr, size_t *cap, size_t bytes);
PICOLM_UNUSED extern int reserve_pinned(float **ptr, size_t *cap, size_t bytes);
extern int reserve_i8(int8_t **ptr, size_t *cap, size_t bytes);
extern int ssm_batch_scratch_ensure(void **buf, size_t *cap, size_t need);

#endif /* BACKEND_GPU_COMMON_CUH */
