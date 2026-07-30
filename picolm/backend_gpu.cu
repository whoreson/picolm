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
#define gpuBlockIdx_x hipBlockIdx_x
#define gpuBlockIdx_y hipBlockIdx_y
#define gpuBlockIdx_z hipBlockIdx_z
#define gpuBlockDim_x hipBlockDim_x
#define gpuGridDim_x hipGridDim_x
#define gpuGridDim_y hipGridDim_y
#define gpuHostRegister hipHostRegister
#define gpuHostUnregister hipHostUnregister
#define gpuMallocManaged hipMallocManaged
#define gpuMemset hipMemset
#define gpuFuncSetAttribute hipFuncSetAttribute
#define gpuFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
/* HIP: no hipMemAdvise equivalent; unified memory is automatic on HIP */
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
#define gpuBlockIdx_x blockIdx.x
#define gpuBlockIdx_y blockIdx.y
#define gpuBlockIdx_z blockIdx.z
#define gpuBlockDim_x blockDim.x
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
#define gpuMemAdvise cudaMemAdvise
#define gpuMemLocation cudaMemLocation
#define gpuMemLocationType cudaMemLocationType
#define gpuCudaMemLocationTypeDevice cudaMemLocationTypeDevice
#define gpuCudaMemLocationTypecpu cudaMemLocationTypecpu
#define gpuCudaMemAdviseSetPreferredLocation cudaMemAdviseSetPreferredLocation
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
#define GPU_BLOCK_Q4_K_SIZE  144 /* uint16_t d + dmin + uint8_t scales[12] + qs[128] */
#define GPU_BLOCK_Q8_0_SIZE  34  /* uint16_t d + int8_t qs[32] */
#define GPU_BLOCK_Q4_K_SIZE 144  /* block_q4_K from quant.h */

/* block_q4_0: 18 bytes = 2B scale(FP16) + 16B qs (32 values)
 * GGUF nibble layout (per vec_dot_q4_0_f32 in quant.c):
 *   qs[0..15] low nibble  -> values 0..15  (qs[j] & 0xF)
 *   qs[0..15] high nibble -> values 16..31  (qs[j] >> 4)
 * This is NOT interleaved! All low nibbles come first, then high nibbles.
 * The old interleaved reading (low0,high0,low1,high1,...) was WRONG
 * and produced garbage output. */
__device__ static inline float dequant_q4_0(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    const uint8_t *qs = b + 2;
    int v;
    if (i < 16) {
        v = qs[i] & 0xF;           /* low nibble: values 0-15 */
    } else {
        v = qs[i - 16] >> 4;       /* high nibble: values 16-31 */
    }
    return (float)(v - 8) * d;
}

/* block_q8_0: 34 bytes = 2B scale(FP16) + 32B qs (32 values)
 * dequant(i) = qs[i] * d */
__device__ static inline float dequant_q8_0(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    return (float)((const int8_t *)(b + 2))[i] * d;
}

/* block_q1_0: 18 bytes = 2B scale(FP16) + 16B qs (128 sign bits)
 * dequant(i) = (bit[i] ? +d : -d) */
__device__ static inline float dequant_q1_0(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    int bit = (b[2 + (i >> 3)] >> (i & 7)) & 1;
    return bit ? d : -d;
}

/* block_q2_0: 34 bytes = 2B scale(FP16) + 32B qs (128 values * 2 bits each)
 * dequant(i) = (qs[i] - 1) * d, where qs[i] in {0,1,2,3} -> {-d, 0, +d, +2d} */
__device__ static inline float dequant_q2_0(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    int byte_idx = i >> 2;          /* 4 values per byte */
    int shift = (i & 3) << 1;       /* 2 bits per value */
    int v = (b[2 + byte_idx] >> shift) & 3;
    return (float)(v - 1) * d;
}

/* F16: raw array of uint16_t, each value is an individual FP16 element.
 * No block structure: dequant(i) = gpu_fp16_to_fp32(weights[i]) */
__device__ static inline float dequant_f16(const void *weights, int i) {
    const uint16_t *w = (const uint16_t *)weights;
    return gpu_fp16_to_fp32(w[i]);
}

/* block_q4_K: 144 bytes for 256 values
 * Layout from quant.h:
 *   uint16_t d;          offset 0  super-block scale (FP16)
 *   uint16_t dmin;       offset 2  super-block min (FP16)
 *   uint8_t  scales[12]; offset 4  packed 6-bit scales+mins (8 pairs)
 *   uint8_t  qs[128];    offset 16 quantized values (256 nibbles)
 *
 * 256 values = 4 superblocks of 64 values each.
 * Each superblock has 2 sub-blocks of 32 values, each with its own scale+min.
 * dequant(i) = d * scale[j] + qs[i] - dmin * min[j]
 * where qs[i] is a raw nibble [0..15], NOT offset by -8.
 *
 * get_scale_min_k4(j, scales, &sc, &mn) extracts scale and min for sub-block j.
 * j=0..7 maps to 8 sub-blocks across 4 superblocks. */
__device__ static inline void gpu_get_scale_min_k4(int j, const uint8_t *scales, uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *mn = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

__device__ static inline float dequant_q4_K(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    uint16_t dmin_raw = b[2] | ((uint16_t)b[3] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    float dmin = gpu_fp16_to_fp32(dmin_raw);
    const uint8_t *scales = b + 4;
    const uint8_t *qs = b + 16;

    int superblk = i / 64;
    int subblk = (i % 64) / 32;
    int offset_in_sub = i % 32;
    int scale_idx = superblk * 2 + subblk;

    uint8_t sc, mn;
    gpu_get_scale_min_k4(scale_idx, scales, &sc, &mn);

    int qs_byte_idx = superblk * 32 + offset_in_sub;
    int v = (qs_byte_idx < 128) ? (subblk == 0 ? (qs[qs_byte_idx] & 0xF) : (qs[qs_byte_idx] >> 4)) : 0;

    return d * (float)sc * (float)v - dmin * (float)mn;
}

/* ---- quant_matmul kernel ----
 *
 * Simple, correct GEMV kernel for all quant formats.
 * 1 thread per output element, shared-memory tree reduce.
 * Not bandwidth-optimal (dequantizes weights on-the-fly) but universally correct.
 *
 * Grid: [O, S], Block: 256 threads
 * Each thread computes y[s*O + o] = sum_i x[s*I + i] * W[o][i]
 */

__global__ void
picolm_quant_matmul(float *y, const float *x, const void *weights,
                    gguf_type_t qtype, int S, int I, int O, int row_bytes) {
    /* bytes_per_block: stride between consecutive blocks in memory */
    int bytes_per_block;
    switch (qtype) {
        case GGUF_TYPE_F32:  bytes_per_block = 4; break;      /* 1 float */
        case GGUF_TYPE_F16:  bytes_per_block = 2; break;      /* 1 uint16_t */
        case 30:             bytes_per_block = 2; break;      /* BF16 */
        case GGUF_TYPE_Q4_0: bytes_per_block = GPU_BLOCK_Q4_0_SIZE; break;  /* 18 */
        case GGUF_TYPE_Q8_0: bytes_per_block = GPU_BLOCK_Q8_0_SIZE; break;  /* 34 */
        case GGUF_TYPE_Q4_K: bytes_per_block = GPU_BLOCK_Q4_K_SIZE; break;  /* 144 */
        case 41:             bytes_per_block = 18; break;      /* Q1_0 */
        case 42:             bytes_per_block = 34; break;      /* Q2_0 */
        default: bytes_per_block = 18; break;
    }
    int o = gpuBlockIdx_x;
    int s = gpuBlockIdx_y;
    if (o >= O || s >= S) return;

    double sum = 0.0;
    const char *wrow = (const char *)weights + (size_t)o * row_bytes;

    switch (qtype) {
    case 0: /* GGUF_TYPE_F32 */
        for (int i = gpuThreadIdx_x; i < I; i += gpuBlockDim_x) {
            sum += x[(size_t)s * I + i] * ((const float *)(wrow))[i];
        }
        break;

    case 1: /* GGUF_TYPE_F16 */
        /* Raw FP16 array, no block structure. Dequant on-the-fly. */
        for (int i = gpuThreadIdx_x; i < I; i += gpuBlockDim_x) {
            sum += x[(size_t)s * I + i] * dequant_f16(wrow, i);
        }
        break;

    case 2: /* GGUF_TYPE_Q4_0 */
        /* I values in 32-value blocks (18 bytes each) */
        {
            int n_blocks = I / 32;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 32; j++) {
                    int i = bi * 32 + j;
                    sum += x[(size_t)s * I + i] * dequant_q4_0(blk, j);
                }
            }
        }
        break;

    case 8: /* GGUF_TYPE_Q8_0 */
        /* I values in 32-value blocks (34 bytes each) */
        {
            int n_blocks = I / 32;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 32; j++) {
                    int i = bi * 32 + j;
                    sum += x[(size_t)s * I + i] * dequant_q8_0(blk, j);
                }
            }
        }
        break;

    case 12: /* GGUF_TYPE_Q4_K */
        /* 256 values per block (144 bytes). Per-element dequant via helper. */
        {
            int n_blocks = I / 256;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 256; j++) {
                    int i = bi * 256 + j;
                    sum += x[(size_t)s * I + i] * dequant_q4_K(blk, j);
                }
            }
        }
        break;

    case 41: /* GGUF_TYPE_Q1_0 */
        /* 128 values per block, 18 bytes */
        {
            int n_blocks = I / 128;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 128; j++) {
                    int i = bi * 128 + j;
                    sum += x[(size_t)s * I + i] * dequant_q1_0(blk, j);
                }
            }
        }
        break;

    case 42: /* GGUF_TYPE_Q2_0 */
        /* 128 values per block, 34 bytes */
        {
            int n_blocks = I / 128;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 128; j++) {
                    int i = bi * 128 + j;
                    sum += x[(size_t)s * I + i] * dequant_q2_0(blk, j);
                }
            }
        }
        break;

    default:
        break;
    }

    /* Shared-memory tree reduce with double precision */
    __shared__ double partial[256];
    partial[gpuThreadIdx_x] = sum;
    gpuSyncthreads();
    for (int n = gpuBlockDim_x >> 1; n; n >>= 1) {
        if (gpuThreadIdx_x < n)
            partial[gpuThreadIdx_x] += partial[gpuThreadIdx_x + n];
        gpuSyncthreads();
    }
    if (!gpuThreadIdx_x)
        y[(size_t)s * O + o] = (float)partial[0];
}

/* ---- GPU-side Q8_0 quantization kernel ----
 *
 * Mirrors CPU quantize_row_q8_0 in quant.c exactly.
 * Per 32-element block: d = max(|x|) / 127, qs[i] = round(x[i] / d).
 * Output: int8_t qs[32] + uint16_t d (FP16) per block = 34 bytes.
 *
 * Grid: [S, n_blocks_per_row], Block: 32 threads.
 * Each block quantizes one 32-element chunk of one input row. */
__global__ void
picolm_quantize_q8_0(int8_t *qs_out, float *d_out,
                      const float *x, int I, int S) {
    int s = gpuBlockIdx_y;           /* row index */
    int block = gpuBlockIdx_x;       /* 32-value block within row */
    int tid = gpuThreadIdx_x;        /* 0..31 */
    int n_blocks = (I + 31) / 32;
    if (block >= n_blocks || s >= S) return;

    const float *xb = x + (size_t)s * I + (size_t)block * 32;

    __shared__ float amax_shared[32];
    float v = (tid < I - block * 32) ? xb[tid] : 0.0f;
    amax_shared[tid] = fabsf(v);
    gpuSyncthreads();

    /* Reduce max across 32 threads */
    for (int stride = 16; stride; stride >>= 1) {
        if (tid < stride)
            amax_shared[tid] = fmaxf(amax_shared[tid], amax_shared[tid + stride]);
        gpuSyncthreads();
    }

    float amax = amax_shared[0];
    float d = amax / 127.0f;
    float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    int q = (int)lrintf(v * id);
    /* Clamp to [-128, 127] */
    if (q > 127) q = 127;
    if (q < -128) q = -128;

    qs_out[(size_t)s * (size_t)n_blocks * 32 + (size_t)block * 32 + tid] = (int8_t)q;
    if (tid == 0)
        d_out[(size_t)s * (size_t)n_blocks + block] = d;
}

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
__global__ void
picolm_q8_q8_matmul(float *y,
                     const int8_t *xq, const float *xd,
                     const void *weights,
                     int S, int I, int O, int row_bytes) {
    int o = gpuBlockIdx_x;
    int s = gpuBlockIdx_y;
    if (o >= O || s >= S) return;

    int n_blocks = I / 32;
    const char *wrow = (const char *)weights + (size_t)o * row_bytes;
    const int8_t *xrow = xq + (size_t)s * I;
    const float *xdrow = xd + (size_t)s * n_blocks;

    double sum = 0.0;
    for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
        const uint8_t *b = (const uint8_t *)wrow + (size_t)bi * GPU_BLOCK_Q8_0_SIZE;
        uint16_t wd_raw = b[0] | ((uint16_t)b[1] << 8);
        float wd = gpu_fp16_to_fp32(wd_raw);
        float xdv = xdrow[bi];

        const int8_t *wq = (const int8_t *)(b + 2);
        const int8_t *xqp = xrow + bi * 32;

        int32_t acc = 0;
#if 0 /* __dp4a: 4-way int8 MAC (sm_61+). Verified bit-exact but on GB10 (sm_121)
       * ptxas lowers __dp4a to scalar IMAD -- no hardware DP4A on Blackwell.
       * Kept in #if 0 for architectures that have it (e.g. Turing/Ampere
       * where __dp4a maps to real DP4A instructions). Guarded to avoid
       * the memcpy overhead on platforms where it doesn't help. */
#ifndef __HIP__
        {
            int32_t wq4[8], xq4[8];
            memcpy(wq4, wq, 32);
            memcpy(xq4, xqp, 32);
#pragma unroll
            for (int j = 0; j < 8; j++)
                acc = __dp4a(wq4[j], xq4[j], acc);
        }
#else
        for (int j = 0; j < 32; j++)
            acc += (int32_t)wq[j] * (int32_t)xqp[j];
#endif
#else
        for (int j = 0; j < 32; j++)
            acc += (int32_t)wq[j] * (int32_t)xqp[j];
#endif

        sum += (double)acc * (double)wd * (double)xdv;
    }

    __shared__ double partial[256];
    partial[gpuThreadIdx_x] = sum;
    gpuSyncthreads();
    for (int n = gpuBlockDim_x >> 1; n; n >>= 1) {
        if (gpuThreadIdx_x < n)
            partial[gpuThreadIdx_x] += partial[gpuThreadIdx_x + n];
        gpuSyncthreads();
    }
    if (!gpuThreadIdx_x)
        y[(size_t)s * O + o] = (float)partial[0];
}

/* Number of sequence positions per tile in the tiled Q8_0 matmul.
 * Shared memory usage (row_bytes, up to ~15KB for a 14336-wide FFN row)
 * doesn't depend on this constant -- positions are handled in an inner
 * loop, not concurrently. Stays well under the default 48KB/block. */
#define Q8_TILE_S 32

/* Tiled Q8_0 matmul: loads each weight row into shared memory once per
 * tile of Q8_TILE_S query positions, then reuses it in the inner loop.
 * Original reads the same weight row from global memory independently
 * for every (output_row, s) pair -- for S=291 prefill, that's 291
 * redundant global reads of the same row.
 * Same arithmetic, same accumulation order per (o,s) as original --
 * bit-exact. For S=1 (decode) reduces to one tile of size 1. */
__global__ void
picolm_q8_q8_matmul_tiled(float *y,
                           const int8_t *xq, const float *xd,
                           const void *weights,
                           int S, int I, int O, int row_bytes) {
    int o = gpuBlockIdx_x;
    int tile = gpuBlockIdx_y;
    if (o >= O) return;
    int s0 = tile * Q8_TILE_S;
    if (s0 >= S) return;
    int s_count = min(Q8_TILE_S, S - s0);

    int n_blocks = I / 32;
    const uint8_t *wrow = (const uint8_t *)weights + (size_t)o * row_bytes;

    /* Load weight row into shared memory (dynamic portion) */
    extern __shared__ uint8_t wrow_sh[];
    for (int b = gpuThreadIdx_x; b < row_bytes; b += gpuBlockDim_x) {
        wrow_sh[b] = wrow[b];
    }
    gpuSyncthreads();

    __shared__ double partial[256];

    for (int ls = 0; ls < s_count; ls++) {
        int s = s0 + ls;
        const int8_t *xrow = xq + (size_t)s * I;
        const float *xdrow = xd + (size_t)s * n_blocks;

        double sum = 0.0;
        for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
            const uint8_t *b = wrow_sh + (size_t)bi * GPU_BLOCK_Q8_0_SIZE;
            uint16_t wd_raw = b[0] | ((uint16_t)b[1] << 8);
            float wd = gpu_fp16_to_fp32(wd_raw);
            float xdv = xdrow[bi];

            const int8_t *wq = (const int8_t *)(b + 2);
            const int8_t *xqp = xrow + bi * 32;

            int32_t acc = 0;
#if 0 /* __dp4a: see picolm_q8_q8_matmul comment above */
#ifndef __HIP__
            {
                int32_t wq4[8], xq4[8];
                memcpy(wq4, wq, 32);
                memcpy(xq4, xqp, 32);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    acc = __dp4a(wq4[j], xq4[j], acc);
            }
#else
            for (int j = 0; j < 32; j++)
                acc += (int32_t)wq[j] * (int32_t)xqp[j];
#endif
#else
            for (int j = 0; j < 32; j++)
                acc += (int32_t)wq[j] * (int32_t)xqp[j];
#endif

            sum += (double)acc * (double)wd * (double)xdv;
        }

        partial[gpuThreadIdx_x] = sum;
        gpuSyncthreads();
        for (int n = gpuBlockDim_x >> 1; n; n >>= 1) {
            if (gpuThreadIdx_x < n)
                partial[gpuThreadIdx_x] += partial[gpuThreadIdx_x + n];
            gpuSyncthreads();
        }
        if (!gpuThreadIdx_x)
            y[(size_t)s * O + o] = (float)partial[0];
        gpuSyncthreads(); /* clean up before next ls iteration */
    }
}

/* ---- silu_mul kernel ----
 * Element-wise: gate[i] = gate[i] / (1 + exp(-gate[i])) * up[i] */
__global__ void
picolm_silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

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
__global__ void
picolm_gpu_attention_decode_kernel(
        float *xb_out,
        const float *q_dev,
        const uint16_t *kv_k,
        const uint16_t *kv_v,
        int layer_ordinal,
        int pos,
        int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
        size_t kv_pos_stride_bytes,
        size_t kv_head_stride_bytes)
{
    int kv_h = (int)gpuBlockIdx_x;
    if (kv_h >= n_kv_heads) return;

    int kv_mul = n_heads / n_kv_heads;
    int first_qh = kv_h * kv_mul;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    size_t layer_base = (size_t)layer_ordinal * max_seq_len * n_kv_heads * head_dim;

    /* Shared memory: K + V (u16), reduce area (256 float), then the
     * accumulator [kv_mul][head_dim] float, then max_score[kv_mul] and
     * sum_exp[kv_mul] float. Sized by the host wrapper. */
    extern __shared__ uint8_t smem[];
    uint16_t *k_sh = (uint16_t *)smem;
    uint16_t *v_sh = k_sh + head_dim;
    float *reduce_sh = (float *)((uint8_t *)v_sh + head_dim * sizeof(uint16_t));
    float *acc_sh = reduce_sh + 256;
    float *max_score_sh = acc_sh + (size_t)kv_mul * head_dim;
    float *sum_exp_sh = max_score_sh + kv_mul;

    __shared__ float rescale_sh, weight_sh;

    for (int i = tid; i < kv_mul * head_dim; i += n_threads) acc_sh[i] = 0.0f;
    for (int g = tid; g < kv_mul; g += n_threads) {
        max_score_sh[g] = -1e30f;
        sum_exp_sh[g] = 0.0f;
    }
    gpuSyncthreads();

    float sqrt_hd = sqrtf((float)head_dim);

    for (int t = 0; t <= pos; t++) {
        size_t k_off = layer_base + (size_t)t * kv_pos_stride_bytes / 2 + kv_h * kv_head_stride_bytes / 2;
        size_t v_off = k_off;
        for (int d = tid; d < head_dim; d += n_threads) {
            k_sh[d] = kv_k[k_off + d];
            v_sh[d] = kv_v[v_off + d];
        }
        gpuSyncthreads();

        for (int g = 0; g < kv_mul; g++) {
            const float *qg = q_dev + (size_t)(first_qh + g) * head_dim;
            float score = 0.0f;
            for (int d = tid; d < head_dim; d += n_threads) {
                score += qg[d] * gpu_fp16_to_fp32(k_sh[d]);
            }
            reduce_sh[tid] = score;
            gpuSyncthreads();
            for (int s = n_threads / 2; s > 0; s >>= 1) {
                if (tid < s) reduce_sh[tid] += reduce_sh[tid + s];
                gpuSyncthreads();
            }
            score = reduce_sh[0] / sqrt_hd;

            if (tid == 0) {
                if (score > max_score_sh[g]) {
                    rescale_sh = expf(max_score_sh[g] - score);
                    weight_sh = 1.0f;
                    sum_exp_sh[g] = sum_exp_sh[g] * rescale_sh + 1.0f;
                    max_score_sh[g] = score;
                } else {
                    rescale_sh = 1.0f;
                    weight_sh = expf(score - max_score_sh[g]);
                    sum_exp_sh[g] += weight_sh;
                }
            }
            gpuSyncthreads();

            float *accg = acc_sh + (size_t)g * head_dim;
            for (int d = tid; d < head_dim; d += n_threads) {
                accg[d] = accg[d] * rescale_sh + weight_sh * gpu_fp16_to_fp32(v_sh[d]);
            }
            gpuSyncthreads(); /* reduce_sh/rescale_sh/weight_sh reused next g/t iter */
        }
    }

    /* Normalize and write output, parallelized across threads */
    for (int g = 0; g < kv_mul; g++) {
        float inv_sum = 1.0f / sum_exp_sh[g];
        float *xbhg = xb_out + (size_t)(first_qh + g) * head_dim;
        float *accg = acc_sh + (size_t)g * head_dim;
        for (int d = tid; d < head_dim; d += n_threads) {
            xbhg[d] = accg[d] * inv_sum;
        }
    }
}

/* ---- Prefill tiled attention kernel ----
 * Processes one (query_head, token_tile) pair.
 * Tiled over KV positions with online-softmax merge.
 *
 * Grid: [n_heads, ceil(n_tokens / TOKEN_TILE)]
 * Each block handles one head and one tile of query tokens.
 * Inner loop tiles over KV positions. */
#define ATTN_TILE_K 64
#define ATTN_TILE_Q 32

__global__ void
picolm_gpu_attention_prefill_kernel(
        float *xb_out,        /* [n_tokens][n_heads][head_dim] */
        const float *q_dev,   /* [n_tokens][n_heads][head_dim] */
        const uint16_t *kv_k, /* [layer][pos][kv_head][head_dim] FP16 */
        const uint16_t *kv_v, /* [layer][pos][kv_head][head_dim] FP16 */
        int layer_ordinal,
        int start_pos, int n_tokens,
        int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
        size_t kv_pos_stride_bytes,
        size_t kv_head_stride_bytes)
{
    int h = (int)gpuBlockIdx_x;       /* query head */
    int tile_q_idx = (int)gpuBlockIdx_y; /* tile of query tokens */
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    int kv_h = h / (n_heads / n_kv_heads);
    float sqrt_hd = sqrtf((float)head_dim);

    /* Layer base offset in uint16_t */
    size_t layer_base = (size_t)layer_ordinal * max_seq_len * n_kv_heads * head_dim;

    /* Which query tokens does this block handle? */
    int q_start = tile_q_idx * ATTN_TILE_Q;
    int q_end = min(q_start + ATTN_TILE_Q, n_tokens);
    int n_q = q_end - q_start;

    /* Shared memory: K tile + V tile (u16) + reduce (256 float) +
     * acc[ATTN_TILE_Q][head_dim] (float) + max_score[ATTN_TILE_Q] +
     * sum_exp[ATTN_TILE_Q] (float). Sized by the host wrapper, which
     * also opts into extended dynamic shared memory since this total
     * exceeds the default 48KB for typical head_dim/tile sizes.
     *
     * Previously acc/max_score/sum_exp were per-thread local arrays
     * (acc[32][256] = 32KB/thread) -- spilled to local memory, declared
     * and zero-initialized redundantly by every one of n_threads
     * threads, and only ever read/written by thread 0 in the update
     * loop below while every other thread idled at the barrier. Same
     * bug class as the decode attention kernel, fixed the same way:
     * one shared copy, update parallelized across threads. */
    extern __shared__ uint8_t smem[];
    uint16_t *k_tile = (uint16_t *)smem;
    uint16_t *v_tile = k_tile + ATTN_TILE_K * head_dim;
    float *reduce_sh = (float *)((uint8_t *)v_tile + ATTN_TILE_K * head_dim * sizeof(uint16_t));
    float *acc_sh = reduce_sh + 256;               /* [ATTN_TILE_Q][head_dim] */
    float *max_score_sh = acc_sh + (size_t)ATTN_TILE_Q * head_dim;  /* [ATTN_TILE_Q] */
    float *sum_exp_sh = max_score_sh + ATTN_TILE_Q; /* [ATTN_TILE_Q] */
    __shared__ float rescale_sh, weight_sh;

    for (int i = tid; i < ATTN_TILE_Q * head_dim; i += n_threads) acc_sh[i] = 0.0f;
    for (int qi = tid; qi < ATTN_TILE_Q; qi += n_threads) {
        max_score_sh[qi] = -1e30f;
        sum_exp_sh[qi] = 0.0f;
    }
    gpuSyncthreads();

    /* Total KV positions visible: start_pos + n_tokens */
    int total_kv = start_pos + n_tokens;

    /* Tile over KV positions */
    for (int t0 = 0; t0 < total_kv; t0 += ATTN_TILE_K) {
        int t_end = min(t0 + ATTN_TILE_K, total_kv);
        int tile_k_size = t_end - t0;

        /* Load K and V tile into shared memory */
        for (int d = tid; d < head_dim; d += n_threads) {
            for (int ti = 0; ti < tile_k_size; ti++) {
                size_t k_off = layer_base + kv_h * kv_head_stride_bytes / 2 + (size_t)(t0 + ti) * kv_pos_stride_bytes / 2 + d;
                k_tile[ti * head_dim + d] = kv_k[k_off];
                v_tile[ti * head_dim + d] = kv_v[k_off];
            }
        }
        gpuSyncthreads();

        /* For each query token in this block, compute attention over this KV tile */
        for (int qi = 0; qi < n_q; qi++) {
            int global_q = q_start + qi;
            int global_pos = start_pos + global_q;

            const float *qg = q_dev + (size_t)(global_q * n_heads + h) * head_dim;
            float *accqi = acc_sh + (size_t)qi * head_dim;

            for (int ti = 0; ti < tile_k_size; ti++) {
                int global_kv = t0 + ti;
                if (global_kv > global_pos) continue;

                /* Compute score with thread parallelization */
                float score = 0.0f;
                for (int d = tid; d < head_dim; d += n_threads) {
                    score += qg[d] * gpu_fp16_to_fp32(k_tile[ti * head_dim + d]);
                }
                /* Tree reduce */
                reduce_sh[tid] = score;
                gpuSyncthreads();
                for (int s = n_threads / 2; s > 0; s >>= 1) {
                    if (tid < s) reduce_sh[tid] += reduce_sh[tid + s];
                    gpuSyncthreads();
                }
                score = reduce_sh[0] / sqrt_hd;

                /* Online softmax branch decision on thread 0, broadcast
                 * via shared scalars; accumulator update parallelized
                 * across all threads below. */
                if (tid == 0) {
                    if (score > max_score_sh[qi]) {
                        rescale_sh = expf(max_score_sh[qi] - score);
                        weight_sh = 1.0f;
                        sum_exp_sh[qi] = sum_exp_sh[qi] * rescale_sh + 1.0f;
                        max_score_sh[qi] = score;
                    } else {
                        rescale_sh = 1.0f;
                        weight_sh = expf(score - max_score_sh[qi]);
                        sum_exp_sh[qi] += weight_sh;
                    }
                }
                gpuSyncthreads();

                for (int d = tid; d < head_dim; d += n_threads) {
                    accqi[d] = accqi[d] * rescale_sh + weight_sh * gpu_fp16_to_fp32(v_tile[ti * head_dim + d]);
                }
                gpuSyncthreads(); /* reduce_sh/rescale_sh/weight_sh reused next ti/qi */
            }
        }
    }

    /* Normalize and write output, parallelized across threads */
    for (int qi = 0; qi < n_q; qi++) {
        int global_q = q_start + qi;
        float inv_sum = 1.0f / sum_exp_sh[qi];
        float *xbhg = xb_out + (size_t)(global_q * n_heads + h) * head_dim;
        float *accqi = acc_sh + (size_t)qi * head_dim;
        for (int d = tid; d < head_dim; d += n_threads) {
            xbhg[d] = accqi[d] * inv_sum;
        }
    }
}

/* ---- SSM alpha/beta batched vec_dot kernel ----
 * Each thread block handles one head, 256 threads process dim elements.
 * Weights: column-major [dim, n_heads], each column is a quantized row. */
__global__ void
picolm_ssm_vecdot_kernel(float *out,
                          const float *x,
                          const void *weights,
                          gguf_type_t qtype,
                          int dim, int n_v_heads,
                          int row_bytes,
                          const int *head_map) {
    int h = gpuBlockIdx_x;
    if (h >= n_v_heads) return;
    int tid = gpuThreadIdx_x;
    int gh = head_map ? head_map[h] : h;
    const char *wrow = (const char *)weights + (size_t)gh * row_bytes;
    double sum = 0.0;

    switch (qtype) {
    case 2: { /* Q4_0 */
        int n_blocks = dim / 32;
        for (int bi = tid; bi < n_blocks; bi += gpuBlockDim_x) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 18;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float d = gpu_fp16_to_fp32(d_raw);
            const uint8_t *qs = blk + 2;
            for (int j = 0; j < 32; j++) {
                int i = bi * 32 + j;
                int v = (j < 16) ? (qs[j] & 0xF) : (qs[j - 16] >> 4);
                sum += x[i] * (float)(v - 8) * d;
            }
        }
        break;
    }
    case 8: { /* Q8_0 */
        int n_blocks = dim / 32;
        for (int bi = tid; bi < n_blocks; bi += gpuBlockDim_x) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 34;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float d = gpu_fp16_to_fp32(d_raw);
            const int8_t *qs = (const int8_t *)(blk + 2);
            for (int j = 0; j < 32; j++) {
                int i = bi * 32 + j;
                sum += x[i] * (float)qs[j] * d;
            }
        }
        break;
    }
    default:
        break;
    }
    /* Tree reduce */
    float local = (float)sum;
    __shared__ float sdata[256];
    sdata[tid] = local;
    gpuSyncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        gpuSyncthreads();
    }
    if (tid == 0) out[h] = sdata[0];
}

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
__global__ void
picolm_ssm_recurrence_kernel(float *state,
                              const float *q_conv,
                              const float *k_conv,
                              const float *v_conv,
                              const float *gate_exp,
                              const float *beta,
                              float *ssm_output,
                              int n_v_heads, int d_state,
                              int repeat) {
    int h = gpuBlockIdx_x;
    if (h >= n_v_heads) return;
    int tid = gpuThreadIdx_x;

    /* Shared memory for intermediate results: sk[d_state] + d_local[d_state] */
    extern __shared__ float sdata[];
    float *sk = sdata;
    float *d_local = sdata + d_state;

    /* Per-head pointers */
    int kh = h / repeat;
    const float *qh = q_conv + (size_t)kh * d_state;
    const float *khv = k_conv + (size_t)kh * d_state;
    const float *vh = v_conv + (size_t)h * d_state;
    float ge = gate_exp[h];
    float bh = beta[h];
    float *st = state + (size_t)h * d_state * d_state;
    float *out_base = ssm_output + h; /* stride n_v_heads in dim-major */

    /* Step 1: Decay state elementwise */
    for (int i = tid; i < d_state * d_state; i += gpuBlockDim_x) {
        st[i] *= ge;
    }
    gpuSyncthreads();

    /* Step 2: sk[row] = state[row] @ k */
    for (int row = tid; row < d_state; row += gpuBlockDim_x) {
        float sum = 0.0f;
        const float *st_row = st + (size_t)row * d_state;
        for (int col = 0; col < d_state; col++) {
            sum += st_row[col] * khv[col];
        }
        sk[row] = sum;
    }
    gpuSyncthreads();

    /* Step 3: d[row] = (v[row] - sk[row]) * beta */
    for (int row = tid; row < d_state; row += gpuBlockDim_x) {
        d_local[row] = (vh[row] - sk[row]) * bh;
    }
    gpuSyncthreads();

    /* Step 4: state[row][col] += k[col] * d[row] */
    for (int i = tid; i < d_state * d_state; i += gpuBlockDim_x) {
        int row = i / d_state;
        int col = i % d_state;
        st[i] += khv[col] * d_local[row];
    }
    gpuSyncthreads();

    /* Step 5: output[row] = state[row] @ q, written dim-major [row*n_v_heads + h] */
    for (int row = tid; row < d_state; row += gpuBlockDim_x) {
        float sum = 0.0f;
        const float *st_row = st + (size_t)row * d_state;
        for (int col = 0; col < d_state; col++) {
            sum += st_row[col] * qh[col];
        }
        out_base[(size_t)row * n_v_heads] = sum;
    }
}

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
__global__ void picolm_w4a16_matmul(float *y, const float *x, const void *weights,
                                     int M, int K, int N, int block_size);
__global__ void picolm_w4a16_gate_up(float *gate, float *up, const float *x,
                                      const void *gate_weights, const void *up_weights,
                                      int M, int K, int N, int block_size);
#endif

#ifdef PICOLM_GPU_WMMA_AVAILABLE

/* Dequant one nibble from block_q4_0 to FP16.
 * GGUF Q4_0: values 0..15 = low nibble of qs[0..15], values 16..31 = high nibble.
 * NOT interleaved (that was the Q4_0 precision bug from earlier). */
__device__ static inline half dequant_q4_0_elem_fp16(const void *blk, int j) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    int v;
    if (j < 16) v = b[2 + j] & 0xF;
    else v = b[2 + j - 16] >> 4;
#ifdef __HIP__
    half d; d.__x = d_raw;
    half val; val.__x = gpu_fp32_to_fp16((float)(v - 8));
    return d * val;
#else
    half d = __ushort_as_half(d_raw);
    half val = __float2half((float)(v - 8));
    return d * val;
#endif
}

__global__ void
picolm_w4a16_matmul(float *y, const float *x, const void *weights,
                     int M, int K, int N, int block_size) {
    int warp = gpuThreadIdx_x >> 5;
    int lane = gpuThreadIdx_x & 31;
    int m0 = gpuBlockIdx_y * 16;
    int n0 = gpuBlockIdx_x * 64 + warp * 16;

#ifdef __HIP__
    using namespace wmma;
    __shared__ half ah[256];
    __shared__ half bh[4][256];
    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.f);
#else
    using namespace nvcuda;
    __shared__ __half ah[256];
    __shared__ __half bh[4][256];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
#endif

    int nblocks = ((K + 31) / 32);  /* blocks per row */
    for (int k0 = 0; k0 < K; k0 += 16) {
        /* Load A tile (FP32 -> FP16) */
        for (int z = gpuThreadIdx_x; z < 256; z += gpuBlockDim_x) {
            int m = z / 16, k = z % 16;
            int gm = m0 + m, gk = k0 + k;
            ah[z] = (gm < M && gk < K) ? __float2half(x[(size_t)gm * K + gk]) : __float2half(0.f);
        }
        /* Dequant B tile from block_q4_0 */
        for (int z = lane; z < 256; z += 32) {
            int n = z / 16, gk = k0 + (z % 16), gn = n0 + n;
            if (gn >= N || gk >= K) {
                bh[warp][z] = __float2half(0.f);
                continue;
            }
            int bi = gk / 32, ji = gk % 32;
            const void *blk_p = (const char *)weights + (size_t)gn * nblocks * block_size + (size_t)bi * block_size;
            bh[warp][z] = dequant_q4_0_elem_fp16(blk_p, ji);
        }
        gpuSyncthreads();

#ifdef __HIP__
        fragment<matrix_a, 16, 16, 16, half, row_major> af;
        fragment<matrix_b, 16, 16, 16, half, col_major> bf;
        load_matrix_sync(af, ah, 16);
        load_matrix_sync(bf, bh[warp], 16);
        mma_sync(acc, af, bf, acc);
#else
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
        wmma::load_matrix_sync(af, ah, 16);
        wmma::load_matrix_sync(bf, bh[warp], 16);
        wmma::mma_sync(acc, af, bf, acc);
#endif
        gpuSyncthreads();
    }

#ifdef __HIP__
    __shared__ float out[4][256];
    store_matrix_sync(out[warp], acc, 16, mem_row_major);
#else
    __shared__ float out[4][256];
    wmma::store_matrix_sync(out[warp], acc, 16, wmma::mem_row_major);
#endif
    for (int z = lane; z < 256; z += 32) {
        int m = z / 16, n = z % 16;
        if (m0 + m < M && n0 + n < N)
            y[(size_t)(m0 + m) * N + n0 + n] = out[warp][z];
    }
}

/* ---- w4a16_gate_up kernel (fused gate+up) ---- */

__global__ void
picolm_w4a16_gate_up(float *gate, float *up, const float *x,
                      const void *gw, const void *uw,
                      int M, int K, int N, int block_size) {
    int warp = gpuThreadIdx_x >> 5;
    int lane = gpuThreadIdx_x & 31;
    int which = warp & 1, tile = warp >> 1;
    int m0 = gpuBlockIdx_y * 16;
    int n0 = gpuBlockIdx_x * 64 + tile * 16;
    const void *w = which ? uw : gw;
    float *y = which ? up : gate;

#ifdef __HIP__
    using namespace wmma;
    __shared__ half ah[256];
    __shared__ half bh[8][256];
    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.f);
#else
    using namespace nvcuda;
    __shared__ __half ah[256];
    __shared__ __half bh[8][256];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
#endif

    int nblocks = ((K + 31) / 32);
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int z = gpuThreadIdx_x; z < 256; z += gpuBlockDim_x) {
            int m = z / 16, k = z % 16;
            int gm = m0 + m, gk = k0 + k;
            ah[z] = (gm < M && gk < K) ? __float2half(x[(size_t)gm * K + gk]) : __float2half(0.f);
        }
        for (int z = lane; z < 256; z += 32) {
            int n = z / 16, gk = k0 + (z % 16), gn = n0 + n;
            if (gn >= N || gk >= K) {
                bh[warp][z] = __float2half(0.f);
                continue;
            }
            int bi = gk / 32, ji = gk % 32;
            const void *blk_p = (const char *)w + (size_t)gn * nblocks * block_size + (size_t)bi * block_size;
            bh[warp][z] = dequant_q4_0_elem_fp16(blk_p, ji);
        }
        gpuSyncthreads();

#ifdef __HIP__
        fragment<matrix_a, 16, 16, 16, half, row_major> af;
        fragment<matrix_b, 16, 16, 16, half, col_major> bf;
        load_matrix_sync(af, ah, 16);
        load_matrix_sync(bf, bh[warp], 16);
        mma_sync(acc, af, bf, acc);
#else
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
        wmma::load_matrix_sync(af, ah, 16);
        wmma::load_matrix_sync(bf, bh[warp], 16);
        wmma::mma_sync(acc, af, bf, acc);
#endif
        gpuSyncthreads();
    }

#ifdef __HIP__
    __shared__ float out[8][256];
    store_matrix_sync(out[warp], acc, 16, mem_row_major);
#else
    __shared__ float out[8][256];
    wmma::store_matrix_sync(out[warp], acc, 16, wmma::mem_row_major);
#endif
    for (int z = lane; z < 256; z += 32) {
        int m = z / 16, n = z % 16;
        if (m0 + m < M && n0 + n < N)
            y[(size_t)(m0 + m) * N + n0 + n] = out[warp][z];
    }
}

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

typedef struct {
    int device;
    int compute_major, compute_minor;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    float *host_x, *host_y;
    size_t host_x_cap, host_y_cap;
    /* Q8_0 GPU matmul scratch buffers */
    int8_t *q8_xq;      /* quantized input activations (int8) */
    float *q8_xd;       /* pre-converted x deltas (fp32) */
    size_t q8_xq_cap;   /* capacity in bytes for q8_xq */
    size_t q8_xd_cap;   /* capacity in bytes for q8_xd */
    gpuStream_t stream;
    size_t tensor_count, tensor_bytes;
    /* Phase 2: fixed-size device-resident pipeline buffers for
     * model_forward_gpu() decode (S=1 only). Allocated once at model
     * load via picolm_gpu_pipeline_alloc(), sized from model config, no
     * grow-on-demand -- deliberately NOT reserve()-based, so the
     * per-token layer chain never risks a mid-pass cudaFree/cudaMalloc
     * racing a still-queued kernel that reads the old pointer. Must not
     * alias x/y/q8_xq/q8_xd above (those stay used internally by
     * picolm_gpu_matmul_dev's Q8_0 path). */
    float *pipe_x, *pipe_xb, *pipe_q, *pipe_k, *pipe_v,
          *pipe_attn_out, *pipe_ffn_norm, *pipe_gate, *pipe_up;
    int pipe_ready; /* 1 once the above are allocated for this device */
    /* Prefill batch buffers: [max_seq_len][dim] for S>1 prefill pipeline.
     * Same set of buffers as decode but sized for n_tokens rows. */
    float *pipe_x_b, *pipe_xb_b, *pipe_q_b, *pipe_k_b, *pipe_v_b,
          *pipe_attn_out_b, *pipe_ffn_norm_b, *pipe_gate_b, *pipe_up_b;
    int pipe_b_ready;
} gpu_device_ctx_t;

static gpu_device_ctx_t g_gpu_ctx[PICOLM_GPU_MAX_DEVICES];
static int g_nctx;

/* Mutex protecting g_gpu_ctx scratch buffer resize (reserve/reserve_pinned).
 * Resize is rare (once per buffer growth), so lock contention is negligible. */
#ifdef _WIN32
static SRWLOCK g_resize_mutex = SRWLOCK_INIT;
#else
static pthread_mutex_t g_resize_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

static int gpu_ok(gpuError_t err, const char *what) {
    if (err == gpuSuccess) return 1;
    fprintf(stderr, "[GPU] %s: %s\n", what, gpuGetErrorString(err));
    return 0;
}

static gpu_device_ctx_t *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++)
        if (g_gpu_ctx[i].device == device) return &g_gpu_ctx[i];
    return NULL;
}

/* Thread-local device cache: avoid redundant cudaSetDevice/hipSetDevice calls */
#ifdef _WIN32
static __declspec(thread) int g_current_device = -1;
#else
static __thread int g_current_device = -1;
#endif

static int select_ctx(gpu_device_ctx_t *ctx) {
    if (!ctx) return 0;
    if (g_current_device == ctx->device) return 1;
    if (!gpu_ok(gpuSetDevice(ctx->device), "select device")) return 0;
    g_current_device = ctx->device;
    return 1;
}

static __host__ void gpu_mutex_lock(void) {
#ifdef _WIN32
    AcquireSRWLockExclusive(&g_resize_mutex);
#else
    pthread_mutex_lock(&g_resize_mutex);
#endif
}
static __host__ void gpu_mutex_unlock(void) {
#ifdef _WIN32
    ReleaseSRWLockExclusive(&g_resize_mutex);
#else
    pthread_mutex_unlock(&g_resize_mutex);
#endif
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    gpu_mutex_lock();
    if (*cap >= bytes) { gpu_mutex_unlock(); return 1; }
    if (*ptr) gpuFree(*ptr);
    *ptr = NULL; *cap = 0;
    /* Use device memory for scratch buffers.
     * cpu writes via explicit cudaMemcpyAsync H2D, gpu reads from device mem,
     * gpu writes to device mem, then cpu reads via explicit cudaMemcpyAsync D2H.
     * This avoids cudaMallocManaged page-fault overhead and err=1 issues
     * seen on GB10 (sm_121) with CUDA 13.0 where managed memory mixed with
     * device memory in kernel arguments causes cudaErrorInvalidValue on first
     * kernel launches. */
    gpuError_t err = gpuMalloc(ptr, bytes);
    if (!gpu_ok(err, "scratch allocation")) {
        gpu_mutex_unlock();
        return 0;
    }
    *cap = bytes;
    gpu_mutex_unlock();
    return 1;
}

static PICOLM_UNUSED int reserve_pinned(float **ptr, size_t *cap, size_t bytes) {
    gpu_mutex_lock();
    if (*cap >= bytes) { gpu_mutex_unlock(); return 1; }
    if (*ptr) gpuFreeHost(*ptr);
    *ptr = NULL; *cap = 0;
    if (!gpu_ok(gpuMallocHost(ptr, bytes), "pinned staging allocation")) {
        gpu_mutex_unlock();
        return 0;
    }
    *cap = bytes;
    gpu_mutex_unlock();
    return 1;
}

/* Reserve int8_t device buffer (for Q8_0 quantized activations) */
static int reserve_i8(int8_t **ptr, size_t *cap, size_t bytes) {
    gpu_mutex_lock();
    if (*cap >= bytes) { gpu_mutex_unlock(); return 1; }
    if (*ptr) gpuFree(*ptr);
    *ptr = NULL; *cap = 0;
    gpuError_t err = gpuMalloc(ptr, bytes);
    if (!gpu_ok(err, "int8 scratch allocation")) {
        gpu_mutex_unlock();
        return 0;
    }
    *cap = bytes;
    gpu_mutex_unlock();
    return 1;
}

/* ---- Public API ---- */

int picolm_gpu_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > PICOLM_GPU_MAX_DEVICES) return 0;
#ifndef __HIP__
    /* CUDA 13 / GB10 (sm_121): multi-engine scheduling can reorder/overlap
     * H2D, kernel, and D2H work even though our code issues everything
     * synchronously per-call, causing run-to-run nondeterminism at temp=0
     * (see gpu_nondeterminism.md -- isolated empirically, fixes it 8/8 with
     * no measured perf cost since we already sync per call). Must be set
     * before the first CUDA driver call, so this has to be the very first
     * thing in init. overwrite=0: respect an explicit user-set value. */
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1", 0);
#endif
    if (!gpu_ok(gpuGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            fprintf(stderr, "[GPU] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0; return 0;
        }
        if (find_ctx(device)) {
            fprintf(stderr, "[GPU] duplicate device %d\n", device);
            g_nctx = 0; return 0;
        }
        gpu_device_ctx_t *ctx = &g_gpu_ctx[g_nctx];
        memset(ctx, 0, sizeof(*ctx));
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        gpuDeviceProp prop;
        if (!gpu_ok(gpuGetDeviceProperties(&prop, device), "device properties")) {
            g_nctx = 0; return 0;
        }
        ctx->compute_major = prop.major;
        ctx->compute_minor = prop.minor;
        if (!gpu_ok(gpuStreamCreateWithFlags(&ctx->stream,
#ifdef __HIP__
                                              hipStreamNonBlocking
#else
                                              cudaStreamNonBlocking
#endif
                                              ), "stream creation")) {
            g_nctx = 0; return 0;
        }
        g_nctx++;
        fprintf(stderr, "[GPU] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                device, prop.name, prop.totalGlobalMem / 1e9,
                prop.major, prop.minor);
    }
    return 1;
}

void picolm_gpu_shutdown(void) {
    /* Free KV cache allocations */
    picolm_gpu_kv_free();
    /* Free pipeline buffers */
    picolm_gpu_pipeline_free();
    for (int i = 0; i < g_nctx; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->x) gpuFree(ctx->x);
        if (ctx->y) gpuFree(ctx->y);
        if (ctx->gate) gpuFree(ctx->gate);
        if (ctx->up) gpuFree(ctx->up);
        if (ctx->q8_xq) gpuFree(ctx->q8_xq);
        if (ctx->q8_xd) gpuFree(ctx->q8_xd);
        if (ctx->host_x) gpuFreeHost(ctx->host_x);
        if (ctx->host_y) gpuFreeHost(ctx->host_y);
        if (ctx->stream) gpuStreamDestroy(ctx->stream);
        ctx->x = ctx->y = ctx->gate = ctx->up = NULL;
        ctx->q8_xq = NULL;
        ctx->q8_xd = NULL;
        ctx->host_x = ctx->host_y = NULL;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->q8_xq_cap = ctx->q8_xd_cap = 0;
        ctx->host_x_cap = ctx->host_y_cap = 0;
        ctx->stream = NULL;
    }
    g_nctx = 0;
}

/* Detect if we're on a unified memory SoC (Grace-Blackwell, Apple Silicon, etc.) */
static PICOLM_UNUSED int is_unified_memory(void) {
#ifdef __HIP__
    return 0; /* HIP: treat as discrete GPU (conservative) */
#else
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    /* On Grace-Blackwell, unifiedAddressing is true and the device has no
     * separate PCIe link width (it's chip-to-chip). Check for integrated
     * memory by looking at the bus type. */
    return prop.unifiedAddressing && prop.l2CacheSize > 0;
#endif
}

int picolm_gpu_device_count(void) { return g_nctx; }

int picolm_gpu_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_gpu_ctx[index].device : -1;
}

int picolm_gpu_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    size_t fb = 0, tb = 0;
    return gpu_ok(gpuMemGetInfo(&fb, &tb), "memory info") && (*free_bytes = fb, *total_bytes = tb, 1);
}

/* Map GGUF_TYPE to block size and values per block.
 * F32 (0), F16 (1), BF16 (30): no blocks, each is an individual element. */
static int gguf_block_size(gguf_type_t qtype) {
    switch (qtype) {
    case 0:  return 0;    /* F32: no blocks */
    case 1:  return 0;    /* F16: no blocks */
    case 30: return 0;    /* BF16: no blocks */
    case 2:  return 18;   /* Q4_0: 18 bytes per 32 values */
    case 8:  return GPU_BLOCK_Q8_0_SIZE; /* Q8_0: 34 bytes per 32 values */
    case 12: return GPU_BLOCK_Q4_K_SIZE;  /* Q4_K: 144 bytes per 256 values */
    case 41: return 18;   /* Q1_0: 18 bytes per 128 values */
    case 42: return 34;   /* Q2_0: 34 bytes per 128 values */
    default: return 0;
    }
}

int picolm_gpu_tensor_upload(void **tensor,
                              const void *weights,
                              gguf_type_t qtype, int I, int O, int device) {
    if (!tensor || !weights || I < 1 || O < 1) return 0;
    picolm_gpu_tensor_t **tp = (picolm_gpu_tensor_t **)tensor;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;
    int bs = gguf_block_size(qtype);
    if (!bs && qtype != 0 && qtype != 1 && qtype != 30) return 0;
    if (*tp) return 1; /* idempotent */

    /* Compute row bytes */
    size_t row_bytes;
    int vals_per_block;
    if (qtype == 0) {
        row_bytes = (size_t)I * sizeof(float);
        vals_per_block = 1;
    } else if (qtype == 1 || qtype == 30) {
        row_bytes = (size_t)I * sizeof(uint16_t);
        vals_per_block = 1;
    } else if (qtype == 12) {
        vals_per_block = 256;
        row_bytes = (size_t)((I + vals_per_block - 1) / vals_per_block) * bs;
    } else if (qtype == 41 || qtype == 42) {
        vals_per_block = 128;
        row_bytes = (size_t)((I + vals_per_block - 1) / vals_per_block) * bs;
    } else {
        vals_per_block = 32;
        row_bytes = (size_t)((I + vals_per_block - 1) / vals_per_block) * bs;
    }
    size_t total = row_bytes * (size_t)O;

    picolm_gpu_tensor_t *t = (picolm_gpu_tensor_t *)calloc(1, sizeof(*t));
    if (!t) return 0;
    t->qtype = qtype; t->I = I; t->O = O; t->device = device;
    t->row_bytes = row_bytes; t->block_size = bs;

    /* Try zero-copy first: register CPU memory with GPU (unified memory SoC) */
    /* NOTE: mmap'd file-backed memory can cause zero-copy to silently produce
     * wrong results on some platforms. cudaHostRegister may succeed but the GPU
     * sees stale or unmapped pages. We use cudaHostRegister with
     * cudaHostRegisterPortable and verify the upload works.
     *
     * For now, skip zero-copy and always copy. The copy happens once at model
     * load time, and on SoC systems with unified memory the effective cost is
     * just a page-table walk, not a real data copy. */
    if (!gpu_ok(gpuMalloc(&t->weights, total), "tensor allocation")) {
        fprintf(stderr, "[GPU] OOM: I=%d O=%d qtype=%d total=%zu MB (gpu_used=%.1f MB)\n",
                I, O, qtype, total/(1024*1024), ctx->tensor_bytes/(1024.0*1024));
        free(t); return 0;
    }
    if (!gpu_ok(gpuMemcpy(t->weights, weights, total, gpuMemcpyHostToDevice),
                "tensor upload")) {
        gpuFree(t->weights); free(t); return 0;
    }
    t->zero_copy = 0;
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += total;
    
    /* Print upload summary for first tensor */
    {
        static int first_print = 1;
        if (first_print) {
            fprintf(stderr, "[GPU] upload mode: %s\n", t->zero_copy ? "zero-copy (unified)" : "copied");
            first_print = 0;
        }
    }
    *tp = t;
    return 1;
}

void picolm_gpu_tensor_free(picolm_gpu_tensor_t *t) {
    if (!t) return;
    gpu_device_ctx_t *ctx = find_ctx(t->device);
    if (ctx && t->tracked) {
        if (select_ctx(ctx)) {
            ctx->tensor_count--;
            ctx->tensor_bytes -= t->row_bytes * (size_t)t->O;
        }
    }
    if (t->weights) {
        if (t->zero_copy) {
            gpuHostUnregister(t->weights);
        } else {
            gpuFree(t->weights);
        }
        t->weights = NULL;
    }
    free(t);
}

size_t picolm_gpu_tensor_bytes(const picolm_gpu_tensor_t *t) {
    return t ? t->row_bytes * (size_t)t->O : 0;
}

int picolm_gpu_tensor_device(const picolm_gpu_tensor_t *t) {
    return t ? t->device : -1;
}

/* Upload a plain host F32 vector (norm weights, RoPE cos/sin tables --
 * anything that isn't a quantized matmul weight matrix, so
 * picolm_gpu_tensor_upload doesn't apply) to a freshly allocated device
 * buffer. Caller owns the returned pointer and must gpuFree() it (or
 * just leak it for the lifetime of the process, same as other one-time
 * per-model uploads here). Returns NULL on failure. */
extern "C" float *
picolm_gpu_upload_f32(const float *host, size_t n, int device) {
    if (!host || n < 1) return NULL;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return NULL;

    float *dev = NULL;
    size_t bytes = n * sizeof(float);
    if (!gpu_ok(gpuMalloc(&dev, bytes), "f32 vector allocation")) return NULL;
    if (!gpu_ok(gpuMemcpy(dev, host, bytes, gpuMemcpyHostToDevice), "f32 vector upload")) {
        gpuFree(dev);
        return NULL;
    }
    return dev;
}

int picolm_gpu_matmul(picolm_gpu_tensor_t *t, float *y, const float *x, int S, int device) {
    if (!t || !y || !x || S < 1) return 0;
    /* Minimum I for GPU to be worthwhile vs CPU kernel launch overhead.
     * Smaller tensors: kernel launch (~200us) dominates compute. */
    if (t->I < 512 || t->O < 256) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    int I = t->I, O = t->O;
    size_t xb = (size_t)S * I * sizeof(float);
    size_t yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) ||
        !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;

    /* ---- Q8_0 special path: quantize x to Q8_0 on GPU, then int8 MAC ----
     * This matches the CPU path in tensor.c exactly:
     *   1. quantize_row_q8_0(x) -> int8 qs + float d per 32-block
     *   2. vec_dot_q8_0_q8_0_deltas(int8_x, float_xd, int8_w, fp16_wd)
     * GPU quantize kernel outputs d as FP32 directly (matching CPU's
     * pre-converted qx_d array), avoiding an extra FP16->FP32 conversion
     * step. The matmul kernel then does pure int8 MAC with fp32 scales. */
    if (t->qtype == GGUF_TYPE_Q8_0) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0; /* must be aligned */

        /* Upload fp32 input */
        if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;

        /* Allocate quantized input buffers: qs (int8_t[S*I]) + d (float[S*n_blocks]) */
        size_t xq_bytes = (size_t)S * I;
        size_t xd_bytes = (size_t)S * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        /* Step 1: Quantize x to Q8_0 on GPU */
        dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, ctx->x, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 quantize sync")) return 0;

        /* Step 2: Q8_0 x Q8_0 integer MAC matmul, tiled over S to reuse
         * each weight row from shared memory across Q8_TILE_S positions.
         * For S=1 (decode), the tiled kernel's shared memory load+sync
         * overhead dominates -- fall back to the original per-block kernel.
         * Also falls back if row too large for shared memory. */
        if (S > 1 && t->row_bytes + 2048 <= 49152) {
            dim3 grid((unsigned)O, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
            picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)t->row_bytes, ctx->stream>>>(
                ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes);
        } else {
            dim3 grid((unsigned)O, (unsigned)S);
            picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
                ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes);
        }
        if (!gpu_ok(gpuGetLastError(), "q8 matmul") ||
            !gpu_ok(gpuDeviceSynchronize(), "q8 matmul sync")) return 0;

        if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
        return 1;
    }

    /* Generic path for all other quant types */
    /* Scratch buffers are in device memory. Explicit H2D copy needed. */
    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;

    dim3 grid((unsigned)O, (unsigned)S);
    picolm_quant_matmul<<<grid, 256, 0, ctx->stream>>>(ctx->y, ctx->x, t->weights,
                                        t->qtype, S, I, O,
                                        (int)t->row_bytes);
    if (!gpu_ok(gpuGetLastError(), "matmul launch") ||
        !gpu_ok(gpuDeviceSynchronize(), "matmul sync")) return 0;

    /* Scratch buffers are in device memory. Explicit D2H copy needed. */
    if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

/* Device-native matmul: x_dev and y_dev are already device-resident
 * pointers owned by the caller (Phase 2 pipeline buffers) -- no H2D, no
 * D2H, no gpuDeviceSynchronize(). Mirrors picolm_gpu_matmul() exactly
 * (same eligibility checks, same kernels, same Q8_0 special-case path)
 * except for the copies and the per-step sync.
 *
 * Safe to omit the sync here: all kernels below are launched on
 * ctx->stream, and CUDA guarantees in-order execution of work queued to
 * the same stream, so the q8 quantize kernel is guaranteed complete
 * before the q8_q8 matmul kernel reads its output, with no explicit sync
 * needed in between. The caller (model_forward_gpu) must call
 * gpuDeviceSynchronize() exactly once, after the very last op of the
 * whole forward pass and before reading any result back via D2H -- not
 * after each layer or each matmul. That single sync is what actually
 * eliminates the ~14-syncs/layer overhead Phase 2 exists to remove.
 *
 * x_dev/y_dev must NOT alias ctx->x/ctx->y/ctx->q8_xq/ctx->q8_xd (those
 * are still used internally here as Q8_0 quantize scratch) -- pass
 * dedicated pipeline buffers. */
extern "C" int
picolm_gpu_matmul_dev(picolm_gpu_tensor_t *t, float *y_dev, const float *x_dev,
                       int S, int device) {
    if (!t || !y_dev || !x_dev || S < 1) return 0;
    if (t->I < 512 || t->O < 256) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    int I = t->I, O = t->O;

    if (t->qtype == GGUF_TYPE_Q8_0) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0; /* must be aligned */

        size_t xq_bytes = (size_t)S * I;
        size_t xd_bytes = (size_t)S * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;

        dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, x_dev, I, S);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize (dev)")) return 0;

        if (S > 1 && t->row_bytes + 2048 <= 49152) {
            dim3 grid((unsigned)O, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
            picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)t->row_bytes, ctx->stream>>>(
                y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes);
        } else {
            dim3 grid((unsigned)O, (unsigned)S);
            picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
                y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes);
        }
        if (!gpu_ok(gpuGetLastError(), "q8 matmul (dev)")) return 0;
        return 1;
    }

    dim3 grid((unsigned)O, (unsigned)S);
    picolm_quant_matmul<<<grid, 256, 0, ctx->stream>>>(y_dev, x_dev, t->weights,
                                        t->qtype, S, I, O,
                                        (int)t->row_bytes);
    if (!gpu_ok(gpuGetLastError(), "matmul launch (dev)")) return 0;
    return 1;
}

int picolm_gpu_expert_mlp(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                           picolm_gpu_tensor_t *down, float *y, const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;

    gpu_device_ctx_t *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;

    int D = gate->I, I = gate->O;
    size_t xb = (size_t)S * D * sizeof(float);
    size_t ib = (size_t)S * I * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) ||
        !reserve(&ctx->y, &ctx->y_cap, xb) ||
        !reserve(&ctx->gate, &ctx->gate_cap, ib) ||
        !reserve(&ctx->up, &ctx->up_cap, ib)) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "expert input")) return 0;

    /* gate projection */
    dim3 hidden_grid((unsigned)I, (unsigned)S);
    picolm_quant_matmul<<<hidden_grid, 256>>>(ctx->gate, ctx->x, gate->weights,
        gate->qtype, S, D, I, (int)gate->row_bytes);
    /* up projection */
    picolm_quant_matmul<<<hidden_grid, 256>>>(ctx->up, ctx->x, up->weights,
        up->qtype, S, D, I, (int)up->row_bytes);
    /* silu(gate) * up */
    size_t n = (size_t)S * I;
    picolm_silu_mul<<<(unsigned)((n + 255) / 256), 256>>>(ctx->gate, ctx->up, n);
    /* down projection */
    dim3 output_grid((unsigned)D, (unsigned)S);
    picolm_quant_matmul<<<output_grid, 256>>>(ctx->y, ctx->gate, down->weights,
        down->qtype, S, I, D, (int)down->row_bytes);

    if (!gpu_ok(gpuGetLastError(), "expert MLP")) return 0;
    if (!gpu_ok(gpuMemcpy(y, ctx->y, xb, gpuMemcpyDeviceToHost), "expert output")) return 0;
    return 1;
}

int picolm_gpu_w4a16_mlp(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                          picolm_gpu_tensor_t *down, float *y, const float *x, int S) {
#ifdef PICOLM_GPU_WMMA_AVAILABLE
    if (!gate || !up || !down || !x || !y || S < 1) return 0;
    if (gate->qtype != 2 || up->qtype != 2 || down->qtype != 2) return 0; /* Q4_0 only */
    if (gate->device != up->device || gate->device != down->device) return 0;
    if (gate->I != up->I || gate->O != up->O || down->I != gate->O || down->O != gate->I)
        return 0;

    gpu_device_ctx_t *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;

    int D = gate->I, I = gate->O;
    size_t xb = (size_t)S * D * sizeof(float);
    size_t ib = (size_t)S * I * sizeof(float);

    if (!reserve(&ctx->x, &ctx->x_cap, xb) ||
        !reserve(&ctx->gate, &ctx->gate_cap, ib) ||
        !reserve(&ctx->up, &ctx->up_cap, ib) ||
        !reserve(&ctx->y, &ctx->y_cap, xb)) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "w4a16 input")) return 0;

    /* fused gate+up via WMMA */
    dim3 hidden((unsigned)((I + 63) / 64), (unsigned)((S + 15) / 16));
    picolm_w4a16_gate_up<<<hidden, 256>>>(ctx->gate, ctx->up, ctx->x,
        gate->weights, up->weights, S, D, I, gate->block_size);
    /* silu(gate) * up */
    size_t n = (size_t)S * I;
    picolm_silu_mul<<<(unsigned)((n + 255) / 256), 256>>>(ctx->gate, ctx->up, n);
    /* down via WMMA */
    dim3 output((unsigned)((D + 63) / 64), (unsigned)((S + 15) / 16));
    picolm_w4a16_matmul<<<output, 256>>>(ctx->y, ctx->gate, down->weights, S, I, D, down->block_size);

    if (!gpu_ok(gpuGetLastError(), "w4a16 launch")) return 0;
#ifdef __HIP__
    if (!gpu_ok(gpuMemcpy(y, ctx->y, xb, gpuMemcpyDeviceToHost), "w4a16 output")) return 0;
#else
    memcpy(y, ctx->y, xb);
#endif
    return 1;
#else
    (void)gate; (void)up; (void)down; (void)y; (void)x; (void)S;
    return 0; /* WMMA not available on this arch */
#endif
}

/* General-purpose WMMA matmul for Q4_0 weights.
 * Handles any tensor, not just expert MLP.
 * M = rows (S in our convention), K = columns (I), N = output (O)
 * Returns 1 on success, 0 to fall back to quant_matmul.
 * Constraints: qtype must be Q4_0, N%64==0, M%16==0, K%32==0. */
int picolm_gpu_w4a16_matmul(picolm_gpu_tensor_t *t, float *y, const float *x, int S, int device) {
#ifdef PICOLM_GPU_WMMA_AVAILABLE
    if (!t || !y || !x || S < 1 || t->qtype != 2) return 0; /* Q4_0 only */
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    int M = S, K = t->I, N = t->O;
    /* WMMA is only worthwhile for large batch sizes.
     * Small M wastes SM resources (16x64 tile for a 1-row output).
     * N must be at least 64 (one grid-x tile). */
    if (M < 16 || N < 64 || K < 64) return 0;

    size_t xb = (size_t)M * K * sizeof(float);
    size_t yb = (size_t)M * N * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) ||
        !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "w4a16 input")) return 0;

    dim3 grid((unsigned)((N + 63) / 64), (unsigned)((M + 15) / 16));
    picolm_w4a16_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, M, K, N, t->block_size);
    if (!gpu_ok(gpuGetLastError(), "w4a16 matmul") ||
        !gpu_ok(gpuDeviceSynchronize(), "w4a16 sync")) return 0;

    if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "w4a16 output")) return 0;
    return 1;
#else
    (void)t; (void)y; (void)x; (void)S; (void)device;
    return 0;
#endif
}

/* ---- SSM recurrence API ---- */
extern "C" int
picolm_gpu_ssm_recurrence(float *state,
                           const float *q_conv,
                           const float *k_conv,
                           const float *v_conv,
                           const float *gate_exp,
                           const float *beta,
                           float *ssm_output,
                           int n_v_heads, int d_state,
                           int repeat, int device) {
    if (n_v_heads <= 0 || d_state <= 0) return 0;
    if (d_state > 256) return 0;

    if (!gpu_ok(gpuSetDevice(device), "ssm device")) return 0;
    gpu_device_ctx_t *ctx = NULL;
    for (int i = 0; i < g_nctx; i++) {
        if (g_gpu_ctx[i].device == device) { ctx = &g_gpu_ctx[i]; break; }
    }
    if (!ctx) return 0;

    /* Upload all CPU data to device memory.
     * On AMD iGPU, CPU malloc'd memory is NOT directly GPU-accessible. */
    size_t state_bytes = (size_t)n_v_heads * d_state * d_state * sizeof(float);
    size_t q_bytes = (size_t)(n_v_heads / repeat) * d_state * sizeof(float);
    size_t k_bytes = q_bytes;
    size_t v_bytes = (size_t)n_v_heads * d_state * sizeof(float);
    size_t scalar_bytes = (size_t)n_v_heads * sizeof(float);
    size_t out_bytes = (size_t)d_state * n_v_heads * sizeof(float);

    void *ds, *dq, *dk, *dv, *dg, *db, *do_;
    if (!gpu_ok(gpuMalloc(&ds, state_bytes), "ssm st") ||
        !gpu_ok(gpuMalloc(&dq, q_bytes), "ssm q") ||
        !gpu_ok(gpuMalloc(&dk, k_bytes), "ssm k") ||
        !gpu_ok(gpuMalloc(&dv, v_bytes), "ssm v") ||
        !gpu_ok(gpuMalloc(&dg, scalar_bytes), "ssm g") ||
        !gpu_ok(gpuMalloc(&db, scalar_bytes), "ssm b") ||
        !gpu_ok(gpuMalloc(&do_, out_bytes), "ssm o")) return 0;

    if (!gpu_ok(gpuMemcpy(ds, state, state_bytes, gpuMemcpyHostToDevice), "ssm st h2d") ||
        !gpu_ok(gpuMemcpy(dq, q_conv, q_bytes, gpuMemcpyHostToDevice), "ssm q h2d") ||
        !gpu_ok(gpuMemcpy(dk, k_conv, k_bytes, gpuMemcpyHostToDevice), "ssm k h2d") ||
        !gpu_ok(gpuMemcpy(dv, v_conv, v_bytes, gpuMemcpyHostToDevice), "ssm v h2d") ||
        !gpu_ok(gpuMemcpy(dg, gate_exp, scalar_bytes, gpuMemcpyHostToDevice), "ssm g h2d") ||
        !gpu_ok(gpuMemcpy(db, beta, scalar_bytes, gpuMemcpyHostToDevice), "ssm b h2d")) {
        gpuFree(ds); gpuFree(dq); gpuFree(dk); gpuFree(dv);
        gpuFree(dg); gpuFree(db); gpuFree(do_); return 0;
    }

    size_t shared_mem = 2 * d_state * sizeof(float);
    dim3 grid((unsigned)n_v_heads, 1, 1);
    picolm_ssm_recurrence_kernel<<<grid, 256, (unsigned)shared_mem, ctx->stream>>>(
        (float *)ds, (const float *)dq, (const float *)dk,
        (const float *)dv, (const float *)dg, (const float *)db,
        (float *)do_, n_v_heads, d_state, repeat);

    if (!gpu_ok(gpuGetLastError(), "ssm recurrence") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm sync")) {
        gpuFree(ds); gpuFree(dq); gpuFree(dk); gpuFree(dv);
        gpuFree(dg); gpuFree(db); gpuFree(do_); return 0;
    }
    gpuMemcpy(ssm_output, do_, out_bytes, gpuMemcpyDeviceToHost);
    gpuMemcpy(state, ds, state_bytes, gpuMemcpyDeviceToHost);
    gpuFree(ds); gpuFree(dq); gpuFree(dk); gpuFree(dv);
    gpuFree(dg); gpuFree(db); gpuFree(do_);
    return 1;
}

/* ---- SSM batched vec_dot API ---- */
extern "C" int
picolm_gpu_ssm_vecdot(float *out_host,
                       const float *x_host,
                       const void *weights_host,
                       gguf_type_t qtype,
                       int dim, int n_v_heads,
                       int row_bytes,
                       const int *head_map,
                       int device) {
    if (n_v_heads <= 0 || dim <= 0) return 0;
    if (qtype != 2 && qtype != 8) return 0;

    if (!gpu_ok(gpuSetDevice(device), "ssm vecdot device")) return 0;
    gpu_device_ctx_t *ctx = NULL;
    for (int i = 0; i < g_nctx; i++) {
        if (g_gpu_ctx[i].device == device) { ctx = &g_gpu_ctx[i]; break; }
    }
    if (!ctx) return 0;

    size_t x_bytes = (size_t)dim * sizeof(float);
    size_t w_bytes = (size_t)n_v_heads * row_bytes;
    size_t out_bytes = (size_t)n_v_heads * sizeof(float);

    if (!reserve(&ctx->x, &ctx->x_cap, x_bytes) ||
        !reserve(&ctx->y, &ctx->y_cap, out_bytes)) return 0;
    void *w_dev = NULL;
    if (!gpu_ok(gpuMalloc(&w_dev, w_bytes), "ssm vecdot w malloc")) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, x_host, x_bytes, gpuMemcpyHostToDevice), "ssm x h2d") ||
        !gpu_ok(gpuMemcpy(w_dev, weights_host, w_bytes, gpuMemcpyHostToDevice), "ssm w h2d")) {
        gpuFree(w_dev);
        return 0;
    }

    dim3 grid((unsigned)n_v_heads, 1, 1);
    void *hm_dev = NULL;
    if (head_map) {
        gpuMalloc(&hm_dev, (size_t)n_v_heads * sizeof(int));
        gpuMemcpy(hm_dev, head_map, (size_t)n_v_heads * sizeof(int), gpuMemcpyHostToDevice);
    }
    picolm_ssm_vecdot_kernel<<<grid, 256, 256 * sizeof(float), ctx->stream>>>(
        ctx->y, ctx->x, w_dev, qtype, dim, n_v_heads, row_bytes,
        hm_dev ? (const int *)hm_dev : NULL);
    gpuFree(w_dev);
    if (hm_dev) gpuFree(hm_dev);

    if (!gpu_ok(gpuDeviceSynchronize(), "ssm vecdot sync") ||
        !gpu_ok(gpuMemcpy(out_host, ctx->y, out_bytes, gpuMemcpyDeviceToHost), "ssm out d2h")) return 0;
    return 1;
}

/* ================================================================
 * Phase 1: GPU-resident KV cache + attention - Host API
 * ================================================================ */

/* Per-device KV cache pointers */
static uint16_t *g_kv_k_dev[PICOLM_GPU_MAX_DEVICES];
static uint16_t *g_kv_v_dev[PICOLM_GPU_MAX_DEVICES];
static size_t g_kv_k_cap[PICOLM_GPU_MAX_DEVICES];
static size_t g_kv_v_cap[PICOLM_GPU_MAX_DEVICES];

extern "C" int
picolm_gpu_kv_alloc(size_t kv_k_bytes, size_t kv_v_bytes, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    /* Align to 256 bytes for coalesced access */
    kv_k_bytes = (kv_k_bytes + 255) & ~(size_t)255;
    kv_v_bytes = (kv_v_bytes + 255) & ~(size_t)255;

    if (g_kv_k_cap[device] < kv_k_bytes) {
        if (g_kv_k_dev[device]) gpuFree(g_kv_k_dev[device]);
        if (!gpu_ok(gpuMalloc(&g_kv_k_dev[device], kv_k_bytes), "KV K allocation")) return 0;
        /* Zero-initialize */
        gpuMemset(g_kv_k_dev[device], 0, kv_k_bytes);
        g_kv_k_cap[device] = kv_k_bytes;
    }
    if (g_kv_v_cap[device] < kv_v_bytes) {
        if (g_kv_v_dev[device]) gpuFree(g_kv_v_dev[device]);
        if (!gpu_ok(gpuMalloc(&g_kv_v_dev[device], kv_v_bytes), "KV V allocation")) return 0;
        gpuMemset(g_kv_v_dev[device], 0, kv_v_bytes);
        g_kv_v_cap[device] = kv_v_bytes;
    }

    return 1;
}

extern "C" void
picolm_gpu_kv_free(void) {
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        if (g_kv_k_dev[i]) { gpuFree(g_kv_k_dev[i]); g_kv_k_dev[i] = NULL; g_kv_k_cap[i] = 0; }
        if (g_kv_v_dev[i]) { gpuFree(g_kv_v_dev[i]); g_kv_v_dev[i] = NULL; g_kv_v_cap[i] = 0; }
    }
}

/* Phase 2: allocate the fixed-size device-resident pipeline buffers for
 * model_forward_gpu() decode (S=1 only). Called once at model load,
 * right after picolm_gpu_kv_alloc(), with sizes derived from model
 * config. Idempotent: safe to call again with the same sizes (no-op if
 * ctx->pipe_ready already set for this device -- these buffers never
 * need to grow, unlike reserve()-based scratch, since decode is always
 * S=1). Returns 1 on success. */
extern "C" int
picolm_gpu_pipeline_alloc(int dim, int q_dim, int kv_dim, int ffn_hidden, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (ctx->pipe_ready) return 1;

    size_t db = (size_t)dim * sizeof(float);
    size_t qb = (size_t)q_dim * sizeof(float);
    size_t kvb = (size_t)kv_dim * sizeof(float);
    size_t fb = (size_t)ffn_hidden * sizeof(float);

    int ok = 1;
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_x, db), "pipe_x alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_xb, db), "pipe_xb alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_q, qb), "pipe_q alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_k, kvb), "pipe_k alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_v, kvb), "pipe_v alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_attn_out, qb), "pipe_attn_out alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_ffn_norm, db), "pipe_ffn_norm alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_gate, fb), "pipe_gate alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_up, fb), "pipe_up alloc");
    if (!ok) return 0;

    ctx->pipe_ready = 1;
    return 1;
}

/* Allocate prefill batch buffers: [max_seq_len][dim] for S>1 pipeline */
extern "C" int
picolm_gpu_pipeline_batch_alloc(int dim, int q_dim, int kv_dim, int ffn_hidden,
                                 int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (ctx->pipe_b_ready) return 1;

    size_t bsz = (size_t)max_seq_len;
    size_t db = bsz * dim * sizeof(float);
    size_t qb = bsz * q_dim * sizeof(float);
    size_t kvb = bsz * kv_dim * sizeof(float);
    size_t fb = bsz * ffn_hidden * sizeof(float);

    int ok = 1;
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_x_b, db), "pipe_x_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_xb_b, db), "pipe_xb_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_q_b, qb), "pipe_q_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_k_b, kvb), "pipe_k_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_v_b, kvb), "pipe_v_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_attn_out_b, qb), "pipe_attn_out_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_ffn_norm_b, db), "pipe_ffn_norm_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_gate_b, fb), "pipe_gate_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_up_b, fb), "pipe_up_b alloc");
    if (!ok) return 0;

    ctx->pipe_b_ready = 1;
    return 1;
}

extern "C" void
picolm_gpu_pipeline_free(void) {
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!ctx->pipe_ready) continue;
        if (ctx->pipe_x) gpuFree(ctx->pipe_x);
        if (ctx->pipe_xb) gpuFree(ctx->pipe_xb);
        if (ctx->pipe_q) gpuFree(ctx->pipe_q);
        if (ctx->pipe_k) gpuFree(ctx->pipe_k);
        if (ctx->pipe_v) gpuFree(ctx->pipe_v);
        if (ctx->pipe_attn_out) gpuFree(ctx->pipe_attn_out);
        if (ctx->pipe_ffn_norm) gpuFree(ctx->pipe_ffn_norm);
        if (ctx->pipe_gate) gpuFree(ctx->pipe_gate);
        if (ctx->pipe_up) gpuFree(ctx->pipe_up);
        ctx->pipe_x = ctx->pipe_xb = ctx->pipe_q = ctx->pipe_k = ctx->pipe_v =
            ctx->pipe_attn_out = ctx->pipe_ffn_norm = ctx->pipe_gate = ctx->pipe_up = NULL;
        ctx->pipe_ready = 0;
    }
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!ctx->pipe_b_ready) continue;
        if (ctx->pipe_x_b) gpuFree(ctx->pipe_x_b);
        if (ctx->pipe_xb_b) gpuFree(ctx->pipe_xb_b);
        if (ctx->pipe_q_b) gpuFree(ctx->pipe_q_b);
        if (ctx->pipe_k_b) gpuFree(ctx->pipe_k_b);
        if (ctx->pipe_v_b) gpuFree(ctx->pipe_v_b);
        if (ctx->pipe_attn_out_b) gpuFree(ctx->pipe_attn_out_b);
        if (ctx->pipe_ffn_norm_b) gpuFree(ctx->pipe_ffn_norm_b);
        if (ctx->pipe_gate_b) gpuFree(ctx->pipe_gate_b);
        if (ctx->pipe_up_b) gpuFree(ctx->pipe_up_b);
        ctx->pipe_x_b = ctx->pipe_xb_b = ctx->pipe_q_b = ctx->pipe_k_b = ctx->pipe_v_b =
            ctx->pipe_attn_out_b = ctx->pipe_ffn_norm_b = ctx->pipe_gate_b = ctx->pipe_up_b = NULL;
        ctx->pipe_b_ready = 0;
    }
}

/* Device pointer accessors, so model.c doesn't need gpu_device_ctx_t's
 * internal layout (it's file-static to backend_gpu.cu). */
extern "C" float *picolm_gpu_pipe_x(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_x : NULL; }
extern "C" float *picolm_gpu_pipe_xb(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_xb : NULL; }
extern "C" float *picolm_gpu_pipe_q(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_q : NULL; }
extern "C" float *picolm_gpu_pipe_k(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_k : NULL; }
extern "C" float *picolm_gpu_pipe_v(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_v : NULL; }
extern "C" float *picolm_gpu_pipe_attn_out(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_attn_out : NULL; }
extern "C" float *picolm_gpu_pipe_ffn_norm(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_ffn_norm : NULL; }
extern "C" float *picolm_gpu_pipe_gate(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_gate : NULL; }
extern "C" float *picolm_gpu_pipe_up(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_up : NULL; }

extern "C" float *picolm_gpu_pipe_x_b(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_x_b : NULL; }
extern "C" float *picolm_gpu_pipe_xb_b(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_xb_b : NULL; }
extern "C" float *picolm_gpu_pipe_q_b(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_q_b : NULL; }
extern "C" float *picolm_gpu_pipe_k_b(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_k_b : NULL; }
extern "C" float *picolm_gpu_pipe_v_b(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_v_b : NULL; }
extern "C" float *picolm_gpu_pipe_attn_out_b(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_attn_out_b : NULL; }
extern "C" float *picolm_gpu_pipe_ffn_norm_b(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_ffn_norm_b : NULL; }
extern "C" float *picolm_gpu_pipe_gate_b(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_gate_b : NULL; }
extern "C" float *picolm_gpu_pipe_up_b(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_up_b : NULL; }

/* Bit-exact device port of the host fp32_to_fp16() in quant.c -- NOT
 * CUDA's __float2half (different rounding/tie behavior in edge cases),
 * so that PICOLM_DBG_PIPELINE logit comparisons against the CPU path
 * stay meaningful all the way through the KV cache, not just up to
 * whatever tolerance a different rounding rule would introduce. */
__device__ __forceinline__ uint16_t
picolm_gpu_fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (((bits >> 23) & 0xFF) == 0) return (uint16_t)sign;
    if (((bits >> 23) & 0xFF) == 0xFF)
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t round_bit = 1U << (shift - 1);
        mant = (mant + round_bit) >> shift;
        return (uint16_t)(sign | mant);
    }
    mant += 0x00001000;
    if (mant & 0x00800000) {
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Pack a device-resident F32 [n_kv_heads*head_dim] vector to F16 and
 * write it directly into the device KV cache row for (layer_ordinal,
 * pos). One thread per element, grid-stride, no shared memory. This
 * replaces the D2H -> CPU convert -> H2D round trip that
 * model_forward_gpu() otherwise needs every layer of every token: with
 * this kernel the KV store never touches the host on the hot path. */
__global__ void
picolm_gpu_kv_pack_store_kernel(uint16_t *dst_row, const float *src, int n) {
    for (int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x; i < n;
         i += (int)gridDim.x * gpuBlockDim_x) {
        dst_row[i] = picolm_gpu_fp32_to_fp16(src[i]);
    }
}

/* Device-native KV store: src_dev is pipe_k/pipe_v (F32, already on
 * device, kv_dim = n_kv_heads*head_dim elements). No transfer, no sync --
 * same ctx->stream ordering argument as the other _dev primitives.
 * Caller must call this (K then V) before picolm_gpu_attention_decode_dev
 * for the same (layer_ordinal, pos), same as picolm_gpu_kv_store_rows. */
extern "C" int
picolm_gpu_kv_store_dev(int is_k, int layer_ordinal, int pos,
                         const float *src_dev, int n_kv_heads, int head_dim,
                         int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;

    int kv_dim = n_kv_heads * head_dim;
    size_t row_bytes = (size_t)kv_dim * sizeof(uint16_t);
    uint16_t *dst_base = is_k ? g_kv_k_dev[device] : g_kv_v_dev[device];
    uint16_t *dst_row = dst_base
        + ((size_t)layer_ordinal * max_seq_len * row_bytes
           + (size_t)pos * row_bytes) / sizeof(uint16_t);

    int n_threads = min(kv_dim, 256);
    int n_blocks = (kv_dim + n_threads - 1) / n_threads;
    picolm_gpu_kv_pack_store_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        dst_row, src_dev, kv_dim);
    if (!gpu_ok(gpuGetLastError(), "kv pack+store (dev)")) return 0;
    return 1;
}

/* Batched variant: src_dev is [S][kv_dim] contiguous F32, positions
 * start_pos..start_pos+S-1. Reuses picolm_gpu_kv_pack_store_kernel --
 * since the device KV cache has no per-row padding, S contiguous source
 * rows map to S contiguous destination rows. One launch for the whole
 * prefill chunk. */
extern "C" int
picolm_gpu_kv_store_dev_batched(int is_k, int layer_ordinal, int start_pos, int n_positions,
                                 const float *src_dev, int n_kv_heads, int head_dim,
                                 int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;

    int kv_dim = n_kv_heads * head_dim;
    size_t row_bytes = (size_t)kv_dim * sizeof(uint16_t);
    uint16_t *dst_base = is_k ? g_kv_k_dev[device] : g_kv_v_dev[device];
    uint16_t *dst_row = dst_base
        + ((size_t)layer_ordinal * max_seq_len * row_bytes
           + (size_t)start_pos * row_bytes) / sizeof(uint16_t);

    int total = n_positions * kv_dim;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 4096);
    picolm_gpu_kv_pack_store_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        dst_row, src_dev, total);
    if (!gpu_ok(gpuGetLastError(), "kv pack+store batched (dev)")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_kv_store_rows(int is_k, int layer_ordinal, int start_pos, int n_positions,
                          const void *host_rows, size_t row_bytes,
                          int n_kv_heads, int head_dim, int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;

    /* Sanity: row_bytes must match the F16 GQA row size this cache was
     * sized for. If a caller passes a quantized row_bytes here, refuse
     * rather than silently corrupting the cache. */
    size_t expect_row_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    if (row_bytes != expect_row_bytes) return 0;

    uint16_t *dst_base = is_k ? g_kv_k_dev[device] : g_kv_v_dev[device];
    /* Layout: [layer_ordinal][pos][kv_head][head_dim], contiguous rows.
     * n_positions contiguous rows starting at start_pos == one memcpy. */
    size_t layer_off_bytes = (size_t)layer_ordinal * max_seq_len * row_bytes;
    size_t pos_off_bytes = (size_t)start_pos * row_bytes;
    uint8_t *dst = (uint8_t *)dst_base + layer_off_bytes + pos_off_bytes;

    size_t copy_bytes = (size_t)n_positions * row_bytes;
    if (!gpu_ok(gpuMemcpyAsync(dst, host_rows, copy_bytes,
                               gpuMemcpyHostToDevice, ctx->stream),
                "kv_store_rows async copy"))
        return 0;
    return 1;
}

extern "C" int
picolm_gpu_attention_decode(float *xb_out, const float *q_host,
                             int layer_ordinal, int pos,
                             int n_heads, int n_kv_heads, int head_dim,
                             int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0; /* stack array limit */

    int kv_mul = n_heads / n_kv_heads;
    if (kv_mul < 1 || kv_mul > 8) return 0;

    size_t x_bytes = (size_t)n_heads * head_dim * sizeof(float);
    size_t y_bytes = x_bytes;
    if (!reserve(&ctx->x, &ctx->x_cap, x_bytes) ||
        !reserve(&ctx->y, &ctx->y_cap, y_bytes)) return 0;

    /* Upload Q */
    if (!gpu_ok(gpuMemcpy(ctx->x, q_host, x_bytes, gpuMemcpyHostToDevice),
                "attn decode Q upload")) return 0;

    /* Compute strides for kernel */
    /* Within a layer: [pos][kv_head][head_dim] in uint16_t */
    /* pos stride: bytes from pos P to P+1 for any kv_head */
    size_t kv_pos_stride_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    /* kv_head stride: bytes from kv_head H to H+1 at same pos */
    size_t kv_head_stride_bytes = head_dim * sizeof(uint16_t);

    /* Shared memory: K(head_dim u16) + V(head_dim u16) + reduce(256 float)
     * + acc[kv_mul][head_dim] float + max_score[kv_mul] + sum_exp[kv_mul]
     * + rescale_sh + weight_sh (static __shared__, 8 bytes) */
    size_t shared_bytes = 2 * head_dim * sizeof(uint16_t) + 256 * sizeof(float)
                         + (size_t)kv_mul * head_dim * sizeof(float)
                         + (size_t)kv_mul * sizeof(float)
                         + (size_t)kv_mul * sizeof(float)
                         + 2 * sizeof(float); /* rescale_sh + weight_sh */
    int block_threads = 256;

    dim3 grid((unsigned)n_kv_heads, 1, 1);
    picolm_gpu_attention_decode_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
        ctx->y, ctx->x,
        g_kv_k_dev[device], g_kv_v_dev[device],
        layer_ordinal, pos, n_heads, n_kv_heads, head_dim, max_seq_len,
        kv_pos_stride_bytes, kv_head_stride_bytes);

    if (!gpu_ok(gpuGetLastError(), "attn decode kernel") ||
        !gpu_ok(gpuDeviceSynchronize(), "attn decode sync")) return 0;

    if (!gpu_ok(gpuMemcpy(xb_out, ctx->y, y_bytes, gpuMemcpyDeviceToHost),
                "attn decode output download")) return 0;
    return 1;
}

/* Device-native decode attention: q_dev and xb_out_dev are already
 * device-resident (Phase 2 pipeline buffers, e.g. the output of the Q
 * projection matmul_dev and rope_apply, both in place on the same
 * buffer). No H2D, no D2H, no gpuDeviceSynchronize() -- same ordering
 * argument as picolm_gpu_matmul_dev: this kernel and picolm_gpu_kv_store_rows
 * (the KV write for this position) are both launched on ctx->stream, so
 * in-stream ordering guarantees the K/V write for `pos` is visible before
 * this kernel reads it, PROVIDED the K/V store call for `pos` happens
 * first in program order on the same device/stream -- caller must store
 * K/V before calling this. q_dev/xb_out_dev must not alias ctx->x/ctx->y. */
extern "C" int
picolm_gpu_attention_decode_dev(float *xb_out_dev, const float *q_dev,
                                 int layer_ordinal, int pos,
                                 int n_heads, int n_kv_heads, int head_dim,
                                 int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0; /* stack array limit */

    int kv_mul = n_heads / n_kv_heads;
    if (kv_mul < 1 || kv_mul > 8) return 0;

    size_t kv_pos_stride_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    size_t kv_head_stride_bytes = head_dim * sizeof(uint16_t);
    size_t shared_bytes = 2 * head_dim * sizeof(uint16_t) + 256 * sizeof(float)
                         + (size_t)kv_mul * head_dim * sizeof(float)
                         + (size_t)kv_mul * sizeof(float)
                         + (size_t)kv_mul * sizeof(float);
    int block_threads = 256;

    dim3 grid((unsigned)n_kv_heads, 1, 1);
    picolm_gpu_attention_decode_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
        xb_out_dev, q_dev,
        g_kv_k_dev[device], g_kv_v_dev[device],
        layer_ordinal, pos, n_heads, n_kv_heads, head_dim, max_seq_len,
        kv_pos_stride_bytes, kv_head_stride_bytes);

    if (!gpu_ok(gpuGetLastError(), "attn decode kernel (dev)")) return 0;
    return 1;
}

/* Total dynamic shared memory picolm_gpu_attention_prefill_kernel needs:
 * K tile + V tile (u16) + reduce (256 float) + acc[ATTN_TILE_Q][head_dim]
 * (float) + max_score[ATTN_TILE_Q] + sum_exp[ATTN_TILE_Q] (float).
 * For head_dim=128 this is ~50KB -- over the default 48KB/block limit,
 * so the caller must opt in via gpuFuncSetAttribute before launching
 * with this size (see ensure_attn_prefill_shared_mem below). */
static size_t
attn_prefill_shared_bytes(int head_dim) {
    /* Dynamic shared memory (extern __shared__): K tile + V tile + reduce +
     * acc + max_score + sum_exp. Plus 8 bytes of static __shared__ for
     * rescale_sh/weight_sh. Total must be set via gpuFuncSetAttribute. */
    return 2 * (size_t)ATTN_TILE_K * head_dim * sizeof(uint16_t)
         + 256 * sizeof(float)
         + (size_t)ATTN_TILE_Q * head_dim * sizeof(float)
         + (size_t)ATTN_TILE_Q * sizeof(float)
         + (size_t)ATTN_TILE_Q * sizeof(float)
         + 2 * sizeof(float); /* rescale_sh + weight_sh (static __shared__) */
}

/* Opts the kernel into a larger dynamic shared memory limit if needed.
 * Idempotent: tracks the largest size already configured so repeated
 * calls (every prefill call, every layer) are a cheap no-op after the
 * first. Returns 1 if `bytes` is safe to launch with, 0 if the opt-in
 * itself failed (e.g. device doesn't support this much shared memory --
 * would need a fallback path, not expected on GB10/Blackwell). */
static int
ensure_attn_prefill_shared_mem(size_t bytes) {
    static size_t configured = 49152; /* default limit, no opt-in needed under this */
    if (bytes <= configured) return 1;
    if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_gpu_attention_prefill_kernel,
                                     gpuFuncAttributeMaxDynamicSharedMemorySize, (int)bytes),
                "attn prefill shared mem opt-in"))
        return 0;
    configured = bytes;
    return 1;
}

extern "C" int
picolm_gpu_attention_prefill(float *xb_out, const float *q_host,
                              int layer_ordinal, int start_pos, int n_tokens,
                              int n_heads, int n_kv_heads, int head_dim,
                              int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0;

    size_t x_bytes = (size_t)n_tokens * n_heads * head_dim * sizeof(float);
    size_t y_bytes = x_bytes;
    if (!reserve(&ctx->x, &ctx->x_cap, x_bytes) ||
        !reserve(&ctx->y, &ctx->y_cap, y_bytes)) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, q_host, x_bytes, gpuMemcpyHostToDevice),
                "attn prefill Q upload")) return 0;

    size_t kv_pos_stride_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    size_t kv_head_stride_bytes = head_dim * sizeof(uint16_t);

    int n_tiles_q = (n_tokens + ATTN_TILE_Q - 1) / ATTN_TILE_Q;
    size_t shared_bytes = attn_prefill_shared_bytes(head_dim);
    if (!ensure_attn_prefill_shared_mem(shared_bytes)) return 0;
    int block_threads = 128; /* enough for head_dim up to 256 */

    dim3 grid((unsigned)n_heads, (unsigned)n_tiles_q, 1);
    picolm_gpu_attention_prefill_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
        ctx->y, ctx->x,
        g_kv_k_dev[device], g_kv_v_dev[device],
        layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
        kv_pos_stride_bytes, kv_head_stride_bytes);

    if (!gpu_ok(gpuGetLastError(), "attn prefill kernel") ||
        !gpu_ok(gpuDeviceSynchronize(), "attn prefill sync")) return 0;

    if (!gpu_ok(gpuMemcpy(xb_out, ctx->y, y_bytes, gpuMemcpyDeviceToHost),
                "attn prefill output download")) return 0;
    return 1;
}

/* Device-native prefill attention: xb_out_dev and q_dev are already
 * device-resident. No H2D, no D2H, no gpuDeviceSynchronize(). */
extern "C" int
picolm_gpu_attention_prefill_dev(float *xb_out_dev, const float *q_dev,
                                  int layer_ordinal, int start_pos, int n_tokens,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0;

    size_t kv_pos_stride_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    size_t kv_head_stride_bytes = head_dim * sizeof(uint16_t);

    int n_tiles_q = (n_tokens + ATTN_TILE_Q - 1) / ATTN_TILE_Q;
    size_t shared_bytes = attn_prefill_shared_bytes(head_dim);
    if (!ensure_attn_prefill_shared_mem(shared_bytes)) return 0;
    int block_threads = 128;

    dim3 grid((unsigned)n_heads, (unsigned)n_tiles_q, 1);
    picolm_gpu_attention_prefill_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
        xb_out_dev, q_dev,
        g_kv_k_dev[device], g_kv_v_dev[device],
        layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
        kv_pos_stride_bytes, kv_head_stride_bytes);
    if (!gpu_ok(gpuGetLastError(), "attn prefill (dev)")) return 0;
    return 1;
}

/* ================================================================
 * Phase 2: Device-resident elementwise kernels (stubs)
 * ================================================================ */

/* RMSNorm kernel: out[d] = x[d] * rsqrt(mean(x^2) + eps) * weight[d].
 * Single block, grid-stride loop -- correct for any dim (an earlier
 * version returned early before __syncthreads() for threads with
 * d >= dim, which deadlocks/UB whenever dim isn't an exact multiple of
 * blockDim). Launched with grid=1 by the host wrapper below. */
__global__ void
picolm_gpu_rmsnorm_kernel(float *out, const float *x, const float *weight,
                           int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = gpuThreadIdx_x; i < dim; i += gpuBlockDim_x) {
        sum_sq += x[i] * x[i];
    }
    __shared__ float ssum[256];
    ssum[gpuThreadIdx_x] = sum_sq;
    gpuSyncthreads();
    for (int s = gpuBlockDim_x / 2; s > 0; s >>= 1) {
        if (gpuThreadIdx_x < s) ssum[gpuThreadIdx_x] += ssum[gpuThreadIdx_x + s];
        gpuSyncthreads();
    }
    float rms = sqrtf(ssum[0] / dim + eps);
    float inv_rms = 1.0f / rms;

    for (int d = gpuThreadIdx_x; d < dim; d += gpuBlockDim_x) {
        out[d] = x[d] * inv_rms * weight[d];
    }
}

/* Batched rmsnorm: one block per row (S rows), same reduction per block.
 * weight is shared across all rows. Mirrors how matmul_dev batches over S. */
__global__ void
picolm_gpu_rmsnorm_batched_kernel(float *out, const float *x, const float *weight,
                                   int dim, float eps) {
    int row = (int)gpuBlockIdx_x;
    const float *xr = x + (size_t)row * dim;
    float *outr = out + (size_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = gpuThreadIdx_x; i < dim; i += gpuBlockDim_x) {
        sum_sq += xr[i] * xr[i];
    }
    __shared__ float ssum[256];
    ssum[gpuThreadIdx_x] = sum_sq;
    gpuSyncthreads();
    for (int s = gpuBlockDim_x / 2; s > 0; s >>= 1) {
        if (gpuThreadIdx_x < s) ssum[gpuThreadIdx_x] += ssum[gpuThreadIdx_x + s];
        gpuSyncthreads();
    }
    float inv_rms = 1.0f / sqrtf(ssum[0] / dim + eps);

    for (int d = gpuThreadIdx_x; d < dim; d += gpuBlockDim_x) {
        outr[d] = xr[d] * inv_rms * weight[d];
    }
}

/* Batched RoPE: each element computes its own absolute position's table
 * row. x is [S][n_heads][head_dim] contiguous, positions start_pos..start_pos+S-1.
 * One launch for the whole prefill chunk. */
__global__ void
picolm_gpu_rope_batched_kernel(float *x, int n_heads, int head_dim,
                                const float *cos_tbl, const float *sin_tbl,
                                int half_dim, int start_pos, int S) {
    int idx = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int per_row = n_heads * head_dim;
    int total = S * per_row;
    for (int i = idx; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        int row = i / per_row;
        int rem = i % per_row;
        int d = rem % head_dim;
        if (d >= half_dim) continue; /* second half handled by paired thread */

        int pos = start_pos + row;
        int h = rem / head_dim;
        float *xr = x + (size_t)row * per_row;
        int h1 = d, h2 = d + half_dim;
        float x1 = xr[h * head_dim + h1];
        float x2 = xr[h * head_dim + h2];
        float c = cos_tbl[(size_t)pos * half_dim + h1];
        float s = sin_tbl[(size_t)pos * half_dim + h1];
        xr[h * head_dim + h1] = x1 * c - x2 * s;
        xr[h * head_dim + h2] = x1 * s + x2 * c;
    }
}

/* RoPE kernel: applies pairwise rotary position embedding */
__global__ void
picolm_gpu_rope_kernel(float *x, int n_heads, int head_dim,
                        const float *cos_tbl, const float *sin_tbl,
                        int half_dim) {
    int idx = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int total = n_heads * head_dim;
    for (int i = idx; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        int h = i / head_dim;
        int d = i % head_dim;
        if (d >= half_dim * 2) {
            /* Beyond rope_dim, copy as-is */
            /* Already in place */
        } else if (d < half_dim) {
            int h1 = d;
            int h2 = d + half_dim;
            float x1 = x[i];
            float x2 = x[h * head_dim + h2];
            float c = cos_tbl[h1];
            float s = sin_tbl[h1];
            x[h * head_dim + h1] = x1 * c - x2 * s;
            x[h * head_dim + h2] = x1 * s + x2 * c;
        }
        /* d >= half_dim: already updated by the paired thread */
    }
}

/* Residual add kernel: out[i] = a[i] + b[i] */
__global__ void
picolm_gpu_residual_add_kernel(float *out, const float *a, const float *b, int n) {
    int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    for (; i < n; i += (int)gridDim.x * gpuBlockDim_x) {
        out[i] = a[i] + b[i];
    }
}

/* Phase 2 host API */
extern "C" int
picolm_gpu_rmsnorm(float *out, const float *x, const float *weight,
                    int dim, float eps, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_kernel<<<1, n_threads, 0, ctx->stream>>>(
        out, x, weight, dim, eps);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm kernel")) return 0;
    return 1;
}

/* Batched rmsnorm: one launch for S rows. */
extern "C" int
picolm_gpu_rmsnorm_batched(float *out, const float *x, const float *weight,
                            int dim, float eps, int S, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;

    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_batched_kernel<<<S, n_threads, 0, ctx->stream>>>(
        out, x, weight, dim, eps);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm batched kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_rope_apply(float *x, int n_heads, int head_dim,
                       const float *cos_tbl, const float *sin_tbl,
                       int half_dim, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    int total = n_heads * head_dim;
    int n_threads = 128;
    int n_blocks = min((total + n_threads - 1) / n_threads, 128);
    picolm_gpu_rope_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        x, n_heads, head_dim, cos_tbl, sin_tbl, half_dim);
    if (!gpu_ok(gpuGetLastError(), "rope kernel")) return 0;
    return 1;
}

/* Batched RoPE: cos_tbl_base/sin_tbl_base are the UNOFFSET [max_seq_len][half_dim]
 * base pointers. Each row computes its own position as start_pos + row. */
extern "C" int
picolm_gpu_rope_apply_batched(float *x, int n_heads, int head_dim,
                               const float *cos_tbl_base, const float *sin_tbl_base,
                               int half_dim, int start_pos, int S, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;

    int total = S * n_heads * head_dim;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 4096);
    picolm_gpu_rope_batched_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        x, n_heads, head_dim, cos_tbl_base, sin_tbl_base, half_dim, start_pos, S);
    if (!gpu_ok(gpuGetLastError(), "rope batched kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_residual_add(float *out, const float *a, const float *b,
                         int n, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    int n_threads = 256;
    int n_blocks = min((n + n_threads - 1) / n_threads, 256);
    picolm_gpu_residual_add_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        out, a, b, n);
    if (!gpu_ok(gpuGetLastError(), "residual_add kernel")) return 0;
    return 1;
}

/* Standalone device-native host wrapper. picolm_silu_mul itself was
 * previously only ever launched inline inside picolm_gpu_expert_mlp()
 * (which does its own H2D/D2H); the Phase 2 pipeline needs to call it
 * on already-device-resident gate/up buffers between matmul_dev calls,
 * with no transfer and no per-call sync (same stream-ordering argument
 * as picolm_gpu_matmul_dev / picolm_gpu_attention_decode_dev). Result is
 * written in place into gate_dev. */
extern "C" int
picolm_gpu_silu_mul_dev(float *gate_dev, const float *up_dev, size_t n, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_silu_mul<<<(unsigned)((n + 255) / 256), 256, 0, ctx->stream>>>(
        gate_dev, up_dev, n);
    if (!gpu_ok(gpuGetLastError(), "silu_mul (dev)")) return 0;
    return 1;
}

/* The single sync point for the whole model_forward_gpu() pass: call
 * this exactly once, after the last device-native op (typically the
 * final rmsnorm), before reading anything back via D2H. Every _dev
 * primitive above is launched on ctx->stream with no internal sync, so
 * this is what actually guarantees all of it has completed. */
extern "C" int
picolm_gpu_sync(int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    return gpu_ok(gpuDeviceSynchronize(), "pipeline sync");
}

/* Synchronous memcpy wrapper for model.c (C file, no CUDA types).
 * dir: 1 = H2D, -1 = D2H. Returns 1 on success. */
extern "C" int
picolm_gpu_memcpy(void *dst, const void *src, size_t bytes, int dir, int device) {
    if (!dst || !src || bytes < 1) return 0;
    (void)device; /* device already selected by caller */
    gpuError_t err = (dir > 0) ?
        gpuMemcpy(dst, src, bytes, gpuMemcpyHostToDevice) :
        gpuMemcpy(dst, src, bytes, gpuMemcpyDeviceToHost);
    return gpu_ok(err, "pipeline memcpy");
}

