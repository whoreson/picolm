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
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

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
#define gpuMemAdvise cudaMemAdvise
#define gpuMemLocation cudaMemLocation
#define gpuMemLocationType cudaMemLocationType
#define gpuCudaMemLocationTypeDevice cudaMemLocationTypeDevice
#define gpuCudaMemLocationTypecpu cudaMemLocationTypecpu
#define gpuCudaMemAdviseSetReadMostly cudaMemAdviseSetReadMostly
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
__host__ __device__ static inline unsigned short gpu_fp32_to_fp16(float f) {
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

extern "C" __global__ void
picolm_quant_matmul(float *y, const float *x, const void *weights,
                    gguf_type_t qtype, int S, int I, int O, int row_bytes) {
    /* bytes_per_block: stride between consecutive blocks in memory */
    int bytes_per_block;
    switch (qtype) {
        case GGUF_TYPE_F32:  bytes_per_block = 4; break;      /* 1 float */
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

/* ---- silu_mul kernel ----
 * Element-wise: gate[i] = gate[i] / (1 + exp(-gate[i])) * up[i] */
extern "C" __global__ void
picolm_silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
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

/* HIP WMMA: only on CDNA2+ (gfx940, gfx941, gfx942). gfx906/908 have no WMMA. */
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define PICOLM_GPU_WMMA_AVAILABLE 1
#include <hip/wmma/wmma.h>
#endif
#endif

/* CUDA WMMA: sm_70+ (Volta and newer)
 * Guard must exclude host compilation phase to avoid CUDAFE stub generation
 * failures (wmma types are device-only). Host gets a forward declaration. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define PICOLM_GPU_WMMA_AVAILABLE 1
#endif

/* Forward declarations for host compilation (extern "C" stubs only, no body) */
#ifndef PICOLM_GPU_WMMA_AVAILABLE
extern "C" {
__global__ void picolm_w4a16_matmul(float *y, const float *x, const void *weights,
                                     int M, int K, int N, int block_size);
__global__ void picolm_w4a16_gate_up(float *gate, float *up, const float *x,
                                      const void *gate_weights, const void *up_weights,
                                      int M, int K, int N, int block_size);
}
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

extern "C" __global__ void
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

extern "C" __global__ void
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
    gpuStream_t stream;
    size_t tensor_count, tensor_bytes;
} gpu_device_ctx_t;

static gpu_device_ctx_t g_gpu_ctx[PICOLM_GPU_MAX_DEVICES];
static int g_nctx;

/* Mutex protecting g_gpu_ctx scratch buffer resize (reserve/reserve_pinned).
 * Resize is rare (once per buffer growth), so lock contention is negligible. */
static pthread_mutex_t g_resize_mutex = PTHREAD_MUTEX_INITIALIZER;

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
static __thread int g_current_device = -1;

static int select_ctx(gpu_device_ctx_t *ctx) {
    if (!ctx) return 0;
    if (g_current_device == ctx->device) return 1;
    if (!gpu_ok(gpuSetDevice(ctx->device), "select device")) return 0;
    g_current_device = ctx->device;
    return 1;
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    pthread_mutex_lock(&g_resize_mutex);
    if (*cap >= bytes) { pthread_mutex_unlock(&g_resize_mutex); return 1; }
    if (*ptr) gpuFree(*ptr);
    *ptr = NULL; *cap = 0;
    /* Use managed memory on CUDA for unified memory systems (Grace-Blackwell SoC).
     * On discrete GPUs, the page fault overhead is tolerable for the small scratch
     * buffers (typically < 10 MB). On SoC, this eliminates the H2D/D2H copies
     * entirely since CPU and GPU share coherent memory.
     * HIP uses regular malloc since hipMallocManaged has different semantics. */
    gpuError_t err;
#ifdef __HIP__
    err = gpuMalloc(ptr, bytes);
#else
    err = gpuMallocManaged(ptr, bytes);
    if (err == gpuSuccess && *ptr) {
        /* Advise: prefer GPU placement for these scratch buffers.
         * They are written by CPU (H2D equivalent), read by GPU kernel, then
         * read by CPU (D2H equivalent). GPU placement minimizes NVLink traffic. */
        gpuMemLocation loc;
        loc.type = gpuCudaMemLocationTypeDevice;
        loc.id = -1; /* any device */
        (void)gpuMemAdvise(*ptr, bytes, gpuCudaMemAdviseSetReadMostly, loc);
    }
#endif
    if (!gpu_ok(err, "scratch allocation")) {
        pthread_mutex_unlock(&g_resize_mutex);
        return 0;
    }
    *cap = bytes;
    pthread_mutex_unlock(&g_resize_mutex);
    return 1;
}

static int reserve_pinned(float **ptr, size_t *cap, size_t bytes) {
    pthread_mutex_lock(&g_resize_mutex);
    if (*cap >= bytes) { pthread_mutex_unlock(&g_resize_mutex); return 1; }
    if (*ptr) gpuFreeHost(*ptr);
    *ptr = NULL; *cap = 0;
    if (!gpu_ok(gpuMallocHost(ptr, bytes), "pinned staging allocation")) {
        pthread_mutex_unlock(&g_resize_mutex);
        return 0;
    }
    *cap = bytes;
    pthread_mutex_unlock(&g_resize_mutex);
    return 1;
}

/* ---- Public API ---- */

int picolm_gpu_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > PICOLM_GPU_MAX_DEVICES) return 0;
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
    for (int i = 0; i < g_nctx; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->x) gpuFree(ctx->x);
        if (ctx->y) gpuFree(ctx->y);
        if (ctx->gate) gpuFree(ctx->gate);
        if (ctx->up) gpuFree(ctx->up);
        if (ctx->host_x) gpuFreeHost(ctx->host_x);
        if (ctx->host_y) gpuFreeHost(ctx->host_y);
        if (ctx->stream) gpuStreamDestroy(ctx->stream);
        ctx->x = ctx->y = ctx->gate = ctx->up = NULL;
        ctx->host_x = ctx->host_y = NULL;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->host_x_cap = ctx->host_y_cap = 0;
        ctx->stream = NULL;
    }
    g_nctx = 0;
}

/* Detect if we're on a unified memory SoC (Grace-Blackwell, Apple Silicon, etc.) */
static int is_unified_memory(void) {
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

/* Map GGUF_TYPE to block size and values per block */
static int gguf_block_size(gguf_type_t qtype) {
    switch (qtype) {
    case 0:  return 0;    /* F32: no blocks */
    case 2:  return 18;   /* Q4_0: 18 bytes per 32 values */
    case 8:  return 34;   /* Q8_0: 34 bytes per 32 values */
    case 12: return GPU_BLOCK_Q4_K_SIZE;  /* Q4_K: 144 bytes per 256 values */
    case 41: return 18;   /* Q1_0: 18 bytes per 128 values */
    case 42: return 34;   /* Q2_0: 34 bytes per 128 values */
    default: return 0;
    }
}

int picolm_gpu_tensor_upload(picolm_gpu_tensor_t **tensor,
                              const void *weights,
                              gguf_type_t qtype, int I, int O, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!tensor || !weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    int bs = gguf_block_size(qtype);
    if (!bs && qtype != 0) return 0;
    if (*tensor) return 1; /* idempotent */

    /* Compute row bytes */
    size_t row_bytes;
    int vals_per_block;
    if (qtype == 0) {
        row_bytes = (size_t)I * sizeof(float);
        vals_per_block = 1;
    } else if (qtype == 12) {
        vals_per_block = 256;
    } else if (qtype == 41 || qtype == 42) {
        vals_per_block = 128;
    } else {
        vals_per_block = 32;
    }
    row_bytes = (size_t)((I + vals_per_block - 1) / vals_per_block) * bs;
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
    *tensor = t;
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

    /* On unified memory (CUDA managed), x and ctx->x are the same address space.
     * We can write directly into ctx->x from CPU, then the GPU kernel reads it.
     * On discrete GPUs (HIP or fallback), we still need explicit copies. */
#ifdef __HIP__
    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "input upload")) return 0;
#else
    /* Managed memory: CPU writes directly, GPU reads directly.
     * memcpy is fast on unified memory (same physical RAM via C2C interconnect). */
    memcpy(ctx->x, x, xb);
#endif

    dim3 grid((unsigned)O, (unsigned)S);
    picolm_quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights,
                                        t->qtype, S, I, O,
                                        (int)t->row_bytes);
    if (!gpu_ok(gpuGetLastError(), "matmul launch") ||
        !gpu_ok(gpuDeviceSynchronize(), "matmul sync")) return 0;

    /* Managed memory: GPU writes directly to ctx->y, CPU reads directly. */
#ifdef __HIP__
    if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "output download")) return 0;
#else
    memcpy(y, ctx->y, yb);
#endif
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

#ifdef __HIP__
    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "expert input")) return 0;
#else
    memcpy(ctx->x, x, xb);
#endif

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
#ifdef __HIP__
    if (!gpu_ok(gpuMemcpy(y, ctx->y, xb, gpuMemcpyDeviceToHost), "expert output")) return 0;
#else
    memcpy(y, ctx->y, xb);
#endif
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

    #ifdef __HIP__
    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "w4a16 input")) return 0;
#else
    memcpy(ctx->x, x, xb);
#endif

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

#ifdef __HIP__
    if (!gpu_ok(gpuMemcpy(ctx->x, x, xb, gpuMemcpyHostToDevice), "w4a16 input")) return 0;
#else
    memcpy(ctx->x, x, xb);
#endif

    dim3 grid((unsigned)((N + 63) / 64), (unsigned)((M + 15) / 16));
    picolm_w4a16_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, M, K, N, t->block_size);
    if (!gpu_ok(gpuGetLastError(), "w4a16 matmul") ||
        !gpu_ok(gpuDeviceSynchronize(), "w4a16 sync")) return 0;

#ifdef __HIP__
    if (!gpu_ok(gpuMemcpy(y, ctx->y, yb, gpuMemcpyDeviceToHost), "w4a16 output")) return 0;
#else
    memcpy(y, ctx->y, yb);
#endif
    return 1;
#else
    (void)t; (void)y; (void)x; (void)S; (void)device;
    return 0;
#endif
}

