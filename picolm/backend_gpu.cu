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

__device__ static inline float dequant_bf16(const void *weights, int i) {
    const uint16_t *w = (const uint16_t *)weights;
    /* Zero-extend BF16 to F32: shift upper 16 bits, reinterpret as float.
     * __bfloat162float is BROKEN on CUDA 13.0 / sm_121 ARM64 (returns raw
     * uint16 value, e.g. 16256 instead of 1.0 for 0x3f80).
     * Use portable bit manipulation matching the CPU bf16_to_fp32(). */
    union { uint32_t u; float f; } o;
    o.u = ((uint32_t)w[i]) << 16;
    return o.f;
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

/* Q2_K dequant: 256 values in 84 bytes.
 * Layout: scales[16] (4-bit sc+mn each), qs[64] (2-bit quants), d (fp16), dmin (fp16).
 * Structure: 2 chunks of 128 values each.
 * Each chunk: 4 groups of 16 values. Each group uses a bit-shift (0,2,4,6) over 32 qs bytes.
 *   Group 0: (qs[0..15] >> 0) & 3, scale[0]
 *   Group 1: (qs[16..31] >> 0) & 3, scale[1]
 *   Group 2: (qs[0..15] >> 2) & 3, scale[2]
 *   Group 3: (qs[16..31] >> 2) & 3, scale[3]
 *   Group 4: (qs[0..15] >> 4) & 3, scale[4]
 *   Group 5: (qs[16..31] >> 4) & 3, scale[5]
 *   Group 6: (qs[0..15] >> 6) & 3, scale[6]
 *   Group 7: (qs[16..31] >> 6) & 3, scale[7]
 * Then same pattern for second chunk (qs[32..63], scales[8..15]).
 * Each scale byte: low nibble = scale, high nibble = min.
 * Result = d * sc * q - dmin * mn. */
__device__ static inline float dequant_q2_K(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    const uint8_t *scales = b;
    const uint8_t *qs = b + 16;
    uint16_t d_raw = b[80] | ((uint16_t)b[81] << 8);
    uint16_t dm_raw = b[82] | ((uint16_t)b[83] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    float dmin = gpu_fp16_to_fp32(dm_raw);

    /* i in 0..255 */
    int chunk = i / 128;              /* 0 or 1 */
    int i_in_chunk = i % 128;         /* 0..127 */
    int group = i_in_chunk / 16;      /* 0..7 */
    int l = i_in_chunk % 16;          /* 0..15 */

    int shift = (group % 4) * 2;      /* 0, 2, 4, 6 */
    int half = group / 4;             /* 0 or 1 (first or second half of qs bytes) */
    int scale_idx = chunk * 8 + group; /* 0..15 */

    uint8_t sc_byte = scales[scale_idx];
    uint8_t sc = sc_byte & 0xF;
    uint8_t mn = sc_byte >> 4;

    int qs_base = chunk * 32 + half * 16;
    int v = (qs[qs_base + l] >> shift) & 3;

    return d * (float)sc * (float)v - dmin * (float)mn;
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

/* Q6_K dequant: 256 values in 210 bytes.
 * Layout: ql[128] (4-bit low), qh[64] (2-bit high), scales[16] (int8), d (fp16).
 * Each element: 6-bit signed value (biased by 32: stored [0..63], actual [-32..31]).
 * 16 scales, one per 16-element sub-block.
 * Result = d * scale[scale_idx] * (quantized_value - 32). */
__device__ static inline float dequant_q6_K(const void *blk, int i) {
    const uint8_t *ql = (const uint8_t *)blk;
    const uint8_t *qh = ql + 128;
    const int8_t *scales = (const int8_t *)(qh + 64);
    uint16_t d_raw = ((const uint16_t *)(scales + 16))[0];
    float d = gpu_fp16_to_fp32(d_raw);

    /* Which 128-element half, and which element within it */
    int chunk = i / 128;            /* 0 or 1 */
    int l = i % 128;                /* 0..127 */

    const uint8_t *ql_c = ql + chunk * 64;
    const uint8_t *qh_c = qh + chunk * 32;

    /* Determine which of the 4 elements (q1/q2/q3/q4) this is */
    int sub_chunk = l / 32;         /* 0=q1, 1=q2, 2=q3, 3=q4 */
    int sub_l = l % 32;             /* 0..31 */
    int is = chunk * 8;             /* scale chunk base: 0 or 8 */
    int sc_idx = is + (sub_chunk * 2) + (sub_l / 16);  /* 0..15 */

    /* Extract 6-bit value: 4 bits from ql, 2 bits from qh */
    uint8_t ql_val, qh_val;
    if (sub_chunk == 0) {           /* q1: ql[sub_l]&0xF, qh[sub_l]>>0 */
        ql_val = ql_c[sub_l] & 0xF;
        qh_val = (qh_c[sub_l] >> 0) & 3;
    } else if (sub_chunk == 1) {    /* q2: ql[sub_l+32]&0xF, qh[sub_l]>>2 */
        ql_val = ql_c[sub_l + 32] & 0xF;
        qh_val = (qh_c[sub_l] >> 2) & 3;
    } else if (sub_chunk == 2) {    /* q3: ql[sub_l]>>4, qh[sub_l]>>4 */
        ql_val = ql_c[sub_l] >> 4;
        qh_val = (qh_c[sub_l] >> 4) & 3;
    } else {                        /* q4: ql[sub_l+32]>>4, qh[sub_l]>>6 */
        ql_val = ql_c[sub_l + 32] >> 4;
        qh_val = (qh_c[sub_l] >> 6) & 3;
    }

    int q = (ql_val | (qh_val << 4)) - 32;
    return d * (float)scales[sc_idx] * (float)q;
}

/* Q5_K dequant: 256 values in 176 bytes.
 * Layout: d (fp16), dm (fp16), scales[12] (packed 6-bit sc+mn via get_scale_min_k4),
 *         qh[32] (1 high bit per quant), qs[128] (4-bit low nibbles, 2 per byte).
 * 4 sub-blocks of 64, each with 2 scales+mins (for two groups of 32).
 * Result = d * sc * (qs_nibble + high_bit*16) - dm * mn. */
__device__ static inline float dequant_q5_K(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    uint16_t dm_raw = b[2] | ((uint16_t)b[3] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    float dm = gpu_fp16_to_fp32(dm_raw);
    const uint8_t *scales = b + 4;   /* 12 bytes packed */
    const uint8_t *qh = b + 16;     /* 32 bytes, 1 high bit per quant */
    const uint8_t *qs = b + 48;     /* 128 bytes, 2 nibbles per byte */

    /* i in 0..255. 4 groups of 64, each group has 2 sub-groups of 32.
     * group = i / 64 (0..3), pos = i % 64 (0..63).
     * Within group: sub = pos / 32 (0 or 1), l = pos % 32 (0..31).
     * scale_idx = group * 2 + sub (0..7 out of 12 packed entries, but only 8 used for 4*2).
     */
    int group = i / 64;
    int sub = (i % 64) / 32;
    int l = i % 32;
    int scale_idx = group * 2 + sub;

    uint8_t sc, mn;
    gpu_get_scale_min_k4(scale_idx, scales, &sc, &mn);

    /* qs layout: for each group of 64, 32 bytes hold the low nibbles.
     * sub=0 reads qs[group*32 + l] & 0xF, sub=1 reads qs[group*32 + l] >> 4.
     * qh layout: 32 bytes, one bit per quant. The bit position cycles.
     * For group g, sub s, position l:
     *   The CPU code shifts u1/u2 by 2 bits per group iteration.
     *   u1 starts at 1, u2 starts at 2, each multiplied by 4 per 64-element chunk.
     *   For the first 32 (sub=0): qh_bit = (qh[l] & u1) != 0
     *   For the next 32 (sub=1): qh_bit = (qh[l] & u2) != 0
     *   u1 = 1 << (2*group), u2 = 2 << (2*group)
     */
    int nibble = sub == 0 ? (qs[group * 32 + l] & 0xF) : (qs[group * 32 + l] >> 4);
    int u = (sub == 0) ? (1 << (2 * group)) : (2 << (2 * group));
    int high_bit = (qh[l] & u) ? 16 : 0;

    return d * (float)sc * (float)(nibble + high_bit) - dm * (float)mn;
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
                    gguf_type_t qtype, int S, int I, int O, int row_bytes, int x_stride, int y_stride) {
    /* bytes_per_block: stride between consecutive blocks in memory */
    int bytes_per_block;
    switch (qtype) {
        case GGUF_TYPE_F32:  bytes_per_block = 4; break;      /* 1 float */
        case GGUF_TYPE_F16:  bytes_per_block = 2; break;      /* 1 uint16_t */
        case 30:             bytes_per_block = 2; break;      /* BF16 */
        case GGUF_TYPE_Q4_0: bytes_per_block = GPU_BLOCK_Q4_0_SIZE; break;  /* 18 */
        case GGUF_TYPE_Q8_0: bytes_per_block = GPU_BLOCK_Q8_0_SIZE; break;  /* 34 */
        case 10:             bytes_per_block = GPU_BLOCK_Q2_K_SIZE; break;  /* Q2_K: 84 */
        case 11:             bytes_per_block = 110; break;                   /* Q3_K: 110 */
        case GGUF_TYPE_Q4_K: bytes_per_block = GPU_BLOCK_Q4_K_SIZE; break;  /* 144 */
        case 13:             bytes_per_block = GPU_BLOCK_Q5_K_SIZE; break;  /* Q5_K: 176 */
        case 14:             bytes_per_block = GPU_BLOCK_Q6_K_SIZE; break;  /* Q6_K: 210 */
        case 41:             bytes_per_block = 18; break;      /* Q1_0 */
        case 42:             bytes_per_block = 34; break;      /* Q2_0 */
        default: bytes_per_block = 18; break;
    }
    int o = gpuBlockIdx_x;
    int s = gpuBlockIdx_y;
    if (o >= O || s >= S) return;
    size_t rs = x_stride > 0 ? (size_t)x_stride : (size_t)I;
    int ys = y_stride > 0 ? y_stride : O;

    double sum = 0.0;
    const char *wrow = (const char *)weights + (size_t)o * row_bytes;

    switch (qtype) {
    case 0: /* GGUF_TYPE_F32 */
        for (int i = gpuThreadIdx_x; i < I; i += gpuBlockDim_x) {
            sum += x[rs*s+i] * ((const float *)(wrow))[i];
        }
        break;

    case 1: /* GGUF_TYPE_F16 */
        /* Raw FP16 array, no block structure. Dequant on-the-fly. */
        for (int i = gpuThreadIdx_x; i < I; i += gpuBlockDim_x) {
            sum += x[rs*s+i] * dequant_f16(wrow, i);
        }
        break;

    case 30: /* GGUF_TYPE_BF16 */
        /* Raw BF16 array, no block structure. Dequant on-the-fly. */
        for (int i = gpuThreadIdx_x; i < I; i += gpuBlockDim_x) {
            sum += x[rs*s+i] * dequant_bf16(wrow, i);
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
                    sum += x[rs*s+i] * dequant_q4_0(blk, j);
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
                    sum += x[rs*s+i] * dequant_q8_0(blk, j);
                }
            }
        }
        break;

    case 10: /* GGUF_TYPE_Q2_K */
        /* 256 values per block (84 bytes). Per-element dequant via helper. */
        {
            int n_blocks = I / 256;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 256; j++) {
                    int i = bi * 256 + j;
                    sum += x[rs*s+i] * dequant_q2_K(blk, j);
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
                    sum += x[rs*s+i] * dequant_q4_K(blk, j);
                }
            }
        }
        break;

    case 13: /* GGUF_TYPE_Q5_K */
        /* 256 values per block (176 bytes). Per-element dequant via helper. */
        {
            int n_blocks = I / 256;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 256; j++) {
                    int i = bi * 256 + j;
                    sum += x[rs*s+i] * dequant_q5_K(blk, j);
                }
            }
        }
        break;

    case 14: /* GGUF_TYPE_Q6_K */
        /* 256 values per block (210 bytes). Per-element dequant via helper. */
        {
            int n_blocks = I / 256;
            for (int bi = gpuThreadIdx_x; bi < n_blocks; bi += gpuBlockDim_x) {
                const void *blk = wrow + (size_t)bi * bytes_per_block;
                for (int j = 0; j < 256; j++) {
                    int i = bi * 256 + j;
                    sum += x[rs*s+i] * dequant_q6_K(blk, j);
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
                    sum += x[rs*s+i] * dequant_q1_0(blk, j);
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
                    sum += x[rs*s+i] * dequant_q2_0(blk, j);
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
        y[(size_t)s * ys + o] = (float)partial[0];
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

/* Strided variant: x_stride > 0 overrides the default s*I stride.
 * Output is always contiguous [S][I]. Only called when x_stride != I. */
__global__ void
picolm_quantize_q8_0_strided(int8_t *qs_out, float *d_out,
                              const float *x, int I, int S, int x_stride) {
    int s = gpuBlockIdx_y;
    int block = gpuBlockIdx_x;
    int tid = gpuThreadIdx_x;
    int n_blocks = (I + 31) / 32;
    if (block >= n_blocks || s >= S) return;
    const float *xb = x + (size_t)s * x_stride + (size_t)block * 32;
    __shared__ float amax_shared[32];
    float v = (tid < I - block * 32) ? xb[tid] : 0.0f;
    amax_shared[tid] = fabsf(v);
    gpuSyncthreads();
    for (int stride = 16; stride; stride >>= 1) {
        if (tid < stride)
            amax_shared[tid] = fmaxf(amax_shared[tid], amax_shared[tid + stride]);
        gpuSyncthreads();
    }
    float amax = amax_shared[0];
    float d = amax / 127.0f;
    float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    int q = (int)lrintf(v * id);
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
                     int S, int I, int O, int row_bytes, int y_stride) {
    int o = gpuBlockIdx_x;
    int s = gpuBlockIdx_y;
    if (o >= O || s >= S) return;
    int ys = y_stride > 0 ? y_stride : O;

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
        y[(size_t)s * ys + o] = (float)partial[0];
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
                           int S, int I, int O, int row_bytes, int y_stride) {
    int o = gpuBlockIdx_x;
    int tile = gpuBlockIdx_y;
    if (o >= O) return;
    int s0 = tile * Q8_TILE_S;
    if (s0 >= S) return;
    int s_count = min(Q8_TILE_S, S - s0);
    int ys = y_stride > 0 ? y_stride : O;

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

        /* Shared-memory tree reduce with double precision */
        partial[gpuThreadIdx_x] = sum;
        gpuSyncthreads();
        for (int n = gpuBlockDim_x >> 1; n; n >>= 1) {
            if (gpuThreadIdx_x < n)
                partial[gpuThreadIdx_x] += partial[gpuThreadIdx_x + n];
            gpuSyncthreads();
        }
        if (!gpuThreadIdx_x)
            y[(size_t)s * ys + o] = (float)partial[0];
        gpuSyncthreads();
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
__global__ void
picolm_gpu_attention_decode_split_kernel(
        float *partial_max, float *partial_sum, float *partial_acc,
        const float *q_dev,
        const uint16_t *kv_k, const uint16_t *kv_v,
        int layer_ordinal, int pos,
        int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
        size_t kv_pos_stride_bytes, size_t kv_head_stride_bytes,
        int n_splits, int chunk_size)
{
    int kv_h = (int)gpuBlockIdx_x;
    int split = (int)gpuBlockIdx_y;
    if (kv_h >= n_kv_heads || split >= n_splits) return;

    int kv_mul = n_heads / n_kv_heads;
    int first_qh = kv_h * kv_mul;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;
    int idx_base = (kv_h * n_splits + split) * kv_mul;

    int t0 = split * chunk_size;
    int t1 = min(t0 + chunk_size, pos + 1);

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

    if (t0 < t1) {
        size_t layer_base = (size_t)layer_ordinal * max_seq_len * n_kv_heads * head_dim;
        float sqrt_hd = sqrtf((float)head_dim);

        for (int t = t0; t < t1; t++) {
            size_t k_off = layer_base + (size_t)t * kv_pos_stride_bytes / 2 + kv_h * kv_head_stride_bytes / 2;
            for (int d = tid; d < head_dim; d += n_threads) {
                k_sh[d] = kv_k[k_off + d];
                v_sh[d] = kv_v[k_off + d];
            }
            gpuSyncthreads();

            for (int g = 0; g < kv_mul; g++) {
                const float *qg = q_dev + (size_t)(first_qh + g) * head_dim;
                float score = 0.0f;
                for (int d = tid; d < head_dim; d += n_threads)
                    score += qg[d] * gpu_fp16_to_fp32(k_sh[d]);
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
                for (int d = tid; d < head_dim; d += n_threads)
                    accg[d] = accg[d] * rescale_sh + weight_sh * gpu_fp16_to_fp32(v_sh[d]);
                gpuSyncthreads();
            }
        }
    }

    for (int g = tid; g < kv_mul; g += n_threads) {
        partial_max[idx_base + g] = max_score_sh[g];
        partial_sum[idx_base + g] = sum_exp_sh[g];
    }
    for (int i = tid; i < kv_mul * head_dim; i += n_threads)
        partial_acc[(size_t)idx_base * head_dim + i] = acc_sh[i];
}

/* Merges n_splits partial states per KV head via the standard online-
 * softmax merge rule, writes final normalized output. Grid = n_kv_heads. */
__global__ void
picolm_gpu_attention_decode_merge_kernel(
        float *xb_out,
        const float *partial_max, const float *partial_sum, const float *partial_acc,
        int n_heads, int n_kv_heads, int head_dim, int n_splits)
{
    int kv_h = (int)gpuBlockIdx_x;
    int kv_mul = n_heads / n_kv_heads;
    int first_qh = kv_h * kv_mul;
    int tid = gpuThreadIdx_x;
    int n_threads = gpuBlockDim_x;

    extern __shared__ float smem_f[];
    float *global_max_sh = smem_f;         /* [kv_mul] */
    float *global_sum_sh = global_max_sh + kv_mul; /* [kv_mul] */

    for (int g = tid; g < kv_mul; g += n_threads) {
        float m = -1e30f;
        for (int sp = 0; sp < n_splits; sp++) {
            int idx = (kv_h * n_splits + sp) * kv_mul + g;
            float v = partial_max[idx];
            if (v > m) m = v;
        }
        global_max_sh[g] = m;
    }
    gpuSyncthreads();

    for (int g = tid; g < kv_mul; g += n_threads) {
        float m = global_max_sh[g];
        float s = 0.0f;
        for (int sp = 0; sp < n_splits; sp++) {
            int idx = (kv_h * n_splits + sp) * kv_mul + g;
            s += partial_sum[idx] * expf(partial_max[idx] - m);
        }
        global_sum_sh[g] = s;
    }
    gpuSyncthreads();

    for (int g = 0; g < kv_mul; g++) {
        float m = global_max_sh[g];
        float inv_sum = 1.0f / global_sum_sh[g];
        float *xbhg = xb_out + (size_t)(first_qh + g) * head_dim;
        for (int d = tid; d < head_dim; d += n_threads) {
            float acc_d = 0.0f;
            for (int sp = 0; sp < n_splits; sp++) {
                int idx = (kv_h * n_splits + sp) * kv_mul + g;
                float w = expf(partial_max[idx] - m);
                acc_d += w * partial_acc[(size_t)idx * head_dim + d];
            }
            xbhg[d] = acc_d * inv_sum;
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
    /* Zero reduce_sh entries used for inter-warp partial sums (max 32 warps). */
    for (int i = tid; i < 32; i += n_threads) reduce_sh[i] = 0.0f;
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

            /* Initialize shared scalars for this token to prevent stale
             * values from the previous qi iteration affecting softmax. */
            if (tid == 0) { rescale_sh = 1.0f; weight_sh = 1.0f; }
            gpuSyncthreads();

            for (int ti = 0; ti < tile_k_size; ti++) {
                int global_kv = t0 + ti;
                if (global_kv > global_pos) continue;

                /* Compute score with thread parallelization */
                float score = 0.0f;
                for (int d = tid; d < head_dim; d += n_threads) {
                    score += qg[d] * gpu_fp16_to_fp32(k_tile[ti * head_dim + d]);
                }
                /* Reduce to thread 0 via shared memory tree reduction (matching decode kernel) */
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

    /* Q8_0-quantized copy of x, shared within the block (only thread 0
     * writes it, but declaring it __shared__ keeps it off each thread's
     * local/register budget). Matches quantize_row_q8_0's scalar path:
     * per 32-element block, scale = max(|x|)/127, round-to-nearest
     * (ties away from zero, via lroundf) clamped to [-127,127]/[-128
     * on the low side], delta round-tripped through fp16 exactly as
     * the CPU's block_q8_0.d field is. */
    __shared__ int8_t xq[PICOLM_SSM_VECDOT_MAX_DIM];
    __shared__ float xq_d[PICOLM_SSM_VECDOT_MAX_DIM / 32];

    if ((qtype == 2 || qtype == 8) && tid == 0) {
        int nb = dim / 32;
        for (int bi = 0; bi < nb; bi++) {
            float asmax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float v = x[bi * 32 + j];
                if (v < 0.0f) v = -v;
                if (v > asmax) asmax = v;
            }
            float d = asmax / 127.0f;
            xq_d[bi] = gpu_fp16_to_fp32(gpu_fp32_to_fp16(d));
            float id = (asmax != 0.0f) ? 127.0f / asmax : 0.0f;
            for (int j = 0; j < 32; j++) {
                int v = (int)lroundf(x[bi * 32 + j] * id);
                if (v > 127) v = 127;
                if (v < -127) v = -128;
                xq[bi * 32 + j] = (int8_t)v;
            }
        }
    }
    gpuSyncthreads();

    if (tid != 0) return;

    float sum = 0.0f;

    switch (qtype) {
    case 0: { /* F32 -- ssm_alpha/ssm_beta stay unquantized on some models
               * (Qwen3.6): these are small, precision-sensitive gating
               * projections. Matches vec_dot_f32_f32's scalar
               * fallback: plain left-to-right float accumulation. */
        const float *wrow_f = (const float *)wrow;
        for (int i = 0; i < dim; i++) sum += wrow_f[i] * x[i];
        break;
    }
    case 2: { /* Q4_0 weight x Q8_0(x) -- matches vec_dot_q4_0_q8_0 scalar tail */
        int n_blocks = dim / 32;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 18;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);
            const uint8_t *qs = blk + 2;
            const int8_t *yq = xq + bi * 32;
            int sumi0 = 0, sumi1 = 0;
            for (int j = 0; j < 16; j++) {
                int v0 = (qs[j] & 0x0F) - 8;
                int v1 = (qs[j] >> 4) - 8;
                sumi0 += v0 * yq[j];
                sumi1 += v1 * yq[j + 16];
            }
            sum += (float)(sumi0 + sumi1) * wd * xq_d[bi];
        }
        break;
    }
    case 8: { /* Q8_0 weight x Q8_0(x) -- matches vec_dot_q8_0_q8_0_deltas scalar tail */
        int n_blocks = dim / 32;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 34;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);
            const int8_t *qs = (const int8_t *)(blk + 2);
            const int8_t *yq = xq + bi * 32;
            int sumi = 0;
            for (int j = 0; j < 32; j++) sumi += (int)yq[j] * (int)qs[j];
            sum += (float)sumi * xq_d[bi] * wd;
        }
        break;
    }
    default:
        break;
    }
    out[h] = sum;
}

/* Batched-over-tokens version of the above: identical per-(token,head)
 * computation (including the same redundant per-block re-quantization
 * of x into Q8_0 -- kept for simplicity/correctness parity with the
 * per-token kernel; x is small enough that this isn't the bottleneck).
 * x is [n_tokens][dim], out is [n_tokens][n_v_heads]. */
__global__ void
picolm_ssm_vecdot_batch_kernel(float *out,
                                const float *x,
                                const void *weights,
                                gguf_type_t qtype,
                                int dim, int n_v_heads, int n_tokens,
                                int row_bytes,
                                const int *head_map, int out_stride) {
    int h = gpuBlockIdx_x;
    int t = gpuBlockIdx_y;
    if (h >= n_v_heads || t >= n_tokens) return;
    int tid = gpuThreadIdx_x;
    int gh = head_map ? head_map[h] : h;
    const char *wrow = (const char *)weights + (size_t)gh * row_bytes;
    const float *xt = x + (size_t)t * dim;

    __shared__ int8_t xq[PICOLM_SSM_VECDOT_MAX_DIM];
    __shared__ float xq_d[PICOLM_SSM_VECDOT_MAX_DIM / 32];

    if ((qtype == 2 || qtype == 8) && tid == 0) {
        int nb = dim / 32;
        for (int bi = 0; bi < nb; bi++) {
            float asmax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float v = xt[bi * 32 + j];
                if (v < 0.0f) v = -v;
                if (v > asmax) asmax = v;
            }
            float d = asmax / 127.0f;
            xq_d[bi] = gpu_fp16_to_fp32(gpu_fp32_to_fp16(d));
            float id = (asmax != 0.0f) ? 127.0f / asmax : 0.0f;
            for (int j = 0; j < 32; j++) {
                int v = (int)lroundf(xt[bi * 32 + j] * id);
                if (v > 127) v = 127;
                if (v < -127) v = -128;
                xq[bi * 32 + j] = (int8_t)v;
            }
        }
    }
    gpuSyncthreads();

    if (tid != 0) return;

    float sum = 0.0f;

    switch (qtype) {
    case 0: {
        const float *wrow_f = (const float *)wrow;
        for (int i = 0; i < dim; i++) sum += wrow_f[i] * xt[i];
        break;
    }
    case 2: {
        int n_blocks = dim / 32;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 18;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);
            const uint8_t *qs = blk + 2;
            const int8_t *yq = xq + bi * 32;
            int sumi0 = 0, sumi1 = 0;
            for (int j = 0; j < 16; j++) {
                int v0 = (qs[j] & 0x0F) - 8;
                int v1 = (qs[j] >> 4) - 8;
                sumi0 += v0 * yq[j];
                sumi1 += v1 * yq[j + 16];
            }
            sum += (float)(sumi0 + sumi1) * wd * xq_d[bi];
        }
        break;
    }
    case 8: {
        int n_blocks = dim / 32;
        for (int bi = 0; bi < n_blocks; bi++) {
            const uint8_t *blk = (const uint8_t *)wrow + (size_t)bi * 34;
            uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
            float wd = gpu_fp16_to_fp32(d_raw);
            const int8_t *qs = (const int8_t *)(blk + 2);
            const int8_t *yq = xq + bi * 32;
            int sumi = 0;
            for (int j = 0; j < 32; j++) sumi += (int)yq[j] * (int)qs[j];
            sum += (float)sumi * xq_d[bi] * wd;
        }
        break;
    }
    case 30: {
        /* BF16: raw bf16 array, no block structure.
         * Use dequant_bf16 (portable zero-extend), not __bfloat162float (broken). */
        const uint16_t *wrow_bf = (const uint16_t *)wrow;
        for (int i = 0; i < dim; i++) sum += dequant_bf16(wrow, i) * xt[i];
        break;
    }
    default:
        break;
    }
    int _os = out_stride > 0 ? out_stride : n_v_heads;
    out[(size_t)t * _os + h] = sum;
}

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
__global__ void
picolm_gpu_ssm_conv1d_kernel(float *conv_output, float *conv_state,
                              const float *new_input, const float *conv1d_w,
                              int conv_dim, int d_conv) {
    int co = (int)gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (co >= conv_dim) return;

    int n_state_rows = d_conv - 1;
    float sum = 0.0f;
    for (int d = 0; d < n_state_rows; d++)
        sum += conv1d_w[d + co * d_conv] * conv_state[d * conv_dim + co];
    sum += conv1d_w[(d_conv - 1) + co * d_conv] * new_input[co];
    conv_output[co] = sum / (1.0f + expf(-sum)); /* silu */

    /* Shift history left, append newest -- safe to do after computing
     * the output above since each thread only ever touches its own
     * channel's column, no cross-thread dependency. */
    for (int r = 0; r < n_state_rows - 1; r++)
        conv_state[r * conv_dim + co] = conv_state[(r + 1) * conv_dim + co];
    if (n_state_rows > 0)
        conv_state[(n_state_rows - 1) * conv_dim + co] = new_input[co];
}

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

__global__ void
picolm_gpu_ssm_conv1d_batch_kernel(float *conv_output,     /* out [n_tokens][stride] */
                                    float *conv_state,      /* in/out [d_conv-1][conv_dim], persistent */
                                    const float *new_input, /* in [n_tokens][stride] */
                                    const float *conv1d_w,  /* in [conv_dim][d_conv] */
                                    int conv_dim, int d_conv, int n_tokens, int stride) {
    int co = (int)gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (co >= conv_dim) return;

    int n_state_rows = d_conv - 1;

    float hist[PICOLM_SSM_CONV_MAX_D_CONV];
    for (int r = 0; r < n_state_rows; r++)
        hist[r] = conv_state[r * conv_dim + co];

    float w[PICOLM_SSM_CONV_MAX_D_CONV];
    for (int d = 0; d < d_conv; d++)
        w[d] = conv1d_w[d + co * d_conv];

    for (int t = 0; t < n_tokens; t++) {
        float new_sample = new_input[(size_t)t * stride + co];
        float sum = 0.0f;
        for (int d = 0; d < n_state_rows; d++)
            sum += w[d] * hist[d];
        sum += w[n_state_rows] * new_sample;
        conv_output[(size_t)t * stride + co] = sum / (1.0f + expf(-sum)); /* silu */

        for (int r = 0; r < n_state_rows - 1; r++)
            hist[r] = hist[r + 1];
        if (n_state_rows > 0)
            hist[n_state_rows - 1] = new_sample;
    }

    for (int r = 0; r < n_state_rows; r++)
        conv_state[r * conv_dim + co] = hist[r];
}

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
    constexpr int GROUP = 4;                         /* matches CPU's 4 strided accumulators */
    constexpr int GROUPS_PER_WARP = GPU_WARP_SIZE / GROUP;
    constexpr int d4 = S_v / GROUP;                  /* S_v always 16/32/64/128, multiple of 4 */

    const int head = gpuBlockIdx_x;
    const int lane = gpuThreadIdx_x;                 /* 0..GPU_WARP_SIZE-1 */
    const int group_in_warp = lane / GROUP;          /* which row within this warp */
    const int sub = lane % GROUP;                    /* 0..3, role of CPU's s0..s3 */
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

    /* Pass 1 (matches CPU's decay+sk loop): decay each of this sub-lane's
     * strided columns and fmaf-accumulate the dot product with k, in
     * ascending vv order, exactly like the CPU's s{sub} accumulator. */
    float s_acc = 0.0f;
    for (int vv = 0; vv < d4; vv++) {
        int c = vv * GROUP + sub;
        float r = st[c] * ge;
        st[c] = r;
        s_acc = fmaf(r, khv[c], s_acc);
    }
    /* (s0+s1)+(s2+s3) via two width-4 XOR shuffles -- see note above. */
    float t1 = s_acc + gpu_shfl_xor_sync(s_acc, 1, GROUP);
    float sk_row = t1 + gpu_shfl_xor_sync(t1, 2, GROUP);

    const float dv = (v0 - sk_row) * bh;

    /* Pass 2 (matches CPU's rank-1 update + output loop): update this
     * sub-lane's strided columns with the rank-1 term and fmaf-accumulate
     * the output dot product with q, same order as the CPU loop. */
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
    if (gpuThreadIdx_x != 0) return;

    /* Thread-0 sequential, matching CPU NEON accumulation order.
     * The DGX Spark runs ARM64 with NEON. The CPU ssm_head_task uses
     * 4-lane strided fmaf accumulators combined by pairwise reduction
     * (vaddvq_f32: (a0+a1)+(a2+a3)). This is a different summation
     * tree from strict left-to-right addition, producing ~1e-7 relative
     * difference that compounds over 48+ SSM layers.
     *
     * Replicate NEON's exact order: 4 independent fmaf() strided
     * accumulators, combined with pairwise (lo+hi) reduction.
     * Decay is applied per-element BEFORE the sk dot product,
     * matching NEON which decays each loaded element before fma. */
    int kh = h / repeat;
    const float *qh = q_conv + (size_t)kh * d_state;
    const float *khv = k_conv + (size_t)kh * d_state;
    const float *vh = v_conv + (size_t)h * d_state;
    float ge = gate_exp[h];
    float bh = beta[h];
    float *st = state + (size_t)h * d_state * d_state;

    int d4 = d_state / 4; /* d_state always 16/32/64/128 */
    float sk_local[256];
    float d_local[256];

    /* Decay + sk: 4-lane strided fmaf per row, matching NEON */
    for (int row = 0; row < d_state; row++) {
        float *sr = st + (size_t)row * d_state;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int vv = 0; vv < d4; vv++) {
            int cb = vv * 4;
            float r0 = sr[cb+0] * ge; sr[cb+0] = r0;
            float r1 = sr[cb+1] * ge; sr[cb+1] = r1;
            float r2 = sr[cb+2] * ge; sr[cb+2] = r2;
            float r3 = sr[cb+3] * ge; sr[cb+3] = r3;
            s0 = fmaf(r0, khv[cb+0], s0);
            s1 = fmaf(r1, khv[cb+1], s1);
            s2 = fmaf(r2, khv[cb+2], s2);
            s3 = fmaf(r3, khv[cb+3], s3);
        }
        sk_local[row] = (s0 + s1) + (s2 + s3);
        d_local[row] = (vh[row] - sk_local[row]) * bh;
    }

    /* Rank-1 update + output: 4-lane strided fmaf per row */
    for (int row = 0; row < d_state; row++) {
        float dv = d_local[row];
        float *sr = st + (size_t)row * d_state;
        float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
        for (int vv = 0; vv < d4; vv++) {
            int cb = vv * 4;
            float r0 = fmaf(khv[cb+0], dv, sr[cb+0]); sr[cb+0] = r0;
            float r1 = fmaf(khv[cb+1], dv, sr[cb+1]); sr[cb+1] = r1;
            float r2 = fmaf(khv[cb+2], dv, sr[cb+2]); sr[cb+2] = r2;
            float r3 = fmaf(khv[cb+3], dv, sr[cb+3]); sr[cb+3] = r3;
            o0 = fmaf(r0, qh[cb+0], o0);
            o1 = fmaf(r1, qh[cb+1], o1);
            o2 = fmaf(r2, qh[cb+2], o2);
            o3 = fmaf(r3, qh[cb+3], o3);
        }
        ssm_output[(size_t)row * n_v_heads + h] = (o0 + o1) + (o2 + o3);
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
    /* Split-K decode attention scratch: [n_heads*n_splits] max + sum
     * floats, then [n_heads*n_splits*head_dim] acc floats, grown via
     * reserve() on first use (size depends on n_splits, which varies
     * with context length, capped at ATTN_DECODE_MAX_SPLITS). */
    float *attn_partial;
    size_t attn_partial_cap;
    /* Prefill batch buffers: [max_seq_len][dim] for S>1 prefill pipeline.
     * Same set of buffers as decode but sized for n_tokens rows. */
    float *pipe_x_b, *pipe_xb_b, *pipe_q_b, *pipe_k_b, *pipe_v_b,
          *pipe_attn_out_b, *pipe_ffn_norm_b, *pipe_gate_b, *pipe_up_b;
    int pipe_b_ready;
    /* SSM pipeline buffers (decode, S=1). Allocated once per model if
     * model has SSM layers. Sized from model's SSM config (conv_dim,
     * ssm_d_inner, dt_rank). These are the working buffers used by the
     * hybrid SSM+attention branch in model_forward_gpu(). */
    float *ssm_qkv_raw;      /* [conv_dim] attn_qkv output, conv1d input */
    float *ssm_conv_out;     /* [conv_dim] conv1d output, split into q/k/v */
    float *ssm_xb2;          /* [ssm_d_inner] attn_gate_ssm output (z-gate) */
    float *ssm_xb2_remap;    /* [ssm_d_inner] remapped xb2 (if do_remap) */
    float *ssm_v_remap;      /* [ssm_d_inner] remapped v_conv (if do_remap) */
    float *ssm_alpha_raw;    /* [n_v_heads] alpha vecdot output */
    float *ssm_beta_raw;     /* [n_v_heads] beta vecdot output */
    float *ssm_gate_exp;     /* [n_v_heads] gate exp output */
    float *ssm_beta;         /* [n_v_heads] beta sigmoid output */
    float *ssm_output;       /* [ssm_d_inner] recurrence output */
    float *ssm_final_output; /* [ssm_d_inner] gated_norm output */
    int ssm_ready;           /* 1 once SSM buffers are allocated */
    /* Prefill SSM scratch buffer: grow-only, used by ssm_chunked_recurrence_dev */
    float *ssm_prefill_scratch;
    size_t ssm_prefill_scratch_cap;
    /* RMSNorm weight cache: maps host weight pointers to device copies */
    void *rmsnorm_w_dev[64];
    const void *rmsnorm_w_keys[64];
    int rmsnorm_w_n;
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

/* Grow-only device buffer helper, mirroring model.c's ssm_mm_alloc /
 * the existing g_ssm_mm_*_dev pattern used for the decode-path GPU
 * matmul scratch buffers. The batched SSM prefill wrappers below
 * (conv1d_batch, l2norm_batch, vecdot_batch, prefill_gated_norm,
 * chunked_recurrence) originally did a fresh gpuMalloc/gpuFree on
 * EVERY call -- since ssm_prefill_layer runs once per layer, that's
 * 30-40 cudaMalloc/cudaFree pairs per layer, ~1500-2000 per prompt for
 * a typical model. Profiling showed cudaMalloc alone costing multiple
 * seconds (individual calls up to several hundred ms, consistent with
 * allocator fragmentation from repeated alloc/free of varying sizes)
 * and cudaMemcpy costing tens of seconds (individual calls blocking
 * for seconds, consistent with waiting behind queued work rather than
 * pure PCIe transfer time) -- exactly what shows up in `perf` as
 * unresolved 0x... addresses inside the driver. Buffers allocated via
 * this helper are grown as needed and never freed until the device
 * changes, eliminating that overhead from the hot path entirely. */
static int ssm_batch_scratch_ensure(void **buf, size_t *cap, size_t need) {
    if (*buf && *cap >= need) return 1;
    if (*buf) { gpuDeviceSynchronize(); gpuFree(*buf); *buf = NULL; *cap = 0; }
    if (!gpu_ok(gpuMalloc(buf, need), "ssm batch scratch grow")) { *cap = 0; return 0; }
    *cap = need;
    return 1;
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    gpu_mutex_lock();
    if (*cap >= bytes) { gpu_mutex_unlock(); return 1; }
    if (*ptr) { gpuDeviceSynchronize(); gpuFree(*ptr); }
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
    /* Zero-init: GB10 cudaMalloc returns stale VRAM contents causing
     * non-determinism in kernels that read before writing all elements. */
    if (*ptr) gpuMemset(*ptr, 0, bytes);
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
    /* MUST synchronize with GPU before freeing, otherwise a kernel
     * launched on ctx->stream may still be reading from the old buffer
     * when gpuFree releases it. This causes non-deterministic memory
     * corruption that varies across runs. */
    if (*ptr) { gpuDeviceSynchronize(); gpuFree(*ptr); }
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
        /* Free cached RMSNorm weights */
        for (int j = 0; j < ctx->rmsnorm_w_n; j++) {
            if (ctx->rmsnorm_w_dev[j]) gpuFree(ctx->rmsnorm_w_dev[j]);
        }
        ctx->rmsnorm_w_n = 0;
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
    case 10: return 84;    /* Q2_K: 84 bytes per 256 values */
    case 11: return 110;   /* Q3_K: 110 bytes per 256 values */
    case 12: return GPU_BLOCK_Q4_K_SIZE;  /* Q4_K: 144 bytes per 256 values */
    case 13: return GPU_BLOCK_Q5_K_SIZE;  /* Q5_K: 176 bytes per 256 values */
    case 14: return GPU_BLOCK_Q6_K_SIZE;  /* Q6_K: 210 bytes per 256 values */
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
    } else if (qtype == 10 || qtype == 11 || qtype == 12 || qtype == 13 || qtype == 14) {
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

    /* Q2_K/Q3_K/Q4_K/Q5_K/Q6_K fast path: convert to Q8_0 at upload time for the int8-MAC kernel.
     * These have per-sub-block scales that the Q8_0 kernel can't handle,
     * but requantizing to Q8_0 (1 scale per 32 elements) is accurate enough
     * and enables the highly-optimized picolm_q8_q8_matmul path. */
    if (qtype == 10 || qtype == 11 || qtype == 12 || qtype == 13 || qtype == 14) {
        /* Dequant Q2_K/Q3_K/Q4_K/Q5_K/Q6_K to F32, then requant to Q8_0 */
        size_t f32_bytes = (size_t)I * (size_t)O * sizeof(float);
        float *f32_buf = (float *)calloc(I * O, sizeof(float));
        if (!f32_buf) { gpuFree(t->weights); free(t); return 0; }

        /* Dequant each row */
        int nb = I / 256;
        int blk_bytes;
        if (qtype == 10) blk_bytes = 84;
        else if (qtype == 11) blk_bytes = 110;
        else if (qtype == 12) blk_bytes = 144;
        else if (qtype == 13) blk_bytes = 176;
        else blk_bytes = 210; /* qtype == 14 */
        for (int row = 0; row < O; row++) {
            float *dst = f32_buf + row * I;
            const uint8_t *row_start = (const uint8_t *)weights + row * nb * blk_bytes;
            if (qtype == 10)
                dequantize_row_q2_K(row_start, dst, I);
            else if (qtype == 11)
                dequantize_row_q3_K(row_start, dst, I);
            else if (qtype == 12)
                dequantize_row_q4_K(row_start, dst, I);
            else if (qtype == 13)
                dequantize_row_q5_K(row_start, dst, I);
            else
                dequantize_row_q6_K(row_start, dst, I);
        }

        /* Requant to Q8_0 */
        size_t q8_total = (size_t)O * ((I + 31) / 32) * 34;  /* 34 bytes per Q8_0 block */
        uint8_t *q8_buf = (uint8_t *)calloc(q8_total, 1);
        if (!q8_buf) { free(f32_buf); gpuFree(t->weights); free(t); return 0; }

        for (int row = 0; row < O; row++) {
            const float *src = f32_buf + row * I;
            block_q8_0 *q8_blocks = (block_q8_0 *)(q8_buf + row * ((I + 31) / 32) * 34);
            quantize_row_q8_0(src, q8_blocks, I);
        }

        free(f32_buf);

        /* Upload Q8_0 bytes, record as Q8_0 type */
        if (!gpu_ok(gpuMalloc(&t->weights, q8_total), "tensor allocation (q8)") ||
            !gpu_ok(gpuMemcpy(t->weights, q8_buf, q8_total, gpuMemcpyHostToDevice),
                    "tensor upload (q8)")) {
            free(q8_buf); gpuFree(t->weights); free(t); return 0;
        }
        free(q8_buf);

        t->qtype = (gguf_type_t)8;  /* Now Q8_0 for the fast path */
        t->block_size = 34;
        t->row_bytes = ((I + 31) / 32) * 34;
        t->zero_copy = 0;
        t->tracked = 1;
        ctx->tensor_count++;
        ctx->tensor_bytes += q8_total;
        {
            static int first_print = 1;
            if (first_print) {
                fprintf(stderr, "[GPU] upload mode: q2/q3/q4/q5/q6->q8 (int8-MAC fast path)\n");
                first_print = 0;
            }
        }
        *tp = t;
        return 1;
    }

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

extern "C" void picolm_gpu_debug_tensor(const char *name, void *tp, int device, int layer, int dump_weights) {
    picolm_gpu_tensor_t *t = (picolm_gpu_tensor_t *)tp;
    if (!t) return;
    fprintf(stderr, "[DBG l%d] %s tensor: qtype=%d I=%d O=%d row_bytes=%zu\n",
        layer, name, (int)t->qtype, t->I, t->O, t->row_bytes);
    if (dump_weights && t->weights) {
        /* Dequantize first 64 floats */
        float wbuf[64] = {0};
        if (t->qtype == 0) {
            /* F32: D2H direct */
            cudaSetDevice(device);
            cudaMemcpy(wbuf, t->weights, 64 * sizeof(float), cudaMemcpyDeviceToHost);
        } else if (t->qtype == 2) {
            /* Q8_0: dump raw blocks then dequantize */
            unsigned char raw[66]; /* 2 blocks: 32+1 + 32+1 = 66 */
            cudaSetDevice(device);
            cudaMemcpy(raw, t->weights, 66, cudaMemcpyDeviceToHost);
            float sc0 = *(const float *)(raw + 32);
            for (int i = 0; i < 32; i++) wbuf[i] = (raw[i] - 128) * sc0;
            float sc1 = *(const float *)(raw + 64);
            for (int i = 0; i < 32; i++) wbuf[32 + i] = (raw[33 + i] - 128) * sc1;
        } else if (t->qtype == 8) {
            /* BF16: D2H and convert */
            unsigned short b16[64];
            cudaSetDevice(device);
            cudaMemcpy(b16, t->weights, 64 * 2, cudaMemcpyDeviceToHost);
            for (int i = 0; i < 64; i++) {
                unsigned int bits = (unsigned int)b16[i] << 16;
                float f; memcpy(&f, &bits, 4);
                wbuf[i] = f;
            }
        }
        fprintf(stderr, "[DBG l%d] %s_w[0][:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f}\n",
            layer, name, wbuf[0],wbuf[1],wbuf[2],wbuf[3],wbuf[4],wbuf[5],wbuf[6],wbuf[7]);
        double wr = 0; for (int i = 0; i < 64; i++) wr += wbuf[i] * wbuf[i];
        fprintf(stderr, "[DBG l%d] %s_w rms64=%.6f\n", layer, name, sqrt(wr / 64));
    }
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
const void *picolm_gpu_tensor_weights(const picolm_gpu_tensor_t *t) {
    return t ? t->weights : NULL;
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

/* Generic device memory allocation. Returns NULL on failure. */
void *picolm_gpu_alloc_device(size_t bytes, int device) {
    if (bytes < 1) return NULL;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return NULL;
    void *ptr = NULL;
    if (!gpu_ok(gpuMalloc(&ptr, bytes), "device alloc")) return NULL;
    return ptr;
}

/* Device memory set to value. Uses gpuMemset (zero-fill). */
int picolm_gpu_device_memset(void *dev_ptr, int value, size_t bytes, int device) {
    if (!dev_ptr || bytes < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    return gpu_ok(gpuMemset(dev_ptr, value, bytes), "device memset");
}

/* Upload a host int32 array to device. Returns device pointer or NULL. */
void *picolm_gpu_upload_int(const int *host, size_t n, int device) {
    if (!host || n < 1) return NULL;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return NULL;
    void *dev = NULL;
    size_t bytes = n * sizeof(int);
    if (!gpu_ok(gpuMalloc(&dev, bytes), "int vector allocation")) return NULL;
    if (!gpu_ok(gpuMemcpy(dev, host, bytes, gpuMemcpyHostToDevice), "int vector upload")) {
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

        if (S > 1 && t->row_bytes + 2048 <= 49152) {
            dim3 grid((unsigned)O, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
            picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)t->row_bytes, ctx->stream>>>(
                ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
        } else {
            dim3 grid((unsigned)O, (unsigned)S);
            picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
                ctx->y, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, O);
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
                                        (int)t->row_bytes, 0, 0);
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
                       int S, int device, int y_stride, int x_stride) {
    if (!t || !y_dev || !x_dev || S < 1) return 0;
    if (t->I < 512 || t->O < 256) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    int I = t->I, O = t->O;
    int ys = y_stride > 0 ? y_stride : O;

    if (t->qtype == GGUF_TYPE_Q8_0 && !getenv("PICOLM_FORCE_F32_MATMUL")) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0;
        size_t xq_bytes = (size_t)S * I;
        size_t xd_bytes = (size_t)S * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;
        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);
        dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        if (x_stride > 0) {
            picolm_quantize_q8_0_strided<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
                ctx->q8_xq, ctx->q8_xd, x_dev, I, S, x_stride);
        } else {
            picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
                ctx->q8_xq, ctx->q8_xd, x_dev, I, S);
        }
        if (!gpu_ok(gpuGetLastError(), "q8 quantize (dev)")) return 0;
        gpuDeviceSynchronize();
        dim3 grid((unsigned)O, (unsigned)S);
        picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q8 matmul (dev)")) return 0;
        return 1;
    }
    if (t->qtype == GGUF_TYPE_Q8_0) {
        /* Fallback: use per-thread F32 dequant for Q8_0 */
        dim3 grid((unsigned)O, (unsigned)S);
        picolm_quant_matmul<<<grid, 256, 0, ctx->stream>>>(y_dev, x_dev, t->weights,
                                            t->qtype, S, I, O,
                                            (int)t->row_bytes, x_stride, ys);
        if (!gpu_ok(gpuGetLastError(), "q8 matmul f32 fallback (dev)")) return 0;
        return 1;
    }
    /* Non-Q8_0 types: use per-thread dequant + F32 accumulation */
    dim3 grid((unsigned)O, (unsigned)S);
    picolm_quant_matmul<<<grid, 256, 0, ctx->stream>>>(y_dev, x_dev, t->weights,
                                        t->qtype, S, I, O,
                                        (int)t->row_bytes, x_stride, ys);
    if (!gpu_ok(gpuGetLastError(), "matmul launch (dev)")) return 0;
    return 1;
}

/* Strided variant: x_stride > 0 overrides default I stride.
 * Only called for SSM output projection where pipe buffer stride != value_dim. */
extern "C" int
picolm_gpu_matmul_dev_strided(picolm_gpu_tensor_t *t, float *y_dev,
                               const float *x_dev, int S, int device, int x_stride, int y_stride) {
    if (!t || !y_dev || !x_dev || S < 1 || x_stride <= 0) return 0;
    if (t->I < 512 || t->O < 256) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;
    int I = t->I, O = t->O;
    int ys = y_stride > 0 ? y_stride : O;
    if (t->qtype == GGUF_TYPE_Q8_0) {
        int n_blocks = I / 32;
        if (n_blocks < 1 || I % 32 != 0) return 0;
        size_t xq_bytes = (size_t)S * I;
        size_t xd_bytes = (size_t)S * n_blocks * sizeof(float);
        if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
            !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) return 0;
        gpuMemsetAsync(ctx->q8_xq, 0, xq_bytes, ctx->stream);
        gpuMemsetAsync(ctx->q8_xd, 0, xd_bytes, ctx->stream);
        dim3 q_grid((unsigned)n_blocks, (unsigned)S);
        picolm_quantize_q8_0_strided<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, x_dev, I, S, x_stride);
        if (!gpu_ok(gpuGetLastError(), "q8 quantize strided (dev)")) return 0;
        gpuDeviceSynchronize();
        dim3 grid((unsigned)O, (unsigned)S);
        picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
            y_dev, ctx->q8_xq, ctx->q8_xd, t->weights, S, I, O, (int)t->row_bytes, ys);
        if (!gpu_ok(gpuGetLastError(), "q8 matmul strided (dev)")) return 0;
        return 1;
    }
    dim3 grid((unsigned)O, (unsigned)S);
    picolm_quant_matmul<<<grid, 256, 0, ctx->stream>>>(y_dev, x_dev, t->weights,
                                        t->qtype, S, I, O, (int)t->row_bytes, x_stride, ys);
    if (!gpu_ok(gpuGetLastError(), "matmul strided (dev)")) return 0;
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

    /* ---- Q8_0 fast path: quantize F32 input to Q8_0, then int8 MAC ----
     * All 3 projections (gate, up, down) use Q8_0 weights. The input to
     * gate+up is the same F32 vector, so we quantize it once and reuse
     * the Q8_0 copy for both. The input to down is silu(gate)*up (F32),
     * which we quantize again before the down projection.
     * This replaces the slow per-block picolm_quant_matmul kernel with
     * the fast picolm_q8_q8_matmul_tiled kernel that tiles the Q8_0
     * weight row in shared memory across 32 query positions. */
    int d_blocks = D / 32;
    int i_blocks = I / 32;
    if (d_blocks < 1 || i_blocks < 1) return 0; /* must be aligned */

    size_t xq_d_bytes = (size_t)S * D;
    size_t xd_d_bytes = (size_t)S * d_blocks * sizeof(float);
    size_t xq_i_bytes = (size_t)S * I;
    size_t xd_i_bytes = (size_t)S * i_blocks * sizeof(float);

    if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_d_bytes > xq_i_bytes ? xq_d_bytes : xq_i_bytes) ||
        !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_d_bytes > xd_i_bytes ? xd_d_bytes : xd_i_bytes)) return 0;

    /* Step 1: Quantize F32 input [D*S] to Q8_0 */
    dim3 q_grid_d((unsigned)d_blocks, (unsigned)S);
    picolm_quantize_q8_0<<<q_grid_d, 32, 32 * sizeof(float), ctx->stream>>>(
        ctx->q8_xq, ctx->q8_xd, ctx->x, D, S);
    if (!gpu_ok(gpuGetLastError(), "expert q8 quantize input")) return 0;

    /* Step 2: gate = q8_q8_tiled(input_q8, gate_weights) -> F32[I*S] */
    int use_tiled = (S > 1 && gate->row_bytes + 2048 <= 49152);
    if (use_tiled) {
        dim3 grid((unsigned)I, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)gate->row_bytes, ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, gate->weights, S, D, I, (int)gate->row_bytes, I);
    } else {
        dim3 grid((unsigned)I, (unsigned)S);
        picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, gate->weights, S, D, I, (int)gate->row_bytes, I);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 gate")) return 0;

    /* Step 3: up = q8_q8_tiled(input_q8, up_weights) -> F32[I*S] */
    if (use_tiled) {
        dim3 grid((unsigned)I, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)up->row_bytes, ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, up->weights, S, D, I, (int)up->row_bytes, I);
    } else {
        dim3 grid((unsigned)I, (unsigned)S);
        picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, up->weights, S, D, I, (int)up->row_bytes, I);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 up")) return 0;

    /* Step 4: silu(gate) * up -> ctx->gate (in-place, F32[I*S]) */
    size_t n = (size_t)S * I;
    picolm_silu_mul<<<(unsigned)((n + 255) / 256), 256, 0, ctx->stream>>>(ctx->gate, ctx->up, n);
    if (!gpu_ok(gpuGetLastError(), "expert silu")) return 0;

    /* Step 5: Quantize F32 silu*up [I*S] to Q8_0 for down projection */
    dim3 q_grid_i((unsigned)i_blocks, (unsigned)S);
    picolm_quantize_q8_0<<<q_grid_i, 32, 32 * sizeof(float), ctx->stream>>>(
        ctx->q8_xq, ctx->q8_xd, ctx->gate, I, S);
    if (!gpu_ok(gpuGetLastError(), "expert q8 quantize hidden")) return 0;

    /* Step 6: y = q8_q8_tiled(hidden_q8, down_weights) -> F32[D*S] */
    int use_tiled_down = (S > 1 && down->row_bytes + 2048 <= 49152);
    if (use_tiled_down) {
        dim3 grid((unsigned)D, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)down->row_bytes, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, down->weights, S, I, D, (int)down->row_bytes, D);
    } else {
        dim3 grid((unsigned)D, (unsigned)S);
        picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
            ctx->y, ctx->q8_xq, ctx->q8_xd, down->weights, S, I, D, (int)down->row_bytes, D);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 down")) return 0;

    if (!gpu_ok(gpuDeviceSynchronize(), "expert MLP sync")) return 0;
    return gpu_ok(gpuMemcpy(y, ctx->y, xb, gpuMemcpyDeviceToHost), "expert output");
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

    /* Warp-shuffle kernel for common d_state values. Falls back to
     * thread-0 kernel for unsupported sizes.
     * Gated behind PICOLM_SSM_WARP_KERNEL_VALIDATED: the rewritten
     * order-matched warp kernel (see comment above
     * picolm_ssm_recurrence_warp_kernel) has not been validated on
     * real hardware against ssm_recurrence_verify.c in this session.
     * Until that validation is done and this macro is defined by the
     * build, always use the thread-0 kernel, which IS bit-exact with
     * CPU NEON (previously verified -- see ssm_gpu_session_findings.md). */
    int warp_launched = 0;
#ifdef PICOLM_SSM_WARP_KERNEL_VALIDATED
    if (d_state == 128) {
        constexpr int S_v = 128;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ds, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 64) {
        constexpr int S_v = 64;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ds, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 32) {
        constexpr int S_v = 32;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ds, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 16) {
        constexpr int S_v = 16;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ds, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    }
#endif /* PICOLM_SSM_WARP_KERNEL_VALIDATED */
    if (!warp_launched) {
        dim3 grid((unsigned)n_v_heads, 1, 1);
        picolm_ssm_recurrence_kernel<<<grid, 256, 0, ctx->stream>>>(
            (float *)ds, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, d_state, repeat);
    }

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

/* Device-native SSM recurrence: takes device-resident state, no per-call
 * malloc/H2D/D2H for state. q/k/v/gate_exp/beta/ssm_output still CPU-side,
 * uploaded/downloaded per call. The key saving is eliminating the persistent
 * state H2D/D2H round-trip per token. */
extern "C" int
picolm_gpu_ssm_recurrence_dev(void *ssm_state_dev,  /* in/out, device [n_v_heads][d_state][d_state] */
                               const float *q_conv,  /* host [n_k_heads][d_state] */
                               const float *k_conv,  /* host [n_k_heads][d_state] */
                               const float *v_conv,  /* host [n_v_heads][head_v_dim==d_state] */
                               const float *gate_exp, /* host [n_v_heads] */
                               const float *beta,    /* host [n_v_heads] */
                               float *ssm_output,    /* out, host [d_state * n_v_heads] */
                               int n_v_heads, int d_state,
                               int repeat, int device) {
    if (n_v_heads <= 0 || d_state <= 0) return 0;
    if (d_state > 256) return 0;
    if (!ssm_state_dev) return 0;

    if (!gpu_ok(gpuSetDevice(device), "ssm device")) return 0;
    gpu_device_ctx_t *ctx = NULL;
    for (int i = 0; i < g_nctx; i++) {
        if (g_gpu_ctx[i].device == device) { ctx = &g_gpu_ctx[i]; break; }
    }
    if (!ctx) return 0;

    size_t q_bytes = (size_t)(n_v_heads / repeat) * d_state * sizeof(float);
    size_t k_bytes = q_bytes;
    size_t v_bytes = (size_t)n_v_heads * d_state * sizeof(float);
    size_t scalar_bytes = (size_t)n_v_heads * sizeof(float);
    size_t out_bytes = (size_t)d_state * n_v_heads * sizeof(float);

    void *dq, *dk, *dv, *dg, *db, *do_;
    if (!gpu_ok(gpuMalloc(&dq, q_bytes), "ssm q") ||
        !gpu_ok(gpuMalloc(&dk, k_bytes), "ssm k") ||
        !gpu_ok(gpuMalloc(&dv, v_bytes), "ssm v") ||
        !gpu_ok(gpuMalloc(&dg, scalar_bytes), "ssm g") ||
        !gpu_ok(gpuMalloc(&db, scalar_bytes), "ssm b") ||
        !gpu_ok(gpuMalloc(&do_, out_bytes), "ssm o")) return 0;

    if (!gpu_ok(gpuMemcpy(dq, q_conv, q_bytes, gpuMemcpyHostToDevice), "ssm q h2d") ||
        !gpu_ok(gpuMemcpy(dk, k_conv, k_bytes, gpuMemcpyHostToDevice), "ssm k h2d") ||
        !gpu_ok(gpuMemcpy(dv, v_conv, v_bytes, gpuMemcpyHostToDevice), "ssm v h2d") ||
        !gpu_ok(gpuMemcpy(dg, gate_exp, scalar_bytes, gpuMemcpyHostToDevice), "ssm g h2d") ||
        !gpu_ok(gpuMemcpy(db, beta, scalar_bytes, gpuMemcpyHostToDevice), "ssm b h2d")) {
        gpuFree(dq); gpuFree(dk); gpuFree(dv);
        gpuFree(dg); gpuFree(db); gpuFree(do_); return 0;
    }

    /* Warp-shuffle kernel for common d_state values.
     * Gated behind PICOLM_SSM_WARP_KERNEL_VALIDATED -- see the longer
     * comment at the first dispatch site in picolm_gpu_ssm_recurrence()
     * above. Not enabled by default: needs on-device validation. */
    int warp_launched = 0;
#ifdef PICOLM_SSM_WARP_KERNEL_VALIDATED
    if (d_state == 128) {
        constexpr int S_v = 128;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 64) {
        constexpr int S_v = 64;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 32) {
        constexpr int S_v = 32;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 16) {
        constexpr int S_v = 16;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    }
#endif /* PICOLM_SSM_WARP_KERNEL_VALIDATED */
    if (!warp_launched) {
        dim3 grid((unsigned)n_v_heads, 1, 1);
        picolm_ssm_recurrence_kernel<<<grid, 256, 0, ctx->stream>>>(
            (float *)ssm_state_dev, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, d_state, repeat);
    }

    if (!gpu_ok(gpuGetLastError(), "ssm recurrence") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm sync")) {
        gpuFree(dq); gpuFree(dk); gpuFree(dv);
        gpuFree(dg); gpuFree(db); gpuFree(do_); return 0;
    }
    gpuMemcpy(ssm_output, do_, out_bytes, gpuMemcpyDeviceToHost);
    /* State remains on device - no D2H needed */
    gpuFree(dq); gpuFree(dk); gpuFree(dv);
    gpuFree(dg); gpuFree(db); gpuFree(do_);
    return 1;
}

/* Fully device-native: ALL of q_conv/k_conv/v_conv/gate_exp/beta/
 * ssm_output are already device-resident pipeline buffers -- no malloc,
 * no H2D/D2H, no sync at all, unlike picolm_gpu_ssm_recurrence_dev above
 * (which still round-trips q/k/v/gate/beta/output through host memory
 * every call, only state is persistent there). This is the one to use
 * from model_forward_gpu's SSM layer branch. */
extern "C" int
picolm_gpu_ssm_recurrence_pipeline_dev(void *ssm_state_dev,
                                        const float *q_conv_dev,
                                        const float *k_conv_dev,
                                        const float *v_conv_dev,
                                        const float *gate_exp_dev,
                                        const float *beta_dev,
                                        float *ssm_output_dev,
                                        int n_v_heads, int d_state,
                                        int repeat, int device) {
    if (n_v_heads <= 0 || d_state <= 0) return 0;
    if (d_state > 256) return 0;
    if (!ssm_state_dev) return 0;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    /* Warp-shuffle kernel for common d_state values.
     * Gated behind PICOLM_SSM_WARP_KERNEL_VALIDATED -- see the longer
     * comment at the first dispatch site in picolm_gpu_ssm_recurrence()
     * above. Not enabled by default: needs on-device validation. */
    int warp_launched = 0;
#ifdef PICOLM_SSM_WARP_KERNEL_VALIDATED
    if (d_state == 128) {
        constexpr int S_v = 128;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, q_conv_dev, k_conv_dev,
            v_conv_dev, gate_exp_dev, beta_dev,
            ssm_output_dev, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 64) {
        constexpr int S_v = 64;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, q_conv_dev, k_conv_dev,
            v_conv_dev, gate_exp_dev, beta_dev,
            ssm_output_dev, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 32) {
        constexpr int S_v = 32;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, q_conv_dev, k_conv_dev,
            v_conv_dev, gate_exp_dev, beta_dev,
            ssm_output_dev, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 16) {
        constexpr int S_v = 16;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, q_conv_dev, k_conv_dev,
            v_conv_dev, gate_exp_dev, beta_dev,
            ssm_output_dev, n_v_heads, repeat);
        warp_launched = 1;
    }
#endif /* PICOLM_SSM_WARP_KERNEL_VALIDATED */
    if (!warp_launched) {
        dim3 grid((unsigned)n_v_heads, 1, 1);
        picolm_ssm_recurrence_kernel<<<grid, 256, 0, ctx->stream>>>(
            (float *)ssm_state_dev, q_conv_dev, k_conv_dev,
            v_conv_dev, gate_exp_dev, beta_dev,
            ssm_output_dev, n_v_heads, d_state, repeat);
    }
    if (!gpu_ok(gpuGetLastError(), "ssm recurrence (pipeline dev)")) return 0;
    return 1;
}

/* ---- SSM gate/beta activation (Finding 6) ----
 * Direct port of the CPU reference (ssm_forward, steps 9-11):
 *   alpha[h]    = alpha_raw[h] + ssm_dt_w[h]   (bias add, easy to miss --
 *                 caught on a second read-through of the CPU reference;
 *                 the vecdot output alone is NOT what softplus takes)
 *   gate[h]     = softplus(alpha[h]) * ssm_a_w[h]
 *   gate_exp[h] = (gate[h] < -50) ? 0 : exp(gate[h])
 *   beta_out[h] = sigmoid(beta_raw[h])  (no bias -- confirmed against
 *                 the CPU reference, beta's vecdot output goes straight
 *                 into sigmoid)
 * Tiny (n_v_heads elements, e.g. 48) and embarrassingly parallel -- one
 * thread per head, no shared memory or sync needed. */
__global__ void
picolm_gpu_ssm_gate_beta_kernel(float *gate_exp_out, float *beta_out,
                                 const float *alpha_in, const float *beta_raw_in,
                                 const float *ssm_a_w, const float *ssm_dt_w,
                                 int n_v_heads) {
    int h = (int)gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (h >= n_v_heads) return;

    float a = alpha_in[h] + ssm_dt_w[h];
    float sp = (a > 20.0f) ? a : (a < -20.0f) ? expf(a) : logf(1.0f + expf(a));
    float gate = sp * ssm_a_w[h];
    gate_exp_out[h] = (gate < -50.0f) ? 0.0f : expf(gate);

    float braw = beta_raw_in[h];
    beta_out[h] = 1.0f / (1.0f + expf(-braw));
}

/* Device-native: all pointers device-resident, no H2D/D2H, no sync. */
extern "C" int
picolm_gpu_ssm_gate_beta_dev(float *gate_exp_out_dev, float *beta_out_dev,
                              const float *alpha_in_dev, const float *beta_raw_in_dev,
                              const float *ssm_a_w_dev, const float *ssm_dt_w_dev,
                              int n_v_heads, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (n_v_heads < 1) return 0;

    int n_threads = min(n_v_heads, 256);
    int n_blocks = (n_v_heads + n_threads - 1) / n_threads;
    picolm_gpu_ssm_gate_beta_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        gate_exp_out_dev, beta_out_dev, alpha_in_dev, beta_raw_in_dev,
        ssm_a_w_dev, ssm_dt_w_dev, n_v_heads);
    if (!gpu_ok(gpuGetLastError(), "ssm gate/beta (dev)")) return 0;
    return 1;
}

/* ---- SSM L2 normalization (Q/K, per k_head) ----
 * Direct port of the CPU reference (ssm_forward, step 7-8): per-head L2
 * norm (NOT RMS -- no division by head_dim, no learned weight), with an
 * optional fused extra_scale applied after normalizing. The CPU
 * reference does Q's 1/sqrt(d_state) scale as a separate pass (decode
 * path) or fused into the norm (prefill path) -- this kernel always
 * fuses it (extra_scale=q_scale for Q, extra_scale=1.0 for K), matching
 * the prefill reference exactly and reducing to the same result as the
 * decode reference's two-pass version (multiplication is associative
 * here, single scalar factor either way).
 * In-place: x is normalized in place. Grid = n_heads. */
__global__ void
picolm_gpu_ssm_l2norm_kernel(float *x, int head_dim, int n_heads,
                              float eps, float extra_scale) {
    int h = (int)gpuBlockIdx_x;
    if (h >= n_heads) return;
    if (gpuThreadIdx_x != 0) return;
    float *xh = x + (size_t)h * head_dim;

    /* Thread-0 sequential: matches the CPU scalar loop's left-to-right
     * sum exactly (a parallel tree reduction sums in a different order
     * and gives a different, non-bit-identical float result). */
    float nrm = 0.0f;
    for (int d = 0; d < head_dim; d++) nrm += xh[d] * xh[d];
    nrm = (1.0f / sqrtf(nrm + eps)) * extra_scale;
    for (int d = 0; d < head_dim; d++) xh[d] *= nrm;
}

/* Batched-over-tokens version: identical per-(token,head) computation,
 * grid.y = n_tokens instead of a host-side loop over tokens.
 * token_stride is the element stride between consecutive tokens' head
 * groups in x -- pass n_heads*head_dim for a tightly-packed
 * [n_tokens][n_heads][head_dim] buffer, or something larger (e.g.
 * conv_dim) if the head group this call operates on (Q or K) is
 * embedded inside a bigger per-token block alongside other data (as in
 * ssm_prefill_layer's conv_batch, where each token's block is
 * [Q][K][V] and Q/K individually don't sit at a tight n_heads*head_dim
 * stride from one token to the next). In-place. */
__global__ void
picolm_gpu_ssm_l2norm_batch_kernel(float *x, int head_dim, int n_heads,
                                    int n_tokens, int token_stride,
                                    float eps, float extra_scale) {
    int h = (int)gpuBlockIdx_x;
    int t = (int)gpuBlockIdx_y;
    if (h >= n_heads || t >= n_tokens) return;
    if (gpuThreadIdx_x != 0) return;
    float *xh = x + (size_t)t * token_stride + (size_t)h * head_dim;

    float nrm = 0.0f;
    for (int d = 0; d < head_dim; d++) nrm += xh[d] * xh[d];
    float nrm_inv = (1.0f / sqrtf(nrm + eps)) * extra_scale;
    for (int d = 0; d < head_dim; d++) xh[d] *= nrm_inv;
}

/* Device-native, in-place. eps must match the CPU reference (1e-12).
 * extra_scale: pass 1/sqrtf(d_state) for Q, 1.0f for K. */
extern "C" int
picolm_gpu_ssm_l2norm_dev(float *x_dev, int head_dim, int n_heads,
                           float eps, float extra_scale, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (head_dim < 1 || n_heads < 1 || head_dim > 256) return 0;

    int n_threads = min(head_dim, 256);
    dim3 grid((unsigned)n_heads, 1, 1);
    picolm_gpu_ssm_l2norm_kernel<<<grid, n_threads, 0, ctx->stream>>>(
        x_dev, head_dim, n_heads, eps, extra_scale);
    if (!gpu_ok(gpuGetLastError(), "ssm l2norm (dev)")) return 0;
    return 1;
}

/* Host-facing, batched-over-tokens L2-norm: takes a CPU pointer to the
 * start of the first token's head group, does one H2D/D2H round trip
 * for the whole batch (in place). See the kernel comment above for
 * token_stride semantics. Same eps/extra_scale contract as the
 * per-token _dev version above. */
extern "C" int
picolm_gpu_ssm_l2norm_batch(float *x_host, int head_dim, int n_heads,
                             int n_tokens, int token_stride,
                             float eps, float extra_scale,
                             int device) {
    if (head_dim < 1 || n_heads < 1 || head_dim > 256 || n_tokens < 1) return 0;
    if (token_stride < n_heads * head_dim) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    /* Copy the full strided span [0, (n_tokens-1)*token_stride + n_heads*head_dim)
     * in one shot -- includes bytes belonging to other data (K/V) sitting
     * between consecutive tokens' head groups when token_stride is larger
     * than n_heads*head_dim, but that's still one H2D/D2H instead of
     * n_tokens of them, and avoids needing a strided-memcpy helper. */
    size_t span = (size_t)(n_tokens - 1) * token_stride + (size_t)n_heads * head_dim;
    size_t bytes = span * sizeof(float);
    static void *d_x = NULL; static size_t d_x_cap = 0;
    if (!ssm_batch_scratch_ensure(&d_x, &d_x_cap, bytes)) return 0;
    if (!gpu_ok(gpuMemcpy(d_x, x_host, bytes, gpuMemcpyHostToDevice), "l2norm batch h2d")) return 0;

    int n_threads = min(head_dim, 256);
    dim3 grid((unsigned)n_heads, (unsigned)n_tokens, 1);
    picolm_gpu_ssm_l2norm_batch_kernel<<<grid, n_threads, 0, ctx->stream>>>(
        (float *)d_x, head_dim, n_heads, n_tokens, token_stride, eps, extra_scale);

    if (!gpu_ok(gpuGetLastError(), "ssm l2norm batch") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm l2norm batch sync")) return 0;
    return gpu_ok(gpuMemcpy(x_host, d_x, bytes, gpuMemcpyDeviceToHost), "l2norm batch d2h");
}

/* ---- SSM head permute (GGUF v-head remap) ----
 * Generic per-head gather: dst[h] = src[head_map[h]], head_dim elements
 * each. Used for BOTH the xb2 (z-gate) remap and the v_conv remap --
 * same head_map (qwen35_vhead_gguf), same head_dim (head_v_dim), same
 * n_heads (n_v_heads) in both cases. dst and src must NOT alias (the
 * CPU reference uses a temp buffer for exactly this reason -- a
 * permutation isn't safe to do purely in place). */
__global__ void
picolm_gpu_ssm_head_permute_kernel(float *dst, const float *src,
                                    const int *head_map,
                                    int head_dim, int n_heads) {
    int h = (int)gpuBlockIdx_x;
    if (h >= n_heads) return;
    int gh = head_map[h];
    int tid = gpuThreadIdx_x, nt = gpuBlockDim_x;
    const float *srch = src + (size_t)gh * head_dim;
    float *dsth = dst + (size_t)h * head_dim;
    for (int d = tid; d < head_dim; d += nt) dsth[d] = srch[d];
}

extern "C" int
picolm_gpu_ssm_head_permute_dev(float *dst_dev, const float *src_dev,
                                 const int *head_map_dev,
                                 int head_dim, int n_heads, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (head_dim < 1 || n_heads < 1) return 0;

    int n_threads = min(head_dim, 256);
    dim3 grid((unsigned)n_heads, 1, 1);
    picolm_gpu_ssm_head_permute_kernel<<<grid, n_threads, 0, ctx->stream>>>(
        dst_dev, src_dev, head_map_dev, head_dim, n_heads);
    if (!gpu_ok(gpuGetLastError(), "ssm head permute (dev)")) return 0;
    return 1;
}

/* ---- SSM gated normalization ----
 * Direct port of the CPU reference ("18. Gated normalization" in
 * ssm_forward): per-head RMSNorm of the SSM output, scaled by a
 * learned per-dim weight (shared across heads) and gated by
 * silu(xb2) (the z-gate computed earlier in the layer).
 *
 * Two different layouts meet here, exactly as in the CPU code -- get
 * this wrong and every hybrid layer's output is silently corrupted:
 *   ssm_output: dim-major  [head_v_dim][n_v_heads], index d*n_v_heads+h
 *   xb2 (gate): head-major [n_v_heads][head_v_dim], index h*head_v_dim+d
 *   output:     head-major [n_v_heads][head_v_dim], index h*head_v_dim+d
 *     (or, if head_map is non-NULL, written at head_map[h] instead of h
 *      -- this fuses the GGUF v-head remap into the output write,
 *      avoiding the CPU reference's separate fo_gguf permute-copy pass)
 *
 * Grid = n_v_heads blocks; each block does one head's RMS reduction
 * then writes its head_v_dim outputs, both parallelized across all
 * threads (no per-thread-array/serial-thread-0 pattern -- learned that
 * lesson three times already this session). */
__global__ void
picolm_gpu_ssm_gated_norm_kernel(float *final_output,
                                  const float *ssm_output,
                                  const float *xb2,
                                  const float *norm_w,
                                  const int *head_map,
                                  int head_v_dim, int n_v_heads, float eps) {
    int h = (int)gpuBlockIdx_x;
    if (h >= n_v_heads) return;
    if (gpuThreadIdx_x != 0) return;

    /* Thread-0 sequential: matches the CPU scalar loop's left-to-right
     * sum exactly (a parallel tree reduction sums in a different order
     * and gives a different, non-bit-identical float result). */
    float nrm = 0.0f;
    for (int d = 0; d < head_v_dim; d++) {
        float v = ssm_output[(size_t)d * n_v_heads + h];
        nrm += v * v;
    }
    nrm = 1.0f / sqrtf(nrm / head_v_dim + eps);

    int gh = head_map ? head_map[h] : h;
    float *out_h = final_output + (size_t)gh * head_v_dim;
    const float *xb2_h = xb2 + (size_t)h * head_v_dim;

    for (int d = 0; d < head_v_dim; d++) {
        float v = ssm_output[(size_t)d * n_v_heads + h];
        float zv = xb2_h[d];
        float silu_z = zv / (1.0f + expf(-zv));
        out_h[d] = v * nrm * norm_w[d] * silu_z;
    }
}

/* Batched-over-tokens version: identical per-(token,head) computation.
 * ssm_output is [n_tokens][head_v_dim][n_v_heads] (dim-major per
 * token), xb2 and final_output are [n_tokens][n_v_heads][head_v_dim]
 * (head-major per token, final_output's head slot remapped through
 * head_map exactly as in the per-token kernel). */
__global__ void
picolm_gpu_ssm_gated_norm_batch_kernel(float *final_output,
                                        const float *ssm_output,
                                        const float *xb2,
                                        const float *norm_w,
                                        const int *head_map,
                                        int head_v_dim, int n_v_heads,
                                        int n_tokens, float eps) {
    int h = (int)gpuBlockIdx_x;
    int t = (int)gpuBlockIdx_y;
    if (h >= n_v_heads || t >= n_tokens) return;
    if (gpuThreadIdx_x != 0) return;

    const float *so_t = ssm_output + (size_t)t * head_v_dim * n_v_heads;

    float nrm = 0.0f;
    for (int d = 0; d < head_v_dim; d++) {
        float v = so_t[(size_t)d * n_v_heads + h];
        nrm += v * v;
    }
    nrm = 1.0f / sqrtf(nrm / head_v_dim + eps);

    int gh = head_map ? head_map[h] : h;
    float *out_h = final_output + (size_t)t * n_v_heads * head_v_dim + (size_t)gh * head_v_dim;
    const float *xb2_h = xb2 + (size_t)t * n_v_heads * head_v_dim + (size_t)h * head_v_dim;

    for (int d = 0; d < head_v_dim; d++) {
        float v = so_t[(size_t)d * n_v_heads + h];
        float zv = xb2_h[d];
        float silu_z = zv / (1.0f + expf(-zv));
        out_h[d] = v * nrm * norm_w[d] * silu_z;
    }
}

/* Device-native: all pointers device-resident, no H2D/D2H, no sync.
 * head_map_dev may be NULL (identity, no remap). eps matches the CPU
 * reference's default (typically 1e-6 / 1e-5 -- confirm against
 * s->ssm_norm_w's actual eps at the call site, don't hardcode here). */
extern "C" int
picolm_gpu_ssm_gated_norm_dev(float *final_output_dev,
                               const float *ssm_output_dev,
                               const float *xb2_dev,
                               const float *norm_w_dev,
                               const int *head_map_dev,
                               int head_v_dim, int n_v_heads, float eps,
                               int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (head_v_dim < 1 || n_v_heads < 1 || head_v_dim > 256) return 0;

    int n_threads = min(head_v_dim, 256);
    dim3 grid((unsigned)n_v_heads, 1, 1);
    picolm_gpu_ssm_gated_norm_kernel<<<grid, n_threads, 0, ctx->stream>>>(
        final_output_dev, ssm_output_dev, xb2_dev, norm_w_dev, head_map_dev,
        head_v_dim, n_v_heads, eps);
    if (!gpu_ok(gpuGetLastError(), "ssm gated norm (dev)")) return 0;
    return 1;
}

/* Host wrapper for ssm_conv1d (moved here to be after helper functions) */
extern "C" int
picolm_gpu_ssm_conv1d_dev(float *conv_output_dev, float *conv_state_dev,
                           const float *new_input_dev, const float *conv1d_w_dev,
                           int conv_dim, int d_conv, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (conv_dim < 1 || d_conv < 1) return 0;

    int n_threads = 256;
    int n_blocks = (conv_dim + n_threads - 1) / n_threads;
    picolm_gpu_ssm_conv1d_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        conv_output_dev, conv_state_dev, new_input_dev, conv1d_w_dev, conv_dim, d_conv);
    if (!gpu_ok(gpuGetLastError(), "ssm conv1d (dev)")) return 0;
    return 1;
}

/* Host-facing, batched-over-tokens conv1d: takes CPU pointers, does its
 * own H2D/D2H/sync, all in one round trip for the whole n_tokens batch
 * (not one per token). conv_state_host is read once and overwritten
 * with the final window once -- same host buffer ssm_prefill_layer and
 * the CPU-hybrid ssm_forward() decode path both already use as the
 * single source of truth (this does NOT touch the separate persistent
 * gw->ssm_conv_state_dev[il] device buffer that only the currently-
 * disabled ssm_forward_gpu() reads/writes).
 * Returns 0 (caller should fall back to the CPU path -- safe, nothing
 * has been mutated yet) if d_conv exceeds PICOLM_SSM_CONV_MAX_D_CONV. */
extern "C" int
picolm_gpu_ssm_conv1d_batch(float *conv_output_host,      /* out [n_tokens][conv_dim] */
                             float *conv_state_host,       /* in/out [d_conv-1][conv_dim] */
                             const float *new_input_host,  /* in [n_tokens][conv_dim] */
                             const float *conv1d_w_host,   /* in [conv_dim][d_conv] */
                             int conv_dim, int d_conv, int n_tokens, int device) {
    if (conv_dim < 1 || d_conv < 1 || n_tokens < 1) return 0;
    if (d_conv > PICOLM_SSM_CONV_MAX_D_CONV) return 0;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    size_t state_bytes = (size_t)(d_conv - 1) * conv_dim * sizeof(float);
    size_t input_bytes = (size_t)n_tokens * conv_dim * sizeof(float);
    size_t w_bytes = (size_t)conv_dim * d_conv * sizeof(float);
    size_t out_bytes = input_bytes;

    static void *d_out = NULL; static size_t d_out_cap = 0;
    static void *d_state = NULL; static size_t d_state_cap = 0;
    static void *d_in = NULL; static size_t d_in_cap = 0;
    static void *d_w = NULL; static size_t d_w_cap = 0;

    if (!ssm_batch_scratch_ensure(&d_out, &d_out_cap, out_bytes) ||
        !ssm_batch_scratch_ensure(&d_state, &d_state_cap, state_bytes > 0 ? state_bytes : 1) ||
        !ssm_batch_scratch_ensure(&d_in, &d_in_cap, input_bytes) ||
        !ssm_batch_scratch_ensure(&d_w, &d_w_cap, w_bytes)) return 0;

    int ok = 1;
    if (state_bytes > 0)
        ok = ok && gpu_ok(gpuMemcpy(d_state, conv_state_host, state_bytes, gpuMemcpyHostToDevice), "conv1d batch state h2d");
    ok = ok && gpu_ok(gpuMemcpy(d_in, new_input_host, input_bytes, gpuMemcpyHostToDevice), "conv1d batch in h2d");
    ok = ok && gpu_ok(gpuMemcpy(d_w, conv1d_w_host, w_bytes, gpuMemcpyHostToDevice), "conv1d batch w h2d");
    if (!ok) return 0;

    int n_threads = 256;
    int n_blocks = (conv_dim + n_threads - 1) / n_threads;
    int stride = conv_dim;
    picolm_gpu_ssm_conv1d_batch_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        (float *)d_out, (float *)d_state, (const float *)d_in, (const float *)d_w,
        conv_dim, d_conv, n_tokens, stride);

    if (!gpu_ok(gpuGetLastError(), "ssm conv1d batch") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm conv1d batch sync")) return 0;
    ok = gpu_ok(gpuMemcpy(conv_output_host, d_out, out_bytes, gpuMemcpyDeviceToHost), "conv1d batch out d2h");
    if (ok && state_bytes > 0)
        ok = gpu_ok(gpuMemcpy(conv_state_host, d_state, state_bytes, gpuMemcpyDeviceToHost), "conv1d batch state d2h");
    return ok;
}

/* Host-side gated norm wrapper: takes CPU pointers, does its own H2D/D2H/sync.
 * For use from ssm_forward() when the pipeline is not active. */
extern "C" int
picolm_gpu_ssm_gated_norm(float *final_output,  /* out, host [head_v_dim * n_v_heads] */
                           const float *ssm_output, /* in, host [d_state * n_v_heads] */
                           const float *xb2,    /* in, host [head_v_dim * n_v_heads] */
                           const float *norm_w, /* in, host [head_v_dim] */
                           const int *head_map, /* in, host [n_v_heads] or NULL */
                           int head_v_dim, int n_v_heads, float eps,
                           int device) {
    if (head_v_dim < 1 || n_v_heads < 1) return 0;
    if (!gpu_ok(gpuSetDevice(device), "ssm gn")) return 0;
    gpu_device_ctx_t *ctx = NULL;
    for (int i = 0; i < g_nctx; i++) {
        if (g_gpu_ctx[i].device == device) { ctx = &g_gpu_ctx[i]; break; }
    }
    if (!ctx) return 0;

    size_t out_bytes = (size_t)n_v_heads * head_v_dim * sizeof(float);
    size_t so_bytes = (size_t)n_v_heads * head_v_dim * sizeof(float); /* ssm_output: d_state==head_v_dim */
    size_t xb2_bytes = (size_t)n_v_heads * head_v_dim * sizeof(float);
    size_t nw_bytes = head_v_dim * sizeof(float);
    size_t hm_bytes = n_v_heads * sizeof(int);

    void *d_final, *d_so, *d_xb2, *d_nw, *d_hm;
    if (!gpu_ok(gpuMalloc(&d_final, out_bytes), "ssm gn final") ||
        !gpu_ok(gpuMalloc(&d_so, so_bytes), "ssm gn so") ||
        !gpu_ok(gpuMalloc(&d_xb2, xb2_bytes), "ssm gn xb2") ||
        !gpu_ok(gpuMalloc(&d_nw, nw_bytes), "ssm gn nw")) return 0;
    d_hm = NULL;
    if (head_map) {
        if (!gpu_ok(gpuMalloc(&d_hm, hm_bytes), "ssm gn hmap")) {
            gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw); return 0;
        }
    }
    if (!gpu_ok(gpuMemcpy(d_so, ssm_output, so_bytes, gpuMemcpyHostToDevice), "ssm gn so h2d") ||
        !gpu_ok(gpuMemcpy(d_xb2, xb2, xb2_bytes, gpuMemcpyHostToDevice), "ssm gn xb2 h2d") ||
        !gpu_ok(gpuMemcpy(d_nw, norm_w, nw_bytes, gpuMemcpyHostToDevice), "ssm gn nw h2d")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
        if (d_hm) gpuFree(d_hm); return 0;
    }
    if (d_hm && !gpu_ok(gpuMemcpy(d_hm, head_map, hm_bytes, gpuMemcpyHostToDevice), "ssm gn hm h2d")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw); gpuFree(d_hm); return 0;
    }

    picolm_gpu_ssm_gated_norm_kernel<<<(unsigned)n_v_heads, min(head_v_dim, 256), 0, ctx->stream>>>(
        (float *)d_final, (const float *)d_so, (const float *)d_xb2,
        (const float *)d_nw, (const int *)d_hm, head_v_dim, n_v_heads, eps);

    if (!gpu_ok(gpuGetLastError(), "ssm gated norm") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm gn sync")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
        if (d_hm) gpuFree(d_hm); return 0;
    }
    gpuMemcpy(final_output, d_final, out_bytes, gpuMemcpyDeviceToHost);
    gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
    if (d_hm) gpuFree(d_hm);
    return 1;
}

/* Batched-over-tokens version of picolm_gpu_ssm_gated_norm above: one
 * H2D/D2H round trip for the whole n_tokens batch instead of one per
 * token. Layouts match the per-token version with an added leading
 * token dimension -- see picolm_gpu_ssm_gated_norm_batch_kernel's
 * comment for exact strides. */
extern "C" int
picolm_gpu_ssm_gated_norm_batch(float *final_output,  /* out, host [n_tokens][n_v_heads][head_v_dim] */
                                 const float *ssm_output, /* in, host [n_tokens][head_v_dim][n_v_heads] */
                                 const float *xb2,     /* in, host [n_tokens][n_v_heads][head_v_dim] */
                                 const float *norm_w,  /* in, host [head_v_dim] */
                                 const int *head_map,  /* in, host [n_v_heads] or NULL */
                                 int head_v_dim, int n_v_heads, int n_tokens, float eps,
                                 int device) {
    if (head_v_dim < 1 || n_v_heads < 1 || n_tokens < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    size_t out_bytes = (size_t)n_tokens * n_v_heads * head_v_dim * sizeof(float);
    size_t so_bytes = out_bytes;
    size_t xb2_bytes = out_bytes;
    size_t nw_bytes = (size_t)head_v_dim * sizeof(float);
    size_t hm_bytes = (size_t)n_v_heads * sizeof(int);

    void *d_final, *d_so, *d_xb2, *d_nw, *d_hm = NULL;
    if (!gpu_ok(gpuMalloc(&d_final, out_bytes), "ssm gn batch final") ||
        !gpu_ok(gpuMalloc(&d_so, so_bytes), "ssm gn batch so") ||
        !gpu_ok(gpuMalloc(&d_xb2, xb2_bytes), "ssm gn batch xb2") ||
        !gpu_ok(gpuMalloc(&d_nw, nw_bytes), "ssm gn batch nw")) return 0;
    if (head_map && !gpu_ok(gpuMalloc(&d_hm, hm_bytes), "ssm gn batch hmap")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw); return 0;
    }
    if (!gpu_ok(gpuMemcpy(d_so, ssm_output, so_bytes, gpuMemcpyHostToDevice), "ssm gn batch so h2d") ||
        !gpu_ok(gpuMemcpy(d_xb2, xb2, xb2_bytes, gpuMemcpyHostToDevice), "ssm gn batch xb2 h2d") ||
        !gpu_ok(gpuMemcpy(d_nw, norm_w, nw_bytes, gpuMemcpyHostToDevice), "ssm gn batch nw h2d")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
        if (d_hm) gpuFree(d_hm); return 0;
    }
    if (d_hm && !gpu_ok(gpuMemcpy(d_hm, head_map, hm_bytes, gpuMemcpyHostToDevice), "ssm gn batch hm h2d")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw); gpuFree(d_hm); return 0;
    }

    dim3 grid((unsigned)n_v_heads, (unsigned)n_tokens, 1);
    picolm_gpu_ssm_gated_norm_batch_kernel<<<grid, min(head_v_dim, 256), 0, ctx->stream>>>(
        (float *)d_final, (const float *)d_so, (const float *)d_xb2,
        (const float *)d_nw, (const int *)d_hm, head_v_dim, n_v_heads, n_tokens, eps);

    if (!gpu_ok(gpuGetLastError(), "ssm gated norm batch") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm gn batch sync")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
        if (d_hm) gpuFree(d_hm); return 0;
    }
    int ok = gpu_ok(gpuMemcpy(final_output, d_final, out_bytes, gpuMemcpyDeviceToHost), "ssm gn batch out d2h");
    gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
    if (d_hm) gpuFree(d_hm);
    return ok;
}

/* ---- SSM prefill gated normalization ----
 * Direct port of ssm_prefill_layer's "8. Gated normalization" CPU loop
 * (model.c). Deliberately NOT shared with picolm_gpu_ssm_gated_norm(_dev)
 * above: those are direct ports of ssm_forward()'s per-token step 18,
 * which takes dim-major ssm_output and fuses the GGUF head_map remap
 * into the output write. Prefill's chunked recurrence
 * (ssm_chunked_recurrence) already writes its output head-major
 * (matching xb2/z's layout) and applies no remap at this stage -- the
 * remap only happens later, in ssm_prefill_layer's do_remap output-
 * projection branch. Forcing this through the decode-path kernel would
 * mean transposing dim-major<->head-major for no reason and getting the
 * remap timing wrong; a second, simpler kernel matching this layout
 * exactly is safer than reusing one built for a different layout.
 *
 * In-place on ssm_out. Same thread-0-per-(token,head) sequential
 * accumulation as every other norm kernel in this file, to bit-match
 * the CPU scalar reference's left-to-right sum. */
__global__ void
picolm_gpu_ssm_prefill_gated_norm_kernel(float *ssm_out,   /* in/out [n_tokens][n_v_heads][head_v_dim] */
                                          const float *z,   /* in [n_tokens][n_v_heads][head_v_dim] */
                                          const float *norm_w, /* in [head_v_dim] */
                                          int head_v_dim, int n_v_heads,
                                          int n_tokens, float eps,
                                          int so_stride, int z_stride) {
    int h = (int)gpuBlockIdx_x;
    int t = (int)gpuBlockIdx_y;
    if (h >= n_v_heads || t >= n_tokens) return;
    if (gpuThreadIdx_x != 0) return;

    float *out_h = ssm_out + (size_t)t * so_stride + (size_t)h * head_v_dim;
    const float *z_h = z + (size_t)t * z_stride + (size_t)h * head_v_dim;

    float nrm = 0.0f;
    for (int d = 0; d < head_v_dim; d++) {
        float v = out_h[d];
        nrm += v * v;
    }
    nrm = 1.0f / sqrtf(nrm / (float)head_v_dim + eps);

    for (int d = 0; d < head_v_dim; d++) {
        float v = out_h[d];
        float zv = z_h[d];
        float silu_z = zv / (1.0f + expf(-zv));
        out_h[d] = v * nrm * norm_w[d] * silu_z;
    }
}

extern "C" int
picolm_gpu_ssm_prefill_gated_norm(float *ssm_out_host,   /* in/out [n_tokens][n_v_heads][head_v_dim] */
                                   const float *z_host,   /* in [n_tokens][n_v_heads][head_v_dim] */
                                   const float *norm_w_host, /* in [head_v_dim] */
                                   int head_v_dim, int n_v_heads, int n_tokens, float eps,
                                   int device) {
    if (head_v_dim < 1 || n_v_heads < 1 || n_tokens < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    size_t so_bytes = (size_t)n_tokens * n_v_heads * head_v_dim * sizeof(float);
    size_t nw_bytes = (size_t)head_v_dim * sizeof(float);

    static void *d_so = NULL; static size_t d_so_cap = 0;
    static void *d_z = NULL; static size_t d_z_cap = 0;
    static void *d_nw = NULL; static size_t d_nw_cap = 0;

    if (!ssm_batch_scratch_ensure(&d_so, &d_so_cap, so_bytes) ||
        !ssm_batch_scratch_ensure(&d_z, &d_z_cap, so_bytes) ||
        !ssm_batch_scratch_ensure(&d_nw, &d_nw_cap, nw_bytes)) return 0;

    if (!gpu_ok(gpuMemcpy(d_so, ssm_out_host, so_bytes, gpuMemcpyHostToDevice), "prefill gn so h2d") ||
        !gpu_ok(gpuMemcpy(d_z, z_host, so_bytes, gpuMemcpyHostToDevice), "prefill gn z h2d") ||
        !gpu_ok(gpuMemcpy(d_nw, norm_w_host, nw_bytes, gpuMemcpyHostToDevice), "prefill gn nw h2d")) return 0;

    int so_stride = n_v_heads * head_v_dim, z_stride = n_v_heads * head_v_dim;
    dim3 grid((unsigned)n_v_heads, (unsigned)n_tokens, 1);
    picolm_gpu_ssm_prefill_gated_norm_kernel<<<grid, min(head_v_dim, 256), 0, ctx->stream>>>(
        (float *)d_so, (const float *)d_z, (const float *)d_nw, head_v_dim, n_v_heads, n_tokens, eps, so_stride, z_stride);

    if (!gpu_ok(gpuGetLastError(), "ssm prefill gated norm") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm prefill gn sync")) return 0;
    return gpu_ok(gpuMemcpy(ssm_out_host, d_so, so_bytes, gpuMemcpyDeviceToHost), "prefill gn out d2h");
}
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
    if (qtype != 0 && qtype != 2 && qtype != 8) return 0;
    if (dim > PICOLM_SSM_VECDOT_MAX_DIM) return 0;

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
    picolm_ssm_vecdot_kernel<<<grid, 256, 0, ctx->stream>>>(
        ctx->y, ctx->x, w_dev, qtype, dim, n_v_heads, row_bytes,
        hm_dev ? (const int *)hm_dev : NULL);

    if (!gpu_ok(gpuDeviceSynchronize(), "ssm vecdot sync") ||
        !gpu_ok(gpuMemcpy(out_host, ctx->y, out_bytes, gpuMemcpyDeviceToHost), "ssm out d2h")) {
        gpuFree(w_dev);
        if (hm_dev) gpuFree(hm_dev);
        return 0;
    }
    gpuFree(w_dev);
    if (hm_dev) gpuFree(hm_dev);
    return 1;
}

/* Batched-over-tokens version of picolm_gpu_ssm_vecdot above: uploads
 * the weight matrix and head_map ONCE for the whole n_tokens batch
 * (instead of once per token, which is what calling picolm_gpu_ssm_vecdot
 * n_tokens times in a loop would do), and does a single H2D for all
 * n_tokens' worth of x and a single D2H for all outputs. */
extern "C" int
picolm_gpu_ssm_vecdot_batch(float *out_host,       /* out [n_tokens][n_v_heads] */
                             const float *x_host,   /* in [n_tokens][dim] */
                             const void *weights_host,
                             gguf_type_t qtype,
                             int dim, int n_v_heads, int n_tokens,
                             int row_bytes,
                             const int *head_map,
                             int device) {
    if (n_v_heads <= 0 || dim <= 0 || n_tokens <= 0) return 0;
    if (qtype != 0 && qtype != 2 && qtype != 8) return 0;
    if (dim > PICOLM_SSM_VECDOT_MAX_DIM) return 0;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    size_t x_bytes = (size_t)n_tokens * dim * sizeof(float);
    size_t w_bytes = (size_t)n_v_heads * row_bytes;
    size_t out_bytes = (size_t)n_tokens * n_v_heads * sizeof(float);
    size_t hm_bytes = (size_t)n_v_heads * sizeof(int);

    static void *x_dev = NULL; static size_t x_dev_cap = 0;
    static void *w_dev = NULL; static size_t w_dev_cap = 0;
    static void *out_dev = NULL; static size_t out_dev_cap = 0;
    static void *hm_dev = NULL; static size_t hm_dev_cap = 0;

    if (!ssm_batch_scratch_ensure(&x_dev, &x_dev_cap, x_bytes) ||
        !ssm_batch_scratch_ensure(&w_dev, &w_dev_cap, w_bytes) ||
        !ssm_batch_scratch_ensure(&out_dev, &out_dev_cap, out_bytes)) return 0;
    if (head_map && !ssm_batch_scratch_ensure(&hm_dev, &hm_dev_cap, hm_bytes)) return 0;

    int ok = gpu_ok(gpuMemcpy(x_dev, x_host, x_bytes, gpuMemcpyHostToDevice), "vecdot batch x h2d") &&
             gpu_ok(gpuMemcpy(w_dev, weights_host, w_bytes, gpuMemcpyHostToDevice), "vecdot batch w h2d");
    if (ok && head_map)
        ok = gpu_ok(gpuMemcpy(hm_dev, head_map, hm_bytes, gpuMemcpyHostToDevice), "vecdot batch hmap h2d");
    if (!ok) return 0;

    dim3 grid((unsigned)n_v_heads, (unsigned)n_tokens, 1);
    picolm_ssm_vecdot_batch_kernel<<<grid, 256, 0, ctx->stream>>>(
        (float *)out_dev, (const float *)x_dev, w_dev, qtype, dim, n_v_heads, n_tokens,
        row_bytes, head_map ? (const int *)hm_dev : NULL, 0);

    if (!gpu_ok(gpuGetLastError(), "ssm vecdot batch") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm vecdot batch sync")) return 0;
    return gpu_ok(gpuMemcpy(out_host, out_dev, out_bytes, gpuMemcpyDeviceToHost), "vecdot batch out d2h");
}

/* Fully device-native SSM vecdot: weights_dev and head_map_dev must
 * already be device-resident, uploaded ONCE at model load (via
 * picolm_gpu_tensor_upload for quantized weights or picolm_gpu_upload_f32
 * for F32 ssm_alpha/ssm_beta, and a one-time int array upload for the
 * head_map) -- NOT re-uploaded every call like picolm_gpu_ssm_vecdot()
 * above does. x_dev/out_dev are pipeline buffers. No malloc, no H2D/D2H,
 * no internal sync -- same ctx->stream ordering argument as every other
 * _dev primitive this session. head_map_dev may be NULL (identity). */
extern "C" int
picolm_gpu_ssm_vecdot_dev(float *out_dev,
                           const float *x_dev,
                           const void *weights_dev,
                           gguf_type_t qtype,
                           int dim, int n_v_heads,
                           int row_bytes,
                           const int *head_map_dev,
                           int device) {
    if (n_v_heads <= 0 || dim <= 0) return 0;
    if (qtype != 0 && qtype != 2 && qtype != 8) return 0;
    if (dim > PICOLM_SSM_VECDOT_MAX_DIM) return 0;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    dim3 grid((unsigned)n_v_heads, 1, 1);
    picolm_ssm_vecdot_kernel<<<grid, 256, 0, ctx->stream>>>(
        out_dev, x_dev, weights_dev, qtype, dim, n_v_heads, row_bytes, head_map_dev);
    if (!gpu_ok(gpuGetLastError(), "ssm vecdot (dev)")) return 0;
    return 1;
}

/* ================================================================
 * Chunked DeltaNet SSM recurrence -- GPU port (prefill).
 *
 * NOT VALIDATED ON REAL HARDWARE IN THIS SESSION. Gated behind
 * PICOLM_SSM_CHUNKED_GPU_VALIDATED (undefined by default -- the host
 * driver at the bottom of this section returns 0 unconditionally
 * unless that macro is defined by the build, so calling it is always
 * safe/no-op until someone has actually run it through the validation
 * plan below and flipped the flag on purpose).
 *
 * Direct, deliberately unoptimized (one-thread-per-output-element, no
 * tiling/shared-memory GEMM) port of the scalar reference path in
 * ssm_chunk_head_task() / ssm_chunked_recurrence() (model.c) -- ported
 * from the #else scalar branch there, not the AVX/NEON micro-kernel
 * branches (multiple SIMD variants exist; the scalar branch is the one
 * unambiguous reference all of them are optimizing). See
 * chunked_ssm_gpu_design.md in the project notes for:
 *   - why bit-exactness with the CPU path is NOT the goal (unlike the
 *     single-token recurrence kernel, these are GEMM-shaped reductions
 *     over d_state~128 elements -- forcing a specific summation order
 *     there would mean giving up tiling, i.e. giving up most of the
 *     performance point)
 *   - the recommended validation plan (per-chunk numerical comparison
 *     against ssm_chunk_head_task before trusting end-to-end output)
 *   - why every kernel below batches per-v-head as one threadblock,
 *     matching the CPU's tensor_parallel_for(n_v_heads, ...) head
 *     parallelization, with chunks processed sequentially on the host
 *     side (state carries chunk-to-chunk within a head)
 *
 * Buffer layout (matches ssm_chunk_head_task's scratch pool exactly,
 * so this can be checked line-by-line against it):
 *   chunk_q, chunk_k:  [n_k_heads][cs][d]   (gathered once per chunk)
 *   chunk_v:           [n_v_heads][cs][d]
 *   chunk_beta, gate_log, cum_g, q_decay:  [n_v_heads][cs]
 *   decay_mask, M_mat, kq:                 [n_v_heads][cs][cs]
 *   v_eff, v_hat, sk, sq, chunk_out:        [n_v_heads][cs][d]
 *   state:                                  [n_v_heads][d][d]
 *
 * `d` below is shared between K/Q (d_state) and V/out/state
 * (head_v_dim) -- this whole algorithm relies on d_state == head_v_dim
 * (state is square, V/out are d_state-wide per head), which
 * ssm_chunk_head_task() on the CPU side already relies on too (it
 * indexes chunk_v, gathered at head_v_dim stride, using ctx->d_state).
 * The host driver checks this invariant explicitly and refuses to
 * dispatch (safe fallback to CPU) if it doesn't hold, rather than
 * assume it silently.
 * ================================================================ */

/* ---- Gather: gate_log/beta/Q/K/V for one chunk, from the token-major
 * conv_batch/alpha_batch/beta_batch buffers, into the head-major
 * layout the rest of this pipeline uses. Direct port of the CPU gather
 * loops in ssm_chunked_recurrence(). Two kernels (Q/K vs V+scalars)
 * since they're indexed by different head counts (n_k_heads vs
 * n_v_heads). ---- */
__global__ void
ssm_chunk_gather_qk_kernel(float *chunk_q, float *chunk_k,
                            const float *conv_batch,
                            int chunk_start, int cs_actual, int conv_stride,
                            int qk_dim, int d_state, int n_k_heads) {
    int h = blockIdx.x;
    if (h >= n_k_heads) return;
    float *cq_h = chunk_q + (size_t)h * cs_actual * d_state;
    float *ck_h = chunk_k + (size_t)h * cs_actual * d_state;
    int n_elem = cs_actual * d_state;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int t = idx / d_state, di = idx % d_state;
        const float *tok = conv_batch + (size_t)(chunk_start + t) * conv_stride;
        cq_h[idx] = tok[h * d_state + di];
        ck_h[idx] = tok[qk_dim + h * d_state + di];
    }
}

__global__ void
ssm_chunk_gather_v_kernel(float *chunk_v, float *chunk_beta, float *gate_log,
                           const float *conv_batch, const float *alpha_batch,
                           const float *beta_batch,
                           int chunk_start, int cs_actual, int conv_stride,
                           int qk_dim, int head_v_dim, int n_v_heads,
                           int ab_stride) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    float *cv_h = chunk_v + (size_t)h * cs_actual * head_v_dim;
    float *cb_h = chunk_beta + (size_t)h * cs_actual;
    float *gl_h = gate_log + (size_t)h * cs_actual;

    int n_elem = cs_actual * head_v_dim;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int t = idx / head_v_dim, di = idx % head_v_dim;
        const float *tok = conv_batch + (size_t)(chunk_start + t) * conv_stride;
        cv_h[idx] = tok[2 * qk_dim + h * head_v_dim + di];
    }
    int as = ab_stride > 0 ? ab_stride : n_v_heads;
    for (int t = threadIdx.x; t < cs_actual; t += blockDim.x) {
        cb_h[t] = beta_batch[(size_t)(chunk_start + t) * as + h];
        /* alpha_batch now contains gate_log directly (log-space), not expf(gate).
         * No logf needed -- direct copy, matching the CPU path. */
        gl_h[t] = alpha_batch[(size_t)(chunk_start + t) * as + h];
    }
}

/* ---- Step 1: cumulative log-decay + decay mask. Direct port of the
 * CPU's "Step 1" + "Build decay mask" blocks. Thread 0 does the
 * inherently-sequential prefix sum (matches CPU's ascending-t loop
 * exactly, same clamping, same expf calls); all threads then fill the
 * cs x cs mask in parallel once cum_g is visible via __syncthreads(). ---- */
__global__ void
ssm_chunk_decay_kernel(float *cum_g, float *q_decay, float *decay_mask,
                        const float *gate_log, int n_v_heads, int cs, int gl_stride) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    const float *gl = gate_log + (size_t)h * cs;
    (void)gl_stride; /* scratch buffer is contiguous head-major */
    float *cg = cum_g + (size_t)h * cs;
    float *qd = q_decay + (size_t)h * cs;
    float *dm = decay_mask + (size_t)h * cs * cs;

    if (threadIdx.x == 0) {
        float cum = 0.0f;
        for (int t = 0; t < cs; t++) {
            cum += gl[t];
            cg[t] = cum;
            float ex = cum;
            if (ex > 50.0f) ex = 50.0f;
            if (ex < -50.0f) ex = -50.0f;
            qd[t] = expf(ex);
        }
    }
    __syncthreads();

    int n_elem = cs * cs;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int i = idx / cs, j = idx % cs;
        float v;
        if (j > i) v = 0.0f;
        else if (i == j) v = 1.0f;
        else {
            float diff = cg[i] - cg[j];
            if (diff > 50.0f) diff = 50.0f;
            if (diff < -50.0f) diff = -50.0f;
            v = expf(diff);
        }
        dm[idx] = v;
    }
}

/* ---- Masked cs x cs GEMM: out[h][i][j] = dot(A[kh][i], B[kh][j]) *
 * decay_mask[h][i][j] for j<=i, else 0. Reused for both M (A=B=chunk_k,
 * "Step 2" in the CPU scalar path) and kq (A=chunk_q, B=chunk_k,
 * part of CPU "Step 5"). One threadblock per v-head, threads stride
 * over the cs*cs output elements; each does a length-d dot product --
 * no tiling, deliberately simple for a first correctness pass (see
 * design doc). ---- */
__global__ void
ssm_chunk_masked_gemm_kernel(float *out_cs_cs, const float *A, const float *B,
                              const float *decay_mask,
                              int n_v_heads, int repeat, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    int kh = h / repeat;
    const float *Ah = A + (size_t)kh * cs * d;
    const float *Bh = B + (size_t)kh * cs * d;
    const float *dm = decay_mask + (size_t)h * cs * cs;
    float *out = out_cs_cs + (size_t)h * cs * cs;

    int n_elem = cs * cs;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int i = idx / cs, j = idx % cs;
        if (j > i) { out[idx] = 0.0f; continue; }
        const float *ai = Ah + (size_t)i * d;
        const float *bj = Bh + (size_t)j * d;
        float dot = 0.0f;
        for (int c = 0; c < d; c++) dot += ai[c] * bj[c];
        out[idx] = dot * dm[idx];
    }
}

/* ---- cs x d matvec-GEMM: out[h][i][r] = sum_c state[h][r][c] * X[kh][i][c],
 * i.e. out = X @ State^T. Reused for sk (X=chunk_k) and sq (X=chunk_q)
 * -- CPU "Kernel 1"/"Kernel 1b" in the SIMD path, inlined into "Step 3"
 * /"Step 5" in the scalar path. ---- */
__global__ void
ssm_chunk_matvec_kernel(float *out, const float *state, const float *chunk_x,
                         int n_v_heads, int repeat, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    int kh = h / repeat;
    const float *x_h = chunk_x + (size_t)kh * cs * d;
    const float *st = state + (size_t)h * d * d;
    float *o = out + (size_t)h * cs * d;

    int n_elem = cs * d;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int i = idx / d, r = idx % d;
        const float *xi = x_h + (size_t)i * d;
        const float *st_row = st + (size_t)r * d;
        float sum = 0.0f;
        for (int c = 0; c < d; c++) sum += st_row[c] * xi[c];
        o[idx] = sum;
    }
}

/* ---- V_eff assembly: direct port of CPU "Step 3". ---- */
__global__ void
ssm_chunk_veff_kernel(float *v_eff, const float *chunk_v, const float *sk,
                       const float *q_decay, const float *chunk_beta,
                       int n_v_heads, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    const float *v_h = chunk_v + (size_t)h * cs * d;
    const float *sk_h = sk + (size_t)h * cs * d;
    const float *qd_h = q_decay + (size_t)h * cs;
    const float *bt_h = chunk_beta + (size_t)h * cs;
    float *ve_h = v_eff + (size_t)h * cs * d;

    int n_elem = cs * d;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int i = idx / d;
        ve_h[idx] = bt_h[i] * (v_h[idx] - qd_h[i] * sk_h[idx]);
    }
}

/* ---- Forward substitution / triangular solve for V_hat: the one
 * inherently sequential part of this whole algorithm. Direct port of
 * CPU "Step 4". One threadblock per v-head; blockDim.x should cover d
 * (loops if not). cs sequential steps over i, ascending, with
 * __syncthreads() between each: v_hat[i] must be fully written by all
 * r-threads before any thread reads it for i+1 (every later i' > i
 * reads all of v_hat[0..i'-1]). This is the piece flagged in the
 * design doc as needing the most dedicated validation (small-cs and
 * boundary-chunk cases especially). ---- */
__global__ void
ssm_chunk_trisolve_kernel(float *v_hat, const float *v_eff, const float *M_mat,
                           const float *chunk_beta,
                           int n_v_heads, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    const float *ve_h = v_eff + (size_t)h * cs * d;
    const float *M_h = M_mat + (size_t)h * cs * cs;
    const float *bt_h = chunk_beta + (size_t)h * cs;
    float *vh_h = v_hat + (size_t)h * cs * d;

    for (int i = 0; i < cs; i++) {
        float bt = bt_h[i];
        const float *Mi = M_h + (size_t)i * cs;
        for (int r = threadIdx.x; r < d; r += blockDim.x) {
            float sum_mv = 0.0f;
            for (int j = 0; j < i; j++) sum_mv += Mi[j] * vh_h[(size_t)j * d + r];
            vh_h[(size_t)i * d + r] = ve_h[(size_t)i * d + r] - bt * sum_mv;
        }
        __syncthreads();
    }
}

/* ---- Output assembly: direct port of CPU "Step 5". kq already has
 * the decay_mask factor folded in (from ssm_chunk_masked_gemm_kernel),
 * matching kq[i][j] == CPU's `attn` exactly -- no separate mask lookup
 * needed here, same as the CPU scalar path's `attn = k_dot_q * dm`
 * being precomputed rather than looked up in the AVX/NEON paths. ---- */
__global__ void
ssm_chunk_output_kernel(float *chunk_out, const float *sq, const float *q_decay,
                         const float *kq, const float *v_hat,
                         int n_v_heads, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    const float *sq_h = sq + (size_t)h * cs * d;
    const float *qd_h = q_decay + (size_t)h * cs;
    const float *kq_h = kq + (size_t)h * cs * cs;
    const float *vh_h = v_hat + (size_t)h * cs * d;
    float *out_h = chunk_out + (size_t)h * cs * d;

    int n_elem = cs * d;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int i = idx / d, r = idx % d;
        float acc = sq_h[idx] * qd_h[i];
        const float *kqi = kq_h + (size_t)i * cs;
        for (int j = 0; j <= i; j++)
            acc += kqi[j] * vh_h[(size_t)j * d + r];
        out_h[idx] = acc;
    }
}

/* ---- State update: direct port of CPU "Step 6" (scalar path),
 * including its exact redundant per-(r,c,j) recomputation of
 * decay_to_end rather than precomputing it once per j -- kept for
 * fidelity to the reference rather than "optimized" into a precompute
 * pass, since this is a correctness-first port. ---- */
__global__ void
ssm_chunk_state_update_kernel(float *state, const float *v_hat, const float *chunk_k,
                               const float *cum_g,
                               int n_v_heads, int repeat, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    int kh = h / repeat;
    const float *vh_h = v_hat + (size_t)h * cs * d;
    const float *k_h = chunk_k + (size_t)kh * cs * d;
    const float *cg_h = cum_g + (size_t)h * cs;
    float *st_h = state + (size_t)h * d * d;

    float cum_last = cg_h[cs - 1];
    float ex_total = cum_last;
    if (ex_total > 50.0f) ex_total = 50.0f;
    if (ex_total < -50.0f) ex_total = -50.0f;
    float total_decay = expf(ex_total);

    int n_elem = d * d;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int r = idx / d, c = idx % d;
        float update = 0.0f;
        for (int j = 0; j < cs; j++) {
            float diff = cum_last - cg_h[j];
            if (diff > 50.0f) diff = 50.0f;
            if (diff < -50.0f) diff = -50.0f;
            float decay_to_end = expf(diff);
            update += vh_h[(size_t)j * d + r] * k_h[(size_t)j * d + c] * decay_to_end;
        }
        st_h[idx] = st_h[idx] * total_decay + update;
    }
}

/* ---- Scatter: chunk_out [n_v_heads][cs][d] -> xb2_batch
 * [n_tokens][value_dim] head-major, matching ssm_chunked_recurrence()'s
 * CPU reshape loop exactly. ---- */
__global__ void
ssm_chunk_scatter_kernel(float *xb2_batch, const float *chunk_out,
                          int chunk_start, int cs_actual, int xb2_stride,
                          int head_v_dim, int n_v_heads) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    const float *co_h = chunk_out + (size_t)h * cs_actual * head_v_dim;
    int n_elem = cs_actual * head_v_dim;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int t = idx / head_v_dim, r = idx % head_v_dim;
        xb2_batch[(size_t)(chunk_start + t) * xb2_stride + h * head_v_dim + r] = co_h[idx];
    }
}

/* ---- Host driver ----
 * Mirrors ssm_chunked_recurrence()'s structure: allocate scratch sized
 * for the max chunk size once, then loop over chunks sequentially
 * (state carries forward within a head across chunks -- chunks
 * themselves cannot be parallelized). Everything (conv_batch, alpha,
 * beta, state, xb2_batch) is uploaded/downloaded ONCE for the whole
 * call, not once per chunk: state and xb2_batch stay device-resident
 * across the entire chunk loop, only leaving the GPU once at the end
 * on success.
 *
 * Safe to fall back to the CPU path on ANY failure, at ANY point in
 * the chunk loop: state_host and xb2_batch_host are only overwritten
 * by the final D2H copies after every chunk has succeeded and the
 * whole thing has been synced -- unlike the decode-path _dev
 * functions, there's no in-place mutation of the sole copy of truth
 * here (same reasoning as picolm_gpu_ssm_conv1d_batch above). On
 * failure, the device-side partial work is simply discarded and
 * state_host/xb2_batch_host are exactly as they were on entry, so the
 * caller's ssm_chunked_recurrence() CPU fallback is always correct.
 *
 * Returns 0 unconditionally unless PICOLM_SSM_CHUNKED_GPU_VALIDATED is
 * defined by the build -- see the section-header comment above. */
extern "C" int
picolm_gpu_ssm_chunked_recurrence(const float *conv_batch_host,
                                   const float *alpha_batch_host,
                                   const float *beta_batch_host,
                                   float *state_host,
                                   float *xb2_batch_host,
                                   int n_tokens, int value_dim,
                                   int d_state, int n_k_heads, int n_v_heads,
                                   int head_v_dim, int repeat,
                                   int conv_dim, int cs, int device) {
#ifndef PICOLM_SSM_CHUNKED_GPU_VALIDATED
    (void)conv_batch_host; (void)alpha_batch_host; (void)beta_batch_host;
    (void)state_host; (void)xb2_batch_host; (void)n_tokens; (void)value_dim;
    (void)d_state; (void)n_k_heads; (void)n_v_heads; (void)head_v_dim;
    (void)repeat; (void)conv_dim; (void)cs; (void)device;
    return 0;
#else
    if (d_state != head_v_dim) return 0; /* architectural invariant this whole port relies on */
    if (cs <= 0) cs = 64;
    if (cs > n_tokens) cs = n_tokens;
    int n_chunks = (n_tokens + cs - 1) / cs;
    if (n_chunks < 1) n_chunks = 1;
    int qk_dim = d_state * n_k_heads;
    int d = d_state;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    /* Per-call, uploaded/downloaded once */
    size_t conv_bytes = (size_t)n_tokens * conv_dim * sizeof(float);
    size_t alpha_bytes = (size_t)n_tokens * n_v_heads * sizeof(float);
    size_t state_bytes = (size_t)n_v_heads * d * d * sizeof(float);
    size_t xb2_bytes = (size_t)n_tokens * value_dim * sizeof(float);

    /* Per-chunk scratch, sized for the max chunk size, reused every chunk */
    size_t qk_sc_bytes = (size_t)n_k_heads * cs * d * sizeof(float);
    size_t v_sc_bytes = (size_t)n_v_heads * cs * d * sizeof(float);
    size_t scalar_sc_bytes = (size_t)n_v_heads * cs * sizeof(float);
    size_t sq_sc_bytes = (size_t)n_v_heads * cs * cs * sizeof(float);

    /* All buffers persistent/grow-only across calls (see
     * ssm_batch_scratch_ensure's comment): this function alone was
     * ~20 gpuMalloc/gpuFree pairs EVERY call, and it's called once per
     * layer -- for a 48-layer model that's ~1000 alloc/free pairs per
     * prompt just from this one function. Grown to the largest chunk
     * size seen so far and never freed until the device changes. */
    static void *d_conv = NULL; static size_t d_conv_cap = 0;
    static void *d_alpha = NULL; static size_t d_alpha_cap = 0;
    static void *d_beta = NULL; static size_t d_beta_cap = 0;
    static void *d_state_buf = NULL; static size_t d_state_buf_cap = 0;
    static void *d_xb2 = NULL; static size_t d_xb2_cap = 0;
    static void *d_chunk_q = NULL; static size_t d_chunk_q_cap = 0;
    static void *d_chunk_k = NULL; static size_t d_chunk_k_cap = 0;
    static void *d_chunk_v = NULL; static size_t d_chunk_v_cap = 0;
    static void *d_chunk_beta = NULL; static size_t d_chunk_beta_cap = 0;
    static void *d_gate_log = NULL; static size_t d_gate_log_cap = 0;
    static void *d_cum_g = NULL; static size_t d_cum_g_cap = 0;
    static void *d_q_decay = NULL; static size_t d_q_decay_cap = 0;
    static void *d_decay_mask = NULL; static size_t d_decay_mask_cap = 0;
    static void *d_M = NULL; static size_t d_M_cap = 0;
    static void *d_kq = NULL; static size_t d_kq_cap = 0;
    static void *d_v_eff = NULL; static size_t d_v_eff_cap = 0;
    static void *d_v_hat = NULL; static size_t d_v_hat_cap = 0;
    static void *d_sk = NULL; static size_t d_sk_cap = 0;
    static void *d_sq = NULL; static size_t d_sq_cap = 0;
    static void *d_chunk_out = NULL; static size_t d_chunk_out_cap = 0;

    int ok = 1;
    ok = ok && ssm_batch_scratch_ensure(&d_conv, &d_conv_cap, conv_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_alpha, &d_alpha_cap, alpha_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_beta, &d_beta_cap, alpha_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_state_buf, &d_state_buf_cap, state_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_xb2, &d_xb2_cap, xb2_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_chunk_q, &d_chunk_q_cap, qk_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_chunk_k, &d_chunk_k_cap, qk_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_chunk_v, &d_chunk_v_cap, v_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_chunk_beta, &d_chunk_beta_cap, scalar_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_gate_log, &d_gate_log_cap, scalar_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_cum_g, &d_cum_g_cap, scalar_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_q_decay, &d_q_decay_cap, scalar_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_decay_mask, &d_decay_mask_cap, sq_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_M, &d_M_cap, sq_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_kq, &d_kq_cap, sq_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_v_eff, &d_v_eff_cap, v_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_v_hat, &d_v_hat_cap, v_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_sk, &d_sk_cap, v_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_sq, &d_sq_cap, v_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_chunk_out, &d_chunk_out_cap, v_sc_bytes);

    if (ok) {
        ok = ok && gpu_ok(gpuMemcpy(d_conv, conv_batch_host, conv_bytes, gpuMemcpyHostToDevice), "chunk conv h2d");
        ok = ok && gpu_ok(gpuMemcpy(d_alpha, alpha_batch_host, alpha_bytes, gpuMemcpyHostToDevice), "chunk alpha h2d");
        ok = ok && gpu_ok(gpuMemcpy(d_beta, beta_batch_host, alpha_bytes, gpuMemcpyHostToDevice), "chunk beta h2d");
        ok = ok && gpu_ok(gpuMemcpy(d_state_buf, state_host, state_bytes, gpuMemcpyHostToDevice), "chunk state h2d");
    }

    if (ok) {
        int n_threads = 256;
        for (int ci = 0; ci < n_chunks && ok; ci++) {
            int cs_actual = (ci == n_chunks - 1) ? (n_tokens - ci * cs) : cs;
            if (cs_actual <= 0) break;
            int chunk_start = ci * cs;

            ssm_chunk_gather_qk_kernel<<<n_k_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_chunk_q, (float *)d_chunk_k, (const float *)d_conv,
                chunk_start, cs_actual, conv_dim, qk_dim, d_state, n_k_heads);
            ssm_chunk_gather_v_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_chunk_v, (float *)d_chunk_beta, (float *)d_gate_log,
                (const float *)d_conv, (const float *)d_alpha, (const float *)d_beta,
                chunk_start, cs_actual, conv_dim, qk_dim, head_v_dim, n_v_heads, 0);

            ssm_chunk_decay_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_cum_g, (float *)d_q_decay, (float *)d_decay_mask,
                (const float *)d_gate_log, n_v_heads, cs_actual, cs_actual);

            ssm_chunk_masked_gemm_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_M, (const float *)d_chunk_k, (const float *)d_chunk_k,
                (const float *)d_decay_mask, n_v_heads, repeat, cs_actual, d);

            ssm_chunk_matvec_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_sk, (const float *)d_state_buf, (const float *)d_chunk_k,
                n_v_heads, repeat, cs_actual, d);

            ssm_chunk_veff_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_v_eff, (const float *)d_chunk_v, (const float *)d_sk,
                (const float *)d_q_decay, (const float *)d_chunk_beta, n_v_heads, cs_actual, d);

            {
                int tri_threads = d < 1024 ? d : 1024;
                ssm_chunk_trisolve_kernel<<<n_v_heads, tri_threads, 0, ctx->stream>>>(
                    (float *)d_v_hat, (const float *)d_v_eff, (const float *)d_M,
                    (const float *)d_chunk_beta, n_v_heads, cs_actual, d);
            }

            ssm_chunk_matvec_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_sq, (const float *)d_state_buf, (const float *)d_chunk_q,
                n_v_heads, repeat, cs_actual, d);

            ssm_chunk_masked_gemm_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_kq, (const float *)d_chunk_q, (const float *)d_chunk_k,
                (const float *)d_decay_mask, n_v_heads, repeat, cs_actual, d);

            ssm_chunk_output_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_chunk_out, (const float *)d_sq, (const float *)d_q_decay,
                (const float *)d_kq, (const float *)d_v_hat, n_v_heads, cs_actual, d);

            ssm_chunk_scatter_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_xb2, (const float *)d_chunk_out,
                chunk_start, cs_actual, value_dim, head_v_dim, n_v_heads);

            /* State update must come after gather/matvec/output above
             * have all read the PRE-update state for this chunk --
             * stream ordering guarantees that since everything above
             * is submitted first on the same stream. */
            ssm_chunk_state_update_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_state_buf, (const float *)d_v_hat, (const float *)d_chunk_k,
                (const float *)d_cum_g, n_v_heads, repeat, cs_actual, d);

            ok = gpu_ok(gpuGetLastError(), "ssm chunked recurrence chunk");
        }
    }

    ok = ok && gpu_ok(gpuDeviceSynchronize(), "ssm chunked recurrence sync");
    if (ok) {
        ok = gpu_ok(gpuMemcpy(xb2_batch_host, d_xb2, xb2_bytes, gpuMemcpyDeviceToHost), "chunk xb2 d2h");
        ok = ok && gpu_ok(gpuMemcpy(state_host, d_state_buf, state_bytes, gpuMemcpyDeviceToHost), "chunk state d2h");
    }
    return ok;
#endif
}

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

/* Allocate SSM pipeline buffers for hybrid SSM+attention layers.
 * Called from model_load after SSM eligibility passes. */
extern "C" int
picolm_gpu_ssm_pipeline_alloc(int conv_dim, int ssm_d_inner, int n_v_heads, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (ctx->ssm_ready) return 1;

    size_t conv_b = (size_t)conv_dim * sizeof(float);
    size_t inner_b = (size_t)ssm_d_inner * sizeof(float);
    size_t heads_b = (size_t)n_v_heads * sizeof(float);

    int ok = 1;
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_qkv_raw, conv_b), "ssm_qkv_raw alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_conv_out, conv_b), "ssm_conv_out alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_xb2, inner_b), "ssm_xb2 alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_xb2_remap, inner_b), "ssm_xb2_remap alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_v_remap, inner_b), "ssm_v_remap alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_alpha_raw, heads_b), "ssm_alpha_raw alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_beta_raw, heads_b), "ssm_beta_raw alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_gate_exp, heads_b), "ssm_gate_exp alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_beta, heads_b), "ssm_beta alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_output, inner_b), "ssm_output alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_final_output, inner_b), "ssm_final_output alloc");
    if (!ok) return 0;

    ctx->ssm_ready = 1;
    return 1;
}

/* Allocate prefill batch buffers: [max_seq_len][max_stride] for S>1 pipeline
 * xb_stride = max(q_dim, conv_dim, dim) to accommodate all uses:
 * - attention: RMSNorm output + QKV projections (q_dim stride)
 * - SSM prefill: attn_qkv output (conv_dim = 2*d_state*n_k + ssm_d_inner)
 * - residual add: dim stride */
extern "C" int
picolm_gpu_pipeline_batch_alloc(int dim, int q_dim, int kv_dim, int ffn_hidden,
                                 int xb_stride, int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (ctx->pipe_b_ready) return 1;

    size_t bsz = (size_t)max_seq_len;
    size_t db = bsz * dim * sizeof(float);
    size_t qb = bsz * q_dim * sizeof(float);
    size_t xb = bsz * xb_stride * sizeof(float);
    size_t kvb = bsz * kv_dim * sizeof(float);
    size_t fb = bsz * ffn_hidden * sizeof(float);

    int ok = 1;
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_x_b, xb), "pipe_x_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_xb_b, xb), "pipe_xb_b alloc");
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
    /* SSM pipeline buffers */
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!ctx->ssm_ready) continue;
        if (ctx->ssm_qkv_raw) gpuFree(ctx->ssm_qkv_raw);
        if (ctx->ssm_conv_out) gpuFree(ctx->ssm_conv_out);
        if (ctx->ssm_xb2) gpuFree(ctx->ssm_xb2);
        if (ctx->ssm_xb2_remap) gpuFree(ctx->ssm_xb2_remap);
        if (ctx->ssm_v_remap) gpuFree(ctx->ssm_v_remap);
        if (ctx->ssm_alpha_raw) gpuFree(ctx->ssm_alpha_raw);
        if (ctx->ssm_beta_raw) gpuFree(ctx->ssm_beta_raw);
        if (ctx->ssm_gate_exp) gpuFree(ctx->ssm_gate_exp);
        if (ctx->ssm_beta) gpuFree(ctx->ssm_beta);
        if (ctx->ssm_output) gpuFree(ctx->ssm_output);
        if (ctx->ssm_final_output) gpuFree(ctx->ssm_final_output);
        ctx->ssm_qkv_raw = ctx->ssm_conv_out = ctx->ssm_xb2 = ctx->ssm_xb2_remap =
            ctx->ssm_v_remap = ctx->ssm_alpha_raw = ctx->ssm_beta_raw = ctx->ssm_gate_exp =
            ctx->ssm_beta = ctx->ssm_output = ctx->ssm_final_output = NULL;
        ctx->ssm_ready = 0;
    }
}

/* Pre-allocate Q8_0 scratch buffers to a fixed maximum size.
 * This eliminates runtime reallocation races where cudaFree of a
 * scratch buffer races with a kernel still reading from it. */
extern "C" int
picolm_gpu_prealloc_q8(size_t max_xq_bytes, size_t max_xd_bytes, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!ctx->q8_xq) {
        if (!gpu_ok(gpuMalloc(&ctx->q8_xq, max_xq_bytes), "prealloc q8_xq")) return 0;
        ctx->q8_xq_cap = max_xq_bytes;
    }
    if (!ctx->q8_xd) {
        if (!gpu_ok(gpuMalloc(&ctx->q8_xd, max_xd_bytes), "prealloc q8_xd")) return 0;
        ctx->q8_xd_cap = max_xd_bytes;
    }
    return 1;
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

/* SSM pipeline buffer accessors */
extern "C" float *picolm_gpu_ssm_qkv_raw(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_qkv_raw : NULL; }
extern "C" float *picolm_gpu_ssm_conv_out(int device)      { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_conv_out : NULL; }
extern "C" float *picolm_gpu_ssm_xb2(int device)           { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_xb2 : NULL; }
extern "C" float *picolm_gpu_ssm_xb2_remap(int device)     { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_xb2_remap : NULL; }
extern "C" float *picolm_gpu_ssm_v_remap(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_v_remap : NULL; }
extern "C" float *picolm_gpu_ssm_alpha_raw(int device)     { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_alpha_raw : NULL; }
extern "C" float *picolm_gpu_ssm_beta_raw(int device)      { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_beta_raw : NULL; }
extern "C" float *picolm_gpu_ssm_gate_exp(int device)      { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_gate_exp : NULL; }
extern "C" float *picolm_gpu_ssm_beta(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_beta : NULL; }
extern "C" float *picolm_gpu_ssm_output(int device)        { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_output : NULL; }
extern "C" float *picolm_gpu_ssm_final_output(int device)  { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_final_output : NULL; }

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

/* Shared dispatch for both picolm_gpu_attention_decode and _dev: chooses
 * single-pass (grid=n_kv_heads, matching the pre-split-K behavior
 * exactly -- no regression risk for short contexts) vs split-K (grid=
 * n_kv_heads*n_splits + a merge pass) based on how many KV positions
 * there are to walk. xb_dev/q_dev must already be device pointers. */
static int
attn_decode_dispatch(float *xb_dev, const float *q_dev,
                      int layer_ordinal, int pos,
                      int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
                      gpu_device_ctx_t *ctx, int device) {
    int kv_mul = n_heads / n_kv_heads;
    size_t kv_pos_stride_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    size_t kv_head_stride_bytes = head_dim * sizeof(uint16_t);

    int total_kv = pos + 1;
    int n_splits = (total_kv + ATTN_DECODE_MIN_CHUNK - 1) / ATTN_DECODE_MIN_CHUNK;
    if (n_splits > ATTN_DECODE_MAX_SPLITS) n_splits = ATTN_DECODE_MAX_SPLITS;
    if (n_splits < 1) n_splits = 1;

    size_t shared_bytes = 2 * head_dim * sizeof(uint16_t) + 256 * sizeof(float)
                         + (size_t)kv_mul * head_dim * sizeof(float)
                         + (size_t)kv_mul * sizeof(float)
                         + (size_t)kv_mul * sizeof(float)
                         + 2 * sizeof(float); /* rescale_sh + weight_sh */

    if (n_splits <= 1) {
        dim3 grid((unsigned)n_kv_heads, 1, 1);
        picolm_gpu_attention_decode_kernel<<<grid, 256, (unsigned)shared_bytes, ctx->stream>>>(
            xb_dev, q_dev, g_kv_k_dev[device], g_kv_v_dev[device],
            layer_ordinal, pos, n_heads, n_kv_heads, head_dim, max_seq_len,
            kv_pos_stride_bytes, kv_head_stride_bytes);
        if (!gpu_ok(gpuGetLastError(), "attn decode kernel")) return 0;
        return 1;
    }

    int chunk_size = (total_kv + n_splits - 1) / n_splits;
    size_t need = (size_t)n_heads * n_splits * (head_dim + 2) * sizeof(float);
    if (!reserve(&ctx->attn_partial, &ctx->attn_partial_cap, need)) return 0;
    float *partial_max = ctx->attn_partial;
    float *partial_sum = partial_max + (size_t)n_heads * n_splits;
    float *partial_acc = partial_sum + (size_t)n_heads * n_splits;

    dim3 grid_split((unsigned)n_kv_heads, (unsigned)n_splits, 1);
    picolm_gpu_attention_decode_split_kernel<<<grid_split, 256, (unsigned)shared_bytes, ctx->stream>>>(
        partial_max, partial_sum, partial_acc,
        q_dev, g_kv_k_dev[device], g_kv_v_dev[device],
        layer_ordinal, pos, n_heads, n_kv_heads, head_dim, max_seq_len,
        kv_pos_stride_bytes, kv_head_stride_bytes, n_splits, chunk_size);
    if (!gpu_ok(gpuGetLastError(), "attn decode split kernel")) return 0;

    size_t merge_shared = (size_t)kv_mul * 2 * sizeof(float);
    dim3 grid_merge((unsigned)n_kv_heads, 1, 1);
    picolm_gpu_attention_decode_merge_kernel<<<grid_merge, 128, (unsigned)merge_shared, ctx->stream>>>(
        xb_dev, partial_max, partial_sum, partial_acc,
        n_heads, n_kv_heads, head_dim, n_splits);
    if (!gpu_ok(gpuGetLastError(), "attn decode merge kernel")) return 0;
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
    if (head_dim > 256) return 0;

    int kv_mul = n_heads / n_kv_heads;
    if (kv_mul < 1 || kv_mul > 8) return 0;

    size_t x_bytes = (size_t)n_heads * head_dim * sizeof(float);
    size_t y_bytes = x_bytes;
    if (!reserve(&ctx->x, &ctx->x_cap, x_bytes) ||
        !reserve(&ctx->y, &ctx->y_cap, y_bytes)) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, q_host, x_bytes, gpuMemcpyHostToDevice),
                "attn decode Q upload")) return 0;

    if (!attn_decode_dispatch(ctx->y, ctx->x, layer_ordinal, pos,
                              n_heads, n_kv_heads, head_dim, max_seq_len,
                              ctx, device)) return 0;

    if (!gpu_ok(gpuDeviceSynchronize(), "attn decode sync")) return 0;

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

    return attn_decode_dispatch(xb_out_dev, q_dev, layer_ordinal, pos,
                                 n_heads, n_kv_heads, head_dim, max_seq_len,
                                 ctx, device);
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
                                   int dim, float eps, int x_stride) {
    int row = (int)gpuBlockIdx_x;
    int stride = (x_stride > 0) ? x_stride : dim;
    const float *xr = x + (size_t)row * stride;
    float *outr = out + (size_t)row * stride;

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
    float rms = sqrtf(ssum[0] / dim + eps);
    float inv_rms = 1.0f / rms;
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
                                int half_dim, int start_pos, int S,
                                int rope_type) {
    int idx = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int per_row = n_heads * head_dim;
    int total = S * per_row;
    for (int i = idx; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        int row = i / per_row;
        int rem = i % per_row;
        int h = rem / head_dim;
        int d = rem % head_dim;
        float *xr = x + (size_t)row * per_row;
        int pos = start_pos + row;

        if (rope_type) {
            /* Qwen2 interleaved: (x[d], x[d+half_dim]) rotated */
            if (d >= half_dim) continue;
            int h1 = d, h2 = d + half_dim;
            float x1 = xr[h * head_dim + h1];
            float x2 = xr[h * head_dim + h2];
            float c = cos_tbl[(size_t)pos * half_dim + h1];
            float s = sin_tbl[(size_t)pos * half_dim + h1];
            xr[h * head_dim + h1] = x1 * c - x2 * s;
            xr[h * head_dim + h2] = x1 * s + x2 * c;
        } else {
            /* Llama pairwise: (x[2i], x[2i+1]) rotated by cos[i], sin[i] */
            if (d & 1) continue; /* odd index: handled by even partner */
            if (d + 1 >= head_dim) continue; /* safety: beyond head */
            int idx_pair = d >> 1; /* 0,1 -> pair 0; 2,3 -> pair 1; etc. */
            if (idx_pair >= half_dim) continue; /* beyond rope range */
            float x0 = xr[h * head_dim + d];
            float x1v = xr[h * head_dim + d + 1];
            float c = cos_tbl[(size_t)pos * half_dim + idx_pair];
            float s = sin_tbl[(size_t)pos * half_dim + idx_pair];
            xr[h * head_dim + d]     = x0 * c - x1v * s;
            xr[h * head_dim + d + 1] = x0 * s + x1v * c;
        }
    }
}

/* RoPE kernel: applies pairwise rotary position embedding */
__global__ void
picolm_gpu_rope_kernel(float *x, int n_heads, int head_dim,
                        const float *cos_tbl, const float *sin_tbl,
                        int half_dim, int rope_type) {
    int idx = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int total = n_heads * head_dim;
    for (int i = idx; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        int h = i / head_dim;
        int d = i % head_dim;

        if (rope_type) {
            /* Qwen2 interleaved: (x[d], x[d+half_dim]) rotated */
            if (d >= half_dim) continue;
            int h1 = d, h2 = d + half_dim;
            float x1 = x[h * head_dim + h1];
            float x2 = x[h * head_dim + h2];
            float c = cos_tbl[h1];
            float s = sin_tbl[h1];
            x[h * head_dim + h1] = x1 * c - x2 * s;
            x[h * head_dim + h2] = x1 * s + x2 * c;
        } else {
            /* Llama pairwise: (x[2i], x[2i+1]) rotated by cos[i], sin[i] */
            if (d & 1) continue; /* odd: handled by even partner */
            if (d + 1 >= head_dim) continue;
            int idx_pair = d >> 1;
            if (idx_pair >= half_dim) continue;
            float x0 = x[h * head_dim + d];
            float x1v = x[h * head_dim + d + 1];
            float c = cos_tbl[idx_pair];
            float s = sin_tbl[idx_pair];
            x[h * head_dim + d]     = x0 * c - x1v * s;
            x[h * head_dim + d + 1] = x0 * s + x1v * c;
        }
    }
}

/* Residual add kernel: out[i] = a[i] + b[i] */
__global__ void
picolm_gpu_residual_add_kernel(float *out, const float *a, const float *b, int n, int dim, int stride) {
    int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int total = n * dim;
    for (; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        int tok = i / dim;
        int off = i % dim;
        out[tok * stride + off] = a[tok * stride + off] + b[tok * stride + off];
    }
}

/* Q+gate de-interleave kernel for SSM attention layers.
 * GGUF stores [Q_0, Gate_0, Q_1, Gate_1, ...] per head, each head_dim floats.
 * Reads raw[head * 2 * head_dim + ...], writes Q to out_q[head * head_dim + ...]
 * and Gate to out_g[head * head_dim + ...].
 * One thread per head, thread-0 of each block does the memmove-like copy. */
__global__ void
picolm_gpu_qg_deinterleave_kernel(const float *raw, float *out_q, float *out_g,
                                   int n_heads, int head_dim) {
    int h = gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (h >= n_heads) return;
    const float *src = raw + (size_t)h * 2 * head_dim;
    float *dst_q = out_q + (size_t)h * head_dim;
    float *dst_g = out_g + (size_t)h * head_dim;
    for (int d = 0; d < head_dim; d++) {
        dst_q[d] = src[d];           /* Q portion */
        dst_g[d] = src[head_dim + d]; /* Gate portion */
    }
}

/* Elementwise sigmoid-multiply kernel: out[i] = a[i] * sigmoid(g[i])
 * For SSM attention: pipe_attn_out *= sigmoid(gate) before output projection. */
__global__ void
picolm_gpu_sigmoid_mul_kernel(float *out, const float *gate, int n) {
    int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    for (; i < n; i += (int)gridDim.x * gpuBlockDim_x) {
        float g = gate[i];
        float sg = (g > 20.0f) ? 1.0f : (g < -20.0f) ? 0.0f : 1.0f / (1.0f + expf(-g));
        out[i] *= sg;
    }
}

/* Batched Q+gate de-interleave: processes S sequences, each with n_heads heads.
 * raw is [S][n_heads * 2 * head_dim], out_q is [S][n_heads * head_dim],
 * out_g is [S][n_heads * head_dim]. */
__global__ void
picolm_gpu_qg_deinterleave_batched_kernel(const float *raw, float *out_q,
                                           float *out_g, int n_heads,
                                           int head_dim, int S) {
    int idx = gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    int total = n_heads * S;
    if (idx >= total) return;
    int s = idx / n_heads;
    int h = idx % n_heads;
    size_t row_stride = (size_t)n_heads * 2 * head_dim;
    size_t q_stride = (size_t)n_heads * head_dim;
    const float *src = raw + (size_t)s * row_stride + (size_t)h * 2 * head_dim;
    float *dst_q = out_q + (size_t)s * q_stride + (size_t)h * head_dim;
    float *dst_g = out_g + (size_t)s * q_stride + (size_t)h * head_dim;
    for (int d = 0; d < head_dim; d++) {
        dst_q[d] = src[d];
        dst_g[d] = src[head_dim + d];
    }
}

/* Batched sigmoid-multiply: processes S sequences, each n elements. */
__global__ void
picolm_gpu_sigmoid_mul_batched_kernel(float *out, const float *gate,
                                       int n, int S) {
    int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int total = n * S;
    for (; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        float g = gate[i];
        float sg = (g > 20.0f) ? 1.0f : (g < -20.0f) ? 0.0f : 1.0f / (1.0f + expf(-g));
        out[i] *= sg;
    }
}

/* Phase 2 host API */
/* Device-native rmsnorm: all pointers are device-resident, no H2D/D2H, no sync.
 * Used from model_forward_gpu() pipeline path. */
extern "C" int
picolm_gpu_rmsnorm_dev(float *out, const float *x, const float *weight,
                        int dim, float eps, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_kernel<<<1, n_threads, 0, ctx->stream>>>(
        out, x, weight, dim, eps);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm dev kernel")) return 0;
    return 1;
}

/* Host-side rmsnorm: takes host pointers, does H2D/D2H/sync.
 * Used from ssm_forward() QK-norm path and other host-side callers. */
extern "C" int
picolm_gpu_rmsnorm(float *out, const float *x, const float *weight,
                    int dim, float eps, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    size_t xb = dim * sizeof(float);
    size_t wb = dim * sizeof(float);
    void *dx, *dw, *do_;
    if (!gpu_ok(gpuMalloc(&dx, xb), "rmsnorm x") ||
        !gpu_ok(gpuMalloc(&dw, wb), "rmsnorm w") ||
        !gpu_ok(gpuMalloc(&do_, xb), "rmsnorm o")) return 0;
    if (!gpu_ok(gpuMemcpy(dx, x, xb, gpuMemcpyHostToDevice), "rmsnorm x h2d") ||
        !gpu_ok(gpuMemcpy(dw, weight, wb, gpuMemcpyHostToDevice), "rmsnorm w h2d")) {
        gpuFree(dx); gpuFree(dw); gpuFree(do_); return 0;
    }
    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_kernel<<<1, n_threads, 0, ctx->stream>>>(
        (float *)do_, (const float *)dx, (const float *)dw, dim, eps);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm kernel") ||
        !gpu_ok(gpuDeviceSynchronize(), "rmsnorm sync")) {
        gpuFree(dx); gpuFree(dw); gpuFree(do_); return 0;
    }
    gpuMemcpy(out, do_, xb, gpuMemcpyDeviceToHost);
    gpuFree(dx); gpuFree(dw); gpuFree(do_);
    return 1;
}

/* Device-native batched rmsnorm: all pointers device-resident, no H2D/D2H/sync.
 * Used from model_forward_gpu() / model_forward_prefill_gpu() pipeline paths. */
extern "C" int
picolm_gpu_rmsnorm_batched_dev(float *out, const float *x, const float *weight,
                                int dim, float eps, int S, int x_stride, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;
    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_batched_kernel<<<S, n_threads, 0, ctx->stream>>>(
        out, x, weight, dim, eps, x_stride);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm batched dev kernel")) return 0;
    return 1;
}

/* Host-side batched rmsnorm: takes host pointers, does H2D/D2H/sync.
 /* Used from ssm_forward() QK-norm path and prefill attention QK-norm.
 * Launches the kernel directly on ctx->stream.
 * x and out must be device-accessible.
 * weight: if it's a device pointer (from gpu_upload_f32), used directly.
 *   If it's a host pointer (from dequantize_row), uploaded once and cached. */
extern "C" int
picolm_gpu_rmsnorm_batched(float *out, const float *x, const float *weight,
                            int dim, float eps, int S, int x_stride, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;

    const void *w_dev = weight;
    gpu_mutex_lock();
    int cached = 0;
    for (int i = 0; i < ctx->rmsnorm_w_n; i++) {
        if (ctx->rmsnorm_w_keys[i] == weight) { w_dev = ctx->rmsnorm_w_dev[i]; cached = 1; break; }
    }
    if (!cached) {
        if ((uintptr_t)weight < 1024ull * 1024 * 1024) {
            /* Host pointer: upload to device and cache */
            size_t wb = dim * sizeof(float);
            if (ctx->rmsnorm_w_n >= 64) { gpu_mutex_unlock(); return 0; }
            void *dw = NULL;
            if (!gpu_ok(gpuMalloc(&dw, wb), "rmsnorm_b w")) { gpu_mutex_unlock(); return 0; }
            gpuDeviceSynchronize();
            if (!gpu_ok(gpuMemcpy(dw, weight, wb, gpuMemcpyHostToDevice), "rmsnorm_b w h2d")) {
                gpuFree(dw); gpu_mutex_unlock(); return 0;
            }
            ctx->rmsnorm_w_keys[ctx->rmsnorm_w_n] = weight;
            ctx->rmsnorm_w_dev[ctx->rmsnorm_w_n] = dw;
            ctx->rmsnorm_w_n++;
            w_dev = dw;
        }
    }
    gpu_mutex_unlock();

    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_batched_kernel<<<S, n_threads, 0, ctx->stream>>>(
        out, x, (const float *)w_dev, dim, eps, x_stride);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm batched kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_rope_apply(float *x, int n_heads, int head_dim,
                       const float *cos_tbl, const float *sin_tbl,
                       int half_dim, int rope_type, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    int total = n_heads * head_dim;
    int n_threads = 128;
    int n_blocks = min((total + n_threads - 1) / n_threads, 128);
    picolm_gpu_rope_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        x, n_heads, head_dim, cos_tbl, sin_tbl, half_dim, rope_type);
    if (!gpu_ok(gpuGetLastError(), "rope kernel")) return 0;
    return 1;
}

/* Batched RoPE: cos_tbl_base/sin_tbl_base are the UNOFFSET [max_seq_len][half_dim]
 * base pointers. Each row computes its own position as start_pos + row. */
extern "C" int
picolm_gpu_rope_apply_batched(float *x, int n_heads, int head_dim,
                               const float *cos_tbl_base, const float *sin_tbl_base,
                               int half_dim, int start_pos, int S,
                               int rope_type, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;

    int total = S * n_heads * head_dim;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 4096);
    picolm_gpu_rope_batched_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        x, n_heads, head_dim, cos_tbl_base, sin_tbl_base, half_dim, start_pos, S, rope_type);
    if (!gpu_ok(gpuGetLastError(), "rope batched kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_residual_add(float *out, const float *a, const float *b,
                         int n, int dim, int stride, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    int n_threads = 256;
    int n_blocks = min((n * dim + n_threads - 1) / n_threads, 256);
    picolm_gpu_residual_add_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        out, a, b, n, dim, stride);
    if (!gpu_ok(gpuGetLastError(), "residual_add kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_qg_deinterleave_dev(const float *raw_dev, float *out_q_dev,
                                float *out_g_dev, int n_heads, int head_dim,
                                int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int n_threads = 256;
    int n_blocks = min((n_heads + n_threads - 1) / n_threads, 256);
    picolm_gpu_qg_deinterleave_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        raw_dev, out_q_dev, out_g_dev, n_heads, head_dim);
    if (!gpu_ok(gpuGetLastError(), "qg_deinterleave kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_sigmoid_mul_dev(float *out_dev, const float *gate_dev,
                            int n, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int n_threads = 256;
    int n_blocks = min((n + n_threads - 1) / n_threads, 256);
    picolm_gpu_sigmoid_mul_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        out_dev, gate_dev, n);
    if (!gpu_ok(gpuGetLastError(), "sigmoid_mul kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_qg_deinterleave_batched_dev(const float *raw_dev, float *out_q_dev,
                                        float *out_g_dev, int n_heads,
                                        int head_dim, int S, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int total = n_heads * S;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 256);
    picolm_gpu_qg_deinterleave_batched_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        raw_dev, out_q_dev, out_g_dev, n_heads, head_dim, S);
    if (!gpu_ok(gpuGetLastError(), "qg_deinterleave_batched kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_sigmoid_mul_batched_dev(float *out_dev, const float *gate_dev,
                                   int n, int S, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int total = n * S;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 256);
    picolm_gpu_sigmoid_mul_batched_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        out_dev, gate_dev, n, S);
    if (!gpu_ok(gpuGetLastError(), "sigmoid_mul_batched kernel")) return 0;
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
 * dir: 1 = H2D, -1 = D2H, 0 = D2D. Returns 1 on success.
 * Uses ctx->stream for the actual transfer to avoid multi-engine
 * reordering with default stream on sm_121 (GB10/CUDA 13). */
extern "C" int
picolm_gpu_memcpy(void *dst, const void *src, size_t bytes, int dir, int device) {
    if (!dst || !src || bytes < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    cudaMemcpyKind kind = (dir > 0) ? gpuMemcpyHostToDevice :
                           (dir < 0) ? gpuMemcpyDeviceToHost :
                                       gpuMemcpyDeviceToDevice;

    /* Use async copy on ctx->stream + sync to avoid default-stream
     * interleaving on GB10 multi-engine scheduler. */
    gpuError_t err = gpuMemcpyAsync(dst, src, bytes, kind, ctx->stream);
    if (!gpu_ok(err, "pipeline memcpy async")) return 0;
    return gpu_ok(gpuDeviceSynchronize(), "pipeline memcpy sync");
}
int picolm_gpu_ssm_conv1d_batch_dev(float *od, float *sd, const float *id, const float *wd,
    int cd, int dc, int nt, int dev, int stride) {
    if(cd<1||dc<1||nt<1) return 0;
    if(dc>PICOLM_SSM_CONV_MAX_D_CONV) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    int nb=(cd+255)/256;
    picolm_gpu_ssm_conv1d_batch_kernel<<<nb,256,0,ctx->stream>>>(od,sd,id,wd,cd,dc,nt,stride);
    return gpu_ok(gpuGetLastError(),"conv1d batch dev");
}

int picolm_gpu_ssm_l2norm_batch_dev(float *xd, int hd, int nh, int nt, int ts, float eps, float es, int dev) {
    if(hd<1||nh<1||hd>256||nt<1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    picolm_gpu_ssm_l2norm_batch_kernel<<<dim3((unsigned)nh,(unsigned)nt),min(hd,256),0,ctx->stream>>>(xd,hd,nh,nt,ts,eps,es);
    return gpu_ok(gpuGetLastError(),"l2norm batch dev");
}

int picolm_gpu_ssm_vecdot_batch_dev(float *od, const float *xd, const void *wd, gguf_type_t qt,
    int dim, int nvh, int nt, int rb, const int *hm, int dev, int out_stride) {
    if(nvh<=0||dim<=0||nt<=0) return 0;
    if(dim>PICOLM_SSM_VECDOT_MAX_DIM) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    picolm_ssm_vecdot_batch_kernel<<<dim3((unsigned)nvh,(unsigned)nt),256,0,ctx->stream>>>(od,xd,wd,qt,dim,nvh,nt,rb,hm,out_stride);
    return gpu_ok(gpuGetLastError(),"vecdot batch dev");
}

int picolm_gpu_ssm_prefill_gated_norm_dev(float *od, const float *zd, const float *nd,
    int hd, int nh, int nt, float eps, int so_stride, int z_stride, int dev) {
    if(hd<1||nh<1||nt<1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    picolm_gpu_ssm_prefill_gated_norm_kernel<<<dim3((unsigned)nh,(unsigned)nt),min(hd,256),0,ctx->stream>>>(od,zd,nd,hd,nh,nt,eps,so_stride,z_stride);
    return gpu_ok(gpuGetLastError(),"gated norm dev");
}

__global__ void
picolm_gpu_ssm_head_permute_batch_kernel(float *dst, const float *src,
                                          const int *head_map,
                                          int head_dim, int n_heads, int n_tokens,
                                          int src_stride, int dst_stride) {
    int h = blockIdx.x; int t = blockIdx.y;
    if (h >= n_heads || t >= n_tokens) return;
    int gh = head_map ? head_map[h] : h;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        dst[(size_t)t * dst_stride + h * head_dim + d] = src[(size_t)t * src_stride + gh * head_dim + d];
}

int picolm_gpu_ssm_head_permute_batch_dev(float *dd, const float *sd, const int *hm,
    int hd, int nh, int nt, int ss, int ds, int dev) {
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_gpu_ssm_head_permute_batch_kernel<<<dim3((unsigned)nh,(unsigned)nt),128,0,ctx->stream>>>(dd,sd,hm,hd,nh,nt,ss,ds);
    return gpu_ok(gpuGetLastError(),"head permute batch dev");
}

__global__ void
/* Gate/beta post-process: outputs gate_log (log-space gate value) directly.
 * Previously this kernel did expf(gate) and the recurrence did logf(expf(gate))
 * -- a useless exp/log round-trip that compounded floating-point error.
 * Now it outputs gate_log = softplus(alpha + dt_w) * a_w, matching the CPU path. */
picolm_ssm_gate_beta_batch_kernel(float *ge, float *be,
    const float *ai, const float *bi, const float *dw, const float *aw,
    int nvh, int nt, int stride) {
    int h = blockIdx.x, t = blockIdx.y;
    if (h >= nvh || t >= nt) return;
    if (threadIdx.x != 0) return;
    int s = stride > 0 ? stride : nvh;
    float a = ai[t*s+h] + dw[h];
    float sp = (a>20.0f)?a:(a<-20.0f)?expf(a):logf(1.0f+expf(a));
    ge[t*s+h] = sp * aw[h];
    be[t*s+h] = 1.0f/(1.0f+expf(-bi[t*s+h]));
}

int picolm_gpu_ssm_gate_beta_batch_dev(float *ge, float *be, const float *ai, const float *bi,
    const float *dw, const float *aw, int nvh, int nt, int dev, int stride) {
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_ssm_gate_beta_batch_kernel<<<dim3((unsigned)nvh,(unsigned)nt),1,0,ctx->stream>>>(ge,be,ai,bi,dw,aw,nvh,nt,stride);
    return gpu_ok(gpuGetLastError(),"gate beta batch dev");
}

int picolm_gpu_expert_mlp_dev(picolm_gpu_tensor_t *g, picolm_gpu_tensor_t *u, picolm_gpu_tensor_t *d,
    float *yd, const float *xd, int S, int x_stride, int y_stride, int dev) {
    if(!g||!u||!d||!xd||!yd||S<1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    int D=g->I, I=g->O;
    if(!reserve(&ctx->gate,&ctx->gate_cap,(size_t)S*I*sizeof(float))||
       !reserve(&ctx->up,&ctx->up_cap,(size_t)S*I*sizeof(float))) return 0;

    /* Q8_0 fast path: quantize F32 input to Q8_0, then int8 MAC */
    int d_blocks = D / 32;
    int i_blocks = I / 32;
    if (d_blocks < 1 || i_blocks < 1) return 0;

    size_t xq_d = (size_t)S * D;
    size_t xd_d = (size_t)S * d_blocks * sizeof(float);
    size_t xq_i = (size_t)S * I;
    size_t xd_i = (size_t)S * i_blocks * sizeof(float);
    if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_d > xq_i ? xq_d : xq_i) ||
        !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_d > xd_i ? xd_d : xd_i)) return 0;

    /* Quantize F32 input [D*S] to Q8_0 */
    if (x_stride > 0 && x_stride != D) {
        picolm_quantize_q8_0_strided<<<dim3((unsigned)d_blocks,(unsigned)S),32,32*sizeof(float),ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, xd, D, S, x_stride);
    } else {
        picolm_quantize_q8_0<<<dim3((unsigned)d_blocks,(unsigned)S),32,32*sizeof(float),ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, xd, D, S);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 quantize input (dev)")) return 0;

    /* gate = q8_q8_tiled(input_q8, gate_weights) */
    int use_tiled = (S > 1 && g->row_bytes + 2048 <= 49152);
    if (use_tiled) {
        dim3 grid((unsigned)I, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid,256,(unsigned)g->row_bytes,ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, g->weights, S, D, I, (int)g->row_bytes, I);
    } else {
        picolm_q8_q8_matmul<<<dim3((unsigned)I,(unsigned)S),256,0,ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, g->weights, S, D, I, (int)g->row_bytes, I);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 gate (dev)")) return 0;

    /* up = q8_q8_tiled(input_q8, up_weights) */
    if (use_tiled) {
        dim3 grid((unsigned)I, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid,256,(unsigned)u->row_bytes,ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, u->weights, S, D, I, (int)u->row_bytes, I);
    } else {
        picolm_q8_q8_matmul<<<dim3((unsigned)I,(unsigned)S),256,0,ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, u->weights, S, D, I, (int)u->row_bytes, I);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 up (dev)")) return 0;

    /* silu(gate) * up -> ctx->gate */
    picolm_silu_mul<<<(unsigned)((S*I+255)/256),256,0,ctx->stream>>>(ctx->gate, ctx->up, S*I);
    if (!gpu_ok(gpuGetLastError(), "expert silu (dev)")) return 0;

    /* Quantize F32 hidden [I*S] to Q8_0 for down */
    picolm_quantize_q8_0<<<dim3((unsigned)i_blocks,(unsigned)S),32,32*sizeof(float),ctx->stream>>>(
        ctx->q8_xq, ctx->q8_xd, ctx->gate, I, S);
    if (!gpu_ok(gpuGetLastError(), "expert q8 quantize hidden (dev)")) return 0;

    /* y = q8_q8_tiled(hidden_q8, down_weights) */
    int use_tiled_down = (S > 1 && d->row_bytes + 2048 <= 49152);
    int ys = y_stride > 0 ? y_stride : D;
    if (use_tiled_down) {
        dim3 grid((unsigned)D, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid,256,(unsigned)d->row_bytes,ctx->stream>>>(
            yd, ctx->q8_xq, ctx->q8_xd, d->weights, S, I, D, (int)d->row_bytes, ys);
    } else {
        picolm_q8_q8_matmul<<<dim3((unsigned)D,(unsigned)S),256,0,ctx->stream>>>(
            yd, ctx->q8_xq, ctx->q8_xd, d->weights, S, I, D, (int)d->row_bytes, ys);
    }
    return gpu_ok(gpuGetLastError(), "expert MLP dev");
}

int picolm_gpu_ssm_chunked_recurrence_dev(const float *conv_dev, const float *alpha_dev,
    const float *beta_dev, float *state_dev, float *xb2_dev,
    int n_tokens, int value_dim, int xb2_stride,
    int d_state, int n_k_heads, int n_v_heads,
    int head_v_dim, int repeat, int conv_dim, int cs, int device) {
    (void)conv_dim; /* conv_dim is used for offset calculations, xb2_stride for striding */
#ifndef PICOLM_SSM_CHUNKED_GPU_VALIDATED
    (void)conv_dev;(void)alpha_dev;(void)beta_dev;(void)state_dev;(void)xb2_dev;
    (void)n_tokens;(void)value_dim;(void)xb2_stride;(void)d_state;(void)n_k_heads;
    (void)n_v_heads;(void)head_v_dim;(void)repeat;(void)cs;(void)device;
    return 0;
#else
    if(d_state!=head_v_dim) return 0;
    if(cs<=0) cs=64; if(cs>n_tokens) cs=n_tokens;
    int nc=(n_tokens+cs-1)/cs; if(nc<1) nc=1;
    int d=d_state, qk_dim=d_state*n_k_heads;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if(!ctx||!select_ctx(ctx)) return 0;
    int n_threads=256;
    /* Use per-buffer allocations identical to host-facing driver,
     * eliminating single-buffer offset arithmetic entirely. */
    float *d_cq=NULL,*d_ck=NULL,*d_cv=NULL,*d_cb=NULL;
    float *d_gl=NULL,*d_cg=NULL,*d_qd=NULL,*d_dm=NULL;
    float *d_M=NULL,*d_kq=NULL,*d_ve=NULL,*d_vh=NULL;
    float *d_sk=NULL,*d_sq=NULL,*d_co=NULL;
    size_t caps[15]={0};
    size_t szs[15]={
        (size_t)n_k_heads*cs*d*sizeof(float),  /* cq */
        (size_t)n_k_heads*cs*d*sizeof(float),  /* ck */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* cv */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* cb */
        (size_t)n_v_heads*cs*sizeof(float),    /* gl */
        (size_t)n_v_heads*cs*sizeof(float),    /* cg */
        (size_t)n_v_heads*cs*sizeof(float),    /* qd */
        (size_t)n_v_heads*cs*cs*sizeof(float), /* dm */
        (size_t)n_v_heads*cs*cs*sizeof(float), /* M */
        (size_t)n_v_heads*cs*cs*sizeof(float), /* kq */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* ve */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* vh */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* sk */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* sq */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* co */
    };
    float **ptrs[15]={&d_cq,&d_ck,&d_cv,&d_cb,&d_gl,&d_cg,&d_qd,&d_dm,
        &d_M,&d_kq,&d_ve,&d_vh,&d_sk,&d_sq,&d_co};
    int ok=1;
    for(int i=0;i<15&&ok;i++){
        ok&=ssm_batch_scratch_ensure((void**)ptrs[i],&caps[i],szs[i]);
        if(ok) ok&=gpu_ok(cudaMemsetAsync(*(void**)ptrs[i],0,szs[i],ctx->stream),"scratch zero");
    }
    if(!ok) return 0;
    for(int ci=0;ci<nc&&ok;ci++){
      int ca=(ci==nc-1)?(n_tokens-ci*cs):cs; if(ca<=0)break;
      int co2=ci*cs;
      ssm_chunk_gather_qk_kernel<<<n_k_heads,n_threads,0,ctx->stream>>>((float*)d_cq,(float*)d_ck,conv_dev,co2,ca,xb2_stride,qk_dim,d_state,n_k_heads);
      if(!gpu_ok(gpuGetLastError(),"gather_qk"))return 0;
      ssm_chunk_gather_v_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_cv,(float*)d_cb,(float*)d_gl,conv_dev,alpha_dev,beta_dev,co2,ca,xb2_stride,qk_dim,head_v_dim,n_v_heads,xb2_stride);
      if(!gpu_ok(gpuGetLastError(),"gather_v"))return 0;
      ssm_chunk_decay_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_cg,(float*)d_qd,(float*)d_dm,(float*)d_gl,n_v_heads,ca,ca);
      if(!gpu_ok(gpuGetLastError(),"decay"))return 0;
#ifdef PICOLM_GPU
      if (getenv("PICOLM_SSM_STEP_VERIFY") && ci == 0) {
          /* D2H decay_mask and gate_log scratch */
          float _dm[16], _gl[40], _cg[40];
          cudaDeviceSynchronize();
          cudaMemcpy(_dm, d_dm, 64, cudaMemcpyDeviceToHost);
          cudaMemcpy(_gl, d_gl, 160, cudaMemcpyDeviceToHost);
          cudaMemcpy(_cg, d_cg, 160, cudaMemcpyDeviceToHost);
          float _dm_full[1300];
          cudaMemcpy(_dm_full, d_dm, 5200, cudaMemcpyDeviceToHost);
          fprintf(stderr, "[STEP l0] dm row0=[%.6f %.6f %.6f %.6f] row1=[%.6f %.6f %.6f %.6f] dm35_0=%.6f dm35_35=%.6f cg0=%.6f cg35=%.6f\n",
              _dm_full[0],_dm_full[1],_dm_full[2],_dm_full[3],
              _dm_full[36],_dm_full[37],_dm_full[38],_dm_full[39],
              _dm_full[35*ca+0], _dm_full[35*ca+35], _cg[0], _cg[35]);
      }
#endif
      ssm_chunk_masked_gemm_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_M,(const float*)d_ck,(const float*)d_ck,(const float*)d_dm,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"M_gemm"))return 0;
      ssm_chunk_matvec_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_sk,state_dev,(const float*)d_ck,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"sk_matvec"))return 0;
      ssm_chunk_veff_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_ve,(const float*)d_cv,(const float*)d_sk,(const float*)d_qd,(const float*)d_cb,n_v_heads,ca,d);
      if(!gpu_ok(gpuGetLastError(),"veff"))return 0;
      {int tr=d<1024?d:1024; ssm_chunk_trisolve_kernel<<<n_v_heads,tr,0,ctx->stream>>>((float*)d_vh,(const float*)d_ve,(const float*)d_M,(const float*)d_cb,n_v_heads,ca,d);}
      if(!gpu_ok(gpuGetLastError(),"trisolve"))return 0;
      ssm_chunk_matvec_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_sq,state_dev,(const float*)d_cq,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"sq_matvec"))return 0;
      ssm_chunk_masked_gemm_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_kq,(const float*)d_cq,(const float*)d_ck,(const float*)d_dm,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"kq_gemm"))return 0;
      ssm_chunk_output_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_co,(const float*)d_sq,(const float*)d_qd,(const float*)d_kq,(const float*)d_vh,n_v_heads,ca,d);
      if(!gpu_ok(gpuGetLastError(),"output"))return 0;
#ifdef PICOLM_GPU
      if (getenv("PICOLM_SSM_STEP_VERIFY") && ci == 0) {
          /* D2H v_eff[0][0..3] and v_hat[0][0..3] for head 0 */
          float _ve[4], _vh[4], _co[4];
          cudaDeviceSynchronize();
          cudaMemcpy(_ve, d_ve, 16, cudaMemcpyDeviceToHost);
          cudaMemcpy(_vh, d_vh, 16, cudaMemcpyDeviceToHost);
          cudaMemcpy(_co, d_co, 16, cudaMemcpyDeviceToHost);
          fprintf(stderr, "[STEP l0] GPU ve[0][0..3]={%.6f,%.6f,%.6f,%.6f} vh[0][0..3]={%.6f,%.6f,%.6f,%.6f} co[0][0..3]={%.6f,%.6f,%.6f,%.6f}\n",
              _ve[0],_ve[1],_ve[2],_ve[3], _vh[0],_vh[1],_vh[2],_vh[3], _co[0],_co[1],_co[2],_co[3]);
      }
#endif
      ssm_chunk_scatter_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>(xb2_dev,(const float*)d_co,co2,ca,xb2_stride,head_v_dim,n_v_heads);
      if(!gpu_ok(gpuGetLastError(),"scatter"))return 0;
      ssm_chunk_state_update_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>(state_dev,(const float*)d_vh,(const float*)d_ck,(const float*)d_cg,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"state_up"))return 0;
    }
    return ok;
#endif
}
