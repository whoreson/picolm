// backend_metal.mm
//
// Apple Metal backend for PicoLM  implements the backend_gpu.h C ABI using the
// Metal framework directly (no MLX-C, no CMake, no external dependencies beyond
// what ships with macOS + the Command Line Tools).
//
// Build:  make metal      (clang++ -std=c++20 -fobjc-arc -framework Metal ...)
//
// Design mirrors backend_gpu.cu (CUDA/HIP): one device context, grow-only
// scratch buffers, idempotent tensor upload, per-tensor MTLBuffer handles.
// The host side implements backend_gpu.h unchanged; model.c / tensor.c /
// model.h are NOT modified  they were already written generically behind
// #ifdef PICOLM_GPU.
//
// The matmul kernels are hand-written Metal Shading Language (compiled from a
// source string at init), NOT direct ports of the CUDA device kernels. The
// device-side dequant is taken from PicoLM's CPU reference in quant.c (the
// source of truth for GGUF block layouts  the CUDA file's Q4_K path is
// unfinished and its Q4_0 nibble order is wrong). The high-traffic types use
// coalesced warp-cooperative loads + simd_sum reduction.
//
// Activation: same as CUDA  set PICOLM_GPU=1 in the environment. The whole
// GPU path is gated by -DPICOLM_GPU=1, added by the `metal` Makefile target.
//
// Supported quants: F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K. Weights are mapped
// zero-copy (page-aligned newBufferWithBytesNoCopy) where possible so large
// models don't double resident memory on Apple Silicon's unified memory.

#include "backend_gpu.h"   // includes quant.h (gguf_type_t enum); extern "C" ABI

#import <metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>       // getpagesize()
#include <time.h>        // clock_gettime (optional profiling)

#define TPB 256             // threads per threadgroup for all matmul kernels

// ---- fp16 -> fp32 device helper (matches quant.c semantics) ----
// as_type<half>(u16) reinterprets the 16 bits; Metal has native half<->float.

#pragma mark - Metal Shading Language kernel source
//
// One source string, compiled once into one MTLLibrary. Seven matmul kernels
// (one per supported GGUF type: F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K) + one
// silu_mul elementwise kernel.
//
// Launch shape (all matmul kernels):
//   grid = (O, S, 1)          -> one threadgroup per (output_row, sample)
//   threadgroup = (TPB,1,1)   -> 256 threads = 8 simdgroups (warps) of 32 lanes
//   buffers: 0=w, 1=x(float), 2=y(float); constants: 3=I, 4=S, 5=O
//
// Reduction: each thread holds a partial dot-product accumulator. We reduce
// within each 32-lane simdgroup with the hardware simd_sum (one op), spill the
// 8 per-simdgroup sums to shared memory, barrier once, and simd_sum again in
// simdgroup 0. This replaces the old 8-barrier shared-memory tree.
//
// Coalescing (the big decode win): for the high-traffic types the 32 lanes of a
// simdgroup read CONTIGUOUS weight bytes, so each load is one coalesced 32- or
// 128-byte transaction instead of 32 scattered single-byte loads:
//   - Q4_K: 1 block (144B, 256 values) per simdgroup; each lane reads 4 qs bytes
//     (128B total, perfectly coalesced) and owns 8 values. Thread->value index
//     derived from quant.c dequantize_row_q4_K and hand-verified.
//   - Q8_0: 1 block (34B, 32 values) per simdgroup; each lane reads 1 int8.
//   - F32/F16: float4 / half4 vector loads, lanes stride by TPB*4.
// Q4_0/Q5_K keep a per-element structure (their layouts are awkward to
// coalesce) but cache block metadata once and use simd_sum. Q6_K is ALSO
// coalesced (1 block/simdgroup, 8 values/lane)  see mm_q6_K below.
//
// IMPORTANT: dequant layouts match PicoLM's CPU dequantize_row_* in quant.c
// (the GGUF/llama.cpp convention). For Q4_0: value j -> byte (j&15), low nibble
// if j<16 else high nibble.

static NSString* KERNEL_SOURCE = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
\n\
inline float f16tof32(uint16_t h) { return (float)as_type<half>(h); }\n\
\n\
#define TPB 256\n\
#define NW (TPB / 32)          // simdgroups per threadgroup (8)\n\
\n\
/* Reduce a per-thread accumulator across the whole threadgroup (256 lanes).\n\
 * simd_sum within each 32-lane simdgroup, then one shared-memory hop + simd_sum\n\
 * across the NW simdgroups. Must be called by ALL threads (contains a barrier).\n\
 * Result is valid everywhere; caller writes it from tid==0. */\n\
inline float reduce_all(float acc, threadgroup float* wb, uint tid) {\n\
    uint warp = tid >> 5, lane = tid & 31;\n\
    acc = simd_sum(acc);\n\
    if (lane == 0) wb[warp] = acc;\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    float v = (warp == 0 && lane < NW) ? wb[lane] : 0.0f;\n\
    return simd_sum(v);\n\
}\n\
\n\
/* Unpack one Q4_K/Q5_K sub-block scale+min (6-bit packed in scales[12]) into\n\
 * d*scale and dmin*min, exactly matching quant.c get_scale_min_k4. */\n\
inline void k4_unpack(device const uint8_t* s, int is,\n\
                      thread float& dsc, thread float& dmn,\n\
                      float d, float dmin) {\n\
    uint8_t sc, mn;\n\
    if (is < 4) { sc = s[is] & 63; mn = s[is + 4] & 63; }\n\
    else { sc = (s[is + 4] & 0xF) | ((s[is - 4] >> 6) << 4);\n\
           mn = (s[is + 4] >> 4)  | ((s[is]     >> 6) << 4); }\n\
    dsc = d * (float)sc;\n\
    dmn = dmin * (float)mn;\n\
}\n\
\n\
/* ---------------- F32 (coalesced float4) ---------------- */\n\
kernel void mm_f32(device const float* w [[buffer(0)]],\n\
                   device const float* x [[buffer(1)]],\n\
                   device float*       y [[buffer(2)]],\n\
                   constant int& I [[buffer(3)]],\n\
                   constant int& S [[buffer(4)]],\n\
                   constant int& O [[buffer(5)]],\n\
                   uint tid [[thread_index_in_threadgroup]],\n\
                   uint2 gp [[threadgroup_position_in_grid]]) {\n\
    int o = (int)gp.x, s = (int)gp.y;\n\
    device const float* wrow = w + (unsigned long)o * (unsigned long)I;\n\
    float acc = 0.0f;\n\
    for (int i = (int)tid * 4; i + 4 <= I; i += TPB * 4) {\n\
        float4 wv = *(device const float4*)(wrow + i);\n\
        float4 xv = *(device const float4*)(x + (unsigned long)s * I + i);\n\
        acc += dot(wv, xv);\n\
    }\n\
    for (int i = (I / 4) * 4 + (int)tid; i < I; i += TPB)\n\
        acc += wrow[i] * x[(unsigned long)s * I + i];\n\
    threadgroup float wb[NW];\n\
    float total = reduce_all(acc, wb, tid);\n\
    if (tid == 0) y[(unsigned long)s * O + o] = total;\n\
}\n\
\n\
/* ---------------- F16 (coalesced half4) ---------------- */\n\
kernel void mm_f16(device const uint8_t* w [[buffer(0)]],\n\
                   device const float*   x [[buffer(1)]],\n\
                   device float*         y [[buffer(2)]],\n\
                   constant int& I [[buffer(3)]],\n\
                   constant int& S [[buffer(4)]],\n\
                   constant int& O [[buffer(5)]],\n\
                   uint tid [[thread_index_in_threadgroup]],\n\
                   uint2 gp [[threadgroup_position_in_grid]]) {\n\
    int o = (int)gp.x, s = (int)gp.y;\n\
    device const half* wrow = (device const half*)(w + (unsigned long)o * (unsigned long)(I * 2));\n\
    float acc = 0.0f;\n\
    for (int i = (int)tid * 4; i + 4 <= I; i += TPB * 4) {\n\
        float4 wv = float4(*(device const half4*)(wrow + i));\n\
        float4 xv = *(device const float4*)(x + (unsigned long)s * I + i);\n\
        acc += dot(wv, xv);\n\
    }\n\
    for (int i = (I / 4) * 4 + (int)tid; i < I; i += TPB)\n\
        acc += (float)wrow[i] * x[(unsigned long)s * I + i];\n\
    threadgroup float wb[NW];\n\
    float total = reduce_all(acc, wb, tid);\n\
    if (tid == 0) y[(unsigned long)s * O + o] = total;\n\
}\n\
\n\
/* ---------------- Q4_0 (18B/32v, SEPARATED nibbles) ----------------\n\
 * Per-element, metadata cached; simd_sum reduce. Tiny blocks (16 qs bytes for\n\
 * 32 values) don't coalesce cleanly across 32 lanes, so we keep this simple. */\n\
kernel void mm_q4_0(device const uint8_t* w [[buffer(0)]],\n\
                    device const float*   x [[buffer(1)]],\n\
                    device float*         y [[buffer(2)]],\n\
                    constant int& I [[buffer(3)]],\n\
                    constant int& S [[buffer(4)]],\n\
                    constant int& O [[buffer(5)]],\n\
                    uint tid [[thread_index_in_threadgroup]],\n\
                    uint2 gp [[threadgroup_position_in_grid]]) {\n\
    int o = (int)gp.x, s = (int)gp.y;\n\
    int nblocks = I / 32;\n\
    device const uint8_t* row = w + (unsigned long)o * (unsigned long)(nblocks * 18);\n\
    float acc = 0.0f;\n\
    for (int bi = (int)tid; bi < nblocks; bi += TPB) {\n\
        device const uint8_t* blk = row + (unsigned long)(bi * 18);\n\
        float d = f16tof32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));\n\
        device const uint8_t* qs = blk + 2;\n\
        for (int j = 0; j < 32; j++) {\n\
            int byte_idx = j & 15;\n\
            int shift = (j < 16) ? 0 : 4;\n\
            int nib = (qs[byte_idx] >> shift) & 0xF;\n\
            acc += x[(unsigned long)s * I + bi * 32 + j] * (float)(nib - 8) * d;\n\
        }\n\
    }\n\
    threadgroup float wb[NW];\n\
    float total = reduce_all(acc, wb, tid);\n\
    if (tid == 0) y[(unsigned long)s * O + o] = total;\n\
}\n\
\n\
/* ---------------- Q8_0 (34B/32v)  COALESCED, 1 block/simdgroup -----------\n\
 * 32 lanes read 32 consecutive int8 qs bytes (one coalesced 32B load); each\n\
 * lane owns value `lane` of the block. d broadcast from L1. */\n\
kernel void mm_q8_0(device const uint8_t* w [[buffer(0)]],\n\
                    device const float*   x [[buffer(1)]],\n\
                    device float*         y [[buffer(2)]],\n\
                    constant int& I [[buffer(3)]],\n\
                    constant int& S [[buffer(4)]],\n\
                    constant int& O [[buffer(5)]],\n\
                    uint tid [[thread_index_in_threadgroup]],\n\
                    uint2 gp [[threadgroup_position_in_grid]]) {\n\
    int o = (int)gp.x, s = (int)gp.y;\n\
    int nblocks = I / 32;\n\
    device const uint8_t* row = w + (unsigned long)o * (unsigned long)(nblocks * 34);\n\
    uint warp = tid >> 5, lane = tid & 31;\n\
    float acc = 0.0f;\n\
    for (int bi = (int)warp; bi < nblocks; bi += NW) {\n\
        device const uint8_t* blk = row + (unsigned long)(bi * 34);\n\
        float d = f16tof32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));\n\
        int8_t q = (int8_t)blk[2 + lane];          // 32 lanes x 1 byte = coalesced\n\
        acc += x[(unsigned long)s * I + bi * 32 + lane] * (float)q * d;\n\
    }\n\
    threadgroup float wb[NW];\n\
    float total = reduce_all(acc, wb, tid);\n\
    if (tid == 0) y[(unsigned long)s * O + o] = total;\n\
}\n\
\n\
/* ---------------- Q4_K (144B/256v)  COALESCED, 1 block/simdgroup ---------\n\
 * The dominant, hot type. One simdgroup (32 lanes) processes one block.\n\
 * Each lane reads qs[lane*4 .. lane*4+4) -> 4 bytes -> 8 nibbles -> 8 values,\n\
 * a single coalesced 128-byte read across the simdgroup for the whole qs[128].\n\
 * Lane g=lane/8 owns superblock group g; within=(lane&7)*4. Indexing derived\n\
 * from quant.c dequantize_row_q4_K (hand-verified for j in {0,32,64,100}). */\n\
kernel void mm_q4_K(device const uint8_t* w [[buffer(0)]],\n\
                    device const float*   x [[buffer(1)]],\n\
                    device float*         y [[buffer(2)]],\n\
                    constant int& I [[buffer(3)]],\n\
                    constant int& S [[buffer(4)]],\n\
                    constant int& O [[buffer(5)]],\n\
                    uint tid [[thread_index_in_threadgroup]],\n\
                    uint2 gp [[threadgroup_position_in_grid]]) {\n\
    int o = (int)gp.x, s = (int)gp.y;\n\
    int nblocks = I / 256;\n\
    device const uint8_t* row = w + (unsigned long)o * (unsigned long)(nblocks * 144);\n\
    uint warp = tid >> 5, lane = tid & 31;\n\
    uint g = lane / 8;\n\
    uint within = (lane & 7) * 4;\n\
    float acc = 0.0f;\n\
    for (int bi = (int)warp; bi < nblocks; bi += NW) {\n\
        device const uint8_t* blk = row + (unsigned long)(bi * 144);\n\
        float d    = f16tof32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));\n\
        float dmin = f16tof32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));\n\
        device const uint8_t* scales = blk + 4;\n\
        device const uint8_t* qs     = blk + 16;\n\
        float d_lo, m_lo, d_hi, m_hi;\n\
        k4_unpack(scales, 2 * (int)g,     d_lo, m_lo, d, dmin);\n\
        k4_unpack(scales, 2 * (int)g + 1, d_hi, m_hi, d, dmin);\n\
        uint qw = *(device const uint*)(qs + lane * 4);   // 4 bytes, coalesced\n\
        uint base = (uint)(bi * 256) + g * 64 + within;\n\
        #pragma unroll\n\
        for (int k = 0; k < 4; k++) {\n\
            uint bv = (qw >> (k * 8)) & 0xFF;\n\
            float lo = (float)(bv & 0xF);\n\
            float hi = (float)(bv >> 4);\n\
            acc += x[(unsigned long)s * I + base + k]      * (d_lo * lo - m_lo);\n\
            acc += x[(unsigned long)s * I + base + k + 32] * (d_hi * hi - m_hi);\n\
        }\n\
    }\n\
    threadgroup float wb[NW];\n\
    float total = reduce_all(acc, wb, tid);\n\
    if (tid == 0) y[(unsigned long)s * O + o] = total;\n\
}\n\
\n\
/* ---------------- Q5_K (176B/256v)  metadata cached + simd_sum -----------\n\
 * block_q5_K: uint16 d, dm, scales[12], qh[32], qs[128]. 5th bit per value in qh. */\n\
kernel void mm_q5_K(device const uint8_t* w [[buffer(0)]],\n\
                    device const float*   x [[buffer(1)]],\n\
                    device float*         y [[buffer(2)]],\n\
                    constant int& I [[buffer(3)]],\n\
                    constant int& S [[buffer(4)]],\n\
                    constant int& O [[buffer(5)]],\n\
                    uint tid [[thread_index_in_threadgroup]],\n\
                    uint2 gp [[threadgroup_position_in_grid]]) {\n\
    int o = (int)gp.x, s = (int)gp.y;\n\
    int nblocks = I / 256;\n\
    device const uint8_t* row = w + (unsigned long)o * (unsigned long)(nblocks * 176);\n\
    float acc = 0.0f;\n\
    for (int bi = (int)tid; bi < nblocks; bi += TPB) {\n\
        device const uint8_t* blk = row + (unsigned long)(bi * 176);\n\
        float d  = f16tof32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));\n\
        float dm = f16tof32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));\n\
        device const uint8_t* scales = blk + 4;\n\
        device const uint8_t* qh = blk + 16;\n\
        device const uint8_t* ql = blk + 48;\n\
        float dsc[8], dmn[8];\n\
        for (int is = 0; is < 8; is++) k4_unpack(scales, is, dsc[is], dmn[is], d, dm);\n\
        for (int j = 0; j < 256; j++) {\n\
            int sb = j / 64, local = j % 64;\n\
            int h = local / 32, l = local % 32;\n\
            int is = sb * 2 + h;\n\
            uint8_t qlb = ql[sb * 32 + l];\n\
            int nib = (h == 0) ? (qlb & 0xF) : (qlb >> 4);\n\
            int hi = (qh[l] >> (sb * 2 + h)) & 1;\n\
            acc += x[(unsigned long)s * I + bi * 256 + j] * (dsc[is] * (float)(nib + hi * 16) - dmn[is]);\n\
        }\n\
    }\n\
    threadgroup float wb[NW];\n\
    float total = reduce_all(acc, wb, tid);\n\
    if (tid == 0) y[(unsigned long)s * O + o] = total;\n\
}\n\
\n\
/* ---------------- Q6_K (210B/256v)  COALESCED, 1 block/simdgroup ---------\n\
 * One simdgroup (32 lanes) processes one 256-value block; each lane owns 8\n\
 * values. ql[128] read in 4 coalesced 32B passes (lane t owns bytes t,32+t,\n\
 * 64+t,96+t); qh[64] in 2 coalesced passes; scales read direct (cached).\n\
 * Indexing derived from quant.c dequantize_row_q6_K; validated vs the CPU\n\
 * reference by probes/metal_matmul_bench (4-11x faster than per-element). */\n\
kernel void mm_q6_K(device const uint8_t* w [[buffer(0)]],\n\
                    device const float*   x [[buffer(1)]],\n\
                    device float*         y [[buffer(2)]],\n\
                    constant int& I [[buffer(3)]],\n\
                    constant int& S [[buffer(4)]],\n\
                    constant int& O [[buffer(5)]],\n\
                    uint tid [[thread_index_in_threadgroup]],\n\
                    uint2 gp [[threadgroup_position_in_grid]]) {\n\
    int o = (int)gp.x, s = (int)gp.y;\n\
    int nblocks = I / 256;\n\
    device const uint8_t* row = w + (unsigned long)o * (unsigned long)(nblocks * 210);\n\
    uint warp = tid >> 5, lane = tid & 31;\n\
    float acc = 0.0f;\n\
    for (int bi = (int)warp; bi < nblocks; bi += NW) {\n\
        device const uint8_t* blk = row + (unsigned long)(bi * 210);\n\
        float d = f16tof32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));\n\
        device const uint8_t* ql = blk;\n\
        device const uint8_t* qh = blk + 128;\n\
        device const int8_t*  sc = (device const int8_t*)(blk + 192);\n\
        #pragma unroll\n\
        for (int k = 0; k < 4; k++) {            /* 4 coalesced 32B ql reads */\n\
            uint b   = (uint)(k * 32) + lane;     /* ql byte this lane owns   */\n\
            uint chk = b >> 6;                     /* chunk = b/64            */\n\
            uint grp = (b & 63) >> 5;             /* (b%64)/32               */\n\
            uint l   = b & 31;                     /* b%32                    */\n\
            uint qlb = ql[b];                      /* coalesced               */\n\
            uint qhb = qh[chk * 32 + l];\n\
            int qraw0 = (int)(qlb & 0xF) | (int)(((qhb >> (2u * grp)) & 3u) << 4);\n\
            int si0   = (int)(chk * 8 + l / 16 + 2u * grp);\n\
            uint j0   = chk * 128 + grp * 32 + l;\n\
            acc += x[(unsigned long)s * I + (unsigned long)bi * 256 + j0] * (d * (float)sc[si0] * (float)(qraw0 - 32));\n\
            int qraw1 = (int)(qlb >> 4) | (int)(((qhb >> (2u * (grp + 2u))) & 3u) << 4);\n\
            int si1   = (int)(chk * 8 + l / 16 + 2u * (grp + 2u));\n\
            uint j1   = chk * 128 + (grp + 2u) * 32 + l;\n\
            acc += x[(unsigned long)s * I + (unsigned long)bi * 256 + j1] * (d * (float)sc[si1] * (float)(qraw1 - 32));\n\
        }\n\
    }\n\
    threadgroup float wb[NW];\n\
    float total = reduce_all(acc, wb, tid);\n\
    if (tid == 0) y[(unsigned long)s * O + o] = total;\n\
}\n\
\n\
/* ---------------- Q4_K MULTI-OUTPUT GEMV (1 warp/output) --------------\n\
 * grid=(O/8,S,1), TG=256 (8 warps). warp w computes output o=gp.x*8+w over\n\
 * ALL blocks; simd_sum within the warp only (no cross-warp barrier/shared).\n\
 * Validated vs CPU by probes/metal_matmul_bench; faster on medium/large O.\n\
 * Host dispatches grid_x=(O+7)/8 for Q4_K/Q6_K (see launch_matmul). */\n\
kernel void mm_q4_K_mo(device const uint8_t* w [[buffer(0)]],\n\
                    device const float*   x [[buffer(1)]],\n\
                    device float*         y [[buffer(2)]],\n\
                    constant int& I [[buffer(3)]],\n\
                    constant int& S [[buffer(4)]],\n\
                    constant int& O [[buffer(5)]],\n\
                    uint tid [[thread_index_in_threadgroup]],\n\
                    uint2 gp [[threadgroup_position_in_grid]]) {\n\
    int o = (int)gp.x * 8 + (int)(tid >> 5), s = (int)gp.y;\n\
    if (o >= O) return;\n\
    int nblocks = I / 256;\n\
    device const uint8_t* row = w + (unsigned long)o * (unsigned long)(nblocks * 144);\n\
    uint lane = tid & 31, g = lane / 8, within = (lane & 7) * 4;\n\
    float acc = 0.0f;\n\
    for (int bi = 0; bi < nblocks; bi++) {\n\
        device const uint8_t* blk = row + (unsigned long)(bi * 144);\n\
        float d    = f16tof32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));\n\
        float dmin = f16tof32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));\n\
        device const uint8_t* scales = blk + 4;\n\
        device const uint8_t* qs     = blk + 16;\n\
        float d_lo, m_lo, d_hi, m_hi;\n\
        k4_unpack(scales, 2 * (int)g,     d_lo, m_lo, d, dmin);\n\
        k4_unpack(scales, 2 * (int)g + 1, d_hi, m_hi, d, dmin);\n\
        uint qw = *(device const uint*)(qs + lane * 4);\n\
        uint base = (uint)(bi * 256) + g * 64 + within;\n\
        for (int k = 0; k < 4; k++) {\n\
            uint bv = (qw >> (k * 8)) & 0xFF;\n\
            float lo = (float)(bv & 0xF);\n\
            float hi = (float)(bv >> 4);\n\
            acc += x[(unsigned long)s * I + base + k]      * (d_lo * lo - m_lo);\n\
            acc += x[(unsigned long)s * I + base + k + 32] * (d_hi * hi - m_hi);\n\
        }\n\
    }\n\
    float total = simd_sum(acc);\n\
    if (lane == 0) y[(unsigned long)s * O + o] = total;\n\
}\n\
/* ---------------- Q6_K MULTI-OUTPUT GEMV (1 warp/output) --------------\n\
 * grid=(O/8,S,1). warp w -> output o=gp.x*8+w over ALL blocks; simd_sum only.\n\
 * Coalesced ql/qh reads as in mm_q6_K. Validated vs CPU; faster on large O. */\n\
kernel void mm_q6_K_mo(device const uint8_t* w [[buffer(0)]],\n\
                    device const float*   x [[buffer(1)]],\n\
                    device float*         y [[buffer(2)]],\n\
                    constant int& I [[buffer(3)]],\n\
                    constant int& S [[buffer(4)]],\n\
                    constant int& O [[buffer(5)]],\n\
                    uint tid [[thread_index_in_threadgroup]],\n\
                    uint2 gp [[threadgroup_position_in_grid]]) {\n\
    int o = (int)gp.x * 8 + (int)(tid >> 5), s = (int)gp.y;\n\
    if (o >= O) return;\n\
    int nblocks = I / 256;\n\
    device const uint8_t* row = w + (unsigned long)o * (unsigned long)(nblocks * 210);\n\
    uint lane = tid & 31;\n\
    float acc = 0.0f;\n\
    for (int bi = 0; bi < nblocks; bi++) {\n\
        device const uint8_t* blk = row + (unsigned long)(bi * 210);\n\
        float d = f16tof32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));\n\
        device const uint8_t* ql = blk;\n\
        device const uint8_t* qh = blk + 128;\n\
        device const int8_t*  sc = (device const int8_t*)(blk + 192);\n\
        for (int k = 0; k < 4; k++) {\n\
            uint b   = (uint)(k * 32) + lane;\n\
            uint chk = b >> 6;\n\
            uint grp = (b & 63) >> 5;\n\
            uint l   = b & 31;\n\
            uint qlb = ql[b];\n\
            uint qhb = qh[chk * 32 + l];\n\
            int qraw0 = (int)(qlb & 0xF) | (int)(((qhb >> (2u * grp)) & 3u) << 4);\n\
            int si0   = (int)(chk * 8 + l / 16 + 2u * grp);\n\
            uint j0   = chk * 128 + grp * 32 + l;\n\
            acc += x[(unsigned long)s * I + (unsigned long)bi * 256 + j0] * (d * (float)sc[si0] * (float)(qraw0 - 32));\n\
            int qraw1 = (int)(qlb >> 4) | (int)(((qhb >> (2u * (grp + 2u))) & 3u) << 4);\n\
            int si1   = (int)(chk * 8 + l / 16 + 2u * (grp + 2u));\n\
            uint j1   = chk * 128 + (grp + 2u) * 32 + l;\n\
            acc += x[(unsigned long)s * I + (unsigned long)bi * 256 + j1] * (d * (float)sc[si1] * (float)(qraw1 - 32));\n\
        }\n\
    }\n\
    float total = simd_sum(acc);\n\
    if (lane == 0) y[(unsigned long)s * O + o] = total;\n\
}\n\
/* ---------------- silu_mul elementwise: gate[i] = silu(gate[i]) * up[i] --- */\n\
kernel void silu_mul(device float*       gate [[buffer(0)]],\n\
                     device const float* up   [[buffer(1)]],\n\
                     constant uint& n   [[buffer(2)]],\n\
                     uint gid [[thread_position_in_grid]]) {\n\
    if (gid < n) {\n\
        float v = gate[gid];\n\
        gate[gid] = (v / (1.0f + exp(-v))) * up[gid];\n\
    }\n\
}\n\
\n\
/* =============== SSM batched quantized vec_dot (1 head / threadgroup) =========\n\
 * Ported from backend_gpu.cu picolm_ssm_vecdot_kernel. One threadgroup per\n\
 * v-head: out[h] = x . dequant(weights[head_map[h]]) over `dim` elements,\n\
 * reduced with reduce_all. Only Q4_0 / Q8_0 reach here (model.c host gate);\n\
 * other types -> CPU fallback. Dequant matches quant.c / mm_q4_0 / mm_q8_0.\n\
 * head_map is non-NULL on the host path (NULL identity is handled by uploading\n\
 * an identity map in the host wrapper). */\n\
kernel void ssm_vecdot(device float*         out      [[buffer(0)]],\n\
                       device const float*   x        [[buffer(1)]],\n\
                       device const uint8_t* weights  [[buffer(2)]],\n\
                       constant int& qtype    [[buffer(3)]],\n\
                       constant int& dim      [[buffer(4)]],\n\
                       constant int& n_v_heads[[buffer(5)]],\n\
                       constant int& row_bytes[[buffer(6)]],\n\
                       device const int*   head_map [[buffer(7)]],\n\
                       uint tid [[thread_index_in_threadgroup]],\n\
                       uint  h  [[threadgroup_position_in_grid]]) {\n\
    if ((int)h >= n_v_heads) return;\n\
    int gh = head_map[h];\n\
    device const uint8_t* wrow = weights + (unsigned long)gh * (unsigned long)row_bytes;\n\
    float acc = 0.0f;\n\
    if (qtype == 2) {                       /* Q4_0: 18B/32v, separated nibbles */\n\
        int nb = dim / 32;\n\
        for (int bi = (int)tid; bi < nb; bi += TPB) {\n\
            device const uint8_t* blk = wrow + (unsigned long)(bi * 18);\n\
            float d = f16tof32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));\n\
            device const uint8_t* qs = blk + 2;\n\
            for (int j = 0; j < 32; j++) {\n\
                int byte_idx = j & 15;\n\
                int shift = (j < 16) ? 0 : 4;\n\
                int nib = (qs[byte_idx] >> shift) & 0xF;\n\
                acc += x[bi * 32 + j] * (float)(nib - 8) * d;\n\
            }\n\
        }\n\
    } else if (qtype == 8) {                /* Q8_0: 34B/32v */\n\
        int nb = dim / 32;\n\
        for (int bi = (int)tid; bi < nb; bi += TPB) {\n\
            device const uint8_t* blk = wrow + (unsigned long)(bi * 34);\n\
            float d = f16tof32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));\n\
            device const int8_t* qs = (device const int8_t*)(blk + 2);\n\
            for (int j = 0; j < 32; j++)\n\
                acc += x[bi * 32 + j] * (float)qs[j] * d;\n\
        }\n\
    }\n\
    threadgroup float wb[NW];\n\
    float total = reduce_all(acc, wb, tid);\n\
    if (tid == 0) out[h] = total;\n\
}\n\
\n\
/* =============== SSM DeltaNet recurrence (1 head / threadgroup) ==============\n\
 * Ported from backend_gpu.cu picolm_ssm_recurrence_kernel; matches the CPU\n\
 * scalar reference in model.c ssm_head_task (same op order). One threadgroup of\n\
 * TPB lanes owns one v-head's [d_state x d_state] state. d_state <= 256 (host\n\
 * returns 0 -> CPU fallback otherwise). Fixed threadgroup scratch\n\
 * sk[256] + d_local[256] = 2 KB covers the max d_state.\n\
 * Per head h (k_head = h/repeat shares q_conv/k_conv):\n\
 *   1. decay state *= gate_exp   2. sk[row]=state[row].k   3. d=(v-sk)*beta\n\
 *   4. state += k^T d (rank-1)   5. out[row*n_v_heads+h] = state[row].q\n\
 * Barriers separate phases; within a phase threads touch disjoint elements. */\n\
kernel void ssm_recurrence(device float*       state    [[buffer(0)]],\n\
                           device const float* q_conv   [[buffer(1)]],\n\
                           device const float* k_conv   [[buffer(2)]],\n\
                           device const float* v_conv   [[buffer(3)]],\n\
                           device const float* gate_exp [[buffer(4)]],\n\
                           device const float* beta     [[buffer(5)]],\n\
                           device float*       ssm_out  [[buffer(6)]],\n\
                           constant int* dims          [[buffer(7)]],\n\
                           uint tid [[thread_index_in_threadgroup]],\n\
                           uint  h  [[threadgroup_position_in_grid]]) {\n\
    int n_v_heads = dims[0];\n\
    int d_state   = dims[1];\n\
    int repeat    = dims[2];\n\
    if ((int)h >= n_v_heads) return;\n\
    threadgroup float sk[256];\n\
    threadgroup float d_local[256];\n\
    int kh = (int)h / repeat;\n\
    device const float* qh  = q_conv + (unsigned long)kh * (unsigned long)d_state;\n\
    device const float* khv = k_conv + (unsigned long)kh * (unsigned long)d_state;\n\
    device const float* vh  = v_conv + (unsigned long)h  * (unsigned long)d_state;\n\
    float ge = gate_exp[h];\n\
    float bh = beta[h];\n\
    device float* st       = state   + (unsigned long)h * (unsigned long)d_state * (unsigned long)d_state;\n\
    device float* out_base = ssm_out + (unsigned long)h;       /* dim-major */\n\
    int N = d_state * d_state;\n\
    /* 1. decay */\n\
    for (int i = (int)tid; i < N; i += TPB) st[i] *= ge;\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    /* 2. sk[row] = state[row] . k */\n\
    for (int row = (int)tid; row < d_state; row += TPB) {\n\
        device const float* st_row = st + (unsigned long)row * (unsigned long)d_state;\n\
        float sum = 0.0f;\n\
        for (int col = 0; col < d_state; col++) sum += st_row[col] * khv[col];\n\
        sk[row] = sum;\n\
    }\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    /* 3. d[row] = (v[row] - sk[row]) * beta */\n\
    for (int row = (int)tid; row < d_state; row += TPB)\n\
        d_local[row] = (vh[row] - sk[row]) * bh;\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    /* 4. state[row][col] += k[col] * d[row]  (rank-1 update) */\n\
    for (int i = (int)tid; i < N; i += TPB) {\n\
        int row = i / d_state, col = i % d_state;\n\
        st[i] += khv[col] * d_local[row];\n\
    }\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    /* 5. out[row*n_v_heads+h] = state[row] . q */\n\
    for (int row = (int)tid; row < d_state; row += TPB) {\n\
        device const float* st_row = st + (unsigned long)row * (unsigned long)d_state;\n\
        float sum = 0.0f;\n\
        for (int col = 0; col < d_state; col++) sum += st_row[col] * qh[col];\n\
        out_base[(unsigned long)row * (unsigned long)n_v_heads] = sum;\n\
    }\n\
}\n\
";

#pragma mark - Device context (single global; Apple Silicon has one GPU)

// Under -fobjc-arc these static globals are strong (retained), released on
// program exit / shutdown. Apple Silicon exposes exactly one Metal device, so
// a single context suffices; the `device` ABI param is validated to be 0.
static id<MTLDevice>            g_device  = nil;
static id<MTLCommandQueue>       g_queue   = nil;
static id<MTLLibrary>            g_library = nil;
static id<MTLComputePipelineState> g_ps_f32, g_ps_f16, g_ps_q4_0, g_ps_q8_0,
                                  g_ps_q4_K, g_ps_q5_K, g_ps_q6_K, g_ps_silu,
                                  g_ps_q4_K_mo, g_ps_q6_K_mo,
                                  g_ps_ssm_vecdot, g_ps_ssm_recurrence;
static int g_ndev = 0;

// Grow-only scratch buffers (mirror CUDA's reserve()). Shared storage mode =
// unified memory: CPU writes x, GPU reads x & writes y, CPU reads y.
//
// These stay hazard-TRACKED (the default): picolm_gpu_expert_mlp issues 4
// separate compute encoders in ONE command buffer with cross-encoder
// read-after-write deps on g_gate/g_up (gate matmul -> silu_mul -> down
// matmul). Default tracking is what serializes those across encoders; making
// them Untracked would race. The per-call waitUntilCompleted below does not
// help with intra-command-buffer ordering.
// g_ssm holds the per-layer SSM recurrence state during a kernel run: it is
// CPU-uploaded (state H2D), read AND written by the recurrence kernel, then
// CPU-read back (state D2H). DefaultCache like g_y (GPU-write/CPU-read).
static id<MTLBuffer> g_x = nil, g_y = nil, g_gate = nil, g_up = nil, g_ssm = nil;
static size_t g_xcap = 0, g_ycap = 0, g_gatecap = 0, g_upcap = 0, g_ssmcap = 0;
// Cached CPU mappings of the scratch buffers (stable for a buffer's lifetime;
// refreshed only on reallocation inside reserve_buf). Saves a
// -[MTLBuffer contents] message send on every dispatch.
static void *g_xptr = NULL, *g_yptr = NULL, *g_gateptr = NULL, *g_upptr = NULL,
           *g_ssmptr = NULL;

// Optional profiling (enable with PICOLM_GPU_PROFILE=1). Accumulates wall time
// of the GPU entry points so a decode run shows where per-token time goes
// (SSM recurrence copy vs GPU, and total GPU-active time vs decode budget).
static int    g_prof = 0;
static long   g_prof_calls = 0;
static double g_prof_gpu_ms = 0;     // total GPU-active time (dispatch+wait)
static double g_prof_rec_copy_ms = 0;// SSM recurrence H2D+D2H memcpy time
static double g_prof_rec_gpu_ms = 0; // SSM recurrence dispatch+wait time
static long   g_prof_rec_calls = 0;
static inline double prof_now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}
static inline void prof_maybe_init(void) {
    static int inited = 0;
    if (!inited) { g_prof = getenv("PICOLM_GPU_PROFILE") != NULL; inited = 1; }
}
static inline void prof_report(void) {
    if (!g_prof || (g_prof_calls % 256) != 0) return;
    static double p_gpu = 0, p_rc = 0, p_rg = 0;
    static long   p_calls = 0, p_rcalls = 0;
    double d_gpu = g_prof_gpu_ms - p_gpu, d_rc = g_prof_rec_copy_ms - p_rc, d_rg = g_prof_rec_gpu_ms - p_rg;
    long   d_calls = g_prof_calls - p_calls, d_rcalls = g_prof_rec_calls - p_rcalls;
    /* ~256 GPU calls ~= one decode token (matmuls + 48 SSM recurrences). */
    fprintf(stderr, "[prof] last %ld GPU calls (~1 tok): gpu_active=%.1f ms | "
                    "ssm_rec %ld layers: copy=%.1f ms gpu=%.1f ms\n",
            d_calls, d_gpu, d_rcalls, d_rc, d_rg);
    p_gpu = g_prof_gpu_ms; p_rc = g_prof_rec_copy_ms; p_rg = g_prof_rec_gpu_ms;
    p_calls = g_prof_calls; p_rcalls = g_prof_rec_calls;
}

static id<MTLComputePipelineState> ps_for_qtype(gguf_type_t q) {
    switch (q) {
        case GGUF_TYPE_F32:  return g_ps_f32;
        case GGUF_TYPE_F16:  return g_ps_f16;
        case GGUF_TYPE_Q4_0: return g_ps_q4_0;
        case GGUF_TYPE_Q8_0: return g_ps_q8_0;
        case GGUF_TYPE_Q4_K: return g_ps_q4_K_mo;
        case GGUF_TYPE_Q5_K: return g_ps_q5_K;
        case GGUF_TYPE_Q6_K: return g_ps_q6_K_mo;
        default:             return nil;   /* unsupported -> caller falls back to CPU */
    }
}

// Grow-only: only (re)allocates when current capacity is too small.
// Parameter qualified __strong so we can pass &g_x (a strong global); the
// default (__autoreleasing*) would reject address-of-strong-global under ARC.
//
// `options` tunes the CPU cache policy for shared storage:
//   - WriteCombined for CPU->GPU upload buffers (g_x): the CPU write streams
//     straight to unified memory without evicting hot host data, and is never
//     read back on the CPU. This is the hot path (every matmul memcpy()s x in).
//   - DefaultCache  for GPU->CPU download buffers (g_y): CPU reads them back.
// `ptr_out` (may be NULL) receives the stable -contents pointer, so callers
// skip the message send per dispatch.
static id<MTLBuffer> reserve_buf(__strong id<MTLBuffer> *buf, size_t *cap,
                                 void **ptr_out, size_t bytes,
                                 MTLResourceOptions options) {
    if (*buf && *cap >= bytes) return *buf;
    *buf = [g_device newBufferWithLength:bytes options:options];
    if (!*buf) { *cap = 0; if (ptr_out) *ptr_out = NULL; return nil; }
    *cap = bytes;
    if (ptr_out) *ptr_out = [*buf contents];
    return *buf;
}

// Row size in bytes for a given qtype / column count (mirrors CUDA gguf_block_size).
static size_t gguf_row_bytes(gguf_type_t q, int I) {
    switch (q) {
        case GGUF_TYPE_F32:  return (size_t)I * 4;
        case GGUF_TYPE_F16:  return (size_t)I * 2;
        case GGUF_TYPE_Q4_0: return (size_t)((I / 32) * 18);
        case GGUF_TYPE_Q8_0: return (size_t)((I / 32) * 34);
        case GGUF_TYPE_Q4_K: return (size_t)((I / 256) * 144);
        case GGUF_TYPE_Q5_K: return (size_t)((I / 256) * 176);
        case GGUF_TYPE_Q6_K: return (size_t)((I / 256) * 210);
        default:             return 0;
    }
}

#pragma mark - Public API (backend_gpu.h)

extern "C" {

int picolm_gpu_init(const int *devices, int count) {
    prof_maybe_init();
    if (g_ndev > 0) return 1;                 /* idempotent */
    if (!devices || count < 1 || count > PICOLM_GPU_MAX_DEVICES) return 0;

    /* Apple Silicon has exactly one GPU; honor only device 0. */
    int want = devices[0];
    if (want != 0) {
        fprintf(stderr, "[Metal] only device 0 is supported on Apple Silicon (asked for %d)\n", want);
        return 0;
    }

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) { fprintf(stderr, "[Metal] no Metal device\n"); return 0; }
    g_queue = [g_device newCommandQueue];
    if (!g_queue) { g_device = nil; return 0; }

    NSError *err = nil;
    g_library = [g_device newLibraryWithSource:KERNEL_SOURCE options:nil error:&err];
    if (!g_library) {
        fprintf(stderr, "[Metal] kernel compile failed: %s\n",
                [[err localizedDescription] UTF8String]);
        g_device = nil; g_queue = nil; return 0;
    }
    /* Eagerly build all pipeline states (one compile per function). */
    g_ps_f32  = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"mm_f32"]  error:&err];
    g_ps_f16  = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"mm_f16"]  error:nil];
    g_ps_q4_0 = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"mm_q4_0"] error:nil];
    g_ps_q8_0 = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"mm_q8_0"] error:nil];
    g_ps_q4_K = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"mm_q4_K"] error:nil];
    g_ps_q5_K = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"mm_q5_K"] error:nil];
    g_ps_q6_K = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"mm_q6_K"] error:nil];
    g_ps_q4_K_mo = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"mm_q4_K_mo"] error:nil];
    g_ps_q6_K_mo = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"mm_q6_K_mo"] error:nil];
    g_ps_silu = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"silu_mul"] error:nil];
    g_ps_ssm_vecdot     = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"ssm_vecdot"]     error:nil];
    g_ps_ssm_recurrence = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"ssm_recurrence"] error:nil];
    if (!g_ps_f32 || !g_ps_f16 || !g_ps_q4_0 || !g_ps_q8_0 || !g_ps_q4_K || !g_ps_q5_K || !g_ps_q6_K || !g_ps_q4_K_mo || !g_ps_q6_K_mo || !g_ps_silu) {
        fprintf(stderr, "[Metal] pipeline state creation failed\n");
        g_device = nil; g_queue = nil; g_library = nil; return 0;
    }
    /* The SSM kernels are new and not exercised by non-SSM models; if either
     * fails to compile we keep GPU matmul/FFN working and just fall the SSM
     * path back to CPU (the wrappers check for nil pipeline and return 0). */
    if (!g_ps_ssm_vecdot || !g_ps_ssm_recurrence)
        fprintf(stderr, "[Metal] warning: SSM kernel compile failed; SSM will run on CPU\n");

    g_ndev = 1;
    fprintf(stderr, "[Metal] device: %s, working set %.1f GB\n",
            [[g_device name] UTF8String],
            (double)[g_device recommendedMaxWorkingSetSize] / 1e9);
    return 1;
}

void picolm_gpu_shutdown(void) {
    g_x = g_y = g_gate = g_up = g_ssm = nil;
    g_xcap = g_ycap = g_gatecap = g_upcap = g_ssmcap = 0;
    g_xptr = g_yptr = g_gateptr = g_upptr = g_ssmptr = NULL;
    g_ps_f32 = g_ps_f16 = g_ps_q4_0 = g_ps_q8_0 = g_ps_q4_K = g_ps_q5_K = g_ps_q6_K = g_ps_silu = nil;
    g_ps_ssm_vecdot = g_ps_ssm_recurrence = nil;
    g_library = nil; g_queue = nil; g_device = nil;
    g_ndev = 0;
}

int  picolm_gpu_device_count(void)            { return g_ndev; }
int  picolm_gpu_device_at(int index)          { return (index == 0 && g_ndev > 0) ? 0 : -1; }

int picolm_gpu_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    if (device != 0 || !g_device || !free_bytes || !total_bytes) return 0;
    size_t ws = [g_device recommendedMaxWorkingSetSize];
    *total_bytes = ws;
    *free_bytes  = ws;    /* Metal exposes no precise "free"; report the budget. */
    return 1;
}

struct picolm_gpu_tensor {
    void *weights;         /* id<MTLBuffer> (bridged-retained under ARC) */
    gguf_type_t qtype;
    int I, O, device;
    size_t row_bytes;
    size_t buf_offset;     /* byte offset of the tensor within its MTLBuffer */
    int zero_copy;
};

// NOTE: param is `void**` (not picolm_gpu_tensor_t**) to match the extern "C"
// declaration in backend_gpu.h. A T** definition against a void** decl does
// NOT match in C++ -> it would emit a mangled symbol and fail to link against
// model.o (compiled as C, which references the unmangled name). Verified.
int picolm_gpu_tensor_upload(void **tensor,
                              const void *weights,
                              gguf_type_t qtype, int I, int O, int device) {
    if (!tensor || !weights || I < 1 || O < 1 || device != 0 || !g_device) return 0;
    if (!ps_for_qtype(qtype)) return 0;       /* unsupported type -> CPU fallback (handle stays NULL) */
    if (*tensor) return 1;                    /* idempotent */

    size_t row_bytes = gguf_row_bytes(qtype, I);
    if (!row_bytes) return 0;
    size_t total = row_bytes * (size_t)O;

    id<MTLBuffer> buf = nil;
    int zero_copy = 0;
    size_t buf_offset = 0;

    /* Zero-copy on Apple Silicon is essential for big models: a 27B model is
     * ~16 GB, and COPYING it into Metal buffers doubles resident memory and
     * blows the GPU working set -> unified-memory paging -> catastrophic
     * slowdown. newBufferWithBytesNoCopy: needs BOTH pointer and length to be
     * page-aligned, which per-tensor pointers rarely are. So we page-align
     * DOWN the pointer and UP the length, register the aligned region, and
     * bind the tensor at its offset within that region. This makes almost every
     * tensor zero-copy with no per-tensor copy and no extra RAM. (If Metal
     * rejects the region  e.g. some overlap cases  we fall back to copy.) */
    size_t page = (size_t)getpagesize();
    uintptr_t addr = (uintptr_t)weights;
    uintptr_t base_addr = addr & ~(uintptr_t)(page - 1);   /* round down to page */
    size_t pad_lo = (size_t)(addr - base_addr);
    size_t aligned_len = (total + pad_lo + page - 1) & ~(page - 1);  /* round up */
    /* Untracked: weights are read-only on the GPU and never written by host or
     * device after upload, so there is no read-after-write hazard to track.
     * Removing per-resource hazard tracking lowers dispatch-encoding overhead
     * for every matmul that binds a weight buffer (the common case). Ordering
     * vs the CPU is provided by the per-call waitUntilCompleted full barrier. */
    buf = [g_device newBufferWithBytesNoCopy:(void *)base_addr
                                      length:aligned_len
                                      options:MTLResourceStorageModeShared |
                                               MTLResourceCPUCacheModeDefaultCache |
                                               MTLResourceHazardTrackingModeUntracked
                                      deallocator:^(void *ptr, NSUInteger len) {
                                          /* mmap'd memory is owned by the model; nothing to free. */
                                          (void)ptr; (void)len;
                                      }];
    if (buf) {
        zero_copy = 1;
        buf_offset = pad_lo;
    } else {
        /* Fallback: one-time copy into a shared-storage buffer (offset 0).
         * Untracked here too (see above): read-only on the GPU. */
        buf = [g_device newBufferWithBytes:weights length:total
                                   options:MTLResourceStorageModeShared |
                                            MTLResourceHazardTrackingModeUntracked];
        buf_offset = 0;
    }
    if (!buf) {
        fprintf(stderr, "[Metal] OOM uploading tensor: I=%d O=%d qtype=%d (%.1f MB)\n",
                I, O, (int)qtype, (double)total / (1024.0 * 1024.0));
        return 0;
    }

    picolm_gpu_tensor_t *t = (picolm_gpu_tensor_t *)calloc(1, sizeof(*t));
    if (!t) { return 0; }
    t->weights    = (__bridge_retained void *)buf;   /* ARC: transfer ownership to void* */
    t->qtype      = qtype;
    t->I          = I;
    t->O          = O;
    t->device     = device;
    t->row_bytes  = row_bytes;
    t->buf_offset = buf_offset;
    t->zero_copy  = zero_copy;

    static int first = 1;
    if (first) { fprintf(stderr, "[Metal] upload mode: %s\n", zero_copy ? "zero-copy" : "copied"); first = 0; }

    *tensor = t;
    return 1;
}

void picolm_gpu_tensor_free(picolm_gpu_tensor_t *t) {
    if (!t) return;
    if (t->weights) {
        /* Reclaim the bridged-retained MTLBuffer so ARC releases it. */
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)t->weights;
        (void)buf;
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

// Encode one quant_matmul: y[S*O] = x[S*I] @ W[O,I]^T, weights already resident.
static int launch_matmul(picolm_gpu_tensor_t *t, id<MTLBuffer> xbuf, id<MTLBuffer> ybuf,
                          int S, id<MTLCommandBuffer> cmd) {
    id<MTLComputePipelineState> ps = ps_for_qtype(t->qtype);
    if (!ps) return 0;
    int I = t->I, O = t->O;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:(__bridge id<MTLBuffer>)t->weights offset:t->buf_offset atIndex:0];
    [enc setBuffer:xbuf offset:0 atIndex:1];
    [enc setBuffer:ybuf offset:0 atIndex:2];
    int32_t cI = I, cS = S, cO = O;
    [enc setBytes:&cI length:sizeof(cI) atIndex:3];
    [enc setBytes:&cS length:sizeof(cS) atIndex:4];
    [enc setBytes:&cO length:sizeof(cO) atIndex:5];
    NSUInteger gx = (t->qtype == GGUF_TYPE_Q4_K || t->qtype == GGUF_TYPE_Q6_K)
                      ? (NSUInteger)((O + 7) / 8) : (NSUInteger)O;
    [enc dispatchThreadgroups:MTLSizeMake(gx, (NSUInteger)S, 1)
        threadsPerThreadgroup:MTLSizeMake(TPB, 1, 1)];
    [enc endEncoding];
    return 1;
}

int picolm_gpu_matmul(picolm_gpu_tensor_t *t, float *y, const float *x, int S, int device) {
    if (!t || !y || !x || S < 1 || device != 0 || !g_device) return 0;
    int I = t->I, O = t->O;
    size_t xb = (size_t)S * (size_t)I * sizeof(float);
    size_t yb = (size_t)S * (size_t)O * sizeof(float);
    /* g_x is CPU-written/GPU-read -> WriteCombined (streaming write, no cache
     * pollution). g_y is GPU-written/CPU-read -> DefaultCache. */
    if (!reserve_buf(&g_x, &g_xcap, &g_xptr, xb,
                     MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined)) return 0;
    if (!reserve_buf(&g_y, &g_ycap, &g_yptr, yb,
                     MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache)) return 0;

    memcpy(g_xptr, x, xb);

    /* Unretained command buffer: skips per-resource retain/release bookkeeping
     * for every bound buffer + pipeline state. Safe because every resource
     * bound here is a static global (scratch buffers, pipeline states) or a
     * model-resident weight tensor that outlives this command buffer. The
     * function holds the command buffer until waitUntilCompleted anyway. */
    id<MTLCommandBuffer> cmd = [g_queue commandBufferWithUnretainedReferences];
    if (!launch_matmul(t, g_x, g_y, S, cmd)) { return 0; }
    double _pt0 = g_prof ? prof_now_ms() : 0;
    [cmd commit];
    [cmd waitUntilCompleted];
    if (g_prof) { g_prof_gpu_ms += prof_now_ms() - _pt0; g_prof_calls++; prof_report(); }

    memcpy(y, g_yptr, yb);
    return 1;
}

int picolm_gpu_expert_mlp(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                           picolm_gpu_tensor_t *down, float *y, const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1) return 0;
    if (gate->device != up->device || gate->device != down->device || gate->device != 0) return 0;
    if (gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;

    int D = gate->I, I = gate->O;
    size_t xb = (size_t)S * (size_t)D * sizeof(float);
    size_t ib = (size_t)S * (size_t)I * sizeof(float);
    /* g_x is the only CPU->GPU upload here (WriteCombined). g_y is read back on
     * the CPU. g_gate/g_up are pure GPU scratch (written and re-read inside the
     * command buffer), so their CPU cache mode is irrelevant; DefaultCache. */
    if (!reserve_buf(&g_x,    &g_xcap,    &g_xptr,    xb,
                     MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined)) return 0;
    if (!reserve_buf(&g_y,    &g_ycap,    &g_yptr,    xb,
                     MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache)) return 0;
    if (!reserve_buf(&g_gate, &g_gatecap, &g_gateptr, ib,
                     MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache)) return 0;
    if (!reserve_buf(&g_up,   &g_upcap,   &g_upptr,   ib,
                     MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache)) return 0;

    memcpy(g_xptr, x, xb);

    /* Unretained command buffer: see picolm_gpu_matmul (every bound buffer
     * outlives this command buffer). */
    id<MTLCommandBuffer> cmd = [g_queue commandBufferWithUnretainedReferences];
    /* gate = W_gate @ x ; up = W_up @ x */
    if (!launch_matmul(gate, g_x, g_gate, S, cmd)) return 0;
    if (!launch_matmul(up,   g_x, g_up,   S, cmd)) return 0;
    /* silu(gate) * up  -> gate */
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ps_silu];
        [enc setBuffer:g_gate offset:0 atIndex:0];
        [enc setBuffer:g_up   offset:0 atIndex:1];
        uint32_t n = (uint32_t)(S * I);
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        NSUInteger ngrid = (n + TPB - 1) / TPB;
        [enc dispatchThreadgroups:MTLSizeMake(ngrid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(TPB, 1, 1)];
        [enc endEncoding];
    }
    /* y = W_down @ gate */
    if (!launch_matmul(down, g_gate, g_y, S, cmd)) return 0;
    double _pt0 = g_prof ? prof_now_ms() : 0;
    [cmd commit];
    [cmd waitUntilCompleted];
    if (g_prof) { g_prof_gpu_ms += prof_now_ms() - _pt0; g_prof_calls++; prof_report(); }

    memcpy(y, g_yptr, xb);
    return 1;
}

// Not implemented in v1: Apple GPUs have their own matrix accelerator but
// exposing it cleanly is a separate task. Returning 0 makes the caller
// (tensor.c / model.c) fall back to the quant_matmul path, which is correct.
int picolm_gpu_w4a16_mlp(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                          picolm_gpu_tensor_t *down, float *y, const float *x, int S) {
    (void)gate; (void)up; (void)down; (void)y; (void)x; (void)S;
    return 0;
}

// The three entry points below are referenced by the host under -DPICOLM_GPU=1
// (tensor.c picolm_gpu_w4a16_matmul; model.c ssm_vecdot/ssm_recurrence) but are
// not implemented for Metal in v1. Returning 0 makes the caller fall back to
// the CPU (quant_matmul / SSM) path, which is correct. These stubs are required
// for `make metal` to link against the current host code (which gained these
// GPU hooks after the Metal backend was first written). Verified needed via:
//   cc -DPICOLM_GPU=1 -c model.c tensor.c && nm -u *.o | grep picolm_gpu
int picolm_gpu_w4a16_matmul(picolm_gpu_tensor_t *t,
                             float *y, const float *x, int S, int device) {
    (void)t; (void)y; (void)x; (void)S; (void)device;
    return 0;   /* CPU fallback (WMMA Tensor Core path not exposed on Apple GPUs yet) */
}

/* ---- SSM batched quantized vec_dot (ported from backend_gpu.cu) ----
 * out[h] = x . dequant(weights[head_map[h]]) for each of n_v_heads heads.
 * Inputs packed into g_x as [x | weights | head_map]; out in g_y. Only Q4_0 /
 * Q8_0 are supported (matches the host gate in model.c); anything else returns 0
 * -> CPU fallback. head_map==NULL means identity (we upload an identity map). */
int picolm_gpu_ssm_vecdot(float *out,
                           const float *x,
                           const void *weights,
                           gguf_type_t qtype,
                           int dim, int n_v_heads,
                           int row_bytes,
                           const int *head_map,
                           int device) {
    if (!out || !x || !weights || n_v_heads <= 0 || dim <= 0 || row_bytes <= 0)
        return 0;
    if (device != 0 || !g_device || !g_ps_ssm_vecdot) return 0;
    if (qtype != GGUF_TYPE_Q4_0 && qtype != GGUF_TYPE_Q8_0) return 0;  /* CPU fallback */

    /* head_map: identity on stack when NULL (cap guards tiny stack array). */
    int idmap[256];
    const int *hmap = head_map;
    if (!hmap) {
        if (n_v_heads > (int)(sizeof(idmap) / sizeof(int))) return 0;
        for (int i = 0; i < n_v_heads; i++) idmap[i] = i;
        hmap = idmap;
    }

    size_t x_bytes    = (size_t)dim * sizeof(float);
    size_t w_bytes    = (size_t)n_v_heads * (size_t)row_bytes;
    size_t map_off    = (x_bytes + w_bytes + 15) & ~(size_t)15;  /* 16-align int head_map */
    size_t map_bytes  = (size_t)n_v_heads * sizeof(int);
    size_t total_in   = map_off + map_bytes;
    size_t out_bytes  = (size_t)n_v_heads * sizeof(float);

    if (!reserve_buf(&g_x, &g_xcap, &g_xptr, total_in,
                     MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined)) return 0;
    if (!reserve_buf(&g_y, &g_ycap, &g_yptr, out_bytes,
                     MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache)) return 0;

    memcpy(g_xptr, x, x_bytes);
    memcpy((uint8_t *)g_xptr + x_bytes, weights, w_bytes);
    memcpy((uint8_t *)g_xptr + map_off, hmap, map_bytes);

    id<MTLCommandBuffer> cmd = [g_queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_ssm_vecdot];
    [enc setBuffer:g_y offset:0                    atIndex:0];   /* out      */
    [enc setBuffer:g_x offset:0                    atIndex:1];   /* x        */
    [enc setBuffer:g_x offset:x_bytes              atIndex:2];   /* weights  */
    int32_t cq = (int32_t)qtype, cd = dim, cn = n_v_heads, cr = row_bytes;
    [enc setBytes:&cq length:sizeof(cq) atIndex:3];
    [enc setBytes:&cd length:sizeof(cd) atIndex:4];
    [enc setBytes:&cn length:sizeof(cn) atIndex:5];
    [enc setBytes:&cr length:sizeof(cr) atIndex:6];
    [enc setBuffer:g_x offset:map_off               atIndex:7];   /* head_map */
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_v_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(TPB, 1, 1)];
    [enc endEncoding];
    double _pt0 = g_prof ? prof_now_ms() : 0;
    [cmd commit];
    [cmd waitUntilCompleted];
    if (g_prof) { g_prof_gpu_ms += prof_now_ms() - _pt0; g_prof_calls++; prof_report(); }

    memcpy(out, g_yptr, out_bytes);
    return 1;
}

/* ---- SSM DeltaNet recurrence (ported from backend_gpu.cu) ----
 * One threadgroup per v-head. State is copied H2D into g_ssm, mutated by the
 * kernel, then copied D2H back (the state is persistent host memory owned by
 * the model). Small per-head inputs (q/k/v/gate_exp/beta) are packed into g_x;
 * ssm_output lands in g_y. d_state<=256 (else CPU fallback). */
int picolm_gpu_ssm_recurrence(float *state,
                               const float *q_conv,
                               const float *k_conv,
                               const float *v_conv,
                               const float *gate_exp,
                               const float *beta,
                               float *ssm_output,
                               int n_v_heads, int d_state,
                               int repeat, int device) {
    if (!state || !q_conv || !k_conv || !v_conv || !gate_exp || !beta || !ssm_output)
        return 0;
    if (n_v_heads <= 0 || d_state <= 0 || d_state > 256 || repeat < 1)
        return 0;
    if (device != 0 || !g_device || !g_ps_ssm_recurrence) return 0;

    int n_k_heads = n_v_heads / repeat;          /* k_head = h/repeat shares q/k */

    /* Pack [q_conv | k_conv | v_conv | gate_exp | beta] into g_x. */
    size_t q_bytes = (size_t)n_k_heads * (size_t)d_state * sizeof(float);
    size_t k_bytes = q_bytes;
    size_t v_bytes = (size_t)n_v_heads * (size_t)d_state * sizeof(float);
    size_t gb_bytes = (size_t)n_v_heads * sizeof(float);
    size_t q_off = 0;
    size_t k_off = q_off + q_bytes;
    size_t v_off = k_off + k_bytes;
    size_t g_off = v_off + v_bytes;
    size_t b_off = g_off + gb_bytes;
    size_t total_in = b_off + gb_bytes;

    size_t state_bytes = (size_t)n_v_heads * (size_t)d_state * (size_t)d_state * sizeof(float);
    size_t out_bytes   = (size_t)d_state * (size_t)n_v_heads * sizeof(float);

    if (!reserve_buf(&g_ssm, &g_ssmcap, &g_ssmptr, state_bytes,
                     MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache)) return 0;
    if (!reserve_buf(&g_x, &g_xcap, &g_xptr, total_in,
                     MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined)) return 0;
    if (!reserve_buf(&g_y, &g_ycap, &g_yptr, out_bytes,
                     MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache)) return 0;

    double _pc0 = g_prof ? prof_now_ms() : 0;
    memcpy(g_ssmptr, state, state_bytes);                 /* H2D state */
    memcpy((uint8_t *)g_xptr + q_off, q_conv,   q_bytes);
    memcpy((uint8_t *)g_xptr + k_off, k_conv,   k_bytes);
    memcpy((uint8_t *)g_xptr + v_off, v_conv,   v_bytes);
    memcpy((uint8_t *)g_xptr + g_off, gate_exp, gb_bytes);
    memcpy((uint8_t *)g_xptr + b_off, beta,     gb_bytes);

    id<MTLCommandBuffer> cmd = [g_queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_ssm_recurrence];
    [enc setBuffer:g_ssm offset:0 atIndex:0];              /* state (rw) */
    [enc setBuffer:g_x   offset:q_off atIndex:1];          /* q_conv     */
    [enc setBuffer:g_x   offset:k_off atIndex:2];          /* k_conv     */
    [enc setBuffer:g_x   offset:v_off atIndex:3];          /* v_conv     */
    [enc setBuffer:g_x   offset:g_off atIndex:4];          /* gate_exp   */
    [enc setBuffer:g_x   offset:b_off atIndex:5];          /* beta       */
    [enc setBuffer:g_y   offset:0   atIndex:6];            /* ssm_output */
    int32_t dims[3] = { n_v_heads, d_state, repeat };
    [enc setBytes:dims length:sizeof(dims) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_v_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(TPB, 1, 1)];
    [enc endEncoding];
    double _pt0 = g_prof ? prof_now_ms() : 0;
    [cmd commit];
    [cmd waitUntilCompleted];
    if (g_prof) {
        double _dt = prof_now_ms() - _pt0;
        g_prof_gpu_ms += _dt; g_prof_calls++; g_prof_rec_gpu_ms += _dt; g_prof_rec_calls++;
        g_prof_rec_copy_ms += _pt0 - _pc0;     /* copy-in time */
        prof_report();
    }

    double _pc1 = g_prof ? prof_now_ms() : 0;
    memcpy(state, g_ssmptr, state_bytes);                  /* D2H state */
    memcpy(ssm_output, g_yptr, out_bytes);
    if (g_prof) g_prof_rec_copy_ms += prof_now_ms() - _pc1; /* copy-out time */
    return 1;
}

}  // extern "C"
