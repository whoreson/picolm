// backend_metal.mm
//
// Apple Metal backend for PicoLM — implements the backend_gpu.h C ABI using the
// Metal framework directly (no MLX-C, no CMake, no external dependencies beyond
// what ships with macOS + the Command Line Tools).
//
// Build:  make metal      (clang++ -std=c++20 -fobjc-arc -framework Metal ...)
//
// Design mirrors backend_gpu.cu (CUDA/HIP): one device context, grow-only
// scratch buffers, idempotent tensor upload, per-tensor MTLBuffer handles.
// The host side implements backend_gpu.h unchanged; model.c / tensor.c /
// model.h are NOT modified — they were already written generically behind
// #ifdef PICOLM_GPU.
//
// The matmul kernels are hand-written Metal Shading Language (compiled from a
// source string at init), NOT direct ports of the CUDA device kernels. The
// device-side dequant is taken from PicoLM's CPU reference in quant.c (the
// source of truth for GGUF block layouts — the CUDA file's Q4_K path is
// unfinished and its Q4_0 nibble order is wrong). The high-traffic types use
// coalesced warp-cooperative loads + simd_sum reduction.
//
// Activation: same as CUDA — set PICOLM_GPU=1 in the environment. The whole
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
// Q4_0/Q5_K/Q6_K keep a per-element structure (their layouts are awkward to
// coalesce) but cache block metadata once and use simd_sum.
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
/* ---------------- Q8_0 (34B/32v) — COALESCED, 1 block/simdgroup -----------\n\
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
/* ---------------- Q4_K (144B/256v) — COALESCED, 1 block/simdgroup ---------\n\
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
/* ---------------- Q5_K (176B/256v) — metadata cached + simd_sum -----------\n\
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
/* ---------------- Q6_K (210B/256v) — metadata cached + simd_sum -----------\n\
 * block_q6_K: ql[128], qh[64], scales[16 int8], uint16 d (offset 208). */\n\
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
    float acc = 0.0f;\n\
    for (int bi = (int)tid; bi < nblocks; bi += TPB) {\n\
        device const uint8_t* blk = row + (unsigned long)(bi * 210);\n\
        float d = f16tof32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));\n\
        device const uint8_t* ql = blk;\n\
        device const uint8_t* qh = blk + 128;\n\
        device const int8_t*  sc = (device const int8_t*)(blk + 192);\n\
        float ds[16];\n\
        for (int i = 0; i < 16; i++) ds[i] = d * (float)sc[i];\n\
        for (int j = 0; j < 256; j++) {\n\
            int chunk = j / 128, within = j % 128;\n\
            int sub = within / 32, l = within % 32;\n\
            int ql_idx = (sub == 1 || sub == 3) ? (l + 32) : l;\n\
            uint8_t qlb = ql[chunk * 64 + ql_idx];\n\
            uint8_t qhb = qh[chunk * 32 + l];\n\
            int qh_shift = (sub == 0) ? 0 : (sub == 1) ? 2 : (sub == 2) ? 4 : 6;\n\
            int qraw = ((sub < 2) ? (int)(qlb & 0xF) : (int)(qlb >> 4))\n\
                     | (int)(((qhb >> qh_shift) & 3) << 4);\n\
            int sc_idx = chunk * 8 + l / 16 + 2 * sub;\n\
            acc += x[(unsigned long)s * I + bi * 256 + j] * (ds[sc_idx] * (float)(qraw - 32));\n\
        }\n\
    }\n\
    threadgroup float wb[NW];\n\
    float total = reduce_all(acc, wb, tid);\n\
    if (tid == 0) y[(unsigned long)s * O + o] = total;\n\
}\n\
\n\
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
";

#pragma mark - Device context (single global; Apple Silicon has one GPU)

// Under -fobjc-arc these static globals are strong (retained), released on
// program exit / shutdown. Apple Silicon exposes exactly one Metal device, so
// a single context suffices; the `device` ABI param is validated to be 0.
static id<MTLDevice>            g_device  = nil;
static id<MTLCommandQueue>       g_queue   = nil;
static id<MTLLibrary>            g_library = nil;
static id<MTLComputePipelineState> g_ps_f32, g_ps_f16, g_ps_q4_0, g_ps_q8_0,
                                  g_ps_q4_K, g_ps_q5_K, g_ps_q6_K, g_ps_silu;
static int g_ndev = 0;

// Grow-only scratch buffers (mirror CUDA's reserve()). Shared storage mode =
// unified memory: CPU writes x, GPU reads x & writes y, CPU reads y.
static id<MTLBuffer> g_x = nil, g_y = nil, g_gate = nil, g_up = nil;
static size_t g_xcap = 0, g_ycap = 0, g_gatecap = 0, g_upcap = 0;

static id<MTLComputePipelineState> ps_for_qtype(gguf_type_t q) {
    switch (q) {
        case GGUF_TYPE_F32:  return g_ps_f32;
        case GGUF_TYPE_F16:  return g_ps_f16;
        case GGUF_TYPE_Q4_0: return g_ps_q4_0;
        case GGUF_TYPE_Q8_0: return g_ps_q8_0;
        case GGUF_TYPE_Q4_K: return g_ps_q4_K;
        case GGUF_TYPE_Q5_K: return g_ps_q5_K;
        case GGUF_TYPE_Q6_K: return g_ps_q6_K;
        default:             return nil;   /* unsupported -> caller falls back to CPU */
    }
}

// Grow-only: only (re)allocates when current capacity is too small.
// Parameter qualified __strong so we can pass &g_x (a strong global); the
// default (__autoreleasing*) would reject address-of-strong-global under ARC.
static id<MTLBuffer> reserve_buf(__strong id<MTLBuffer> *buf, size_t *cap, size_t bytes) {
    if (*buf && *cap >= bytes) return *buf;
    *buf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!*buf) { *cap = 0; return nil; }
    *cap = bytes;
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
    g_ps_silu = [g_device newComputePipelineStateWithFunction:[g_library newFunctionWithName:@"silu_mul"] error:nil];
    if (!g_ps_f32 || !g_ps_f16 || !g_ps_q4_0 || !g_ps_q8_0 || !g_ps_q4_K || !g_ps_q5_K || !g_ps_q6_K || !g_ps_silu) {
        fprintf(stderr, "[Metal] pipeline state creation failed\n");
        g_device = nil; g_queue = nil; g_library = nil; return 0;
    }

    g_ndev = 1;
    fprintf(stderr, "[Metal] device: %s, working set %.1f GB\n",
            [[g_device name] UTF8String],
            (double)[g_device recommendedMaxWorkingSetSize] / 1e9);
    return 1;
}

void picolm_gpu_shutdown(void) {
    g_x = g_y = g_gate = g_up = nil;
    g_xcap = g_ycap = g_gatecap = g_upcap = 0;
    g_ps_f32 = g_ps_f16 = g_ps_q4_0 = g_ps_q8_0 = g_ps_q4_K = g_ps_q5_K = g_ps_q6_K = g_ps_silu = nil;
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

int picolm_gpu_tensor_upload(picolm_gpu_tensor_t **tensor,
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
     * rejects the region — e.g. some overlap cases — we fall back to copy.) */
    size_t page = (size_t)getpagesize();
    uintptr_t addr = (uintptr_t)weights;
    uintptr_t base_addr = addr & ~(uintptr_t)(page - 1);   /* round down to page */
    size_t pad_lo = (size_t)(addr - base_addr);
    size_t aligned_len = (total + pad_lo + page - 1) & ~(page - 1);  /* round up */
    buf = [g_device newBufferWithBytesNoCopy:(void *)base_addr
                                      length:aligned_len
                                      options:MTLResourceStorageModeShared |
                                               MTLResourceCPUCacheModeDefaultCache
                                      deallocator:^(void *ptr, NSUInteger len) {
                                          /* mmap'd memory is owned by the model; nothing to free. */
                                          (void)ptr; (void)len;
                                      }];
    if (buf) {
        zero_copy = 1;
        buf_offset = pad_lo;
    } else {
        /* Fallback: one-time copy into a shared-storage buffer (offset 0). */
        buf = [g_device newBufferWithBytes:weights length:total options:MTLResourceStorageModeShared];
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
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)O, (NSUInteger)S, 1)
        threadsPerThreadgroup:MTLSizeMake(TPB, 1, 1)];
    [enc endEncoding];
    return 1;
}

int picolm_gpu_matmul(picolm_gpu_tensor_t *t, float *y, const float *x, int S, int device) {
    if (!t || !y || !x || S < 1 || device != 0 || !g_device) return 0;
    int I = t->I, O = t->O;
    size_t xb = (size_t)S * I * sizeof(float);
    size_t yb = (size_t)S * O * sizeof(float);
    if (!reserve_buf(&g_x, &g_xcap, xb)) return 0;
    if (!reserve_buf(&g_y, &g_ycap, yb)) return 0;

    memcpy([g_x contents], x, xb);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    if (!launch_matmul(t, g_x, g_y, S, cmd)) { return 0; }
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(y, [g_y contents], yb);
    return 1;
}

int picolm_gpu_expert_mlp(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                           picolm_gpu_tensor_t *down, float *y, const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1) return 0;
    if (gate->device != up->device || gate->device != down->device || gate->device != 0) return 0;
    if (gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;

    int D = gate->I, I = gate->O;
    size_t xb = (size_t)S * D * sizeof(float);
    size_t ib = (size_t)S * I * sizeof(float);
    if (!reserve_buf(&g_x,    &g_xcap,    xb)) return 0;
    if (!reserve_buf(&g_y,    &g_ycap,    xb)) return 0;
    if (!reserve_buf(&g_gate, &g_gatecap, ib)) return 0;
    if (!reserve_buf(&g_up,   &g_upcap,   ib)) return 0;

    memcpy([g_x contents], x, xb);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
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
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(y, [g_y contents], xb);
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

}  // extern "C"
