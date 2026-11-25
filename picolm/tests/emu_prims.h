/* emu_prims.h: CPU emulation of the GPU primitives used by the hand-written mma
 * kernels (backend_gpu_mma.cuh): ldmatrix(.trans), mma.m16n8k16, cp.async,
 * cvt.rn.f16x2.f32, packed fp16 sub/mul, warp shuffles/votes, __syncthreads.
 * One OS thread per CUDA thread, real barriers. Include BEFORE the kernel header,
 * with -DPICOLM_FA3_EMU. Shared by tests/fa3_emu.cpp and tests/gemm_emu.cpp. */
#ifndef EMU_PRIMS_H
#define EMU_PRIMS_H
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <functional>
#include <algorithm>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __restrict__ __restrict
struct uint4 { uint32_t x, y, z, w; };
struct float4 { float x, y, z, w; };
struct float2 { float x, y; };
struct uint2 { uint32_t x, y; };
static inline float4 make_float4(float x, float y, float z, float w) { float4 r = {x, y, z, w}; return r; }
static inline uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { return {x, y, z, w}; }
static inline float2 make_float2(float x, float y) { return {x, y}; }
struct Dim3 { unsigned x, y, z; };
thread_local Dim3 threadIdx, blockIdx;

/* ---- block / warp execution context --------------------------------------
 * A hand-rolled barrier instead of C++20 std::barrier: this file must build
 * with -std=c++11 on any host with threads, and only the emulator's own
 * correctness depends on it. Simple sense-reversing barrier over mutex+cv,
 * which is all the kernel's __syncthreads()/__syncwarp() need. */
#include <mutex>
#include <condition_variable>
struct EmuBarrier {
    std::mutex m;
    std::condition_variable cv;
    unsigned n, count, gen;
    explicit EmuBarrier(unsigned threads) : n(threads), count(0), gen(0) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m);
        const unsigned g = gen;
        if (++count == n) { count = 0; gen++; cv.notify_all(); return; }
        while (gen == g) cv.wait(lk);
    }
};
struct Ctx {
    EmuBarrier *blk;
    EmuBarrier *wbar[8];       /* up to 8 warps (the GEMM uses 8, FA3 uses 4) */
    uint8_t *smem;
};
static Ctx *g_ctx;
static float     g_wf[8][32];
static uint32_t  g_wu[8][32][8];
static const void *g_wp[8][32];
static int       g_wi[8][32];
static inline int emu_warp() { return threadIdx.x >> 5; }
static inline int emu_lane() { return threadIdx.x & 31; }
static inline void wbar() { g_ctx->wbar[emu_warp()]->wait(); }
#define __syncthreads() g_ctx->blk->wait()
#define FA3_DYN_SMEM_PTR(name) uint8_t *name = g_ctx->smem

/* ---- fp16 helpers ---------------------------------------------------------
 * Software conversions rather than _Float16: gcc on x86 only exposes _Float16
 * with AVX512-FP16 support (gcc 12+), and this test has to build on anything.
 * They implement exactly the PTX/IEEE binary16 semantics the kernel relies on:
 * f2h is round-to-nearest-even from binary32 (what cvt.rn.f16x2.f32 does), and
 * h2f is exact. Subnormals and overflow are handled, because the dequant path
 * multiplies small scales by small codes and would otherwise disagree with the
 * GPU in exactly the cases the test is meant to catch. */
static inline uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mag = x & 0x7FFFFFFFu;
    if (mag >= 0x7F800000u)                       /* Inf / NaN */
        return (uint16_t)(sign | (mag > 0x7F800000u ? 0x7E00u : 0x7C00u));
    if (mag >= 0x47800000u)                       /* overflows binary16 range */
        return (uint16_t)(sign | 0x7C00u);
    if (mag < 0x38800000u) {                      /* subnormal or zero */
        if (mag < 0x33000000u) return (uint16_t)sign;
        const int e = (int)(mag >> 23) - 127;
        uint32_t m = (mag & 0x7FFFFFu) | 0x800000u;
        const int shift = 13 - e - 1 + 1;         /* to a 1.10 fixed-point */
        uint32_t q = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1u))) q++;
        return (uint16_t)(sign | q);
    }
    uint32_t m = (mag - 0x38000000u) >> 13;       /* rebase exponent, keep 10 bits */
    const uint32_t rem = (mag >> 13) & 0x1u;
    if (rem && ((mag & 0x1FFFu) > 0x1000u || ((mag & 0x1FFFu) == 0x1000u && (m & 1u)))) m++;
    return (uint16_t)(sign | m);
}
static inline float h2f(uint16_t b) {
    const uint32_t sign = (uint32_t)(b & 0x8000u) << 16;
    const uint32_t exp  = (b >> 10) & 0x1Fu;
    const uint32_t man  = b & 0x3FFu;
    uint32_t r;
    if (exp == 0) {
        if (man == 0) r = sign;
        else {                                    /* subnormal: normalise */
            int e = -1; uint32_t m = man;
            do { m <<= 1; e++; } while (!(m & 0x400u));
            r = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        r = sign | 0x7F800000u | (man << 13);
    } else {
        r = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f; memcpy(&f, &r, 4); return f;
}

/* ---- PTX primitive emulation (same names the kernel uses) ----------------- */
static inline void ldsm_impl(uint32_t r[4], const void *p, bool trans) {
    const int w = emu_warp(), l = emu_lane();
    g_wp[w][l] = p; wbar();
    for (int j = 0; j < 4; j++) {
        if (!trans) {
            r[j] = *(const uint32_t *)((const uint8_t *)g_wp[w][8 * j + (l >> 2)] + (l & 3) * 4);
        } else {
            const int rr = 2 * (l & 3), c = l >> 2;
            const uint16_t lo = *(const uint16_t *)((const uint8_t *)g_wp[w][8 * j + rr]     + c * 2);
            const uint16_t hi = *(const uint16_t *)((const uint8_t *)g_wp[w][8 * j + rr + 1] + c * 2);
            r[j] = (uint32_t)lo | ((uint32_t)hi << 16);
        }
    }
    wbar();
}
static inline void fa3_ldsm_x4(uint32_t r[4], const void *p)       { ldsm_impl(r, p, false); }
static inline void fa3_ldsm_x4_trans(uint32_t r[4], const void *p) { ldsm_impl(r, p, true); }
static inline void fa3_mma(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    const int w = emu_warp(), l = emu_lane(), g = l >> 2, t = l & 3;
    for (int i = 0; i < 4; i++) g_wu[w][l][i] = a[i];
    g_wu[w][l][4] = b0; g_wu[w][l][5] = b1;
    wbar();
    auto A = [&](int r, int k) {   /* a0:(g,2t..) a1:(g+8,2t..) a2:(g,2t+8..) a3:(g+8,2t+8..) */
        uint32_t word = g_wu[w][(r & 7) * 4 + (k & 7) / 2][(r >> 3) + 2 * (k >> 3)];
        return h2f((uint16_t)((word >> (16 * (k & 1))) & 0xFFFF));
    };
    auto B = [&](int k, int n) {   /* b0:(k=2t..,n=g)  b1:(k=2t+8..,n=g) */
        uint32_t word = g_wu[w][n * 4 + (k & 7) / 2][4 + (k >> 3)];
        return h2f((uint16_t)((word >> (16 * (k & 1))) & 0xFFFF));
    };
    float res[4];
    for (int i = 0; i < 4; i++) {
        const int row = g + (i >> 1) * 8, col = 2 * t + (i & 1);
        double acc = d[i];
        for (int k = 0; k < 16; k++) acc += (double)A(row, k) * (double)B(k, col);
        res[i] = (float)acc;
    }
    wbar();
    for (int i = 0; i < 4; i++) d[i] = res[i];
}
static inline void fa3_cp16(void *dst, const void *src) { memcpy(dst, src, 16); }
static inline void fa3_cp4(void *dst, const void *src)  { memcpy(dst, src, 4); }
static inline void fa3_cp_commit() {}
static inline void fa3_cp_wait1() {}
static inline uint32_t fa3_pack2(float lo, float hi) { return (uint32_t)f2h(lo) | ((uint32_t)f2h(hi) << 16); }
static inline uint32_t fa3_h2sub(uint32_t a, uint32_t b) {
    return (uint32_t)f2h(h2f(a & 0xFFFF) - h2f(b & 0xFFFF)) | ((uint32_t)f2h(h2f(a >> 16) - h2f(b >> 16)) << 16);
}
static inline uint32_t fa3_h2mul(uint32_t a, uint32_t b) {
    return (uint32_t)f2h(h2f(a & 0xFFFF) * h2f(b & 0xFFFF)) | ((uint32_t)f2h(h2f(a >> 16) * h2f(b >> 16)) << 16);
}
static inline float fa3_f16_to_f32(uint32_t bits) { return h2f((uint16_t)bits); }
static inline float fa3_shfl_xor(float v, int m) {
    const int w = emu_warp(), l = emu_lane();
    g_wf[w][l] = v; wbar(); float r = g_wf[w][l ^ m]; wbar(); return r;
}
static inline int fa3_any(int p) {
    const int w = emu_warp(), l = emu_lane();
    g_wi[w][l] = p; wbar(); int r = 0; for (int i = 0; i < 32; i++) r |= g_wi[w][i] != 0; wbar(); return r;
}


#endif /* EMU_PRIMS_H */
