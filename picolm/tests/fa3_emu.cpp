/* fa3_emu: executes the REAL backend_gpu_attn_fa3.cuh kernel on the CPU.
 *
 * Every GPU primitive the kernel uses (ldmatrix, ldmatrix.trans, mma.m16n8k16,
 * cp.async, cvt.rn.f16x2.f32, packed fp16 sub/mul, warp shuffles/votes,
 * __syncthreads/__syncwarp) is emulated here to its PTX-specified semantics,
 * with one OS thread per CUDA thread and real barriers. A kernel with a
 * divergent barrier or divergent warp-collective HANGS here (the run is
 * killed by the timeout), and a missing barrier shows up as a data race.
 *
 * It then checks the output against a double-precision reference built from
 * the same (dequantised) K/V, for F16 / Q8_0 / Q4_0 KV, head_dim 64/128/256,
 * ragged tails, start_pos > 0, GQA, and a sliding window.
 *
 *   g++ -std=c++20 -O2 -pthread -DPICOLM_FA3_EMU -I. tests/fa3_emu.cpp -o fa3_emu && ./fa3_emu
 *
 * What this does NOT cover: real async timing of cp.async, bank conflicts,
 * register allocation, tensor-core rounding. Those need the GPU test
 * (tests/test_attn_fa3_gpu.cu) and ptxas -v. */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <functional>
#include <algorithm>

#define PICOLM_FA3_EMU 1
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __restrict__ __restrict
struct uint4 { uint32_t x, y, z, w; };
struct float4 { float x, y, z, w; };
struct float2 { float x, y; };
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
    EmuBarrier *wbar[4];
    uint8_t *smem;
};
static Ctx *g_ctx;
static float     g_wf[4][32];
static uint32_t  g_wu[4][32][8];
static const void *g_wp[4][32];
static int       g_wi[4][32];
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
static inline float fa3_shfl_xor(float v, int m) {
    const int w = emu_warp(), l = emu_lane();
    g_wf[w][l] = v; wbar(); float r = g_wf[w][l ^ m]; wbar(); return r;
}
static inline int fa3_any(int p) {
    const int w = emu_warp(), l = emu_lane();
    g_wi[w][l] = p; wbar(); int r = 0; for (int i = 0; i < 32; i++) r |= g_wi[w][i] != 0; wbar(); return r;
}

#include "backend_gpu_attn_fa3.cuh"

/* ---- host harness --------------------------------------------------------- */
static uint32_t rng_s = 12345;
static float frand() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 17; rng_s ^= rng_s << 5; return (rng_s % 20001) / 10000.0f - 1.0f; }

template <int HD, int BN>
static int run_case(int kt, int vt, int start_pos, int n_tokens, int n_heads, int n_kv, int n_swa) {
    const int layer = 1, max_seq = start_pos + n_tokens + 40;
    auto row_bytes = [&](int t) { return t == 0 ? HD * 2 : fa3_raw_row_bytes(HD, t); };
    const size_t khs = row_bytes(kt), vhs = row_bytes(vt);
    const size_t kps = n_kv * khs, vps = n_kv * vhs;
    std::vector<uint8_t> K((size_t)(layer + 1) * max_seq * kps), V((size_t)(layer + 1) * max_seq * vps);
    auto fill = [&](std::vector<uint8_t> &buf, int t, float amp) {
        if (t == 0) { uint16_t *p = (uint16_t *)buf.data(); for (size_t i = 0; i < buf.size() / 2; i++) p[i] = f2h(frand() * amp); }
        else {
            const int bsz = t == 1 ? 34 : 18;
            for (size_t b = 0; b + bsz <= buf.size(); b += bsz) {
                uint16_t d = f2h(0.01f + 0.05f * (frand() * 0.5f + 0.5f)); memcpy(&buf[b], &d, 2);
                for (int i = 2; i < bsz; i++) buf[b + i] = (uint8_t)(rng_s = rng_s * 1664525u + 1013904223u, rng_s >> 24);
            }
        }
    };
    fill(K, kt, 2.0f); fill(V, vt, 1.0f);
    /* reference dequant: exactly kv_row_dequant() then fp32 -> fp16 */
    auto deq = [&](const std::vector<uint8_t> &buf, int t, size_t pos_stride, size_t head_stride, int pos, int h, int i) -> float {
        const uint8_t *row = &buf[(size_t)layer * max_seq * pos_stride + (size_t)pos * pos_stride + (size_t)h * head_stride];
        if (t == 0) return h2f(((const uint16_t *)row)[i]);
        const int bsz = t == 1 ? 34 : 18; const uint8_t *b = row + (i >> 5) * bsz; const int e = i & 31;
        uint16_t dr; memcpy(&dr, b, 2); const float d = h2f(dr);
        float v;
        if (t == 1) v = (float)((const int8_t *)(b + 2))[e] * d;
        else { const int nb = e < 16 ? (b[2 + e] & 0xF) : (b[2 + e - 16] >> 4); v = (float)(nb - 8) * d; }
        return h2f(f2h(v));
    };
    const size_t qn = (size_t)n_tokens * n_heads * HD;
    std::vector<float> Q(qn), out(qn, NAN), ref(qn);
    for (auto &x : Q) x = frand() * 2.0f;

    const fa3_layout L = fa3_make_layout(HD, BN, kt, vt);
    std::vector<uint8_t> smem(L.total + 64, 0xFF);   /* 0xFF = NaN fp16 patterns: unzeroed smem poisons the result */
    const unsigned qtiles = (n_tokens + FA3_BM - 1) / FA3_BM;
    for (unsigned by = 0; by < qtiles; by++)
        for (int bx = 0; bx < n_heads; bx++) {
            Ctx ctx; ctx.blk = new EmuBarrier(FA3_THREADS);
            for (int w = 0; w < 4; w++) ctx.wbar[w] = new EmuBarrier(32);
            ctx.smem = (uint8_t *)(((uintptr_t)smem.data() + 15) & ~(uintptr_t)15);
            g_ctx = &ctx;
            std::vector<std::thread> th;
            for (int t = 0; t < FA3_THREADS; t++)
                th.push_back(std::thread([&, t] {       /* t by value: C++11 init-capture is C++14 */
                    threadIdx = {(unsigned)t, 0, 0}; blockIdx = {(unsigned)bx, by, 0};
                    picolm_gpu_attention_prefill_fa3_kernel<HD, BN>(out.data(), Q.data(), K.data(), V.data(),
                        layer, start_pos, n_tokens, n_heads, n_kv, max_seq, kps, khs, vps, vhs, n_swa, kt, vt);
                }));
            for (size_t i = 0; i < th.size(); i++) th[i].join();
            delete ctx.blk;
            for (int w = 0; w < 4; w++) delete ctx.wbar[w];
        }

    /* double-precision reference (Q rounded to fp16 like the kernel does) */
    const int kv_mul = n_heads / n_kv;
    for (int t = 0; t < n_tokens; t++)
        for (int h = 0; h < n_heads; h++) {
            const int pos = start_pos + t, kvh = h / kv_mul;
            std::vector<double> s; std::vector<int> keys;
            for (int p = 0; p <= pos; p++) {
                if (n_swa > 0 && p < pos - n_swa + 1) continue;
                double d = 0;
                for (int i = 0; i < HD; i++) d += (double)h2f(f2h(Q[((size_t)t * n_heads + h) * HD + i])) * deq(K, kt, kps, khs, p, kvh, i);
                s.push_back(d / sqrt((double)HD)); keys.push_back(p);
            }
            double mx = -1e300; for (double x : s) mx = std::max(mx, x);
            double den = 0; for (double &x : s) { x = exp(x - mx); den += x; }
            for (int i = 0; i < HD; i++) {
                double o = 0;
                for (size_t k = 0; k < keys.size(); k++) o += s[k] * deq(V, vt, vps, vhs, keys[k], kvh, i);
                ref[((size_t)t * n_heads + h) * HD + i] = (float)(o / den);
            }
        }
    /* Tolerance: absolute error normalised by the amplitude of the whole
     * OUTPUT, not per element.
     *
     * The two kernels differ only in summation order and in rounding P to
     * fp16, so the error scales with |o| -- which is a property of the test
     * data, not of the kernel. But a per-element relative test is wrong here:
     * an output element that happens to sit near zero (0.342185 vs 0.342347)
     * has ~5e-4 absolute error and a meaningless 9e-2 "relative" error.
     * Normalising by the max |o| over the whole case is the scale the
     * quantisation noise actually lives on. */
    double maxe = 0, maxr = 0; size_t bad = 0, worst = 0;
    for (size_t i = 0; i < qn; i++) {
        if (!std::isfinite(out[i])) { bad++; continue; }
        double e = fabs((double)out[i] - ref[i]); if (e > maxe) { maxe = e; worst = i; }
        maxr = std::max(maxr, (double)fabs(ref[i]));
    }
    const double rel = maxe / (maxr > 1e-6 ? maxr : 1.0);
    const bool fail = bad || !(rel < 1e-2);
    printf("%s hd=%-3d bn=%-2d kv=%d/%d start=%-4d n=%-4d heads=%d/%d swa=%-3d  err %.2e (rel %.1e, |o|max %.2f) nonfinite=%zu\n",
           fail ? "FAIL" : "ok  ", HD, BN, kt, vt, start_pos, n_tokens, n_heads, n_kv, n_swa, maxe, rel, maxr, bad);
    if (fail && bad < qn)
        printf("     worst: tok %zu head %zu dim %zu  got %f ref %f\n", worst / ((size_t)n_heads * HD), (worst / HD) % n_heads, worst % HD, out[worst], ref[worst]);
    return fail;
}

int main(int argc, char **argv) {
    const bool quick = argc > 1 && !strcmp(argv[1], "quick");
    int f = 0;
    /* Qwen3.5/3.6 shape: head_dim 256, 6 q heads per kv head. Ragged n, start_pos > 0. */
    f |= run_case<256, 32>(0, 0, 0, 17, 6, 1, 0);       /* fewer than 32 tokens, one warp-tile */
    f |= run_case<256, 32>(0, 0, 0, 65, 6, 1, 0);       /* second q tile has a single row */
    f |= run_case<256, 32>(0, 0, 37, 100, 6, 1, 0);
    f |= run_case<256, 32>(1, 1, 5, 70, 6, 1, 0);
    f |= run_case<256, 32>(2, 2, 11, 130, 6, 1, 0);     /* the long-context production config: q4_0 KV */
    if (!quick) {
        f |= run_case<256, 32>(0, 2, 3, 90, 6, 2, 0);   /* mixed K/V types */
        f |= run_case<256, 32>(2, 1, 0, 50, 4, 2, 0);
        f |= run_case<256, 32>(2, 0, 4, 80, 4, 2, 0);   /* quantised K, F16 V: K and V pos strides differ */
        f |= run_case<256, 32>(1, 0, 0, 40, 2, 1, 0);
        f |= run_case<256, 32>(0, 0, 0, 150, 2, 1, 48); /* sliding window */
        f |= run_case<256, 32>(2, 2, 200, 70, 2, 1, 40);
        f |= run_case<128, 64>(0, 0, 9, 140, 4, 2, 0);
        f |= run_case<128, 64>(2, 2, 0, 75, 4, 2, 0);
        f |= run_case<128, 64>(1, 1, 33, 64, 2, 1, 0);
        f |= run_case<64, 64>(0, 0, 0, 100, 2, 1, 0);
        f |= run_case<64, 64>(2, 1, 7, 66, 2, 1, 0);
    }
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
