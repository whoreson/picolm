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
#include "emu_prims.h"
/* the header's TQ dequant uses fa3_h2f; the emulator's software h2f IS the
 * IEEE binary16 decode the GPU performs, so reuse it. */
#define fa3_h2f h2f
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
            const int bsz = fa3_block_bytes(t);   /* 34 Q8_0 / 18 Q4_0 / 14 TQ3 / 18 TQ4 */
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
        const int bsz = fa3_block_bytes(t); const uint8_t *b = row + (i >> 5) * bsz; const int e = i & 31;
        uint16_t dr; memcpy(&dr, b, 2);
        float v;
        if (t == 1) { const float d = h2f(dr); v = (float)((const int8_t *)(b + 2))[e] * d; }
        else if (t == 2) { const float d = h2f(dr); const int nb = e < 16 ? (b[2 + e] & 0xF) : (b[2 + e - 16] >> 4); v = (float)(nb - 8) * d; }
        else {
            /* TQ3/TQ4: reuse the header's own inverse transform so this
             * reference is independent of the kernel's fp16 packing, only
             * sharing the (fixed) codebook/WHT definition. */
            float rot[32];
            uint8_t idx[8];
            for (int g = 0; g < 4; g++) {
                if (t == 3) fa3_tq_unpack3(idx, b + 2 + g * 3); else fa3_tq_unpack4(idx, b + 2 + g * 4);
                for (int k = 0; k < 8; k++)
                    rot[g * 8 + k] = (t == 3) ? tq3_codebook_dev[idx[k]] : tq4_codebook_dev[idx[k]];
            }
            for (int step = 1; step < 32; step <<= 1)
                for (int ii = 0; ii < 32; ii += step << 1)
                    for (int j = ii; j < ii + step; j++) {
                        const float a = rot[j], c = rot[j + step];
                        rot[j] = a + c; rot[j + step] = a - c;
                    }
            return h2f(dr) * rot[e] * 0.176776695f * tq3_signs_dev[e];
        }
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
    f |= run_case<256, 32>(3, 3, 7, 96, 6, 1, 0);       /* TQ3 */
    f |= run_case<256, 32>(4, 4, 13, 110, 6, 1, 0);     /* TQ4 */
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
        f |= run_case<128, 64>(3, 3, 0, 70, 2, 1, 0);   /* TQ3 */
        f |= run_case<128, 64>(4, 4, 5, 80, 2, 1, 0);   /* TQ4 */
        f |= run_case<256, 32>(3, 0, 4, 60, 4, 2, 0);   /* TQ3 K, F16 V: differing pos strides */
        f |= run_case<256, 32>(4, 1, 9, 90, 4, 2, 0);   /* TQ4 K, Q8_0 V */
        f |= run_case<64, 64>(0, 0, 0, 100, 2, 1, 0);
        f |= run_case<64, 64>(2, 1, 7, 66, 2, 1, 0);
    }
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
