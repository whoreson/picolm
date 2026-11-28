/* test_gemm_f16_gpu: the fp16 tensor-core prefill GEMM (picolm_gpu_gemm_f16_launch)
 * for Q5_K / Q6_K / Q8_0 weights.
 *
 * Correctness: sampled outputs (a full CPU reference at Qwen shapes would take
 * minutes) are compared with a double-precision reference built from the project's
 * own CPU dequantisers in quant.c, with fp16-rounded X and W exactly like the kernel.
 * Pass = worst error relative to the sampled token row's own scale <= 5e-6, and every
 * output finite (y is pre-filled with NaN, so an unwritten element fails).
 *
 * Comparison: for Q5_K and Q6_K the old int8 kernels (Q8_0 activations + per-sub-block
 * float epilogue) run on the same data, so the table shows BOTH kernels' error against
 * the same oracle and their times. The int8 path's error is dominated by activation
 * quantisation (~1e-3 relative); the fp16 path should sit around 1e-6.
 *
 *   make test-gemm-f16-gpu && ./test_gemm_f16_gpu               (needs sm_80+) */
#include "backend_gpu_kernels.cuh"
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
extern "C" {
#include "quant.h"
}

static uint32_t rs = 4242;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float frand() { return (rnd() % 20001) / 10000.0f - 1.0f; }
static uint16_t f2h(float f) { __half h = __float2half(f); return *(uint16_t *)&h; }
static float h2f(uint16_t b) { __half h = *(__half *)&b; return __half2float(h); }

static size_t row_bytes_of(int fmt, int I) {
    switch (fmt) {
    case 5: return (size_t)(I / 256) * 176;   /* Q5_K */
    case 6: return (size_t)(I / 256) * 210;   /* Q6_K */
    case 7: return (size_t)(I / 256) * 212;   /* IQ6_K */
    case 9: return (size_t)(I / 256) * 144;   /* IQ4_K */
    default: return (size_t)(I / 32) * 34;    /* Q8_0 */
    }
}

static void make_weights(std::vector<uint8_t> &w, int fmt, int I, int O) {
    const size_t rb = row_bytes_of(fmt, I);
    w.resize((size_t)O * rb);
    for (size_t i = 0; i < w.size(); i++) w[i] = (uint8_t)(rnd() >> 11);
    for (int o = 0; o < O; o++)
        for (size_t b = 0; b < rb;) {
            uint8_t *blk = &w[(size_t)o * rb + b];
            const uint16_t d = f2h(0.0005f + 0.003f * (frand() * 0.5f + 0.5f));
            const uint16_t dm = f2h(0.0005f + 0.003f * (frand() * 0.5f + 0.5f));
            if (fmt == 5) { memcpy(blk, &d, 2); memcpy(blk + 2, &dm, 2); b += 176; }
            else if (fmt == 6) { memcpy(blk + 208, &d, 2); for (int k = 0; k < 16; k++) blk[192 + k] = (uint8_t)(int8_t)((int)(rnd() % 41) - 20); b += 210; }
            else if (fmt == 7) { /* IQ6_K: d at 0, uint16 extra (any bits valid), int8 scales at 4 */
                memcpy(blk, &d, 2);
                for (int k = 0; k < 16; k++) blk[4 + k] = (uint8_t)(int8_t)((int)(rnd() % 41) - 20);
                b += 212; }
            else if (fmt == 9) { /* IQ4_K: d at 0, uint16 extra, scales_h[4]+scales_l[8]
                                  * (random bytes cover all 6-bit scale / LUT-select
                                  * combinations); qs stays random. */
                memcpy(blk, &d, 2);
                b += 144; }
            else { memcpy(blk, &d, 2); b += 34; }
        }
}

/* worst error of y relative to its token row's scale, over a sample of (s, o) pairs */
static void check(const float *y, int ys, const std::vector<float> &x, int S, int I, int O, int fmt,
                  const std::vector<uint8_t> &w, double *worst_rel, size_t *nonfinite_total, const float *y_full_scan, size_t n_full) {
    const size_t rb = row_bytes_of(fmt, I);
    const int NS = std::min(S, 48), NO = std::min(O, 48);
    std::vector<int> ss(NS), oo(NO);
    for (int i = 0; i < NS; i++) ss[i] = (int)(((long)i * 2654435761u + 7) % S);
    for (int i = 0; i < NO; i++) oo[i] = (int)(((long)i * 40503u + 11) % O);
    std::vector<std::vector<float>> Wr(NO, std::vector<float>(I));
    for (int j = 0; j < NO; j++) {
        const uint8_t *r = &w[(size_t)oo[j] * rb];
        if (fmt == 5) dequantize_row_q5_K(r, Wr[j].data(), I); else if (fmt == 6) dequantize_row_q6_K(r, Wr[j].data(), I);
        else if (fmt == 7) dequantize_row_iq6_k(r, Wr[j].data(), I);
        else if (fmt == 9) dequantize_row_iq4_k(r, Wr[j].data(), I); else dequantize_row_q8_0(r, Wr[j].data(), I);
        for (int k = 0; k < I; k++) Wr[j][k] = h2f(f2h(Wr[j][k]));
    }
    *worst_rel = 0; *nonfinite_total = 0;
    for (int i = 0; i < NS; i++) {
        std::vector<double> ref(NO); double rmax = 0;
        for (int j = 0; j < NO; j++) {
            double a = 0;
            for (int k = 0; k < I; k++) a += (double)h2f(f2h(fminf(65504.f, fmaxf(-65504.f, x[(size_t)ss[i] * I + k])))) * (double)Wr[j][k];
            ref[j] = a; rmax = std::max(rmax, fabs(a));
        }
        for (int j = 0; j < NO; j++) {
            const float g = y[(size_t)ss[i] * ys + oo[j]];
            if (!isfinite(g)) { (*nonfinite_total)++; continue; }
            *worst_rel = std::max(*worst_rel, fabs((double)g - ref[j]) / (rmax > 0 ? rmax : 1.0));
        }
    }
    if (y_full_scan) for (size_t i = 0; i < n_full; i++) if (!isfinite(y_full_scan[i])) (*nonfinite_total)++;
}

static int run(int fmt, int S, int I, int O, const char *label) {
    std::vector<uint8_t> w; make_weights(w, fmt, I, O);
    std::vector<float> x((size_t)S * I);
    for (auto &v : x) v = frand() * 2.0f;
    const size_t rb = row_bytes_of(fmt, I);
    uint8_t *dw; float *dx, *dy;
    cudaMalloc(&dw, w.size()); cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dy, (size_t)S * O * 4);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice); cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> y((size_t)S * O);
    const double flop = 2.0 * S * I * O;

    /* ---- new fp16 GEMM ---- */
    cudaMemset(dy, 0xFF, (size_t)S * O * 4);
    int launched = picolm_gpu_gemm_f16_launch(0, fmt, dy, dx, dw, S, I, O, rb, 0, 0);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
    double rel_new; size_t nf_new;
    check(y.data(), O, x, S, I, O, fmt, w, &rel_new, &nf_new, y.data(), y.size());
    const int iters = 10;
    cudaEventRecord(a); for (int i = 0; i < iters; i++) picolm_gpu_gemm_f16_launch(0, fmt, dy, dx, dw, S, I, O, rb, 0, 0); cudaEventRecord(b);
    cudaEventSynchronize(b); float ms_new = 0; cudaEventElapsedTime(&ms_new, a, b); ms_new /= iters;

    /* ---- old int8 kernels (Q5_K single-block-superblock "sb", Q6_K multi-tile).
     * IQ6_K has no int8 IMMA kernel (its old path is the generic per-element
     * dequant matmul), and Q8_0's old path is the activation-quantizing IMMA
     * kernel measured by the Q8_0 rows of bench_q8_gemm -- neither is wired
     * here, so only the fp16 GEMM is timed for fmts 7 and 8. ---- */
    double rel_old = -1, ms_old = -1; size_t nf_old = 0;
    if (fmt == 5 || fmt == 6) {
        const int nb = I / 32, Sp = (S + 15) & ~15;
        int8_t *xq; float *xd;
        cudaMalloc(&xq, (size_t)Sp * I); cudaMalloc(&xd, (size_t)Sp * nb * 4);
        auto old_launch = [&]() {
            cudaMemset(xq, 0, (size_t)Sp * I); cudaMemset(xd, 0, (size_t)Sp * nb * 4);
            picolm_quantize_q8_0<<<dim3(nb, S), 32, 32 * sizeof(float)>>>(xq, xd, dx, I, S);
            if (fmt == 5) {
                dim3 grid((O + 8 * Q5KW_WARPS - 1) / (8 * Q5KW_WARPS), (S + 16 * Q5K_TILES_S - 1) / (16 * Q5K_TILES_S));
                picolm_q5_k_q8_matmul_imma_sb<<<grid, 32 * Q5KW_WARPS>>>(dy, xq, xd, dw, S, I, O, (int)rb, O);
            } else {
                dim3 grid((O + 7) / 8, (S + 16 * Q6K_TILES_S - 1) / (16 * Q6K_TILES_S));
                picolm_q6_k_q8_matmul_imma_mt<<<grid, 32, Q6K_TILES_S * 32 * 4 * sizeof(float)>>>(dy, xq, xd, dw, S, I, O, (int)rb, O);
            }
        };
        cudaMemset(dy, 0xFF, (size_t)S * O * 4);
        old_launch(); cudaDeviceSynchronize();
        cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
        check(y.data(), O, x, S, I, O, fmt, w, &rel_old, &nf_old, NULL, 0);
        cudaEventRecord(a); for (int i = 0; i < iters; i++) old_launch(); cudaEventRecord(b);
        cudaEventSynchronize(b); float ms = 0; cudaEventElapsedTime(&ms, a, b); ms_old = ms / iters;
        cudaFree(xq); cudaFree(xd);
    }
    const int fail = !launched || err != cudaSuccess || nf_new || !(rel_new <= 5e-6);
    printf("%s fmt=%d %-14s S=%-4d I=%-5d O=%-5d | fp16 gemm %7.3f ms %6.1f TFLOP/s err %.1e", fail ? "FAIL" : "ok  ", fmt, label, S, I, O,
           ms_new, flop / (ms_new * 1e-3) / 1e12, rel_new);
    if (ms_old > 0) printf(" | int8 %7.3f ms %5.1f TFLOP/s err %.1e | %4.2fx", ms_old, flop / (ms_old * 1e-3) / 1e12, rel_old, ms_old / ms_new);
    if (!launched) printf("  [launcher returned 0]"); else if (err != cudaSuccess) printf("  %s", cudaGetErrorString(err)); else if (nf_new) printf("  nonfinite=%zu", nf_new);
    printf("\n");
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    return fail;
}

int main(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (sm_%d%d, %d SMs), max opt-in smem/block %zu B\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.sharedMemPerBlockOptin);
    if (prop.major < 8) { printf("needs sm_80+: skipped\n"); return 0; }
    fp16_table_init();     /* quant.c's Q6_K / Q8_0 dequantisers read d through a lookup table */
    int f = 0;
    const int fmts[5] = {5, 6, 7, 9, 8};
    for (int i = 0; i < 5; i++) {
        const int fmt = fmts[i];
        /* correctness: ragged shapes, then the real Qwen3.6-27B projections */
        f |= run(fmt, 100, 512, 136, "ragged");
        f |= run(fmt, 33, 768, 200, "ragged");
        f |= run(fmt, 512, 5120, 6144, "ssm in-proj");
        f |= run(fmt, 512, 5120, 17408, "ffn up/gate");
        f |= run(fmt, 512, 17408, 5120, "ffn down");
        f |= run(fmt, 512, 5120, 5120, "attn o-proj");
        f |= run(fmt, 86, 5120, 17408, "bench chunk");
        f |= run(fmt, 2048, 5120, 17408, "big batch");
    }
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
