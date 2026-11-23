/* test_attn_fa3_gpu: the tensor-core FA3 prefill kernel (picolm_gpu_attention_
 * prefill_fa3_launch) must agree with the trusted scalar picolm_gpu_attention_
 * prefill_warpgrp_kernel to within fp16 rounding (max abs diff < 4e-3), for
 * F16 / Q8_0 / Q4_0 KV, head_dim 256 and 128, GQA, start_pos > 0, ragged tails
 * and a sliding window. Mixed K/V types are covered by tests/fa3_emu.cpp (the
 * reference kernel takes a single kv_type).
 *
 * The output is poisoned with 0xFF (NaN) before the FA3 launch, so an element
 * the kernel never writes fails instead of reading stale data.
 *
 * It also times both kernels and prints achieved attention TFLOP/s.
 *
 *   make test-attn-fa3-gpu && ./test_attn_fa3_gpu            (needs sm_80+) */
#include "backend_gpu_kernels.cuh"
#include "backend_gpu_attn_fa3.cuh"
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t rng = 1234;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (rng % 20001) / 10000.0f - 1.0f; }
static uint16_t f2h(float f) { __half h = __float2half(f); return *(uint16_t *)&h; }

static size_t head_bytes(int hd, int t) {
    if (t == 0) return (size_t)hd * 2;
    return (size_t)(hd / 32) * (size_t)fa3_block_bytes(t);   /* 34/18/14/18 */
}

static void fill_kv(uint8_t *buf, size_t bytes, int t, float amp) {
    if (t == 0) {
        for (size_t i = 0; i + 1 < bytes; i += 2) { uint16_t h = f2h(frand() * amp); memcpy(buf + i, &h, 2); }
    } else {
        const int bsz = fa3_block_bytes(t);
        for (size_t b = 0; b + bsz <= bytes; b += bsz) {
            uint16_t d = f2h(0.01f + 0.05f * (frand() * 0.5f + 0.5f)); memcpy(buf + b, &d, 2);
            for (int i = 2; i < bsz; i++) { rng = rng * 1664525u + 1013904223u; buf[b + i] = (uint8_t)(rng >> 24); }
        }
    }
}

static int run(int HD, int kvt, int start_pos, int n_tokens, int n_heads, int n_kv, int n_swa, int bench) {
    const int layer = 1, max_seq = start_pos + n_tokens + 64;
    const size_t hs = head_bytes(HD, kvt), ps = (size_t)n_kv * hs;
    const size_t kvb = (size_t)(layer + 1) * max_seq * ps;
    uint8_t *hk = (uint8_t *)malloc(kvb), *hv = (uint8_t *)malloc(kvb);
    fill_kv(hk, kvb, kvt, 2.0f); fill_kv(hv, kvb, kvt, 1.0f);
    const size_t qn = (size_t)n_tokens * n_heads * HD;
    float *hq = (float *)malloc(qn * 4), *o0 = (float *)malloc(qn * 4), *o1 = (float *)malloc(qn * 4);
    for (size_t i = 0; i < qn; i++) hq[i] = frand() * 2;
    uint8_t *dk, *dv; float *dq, *dy;
    cudaMalloc(&dk, kvb); cudaMalloc(&dv, kvb); cudaMalloc(&dq, qn * 4); cudaMalloc(&dy, qn * 4);
    cudaMemcpy(dk, hk, kvb, cudaMemcpyHostToDevice); cudaMemcpy(dv, hv, kvb, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, hq, qn * 4, cudaMemcpyHostToDevice);

    /* reference: scalar warpgrp kernel (layer offset is applied inside via layer_ordinal) */
    const int tile_q = 16;
    const size_t sh0 = (size_t)ATTN_TILE_K * HD * 2 * 2 + (size_t)tile_q * HD * 4 + 2 * tile_q * 4;
    cudaFuncSetAttribute((const void *)picolm_gpu_attention_prefill_warpgrp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh0);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float ms0 = 0, ms1 = 0;
    auto ref_launch = [&]() {
        picolm_gpu_attention_prefill_warpgrp_kernel<<<dim3(n_heads, (n_tokens + tile_q - 1) / tile_q), 256, sh0>>>(
            dy, dq, (const uint16_t *)dk, (const uint16_t *)dv, layer, start_pos, n_tokens, n_heads, n_kv, HD, max_seq,
            ps, hs, hs, tile_q, n_swa, kvt);
    };
    ref_launch(); cudaDeviceSynchronize();
    cudaMemcpy(o0, dy, qn * 4, cudaMemcpyDeviceToHost);
    const int iters = bench ? 5 : 1;
    cudaEventRecord(a); for (int i = 0; i < iters; i++) ref_launch(); cudaEventRecord(b); cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms0, a, b); ms0 /= iters;

    /* FA3 */
    cudaMemset(dy, 0xFF, qn * 4);
    auto fa3_launch = [&]() {
        return picolm_gpu_attention_prefill_fa3_launch(0, dy, dq, dk, dv, layer, start_pos, n_tokens, n_heads, n_kv,
                                                       HD, max_seq, ps, hs, ps, hs, n_swa, kvt, kvt);
    };
    int launched = fa3_launch(); cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    cudaMemcpy(o1, dy, qn * 4, cudaMemcpyDeviceToHost);
    cudaEventRecord(a); for (int i = 0; i < iters; i++) fa3_launch(); cudaEventRecord(b); cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms1, a, b); ms1 /= iters;

    /* The tolerance has to be RELATIVE to the output amplitude, not a fixed
     * absolute bound. The two kernels differ only in summation order and in
     * rounding P to fp16, so the error scales with |o|. That amplitude is a
     * property of the TEST DATA, not of the kernel: the filler below uses the
     * same block scale for every format, so a Q8_0 cache holds +-128*d while a
     * Q4_0 cache holds +-8*d -- a factor of 16 -- and outputs differ by the
     * same factor. A fixed 4e-3 threshold therefore passes Q4_0 and fails
     * Q8_0 at 6e-3 on |o| of 7.4, which is 8e-4 relative, i.e. fine.
     *
     * 1e-2 relative is ~20x the worst observed (5e-4 for F16) and still far
     * below any real disagreement. */
    double max_abs = 0, max_ref = 0; size_t worst = 0, nan_cnt = 0;
    for (size_t i = 0; i < qn; i++) {
        if (isnan(o1[i]) || isinf(o1[i])) { nan_cnt++; continue; }
        double e = fabs(o0[i] - o1[i]);
        if (e > max_abs) { max_abs = e; worst = i; }
        if (fabs(o0[i]) > max_ref) max_ref = fabs(o0[i]);
    }
    /* Normalise by the amplitude of the whole case, not per element: an
     * output element near zero has a large per-element ratio for a
     * negligible absolute error. */
    const double max_rel = max_abs / (max_ref > 1e-6 ? max_ref : 1.0);
    const int fail = !launched || err != cudaSuccess || nan_cnt || !(max_rel < 1e-2);
    const double ctx_sum = (double)n_tokens * start_pos + (double)n_tokens * n_tokens / 2.0;   /* causal (token,key) pairs */
    const double flops = 4.0 * HD * n_heads * ctx_sum;
    printf("%s hd=%-3d kv=%d start=%-5d n=%-5d heads=%d/%d swa=%-3d  scalar %8.2f ms  fa3 %7.2f ms (%5.1fx, %5.1f TFLOP/s)  err %.1e (rel %.1e, |o| %.2f) nonfinite=%zu%s\n",
           fail ? "FAIL" : "ok  ", HD, kvt, start_pos, n_tokens, n_heads, n_kv, n_swa, ms0, ms1, ms0 / ms1,
           flops / (ms1 * 1e-3) / 1e12, max_abs, max_rel, max_ref, nan_cnt,
           !launched ? "  [launcher returned 0]" : err != cudaSuccess ? cudaGetErrorString(err) : "");
    if (fail && nan_cnt < qn)
        printf("   worst element %zu (tok %zu head %zu dim %zu): ref %f fa3 %f\n", worst,
               worst / ((size_t)n_heads * HD), (worst / HD) % n_heads, worst % HD, o0[worst], o1[worst]);
    cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dy); free(hk); free(hv); free(hq); free(o0); free(o1);
    return fail;
}

int main(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (sm_%d%d), max opt-in smem/block %zu B\n", prop.name, prop.major, prop.minor, prop.sharedMemPerBlockOptin);
    if (prop.major < 8) { printf("needs sm_80+ (mma.m16n8k16, cp.async): skipped\n"); return 0; }
    int f = 0;
    for (int kvt = 0; kvt <= 4; kvt++) {
        /* Qwen3.5/3.6 shape (head_dim 256, 24 q heads / 4 kv heads), ragged tails first */
        f |= run(256, kvt, 0, 32, 24, 4, 0, 0);
        f |= run(256, kvt, 0, 33, 24, 4, 0, 0);
        f |= run(256, kvt, 0, 100, 24, 4, 0, 0);
        f |= run(256, kvt, 37, 200, 24, 4, 0, 0);
        f |= run(256, kvt, 512, 512, 24, 4, 0, 0);     /* second ubatch of a prompt */
        f |= run(256, kvt, 0, 300, 8, 4, 64, 0);       /* sliding window */
        f |= run(128, kvt, 9, 140, 8, 2, 0, 0);
    }
    /* performance: a ubatch at depth, the case the benchmark shows collapsing */
    printf("--- performance (512-token ubatch at depth, Qwen shape) ---\n");
    for (int kvt = 0; kvt <= 4; kvt++) {
        f |= run(256, kvt, 0, 512, 24, 4, 0, 1);
        f |= run(256, kvt, 4096, 512, 24, 4, 0, 1);
        f |= run(256, kvt, 16384, 512, 24, 4, 0, 1);
    }
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
