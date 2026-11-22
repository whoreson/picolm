/* test_attn_fa3_gpu: picolm_gpu_attention_prefill_fa3_kernel (register-resident
 * O, head_dim 256, tensor-core scores) must agree with the trusted scalar
 * picolm_gpu_attention_prefill_warpgrp_kernel to within fp16 rounding
 * (max abs diff < 1e-2 on outputs of magnitude ~1), over the Qwen3.5/3.6 shape
 * (24 heads / 4 KV heads, head_dim 256), start_pos > 0, and ragged tails --
 * the n_tokens that are NOT a multiple of 64 are the cases that exposed the
 * divergent-barrier and per-warp-horizon bugs.
 *
 * The output buffer is poisoned with 0xFF (NaN) before the FA3 launch, so an
 * element the kernel never writes fails instead of reading stale data.
 *
 *   make test-attn-fa3-gpu && ./test_attn_fa3_gpu            (needs sm_80+) */
#include "backend_gpu_kernels.cuh"
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static uint32_t rng = 1234;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (rng % 20001) / 10000.0f - 1.0f; }
static int run(int start_pos, int n_tokens, int n_heads, int n_kv, int n_swa) {
    const int D = 256, TK = 32;
    int max_seq = start_pos + n_tokens + 64;
    size_t pos_stride = (size_t)n_kv * D * 2, head_stride = (size_t)D * 2;
    size_t kvn = (size_t)max_seq * n_kv * D;
    uint16_t *hk = (uint16_t *)malloc(kvn * 2), *hv = (uint16_t *)malloc(kvn * 2);
    for (size_t i = 0; i < kvn; i++) { __half a = __float2half(frand() * 2), b = __float2half(frand()); hk[i] = *(uint16_t *)&a; hv[i] = *(uint16_t *)&b; }
    size_t qn = (size_t)n_tokens * n_heads * D;
    float *hq = (float *)malloc(qn * 4), *o0 = (float *)malloc(qn * 4), *o1 = (float *)malloc(qn * 4);
    for (size_t i = 0; i < qn; i++) hq[i] = frand() * 2;
    uint16_t *dk, *dv; float *dq, *dy;
    cudaMalloc(&dk, kvn * 2); cudaMalloc(&dv, kvn * 2); cudaMalloc(&dq, qn * 4); cudaMalloc(&dy, qn * 4);
    cudaMemcpy(dk, hk, kvn * 2, cudaMemcpyHostToDevice); cudaMemcpy(dv, hv, kvn * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, hq, qn * 4, cudaMemcpyHostToDevice);

    /* reference: scalar warpgrp kernel */
    int tile_q = 16;
    size_t sh0 = (size_t)ATTN_TILE_K * D * 2 * 2 + (size_t)tile_q * D * 4 + 2 * tile_q * 4;
    cudaFuncSetAttribute((const void *)picolm_gpu_attention_prefill_warpgrp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh0);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float ms0, ms1;
    cudaEventRecord(a);
    picolm_gpu_attention_prefill_warpgrp_kernel<<<dim3(n_heads, (n_tokens + tile_q - 1) / tile_q), 256, sh0>>>(
        dy, dq, dk, dv, 0, start_pos, n_tokens, n_heads, n_kv, D, max_seq, pos_stride, head_stride, head_stride, tile_q, n_swa, 0);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms0, a, b);
    cudaMemcpy(o0, dy, qn * 4, cudaMemcpyDeviceToHost);

    /* FA3 */
    cudaMemset(dy, 0xFF, qn * 4);
    size_t sh1 = (size_t)2 * TK * D * 2 + (size_t)FA3_WARPS * FA3_ROWS * D * 2 + (size_t)FA3_WARPS * FA3_ROWS * TK * 4;
    cudaFuncSetAttribute((const void *)picolm_gpu_attention_prefill_fa3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh1);
    cudaEventRecord(a);
    picolm_gpu_attention_prefill_fa3_kernel<<<dim3(n_heads, (n_tokens + FA3_ROWS * FA3_WARPS - 1) / (FA3_ROWS * FA3_WARPS)), 32 * FA3_WARPS, sh1>>>(
        dy, dq, dk, dv, 0, start_pos, n_tokens, n_heads, n_kv, D, max_seq, pos_stride, head_stride, head_stride, n_swa, 0, TK);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms1, a, b);
    cudaMemcpy(o1, dy, qn * 4, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();

    double max_abs = 0, max_ref = 0; size_t worst = 0, nan_cnt = 0;
    for (size_t i = 0; i < qn; i++) {
        if (isnan(o1[i]) || isinf(o1[i])) { nan_cnt++; continue; }
        double e = fabs(o0[i] - o1[i]); if (e > max_abs) { max_abs = e; worst = i; }
        if (fabs(o0[i]) > max_ref) max_ref = fabs(o0[i]);
    }
    int fail = err != cudaSuccess || nan_cnt || !(max_abs < 1e-2);
    printf("start=%-5d n=%-5d heads=%d/%d swa=%-4d warpgrp %8.2f ms  fa3 %7.2f ms (%5.1fx)  max_abs_diff %.1e (max|o| %.2f) nonfinite=%zu %s\n",
           start_pos, n_tokens, n_heads, n_kv, n_swa, ms0, ms1, ms0 / ms1, max_abs, max_ref, nan_cnt,
           err != cudaSuccess ? cudaGetErrorString(err) : fail ? "FAIL" : "ok");
    if (fail && nan_cnt < qn) printf("   worst element %zu (tok %zu head %zu dim %zu): ref %f fa3 %f\n", worst,
           worst / ((size_t)n_heads * D), (worst / D) % n_heads, worst % D, o0[worst], o1[worst]);
    cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dy); free(hk); free(hv); free(hq); free(o0); free(o1);
    return fail;
}
int main(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) { printf("needs sm_80+ (mma.m16n8k16): skipped\n"); return 0; }
    int f = 0;
    /* ragged tails first: n % 64 != 0 is what broke the barrier logic */
    f |= run(0, 32, 4, 1, 0);
    f |= run(0, 33, 4, 1, 0);
    f |= run(0, 64, 4, 1, 0);
    f |= run(0, 100, 24, 4, 0);
    f |= run(0, 130, 24, 4, 0);
    f |= run(37, 200, 24, 4, 0);
    f |= run(512, 512, 24, 4, 0);      /* second ubatch of a prompt */
    f |= run(4000, 423, 24, 4, 0);     /* deep, ragged */
    f |= run(0, 300, 8, 4, 64);        /* sliding window */
    f |= run(0, 2048, 24, 4, 0);
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
