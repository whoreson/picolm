/* test_attn_hmma_gpu: on Volta, picolm_gpu_attention_prefill_hmma_kernel
 * must agree with picolm_gpu_attention_prefill_warpgrp_kernel to within
 * fp16 rounding (max abs diff < 1e-2 on outputs of magnitude ~1; observed
 * ~3e-4), over GQA, sliding window, start_pos > 0 and ragged tails.
 *
 *   make test-attn-hmma-gpu && ./test_attn_hmma_gpu */
#include "backend_gpu_kernels.cuh"
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static uint32_t rng = 99;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (rng % 20001) / 10000.0f - 1.0f; }
static int run(int start_pos, int n_tokens, int n_heads, int n_kv, int n_swa) {
    int D = 128, max_seq = start_pos + n_tokens + 64, layers = 1;
    size_t pos_stride = (size_t)n_kv * D * 2, head_stride = (size_t)D * 2;
    size_t kvn = (size_t)layers * max_seq * n_kv * D;
    uint16_t *hk = (uint16_t *)malloc(kvn * 2), *hv = (uint16_t *)malloc(kvn * 2);
    for (size_t i = 0; i < kvn; i++) { __half a = __float2half(frand() * 2), b = __float2half(frand()); hk[i] = *(uint16_t *)&a; hv[i] = *(uint16_t *)&b; }
    size_t qn = (size_t)n_tokens * n_heads * D;
    float *hq = (float *)malloc(qn * 4), *o0 = (float *)malloc(qn * 4), *o1 = (float *)malloc(qn * 4);
    for (size_t i = 0; i < qn; i++) hq[i] = frand() * 2;
    uint16_t *dk, *dv; float *dq, *dy;
    cudaMalloc(&dk, kvn * 2); cudaMalloc(&dv, kvn * 2); cudaMalloc(&dq, qn * 4); cudaMalloc(&dy, qn * 4);
    cudaMemcpy(dk, hk, kvn * 2, cudaMemcpyHostToDevice); cudaMemcpy(dv, hv, kvn * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, hq, qn * 4, cudaMemcpyHostToDevice);
    int tile_q = 16;
    size_t sh = (size_t)ATTN_TILE_K * D * 2 * 2 + (size_t)tile_q * D * 4 + 2 * tile_q * 4;
    cudaFuncSetAttribute((const void *)picolm_gpu_attention_prefill_warpgrp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float ms0, ms1;
    cudaEventRecord(a);
    picolm_gpu_attention_prefill_warpgrp_kernel<<<dim3(n_heads, (n_tokens + tile_q - 1) / tile_q), 256, sh>>>(
        dy, dq, dk, dv, 0, start_pos, n_tokens, n_heads, n_kv, D, max_seq, pos_stride, head_stride, head_stride, tile_q, n_swa, 0);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms0, a, b);
    cudaMemcpy(o0, dy, qn * 4, cudaMemcpyDeviceToHost);
    cudaMemset(dy, 0xFF, qn * 4);
    cudaEventRecord(a);
    picolm_gpu_attention_prefill_hmma_kernel<<<dim3(n_heads, (n_tokens + 63) / 64), 128>>>(
        dy, dq, dk, dv, 0, start_pos, n_tokens, n_heads, n_kv, max_seq, pos_stride, head_stride, head_stride, n_swa);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms1, a, b);
    cudaMemcpy(o1, dy, qn * 4, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < qn; i++) { double e = fabs(o0[i] - o1[i]); if (!(e <= max_abs)) max_abs = e; if (fabs(o0[i]) > max_ref) max_ref = fabs(o0[i]); }
    int fail = err != cudaSuccess || !(max_abs < 1e-2);
    printf("start=%-5d n=%-5d heads=%d/%d swa=%-4d  warpgrp %8.2f ms  hmma %7.2f ms (%5.1fx)  max_abs_diff %.1e (max |o| %.2f) %s\n",
           start_pos, n_tokens, n_heads, n_kv, n_swa, ms0, ms1, ms0 / ms1, max_abs, max_ref, err != cudaSuccess ? cudaGetErrorString(err) : fail ? "FAIL" : "ok");
    cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dy); free(hk); free(hv); free(hq); free(o0); free(o1);
    return fail;
}
int main(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!(prop.major == 7 && prop.minor == 0)) { printf("not a Volta GPU: skipped\n"); return 0; }
    int f = 0;
    f |= run(0, 64, 4, 2, 0);
    f |= run(0, 100, 4, 1, 0);
    f |= run(37, 200, 8, 2, 0);
    f |= run(0, 300, 4, 4, 64);
    f |= run(500, 129, 8, 8, 100);
    f |= run(0, 1900, 32, 8, 0);
    f |= run(0, 5500, 32, 8, 0);
    /* short chunks: a chat turn on a long cached history, and a short prompt */
    f |= run(2048, 5, 32, 8, 0);
    f |= run(2048, 27, 32, 8, 0);
    f |= run(6000, 33, 32, 8, 0);
    f |= run(0, 5, 32, 8, 0);
    f |= run(0, 27, 32, 8, 0);
    f |= run(100, 2, 32, 8, 0);
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
