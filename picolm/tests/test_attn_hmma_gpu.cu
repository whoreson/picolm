/* test_attn_hmma_gpu: on Volta, picolm_gpu_attention_prefill_hmma_kernel
 * must agree with picolm_gpu_attention_prefill_warpgrp_kernel to within
 * fp16 rounding (max abs diff < 1e-2 on outputs of magnitude ~1; observed
 * ~3e-4), over GQA, sliding window, start_pos > 0, ragged tails and
 * F16/Q8_0/Q4_0 KV caches.
 *
 *   make test-attn-hmma-gpu && ./test_attn_hmma_gpu */
#include "backend_gpu_kernels.cuh"
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static uint32_t rng = 99;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (rng % 20001) / 10000.0f - 1.0f; }

/* Encode one head row of D floats as kv_type 0 (F16), 1 (Q8_0) or 2 (Q4_0). */
static size_t kv_row_bytes(int kvt, int D) { return kvt == 0 ? (size_t)D * 2 : (size_t)(D / 32) * (kvt == 1 ? 34 : 18); }
static void kv_encode(uint8_t *dst, const float *x, int D, int kvt) {
    if (kvt == 0) { for (int i = 0; i < D; i++) { __half h = __float2half(x[i]); memcpy(dst + 2 * i, &h, 2); } return; }
    for (int b = 0; b < D / 32; b++) {
        const float *v = x + 32 * b; uint8_t *o = dst + (size_t)b * (kvt == 1 ? 34 : 18);
        float amax = 0, mx = 0;
        for (int j = 0; j < 32; j++) if (fabsf(v[j]) > amax) { amax = fabsf(v[j]); mx = v[j]; }
        float d = kvt == 1 ? amax / 127.0f : mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        __half hd = __float2half(d); memcpy(o, &hd, 2);
        if (kvt == 1) for (int j = 0; j < 32; j++) o[2 + j] = (uint8_t)(int8_t)lroundf(v[j] * id);
        else for (int j = 0; j < 16; j++) {
            int a = (int)(v[j] * id + 8.5f), c = (int)(v[j + 16] * id + 8.5f);
            a = a > 15 ? 15 : a; c = c > 15 ? 15 : c;
            o[2 + j] = (uint8_t)(a | (c << 4));
        }
    }
}
static int run(int start_pos, int n_tokens, int n_heads, int n_kv, int n_swa, int kvt) {
    int D = 128, max_seq = start_pos + n_tokens + 64, layers = 1;
    size_t head_stride = kv_row_bytes(kvt, D), pos_stride = (size_t)n_kv * head_stride;
    size_t kvn = (size_t)layers * max_seq * n_kv * D, kvb = (size_t)layers * max_seq * pos_stride;
    uint8_t *hk = (uint8_t *)malloc(kvb), *hv = (uint8_t *)malloc(kvb);
    { float *fk = (float *)malloc(kvn * 4), *fv = (float *)malloc(kvn * 4);
      for (size_t i = 0; i < kvn; i++) { fk[i] = frand() * 2; fv[i] = frand(); }
      for (size_t r = 0; r < kvn / D; r++) { kv_encode(hk + r * head_stride, fk + r * D, D, kvt); kv_encode(hv + r * head_stride, fv + r * D, D, kvt); }
      free(fk); free(fv); }
    size_t qn = (size_t)n_tokens * n_heads * D;
    float *hq = (float *)malloc(qn * 4), *o0 = (float *)malloc(qn * 4), *o1 = (float *)malloc(qn * 4);
    for (size_t i = 0; i < qn; i++) hq[i] = frand() * 2;
    uint16_t *dk, *dv; float *dq, *dy;
    cudaMalloc(&dk, kvb); cudaMalloc(&dv, kvb); cudaMalloc(&dq, qn * 4); cudaMalloc(&dy, qn * 4);
    cudaMemcpy(dk, hk, kvb, cudaMemcpyHostToDevice); cudaMemcpy(dv, hv, kvb, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, hq, qn * 4, cudaMemcpyHostToDevice);
    int tile_q = 16;
    size_t sh = (size_t)ATTN_TILE_K * D * 2 * 2 + (size_t)tile_q * D * 4 + 2 * tile_q * 4;
    cudaFuncSetAttribute((const void *)picolm_gpu_attention_prefill_warpgrp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float ms0, ms1;
    cudaEventRecord(a);
    picolm_gpu_attention_prefill_warpgrp_kernel<<<dim3(n_heads, (n_tokens + tile_q - 1) / tile_q), 256, sh>>>(
        dy, dq, dk, dv, 0, start_pos, n_tokens, n_heads, n_kv, D, max_seq, pos_stride, head_stride, head_stride, tile_q, n_swa, kvt);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms0, a, b);
    cudaMemcpy(o0, dy, qn * 4, cudaMemcpyDeviceToHost);
    cudaMemset(dy, 0xFF, qn * 4);
    cudaEventRecord(a);
    picolm_gpu_attention_prefill_hmma_kernel<<<dim3(n_heads, (n_tokens + 63) / 64), 128>>>(
        dy, dq, dk, dv, 0, start_pos, n_tokens, n_heads, n_kv, max_seq, pos_stride, head_stride, head_stride, n_swa, kvt);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms1, a, b);
    cudaMemcpy(o1, dy, qn * 4, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < qn; i++) { double e = fabs(o0[i] - o1[i]); if (!(e <= max_abs)) max_abs = e; if (fabs(o0[i]) > max_ref) max_ref = fabs(o0[i]); }
    int fail = err != cudaSuccess || !(max_abs < 1e-2);
    printf("kv=%d start=%-5d n=%-5d heads=%d/%d swa=%-4d  warpgrp %8.2f ms  hmma %7.2f ms (%5.1fx)  max_abs_diff %.1e (max |o| %.2f) %s\n",
           kvt, start_pos, n_tokens, n_heads, n_kv, n_swa, ms0, ms1, ms0 / ms1, max_abs, max_ref, err != cudaSuccess ? cudaGetErrorString(err) : fail ? "FAIL" : "ok");
    cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dy); free(hk); free(hv); free(hq); free(o0); free(o1);
    return fail;
}
int main(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!(prop.major == 7 && prop.minor == 0)) { printf("not a Volta GPU: skipped\n"); return 0; }
    int f = 0;
    for (int kvt = 0; kvt < 3; kvt++) f |= run(0, 64, 4, 2, 0, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(0, 100, 4, 1, 0, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(37, 200, 8, 2, 0, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(0, 300, 4, 4, 64, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(500, 129, 8, 8, 100, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(0, 1900, 32, 8, 0, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(0, 5500, 32, 8, 0, kvt);
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
