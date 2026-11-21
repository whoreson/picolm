/* test_attn_decode_gpu: picolm_gpu_attention_decode_warp_kernel + merge must
 * agree with picolm_gpu_attention_decode_split_kernel + merge (max abs diff
 * < 1e-3; observed ~4e-7) over context length, GQA ratio, head_dim 64/128/256,
 * sliding window and F16/Q8_0/Q4_0 KV, and prints both timings.
 *
 *   make test-attn-decode-gpu && ./test_attn_decode_gpu */
#include "backend_gpu_kernels.cuh"
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static uint32_t rng = 5;
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
static int run(int pos, int n_heads, int n_kv, int D, int n_swa, int splits_new, int kvt) {
    int max_seq = pos + 8, kv_mul = n_heads / n_kv;
    size_t head_stride = kv_row_bytes(kvt, D), pos_stride = (size_t)n_kv * head_stride;
    size_t kvn = (size_t)max_seq * n_kv * D, kvb = (size_t)max_seq * pos_stride;
    uint8_t *hk = (uint8_t *)malloc(kvb), *hv = (uint8_t *)malloc(kvb);
    { float *fk = (float *)malloc(kvn * 4), *fv = (float *)malloc(kvn * 4);
      for (size_t i = 0; i < kvn; i++) { fk[i] = frand() * 3; fv[i] = frand(); }
      for (size_t r = 0; r < kvn / D; r++) { kv_encode(hk + r * head_stride, fk + r * D, D, kvt); kv_encode(hv + r * head_stride, fv + r * D, D, kvt); }
      free(fk); free(fv); }
    size_t qn = (size_t)n_heads * D;
    float *hq = (float *)malloc(qn * 4), *o0 = (float *)malloc(qn * 4), *o1 = (float *)malloc(qn * 4);
    for (size_t i = 0; i < qn; i++) hq[i] = frand() * 2;
    uint16_t *dk, *dv; float *dq, *dy, *part;
    cudaMalloc(&dk, kvb); cudaMalloc(&dv, kvb); cudaMalloc(&dq, qn * 4); cudaMalloc(&dy, qn * 4);
    cudaMalloc(&part, (size_t)n_heads * 512 * (D + 2) * 4);
    cudaMemcpy(dk, hk, kvb, cudaMemcpyHostToDevice); cudaMemcpy(dv, hv, kvb, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, hq, qn * 4, cudaMemcpyHostToDevice);
    float ms[2];
    for (int which = 0; which < 2; which++) {
        int total = pos + 1;
        int ns = which == 0 ? (total + 63) / 64 : splits_new;
        if (ns > 512) ns = 512;
        int chunk = (pos + 1 + ns - 1) / ns;
        float *pm = part, *psum = pm + (size_t)n_heads * ns, *pacc = psum + (size_t)n_heads * ns;
        size_t sh = 2 * D * 2 + 256 * 4 + (size_t)kv_mul * D * 4 + kv_mul * 8 + 8;
        size_t msh = (size_t)kv_mul * 2 * 4 + (size_t)kv_mul * 512 * 4;
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        for (int rep = 0; rep < 2; rep++) {
            cudaEventRecord(a);
            if (which == 0)
                picolm_gpu_attention_decode_split_kernel<<<dim3(n_kv, ns), 256, sh>>>(pm, psum, pacc, dq, dk, dv, 0, pos, n_heads, n_kv, D, max_seq, pos_stride, head_stride, head_stride, ns, chunk, n_swa, kvt);
            else
                picolm_gpu_attention_decode_warp_launch(dim3(n_kv, ns), 0, pm, psum, pacc, dq, dk, dv, 0, pos, n_heads, n_kv, D, max_seq, pos_stride, head_stride, head_stride, ns, chunk, n_swa, kvt);
            picolm_gpu_attention_decode_merge_kernel<<<n_kv, 128, msh>>>(dy, pm, psum, pacc, n_heads, n_kv, D, ns);
            cudaEventRecord(b); cudaEventSynchronize(b);
        }
        cudaEventElapsedTime(&ms[which], a, b);
        cudaMemcpy(which ? o1 : o0, dy, qn * 4, cudaMemcpyDeviceToHost);
    }
    cudaError_t err = cudaGetLastError();
    double mx = 0; for (size_t i = 0; i < qn; i++) { double e = fabs(o0[i] - o1[i]); if (!(e <= mx)) mx = e; }
    int fail = err != cudaSuccess || !(mx < 1e-3);
    printf("kv=%d pos=%-6d heads=%d/%d D=%d swa=%-4d splits=%-3d  split %7.3f ms  warp %7.3f ms (%5.1fx)  max_abs_diff %.1e %s\n",
           kvt, pos, n_heads, n_kv, D, n_swa, splits_new, ms[0], ms[1], ms[0] / ms[1], mx, err != cudaSuccess ? cudaGetErrorString(err) : fail ? "FAIL" : "ok");
    cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dy); cudaFree(part); free(hk); free(hv); free(hq); free(o0); free(o1);
    return fail;
}
int main(void) {
    int f = 0;
    for (int kvt = 0; kvt < 3; kvt++) f |= run(0, 32, 8, 128, 0, 1, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(225, 32, 8, 128, 0, 15, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(225, 32, 8, 128, 0, 40, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(1999, 32, 8, 128, 0, 40, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(5999, 32, 8, 128, 0, 40, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(5999, 32, 8, 128, 512, 40, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(3000, 16, 2, 256, 0, 64, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(700, 12, 4, 64, 0, 20, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(900, 24, 4, 128, 0, 30, kvt);
    for (int kvt = 0; kvt < 3; kvt++) f |= run(400, 8, 8, 128, 0, 24, kvt);
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
