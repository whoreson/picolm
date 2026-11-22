/* test_attn_decode_gpu: picolm_gpu_attention_decode_warp_kernel + merge must
 * agree with picolm_gpu_attention_decode_split_kernel + merge (max abs diff
 * < 1e-3; observed ~4e-7) over context length, GQA ratio, head_dim 64/128/256
 * and sliding window, and prints both timings.
 *
 *   make test-attn-decode-gpu && ./test_attn_decode_gpu */
#include "backend_gpu_kernels.cuh"
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static uint32_t rng = 5;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (rng % 20001) / 10000.0f - 1.0f; }
static int run(int pos, int n_heads, int n_kv, int D, int n_swa, int splits_new) {
    int max_seq = pos + 8, kv_mul = n_heads / n_kv;
    size_t pos_stride = (size_t)n_kv * D * 2, head_stride = (size_t)D * 2;
    size_t kvn = (size_t)max_seq * n_kv * D;
    uint16_t *hk = (uint16_t *)malloc(kvn * 2), *hv = (uint16_t *)malloc(kvn * 2);
    for (size_t i = 0; i < kvn; i++) { __half a = __float2half(frand() * 3), b = __float2half(frand()); hk[i] = *(uint16_t *)&a; hv[i] = *(uint16_t *)&b; }
    size_t qn = (size_t)n_heads * D;
    float *hq = (float *)malloc(qn * 4), *o0 = (float *)malloc(qn * 4), *o1 = (float *)malloc(qn * 4);
    for (size_t i = 0; i < qn; i++) hq[i] = frand() * 2;
    uint16_t *dk, *dv; float *dq, *dy, *part;
    cudaMalloc(&dk, kvn * 2); cudaMalloc(&dv, kvn * 2); cudaMalloc(&dq, qn * 4); cudaMalloc(&dy, qn * 4);
    cudaMalloc(&part, (size_t)n_heads * 512 * (D + 2) * 4);
    cudaMemcpy(dk, hk, kvn * 2, cudaMemcpyHostToDevice); cudaMemcpy(dv, hv, kvn * 2, cudaMemcpyHostToDevice);
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
                picolm_gpu_attention_decode_split_kernel<<<dim3(n_kv, ns), 256, sh>>>(pm, psum, pacc, dq, dk, dv, 0, pos, n_heads, n_kv, D, max_seq, pos_stride, head_stride, head_stride, ns, chunk, n_swa, 0);
            else
                picolm_gpu_attention_decode_warp_launch(dim3(n_kv, ns), 0, pm, psum, pacc, dq, dk, dv, 0, pos, n_heads, n_kv, D, max_seq, pos_stride, head_stride, head_stride, ns, chunk, n_swa);
            picolm_gpu_attention_decode_merge_kernel<<<dim3(n_kv, kv_mul), 128, msh>>>(dy, pm, psum, pacc, n_heads, n_kv, D, ns);
            cudaEventRecord(b); cudaEventSynchronize(b);
        }
        cudaEventElapsedTime(&ms[which], a, b);
        cudaMemcpy(which ? o1 : o0, dy, qn * 4, cudaMemcpyDeviceToHost);
    }
    cudaError_t err = cudaGetLastError();
    double mx = 0; for (size_t i = 0; i < qn; i++) { double e = fabs(o0[i] - o1[i]); if (!(e <= mx)) mx = e; }
    int fail = err != cudaSuccess || !(mx < 1e-3);
    printf("pos=%-6d heads=%d/%d D=%d swa=%-4d splits=%-3d  split %7.3f ms  warp %7.3f ms (%5.1fx)  max_abs_diff %.1e %s\n",
           pos, n_heads, n_kv, D, n_swa, splits_new, ms[0], ms[1], ms[0] / ms[1], mx, err != cudaSuccess ? cudaGetErrorString(err) : fail ? "FAIL" : "ok");
    cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dy); cudaFree(part); free(hk); free(hv); free(hq); free(o0); free(o1);
    return fail;
}
int main(void) {
    int f = 0;
    f |= run(0, 32, 8, 128, 0, 1);
    f |= run(225, 32, 8, 128, 0, 15);
    f |= run(225, 32, 8, 128, 0, 40);
    f |= run(1999, 32, 8, 128, 0, 40);
    f |= run(5999, 32, 8, 128, 0, 40);
    f |= run(5999, 32, 8, 128, 512, 40);
    f |= run(3000, 16, 2, 256, 0, 64);
    f |= run(700, 12, 4, 64, 0, 20);
    f |= run(900, 24, 4, 128, 0, 30);
    f |= run(400, 8, 8, 128, 0, 24);
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
