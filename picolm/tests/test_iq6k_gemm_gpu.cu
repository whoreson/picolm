/* test_iq6k_gemm_gpu: on Volta, picolm_iq6_k_q8_K_matmul_hmma must be within
 * 1e-6 (of the sum of |terms|) of a double-precision CPU reference of the
 * CPU's IQ6_K x Q8_K math, and prints its speed on Moistral-11B shapes.
 *
 *   make test-iq6k-gemm-gpu && ./test_iq6k_gemm_gpu */
#include "backend_gpu_kernels.cuh"
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static const uint8_t lut[64] = {1,7,13,19,24,30,35,40,44,49,54,58,62,66,70,74,77,81,84,88,91,94,97,100,103,106,109,112,115,117,120,123,126,128,131,134,137,140,142,145,148,151,155,158,161,164,168,172,175,179,183,187,191,196,200,205,210,215,220,226,231,237,243,249};
static uint32_t rng = 31;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float h2f(uint16_t h) { uint32_t u = ((uint32_t)(h & 0x8000) << 16) | ((uint32_t)(((h >> 10) & 0x1F) - 15 + 127) << 23) | ((uint32_t)(h & 0x3FF) << 13); float f; memcpy(&f, &u, 4); return f; }
static int run(int S, int I, int O) {
    int ns = I / 256, rb = ns * 212;
    uint8_t *w = (uint8_t *)malloc((size_t)O * rb); int8_t *xq = (int8_t *)malloc((size_t)S * I);
    float *xd = (float *)malloc((size_t)S * ns * 4);
    for (size_t i = 0; i < (size_t)O * rb; i++) w[i] = rnd();
    for (size_t i = 0; i < (size_t)O * ns; i++) { w[i * 212] = rnd(); w[i * 212 + 1] = 0x10 + rnd() % 0x18; }
    for (size_t i = 0; i < (size_t)S * I; i++) xq[i] = (int8_t)(rnd() % 255 - 127);
    for (size_t i = 0; i < (size_t)S * ns; i++) xd[i] = (rnd() % 100000 + 1) / 1e7f;
    void *dw; int8_t *dq; float *dd, *dy;
    cudaMalloc(&dw, (size_t)O * rb); cudaMalloc(&dq, (size_t)S * I); cudaMalloc(&dd, (size_t)S * ns * 4); cudaMalloc(&dy, (size_t)S * O * 4);
    cudaMemcpy(dw, w, (size_t)O * rb, cudaMemcpyHostToDevice); cudaMemcpy(dq, xq, (size_t)S * I, cudaMemcpyHostToDevice);
    cudaMemcpy(dd, xd, (size_t)S * ns * 4, cudaMemcpyHostToDevice);
    dim3 g((O + 63) / 64, (S + 127) / 128);
    picolm_iq6_k_q8_K_matmul_hmma<<<g, 128>>>(dy, dq, dd, dw, S, I, O, rb, O);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < 3; r++) picolm_iq6_k_q8_K_matmul_hmma<<<g, 128>>>(dy, dq, dd, dw, S, I, O, rb, O);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); ms /= 3;
    float *hy = (float *)malloc((size_t)S * O * 4); cudaMemcpy(hy, dy, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    double mx = 0; int n = 0;
    for (int s = 0; s < S; s += (S > 64 ? 41 : 1))
        for (int o = 0; o < O; o += (O > 256 ? 67 : 1)) {
            double ref = 0, mag = 0;
            for (int sb = 0; sb < ns; sb++) {
                const uint8_t *bk = w + (size_t)o * rb + (size_t)sb * 212;
                double d = h2f((uint16_t)(bk[0] | bk[1] << 8)); int extra = bk[2] | bk[3] << 8;
                for (int ib64 = 0; ib64 < 4; ib64++)
                    for (int gg = 0; gg < 4; gg++) {
                        int mn = (extra >> (4 * ib64 + gg)) & 1, acc = 0;
                        const uint8_t *qs = bk + 20 + 32 * ib64 + 16 * (gg & 1), *qh = bk + 148 + 32 * (ib64 >> 1) + 16 * (gg & 1);
                        int shift = 4 * (ib64 & 1) + 2 * (gg >> 1);
                        for (int j = 0; j < 16; j++) {
                            int lo = gg < 2 ? (qs[j] & 15) : (qs[j] >> 4), q = lo | (((qh[j] >> shift) & 3) << 4);
                            acc += ((int)lut[q] - 128 + mn) * xq[(size_t)s * I + sb * 256 + ib64 * 64 + gg * 16 + j];
                        }
                        double t = acc * d * (int8_t)bk[4 + 4 * ib64 + gg] * xd[(size_t)s * ns + sb];
                        ref += t; mag += fabs(t);
                    }
            }
            double e = fabs(hy[(size_t)s * O + o] - ref) / mag; if (!(e <= mx)) mx = e; n++;
        }
    int fail = err != cudaSuccess || !(mx < 1e-6);
    printf("S=%-4d I=%-6d O=%-5d  %7.2f ms %5.1f TOPS  max_err %.1e (%d checked) %s\n", S, I, O, ms, 2.0 * S * I * O / ms / 1e9, mx, n,
           err != cudaSuccess ? cudaGetErrorString(err) : fail ? "FAIL" : "ok");
    cudaFree(dw); cudaFree(dq); cudaFree(dd); cudaFree(dy); free(w); free(xq); free(xd); free(hy);
    return fail;
}
int main(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!(prop.major == 7 && prop.minor == 0)) { printf("not a Volta GPU: skipped\n"); return 0; }
    int f = 0;
    f |= run(5, 256, 7); f |= run(37, 1024, 129); f |= run(200, 4096, 300);
    f |= run(1900, 4096, 4096); f |= run(1900, 4096, 14336); f |= run(1900, 14336, 4096);
    printf(f ? "FAILED\n" : "all ok\n"); return f;
}
