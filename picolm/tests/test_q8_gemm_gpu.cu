/* test_q8_gemm_gpu: picolm_q8_q8_matmul_dp4a must match
 * picolm_q8_q8_matmul_tiled and picolm_q8_q8_matmul bit for bit (and be close to a
 * double-precision CPU reference). On Volta, picolm_q8_q8_matmul_hmma must
 * be within 1e-6 of that reference (relative to the sum of |terms|). Random Q8_0 weights and activations over
 * shapes that hit every tail: K not a multiple of 1024, K > 8192 (slots with
 * two terms), O and S not multiples of the tiles.
 *
 *   make test-q8-gemm-gpu && ./test_q8_gemm_gpu */
#include "backend_gpu_kernels.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t rng = 12345;
static int g_hmma;   /* Volta: also test picolm_q8_q8_matmul_hmma */
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static uint16_t f32_to_f16(float f) {   /* normal range only, round toward zero */
    uint32_t u; memcpy(&u, &f, 4);
    int e = (int)((u >> 23) & 0xFF) - 127 + 15;
    return (uint16_t)(((u >> 16) & 0x8000) | (e << 10) | ((u >> 13) & 0x3FF));
}
static float f16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)(h & 0x8000) << 16) | ((uint32_t)(((h >> 10) & 0x1F) - 15 + 127) << 23)
               | ((uint32_t)(h & 0x3FF) << 13);
    float f; memcpy(&f, &u, 4); return f;
}

static int run(int S, int I, int O) {
    int nb = I / 32, row_bytes = nb * GPU_BLOCK_Q8_0_SIZE;
    size_t wn = (size_t)O * row_bytes;
    uint8_t *w = (uint8_t *)malloc(wn);
    int8_t *xq = (int8_t *)malloc((size_t)S * I);
    float *xd = (float *)malloc((size_t)S * nb * sizeof(float));
    float *y0 = (float *)malloc((size_t)S * O * sizeof(float));
    float *y1 = (float *)malloc((size_t)S * O * sizeof(float));
    float *y2 = (float *)calloc((size_t)S * O, sizeof(float));
    float *y3 = (float *)malloc((size_t)S * O * sizeof(float));

    for (int o = 0; o < O; o++)
        for (int b = 0; b < nb; b++) {
            uint8_t *p = w + (size_t)o * row_bytes + (size_t)b * GPU_BLOCK_Q8_0_SIZE;
            uint16_t d = f32_to_f16((float)(rnd() % 1000 + 1) / 4096.0f);
            p[0] = d & 0xFF; p[1] = d >> 8;
            for (int i = 0; i < 32; i++) p[2 + i] = (uint8_t)(rnd() % 255 - 127);
        }
    for (size_t i = 0; i < (size_t)S * I; i++) xq[i] = (int8_t)(rnd() % 255 - 127);
    for (size_t i = 0; i < (size_t)S * nb; i++) xd[i] = (float)(rnd() % 100000 + 1) / 1e7f;

    void *dw; int8_t *dxq; float *dxd, *dy;
    cudaMalloc(&dw, wn); cudaMalloc(&dxq, (size_t)S * I);
    cudaMalloc(&dxd, (size_t)S * nb * sizeof(float)); cudaMalloc(&dy, (size_t)S * O * sizeof(float));
    cudaMemcpy(dw, w, wn, cudaMemcpyHostToDevice);
    cudaMemcpy(dxq, xq, (size_t)S * I, cudaMemcpyHostToDevice);
    cudaMemcpy(dxd, xd, (size_t)S * nb * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(dy, 0xFF, (size_t)S * O * sizeof(float));
    picolm_q8_q8_matmul_tiled<<<dim3(O, (S + Q8_TILE_S - 1) / Q8_TILE_S), 256, row_bytes>>>(
        dy, dxq, dxd, dw, S, I, O, row_bytes, O);
    cudaMemcpy(y0, dy, (size_t)S * O * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemset(dy, 0xFF, (size_t)S * O * sizeof(float));
    picolm_q8_q8_matmul_dp4a<<<dim3((O + 7) / 8, (S + 7) / 8), 256>>>(
        dy, dxq, dxd, dw, S, I, O, row_bytes, O);
    cudaMemcpy(y1, dy, (size_t)S * O * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemset(dy, 0xFF, (size_t)S * O * sizeof(float));
    picolm_q8_q8_matmul<<<dim3(O, S), 256>>>(dy, dxq, dxd, dw, S, I, O, row_bytes, O);
    cudaMemcpy(y3, dy, (size_t)S * O * sizeof(float), cudaMemcpyDeviceToHost);
    if (g_hmma) {
        cudaMemset(dy, 0xFF, (size_t)S * O * sizeof(float));
        picolm_q8_q8_matmul_hmma<<<dim3((O + 63) / 64, (S + 127) / 128), 128>>>(
            dy, dxq, dxd, dw, S, I, O, row_bytes, O);
        cudaMemcpy(y2, dy, (size_t)S * O * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaError_t err = cudaDeviceSynchronize();

    int diff = 0; double max_rel = 0.0, max_rel2 = 0.0;
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            size_t k = (size_t)s * O + o;
            if (memcmp(&y0[k], &y1[k], 4) || memcmp(&y3[k], &y1[k], 4)) diff++;
            double ref = 0.0, mag = 0.0;
            for (int b = 0; b < nb; b++) {
                const uint8_t *p = w + (size_t)o * row_bytes + (size_t)b * GPU_BLOCK_Q8_0_SIZE;
                int acc = 0;
                for (int i = 0; i < 32; i++) acc += (int8_t)p[2 + i] * xq[(size_t)s * I + b * 32 + i];
                double t = (double)acc * f16_to_f32((uint16_t)(p[0] | p[1] << 8)) * xd[(size_t)s * nb + b];
                ref += t; mag += fabs(t);
            }
            double rel = fabs(y1[k] - ref) / mag;   /* relative to sum |terms| */
            if (rel > max_rel) max_rel = rel;
            rel = fabs(y2[k] - ref) / mag;
            if (g_hmma && rel > max_rel2) max_rel2 = rel;
        }
    int fail = err != cudaSuccess || diff || max_rel > 1e-6 || max_rel2 > 1e-6;
    printf("S=%-4d I=%-6d O=%-5d  differing=%d/%d  max_err_vs_cpu=%.1e  hmma=%.1e  %s\n",
           S, I, O, diff, S * O, max_rel, max_rel2,
           err != cudaSuccess ? cudaGetErrorString(err) : fail ? "FAIL" : "ok");
    cudaFree(dw); cudaFree(dxq); cudaFree(dxd); cudaFree(dy);
    free(w); free(xq); free(xd); free(y0); free(y1); free(y2); free(y3);
    return fail;
}

int main(void) {
    static const int shapes[][3] = {   /* S, I, O */
        {1, 32, 1}, {5, 96, 7}, {33, 4096, 13}, {64, 4096, 256}, {37, 5120, 129},
        {128, 14336, 64}, {19, 8192 + 160, 9}, {200, 1024, 4096}, {512, 4096, 1024},
    };
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    g_hmma = prop.major == 7 && prop.minor == 0;
    int fail = 0;
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
        fail |= run(shapes[i][0], shapes[i][1], shapes[i][2]);
    printf(fail ? "FAILED\n" : g_hmma ? "all bit-identical, hmma within 1e-6\n" : "all bit-identical\n");
    return fail;
}
