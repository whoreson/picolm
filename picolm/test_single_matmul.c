#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "quant.h"
#include "backend_gpu.h"

void fp16_table_init(void);

int main(void) {
    fp16_table_init();

    int S = 3, I = 576, O = 576;
    float *x = calloc(S * I, sizeof(float));
    float *y_gpu = calloc(S * O, sizeof(float));
    float *y_cpu = calloc(S * O, sizeof(float));

    // Deterministic input
    for (int i = 0; i < S * I; i++) x[i] = (float)((i * 7 + 3) % 97 - 48) / 48.0f;

    // Create Q8_0 weight matrix with known values
    size_t rb = ((I + 31) / 32) * 34;
    uint8_t *w = (uint8_t *)calloc(O * rb, 1);
    for (int o = 0; o < O; o++) {
        for (int bi = 0; bi < (I+31)/32; bi++) {
            uint16_t d16 = 0x3C00; // d=1.0
            memcpy(w + o * rb + bi * 34, &d16, 2);
            for (int v = 0; v < 32; v++) {
                int vi = bi * 32 + v;
                if (vi >= I) break;
                int8_t val = (int8_t)(((o * 7 + vi * 13) % 251) - 127);
                w[o * rb + bi * 34 + 2 + v] = (uint8_t)val;
            }
        }
    }

    // CPU reference
    for (int s = 0; s < S; s++) {
        for (int o = 0; o < O; o++) {
            float sum = 0.0f;
            const block_q8_0 *blk = (const block_q8_0 *)(w + o * rb);
            int nblocks = (I + 31) / 32;
            for (int bi = 0; bi < nblocks; bi++) {
                float d = fp16_to_fp32_lookup(blk[bi].d);
                for (int v = 0; v < 32; v++) {
                    int vi = bi * 32 + v;
                    if (vi >= I) break;
                    sum += x[s * I + vi] * (float)blk[bi].qs[v] * d;
                }
            }
            y_cpu[s * O + o] = sum;
        }
    }

    // GPU via Vulkan backend
    printf("Initializing Vulkan...\n");
    int dev[] = {0};
    if (!picolm_gpu_init(dev, 1)) { fprintf(stderr, "GPU init failed\n"); return 1; }

    picolm_gpu_tensor_t *t;
    if (!picolm_gpu_tensor_upload((void *)&t, w, GGUF_TYPE_Q8_0, I, O, 0)) {
        fprintf(stderr, "Upload failed\n"); picolm_gpu_shutdown(); return 1;
    }

    printf("Running GPU matmul S=%d I=%d O=%d...\n", S, I, O);
    if (!picolm_gpu_matmul(t, y_gpu, x, S, 0)) {
        fprintf(stderr, "Matmul failed\n"); picolm_gpu_shutdown(); return 1;
    }

    // Compare
    float max_abs_diff = 0, max_rel_diff = 0;
    int first_diff_at = -1;
    for (int i = 0; i < S * O; i++) {
        float d = y_gpu[i] - y_cpu[i];
        if (d < 0) d = -d;
        float r = y_cpu[i] != 0 ? d / (float)abs(y_cpu[i]) : 0;
        if (d < 0) d = -d;
        if (d > max_abs_diff) max_abs_diff = d;
        if (r > max_rel_diff) max_rel_diff = r;
        if (d > 0.001f && first_diff_at < 0) first_diff_at = i;
    }

    printf("Max abs diff: %f\n", max_abs_diff);
    printf("Max rel diff: %f\n", max_rel_diff);
    printf("First diff at: %d (s=%d, o=%d)\n", first_diff_at, first_diff_at / O, first_diff_at % O);

    if (first_diff_at >= 0) {
        printf("GPU[0..4]={%f %f %f %f %f}\n", y_gpu[0],y_gpu[1],y_gpu[2],y_gpu[3],y_gpu[4]);
        printf("CPU[0..4]={%f %f %f %f %f}\n", y_cpu[0],y_cpu[1],y_cpu[2],y_cpu[3],y_cpu[4]);
        int o = first_diff_at % O;
        int s = first_diff_at / O;
        printf("GPU[s=%d,o=%d..%d]={%f %f %f %f}\n", s, o,o+1,o+2,o+3,
               y_gpu[first_diff_at], y_gpu[first_diff_at+1], y_gpu[first_diff_at+2], y_gpu[first_diff_at+3]);
        printf("CPU[s=%d,o=%d..%d]={%f %f %f %f}\n", s, o,o+1,o+2,o+3,
               y_cpu[first_diff_at], y_cpu[first_diff_at+1], y_cpu[first_diff_at+2], y_cpu[first_diff_at+3]);
    }

    picolm_gpu_tensor_free(t);
    picolm_gpu_shutdown();
    free(x); free(y_gpu); free(y_cpu); free(w);
    return max_abs_diff > 0.01f ? 1 : 0;
}

