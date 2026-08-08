/* test_batch4.c: verify vec_dot_q8_0_f32_batch4 matches 4x vec_dot_q8_0_f32 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* Minimal definitions needed */
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;

void fp16_table_init(void);
float fp16_to_fp32(uint16_t h);
float vec_dot_q8_0_f32(const void *src, const float *x, int n);
void vec_dot_q8_0_f32_batch4(const void *qx0, const void *qx1, const void *qx2, const void *qx3,
                              const float *w, int n,
                              float *out0, float *out1, float *out2, float *out3);

/* Convert float to fp16 (IEEE 754 half precision) */
static uint16_t float_to_fp16(float f) {
    union { uint32_t u; float f; } u;
    u.f = f;
    uint32_t x = u.u;
    uint16_t sign = (x >> 31) & 1;
    int32_t  e = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) { e = 0; m = 0; }
    if (e > 31) { e = 31; m = 0; }
    return (sign << 15) | (e << 10) | (m >> 13);
}

static void rand_block(block_q8_0 *b, float scale) {
    b->d = float_to_fp16(scale);
    for (int i = 0; i < 32; i++)
        b->qs[i] = (int8_t)(rand() % 256) - 128;
}

int main(void) {
    fp16_table_init();
    srand(42);
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048};
    int ntests = sizeof(sizes) / sizeof(sizes[0]);
    int total_fail = 0;

    for (int t = 0; t < ntests; t++) {
        int n = sizes[t];
        int nb = n / 32;
        for (int seed = 0; seed < 5; seed++) {
            srand(seed * 1000 + n);

            /* 4 tokens with different scales to catch delta bugs */
            float scales[4] = {0.1f, 1.0f, 5.0f, 30.0f};
            block_q8_0 *tok[4];
            float *w = (float *)malloc(n * sizeof(float));
            for (int k = 0; k < 4; k++) {
                tok[k] = (block_q8_0 *)malloc(nb * sizeof(block_q8_0));
                for (int i = 0; i < nb; i++)
                    rand_block(&tok[k][i], scales[k]);
            }
            for (int i = 0; i < n; i++)
                w[i] = (float)(rand() % 200 - 100) / 10.0f;

            float ref0 = vec_dot_q8_0_f32(tok[0], w, n);
            float ref1 = vec_dot_q8_0_f32(tok[1], w, n);
            float ref2 = vec_dot_q8_0_f32(tok[2], w, n);
            float ref3 = vec_dot_q8_0_f32(tok[3], w, n);

            float out0, out1, out2, out3;
            vec_dot_q8_0_f32_batch4(tok[0], tok[1], tok[2], tok[3], w, n,
                                    &out0, &out1, &out2, &out3);

            float refs[4] = {ref0, ref1, ref2, ref3};
            float outs[4] = {out0, out1, out2, out3};
            for (int k = 0; k < 4; k++) {
                float rel = fabsf(refs[k] - outs[k]) / (fabsf(refs[k]) + 1e-6f);
                if (rel > 1e-5f) {
                    printf("FAIL: n=%d seed=%d tok=%d: ref=%.6f batch4=%.6f rel_err=%.2e\n",
                           n, seed, k, refs[k], outs[k], rel);
                    total_fail++;
                }
            }
            for (int k = 0; k < 4; k++) free(tok[k]);
            free(w);
        }
    }
    if (total_fail == 0)
        printf("ALL PASS (%d sizes x 5 seeds x 4 tokens)\n", ntests);
    else
        printf("%d FAILURES\n", total_fail);
    return total_fail ? 1 : 0;
}
