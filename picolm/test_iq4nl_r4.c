#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "quant.h"

/* Force inclusion of implementations */
extern void dequantize_row_iq4_nl_r4_single(const block_iq4_nl_r4 *x, float *dst, int n, int row);
extern void vec_dot_iq4_nl_r4_q8_0_avx2(const void *vx, const void *wy, int n, float *out, int nrows);
extern float vec_dot_iq4_nl_r4_q8_0(const void *vx, const void *wy, int n);
extern void vec_dot_iq4_nl_r4_q8_0_batch4(const void *vx, const void *wy, int n, float *out);

extern void fp16_table_init(void);

int main(void) {
    fp16_table_init();
    /* Create a single block of IQ4_NL_R4 with known values */
    block_iq4_nl_r4 wb;
    memset(&wb, 0, sizeof(wb));
    wb.d[0] = 0x3C00; /* 1.0f */
    wb.d[1] = 0x3C00; /* 1.0f */
    wb.d[2] = 0x3C00; /* 1.0f */
    wb.d[3] = 0x3C00; /* 1.0f */

    /* Fill qs with known pattern: each byte = 0x53 (nibbles 0x5, 0x3)
     * kvalues_iq4nl[0x5] = -22, kvalues_iq4nl[0x3] = -65 */
    for (int j = 0; j < 64; j++) {
        wb.qs[j] = 0x53;
    }

    /* Create a Q8_0 activation block */
    block_q8_0 yb;
    yb.d = 0x3C00; /* 1.0f */
    for (int j = 0; j < 32; j++) {
        yb.qs[j] = 1; /* all ones */
    }

    /* Check scale conversion */
    printf("fp16_to_fp32_lookup(0x3C00) = %.1f\n", fp16_to_fp32_lookup(0x3C00));
    printf("kvalues_iq4nl[0x5] = %d, kvalues_iq4nl[0x3] = %d\n", kvalues_iq4nl[0x5], kvalues_iq4nl[0x3]);

    /* Dump what scalar sees */
    printf("wb.d[0]=%u wb.d[1]=%u wb.d[2]=%u wb.d[3]=%u\n", wb.d[0], wb.d[1], wb.d[2], wb.d[3]);
    printf("qs[0]=%02x qs[1]=%02x qs[4]=%02x qs[16]=%02x\n", wb.qs[0], wb.qs[1], wb.qs[4], wb.qs[16]);

    /* Dequantize each row using scalar */
    float deq[4][32];
    for (int r = 0; r < 4; r++) {
        dequantize_row_iq4_nl_r4_single(&wb, deq[r], 32, r);
        printf("Scalar dequant row %d: ", r);
        for (int j = 0; j < 32; j++) printf("%.1f ", deq[r][j]);
        printf("\n");
    }

    /* Batch4 (AVX2 path) */
    float out4[4];
    vec_dot_iq4_nl_r4_q8_0_batch4(&wb, &yb, 32, out4);
    printf("Batch4 (AVX2):   ");
    for (int r = 0; r < 4; r++) printf("%.1f ", out4[r]);
    printf("\n");

    /* Scalar single row */
    float out0 = vec_dot_iq4_nl_r4_q8_0(&wb, &yb, 32);
    printf("Scalar row0:     %.1f\n", out0);

    /* Manual computation for row 0 */
    /* All nibbles are 0x5 (low) and 0x3 (high)
     * kvalues_iq4nl[5] = -22, kvalues_iq4nl[3] = -65
     * Each activation is 1, so sum = 32 * (kvalues_iq4nl[5] + kvalues_iq4nl[3]) / 2
     * Wait, each value is either -22 or -65, and there are 32 values
     * 16 values of -22, 16 values of -65
     * sum = 16*(-22) + 16*(-65) = 16*(-87) = -1392 */
    printf("Expected row0:   -1392.0\n");

    /* Check if scalar and AVX2 agree */
    if (out4[0] != out0) {
        printf("MISMATCH: scalar=%.1f avx2=%.1f\n", out0, out4[0]);
    } else {
        printf("MATCH: scalar=avx2=%.1f\n", out0);
    }

    return 0;
}
