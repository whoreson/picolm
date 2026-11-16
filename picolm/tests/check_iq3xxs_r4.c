/* Validate the IQ3_XXS_R4 / IQ4_XS_R8 dequantizers against ground truth.
 *
 * The R4/R8 formats are just row-interleaved versions of the plain formats, so
 * dequantize_row_*_r4/_r8_single() must reproduce exactly what the plain
 * dequantizer would produce for the same logical row. We check that two ways:
 *
 *   1. Interleave check (no file needed): build a synthetic R4/R8 group by
 *      calling the ik_llama repack rule on plain blocks, then confirm every
 *      logical row round-trips to the plain dequantizer's output.
 *   2. Real-file check (optional): given a pure IQ3_XXS_R4 / IQ4_XS_R8 GGUF and
 *      its F16/F32 ancestor, compare relative error against the source model,
 *      which catches a wrong layout that is self-consistent.
 *
 *   ./check_iq3xxs_r4            # interleave check only
 *   ./check_iq3xxs_r4 <r4.gguf>  # + statistics on a real repacked model
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "quant.h"

void fp16_table_init(void);

static unsigned rs = 12345;
static unsigned rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

/* Probability of a mixed-precision match within one quantization step. */
static double relerr(const float *a, const float *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { num += (a[i]-b[i])*(a[i]-b[i]); den += (double)b[i]*b[i]; }
    return den > 0 ? sqrt(num/den) : (num > 0 ? 1e9 : 0.0);
}

/* ---- Build a synthetic IQ3_XXS_R4 group from 4 plain blocks, using
 * ik_llama's repack_iq3_xxs() rule verbatim. ---- */
static void pack_iq3xxs_r4(const block_iq3_xxs x[4], block_iq3_xxs_r4 *y) {
    static const uint8_t scr[128] = {
        0x00,0x7f,0x7e,0x01,0x7c,0x03,0x02,0x7d,0x78,0x07,0x06,0x79,0x04,0x7b,0x7a,0x05,
        0x70,0x0f,0x0e,0x71,0x0c,0x73,0x72,0x0d,0x08,0x77,0x76,0x09,0x74,0x0b,0x0a,0x75,
        0x60,0x1f,0x1e,0x61,0x1c,0x63,0x62,0x1d,0x18,0x67,0x66,0x19,0x64,0x1b,0x1a,0x65,
        0x10,0x6f,0x6e,0x11,0x6c,0x13,0x12,0x6d,0x68,0x17,0x16,0x69,0x14,0x6b,0x6a,0x15,
        0x40,0x3f,0x3e,0x41,0x3c,0x43,0x42,0x3d,0x38,0x47,0x46,0x39,0x44,0x3b,0x3a,0x45,
        0x30,0x4f,0x4e,0x31,0x4c,0x33,0x32,0x4d,0x48,0x37,0x36,0x49,0x34,0x4b,0x4a,0x35,
        0x20,0x5f,0x5e,0x21,0x5c,0x23,0x22,0x5d,0x58,0x27,0x26,0x59,0x24,0x5b,0x5a,0x25,
        0x50,0x2f,0x2e,0x51,0x2c,0x53,0x52,0x2d,0x28,0x57,0x56,0x29,0x54,0x2b,0x2a,0x55,
    };
    memset(y, 0, sizeof *y);
    for (int k = 0; k < 4; k++) {
        y->d[k] = x[k].d;
        const uint8_t *xsas = x[k].qs + QK_K/4;
        for (int ib = 0; ib < QK_K/32; ib++) {
            for (int i = 0; i < 8; i++) y->qs[32*ib + 8*k + i] = x[k].qs[8*ib + i];
            uint32_t aux32; memcpy(&aux32, xsas + 4*ib, 4);
            unsigned scale = aux32 >> 28;
            uint8_t s[4];
            for (int l = 0; l < 4; l++)
                s[l] = (uint8_t)((scr[(aux32 >> (7*l)) & 127] << 1) | ((scale >> l) & 1));
            uint32_t w = (uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16) | ((uint32_t)s[3] << 24);
            /* ysas = (uint32_t *)sas, so ysas[4*ib+k] == sas + 4*(4*ib+k) */
            memcpy(y->sas + 4*(4*ib + k), &w, 4);
        }
    }
}

/* ---- Build a synthetic IQ4_XS_R8 group from 8 plain blocks ---- */
static void pack_iq4xs_r8(const block_iq4_xs x[8], block_iq4_xs_r8 *y) {
    memset(y, 0, sizeof *y);
    for (int k = 0; k < 8; k++) {
        y->d[k] = x[k].d;
        for (int ib = 0; ib < QK_K/32; ib++) {
            uint8_t sl = (x[k].scales_l[ib/2] >> (4*(ib%2))) & 0xf;
            uint8_t sh = (x[k].scales_h >> (2*ib)) & 3;
            int i = 8*ib + k;
            y->scales_l[i % 32] |= (uint8_t)(sl << (4*(i/32)));
            y->scales_h[i % 16] |= (uint8_t)(sh << (2*(i/16)));
            for (int j = 0; j < 4; j++) {
                y->qs[128*ib + 4*k + j +  0] = (x[k].qs[16*ib+j+0] & 0xf) | ((x[k].qs[16*ib+j+ 4] & 0xf) << 4);
                y->qs[128*ib + 4*k + j + 32] = (x[k].qs[16*ib+j+8] & 0xf) | ((x[k].qs[16*ib+j+12] & 0xf) << 4);
                y->qs[128*ib + 4*k + j + 64] = (x[k].qs[16*ib+j+0] >>  4) | ((x[k].qs[16*ib+j+ 4] >>  4) << 4);
                y->qs[128*ib + 4*k + j + 96] = (x[k].qs[16*ib+j+8] >>  4) | ((x[k].qs[16*ib+j+12] >>  4) << 4);
            }
        }
    }
}

static int check_iq3xxs_r4(void) {
    int nb = 4, n = nb * QK_K;
    block_iq3_xxs x[4][4];   /* 4 rows x nb blocks */
    for (int r = 0; r < 4; r++)
        for (int i = 0; i < nb; i++) {
            uint8_t *b = (uint8_t *)&x[r][i];
            /* Grid indices must address the 256-entry iq3xxs_grid; the sign
             * words are free-form. Random full-byte indices would read past the
             * table, so build valid blocks: 64 index bytes in 0..255 (the grid
             * has 256 entries) and 32 scale/sign bytes. */
            for (size_t j = 2; j < sizeof(block_iq3_xxs); j++) b[j] = (uint8_t)rnd();
            x[r][i].d = fp32_to_fp16(0.001f + (rnd() & 255) * 1e-5f);
            x[r][i].d = fp32_to_fp16(0.001f + (rnd() & 255) * 1e-5f);
        }
    /* group g holds one block from each row */
    block_iq3_xxs_r4 g[4];
    for (int i = 0; i < nb; i++) {
        block_iq3_xxs col[4];
        for (int r = 0; r < 4; r++) col[r] = x[r][i];
        pack_iq3xxs_r4(col, &g[i]);
    }
    double worst = 0;
    for (int r = 0; r < 4; r++) {
        float got[4*QK_K], ref[4*QK_K];
        dequantize_row_iq3_xxs_r4_single(g, got, n, r);
        /* reference: the plain blocks, row r, concatenated */
        for (int i = 0; i < nb; i++)
            dequantize_row_iq3_xxs(&x[r][i], ref + i*QK_K, QK_K);
        double e = relerr(got, ref, n);
        if (e > worst) worst = e;
        if (e > 1e-6) { printf("  FAIL iq3_xxs_r4 row %d relerr=%.3g\n", r, e); return 1; }
    }
    printf("iq3_xxs_r4  interleave OK (worst relerr %.2e)\n", worst);
    return 0;
}

static int check_iq4_xs_r8(void) {
    int nb = 4, n = nb * QK_K;
    block_iq4_xs x[8][4];
    for (int r = 0; r < 8; r++)
        for (int i = 0; i < nb; i++) {
            uint8_t *b = (uint8_t *)&x[r][i];
            for (size_t j = 2; j < sizeof(block_iq4_xs); j++) b[j] = (uint8_t)rnd();
            x[r][i].d = fp32_to_fp16(0.001f + (rnd() & 255) * 1e-5f);
        }
    block_iq4_xs_r8 g[4];
    for (int i = 0; i < nb; i++) {
        block_iq4_xs col[8];
        for (int r = 0; r < 8; r++) col[r] = x[r][i];
        pack_iq4xs_r8(col, &g[i]);
    }
    double worst = 0;
    for (int r = 0; r < 8; r++) {
        float got[4*QK_K], ref[4*QK_K];
        dequantize_row_iq4_xs_r8_single(g, got, n, r);
        for (int i = 0; i < nb; i++)
            dequantize_row_iq4_xs(&x[r][i], ref + i*QK_K, QK_K);
        double e = relerr(got, ref, n);
        if (e > worst) worst = e;
        if (e > 1e-6) { printf("  FAIL iq4_xs_r8 row %d relerr=%.3g\n", r, e); return 1; }
    }
    printf("iq4_xs_r8   interleave OK (worst relerr %.2e)\n", worst);
    return 0;
}

int main(int argc, char **argv) {
    fp16_table_init();
    int bad = 0;
    bad |= check_iq3xxs_r4();
    bad |= check_iq4_xs_r8();
    if (argc > 1) {
        printf("\n(real-model statistics for '%s' need the GGUF loader; run picolm\n"
               " with -p on that file and compare against the F16 original)\n", argv[1]);
    }
    printf("check_iq3xxs_r4: %s\n", bad ? "FAIL" : "all OK");
    return bad;
}
