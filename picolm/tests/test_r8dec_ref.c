/* Isolate picolm_sgemm_r8dec_avx2t against a straightforward reference.
 *
 * Motivation: tools/oracle_check.sh reported "R8 decode-once GEMM vs fallback"
 * disagreeing by ~4e-2 max |dlogit| on the tiny SSM model. The oracle hashed
 * generated text, so this hid for a long time. Before touching any kernel, pin
 * down which atype/repack path is wrong and by how much.
 *
 * Reference is built the dumb way: dequantize the weights to float, dequantize
 * the Q8_0 activation columns to float, and dot them. No R8 transpose, no
 * panels, no dpbusd. Anything the kernel does wrong shows up as a difference
 * here, with no model or SSM machinery in the way.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "quant.h"
#include "sgemm.h"

/* Thread count used when driving the kernel, matching the real caller. */
#define R8T_NTH 4

/* Deterministic pseudo-random so failures reproduce. */
static uint32_t g_seed = 12345;
static uint32_t rnd(void) { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }
static float rndf(float s) { return ((float)(rnd() % 20001) / 10000.0f - 1.0f) * s; }

static void put_fp16(uint16_t *dst, float v);
static float get_fp16(uint16_t h);

/* Fill one IQ4_NL_R4 block with random nibbles and per-row deltas. */
static void fill_iq4nl_r4(block_iq4_nl_r4 *b, int nb) {
    for (int ib = 0; ib < nb; ib++) {
        for (int r = 0; r < 4; r++) put_fp16(&b[ib].d[r], rndf(0.05f) + 0.06f);
        for (int i = 0; i < 64; i++) b[ib].qs[i] = (uint8_t)(rnd() & 0xff);
    }
}

/* Dequantize row `row` of the R4 block layout to 32*nb floats.
 * Layout per the header: qs[i + 0], qs[i + 16], qs[i + 32], qs[i + 48] hold the
 * four groups of 8 bytes for this row; low nibble -> v[i + 8*k], high nibble ->
 * v[i + 8*k + 4]. This mirrors r8a_repack_r4_iq4nl's own decode order. */
static void deq_iq4nl_r4_row(const block_iq4_nl_r4 *b, int row, int nb, float *dst) {
    for (int ib = 0; ib < nb; ib++) {
        const float d = get_fp16(b[ib].d[row]);
        const uint8_t *qs = b[ib].qs;
        float v[32];
        for (int i = 0; i < 4; i++) {
            const uint8_t b0 = qs[4 * row + i +  0];
            const uint8_t b1 = qs[4 * row + i + 16];
            const uint8_t b2 = qs[4 * row + i + 32];
            const uint8_t b3 = qs[4 * row + i + 48];
            v[i +  0] = kvalues_iq4nl[b0 & 0xf];
            v[i +  8] = kvalues_iq4nl[b0 >> 4];
            v[i + 16] = kvalues_iq4nl[b1 & 0xf];
            v[i + 24] = kvalues_iq4nl[b1 >> 4];
            v[i +  4] = kvalues_iq4nl[b2 & 0xf];
            v[i + 12] = kvalues_iq4nl[b2 >> 4];
            v[i + 20] = kvalues_iq4nl[b3 & 0xf];
            v[i + 28] = kvalues_iq4nl[b3 >> 4];
        }
        for (int j = 0; j < 32; j++) dst[ib * 32 + j] = v[j] * d;
    }
}

static void put_fp16(uint16_t *dst, float v) {
    *dst = fp32_to_fp16(v);
}
static float get_fp16(uint16_t h) {
    return fp16_to_fp32(h);
}

/* Quantize one row of `n` floats to Q8_0 (reuse the library's quantizer). */
static void quant_row_q8_0(const float *x, block_q8_0 *dst, int n) {
    quantize_row_q8_0(x, dst, n);
}

static int run_case(int atype, const char *name, int m, int n, int nb) {
    const int k = nb * 32;

    /* Weights in the requested native layout. */
    void *A = NULL;
    size_t ablocks = 0;
    if (atype == GGUF_TYPE_IQ4_NL_R4) { ablocks = (size_t)(m / 4) * nb; }
    else if (atype == GGUF_TYPE_Q4_0 || atype == GGUF_TYPE_IQ4_NL) { ablocks = (size_t)m * nb; }
    else { ablocks = (size_t)m * nb; }
    size_t abytes;
    switch (atype) {
    case GGUF_TYPE_IQ4_NL_R4: abytes = ablocks * sizeof(block_iq4_nl_r4); break;
    case GGUF_TYPE_Q4_0:
    case GGUF_TYPE_IQ4_NL:    abytes = ablocks * sizeof(block_q4_0); break;
    default:                  abytes = ablocks * sizeof(block_q8_0); break;
    }
    A = calloc(1, abytes + 64);
    if (!A) return 1;

    if (atype == GGUF_TYPE_IQ4_NL_R4) {
        fill_iq4nl_r4((block_iq4_nl_r4 *)A, (int)ablocks);
    } else if (atype == GGUF_TYPE_Q4_0 || atype == GGUF_TYPE_IQ4_NL) {
        /* Q4_0 / IQ4_NL weights: block_q4_0 (18 bytes) = fp16 d + 16 nibble bytes. */
        block_q4_0 *w = (block_q4_0 *)A;
        for (size_t i = 0; i < ablocks; i++) {
            put_fp16(&w[i].d, rndf(0.05f) + 0.06f);
            for (int j = 0; j < 16; j++) w[i].qs[j] = (uint8_t)(rnd() & 0xff);
        }
    } else {
        /* Q8_0 weights: random int8 + fp16 scale. */
        block_q8_0 *w = (block_q8_0 *)A;
        for (size_t i = 0; i < ablocks; i++) {
            put_fp16(&w[i].d, rndf(0.05f) + 0.06f);
            for (int j = 0; j < 32; j++) w[i].qs[j] = (int8_t)((int)(rnd() % 255) - 127);
        }
    }

    /* Activations: n random columns, quantized to Q8_0. */
    float *xa = malloc(sizeof(float) * (size_t)n * k);
    block_q8_0 *B = calloc((size_t)n * nb, sizeof(block_q8_0));
    float *C = calloc((size_t)n * (size_t)m, sizeof(float));
    float *ref = calloc((size_t)n * (size_t)m, sizeof(float));
    if (!xa || !B || !C || !ref) return 1;

    for (int c = 0; c < n; c++) {
        for (int j = 0; j < k; j++) xa[(size_t)c * k + j] = rndf(1.0f);
        quant_row_q8_0(xa + (size_t)c * k, B + (size_t)c * nb, k);
    }

    /* Reference: dequantize both sides, dot. */
    float *wrow = malloc(sizeof(float) * (size_t)k);
    float *yrow = malloc(sizeof(float) * (size_t)k);
    if (!wrow || !yrow) return 1;
    for (int r = 0; r < m; r++) {
        if (atype == GGUF_TYPE_IQ4_NL_R4) {
            const block_iq4_nl_r4 *blk = (const block_iq4_nl_r4 *)A;
            /* Row r lives in R4 group r/4, position r%4, blocks at (r/4)*nb. */
            deq_iq4nl_r4_row(blk + (size_t)(r / 4) * nb, r % 4, nb, wrow);
        } else if (atype == GGUF_TYPE_Q4_0 || atype == GGUF_TYPE_IQ4_NL) {
            /* Q4_0: 32 nibbles per 16 bytes, low nibble first, minus 8.
             * IQ4_NL: same byte layout but the nibble indexes kvalues_iq4nl. */
            for (int ib = 0; ib < nb; ib++) {
                const block_q4_0 *b = (const block_q4_0 *)A + (size_t)r * nb + ib;
                const float d = get_fp16(b->d);
                for (int j = 0; j < 16; j++) {
                    const int lo = b->qs[j] & 0xf, hi = b->qs[j] >> 4;
                    if (atype == GGUF_TYPE_Q4_0) {
                        wrow[ib * 32 + j]      = (float)(lo - 8) * d;
                        wrow[ib * 32 + j + 16] = (float)(hi - 8) * d;
                    } else {
                        wrow[ib * 32 + j]      = (float)kvalues_iq4nl[lo] * d;
                        wrow[ib * 32 + j + 16] = (float)kvalues_iq4nl[hi] * d;
                    }
                }
            }
        } else {
            for (int ib = 0; ib < nb; ib++) {
                const float d = get_fp16(((const block_q8_0 *)A)[(size_t)r * nb + ib].d);
                for (int j = 0; j < 32; j++)
                    wrow[ib * 32 + j] = (float)((const block_q8_0 *)A)[(size_t)r * nb + ib].qs[j] * d;
            }
        }
        for (int c = 0; c < n; c++) {
            for (int ib = 0; ib < nb; ib++) {
                const float d = get_fp16(B[(size_t)c * nb + ib].d);
                for (int j = 0; j < 32; j++)
                    yrow[ib * 32 + j] = (float)B[(size_t)c * nb + ib].qs[j] * d;
            }
            double acc = 0.0;
            for (int j = 0; j < k; j++) acc += (double)wrow[j] * (double)yrow[j];
            ref[(size_t)c * m + r] = (float)acc;
        }
    }

    /* Kernel under test: the R8 decode-once kernel.
     * Run EVERY thread of an nth-thread partition into the same C buffer, which
     * is exactly how the real caller drives it (gemm_table_task passes
     * nth = pool_total_threads(1) and each idx covers a slice of the row
     * groups). Testing only ith=0/nth=1 would miss a partition bug. */
    int ok1 = 1;
    for (int t = 0; t < R8T_NTH; t++)
        ok1 &= picolm_sgemm_r8dec_avx2t(atype, m, n, nb, A, nb, B, nb, C, m, t, R8T_NTH);
    int ok = ok1;
    if (!ok) {
        printf("%-14s SKIP (kernel declined: atype=%d m=%d n=%d nb=%d)\n", name, atype, m, n, nb);
        free(A); free(xa); free(B); free(C); free(ref); free(wrow); free(yrow);
        return 0;
    }

    /* Also run the non-R8 fallback that PICOLM_Q4_R8=0 selects, so a mismatch
     * can be attributed to the R8 kernel or to its fallback rather than to
     * "the R8 path" as a whole. */
    float *Cf = NULL;
    if (atype == GGUF_TYPE_IQ4_NL_R4) {
        Cf = calloc((size_t)n * (size_t)m, sizeof(float));
        if (Cf) sgemm_iq4_nl_r4_q8_0_avx2(m, n, k, A, B, Cf, (size_t)m, 0, 1);
    }

    double maxd = 0, maxrel = 0, maxmag = 0;
    int worst_r = -1, worst_c = -1;
    for (int c = 0; c < n; c++) {
        for (int r = 0; r < m; r++) {
            double a = C[(size_t)c * m + r], b = ref[(size_t)c * m + r];
            double d = fabs(a - b);
            double rel = (fabs(b) > 1e-9) ? d / fabs(b) : 0.0;
            if (d > maxd) { maxd = d; worst_r = r; worst_c = c; }
            if (rel > maxrel) maxrel = rel;
            if (fabs(b) > maxmag) maxmag = fabs(b);
        }
    }
    printf("%-14s atype=%-3d m=%-3d n=%-4d nb=%-3d max|d|=%.3e maxrel=%.3e max|ref|=%.2f",
           name, atype, m, n, nb, maxd, maxrel, maxmag);
    if (worst_r >= 0 && maxd > 1e-3 * (maxmag > 1 ? maxmag : 1))
        printf("   worst r=%d c=%d (got %.6f want %.6f)", worst_r, worst_c,
               C[(size_t)worst_c * m + worst_r], ref[(size_t)worst_c * m + worst_r]);
    printf("%s\n", (maxd <= 1e-3 * (maxmag > 1 ? maxmag : 1)) ? "" : "   <-- MISMATCH");

    if (Cf) {
        double fd = 0, fmag = 0;
        for (int c = 0; c < n; c++)
            for (int r = 0; r < m; r++) {
                double b = ref[(size_t)c * m + r];
                double d = fabs(Cf[(size_t)c * m + r] - b);
                if (d > fd) fd = d;
                if (fabs(b) > fmag) fmag = fabs(b);
            }
        printf("               fallback (PICOLM_Q4_R8=0 path): max|d|=%.3e max|ref|=%.2f%s\n",
               fd, fmag, (fd <= 1e-3 * (fmag > 1 ? fmag : 1)) ? "" : "   <-- FALLBACK MISMATCH");
        free(Cf);
    }

    int bad = (maxd > 1e-3 * (maxmag > 1 ? maxmag : 1));
    free(A); free(xa); free(B); free(C); free(ref); free(wrow); free(yrow);
    return bad;
}

int main(void) {
    int bad = 0;
    /* n=1 avoids the column-panel path; larger n exercises r8a_panel_cols tiling. */
    bad |= run_case(GGUF_TYPE_Q8_0,      "q8_0",      8, 1, 4);
    bad |= run_case(GGUF_TYPE_Q8_0,      "q8_0",     16, 7, 4);
    bad |= run_case(GGUF_TYPE_Q4_0,      "q4_0",     16, 7, 4);
    bad |= run_case(GGUF_TYPE_IQ4_NL,    "iq4_nl",   16, 7, 4);
    bad |= run_case(GGUF_TYPE_IQ4_NL_R4, "iq4nl_r4", 16, 7, 4);
    /* Exact FFN gate/up shape from the failing tiny model: d=320 n=256 batch=38. */
    bad |= run_case(GGUF_TYPE_IQ4_NL_R4, "iq4nl_r4", 256, 38, 10);
    /* Same, with ncols above the default PICOLM_Q4_R8_MIN_N (16) and panel
     * boundaries exercised (r8a_panel_cols). */
    bad |= run_case(GGUF_TYPE_IQ4_NL_R4, "iq4nl_r4-pc", 256, 200, 10);
    printf("\n%s\n", bad ? "FAIL: at least one path disagrees with the reference"
                          : "all R8 GEMM paths match the float reference");
    return bad ? 1 : 0;
}
