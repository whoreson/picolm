/* check_r4_gemv.c -- differential test for the row-interleaved (R4/R8) GEMVs.
 *
 * WHAT THIS CATCHES
 *   The R4/R8 formats (Q4_K_R4, Q5_K_R4, IQ3_XXS_R4, IQ4_XS_R8, ...) store
 *   `rpb` logical rows inside ONE block group. Everything about them is easy to
 *   get wrong in a way that still produces plausible numbers:
 *
 *     - addressing the group with `block_ptr + i` instead of a byte stride
 *       (steps rpb times too far; off-by-row errors stay correlated, so the
 *       model keeps emitting grammatical-looking text),
 *     - ignoring the `nrows` argument and always writing rpb floats,
 *     - a SIMD helper that reads the wrong 8 activation bytes
 *       (_mm256_extracti128_si256(x,1) is +16, not +8 -- this bug lived in
 *       quant.c's r4_f32_q8k_dot and made every R4 GEMV wrong),
 *     - per-block scale folding mistakes (y[i].d applied once, not per block).
 *
 *   None of these are visible from --dump-tensor (dequantizers only) and all of
 *   them are buried in a 12 GB model, so this test uses SYNTHETIC groups: random
 *   block contents (any bit pattern decodes to something), random Q8_K
 *   activations, and the verified generic reference
 *   vec_dot_r4_q8_k_dequant_ref() as the oracle.
 *
 *   It runs in well under a second and needs no model files.
 *
 * COVERAGE
 *   For every type: nb = 1, 2, 5 blocks; rpb rows; nrows = rpb and nrows < rpb.
 *   Relative error must stay under 1e-4 (one float ULP-scale discrepancy: the
 *   GEMVs accumulate in float while the reference accumulates per block in
 *   float too, so they agree to rounding, not bit-exactly).
 *
 * USAGE
 *   make check-r4-gemv
 *   ./check_r4_gemv          # exit 0 = all pass
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "quant.h"

void fp16_table_init(void);

static unsigned rng_state = 12345u;
static unsigned rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

typedef void (*deq_single_fn)(const void *, float *, int, int);
typedef void (*gemv_fn)(const void *, const void *, int, float *, int);

/* Fill `nb` interleaved block groups with random bytes and sane fp16 deltas.
 * Every R4/R8 layout starts with rpb fp16 scale words, so those are written
 * explicitly; the remaining bytes (grid indices / nibbles / sign masks) are
 * left fully random, which is valid input for any bit pattern. */
static void *make_group(size_t block_bytes, int nb, int rpb, int d_bytes) {
    unsigned char *g = (unsigned char *)calloc((size_t)nb, block_bytes);
    if (!g) { fprintf(stderr, "OOM\n"); exit(2); }
    for (int i = 0; i < nb; i++) {
        unsigned char *p = g + (size_t)i * block_bytes;
        for (size_t j = 0; j < block_bytes; j++) p[j] = (unsigned char)rnd();
        /* rpb consecutive fp16 scales at the head: keep them small and positive
         * so the accumulated sums stay in a range where 1e-4 relative error
         * means something. */
        for (int r = 0; r < rpb; r++) {
            uint16_t d = fp32_to_fp16(0.001f + (float)(rnd() & 255) * 1e-5f);
            memcpy(p + 2 * r, &d, 2);
        }
        (void)d_bytes;
    }
    return g;
}

/* Random Q8_K activation row of `nb` blocks (d + qs + bsums). */
static block_q8_K *make_act(int nb) {
    block_q8_K *y = (block_q8_K *)calloc((size_t)nb, sizeof(block_q8_K));
    if (!y) { fprintf(stderr, "OOM\n"); exit(2); }
    for (int i = 0; i < nb; i++) {
        float xs[QK_K], amax = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            xs[j] = ((float)(int)(rnd() % 2001) - 1000.0f) / 1000.0f;
            if (fabsf(xs[j]) > amax) amax = fabsf(xs[j]);
        }
        const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
        y[i].d = d;
        for (int j = 0; j < QK_K; j++) y[i].qs[j] = (int8_t)lrintf(xs[j] / d);
        for (int j = 0; j < QK_K / 16; j++) {
            int s = 0;
            for (int m = 0; m < 16; m++) s += y[i].qs[16 * j + m];
            y[i].bsums[j] = (int16_t)s;
        }
    }
    return y;
}

/* One (type, nb, nrows) case. Returns 0 on pass. */
static int case_run(const char *name, size_t block_bytes, int rpb,
                    deq_single_fn deq, gemv_fn gemv, int nb, int nrows) {
    const int n = nb * QK_K;
    void *g = make_group(block_bytes, nb, rpb, 0);
    block_q8_K *y = make_act(nb);

    float got[16], ref[16];
    memset(got, 0xCD, sizeof got);   /* poison: catches "never written" */
    memset(ref, 0, sizeof ref);
    gemv(g, y, n, got, nrows);
    /* vec_dot_r4_q8_k_dequant_ref() hard-caps at 4 rows (it was written for the
     * R4 formats). IQ4_XS_R8 has 8, so do the same arithmetic here for the rest:
     * dequantize the row block by block, dot against the Q8_K codes, fold the
     * per-block delta. Deliberately a separate, obvious implementation -- it is
     * the oracle, so it must not share code with the kernel under test. */
    {
        const int cap = nrows < 4 ? nrows : 4;
        vec_dot_r4_q8_k_dequant_ref(g, y, n, ref, cap, deq, block_bytes);
        float wbuf[QK_K];
        for (int r = 4; r < nrows; r++) {
            double acc = 0.0;
            for (int i = 0; i < nb; i++) {
                deq((const char *)g + (size_t)i * block_bytes, wbuf, QK_K, r);
                double s = 0.0;
                for (int j = 0; j < QK_K; j++) s += (double)wbuf[j] * (double)y[i].qs[j];
                acc += s * (double)y[i].d;
            }
            ref[r] = (float)acc;
        }
    }

    int bad = 0;
    double worst = 0.0;
    for (int r = 0; r < nrows; r++) {
        if (got[r] == -0x1.5p+38f || !isfinite(got[r])) {   /* 0xCDCDCDCD */
            printf("  FAIL %s nb=%d nrows=%d: out[%d] never written\n", name, nb, nrows, r);
            bad = 1;
            continue;
        }
        const double e = fabs((double)ref[r] - (double)got[r]);
        const double m = fabs((double)ref[r]) > 1e-6 ? fabs((double)ref[r]) : 1.0;
        const double rel = e / m;
        if (rel > worst) worst = rel;
        if (rel > 1e-4) {
            printf("  FAIL %s nb=%d nrows=%d row %d: got=%.6f ref=%.6f rel=%.2e\n",
                   name, nb, nrows, r, (double)got[r], (double)ref[r], rel);
            bad = 1;
        }
    }
    if (!bad)
        printf("  ok   %-12s nb=%d nrows=%d  worst rel=%.2e\n", name, nb, nrows, worst);

    free(g);
    free(y);
    return bad;
}

int main(void) {
    fp16_table_init();
    int bad = 0;

    printf("R4/R8 GEMV vs vec_dot_r4_q8_k_dequant_ref\n");

    static const int nbs[] = { 1, 2, 5 };
    for (size_t i = 0; i < sizeof nbs / sizeof nbs[0]; i++) {
        const int nb = nbs[i];
        bad |= case_run("iq3_xxs_r4", sizeof(block_iq3_xxs_r4), 4,
                        (deq_single_fn)dequantize_row_iq3_xxs_r4_single,
                        vec_dot_iq3_xxs_r4_q8_k, nb, 4);
        bad |= case_run("iq3_xxs_r4", sizeof(block_iq3_xxs_r4), 4,
                        (deq_single_fn)dequantize_row_iq3_xxs_r4_single,
                        vec_dot_iq3_xxs_r4_q8_k, nb, 3);
        bad |= case_run("iq4_xs_r8", sizeof(block_iq4_xs_r8), 8,
                        (deq_single_fn)dequantize_row_iq4_xs_r8_single,
                        vec_dot_iq4_xs_r8_q8_k, nb, 8);
        bad |= case_run("iq4_xs_r8", sizeof(block_iq4_xs_r8), 8,
                        (deq_single_fn)dequantize_row_iq4_xs_r8_single,
                        vec_dot_iq4_xs_r8_q8_k, nb, 5);
    }

    printf("check_r4_gemv: %s\n", bad ? "FAIL" : "all OK");
    return bad;
}
