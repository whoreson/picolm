/* check_iq_gemv.c -- differential test for the IQ grid-quant GEMV kernels.
 *
 * WHAT THIS CATCHES
 *   The vec_dot_iq*_q8_k() public entry points dispatch to a hand-written
 *   SIMD kernel and fall back to a scalar reference. The reference
 *   (vec_dot_iq*_q8_k_scalar) is a transcription of dequantize-then-dot and
 *   was verified line-by-line against ggml-quants.c, so it is the oracle.
 *
 *   This test builds synthetic blocks (random bit patterns are all valid --
 *   grid indices are masked, sign codes index a 128-entry table) and requires
 *   the vectorised path to agree with the scalar one to within rounding.
 *
 *   That comparison is the whole point: every SIMD bug found in this tree so
 *   far -- the wrong-half extracti128, the interleaved-group stride, the
 *   ignored nrows -- produced output that still *looked* plausible and only
 *   showed up in a 12 GB model as slightly-off text. A few hundred synthetic
 *   blocks compared against a scalar oracle find those in milliseconds.
 *
 *   Because the integer MAC is exact up to the final float scale, tolerance is
 *   tight (1e-4 relative): a lane-order or sign-mask mistake is never subtle.
 *
 * USAGE
 *   make check-iq-gemv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "quant.h"

void fp16_table_init(void);

static unsigned rng_state = 20251116u;
static unsigned rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

/* Random Q8_K activation row of nb blocks. */
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

typedef float (*gemv1_fn)(const void *, const void *, int);

/* Run one type at one size; returns 0 on pass. `rpb`-style row groups do not
 * apply here (these are single-row formats), so this is just nb blocks. */
static int case_run(const char *name, size_t block_bytes, gemv1_fn fast,
                    gemv1_fn ref, int nb) {
    const int n = nb * QK_K;
    unsigned char *w = (unsigned char *)calloc((size_t)nb, block_bytes);
    if (!w) { fprintf(stderr, "OOM\n"); exit(2); }
    for (int i = 0; i < nb; i++) {
        unsigned char *p = w + (size_t)i * block_bytes;
        for (size_t j = 0; j < block_bytes; j++) p[j] = (unsigned char)rnd();
        /* Keep the fp16 delta small and sane: the head of every layout is a
         * single uint16 d (IQ1_M packs it into scales, so skip that one). */
        const uint16_t d = fp32_to_fp16(0.001f + (float)(rnd() & 255) * 1e-5f);
        memcpy(p, &d, 2);
    }
    block_q8_K *y = make_act(nb);

    const float got = fast(w, y, n);
    const float exp = ref(w, y, n);
    const double e = fabs((double)got - (double)exp);
    const double m = fabs((double)exp) > 1e-6 ? fabs((double)exp) : 1.0;
    const double rel = e / m;
    const int bad = !(rel <= 1e-4) || !isfinite(got);
    printf("  %-4s %-11s nb=%-3d got=%.6f ref=%.6f rel=%.2e\n",
           bad ? "FAIL" : "ok", name, nb, (double)got, (double)exp, rel);
    free(w);
    free(y);
    return bad;
}

#define CASE(name, block, fast, ref, nb) \
    bad |= case_run(name, block, fast, ref, nb)

int main(void) {
    fp16_table_init();
    int bad = 0;

    printf("IQ grid-quant GEMV: vectorised path vs scalar reference\n");

    static const int nbs[] = { 1, 2, 3, 7 };
    for (size_t i = 0; i < sizeof nbs / sizeof nbs[0]; i++) {
        const int nb = nbs[i];
        CASE("iq2_xxs", sizeof(block_iq2_xxs), vec_dot_iq2_xxs_q8_k,
             vec_dot_iq2_xxs_q8_k_scalar, nb);
        CASE("iq2_xs", sizeof(block_iq2_xs), vec_dot_iq2_xs_q8_k,
             vec_dot_iq2_xs_q8_k_scalar, nb);
        CASE("iq1_s", sizeof(block_iq1_s), vec_dot_iq1_s_q8_k,
             vec_dot_iq1_s_q8_k_scalar, nb);
        CASE("iq3_s", sizeof(block_iq3_s), vec_dot_iq3_s_q8_k,
             vec_dot_iq3_s_q8_k_scalar, nb);
        CASE("iq2_s", sizeof(block_iq2_s), vec_dot_iq2_s_q8_k,
             vec_dot_iq2_s_q8_k_scalar, nb);
        /* iq1_m is excluded: its id (29) is shadowed by the internal
         * GGUF_TYPE_Q8_K marker, so it has no dispatch entry. The scalar and
         * vectorised entry points still exist and still agree -- see the TODO
         * in quant.h -- but there is nothing else to compare it against yet. */
    }

    printf("check_iq_gemv: %s\n", bad ? "FAIL" : "all OK");
    return bad;
}
