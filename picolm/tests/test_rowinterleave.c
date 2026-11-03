/* test_rowinterleave.c -- row-interleaved (R4/R8) quant layout regression test.
 *
 * WHY THIS EXISTS
 * ---------------
 * Two shipped bugs (see notes/picolm/qwen3.6_iq4_nl_r4_garbage_DEBUG_JOURNAL.txt)
 * were both "code assumed the wrong memory layout for an interleaved type":
 *
 *   BUG A  vec_dot() on an interleaved type only ever reads ROW 0 of a block
 *          group, and call sites addressed rows as base + row * row_bytes.
 *          For R4 that is wrong for 3 of every 4 rows.  -> NaN at layer 8.
 *   BUG B  r8a_repack_r4_iq4nl() decoded IQ4_NL_R4 nibbles with the Q4_0_4_4
 *          byte-contiguous layout instead of type 220's four-window spread.
 *          -> ~100% wrong prefill GEMM at batch >= 16.
 *
 * Neither was caught, because *no* test in check-all touched a single
 * interleaved type.  This file fixes that.  It is model-free (no GGUF), runs in
 * well under a second, and asserts on the two invariants that were violated:
 *
 *   1. EVERY logical row of a group dequantizes to what an independent scalar
 *      reference says it should -- not just row 0.
 *   2. GEMM output equals a scalar reference at batch sizes that straddle the
 *      decode-once R8 minimum-batch threshold (default 16), so the R8 repack
 *      path is actually exercised in CI.
 *
 * THE REFERENCE MUST BE INDEPENDENT.  Do not call the function under test to
 * build the expected value: the whole point is to encode the *file format*
 * by hand from the layout comment and compare.  (The original debugging session
 * wasted a cycle on a "reference" that reused the same pointer and the same
 * dequantizer, which therefore agreed with the bug.)
 *
 * Build:   make test_rowinterleave
 * Run:     ./test_rowinterleave [-v]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "quant.h"
#include "sgemm.h"

static int verbose = 0;
static int tests_run, tests_pass, tests_fail, tests_skipped;

static void pass_t(const char *n) {
    tests_pass++;
    if (verbose) printf("  PASS %s\n", n);
}
static void fail_t(const char *n, const char *why, double got, double want) {
    tests_fail++;
    fprintf(stderr, "  FAIL %s: %s (got %.6g want %.6g)\n", n, why, got, want);
}

/* ---------------------------------------------------------------- helpers */

static uint32_t rng_state = 0x12345678u;
static uint32_t rnd(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state >> 8; }
static float frand(void) { return (float)((rnd() & 0xFFFF) / 32768.0) - 1.0f; }

/* Aggregate comparison: rms(diff) / rms(ref).  NEVER per-element relative error
 * against a possibly-near-zero value -- that reports huge ratios for numerically
 * meaningless differences and is the classic false-positive generator. */
static double rel_rms(const float *got, const float *want, size_t n) {
    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - (double)want[i];
        se += d * d;
        sr += (double)want[i] * (double)want[i];
    }
    if (sr == 0.0) return se == 0.0 ? 0.0 : 1e30;
    return sqrt(se / sr);
}

static void check_rel(const char *name, const float *got, const float *want,
                      size_t n, double tol) {
    tests_run++;
    double r = rel_rms(got, want, n);
    if (r <= tol) pass_t(name);
    else fail_t(name, "rel_rms over tolerance", r, tol);
}

/* ======================================================================
 * Independent IQ4_NL_R4 row dequantizer, written from the layout comment
 * in quant.h / dequantize_row_iq4_nl_r4_single -- deliberately NOT calling
 * dequantize_row_iq4_nl_r4_single.
 *
 * Row r's 32 values come from four 4-byte windows spread across the whole
 * 64-byte qs[] of the block:
 *     qs[4r+i+ 0] -> value[i+ 0] (low nibble), value[i+ 8] (high nibble)
 *     qs[4r+i+16] -> value[i+16] (low),        value[i+24] (high)
 *     qs[4r+i+32] -> value[i+ 4] (low),        value[i+12] (high)
 *     qs[4r+i+48] -> value[i+20] (low),        value[i+28] (high)
 * ====================================================================== */
static void ref_iq4_nl_r4_row(const block_iq4_nl_r4 *x, int nb, int row, float *dst) {
    for (int ib = 0; ib < nb; ib++) {
        const float d = fp16_to_fp32(x[ib].d[row]);
        const uint8_t *qs = x[ib].qs;
        float *dp = dst + (size_t)ib * 32;
        for (int i = 0; i < 4; i++) {
            const uint8_t b0 = qs[4 * row + i +  0];
            const uint8_t b1 = qs[4 * row + i + 16];
            const uint8_t b2 = qs[4 * row + i + 32];
            const uint8_t b3 = qs[4 * row + i + 48];
            dp[i +  0] = d * (float)kvalues_iq4nl[b0 & 0xF];
            dp[i +  8] = d * (float)kvalues_iq4nl[b0 >> 4];
            dp[i + 16] = d * (float)kvalues_iq4nl[b1 & 0xF];
            dp[i + 24] = d * (float)kvalues_iq4nl[b1 >> 4];
            dp[i +  4] = d * (float)kvalues_iq4nl[b2 & 0xF];
            dp[i + 12] = d * (float)kvalues_iq4nl[b2 >> 4];
            dp[i + 20] = d * (float)kvalues_iq4nl[b3 & 0xF];
            dp[i + 28] = d * (float)kvalues_iq4nl[b3 >> 4];
        }
    }
}

/* Deterministic pseudo-random IQ4_NL_R4 weights: `nrows` logical rows, k values
 * each, packed rpb rows per 72-byte block.  Scales are kept small and sane so a
 * layout error shows up as a numeric mismatch rather than inf/NaN noise. */
static void fill_iq4_nl_r4(void *buf, int nrows, int k) {
    const int nb = k / 32, rpb = 4, groups = nrows / rpb;
    block_iq4_nl_r4 *w = (block_iq4_nl_r4 *)buf;
    for (int g = 0; g < groups; g++) {
        for (int ib = 0; ib < nb; ib++) {
            block_iq4_nl_r4 *b = &w[(size_t)g * nb + ib];
            for (int r = 0; r < rpb; r++) b->d[r] = fp32_to_fp16(0.01f + (float)(rnd() & 255) * 1e-4f);
            for (int j = 0; j < 64; j++) b->qs[j] = (uint8_t)rnd();
        }
    }
}

/* ---- Test 1: EVERY row of the group, not just row 0 --------------------
 * This is the assertion that BUG A would have failed.  The old vec_dot path
 * used row 0 unconditionally, so rows 1..3 were silently wrong. */
static void test_iq4_nl_r4_rows(void) {
    enum { NROWS = 8, K = 256 };
    const int nb = K / 32;
    const size_t wbytes = (size_t)(NROWS / 4) * nb * sizeof(block_iq4_nl_r4);
    block_iq4_nl_r4 *W = (block_iq4_nl_r4 *)malloc(wbytes);
    fill_iq4_nl_r4(W, NROWS, K);

    float *x = (float *)malloc(K * sizeof(float));
    float *want = (float *)malloc(K * sizeof(float));
    float *got = (float *)malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) x[i] = frand();

    char name[128];
    for (int row = 0; row < NROWS; row++) {
        ref_iq4_nl_r4_row((const block_iq4_nl_r4 *)gguf_type_row_ptr(W, GGUF_TYPE_IQ4_NL_R4,
                                                                     gguf_type_row_size(GGUF_TYPE_IQ4_NL_R4, K), row),
                          nb, gguf_type_row_in_block(GGUF_TYPE_IQ4_NL_R4, row), want);
        dequantize_row_iq4_nl_r4_single((const block_iq4_nl_r4 *)gguf_type_row_ptr(W, GGUF_TYPE_IQ4_NL_R4,
                                          gguf_type_row_size(GGUF_TYPE_IQ4_NL_R4, K), row),
                                        got, K, gguf_type_row_in_block(GGUF_TYPE_IQ4_NL_R4, row));
        snprintf(name, sizeof name, "IQ4_NL_R4 dequant row %d", row);
        check_rel(name, got, want, K, 1e-6);
    }

    /* Row-aware dot must match dequantize-then-dot for every row. */
    for (int row = 0; row < NROWS; row++) {
        ref_iq4_nl_r4_row((const block_iq4_nl_r4 *)gguf_type_row_ptr(W, GGUF_TYPE_IQ4_NL_R4,
                                                                     gguf_type_row_size(GGUF_TYPE_IQ4_NL_R4, K), row),
                          nb, gguf_type_row_in_block(GGUF_TYPE_IQ4_NL_R4, row), want);
        double want_dot = 0.0;
        for (int i = 0; i < K; i++) want_dot += (double)want[i] * x[i];
        float got_dot = vec_dot_row(W, x, K, GGUF_TYPE_IQ4_NL_R4, row);
        snprintf(name, sizeof name, "IQ4_NL_R4 vec_dot_row row %d", row);
        tests_run++;
        double r = fabs((double)got_dot - want_dot) / (fabs(want_dot) + 1e-30);
        if (r <= 1e-5) pass_t(name);
        else fail_t(name, "dot mismatch vs independent reference", got_dot, want_dot);
    }

    /* Regression guard for BUG A specifically: rows 1..3 must NOT be silently
     * equal to row 0 (which is what the old unconditional-row-0 read produced). */
    tests_run++;
    {
        int distinct = 1;
        for (int row = 1; row < 4; row++) {
            float a = vec_dot_row(W, x, K, GGUF_TYPE_IQ4_NL_R4, 0);
            float b = vec_dot_row(W, x, K, GGUF_TYPE_IQ4_NL_R4, row);
            if (fabsf(a - b) < 1e-9f) distinct = 0;
        }
        if (distinct) pass_t("IQ4_NL_R4 rows 1..3 differ from row 0 (no row-0 aliasing)");
        else fail_t("IQ4_NL_R4 rows 1..3 differ from row 0 (no row-0 aliasing)",
                    "vec_dot_row returned row 0 for every row -- BUG A regression", 0, 1);
    }

    free(W); free(x); free(want); free(got);
}

/* ---- Test 2: GEMM across the R8 minimum-batch threshold ----------------
 * BUG B lived in the decode-once R8 prefill GEMM, which only engages at
 * ncols >= PICOLM_Q4_R8_MIN_N (default 16) and nrows % 8 == 0.  Testing a
 * single batch size -- or only small ones -- misses it entirely. */
static void test_iq4_nl_r4_gemm(void) {
    /* Small enough to run in milliseconds, large enough to engage R8. */
    enum { NROWS = 32, K = 256 };
    const int nb = K / 32;
    const size_t wbytes = (size_t)(NROWS / 4) * nb * sizeof(block_iq4_nl_r4);
    block_iq4_nl_r4 *W = (block_iq4_nl_r4 *)malloc(wbytes);
    fill_iq4_nl_r4(W, NROWS, K);

    static const int batches[] = { 1, 3, 15, 16, 17, 33 };
    const int nmax = 33;

    float *x = (float *)malloc((size_t)nmax * K * sizeof(float));
    for (int i = 0; i < nmax * K; i++) x[i] = frand();

    /* Reference: dequantize every weight row once, then plain f32 dot. */
    float *Wf = (float *)malloc((size_t)NROWS * K * sizeof(float));
    for (int r = 0; r < NROWS; r++) {
        ref_iq4_nl_r4_row((const block_iq4_nl_r4 *)gguf_type_row_ptr(W, GGUF_TYPE_IQ4_NL_R4,
                                                                    gguf_type_row_size(GGUF_TYPE_IQ4_NL_R4, K), r),
                          nb, gguf_type_row_in_block(GGUF_TYPE_IQ4_NL_R4, r), Wf + (size_t)r * K);
    }

    /* Quantize activations the way matmul_batch does (Q8_0, ACT_FMT_Q8_0). */
    const int qrb = (K / 32) * (int)sizeof(block_q8_0);
    block_q8_0 *Q = (block_q8_0 *)malloc((size_t)nmax * qrb);
    for (int b = 0; b < nmax; b++) quantize_row_q8_0(x + (size_t)b * K, (char *)Q + (size_t)b * qrb, K);

    char name[128];
    for (size_t bi = 0; bi < sizeof batches / sizeof batches[0]; bi++) {
        const int nc = batches[bi];
        /* Dequantized-activation reference path: reference must be independent
         * of the kernel, so dequantize Q back to f32 and dot that. */
        float *Xf = (float *)malloc((size_t)nc * K * sizeof(float));
        for (int b = 0; b < nc; b++) dequantize_row_q8_0((const char *)Q + (size_t)b * qrb, Xf + (size_t)b * K, K);

        float *want = (float *)malloc((size_t)nc * NROWS * sizeof(float));
        for (int b = 0; b < nc; b++)
            for (int r = 0; r < NROWS; r++) {
                double acc = 0.0;
                for (int i = 0; i < K; i++) acc += (double)Wf[(size_t)r * K + i] * Xf[(size_t)b * K + i];
                want[(size_t)b * NROWS + r] = (float)acc;
            }

        float *got = (float *)calloc((size_t)nc * NROWS, sizeof(float));
        /* out[c * bs + r], bs = NROWS -- the convention gemm_table_task uses. */
        /* Call the table's registered fn_gemm target directly: pulling in
         * type_dispatch.c would drag the thread pool into this test, and the
         * point here is the kernel, not the dispatch plumbing. */
        int ok = sgemm_iq4_nl_r4_q8_0_best(NROWS, nc, K, W, Q, got, NROWS, 0, 1);

        snprintf(name, sizeof name, "IQ4_NL_R4 GEMM ncols=%d", nc);
        tests_run++;
        if (!ok) {
#if defined(__AVX2__) || defined(__ARM_NEON)
            /* A SIMD build MUST have this GEMM. Returning 0 here means the
             * kernel was compiled out or its entry point regressed. */
            fail_t(name, "fn_gemm returned 0 in a SIMD build", 0, 1);
#else
            /* Scalar-only build: there is no R4 GEMM to test. Not a failure,
             * just no coverage -- say so rather than pass silently. */
            tests_skipped++;
            if (verbose) printf("  SKIP %s (scalar build, no R4 GEMM)\n", name);
#endif
        } else {
            double r = rel_rms(got, want, (size_t)nc * NROWS);
            /* Tolerance covers int8 activation quantisation (~0.5% at k=256). */
            if (r <= 1e-2) pass_t(name);
            else fail_t(name, "rel_rms over tolerance", r, 1e-2);
        }
        free(Xf); free(want); free(got);
    }

    free(W); free(x); free(Wf); free(Q);
}

/* ---- Test 3: the layout helpers themselves -----------------------------
 * gguf_type_row_ptr/row_in_block are the abstraction both fixes rely on. */
static void test_layout_helpers(void) {
    const int k = 256;
    const size_t rs = gguf_type_row_size(GGUF_TYPE_IQ4_NL_R4, k);

    tests_run++;
    if (gguf_type_rows_per_block(GGUF_TYPE_IQ4_NL_R4) == 4) pass_t("rows_per_block(IQ4_NL_R4) == 4");
    else fail_t("rows_per_block(IQ4_NL_R4) == 4", "wrong value",
                gguf_type_rows_per_block(GGUF_TYPE_IQ4_NL_R4), 4);

    tests_run++;
    if (gguf_type_rows_per_block(GGUF_TYPE_Q8_0) == 1) pass_t("rows_per_block(Q8_0) == 1");
    else fail_t("rows_per_block(Q8_0) == 1", "plain type must be 1",
                gguf_type_rows_per_block(GGUF_TYPE_Q8_0), 1);

    /* Rows 0..3 share one group base; 4..7 share the next. */
    char *base = (char *)calloc(1, 8 * rs + 64);
    tests_run++;
    {
        int ok = 1;
        for (int r = 0; r < 4; r++) {
            if ((char *)gguf_type_row_ptr(base, GGUF_TYPE_IQ4_NL_R4, rs, r) != base) ok = 0;
            if (gguf_type_row_in_block(GGUF_TYPE_IQ4_NL_R4, r) != r) ok = 0;
        }
        for (int r = 4; r < 8; r++) {
            if ((char *)gguf_type_row_ptr(base, GGUF_TYPE_IQ4_NL_R4, rs, r) != base + 4 * rs) ok = 0;
            if (gguf_type_row_in_block(GGUF_TYPE_IQ4_NL_R4, r) != r - 4) ok = 0;
        }
        if (ok) pass_t("row_ptr groups 0-3 / 4-7 correctly");
        else fail_t("row_ptr groups 0-3 / 4-7 correctly", "group addressing wrong", 0, 1);
    }

    /* A plain type must keep the simple row*stride addressing. */
    tests_run++;
    if ((char *)gguf_type_row_ptr(base, GGUF_TYPE_Q8_0, rs, 3) == base + 3 * rs &&
        gguf_type_row_in_block(GGUF_TYPE_Q8_0, 3) == 0)
        pass_t("row_ptr(Q8_0) is row*stride");
    else
        fail_t("row_ptr(Q8_0) is row*stride", "plain-type addressing changed", 0, 1);

    free(base);
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "-v") == 0) verbose = 1;

    /* The decode-once R8 prefill path is gated by PICOLM_R8_MIN_BLOCKS
     * (default 16384 blocks = e.g. 512 rows at k=1024).  A CI-sized test can
     * never reach that, so the gate -- not just the batch threshold -- is what
     * kept BUG B invisible.  Force it open here: the whole point of this test
     * is to exercise the fast path at a shape small enough to run in CI.
     * setenv before any kernel reads it (they cache on first call). */
    setenv("PICOLM_R8_MIN_BLOCKS", "0", 1);

    printf("PicoLM row-interleaved (R4/R8) layout tests\n");
    printf("Platform:");
#if defined(PICOLM_AVX512)
    printf(" AVX-512");
#elif defined(PICOLM_AVX2)
    printf(" AVX2");
#elif defined(PICOLM_NEON)
    printf(" NEON");
#else
    printf(" Scalar");
#endif
    printf("\n");

    fp16_table_init();

    test_layout_helpers();
    test_iq4_nl_r4_rows();
    test_iq4_nl_r4_gemm();

    if (tests_skipped)
        printf("%d skipped (no SIMD GEMM in this build)\n", tests_skipped);
    printf("\n%d tests: %d passed, %d failed\n", tests_run, tests_pass, tests_fail);
    return tests_fail > 0 ? 1 : 0;
}
