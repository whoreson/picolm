/* test_bf16_simd.c -- Differential test: BF16 SIMD dequant/vec_dot vs scalar ref.
 *
 * Compares the SIMD implementations of vec_dot_bf16_f32() and
 * dequantize_row_bf16() against the scalar bf16_to_fp32() path.
 *
 * BF16 bitcast is exact (shift << 16), so all SIMD paths must agree
 * bit-exact with the scalar reference. Any deviation is a bug.
 *
 * Also benchmarks each function at realistic sizes.
 *
 * Build: make test_bf16_simd
 * Run:   ./test_bf16_simd [-v] [-b reps]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "quant.h"

/* These are defined in quant.c but not declared in quant.h */
float vec_dot_bf16_f32(const void *src, const float *x, int n);
void dequantize_row_bf16(const void *src, float *dst, int n);

static int verbose = 0;
static int bench_reps = 1000;
static int tests_run, tests_pass, tests_fail;

static void pass_t(const char *n) { tests_pass++; if (verbose) printf("  PASS %s\n", n); }
static void fail_t(const char *n, const char *msg) {
    tests_fail++;
    fprintf(stderr, "  FAIL %s: %s\n", n, msg);
}

static uint32_t rs = 0xBEEFu;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

static inline double get_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

/* Generate random bf16 values: normal floats, infs, nans, zeros. */
static void fill_bf16(uint16_t *buf, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t r = rnd();
        if (r % 100 == 0) buf[i] = 0x7F80;       /* +inf */
        else if (r % 100 == 1) buf[i] = 0xFF80;   /* -inf */
        else if (r % 100 == 2) buf[i] = 0x7FFF;   /* nan */
        else if (r % 100 == 3) buf[i] = 0;         /* zero */
        else {
            /* Random normal: sign, exp 1..254, mantissa */
            uint32_t fbits = (r & 0x80000000u) |
                              (((r >> 8) % 254 + 1) << 23) |
                              (r & 0x7FFFFF);
            buf[i] = (uint16_t)(fbits >> 16);
        }
    }
}

static void fill_f32(float *buf, int n) {
    for (int i = 0; i < n; i++) {
        buf[i] = ((float)(rnd() & 0xFFFF) - 32767.5f) * 1e-3f;
    }
}

/* Scalar reference for vec_dot_bf16_f32 */
static float vec_dot_bf16_f32_ref(const uint16_t *bf16, const float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += bf16_to_fp32(bf16[i]) * x[i];
    return sum;
}

/* Scalar reference for dequantize_row_bf16 */
static void dequantize_row_bf16_ref(const uint16_t *bf16, float *dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = bf16_to_fp32(bf16[i]);
}

static void test_vec_dot_sizes(void) {
    /* Test various sizes including non-aligned counts. */
    static const int sizes[] = {1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64,
                                 127, 128, 255, 256, 511, 512, 1024, 2048, 4096, 8192};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        uint16_t *bf16 = malloc(n * sizeof(uint16_t));
        float *x = malloc(n * sizeof(float));
        float *dst = calloc(n * 2, sizeof(float)); /* simd in [0..n), ref in [n..2n) */

        fill_bf16(bf16, n);
        fill_f32(x, n);

        /* vec_dot */
        float got = vec_dot_bf16_f32(bf16, x, n);
        float want = vec_dot_bf16_f32_ref(bf16, x, n);

        /* Both should handle inf/nan consistently. For normal results,
         * the bitcast is exact so results should match to float rounding.
         * Use a generous tolerance for accumulated rounding differences. */
        if (isnan(want)) {
            if (!isnan(got))
                fail_t("vec_dot n=%d", "expected nan");
            else { tests_run++; tests_pass++; if (verbose) printf("  PASS vec_dot n=%d (nan)\n", n); }
        } else if (isinf(want)) {
            if (!isinf(got) || (got > 0) != (want > 0))
                fail_t("vec_dot n=%d", "expected inf");
            else { tests_run++; tests_pass++; if (verbose) printf("  PASS vec_dot n=%d (inf)\n", n); }
        } else {
            double rel = fabs((double)got - (double)want) / (fabs((double)want) + 1e-30);
            if (rel > 1e-6) {
                fprintf(stderr, "  FAIL vec_dot n=%d: rel_err=%.3g (got=%.9g want=%.9g)\n",
                        n, rel, (double)got, (double)want);
                tests_fail++;
            } else {
                tests_run++; tests_pass++;
                if (verbose) printf("  PASS vec_dot n=%d (rel_err=%.3e)\n", n, rel);
            }
        }
        tests_run++;

        /* dequantize_row */
        dequantize_row_bf16(bf16, dst, n);
        dequantize_row_bf16_ref(bf16, dst + n, n);

        int deq_ok = 1;
        for (int i = 0; i < n && deq_ok; i++) {
            if (memcmp(&dst[i], &dst[n + i], sizeof(float)) != 0) {
                deq_ok = 0;
                fprintf(stderr, "  FAIL dequant n=%d i=%d: got=0x%08x want=0x%08x\n",
                        n, i, *(uint32_t*)&dst[i], *(uint32_t*)&dst[n + i]);
            }
        }
        tests_run++;
        if (deq_ok) { tests_pass++; if (verbose) printf("  PASS dequant n=%d\n", n); }
        else tests_fail++;

        free(bf16); free(x); free(dst);
    }
}

typedef struct { int n; const char *label; } bench_shape_t;

static void benchmark(void) {
    /* Benchmark at realistic sizes. */
    static const bench_shape_t shapes[] = {
        {256, "hd=256"}, {1024, "hd=1024"}, {4096, "hd=4096"},
        {8192, "hd=8192"}, {14336, "hd=14336"}, {40960, "hd=40960"}
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    printf("\nBenchmark (PICOLM_BF16_BENCH=%d reps):\n", bench_reps);
    printf("  %-12s  %10s  %10s\n", "shape", "vec_dot ms", "deq ms");

    for (int si = 0; si < n_shapes; si++) {
        int n = shapes[si].n;
        uint16_t *bf16 = malloc(n * sizeof(uint16_t));
        float *x = malloc(n * sizeof(float));
        float *dst = malloc(n * sizeof(float));
        fill_bf16(bf16, n);
        fill_f32(x, n);

        /* vec_dot benchmark */
        double t0 = get_ms();
        volatile float sink = 0;
        for (int r = 0; r < bench_reps; r++)
            sink += vec_dot_bf16_f32(bf16, x, n);
        double vd_ms = (get_ms() - t0) / bench_reps;
        (void)sink;

        /* dequant benchmark */
        t0 = get_ms();
        for (int r = 0; r < bench_reps; r++)
            dequantize_row_bf16(bf16, dst, n);
        double dq_ms = (get_ms() - t0) / bench_reps;

        printf("  %-12s  %10.4f  %10.4f\n", shapes[si].label, vd_ms, dq_ms);

        free(bf16); free(x); free(dst);
    }
}

int main(int argc, char **argv) {
    fp16_table_init();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) verbose = 1;
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc)
            bench_reps = atoi(argv[++i]);
    }

    printf("BF16 SIMD correctness test\n");
    test_vec_dot_sizes();

    printf("\n%d tests: %d pass, %d fail\n", tests_run, tests_pass, tests_fail);

    if (tests_fail == 0) {
        benchmark();
    }

    return tests_fail > 0 ? 1 : 0;
}
