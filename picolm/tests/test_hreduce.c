/* Regression test: horizontal SIMD reductions.
 *
 * hreduce256_ps() used the form
 *     t3 = _mm_add_ss(t3, _mm_movehl_ps(t3, t3));
 * _mm_movehl_ps(dst, src) returns {dst[2], dst[3], src[2], src[3]} -- it writes
 * lanes 2..3 and only reads lanes 2..3 of `src`. So that line added t3[0] to
 * itself and the following shuffle+add re-added lane 0, giving
 * 2*lane0 + lane1 + lane2 instead of lane0+lane1+lane2+lane3. Lanes 3 and 7
 * were dropped and lane 0 double-counted, i.e. the 8-lane sum came out 0.75x
 * too small.
 *
 * Because every SSM layer's query/state dot products go through this helper,
 * that single line made AVX2 builds produce different (worse) output than
 * AVX-512 on the same model. All-ones is the sharpest check: the correct sum
 * of eight 1.0f is exactly 8.0f, while the buggy form returns 6.0f.
 *
 * Build:
 *   cc -O2 -I. -DPICOLM_AVX2=1 -mavx2 -mf16c -o /tmp/t tests/test_hreduce.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

/* Use the real implementation, never a copy: a duplicated helper drifts from
 * the shipping one and the test then proves nothing. */
#include "model.h"
#include "model_internal.h"

static int failures;

static void check_exact(__m256 v, float want, const char *what) {
    float got = hreduce256_ps(v);
    if (got != want) {
        printf("FAIL %s: got %g want %g\n", what, got, want);
        failures++;
    }
}

int main(void) {
    /* Sharpest case: the buggy implementation returns 6.0f here. */
    check_exact(_mm256_set1_ps(1.0f), 8.0f, "all-ones");

    /* Box filter: lane values 0..7 must total 28. */
    check_exact(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), 28.0f, "0..7");

    /* Isolate each lane: dropping lane 3 or 7, or double-counting lane 0,
     * shows up immediately with a single non-zero lane. */
    for (int lane = 0; lane < 8; lane++) {
        float v[8] = {0};
        v[lane] = 1.0f;
        char msg[64];
        snprintf(msg, sizeof(msg), "only lane %d set", lane);
        check_exact(_mm256_loadu_ps(v), 1.0f, msg);
    }

    /* Lane 0 and lane 7 must each contribute exactly once. */
    {
        float v[8] = {0};
        v[0] = 100.0f; v[7] = 1.0f;
        check_exact(_mm256_loadu_ps(v), 101.0f, "lane0 + lane7");
    }

    /* Randomised cross-check against a scalar sum. */
    srand(1234);
    int rnd_bad = 0;
    for (int t = 0; t < 5000; t++) {
        float v[8]; double s = 0;
        for (int i = 0; i < 8; i++) {
            v[i] = (float)(((rand() % 20000) - 10000) / 1000.0);
            s += v[i];
        }
        float got = hreduce256_ps(_mm256_loadu_ps(v));
        double tol = 1e-5 * (fabs(s) > 1.0 ? fabs(s) : 1.0);
        if (fabs((double)got - s) > tol) {
            if (rnd_bad < 3) printf("FAIL random: got %g want %g\n", got, s);
            rnd_bad++;
        }
    }
    if (rnd_bad) { printf("FAIL %d/5000 randomised sums\n", rnd_bad); failures++; }

    if (failures) { printf("%d FAILURES\n", failures); return 1; }
    printf("hreduce256_ps: all checks OK\n");
    return 0;
}
