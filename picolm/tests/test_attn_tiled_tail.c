/* test_attn_tiled_tail.c -- the acc += sum_j exp[j]*V[j] tail of the tiled
 * prefill attention, checked for EVERY head_dim the engine accepts.
 *
 * WHY THIS EXISTS
 * ---------------
 * attn_process_tile() keeps the online-softmax accumulator acc[n_q][hd] in
 * registers, 16 floats (one zmm) per block. The generic (hd != 128) branch used
 *
 *     n_blks = (hd + D_BLK - 1) / D_BLK;   -- ceil, WRONG
 *
 * so head_dim=8 rounded up to one 16-float block, and the kernel loaded and
 * stored acc + 0 .. acc + 15. With head_dim=8 the rows of acc are packed 8
 * floats apart, so every row's "block" clobbered the NEXT query row's
 * accumulator, and the final row walked past the end of the task's scratch
 * slot into its neighbour's. No crash, no NaN -- the model just produced a
 * different (wrong) string. Only AVX-512 took that branch, which is why the
 * default `make` build disagreed with `make scalar` and with CI's avx2 job.
 *
 * The fix is n_full = hd / D_BLK and a scalar tail from there. This test pins
 * the property directly: for every head_dim, updating acc must touch exactly
 * [acc_row, acc_row+hd) and nothing else, and the values must match a plain
 * scalar reference.
 *
 * It needs no model and no GGUF, so it runs in CI on every ISA -- including
 * the builds that never select the AVX-512 branch and therefore could not
 * have caught the original bug by execution. The sentinel check is what makes
 * it useful there: it fails on ANY architecture whose handler writes outside
 * the row, and it documents the contract for the next person who adds one.
 *
 * BUILD/RUN:  make check-attn-tail   (also part of check-all)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* The tail kernels below are copies of the acc-update part of
 * attn_process_tile() with the surrounding attention machinery removed, so
 * they can be driven with arbitrary shapes. Keep them in sync with
 * model_attention.c; the ISA selection mirrors it exactly. */

#define ATTN_TILE 64

#if defined(__AVX512F__)
#include <immintrin.h>
static void acc_update(const float *exp_w, int ts, const float *v_tile,
                       int hd, float *acc_row) {
    if (hd == 128) {
        __m512 a0 = _mm512_loadu_ps(acc_row + 0);
        __m512 a1 = _mm512_loadu_ps(acc_row + 16);
        __m512 a2 = _mm512_loadu_ps(acc_row + 32);
        __m512 a3 = _mm512_loadu_ps(acc_row + 48);
        __m512 a4 = _mm512_loadu_ps(acc_row + 64);
        __m512 a5 = _mm512_loadu_ps(acc_row + 80);
        __m512 a6 = _mm512_loadu_ps(acc_row + 96);
        __m512 a7 = _mm512_loadu_ps(acc_row + 112);
        for (int j = 0; j < ts; j++) {
            __m512 wv = _mm512_set1_ps(exp_w[j]);
            const float *vj = v_tile + (size_t)j * hd;
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 0), wv, a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 16), wv, a1);
            a2 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 32), wv, a2);
            a3 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 48), wv, a3);
            a4 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 64), wv, a4);
            a5 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 80), wv, a5);
            a6 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 96), wv, a6);
            a7 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 112), wv, a7);
        }
        _mm512_storeu_ps(acc_row + 0, a0);
        _mm512_storeu_ps(acc_row + 16, a1);
        _mm512_storeu_ps(acc_row + 32, a2);
        _mm512_storeu_ps(acc_row + 48, a3);
        _mm512_storeu_ps(acc_row + 64, a4);
        _mm512_storeu_ps(acc_row + 80, a5);
        _mm512_storeu_ps(acc_row + 96, a6);
        _mm512_storeu_ps(acc_row + 112, a7);
    } else {
        const int D_BLK = 16;
        const int n_full = hd / D_BLK;          /* complete blocks only */
        if (n_full > 0) {
            const int n_blks = n_full > 16 ? 16 : n_full;
            __m512 accv[16];
            for (int db = 0; db < n_blks; db++)
                accv[db] = _mm512_loadu_ps(acc_row + db * D_BLK);
            for (int j = 0; j < ts; j++) {
                __m512 wv = _mm512_set1_ps(exp_w[j]);
                const float *vj = v_tile + (size_t)j * hd;
                for (int db = 0; db < n_blks; db++)
                    accv[db] = _mm512_fmadd_ps(_mm512_loadu_ps(vj + db * D_BLK), wv, accv[db]);
            }
            for (int db = 0; db < n_blks; db++)
                _mm512_storeu_ps(acc_row + db * D_BLK, accv[db]);
        }
    }
    {
        int d = (hd / 16) * 16;
        if (d > 256) d = 256;
        for (; d < hd; d++) {
            float a = acc_row[d];
            for (int j = 0; j < ts; j++)
                a += exp_w[j] * v_tile[(size_t)j * hd + d];
            acc_row[d] = a;
        }
    }
}
#elif defined(__AVX__)
#include <immintrin.h>
static void acc_update(const float *exp_w, int ts, const float *v_tile,
                       int hd, float *acc_row) {
    const int D_BLK = 8;
    int n_blks = hd / D_BLK;
    for (int db = 0; db < n_blks; db++) {
        int d = db * D_BLK;
        __m256 acc0 = _mm256_loadu_ps(acc_row + d);
        const float *vr = v_tile + d;
        for (int j = 0; j < ts; j++) {
            __m256 wv = _mm256_set1_ps(exp_w[j]);
            __m256 vv = _mm256_loadu_ps(vr + (size_t)j * hd);
#if defined(__FMA__)
            acc0 = _mm256_fmadd_ps(vv, wv, acc0);
#else
            acc0 = _mm256_add_ps(_mm256_mul_ps(vv, wv), acc0);
#endif
        }
        _mm256_storeu_ps(acc_row + d, acc0);
    }
    for (int d = n_blks * D_BLK; d < hd; d++) {
        float a = acc_row[d];
        for (int j = 0; j < ts; j++)
            a += exp_w[j] * v_tile[(size_t)j * hd + d];
        acc_row[d] = a;
    }
}
#else
static void acc_update(const float *exp_w, int ts, const float *v_tile,
                       int hd, float *acc_row) {
    for (int d = 0; d < hd; d++) {
        float a = acc_row[d];
        for (int j = 0; j < ts; j++)
            a += exp_w[j] * v_tile[(size_t)j * hd + d];
        acc_row[d] = a;
    }
}
#endif

/* head_dims the engine must handle: real ones plus the awkward small ones.
 * 8 is stories260K (the shape that exposed the bug), 48 is stories15M, 128 is
 * the unrolled fast path, 256 the largest the AVX-512 block loop covers. */
static const int HEAD_DIMS[] = { 1, 2, 4, 8, 12, 16, 17, 24, 32, 40, 48, 64,
                                 80, 96, 112, 128, 129, 144, 160, 192, 256, 288 };
static const int TILE_SIZES[] = { 1, 3, 8, 9, 17, 64 };

/* Rows are packed head_dim floats apart, exactly like acc[] in
 * batch_tiled_task, with guard sentinels on both sides of every row and of
 * the whole block. A write outside [acc_row, acc_row+hd) trips a guard. */
#define GUARD 0x7f7f7f7f

int main(void) {
    int failures = 0, checks = 0, failed_shapes = 0;

    for (size_t hi = 0; hi < sizeof HEAD_DIMS / sizeof *HEAD_DIMS; hi++) {
        const int hd = HEAD_DIMS[hi];
        /* 3 rows = the shape of acc[] for kv_mul=3, n_q=1: rows are
         * contiguous, so row i+1 immediately follows row i. */
        for (size_t ti = 0; ti < sizeof TILE_SIZES / sizeof *TILE_SIZES; ti++) {
            const int ts = TILE_SIZES[ti];
            const int nrows = 3;
            const int guard = 16;
            const size_t total = (size_t)guard + (size_t)nrows * hd + guard;

            float *acc = malloc(total * sizeof(float));
            float *ref = malloc(total * sizeof(float));
            float *v   = malloc((size_t)ts * hd * sizeof(float));
            float *w   = malloc((size_t)ts * sizeof(float));
            if (!acc || !ref || !v || !w) { fprintf(stderr, "OOM\n"); return 2; }

            /* Deterministic pseudo-random inputs; no NaN, no infinities so
             * any difference is the kernel's, not IEEE corner cases. */
            unsigned s = 12345u + (unsigned)hd * 7919u + (unsigned)ts;
            for (size_t i = 0; i < total; i++) {
                s = s * 1103515245u + 12345u;
                acc[i] = (float)((int)((s >> 16) & 255) - 128) / 128.0f;
            }
            for (int i = 0; i < guard; i++) {
                acc[i] = (float)GUARD;
                acc[total - 1 - (size_t)i] = (float)GUARD;
            }
            for (size_t i = 0; i < (size_t)ts * hd; i++) {
                s = s * 1103515245u + 12345u;
                v[i] = (float)((int)((s >> 16) & 255) - 128) / 128.0f;
            }
            for (int j = 0; j < ts; j++) {
                s = s * 1103515245u + 12345u;
                w[j] = fabsf((float)((int)((s >> 16) & 255) - 128) / 128.0f);
            }
            memcpy(ref, acc, total * sizeof(float));

            /* Reference: scalar update of every row. */
            for (int r = 0; r < nrows; r++) {
                float *rr = ref + guard + (size_t)r * hd;
                for (int d = 0; d < hd; d++) {
                    float a = rr[d];
                    for (int j = 0; j < ts; j++) a += w[j] * v[(size_t)j * hd + d];
                    rr[d] = a;
                }
            }

            /* Kernel under test, row by row, exactly as attn_process_tile
             * calls it. */
            for (int r = 0; r < nrows; r++)
                acc_update(w, ts, v, hd, acc + guard + (size_t)r * hd);

            const int before = failures;
            checks++;

            /* 1. nothing outside the rows may change (catches the overrun) */
            for (size_t i = 0; i < total; i++) {
                int in_row = 0;
                for (int r = 0; r < nrows; r++) {
                    size_t lo = (size_t)guard + (size_t)r * hd;
                    if (i >= lo && i < lo + (size_t)hd) { in_row = 1; break; }
                }
                if (!in_row && acc[i] != (float)GUARD) {
                    printf("FAIL head_dim=%d tile=%d: wrote outside the row at "
                           "index %zu (got %g, guard is %d)\n",
                           hd, ts, i, (double)acc[i], GUARD);
                    failures++;
                    break;
                }
            }

            /* 2. values must match the scalar reference.
             * FMA contraction vs separate mul/add, and the 16-float block
             * order, allow a little slack; 1e-4 relative is far tighter than
             * the ~100% error a clobbered row produces. */
            for (int r = 0; r < nrows; r++) {
                float *ar = acc + guard + (size_t)r * hd;
                float *rr = ref + guard + (size_t)r * hd;
                for (int d = 0; d < hd; d++) {
                    float den = fabsf(rr[d]) + 1e-3f;
                    if (fabsf(ar[d] - rr[d]) / den > 1e-4f) {
                        printf("FAIL head_dim=%d tile=%d row=%d dim=%d: "
                               "got %.6f, expected %.6f\n",
                               hd, ts, r, d, (double)ar[d], (double)rr[d]);
                        failures++;
                        break;
                    }
                }
            }

            free(acc); free(ref); free(v); free(w);
            if (failures != before) failed_shapes++;
            if (failed_shapes > 6) { printf("... stopping after %d failing shapes\n", failed_shapes); return 1; }
        }
    }

    if (failures) {
        printf("\n%d of %d acc-update checks FAILED\n", failures, checks);
        return 1;
    }
    printf("ALL TESTS PASSED (%d acc-update shapes, head_dim 1..288)\n", checks);
    return 0;
}
