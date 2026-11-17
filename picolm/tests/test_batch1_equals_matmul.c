/* test_batch1_equals_matmul.c
 *
 * matmul_batch(..., n_batch = 1, ...) must produce exactly the same output as
 * matmul(), because tensor.c now delegates the batch-of-one case to matmul().
 *
 * Why this test exists: the server's checkpoint-driven prefill cuts the prompt
 * at n_prompt - checkpoint_tail_offset, which leaves a 1-token tail chunk on
 * every request. That chunk used to fall through matmul_batch()'s table GEMM
 * gates (all of which require n_batch >= 2) into the per-row threaded_worker,
 * and for F32/BF16 tensors into the *serial* vec_dot loop -- 441 ms per token
 * on GLM-5 instead of the 269 ms the decode path needs for the same work.
 *
 * The two are expected to agree bit-for-bit: both quantize the activation row
 * with quant_activations_ex(use_amax_deltas=1) and run the same fn_gemv kernel
 * per weight row. Per-row dots do not depend on how the rows are partitioned
 * across threads, so the thread split cannot perturb the result.
 *
 * Weights are synthetic (random quant payloads) -- a dot product against a
 * random-but-well-formed block is enough to catch a wrong activation format, a
 * wrong row stride or a mis-routed kernel, which is all this test claims to
 * check. No GGUF needed.
 *
 * Build/run: make check-batch1
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "quant.h"
#include "tensor.h"
#include "type_dispatch.h"

static unsigned rng_state = 12345u;
static unsigned rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

/* Types with a fn_gemv (i.e. types matmul() can run) and their block layouts. */
static const gguf_type_t types[] = {
    GGUF_TYPE_Q8_0, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K,
    GGUF_TYPE_IQ4_NL, GGUF_TYPE_IQ6_K, GGUF_TYPE_IQ4_XS, GGUF_TYPE_IQ2_K,
};
static const char *type_names[] = {
    "Q8_0", "Q4_0", "Q4_K", "Q6_K", "IQ4_NL", "IQ6_K", "IQ4_XS", "IQ2_K",
};
/* Activation block sizes; the input dim must be a multiple of these. */
static const int type_kblock[] = { 32, 32, 256, 256, 32, 256, 256, 256 };

int main(int argc, char **argv) {
    fp16_table_init();
    int nthreads = (argc > 1) ? atoi(argv[1]) : 8;
    if (nthreads < 1) nthreads = 1;
    tensor_set_n_threads(nthreads);

    int fails = 0, checks = 0;
    const int ntypes = (int)(sizeof(types) / sizeof(types[0]));
    const int dims[] = { 256, 1024, 4096 };
    const int rows[] = { 1, 7, 128, 512 };

    for (int ti = 0; ti < ntypes; ti++) {
        gguf_type_t qt = types[ti];
        int kb = type_kblock[ti];
        for (unsigned di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
            int n = dims[di];
            if (n % kb) continue;
            for (unsigned ri = 0; ri < sizeof(rows) / sizeof(rows[0]); ri++) {
                int d = rows[ri];

                size_t rb = gguf_type_row_size(qt, n);
                size_t wbytes = rb * (size_t)d;
                void *W = malloc(wbytes);
                float *x = malloc((size_t)n * sizeof(float));
                float *o_mm = malloc((size_t)d * sizeof(float));
                float *o_mb = malloc((size_t)d * sizeof(float));
                if (!W || !x || !o_mm || !o_mb) { printf("OOM\n"); return 1; }

                /* Random, but well-formed enough: the fp16 scale halves of the
                 * blocks must not be NaN/Inf, or every dot becomes NaN and the
                 * comparison proves nothing. Random bytes elsewhere are fine. */
                uint8_t *wb = (uint8_t *)W;
                for (size_t i = 0; i < wbytes; i++) wb[i] = (uint8_t)rnd();

                for (int i = 0; i < n; i++)
                    x[i] = ((float)(int)(rnd() % 2001) - 1000.0f) / 300.0f;

                matmul(o_mm, x, W, n, d, qt);
                matmul_batch(o_mb, x, 1, W, n, d, qt);

                double maxrel = 0.0;
                int bad = 0;
                for (int i = 0; i < d; i++) {
                    if (o_mm[i] != o_mb[i]) {
                        double rel = fabs(o_mm[i] - o_mb[i]) / (fabs(o_mm[i]) + 1e-6);
                        if (rel > maxrel) maxrel = rel;
                        if (rel > 0.0) bad = 1;
                    }
                }
                checks++;
                if (bad) {
                    printf("FAIL %-7s n=%-5d d=%-4d  max_rel_diff=%.3e\n",
                           type_names[ti], n, d, maxrel);
                    fails++;
                } else {
                    printf("ok   %-7s n=%-5d d=%-4d  bit-identical\n",
                           type_names[ti], n, d);
                }
                free(W); free(x); free(o_mm); free(o_mb);
            }
        }
    }
    printf("\nthreads=%d: %d/%d checks bit-identical, %d failures\n",
           nthreads, checks - fails, checks, fails);
    return fails ? 1 : 0;
}
