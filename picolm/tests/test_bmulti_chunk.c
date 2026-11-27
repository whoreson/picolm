/* test_bmulti_chunk: is a quant type's fn_gemm safe to call on a ROW SUB-RANGE?
 *
 * This is the contract tensor_matmul_batch_multi() depends on. It splits each
 * expert's weight matrix into BMULTI_ROWS-row chunks and hands each chunk to
 * that type's fn_gemm as
 *
 *     fn_gemm(nr, ncols, n, (char*)W + r0*row_bytes, qbuf, out + r0, bs=d, 0, 1)
 *
 * -- a sliced weight base, a sliced output base, and ith=0/nth=1. That is only
 * valid if the kernel computes exactly rows [r0, r0+nr) and writes nothing
 * else. A tile-shaped kernel that assumes nrows is a multiple of its tile, or
 * that writes to out[0..] instead of out[r0..], silently produces wrong values
 * or corrupts a neighbouring chunk, with no diagnostic at the call site.
 *
 * tensor_matmul_batch_multi_ok() is a TRUST LIST of quant types proven here.
 * Every quant type a user can put in a GGUF needs to be on it eventually; see
 * the TODO there.
 *
 * ---------------------------------------------------------------- pitfalls
 * Two mistakes make this test silently vacuous, and both were made while
 * writing it, so they are called out here:
 *
 *  1. fp16_table_init() must be called. quant.c reads every block scale
 *     through a 64K-entry lookup table that is NOT initialised automatically;
 *     without it all dequantised weights are 0 and everything "agrees".
 *
 *  2. The `a` argument is QUANTIZED ACTIVATIONS, not floats. The real caller
 *     passes bmulti_quant_task's Q8_K (or Q8_0) buffer. Passing raw floats
 *     makes fn_gemm reinterpret float bits as quant blocks: results are wrong
 *     by ~100x and LOOK like a real kernel bug. The activation format comes
 *     from qtype_act_format(), never assumed.
 *
 * ORACLE: expected values come from dequantising each weight row and taking a
 * plain float dot product against the SAME dequantised activations the kernel
 * sees, so this stays independent of which GEMM kernel the full-batch path
 * picks. A failure here means the chunked kernel is wrong, not that two
 * kernels tile differently.
 *
 * Build/run:  make check-bmulti-chunk
 * Env:        PICOLM_BMULTI_TEST_TYPES=iq6_k,iq3_xxs   (exact type name)
 *             PICOLM_BMULTI_TEST_VERBOSE=1             (also print passes)
 */
#include "quant.h"
#include "tensor.h"
#include "type_dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t rs = 0x12345678u;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float frand(void) { return (float)((int)(rnd() % 20001) - 10000) / 10000.0f; }

/* Small finite fp16 bit patterns forced into every block's scale slot(s). */
static const uint16_t k_scale_half  = 0x2C00;   /* 0.0625   */
static const uint16_t k_scale2_half = 0x2400;   /* 0.015625 */

/* Blocks per row and block byte size, derived from the type's own row size so
 * no second layout table is needed. Two block families exist: 256-value
 * superblocks (K-quants, IQ*) and 32-value legacy blocks (Q8_0/Q4_0/Q5_0/
 * IQ4_NL). Guessing wrong here (e.g. treating Q8_0's 64 blocks as 8) leaves
 * 56 blocks with random fp16 scales and the oracle itself returns 1e6. */
static size_t block_bytes_of(gguf_type_t t, int n)
{
    const size_t rb = gguf_type_row_size(t, n);
    if (rb == 0) return 0;
    static const int vals[2] = { 256, 32 };
    for (int i = 0; i < 2; i++) {
        const int nb = n / vals[i];
        if (nb > 0 && (n % vals[i]) == 0 && (rb % (size_t)nb) == 0) {
            const size_t bb = rb / (size_t)nb;
            if (bb >= 16) return bb;
        }
    }
    return rb;
}

/* Where the fp16 scale word(s) live inside one block.
 *
 * Most formats put `d` first, but three do not, and guessing wrong leaves the
 * REAL scale as random fp16 exponent bits (up to ~1e5):
 *   Q2_K: scales[16] qs[64] d dmin   -> d at 82, dmin at 82+2 (84-byte block)
 *   Q3_K: hmask[32] qs[64] scales[12] d -> d at 108 (110-byte block)
 *   Q6_K: ... scales[16] at 192, d at 208 (210-byte block)
 * Everything else (Q4_K/Q5_K/IQ*, Q8_0/Q4_0/Q5_0/IQ4_NL/Q1_0/Q2_0) starts
 * with d, and K-quant superblocks have dmin right after it. */
static void stomp_block_scales(uint8_t *blk, size_t bb, gguf_type_t t)
{
    switch (t) {
    case GGUF_TYPE_Q2_K:
        if (bb >= 84) { memcpy(blk + 82, &k_scale_half, 2); memcpy(blk + 80, &k_scale2_half, 2); }
        return;
    case GGUF_TYPE_Q3_K:
        if (bb >= 110) memcpy(blk + 108, &k_scale_half, 2);
        return;
    case GGUF_TYPE_Q6_K:
        if (bb >= 210) memcpy(blk + 208, &k_scale_half, 2);
        return;
    default:
        break;
    }
    /* Leading-d formats. */
    if (bb >= 4) {
        memcpy(blk, &k_scale_half, 2);
        memcpy(blk + 2, &k_scale2_half, 2);
    } else if (bb >= 2) {
        memcpy(blk, &k_scale_half, 2);
    }
}

/* Write a weights buffer for `t` with a sane numeric range, index bits random.
 *
 * Every block (not just the row!) needs its scale stomped. Stomping only the
 * first block of each row leaves the other blocks' `d` as random fp16 bits, and
 * a random fp16 exponent is routinely 1e4..1e5 -- enough that the oracle itself
 * returns 1e9 and every comparison is meaningless. (That is precisely how the
 * first version of this test managed to "fail" on IQ6_K and Q8_0, the two
 * simplest layouts.) data_is_sane() now checks for this before any comparison
 * and reports DATA-BAD instead of a bogus CHUNK-UNSAFE verdict. */
static void *make_weights(gguf_type_t t, int n, int d)
{
    const size_t rb = gguf_type_row_size(t, n);
    const size_t bb = block_bytes_of(t, n);
    if (rb == 0 || bb == 0) return NULL;
    uint8_t *W = (uint8_t *)malloc(rb * (size_t)d);
    if (!W) return NULL;
    for (int r = 0; r < d; r++) {
        uint8_t *row = W + (size_t)r * rb;
        for (size_t b = 0; b < rb; b++) row[b] = (uint8_t)(rnd() & 0xFF);
        for (size_t o = 0; o + bb <= rb; o += bb) stomp_block_scales(row + o, bb, t);
    }
    return W;
}

/* Quantize `ncols` activation rows into the format this weight type expects,
 * exactly as bmulti_quant_task does (one contiguous buffer, row stride = the
 * format's block size). Caller frees. */
static void *make_acts(gguf_type_t t, int n, int ncols, float *x)
{
    const act_fmt_t fmt = qtype_act_format(t);
    size_t stride;
    switch (fmt) {
        case ACT_FMT_Q8_0: stride = (size_t)(n / 32) * sizeof(block_q8_0);  break;
        case ACT_FMT_Q8_K: stride = (size_t)(n / 256) * sizeof(block_q8_K); break;
        default:           return NULL;   /* F32: no table GEMM anyway */
    }
    char *q = (char *)malloc(stride * (size_t)ncols);
    if (!q) return NULL;
    for (int c = 0; c < ncols; c++) {
        if (fmt == ACT_FMT_Q8_0) quantize_row_q8_0(x + (size_t)c * n, q + (size_t)c * stride, n);
        else                     quantize_row_q8_K(x + (size_t)c * n, q + (size_t)c * stride, n);
    }
    return q;
}

/* Oracle: dequantise the weights AND the quantized activations, then take a
 * plain float dot product. Independent of any GEMM tiling. */
static void oracle(const void *W, const char *qbuf, gguf_type_t t,
                   int n, int d, int ncols, float *out)
{
    const size_t rb = gguf_type_row_size(t, n);
    const act_fmt_t fmt = qtype_act_format(t);
    float *wrow = (float *)malloc((size_t)n * sizeof(float));
    float *arow = (float *)malloc((size_t)ncols * n * sizeof(float));

    const size_t qstride = (fmt == ACT_FMT_Q8_0) ? (size_t)(n / 32) * sizeof(block_q8_0)
                                                 : (size_t)(n / 256) * sizeof(block_q8_K);
    for (int c = 0; c < ncols; c++) {
        if (fmt == ACT_FMT_Q8_0) {
            dequantize_row_q8_0(qbuf + (size_t)c * qstride, arow + (size_t)c * n, n);
        } else {
            /* No public dequantize_row_q8_K(); inline it from the block struct
             * (block_q8_K = { float d; int8_t qs[256]; int16_t bsums[16]; }). */
            const block_q8_K *qk = (const block_q8_K *)(qbuf + (size_t)c * qstride);
            float *dst = arow + (size_t)c * n;
            const int nb = n / 256;
            for (int b = 0; b < nb; b++)
                for (int j = 0; j < 256; j++)
                    dst[b * 256 + j] = qk[b].d * (float)qk[b].qs[j];
        }
    }

    for (int r = 0; r < d; r++) {
        dequantize_row((const uint8_t *)W + (size_t)r * rb, wrow, n, t);
        for (int c = 0; c < ncols; c++) {
            const float *xc = arow + (size_t)c * n;
            double acc = 0.0;
            for (int k = 0; k < n; k++) acc += (double)wrow[k] * (double)xc[k];
            out[(size_t)c * d + r] = (float)acc;
        }
    }
    free(wrow);
    free(arow);
}

/* Sanity of the SYNTHETIC data, checked once per type before any comparison.
 *
 * A quant format whose scale/offset slots we fail to stomp produces weights
 * with fp16 exponent bits left random -- routinely 1e4..1e9 -- and then the
 * oracle overflows, every comparison fails, and it looks exactly like a
 * chunk-safety bug. That wastes an afternoon. So separate the two verdicts:
 * if the data is bad, report DATA-BAD and do not claim the type is unsafe. */
static int data_is_sane(gguf_type_t t, int n, int verbose)
{
    void *W = make_weights(t, n, 1);
    if (!W) return 0;
    float *row = (float *)malloc((size_t)n * sizeof(float));
    dequantize_row(W, row, n, t);
    double mx = 0;
    for (int i = 0; i < n; i++) {
        const double v = fabs((double)row[i]);
        if (v > mx) mx = v;
    }
    const int ok = (mx > 1e-6) && (mx < 1e3) && !isnan(mx);
    if (!ok || verbose)
        printf("  [data] %-8s n=%-5d dequant |max| = %.3e%s\n",
               " ", n, mx, ok ? "" : "   <-- generator cannot produce sane weights for this type");
    free(W); free(row);
    return ok;
}

static int run_case(gguf_type_t t, const char *name, int n, int ncols, int d, int chunk,
                    int verbose)
{
    const type_info_t *ti = typeinfo(t);
    if (!ti || !ti->fn_gemm) return 0;          /* no table GEMM: nothing to prove */

    void *W = make_weights(t, n, d);
    float *x = (float *)malloc((size_t)ncols * n * sizeof(float));
    float *exp = (float *)malloc((size_t)ncols * d * sizeof(float));
    float *got = (float *)calloc((size_t)ncols * d, sizeof(float));
    if (!W || !x || !exp || !got) { free(W); free(x); free(exp); free(got); return 1; }

    for (size_t i = 0; i < (size_t)ncols * n; i++) x[i] = frand();
    void *q = make_acts(t, n, ncols, x);
    if (!q) { free(W); free(x); free(exp); free(got); return 0; }

    oracle(W, (const char *)q, t, n, d, ncols, exp);

    /* Chunked: exactly bmulti_gemm_task's call shape, serial with a fixed
     * chunk so boundary cases are deterministic. */
    const size_t rb = gguf_type_row_size(t, n);
    for (int r0 = 0; r0 < d; r0 += chunk) {
        const int nr = (d - r0 < chunk) ? (d - r0) : chunk;
        const int r = ti->fn_gemm(nr, ncols, n, (const char *)W + (size_t)r0 * rb,
                                  q, got + r0, (size_t)d, 0, 1);
        if (r == 0) {
            printf("  %-8s n=%-5d ncols=%-3d d=%-5d chunk=%-4d  FAIL: fn_gemm returned 0 (nr=%d)\n",
                   name, n, ncols, d, chunk, nr);
            free(W); free(x); free(exp); free(got); free(q);
            return 1;
        }
    }

    double md = 0, scale = 0;
    for (size_t i = 0; i < (size_t)ncols * d; i++) {
        const double a = (double)exp[i], b = (double)got[i];
        if (isnan(a) != isnan(b)) { md = INFINITY; break; }
        const double df = fabs(a - b);
        if (df > md) md = df;
        if (fabs(a) > scale) scale = fabs(a);
    }
    /* Relative to the oracle's own magnitude. Same-kernel chunking should be
     * bit-identical for types that keep a fixed accumulation order; we allow
     * 2e-5 for types whose kernel changes tiling with nrows. Structural
     * corruption is O(1) relative and cannot hide under this. */
    const double rel = (scale > 1e-6) ? md / scale : md;
    const int fail = !(rel <= 2e-5);
    if (fail || verbose)
        printf("  %-8s n=%-5d ncols=%-3d d=%-5d chunk=%-4d  %s maxabs=%.3e rel=%.3e (|ref|max=%.2e)\n",
               name, n, ncols, d, chunk, fail ? "FAIL" : "ok  ", md, rel, scale);
    free(W); free(x); free(exp); free(got); free(q);
    return fail;
}

int main(void)
{
    /* NOT optional, but the call is also *verified* below rather than merely
     * trusted: if a future refactor removes the lazy init from quant.c, the
     * self-check fails loudly instead of the whole suite going vacuous. */
    fp16_table_init();
    {
        /* 0x2C00 is 0.0625 in fp16. Through the table it must come back as
         * 0.0625; before fp16_table_init() every entry is 0.0f, which is what
         * makes comparisons silently "pass" (see the header comment on
         * fp16_table_init() in quant.h). */
        const float got = fp16_to_fp32_lookup(0x2C00);
        if (!(got > 0.0624f && got < 0.0626f)) {
            printf("FATAL: fp16 lookup table is not initialised (fp16 0x2C00 -> %g, want 0.0625).\n"
                   "       Every dequantised weight would be 0.0 and this suite would prove nothing.\n", got);
            return 2;
        }
    }

    const char *only = getenv("PICOLM_BMULTI_TEST_TYPES");
    const int verbose = getenv("PICOLM_BMULTI_TEST_VERBOSE") != NULL;

    static const struct { const char *n; gguf_type_t t; } types[] = {
        { "iq6_k",   GGUF_TYPE_IQ6_K   },   /* the current trust-list entry */
        { "iq3_xxs", GGUF_TYPE_IQ3_XXS },   /* DeepSeek-V3 routed experts     */
        { "iq3_s",   GGUF_TYPE_IQ3_S   },   /* DeepSeek-V3 attention          */
        { "iq3_k",   GGUF_TYPE_IQ3_K   },
        { "q6_k",    GGUF_TYPE_Q6_K    },
        { "q4_k",    GGUF_TYPE_Q4_K    },
        { "q5_k",    GGUF_TYPE_Q5_K    },
        { "q3_k",    GGUF_TYPE_Q3_K    },
        { "q2_k",    GGUF_TYPE_Q2_K    },
        { "iq4_xs",  GGUF_TYPE_IQ4_XS  },
        { "iq2_k",   GGUF_TYPE_IQ2_K   },
        { "iq4_nl",  GGUF_TYPE_IQ4_NL  },
        { "q8_0",    GGUF_TYPE_Q8_0    },
        { "q4_0",    GGUF_TYPE_Q4_0    },
        { NULL, 0 }
    };

    /* Shapes mirror the MoE FFN: gate/up are D->FF (K=7168 d=2048), down is
     * FF->D (K=2048 d=7168). ncols mirrors real bucket sizes: a prefill of ~86
     * tokens over 256 experts gives mostly 1-8, occasionally larger. */
    static const int kcols[]   = { 1, 2, 3, 5, 8, 17 };
    static const int kchunks[] = { 64, 128, 8, 24, 100, 17 };
    static const int kns[]     = { 2048, 7168 };
    static const int kds[]     = { 2048, 7168 };

    int fails = 0, cases = 0, skipped = 0, data_bad = 0;
    for (int ti = 0; types[ti].n; ti++) {
        if (only && !strstr(only, types[ti].n)) continue;
        printf("== %s ==\n", types[ti].n);
        const type_info_t *ti_ = typeinfo(types[ti].t);
        if (!ti_ || !ti_->fn_gemm) {
            printf("   -> no table GEMM for this type; not applicable\n");
            continue;
        }
        if (!data_is_sane(types[ti].t, 2048, verbose)) {
            printf("   -> DATA-BAD: synthetic weights unsupported for this type.\n"
                   "      NOT a chunk-safety verdict. Fix make_weights() first.\n");
            data_bad++;
            continue;
        }
        int t_fails = 0, t_cases = 0;
        for (unsigned a = 0; a < sizeof kns / sizeof kns[0]; a++)
        for (unsigned b = 0; b < sizeof kds / sizeof kds[0]; b++)
        for (unsigned c = 0; c < sizeof kcols / sizeof kcols[0]; c++)
        for (unsigned e = 0; e < sizeof kchunks / sizeof kchunks[0]; e++) {
            if (gguf_type_row_size(types[ti].t, kns[a]) == 0) continue;
            int f = run_case(types[ti].t, types[ti].n, kns[a], kcols[c], kds[b],
                             kchunks[e], verbose);
            t_fails += f; t_cases++; cases++;
        }
        if (t_cases == 0) skipped++;
        fails += t_fails;
        printf("   -> %s (%d/%d failed)\n", t_fails ? "CHUNK-UNSAFE" : "chunk-safe",
               t_fails, t_cases);
    }
    printf("\n%d cases, %d failures, %d types skipped as DATA-BAD\n",
           cases, fails, data_bad);
    return fails ? 1 : 0;
}
