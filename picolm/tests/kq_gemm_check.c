/* Validate every ARM K-quant x Q8_K GEMM against the scalar vec_dot.
 *
 * For each type and thread count, compare the table GEMM's output element by
 * element against vec_dot_*_q8_K(). The accumulation is integer in all kernels,
 * so agreement is to float-rounding only; anything larger is a real bug.
 *
 *   cc -O3 -I. -o kq_gemm_check tests/kq_gemm_check.c <objects> -lm -fopenmp
 *   ./kq_gemm_check
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include "quant.h"
#include "sgemm.h"
#include "type_dispatch.h"
#include "tensor.h"

void fp16_table_init(void);

static unsigned rs = 20251108;
static unsigned rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

typedef struct {
    const char *name;
    gguf_type_t gt;
    size_t bsz, d_off, dmin_off;
    float (*dot)(const void *, const void *, int);
} kt_t;

static const kt_t KT[] = {
    {"q2_K", GGUF_TYPE_Q2_K, sizeof(block_q2_K), offsetof(block_q2_K, d), offsetof(block_q2_K, dmin), vec_dot_q2_K_q8_K},
    {"q3_K", GGUF_TYPE_Q3_K, sizeof(block_q3_K), offsetof(block_q3_K, d), (size_t)-1,                  vec_dot_q3_K_q8_K},
    {"q4_K", GGUF_TYPE_Q4_K, sizeof(block_q4_K), offsetof(block_q4_K, d), offsetof(block_q4_K, dmin), vec_dot_q4_K_q8_K},
    {"q5_K", GGUF_TYPE_Q5_K, sizeof(block_q5_K), offsetof(block_q5_K, d), offsetof(block_q5_K, dm),   vec_dot_q5_K_q8_K},
    {"q6_K", GGUF_TYPE_Q6_K, sizeof(block_q6_K), offsetof(block_q6_K, d), (size_t)-1,                  vec_dot_q6_K_q8_K},
};

static int NTH = 4;

typedef struct { int m, n, k; const void *w, *a; float *c; size_t bs; type_gemm_fn f; } gctx_t;

/* IQ4_NL goes through picolm_sgemm_d (Q8_0 activations + float deltas), not the
 * Q8_K table ABI, so it gets its own check. */
#include "sgemm.h"
typedef struct { int m, n, k, nb; const void *w, *a; const float *ad; float *c; size_t bs; } ictx_t;
static void itask(int idx, void *vp) {
    ictx_t *g = (ictx_t *)vp;
    picolm_sgemm_d(g->m, g->n, g->nb, g->w, g->nb, (const block_q8_0 *)g->a, g->nb,
                   g->ad, g->nb, g->c, (int)g->bs, (int)GGUF_TYPE_IQ4_NL, idx, NTH);
}
static void gtask(int idx, void *vp) {
    gctx_t *g = (gctx_t *)vp;
    g->f(g->m, g->n, g->k, g->w, g->a, g->c, g->bs, idx, NTH);
}

int main(void) {
    fp16_table_init();
    setenv("PICOLM_R8_MIN_BLOCKS", "0", 0);

    const int ms[] = {8, 24, 72, 136};
    const int ns[] = {4, 7, 16, 17, 44};
    const int ks[] = {256, 512, 4096};
    const int nts[] = {1, 3, 4};

    int bad = 0, cnt = 0;
    for (unsigned t = 0; t < sizeof KT / sizeof *KT; t++) {
        const kt_t *q = &KT[t];
        type_gemm_fn f = typeinfo(q->gt)->fn_gemm;
        if (!f) { printf("%-5s no fn_gemm, skipped\n", q->name); continue; }
        double worst = 0;
        for (unsigned a = 0; a < sizeof ms / sizeof *ms; a++)
        for (unsigned b = 0; b < sizeof ns / sizeof *ns; b++)
        for (unsigned c = 0; c < sizeof ks / sizeof *ks; c++)
        for (unsigned e = 0; e < sizeof nts / sizeof *nts; e++) {
            const int m = ms[a], n = ns[b], k = ks[c];
            NTH = nts[e];
            const int nb = k / 256;
            const size_t wrow = (size_t)nb * q->bsz;

            unsigned char *W = malloc((size_t)m * wrow);
            for (size_t i = 0; i < (size_t)m * wrow; i++) W[i] = (unsigned char)rnd();
            for (size_t i = 0; i < (size_t)m * nb; i++) {
                unsigned char *blk = W + i * q->bsz;
                uint16_t d = fp32_to_fp16(0.001f + (rnd() & 255) * 1e-5f);
                memcpy(blk + q->d_off, &d, 2);
                if (q->dmin_off != (size_t)-1) {
                    uint16_t dm = fp32_to_fp16(0.0005f + (rnd() & 255) * 1e-5f);
                    memcpy(blk + q->dmin_off, &dm, 2);
                }
            }
            block_q8_K *B = malloc((size_t)n * nb * sizeof(block_q8_K));
            float *tmp = malloc(sizeof(float) * k);
            for (int col = 0; col < n; col++) {
                for (int i = 0; i < k; i++) tmp[i] = (rnd() & 0xFFFF) / 32768.0f - 1.0f;
                quantize_row_q8_K(tmp, B + (size_t)col * nb, k);
            }
            float *C = malloc(sizeof(float) * (size_t)n * m);
            for (size_t i = 0; i < (size_t)n * m; i++) C[i] = 1e30f;   /* poison */
            gctx_t g = {m, n, k, W, B, C, (size_t)m, f};
            tensor_parallel_for(NTH, gtask, &g);

            double md = 0, rm = 0;
            int this_bad = 0;
            for (int col = 0; col < n; col++)
                for (int i = 0; i < m; i++) {
                    double ref = q->dot(W + (size_t)i * wrow, B + (size_t)col * nb, k);
                    double got = C[(size_t)col * m + i];
                    double dd = fabs(ref - got);
                    if (!(dd == dd) || got > 1e29f) this_bad = 1;    /* unwritten */
                    if (dd > md) md = dd;
                    rm += ref * ref;
                }
            rm = sqrt(rm / ((double)m * n) + 1e-30);
            if (md > 1e-3 * rm + 1e-3) this_bad = 1;
            if (md > worst) worst = md;
            if (this_bad) {
                printf("  FAIL %-5s m=%d n=%d k=%d nth=%d maxdiff=%.3g rms=%.3g\n",
                       q->name, m, n, k, NTH, md, rm);
                bad = 1;
            }
            cnt++;
            free(W); free(B); free(tmp); free(C);
        }
        printf("%-5s %s  (worst abs diff over all shapes: %.3g)\n",
               q->name, bad ? "see failures above" : "OK", worst);
    }
    /* ---- IQ4_NL x Q8_0 through the real dispatch ---- */
    {
        const int ms2[] = {12, 24, 44, 72, 136};
        const int ns2[] = {4, 7, 22, 44};
        const int ks2[] = {32, 4096};
        const int nts2[] = {1, 3, 4};
        double worst4 = 0;
        int bad4 = 0;
        for (unsigned a = 0; a < 5; a++)
        for (unsigned b = 0; b < 4; b++)
        for (unsigned c = 0; c < 2; c++)
        for (unsigned e = 0; e < 3; e++) {
            const int m = ms2[a], n = ns2[b], k = ks2[c], nb = k / 32;
            NTH = nts2[e];
            block_iq4_nl *W = malloc(sizeof(block_iq4_nl) * (size_t)m * nb);
            for (size_t i = 0; i < (size_t)m * nb; i++) {
                W[i].d = fp32_to_fp16(0.01f + (rnd() & 63) * 1e-4f);
                for (int j = 0; j < 16; j++) W[i].qs[j] = (uint8_t)rnd();
            }
            block_q8_0 *B = malloc(sizeof(block_q8_0) * (size_t)n * nb);
            float *Bd = malloc(sizeof(float) * (size_t)n * nb);
            float *tmp = malloc(sizeof(float) * k);
            for (int col = 0; col < n; col++) {
                for (int i = 0; i < k; i++) tmp[i] = (rnd() & 0xFFFF) / 32768.0f - 1.0f;
                quantize_row_q8_0(tmp, B + (size_t)col * nb, k);
                for (int l = 0; l < nb; l++)
                    Bd[(size_t)col * nb + l] = fp16_to_fp32(B[(size_t)col * nb + l].d);
            }
            float *C = malloc(sizeof(float) * (size_t)n * m);
            for (size_t i = 0; i < (size_t)n * m; i++) C[i] = 1e30f;
            ictx_t g = {m, n, k, nb, W, B, Bd, C, (size_t)m};
            tensor_parallel_for(NTH, itask, &g);
            double md = 0, rm = 0;
            int this_bad = 0;
            for (int col = 0; col < n; col++)
                for (int i = 0; i < m; i++) {
                    double ref = vec_dot_iq4_nl_q8_0(W + (size_t)i * nb, B + (size_t)col * nb, k);
                    double got = C[(size_t)col * m + i];
                    double dd = fabs(ref - got);
                    if (!(dd == dd) || got > 1e29f) this_bad = 1;
                    if (dd > md) md = dd;
                    rm += ref * ref;
                }
            rm = sqrt(rm / ((double)m * n) + 1e-30);
            if (md > 1e-3 * rm + 1e-3) this_bad = 1;
            if (md > worst4) worst4 = md;
            if (this_bad) {
                printf("  FAIL iq4_nl m=%d n=%d k=%d nth=%d maxdiff=%.3g rms=%.3g\n",
                       m, n, k, NTH, md, rm);
                bad4 = 1;
            }
            cnt++;
            free(W); free(B); free(Bd); free(tmp); free(C);
        }
        printf("%-5s %s  (worst abs diff over all shapes: %.3g)\n",
               "iq4_nl", bad4 ? "see failures above" : "OK", worst4);
        bad |= bad4;
    }

    printf("kq_gemm_check: %d shape/thread combinations, %s\n", cnt, bad ? "FAIL" : "all OK");
    return bad;
}
