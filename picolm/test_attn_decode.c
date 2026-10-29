/* Differential + timing test: decode attention (attention_decode / attention_group,
 * blocked softmax + split-KV) vs attention_group_ref (original per-position path).
 *
 * usage: test_attn_decode [heads kv hd pos kvtype swa threads reps seed vtype]
 *   kvtype: 0=F16 1=Q8_0 2=Q4_0 3=TQ3 4=TQ4    pos = last position (ctx = pos+1)
 * env: PICOLM_TEST_QSCALE (peaky softmax), PICOLM_TEST_KRAMP, plus the runtime
 *      knobs PICOLM_ATTN_SPLIT / _SPLIT_MIN / _PF.
 * Prints MATCH/MISMATCH, then median ms per call and effective KV bandwidth. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "model.h"
#include "model_internal.h"
#include "quant.h"
#include "tensor.h"

void fp16_table_init(void);
double get_time_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static uint64_t rs = 0x123456789abcdefULL;
static float frand(void) {
    rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((rs >> 40) & 0xFFFF) / 65535.0f - 0.5f;
}
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static size_t head_bytes(int type, int hd) {
    switch (type) {
    case KV_CACHE_Q8_0: return (size_t)(hd / 32) * sizeof(block_q8_0);
    case KV_CACHE_Q4_0: return (size_t)(hd / 32) * sizeof(block_q4_0);
    case KV_CACHE_TQ3:  return (size_t)(hd / 32) * sizeof(block_tq3);
    case KV_CACHE_TQ4:  return (size_t)(hd / 32) * sizeof(block_tq4);
    default:            return (size_t)hd * 2;
    }
}
static void put_row(int type, const float *x, uint8_t *dst, int hd) {
    switch (type) {
    case KV_CACHE_Q8_0: quantize_row_q8_0(x, dst, hd); break;
    case KV_CACHE_Q4_0: quantize_row_q4_0(x, dst, hd); break;
    case KV_CACHE_TQ3:  quantize_row_tq3(x, dst, hd); break;
    case KV_CACHE_TQ4:  quantize_row_tq4(x, dst, hd); break;
    default: for (int i = 0; i < hd; i++) ((uint16_t *)dst)[i] = fp32_to_fp16(x[i]);
    }
}

int main(int argc, char **argv) {
    int nh   = argc > 1 ? atoi(argv[1]) : 32;
    int nkv  = argc > 2 ? atoi(argv[2]) : 8;
    int hd   = argc > 3 ? atoi(argv[3]) : 128;
    int pos  = argc > 4 ? atoi(argv[4]) : 4095;
    int kvt  = argc > 5 ? atoi(argv[5]) : 0;
    int swa  = argc > 6 ? atoi(argv[6]) : 0;
    int thr  = argc > 7 ? atoi(argv[7]) : 1;
    int reps = argc > 8 ? atoi(argv[8]) : 20;
    int seed = argc > 9 ? atoi(argv[9]) : 1;
    int vkvt = argc > 10 ? atoi(argv[10]) : kvt;   /* optional separate V cache type */
    tensor_set_n_threads(thr);
    fp16_table_init();
    rs ^= (uint64_t)seed * 0x9e3779b97f4a7c15ULL;

    int kvm = nh / nkv, total = pos + 1;
    size_t hb = head_bytes(kvt, hd), row = (size_t)nkv * hb;
    size_t hbv = head_bytes(vkvt, hd), rowv = (size_t)nkv * hbv;
    uint8_t *kc = malloc((size_t)total * row), *vc = malloc((size_t)total * rowv);
    const char *kr = getenv("PICOLM_TEST_KRAMP"); float kramp = kr ? (float)atof(kr) : 0.0f;
    float *tmp = malloc(sizeof(float) * hd);
    for (int p = 0; p < total; p++) {
        float ks = 1.0f + kramp * (float)p / (float)total;
        for (int h = 0; h < nkv; h++) {
            for (int i = 0; i < hd; i++) tmp[i] = frand() * ks;
            put_row(kvt, tmp, kc + (size_t)p * row + h * hb, hd);
            for (int i = 0; i < hd; i++) tmp[i] = frand();
            put_row(vkvt, tmp, vc + (size_t)p * rowv + h * hbv, hd);
        }
    }
    const char *qs = getenv("PICOLM_TEST_QSCALE"); float qscale = qs ? (float)atof(qs) : 1.0f;
    float *q = malloc(sizeof(float) * nh * hd);
    for (int i = 0; i < nh * hd; i++) q[i] = frand() * 2.0f * qscale;
    float *xr = calloc(nh * hd, sizeof(float)), *xn = calloc(nh * hd, sizeof(float));

    attn_group_ctx_t g; memset(&g, 0, sizeof g);
    g.kv_mul = kvm; g.head_dim = hd; g.pos = pos; g.kv_type_k = kvt; g.kv_type_v = vkvt;
    g.kv_row_size_k = row; g.kv_row_size_v = rowv; g.kv_head_stride_k = hb; g.kv_head_stride_v = hbv;
    g.kcache = kc; g.vcache = vc; g.q = q; g.n_kv_heads = nkv;
    g.attn_scale = 1.0f / sqrtf((float)hd); g.n_swa = swa;

    g.xb = xr; tensor_parallel_for(nkv, attention_group_ref, &g);
    g.xb = xn; attention_decode(&g);

    double md = 0, rms = 0; int at = 0;
    for (int i = 0; i < nh * hd; i++) {
        double d = fabs((double)xr[i] - xn[i]); if (!isfinite(d)) d = INFINITY;
        if (d > md) { md = d; at = i; }
        rms += (double)xr[i] * xr[i];
    }
    rms = sqrt(rms / (nh * hd));
    /* mixed abs/rel tolerance: new path uses a degree-6 exp2 (1.3 ulp) and a different
     * summation order; allow 2e-5 of the output rms + 1e-6 absolute. */
    int ok = md <= 2e-5 * rms + 1e-6;
    /* float64 oracle (F16 caches only): when extreme scores make fp32 rounding visible,
     * judge each implementation against the truth instead of against each other. */
    double e_ref = -1, e_new = -1;
    if (kvt == KV_CACHE_F16 && vkvt == KV_CACHE_F16) {
        double *o = malloc(sizeof(double) * hd), *sc = malloc(sizeof(double) * total);
        int t_start = swa > 0 ? pos - swa + 1 : 0; if (t_start < 0) t_start = 0;
        e_ref = e_new = 0;
        for (int h = 0; h < nh; h++) {
            int kh = h / kvm; double mx = -1e300;
            for (int t = t_start; t <= pos; t++) {
                const uint16_t *k = (const uint16_t *)(kc + (size_t)t * row + kh * hb);
                double a = 0; for (int d = 0; d < hd; d++) a += (double)q[h * hd + d] * fp16_to_fp32(k[d]);
                sc[t] = a * g.attn_scale; if (sc[t] > mx) mx = sc[t];
            }
            double L = 0; for (int d = 0; d < hd; d++) o[d] = 0;
            for (int t = t_start; t <= pos; t++) {
                double w = exp(sc[t] - mx); L += w;
                const uint16_t *v = (const uint16_t *)(vc + (size_t)t * rowv + kh * hbv);
                for (int d = 0; d < hd; d++) o[d] += w * fp16_to_fp32(v[d]);
            }
            for (int d = 0; d < hd; d++) {
                double tr = o[d] / L;
                double a = fabs(tr - xr[h * hd + d]), b = fabs(tr - xn[h * hd + d]);
                if (a > e_ref) e_ref = a; if (b > e_new) e_new = b;
            }
        }
        free(o); free(sc);
        if (!ok && e_new <= 1.5 * e_ref + 1e-7) ok = 1;   /* not worse than the reference vs truth */
    }
    printf("CONFIG heads=%d kv=%d hd=%d ctx=%d ktype=%d vtype=%d swa=%d threads=%d\n", nh, nkv, hd, total, kvt, vkvt, swa, thr);
    printf("RESULT max_abs_diff=%.3g (at %d) rms_ref=%.4g  %s\n", md, at, rms, ok ? "MATCH" : "MISMATCH");
    if (e_ref >= 0) printf("ORACLE err_vs_f64: ref=%.3g new=%.3g\n", e_ref, e_new);

    if (reps > 0) {
        size_t span = swa > 0 ? (size_t)(swa < total ? swa : total) : (size_t)total;
        double bytes = (double)span * nkv * (hb + hbv);
        for (int variant = 0; variant < 2; variant++) {
            double *t = malloc(sizeof(double) * reps);
            for (int r = -2; r < reps; r++) {
                double t0 = get_time_ms();
                g.xb = xn;
                if (variant == 0) tensor_parallel_for(nkv, attention_group_ref, &g);
                else attention_decode(&g);
                double dt = get_time_ms() - t0;
                if (r >= 0) t[r] = dt;
            }
            qsort(t, reps, sizeof(double), cmp_d);
            double med = t[reps / 2];
            printf("TIME %-4s median %8.3f ms/call  %7.2f ns/pos/kvhead  %6.2f GB/s KV\n",
                   variant == 0 ? "ref" : "new", med, med * 1e6 / ((double)span * nkv), bytes / (med * 1e-3) / 1e9);
            free(t);
        }
    }
    return ok ? 0 : 1;
}
