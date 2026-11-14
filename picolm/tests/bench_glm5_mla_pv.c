/* Is the MLA PV accumulate bound by the accumulator (L1 load/FMA/store) or by
 * reading the latent cache from L3?  Two kernels, bit-identical to each other,
 * timed with the cache footprint growing from L2-resident to 14 MB (7k pos):
 *   current block : ctx accumulators live in memory (load+FMA+store per vector)
 *   reg-tiled     : a 128-float column tile of both heads' accumulators stays in
 *                   zmm registers across the whole position loop
 * If reg-tiled wins big while the cache is L2-resident but the two converge
 * at 14 MB with GB/s flat near the per-core L3 rate, the pass is L3-bound and
 * only fewer cache bytes (F16 cache, fewer passes) can help.
 * Build like the other glm5 tests (needs -DPICOLM_AVX512), no Makefile target. */
#include "model_glm5.c"
#include <time.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }
static unsigned long long rs = 88172645463325252ULL;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (float)((double)(rs >> 11) / 9007199254740992.0) * 2.f - 1.f; }

/* Register-tiled PV accumulate for hn==2 heads: accumulators for a TILE of
 * columns stay in zmm registers across the whole position loop. Same fused
 * op, same ascending order, same p<1e-10 skip -> bit-identical. */
#define TILE 8   /* zmm per head per tile (128 floats) */
static void pv_tiled2(float *ctx0, const float *sc0, const float *inv, int n, int hn,
                      const float *cache, const int *sel, int dense, int KVL)
{
    (void)hn;
    const float *s0 = sc0, *s1 = sc0 + n;
    for (int c0 = 0; c0 < KVL; c0 += TILE*16) {
        __m512 a0[TILE], a1[TILE];
        #pragma GCC unroll 8
        for (int j = 0; j < TILE; j++) { a0[j] = _mm512_setzero_ps(); a1[j] = _mm512_setzero_ps(); }
        for (int i = 0; i < n; i++) {
            const int t = dense ? i : sel[i];
            const float *row = cache + (size_t)t * KVL + c0;
            const float p0 = s0[i] * inv[0], p1 = s1[i] * inv[1];
            const int u0 = !(p0 < 1e-10f), u1 = !(p1 < 1e-10f);
            if (u0) { __m512 v = _mm512_set1_ps(p0);
                #pragma GCC unroll 8
                for (int j = 0; j < TILE; j++) a0[j] = _mm512_fmadd_ps(v, _mm512_loadu_ps(row + 16*j), a0[j]); }
            if (u1) { __m512 v = _mm512_set1_ps(p1);
                #pragma GCC unroll 8
                for (int j = 0; j < TILE; j++) a1[j] = _mm512_fmadd_ps(v, _mm512_loadu_ps(row + 16*j), a1[j]); }
        }
        #pragma GCC unroll 8
        for (int j = 0; j < TILE; j++) {
            _mm512_storeu_ps(ctx0 + c0 + 16*j, a0[j]);
            _mm512_storeu_ps(ctx0 + KVL + c0 + 16*j, a1[j]);
        }
    }
}

int main(void) {
    const int KVL = 512, hn = 2;
    int NS[] = {128, 512, 2048, 7111};
    for (int ni = 0; ni < 4; ni++) {
        const int n = NS[ni];
        /* many layers' worth of distinct caches so each rep really comes from L3, not L2 */
        const int NL = (n * KVL * 4 < (1<<20)) ? 1 : 12;
        float **cache = malloc(NL * sizeof(float*));
        for (int l = 0; l < NL; l++) { cache[l] = malloc((size_t)n*KVL*4);
            for (size_t i = 0; i < (size_t)n*KVL; i++) cache[l][i] = rnd(); }
        float *s = malloc((size_t)hn*n*4), *sA = malloc((size_t)hn*n*4);
        float cA[2*512], cB[2*512];
        for (size_t i = 0; i < (size_t)hn*n; i++) s[i] = rnd() * 12.f;
        const int REP = (ni < 2) ? 400 : 60;
        double tb = 0, tt = 0; int bad = 0;
        for (int r = 0; r < REP; r++) {
            const float *C = cache[r % NL];
            memcpy(sA, s, (size_t)hn*n*4);
            double t0 = get_time_ms();
            glm5_mla_reduce_block(cA, sA, n, hn, C, NULL, 1, KVL);
            double t1 = get_time_ms();
            float inv[2];
            tb += t1 - t0;
            /* tiled: needs normalised exps + inv; recompute them the same way */
            memcpy(sA, s, (size_t)hn*n*4);
            for (int k = 0; k < hn; k++) {
                float *sc = sA + (size_t)k*n; float mx = -INFINITY;
                for (int i = 0; i < n; i++) if (sc[i] > mx) mx = sc[i];
                float sum = 0.f;
                for (int i = 0; i < n; i++) { float e = expf(sc[i]-mx); sc[i] = e; sum += e; }
                inv[k] = 1.0f / sum;
            }
            double t2 = get_time_ms();
            pv_tiled2(cB, sA, inv, n, hn, C, NULL, 1, KVL);
            double t3 = get_time_ms();
            tt += t3 - t2;
            if (memcmp(cA, cB, sizeof cA)) bad = 1;
        }
        /* subtract softmax cost from the tiled number for an apples-to-apples PV compare */
        double tsm = 0; { memcpy(sA, s, (size_t)hn*n*4); double a = get_time_ms();
            for (int r = 0; r < REP; r++) for (int k = 0; k < hn; k++) { float *sc = sA + (size_t)k*n; float mx=-INFINITY;
                for (int i = 0; i < n; i++) if (sc[i] > mx) mx = sc[i]; float sum=0; for (int i=0;i<n;i++){ sum += expf(sc[i]-mx);} (void)sum; }
            tsm = get_time_ms() - a; }
        const double mb = (double)n*KVL*4/1048576.0;
        printf("n=%-5d cache %5.1f MB (x%-2d layers)  current block %.3f ms  reg-tiled %.3f ms  "
               "-> %.1f GB/s vs %.1f GB/s of cache read  %s\n",
               n, mb, NL, tb/REP, tt/REP, mb/1024.0/(tb/REP/1e3), mb/1024.0/(tt/REP/1e3),
               bad ? "MISMATCH" : "bit-identical");
    }
    return 0;
}
