/* Regression: the F16 shadow of the MLA latent cache must be populated by the
 * snapshot restore (and the batched-prefill store), not only by the decode
 * store. Decode attention reads only the shadow; a row that was written to the
 * F32 master but not mirrored is an all-zero row, i.e. invisible context. */
#include "model_glm5.c"
#include <time.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }

int main(void) {
    const int KVL = 512, MAXSEQ = 64, NPOS = 37;
    model_t *m = calloc(1, sizeof *m);
    model_config_t *c = &m->config;
    c->is_glm5 = 1; c->n_layers = 2; c->max_seq_len = MAXSEQ; c->glm5_kv_lora = KVL;
    c->glm5_n_mla = 2;
    for (int l = 0; l < 2; l++) { c->glm5_layer_is_mla[l] = 1; c->glm5_layer_mla_ord[l] = (int16_t)l; }
    run_state_t *s = &m->state;
    size_t n = (size_t)2 * MAXSEQ * KVL;
    s->glm5.mla_kv   = calloc(n, sizeof(float));
    s->glm5.mla_kv16 = calloc(n, sizeof(uint16_t));

    size_t need = model_glm5_snapshot_size(m, NPOS);
    uint8_t *buf = calloc(1, need);
    float *rows = (float *)buf;                       /* MLA latents come first in the blob */
    unsigned long long r = 88172645463325252ULL;
    for (size_t i = 0; i < (size_t)2 * NPOS * KVL; i++) {
        r ^= r << 13; r ^= r >> 7; r ^= r << 17;
        rows[i] = ((float)((double)(r >> 11) / 9007199254740992.0) * 2.f - 1.f) * 3.f;
    }
    if (model_glm5_state_restore(m, buf, need, NPOS) == 0) { puts("restore failed"); return 2; }

    int fails = 0, zero_rows = 0;
    for (int ord = 0; ord < 2; ord++) {
        for (int pos = 0; pos < MAXSEQ; pos++) {
            const size_t base = ((size_t)ord * MAXSEQ + pos) * KVL;
            int all_zero = 1;
            for (int d = 0; d < KVL; d++) {
                uint16_t want = pos < NPOS ? fp32_to_fp16(s->glm5.mla_kv[base + d]) : 0;
                if (s->glm5.mla_kv16[base + d] != want) fails++;
                if (s->glm5.mla_kv16[base + d]) all_zero = 0;
            }
            if (pos < NPOS && all_zero) zero_rows++;
        }
    }
    printf("restore: %d shadow mismatches, %d restored rows left all-zero (want 0, 0)\n", fails, zero_rows);

    /* batched-prefill store helper: rows [10, 15) of ordinal 1 rewritten in F32 */
    for (int pos = 10; pos < 15; pos++)
        for (int d = 0; d < KVL; d++) s->glm5.mla_kv[((size_t)1*MAXSEQ + pos)*KVL + d] = 1.5f + d * 1e-3f;
    glm5_mla_shadow_sync(s, MAXSEQ, KVL, 1, 10, 5);
    int pf_bad = 0;
    for (int pos = 10; pos < 15; pos++)
        for (int d = 0; d < KVL; d++) {
            size_t i = ((size_t)1*MAXSEQ + pos)*KVL + d;
            if (s->glm5.mla_kv16[i] != fp32_to_fp16(s->glm5.mla_kv[i])) pf_bad++;
        }
    printf("prefill-store sync: %d mismatches (want 0)\n", pf_bad);
    fails += pf_bad + zero_rows;
    puts(fails ? "FAIL" : "PASS");
    return fails != 0;
}
