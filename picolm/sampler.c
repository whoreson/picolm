#include "sampler.h"
#include "tensor.h"  /* for softmax */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- xorshift64 RNG ---- */

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float rand_float(uint64_t *state) {
    /* Generate a float in [0, 1) */
    return (float)(xorshift64(state) >> 11) / (float)(1ULL << 53);
}

/* ---- Comparison for sorting by probability (descending) ---- */

typedef struct { float prob; int index; } prob_index_t;

static int cmp_prob_desc(const void *a, const void *b) {
    float pa = ((const prob_index_t *)a)->prob;
    float pb = ((const prob_index_t *)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

/* ---- Public API ---- */

void sampler_init(sampler_t *s, float temperature, float top_p, uint64_t seed) {
    s->temperature = temperature;
    s->top_p = top_p;
    s->rng_state = seed ? seed : 42;
}

int sampler_sample(sampler_t *s, float *logits, int vocab_size) {
    /* Greedy (temperature 0) */
    if (s->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    /* Apply temperature */
    float inv_temp = 1.0f / s->temperature;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }

    /* Softmax */
    softmax(logits, vocab_size);

    /* If top_p >= 1.0, still use sorted sampling for consistency */

    /* Top-p (nucleus) sampling */
    /* Sort indices by probability descending */
    prob_index_t *sorted = (prob_index_t *)malloc((size_t)vocab_size * sizeof(prob_index_t));
    for (int i = 0; i < vocab_size; i++) {
        sorted[i].prob  = logits[i];
        sorted[i].index = i;
    }
    qsort(sorted, (size_t)vocab_size, sizeof(prob_index_t), cmp_prob_desc);

    /* Find cutoff where cumulative probability exceeds top_p */
    float cum = 0.0f;
    int cutoff = 0;
    for (int i = 0; i < vocab_size; i++) {
        cum += sorted[i].prob;
        cutoff = i + 1;
        if (cum >= s->top_p) break;
    }

    /* Sample from truncated distribution */
    float r = rand_float(&s->rng_state) * cum;
    float acc = 0.0f;
    int result = sorted[0].index;
    for (int i = 0; i < cutoff; i++) {
        acc += sorted[i].prob;
        if (acc > r) {
            result = sorted[i].index;
            break;
        }
    }

    free(sorted);
    return result;
}
