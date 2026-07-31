#include "sampler.h"
#include "tensor.h"  /* for softmax */
#include <math.h>
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

void sampler_init(sampler_t *s, float temperature, float top_p, int top_k,
                  float min_p, uint64_t seed) {
    s->temperature = temperature;
    s->top_p       = top_p;
    s->top_k       = top_k;
    s->min_p       = min_p;
    s->rng_state   = seed ? seed : 42;
}

/* llama.cpp order: penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature
   PicoLM implements: top_k -> top_p -> min_p -> temperature -> sample */
int sampler_sample(sampler_t *s, float *logits, int vocab_size) {
    /* Greedy (temperature 0) */
    if (s->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    /* 1. Softmax on raw logits */
    softmax(logits, vocab_size);

    /* 2. Sort by probability descending */
    prob_index_t *sorted = (prob_index_t *)malloc((size_t)vocab_size * sizeof(prob_index_t));
    for (int i = 0; i < vocab_size; i++) {
        sorted[i].prob  = logits[i];
        sorted[i].index = i;
    }
    qsort(sorted, (size_t)vocab_size, sizeof(prob_index_t), cmp_prob_desc);

    /* 3. Apply top-k: restrict to k most probable */
    int k_limit = vocab_size;
    if (s->top_k > 0 && s->top_k < vocab_size) {
        k_limit = s->top_k;
    }

    /* 4. Apply top-p: find cutoff where cumulative >= top_p (within top-k) */
    float cum = 0.0f;
    int cutoff = 0;
    for (int i = 0; i < k_limit; i++) {
        cum += sorted[i].prob;
        cutoff = i + 1;
        if (cum >= s->top_p) break;
    }

    /* 5. Apply min_p: remove tokens with prob < min_p * max_prob */
    if (s->min_p > 0.0f && cutoff > 1) {
        float min_threshold = s->min_p * sorted[0].prob;
        int new_cutoff = 0;
        for (int i = 0; i < cutoff; i++) {
            if (sorted[i].prob >= min_threshold) {
                new_cutoff = i + 1;
            }
        }
        cutoff = new_cutoff;
        if (cutoff == 0) cutoff = 1; /* keep at least top token */
    }

    /* 6. Apply temperature: scale to p^(1/T), renormalize */
    float inv_temp = 1.0f / s->temperature;
    float sum_scaled = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        sorted[i].prob = powf(sorted[i].prob, inv_temp);
        sum_scaled += sorted[i].prob;
    }
    if (sum_scaled > 0.0f) {
        float inv_sum = 1.0f / sum_scaled;
        for (int i = 0; i < cutoff; i++) {
            sorted[i].prob *= inv_sum;
        }
    }

    /* 7. Sample */
    float r = rand_float(&s->rng_state);
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
