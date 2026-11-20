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

/* Cap on how many entries a top-k DISABLED sample (top_k <= 0, i.e. llama.cpp's
 * "top-k off") selects. Verified against the retained probability mass below,
 * so this bounds the per-token cost without changing what gets sampled. */
#ifndef SAMPLER_TOPK_UNBOUNDED
#define SAMPLER_TOPK_UNBOUNDED 1024
#endif

/* Above this k the sorted-insertion selection loses to a plain qsort (the
 * in-place memmove grows with k), so fall back to the sort. */
#ifndef SAMPLER_SELECT_MAX_K
#define SAMPLER_SELECT_MAX_K 8192
#endif

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

    /* 2. Top-k (if requested) by partial selection, else a full sort.
     *
     * This used to qsort the whole vocabulary every token: 154880 elements for
     * GLM-5, i.e. a malloc plus ~2.6M comparisons per token, and profiler showed
     * cmp_prob_desc + msort_with_tmp at 8.5% of the main thread during decode
     * (5.74% + 2.79%). Only the first `k_limit` entries are ever read, so a
     * partial selection is enough -- and it also lets the buffer be sized to
     * k_limit instead of the vocabulary.
     *
     * Selection: repeatedly take the max of the remaining range and swap it into
     * place. O(vocab * k) copies of a float compare, no function-pointer
     * indirection per comparison, no cache-hostile full sort. With the default
     * top_k = 40 that is 40 linear scans instead of a sort. Order among equal
     * probabilities can differ from qsort's, so outputs are only identical for
     * distinct probabilities; logits after softmax are effectively never tied.
     *
     * top_k <= 0 means "top-k DISABLED" (llama.cpp semantics -- and exactly
     * what a client sending `"top_k":0` asks for; llama.cpp's own default
     * request body does that). Keeping a CAP for that case is a large win:
     * the full sort costs ~35 ms/token on the 27B's 248,320-token vocabulary,
     * which was 42% of the token time and dropped the HTTP server from 19.7 to
     * 11.9 tok/s with the GPU 40% idle -- the server looked ~35% slower than
     * the CLI simply because the CLI's default top_k = 40 takes the cheap path.
     *
     * SAMPLER_TOPK_UNBOUNDED is only used when the retained mass still covers
     * top_p; otherwise the full sort below runs (correct, and cheap to detect).
     * Real LM logits are peaked enough that the cap holds in practice, but the
     * check makes that a measurement rather than an assumption. */
    int want_k = (s->top_k > 0 && s->top_k < vocab_size) ? s->top_k : SAMPLER_TOPK_UNBOUNDED;
    if (want_k > vocab_size) want_k = vocab_size;
    /* The selection below keeps a sorted k-element list, so it must stay cheap:
     * with k near vocab_size the memmove inside the k-array dominates. Above
     * this bound the plain qsort is both faster and exactly equivalent. */
    int use_partial = want_k <= SAMPLER_SELECT_MAX_K;

    int k_limit = use_partial ? want_k : vocab_size;
    prob_index_t *sorted = (prob_index_t *)malloc((size_t)k_limit * sizeof(prob_index_t));
    if (!sorted) return 0;
    if (use_partial) {
        /* Single linear pass over the vocabulary, keeping the best `want_k` in
         * descending order: for each entry that beats the current k-th largest,
         * binary-search its slot and memmove once. O(vocab) in the common case,
         * O(vocab*log k) worst. NOT k repeated full scans of the vocabulary:
         * that is O(vocab*k) = 250M+ ops at k=1024 and measured 397 ms/token. */
        int n = 0;
        float vmin = -INFINITY;         /* sorted[n-1].prob once n == want_k */
        for (int j = 0; j < vocab_size; j++) {
            float v = logits[j];
            if (n == want_k && v <= vmin) continue;
            int lo = 0, hi = n;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (sorted[mid].prob >= v) lo = mid + 1; else hi = mid;
            }
            if (n < want_k) n++;
            if (lo < n - 1)
                memmove(&sorted[lo + 1], &sorted[lo], (size_t)(n - 1 - lo) * sizeof(prob_index_t));
            sorted[lo].prob  = v;
            sorted[lo].index = j;
            if (n == want_k) vmin = sorted[n - 1].prob;
        }
        k_limit = n;

        /* If top-k is disabled, top_p is applied over the FULL vocabulary, so
         * a cap is only safe when the retained entries cover it. Verify, and
         * fall back to the full sort otherwise (flat/untrained logits). */
        if (s->top_k <= 0 && k_limit < vocab_size) {
            double kept = 0.0;
            for (int i = 0; i < k_limit; i++) kept += (double)sorted[i].prob;
            if (kept + 1e-6 < (double)s->top_p) use_partial = 0;
        }
        if (!use_partial) { free(sorted); sorted = NULL; }
    }
    if (!use_partial) {
        k_limit = vocab_size;
        sorted = (prob_index_t *)malloc((size_t)k_limit * sizeof(prob_index_t));
        if (!sorted) return 0;
        /* Full sort: O(vocab * log vocab). */
        for (int i = 0; i < vocab_size; i++) {
            sorted[i].prob  = logits[i];
            sorted[i].index = i;

        }
        qsort(sorted, (size_t)vocab_size, sizeof(prob_index_t), cmp_prob_desc);
    }

    /* 3. Apply top-k: already restricted above (k_limit entries in sorted[]). */

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
