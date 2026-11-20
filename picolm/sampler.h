#ifndef PICOLM_SAMPLER_H
#define PICOLM_SAMPLER_H

#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  Sampling parameters -- ONE definition for every caller             */
/*                                                                     */
/*  Everything the sampler can act on lives here, and every front end   */
/*  (CLI, HTTP server, --benchmark-ctx) fills this struct instead of    */
/*  calling sampler_init() with a position-dependent argument list.     */
/*                                                                     */
/*  This exists because the previous hand-rolled argument lists drifted: */
/*    - min_p was hardcoded 0.05 at all three call sites while /props    */
/*      advertised it as settable for *every* request (including /props  */
/*      and /slots, which print DEFAULT_SAMPLING_PARAMS directly);       */
/*    - repeat_penalty / repeat_last_n were parsed from the request and  */
/*      echoed back in generation_settings but never reached the sampler,*/
/*      so the API described a sampler that did not exist;               */
/*    - --benchmark-ctx passed a literal 0.95f for top_p, ignoring       */
/*      --top-p entirely;                                                */
/*    - /completion defaulted top_p to 0.90 while /v1/completions        */
/*      defaulted it to 0.95.                                            */
/*  A struct with one initialiser makes each of those a compile-time or  */
/*  single-point change instead of a silent divergence.                 */
/* ------------------------------------------------------------------ */

/* llama.cpp-compatible defaults. Kept in one place; every caller that has no
 * opinion uses these rather than spelling the numbers out again. */
#define PICOLM_DEF_TEMPERATURE   0.80f
#define PICOLM_DEF_TOP_P         0.90f   /* was 0.90 in /completion, 0.95 in /v1 */
#define PICOLM_DEF_TOP_K         40
#define PICOLM_DEF_MIN_P         0.05f
#define PICOLM_DEF_REPEAT_PENALTY 1.00f  /* 1.0 = disabled (llama.cpp default) */
#define PICOLM_DEF_REPEAT_LAST_N 64

typedef struct {
    float    temperature;      /* <= 0 = greedy                                */
    float    top_p;            /* 1.0 = disabled                               */
    int      top_k;            /* <= 0 = disabled                              */
    float    min_p;            /* 0.0 = disabled                               */
    /* Repetition penalties, applied to RAW LOGITS before softmax, in
     * llama.cpp's order and with its sign convention:
     *   logit <= 0 : logit *= repeat_penalty
     *   logit >  0 : logit /= repeat_penalty
     * A token occurring `count` times in the last repeat_last_n generated
     * tokens is also reduced by count*freq + (count>0)*present; freq/present
     * are 0 in this engine (no API for them yet) but the slots document where
     * they belong, and the loop already computes `count`. */
    float    repeat_penalty;   /* 1.0 = disabled                               */
    int      repeat_last_n;    /* 0 = disabled                                 */
} sampling_params_t;

/* Defaults, usable as an initialiser: sampling_params_t p = SAMPLING_DEFAULTS; */
#define SAMPLING_DEFAULTS {                                   \
    PICOLM_DEF_TEMPERATURE, PICOLM_DEF_TOP_P, PICOLM_DEF_TOP_K, \
    PICOLM_DEF_MIN_P, PICOLM_DEF_REPEAT_PENALTY, PICOLM_DEF_REPEAT_LAST_N }

typedef struct {
    sampling_params_t params;
    uint64_t          rng_state;   /* xorshift64 state */
    /* Repetition-penalty history. The sampler maintains this itself: every
     * returned token is appended, so no call site has to remember to do it (a
     * per-caller ring is exactly the kind of duplicated bookkeeping that let
     * repeat_penalty be parsed and then ignored).
     * Seeded with the prompt via sampler_set_history(); grows to
     * repeat_last_n and then wraps. */
    int              *history;     /* owned, capacity HIST_CAP */
    int               n_history;   /* valid entries (<= HIST_CAP) */
    int               history_head;/* next write index */
} sampler_t;

/* How many tokens the sampler can remember. repeat_last_n is clamped to this. */
#define SAMPLER_HISTORY_CAP 4096

/* ---- Construction ---- */

/* Default parameter set (identical to SAMPLING_DEFAULTS). */
sampling_params_t sampling_params_default(void);

/* Validate/normalise in place. Clamps nothing silently that a user could not
 * see coming; only repairs values that cannot mean anything (negative top_k,
 * NaN temperature, repeat_last_n < 0). Returns 0 on success, -1 if a value is
 * so far out of range that the request should be rejected. */
int sampling_params_check(sampling_params_t *p);

/* Initialise with an explicit parameter set. `seed` 0 selects the default. */
void sampler_init(sampler_t *s, const sampling_params_t *p, uint64_t seed);

/* Convenience wrapper for the common "temperature/top_p/top_k/min_p" case with
 * penalties disabled. Kept so existing call sites stay short, but it fills the
 * SAME struct, so nothing can be silently omitted. */
void sampler_init_simple(sampler_t *s, float temperature, float top_p, int top_k,
                         float min_p, uint64_t seed);

/* Name kept for the previous API: same as sampler_init_simple(). */
void sampler_init_legacy(sampler_t *s, float temperature, float top_p, int top_k,
                         float min_p, uint64_t seed);

/* Seed the repetition-penalty history with the prompt (or the tail of it).
 * The sampler copies the last SAMPLER_HISTORY_CAP tokens; generated tokens are
 * appended automatically by sampler_sample(). Calling this again restarts the
 * history, which is what a new request wants. */
void sampler_set_history(sampler_t *s, const int *tokens, int n_tokens);

/* Free the history. Safe on a zeroed/never-initialised sampler. */
void sampler_free(sampler_t *s);

/* ---- Sampling ---- */

/* Sample the next token. `logits` are RAW model logits and are modified in
 * place (penalties, then softmax), matching the previous behaviour. */
int  sampler_sample(sampler_t *s, float *logits, int vocab_size);

#endif /* PICOLM_SAMPLER_H */
