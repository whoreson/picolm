#include "grammar.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define NEG_INF (-1e30f)

/* ---- Per-token analysis ---- */

/* Count net brace/bracket deltas and check for unmatched quotes in a token string.
 * This is done once at init time, not at every sampling step. */
static void analyze_token(const char *str, int8_t *brace_delta, int8_t *bracket_delta,
                          int8_t *has_unmatched_quote, uint8_t *first_byte) {
    *first_byte = (uint8_t)str[0];
    int bd = 0, bkd = 0, quotes = 0, escape = 0;

    for (const char *p = str; *p; p++) {
        if (escape) {
            escape = 0;
            continue;
        }
        switch (*p) {
            case '\\': escape = 1; break;
            case '{':  bd++;  break;
            case '}':  bd--;  break;
            case '[':  bkd++; break;
            case ']':  bkd--; break;
            case '"':  quotes++; break;
        }
    }

    /* Clamp to int8 range */
    if (bd > 127) bd = 127;
    if (bd < -128) bd = -128;
    if (bkd > 127) bkd = 127;
    if (bkd < -128) bkd = -128;

    *brace_delta = (int8_t)bd;
    *bracket_delta = (int8_t)bkd;
    *has_unmatched_quote = (int8_t)(quotes & 1);
}

/* ---- Public API ---- */

void grammar_init(grammar_state_t *g, grammar_mode_t mode,
                  const tokenizer_t *tok) {
    memset(g, 0, sizeof(*g));
    g->mode = mode;
    g->vocab_size = tok->vocab_size;
    g->eos_id = (int)tok->eos_id;

    if (mode == GRAMMAR_NONE) return;

    /* The native Qwen BPE tokenizer keeps its vocabulary in qwen_enc_t, not
     * in tokenizer_t::vocab, so tok->vocab is NULL for those models. The
     * brace-counting JSON mode has always needed tokenizer_t; degrade to no
     * constraining rather than dereferencing NULL. */
    if (!tok->vocab) {
        fprintf(stderr, "WARN: --json needs a tokenizer with a vocab table "
                        "(not available for this tokenizer); JSON mode disabled\n");
        g->mode = GRAMMAR_NONE;
        return;
    }
    int vs = tok->vocab_size;
    g->token_brace_delta    = (int8_t *)calloc((size_t)vs, sizeof(int8_t));
    g->token_bracket_delta  = (int8_t *)calloc((size_t)vs, sizeof(int8_t));
    g->token_first_byte     = (uint8_t *)calloc((size_t)vs, sizeof(uint8_t));
    g->token_has_unmatched_quote = (int8_t *)calloc((size_t)vs, sizeof(int8_t));

    /* Pre-analyze every token */
    for (int i = 0; i < vs; i++) {
        const char *s = tok->vocab[i];
        if (s && *s) {
            analyze_token(s, &g->token_brace_delta[i],
                         &g->token_bracket_delta[i],
                         &g->token_has_unmatched_quote[i],
                         &g->token_first_byte[i]);
        }
    }
}

void grammar_apply(grammar_state_t *g, float *logits, int vocab_size) {
    if (g->mode == GRAMMAR_NONE) return;

    int total_depth = g->brace_depth + g->bracket_depth;

    if (!g->started) {
        /* Force first token to start JSON: must begin with '{' or '[' */
        for (int i = 0; i < vocab_size; i++) {
            uint8_t fb = g->token_first_byte[i];
            if (fb != '{' && fb != '[' && fb != ' ' && fb != '\n') {
                logits[i] = NEG_INF;
            }
        }
        /* Also prevent EOS before we even start */
        logits[g->eos_id] = NEG_INF;
        return;
    }

    /* During generation */
    for (int i = 0; i < vocab_size; i++) {
        if (i == g->eos_id) continue; /* handle EOS separately */

        int new_brace   = g->brace_depth   + g->token_brace_delta[i];
        int new_bracket = g->bracket_depth  + g->token_bracket_delta[i];
        int new_total   = new_brace + new_bracket;

        /* Prevent depth from going negative (unmatched closing) */
        if (new_brace < 0 || new_bracket < 0) {
            logits[i] = NEG_INF;
            continue;
        }

        /* If we're in a string, don't constrain (allow any content) */
        if (g->in_string) continue;

        /* Prevent excessively deep nesting (runaway) */
        if (new_total > 50) {
            logits[i] = NEG_INF;
            continue;
        }
    }

    /* Handle EOS: only allow when all braces/brackets are balanced */
    if (total_depth > 0) {
        logits[g->eos_id] = NEG_INF;
    }

    /* If depth is 0 and we've started, boost EOS to encourage stopping */
    if (total_depth == 0 && g->started) {
        /* Strongly encourage EOS when JSON is complete */
        float max_logit = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
        logits[g->eos_id] = max_logit + 5.0f;
    }
}

void grammar_advance(grammar_state_t *g, const tokenizer_t *tok, int token) {
    if (g->mode == GRAMMAR_NONE) return;
    if (token < 0 || token >= g->vocab_size) return;

    const char *str = tok->vocab[token];
    if (!str) return;

    /* Track state character by character */
    for (const char *p = str; *p; p++) {
        if (g->escape_next) {
            g->escape_next = 0;
            continue;
        }

        char c = *p;

        if (g->in_string) {
            if (c == '\\') {
                g->escape_next = 1;
            } else if (c == '"') {
                g->in_string = 0;
            }
        } else {
            switch (c) {
                case '{': g->brace_depth++; g->started = 1; break;
                case '}': g->brace_depth--; break;
                case '[': g->bracket_depth++; g->started = 1; break;
                case ']': g->bracket_depth--; break;
                case '"': g->in_string = 1; break;
            }
        }
    }
}

int grammar_is_complete(const grammar_state_t *g) {
    if (g->mode == GRAMMAR_NONE) return 0;
    return g->started && g->brace_depth == 0 && g->bracket_depth == 0 && !g->in_string;
}

void grammar_free(grammar_state_t *g) {
    free(g->token_brace_delta);
    free(g->token_bracket_delta);
    free(g->token_first_byte);
    free(g->token_has_unmatched_quote);
    memset(g, 0, sizeof(*g));
}
