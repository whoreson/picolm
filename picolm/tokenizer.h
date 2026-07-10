#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "model.h"
#include <stdint.h>

typedef struct {
    char  **vocab;       /* vocab[i] = string for token i */
    float  *scores;      /* BPE merge scores */
    int     vocab_size;
    int    *sorted_idx;  /* indices sorted by vocab string for binary search */
    uint32_t bos_id;
    uint32_t eos_id;
    int space_marker; /* 0=U+2581 (default), 1=U+0100 (smollm), 2=literal space (qwen35/gpt2) */
} tokenizer_t;

/* Load tokenizer data from GGUF metadata pointers in model.
 * Returns 0 on success. */
int tokenizer_load(tokenizer_t *t, const model_t *m);

/* Encode a text string into token IDs.
 * tokens must have space for at least max_tokens entries.
 * Returns number of tokens produced. */
int tokenizer_encode(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos);

/* Decode a single token ID to its string representation.
 * Returns pointer to static/vocab string (do not free). */
const char *tokenizer_decode(const tokenizer_t *t, int prev_token, int token);

/* Free tokenizer resources */
void tokenizer_free(tokenizer_t *t);

#endif /* TOKENIZER_H */
