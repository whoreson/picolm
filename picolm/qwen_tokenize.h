#ifndef QWEN_TOKENIZE_H
#define QWEN_TOKENIZE_H

#include "model.h"

typedef struct {
    const char *lp; int ll; const char *rp; int rl;
} qwen_merge_t;

/* Which pretokenizer regex to use. The BPE engine (hash tables, merge loop,
 * byte<->codepoint mapping) is shared; only the pretoken-boundary function
 * differs between model families. */
enum {
    QWEN_PRETOK_DEFAULT = 0, /* Qwen3.5/3.6 / GPT-2 style pretokenizer */
    QWEN_PRETOK_TEKKEN  = 1, /* Mistral Tekken pretokenizer (Mistral-Nemo-2407+) */
};

typedef struct {
    int *tok_tab;
    int *mrg_tab;
    qwen_merge_t *merges; int n_merges;
    int *spec; int n_spec;
    const char **vocab;
    int *vocab_len;
    int vocab_size;
    int bos_id;
    int eos_id;
    int pretok_type; /* QWEN_PRETOK_DEFAULT or QWEN_PRETOK_TEKKEN */
} qwen_enc_t;

/* Check if a model should use the Qwen tokenizer */
int qwen_tokenize_should_use(const model_t *m);

/* Initialize the Qwen tokenizer from model data */
int qwen_tokenize_init(qwen_enc_t *enc, const model_t *m);
void qwen_tokenize_free(qwen_enc_t *enc);

/* Encode text to token IDs. Returns number of tokens produced. */
int qwen_tokenize_encode(qwen_enc_t *enc, const char *text, int *out, int cap);

/* Decode one token ID to UTF-8 text. Returns bytes written.
 * add_special=0: hide BOS/EOS (default behavior).
 * add_special=1: print BOS/EOS too.
 * All other special tokens (tool_call tags, etc.) are always printed. */
int qwen_tokenize_decode2(const qwen_enc_t *enc, int id, char *buf, int cap, int add_special);
int qwen_tokenize_decode(const qwen_enc_t *enc, int id, char *buf, int cap);

#endif /* QWEN_TOKENIZE_H */

