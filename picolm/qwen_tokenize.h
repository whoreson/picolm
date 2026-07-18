#ifndef QWEN_TOKENIZE_H
#define QWEN_TOKENIZE_H

#include "model.h"

typedef struct {
    const char *lp; int ll; const char *rp; int rl;
} qwen_merge_t;

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
} qwen_enc_t;

/* Check if a model should use the Qwen tokenizer */
int qwen_tokenize_should_use(const model_t *m);

/* Initialize the Qwen tokenizer from model data */
int qwen_tokenize_init(qwen_enc_t *enc, const model_t *m);
void qwen_tokenize_free(qwen_enc_t *enc);

/* Encode text to token IDs. Returns number of tokens produced. */
int qwen_tokenize_encode(qwen_enc_t *enc, const char *text, int *out, int cap);

/* Decode one token ID to UTF-8 text. Returns bytes written. */
int qwen_tokenize_decode(const qwen_enc_t *enc, int id, char *buf, int cap);

#endif /* QWEN_TOKENIZE_H */

