#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- GGUF string reader (reused from model.c logic) ---- */

static uint64_t read_u64_at(const uint8_t **p) {
    uint64_t v;
    memcpy(&v, *p, 8);
    *p += 8;
    return v;
}

/* ---- Sorted index for binary search ---- */

static char **g_vocab_for_sort; /* global for qsort comparison */

static int cmp_sorted(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return strcmp(g_vocab_for_sort[ia], g_vocab_for_sort[ib]);
}

static int vocab_lookup(const tokenizer_t *t, const char *str, int len) {
    /* Binary search in sorted vocabulary */
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int idx = t->sorted_idx[mid];
        int cmp = strncmp(t->vocab[idx], str, (size_t)len);
        if (cmp == 0) {
            /* Check exact length match */
            if (t->vocab[idx][len] == '\0') return idx;
            if (t->vocab[idx][len] > '\0') { hi = mid - 1; }
            else { lo = mid + 1; }
        } else if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1; /* not found */
}

/* ---- Public API ---- */

int tokenizer_load(tokenizer_t *t, const model_t *m) {
    int vs = m->config.vocab_size;
    t->vocab_size = vs;
    t->bos_id = m->tok_bos_id;
    t->eos_id = m->tok_eos_id;
    t->space_marker = m->tok_space_marker;

    /* Allocate vocab and scores arrays */
    t->vocab = (char **)calloc((size_t)vs, sizeof(char *));
    t->scores = (float *)calloc((size_t)vs, sizeof(float));
    t->sorted_idx = (int *)malloc((size_t)vs * sizeof(int));
    if (!t->vocab || !t->scores || !t->sorted_idx) {
        fprintf(stderr, "OOM allocating tokenizer\n");
        return -1;
    }

    /* Read vocab strings from GGUF metadata array */
    if (m->tok_tokens_data && m->tok_n_tokens > 0) {
        const uint8_t *p = (const uint8_t *)m->tok_tokens_data;
        uint64_t n = m->tok_n_tokens;
        if ((int)n > vs) n = (uint64_t)vs;

        for (uint64_t i = 0; i < n; i++) {
            uint64_t slen = read_u64_at(&p);
            /* Allocate and copy the string with null terminator */
            t->vocab[i] = (char *)malloc((size_t)slen + 1);
            if (t->vocab[i]) {
                memcpy(t->vocab[i], p, (size_t)slen);
                t->vocab[i][slen] = '\0';
            }
            p += slen;
        }
    }

    /* Fill any remaining entries with empty strings */
    for (int i = 0; i < vs; i++) {
        if (!t->vocab[i]) {
            t->vocab[i] = (char *)calloc(1, 1);
        }
    }

    /* Read scores */
    if (m->tok_scores_data && m->tok_n_scores > 0) {
        uint64_t n = m->tok_n_scores;
        if ((int)n > vs) n = (uint64_t)vs;
        memcpy(t->scores, m->tok_scores_data, (size_t)n * sizeof(float));
    }

    /* Build sorted index */
    for (int i = 0; i < vs; i++) {
        t->sorted_idx[i] = i;
    }
    g_vocab_for_sort = t->vocab;
    qsort(t->sorted_idx, (size_t)vs, sizeof(int), cmp_sorted);

    fprintf(stderr, "Tokenizer loaded: %d tokens, bos=%u, eos=%u\n",
            vs, t->bos_id, t->eos_id);
    return 0;
}

int tokenizer_encode(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos) {
    int n_tokens = 0;

    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = (int)t->bos_id;
    }

    if (!text || !*text) return n_tokens;

    /* Replace spaces with model's space marker:
     * - Most models: U+2581 (0xE2 0x96 0x81)
     * - SmolLM: U+0100 (0xC4 0xA0) */
    const unsigned char *sp;
    int sp_len;
    if (t->space_marker == 1) {
        static const unsigned char sp_0100[] = {0xC4, 0xA0};
        sp = sp_0100; sp_len = 2;
    } else {
        static const unsigned char sp_2581[] = {0xE2, 0x96, 0x81};
        sp = sp_2581; sp_len = 3;
    }

    int text_len = (int)strlen(text);
    int norm_cap = text_len * (sp_len + 1) + sp_len + 4;
    char *norm = (char *)malloc((size_t)norm_cap);
    int norm_len = 0;

    for (int i = 0; i < sp_len; i++) norm[norm_len++] = (char)sp[i];
    for (int i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
            for (int j = 0; j < sp_len; j++) norm[norm_len++] = (char)sp[j];
        } else {
            norm[norm_len++] = text[i];
        }
    }
    norm[norm_len] = '\0';

    /* Step 1: Convert normalized text to individual character tokens.
     * Each UTF-8 character (including ▁) gets looked up in the vocab. */
    /* Worst case: one token per byte of normalized text */
    int *merge_buf = (int *)malloc((size_t)(norm_len + 1) * sizeof(int));
    int merge_len = 0;

    for (int i = 0; i < norm_len; ) {
        /* Determine UTF-8 character length */
        int clen = 1;
        unsigned char c = (unsigned char)norm[i];
        if (c >= 0xF0) clen = 4;
        else if (c >= 0xE0) clen = 3;
        else if (c >= 0xC0) clen = 2;

        if (i + clen > norm_len) clen = norm_len - i;

        /* Try to find this character in vocab */
        int tok = vocab_lookup(t, norm + i, clen);
        if (tok >= 0) {
            merge_buf[merge_len++] = tok;
            i += clen;
        } else {
            /* Fall back to byte tokens: <0xHH> */
            char byte_tok[8];
            snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)norm[i]);
            tok = vocab_lookup(t, byte_tok, (int)strlen(byte_tok));
            if (tok >= 0) {
                merge_buf[merge_len++] = tok;
            }
            i++;
        }
    }
    free(norm);

    /* Step 2: BPE merge loop — iteratively find best adjacent pair */
    while (merge_len >= 2) {
        float best_score = -1e30f;
        int best_idx = -1;
        int best_tok = -1;

        for (int i = 0; i < merge_len - 1; i++) {
            /* Build the merged string */
            const char *s1 = t->vocab[merge_buf[i]];
            const char *s2 = t->vocab[merge_buf[i + 1]];
            int l1 = (int)strlen(s1);
            int l2 = (int)strlen(s2);

            /* Build concatenation in stack buffer */
            char merged[256];
            if (l1 + l2 >= (int)sizeof(merged)) continue;
            memcpy(merged, s1, (size_t)l1);
            memcpy(merged + l1, s2, (size_t)l2);
            merged[l1 + l2] = '\0';

            int tok = vocab_lookup(t, merged, l1 + l2);
            if (tok >= 0 && t->scores[tok] > best_score) {
                best_score = t->scores[tok];
                best_idx = i;
                best_tok = tok;
            }
        }

        if (best_idx < 0) break; /* no more merges possible */

        /* Apply the merge */
        merge_buf[best_idx] = best_tok;
        /* Shift left */
        for (int i = best_idx + 1; i < merge_len - 1; i++) {
            merge_buf[i] = merge_buf[i + 1];
        }
        merge_len--;
    }

    /* Copy to output */
    for (int i = 0; i < merge_len && n_tokens < max_tokens; i++) {
        tokens[n_tokens++] = merge_buf[i];
    }

    free(merge_buf);
    return n_tokens;
}

const char *tokenizer_decode(const tokenizer_t *t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) return "";

    const char *str = t->vocab[token];

    /* Handle byte tokens: <0xHH> -> single byte */
    if (str[0] == '<' && str[1] == '0' && str[2] == 'x' && str[5] == '>') {
        /* Decode hex byte */
        static char byte_buf[2];
        unsigned int val = 0;
        for (int i = 3; i < 5; i++) {
            val <<= 4;
            char c = str[i];
            if (c >= '0' && c <= '9') val += (unsigned)(c - '0');
            else if (c >= 'A' && c <= 'F') val += (unsigned)(c - 'A' + 10);
            else if (c >= 'a' && c <= 'f') val += (unsigned)(c - 'a' + 10);
        }
        byte_buf[0] = (char)val;
        byte_buf[1] = '\0';
        return byte_buf;
    }

    /* Handle SentencePiece leading space marker "▁" -> " " */
    /* The "▁" character is U+2581, encoded as 0xE2 0x96 0x81 in UTF-8 */
    if ((unsigned char)str[0] == 0xE2 && (unsigned char)str[1] == 0x96 && (unsigned char)str[2] == 0x81) {
        /* Replace leading ▁ with space, but not after BOS */
        static char space_buf[256];
        if (prev_token == (int)t->bos_id) {
            /* After BOS, strip the leading space */
            int len = (int)strlen(str + 3);
            if (len >= (int)sizeof(space_buf)) len = (int)sizeof(space_buf) - 1;
            memcpy(space_buf, str + 3, (size_t)len);
            space_buf[len] = '\0';
            return space_buf;
        }
        space_buf[0] = ' ';
        int len = (int)strlen(str + 3);
        if (len >= (int)sizeof(space_buf) - 1) len = (int)sizeof(space_buf) - 2;
        memcpy(space_buf + 1, str + 3, (size_t)len);
        space_buf[1 + len] = '\0';
        return space_buf;
    }

    /* Handle U+00A0 NO-BREAK SPACE -> regular space */
    if ((unsigned char)str[0] == 0xC2 && (unsigned char)str[1] == 0xA0) {
        static char nbsp_buf[256];
        int len = (int)strlen(str + 2);
        if (len >= (int)sizeof(nbsp_buf)) len = (int)sizeof(nbsp_buf) - 1;
        nbsp_buf[0] = ' ';
        if (len > 0) {
            memcpy(nbsp_buf + 1, str + 2, (size_t)len);
            nbsp_buf[1 + len] = '\0';
        } else {
            nbsp_buf[1] = '\0';
        }
        return nbsp_buf;
    }

    /* Handle U+0100 (LATIN SMALL LETTER A WITH OGONEK) as space marker */
    /* Some GGUF tokenizers use U+0100 (0xC4 0xA0) instead of U+2581 */
    if ((unsigned char)str[0] == 0xC4 && (unsigned char)str[1] == 0xA0) {
        static char u100_buf[256];
        int len = (int)strlen(str + 2);
        if (len >= (int)sizeof(u100_buf)) len = (int)sizeof(u100_buf) - 1;
        u100_buf[0] = ' ';
        if (len > 0) {
            memcpy(u100_buf + 1, str + 2, (size_t)len);
            u100_buf[1 + len] = '\0';
        } else {
            u100_buf[1] = '\0';
        }
        return u100_buf;
    }

    return str;
}

void tokenizer_free(tokenizer_t *t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) {
            free(t->vocab[i]);
        }
        free(t->vocab);
        t->vocab = NULL;
    }
    free(t->scores);
    t->scores = NULL;
    free(t->sorted_idx);
    t->sorted_idx = NULL;
}
