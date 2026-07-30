#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

/* ---- GGUF string reader (reused from model.c logic) ---- */

static uint64_t read_u64_at(const uint8_t **p) {
    uint64_t v;
    memcpy(&v, *p, 8);
    *p += 8;
    return GGUF_LE64(v);
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
    t->add_space_prefix = m->tok_add_space_prefix;
    t->token_type = NULL;
    t->n_token_type = 0;

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
#if defined(__APPLE__) && defined(__ppc__)
        /* GGUF stores F32 as little-endian; swap on big-endian */
        { const uint32_t *src32 = (const uint32_t *)m->tok_scores_data;
          for (uint64_t i = 0; i < n; i++) {
              uint32_t vi = GGUF_LE32(src32[i]);
              memcpy(&t->scores[i], &vi, 4);
          }
        }
#else
        memcpy(t->scores, m->tok_scores_data, (size_t)n * sizeof(float));
#endif
    }

    /* Read token_type array (3=control, 4=user_defined) */
    if (m->tok_token_type_data && m->tok_n_token_type > 0) {
        t->token_type = (const int32_t *)m->tok_token_type_data;
        t->n_token_type = (int)m->tok_n_token_type;
        if (t->n_token_type > vs) t->n_token_type = vs;
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
     * - SmolLM: U+0100 (0xC4 0xA0)
     * - Qwen3.5: U+0100 but NO prefix on first token (marker=3) */
    const unsigned char *sp;
    int sp_len;
    if (t->space_marker == 1) {
        static const unsigned char sp_0100[] = {0xC4, 0xA0};
        sp = sp_0100; sp_len = 2;
    } else if (t->space_marker == 2) {
        static const unsigned char sp_literal[] = {' '};
        sp = sp_literal; sp_len = 1;
    } else if (t->space_marker == 3) {
        static const unsigned char sp_0100[] = {0xC4, 0xA0};
        sp = sp_0100; sp_len = 2;
    } else {
        static const unsigned char sp_2581[] = {0xE2, 0x96, 0x81};
        sp = sp_2581; sp_len = 3;
    }

    /* Determine whether to prepend a space marker.
     * Controlled by the GGUF key tokenizer.ggml.add_space_prefix.
     * - SentencePiece (llama) models: typically true (implicit)
     * - GPT-2 BPE models: typically false
     * - Qwen3.5 (marker=3): never prepend */
    int prepend_space = (t->space_marker != 3) && t->add_space_prefix;

    int text_len = (int)strlen(text);
    int norm_cap = text_len * (sp_len + 1) + (prepend_space ? sp_len : 0) + 4;
    char *norm = (char *)malloc((size_t)norm_cap);
    int norm_len = 0;

    if (prepend_space) {
        for (int i = 0; i < sp_len; i++) norm[norm_len++] = (char)sp[i];
    }
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

    /* Suppress special tokens (control=3, user_defined=4) */
    if (t->token_type) {
        int nn = t->n_token_type;
        if (token < nn) {
            int32_t ty = GGUF_LE32((uint32_t)t->token_type[token]);
            if (ty == 3 || ty == 4) return "";
        }
    }

    const char *str = t->vocab[token];

    /* Handle byte tokens: <0xHH> -> single byte */
    /* Must check length first to avoid OOB read on short strings */
    {
        int slen = (int)strlen(str);
        if (slen == 6 && str[0] == '<' && str[1] == '0' && str[2] == 'x' && str[5] == '>') {
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
    }

    /* Unescape whitespace markers in the token string.
     * llama.cpp: replace_all(word, "\xe2\x96\x81", " ");
     * Must handle ALL occurrences, not just the leading one,
     * because tokens like "  " (4-space indent) are stored as
     * multiple U+2581 markers, each representing one space.
     *
     * After BOS, ALL leading space markers are stripped (not just the
     * first one), matching SentencePiece spec where the first
     * non-control token after BOS should have all leading spaces removed.
     *
     * Only U+2581 (SentencePiece) and U+0100 (SmolLM/Qwen) are treated
     * as space markers. U+00A0 is a literal non-breaking space and is
     * NOT converted. */
    static char space_buf[256];
    int out = 0;
    const char *s = str;
    int skip_leading = (prev_token == (int)t->bos_id);
    int changed = 0;

    while (*s && out < (int)sizeof(space_buf) - 1) {
        if ((unsigned char)s[0] == 0xE2 && (unsigned char)s[1] == 0x96 && (unsigned char)s[2] == 0x81) {
            /* U+2581 SentencePiece space marker */
            if (!skip_leading) {
                space_buf[out++] = ' ';
            } else {
                /* Skip all leading space markers after BOS */
            }
            s += 3;
            changed = 1;
        } else if ((unsigned char)s[0] == 0xC4 && (unsigned char)s[1] == 0xA0) {
            /* U+0100 (SmolLM/Qwen-style space marker) */
            if (!skip_leading) {
                space_buf[out++] = ' ';
            } else {
                /* Skip all leading space markers after BOS */
            }
            s += 2;
            changed = 1;
        } else {
            /* First non-space character: stop skipping */
            skip_leading = 0;
            space_buf[out++] = *s++;
        }
    }
    if (changed) {
        space_buf[out] = '\0';
        return space_buf;
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
