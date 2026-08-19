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

    /* Build special tokens cache (for SPM partitioning).
     * Collect all tokens with type 3 (control), 4 (user_defined), 5 (unknown).
     * Sorted by string length descending so longer matches are found first. */
    t->n_special_tokens = 0;
    t->special_tokens = NULL;
    if (t->token_type) {
        /* First count */
        int nspec = 0;
        int nn = t->n_token_type;
        for (int i = 0; i < nn && i < vs; i++) {
            int32_t ty = t->token_type[i];
            if (ty == 3 || ty == 4 || ty == 5) nspec++;
        }
        if (nspec > 0) {
            t->special_tokens = (int *)malloc((size_t)nspec * sizeof(int));
            int idx = 0;
            for (int i = 0; i < nn && i < vs; i++) {
                int32_t ty = t->token_type[i];
                if (ty == 3 || ty == 4 || ty == 5) {
                    t->special_tokens[idx++] = i;
                }
            }
            t->n_special_tokens = nspec;
            /* Sort by string length descending (longest first) */
            for (int i = 0; i < nspec - 1; i++) {
                for (int j = i + 1; j < nspec; j++) {
                    int li = (int)strlen(t->vocab[t->special_tokens[i]]);
                    int lj = (int)strlen(t->vocab[t->special_tokens[j]]);
                    if (lj > li) {
                        int tmp = t->special_tokens[i];
                        t->special_tokens[i] = t->special_tokens[j];
                        t->special_tokens[j] = tmp;
                    }
                }
            }
        }
    }

    fprintf(stderr, "Tokenizer loaded: %d tokens, bos=%u, eos=%u, special=%d\n",
            vs, t->bos_id, t->eos_id, t->n_special_tokens);
    return 0;
}

/* ---- SPM helpers ---- */

/* Tokenize a single raw text fragment using SPM merge.
 * Writes tokens starting at tokens[out_pos], increments *out_pos.
 * Returns the number of new tokens appended. */
static int spm_tokenize_fragment(const tokenizer_t *t, const char *frag, int frag_len,
                                  int *tokens, int max_tokens, int *out_pos,
                                  int prepend_space) {
    const unsigned char *sp;
    int sp_len;
    if (t->space_marker == 1 || t->space_marker == 3) {
        static const unsigned char sp_0100[] = {0xC4, 0xA0};
        sp = sp_0100; sp_len = 2;
    } else if (t->space_marker == 2) {
        static const unsigned char sp_literal[] = {' '};
        sp = sp_literal; sp_len = 1;
    } else {
        static const unsigned char sp_2581[] = {0xE2, 0x96, 0x81};
        sp = sp_2581; sp_len = 3;
    }

    /* Normalize: replace spaces with space marker, optionally prepend */
    int norm_cap = frag_len * (sp_len + 1) + (prepend_space ? sp_len : 0) + 4;
    char *norm = (char *)malloc((size_t)norm_cap);
    int norm_len = 0;

    if (prepend_space) {
        for (int i = 0; i < sp_len; i++) norm[norm_len++] = (char)sp[i];
    }
    for (int i = 0; i < frag_len; i++) {
        if (frag[i] == ' ') {
            for (int j = 0; j < sp_len; j++) norm[norm_len++] = (char)sp[j];
        } else {
            norm[norm_len++] = frag[i];
        }
    }
    norm[norm_len] = '\0';

    /* Step 1: Split into character tokens */
    int *merge_buf = (int *)malloc((size_t)(norm_len + 1) * sizeof(int));
    int merge_len = 0;

    for (int i = 0; i < norm_len; ) {
        int clen = 1;
        unsigned char c = (unsigned char)norm[i];
        if (c >= 0xF0) clen = 4;
        else if (c >= 0xE0) clen = 3;
        else if (c >= 0xC0) clen = 2;
        if (i + clen > norm_len) clen = norm_len - i;

        int tok = vocab_lookup(t, norm + i, clen);
        if (tok >= 0) {
            merge_buf[merge_len++] = tok;
            i += clen;
        } else {
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

    /* Step 2: SPM merge loop */
    while (merge_len >= 2) {
        float best_score = -1e30f;
        int best_idx = -1;
        int best_tok = -1;

        for (int i = 0; i < merge_len - 1; i++) {
            const char *s1 = t->vocab[merge_buf[i]];
            const char *s2 = t->vocab[merge_buf[i + 1]];
            int l1 = (int)strlen(s1);
            int l2 = (int)strlen(s2);

            char merged[256];
            if (l1 + l2 >= (int)sizeof(merged)) continue;
            memcpy(merged, s1, (size_t)l1);
            memcpy(merged + l1, s2, (size_t)l2);
            merged[l1 + l2] = '\0';

            int merged_tok = vocab_lookup(t, merged, l1 + l2);
            if (merged_tok >= 0 && t->scores[merged_tok] > best_score) {
                best_score = t->scores[merged_tok];
                best_idx = i;
                best_tok = merged_tok;
            }
        }

        if (best_idx < 0) break;

        merge_buf[best_idx] = best_tok;
        for (int i = best_idx + 1; i < merge_len - 1; i++) {
            merge_buf[i] = merge_buf[i + 1];
        }
        merge_len--;
    }

    /* Copy to output */
    int count = 0;
    for (int i = 0; i < merge_len && *out_pos < max_tokens; i++) {
        tokens[(*out_pos)++] = merge_buf[i];
        count++;
    }

    free(merge_buf);
    return count;
}

/* Fragment types for partitioning */
#define FRAG_TYPE_RAW   0
#define FRAG_TYPE_TOKEN 1

typedef struct {
    int type;          /* FRAG_TYPE_RAW or FRAG_TYPE_TOKEN */
    int offset;        /* offset into original text (for raw) */
    int length;        /* length in bytes (for raw) */
    int token_id;      /* token ID (for FRAG_TYPE_TOKEN) */
} frag_t;

int tokenizer_encode(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos) {
    int n_tokens = 0;

    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = (int)t->bos_id;
    }

    if (!text || !*text) return n_tokens;

    int text_len = (int)strlen(text);

    /* If no special tokens are defined, fall through to simple SPM on the whole text.
     * This preserves backward compatibility for models without special token types. */
    if (t->n_special_tokens == 0) {
        /* Use the fragment tokenizer without partitioning */
        int out_pos = n_tokens;
        int prepend_space = (t->space_marker != 3) && t->add_space_prefix;
        (void)spm_tokenize_fragment(t, text, text_len, tokens, max_tokens, &out_pos, prepend_space);
        return out_pos;
    }

    /* ---- Special token partitioning ----
     * Scan text for special token occurrences (longest first),
     * splitting into fragments of raw text and token IDs. */

    /* Allocate worst-case fragments: every character could be a special match boundary.
     * In practice, it'll be far fewer. */
    int max_frags = text_len + 1;
    frag_t *frags = (frag_t *)calloc((size_t)max_frags, sizeof(frag_t));
    int n_frags = 0;

    /* We process the text left-to-right, finding the earliest special token match.
     * Among all special tokens, we find the one with the earliest occurrence;
     * if there's a tie, we prefer the longest match (since they're sorted longest-first,
     * we check in order and take the earliest offset). */
    int pos = 0; /* current position in text */
    while (pos < text_len) {
        /* Find the earliest special token match at or after pos */
        int best_offset = -1;
        int best_tok = -1;
        int best_slen = 0;

        for (int si = 0; si < t->n_special_tokens; si++) {
            int tok_id = t->special_tokens[si];
            const char *tok_str = t->vocab[tok_id];
            int tok_len = (int)strlen(tok_str);
            if (tok_len == 0) continue;

            /* Search for this special token at or after pos */
            for (int search_pos = pos; search_pos <= text_len - tok_len; search_pos++) {
                if (strncmp(text + search_pos, tok_str, (size_t)tok_len) == 0) {
                    if (best_offset < 0 || search_pos < best_offset ||
                        (search_pos == best_offset && tok_len > best_slen)) {
                        best_offset = search_pos;
                        best_tok = tok_id;
                        best_slen = tok_len;
                    }
                    break; /* found earliest occurrence of this special token */
                }
            }
        }

        if (best_offset < 0) {
            /* No more special tokens found — rest is raw text */
            if (pos < text_len) {
                frags[n_frags].type = FRAG_TYPE_RAW;
                frags[n_frags].offset = pos;
                frags[n_frags].length = text_len - pos;
                n_frags++;
            }
            break;
        }

        /* Raw text fragment before the special token match */
        if (best_offset > pos) {
            frags[n_frags].type = FRAG_TYPE_RAW;
            frags[n_frags].offset = pos;
            frags[n_frags].length = best_offset - pos;
            n_frags++;
        }

        /* Special token fragment */
        frags[n_frags].type = FRAG_TYPE_TOKEN;
        frags[n_frags].token_id = best_tok;
        n_frags++;

        pos = best_offset + best_slen;
    }

    /* Now process fragments: raw text gets SPM tokenized, tokens are inserted directly */
    int out_pos = n_tokens;
    int prev_was_special = 1; /* start of text counts as "after special" for space prefix */
    int prepend_space = (t->space_marker != 3) && t->add_space_prefix;

    for (int fi = 0; fi < n_frags; fi++) {
        if (frags[fi].type == FRAG_TYPE_TOKEN) {
            if (out_pos < max_tokens) {
                tokens[out_pos++] = frags[fi].token_id;
            }
            prev_was_special = 1;
        } else {
            /* Raw text fragment — SPM tokenize with whitespace normalization.
             * If add_space_prefix is true and previous fragment was special,
             * prepend a space marker. */
            int do_prepend = prepend_space && prev_was_special;
            (void)spm_tokenize_fragment(t, text + frags[fi].offset, frags[fi].length,
                                         tokens, max_tokens, &out_pos, do_prepend);
            prev_was_special = 0;
        }
    }

    free(frags);
    return out_pos;
}

const char *tokenizer_decode(const tokenizer_t *t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) return "";

    /* Only hide BOS/EOS; other special tokens (tool_call tags, etc.)
     * must still be printed so stop-word matching can see them. */
    if ((uint32_t)token == t->bos_id || (uint32_t)token == t->eos_id) return "";

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
    free(t->special_tokens);
    t->special_tokens = NULL;
    t->n_special_tokens = 0;
}
