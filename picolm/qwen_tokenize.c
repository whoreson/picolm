/* Native GPT-2 byte-level BPE tokenizer, shared by Qwen3.5/3.6 and Mistral
 * Tekken (Mistral-Nemo-2407 and later) models.
 * Qwen support ported from q36 (https://github.com/ambud-sh/q36) for PicoLM.
 *
 * Uses the model's GGUF metadata directly: vocab strings, merges, token_type.
 * The BPE engine (byte<->codepoint mapping, FNV hash tables, rank-based merge
 * loop) is shared across model families. Only the pretokenizer -- the
 * function that decides where to cut the input into initial chunks before
 * BPE merging -- differs, since each family ships its own pretokenizer
 * regex in tokenizer.json:
 *   - QWEN_PRETOK_DEFAULT: Qwen3.5/3.6 rules (contractions, L/N categories,
 *     whitespace handling). Also used as-is for plain GPT-2 models.
 *   - QWEN_PRETOK_TEKKEN: Mistral's Tekken pretokenizer regex (see
 *     pretok_next_tekken() below for the exact pattern it approximates).
 *
 * Selected at load time by qwen_tokenize_should_use() / qwen_tokenize_init()
 * based on model.config.is_qwen / is_gpt2 / is_tekken / from_safetensors.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "model.h"
#include "qwen_tokenize.h"

/* ---- GPT-2 byte <-> codepoint mapping ---- */
static int g_rev[512];   /* codepoint (<512) -> byte, or -1 */
static int g_fwd[256];   /* byte -> mapped codepoint */
static int g_maps_ready = 0;

static void build_maps(void) {
    int keep[256];
    memset(keep, 0, sizeof keep);
    for (int b = '!'; b <= '~'; b++) keep[b] = 1;
    for (int b = 0xA1; b <= 0xAC; b++) keep[b] = 1;
    for (int b = 0xAE; b <= 0xFF; b++) keep[b] = 1;
    for (int i = 0; i < 512; i++) g_rev[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int cp;
        if (keep[b]) cp = b;
        else { cp = 256 + n; n++; }
        if (cp < 512) g_rev[cp] = b;
    }
    for (int cp = 0; cp < 512; cp++) {
        if (g_rev[cp] >= 0) g_fwd[g_rev[cp]] = cp;
    }
    g_maps_ready = 1;
}

/* ---- Unicode classification for pretokenizer ---- */
typedef struct { uint32_t a, b; } urange;

static const urange UL[] = {
    {0x41,0x5A},{0x61,0x7A},{0xAA,0xAA},{0xB5,0xB5},{0xBA,0xBA},
    {0xC0,0xD6},{0xD8,0xF6},{0xF8,0x2FF},{0x370,0x5FF},{0x620,0x64A},
    {0x671,0x6D3},{0x710,0x74F},{0x780,0x7B1},{0x900,0x963},{0x971,0x9FF},
    {0xA00,0xDFF},{0xE01,0xE4F},{0xE81,0xEDF},{0xF00,0xFFF},{0x10A0,0x13FF},
    {0x1E00,0x1FFF},{0x2C00,0x2FEF},{0x3040,0x30FF},{0x3105,0x318E},
    {0x31A0,0x31BF},{0x3400,0x4DBF},{0x4E00,0x9FFF},{0xA000,0xA48F},
    {0xAC00,0xD7A3},{0xF900,0xFAFF},{0xFB00,0xFB4F},{0xFE70,0xFEFC},
    {0xFF21,0xFF3A},{0xFF41,0xFF5A},{0xFF66,0xFFDC},{0x20000,0x2FA1F}
};
static const urange UN[] = {
    {0x30,0x39},{0x660,0x669},{0x6F0,0x6F9},{0x966,0x96F},
    {0x9E6,0x9EF},{0xA66,0xA6F},{0xAE6,0xAEF},{0xB66,0xB6F},
    {0xBE6,0xBEF},{0xC66,0xC6F},{0xCE6,0xCEF},{0xD66,0xD6F},
    {0xE50,0xE59},{0xED0,0xED9},{0xF20,0xF29},{0xFF10,0xFF19}
};

static int in_ranges(uint32_t cp, const urange *r, int n) {
    for (int i = 0; i < n; i++) if (cp >= r[i].a && cp <= r[i].b) return 1;
    return 0;
}
static int is_L(uint32_t cp) { return in_ranges(cp, UL, (int)(sizeof UL/sizeof*UL)); }
static int is_N(uint32_t cp) { return in_ranges(cp, UN, (int)(sizeof UN/sizeof*UN)); }
static int is_WS(uint32_t cp) {
    return cp==' '||cp=='\t'||cp=='\n'||cp=='\r'||cp==0x0B||cp==0x0C||cp==0xA0||
           (cp>=0x2000&&cp<=0x200A)||cp==0x2028||cp==0x2029||cp==0x202F||cp==0x205F||cp==0x3000;
}

/* ---- UTF-8 decoder ---- */
static int utf8_next(const char *s, int len, int *cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c >> 5) == 0x6 && len >= 2) { *cp = ((c&0x1F)<<6)|((unsigned char)s[1]&0x3F); return 2; }
    if ((c >> 4) == 0xE && len >= 3) { *cp = ((c&0x0F)<<12)|(((unsigned char)s[1]&0x3F)<<6)|((unsigned char)s[2]&0x3F); return 3; }
    if ((c >> 3) == 0x1E && len >= 4) { *cp = ((c&0x07)<<18)|(((unsigned char)s[1]&0x3F)<<12)|(((unsigned char)s[2]&0x3F)<<6)|((unsigned char)s[3]&0x3F); return 4; }
    *cp = c; return 1;
}

/* ---- Hash functions ---- */
static uint32_t fnv(const char *s, int n, uint32_t seed) {
    uint32_t h = 2166136261u ^ seed;
    for (int i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}

/* ---- Qwen tokenizer state ---- */
enum { QWEN_TOKHASH_BITS = 19, QWEN_MRGHASH_BITS = 20 };

/* qwen_merge_t and qwen_enc_t typedefs are in qwen_tokenize.h */

static int tok_lookup(const char **vocab, int *vocab_len, int *tok_tab, int vs,
                      const char *s, int n) {
    (void)vs;
    uint32_t m = (1u << QWEN_TOKHASH_BITS) - 1;
    uint32_t h = fnv(s, n, 0) & m;
    for (;;) {
        int id = tok_tab[h];
        if (id < 0) return -1;
        if (vocab_len[id] == n && memcmp(vocab[id], s, n) == 0) return id;
        h = (h + 1) & m;
    }
}

static int merge_rank(const qwen_merge_t *merges, int *mrg_tab,
                      const char *l, int ll, const char *r, int rl) {
    uint32_t m = (1u << QWEN_MRGHASH_BITS) - 1;
    uint32_t h = (fnv(l, ll, 0) * 31u + fnv(r, rl, 7)) & m;
    for (;;) {
        int idx = mrg_tab[h];
        if (idx < 0) return -1;
        const qwen_merge_t *mg = &merges[idx];
        if (mg->ll == ll && mg->rl == rl &&
            memcmp(mg->lp, l, ll) == 0 && memcmp(mg->rp, r, rl) == 0)
            return idx;
        h = (h + 1) & m;
    }
}

/* ---- Preetokenizer ---- */
static int pretok_next(const char *s, int n) {
    int cp, adv;

    /* 1: contractions (case-insensitive) */
    if (s[0] == '\'' && n >= 2) {
        char c1 = s[1] | 32, c2 = (n >= 3) ? (s[2] | 32) : 0;
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
            (c1 == 'l' && c2 == 'l')) return 3;
    }

    adv = utf8_next(s, n, &cp);

    /* 2: optional non-[\r\n L N] prefix + letters */
    {
        int j = 0, c0 = cp, a0 = adv;
        if (c0 != '\r' && c0 != '\n' && !is_L(c0) && !is_N(c0)) {
            if (a0 < n) {
                int c2, a2 = utf8_next(s + a0, n - a0, &c2);
                if (is_L(c2)) {
                    j = a0 + a2;
                    while (j < n) { int c3, a3 = utf8_next(s + j, n - j, &c3); if (!is_L(c3)) break; j += a3; }
                    return j;
                }
            }
        } else if (is_L(c0)) {
            j = a0;
            while (j < n) { int c3, a3 = utf8_next(s + j, n - j, &c3); if (!is_L(c3)) break; j += a3; }
            return j;
        }
    }

    /* 3: single number */
    if (is_N(cp)) return adv;

    /* 4: optional space + punct run + newlines */
    {
        int j = 0;
        if (cp == ' ' && adv < n) {
            int c2;
            (void)utf8_next(s + adv, n - adv, &c2);
            if (!is_WS(c2) && !is_L(c2) && !is_N(c2)) j = adv;
        }
        if (j > 0 || (!is_WS(cp) && !is_L(cp) && !is_N(cp))) {
            int k = j;
            while (k < n) {
                int c3, a3 = utf8_next(s + k, n - k, &c3);
                if (is_WS(c3) || is_L(c3) || is_N(c3)) break;
                k += a3;
            }
            if (k > j || j > 0) {
                while (k < n && (s[k] == '\r' || s[k] == '\n')) k++;
                if (k > 0) return k;
            }
        }
    }

    /* 5/6/7: whitespace forms */
    if (is_WS(cp)) {
        int j = 0, last_nl = -1;
        while (j < n) {
            int c3, a3 = utf8_next(s + j, n - j, &c3);
            if (!is_WS(c3)) break;
            if (c3 == '\r' || c3 == '\n') last_nl = j + a3;
            j += a3;
        }
        if (last_nl > 0) return last_nl;
        if (j < n && j > adv) return (j - 1 == 0) ? adv : j - 1;
        return j;
    }

    return adv; /* fallback: single codepoint */
}

/* ---- Tekken pretokenizer (Mistral-Nemo-2407 and later) ----
 *
 * Approximates llama.cpp's LLAMA_VOCAB_PRE_TYPE_TEKKEN regex
 * (src/llama-vocab.cpp), which is itself an ASCII-only-lookahead
 * approximation of the original tokenizer.json pattern:
 *
 *   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
 *   |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
 *   |\p{N}
 *   | ?[^\s\p{L}\p{N}]+[\r\n/]*
 *   |\s*[\r\n]+
 *   |\s+(?!\S)
 *   |\s+
 *
 * Differences from the QWEN_PRETOK_DEFAULT pattern above:
 *   - No contraction rule ('s/'t/'re/...) -- an apostrophe is just another
 *     "optional prefix" character and naturally attaches to a following
 *     letter run (e.g. "it's" -> "it" + "'s"), matching Tekken's regex
 *     (which has no contraction alternative at all).
 *   - The letter-run alternatives split on case: a run of "upper-like"
 *     codepoints (Lu/Lt/Lm/Lo/M) followed by a run of "lower-like"
 *     codepoints (Ll/Lm/Lo/M) is one pretoken; this is what makes
 *     "HelloWorld" split into "Hello"+"World" while "HELLO" and
 *     "HELLOworld" each stay whole. Letters with no case (CJK, Devanagari,
 *     Arabic, etc.) count as *both* upper-like and lower-like, so they
 *     behave the same as the QWEN_PRETOK_DEFAULT single-category letters.
 *   - The trailing run after a punctuation span also swallows '/', not
 *     just '\r'/'\n' (so "path/to/file" -> "path" + "/to" + "/file").
 *
 * Unicode classification here (is_upper/is_lower/is_mark) is a curated
 * approximation covering ASCII, Latin-1 Supplement, Latin Extended-A,
 * Greek and Cyrillic (the scripts where case actually matters for
 * Mistral's training data) plus the common combining-mark blocks, in the
 * same spirit as the is_L/is_N tables above. Any letter outside these
 * explicit upper/lower ranges (CJK, Hangul, Devanagari, Arabic, Hebrew,
 * Thai, etc., all already covered by is_L) is treated as caseless and
 * folds into both upper-like and lower-like, which is exactly the
 * behavior \p{Lo} gets in the original regex. */

static int is_tekken_upper(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return 1;
    if (cp >= 0xC0 && cp <= 0xD6) return 1;             /* Latin-1 uppercase */
    if (cp >= 0xD8 && cp <= 0xDE) return 1;
    if (cp >= 0x100 && cp <= 0x137 && (cp & 1) == 0) return 1; /* Latin Ext-A (approx) */
    if (cp >= 0x139 && cp <= 0x148 && (cp & 1) == 1) return 1;
    if (cp >= 0x14A && cp <= 0x177 && (cp & 1) == 0) return 1;
    if (cp == 0x178 || cp == 0x179 || cp == 0x17B || cp == 0x17D) return 1;
    if (cp == 0x386 || (cp >= 0x388 && cp <= 0x38F)) return 1; /* Greek accented caps */
    if (cp >= 0x391 && cp <= 0x3A1) return 1;            /* Greek Alpha..Rho */
    if (cp >= 0x3A3 && cp <= 0x3AB) return 1;            /* Greek Sigma..Omega/Iota/Upsilon diaeresis */
    if (cp >= 0x400 && cp <= 0x42F) return 1;            /* Cyrillic uppercase */
    return 0;
}

static int is_tekken_lower(uint32_t cp) {
    if (cp >= 'a' && cp <= 'z') return 1;
    if (cp >= 0xDF && cp <= 0xF6) return 1;              /* Latin-1 lowercase */
    if (cp >= 0xF8 && cp <= 0xFF) return 1;
    if (cp >= 0x100 && cp <= 0x137 && (cp & 1) == 1) return 1; /* Latin Ext-A (approx) */
    if (cp >= 0x139 && cp <= 0x148 && (cp & 1) == 0) return 1;
    if (cp >= 0x14A && cp <= 0x177 && (cp & 1) == 1) return 1;
    if (cp == 0x17A || cp == 0x17C || cp == 0x17E) return 1;
    if (cp >= 0x3AC && cp <= 0x3CE) return 1;            /* Greek lowercase (incl. accented/final sigma) */
    if (cp >= 0x430 && cp <= 0x45F) return 1;            /* Cyrillic lowercase */
    return 0;
}

static int is_tekken_mark(uint32_t cp) {
    if (cp >= 0x0300 && cp <= 0x036F) return 1;  /* combining diacritical marks */
    if (cp >= 0x0483 && cp <= 0x0489) return 1;  /* Cyrillic combining */
    if (cp >= 0x0591 && cp <= 0x05C7) return 1;  /* Hebrew points (approx) */
    if (cp >= 0x0610 && cp <= 0x061A) return 1;  /* Arabic marks */
    if (cp >= 0x064B && cp <= 0x065F) return 1;  /* Arabic diacritics */
    if (cp == 0x0670) return 1;
    if (cp >= 0x06D6 && cp <= 0x06ED) return 1;  /* Arabic marks (approx range) */
    if (cp >= 0x0900 && cp <= 0x0963) return 1;  /* Devanagari marks (approx range) */
    if (cp >= 0x1AB0 && cp <= 0x1AFF) return 1;
    if (cp >= 0x1DC0 && cp <= 0x1DFF) return 1;
    if (cp >= 0x20D0 && cp <= 0x20FF) return 1;
    if (cp >= 0xFE20 && cp <= 0xFE2F) return 1;
    return 0;
}

/* Lu|Lt|Lm|Lo|M : upper/titlecase, plus caseless letters and marks */
static int is_tekken_upper_like(uint32_t cp) {
    if (is_tekken_upper(cp) || is_tekken_mark(cp)) return 1;
    return is_L(cp) && !is_tekken_lower(cp); /* caseless letter */
}
/* Ll|Lm|Lo|M : lowercase, plus caseless letters and marks */
static int is_tekken_lower_like(uint32_t cp) {
    if (is_tekken_lower(cp) || is_tekken_mark(cp)) return 1;
    return is_L(cp) && !is_tekken_upper(cp); /* caseless letter */
}

static int pretok_next_tekken(const char *s, int n) {
    int cp, adv;
    adv = utf8_next(s, n, &cp);

    /* 1/2: optional non-[\r\n L N] prefix, then a case-aware letter run:
     *      upper-like* followed by lower-like+ (alt1), or upper-like+
     *      optionally followed by lower-like* (alt2). */
    {
        int start = -1; /* byte offset (from s) where the letter run itself begins;
                          * the overall match always starts at offset 0 (either the
                          * prefix char at 0, or the letter itself at 0). */

        if (cp != '\r' && cp != '\n' && !is_L(cp) && !is_N(cp)) {
            /* cp is a candidate optional-prefix character; only consume it
             * if a letter actually follows (regex prefix is greedy but the
             * overall alternative still requires a letter run). */
            if (adv < n) {
                int c2 = 0;
                (void)utf8_next(s + adv, n - adv, &c2);
                if (is_L(c2)) start = adv;
            }
        } else if (is_L(cp)) {
            start = 0;
        }

        if (start >= 0) {
            int j = start;
            while (j < n) {
                int c3, a3 = utf8_next(s + j, n - j, &c3);
                if (!is_tekken_upper_like(c3)) break;
                j += a3;
            }
            int upper_end = j;
            while (j < n) {
                int c3, a3 = utf8_next(s + j, n - j, &c3);
                if (!is_tekken_lower_like(c3)) break;
                j += a3;
            }
            int lower_end = j;

            if (lower_end > upper_end) return lower_end;   /* alt1 */
            if (upper_end > start) return upper_end;       /* alt2 */
            /* Shouldn't happen (is_L implies upper_like||lower_like), but
             * fall through defensively to the remaining alternatives. */
        }
    }

    /* 3: single number */
    if (is_N(cp)) return adv;

    /* 4: optional space + punct run + trailing \r\n/ */
    {
        int j = 0;
        if (cp == ' ' && adv < n) {
            int c2;
            (void)utf8_next(s + adv, n - adv, &c2);
            if (!is_WS(c2) && !is_L(c2) && !is_N(c2)) j = adv;
        }
        if (j > 0 || (!is_WS(cp) && !is_L(cp) && !is_N(cp))) {
            int k = j;
            while (k < n) {
                int c3, a3 = utf8_next(s + k, n - k, &c3);
                if (is_WS(c3) || is_L(c3) || is_N(c3)) break;
                k += a3;
            }
            if (k > j || j > 0) {
                while (k < n && (s[k] == '\r' || s[k] == '\n' || s[k] == '/')) k++;
                if (k > 0) return k;
            }
        }
    }

    /* 5/6/7: whitespace forms (identical semantics to the default
     * pretokenizer's \s*[\r\n]+ / \s+(?!\S) / \s+ handling). */
    if (is_WS(cp)) {
        int j = 0, last_nl = -1;
        while (j < n) {
            int c3, a3 = utf8_next(s + j, n - j, &c3);
            if (!is_WS(c3)) break;
            if (c3 == '\r' || c3 == '\n') last_nl = j + a3;
            j += a3;
        }
        if (last_nl > 0) return last_nl;
        if (j < n && j > adv) return (j - 1 == 0) ? adv : j - 1;
        return j;
    }

    return adv; /* fallback: single codepoint */
}

/* ---- BPE encode one pretoken ---- */
static int bpe_piece(qwen_enc_t *enc, const char *s, int n, int *out, int cap) {
    char buf[2048];
    int st[512], ln[512], ns = 0, w = 0;

    for (int i = 0; i < n && ns < 512; i++) {
        int cp = g_fwd[(uint8_t)s[i]];
        st[ns] = w;
        if (cp < 0x80) buf[w++] = (char)cp;
        else { buf[w++] = (char)(0xC0 | (cp >> 6)); buf[w++] = (char)(0x80 | (cp & 0x3F)); }
        ln[ns] = w - st[ns];
        ns++;
    }

    for (;;) {
        int best = -1, bi = -1;
        for (int i = 0; i + 1 < ns; i++) {
            int r = merge_rank(enc->merges, enc->mrg_tab,
                               buf + st[i], ln[i], buf + st[i + 1], ln[i + 1]);
            if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; }
        }
        if (bi < 0) break;
        ln[bi] += ln[bi + 1];
        memmove(&st[bi + 1], &st[bi + 2], (ns - bi - 2) * sizeof(int));
        memmove(&ln[bi + 1], &ln[bi + 2], (ns - bi - 2) * sizeof(int));
        ns--;
    }

    int m2 = 0;
    for (int i = 0; i < ns && m2 < cap; i++) {
        int id = tok_lookup(enc->vocab, enc->vocab_len, enc->tok_tab, enc->vocab_size,
                            buf + st[i], ln[i]);
        if (id >= 0) { out[m2++] = id; }
        else {
            for (int j = 0; j < ln[i] && m2 < cap; j++) {
                int id2 = tok_lookup(enc->vocab, enc->vocab_len, enc->tok_tab, enc->vocab_size,
                                     buf + st[i] + j, 1);
                if (id2 >= 0) out[m2++] = id2;
            }
        }
    }
    return m2;
}

/* ---- Public API ---- */

/* Check if a model should use the native GPT-2 BPE tokenizer (this file). */
int qwen_tokenize_should_use(const model_t *m) {
    /* Use the native BPE tokenizer for Qwen3/Qwen3.5 architectures, GPT-2
     * models, and Mistral Tekken models (tokenizer.ggml.pre == "tekken",
     * e.g. Mistral-Nemo-2407). All of these ship a GPT-2-style byte-level
     * BPE vocab/merges in GGUF, just with different pretokenizer regexes
     * (see qwen_enc_t.pretok_type / pretok_next vs pretok_next_tekken).
     * Llama SPM and other architectures should use the old tokenizer.
     * Safetensors Qwen models also use this tokenizer (from_safetensors path). */
    return m->config.is_qwen || m->config.is_gpt2 || m->config.is_tekken || m->from_safetensors;
}

/* Initialize the Qwen tokenizer from model data */
int qwen_tokenize_init(qwen_enc_t *enc, const model_t *m) {
    if (!g_maps_ready) build_maps();

    enc->pretok_type = m->config.is_tekken ? QWEN_PRETOK_TEKKEN : QWEN_PRETOK_DEFAULT;

    /* Guard: Qwen tokenizer requires tok_tokens_data from GGUF */
    if (!m->tok_tokens_data || !m->tok_n_tokens) {
        fprintf(stderr, "ERROR: Qwen tokenizer requested but tok_tokens_data is missing\n");
        return -1;
    }

    /* Parse vocab strings from tok_tokens_data.
     * Two formats:
     * - Safetensors: [u64 len][bytes]... (native endian, written by safetensors.c)
     * - GGUF:         [u32_le len][bytes]... (raw GGUF string array, little-endian)
     * GGUF strings are not necessarily aligned; read byte-by-byte. */
    {
        const uint8_t *pp = (const uint8_t *)m->tok_tokens_data;
        uint64_t nn = m->tok_n_tokens;
        enc->vocab_size = (int)nn;
        enc->vocab = (const char **)malloc(nn * sizeof(char*));
        if (!enc->vocab) return -1;
        enc->vocab_len = (int *)malloc(nn * sizeof(int));
        if (!enc->vocab_len) return -1;
        for (uint64_t i = 0; i < nn; i++) {
            uint32_t sl;
            /* Both Safetensors and GGUF v3 use [u64 LE len][bytes] format.
               GGUF v2 used [u32 LE len] but all current files are v3.
               Read as LE u64 to work on both BE and LE architectures. */
            uint32_t sl32 = (uint32_t)pp[0] | ((uint32_t)pp[1] << 8) |
                            ((uint32_t)pp[2] << 16) | ((uint32_t)pp[3] << 24);
            uint32_t sl32hi = (uint32_t)pp[4] | ((uint32_t)pp[5] << 8) |
                              ((uint32_t)pp[6] << 16) | ((uint32_t)pp[7] << 24);
            uint64_t sl64 = (uint64_t)sl32 | ((uint64_t)sl32hi << 32);
            sl = (uint32_t)sl64;
            pp += 8;
            enc->vocab_len[i] = (int)sl;
            enc->vocab[i] = (const char *)pp;
            pp += sl;
        }
    }
    enc->bos_id = (int)m->tok_bos_id;
    enc->eos_id = m->tok_eos_id; /* fallback: same as bos if GGUF missing it */

    /* Build token lookup hash table */
    {
        uint32_t tc = 1u << QWEN_TOKHASH_BITS;
        enc->tok_tab = (int *)malloc(tc * sizeof(int));
        if (!enc->tok_tab) return -1;
        memset(enc->tok_tab, -1, tc * sizeof(int));
        uint32_t mask = tc - 1;
        for (int id = 0; id < enc->vocab_size; id++) {
            uint32_t h = fnv(enc->vocab[id], enc->vocab_len[id], 0) & mask;
            while (enc->tok_tab[h] >= 0) h = (h + 1) & mask;
            enc->tok_tab[h] = id;
        }
    }

    /* Parse merges from GGUF metadata */
    if (!m->tok_merges_data) {
        fprintf(stderr, "ERROR: Qwen tokenizer requested but tok_merges_data is missing\n");
        return -1;
    }
    const uint8_t *p = (const uint8_t *)m->tok_merges_data;
    uint64_t n_merges = m->tok_n_merges;
    enc->n_merges = (int)n_merges;
    enc->merges = (qwen_merge_t *)malloc(sizeof(qwen_merge_t) * (size_t)enc->n_merges);
    if (!enc->merges) return -1;

    {
        uint32_t mc = 1u << QWEN_MRGHASH_BITS;
        enc->mrg_tab = (int *)malloc(mc * sizeof(int));
        if (!enc->mrg_tab) return -1;
        memset(enc->mrg_tab, -1, mc * sizeof(int));
        uint32_t mm = mc - 1;

        for (int i = 0; i < enc->n_merges; i++) {
            /* GGUF v3: [u64 LE len][bytes]. Same format as vocab strings. */
            uint32_t l32 = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
            uint32_t l32hi = (uint32_t)p[4] | ((uint32_t)p[5] << 8) |
                             ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
            uint64_t l64 = (uint64_t)l32 | ((uint64_t)l32hi << 32);
            uint32_t l = (uint32_t)l64;
            p += 8;
            const char *s = (const char *)p; p += l;
            int sp = -1;
            for (int j = 0; j < (int)l; j++) { if (s[j] == ' ') { sp = j; break; } }
            enc->merges[i].lp = s;
            enc->merges[i].ll = sp;
            enc->merges[i].rp = s + sp + 1;
            enc->merges[i].rl = (int)l - sp - 1;
            uint32_t h = (fnv(s, sp, 0) * 31u + fnv(s + sp + 1, (int)l - sp - 1, 7)) & mm;
            while (enc->mrg_tab[h] >= 0) h = (h + 1) & mm;
            enc->mrg_tab[h] = i;
        }
    }

    /* Parse special tokens from token_type (3=control, 4=user_defined) */
    enc->n_spec = 0;
    enc->spec = (int *)malloc(sizeof(int) * 512);
    if (m->tok_token_type_data) {
        const int32_t *ty = (const int32_t *)m->tok_token_type_data;
        uint64_t n = m->tok_n_token_type;
        if ((int)n > enc->vocab_size) n = (uint64_t)enc->vocab_size;
        for (int id = 0; id < (int)n && enc->n_spec < 512; id++) {
            if (ty[id] == 3 || ty[id] == 4) enc->spec[enc->n_spec++] = id;
        }
    }
    /* For GPT-2 models, explicitly add BOS/EOS as special tokens */
    if (m->config.is_gpt2 && enc->n_spec < 511) {
        if (enc->bos_id >= 0 && enc->bos_id < enc->vocab_size) {
            enc->spec[enc->n_spec++] = enc->bos_id;
        }
        if (enc->eos_id != enc->bos_id && enc->eos_id >= 0 && enc->eos_id < enc->vocab_size) {
            enc->spec[enc->n_spec++] = enc->eos_id;
        }
    }

    return 0;
}

void qwen_tokenize_free(qwen_enc_t *enc) {
    free(enc->tok_tab); enc->tok_tab = NULL;
    free(enc->mrg_tab); enc->mrg_tab = NULL;
    free(enc->merges); enc->merges = NULL;
    free(enc->spec); enc->spec = NULL;
    free(enc->vocab); enc->vocab = NULL;
    free(enc->vocab_len); enc->vocab_len = NULL;
}

/* Encode text to token IDs. Returns number of tokens produced. */
int qwen_tokenize_encode(qwen_enc_t *enc, const char *text, int *out, int cap) {
    int n = (int)strlen(text);
    int pos = 0, m2 = 0;

    while (pos < n && m2 < cap) {
        /* Longest special-token match at pos */
        int sid = -1, slen = 0;
        for (int i = 0; i < enc->n_spec; i++) {
            int id = enc->spec[i];
            int l = enc->vocab_len[id];
            if (l > slen && pos + l <= n && memcmp(text + pos, enc->vocab[id], l) == 0) {
                sid = id; slen = l;
            }
        }
        if (sid >= 0) { out[m2++] = sid; pos += slen; continue; }

        /* Ordinary span up to the next special token */
        int end = n;
        for (int p2 = pos + 1; p2 < n; p2++) {
            for (int i = 0; i < enc->n_spec; i++) {
                int id = enc->spec[i];
                int l = enc->vocab_len[id];
                if (p2 + l <= n && memcmp(text + p2, enc->vocab[id], l) == 0) {
                    end = p2; goto found;
                }
            }
        }
        found:;

        int q = pos;
        while (q < end && m2 < cap) {
            int pl = (enc->pretok_type == QWEN_PRETOK_TEKKEN)
                      ? pretok_next_tekken(text + q, end - q)
                      : pretok_next(text + q, end - q);
            if (pl <= 0) pl = 1;
            m2 += bpe_piece(enc, text + q, pl, out + m2, cap - m2);
            q += pl;
        }
        pos = end;
    }
    return m2;
}

/* Decode one token ID to UTF-8 text. Returns bytes written.
 * By default, BOS and EOS are hidden (return 0).
 * Other special tokens (tool_call tags, etc.) are printed as their raw vocab bytes.
 * If add_special is 1, BOS/EOS are also printed. */
int qwen_tokenize_decode2(const qwen_enc_t *enc, int id, char *buf, int cap, int add_special) {
    if (id < 0 || id >= enc->vocab_size) return 0;

    /* Check if this is a special token */
    int is_special = 0;
    for (int si = 0; si < enc->n_spec; si++) {
        if (enc->spec[si] == id) { is_special = 1; break; }
    }

    const char *s = enc->vocab[id];
    int sl = enc->vocab_len[id];

    /* For BOS/EOS: hide unless add_special is set */
    if (is_special && id == enc->bos_id && !add_special) { buf[0] = '\0'; return 0; }
    if (is_special && id == enc->eos_id && !add_special) { buf[0] = '\0'; return 0; }

    /* For all other special tokens (tool_call, etc.), emit the raw vocab bytes */
    int w = 0;
    for (int i = 0; i < sl && w < cap - 4; ) {
        int cp;
        i += utf8_next(s + i, sl - i, &cp);
        if (cp < 512 && g_rev[cp] >= 0) {
            buf[w++] = (char)g_rev[cp];
        } else {
            /* Control bytes and special codepoints: emit raw UTF-8 */
            if (cp < 0x80) buf[w++] = (char)cp;
            else if (cp < 0x800) {
                buf[w++] = (char)(0xC0 | (cp >> 6));
                buf[w++] = (char)(0x80 | (cp & 0x3F));
            } else {
                buf[w++] = (char)(0xE0 | (cp >> 12));
                buf[w++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                buf[w++] = (char)(0x80 | (cp & 0x3F));
            }
        }
    }
    buf[w] = '\0';
    return w;
}

/* Compatibility wrapper: add_special=0 */
int qwen_tokenize_decode(const qwen_enc_t *enc, int id, char *buf, int cap) {
    return qwen_tokenize_decode2(enc, id, buf, cap, 0);
}

