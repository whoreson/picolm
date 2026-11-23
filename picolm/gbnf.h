#ifndef GBNF_H
#define GBNF_H

/* GBNF grammar-constrained sampling.
 *
 * This is a C99 port of llama.cpp's llama-grammar (src/llama-grammar.cpp,
 * MIT). The algorithm is unchanged in substance:
 *
 *   parse  grammar text  ->  flat rule table of "elements"
 *   match  a token's UTF-8 text against a set of pushdown stacks
 *   mask   logits so only tokens that can still lead to a valid
 *          continuation survive
 *
 * The flat representation is the trick that makes this simple: every rule is
 * a sequence of elements, alternatives are separated by GRETYPE_ALT, and a
 * rule ends with GRETYPE_END. So a rule reference is just "push a pointer into
 * that rule's table", and a character class [a-z] / [^abc] / "." is a run of
 * elements treated as one unit. No AST, no interpretation.
 *
 * Why a separate engine and not the old grammar.c: the existing
 * GRAMMAR_JSON mode only tracks brace/bracket depth. It cannot express
 * "no em dash", "this field is one of three words", or anything else a real
 * BNF can. GBNF is the format llama.cpp users and GGUF metadata already use,
 * so this replaces the heuristic when a grammar is supplied and leaves
 * --json alone.
 */

#include <stdint.h>

/* Grammar element type. Values match llama.cpp's llama_gretype so that a
 * grammar dumped from llama.cpp tooling reads the same here. */
typedef enum {
    GRETYPE_END            = 0,  /* end of rule / alternate              */
    GRETYPE_ALT            = 1,  /* start of alternate definition        */
    GRETYPE_RULE_REF       = 2,  /* non-terminal: value = rule id        */
    GRETYPE_CHAR           = 3,  /* terminal: value = code point         */
    GRETYPE_CHAR_NOT       = 4,  /* inverse char class [^...]            */
    GRETYPE_CHAR_RNG_UPPER = 5,  /* upper bound of a range [a-z]         */
    GRETYPE_CHAR_ALT       = 6,  /* another char in the same class       */
    GRETYPE_CHAR_ANY       = 7,  /* .                                    */
} gretype_t;

typedef struct {
    uint8_t  type;
    uint32_t value;             /* code point, or rule id for GRETYPE_RULE_REF */
    /* For the FIRST element of a character class only: a precomputed
     * membership bitmap for code points 0..255, plus a flag for "this class
     * also contains code points above 255" (checked the slow way).
     *
     * Without this, testing one character walked the whole class element by
     * element: nosthit.gbnf's "punctuation" is 30+ elements, so every
     * character cost ~100 comparisons and match_char was 38% of the profile.
     * The class is a set of code points, so a bitmap is the natural form.
     * See gbnf_build_class_bits() in gbnf.c. */
    uint8_t  is_class_head;     /* 1 on the first element of a class */
    uint8_t  has_high;          /* class has code points > 255 */
    uint16_t class_len;         /* elements in the class, head included */
    uint8_t  bits[32];          /* 256 bits for code points 0..255 */
} gbnf_element;

typedef struct {
    gbnf_element *elems;
    int           n;
} gbnf_rule;

/* A partially-decoded UTF-8 sequence, so a token that ends mid-character can
 * still be judged against a character class. */
typedef struct {
    uint32_t value;    /* bits accumulated so far, unshifted */
    int      n_remain; /* bytes still expected; -1 = invalid sequence */
} gbnf_utf8;

typedef struct {
    uint32_t index;    /* token id */
    gbnf_utf8 partial; /* partial UTF-8 state after this token */
} gbnf_candidate;

/* Pushdown stacks, arena-backed.
 *
 * The matcher is run once per vocabulary entry per generated token (~150k x
 * per token), so per-stack malloc/free is not affordable: the first version
 * did exactly that and cost 1.6 s/token while pinning one core. Everything
 * here is a fixed-size array inside one struct, allocated once per thread.
 *
 * Capacities are generous relative to real grammars (a stack is one pointer
 * per nested rule reference plus per pending continuation) and overflow is
 * handled by DROPPING the excess stack rather than corrupting memory: for
 * masking purposes a dropped stack can only make us reject a token, which is
 * visible as truncated output, not as a crash or wrong accept. */
#define GBNF_MAX_STACK_DEPTH  128
#define GBNF_MAX_STACKS        32

typedef struct {
    const gbnf_element *ptrs[GBNF_MAX_STACK_DEPTH];
    int                 n;
} gbnf_stack;

typedef struct {
    gbnf_stack v[GBNF_MAX_STACKS];
    int        n;
    int        overflow;   /* set when a stack was dropped */
} gbnf_stacks;

typedef struct {
    gbnf_rule *rules;
    int        n_rules;

    /* symbol name -> rule id, for diagnostics and for looking up "root" */
    char     **names;
    int       *name_ids;
    int        n_names;

    /* Per-token metadata, computed once: the token's raw UTF-8 BYTES, with
     * the byte length. Bytes, not code points, because a byte-level BPE
     * vocabulary can end a token in the middle of a multi-byte character (a
     * lone 0xC3 completed by the next token); only bytes let us track that. */
    uint8_t **token_bytes;
    int      *token_len;
    int        vocab_size;
    int        eos_id;

    /* Live parse state */
    gbnf_stacks stacks;
    gbnf_utf8   partial_utf8;

    int started;   /* fed at least one token */

    /* Per-token verdict cache for the CURRENT state.
     *
     * Judging a token costs a stack walk; doing that for ~150k vocab entries on
     * every sampling step was 560 ms/token and pinned one core (the whole
     * pipeline is otherwise ~24 cores). The verdicts only change when the
     * parse state changes, i.e. once per generated token, so they are computed
     * once per state (in parallel) and reused by gbnf_apply() until the next
     * gbnf_accept_token(). */
    uint8_t *accept_cache;   /* [vocab_size]: 1 = allowed in this state */
    int      cache_valid;

    /* Cross-token verdict cache.
     *
     * The verdict set depends only on the parse state (the stack set + the
     * partial UTF-8). Loop rules like en-char+ revisit the same handful of
     * states over and over: 30 generated tokens of a paragraph cycle through
     * FOUR distinct states. Recomputing 248k token verdicts for each of them
     * cost 90 ms per token against 14 ms of model compute.
     *
     * So remember the last few states' verdict arrays keyed by a signature of
     * the state. A hit copies the saved array (a 248 KB memcpy) instead of
     * redoing the walks. Sized in bytes, allocated lazily. */
    #define GBNF_STATE_CACHE_SLOTS 8
    uint64_t state_sig[GBNF_STATE_CACHE_SLOTS];
    uint8_t *state_cache[GBNF_STATE_CACHE_SLOTS];
    int      state_cache_n;
    int      state_cache_next;

    /* Fast prefilter: for the CURRENT state, first_ok[b] = 1 if a token whose
     * first byte is b can possibly be accepted. Almost every vocabulary entry
     * dies on its first byte for a constraining grammar, so this rejects them
     * without touching the stack machinery at all.
     * first_partial[b] = 1 if b is a lead byte of a multi-byte sequence that
     * could still be completed (needs the full check). */
    uint8_t  first_ok[256];
    uint8_t  first_partial[256];
} gbnf_t;

/* Parse `text` as GBNF. `root_name` is the start rule (NULL = "root").
 * Returns 0 on success, -1 on a parse error (message on stderr). */
int gbnf_parse(gbnf_t *g, const char *text, const char *root_name);

/* Precompute per-token code points. vocab[i] is the token string. */
void gbnf_set_vocab(gbnf_t *g, char **vocab, int vocab_size, int eos_id);

/* Starting stack set, after parsing. */
int gbnf_start(gbnf_t *g);

/* Mask logits: tokens that cannot continue any live stack become -1e30.
 * Must be called before sampling each token. Cheap: uses the verdict cache
 * built by gbnf_accept_token()/gbnf_start(). */
void gbnf_apply(const gbnf_t *g, float *logits, int vocab_size);

/* Rebuild the per-token verdict cache for the current state, using `n_threads`
 * threads (0 = auto). Called automatically by gbnf_start/gbnf_accept_token;
 * exposed for tests and for callers that want to control the count. */
void gbnf_refresh_cache(gbnf_t *g, int n_threads);

/* Advance the parse state with the token that was committed. */
void gbnf_accept_token(gbnf_t *g, int token);

/* 1 when every live stack has consumed the whole grammar (nothing left to
 * match), i.e. EOS is the only sensible thing to emit. */
int gbnf_is_complete(const gbnf_t *g);

void gbnf_free(gbnf_t *g);

#endif /* GBNF_H */
