/* GBNF grammar engine. C99 port of llama.cpp's llama-grammar.cpp (MIT).
 *
 * See gbnf.h for the data model. Two halves:
 *
 *   PARSER   text -> flat rule table (+ repetition rewriting, since GBNF has
 *            S* S+ S? S{n,m} but the matcher only understands "rule ref",
 *            "alt" and character classes)
 *   MATCHER  a set of pushdown stacks, advanced one code point at a time
 *
 * The hot path is gbnf_apply(), called per generated token: it must decide,
 * for every vocabulary entry, whether that token can continue the grammar.
 * That is ~250k entries for a large Qwen vocab, so:
 *   - tokens are stored as raw bytes (a byte-level BPE token may end in the
 *     middle of a multi-byte character, completed by the next token);
 *   - character classes are precomputed into 256-bit bitmaps;
 *   - the verdict array is cached per parse state, because loop rules cycle
 *     through a handful of states (a whole paragraph used 4).
 */

#include "gbnf.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GBNF_NEG_INF (-1e30f)
#define GBNF_MAX_REPETITIONS 1000
#define GBNF_MAX_DEPTH 1000

/* ------------------------------------------------------------------ */
/* Small helpers                                                      */
/* ------------------------------------------------------------------ */

static void gbnf_build_class_bits(gbnf_t *g);

static int prof_on(void) {
    static int on = -1;
    if (on < 0) on = getenv("PICOLM_GBNF_PROF") ? 1 : 0;
    return on;
}

/* Profiling only. Uses the project's portable get_time_ms() (Win32
 * QueryPerformanceCounter / DOS clock() / POSIX gettimeofday) instead of
 * clock_gettime(CLOCK_MONOTONIC), which MSVC has no declaration for and which
 * does not exist at all on DOS or on older Unix. */
extern double get_time_ms(void);
static double now_ms(void) { return get_time_ms(); }

/* Length and value bits of a UTF-8 lead byte. *len is always >= 1 so callers
 * make progress even on garbage. */
static uint32_t lead_byte(uint8_t b, int *len) {
    if (b < 0x80)          { *len = 1; return b; }
    if ((b & 0xE0) == 0xC0) { *len = 2; return b & 0x1Fu; }
    if ((b & 0xF0) == 0xE0) { *len = 3; return b & 0x0Fu; }
    if ((b & 0xF8) == 0xF0) { *len = 4; return b & 0x07u; }
    *len = 1; return b;
}

/* ------------------------------------------------------------------ */
/* Pushdown stack sets                                                */
/* ------------------------------------------------------------------ */

/* Add a stack unless an identical one is already present. */
static void stacks_add(gbnf_stacks *ss, const gbnf_element *const *ptrs, int n) {
    for (int i = 0; i < ss->n; i++) {
        if (ss->v[i].n != n) continue;
        if (n == 0 || memcmp(ss->v[i].ptrs, ptrs, (size_t)n * sizeof(*ptrs)) == 0)
            return;
    }
    if (ss->n >= GBNF_MAX_STACKS) { ss->overflow = 1; return; }
    if (n > GBNF_MAX_STACK_DEPTH) { ss->overflow = 1; return; }
    gbnf_stack *d = &ss->v[ss->n];
    if (n) memcpy(d->ptrs, ptrs, (size_t)n * sizeof(*ptrs));
    d->n = n;
    ss->n++;
}

static void stacks_clear(gbnf_stacks *ss) { ss->n = 0; ss->overflow = 0; }

/* ------------------------------------------------------------------ */
/* Parser                                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    gbnf_t *g;
    const char *err;
} parser_t;

static int rule_add_elem(gbnf_rule *r, uint8_t type, uint32_t value) {
    gbnf_element *n = (gbnf_element *)realloc(r->elems,
        (size_t)(r->n + 1) * sizeof(gbnf_element));
    if (!n) return -1;
    r->elems = n;
    memset(&r->elems[r->n], 0, sizeof(gbnf_element));
    r->elems[r->n].type = type;
    r->elems[r->n].value = value;
    r->n++;
    return 0;
}

static uint32_t gbnf_symbol_id(gbnf_t *g, const char *name, int len) {
    for (int i = 0; i < g->n_names; i++) {
        if ((int)strlen(g->names[i]) == len && memcmp(g->names[i], name, (size_t)len) == 0)
            return (uint32_t)g->name_ids[i];
    }
    int id = g->n_rules;   /* dense, first-seen order, like llama.cpp */
    char **nn = (char **)realloc(g->names, (size_t)(g->n_names + 1) * sizeof(char *));
    if (!nn) return 0;
    g->names = nn;
    int *ni = (int *)realloc(g->name_ids, (size_t)(g->n_names + 1) * sizeof(int));
    if (!ni) return 0;
    g->name_ids = ni;
    char *nm = (char *)malloc((size_t)len + 1);
    if (!nm) return 0;
    memcpy(nm, name, (size_t)len);
    nm[len] = '\0';
    g->names[g->n_names] = nm;
    g->name_ids[g->n_names] = id;
    g->n_names++;

    if (id >= g->n_rules) {
        gbnf_rule *nr = (gbnf_rule *)realloc(g->rules, (size_t)(id + 1) * sizeof(gbnf_rule));
        if (!nr) return 0;
        g->rules = nr;
        for (int i = g->n_rules; i <= id; i++) {
            g->rules[i].elems = NULL;
            g->rules[i].n = 0;
        }
        g->n_rules = id + 1;
    }
    return (uint32_t)id;
}

static uint32_t gbnf_generate_symbol_id(gbnf_t *g, const char *base) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s_%d", base, g->n_rules);
    return gbnf_symbol_id(g, buf, (int)strlen(buf));
}

static const char *parse_space(const char *src, int newline_ok) {
    const char *p = src;
    while (*p == ' ' || *p == '\t' || *p == '#' || (newline_ok && (*p == '\r' || *p == '\n'))) {
        if (*p == '#') { while (*p && *p != '\r' && *p != '\n') p++; }
        else p++;
    }
    return p;
}

static int is_word_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '-' || c == '_';
}

static const char *parse_name(const char *src) {
    const char *p = src;
    while (is_word_char(*p)) p++;
    return (p == src) ? NULL : p;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static const char *parse_hex(const char *src, int size, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < size; i++) {
        int h = hexval(src[i]);
        if (h < 0) { *out = 0; return NULL; }
        v = (v << 4) | (uint32_t)h;
    }
    *out = v;
    return src + size;
}

static const char *parse_char(const char *src, uint32_t *out) {
    if (*src == '\\') {
        switch (src[1]) {
            case 'x': return parse_hex(src + 2, 2, out);
            case 'u': return parse_hex(src + 2, 4, out);
            case 'U': return parse_hex(src + 2, 8, out);
            case 't': *out = '\t'; return src + 2;
            case 'r': *out = '\r'; return src + 2;
            case 'n': *out = '\n'; return src + 2;
            case '\\': case '"': case '[': case ']': case '-':
                *out = (uint32_t)(unsigned char)src[1];
                return src + 2;
            default: return NULL;
        }
    }
    if (!*src) return NULL;
    uint8_t b = (uint8_t)*src;
    int len;
    uint32_t c = lead_byte(b, &len);
    if (len == 1) { *out = c; return src + 1; }
    for (int i = 1; i < len; i++) {
        if (((uint8_t)src[i] & 0xC0) != 0x80) { *out = b; return src + 1; }
        c = (c << 6) | (uint32_t)((uint8_t)src[i] & 0x3F);
    }
    *out = c;
    return src + len;
}

static const char *parse_sequence(parser_t *ps, const char *src,
                                  const char *rule_name, gbnf_rule *rule,
                                  int is_nested, uint32_t depth);

/* S{m,n} and the * + ? shorthands, expanded into rules exactly as llama.cpp
 * does:
 *   S{m,n} => S...S (m times) S'(n-m),  S'(k) => S S'(k-1) |,  S'(1) => S |
 *   S{m,}  => S...S (m times) S',       S'    => S S' |
 *   S*     => S{0,}   S+ => S{1,}   S? => S{0,1}
 */
static int handle_repetitions(parser_t *ps, const char *rule_name,
                              gbnf_rule *rule, size_t last_sym_start,
                              uint64_t min_times, uint64_t max_times) {
    gbnf_t *g = ps->g;
    int no_max = (max_times == UINT64_MAX);

    if (last_sym_start == (size_t)rule->n) {
        ps->err = "expecting preceding item before *, +, ? or {}";
        return -1;
    }

    gbnf_rule prev;
    prev.n = rule->n - (int)last_sym_start;
    prev.elems = NULL;
    if (prev.n > 0) {
        prev.elems = (gbnf_element *)malloc((size_t)prev.n * sizeof(gbnf_element));
        if (!prev.elems) { ps->err = "out of memory"; return -1; }
        memcpy(prev.elems, rule->elems + last_sym_start, (size_t)prev.n * sizeof(gbnf_element));
    }

    if (min_times == 0) rule->n = (int)last_sym_start;
    else for (uint64_t i = 1; i < min_times; i++)
        for (int e = 0; e < prev.n; e++)
            rule_add_elem(rule, prev.elems[e].type, prev.elems[e].value);

    uint64_t n_opt = no_max ? 1 : (max_times - min_times);
    if (n_opt > GBNF_MAX_REPETITIONS) n_opt = GBNF_MAX_REPETITIONS;

    uint32_t last_rec = 0;
    int have_rec = 0;
    for (uint64_t i = 0; i < n_opt; i++) {
        gbnf_rule rec;
        rec.n = 0;
        rec.elems = NULL;
        for (int e = 0; e < prev.n; e++)
            rule_add_elem(&rec, prev.elems[e].type, prev.elems[e].value);

        uint32_t rec_id = gbnf_generate_symbol_id(g, rule_name);
        if (i > 0 || no_max)
            rule_add_elem(&rec, GRETYPE_RULE_REF, no_max ? rec_id : last_rec);
        rule_add_elem(&rec, GRETYPE_ALT, 0);
        rule_add_elem(&rec, GRETYPE_END, 0);

        gbnf_rule *dst = &g->rules[rec_id];
        free(dst->elems);
        dst->elems = rec.elems;
        dst->n = rec.n;

        last_rec = rec_id;
        have_rec = 1;
    }
    if (have_rec && n_opt > 0) rule_add_elem(rule, GRETYPE_RULE_REF, last_rec);

    free(prev.elems);
    return 0;
}

static const char *parse_alternates(parser_t *ps, const char *src,
                                    const char *rule_name, uint32_t rule_id,
                                    int is_nested, uint32_t depth) {
    gbnf_rule rule;
    rule.elems = NULL;
    rule.n = 0;

    const char *pos = parse_sequence(ps, src, rule_name, &rule, is_nested, depth);
    while (pos && *pos == '|') {
        rule_add_elem(&rule, GRETYPE_ALT, 0);
        pos = parse_space(pos + 1, 1);
        pos = parse_sequence(ps, pos, rule_name, &rule, is_nested, depth);
    }
    if (!pos) { free(rule.elems); return NULL; }
    rule_add_elem(&rule, GRETYPE_END, 0);

    gbnf_rule *dst = &ps->g->rules[rule_id];
    free(dst->elems);
    dst->elems = rule.elems;
    dst->n = rule.n;
    return pos;
}

static const char *parse_sequence(parser_t *ps, const char *src,
                                  const char *rule_name, gbnf_rule *rule,
                                  int is_nested, uint32_t depth) {
    size_t last_sym_start = (size_t)rule->n;
    const char *pos = src;

    while (*pos) {
        if (*pos == '"') {                                  /* literal string */
            pos++;
            last_sym_start = (size_t)rule->n;
            while (*pos != '"') {
                if (!*pos) { ps->err = "unterminated string"; return NULL; }
                uint32_t c;
                const char *np = parse_char(pos, &c);
                if (!np) { ps->err = "bad escape in string"; return NULL; }
                pos = np;
                rule_add_elem(rule, GRETYPE_CHAR, c);
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '[') {                           /* character class */
            pos++;
            uint8_t start_type = GRETYPE_CHAR;
            if (*pos == '^') { pos++; start_type = GRETYPE_CHAR_NOT; }
            last_sym_start = (size_t)rule->n;
            while (*pos != ']') {
                if (!*pos) { ps->err = "unterminated character class"; return NULL; }
                uint32_t c;
                const char *np = parse_char(pos, &c);
                if (!np) { ps->err = "bad escape in character class"; return NULL; }
                pos = np;
                uint8_t type = (last_sym_start < (size_t)rule->n)
                             ? GRETYPE_CHAR_ALT : start_type;
                rule_add_elem(rule, type, c);
                if (pos[0] == '-' && pos[1] != ']') {
                    uint32_t hi;
                    const char *hp = parse_char(pos + 1, &hi);
                    if (!hp) { ps->err = "bad range upper bound"; return NULL; }
                    pos = hp;
                    rule_add_elem(rule, GRETYPE_CHAR_RNG_UPPER, hi);
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (is_word_char(*pos)) {                    /* rule reference */
            const char *name_end = parse_name(pos);
            if (!name_end) { ps->err = "bad rule name"; return NULL; }
            uint32_t ref = gbnf_symbol_id(ps->g, pos, (int)(name_end - pos));
            pos = parse_space(name_end, is_nested);
            last_sym_start = (size_t)rule->n;
            rule_add_elem(rule, GRETYPE_RULE_REF, ref);
        } else if (*pos == '(') {                           /* group */
            if (depth >= GBNF_MAX_DEPTH) { ps->err = "nesting too deep"; return NULL; }
            pos = parse_space(pos + 1, 1);
            uint32_t sub_id = gbnf_generate_symbol_id(ps->g, rule_name);
            pos = parse_alternates(ps, pos, rule_name, sub_id, 1, depth + 1);
            if (!pos) return NULL;
            last_sym_start = (size_t)rule->n;
            rule_add_elem(rule, GRETYPE_RULE_REF, sub_id);
            if (*pos != ')') { ps->err = "expecting ')'"; return NULL; }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '.') {                           /* any char */
            last_sym_start = (size_t)rule->n;
            rule_add_elem(rule, GRETYPE_CHAR_ANY, 0);
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '*') {
            pos = parse_space(pos + 1, is_nested);
            if (handle_repetitions(ps, rule_name, rule, last_sym_start, 0, UINT64_MAX)) return NULL;
        } else if (*pos == '+') {
            pos = parse_space(pos + 1, is_nested);
            if (handle_repetitions(ps, rule_name, rule, last_sym_start, 1, UINT64_MAX)) return NULL;
        } else if (*pos == '?') {
            pos = parse_space(pos + 1, is_nested);
            if (handle_repetitions(ps, rule_name, rule, last_sym_start, 0, 1)) return NULL;
        } else if (*pos == '{') {
            pos = parse_space(pos + 1, is_nested);
            if (!(*pos >= '0' && *pos <= '9')) { ps->err = "expecting an integer in {}"; return NULL; }
            uint64_t mn = 0;
            while (*pos >= '0' && *pos <= '9') { mn = mn * 10 + (uint64_t)(*pos - '0'); pos++; }
            pos = parse_space(pos, is_nested);
            uint64_t mx = UINT64_MAX;
            if (*pos == '}') {
                mx = mn;
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == ',') {
                pos = parse_space(pos + 1, is_nested);
                if (*pos >= '0' && *pos <= '9') {
                    mx = 0;
                    while (*pos >= '0' && *pos <= '9') { mx = mx * 10 + (uint64_t)(*pos - '0'); pos++; }
                    pos = parse_space(pos, is_nested);
                }
                if (*pos != '}') { ps->err = "expecting '}'"; return NULL; }
                pos = parse_space(pos + 1, is_nested);
            } else {
                ps->err = "expecting ','";
                return NULL;
            }
            if (mn > GBNF_MAX_REPETITIONS) { ps->err = "repetition count too large"; return NULL; }
            if (mx != UINT64_MAX && mx > GBNF_MAX_REPETITIONS) mx = UINT64_MAX;
            if (handle_repetitions(ps, rule_name, rule, last_sym_start, mn, mx)) return NULL;
        } else {
            break;
        }
    }
    return pos;
}

static const char *parse_rule(parser_t *ps, const char *src) {
    const char *name_end = parse_name(src);
    if (!name_end) { ps->err = "expecting a rule name"; return NULL; }
    const char *pos = parse_space(name_end, 0);
    uint32_t id = gbnf_symbol_id(ps->g, src, (int)(name_end - src));
    char name[256];
    int nl = (int)(name_end - src);
    if (nl >= (int)sizeof(name)) nl = (int)sizeof(name) - 1;
    memcpy(name, src, (size_t)nl);
    name[nl] = '\0';

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        ps->err = "expecting ::=";
        return NULL;
    }
    pos = parse_space(pos + 3, 1);
    pos = parse_alternates(ps, pos, name, id, 0, 0);
    if (!pos) return NULL;

    if (*pos == '\r') pos += (pos[1] == '\n') ? 2 : 1;
    else if (*pos == '\n') pos++;
    else if (*pos) { ps->err = "expecting newline or end of input"; return NULL; }

    return parse_space(pos, 1);
}

int gbnf_parse(gbnf_t *g, const char *text, const char *root_name) {
    if (!g) return -1;
    memset(g, 0, sizeof(*g));

    parser_t ps;
    ps.g = g;
    ps.err = NULL;

    /* Seed "root" first so it gets id 0, matching llama.cpp. */
    gbnf_symbol_id(g, root_name ? root_name : "root",
                   (int)strlen(root_name ? root_name : "root"));

    const char *pos = parse_space(text, 1);
    while (*pos) {
        const char *np = parse_rule(&ps, pos);
        if (!np) {
            fprintf(stderr, "gbnf: parse error: %s\n  near: %.60s\n",
                    ps.err ? ps.err : "unknown", pos);
            gbnf_free(g);
            return -1;
        }
        pos = np;
    }

    /* Every referenced rule must exist and be non-empty; a grammar that
     * references an undefined symbol would otherwise match nothing and mask
     * the entire vocabulary, which looks like a sampler bug. */
    for (int i = 0; i < g->n_rules; i++) {
        if (g->rules[i].n == 0) {
            for (int k = 0; k < g->n_names; k++)
                if (g->name_ids[k] == i)
                    fprintf(stderr, "gbnf: rule '%s' is referenced but never defined\n", g->names[k]);
            gbnf_free(g);
            return -1;
        }
        for (int e = 0; e < g->rules[i].n; e++) {
            gbnf_element *el = &g->rules[i].elems[e];
            if (el->type == GRETYPE_RULE_REF &&
                (el->value >= (uint32_t)g->n_rules || g->rules[el->value].n == 0)) {
                fprintf(stderr, "gbnf: rule %d references undefined rule %u\n", i, el->value);
                gbnf_free(g);
                return -1;
            }
        }
    }

    gbnf_build_class_bits(g);

    if (!root_name) root_name = "root";
    uint32_t root_id = (uint32_t)-1;
    for (int k = 0; k < g->n_names; k++) {
        if (strcmp(g->names[k], root_name) == 0) { root_id = (uint32_t)g->name_ids[k]; break; }
    }
    if (root_id == (uint32_t)-1) {
        fprintf(stderr, "gbnf: no rule named '%s'\n", root_name);
        gbnf_free(g);
        return -1;
    }
    return (int)root_id;
}

/* ------------------------------------------------------------------ */
/* Character-class bitmaps                                            */
/* ------------------------------------------------------------------ */

/* Fill each character class head's 256-bit membership map, so an ASCII
 * membership test is one bit test rather than a walk over the class.
 *
 * This is not a micro-optimisation: nosthit.gbnf's "punctuation" class has 30+
 * elements, and match_char() was 38% of the profile before the bitmaps went
 * in, because every character of every candidate token walked the list. */
static void gbnf_build_class_bits(gbnf_t *g) {
    for (int r = 0; r < g->n_rules; r++) {
        gbnf_rule *rule = &g->rules[r];
        for (int i = 0; i < rule->n; i++) {
            gbnf_element *e = &rule->elems[i];
            if (!(e->type == GRETYPE_CHAR || e->type == GRETYPE_CHAR_NOT ||
                  e->type == GRETYPE_CHAR_ANY))
                continue;
            /* Only the first element of a class carries the bitmap; the rest
             * are CHAR_ALT / CHAR_RNG_UPPER continuations. */
            if (i > 0) {
                uint8_t pt = rule->elems[i - 1].type;
                if (pt == GRETYPE_CHAR_ALT || pt == GRETYPE_CHAR_RNG_UPPER) continue;
            }
            e->is_class_head = 1;
            memset(e->bits, 0, sizeof(e->bits));
            e->has_high = 0;
            int clen = 0;

            const gbnf_element *p = e;
            for (;;) {
                if (p[1].type == GRETYPE_CHAR_RNG_UPPER) {
                    uint32_t lo = p->value, hi = p[1].value;
                    if (hi > 255 || lo > 255) e->has_high = 1;
                    if (lo <= 255) {
                        if (hi > 255) hi = 255;
                        for (uint32_t c = lo; c <= hi; c++)
                            e->bits[c >> 3] |= (uint8_t)(1u << (c & 7));
                    }
                    p += 2; clen += 2;
                } else if (p->type == GRETYPE_CHAR_ANY) {
                    memset(e->bits, 0xFF, sizeof(e->bits));
                    e->has_high = 1;
                    p += 1; clen += 1;
                } else {
                    if (p->value > 255) e->has_high = 1;
                    else e->bits[p->value >> 3] |= (uint8_t)(1u << (p->value & 7));
                    p += 1; clen += 1;
                }
                if (p->type != GRETYPE_CHAR_ALT) break;
            }
            e->class_len = (uint16_t)clen;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Matcher                                                            */
/* ------------------------------------------------------------------ */

static int is_end_of_sequence(const gbnf_element *e) {
    return e->type == GRETYPE_END || e->type == GRETYPE_ALT;
}

static int is_char_element(const gbnf_element *e) {
    return e->type >= GRETYPE_CHAR && e->type <= GRETYPE_CHAR_ANY;
}

static int match_char(const gbnf_element *pos, uint32_t chr, const gbnf_element **next) {
    int positive = (pos->type == GRETYPE_CHAR || pos->type == GRETYPE_CHAR_ANY);

    if (pos->is_class_head && chr < 256) {
        /* One bit test. Exact also for [^...]: a code point above 255 can
         * never be a member of an ASCII-only listed set. */
        *next = pos + pos->class_len;
        int found = (pos->bits[chr >> 3] >> (chr & 7)) & 1;
        return (found == positive);
    }

    int found = 0;
    for (;;) {
        if (pos[1].type == GRETYPE_CHAR_RNG_UPPER) {
            if (pos->value <= chr && chr <= pos[1].value) found = 1;
            pos += 2;
        } else if (pos->type == GRETYPE_CHAR_ANY) {
            found = 1;
            pos += 1;
        } else {
            if (pos->value == chr) found = 1;
            pos += 1;
        }
        if (pos->type != GRETYPE_CHAR_ALT) break;
    }
    *next = pos;
    return (found == positive);
}

/* Expand rule references until every stack ends on a terminal, pushing the
 * remainder of each sequence under the expansion. The only place that grows
 * the stack; dedup stops ambiguous grammars exploding. */
static void advance_stack(const gbnf_rule *rules, const gbnf_element *const *stack, int n,
                          gbnf_stacks *out) {
    gbnf_stacks todo;
    stacks_clear(&todo);
    stacks_add(&todo, stack, n);

    while (todo.n > 0) {
        todo.n--;
        gbnf_stack cur = todo.v[todo.n];

        if (cur.n == 0) { stacks_add(out, NULL, 0); continue; }

        const gbnf_element *top = cur.ptrs[cur.n - 1];
        if (top->type == GRETYPE_RULE_REF) {
            const gbnf_rule *sub = &rules[top->value];
            const gbnf_element *base[GBNF_MAX_STACK_DEPTH];
            int base_n = cur.n - 1;
            for (int i = 0; i < base_n; i++) base[i] = cur.ptrs[i];

            for (int alt = 0; alt < sub->n; ) {
                const gbnf_element *subpos = &sub->elems[alt];
                const gbnf_element *ns[GBNF_MAX_STACK_DEPTH];
                int nn = 0;
                for (int i = 0; i < base_n; i++) ns[nn++] = base[i];
                if (!is_end_of_sequence(top + 1)) ns[nn++] = top + 1;
                if (!is_end_of_sequence(subpos)) ns[nn++] = subpos;
                stacks_add(&todo, ns, nn);

                while (!is_end_of_sequence(subpos)) subpos++;
                if (subpos->type == GRETYPE_ALT) alt = (int)(subpos - sub->elems) + 1;
                else break;
            }
        } else {
            stacks_add(out, cur.ptrs, cur.n);
        }
    }
}

/* Feed one code point to one stack, producing any resulting stacks. */
static void accept_chr(const gbnf_rule *rules, const gbnf_element *const *stack, int n,
                       uint32_t chr, gbnf_stacks *out) {
    if (n == 0) return;
    const gbnf_element *top = stack[n - 1];
    if (!is_char_element(top)) return;

    const gbnf_element *next = NULL;
    if (match_char(top, chr, &next)) {
        const gbnf_element *ns[GBNF_MAX_STACK_DEPTH];
        int nn = 0;
        for (int i = 0; i < n - 1; i++) ns[nn++] = stack[i];
        if (!is_end_of_sequence(next)) ns[nn++] = next;
        advance_stack(rules, ns, nn, out);
    }
}

/* ------------------------------------------------------------------ */
/* Walking token bytes through a stack set                            */
/* ------------------------------------------------------------------ */

/* Walk `bytes` through one stack.
 *
 * Bytes, not code points: a byte-level BPE token can end in the middle of a
 * multi-byte character (a lone lead byte completed by the next token), and
 * only a byte walk can track that. `partial` carries over between tokens.
 *
 * With commit==0 only reachability matters. With commit==1 the successful
 * stacks are written to `commit_stacks` (the caller then adopts them) and the
 * final partial state to `partial`.
 */
static int walk_bytes(const gbnf_rule *rules,
                      const gbnf_element *const *stack, int n,
                      const uint8_t *bytes, int n_bytes,
                      gbnf_utf8 *partial,
                      gbnf_stacks *commit_stacks) {
    static _Thread_local gbnf_stacks buf_a, buf_b;
    gbnf_stacks *cur = &buf_a, *nxt = &buf_b;
    stacks_clear(cur);
    stacks_add(cur, stack, n);

    gbnf_utf8 p = *partial;

    for (int i = 0; i < n_bytes && cur->n > 0 && p.n_remain >= 0; ) {
        uint32_t chr;
        int need;

        if (p.n_remain > 0) {
            /* completing a sequence started in an earlier token */
            uint8_t b = bytes[i];
            if ((b & 0xC0) != 0x80) { p.n_remain = -1; break; }
            p.value = (p.value << 6) | (uint32_t)(b & 0x3F);
            p.n_remain--;
            i++;
            if (p.n_remain != 0) continue;
            chr = p.value;
        } else {
            uint8_t b = bytes[i];
            chr = lead_byte(b, &need);
            if (need == 1) i += 1;
            else if (i + need > n_bytes) {
                /* ends mid-character: remember the prefix and stop */
                int got = n_bytes - i - 1;
                uint32_t v = chr;
                int ok = 1;
                for (int k = 0; k < got; k++) {
                    uint8_t cb = bytes[i + 1 + k];
                    if ((cb & 0xC0) != 0x80) { ok = 0; break; }
                    v = (v << 6) | (uint32_t)(cb & 0x3F);
                }
                if (!ok) { p.n_remain = -1; break; }
                p.value = v;
                p.n_remain = need - 1 - got;
                i = n_bytes;
                break;
            } else {
                int ok = 1;
                for (int k = 1; k < need; k++) {
                    uint8_t cb = bytes[i + k];
                    if ((cb & 0xC0) != 0x80) { ok = 0; break; }
                    chr = (chr << 6) | (uint32_t)(cb & 0x3F);
                }
                if (!ok) { p.n_remain = -1; break; }
                i += need;
            }
        }

        stacks_clear(nxt);
        for (int s = 0; s < cur->n; s++)
            accept_chr(rules, cur->v[s].ptrs, cur->v[s].n, chr, nxt);
        gbnf_stacks *t = cur; cur = nxt; nxt = t;
    }

    int ok = (cur->n > 0) && (p.n_remain >= 0);
    *partial = p;
    if (ok && commit_stacks) *commit_stacks = *cur;
    return ok;
}

/* ------------------------------------------------------------------ */
/* Verdict cache                                                      */
/* ------------------------------------------------------------------ */

/* First-byte prefilter for the current state: first_ok[b] for ASCII, and
 * first_partial[b] for multi-byte lead bytes (those need the full walk). */
static void build_first_byte_filter(gbnf_t *g) {
    memset(g->first_ok, 0, sizeof(g->first_ok));
    memset(g->first_partial, 0, sizeof(g->first_partial));

    if (g->partial_utf8.n_remain != 0)
        for (int b = 0x80; b < 0xC0; b++) g->first_partial[b] = 1;

    for (int b = 0; b < 0x80; b++) {
        for (int s = 0; s < g->stacks.n; s++) {
            if (g->stacks.v[s].n == 0) continue;
            const gbnf_element *top = g->stacks.v[s].ptrs[g->stacks.v[s].n - 1];
            if (!is_char_element(top)) continue;
            const gbnf_element *next = NULL;
            if (match_char(top, (uint32_t)b, &next)) { g->first_ok[b] = 1; break; }
        }
    }
    int any = 0;
    for (int s = 0; s < g->stacks.n; s++)
        if (g->stacks.v[s].n > 0 &&
            is_char_element(g->stacks.v[s].ptrs[g->stacks.v[s].n - 1])) any = 1;
    for (int b = 0xC0; b < 0x100; b++) g->first_partial[b] = (uint8_t)any;
}

static int token_accepted_now(const gbnf_t *g, int i) {
    const uint8_t *bytes = g->token_bytes[i];
    if (!bytes) return 0;
    int n = g->token_len[i];
    if (n == 0) return 1;               /* consumes nothing */

    uint8_t b0 = bytes[0];
    if (b0 < 0x80) { if (!g->first_ok[b0]) return 0; }
    else if (!g->first_partial[b0]) return 0;

    for (int s = 0; s < g->stacks.n; s++) {
        gbnf_utf8 partial = g->partial_utf8;
        if (walk_bytes(g->rules, g->stacks.v[s].ptrs, g->stacks.v[s].n,
                       bytes, n, &partial, NULL))
            return 1;
    }
    return 0;
}

/* Signature of everything token verdicts depend on. */
static uint64_t gbnf_state_sig(const gbnf_t *g) {
    uint64_t h = 1469598103934665603ull;
    h ^= (uint64_t)g->stacks.n; h *= 1099511628211ull;
    h ^= ((uint64_t)g->partial_utf8.value << 8) ^ (uint64_t)(g->partial_utf8.n_remain + 2);
    h *= 1099511628211ull;
    for (int s = 0; s < g->stacks.n; s++) {
        h ^= (uint64_t)g->stacks.v[s].n; h *= 1099511628211ull;
        for (int q = 0; q < g->stacks.v[s].n; q++) {
            h ^= (uint64_t)(uintptr_t)g->stacks.v[s].ptrs[q];
            h *= 1099511628211ull;
        }
    }
    return h;
}

void gbnf_refresh_cache(gbnf_t *g, int n_threads) {
    (void)n_threads;
    if (!g || !g->token_bytes || g->vocab_size <= 0) return;
    if (!g->accept_cache) {
        g->accept_cache = (uint8_t *)malloc((size_t)g->vocab_size);
        if (!g->accept_cache) return;
    }
    int vs = g->vocab_size;

    /* Verdicts are a pure function of the parse state, and loop rules cycle
     * through a handful of states (a paragraph used 4), so try the cache. */
    uint64_t sig = gbnf_state_sig(g);
    for (int i = 0; i < g->state_cache_n; i++) {
        if (g->state_sig[i] == sig && g->state_cache[i]) {
            memcpy(g->accept_cache, g->state_cache[i], (size_t)vs);
            g->cache_valid = 1;
            return;
        }
    }

    int prof = prof_on();
    double t0 = prof ? now_ms() : 0.0;
    build_first_byte_filter(g);
    double t1 = prof ? now_ms() : 0.0;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < vs; i++)
        g->accept_cache[i] = (uint8_t)token_accepted_now(g, i);

    if (prof) {
        int acc = 0;
        for (int i = 0; i < vs; i++) acc += g->accept_cache[i];
        fprintf(stderr, "[GBNF] state MISS %016llx filter=%.2fms walks=%.2fms accepted=%d/%d\n",
                (unsigned long long)sig, t1 - t0, now_ms() - t1, acc, vs);
    }

    g->cache_valid = 1;

    int slot;
    if (g->state_cache_n < GBNF_STATE_CACHE_SLOTS) slot = g->state_cache_n++;
    else {
        slot = g->state_cache_next;
        g->state_cache_next = (g->state_cache_next + 1) % GBNF_STATE_CACHE_SLOTS;
    }
    if (!g->state_cache[slot]) {
        g->state_cache[slot] = (uint8_t *)malloc((size_t)vs);
        if (!g->state_cache[slot]) return;
    }
    memcpy(g->state_cache[slot], g->accept_cache, (size_t)vs);
    g->state_sig[slot] = sig;
}

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

int gbnf_start(gbnf_t *g) {
    uint32_t root = 0;
    for (int k = 0; k < g->n_names; k++)
        if (strcmp(g->names[k], "root") == 0) { root = (uint32_t)g->name_ids[k]; break; }

    const gbnf_element *start = g->rules[root].elems;
    stacks_clear(&g->stacks);
    advance_stack(g->rules, &start, 1, &g->stacks);
    g->partial_utf8.value = 0;
    g->partial_utf8.n_remain = 0;
    g->started = 0;
    g->state_cache_n = 0;
    g->state_cache_next = 0;
    gbnf_refresh_cache(g, 0);
    return g->stacks.n;
}

void gbnf_set_vocab(gbnf_t *g, char **vocab, int vocab_size, int eos_id) {
    g->vocab_size = vocab_size;
    g->eos_id = eos_id;
    g->token_bytes = (uint8_t **)calloc((size_t)vocab_size, sizeof(uint8_t *));
    g->token_len   = (int *)calloc((size_t)vocab_size, sizeof(int));
    if (!g->token_bytes || !g->token_len) return;
    for (int i = 0; i < vocab_size; i++) {
        if (!vocab[i]) continue;
        int n = (int)strlen(vocab[i]);
        g->token_len[i] = n;
        g->token_bytes[i] = (uint8_t *)malloc((size_t)n + 1);
        if (!g->token_bytes[i]) { g->token_len[i] = 0; continue; }
        memcpy(g->token_bytes[i], vocab[i], (size_t)n + 1);
    }
}

void gbnf_apply(const gbnf_t *g, float *logits, int vocab_size) {
    if (!g || !g->token_bytes) return;
    if (!g->accept_cache || !g->cache_valid) return;

    /* A DEAD state (no live stack) must fail CLOSED.
     *
     * This used to be part of the early return above: `|| g->stacks.n == 0`.
     * That made a dead grammar behave exactly like no grammar at all -- every
     * token stayed allowed and generation continued completely unconstrained.
     * The state is reachable with ordinary models: a byte-level BPE vocab
     * contains tokens that are UTF-8 fragments (GLM-5's id 21056 is
     * "\x20\xE2\x89"), the matcher deliberately admits a token that ends
     * mid-character, and if the next token then completes a character the
     * grammar forbids, every stack dies. From that point the old code masked
     * nothing -- so a grammar that had correctly rejected U+2248 let the model
     * emit anything at all.
     *
     * Masking everything (EOS included) is correct here: there is no legal
     * continuation, so the only sane outcomes are "stop" or "this state is a
     * bug". The caller sees gbnf_is_complete() == 1 and ends generation. */
    if (g->stacks.n == 0) {
        int vs_dead = (vocab_size < g->vocab_size) ? vocab_size : g->vocab_size;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < vs_dead; i++) logits[i] = GBNF_NEG_INF;
        return;
    }

    int vs = (vocab_size < g->vocab_size) ? vocab_size : g->vocab_size;

    /* EOS is legal iff some stack consumed the whole grammar. That is a
     * property of the state, not of a token. */
    int complete = 0;
    for (int s = 0; s < g->stacks.n; s++)
        if (g->stacks.v[s].n == 0) { complete = 1; break; }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < vs; i++) {
        if (i == g->eos_id) {
            if (!complete) logits[i] = GBNF_NEG_INF;
            continue;
        }
        if (!g->accept_cache[i]) logits[i] = GBNF_NEG_INF;
    }
}

void gbnf_accept_token(gbnf_t *g, int token) {
    if (!g || !g->token_bytes) return;
    if (token < 0 || token >= g->vocab_size) return;
    const uint8_t *bytes = g->token_bytes[token];
    int n = g->token_len[token];
    if (!bytes || n == 0) return;

    /* Commit: walk each live stack and adopt the union of the survivors. */
    gbnf_stacks acc;
    stacks_clear(&acc);
    gbnf_utf8 result_partial = g->partial_utf8;

    for (int s = 0; s < g->stacks.n; s++) {
        gbnf_utf8 partial = g->partial_utf8;
        gbnf_stacks out;
        stacks_clear(&out);
        if (walk_bytes(g->rules, g->stacks.v[s].ptrs, g->stacks.v[s].n,
                       bytes, n, &partial, &out)) {
            for (int k = 0; k < out.n; k++)
                stacks_add(&acc, out.v[k].ptrs, out.v[k].n);
            result_partial = partial;   /* all survivors agree on this */
        }
    }

    g->stacks = acc;
    g->partial_utf8 = result_partial;
    g->started = 1;
    gbnf_refresh_cache(g, 0);
}

int gbnf_is_complete(const gbnf_t *g) {
    /* A dead state is "finished" in the only sense the caller cares about:
     * nothing can be emitted. Returning 0 here (the old behaviour) meant
     * gbnf_apply masked every token and generation then sampled from an
     * all-negative-infinity array. The caller must stop instead. */
    if (!g || !g->started) return 0;
    if (g->stacks.n == 0) return 1;
    /* Complete means "no way to produce another character", i.e. EVERY live
     * stack is exhausted. An empty stack alone does not mean that: for
     * root ::= a (b)*, after "a" one stack is empty while others can still
     * take "b". Returning 1 there ended generation after one token. */
    for (int s = 0; s < g->stacks.n; s++)
        if (g->stacks.v[s].n != 0) return 0;
    return 1;
}

void gbnf_free(gbnf_t *g) {
    if (!g) return;
    for (int i = 0; i < g->n_rules; i++) free(g->rules[i].elems);
    free(g->rules);
    for (int i = 0; i < g->n_names; i++) free(g->names[i]);
    free(g->names);
    free(g->name_ids);
    if (g->token_bytes) {
        for (int i = 0; i < g->vocab_size; i++) free(g->token_bytes[i]);
        free(g->token_bytes);
    }
    free(g->token_len);
    free(g->accept_cache);
    for (int i = 0; i < GBNF_STATE_CACHE_SLOTS; i++) free(g->state_cache[i]);
    memset(g, 0, sizeof(*g));
}
