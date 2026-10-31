/* Regression test: kvcache_save()'s token-sequence writer.
 *
 * The writer streams n_pos uint32 token ids through a fixed 128 KiB buffer.
 * It used to do a single `if (buf_off + tk_sz > buf_size) flush();` and then
 * write all n_pos entries unconditionally, so any prompt longer than
 * 32768 tokens (128 KiB / 4) wrote past the buffer -- 282 KiB against a
 * 128 KiB alloca at 70K tokens, i.e. 154 KiB past the end, which segfaulted
 * inside the /slots save of a long agentic session.
 *
 * This test replays the exact loop against deliberately tiny buffers (so the
 * chunking path is exercised thousands of times, including non-4-aligned
 * offsets) and checks the emitted byte stream. Build with
 * -fsanitize=address,undefined to also catch the misaligned uint32_t store an
 * earlier revision of the fix used.
 *
 *   cc -O2 -fsanitize=address,undefined -o /tmp/t tests/test_kvcache_tokwrite.c
 *   /tmp/t
 */
/* Standalone verification of the chunked token writer used by kvcache_save:
 * emulate the write loop against a small buffer and check the byte stream. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* --- reference: what _kv_flush does into a growable file image --- */
typedef struct { uint8_t *p; size_t n, cap; } out_t;
static int flush(out_t *o, uint8_t *buf, size_t *off) {
    if (*off) {
        if (o->n + *off > o->cap) { o->cap = (o->n + *off) * 2; o->p = realloc(o->p, o->cap); }
        memcpy(o->p + o->n, buf, *off); o->n += *off; *off = 0;
    }
    return 0;
}

/* --- the NEW loop, verbatim from model_kv_cache.c --- */
static int write_tokens(out_t *o, uint8_t *buf, size_t buf_size, size_t *buf_off,
                        const int *tokens, int n_pos) {
    if (tokens && n_pos > 0) {
        size_t tk_i = 0;
        while (tk_i < (size_t)n_pos) {
            if (*buf_off + 4 > buf_size) { if (flush(o, buf, buf_off)) return -1; }
            size_t room = buf_size - *buf_off;
            size_t nfit = room / 4;
            size_t left = (size_t)n_pos - tk_i;
            if (nfit > left) nfit = left;
            if (nfit == 0) { if (flush(o, buf, buf_off)) return -1; continue; }
            for (size_t j = 0; j < nfit; j++) {
                uint32_t v = (uint32_t)tokens[tk_i + j];
                memcpy(buf + *buf_off + j * 4, &v, 4);
            }
            *buf_off += nfit * 4;
            tk_i += nfit;
        }
    }
    return 0;
}

int main(void) {
    /* small buffer (34 bytes) to force many chunks, incl. odd sizes */
    size_t sizes[] = { 34, 35, 36, 8, 5, 131072 };
    int fails = 0;
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        size_t buf_size = sizes[si];
        for (int n_pos = 0; n_pos <= 300; n_pos++) {
            int *tokens = malloc(sizeof(int) * (n_pos ? n_pos : 1));
            for (int i = 0; i < n_pos; i++) tokens[i] = 1000000 + i * 7;
            uint8_t *buf = calloc(1, buf_size);
            out_t o = {0};
            size_t off = 0;
            /* seed a few bytes like the header would */
            size_t seed = buf_size >= 8 ? 5 : buf_size; memset(buf, 0xAA, seed); off = seed;
            write_tokens(&o, buf, buf_size, &off, tokens, n_pos);
            flush(&o, buf, &off);
            /* verify: 5 bytes of 0xAA then n_pos uint32 values */
            size_t want = seed + (size_t)n_pos * 4;
            int bad = 0;
            if (o.n != want) bad = 1;
            else {
                for (int i = 0; i < n_pos; i++) {
                    uint32_t v; memcpy(&v, o.p + seed + i*4, 4);
                    if (v != (uint32_t)tokens[i]) { bad = 1; break; }
                }
            }
            if (bad) { printf("FAIL buf_size=%zu n_pos=%d got=%zu want=%zu\n", buf_size, n_pos, o.n, want); fails++; }
            free(buf); free(o.p); free(tokens);
        }
    }
    printf("%s (%d failures)\n", fails ? "FAILURES" : "all token-write shapes OK", fails);
    return fails != 0;
}
