/* test_kv_watermark: randomized check of the host-mirror watermark (backend_kv_watermark.h),
 * the exact logic the GPU backend runs.
 *
 * Model: `dev` is the device KV cache and `host` the host mirror, one int per position (the
 * id of the write that produced it; 0 = zero-initialised). Every device writer calls
 * kvwm_note_write(); sync copies exactly what kvwm_plan_sync() says. After EVERY sync the
 * mirror must equal the device on [0, min(pos_to, max_seq)).
 *
 * Operations are random: prefill chunks (at any start, including rewinds), single-token
 * decode, rewind-then-DECODE-replay (the case the old model-level watermark missed), bulk
 * upload, clear, and syncs with pos_to past the end of the cache (overflow).
 *
 *   cc -O2 -I. tests/test_kv_watermark.c -o test_kv_watermark && ./test_kv_watermark
 *   ./test_kv_watermark --no-upload-clear   drop bulk upload/clear (isolates rewind, decode-replay, overflow)
 *   ./test_kv_watermark --old    runs the PREVIOUS model-level logic (min(wm,start_pos), unclamped
 *                                `to`) on the same sequences; it must FAIL, proving the test can see
 *                                the reported bugs. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "backend_kv_watermark.h"

#define MAXSEQ 64
static int dev[MAXSEQ], host[MAXSEQ], valid, next_id = 1, copied_rows;
static int old_wm;                    /* the previous model-level watermark */
static int use_old, restricted;      /* restricted: no bulk upload / clear (operations the old logic never modelled) */
static unsigned rs = 12345;
static int rnd(int n) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return (int)(rs % (unsigned)n); }

static void dev_write(int pos) { if (!use_old) kvwm_note_write(&valid, pos); dev[pos] = next_id++; }
static void do_clear(void)     { memset(dev, 0, sizeof dev); if (!use_old) valid = 0; /* host keeps stale rows */ }
static void do_upload(int n)   { if (!use_old) kvwm_note_write(&valid, 0); for (int i = 0; i < n; i++) dev[i] = next_id++; }

/* returns number of positions checked; sets *bad on a mismatch */
static int do_sync(int start_pos, int to_arg, int *bad) {
    int from, to;
    if (use_old) {
        /* previous behaviour: flush [min(wm, start_pos), to) with to clamped inside the backend,
         * but the model records the UNCLAMPED to */
        int f = old_wm < start_pos ? old_wm : start_pos; if (f < 0) f = 0;
        int t = to_arg > MAXSEQ ? MAXSEQ : to_arg;
        if (t > f) { for (int i = f; i < t; i++) host[i] = dev[i]; copied_rows += t - f; }
        old_wm = to_arg;
        from = 0; to = t;
    } else {
        if (kvwm_plan_sync(valid, to_arg, MAXSEQ, &from, &to)) {
            for (int i = from; i < to; i++) host[i] = dev[i];
            copied_rows += to - from;
            kvwm_commit(&valid, to);
        }
    }
    for (int i = 0; i < to; i++) if (host[i] != dev[i]) { *bad = i; return to; }
    return to;
}

int main(int argc, char **argv) {
    for (int a = 1; a < argc; a++) { if (!strcmp(argv[a], "--old")) use_old = 1; if (!strcmp(argv[a], "--no-upload-clear")) restricted = 1; }
    long syncs = 0, failures = 0; int first_fail_shown = 0;
    for (int run = 0; run < 3000; run++) {
        memset(dev, 0, sizeof dev); memset(host, 0, sizeof host); valid = 0; old_wm = 0; next_id = 1;
        int pos = 0;                                   /* the engine's current position */
        for (int step = 0; step < 40; step++) {
            int op = rnd(100);
            if (restricted && op >= 85 && op < 93) op = 0;     /* replace upload/clear by a prefill */
            if (op < 30) {                             /* prefill chunk at pos, then sync */
                int n = 1 + rnd(12); if (pos + n > MAXSEQ) n = MAXSEQ - pos; if (n < 1) { pos = rnd(MAXSEQ); continue; }
                for (int i = 0; i < n; i++) dev_write(pos + i);
                int bad = -1; do_sync(pos, pos + n, &bad); syncs++;
                if (bad >= 0) { failures++; if (!first_fail_shown) { first_fail_shown = 1; printf("  first failure: run %d step %d after prefill(start=%d n=%d): host[%d]=%d != dev[%d]=%d\n", run, step, pos, n, bad, host[bad], bad, dev[bad]); } }
                pos += n;
            } else if (op < 55) {                      /* GPU decode of 1..6 tokens (device only) */
                int k = 1 + rnd(6);
                for (int i = 0; i < k && pos < MAXSEQ; i++) dev_write(pos++);
            } else if (op < 75) {                      /* rewind (e.g. checkpoint restore) */
                pos = rnd(pos + 1);
            } else if (op < 85) {                      /* rewind then replay through DECODE, then a prefill sync */
                pos = rnd(pos + 1);
                int k = 1 + rnd(8);
                for (int i = 0; i < k && pos < MAXSEQ; i++) dev_write(pos++);
                int bad = -1; do_sync(pos, pos + 1 + rnd(4), &bad); syncs++;
                if (bad >= 0) { failures++; if (!first_fail_shown) { first_fail_shown = 1; printf("  first failure: run %d step %d after rewind+decode-replay: host[%d]=%d != dev[%d]=%d\n", run, step, bad, host[bad], bad, dev[bad]); } }
            } else if (op < 90) {                      /* bulk upload */
                do_upload(1 + rnd(MAXSEQ));
            } else if (op < 93) {                      /* clear */
                do_clear(); pos = 0;
            } else {                                   /* sync with a position past the end (context overflow) */
                int bad = -1; do_sync(pos, MAXSEQ + 1 + rnd(40), &bad); syncs++;
                if (bad >= 0) { failures++; if (!first_fail_shown) { first_fail_shown = 1; printf("  first failure: run %d step %d after overflow sync: host[%d]=%d != dev[%d]=%d\n", run, step, bad, host[bad], bad, dev[bad]); } }
            }
        }
    }
    printf("%s logic: %ld syncs, %ld failures, %d rows copied in total\n", use_old ? "OLD (model-level)" : "NEW (backend-owned)", syncs, failures, copied_rows);
    if (use_old) { printf(failures ? "old logic fails as expected (test is sensitive to the reported bugs)\n" : "WARNING: old logic did not fail; the test is too weak\n"); return failures ? 0 : 1; }
    printf(failures ? "FAILED\n" : "all ok\n");
    return failures != 0;
}
