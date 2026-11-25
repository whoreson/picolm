#ifndef BACKEND_KV_WATERMARK_H
#define BACKEND_KV_WATERMARK_H
/* Host-mirror watermark for the GPU KV cache. Pure integer logic, no GPU types, so
 * tests/test_kv_watermark.c can run the exact code the backend runs.
 *
 * INVARIANT: the host mirror (s->key_cache / s->val_cache) is a faithful copy of the
 * device cache for positions [0, valid).
 *
 * It is owned by the backend because EVERY device KV write goes through the backend:
 * each writer calls kvwm_note_write() with the first position it touches, which lowers
 * `valid` to that position. So a rewind followed by a replay through ANY path (decode,
 * prefill, bulk upload) invalidates exactly the rows it changed, instead of the flush
 * having to infer it from the caller's start_pos. Allocation, clear and free reset it to 0.
 *
 * Syncing the host mirror then only ever needs to copy [valid, pos_to). */

static inline void kvwm_note_write(int *valid, int first_pos) {
    if (first_pos < 0) first_pos = 0;
    if (first_pos < *valid) *valid = first_pos;
}

/* Plan a sync up to pos_to. pos_to is CLAMPED to [0, max_seq] here, once, so no caller can
 * record a position beyond the cache. Returns 1 if rows [*from, *to) must be copied,
 * 0 if the host mirror is already current up to pos_to (nothing to do, `valid` unchanged). */
static inline int kvwm_plan_sync(int valid, int pos_to, int max_seq, int *from, int *to) {
    if (pos_to > max_seq) pos_to = max_seq;
    if (pos_to < 0) pos_to = 0;
    *to = pos_to;
    *from = valid < pos_to ? valid : pos_to;
    return pos_to > valid;
}

/* Record a SUCCESSFUL copy of [valid, to) (or of [0, to) for a full flush). Never lowers. */
static inline void kvwm_commit(int *valid, int to) {
    if (to > *valid) *valid = to;
}
#endif /* BACKEND_KV_WATERMARK_H */
