#include "tensor.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define WINVER 0x0600
#define _WIN32_WINNT 0x0600
#include <windows.h>
#include <synchapi.h>
#else
#include <pthread.h>
#endif

/* Thread pool (Windows-native or POSIX) */
#ifdef _WIN32
typedef HANDLE            win_thread_t;
typedef SRWLOCK           win_mutex_t;
typedef CONDITION_VARIABLE win_cond_t;
#else
typedef pthread_t  win_thread_t;
typedef pthread_mutex_t win_mutex_t;
typedef pthread_cond_t  win_cond_t;
#endif

/* ---- Scratch buffer (kept for dequantize_row in model.c) ---- */

static float *scratch_buf = NULL;
static int    scratch_size = 0;

/* Repacked Q4_0->Q4_0x8 weight pointer (set per-matmul by caller on AVX2) */
static const void *wptr_repacked = NULL;

void tensor_set_repacked(const void *ptr) {
    wptr_repacked = ptr;
}

void tensor_init_scratch(float *buf, int size) {
    scratch_buf  = buf;
    scratch_size = size;
}

/* ---- Threading for matmul ---- */

static int n_threads = 1;

void tensor_set_threads(int t) {
    if (t < 1) t = 1;
    if (t > MAX_THREADS) t = MAX_THREADS;
    n_threads = t;
}

/* Threshold: skip threading if output vector is smaller than this.
 * Avoids mutex overhead for tiny matmuls. */
static int matmul_min_rows = 1024;

void tensor_set_matmul_min_rows(int r) {
    if (r < 0) r = 0;
    matmul_min_rows = r;
}

int tensor_get_threads(void) {
    return n_threads;
}

typedef struct {
    float       *out;
    const float *x;
    const float *x_d;     /* pre-converted fp32 deltas for Q8_0 quantized x */
    const char  *W;
    size_t       row_bytes;
    int          n;        /* input dimension */
    int          start;    /* first output row */
    int          end;      /* one past last output row */
    gguf_type_t  qtype;
} matmul_task_t;

/* ---- Persistent thread pool (port from picolm-evilbinary, simplified) ---- */

/* Pool state: generation-counter barrier pattern.
 *
 * Workers loop:
 *   - Wait on pool_cond while pool_gen == last_gen
 *   - When pool_gen changes, wake up, read pool_tasks[tid], execute matmul_worker_f
 *   - Increment pool_done, broadcast pool_cond, loop back
 *
 * Dispatcher:
 *   - Fill pool_tasks[0..nt-1], increment pool_gen, broadcast pool_cond
 *   - Main thread executes task 0
 *   - Wait on pool_cond until pool_done == nt - 1
 *   - Reset pool_done = 0
 */

static win_thread_t    pool_threads[MAX_THREADS];
static win_mutex_t     pool_mutex;
static win_cond_t      pool_cond;
static volatile int    pool_gen   = 0;
static volatile int    pool_done  = 0;
static volatile int    pool_shutdown = 0;
static volatile int    pool_nworkers = 0;
static matmul_task_t   pool_tasks[MAX_THREADS];

/* Cross-platform mutex/cond helpers */
#ifdef _WIN32
static void win_mutex_init(win_mutex_t *m) { InitializeSRWLock(m); }
static void win_mutex_lock(win_mutex_t *m) { AcquireSRWLockExclusive(m); }
static void win_mutex_unlock(win_mutex_t *m) { ReleaseSRWLockExclusive(m); }
static void win_cond_init(win_cond_t *c) { InitializeConditionVariable(c); }
static void win_cond_wait(win_cond_t *c, win_mutex_t *m) { SleepConditionVariableSRW(c, m, INFINITE, 0); }
static void win_cond_broadcast(win_cond_t *c) { WakeAllConditionVariable(c); }
static void win_cond_destroy(win_cond_t *c) { (void)c; }
static void win_thread_create(win_thread_t *t, DWORD (WINAPI *fn)(void*), void *arg) { *t = CreateThread(NULL, 0, fn, arg, 0, NULL); }
static void win_thread_join(win_thread_t *t) { WaitForSingleObject(*t, INFINITE); CloseHandle(*t); }
#else
static void win_mutex_init(win_mutex_t *m) { pthread_mutex_init(m, NULL); }
static void win_mutex_lock(win_mutex_t *m) { pthread_mutex_lock(m); }
static void win_mutex_unlock(win_mutex_t *m) { pthread_mutex_unlock(m); }
static void win_cond_init(win_cond_t *c) { pthread_cond_init(c, NULL); }
static void win_cond_wait(win_cond_t *c, win_mutex_t *m) { pthread_cond_wait(c, m); }
static void win_cond_broadcast(win_cond_t *c) { pthread_cond_broadcast(c); }
static void win_cond_destroy(win_cond_t *c) { pthread_cond_destroy(c); }
static void win_thread_create(win_thread_t *t, void *(*fn)(void*), void *arg) { pthread_create(t, NULL, fn, arg); }
static void win_thread_join(win_thread_t *t) { pthread_join(*t, NULL); }
#endif

/* Core worker logic (same as before: handles Q8_0/Q4_K/Q4_0 fast paths) */
static void matmul_worker_f(matmul_task_t *t) {
    if (t->qtype == GGUF_TYPE_Q8_0 && t->x) {
        const block_q8_0 *qx = (const block_q8_0 *)t->x;
        const float *qx_d = t->x_d;
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot_q8_0_q8_0_deltas(qx, qx_d,
                                                  t->W + (size_t)i * t->row_bytes, t->n);
        }
#if defined(PICOLM_AVX2) || defined(PICOLM_AVX)
    } else if (t->qtype == GGUF_TYPE_Q4_K && t->x) {
        const block_q8_K *qx = (const block_q8_K *)t->x;
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot_q4_K_q8_K(
                t->W + (size_t)i * t->row_bytes, qx, t->n);
        }
    } else if (t->qtype == GGUF_TYPE_Q4_0 && t->x) {
        const block_q8_0 *qx = (const block_q8_0 *)t->x;
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot_q4_0_q8_0(
                t->W + (size_t)i * t->row_bytes, qx, t->n);
        }
    } else if (t->qtype == GGUF_TYPE_Q4_0_4_4 && t->x) {
        /* Q4_0_4_4: process 4 rows at a time from interleaved layout */
        const block_q8_0 *qx = (const block_q8_0 *)t->x;
        int start4 = (t->start / 4) * 4;
        int end4 = (t->end + 3) / 4 * 4;
        for (int i = start4; i < end4; i += 4) {
            vec_dot_q4_0x4_q8_0(
                t->W + (size_t)i * t->row_bytes, qx, t->n,
                t->out + i, 4);
        }
    } else if (t->qtype == GGUF_TYPE_Q4_0_8_8 && t->x) {
        /* Q4_0_8_8: process 8 rows at a time from interleaved layout */
#if defined(PICOLM_AVX2)
        const block_q8_0 *qx = (const block_q8_0 *)t->x;
        int start8 = (t->start / 8) * 8;
        int end8 = (t->end + 7) / 8 * 8;
        for (int i = start8; i < end8; i += 8) {
            int nrows = (i + 8 < end8) ? 8 : (end8 - i);
            vec_dot_q4_0x8_q8_0_avx2(
                t->W + (size_t)i * t->row_bytes, qx, t->n,
                t->out + i, nrows);
        }
#else
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot(t->W + (size_t)i * t->row_bytes,
                                t->x, t->n, t->qtype);
        }
#endif
#endif
    } else {
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot(t->W + (size_t)i * t->row_bytes,
                                t->x, t->n, t->qtype);
        }
    }
}

#ifdef _WIN32
static DWORD WINAPI pool_worker(void *arg) {
#else
static void *pool_worker(void *arg) {
#endif
    int tid = (int)(size_t)arg;
    int last_gen = 0;
    win_mutex_lock(&pool_mutex);
    while (1) {
        while (pool_gen == last_gen && !pool_shutdown) {
            win_cond_wait(&pool_cond, &pool_mutex);
        }
        if (pool_shutdown) break;
        last_gen = pool_gen;
        win_mutex_unlock(&pool_mutex);
        matmul_worker_f(&pool_tasks[tid]);
        win_mutex_lock(&pool_mutex);
        pool_done++;
        win_cond_broadcast(&pool_cond);
    }
    win_mutex_unlock(&pool_mutex);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static void pool_init(int nt) {
    if (pool_nworkers > 0) return;
    if (nt <= 1) return;
    win_mutex_init(&pool_mutex);
    win_cond_init(&pool_cond);
    pool_gen = 0;
    pool_done = 0;
    pool_shutdown = 0;
    pool_nworkers = nt - 1;
    for (int i = 1; i < nt; i++) {
        win_thread_create(&pool_threads[i], pool_worker, (void *)(size_t)i);
    }
}

static void pool_wake(int nt) {
    (void)nt;
    win_mutex_lock(&pool_mutex);
    pool_gen++;
    win_cond_broadcast(&pool_cond);
    win_mutex_unlock(&pool_mutex);
}

static void pool_wait(int nt) {
    win_mutex_lock(&pool_mutex);
    while (pool_done < nt - 1) {
        win_cond_wait(&pool_cond, &pool_mutex);
    }
    pool_done = 0;
    win_mutex_unlock(&pool_mutex);
}
/* ---- Public API ---- */

void tensor_threadpool_init(int n_threads) {
    if (n_threads > 1) pool_init(n_threads);
}

void tensor_threadpool_free(void) {
    if (pool_nworkers <= 0) return;
    win_mutex_lock(&pool_mutex);
    pool_shutdown = 1;
    win_cond_broadcast(&pool_cond);
    win_mutex_unlock(&pool_mutex);
    for (int i = 1; i <= pool_nworkers; i++) {
        win_thread_join(&pool_threads[i]);
    }
    pool_nworkers = 0;
    win_cond_destroy(&pool_cond);
#ifdef _WIN32
    /* SRWLOCK has no destroy */
#else
    pthread_mutex_destroy(&pool_mutex);
#endif
}

void matmul(float *out, const float *x, const void *W, int n, int d, gguf_type_t qtype) {
    size_t row_bytes = gguf_type_row_size(qtype, n);
    const char *wptr = (const char *)W;

    /* Q8_0 fast path: quantize x once, then use int8 MAC for all rows.
     * This mirrors llama.cpp's approach: the vec_dot_type for Q8_0 is Q8_0,
     * and the input is quantized to Q8_0 before the matmul loop.
     * This avoids float32 materialization of weights and uses fast
     * maddubs_epi16 integer multiply-accumulate. */
    if (qtype == GGUF_TYPE_Q8_0 && n > 0) {
        size_t q8_row_size = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
        int nb = n / 32;
        /* Pre-allocate buffer for: quantized x + pre-converted x deltas */
        size_t total_buf = q8_row_size + nb * sizeof(float);
        /* When threading is used, always malloc to avoid data races on scratch_buf */
        float *qx_buf = (n_threads > 1) ? (float *)malloc(total_buf) :
                        (scratch_buf ? (float *)scratch_buf : (float *)malloc(total_buf));
        if (!qx_buf) qx_buf = (float *)malloc(total_buf);
        if (qx_buf) {
            quantize_row_q8_0(x, qx_buf, n);
            const block_q8_0 *qx = (const block_q8_0 *)qx_buf;
            /* Pre-convert x deltas once for all rows */
            float *qx_d = qx_buf + (q8_row_size / sizeof(float));
            for (int bi = 0; bi < nb; bi++) qx_d[bi] = fp16_to_fp32(qx[bi].d);

            if (n_threads <= 1 || d < 4) {
                for (int i = 0; i < d; i++) {
                    out[i] = vec_dot_q8_0_q8_0_deltas(qx, qx_d, wptr + (size_t)i * row_bytes, n);
                }
                if (qx_buf != scratch_buf) free(qx_buf);
                return;
            }

            /* Threaded dispatch: main thread does task 0, workers do tasks 1..nt-1 */
            {
                int nt = n_threads;
                if (nt > d) nt = d;
                int rows_per = d / nt, extra = d % nt, tstart = 0;
                for (int t = 0; t < nt; t++) {
                    int tend = tstart + rows_per + (t < extra ? 1 : 0);
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = qx_d; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q8_0;
                    pool_tasks[t].start = tstart; pool_tasks[t].end = tend;
                    tstart = tend;
                }
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_buf != scratch_buf) free(qx_buf);
            return;
        }
        /* If allocation failed, fall through to generic path */
#if defined(PICOLM_AVX2) || defined(PICOLM_AVX)
    } else if (qtype == GGUF_TYPE_Q4_0) {
        /* Q4_0 fast path: quantize x to Q8_0 once, then vec_dot_q4_0_q8_0
         * On AVX2 with d%8==0, use repacked Q4_0_8x8 path for 8x row throughput. */
        size_t qx_size = (n / 32) * sizeof(block_q8_0);
        block_q8_0 *qx = NULL;
        int qx_owned = 0;
        /* When threading, always malloc to avoid data races */
        if (n_threads <= 1 && scratch_buf != NULL && qx_size <= (size_t)scratch_size) {
            qx = (block_q8_0 *)scratch_buf;
        } else {
            qx = (block_q8_0 *)malloc(qx_size);
            qx_owned = 1;
        }
        if (qx != NULL) {
            quantize_row_q8_0(x, qx, n);

#if defined(PICOLM_AVX2)
            /* Check if we have a repacked Q4_0x8 buffer available */
            const void *wptr8 = wptr_repacked; /* set by model_load via tensor_set_repacked() */
            if (wptr8 && d % 8 == 0 && n % 32 == 0) {
                /* Use AVX2 Q4_0_8x8 path: processes 8 rows per call */
                vec_dot_q4_0x8_q8_0_avx2(wptr8, qx, n, out, d);
                if (qx_owned) free(qx);
                return;
            }
#endif

            if (n_threads <= 1 || d < 4) {
                for (int i = 0; i < d; i++) {
                    out[i] = vec_dot_q4_0_q8_0(wptr + (size_t)i * row_bytes, qx, n);
                }
                if (qx_owned) free(qx);
                return;
            }

            int nt = n_threads;
            if (nt > d) nt = d;

            {
                int rows_per = d / nt, extra = d % nt, tstart = 0;
                for (int t = 0; t < nt; t++) {
                    int tend = tstart + rows_per + (t < extra ? 1 : 0);
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q4_0;
                    pool_tasks[t].start = tstart; pool_tasks[t].end = tend;
                    tstart = tend;
                }
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_owned) free(qx);
            return;
        }
        /* If allocation failed, fall through to generic path */
    } else if (qtype == GGUF_TYPE_Q4_0_4_4) {
        /* Q4_0_4_4 fast path: quantize x to Q8_0 once, then vec_dot_q4_0x4_q8_0
         * processes 4 rows simultaneously from the interleaved weight layout. */
        size_t qx_size = (n / 32) * sizeof(block_q8_0);
        block_q8_0 *qx = NULL;
        int qx_owned = 0;
        if (n_threads <= 1 && scratch_buf != NULL && qx_size <= (size_t)scratch_size) {
            qx = (block_q8_0 *)scratch_buf;
        } else {
            qx = (block_q8_0 *)malloc(qx_size);
            qx_owned = 1;
        }
        if (qx != NULL) {
            quantize_row_q8_0(x, qx, n);

            /* Align to multiples of 4 rows */
            int d4 = (d / 4) * 4;

            if (n_threads <= 1 || d < 4) {
                for (int i = 0; i < d4; i += 4) {
                    vec_dot_q4_0x4_q8_0(wptr + (size_t)i * row_bytes, qx, n,
                                         out + i, 4);
                }
                /* Scalar tail for remaining rows */
                for (int i = d4; i < d; i++) {
                    out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
                }
                if (qx_owned) free(qx);
                return;
            }

            int nt = n_threads;
            if (nt > d) nt = d;
            {
                int rows_per = d / nt, extra = d % nt, tstart = 0;
                for (int t = 0; t < nt; t++) {
                    int tend = tstart + rows_per + (t < extra ? 1 : 0);
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q4_0_4_4;
                    pool_tasks[t].start = tstart; pool_tasks[t].end = tend;
                    tstart = tend;
                }
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_owned) free(qx);
            return;
        }
    } else if (qtype == GGUF_TYPE_Q4_0_8_8) {
        /* Q4_0_8_8 fast path: weights are already in block_q4_0x8 interleaved format.
         * On AVX2, use vec_dot_q4_0x8_q8_0_avx2 for 8x row throughput.
         * Otherwise fall through to generic vec_dot path. */
#if defined(PICOLM_AVX2)
        size_t qx_size = (n / 32) * sizeof(block_q8_0);
        block_q8_0 *qx = NULL;
        int qx_owned = 0;
        if (n_threads <= 1 && scratch_buf != NULL && qx_size <= (size_t)scratch_size) {
            qx = (block_q8_0 *)scratch_buf;
        } else {
            qx = (block_q8_0 *)malloc(qx_size);
            qx_owned = 1;
        }
        if (qx != NULL) {
            quantize_row_q8_0(x, qx, n);

            /* Single-threaded: process all rows at once */
            if (n_threads <= 1 || d < 8) {
                vec_dot_q4_0x8_q8_0_avx2(wptr, qx, n, out, d);
                if (qx_owned) free(qx);
                return;
            }

            int nt = n_threads;
            if (nt > d) nt = d;

            {
                int rows_per = d / nt, extra = d % nt, tstart = 0;
                for (int t = 0; t < nt; t++) {
                    int tend = tstart + rows_per + (t < extra ? 1 : 0);
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q4_0_8_8;
                    pool_tasks[t].start = tstart; pool_tasks[t].end = tend;
                    tstart = tend;
                }
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_owned) free(qx);
            return;
        }
#endif
        /* Non-AVX2: fall through to generic vec_dot path */
    } else if (qtype == GGUF_TYPE_Q4_K) {
        /* Q4_K fast path: quantize x to Q8_K once, then vec_dot_q4_K_q8_K */
        /* Only enabled on x86 (AVX/AVX2). On NEON, use vec_dot_q4_K_f32 instead. */
        size_t qx_size = (n / 256) * sizeof(block_q8_K);
        block_q8_K *qx = NULL;
        int qx_owned = 0;  /* whether we malloc'd qx */
        /* When threading, always malloc to avoid data races */
        if (n_threads <= 1 && scratch_buf != NULL && qx_size <= (size_t)scratch_size) {
            qx = (block_q8_K *)scratch_buf;
        } else {
            qx = (block_q8_K *)malloc(qx_size);
            qx_owned = 1;
        }
        if (qx != NULL) {
            quantize_row_q8_K(x, qx, n);

            if (n_threads <= 1 || d < 4) {
                for (int i = 0; i < d; i++) {
                    out[i] = vec_dot_q4_K_q8_K(wptr + (size_t)i * row_bytes, qx, n);
                }
                if (qx_owned) free(qx);
                return;
            }

            int nt = n_threads;
            if (nt > d) nt = d;

            {
                int rows_per = d / nt, extra = d % nt, tstart = 0;
                for (int t = 0; t < nt; t++) {
                    int tend = tstart + rows_per + (t < extra ? 1 : 0);
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q4_K;
                    pool_tasks[t].start = tstart; pool_tasks[t].end = tend;
                    tstart = tend;
                }
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_owned) free(qx);
            return;
        }
        /* If allocation failed, fall through to generic path */
#endif
    }

    if (n_threads <= 1 || d < 4) {
        for (int i = 0; i < d; i++) {
            out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
        }
        return;
    }

    int nt = n_threads;
    if (nt > d) nt = d;

    {
        int rows_per = d / nt, extra = d % nt, tstart = 0;
        for (int t = 0; t < nt; t++) {
            int tend = tstart + rows_per + (t < extra ? 1 : 0);
            pool_tasks[t].out = out; pool_tasks[t].x = x;
            pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
            pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
            pool_tasks[t].qtype = qtype;
            pool_tasks[t].start = tstart; pool_tasks[t].end = tend;
            tstart = tend;
        }
        pool_init(nt);
        pool_wake(nt);
        matmul_worker_f(&pool_tasks[0]);
        pool_wait(nt);
    }
}

/* ================================================================
 * Batch matmul for prefill (weights read once for all tokens)
 * Output layout: out[batch * d + row]
 * ================================================================ */

void matmul_batch(float *out, const float *x, int n_batch,
                   const void *W, int n, int d, gguf_type_t qtype) {
    size_t row_bytes = gguf_type_row_size(qtype, n);
    const char *wptr = (const char *)W;
    for (int i = 0; i < d; i++) {
        const char *wrow = wptr + (size_t)i * row_bytes;
        for (int b = 0; b < n_batch; b++)
            out[b * d + i] = vec_dot(wrow, x + b * n, n, qtype);
    }
}

void matmul_dual_batch(float *out1, float *out2, const float *x, int n_batch,
                        const void *W1, const void *W2,
                        int n, int d, gguf_type_t qtype1, gguf_type_t qtype2) {
    size_t row_bytes = gguf_type_row_size(qtype1, n);
    size_t row_bytes2 = gguf_type_row_size(qtype2, n);
    const char *w1 = (const char *)W1;
    const char *w2 = (const char *)W2;
    for (int i = 0; i < d; i++) {
        const char *wr1 = w1 + (size_t)i * row_bytes;
        const char *wr2 = w2 + (size_t)i * row_bytes2;
        for (int b = 0; b < n_batch; b++) {
            const float *xb = x + b * n;
            out1[b * d + i] = vec_dot(wr1, xb, n, qtype1);
            out2[b * d + i] = vec_dot(wr2, xb, n, qtype2);
        }
    }
}

/* ================================================================
 * SIMD-accelerated basic operations
 * ================================================================ */

void rmsnorm(float *out, const float *x, const float *weight, int size, float eps) {
    float ss = 0.0f;

#ifdef PICOLM_NEON
    float32x4_t acc = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vmlaq_f32(acc, v, v);
    }
    ss = vaddvq_f32_compat(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#elif defined(PICOLM_AVX)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
    }
    ss = hsum_avx(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#elif defined(PICOLM_SSE2)
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
    }
    ss = hsum_sse(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#else
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
#endif

    ss = 1.0f / sqrtf(ss / (float)size + eps);

#ifdef PICOLM_NEON
    float32x4_t scale = vdupq_n_f32(ss);
    i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t w = vld1q_f32(weight + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, scale), w));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
#elif defined(PICOLM_AVX)
    __m256 scale = _mm256_set1_ps(ss);
    i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(v, scale), w));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
#elif defined(PICOLM_SSE2)
    __m128 scale = _mm_set1_ps(ss);
    i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        __m128 w = _mm_loadu_ps(weight + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_mul_ps(v, scale), w));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
#else
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
#endif
}

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;

#ifdef PICOLM_NEON
    float32x4_t inv_v = vdupq_n_f32(inv);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), inv_v));
    }
    for (; i < size; i++) x[i] *= inv;
#elif defined(PICOLM_AVX)
    __m256 inv_v = _mm256_set1_ps(inv);
    int i = 0;
    for (; i + 7 < size; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), inv_v));
    }
    for (; i < size; i++) x[i] *= inv;
#elif defined(PICOLM_SSE2)
    __m128 inv_v = _mm_set1_ps(inv);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(x + i, _mm_mul_ps(_mm_loadu_ps(x + i), inv_v));
    }
    for (; i < size; i++) x[i] *= inv;
#else
    for (int i = 0; i < size; i++) x[i] *= inv;
#endif
}

/* AVX RoPE: 4 complex pairs/iter; addsub handles r*cos-i*sin / r*sin+i*cos in one op */
#ifdef PICOLM_AVX
static void rope_avx(float *h, int half, const float *cos_pos, const float *sin_pos) {
    int i = 0;
    for (; i + 3 < half; i += 4) {
        __m256 v   = _mm256_loadu_ps(h + i * 2);
        __m128 c4  = _mm_loadu_ps(cos_pos + i);
        __m128 s4  = _mm_loadu_ps(sin_pos + i);
        __m256 cv  = _mm256_set_m128(_mm_unpackhi_ps(c4, c4), _mm_unpacklo_ps(c4, c4));
        __m256 sv  = _mm256_set_m128(_mm_unpackhi_ps(s4, s4), _mm_unpacklo_ps(s4, s4));
        __m256 sw  = _mm256_permute_ps(v, 0xB1); /* swap r,i within each pair */
        _mm256_storeu_ps(h + i * 2,
            _mm256_addsub_ps(_mm256_mul_ps(v, cv), _mm256_mul_ps(sw, sv)));
    }
    for (; i < half; i++) {
        float r = h[i * 2], im = h[i * 2 + 1];
        h[i * 2]     = r * cos_pos[i] - im * sin_pos[i];
        h[i * 2 + 1] = r * sin_pos[i] + im * cos_pos[i];
    }
}
#endif

/* SSE2/SSE3 RoPE: 2 pairs/iter; SSE3 uses addsub, SSE2 uses sign-mask to negate even lanes */
#if defined(PICOLM_SSE2) && !defined(PICOLM_AVX)
static void rope_sse(float *h, int half, const float *cos_pos, const float *sin_pos) {
    int i = 0;
#ifdef PICOLM_SSE3
    for (; i + 1 < half; i += 2) {
        __m128 v  = _mm_loadu_ps(h + i * 2);
        __m128 c2 = _mm_unpacklo_ps(_mm_load_ss(cos_pos + i), _mm_load_ss(cos_pos + i + 1));
        __m128 s2 = _mm_unpacklo_ps(_mm_load_ss(sin_pos + i), _mm_load_ss(sin_pos + i + 1));
        __m128 cv = _mm_shuffle_ps(c2, c2, _MM_SHUFFLE(1,1,0,0));
        __m128 sv = _mm_shuffle_ps(s2, s2, _MM_SHUFFLE(1,1,0,0));
        __m128 sw = _mm_shuffle_ps(v,  v,  _MM_SHUFFLE(2,3,0,1));
        _mm_storeu_ps(h + i * 2, _mm_addsub_ps(_mm_mul_ps(v, cv), _mm_mul_ps(sw, sv)));
    }
#else
    const __m128 sign = _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f);
    for (; i + 1 < half; i += 2) {
        __m128 v  = _mm_loadu_ps(h + i * 2);
        __m128 c2 = _mm_unpacklo_ps(_mm_load_ss(cos_pos + i), _mm_load_ss(cos_pos + i + 1));
        __m128 s2 = _mm_unpacklo_ps(_mm_load_ss(sin_pos + i), _mm_load_ss(sin_pos + i + 1));
        __m128 cv = _mm_shuffle_ps(c2, c2, _MM_SHUFFLE(1,1,0,0));
        __m128 sv = _mm_shuffle_ps(s2, s2, _MM_SHUFFLE(1,1,0,0));
        __m128 sw = _mm_shuffle_ps(v,  v,  _MM_SHUFFLE(2,3,0,1));
        __m128 a  = _mm_mul_ps(v, cv);
        __m128 b  = _mm_mul_ps(_mm_mul_ps(sign, sw), sv);
        _mm_storeu_ps(h + i * 2, _mm_add_ps(a, b));
    }
#endif
    for (; i < half; i++) {
        float r = h[i * 2], im = h[i * 2 + 1];
        h[i * 2]     = r * cos_pos[i] - im * sin_pos[i];
        h[i * 2 + 1] = r * sin_pos[i] + im * cos_pos[i];
    }
}
#endif

/* Rotary position encoding using pre-computed cos/sin tables */
void rope(float *q, float *k, int head_dim, int n_heads, int n_kv_heads,
          const float *cos_pos, const float *sin_pos, int rope_type) {
    int half = head_dim / 2;

    if (rope_type) {
        /* Qwen2 interleaved style: q[i] and q[i+half] are paired */
        for (int h = 0; h < n_heads; h++) {
            float *qh = q + h * head_dim;
            for (int i = 0; i < half; i++) {
                float q0 = qh[i], q1 = qh[i + half];
                qh[i]     = q0 * cos_pos[i] - q1 * sin_pos[i];
                qh[i + half] = q0 * sin_pos[i] + q1 * cos_pos[i];
            }
        }
        for (int h = 0; h < n_kv_heads; h++) {
            float *kh = k + h * head_dim;
            for (int i = 0; i < half; i++) {
                float k0 = kh[i], k1 = kh[i + half];
                kh[i]     = k0 * cos_pos[i] - k1 * sin_pos[i];
                kh[i + half] = k0 * sin_pos[i] + k1 * cos_pos[i];
            }
        }
    } else {
        /* Llama pairwise style: (q[2i], q[2i+1]) paired */
        for (int h = 0; h < n_heads; h++) {
            float *qh = q + h * head_dim;
#ifdef PICOLM_NEON
            int i = 0;
            for (; i + 3 < half; i += 4) {
                float32x4x2_t qv = vld2q_f32(qh + i * 2);
                float32x4_t cv = vld1q_f32(cos_pos + i);
                float32x4_t sv = vld1q_f32(sin_pos + i);
                float32x4_t new_even = vmlsq_f32(vmulq_f32(qv.val[0], cv), qv.val[1], sv);
                float32x4_t new_odd  = vmlaq_f32(vmulq_f32(qv.val[0], sv), qv.val[1], cv);
                float32x4x2_t result = {{ new_even, new_odd }};
                vst2q_f32(qh + i * 2, result);
            }
            for (; i < half; i++) {
                float q0 = qh[i * 2], q1 = qh[i * 2 + 1];
                qh[i * 2]     = q0 * cos_pos[i] - q1 * sin_pos[i];
                qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
            }
#elif defined(PICOLM_AVX)
            rope_avx(qh, half, cos_pos, sin_pos);
#elif defined(PICOLM_SSE2)
            rope_sse(qh, half, cos_pos, sin_pos);
#else
            for (int i = 0; i < half; i++) {
                float q0 = qh[i * 2], q1 = qh[i * 2 + 1];
                qh[i * 2]     = q0 * cos_pos[i] - q1 * sin_pos[i];
                qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
            }
#endif
        }
        for (int h = 0; h < n_kv_heads; h++) {
            float *kh = k + h * head_dim;
#ifdef PICOLM_NEON
            int i = 0;
            for (; i + 3 < half; i += 4) {
                float32x4x2_t kv = vld2q_f32(kh + i * 2);
                float32x4_t cv = vld1q_f32(cos_pos + i);
                float32x4_t sv = vld1q_f32(sin_pos + i);
                float32x4_t new_even = vmlsq_f32(vmulq_f32(kv.val[0], cv), kv.val[1], sv);
                float32x4_t new_odd  = vmlaq_f32(vmulq_f32(kv.val[0], sv), kv.val[1], cv);
                float32x4x2_t result = {{ new_even, new_odd }};
                vst2q_f32(kh + i * 2, result);
            }
            for (; i < half; i++) {
                float k0 = kh[i * 2], k1 = kh[i * 2 + 1];
                kh[i * 2]     = k0 * cos_pos[i] - k1 * sin_pos[i];
                kh[i * 2 + 1] = k0 * sin_pos[i] + k1 * cos_pos[i];
            }
#elif defined(PICOLM_AVX)
            rope_avx(kh, half, cos_pos, sin_pos);
#elif defined(PICOLM_SSE2)
            rope_sse(kh, half, cos_pos, sin_pos);
#else
            for (int i = 0; i < half; i++) {
                float k0 = kh[i * 2], k1 = kh[i * 2 + 1];
                kh[i * 2]     = k0 * cos_pos[i] - k1 * sin_pos[i];
                kh[i * 2 + 1] = k0 * sin_pos[i] + k1 * cos_pos[i];
            }
#endif
        }
    }
}

void silu(float *x, int size) {
    /* swish/silu: x / (1 + exp(-x))
     * No hardware exp instruction in SSE/AVX, so use scalar expf.
     * Vectorized expf exists in libsvml (Intel) or libxsmm, but we
     * avoid those dependencies. The compiler may auto-vectorize
     * expf with -ffast-math. */
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void elemwise_mul(float *out, const float *a, const float *b, int size) {
#ifdef PICOLM_NEON
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < size; i++) out[i] = a[i] * b[i];
#elif defined(PICOLM_AVX)
    int i = 0;
    for (; i + 7 < size; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < size; i++) out[i] = a[i] * b[i];
#elif defined(PICOLM_SSE2)
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    for (; i < size; i++) out[i] = a[i] * b[i];
#else
    for (int i = 0; i < size; i++) out[i] = a[i] * b[i];
#endif
}

void vec_add(float *a, const float *b, int size) {
#ifdef PICOLM_NEON
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(a + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < size; i++) a[i] += b[i];
#elif defined(PICOLM_AVX)
    int i = 0;
    for (; i + 7 < size; i += 8) {
        _mm256_storeu_ps(a + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < size; i++) a[i] += b[i];
#elif defined(PICOLM_SSE2)
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(a + i, _mm_add_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    for (; i < size; i++) a[i] += b[i];
#else
    for (int i = 0; i < size; i++) a[i] += b[i];
#endif
}
