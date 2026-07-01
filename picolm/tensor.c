#include "tensor.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* ---- Scratch buffer (kept for dequantize_row in model.c) ---- */

static float *scratch_buf = NULL;
static int    scratch_size = 0;

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

#ifndef _WIN32

/* Persistent thread pool -- correct implementation.
 *
 * Design:
 *   - Workers loop: wait on tp_cond until tp_tasks >= my_idx (their turn).
 *   - Dispatcher: fills tasks, signals workers via tp_cond, waits on tp_done
 *     until all workers have completed (tp_active == 0).
 *   - The critical fix: workers must signal tp_done when ALL are done, not just
 *     the last one. Using a counter + broadcast ensures no missed signals.
 */
static pthread_t       tp_threads[MAX_THREADS];
static pthread_mutex_t tp_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  tp_cond  = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  tp_done  = PTHREAD_COND_INITIALIZER;
static volatile int    tp_nt    = 0;
static volatile int    tp_stop  = 0;
static volatile int    tp_tasks = 0;
static volatile int    tp_active = 0;
static matmul_task_t   tp_tasks_arr[MAX_THREADS];

/* Core worker logic */
static void matmul_worker_f(matmul_task_t *t) {
    if (t->qtype == GGUF_TYPE_Q8_0 && t->x) {
        const block_q8_0 *qx = (const block_q8_0 *)t->x;
        const float *qx_d = t->x_d;
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot_q8_0_q8_0_deltas(qx, qx_d,
                                                  t->W + (size_t)i * t->row_bytes, t->n);
        }
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
    } else {
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot(t->W + (size_t)i * t->row_bytes,
                                t->x, t->n, t->qtype);
        }
    }
}

/* Worker thread: waits for task, executes it, signals completion */
static void *tp_worker(void *arg) {
    int my_idx = *(int *)arg;
    for (;;) {
        pthread_mutex_lock(&tp_mutex);
        /* Wait until my task is ready or we're stopping */
        while (tp_tasks <= my_idx && !tp_stop) {
            pthread_cond_wait(&tp_cond, &tp_mutex);
        }
        if (tp_stop) {
            pthread_mutex_unlock(&tp_mutex);
            break;
        }
        tp_active++;
        /* Signal dispatcher: a worker has started */
        pthread_cond_signal(&tp_done);
        pthread_mutex_unlock(&tp_mutex);

        /* Execute task */
        matmul_worker_f(&tp_tasks_arr[my_idx]);

        pthread_mutex_lock(&tp_mutex);
        tp_active--;
        /* Signal dispatcher: a worker has completed */
        pthread_cond_signal(&tp_done);
        pthread_mutex_unlock(&tp_mutex);
    }
    return NULL;
}

void tensor_threadpool_init(int n_threads) {
    if (tp_nt > 0) return;
    int nw = n_threads - 1;
    if (nw <= 0) return;
    tp_stop = 0;
    tp_active = 0;
    tp_tasks = 0;
    for (int i = 0; i < nw; i++) {
        int *idx = malloc(sizeof(int));
        *idx = i + 1;
        pthread_create(&tp_threads[i], NULL, tp_worker, idx);
    }
    tp_nt = nw;
}

void tensor_threadpool_free(void) {
    if (tp_nt <= 0) return;
    tp_stop = 1;
    pthread_cond_broadcast(&tp_cond);
    for (int i = 0; i < tp_nt; i++) {
        pthread_join(tp_threads[i], NULL);
    }
    tp_nt = 0;
}

/* Set task for worker thread idx */
static void tp_set_task(int idx, float *out, const float *x, const float *x_d,
                        const char *W, size_t row_bytes, int n,
                        int start, int end, gguf_type_t qtype) {
    tp_tasks_arr[idx].out       = out;
    tp_tasks_arr[idx].x         = x;
    tp_tasks_arr[idx].x_d       = x_d;
    tp_tasks_arr[idx].W         = W;
    tp_tasks_arr[idx].row_bytes = row_bytes;
    tp_tasks_arr[idx].n         = n;
    tp_tasks_arr[idx].start     = start;
    tp_tasks_arr[idx].end       = end;
    tp_tasks_arr[idx].qtype     = qtype;
}

/* Dispatch: signal workers, wait for all to complete.
 *
 * Two-phase barrier:
 * Phase 1: Signal workers, wait until all have started (tp_active == nt-1)
 * Phase 2: Reset tp_tasks so workers know they're done, wait until all complete
 */
static void tp_dispatch(int nt) {
    /* Safety: cap nt to available workers + 1 (main thread) */
    if (nt > tp_nt + 1) nt = tp_nt + 1;
    if (nt <= 1) return;  /* No workers to dispatch to */
    pthread_mutex_lock(&tp_mutex);
    tp_active = 0;
    tp_tasks = nt;
    /* Signal all workers */
    pthread_cond_broadcast(&tp_cond);
    /* Phase 1: wait until all nt-1 workers have started */
    while (tp_active < nt - 1) {
        pthread_cond_wait(&tp_done, &tp_mutex);
    }
    /* All workers are running. Reset tp_tasks so workers can stop after
     * completing their current task. */
    tp_tasks = 0;
    /* Phase 2: wait until all workers have completed */
    while (tp_active > 0) {
        pthread_cond_wait(&tp_done, &tp_mutex);
    }
    pthread_mutex_unlock(&tp_mutex);
}

#else

static void matmul_worker_f(matmul_task_t *t) {
    if (t->qtype == GGUF_TYPE_Q8_0 && t->x) {
        const block_q8_0 *qx = (const block_q8_0 *)t->x;
        const float *qx_d = t->x_d;
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot_q8_0_q8_0_deltas(qx, qx_d,
                                                  t->W + (size_t)i * t->row_bytes, t->n);
        }
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
    } else {
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot(t->W + (size_t)i * t->row_bytes,
                                t->x, t->n, t->qtype);
        }
    }
}

/* Windows thread entry point - CreateThread requires DWORD WINAPI signature */
static DWORD WINAPI matmul_worker(LPVOID lpParam) {
    matmul_worker_f((matmul_task_t *)lpParam);
    return 0;
}

void tensor_threadpool_init(int n_threads) {}
void tensor_threadpool_free(void) {}

#endif

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

            /* Threaded Q8_0 path via persistent pool */
            int nt = n_threads;
            if (nt > d) nt = d;

            int rows_per = d / nt;
            int extra = d % nt;
            int row = rows_per + (0 < extra ? 1 : 0);

            for (int i = 0; i < row; i++) {
                out[i] = vec_dot_q8_0_q8_0_deltas(qx, qx_d, wptr + (size_t)i * row_bytes, n);
            }

            if (nt > 1 && d >= matmul_min_rows) {
#ifdef _WIN32
                HANDLE threads[MAX_THREADS];
                matmul_task_t tasks[MAX_THREADS];
                row = rows_per + (0 < extra ? 1 : 0);
                for (int t = 1; t < nt; t++) {
                    int tstart = row;
                    row += rows_per + (t < extra ? 1 : 0);
                    tasks[t].out = out; tasks[t].x = (const float *)qx;
                    tasks[t].x_d = qx_d; tasks[t].W = wptr;
                    tasks[t].row_bytes = row_bytes; tasks[t].n = n;
                    tasks[t].qtype = GGUF_TYPE_Q8_0;
                    tasks[t].start = tstart; tasks[t].end = row;
                    threads[t] = CreateThread(NULL, 0, matmul_worker, &tasks[t], 0, NULL);
                }
                for (int t = 1; t < nt; t++) {
                    WaitForSingleObject(threads[t], INFINITE);
                    CloseHandle(threads[t]);
                }
#else
                for (int t = 1; t < nt; t++) {
                    int tstart = row;
                    row += rows_per + (t < extra ? 1 : 0);
                    tp_set_task(t, out, (const float *)qx, qx_d, wptr, row_bytes, n,
                                tstart, row, GGUF_TYPE_Q8_0);
                }
                tp_dispatch(nt);
#endif
            } else {
                /* Threading disabled (too few rows or single-thread): compute remaining rows */
                for (int i = row; i < d; i++) {
                    out[i] = vec_dot_q8_0_q8_0_deltas(qx, qx_d, wptr + (size_t)i * row_bytes, n);
                }
            }

            if (qx_buf != scratch_buf) free(qx_buf);
            return;
        }
        /* If allocation failed, fall through to generic path */
#if defined(PICOLM_AVX2) || defined(PICOLM_AVX)
    } else if (qtype == GGUF_TYPE_Q4_0) {
        /* Q4_0 fast path: quantize x to Q8_0 once, then vec_dot_q4_0_q8_0 */
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

            if (n_threads <= 1 || d < 4) {
                for (int i = 0; i < d; i++) {
                    out[i] = vec_dot_q4_0_q8_0(wptr + (size_t)i * row_bytes, qx, n);
                }
                if (qx_owned) free(qx);
                return;
            }

            int nt = n_threads;
            if (nt > d) nt = d;

            int rows_per = d / nt;
            int extra = d % nt;
            int row = rows_per + (0 < extra ? 1 : 0);

            for (int i = 0; i < row; i++) {
                out[i] = vec_dot_q4_0_q8_0(wptr + (size_t)i * row_bytes, qx, n);
            }

            if (nt > 1 && d >= matmul_min_rows) {
#ifdef _WIN32
                HANDLE threads[MAX_THREADS];
                matmul_task_t tasks[MAX_THREADS];
                row = rows_per + (0 < extra ? 1 : 0);
                for (int t = 1; t < nt; t++) {
                    int tstart = row;
                    row += rows_per + (t < extra ? 1 : 0);
                    tasks[t].out = out; tasks[t].x = (const float *)qx; tasks[t].x_d = NULL;
                    tasks[t].W = wptr; tasks[t].row_bytes = row_bytes; tasks[t].n = n;
                    tasks[t].qtype = GGUF_TYPE_Q4_0; tasks[t].start = tstart; tasks[t].end = row;
                    threads[t] = CreateThread(NULL, 0, matmul_worker, &tasks[t], 0, NULL);
                }
                for (int t = 1; t < nt; t++) {
                    WaitForSingleObject(threads[t], INFINITE);
                    CloseHandle(threads[t]);
                }
#else
                for (int t = 1; t < nt; t++) {
                    int tstart = row;
                    row += rows_per + (t < extra ? 1 : 0);
                    tp_set_task(t, out, (const float *)qx, NULL, wptr, row_bytes, n,
                                tstart, row, GGUF_TYPE_Q4_0);
                }
                tp_dispatch(nt);
#endif
            } else {
                for (int i = row; i < d; i++) {
                    out[i] = vec_dot_q4_0_q8_0(wptr + (size_t)i * row_bytes, qx, n);
                }
            }

            if (qx_owned) free(qx);
            return;
        }
        /* If allocation failed, fall through to generic path */
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

            int rows_per = d / nt;
            int extra = d % nt;
            int row = rows_per + (0 < extra ? 1 : 0);

            for (int i = 0; i < row; i++) {
                out[i] = vec_dot_q4_K_q8_K(wptr + (size_t)i * row_bytes, qx, n);
            }

            if (nt > 1 && d >= matmul_min_rows) {
#ifdef _WIN32
                HANDLE threads[MAX_THREADS];
                matmul_task_t tasks[MAX_THREADS];
                row = rows_per + (0 < extra ? 1 : 0);
                for (int t = 1; t < nt; t++) {
                    int tstart = row;
                    row += rows_per + (t < extra ? 1 : 0);
                    tasks[t].out = out; tasks[t].x = (const float *)qx; tasks[t].x_d = NULL;
                    tasks[t].W = wptr; tasks[t].row_bytes = row_bytes; tasks[t].n = n;
                    tasks[t].qtype = GGUF_TYPE_Q4_K; tasks[t].start = tstart; tasks[t].end = row;
                    threads[t] = CreateThread(NULL, 0, matmul_worker, &tasks[t], 0, NULL);
                }
                for (int t = 1; t < nt; t++) {
                    WaitForSingleObject(threads[t], INFINITE);
                    CloseHandle(threads[t]);
                }
#else
                for (int t = 1; t < nt; t++) {
                    int tstart = row;
                    row += rows_per + (t < extra ? 1 : 0);
                    tp_set_task(t, out, (const float *)qx, NULL, wptr, row_bytes, n,
                                tstart, row, GGUF_TYPE_Q4_K);
                }
                tp_dispatch(nt);
#endif
            } else {
                for (int i = row; i < d; i++) {
                    out[i] = vec_dot_q4_K_q8_K(wptr + (size_t)i * row_bytes, qx, n);
                }
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

    int rows_per = d / nt;
    int extra = d % nt;
    int row = rows_per + (0 < extra ? 1 : 0);

    for (int i = 0; i < row; i++) {
        out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
    }

    if (nt > 1 && d >= matmul_min_rows) {
#ifdef _WIN32
        HANDLE threads[MAX_THREADS];
        matmul_task_t tasks[MAX_THREADS];
        row = rows_per + (0 < extra ? 1 : 0);
        for (int t = 1; t < nt; t++) {
            int tstart = row;
            row += rows_per + (t < extra ? 1 : 0);
            tasks[t].out = out; tasks[t].x = x; tasks[t].x_d = NULL;
            tasks[t].W = wptr; tasks[t].row_bytes = row_bytes; tasks[t].n = n;
            tasks[t].qtype = qtype; tasks[t].start = tstart; tasks[t].end = row;
            threads[t] = CreateThread(NULL, 0, matmul_worker, &tasks[t], 0, NULL);
        }
        for (int t = 1; t < nt; t++) {
            WaitForSingleObject(threads[t], INFINITE);
            CloseHandle(threads[t]);
        }
#else
        for (int t = 1; t < nt; t++) {
            int tstart = row;
            row += rows_per + (t < extra ? 1 : 0);
            tp_set_task(t, out, x, NULL, wptr, row_bytes, n, tstart, row, qtype);
        }
        tp_dispatch(nt);
#endif
    } else {
        for (int i = row; i < d; i++) {
            out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
        }
    }
}

/* ================================================================
 * SIMD-accelerated basic operations
 * ================================================================ */

void rmsnorm(float *out, const float *x, const float *weight, int size) {
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

    ss = 1.0f / sqrtf(ss / (float)size + 1e-5f);

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
          const float *cos_pos, const float *sin_pos) {
    int half = head_dim / 2;

    /* Apply RoPE to all query heads */
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
#ifdef PICOLM_NEON
        int i = 0;
        for (; i + 3 < half; i += 4) {
            /* Load pairs: (q0,q1), (q2,q3), ... as interleaved */
            float32x4x2_t qv = vld2q_f32(qh + i * 2);
            float32x4_t cv = vld1q_f32(cos_pos + i);
            float32x4_t sv = vld1q_f32(sin_pos + i);
            /* q_even = q0*cos - q1*sin, q_odd = q0*sin + q1*cos */
            float32x4_t new_even = vmlsq_f32(vmulq_f32(qv.val[0], cv), qv.val[1], sv);
            float32x4_t new_odd  = vmlaq_f32(vmulq_f32(qv.val[0], sv), qv.val[1], cv);
            float32x4x2_t result = {{ new_even, new_odd }};
            vst2q_f32(qh + i * 2, result);
        }
        for (; i < half; i++) {
            float q0 = qh[i * 2];
            float q1 = qh[i * 2 + 1];
            qh[i * 2]     = q0 * cos_pos[i] - q1 * sin_pos[i];
            qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
        }
#elif defined(PICOLM_AVX)
        rope_avx(qh, half, cos_pos, sin_pos);
#elif defined(PICOLM_SSE2)
        rope_sse(qh, half, cos_pos, sin_pos);
#else
        for (int i = 0; i < half; i++) {
            float q0 = qh[i * 2];
            float q1 = qh[i * 2 + 1];
            qh[i * 2]     = q0 * cos_pos[i] - q1 * sin_pos[i];
            qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
        }
#endif
    }

    /* Apply RoPE to all KV heads */
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
#ifdef PICOLM_AVX
        rope_avx(kh, half, cos_pos, sin_pos);
#elif defined(PICOLM_SSE2)
        rope_sse(kh, half, cos_pos, sin_pos);
#else
        for (int i = 0; i < half; i++) {
            float k0 = kh[i * 2];
            float k1 = kh[i * 2 + 1];
            kh[i * 2]     = k0 * cos_pos[i] - k1 * sin_pos[i];
            kh[i * 2 + 1] = k0 * sin_pos[i] + k1 * cos_pos[i];
        }
#endif
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
