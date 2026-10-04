#include <assert.h>
#include "tensor.h"
#include "sgemm.h"
/* Forward decl for NEON GEMM task wrapper */
#if defined(PICOLM_NEON)
extern int sgemm_iq2_k_q8_k_neon(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
extern int sgemm_iq3_k_q8_k_neon(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
extern int sgemm_iq4_k_q8_k_avx2(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
extern int sgemm_iq4_k_q8_k_neon(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
extern int sgemm_q8_k_r8_q8_k_neon(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
extern void vec_dot_q8_k_r8_q8_k_neon(const void *vx, const void *wy, int n, float *out, int nrows);
/* NEON R4 GEMV declarations */
extern int sgemm_iq6_k_q8_k_neon(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
extern void vec_dot_iq6_k_q8_k_neon(const void *vx, const void *wy, int n, float *out);
extern void vec_dot_q4_k_r4_q8_k_neon(const void *vx, const void *wy, int n, float *out, int nrows);
#endif
/* AVX2 GEMM declarations */
#if defined(PICOLM_AVX2)
extern int sgemm_iq6_k_q8_k_avx2(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
extern int sgemm_q4_k_r4_q8_k_avx2(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
extern void vec_dot_iq2_k_r4_q8_k_neon(const void *vx, const void *wy, int n, float *out, int nrows);
extern void vec_dot_iq3_k_r4_q8_k_neon(const void *vx, const void *wy, int n, float *out, int nrows);
extern void vec_dot_iq4_k_r4_q8_k_neon(const void *vx, const void *wy, int n, float *out, int nrows);
#endif
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef WINVER
#define WINVER 0x0600
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <windows.h>
#include <synchapi.h>
#elif !defined(PICOLM_DOS)
#include <pthread.h>
#include <unistd.h>
#include <dirent.h>
#include <time.h>
#include <stdlib.h>
#endif

#ifdef PICOLM_GPU
#include "backend_gpu.h"
/* GPU dispatch: if a GPU tensor handle is registered for this weight matrix,
 * offload to GPU. The handle is stored per-tensor via tensor_set_gpu_tensor().
 * This is checked at the very start of matmul/matmul_batch before any CPU path.
 *
 * Load-bearing invariant: gpu_tensor/gpu_device are accessed ONLY from the
 * orchestrator thread (the same thread that calls model_forward/model_forward_prefill).
 * All task functions passed to tensor_parallel_for must NOT call matmul/matmul_batch.
 *
 * We record the orchestrator thread ID and abort if any GPU dispatch
 * or tensor_set_gpu_tensor call originates from a different thread. This catches
 * silent corruption (wrong weight matrix used) by converting it into a loud crash. */
static picolm_gpu_tensor_t *gpu_tensor = NULL;
static int gpu_device = 0;
#ifdef _WIN32
static DWORD gpu_orchestrator_thread = 0;
#else
static pthread_t gpu_orchestrator_thread = 0;
#endif
static int gpu_orchestrator_set = 0;

static void gpu_assert_orchestrator(const char *fn) {
    if (!gpu_orchestrator_set) return;
#ifdef _WIN32
    if (GetCurrentThreadId() != gpu_orchestrator_thread) {
#else
    if (pthread_self() != gpu_orchestrator_thread) {
#endif
        fprintf(stderr, "FATAL: GPU %s called from non-orchestrator thread (corruption risk)\n", fn);
        abort();
    }
}

void tensor_set_gpu_tensor(picolm_gpu_tensor_t *t, int device) {
    if (!gpu_orchestrator_set) {
#ifdef _WIN32
        gpu_orchestrator_thread = GetCurrentThreadId();
#else
        gpu_orchestrator_thread = pthread_self();
#endif
        gpu_orchestrator_set = 1;
    }
    gpu_assert_orchestrator("tensor_set_gpu_tensor");
    gpu_tensor = t;
    gpu_device = device;
}
#endif /* PICOLM_GPU */

/* Mac OS X < 10.5 compat: _SC_NPROCESSORS_ONLN may not be defined */
#if defined(__APPLE__) && !defined(_SC_NPROCESSORS_ONLN)
#include <sys/types.h>
#include <sys/sysctl.h>
#define _SC_NPROCESSORS_ONLN 99
static int sysconf_compat(int name) {
    if (name == _SC_NPROCESSORS_ONLN) {
        int mib[2] = { CTL_HW, HW_NCPU };
        int ncpu; size_t len = sizeof(ncpu);
        if (sysctl(mib, 2, &ncpu, &len, NULL, 0) == 0) return ncpu;
    }
    return 1;
}
#define sysconf(n) sysconf_compat(n)
#endif

/* Thread pool (Windows-native or POSIX) */
#ifdef _WIN32
typedef HANDLE            win_thread_t;
typedef SRWLOCK           win_mutex_t;
typedef CONDITION_VARIABLE win_cond_t;
#elif defined(PICOLM_DOS)
/* DOS: no threading. Stubs for compilation. */
typedef int               win_thread_t;
typedef int               win_mutex_t;
typedef int               win_cond_t;
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

int tensor_get_n_threads(void) { return n_threads; }
void tensor_set_n_threads(int n) { n_threads = n; }

/* ---- Physical core enumeration ---- */

/* ---- big.LITTLE CPU group detection (Linux only) ---- */

#ifndef _WIN32

typedef struct {
    int ids[256];        /* CPU logical IDs in this group */
    int nids;           /* number of IDs */
    uint32_t min_freq;  /* kHz (cpufreq units) */
    uint32_t max_freq;  /* kHz */
} cpu_group_t;

/* Detect CPU core groups from cpufreq topology.
 * Returns number of groups found (0 = fallback to single group).
 * Groups are sorted by max_freq ascending (little first, big last). */
static int detect_cpu_groups(cpu_group_t groups[], int max_groups) {
#if !defined(PICOLM_DOS)
    DIR *root = opendir("/sys/devices/system/cpu/cpufreq");
    if (!root) return 0;

    int ng = 0;
    struct dirent *ent;
    while ((ent = readdir(root)) != NULL && ng < max_groups) {
        /* Match policy0, policy1, etc. */
        if (ent->d_name[0] != 'p') continue;
        if (strlen(ent->d_name) > 120) continue; /* sanity limit */
        char dir[256];
        /* Prefix is 37 chars + max 120 from d_name = 157 < 256, no truncation */
        snprintf(dir, sizeof(dir), "/sys/devices/system/cpu/cpufreq/%.218s", ent->d_name);

        cpu_group_t *g = &groups[ng];
        g->nids = 0;
        g->min_freq = 0;
        g->max_freq = 0;

        /* Read affected_cpus */
        {
            char path[384];
            snprintf(path, sizeof(path), "%s/affected_cpus", dir);
            FILE *f = fopen(path, "r");
            if (f) {
                char buf[256] = {0};
                if (fgets(buf, sizeof(buf), f)) {
                    char *tok = strtok(buf, ", \t\n");
                    while (tok && g->nids < 256) {
                        g->ids[g->nids++] = atoi(tok);
                        tok = strtok(NULL, ", \t\n");
                    }
                }
                fclose(f);
            }
        }

        /* Read cpuinfo_min_freq and cpuinfo_max_freq (in kHz) */
        {
            char path[384];
            FILE *f;
            snprintf(path, sizeof(path), "%s/cpuinfo_min_freq", dir);
            f = fopen(path, "r");
            if (f) { fscanf(f, "%u", &g->min_freq); fclose(f); }
            snprintf(path, sizeof(path), "%s/cpuinfo_max_freq", dir);
            f = fopen(path, "r");
            if (f) { fscanf(f, "%u", &g->max_freq); fclose(f); }
        }

        if (g->nids > 0) ng++;
    }
    closedir(root);
#else
    return 0;
#endif

#if !defined(PICOLM_DOS)
    /* Sort groups by max_freq ascending (bubble sort, small N) */
    for (int i = 0; i < ng - 1; i++) {
        for (int j = i + 1; j < ng; j++) {
            if (groups[j].max_freq < groups[i].max_freq) {
                cpu_group_t tmp = groups[i];
                groups[i] = groups[j];
                groups[j] = tmp;
            }
        }
    }

    /* Merge groups with same max_freq */
    int merged = 0;
    for (int i = 0; i < ng; i++) {
        if (merged > 0 && groups[merged-1].max_freq == groups[i].max_freq) {
            cpu_group_t *prev = &groups[merged-1];
            for (int k = 0; k < groups[i].nids; k++) {
                if (prev->nids < 256) prev->ids[prev->nids++] = groups[i].ids[k];
            }
            if (groups[i].min_freq < prev->min_freq) prev->min_freq = groups[i].min_freq;
        } else {
            if (merged > 0 && merged != i) {
                groups[merged] = groups[i];
            }
            merged++;
        }
    }
    return merged;
#else
    return 0;
#endif
}

#endif /* _WIN32 */

/* Count physical CPU cores (excluding hyperthread siblings).
 * Linux: parses /sys/devices/system/cpu/ topology files.
 * Windows: uses GetLogicalProcessorInformation with RelationProcessorCore.
 * Fallback: sysconf(_SC_NPROCESSORS_ONLN) on POSIX, GetProcessorCount() on Win. */
static int count_physical_cores(void) {
#ifdef _WIN32
    {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf = NULL;
        DWORD len = 0;
        GetLogicalProcessorInformation(NULL, &len);
        buf = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(len);
        if (buf && GetLogicalProcessorInformation(buf, &len)) {
            int cores = 0;
            for (DWORD i = 0; i < len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); i++) {
                if (buf[i].Relationship == RelationProcessorCore) {
                    cores++;
                }
            }
            free(buf);
            if (cores > 0) return cores;
        } else if (buf) {
            free(buf);
        }
        /* Fallback: total logical processors (works on Win7+) */
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (int)si.dwNumberOfProcessors;
    }
#elif defined(PICOLM_DOS)
    /* DOS: single thread */
    return 1;
#else /* Linux / POSIX */
    {
        /* Count physical cores from /sys topology files.
         * Each /sys/devices/system/cpu/cpuN/topology/core_id gives the core
         * number for that logical CPU. Unique count = physical cores. */
        {
            int max_cpus = 256;
            int *core_ids = (int *)calloc(max_cpus, sizeof(int));
            if (!core_ids) return (int)sysconf(_SC_NPROCESSORS_ONLN);
            int n_cpus = 0;
            for (int i = 0; i < max_cpus; i++) {
                char path[128];
                snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
                FILE *cf = fopen(path, "r");
                if (!cf) break;
                int cid;
                if (fscanf(cf, "%d", &cid) == 1) {
                    core_ids[n_cpus++] = cid;
                }
                fclose(cf);
            }
            /* Count unique core_ids */
            int unique = 0;
            for (int i = 0; i < n_cpus; i++) {
                int found = 0;
                for (int j = 0; j < unique; j++) {
                    if (core_ids[i] == core_ids[j]) { found = 1; break; }
                }
                if (!found) {
                    core_ids[unique++] = core_ids[i];
                }
            }
            free(core_ids);
            if (unique > 0) return unique;
        }
        /* Fallback */
        return (int)sysconf(_SC_NPROCESSORS_ONLN);
    }
#endif
}

void tensor_set_threads(int t) {
    if (t < 1) t = 1;
    if (t > MAX_THREADS) t = MAX_THREADS;
    n_threads = t;
}

/* Return the default thread count based on physical core enumeration.
 * Uses only physical cores (no HT siblings) for generation performance.
 * On big.LITTLE systems, prefers big cores only. */
int tensor_default_threads(void) {
    int cores = count_physical_cores();
    if (cores < 1) cores = 4; /* fallback */

#ifndef _WIN32
    /* Try to detect big.LITTLE groups. If found, prefer big cores only. */
    cpu_group_t groups[256];
    int ng = detect_cpu_groups(groups, 256);
    if (ng >= 2) {
        /* Multiple groups detected: use only the fastest (last group = big cores) */
        int big_cores = groups[ng - 1].nids;
        fprintf(stderr, "CPU: detected %d groups, using %d big cores (skipping %d little)\n",
                ng, big_cores, cores - big_cores);
        cores = big_cores;
        /* Cap default threads on big.LITTLE to avoid over-subscription on decode
         * workloads. The thread pool's broadcast-wait model has O(n_threads)
         * overhead per matmul. For decode (S=1), each layer has ~8 matmuls
         * with 4096-14336 output rows. Empirical sweet spot: 4-8 threads.
         * Users who need more threads for prefill can use -j explicitly. */
        if (cores > 6) cores = 6;
    }
#endif

    if (cores > MAX_THREADS) cores = MAX_THREADS;
    return cores;
}

/* Return the number of big cores on a big.LITTLE system, or total physical
 * cores if no big.LITTLE detected. Returns 0 if detection fails. */
int tensor_get_big_cores(void) {
#ifdef _WIN32
    int cores = count_physical_cores();
    return (cores > 0) ? cores : 0;
#else
    cpu_group_t groups[256];
    int ng = detect_cpu_groups(groups, 256);
    if (ng >= 2) {
        return groups[ng - 1].nids;
    }
    int cores = count_physical_cores();
    return (cores > 0) ? cores : 0;
#endif
}

/* Threshold: skip threading if output vector is smaller than this.
 * Avoids mutex overhead for tiny matmuls. */
static int matmul_min_rows = 256;  /* lowered from 1024 to cover d=512 (MoE shared expert) */

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
    int          x_is_q8k; /* 1 if x points to Q8_K activations (for IQ2_K/IQ3_K) */
    const char  *W;
    size_t       row_bytes;
    int          n;        /* input dimension */
    int          d;        /* output dimension (for batched: output stride) */
    int          start;    /* first output row */
    int          end;      /* one past last output row */
    gguf_type_t  qtype;
    int          n_batch;  /* batch count (0 = single, >0 = batched) */
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

/* Generic task type: lets the same worker threads run an arbitrary
 * per-index function (e.g. per-attention-head work), not just matmul rows.
 * pool_mode selects which task array/dispatcher a woken worker should use. */
typedef struct {
    void (*fn)(int idx, void *ctx);
    void *ctx;
    int   start, end;
} generic_task_t;

static generic_task_t  generic_tasks[MAX_THREADS];
static volatile int    pool_mode = 0; /* 0 = matmul_task_t, 1 = generic_task_t */
static int pool_total_threads(int requested);
#if defined(_WIN32) || defined(__APPLE__) || defined(__osf__)
/* MSVC, old Mac OS X, OSF/1 Alpha (GCC 3.x) don't support __thread */
static int             pool_my_tid = 0;
#else
static __thread int    pool_my_tid = 0;
#endif /* TLS: current thread's pool index (0 for main) */

/* Return current thread's pool index (0 for main thread, 1..n-1 for workers) */
int tensor_get_thread_id(void) {
    return pool_my_tid;
}

static void generic_worker_f(generic_task_t *t) {
    for (int i = t->start; i < t->end; i++) t->fn(i, t->ctx);
}

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
#elif defined(PICOLM_DOS)
static void win_mutex_init(win_mutex_t *m) { (void)m; }
static void win_mutex_lock(win_mutex_t *m) { (void)m; }
static void win_mutex_unlock(win_mutex_t *m) { (void)m; }
static void win_cond_init(win_cond_t *c) { (void)c; }
static void win_cond_wait(win_cond_t *c, win_mutex_t *m) { (void)c; (void)m; }
static void win_cond_broadcast(win_cond_t *c) { (void)c; }
static void win_cond_destroy(win_cond_t *c) { (void)c; }
static void win_thread_create(win_thread_t *t, void (*fn)(void*), void *arg) { (void)t; (void)fn; (void)arg; }
static void win_thread_join(win_thread_t *t) { (void)t; }
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
/* ---- Table-driven matmul_worker_f (replaces the old if-else chain) ---- */

#include "type_dispatch.h"

static void matmul_worker_f(matmul_task_t *t) {
    const type_info_t *ti = typeinfo(t->qtype);
    int nb = t->n_batch;
    int out_stride = nb > 0 ? t->d : t->end - t->start;

    /* F32/F16: no quantization, use generic vec_dot */
    if (!ti || ti->act_fmt == ACT_FMT_F32) {
        if (nb > 0) {
            for (int i = t->start; i < t->end; i++) {
                const char *wrow = t->W + (size_t)i * t->row_bytes;
                for (int b = 0; b < nb; b++)
                    t->out[b * out_stride + i] = vec_dot(wrow, t->x + b * t->n, t->n, t->qtype);
            }
        } else {
            for (int i = t->start; i < t->end; i++) {
                t->out[i] = vec_dot(t->W + (size_t)i * t->row_bytes, t->x, t->n, t->qtype);
            }
        }
        return;
    }

    /* Determine activation row stride based on format */
    size_t qstride;
    switch (ti->act_fmt) {
    case ACT_FMT_Q8_0: qstride = (t->n / 32) * sizeof(block_q8_0); break;
    case ACT_FMT_Q8_K: qstride = (t->n / 256) * sizeof(block_q8_K); break;
    case ACT_FMT_Q8_2: qstride = (t->n / 32) * sizeof(block_q8_2); break;
    default: qstride = 0; break;
    }
    const char *qx_base = (const char *)t->x;
    int nblk = t->n / 32; /* for Q8_0 delta indexing */

    if (nb > 0) {
        /* ---- Batched mode ---- */
        if (ti->rows_per_block > 1) {
            /* Interleaved (R4/R8): process rows in groups */
            int rpb = ti->rows_per_block;
            int start_g = (t->start / rpb) * rpb;
            int end_g = (t->end + rpb - 1) / rpb * rpb;
            size_t block_stride = t->row_bytes * rpb;

            for (int g = start_g; g < end_g; g += rpb) {
                const char *wblock = t->W + (g / rpb) * block_stride;
                for (int b = 0; b < nb; b++) {
                    const char *xb = qx_base + (size_t)b * qstride;
                    float results[TYPE_INFO_MAX_RPB];
                    ti->fn_gemv(wblock, xb, t->n, results, rpb);
                    for (int r = 0; r < rpb && g + r < t->end; r++)
                        t->out[b * out_stride + g + r] = results[r];
                }
            }
        } else {
            /* Plain: one row at a time */
            /* Special case: Q8_0 uses FP32 deltas appended after blocks */
            if (t->qtype == GGUF_TYPE_Q8_0 && t->x_d) {
                for (int i = t->start; i < t->end; i++) {
                    const char *wrow = t->W + (size_t)i * t->row_bytes;
                    for (int b = 0; b < nb; b++) {
                        const char *xb = qx_base + (size_t)b * qstride;
                        const float *xqd = t->x_d + (size_t)b * nblk;
                        t->out[b * out_stride + i] = vec_dot_q8_0_q8_0_deltas(xb, xqd, wrow, t->n);
                    }
                }
            }
            /* Special case: Q4_0 uses shadow Q8_0 conversion for batch reuse */
            else if (t->qtype == GGUF_TYPE_Q4_0 && t->x_d) {
                size_t q8_rb = (t->n / 32) * sizeof(block_q8_0);
                uint8_t shadow_stack[32768];
                void *shadow = (q8_rb <= sizeof(shadow_stack)) ? (void *)shadow_stack : malloc(q8_rb);
                for (int i = t->start; i < t->end; i++) {
                    const char *wrow = t->W + (size_t)i * t->row_bytes;
                    q4_0_row_to_q8_0_shadow(wrow, shadow, t->n);
                    for (int b = 0; b < nb; b++) {
                        const char *xb = qx_base + (size_t)b * q8_rb;
                        const float *xqd = t->x_d + (size_t)b * nblk;
                        t->out[b * out_stride + i] = vec_dot_q8_0_q8_0_deltas(xb, xqd, shadow, t->n);
                    }
                }
                if (shadow != shadow_stack) free(shadow);
            }
            /* Special case: IQ4_NL uses shadow Q8_0 conversion */
            else if (t->qtype == GGUF_TYPE_IQ4_NL && t->x_d) {
                size_t q8_rb = (t->n / 32) * sizeof(block_q8_0);
                uint8_t shadow_stack[32768] __attribute__((aligned(64)));
                void *shadow = (q8_rb <= sizeof(shadow_stack)) ? (void *)shadow_stack : malloc(q8_rb);
                for (int i = t->start; i < t->end; i++) {
                    const char *wrow = t->W + (size_t)i * t->row_bytes;
                    iq4_nl_row_to_q8_0_shadow(wrow, shadow, t->n);
                    for (int b = 0; b < nb; b++) {
                        const char *xb = qx_base + (size_t)b * q8_rb;
                        const float *xqd = t->x_d + (size_t)b * nblk;
                        t->out[b * out_stride + i] = vec_dot_q8_0_q8_0_deltas(xb, xqd, shadow, t->n);
                    }
                }
                if (shadow != shadow_stack) free(shadow);
            }
            /* Generic table-driven path */
            else if (ti->fn_gemv) {
                for (int i = t->start; i < t->end; i++) {
                    const char *wrow = t->W + (size_t)i * t->row_bytes;
                    for (int b = 0; b < nb; b++) {
                        const char *xb = qx_base + (size_t)b * qstride;
                        float result;
                        ti->fn_gemv(wrow, xb, t->n, &result, 1);
                        t->out[b * out_stride + i] = result;
                    }
                }
            }
            /* Fallback: generic vec_dot with F32 activations */
            else {
                for (int i = t->start; i < t->end; i++) {
                    const char *wrow = t->W + (size_t)i * t->row_bytes;
                    for (int b = 0; b < nb; b++)
                        t->out[b * out_stride + i] = vec_dot(wrow, t->x + b * t->n, t->n, t->qtype);
                }
            }
        }
        return;
    }

    /* ---- Non-batched (decode) mode ---- */
    if (ti->rows_per_block > 1) {
        /* Interleaved (R4/R8): process rows in groups */
        int rpb = ti->rows_per_block;
        int start_g = (t->start / rpb) * rpb;
        int end_g = (t->end + rpb - 1) / rpb * rpb;
        size_t block_stride = t->row_bytes * rpb;

        for (int g = start_g; g < end_g; g += rpb) {
            const char *wblock = t->W + (g / rpb) * block_stride;
            float results[TYPE_INFO_MAX_RPB];
            ti->fn_gemv(wblock, t->x, t->n, results, rpb);
            for (int r = 0; r < rpb && g + r < t->end; r++)
                t->out[g + r] = results[r];
        }
    } else {
        /* Plain: one row at a time */
        if (ti->fn_gemv) {
            for (int i = t->start; i < t->end; i++) {
                float result;
                ti->fn_gemv(t->W + (size_t)i * t->row_bytes, t->x, t->n, &result, 1);
                t->out[i] = result;
            }
        } else {
            for (int i = t->start; i < t->end; i++) {
                t->out[i] = vec_dot(t->W + (size_t)i * t->row_bytes, t->x, t->n, t->qtype);
            }
        }
    }
}


#ifdef _WIN32
static DWORD WINAPI pool_worker(void *arg) {
#else
static void *pool_worker(void *arg) {
#endif
    int tid = (int)(size_t)arg;
    pool_my_tid = tid;
    int last_gen = 0;
    win_mutex_lock(&pool_mutex);
    while (1) {
        while (pool_gen == last_gen && !pool_shutdown) {
            win_cond_wait(&pool_cond, &pool_mutex);
        }
        if (pool_shutdown) break;
        last_gen = pool_gen;
        win_mutex_unlock(&pool_mutex);
        if (pool_mode) generic_worker_f(&generic_tasks[tid]);
        else matmul_worker_f(&pool_tasks[tid]);
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

void tensor_threadpool_init(int n_req_threads) {
    if (n_req_threads > MAX_THREADS) n_req_threads = MAX_THREADS;
    if (n_req_threads > 1) pool_init(n_req_threads);
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

/* Shared safe dispatch helpers (see tensor_parallel_for's comment for the
 * full explanation of why partial-slot dispatch is unsafe): every call
 * that wakes the pool must refresh every slot in [0, n_threads), because
 * pool_wake() broadcasts to every worker thread that was ever spawned,
 * not just the ones a given call needs.
 *
 * pool_assign_rows() splits `d` output rows across `count` task slots
 * starting at `base`, writing only .start/.end -- callers fill the other
 * fields (out/x/W/qtype/...) for slots [base, base+active) afterwards.
 * pool_clear_unused() gives the remaining slots an empty range so they
 * safely no-op instead of running a stale task from a previous call. */
static int pool_assign_rows(int base, int count, int d) {
    if (count <= 0) return 0;
    int active = count > d ? d : count;
    if (active < 1) active = 1;
    int rows_per = d / active, extra = d % active, tstart = 0;
    for (int t = 0; t < active; t++) {
        int tend = tstart + rows_per + (t < extra ? 1 : 0);
        pool_tasks[base + t].start = tstart;
        pool_tasks[base + t].end = tend;
        tstart = tend;
    }
    return active;
}

static void pool_clear_unused(int from, int to) {
    for (int t = from; t < to; t++) {
        pool_tasks[t].start = 0;
        pool_tasks[t].end = 0;
    }
}

/* Returns the pool's actual physical thread count (workers + main thread),
 * creating the pool on first use sized to `requested`.
 *
 * This is deliberately NOT the same as the mutable `n_threads` global.
 * Once created the pool's physical size is fixed forever. pool_wake()
 * broadcasts to every physical worker, so the count passed to
 * pool_wake()/pool_wait() must equal the pool's true physical size. */
static int pool_total_threads(int requested) {
    pool_init(requested);
    return (pool_nworkers > 0) ? (pool_nworkers + 1) : 1;
}

/* ---- Table-driven matmul (replaces ~1700 lines of if-else chains) ---- */

void matmul(float *out, const float *x, const void *W, int n, int d, gguf_type_t qtype) {
    if (!W) {
        static int null_warn;
        if (!null_warn) {
            fprintf(stderr, "WARN: matmul NULL weight (n=%d d=%d qtype=%d) -> zero output\n", n, d, qtype);
            null_warn = 1;
        }
        if (out) memset(out, 0, (size_t)d * sizeof(float));
        return;
    }
#ifdef PICOLM_GPU
    if (gpu_tensor && d > 0 && n > 0 && !getenv("PICOLM_PREFILL_CPU") && !getenv("PICOLM_SSM_PREFILL_CPU")) {
        gpu_assert_orchestrator("matmul GPU dispatch");
        if (picolm_gpu_matmul(gpu_tensor, out, x, 1, gpu_device)) {
            static int gpu_matmul_count = 0;
            if (gpu_matmul_count++ == 0) fprintf(stderr, "INFO: GPU matmul active\n");
            return;
        }
    }
#endif
    const type_info_t *ti = typeinfo(qtype);
    size_t row_bytes = gguf_type_row_size(qtype, n);
    const char *wptr = (const char *)W;

    /* F32/F16: use generic vec_dot or tiled GEMM */
    if (!ti || ti->act_fmt == ACT_FMT_F32) {
        if (d < matmul_min_rows || n_threads <= 1) {
            for (int i = 0; i < d; i++)
                out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
            return;
        }
        /* Threaded F32/F16 */
        int nt = pool_total_threads(n_threads);
        int want = n_threads < nt ? n_threads : nt;
        int active = pool_assign_rows(0, want, d);
        for (int t = 0; t < active; t++) {
            pool_tasks[t].out = out; pool_tasks[t].x = x;
            pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
            pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
            pool_tasks[t].qtype = qtype; pool_tasks[t].n_batch = 0;
            pool_tasks[t].x_is_q8k = 0;
        }
        pool_clear_unused(active, nt);
        pool_init(nt); pool_wake(nt);
        matmul_worker_f(&pool_tasks[0]);
        pool_wait(nt);
        return;
    }

    /* Quantized path: quantize activations once, then dispatch.
     * Use amax deltas to match old matmul() decode behavior. */
    quant_buf_t qb = quant_activations_ex(qtype, x, 1, n,
                                           (n_threads <= 1) ? scratch_buf : NULL,
                                           (n_threads <= 1) ? (size_t)scratch_size : 0, 1);
    if (!qb.qbuf) {
        /* Allocation failed: fall back to generic vec_dot */
        for (int i = 0; i < d; i++)
            out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
        return;
    }

    /* Special case: Q4_0_8_8 uses repacked weights if available */
    if (ti->use_repacked) {
        const void *wptr8 = wptr_repacked;
        if (wptr8 && d % 8 == 0 && n % 32 == 0) {
#if defined(PICOLM_AVX2)
            vec_dot_q4_0x8_q8_0_avx2(wptr8, qb.qbuf, n, out, d);
            free_quant_buf(&qb, scratch_buf);
            return;
#endif
        }
    }

    /* Single-threaded or small d: direct GEMV */
    if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
        if (ti->rows_per_block > 1) {
            /* Interleaved: process rows in groups */
            int rpb = ti->rows_per_block;
            int dg = (d / rpb) * rpb;
            size_t block_stride = row_bytes * rpb;
            for (int i = 0; i < dg; i += rpb) {
                ti->fn_gemv(wptr + (i / rpb) * block_stride, qb.qbuf, n, out + i, rpb);
            }
            /* Tail rows */
            for (int i = dg; i < d; i++) {
                out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
            }
        } else if (ti->fn_gemv) {
            /* Plain: use table GEMV kernel */
            for (int i = 0; i < d; i++) {
                float result;
                ti->fn_gemv(wptr + (size_t)i * row_bytes, qb.qbuf, n, &result, 1);
                out[i] = result;
            }
        } else {
            /* No GEMV kernel: generic vec_dot */
            for (int i = 0; i < d; i++)
                out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
        }
        free_quant_buf(&qb, scratch_buf);
        return;
    }

    /* Threaded dispatch */
    {
        int nt = pool_total_threads(n_threads);
        int want = n_threads < nt ? n_threads : nt;
        int active = pool_assign_rows(0, want, d);
        for (int t = 0; t < active; t++) {
            pool_tasks[t].out = out;
            pool_tasks[t].x = (const float *)qb.qbuf;
            pool_tasks[t].x_d = qb.dbuf;
            pool_tasks[t].W = wptr;
            pool_tasks[t].row_bytes = row_bytes;
            pool_tasks[t].n = n;
            pool_tasks[t].qtype = qtype;
            pool_tasks[t].n_batch = 0;
            pool_tasks[t].x_is_q8k = (ti->act_fmt == ACT_FMT_Q8_K) ? 1 : 0;
        }
        pool_clear_unused(active, nt);
        pool_init(nt); pool_wake(nt);
        matmul_worker_f(&pool_tasks[0]);
        pool_wait(nt);
    }
    free_quant_buf(&qb, scratch_buf);
}

/* ================================================================
 * Batch matmul for prefill (weights read once for all tokens)
 * Output layout: out[batch * d + row]
 * ================================================================ */

typedef struct {
    const char *wptr;
    const void *qx_buf;
    size_t q8_row_bytes;
    float *out;
    int n, d;
} q4_0_4_4_batch_ctx_t;

static void q4_0_4_4_batch_task(int b, void *ctxp) {
    q4_0_4_4_batch_ctx_t *ctx = (q4_0_4_4_batch_ctx_t *)ctxp;
    const char *xb = (const char *)ctx->qx_buf + (size_t)b * ctx->q8_row_bytes;
    vec_dot_q4_0x4_q8_0(ctx->wptr, xb, ctx->n, ctx->out + (size_t)b * ctx->d, ctx->d);
}

#ifdef PICOLM_AVX2
typedef struct {
    const char *wptr;
    size_t w_row_bytes;      /* Q4_0_8_8 row_bytes for weight offset calc */
    const void *qx_buf;
    size_t q8_row_bytes;     /* Q8_0 row_bytes for activation offset calc */
    float *out;
    int n, d, n_batch;
    int row_offset;          /* skip first N rows (already processed by GEMM) */
} q4_0_8_8_batch_ctx_t;

/* Threaded Q4_0_8_8 dual-batch GEMM context */
typedef struct {
    int k;
    const void *w;
    void *abuf;
    int n_batch, n_batch_padded;
    int d;
    gguf_type_t qtype1, qtype2;
    const void *W1, *W2;
    float *out1, *out2;
} qgemm_q4x8_dual_ctx_t;

/* Process one (token, 8-row-group) pair. The AVX2 kernel handles partial
 * groups at the tail. This enables row-level parallelism within each
 * token, matching the matmul() single-token threaded path and closing
 * the perf gap vs Q8_0's row-parallel matmul_batch. */
static void q4_0_8_8_batch_task(int idx, void *ctxp) {
    q4_0_8_8_batch_ctx_t *ctx = (q4_0_8_8_batch_ctx_t *)ctxp;
    int total_groups = (ctx->d + 7) / 8;
    int b = idx / total_groups;
    int g = idx % total_groups;
    int row_start = g * 8;
    int nrows = (row_start + 8 < ctx->d) ? 8 : (ctx->d - row_start);
    if (row_start >= ctx->d) return;
    if (b < ctx->row_offset) return;  /* skip rows already processed by GEMM */

    const char *xb = (const char *)ctx->qx_buf + (size_t)b * ctx->q8_row_bytes;
    const void *wp = (const char *)ctx->wptr + (size_t)row_start * ctx->w_row_bytes;
    float *op = ctx->out + (size_t)b * ctx->d + row_start;
    vec_dot_q4_0x8_q8_0_avx2(wp, xb, ctx->n, op, nrows);
}

#if defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VNNI__)
/* Threaded task: distribute 16-row tiles across threads.
 * Each thread processes tiles spaced by nth to avoid false sharing.
 * tmp1/tmp2 are passed via out1/out2 fields of the context. */
static void qgemm_q4x8_dual_task(int idx, void *ctxp) {
    qgemm_q4x8_dual_ctx_t *c = (qgemm_q4x8_dual_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    int total_tiles = c->n_batch_padded / 16;
    int nb = c->k / 32;

    for (int t = idx; t < total_tiles; t += nth) {
        int nr_start = t * 16;
        block_q8_0x4 *ap = (block_q8_0x4 *)c->abuf + (nr_start / 4) * nb;

        if (c->qtype1 == GGUF_TYPE_Q4_0_8_8) {
            float *o1 = (float *)c->out1 + nr_start * c->d;
            sgemm_q4_0x8_q8_0x4(16, c->d, c->k, c->W1, ap, o1, c->d, idx, nth);
        } else if (c->qtype1 == GGUF_TYPE_Q4I_0_8_8) {
            float *o1 = (float *)c->out1 + nr_start * c->d;
            sgemm_q4i_0x8_q8_0x4(16, c->d, c->k, c->W1, ap, o1, c->d, idx, nth);
        }
        if (c->qtype2 == GGUF_TYPE_Q4_0_8_8) {
            float *o2 = (float *)c->out2 + nr_start * c->d;
            sgemm_q4_0x8_q8_0x4(16, c->d, c->k, c->W2, ap, o2, c->d, idx, nth);
        } else if (c->qtype2 == GGUF_TYPE_Q4I_0_8_8) {
            float *o2 = (float *)c->out2 + nr_start * c->d;
            sgemm_q4i_0x8_q8_0x4(16, c->d, c->k, c->W2, ap, o2, c->d, idx, nth);
        }
    }
}
#endif
#endif

/* matmul_q8: matmul with pre-quantized Q8_0 activation.
 * qx: pre-quantized input as block_q8_0[], same layout as quantize_row_q8_0 output.
 * qx_d: pre-converted delta values (fp32) for each Q8_0 block of x.
 * This avoids re-quantizing x for every call, critical for MoE where x
 * is projected against 16+ expert weight matrices per layer.
 * n = input dimension (must be multiple of 32)
 * d = number of output rows (weight matrix columns) */
void matmul_q8(float *out, const void *qx, const float *qx_d,
               const void *W, int n, int d, gguf_type_t qtype) {
    size_t row_bytes = gguf_type_row_size(qtype, n);
    const char *wptr = (const char *)W;

    if (qtype == GGUF_TYPE_Q8_0 && n > 0) {
        /* Use Q8xQ8 dot products with pre-converted deltas */
        if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
            for (int i = 0; i < d; i++) {
                out[i] = vec_dot_q8_0_q8_0_deltas(qx, qx_d, wptr + (size_t)i * row_bytes, n);
            }
            return;
        }

        /* Threaded dispatch: main thread does task 0, workers do tasks 1..nt-1 */
        {
            int nt = pool_total_threads(n_threads);
            int want = n_threads < nt ? n_threads : nt;
            int active = pool_assign_rows(0, want, d);
            for (int t = 0; t < active; t++) {
                pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                pool_tasks[t].x_d = (float *)qx_d; pool_tasks[t].W = wptr;
                pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                pool_tasks[t].qtype = GGUF_TYPE_Q8_0;
                pool_tasks[t].n_batch = 0; pool_tasks[t].x_is_q8k = 0;
            }
            pool_clear_unused(active, nt);
            pool_init(nt);
            pool_wake(nt);
            matmul_worker_f(&pool_tasks[0]);
            pool_wait(nt);
        }
        return;
    }
    /* For non-Q8_0 types, dequantize Q8_0 activation to float, then use vec_dot.
     * Threaded dispatch is not supported for non-Q8_0 weights (matmul_worker_f
     * has no non-batched float-activation fallback); always use single-threaded. */
    {
        float *xf = (float *)malloc((size_t)n * sizeof(float));
        dequantize_row_q8_0(qx, xf, n);
        for (int i = 0; i < d; i++) {
            out[i] = vec_dot(wptr + (size_t)i * row_bytes, xf, n, qtype);
        }
        free(xf);
    }
}
/* matmul_q8_seq: sequential Q8xQ8 matmul that ignores n_threads.
 * Safe to call from inside tensor_parallel_for workers. */
void matmul_q8_seq(float *out, const void *qx, const float *qx_d,
                   const void *W, int n, int d, gguf_type_t qtype) {
    size_t row_bytes = gguf_type_row_size(qtype, n);
    const char *wptr = (const char *)W;

    if (qtype == GGUF_TYPE_Q8_0 && n > 0) {
        for (int i = 0; i < d; i++) {
            out[i] = vec_dot_q8_0_q8_0_deltas(qx, qx_d, wptr + (size_t)i * row_bytes, n);
        }
        return;
    }
    /* Non-Q8_0 weights: dequantize Q8_0 activation to float, then use vec_dot */
    {
        float *xf = (float *)malloc((size_t)n * sizeof(float));
        dequantize_row_q8_0(qx, xf, n);
        for (int i = 0; i < d; i++) {
            out[i] = vec_dot(wptr + (size_t)i * row_bytes, xf, n, qtype);
        }
        free(xf);
    }
}

/* matmul_q8_batch: batched Q8×Q8 matmul with pre-quantized activations.
 *
 * Strategy: for each weight row, process 4 tokens at a time using
 * vec_dot_q8_0_q8_0_deltas_batch4, which loads the weight block once
 * and reuses it across 4 different activations. This reduces weight
 * memory traffic by ~4x compared to sequential single-token calls.
 *
 * qx_all: array of n_batch pre-quantized Q8_0 inputs, each of size
 *         q8_buf_per_token floats.
 * qx_d_off: offset in floats from token buffer start to delta array.
 * out: output [n_batch * d], row-major: out[b*d + i] = token b, row i.
 * W: weight matrix [d rows × n cols] in Q8_0 format. */
void matmul_q8_batch(float *out, const float *qx_all, int qx_d_off,
                     int q8_buf_per_token, const void *W,
                     int n, int d, int n_batch, gguf_type_t qtype) {
    if (n <= 0 || d <= 0 || n_batch <= 0) return;

    if (qtype != GGUF_TYPE_Q8_0) {
        /* Non-Q8_0 weights: dequantize Q8_0 activation to float, then use vec_dot.
         * This is slower than Q8xQ8 but handles all weight types (Q4_K, Q6_K, etc.) */
        {
            size_t row_bytes = gguf_type_row_size(qtype, n);
            const char *wptr = (const char *)W;
            float *xf = (float *)malloc((size_t)n * sizeof(float));
            for (int b = 0; b < n_batch; b++) {
                const float *tbuf = qx_all + b * q8_buf_per_token;
                dequantize_row_q8_0(tbuf, xf, n);
                for (int i = 0; i < d; i++) {
                    out[b * d + i] = vec_dot(wptr + (size_t)i * row_bytes, xf, n, qtype);
                }
            }
            free(xf);
        }
        return;
    }

    size_t row_bytes = gguf_type_row_size(qtype, n);
    const char *wptr = (const char *)W;

    /* Process 4 tokens at a time per weight row */
    int n_batch4 = (n_batch / 4) * 4;

    for (int i = 0; i < d; i++) {
        const char *wi = wptr + (size_t)i * row_bytes;

        /* Batch-4 kernel: weight loaded once per block, reused across 4 tokens */
        for (int b = 0; b < n_batch4; b += 4) {
            const float *tb[4];
            const void *qx[4];
            const float *qx_d[4];
            float fout[4];

            for (int k = 0; k < 4; k++) {
                tb[k] = qx_all + (b + k) * q8_buf_per_token;
                qx[k] = (const void *)tb[k];
                qx_d[k] = tb[k] + qx_d_off;
            }

            vec_dot_q8_0_q8_0_deltas_batch4(qx[0], qx_d[0], qx[1], qx_d[1],
                                             qx[2], qx_d[2], qx[3], qx_d[3],
                                             wi, n,
                                             &fout[0], &fout[1], &fout[2], &fout[3]);

            for (int k = 0; k < 4; k++) {
                out[(b + k) * d + i] = fout[k];
            }
        }

        /* Remainder: single-token vec_dot */
        for (int b = n_batch4; b < n_batch; b++) {
            const float *tbuf = qx_all + b * q8_buf_per_token;
            out[b * d + i] = vec_dot_q8_0_q8_0_deltas((const void *)tbuf,
                tbuf + qx_d_off, wi, n);
        }
    }
}

/* ================================================================
 * Parallel mm_id expert dispatch
 *
 * Context struct shared between main thread and worker threads.
 * Worker functions are declared at file scope for C compliance.
 * ================================================================ */

/* Precomputed routing map: for each expert, list of (token_id << 8 | slot) entries.
 * Using packed token+slot to save space. expert_counts[e] tells how many entries. */
#define MOE_ROUTE_PACK(t, s)  ((t) << 8 | (s))
#define MOE_ROUTE_TOKEN(v)    ((v) >> 8)
#define MOE_ROUTE_SLOT(v)     ((v) & 0xff)

typedef struct {
    float *gate_out; float *up_out;
    const float *qx_all; int qx_d_off; int q8_buf_per_token;
    const void *gate_w_base; const void *up_w_base;
    const int *expert_assignments; /* [n_expert][n_tokens] packed token+slot */
    const int *expert_counts;      /* [n_expert] number of assigned tokens */
    int n_tokens; int n_used; int dim; int n_ff; int n_expert;
    gguf_type_t type;
    size_t row_bytes;
} mm_id_gate_ctx_t;

typedef struct {
    float *down_out;
    const float *expert_out;
    const void *down_w_base;
    const int *expert_assignments;
    const int *expert_counts;
    int n_tokens; int n_used; int dim; int n_ff; int n_expert;
    gguf_type_t type;
    size_t row_bytes; size_t dnb; size_t qx_d_off;
    block_q8_0 *scratch_qx; float *scratch_qx_d;
    block_q8_0 *exp_down_qx_all;
    int q8_per_token; int max_threads;
} mm_id_down_ctx_t;

static void mm_id_gate_expert_task(int idx, void *ctxp);
static void mm_id_down_expert_task(int idx, void *ctxp);

/* matmul_mm_id: llama.cpp-style MoE batched GEMM.
 *
 * Input layout (PicoLM native, no repacking needed):
 *   W: expert weights [n_expert][n_ff][dim] in Q8_0
 *   x_batch: input activations [n_tokens * dim] in F32
 *   ids: [n_tokens][n_used] — expert ID for each token's slot
 *   weights: [n_tokens][n_used] — routing weight for each slot
 *
 * Strategy: for each expert, gather the tokens that selected it,
 * process through gate/up/down, accumulate into output.
 * Uses pre-quantized Q8_0 activations to avoid redundant quantization.
 *
 * out: [n_tokens * n_used * n_ff] — per-token, per-slot expert output
 * qx_all: pre-quantized activations [n_tokens * q8_buf_per_token]
 */
void matmul_mm_id_gate_up(float *gate_out, float *up_out,
    const float *qx_all, int qx_d_off, int q8_buf_per_token,
    const void *gate_w_base, const void *up_w_base,
    const int *expert_assignments, const int *expert_counts,
    int n_tokens, int n_used, int dim, int n_ff, int n_expert,
    gguf_type_t type) {

    if (n_tokens <= 0 || n_used <= 0 || dim <= 0 || n_ff <= 0) return;

    size_t row_bytes = gguf_type_row_size(type, dim);

    /* Count active experts — only dispatch those with >0 assigned tokens */
    int active_eids[256];
    int n_active = 0;
    for (int eid = 0; eid < n_expert; eid++) {
        if (expert_counts[eid] > 0) active_eids[n_active++] = eid;
    }

    /* Use sequential path for small batches (dispatch overhead > benefit) */
    if (n_active < 8 || n_tokens <= 2) {
        for (int i = 0; i < n_active; i++) {
            mm_id_gate_expert_task(active_eids[i],
                (void *)&(mm_id_gate_ctx_t){ gate_out, up_out, qx_all, qx_d_off, q8_buf_per_token,
                    gate_w_base, up_w_base, expert_assignments, expert_counts,
                    n_tokens, n_used, dim, n_ff, n_expert, type, row_bytes });
        }
        return;
    }

    mm_id_gate_ctx_t ctx = { gate_out, up_out, qx_all, qx_d_off, q8_buf_per_token,
        gate_w_base, up_w_base, expert_assignments, expert_counts,
        n_tokens, n_used, dim, n_ff, n_expert, type, row_bytes };

    tensor_parallel_for(n_expert, mm_id_gate_expert_task, &ctx);
}

/* matmul_mm_id_down: down projection with mm_id pattern.
 *
 * Input: expert_out[t * n_used * n_ff + slot * n_ff] = SwiGLU output per token per expert slot
 * Output: down_out[t * n_used * dim + slot * dim] = down projection result
 *
 * Each expert's down weights: [n_ff, dim] in Q8_0
 *
 * Strategy: for each expert, gather assigned tokens, quantize their SwiGLU
 * outputs into pre-allocated Q8_0 buffers, then use matmul_q8_batch_expert
 * (with batch4 kernel) for the down projection.
 */
void matmul_mm_id_down(float *down_out,
    const float *expert_out, /* [n_tokens * n_used * n_ff] */
    const void *down_w_base,
    const int *expert_assignments, const int *expert_counts,
    int n_tokens, int n_used, int dim, int n_ff, int n_expert,
    gguf_type_t type,
    /* Scratch buffers */
    block_q8_0 *scratch_qx, float *scratch_qx_d,
    /* Per-expert batched quantization buffers */
    block_q8_0 *exp_down_qx_all, float *exp_down_qx_d_all, int q8_per_token) {
    /* exp_down_qx_all uses the same layout as moe_qx_all: contiguous float buffer
     * where each entry is [Q8_0 blocks as float-aligned bytes] + [dnb deltas].
     * q8_per_token = moe_q8_buf_per_token from run_state_t. */

    if (n_tokens <= 0 || n_used <= 0 || dim <= 0 || n_ff <= 0) return;

    (void)exp_down_qx_d_all;

    size_t row_bytes = gguf_type_row_size(type, n_ff);
    size_t dnb = n_ff / 32;

    /* Compute qx_d_off from q8_per_token (same as moe_qx_d_off calculation) */
    size_t q8_rb = n_ff / 32;
    size_t qx_d_off = (q8_rb * sizeof(block_q8_0) + sizeof(float) - 1) / sizeof(float);

    mm_id_down_ctx_t ctx = { down_out, expert_out, down_w_base,
        expert_assignments, expert_counts, n_tokens, n_used, dim, n_ff, n_expert,
        type, row_bytes, dnb, qx_d_off,
        scratch_qx, scratch_qx_d, exp_down_qx_all, q8_per_token, tensor_get_n_threads() };

    tensor_parallel_for(n_expert, mm_id_down_expert_task, &ctx);
}


/* ================================================================
 * mm_id parallel expert tasks (file-scope static functions)
 * ================================================================ */
static void mm_id_gate_expert_task(int idx, void *ctxp) {
    mm_id_gate_ctx_t *c = (mm_id_gate_ctx_t *)ctxp;
    int eid = idx;

    /* Use precomputed routing map */
    int n_assigned = c->expert_counts[eid];
    if (n_assigned == 0) return;

    const int *assignments = c->expert_assignments + eid * c->n_tokens;

    int assigned[256]; /* max tokens that can select one expert */
    int slots[256];
    int n_useful = n_assigned < 256 ? n_assigned : 256;
    for (int a = 0; a < n_useful; a++) {
        assigned[a] = MOE_ROUTE_TOKEN(assignments[a]);
        slots[a] = MOE_ROUTE_SLOT(assignments[a]);
    }
    n_assigned = n_useful;

    const char *gate_exp = (const char *)c->gate_w_base + (size_t)eid * c->n_ff * c->row_bytes;
    const char *up_exp = (const char *)c->up_w_base + (size_t)eid * c->n_ff * c->row_bytes;

#ifdef PICOLM_AVX512
    if (c->type == GGUF_TYPE_Q8_0) {
        /* Block-interleaved kernel: Q8_0 weights only.
         * For other types, fall through to scalar_gateup below. */
        int nblk = c->dim / 32;  /* 64 blocks per row for dim=2048 */

        /* Precompute token buffer pointers */
        const float *tbufs[256];
        const float *qx_d_ptrs[256];
    for (int a = 0; a < n_assigned; a++) {
        tbufs[a] = c->qx_all + assigned[a] * c->q8_buf_per_token;
        qx_d_ptrs[a] = tbufs[a] + c->qx_d_off;
    }

    /* Block-interleaved kernel: for each weight row, process all blocks
     * in the outer loop and all assigned tokens in the inner loop.
     * This keeps weight blocks in L1 cache across all tokens.
     * Per-token accumulators are __m512 registers (max 8 inline). */
    if (n_assigned <= 8) {
        for (int row = 0; row < c->n_ff; row++) {
            const block_q8_0 *gw = (const block_q8_0 *)(gate_exp + (size_t)row * c->row_bytes);
            const block_q8_0 *uw = (const block_q8_0 *)(up_exp + (size_t)row * c->row_bytes);

            __m512 g_acc[8], u_acc[8];
            for (int a = 0; a < n_assigned; a++) {
                g_acc[a] = _mm512_setzero_ps();
                u_acc[a] = _mm512_setzero_ps();
            }

            int bi;
            for (bi = 0; bi + 1 < nblk; bi += 2) {
                /* Load gate weight blocks (once, shared across tokens) */
                __m256i wg0 = _mm256_loadu_si256((const __m256i *)gw[bi].qs);
                __m256i wg1 = _mm256_loadu_si256((const __m256i *)gw[bi + 1].qs);
                __m512i ww_g = _mm512_inserti64x4(_mm512_castsi256_si512(wg0), wg1, 1);
                float wd_g0 = fp16_to_fp32_lookup(gw[bi].d);
                float wd_g1 = fp16_to_fp32_lookup(gw[bi + 1].d);

                /* Load up weight blocks (once, shared across tokens) */
                __m256i wu0 = _mm256_loadu_si256((const __m256i *)uw[bi].qs);
                __m256i wu1 = _mm256_loadu_si256((const __m256i *)uw[bi + 1].qs);
                __m512i ww_u = _mm512_inserti64x4(_mm512_castsi256_si512(wu0), wu1, 1);
                float wd_u0 = fp16_to_fp32_lookup(uw[bi].d);
                float wd_u1 = fp16_to_fp32_lookup(uw[bi + 1].d);

                for (int a = 0; a < n_assigned; a++) {
                    const block_q8_0 *xq = (const block_q8_0 *)tbufs[a];
                    const float *qx_d = qx_d_ptrs[a];

                    __m256i x0 = _mm256_loadu_si256((const __m256i *)xq[bi].qs);
                    __m256i x1 = _mm256_loadu_si256((const __m256i *)xq[bi + 1].qs);
                    __m512i xx = _mm512_inserti64x4(_mm512_castsi256_si512(x0), x1, 1);

                    /* Gate dot product */
                    __m512i dot_g = mul_sum_i8_pairs_avx512(xx, ww_g);
                    __m512 f_g = _mm512_cvtepi32_ps(dot_g);
                    float dg0 = qx_d[bi] * wd_g0;
                    float dg1 = qx_d[bi + 1] * wd_g1;
                    __m512 dvec_g = _mm512_insertf32x8(
                        _mm512_castps256_ps512(_mm256_set1_ps(dg0)),
                        _mm256_set1_ps(dg1), 1);
                    g_acc[a] = _mm512_fmadd_ps(f_g, dvec_g, g_acc[a]);

                    /* Up dot product (same activation, different weights) */
                    __m512i dot_u = mul_sum_i8_pairs_avx512(xx, ww_u);
                    __m512 f_u = _mm512_cvtepi32_ps(dot_u);
                    float du0 = qx_d[bi] * wd_u0;
                    float du1 = qx_d[bi + 1] * wd_u1;
                    __m512 dvec_u = _mm512_insertf32x8(
                        _mm512_castps256_ps512(_mm256_set1_ps(du0)),
                        _mm256_set1_ps(du1), 1);
                    u_acc[a] = _mm512_fmadd_ps(f_u, dvec_u, u_acc[a]);
                }
            }

            /* Reduce and scatter */
            for (int a = 0; a < n_assigned; a++) {
                const block_q8_0 *xq = (const block_q8_0 *)tbufs[a];
                const float *qx_d = qx_d_ptrs[a];
                float gs = _mm512_reduce_add_ps(g_acc[a]);
                float us = _mm512_reduce_add_ps(u_acc[a]);
                /* Scalar remainder for odd nblk */
                for (; bi < nblk; bi++) {
                    int si_g = 0, si_u = 0;
                    for (int j = 0; j < 32; j++) {
                        si_g += xq[bi].qs[j] * gw[bi].qs[j];
                        si_u += xq[bi].qs[j] * uw[bi].qs[j];
                    }
                    gs += (float)si_g * qx_d[bi] * fp16_to_fp32_lookup(gw[bi].d);
                    us += (float)si_u * qx_d[bi] * fp16_to_fp32_lookup(uw[bi].d);
                }
                int t = assigned[a];
                size_t idx = (size_t)t * c->n_used * c->n_ff + (size_t)slots[a] * c->n_ff + row;
                c->gate_out[idx] = gs;
                c->up_out[idx] = us;
            }
        }
    } else {
        /* >8 tokens per expert: use batch4 + scalar fallback */
        int n_batch4 = (n_assigned / 4) * 4;
        for (int row = 0; row < c->n_ff; row++) {
            const char *gw = gate_exp + row * c->row_bytes;
            const char *uw = up_exp + row * c->row_bytes;
            for (int a = 0; a < n_batch4; a += 4) {
                const void *qx[4];
                const float *qx_d_arr[4];
                float fout[4];
                for (int k = 0; k < 4; k++) {
                    qx[k] = (const void *)tbufs[a + k];
                    qx_d_arr[k] = qx_d_ptrs[a + k];
                }
                vec_dot_q8_0_q8_0_deltas_batch4(qx[0], qx_d_arr[0], qx[1], qx_d_arr[1],
                    qx[2], qx_d_arr[2], qx[3], qx_d_arr[3], gw, c->dim,
                    &fout[0], &fout[1], &fout[2], &fout[3]);
                for (int k = 0; k < 4; k++) {
                    size_t idx = (size_t)assigned[a + k] * c->n_used * c->n_ff
                               + (size_t)slots[a + k] * c->n_ff + row;
                    c->gate_out[idx] = fout[k];
                }
                vec_dot_q8_0_q8_0_deltas_batch4(qx[0], qx_d_arr[0], qx[1], qx_d_arr[1],
                    qx[2], qx_d_arr[2], qx[3], qx_d_arr[3], uw, c->dim,
                    &fout[0], &fout[1], &fout[2], &fout[3]);
                for (int k = 0; k < 4; k++) {
                    size_t idx = (size_t)assigned[a + k] * c->n_used * c->n_ff
                               + (size_t)slots[a + k] * c->n_ff + row;
                    c->up_out[idx] = fout[k];
                }
            }
            for (int a = n_batch4; a < n_assigned; a++) {
                size_t idx = (size_t)assigned[a] * c->n_used * c->n_ff
                           + (size_t)slots[a] * c->n_ff + row;
                c->gate_out[idx] = vec_dot_q8_0_q8_0_deltas(tbufs[a], qx_d_ptrs[a], gw, c->dim);
                c->up_out[idx] = vec_dot_q8_0_q8_0_deltas(tbufs[a], qx_d_ptrs[a], uw, c->dim);
            }
        }
    }
    } /* end if (c->type == GGUF_TYPE_Q8_0) */
    if (c->type == GGUF_TYPE_Q8_0) return;
    /* For non-Q8_0 weights, dequantize each row to float once, then
     * compute dot products with Q8_0 activations. AVX-512 batch4 path
     * keeps the weight row resident while processing 4 tokens at a time. */
    float *wrow_f = (float *)malloc(c->dim * sizeof(float));
    if (!wrow_f) { fprintf(stderr, "OOM: scalar_gateup\n"); exit(1); }

    int n_batch4 = (n_assigned / 4) * 4;
    for (int row = 0; row < c->n_ff; row++) {
        const char *gw = gate_exp + row * c->row_bytes;
        const char *uw = up_exp + row * c->row_bytes;

        dequantize_row(gw, wrow_f, c->dim, c->type);
        for (int a = 0; a < n_batch4; a += 4) {
            float fout[4];
            const float *tb[4];
            for (int k = 0; k < 4; k++)
                tb[k] = c->qx_all + assigned[a + k] * c->q8_buf_per_token;
            vec_dot_q8_0_f32_batch4(tb[0], tb[1], tb[2], tb[3], wrow_f, c->dim,
                                    &fout[0], &fout[1], &fout[2], &fout[3]);
            for (int k = 0; k < 4; k++) {
                size_t idx = (size_t)assigned[a + k] * c->n_used * c->n_ff
                           + (size_t)slots[a + k] * c->n_ff + row;
                c->gate_out[idx] = fout[k];
            }
        }
        for (int a = n_batch4; a < n_assigned; a++) {
            int t = assigned[a];
            const float *tbuf = c->qx_all + t * c->q8_buf_per_token;
            size_t idx = (size_t)t * c->n_used * c->n_ff + (size_t)slots[a] * c->n_ff + row;
            c->gate_out[idx] = vec_dot_q8_0_f32(tbuf, wrow_f, c->dim);
        }

        dequantize_row(uw, wrow_f, c->dim, c->type);
        for (int a = 0; a < n_batch4; a += 4) {
            float fout[4];
            const float *tb[4];
            for (int k = 0; k < 4; k++)
                tb[k] = c->qx_all + assigned[a + k] * c->q8_buf_per_token;
            vec_dot_q8_0_f32_batch4(tb[0], tb[1], tb[2], tb[3], wrow_f, c->dim,
                                    &fout[0], &fout[1], &fout[2], &fout[3]);
            for (int k = 0; k < 4; k++) {
                size_t idx = (size_t)assigned[a + k] * c->n_used * c->n_ff
                           + (size_t)slots[a + k] * c->n_ff + row;
                c->up_out[idx] = fout[k];
            }
        }
        for (int a = n_batch4; a < n_assigned; a++) {
            int t = assigned[a];
            const float *tbuf = c->qx_all + t * c->q8_buf_per_token;
            size_t idx = (size_t)t * c->n_used * c->n_ff + (size_t)slots[a] * c->n_ff + row;
            c->up_out[idx] = vec_dot_q8_0_f32(tbuf, wrow_f, c->dim);
        }
    }
    free(wrow_f);
#else
    /* Non-AVX512: scalar path for all weight types.
     * For Q8_0 weights, use vec_dot_q8_0_q8_0_deltas directly.
     * For non-Q8_0 weights, dequantize to float and use vec_dot_q8_0_f32. */
    if (c->type == GGUF_TYPE_Q8_0) {
        for (int row = 0; row < c->n_ff; row++) {
            const char *gw = gate_exp + row * c->row_bytes;
            const char *uw = up_exp + row * c->row_bytes;
            for (int a = 0; a < n_assigned; a++) {
                int t = assigned[a];
                const float *tbuf = c->qx_all + t * c->q8_buf_per_token;
                size_t idx = (size_t)t * c->n_used * c->n_ff + (size_t)slots[a] * c->n_ff + row;
                c->gate_out[idx] = vec_dot_q8_0_q8_0_deltas((const void *)tbuf,
                    tbuf + c->qx_d_off, gw, c->dim);
                c->up_out[idx] = vec_dot_q8_0_q8_0_deltas((const void *)tbuf,
                    tbuf + c->qx_d_off, uw, c->dim);
            }
        }
    } else {
        /* Non-Q8_0: dequantize each row to float, then dot with Q8_0 activations */
        float *wrow_f = (float *)malloc(c->dim * sizeof(float));
        if (!wrow_f) { fprintf(stderr, "OOM: scalar_gateup\n"); exit(1); }
        for (int row = 0; row < c->n_ff; row++) {
            const char *gw = gate_exp + row * c->row_bytes;
            const char *uw = up_exp + row * c->row_bytes;
            dequantize_row(gw, wrow_f, c->dim, c->type);
            for (int a = 0; a < n_assigned; a++) {
                int t = assigned[a];
                const void *tbuf = (const void *)(c->qx_all + t * c->q8_buf_per_token);
                size_t idx = (size_t)t * c->n_used * c->n_ff + (size_t)slots[a] * c->n_ff + row;
                c->gate_out[idx] = vec_dot_q8_0_f32(tbuf, wrow_f, c->dim);
            }
            dequantize_row(uw, wrow_f, c->dim, c->type);
            for (int a = 0; a < n_assigned; a++) {
                int t = assigned[a];
                const void *tbuf = (const void *)(c->qx_all + t * c->q8_buf_per_token);
                size_t idx = (size_t)t * c->n_used * c->n_ff + (size_t)slots[a] * c->n_ff + row;
                c->up_out[idx] = vec_dot_q8_0_f32(tbuf, wrow_f, c->dim);
            }
        }
        free(wrow_f);
    }
#endif
}

static void mm_id_down_expert_task(int idx, void *ctxp) {
    mm_id_down_ctx_t *c = (mm_id_down_ctx_t *)ctxp;
    int eid = idx;

    /* Use precomputed routing map */
    int n_assigned = c->expert_counts[eid];
    if (n_assigned == 0) return;

    const int *assignments = c->expert_assignments + eid * c->n_tokens;

    int assigned[256];
    int slots[256];
    int n_useful = n_assigned < 256 ? n_assigned : 256;
    for (int a = 0; a < n_useful; a++) {
        assigned[a] = MOE_ROUTE_TOKEN(assignments[a]);
        slots[a] = MOE_ROUTE_SLOT(assignments[a]);
    }
    n_assigned = n_useful;

    const char *dw = (const char *)c->down_w_base + (size_t)eid * c->dim * c->row_bytes;

    int tid = tensor_get_thread_id();
    float *my_qx_base = (float *)c->exp_down_qx_all + tid * 256 * c->q8_per_token;

    /* Quantize all assigned tokens' SwiGLU outputs to Q8_0 */
    const void *qx_bufs[256];
    const float *qx_d_arr[256];

    for (int a = 0; a < n_assigned; a++) {
        int t = assigned[a];
        int slot = slots[a];

        const float *swiglu = c->expert_out + (size_t)t * c->n_used * c->n_ff + (size_t)slot * c->n_ff;
        float *entry = my_qx_base + a * c->q8_per_token;
        block_q8_0 *exp_qx = (block_q8_0 *)entry;
        float *exp_qx_d = entry + c->qx_d_off;
        quantize_row_q8_0(swiglu, exp_qx, c->n_ff);
        for (size_t bi = 0; bi < c->dnb; bi++)
            exp_qx_d[bi] = fp16_to_fp32(exp_qx[bi].d);

        qx_bufs[a] = (const void *)exp_qx;
        qx_d_arr[a] = exp_qx_d;
    }

#ifdef PICOLM_AVX512
    int nblk = c->n_ff / 32;  /* 16 blocks per row for n_ff=512 */

    /* Block-interleaved kernel: Q8_0 weights only.
     * For non-Q8_0 types, fall through to matmul_q8_seq below. */
    if (c->type == GGUF_TYPE_Q8_0 && n_assigned <= 8) {
        for (int row = 0; row < c->dim; row++) {
            const block_q8_0 *dw_blocks = (const block_q8_0 *)(dw + (size_t)row * c->row_bytes);

            __m512 acc[8];
            for (int a = 0; a < n_assigned; a++)
                acc[a] = _mm512_setzero_ps();

            int bi;
            for (bi = 0; bi + 1 < nblk; bi += 2) {
                /* Load down weight blocks (once, shared across tokens) */
                __m256i wd0 = _mm256_loadu_si256((const __m256i *)dw_blocks[bi].qs);
                __m256i wd1 = _mm256_loadu_si256((const __m256i *)dw_blocks[bi + 1].qs);
                __m512i ww = _mm512_inserti64x4(_mm512_castsi256_si512(wd0), wd1, 1);
                float wd_d0 = fp16_to_fp32_lookup(dw_blocks[bi].d);
                float wd_d1 = fp16_to_fp32_lookup(dw_blocks[bi + 1].d);

                for (int a = 0; a < n_assigned; a++) {
                    const block_q8_0 *xq = (const block_q8_0 *)qx_bufs[a];
                    const float *qx_d = qx_d_arr[a];

                    __m256i x0 = _mm256_loadu_si256((const __m256i *)xq[bi].qs);
                    __m256i x1 = _mm256_loadu_si256((const __m256i *)xq[bi + 1].qs);
                    __m512i xx = _mm512_inserti64x4(_mm512_castsi256_si512(x0), x1, 1);

                    __m512i dot = mul_sum_i8_pairs_avx512(xx, ww);
                    __m512 f = _mm512_cvtepi32_ps(dot);
                    float d0 = qx_d[bi] * wd_d0;
                    float d1 = qx_d[bi + 1] * wd_d1;
                    __m512 dvec = _mm512_insertf32x8(
                        _mm512_castps256_ps512(_mm256_set1_ps(d0)),
                        _mm256_set1_ps(d1), 1);
                    acc[a] = _mm512_fmadd_ps(f, dvec, acc[a]);
                }
            }

            /* Reduce and scatter */
            for (int a = 0; a < n_assigned; a++) {
                const block_q8_0 *xq = (const block_q8_0 *)qx_bufs[a];
                const float *qx_d = qx_d_arr[a];
                float result = _mm512_reduce_add_ps(acc[a]);
                /* Scalar remainder for odd nblk */
                for (; bi < nblk; bi++) {
                    int si = 0;
                    for (int j = 0; j < 32; j++)
                        si += xq[bi].qs[j] * dw_blocks[bi].qs[j];
                    result += (float)si * qx_d[bi] * fp16_to_fp32_lookup(dw_blocks[bi].d);
                }
                int t = assigned[a];
                size_t idx = (size_t)t * c->n_used * c->dim + (size_t)slots[a] * c->dim + row;
                c->down_out[idx] = result;
            }
        }
    } else if (n_assigned == 1) {
        float *dout = c->down_out + (size_t)assigned[0] * c->n_used * c->dim + (size_t)slots[0] * c->dim;
        matmul_q8_seq(dout, qx_bufs[0], qx_d_arr[0], dw, c->n_ff, c->dim, c->type);
    } else if (c->type == GGUF_TYPE_Q8_0) {
        /* >8 tokens per expert: batch4 + scalar fallback (Q8_0 only) */
        int n_batch4 = (n_assigned / 4) * 4;
        for (int row = 0; row < c->dim; row++) {
            const char *wi = dw + row * c->row_bytes;
            for (int a = 0; a < n_batch4; a += 4) {
                float fout[4];
                vec_dot_q8_0_q8_0_deltas_batch4(qx_bufs[a], qx_d_arr[a],
                    qx_bufs[a+1], qx_d_arr[a+1],
                    qx_bufs[a+2], qx_d_arr[a+2],
                    qx_bufs[a+3], qx_d_arr[a+3],
                    wi, c->n_ff,
                    &fout[0], &fout[1], &fout[2], &fout[3]);
                for (int k = 0; k < 4; k++) {
                    int t = assigned[a + k];
                    size_t idx = (size_t)t * c->n_used * c->dim + (size_t)slots[a + k] * c->dim + row;
                    c->down_out[idx] = fout[k];
                }
            }
            for (int a = n_batch4; a < n_assigned; a++) {
                int t = assigned[a];
                size_t idx = (size_t)t * c->n_used * c->dim + (size_t)slots[a] * c->dim + row;
                c->down_out[idx] = vec_dot_q8_0_q8_0_deltas(qx_bufs[a], qx_d_arr[a], wi, c->n_ff);
            }
        }
    } else {
        /* Non-Q8_0 weights: use matmul_q8_seq which handles all types */
        for (int a = 0; a < n_assigned; a++) {
            int t = assigned[a];
            int slot = slots[a];
            float *dout = c->down_out + (size_t)t * c->n_used * c->dim + (size_t)slot * c->dim;
            matmul_q8_seq(dout, qx_bufs[a], qx_d_arr[a], dw, c->n_ff, c->dim, c->type);
        }
    }
#else
    /* Non-AVX512: use matmul_q8_seq for all weight types */
    for (int a = 0; a < n_assigned; a++) {
        int t = assigned[a];
        int slot = slots[a];
        float *dout = c->down_out + (size_t)t * c->n_used * c->dim + (size_t)slot * c->dim;
        matmul_q8_seq(dout, qx_bufs[a], qx_d_arr[a], dw, c->n_ff, c->dim, c->type);
    }
#endif
}

/* GEMM threading context: dispatches picolm_sgemm_d via tensor_parallel_for.
 * Each worker thread gets a distinct idx (its ith), ensuring ALL output tiles
 * are computed. Without this, picolm_sgemm_d(..., ith=0, nth=16) would only
 * compute 1/16 of the output matrix. */
#if (defined(__AVX2__) && defined(__F16C__)) || defined(__ARM_NEON)
typedef struct {
    int m, n, k_blocks;
    const void *A; int lda;
    const block_q8_0 *B; int ldb;
    const float *B_d; int ldb_d;
    float *C; int ldc;
    int Atype;
    int nth;
    volatile int failed;   /* set by any task whose picolm_sgemm_d() returned 0 */
} qgemm_d_ctx_t;

/* Table-driven GEMM task wrapper: dispatches through type_info_t->fn_gemm with
 * correct per-thread ith/nth. Used by matmul_batch's GEMM_table path. */
typedef struct { int nrows, ncols, k; const void *w, *a; float *out; size_t bs;
                 type_gemm_fn fn; } gemm_table_ctx_t;

static void gemm_table_task(int idx, void *ctxp) {
    gemm_table_ctx_t *c = (gemm_table_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    c->fn(c->nrows, c->ncols, c->k, c->w, c->a, c->out, c->bs, idx, nth);
}

static void qgemm_d_task(int idx, void *ctxp) {
    qgemm_d_ctx_t *c = (qgemm_d_ctx_t *)ctxp;
    int ok = picolm_sgemm_d(c->m, c->n, c->k_blocks, c->A, c->lda,
                   c->B, c->ldb, c->B_d, c->ldb_d,
                   c->C, c->ldc, c->Atype, idx, c->nth);
    /* If picolm_sgemm_d returns 0 here, the output buffer C is partially
     * uninitialized. This should not happen because the caller checks
     * PICOLM_SGEMM=0 before entering this path. */
    if (!ok) {
        c->failed = 1;   /* callers that can fall back check this */
        if (idx == 0)
            fprintf(stderr, "WARN: qgemm_d_task returned 0 (m=%d n=%d k=%d)\n",
                    c->m, c->n, c->k_blocks);
    }
}

/* Non-delta GEMM worker: dispatches picolm_sgemm with Btype=Q8_0.
 *
 * TODO: This worker routes to sgemm_q4_q8_neon/sgemm_q5_q8_neon which are SLOWER
 * than the delta path. Only used when PICOLM_NEON_Q4=nondelta. The delta path
 * (qgemm_d_task -> picolm_sgemm_d) is the default and fastest on basic NEON. */
typedef struct {
    int m, n, k_blocks;
    const void *A; int lda;
    const void *B; int ldb;
    float *C; int ldc;
    int Atype, Btype;
    int nth;
} sgemm_q8_ctx_t;

static void sgemm_q8_worker(int idx, void *ctxp) {
    sgemm_q8_ctx_t *c = (sgemm_q8_ctx_t *)ctxp;
    int ok = picolm_sgemm(c->m, c->n, c->k_blocks, c->A, c->lda,
                 c->B, c->ldb, c->C, c->ldc,
                 c->Atype, c->Btype, idx, c->nth);
    if (!ok && idx == 0) {
        fprintf(stderr, "WARN: sgemm_q8_worker returned 0 (m=%d n=%d k=%d) -- output may be garbage\n",
                c->m, c->n, c->k_blocks);
    }
}
#endif /* AVX2+F16C || ARM NEON for qgemm_d_ctx_t / sgemm_q8_ctx_t */

/* Q4_K GEMM worker (AVX2+F16C/AVX1 or ARM NEON) */
#if (defined(__AVX2__) && defined(__F16C__)) || defined(__AVX__) || defined(__ARM_NEON)
typedef struct {
    int m, n, k_blocks_q4k;
    const void *A; int lda_q4k;
    const void *B; int ldb_q8k;
    float *C; int ldc;
    int nth;
} q4k_gemm_ctx_t;

static void q4k_gemm_task(int idx, void *ctxp) {
    q4k_gemm_ctx_t *c = (q4k_gemm_ctx_t *)ctxp;
    picolm_sgemm_d_q4k(c->m, c->n, c->k_blocks_q4k, c->A, c->lda_q4k,
                       c->B, c->ldb_q8k, c->C, c->ldc, idx, c->nth);
}
#endif /* AVX2+F16C || AVX1 || ARM_NEON for q4k_gemm_ctx_t */

#if defined(__AVX2__)
typedef struct {
    int m, n, k_blocks_q4xs;
    const void *A; int lda_q4xs;
    const void *B; int ldb_q8k;
    float *C; int ldc;
    int nth;
} iq4xs_gemm_ctx_t;

static void iq4xs_gemm_task(int idx, void *ctxp) {
    iq4xs_gemm_ctx_t *c = (iq4xs_gemm_ctx_t *)ctxp;
    picolm_sgemm_d_iq4xs(c->m, c->n, c->k_blocks_q4xs, c->A, c->lda_q4xs,
                         c->B, c->ldb_q8k, c->C, c->ldc, idx, c->nth);
}
#endif /* AVX2 for iq4xs_gemm_ctx_t */

static int picolm_sgemm_disabled_tensor(void) {
    static int checked = 0, disabled = 0;
    if (!checked) {
        const char *sv = getenv("PICOLM_SGEMM");
        disabled = sv && (sv[0] == '0' || (sv[0] == 'f' && sv[1] == 'a'));
        checked = 1;
    }
    return disabled;
}

/* qgemm_q4r8_ctx_t: used by both AVX2 and NEON GEMM task wrappers.
 * Must be defined outside #if PICOLM_AVX2 for NEON-only builds. */
typedef struct {
    int nr, nc, k;
    const void *w;
    const void *abuf;
    float *out;
    size_t bs;
} qgemm_q4r8_ctx_t;

#if defined(PICOLM_AVX2)
typedef struct {
    int nr, nc, k;
    const void *w;
    const void *abuf;
    float *out;
    size_t bs;
} qgemm_q4x8_ctx_t;

typedef struct {
    int nr, nc, k;
    const void *w;
    const void *abuf;
    float *out;
    size_t bs;
} qgemm_q4ix8_ctx_t;

static void qgemm_q4x8_task(int idx, void *ctxp) {
    qgemm_q4x8_ctx_t *c = (qgemm_q4x8_ctx_t *)ctxp;
    if (!picolm_sgemm_disabled_tensor()) {
        int nth = pool_total_threads(1);
        sgemm_q4_0x8_q8_0x4(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
    }
    /* If SGEMM disabled, output buffer is left untouched. Caller must
     * handle fallback. The Q4_0_8_8 dispatch in matmul_batch should
     * not be reached when PICOLM_SGEMM=0 (guarded at entry). */
}

/* Q4I_0_8_8 task (struct now in PICOLM_AVX2 block above) */
static void qgemm_q4ix8_task(int idx, void *ctxp) {
    qgemm_q4ix8_ctx_t *c = (qgemm_q4ix8_ctx_t *)ctxp;
    if (!picolm_sgemm_disabled_tensor()) {
        int nth = pool_total_threads(1);
        sgemm_q4i_0x8_q8_0x4(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
    }
}

/* Q4_0_R8 tiled GEMM task (Q8_2 activations with precomputed sum) */
static void qgemm_q4r8_task(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_q4_0_r8_q8_2_avx2(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* Q8_K_R8 tiled GEMM task (Q8_K activations, sign trick) */
static void qgemm_q8kr8_task(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_q8_k_r8_q8_k_avx2(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* IQ2_K_R4 tiled GEMM task (Q8_K activations, LUT dequant) */
static void qgemm_iq2kr4_task(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_iq2_k_r4_q8_k_avx2(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* IQ3_K_R4 tiled GEMM task (Q8_K activations, LUT dequant) */
static void qgemm_iq3kr4_task(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_iq3_k_r4_q8_k_avx2(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* IQ4_K_R4 tiled GEMM task (Q8_K activations, LUT dequant) */
static void qgemm_iq4kr4_task(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_iq4_k_r4_q8_k_avx2(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* Q4_K_R4 tiled GEMM task (Q8_K activations, raw 4-bit + bias correction) */
static void qgemm_q4kr4_task(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_q4_k_r4_q8_k_avx2(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* Q6_K_R4 tiled GEMM task (Q8_K activations, bsums bias correction) */
static void qgemm_q6kr4_task(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_q6_k_r4_q8_k(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* IQ6_K plain GEMM task (Q8_K activations, decode-once + sign trick) */
static void qgemm_iq6k_task(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_iq6_k_q8_k_avx2(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}
#endif /* PICOLM_AVX2 */

#ifdef PICOLM_NEON
/* NEON GEMM task wrappers */
static void qgemm_iq2kr4_task_neon(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_iq2_k_r4_q8_k_neon(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* Plain IQ2_K NEON GEMM task wrapper */
static void qgemm_iq2k_task_neon(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_iq2_k_q8_k_neon(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* Plain IQ3_K NEON GEMM task wrapper */
static void qgemm_iq3k_task_neon(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_iq3_k_q8_k_neon(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* Plain IQ4_K NEON GEMM task wrapper */
static void qgemm_iq4k_task_neon(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_iq4_k_q8_k_neon(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* Q8_K_R8 NEON GEMM task wrapper */
static void qgemm_q8kr8_task_neon(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_q8_k_r8_q8_k_neon(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}

/* IQ6_K plain NEON GEMM task wrapper */
static void qgemm_iq6k_task_neon(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_iq6_k_q8_k_neon(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}
#endif /* PICOLM_NEON */

/* Plain IQ4_K AVX2 GEMM task wrapper (outside NEON block) */
#if defined(PICOLM_AVX2)
static void qgemm_iq4k_task_avx2(int idx, void *ctxp) {
    qgemm_q4r8_ctx_t *c = (qgemm_q4r8_ctx_t *)ctxp;
    int nth = pool_total_threads(1);
    sgemm_iq4_k_q8_k_avx2(c->nr, c->nc, c->k, c->w, c->abuf, c->out, c->bs, idx, nth);
}
#endif

/* Profiling: per-path timing for matmul_batch (PICOLM_PROFILE=1) */
#ifdef _MSC_VER
#include <windows.h>
static inline double picolm_now(void) {
    static double freq = 0;
    if (!freq) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); freq = (double)f.QuadPart; }
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart / freq;
}
#else
#include <time.h>
static inline double picolm_now(void) {
#ifdef CLOCK_MONOTONIC
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}
#endif
static double prof_f32_gemm, prof_q8_d, prof_q4_d, prof_q5_d, prof_scalar_par, prof_scalar_seq, prof_other;
static int    cnt_f32_gemm, cnt_q8_d, cnt_q4_d, cnt_q5_d, cnt_scalar_par, cnt_scalar_seq, cnt_other;
static int    prof_active;
static void prof_print(void) {
    double total = prof_f32_gemm + prof_q8_d + prof_q4_d + prof_q5_d + prof_scalar_par + prof_scalar_seq + prof_other;
    if (total < 0.001) return;
    fprintf(stderr, "\n=== matmul_batch profile (total %.1fms) ===\n", total*1000);
    #define P(l,v,n) if(n) fprintf(stderr, "  %16s: %4d calls %7.1fms %5.1f%%\n", l, n, v*1000, v/total*100)
    P("f32_gemm", prof_f32_gemm, cnt_f32_gemm);
    P("neon_q4_d", prof_q4_d, cnt_q4_d);
    P("neon_q5_d", prof_q5_d, cnt_q5_d);
    P("neon_q8_d", prof_q8_d, cnt_q8_d);
    P("scalar_par", prof_scalar_par, cnt_scalar_par);
    P("scalar_seq", prof_scalar_seq, cnt_scalar_seq);
    P("other", prof_other, cnt_other);
    fprintf(stderr, "===========================================\n");
    #undef P
}

/* Forward decls for R8/R4 scalar worker tasks (defined after matmul_dual_batch). */
typedef struct { const void *W; float *out; const float *x; int n; int d; int d8; int ng; int n_batch; size_t rb; size_t block_stride; } q8kr8_batch_ctx_t;
typedef struct { const void *W; float *out; const float *x; int n; int d; int d8; int ng; int n_batch; size_t rb; size_t block_stride; } q4r8_batch_ctx_t;
typedef struct { float *out1; const void *W1; gguf_type_t qtype1; size_t rb1; float *out2; const void *W2; gguf_type_t qtype2; size_t rb2; const float *x; int n; int d; int d8; int ng; int n_batch; } r8_dual_ctx_t;
typedef struct { float *out1; const void *W1; gguf_type_t qtype1; int is_r4_1; float *out2; const void *W2; gguf_type_t qtype2; int is_r4_2; const float *x; int n; int d; int n_batch; size_t rb1; size_t rb2; } r4_scalar_dual_ctx_t;
static void q8kr8_batch_task(int idx, void *cp);
static void q4r8_batch_task(int idx, void *cp);
static void r8_dual_task(int idx, void *cp);
static void r4_scalar_dual_task(int idx, void *cp);
static void matmul_batch_scalar_threaded(float *out, const float *x, int n_batch, const void *W, int n, int d, gguf_type_t qtype, size_t rb);

/* ---- Table-driven matmul_batch (replaces ~2200 lines of if-else chains) ---- */

void matmul_batch(float *out, const float *x, int n_batch,
                   const void *W, int n, int d, gguf_type_t qtype) {
    /* Types without GEMM or qgemm_d support fall through to the threaded
     * worker path below, which uses fn_gemv per-row dispatch. */
    const type_info_t *ti_check = typeinfo(qtype);
    if (!ti_check) {
        fprintf(stderr, "ERROR: matmul_batch unknown qtype %d\n", qtype);
        memset(out, 0, (size_t)n_batch * d * sizeof(float));
        return;
    }

    static int init;
    if (!init && getenv("PICOLM_PROFILE")) { init = 1; prof_active = 1; atexit(prof_print); }
    double t0 = prof_active ? picolm_now() : 0;
    static const char *_dispatch_env = NULL;
    if (!_dispatch_env) _dispatch_env = getenv("PICOLM_DISPATCH");
    #define __DISPATCH_UNIQ_N(x, y) x##y
    #define __DISPATCH_UNIQ(x, y) __DISPATCH_UNIQ_N(x, y)
    #define DISPATCH(fmt) do { static int __DISPATCH_UNIQ(_d, __LINE__); if (_dispatch_env && !__DISPATCH_UNIQ(_d, __LINE__)) { __DISPATCH_UNIQ(_d, __LINE__)=1; fprintf(stderr, "DISPATCH matmul_batch: d=%d n=%d batch=%d qtype=%d -> " fmt "\n", d, n, n_batch, qtype); } } while(0)

    if (!W) {
        static int null_warn;
        if (!null_warn) {
            fprintf(stderr, "WARN: matmul_batch NULL weight (n=%d d=%d batch=%d qtype=%d) -> zero output\n", n, d, n_batch, qtype);
            null_warn = 1;
        }
        if (out) memset(out, 0, (size_t)n_batch * d * sizeof(float));
        DISPATCH("NULL (zero-filled)");
        return;
    }
#ifdef PICOLM_GPU
    if (gpu_tensor && n_batch > 0 && d > 0 && n > 0 && !getenv("PICOLM_PREFILL_CPU") && !getenv("PICOLM_SSM_PREFILL_CPU")) {
        gpu_assert_orchestrator("matmul_batch GPU dispatch");
        if (picolm_gpu_w4a16_matmul(gpu_tensor, out, x, n_batch, gpu_device)) {
            static int gpu_batch_count = 0;
            if (gpu_batch_count++ == 0) fprintf(stderr, "INFO: GPU WMMA batch matmul active\n");
            DISPATCH("GPU_WMMA");
            return;
        }
        if (picolm_gpu_matmul(gpu_tensor, out, x, n_batch, gpu_device)) {
            DISPATCH("GPU_host");
            return;
        }
    }
#endif
    const type_info_t *ti = typeinfo(qtype);
    size_t row_bytes = gguf_type_row_size(qtype, n);
    const char *wptr = (const char *)W;

    /* F32/F16: tiled GEMM */
    if (!ti || ti->act_fmt == ACT_FMT_F32) {
        if (picolm_sgemm(d, n_batch, n, wptr, n, x, n, out, d, qtype, GGUF_TYPE_F32, 0, 1)) {
            if (prof_active) { double dt = picolm_now()-t0; prof_f32_gemm+=dt; cnt_f32_gemm++; }
            DISPATCH("SGEMM_F32");
            return;
        }
        /* Fall back to generic vec_dot */
        if (n_batch <= 4 || d < matmul_min_rows) {
            for (int b = 0; b < n_batch; b++)
                for (int i = 0; i < d; i++)
                    out[b * d + i] = vec_dot(wptr + (size_t)i * row_bytes, x + (size_t)b * n, n, qtype);
            DISPATCH("vec_dot_F32");
            return;
        }
        /* Threaded F32 */
        int nt = pool_total_threads(n_threads);
        int want = n_threads < nt ? n_threads : nt;
        int active = pool_assign_rows(0, want, d);
        for (int t = 0; t < active; t++) {
            pool_tasks[t].out = out; pool_tasks[t].x = x;
            pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
            pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
            pool_tasks[t].d = d; pool_tasks[t].qtype = qtype;
            pool_tasks[t].n_batch = n_batch; pool_tasks[t].x_is_q8k = 0;
        }
        pool_clear_unused(active, nt);
        pool_init(nt); pool_wake(nt);
        matmul_worker_f(&pool_tasks[0]);
        pool_wait(nt);
        DISPATCH("threaded_F32");
        return;
    }

    /* Quantized path: try GEMM first, then fall back to threaded worker */
    quant_buf_t qb = quant_activations(qtype, x, n_batch, n, NULL, 0);
    if (!qb.qbuf) {
        /* Allocation failed: generic vec_dot fallback */
        for (int b = 0; b < n_batch; b++)
            for (int i = 0; i < d; i++)
                out[b * d + i] = vec_dot(wptr + (size_t)i * row_bytes, x + (size_t)b * n, n, qtype);
        DISPATCH("vec_dot_fallback");
        return;
    }

    /* Try tiled GEMM if available for this type.
     * Must use tensor_parallel_for to dispatch with correct ith/nth per thread. */
#if (defined(__AVX2__) && defined(__F16C__)) || defined(__ARM_NEON)
    if (ti->fn_gemm && n_batch >= 4) {
        int nth = pool_total_threads(1);
        /* Use a task wrapper that passes the table's fn_gemm with correct ith */
        gemm_table_ctx_t ctx = { d, n_batch, n, W, qb.qbuf, out, d, ti->fn_gemm };
        tensor_parallel_for(nth, gemm_table_task, &ctx);
        DISPATCH("GEMM_table");
        free_quant_buf(&qb, NULL);
        return;
    }
#endif

    /* Q4_K GEMM: uses picolm_sgemm_d_q4k via q4k_gemm_task */
#if (defined(__AVX2__) && defined(__F16C__)) || defined(__ARM_NEON)
    if (qtype == GGUF_TYPE_Q4_K && n_batch >= 8 && n % 256 == 0 && d >= 4) {
        int k_blocks = n / 256;
        int nth = pool_total_threads(1);
        q4k_gemm_ctx_t ctx4k = {
            .m = d, .n = n_batch, .k_blocks_q4k = k_blocks,
            .A = W, .lda_q4k = k_blocks,
            .B = qb.qbuf, .ldb_q8k = k_blocks,
            .C = out, .ldc = d, .nth = nth,
        };
        tensor_parallel_for(nth, q4k_gemm_task, &ctx4k);
        DISPATCH("Q4_K_GEMM_d");
        free_quant_buf(&qb, NULL);
        return;
    }
#endif

    /* IQ4_XS GEMM: uses picolm_sgemm_d_iq4xs via iq4xs_gemm_task */
#if defined(__AVX2__)
    if (qtype == GGUF_TYPE_IQ4_XS && n_batch >= 2 && n % 256 == 0 && d >= 4) {
        int k_blocks = n / 256;
        int nth = pool_total_threads(1);
        iq4xs_gemm_ctx_t ctxxs = {
            .m = d, .n = n_batch, .k_blocks_q4xs = k_blocks,
            .A = W, .lda_q4xs = k_blocks,
            .B = qb.qbuf, .ldb_q8k = k_blocks,
            .C = out, .ldc = d, .nth = nth,
        };
        tensor_parallel_for(nth, iq4xs_gemm_task, &ctxxs);
        DISPATCH("IQ4_XS_GEMM_d");
        free_quant_buf(&qb, NULL);
        return;
    }
#endif

    /* Try qgemm_d for standard Q8_0-activation types */
#if (defined(__AVX2__) && defined(__F16C__)) || defined(__ARM_NEON)
    if (ti->has_qgemm_d && n_batch >= 2) {
        /* picolm_sgemm_d handles Q8_0/Q4_0/IQ4_NL with pre-quantized activations.
         * Must use tensor_parallel_for to dispatch with correct ith/nth. */
        int nb = n / 32;
        int nth = pool_total_threads(n_threads);
        qgemm_d_ctx_t ctx = {
            .m = d, .n = n_batch, .k_blocks = nb,
            .A = W, .lda = n / 32,
            .B = (const block_q8_0 *)qb.qbuf, .ldb = nb,
            .B_d = qb.dbuf, .ldb_d = nb,
            .C = out, .ldc = d,
            .Atype = qtype, .nth = nth, .failed = 0,
        };
        tensor_parallel_for(nth, qgemm_d_task, &ctx);
        if (!ctx.failed) {
            DISPATCH("qgemm_d");
            free_quant_buf(&qb, NULL);
            return;
        }
        DISPATCH("qgemm_d_failed");
    }
#endif

    /* Threaded worker fallback */
    {
        int nt = pool_total_threads(n_threads);
        int want = n_threads < nt ? n_threads : nt;
        int active = pool_assign_rows(0, want, d);
        for (int t = 0; t < active; t++) {
            pool_tasks[t].out = out;
            pool_tasks[t].x = (const float *)qb.qbuf;
            pool_tasks[t].x_d = qb.dbuf;
            pool_tasks[t].W = wptr;
            pool_tasks[t].row_bytes = row_bytes;
            pool_tasks[t].n = n;
            pool_tasks[t].d = d;
            pool_tasks[t].qtype = qtype;
            pool_tasks[t].n_batch = n_batch;
            pool_tasks[t].x_is_q8k = (ti->act_fmt == ACT_FMT_Q8_K) ? 1 : 0;
        }
        pool_clear_unused(active, nt);
        pool_init(nt); pool_wake(nt);
        matmul_worker_f(&pool_tasks[0]);
        pool_wait(nt);
    }
    free_quant_buf(&qb, NULL);
    DISPATCH("threaded_worker");
    #undef DISPATCH
    #undef __DISPATCH_UNIQ
    #undef __DISPATCH_UNIQ_N
}

/* Dual-batch Q8_0 row task: one task per output row, processes all tokens */
typedef struct {
    float *out1, *out2;
    const void *W1, *W2;
    const float *x;              /* original float activations (for fallback) */
    const void *qx1_buf, *qx2_buf;
    const float *qx1_d, *qx2_d;
    size_t row_bytes1, row_bytes2, qx_stride1, qx_stride2;
    int n, d, n_batch;
    int have_qx1, have_qx2;     /* 1 if pre-quantized buffer valid */
    int out1_done, out2_done;   /* 1 if already computed by specialized handler */
    gguf_type_t qtype1, qtype2;
} dual_q8_row_ctx_t;

static void dual_q8_row_task(int i, void *ctxp) {
    dual_q8_row_ctx_t *c = (dual_q8_row_ctx_t *)ctxp;
    if (c->out1_done && c->out2_done) return;
    const char *wr1 = (const char *)c->W1 + (size_t)i * c->row_bytes1;
    const char *wr2 = (const char *)c->W2 + (size_t)i * c->row_bytes2;
    for (int b = 0; b < c->n_batch; b++) {
        if (!c->out1_done && c->have_qx1) {
            if (c->qtype1 == GGUF_TYPE_Q8_0) {
                const block_q8_0 *xq = (const block_q8_0 *)((const char *)c->qx1_buf + (size_t)b * c->qx_stride1);
                const float *xqd = c->qx1_d + (size_t)b * (c->n / 32);
                c->out1[b * c->d + i] = vec_dot_q8_0_q8_0_deltas(xq, xqd, (const block_q8_0*)wr1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_Q4_0) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_q4_0_q8_0(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_Q4_K) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_q4_K_q8_K(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_Q6_K) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_q6_K_q8_K(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_Q3_K) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_q3_K_q8_K(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_Q5_K) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_q5_K_q8_K(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_Q1_0) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_q1_0_q8_0(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_Q2_0) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_q2_0_q8_0(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_Q2_K) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_q2_K_q8_K(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_IQ4_NL) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_iq4_nl_q8_0(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_IQ2_K) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_iq2_k_q8_k(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_IQ3_K) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_iq3_k_q8_k(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_IQ4_K) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_iq4_k_q8_k(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_IQ6_K) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_iq6_k_q8_k(wr1, xb1, c->n);
            } else if (c->qtype1 == GGUF_TYPE_IQ4_XS) {
                const char *xb1 = (const char *)c->qx1_buf + (size_t)b * c->qx_stride1;
                c->out1[b * c->d + i] = vec_dot_iq4_xs_q8_k(wr1, xb1, c->n);
            } else {
                c->out1[b * c->d + i] = vec_dot(wr1, c->x + b * c->n, c->n, c->qtype1);
            }
        } else if (!c->out1_done) {
            c->out1[b * c->d + i] = vec_dot(wr1, c->x + b * c->n, c->n, c->qtype1);
        }
        if (!c->out2_done && c->have_qx2) {
            if (c->qtype2 == GGUF_TYPE_Q8_0) {
                const block_q8_0 *xq = (const block_q8_0 *)((const char *)c->qx2_buf + (size_t)b * c->qx_stride2);
                const float *xqd = c->qx2_d + (size_t)b * (c->n / 32);
                c->out2[b * c->d + i] = vec_dot_q8_0_q8_0_deltas(xq, xqd, (const block_q8_0*)wr2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_Q4_0) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_q4_0_q8_0(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_Q4_K) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_q4_K_q8_K(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_Q6_K) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_q6_K_q8_K(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_Q3_K) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_q3_K_q8_K(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_Q5_K) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_q5_K_q8_K(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_Q1_0) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_q1_0_q8_0(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_Q2_0) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_q2_0_q8_0(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_Q2_K) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_q2_K_q8_K(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_IQ4_NL) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_iq4_nl_q8_0(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_IQ2_K) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_iq2_k_q8_k(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_IQ3_K) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_iq3_k_q8_k(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_IQ4_K) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_iq4_k_q8_k(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_IQ6_K) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_iq6_k_q8_k(wr2, xb2, c->n);
            } else if (c->qtype2 == GGUF_TYPE_IQ4_XS) {
                const char *xb2 = (const char *)c->qx2_buf + (size_t)b * c->qx_stride2;
                c->out2[b * c->d + i] = vec_dot_iq4_xs_q8_k(wr2, xb2, c->n);
            } else {
                c->out2[b * c->d + i] = vec_dot(wr2, c->x + b * c->n, c->n, c->qtype2);
            }
        } else if (!c->out2_done) {
            c->out2[b * c->d + i] = vec_dot(wr2, c->x + b * c->n, c->n, c->qtype2);
        }
    }
}

/* Q4_0_8_8 GEMM helpers for dual-batch (used under PICOLM_AVX2, not tied to GCC intrinsics) */
#if defined(PICOLM_AVX2)
static void qgemm_q4x8_dual_single(const float *x, int n_batch, int d, int n,
                                    const void *W, float *out)
{
    int n_batch_padded = (n_batch + 15) & ~15;
    int n_act_rg = (n_batch_padded + 3) / 4;
    int nb = n / 32;
    size_t abuf_size = (size_t)n_act_rg * nb * sizeof(block_q8_0x4);
    void *abuf = malloc(abuf_size);
    if (!abuf) return;
    for (int rg = 0; rg < n_act_rg; rg++) {
        if (rg * 4 + 4 <= n_batch) {
            float *xr = (float *)x + (size_t)(rg * 4) * n;
            block_q8_0x4 *y = (block_q8_0x4 *)abuf + rg * nb;
            quantize_mat_q8_0x4(xr, y, n, n);
        } else if (rg * 4 < n_batch) {
            /* Tail row-group: see the matching comment in matmul()'s
             * Q4I_0_8_8/Q4_0_8_8 GEMM branches -- zeroing the whole
             * group here would silently drop 1..3 real rows. */
            int real = n_batch - rg * 4;
            float *scratch = (float *)calloc(4 * (size_t)n, sizeof(float));
            if (scratch) {
                memcpy(scratch, x + (size_t)(rg * 4) * n, (size_t)real * n * sizeof(float));
                block_q8_0x4 *y = (block_q8_0x4 *)abuf + rg * nb;
                quantize_mat_q8_0x4(scratch, y, n, n);
                free(scratch);
            } else {
                memset((block_q8_0x4 *)abuf + rg * nb, 0, (size_t)nb * sizeof(block_q8_0x4));
            }
        } else {
            memset((block_q8_0x4 *)abuf + rg * nb, 0, (size_t)nb * sizeof(block_q8_0x4));
        }
    }
    float *tmp = calloc(1, (size_t)n_batch_padded * d * sizeof(float));
    if (!tmp) { free(abuf); return; }
    int nth = pool_total_threads(1);
    /* sgemm_q4_0x8_q8_0x4's AVX2 kernel tiles 4 activation-rows x 8
     * weight-cols (see sgemm_q4_0x8.c), not the 16x16 AVX-512 tile size
     * this throttling heuristic was originally written for. */
    int n_tiles = (n_batch_padded / 4) * (d / 8);
    if (n_tiles < nth * 2) nth = n_tiles / 2;
    if (nth < 1) nth = 1;
    memset(tmp, 0, (size_t)n_batch_padded * d * sizeof(float));
    qgemm_q4x8_ctx_t ctx = {
        .nr = n_batch_padded, .nc = d, .k = n,
        .w = W, .abuf = abuf, .out = tmp, .bs = d,
    };
    tensor_parallel_for(nth, qgemm_q4x8_task, &ctx);
    for (int t = 0; t < n_batch; t++)
        memcpy(out + (size_t)t * d, tmp + (size_t)t * d, (size_t)d * sizeof(float));
    free(tmp); free(abuf);
}

/* Q4I_0_8_8 version: same structure but uses the Q4I GEMM kernel */
static void qgemm_q4ix8_dual_single(const float *x, int n_batch, int d, int n,
                                    const void *W, float *out)
{
    int n_batch_padded = (n_batch + 15) & ~15;
    int n_act_rg = (n_batch_padded + 3) / 4;
    int nb = n / 32;
    size_t abuf_size = (size_t)n_act_rg * nb * sizeof(block_q8_0x4);
    void *abuf = malloc(abuf_size);
    if (!abuf) return;
    for (int rg = 0; rg < n_act_rg; rg++) {
        if (rg * 4 + 4 <= n_batch) {
            float *xr = (float *)x + (size_t)(rg * 4) * n;
            block_q8_0x4 *y = (block_q8_0x4 *)abuf + rg * nb;
            quantize_mat_q8_0x4(xr, y, n, n);
        } else if (rg * 4 < n_batch) {
            int real = n_batch - rg * 4;
            float *scratch = (float *)calloc(4 * (size_t)n, sizeof(float));
            if (scratch) {
                memcpy(scratch, x + (size_t)(rg * 4) * n, (size_t)real * n * sizeof(float));
                block_q8_0x4 *y = (block_q8_0x4 *)abuf + rg * nb;
                quantize_mat_q8_0x4(scratch, y, n, n);
                free(scratch);
            } else {
                memset((block_q8_0x4 *)abuf + rg * nb, 0, (size_t)nb * sizeof(block_q8_0x4));
            }
        } else {
            memset((block_q8_0x4 *)abuf + rg * nb, 0, (size_t)nb * sizeof(block_q8_0x4));
        }
    }
    float *tmp = calloc(1, (size_t)n_batch_padded * d * sizeof(float));
    if (!tmp) { free(abuf); return; }
    int nth = pool_total_threads(1);
    int n_tiles = (n_batch_padded / 16) * (d / 16);
    if (n_tiles < nth * 2) nth = n_tiles / 2;
    if (nth < 1) nth = 1;
    memset(tmp, 0, (size_t)n_batch_padded * d * sizeof(float));
    qgemm_q4ix8_ctx_t ctx = {
        .nr = n_batch_padded, .nc = d, .k = n,
        .w = W, .abuf = abuf, .out = tmp, .bs = d,
    };
    tensor_parallel_for(nth, qgemm_q4ix8_task, &ctx);
    for (int t = 0; t < n_batch; t++)
        memcpy(out + (size_t)t * d, tmp + (size_t)t * d, (size_t)d * sizeof(float));
    free(tmp); free(abuf);
}

static void qgemm_q4x8_fallback(const float *x, int n_batch, int d, int n,
                                 const void *W, float *out, gguf_type_t qtype)
{
    size_t rb = gguf_type_row_size(qtype, n);
    for (int i = 0; i < d; i++) {
        const char *wr = (const char *)W + (size_t)i * rb;
        for (int b = 0; b < n_batch; b++)
            out[b * d + i] = vec_dot(wr, x + b * n, n, qtype);
    }
}
#endif /* PICOLM_AVX2 */

/* ---- Generalized R4-interleaved dispatch (matmul_dual_batch) ----
 * IQ2_K_R4/IQ3_K_R4/IQ4_K_R4 (and possibly more added later) all share the
 * same on-disk shape: 4 logical rows interleaved per shared block, so a
 * row's true byte offset is (row/4)*block_stride, NOT row*row_bytes. The
 * generic vec_dot(W + i*row_bytes, ...) fallback used for ordinary
 * (non-interleaved) quant types is silently wrong for any of these.
 *
 * Earlier fixes patched this one (type1, type2) pair at a time (see
 * commit 0a4e769, which added IQ4_K_R4-specific branches only inside the
 * IQ2_K_R4-triggered gate blocks). That approach doesn't generalize: e.g.
 * K=IQ3_K_R4 + V=IQ4_K_R4 still falls through to the broken generic
 * vec_dot for whichever side isn't IQ3_K_R4, silently reproducing the
 * exact "wrong stride -> huge garbage values -> INF in F16 KV cache ->
 * NaN cascade" failure -- and it does so *before* reaching the protective
 * assertion further down in this function, so it wouldn't even print a
 * warning. Any newly added R4 type would need the same manual patching
 * across every (type1, type2) ordering and every AVX2/scalar branch.
 *
 * This table-driven dispatch handles any combination of R4 types (and any
 * R4 type paired with a non-R4 type) uniformly, so adding a new R4 type
 * only requires one entry here instead of N new special cases. */
typedef void (*r4_vecdot4_fn)(const void *, const void *, int, float *, int);
typedef void (*r4_dequant_row_fn)(const void *, float *, int, int);

/* R4 dual-batch lookup: returns vecdot kernel, dequant function, and activation format.
 * The activation format is authoritative for choosing which buffer to pass.
 * Adding a new R4 type requires exactly one entry here. */
static int r4_dual_lookup(gguf_type_t qtype, r4_vecdot4_fn *vd, r4_dequant_row_fn *dq,
                           act_fmt_t *act_fmt) {
    switch (qtype) {
    case GGUF_TYPE_IQ2_K_R4:
        *vd = vec_dot_iq2_k_r4_q8_k_avx2;
        *dq = (r4_dequant_row_fn)dequantize_row_iq2_k_r4_single;
        *act_fmt = ACT_FMT_Q8_K;
        return 1;
    case GGUF_TYPE_IQ3_K_R4:
        *vd = vec_dot_iq3_k_r4_q8_k_avx2;
        *dq = (r4_dequant_row_fn)dequantize_row_iq3_k_r4_single;
        *act_fmt = ACT_FMT_Q8_K;
        return 1;
    case GGUF_TYPE_IQ4_K_R4:
        *vd = vec_dot_iq4_k_r4_q8_k_avx2;
        *dq = (r4_dequant_row_fn)dequantize_row_iq4_k_r4_single;
        *act_fmt = ACT_FMT_Q8_K;
        return 1;
    case GGUF_TYPE_IQ4_NL_R4:
        *vd = vec_dot_iq4_nl_r4_q8_0_avx2;
        *dq = (r4_dequant_row_fn)dequantize_row_iq4_nl_r4_single;
        *act_fmt = ACT_FMT_Q8_0;
        return 1;
    case GGUF_TYPE_Q4_K_R4:
        *vd = vec_dot_q4_k_r4_q8_k_avx2;
        *dq = (r4_dequant_row_fn)dequantize_row_q4_k_r4_single;
        *act_fmt = ACT_FMT_Q8_K;
        return 1;
    default:
        return 0;
    }
}

static int is_r4_dual_type(gguf_type_t qtype) {
    r4_vecdot4_fn vd; r4_dequant_row_fn dq; act_fmt_t af;
    return r4_dual_lookup(qtype, &vd, &dq, &af);
}

/* Generic R4 serial helper: dispatches to the correct vecdot kernel based on
 * activation format. The act_fmt parameter tells us whether to cast qx as
 * Q8_K or Q8_0. This eliminates the need for separate r4_dual_side_avx2 and
 * r4_dual_side_q8_0_avx2 functions. */
static void r4_dual_side_avx2(float *out_col, const void *W, gguf_type_t qtype,
                               const void *qx, int n, int d, act_fmt_t act_fmt) {
    r4_vecdot4_fn vd; r4_dequant_row_fn dq; act_fmt_t af;
    r4_dual_lookup(qtype, &vd, &dq, &af);
    assert(af == act_fmt);  /* Caller must pass the right buffer type */
    size_t block_stride = gguf_type_row_size(qtype, n) * 4;
    int d4 = (d / 4) * 4;
    for (int i = 0; i < d4; i += 4) {
        float results[4] = {0, 0, 0, 0};
        vd((const char *)W + (i / 4) * block_stride, qx, n, results, 4);
        for (int r = 0; r < 4; r++) out_col[i + r] = results[r];
    }
    for (int i = d4; i < d; i++) {
        float results[4] = {0, 0, 0, 0};
        vd((const char *)W + (i / 4) * block_stride, qx, n, results, 4);
        out_col[i] = results[i % 4];
    }
}

/* Generic R4 threaded task: activation buffer type is opaque, row bytes
 * are pre-computed by the dispatcher. One struct type for both Q8_K and Q8_0. */
typedef struct {
    const void *W;         /* weight matrix (R4 interleaved) */
    float *out;            /* output column (d floats per token) */
    const char *qbuf;      /* quantized activations, n_batch rows */
    int n;                 /* input dim (per token) */
    int d;                 /* output dim (per token) */
    int n_batch;           /* number of token rows */
    size_t q8_rb;          /* activation row size in bytes (Q8_K or Q8_0) */
    size_t block_stride;   /* R4 block-group stride (row_size * 4) */
    r4_vecdot4_fn vd;      /* 4-row vec_dot kernel */
} r4_dual_thread_task_t;

static void r4_dual_thread_task(int idx, void *ctxp) {
    r4_dual_thread_task_t *c = (r4_dual_thread_task_t *)ctxp;
    int i = idx * 4;
    const char *wgrp = (const char *)c->W + (size_t)idx * c->block_stride;
    for (int b = 0; b < c->n_batch; b++) {
        const void *qx = (const void *)(c->qbuf + (size_t)b * c->q8_rb);
        float results[4] = {0, 0, 0, 0};
        c->vd(wgrp, qx, c->n, results, 4);
        for (int r = 0; r < 4; r++)
            c->out[(size_t)b * c->d + i + r] = results[r];
    }
}

/* Threaded one-side R4 matmul: parallelize over d/4 row groups.
 * The activation format is looked up from the type table, so the right
 * buffer (Q8_K or Q8_0) is passed automatically. */
static int r4_dual_side_threaded(float *out_col, const void *W, gguf_type_t qtype,
                                  const char *qbuf, int n, int d, int n_batch) {
    r4_vecdot4_fn vd; r4_dequant_row_fn dq; act_fmt_t af;
    if (!r4_dual_lookup(qtype, &vd, &dq, &af)) return 0;
    size_t q8_rb = af == ACT_FMT_Q8_0 ?
        gguf_type_row_size(GGUF_TYPE_Q8_0, n) :
        (size_t)(n / 256) * sizeof(block_q8_K);
    size_t block_stride = gguf_type_row_size(qtype, n) * 4;
    if (d < 4 || d % 4 != 0) return 0;

    r4_dual_thread_task_t ctx = {
        .W = W, .out = out_col, .qbuf = qbuf,
        .n = n, .d = d, .n_batch = n_batch,
        .q8_rb = q8_rb, .block_stride = block_stride, .vd = vd,
    };
    int ngroups = d / 4;
    int nth = pool_total_threads(1);
    int want = n_threads < nth ? n_threads : nth;
    int active = want > ngroups ? ngroups : want;
    if (active < 2) {
        for (int g = 0; g < ngroups; g++) r4_dual_thread_task(g, &ctx);
    } else {
        tensor_parallel_for(ngroups, r4_dual_thread_task, &ctx);
    }
    return 1;
}

/* Portable scalar fallback: dequantize each row, plain F32 dot product. */
static void r4_dual_side_scalar(float *out_col, const void *W, gguf_type_t qtype,
                                 const float *xrow, int n, int d) {
    r4_vecdot4_fn vd; r4_dequant_row_fn dq; act_fmt_t af;
    r4_dual_lookup(qtype, &vd, &dq, &af);
    size_t block_stride = gguf_type_row_size(qtype, n) * 4;
    int d4 = (d / 4) * 4;
    float *tmp = (float *)malloc((size_t)n * sizeof(float));
    if (!tmp) return;
    for (int i = 0; i < d4; i += 4) {
        const char *wblock = (const char *)W + (i / 4) * block_stride;
        for (int r = 0; r < 4; r++) {
            dq(wblock, tmp, n, r);
            out_col[i + r] = vec_dot_f32_f32(tmp, xrow, n);
        }
    }
    for (int i = d4; i < d; i++) {
        const char *wblock = (const char *)W + (i / 4) * block_stride;
        dq(wblock, tmp, n, i % 4);
        out_col[i] = vec_dot_f32_f32(tmp, xrow, n);
    }
    free(tmp);
}

/* Q8_K_R8 threaded batch worker: one task = one 8-row group for one batch
 * token. Dequantizes to F32, then vec_dot_f32_f32 (AVX-optimized). */
static void q8kr8_batch_task(int idx, void *cp) {
    q8kr8_batch_ctx_t *c = (q8kr8_batch_ctx_t *)cp;
    int b = idx / c->ng;
    int g = idx % c->ng;
    const float *xb = c->x + (size_t)b * c->n;
    int i = g * 8;
    float *o = c->out + (size_t)b * c->d + i;
    const char *wblock = (const char *)c->W + (size_t)g * c->block_stride;
    float *tmp = (float *)malloc((size_t)c->n * sizeof(float));
    if (g * 8 < c->d8) {
        for (int r = 0; r < 8; r++) {
            dequantize_row_q8_k_r8_single(wblock, tmp, c->n, r);
            o[r] = vec_dot_f32_f32(tmp, xb, c->n);
        }
    } else {
        for (int r = 0; i + r < c->d; r++) {
            dequantize_row_q8_k_r8_single(wblock, tmp, c->n, r);
            o[r] = vec_dot_f32_f32(tmp, xb, c->n);
        }
    }
    free(tmp);
}

/* Q4_0_R8 threaded batch worker: one task = one 8-row group for one batch
 * token. Dequantizes to F32, then vec_dot_f32_f32 (AVX-optimized). */
static void q4r8_batch_task(int idx, void *cp) {
    q4r8_batch_ctx_t *c = (q4r8_batch_ctx_t *)cp;
    int b = idx / c->ng;
    int g = idx % c->ng;
    const float *xb = c->x + (size_t)b * c->n;
    int i = g * 8;
    float *o = c->out + (size_t)b * c->d + i;
    const char *wblock = (const char *)c->W + (size_t)g * c->block_stride;
    float *tmp = (float *)malloc((size_t)c->n * sizeof(float));
    if (g * 8 < c->d8) {
        for (int r = 0; r < 8; r++) {
            dequantize_row_q4_0_r8_single(wblock, tmp, c->n, r);
            o[r] = vec_dot_f32_f32(tmp, xb, c->n);
        }
    } else {
        for (int r = 0; i + r < c->d; r++) {
            dequantize_row_q4_0_r8_single(wblock, tmp, c->n, r);
            o[r] = vec_dot_f32_f32(tmp, xb, c->n);
        }
    }
    free(tmp);
}

/* Q8_K_R8 / Q4_0_R8 threaded dual-batch worker: one task = one 8-row group
 * for one batch token. Handles both W1 and W2, supports mixed R8/non-R8. */
static void r8_dual_task(int idx, void *cp) {
    r8_dual_ctx_t *c = (r8_dual_ctx_t *)cp;
    int b = idx / c->ng;
    int g = idx % c->ng;
    const float *xb = c->x + (size_t)b * c->n;
    int i = g * 8;
    float *o1 = c->out1 + (size_t)b * c->d + i;
    float *o2 = c->out2 + (size_t)b * c->d + i;
    float *tmp = (float *)malloc((size_t)c->n * sizeof(float));

    if (c->qtype1 == GGUF_TYPE_Q8_K_R8) {
        size_t bs = c->rb1 * 8;
        const char *wb = (const char *)c->W1 + (size_t)g * bs;
        int rows = (g * 8 < c->d8) ? 8 : (c->d - i);
        for (int r = 0; r < rows; r++) {
            dequantize_row_q8_k_r8_single(wb, tmp, c->n, r);
            o1[r] = vec_dot_f32_f32(tmp, xb, c->n);
        }
    } else if (c->qtype1 == GGUF_TYPE_Q4_0_R8) {
        size_t bs = c->rb1 * 8;
        const char *wb = (const char *)c->W1 + (size_t)g * bs;
        int rows = (g * 8 < c->d8) ? 8 : (c->d - i);
        for (int r = 0; r < rows; r++) {
            dequantize_row_q4_0_r8_single(wb, tmp, c->n, r);
            o1[r] = vec_dot_f32_f32(tmp, xb, c->n);
        }
    } else {
        for (int ri = 0; i + ri < c->d; ri++)
            o1[ri] = vec_dot((const char *)c->W1 + (i + ri) * c->rb1, xb, c->n, c->qtype1);
    }

    if (c->qtype2 == GGUF_TYPE_Q8_K_R8) {
        size_t bs = c->rb2 * 8;
        const char *wb = (const char *)c->W2 + (size_t)g * bs;
        int rows = (g * 8 < c->d8) ? 8 : (c->d - i);
        for (int r = 0; r < rows; r++) {
            dequantize_row_q8_k_r8_single(wb, tmp, c->n, r);
            o2[r] = vec_dot_f32_f32(tmp, xb, c->n);
        }
    } else if (c->qtype2 == GGUF_TYPE_Q4_0_R8) {
        size_t bs = c->rb2 * 8;
        const char *wb = (const char *)c->W2 + (size_t)g * bs;
        int rows = (g * 8 < c->d8) ? 8 : (c->d - i);
        for (int r = 0; r < rows; r++) {
            dequantize_row_q4_0_r8_single(wb, tmp, c->n, r);
            o2[r] = vec_dot_f32_f32(tmp, xb, c->n);
        }
    } else {
        for (int ri = 0; i + ri < c->d; ri++)
            o2[ri] = vec_dot((const char *)c->W2 + (i + ri) * c->rb2, xb, c->n, c->qtype2);
    }

    free(tmp);
}

/* Threaded scalar R4 dual-batch worker: one task = one 4-row group for one
 * batch token. Processes both W1 and W2 sides. */
static void r4_scalar_dual_task(int idx, void *cp) {
    r4_scalar_dual_ctx_t *c = (r4_scalar_dual_ctx_t *)cp;
    int d4 = c->d / 4;
    int b = idx / d4;
    int gr = idx % d4;
    const float *xb = c->x + (size_t)b * c->n;
    int i = gr * 4;
    float *o1 = c->out1 + (size_t)b * c->d + i;
    float *o2 = c->out2 + (size_t)b * c->d + i;

    if (c->is_r4_1) {
        r4_dequant_row_fn dq; r4_vecdot4_fn vd; act_fmt_t af;
        r4_dual_lookup(c->qtype1, &vd, &dq, &af);
        size_t block_stride = c->rb1 * 4;
        const char *wblock = (const char *)c->W1 + (size_t)gr * block_stride;
        float *tmp = (float *)malloc((size_t)c->n * sizeof(float));
        for (int r = 0; r < 4; r++) {
            dq(wblock, tmp, c->n, r);
            o1[r] = vec_dot_f32_f32(tmp, xb, c->n);
        }
        free(tmp);
    } else {
        for (int ri = 0; ri < 4; ri++)
            o1[ri] = vec_dot((const char *)c->W1 + (i + ri) * c->rb1, xb, c->n, c->qtype1);
    }

    if (c->is_r4_2) {
        r4_dequant_row_fn dq; r4_vecdot4_fn vd; act_fmt_t af;
        r4_dual_lookup(c->qtype2, &vd, &dq, &af);
        size_t block_stride = c->rb2 * 4;
        const char *wblock = (const char *)c->W2 + (size_t)gr * block_stride;
        float *tmp = (float *)malloc((size_t)c->n * sizeof(float));
        for (int r = 0; r < 4; r++) {
            dq(wblock, tmp, c->n, r);
            o2[r] = vec_dot_f32_f32(tmp, xb, c->n);
        }
        free(tmp);
    } else {
        for (int ri = 0; ri < 4; ri++)
            o2[ri] = vec_dot((const char *)c->W2 + (i + ri) * c->rb2, xb, c->n, c->qtype2);
    }
}

/* Threaded scalar vec_dot for non-interleaved types in dual-batch.
 * One task per (batch, row) pair. */
typedef struct { const void *W; float *out; const float *x; int n; int d; int n_batch; size_t rb; gguf_type_t qt; } nr8_batch_ctx_t;

static void nr8_batch_task(int idx, void *cp) {
    nr8_batch_ctx_t *c = (nr8_batch_ctx_t *)cp;
    int b = idx / c->d;
    int i = idx % c->d;
    c->out[b * c->d + i] = vec_dot((const char *)c->W + (size_t)i * c->rb, c->x + (size_t)b * c->n, c->n, c->qt);
}

static void matmul_batch_scalar_threaded(float *out, const float *x, int n_batch, const void *W, int n, int d, gguf_type_t qtype, size_t rb) {
    nr8_batch_ctx_t ctx = { .W = W, .out = out, .x = x, .n = n, .d = d, .n_batch = n_batch, .rb = rb, .qt = qtype };
    tensor_parallel_for(n_batch * d, nr8_batch_task, &ctx);
}

void matmul_dual_batch(float *out1, float *out2, const float *x, int n_batch,
                        const void *W1, const void *W2,
                        int n, int d, gguf_type_t qtype1, gguf_type_t qtype2) {
    static const char *_dispatch_env2 = NULL;
    if (!_dispatch_env2) _dispatch_env2 = getenv("PICOLM_DISPATCH");
    #define __DISPATCH2_UNIQ_N(x, y) x##y
    #define __DISPATCH2_UNIQ(x, y) __DISPATCH2_UNIQ_N(x, y)
    #define DISPATCH2(fmt) do { static int __DISPATCH2_UNIQ(_d, __LINE__); if (_dispatch_env2 && !__DISPATCH2_UNIQ(_d, __LINE__)) { __DISPATCH2_UNIQ(_d, __LINE__)=1; fprintf(stderr, "DISPATCH matmul_dual_batch: d=%d n=%d batch=%d qtype1=%d qtype2=%d -> " fmt "\n", d, n, n_batch, qtype1, qtype2); } } while(0)

    /* Null weight handling: zero-fill output and skip that side. */
    int out1_done = 0, out2_done = 0;
    if (!W1 && out1) { memset(out1, 0, (size_t)n_batch * d * sizeof(float)); out1_done = 1; DISPATCH2("NULL (W1 zero-filled)"); }
    if (!W2 && out2) { memset(out2, 0, (size_t)n_batch * d * sizeof(float)); out2_done = 1; DISPATCH2("NULL (W2 zero-filled)"); }
    if (!W1 && !W2) return;

#ifdef PICOLM_GPU
    if (getenv("PICOLM_GPU")) {
        fprintf(stderr, "WARN: matmul_dual_batch (CPU only, no GPU path) n=%d d=%d batch=%d qtype=%d/%d\n",
            n, d, n_batch, qtype1, qtype2);
    }
#endif

    /* Table-driven dispatch for both sides.
     * Strategy: check if both sides are "standard" types (qgemm_d compatible).
     * If yes, quantize activations once and dispatch both sides via qgemm_d.
     * If one or both sides are specialized (R4, R8, Q4_0_8_8, etc.), fall back
     * to the old implementation which handles those cases correctly. */
    const type_info_t *ti1 = typeinfo(qtype1);
    const type_info_t *ti2 = typeinfo(qtype2);

    /* Both sides standard Q8_0-activation types: use qgemm_d for both. */
#if (defined(__AVX2__) && defined(__F16C__)) || defined(__ARM_NEON)
    if (ti1 && ti2 && ti1->has_qgemm_d && ti2->has_qgemm_d && n_batch >= 8 && d >= 4 && n > 0) {
        /* PICOLM_SGEMM guard */
        { static const char *_sgemm_env = NULL;
          if (!_sgemm_env) _sgemm_env = getenv("PICOLM_SGEMM");
          int _sgemm_off = _sgemm_env && (_sgemm_env[0] == '0' || (_sgemm_env[0] == 'f' && _sgemm_env[1] == 'a'));
          if (!_sgemm_off) {
            size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
            int nb = n / 32;
            void *qbuf = malloc((size_t)n_batch * q8_rb);
            float *dbuf = (float *)malloc((size_t)n_batch * nb * sizeof(float));
            if (qbuf && dbuf) {
                for (int b = 0; b < n_batch; b++) {
                    quantize_row_q8_0(x + (size_t)b * n, (char *)qbuf + (size_t)b * q8_rb, n);
                    const block_q8_0 *blk = (const block_q8_0 *)((char *)qbuf + (size_t)b * q8_rb);
                    for (int k = 0; k < nb; k++) dbuf[(size_t)b * nb + k] = fp16_to_fp32(blk[k].d);
                }
                int nth = pool_total_threads(1);
                qgemm_d_ctx_t ctx1 = {
                    .m = d, .n = n_batch, .k_blocks = nb,
                    .A = W1, .lda = nb,
                    .B = (const block_q8_0*)qbuf, .ldb = nb,
                    .B_d = dbuf, .ldb_d = nb,
                    .C = out1, .ldc = d,
                    .Atype = qtype1, .nth = nth,
                };
                tensor_parallel_for(nth, qgemm_d_task, &ctx1);
                qgemm_d_ctx_t ctx2 = {
                    .m = d, .n = n_batch, .k_blocks = nb,
                    .A = W2, .lda = nb,
                    .B = (const block_q8_0*)qbuf, .ldb = nb,
                    .B_d = dbuf, .ldb_d = nb,
                    .C = out2, .ldc = d,
                    .Atype = qtype2, .nth = nth,
                };
                tensor_parallel_for(nth, qgemm_d_task, &ctx2);
                free(qbuf); free(dbuf);
                DISPATCH2("GEMM_d_dual_table");
                return;
            }
          }
        }
    }
#endif

    /* Fall back: per-row dispatch matching the old dual-batch's scalar_vec_dot path.
     * The old code had an explicit if-else handler for specific types (quantizing
     * activations to Q8_0/Q8_K), and fell through to vec_dot() with F32 activations
     * for unhandled types. We replicate this by checking has_dual_scalar_path. */
    if (ti1 && ti2 && ti1->fn_gemv && ti2->fn_gemv && ti1->has_dual_scalar_path && ti2->has_dual_scalar_path) {
        /* Both sides have fn_gemv: quantize activations and dispatch per-row.
         * For Q8_0: quantize to Q8_0 + compute FP16 round-trip deltas.
         * For Q8_K types: quantize to Q8_K. */
        act_fmt_t af1 = ti1->act_fmt;
        act_fmt_t af2 = ti2->act_fmt;

        /* Quantize side1 */
        size_t qstride1 = 0;
        void *qbuf1 = NULL;
        float *dbuf1 = NULL;
        if (af1 == ACT_FMT_Q8_0) {
            /* Q8_0: buffer = qs + deltas appended */
            size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
            int nb = n / 32;
            qstride1 = q8_rb + (size_t)nb * sizeof(float);
            qbuf1 = malloc((size_t)n_batch * qstride1);
            if (qbuf1) {
                for (int b = 0; b < n_batch; b++) {
                    char *qb = (char *)qbuf1 + b * qstride1;
                    quantize_row_q8_0(x + b * n, qb, n);
                    /* Compute FP16 round-trip deltas (matching old dual-batch) */
                    const block_q8_0 *blk = (const block_q8_0 *)qb;
                    float *db = (float *)(qb + q8_rb);
                    for (int k = 0; k < nb; k++) db[k] = fp16_to_fp32(blk[k].d);
                }
            }
        } else if (af1 == ACT_FMT_Q8_K) {
            qstride1 = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
            qbuf1 = malloc((size_t)n_batch * qstride1);
            if (qbuf1) {
                for (int b = 0; b < n_batch; b++)
                    quantize_row_q8_K(x + b * n, (char *)qbuf1 + b * qstride1, n);
            }
        }

        /* Quantize side2 (reuse side1 if same format) */
        size_t qstride2 = 0;
        void *qbuf2 = NULL;
        float *dbuf2 = NULL;
        if (af1 == af2 && qbuf1) {
            qbuf2 = qbuf1; qstride2 = qstride1; dbuf2 = dbuf1;
        } else if (af2 == ACT_FMT_Q8_0) {
            size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
            int nb = n / 32;
            qstride2 = q8_rb + (size_t)nb * sizeof(float);
            qbuf2 = malloc((size_t)n_batch * qstride2);
            if (qbuf2) {
                for (int b = 0; b < n_batch; b++) {
                    char *qb = (char *)qbuf2 + b * qstride2;
                    quantize_row_q8_0(x + b * n, qb, n);
                    const block_q8_0 *blk = (const block_q8_0 *)qb;
                    float *db = (float *)(qb + q8_rb);
                    for (int k = 0; k < nb; k++) db[k] = fp16_to_fp32(blk[k].d);
                }
            }
        } else if (af2 == ACT_FMT_Q8_K) {
            qstride2 = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
            qbuf2 = malloc((size_t)n_batch * qstride2);
            if (qbuf2) {
                for (int b = 0; b < n_batch; b++)
                    quantize_row_q8_K(x + b * n, (char *)qbuf2 + b * qstride2, n);
            }
        }

        /* Dispatch per-row using fn_gemv */
        size_t rb1 = gguf_type_row_size(qtype1, n);
        size_t rb2 = gguf_type_row_size(qtype2, n);
        int rpb1 = ti1->rows_per_block;
        int rpb2 = ti2->rows_per_block;
        size_t bs1 = rpb1 > 1 ? rb1 * rpb1 : rb1;
        size_t bs2 = rpb2 > 1 ? rb2 * rpb2 : rb2;

        for (int b = 0; b < n_batch; b++) {
            const char *xb1 = qbuf1 ? (const char *)qbuf1 + b * qstride1 : NULL;
            const char *xb2 = qbuf2 ? (const char *)qbuf2 + b * qstride2 : NULL;
            if (rpb1 == 1 && rpb2 == 1) {
                for (int i = 0; i < d; i++) {
                    if (ti1->fn_gemv && xb1) { float r; ti1->fn_gemv((const char *)W1 + i * rb1, xb1, n, &r, 1); out1[b * d + i] = r; }
                    else { out1[b * d + i] = vec_dot((const char *)W1 + i * rb1, x + b * n, n, qtype1); }
                    if (ti2->fn_gemv && xb2) { float r; ti2->fn_gemv((const char *)W2 + i * rb2, xb2, n, &r, 1); out2[b * d + i] = r; }
                    else { out2[b * d + i] = vec_dot((const char *)W2 + i * rb2, x + b * n, n, qtype2); }
                }
            } else {
                /* Interleaved: process in groups */
                int d1g = (d + rpb1 - 1) / rpb1 * rpb1;
                int d2g = (d + rpb2 - 1) / rpb2 * rpb2;
                int dg = d1g > d2g ? d1g : d2g;
                for (int g = 0; g < dg; g++) {
                    int i1 = (rpb1 == 1) ? g : (g / rpb1) * rpb1;
                    int i2 = (rpb2 == 1) ? g : (g / rpb2) * rpb2;
                    if (i1 >= d && i2 >= d) break;
                    float res1[TYPE_INFO_MAX_RPB], res2[TYPE_INFO_MAX_RPB];
                    if (ti1->fn_gemv && xb1 && i1 < d) {
                        ti1->fn_gemv((const char *)W1 + (i1 / rpb1) * bs1, xb1, n, res1, rpb1);
                        for (int r = 0; r < rpb1 && i1 + r < d; r++) out1[b * d + i1 + r] = res1[r];
                    }
                    if (ti2->fn_gemv && xb2 && i2 < d) {
                        ti2->fn_gemv((const char *)W2 + (i2 / rpb2) * bs2, xb2, n, res2, rpb2);
                        for (int r = 0; r < rpb2 && i2 + r < d; r++) out2[b * d + i2 + r] = res2[r];
                    }
                }
            }
        }

        free(qbuf1);
        if (qbuf2 && qbuf2 != qbuf1) free(qbuf2);
        DISPATCH2("dual_fallback_gemv");
    } else {
        /* Fall back: dispatch each side independently via matmul_batch. */
        matmul_batch(out1, x, n_batch, W1, n, d, qtype1);
        matmul_batch(out2, x, n_batch, W2, n, d, qtype2);
        DISPATCH2("dual_fallback_matmul_batch");
    }
}

/* ================================================================
 * SIMD-accelerated basic operations
 * ================================================================ */

void rmsnorm(float *out, const float *x, const float *weight, int size, float eps) {
    float ss = 0.0f;
    int i;

#ifdef PICOLM_AVX512
    __m512 acc = _mm512_setzero_ps();
    i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    ss = _mm512_reduce_add_ps(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#elif defined(PICOLM_NEON)
    float32x4_t acc = vdupq_n_f32(0);
    i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vmlaq_f32(acc, v, v);
    }
    ss = vaddvq_f32_compat(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#elif defined(PICOLM_AVX)
    __m256 acc = _mm256_setzero_ps();
    i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
    }
    ss = hsum_avx(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#elif defined(PICOLM_SSE2)
    __m128 acc = _mm_setzero_ps();
    i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
    }
    ss = hsum_sse(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#elif defined(PICOLM_ALTIVEC)
    {        static char __rnbuf[192];
        unsigned long ba = (unsigned long)__rnbuf + 63;
        ba = ba / 64 * 64;
        vector float zero = (vector float)vec_splat_u32(0);
        vector float acc = zero;
        i = 0;
        for (; i + 3 < size; i += 4) {
            memcpy((void*)ba, x + i, 16);
            vector float v = vec_ld(0, (float*)ba);
            acc = vec_madd(v, v, acc);
        }
        ss = hsum_altivec(acc);
        for (; i < size; i++) ss += x[i] * x[i];
    }
#else
    for (i = 0; i < size; i++) ss += x[i] * x[i];
#endif

    ss = 1.0f / sqrtf(ss / (float)size + eps);

#ifdef PICOLM_AVX512
    __m512 scale = _mm512_set1_ps(ss);
    i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 w = _mm512_loadu_ps(weight + i);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(v, scale), w));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
#elif defined(PICOLM_NEON)
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
#elif defined(PICOLM_ALTIVEC)
    {
        static char __rnbuf2[192];
        unsigned long ba = (unsigned long)__rnbuf2 + 63;
        ba = ba / 64 * 64;
        unsigned long bs = ba;
        ((float*)(void*)bs)[0] = ss;
        vector float sc = vec_splat(vec_ld(0, (float*)bs), 0);
        vector float zero = (vector float)vec_splat_u32(0);
        i = 0;
        for (; i + 3 < size; i += 4) {
            memcpy((void*)(ba + 16), x + i, 16);
            memcpy((void*)(ba + 32), weight + i, 16);
            vector float vx2 = vec_ld(0, (float*)(ba + 16));
            vector float wx = vec_ld(0, (float*)(ba + 32));
            vector float result = vec_madd(vx2, sc, zero);
            result = vec_madd(result, wx, zero);
            vec_st(result, 0, (float*)(ba + 48));
            memcpy(out + i, (void*)(ba + 48), 16);
        }
        for (; i < size; i++) out[i] = x[i] * ss * weight[i];
    }
#else
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
#endif
}

/* LayerNorm (with weight and bias) - GPT-2 style.
 * out = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias */
void layernorm(float *out, const float *x, const float *weight, const float *bias, int size, float eps) {
    float ss = 0.0f;
    int i;

    /* Compute mean */
#ifdef PICOLM_AVX512
    __m512 acc = _mm512_setzero_ps();
    i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        acc = _mm512_add_ps(acc, v);
    }
    ss = _mm512_reduce_add_ps(acc);
    for (; i < size; i++) ss += x[i];
#elif defined(PICOLM_NEON)
    float32x4_t acc = vdupq_n_f32(0);
    i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vaddq_f32(acc, v);
    }
    ss = vaddvq_f32_compat(acc);
    for (; i < size; i++) ss += x[i];
#elif defined(PICOLM_AVX)
    __m256 acc = _mm256_setzero_ps();
    i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_add_ps(acc, v);
    }
    ss = hsum_avx(acc);
    for (; i < size; i++) ss += x[i];
#elif defined(PICOLM_SSE2)
    __m128 acc = _mm_setzero_ps();
    i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        acc = _mm_add_ps(acc, v);
    }
    ss = hsum_sse(acc);
    for (; i < size; i++) ss += x[i];
#else
    for (i = 0; i < size; i++) ss += x[i];
#endif
    float mean = ss / (float)size;

    /* Compute variance and apply normalization */
    ss = 0.0f;
#ifdef PICOLM_AVX512
    __m512 mean_v = _mm512_set1_ps(mean);
    __m512 acc2 = _mm512_setzero_ps();
    i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 d = _mm512_sub_ps(v, mean_v);
        acc2 = _mm512_fmadd_ps(d, d, acc2);
    }
    ss = _mm512_reduce_add_ps(acc2);
    for (; i < size; i++) { float d = x[i] - mean; ss += d * d; }
    __m512 scale = _mm512_set1_ps(1.0f / sqrtf(ss / (float)size + eps));
    i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 w = _mm512_loadu_ps(weight + i);
        __m512 b = _mm512_loadu_ps(bias + i);
        __m512 d = _mm512_sub_ps(v, mean_v);
        _mm512_storeu_ps(out + i, _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(d, scale), w), b));
    }
    for (; i < size; i++) out[i] = (x[i] - mean) * (float)(1.0f / sqrtf(ss / (float)size + eps)) * weight[i] + bias[i];
#elif defined(PICOLM_NEON)
    float32x4_t mean_v = vdupq_n_f32(mean);
    float32x4_t acc2 = vdupq_n_f32(0);
    i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t d = vsubq_f32(v, mean_v);
        acc2 = vmlaq_f32(acc2, d, d);
    }
    ss = vaddvq_f32_compat(acc2);
    for (; i < size; i++) { float d = x[i] - mean; ss += d * d; }
    float scale = 1.0f / sqrtf(ss / (float)size + eps);
    float32x4_t scale_v = vdupq_n_f32(scale);
    i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t w = vld1q_f32(weight + i);
        float32x4_t b = vld1q_f32(bias + i);
        float32x4_t d = vsubq_f32(v, mean_v);
        vst1q_f32(out + i, vaddq_f32(vmulq_f32(vmulq_f32(d, scale_v), w), b));
    }
    for (; i < size; i++) out[i] = (x[i] - mean) * scale * weight[i] + bias[i];
#elif defined(PICOLM_AVX)
    __m256 mean_v = _mm256_set1_ps(mean);
    __m256 acc2 = _mm256_setzero_ps();
    i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 d = _mm256_sub_ps(v, mean_v);
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d, d));
    }
    ss = hsum_avx(acc2);
    for (; i < size; i++) { float d = x[i] - mean; ss += d * d; }
    float scale = 1.0f / sqrtf(ss / (float)size + eps);
    __m256 scale_v = _mm256_set1_ps(scale);
    __m256 mean_v2 = _mm256_set1_ps(mean);
    i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        __m256 b = _mm256_loadu_ps(bias + i);
        __m256 d = _mm256_sub_ps(v, mean_v2);
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(d, scale_v), w), b));
    }
    for (; i < size; i++) out[i] = (x[i] - mean) * scale * weight[i] + bias[i];
#elif defined(PICOLM_SSE2)
    __m128 mean_v = _mm_set1_ps(mean);
    __m128 acc2 = _mm_setzero_ps();
    i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        __m128 d = _mm_sub_ps(v, mean_v);
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(d, d));
    }
    ss = hsum_sse(acc2);
    for (; i < size; i++) { float d = x[i] - mean; ss += d * d; }
    float scale = 1.0f / sqrtf(ss / (float)size + eps);
    __m128 scale_v = _mm_set1_ps(scale);
    __m128 mean_v2 = _mm_set1_ps(mean);
    i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        __m128 w = _mm_loadu_ps(weight + i);
        __m128 b = _mm_loadu_ps(bias + i);
        __m128 d = _mm_sub_ps(v, mean_v2);
        _mm_storeu_ps(out + i, _mm_add_ps(_mm_mul_ps(_mm_mul_ps(d, scale_v), w), b));
    }
    for (; i < size; i++) out[i] = (x[i] - mean) * scale * weight[i] + bias[i];
#else
    /* Scalar variance accumulation (missing from SSSE3 path below) */
    for (i = 0; i < size; i++) { float d = x[i] - mean; ss += d * d; }
    float variance = ss / (float)size;
    float inv_std = 1.0f / sqrtf(variance + eps);
    for (i = 0; i < size; i++) {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
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

#ifdef PICOLM_AVX512
    __m512 inv_v = _mm512_set1_ps(inv);
    int i = 0;
    for (; i + 15 < size; i += 16) {
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), inv_v));
    }
    for (; i < size; i++) x[i] *= inv;
#elif defined(PICOLM_NEON)
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
#if defined(PICOLM_AVX) && !defined(PICOLM_AVX512)
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

/* AVX-512 RoPE: 8 complex pairs/iter */
#ifdef PICOLM_AVX512
static void rope_avx512(float *h, int half, const float *cos_pos, const float *sin_pos) {
    int i = 0;
    for (; i + 7 < half; i += 8) {
        __m512 v = _mm512_loadu_ps(h + i * 2);
        __m256 c8 = _mm256_loadu_ps(cos_pos + i);
        __m256 s8 = _mm256_loadu_ps(sin_pos + i);
        /* Broadcast each cos/sin to 2 copies: [c0,c0,c1,c1,...,c7,c7] */
        __m128 c03 = _mm256_castps256_ps128(c8);
        __m128 c47 = _mm256_extractf128_ps(c8, 1);
        __m128 c00 = _mm_unpacklo_ps(c03, c03); /* [c0,c0,c1,c1] */
        __m128 c22 = _mm_unpackhi_ps(c03, c03); /* [c2,c2,c3,c3] */
        __m128 c44 = _mm_unpacklo_ps(c47, c47); /* [c4,c4,c5,c5] */
        __m128 c66 = _mm_unpackhi_ps(c47, c47); /* [c6,c6,c7,c7] */
        __m512 cv = _mm512_castps128_ps512(c00);
        cv = _mm512_insertf32x4(cv, c22, 1);
        cv = _mm512_insertf32x4(cv, c44, 2);
        cv = _mm512_insertf32x4(cv, c66, 3);
        __m128 s03 = _mm256_castps256_ps128(s8);
        __m128 s47 = _mm256_extractf128_ps(s8, 1);
        __m128 s00 = _mm_unpacklo_ps(s03, s03); /* [s0,s0,s1,s1] */
        __m128 s22 = _mm_unpackhi_ps(s03, s03); /* [s2,s2,s3,s3] */
        __m128 s44 = _mm_unpacklo_ps(s47, s47); /* [s4,s4,s5,s5] */
        __m128 s66 = _mm_unpackhi_ps(s47, s47); /* [s6,s6,s7,s7] */
        __m512 sv = _mm512_castps128_ps512(s00);
        sv = _mm512_insertf32x4(sv, s22, 1);
        sv = _mm512_insertf32x4(sv, s44, 2);
        sv = _mm512_insertf32x4(sv, s66, 3);
        __m512 sw = _mm512_shuffle_ps(v, v, 0xB1);
        /* addsub replacement: a + xor(b, sign_mask) where sign_mask negates odd lanes */
        __m512 a = _mm512_mul_ps(v, cv);
        __m512 b = _mm512_mul_ps(sw, sv);
        __m512i mask = _mm512_set_epi32(0,0x80000000,0,0x80000000,0,0x80000000,0,0x80000000,
                                        0,0x80000000,0,0x80000000,0,0x80000000,0,0x80000000);
        _mm512_storeu_ps(h + i * 2,
            _mm512_add_ps(a, _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(b), mask))));
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
          const float *cos_pos, const float *sin_pos, int rope_type, int half) {

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
#elif defined(PICOLM_AVX512)
            rope_avx512(qh, half, cos_pos, sin_pos);
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
#elif defined(PICOLM_AVX512)
            rope_avx512(kh, half, cos_pos, sin_pos);
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

void gelu(float *x, int size) {
    /* Always use F32 GELU for numerical accuracy.
     * The F16 table lookup introduces quantization error that compounds
     * over 48 layers, especially problematic for GPT-2 on GPU where the
     * GPU GELU uses F32 tanh approximation. Matching GELU ensures GPU
     * and CPU produce consistent intermediate values. */
    picolm_gelu_f32(x, size);
}
void elemwise_mul(float *out, const float *a, const float *b, int size) {
#ifdef PICOLM_AVX512
    int i = 0;
    for (; i + 15 < size; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
    for (; i < size; i++) out[i] = a[i] * b[i];
#elif defined(PICOLM_NEON)
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
#ifdef PICOLM_AVX512
    int i = 0;
    for (; i + 15 < size; i += 16) {
        _mm512_storeu_ps(a + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
    for (; i < size; i++) a[i] += b[i];
#elif defined(PICOLM_NEON)
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
/* tensor_parallel_for: dispatches [0, count) across the thread pool.
 *
 * IMPORTANT: pool_wake() broadcasts unconditionally to every worker thread
 * that was ever spawned (pool_nworkers, fixed at startup by
 * tensor_threadpool_init()), not just the first `nt` of them. If this
 * function only filled generic_tasks[0..nt-1] and nt < pool_nworkers+1
 * (which happens any time count < n_threads -- e.g. attention with
 * n_heads=24 on a 32-thread build), the remaining worker threads still
 * wake up and execute whatever *stale* generic_tasks[] entry is sitting
 * in their slot from a previous, unrelated call -- often a dangling
 * pointer to a stack frame that has already returned. Worse, pool_wait()
 * only waits for `nt-1` completions total, counted across *all* workers
 * that happen to finish first; if a stale-slot worker finishes its
 * garbage task quickly, it can satisfy that count before a real worker
 * on a valid slot has finished writing its output -- a silent data race,
 * not just wasted CPU. This is the same hazard the matmul dispatchers in
 * this file have (they also only fill [0, nt)); it matters more here
 * because n_heads is usually smaller than n_threads on multi-core boxes,
 * so this path hits the stale case on every single token, whereas most
 * matmuls have output dims larger than the thread count and rarely do.
 *
 * Fix: always refresh every slot in [0, n_threads), giving unused slots
 * an empty (start == end) range so they safely no-op instead of running
 * leftover data. */
void tensor_parallel_for(int count, void (*fn)(int idx, void *ctx), void *ctx) {
    if (n_threads <= 1 || count < 2) {
        for (int i = 0; i < count; i++) fn(i, ctx);
        return;
    }
    int nt = pool_total_threads(n_threads);
    int want = n_threads < nt ? n_threads : nt;
    int active = want > count ? count : want;
    int chunk = (count + active - 1) / active;
    for (int t = 0; t < active; t++) {
        generic_tasks[t].fn = fn;
        generic_tasks[t].ctx = ctx;
        generic_tasks[t].start = t * chunk;
        generic_tasks[t].end = (t + 1) * chunk;
        if (generic_tasks[t].end > count) generic_tasks[t].end = count;
    }
    for (int t = active; t < nt; t++) {
        generic_tasks[t].start = 0;
        generic_tasks[t].end = 0;
    }
    pool_mode = 1;
    pool_wake(nt);
    /* main thread does task 0 */
    generic_worker_f(&generic_tasks[0]);
    pool_wait(nt);
    pool_mode = 0;
}
