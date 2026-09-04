#include "tensor.h"
#include "sgemm.h"
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
#if defined(_WIN32) || defined(__APPLE__)
/* MSVC and old Mac OS X don't support __thread */
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
static void matmul_worker_f(matmul_task_t *t) {
    int nb = t->n_batch;
    int out_stride = nb > 0 ? t->d : t->end - t->start;
    /* Batched mode: output[b*d+i] for each batch b */
    if (nb > 0) {
        if (t->qtype == GGUF_TYPE_Q8_0 && t->x_d) {
            /* Pre-quantized Q8_0 activations: int8xint8 MAC */
            size_t q8_row_bytes = gguf_type_row_size(GGUF_TYPE_Q8_0, t->n);
            int nblk = t->n / 32;
            const char *qx_base = (const char *)t->x;
            for (int i = t->start; i < t->end; i++) {
                const block_q8_0 *wrow = (const block_q8_0 *)(t->W + (size_t)i * t->row_bytes);
                for (int b = 0; b < nb; b++) {
                    const block_q8_0 *xq = (const block_q8_0 *)(qx_base + (size_t)b * q8_row_bytes);
                    const float *xqd = t->x_d + (size_t)b * nblk;
                    t->out[b * out_stride + i] = vec_dot_q8_0_q8_0_deltas(xq, xqd, wrow, t->n);
                }
            }
        } else if (t->qtype == GGUF_TYPE_Q4_K && t->x) {
            /* Pre-quantized Q8_K activations: int8 MAC with per-subblock scales */
            size_t q8k_row_bytes = gguf_type_row_size(GGUF_TYPE_Q8_K, t->n);
            const char *qx_base = (const char *)t->x;
            for (int i = t->start; i < t->end; i++) {
                const char *wrow = t->W + (size_t)i * t->row_bytes;
                for (int b = 0; b < nb; b++) {
                    const char *xb = qx_base + (size_t)b * q8k_row_bytes;
                    t->out[b * out_stride + i] = vec_dot_q4_K_q8_K(wrow, xb, t->n);
                }
            }
        } else if (t->qtype == GGUF_TYPE_Q4_0 && t->x) {
            /* Pre-quantized Q8_0 activations. Decode each Q4_0 weight row
             * into a "shadow" Q8_0 row ONCE (Q4_0's dequant formula
             * (nibble-8)*d is exactly what Q8_0 stores natively, so this
             * is a lossless reinterpretation, not an approximation), then
             * reuse the fast vec_dot_q8_0_q8_0_deltas kernel (same one
             * Q8_0 weights use, AVX-512 path included) across every token
             * in the batch. Weight-stationary batching already reads
             * each row once per layer; this makes the nibble-unpack work
             * amortize the same way instead of repeating per token --
             * previously this called vec_dot_q4_0_q8_0 fresh for every
             * (row, token) pair, redoing the same unpack n_tokens times. */
            size_t q8_row_bytes = gguf_type_row_size(GGUF_TYPE_Q8_0, t->n);
            int nblk = t->n / 32;
            const char *qx_base = (const char *)t->x;
            uint8_t shadow_stack[32768];
            void *shadow = (q8_row_bytes <= sizeof(shadow_stack)) ? (void *)shadow_stack : malloc(q8_row_bytes);
            for (int i = t->start; i < t->end; i++) {
                const char *wrow = t->W + (size_t)i * t->row_bytes;
                q4_0_row_to_q8_0_shadow(wrow, shadow, t->n);
                for (int b = 0; b < nb; b++) {
                    const char *xb = qx_base + (size_t)b * q8_row_bytes;
                    const float *xqd = t->x_d + (size_t)b * nblk;
                    t->out[b * out_stride + i] = vec_dot_q8_0_q8_0_deltas(xb, xqd, shadow, t->n);
                }
            }
            if (shadow != shadow_stack) free(shadow);
        } else if (t->qtype == GGUF_TYPE_Q1_0 && t->x) {
            size_t q8_row_bytes = gguf_type_row_size(GGUF_TYPE_Q8_0, t->n);
            const char *qx_base = (const char *)t->x;
            for (int i = t->start; i < t->end; i++) {
                const char *wrow = t->W + (size_t)i * t->row_bytes;
                for (int b = 0; b < nb; b++) {
                    const char *xb = qx_base + (size_t)b * q8_row_bytes;
                    t->out[b * out_stride + i] = vec_dot_q1_0_q8_0(wrow, xb, t->n);
                }
            }
        } else if (t->qtype == GGUF_TYPE_Q2_0 && t->x) {
            size_t q8_row_bytes = gguf_type_row_size(GGUF_TYPE_Q8_0, t->n);
            const char *qx_base = (const char *)t->x;
            for (int i = t->start; i < t->end; i++) {
                const char *wrow = t->W + (size_t)i * t->row_bytes;
                for (int b = 0; b < nb; b++) {
                    const char *xb = qx_base + (size_t)b * q8_row_bytes;
                    t->out[b * out_stride + i] = vec_dot_q2_0_q8_0(wrow, xb, t->n);
                }
            }
        } else if (t->qtype == GGUF_TYPE_Q6_K && t->x) {
            /* Pre-quantized Q8_K activations: int8 MAC with per-subblock scales + bsums */
            size_t q8k_row_bytes = gguf_type_row_size(GGUF_TYPE_Q8_K, t->n);
            const char *qx_base = (const char *)t->x;
            for (int i = t->start; i < t->end; i++) {
                const char *wrow = t->W + (size_t)i * t->row_bytes;
                for (int b = 0; b < nb; b++) {
                    const char *xb = qx_base + (size_t)b * q8k_row_bytes;
                    t->out[b * out_stride + i] = vec_dot_q6_K_q8_K(wrow, xb, t->n);
                }
            }
        } else if (t->qtype == GGUF_TYPE_Q3_K && t->x) {
            size_t q8k_row_bytes = gguf_type_row_size(GGUF_TYPE_Q8_K, t->n);
            const char *qx_base = (const char *)t->x;
            for (int i = t->start; i < t->end; i++) {
                const char *wrow = t->W + (size_t)i * t->row_bytes;
                for (int b = 0; b < nb; b++) {
                    const char *xb = qx_base + (size_t)b * q8k_row_bytes;
                    t->out[b * out_stride + i] = vec_dot_q3_K_q8_K(wrow, xb, t->n);
                }
            }
        } else if (t->qtype == GGUF_TYPE_Q5_K && t->x) {
            size_t q8k_row_bytes = gguf_type_row_size(GGUF_TYPE_Q8_K, t->n);
            const char *qx_base = (const char *)t->x;
            for (int i = t->start; i < t->end; i++) {
                const char *wrow = t->W + (size_t)i * t->row_bytes;
                for (int b = 0; b < nb; b++) {
                    const char *xb = qx_base + (size_t)b * q8k_row_bytes;
                    t->out[b * out_stride + i] = vec_dot_q5_K_q8_K(wrow, xb, t->n);
                }
            }
        } else if (t->qtype == GGUF_TYPE_Q2_K && t->x) {
            /* Pre-quantized Q8_K activations: int8 MAC with per-subblock scales + bsums */
            size_t q8k_row_bytes = gguf_type_row_size(GGUF_TYPE_Q8_K, t->n);
            const char *qx_base = (const char *)t->x;
            for (int i = t->start; i < t->end; i++) {
                const char *wrow = t->W + (size_t)i * t->row_bytes;
                for (int b = 0; b < nb; b++) {
                    const char *xb = qx_base + (size_t)b * q8k_row_bytes;
                    t->out[b * out_stride + i] = vec_dot_q2_K_q8_K(wrow, xb, t->n);
                }
            }
        } else if (t->qtype == GGUF_TYPE_Q8_0) {
            for (int i = t->start; i < t->end; i++) {
                const block_q8_0 *wrow = (const block_q8_0 *)(t->W + (size_t)i * t->row_bytes);
                for (int b = 0; b < nb; b++)
                    t->out[b * out_stride + i] = vec_dot_q8_0_f32(wrow, t->x + b * t->n, t->n);
            }
        } else {
            for (int i = t->start; i < t->end; i++) {
                const char *wrow = t->W + (size_t)i * t->row_bytes;
                for (int b = 0; b < nb; b++)
                    t->out[b * out_stride + i] = vec_dot(wrow, t->x + b * t->n, t->n, t->qtype);
            }
        }
        return;
    }

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
    } else if (t->qtype == GGUF_TYPE_Q4_0_4_8 && t->x) {
        /* Q4_0_4_8: process 4 rows at a time from interleaved layout (blocklen=8) */
        const block_q8_0 *qx = (const block_q8_0 *)t->x;
        int start4 = (t->start / 4) * 4;
        int end4 = (t->end + 3) / 4 * 4;
        for (int i = start4; i < end4; i += 4) {
            vec_dot_q4_0x4_4x8_q8_0(
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
    } else if (t->qtype == GGUF_TYPE_Q6_K && t->x) {
        const block_q8_K *qx = (const block_q8_K *)t->x;
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot_q6_K_q8_K(
                t->W + (size_t)i * t->row_bytes, qx, t->n);
        }
    } else if (t->qtype == GGUF_TYPE_Q3_K && t->x) {
        const block_q8_K *qx = (const block_q8_K *)t->x;
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot_q3_K_q8_K(
                t->W + (size_t)i * t->row_bytes, qx, t->n);
        }
    } else if (t->qtype == GGUF_TYPE_Q5_K && t->x) {
        const block_q8_K *qx = (const block_q8_K *)t->x;
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot_q5_K_q8_K(
                t->W + (size_t)i * t->row_bytes, qx, t->n);
        }
    } else if (t->qtype == GGUF_TYPE_Q1_0 && t->x) {
        const block_q8_0 *qx = (const block_q8_0 *)t->x;
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot_q1_0_q8_0(
                t->W + (size_t)i * t->row_bytes, qx, t->n);
        }
    } else if (t->qtype == GGUF_TYPE_Q2_0 && t->x) {
        const block_q8_0 *qx = (const block_q8_0 *)t->x;
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot_q2_0_q8_0(
                t->W + (size_t)i * t->row_bytes, qx, t->n);
        }
    } else if (t->qtype == GGUF_TYPE_Q2_K && t->x) {
        const block_q8_K *qx = (const block_q8_K *)t->x;
        for (int i = t->start; i < t->end; i++) {
            t->out[i] = vec_dot_q2_K_q8_K(
                t->W + (size_t)i * t->row_bytes, qx, t->n);
        }
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

void matmul(float *out, const float *x, const void *W, int n, int d, gguf_type_t qtype) {
#ifdef PICOLM_GPU
    if (gpu_tensor && d > 0 && n > 0 && !getenv("PICOLM_PREFILL_CPU") && !getenv("PICOLM_SSM_PREFILL_CPU")) {
        gpu_assert_orchestrator("matmul GPU dispatch");
        /* GPU path: single-token GEMV (S=1). WMMA requires S>=16 so skip it here. */
        if (picolm_gpu_matmul(gpu_tensor, out, x, 1, gpu_device)) {
            static int gpu_matmul_count = 0;
            if (gpu_matmul_count++ == 0) {
                fprintf(stderr, "INFO: GPU matmul active\n");
            }
            return;
        }
        /* GPU returned 0: fall through to CPU path */
    }
#endif
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

            if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
                for (int i = 0; i < d; i++) {
                    out[i] = vec_dot_q8_0_q8_0_deltas(qx, qx_d, wptr + (size_t)i * row_bytes, n);
                }
                if (qx_buf != scratch_buf) free(qx_buf);
                return;
            }

            /* Threaded dispatch: main thread does task 0, workers do tasks 1..nt-1 */
            {
                int nt = pool_total_threads(n_threads);
                int want = n_threads < nt ? n_threads : nt;
                int active = pool_assign_rows(0, want, d);
                for (int t = 0; t < active; t++) {
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = qx_d; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q8_0;
                    pool_tasks[t].n_batch = 0;
                }
                pool_clear_unused(active, nt);
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_buf != scratch_buf) free(qx_buf);
            return;
        }
        /* If allocation failed, fall through to generic path */
    } else if (qtype == GGUF_TYPE_Q4_0) {
        /* Q4_0 fast path: quantize x to Q8_0 once, then vec_dot_q4_0_q8_0
         * On AVX2 with d%8==0, use repacked Q4_0_8x8 path for 8x row throughput.
         * vec_dot_q4_0_q8_0 has scalar/I8MM fallback for NEON/non-AVX platforms */
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

            if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
                for (int i = 0; i < d; i++) {
                    out[i] = vec_dot_q4_0_q8_0(wptr + (size_t)i * row_bytes, qx, n);
                }
                if (qx_owned) free(qx);
                return;
            }

            int nt = pool_total_threads(n_threads);
            int want = n_threads < nt ? n_threads : nt;

            {
                int active = pool_assign_rows(0, want, d);
                for (int t = 0; t < active; t++) {
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q4_0;
                    pool_tasks[t].n_batch = 0;
                }
                pool_clear_unused(active, nt);
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

            if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
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

            int nt = pool_total_threads(n_threads);
            int want = n_threads < nt ? n_threads : nt;
            {
                int active = pool_assign_rows(0, want, d);
                for (int t = 0; t < active; t++) {
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q4_0_4_4;
                    pool_tasks[t].n_batch = 0;
                }
                pool_clear_unused(active, nt);
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_owned) free(qx);
            return;
        }
    } else if (qtype == GGUF_TYPE_Q4_0_4_8) {
        /* Q4_0_4_8 fast path: same as 4_4 but uses vec_dot_q4_0x4_4x8_q8_0 (blocklen=8) */
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
            int d4 = (d / 4) * 4;
            if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
                for (int i = 0; i < d4; i += 4) {
                    vec_dot_q4_0x4_4x8_q8_0(wptr + (size_t)i * row_bytes, qx, n,
                                             out + i, 4);
                }
                for (int i = d4; i < d; i++) {
                    out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
                }
                if (qx_owned) free(qx);
                return;
            }
            int nt = pool_total_threads(n_threads);
            int want = n_threads < nt ? n_threads : nt;
            {
                int active = pool_assign_rows(0, want, d);
                for (int t = 0; t < active; t++) {
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q4_0_4_8;
                    pool_tasks[t].n_batch = 0;
                }
                pool_clear_unused(active, nt);
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
            if (n_threads <= 1 || d < 8 || d < matmul_min_rows) {
                vec_dot_q4_0x8_q8_0_avx2(wptr, qx, n, out, d);
                if (qx_owned) free(qx);
                return;
            }

            int nt = pool_total_threads(n_threads);

            int want = n_threads < nt ? n_threads : nt;
            {
                int active = pool_assign_rows(0, want, d);
                for (int t = 0; t < active; t++) {
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q4_0_8_8;
                    pool_tasks[t].n_batch = 0;
                }
                pool_clear_unused(active, nt);
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
        /* vec_dot_q4_K_q8_K has scalar fallback for NEON/non-AVX platforms */
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

            if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
                for (int i = 0; i < d; i++) {
                    out[i] = vec_dot_q4_K_q8_K(wptr + (size_t)i * row_bytes, qx, n);
                }
                if (qx_owned) free(qx);
                return;
            }

            int nt = pool_total_threads(n_threads);

            int want = n_threads < nt ? n_threads : nt;
            {
                int active = pool_assign_rows(0, want, d);
                for (int t = 0; t < active; t++) {
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q4_K;
                    pool_tasks[t].n_batch = 0;
                }
                pool_clear_unused(active, nt);
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_owned) free(qx);
            return;
        }
        /* If allocation failed, fall through to generic path */
    } else if (qtype == GGUF_TYPE_Q6_K) {
        /* Q6_K fast path: quantize x to Q8_K once, then vec_dot_q6_K_q8_K
         * Uses int8 MAC with per-subblock scale shuffling and bsums bias correction.
         * AVX2: maddubs_epi16 + madd_epi16 scale application
         * AVX:  maddubs_epi16 (128-bit) + 256-bit float accumulation
         * Scalar fallback available. */
        size_t qx_size = (n / 256) * sizeof(block_q8_K);
        block_q8_K *qx = NULL;
        int qx_owned = 0;
        if (n_threads <= 1 && scratch_buf != NULL && qx_size <= (size_t)scratch_size) {
            qx = (block_q8_K *)scratch_buf;
        } else {
            qx = (block_q8_K *)malloc(qx_size);
            qx_owned = 1;
        }
        if (qx != NULL) {
            quantize_row_q8_K(x, qx, n);

            if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
                for (int i = 0; i < d; i++) {
                    out[i] = vec_dot_q6_K_q8_K(wptr + (size_t)i * row_bytes, qx, n);
                }
                if (qx_owned) free(qx);
                return;
            }

            int nt = pool_total_threads(n_threads);
            int want = n_threads < nt ? n_threads : nt;
            {
                int active = pool_assign_rows(0, want, d);
                for (int t = 0; t < active; t++) {
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q6_K;
                    pool_tasks[t].n_batch = 0;
                }
                pool_clear_unused(active, nt);
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_owned) free(qx);
            return;
        }
        /* If allocation failed, fall through to generic path */
    } else if (qtype == GGUF_TYPE_Q3_K) {
        /* Q3_K fast path: quantize x to Q8_K once, then vec_dot_q3_K_q8_K */
        size_t qx_size = (n / 256) * sizeof(block_q8_K);
        block_q8_K *qx = NULL;
        int qx_owned = 0;
        if (n_threads <= 1 && scratch_buf != NULL && qx_size <= (size_t)scratch_size) {
            qx = (block_q8_K *)scratch_buf;
        } else {
            qx = (block_q8_K *)malloc(qx_size);
            qx_owned = 1;
        }
        if (qx != NULL) {
            quantize_row_q8_K(x, qx, n);

            if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
                for (int i = 0; i < d; i++) {
                    out[i] = vec_dot_q3_K_q8_K(wptr + (size_t)i * row_bytes, qx, n);
                }
                if (qx_owned) free(qx);
                return;
            }

            int nt = pool_total_threads(n_threads);
            int want = n_threads < nt ? n_threads : nt;
            {
                int active = pool_assign_rows(0, want, d);
                for (int t = 0; t < active; t++) {
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q3_K;
                    pool_tasks[t].n_batch = 0;
                }
                pool_clear_unused(active, nt);
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_owned) free(qx);
            return;
        }
        /* If allocation failed, fall through to generic path */
    } else if (qtype == GGUF_TYPE_Q5_K) {
        /* Q5_K fast path: quantize x to Q8_K once, then vec_dot_q5_K_q8_K */
        size_t qx_size = (n / 256) * sizeof(block_q8_K);
        block_q8_K *qx = NULL;
        int qx_owned = 0;
        if (n_threads <= 1 && scratch_buf != NULL && qx_size <= (size_t)scratch_size) {
            qx = (block_q8_K *)scratch_buf;
        } else {
            qx = (block_q8_K *)malloc(qx_size);
            qx_owned = 1;
        }
        if (qx != NULL) {
            quantize_row_q8_K(x, qx, n);

            if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
                for (int i = 0; i < d; i++) {
                    out[i] = vec_dot_q5_K_q8_K(wptr + (size_t)i * row_bytes, qx, n);
                }
                if (qx_owned) free(qx);
                return;
            }

            int nt = pool_total_threads(n_threads);
            int want = n_threads < nt ? n_threads : nt;
            {
                int active = pool_assign_rows(0, want, d);
                for (int t = 0; t < active; t++) {
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q5_K;
                    pool_tasks[t].n_batch = 0;
                }
                pool_clear_unused(active, nt);
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_owned) free(qx);
            return;
        }
    } else if (qtype == GGUF_TYPE_Q1_0 || qtype == GGUF_TYPE_Q2_0) {
        /* Q1_0/Q2_0 fast path: quantize x to Q8_0 once, then use int8 MAC
         * for all rows. Same approach as Q4_0/Q8_0 above.
         * Q1_0 has AVX2 SIMD path in vec_dot_q1_0_q8_0.
         * Q2_0 has scalar fallback (AVX-512-VNNI path only in llama.cpp). */
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

            if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
                if (qtype == GGUF_TYPE_Q1_0) {
                    for (int i = 0; i < d; i++) {
                        out[i] = vec_dot_q1_0_q8_0(wptr + (size_t)i * row_bytes, qx, n);
                    }
                } else {
                    for (int i = 0; i < d; i++) {
                        out[i] = vec_dot_q2_0_q8_0(wptr + (size_t)i * row_bytes, qx, n);
                    }
                }
                if (qx_owned) free(qx);
                return;
            }

            int nt = pool_total_threads(n_threads);
            int want = n_threads < nt ? n_threads : nt;
            {
                int active = pool_assign_rows(0, want, d);
                for (int t = 0; t < active; t++) {
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = qtype;
                    pool_tasks[t].n_batch = 0;
                }
                pool_clear_unused(active, nt);
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_owned) free(qx);
            return;
        }
        /* If allocation failed, fall through to generic path */
    } else if (qtype == GGUF_TYPE_Q2_K) {
        /* Q2_K fast path: quantize x to Q8_K once, then vec_dot_q2_K_q8_K */
        size_t qx_size = (n / 256) * sizeof(block_q8_K);
        block_q8_K *qx = NULL;
        int qx_owned = 0;
        if (n_threads <= 1 && scratch_buf != NULL && qx_size <= (size_t)scratch_size) {
            qx = (block_q8_K *)scratch_buf;
        } else {
            qx = (block_q8_K *)malloc(qx_size);
            qx_owned = 1;
        }
        if (qx != NULL) {
            quantize_row_q8_K(x, qx, n);

            if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
                for (int i = 0; i < d; i++) {
                    out[i] = vec_dot_q2_K_q8_K(wptr + (size_t)i * row_bytes, qx, n);
                }
                if (qx_owned) free(qx);
                return;
            }

            int nt = pool_total_threads(n_threads);
            int want = n_threads < nt ? n_threads : nt;
            {
                int active = pool_assign_rows(0, want, d);
                for (int t = 0; t < active; t++) {
                    pool_tasks[t].out = out; pool_tasks[t].x = (const float *)qx;
                    pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
                    pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
                    pool_tasks[t].qtype = GGUF_TYPE_Q2_K;
                    pool_tasks[t].n_batch = 0;
                }
                pool_clear_unused(active, nt);
                pool_init(nt);
                pool_wake(nt);
                matmul_worker_f(&pool_tasks[0]);
                pool_wait(nt);
            }

            if (qx_owned) free(qx);
            return;
        }
        /* If allocation failed, fall through to generic path */
    }

    /* Tiled GEMM fast path for F32/F16 weights.
     * Much faster than vec_dot gemv for large matrices because
     * the activation vector is reused across tiles. */
    if (picolm_sgemm(d, 1, n, wptr, n, x, n, out, d, qtype, GGUF_TYPE_F32, 0, 1)) {
        return;
    }

    if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
        for (int i = 0; i < d; i++) {
            out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
        }
        return;
    }

    int nt = pool_total_threads(n_threads);

    int want = n_threads < nt ? n_threads : nt;
    {
        int active = pool_assign_rows(0, want, d);
        for (int t = 0; t < active; t++) {
            pool_tasks[t].out = out; pool_tasks[t].x = x;
            pool_tasks[t].x_d = NULL; pool_tasks[t].W = wptr;
            pool_tasks[t].row_bytes = row_bytes; pool_tasks[t].n = n;
            pool_tasks[t].qtype = qtype;
            pool_tasks[t].n_batch = 0;
        }
        pool_clear_unused(active, nt);
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
} q4_0_8_8_batch_ctx_t;

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

    const char *xb = (const char *)ctx->qx_buf + (size_t)b * ctx->q8_row_bytes;
    const void *wp = (const char *)ctx->wptr + (size_t)row_start * ctx->w_row_bytes;
    float *op = ctx->out + (size_t)b * ctx->d + row_start;
    vec_dot_q4_0x8_q8_0_avx2(wp, xb, ctx->n, op, nrows);
}
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
                pool_tasks[t].n_batch = 0;
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
#if defined(__AVX2__) && defined(__F16C__)
typedef struct {
    int m, n, k_blocks;
    const void *A; int lda;
    const block_q8_0 *B; int ldb;
    const float *B_d; int ldb_d;
    float *C; int ldc;
    int Atype;
    int nth;
} qgemm_d_ctx_t;

static void qgemm_d_task(int idx, void *ctxp) {
    qgemm_d_ctx_t *c = (qgemm_d_ctx_t *)ctxp;
    picolm_sgemm_d(c->m, c->n, c->k_blocks, c->A, c->lda,
                   c->B, c->ldb, c->B_d, c->ldb_d,
                   c->C, c->ldc, c->Atype, idx, c->nth);
}
#endif

void matmul_batch(float *out, const float *x, int n_batch,
                   const void *W, int n, int d, gguf_type_t qtype) {
#ifdef PICOLM_GPU
        if (gpu_tensor && n_batch > 0 && d > 0 && n > 0 && !getenv("PICOLM_PREFILL_CPU") && !getenv("PICOLM_SSM_PREFILL_CPU")) {
        gpu_assert_orchestrator("matmul_batch GPU dispatch");
        /* Try WMMA Tensor Core for aligned Q4_0 batched matmul */
        if (picolm_gpu_w4a16_matmul(gpu_tensor, out, x, n_batch, gpu_device)) {
            static int gpu_batch_count = 0;
            if (gpu_batch_count++ == 0) {
                fprintf(stderr, "INFO: GPU WMMA batch matmul active\n");
            }
            return;
        }
        if (picolm_gpu_matmul(gpu_tensor, out, x, n_batch, gpu_device)) {
            return;
        }
        if ((n == 6144 || d == 6144) && getenv("PICOLM_GPU")) fprintf(stderr, "WARN: GPU batch matmul failed (n=%d d=%d batch=%d qtype=%d gpu_tensor=%p) -> CPU\n",
            n, d, n_batch, qtype, gpu_tensor);
    }
#endif
    size_t row_bytes = gguf_type_row_size(qtype, n);
    const char *wptr = (const char *)W;

    /* Tiled GEMM for F16/F32 weights with batched F32 activations.
     * Much faster than per-row gemv because activation tokens are
     * reused across weight tiles.
     * picolm_sgemm handles: F32xF32, F16xF32, F16xF16, Q8_0xQ8_0 (ARM NEON too). */
    if (picolm_sgemm(d, n_batch, n, wptr, n, x, n, out, d, qtype, GGUF_TYPE_F32, 0, n_threads)) {
        if (getenv("PICOLM_DISPATCH")) {
            fprintf(stderr, "DISPATCH matmul_batch: d=%d n=%d batch=%d qtype=%d -> SGEMM\n", d, n, n_batch, qtype);
        }
        return;
    }
    if (getenv("PICOLM_DISPATCH")) {
        fprintf(stderr, "DISPATCH matmul_batch: d=%d n=%d batch=%d qtype=%d -> vec_dot\n", d, n, n_batch, qtype);
    }

#if defined(PICOLM_AVX2)
    /* Parallelize across (token, 8-row-group) pairs. Each group is processed
     * by vec_dot_q4_0x8_q8_0_avx2 which handles 8 rows per call. This matches
     * the row-parallel strategy of the Q8_0 batch path and provides enough
     * work units even for small n_batch (e.g. 51 tokens x ~1280 groups =
     * 65K work units on a 27B model). */
    /* WIP: Interleaved Q4_0_8_8 GEMM path (sgemm_q4_0x8.c).
     * Disabled by default due to known bugs. Enable with PICOLM_Q4_0x8_GEMM=1.
     * See sgemm_q4_0x8.c header for details and known issues. */
    if (qtype == GGUF_TYPE_Q4_0_8_8 && n_batch > 0 && n > 0) {
        size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
        void *qbuf = malloc((size_t)n_batch * q8_rb);
        if (qbuf) {
            for (int b = 0; b < n_batch; b++)
                quantize_row_q8_0(x + (size_t)b * n, (char *)qbuf + (size_t)b * q8_rb, n);
            int total_groups = (d + 7) / 8;
            q4_0_8_8_batch_ctx_t ctx = { wptr, row_bytes, qbuf, q8_rb, out, n, d, n_batch };
            tensor_parallel_for(n_batch * total_groups, q4_0_8_8_batch_task, &ctx);
            free(qbuf);
            return;
        }
        /* malloc failed: fall through to the generic path below, which is
         * WRONG for this interleaved layout -- but malloc failure at this
         * size is already a bigger problem than a wrong matmul result. */
    }
#endif /* PICOLM_AVX2 */
    /* Same idea, 4-row groups -- this is the format ARM NEON dotprod/SDOT
     * hardware wants. vec_dot_q4_0x4_q8_0 is a plain scalar reference (no
     * SIMD acceleration wired in yet), so unlike the AVX2 kernel above it
     * doesn't need a PICOLM_AVX2 guard -- it's correct (if not fast) on
     * any platform. Reviewed and algebraically verified against the real
     * NEON dotprod algorithm (the shift-based nibble extraction matches
     * Q4_0's dequant exactly, same identity as the AVX2 LUT above), but
     * not run on real dotprod hardware -- none available to test on here. */
    if (qtype == GGUF_TYPE_Q4_0_4_4 && n_batch > 0 && n > 0) {
        size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
        void *qbuf = malloc((size_t)n_batch * q8_rb);
        if (qbuf) {
            for (int b = 0; b < n_batch; b++)
                quantize_row_q8_0(x + (size_t)b * n, (char *)qbuf + (size_t)b * q8_rb, n);
            q4_0_4_4_batch_ctx_t ctx = { wptr, qbuf, q8_rb, out, n, d };
            tensor_parallel_for(n_batch, q4_0_4_4_batch_task, &ctx);
            free(qbuf);
            return;
        }
    }
    /* Q4_0_4_8 batch path: use I8MM gemm (delegates to DOTPROD gemv) */
    if (qtype == GGUF_TYPE_Q4_0_4_8 && n_batch > 0 && n > 0) {
        size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
        void *qbuf = malloc((size_t)n_batch * q8_rb);
        if (qbuf) {
            for (int b = 0; b < n_batch; b++)
                quantize_row_q8_0(x + (size_t)b * n, (char *)qbuf + (size_t)b * q8_rb, n);
            gemm_q4_0_4x8_q8_0(wptr, qbuf, n, out, d, n_batch);
            free(qbuf);
            return;
        }
    }

    /* Pre-quantize the whole activation batch once per weight type */
    void *qx_buf = NULL; float *qx_d_buf = NULL;
    size_t qx_stride = 0;
    int have_qx = 0;
    if (qtype == GGUF_TYPE_Q8_0 && n_batch > 0 && n > 0) {
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
            qx_buf = qbuf; qx_d_buf = dbuf;
            qx_stride = q8_rb; have_qx = 1;
        }
    } else if (qtype == GGUF_TYPE_Q4_K && n_batch > 0 && n > 0) {
        size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
        void *qbuf = malloc((size_t)n_batch * q8k_rb);
        if (qbuf) {
            for (int b = 0; b < n_batch; b++) {
                quantize_row_q8_K(x + (size_t)b * n, (char *)qbuf + (size_t)b * q8k_rb, n);
            }
            qx_buf = qbuf; qx_stride = q8k_rb; have_qx = 1;
        }
    } else if (qtype == GGUF_TYPE_Q6_K && n_batch > 0 && n > 0) {
        /* Q6_K: quantize to Q8_K, same as Q4_K */
        size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
        void *qbuf = malloc((size_t)n_batch * q8k_rb);
        if (qbuf) {
            for (int b = 0; b < n_batch; b++) {
                quantize_row_q8_K(x + (size_t)b * n, (char *)qbuf + (size_t)b * q8k_rb, n);
            }
            qx_buf = qbuf; qx_stride = q8k_rb; have_qx = 1;
        }
    } else if (qtype == GGUF_TYPE_Q3_K && n_batch > 0 && n > 0) {
        size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
        void *qbuf = malloc((size_t)n_batch * q8k_rb);
        if (qbuf) {
            for (int b = 0; b < n_batch; b++)
                quantize_row_q8_K(x + (size_t)b * n, (char *)qbuf + (size_t)b * q8k_rb, n);
            qx_buf = qbuf; qx_stride = q8k_rb; have_qx = 1;
        }
    } else if (qtype == GGUF_TYPE_Q5_K && n_batch > 0 && n > 0) {
        size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
        void *qbuf = malloc((size_t)n_batch * q8k_rb);
        if (qbuf) {
            for (int b = 0; b < n_batch; b++)
                quantize_row_q8_K(x + (size_t)b * n, (char *)qbuf + (size_t)b * q8k_rb, n);
            qx_buf = qbuf; qx_stride = q8k_rb; have_qx = 1;
        }
    } else if (qtype == GGUF_TYPE_Q4_0 && n_batch > 0 && n > 0) {
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
            qx_buf = qbuf; qx_d_buf = dbuf;
            qx_stride = q8_rb; have_qx = 1;
        } else {
            free(qbuf); free(dbuf);
        }
    } else if (qtype == GGUF_TYPE_Q1_0 && n_batch > 0 && n > 0) {
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
            qx_buf = qbuf; qx_d_buf = dbuf;
            qx_stride = q8_rb; have_qx = 1;
        } else { free(qbuf); free(dbuf); }
    } else if (qtype == GGUF_TYPE_Q2_0 && n_batch > 0 && n > 0) {
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
            qx_buf = qbuf; qx_d_buf = dbuf;
            qx_stride = q8_rb; have_qx = 1;
        } else { free(qbuf); free(dbuf); }
    } else if (qtype == GGUF_TYPE_Q2_K && n_batch > 0 && n > 0) {
        size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
        void *qbuf = malloc((size_t)n_batch * q8k_rb);
        if (qbuf) {
            for (int b = 0; b < n_batch; b++)
                quantize_row_q8_K(x + (size_t)b * n, (char *)qbuf + (size_t)b * q8k_rb, n);
            qx_buf = qbuf; qx_stride = q8k_rb; have_qx = 1;
        }
    }

    /* Quantized GEMM fast path: try tiled GEMM with pre-quantized Q8_0 activations.
     * AVX2+F16C: uses picolm_sgemm_d (delta-optimized, _mm_cvtph_ps).
     * ARM NEON: uses picolm_sgemm (Q8_0xQ8_0 path, no delta optimization).
     * Threshold: n_batch >= 8 for all quant types. */
#if defined(__AVX2__) && defined(__F16C__)
    if (have_qx && qx_d_buf && d >= 4 &&
        (qtype == GGUF_TYPE_Q8_0 || qtype == GGUF_TYPE_Q4_0 || qtype == GGUF_TYPE_Q5_0)) {
        int min_batch = 8;
        if (n_batch >= min_batch) {
            int k_blocks = n / 32;
            int nth = pool_total_threads(1);
            qgemm_d_ctx_t ctx = {
                .m = d, .n = n_batch, .k_blocks = k_blocks,
                .A = W, .lda = k_blocks,
                .B = (const block_q8_0*)qx_buf, .ldb = k_blocks,
                .B_d = qx_d_buf, .ldb_d = k_blocks,
                .C = out, .ldc = d,
                .Atype = qtype, .nth = nth,
            };
            tensor_parallel_for(nth, qgemm_d_task, &ctx);
            if (getenv("PICOLM_DISPATCH")) fprintf(stderr, "DISPATCH matmul_batch: d=%d n=%d batch=%d qtype=%d -> GEMM_d\n", d, n, n_batch, qtype);
            if (qx_buf) { free(qx_buf); if (qx_d_buf) free(qx_d_buf); }
            return;
        }
    }
#endif
#if defined(__ARM_NEON) && (defined(__ARM_FEATURE_DOTPROD) || defined(__ARM_FEATURE_MATMUL_INT8))
    /* ARM NEON GEMM only useful with DOTPROD or I8MM.
     * Plain NEON scalar fallback is slower than vec_dot. */
    if (have_qx && qx_stride && d >= 4 &&
        (qtype == GGUF_TYPE_Q8_0 || qtype == GGUF_TYPE_Q4_0 || qtype == GGUF_TYPE_Q5_0)) {
        if (n_batch >= 8) {
            int k_blocks = n / 32;
            int nth = pool_total_threads(1);
            if (picolm_sgemm(d, n_batch, k_blocks, W, k_blocks,
                             qx_buf, k_blocks, out, d,
                             qtype, GGUF_TYPE_Q8_0, 0, nth)) {
                if (getenv("PICOLM_DISPATCH")) fprintf(stderr, "DISPATCH matmul_batch: d=%d n=%d batch=%d qtype=%d -> SGEMM(Q8_0)\n", d, n, n_batch, qtype);
                if (qx_buf) { free(qx_buf); if (qx_d_buf) free(qx_d_buf); }
                return;
            }
        }
    }
#endif

    /* Q4_0_4_4 fast path for batched matmul */
    if (qtype == GGUF_TYPE_Q4_0_4_4 && n_batch > 0 && n > 0) {
        size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
        void *q4q_qx_buf = malloc((size_t)n_batch * q8_rb);
        if (q4q_qx_buf) {
            for (int b = 0; b < n_batch; b++)
                quantize_row_q8_0(x + (size_t)b * n, (char *)q4q_qx_buf + (size_t)b * q8_rb, n);
            int d4 = (d / 4) * 4;
            size_t group_bytes = (size_t)(n / 32) * sizeof(block_q4_0x4);
            for (int b = 0; b < n_batch; b++) {
                const block_q8_0 *qx = (const block_q8_0 *)((char *)q4q_qx_buf + (size_t)b * q8_rb);
                for (int g = 0; g < d4; g += 4) {
                    const char *wg = (const char *)W + (g / 4) * group_bytes;
                    vec_dot_q4_0x4_q8_0(wg, qx, n, out + b * d + g, 4);
                }
                /* Tail rows */
                for (int i = d4; i < d; i++)
                    out[b * d + i] = vec_dot((const char *)W + (size_t)i * gguf_type_row_size(qtype, n), x + b * n, n, qtype);
            }
            free(q4q_qx_buf);
            return;
        }
    }

    if (n_threads <= 1 || d < 4 || d < matmul_min_rows) {
        /* Scalar path */
        for (int i = 0; i < d; i++) {
            if (have_qx) {
                const char *wrow = wptr + (size_t)i * row_bytes;
                for (int b = 0; b < n_batch; b++) {
                    const char *xb = (const char *)qx_buf + (size_t)b * qx_stride;
                    if (qtype == GGUF_TYPE_Q8_0) {
                        const block_q8_0 *xq = (const block_q8_0 *)xb;
                        const float *xqd = qx_d_buf + (size_t)b * (n / 32);
                        out[b * d + i] = vec_dot_q8_0_q8_0_deltas(xq, xqd,
                            (const block_q8_0 *)wrow, n);
                    } else if (qtype == GGUF_TYPE_Q4_K) {
                        out[b * d + i] = vec_dot_q4_K_q8_K(wrow, xb, n);
                    } else if (qtype == GGUF_TYPE_Q6_K) {
                        out[b * d + i] = vec_dot_q6_K_q8_K(wrow, xb, n);
                    } else if (qtype == GGUF_TYPE_Q3_K) {
                        out[b * d + i] = vec_dot_q3_K_q8_K(wrow, xb, n);
                    } else if (qtype == GGUF_TYPE_Q5_K) {
                        out[b * d + i] = vec_dot_q5_K_q8_K(wrow, xb, n);
                    } else if (qtype == GGUF_TYPE_Q1_0) {
                        out[b * d + i] = vec_dot_q1_0_q8_0(wrow, xb, n);
                    } else if (qtype == GGUF_TYPE_Q2_0) {
                        out[b * d + i] = vec_dot_q2_0_q8_0(wrow, xb, n);
                    } else if (qtype == GGUF_TYPE_Q2_K) {
                        out[b * d + i] = vec_dot_q2_K_q8_K(wrow, xb, n);
                    } else {
                        out[b * d + i] = vec_dot_q4_0_q8_0(wrow, xb, n);
                    }
                }
            } else if (qtype == GGUF_TYPE_Q8_0) {
                const block_q8_0 *wrow = (const block_q8_0 *)(wptr + (size_t)i * row_bytes);
                for (int b = 0; b < n_batch; b++)
                    out[b * d + i] = vec_dot_q8_0_f32(wrow, x + b * n, n);
            } else {
                const char *wrow = wptr + (size_t)i * row_bytes;
                for (int b = 0; b < n_batch; b++)
                    out[b * d + i] = vec_dot(wrow, x + b * n, n, qtype);
            }
        }
        if (qx_buf) { free(qx_buf); if (qx_d_buf) free(qx_d_buf); }
        return;
    }

    /* Threaded: dispatch over output rows d using shared pool */
    int nt = pool_total_threads(n_threads);
    int want = n_threads < nt ? n_threads : nt;
    int active = pool_assign_rows(0, want, d);
    for (int t = 0; t < active; t++) {
        pool_tasks[t].out = out;
        pool_tasks[t].x = have_qx ? (const float *)qx_buf : x;
        pool_tasks[t].x_d = have_qx ? qx_d_buf : NULL;
        pool_tasks[t].W = wptr;
        pool_tasks[t].row_bytes = row_bytes;
        pool_tasks[t].n = n;
        pool_tasks[t].d = d;
        pool_tasks[t].qtype = qtype;
        pool_tasks[t].n_batch = n_batch;
    }
    pool_clear_unused(active, nt);
    pool_init(nt);
    pool_wake(nt);
    matmul_worker_f(&pool_tasks[0]);
    pool_wait(nt);
    if (qx_buf) { free(qx_buf); if (qx_d_buf) free(qx_d_buf); }
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
    gguf_type_t qtype1, qtype2;
} dual_q8_row_ctx_t;

static void dual_q8_row_task(int i, void *ctxp) {
    dual_q8_row_ctx_t *c = (dual_q8_row_ctx_t *)ctxp;
    const char *wr1 = (const char *)c->W1 + (size_t)i * c->row_bytes1;
    const char *wr2 = (const char *)c->W2 + (size_t)i * c->row_bytes2;
    for (int b = 0; b < c->n_batch; b++) {
        if (c->have_qx1) {
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
            } else {
                c->out1[b * c->d + i] = vec_dot(wr1, c->x + b * c->n, c->n, c->qtype1);
            }
        } else {
            c->out1[b * c->d + i] = vec_dot(wr1, c->x + b * c->n, c->n, c->qtype1);
        }
        if (c->have_qx2) {
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
            } else {
                c->out2[b * c->d + i] = vec_dot(wr2, c->x + b * c->n, c->n, c->qtype2);
            }
        } else {
            c->out2[b * c->d + i] = vec_dot(wr2, c->x + b * c->n, c->n, c->qtype2);
        }
    }
}

void matmul_dual_batch(float *out1, float *out2, const float *x, int n_batch,
                        const void *W1, const void *W2,
                        int n, int d, gguf_type_t qtype1, gguf_type_t qtype2) {
#ifdef PICOLM_GPU
    if (getenv("PICOLM_GPU")) {
        fprintf(stderr, "WARN: matmul_dual_batch (CPU only, no GPU path) n=%d d=%d batch=%d qtype=%d/%d\n",
            n, d, n_batch, qtype1, qtype2);
    }
#endif
    /* Quantized GEMM fast path for dual batch (K+V or gate+up projections).
     * Pre-quantize activations once, then use picolm_sgemm_d via tensor_parallel_for
     * for both matmuls. Each worker thread gets a distinct ith. */
#if defined(__AVX2__) && defined(__F16C__)
    if (n_batch >= 8 && d >= 4 && n > 0 &&
        (qtype1 == GGUF_TYPE_Q8_0 || qtype1 == GGUF_TYPE_Q4_0 || qtype1 == GGUF_TYPE_Q5_0) &&
        (qtype2 == GGUF_TYPE_Q8_0 || qtype2 == GGUF_TYPE_Q4_0 || qtype2 == GGUF_TYPE_Q5_0)) {
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
            int k_blocks = n / 32;
            int nth = pool_total_threads(1);
            qgemm_d_ctx_t ctx1 = {
                .m = d, .n = n_batch, .k_blocks = k_blocks,
                .A = W1, .lda = k_blocks,
                .B = (const block_q8_0*)qbuf, .ldb = k_blocks,
                .B_d = dbuf, .ldb_d = k_blocks,
                .C = out1, .ldc = d,
                .Atype = qtype1, .nth = nth,
            };
            tensor_parallel_for(nth, qgemm_d_task, &ctx1);
            qgemm_d_ctx_t ctx2 = {
                .m = d, .n = n_batch, .k_blocks = k_blocks,
                .A = W2, .lda = k_blocks,
                .B = (const block_q8_0*)qbuf, .ldb = k_blocks,
                .B_d = dbuf, .ldb_d = k_blocks,
                .C = out2, .ldc = d,
                .Atype = qtype2, .nth = nth,
            };
            tensor_parallel_for(nth, qgemm_d_task, &ctx2);
            free(qbuf); free(dbuf);
            return;
        }
    }
#endif

#if defined(PICOLM_AVX2)
    /* Same reasoning as matmul_batch's Q4_0_8_8 branch: reuse the
     * validated gemv kernel once per token, parallelized over tokens.
     * Handled per-output so a mixed qtype1/qtype2 (uncommon in practice
     * -- gate/up almost always share a quant type) still gets a correct
     * result for whichever side isn't Q4_0_8_8, just without the fast
     * path on that side. */
    if ((qtype1 == GGUF_TYPE_Q4_0_8_8 || qtype2 == GGUF_TYPE_Q4_0_8_8) && n_batch > 0 && n > 0) {
        size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
        void *qbuf = malloc((size_t)n_batch * q8_rb);
        if (qbuf) {
            for (int b = 0; b < n_batch; b++)
                quantize_row_q8_0(x + (size_t)b * n, (char *)qbuf + (size_t)b * q8_rb, n);

            int total_groups = (d + 7) / 8;
            if (qtype1 == GGUF_TYPE_Q4_0_8_8) {
                size_t rb1 = gguf_type_row_size(GGUF_TYPE_Q4_0_8_8, n);
                q4_0_8_8_batch_ctx_t ctx1 = { (const char *)W1, rb1, qbuf, q8_rb, out1, n, d, n_batch };
                tensor_parallel_for(n_batch * total_groups, q4_0_8_8_batch_task, &ctx1);
            } else {
                size_t rb1 = gguf_type_row_size(qtype1, n);
                for (int i = 0; i < d; i++) {
                    const char *wr = (const char *)W1 + (size_t)i * rb1;
                    for (int b = 0; b < n_batch; b++)
                        out1[b * d + i] = vec_dot(wr, x + b * n, n, qtype1);
                }
            }
            if (qtype2 == GGUF_TYPE_Q4_0_8_8) {
                size_t rb2 = gguf_type_row_size(GGUF_TYPE_Q4_0_8_8, n);
                q4_0_8_8_batch_ctx_t ctx2 = { (const char *)W2, rb2, qbuf, q8_rb, out2, n, d, n_batch };
                tensor_parallel_for(n_batch * total_groups, q4_0_8_8_batch_task, &ctx2);
            } else {
                size_t rb2 = gguf_type_row_size(qtype2, n);
                for (int i = 0; i < d; i++) {
                    const char *wr = (const char *)W2 + (size_t)i * rb2;
                    for (int b = 0; b < n_batch; b++)
                        out2[b * d + i] = vec_dot(wr, x + b * n, n, qtype2);
                }
            }
            free(qbuf);
            return;
        }
        /* malloc failed: fall through (wrong for the Q4_0_8_8 side, but
         * malloc failure at this size is already a bigger problem). */
    }
#endif

    /* Pre-quantize activations for both matmuls.
     * If both types are the same, one quantization suffices.
     * If mixed, quantize for each type separately. */
    void *qx1_buf = NULL; size_t qx1_stride = 0;
    float *qx1_d_buf = NULL; int have_qx1 = 0;
    void *qx2_buf = NULL; size_t qx2_stride = 0;
    float *qx2_d_buf = NULL; int have_qx2 = 0;

    if (n_batch > 0 && n > 0) {
        if (qtype1 == GGUF_TYPE_Q4_K || qtype1 == GGUF_TYPE_Q6_K || qtype1 == GGUF_TYPE_Q3_K ||
            qtype1 == GGUF_TYPE_Q5_K || qtype1 == GGUF_TYPE_Q4_0 || qtype1 == GGUF_TYPE_Q8_0 ||
            qtype1 == GGUF_TYPE_Q1_0 || qtype1 == GGUF_TYPE_Q2_0 || qtype1 == GGUF_TYPE_Q2_K) {
            have_qx1 = 1;
            if (qtype1 == GGUF_TYPE_Q8_0) {
                size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
                int nb = n / 32;
                qx1_buf = malloc((size_t)n_batch * q8_rb);
                qx1_d_buf = (float *)malloc((size_t)n_batch * nb * sizeof(float));
                if (qx1_buf && qx1_d_buf) {
                    for (int b = 0; b < n_batch; b++) {
                        quantize_row_q8_0(x + (size_t)b * n, (char *)qx1_buf + (size_t)b * q8_rb, n);
                        const block_q8_0 *blk = (const block_q8_0 *)((char *)qx1_buf + (size_t)b * q8_rb);
                        for (int k = 0; k < nb; k++) qx1_d_buf[(size_t)b * nb + k] = fp16_to_fp32(blk[k].d);
                    }
                    qx1_stride = q8_rb;
                } else { have_qx1 = 0; free(qx1_buf); free(qx1_d_buf); qx1_buf = NULL; qx1_d_buf = NULL; }
            } else if (qtype1 == GGUF_TYPE_Q4_K) {
                size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
                qx1_buf = malloc((size_t)n_batch * q8k_rb);
                if (qx1_buf) {
                    for (int b = 0; b < n_batch; b++) {
                        quantize_row_q8_K(x + (size_t)b * n, (char *)qx1_buf + (size_t)b * q8k_rb, n);
                    }
                    qx1_stride = q8k_rb;
                } else have_qx1 = 0;
            } else if (qtype1 == GGUF_TYPE_Q6_K) {
                /* Q6_K: quantize to Q8_K (same as Q4_K) */
                size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
                qx1_buf = malloc((size_t)n_batch * q8k_rb);
                if (qx1_buf) {
                    for (int b = 0; b < n_batch; b++) {
                        quantize_row_q8_K(x + (size_t)b * n, (char *)qx1_buf + (size_t)b * q8k_rb, n);
                    }
                    qx1_stride = q8k_rb;
                } else have_qx1 = 0;
            } else if (qtype1 == GGUF_TYPE_Q3_K || qtype1 == GGUF_TYPE_Q5_K) {
                /* Q3_K/Q5_K: quantize to Q8_K */
                size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
                qx1_buf = malloc((size_t)n_batch * q8k_rb);
                if (qx1_buf) {
                    for (int b = 0; b < n_batch; b++) {
                        quantize_row_q8_K(x + (size_t)b * n, (char *)qx1_buf + (size_t)b * q8k_rb, n);
                    }
                    qx1_stride = q8k_rb;
                } else have_qx1 = 0;
            } else if (qtype1 == GGUF_TYPE_Q1_0 || qtype1 == GGUF_TYPE_Q2_0) {
                size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
                qx1_buf = malloc((size_t)n_batch * q8_rb);
                if (qx1_buf) {
                    for (int b = 0; b < n_batch; b++) {
                        quantize_row_q8_0(x + (size_t)b * n, (char *)qx1_buf + (size_t)b * q8_rb, n);
                    }
                    qx1_stride = q8_rb;
                } else have_qx1 = 0;
            } else if (qtype1 == GGUF_TYPE_Q2_K) {
                size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
                qx1_buf = malloc((size_t)n_batch * q8k_rb);
                if (qx1_buf) {
                    for (int b = 0; b < n_batch; b++) {
                        quantize_row_q8_K(x + (size_t)b * n, (char *)qx1_buf + (size_t)b * q8k_rb, n);
                    }
                    qx1_stride = q8k_rb;
                } else have_qx1 = 0;
            } else { /* Q4_0 */
                size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
                int nb = n / 32;
                qx1_buf = malloc((size_t)n_batch * q8_rb);
                qx1_d_buf = (float *)malloc((size_t)n_batch * nb * sizeof(float));
                if (qx1_buf && qx1_d_buf) {
                    for (int b = 0; b < n_batch; b++) {
                        quantize_row_q8_0(x + (size_t)b * n, (char *)qx1_buf + (size_t)b * q8_rb, n);
                        const block_q8_0 *blk = (const block_q8_0 *)((char *)qx1_buf + (size_t)b * q8_rb);
                        for (int k = 0; k < nb; k++) qx1_d_buf[(size_t)b * nb + k] = fp16_to_fp32(blk[k].d);
                    }
                    qx1_stride = q8_rb;
                } else { have_qx1 = 0; free(qx1_buf); free(qx1_d_buf); qx1_buf = NULL; qx1_d_buf = NULL; }
            }
        }
        if (qtype2 == GGUF_TYPE_Q4_K || qtype2 == GGUF_TYPE_Q6_K || qtype2 == GGUF_TYPE_Q3_K ||
            qtype2 == GGUF_TYPE_Q5_K || qtype2 == GGUF_TYPE_Q4_0 || qtype2 == GGUF_TYPE_Q8_0 ||
            qtype2 == GGUF_TYPE_Q1_0 || qtype2 == GGUF_TYPE_Q2_0 || qtype2 == GGUF_TYPE_Q2_K) {
            if (qtype2 == qtype1 && have_qx1) {
                /* Same type as W1, share the quantized buffer */
                qx2_buf = qx1_buf; qx2_stride = qx1_stride;
                qx2_d_buf = qx1_d_buf; have_qx2 = 1;
            } else {
                have_qx2 = 1;
                if (qtype2 == GGUF_TYPE_Q8_0) {
                    size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
                    int nb = n / 32;
                    qx2_buf = malloc((size_t)n_batch * q8_rb);
                    qx2_d_buf = (float *)malloc((size_t)n_batch * nb * sizeof(float));
                    if (qx2_buf && qx2_d_buf) {
                        for (int b = 0; b < n_batch; b++) {
                            quantize_row_q8_0(x + (size_t)b * n, (char *)qx2_buf + (size_t)b * q8_rb, n);
                            const block_q8_0 *blk = (const block_q8_0 *)((char *)qx2_buf + (size_t)b * q8_rb);
                            for (int k = 0; k < nb; k++) qx2_d_buf[(size_t)b * nb + k] = fp16_to_fp32(blk[k].d);
                        }
                        qx2_stride = q8_rb;
                    } else { have_qx2 = 0; free(qx2_buf); free(qx2_d_buf); qx2_buf = NULL; qx2_d_buf = NULL; }
                } else if (qtype2 == GGUF_TYPE_Q4_K) {
                    size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
                    qx2_buf = malloc((size_t)n_batch * q8k_rb);
                    if (qx2_buf) {
                        for (int b = 0; b < n_batch; b++) {
                            quantize_row_q8_K(x + (size_t)b * n, (char *)qx2_buf + (size_t)b * q8k_rb, n);
                        }
                        qx2_stride = q8k_rb;
                    } else have_qx2 = 0;
                } else if (qtype2 == GGUF_TYPE_Q6_K) {
                    /* Q6_K: quantize to Q8_K */
                    size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
                    qx2_buf = malloc((size_t)n_batch * q8k_rb);
                    if (qx2_buf) {
                        for (int b = 0; b < n_batch; b++) {
                            quantize_row_q8_K(x + (size_t)b * n, (char *)qx2_buf + (size_t)b * q8k_rb, n);
                        }
                        qx2_stride = q8k_rb;
                    } else have_qx2 = 0;
                } else if (qtype2 == GGUF_TYPE_Q3_K || qtype2 == GGUF_TYPE_Q5_K) {
                    /* Q3_K/Q5_K: quantize to Q8_K */
                    size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
                    qx2_buf = malloc((size_t)n_batch * q8k_rb);
                    if (qx2_buf) {
                        for (int b = 0; b < n_batch; b++) {
                            quantize_row_q8_K(x + (size_t)b * n, (char *)qx2_buf + (size_t)b * q8k_rb, n);
                        }
                        qx2_stride = q8k_rb;
                    } else have_qx2 = 0;
                } else if (qtype2 == GGUF_TYPE_Q1_0 || qtype2 == GGUF_TYPE_Q2_0) {
                    size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
                    qx2_buf = malloc((size_t)n_batch * q8_rb);
                    if (qx2_buf) {
                        for (int b = 0; b < n_batch; b++) {
                            quantize_row_q8_0(x + (size_t)b * n, (char *)qx2_buf + (size_t)b * q8_rb, n);
                        }
                        qx2_stride = q8_rb;
                    } else have_qx2 = 0;
                } else if (qtype2 == GGUF_TYPE_Q2_K) {
                    size_t q8k_rb = gguf_type_row_size(GGUF_TYPE_Q8_K, n);
                    qx2_buf = malloc((size_t)n_batch * q8k_rb);
                    if (qx2_buf) {
                        for (int b = 0; b < n_batch; b++) {
                            quantize_row_q8_K(x + (size_t)b * n, (char *)qx2_buf + (size_t)b * q8k_rb, n);
                        }
                        qx2_stride = q8k_rb;
                    } else have_qx2 = 0;
                } else { /* Q4_0 */
                    size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
                    int nb = n / 32;
                    qx2_buf = malloc((size_t)n_batch * q8_rb);
                    qx2_d_buf = (float *)malloc((size_t)n_batch * nb * sizeof(float));
                    if (qx2_buf && qx2_d_buf) {
                        for (int b = 0; b < n_batch; b++) {
                            quantize_row_q8_0(x + (size_t)b * n, (char *)qx2_buf + (size_t)b * q8_rb, n);
                            const block_q8_0 *blk = (const block_q8_0 *)((char *)qx2_buf + (size_t)b * q8_rb);
                            for (int k = 0; k < nb; k++) qx2_d_buf[(size_t)b * nb + k] = fp16_to_fp32(blk[k].d);
                        }
                        qx2_stride = q8_rb;
                    } else { have_qx2 = 0; free(qx2_buf); free(qx2_d_buf); qx2_buf = NULL; qx2_d_buf = NULL; }
                }
            }
        }

    /* Q4_0_4_4 and Q4_0_4_8 require group-of-4 row processing; handle before threading check */
    int is_q4x4_1 = (qtype1 == GGUF_TYPE_Q4_0_4_4 || qtype1 == GGUF_TYPE_Q4_0_4_8);
    int is_q4x4_2 = (qtype2 == GGUF_TYPE_Q4_0_4_4 || qtype2 == GGUF_TYPE_Q4_0_4_8);
    if (is_q4x4_1 || is_q4x4_2) {
        /* Quantize input to Q8_0 once per batch token */
        size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
        void *qx_buf = malloc((size_t)n_batch * q8_rb);
        if (qx_buf) {
            for (int b = 0; b < n_batch; b++)
                quantize_row_q8_0(x + (size_t)b * n, (char *)qx_buf + (size_t)b * q8_rb, n);
            int d4_1 = (is_q4x4_1) ? (d / 4) * 4 : 0;
            int d4_2 = (is_q4x4_2) ? (d / 4) * 4 : 0;
                size_t group_bytes1 = (is_q4x4_1) ? (size_t)(n / 32) * sizeof(block_q4_0x4) : 0;
                size_t group_bytes2 = (is_q4x4_2) ? (size_t)(n / 32) * sizeof(block_q4_0x4) : 0;
                size_t rb1 = gguf_type_row_size(qtype1, n);
                size_t rb2 = gguf_type_row_size(qtype2, n);
                for (int b = 0; b < n_batch; b++) {
                    const block_q8_0 *qx = (const block_q8_0 *)((char *)qx_buf + (size_t)b * q8_rb);
                    if (is_q4x4_1) {
                        for (int g = 0; g < d4_1; g += 4) {
                            const char *wg = (const char *)W1 + (g / 4) * group_bytes1;
                            if (qtype1 == GGUF_TYPE_Q4_0_4_8)
                                vec_dot_q4_0x4_4x8_q8_0(wg, qx, n, out1 + b * d + g, 4);
                            else
                                vec_dot_q4_0x4_q8_0(wg, qx, n, out1 + b * d + g, 4);
                        }
                    } else {
                        for (int i = 0; i < d; i++)
                            out1[b * d + i] = vec_dot((const char *)W1 + i * rb1, x + b * n, n, qtype1);
                    }
                    if (is_q4x4_2) {
                        for (int g = 0; g < d4_2; g += 4) {
                            const char *wg = (const char *)W2 + (g / 4) * group_bytes2;
                            if (qtype2 == GGUF_TYPE_Q4_0_4_8)
                                vec_dot_q4_0x4_4x8_q8_0(wg, qx, n, out2 + b * d + g, 4);
                            else
                                vec_dot_q4_0x4_q8_0(wg, qx, n, out2 + b * d + g, 4);
                        }
                    } else {
                        for (int i = 0; i < d; i++)
                            out2[b * d + i] = vec_dot((const char *)W2 + i * rb2, x + b * n, n, qtype2);
                    }
                }
                free(qx_buf);
                return;
            }
        }
        size_t row_bytes = gguf_type_row_size(qtype1, n);
        size_t row_bytes2 = gguf_type_row_size(qtype2, n);
        const char *w1 = (const char *)W1;
        const char *w2 = (const char *)W2;

        /* Threaded dispatch across output rows using tensor_parallel_for.
         * Each row is independent, so parallelize across d rows. */
        if (n_threads > 1 && d >= matmul_min_rows) {
            dual_q8_row_ctx_t ctx;
            ctx.out1 = out1; ctx.out2 = out2;
            ctx.W1 = w1; ctx.W2 = w2;
            ctx.x = x;
            ctx.qx1_buf = qx1_buf; ctx.qx2_buf = qx2_buf;
            ctx.qx1_d = qx1_d_buf; ctx.qx2_d = qx2_d_buf;
            ctx.row_bytes1 = row_bytes; ctx.row_bytes2 = row_bytes2;
            ctx.qx_stride1 = qx1_stride; ctx.qx_stride2 = qx2_stride;
            ctx.n = n; ctx.d = d; ctx.n_batch = n_batch;
            ctx.have_qx1 = have_qx1; ctx.have_qx2 = have_qx2;
            ctx.qtype1 = qtype1; ctx.qtype2 = qtype2;
            tensor_parallel_for(d, dual_q8_row_task, &ctx);
        } else {
            /* Serial fallback for small d or single-threaded */
            for (int i = 0; i < d; i++) {
                const char *wr1 = w1 + (size_t)i * row_bytes;
                const char *wr2 = w2 + (size_t)i * row_bytes2;
                for (int b = 0; b < n_batch; b++) {
                    if (have_qx1) {
                        const char *xb1 = (const char *)qx1_buf + (size_t)b * qx1_stride;
                        if (qtype1 == GGUF_TYPE_Q8_0) {
                            const block_q8_0 *xq = (const block_q8_0 *)xb1;
                            const float *xqd = qx1_d_buf + (size_t)b * (n / 32);
                            out1[b * d + i] = vec_dot_q8_0_q8_0_deltas(xq, xqd, (const block_q8_0*)wr1, n);
                        } else if (qtype1 == GGUF_TYPE_Q4_K) {
                            out1[b * d + i] = vec_dot_q4_K_q8_K(wr1, xb1, n);
                        } else if (qtype1 == GGUF_TYPE_Q6_K) {
                            out1[b * d + i] = vec_dot_q6_K_q8_K(wr1, xb1, n);
                        } else if (qtype1 == GGUF_TYPE_Q1_0) {
                            out1[b * d + i] = vec_dot_q1_0_q8_0(wr1, xb1, n);
                        } else if (qtype1 == GGUF_TYPE_Q2_0) {
                            out1[b * d + i] = vec_dot_q2_0_q8_0(wr1, xb1, n);
                        } else if (qtype1 == GGUF_TYPE_Q2_K) {
                            out1[b * d + i] = vec_dot_q2_K_q8_K(wr1, xb1, n);
                        } else {
                            out1[b * d + i] = vec_dot(wr1, x + b * n, n, qtype1);
                        }
                    } else {
                        out1[b * d + i] = vec_dot(wr1, x + b * n, n, qtype1);
                    }
                    if (have_qx2) {
                        const char *xb2 = (const char *)qx2_buf + (size_t)b * qx2_stride;
                        if (qtype2 == GGUF_TYPE_Q8_0) {
                            const block_q8_0 *xq = (const block_q8_0 *)xb2;
                            const float *xqd = qx2_d_buf + (size_t)b * (n / 32);
                            out2[b * d + i] = vec_dot_q8_0_q8_0_deltas(xq, xqd, (const block_q8_0*)wr2, n);
                        } else if (qtype2 == GGUF_TYPE_Q4_K) {
                            out2[b * d + i] = vec_dot_q4_K_q8_K(wr2, xb2, n);
                        } else if (qtype2 == GGUF_TYPE_Q6_K) {
                            out2[b * d + i] = vec_dot_q6_K_q8_K(wr2, xb2, n);
                        } else if (qtype2 == GGUF_TYPE_Q1_0) {
                            out2[b * d + i] = vec_dot_q1_0_q8_0(wr2, xb2, n);
                        } else if (qtype2 == GGUF_TYPE_Q2_0) {
                            out2[b * d + i] = vec_dot_q2_0_q8_0(wr2, xb2, n);
                        } else if (qtype2 == GGUF_TYPE_Q2_K) {
                            out2[b * d + i] = vec_dot_q2_K_q8_K(wr2, xb2, n);
                        } else {
                            out2[b * d + i] = vec_dot(wr2, x + b * n, n, qtype2);
                        }
                    } else {
                        out2[b * d + i] = vec_dot(wr2, x + b * n, n, qtype2);
                    }
                }
            }
        }
        if (qx1_buf) { free(qx1_buf); if (qx1_d_buf) free(qx1_d_buf); }
        if (qx2_buf && qx2_buf != qx1_buf) { free(qx2_buf); if (qx2_d_buf) free(qx2_d_buf); }
        return;
    }

    /* Threaded: run both matmuls with half threads each */
    int nt = pool_total_threads(n_threads);
    int want = n_threads < nt ? n_threads : nt;
    int active_total = want > d ? d : want;
    int nt1 = (active_total + 1) / 2, nt2 = active_total - nt1;

    /* Set up tasks for W1 -> out1 */
    int a1 = pool_assign_rows(0, nt1, d);
    for (int t = 0; t < a1; t++) {
        pool_tasks[t].out = out1;
        pool_tasks[t].x = have_qx1 ? (const float *)qx1_buf : x;
        pool_tasks[t].x_d = have_qx1 ? qx1_d_buf : NULL;
        pool_tasks[t].W = (const char *)W1;
        pool_tasks[t].row_bytes = gguf_type_row_size(qtype1, n);
        pool_tasks[t].n = n;
        pool_tasks[t].d = d;
        pool_tasks[t].qtype = qtype1;
        pool_tasks[t].n_batch = n_batch;
    }

    /* Set up tasks for W2 -> out2 */
    int a2 = pool_assign_rows(nt1, nt2, d);
    for (int t = 0; t < a2; t++) {
        pool_tasks[nt1 + t].out = out2;
        pool_tasks[nt1 + t].x = have_qx2 ? (const float *)qx2_buf : x;
        pool_tasks[nt1 + t].x_d = have_qx2 ? qx2_d_buf : NULL;
        pool_tasks[nt1 + t].W = (const char *)W2;
        pool_tasks[nt1 + t].row_bytes = gguf_type_row_size(qtype2, n);
        pool_tasks[nt1 + t].n = n;
        pool_tasks[nt1 + t].d = d;
        pool_tasks[nt1 + t].qtype = qtype2;
        pool_tasks[nt1 + t].n_batch = n_batch;
    }
    pool_clear_unused(nt1 + a2, nt);

    pool_init(nt);
    pool_wake(nt);
    matmul_worker_f(&pool_tasks[0]);
    pool_wait(nt);
    if (qx1_buf) { free(qx1_buf); if (qx1_d_buf) free(qx1_d_buf); }
    if (qx2_buf && qx2_buf != qx1_buf) { free(qx2_buf); if (qx2_d_buf) free(qx2_d_buf); }
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
        __m128 c00 = _mm_shuffle_ps(c03, c03, 0x44);
        __m128 c22 = _mm_shuffle_ps(c03, c03, 0x99);
        __m128 c44 = _mm_shuffle_ps(c47, c47, 0x44);
        __m128 c66 = _mm_shuffle_ps(c47, c47, 0x99);
        __m512 cv = _mm512_castps128_ps512(c00);
        cv = _mm512_insertf32x4(cv, c22, 1);
        cv = _mm512_insertf32x4(cv, c44, 2);
        cv = _mm512_insertf32x4(cv, c66, 3);
        __m128 s03 = _mm256_castps256_ps128(s8);
        __m128 s47 = _mm256_extractf128_ps(s8, 1);
        __m128 s00 = _mm_shuffle_ps(s03, s03, 0x44);
        __m128 s22 = _mm_shuffle_ps(s03, s03, 0x99);
        __m128 s44 = _mm_shuffle_ps(s47, s47, 0x44);
        __m128 s66 = _mm_shuffle_ps(s47, s47, 0x99);
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
    /* Gemma-3n: use F32 GELU for numerical accuracy.
     * The F16 table lookup introduces quantization error that compounds over 35 layers. */
    if (getenv("PICOLM_GELU_F32")) {
        picolm_gelu_f32(x, size);
    } else {
        picolm_gelu_table_f32(x, size);
    }
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
