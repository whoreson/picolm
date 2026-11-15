/* compat/msvc_c11_compat.h -- C11 <stdatomic.h> and <pthread.h> shims for MSVC.
 *
 * viz.c is written against C11 atomics and pthreads, which MSVC does not
 * provide: it has no usable <stdatomic.h> (verified on 14.44 and 14.51, with
 * and without /std:c11) and no <pthread.h>. Rather than fork viz.c into a
 * Windows dialect -- which would mean every future change to viz.c needing to
 * be made twice -- this header supplies the small subset both use, mapping
 * onto Win32 Interlocked intrinsics and the Win32 thread API.
 *
 * Only the operations viz.c actually uses are provided. Adding more is
 * straightforward, but anything not listed here has not been thought through
 * for Windows, so it should be added deliberately rather than assumed.
 *
 * Used only when _MSC_VER is defined; GCC/Clang/Clang-cl on POSIX and MinGW
 * take the real system headers. MinGW has both, so it is deliberately NOT
 * covered by this shim (test with #if defined(_MSC_VER), not _WIN32).
 *
 * No C11 compiler features are required: no _Static_assert, no _Generic,
 * no typeof. MSVC compiles this as C89/C99.
 */
#ifndef PICOLM_MSVC_C11_COMPAT_H
#define PICOLM_MSVC_C11_COMPAT_H

#if defined(_MSC_VER) && !defined(__cplusplus)

#include <intrin.h>
#include <stddef.h>
#include <stdlib.h>   /* malloc/free for the thread trampoline */
#include <process.h>  /* _beginthreadex */

/* Deliberately NOT <windows.h>. windows.h drags in winsock.h (v1), which makes
 * a later #include <winsock2.h> fail with "struct type redefinition" for
 * fd_set, timeval, hostent, WSAData and every socket function. viz.h is
 * included by server.c and picolm.c before their own winsock2.h, so pulling in
 * windows.h here poisoned those translation units.
 *
 * The Win32 types needed are therefore opaque:
 *   - the mutex is a byte blob large enough for a CRITICAL_SECTION and aligned
 *     for pointers. Its layout is not touched here; Initialize/Enter/Leave/
 *     DeleteCriticalSection all take a pointer, and those are declared below
 *     with matching prototypes so no Win32 header is required.
 *   - a thread handle is a void *. */
typedef void *pthread_handle_t;

/* ============================================================
 * _Atomic
 *
 * MSVC has no _Atomic qualifier. Every use in the tree is the prefix form
 * ("_Atomic int x;", "_Atomic unsigned y;") on a scalar that is only ever
 * accessed through the atomic_* functions below, which take a pointer and use
 * Interlocked*, so plain storage is sufficient and correct. Expanding _Atomic
 * to nothing keeps those declarations valid as written.
 *
 * Note this is a plain macro, not function-like: the code writes
 * "_Atomic int x", never "_Atomic(int) x", and a function-like macro would
 * leave the bare token behind.
 * ============================================================ */
#define _Atomic

/* ============================================================
 * memory_order
 *
 * Interlocked* operations are full barriers on x86/x64 (LOCK prefix or an
 * implicit one), which is at least as strong as any of these, so the orders
 * collapse to a single implementation. The enum is still defined so that
 * viz.c's explicit annotations compile and stay readable.
 * ============================================================ */
typedef enum {
    memory_order_relaxed = 0,
    memory_order_consume = 1,
    memory_order_acquire = 2,
    memory_order_release = 3,
    memory_order_acq_rel = 4,
    memory_order_seq_cst = 5
} memory_order;

typedef int          atomic_int;
typedef unsigned int atomic_uint;
typedef long         atomic_long;
typedef void        *atomic_ptr;

/* ---- load --------------------------------------------------- */
/* A plain read is a single naturally-aligned load on x86/x64, which cannot
 * tear. _ReadWriteBarrier stops the compiler from reordering or caching it
 * across other atomic operations. */
static __inline int atomic_load_explicit(const volatile int *p, memory_order mo) {
    int v = *(const volatile int *)p;
    (void)mo;
    _ReadWriteBarrier();
    return v;
}
#define atomic_load(p) atomic_load_explicit((p), memory_order_seq_cst)

static __inline unsigned atomic_load_u_explicit(const volatile unsigned *p, memory_order mo) {
    unsigned v = *(const volatile unsigned *)p;
    (void)mo;
    _ReadWriteBarrier();
    return v;
}

/* ---- store -------------------------------------------------- */
static __inline void atomic_store_explicit(volatile int *p, int v, memory_order mo) {
    (void)mo;
    _ReadWriteBarrier();
    *(volatile int *)p = v;
    _ReadWriteBarrier();
}
#define atomic_store(p, v) atomic_store_explicit((p), (v), memory_order_seq_cst)

static __inline void atomic_store_u_explicit(volatile unsigned *p, unsigned v, memory_order mo) {
    (void)mo;
    _ReadWriteBarrier();
    *(volatile unsigned *)p = v;
    _ReadWriteBarrier();
}

/* ---- fetch_add ---------------------------------------------- */
static __inline int atomic_fetch_add_explicit(volatile int *p, int v, memory_order mo) {
    (void)mo;
    return (int)_InterlockedExchangeAdd((volatile long *)p, (long)v);
}
#define atomic_fetch_add(p, v) atomic_fetch_add_explicit((p), (v), memory_order_seq_cst)

static __inline unsigned atomic_fetch_add_u_explicit(volatile unsigned *p, unsigned v, memory_order mo) {
    (void)mo;
    return (unsigned)_InterlockedExchangeAdd((volatile long *)p, (long)v);
}

/* ---- exchange ----------------------------------------------- */
static __inline int atomic_exchange_explicit(volatile int *p, int v, memory_order mo) {
    (void)mo;
    return (int)_InterlockedExchange((volatile long *)p, (long)v);
}
#define atomic_exchange(p, v) atomic_exchange_explicit((p), (v), memory_order_seq_cst)

/* ---- fence --------------------------------------------------- */
/* On x86/x64 every load is acquire and every store is release, so a compiler
 * barrier is all a fence needs; hardware reordering of this shape does not
 * occur. */
static __inline void atomic_thread_fence(memory_order mo) {
    (void)mo;
    _ReadWriteBarrier();
}

/* ============================================================
 * pthread subset
 *
 * viz.c uses exactly one thread and one mutex. Both map onto Win32 directly.
 * pthread_t is a HANDLE; pthread_create takes the same signature MSVC's
 * _beginthreadex wrapper needs, so the thread function is unchanged.
 * ============================================================ */
/* Declared directly so that <windows.h> is not needed. Signatures match the
 * SDK's (which are __declspec(dllimport) void*); taking void* keeps the ABI
 * identical without pulling in windows.h. */
__declspec(dllimport) void __stdcall InitializeCriticalSection(void *);
__declspec(dllimport) void __stdcall EnterCriticalSection(void *);
__declspec(dllimport) void __stdcall LeaveCriticalSection(void *);
__declspec(dllimport) void __stdcall DeleteCriticalSection(void *);
__declspec(dllimport) unsigned long __stdcall WaitForSingleObject(void *, unsigned long);
__declspec(dllimport) int __stdcall CloseHandle(void *);

/* 40 bytes holds a CRITICAL_SECTION on x64; 64 leaves headroom and keeps the
 * type aligned. Never accessed field-by-field. */
typedef struct { void *p[8]; } pthread_mutex_impl_t;
typedef pthread_mutex_impl_t  pthread_mutex_t;
typedef pthread_handle_t      pthread_t;

static __inline int pthread_mutex_init(pthread_mutex_t *m, const void *attr) {
    (void)attr;
    InitializeCriticalSection((void *)m);
    return 0;
}

static __inline int pthread_mutex_destroy(pthread_mutex_t *m) {
    DeleteCriticalSection((void *)m);
    return 0;
}

static __inline int pthread_mutex_lock(pthread_mutex_t *m) {
    EnterCriticalSection((void *)m);
    return 0;
}

static __inline int pthread_mutex_unlock(pthread_mutex_t *m) {
    LeaveCriticalSection((void *)m);
    return 0;
}

/* __stdcall matches viz.c's thread entry point. The trampoline adapts the
 * void*(*)(void*) convention to Win32's unsigned __stdcall(unsigned, void*). */
typedef struct {
    void *(*fn)(void *);
    void  *arg;
} picolm_thread_tramp_t;

static unsigned __stdcall picolm_thread_tramp(void *ctx) {
    picolm_thread_tramp_t *t = (picolm_thread_tramp_t *)ctx;
    void *(*fn)(void *) = t->fn;
    void *arg = t->arg;
    free(t);
    fn(arg);
    return 0;
}

static __inline int pthread_create(pthread_t *th, const void *attr,
                                   void *(*fn)(void *), void *arg) {
    picolm_thread_tramp_t *t;
    (void)attr;
    t = (picolm_thread_tramp_t *)malloc(sizeof(*t));
    if (!t) return -1;
    t->fn = fn;
    t->arg = arg;
    *th = (pthread_handle_t)_beginthreadex(NULL, 0, picolm_thread_tramp, t, 0, NULL);
    if (*th == 0) { free(t); return -1; }
    return 0;
}

static __inline int pthread_join(pthread_t th, void **retval) {
    (void)retval;
    WaitForSingleObject((void *)th, 0xFFFFFFFFu);
    CloseHandle((void *)th);
    return 0;
}

#endif /* _MSC_VER && !__cplusplus */
#endif /* PICOLM_MSVC_C11_COMPAT_H */
