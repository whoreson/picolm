/* dos_stubs.c - Minimal DJGPP runtime stubs for MS-DOS builds.
 * Provides symbols that the DJGPP libc.a expects from stubify's output object.
 * A real DJGPP build would use stubify.exe to generate these properly.
 * This stub allows linking; the executable will work for basic operations
 * as long as environment variables and DPMI-specific features aren't used. */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* DJGPP runtime globals (normally provided by stubify).
 * Declared as COMMON (uninitialized) so the linker places them in BSS.
 * The DJGPP go32 extender patches these at runtime with real values
 * (DPMI base address, data segment selector, stack boundaries, etc.).
 * This is the cross-platform substitute for stubify.exe which can't
 * run on Linux build hosts. */
struct _reent;
struct _reent *_impure_ptr;
char **_environ;
unsigned char _ctype_[257];
unsigned long ___djgpp_base_address;
unsigned long ___djgpp_ds_alias;
unsigned long ___djgpp_stack_top;
unsigned long ___djgpp_stack_limit;
unsigned long ___djgpp_selector_limit;
unsigned long _stklen;
unsigned long _crt0_startup_flags;
unsigned long _crt0_init_mcount;
unsigned long _stubinfo;

int *__errno(void) {
    static int errno_val;
    return &errno_val;
}

void __assert_func(const char *file, int line, const char *func, const char *expr) {
    fprintf(stderr, "Assertion failed: %s, file %s, line %d, function %s\n",
            expr, file, line, func ? func : "(null)");
    abort();
}

/* C11/POSIX functions not available on DOS */
void *aligned_alloc(size_t alignment, size_t size) {
    (void)alignment;
    return malloc(size);
}

float fmaf(float a, float b, float c) { return a * b + c; }

/* POSIX memory locking - no-op on DOS */
int mlock(const void *addr, size_t len) { (void)addr; (void)len; return 0; }
int munlock(const void *addr, size_t len) { (void)addr; (void)len; return 0; }

/* POSIX mmap/munmap - stubs (real DOS uses fread) */
void *mmap(void *addr, size_t len, int prot, int flags, int fd, long offset) {
    (void)addr; (void)len; (void)prot; (void)flags; (void)fd; (void)offset;
    return (void*)-1;
}
int munmap(void *addr, size_t len) { (void)addr; (void)len; return 0; }

/* Threading stubs */
int pthread_mutex_destroy(void *m) { (void)m; return 0; }

/* Server stub (server.c not compiled for DOS) */
int server_main(void *sc) { (void)sc; return 0; }

/* Misc POSIX stubs */
struct timeval { long tv_sec; long tv_usec; };
int gettimeofday(struct timeval *tv, void *tz) { (void)tv; (void)tz; return 0; }
int isatty(int fd) { (void)fd; return 0; }
int access(const char *path, int amode) { (void)path; (void)amode; return -1; }
