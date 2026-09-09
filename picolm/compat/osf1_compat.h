/*
 * OSF/1 Tru64 UNIX compatibility layer (DEC Alpha, GCC 3.x)
 *
 * Provides:
 *  - snprintf wrapper (float format support via system vsprintf)
 *  - atoll, roundf implementations
 *  - fmaf macro (avoids GCC 3.3.6 Alpha codegen bugs)
 *  - Declarations for portable_snprintf/portable_vsnprintf from snprintf.c
 */

#ifndef PICOLM_COMPAT_OSF1_H
#define PICOLM_COMPAT_OSF1_H

#include <stdarg.h>
#include <stddef.h>

/* From portable snprintf (Mark Martinec, compiled with USE_SNPRINTF+HAVE_SNPRINTF) */
extern int portable_snprintf(char *str, size_t str_m, const char *fmt, ...);
extern int portable_vsnprintf(char *str, size_t str_m, const char *fmt, va_list ap);

/* Our supplemented snprintf -- handles float formats via system vsprintf */
int snprintf(char *str, size_t str_m, const char *fmt, ...);

/* Math/utility stubs not present in OSF/1 libc with -std=c99 */
long long atoll(const char *str);
float roundf(float x);
float fmaf(float x, float y, float z);
/* Also define as macro for inlining -- avoids GCC 3.3.6 Alpha codegen bugs.
 * The function symbol exists for files that don't include this header. */
#define fmaf(x, y, z) ((x) * (y) + (z))

/* Memory locking: OSF/1 libc_r has memlk/memunlk in <sys/mman.h>.
 * Don't redeclare them here -- they're real functions, not stubs.
 * The model code uses them via model.h guards (#ifdef RLIMIT_MEMLOCK). */

#endif /* PICOLM_COMPAT_OSF1_H */

