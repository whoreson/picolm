/*
 * Supplement for portable snprintf on OSF/1 Alpha (GCC 3.3.6).
 *
 * The portable snprintf (Mark Martinec) doesn't handle f/e/g/E/a/A
 * floating-point format specifiers. This wrapper intercepts those
 * and uses the system's sprintf for float formatting, then assembles
 * the final result into the snprintf buffer with proper truncation.
 *
 * Also provides: atoll, roundf, fmaf, memlk, memunlk stubs.
 */

#include "osf1_compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Check if a format string contains float conversion specifiers
 * (f, F, e, E, g, G, a, A). If so, we can't use portable_snprintf.
 */
static int has_float_format(const char *fmt) {
    while (*fmt) {
        if (*fmt == '%') {
            fmt++;
            /* skip flags */
            while (*fmt == '-' || *fmt == '+' || *fmt == ' ' || *fmt == '#' || *fmt == '0')
                fmt++;
            /* skip width */
            while (*fmt >= '0' && *fmt <= '9') fmt++;
            if (*fmt == '*') { fmt++; }
            /* skip precision */
            if (*fmt == '.') {
                fmt++;
                if (*fmt == '*') { fmt++; }
                while (*fmt >= '0' && *fmt <= '9') fmt++;
            }
            /* skip length modifier */
            while (*fmt == 'h' || *fmt == 'l' || *fmt == 'L' || *fmt == 'z' || *fmt == 'j' || *fmt == 't')
                fmt++;
            /* check conversion specifier */
            if (*fmt == 'f' || *fmt == 'F' || *fmt == 'e' || *fmt == 'E' ||
                *fmt == 'g' || *fmt == 'G' || *fmt == 'a' || *fmt == 'A')
                return 1;
        }
        fmt++;
    }
    return 0;
}

/*
 * Our snprintf wrapper: if format has float specifiers, use the system's
 * snprintf via vfprintf to a temp buffer, then copy to dest with truncation.
 * Otherwise, use the portable snprintf.
 */
int snprintf(char *str, size_t str_m, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);

    if (has_float_format(fmt)) {
        /* Use system's vsprintf for float formatting.
         * We need a temp buffer large enough. */
        char tmp[4096];
        int n;
        /* vsprintf doesn't have length limit, but our temp buffer is large enough
         * for all current uses in picolm code. */
        n = vsprintf(tmp, fmt, ap);
        /* Copy to destination with proper truncation */
        if (str_m > 0) {
            size_t copy_len = (size_t)n < str_m - 1 ? (size_t)n : str_m - 1;
            memcpy(str, tmp, copy_len);
            str[copy_len] = '\0';
        }
        va_end(ap);
        return n;
    }

    /* No float formats: use portable snprintf */
    int result = portable_vsnprintf(str, str_m, fmt, ap);
    va_end(ap);
    return result;
}

/* atoll: convert string to long long (strtoll may not exist on OSF/1) */
long long atoll(const char *str) {
    long long result = 0;
    int sign = 1;
    while (*str == ' ' || *str == '\t') str++;
    if (*str == '-') { sign = -1; str++; }
    else if (*str == '+') { str++; }
    while (*str >= '0' && *str <= '9') {
        result = result * 10 + (*str - '0');
        str++;
    }
    return result * sign;
}

/* roundf: round float to nearest integer */
float roundf(float x) {
    return (x >= 0.0f) ? (float)((int)(x + 0.5f)) : (float)((int)(x - 0.5f));
}

/* fmaf: provide a function symbol for the linker (other .o files may not
 * include osf1_compat.h and need the fmaf symbol). Undef the macro first
 * so the function name doesn't get expanded. */
#ifdef fmaf
#define _fmaf_macro_defined
#undef fmaf
#endif
float fmaf(float x, float y, float z) {
    return x * y + z;
}
#ifdef _fmaf_macro_defined
#undef _fmaf_macro_defined
#define fmaf(x, y, z) ((x) * (y) + (z))
#endif

/* memlk/memunlk: declared in <sys/mman.h> but not implemented in libc_r.
 * Provide no-op stubs so the linker is happy. */
int memlk(const void *addr, size_t len) {
    (void)addr; (void)len;
    return -1;
}

int memunlk(const void *addr, size_t len) {
    (void)addr; (void)len;
    return -1;
}

