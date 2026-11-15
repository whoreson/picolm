/* tools/msvc_cpu_probe.c -- report the CPU's SIMD tiers for `make msvc`.
 *
 * Why a compiled probe instead of /proc/cpuinfo or WMI:
 *   - /proc/cpuinfo only exists under MSYS2/Cygwin, not from a plain CMD
 *     prompt or FAR Manager, which is where users actually build.
 *   - WMI (wmic/powershell) describes the CPU model, not its feature bits,
 *     and is deprecated on Windows 11.
 *   - __cpuid is intrinsic on every MSVC and needs no external tool.
 *
 * Output: one space-separated line, highest tier last, e.g.
 *   fma f16c avx512f avx2 avx
 * The consumer (tools/msvc_build.bat) greps for "avx2 " / "avx512f " so the
 * substring quirks ("avx" is a prefix of "avx2") are avoided by the trailing
 * space in the search string.
 *
 * Deliberately plain C89: this must compile with any cl.exe, including the
 * oldest toolset a user might have.
 */
#include <stdio.h>

#if defined(_MSC_VER)
#include <intrin.h>
#define CPUID(info, leaf)        __cpuid((info), (leaf))
#define CPUIDX(info, leaf, sub)  __cpuidex((info), (leaf), (sub))
#else
/* Non-MSVC: fall back to compiling to no output so the script can treat an
 * empty result as "no SIMD detected". Keeps this file buildable anywhere
 * (people do run `cc tools/msvc_cpu_probe.c` to sanity-check it). */
#define CPUID(info, leaf)        ((void)0)
#define CPUIDX(info, leaf, sub)  ((void)0)
#endif

int main(void) {
    int info[4];
    int max_leaf;
    int have_avx = 0, have_avx2 = 0, have_avx512f = 0;
    int have_fma = 0, have_f16c = 0;

    info[0] = info[1] = info[2] = info[3] = 0;
    CPUID(info, 0);
    max_leaf = info[0];

    if (max_leaf >= 1) {
        CPUID(info, 1);
        have_avx   = (info[2] >> 28) & 1;   /* ECX bit 28: AVX   */
        have_fma   = (info[2] >> 12) & 1;   /* ECX bit 12: FMA3  */
        have_f16c  = (info[2] >> 29) & 1;   /* ECX bit 29: F16C  */
    }
    if (max_leaf >= 7) {
        CPUIDX(info, 7, 0);
        have_avx2    = (info[1] >>  5) & 1; /* EBX bit  5: AVX2  */
        have_avx512f = (info[1] >> 16) & 1; /* EBX bit 16: AVX512F */
    }

    if (have_fma)      printf("fma ");
    if (have_f16c)     printf("f16c ");
    if (have_avx512f)  printf("avx512f ");
    if (have_avx2)     printf("avx2 ");
    if (have_avx)      printf("avx ");
    printf("\n");
    return 0;
}
