#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Portable thread-local storage.
 * MSVC uses __declspec(thread), GCC >= 4 uses __thread.
 * Old Mac OS X (Tiger/Leopard, GCC 3.3) and OSF/1 (GCC 3.x) don't support __thread. */
#if defined(_MSC_VER)
#define PICOLM_TLS __declspec(thread)
#elif defined(__APPLE__) || defined(__osf__) || (defined(__GNUC__) && __GNUC__ < 4)
#define PICOLM_TLS
#else
#define PICOLM_TLS __thread
#endif

/* restrict: not supported in GCC < 4 (e.g. Mac OS X Tiger GCC 3.3, OSF/1) */
#if !defined(__GNUC__) || __GNUC__ >= 4
#define PICOLM_RESTRICT restrict
#else
#define PICOLM_RESTRICT
#endif

/* Mac OS X uses MAP_ANON instead of MAP_ANONYMOUS */
#if defined(__APPLE__) && !defined(MAP_ANONYMOUS)
#define MAP_ANONYMOUS MAP_ANON
#endif

/* OSF/1 compatibility: provides fmaf macro, snprintf, atoll, roundf. */
#if defined(__osf__)
#include "compat/osf1_compat.h"
#endif

/* Fallback for systems without <inttypes.h> (OSF/1 V4.0, GCC 3.x, etc.) */
#ifndef PRIu64
#define PRIu64 "llu"
#endif
#ifndef PRId64
#define PRId64 "lld"
#endif

/* MSVC compatibility: GCC-only constructs not available in MSVC's compiler. */
#ifndef PICOLM_ALIGNED
#if defined(_MSC_VER)
#define PICOLM_ALIGNED(N) __declspec(align(N))
#else
#define PICOLM_ALIGNED(N) __attribute__((aligned(N)))
#endif
#endif
#ifndef PICOLM_CONSTRUCTOR
/* GCC: __attribute__((constructor)). MSVC does not support this; empty on
 * MSVC (register functions are only called explicitly if needed). */
#if defined(_MSC_VER)
#define PICOLM_CONSTRUCTOR
#else
#define PICOLM_CONSTRUCTOR __attribute__((constructor))
#endif
#endif
/* GCC-only constructs not available in MSVC's compiler. */
#if defined(_MSC_VER)
#ifndef ssize_t
#define ssize_t SSIZE_T
#endif
/* off_t is provided by the Windows SDK's sys/types.h on MSVC. */
#ifndef __thread
#define __thread __declspec(thread)
#endif
#endif
/* PICOLM_ALWAYS_INLINE: GCC __attribute__((always_inline)) vs MSVC __forceinline. */
#ifndef PICOLM_ALWAYS_INLINE
#if defined(_MSC_VER)
#define PICOLM_ALWAYS_INLINE __forceinline
#else
#define PICOLM_ALWAYS_INLINE __attribute__((always_inline))
#endif
#endif
/* PICOLM_NOINLINE: GCC __attribute__((noinline)) vs MSVC __declspec(noinline).
 * Used as a compiler workaround (e.g. GCC 4.2.1 ICE on SSSE3+SSE4.1 combo). */
#ifndef PICOLM_NOINLINE
#if defined(_MSC_VER)
#define PICOLM_NOINLINE __declspec(noinline)
#else
#define PICOLM_NOINLINE __attribute__((noinline))
#endif
#endif

/* Atomic compatibility: GCC builtins vs MSVC Interlocked.
 * GCC 4.7+ has __atomic_fetch_add/__atomic_store_n natively.
 * MSVC uses InterlockedExchangeAdd/InterlockedExchange.
 * Older GCC and other compilers fall back to plain operations
 * (unsafe for multi-threaded, but acceptable on single-core/embedded). */
#if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7))
#define PICOLM_ATOMIC_FETCH_ADD(p, v) __atomic_fetch_add((p), (v), __ATOMIC_RELAXED)
#define PICOLM_ATOMIC_ADD_FETCH(p, v) __atomic_add_fetch((p), (v), __ATOMIC_SEQ_CST)
#define PICOLM_ATOMIC_SUB_FETCH(p, v) __atomic_sub_fetch((p), (v), __ATOMIC_SEQ_CST)
#define PICOLM_ATOMIC_LOAD(p)         __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define PICOLM_ATOMIC_LOAD_SEQCST(p)  __atomic_load_n((p), __ATOMIC_SEQ_CST)
#define PICOLM_ATOMIC_STORE(p, v)     __atomic_store_n((p), (v), __ATOMIC_SEQ_CST)
#define PICOLM_ATOMIC_STORE_RELAXED(p, v) __atomic_store_n((p), (v), __ATOMIC_RELAXED)
/* GCC 4.7+ handles any type natively, so 64-bit macros alias the 32-bit ones */
#define PICOLM_ATOMIC_FETCH_ADD64(p, v) PICOLM_ATOMIC_FETCH_ADD(p, v)
#define PICOLM_ATOMIC_ADD_FETCH64(p, v) PICOLM_ATOMIC_ADD_FETCH(p, v)
#define PICOLM_ATOMIC_LOAD64(p)         PICOLM_ATOMIC_LOAD(p)
#define PICOLM_ATOMIC_LOAD64_SEQCST(p)  PICOLM_ATOMIC_LOAD_SEQCST(p)
#define PICOLM_ATOMIC_STORE64(p, v)     PICOLM_ATOMIC_STORE(p, v)
#elif defined(_MSC_VER)
#include <intrin.h>
#define PICOLM_ATOMIC_FETCH_ADD(p, v) ((int)_InterlockedExchangeAdd((volatile long *)(p), (long)(v)))
#define PICOLM_ATOMIC_ADD_FETCH(p, v) ((int)_InterlockedExchangeAdd((volatile long *)(p), (long)(v)) + (int)(v))
#define PICOLM_ATOMIC_SUB_FETCH(p, v) ((int)_InterlockedExchangeAdd((volatile long *)(p), -(long)(v)) - (int)(v))
#define PICOLM_ATOMIC_LOAD(p)         (*(const volatile int *)(p))
#define PICOLM_ATOMIC_LOAD_SEQCST(p)  (*(const volatile int *)(p))
#define PICOLM_ATOMIC_STORE(p, v)     ((*(volatile int *)(p)) = (int)(v))
#define PICOLM_ATOMIC_STORE_RELAXED(p, v) ((*(volatile int *)(p)) = (int)(v))
/* 64-bit atomic operations for pool_done (monotonic counter) */
#define PICOLM_ATOMIC_FETCH_ADD64(p, v) ((long long)_InterlockedExchangeAdd64((volatile long long *)(p), (long long)(v)))
#define PICOLM_ATOMIC_ADD_FETCH64(p, v) ((long long)_InterlockedExchangeAdd64((volatile long long *)(p), (long long)(v)) + (long long)(v))
#define PICOLM_ATOMIC_LOAD64(p)         (*(const volatile long long *)(p))
#define PICOLM_ATOMIC_LOAD64_SEQCST(p)  (*(const volatile long long *)(p))
#define PICOLM_ATOMIC_STORE64(p, v)     ((*(volatile long long *)(p)) = (long long)(v))
#else
#define PICOLM_ATOMIC_FETCH_ADD(p, v) ((*(p)) += (int)(v), (*(p) - (int)(v)))
#define PICOLM_ATOMIC_ADD_FETCH(p, v) ((*(p)) += (int)(v))
#define PICOLM_ATOMIC_SUB_FETCH(p, v) ((*(p)) -= (int)(v))
#define PICOLM_ATOMIC_LOAD(p)         (*(p))
#define PICOLM_ATOMIC_LOAD_SEQCST(p)  (*(p))
#define PICOLM_ATOMIC_STORE(p, v)     ((*(p)) = (int)(v))
#define PICOLM_ATOMIC_STORE_RELAXED(p, v) ((*(p)) = (int)(v))
/* Fallback 64-bit macros */
#define PICOLM_ATOMIC_FETCH_ADD64(p, v) ((*(p)) += (long long)(v), (*(p) - (long long)(v)))
#define PICOLM_ATOMIC_ADD_FETCH64(p, v) ((*(p)) += (long long)(v))
#define PICOLM_ATOMIC_LOAD64(p)         (*(p))
#define PICOLM_ATOMIC_LOAD64_SEQCST(p)  (*(p))
#define PICOLM_ATOMIC_STORE64(p, v)     ((*(p)) = (long long)(v))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- SIMD detection ----
 *
 * Each level explicitly implies all lower levels so that code only needs to
 * check a single flag.  The order of checks (highest first) lets each block
 * upgrade flags that a lower-level check would otherwise miss if the compiler
 * only predefines the highest applicable macro.
 *
 * Hierarchy (x86):
 *   PICOLM_SSE2 ⊂ PICOLM_SSE3 ⊂ PICOLM_AVX ⊂ PICOLM_AVX2
 *
 * ARM:
 *   PICOLM_NEON (independent)
 */

/* Forward declarations for use in inline helpers below */
/* Note: fp16_to_fp32() uses pure integer arithmetic (no FPU ops). */
float fp16_to_fp32(uint16_t h);
float fp16_to_fp32_lookup(uint16_t h);
float bf16_to_fp32(uint16_t x);

/* ===================================================================
 * fp16_table_init() -- CALL THIS BEFORE ANY DEQUANTISER OR QUANT GEMM.
 * ===================================================================
 *
 * Every block-quant format stores its scale as an fp16 word, and quant.c reads
 * it through a 64K-entry lookup table (fp16_to_fp32_lookup) rather than doing
 * the conversion inline. That table is NOT initialised automatically and
 * float zero-initialises to 0.0f.
 *
 * The failure mode is the worst kind: no crash, no warning, no wrong pointer.
 * Every scale silently becomes 0.0, so every dequantised weight and every dot
 * product is 0. Consequences that have actually bitten this tree:
 *
 *  - A test that compares two implementations "passes" because 0 == 0.
 *  - A benchmark measures nothing, because the kernels have no data.
 *  - Formats whose dequantiser does NOT use the table (F32, and F16 on SIMD
 *    builds where it uses native cvt instructions) still look correct, so a
 *    mixed-format run makes it look like a format-specific bug.
 *
 * Callers: picolm/main initialisation, model_gguf.c's loader, every test that
 * touches quant.c (see tests/test_gemm_f16_gpu.cu and
 * tests/test_bmulti_chunk.c), and anything that calls dequantize_row*(),
 * vec_dot*(), matmul*() or picolm_sgemm*() on quantised data.
 *
 * It is idempotent and cheap (one 64K pass), so calling it defensively at the
 * top of a tool's main() is always correct. */
void fp16_table_init(void);

/* ARMv8.2 SDOT/UDOT (vdotq_laneq_s32 etc). Distinct from PICOLM_NEON since
 * plain NEON hardware (no dotprod) can't run this path -- per llama.cpp's
 * own findings (and this project's own AVX1-machine test), running an
 * interleaved Q4_0x4/x8 layout through a kernel that *doesn't* have the
 * matching accelerated instruction is actively worse than the plain,
 * non-interleaved path, not just "not faster". */
#if defined(__ARM_FEATURE_DOTPROD)
#  define PICOLM_DOTPROD 1
#endif

/* ARMv8.2 I8MM (vmmlaq_s32 / vmmlaq_u32). 16 int8 x int8 -> 4 int32 lanes.
 * Each lane is an 8-element dot product. vmmlaq_s32 computes a 2x2 block
 * matmul: lanes 0,1,2,3 = a_lo.b_lo, a_lo.b_hi, a_hi.b_lo, a_hi.b_hi.
 * Requires __ARM_FEATURE_SVE_MATMUL_INT8 or explicit -march=armv8.2-a+i8mm. */
#if defined(__ARM_FEATURE_MATMUL_INT8) || defined(__ARM_FEATURE_SVE_MATMUL_INT8)
#  define PICOLM_I8MM 1
#endif

/* --- ARM NEON --- */
/* Guard against CUDA device compilation: nvcc cannot handle arm_neon.h.
 * __ARM_NEON/__ARM_NEON__ is auto-defined by the compiler when NEON is
 * enabled (64-bit ARM always, 32-bit ARM requires -mfpu=neon or
 * -march=armv8-a+simd).  PICOLM_NEON may also be manually defined via
 * -DPICOLM_NEON (e.g. make pi), in which case we still need the header.
 *
 * On aarch64, all NEON intrinsics are available (including FP16 vector
 * load/store/convert via the base ISA).  On 32-bit ARM (armv7), NEON is
 * available but lacks __fp16 type, float16x4_t, vld1_f16, vcvt_f32_f16,
 * vaddvq_f32, vaddvq_s32, vmaxvq_f32, vcvtnq_s32_f32, vmull_high_s8.
 * PICOLM_NEON_AARCH64 is defined when we have the full aarch64 NEON set. */
#if (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(PICOLM_NEON)) && !defined(__CUDACC__)
#  define PICOLM_NEON 1
#  include <arm_neon.h>
#  ifdef __aarch64__
#    define PICOLM_NEON_AARCH64 1
#  endif
static inline float vaddvq_f32_compat(float32x4_t v) {
#  ifdef PICOLM_NEON_AARCH64
    return vaddvq_f32(v);
#  else
    float32x2_t r = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(r, r), 0);
#  endif
}

/* --- 32-bit ARM NEON compatibility shims --- */
/* On armv7 (32-bit), several aarch64 NEON intrinsics are missing.
 * Provide compat wrappers so existing PICOLM_NEON code compiles. */
#if defined(PICOLM_NEON) && !defined(PICOLM_NEON_AARCH64)

/* Horizontal sum/reduction intrinsics */
static inline float vaddvq_f32(float32x4_t v) {
    float32x2_t r = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}

static inline int32_t vaddvq_s32(int32x4_t v) {
    int32x2_t r = vpadd_s32(vget_low_s32(v), vget_high_s32(v));
    return vget_lane_s32(vpadd_s32(r, r), 0);
}

static inline float vmaxvq_f32(float32x4_t v) {
    float32x2_t r = vmax_f32(vget_low_f32(v), vget_high_f32(v));
    float32x2_t r2 = vpmax_f32(r, r);
    return vget_lane_f32(r2, 0);
}

/* FP16 vector types and intrinsics (aarch64-only, shim for 32-bit) */
typedef uint16x4_t float16x4_t;
typedef uint16_t float16_t;

static inline float16x4_t vld1_f16(const float16_t *p) {
    return vld1_u16(p);
}

/* Forward decl: fp16x4_to_fp32_inline defined later in this header */
static inline float32x4_t fp16x4_to_fp32_inline(const uint16_t *p);

static inline float32x4_t vcvt_f32_f16(float16x4_t v) {
    uint16_t buf[4]; vst1_u16(buf, v);
    return fp16x4_to_fp32_inline(buf);
}

/* vcvtnq_s32_f32: float32x4 -> int32x4 (round-to-nearest, saturate)
 * On aarch64 this is a single instruction. On 32-bit ARM, emulate using
 * bit manipulation to avoid FP comparison intrinsics. */
static inline int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    /* Extract sign bits, work with absolute values */
    int32x4_t sign_bits = vreinterpretq_s32_f32(v);
    int32x4_t abs_bits = vbicq_s32(sign_bits, vdupq_n_s32(0x80000000));
    float32x4_t absv = vreinterpretq_f32_s32(abs_bits);
    float32x4_t rounded = vaddq_f32(absv, vdupq_n_f32(0.5f));
    int32x4_t result = vcvtq_s32_f32(rounded);
    /* Restore sign: if sign bit was set, negate */
    int32x4_t negated = vnegq_s32(result);
    uint32x4_t sign_mask = vreinterpretq_u32_s32(vshrq_n_s32(sign_bits, 31));
    return vbslq_s32(sign_mask, negated, result);
}

/* vmull_high_s8: multiply high 8 lanes of two int8x16, return int16x8 */
static inline int16x8_t vmull_high_s8(int8x16_t a, int8x16_t b) {
    return vmull_s8(vget_high_s8(a), vget_high_s8(b));
}

/* vaddl_high_s16: widen-add high 4 lanes of two int16x8, return int32x4 */
static inline int32x4_t vaddl_high_s16(int16x8_t a, int16x8_t b) {
    int16x4_t ah = vget_high_s16(a);
    int16x4_t bh = vget_high_s16(b);
    int32x4_t sa = vmovl_s16(ah);
    int32x4_t sb = vmovl_s16(bh);
    return vaddq_s32(sa, sb);
}

/* vqtbl1q_u8: 16-wide LUT table lookup
 * 32-bit ARM has no 16-wide table lookup; use vtbl2_u8 (two 8-byte tables
 * forming a 16-byte contiguous table) with 8 indices at a time. */
static inline uint8x16_t vqtbl1q_u8(uint8x16_t lut, uint8x16_t idx) {
    uint8x8x2_t tbl; tbl.val[0] = vget_low_u8(lut); tbl.val[1] = vget_high_u8(lut);
    uint8x8_t lo = vtbl2_u8(tbl, vget_low_u8(idx));
    uint8x8_t hi = vtbl2_u8(tbl, vget_high_u8(idx));
    return vcombine_u8(lo, hi);
}

/* Note: __fp16 is NOT shimmed on 32-bit ARM. Code using __fp16 must
 * guard with #ifdef PICOLM_NEON_AARCH64 or use fp16_to_fp32_lookup(). */

#endif  /* PICOLM_NEON && !PICOLM_NEON_AARCH64 */
#endif  /* main PICOLM_NEON block */

/* --- ARM NEON feature detection (compiler-only) --- */
/* Hardware FP16 vector conversion (ARMv8.2-A asimdhp).
 * On older NEON hardware (ARMv8.0/8.1), this path is not available and
 * the scalar fp16_to_fp32 lookup table is used instead. */
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(__CUDACC__)
#  define PICOLM_FP16_HW 1
#  include <arm_acle.h>
#endif

/* FP16 hardware conversion helpers (ARMv8.2-A only) */
#if defined(PICOLM_FP16_HW)
static inline float32x4_t fp16x4_to_f32_hw(const uint16_t *p) {
    return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p)));
}
static inline void f32x4_to_fp16_hw(uint16_t *p, float32x4_t v) {
    vst1_u16(p, vreinterpret_u16_f16(vcvt_f16_f32(v)));
}
#endif

/* --- MSVC feature-macro shim ------------------------------------------------
 *
 * MSVC only predefines __AVX__, __AVX2__ and __AVX512F__ for its /arch switch.
 * It never defines the GCC-style sub-feature macros, even though the same
 * /arch level enables those instructions and MSVC's <immintrin.h> exposes the
 * intrinsics for them:
 *
 *   /arch:AVX2   ->  FMA3 (__FMA__)  and  F16C (__F16C__)
 *
 * Every `#if defined(__FMA__)` / `defined(__F16C__)` gate in this codebase then
 * evaluates false on MSVC, silently compiling whole SIMD kernels out. That is
 * not a performance-only problem: the guards also pick the *layout variant*
 * a kernel expects. On MSVC/AVX2 the tiled prefill still asked
 * attn_v_prep_run() for a slice-major V tile (v_sm=1, head_dim % 8 == 0) while
 * the PV multiply fell through every `&& defined(__FMA__)` branch to the
 * generic row-major loop -- reading the tile transposed. Result: garbled
 * output on any prefill batch long enough to engage the tiled path, while the
 * /arch:AVX512 build was fine. tests/check_simd_parity.sh flags it immediately.
 *
 * Defining them here (before any use) makes MSVC builds select the same code
 * as GCC -mavx2 -mfma -mf16c. Verified: /arch:AVX2 accepts _mm256_fmadd_ps
 * and _mm256_cvtph_ps and produces correct results.
 *
 * AVX-512 VNNI is NOT derivable on MSVC and must be passed in from outside.
 * Earlier notes here claimed "MSVC defines __AVX512VNNI__ only for targets
 * that actually have it"; that is wrong -- cl.exe 14.44 with /arch:AVX512
 * defines __AVX512F__ and __AVX512VL__ but NEVER __AVX512VNNI__, even though
 * it happily emits vpdpbusd (verified by dumping the asm). The consequence was
 * that PICOLM_VNNI never fired on MSVC, so every VNNI kernel compiled out and
 * a VNNI-capable CPU silently ran the AVX2 fallbacks:
 *     sgemm_iq6_k_q8_k_r8t       instead of ..._r8v
 *     sgemm_q8_0_r8_avx2t_q8_0   instead of ..._vnni_q8_0
 * tools/msvc_build.bat now passes -D PICOLM_VNNI=1, but only after its CPU
 * probe reports the VNNI feature bit AND ZMM state is enabled, so no build can
 * execute vpdpbusd on a Skylake-X or a VM that lacks it. Keep that gating if
 * this flag is ever set by hand; PICOLM_VNNI is intentionally not part of the
 * /arch consistency guard below because it is a CPU property, not an /arch one.
 *
 * The ~32 VNNI kernels in sgemm_*.c / model_attention.c are guarded by
 * `#if defined(__AVX512VNNI__)` directly rather than by PICOLM_VNNI (they
 * predate this note, and gcc/clang do define the macro). So opting in has to
 * define the MACRO, not just PICOLM_VNNI -- exactly the same shim that is
 * already here for __FMA__ and __F16C__, which MSVC also enables without
 * defining. Doing it in one place beats editing every kernel guard.
 * ------------------------------------------------------------------------- */
#if defined(_MSC_VER) && !defined(__CUDACC__) && !defined(__HIP_DEVICE_COMPILE__)
#  if defined(__AVX2__) || defined(__AVX512F__)
#    ifndef __FMA__
#      define __FMA__ 1
#    endif
#    ifndef __F16C__
#      define __F16C__ 1
#    endif
#  endif
#endif

/* AVX-512 VNNI: promoted from the build's opt-in to the compiler macro the
 * kernels actually test. See the long note above for why MSVC cannot define
 * this itself and why it must only ever come from tools/msvc_build.bat's CPU
 * probe (or a deliberate -D on a CPU known to have VNNI).
 *
 * AVX512VL is required by most of those kernels and MSVC does define it for
 * /arch:AVX512, but define it here too so a hand-set -D does not half-compile
 * the VNNI path. Guarded to x86 MSVC builds only. */
#if defined(PICOLM_VNNI) && !defined(__CUDACC__) && !defined(__HIP_DEVICE_COMPILE__)
#  if !defined(__AVX512VNNI__)
#    define __AVX512VNNI__ 1
#  endif
#  if defined(_MSC_VER) && !defined(__AVX512VL__)
#    define __AVX512VL__ 1
#  endif
#endif

/* --- x86 SIMD: detect highest level, then propagate downward --- */
/* Never enable x86 SIMD intrinsics in CUDA/HIP device code */
#if !defined(__CUDACC__) && !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_VERSION__)
/* MSVC: /arch:AVX512 defines __AVX512F__ on VS2019+ but also __AVX512__ on some versions */
#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(__AVX512__))
#  define PICOLM_AVX512 1
/* VNNI with 256-bit (VL) enables dpbssd/dpbusd int8 MAC instructions */
#  if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#    define PICOLM_VNNI 1
#  endif
#  define PICOLM_AVX2   1
#  define PICOLM_AVX    1
#  define PICOLM_SSE3   1
#  define PICOLM_SSE2   1
/* MSVC: /arch:AVX2 defines __AVX2__ */
#elif defined(__AVX2__)
#  define PICOLM_AVX2 1
#  define PICOLM_AVX  1
#  define PICOLM_SSE3 1
#  define PICOLM_SSE2 1
/* AVX implies SSE3 + SSE2 */
#elif defined(__AVX__)
#  define PICOLM_AVX  1
#  define PICOLM_SSE3 1
#  define PICOLM_SSE2 1
/* SSE3 implies SSE2 */
#elif defined(__SSE3__)
#  if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 2)
#    define PICOLM_SSE3 1
#  endif
#  ifdef __SSSE3__
#    define PICOLM_SSSE3 1
#  endif
#  define PICOLM_SSE2 1
/* SSE2 baseline (also the default for all x86-64 targets) */
#elif defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
#  define PICOLM_SSE2 1
#endif
#endif /* !__CUDA__ && !__HIP__ */

/* ---- /arch consistency guard ----------------------------------------------
 *
 * PICOLM_* can be set two ways: derived from the compiler's predefined macros
 * above, or force-defined on the command line (-D PICOLM_AVX2=1). Force-defining
 * is what the Windows targets do, and if it disagrees with what the compiler was
 * actually told to generate (/arch:, -march=) the result is NOT a slowdown but
 * wrong code: kernels gated on the macro get compiled with intrinsics the
 * compiler will not emit, and others silently drop out, changing which code
 * path a data layout is paired with. That is how the MSVC/AVX2 V-tile transpose
 * shipped.
 *
 * So: refuse to build on a mismatch. #error is C89 and works on every compiler
 * we target (MSVC, gcc 3.x/4.x, DJGPP, Alpha cc, PPC gcc 4.0) -- unlike
 * _Static_assert, which is C11.  These blocks are all no-ops on non-x86 targets
 * (DOS/DJGPP sets PICOLM_DOS, PPC only sets PICOLM_ALTIVEC, Alpha sets none).
 *
 * Only checked when the compiler can tell us: MSVC's predefined macros are the
 * ones that lie (it defines __AVX2__ but not __FMA__/__F16C__; see the shim at
 * the top of this file), so a command-line -D PICOLM_AVX2 without a matching
 * /arch:AVX2 is always a mistake.
 * ------------------------------------------------------------------------- */
#if defined(_MSC_VER)
#  if defined(PICOLM_AVX512) && !defined(__AVX512F__)
#    error "PICOLM_AVX512 is defined but cl.exe was not given /arch:AVX512"
#  endif
#  if defined(PICOLM_AVX2) && !defined(__AVX2__)
#    error "PICOLM_AVX2 is defined but cl.exe was not given /arch:AVX2 (or /arch:AVX512)"
#  endif
#  if defined(PICOLM_AVX) && !defined(__AVX__)
#    error "PICOLM_AVX is defined but cl.exe was not given /arch:AVX or higher"
#  endif
#endif
/* Same idea for GCC/Clang, but only when the user clearly selected a -march
 * tier: __AVX2__ absent while PICOLM_AVX2 was force-defined. Guarded on
 * __GNUC__ so DOS/DJGPP (which defines __GNUC__ but has no AVX at all) is
 * excluded via PICOLM_DOS. */
#if defined(__GNUC__) && !defined(PICOLM_DOS) && !defined(__CUDACC__) && \
    !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_VERSION__)
#  if defined(PICOLM_AVX512) && !defined(__AVX512F__)
#    error "PICOLM_AVX512 is defined but the compiler was not given -mavx512f"
#  endif
#  if defined(PICOLM_AVX2) && !defined(__AVX2__)
#    error "PICOLM_AVX2 is defined but the compiler was not given -mavx2"
#  endif
#  if defined(PICOLM_FMA) && !defined(__FMA__)
#    error "PICOLM_FMA is defined but the compiler was not given -mfma"
#  endif
#endif

/* FMA3 (FMA) detection -- independent of AVX level.
 * AMD Piledriver (FX-8350 etc.) has AVX+FMA but no AVX2.
 * Intel Haswell+ has AVX2+FMA.  Sandy/Ivy Bridge have AVX but no FMA. */
#if defined(__FMA__)
#  define PICOLM_FMA 1
#endif

/* PPC Altivec (VMX) - separate from x86 chain */
#if defined(__ALTIVEC__)
#  define PICOLM_ALTIVEC 1
#  include <altivec.h>
#  undef bool
#  undef pixel
#  undef vec_add
#endif

/* Compile-time warning if no SIMD path was detected.
 * Falling back to scalar code is correct but dramatically slower.
 * This typically means -march=native was not used, or the compiler
 * did not define the expected architecture macros. */
#if !defined(PICOLM_AVX512) && !defined(PICOLM_AVX2) && !defined(PICOLM_AVX) && \
    !defined(PICOLM_SSE3) && !defined(PICOLM_SSSE3) && !defined(PICOLM_SSE2) && \
    !defined(PICOLM_NEON) && !defined(PICOLM_ALTIVEC)
#ifdef _MSC_VER
#  pragma message("PICOLM: no SIMD detected (AVX/AVX2/AVX512/SSE/NEON/Altivec). Scalar fallback active. Use -march=native for best performance.")
#else
#  warning "PICOLM: no SIMD detected (AVX/AVX2/AVX512/SSE/NEON/Altivec). Scalar fallback active. Use -march=native for best performance."
#endif
#endif

/* Include x86 SIMD header once for any x86 SIMD level.
 * <immintrin.h> is the modern umbrella header (GCC 4.7+).
 * Older compilers (e.g. GCC 4.2 on Mac OS X 10.6) need individual headers. */
#ifdef PICOLM_SSE2
#  ifdef _MSC_VER
#    include <intrin.h>  /* MSVC: provides __m128, __m256, __m512 built-in types + intrinsics */
#    include <immintrin.h>
#  elif __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 7)
#    include <xmmintrin.h>  /* SSE/SSE2 */
#    include <emmintrin.h>  /* SSE3 */
#    ifdef __SSSE3__
#      include <tmmintrin.h>  /* SSSE3: _mm_maddubs_epi16, _mm_sign_epi8 */
#    endif
#    ifdef __SSE4_1__
#      include <smmintrin.h>  /* SSE4.1: _mm_blendv_epi8, _mm_blendv_ps */
#    endif
#  else
#    include <immintrin.h>
#  endif
static inline float hsum_sse(__m128 v) {
    __m128 shuf = _mm_movehl_ps(v, v);
    __m128 sum  = _mm_add_ps(v, shuf);
    shuf = _mm_shuffle_ps(sum, sum, 1);
    sum  = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}
#endif

#ifdef PICOLM_AVX
static inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return hsum_sse(_mm_add_ps(lo, hi));
}

/* Horizontal sum of 8 int32_t in a 256-bit register */
static inline int hsum_i32_8(__m256i a) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i sum64 = _mm_add_epi32(hi64, sum128);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

/* Convert 8 FP16 to 8 FP32 in AVX register */
static inline __m256 fp16x8_to_fp32_inline(const uint16_t *p) {
#ifdef __F16C__
    __m128 lo = _mm_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
    __m128 hi = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(p + 4)));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
#else
    /* No F16C: use lookup table for faster FP16->FP32 conversion */
    return _mm256_set_ps(
        fp16_to_fp32_lookup(p[7]), fp16_to_fp32_lookup(p[6]), fp16_to_fp32_lookup(p[5]), fp16_to_fp32_lookup(p[4]),
        fp16_to_fp32_lookup(p[3]), fp16_to_fp32_lookup(p[2]), fp16_to_fp32_lookup(p[1]), fp16_to_fp32_lookup(p[0]));
#endif
}
#endif

/* --- PPC Altivec SIMD helpers --- */
#ifdef PICOLM_ALTIVEC
static inline float hsum_altivec(vector float v) {
    static char __hsbuf[128];
    unsigned long hba = (unsigned long)__hsbuf + 63;
    hba = hba / 64 * 64;
    ((float*)(void*)hba)[0] = 1.0f;
    vector float one = vec_splat(vec_ld(0, (float*)hba), 0);
    vector float h1 = vec_sld(v, v, 8);
    vector float h2 = vec_madd(h1, one, v);
    vector float h3 = vec_sld(h2, h2, 12);
    vector float h4 = vec_madd(h3, one, h2);
    float result;
    vec_st(h4, 0, &result);
    return result;
}
#endif

/* --- ARM NEON SIMD helpers --- */
#ifdef PICOLM_NEON
/* arm_neon.h already included above */
static inline float hsum_neon(float32x4_t v) {
    float32x2_t r = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}

static inline float32x4_t fp16x4_to_fp32_inline(const uint16_t *p) {
#if defined(__fp16)
    __fp16 h0, h1, h2, h3;
    memcpy(&h0, &p[0], 2); memcpy(&h1, &p[1], 2);
    memcpy(&h2, &p[2], 2); memcpy(&h3, &p[3], 2);
    return vsetq_lane_f32((float)h3,
           vsetq_lane_f32((float)h2,
           vsetq_lane_f32((float)h1,
           vdupq_n_f32((float)h0), 1), 2), 3);
#else
    /* 32-bit ARM lacks __fp16 type; use scalar lookup table. */
    return vsetq_lane_f32(fp16_to_fp32_lookup(p[3]),
           vsetq_lane_f32(fp16_to_fp32_lookup(p[2]),
           vsetq_lane_f32(fp16_to_fp32_lookup(p[1]),
           vdupq_n_f32(fp16_to_fp32_lookup(p[0])), 1), 2), 3);
#endif
}
#endif

#ifdef PICOLM_SSE2
/* Convert 4 FP16 to 4 FP32 in SSE register */
static inline __m128 fp16x4_to_fp32_inline(const uint16_t *p) {
#ifdef __F16C__
    return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)p));
#else
    /* No F16C: use lookup table */
    return _mm_set_ps(
        fp16_to_fp32_lookup(p[3]), fp16_to_fp32_lookup(p[2]), fp16_to_fp32_lookup(p[1]), fp16_to_fp32_lookup(p[0]));
#endif
}
#endif

/* AVX-512 specific helpers */
#ifdef PICOLM_AVX512
/* AVX-512 signed int8 x int8 dot product, 64 pairs per call (2 q8_0 blocks).
 * Used by vec_dot_q8_0_q8_0_deltas, vec_dot_q8_0_q8_0_deltas_batch4, and
 * the block-interleaved MoE GEMM kernels in tensor.c. */
static inline __m512i mul_sum_i8_pairs_avx512(const __m512i x, const __m512i y) {
    __m512i ax = _mm512_abs_epi8(x);
    __mmask64 neg_mask = _mm512_movepi8_mask(x);
    __m512i neg_y = _mm512_sub_epi8(_mm512_setzero_si512(), y);
    __m512i sy = _mm512_mask_blend_epi8(neg_mask, y, neg_y);
#if defined(__AVX512VNNI__)
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), ax, sy);
#else
    __m512i dot = _mm512_maddubs_epi16(ax, sy);
    __m512i ones = _mm512_set1_epi16(1);
    return _mm512_madd_epi16(ones, dot);
#endif
}
/* Convert 16 FP16 to 16 FP32 in AVX-512 register */
static inline __m512 fp16x16_to_fp32_inline(const uint16_t *p) {
#ifdef __F16C__
    __m128 a = _mm_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
    __m128 b = _mm_cvtph_ps(_mm_loadu_si128((const __m128i *)(p + 4)));
    __m128 c = _mm_cvtph_ps(_mm_loadu_si128((const __m128i *)(p + 8)));
    __m128 d = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(p + 12)));
    __m512 r = _mm512_castps128_ps512(a);
    r = _mm512_insertf32x4(r, b, 1);
    r = _mm512_insertf32x4(r, c, 2);
    r = _mm512_insertf32x4(r, d, 3);
    return r;
#else
    return _mm512_set_ps(
        fp16_to_fp32_lookup(p[15]), fp16_to_fp32_lookup(p[14]), fp16_to_fp32_lookup(p[13]), fp16_to_fp32_lookup(p[12]),
        fp16_to_fp32_lookup(p[11]), fp16_to_fp32_lookup(p[10]), fp16_to_fp32_lookup(p[9]), fp16_to_fp32_lookup(p[8]),
        fp16_to_fp32_lookup(p[7]), fp16_to_fp32_lookup(p[6]), fp16_to_fp32_lookup(p[5]), fp16_to_fp32_lookup(p[4]),
        fp16_to_fp32_lookup(p[3]), fp16_to_fp32_lookup(p[2]), fp16_to_fp32_lookup(p[1]), fp16_to_fp32_lookup(p[0]));
#endif
}

/* Vectorised expf for 16 float lanes (AVX-512).
 * Cephes range reduction + degree-6 polynomial, scaled with _mm512_scalef_ps.
 * Overflow -> +inf, underflow -> 0 like libm.  Max error ~2 ULP vs glibc expf.
 * Used by flash-style PFA attention (glm5, deepseek2) and KDA decay/gate. */
static inline __m512 picolm_expf16_avx512(__m512 x)
{
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-104.0f)), _mm512_set1_ps(89.0f));
    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 c1 = _mm512_set1_ps(0.693359375f), c2 = _mm512_set1_ps(-2.12194440e-4f);
    const __m512 fx = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e),
                                           _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(fx, c1, x);
    r = _mm512_fnmadd_ps(fx, c2, r);
    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.0000001201E-1f));
    const __m512 r2 = _mm512_mul_ps(r, r);
    y = _mm512_fmadd_ps(y, r2, _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
    return _mm512_scalef_ps(y, fx);
}
#endif

/* GGUF tensor data types */
typedef enum {
    GGUF_TYPE_F32   = 0,
    GGUF_TYPE_F16   = 1,
    GGUF_TYPE_Q4_0  = 2,
    GGUF_TYPE_Q4_1  = 3,
    GGUF_TYPE_Q5_0  = 6,
    GGUF_TYPE_Q5_1  = 7,
    GGUF_TYPE_Q8_0  = 8,
    GGUF_TYPE_Q8_1  = 9,
    GGUF_TYPE_Q5_K  = 13,
    GGUF_TYPE_Q2_K  = 10,
    GGUF_TYPE_Q3_K  = 11,
    GGUF_TYPE_Q4_K  = 12,
    /* TODO(type-ids): GGUF_TYPE_Q8_K = 29 is a PicoLM-internal ACTIVATION tag,
     * not a real on-disk GGUF/GGML tensor type id. It works, and nothing reads
     * a file and dispatches on it, so it is left alone deliberately -- but it
     * is wrong in two ways that will bite anyone extending the type table:
     *
     *   1. The real id for on-disk Q8_K is GGML_TYPE_Q8_K = 15 (see
     *      ggml/include/ggml.h in ik_llama.cpp / llama.cpp). 29 is
     *      GGML_TYPE_IQ1_M. So a GGUF whose tensors are genuinely IQ1_M puts
     *      29 in tinfos[i].type, which lands on THIS table entry instead of an
     *      IQ1_M one. Nothing in-tree produces such a file, but the failure
     *      would be silent (Q8_K-shaped reads of IQ1_M data), not an error.
     *
     *   2. Even using 15 here would not be a fix: llama.cpp's block_q8_K is
     *      596 bytes (float d; float sum; int8 qs[256]; int16 bsums[16]),
     *      while PicoLM's block_q8_K below is 292 bytes -- it drops `sum`. The
     *      two layouts are not interchangeable, so PicoLM cannot read real
     *      Q8_K tensor data until `sum` is added (which shifts every qs/bsums
     *      offset and touches all Q8_K kernels).
     *
     * If that ever needs fixing, the clean form is to move internal tags OUT of
     * the GGUF id space entirely (e.g. rename to PICOLM_ACT_Q8_K with a value
     * above 399, the highest id in ik_llama's ggml_type), free 29 for IQ1_M,
     * and add a static assert that keeps future internal tags above the real
     * range. Note also the adjacent trap documented on GGUF_TYPE_IQ4_NL_R4
     * below: `general.file_type` (LLAMA_FTYPE_MOSTLY_*) is a THIRD, unrelated
     * enum, and it is the usual source of confusion here. */
    GGUF_TYPE_Q8_K = 29,  /* internal: Q8_K intermediate quantization (see TODO above) */
    GGUF_TYPE_Q6_K       = 14,
    GGUF_TYPE_Q4_0_4_4   = 31,  /* 4-row interleaved Q4_0, blocklen=4 (pre-repacked) */
    GGUF_TYPE_Q4_0_4_8   = 32,  /* 4-row interleaved Q4_0, blocklen=8 (pre-repacked, I8MM target) */
    GGUF_TYPE_Q4_0_8_8   = 33,  /* 8-row interleaved Q4_0 (pre-repacked, AVX2) */
    GGUF_TYPE_Q4I_0_8_8  = 34,  /* 8-row pre-dequantized int8 Q4_0 (dpbusd lane order, AVX-512) */
    GGUF_TYPE_BF16      = 30,  /* Brain Float 16 (GGUF type 30) */
    GGUF_TYPE_IQ4_NL    = 20,  /* Non-linear 4-bit quant (LUT-based, same layout as Q4_0) */
    /* 4-row interleaved IQ4_NL.
     *
     * 220 is the per-tensor type id (GGML_TYPE_IQ4_NL_R4 in ggml.h) and is what
     * appears in every tensor header. Do not confuse it with the *file-level*
     * tag `general.file_type`, which for an IQ4_NL_R4 model is a different
     * number entirely (LLAMA_FTYPE_MOSTLY_IQ4_NL_R4 = 225 in include/llama.h).
     * We only ever read per-tensor ids, so only 220 matters here. ik_llama's
     * `llama-quantize --help` prints the file-type number, which is why it
     * looks like a competing type id; it is not one. */
    GGUF_TYPE_IQ4_NL_R4 = 220, /* 4-row interleaved IQ4_NL (repacked, AVX-512/AVX2 target) */
    GGUF_TYPE_IQ4_XS    = 23,  /* Non-linear 4-bit quant with per-subblock scales (K-style, GGUF type 23) */
    GGUF_TYPE_IQ2_K     = 137, /* Plain IQ2_K (non-interleaved, GGUF type 137) */
    GGUF_TYPE_IQ3_K     = 138, /* Plain IQ3_K (non-interleaved, GGUF type 138) */
    GGUF_TYPE_IQ4_K     = 139, /* Plain IQ4_K (non-interleaved, GGUF type 139) */
    GGUF_TYPE_IQ6_K     = 141, /* Plain IQ6_K (non-interleaved, GGUF type 141) */
    GGUF_TYPE_IQ2_K_R4  = 337, /* 4-row interleaved IQ2_K (repacked, AVX-512/AVX2 target) */
    GGUF_TYPE_IQ3_K_R4  = 338, /* 4-row interleaved IQ3_K (repacked, AVX-512/AVX2 target) */
    GGUF_TYPE_IQ4_K_R4  = 339, /* 4-row interleaved IQ4_K (repacked, AVX-512/AVX2 target) */
    GGUF_TYPE_Q1_0       = 41,  /* 1-bit sign + scale, 128 values/block */
    GGUF_TYPE_Q2_0       = 42,  /* 2-bit values + scale, 128 values/block */
    GGUF_TYPE_Q6_0       = 133, /* 6-bit values + FP16 scale, 32 values/block (legacy GGML) */
    GGUF_TYPE_Q4_0_R8    = 202, /* 8-row interleaved Q4_0 (GGUF type 202, llama.cpp ik branch) */
    /* TODO(type-ids): 203 is not ik_llama's id for Q8_0_R8 -- ggml.h says
     * GGML_TYPE_Q8_0_R8 = 208 (203 is unused there). model_gguf.c papers over
     * this by normalising an incoming 208 to this constant, so current GGUFs
     * load correctly; the constant itself is simply misnumbered. Changing it
     * to 208 would be a no-op for every model in use (208 already normalises
     * to it) but could change behaviour for any file that actually carries 203.
     * Left as-is: it works today. Do not "fix" this without checking
     * model_gguf.c's 208 normalisation at the same time. */
    GGUF_TYPE_Q8_0_R8    = 203, /* 8-row interleaved Q8_0 (TODO: ik says 208) */
    GGUF_TYPE_Q6_K_R4    = 214, /* 4-row interleaved Q6_K (GGUF type 214, llama.cpp ik branch) */
    GGUF_TYPE_Q8_K_R8    = 399, /* 8-row interleaved Q8_K (GGUF type 399, llama.cpp ik branch) */
    GGUF_TYPE_Q4_K_R4    = 212, /* 4-row interleaved Q4_K (GGUF type 212, llama.cpp ik branch) */
    GGUF_TYPE_Q5_K_R4    = 213, /* 4-row interleaved Q5_K (GGUF type 213, llama.cpp ik branch) */
    GGUF_TYPE_Q2_K_R4    = 210, /* 4-row interleaved Q2_K (GGUF type 210, llama.cpp ik branch) */
    GGUF_TYPE_IQ3_XXS    = 18,  /* 3-bit "almost true" quant, 16-bit block scale (ggml type 18) */
    GGUF_TYPE_IQ3_XXS_R4 = 218, /* 4-row interleaved IQ3_XXS (ik_llama GGML type 218) */
    GGUF_TYPE_IQ4_XS_R8  = 223, /* 8-row interleaved IQ4_XS (ik_llama GGML type 223) */
    /* Remaining standard GGUF IQ grid quants. Ids are GGML_TYPE_* from
     * ik_llama.cpp / llama.cpp ggml.h, i.e. the 4-byte per-tensor type field --
     * NOT `general.file_type` (LLAMA_FTYPE_MOSTLY_*), which is a separate enum
     * and a frequent source of confusion here. Block sizes: 66/74/50/110/82/56. */
    GGUF_TYPE_IQ2_XXS    = 16,  /* 2-bit grid, 8-bit scales */
    GGUF_TYPE_IQ2_XS     = 17,  /* 2-bit grid, per-32 scales */
    GGUF_TYPE_IQ1_S      = 19,  /* 1-bit sign grid + delta */
    GGUF_TYPE_IQ3_S      = 21,  /* 3-bit grid + signs */
    GGUF_TYPE_IQ2_S      = 22,  /* 2-bit grid + signs */
    GGUF_TYPE_IQ1_M      = 29,  /* 1-bit, multi-block scale (see TODO on
                                 * GGUF_TYPE_Q8_K above: that internal marker
                                 * currently occupies this id) */
} gguf_type_t;

/* One past the highest type ID, for table sizing */
#define GGUF_TYPE_COUNT 400

/* Activation format for a given weight quantization type.
 * Single source of truth: every matmul dispatch path uses this to decide
 * which quantization function to call on the activation buffer.
 *
 * ACT_FMT_F32   - no quantization, activation stays float32 (vec_dot fallback)
 * ACT_FMT_Q8_0  - quantize to block_q8_0 (32 values/block, uint16_t d)
 * ACT_FMT_Q8_K  - quantize to block_q8_K (256 values/block, float d)
 */
typedef enum {
    ACT_FMT_F32  = 0,
    ACT_FMT_Q8_0 = 1,
    ACT_FMT_Q8_K = 2,
    ACT_FMT_Q8_2 = 3,  /* block_q8_2: Q8_0 + int16 sum (Q4_0_R8) */
} act_fmt_t;

/* Return the activation format for a weight type.
 * Default is Q8_K (K-quant family uses per-subblock scales).
 * Q8_0 family uses the simpler block_q8_0 format. */
static inline act_fmt_t qtype_act_format(gguf_type_t qtype) {
    switch (qtype) {
    /* Q8_0 activation family: simple block_q8_0 (32 values, uint16_t d) */
    case GGUF_TYPE_Q8_0:
    case GGUF_TYPE_Q4_0:
    case GGUF_TYPE_Q5_0:
    case GGUF_TYPE_Q5_1:
    case GGUF_TYPE_Q1_0:
    case GGUF_TYPE_IQ4_NL:
    case GGUF_TYPE_IQ4_NL_R4:
    case GGUF_TYPE_Q4_0_4_4:
    case GGUF_TYPE_Q4_0_4_8:
    case GGUF_TYPE_Q4_0_8_8:
    case GGUF_TYPE_Q4_0_R8:
    case GGUF_TYPE_Q8_0_R8:
        return ACT_FMT_Q8_0;
    /* F32 activation family: no quantization (vec_dot fallback only) */
    case GGUF_TYPE_F32:
    case GGUF_TYPE_F16:
    case GGUF_TYPE_BF16:
        return ACT_FMT_F32;
    /* Everything else (K-quant family, IQ*, Q4I*, Q5*, Q6*, etc.) uses Q8_K */
    default:
        return ACT_FMT_Q8_K;
    }
}

/* Number of logical rows packed into one physical block group on disk.
 * 1 for ordinary (row-major) quant types; 4 for the "R4" row-interleaved
 * types, 8 for the "R8" ones. Interleaved formats store the first 4 (or 8)
 * logical rows of a tensor side by side inside each block, so the byte
 * offset of a logical row is NOT row * gguf_type_row_size():
 * the containing group starts at (row / rpb) * row_size * rpb, and the row
 * itself is row % rpb inside that group. Any code that walks head/row
 * pointers by a single row stride must go through
 * gguf_type_row_ptr() below. */
static inline int gguf_type_rows_per_block(gguf_type_t type) {
    switch (type) {
    case GGUF_TYPE_IQ2_K_R4:
    case GGUF_TYPE_IQ3_K_R4:
    case GGUF_TYPE_IQ4_K_R4:
    case GGUF_TYPE_IQ4_NL_R4:
    case GGUF_TYPE_IQ3_XXS_R4:
    case GGUF_TYPE_Q4_K_R4:
    case GGUF_TYPE_Q5_K_R4:
    case GGUF_TYPE_Q6_K_R4:
    case GGUF_TYPE_Q4_0_4_4:
    case GGUF_TYPE_Q4_0_4_8:
        return 4;
    case GGUF_TYPE_Q4_0_8_8:
    case GGUF_TYPE_Q4I_0_8_8:
    case GGUF_TYPE_Q4_0_R8:
    case GGUF_TYPE_Q8_0_R8:
    case GGUF_TYPE_IQ4_XS_R8:
        return 8;
    default:
        return 1;
    }
}

/* Start of the on-disk block group that contains logical row `row`.
 * For ordinary types this is just base + row * row_size. For R4/R8 it is the
 * group base; pair it with gguf_type_row_in_block() to pick the row inside.
 * `row_size` must be gguf_type_row_size(type, n). */
static inline const void *gguf_type_row_ptr(const void *base, gguf_type_t type,
                                            size_t row_size, int row) {
    int rpb = gguf_type_rows_per_block(type);
    if (rpb <= 1) return (const char *)base + (size_t)row * row_size;
    return (const char *)base + (size_t)(row / rpb) * row_size * (size_t)rpb;
}

/* Index of logical row `row` inside the group returned by gguf_type_row_ptr(). */
static inline int gguf_type_row_in_block(gguf_type_t type, int row) {
    int rpb = gguf_type_rows_per_block(type);
    return rpb <= 1 ? 0 : (row % rpb);
}

/* Packed struct attribute: empty on mainstream compilers where #pragma pack works,
 * but adds __attribute__((packed)) for LLVM-GCC 4.0.1 (iPhoneOS 1) which ignores
 * #pragma pack and pads structs to 4-byte boundaries.
 * Identified by: __llvm__ defined, __GNUC__==4, __GNUC_MINOR__==0 */
#if defined(__llvm__) && (__GNUC__ == 4) && (__GNUC_MINOR__ == 0)
#define PICOLM_PACKED_ATTR __attribute__((packed))
#else
#define PICOLM_PACKED_ATTR
#endif

/* Q4_K block: 256 weights in 144 bytes */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* super-block scale (FP16) */
    uint16_t dmin;       /* super-block min   (FP16) */
    uint8_t  scales[12]; /* packed 6-bit scales and mins for 8 sub-blocks */
    uint8_t  qs[128];    /* 4-bit quantized values (256 nibbles) */
} block_q4_K;            /* 144 bytes */
#pragma pack(pop)

/* Q8_K block: 256 weights, used for intermediate quantization in Q4_K/Q6_K matmul */
/* d: float, qs: 256 int8, bsums: 16 int16 (sum of quants in groups of 16) */
#ifndef QK_K
#define QK_K 256  /* Q8_K block length */
#endif
typedef struct PICOLM_PACKED_ATTR {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
} block_q8_K;            /* 4 + 256 + 32 = 292 bytes */

/* Q8_K_R8: 8-row interleaved Q8_K (GGUF type 399, llama.cpp ik branch)
 * Used for high-precision intermediate quantization in matmul.
 * Layout: 8 FP16 scales + 8x256 int8 values, interleaved per 32-byte chunks.
 * Each 32-byte chunk interleaves 4 consecutive bytes from each of 8 rows:
 *   qs[32*ib + 4*k + i]  ib=0..63, k=0..7 (rows), i=0..3 (values)
 *   bytes 0-3: row 0, bytes 4-7: row 1, ..., bytes 28-31: row 7
 * Total: 16 + 2048 = 2064 bytes per block. */
typedef struct {
    uint16_t  d[8];        // 8 FP16 scales = 16 bytes
    int8_t    qs[2048];    // 8 rows x 256 values, interleaved = 2048 bytes
} block_q8_k_r8;          // Total: 2064 bytes per block

/* Q4_K_R4 block: 4 rows of Q4_K repacked together for SIMD efficiency.
 * GGUF type 212. Size = 576 bytes = 4 * 144.
 * Each row covers QK_K=256 values. Total = 1024 values per block.
 *
 * Layout:
 *   d[8]:       FP16: d[0..3] = scales per row, d[4..7] = mins per row (16 bytes)
 *   scales_h[16]: 2 bits per scale (16 bytes, covers 64 scale entries)
 *   scales_l[32]: 4-bit magnitude scales (32 bytes, covers 64 scale entries)
 *   qs[512]:     4-bit quantized values interleaved across 4 rows (512 bytes)
 *
 * Scale encoding: 6-bit unsigned = (scales_l low nibble | scales_h 2-bit << 4)
 * Dequant: val = d[k] * scale * q - m[k] * min  (where k = row 0..3)
 * No LUT (unlike IQ4_K) -- raw 4-bit values 0..15.
 */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[8];         /* 8 FP16: d[0..3]=scales, d[4..7]=mins */
    uint8_t  scales_h[16]; /* 64 packed 2-bit scale extensions (16 bytes) */
    uint8_t  scales_l[32]; /* 64 packed 4-bit magnitude scales (32 bytes) */
    uint8_t  qs[512];      /* 1024 packed 4-bit values (512 bytes) */
} block_q4_k_r4;           /* 576 bytes = 4 * 144 */
#pragma pack(pop)

/* Q5_K_R4 block: 4 rows of Q5_K repacked together (ik_llama GGML type 213).
 * Size = 704 bytes = 4 * sizeof(block_q5_K). Layout mirrors block_q4_k_r4 but
 * with the 5th bit stored separately:
 *   d[8]:         FP16: d[0..3] = row scales, d[4..7] = row mins
 *   scales_h[16]: 2 high bits per 6-bit scale/min entry
 *   scales_l[32]: (scale & 0xf) | ((min & 0xf) << 4) per 32-value sub-block
 *   qh[128]:      8 high bits per (row, sub-block) byte-group, one per value
 *   qs[512]:      low 4 bits, 4 rows interleaved
 * From ik_llama repack_q5_k(); dequant = d[row]*scale*q - dmin[row]*min with
 * q = low nibble | (high bit << 4). */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[8];         /* 4 row scales, then 4 row mins */
    uint8_t  scales_h[16];
    uint8_t  scales_l[32];
    uint8_t  qh[128];
    uint8_t  qs[512];
} block_q5_k_r4;           /* 704 bytes = 4 * 176 */
#pragma pack(pop)

/* Q4_0_4_4 interleaved block: 4 rows of Q4_0 packed together for SIMD efficiency.
 * Layout: 4 FP16 deltas, then interleaved nibble-bytes from 4 standard Q4_0 blocks.
 * Nibbles are XOR'd with 0x88 (per byte) during repacking to convert from bias form
 * [0..15] to sign form [-8..7], eliminating the subtract-8 during dequantization.
 *
 * Interleaving pattern (blocklen=4):
 *   qs[0..3]   = row0 qs[0..3]   ^ 0x88
 *   qs[4..7]   = row1 qs[0..3]   ^ 0x88
 *   qs[8..11]  = row2 qs[0..3]   ^ 0x88
 *   qs[12..15] = row3 qs[0..3]   ^ 0x88
 *   qs[16..19] = row0 qs[4..7]   ^ 0x88
 *   ... (continues for all 16 nibble-bytes per row)
 *
 * Total: 4*2 + 4*16 = 72 bytes (same as 4 standard Q4_0 blocks)
 * Each block covers 4 rows x 32 values = 128 values. */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[4];      /* 4 FP16 deltas, one per row */
    uint8_t  qs[64];    /* interleaved nibble-bytes (4 rows x 16 bytes, XOR'd with 0x88) */
} block_q4_0x4;         /* 72 bytes */
#pragma pack(pop)

/* Q4_0_8x8 interleaved block: 8 rows of Q4_0 packed together for AVX2 SIMD.
 * Layout: 8 FP16 deltas, then interleaved nibble-bytes from 8 standard Q4_0 blocks.
 * Nibbles are XOR'd with 0x88 (per byte) during repacking to convert from bias form
 * [0..15] to sign form [-8..7], eliminating the subtract-8 during dequantization.
 *
 * Interleaving (blocklen=8): for each chunk k=0..3:
 *   qs[k*128 + 0..7]    = row0 qs[k*8..k*8+7] ^ 0x88
 *   qs[k*128 + 8..15]   = row1 qs[k*8..k*8+7] ^ 0x88
 *   ...
 *   qs[k*128 + 56..63]  = row7 qs[k*8..k*8+7] ^ 0x88
 *
 * Total: 8*2 + 8*16 = 144 bytes (same as 8 standard Q4_0 blocks)
 * Each block covers 8 rows x 32 values = 256 values.
 * Used by AVX2 kernel that processes 8 output rows simultaneously. */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[8];      /* 8 FP16 deltas, one per row */
    uint8_t  qs[128];   /* interleaved nibble-bytes (8 rows x 16 bytes, XOR'd with 0x88) */
} block_q4_0x8;         /* 144 bytes */
#pragma pack(pop)

/* Q4I_0_8_8 (GGUF type 34): pre-dequantized int8 format for AVX-512 dpbusd.
 * Layout (272 bytes): d[8] (16B FP16 scales) + qs[256] (8 rows x 32 signed int8).
 * qs arranged in dpbusd lane order to eliminate blend+permute and LUT shuffles. */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[8];      /* 8 FP16 deltas, one per row */
    int8_t   qs[256];   /* 8 rows x 32 signed int8, dpbusd lane order */
} block_q4i_0x8;        /* 272 bytes */
#pragma pack(pop)

/* Q3_K block: 256 weights in 110 bytes, layout: hmask[32] + qs[64] + scales[12] + d[2] */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint8_t  hmask[32];  /* high bit mask */
    uint8_t  qs[64];     /* 2-bit low quants */
    uint8_t  scales[12]; /* packed 6-bit scales */
    uint16_t d;          /* super-block scale (FP16) */
} block_q3_K;            /* 110 bytes */
#pragma pack(pop)

/* Q2_K block: 256 weights in 84 bytes */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint8_t  scales[16]; /* packed scales and mins (4-bit each) */
    uint8_t  qs[64];     /* 2-bit quantized values */
    uint16_t d;          /* super-block scale (FP16) */
    uint16_t dmin;       /* super-block min   (FP16) */
} block_q2_K;            /* 84 bytes */
#pragma pack(pop)

/* Q8_0 block: 32 weights */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* scale (FP16) */
    int8_t   qs[32];     /* 8-bit quantized values */
} block_q8_0;            /* 34 bytes */

/* TurboQuant TQ3 block: 32 values in 14 bytes (3.5 bits/value)
 * Pipeline: RMS normalize -> WHT forward (sign-randomized) -> Lloyd-Max 3-bit
 * Dequant: unpack indices -> codebook lookup -> WHT inverse -> scale restore
 *
 * Layout: 2 bytes fp16 RMS scale + 12 bytes packed 3-bit indices.
 * 8-element Lloyd-Max codebook for N(0,1): {-2.15, -1.34, -0.76, -0.25, 0.25, 0.76, 1.34, 2.15} */
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* RMS scale (FP16) */
    uint8_t  qs[12];     /* 32 values x 3-bit packed = 12 bytes */
} block_tq3;             /* 14 bytes */

/* TurboQuant TQ4 block: 32 values in 18 bytes (4.5 bits/value)
 * Same WHT rotation as TQ3, but 16-entry Lloyd-Max codebook + nibble packing.
 * D_mse ~0.0095 (vs TQ3 D_mse ~0.032) -- 3.4x better accuracy.
 * Layout: 2 bytes fp16 RMS scale + 16 bytes packed 4-bit indices. */
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* RMS scale (FP16) */
    uint8_t  qs[16];     /* 32 values x 4-bit packed = 16 bytes */
} block_tq4;             /* 18 bytes */

/* TQ3 Lloyd-Max codebook centroids for N(0,1) */
#define TQ3_CODEBOOK_8 { -2.1519f, -1.3439f, -0.7560f, -0.2451f, \
                          0.2451f,  0.7560f,  1.3439f,  2.1519f }

/* TQ3 decision boundaries (midpoints between consecutive centroids) */
#define TQ3_BOUNDARIES_7 { -1.7479f, -1.0500f, -0.5005f, 0.0f, 0.5005f, 1.0500f, 1.7479f }

/* TQ3 deterministic sign pattern (golden ratio hash) */
#define TQ3_SIGNS_32 { \
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, \
   -1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f, \
   -1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f, \
   -1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f }

/* TQ3 block size */
#define TQ3_BLOCK_SIZE 32

/* TQ3 data exposed for use in model.c attention path */
extern const float tq3_codebook[8];
extern const float tq3_boundaries[7];
extern const float tq3_signs[32];

/* TQ4 block size (same as TQ3) */
#define TQ4_BLOCK_SIZE 32

/* TQ4 Lloyd-Max 16-entry codebook centroids for N(0,1) */
/* D_mse ~0.0095 vs TQ3 ~0.032 */
extern const float tq4_codebook[16];
extern const float tq4_boundaries[15];

static inline void tq4_unpack_4bit_8(uint8_t *dst, const uint8_t *src) {
    dst[0] = src[0] & 0xF;
    dst[1] = src[0] >> 4;
    dst[2] = src[1] & 0xF;
    dst[3] = src[1] >> 4;
    dst[4] = src[2] & 0xF;
    dst[5] = src[2] >> 4;
    dst[6] = src[3] & 0xF;
    dst[7] = src[3] >> 4;
}

static inline void tq3_unpack_3bit_8(uint8_t *dst, const uint8_t *src) {
    dst[0] = src[0] & 7;
    dst[1] = (src[0] >> 3) & 7;
    dst[2] = ((src[0] >> 6) | (src[1] << 2)) & 7;
    dst[3] = (src[1] >> 1) & 7;
    dst[4] = (src[1] >> 4) & 7;
    dst[5] = ((src[1] >> 7) | (src[2] << 1)) & 7;
    dst[6] = (src[2] >> 2) & 7;
    dst[7] = (src[2] >> 5) & 7;
}

#pragma pack(pop)

/* I8MM repacked activation for Q4_0_4x8 gemm
 * Each Q8_0 block (32 int8 values) is repacked into two int8x16 vectors:
 *   B0 = [act[0..7], act[16..23]]   -- low segments zipped with high segments
 *   B1 = [act[8..15], act[24..31]]
 * This layout matches smmla's 2x8 input format:
 *   smmla(A=[row_lo, row_hi], B=B0) gives [dot(lo,lo), dot(lo,hi), dot(hi,lo), dot(hi,hi)]
 *   where [0]+[3] = dot product of expanded nibbles with full activation.
 *
 * Storage: int8_t[32] per block (same as Q8_0 qs, but rearranged). */

/* Q5_K block: 256 weights in 176 bytes */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* super-block scale (FP16) */
    uint16_t dm;         /* super-block min   (FP16) */
    uint8_t  scales[12]; /* packed 6-bit scales+mins (get_scale_min_k4) */
    uint8_t  qh[32];     /* high bit (1 per quant) */
    uint8_t  qs[128];    /* low 4 bits (2 per byte) */
} block_q5_K;            /* 176 bytes */
#pragma pack(pop)

/* Q6_K block: 256 weights in 210 bytes */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint8_t  ql[128];    /* low 4 bits of quants */
    uint8_t  qh[64];     /* high 2 bits of quants */
    int8_t   scales[16]; /* 8-bit scales */
    uint16_t d;          /* super-block scale (FP16) */
} block_q6_K;            /* 210 bytes */
#pragma pack(pop)

/* Q6_K_R4 block: 4 interleaved Q6_K rows of 256 weights each, 840 bytes.
 * Matches llama.cpp ik branch's block_q6_k_r4 (ggml-common.h) field order
 * and byte layout exactly (see repack_q6_k()/dequantize_row_q6_k_r4()).
 * quant_size for this type is defined as sizeof(block_q6_K) (210, the
 * per-row-equivalent byte count), matching the GGUF_TYPE_Q4_0_R8 convention:
 * gguf_type_row_size() then naturally lines up row i's byte offset with the
 * base of block group i/4 whenever i is a multiple of 4. */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[4];       /* super-block scale (FP16), one per row */
    int8_t   scales[64]; /* 8-bit scales, interleaved: scales[8*ib+k+{0,4}] */
    uint8_t  qh[256];    /* high 2 bits of quants, interleaved across rows */
    uint8_t  ql[512];    /* low 4 bits of quants, interleaved across rows */
} block_q6_K_R4;         /* 840 bytes */
#pragma pack(pop)

/* Q4_0 block: 32 weights */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* scale (FP16) */
    uint8_t  qs[16];     /* 4-bit quantized values */
} block_q4_0;            /* 18 bytes */

/* IQ4_NL block: 32 weights (non-linear 4-bit quant, GGUF type 20)
 * Identical layout to Q4_0 (18 bytes), but dequant uses a 16-entry LUT
 * instead of linear (nibble-8) mapping.
 * LUT values: -127, -104, -83, -65, -49, -35, -22, -10,
 *              1,   13,   25,  38,  53,  69,  89, 113
 * Dequant: val = kvalues_iq4nl[nibble] * d
 * Provides better accuracy than Q4_0 at same 2.25 BPW for typical
 * LLM weight distributions (mass near zero, heavy tails). */
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* scale (FP16) */
    uint8_t  qs[16];     /* 4-bit quantized values (indices into LUT) */
} block_iq4_nl;           /* 18 bytes, identical to block_q4_0 */
#pragma pack(pop)
/* Compile-time check: block_iq4_nl must match block_q4_0 layout (18 bytes) */
#if !defined(_MSC_VER)
typedef char __compiletime_assert_iq4_nl_size[(sizeof(block_iq4_nl) == sizeof(block_q4_0)) ? 1 : -1] __attribute__((unused));
#endif
/* MSVC: checked at link time by sizeof consistency, no compile-time assert needed */

/* IQ4_NL dequantization lookup table (16 entries, int8) */
/* Derived from llama.cpp ggml-common.h GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16) */
extern const int8_t kvalues_iq4nl[16];

/* IQ4_NL_R4 block (GGUF type 220): 4-row interleaved IQ4_NL.
 * Identical layout to block_q4_0x4 (72 bytes = 4*18), but dequant uses
 * the non-linear IQ4_NL LUT instead of signed nibble subtraction.
 *
 * Layout:
 *   d[4]:  4 FP16 global scales, one per row
 *   qs[64]: 4 rows x 16 bytes, interleaved with stride 4:
 *     For block group ib, row r, byte offset j:
 *       qs[ib*64 + r*16 + j]  (r=0..3, j=0..15)
 *   Each row: 32 values per block group (QK4_NL=32)
 *   Dequant: val = kvalues_iq4nl[nibble] * d[row]
 *   No XOR with 0x88 (unlike Q4_0_4_4).
 */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[4];      /* 4 FP16 deltas, one per row */
    uint8_t  qs[64];    /* interleaved nibble-bytes (4 rows x 16 bytes) */
} block_iq4_nl_r4;      /* 72 bytes = 4 * 18 */
#pragma pack(pop)
#if !defined(_MSC_VER)
typedef char __compiletime_assert_iq4_nl_r4_size[(sizeof(block_iq4_nl_r4) == 72) ? 1 : -1] __attribute__((unused));
#endif

/* IQ4_XS block (GGUF type 23): 256 weights (non-linear 4-bit quant with per-subblock scales).
 * Layout (from ik_llama.cpp ggml-common.h):
 *   d:         FP16 global scale
 *   scales_h:  16-bit: high 2 bits for each of 8 sub-scales (bits 0-1, 2-3, ..., 14-15)
 *   scales_l[4]: low 4 bits for each of 8 sub-scales (packed 2 per byte)
 *   qs[128]:   4-bit quantized values (indices into kvalues_iq4nl LUT)
 *
 * Each sub-scale = (scales_l_nib | ((scales_h_nib) << 4)) - 32, where:
 *   scales_l_nib = (scales_l[ib/2] >> (4*(ib%2))) & 0xf
 *   scales_h_nib = (scales_h >> (2*ib)) & 3
 *
 * Dequant: val = d * scale[subblock] * kvalues_iq4nl[nibble]
 * Size: 2 + 2 + 4 + 128 = 136 bytes per block (QK_K=256 values). */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;             /* global scale (FP16) */
    uint16_t scales_h;      /* high 2 bits x 8 sub-scales (16 bits) */
    uint8_t  scales_l[4];   /* low 4 bits x 8 sub-scales (8 nibbles = 4 bytes) */
    uint8_t  qs[128];       /* 4-bit quantized values (256 nibbles = 128 bytes) */
} block_iq4_xs;            /* 136 bytes */
#pragma pack(pop)
#if !defined(_MSC_VER)
typedef char __compiletime_assert_iq4_xs_size[(sizeof(block_iq4_xs) == 136) ? 1 : -1] __attribute__((unused));
#endif

/* IQ2_K plain block (GGUF type 137): single-row 2-bit non-linear quant.
 * Size: 76 bytes per block (QK_K=256 values).
 *
 * Layout (from ik_llama.cpp ggml-common.h):
 *   d:      FP16 global scale (2 bytes)
 *   extra:  16 bits, 2 bits per 32-value subblock (LUT table select) (2 bytes)
 *   scales: 8 bytes, 4-bit signed scales (2 per byte, offset by -8)
 *   qs:     64 bytes, 2-bit values (4 per byte)
 */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;              /* FP16 global scale */
    uint16_t extra;          /* LUT table selection (2 bits per subblock) */
    uint8_t  scales[8];      /* 16 packed 4-bit signed scales (2 per byte) */
    uint8_t  qs[64];         /* 256 packed 2-bit values (4 per byte) */
} block_iq2_k;              /* 76 bytes */
#pragma pack(pop)

/* IQ3_K plain block (GGUF type 138): single-row 3-bit non-linear quant.
 * Size: 110 bytes per block (QK_K=256 values).
 *
 * Layout (from ik_llama.cpp ggml-common.h):
 *   d:        FP16 global scale (2 bytes)
 *   extra:    16 bits, 2 bits per 32-value subblock (LUT table select) (2 bytes)
 *   scales_h: 16 bits, sign bits for 8 scale pairs (1=pos, 0=neg) (2 bytes)
 *   scales_l: 8 bytes, 4-bit magnitude per scale (2 per byte)
 *   qs:       64 bytes, 2-bit low values (4 per byte)
 *   qh:       32 bytes, 1 high bit per value (8 per byte)
 */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;              /* FP16 global scale */
    uint16_t extra;          /* LUT table selection (2 bits per subblock) */
    uint16_t scales_h;       /* Sign bits (16 bits, 1 per scale) */
    uint8_t  scales_l[8];    /* 4-bit magnitude scales (2 per byte) */
    uint8_t  qs[64];         /* 2-bit low values (4 per byte) */
    uint8_t  qh[32];         /* 1 high bit per value (8 per byte) */
} block_iq3_k;              /* 110 bytes */
#pragma pack(pop)

/* IQ2_K_R4 block: 4 rows of IQ2_K repacked together for SIMD efficiency.
 * GGUF type 337. Size = 4 * sizeof(block_iq2_k) = 304 bytes.
 * Each row covers QK_K=256 values. Total = 1024 values per block.
 *
 * Layout:
 *   d[4]:     FP16 global scales, one per row (8 bytes)
 *   extra[8]: uint8_t LUT selection flags (8 bytes)
 *     extra[0..3] = low-half flags for rows 0..3 (1 bit per sub-block)
 *     extra[4..7] = high-half flags for rows 0..3 (1 bit per sub-block)
 *   scales[32]: 4-bit signed scales interleaved across 4 rows (32 bytes)
 *     16 scales per row (8 sub-blocks x 2 scales each), 64 total = 32 bytes
 *     Packing: scales[row*16+idx] stored 2-per-byte with row interleaving
 *   qs[256]:  2-bit quantized values interleaved across 4 rows (256 bytes)
 *     64 bytes per row * 4 rows = 256 bytes
 *     Layout: 4 rows' 2-bit values packed as 2 bits per row per byte
 */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[4];        /* 4 FP16 global scales */
    uint8_t  extra[8];    /* LUT selection: extra[0..3]=low-half rows, extra[4..7]=high-half rows */
    uint8_t  scales[32];  /* 64 packed 4-bit scales (16 per row x 4 rows) */
    uint8_t  qs[256];     /* 1024 packed 2-bit values (256 per row x 4 rows) */
} block_iq2_k_r4;         /* 304 bytes = 4 * 76 */
#pragma pack(pop)

/* IQ3_K_R4 block: 4 rows of IQ3_K repacked together for SIMD efficiency.
 * GGUF type 338. Size = 4 * sizeof(block_iq3_k) = 440 bytes.
 * Each row covers QK_K=256 values. Total = 1024 values per block.
 *
 * Layout:
 *   d[4]:       FP16 global scales, one per row (8 bytes)
 *   extra[8]:   uint8_t LUT selection flags (8 bytes)
 *     extra[0..3] = low-half flags for rows 0..3 (1 bit per sub-block)
 *     extra[4..7] = high-half flags for rows 0..3 (1 bit per sub-block)
 *   scales_h[8]:  packed sign bits for 64 scales (8 bytes)
 *     1 bit per scale, 64 scales total (16 per row x 4 rows)
 *   scales_l[32]: 4-bit magnitude scales interleaved (32 bytes)
 *     Each scale = (mag & 0xf) * 2 + 1 (odd values 1..31)
 *   qs[256]:   3-bit quantized values interleaved across 4 rows (256 bytes)
 *     64 bytes per row * 4 rows = 256 bytes
 *   qh[128]:   1 high bit per value, packed 8 per byte (128 bytes)
 *     16 bytes per row * 4 rows = 64 bytes... wait, 256/2 = 128
 */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[4];        /* 4 FP16 global scales */
    uint8_t  extra[8];    /* LUT selection: extra[0..3]=low-half rows, extra[4..7]=high-half rows */
    uint8_t  scales_h[8]; /* 64 sign bits packed (1 bit per scale) */
    uint8_t  scales_l[32];/* 64 packed 4-bit magnitude scales */
    uint8_t  qs[256];     /* 1024 packed 3-bit values */
    uint8_t  qh[128];     /* 1024 high bits (1 per value, 8 values per byte) */
} block_iq3_k_r4;         /* 440 bytes = 4 * 110 */

/* Q2_K_R4 block (GGUF type 210): 4 rows of Q2_K repacked together.
 * Size = 8*2 + QK_K/4 + QK_K = 16 + 64 + 256 = 336 bytes (4 * 84).
 *
 * Layout (from ik_llama.cpp ggml-common.h / repack_q2_k()):
 *   d[8]:         FP16: d[0..3] = the 4 rows' super-block scales,
 *                        d[4..7] = the 4 rows' dmin values
 *   scales[64]:   QK_K/16 = 16 sub-block (min<<4 | scale) bytes per row,
 *                 interleaved as scales[4*ib + row]
 *   qs[256]:      QK_K = 256 bytes = 1024 packed 2-bit values.
 *                 Repacked as, per 32-value sub-block ib (0..7):
 *                   qs[32*ib + 4*row + i +  0] packs L[32*ib + i + 0/4/8/12]
 *                   qs[32*ib + 4*row + i + 16] packs L[32*ib + i + 16/20/24/28]
 *                 i.e. the same 4-row-interleaved nibble grouping the other R4
 *                 K-quants use. NOTE the q2 values are 2-bit in the low two
 *                 bits of each byte slot (values 0..3), NOT nibbles.
 * Dequant: val = d[row]*scale*q - dmin[row]*min */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[8];        /* 4 FP16 scales then 4 FP16 mins */
    uint8_t  scales[64];  /* 16 sub-blocks * 4 rows, (min<<4 | scale) */
    uint8_t  qs[256];     /* 4 rows x 64 bytes of 2-bit quants, row-interleaved */
} block_q2_k_r4;          /* 336 bytes = 4 * 84 */
#pragma pack(pop)

/* IQ3_XXS block (ggml type 18): 3.0625 bpw, QK_K values in 98 bytes.
 *
 * Layout (from ik_llama.cpp ggml-common.h / dequantize_row_iq3_xxs):
 *   d:  FP16 super-block scale
 *   qs: 3*QK_K/8 = 96 bytes. First QK_K/4 = 64 bytes are grid indices,
 *       the remaining 32 bytes are scales_and_signs.
 * Each 32-value sub-block ib32 (0..7) uses:
 *   aux32 = scales_and_signs[4*ib32 .. +3]
 *   db    = d * (0.5f + (aux32 >> 28)) * 0.5f
 *   for l in 0..3: signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]
 *                  grid1 = iq3xxs_grid[qs[2*l+0]], grid2 = iq3xxs_grid[qs[2*l+1]]
 *                  8 values from two 4-entry grids, sign-flipped by `signs`
 *   then qs += 8. */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;
    uint8_t  qs[3 * 256 / 8];  /* 96 bytes: 64 index bytes + 32 scale/sign bytes */
} block_iq3_xxs;               /* 98 bytes */

/* ---- Block layouts (ggml-common.h, QK_K = 256) ----
 *
 * Same packing convention as every other block_* struct in quant.h:
 * #pragma pack(1) plus PICOLM_PACKED_ATTR, which adds __attribute__((packed))
 * for LLVM-GCC 4.0.1 (iPhoneOS 1), the one compiler that ignores #pragma pack.
 * Byte sizes are asserted at the end of this section. */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR { uint16_t d; uint16_t qs[QK_K/8]; } block_iq2_xxs;
/* 2 + 64 = 66 bytes */

typedef struct PICOLM_PACKED_ATTR { uint16_t d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; } block_iq2_xs;
/* 2 + 64 + 8 = 74 bytes */

typedef struct PICOLM_PACKED_ATTR { uint16_t d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; } block_iq1_s;
/* 2 + 32 + 16 = 50 bytes */

#define IQ3S_N_SCALE (QK_K/64)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
/* 2 + 64 + 8 + 32 + 4 = 110 bytes */

typedef struct PICOLM_PACKED_ATTR { uint16_t d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; } block_iq2_s;
/* 2 + 64 + 8 + 8 = 82 bytes */

typedef struct PICOLM_PACKED_ATTR {
    uint8_t qs[QK_K/8];      /* grid index, low 8 bits */
    uint8_t qh[QK_K/16];     /* grid index, high 3 bits + grid shift bit */
    uint8_t scales[QK_K/32]; /* 3-bit block scales */
} block_iq1_m;
/* 32 + 16 + 8 = 56 bytes */
#pragma pack(pop)

/* Layout guard: a wrong block size here means every tensor of that type is read
 * at the wrong stride, silently. C11 gets _Static_assert; older compilers
 * (DJGPP, gcc 3.3, MSVC pre-C11) get the negative-array-size trick, which every
 * one of them supports and which costs nothing at run time. */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(block_iq2_xxs) == 66,  "block_iq2_xxs must be 66 bytes");
_Static_assert(sizeof(block_iq2_xs)  == 74,  "block_iq2_xs must be 74 bytes");
_Static_assert(sizeof(block_iq1_s)   == 50,  "block_iq1_s must be 50 bytes");
_Static_assert(sizeof(block_iq3_s)   == 110, "block_iq3_s must be 110 bytes");
_Static_assert(sizeof(block_iq2_s)   == 82,  "block_iq2_s must be 82 bytes");
_Static_assert(sizeof(block_iq1_m)   == 56,  "block_iq1_m must be 56 bytes");
#else
typedef char picolm_iq_block_size_check[
    (sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 &&
     sizeof(block_iq1_s)   == 50 && sizeof(block_iq3_s)  == 110 &&
     sizeof(block_iq2_s)   == 82 && sizeof(block_iq1_m)  == 56) ? 1 : -1];
#endif

/* IQ3_XXS_R4 block (ik_llama GGML type 218): 4 IQ3_XXS rows interleaved.
 * Size = 4 * sizeof(block_iq3_xxs) = 392 bytes, i.e. 98 bytes per row of 256.
 *
 * Layout (from ik_llama.cpp repack_iq3_xxs()):
 *   d[4]:     the 4 rows' FP16 super-block scales
 *   qs[256]:  per 32-value sub-block ib (0..7), qs[32*ib + 8*k + i] holds
 *             row k's grid indices qs[8*ib + i] -- i.e. 8 index bytes per row
 *   sas[128]: per 256-value block, the 32-bit word at
 *             ((uint32_t*)sas)[8*ibl + 4*ib + k] belongs to row k of
 *             sub-block ib. Bit 0 of each byte is one bit of the 4-bit
 *             sub-block scale; the other 7 bits are that 8-value run's sign
 *             mask, stored scrambled and undone at decode time with
 *             aux &= 0xfefefefe; aux ^= aux >> 1
 *             See dequantize_row_iq3_xxs_r4_single() in quant.c. */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[4];      /* 4 FP16 row scales */
    uint8_t  sas[128];  /* 4 rows x 8 sub-blocks x 4 bytes: sign7<<1 | scale bit */
    uint8_t  qs[256];   /* 4 rows x 64 grid indices, row-interleaved per 8-value run */
} block_iq3_xxs_r4;     /* 392 bytes = 4 * 98 */
#pragma pack(pop)

/* IQ4_XS_R8 block (ik_llama GGML type 223): 8 IQ4_XS rows interleaved.
 * Size = 8 * sizeof(block_iq4_xs) = 1088 bytes, i.e. 136 bytes per row of 256.
 *
 * Layout (from ik_llama.cpp repack_iq4_xs()):
 *   d[8]:          the 8 rows' FP16 super-block scales
 *   scales_h[16]:  for value-index i = 8*ib + k, scales_h[i % 16] |= sh << 2*(i/16)
 *   scales_l[32]:  for value-index i = 8*ib + k, scales_l[i % 32] |= sl << 4*(i/32)
 *   qs[1024]:      per 32-value sub-block ib, 128 bytes =
 *                    qs[128*ib + 4*k + i + {0,32,64,96}] packs row k's nibbles
 *                  (low nibbles of qs[16*ib + i + {0,4}] / {8,12} into +0/+32,
 *                   high nibbles of the same into +64/+96) */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[8];        /* 8 FP16 row scales */
    uint8_t  scales_h[16];
    uint8_t  scales_l[32];
    uint8_t  qs[1024];
} block_iq4_xs_r8;        /* 1088 bytes = 8 * 136 */
#pragma pack(pop)
#pragma pack(pop)

/* IQ4_K plain block (GGUF type 139): single-row 4-bit non-linear quant.
 * Size: 144 bytes per block (QK_K=256 values).
 *
 * Layout (from ik_llama.cpp ggml-common.h):
 *   d:        FP16 global scale (2 bytes)
 *   extra:    16 bits, 2 bits per 32-value subblock (LUT table select) (2 bytes)
 *   scales_h: 4 bytes, 2 bits per scale (16 scales total, 2 per subblock)
 *   scales_l: 8 bytes, 4-bit magnitude per scale (2 per byte, 16 total)
 *   qs:       128 bytes, 4-bit values (2 per byte, 256 values)
 *
 * Scale encoding: 6-bit signed = (scales_l 4-bit | scales_h 2-bit) - 32
 * LUT: 32 entries (2 tables of 16), selected by extra bits
 * No qh field (unlike IQ3_K) -- all 4 bits in qs.
 */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;              /* FP16 global scale */
    uint16_t extra;          /* LUT table selection (2 bits per subblock) */
    uint8_t  scales_h[4];    /* 2 bits per scale (16 scales) */
    uint8_t  scales_l[8];    /* 4-bit magnitude scales (16 total) */
    uint8_t  qs[128];        /* 256 packed 4-bit values (2 per byte) */
} block_iq4_k;              /* 144 bytes */

/* IQ6_K block (GGUF type 141): single-row 6-bit non-linear quant.
 * Size: 212 bytes per block (QK_K=256 values). */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;              /* FP16 global scale */
    uint16_t extra;          /* LUT table select: 1 bit per 16-value group (4 per 64-value subblock, 4 subblocks = 16 bits) */
    int8_t   scales[16];     /* Signed per-subblock scales (4 per subblock) */
    uint8_t  qs[128];        /* 256 packed 4-bit low values (2 per byte) */
    uint8_t  qh[64];         /* 256 packed 2-bit high values (4 per byte) */
} block_iq6_k;              /* 212 bytes */
#pragma pack(pop)

/* IQ6_K non-linear LUT: 64 entries, uint8 offset by +128.
 * iq6nl_lut[q] - 128 gives the signed dequantized value for index q (0..63). */
extern const uint8_t iq6nl_lut[64];

/* IQ4_K_R4 block: 4 rows of IQ4_K repacked together for SIMD efficiency.
 * GGUF type 339. Size = 4 * sizeof(block_iq4_k) = 576 bytes.
 * Each row covers QK_K=256 values. Total = 1024 values per block.
 *
 * Layout:
 *   d[4]:       FP16 global scales, one per row (8 bytes)
 *   extra[8]:   uint8_t LUT selection flags (8 bytes)
 *     extra[0..3] = low-half flags for rows 0..3 (1 bit per sub-block)
 *     extra[4..7] = high-half flags for rows 0..3 (1 bit per sub-block)
 *   scales_h[8]:  2 bits per scale (8 bytes, covers 32 scale entries)
 *     16 scales per row x 4 rows = 64 scales, but packed as 32 entries
 *     (2 scales per subblock per row, 8 subblocks)
 *   scales_l[32]: 4-bit magnitude scales interleaved (32 bytes)
 *     64 scales total (16 per row x 4 rows)
 *   qs[512]:     4-bit quantized values interleaved across 4 rows (512 bytes)
 *     128 bytes per row * 4 rows = 512 bytes
 */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[4];         /* 4 FP16 global scales */
    uint8_t  extra[8];     /* LUT selection: extra[0..3]=low-half, extra[4..7]=high-half */
    uint8_t  scales_h[16]; /* 64 packed 2-bit scale extensions (16 per row x 4 rows) */
    uint8_t  scales_l[32]; /* 64 packed 4-bit magnitude scales (16 per row x 4 rows) */
    uint8_t  qs[512];      /* 1024 packed 4-bit values (256 per row x 4 rows) */
} block_iq4_k_r4;         /* 576 bytes = 4 * 144 */
#pragma pack(pop)
#pragma pack(pop)

/* IQ2_K non-linear values table (8 entries, int8)
 * Derived from llama.cpp ggml-common.h GGML_TABLE_BEGIN(int8_t, iq2nl_values, 8)
 * Normal table (indices 0-3): {-31, -13, 1, 17}
 * Shifted table (indices 4-7): {-26, -8, 6, 22}
 * The extra bits select which table per sub-block half. */
extern const int8_t iq2nl_values[8];

/* IQ3_K non-linear values table (16 entries, int8)
 * Derived from llama.cpp ggml-common.h GGML_TABLE_BEGIN(int8_t, iq3nl_values, 16)
 * Normal table (indices 0-7):  {-63, -40, -23, -10, 1, 13, 28, 47}
 * Shifted table (indices 8-15): {-59, -36, -19, -6, 5, 17, 32, 51}
 * The extra bits select which table per sub-block half. */
extern const int8_t iq3nl_values[16];

/* IQ4_K non-linear values table (32 entries, int8)
 * Derived from llama.cpp ggml-common.h GGML_TABLE_BEGIN(int8_t, iq4k_values, 32)
 * Table 0 (indices 0-15):  {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113}
 * Table 1 (indices 16-31): {-123,-100,-79,-61,-45,-31,-18, -6, 5,17,29,42,57,73,93,117}
 * The extra bits select which table per sub-block half. */
extern const int8_t iq4k_values[32];

/* Q4_1 block: 32 weights (old GGML format, used by some GGUF models)
 * Layout: half d (scale), half m (min), uchar qs[16] (nibbles)
 * Dequant: val = qs[j] * d + m  (unsigned nibble, no sign extension) */
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* scale (FP16) */
    uint16_t m;          /* min (FP16) */
    uint8_t  qs[16];     /* 4-bit quantized values */
} block_q4_1;            /* 20 bytes */

/* ---- Little-endian-safe loaders for multi-byte bitmask fields ----
 *
 * Q5_0/Q5_1 store their 5th bits as `uint8_t qh[4]`, which is read as a
 * uint32_t bitmask: value j's 5th bit is bit j. That bitmask is written
 * little-endian in the GGUF, so a bare memcpy() of the 4 bytes into a uint32_t
 * is only correct on a little-endian host. On big-endian (Wii/Broadway, Mac
 * PPC) the bytes land reversed and every value gets the WRONG 5th bit.
 *
 * The runtime BE swap in model_gguf.c only byte-swaps the FP16 d/m scales, and
 * the offline gguf_be_swap tool likewise touches only d/m, so nothing else
 * repairs qh. Hence these helpers: assemble the uint32 from the bytes
 * explicitly, which is correct on every byte order. */
static inline uint32_t picolm_qh_load(const uint8_t *qh) {
    return (uint32_t)qh[0]
         | ((uint32_t)qh[1] << 8)
         | ((uint32_t)qh[2] << 16)
         | ((uint32_t)qh[3] << 24);
}

/* Q5_0 block: 32 weights (5-bit values + FP16 scale, 22 bytes)
 * Layout: half d (scale), uchar qh[4] (5th bits), uchar qs[16] (low 4 bits)
 * Dequant: val = ((qs & 0xF) | (qh_bit << 4)) * d
 * qh holds 1 bit per value: value j's 5th bit is bit j of the 32-bit qh. */
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* scale (FP16) */
    uint8_t  qh[4];      /* 5th bit of each quant (32 bits) */
    uint8_t  qs[16];     /* low 4 bits of each quant */
} block_q5_0;            /* 22 bytes */

/* Q5_1 block: 32 weights (5-bit values + FP16 scale + FP16 min, 24 bytes)
 * Layout: half d (scale), half m (min), uchar qh[4], uchar qs[16]
 * Dequant: val = ((qs & 0xF) | (qh_bit << 4)) * d + m */
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* scale (FP16) */
    uint16_t m;          /* min (FP16) */
    uint8_t  qh[4];      /* 5th bit of each quant (32 bits) */
    uint8_t  qs[16];     /* low 4 bits of each quant */
} block_q5_1;            /* 24 bytes */

/* Q1_0 block: 128 weights (1-bit sign + FP16 scale, 18 bytes)
 * Dequant: val[j] = (bit[j] ? +d : -d) */
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* scale (FP16) = mean(|values|) */
    uint8_t  qs[16];     /* 128 sign bits (1 bit per value) */
} block_q1_0;            /* 18 bytes */

/* Q2_0 block: 128 weights (2-bit values + FP16 scale, 34 bytes)
 * Dequant: val[j] = ((qs[j] - 1) * d), {0,1,2,3} -> {-d, 0, +d, +2d} */
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* scale (FP16) = max(|values|) */
    uint8_t  qs[32];     /* 128 values * 2 bits each */
} block_q2_0;            /* 34 bytes */

/* Q6_0 block: 32 weights (6-bit values + FP16 scale, 26 bytes)
 * Legacy GGML format (GGUF type 133). No SIMD vec_dot exists in upstream.
 * Layout: half d (scale, F16), uchar qh[8] (5th+6th bits), uchar qs[16] (low 4 bits)
 * Dequant: each value j (0..31) stored as two halves:
 *   j0 = j (first half), j1 = j + 16 (second half)
 *   low4 = qs[j] & 0x0F  |  ((qh[j%8] >> (4*(j/8))) & 0x30)
 *   low4 = (qs[j] >> 4)  |  ((qh[j%8] >> (4*(j/8)-4)) & 0x30)
 *   val = (low4 - 32) * d
 * qh stores the 5th and 6th bits packed: 2 bits per value, 4 values per byte.
 * d uses LLMAMA-style optimal scale correction (sum(qw*x*q)/sum(qw*q)). */
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* scale (FP16) */
    uint8_t  qh[8];      /* 5th+6th bits of 32 quants (2 bits each, packed) */
    uint8_t  qs[16];     /* low 4 bits of 32 quants (nibbles) */
} block_q6_0;            /* 26 bytes */

#pragma pack(pop)

/* ---- FP16 conversion ---- */
uint16_t fp32_to_fp16(float f);

/* ---- Dequantize a row of weights into float output buffer ---- */
void dequantize_row_q4_K(const void *src, float *dst, int n);
void dequantize_row_q3_K(const void *src, float *dst, int n);
void dequantize_row_q2_K(const void *src, float *dst, int n);
void dequantize_row_q8_0(const void *src, float *dst, int n);
void dequantize_row_q6_K(const void *src, float *dst, int n);
/* Dequantize one Q6_K_R4 block group (4 interleaved rows) to F32.
 * src has (n/256) block_q6_K_R4 blocks. n = elements per logical row
 * (must be a multiple of 256). dst must hold 4*n floats, row-major
 * (row k at dst + k*n), matching the dequantize_row_q4_0_r8 convention. */
void dequantize_row_q6_K_R4(const void *src, float *dst, int n);
void dequantize_row_q6_0(const void *src, float *dst, int n);
void dequantize_row_q5_K(const void *src, float *dst, int n);
void dequantize_row_q4_0(const void *src, float *dst, int n);
/* Older-generation repacked Q4_0 (NEON 4x4/4x8, AVX2 8x8). Each block holds
 * N interleaved rows of Q4_0, nibbles pre-XOR'd with 0x88. Superseded by the
 * _R4/_R8 family but still written by llama-quantize, so they must decode.
 * The _single variants are row-aware: pass row in [0, N). */
void dequantize_row_q4_0_4_4(const void *src, float *dst, int n);
void dequantize_row_q4_0_4_4_single(const block_q4_0x4 *x, float *dst, int n, int row);
void dequantize_row_q4_0_4_8(const void *src, float *dst, int n);
void dequantize_row_q4_0_4_8_single(const block_q4_0x4 *x, float *dst, int n, int row);
void dequantize_row_q4_0_8_8(const void *src, float *dst, int n);
void dequantize_row_q4_0_8_8_single(const block_q4_0x8 *x, float *dst, int n, int row);
/* Q5_0 dequantize (22-byte blocks). Was defined in quant.c but never declared
 * here, so any test or tool that called it got an implicit int-returning
 * declaration -- which silently truncates float results. */
void dequantize_row_q5_0(const void *src, float *dst, int n);
void dequantize_row_iq4_nl(const void *src, float *dst, int n);
void dequantize_row_f16(const void *src, float *dst, int n);
void dequantize_row_f32(const void *src, float *dst, int n);

/* Generic dispatch by type */
void dequantize_row(const void *src, float *dst, int n, gguf_type_t type);

/* Block size (number of weights per block) */
int gguf_type_block_size(gguf_type_t type);

/* Bytes per block of quantized data */
int gguf_type_quant_size(gguf_type_t type);

/* Bytes for n elements of the given type */
size_t gguf_type_row_size(gguf_type_t type, int n);

/* ---- Fused dot products (dequant + dot in one pass, no scratch buffer) ---- */
float vec_dot_q4_K_f32(const void *src, const float *x, int n);
float vec_dot_q6_K_f32(const void *src, const float *x, int n);
float vec_dot_q6_0_f32(const void *src, const float *x, int n);
float vec_dot_q6_0_q8_0(const void *src_q6, const void *src_q8, int n);
float vec_dot_q6_K_q8_K(const void *src_q6, const void *src_q8, int n);
/* Q6_K_R4 GEMV: dot 4 interleaved weight rows (one block group) against one
 * Q8_K-quantized activation row. vx points at (n/256) block_q6_K_R4 blocks
 * (the start of a row group; group index = row_idx/4). vy points at (n/256)
 * block_q8_K blocks for the activation. Writes 4 floats (one per row,
 * row-major) to out[0..3]. AVX2-accelerated when available, portable scalar
 * fallback otherwise. Ported from ik_llama.cpp's mul_mat_q6_k_r4_q8_k. */
void vec_dot_q6_K_R4_q8_K(const void *vx, const void *vy, int n, float *out);
/* Q6_K_R4 x Q8_K GEMM: nrows (multiple of 4) weight rows x ncols activation
 * columns. vx: nrows/4 row groups of (k/256) block_q6_K_R4 each, row-major
 * by group (stride = gguf_type_row_size(GGUF_TYPE_Q6_K_R4,k)*4 bytes per
 * group -- i.e. plain per-row striding with row_bytes computed the usual
 * way). vy: ncols activation rows of (k/256) block_q8_K each, contiguous.
 * out[w*4 + r + c*bs] receives row (w*4+r), column c. Returns the number of
 * rows actually processed (a multiple of 4), or 0 if unsupported (k not a
 * multiple of 256, or nrows < 4). */
int sgemm_q6_k_r4_q8_k(int nrows, int ncols, int k,
                        const void *vx, const void *vy,
                        float *out, size_t bs,
                        int ith, int nth);
int sgemm_q4_k_r4_q8_k_avx2(int nrows, int ncols, int k,
                             const void *vx, const void *vy,
                             float *out, size_t bs,
                             int ith, int nth);
float vec_dot_f32_f32(const void *src, const float *x, int n);
float vec_dot_q8_0_f32(const void *src, const float *x, int n);
void vec_dot_q8_0_f32_batch4(const void *qx0, const void *qx1, const void *qx2, const void *qx3,
                              const float *w, int n,
                              float *out0, float *out1, float *out2, float *out3);
float vec_dot_q4_0_f32(const void *src, const float *x, int n);
float vec_dot_q4_1_f32(const void *src, const float *x, int n);
/* fp16-fp32 dot product: sum of fp16_to_fp32(k[i]) * x[i] */
float vec_dot_f16_f32(const void *src, const float *x, int n);
float vec_dot_q8_0_q8_0(const void *qx, const void *qw, int n);
float vec_dot_q8_0_q8_0_deltas(const void *qx, const float *qx_d, const void *qw, int n);
/* Batch-4: compute 4 dot products with 1 shared weight row */
void vec_dot_q8_0_q8_0_deltas_batch4(
        const void *qx0, const float *qx_d0,
        const void *qx1, const float *qx_d1,
        const void *qx2, const float *qx_d2,
        const void *qx3, const float *qx_d3,
        const void *qw, int n,
        float *out0, float *out1, float *out2, float *out3);
/* Q1_0 * Q8_0 dot product: 1-bit weights with pre-quantized Q8_0 input */
float vec_dot_q1_0_q8_0(const void *src_q1, const void *src_q8, int n);
/* Q2_0 * Q8_0 dot product: 2-bit weights with pre-quantized Q8_0 input */
float vec_dot_q2_0_q8_0(const void *src_q2, const void *src_q8, int n);
/* Q4_0 * Q8_0 dot product: Q4_0 weights with pre-quantized Q8_0 input */
float vec_dot_q4_0_q8_0(const void *src_q4, const void *src_q8, int n);
/* Q5_0 x Q8_0: portable scalar (no SIMD). Q5_0 is the only legacy type that
 * lacked a fused Q8_0-activation dot, which left it unusable on scalar-only
 * builds. See the comment on the definition in quant.c. */
float vec_dot_q5_0_q8_0(const void *src_q5, const void *src_q8, int n);
/* Q4_1 x Q8_0: portable scalar. Q4_1 had no type-table entry at all, so it
 * reached the F32 vec_dot with a quantized buffer. See quant.c. */
float vec_dot_q4_1_q8_0(const void *src_q4, const void *src_q8, int n);
/* IQ4_NL * Q8_0 dot product: IQ4_NL weights with pre-quantized Q8_0 input */
float vec_dot_iq4_nl_q8_0(const void *src_iq4, const void *src_q8, int n);
/* IQ4_NL * F32 dot product: fused dequant + dot (scalar fallback) */
float vec_dot_iq4_nl_f32(const void *src_iq4, const float *x, int n);
/* IQ4_XS dequantize: convert IQ4_XS bytes to float32 */
void dequantize_row_iq4_xs(const void *src, float *dst, int n);
/* IQ4_XS x Q8_K SIMD GEMV kernels (sgemm_iq4_xs.c) */
void vec_dot_iq4_xs_q8_k_avx2(const void *vx, const void *wy, int n, float *out);
void vec_dot_iq4_xs_q8_k_neon(const void *vx, const void *wy, int n, float *out);
/* IQ4_XS * Q8_K dot product: IQ4_XS weights with pre-quantized Q8_K input */
float vec_dot_iq4_xs_q8_k(const void *src_iq4, const void *src_q8, int n);
/* IQ4_XS * F32 dot product: fused dequant + dot (scalar fallback) */
float vec_dot_iq4_xs_f32(const void *src_iq4, const float *x, int n);
/* Q4_K * Q8_K dot product: Q4_K weights with pre-quantized Q8_K input */
float vec_dot_q4_K_q8_K(const void *src_q4, const void *src_q8, int n);
/* Q5_K * Q8_K dot product: Q5_K weights with pre-quantized Q8_K input */
float vec_dot_q5_K_q8_K(const void *src_q5, const void *src_q8, int n);
/* Q5_1 * Q8_0 dot product: Q5_1 weights with pre-quantized Q8_0 input */
float vec_dot_q5_1_q8_0(const void *src_q5, const void *src_q8, int n);
/* Q5_1 row to Q8_0 shadow: for weight-stationary batching */
void q5_1_row_to_q8_0_shadow(const void *src, void *dst, int n);
/* Q3_K * Q8_K dot product: Q3_K weights with pre-quantized Q8_K input */
float vec_dot_q3_K_q8_K(const void *src_q3, const void *src_q8, int n);
/* Q2_K * Q8_K dot product: Q2_K weights with pre-quantized Q8_K input */
float vec_dot_q2_K_q8_K(const void *src_q2, const void *src_q8, int n);
/* Q2_K * F32 dot product: Q2_K weights with float activations */
float vec_dot_q2_K_f32(const void *src, const float *x, int n);
/* Q4_0_4_4 interleaved weights x Q8_0 input (blocklen=4): processes nrows (multiple of 4) */
void vec_dot_q4_0x4_q8_0(const void *vx, const void *wy, int n, float *out, int nrows);
/* Q4_0_4_8 interleaved weights x Q8_0 input (blocklen=8): processes nrows (multiple of 4) */
void vec_dot_q4_0x4_4x8_q8_0(const void *vx, const void *wy, int n, float *out, int nrows);
/* Q4_0_4_8 I8MM batched gemm: processes d output rows x n_batch activations */
void gemm_q4_0_4x8_q8_0(const void *W, const void *X, int n, float *out, int d, int n_batch);
/* Q4_0_8x8 interleaved weights x Q8_0 input (AVX2): processes nrows (multiple of 8) simultaneously */
void vec_dot_q4_0x8_q8_0_avx2(const void *vx, const void *wy, int n, float *out, int nrows);
/* Q4I_0_8_8 pre-dequantized int8 x Q8_0: AVX-512 VNNI (dpbusd) / maddubs / scalar fallback */
void vec_dot_q4i_0x8_q8_0(const void *vx, const void *wy, int n, float *out, int nrows);
/* Q4_0_R8 x Q8_0 GEMV (AVX2): 8 weight rows x 1 activation row.
 * Uses block_q8_0 activations, computes activation sum in scalar loop. */
void vec_dot_q4_0_r8_q8_0_avx2(const void *vx, const void *wy, int n,
                                 float *out, int nrows);
/* Q4_0_R8 x Q8_0 batched GEMM (AVX2). */
int sgemm_q4_0_r8_q8_0_avx2(int nrows, int ncols, int k,
                             const void *vx, const void *vy,
                             float *out, size_t bs,
                             int ith, int nth);
/* Q4_0_R8 x Q8_2 GEMV (AVX2): 8 weight rows x 1 activation row.
 * Uses block_q8_2 activations with precomputed int16 sum (no scalar loop). */
void vec_dot_q4_0_r8_q8_2_avx2(const void *vx, const void *wy, int n,
                                 float *out, int nrows);
/* Q4_0_R8 x Q8_2 batched GEMM (AVX2). */
int sgemm_q4_0_r8_q8_2_avx2(int nrows, int ncols, int k,
                             const void *vx, const void *vy,
                             float *out, size_t bs,
                             int ith, int nth);
/* Quantize 8 rows of F32 to Q4_0_R8 interleaved format.
 * dst must have space for (n/32) * sizeof(block_q4_0x8) bytes per row group. */
void quantize_row_q4_0_r8(const float *x, void *dst, int n);
/* Dequantize 8 rows of Q4_0_R8 to F32. dst must hold 8*n floats. */
void dequantize_row_q4_0_r8(const void *src, float *dst, int n);
/* Dequantize a single row from the 8-row interleaved Q4_0_R8 group. */
void dequantize_row_q4_0_r8_single(const void *src, float *dst, int n, int row);
/* Quantize 8 rows of F32 to Q8_0_R8 interleaved format. */
void quantize_row_q8_0_r8(const float *x, void *dst, int n);
/* Repack 8-row groups of plain Q8_0 rows (row stride n/32 blocks) into
 * block_q8_0_r8 interleaved groups (dst = nrows/8 groups). Portable. */
void repack_q8_0_to_q8_0_r8(const void *src, void *dst, int nrows, int n);
/* Dequantize 8 rows of Q8_0_R8 to F32. dst must hold 8*n floats. */
void dequantize_row_q8_0_r8(const void *src, float *dst, int n);
void dequantize_row_q8_0_r8_single(const void *src, float *dst, int n, int row);
void vec_dot_q8_0_r8_q8_0_avx2(const void *vx, const void *wy, int n, float *out, int nrows);
int sgemm_q8_0_r8_q8_0_avx2(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q8_0_r8_vnni_q8_0(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);  /* AVX-512VL VNNI; returns 0 if unavailable */
int sgemm_q8_0_r8_avx2t_q8_0(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);  /* panel-blocked AVX2; returns 0 if unavailable */
int picolm_sgemm_q8_0_avx2t(int m, int n, int k_blocks, const void *vA, int lda, const void *vB, int ldb, float *C, int ldc, int ith, int nth);
int picolm_sgemm_r8dec_avx2t(int atype, int m, int n, int k_blocks, const void *vA, int lda, const void *vB, int ldb, float *C, int ldc, int ith, int nth);  /* atype: Q8_0 / Q4_0 / IQ4_NL */
int picolm_sgemm_r8dec_vnni(int atype, int m, int n, int k_blocks, const void *vA, int lda, const void *vB, int ldb, float *C, int ldc, int ith, int nth);
int picolm_r8_big_enough(int m, int k_blocks);   /* size gate for the decode-once R8 prefill paths, see sgemm_q8_0_r8.c */
int sgemm_iq4_nl_r4_q8_0_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);  /* decode-once R8 path, falls back to the 1x4 tile kernel */
int sgemm_q8_0_r8_q8_0_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);  /* VNNI when available, else AVX2; always computes */
/* Q8_K_R8 x Q8_K GEMV (AVX2): 8 weight rows x 1 activation row.
 * Uses block_q8_k_r8 weights (sign trick, no bias correction).
 * Activations in block_q8_K format. */
void vec_dot_q8_k_r8_q8_k_avx2(const void *vx, const void *wy, int n,
                                 float *out, int nrows);
/* Q8_K_R8 x Q8_K batched GEMM (AVX2). */
int sgemm_q8_k_r8_q8_k_avx2(int nrows, int ncols, int k,
                             const void *vx, const void *vy,
                             float *out, size_t bs,
                             int ith, int nth);
/* Quantize 8 rows of F32 to Q8_K_R8 interleaved format. */
void quantize_row_q8_k_r8(const float *x, void *dst, int n);
/* Dequantize 8 rows of Q8_K_R8 to F32. dst must hold 8*n floats. */
void dequantize_row_q8_k_r8(const void *src, float *dst, int n);
/* Dequantize a single row from the 8-row interleaved Q8_K_R8 group. */
void dequantize_row_q8_k_r8_single(const void *src, float *dst, int n, int row);
/* Convert F32 activations to Q8_2 blocks (with row sum).
 * dst must have space for (n/32) * sizeof(block_q8_2) bytes. */
void quantize_row_q8_2(const float *x, void *dst, int n);
/* Convert F32 activations to Q8_2_x4 interleaved format (4 rows).
 * dst must have space for (n/32) * sizeof(block_q8_2_x4) bytes per group. */
void quantize_mat_q8_2_x4(const float *x, void *dst, int n, int row_stride);

/* IQ2_K plain * Q8_K dot product: single-row IQ2_K weights with pre-quantized Q8_K input.
 * n must be a multiple of 256 (QK_K). */
float vec_dot_iq2_k_q8_k(const void *vx, const void *wy, int n);
/* IQ2_K plain dequantize: scalar reference. dst must have space for n floats. */
void dequantize_row_iq2_k(const void *src, float *dst, int n);
/* IQ2_K plain x Q8_K AVX2 GEMV. out: 1 float. n must be multiple of 256. */
void vec_dot_iq2_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out);
void convert_iq2_k_to_q8_k_r8(int n, const void *vx, size_t bx, void *vy, int nrc_x);
/* IQ2_K plain x Q8_K NEON GEMV. out: 1 float. n must be multiple of 256. */
void vec_dot_iq2_k_q8_k_neon(const void *vx, const void *wy, int n, float *out);
/* IQ2_K plain x Q8_K batched GEMM (AVX2). */
int sgemm_iq2_k_q8_k_avx2(int nrows, int ncols, int k,
                            const void *vx, const void *vy,
                            float *out, size_t bs,
                            int ith, int nth);
/* IQ2_K plain x Q8_K batched GEMM (NEON). */
int sgemm_iq2_k_q8_k_neon(int nrows, int ncols, int k,
                            const void *vx, const void *vy,
                            float *out, size_t bs,
                            int ith, int nth);

/* IQ3_K plain * Q8_K dot product: single-row IQ3_K weights with pre-quantized Q8_K input.
 * n must be a multiple of 256 (QK_K). */
float vec_dot_iq3_k_q8_k(const void *vx, const void *wy, int n);
/* IQ3_K plain dequantize: scalar reference. dst must have space for n floats. */
void dequantize_row_iq3_k(const void *src, float *dst, int n);
/* IQ3_K plain x Q8_K AVX2 GEMV. out: 1 float. n must be multiple of 256. */
void vec_dot_iq3_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out);
/* IQ3_K plain x Q8_K batched GEMM (AVX2). */
int sgemm_iq3_k_q8_k_avx2(int nrows, int ncols, int k,
                            const void *vx, const void *vy,
                            float *out, size_t bs,
                            int ith, int nth);

/* IQ2_K_R4 * Q8_K dot product: IQ2_K_R4 weights (4-row interleaved) with pre-quantized Q8_K input
 * Returns dot product for a single row. Use vec_dot_iq2_k_r4_q8_k_batch4() for 4 rows.
 * n must be a multiple of 256 (QK_K). */
float vec_dot_iq2_k_r4_q8_k(const void *vx, const void *wy, int n);
/* IQ2_K_R4 * Q8_K batched dot product: computes 4 row dot products at once.
 * out[0..3] = dot products for rows 0..3 of the interleaved block.
 * n must be a multiple of 256 (QK_K). */
void vec_dot_iq2_k_r4_q8_k_batch4(const void *vx, const void *wy, int n, float *out);
/* IQ2_K_R4 dequantize: scalar reference implementation for validation.
 * Dequantizes a single row (k/4 values) from the 4-row interleaved block.
 * dst must have space for k/4 floats. */
void dequantize_row_iq2_k_r4(const void *src, float *dst, int n);
/* IQ2_K_R4 x Q8_K AVX2 GEMV: 4 weight rows x 1 activation row.
 * out: 4 output floats. nrows must be 4. n must be multiple of 256. */
void vec_dot_iq2_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows);
/* IQ2_K_R4 x Q8_K NEON GEMV: 4 weight rows x 1 activation row.
 * out: 4 output floats. nrows must be 4. n must be multiple of 256. */
void vec_dot_iq2_k_r4_q8_k_neon(const void *vx, const void *wy, int n,
                                  float *out, int nrows);
/* IQ2_K_R4 x Q8_K batched GEMM (AVX2). */
int sgemm_iq2_k_r4_q8_k_avx2(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth);
/* IQ2_K_R4 x Q8_K batched GEMM (NEON). */
int sgemm_iq2_k_r4_q8_k_neon(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth);

/* IQ3_K_R4 * Q8_K dot product: IQ3_K_R4 weights (4-row interleaved) with pre-quantized Q8_K input
 * Returns dot product for a single row. Use vec_dot_iq3_k_r4_q8_k_batch4() for 4 rows.
 * n must be a multiple of 256 (QK_K). */
float vec_dot_iq3_k_r4_q8_k(const void *vx, const void *wy, int n);
/* IQ3_K_R4 * Q8_K batched dot product: computes 4 row dot products at once.
 * out[0..3] = dot products for rows 0..3 of the interleaved block.
 * n must be a multiple of 256 (QK_K). */
void vec_dot_iq3_k_r4_q8_k_batch4(const void *vx, const void *wy, int n, float *out);
/* IQ3_K_R4 dequantize: scalar reference implementation for validation.
 * Dequantizes a single row (k/4 values) from the 4-row interleaved block.
 * dst must have space for k/4 floats. */
void dequantize_row_iq3_k_r4(const void *src, float *dst, int n);
void dequantize_row_iq2_k_r4_single(const block_iq2_k_r4 *x, float *dst, int n, int row);
void dequantize_row_iq3_k_r4_single(const block_iq3_k_r4 *x, float *dst, int n, int row);
void dequantize_row_iq4_k_r4_single(const block_iq4_k_r4 *x, float *dst, int n, int row);

/* IQ4_NL_R4 dequantize: single row from 4-row interleaved block.
 * dst must have space for n floats. n is per-row (total = n*4 in GGUF). */
void dequantize_row_iq4_nl_r4(const void *src, float *dst, int n);
void dequantize_row_iq4_nl_r4_single(const block_iq4_nl_r4 *x, float *dst, int n, int row);
/* IQ4_NL_R4 x Q8_0 scalar vec_dot (single row). */
float vec_dot_iq4_nl_r4_q8_0(const void *vx, const void *wy, int n);
/* IQ4_NL_R4 x Q8_0 batched vec_dot: 4 rows at once. */
void vec_dot_iq4_nl_r4_q8_0_batch4(const void *vx, const void *wy, int n, float *out);
/* IQ4_NL_R4 x Q8_0 AVX2 GEMV: 4 weight rows x 1 activation row. */
void vec_dot_iq4_nl_r4_q8_0_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows);
/* IQ4_NL_R4 x Q8_0 NEON GEMV: 4 weight rows x 1 activation row. */
void vec_dot_iq4_nl_r4_q8_0_neon(const void *vx, const void *wy, int n,
                                  float *out, int nrows);
/* IQ4_NL_R4 x Q8_0 batched GEMM (AVX2). */
int sgemm_iq4_nl_r4_q8_0_avx2(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth);
/* IQ4_NL_R4 x Q8_0 batched GEMM (NEON). */
int sgemm_iq4_nl_r4_q8_0_neon(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth);

/* IQ3_K_R4 x Q8_K AVX2 GEMV: 4 weight rows x 1 activation row.
 * out: 4 output floats. nrows must be 4. n must be multiple of 256. */
void vec_dot_iq3_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows);
/* IQ3_K_R4 x Q8_K NEON GEMV: 4 weight rows x 1 activation row.
 * out: 4 output floats. nrows must be 4. n must be multiple of 256. */
void vec_dot_iq3_k_r4_q8_k_neon(const void *vx, const void *wy, int n,
                                  float *out, int nrows);
/* IQ3_K_R4 x Q8_K batched GEMM (AVX2). */
int sgemm_iq3_k_r4_q8_k_avx2(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth);

/* IQ4_K plain x Q8_K AVX2 GEMV. */
void vec_dot_iq4_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out);
/* IQ4_K plain x Q8_K scalar vec_dot. */
float vec_dot_iq4_k_q8_k(const void *vx, const void *wy, int n);
/* IQ4_K plain row dequantizer (GPU fp16-GEMM emulator reference). */
void dequantize_row_iq4_k(const void *src, float *dst, int n);
/* IQ6_K plain (GGUF type 141): 6-bit non-linear quant, 212-byte blocks. */
void dequantize_row_iq6_k(const void *src, float *dst, int n);
/* Max activation rows decoded-weight reuse covers per kernel call. */
#define IQ6K_BATCH_TILE 4
/* One IQ6_K weight row x ncols Q8_K activation rows (row c at wy + c*y_stride bytes).
 * Decodes each weight block once. ncols may exceed IQ6K_BATCH_TILE. */
void vec_dot_iq6_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out);
void vec_dot_iq6_k_q8_k_avx2_batch(const void *vx, const void *wy, size_t y_stride,
                                   int n, int ncols, float *out);
void vec_dot_iq6_k_q8_k_vnni(const void *vx, const void *wy, int n, float *out);
void vec_dot_iq6_k_q8_k_vnni_batch(const void *vx, const void *wy, size_t y_stride,
                                   int n, int ncols, float *out);
float vec_dot_iq6_k_q8_k(const void *vx, const void *wy, int n);
void vec_dot_iq6_k_q8_k_batch(const void *vx, const void *wy, size_t y_stride,
                              int n, int ncols, float *out);
int sgemm_iq6_k_q8_k_avx2(int nrows, int ncols, int k, const void *vx, const void *vy,
                           float *out, size_t bs, int ith, int nth);
int sgemm_iq6_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy,
                         float *out, size_t bs, int ith, int nth);   /* decode-once R8 GEMM; returns 0 if unsupported */
int sgemm_iq6_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy,
                         float *out, size_t bs, int ith, int nth);   /* VNNI variant; returns 0 if unavailable */
int sgemm_iq4_xs_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);  /* IQ4_XS decode-once, AVX2 */
int sgemm_iq4_xs_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);  /* AVX-512 VNNI+VL; 0 if unavailable */
int sgemm_q6_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q6_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q3_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q3_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q6_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);   /* table fn_gemm: always computes the full share */
int sgemm_q3_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q4_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q4_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q5_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q5_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q2_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q2_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q5_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q2_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q4_k_q8_k_fast(int nrows, int ncols, int k_blocks, const void *vx, int lda, const void *vy, int ldb, float *out, size_t bs, int ith, int nth);   /* 1 if handled, else fall through */
int sgemm_q4_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q4_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q4_k_r4_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);   /* table fn_gemm for Q4_K_R4 */
int sgemm_q6_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q6_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_q6_k_r4_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);   /* table fn_gemm for Q6_K_R4 */
int sgemm_iq4_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq4_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq4_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq4_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq4_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq4_k_r4_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq2_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq2_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq3_k_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq3_k_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq2_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq2_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq3_k_r4_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq3_k_r4_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq2_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq3_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq2_k_r4_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq3_k_r4_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq6_k_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy,
                          float *out, size_t bs, int ith, int nth);   /* r8t when worthwhile, else tile kernel; always computes */
/* IQ3_XXS x Q8_K table fn_gemm: decode-once R8 (VNNI/AVX2) for big shapes, else the
 * 1-row GEMV over this thread's rows. Always computes its whole share. */
int sgemm_iq3_xxs_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq3_s_q8_k_best(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq3_s_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq3_s_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq3_xxs_q8_k_r8t(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
int sgemm_iq3_xxs_q8_k_r8v(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
/* IQ3_XXS x Q8_K ARM NEON tiled GEMM (I8MM / DOTPROD / plain NEON tiers).
 * UNTESTED: has not been executed on real ARM hardware yet. Returns 0 when
 * unsupported so the caller can fall back to the per-column GEMV. */
int sgemm_iq3_xxs_q8_k_neon(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth);
/* IQ6_K plain x Q8_K ARM NEON GEMV/GEMM. */
void vec_dot_iq6_k_q8_k_neon(const void *vx, const void *wy, int n, float *out);
void vec_dot_iq6_k_q8_k_neon_batch(const void *vx, const void *wy, size_t y_stride,
                                   int n, int ncols, float *out);
int sgemm_iq6_k_q8_k_neon(int nrows, int ncols, int k, const void *vx, const void *vy,
                          float *out, size_t bs, int ith, int nth);
/* IQ4_K_R4 x Q8_K scalar vec_dot (row 0 only). */
float vec_dot_iq4_k_r4_q8_k(const void *vx, const void *wy, int n);
/* IQ4_K_R4 x Q8_K AVX2 GEMV: 4 weight rows x 1 activation row.
 * out: 4 output floats. nrows must be 4. n must be multiple of 256. */
void vec_dot_iq4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows);
/* IQ4_K_R4 x Q8_K NEON GEMV: 4 weight rows x 1 activation row.
 * out: 4 output floats. nrows must be 4. n must be multiple of 256. */
void vec_dot_iq4_k_r4_q8_k_neon(const void *vx, const void *wy, int n,
                                  float *out, int nrows);
/* IQ4_K_R4 x Q8_K batched GEMV dispatcher: 4 weight rows x 1 activation row.
 * out: 4 output floats. n must be multiple of 256. */
void vec_dot_iq4_k_r4_q8_k_batch4(const void *vx, const void *vy, int n, float *out);

/* Q4_K_R4 dequantize: scalar reference for a single row from the 4-row interleaved block.
 * dst must have space for n floats. row must be 0..3. */
void dequantize_row_q4_k_r4_single(const block_q4_k_r4 *x, float *dst, int n, int row);
void dequantize_row_q5_k_r4_single(const block_q5_k_r4 *x, float *dst, int n, int row);
void dequantize_row_q5_k_r4(const void *src, float *dst, int n);
void dequantize_row_q2_k_r4_single(const block_q2_k_r4 *x, float *dst, int n, int row);
void dequantize_row_q2_k_r4(const void *src, float *dst, int n);
float vec_dot_q2_k_r4_q8_k(const void *vx, const void *vy, int n);
void vec_dot_q2_k_r4_q8_k_batch4(const void *vx, const void *vy, int n, float *out, int nrows);
void dequantize_row_iq3_xxs(const void *src, float *dst, int n);
/* Standard GGUF IQ grid quants (quant.c, 256-value blocks). Dequantizers are
 * the reference implementations; the vec_dot_* are the simple
 * dequantize-then-dot form (correct, not fast). */
void dequantize_row_iq2_xxs(const void *src, float *dst, int n);
void dequantize_row_iq2_xs(const void *src, float *dst, int n);
void dequantize_row_iq1_s(const void *src, float *dst, int n);
void dequantize_row_iq3_s(const void *src, float *dst, int n);
void dequantize_row_iq2_s(const void *src, float *dst, int n);
void dequantize_row_iq1_m(const void *src, float *dst, int n);
float vec_dot_iq2_xxs_q8_k(const void *vx, const void *vy, int n);
float vec_dot_iq2_xs_q8_k(const void *vx, const void *vy, int n);
float vec_dot_iq1_s_q8_k(const void *vx, const void *vy, int n);
float vec_dot_iq3_s_q8_k(const void *vx, const void *vy, int n);
float vec_dot_iq2_s_q8_k(const void *vx, const void *vy, int n);
float vec_dot_iq1_m_q8_k(const void *vx, const void *vy, int n);
/* Scalar reference for each of the above: dequantize-then-dot, transcribed
 * from ggml-quants.c. Exposed so tests/check_iq_gemv.c can use it as the
 * oracle for the vectorised kernels. */
float vec_dot_iq2_xxs_q8_k_scalar(const void *vx, const void *vy, int n);
float vec_dot_iq2_xs_q8_k_scalar(const void *vx, const void *vy, int n);
float vec_dot_iq1_s_q8_k_scalar(const void *vx, const void *vy, int n);
float vec_dot_iq3_s_q8_k_scalar(const void *vx, const void *vy, int n);
float vec_dot_iq2_s_q8_k_scalar(const void *vx, const void *vy, int n);
float vec_dot_iq1_m_q8_k_scalar(const void *vx, const void *vy, int n);
/* IQ3_XXS_R4 / IQ4_XS_R8 (ik_llama GGML types 218 / 223): row-interleaved
 * formats. The _single variants extract one logical row from a group base and
 * write exactly n floats; the group variants write rpb rows of n floats each. */
void dequantize_row_iq3_xxs_r4_single(const void *src, float *dst, int n, int row);
void dequantize_row_iq3_xxs_r4(const void *src, float *dst, int n);
void dequantize_row_iq4_xs_r8_single(const void *src, float *dst, int n, int row);
void dequantize_row_iq4_xs_r8(const void *src, float *dst, int n);
/* R4/R8 GEMV: 4 (resp. 8) weight rows x 1 Q8_K activation row. */
void vec_dot_iq3_xxs_r4_q8_k(const void *vx, const void *vy, int n, float *out, int nrows);
void vec_dot_iq4_xs_r8_q8_k(const void *vx, const void *vy, int n, float *out, int nrows);
float vec_dot_iq3_xxs_q8_k(const void *vx, const void *vy, int n);
/* NC columns at once for one weight row: decode shared, bit-identical to NC calls of the above.
 * ystride = byte stride between activation columns; res[c] gets column c's dot. */
void vec_dot_iq3_xxs_q8_k_cols(const void *vx, const void *vy, size_t ystride, int nc, int n, float *res);
/* Scalar (integer-exact) reference for vec_dot_iq3_xxs_q8_k, used as the
 * validation oracle by tests/kq_gemm_check.c. */
float vec_dot_iq3_xxs_q8_k_ref(const void *vx, const void *vy, int n);
/* IQ3_XXS reference tables (see quant.c), shared by the GEMM decode path. */
const uint32_t *picolm_iq3xxs_grid(void);    /* 256 x packed 4-byte grids */
const uint32_t *picolm_iq3s_grid(void);      /* 512 x packed 4-byte grids (IQ3_S) */
const uint8_t  *picolm_iq3xxs_ksigns(void);  /* 128 sign masks */
const uint8_t  *picolm_iq3xxs_kmask(void);   /* 8 bit masks */

/* Correct 4-row R4 vecdot built on a verified per-row dequantizer. `block_bytes`
 * is the size of one R4 block (e.g. sizeof(block_iq4_k_r4)). */
void vec_dot_r4_q8_k_dequant_ref(const void *vx, const void *vy, int n,
                                 float *out, int nrows,
                                 void (*deq_single)(const void *, float *, int, int),
                                 size_t block_bytes);

/* Q4_K_R4 x Q8_K scalar vec_dot (single row). */
float vec_dot_q4_k_r4_q8_k(const void *vx, const void *wy, int n);
/* Q4_K_R4 x Q8_K batch4 vec_dot: computes 4 row dot products at once.
 * out[0..3] = dot products for rows 0..3 of the interleaved block. */
void vec_dot_q4_k_r4_q8_k_batch4(const void *vx, const void *wy, int n, float *out);
/* Q4_K_R4 x Q8_K AVX2 GEMV: 4 weight rows x 1 activation row.
 * out: 4 output floats. nrows must be 4. n must be multiple of 256. */
void vec_dot_q4_k_r4_q8_k_avx2(const void *vx, const void *wy, int n,
                                  float *out, int nrows);
/* Q4_K_R4 x Q8_K ARM NEON GEMV: 4 weight rows x 1 activation row.
 * out: 4 output floats. nrows must be 4. n must be multiple of 256. */
void vec_dot_q4_k_r4_q8_k_neon(const void *vx, const void *wy, int n,
                                float *out, int nrows);

/* Repack standard Q4_0 weights to Q4_0_8x8 interleaved format (for AVX2).
 * dst must have the same size as src (1:1 byte mapping, just reordered). */
void repack_q4_0_to_q4_0x8(const void *src, void *dst, int nrows, int ncols);
/* Repack standard Q4_0 weights to Q4_0_4x4 interleaved format (for NEON dotprod). */
void repack_q4_0_to_q4_0x4(const void *src, void *dst, int nrows, int ncols);

/* Quantize a float32 vector to Q8_0 blocks in-place or to a separate buffer.
 * dst must have space for (n / 32) * sizeof(block_q8_0) bytes. */
void quantize_row_q8_0(const float *x, void *dst, int n);
void quantize_row_q8_2(const float *x, void *dst, int n);

/* Q8_0x4 interleaved block: 4 rows of Q8_0 packed for AVX2/AVX-512 GEMM.
 * 4 FP16 deltas + 128 interleaved int8 values. */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[4];      /* 4 FP16 deltas */
    int8_t   qs[128];    /* interleaved int8 (4 rows x 32 values) */
} block_q8_0x4;          /* 136 bytes */
#pragma pack(pop)

/* Q8_0_R8 interleaved block: 8 rows of Q8_0 packed for Q4_0_R8 GEMM.
 * 8 FP16 deltas + 256 interleaved int8 values. */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[8];      /* 8 FP16 deltas */
    int8_t   qs[256];    /* interleaved int8 (8 rows x 32 values) */
} block_q8_0_r8;         /* 272 bytes */
#pragma pack(pop)

/* Q8_2 block: Q8_0 with FP16 delta + int16 sum for delta-based GEMM.
 * Used as the activation format in Q4_0_R8 x Q8_2 GEMM kernels.
 * d = FP16 scale, s = int16 row sum of qs values. */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* scale (FP16) */
    int16_t  s;          /* sum of qs (int16) */
    int8_t   qs[32];     /* signed int8 values */
} block_q8_2;            /* 36 bytes */
#pragma pack(pop)

/* Q8_2_x4 interleaved block: 4 Q8_2 blocks packed for AVX2 SIMD.
 * 8 FP16 deltas (d[0..7] for blocks 0..3, each block has d+s).
 * Layout: d[8] + qs[128] (4 rows x 32 int8 values). */
#pragma pack(push, 1)
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d[8];      /* 8 FP16 deltas, one per Q8_2 block (d[2*k] for block k) */
    int8_t   qs[128];   /* 4 Q8_2 blocks x 32 int8 = 128 */
} block_q8_2_x4;         /* 144 bytes */
#pragma pack(pop)

/* Quantize 4 rows of F32 to interleaved block_q8_0x4. */
void quantize_mat_q8_0x4(const float *x, void *dst, int n, int row_stride);

/* Quantize a float32 vector to Q4_0 blocks.
 * dst must have space for (n / 32) * sizeof(block_q4_0) bytes. */
void quantize_row_q4_0(const float *x, void *dst, int n);

/* Quantize a float32 vector to IQ4_NL blocks (Lloyd-Max 4-bit non-linear).
 * dst must have space for (n / 32) * sizeof(block_iq4_nl) bytes.
 * Identical layout to Q4_0 but uses non-linear LUT for better accuracy. */
void quantize_row_iq4_nl(const float *x, void *dst, int n);

/* Quantize a float32 vector to TQ3 blocks (TurboQuant 3-bit codebook).
 * dst must have space for (n / 32) * sizeof(block_tq3) bytes.
 * n must be a multiple of 32. */
void quantize_row_tq3(const float *x, void *dst, int n);

/* Quantize a float32 vector to TQ4 blocks (TurboQuant 4-bit codebook).
 * dst must have space for (n / 32) * sizeof(block_tq4) bytes.
 * n must be a multiple of 32. */
void quantize_row_tq4(const float *x, void *dst, int n);

/* Converts one Q4_0 weight row to a "shadow" Q8_0 representation: same
 * per-block delta, values unpacked to (nibble - 8) directly as int8.
 * Q4_0's dequant formula is exactly (nibble-8)*d, which is exactly what
 * Q8_0 stores natively -- so this is a lossless format conversion, not
 * an approximation. Lets batched/prefill matmul decode a weight row's
 * nibbles ONCE and reuse the fast vec_dot_q8_0_q8_0_deltas kernel across
 * every token in the batch, instead of re-unpacking the same row from
 * scratch for every single token. dst must be sized
 * gguf_type_row_size(GGUF_TYPE_Q8_0, n) bytes. */
void q4_0_row_to_q8_0_shadow(const void *q4_row, void *q8_row_out, int n);
void iq4_nl_row_to_q8_0_shadow(const void *iq4_row, void *q8_row_out, int n);

/* Quantize a float32 vector to Q8_K blocks (for Q4_K/Q6_K matmul).
 * dst must have space for (n / 256) * sizeof(block_q8_K) bytes. */
void quantize_row_q8_K(const float *x, void *dst, int n);

/* Generic fused dot product dispatch. Returns dot(dequant(src), x) for n elements. */
float vec_dot(const void *src, const float *x, int n, gguf_type_t type);
/* Row-aware vec_dot for row-interleaved (R4/R8) weight types.
 *
 * gguf_type_row_ptr()/gguf_type_row_in_block() resolve WHICH logical row a
 * caller wants, but plan vec_dot() only ever reads row 0 of the block it is
 * handed. Handing it (base + row * row_size) from an interleaved tensor
 * therefore silently computes the wrong row for 3 of every 4 (R4) or 7 of
 * every 8 (R8) rows -- and reads a misaligned block for the rest, which can
 * even produce inf/NaN. Use this when walking individual rows of such a type:
 *   vec_dot_row(w, x, n, type, row)
 * It is identical to vec_dot() for non-interleaved types. */
float vec_dot_row(const void *base, const float *x, int n, gguf_type_t type, int row);

/* Scale-and-add: dst[i] += scale * dequant(src[i]).
 * Used for V-cache accumulation in attention with quantized V cache. */
void scale_add_q8_0_f32(float *dst, float scale, const void *src, int n);
void scale_add_q4_0_f32(float *dst, float scale, const void *src, int n);

/* FMA-style: dst[i] = dst[i] * correction + dequant(src[i]).
 * Used for the online softmax "new max" path in attention. */
void fma_scale_q8_0_f32(float *dst, float correction, const void *src, int n);
void fma_scale_q4_0_f32(float *dst, float correction, const void *src, int n);

/* ---- TQ3 KV cache helpers ---- */
/* vec_dot_tq3_f32: dot product of TQ3-encoded K with float32 Q.
 * Q must be pre-rotated with WHT forward (block=32) before calling.
 * Returns: scale * sum(q_rotated[i] * codebook[idx[i]]). */
float vec_dot_tq3_f32(const void *src, const float *q_rotated, int n);

/* scale_add_tq3_f32: dst[i] += scale * tq3_dequant(src[i]).
 * Result is in the ORIGINAL domain (inverse WHT applied). */
void scale_add_tq3_f32(float *dst, float scale, const void *src, int n);

/* fma_scale_tq3_f32: dst[i] = dst[i] * correction + tq3_dequant(src[i]).
 * Result is in the ORIGINAL domain (inverse WHT applied). */
void fma_scale_tq3_f32(float *dst, float correction, const void *src, int n);

/* ---- TQ4 KV cache helpers ---- */
/* TQ4: same WHT as TQ3 but 16-entry Lloyd-Max codebook (4 bits/value).
 * Dot product uses the codebook-lookup approach in the rotated domain
 * (no inverse WHT needed, via Parseval's theorem for orthogonal transforms).
 * Q must be pre-rotated with WHT forward (block=32) before calling vec_dot. */
float vec_dot_tq4_f32(const void *src, const float *q_rotated, int n);

/* scale_add_tq4_f32: dst[i] += scale * tq4_dequant(src[i]).
 * Result is in the ORIGINAL domain (inverse WHT applied). */
void scale_add_tq4_f32(float *dst, float scale, const void *src, int n);

/* fma_scale_tq4_f32: dst[i] = dst[i] * correction + tq4_dequant(src[i]).
 * Result is in the ORIGINAL domain (inverse WHT applied). */
void fma_scale_tq4_f32(float *dst, float correction, const void *src, int n);

/* ---- Walsh-Hadamard Transform (for KV cache rotation) ---- */
/* In-place Walsh-Hadamard transform of n elements.
 * n must be a power of 2. Self-inverting: applying twice restores the input.
 * If signs is non-NULL, apply deterministic sign flips before butterfly stages. */
void picolm_fast_ht(float *x, int n, const float *signs);

/* Apply fast_ht(nrot) to each nrot-element block in x.
 * x has head_dim elements. head_dim % nrot == 0. */
void picolm_hadamard_transform(float *x, int head_dim, int nrot);

/* GELU lookup table: 256K entries for F16->GELU mapping */
extern uint16_t picolm_gelu_table[65536];
void picolm_init_gelu_table(void);
/* NEON-accelerated GELU using lookup table (same formula as GPT-2 gelu) */
void picolm_gelu_table_f32(float *x, int size);
void picolm_gelu_f32(float *x, int size);

/* Vectorised exp(x) for 8 floats (AVX/FMA builds only; |x| clamped to ~+-88 so the
 * result stays a normal float). Returns 0 when the build has no FMA.
 * Used by the pointwise activations (GELU here, SiLU in tensor.c) that used to call
 * scalar expf/tanhf once per element on the prefill hot path. */
#if (defined(PICOLM_AVX512) || defined(PICOLM_AVX2)) && defined(__FMA__)
#include <immintrin.h>
__m256 picolm_exp256_ps(__m256 x);
#define PICOLM_HAVE_EXP256 1
#endif

/* PowerPC paired-singles intrinsics (Broadway/WiiLinux).
 * Requires:
 *   - PowerPC 750-class CPU with paired-singles support
 *   - GQR7 set to s8 type via ps_init_gqr() (called from fp16_table_init)
 *   - HID2[LSQE] enabled by kernel (non-float psq_l is supervisor-only)
 *   - Pointers 2-byte aligned (block_q8_0 qs at offset 2 is sufficient for
 *     psq_l's 2-byte int8-pair loads; 8-byte alignment is NOT required) */
#ifdef PICOLM_PS
/* These are the paired-singles asm kernels. Every one of them is marked
 * noinline, which is NOT an optimisation hint but a compiler-workaround:
 *
 *   GCC 4.3.2 (WiiLinux) ICEs in reg_overlap_mentioned_p (rtlanal.c:1398)
 *   when these asm statements are inlined into a caller at -O2, e.g.
 *   "quant.c: In function 'vec_dot2_q8_0_f32': internal compiler error:
 *   in reg_overlap_mentioned_p".
 *
 * The trigger is register pressure from the paired-singles asm operand list
 * (10 earlyclobber "=&f" doubles plus pointer operands) meeting the inliner --
 * it is an inliner/RA bug, not a problem with the asm text itself: the
 * identical code builds at -O0 and -O1, and builds fine as a non-inlined
 * function at -O2. These kernels run once per 32-element block, so the call
 * overhead is noise; correctness on the target compiler wins.
 *
 * If you add another paired-singles asm kernel, mark it the same way. */
#define PICOLM_PS_NOINLINE PICOLM_NOINLINE

void ps_init_gqr(void);
PICOLM_PS_NOINLINE void ps_q8_row_dot(const int8_t *x, const int8_t *w, int n, float *out);
PICOLM_PS_NOINLINE void ps_q8_row_dot2(const float *x, const int8_t *w0, const int8_t *w1, int n, float *out);
PICOLM_PS_NOINLINE void ps_axpy_s8(float *xb, const int8_t *v, float a, int octets);
void vec_dot2_q8_0_f32(const void *w0, const void *w1, const float *x, int n, float *out);
/* ps_qk4_s8: exported but currently dead code -- flat int8 KV cache not used */
PICOLM_PS_NOINLINE void ps_qk4_s8(const float *q, const int8_t *k, int kv_dim, int pairs, float *out);
#endif

#ifdef __cplusplus
}
#endif

#endif /* QUANT_H */
