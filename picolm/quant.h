#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

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
    GGUF_TYPE_Q8_K = 29,  /* internal: Q8_K intermediate quantization */
    GGUF_TYPE_Q6_K       = 14,
    GGUF_TYPE_Q4_0_4_4   = 31,  /* 4-row interleaved Q4_0, blocklen=4 (pre-repacked) */
    GGUF_TYPE_Q4_0_4_8   = 32,  /* 4-row interleaved Q4_0, blocklen=8 (pre-repacked, I8MM target) */
    GGUF_TYPE_Q4_0_8_8   = 33,  /* 8-row interleaved Q4_0 (pre-repacked, AVX2) */
    GGUF_TYPE_Q4I_0_8_8  = 34,  /* 8-row pre-dequantized int8 Q4_0 (dpbusd lane order, AVX-512) */
    GGUF_TYPE_BF16      = 30,  /* Brain Float 16 (GGUF type 30) */
    GGUF_TYPE_IQ4_NL    = 20,  /* Non-linear 4-bit quant (LUT-based, same layout as Q4_0) */
    GGUF_TYPE_IQ2_K_R4  = 337, /* 4-row interleaved IQ2_K (repacked, AVX-512/AVX2 target) */
    GGUF_TYPE_Q1_0       = 41,  /* 1-bit sign + scale, 128 values/block */
    GGUF_TYPE_Q2_0       = 42,  /* 2-bit values + scale, 128 values/block */
    GGUF_TYPE_Q6_0       = 133, /* 6-bit values + FP16 scale, 32 values/block (legacy GGML) */
    GGUF_TYPE_Q4_0_R8    = 202, /* 8-row interleaved Q4_0 (GGUF type 202, llama.cpp ik branch) */
    GGUF_TYPE_Q8_0_R8    = 203, /* 8-row interleaved Q8_0 (GGUF type 203) */
    GGUF_TYPE_Q6_K_R4    = 214, /* 4-row interleaved Q6_K (GGUF type 214, llama.cpp ik branch) */
    GGUF_TYPE_Q8_K_R8    = 399, /* 8-row interleaved Q8_K (GGUF type 399, llama.cpp ik branch) */
} gguf_type_t;

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
/* Compile-time check: block_iq4_nl must match block_q4_0 layout (18 bytes) */
#if !defined(_MSC_VER)
typedef char __compiletime_assert_iq4_nl_size[(sizeof(block_iq4_nl) == sizeof(block_q4_0)) ? 1 : -1] __attribute__((unused));
#endif
/* MSVC: checked at link time by sizeof consistency, no compile-time assert needed */

/* IQ4_NL dequantization lookup table (16 entries, int8) */
/* Derived from llama.cpp ggml-common.h GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16) */
extern const int8_t kvalues_iq4nl[16];

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

/* IQ2_K non-linear values table (8 entries, int8)
 * Derived from llama.cpp ggml-common.h GGML_TABLE_BEGIN(int8_t, iq2nl_values, 8)
 * Normal table (indices 0-3): {-31, -13, 1, 17}
 * Shifted table (indices 4-7): {-26, -8, 6, 22}
 * The extra bits select which table per sub-block half. */
extern const int8_t iq2nl_values[8];

/* Q4_1 block: 32 weights (old GGML format, used by some GGUF models)
 * Layout: half d (scale), half m (min), uchar qs[16] (nibbles)
 * Dequant: val = qs[j] * d + m  (unsigned nibble, no sign extension) */
typedef struct PICOLM_PACKED_ATTR {
    uint16_t d;          /* scale (FP16) */
    uint16_t m;          /* min (FP16) */
    uint8_t  qs[16];     /* 4-bit quantized values */
} block_q4_1;            /* 20 bytes */

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
/* IQ4_NL * Q8_0 dot product: IQ4_NL weights with pre-quantized Q8_0 input */
float vec_dot_iq4_nl_q8_0(const void *src_iq4, const void *src_q8, int n);
/* IQ4_NL * F32 dot product: fused dequant + dot (scalar fallback) */
float vec_dot_iq4_nl_f32(const void *src_iq4, const float *x, int n);
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
/* Quantize 8 rows of F32 to Q8_0_R8 interleaved format. */
void quantize_row_q8_0_r8(const float *x, void *dst, int n);
/* Dequantize 8 rows of Q8_0_R8 to F32. dst must hold 8*n floats. */
void dequantize_row_q8_0_r8(const void *src, float *dst, int n);
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
/* Convert F32 activations to Q8_2 blocks (with row sum).
 * dst must have space for (n/32) * sizeof(block_q8_2) bytes. */
void quantize_row_q8_2(const float *x, void *dst, int n);
/* Convert F32 activations to Q8_2_x4 interleaved format (4 rows).
 * dst must have space for (n/32) * sizeof(block_q8_2_x4) bytes per group. */
void quantize_mat_q8_2_x4(const float *x, void *dst, int n, int row_stride);

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
/* IQ2_K_R4 x Q8_K batched GEMM (AVX2). */
int sgemm_iq2_k_r4_q8_k_avx2(int nrows, int ncols, int k,
                               const void *vx, const void *vy,
                               float *out, size_t bs,
                               int ith, int nth);
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

#ifdef __cplusplus
}
#endif

#endif /* QUANT_H */
