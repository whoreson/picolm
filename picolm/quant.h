#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>
#include <stddef.h>

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
float fp16_to_fp32(uint16_t h);
float fp16_to_fp32_lookup(uint16_t h);
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
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
#  define PICOLM_I8MM 1
#endif

/* --- ARM NEON --- */
/* Guard against CUDA device compilation: nvcc cannot handle arm_neon.h */
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && !defined(__CUDACC__)
#  define PICOLM_NEON 1
#  include <arm_neon.h>
static inline float vaddvq_f32_compat(float32x4_t v) {
#  if defined(__aarch64__)
    return vaddvq_f32(v);
#  else
    float32x2_t r = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(r, r), 0);
#  endif
}
#endif

/* --- ARM NEON detection --- */
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && !defined(__CUDACC__)
#  define PICOLM_NEON 1
/* Hardware FP16 vector conversion (ARMv8.2-A asimdhp).
 * On older NEON hardware (ARMv8.0/8.1), this path is not available and
 * the scalar fp16_to_fp32 lookup table is used instead. */
#  if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#    define PICOLM_FP16_HW 1
#    include <arm_acle.h>
#  endif
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

/* AVX512 implies AVX2 + AVX + SSE3 + SSE2 */
#if defined(__AVX512F__)
#  define PICOLM_AVX512 1
/* VNNI with 256-bit (VL) enables dpbssd/dpbusd int8 MAC instructions */
#  if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#    define PICOLM_VNNI 1
#  endif
#  define PICOLM_AVX2   1
#  define PICOLM_AVX    1
#  define PICOLM_SSE3   1
#  define PICOLM_SSE2   1
/* AVX2 implies AVX + SSE3 + SSE2 */
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

/* PPC Altivec (VMX) - separate from x86 chain */
#if defined(__ALTIVEC__)
#  define PICOLM_ALTIVEC 1
#  include <altivec.h>
#  undef bool
#  undef pixel
#  undef vec_add
#endif

/* Include x86 SIMD header once for any x86 SIMD level.
 * <immintrin.h> is the modern umbrella header (GCC 4.7+).
 * Older compilers (e.g. GCC 4.2 on Mac OS X 10.6) need individual headers. */
#ifdef PICOLM_SSE2
#  if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 7)
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
    __m128 hi = _mm_cvtph_ps(_mm_loadu_si128((const __m128i *)(p + 4)));
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
    float32x4_t r = vdupq_n_f32(fp16_to_fp32_lookup(p[0]));
    r = vsetq_lane_f32(fp16_to_fp32_lookup(p[1]), r, 1);
    r = vsetq_lane_f32(fp16_to_fp32_lookup(p[2]), r, 2);
    r = vsetq_lane_f32(fp16_to_fp32_lookup(p[3]), r, 3);
    return r;
}
#endif

#ifdef PICOLM_SSE2
/* Convert 4 FP16 to 4 FP32 in SSE register */
static inline __m128 fp16x4_to_fp32_inline(const uint16_t *p) {
#ifdef __F16C__
    return _mm_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
#else
    /* No F16C: use lookup table */
    return _mm_set_ps(
        fp16_to_fp32_lookup(p[3]), fp16_to_fp32_lookup(p[2]), fp16_to_fp32_lookup(p[1]), fp16_to_fp32_lookup(p[0]));
#endif
}
#endif

/* AVX-512 specific helpers */
#ifdef PICOLM_AVX512
/* Convert 16 FP16 to 16 FP32 in AVX-512 register */
static inline __m512 fp16x16_to_fp32_inline(const uint16_t *p) {
#ifdef __F16C__
    __m128 a = _mm_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
    __m128 b = _mm_cvtph_ps(_mm_loadu_si128((const __m128i *)(p + 4)));
    __m128 c = _mm_cvtph_ps(_mm_loadu_si128((const __m128i *)(p + 8)));
    __m128 d = _mm_cvtph_ps(_mm_loadu_si128((const __m128i *)(p + 12)));
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
    GGUF_TYPE_BF16      = 30,  /* Brain Float 16 (GGUF type 30) */
    GGUF_TYPE_Q1_0       = 41,  /* 1-bit sign + scale, 128 values/block */
    GGUF_TYPE_Q2_0       = 42,  /* 2-bit values + scale, 128 values/block */
} gguf_type_t;

/* Q4_K block: 256 weights in 144 bytes */
#pragma pack(push, 1)
typedef struct {
    uint16_t d;          /* super-block scale (FP16) */
    uint16_t dmin;       /* super-block min   (FP16) */
    uint8_t  scales[12]; /* packed 6-bit scales and mins for 8 sub-blocks */
    uint8_t  qs[128];    /* 4-bit quantized values (256 nibbles) */
} block_q4_K;            /* 144 bytes */
#pragma pack(pop)

/* Q8_K block: 256 weights, used for intermediate quantization in Q4_K/Q6_K matmul */
/* d: float, qs: 256 int8, bsums: 16 int16 (sum of quants in groups of 16) */
typedef struct {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
} block_q8_K;            /* 4 + 256 + 32 = 292 bytes */

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
typedef struct {
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
typedef struct {
    uint16_t d[8];      /* 8 FP16 deltas, one per row */
    uint8_t  qs[128];   /* interleaved nibble-bytes (8 rows x 16 bytes, XOR'd with 0x88) */
} block_q4_0x8;         /* 144 bytes */
#pragma pack(pop)

/* Q3_K block: 256 weights in 110 bytes */
#pragma pack(push, 1)
typedef struct {
    uint16_t d;          /* super-block scale (FP16) */
    uint8_t  qs[64];     /* 2-bit low quants */
    uint8_t  hmask[32];  /* high bit mask */
    uint8_t  scales[12]; /* packed 6-bit scales */
} block_q3_K;            /* 110 bytes */
#pragma pack(pop)

/* Q2_K block: 256 weights in 84 bytes */
#pragma pack(push, 1)
typedef struct {
    uint8_t  scales[16]; /* packed scales and mins (4-bit each) */
    uint8_t  qs[64];     /* 2-bit quantized values */
    uint16_t d;          /* super-block scale (FP16) */
    uint16_t dmin;       /* super-block min   (FP16) */
} block_q2_K;            /* 84 bytes */
#pragma pack(pop)

/* Q8_0 block: 32 weights */
#pragma pack(push, 1)
typedef struct {
    uint16_t d;          /* scale (FP16) */
    int8_t   qs[32];     /* 8-bit quantized values */
} block_q8_0;            /* 34 bytes */
#pragma pack(pop)

/* Q5_K block: 256 weights in 176 bytes */
#pragma pack(push, 1)
typedef struct {
    uint16_t d;          /* super-block scale (FP16) */
    uint16_t dm;         /* super-block min   (FP16) */
    uint8_t  scales[12]; /* packed 6-bit scales+mins (get_scale_min_k4) */
    uint8_t  qh[32];     /* high bit (1 per quant) */
    uint8_t  qs[128];    /* low 4 bits (2 per byte) */
} block_q5_K;            /* 176 bytes */
#pragma pack(pop)

/* Q6_K block: 256 weights in 210 bytes */
#pragma pack(push, 1)
typedef struct {
    uint8_t  ql[128];    /* low 4 bits of quants */
    uint8_t  qh[64];     /* high 2 bits of quants */
    int8_t   scales[16]; /* 8-bit scales */
    uint16_t d;          /* super-block scale (FP16) */
} block_q6_K;            /* 210 bytes */
#pragma pack(pop)

/* Q4_0 block: 32 weights */
#pragma pack(push, 1)
typedef struct {
    uint16_t d;          /* scale (FP16) */
    uint8_t  qs[16];     /* 4-bit quantized values */
} block_q4_0;            /* 18 bytes */

/* Q4_1 block: 32 weights (old GGML format, used by some GGUF models)
 * Layout: half d (scale), half m (min), uchar qs[16] (nibbles)
 * Dequant: val = qs[j] * d + m  (unsigned nibble, no sign extension) */
typedef struct {
    uint16_t d;          /* scale (FP16) */
    uint16_t m;          /* min (FP16) */
    uint8_t  qs[16];     /* 4-bit quantized values */
} block_q4_1;            /* 20 bytes */

/* Q1_0 block: 128 weights (1-bit sign + FP16 scale, 18 bytes)
 * Dequant: val[j] = (bit[j] ? +d : -d) */
typedef struct {
    uint16_t d;          /* scale (FP16) = mean(|values|) */
    uint8_t  qs[16];     /* 128 sign bits (1 bit per value) */
} block_q1_0;            /* 18 bytes */

/* Q2_0 block: 128 weights (2-bit values + FP16 scale, 34 bytes)
 * Dequant: val[j] = ((qs[j] - 1) * d), {0,1,2,3} -> {-d, 0, +d, +2d} */
typedef struct {
    uint16_t d;          /* scale (FP16) = max(|values|) */
    uint8_t  qs[32];     /* 128 values * 2 bits each */
} block_q2_0;            /* 34 bytes */

#pragma pack(pop)

/* ---- FP16 conversion ---- */
uint16_t fp32_to_fp16(float f);

/* ---- Dequantize a row of weights into float output buffer ---- */
void dequantize_row_q4_K(const void *src, float *dst, int n);
void dequantize_row_q3_K(const void *src, float *dst, int n);
void dequantize_row_q2_K(const void *src, float *dst, int n);
void dequantize_row_q8_0(const void *src, float *dst, int n);
void dequantize_row_q6_K(const void *src, float *dst, int n);
void dequantize_row_q5_K(const void *src, float *dst, int n);
void dequantize_row_q4_0(const void *src, float *dst, int n);
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
float vec_dot_f32_f32(const void *src, const float *x, int n);
float vec_dot_q8_0_f32(const void *src, const float *x, int n);
float vec_dot_q4_0_f32(const void *src, const float *x, int n);
float vec_dot_q4_1_f32(const void *src, const float *x, int n);
/* fp16-fp32 dot product: sum of fp16_to_fp32(k[i]) * x[i] */
float vec_dot_f16_f32(const void *src, const float *x, int n);
float vec_dot_q8_0_q8_0(const void *qx, const void *qw, int n);
float vec_dot_q8_0_q8_0_deltas(const void *qx, const float *qx_d, const void *qw, int n);
/* Q1_0 * Q8_0 dot product: 1-bit weights with pre-quantized Q8_0 input */
float vec_dot_q1_0_q8_0(const void *src_q1, const void *src_q8, int n);
/* Q2_0 * Q8_0 dot product: 2-bit weights with pre-quantized Q8_0 input */
float vec_dot_q2_0_q8_0(const void *src_q2, const void *src_q8, int n);
/* Q4_0 * Q8_0 dot product: Q4_0 weights with pre-quantized Q8_0 input */
float vec_dot_q4_0_q8_0(const void *src_q4, const void *src_q8, int n);
/* Q4_K * Q8_K dot product: Q4_K weights with pre-quantized Q8_K input */
float vec_dot_q4_K_q8_K(const void *src_q4, const void *src_q8, int n);
/* Q4_0_4_4 interleaved weights x Q8_0 input (blocklen=4): processes nrows (multiple of 4) */
void vec_dot_q4_0x4_q8_0(const void *vx, const void *wy, int n, float *out, int nrows);
/* Q4_0_4_8 interleaved weights x Q8_0 input (blocklen=8): processes nrows (multiple of 4) */
void vec_dot_q4_0x4_4x8_q8_0(const void *vx, const void *wy, int n, float *out, int nrows);
/* Q4_0_8x8 interleaved weights x Q8_0 input (AVX2): processes nrows (multiple of 8) simultaneously */
void vec_dot_q4_0x8_q8_0_avx2(const void *vx, const void *wy, int n, float *out, int nrows);
/* Repack standard Q4_0 weights to Q4_0_8x8 interleaved format (for AVX2).
 * dst must have the same size as src (1:1 byte mapping, just reordered). */
void repack_q4_0_to_q4_0x8(const void *src, void *dst, int nrows, int ncols);
/* Repack standard Q4_0 weights to Q4_0_4x4 interleaved format (for NEON dotprod). */
void repack_q4_0_to_q4_0x4(const void *src, void *dst, int nrows, int ncols);

/* Quantize a float32 vector to Q8_0 blocks in-place or to a separate buffer.
 * dst must have space for (n / 32) * sizeof(block_q8_0) bytes. */
void quantize_row_q8_0(const float *x, void *dst, int n);

/* Quantize a float32 vector to Q4_0 blocks.
 * dst must have space for (n / 32) * sizeof(block_q4_0) bytes. */
void quantize_row_q4_0(const float *x, void *dst, int n);

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

#endif /* QUANT_H */
