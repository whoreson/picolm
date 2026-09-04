/* ================================================================
 * TODO/WIP: Interleaved Q4_0_8x8 x Q8_0_4x8 GEMM
 * ================================================================
 * STATUS: WIP (WORK IN PROGRESS) -- NOT YET FUNCTIONAL
 *
 * Port of llama.cpp's interleaved GEMM infrastructure:
 *   Source: ~/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp
 *   Routine: gemm_q4_b32_8x8_q8_0_lut_avx (AVX-512 path, ~line 641)
 *   Routine: gemv_q4_b32_8x8_q8_0_lut_avx (AVX2 path, ~line 522)
 *   Repack:  ~/llama.cpp/ggml/src/ggml-cpu/repack.cpp
 *            ggml_quantize_mat_q8_0_4x8_generic (~line 173)
 *
 * BUGS FIXED (2025-09-04, based on code analysis):
 *   0. Unterminated comment fragment (line 35) -- FIXED: removed orphan text
 *   1. g_nibble_lut broadcast -- FIXED: use _mm_set_epi8 + permute2f128
 *      (was _mm256_set1_epi64x of raw pointer, only loaded 8 of 16 entries)
 *   2. AVX2 acc[4] OOB write -- FIXED: expanded to acc[16], compute all 4
 *      activation row groups (was only computing 2, missing rows 2-3)
 *   3. AVX-512 rs unreplicated -- FIXED: use GGML_F32Cx16_REPEAT_LOAD pattern
 *      (was _mm512_castps256_ps512 leaving upper 256 bits undefined)
 *   4. Activation delta over-read -- FIXED: use _mm_loadl_epi64 for d[4]
 *      (was _mm_loadu_si128 reading 16 bytes from 8-byte field)
 *   5. _mm512_permute_ps wrong intrinsic -- FIXED: replaced with
 *      _mm512_cvtepi32_ps (llama.cpp style: convert all 16 int32 at once)
 *
 * REMAINING TODO:
 *   6. dpbusd accumulator straightening shuffles need verification against
 *      llama.cpp's mul_sum_i8_pairs_acc_int32x16. The blend/permute pattern
 *      was ported structurally but not exhaustively hand-traced.
 *
 * TO ENABLE (when bug #6 is verified):
 *   Set PICOLM_Q4_0x8_GEMM=1 in tensor.c to bypass vec_dot path.
 *
 * Performance target: 2-3x over vec_dot Q4_0_8_8 path (18 tok/s -> 40+ tok/s)
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>
#include "quant.h"

/* ============================================================
 * Helpers: sign-trick dpbusd MAC
 * ============================================================ */
/* 512-bit dpbusd MAC: sign trick + dpbusd or maddubs fallback.
 * Note: _mm512_sign_epi8 is NOT available in gcc 15 headers.
 * We implement it via two _mm256_sign_epi8 + merge. */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
static inline __m512i dpbusd_512(const __m512i acc, const __m512i x, const __m512i y) {
    /* sign(x,x) = x (identity), sign(y,x) via 256-bit halves */
    __m256i xlo = _mm512_castsi512_si256(x);
    __m256i xhi = _mm512_extracti32x8_epi32(x, 1);
    __m256i ylo = _mm512_castsi512_si256(y);
    __m256i yhi = _mm512_extracti32x8_epi32(y, 1);
    __m256i sylo = _mm256_sign_epi8(ylo, xlo);
    __m256i syhi = _mm256_sign_epi8(yhi, xhi);
    __m512i sy = _mm512_inserti32x8(_mm512_castsi256_si512(sylo), syhi, 1);
    return _mm512_dpbusd_epi32(acc, x, sy);
}
#elif defined(__AVX512BW__) && defined(__AVX512DQ__)
/* Fallback: maddubs + madd (no VNNI) */
static inline __m512i dpbusd_512(const __m512i acc, const __m512i x, const __m512i y) {
    __m256i xlo = _mm512_castsi512_si256(x);
    __m256i xhi = _mm512_extracti32x8_epi32(x, 1);
    __m256i ylo = _mm512_castsi512_si256(y);
    __m256i yhi = _mm512_extracti32x8_epi32(y, 1);
    __m256i sylo = _mm256_sign_epi8(ylo, xlo);
    __m256i syhi = _mm256_sign_epi8(yhi, xhi);
    __m512i sy = _mm512_inserti32x8(_mm512_castsi256_si512(sylo), syhi, 1);
    const __m512i dot = _mm512_maddubs_epi16(x, sy);
    return _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_set1_epi16(1), dot));
}
#endif

/* 256-bit: 8 int32 accumulators from 32 int8 x 32 int8 */
static inline __m256i dpbusd_256(const __m256i acc, const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, ax, sy);
#elif defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(acc, ax, sy);
#else
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi16(1), dot));
#endif
}

/* ============================================================
 * Helpers: FP16 -> FP32
 * ============================================================ */

/* Load 8 FP16 values (16 bytes) -> 8 FP32.
 * For block_q4_0x8.d[8] (weights, 16 bytes). */
static inline __m256 fp16x8_to_fp32(const uint16_t *d) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)d));
}

/* Load 4 FP16 values (8 bytes) -> 4 FP32 in low lane of __m256.
 * For block_q8_0x4.d[4] (activations, 8 bytes).
 * FIXED: was _mm_loadu_si128 reading 16 bytes (OOB into qs field).
 * Now uses _mm_loadl_epi64 to read exactly 8 bytes, then _mm256_cvtph_ps. */
static inline __m256 fp16x4_to_fp32(const uint16_t *d) {
    return _mm256_cvtph_ps(_mm_loadl_epi64((const __m128i*)d));
}

#if defined(__AVX512F__)
/* Load 16 FP16 values (32 bytes, two 16-byte sources) -> 16 FP32.
 * For two block_q4_0x8.d[8] pairs (weights). */
static inline __m512 fp16x16_to_fp32(const uint16_t *d0, const uint16_t *d1) {
    return _mm512_cvtph_ps(_mm256_set_m128i(
        _mm_loadu_si128((const __m128i*)d1),
        _mm_loadu_si128((const __m128i*)d0)));
}

/* GGML_F32Cx16_REPEAT_LOAD: load 4 FP16 (8 bytes), convert to 4 FP32,
 * replicate to all 16 lanes of __m512.
 * For block_q8_0x4.d[4] (activations).
 * This is the llama.cpp pattern for row_scale_f32. */
#define PICOLM_F32Cx16_REPEAT_LOAD(x) \
    _mm512_cvtph_ps(_mm256_set_m128i(x, x))
#endif

/* ============================================================
 * Helpers: 4-bit nibble dequant via LUT
 * ============================================================ */

/* Sign-extension LUT: maps nibble (0..15) to signed byte (-8..7).
 * Entry 0->0, 1->1, ..., 7->7, 8->-1, 9->-2, ..., 15->-8.
 * Note: _mm_set_epi8 is high-byte-first, so args are reversed. */
static const int8_t g_nibble_lut_bytes[16] =
    {-1,-2,-3,-4,-5,-6,-7,-8, 7,6,5,4,3,2,1,0};

#if defined(__AVX512BW__)
static inline __m512i nibble_to_i8_512(const __m512i raw, const __m512i lut) {
    return _mm512_shuffle_epi8(lut, _mm512_and_si512(raw, _mm512_set1_epi8(0x0F)));
}
static inline __m512i nibble_to_i8_hi_512(const __m512i raw, const __m512i lut) {
    return _mm512_shuffle_epi8(lut,
        _mm512_and_si512(_mm512_srli_epi16(raw, 4), _mm512_set1_epi8(0x0F)));
}
#endif

#if defined(__AVX2__)
static inline __m256i nibble_to_i8_256(const __m256i raw, const __m256i lut) {
    return _mm256_shuffle_epi8(lut, _mm256_and_si256(raw, _mm256_set1_epi8(0x0F)));
}
static inline __m256i nibble_to_i8_hi_256(const __m256i raw, const __m256i lut) {
    return _mm256_shuffle_epi8(lut,
        _mm256_and_si256(_mm256_srli_epi16(raw, 4), _mm256_set1_epi8(0x0F)));
}
#endif

/* ============================================================
 * Build the LUT properly (FIX for Bug #1)
 *
 * llama.cpp reference:
 *   __m256i signextendlut = _mm256_castsi128_si256(
 *       _mm_set_epi8(-1,-2,-3,-4,-5,-6,-7,-8,7,6,5,4,3,2,1,0));
 *   signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
 *
 * _mm_set_epi8 is high-byte-first: arg[0] goes to byte[15], arg[15] to byte[0].
 * So (-1,-2,...,0) produces bytes [0,1,2,3,4,5,6,7,-8,-7,-6,-5,-4,-3,-2,-1].
 * This is the correct LUT: index 0->0, 1->1, ..., 7->7, 8->-8, ..., 15->-1.
 *
 * The old code used _mm256_set1_epi64x(*(uint64_t*)g_nibble_lut) which
 * only loaded the first 8 bytes {-1,-2,-3,-4,-5,-6,-7,-8} and broadcast
 * them, missing the positive half {7,6,5,4,3,2,1,0}.
 * ============================================================ */
static inline __m256i build_lut256(void) {
    __m128i lut128 = _mm_set_epi8(-1,-2,-3,-4,-5,-6,-7,-8,7,6,5,4,3,2,1,0);
    return _mm256_permute2f128_si256(_mm256_castsi128_si256(lut128),
                                     _mm256_castsi128_si256(lut128), 0);
}

#if defined(__AVX512BW__)
static inline __m512i build_lut512(void) {
    __m256i lut256 = build_lut256();
    return _mm512_inserti32x8(_mm512_castsi256_si512(lut256), lut256, 1);
}
#endif

/* ============================================================
 * AVX-512 path: 16x16 output tiles
 * ============================================================ */
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
static int sgemm_q4x8_q8x4_avx512(
        int k, const block_q4_0x8 *bp_start,
        const block_q8_0x4 *ap_start,
        float *s, size_t bs, int nr, int nc)
{
    const int nb = k / 32;
    const __m512i lut512 = build_lut512();
    const __m256i reorder = _mm256_set_epi32(3,2,1,0,7,6,5,4);
    int y = 0;
    const int anr = nr - nr % 16;
    const int anc = nc - nc % 16;

    for (; y < anr / 4; y += 4) {
        const block_q8_0x4 *ap[4];
        ap[0] = ap_start + (y * nb);
        for (int i = 0; i < 3; i++) ap[i+1] = ap[i] + nb;

        for (int x = 0; x < anc / 8; x += 2) {
            const block_q4_0x8 *bp0 = bp_start + (x * nb);
            const block_q4_0x8 *bp1 = bp_start + ((x+1) * nb);

            __m512 acc[16];
            for (int i = 0; i < 16; i++) acc[i] = _mm512_setzero_ps();

            for (int b = 0; b < nb; b++) {
                /* Load 16 rows of Q4_0 nibbles (2x block_q4_0x8) */
                const __m256i w00 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs));
                const __m256i w10 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs+32));
                const __m256i w01 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs+64));
                const __m256i w11 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs+96));
                const __m256i w20 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs));
                const __m256i w30 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs+32));
                const __m256i w21 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs+64));
                const __m256i w31 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs+96));

                /* Blend: separate even/odd row groups */
                const __m256i we00 = _mm256_blend_epi32(w00, _mm256_permutevar8x32_epi32(w10,reorder), 240);
                const __m256i wo00 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(w00,reorder), w10, 240);
                const __m256i we01 = _mm256_blend_epi32(w01, _mm256_permutevar8x32_epi32(w11,reorder), 240);
                const __m256i wo01 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(w01,reorder), w11, 240);
                const __m256i we10 = _mm256_blend_epi32(w20, _mm256_permutevar8x32_epi32(w30,reorder), 240);
                const __m256i wo10 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(w20,reorder), w30, 240);
                const __m256i we11 = _mm256_blend_epi32(w21, _mm256_permutevar8x32_epi32(w31,reorder), 240);
                const __m256i wo11 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(w21,reorder), w31, 240);

                /* Merge to 512-bit */
                const __m512i we010 = _mm512_inserti32x8(_mm512_castsi256_si512(we00), we10, 1);
                const __m512i wo010 = _mm512_inserti32x8(_mm512_castsi256_si512(wo00), wo10, 1);
                const __m512i we011 = _mm512_inserti32x8(_mm512_castsi256_si512(we01), we11, 1);
                const __m512i wo011 = _mm512_inserti32x8(_mm512_castsi256_si512(wo01), wo11, 1);

                /* 4-bit -> 8-bit via LUT (4 chunks per group: 2 low + 2 high) */
                const __m512i rl0 = nibble_to_i8_512(we010, lut512);
                const __m512i rl1 = nibble_to_i8_512(we011, lut512);
                const __m512i rl2 = nibble_to_i8_hi_512(we010, lut512);
                const __m512i rl3 = nibble_to_i8_hi_512(we011, lut512);
                const __m512i ro0 = nibble_to_i8_512(wo010, lut512);
                const __m512i ro1 = nibble_to_i8_512(wo011, lut512);
                const __m512i ro2 = nibble_to_i8_hi_512(wo010, lut512);
                const __m512i ro3 = nibble_to_i8_hi_512(wo011, lut512);

                /* Shuffle: dup each 4-int8 group for dpbusd (16 lanes) */
                const __m512i rl0s1 = _mm512_shuffle_epi32(rl0, (_MM_PERM_ENUM)136);
                const __m512i rl1s1 = _mm512_shuffle_epi32(rl1, (_MM_PERM_ENUM)136);
                const __m512i rl2s1 = _mm512_shuffle_epi32(rl2, (_MM_PERM_ENUM)136);
                const __m512i rl3s1 = _mm512_shuffle_epi32(rl3, (_MM_PERM_ENUM)136);
                const __m512i ro0s1 = _mm512_shuffle_epi32(ro0, (_MM_PERM_ENUM)136);
                const __m512i ro1s1 = _mm512_shuffle_epi32(ro1, (_MM_PERM_ENUM)136);
                const __m512i ro2s1 = _mm512_shuffle_epi32(ro2, (_MM_PERM_ENUM)136);
                const __m512i ro3s1 = _mm512_shuffle_epi32(ro3, (_MM_PERM_ENUM)136);
                const __m512i rl0s2 = _mm512_shuffle_epi32(rl0, (_MM_PERM_ENUM)221);
                const __m512i rl1s2 = _mm512_shuffle_epi32(rl1, (_MM_PERM_ENUM)221);
                const __m512i rl2s2 = _mm512_shuffle_epi32(rl2, (_MM_PERM_ENUM)221);
                const __m512i rl3s2 = _mm512_shuffle_epi32(rl3, (_MM_PERM_ENUM)221);
                const __m512i ro0s2 = _mm512_shuffle_epi32(ro0, (_MM_PERM_ENUM)221);
                const __m512i ro1s2 = _mm512_shuffle_epi32(ro1, (_MM_PERM_ENUM)221);
                const __m512i ro2s2 = _mm512_shuffle_epi32(ro2, (_MM_PERM_ENUM)221);
                const __m512i ro3s2 = _mm512_shuffle_epi32(ro3, (_MM_PERM_ENUM)221);

                /* Weight column scales (16 rows = 2 blocks x 8 FP16 each) */
                __m512 cs = fp16x16_to_fp32(bp0[b].d, bp1[b].d);

                /* Process 4 activation row groups (16 rows) */
                for (int rp = 0; rp < 4; rp++) {
                    const block_q8_0x4 *a = ap[rp];

                    /* Load Q8_0 activations (128 int8 = 4 x 32) */
                    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a[b].qs));
                    __m256i a1 = _mm256_loadu_si256((const __m256i*)(a[b].qs+32));
                    __m256i a2 = _mm256_loadu_si256((const __m256i*)(a[b].qs+64));
                    __m256i a3 = _mm256_loadu_si256((const __m256i*)(a[b].qs+96));

                    /* Split into halves, expand to 512 (dup each 16-byte half to 32-byte) */
                    __m256i a0l = _mm256_permute2f128_si256(a0, a0, 0);
                    __m256i a0h = _mm256_permute2f128_si256(a0, a0, 17);
                    __m256i a1l = _mm256_permute2f128_si256(a1, a1, 0);
                    __m256i a1h = _mm256_permute2f128_si256(a1, a1, 17);
                    __m256i a2l = _mm256_permute2f128_si256(a2, a2, 0);
                    __m256i a2h = _mm256_permute2f128_si256(a2, a2, 17);
                    __m256i a3l = _mm256_permute2f128_si256(a3, a3, 0);
                    __m256i a3h = _mm256_permute2f128_si256(a3, a3, 17);

                    const __m512i v00 = _mm512_inserti32x8(_mm512_castsi256_si512(a0l), a0l, 1);
                    const __m512i v01 = _mm512_inserti32x8(_mm512_castsi256_si512(a1l), a1l, 1);
                    const __m512i v02 = _mm512_inserti32x8(_mm512_castsi256_si512(a2l), a2l, 1);
                    const __m512i v03 = _mm512_inserti32x8(_mm512_castsi256_si512(a3l), a3l, 1);
                    const __m512i v20 = _mm512_inserti32x8(_mm512_castsi256_si512(a0h), a0h, 1);
                    const __m512i v21 = _mm512_inserti32x8(_mm512_castsi256_si512(a1h), a1h, 1);
                    const __m512i v22 = _mm512_inserti32x8(_mm512_castsi256_si512(a2h), a2h, 1);
                    const __m512i v23 = _mm512_inserti32x8(_mm512_castsi256_si512(a3h), a3h, 1);

                    const __m512i v00s1 = _mm512_shuffle_epi32(v00, (_MM_PERM_ENUM)136);
                    const __m512i v01s1 = _mm512_shuffle_epi32(v01, (_MM_PERM_ENUM)136);
                    const __m512i v02s1 = _mm512_shuffle_epi32(v02, (_MM_PERM_ENUM)136);
                    const __m512i v03s1 = _mm512_shuffle_epi32(v03, (_MM_PERM_ENUM)136);
                    const __m512i v20s1 = _mm512_shuffle_epi32(v20, (_MM_PERM_ENUM)136);
                    const __m512i v21s1 = _mm512_shuffle_epi32(v21, (_MM_PERM_ENUM)136);
                    const __m512i v22s1 = _mm512_shuffle_epi32(v22, (_MM_PERM_ENUM)136);
                    const __m512i v23s1 = _mm512_shuffle_epi32(v23, (_MM_PERM_ENUM)136);
                    const __m512i v00s2 = _mm512_shuffle_epi32(v00, (_MM_PERM_ENUM)221);
                    const __m512i v01s2 = _mm512_shuffle_epi32(v01, (_MM_PERM_ENUM)221);
                    const __m512i v02s2 = _mm512_shuffle_epi32(v02, (_MM_PERM_ENUM)221);
                    const __m512i v03s2 = _mm512_shuffle_epi32(v03, (_MM_PERM_ENUM)221);
                    const __m512i v20s2 = _mm512_shuffle_epi32(v20, (_MM_PERM_ENUM)221);
                    const __m512i v21s2 = _mm512_shuffle_epi32(v21, (_MM_PERM_ENUM)221);
                    const __m512i v22s2 = _mm512_shuffle_epi32(v22, (_MM_PERM_ENUM)221);
                    const __m512i v23s2 = _mm512_shuffle_epi32(v23, (_MM_PERM_ENUM)221);

                    /* dpbusd MAC: 4 activation chunks x 4 weight chunks x 2 shuffles */
                    const __m512i zero = _mm512_setzero_epi32();

                    __m512i i00 = _mm512_add_epi32(
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(zero,
                            v03s1, rl3s1), v02s1, rl2s1), v01s1, rl1s1), v00s1, rl0s1),
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(zero,
                            v03s2, rl3s2), v02s2, rl2s2), v01s2, rl1s2), v00s2, rl0s2));
                    __m512i i01 = _mm512_add_epi32(
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(zero,
                            v03s1, ro3s1), v02s1, ro2s1), v01s1, ro1s1), v00s1, ro0s1),
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(zero,
                            v03s2, ro3s2), v02s2, ro2s2), v01s2, ro1s2), v00s2, ro0s2));
                    __m512i i10 = _mm512_add_epi32(
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(zero,
                            v23s1, rl3s1), v22s1, rl2s1), v21s1, rl1s1), v20s1, rl0s1),
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(zero,
                            v23s2, rl3s2), v22s2, rl2s2), v21s2, rl1s2), v20s2, rl0s2));
                    __m512i i11 = _mm512_add_epi32(
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(zero,
                            v23s1, ro3s1), v22s1, ro2s1), v21s1, ro1s1), v20s1, ro0s1),
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(zero,
                            v23s2, ro3s2), v22s2, ro2s2), v21s2, ro1s2), v20s2, ro0s2));

                    /* Straighten to 4 row vectors (4 int32 each) */
                    const __m256i i00lo = _mm512_castsi512_si256(i00);
                    const __m256i i01lo = _mm512_castsi512_si256(i01);
                    const __m256i i10lo = _mm512_castsi512_si256(i10);
                    const __m256i i11lo = _mm512_castsi512_si256(i11);

                    __m256i row0 = _mm256_blend_epi32(i00lo, _mm256_shuffle_epi32(i01lo, 78), 204);
                    __m256i row1 = _mm256_blend_epi32(_mm256_shuffle_epi32(i00lo, 78), i01lo, 204);
                    __m256i row2 = _mm256_blend_epi32(i10lo, _mm256_shuffle_epi32(i11lo, 78), 204);
                    __m256i row3 = _mm256_blend_epi32(_mm256_shuffle_epi32(i10lo, 78), i11lo, 204);

                    /* Activation row scales (FIX for Bug #3 + #4):
                     * Use PICOLM_F32Cx16_REPEAT_LOAD which:
                     * 1. Loads exactly 8 bytes from d[4] via _mm_loadl_epi64 (not 16)
                     * 2. Replicates                     * 2. Replicates the 4 FP32 to all 16 lanes of __m512
                     * This matches llama.cpp's GGML_F32Cx16_REPEAT_LOAD pattern. */
                    __m128i row_scale_f16 = _mm_loadl_epi64((const __m128i*)a[b].d);
                    __m512 rs = PICOLM_F32Cx16_REPEAT_LOAD(row_scale_f16);

                    /* int32[4] -> float[4] -> broadcast to 16 lanes.
                     * FIX Bug #5: row0..row3 are __m256i with 4 int32 each.
                     * Convert to __m128 (4 FP32), broadcast to __m512 (16 lanes). */
                    int rb = rp * 4 + x * 8;
                    /* row0..row3 are __m256i with 4 int32 in the low lane.
                     * Extract low 128 bits -> convert 4 int32 to 4 FP32 -> broadcast to 16. */
                    __m512 r0f = _mm512_castps128_ps512(
                        _mm_cvtepi32_ps(_mm256_castsi256_si128(row0)));
                    r0f = _mm512_shuffle_f32x4(r0f, r0f, 0);
                    __m512 r1f = _mm512_castps128_ps512(
                        _mm_cvtepi32_ps(_mm256_castsi256_si128(row1)));
                    r1f = _mm512_shuffle_f32x4(r1f, r1f, 0);
                    __m512 r2f = _mm512_castps128_ps512(
                        _mm_cvtepi32_ps(_mm256_castsi256_si128(row2)));
                    r2f = _mm512_shuffle_f32x4(r2f, r2f, 0);
                    __m512 r3f = _mm512_castps128_ps512(
                        _mm_cvtepi32_ps(_mm256_castsi256_si128(row3)));
                    r3f = _mm512_shuffle_f32x4(r3f, r3f, 0);

                    acc[rb+0] = _mm512_fmadd_ps(
                        r0f, _mm512_mul_ps(cs, _mm512_shuffle_ps(rs,rs,0)),  acc[rb+0]);
                    acc[rb+1] = _mm512_fmadd_ps(
                        r1f, _mm512_mul_ps(cs, _mm512_shuffle_ps(rs,rs,85)), acc[rb+1]);
                    acc[rb+2] = _mm512_fmadd_ps(
                        r2f, _mm512_mul_ps(cs, _mm512_shuffle_ps(rs,rs,170)),acc[rb+2]);
                    acc[rb+3] = _mm512_fmadd_ps(
                        r3f, _mm512_mul_ps(cs, _mm512_shuffle_ps(rs,rs,255)),acc[rb+3]);
                }
            }

            /* Store output */
            for (int i = 0; i < 16; i++) {
                _mm512_storeu_ps((float*)(s + ((y*4+i)*bs + x*8)), acc[i]);
            }
        }
    }
    return y;
}
#endif /* AVX512BW + AVX512DQ */

/* ============================================================
 * AVX2 path: 16x8 output tiles
 * ============================================================ */
#if defined(__AVX2__) && defined(__F16C__)
static int sgemm_q4x8_q8x4_avx2(
        int k, const block_q4_0x8 *bp_start,
        const block_q8_0x4 *ap_start,
        float *s, size_t bs, int nr, int nc)
{
    const int nb = k / 32;
    const __m256i lut256 = build_lut256();
    const __m256i reorder = _mm256_set_epi32(3,2,1,0,7,6,5,4);
    int y = 0;
    const int anr = nr - nr % 16;
    const int anc = nc - nc % 8;

    for (; y < anr / 4; y += 4) {
        const block_q8_0x4 *ap[4];
        ap[0] = ap_start + (y * nb);
        for (int i = 0; i < 3; i++) ap[i+1] = ap[i] + nb;

        for (int x = 0; x < anc / 8; x++) {
            const block_q4_0x8 *bp = bp_start + (x * nb);

            /* FIX Bug #2: acc needs 16 entries (4 rp x 4 rows), not 4 */
            __m256 acc[16];
            for (int i = 0; i < 16; i++) acc[i] = _mm256_setzero_ps();

            for (int b = 0; b < nb; b++) {
                /* Load 8 rows of Q4_0 nibbles (1 block_q4_0x8) */
                const __m256i w00 = _mm256_loadu_si256((const __m256i*)(bp[b].qs));
                const __m256i w10 = _mm256_loadu_si256((const __m256i*)(bp[b].qs+32));
                const __m256i w01 = _mm256_loadu_si256((const __m256i*)(bp[b].qs+64));
                const __m256i w11 = _mm256_loadu_si256((const __m256i*)(bp[b].qs+96));

                /* Blend even/odd row groups */
                const __m256i we0 = _mm256_blend_epi32(
                    w00, _mm256_permutevar8x32_epi32(w10, reorder), 240);
                const __m256i wo0 = _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(w00, reorder), w10, 240);
                const __m256i we1 = _mm256_blend_epi32(
                    w01, _mm256_permutevar8x32_epi32(w11, reorder), 240);
                const __m256i wo1 = _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(w01, reorder), w11, 240);

                /* 4-bit -> 8-bit via LUT */
                const __m256i rl0 = nibble_to_i8_256(we0, lut256);
                const __m256i rl1 = nibble_to_i8_256(we1, lut256);
                const __m256i rl2 = nibble_to_i8_hi_256(we0, lut256);
                const __m256i rl3 = nibble_to_i8_hi_256(we1, lut256);
                const __m256i ro0 = nibble_to_i8_256(wo0, lut256);
                const __m256i ro1 = nibble_to_i8_256(wo1, lut256);
                const __m256i ro2 = nibble_to_i8_hi_256(wo0, lut256);
                const __m256i ro3 = nibble_to_i8_hi_256(wo1, lut256);

                /* Shuffle patterns for dpbusd */
                const __m256i rl0s1 = _mm256_shuffle_epi32(rl0, 136);
                const __m256i rl1s1 = _mm256_shuffle_epi32(rl1, 136);
                const __m256i rl2s1 = _mm256_shuffle_epi32(rl2, 136);
                const __m256i rl3s1 = _mm256_shuffle_epi32(rl3, 136);
                const __m256i ro0s1 = _mm256_shuffle_epi32(ro0, 136);
                const __m256i ro1s1 = _mm256_shuffle_epi32(ro1, 136);
                const __m256i ro2s1 = _mm256_shuffle_epi32(ro2, 136);
                const __m256i ro3s1 = _mm256_shuffle_epi32(ro3, 136);
                const __m256i rl0s2 = _mm256_shuffle_epi32(rl0, 221);
                const __m256i rl1s2 = _mm256_shuffle_epi32(rl1, 221);
                const __m256i rl2s2 = _mm256_shuffle_epi32(rl2, 221);
                const __m256i rl3s2 = _mm256_shuffle_epi32(rl3, 221);
                const __m256i ro0s2 = _mm256_shuffle_epi32(ro0, 221);
                const __m256i ro1s2 = _mm256_shuffle_epi32(ro1, 221);
                const __m256i ro2s2 = _mm256_shuffle_epi32(ro2, 221);
                const __m256i ro3s2 = _mm256_shuffle_epi32(ro3, 221);

                /* Weight column scales (8 rows, 16 bytes) */
                __m256 cs = fp16x8_to_fp32(bp[b].d);

                /* Process 4 activation row groups (FIX Bug #2: was only 2) */
                for (int rp = 0; rp < 4; rp++) {
                    const block_q8_0x4 *a = ap[rp];

                    /* Load Q8_0 activations (128 int8 = 4 x 32) */
                    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a[b].qs));
                    __m256i a1 = _mm256_loadu_si256((const __m256i*)(a[b].qs+32));
                    __m256i a2 = _mm256_loadu_si256((const __m256i*)(a[b].qs+64));
                    __m256i a3 = _mm256_loadu_si256((const __m256i*)(a[b].qs+96));

                    /* Shuffle activations for dpbusd */
                    const __m256i a0s1 = _mm256_shuffle_epi32(a0, 136);
                    const __m256i a1s1 = _mm256_shuffle_epi32(a1, 136);
                    const __m256i a2s1 = _mm256_shuffle_epi32(a2, 136);
                    const __m256i a3s1 = _mm256_shuffle_epi32(a3, 136);
                    const __m256i a0s2 = _mm256_shuffle_epi32(a0, 221);
                    const __m256i a1s2 = _mm256_shuffle_epi32(a1, 221);
                    const __m256i a2s2 = _mm256_shuffle_epi32(a2, 221);
                    const __m256i a3s2 = _mm256_shuffle_epi32(a3, 221);

                    /* dpbusd MAC: 4 activation x 4 weight x 2 shuffle */
                    const __m256i zero = _mm256_setzero_si256();
                    __m256i i00 = _mm256_add_epi32(
                        dpbusd_256(dpbusd_256(dpbusd_256(dpbusd_256(zero,
                            a3s1, rl3s1), a2s1, rl2s1), a1s1, rl1s1), a0s1, rl0s1),
                        dpbusd_256(dpbusd_256(dpbusd_256(dpbusd_256(zero,
                            a3s2, rl3s2), a2s2, rl2s2), a1s2, rl1s2), a0s2, rl0s2));
                    __m256i i01 = _mm256_add_epi32(
                        dpbusd_256(dpbusd_256(dpbusd_256(dpbusd_256(zero,
                            a3s1, ro3s1), a2s1, ro2s1), a1s1, ro1s1), a0s1, ro0s1),
                        dpbusd_256(dpbusd_256(dpbusd_256(dpbusd_256(zero,
                            a3s2, ro3s2), a2s2, ro2s2), a1s2, ro1s2), a0s2, ro0s2));
                    __m256i i10 = _mm256_add_epi32(
                        dpbusd_256(dpbusd_256(dpbusd_256(dpbusd_256(zero,
                            a3s1, rl3s1), a2s1, rl2s1), a1s1, rl1s1), a0s1, rl0s1),
                        dpbusd_256(dpbusd_256(dpbusd_256(dpbusd_256(zero,
                            a3s2, rl3s2), a2s2, rl2s2), a1s2, rl1s2), a0s2, rl0s2));
                    __m256i i11 = _mm256_add_epi32(
                        dpbusd_256(dpbusd_256(dpbusd_256(dpbusd_256(zero,
                            a3s1, ro3s1), a2s1, ro2s1), a1s1, ro1s1), a0s1, ro0s1),
                        dpbusd_256(dpbusd_256(dpbusd_256(dpbusd_256(zero,
                            a3s2, ro3s2), a2s2, ro2s2), a1s2, ro1s2), a0s2, ro0s2));

                    /* Straighten to 4 row vectors (4 int32 each) */
                    __m256i row0 = _mm256_blend_epi32(i00, _mm256_shuffle_epi32(i01,78), 204);
                    __m256i row1 = _mm256_blend_epi32(_mm256_shuffle_epi32(i00,78), i01, 204);
                    __m256i row2 = _mm256_blend_epi32(i10, _mm256_shuffle_epi32(i11,78), 204);
                    __m256i row3 = _mm256_blend_epi32(_mm256_shuffle_epi32(i10,78), i11, 204);

                    /* Activation scales (FIX Bug #3 + #4): load 4 FP16, convert to 16 FP32 */
                    __m128i rs_f16 = _mm_loadl_epi64((const __m128i*)a[b].d);
                    __m256 rs = _mm256_cvtph_ps(rs_f16);

                    /* Scale and accumulate: all 4 row groups */
                    int rb = rp * 4 + x * 8;
                    acc[rb+0] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(row0), _mm256_mul_ps(cs,
                        _mm256_shuffle_ps(rs,rs,0)), acc[rb+0]);
                    acc[rb+1] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(row1), _mm256_mul_ps(cs,
                        _mm256_shuffle_ps(rs,rs,85)), acc[rb+1]);
                    acc[rb+2] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(row2), _mm256_mul_ps(cs,
                        _mm256_shuffle_ps(rs,rs,170)), acc[rb+2]);
                    acc[rb+3] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(row3), _mm256_mul_ps(cs,
                        _mm256_shuffle_ps(rs,rs,255)), acc[rb+3]);
                }
            }

            /* Store output (16 rows) */
            for (int i = 0; i < 16; i++) {
                _mm256_storeu_ps((float*)(s + ((y*4+i)*bs + x*8)), acc[i]);
            }
        }
    }
    return y;
}
#endif /* AVX2 + F16C */

/* ============================================================
 * Top-level dispatcher
 * ============================================================ */
int sgemm_q4_0x8_q8_0x4(int m, int n, int k,
        const void *vx, const void *vy, float *s, size_t bs)
{
    const block_q4_0x8 *bp = (const block_q4_0x8 *)vx;
    const block_q8_0x4 *ap = (const block_q8_0x4 *)vy;

    if (m < 16 || n < 8 || k % 32 != 0) return 0;

#if defined(__AVX512BW__) && defined(__AVX512DQ__)
    {
    int yd = sgemm_q4x8_q8x4_avx512(k, bp, ap, s, bs, m, n);
    if (yd >= m) return 1;
    }
#endif

#if defined(__AVX2__) && defined(__F16C__)
    {
    int yd = sgemm_q4x8_q8x4_avx2(k, bp, ap, s, bs, m, n);
    if (yd >= m) return 1;
    }
#endif

    return 0;
}
