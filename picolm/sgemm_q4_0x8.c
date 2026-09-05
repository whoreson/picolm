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
 * BUGS FIXED:
 *   0. Unterminated comment fragment -- FIXED
 *   1. g_nibble_lut broadcast (only 8 of 16 entries) -- FIXED
 *   2. AVX2 acc[4] OOB write -- FIXED (expanded to acc[16])
 *   3. AVX-512 rs unreplicated -- FIXED (PICOLM_F32Cx16_REPEAT_LOAD)
 *   4. Activation delta over-read (16 bytes from 8-byte d[4]) -- FIXED
 *   5. _mm512_permute_ps wrong intrinsic -- FIXED (_mm_cvtepi32_ps)
 *   7. dpbusd_512 missing abs(x) -- FIXED (sign trick for first operand)
 *   8. quantize_mat_q8_0x4 non-interleaved -- FIXED (proper 8-byte chunk interleave)
 *   9. AVX2 path row 2/3 copy-pasted from 0/1 -- FIXED (rewrote as GEMV)
 *  10. Return value in wrong units (y vs anr) -- FIXED (return anr)
 *  11. rb = rp*4+x*8 OOB -- FIXED (rb = rp*4, matching llama.cpp)
 *
 * REMAINING BUGS (blocks enabling):
 *   6. dpbusd accumulator straightening shuffles unverified against reference.
 *      The blend/permute pattern (78, 204, 0xCCCC) was ported structurally
 *      but not exhaustively hand-traced. Without this, even with all other
 *      fixes, the kernel produces wrong output ("read more to identify the N"
 *      instead of "Paris" for Q4_0_8_8 prefill).
 *
 *   Additional issues found during integration:
 *   - Output store layout: llama.cpp writes C[nr][nc] with contiguous columns.
 *     Our caller needs a transposed layout. Requires temp buffer + transpose.
 *   - Parameter convention: llama.cpp's nr/nc map to our n_batch/d differently.
 *     The dispatcher handles this swap.
 *
 * NOT WIRED INTO tensor.c dispatch. The vec_dot fallback remains active.
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
 * dpbusd/maddubs require the first operand to be UNSIGNED.
 * The sign trick: |x| * sign(y,x) = x*y.
 * sign(x,x) = |x| (absolute value), NOT x.
 * Note: _mm512_sign_epi8 is NOT available in gcc 15 headers.
 * We implement it via two _mm256_sign_epi8 + merge. */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
static inline __m512i dpbusd_512(const __m512i acc, const __m512i x, const __m512i y) {
    __m256i xlo = _mm512_castsi512_si256(x);
    __m256i xhi = _mm512_extracti32x8_epi32(x, 1);
    __m256i ylo = _mm512_castsi512_si256(y);
    __m256i yhi = _mm512_extracti32x8_epi32(y, 1);
    __m256i axlo = _mm256_sign_epi8(xlo, xlo);  /* |x| low */
    __m256i axhi = _mm256_sign_epi8(xhi, xhi);  /* |x| high */
    __m256i sylo = _mm256_sign_epi8(ylo, xlo);  /* sign(y,x) low */
    __m256i syhi = _mm256_sign_epi8(yhi, xhi);  /* sign(y,x) high */
    __m512i ax = _mm512_inserti32x8(_mm512_castsi256_si512(axlo), axhi, 1);
    __m512i sy = _mm512_inserti32x8(_mm512_castsi256_si512(sylo), syhi, 1);
    return _mm512_dpbusd_epi32(acc, ax, sy);
}
#elif defined(__AVX512BW__) && defined(__AVX512DQ__)
/* Fallback: maddubs + madd (no VNNI) */
static inline __m512i dpbusd_512(const __m512i acc, const __m512i x, const __m512i y) {
    __m256i xlo = _mm512_castsi512_si256(x);
    __m256i xhi = _mm512_extracti32x8_epi32(x, 1);
    __m256i ylo = _mm512_castsi512_si256(y);
    __m256i yhi = _mm512_extracti32x8_epi32(y, 1);
    __m256i axlo = _mm256_sign_epi8(xlo, xlo);  /* |x| low */
    __m256i axhi = _mm256_sign_epi8(xhi, xhi);  /* |x| high */
    __m256i sylo = _mm256_sign_epi8(ylo, xlo);  /* sign(y,x) low */
    __m256i syhi = _mm256_sign_epi8(yhi, xhi);  /* sign(y,x) high */
    __m512i ax = _mm512_inserti32x8(_mm512_castsi256_si512(axlo), axhi, 1);
    __m512i sy = _mm512_inserti32x8(_mm512_castsi256_si512(sylo), syhi, 1);
    const __m512i dot = _mm512_maddubs_epi16(ax, sy);
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
    const int anr = nr - nr % 16;
    const int anc = nc - nc % 16;

    /* Thread over (y,x) tiles: each tile is independent.
     * Total tiles = (anr/4) * (anc/8) / 2, divided by 2 because x steps by 2.
     * For n_batch=44, d=4096: 2 * 256 = 512 tiles. */
    const int n_tiles = (anr / 4) * (anc / 16);
    const int nth = getenv("PICOLM_Q4_0x8_THREADS") ? atoi(getenv("PICOLM_Q4_0x8_THREADS")) : 16;

#pragma omp parallel num_threads(nth)
    {
        const __m512i lut512 = build_lut512();
        const __m256i reorder = _mm256_set_epi32(3,2,1,0,7,6,5,4);
        int tile;
#pragma omp for schedule(static)
        for (tile = 0; tile < n_tiles; tile++) {
            int y = (tile / (anc / 16)) * 4;
            int x = (tile % (anc / 16)) * 8;

            const block_q8_0x4 *ap[4];
            ap[0] = ap_start + (y * nb);
            for (int i = 0; i < 3; i++) ap[i+1] = ap[i] + nb;

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

                    /* dpbusd MAC: 4 activation chunks x 4 weight chunks x 2 shuffles
                     * Each dpbusd_512 call produces 16 int32 results (one per weight row B0-B15).
                     * The i00/i01/i10/i11 each have 16 int32 = dot products for 16 weight rows
                     * against a single activation row (A0/A1/A2/A3 respectively).
                     * No straightening needed: each lane already corresponds to one weight row. */
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

                    /* FIX for Bug #6: Direct conversion, no straightening blend/permute.
                     * i00 = 16 int32 for weight rows B0-B15 x activation A0
                     * i01 = 16 int32 for weight rows B0-B15 x activation A1
                     * i10 = 16 int32 for weight rows B0-B15 x activation A2
                     * i11 = 16 int32 for weight rows B0-B15 x activation A3
                     * Convert each to 16 FP32, multiply by cs * rs, accumulate into acc[0..15].
                     * Each acc[i] accumulates dot products for weight row B_i across all 4 activation rows. */

                    __m512 f00 = _mm512_cvtepi32_ps(i00);
                    __m512 f01 = _mm512_cvtepi32_ps(i01);
                    __m512 f10 = _mm512_cvtepi32_ps(i10);
                    __m512 f11 = _mm512_cvtepi32_ps(i11);

                    /* Activation scales: 4 FP16 -> 4 FP32, each broadcast to 16 lanes */
                    /* rs0 = activation A0 scale, rs1 = A1, rs2 = A2, rs3 = A3 */
                    __m128i row_scale_f16 = _mm_loadl_epi64((const __m128i*)a[b].d);
                    __m512 rs = PICOLM_F32Cx16_REPEAT_LOAD(row_scale_f16);

                    /* Shuffle rs to extract each of the 4 activation scales, broadcast to 16 lanes */
                    __m512 rs0 = _mm512_shuffle_ps(rs, rs, 0);    /* lane 0 -> all 16 */
                    __m512 rs1 = _mm512_shuffle_ps(rs, rs, 85);   /* lane 4 -> all 16 */
                    __m512 rs2 = _mm512_shuffle_ps(rs, rs, 170);  /* lane 8 -> all 16 */
                    __m512 rs3 = _mm512_shuffle_ps(rs, rs, 255);  /* lane 12 -> all 16 */

                    /* Combine weight scales * activation scales */
                    __m512 cs0 = _mm512_mul_ps(cs, rs0);
                    __m512 cs1 = _mm512_mul_ps(cs, rs1);
                    __m512 cs2 = _mm512_mul_ps(cs, rs2);
                    __m512 cs3 = _mm512_mul_ps(cs, rs3);

                    /* FIX: Direct accumulation using the 16-lane __m512 vectors.
                     * f00[0..15] = dot products for weight rows B0-B15 x activation A0
                     * cs0[0..15] = combined scales for B0-B15 x activation A0
                     * acc[0] accumulates all activations for weight row B0, etc.
                     * We broadcast each lane of f00*cs0 into acc[i], then do the same for
                     * f01*cs1, f10*cs2, f11*cs3. */

                    /* Pre-compute the scaled results for each activation row.
                     * Each has 16 FP32 values, one per weight row B0-B15. */
                    __m512 s00 = _mm512_mul_ps(f00, cs0);
                    __m512 s01 = _mm512_mul_ps(f01, cs1);
                    __m512 s10 = _mm512_mul_ps(f10, cs2);
                    __m512 s11 = _mm512_mul_ps(f11, cs3);

                    /* Transpose 4x16: extract 4 lanes from each of the 4 vectors,
                     * sum them, and accumulate into acc[0..15].
                     * Each acc[i] gets one lane from each of s00/s01/s10/s11, summed.
                     * Process 4 lanes at a time using 4 __m128 extractions. */
                    {
                        __m128 e00a = _mm512_extractf32x4_ps(s00, 0);
                        __m128 e01a = _mm512_extractf32x4_ps(s01, 0);
                        __m128 e10a = _mm512_extractf32x4_ps(s10, 0);
                        __m128 e11a = _mm512_extractf32x4_ps(s11, 0);
                        __m128 ea = _mm_add_ps(_mm_add_ps(e00a, e01a), _mm_add_ps(e10a, e11a));
                        /* Broadcast each of 4 lanes to full __m512 and fmadd into acc[0..3] */
                        acc[0] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ea,ea,0)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[0]);
                        acc[1] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ea,ea,85)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[1]);
                        acc[2] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ea,ea,170)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[2]);
                        acc[3] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ea,ea,255)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[3]);

                        __m128 e00b = _mm512_extractf32x4_ps(s00, 1);
                        __m128 e01b = _mm512_extractf32x4_ps(s01, 1);
                        __m128 e10b = _mm512_extractf32x4_ps(s10, 1);
                        __m128 e11b = _mm512_extractf32x4_ps(s11, 1);
                        __m128 eb = _mm_add_ps(_mm_add_ps(e00b, e01b), _mm_add_ps(e10b, e11b));
                        acc[4] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(eb,eb,0)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[4]);
                        acc[5] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(eb,eb,85)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[5]);
                        acc[6] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(eb,eb,170)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[6]);
                        acc[7] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(eb,eb,255)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[7]);

                        __m128 e00c = _mm512_extractf32x4_ps(s00, 2);
                        __m128 e01c = _mm512_extractf32x4_ps(s01, 2);
                        __m128 e10c = _mm512_extractf32x4_ps(s10, 2);
                        __m128 e11c = _mm512_extractf32x4_ps(s11, 2);
                        __m128 ec = _mm_add_ps(_mm_add_ps(e00c, e01c), _mm_add_ps(e10c, e11c));
                        acc[8] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ec,ec,0)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[8]);
                        acc[9] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ec,ec,85)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[9]);
                        acc[10] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ec,ec,170)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[10]);
                        acc[11] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ec,ec,255)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[11]);

                        __m128 e00d = _mm512_extractf32x4_ps(s00, 3);
                        __m128 e01d = _mm512_extractf32x4_ps(s01, 3);
                        __m128 e10d = _mm512_extractf32x4_ps(s10, 3);
                        __m128 e11d = _mm512_extractf32x4_ps(s11, 3);
                        __m128 ed = _mm_add_ps(_mm_add_ps(e00d, e01d), _mm_add_ps(e10d, e11d));
                        acc[12] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ed,ed,0)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[12]);
                        acc[13] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ed,ed,85)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[13]);
                        acc[14] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ed,ed,170)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[14]);
                        acc[15] = _mm512_fmadd_ps(_mm512_castps128_ps512(_mm_shuffle_ps(ed,ed,255)),
                               _mm512_castps128_ps512(_mm_set1_ps(1.0f)), acc[15]);
                    }
                }
            }

            /* Store output: C[nr][nc] = C[n_batch][d], bs = nc (= d)
             * llama.cpp: s[(y*4+i)*bs + x*8]
             * Each acc[i] = 16 FP32 values for output cols x*8..x*8+15 */
            for (int i = 0; i < 16; i++) {
                _mm512_storeu_ps((float*)(s + ((y*4+i)*bs + x)), acc[i]);
            }
        }
    } /* #pragma omp parallel */
    return anr;
}
#endif /* AVX512BW + AVX512DQ */

/* ============================================================
 * AVX2 path: GEMV (single output row at a time)
 * Processes 8 weight rows x 1 activation row per dpbusd call.
 * This follows llama.cpp's approach: AVX2 has only gemv, not gemm,
 * for Q4_0_8_8. The gemv is still faster than scalar vec_dot
 * due to AVX2 maddubs throughput and LUT-based dequant.
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
    const int anr = nr - nr % 16;
    const int anc = nc - nc % 8;

    for (int y = 0; y < anr; y++) {
        /* Activation row group: 4 rows starting at y, we process row y only */
        int y_block = y / 4;
        int y_off = y % 4;
        const block_q8_0x4 *ap = ap_start + y_block * nb;

        /* Extract one activation row from the interleaved block_q8_0x4:
         * row y_off's data is at qs offsets: for each 8-byte chunk,
         * the y_off-th group of 8 bytes. I.e., qs[y_off*8 .. y_off*8+7],
         * qs[(y_off*8+32) .. (y_off*8+39)], etc.
         * This gives us 32 int8 values per block. */

        for (int x = 0; x < anc; x++) {
            /* Output row y, column x (one activation row) */
            int x_block = x / 8;
            int x_off = x % 8;
            const block_q4_0x8 *bp = bp_start + x_block * nb;

            float sum = 0.0f;

            for (int b = 0; b < nb; b++) {
                /* Load 8 rows of Q4_0 nibbles */
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

                /* Dequant 4-bit -> 8-bit via LUT */
                const __m256i rl0 = nibble_to_i8_256(we0, lut256);
                const __m256i rl1 = nibble_to_i8_256(we1, lut256);
                const __m256i rl2 = nibble_to_i8_hi_256(we0, lut256);
                const __m256i rl3 = nibble_to_i8_hi_256(we1, lut256);
                const __m256i ro0 = nibble_to_i8_256(wo0, lut256);
                const __m256i ro1 = nibble_to_i8_256(wo1, lut256);
                const __m256i ro2 = nibble_to_i8_hi_256(wo0, lut256);
                const __m256i ro3 = nibble_to_i8_hi_256(wo1, lut256);

                /* Weight scale for row x_off (runtime index, need extract not shuffle) */
                __m256 cs = fp16x8_to_fp32(bp[b].d);
                float wscale[8];
                _mm256_storeu_ps(wscale, cs);
                float wscale_v = wscale[x_off];

                /* Extract one activation row from interleaved block:
                 * row y_off is at byte offset y_off*8 in each 32-byte group.
                 * We need 32 bytes total (4 groups of 8). */
                const uint8_t *aqs = ap[b].qs;
                __m128i c0 = _mm_loadl_epi64((const __m128i*)(aqs + 0*32 + y_off*8));
                __m128i c1 = _mm_loadl_epi64((const __m128i*)(aqs + 1*32 + y_off*8));
                __m128i c2 = _mm_loadl_epi64((const __m128i*)(aqs + 2*32 + y_off*8));
                __m128i c3 = _mm_loadl_epi64((const __m128i*)(aqs + 3*32 + y_off*8));
                /* Use _mm256_set_m128i (lo, hi) to avoid inserti128 immediate issues */
                __m256i alo = _mm256_set_m128i(c1, c0);
                __m256i ahi = _mm256_set_m128i(c3, c2);
                __m256i a = _mm256_permute2x128_si256(alo, ahi, 0x20);

                /* Sign trick: |a| * sign(w,a) = a*w */
                __m256i ax = _mm256_sign_epi8(a, a);

                /* Shuffle for dpbusd: split 32 bytes into two 16-byte groups */
                __m256i as1 = _mm256_shuffle_epi32(a, 136);
                __m256i as2 = _mm256_shuffle_epi32(a, 221);
                __m256i axs1 = _mm256_shuffle_epi32(ax, 136);
                __m256i axs2 = _mm256_shuffle_epi32(ax, 221);

                __m256i acc = _mm256_setzero_si256();
                for (int w = 0; w < 4; w++) {
                    const __m256i *wv_p, *wo_p;
                    if (w < 2) { wv_p = &rl0 + w; wo_p = &ro0 + w; }
                    else { wv_p = &rl2 + w-2; wo_p = &ro2 + w-2; }
                    __m256i sy1 = _mm256_sign_epi8(*wv_p, as1);
                    __m256i sy2 = _mm256_sign_epi8(*wo_p, as2);
                    acc = dpbusd_256(dpbusd_256(acc, axs1, sy1), axs2, sy2);
                }

                /* Horizontal sum of 8 int32 */
                float dsum = 0;
                for (int i = 0; i < 8; i++) {
                    int32_t v = _mm256_extract_epi32(acc, i);
                    dsum += (float)v;
                }
                sum += dsum * wscale_v;
            }

            /* Activation scale for row y_off */
            float ascale = fp16_to_fp32(ap[0].d[y_off]);
            /* C[nr][nc] = C[n_batch][d], bs = nc = d
             * y = token index (row in C), x = weight row index (col in C) */
            s[y * bs + x] = sum * ascale;
        }
    }
    return anr;
}
#endif /* AVX2 + F16C */

/* ============================================================
 * Top-level dispatcher
 * ============================================================
 * llama.cpp convention: C[nr][nc] = A[nr][k] @ B[nc][k]^T
 *   nr = activation rows (= n_batch, tokens)
 *   nc = weight rows (= d, model dim)
 *   k = inner dimension (aligned to 32)
 *   vx = weights in block_q4_0x8[nc/8][k/32]
 *   vy = activations in block_q8_0x4[nr/4][k/32]
 *   bs = nc (= d, output stride)
 */
int sgemm_q4_0x8_q8_0x4(int nr, int nc, int k,
        const void *vx, const void *vy, float *s, size_t bs)
{
    const block_q4_0x8 *bp = (const block_q4_0x8 *)vx;
    const block_q8_0x4 *ap = (const block_q8_0x4 *)vy;

    if (nc < 8 || nr < 4 || k % 32 != 0) return 0;

#if defined(__AVX512BW__) && defined(__AVX512DQ__)
    {
    int yd = sgemm_q4x8_q8x4_avx512(k, bp, ap, s, bs, nr, nc);
    if (yd >= nr) return 1;
    }
#endif

#if defined(__AVX2__) && defined(__F16C__)
    {
    int yd = sgemm_q4x8_q8x4_avx2(k, bp, ap, s, bs, nr, nc);
    if (yd >= nr) return 1;
    }
#endif

    return 0;
}
