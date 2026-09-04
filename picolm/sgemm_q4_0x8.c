/* ================================================================
 * TODO/WIP: Interleaved Q4_0_8x8 x Q8_0_4x8 GEMM
 * ================================================================
 * STATUS: WIP (WORK IN PROGRESS) -- NOT YET FUNCTIONAL
 *
 * This is a port of llama.cpp's interleaved GEMM infrastructure:
 *   Source: ~/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp
 *   Routine: gemm_q4_b32_8x8_q8_0_lut_avx (AVX-512 path, ~line 641)
 *   Routine: gemm_q4_b32_8x8_q8_0_lut_avx2 (AVX2 path)
 *   Repack:  ~/llama.cpp/ggml/src/ggml-cpu/repack.cpp
 *            ggml_quantize_mat_q8_0_4x8_generic (~line 173)
 *
 * KNOWN BUGS (as of 2025-09-04):
 *   1. _mm512_permute_ps(0x33221100u) compile error: gcc 15 requires
 *      8-bit immediate for AVX-512 shuffle intrinsics. Need to use
 *      _mm512_permutexvar_epi32 or split into 128-bit shuffles.
 *   2. Output store formula uses bs=48 (n_out) but the tile loop
 *      indices (y, i, x) may not match the expected output layout.
 *   3. The dpbusd accumulator straightening (i00..i11 -> row0..row3)
 *      shuffle/blend pattern needs verification against llama.cpp's
 *      mul_sum_i8_pairs_acc_int32x16 reference.
 *
 * TO ENABLE (when fixed):
 *   Set PICOLM_Q4_0x8_GEMM=1 to bypass the vec_dot Q4_0_8_8 path
 *   and use this GEMM instead. Currently disabled by default.
 *
 * Performance target: 2-3x over vec_dot Q4_0_8_8 path (18 tok/s -> 40+ tok/s)
 * by using AVX-512 512-bit vpdpbusd for 16x16 output tiles.
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
 * Ported from llama.cpp gemm_q4_b32_8x8_q8_0_lut_avx
 *
 * Weights:  block_q4_0x8 (8 rows interleaved, k/32 blocks per row)
 * Activations: block_q8_0x4 (4 rows interleaved, k/32 blocks per row)
 * Output:   float C[m][n], row-major
 * ================================================================ */

#include <stdint.h>
#include <string.h>
#include "quant.h"

/* ============================================================
 * Helpers: sign trick dpbusd
 * ============================================================ */

/* 512-bit: 16 int32 accumulators from 64 int8 x 64 int8 */
#if defined(__AVX512F__)
static inline __m512i dpbusd_512(const __m512i acc, const __m512i x, const __m512i y) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ax = _mm512_abs_epi8(x);
    __mmask64 m = _mm512_movepi8_mask(x);
    const __m512i sy = _mm512_mask_sub_epi8(y, m, zero, y);
#if defined(__AVX512VNNI__)
    return _mm512_dpbusd_epi32(acc, ax, sy);
#else
    const __m512i dot = _mm512_maddubs_epi16(ax, sy);
    return _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_set1_epi16(1), dot));
#endif
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
static inline __m256 fp16x8_to_fp32(const uint16_t *d) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)d));
}

#if defined(__AVX512F__)
static inline __m512 fp16x16_to_fp32(const uint16_t *d0, const uint16_t *d1) {
    return _mm512_cvtph_ps(_mm256_set_m128i(
        _mm_loadu_si128((const __m128i*)d1),
        _mm_loadu_si128((const __m128i*)d0)));
}
#endif

/* ============================================================
 * Helpers: 4-bit nibble dequant via LUT
 * ============================================================ */
static const int8_t g_nibble_lut[16] = {-1,-2,-3,-4,-5,-6,-7,-8,7,6,5,4,3,2,1,0};

#if defined(__AVX512BW__)
static inline __m512i nibble_to_i8_512(const __m512i raw, const __m512i lut) {
    return _mm512_shuffle_epi8(lut, _mm512_and_si512(raw, _mm512_set1_epi8(0x0F)));
}
static inline __m512i nibble_to_i8_hi_512(const __m512i raw, const __m512i lut) {
    return _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(raw, 4), _mm512_set1_epi8(0x0F)));
}
#endif

#if defined(__AVX2__)
static inline __m256i nibble_to_i8_256(const __m256i raw, const __m256i lut) {
    return _mm256_shuffle_epi8(lut, _mm256_and_si256(raw, _mm256_set1_epi8(0x0F)));
}
static inline __m256i nibble_to_i8_hi_256(const __m256i raw, const __m256i lut) {
    return _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw, 4), _mm256_set1_epi8(0x0F)));
}
#endif

/* ============================================================
 * Helpers: dpbusd shuffle patterns
 * 
 * For 512-bit dpbusd (16 int32 outputs): each output element sums
 * products of 4 int8 pairs. We need to duplicate each int8 value
 * across 4 lanes within a 16-lane vector.
 * 
 * Pattern 136 = 0b10001000 = [0,0,1,1,4,4,5,5,8,8,9,9,12,12,13,13]
 *   Duplicates even lanes: pairs (0,0), (1,1), (4,4), (5,5), ...
 * Pattern 221 = 0b11011101 = [2,2,3,3,6,6,7,7,10,10,11,11,14,14,15,15]
 *   Duplicates odd lanes: pairs (2,2), (3,3), (6,6), (7,7), ...
 *
 * For 256-bit dpbusd (8 int32 outputs):
 * Pattern 136 = [0,0,4,4,8,8,12,12] (even)
 * Pattern 221 = [2,2,6,6,10,10,14,14] (odd)
 * ============================================================ */

/* ============================================================
 * AVX-512 path: 16x16 output tiles
 * Requires: AVX-512 BW + DQ (+ VNNI for best perf)
 *
 * Processes 16 weight rows (2 block_q4_0x8) x 16 activation rows (4 block_q8_0x4)
 * per kernel call. m aligned to 16, n aligned to 16.
 * ============================================================ */
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
static int sgemm_q4x8_q8x4_avx512(
        int k, const block_q4_0x8 *bp_start,
        const block_q8_0x4 *ap_start,
        float *s, size_t bs, int nr, int nc)
{
    const int nb = k / 32;
    const __m256i lut256 = _mm256_set1_epi64x(*(uint64_t*)g_nibble_lut);
    const __m512i lut512 = _mm512_set1_epi64(*(const long long*)g_nibble_lut);
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
                /* === Load 16 rows of Q4_0 nibbles (2x block_q4_0x8) === */
                /* Each block_q4_0x8: qs[128] = 8 rows x 16 bytes interleaved */
                /* Load as 4x __m256i per block */
                const __m256i w00 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs));
                const __m256i w10 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs+32));
                const __m256i w01 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs+64));
                const __m256i w11 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs+96));
                const __m256i w20 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs));
                const __m256i w30 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs+32));
                const __m256i w21 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs+64));
                const __m256i w31 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs+96));

                /* Blend: separate even/odd row groups to match dpbusd lane ordering */
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

                /* Weight column scales (16 rows) */
                __m512 cs = fp16x16_to_fp32(bp0[b].d, bp1[b].d);

                /* === Process 4 activation row groups (16 rows) === */
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

                    /* Shuffle activations for dpbusd */
                    const __m512i v00s1 = _mm512_shuffle_epi32(v00, (_MM_PERM_ENUM)160);
                    const __m512i v01s1 = _mm512_shuffle_epi32(v01, (_MM_PERM_ENUM)160);
                    const __m512i v02s1 = _mm512_shuffle_epi32(v02, (_MM_PERM_ENUM)160);
                    const __m512i v03s1 = _mm512_shuffle_epi32(v03, (_MM_PERM_ENUM)160);
                    const __m512i v20s1 = _mm512_shuffle_epi32(v20, (_MM_PERM_ENUM)160);
                    const __m512i v21s1 = _mm512_shuffle_epi32(v21, (_MM_PERM_ENUM)160);
                    const __m512i v22s1 = _mm512_shuffle_epi32(v22, (_MM_PERM_ENUM)160);
                    const __m512i v23s1 = _mm512_shuffle_epi32(v23, (_MM_PERM_ENUM)160);
                    const __m512i v00s2 = _mm512_shuffle_epi32(v00, (_MM_PERM_ENUM)245);
                    const __m512i v01s2 = _mm512_shuffle_epi32(v01, (_MM_PERM_ENUM)245);
                    const __m512i v02s2 = _mm512_shuffle_epi32(v02, (_MM_PERM_ENUM)245);
                    const __m512i v03s2 = _mm512_shuffle_epi32(v03, (_MM_PERM_ENUM)245);
                    const __m512i v20s2 = _mm512_shuffle_epi32(v20, (_MM_PERM_ENUM)245);
                    const __m512i v21s2 = _mm512_shuffle_epi32(v21, (_MM_PERM_ENUM)245);
                    const __m512i v22s2 = _mm512_shuffle_epi32(v22, (_MM_PERM_ENUM)245);
                    const __m512i v23s2 = _mm512_shuffle_epi32(v23, (_MM_PERM_ENUM)245);

                    /* dpbusd MAC: 4 activation chunks x 4 weight chunks x 2 shuffles */
                    const __m512i zero = _mm512_setzero_epi32();

                    /* Accumulate: v00 x rl (sp1+sp2) = 8 dpbusd calls */
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
                    /* Extract low 256 bits from each __m512i accumulator */
                    const __m256i i00lo = _mm512_castsi512_si256(i00);
                    const __m256i i00hi = _mm512_extracti32x8_epi32(i00, 1);
                    const __m256i i01lo = _mm512_castsi512_si256(i01);
                    const __m256i i01hi = _mm512_extracti32x8_epi32(i01, 1);
                    const __m256i i10lo = _mm512_castsi512_si256(i10);
                    const __m256i i10hi = _mm512_extracti32x8_epi32(i10, 1);
                    const __m256i i11lo = _mm512_castsi512_si256(i11);
                    const __m256i i11hi = _mm512_extracti32x8_epi32(i11, 1);

                    /* Interleave: [i00lo[0],i01lo[0],i00lo[1],i01lo[1],...] -> row0, row1, row2, row3 */
                    __m256i row0 = _mm256_blend_epi32(i00lo, _mm256_shuffle_epi32(i01lo, 78), 204);
                    __m256i row1 = _mm256_blend_epi32(_mm256_shuffle_epi32(i00lo, 78), i01lo, 204);
                    __m256i row2 = _mm256_blend_epi32(i10lo, _mm256_shuffle_epi32(i11lo, 78), 204);
                    __m256i row3 = _mm256_blend_epi32(_mm256_shuffle_epi32(i10lo, 78), i11lo, 204);

                    /* Activation row scales (4 FP16 -> 4 FP32, then promote to __m512) */
                    __m512 rs = _mm512_castps256_ps512(fp16x8_to_fp32(a[b].d));

                    /* int32[4] -> float[4] -> float[16] via broadcast */
                    int rb = rp * 4 + x * 8;
                    /* int32[4] -> float[4] (low lane) -> broadcast to float[16] */
                    __m128 r0p = _mm_cvtepi32_ps(_mm256_castsi256_si128(row0));
                    __m128 r1p = _mm_cvtepi32_ps(_mm256_castsi256_si128(row1));
                    __m128 r2p = _mm_cvtepi32_ps(_mm256_castsi256_si128(row2));
                    __m128 r3p = _mm_cvtepi32_ps(_mm256_castsi256_si128(row3));
                    /* Each __m128 has 4 FP32 [s0,s1,s2,s3], one per weight row group.
                     * Broadcast to 16 lanes: [s0,s0,s0,s0, s1,s1,s1,s1, s2,s2,s2,s2, s3,s3,s3,s3] */
                    __m512 r0f = _mm512_castps128_ps512(r0p);
                    r0f = _mm512_shuffle_f32x4(r0f, r0f, 0);
                    r0f = _mm512_permute_ps(r0f, 0x33221100u);
                    __m512 r1f = _mm512_castps128_ps512(r1p);
                    r1f = _mm512_shuffle_f32x4(r1f, r1f, 0);
                    r1f = _mm512_permute_ps(r1f, 0x33221100u);
                    __m512 r2f = _mm512_castps128_ps512(r2p);
                    r2f = _mm512_shuffle_f32x4(r2f, r2f, 0);
                    r2f = _mm512_permute_ps(r2f, 0x33221100u);
                    __m512 r3f = _mm512_castps128_ps512(r3p);
                    r3f = _mm512_shuffle_f32x4(r3f, r3f, 0);
                    r3f = _mm512_permute_ps(r3f, 0x33221100u);

                    acc[rb+0] = _mm512_fmadd_ps(r0f, _mm512_mul_ps(cs, _mm512_shuffle_ps(rs,rs,0)),  acc[rb+0]);
                    acc[rb+1] = _mm512_fmadd_ps(r1f, _mm512_mul_ps(cs, _mm512_shuffle_ps(rs,rs,85)), acc[rb+1]);
                    acc[rb+2] = _mm512_fmadd_ps(r2f, _mm512_mul_ps(cs, _mm512_shuffle_ps(rs,rs,170)),acc[rb+2]);
                    acc[rb+3] = _mm512_fmadd_ps(r3f, _mm512_mul_ps(cs, _mm512_shuffle_ps(rs,rs,255)),acc[rb+3]);
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
 * Requires: AVX2 + F16C
 * ============================================================ */
#if defined(__AVX2__) && defined(__F16C__)
static int sgemm_q4x8_q8x4_avx2(
        int k, const block_q4_0x8 *bp_start,
        const block_q8_0x4 *ap_start,
        float *s, size_t bs, int nr, int nc)
{
    const int nb = k / 32;
    const __m256i lut256 = _mm256_set1_epi64x(*(uint64_t*)g_nibble_lut);
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

            __m256 acc[4];
            for (int i = 0; i < 4; i++) acc[i] = _mm256_setzero_ps();

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

                /* Weight column scales (8 rows) */
                __m256 cs = fp16x8_to_fp32(bp[b].d);

                /* Process 4 activation row groups */
                for (int rp = 0; rp < 4; rp++) {
                    const block_q8_0x4 *a = ap[rp];

                    /* Load Q8_0 activations */
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

                    /* dpbusd MAC: 4 activation chunks x 4 weight chunks x 2 shuffles */
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

                    /* Straighten to 2 row vectors */
                    __m256i row0 = _mm256_blend_epi32(i00, _mm256_shuffle_epi32(i01,78), 204);
                    __m256i row1 = _mm256_blend_epi32(_mm256_shuffle_epi32(i00,78), i01, 204);

                    /* Activation scales */
                    __m256 rs = fp16x8_to_fp32(a[b].d);

                    /* Scale and accumulate */
                    int rb = rp * 2 + x * 2;
                    acc[rb+0] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(row0), _mm256_mul_ps(cs,
                        _mm256_shuffle_ps(rs,rs,0)), acc[rb+0]);
                    acc[rb+1] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(row1), _mm256_mul_ps(cs,
                        _mm256_shuffle_ps(rs,rs,85)), acc[rb+1]);
                }
            }

            /* Store */
            for (int i = 0; i < 4; i++) {
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
