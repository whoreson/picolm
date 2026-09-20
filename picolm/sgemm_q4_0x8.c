/* ================================================================
 * Interleaved Q4_0_8x8 x Q8_0_4x8 Tiled GEMM (AVX-512 + AVX2)
 * ================================================================
 * Port of llama.cpp's gemm_q4_b32_8x8_q8_0_lut_avx.
 *
 * C[nr][nc] = A[nr][k] @ B[nc][k]^T
 * nr=tokens, nc=model_dim, k=inner_dim (multiple of 32)
 * Weights: block_q4_0x8[nc/8][k/32]
 * Activations: block_q8_0x4[nr/4][k/32]
 *
 * 16x16 output tiles. AVX-512 tiled GEMM, AVX2 GEMV fallback.
 * STATUS: WORKING
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "quant.h"

/* ============================================================
 * dpbusd MAC helpers (sign trick: |x| * sign(y,x) = x*y)
 * ============================================================ */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
static inline __m512i dpbusd_512(const __m512i acc, const __m512i x, const __m512i y) {
    __m256i xlo = _mm512_castsi512_si256(x);
    __m256i xhi = _mm512_extracti32x8_epi32(x, 1);
    __m256i ylo = _mm512_castsi512_si256(y);
    __m256i yhi = _mm512_extracti32x8_epi32(y, 1);
    __m256i axlo = _mm256_sign_epi8(xlo, xlo);
    __m256i axhi = _mm256_sign_epi8(xhi, xhi);
    __m256i sylo = _mm256_sign_epi8(ylo, xlo);
    __m256i syhi = _mm256_sign_epi8(yhi, xhi);
    __m512i ax = _mm512_inserti32x8(_mm512_castsi256_si512(axlo), axhi, 1);
    __m512i sy = _mm512_inserti32x8(_mm512_castsi256_si512(sylo), syhi, 1);
    return _mm512_dpbusd_epi32(acc, ax, sy);
}
#elif defined(__AVX512BW__) && defined(__AVX512DQ__)
static inline __m512i dpbusd_512(const __m512i acc, const __m512i x, const __m512i y) {
    __m256i xlo = _mm512_castsi512_si256(x);
    __m256i xhi = _mm512_extracti32x8_epi32(x, 1);
    __m256i ylo = _mm512_castsi512_si256(y);
    __m256i yhi = _mm512_extracti32x8_epi32(y, 1);
    __m256i axlo = _mm256_sign_epi8(xlo, xlo);
    __m256i axhi = _mm256_sign_epi8(xhi, xhi);
    __m256i sylo = _mm256_sign_epi8(ylo, xlo);
    __m256i syhi = _mm256_sign_epi8(yhi, xhi);
    __m512i ax = _mm512_inserti32x8(_mm512_castsi256_si512(axlo), axhi, 1);
    __m512i sy = _mm512_inserti32x8(_mm512_castsi256_si512(sylo), syhi, 1);
    const __m512i dot = _mm512_maddubs_epi16(ax, sy);
    return _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_set1_epi16(1), dot));
}
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
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
 * FP16 -> FP32 converters
 * ============================================================ */
static inline __m256 fp16x8_to_fp32(const uint16_t *d) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)d));
}
static inline __m256 fp16x4_to_fp32(const uint16_t *d) {
    return _mm256_cvtph_ps(_mm_loadl_epi64((const __m128i*)d));
}

#if defined(__AVX512F__)
static inline __m512 fp16x16_to_fp32(const uint16_t *d0, const uint16_t *d1) {
    return _mm512_cvtph_ps(_mm256_set_m128i(
        _mm_loadu_si128((const __m128i*)d1),
        _mm_loadu_si128((const __m128i*)d0)));
}
#define PICOLM_F32Cx16_REPEAT_LOAD(x) \
    _mm512_cvtph_ps(_mm256_set_m128i(x, x))
#endif

/* ============================================================
 * 4-bit LUT dequant helpers
 * ============================================================ */
static inline __m256i build_lut256(void) {
    __m128i l = _mm_set_epi8(-1,-2,-3,-4,-5,-6,-7,-8,7,6,5,4,3,2,1,0);
    return _mm256_permute2f128_si256(_mm256_castsi128_si256(l),
                                     _mm256_castsi128_si256(l), 0);
}
#if defined(__AVX512BW__)
static inline __m512i build_lut512(void) {
    __m256i l = build_lut256();
    return _mm512_inserti32x8(_mm512_castsi256_si512(l), l, 1);
}
#endif

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
 * AVX-512: 16x16 tiled GEMM
 * Directly follows llama.cpp's gemm_q4_b32_8x8_q8_0_lut_avx structure.
 * Kept but not wired up by default -- see sgemm_q4_0x8_q8_0x4() below.
 * Only compiled in if PICOLM_Q4_0X8_TRY_AVX512 is defined, to avoid an
 * unused-static-function warning on AVX-512 hosts (which now use the
 * AVX2 path instead).
 * ============================================================ */
#if defined(PICOLM_Q4_0X8_TRY_AVX512) && defined(__AVX512BW__) && defined(__AVX512DQ__)
static int sgemm_q4x8_q8x4_avx512(
        int k, const block_q4_0x8 *bp,
        const block_q8_0x4 *ap,
        float *s, size_t bs, int nr, int nc,
        int ith, int nth)
{
    const int nb = k / 32;
    const int anr = nr - nr % 16;
    const int anc = nc - nc % 16;
    const __m512i lut = build_lut512();
    const __m256i reorder = _mm256_set_epi32(3,2,1,0,7,6,5,4);

    /* Tile grid: (anr/16) activation tile rows x (anc/16) weight tile cols.
     * Each tile is 16 activation rows x 16 weight rows.
     * Distribute tiles across nth threads. */
    int n_ytiles = anr / 16;
    int n_xtiles = anc / 16;
    int total_tiles = n_ytiles * n_xtiles;
    int duty = (total_tiles + nth - 1) / nth;
    int start = duty * ith;
    int end = start + duty;
    if (end > total_tiles) end = total_tiles;

    for (int job = start; job < end; job++) {
        int yt = job / n_xtiles;       /* activation tile row index */
        int xt = job % n_xtiles;       /* weight tile col index */
        int y = yt * 4;                /* activation row group index (4 groups = 16 rows) */
        int xg = xt * 2;               /* weight row group index (each +2 = 16 weight rows) */

        /* 4 groups of activation rows, each group = nb blocks */
        const block_q8_0x4 *ap4[4];
        ap4[0] = ap + (y * nb);
        for (int i = 0; i < 3; i++) ap4[i+1] = ap4[i] + nb;

        /* Two column groups = 16 weight rows */
        const block_q4_0x8 *bp0 = bp + (xg * nb);
        const block_q4_0x8 *bp1 = bp + ((xg+1) * nb);

        __m512 acc[16];
        for (int i = 0; i < 16; i++) acc[i] = _mm512_setzero_ps();

        for (int b = 0; b < nb; b++) {
            /* Load 8 x 32 = 256 bytes of nibble data (2 blocks x 4 chunks) */
            const __m256i w00 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs));
            const __m256i w10 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs+32));
            const __m256i w01 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs+64));
            const __m256i w11 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs+96));
            const __m256i w20 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs));
            const __m256i w30 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs+32));
            const __m256i w21 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs+64));
            const __m256i w31 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs+96));

                /* Blend+permute to separate even/odd weight row groups.
                 * "even" = rows 0,1,4,5 from first block + 8,9,C,D from second
                 * "odd"  = rows 2,3,6,7 from first block + A,B,E,F from second */
                const __m256i we00 = _mm256_blend_epi32(w00, _mm256_permutevar8x32_epi32(w10,reorder), 240);
                const __m256i wo00 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(w00,reorder), w10, 240);
                const __m256i we01 = _mm256_blend_epi32(w01, _mm256_permutevar8x32_epi32(w11,reorder), 240);
                const __m256i wo01 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(w01,reorder), w11, 240);
                const __m256i we10 = _mm256_blend_epi32(w20, _mm256_permutevar8x32_epi32(w30,reorder), 240);
                const __m256i wo10 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(w20,reorder), w30, 240);
                const __m256i we11 = _mm256_blend_epi32(w21, _mm256_permutevar8x32_epi32(w31,reorder), 240);
                const __m256i wo11 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(w21,reorder), w31, 240);

                /* Merge to 512-bit: even group (chunks 0+1) and odd group (chunks 0+1) */
                const __m512i rhs_e0 = _mm512_inserti32x8(_mm512_castsi256_si512(we00), we10, 1);
                const __m512i rhs_e1 = _mm512_inserti32x8(_mm512_castsi256_si512(we01), we11, 1);
                const __m512i rhs_o0 = _mm512_inserti32x8(_mm512_castsi256_si512(wo00), wo10, 1);
                const __m512i rhs_o1 = _mm512_inserti32x8(_mm512_castsi256_si512(wo01), wo11, 1);

                /* LUT dequant: 4-bit -> signed 8-bit (low + high nibble, 4 chunks) */
                const __m512i re0 = nibble_to_i8_512(rhs_e0, lut);
                const __m512i re1 = nibble_to_i8_512(rhs_e1, lut);
                const __m512i re2 = nibble_to_i8_hi_512(rhs_e0, lut);
                const __m512i re3 = nibble_to_i8_hi_512(rhs_e1, lut);
                const __m512i ro0 = nibble_to_i8_512(rhs_o0, lut);
                const __m512i ro1 = nibble_to_i8_512(rhs_o1, lut);
                const __m512i ro2 = nibble_to_i8_hi_512(rhs_o0, lut);
                const __m512i ro3 = nibble_to_i8_hi_512(rhs_o1, lut);

                /* Shuffle weights for dpbusd (broadcast within quad) */
                const __m512i re0s1 = _mm512_shuffle_epi32(re0, 136);
                const __m512i re1s1 = _mm512_shuffle_epi32(re1, 136);
                const __m512i re2s1 = _mm512_shuffle_epi32(re2, 136);
                const __m512i re3s1 = _mm512_shuffle_epi32(re3, 136);
                const __m512i ro0s1 = _mm512_shuffle_epi32(ro0, 136);
                const __m512i ro1s1 = _mm512_shuffle_epi32(ro1, 136);
                const __m512i ro2s1 = _mm512_shuffle_epi32(ro2, 136);
                const __m512i ro3s1 = _mm512_shuffle_epi32(ro3, 136);
                const __m512i re0s2 = _mm512_shuffle_epi32(re0, 221);
                const __m512i re1s2 = _mm512_shuffle_epi32(re1, 221);
                const __m512i re2s2 = _mm512_shuffle_epi32(re2, 221);
                const __m512i re3s2 = _mm512_shuffle_epi32(re3, 221);
                const __m512i ro0s2 = _mm512_shuffle_epi32(ro0, 221);
                const __m512i ro1s2 = _mm512_shuffle_epi32(ro1, 221);
                const __m512i ro2s2 = _mm512_shuffle_epi32(ro2, 221);
                const __m512i ro3s2 = _mm512_shuffle_epi32(ro3, 221);

                /* Weight column scales: 16 FP16 -> 16 FP32 */
                __m512 cs = fp16x16_to_fp32(bp0[b].d, bp1[b].d);

                /* Process activation row groups (4 groups of 4 rows = 16 rows) */
                for (int rp = 0; rp < 4; rp++) {
                    const block_q8_0x4 *a = ap4[rp];

                    /* Load 4 x 32-byte interleaved activation chunks */
                    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a[b].qs));
                    __m256i a1 = _mm256_loadu_si256((const __m256i*)(a[b].qs+32));
                    __m256i a2 = _mm256_loadu_si256((const __m256i*)(a[b].qs+64));
                    __m256i a3 = _mm256_loadu_si256((const __m256i*)(a[b].qs+96));

                    /* Split each 32-byte chunk: low 16 bytes = A0/A1, high 16 = A2/A3 */
                    __m256i a0l = _mm256_permute2f128_si256(a0, a0, 0);
                    __m256i a0h = _mm256_permute2f128_si256(a0, a0, 17);
                    __m256i a1l = _mm256_permute2f128_si256(a1, a1, 0);
                    __m256i a1h = _mm256_permute2f128_si256(a1, a1, 17);
                    __m256i a2l = _mm256_permute2f128_si256(a2, a2, 0);
                    __m256i a2h = _mm256_permute2f128_si256(a2, a2, 17);
                    __m256i a3l = _mm256_permute2f128_si256(a3, a3, 0);
                    __m256i a3h = _mm256_permute2f128_si256(a3, a3, 17);

                    /* Expand to 512-bit (duplicate 256-bit -> 512-bit) */
                    const __m512i l01_0 = _mm512_inserti32x8(_mm512_castsi256_si512(a0l), a0l, 1);
                    const __m512i l23_0 = _mm512_inserti32x8(_mm512_castsi256_si512(a0h), a0h, 1);
                    const __m512i l01_1 = _mm512_inserti32x8(_mm512_castsi256_si512(a1l), a1l, 1);
                    const __m512i l23_1 = _mm512_inserti32x8(_mm512_castsi256_si512(a1h), a1h, 1);
                    const __m512i l01_2 = _mm512_inserti32x8(_mm512_castsi256_si512(a2l), a2l, 1);
                    const __m512i l23_2 = _mm512_inserti32x8(_mm512_castsi256_si512(a2h), a2h, 1);
                    const __m512i l01_3 = _mm512_inserti32x8(_mm512_castsi256_si512(a3l), a3l, 1);
                    const __m512i l23_3 = _mm512_inserti32x8(_mm512_castsi256_si512(a3h), a3h, 1);

                    /* Shuffle activations: interleave A0/A1 (or A2/A3) pairs.
                     * 160 (0xA0): [L0,L0,L2,L2] -> A0/A0/A1/A1 per quad
                     * 245 (0xF5): [L1,L1,L3,L3] -> A0[4-7]/A0[4-7]/A1[4-7]/A1[4-7] */
                    const __m512i l01_0s1 = _mm512_shuffle_epi32(l01_0, 160);
                    const __m512i l01_1s1 = _mm512_shuffle_epi32(l01_1, 160);
                    const __m512i l01_2s1 = _mm512_shuffle_epi32(l01_2, 160);
                    const __m512i l01_3s1 = _mm512_shuffle_epi32(l01_3, 160);
                    const __m512i l23_0s1 = _mm512_shuffle_epi32(l23_0, 160);
                    const __m512i l23_1s1 = _mm512_shuffle_epi32(l23_1, 160);
                    const __m512i l23_2s1 = _mm512_shuffle_epi32(l23_2, 160);
                    const __m512i l23_3s1 = _mm512_shuffle_epi32(l23_3, 160);
                    const __m512i l01_0s2 = _mm512_shuffle_epi32(l01_0, 245);
                    const __m512i l01_1s2 = _mm512_shuffle_epi32(l01_1, 245);
                    const __m512i l01_2s2 = _mm512_shuffle_epi32(l01_2, 245);
                    const __m512i l01_3s2 = _mm512_shuffle_epi32(l01_3, 245);
                    const __m512i l23_0s2 = _mm512_shuffle_epi32(l23_0, 245);
                    const __m512i l23_1s2 = _mm512_shuffle_epi32(l23_1, 245);
                    const __m512i l23_2s2 = _mm512_shuffle_epi32(l23_2, 245);
                    const __m512i l23_3s2 = _mm512_shuffle_epi32(l23_3, 245);

                    /* dpbusd: pair activations against even/odd weight groups.
                     * 4 chunks x 2 shuffles = 8 dpbusd per accumulator.
                     * i00: A0/A1 vs even weights (B0,B1,B4,B5,B8,B9,BC,BD)
                     * i01: A0/A1 vs odd weights  (B2,B3,B6,B7,BA,BB,BE,BF)
                     * i10: A2/A3 vs even weights
                     * i11: A2/A3 vs odd weights */
                    __m512i i00 = _mm512_add_epi32(
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(_mm512_setzero_epi32(),l01_3s1,re3s1),l01_2s1,re2s1),l01_1s1,re1s1),l01_0s1,re0s1),
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(_mm512_setzero_epi32(),l01_3s2,re3s2),l01_2s2,re2s2),l01_1s2,re1s2),l01_0s2,re0s2));
                    __m512i i01 = _mm512_add_epi32(
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(_mm512_setzero_epi32(),l01_3s1,ro3s1),l01_2s1,ro2s1),l01_1s1,ro1s1),l01_0s1,ro0s1),
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(_mm512_setzero_epi32(),l01_3s2,ro3s2),l01_2s2,ro2s2),l01_1s2,ro1s2),l01_0s2,ro0s2));
                    __m512i i10 = _mm512_add_epi32(
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(_mm512_setzero_epi32(),l23_3s1,re3s1),l23_2s1,re2s1),l23_1s1,re1s1),l23_0s1,re0s1),
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(_mm512_setzero_epi32(),l23_3s2,re3s2),l23_2s2,re2s2),l23_1s2,re1s2),l23_0s2,re0s2));
                    __m512i i11 = _mm512_add_epi32(
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(_mm512_setzero_epi32(),l23_3s1,ro3s1),l23_2s1,ro2s1),l23_1s1,ro1s1),l23_0s1,ro0s1),
                        dpbusd_512(dpbusd_512(dpbusd_512(dpbusd_512(_mm512_setzero_epi32(),l23_3s2,ro3s2),l23_2s2,ro2s2),l23_1s2,ro1s2),l23_0s2,ro0s2));

                    /* Straighten: extract per-activation-row results.
                     * Each iXX has interleaved activation row results (A0/A1 or A2/A3)
                     * mixed between even and odd weight groups.
                     * mask_blend(0xCCCC, a, shuffle(b,78)):
                     *   - 0xCCCC selects even-indexed lanes from a, odd from shuffle(b,78)
                     *   - shuffle(b,78) swaps adjacent pairs within each quad
                     * This merges the even/odd weight groups into sequential order. */
                    __m512i row0 = _mm512_mask_blend_epi32(0xCCCC, i00, _mm512_shuffle_epi32(i01, 78));
                    __m512i row1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(i00, 78), i01);
                    __m512i row2 = _mm512_mask_blend_epi32(0xCCCC, i10, _mm512_shuffle_epi32(i11, 78));
                    __m512i row3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(i10, 78), i11);

                    /* Activation row scales: 4 FP16 -> 16 FP32.
                     * _mm_loadl_epi64 loads 8 bytes into low 64 bits, zeros high 64 bits.
                     * shuffle(68) duplicates low 32-bit words to fill all 8 FP16 slots,
                     * matching llama.cpp's __avx512_repeat_f32cx16_load behavior. */
                    __m128i rs_f16 = _mm_loadl_epi64((const __m128i*)a[b].d);
                    rs_f16 = _mm_shuffle_epi32(rs_f16, 68);
                    __m512 rs = PICOLM_F32Cx16_REPEAT_LOAD(rs_f16);

                    __m512 rs0 = _mm512_shuffle_ps(rs, rs, 0);    /* broadcast lane 0 (A0 scale) */
                    __m512 rs1 = _mm512_shuffle_ps(rs, rs, 85);   /* broadcast lane 4 (A1 scale) */
                    __m512 rs2 = _mm512_shuffle_ps(rs, rs, 170);  /* broadcast lane 8 (A2 scale) */
                    __m512 rs3 = _mm512_shuffle_ps(rs, rs, 255);  /* broadcast lane 12 (A3 scale) */

                    /* Multiply by combined scales and accumulate into acc[0..15] */
                    acc[rp*4]     = _mm512_fmadd_ps(_mm512_cvtepi32_ps(row0), _mm512_mul_ps(cs, rs0), acc[rp*4]);
                    acc[rp*4 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(row1), _mm512_mul_ps(cs, rs1), acc[rp*4+1]);
                    acc[rp*4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(row2), _mm512_mul_ps(cs, rs2), acc[rp*4+2]);
                    acc[rp*4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(row3), _mm512_mul_ps(cs, rs3), acc[rp*4+3]);
                }
            }

            /* Store output: C[nr][nc], bs = nc (= d)
             * Each acc[i] = 16 FP32 for output cols xg*8..xg*8+15 */
            for (int i = 0; i < 16; i++) {
                _mm512_storeu_ps((float*)(s + ((y*4+i)*bs + xg*8)), acc[i]);
            }
        } /* end job loop */
    return anr;
}
#endif /* AVX512BW + AVX512DQ */


/* ============================================================
 * AVX2 tiled GEMM: 8 weight rows (one block_q4_0x8) x 4 activation
 * rows (one block_q8_0x4) computed together.
 *
 * STATUS: WORKING. This replaces a previous AVX2 "GEMM" that was
 * never actually wired into sgemm_q4_0x8_q8_0x4() (see git history)
 * and, on inspection, also made a wrong assumption about both the
 * block_q4_0x8 nibble interleaving *and* the block_q8_0x4 activation
 * layout (it assumed a byte-interleaved activation format; picolm's
 * quantize_mat_q8_0x4() actually stores 4 rows back-to-back as
 * qs[r*32 .. r*32+32) -- see quant.c). That mismatch is what produced
 * garbage output and led to the kernel being disabled wholesale.
 *
 * This version is a port of ik_llama.cpp's (ikawrakow/ik_llama.cpp)
 * mul_mat_q4_0_r8_q8_2_avx2() strategy from
 * ggml/src/iqk/iqk_gemm_legacy_quants.cpp: for each k-block, unpack
 * the 8 interleaved weight rows into signed int8 lanes *once*
 * (prepare_q4_0_quants_avx2 in ik_llama's terms) and then reuse that
 * unpacked tile across every activation row in the group
 * (accum_q4_0_quants), instead of re-unpacking the weight nibbles
 * once per activation row the way a GEMV kernel must. That's the
 * actual "GEMM" win: the nibble/LUT/blend work is amortized 4x.
 *
 * The nibble-unpack and blend sequence itself is copied verbatim
 * from vec_dot_q4_0x8_q8_0_avx2() (quant.c), which is the verified,
 * bug-fixed GEMV kernel for this exact block_q4_0x8 layout -- see the
 * comment on Q4_0X8_MULSUM there for why the sign-trick maddubs
 * reduction is correct. picolm's Q8_0 activations are symmetric
 * per-block-delta only (no bsums/min term), unlike ik_llama's
 * asymmetric block_q8_2, so no extra "-8 * activation_sum" correction
 * term is needed here (that term in ik_llama's kernel exists purely
 * because ik_llama's Q4_0_R8 dequant is unsigned-nibble based and
 * corrected via the block_q8_2 min; picolm's XOR-0x88 repacking
 * already produces signed nibbles, matching Q4_0's own convention).
 * ============================================================ */
#if defined(__AVX2__) && defined(__F16C__)

#if defined(__FMA__)
#define Q4X8_FMADD(a,b,c) _mm256_fmadd_ps((a),(b),(c))
#else
#define Q4X8_FMADD(a,b,c) _mm256_add_ps(_mm256_mul_ps((a),(b)),(c))
#endif

/* Computes an 8(weight-cols) x 4(activation-rows) output tile for one
 * (weight-group, activation-group) pair, summed over nb k-blocks.
 * bp: nb consecutive block_q4_0x8 (one weight-row-group)
 * ap: nb consecutive block_q8_0x4 (one activation-row-group)
 * out4x8[r][0..7]: output for activation row r, weight cols 0..7 */
static inline void sgemm_q4x8_q8x4_avx2_tile(
        const block_q4_0x8 *bp, const block_q8_0x4 *ap, int nb,
        float out4x8[4][8])
{
    __m256i signextendlut = _mm256_castsi128_si256(
        _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
    const __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m128i changemask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4,
                                             11, 10, 3, 2, 9, 8, 1, 0);

    __m256 acc[4];
    acc[0] = acc[1] = acc[2] = acc[3] = _mm256_setzero_ps();

#define Q4_0X8_MULSUM_INTO(iacc, bvec, avec) \
    (iacc) = _mm256_add_epi32((iacc), _mm256_madd_epi16(_mm256_set1_epi16(1), \
        _mm256_maddubs_epi16(_mm256_sign_epi8((bvec), (bvec)), _mm256_sign_epi8((avec), (bvec)))))

    for (int b = 0; b < nb; b++) {
        /* ---- Weight side: unpack once, reused for all 4 activation rows ---- */
        const __m256i rhs0_0 = _mm256_loadu_si256((const __m256i *)bp[b].qs);
        const __m256i rhs1_0 = _mm256_loadu_si256((const __m256i *)(bp[b].qs + 32));
        const __m256i rhs0_1 = _mm256_loadu_si256((const __m256i *)(bp[b].qs + 64));
        const __m256i rhs1_1 = _mm256_loadu_si256((const __m256i *)(bp[b].qs + 96));

        const __m256i r00 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs0_0, m4b));
        const __m256i r10 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs1_0, m4b));
        const __m256i r01 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs0_1, m4b));
        const __m256i r11 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs1_1, m4b));
        const __m256i r02 = _mm256_shuffle_epi8(signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs0_0, 4), m4b));
        const __m256i r12 = _mm256_shuffle_epi8(signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs1_0, 4), m4b));
        const __m256i r03 = _mm256_shuffle_epi8(signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs0_1, 4), m4b));
        const __m256i r13 = _mm256_shuffle_epi8(signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs1_1, 4), m4b));

        const __m256 col_scales = _mm256_cvtph_ps(
            _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)bp[b].d), changemask));

        /* Blended forms depend only on the weight tile, not the
         * activation row -- hoisted out of the row loop below. */
        const __m256i bl_lo1 = _mm256_blend_epi32(r00, _mm256_shuffle_epi32(r10, 177), 170);
        const __m256i bh_lo1 = _mm256_blend_epi32(_mm256_shuffle_epi32(r00, 177), r10, 170);
        const __m256i bl_lo2 = _mm256_blend_epi32(r01, _mm256_shuffle_epi32(r11, 177), 170);
        const __m256i bh_lo2 = _mm256_blend_epi32(_mm256_shuffle_epi32(r01, 177), r11, 170);
        const __m256i bl_hi1 = _mm256_blend_epi32(r02, _mm256_shuffle_epi32(r12, 177), 170);
        const __m256i bh_hi1 = _mm256_blend_epi32(_mm256_shuffle_epi32(r02, 177), r12, 170);
        const __m256i bl_hi2 = _mm256_blend_epi32(r03, _mm256_shuffle_epi32(r13, 177), 170);
        const __m256i bh_hi2 = _mm256_blend_epi32(_mm256_shuffle_epi32(r03, 177), r13, 170);

        /* ---- Activation side: one pass per row, reusing the tile above ---- */
        for (int r = 0; r < 4; r++) {
            const uint8_t *aq = (const uint8_t *)ap[b].qs + r * 32;
            __m256i a0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)aq));
            __m256i a1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(aq + 16)));
            a0 = _mm256_permute2f128_si256(a0, a0, 0);
            a1 = _mm256_permute2f128_si256(a1, a1, 0);

            const __m256 row_scale = _mm256_set1_ps(fp16_to_fp32_lookup((uint16_t)ap[b].d[r]));
            const __m256 sd = _mm256_mul_ps(col_scales, row_scale);

            __m256i iacc = _mm256_setzero_si256();
            Q4_0X8_MULSUM_INTO(iacc, bl_lo1, _mm256_shuffle_epi32(a0, 0));
            Q4_0X8_MULSUM_INTO(iacc, bh_lo1, _mm256_shuffle_epi32(a0, 85));
            Q4_0X8_MULSUM_INTO(iacc, bl_lo2, _mm256_shuffle_epi32(a0, 170));
            Q4_0X8_MULSUM_INTO(iacc, bh_lo2, _mm256_shuffle_epi32(a0, 255));
            Q4_0X8_MULSUM_INTO(iacc, bl_hi1, _mm256_shuffle_epi32(a1, 0));
            Q4_0X8_MULSUM_INTO(iacc, bh_hi1, _mm256_shuffle_epi32(a1, 85));
            Q4_0X8_MULSUM_INTO(iacc, bl_hi2, _mm256_shuffle_epi32(a1, 170));
            Q4_0X8_MULSUM_INTO(iacc, bh_hi2, _mm256_shuffle_epi32(a1, 255));

            acc[r] = Q4X8_FMADD(_mm256_cvtepi32_ps(iacc), sd, acc[r]);
        }
    }
#undef Q4_0X8_MULSUM_INTO

    for (int r = 0; r < 4; r++) {
        __m256 result = _mm256_permutevar8x32_ps(acc[r], finalpermutemask);
        _mm256_storeu_ps(out4x8[r], result);
    }
}

/* Tiles the full [nr x nc] output over (nr/4) x (nc/8) 4x8 tiles,
 * distributing tiles across [ith, nth) threads the same way the
 * AVX-512 path above distributes 16x16 tiles.
 * Returns the number of activation rows actually covered (nr rounded
 * down to a multiple of 4), or 0 if the shapes don't meet minimums. */
static int sgemm_q4x8_q8x4_avx2(
        int k, const block_q4_0x8 *bp_start,
        const block_q8_0x4 *ap_start,
        float *s, size_t bs, int nr, int nc,
        int ith, int nth)
{
    const int nb = k / 32;
    const int anr = nr - nr % 4;
    const int anc = nc - nc % 8;
    if (anr <= 0 || anc <= 0) return 0;

    const int n_ytiles = anr / 4;
    const int n_xtiles = anc / 8;
    const int total_tiles = n_ytiles * n_xtiles;
    if (nth < 1) nth = 1;
    const int duty = (total_tiles + nth - 1) / nth;
    int start = duty * ith;
    int end = start + duty;
    if (end > total_tiles) end = total_tiles;

    for (int job = start; job < end; job++) {
        const int yt = job / n_xtiles;
        const int xt = job % n_xtiles;
        const int y = yt * 4;
        const int x = xt * 8;

        const block_q4_0x8 *bp = bp_start + (size_t)xt * nb;
        const block_q8_0x4 *ap = ap_start + (size_t)yt * nb;

        float out4x8[4][8];
        sgemm_q4x8_q8x4_avx2_tile(bp, ap, nb, out4x8);

        for (int r = 0; r < 4; r++) {
            memcpy(s + (size_t)(y + r) * bs + x, out4x8[r], 8 * sizeof(float));
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
        const void *vx, const void *vy, float *s, size_t bs,
        int ith, int nth)
{
    const block_q4_0x8 *bp = (const block_q4_0x8 *)vx;
    const block_q8_0x4 *ap = (const block_q8_0x4 *)vy;

    if (nc < 8 || nr < 4 || k % 32 != 0) return 0;

    int done = 0;

    /* AVX2 path is the verified/primary path (see sgemm_q4x8_q8x4_avx2
     * header comment). It only needs AVX2+F16C, which AVX-512 machines
     * also have, so it is tried unconditionally first. The AVX-512
     * 16x16-tile kernel below is kept for reference but is NOT wired
     * up by default: it has not been re-verified against the fixed
     * activation-layout understanding used here, and correctness of
     * the AVX2 path already covers every host in this project (see
     * project_summary.txt: no host lacks AVX2 among the ones that
     * reach this function at all -- non-AVX2 hosts return 0 here and
     * the caller falls back to the scalar vec_dot path). Define
     * PICOLM_Q4_0X8_TRY_AVX512 to opt back into it for experimentation. */
#if defined(__AVX2__) && defined(__F16C__)
    {
        int yd = sgemm_q4x8_q8x4_avx2(k, bp, ap, s, bs, nr, nc, ith, nth);
        if (yd > 0) done = yd;
    }
#elif defined(PICOLM_Q4_0X8_TRY_AVX512) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    {
        int yd = sgemm_q4x8_q8x4_avx512(k, bp, ap, s, bs, nr, nc, ith, nth);
        if (yd > 0) done = yd;
    }
#endif

    return done;
}
#endif

#if !defined(__AVX2__) && !defined(__AVX512F__)
int sgemm_q4_0x8_q8_0x4(int nr, int nc, int k,
        const void *vx, const void *vy, float *s, size_t bs,
        int ith, int nth)
{
    (void)nr; (void)nc; (void)k; (void)vx; (void)vy;
    (void)s; (void)bs; (void)ith; (void)nth;
    return 0;
}
#endif
