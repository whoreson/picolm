/* ================================================================
 * Q4I_0_8_8 (pre-dequantized int8) x Q8_0_4x8 Tiled GEMM
 * ================================================================
 * Optimized kernel for Q4I_0_8_8 (GGUF type 34) where weights are
 * pre-dequantized to signed int8 and pre-arranged in dpbusd lane order.
 *
 * This eliminates the blend+permute and LUT dequant shuffle stages
 * from the original Q4_0_8_8 kernel (~48 fewer shuffle instructions).
 *
 * C[nr][nc] = A[nr][k] @ B[nc][k]^T
 * nr=tokens, nc=model_dim, k=inner_dim (multiple of 32)
 * Weights: block_q4i_0x8[nc/8][k/32]
 * Activations: block_q8_0x4[nr/4][k/32]
 *
 * AVX2: 4x8 tiles (primary, verified correct)
 * AVX-512: 16x16 tiles (secondary, disabled due to activation bug)
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

#if defined(__AVX2__)
static inline __m256 fp16x8_to_fp32(const uint16_t *d) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)d));
}
#endif

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
 * AVX2: 4x8 tiled GEMM for Q4I_0_8_8 (pre-dequantized int8)
 *
 * Same strategy as sgemm_q4x8_q8x4_avx2 (sgemm_q4_0x8.c) but
 * skips the LUT dequant step since Q4I data is already int8.
 * The weight reordering from even/odd chunks to the interleaved
 * {0,4,1,5,2,6,3,7} lane pattern is done with permutevar8x32.
 * ============================================================ */
#if defined(__AVX2__) && defined(__F16C__)

#if defined(__FMA__)
#define Q4IX8_FMADD(a,b,c) _mm256_fmadd_ps((a),(b),(c))
#else
#define Q4IX8_FMADD(a,b,c) _mm256_add_ps(_mm256_mul_ps((a),(b)),(c))
#endif

/* Computes an 8(weight-cols) x 4(activation-rows) output tile for one
 * (weight-group, activation-group) pair, summed over nb k-blocks.
 * bp: nb consecutive block_q4i_0x8 (one weight-row-group)
 * ap: nb consecutive block_q8_0x4 (one activation-row-group)
 * out4x8[r][0..7]: output for activation row r, weight cols 0..7 */
static inline void sgemm_q4ix8_q8x4_avx2_tile(
        const block_q4i_0x8 *bp, const block_q8_0x4 *ap, int nb,
        float out4x8[4][8])
{
    const __m256i perm_row = _mm256_set_epi32(7,3,5,1,6,2,4,0);
    const __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    const __m128i changemask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4,
                                             11, 10, 3, 2, 9, 8, 1, 0);

    __m256 acc[4];
    acc[0] = acc[1] = acc[2] = acc[3] = _mm256_setzero_ps();

#define Q4IX8_MULSUM_INTO(iacc, bvec, avec) \
    (iacc) = _mm256_add_epi32((iacc), _mm256_madd_epi16(_mm256_set1_epi16(1), \
        _mm256_maddubs_epi16(_mm256_sign_epi8((bvec), (bvec)), _mm256_sign_epi8((avec), (bvec)))))

    for (int b = 0; b < nb; b++) {
        /* ---- Weight side: load pre-dequantized int8, reorder once ---- */
        /* Even group chunks 0-3 (rows {0,1,4,5}) */
        const __m256i ev0 = _mm256_loadu_si256((const __m256i *)(bp[b].qs));
        const __m256i ev1 = _mm256_loadu_si256((const __m256i *)(bp[b].qs + 32));
        const __m256i ev2 = _mm256_loadu_si256((const __m256i *)(bp[b].qs + 64));
        const __m256i ev3 = _mm256_loadu_si256((const __m256i *)(bp[b].qs + 96));
        /* Odd group chunks 4-7 (rows {2,3,6,7}) */
        const __m256i od0 = _mm256_loadu_si256((const __m256i *)(bp[b].qs + 128));
        const __m256i od1 = _mm256_loadu_si256((const __m256i *)(bp[b].qs + 160));
        const __m256i od2 = _mm256_loadu_si256((const __m256i *)(bp[b].qs + 192));
        const __m256i od3 = _mm256_loadu_si256((const __m256i *)(bp[b].qs + 224));

        /* Reorder lanes: [r0,r1,r4,r5] -> [r0,r4,r1,r5] for even,
         *                [r2,r3,r6,r7] -> [r2,r6,r3,r7] for odd. */
        const __m256i ev0p = _mm256_permutevar8x32_epi32(ev0, perm_row);
        const __m256i od0p = _mm256_permutevar8x32_epi32(od0, perm_row);
        const __m256i ev1p = _mm256_permutevar8x32_epi32(ev1, perm_row);
        const __m256i od1p = _mm256_permutevar8x32_epi32(od1, perm_row);
        const __m256i ev2p = _mm256_permutevar8x32_epi32(ev2, perm_row);
        const __m256i od2p = _mm256_permutevar8x32_epi32(od2, perm_row);
        const __m256i ev3p = _mm256_permutevar8x32_epi32(ev3, perm_row);
        const __m256i od3p = _mm256_permutevar8x32_epi32(od3, perm_row);

        /* Merge even+odd: bl = vals 0-3, bh = vals 4-7 per chunk */
        const __m256i bl_lo1 = _mm256_permute2x128_si256(ev0p, od0p, 0x20);
        const __m256i bh_lo1 = _mm256_permute2x128_si256(ev0p, od0p, 0x31);
        const __m256i bl_lo2 = _mm256_permute2x128_si256(ev1p, od1p, 0x20);
        const __m256i bh_lo2 = _mm256_permute2x128_si256(ev1p, od1p, 0x31);
        const __m256i bl_hi1 = _mm256_permute2x128_si256(ev2p, od2p, 0x20);
        const __m256i bh_hi1 = _mm256_permute2x128_si256(ev2p, od2p, 0x31);
        const __m256i bl_hi2 = _mm256_permute2x128_si256(ev3p, od3p, 0x20);
        const __m256i bh_hi2 = _mm256_permute2x128_si256(ev3p, od3p, 0x31);

        const __m256 col_scales = _mm256_cvtph_ps(
            _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)bp[b].d), changemask));

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
            Q4IX8_MULSUM_INTO(iacc, bl_lo1, _mm256_shuffle_epi32(a0, 0));
            Q4IX8_MULSUM_INTO(iacc, bh_lo1, _mm256_shuffle_epi32(a0, 85));
            Q4IX8_MULSUM_INTO(iacc, bl_lo2, _mm256_shuffle_epi32(a0, 170));
            Q4IX8_MULSUM_INTO(iacc, bh_lo2, _mm256_shuffle_epi32(a0, 255));
            Q4IX8_MULSUM_INTO(iacc, bl_hi1, _mm256_shuffle_epi32(a1, 0));
            Q4IX8_MULSUM_INTO(iacc, bh_hi1, _mm256_shuffle_epi32(a1, 85));
            Q4IX8_MULSUM_INTO(iacc, bl_hi2, _mm256_shuffle_epi32(a1, 170));
            Q4IX8_MULSUM_INTO(iacc, bh_hi2, _mm256_shuffle_epi32(a1, 255));

            acc[r] = Q4IX8_FMADD(_mm256_cvtepi32_ps(iacc), sd, acc[r]);
        }
    }
#undef Q4IX8_MULSUM_INTO

    for (int r = 0; r < 4; r++) {
        __m256 result = _mm256_permutevar8x32_ps(acc[r], finalpermutemask);
        _mm256_storeu_ps(out4x8[r], result);
    }
}

/* Tiles the full [nr x nc] output over (nr/4) x (nc/8) 4x8 tiles */
static int sgemm_q4ix8_q8x4_avx2(
        int k, const block_q4i_0x8 *bp_start,
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

        const block_q4i_0x8 *bp = bp_start + (size_t)xt * nb;
        const block_q8_0x4 *ap = ap_start + (size_t)yt * nb;

        float out4x8[4][8];
        sgemm_q4ix8_q8x4_avx2_tile(bp, ap, nb, out4x8);

        for (int r = 0; r < 4; r++) {
            memcpy(s + (size_t)(y + r) * bs + x, out4x8[r], 8 * sizeof(float));
        }
    }
    return anr;
}
#endif /* AVX2 + F16C */

/* ============================================================
 * AVX-512: 16x16 tiled GEMM for Q4I_0_8_8 (pre-dequantized int8)
 * NOTE: Disabled -- has activation-layout bug (assumes byte-interleaved
 * activations instead of contiguous rows). See Q4_0_8_8 AVX-512 note.
 * ============================================================ */
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
static int sgemm_q4ix8_q8x4_avx512(
        int k, const block_q4i_0x8 *bp,
        const block_q8_0x4 *ap,
        float *s, size_t bs, int nr, int nc,
        int ith, int nth)
{
    const int nb = k / 32;
    const int anr = nr - nr % 16;
    const int anc = nc - nc % 16;

    int n_ytiles = anr / 16;
    int n_xtiles = anc / 16;
    int total_tiles = n_ytiles * n_xtiles;
    int duty = (total_tiles + nth - 1) / nth;
    int start = duty * ith;
    int end = start + duty;
    if (end > total_tiles) end = total_tiles;

    for (int job = start; job < end; job++) {
        int yt = job / n_xtiles;
        int xt = job % n_xtiles;
        int y = yt * 4;
        int xg = xt * 2;

        const block_q8_0x4 *ap4[4];
        ap4[0] = ap + (y * nb);
        for (int i = 0; i < 3; i++) ap4[i+1] = ap4[i] + nb;

        const block_q4i_0x8 *bp0 = bp + (xg * nb);
        const block_q4i_0x8 *bp1 = bp + ((xg+1) * nb);

        __m512 acc[16];
        for (int i = 0; i < 16; i++) acc[i] = _mm512_setzero_ps();

        for (int b = 0; b < nb; b++) {
            /* Load pre-dequantized int8 weights directly.
             * Each block_q4i_0x8.qs[256] has 8 chunks of 32 bytes:
             *   chunks 0-3: even rows {0,1,4,5} vals 0-7,8-15,16-23,24-31
             *   chunks 4-7: odd  rows {2,3,6,7} vals 0-7,8-15,16-23,24-31
             *
             * We load 256-bit chunks from bp0 and bp1, merge to 512-bit,
             * then shuffle for dpbusd.
             */

            /* Even group: chunks 0-3 from bp0 and bp1 */
            const __m256i we00 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs + 0));
            const __m256i we01 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs + 32));
            const __m256i we02 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs + 64));
            const __m256i we03 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs + 96));
            const __m256i we10 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs + 0));
            const __m256i we11 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs + 32));
            const __m256i we12 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs + 64));
            const __m256i we13 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs + 96));

            /* Odd group: chunks 4-7 (offset 128) from bp0 and bp1 */
            const __m256i wo00 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs + 128));
            const __m256i wo01 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs + 160));
            const __m256i wo02 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs + 192));
            const __m256i wo03 = _mm256_loadu_si256((const __m256i*)(bp0[b].qs + 224));
            const __m256i wo10 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs + 128));
            const __m256i wo11 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs + 160));
            const __m256i wo12 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs + 192));
            const __m256i wo13 = _mm256_loadu_si256((const __m256i*)(bp1[b].qs + 224));

            /* Merge to 512-bit: even and odd groups */
            const __m512i re0 = _mm512_inserti32x8(_mm512_castsi256_si512(we00), we10, 1);
            const __m512i re1 = _mm512_inserti32x8(_mm512_castsi256_si512(we01), we11, 1);
            const __m512i re2 = _mm512_inserti32x8(_mm512_castsi256_si512(we02), we12, 1);
            const __m512i re3 = _mm512_inserti32x8(_mm512_castsi256_si512(we03), we13, 1);
            const __m512i ro0 = _mm512_inserti32x8(_mm512_castsi256_si512(wo00), wo10, 1);
            const __m512i ro1 = _mm512_inserti32x8(_mm512_castsi256_si512(wo01), wo11, 1);
            const __m512i ro2 = _mm512_inserti32x8(_mm512_castsi256_si512(wo02), wo12, 1);
            const __m512i ro3 = _mm512_inserti32x8(_mm512_castsi256_si512(wo03), wo13, 1);

            /* Shuffle weights for dpbusd (broadcast within quad)
             * 136 = 0x88: broadcast lane 0 of each row-pair
             * 221 = 0xDD: broadcast lane 1 of each row-pair */
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

                /* Split: low 16 bytes = A0/A1, high 16 = A2/A3 */
                __m256i a0l = _mm256_permute2f128_si256(a0, a0, 0);
                __m256i a0h = _mm256_permute2f128_si256(a0, a0, 17);
                __m256i a1l = _mm256_permute2f128_si256(a1, a1, 0);
                __m256i a1h = _mm256_permute2f128_si256(a1, a1, 17);
                __m256i a2l = _mm256_permute2f128_si256(a2, a2, 0);
                __m256i a2h = _mm256_permute2f128_si256(a2, a2, 17);
                __m256i a3l = _mm256_permute2f128_si256(a3, a3, 0);
                __m256i a3h = _mm256_permute2f128_si256(a3, a3, 17);

                /* Expand to 512-bit */
                const __m512i l01_0 = _mm512_inserti32x8(_mm512_castsi256_si512(a0l), a0l, 1);
                const __m512i l23_0 = _mm512_inserti32x8(_mm512_castsi256_si512(a0h), a0h, 1);
                const __m512i l01_1 = _mm512_inserti32x8(_mm512_castsi256_si512(a1l), a1l, 1);
                const __m512i l23_1 = _mm512_inserti32x8(_mm512_castsi256_si512(a1h), a1h, 1);
                const __m512i l01_2 = _mm512_inserti32x8(_mm512_castsi256_si512(a2l), a2l, 1);
                const __m512i l23_2 = _mm512_inserti32x8(_mm512_castsi256_si512(a2h), a2h, 1);
                const __m512i l01_3 = _mm512_inserti32x8(_mm512_castsi256_si512(a3l), a3l, 1);
                const __m512i l23_3 = _mm512_inserti32x8(_mm512_castsi256_si512(a3h), a3h, 1);

                /* Shuffle activations: interleave A0/A1 pairs */
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

                /* dpbusd MAC */
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

                /* Straighten */
                __m512i row0 = _mm512_mask_blend_epi32(0xCCCC, i00, _mm512_shuffle_epi32(i01, 78));
                __m512i row1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(i00, 78), i01);
                __m512i row2 = _mm512_mask_blend_epi32(0xCCCC, i10, _mm512_shuffle_epi32(i11, 78));
                __m512i row3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(i10, 78), i11);

                /* Activation scales */
                __m128i rs_f16 = _mm_loadl_epi64((const __m128i*)a[b].d);
                rs_f16 = _mm_shuffle_epi32(rs_f16, 68);
                __m512 rs = PICOLM_F32Cx16_REPEAT_LOAD(rs_f16);

                __m512 rs0 = _mm512_shuffle_ps(rs, rs, 0);
                __m512 rs1 = _mm512_shuffle_ps(rs, rs, 85);
                __m512 rs2 = _mm512_shuffle_ps(rs, rs, 170);
                __m512 rs3 = _mm512_shuffle_ps(rs, rs, 255);

                /* Scale and accumulate */
                acc[rp*4]     = _mm512_fmadd_ps(_mm512_cvtepi32_ps(row0), _mm512_mul_ps(cs, rs0), acc[rp*4]);
                acc[rp*4 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(row1), _mm512_mul_ps(cs, rs1), acc[rp*4+1]);
                acc[rp*4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(row2), _mm512_mul_ps(cs, rs2), acc[rp*4+2]);
                acc[rp*4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(row3), _mm512_mul_ps(cs, rs3), acc[rp*4+3]);
            }
        }

        /* Store output */
        for (int i = 0; i < 16; i++) {
            _mm512_storeu_ps((float*)(s + ((y*4+i)*bs + xg*8)), acc[i]);
        }
    }
    return anr;
}
#endif

/* ============================================================
 * Top-level dispatcher for Q4I_0_8_8
 * ============================================================ */
int sgemm_q4i_0x8_q8_0x4(int nr, int nc, int k,
        const void *vx, const void *vy, float *s, size_t bs,
        int ith, int nth)
{
    const block_q4i_0x8 *bp = (const block_q4i_0x8 *)vx;
    const block_q8_0x4 *ap = (const block_q8_0x4 *)vy;

    if (nc < 8 || nr < 4 || k % 32 != 0) return 0;

    /* AVX2 path is the verified/primary path (same strategy as
     * sgemm_q4_0x8_q8_0x4 which uses AVX2 on AVX-512 hosts too).
     * The AVX-512 16x16 kernel below has the same activation-layout
     * bug as the Q4_0_8_8 AVX-512 kernel and is kept for reference. */
#if defined(__AVX2__) && defined(__F16C__)
    return sgemm_q4ix8_q8x4_avx2(k, bp, ap, s, bs, nr, nc, ith, nth);
#else
    (void)nr; (void)nc; (void)k; (void)vx; (void)vy;
    (void)s; (void)bs; (void)ith; (void)nth;
    (void)bp; (void)ap;
    return 0;
#endif
}
