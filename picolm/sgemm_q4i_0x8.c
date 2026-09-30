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
 * AVX-512: 4x16 tiled GEMM for Q4I_0_8_8 (pre-dequantized int8)
 *
 * Same activation-layout strategy as the AVX2 kernel: per-row loads
 * from block_q8_0x4.qs[r*32..r*32+31], broadcast to 512-bit.
 * Weight side: permutevar8x32 + permute2x128 to reorder even/odd
 * chunks into the interleaved {0,4,1,5,2,6,3,7} pattern, then
 * merge bp0+bp1 to 512-bit (16 weight rows per register).
 *
 * Each tile: 4 activation rows x 16 weight columns.
 * ============================================================ */
#if defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VNNI__)
static int sgemm_q4ix8_q8x4_avx512(
        int k, const block_q4i_0x8 *bp_start,
        const block_q8_0x4 *ap_start,
        float *s, size_t bs, int nr, int nc,
        int ith, int nth)
{
    const int nb = k / 32;
    const int anr = nr - nr % 4;
    const int anc = nc - nc % 16;
    if (anr <= 0 || anc <= 0) return 0;

    const int n_ytiles = anr / 4;
    const int n_xtiles = anc / 16;
    const int total_tiles = n_ytiles * n_xtiles;
    if (nth < 1) nth = 1;
    const int duty = (total_tiles + nth - 1) / nth;
    int start = duty * ith;
    int end = start + duty;
    if (end > total_tiles) end = total_tiles;

    /* Lane reorder: [r0,r1,r4,r5] -> [r0,r4,r1,r5] within each half */
    const __m256i perm_row = _mm256_set_epi32(7,3,5,1,6,2,4,0);

    /* Final output permute: from interleaved lane order to sequential rows.
     * Lanes {0..7} = bp0 rows {0,4,1,5,2,6,3,7}
     * Lanes {8..15} = bp1 rows {8,12,9,13,10,14,11,15}
     * Sequential: row0=lane0, row1=lane2, row2=lane4, row3=lane6,
     *             row4=lane1, row5=lane3, row6=lane5, row7=lane7,
     *             row8=lane8, row9=lane10, row10=lane12, row11=lane14,
     *             row12=lane9, row13=lane11, row14=lane13, row15=lane15 */
    const __m512i out_perm = _mm512_set_epi32(15,13,11,9,14,12,10,8,7,5,3,1,6,4,2,0);

    for (int job = start; job < end; job++) {
        const int yt = job / n_xtiles;
        const int xt = job % n_xtiles;
        const int y = yt * 4;
        const int x = xt * 16;

        const block_q4i_0x8 *bp0 = bp_start + (size_t)(xt * 2) * nb;
        const block_q4i_0x8 *bp1 = bp_start + (size_t)(xt * 2 + 1) * nb;
        const block_q8_0x4 *ap = ap_start + (size_t)yt * nb;

        __m512 acc[4];
        acc[0] = acc[1] = acc[2] = acc[3] = _mm512_setzero_ps();

        for (int b = 0; b < nb; b++) {
            /* ---- Weight side: load + reorder, reused for all 4 act rows ---- */
            /* Even chunks: rows {0,1,4,5} from bp0, {8,9,12,13} from bp1 */
            const __m256i ev0_lo = _mm256_loadu_si256((const __m256i *)(bp0[b].qs));
            const __m256i ev0_hi = _mm256_loadu_si256((const __m256i *)(bp1[b].qs));
            const __m256i ev1_lo = _mm256_loadu_si256((const __m256i *)(bp0[b].qs + 32));
            const __m256i ev1_hi = _mm256_loadu_si256((const __m256i *)(bp1[b].qs + 32));
            const __m256i ev2_lo = _mm256_loadu_si256((const __m256i *)(bp0[b].qs + 64));
            const __m256i ev2_hi = _mm256_loadu_si256((const __m256i *)(bp1[b].qs + 64));
            const __m256i ev3_lo = _mm256_loadu_si256((const __m256i *)(bp0[b].qs + 96));
            const __m256i ev3_hi = _mm256_loadu_si256((const __m256i *)(bp1[b].qs + 96));

            /* Odd chunks: rows {2,3,6,7} from bp0, {10,11,14,15} from bp1 */
            const __m256i od0_lo = _mm256_loadu_si256((const __m256i *)(bp0[b].qs + 128));
            const __m256i od0_hi = _mm256_loadu_si256((const __m256i *)(bp1[b].qs + 128));
            const __m256i od1_lo = _mm256_loadu_si256((const __m256i *)(bp0[b].qs + 160));
            const __m256i od1_hi = _mm256_loadu_si256((const __m256i *)(bp1[b].qs + 160));
            const __m256i od2_lo = _mm256_loadu_si256((const __m256i *)(bp0[b].qs + 192));
            const __m256i od2_hi = _mm256_loadu_si256((const __m256i *)(bp1[b].qs + 192));
            const __m256i od3_lo = _mm256_loadu_si256((const __m256i *)(bp0[b].qs + 224));
            const __m256i od3_hi = _mm256_loadu_si256((const __m256i *)(bp1[b].qs + 224));

            /* Permute lanes: reorder rows within each 256-bit chunk */
            const __m256i ev0p_lo = _mm256_permutevar8x32_epi32(ev0_lo, perm_row);
            const __m256i ev0p_hi = _mm256_permutevar8x32_epi32(ev0_hi, perm_row);
            const __m256i od0p_lo = _mm256_permutevar8x32_epi32(od0_lo, perm_row);
            const __m256i od0p_hi = _mm256_permutevar8x32_epi32(od0_hi, perm_row);
            const __m256i ev1p_lo = _mm256_permutevar8x32_epi32(ev1_lo, perm_row);
            const __m256i ev1p_hi = _mm256_permutevar8x32_epi32(ev1_hi, perm_row);
            const __m256i od1p_lo = _mm256_permutevar8x32_epi32(od1_lo, perm_row);
            const __m256i od1p_hi = _mm256_permutevar8x32_epi32(od1_hi, perm_row);
            const __m256i ev2p_lo = _mm256_permutevar8x32_epi32(ev2_lo, perm_row);
            const __m256i ev2p_hi = _mm256_permutevar8x32_epi32(ev2_hi, perm_row);
            const __m256i od2p_lo = _mm256_permutevar8x32_epi32(od2_lo, perm_row);
            const __m256i od2p_hi = _mm256_permutevar8x32_epi32(od2_hi, perm_row);
            const __m256i ev3p_lo = _mm256_permutevar8x32_epi32(ev3_lo, perm_row);
            const __m256i ev3p_hi = _mm256_permutevar8x32_epi32(ev3_hi, perm_row);
            const __m256i od3p_lo = _mm256_permutevar8x32_epi32(od3_lo, perm_row);
            const __m256i od3p_hi = _mm256_permutevar8x32_epi32(od3_hi, perm_row);

            /* Merge even+odd halves within each block, then merge bp0+bp1 to 512-bit.
             * bl_k = vals k*8..k*8+3 for all 16 rows
             * bh_k = vals k*8+4..k*8+7 for all 16 rows
             * Lane order: bp0 rows {0,4,1,5,2,6,3,7}, bp1 rows {8,12,9,13,10,14,11,15} */
            const __m512i bl_0 = _mm512_inserti32x8(_mm512_castsi256_si512(
                _mm256_permute2x128_si256(ev0p_lo, od0p_lo, 0x20)),
                _mm256_permute2x128_si256(ev0p_hi, od0p_hi, 0x20), 1);
            const __m512i bh_0 = _mm512_inserti32x8(_mm512_castsi256_si512(
                _mm256_permute2x128_si256(ev0p_lo, od0p_lo, 0x31)),
                _mm256_permute2x128_si256(ev0p_hi, od0p_hi, 0x31), 1);
            const __m512i bl_1 = _mm512_inserti32x8(_mm512_castsi256_si512(
                _mm256_permute2x128_si256(ev1p_lo, od1p_lo, 0x20)),
                _mm256_permute2x128_si256(ev1p_hi, od1p_hi, 0x20), 1);
            const __m512i bh_1 = _mm512_inserti32x8(_mm512_castsi256_si512(
                _mm256_permute2x128_si256(ev1p_lo, od1p_lo, 0x31)),
                _mm256_permute2x128_si256(ev1p_hi, od1p_hi, 0x31), 1);
            const __m512i bl_2 = _mm512_inserti32x8(_mm512_castsi256_si512(
                _mm256_permute2x128_si256(ev2p_lo, od2p_lo, 0x20)),
                _mm256_permute2x128_si256(ev2p_hi, od2p_hi, 0x20), 1);
            const __m512i bh_2 = _mm512_inserti32x8(_mm512_castsi256_si512(
                _mm256_permute2x128_si256(ev2p_lo, od2p_lo, 0x31)),
                _mm256_permute2x128_si256(ev2p_hi, od2p_hi, 0x31), 1);
            const __m512i bl_3 = _mm512_inserti32x8(_mm512_castsi256_si512(
                _mm256_permute2x128_si256(ev3p_lo, od3p_lo, 0x20)),
                _mm256_permute2x128_si256(ev3p_hi, od3p_hi, 0x20), 1);
            const __m512i bh_3 = _mm512_inserti32x8(_mm512_castsi256_si512(
                _mm256_permute2x128_si256(ev3p_lo, od3p_lo, 0x31)),
                _mm256_permute2x128_si256(ev3p_hi, od3p_hi, 0x31), 1);

            /* Weight column scales: 16 FP16 -> 16 FP32.
             * Reorder from sequential {d[0],d[1],..,d[7]} to interleaved
             * {d[0],d[4],d[1],d[5],d[2],d[6],d[3],d[7]} to match the
             * permuted weight lane order. */
            const __m128i changemask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4,
                                                     11, 10, 3, 2, 9, 8, 1, 0);
            const __m128i d0_shuf = _mm_shuffle_epi8(
                _mm_loadu_si128((const __m128i *)bp0[b].d), changemask);
            const __m128i d1_shuf = _mm_shuffle_epi8(
                _mm_loadu_si128((const __m128i *)bp1[b].d), changemask);
            const __m512 cs = _mm512_cvtph_ps(_mm256_set_m128i(d1_shuf, d0_shuf));

            /* ---- Activation side: per-row loads (correct layout) ---- */
            for (int r = 0; r < 4; r++) {
                const uint8_t *aq = (const uint8_t *)ap[b].qs + r * 32;
                __m256i a0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)aq));
                __m256i a1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(aq + 16)));
                a0 = _mm256_permute2f128_si256(a0, a0, 0);
                a1 = _mm256_permute2f128_si256(a1, a1, 0);

                /* Expand to 512-bit: duplicate 256-bit halves */
                const __m512i a0_512 = _mm512_inserti32x8(_mm512_castsi256_si512(a0), a0, 1);
                const __m512i a1_512 = _mm512_inserti32x8(_mm512_castsi256_si512(a1), a1, 1);

                const __m512 row_scale = _mm512_set1_ps(fp16_to_fp32_lookup((uint16_t)ap[b].d[r]));
                const __m512 sd = _mm512_mul_ps(cs, row_scale);

                /* dpbusd MAC: 8 chunks x 2 halves = 8 dpbusd per row */
                __m512i iacc = _mm512_setzero_epi32();
                iacc = dpbusd_512(iacc, bl_0, _mm512_shuffle_epi32(a0_512, 0));
                iacc = dpbusd_512(iacc, bh_0, _mm512_shuffle_epi32(a0_512, 85));
                iacc = dpbusd_512(iacc, bl_1, _mm512_shuffle_epi32(a0_512, 170));
                iacc = dpbusd_512(iacc, bh_1, _mm512_shuffle_epi32(a0_512, 255));
                iacc = dpbusd_512(iacc, bl_2, _mm512_shuffle_epi32(a1_512, 0));
                iacc = dpbusd_512(iacc, bh_2, _mm512_shuffle_epi32(a1_512, 85));
                iacc = dpbusd_512(iacc, bl_3, _mm512_shuffle_epi32(a1_512, 170));
                iacc = dpbusd_512(iacc, bh_3, _mm512_shuffle_epi32(a1_512, 255));

                acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc), sd, acc[r]);
            }
        }

        /* Store: permute from interleaved lane order to sequential rows */
        for (int r = 0; r < 4; r++) {
            __m512 result = _mm512_permutexvar_ps(out_perm, acc[r]);
            _mm512_storeu_ps((float*)(s + ((y + r) * bs + x)), result);
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

    int done = 0;

    /* AVX-512 path: 4x16 tiles, VNNI dpbusd, per-row activation loads.
     * Set PICOLM_Q4I_AVX512=0 to disable and use AVX2 instead. */
#if defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VNNI__)
    {
        static int use_avx512 = -1;
        if (use_avx512 < 0) {
            const char *env = getenv("PICOLM_Q4I_AVX512");
            use_avx512 = (!env || env[0] != '0');
        }
        if (use_avx512)
            done = sgemm_q4ix8_q8x4_avx512(k, bp, ap, s, bs, nr, nc, ith, nth);
    }
#endif

    /* AVX2 path: 4x8 tiles, maddubs+madd, per-row activation loads.
     * Verified correct (same strategy as Q4_0_8_8 AVX2 kernel). */
#if defined(__AVX2__) && defined(__F16C__)
    if (!done)
        done = sgemm_q4ix8_q8x4_avx2(k, bp, ap, s, bs, nr, nc, ith, nth);
#endif

    if (!done) {
        (void)nr; (void)nc; (void)k; (void)vx; (void)vy;
        (void)s; (void)bs; (void)ith; (void)nth;
        (void)bp; (void)ap;
    }
    return done;
}
