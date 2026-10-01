/* ================================================================
 * IQ2_K (plain, GGUF type 137) x Q8_K AVX2 GEMV and GEMM kernels
 * ================================================================
 * Weights: block_iq2_k (GGUF type 137), 76 bytes/block
 * Activations: block_q8_K
 *
 * Per subblock: 32 values from qs[j]+qs[j+16] (j=0..15), shift 0,2,4,6
 * ib32 0-3 use qs[0..31], ib32 4-7 use qs[32..63]
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "quant.h"
#if defined(__AVX2__)

#define QK_K 256

/* Method (per 256-value block):
 *   1. Decode all 8 sub-blocks (32 values each) to signed int8: w[i].
 *      qs chunk (32 bytes) is loaded as one ymm: lane 0 = values 0..15,
 *      lane 1 = values 16..31, i.e. natural order. 2-bit index =
 *      (qs >> 2*(i&3)) & 3, plus 4 where the matching extra bit is set
 *      (bit 2i for lane 0, bit 2i+1 for lane 1), then one pshufb into the
 *      8-entry table (iq2nl_values, repeated twice per 128-bit lane).
 *   2. sp[i] = int16 scale vector: lanes 0-7 = scale of the first 16 values,
 *      lanes 8-15 = scale of the next 16 (nibble - 8).
 *   3. Dot: abs/sign + maddubs (int16) + madd_epi16 with sp[i] (applies the
 *      scale, reduces to int32x8). One float conversion per block, one
 *      horizontal sum per row.
 * Int range: |w| <= 31, |y| <= 127, |scale| <= 8 -> no overflow anywhere. */

static inline float iq2_hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

/* 8-entry table {normal x4, shifted x4}, repeated so every 128-bit lane
 * holds it twice (pshufb index is 0..7 only). Single source: iq2nl_values. */
static inline __m256i iq2_lut(void) {
    int64_t t;
    memcpy(&t, iq2nl_values, sizeof(t));
    return _mm256_set1_epi64x(t);
}

static inline void iq2_decode_block(const block_iq2_k *x, __m256i lut,
                                    __m256i w[8], __m256i sp[8]) {
    const __m256i m3   = _mm256_set1_epi8(3);
    const __m256i four = _mm256_set1_epi8(4);
    const __m256i ex   = _mm256_set1_epi16((short)x->extra);
    const __m256i ctrl = _mm256_setr_epi8(0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
                                          2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3);

    /* scales[i]: low nibble = group 2i, high nibble = group 2i+1 */
    const __m128i s8  = _mm_loadl_epi64((const __m128i *)x->scales);
    const __m128i lo4 = _mm_and_si128(s8, _mm_set1_epi8(0x0f));
    const __m128i hi4 = _mm_and_si128(_mm_srli_epi16(s8, 4), _mm_set1_epi8(0x0f));
    const __m128i sg  = _mm_sub_epi8(_mm_unpacklo_epi8(lo4, hi4), _mm_set1_epi8(8));
    const __m256i sc16 = _mm256_cvtepi8_epi16(sg);              /* int16[g], g = 0..15 */

    for (int i = 0; i < 8; ++i) {
        const __m256i qs = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * (i >> 2)));
        const __m256i q  = _mm256_and_si256(_mm256_srl_epi16(qs, _mm_cvtsi32_si128(2 * (i & 3))), m3);

        const __m256i bm  = _mm256_inserti128_si256(_mm256_set1_epi16((short)(1u << (2 * i))),
                                                    _mm_set1_epi16((short)(1u << (2 * i + 1))), 1);
        const __m256i sel = _mm256_cmpeq_epi16(_mm256_and_si256(ex, bm), bm);
        w[i] = _mm256_shuffle_epi8(lut, _mm256_add_epi8(q, _mm256_and_si256(sel, four)));

        /* element i of sc16 viewed as int32 = (scale[2i], scale[2i+1]) */
        sp[i] = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(sc16, _mm256_set1_epi32(i)), ctrl);
    }
}

static inline __m256i iq2_block_dot(const __m256i w[8], const __m256i sp[8], const block_q8_K *yb) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < 8; ++i) {
        const __m256i y = _mm256_loadu_si256((const __m256i *)(yb->qs + 32 * i));
        const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w[i], w[i]), _mm256_sign_epi8(y, w[i]));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, sp[i]));
    }
    return acc;
}

void vec_dot_iq2_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) {
    assert(n % QK_K == 0);
    const block_iq2_k *x = (const block_iq2_k *)vx;
    const block_q8_K *y = (const block_q8_K *)wy;
    const int nb = n / QK_K;
    const __m256i lut = iq2_lut();

    __m256 fsum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        __m256i w[8], sp[8];
        iq2_decode_block(&x[ibl], lut, w, sp);
        const __m256i acc = iq2_block_dot(w, sp, &y[ibl]);
        const float dy = fp16_to_fp32_lookup(x[ibl].d) * y[ibl].d;
        fsum = _mm256_add_ps(fsum, _mm256_mul_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(dy)));
    }
    *out = iq2_hsum_ps(fsum);
}

/* Weight rows [ii] x activation columns, tiled IQ2K_GEMM_TILE columns per job:
 * each weight block is decoded once and reused for every column in the tile.
 * out[row + col * bs]. */
#define IQ2K_GEMM_TILE 4

int sgemm_iq2_k_q8_k_avx2(int nrows, int ncols, int k,
                           const void *vx, const void *vy,
                           float *out, size_t bs,
                           int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k % QK_K != 0)
        return 0;

    const int nb = k / QK_K;
    const __m256i lut = iq2_lut();

    int64_t ytiles = nrows;
    int64_t xtiles = ncols / IQ2K_GEMM_TILE;
    int64_t n_tail = ncols - xtiles * IQ2K_GEMM_TILE;
    int64_t xtiles_ext = xtiles + (n_tail > 0 ? 1 : 0);
    int64_t tiles = ytiles * xtiles_ext;
    if (tiles <= 0) return 0;

    int64_t duty = (tiles + nth - 1) / nth;
    int64_t start = duty * ith;
    int64_t end = start + duty;
    if (end > tiles) end = tiles;

    size_t w_row_bytes = nb * sizeof(block_iq2_k);
    size_t a_row_bytes = nb * sizeof(block_q8_K);

    for (int64_t job = start; job < end; job++) {
        int64_t ii = job / xtiles_ext;
        int64_t xt = job % xtiles_ext;
        int64_t jj = xt * IQ2K_GEMM_TILE;
        int nc = (xt < xtiles) ? IQ2K_GEMM_TILE : (int)n_tail;
        if (nc < 1) nc = 1;

        const block_iq2_k *x = (const block_iq2_k *)((const char *)vx + ii * w_row_bytes);

        __m256 fs[IQ2K_GEMM_TILE];
        for (int c = 0; c < nc; c++) fs[c] = _mm256_setzero_ps();

        for (int ibl = 0; ibl < nb; ibl++) {
            __m256i w[8], sp[8];
            iq2_decode_block(&x[ibl], lut, w, sp);
            const float d = fp16_to_fp32_lookup(x[ibl].d);
            for (int c = 0; c < nc; c++) {
                const block_q8_K *yb = (const block_q8_K *)((const char *)vy + (jj + c) * a_row_bytes) + ibl;
                const __m256i acc = iq2_block_dot(w, sp, yb);
                fs[c] = _mm256_add_ps(fs[c],
                        _mm256_mul_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(d * yb->d)));
            }
        }
        for (int c = 0; c < nc; c++)
            out[ii + (jj + c) * bs] = iq2_hsum_ps(fs[c]);
    }
    return nrows;
}
#else
/* Non-AVX2 stubs for cross-compilation compatibility. Callers must guard on __AVX2__. */
void vec_dot_iq2_k_q8_k_avx2(const void *vx, const void *wy, int n, float *out) { (void)vx; (void)wy; (void)n; if (out) *out = 0.0f; }
int sgemm_iq2_k_q8_k_avx2(int nrows, int ncols, int k, const void *vx, const void *vy, float *out, size_t bs, int ith, int nth) { (void)vx; (void)vy; (void)nrows; (void)ncols; (void)k; (void)out; (void)bs; (void)ith; (void)nth; return 0; }
#endif

/* ================================================================
 * IQ2_K plain x Q8_K ARM NEON GEMV kernel
 * ================================================================
 * Uses vqtbl1q_u8 for LUT-based 2-bit dequantization, then
 * vmull_s8/vmlal_s8 for int8 MAC with Q8_K activations.
 * Per-subblock scales applied via vreinterpretq_s32_u32 + vmulq.
 * ================================================================ */
#if defined(PICOLM_NEON)
#include <arm_neon.h>

void vec_dot_iq2_k_q8_k_neon(const void *vx, const void *wy, int n, float *out) {
    assert(n % QK_K == 0);

    const block_iq2_k *iq2 = (const block_iq2_k *)vx;
    const block_q8_K *qk = (const block_q8_K *)wy;
    const int nb = n / QK_K;

    /* LUT: 8 entries (4 normal + 4 shifted), repeated 4x for 32-byte alignment.
     * vqtbl1q_u8 uses the low 4 bits of each index byte, so indices 0-7 map
     * to the first 8 entries. */
    static const int8_t kvalues_iq2nl[32] = {
        -31, -13, 1, 17, -26, -8, 6, 22,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    };
    const uint8x16_t lut = vld1q_u8((const uint8_t *)kvalues_iq2nl);
    const uint8x16_t m03 = vdupq_n_u8(0x03);

    float result = 0.0f;

    for (int ibl = 0; ibl < nb; ibl++) {
        float d = fp16_to_fp32_lookup(iq2[ibl].d);
        float q8_scale = qk[ibl].d;
        float dy = d * q8_scale;

        uint16_t extra = iq2[ibl].extra;
        const uint8_t *qs = iq2[ibl].qs;
        const int8_t *q8_base = qk[ibl].qs;

        int32x4_t isum = vdupq_n_s32(0);

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            int scale_lo = (iq2[ibl].scales[ib32] & 0xf) - 8;
            int scale_hi = (iq2[ibl].scales[ib32] >> 4) - 8;

            /* LUT table selection: each extra bit selects normal (0) or shifted (+4) */
            int shift_idx  = ((extra >> 0) & 1) * 4;
            int shift_idx2 = ((extra >> 1) & 1) * 4;
            extra >>= 2;

            int shift = 2 * (ib32 % 4);
            int qs_off = (ib32 / 4) * 32;

            /* Load 32 activations from q8_base + ib32*32 */
            int8x16_t y_lo = vld1q_s8(q8_base + ib32 * 32);
            int8x16_t y_hi = vld1q_s8(q8_base + ib32 * 32 + 16);

            /* Extract low 2 bits from qs[qs_off .. qs_off+15] and qs[qs_off+16 .. qs_off+31] */
            uint8x16_t lb_lo = vld1q_u8(qs + qs_off + 0);
            uint8x16_t lb_hi = vld1q_u8(qs + qs_off + 16);

            /* Right-shift to extract the 2-bit values at the current shift position */
            uint8x16_t ql_lo, ql_hi;
            if (shift == 0) {
                ql_lo = lb_lo;
                ql_hi = lb_hi;
            } else if (shift == 2) {
                ql_lo = vshrq_n_u8(lb_lo, 2);
                ql_hi = vshrq_n_u8(lb_hi, 2);
            } else if (shift == 4) {
                ql_lo = vshrq_n_u8(lb_lo, 4);
                ql_hi = vshrq_n_u8(lb_hi, 4);
            } else {
                ql_lo = vshrq_n_u8(lb_lo, 6);
                ql_hi = vshrq_n_u8(lb_hi, 6);
            }
            ql_lo = vandq_u8(ql_lo, m03);
            ql_hi = vandq_u8(ql_hi, m03);

            /* LUT lookup: 2-bit index -> int8 dequantized value.
             * Add shift_idx to select the correct LUT table. */
            int8x16_t qx_lo = vreinterpretq_s8_u8(vqtbl1q_u8(lut,
                vaddq_u8(ql_lo, vdupq_n_u8(shift_idx))));
            int8x16_t qx_hi = vreinterpretq_s8_u8(vqtbl1q_u8(lut,
                vaddq_u8(ql_hi, vdupq_n_u8(shift_idx2))));

            /* int8 MAC: sum(qx_lo * y_lo) and sum(qx_hi * y_hi)
             * Using vmull_s8 (widening multiply) + vpaddlq_s16 -> int32 */
            int32x4_t s_lo = vaddq_s32(
                vpaddlq_s16(vmull_s8(vget_low_s8(qx_lo), vget_low_s8(y_lo))),
                vpaddlq_s16(vmull_high_s8(qx_lo, y_lo)));
            int32x4_t s_hi = vaddq_s32(
                vpaddlq_s16(vmull_s8(vget_low_s8(qx_hi), vget_low_s8(y_hi))),
                vpaddlq_s16(vmull_high_s8(qx_hi, y_hi)));

            /* Apply per-subblock scales: s_lo *= scale_lo, s_hi *= scale_hi */
            int32x4_t s_lo_s = vmulq_n_s32(s_lo, scale_lo);
            int32x4_t s_hi_s = vmulq_n_s32(s_hi, scale_hi);

            isum = vaddq_s32(isum, vaddq_s32(s_lo_s, s_hi_s));
        }

        /* Horizontal sum of 4 int32 lanes -> single int32 */
        int total = vaddlvq_s32(isum);
        result += (float)total * dy;
    }

    *out = result;
}

/* IQ2_K plain x Q8_K ARM NEON GEMM kernel.
 * Tiled: 4 weight rows x 2 activation rows per tile. */
int sgemm_iq2_k_q8_k_neon(int nrows, int ncols, int k,
                           const void *vx, const void *vy,
                           float *out, size_t bs,
                           int ith, int nth) {
    if (nrows < 1 || ncols < 1 || k % QK_K != 0)
        return 0;

    const int nb = k / QK_K;

    static const int8_t kvalues_iq2nl[32] = {
        -31, -13, 1, 17, -26, -8, 6, 22,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    };
    const uint8x16_t lut = vld1q_u8((const uint8_t *)kvalues_iq2nl);
    const uint8x16_t m03 = vdupq_n_u8(0x03);

    int64_t ytiles = nrows / 4;
    int64_t xtiles = ncols / 2;
    int64_t n_tail = ncols - xtiles * 2;
    int64_t xtiles_ext = xtiles + (n_tail > 0 ? 1 : 0);
    int64_t tiles = ytiles * xtiles_ext;
    if (tiles <= 0) return 0;

    int64_t duty = (tiles + nth - 1) / nth;
    int64_t start = duty * ith;
    int64_t end = start + duty;
    if (end > tiles) end = tiles;

    size_t w_row_bytes = nb * sizeof(block_iq2_k);
    size_t a_row_bytes = nb * sizeof(block_q8_K);

    for (int64_t job = start; job < end; job++) {
        int64_t ii = (job / xtiles_ext) * 4;
        int64_t xt = job % xtiles_ext;
        int64_t jj = xt * 2;
        int64_t ncols_tile = (xt < xtiles) ? 2 : n_tail;
        if (ncols_tile < 1) ncols_tile = 1;

        const block_iq2_k *iq2_row[4];
        for (int r = 0; r < 4; r++) {
            iq2_row[r] = (const block_iq2_k *)
                ((const char *)vx + (ii + r) * w_row_bytes);
        }

        float acc[4][2] = { {{0,0}},{{0,0}},{{0,0}},{{0,0}} };

        const block_q8_K *qk_ptr[2];
        for (int c = 0; c < ncols_tile; c++) {
            qk_ptr[c] = (const block_q8_K *)
                ((const char *)vy + (jj + c) * a_row_bytes);
        }

        for (int ibl = 0; ibl < nb; ibl++) {
            for (int r = 0; r < 4; r++) {
                float d = fp16_to_fp32_lookup(iq2_row[r][ibl].d);
                uint16_t extra = iq2_row[r][ibl].extra;
                const uint8_t *qs = iq2_row[r][ibl].qs;

                for (int c = 0; c < ncols_tile; c++) {
                    float q8_scale = qk_ptr[c][ibl].d;
                    float dy = d * q8_scale;
                    const int8_t *q8 = qk_ptr[c][ibl].qs;

                    int32x4_t isum = vdupq_n_s32(0);
                    uint16_t extra_c = extra;

                    const int8_t *q8_base = q8;
                    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                        int scale_lo = (iq2_row[r][ibl].scales[ib32] & 0xf) - 8;
                        int scale_hi = (iq2_row[r][ibl].scales[ib32] >> 4) - 8;

                        int shift_idx  = ((extra_c >> 0) & 1) * 4;
                        int shift_idx2 = ((extra_c >> 1) & 1) * 4;
                        extra_c >>= 2;

                        int shift = 2 * (ib32 % 4);
                        int qs_off = (ib32 / 4) * 32;

                        int8x16_t y_lo = vld1q_s8(q8_base + ib32 * 32);
                        int8x16_t y_hi = vld1q_s8(q8_base + ib32 * 32 + 16);

                        uint8x16_t lb_lo = vld1q_u8(qs + qs_off + 0);
                        uint8x16_t lb_hi = vld1q_u8(qs + qs_off + 16);

                        uint8x16_t ql_lo, ql_hi;
                        if (shift == 0) {
                            ql_lo = lb_lo; ql_hi = lb_hi;
                        } else if (shift == 2) {
                            ql_lo = vshrq_n_u8(lb_lo, 2); ql_hi = vshrq_n_u8(lb_hi, 2);
                        } else if (shift == 4) {
                            ql_lo = vshrq_n_u8(lb_lo, 4); ql_hi = vshrq_n_u8(lb_hi, 4);
                        } else {
                            ql_lo = vshrq_n_u8(lb_lo, 6); ql_hi = vshrq_n_u8(lb_hi, 6);
                        }
                        ql_lo = vandq_u8(ql_lo, m03);
                        ql_hi = vandq_u8(ql_hi, m03);

                        int8x16_t qx_lo = vreinterpretq_s8_u8(vqtbl1q_u8(lut,
                            vaddq_u8(ql_lo, vdupq_n_u8(shift_idx))));
                        int8x16_t qx_hi = vreinterpretq_s8_u8(vqtbl1q_u8(lut,
                            vaddq_u8(ql_hi, vdupq_n_u8(shift_idx2))));

                        int32x4_t s_lo = vaddq_s32(
                            vpaddlq_s16(vmull_s8(vget_low_s8(qx_lo), vget_low_s8(y_lo))),
                            vpaddlq_s16(vmull_high_s8(qx_lo, y_lo)));
                        int32x4_t s_hi = vaddq_s32(
                            vpaddlq_s16(vmull_s8(vget_low_s8(qx_hi), vget_low_s8(y_hi))),
                            vpaddlq_s16(vmull_high_s8(qx_hi, y_hi)));

                        int32x4_t s_lo_s = vmulq_n_s32(s_lo, scale_lo);
                        int32x4_t s_hi_s = vmulq_n_s32(s_hi, scale_hi);
                        isum = vaddq_s32(isum, vaddq_s32(s_lo_s, s_hi_s));
                    }

                    int total = vaddlvq_s32(isum);
                    acc[r][c] += (float)total * dy;
                }
            }
        }

        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < ncols_tile; c++) {
                out[ii + r + (jj + c) * bs] = acc[r][c];
            }
        }
    }
    return nrows;
}
#endif /* PICOLM_NEON */
