#include "quant.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* ================================================================
 * FP16 <-> FP32 lookup table (mirrors llama.cpp's ggml_table_f32_f16)
 *
 * 64KB table initialized once at startup. Each entry maps a uint16_t
 * FP16 bit pattern to its FP32 value. This is significantly faster
 * than computing the conversion on-the-fly, especially without F16C.
 * ================================================================ */

static float fp16_to_fp32_table[1 << 16];
static int fp16_table_initialized = 0;

void fp16_table_init(void) {
    if (fp16_table_initialized) return;
    for (int i = 0; i < (1 << 16); i++) {
        fp16_to_fp32_table[i] = fp16_to_fp32((uint16_t)i);
    }
    fp16_table_initialized = 1;
}

/* Fast lookup-based FP16->FP32 conversion */
float fp16_to_fp32_lookup(uint16_t h) {
#ifdef PICOLM_NEON
    __fp16 tmp;
    memcpy(&tmp, &h, 2);
    return (float)tmp;
#else
    return fp16_to_fp32_table[h];
#endif
}

/* ================================================================
 * FP16 <-> FP32 conversion (software, no hardware dependency)
 * ================================================================ */

/* fp16-fp32 dot product: sum of fp16_to_fp32_lookup(k[i]) * x[i] for i=0..n-1 */
float vec_dot_f16_f32(const void *src, const float *x, int n) {
    const uint16_t *k = (const uint16_t *)src;
#ifdef PICOLM_AVX512
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 kf = fp16x16_to_fp32_inline(k + i);
        __m512 xf = _mm512_loadu_ps(x + i);
        acc = _mm512_fmadd_ps(kf, xf, acc);
    }
    float sumf = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) sumf += fp16_to_fp32_lookup(k[i]) * x[i];
    return sumf;
#elif defined(PICOLM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        /* vcvt_f32_f16: hardware vector F16->F32 conversion (4 at a time) */
        float16x4_t hf0 = vld1_f16((const float16_t *)k + i);
        float32x4_t kf0 = vcvt_f32_f16(hf0);
        float32x4_t xf0 = vld1q_f32(x + i);
        acc0 = vmlaq_f32(acc0, kf0, xf0);
        float16x4_t hf1 = vld1_f16((const float16_t *)k + i + 4);
        float32x4_t kf1 = vcvt_f32_f16(hf1);
        float32x4_t xf1 = vld1q_f32(x + i + 4);
        acc1 = vmlaq_f32(acc1, kf1, xf1);
    }
    float sumf = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sumf += fp16_to_fp32_lookup(k[i]) * x[i];
    return sumf;
#elif defined(PICOLM_AVX)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 kf = fp16x8_to_fp32_inline(k + i);
        __m256 xf = _mm256_loadu_ps(x + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(kf, xf));
    }
    float sumf = hsum_avx(acc);
    for (; i < n; i++) sumf += fp16_to_fp32_lookup(k[i]) * x[i];
    return sumf;
#elif defined(PICOLM_SSE2)
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __m128 kf = fp16x4_to_fp32_inline(k + i);
        __m128 xf = _mm_loadu_ps(x + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(kf, xf));
    }
    float sumf = hsum_sse(acc);
    for (; i < n; i++) sumf += fp16_to_fp32_lookup(k[i]) * x[i];
    return sumf;
#else
    float sumf = 0.0f;
    for (int i = 0; i < n; i++) sumf += fp16_to_fp32_lookup(k[i]) * x[i];
    return sumf;
#endif
}

float fp16_to_fp32(uint16_t h) {
    /* Mirrors llama.cpp's ggml_compute_fp16_to_fp32 for correct subnormal handling */
    uint32_t w = (uint32_t)h << 16;
    uint32_t sign = w & 0x80000000U;
    uint32_t two_w = w + w;

    uint32_t exp_offset = 0xE0U << 23;
    float exp_scale;
    { uint32_t escale = 0x07800000U; memcpy(&exp_scale, &escale, sizeof(float)); } /* 0x07800000 = 2^-112 */
    float normalized_value;
    { uint32_t nbits = (two_w >> 4) + exp_offset; memcpy(&normalized_value, &nbits, sizeof(float)); }
    normalized_value *= exp_scale;

    uint32_t magic_mask = 126U << 23;
    float magic_bias = 0.5f;
    float denormalized_value;
    { uint32_t dbits = (two_w >> 17) | magic_mask; memcpy(&denormalized_value, &dbits, sizeof(float)); }
    denormalized_value -= magic_bias;

    uint32_t denormalized_cutoff = 1U << 27;
    uint32_t result;
    float rv;
    if (two_w < denormalized_cutoff) {
        memcpy(&result, &denormalized_value, sizeof(float));
    } else {
        memcpy(&result, &normalized_value, sizeof(float));
    }
    result |= sign;
    memcpy(&rv, &result, sizeof(float));
    return rv;
}

/* BF16 -> FP32 conversion */
float bf16_to_fp32(uint16_t x) {
    union { uint32_t u; float f; } o;
    o.u = ((uint32_t)x) << 16;
    return o.f;
}

float vec_dot_bf16_f32(const void *src, const float *x, int n) {
    const uint16_t *bf16 = (const uint16_t *)src;
#ifdef PICOLM_AVX512
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        /* _mm512_cvtepu16_epi32: 16 x uint16 -> 16 x uint32 (zero-extended) */
        __m512i bits = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(bf16 + i)));
        __m512 bf = _mm512_castsi512_ps(_mm512_slli_epi32(bits, 16));
        __m512 xf = _mm512_loadu_ps(x + i);
        acc = _mm512_fmadd_ps(bf, xf, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) sum += bf16_to_fp32(bf16[i]) * x[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += bf16_to_fp32(bf16[i]) * x[i];
    return sum;
#endif
}

void dequantize_row_bf16(const void *src, float *dst, int n) {
    const uint16_t *bf16 = (const uint16_t *)src;
    for (int i = 0; i < n; i++) dst[i] = bf16_to_fp32(bf16[i]);
}

uint16_t fp32_to_fp16(float f) {
#ifdef PICOLM_NEON
    __fp16 tmp = f;
    uint16_t res;
    memcpy(&res, &tmp, 2);
    return res;
#else
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 16) & 0x8000;
    int      exp  = (int)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (((bits >> 23) & 0xFF) == 0) {
        return (uint16_t)sign; /* zero or f32 subnormal -> fp16 zero */
    }
    if (((bits >> 23) & 0xFF) == 0xFF) {
        /* inf / nan */
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00); /* overflow -> inf */
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; /* too small -> zero */
        /* subnormal fp16 */
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        /* round to nearest */
        uint32_t round_bit = 1U << (shift - 1);
        mant = (mant + round_bit) >> shift;
        return (uint16_t)(sign | mant);
    }

    /* round to nearest even */
    mant += 0x00001000; /* bit 12 */
    if (mant & 0x00800000) {
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
#endif
}

/* ---- Q4_K helpers ---- */

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *mn = (uint8_t)((q[j + 4] >>  4) | ((q[j    ] >> 6) << 4));
    }
}

/* ================================================================
 * Dequantization kernels (scalar — used for embedding lookup etc.)
 * ================================================================ */

void dequantize_row_q4_K(const void *src, float *dst, int n) {
    const block_q4_K *blocks = (const block_q4_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q4_K *b = &blocks[i];
        float d    = fp16_to_fp32_lookup(b->d);
        float dmin = fp16_to_fp32_lookup(b->dmin);
        const uint8_t *q = b->qs;
        float *y = dst + i * 256;

        int is = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc, mn;
            get_scale_min_k4(is, b->scales, &sc, &mn);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)mn;
            get_scale_min_k4(is + 1, b->scales, &sc, &mn);
            float d2 = d * (float)sc;
            float m2 = dmin * (float)mn;

            for (int l = 0; l < 32; l++) {
                y[l]      = d1 * (float)(q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; l++) {
                y[l + 32] = d2 * (float)(q[l] >> 4)  - m2;
            }
            y  += 64;
            q  += 32;
            is += 2;
        }
    }
}

void dequantize_row_q3_K(const void *src, float *dst, int n) {
    /* Ported from llama.cpp ggml-quants.c dequantize_row_q3_K */
    const block_q3_K *blocks = (const block_q3_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q3_K *b = &blocks[i];
        float d = fp16_to_fp32_lookup(b->d);

        /* Unpack 16 scales from 12 bytes using llama.cpp's bit-interleaving */
        uint32_t aux[4];
        memcpy(aux, b->scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
        aux[0] = (aux[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4);
        aux[1] = (aux[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4);
        const int8_t *scales = (const int8_t *)aux;

        const uint8_t *q = b->qs;
        const uint8_t *hm = b->hmask;
        uint8_t m = 1;

        float *y = dst + i * 256;
        int is = 0;
        float dl;
        for (int chunk = 0; chunk < 256; chunk += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * (float)((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                }
                dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * (float)((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

void dequantize_row_q2_K(const void *src, float *dst, int n) {
    /* Q2_K: 256 values per block, 84 bytes
     * Dequant: val = d * scale * q - dmin * min
     * q is 2-bit, scale and min are 4-bit each, packed in scales[16]
     * Quant layout: for each 128-value chunk, 4 bit-shifts (0,2,4,6)
     * across 32 qs bytes (2 groups of 16), giving 128 values per chunk.
     * Two chunks per block = 256 values total.
     * Ported from llama.cpp ggml-quants.c dequantize_row_q2_K */
    const block_q2_K *blocks = (const block_q2_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q2_K *b = &blocks[i];
        float d    = fp16_to_fp32_lookup(b->d);
        float dmin = fp16_to_fp32_lookup(b->dmin);

        const uint8_t *q = b->qs;
        float *y = dst + i * 256;

        int is = 0;
        float dl, ml;
        for (int chunk = 0; chunk < 256; chunk += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = b->scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * (int8_t)((q[l] >> shift) & 3) - ml;

                sc = b->scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * (int8_t)((q[l+16] >> shift) & 3) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_row_q6_K(const void *src, float *dst, int n) {
    const block_q6_K *blocks = (const block_q6_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d);
        const uint8_t *ql = blocks[i].ql;
        const uint8_t *qh = blocks[i].qh;
        const int8_t  *sc = blocks[i].scales;
        float *y = dst + i * 256;

        for (int chunk = 0; chunk < 256; chunk += 128) {
            int is = chunk / 16;
            for (int l = 0; l < 32; l++) {
                int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                int is_l = is + (l / 16);
                y[l]      = d * (float)sc[is_l + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is_l + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is_l + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is_l + 6] * (float)q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
        }
    }
}

void dequantize_row_q8_0(const void *src, float *dst, int n) {
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            dst[i * 32 + j] = d * (float)blocks[i].qs[j];
        }
    }
}

void dequantize_row_q4_0(const void *src, float *dst, int n) {
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d);
        const uint8_t *qs = blocks[i].qs;
        float *dp = dst + i * 32;
        /* GGUF Q4_0: qs[j] = {b[j] low nibble, b[j+16] high nibble} */
        for (int j = 0; j < 16; j++) {
            dp[j]      = d * ((float)(qs[j] & 0xF) - 8.0f);
            dp[j + 16] = d * ((float)(qs[j] >> 4) - 8.0f);
        }
    }
}

/* Dequantize Q4_1: val = qs[j] * d + m (unsigned nibble) */
void dequantize_row_q4_1(const void *src, float *dst, int n) {
    const block_q4_1 *blocks = (const block_q4_1 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d);
        float m = fp16_to_fp32_lookup(blocks[i].m);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blocks[i].qs[j];
            dst[i * 32 + j]        = (float)(byte & 0x0F) * d + m;
            dst[i * 32 + j + 16]   = (float)(byte >> 4)   * d + m;
        }
    }
}

/* Dequantize Q1_0: val[j] = (bit[j] ? +d : -d) */
void dequantize_row_q1_0(const void *src, float *dst, int n) {
    const block_q1_0 *blocks = (const block_q1_0 *)src;
    int nb = n / 128;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d);
        float neg_d = -d;
        for (int j = 0; j < 128; j++) {
            int byte_idx = j / 8;
            int bit_off  = j % 8;
            dst[i * 128 + j] = ((blocks[i].qs[byte_idx] >> bit_off) & 1) ? d : neg_d;
        }
    }
}

/* Dequantize Q2_0: val[j] = ((qs[j] & 3) - 1) * d */
void dequantize_row_q2_0(const void *src, float *dst, int n) {
    const block_q2_0 *blocks = (const block_q2_0 *)src;
    int nb = n / 128;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d);
        for (int j = 0; j < 128; j++) {
            int byte_idx = j / 4;
            int bit_off  = (j % 4) * 2;
            int q = (blocks[i].qs[byte_idx] >> bit_off) & 0x03;
            dst[i * 128 + j] = (float)(q - 1) * d;
        }
    }
}

/* Dequantize a single row from Q4_0_4_4 interleaved format to float32.
 * Dequantizes row 0 from a group of 4 interleaved rows.
 * Interleaving: qs[k*16 + r*4 + j] = row_r.qs[k*4 + j] ^ 0x88
 * For row 0: qs[k*16 + j] for k=0..3, j=0..3 */
void dequantize_row_q4_0_4_4(const void *src, float *dst, int n) {
    const block_q4_0x4 *blocks = (const block_q4_0x4 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d[0]);
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 4; j++) {
                uint8_t byte = blocks[i].qs[k * 16 + j];
                int v0 = (int8_t)(byte << 4) >> 4;
                int v1 = (int8_t)(byte & 0xF0) >> 4;
                dst[i * 32 + k * 8 + j * 2] = d * (float)v0;
                dst[i * 32 + k * 8 + j * 2 + 1] = d * (float)v1;
            }
        }
    }
}

/* Dequantize row 0 from Q4_0_4_8 interleaved format (blocklen=8).
 * Same block_q4_0x4 struct but different interleaving: 8 bytes per row per group. */
void dequantize_row_q4_0_4_8(const void *src, float *dst, int n) {
    const block_q4_0x4 *blocks = (const block_q4_0x4 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d[0]);
        for (int k = 0; k < 2; k++) {
            for (int j = 0; j < 8; j++) {
                uint8_t byte = blocks[i].qs[k * 32 + j];
                int v0 = (int8_t)(byte << 4) >> 4;
                int v1 = (int8_t)(byte & 0xF0) >> 4;
                dst[i * 32 + k * 16 + j * 2] = d * (float)v0;
                dst[i * 32 + k * 16 + j * 2 + 1] = d * (float)v1;
            }
        }
    }
}

/* Dequantize a single row from Q4_0_8_8 interleaved format to float32.
 *
 * The GGUF stores 8 consecutive rows interleaved as block_q4_0x8 structures.
 * src MUST point to the start of an 8-row group (i.e., aligned to nb*144 bytes).
 * This function dequantifies row 0 of the group.
 *
 * For arbitrary row extraction, use the inline code in model.c's embedding lookup. */
void dequantize_row_q4_0_8_8(const void *src, float *dst, int n) {
    const block_q4_0x8 *blocks = (const block_q4_0x8 *)src;
    int nb = n / 32;
    int row_in_group = 0;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d[row_in_group]);
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 8; j++) {
                uint8_t byte = blocks[i].qs[k * 128 + row_in_group * 8 + j];
                int v0 = (int8_t)(byte << 4);
                int v1 = (int8_t)(byte & 0xF0);
                dst[i * 32 + k * 8 + j * 2]     = d * (float)(v0 >> 4);
                dst[i * 32 + k * 8 + j * 2 + 1] = d * (float)(v1 >> 4);
            }
        }
    }
}

void dequantize_row_f16(const void *src, float *dst, int n) {
    const uint16_t *fp16 = (const uint16_t *)src;
#ifdef PICOLM_NEON
    int i;
    for (i = 0; i + 3 < n; i += 4) {
        float16x4_t hf = vld1_f16((const float16_t *)fp16 + i);
        float32x4_t xf = vcvt_f32_f16(hf);
        vst1q_f32(dst + i, xf);
    }
    __fp16 tmp;
    for (; i < n; i++) {
        memcpy(&tmp, &fp16[i], 2);
        dst[i] = (float)tmp;
    }
#else
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32_lookup(fp16[i]);
    }
#endif
}

void dequantize_row_f32(const void *src, float *dst, int n) {
#if defined(__APPLE__) && defined(__ppc__) && defined(__ALTIVEC__)
    /* GGUF stores F32 as little-endian; swap on big-endian using Altivec vec_perm */
    static char __f32swap_data[256];
    static int __f32swap_init = 0;
    unsigned long ba = (unsigned long)__f32swap_data + 63;
    ba = ba / 64 * 64;
    char *buf = (char*)ba + 16;
    int i;
    if (!__f32swap_init) {
        ((unsigned char*)(void*)ba)[0] = 3; ((unsigned char*)(void*)ba)[1] = 2;
        ((unsigned char*)(void*)ba)[2] = 1; ((unsigned char*)(void*)ba)[3] = 0;
        ((unsigned char*)(void*)ba)[4] = 7; ((unsigned char*)(void*)ba)[5] = 6;
        ((unsigned char*)(void*)ba)[6] = 5; ((unsigned char*)(void*)ba)[7] = 4;
        ((unsigned char*)(void*)ba)[8] = 11; ((unsigned char*)(void*)ba)[9] = 10;
        ((unsigned char*)(void*)ba)[10] = 9; ((unsigned char*)(void*)ba)[11] = 8;
        ((unsigned char*)(void*)ba)[12] = 15; ((unsigned char*)(void*)ba)[13] = 14;
        ((unsigned char*)(void*)ba)[14] = 13; ((unsigned char*)(void*)ba)[15] = 12;
        __f32swap_init = 1;
    }
    vector unsigned char vmask = (vector unsigned char)vec_ld(0, (unsigned char*)ba);
    for (i = 0; i + 3 < n; i += 4) {
        memcpy(buf, (const char*)src + i * 4, 16);
        vector unsigned char v = (vector unsigned char)vec_ld(0, (unsigned char*)buf);
        vector unsigned char s = vec_perm(v, v, vmask);
        vec_st(s, 0, (unsigned char*)buf);
        memcpy(dst + i, buf, 16);
    }
    for (; i < n; i++) {
        uint32_t val = *(const uint32_t*)((const char*)src + i * 4);
        ((uint32_t*)dst)[i] = (val >> 24) | ((val >> 8) & 0xff00) | ((val << 8) & 0xff0000) | (val << 24);
    }
#elif defined(__APPLE__) && defined(__ppc__)
    /* Scalar fallback for PPC without Altivec */
    const uint32_t *src32 = (const uint32_t *)src;
    uint32_t *dst32 = (uint32_t *)dst;
    for (int i = 0; i < n; i++) {
        dst32[i] = (src32[i] >> 24) | ((src32[i] >> 8) & 0xff00) | ((src32[i] << 8) & 0xff0000) | (src32[i] << 24);
    }
#else
    memcpy(dst, src, (size_t)n * sizeof(float));
#endif
}

/* Q5_K dequantize: 256 elements per block, 5-bit quants with per-subblock scale+min */
void dequantize_row_q5_K(const void *src, float *dst, int n) {
    const block_q5_K *x = (const block_q5_K *)src;
    const int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const float dm = fp16_to_fp32(x[i].dm);
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(j/32 + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = dm * m;
            get_scale_min_k4(j/32 + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = dm * m;
            for (int l = 0; l < 32; ++l) *dst++ = d1 * (float)((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *dst++ = d2 * (float)((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; u1 <<= 2; u2 <<= 2;
        }
    }
}

void dequantize_row(const void *src, float *dst, int n, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   dequantize_row_f32(src, dst, n);  break;
        case GGUF_TYPE_F16:   dequantize_row_f16(src, dst, n);  break;
        case GGUF_TYPE_BF16:  dequantize_row_bf16(src, dst, n); break;
        case GGUF_TYPE_Q4_0:  dequantize_row_q4_0(src, dst, n); break;
        case GGUF_TYPE_Q4_1:  dequantize_row_q4_1(src, dst, n); break;
        case GGUF_TYPE_Q8_0:  dequantize_row_q8_0(src, dst, n); break;
        case GGUF_TYPE_Q2_K:  dequantize_row_q2_K(src, dst, n); break;
        case GGUF_TYPE_Q3_K:  dequantize_row_q3_K(src, dst, n); break;
        case GGUF_TYPE_Q4_K:  dequantize_row_q4_K(src, dst, n); break;
        case GGUF_TYPE_Q5_K:  dequantize_row_q5_K(src, dst, n); break;
        case GGUF_TYPE_Q6_K:  dequantize_row_q6_K(src, dst, n); break;
        case GGUF_TYPE_Q4_0_4_4: dequantize_row_q4_0_4_4(src, dst, n); break;
        case GGUF_TYPE_Q4_0_4_8: dequantize_row_q4_0_4_8(src, dst, n); break;
        case GGUF_TYPE_Q4_0_8_8: dequantize_row_q4_0_8_8(src, dst, n); break;
        case GGUF_TYPE_Q1_0:     dequantize_row_q1_0(src, dst, n); break;
        case GGUF_TYPE_Q2_0:     dequantize_row_q2_0(src, dst, n); break;
        default:
            fprintf(stderr, "dequantize_row: unsupported type %d\n", type);
            exit(1);
    }
}

/* ---- Type info ---- */

int gguf_type_block_size(gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   return 1;
        case GGUF_TYPE_F16:   return 1;
        case GGUF_TYPE_Q4_0:  return 32;
        case GGUF_TYPE_Q4_1:  return 32;
        case GGUF_TYPE_Q5_0:  return 32;
        case GGUF_TYPE_Q5_1:  return 32;
        case GGUF_TYPE_Q8_0:  return 32;
        case GGUF_TYPE_Q8_1:  return 32;
        case GGUF_TYPE_Q2_K:  return 256;
        case GGUF_TYPE_Q3_K:  return 256;
        case GGUF_TYPE_Q4_K:  return 256;
        case GGUF_TYPE_Q5_K:  return 256;
        case GGUF_TYPE_Q6_K:  return 256;
        case GGUF_TYPE_Q8_K:  return 256;
        case GGUF_TYPE_Q4_0_4_4: return 32;  /* each block covers 32 values per row */
        case GGUF_TYPE_Q4_0_4_8: return 32;
        case GGUF_TYPE_Q4_0_8_8: return 32;
        case GGUF_TYPE_BF16:     return 1;  /* BF16: 1 element per block, 2 bytes each */
        case GGUF_TYPE_Q1_0:     return 128;
        case GGUF_TYPE_Q2_0:     return 128;
        default: return 0;
    }
}

int gguf_type_quant_size(gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   return 4;
        case GGUF_TYPE_F16:   return 2;
        case GGUF_TYPE_Q4_0:  return 18;
        case GGUF_TYPE_Q4_1:  return 20;
        case GGUF_TYPE_Q5_0:  return 22;
        case GGUF_TYPE_Q5_1:  return 24;
        case GGUF_TYPE_Q8_0:  return 34;
        case GGUF_TYPE_Q8_1:  return 40;
        case GGUF_TYPE_Q2_K:  return 84;
        case GGUF_TYPE_Q3_K:  return 110;
        case GGUF_TYPE_Q4_K:  return 144;
        case GGUF_TYPE_Q5_K:  return 176;
        case GGUF_TYPE_Q6_K:  return 210;
        case GGUF_TYPE_Q8_K:  return (int)sizeof(block_q8_K);
        case GGUF_TYPE_Q4_0_4_4: return (int)sizeof(block_q4_0);  /* 18: GGUF stores same layout as Q4_0 */
        case GGUF_TYPE_Q4_0_4_8: return (int)sizeof(block_q4_0);  /* 18: same per-row stride as Q4_0 */
        case GGUF_TYPE_Q4_0_8_8: return (int)sizeof(block_q4_0);  /* 18: GGUF stores same layout as Q4_0 */
        case GGUF_TYPE_BF16:     return 2;  /* BF16: 2 bytes per element */
        case GGUF_TYPE_Q1_0:     return 18;
        case GGUF_TYPE_Q2_0:     return 34;
        default: return 0;
    }
}

size_t gguf_type_row_size(gguf_type_t type, int n) {
    int bs = gguf_type_block_size(type);
    int qs = gguf_type_quant_size(type);
    if (bs == 0 || qs == 0) return 0;
    return (size_t)(n / bs) * (size_t)qs;
}

/* ================================================================
 * Fused dequant + dot-product: compute dot(dequant(row), x) without
 * materializing the full dequantized row.
 *
 * Three tiers per format:
 *   1. NEON (ARM Pi 3/4/5)
 *   2. SSE2 (x86 development)
 *   3. Scalar fallback
 * ================================================================ */

/* ---- vec_dot_f32_f32 ---- */

float vec_dot_f32_f32(const void *src, const float *x, int n) {
    const float *w = (const float *)src;

#ifdef PICOLM_NEON
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(w + i),     vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
    }
    float sum = vaddvq_f32_compat(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;

#elif defined(PICOLM_AVX512)
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(w + i), _mm512_loadu_ps(x + i), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;

#elif defined(PICOLM_AVX)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(w + i),     _mm256_loadu_ps(x + i)));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8)));
    }
    /* pick up a trailing group of 8 (common: hidden sizes are multiples of 8 not 16) */
    for (; i + 7 < n; i += 8)
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i)));
    float sum = hsum_avx(_mm256_add_ps(acc0, acc1));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;

#elif defined(PICOLM_SSE2)
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(w + i),     _mm_loadu_ps(x + i)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(w + i + 4), _mm_loadu_ps(x + i + 4)));
    }
    float sum = hsum_sse(_mm_add_ps(acc0, acc1));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;

#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += w[i] * x[i];
    }
    return sum;
#endif
}

/* ---- vec_dot_q4_K_f32 ---- */

float vec_dot_q4_K_f32(const void *src, const float *x, int n) {
    const block_q4_K *blocks = (const block_q4_K *)src;
    int nb = n / 256;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const block_q4_K *b = &blocks[i];
        float d    = fp16_to_fp32_lookup(b->d);
        float dmin = fp16_to_fp32_lookup(b->dmin);
        const uint8_t *q = b->qs;
        const float *xp = x + i * 256;

        int is = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc, mn;
            get_scale_min_k4(is, b->scales, &sc, &mn);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)mn;
            get_scale_min_k4(is + 1, b->scales, &sc, &mn);
            float d2 = d * (float)sc;
            float m2 = dmin * (float)mn;

#ifdef PICOLM_NEON
            float32x4_t sum_qx1_v = vdupq_n_f32(0);
            float32x4_t sum_x1_v  = vdupq_n_f32(0);
            float32x4_t sum_qx2_v = vdupq_n_f32(0);
            float32x4_t sum_x2_v  = vdupq_n_f32(0);

            for (int l = 0; l < 32; l += 8) {
                /* Load 8 quantized bytes, extract nibbles */
                uint8x8_t qbytes = vld1_u8(q + l);
                uint8x8_t q_lo_8 = vand_u8(qbytes, vdup_n_u8(0xF));
                uint8x8_t q_hi_8 = vshr_n_u8(qbytes, 4);

                /* Widen to 16-bit */
                uint16x8_t q_lo_16 = vmovl_u8(q_lo_8);
                uint16x8_t q_hi_16 = vmovl_u8(q_hi_8);

                /* First 4 elements */
                float32x4_t qf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_lo_16)));
                float32x4_t xv0 = vld1q_f32(xp + l);
                sum_qx1_v = vmlaq_f32(sum_qx1_v, qf0, xv0);
                sum_x1_v  = vaddq_f32(sum_x1_v, xv0);

                float32x4_t qf0h = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_hi_16)));
                float32x4_t xv0h = vld1q_f32(xp + l + 32);
                sum_qx2_v = vmlaq_f32(sum_qx2_v, qf0h, xv0h);
                sum_x2_v  = vaddq_f32(sum_x2_v, xv0h);

                /* Next 4 elements */
                float32x4_t qf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q_lo_16)));
                float32x4_t xv1 = vld1q_f32(xp + l + 4);
                sum_qx1_v = vmlaq_f32(sum_qx1_v, qf1, xv1);
                sum_x1_v  = vaddq_f32(sum_x1_v, xv1);

                float32x4_t qf1h = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q_hi_16)));
                float32x4_t xv1h = vld1q_f32(xp + l + 32 + 4);
                sum_qx2_v = vmlaq_f32(sum_qx2_v, qf1h, xv1h);
                sum_x2_v  = vaddq_f32(sum_x2_v, xv1h);
            }

            float sum_qx1 = vaddvq_f32_compat(sum_qx1_v);
            float sum_x1  = vaddvq_f32_compat(sum_x1_v);
            float sum_qx2 = vaddvq_f32_compat(sum_qx2_v);
            float sum_x2  = vaddvq_f32_compat(sum_x2_v);
#elif defined(PICOLM_AVX2)
            /* AVX2: 256-bit integer ops allow zero-extending 8 uint8 nibbles
             * to 8 int32 in one _mm256_cvtepu8_epi32 instruction, then a
             * single _mm256_cvtepi32_ps — no multi-step unpack chain needed. */
            __m256 sum_qx1_v = _mm256_setzero_ps();
            __m256 sum_x1_v  = _mm256_setzero_ps();
            __m256 sum_qx2_v = _mm256_setzero_ps();
            __m256 sum_x2_v  = _mm256_setzero_ps();
            const __m128i mask4 = _mm_set1_epi8(0x0F);

            for (int l = 0; l < 32; l += 8) {
                __m128i qb  = _mm_loadl_epi64((const __m128i *)(q + l));
                __m128i lo8 = _mm_and_si128(qb, mask4);
                __m128i hi8 = _mm_and_si128(_mm_srli_epi16(qb, 4), mask4);

                /* AVX2: zero-extend 8 uint8 → 8 int32 → 8 float in 2 ops */
                __m256 qf_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo8));
                __m256 qf_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi8));

                __m256 xv_lo = _mm256_loadu_ps(xp + l);
                __m256 xv_hi = _mm256_loadu_ps(xp + l + 32);

                sum_qx1_v = _mm256_add_ps(sum_qx1_v, _mm256_mul_ps(qf_lo, xv_lo));
                sum_x1_v  = _mm256_add_ps(sum_x1_v,  xv_lo);
                sum_qx2_v = _mm256_add_ps(sum_qx2_v, _mm256_mul_ps(qf_hi, xv_hi));
                sum_x2_v  = _mm256_add_ps(sum_x2_v,  xv_hi);
            }
            float sum_qx1 = hsum_avx(sum_qx1_v);
            float sum_x1  = hsum_avx(sum_x1_v);
            float sum_qx2 = hsum_avx(sum_qx2_v);
            float sum_x2  = hsum_avx(sum_x2_v);
#elif defined(PICOLM_AVX)
            /* AVX: 128-bit nibble extraction (no AVX2 int needed), 256-bit float accumulators */
            __m256 sum_qx1_v = _mm256_setzero_ps();
            __m256 sum_x1_v  = _mm256_setzero_ps();
            __m256 sum_qx2_v = _mm256_setzero_ps();
            __m256 sum_x2_v  = _mm256_setzero_ps();
            const __m128i mask4  = _mm_set1_epi8(0x0F);
            const __m128i zero_i = _mm_setzero_si128();

            for (int l = 0; l < 32; l += 8) {
                __m128i qb  = _mm_loadl_epi64((const __m128i *)(q + l));
                __m128i lo8 = _mm_and_si128(qb, mask4);
                __m128i hi8 = _mm_and_si128(_mm_srli_epi16(qb, 4), mask4);

                __m128i lo16 = _mm_unpacklo_epi8(lo8, zero_i);
                __m128i hi16 = _mm_unpacklo_epi8(hi8, zero_i);

                /* Combine two __m128 → one __m256 of 8 floats */
                __m256 qf_lo = _mm256_set_m128(
                    _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo16, zero_i)),
                    _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo16, zero_i)));
                __m256 qf_hi = _mm256_set_m128(
                    _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi16, zero_i)),
                    _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi16, zero_i)));

                __m256 xv_lo = _mm256_loadu_ps(xp + l);
                __m256 xv_hi = _mm256_loadu_ps(xp + l + 32);

                sum_qx1_v = _mm256_add_ps(sum_qx1_v, _mm256_mul_ps(qf_lo, xv_lo));
                sum_x1_v  = _mm256_add_ps(sum_x1_v,  xv_lo);
                sum_qx2_v = _mm256_add_ps(sum_qx2_v, _mm256_mul_ps(qf_hi, xv_hi));
                sum_x2_v  = _mm256_add_ps(sum_x2_v,  xv_hi);
            }
            float sum_qx1 = hsum_avx(sum_qx1_v);
            float sum_x1  = hsum_avx(sum_x1_v);
            float sum_qx2 = hsum_avx(sum_qx2_v);
            float sum_x2  = hsum_avx(sum_x2_v);
#elif defined(PICOLM_SSE2)
            /* SSE2: lo nibble → group1 (xp+l), hi nibble → group2 (xp+l+32) */
            __m128 sum_qx1_v = _mm_setzero_ps();
            __m128 sum_x1_v  = _mm_setzero_ps();
            __m128 sum_qx2_v = _mm_setzero_ps();
            __m128 sum_x2_v  = _mm_setzero_ps();
            const __m128i mask4 = _mm_set1_epi8(0x0F);
            const __m128i zero_i = _mm_setzero_si128();

            for (int l = 0; l < 32; l += 8) {
                /* Load 8 quantized bytes -> 8 lo + 8 hi nibbles */
                __m128i qb  = _mm_loadl_epi64((const __m128i *)(q + l));
                __m128i lo8 = _mm_and_si128(qb, mask4);
                __m128i hi8 = _mm_and_si128(_mm_srli_epi16(qb, 4), mask4);

                /* Widen uint8 nibbles -> int32 -> float (8 values each) */
                __m128i lo16 = _mm_unpacklo_epi8(lo8, zero_i);
                __m128i hi16 = _mm_unpacklo_epi8(hi8, zero_i);
                __m128 qf_lo0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo16, zero_i));
                __m128 qf_lo1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo16, zero_i));
                __m128 qf_hi0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi16, zero_i));
                __m128 qf_hi1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi16, zero_i));

                __m128 xv_lo0 = _mm_loadu_ps(xp + l);
                __m128 xv_lo1 = _mm_loadu_ps(xp + l + 4);
                __m128 xv_hi0 = _mm_loadu_ps(xp + l + 32);
                __m128 xv_hi1 = _mm_loadu_ps(xp + l + 36);

                sum_qx1_v = _mm_add_ps(sum_qx1_v,
                    _mm_add_ps(_mm_mul_ps(qf_lo0, xv_lo0), _mm_mul_ps(qf_lo1, xv_lo1)));
                sum_x1_v  = _mm_add_ps(sum_x1_v,  _mm_add_ps(xv_lo0, xv_lo1));
                sum_qx2_v = _mm_add_ps(sum_qx2_v,
                    _mm_add_ps(_mm_mul_ps(qf_hi0, xv_hi0), _mm_mul_ps(qf_hi1, xv_hi1)));
                sum_x2_v  = _mm_add_ps(sum_x2_v,  _mm_add_ps(xv_hi0, xv_hi1));
            }
            float sum_qx1 = hsum_sse(sum_qx1_v);
            float sum_x1  = hsum_sse(sum_x1_v);
            float sum_qx2 = hsum_sse(sum_qx2_v);
            float sum_x2  = hsum_sse(sum_x2_v);
#else
            float sum_qx1 = 0.0f, sum_x1 = 0.0f;
            float sum_qx2 = 0.0f, sum_x2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                float x_lo = xp[l];
                float x_hi = xp[l + 32];
                sum_qx1 += (float)(q[l] & 0xF) * x_lo;
                sum_x1  += x_lo;
                sum_qx2 += (float)(q[l] >> 4) * x_hi;
                sum_x2  += x_hi;
            }
#endif
            sumf += d1 * sum_qx1 - m1 * sum_x1 + d2 * sum_qx2 - m2 * sum_x2;

            xp += 64;
            q  += 32;
            is += 2;
        }
    }
    return sumf;
}

/* ---- vec_dot_q6_K_f32 ---- */

float vec_dot_q6_K_f32(const void *src, const float *x, int n) {
    const block_q6_K *blocks = (const block_q6_K *)src;
    int nb = n / 256;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d);
        const uint8_t *ql = blocks[i].ql;
        const uint8_t *qh = blocks[i].qh;
        const int8_t  *sc = blocks[i].scales;
        const float *xp = x + i * 256;

        float sums[16] = {0};

/* sign-extend packed int8 → two __m128 floats; used by AVX and SSE2 paths.
 * Idiom: unpacklo_epi8(zero, x) places each byte in the HIGH byte of a 16-bit
 * lane; srai_epi16(..., 8) then arithmetic-shifts it down, propagating the sign
 * bit — equivalent to a sign-extending byte→int16 widening without SSE4.1. */
#if defined(PICOLM_AVX) || defined(PICOLM_SSE2)
#define Q6K_CONV(qi8, fa, fb) do { \
    __m128i w16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, qi8), 8); \
    fa = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(zero_i, w16), 16)); \
    fb = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(zero_i, w16), 16)); \
} while (0)
#endif

#ifdef PICOLM_AVX2
        /* AVX2: _mm256_cvtepi8_epi32 replaces the 4-op Q6K_CONV sign-extension chain */
        const __m128i mask4  = _mm_set1_epi8(0x0F);
        const __m128i mask3  = _mm_set1_epi8(0x03);
        const __m128i sub32  = _mm_set1_epi8(32);

        for (int chunk = 0; chunk < 2; chunk++) {
            int is = chunk * 8;
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const float   *xp_c = xp + chunk * 128;

            for (int half = 0; half < 2; half++) {
                int l0   = half * 16;
                int sidx = is + half;
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                __m256 acc4 = _mm256_setzero_ps();

                for (int l = l0; l < l0 + 16; l += 8) {
                    __m128i qla = _mm_loadl_epi64((const __m128i *)(ql_c + l));
                    __m128i qlb = _mm_loadl_epi64((const __m128i *)(ql_c + l + 32));
                    __m128i qhv = _mm_loadl_epi64((const __m128i *)(qh_c + l));

                    __m128i lo_a = _mm_and_si128(qla, mask4);
                    __m128i hi_a = _mm_and_si128(_mm_srli_epi16(qla, 4), mask4);
                    __m128i lo_b = _mm_and_si128(qlb, mask4);
                    __m128i hi_b = _mm_and_si128(_mm_srli_epi16(qlb, 4), mask4);

                    __m128i h01 = _mm_and_si128(qhv, mask3);
                    __m128i h23 = _mm_and_si128(_mm_srli_epi16(qhv, 2), mask3);
                    __m128i h45 = _mm_and_si128(_mm_srli_epi16(qhv, 4), mask3);
                    __m128i h67 = _mm_and_si128(_mm_srli_epi16(qhv, 6), mask3);

                    __m128i q1_i8 = _mm_sub_epi8(_mm_or_si128(lo_a, _mm_slli_epi16(h01, 4)), sub32);
                    __m128i q2_i8 = _mm_sub_epi8(_mm_or_si128(lo_b, _mm_slli_epi16(h23, 4)), sub32);
                    __m128i q3_i8 = _mm_sub_epi8(_mm_or_si128(hi_a, _mm_slli_epi16(h45, 4)), sub32);
                    __m128i q4_i8 = _mm_sub_epi8(_mm_or_si128(hi_b, _mm_slli_epi16(h67, 4)), sub32);

                    /* AVX2: sign-extend 8 int8 → 8 int32 → 8 float in 2 ops */
                    __m256 qf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1_i8));
                    __m256 qf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2_i8));
                    __m256 qf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3_i8));
                    __m256 qf4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q4_i8));

                    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(qf1, _mm256_loadu_ps(xp_c + l)));
                    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(qf2, _mm256_loadu_ps(xp_c + l + 32)));
                    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(qf3, _mm256_loadu_ps(xp_c + l + 64)));
                    acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(qf4, _mm256_loadu_ps(xp_c + l + 96)));
                }
                sums[sidx + 0] += hsum_avx(acc1);
                sums[sidx + 2] += hsum_avx(acc2);
                sums[sidx + 4] += hsum_avx(acc3);
                sums[sidx + 6] += hsum_avx(acc4);
            }
        }
#elif defined(PICOLM_AVX)
        /* AVX: 128-bit integer extraction, 256-bit float accumulators */
        const __m128i mask4  = _mm_set1_epi8(0x0F);
        const __m128i mask3  = _mm_set1_epi8(0x03);
        const __m128i sub32  = _mm_set1_epi8(32);
        const __m128i zero_i = _mm_setzero_si128();

        for (int chunk = 0; chunk < 2; chunk++) {
            int is = chunk * 8;
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const float   *xp_c = xp + chunk * 128;

            for (int half = 0; half < 2; half++) {
                int l0   = half * 16;
                int sidx = is + half;
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                __m256 acc4 = _mm256_setzero_ps();

                for (int l = l0; l < l0 + 16; l += 8) {
                    __m128i qla = _mm_loadl_epi64((const __m128i *)(ql_c + l));
                    __m128i qlb = _mm_loadl_epi64((const __m128i *)(ql_c + l + 32));
                    __m128i qhv = _mm_loadl_epi64((const __m128i *)(qh_c + l));

                    __m128i lo_a = _mm_and_si128(qla, mask4);
                    __m128i hi_a = _mm_and_si128(_mm_srli_epi16(qla, 4), mask4);
                    __m128i lo_b = _mm_and_si128(qlb, mask4);
                    __m128i hi_b = _mm_and_si128(_mm_srli_epi16(qlb, 4), mask4);

                    __m128i h01 = _mm_and_si128(qhv, mask3);
                    __m128i h23 = _mm_and_si128(_mm_srli_epi16(qhv, 2), mask3);
                    __m128i h45 = _mm_and_si128(_mm_srli_epi16(qhv, 4), mask3);
                    __m128i h67 = _mm_and_si128(_mm_srli_epi16(qhv, 6), mask3);

                    __m128i q1_i8 = _mm_sub_epi8(_mm_or_si128(lo_a, _mm_slli_epi16(h01, 4)), sub32);
                    __m128i q2_i8 = _mm_sub_epi8(_mm_or_si128(lo_b, _mm_slli_epi16(h23, 4)), sub32);
                    __m128i q3_i8 = _mm_sub_epi8(_mm_or_si128(hi_a, _mm_slli_epi16(h45, 4)), sub32);
                    __m128i q4_i8 = _mm_sub_epi8(_mm_or_si128(hi_b, _mm_slli_epi16(h67, 4)), sub32);

                    __m128 qf1a, qf1b, qf2a, qf2b, qf3a, qf3b, qf4a, qf4b;
                    Q6K_CONV(q1_i8, qf1a, qf1b);
                    Q6K_CONV(q2_i8, qf2a, qf2b);
                    Q6K_CONV(q3_i8, qf3a, qf3b);
                    Q6K_CONV(q4_i8, qf4a, qf4b);

                    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_set_m128(qf1b, qf1a), _mm256_loadu_ps(xp_c + l)));
                    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_set_m128(qf2b, qf2a), _mm256_loadu_ps(xp_c + l + 32)));
                    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_set_m128(qf3b, qf3a), _mm256_loadu_ps(xp_c + l + 64)));
                    acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(_mm256_set_m128(qf4b, qf4a), _mm256_loadu_ps(xp_c + l + 96)));
                }
                sums[sidx + 0] += hsum_avx(acc1);
                sums[sidx + 2] += hsum_avx(acc2);
                sums[sidx + 4] += hsum_avx(acc3);
                sums[sidx + 6] += hsum_avx(acc4);
            }
        }
#elif defined(PICOLM_SSE2)
        /* SSE2: 6-bit values = lo4(ql) | hi2(qh)<<4, biased by 32 */
        const __m128i mask4  = _mm_set1_epi8(0x0F);
        const __m128i mask3  = _mm_set1_epi8(0x03);
        const __m128i sub32  = _mm_set1_epi8(32);
        const __m128i zero_i = _mm_setzero_si128();

        for (int chunk = 0; chunk < 2; chunk++) {
            int is = chunk * 8;
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const float   *xp_c = xp + chunk * 128;

            for (int half = 0; half < 2; half++) { /* half=0 -> sums[+0,2,4,6], half=1 -> [+1,3,5,7] */
                int l0   = half * 16;
                int sidx = is + half;
                __m128 acc1a = _mm_setzero_ps(), acc1b = _mm_setzero_ps();
                __m128 acc2a = _mm_setzero_ps(), acc2b = _mm_setzero_ps();
                __m128 acc3a = _mm_setzero_ps(), acc3b = _mm_setzero_ps();
                __m128 acc4a = _mm_setzero_ps(), acc4b = _mm_setzero_ps();

                for (int l = l0; l < l0 + 16; l += 8) {
                    __m128i qla = _mm_loadl_epi64((const __m128i *)(ql_c + l));
                    __m128i qlb = _mm_loadl_epi64((const __m128i *)(ql_c + l + 32));
                    __m128i qhv = _mm_loadl_epi64((const __m128i *)(qh_c + l));

                    __m128i lo_a = _mm_and_si128(qla, mask4);
                    __m128i hi_a = _mm_and_si128(_mm_srli_epi16(qla, 4), mask4);
                    __m128i lo_b = _mm_and_si128(qlb, mask4);
                    __m128i hi_b = _mm_and_si128(_mm_srli_epi16(qlb, 4), mask4);

                    /* epi16 shifts on qh: avoids byte-lane bleed from epi8 shifts */
                    __m128i h01 = _mm_and_si128(qhv, mask3);
                    __m128i h23 = _mm_and_si128(_mm_srli_epi16(qhv, 2), mask3);
                    __m128i h45 = _mm_and_si128(_mm_srli_epi16(qhv, 4), mask3);
                    __m128i h67 = _mm_and_si128(_mm_srli_epi16(qhv, 6), mask3);

                    __m128i q1_i8 = _mm_sub_epi8(_mm_or_si128(lo_a, _mm_slli_epi16(h01, 4)), sub32);
                    __m128i q2_i8 = _mm_sub_epi8(_mm_or_si128(lo_b, _mm_slli_epi16(h23, 4)), sub32);
                    __m128i q3_i8 = _mm_sub_epi8(_mm_or_si128(hi_a, _mm_slli_epi16(h45, 4)), sub32);
                    __m128i q4_i8 = _mm_sub_epi8(_mm_or_si128(hi_b, _mm_slli_epi16(h67, 4)), sub32);

                    __m128 qf1a, qf1b, qf2a, qf2b, qf3a, qf3b, qf4a, qf4b;
                    Q6K_CONV(q1_i8, qf1a, qf1b);
                    Q6K_CONV(q2_i8, qf2a, qf2b);
                    Q6K_CONV(q3_i8, qf3a, qf3b);
                    Q6K_CONV(q4_i8, qf4a, qf4b);

                    acc1a = _mm_add_ps(acc1a, _mm_mul_ps(qf1a, _mm_loadu_ps(xp_c + l)));
                    acc1b = _mm_add_ps(acc1b, _mm_mul_ps(qf1b, _mm_loadu_ps(xp_c + l + 4)));
                    acc2a = _mm_add_ps(acc2a, _mm_mul_ps(qf2a, _mm_loadu_ps(xp_c + l + 32)));
                    acc2b = _mm_add_ps(acc2b, _mm_mul_ps(qf2b, _mm_loadu_ps(xp_c + l + 36)));
                    acc3a = _mm_add_ps(acc3a, _mm_mul_ps(qf3a, _mm_loadu_ps(xp_c + l + 64)));
                    acc3b = _mm_add_ps(acc3b, _mm_mul_ps(qf3b, _mm_loadu_ps(xp_c + l + 68)));
                    acc4a = _mm_add_ps(acc4a, _mm_mul_ps(qf4a, _mm_loadu_ps(xp_c + l + 96)));
                    acc4b = _mm_add_ps(acc4b, _mm_mul_ps(qf4b, _mm_loadu_ps(xp_c + l + 100)));
                }
                sums[sidx + 0] += hsum_sse(_mm_add_ps(acc1a, acc1b));
                sums[sidx + 2] += hsum_sse(_mm_add_ps(acc2a, acc2b));
                sums[sidx + 4] += hsum_sse(_mm_add_ps(acc3a, acc3b));
                sums[sidx + 6] += hsum_sse(_mm_add_ps(acc4a, acc4b));
            }
        }
#else
        for (int chunk = 0; chunk < 2; chunk++) {
            int is = chunk * 8;
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const float *xp_c = xp + chunk * 128;

            for (int l = 0; l < 16; l++) {
                int q1 = (int)((ql_c[l]      & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_c[l]      >> 4)  | (((qh_c[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_c[l + 32] >> 4)  | (((qh_c[l] >> 6) & 3) << 4)) - 32;
                sums[is + 0] += (float)q1 * xp_c[l];
                sums[is + 2] += (float)q2 * xp_c[l + 32];
                sums[is + 4] += (float)q3 * xp_c[l + 64];
                sums[is + 6] += (float)q4 * xp_c[l + 96];
            }
            for (int l = 16; l < 32; l++) {
                int q1 = (int)((ql_c[l]      & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_c[l]      >> 4)  | (((qh_c[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_c[l + 32] >> 4)  | (((qh_c[l] >> 6) & 3) << 4)) - 32;
                sums[is + 1] += (float)q1 * xp_c[l];
                sums[is + 3] += (float)q2 * xp_c[l + 32];
                sums[is + 5] += (float)q3 * xp_c[l + 64];
                sums[is + 7] += (float)q4 * xp_c[l + 96];
            }
        }
#endif

#undef Q6K_CONV

        for (int j = 0; j < 16; j++) {
            sumf += d * (float)sc[j] * sums[j];
        }
    }
    return sumf;
}

/* ================================================================
 * vec_dot_q6_K_q8_K: int8 MAC for Q6_K weights * Q8_K input
 *
 * Q6_K stores 6-bit values per weight, biased by 32: stored [0..63], actual [-32..31].
 * 256 weights per block, 16 per-subblock scales, 16 bsums precomputed for Q8_K input.
 *
 * The key optimization from llama.cpp:
 *   sum((q6_stored[j] - 32) * q8[j] * scale[sub])
 * = sum(q6_stored[j] * q8[j] * scale[sub]) - 32 * bsums[sub] * scale[sub]
 *
 * We do unsigned int8 MAC with stored [0..63] values, then subtract
 * the bias correction: 32 * bsums[j] * scales[j], precomputed per block.
 *
 * AVX-512 VNNI: _mm256_dpbusd_epi32 for 32-wide dot per sub-block
 * AVX2:         _mm256_maddubs_epi16 for 32-wide multiply-accumulate
 * AVX:          _mm_maddubs_epi16 for 16-wide (two lanes)
 * ================================================================ */

#if defined(PICOLM_AVX2) || defined(PICOLM_AVX)
/* Scale shuffle for Q6_K: 16 scales, each repeated 8 times per 32-byte lane.
 * Q6_K has one scale per 16-element sub-block. For AVX2 32-wide ops, each
 * scale needs to be duplicated across the lanes it applies to.
 * Each scale index is repeated 8 times (covering 8 values per 32-lane register).
 * Two consecutive scales cover 16 values each = 32 values per chunk of the shuffle.
 * Used by both AVX2 (32-wide) and AVX (16-wide) code paths. */
static const uint8_t get_scale_shuffle_k6[144] = {
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
     6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
     8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
    10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,
    12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,
    14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,
};
#endif

float vec_dot_q6_K_q8_K(const void *src_q6, const void *src_q8, int n) {
    const block_q6_K *x = (const block_q6_K *)src_q6;
    const block_q8_K *y = (const block_q8_K *)src_q8;
    int nb = n / 256;

/* AVX2: 256-bit integer SIMD for 2x wider dequantization vs AVX 128-bit.
 * AVX-512 (with AVX-512VL) also uses this path since _mm256_maddubs_epi16
 * is available. The scale shuffle uses llama.cpp's proven approach. */
#if defined(PICOLM_AVX2)
    const __m256i m3  = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);

        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *q8 = y[i].qs;

        /* Bias correction: 32 * bsums[j] * scales[j], computed as 256-bit */
        const __m256i q8sums   = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i scales   = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < 2; j++) {  /* 2 chunks of 128 values per block */
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)ql); ql += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)ql); ql += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*)(get_scale_shuffle_k6 + 16*(is+0))));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*)(get_scale_shuffle_k6 + 16*(is+1))));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*)(get_scale_shuffle_k6 + 16*(is+2))));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*)(get_scale_shuffle_k6 + 16*(is+3))));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }
    float result = hsum_avx(acc);
    return result;

#elif defined(PICOLM_AVX)
    /* AVX-only: 128-bit integer, 256-bit float accumulation (proven correct) */
    const __m128i m3_128  = _mm_set1_epi8(3);
    const __m128i m15_128 = _mm_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);

        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *q8 = y[i].qs;

        /* Bias correction using bsums */
        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)y[i].bsums);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)y[i].bsums + 1);
        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales_16_0 = _mm_cvtepi8_epi16(scales);
        const __m128i scales_16_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scales, 8));
        const __m128i q8sclsub_0 = _mm_slli_epi32(_mm_madd_epi16(q8sums_0, scales_16_0), 5);
        const __m128i q8sclsub_1 = _mm_slli_epi32(_mm_madd_epi16(q8sums_1, scales_16_1), 5);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();
        int is = 0;

        for (int j = 0; j < 2; j++) {  /* 2 chunks of 128 */
            const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i*)qh); qh += 16;
            const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i*)qh); qh += 16;

            const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3_128), 4);
            const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3_128), 4);
            const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(12)), 2);
            const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(12)), 2);
            const __m128i q4h_4 = _mm_and_si128(q4bitsH_0, _mm_set1_epi8(48));
            const __m128i q4h_5 = _mm_and_si128(q4bitsH_1, _mm_set1_epi8(48));
            const __m128i q4h_6 = _mm_srli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(-64)), 2);
            const __m128i q4h_7 = _mm_srli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(-64)), 2);

            const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i*)ql); ql += 16;
            const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i*)ql); ql += 16;
            const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i*)ql); ql += 16;
            const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i*)ql); ql += 16;

            const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m15_128), q4h_0);
            const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m15_128), q4h_1);
            const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m15_128), q4h_2);
            const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m15_128), q4h_3);
            const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m15_128), q4h_4);
            const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m15_128), q4h_5);
            const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m15_128), q4h_6);
            const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m15_128), q4h_7);

            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*)(get_scale_shuffle_k6 + 16*(is+0))));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*)(get_scale_shuffle_k6 + 16*(is+1))));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*)(get_scale_shuffle_k6 + 16*(is+2))));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*)(get_scale_shuffle_k6 + 16*(is+3))));
            is += 4;

            p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_0, 8)), p16_1);
            p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
            p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_1, 8)), p16_3);
            p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
            p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_2, 8)), p16_5);
            p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
            p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_3, 8)), p16_7);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));
        }

        sumi_0 = _mm_sub_epi32(sumi_0, q8sclsub_0);
        sumi_1 = _mm_sub_epi32(sumi_1, q8sclsub_1);
        const __m256i sumi = _mm256_insertf128_si256(_mm256_castsi128_si256(sumi_0), sumi_1, 1);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc);
    }
    float result = hsum_avx(acc);
    return result;

#elif defined(PICOLM_NEON)
    {
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *q8 = y[i].qs;
        const int8_t  *sc = x[i].scales;

        int block_sum = 0;
        {
            int32_t sums[16] = {0};
            for (int chunk = 0; chunk < 2; chunk++) {
                int is = chunk * 8;
                const uint8_t *ql_c = ql + chunk * 64;
                const uint8_t *qh_c = qh + chunk * 32;
                const int8_t  *q8_c = q8 + chunk * 128;

                /* Process all 32 qh bytes, 16 at a time */
                for (int half = 0; half < 2; half++) {
                    const uint8_t *ql_h = ql_c + half * 16;
                    const uint8_t *ql_h2 = ql_c + 32 + half * 16;
                    const uint8_t *qh_h = qh_c + half * 16;

                    const uint8x16_t ql_lo = vld1q_u8(ql_h);
                    const uint8x16_t ql_lo2 = vld1q_u8(ql_h2);
                    const uint8x16_t ql_hi = vshrq_n_u8(ql_lo, 4);
                    const uint8x16_t ql_hi2 = vshrq_n_u8(ql_lo2, 4);
                    const uint8x16_t ql_lo_masked = vandq_u8(ql_lo, vdupq_n_u8(0x0F));
                    const uint8x16_t ql_lo2_masked = vandq_u8(ql_lo2, vdupq_n_u8(0x0F));

                    /* 4 sub-blocks from each 16-byte qh slice */
                    const uint8x16_t qh_hv = vld1q_u8(qh_h);
                    const uint8x16_t qh0 = vandq_u8(qh_hv, vdupq_n_u8(0x03));
                    const uint8x16_t qh1 = vandq_u8(vshrq_n_u8(qh_hv, 2), vdupq_n_u8(0x03));
                    const uint8x16_t qh2 = vandq_u8(vshrq_n_u8(qh_hv, 4), vdupq_n_u8(0x03));
                    const uint8x16_t qh3 = vandq_u8(vshrq_n_u8(qh_hv, 6), vdupq_n_u8(0x03));

                    /* Combine: 4 low + 2 high bits = 6-bit value */
                    const uint8x16_t q6_0 = vorrq_u8(ql_lo_masked, vshlq_n_u8(qh0, 4));
                    const uint8x16_t q6_1 = vorrq_u8(ql_lo2_masked, vshlq_n_u8(qh1, 4));
                    const uint8x16_t q6_2 = vorrq_u8(ql_hi, vshlq_n_u8(qh2, 4));
                    const uint8x16_t q6_3 = vorrq_u8(ql_hi2, vshlq_n_u8(qh3, 4));

                    /* Multiply-accumulate: half=0 -> even sb, half=1 -> odd sb */
                    int sb_off = half;  /* 0 for even, 1 for odd */
                    /* Sub-block is+(sb_off+0): q6_0 * q8_c[half*16 .. half*16+15] */
                    { const int8x16_t q8v = vld1q_s8(q8_c + half * 16);
                      int16x8_t p0 = vmull_s8(vget_low_s8(vreinterpretq_s8_u8(q6_0)), vget_low_s8(q8v));
                      int16x8_t p1 = vmull_s8(vget_high_s8(vreinterpretq_s8_u8(q6_0)), vget_high_s8(q8v));
                      sums[is + sb_off + 0] += vaddvq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))); }
                    /* Sub-block is+(sb_off+2): q6_1 * q8_c[32+half*16 .. ] */
                    { const int8x16_t q8v = vld1q_s8(q8_c + 32 + half * 16);
                      int16x8_t p0 = vmull_s8(vget_low_s8(vreinterpretq_s8_u8(q6_1)), vget_low_s8(q8v));
                      int16x8_t p1 = vmull_s8(vget_high_s8(vreinterpretq_s8_u8(q6_1)), vget_high_s8(q8v));
                      sums[is + sb_off + 2] += vaddvq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))); }
                    /* Sub-block is+(sb_off+4): q6_2 * q8_c[64+half*16 .. ] */
                    { const int8x16_t q8v = vld1q_s8(q8_c + 64 + half * 16);
                      int16x8_t p0 = vmull_s8(vget_low_s8(vreinterpretq_s8_u8(q6_2)), vget_low_s8(q8v));
                      int16x8_t p1 = vmull_s8(vget_high_s8(vreinterpretq_s8_u8(q6_2)), vget_high_s8(q8v));
                      sums[is + sb_off + 4] += vaddvq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))); }
                    /* Sub-block is+(sb_off+6): q6_3 * q8_c[96+half*16 .. ] */
                    { const int8x16_t q8v = vld1q_s8(q8_c + 96 + half * 16);
                      int16x8_t p0 = vmull_s8(vget_low_s8(vreinterpretq_s8_u8(q6_3)), vget_low_s8(q8v));
                      int16x8_t p1 = vmull_s8(vget_high_s8(vreinterpretq_s8_u8(q6_3)), vget_high_s8(q8v));
                      sums[is + sb_off + 6] += vaddvq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))); }
                }
            }
            for (int j = 0; j < 16; j++) {
                block_sum += sc[j] * sums[j] - 32 * y[i].bsums[j] * sc[j];
            }
        }
        sumf += d * (float)block_sum;
    }
    return sumf;
}

#else
    /* Scalar fallback */
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *q8 = y[i].qs;
        const int8_t  *sc = x[i].scales;

        int block_sum = 0;
        {
            int sums[16] = {0};
            for (int chunk = 0; chunk < 2; chunk++) {
                int is = chunk * 8;
                const uint8_t *ql_c = ql + chunk * 64;
                const uint8_t *qh_c = qh + chunk * 32;
                const int8_t  *q8_c = q8 + chunk * 128;

                for (int l = 0; l < 16; l++) {
                    int q1 = (ql_c[l]      & 0xF) | (((qh_c[l] >> 0) & 3) << 4);
                    int q2 = (ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4);
                    int q3 = (ql_c[l]      >> 4)  | (((qh_c[l] >> 4) & 3) << 4);
                    int q4 = (ql_c[l + 32] >> 4)  | (((qh_c[l] >> 6) & 3) << 4);
                    sums[is + 0] += q1 * q8_c[l];
                    sums[is + 2] += q2 * q8_c[l + 32];
                    sums[is + 4] += q3 * q8_c[l + 64];
                    sums[is + 6] += q4 * q8_c[l + 96];
                }
                for (int l = 16; l < 32; l++) {
                    int q1 = (ql_c[l]      & 0xF) | (((qh_c[l] >> 0) & 3) << 4);
                    int q2 = (ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4);
                    int q3 = (ql_c[l]      >> 4)  | (((qh_c[l] >> 4) & 3) << 4);
                    int q4 = (ql_c[l + 32] >> 4)  | (((qh_c[l] >> 6) & 3) << 4);
                    sums[is + 1] += q1 * q8_c[l];
                    sums[is + 3] += q2 * q8_c[l + 32];
                    sums[is + 5] += q3 * q8_c[l + 64];
                    sums[is + 7] += q4 * q8_c[l + 96];
                }
            }
            for (int j = 0; j < 16; j++) {
                block_sum += sc[j] * sums[j] - 32 * y[i].bsums[j] * sc[j];
            }
        }
        sumf += d * (float)block_sum;
    }
    return sumf;
#endif
}

/* ================================================================
 * quantize_row_q8_0: quantize float32 -> Q8_0 blocks
 *
 * Adapted from llama.cpp's AVX quantize_row_q8_0.
 * ================================================================ */

void quantize_row_q8_0(const float *x, void *dst, int n) {
    block_q8_0 *y = (block_q8_0 *)dst;
    int nb = n / 32;

#ifdef PICOLM_AVX512
    { int i = 0;
    /* AVX-512: cvtps_epi32 -> cvtepi32_epi8 (16 i32 -> 16 i8 each).
     * Two calls per 32-element block, stored low+high. */
    for (; i + 1 < nb; i += 2) {
        __m512 v0 = _mm512_loadu_ps(x);
        __m512 v1 = _mm512_loadu_ps(x + 16);
        __m512 v2 = _mm512_loadu_ps(x + 32);
        __m512 v3 = _mm512_loadu_ps(x + 48);
        x += 64;

        __m512 maxAbs = _mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1));
        float maxS = _mm512_reduce_max_ps(maxAbs);
        y[i].d = fp32_to_fp16(maxS / 127.0f);
        float id = (maxS != 0.0f) ? 127.0f / maxS : 0.0f;
        __m512 mul = _mm512_set1_ps(id);
        __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, mul));
        __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, mul));
        __m128i q0l = _mm512_cvtepi32_epi8(i0);  /* i32[0..15] -> i8[0..15] */
        __m128i q0h = _mm512_cvtepi32_epi8(i1);  /* i32[16..31] -> i8[16..31] */
        _mm_storeu_si128((__m128i *)(y[i].qs), q0l);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), q0h);

        maxAbs = _mm512_max_ps(_mm512_abs_ps(v2), _mm512_abs_ps(v3));
        float maxS2 = _mm512_reduce_max_ps(maxAbs);
        y[i+1].d = fp32_to_fp16(maxS2 / 127.0f);
        float id2 = (maxS2 != 0.0f) ? 127.0f / maxS2 : 0.0f;
        __m512 mul2 = _mm512_set1_ps(id2);
        __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(v2, mul2));
        __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(v3, mul2));
        __m128i q1l = _mm512_cvtepi32_epi8(i2);
        __m128i q1h = _mm512_cvtepi32_epi8(i3);
        _mm_storeu_si128((__m128i *)(y[i+1].qs), q1l);
        _mm_storeu_si128((__m128i *)(y[i+1].qs + 16), q1h);
    }
    for (; i < nb; i++) {
        float asmax = 0.0f;
        for (int j = 0; j < 32; j++) { float v = x[j]; if (v < 0) v = -v; if (v > asmax) asmax = v; }
        float d = asmax / 127.0f;
        y[i].d = fp32_to_fp16(d);
        float id = (asmax != 0.0f) ? 127.0f / asmax : 0.0f;
        for (int j = 0; j < 32; j++) {
            int v = (int)lroundf(x[j] * id);
            if (v > 127) v = 127;
            if (v < -127) v = -128;
            y[i].qs[j] = (int8_t)v;
        }
        x += 32;
    }
    } /* end AVX-512 block */
#elif defined(PICOLM_AVX2)
    /* AVX2: same as AVX but uses _mm256_permutevar8x32_epi32 for shuffle */
    for (int i = 0; i < nb; i++) {
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;

        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        float maxScalar = _mm_cvtss_f32(max4);

        float d = maxScalar / 127.0f;
        y[i].d = fp32_to_fp16(d);
        float id = (maxScalar != 0.0f) ? 127.0f / maxScalar : 0.0f;
        __m256 mul = _mm256_set1_ps(id);

        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);
        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
    }
#elif defined(PICOLM_AVX)
    for (int i = 0; i < nb; i++) {
        /* Compute max(abs(e)) for the block */
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;

        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        float maxScalar = _mm_cvtss_f32(max4);

        float d = maxScalar / 127.0f;
        y[i].d = fp32_to_fp16(d);
        float id = (maxScalar != 0.0f) ? 127.0f / maxScalar : 0.0f;
        __m256 mul = _mm256_set1_ps(id);

        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

        /* int32 -> int16 -> int8 via packs (same as llama.cpp AVX path) */
        __m128i ni0 = _mm256_castsi256_si128(i0);
        __m128i ni1 = _mm256_extractf128_si256(i0, 1);
        __m128i ni2 = _mm256_castsi256_si128(i1);
        __m128i ni3 = _mm256_extractf128_si256(i1, 1);
        __m128i ni4 = _mm256_castsi256_si128(i2);
        __m128i ni5 = _mm256_extractf128_si256(i2, 1);
        __m128i ni6 = _mm256_castsi256_si128(i3);
        __m128i ni7 = _mm256_extractf128_si256(i3, 1);

        ni0 = _mm_packs_epi32(ni0, ni1);
        ni2 = _mm_packs_epi32(ni2, ni3);
        ni4 = _mm_packs_epi32(ni4, ni5);
        ni6 = _mm_packs_epi32(ni6, ni7);
        ni0 = _mm_packs_epi16(ni0, ni2);
        ni4 = _mm_packs_epi16(ni4, ni6);

        _mm_storeu_si128((__m128i *)(y[i].qs + 0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
    }
#elif defined(PICOLM_NEON)
    /* NEON quantize_row_q8_0: adapted from llama.cpp ARM impl */
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv[8], asrcv[8], amaxv[8];
        for (int j = 0; j < 8; j++) srcv[j] = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);
        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);
        const float amax = vmaxvq_f32(amaxv[0]);
        const float d = amax / 127.0f;
        y[i].d = fp32_to_fp16(d);
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int j = 0; j < 8; j++) {
            const float32x4_t v = vmulq_n_f32(srcv[j], id);
            const int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[4*j+0] = (int8_t)vgetq_lane_s32(vi, 0);
            y[i].qs[4*j+1] = (int8_t)vgetq_lane_s32(vi, 1);
            y[i].qs[4*j+2] = (int8_t)vgetq_lane_s32(vi, 2);
            y[i].qs[4*j+3] = (int8_t)vgetq_lane_s32(vi, 3);
        }
    }
#elif defined(PICOLM_SSE2)
    for (int i = 0; i < nb; i++) {
        float maxAbs = 0.0f;
        for (int j = 0; j < 32; j++) {
            float a = x[j] < 0 ? -x[j] : x[j];
            if (a > maxAbs) maxAbs = a;
        }
        float d = maxAbs / 127.0f;
        y[i].d = fp32_to_fp16(d);
        float id = (maxAbs != 0.0f) ? 127.0f / maxAbs : 0.0f;
        for (int j = 0; j < 32; j++) {
            y[i].qs[j] = (int8_t)((int)(x[j] * id + (x[j] >= 0 ? 0.5f : -0.5f)));
        }
        x += 32;
    }
#else
    for (int i = 0; i < nb; i++) {
        float maxAbs = 0.0f;
        for (int j = 0; j < 32; j++) {
            float a = x[j] < 0 ? -x[j] : x[j];
            if (a > maxAbs) maxAbs = a;
        }
        float d = maxAbs / 127.0f;
        y[i].d = fp32_to_fp16(d);
        float id = (maxAbs != 0.0f) ? 127.0f / maxAbs : 0.0f;
        for (int j = 0; j < 32; j++) {
            y[i].qs[j] = (int8_t)((int)(x[j] * id + (x[j] >= 0 ? 0.5f : -0.5f)));
        }
        x += 32;
    }
#endif
}

/* ================================================================
 * quantize_row_q8_K: quantize float32 -> Q8_K blocks
 * Used for intermediate quantization in Q4_K/Q6_K matmul
 * Adapted from llama.cpp's quantize_row_q8_K_ref
 */
void quantize_row_q8_K(const float *x, void *dst, int n) {
    block_q8_K *y = (block_q8_K *)dst;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < 256; ++j) {
            float ax = x[j] < 0 ? -x[j] : x[j];
            if (ax > amax) amax = ax;
        }
        float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
        y[i].d = 1.0f / id;
        
#ifdef PICOLM_NEON
        for (int j = 0; j < 256; j += 8) {
            float32x4_t v0 = vld1q_f32(x + j);
            float32x4_t v1 = vld1q_f32(x + j + 4);
            int32x4_t vi0 = vcvtnq_s32_f32(vmulq_n_f32(v0, id));
            int32x4_t vi1 = vcvtnq_s32_f32(vmulq_n_f32(v1, id));
            int16x4_t s0 = vmovn_s32(vi0);
            int16x4_t s1 = vmovn_s32(vi1);
            int16x8_t s8 = vcombine_s16(s0, s1);
            int8x8_t qi = vmovn_s16(s8);
            vst1_s8(y[i].qs + j, qi);
        }
#elif defined(PICOLM_AVX)
        const __m256 v_id = _mm256_set1_ps(id);
        for (int j = 0; j < 256; j += 32) {
            __m256 v0 = _mm256_loadu_ps(x + j + 0);
            __m256 v1 = _mm256_loadu_ps(x + j + 8);
            __m256 v2 = _mm256_loadu_ps(x + j + 16);
            __m256 v3 = _mm256_loadu_ps(x + j + 24);
            __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, v_id), _MM_ROUND_NEAREST));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, v_id), _MM_ROUND_NEAREST));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, v_id), _MM_ROUND_NEAREST));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, v_id), _MM_ROUND_NEAREST));
            __m128i p0 = _mm256_castsi256_si128(i0);
            __m128i p1 = _mm256_extractf128_si256(i0, 1);
            __m128i p2 = _mm256_castsi256_si128(i1);
            __m128i p3 = _mm256_extractf128_si256(i1, 1);
            p0 = _mm_packs_epi32(p0, p1);
            p2 = _mm_packs_epi32(p2, p3);
            p0 = _mm_packs_epi16(p0, p2);
            _mm_storeu_si128((__m128i *)(y[i].qs + j), p0);
            /* Next 16 */
            p0 = _mm256_castsi256_si128(i2);
            p1 = _mm256_extractf128_si256(i2, 1);
            p2 = _mm256_castsi256_si128(i3);
            p3 = _mm256_extractf128_si256(i3, 1);
            p0 = _mm_packs_epi32(p0, p1);
            p2 = _mm_packs_epi32(p2, p3);
            p0 = _mm_packs_epi16(p0, p2);
            _mm_storeu_si128((__m128i *)(y[i].qs + j + 16), p0);
        }
#else
        for (int j = 0; j < 256; j++) {
            y[i].qs[j] = (int8_t)((int)(x[j] * id + (x[j] >= 0 ? 0.5f : -0.5f)));
        }
#endif

        /* Compute bsums: sum of quants in groups of 16 */
#if defined(PICOLM_AVX2)
        {
            const __m256i ones = _mm256_set1_epi16(1);
            for (int j = 0; j < 16; ++j) {
                __m128i qs = _mm_loadu_si128((const __m128i*)(y[i].qs + j * 16));
                __m256i qs16 = _mm256_cvtepi8_epi16(qs);  /* 16 int8 -> 16 int16 (sign-extended) */
                __m256i s0 = _mm256_madd_epi16(qs16, ones);
                /* Horizontal sum of 16 int32 -> single int32 */
                __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(s0), _mm256_extractf128_si256(s0, 1));
                s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0x1B));
                s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0x4E));
                y[i].bsums[j] = (int16_t)_mm_cvtsi128_si32(s128);
            }
        }
#elif defined(PICOLM_AVX)
        {
            const __m128i ones = _mm_set1_epi16(1);
            for (int j = 0; j < 16; ++j) {
                __m128i qs = _mm_loadu_si128((const __m128i*)(y[i].qs + j * 16));
                /* Sign-extend low 8 bytes -> 8 int16, then high 8 bytes -> 8 int16 */
                __m128i lo = _mm_cvtepi8_epi16(qs);          /* sign-extend bytes 0-7 */
                __m128i hi = _mm_cvtepi8_epi16(_mm_bsrli_si128(qs, 8));  /* sign-extend bytes 8-15 */
                __m128i s0 = _mm_madd_epi16(lo, ones);
                __m128i s1 = _mm_madd_epi16(hi, ones);
                s0 = _mm_add_epi32(s0, s1);
                s0 = _mm_add_epi32(s0, _mm_shuffle_epi32(s0, 0x1B));
                s0 = _mm_add_epi32(s0, _mm_shuffle_epi32(s0, 0x4E));
                y[i].bsums[j] = (int16_t)_mm_cvtsi128_si32(s0);
            }
        }
#else
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j * 16 + ii];
            }
            y[i].bsums[j] = (int16_t)sum;
        }
#endif
        x += 256;
    }
}

/* ================================================================ */
#ifdef PICOLM_AVX2
/* Q2_K scale shuffle table for AVX2.
 * The pattern is identical for both chunks (j=0 and j=1) since scales[j]
 * holds 8 local scale int16s duplicated across both 128-bit lanes.
 * For shift s: low 8 lanes need scale 2s (bytes 4s,4s+1),
 * high 8 lanes need scale 2s+1 (bytes 4s+2,4s+3).
 * Matches llama.cpp's get_scale_shuffle_q3k (128 bytes, 4 entries). */
static inline __m256i get_scale_shuffle_q2k(int shift) {
    #ifdef _MSC_VER
    static const __declspec(align(32)) uint8_t k_shuffle[128] = {
#else
    static const uint8_t k_shuffle[128] __attribute__((aligned(32))) = {
#endif
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,  6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13, 14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + shift);
}
#endif

/* vec_dot_q4_K_q8_K: int8 MAC for Q4_K weights * Q8_K input
 * Adapted from llama.cpp's ggml_vec_dot_q4_K_q8_K (AVX2, AVX1, NEON, scalar)
 * The key optimization: nibble extraction to int8, int8 MAC with
 * per-subblock scale factors, only 8 final float ops per block.
 */
float vec_dot_q4_K_q8_K(const void *src_q4, const void *src_q8, int n) {
    const block_q4_K *x = (const block_q4_K *)src_q4;
    const block_q8_K *y = (const block_q8_K *)src_q8;
    int nb = n / 256;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ifdef PICOLM_AVX2
    /* AVX2 path: 256-bit SIMD nibble extraction + maddubs_epi16 */
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32_lookup(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales256 = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

        static const uint8_t k_shuffle[256] = {
             0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
             6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
             8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
            10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
            12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
            14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
        };
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales256,
                _mm256_loadu_si256((const __m256i*)k_shuffle + 2*j));
            const __m256i scale_h = _mm256_shuffle_epi8(scales256,
                _mm256_loadu_si256((const __m256i*)k_shuffle + 2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    __m128 res = _mm256_extractf128_ps(acc, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));

    return _mm_cvtss_f32(res) + _mm_cvtss_f32(acc_m);

#elif defined(PICOLM_AVX)
    /* AVX1 path: 128-bit integer + 256-bit float accumulation */
    const __m128i m4_128 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32_lookup(x[i].dmin);

        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales128 = _mm_unpacklo_epi8(utmps, _mm_setzero_si128());
        const __m128i mins128 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(utmps, utmps), _mm_setzero_si128());

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins128, q8s);
        acc_m = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < 4; ++j) {
            const __m128i scale_l = _mm_shuffle_epi8(scales128, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_h = _mm_shuffle_epi8(scales128, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            __m128i q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_0 = _mm_and_si128(q4bits, m4_128);
            const __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4_128);
            q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_1 = _mm_and_si128(q4bits, m4_128);
            const __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4_128);

            const __m128i q8l_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16l = _mm_maddubs_epi16(q4l_0, q8l_0);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_0 = _mm_add_epi32(sumi_0, p16l);
            const __m128i q8l_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16l = _mm_maddubs_epi16(q4l_1, q8l_1);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_1 = _mm_add_epi32(sumi_1, p16l);

            const __m128i q8h_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16h = _mm_maddubs_epi16(q4h_0, q8h_0);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_0 = _mm_add_epi32(sumi_0, p16h);
            const __m128i q8h_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16h = _mm_maddubs_epi16(q4h_1, q8h_1);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_1 = _mm_add_epi32(sumi_1, p16h);
        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = _mm256_insertf128_si256(_mm256_castsi128_si256(sumi_0), sumi_1, 1);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    __m128 res = _mm256_extractf128_ps(acc, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));

    return _mm_cvtss_f32(res) + _mm_cvtss_f32(acc_m);

#elif defined(PICOLM_SSSE3)
    /* SSSE3/SSE3 path: 128-bit integer + 128-bit float accumulation.
     * Uses pmaddubsw (_mm_maddubs_epi16) for signed int8 MAC.
     * Same algorithm as AVX1 path but with SSE registers throughout. */
    const __m128i m4_128 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);
    __m128 acc = _mm_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32_lookup(x[i].dmin);

        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales128 = _mm_unpacklo_epi8(utmps, _mm_setzero_si128());
        const __m128i mins128 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(utmps, utmps), _mm_setzero_si128());

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins128, q8s);
        acc_m = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < 4; ++j) {
            const __m128i scale_l = _mm_shuffle_epi8(scales128, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_h = _mm_shuffle_epi8(scales128, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            __m128i q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_0 = _mm_and_si128(q4bits, m4_128);
            const __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4_128);
            q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_1 = _mm_and_si128(q4bits, m4_128);
            const __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4_128);

            const __m128i q8l_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16l = _mm_maddubs_epi16(q4l_0, q8l_0);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_0 = _mm_add_epi32(sumi_0, p16l);
            const __m128i q8l_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16l = _mm_maddubs_epi16(q4l_1, q8l_1);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_1 = _mm_add_epi32(sumi_1, p16l);

            const __m128i q8h_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16h = _mm_maddubs_epi16(q4h_0, q8h_0);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_0 = _mm_add_epi32(sumi_0, p16h);
            const __m128i q8h_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16h = _mm_maddubs_epi16(q4h_1, q8h_1);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_1 = _mm_add_epi32(sumi_1, p16h);
        }

        const __m128 vd = _mm_set1_ps(d);
        __m128 sf = _mm_add_ps(_mm_cvtepi32_ps(sumi_0), _mm_cvtepi32_ps(sumi_1));
        acc = _mm_add_ps(acc, _mm_mul_ps(vd, sf));
    }

    acc_m = _mm_add_ps(acc_m, acc);
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    return _mm_cvtss_f32(acc_m);

#elif defined(PICOLM_I8MM)
    /* I8MM: vmmlaq_s32 for 16x int8 MAC -> 4x int32 lanes per call.
     * Q4_K has 8 sub-blocks of 32 values each (256 total).
     * Each sub-block has its own 6-bit scale factor.
     *
     * Layout: qs[0..127] (128 bytes = 256 nibbles).
     * Scalar extraction: for j=0..3, read qs[j*32..j*32+31] (32 bytes):
     *   low nibbles  (32 values) -> sub-block j*2
     *   high nibbles (32 values) -> sub-block j*2+1
     * Each sub-block pairs with 32 consecutive Q8_K values.
     *
     * vmmlaq_s32 processes 16 int8 x int8 -> 4 int32 lanes.
     * Lanes 0 and 3 each accumulate an 8-element dot product.
     * For 32 values, need 2 vmmlaq calls (16+16). */
    const uint8x16_t m4 = vdupq_n_u8(0x0F);

    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        /* Decode 6-bit scales and mins from packed 12 bytes */
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t *scales8 = (const uint8_t *)&utmp[0];
        const uint8_t *mins8   = (const uint8_t *)&utmp[2];

        /* Compute dmin * sum(bsums * mins) for bias correction */
        int sumi = 0;
        for (int j = 0; j < 16; j++) sumi += y[i].bsums[j] * (int)mins8[j / 2];

        const float d    = fp16_to_fp32_lookup(x[i].d) * y[i].d;
        const float dmin = fp16_to_fp32_lookup(x[i].dmin) * y[i].d;

        int32_t sub_sums[8] = {0};

        /* Process 4 groups of 32 bytes (64 nibbles = 32 low + 32 high) */
        for (int j = 0; j < 4; j++) {
            /* Load 32 Q4_K bytes -> 64 nibbles: 32 low + 32 high */
            const uint8x16_t q4a = vld1q_u8(q4);
            const uint8x16_t q4b = vld1q_u8(q4 + 16);
            q4 += 32;

            const int8x16_t q4lo_a = vreinterpretq_s8_u8(vandq_u8(q4a, m4));
            const int8x16_t q4lo_b = vreinterpretq_s8_u8(vandq_u8(q4b, m4));
            const int8x16_t q4hi_a = vreinterpretq_s8_u8(vshrq_n_u8(q4a, 4));
            const int8x16_t q4hi_b = vreinterpretq_s8_u8(vshrq_n_u8(q4b, 4));

            /* Sub-block j*2: low nibbles (32 values) vs q8[32*j*2 .. 32*j*2+31] */
            int sb_lo = j * 2;
            const int8x16_t q8a = vld1q_s8(q8); q8 += 16;
            const int8x16_t q8b = vld1q_s8(q8); q8 += 16;
            int32x4_t s0 = vmmlaq_s32(vdupq_n_s32(0), q4lo_a, q8a);
            int32x4_t s1 = vmmlaq_s32(vdupq_n_s32(0), q4lo_b, q8b);
            sub_sums[sb_lo] += vaddvq_s32(vaddq_s32(s0, s1)) * scales8[sb_lo];

            /* Sub-block j*2+1: high nibbles (32 values) vs q8[32*(j*2+1) .. 32*(j*2+1)+31] */
            int sb_hi = j * 2 + 1;
            const int8x16_t q8c = vld1q_s8(q8); q8 += 16;
            const int8x16_t q8d = vld1q_s8(q8); q8 += 16;
            int32x4_t s2 = vmmlaq_s32(vdupq_n_s32(0), q4hi_a, q8c);
            int32x4_t s3 = vmmlaq_s32(vdupq_n_s32(0), q4hi_b, q8d);
            sub_sums[sb_hi] += vaddvq_s32(vaddq_s32(s2, s3)) * scales8[sb_hi];
        }

        /* Sum all 8 sub-block contributions */
        int32_t total = 0;
        for (int j = 0; j < 8; j++) total += sub_sums[j];

        sumf += d * (float)total - dmin * (float)sumi;
    }
    return sumf;

#elif defined(PICOLM_NEON)
    {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        { const uint32_t uaux = utmp[1] & kmask1; utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4); utmp[2] = uaux; }
        utmp[0] &= kmask1;
        const uint8_t *scales8 = (const uint8_t *)&utmp[0];
        const uint8_t *mins8 = (const uint8_t *)&utmp[2];
        int sumi = 0;
        for (int j = 0; j < 16; j++) sumi += y[i].bsums[j] * (int)mins8[j / 2];
        const float d = fp16_to_fp32_lookup(x[i].d) * y[i].d;
        const float dmin = fp16_to_fp32_lookup(x[i].dmin) * y[i].d;
        int32_t sub_sums[8] = {0};
        for (int j = 0; j < 4; j++) {
            const uint8x16_t q4a = vld1q_u8(q4);
            const uint8x16_t q4b = vld1q_u8(q4 + 16);
            q4 += 32;
            const int8x16_t q4lo_a = vreinterpretq_s8_u8(vandq_u8(q4a, m4));
            const int8x16_t q4lo_b = vreinterpretq_s8_u8(vandq_u8(q4b, m4));
            const int8x16_t q4hi_a = vreinterpretq_s8_u8(vshrq_n_u8(q4a, 4));
            const int8x16_t q4hi_b = vreinterpretq_s8_u8(vshrq_n_u8(q4b, 4));
            {
                const int8x16_t q8a = vld1q_s8(q8); q8 += 16;
                const int8x16_t q8b = vld1q_s8(q8); q8 += 16;
                int16x8_t p0 = vmull_s8(vget_low_s8(q4lo_a), vget_low_s8(q8a));
                int16x8_t p1 = vmull_s8(vget_high_s8(q4lo_a), vget_high_s8(q8a));
                int16x8_t p2 = vmull_s8(vget_low_s8(q4lo_b), vget_low_s8(q8b));
                int16x8_t p3 = vmull_s8(vget_high_s8(q4lo_b), vget_high_s8(q8b));
                int32x4_t s = vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)), vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)));
                sub_sums[j*2] += vaddvq_s32(s) * (int)scales8[j*2];
            }
            {
                const int8x16_t q8c = vld1q_s8(q8); q8 += 16;
                const int8x16_t q8d = vld1q_s8(q8); q8 += 16;
                int16x8_t p0 = vmull_s8(vget_low_s8(q4hi_a), vget_low_s8(q8c));
                int16x8_t p1 = vmull_s8(vget_high_s8(q4hi_a), vget_high_s8(q8c));
                int16x8_t p2 = vmull_s8(vget_low_s8(q4hi_b), vget_low_s8(q8d));
                int16x8_t p3 = vmull_s8(vget_high_s8(q4hi_b), vget_high_s8(q8d));
                int32x4_t s = vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)), vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)));
                sub_sums[j*2+1] += vaddvq_s32(s) * (int)scales8[j*2+1];
            }
        }
        int32_t total = 0;
        for (int j = 0; j < 8; j++) total += sub_sums[j];
        sumf += d * (float)total - dmin * (float)sumi;
    }
    return sumf;
}

#else
    /* Scalar fallback: nibble extraction + int8 MAC */
    /* Used on x86 without AVX or NEON without I8MM */
    const uint8_t *scales = (const uint8_t *)&utmp[0];
    const uint8_t *mins   = (const uint8_t *)&utmp[2];
    int8_t  aux8[256];
    int16_t aux16[8];
    float   sums[8] = {0};
    int32_t aux32[8];

    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        int8_t *a = aux8;
        for (int j = 0; j < 4; j++) {
            for (int l = 0; l < 32; l++) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; l++) a[l] = (int8_t)(q4[l] >> 4);
            a += 32;
            q4 += 32;
        }

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < 16; j++) sumi += y[i].bsums[j] * (int)mins[j / 2];

        memset(aux32, 0, sizeof(aux32));
        a = aux8;
        int is = 0;
        for (int j = 0; j < 8; j++) {
            int32_t scale = (int32_t)scales[is++];
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }

        const float d = fp16_to_fp32_lookup(x[i].d) * y[i].d;
        for (int l = 0; l < 8; l++) sums[l] += d * (float)aux32[l];

        const float dmin = fp16_to_fp32_lookup(x[i].dmin) * y[i].d;
        sumf -= dmin * (float)sumi;
    }
    for (int l = 0; l < 8; l++) sumf += sums[l];
    return sumf;
#endif
}

/* ================================================================
 * vec_dot_q8_0_q8_0: int8 MAC for two Q8_0 vectors
 *
 * Three tiers (mirrors llama.cpp/llamafile approach):
 *   AVX2: 256-bit int8 MAC via _mm256_maddubs_epi16, 32 pairs/block
 *   AVX:  128-bit int8 MAC via _mm_maddubs_epi16, 256-bit float accum
 *   SSE2: 128-bit int8 MAC, scalar float accum
 * ================================================================ */

/* mul_sum_i8_pairs_avx512 is now defined in quant.h (shared with tensor.c) */

#if defined(PICOLM_AVX2) || defined(PICOLM_AVX) || defined(PICOLM_SSSE3)
static inline __m128i mul_sum_i8_pairs_sse(const __m128i x, const __m128i y) {
    __m128i ax = _mm_sign_epi8(x, x);
    __m128i sy = _mm_sign_epi8(y, x);
    __m128i dot = _mm_maddubs_epi16(ax, sy);
    __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, dot);
}
#endif

#if defined(PICOLM_AVX2)
/* AVX2-only helpers */
static inline __m256i mul_sum_i8_pairs_avx2(const __m256i x, const __m256i y) {
    __m256i ax = _mm256_sign_epi8(x, x);
    __m256i sy = _mm256_sign_epi8(y, x);
    __m256i dot = _mm256_maddubs_epi16(ax, sy);
    __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(ones, dot);
}

static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, x));
}

static inline __m256i bytes_from_nibbles_32(const uint8_t *qs) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)qs);
    __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    return _mm256_and_si256(bytes, _mm256_set1_epi8(0xF));
}

/* Expand 32 bits into a 256-bit vector of 0x00 or 0xFF per byte.
 * Each input byte contains 8 sign bits; output has 32 bytes (4 lanes of 8).
 * Adapted from llama.cpp bytes_from_bits_32. */
static inline __m256i bytes_from_bits_32(const uint8_t *x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
            0x0303030303030303ULL, 0x0202020202020202ULL,
            0x0101010101010101ULL, 0x0000000000000000ULL);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeULL);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
}
#elif defined(PICOLM_AVX)
/* AVX-only helpers (no AVX2 intrinsics) */
static inline __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i sh = _mm_madd_epi16(ones, xh);
    const __m128i sl = _mm_madd_epi16(ones, xl);
    return _mm256_cvtepi32_ps(_mm256_set_m128i(sh, sl));
}

static inline __m128i mul_add_epi8_sse(const __m128i x, const __m128i y) {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    return _mm_maddubs_epi16(ax, sy);
}
#endif
/* ================================================================
 * vec_dot_q5_K_q8_K: int8 MAC for Q5_K weights * Q8_K input
 * Adapted from llama.cpp's ggml_vec_dot_q5_K_q8_K (AVX2, AVX, scalar).
 *
 * Q5_K: 256 values/block, 5-bit values (4 low + 1 high).
 * 8 sub-blocks of 32, each with 6-bit scale and 6-bit min (packed 12 bytes).
 * ================================================================ */
/* ================================================================
 * vec_dot_q5_K_q8_K: int8 MAC for Q5_K weights * Q8_K input
 * Directly ported from llama.cpp ggml_vec_dot_q5_K_q8_K (x86 quants.c)
 * ================================================================ */
float vec_dot_q5_K_q8_K(const void *src_q5, const void *src_q8, int n) {
    const block_q5_K *x = (const block_q5_K *)src_q5;
    const block_q8_K *y = (const block_q8_K *)src_q8;
    const int nb = n / 256;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

#if defined(PICOLM_AVX2)
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t *q5 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32_lookup(x[i].dm);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * (float)_mm_extract_epi32(hsum, 0);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i hmask = mone;
        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;

        for (int j = 0; j < 4; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, _mm256_loadu_si256((const __m256i*)k_shuffle + 2*j+0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, _mm256_loadu_si256((const __m256i*)k_shuffle + 2*j+1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_avx(acc) + summs;

#elif defined(PICOLM_AVX)
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m128i mone  = _mm_set1_epi8(1);
    const __m128i m2 = _mm_set1_epi8(2);
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32_lookup(x[i].dm);
        const uint8_t *q5 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_unpacklo_epi8(utmps, _mm_setzero_si128());
        const __m128i mins = _mm_unpacklo_epi8(_mm_unpackhi_epi64(utmps, utmps), _mm_setzero_si128());

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins, q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * (float)_mm_extract_epi32(hsum, 0);

        const __m128i hbits_0 = _mm_loadu_si128((const __m128i*)&x[i].qh[0]);
        const __m128i hbits_1 = _mm_loadu_si128((const __m128i*)&x[i].qh[16]);
        __m128i hmask = mone;
        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();
        int bit = 0;
        __m128i shuffle = _mm_set1_epi16(0x0100);

        for (int j = 0; j < 4; ++j) {
            const __m128i scale_0 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_1 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            const __m128i q5bits_0 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;
            const __m128i q5bits_1 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;

            __m128i q5l_0 = _mm_and_si128(q5bits_0, m4);
            __m128i q5l_1 = _mm_and_si128(q5bits_1, m4);
            __m128i q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), bit), 4);
            __m128i q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), bit++), 4);
            __m128i q5_0  = _mm_add_epi8(q5l_0, q5h_0);
            __m128i q5_1  = _mm_add_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16_0 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm_madd_epi16(scale_0, p16_0);
            p16_1 = _mm_madd_epi16(scale_0, p16_1);

            q5l_0 = _mm_and_si128(_mm_srli_epi16(q5bits_0, 4), m4);
            q5l_1 = _mm_and_si128(_mm_srli_epi16(q5bits_1, 4), m4);
            q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), bit), 4);
            q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), bit++), 4);
            q5_0  = _mm_add_epi8(q5l_0, q5h_0);
            q5_1  = _mm_add_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16_2 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_3 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_2 = _mm_madd_epi16(scale_1, p16_2);
            p16_3 = _mm_madd_epi16(scale_1, p16_3);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = _mm256_insertf128_si256(_mm256_castsi128_si256(sumi_0), sumi_1, 1);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);
    }
    return hsum_avx(acc) + summs;

#elif defined(PICOLM_NEON)
    {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q4 = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const int8_t  *q8 = y[i].qs;
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        { const uint32_t uaux = utmp[1] & kmask1; utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4); utmp[2] = uaux; }
        utmp[0] &= kmask1;
        const uint8_t *scales8 = (const uint8_t *)&utmp[0];
        const uint8_t *mins8 = (const uint8_t *)&utmp[2];
        int sumi = 0;
        for (int j = 0; j < 16; j++) sumi += y[i].bsums[j] * (int)mins8[j / 2];
        const float d = fp16_to_fp32_lookup(x[i].d) * y[i].d;
        const float dmin = fp16_to_fp32_lookup(x[i].dm) * y[i].d;
        int32_t sub_sums[8] = {0};
        int qh_bit = 1;
        for (int j = 0; j < 4; j++) {
            const uint8x16_t q4a = vld1q_u8(q4);
            const uint8x16_t q4b = vld1q_u8(q4 + 16);
            q4 += 32;
            const int8x16_t q4lo_a = vreinterpretq_s8_u8(vandq_u8(q4a, m4));
            const int8x16_t q4lo_b = vreinterpretq_s8_u8(vandq_u8(q4b, m4));
            const int8x16_t q4hi_a = vreinterpretq_s8_u8(vshrq_n_u8(q4a, 4));
            const int8x16_t q4hi_b = vreinterpretq_s8_u8(vshrq_n_u8(q4b, 4));
            /* Build high-bit correction arrays */
            int8_t qh_a[16], qh_b[16], qh_c[16], qh_d[16];
            for (int l = 0; l < 16; l++) {
                qh_a[l] = (qh[l] & qh_bit) ? 16 : 0;
                qh_b[l] = (qh[l+16] & qh_bit) ? 16 : 0;
                qh_c[l] = (qh[l+32] & qh_bit) ? 16 : 0;
                qh_d[l] = (qh[l+48] & qh_bit) ? 16 : 0;
            }
            const int8x16_t qha = vld1q_s8(qh_a);
            const int8x16_t qhb = vld1q_s8(qh_b);
            const int8x16_t qhc = vld1q_s8(qh_c);
            const int8x16_t qhd = vld1q_s8(qh_d);
            /* Sub-block j*2: low nibbles + high bit */
            {
                const int8x16_t q5a = vaddq_s8(q4lo_a, qha);
                const int8x16_t q5b = vaddq_s8(q4lo_b, qhb);
                const int8x16_t q8a = vld1q_s8(q8); q8 += 16;
                const int8x16_t q8b = vld1q_s8(q8); q8 += 16;
                int16x8_t p0 = vmull_s8(vget_low_s8(q5a), vget_low_s8(q8a));
                int16x8_t p1 = vmull_s8(vget_high_s8(q5a), vget_high_s8(q8a));
                int16x8_t p2 = vmull_s8(vget_low_s8(q5b), vget_low_s8(q8b));
                int16x8_t p3 = vmull_s8(vget_high_s8(q5b), vget_high_s8(q8b));
                int32x4_t s = vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)), vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)));
                sub_sums[j*2] += vaddvq_s32(s) * (int)scales8[j*2];
            }
            /* Sub-block j*2+1: high nibbles + high bit */
            {
                const int8x16_t q5a = vaddq_s8(q4hi_a, qhc);
                const int8x16_t q5b = vaddq_s8(q4hi_b, qhd);
                const int8x16_t q8c = vld1q_s8(q8); q8 += 16;
                const int8x16_t q8d = vld1q_s8(q8); q8 += 16;
                int16x8_t p0 = vmull_s8(vget_low_s8(q5a), vget_low_s8(q8c));
                int16x8_t p1 = vmull_s8(vget_high_s8(q5a), vget_high_s8(q8c));
                int16x8_t p2 = vmull_s8(vget_low_s8(q5b), vget_low_s8(q8d));
                int16x8_t p3 = vmull_s8(vget_high_s8(q5b), vget_high_s8(q8d));
                int32x4_t s = vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)), vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)));
                sub_sums[j*2+1] += vaddvq_s32(s) * (int)scales8[j*2+1];
            }
            qh_bit <<= 1;
        }
        int32_t total = 0;
        for (int j = 0; j < 8; j++) total += sub_sums[j];
        sumf += d * (float)total - dmin * (float)sumi;
    }
    return sumf;
}

#else
    /* Scalar fallback: ported from llama.cpp ggml_vec_dot_q5_K_q8_K_generic */
    {
        int8_t aux8[256];
        int32_t aux32[8];
        float sums[8];
        float sumf = 0.0f;

        for (int s = 0; s < 8; s++) sums[s] = 0.0f;

        for (int i = 0; i < nb; i++) {
            const uint8_t *q4 = x[i].qs;
            const uint8_t *hm = x[i].qh;
            const int8_t *q8 = y[i].qs;

            for (int s = 0; s < 8; s++) aux32[s] = 0;

            int8_t *a = aux8;
            uint8_t m = 1;
            for (int j = 0; j < 4; ++j) {
                for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
                for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
                a += 32; m <<= 1;
                for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
                for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
                a += 32; m <<= 1;
                q4 += 32;
            }
            memcpy(utmp, x[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            { const uint32_t uaux = utmp[1] & kmask1; utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4); utmp[2] = uaux; }
            utmp[0] &= kmask1;

            const uint8_t *scales = (const uint8_t *)&utmp[0];
            const uint8_t *mins = (const uint8_t *)&utmp[2];
            
            int sumi = 0;
            for (int j = 0; j < 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];

            a = aux8;
            int is = 0;
            for (int j = 0; j < 8; ++j) {
                int32_t scale = scales[is++];
                for (int g = 0; g < 4; g++) {
                    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8[l] * a[l]);
                    q8 += 8; a += 8;
                }
            }

            float d = fp16_to_fp32_lookup(x[i].d) * y[i].d;
            for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];

            float dmin = fp16_to_fp32_lookup(x[i].dm) * y[i].d;
            sumf -= dmin * sumi;
        }
        for (int l = 0; l < 8; ++l) sumf += sums[l];
        return sumf;
    }
#endif
}

/* ================================================================
 * vec_dot_q3_K_q8_K: int8 MAC for Q3_K weights * Q8_K input
 * Directly ported from llama.cpp ggml_vec_dot_q3_K_q8_K (x86 quants.c)
 * ================================================================ */
float vec_dot_q3_K_q8_K(const void *src_q3, const void *src_q8, int n) {
    const block_q3_K *x = (const block_q3_K *)src_q3;
    const block_q8_K *y = (const block_q8_K *)src_q8;
    const int nb = n / 256;

#if defined(PICOLM_AVX2)
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[3];
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
        const uint8_t *q3 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {
            _mm256_insertf128_si256(_mm256_castsi128_si256(l_scales), l_scales, 1),
            _mm256_insertf128_si256(_mm256_castsi128_si256(h_scales), h_scales, 1)
        };

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);
        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;
        int is = 0;

        for (int j = 0; j < 2; ++j) {
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;

            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], _mm256_loadu_si256((const __m256i*)k_shuffle + (is + 0))), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], _mm256_loadu_si256((const __m256i*)k_shuffle + (is + 1))), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], _mm256_loadu_si256((const __m256i*)k_shuffle + (is + 2))), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], _mm256_loadu_si256((const __m256i*)k_shuffle + (is + 3))), p16_3);

            p16_0 = _mm256_add_epi32(p16_0, p16_1);
            p16_2 = _mm256_add_epi32(p16_2, p16_3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_avx(acc);

#elif defined(PICOLM_AVX)
    const __m128i m3   = _mm_set1_epi8(3);
    const __m128i mone = _mm_set1_epi8(1);
    const __m128i m32  = _mm_set1_epi8(32);
    const __m128i m2   = _mm_set1_epi8(2);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    __m256 acc = _mm256_setzero_ps();
    uint32_t auxv[3];

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
        const uint8_t *q3 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        memcpy(auxv, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(
                ((auxv[1] >> 4) & kmask2) | (((auxv[2] >> 6) & kmask1) << 4),
                ((auxv[0] >> 4) & kmask2) | (((auxv[2] >> 4) & kmask1) << 4),
                (auxv[1] & kmask2) | (((auxv[2] >> 2) & kmask1) << 4),
                (auxv[0] & kmask2) | (((auxv[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m128i scales_0 = _mm_cvtepi8_epi16(scales128);
        const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales128, scales128));
        const __m128i scales[2] = {scales_0, scales_1};

        const __m128i hbits_0 = _mm_loadu_si128((const __m128i*)&x[i].hmask[0]);
        const __m128i hbits_1 = _mm_loadu_si128((const __m128i*)&x[i].hmask[16]);
        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        for (int j = 0; j < 2; ++j) {
            const __m128i q3bits_0 = _mm_loadu_si128((const __m128i*)q3); q3 += 16;
            const __m128i q3bits_1 = _mm_loadu_si128((const __m128i*)q3); q3 += 16;
            const int bit = j << 2;

            const __m128i q3l_0 = _mm_and_si128(q3bits_0, m3);
            const __m128i q3l_1 = _mm_and_si128(q3bits_1, m3);
            const __m128i q3h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit)), bit), 2);
            const __m128i q3h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit)), bit), 2);
            const __m128i q3l_2 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 2), m3);
            const __m128i q3l_3 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 2), m3);
            const __m128i q3h_2 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+1)), bit+1), 2);
            const __m128i q3h_3 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+1)), bit+1), 2);
            const __m128i q3l_4 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 4), m3);
            const __m128i q3l_5 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 4), m3);
            const __m128i q3h_4 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+2)), bit+2), 2);
            const __m128i q3h_5 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+2)), bit+2), 2);
            const __m128i q3l_6 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 6), m3);
            const __m128i q3l_7 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 6), m3);
            const __m128i q3h_6 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+3)), bit+3), 2);
            const __m128i q3h_7 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+3)), bit+3), 2);

            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)(q8+0)); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)(q8+0)); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)(q8+0)); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)(q8+0)); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)(q8+0)); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)(q8+0)); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)(q8+0)); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)(q8+0)); q8 += 16;

            __m128i q8s_0 = _mm_maddubs_epi16(q3h_0, q8_0);
            __m128i q8s_1 = _mm_maddubs_epi16(q3h_1, q8_1);
            __m128i q8s_2 = _mm_maddubs_epi16(q3h_2, q8_2);
            __m128i q8s_3 = _mm_maddubs_epi16(q3h_3, q8_3);
            __m128i q8s_4 = _mm_maddubs_epi16(q3h_4, q8_4);
            __m128i q8s_5 = _mm_maddubs_epi16(q3h_5, q8_5);
            __m128i q8s_6 = _mm_maddubs_epi16(q3h_6, q8_6);
            __m128i q8s_7 = _mm_maddubs_epi16(q3h_7, q8_7);

            __m128i p16_0 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_0, q8_0), q8s_0);
            __m128i p16_1 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_1, q8_1), q8s_1);
            __m128i p16_2 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_2, q8_2), q8s_2);
            __m128i p16_3 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_3, q8_3), q8s_3);
            __m128i p16_4 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_4, q8_4), q8s_4);
            __m128i p16_5 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_5, q8_5), q8s_5);
            __m128i p16_6 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_6, q8_6), q8s_6);
            __m128i p16_7 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_7, q8_7), q8s_7);

            __m128i shuffle = _mm_set1_epi16(0x0100);
            p16_0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_0);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_1);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_2);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_3);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_4);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_5);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_6);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_7);

            p16_0 = _mm_add_epi32(p16_0, p16_1);
            p16_2 = _mm_add_epi32(p16_2, p16_3);
            p16_4 = _mm_add_epi32(p16_4, p16_5);
            p16_6 = _mm_add_epi32(p16_6, p16_7);
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_4, p16_6));
        }
        __m256i sumi = _mm256_insertf128_si256(_mm256_castsi128_si256(sumi_1), sumi_0, 1);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi)), acc);
    }
    return hsum_avx(acc);

#elif defined(PICOLM_NEON)
    {
    const uint8x16_t m3 = vdupq_n_u8(3);
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
        const uint8_t *q3 = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        const int8_t  *q8 = y[i].qs;
        uint32_t auxs[4];
        memcpy(auxs, x[i].scales, 12);
        { uint32_t tmp = auxs[2];
          auxs[2] = ((auxs[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
          auxs[3] = ((auxs[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
          auxs[0] = (auxs[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4);
          auxs[1] = (auxs[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4); }
        const int8_t *scales = (const int8_t *)auxs;
        int32_t sub_sums[16] = {0};
        int is = 0;
        uint16_t hmv = 1;
        for (int chunk = 0; chunk < 2; chunk++) {
            const uint8x16_t q3a_raw = vld1q_u8(q3);
            const uint8x16_t q3b_raw = vld1q_u8(q3 + 16);
            /* vshrq_n_u8 requires compile-time constant shift; accumulate by 2 each iteration */
            uint8x16_t q3a_shifted = q3a_raw, q3b_shifted = q3b_raw;
            for (int j = 0; j < 4; j++) {
                uint8x16_t q3l_a = vandq_u8(q3a_shifted, m3);
                uint8x16_t q3l_b = vandq_u8(q3b_shifted, m3);
                q3a_shifted = vshrq_n_u8(q3a_shifted, 2);
                q3b_shifted = vshrq_n_u8(q3b_shifted, 2);
                uint8_t hm_sub_a[16], hm_sub_b[16];
                for (int l = 0; l < 16; l++) {
                    hm_sub_a[l] = (hm[l] & hmv) ? 0 : 4;
                    hm_sub_b[l] = (hm[l+16] & hmv) ? 0 : 4;
                }
                const int8x16_t q3v_a = vreinterpretq_s8_u8(vsubq_u8(q3l_a, vld1q_u8(hm_sub_a)));
                const int8x16_t q3v_b = vreinterpretq_s8_u8(vsubq_u8(q3l_b, vld1q_u8(hm_sub_b)));
                {
                    const int8x16_t q8a = vld1q_s8(q8); q8 += 16;
                    int16x8_t p0 = vmull_s8(vget_low_s8(q3v_a), vget_low_s8(q8a));
                    int16x8_t p1 = vmull_s8(vget_high_s8(q3v_a), vget_high_s8(q8a));
                    int32x4_t s = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
                    int scale = scales[is] - 32;
                    sub_sums[is++] += vaddvq_s32(s) * scale;
                }
                {
                    const int8x16_t q8b = vld1q_s8(q8); q8 += 16;
                    int16x8_t p0 = vmull_s8(vget_low_s8(q3v_b), vget_low_s8(q8b));
                    int16x8_t p1 = vmull_s8(vget_high_s8(q3v_b), vget_high_s8(q8b));
                    int32x4_t s = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
                    int scale = scales[is] - 32;
                    sub_sums[is++] += vaddvq_s32(s) * scale;
                }
                hmv <<= 1;
            }
            q3 += 32;
        }
        int32_t total = 0;
        for (int j = 0; j < 16; j++) total += sub_sums[j];
        sumf += d * (float)total;
    }
    return sumf;
}

#else
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
        const uint8_t *q3 = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        const int8_t  *q8 = y[i].qs;
        uint32_t auxs[4];
        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
        auxs[3] = ((auxs[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
        auxs[0] = (auxs[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4);
        auxs[1] = (auxs[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4);
        const int8_t *scales = (const int8_t *)auxs;
        int is = 0;
        uint16_t hmv = 1;
        for (int chunk = 0; chunk < 2; chunk++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int8_t v = (int8_t)(((q3[l] >> shift) & 3) - ((hm[l] & hmv) ? 0 : 4));
                    sumf += dl * (float)v * q8[l];
                }
                float dl2 = d * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int8_t v = (int8_t)(((q3[l+16] >> shift) & 3) - ((hm[l+16] & hmv) ? 0 : 4));
                    sumf += dl2 * (float)v * q8[16+l];
                }
                shift += 2;
                hmv <<= 1;
                q8 += 32;
            }
            q3 += 32;
        }
    }
    return sumf;
#endif
}





/* ================================================================
 * vec_dot_q4_0_q8_0: Q4_0 weights x Q8_0 input (int8 MAC)
 * Adapted from llama.cpp ggml_vec_dot_q4_0_q8_0.
 *
 * Q4_0: 16 bytes qs + 2 bytes d(FP16) per 32 values.
 * Q8_0: 32 bytes qs + 2 bytes d(FP16) per 32 values.
 * Both use block size 32.
 * ================================================================ */
float vec_dot_q4_0_q8_0(const void *vx, const void *wy, int n) {
    const block_q4_0 *x = (const block_q4_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)wy;
    int nb = n / 32;
    int ib = 0;
    float sumf = 0;

#if defined(PICOLM_AVX2)
    {
        __m256 acc = _mm256_setzero_ps();
        const __m256i off = _mm256_set1_epi8(8);

        for (; ib < nb; ++ib) {
            const __m256 d = _mm256_set1_ps(
                fp16_to_fp32_lookup(x[ib].d) * fp16_to_fp32_lookup(y[ib].d));
            __m256i qx = _mm256_sub_epi8(bytes_from_nibbles_32(x[ib].qs), off);
            __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
            acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float(qx, qy), acc);
        }
        sumf = hsum_avx(acc);
    }

#elif defined(PICOLM_AVX)
    {
        __m256 accum = _mm256_setzero_ps();
        const __m128i mask4 = _mm_set1_epi8(15);
        const __m128i off = _mm_set1_epi8(8);

        for (; ib + 1 < nb; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs);
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs + 1);
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

            const __m128i q4b_1_0 = _mm_sub_epi8(_mm_and_si128(mask4, q4bits_1), off);
            const __m128i q4b_1_1 = _mm_sub_epi8(_mm_and_si128(mask4, _mm_srli_epi16(q4bits_1, 4)), off);
            const __m128i q4b_2_0 = _mm_sub_epi8(_mm_and_si128(mask4, q4bits_2), off);
            const __m128i q4b_2_1 = _mm_sub_epi8(_mm_and_si128(mask4, _mm_srli_epi16(q4bits_2, 4)), off);

            const __m128i p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
            const __m128i p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
            const __m128i p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
            const __m128i p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
            const __m128i p_1 = _mm_add_epi16(p16_1_0, p16_1_1);
            const __m128i p_2 = _mm_add_epi16(p16_2_0, p16_2_1);

            __m256 p = sum_i16_pairs_float(p_2, p_1);
            float d0 = fp16_to_fp32_lookup(x[ib].d) * fp16_to_fp32_lookup(y[ib].d);
            float d1 = fp16_to_fp32_lookup(x[ib + 1].d) * fp16_to_fp32_lookup(y[ib + 1].d);
            __m256 deltas = _mm256_set_m128(_mm_set1_ps(d1), _mm_set1_ps(d0));
            accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
        }
        sumf = hsum_avx(accum);
    }

#elif defined(PICOLM_I8MM)
    /* Fused I8MM: vmmlaq_s32 for int8 x int8 -> int32 MAC,
     * with post-processing (scvtf + scale multiply) fused in NEON.
     *
     * Strategy: accumulate int32 dot products in float32x4 accumulators
     * per block, using vmlaq_n_f32 to fuse the scale multiply.
     * This mirrors MNN's fused epilogue: int32 -> scvtf -> fmul(scale)
     * stays entirely in SIMD registers.
     *
     * vmmlaq_s32 on 16 int8 x 16 int8 produces 4 int32 lanes:
     *   lane 0 = dot of first 8 pairs, lane 3 = dot of last 8 pairs.
     * After low+high vmmlaq vadd: lane 0 + lane 3 = 32-element dot.
     * We use vaddvq_s32 for the horizontal sum (all 4 lanes) since
     * lanes 1+2 are zero after our two-vaddq pattern. */
    {
        const uint8x16_t mask4 = vdupq_n_u8(0x0F);
        const int8x16_t offset8 = vdupq_n_s8(8);
        float32x4_t acc = vdupq_n_f32(0.0f);

        for (; ib + 3 < nb; ib += 4) {
            /* Process 4 blocks per iteration */
            float scales[4];
            int32x4_t sums[4];
            for (int k = 0; k < 4; k++) {
                int ki = ib + k;
                const uint8x16_t qx4k = vld1q_u8(x[ki].qs);
                const int8x16_t qy0k = vld1q_s8(y[ki].qs);
                const int8x16_t qy1k = vld1q_s8(y[ki].qs + 16);
                int8x16_t qx_lo_k = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qx4k, mask4)), offset8);
                int8x16_t qx_hi_k = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qx4k, 4)), offset8);
                sums[k] = vaddq_s32(vmmlaq_s32(vdupq_n_s32(0), qx_lo_k, qy0k),
                                     vmmlaq_s32(vdupq_n_s32(0), qx_hi_k, qy1k));
                scales[k] = fp16_to_fp32_lookup(x[ki].d) * fp16_to_fp32_lookup(y[ki].d);
            }
            /* Fused post-processing: scvtf + fmul(scale) per block, accumulate in float32x4 */
            float di = scales[0];
            float dj = scales[1];
            float dk = scales[2];
            float dl = scales[3];
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(sums[0]), di);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(sums[1]), dj);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(sums[2]), dk);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(sums[3]), dl);
        }
        sumf = vaddvq_f32(acc);

        /* Tail: 1-3 blocks */
        for (; ib + 1 < nb; ib += 2) {
            const uint8x16_t qx4 = vld1q_u8(x[ib].qs);
            const int8x16_t qy0 = vld1q_s8(y[ib].qs);
            const int8x16_t qy1 = vld1q_s8(y[ib].qs + 16);
            int8x16_t qx_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qx4, mask4)), offset8);
            int8x16_t qx_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qx4, 4)), offset8);
            int32x4_t s = vaddq_s32(vmmlaq_s32(vdupq_n_s32(0), qx_lo, qy0),
                                     vmmlaq_s32(vdupq_n_s32(0), qx_hi, qy1));
            float di = fp16_to_fp32_lookup(x[ib].d) * fp16_to_fp32_lookup(y[ib].d);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(s), di);

            const uint8x16_t qx4b = vld1q_u8(x[ib+1].qs);
            const int8x16_t qyb0 = vld1q_s8(y[ib+1].qs);
            const int8x16_t qyb1 = vld1q_s8(y[ib+1].qs + 16);
            int8x16_t qx_lo_b = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qx4b, mask4)), offset8);
            int8x16_t qx_hi_b = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qx4b, 4)), offset8);
            int32x4_t sb = vaddq_s32(vmmlaq_s32(vdupq_n_s32(0), qx_lo_b, qyb0),
                                      vmmlaq_s32(vdupq_n_s32(0), qx_hi_b, qyb1));
            float dj = fp16_to_fp32_lookup(x[ib+1].d) * fp16_to_fp32_lookup(y[ib+1].d);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(sb), dj);
        }
        sumf += vaddvq_f32(acc);

        /* Final single block */
        for (; ib < nb; ib++) {
            const uint8x16_t qx4 = vld1q_u8(x[ib].qs);
            const int8x16_t qy0 = vld1q_s8(y[ib].qs);
            const int8x16_t qy1 = vld1q_s8(y[ib].qs + 16);
            int8x16_t qx_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qx4, mask4)), offset8);
            int8x16_t qx_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qx4, 4)), offset8);
            int32x4_t s = vaddq_s32(vmmlaq_s32(vdupq_n_s32(0), qx_lo, qy0),
                                     vmmlaq_s32(vdupq_n_s32(0), qx_hi, qy1));
            float di = fp16_to_fp32_lookup(x[ib].d) * fp16_to_fp32_lookup(y[ib].d);
            int32_t dot_i = vgetq_lane_s32(s, 0) + vgetq_lane_s32(s, 3);
            sumf += (float)dot_i * di;
        }
    }

#elif defined(PICOLM_NEON)
    /* Plain NEON: vmull_s8 + vpaddlq_s16.
     * Two blocks per iteration for better throughput. */
    {
        const uint8x16_t mask4 = vdupq_n_u8(0x0F);
        const int8x16_t off = vdupq_n_s8(8);

        for (; ib + 1 < nb; ib += 2) {
            float d0 = fp16_to_fp32_lookup(x[ib + 0].d) * fp16_to_fp32_lookup(y[ib + 0].d);
            float d1 = fp16_to_fp32_lookup(x[ib + 1].d) * fp16_to_fp32_lookup(y[ib + 1].d);

            /* Block ib */
            {
                const uint8x16_t qx = vld1q_u8(x[ib].qs);
                const int8x16_t qy0 = vld1q_s8(y[ib].qs);
                const int8x16_t qy1 = vld1q_s8(y[ib].qs + 16);
                int8x16_t qxl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qx, mask4)), off);
                int8x16_t qxh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qx, 4)), off);
                int16x8_t p0 = vmull_s8(vget_low_s8(qxl), vget_low_s8(qy0));
                int16x8_t p1 = vmull_s8(vget_high_s8(qxl), vget_high_s8(qy0));
                int16x8_t p2 = vmull_s8(vget_low_s8(qxh), vget_low_s8(qy1));
                int16x8_t p3 = vmull_s8(vget_high_s8(qxh), vget_high_s8(qy1));
                int32x4_t s = vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
                                        vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)));
                sumf += d0 * (float)vaddvq_s32(s);
            }
            /* Block ib+1 */
            {
                const uint8x16_t qx = vld1q_u8(x[ib + 1].qs);
                const int8x16_t qy0 = vld1q_s8(y[ib + 1].qs);
                const int8x16_t qy1 = vld1q_s8(y[ib + 1].qs + 16);
                int8x16_t qxl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qx, mask4)), off);
                int8x16_t qxh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qx, 4)), off);
                int16x8_t p0 = vmull_s8(vget_low_s8(qxl), vget_low_s8(qy0));
                int16x8_t p1 = vmull_s8(vget_high_s8(qxl), vget_high_s8(qy0));
                int16x8_t p2 = vmull_s8(vget_low_s8(qxh), vget_low_s8(qy1));
                int16x8_t p3 = vmull_s8(vget_high_s8(qxh), vget_high_s8(qy1));
                int32x4_t s = vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
                                        vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)));
                sumf += d1 * (float)vaddvq_s32(s);
            }
        }
        /* Tail: single block */
        for (; ib < nb; ib++) {
            float d0 = fp16_to_fp32_lookup(x[ib].d) * fp16_to_fp32_lookup(y[ib].d);
            const uint8x16_t qx = vld1q_u8(x[ib].qs);
            const int8x16_t qy0 = vld1q_s8(y[ib].qs);
            const int8x16_t qy1 = vld1q_s8(y[ib].qs + 16);
            int8x16_t qxl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qx, mask4)), off);
            int8x16_t qxh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qx, 4)), off);
            int16x8_t p0 = vmull_s8(vget_low_s8(qxl), vget_low_s8(qy0));
            int16x8_t p1 = vmull_s8(vget_high_s8(qxl), vget_high_s8(qy0));
            int16x8_t p2 = vmull_s8(vget_low_s8(qxh), vget_low_s8(qy1));
            int16x8_t p3 = vmull_s8(vget_high_s8(qxh), vget_high_s8(qy1));
            int32x4_t s = vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
                                    vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)));
            sumf += d0 * (float)vaddvq_s32(s);
        }
    }

#elif defined(PICOLM_SSSE3)
    {
        const __m128i mask4 = _mm_set1_epi8(15);
        const __m128i off = _mm_set1_epi8(8);
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();
        __m128 acc2 = _mm_setzero_ps();
        __m128 acc3 = _mm_setzero_ps();

        for (; ib + 1 < nb; ib += 2) {
            const __m128 d_0_1 = _mm_set1_ps(
                fp16_to_fp32_lookup(x[ib].d) * fp16_to_fp32_lookup(y[ib].d));
            const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i *)x[ib].qs);
            __m128i bx_0 = _mm_sub_epi8(_mm_and_si128(mask4, tmp_0_1), off);
            const __m128i i32_0 = mul_sum_i8_pairs_sse(bx_0, _mm_loadu_si128((const __m128i *)y[ib].qs));
            __m128i bx_1 = _mm_sub_epi8(_mm_and_si128(mask4, _mm_srli_epi64(tmp_0_1, 4)), off);
            const __m128i i32_1 = mul_sum_i8_pairs_sse(bx_1, _mm_loadu_si128((const __m128i *)(y[ib].qs + 16)));

            const __m128 d_2_3 = _mm_set1_ps(
                fp16_to_fp32_lookup(x[ib + 1].d) * fp16_to_fp32_lookup(y[ib + 1].d));
            const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
            __m128i bx_2 = _mm_sub_epi8(_mm_and_si128(mask4, tmp_2_3), off);
            const __m128i i32_2 = mul_sum_i8_pairs_sse(bx_2, _mm_loadu_si128((const __m128i *)y[ib + 1].qs));
            __m128i bx_3 = _mm_sub_epi8(_mm_and_si128(mask4, _mm_srli_epi64(tmp_2_3, 4)), off);
            const __m128i i32_3 = mul_sum_i8_pairs_sse(bx_3, _mm_loadu_si128((const __m128i *)(y[ib + 1].qs + 16)));

            acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_cvtepi32_ps(i32_0), d_0_1));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_cvtepi32_ps(i32_1), d_0_1));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_cvtepi32_ps(i32_2), d_2_3));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_cvtepi32_ps(i32_3), d_2_3));
        }
        sumf = hsum_sse(_mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3)));
    }

#else
    /* Scalar fallback (same as llama.cpp generic path) */
#endif

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;
        for (int j = 0; j < 16; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >> 4) - 8;
            sumi0 += v0 * y[ib].qs[j];
            sumi1 += v1 * y[ib].qs[j + 16];
        }
        sumf += (float)(sumi0 + sumi1) * fp16_to_fp32_lookup(x[ib].d) * fp16_to_fp32_lookup(y[ib].d);
    }

    return sumf;
}

void q4_0_row_to_q8_0_shadow(const void *q4_row, void *q8_row_out, int n) {
    const block_q4_0 *q4 = (const block_q4_0 *)q4_row;
    block_q8_0 *q8 = (block_q8_0 *)q8_row_out;
    int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        q8[b].d = q4[b].d; /* same fp16 scale bits -- no conversion needed */
        for (int j = 0; j < 16; j++) {
            uint8_t byte = q4[b].qs[j];
            q8[b].qs[j]      = (int8_t)(byte & 0x0F) - 8;
            q8[b].qs[j + 16] = (int8_t)(byte >> 4)    - 8;
        }
    }
}

float vec_dot_q8_0_q8_0(const void *qx, const void *qw, int n) {
    const block_q8_0 *x = (const block_q8_0 *)qx;
    const block_q8_0 *w = (const block_q8_0 *)qw;
    int nb = n / 32;
    float sumf = 0.0f;
    int i = 0;
#if !defined(PICOLM_VNNI) && !defined(PICOLM_I8MM) && !defined(PICOLM_NEON) && !defined(PICOLM_AVX2) && !defined(PICOLM_AVX) && !defined(PICOLM_SSSE3)
    /* Pure scalar path: no SIMD available */
    for (i = 0; i < nb; i++) {
        int sumi = 0;
        for (int j = 0; j < 32; j++) sumi += x[i].qs[j] * w[i].qs[j];
        sumf += (float)sumi * fp16_to_fp32_lookup(x[i].d) * fp16_to_fp32_lookup(w[i].d);
    }
    return sumf;
#endif

#ifdef PICOLM_VNNI
    /* AVX-512 VNNI: dpbusd + sign trick for signed int8 MAC.
     * dpbusd(zero, abs(x), sign(y,x)) = sum(x*y) for signed int8.
     * Server CPUs have AVX512-VNNI (not client AVX-VNNI),
     * so we use 256-bit dpbusd via __AVX512VL__. */
    __m256 acc = _mm256_setzero_ps();
    for (i = 0; i + 1 < nb; i += 2) {
        __m256i qx0 = _mm256_loadu_si256((const __m256i *)x[i].qs);
        __m256i qw0 = _mm256_loadu_si256((const __m256i *)w[i].qs);
        __m256i qx1 = _mm256_loadu_si256((const __m256i *)x[i+1].qs);
        __m256i qw1 = _mm256_loadu_si256((const __m256i *)w[i+1].qs);

        __m256i ax0 = _mm256_sign_epi8(qx0, qx0);
        __m256i sx0 = _mm256_sign_epi8(qw0, qx0);
        __m256i s0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax0, sx0);

        __m256i ax1 = _mm256_sign_epi8(qx1, qx1);
        __m256i sx1 = _mm256_sign_epi8(qw1, qx1);
        __m256i s1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax1, sx1);

        __m256 f0 = _mm256_cvtepi32_ps(s0);
        __m256 f1 = _mm256_cvtepi32_ps(s1);

        float d0 = fp16_to_fp32_lookup(x[i].d) * fp16_to_fp32_lookup(w[i].d);
        float d1 = fp16_to_fp32_lookup(x[i+1].d) * fp16_to_fp32_lookup(w[i+1].d);
        __m256 dd0 = _mm256_set1_ps(d0);
        __m256 dd1 = _mm256_set1_ps(d1);

        acc = _mm256_add_ps(_mm256_mul_ps(f0, dd0), _mm256_add_ps(_mm256_mul_ps(f1, dd1), acc));
    }
    sumf = hsum_avx(acc);

#elif defined(PICOLM_I8MM)
    /* Fused I8MM: vmmlaq_s32 for int8 x int8 -> int32 MAC,
     * with scvtf + scale multiply fused in NEON (MNN-style epilogue).
     *
     * vmmlaq_s32 on 32 int8 x int8 (split as 16+16) produces:
     *   lane 0 = dot(8 pairs), lane 3 = dot(8 pairs) for each half.
     * After vaddq of two vmmlaq: lane 0 = dot(16), lane 3 = dot(16).
     * Total dot = lane 0 + lane 3.
     *
     * We accumulate in float32x4 using vmlaq_n_f32 to fuse scvtf+fmul,
     * processing 4 blocks per iteration for register efficiency. */
    {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (i = 0; i + 3 < nb; i += 4) {
            int32x4_t sums[4];
            float scales[4];
            for (int k = 0; k < 4; k++) {
                int ki = i + k;
                const int8x16_t xi0 = vld1q_s8(x[ki].qs);
                const int8x16_t xi1 = vld1q_s8(x[ki].qs + 16);
                const int8x16_t wi0 = vld1q_s8(w[ki].qs);
                const int8x16_t wi1 = vld1q_s8(w[ki].qs + 16);
                sums[k] = vaddq_s32(vmmlaq_s32(vdupq_n_s32(0), xi0, wi0),
                                     vmmlaq_s32(vdupq_n_s32(0), xi1, wi1));
                scales[k] = fp16_to_fp32_lookup(x[ki].d) * fp16_to_fp32_lookup(w[ki].d);
            }
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(sums[0]), scales[0]);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(sums[1]), scales[1]);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(sums[2]), scales[2]);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(sums[3]), scales[3]);
        }
        sumf = vaddvq_f32(acc);

        /* Tail: 1-3 blocks via scalar extraction */
        for (; i + 1 < nb; i += 2) {
            const int8x16_t xi_0 = vld1q_s8(x[i].qs);
            const int8x16_t xi_1 = vld1q_s8(x[i].qs + 16);
            const int8x16_t wi_0 = vld1q_s8(w[i].qs);
            const int8x16_t wi_1 = vld1q_s8(w[i].qs + 16);
            int32x4_t si = vaddq_s32(vmmlaq_s32(vdupq_n_s32(0), xi_0, wi_0),
                                      vmmlaq_s32(vdupq_n_s32(0), xi_1, wi_1));
            int32_t dot_i = vgetq_lane_s32(si, 0) + vgetq_lane_s32(si, 3);
            sumf += (float)dot_i * fp16_to_fp32_lookup(x[i].d) * fp16_to_fp32_lookup(w[i].d);

            const int8x16_t xj_0 = vld1q_s8(x[i+1].qs);
            const int8x16_t xj_1 = vld1q_s8(x[i+1].qs + 16);
            const int8x16_t wj_0 = vld1q_s8(w[i+1].qs);
            const int8x16_t wj_1 = vld1q_s8(w[i+1].qs + 16);
            int32x4_t sj = vaddq_s32(vmmlaq_s32(vdupq_n_s32(0), xj_0, wj_0),
                                      vmmlaq_s32(vdupq_n_s32(0), xj_1, wj_1));
            int32_t dot_j = vgetq_lane_s32(sj, 0) + vgetq_lane_s32(sj, 3);
            sumf += (float)dot_j * fp16_to_fp32_lookup(x[i+1].d) * fp16_to_fp32_lookup(w[i+1].d);
        }
        for (; i < nb; i++) {
            const int8x16_t xi_0 = vld1q_s8(x[i].qs);
            const int8x16_t xi_1 = vld1q_s8(x[i].qs + 16);
            const int8x16_t wi_0 = vld1q_s8(w[i].qs);
            const int8x16_t wi_1 = vld1q_s8(w[i].qs + 16);
            int32x4_t si = vaddq_s32(vmmlaq_s32(vdupq_n_s32(0), xi_0, wi_0),
                                      vmmlaq_s32(vdupq_n_s32(0), xi_1, wi_1));
            int32_t dot_i = vgetq_lane_s32(si, 0) + vgetq_lane_s32(si, 3);
            sumf += (float)dot_i * fp16_to_fp32_lookup(x[i].d) * fp16_to_fp32_lookup(w[i].d);
        }
    }

#elif defined(PICOLM_NEON)
    /* NEON: optimized int8 MAC via vpaddlq_s16, 2 blocks/iter (mirrors llama.cpp) */
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);
    for (i = 0; i + 1 < nb; i += 2) {
        const int8x16_t x0_0 = vld1q_s8(x[i].qs);
        const int8x16_t x0_1 = vld1q_s8(x[i].qs + 16);
        const int8x16_t w0_0 = vld1q_s8(w[i].qs);
        const int8x16_t w0_1 = vld1q_s8(w[i].qs + 16);
        {
            const int16x8_t p0 = vmull_s8(vget_low_s8(x0_0), vget_low_s8(w0_0));
            const int16x8_t p1 = vmull_s8(vget_high_s8(x0_0), vget_high_s8(w0_0));
            const int16x8_t p2 = vmull_s8(vget_low_s8(x0_1), vget_low_s8(w0_1));
            const int16x8_t p3 = vmull_s8(vget_high_s8(x0_1), vget_high_s8(w0_1));
            const int32x4_t s = vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
                                          vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)));
            const float d = fp16_to_fp32_lookup(x[i].d) * fp16_to_fp32_lookup(w[i].d);
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(s), d);
        }
        const int8x16_t x1_0 = vld1q_s8(x[i+1].qs);
        const int8x16_t x1_1 = vld1q_s8(x[i+1].qs + 16);
        const int8x16_t w1_0 = vld1q_s8(w[i+1].qs);
        const int8x16_t w1_1 = vld1q_s8(w[i+1].qs + 16);
        {
            const int16x8_t p0 = vmull_s8(vget_low_s8(x1_0), vget_low_s8(w1_0));
            const int16x8_t p1 = vmull_s8(vget_high_s8(x1_0), vget_high_s8(w1_0));
            const int16x8_t p2 = vmull_s8(vget_low_s8(x1_1), vget_low_s8(w1_1));
            const int16x8_t p3 = vmull_s8(vget_high_s8(x1_1), vget_high_s8(w1_1));
            const int32x4_t s = vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
                                          vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)));
            const float d = fp16_to_fp32_lookup(x[i+1].d) * fp16_to_fp32_lookup(w[i+1].d);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(s), d);
        }
    }
    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

#elif defined(PICOLM_AVX2)
    /* AVX2: load 32 int8 with _mm256_loadu_si256, 256-bit maddubs */
    __m256 acc = _mm256_setzero_ps();
    for (i = 0; i + 1 < nb; i += 2) {
        __m256i qx0 = _mm256_loadu_si256((const __m256i *)x[i].qs);
        __m256i qx1 = _mm256_loadu_si256((const __m256i *)x[i + 1].qs);
        __m256i qw0 = _mm256_loadu_si256((const __m256i *)w[i].qs);
        __m256i qw1 = _mm256_loadu_si256((const __m256i *)w[i + 1].qs);

        __m256i s0 = mul_sum_i8_pairs_avx2(qx0, qw0);
        __m256i s1 = mul_sum_i8_pairs_avx2(qx1, qw1);

        __m256 f0 = _mm256_cvtepi32_ps(s0);
        __m256 f1 = _mm256_cvtepi32_ps(s1);

        float d0 = fp16_to_fp32_lookup(x[i].d) * fp16_to_fp32_lookup(w[i].d);
        float d1 = fp16_to_fp32_lookup(x[i + 1].d) * fp16_to_fp32_lookup(w[i + 1].d);
        __m256 dd0 = _mm256_set1_ps(d0);
        __m256 dd1 = _mm256_set1_ps(d1);

        acc = _mm256_fmadd_ps(f0, dd0, _mm256_fmadd_ps(f1, dd1, acc));
    }
    sumf = hsum_avx(acc);

#elif defined(PICOLM_AVX)
    /* AVX (no AVX2): SSE4.1 maddubs_epi16, 256-bit float accum */
    /* Process 2 blocks per iteration for instruction-level parallelism */
    __m256 acc = _mm256_setzero_ps();
    for (i = 0; i + 1 < nb; i += 2) {
        __m128i qx0 = _mm_loadu_si128((const __m128i *)x[i].qs);
        __m128i qx1 = _mm_loadu_si128((const __m128i *)x[i].qs + 1);
        __m128i qw0 = _mm_loadu_si128((const __m128i *)w[i].qs);
        __m128i qw1 = _mm_loadu_si128((const __m128i *)w[i].qs + 1);
        __m128i qx2 = _mm_loadu_si128((const __m128i *)x[i + 1].qs);
        __m128i qx3 = _mm_loadu_si128((const __m128i *)x[i + 1].qs + 1);
        __m128i qw2 = _mm_loadu_si128((const __m128i *)w[i + 1].qs);
        __m128i qw3 = _mm_loadu_si128((const __m128i *)w[i + 1].qs + 1);

        __m128i p0 = mul_sum_i8_pairs_sse(qx0, qw0);
        __m128i p1 = mul_sum_i8_pairs_sse(qx1, qw1);
        __m128i p2 = mul_sum_i8_pairs_sse(qx2, qw2);
        __m128i p3 = mul_sum_i8_pairs_sse(qx3, qw3);

        __m128i sum0 = _mm_add_epi32(p0, p1);
        __m128i sum1 = _mm_add_epi32(p2, p3);
        __m256 sums = _mm256_cvtepi32_ps(_mm256_set_m128i(sum1, sum0));

        float d0 = fp16_to_fp32_lookup(x[i].d) * fp16_to_fp32_lookup(w[i].d);
        float d1 = fp16_to_fp32_lookup(x[i + 1].d) * fp16_to_fp32_lookup(w[i + 1].d);
        __m256 deltas = _mm256_set_m128(_mm_set1_ps(d1), _mm_set1_ps(d0));
        acc = _mm256_add_ps(_mm256_mul_ps(deltas, sums), acc);
    }
    sumf = hsum_avx(acc);

#elif defined(PICOLM_SSSE3)
    /* SSSE3: process 2 blocks per iteration with 2 accumulators */
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    for (i = 0; i + 1 < nb; i += 2) {
        /* Block i */
        __m128i qx0 = _mm_loadu_si128((const __m128i *)x[i].qs);
        __m128i qx1 = _mm_loadu_si128((const __m128i *)x[i].qs + 1);
        __m128i qw0 = _mm_loadu_si128((const __m128i *)w[i].qs);
        __m128i qw1 = _mm_loadu_si128((const __m128i *)w[i].qs + 1);
        __m128i s0 = _mm_add_epi32(mul_sum_i8_pairs_sse(qx0, qw0),
                                    mul_sum_i8_pairs_sse(qx1, qw1));
        float d0 = fp16_to_fp32_lookup(x[i].d) * fp16_to_fp32_lookup(w[i].d);
        __m128 dd0 = _mm_set1_ps(d0);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_cvtepi32_ps(s0), dd0));

        /* Block i+1 */
        __m128i qx2 = _mm_loadu_si128((const __m128i *)x[i + 1].qs);
        __m128i qx3 = _mm_loadu_si128((const __m128i *)x[i + 1].qs + 1);
        __m128i qw2 = _mm_loadu_si128((const __m128i *)w[i + 1].qs);
        __m128i qw3 = _mm_loadu_si128((const __m128i *)w[i + 1].qs + 1);
        __m128i s1 = _mm_add_epi32(mul_sum_i8_pairs_sse(qx2, qw2),
                                    mul_sum_i8_pairs_sse(qx3, qw3));
        float d1 = fp16_to_fp32_lookup(x[i + 1].d) * fp16_to_fp32_lookup(w[i + 1].d);
        __m128 dd1 = _mm_set1_ps(d1);
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_cvtepi32_ps(s1), dd1));
    }
    sumf = hsum_sse(_mm_add_ps(acc0, acc1));
#endif

    /* Scalar tail for remaining blocks */
    for (; i < nb; i++) {
        int sumi = 0;
        for (int j = 0; j < 32; j++) {
            sumi += x[i].qs[j] * w[i].qs[j];
        }
        sumf += (float)sumi * fp16_to_fp32_lookup(x[i].d) * fp16_to_fp32_lookup(w[i].d);
    }
    return sumf;
}

/* ================================================================
 * vec_dot_q4_0x4_q8_0: Q4_0_4_4 interleaved weights x Q8_0 input
 *
 * Processes 4 rows simultaneously from the interleaved Q4_0_4_4 layout.
 * Adapted from llama.cpp ggml_gemv_q4_0_4x4_q8_0_generic.
 *
 * Layout: block_q4_0x4 with d[4] (FP16) + qs[64] (interleaved nibbles)
 * Interleaving (blocklen=4): for each k=0..3:
 *   qs[k*16 + 0..3]   = row0 nibble-bytes at offset k*4..k*4+3
 *   qs[k*16 + 4..7]   = row1 nibble-bytes at offset k*4..k*4+3
 *   qs[k*16 + 8..11]  = row2 nibble-bytes at offset k*4..k*4+3
 *   qs[k*16 + 12..15] = row3 nibble-bytes at offset k*4..k*4+3
 *
 * Nibbles are XOR'd with 0x88 during repacking. Extraction uses:
 *   v0 = (int8_t)(byte << 4) >> 4  -- sign-extends low nibble
 *   v1 = (int8_t)(byte & 0xF0) >> 4  -- extracts signed high nibble
 *
 * Q8_0 input (y): k*4+0..3 -> first half, k*4+0..3 + 16 -> second half.
 * ================================================================ */
void vec_dot_q4_0x4_q8_0(const void *vx, const void *wy, int n, float *out, int nrows) {
    const block_q4_0x4 *xb = (const block_q4_0x4 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)wy;
    int nb = n / 32;  /* blocks per row */

#if defined(PICOLM_DOTPROD)
    /* Ported from ggml's ggml_gemv_q4_0_4x4_q8_0 (ggml-cpu/arch/arm/repack.cpp).
     * vdotq_laneq_s32 computes a 4-way signed dot-product-accumulate in one
     * instruction -- no shuffle/blend choreography needed, unlike the AVX2
     * LUT approach above. The `<<4` / `&0xf0U` trick isolates each nibble
     * into the top of its byte; interpreted as signed int8 that's exactly
     * 16*sign_extend_4bit(nibble), and since the repack step already XORs
     * every nibble with 8, sign_extend_4bit(nibble^8) == nibble-8 for all
     * 16 values (same identity verified for the AVX2 LUT, just via shift
     * instead of table lookup) -- confirmed both algebraically and by
     * running this exact kernel under QEMU's cortex-a76 dotprod emulation
     * against the scalar reference before shipping it, since there's no
     * real dotprod hardware available to test on directly. */
    for (int row_group = 0; row_group < nrows; row_group += 4) {
        const block_q4_0x4 *b_ptr = xb + (size_t)(row_group / 4) * (size_t)nb;
        const block_q8_0 *a_ptr = y;
        float32x4_t acc = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            int8x16_t b0 = vld1q_s8((const int8_t *)b_ptr->qs);
            int8x16_t b1 = vld1q_s8((const int8_t *)b_ptr->qs + 16);
            int8x16_t b2 = vld1q_s8((const int8_t *)b_ptr->qs + 32);
            int8x16_t b3 = vld1q_s8((const int8_t *)b_ptr->qs + 48);
            float16x4_t bd = vld1_f16((const __fp16 *)b_ptr->d);

            int8x16_t a0 = vld1q_s8(a_ptr->qs);
            int8x16_t a1 = vld1q_s8(a_ptr->qs + 16);
            float16x4_t ad = vld1_dup_f16((const __fp16 *)&a_ptr->d);

            int32x4_t ret = vdupq_n_s32(0);
            ret = vdotq_laneq_s32(ret, vshlq_n_s8(b0, 4), a0, 0);
            ret = vdotq_laneq_s32(ret, vshlq_n_s8(b1, 4), a0, 1);
            ret = vdotq_laneq_s32(ret, vshlq_n_s8(b2, 4), a0, 2);
            ret = vdotq_laneq_s32(ret, vshlq_n_s8(b3, 4), a0, 3);
            ret = vdotq_laneq_s32(ret, vandq_s8(b0, vdupq_n_s8((int8_t)0xf0U)), a1, 0);
            ret = vdotq_laneq_s32(ret, vandq_s8(b1, vdupq_n_s8((int8_t)0xf0U)), a1, 1);
            ret = vdotq_laneq_s32(ret, vandq_s8(b2, vdupq_n_s8((int8_t)0xf0U)), a1, 2);
            ret = vdotq_laneq_s32(ret, vandq_s8(b3, vdupq_n_s8((int8_t)0xf0U)), a1, 3);

            acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                             vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
            a_ptr++;
            b_ptr++;
        }
        vst1q_f32(out + row_group, acc);
    }
    return;
#endif

/* Scalar: process each row individually from interleaved data.
     * Adapted from llama.cpp ggml_gemv_q4_0_4x4_q8_0_generic.
     * Groups of 4 rows are stored consecutively (each group = nb
     * consecutive block_q4_0x4 structs), so a row beyond the first group
     * needs both its group offset (group * nb) and its local index
     * within that group (row % 4) -- using `row` directly as the local
     * index only worked by accident for the first 4 rows. */
    for (int row = 0; row < nrows; row++) {
        int group = row / 4;
        int local_row = row % 4;
        const block_q4_0x4 *group_base = xb + (size_t)group * (size_t)nb;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++) {
            const block_q4_0x4 *b = group_base + ib;
            float dd = fp16_to_fp32_lookup(b->d[local_row]) * fp16_to_fp32_lookup(y[ib].d);
            int sumi = 0;
            for (int k = 0; k < 4; k++) {
                for (int i = 0; i < 4; i++) {
                    uint8_t byte = b->qs[k * 16 + local_row * 4 + i];
                    int v0 = (int8_t)(byte << 4) >> 4;
                    int v1 = (int8_t)(byte & 0xF0) >> 4;
                    sumi += ((v0 * y[ib].qs[k * 4 + i]) +
                             (v1 * y[ib].qs[k * 4 + i + 16]));
                }
            }
            sumf += (float)sumi * dd;
        }
        out[row] = sumf;
    }
}

/* ---- vec_dot_q4_0x4_4x8_q8_0: Q4_0_4x8 interleaved x Q8_0 (DOTPROD) ----
 * Processes 4 output rows simultaneously.
 * Adapted from llama.cpp's ggml_gemv_q4_0_4x8_q8_0.
 *
 * Q4_0_4_8 layout (blocklen=8): 4 rows interleaved in blocks of 8 bytes.
 * qs[k*32 + r*8 + j] for k=0..1, r=0..3, j=0..7.
 * Each block has 2 groups of 32 bytes, each group has 4 rows x 8 bytes.
 *
 * The DOTPROD kernel uses vdotq_s32 (not vdotq_laneq_s32) because blocklen=8.
 * Each vdotq_s32 computes 4 dot products of 4-byte segments.
 * The Q8_0 input (32 bytes = 2x16) is loaded as 4x8-byte duplicated chunks. */
void vec_dot_q4_0x4_4x8_q8_0(const void *vx, const void *wy, int n, float *out, int nrows) {
    const block_q4_0x4 *xb = (const block_q4_0x4 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)wy;
    int nb = n / 32;

#if defined(PICOLM_DOTPROD)
    for (int row_group = 0; row_group < nrows; row_group += 4) {
        const block_q4_0x4 *b_ptr = xb + (size_t)(row_group / 4) * (size_t)nb;
        const block_q8_0 *a_ptr = y;
        float32x4_t acc = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            int8x16_t b0 = vld1q_s8((const int8_t *)b_ptr->qs);
            int8x16_t b1 = vld1q_s8((const int8_t *)b_ptr->qs + 16);
            int8x16_t b2 = vld1q_s8((const int8_t *)b_ptr->qs + 32);
            int8x16_t b3 = vld1q_s8((const int8_t *)b_ptr->qs + 48);
            float16x4_t bd = vld1_f16((const __fp16 *)b_ptr->d);

            /* Q8_0 input: duplicate each 8-byte chunk to fill 16-byte register */
            int8x16_t a0 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs);
            int8x16_t a1 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 1);
            int8x16_t a2 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 2);
            int8x16_t a3 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 3);
            float16x4_t ad = vld1_dup_f16((const __fp16 *)&a_ptr->d);

            int32x4_t ret0 = vdupq_n_s32(0);
            int32x4_t ret1 = vdupq_n_s32(0);

            /* Low nibbles: b0/b1 with a0, b2/b3 with a1 */
            ret0 = vdotq_s32(ret0, vshlq_n_s8(b0, 4), a0);
            ret1 = vdotq_s32(ret1, vshlq_n_s8(b1, 4), a0);
            ret0 = vdotq_s32(ret0, vshlq_n_s8(b2, 4), a1);
            ret1 = vdotq_s32(ret1, vshlq_n_s8(b3, 4), a1);

            /* High nibbles: b0/b1 with a2, b2/b3 with a3 */
            ret0 = vdotq_s32(ret0, vandq_s8(b0, vdupq_n_s8((int8_t)0xf0U)), a2);
            ret1 = vdotq_s32(ret1, vandq_s8(b1, vdupq_n_s8((int8_t)0xf0U)), a2);
            ret0 = vdotq_s32(ret0, vandq_s8(b2, vdupq_n_s8((int8_t)0xf0U)), a3);
            ret1 = vdotq_s32(ret1, vandq_s8(b3, vdupq_n_s8((int8_t)0xf0U)), a3);

            /* Horizontal pair-add to combine ret0+ret1 */
            int32x4_t ret = vpaddq_s32(ret0, ret1);

            acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                             vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
            a_ptr++;
            b_ptr++;
        }
        vst1q_f32(out + row_group, acc);
    }
    return;
#endif

    /* Scalar fallback for Q4_0_4_8 (blocklen=8) */
    for (int row = 0; row < nrows; row++) {
        int group = row / 4;
        int local_row = row % 4;
        const block_q4_0x4 *group_base = xb + (size_t)group * (size_t)nb;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++) {
            const block_q4_0x4 *b = group_base + ib;
            float dd = fp16_to_fp32_lookup(b->d[local_row]) * fp16_to_fp32_lookup(y[ib].d);
            int sumi = 0;
            for (int k = 0; k < 2; k++) {
                for (int i = 0; i < 8; i++) {
                    uint8_t byte = b->qs[k * 32 + local_row * 8 + i];
                    int v0 = (int8_t)(byte << 4) >> 4;
                    int v1 = (int8_t)(byte & 0xF0) >> 4;
                    sumi += ((v0 * y[ib].qs[k * 8 + i]) +
                             (v1 * y[ib].qs[k * 8 + i + 16]));
                }
            }
            sumf += (float)sumi * dd;
        }
        out[row] = sumf;
    }
}

/* ---- repack_q8_0_for_i8mm: rearrange Q8_0 activations for smmla ----
 *
 * Standard Q8_0 block: d[2] + qs[0..31]
 * Repacked: qs[0..7] + qs[16..23] + qs[8..15] + qs[24..31]
 * Stored as int8_t[32] per block (no delta, deltas stored separately).
 *
 * After repack, smmla(A=[weight_lo, weight_hi], B=[repacked_0..7, repacked_8..15])
 * gives [dot(lo,lo), dot(lo,hi), dot(hi,lo), dot(hi,hi)]
 * where [0]+[3] = the correct dot product for one row block. */
#if defined(PICOLM_I8MM)
void repack_q8_0_for_i8mm(const void *src, int8_t *dst, float *dst_d, int n, int n_batch) {
    int nb = n / 32;
    for (int b = 0; b < n_batch; b++) {
        const block_q8_0 *x = (const block_q8_0 *)((const int8_t *)src + (size_t)b * gguf_type_row_size(GGUF_TYPE_Q8_0, n));
        /* Store per-block activation deltas */
        for (int ib = 0; ib < nb; ib++)
            dst_d[(size_t)b * (size_t)nb + (size_t)ib] = fp16_to_fp32_lookup(x[ib].d);
        for (int ib = 0; ib < nb; ib++) {
            int8_t *out = dst + (size_t)b * (size_t)nb * 32 + (size_t)ib * 32;
#if defined(PICOLM_I8MM)
            /* Repack: [qs[0..7], qs[16..23], qs[8..15], qs[24..31]]
             * This lets smmla extract dot(lo, qs[0..7]) + dot(hi, qs[16..23]) as diagonal. */
            int8x8_t a0 = vld1_s8(x[ib].qs);      /* qs[0..7] */
            int8x8_t a1 = vld1_s8(x[ib].qs + 16); /* qs[16..23] */
            int8x8_t a2 = vld1_s8(x[ib].qs + 8);  /* qs[8..15] */
            int8x8_t a3 = vld1_s8(x[ib].qs + 24); /* qs[24..31] */
            vst1_s8(out + 0, a0);
            vst1_s8(out + 8, a1);
            vst1_s8(out + 16, a2);
            vst1_s8(out + 24, a3);
#else
            /* Scalar repack: simple copy */
            for (int i = 0; i < 8; i++) {
                out[i] = x[ib].qs[i];
                out[i + 8] = x[ib].qs[i + 16];
                out[i + 16] = x[ib].qs[i + 8];
                out[i + 24] = x[ib].qs[i + 24];
            }
#endif
        }
    }
}
#endif /* PICOLM_I8MM */

/* ---- gemm_q4_0_4x8_i8mm: I8MM batched matmul with repacked activations ----
 *
 * Weights: Q4_0_4x8 interleaved (standard GGUF layout)
 * Activations: repacked via repack_q8_0_for_i8mm
 * Output: float32, [n_batch x d], row-major
 *
 * For each weight row:
 *   Expand 8 nibble bytes to low[8]+high[8] int8.
 *   smmla(A=[low, high], B=[repacked[0..7], repacked[8..15]])
 *   -> [0]=dot(low, repacked[0..7]), [3]=dot(high, repacked[8..15])
 *   -> [0]+[3] = correct dot product for this block
 *
 * Process 4 rows at a time, accumulating int32 sums across nb blocks. */
#if defined(PICOLM_I8MM)
void gemm_q4_0_4x8_i8mm(const void *W, const int8_t *X_repacked, const float *ad,
                         int n, float *out, int d, int n_batch) {
    int nb = n / 32;
    int d4 = (d / 4) * 4;

    for (int br = 0; br < d4; br += 4) {
        const block_q4_0x4 *wb = (const block_q4_0x4 *)W + (size_t)(br / 4) * (size_t)nb;

        for (int bc = 0; bc < n_batch; bc++) {
            /* Fused I8MM: process 4 blocks per group, accumulating int32
             * in NEON registers, then doing scvtf + fmul(scale) in float32x4.
             *
             * Each block has different wd[r] and adelta, so we cannot
             * accumulate raw int32 across all blocks and multiply once.
             * Instead, we group 4 blocks: accumulate int32 per block in
             * 4 int32x4 registers, then convert to float32 and multiply
             * by the combined scale (wd[r]*adelta), adding into a running
             * float32 accumulator. This mirrors MNN's approach where
             * scale multiply happens inside the SIMD kernel. */
            float row_sum[4] = {0};
            const int8_t *xrep = X_repacked + (size_t)bc * (size_t)nb * 32;
            const float *ad_bc = ad + (size_t)bc * (size_t)nb;

            /* Process 4 blocks per group */
            int ib;
            for (ib = 0; ib + 3 < nb; ib += 4) {
                float32x4_t f_acc[4];
                f_acc[0] = vdupq_n_f32(0.0f);
                f_acc[1] = vdupq_n_f32(0.0f);
                f_acc[2] = vdupq_n_f32(0.0f);
                f_acc[3] = vdupq_n_f32(0.0f);

                for (int k = 0; k < 4; k++) {
                    int ki = ib + k;
                    const block_q4_0x4 *b = wb + ki;
                    const uint8_t *wqs = (const uint8_t *)b->qs;
                    const int8_t *aq = xrep + ki * 32;
                    float adelta = ad_bc[ki];

                    int8x16_t B0 = vld1q_s8(aq);
                    int8x16_t B1 = vld1q_s8(aq + 16);

                    for (int r = 0; r < 4; r++) {
                        uint8x8_t wb0 = vld1_u8(wqs + r * 8);
                        int8x8_t lo0 = vshl_n_s8(vreinterpret_s8_u8(wb0), 4);
                        lo0 = vshr_n_s8(lo0, 4);
                        int8x8_t hi0 = vshr_n_s8(vreinterpret_s8_u8(wb0), 4);

                        uint8x8_t wb1 = vld1_u8(wqs + 32 + r * 8);
                        int8x8_t lo1 = vshl_n_s8(vreinterpret_s8_u8(wb1), 4);
                        lo1 = vshr_n_s8(lo1, 4);
                        int8x8_t hi1 = vshr_n_s8(vreinterpret_s8_u8(wb1), 4);

                        int8x16_t A0 = vcombine_s8(lo0, hi0);
                        int8x16_t A1 = vcombine_s8(lo1, hi1);

                        int32x4_t r0 = vmmlaq_s32(vdupq_n_s32(0), A0, B0);
                        int32x4_t r1 = vmmlaq_s32(vdupq_n_s32(0), A1, B1);
                        int32x4_t sum = vaddq_s32(r0, r1);

                        float scale = fp16_to_fp32_lookup(b->d[r]) * adelta;
                        f_acc[r] = vmlaq_n_f32(f_acc[r], vcvtq_f32_s32(sum), scale);
                    }
                }

                for (int r = 0; r < 4; r++)
                    row_sum[r] += vaddvq_f32(f_acc[r]);
            }

            /* Tail: 1-3 blocks */
            for (; ib < nb; ib++) {
                const block_q4_0x4 *b = wb + ib;
                const uint8_t *wqs = (const uint8_t *)b->qs;
                const int8_t *aq = xrep + ib * 32;
                float adelta = ad_bc[ib];

                int8x16_t B0 = vld1q_s8(aq);
                int8x16_t B1 = vld1q_s8(aq + 16);

                for (int r = 0; r < 4; r++) {
                    uint8x8_t wb0 = vld1_u8(wqs + r * 8);
                    int8x8_t lo0 = vshl_n_s8(vreinterpret_s8_u8(wb0), 4);
                    lo0 = vshr_n_s8(lo0, 4);
                    int8x8_t hi0 = vshr_n_s8(vreinterpret_s8_u8(wb0), 4);

                    uint8x8_t wb1 = vld1_u8(wqs + 32 + r * 8);
                    int8x8_t lo1 = vshl_n_s8(vreinterpret_s8_u8(wb1), 4);
                    lo1 = vshr_n_s8(lo1, 4);
                    int8x8_t hi1 = vshr_n_s8(vreinterpret_s8_u8(wb1), 4);

                    int8x16_t A0 = vcombine_s8(lo0, hi0);
                    int8x16_t A1 = vcombine_s8(lo1, hi1);

                    int32x4_t r0 = vmmlaq_s32(vdupq_n_s32(0), A0, B0);
                    int32x4_t r1 = vmmlaq_s32(vdupq_n_s32(0), A1, B1);
                    int32x4_t sum = vaddq_s32(r0, r1);
                    int32_t ra[4]; vst1q_s32(ra, sum);
                    row_sum[r] += (float)(ra[0] + ra[3]) *
                        fp16_to_fp32_lookup(b->d[r]) * adelta;
                }
            }

            for (int r = 0; r < 4; r++)
                out[(size_t)bc * (size_t)d + (size_t)br + (size_t)r] = row_sum[r];
        }
    }
}
#endif /* PICOLM_I8MM */

/* ---- gemm_q4_0_4x8_q8_0: dispatch with optional I8MM path ----
 *
 * When n_batch is large enough, repack activations and use I8MM smmla.
 * Otherwise fall back to DOTPROD gemv per token. */
void gemm_q4_0_4x8_q8_0(const void *W, const void *X, int n,
                         float *out, int d, int n_batch) {
    int b;
#if defined(PICOLM_I8MM)
    /* Use I8MM for batched prefill (n_batch >= 8) */
    if (n_batch >= 8) {
        int nb = n / 32;
        int8_t *xrep = malloc((size_t)n_batch * (size_t)nb * 32);
        float *ad = malloc((size_t)n_batch * (size_t)nb * sizeof(float));
        if (xrep && ad) {
            int d4 = (d / 4) * 4;
            repack_q8_0_for_i8mm(X, xrep, ad, n, n_batch);

            gemm_q4_0_4x8_i8mm(W, xrep, ad, n, out, d4, n_batch);

            /* Tail rows via DOTPROD */
            if (d4 < d) {
                size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
                for (int bi = 0; bi < n_batch; bi++) {
                    const void *xb = (const int8_t *)X + (size_t)bi * q8_rb;
                    vec_dot_q4_0x4_4x8_q8_0(W, xb, n, out + (size_t)bi * (size_t)d + (size_t)d4, d - d4);
                }
            }
            free(xrep);
            free(ad);
            return;
        }
    }
#endif
    /* DOTPROD gemv fallback */
    size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (b = 0; b < n_batch; b++) {
        const void *xb = (const int8_t *)X + (size_t)b * q8_rb;
        vec_dot_q4_0x4_4x8_q8_0(W, xb, n, out + (size_t)b * (size_t)d, d);
    }
}


/* ================================================================
 * vec_dot_q8_0_q8_0_deltas: int8 MAC with pre-converted x deltas
 *
 * Same as vec_dot_q8_0_q8_0 but takes pre-converted float32 deltas
 * for the quantized input x, avoiding per-block fp16_to_fp32 calls.
 * This saves ~nb/2 fp16_to_fp32 calls per vec_dot invocation.
 * ================================================================ */

float vec_dot_q8_0_q8_0_deltas(const void *qx, const float *qx_d, const void *qw, int n) {
    static int vd_cnt = 0;
    if (++vd_cnt <= 5 && getenv("PICOLM_GPU")) fprintf(stderr, "WARN: vec_dot_q8_0_q8_0_deltas called (call #%d, n=%d)\n", vd_cnt, n);
    const block_q8_0 *x = (const block_q8_0 *)qx;
    const block_q8_0 *w = (const block_q8_0 *)qw;
    int nb = n / 32;
    float sumf = 0.0f;
    int i = 0;
#if !defined(PICOLM_I8MM) && !defined(PICOLM_NEON) && !defined(PICOLM_AVX512) && !defined(PICOLM_AVX2) && !defined(PICOLM_AVX) && !defined(PICOLM_SSSE3)
    /* Pure scalar path: no SIMD available */
    for (i = 0; i < nb; i++) {
        int sumi = 0;
        for (int j = 0; j < 32; j++) sumi += x[i].qs[j] * w[i].qs[j];
        sumf += (float)sumi * qx_d[i] * fp16_to_fp32_lookup(w[i].d);
    }
    return sumf;
#endif

#ifdef PICOLM_I8MM
    /* I8MM: vmmlaq_s32 processes 16 int8 elements -> 4 int32 lanes.
     * Layout: lanes 0,1,2,3 = a_lo.b_lo, a_lo.b_hi, a_hi.b_lo, a_hi.b_hi
     * where lo=qs[0..7], hi=qs[8..15]. For dot product, use lanes 0+3.
     * Two vmmlaq calls per Q8_0 block (32 elements) -> lanes 0+3 from each. */
    for (i = 0; i + 1 < nb; i += 2) {
        const int8x16_t xi_0 = vld1q_s8(x[i].qs);
        const int8x16_t xi_1 = vld1q_s8(x[i].qs + 16);
        const int8x16_t wi_0 = vld1q_s8(w[i].qs);
        const int8x16_t wi_1 = vld1q_s8(w[i].qs + 16);
        int32x4_t si = vaddq_s32(vmmlaq_s32(vdupq_n_s32(0), xi_0, wi_0),
                                  vmmlaq_s32(vdupq_n_s32(0), xi_1, wi_1));
        int32_t dot_i = vgetq_lane_s32(si, 0) + vgetq_lane_s32(si, 3);
        float d = qx_d[i] * fp16_to_fp32_lookup(w[i].d);
        sumf += (float)dot_i * d;

        const int8x16_t xj_0 = vld1q_s8(x[i+1].qs);
        const int8x16_t xj_1 = vld1q_s8(x[i+1].qs + 16);
        const int8x16_t wj_0 = vld1q_s8(w[i+1].qs);
        const int8x16_t wj_1 = vld1q_s8(w[i+1].qs + 16);
        int32x4_t sj = vaddq_s32(vmmlaq_s32(vdupq_n_s32(0), xj_0, wj_0),
                                  vmmlaq_s32(vdupq_n_s32(0), xj_1, wj_1));
        int32_t dot_j = vgetq_lane_s32(sj, 0) + vgetq_lane_s32(sj, 3);
        float dj = qx_d[i+1] * fp16_to_fp32_lookup(w[i+1].d);
        sumf += (float)dot_j * dj;
    }

#elif defined(PICOLM_NEON)
    /* NEON: optimized int8 MAC via vpaddlq_s16, 2 blocks/iter */
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);
    for (i = 0; i + 1 < nb; i += 2) {
        const int8x16_t x0_0 = vld1q_s8(x[i].qs);
        const int8x16_t x0_1 = vld1q_s8(x[i].qs + 16);
        const int8x16_t w0_0 = vld1q_s8(w[i].qs);
        const int8x16_t w0_1 = vld1q_s8(w[i].qs + 16);
        {
            const int16x8_t p0 = vmull_s8(vget_low_s8(x0_0), vget_low_s8(w0_0));
            const int16x8_t p1 = vmull_s8(vget_high_s8(x0_0), vget_high_s8(w0_0));
            const int16x8_t p2 = vmull_s8(vget_low_s8(x0_1), vget_low_s8(w0_1));
            const int16x8_t p3 = vmull_s8(vget_high_s8(x0_1), vget_high_s8(w0_1));
            const int32x4_t s = vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
                                          vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)));
            const float d = qx_d[i] * fp16_to_fp32_lookup(w[i].d);
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(s), d);
        }
        const int8x16_t x1_0 = vld1q_s8(x[i+1].qs);
        const int8x16_t x1_1 = vld1q_s8(x[i+1].qs + 16);
        const int8x16_t w1_0 = vld1q_s8(w[i+1].qs);
        const int8x16_t w1_1 = vld1q_s8(w[i+1].qs + 16);
        {
            const int16x8_t p0 = vmull_s8(vget_low_s8(x1_0), vget_low_s8(w1_0));
            const int16x8_t p1 = vmull_s8(vget_high_s8(x1_0), vget_high_s8(w1_0));
            const int16x8_t p2 = vmull_s8(vget_low_s8(x1_1), vget_low_s8(w1_1));
            const int16x8_t p3 = vmull_s8(vget_high_s8(x1_1), vget_high_s8(w1_1));
            const int32x4_t s = vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
                                          vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)));
            const float d = qx_d[i+1] * fp16_to_fp32_lookup(w[i+1].d);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(s), d);
        }
    }
    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

#elif defined(PICOLM_AVX512)
    /* 512-bit wide: 2 q8_0 blocks (64 int8 elements) per iteration,
     * instead of AVX2's 1 block (32 elements) per iteration -- see
     * mul_sum_i8_pairs_avx512's comment for the VNNI vs. maddubs+madd
     * choice. Each 512-bit dot-product result holds 16 int32 lanes:
     * lanes 0-7 are block i's partial sums, lanes 8-15 are block i+1's,
     * so the two blocks' (different) per-block deltas are packed into
     * a matching low-8/high-8 float vector and applied with one fmadd
     * instead of extracting and horizontally summing each half
     * separately -- only the running accumulator needs a final
     * reduce, same as the AVX2 path's single hsum_avx after the loop. */
    __m512 acc = _mm512_setzero_ps();

    for (i = 0; i + 1 < nb; i += 2) {
        __m256i qx0 = _mm256_loadu_si256((const __m256i *)x[i].qs);
        __m256i qx1 = _mm256_loadu_si256((const __m256i *)x[i + 1].qs);
        __m256i qw0 = _mm256_loadu_si256((const __m256i *)w[i].qs);
        __m256i qw1 = _mm256_loadu_si256((const __m256i *)w[i + 1].qs);
        __m512i xx = _mm512_inserti64x4(_mm512_castsi256_si512(qx0), qx1, 1);
        __m512i ww = _mm512_inserti64x4(_mm512_castsi256_si512(qw0), qw1, 1);

        __m512i dot = mul_sum_i8_pairs_avx512(xx, ww);
        __m512 f = _mm512_cvtepi32_ps(dot);

        float d0 = qx_d[i] * fp16_to_fp32_lookup(w[i].d);
        float d1 = qx_d[i + 1] * fp16_to_fp32_lookup(w[i + 1].d);
        __m512 dvec = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(d0)),
                                          _mm256_set1_ps(d1), 1);
        acc = _mm512_fmadd_ps(f, dvec, acc);
    }
    sumf = _mm512_reduce_add_ps(acc);

#elif defined(PICOLM_AVX2)
    __m256 acc = _mm256_setzero_ps();

    for (i = 0; i + 1 < nb; i += 2) {
        __m256i qx0 = _mm256_loadu_si256((const __m256i *)x[i].qs);
        __m256i qx1 = _mm256_loadu_si256((const __m256i *)x[i + 1].qs);
        __m256i qw0 = _mm256_loadu_si256((const __m256i *)w[i].qs);
        __m256i qw1 = _mm256_loadu_si256((const __m256i *)w[i + 1].qs);

        __m256i s0 = mul_sum_i8_pairs_avx2(qx0, qw0);
        __m256i s1 = mul_sum_i8_pairs_avx2(qx1, qw1);
        __m256 f0 = _mm256_cvtepi32_ps(s0);
        __m256 f1 = _mm256_cvtepi32_ps(s1);

        float d0 = qx_d[i] * fp16_to_fp32_lookup(w[i].d);
        float d1 = qx_d[i + 1] * fp16_to_fp32_lookup(w[i + 1].d);
        __m256 dd0 = _mm256_set1_ps(d0);
        __m256 dd1 = _mm256_set1_ps(d1);
        acc = _mm256_fmadd_ps(f0, dd0, _mm256_fmadd_ps(f1, dd1, acc));
    }
    sumf = hsum_avx(acc);

#elif defined(PICOLM_AVX)
    __m256 acc = _mm256_setzero_ps();

    for (i = 0; i + 1 < nb; i += 2) {
        __m128i qx0 = _mm_loadu_si128((const __m128i *)x[i].qs);
        __m128i qx1 = _mm_loadu_si128((const __m128i *)x[i].qs + 1);
        __m128i qw0 = _mm_loadu_si128((const __m128i *)w[i].qs);
        __m128i qw1 = _mm_loadu_si128((const __m128i *)w[i].qs + 1);
        __m128i qx2 = _mm_loadu_si128((const __m128i *)x[i + 1].qs);
        __m128i qx3 = _mm_loadu_si128((const __m128i *)x[i + 1].qs + 1);
        __m128i qw2 = _mm_loadu_si128((const __m128i *)w[i + 1].qs);
        __m128i qw3 = _mm_loadu_si128((const __m128i *)w[i + 1].qs + 1);

        __m128i p0 = mul_sum_i8_pairs_sse(qx0, qw0);
        __m128i p1 = mul_sum_i8_pairs_sse(qx1, qw1);
        __m128i p2 = mul_sum_i8_pairs_sse(qx2, qw2);
        __m128i p3 = mul_sum_i8_pairs_sse(qx3, qw3);
        __m128i sum0 = _mm_add_epi32(p0, p1);
        __m128i sum1 = _mm_add_epi32(p2, p3);
        __m256 sums = _mm256_cvtepi32_ps(_mm256_set_m128i(sum1, sum0));

        float d0 = qx_d[i] * fp16_to_fp32_lookup(w[i].d);
        float d1 = qx_d[i + 1] * fp16_to_fp32_lookup(w[i + 1].d);
        __m256 deltas = _mm256_set_m128(_mm_set1_ps(d1), _mm_set1_ps(d0));
        acc = _mm256_add_ps(_mm256_mul_ps(deltas, sums), acc);
    }
    sumf = hsum_avx(acc);

#elif defined(PICOLM_SSSE3)
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();

    for (i = 0; i + 1 < nb; i += 2) {
        __m128i qx0 = _mm_loadu_si128((const __m128i *)x[i].qs);
        __m128i qx1 = _mm_loadu_si128((const __m128i *)x[i].qs + 1);
        __m128i qw0 = _mm_loadu_si128((const __m128i *)w[i].qs);
        __m128i qw1 = _mm_loadu_si128((const __m128i *)w[i].qs + 1);
        __m128i s0 = _mm_add_epi32(mul_sum_i8_pairs_sse(qx0, qw0),
                                    mul_sum_i8_pairs_sse(qx1, qw1));
        float d0 = qx_d[i] * fp16_to_fp32_lookup(w[i].d);
        __m128 dd0 = _mm_set1_ps(d0);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_cvtepi32_ps(s0), dd0));

        __m128i qx2 = _mm_loadu_si128((const __m128i *)x[i + 1].qs);
        __m128i qx3 = _mm_loadu_si128((const __m128i *)x[i + 1].qs + 1);
        __m128i qw2 = _mm_loadu_si128((const __m128i *)w[i + 1].qs);
        __m128i qw3 = _mm_loadu_si128((const __m128i *)w[i + 1].qs + 1);
        __m128i s1 = _mm_add_epi32(mul_sum_i8_pairs_sse(qx2, qw2),
                                    mul_sum_i8_pairs_sse(qx3, qw3));
        float d1 = qx_d[i + 1] * fp16_to_fp32_lookup(w[i + 1].d);
        __m128 dd1 = _mm_set1_ps(d1);
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_cvtepi32_ps(s1), dd1));
    }
    sumf = hsum_sse(_mm_add_ps(acc0, acc1));
#endif

    for (; i < nb; i++) {
        int sumi = 0;
        for (int j = 0; j < 32; j++) sumi += x[i].qs[j] * w[i].qs[j];
        sumf += (float)sumi * qx_d[i] * fp16_to_fp32_lookup(w[i].d);
    }
    return sumf;
}

/* ================================================================
 * vec_dot_q8_0_q8_0_deltas_batch4: compute 4 dot products
 *
 * Takes 4 pre-quantized token activations (x0..x3) against a single
 * weight row (w), returning 4 dot products. The weight is loaded once
 * per block and reused across all 4 activations, multiplying memory
 * bandwidth efficiency by 4x for the weight portion.
 *
 * Each activation has its own per-block delta (qx_d0..qx_d3), while
 * the weight has a shared per-block delta (w[i].d).
 * ================================================================ */
void vec_dot_q8_0_q8_0_deltas_batch4(
        const void *qx0, const float *qx_d0,
        const void *qx1, const float *qx_d1,
        const void *qx2, const float *qx_d2,
        const void *qx3, const float *qx_d3,
        const void *qw, int n,
        float *out0, float *out1, float *out2, float *out3) {
    const block_q8_0 *x[4] = {
        (const block_q8_0 *)qx0, (const block_q8_0 *)qx1,
        (const block_q8_0 *)qx2, (const block_q8_0 *)qx3
    };
    const float *qx_d[4] = { qx_d0, qx_d1, qx_d2, qx_d3 };
    float *out[4] = { out0, out1, out2, out3 };
    const block_q8_0 *w = (const block_q8_0 *)qw;
    int nb = n / 32;
    int i = 0;

#ifdef PICOLM_AVX512
    __m512 acc[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps()
    };

    for (i = 0; i + 1 < nb; i += 2) {
        __m256i qw0 = _mm256_loadu_si256((const __m256i *)w[i].qs);
        __m256i qw1 = _mm256_loadu_si256((const __m256i *)w[i + 1].qs);
        __m512i ww = _mm512_inserti64x4(_mm512_castsi256_si512(qw0), qw1, 1);

        float wd0 = fp16_to_fp32_lookup(w[i].d);
        float wd1 = fp16_to_fp32_lookup(w[i + 1].d);

        for (int b = 0; b < 4; b++) {
            __m256i qx0v = _mm256_loadu_si256((const __m256i *)x[b][i].qs);
            __m256i qx1v = _mm256_loadu_si256((const __m256i *)x[b][i + 1].qs);
            __m512i xx = _mm512_inserti64x4(_mm512_castsi256_si512(qx0v), qx1v, 1);

            __m512i dot = mul_sum_i8_pairs_avx512(xx, ww);
            __m512 f = _mm512_cvtepi32_ps(dot);

            float d0 = qx_d[b][i] * wd0;
            float d1 = qx_d[b][i + 1] * wd1;
            __m512 dvec = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(d0)),
                                              _mm256_set1_ps(d1), 1);
            acc[b] = _mm512_fmadd_ps(f, dvec, acc[b]);
        }
    }
    for (int b = 0; b < 4; b++) *out[b] = _mm512_reduce_add_ps(acc[b]);

#elif defined(PICOLM_AVX2)
    __m256 acc[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()
    };

    for (i = 0; i + 1 < nb; i += 2) {
        __m256i qw0 = _mm256_loadu_si256((const __m256i *)w[i].qs);
        __m256i qw1 = _mm256_loadu_si256((const __m256i *)w[i + 1].qs);

        float wd0 = fp16_to_fp32_lookup(w[i].d);
        float wd1 = fp16_to_fp32_lookup(w[i + 1].d);

        for (int b = 0; b < 4; b++) {
            __m256i qx0v = _mm256_loadu_si256((const __m256i *)x[b][i].qs);
            __m256i qx1v = _mm256_loadu_si256((const __m256i *)x[b][i + 1].qs);

            __m256i s0 = mul_sum_i8_pairs_avx2(qx0v, qw0);
            __m256i s1 = mul_sum_i8_pairs_avx2(qx1v, qw1);
            __m256 f0 = _mm256_cvtepi32_ps(s0);
            __m256 f1 = _mm256_cvtepi32_ps(s1);

            acc[b] = _mm256_fmadd_ps(f0, _mm256_set1_ps(qx_d[b][i] * wd0),
                       _mm256_fmadd_ps(f1, _mm256_set1_ps(qx_d[b][i + 1] * wd1), acc[b]));
        }
    }
    for (int b = 0; b < 4; b++) *out[b] = hsum_avx(acc[b]);

#else
    /* Fallback: 4 separate scalar dot products */
    for (int b = 0; b < 4; b++) {
        float s = 0.0f;
        for (i = 0; i < nb; i++) {
            int sumi = 0;
            for (int j = 0; j < 32; j++) sumi += x[b][i].qs[j] * w[i].qs[j];
            s += (float)sumi * qx_d[b][i] * fp16_to_fp32_lookup(w[i].d);
        }
        *out[b] = s;
    }
    return;
#endif

    /* Scalar remainder (same for all 4 outputs) */
    for (int b = 0; b < 4; b++) {
        float s = 0.0f;
        for (; i < nb; i++) {
            int sumi = 0;
            for (int j = 0; j < 32; j++) sumi += x[b][i].qs[j] * w[i].qs[j];
            s += (float)sumi * qx_d[b][i] * fp16_to_fp32_lookup(w[i].d);
        }
        *out[b] += s;
    }
}

/* ================================================================
 * vec_dot_q8_0_f32: fused dequant + dot for Q8_0 x float32
 *
 * Strategy: load int8 quantized values, widen to int32, multiply
 * by float32 input x, accumulate.
 *
 * With SSE2: process 8 elements per iteration (8 int8 -> 4x2 int32)
 * With AVX:  process 8 elements per iteration with 256-bit float accum
 * ================================================================ */

float vec_dot_q8_0_f32(const void *src, const float *x, int n) {
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    int nb = n / 32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const block_q8_0 *b = &blocks[i];
        float d = fp16_to_fp32_lookup(b->d);
        const int8_t *qs = b->qs;
        const float *xp = x + i * 32;

#ifdef PICOLM_NEON
        float32x4_t acc = vdupq_n_f32(0);

        for (int j = 0; j < 32; j += 8) {
            int8x8_t q8 = vld1_s8(qs + j);
            int16x8_t q16 = vmovl_s8(q8);
            /* Widen int16 -> int32, then convert to float32 */
            float32x4_t qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
            float32x4_t xf  = vld1q_f32(xp + j);
            acc = vmlaq_f32(acc, qf0, xf);

            float32x4_t qf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
            float32x4_t xf1 = vld1q_f32(xp + j + 4);
            acc = vmlaq_f32(acc, qf1, xf1);
        }
        sumf += d * vaddvq_f32_compat(acc);

#elif defined(PICOLM_AVX)
        /* AVX: load 8 int8s, widen to int32, convert to float, mul by x, accumulate */
        __m256 acc = _mm256_setzero_ps();
        const __m128i zero_i = _mm_setzero_si128();

        for (int j = 0; j < 32; j += 8) {
            __m128i q8 = _mm_loadl_epi64((const __m128i *)(qs + j));
            /* sign-extend int8 -> int16: unpack with zero in high byte, then srai 8 */
            __m128i q16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, q8), 8);
            /* sign-extend int16 -> int32: same trick */
            __m128i q32lo = _mm_srai_epi32(_mm_unpacklo_epi16(zero_i, q16), 16);
            __m128i q32hi = _mm_srai_epi32(_mm_unpackhi_epi16(zero_i, q16), 16);
            __m256 qf = _mm256_cvtepi32_ps(_mm256_set_m128i(q32hi, q32lo));
            __m256 xf = _mm256_loadu_ps(xp + j);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(qf, xf));
        }
        sumf += d * hsum_avx(acc);

#elif defined(PICOLM_SSE2)
        /* SSE2: load 8 int8s, widen to int32, convert to float, mul by x, accumulate */
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();
        const __m128i zero_i = _mm_setzero_si128();

        for (int j = 0; j < 32; j += 8) {
            __m128i q8 = _mm_loadl_epi64((const __m128i *)(qs + j));
            __m128i q16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, q8), 8);
            __m128i q32lo = _mm_srai_epi32(_mm_unpacklo_epi16(zero_i, q16), 16);
            __m128i q32hi = _mm_srai_epi32(_mm_unpackhi_epi16(zero_i, q16), 16);
            __m128 qf0 = _mm_cvtepi32_ps(q32lo);
            __m128 qf1 = _mm_cvtepi32_ps(q32hi);
            __m128 xf0 = _mm_loadu_ps(xp + j);
            __m128 xf1 = _mm_loadu_ps(xp + j + 4);
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(qf0, xf0));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(qf1, xf1));
        }
        sumf += d * hsum_sse(_mm_add_ps(acc0, acc1));

#else
        /* Scalar fallback */
        {
            float block_sum = 0.0f;
            for (int j = 0; j < 32; j++) {
                block_sum += (float)qs[j] * xp[j];
            }
            sumf += d * block_sum;
        }
#endif
    }
    return sumf;
}

/* vec_dot_q8_0_f32_batch4: dot product of 4 Q8_0 activations against the same
 * float32 weight vector. Keeps the weight vector w resident in cache/registers
 * while processing 4 tokens simultaneously. Each token has its own per-block
 * quantization delta (d0..d3), applied correctly per-token.
 *
 * Correct implementation: for each Q8_0 block, load 8 qs from each of the 4
 * tokens, sign-extend to float32, multiply by the same 8 w floats, accumulate
 * into 4 separate per-token accumulators. After the block, scale each
 * accumulator by its token's delta and add to the running sum. */
void vec_dot_q8_0_f32_batch4(const void *qx0, const void *qx1, const void *qx2, const void *qx3,
                              const float *w, int n,
                              float *out0, float *out1, float *out2, float *out3) {
#ifdef PICOLM_AVX2
    const block_q8_0 *b0 = (const block_q8_0 *)qx0;
    const block_q8_0 *b1 = (const block_q8_0 *)qx1;
    const block_q8_0 *b2 = (const block_q8_0 *)qx2;
    const block_q8_0 *b3 = (const block_q8_0 *)qx3;
    int nb = n / 32;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    const __m128i zero_i = _mm_setzero_si128();

    for (int i = 0; i < nb; i++) {
        float d0 = fp16_to_fp32_lookup(b0[i].d);
        float d1 = fp16_to_fp32_lookup(b1[i].d);
        float d2 = fp16_to_fp32_lookup(b2[i].d);
        float d3 = fp16_to_fp32_lookup(b3[i].d);
        const int8_t *qs0 = b0[i].qs, *qs1 = b1[i].qs, *qs2 = b2[i].qs, *qs3 = b3[i].qs;
        const float *wp = w + i * 32;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();

        for (int j = 0; j < 32; j += 8) {
            __m256 xf = _mm256_loadu_ps(wp + j);
            __m128i q8_0 = _mm_loadl_epi64((const __m128i *)(qs0 + j));
            __m128i q8_1 = _mm_loadl_epi64((const __m128i *)(qs1 + j));
            __m128i q8_2 = _mm_loadl_epi64((const __m128i *)(qs2 + j));
            __m128i q8_3 = _mm_loadl_epi64((const __m128i *)(qs3 + j));
            __m128i q16_0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, q8_0), 8);
            __m128i q16_1 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, q8_1), 8);
            __m128i q16_2 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, q8_2), 8);
            __m128i q16_3 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, q8_3), 8);
            __m128i q32lo_0 = _mm_srai_epi32(_mm_unpacklo_epi16(zero_i, q16_0), 16);
            __m128i q32hi_0 = _mm_srai_epi32(_mm_unpackhi_epi16(zero_i, q16_0), 16);
            __m128i q32lo_1 = _mm_srai_epi32(_mm_unpacklo_epi16(zero_i, q16_1), 16);
            __m128i q32hi_1 = _mm_srai_epi32(_mm_unpackhi_epi16(zero_i, q16_1), 16);
            __m128i q32lo_2 = _mm_srai_epi32(_mm_unpacklo_epi16(zero_i, q16_2), 16);
            __m128i q32hi_2 = _mm_srai_epi32(_mm_unpackhi_epi16(zero_i, q16_2), 16);
            __m128i q32lo_3 = _mm_srai_epi32(_mm_unpacklo_epi16(zero_i, q16_3), 16);
            __m128i q32hi_3 = _mm_srai_epi32(_mm_unpackhi_epi16(zero_i, q16_3), 16);
            __m256 qf0 = _mm256_cvtepi32_ps(_mm256_set_m128i(q32hi_0, q32lo_0));
            __m256 qf1 = _mm256_cvtepi32_ps(_mm256_set_m128i(q32hi_1, q32lo_1));
            __m256 qf2 = _mm256_cvtepi32_ps(_mm256_set_m128i(q32hi_2, q32lo_2));
            __m256 qf3 = _mm256_cvtepi32_ps(_mm256_set_m128i(q32hi_3, q32lo_3));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(qf0, xf));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(qf1, xf));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(qf2, xf));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(qf3, xf));
        }
        sum0 += d0 * hsum_avx(acc0);
        sum1 += d1 * hsum_avx(acc1);
        sum2 += d2 * hsum_avx(acc2);
        sum3 += d3 * hsum_avx(acc3);
    }
    *out0 = sum0;
    *out1 = sum1;
    *out2 = sum2;
    *out3 = sum3;
#elif defined(PICOLM_AVX)
    const block_q8_0 *b0 = (const block_q8_0 *)qx0;
    const block_q8_0 *b1 = (const block_q8_0 *)qx1;
    const block_q8_0 *b2 = (const block_q8_0 *)qx2;
    const block_q8_0 *b3 = (const block_q8_0 *)qx3;
    int nb = n / 32;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    const __m128i zero_i = _mm_setzero_si128();

    for (int i = 0; i < nb; i++) {
        float d0 = fp16_to_fp32_lookup(b0[i].d);
        float d1 = fp16_to_fp32_lookup(b1[i].d);
        float d2 = fp16_to_fp32_lookup(b2[i].d);
        float d3 = fp16_to_fp32_lookup(b3[i].d);
        const int8_t *qs0 = b0[i].qs, *qs1 = b1[i].qs, *qs2 = b2[i].qs, *qs3 = b3[i].qs;
        const float *wp = w + i * 32;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();

        for (int j = 0; j < 32; j += 8) {
            __m256 xf = _mm256_loadu_ps(wp + j);
            __m128i q8_0 = _mm_loadl_epi64((const __m128i *)(qs0 + j));
            __m128i q8_1 = _mm_loadl_epi64((const __m128i *)(qs1 + j));
            __m128i q8_2 = _mm_loadl_epi64((const __m128i *)(qs2 + j));
            __m128i q8_3 = _mm_loadl_epi64((const __m128i *)(qs3 + j));
            __m128i q16_0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, q8_0), 8);
            __m128i q16_1 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, q8_1), 8);
            __m128i q16_2 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, q8_2), 8);
            __m128i q16_3 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, q8_3), 8);
            __m128i q32lo_0 = _mm_srai_epi32(_mm_unpacklo_epi16(zero_i, q16_0), 16);
            __m128i q32hi_0 = _mm_srai_epi32(_mm_unpackhi_epi16(zero_i, q16_0), 16);
            __m128i q32lo_1 = _mm_srai_epi32(_mm_unpacklo_epi16(zero_i, q16_1), 16);
            __m128i q32hi_1 = _mm_srai_epi32(_mm_unpackhi_epi16(zero_i, q16_1), 16);
            __m128i q32lo_2 = _mm_srai_epi32(_mm_unpacklo_epi16(zero_i, q16_2), 16);
            __m128i q32hi_2 = _mm_srai_epi32(_mm_unpackhi_epi16(zero_i, q16_2), 16);
            __m128i q32lo_3 = _mm_srai_epi32(_mm_unpacklo_epi16(zero_i, q16_3), 16);
            __m128i q32hi_3 = _mm_srai_epi32(_mm_unpackhi_epi16(zero_i, q16_3), 16);
            __m256 qf0 = _mm256_cvtepi32_ps(_mm256_set_m128i(q32hi_0, q32lo_0));
            __m256 qf1 = _mm256_cvtepi32_ps(_mm256_set_m128i(q32hi_1, q32lo_1));
            __m256 qf2 = _mm256_cvtepi32_ps(_mm256_set_m128i(q32hi_2, q32lo_2));
            __m256 qf3 = _mm256_cvtepi32_ps(_mm256_set_m128i(q32hi_3, q32lo_3));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(qf0, xf));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(qf1, xf));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(qf2, xf));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(qf3, xf));
        }
        sum0 += d0 * hsum_avx(acc0);
        sum1 += d1 * hsum_avx(acc1);
        sum2 += d2 * hsum_avx(acc2);
        sum3 += d3 * hsum_avx(acc3);
    }
    *out0 = sum0;
    *out1 = sum1;
    *out2 = sum2;
    *out3 = sum3;
#elif defined(PICOLM_NEON)
    const block_q8_0 *b0 = (const block_q8_0 *)qx0;
    const block_q8_0 *b1 = (const block_q8_0 *)qx1;
    const block_q8_0 *b2 = (const block_q8_0 *)qx2;
    const block_q8_0 *b3 = (const block_q8_0 *)qx3;
    int nb = n / 32;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d0 = fp16_to_fp32_lookup(b0[i].d);
        float d1 = fp16_to_fp32_lookup(b1[i].d);
        float d2 = fp16_to_fp32_lookup(b2[i].d);
        float d3 = fp16_to_fp32_lookup(b3[i].d);
        const int8_t *qs0 = b0[i].qs, *qs1 = b1[i].qs, *qs2 = b2[i].qs, *qs3 = b3[i].qs;
        const float *wp = w + i * 32;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

        for (int j = 0; j < 32; j += 8) {
            float32x4_t xf = vld1q_f32(wp + j);
            int8x8_t q8_0 = vld1_s8(qs0 + j);
            int8x8_t q8_1 = vld1_s8(qs1 + j);
            int8x8_t q8_2 = vld1_s8(qs2 + j);
            int8x8_t q8_3 = vld1_s8(qs3 + j);
            int16x8_t q16_0 = vmovl_s8(q8_0);
            int16x8_t q16_1 = vmovl_s8(q8_1);
            int16x8_t q16_2 = vmovl_s8(q8_2);
            int16x8_t q16_3 = vmovl_s8(q8_3);
            float32x4_t qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_0)));
            float32x4_t qf1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_1)));
            float32x4_t qf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_2)));
            float32x4_t qf3 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_3)));
            acc0 = vmlaq_f32(acc0, qf0, xf);
            acc1 = vmlaq_f32(acc1, qf1, xf);
            acc2 = vmlaq_f32(acc2, qf2, xf);
            acc3 = vmlaq_f32(acc3, qf3, xf);
            qf0 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_0)));
            qf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_1)));
            qf2 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_2)));
            qf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_3)));
            xf  = vld1q_f32(wp + j + 4);
            acc0 = vmlaq_f32(acc0, qf0, xf);
            acc1 = vmlaq_f32(acc1, qf1, xf);
            acc2 = vmlaq_f32(acc2, qf2, xf);
            acc3 = vmlaq_f32(acc3, qf3, xf);
        }
        sum0 += d0 * vaddvq_f32_compat(acc0);
        sum1 += d1 * vaddvq_f32_compat(acc1);
        sum2 += d2 * vaddvq_f32_compat(acc2);
        sum3 += d3 * vaddvq_f32_compat(acc3);
    }
    *out0 = sum0;
    *out1 = sum1;
    *out2 = sum2;
    *out3 = sum3;
#else
    /* Scalar fallback: 4 independent calls */
    *out0 = vec_dot_q8_0_f32(qx0, w, n);
    *out1 = vec_dot_q8_0_f32(qx1, w, n);
    *out2 = vec_dot_q8_0_f32(qx2, w, n);
    *out3 = vec_dot_q8_0_f32(qx3, w, n);
#endif
}

/* ================================================================
 * vec_dot_q4_0_f32: fused dequant + dot for Q4_0 x float32
 *
 * Q4_0 block: 16 bytes qs + 2 bytes d(FP16) = 18 bytes for 32 values.
 *   vals[0..15]  = low  nibble of qs[0..15]  (qs[j] & 0xF)
 *   vals[16..31] = high nibble of qs[0..15]  (qs[j] >> 4)
 * Each val in [0..15], offset by -8 to get signed [-8..+7].
 * ================================================================ */

float vec_dot_q4_0_f32(const void *src, const float *x, int n) {
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    int nb = n / 32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d);
        const uint8_t *qs = blocks[i].qs;
        const float *xp = x + i * 32;
        float block_sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            block_sum += (float)((qs[j] & 0xF) - 8) * xp[j];
            block_sum += (float)((qs[j] >> 4) - 8) * xp[j + 16];
        }
        sumf += d * block_sum;
    }
    return sumf;
}

/* vec_dot_q4_1_f32: fused dequant + dot for Q4_1 x float32
 * Q4_1 block: 16 bytes qs + 2 bytes d(FP16) + 2 bytes m(FP16) = 20 bytes for 32 values.
 *   vals[0..15]  = low  nibble of qs[0..15]  (unsigned, 0..15)
 *   vals[16..31] = high nibble of qs[0..15]  (unsigned, 0..15)
 * Dequant: val = qs[j] * d + m */
float vec_dot_q4_1_f32(const void *src, const float *x, int n) {
    const block_q4_1 *blocks = (const block_q4_1 *)src;
    int nb = n / 32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d);
        float m = fp16_to_fp32_lookup(blocks[i].m);
        const uint8_t *qs = blocks[i].qs;
        const float *xp = x + i * 32;
        float block_sum = 0.0f;
        float m_sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            uint8_t lo = qs[j] & 0xF;
            uint8_t hi = qs[j] >> 4;
            block_sum += (float)lo * xp[j];
            block_sum += (float)hi * xp[j + 16];
            m_sum += xp[j];
            m_sum += xp[j + 16];
        }
        sumf += d * block_sum + m * m_sum;
    }
    return sumf;
}

/* vec_dot_q1_0_f32: fused dequant + dot for Q1_0 x float32
 * Q1_0 block: 16 bytes qs (128 bits) + 2 bytes d(FP16) = 18 bytes per 128 values.
 * Dequant: val[j] = (bit[j] ? +d : -d) */
float vec_dot_q1_0_f32(const void *src, const float *x, int n) {
    /* Pre-quantize x to Q8_0 and delegate to vec_dot_q1_0_q8_0.
     * This allows the AVX2/VNNI path to be used even with F32 activations. */
    if (n >= 128 && n % 128 == 0) {
        size_t nq8 = (size_t)(n / 32) * sizeof(block_q8_0);
        block_q8_0 qx_buf[4]; /* stack buffer for typical sizes */
        block_q8_0 *qx;
        int qx_owned = 0;

        if (nq8 <= sizeof(qx_buf)) {
            qx = qx_buf;
        } else {
            qx = (block_q8_0 *)malloc(nq8);
            if (!qx) goto scalar_path;
            qx_owned = 1;
        }
        quantize_row_q8_0(x, qx, n);
        float result = vec_dot_q1_0_q8_0(src, qx, n);
        if (qx_owned) free(qx);
        return result;

    scalar_path:
        ;
    }

    const block_q1_0 *blocks = (const block_q1_0 *)src;
    int nb = n / 128;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d);
        const uint8_t *qs = blocks[i].qs;
        const float *xp = x + i * 128;
        float block_sum = 0.0f;
        for (int j = 0; j < 128; j++) {
            int byte_idx = j / 8;
            int bit_off  = j % 8;
            int bit = (qs[byte_idx] >> bit_off) & 1;
            block_sum += bit ? xp[j] : -xp[j];
        }
        sumf += d * block_sum;
    }
    return sumf;
}

/* vec_dot_q2_0_f32: fused dequant + dot for Q2_0 x float32
 * Q2_0 block: 32 bytes qs + 2 bytes d(FP16) = 34 bytes per 128 values.
 * Dequant: val[j] = ((qs[j] & 3) - 1) * d */
float vec_dot_q2_0_f32(const void *src, const float *x, int n) {
    /* Pre-quantize x to Q8_0 and delegate to vec_dot_q2_0_q8_0.
     * This allows the VNNI path to be used even with F32 activations.
     * For small n (<128), the quantization overhead dominates; fall back
     * to scalar. */
    if (n >= 128 && n % 128 == 0) {
        size_t nq8 = (size_t)(n / 32) * sizeof(block_q8_0);
        block_q8_0 qx_buf[4]; /* stack buffer for typical sizes */
        block_q8_0 *qx;
        int qx_owned = 0;

        if (nq8 <= sizeof(qx_buf)) {
            qx = qx_buf;
        } else {
            qx = (block_q8_0 *)malloc(nq8);
            if (!qx) goto scalar_path;
            qx_owned = 1;
        }
        quantize_row_q8_0(x, qx, n);
        float result = vec_dot_q2_0_q8_0(src, qx, n);
        if (qx_owned) free(qx);
        return result;

    scalar_path:
        ;
    }

    const block_q2_0 *blocks = (const block_q2_0 *)src;
    int nb = n / 128;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d);
        const uint8_t *qs = blocks[i].qs;
        const float *xp = x + i * 128;
        float block_sum = 0.0f;
        for (int j = 0; j < 128; j++) {
            int byte_idx = j / 4;
            int bit_off  = (j % 4) * 2;
            int q = (qs[byte_idx] >> bit_off) & 0x03;
            block_sum += (float)(q - 1) * xp[j];
        }
        sumf += d * block_sum;
    }
    return sumf;
}

/* vec_dot_q1_0_q8_0: Q1_0 weights x Q8_0 input (int8 MAC)
 * Ported from llama.cpp ggml_vec_dot_q1_0_q8_0.
 * Q1_0: 16 bytes qs (128 bits) + 2 bytes d per 128 values.
 * Q8_0: 32 bytes qs + 2 bytes d per 32 values.
 * Each Q1_0 block (128 vals) matches 4 Q8_0 blocks (4*32 = 128 vals).
 *
 * Algorithm: For each Q1_0 block, expand the 128 sign bits into sign masks,
 * then for each of the 4 Q8_0 sub-blocks, use sign masks to conditionally
 * negate Q8_0 values, then sum with maddubs_epi16 + madd_epi16.
 * The result is sum of signed Q8_0 values, scaled by d0*d1. */
float vec_dot_q1_0_q8_0(const void *vx, const void *wy, int n) {
    const block_q1_0 *x = (const block_q1_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)wy;
    int nb = n / 128;

#if defined(PICOLM_AVX2)
    {
        const __m256i ones_8  = _mm256_set1_epi8(1);
        const __m256i ones_16 = _mm256_set1_epi16(1);
        const __m256i byte_shuf = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3);
        const __m256i bit_masks = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, (char)-128,
            1, 2, 4, 8, 16, 32, 64, (char)-128,
            1, 2, 4, 8, 16, 32, 64, (char)-128,
            1, 2, 4, 8, 16, 32, 64, (char)-128);
        const __m256i zero = _mm256_setzero_si256();
        float sumf = 0.0f;
        __m256 acc = _mm256_setzero_ps();

        for (int ib = 0; ib < nb; ib++) {
            float d0 = fp16_to_fp32_lookup(x[ib].d);
            const uint32_t *qs32 = (const uint32_t *)x[ib].qs;
            const block_q8_0 *yp = &y[ib * 4];
            __m256 acc_block = _mm256_setzero_ps();

            for (int K = 0; K < 4; K++) {
                __m256i qy = _mm256_loadu_si256((const __m256i *)yp[K].qs);
                __m256i qs_vec = _mm256_set1_epi32(qs32[K]);
                __m256i sm = _mm256_cmpeq_epi8(
                    _mm256_and_si256(
                        _mm256_shuffle_epi8(qs_vec, byte_shuf),
                        bit_masks), zero);
                __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
                __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
                float d1 = fp16_to_fp32_lookup(yp[K].d);
                acc_block = _mm256_fmadd_ps(
                    _mm256_set1_ps(d1), _mm256_cvtepi32_ps(s32), acc_block);
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d0), acc_block, acc);
        }
        sumf = hsum_avx(acc);
        return sumf;
    }
#elif defined(PICOLM_AVX)
    /* AVX path: broadcast 32-bit sign value via AVX, split into two
     * 128-bit halves for pshufb expansion + maddubs (SSE4.1).
     * Uses a correct 256-bit bit expansion via _mm256_set1_epi32
     * (AVX, not AVX2) + two 128-bit _mm_shuffle_epi8 calls. */
    {
        const __m128i ones_8  = _mm_set1_epi8(1);
        const __m128i ones_16 = _mm_set1_epi16(1);
        const __m128i zero = _mm_setzero_si128();
        /* pshufb index: replicate each of the 4 bytes 8 times */
        const __m128i byte_shuf = _mm_setr_epi8(
            0,0,0,0,0,0,0,0,
            1,1,1,1,1,1,1,1);
        const __m128i bit_masks = _mm_setr_epi8(
            1,2,4,8,16,32,64,(char)-128,
            1,2,4,8,16,32,64,(char)-128);
        float sumf;
        __m256 acc = _mm256_setzero_ps();

        for (int ib = 0; ib < nb; ib++) {
            float d0 = fp16_to_fp32_lookup(x[ib].d);
            const uint32_t *qs32 = (const uint32_t *)x[ib].qs;
            const block_q8_0 *yp = &y[ib * 4];
            __m256 acc_block = _mm256_setzero_ps();

            for (int K = 0; K < 4; K++) {
                /* Expand 32 bits into 32 bytes of 0x00/0xFF sign mask.
                 * Split the 32-bit value into two 16-bit halves (low=bytes 0,1,
                 * high=bytes 2,3), replicate each to all 16-bit slots, then
                 * use pshufb to replicate each byte 8x for bit testing. */
                __m128i qs16 = _mm_cvtsi32_si128((int)qs32[K]);
                __m128i rep_lo = _mm_shufflelo_epi16(qs16, 0);
                __m128i sm_lo = _mm_cmpeq_epi8(
                    _mm_and_si128(_mm_shuffle_epi8(rep_lo, byte_shuf), bit_masks), zero);
                __m128i rep_hi = _mm_shufflelo_epi16(qs16, 0x11);
                __m128i sm_hi = _mm_cmpeq_epi8(
                    _mm_and_si128(_mm_shuffle_epi8(rep_hi, byte_shuf), bit_masks), zero);
                /* Load Q8_0 in two 128-bit chunks */
                __m128i qy0 = _mm_loadu_si128((const __m128i *)yp[K].qs);
                __m128i qy1 = _mm_loadu_si128((const __m128i *)(yp[K].qs + 16));
                __m128i sy0 = _mm_sub_epi8(_mm_xor_si128(qy0, sm_lo), sm_lo);
                __m128i sy1 = _mm_sub_epi8(_mm_xor_si128(qy1, sm_hi), sm_hi);
                __m128i s16_0 = _mm_maddubs_epi16(ones_8, sy0);
                __m128i s16_1 = _mm_maddubs_epi16(ones_8, sy1);
                __m128i s32_0 = _mm_madd_epi16(s16_0, ones_16);
                __m128i s32_1 = _mm_madd_epi16(s16_1, ones_16);
                __m256 q = _mm256_cvtepi32_ps(_mm256_set_m128i(s32_1, s32_0));
                float d1 = fp16_to_fp32_lookup(yp[K].d);
                acc_block = _mm256_add_ps(acc_block, _mm256_mul_ps(_mm256_set1_ps(d1), q));
            }
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d0), acc_block));
        }
        sumf = hsum_avx(acc);
        return sumf;
    }
#elif defined(PICOLM_SSE3)
    /* SSSE3 path: 128-bit bit expansion + 128-bit maddubs (SSE4.1).
     * Uses 4 independent __m128 accumulators for the 4 Q8_0 sub-blocks.
     * Ported from llama.cpp. */
    {
        float sumf;
        const __m128i ones_8  = _mm_set1_epi8(1);
        const __m128i ones_16 = _mm_set1_epi16(1);
        const __m128i byte_shuf = _mm_setr_epi8(
            0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1);
        const __m128i bit_masks = _mm_setr_epi8(
            1,2,4,8,16,32,64,(char)-128,
            1,2,4,8,16,32,64,(char)-128);
        const __m128i zero = _mm_setzero_si128();
        __m128 acc_0 = _mm_setzero_ps();
        __m128 acc_1 = _mm_setzero_ps();
        __m128 acc_2 = _mm_setzero_ps();
        __m128 acc_3 = _mm_setzero_ps();

        for (int ib = 0; ib < nb; ib++) {
            __m128 d0 = _mm_set1_ps(fp16_to_fp32_lookup(x[ib].d));
            const uint32_t *qs32 = (const uint32_t *)x[ib].qs;
            const block_q8_0 *yp = &y[ib * 4];

#define Q1_SSSE3_BLOCK(QS_OFF, Y_IDX, ACC) \
            { \
                __m128i qs16 = _mm_cvtsi32_si128((int)qs32[QS_OFF / 4]); \
                /* Expand low 16 bits (bytes 0,1) into 16 sign mask bytes */ \
                __m128i rep_lo = _mm_shufflelo_epi16(qs16, 0); \
                __m128i sm0 = _mm_cmpeq_epi8( \
                    _mm_and_si128(_mm_shuffle_epi8(rep_lo, byte_shuf), bit_masks), zero); \
                /* Expand high 16 bits (bytes 2,3) into 16 sign mask bytes */ \
                __m128i rep_hi = _mm_shufflelo_epi16(qs16, 0x11); \
                __m128i sm1 = _mm_cmpeq_epi8( \
                    _mm_and_si128(_mm_shuffle_epi8(rep_hi, byte_shuf), bit_masks), zero); \
                __m128i qy0 = _mm_loadu_si128((const __m128i *)yp[Y_IDX].qs); \
                __m128i qy1 = _mm_loadu_si128((const __m128i *)(yp[Y_IDX].qs + 16)); \
                __m128i sy0 = _mm_sub_epi8(_mm_xor_si128(qy0, sm0), sm0); \
                __m128i sy1 = _mm_sub_epi8(_mm_xor_si128(qy1, sm1), sm1); \
                __m128i sum_0 = _mm_madd_epi16(_mm_maddubs_epi16(ones_8, sy0), ones_16); \
                __m128i sum_1 = _mm_madd_epi16(_mm_maddubs_epi16(ones_8, sy1), ones_16); \
                __m128 q = _mm_cvtepi32_ps(_mm_add_epi32(sum_0, sum_1)); \
                (ACC) = _mm_add_ps((ACC), _mm_mul_ps(_mm_mul_ps(d0, \
                    _mm_set1_ps(fp16_to_fp32_lookup(yp[Y_IDX].d))), q)); \
            }
            Q1_SSSE3_BLOCK(0, 0, acc_0)
            Q1_SSSE3_BLOCK(4, 1, acc_1)
            Q1_SSSE3_BLOCK(8, 2, acc_2)
            Q1_SSSE3_BLOCK(12, 3, acc_3)
#undef Q1_SSSE3_BLOCK
        }
        sumf = hsum_sse(_mm_add_ps(_mm_add_ps(acc_0, acc_1), _mm_add_ps(acc_2, acc_3)));
        return sumf;
    }
#elif defined(PICOLM_NEON)
    /* Plain NEON: expand 32 sign bits -> 32 sign masks via vcreate_u8 from
     * precomputed lookup table (mirrors llama.cpp approach), XOR+subtract Q8_0,
     * accumulate with vpadalq_s16. 1 bit/value, 128 values/block.
     *
     * table_q1_mask: 256 entries of 8 bytes each. For each input byte value
     * 0..255, each of its 8 bits expands to a sign mask byte:
     *   bit=1 -> 0x00 (keep Q8_0 sign), bit=0 -> 0xFF (negate Q8_0).
     * vcreate_u8 converts each 64-bit table entry directly into a uint8x8_t. */
    {
        /* Generate table at first call: for byte value v, each bit i
         * produces byte (v & (1<<i)) ? 0x00 : 0xFF */
        static uint64_t table_q1_mask[256];
        static int tbl_init = 0;
        if (!tbl_init) {
            for (int v = 0; v < 256; v++) {
                uint64_t entry = 0;
                for (int i = 0; i < 8; i++) {
                    entry |= (uint64_t)((v & (1 << i)) ? 0x00 : 0xFF) << (i * 8);
                }
                table_q1_mask[v] = entry;
            }
            tbl_init = 1;
        }

        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++) {
            float d0 = fp16_to_fp32_lookup(x[ib].d);
            const uint8_t *qs = x[ib].qs;
            const block_q8_0 *yp = &y[ib * 4];
            float sumi = 0.0f;

            for (int K = 0; K < 4; K++) {
                const uint8_t *bits = &qs[K * 4];

                /* Expand 4 bytes of sign bits -> 32 sign mask bytes
                 * via 4 table lookups + vcreate_u8 -> 2 int8x16 vectors */
                const int8x16_t sm0 = vreinterpretq_s8_u8(
                    vcombine_u8(vcreate_u8(table_q1_mask[bits[0]]),
                                vcreate_u8(table_q1_mask[bits[1]])));
                const int8x16_t sm1 = vreinterpretq_s8_u8(
                    vcombine_u8(vcreate_u8(table_q1_mask[bits[2]]),
                                vcreate_u8(table_q1_mask[bits[3]])));

                /* sy = xor(qy, sm) - sm: bit=1 -> +qy, bit=0 -> -qy */
                const int8x16_t qy0 = vld1q_s8(yp[K].qs);
                const int8x16_t qy1 = vld1q_s8(yp[K].qs + 16);
                const int8x16_t sy0 = vsubq_s8(veorq_s8(qy0, sm0), sm0);
                const int8x16_t sy1 = vsubq_s8(veorq_s8(qy1, sm1), sm1);

                /* Horizontal sum: pairwise add long accumulate */
                int32x4_t p = vdupq_n_s32(0);
                p = vpadalq_s16(p, vpaddlq_s8(sy0));
                p = vpadalq_s16(p, vpaddlq_s8(sy1));
                float d1 = fp16_to_fp32_lookup(yp[K].d);
                sumi += d1 * (float)vaddvq_s32(p);
            }
            sumf += d0 * sumi;
        }
        return sumf;
    }

#else
    /* Scalar fallback */
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        float d0 = fp16_to_fp32_lookup(x[ib].d);
        float sumi = 0.0f;
        for (int K = 0; K < 4; K++) {
            float d1 = fp16_to_fp32_lookup(y[ib * 4 + K].d);
            int sumi_block = 0;
            const uint8_t *bits = &x[ib].qs[K * 4];
            const int8_t *qy = y[ib * 4 + K].qs;
            for (int b = 0; b < 4; b++, qy += 8) {
                unsigned mask = bits[b];
                sumi_block += ((mask & 0x01) ? qy[0] : -qy[0])
                            + ((mask & 0x02) ? qy[1] : -qy[1])
                            + ((mask & 0x04) ? qy[2] : -qy[2])
                            + ((mask & 0x08) ? qy[3] : -qy[3])
                            + ((mask & 0x10) ? qy[4] : -qy[4])
                            + ((mask & 0x20) ? qy[5] : -qy[5])
                            + ((mask & 0x40) ? qy[6] : -qy[6])
                            + ((mask & 0x80) ? qy[7] : -qy[7]);
            }
            sumi += d1 * (float)sumi_block;
        }
        sumf += d0 * sumi;
    }
    return sumf;
#endif
}

/* vec_dot_q2_0_q8_0: Q2_0 weights x Q8_0 input
 * Q2_0: 32 bytes qs + 2 bytes d per 128 values.
 * Q8_0: 32 bytes qs + 2 bytes d per 32 values.
 * Each Q2_0 block (128 vals) matches 4 Q8_0 blocks. */
float vec_dot_q2_0_q8_0(const void *vx, const void *wy, int n) {
    const block_q2_0 *x = (const block_q2_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)wy;
    int nb = n / 128;
    float sumf = 0.0f;

#ifdef PICOLM_VNNI
    /* AVX-512-VNNI path: unpack 2-bit codes c in {0,1,2,3} (value = c-1),
     * then dot((c-1), qy) = dpbusd(c, qy) - dpbusd(1, qy).
     * Ported from llama.cpp ggml_vec_dot_q2_0_q8_0. */
    {
        const __m256i ones   = _mm256_set1_epi8(1);
        const __m128i idxlo  = _mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
        const __m128i idxhi  = _mm_setr_epi8(4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);
        const __m256i mul    = _mm256_setr_epi16(64,16,4,1, 64,16,4,1, 64,16,4,1, 64,16,4,1);
        const __m256i three  = _mm256_set1_epi16(3);

        for (int i = 0; i < nb; i++) {
            const float d0 = fp16_to_fp32_lookup(x[i].d);
            float sumi = 0.0f;
            for (int k = 0; k < 4; k++) {
                const block_q8_0 *yb = &y[i * 4 + k];
                const float d1 = fp16_to_fp32_lookup(yb->d);
                const __m256i qy = _mm256_loadu_si256((const __m256i *)yb->qs);
                /* Load 8 bytes of qs (covers 32 2-bit values for this Q8_0 block) */
                const __m128i src = _mm_loadl_epi64((const __m128i *)&x[i].qs[k * 8]);
                /* Replicate each byte 4x -> 32 bytes in low+high 128-bit lanes */
                const __m256i rep = _mm256_set_m128i(
                    _mm_shuffle_epi8(src, idxhi), _mm_shuffle_epi8(src, idxlo));
                /* Expand bytes to 16-bit, extract 2-bit fields via multiply+shift */
                __m256i r0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rep));
                __m256i r1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(rep, 1));
                r0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r0, mul), 6), three);
                r1 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r1, mul), 6), three);
                /* Pack back to 32 ordered codes */
                __m256i codes = _mm256_permute4x64_epi64(_mm256_packus_epi16(r0, r1), 0xD8);
                const int dp = hsum_i32_8(_mm256_dpbusd_epi32(_mm256_setzero_si256(), codes, qy));
                const int sy = hsum_i32_8(_mm256_dpbusd_epi32(_mm256_setzero_si256(), ones,  qy));
                sumi += d1 * (float)(dp - sy);
            }
            sumf += d0 * sumi;
        }
    }
#elif defined(PICOLM_AVX2)
    /* AVX2 path: same 2-bit unpack as VNNI (pshufb + cvtepu8 + mul+shift),
     * but dot product via pmaddubsw + pmaddwd instead of dpbusd.
     * codes in {0,1,2,3} -> sign-extend to int8 -> pmaddubsw with qy.
     * Then subtract dot(1, qy) for the offset. */
    {
        const __m128i idxlo  = _mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
        const __m128i idxhi  = _mm_setr_epi8(4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);
        const __m256i mul    = _mm256_setr_epi16(64,16,4,1, 64,16,4,1, 64,16,4,1, 64,16,4,1);
        const __m256i three  = _mm256_set1_epi16(3);
        const __m256i ones   = _mm256_set1_epi8(1);

        for (int i = 0; i < nb; i++) {
            const float d0 = fp16_to_fp32_lookup(x[i].d);
            float sumi = 0.0f;
            for (int k = 0; k < 4; k++) {
                const block_q8_0 *yb = &y[i * 4 + k];
                const float d1 = fp16_to_fp32_lookup(yb->d);
                const __m256i qy = _mm256_loadu_si256((const __m256i *)yb->qs);
                /* Unpack 8 bytes of qs -> 32 2-bit codes (same as VNNI) */
                const __m128i src = _mm_loadl_epi64((const __m128i *)&x[i].qs[k * 8]);
                const __m256i rep = _mm256_set_m128i(
                    _mm_shuffle_epi8(src, idxhi), _mm_shuffle_epi8(src, idxlo));
                __m256i r0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rep));
                __m256i r1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(rep, 1));
                r0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r0, mul), 6), three);
                r1 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r1, mul), 6), three);
                __m256i codes = _mm256_permute4x64_epi64(_mm256_packus_epi16(r0, r1), 0xD8);
                /* Dot product: codes are unsigned {0,1,2,3}, qy is signed int8.
                 * pmaddubsw: unsigned x signed -> signed 16-bit (16 pairs).
                 * pmaddwd: sum adjacent 16-bit -> 32-bit (8 sums). */
                __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(codes, qy),
                                                _mm256_set1_epi16(1));
                int dp = hsum_i32_8(p);
                /* Subtract offset: dot(1, qy) */
                __m256i sy_p = _mm256_madd_epi16(_mm256_maddubs_epi16(ones, qy),
                                                   _mm256_set1_epi16(1));
                int sy = hsum_i32_8(sy_p);
                sumi += d1 * (float)(dp - sy);
            }
            sumf += d0 * sumi;
        }
    }
#elif defined(PICOLM_FMA)
    /* AVX+FMA path: 128-bit unpack + maddubs. Process one Q2_0 block at a time
     * using 128-bit pshufb for the 2-bit expansion. Two 128-bit halves per block.
     * Uses SSE2 punpcklbw/punpckhbw for byte->16-bit expansion (AVX2-only
     * _mm_cvtepu8_epi16 not available without AVX2). */
    {
        const __m128i idxlo   = _mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
        const __m128i idxhi   = _mm_setr_epi8(4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);
        const __m128i ones_16 = _mm_set1_epi16(1);
        const __m128i ones_8  = _mm_set1_epi8(1);
        const __m128i mul     = _mm_setr_epi16(64,16,4,1, 64,16,4,1, 0,0,0,0, 0,0,0,0);
        const __m128i mulhi   = _mm_setr_epi16(0,0,0,0, 0,0,0,0, 64,16,4,1, 64,16,4,1);
        const __m128i three   = _mm_set1_epi16(3);
        __m256 acc = _mm256_setzero_ps();

        for (int i = 0; i < nb; i++) {
            const float d0 = fp16_to_fp32_lookup(x[i].d);
            __m256 acc_block = _mm256_setzero_ps();
            for (int k = 0; k < 4; k++) {
                const block_q8_0 *yb = &y[i * 4 + k];
                const float d1 = fp16_to_fp32_lookup(yb->d);
                const __m128i src = _mm_loadl_epi64((const __m128i *)&x[i].qs[k * 8]);

                /* Unpack low 4 bytes of qs (16 2-bit codes) -> 16 int8 values */
                {
                    const __m128i rep = _mm_shuffle_epi8(src, idxlo);
                    /* punpcklbw + punpckhbw replaces AVX2-only _mm_cvtepu8_epi16 */
                    __m128i r0 = _mm_unpacklo_epi8(rep, _mm_setzero_si128());
                    r0 = _mm_and_si128(_mm_srli_epi16(_mm_mullo_epi16(r0, mul), 6), three);
                    __m128i r1 = _mm_unpackhi_epi8(rep, _mm_setzero_si128());
                    r1 = _mm_and_si128(_mm_srli_epi16(_mm_mullo_epi16(r1, mulhi), 6), three);
                    __m128i codes = _mm_packus_epi16(r0, r1);
                    /* codes: 16 unsigned bytes {0,1,2,3}; qy0: 16 signed bytes */
                    const __m128i qy0 = _mm_loadl_epi64((const __m128i *)&yb->qs[0]);
                    __m128i p = _mm_madd_epi16(_mm_maddubs_epi16(codes, qy0), ones_16);
                    __m128i sy = _mm_madd_epi16(_mm_maddubs_epi16(ones_8, qy0), ones_16);
                    p = _mm_sub_epi32(p, sy);
                    acc_block = _mm256_fmadd_ps(_mm256_castps128_ps256(_mm_cvtepi32_ps(p)),
                                                 _mm256_set1_ps(d1), acc_block);
                }
                /* Unpack high 4 bytes of qs (16 2-bit codes) -> 16 int8 values */
                {
                    const __m128i rep = _mm_shuffle_epi8(src, idxhi);
                    __m128i r0 = _mm_unpacklo_epi8(rep, _mm_setzero_si128());
                    r0 = _mm_and_si128(_mm_srli_epi16(_mm_mullo_epi16(r0, mul), 6), three);
                    __m128i r1 = _mm_unpackhi_epi8(rep, _mm_setzero_si128());
                    r1 = _mm_and_si128(_mm_srli_epi16(_mm_mullo_epi16(r1, mulhi), 6), three);
                    __m128i codes = _mm_packus_epi16(r0, r1);
                    const __m128i qy0 = _mm_loadl_epi64((const __m128i *)&yb->qs[16]);
                    __m128i p = _mm_madd_epi16(_mm_maddubs_epi16(codes, qy0), ones_16);
                    __m128i sy = _mm_madd_epi16(_mm_maddubs_epi16(ones_8, qy0), ones_16);
                    p = _mm_sub_epi32(p, sy);
                    acc_block = _mm256_fmadd_ps(_mm256_castps128_ps256(_mm_cvtepi32_ps(p)),
                                                 _mm256_set1_ps(d1), acc_block);
                }
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d0), acc_block, acc);
        }
        sumf = hsum_avx(acc);
    }
#elif defined(PICOLM_NEON)
    /* Plain NEON: expand 2-bit codes from qs via vqtbl1q_u8 table lookup,
     * subtract offset 1, multiply by Q8_0, sum with vmull_s8 + vpaddlq_s16.
     * Mirrors llama.cpp approach but without DOTPROD.
     *
     * Each Q2_0 block: 32 bytes qs = 128 2-bit values, matching 4 Q8_0 blocks.
     * Per Q8_0 sub-block: 8 bytes of qs -> 32 2-bit codes -> 32 int8 values. */
    {
        /* Replicate pattern: each of 4 bytes repeated 4 times */
        static const uint8_t tbl_idx_lo[16] = {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3};
        static const uint8_t tbl_idx_hi[16] = {4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7};
        /* Arithmetic right-shift by 0,2,4,6 for each 2-bit field */
        static const int8_t shift_vals[16] = {0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6};
        const uint8x16_t idx_lo = vld1q_u8(tbl_idx_lo);
        const uint8x16_t idx_hi = vld1q_u8(tbl_idx_hi);
        const int8x16_t shifts = vld1q_s8(shift_vals);
        const uint8x16_t mask2 = vdupq_n_u8(3);
        const int8x16_t one = vdupq_n_s8(1);

        for (int i = 0; i < nb; i++) {
            const float d0 = fp16_to_fp32_lookup(x[i].d);
            float sumi = 0.0f;

            for (int k = 0; k < 4; k++) {
                const block_q8_0 *yb = &y[i * 4 + k];
                const float d1 = fp16_to_fp32_lookup(yb->d);

                /* Load 8 bytes of packed 2-bit values, duplicate to fill 16 */
                const uint8x8_t raw = vld1_u8(&x[i].qs[k * 8]);
                const uint8x16_t raw16 = vcombine_u8(raw, raw);

                /* Expand bytes 0-3: replicate each byte 4x, shift, mask -> 16 2-bit codes */
                uint8x16_t bytes0 = vqtbl1q_u8(raw16, idx_lo);
                int8x16_t qv0 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshlq_u8(bytes0, shifts), mask2)),
                    one);

                /* Expand bytes 4-7: replicate each byte 4x, shift, mask -> 16 2-bit codes */
                uint8x16_t bytes1 = vqtbl1q_u8(raw16, idx_hi);
                int8x16_t qv1 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshlq_u8(bytes1, shifts), mask2)),
                    one);

                /* Dot product: vmull_s8 + vpaddlq_s16 (no DOTPROD) */
                const int8x16_t y0 = vld1q_s8(yb->qs);
                const int8x16_t y1 = vld1q_s8(yb->qs + 16);

                /* qv0 . y0 */
                int16x8_t p0 = vmull_s8(vget_low_s8(qv0), vget_low_s8(y0));
                int16x8_t p1 = vmull_s8(vget_high_s8(qv0), vget_high_s8(y0));
                /* qv1 . y1 */
                int16x8_t p2 = vmull_s8(vget_low_s8(qv1), vget_low_s8(y1));
                int16x8_t p3 = vmull_s8(vget_high_s8(qv1), vget_high_s8(y1));

                int32x4_t s0 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
                int32x4_t s1 = vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3));
                int32_t total = vaddvq_s32(vaddq_s32(s0, s1));

                sumi += d1 * (float)total;
            }
            sumf += d0 * sumi;
        }
    }
#else
    /* Scalar fallback */
    for (int ib = 0; ib < nb; ib++) {
        float d0 = fp16_to_fp32_lookup(x[ib].d);
        float sumi = 0.0f;
        for (int K = 0; K < 4; K++) {
            float d1 = fp16_to_fp32_lookup(y[ib * 4 + K].d);
            int sumi_block = 0;
            const uint8_t *qs = &x[ib].qs[K * 8];
            const int8_t *qy = y[ib * 4 + K].qs;
            for (int b = 0; b < 8; b++) {
                uint8_t byte = qs[b];
                sumi_block += ((int)((byte >> 0) & 3) - 1) * qy[b*4 + 0];
                sumi_block += ((int)((byte >> 2) & 3) - 1) * qy[b*4 + 1];
                sumi_block += ((int)((byte >> 4) & 3) - 1) * qy[b*4 + 2];
                sumi_block += ((int)((byte >> 6) & 3) - 1) * qy[b*4 + 3];
            }
            sumi += d1 * (float)sumi_block;
        }
        sumf += d0 * sumi;
    }
#endif
    return sumf;
}

/* vec_dot for Q4_0_4_4: dequantize row 0 of interleaved block, then dot
 * Interleaving: qs[k*16 + r*4 + j] = row_r.qs[k*4 + j] ^ 0x88
 * For row 0: qs[k*16 + j] for k=0..3, j=0..3 */
float vec_dot_q4_0_4_4_f32(const void *src, const float *x, int n) {
    const block_q4_0x4 *blocks = (const block_q4_0x4 *)src;
    int nb = n / 32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d[0]);
        const float *xp = x + i * 32;
        float block_sum = 0.0f;
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 4; j++) {
                uint8_t byte = blocks[i].qs[k * 16 + j];
                int v0 = (int8_t)(byte << 4) >> 4;
                int v1 = (int8_t)(byte & 0xF0) >> 4;
                block_sum += (float)v0 * xp[k * 8 + j * 2];
                block_sum += (float)v1 * xp[k * 8 + j * 2 + 1];
            }
        }
        sumf += d * block_sum;
    }
    return sumf;
}

/* vec_dot for Q4_0_4_8: dequantize row 0 of interleaved block (blocklen=8), then dot
 * Interleaving: qs[k*32 + r*8 + j] = row_r.qs[k*8 + j] ^ 0x88
 * For row 0: qs[k*32 + j] for k=0..1, j=0..7 */
float vec_dot_q4_0_4_8_f32(const void *src, const float *x, int n) {
    const block_q4_0x4 *blocks = (const block_q4_0x4 *)src;
    int nb = n / 32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d[0]);
        const float *xp = x + i * 32;
        float block_sum = 0.0f;
        for (int k = 0; k < 2; k++) {
            for (int j = 0; j < 8; j++) {
                uint8_t byte = blocks[i].qs[k * 32 + j];
                int v0 = (int8_t)(byte << 4) >> 4;
                int v1 = (int8_t)(byte & 0xF0) >> 4;
                block_sum += (float)v0 * xp[k * 16 + j * 2];
                block_sum += (float)v1 * xp[k * 16 + j * 2 + 1];
            }
        }
        sumf += d * block_sum;
    }
    return sumf;
}

/* vec_dot for Q4_0_8_8: dequantize row 0 of interleaved block, then dot product */
float vec_dot_q4_0_8_8_f32(const void *src, const float *x, int n) {
    const block_q4_0x8 *blocks = (const block_q4_0x8 *)src;
    int nb = n / 32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d[0]);
        const float *xp = x + i * 32;
        float block_sum = 0.0f;
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 8; j++) {
                uint8_t byte = blocks[i].qs[k * 128 + j * 8];
                int v0 = (int8_t)(byte << 4) >> 4;
                int v1 = (int8_t)(byte & 0xF0) >> 4;
                block_sum += (float)v0 * xp[k * 8 + j];
                block_sum += (float)v1 * xp[k * 8 + j + 4];
            }
        }
        sumf += d * block_sum;
    }
    return sumf;
}

/* ================================================================
 * vec_dot_q2_K_q8_K: int8 MAC for Q2_K weights * Q8_K input
 * Adapted from llama.cpp ggml_vec_dot_q2_K_q8_K (arm NEON path + generic).
 *
 * Q2_K block: 256 values, 84 bytes
 *   scales[16] - packed 4-bit scales+mins (lower 4 = scale, upper 4 = min)
 *   qs[64]     - 2-bit quantized values (2 bits * 64 * 2 chunks = 256)
 *   d          - super-block scale (FP16)
 *   dmin       - super-block min   (FP16)
 *
 * Dequant formula: val = d * scale * q - dmin * min
 * where q in {0,1,2,3} and scale/min are 4-bit each.
 *
 * Dot product: sum(d * scale * q * q8 - dmin * min * q8)
 *            = d * sum(scale * (q*q8)) - dmin * sum(min * sum(q8_per_group))
 *
 * The second term uses bsums (precomputed sum of q8 in groups of 16).
 * ================================================================ */
float vec_dot_q2_K_q8_K(const void *src_q2, const void *src_q8, int n) {
    const block_q2_K *x = (const block_q2_K *)src_q2;
    const block_q8_K *y = (const block_q8_K *)src_q8;
    const int nb = n / 256;
    float sumf = 0.0f;

#ifdef PICOLM_I8MM
    /* I8MM: vmmlaq_s32 for 16x int8 MAC -> 4x int32 lanes per call.
     * Each Q2_K block = 256 values in 2 chunks of 128.
     * Per chunk: 4 shifts x 2 halves (16 values each) = 8 sub-blocks of 16.
     * vmmlaq_s32: 16 int8 x int8 -> 4 int32 lanes.
     * Lanes 0+3 = full 16-element dot product.
     * One vmmlaq call per 16-element group. */
    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const uint8x16_t m4 = vdupq_n_u8(0xF);

    for (int i = 0; i < nb; ++i) {
        const float d    = fp16_to_fp32_lookup(x[i].d)    * y[i].d;
        const float dmin = fp16_to_fp32_lookup(x[i].dmin) * y[i].d;

        const uint8_t *q2 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        /* Extract scales (lower 4 bits) */
        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales_v = vandq_u8(mins_and_scales, m4);
        const int8x16_t  scales_s8 = vreinterpretq_s8_u8(scales_v);

        /* Min correction: dmin * sum(bsums[j] * min[j]) */
        const uint8x16_t mins_v = vshrq_n_u8(mins_and_scales, 4);
        const int16x4_t b0 = vld1_s16(y[i].bsums);
        const int16x4_t b1 = vld1_s16(y[i].bsums + 4);
        const int16x4_t b2 = vld1_s16(y[i].bsums + 8);
        const int16x4_t b3 = vld1_s16(y[i].bsums + 12);
        const int32x4_t mm0 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins_v)))), b0);
        const int32x4_t mm1 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins_v)))), b1);
        const int32x4_t mm2 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins_v)))), b2);
        const int32x4_t mm3 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins_v)))), b3);
        const int32x4_t msum = vaddq_s32(vaddq_s32(mm0, mm1), vaddq_s32(mm2, mm3));
        const int summs = vaddvq_s32(msum);

        int isum = 0;
        int is = 0;
        for (int chunk = 0; chunk < 2; ++chunk) {
            uint8x16_t q2a = vld1q_u8(q2);
            uint8x16_t q2b = vld1q_u8(q2 + 16);
            q2 += 32;

            for (int shift_i = 0; shift_i < 4; ++shift_i) {
                const int8x16_t qa = vreinterpretq_s8_u8(vandq_u8(q2a, m3));
                const int8x16_t qb = vreinterpretq_s8_u8(vandq_u8(q2b, m3));
                q2a = vshrq_n_u8(q2a, 2);
                q2b = vshrq_n_u8(q2b, 2);

                const int8x16_t q8a = vld1q_s8(q8);
                const int8x16_t q8b = vld1q_s8(q8 + 16);
                q8 += 32;

                int32x4_t s0 = vmmlaq_s32(vdupq_n_s32(0), qa, q8a);
                int32x4_t s1 = vmmlaq_s32(vdupq_n_s32(0), qb, q8b);
                int dot_a = vgetq_lane_s32(s0, 0) + vgetq_lane_s32(s0, 3);
                int dot_b = vgetq_lane_s32(s1, 0) + vgetq_lane_s32(s1, 3);

                isum += dot_a * (int)scales_s8[is] + dot_b * (int)scales_s8[is + 1];
                is += 2;
            }
        }

        sumf += d * (float)isum - dmin * (float)summs;
    }

#elif defined(PICOLM_DOTPROD)
    /* ARMv8.2 dotprod: vdotq_s32 for 16x int8 MAC -> 1 int32 accumulation.
     * Each 16-element group uses one vdotq_s32 call.
     * Pattern mirrors llama.cpp's NEON dotprod Q2_K path. */
    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const uint8x16_t m4 = vdupq_n_u8(0xF);

    for (int i = 0; i < nb; ++i) {
        const float d    = fp16_to_fp32_lookup(x[i].d)    * y[i].d;
        const float dmin = fp16_to_fp32_lookup(x[i].dmin) * y[i].d;

        const uint8_t *q2 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales_v = vandq_u8(mins_and_scales, m4);
        const int8x16_t  scales_s8 = vreinterpretq_s8_u8(scales_v);

        const uint8x16_t mins_v = vshrq_n_u8(mins_and_scales, 4);
        const int16x4_t b0 = vld1_s16(y[i].bsums);
        const int16x4_t b1 = vld1_s16(y[i].bsums + 4);
        const int16x4_t b2 = vld1_s16(y[i].bsums + 8);
        const int16x4_t b3 = vld1_s16(y[i].bsums + 12);
        const int32x4_t mm0 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins_v)))), b0);
        const int32x4_t mm1 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins_v)))), b1);
        const int32x4_t mm2 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins_v)))), b2);
        const int32x4_t mm3 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins_v)))), b3);
        const int32x4_t msum = vaddq_s32(vaddq_s32(mm0, mm1), vaddq_s32(mm2, mm3));
        const int summs = vaddvq_s32(msum);

        int isum = 0;
        int is = 0;
        const int32x4_t vzero = vdupq_n_s32(0);

        for (int chunk = 0; chunk < 2; ++chunk) {
            uint8x16_t q2a = vld1q_u8(q2);
            uint8x16_t q2b = vld1q_u8(q2 + 16);
            q2 += 32;

            for (int shift_i = 0; shift_i < 4; ++shift_i) {
                const int8x16_t qa = vreinterpretq_s8_u8(vandq_u8(q2a, m3));
                const int8x16_t qb = vreinterpretq_s8_u8(vandq_u8(q2b, m3));
                q2a = vshrq_n_u8(q2a, 2);
                q2b = vshrq_n_u8(q2b, 2);

                const int8x16_t q8a = vld1q_s8(q8);
                const int8x16_t q8b = vld1q_s8(q8 + 16);
                q8 += 32;

                int dot_a = vaddvq_s32(vdotq_s32(vzero, qa, q8a));
                int dot_b = vaddvq_s32(vdotq_s32(vzero, qb, q8b));

                isum += dot_a * (int)scales_s8[is] + dot_b * (int)scales_s8[is + 1];
                is += 2;
            }
        }

        sumf += d * (float)isum - dmin * (float)summs;
    }

#elif defined(PICOLM_NEON)
    /* Plain NEON (no dotprod, no I8MM): vmull_s8 + vpaddlq_s16.
     * Same algorithm as above but without dotprod or I8MM instructions. */
    const uint8x16_t m3  = vdupq_n_u8(0x3);
    const uint8x16_t m4  = vdupq_n_u8(0xF);

    for (int i = 0; i < nb; ++i) {
        const float d    = fp16_to_fp32_lookup(x[i].d)    * y[i].d;
        const float dmin = fp16_to_fp32_lookup(x[i].dmin) * y[i].d;

        const uint8_t *q2 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales_v = vandq_u8(mins_and_scales, m4);
        const int8x16_t  scales_s8 = vreinterpretq_s8_u8(scales_v);

        const uint8x16_t mins_v = vshrq_n_u8(mins_and_scales, 4);
        const int16x4_t b0 = vld1_s16(y[i].bsums);
        const int16x4_t b1 = vld1_s16(y[i].bsums + 4);
        const int16x4_t b2 = vld1_s16(y[i].bsums + 8);
        const int16x4_t b3 = vld1_s16(y[i].bsums + 12);
        const int32x4_t mm0 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins_v)))), b0);
        const int32x4_t mm1 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins_v)))), b1);
        const int32x4_t mm2 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins_v)))), b2);
        const int32x4_t mm3 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins_v)))), b3);
        const int32x4_t msum = vaddq_s32(vaddq_s32(mm0, mm1), vaddq_s32(mm2, mm3));
        const int summs = vaddvq_s32(msum);

        int isum = 0;
        int is = 0;
        for (int chunk = 0; chunk < 2; ++chunk) {
            uint8x16_t q2a = vld1q_u8(q2);
            uint8x16_t q2b = vld1q_u8(q2 + 16);
            q2 += 32;

            for (int shift_i = 0; shift_i < 4; ++shift_i) {
                const int8x16_t qa = vreinterpretq_s8_u8(vandq_u8(q2a, m3));
                const int8x16_t qb = vreinterpretq_s8_u8(vandq_u8(q2b, m3));
                q2a = vshrq_n_u8(q2a, 2);
                q2b = vshrq_n_u8(q2b, 2);

                const int8x16_t q8a = vld1q_s8(q8);
                const int8x16_t q8b = vld1q_s8(q8 + 16);
                q8 += 32;

                const int16x8_t p0a = vmull_s8(vget_low_s8(qa), vget_low_s8(q8a));
                const int16x8_t p1a = vmull_s8(vget_high_s8(qa), vget_high_s8(q8a));
                const int32x4_t s0a = vaddq_s32(vpaddlq_s16(p0a), vpaddlq_s16(p1a));
                int dot_a = vaddvq_s32(s0a);

                const int16x8_t p0b = vmull_s8(vget_low_s8(qb), vget_low_s8(q8b));
                const int16x8_t p1b = vmull_s8(vget_high_s8(qb), vget_high_s8(q8b));
                const int32x4_t s0b = vaddq_s32(vpaddlq_s16(p0b), vpaddlq_s16(p1b));
                int dot_b = vaddvq_s32(s0b);

                isum += dot_a * (int)scales_s8[is] + dot_b * (int)scales_s8[is + 1];
                is += 2;
            }
        }

        sumf += d * (float)isum - dmin * (float)summs;
    }

#elif defined(PICOLM_AVX2)
    /* AVX2 path: 256-bit SIMD for Q2_K dot product.
     * Extracts 4 shifts of 2-bit quants (0,2,4,6) from 32 bytes per iteration,
     * processes with maddubs+madd_epi16. Adapted from llama.cpp. */
    {
        const __m256i m3 = _mm256_set1_epi8(3);
        const __m128i m4 = _mm_set1_epi8(0xF);

        __m256 acc = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            const float d = y[i].d * fp16_to_fp32_lookup(x[i].d);
            const float dmin = -y[i].d * fp16_to_fp32_lookup(x[i].dmin);

            const uint8_t *q2 = x[i].qs;
            const int8_t  *q8 = y[i].qs;

            /* Load all 16 scale+min bytes at once */
            const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
            const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
            const __m128i mins8   = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);

            /* Min correction: dmin * sum(bsums[j] * mins[j]) */
            const __m256i mins = _mm256_cvtepi8_epi16(mins8);
            const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y[i].bsums));
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

            /* Prepare scales: convert 16 scale bytes to 16 int16, split into
             * low 8 and high 8, each duplicated into a __m256i for shuffling. */
            const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
            const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
            const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
            const __m256i scales[2] = {
                _mm256_set_m128i(l_scales, l_scales),
                _mm256_set_m128i(h_scales, h_scales),
            };

            __m256i sumi = _mm256_setzero_si256();

            /* 256 q2 values = 64 bytes = 2 chunks of 32 bytes.
             * Each chunk: 4 shifts (bits 0,2,4,6), 32 q2 values per shift. */
            for (int j = 0; j < 2; ++j) {
                const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2);
                q2 += 32;

                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

                const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
                const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
                const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
                const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

                __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
                __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
                __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
                __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

                /* Shuffle scales: same pattern for both chunks, indexed by shift only.
                 * Each shift uses 2 consecutive scales (one per 16-element half). */
                p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q2k(0)), p0);
                p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q2k(1)), p1);
                p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q2k(2)), p2);
                p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q2k(3)), p3);

                p0 = _mm256_add_epi32(p0, p1);
                p2 = _mm256_add_epi32(p2, p3);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
            }

            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
        }

        sumf = hsum_avx(acc);
    }

#elif defined(PICOLM_AVX)
    /* AVX1 path: 128-bit integer + 256-bit float accumulation.
     * Processes 16 q2 bytes per iteration (8 shifts of 2-bit quants).
     * Adapted from llama.cpp. */
    {
        const __m128i m3 = _mm_set1_epi8(0x3);
        const __m128i m4 = _mm_set1_epi8(0xF);
        const __m128i m2 = _mm_set1_epi8(0x2);

        __m256 acc = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            const float dall = y[i].d * fp16_to_fp32_lookup(x[i].d);
            const float dmin = -y[i].d * fp16_to_fp32_lookup(x[i].dmin);

            const uint8_t *q2 = x[i].qs;
            const int8_t  *q8 = y[i].qs;

            /* Min correction */
            const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
            const __m128i scales16 = _mm_and_si128(mins_and_scales, m4);
            const __m128i mins16 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
            const __m128i mins_0 = _mm_cvtepi8_epi16(mins16);
            const __m128i mins_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(mins16, mins16));
            const __m128i summs_0 = _mm_madd_epi16(mins_0, _mm_loadu_si128((const __m128i*)&y[i].bsums[0]));
            const __m128i summs_1 = _mm_madd_epi16(mins_1, _mm_loadu_si128((const __m128i*)&y[i].bsums[8]));
            acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dmin),
                    _mm256_cvtepi32_ps(_mm256_set_m128i(summs_1, summs_0))), acc);

            /* Prepare scales */
            const __m128i scales_0 = _mm_cvtepi8_epi16(scales16);
            const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales16, scales16));
            const __m128i scales[2] = {scales_0, scales_1};

            __m128i sumi_0 = _mm_setzero_si128();
            __m128i sumi_1 = _mm_setzero_si128();

            for (int j = 0; j < 2; ++j) {
                /* Load 32 q2 bytes as 2x16, extract 8 shifts total */
                __m128i q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
                const __m128i q2_0 = _mm_and_si128(q2bits, m3);
                const __m128i q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
                const __m128i q2_4 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
                const __m128i q2_6 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);
                q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
                const __m128i q2_1 = _mm_and_si128(q2bits, m3);
                const __m128i q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
                const __m128i q2_5 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
                const __m128i q2_7 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);

                /* Load 8x16 q8 bytes */
                const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
                const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
                const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
                const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
                const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
                const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
                const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
                const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

                __m128i p0 = _mm_maddubs_epi16(q2_0, q8_0);
                __m128i p1 = _mm_maddubs_epi16(q2_1, q8_1);
                __m128i p2 = _mm_maddubs_epi16(q2_2, q8_2);
                __m128i p3 = _mm_maddubs_epi16(q2_3, q8_3);
                __m128i p4 = _mm_maddubs_epi16(q2_4, q8_4);
                __m128i p5 = _mm_maddubs_epi16(q2_5, q8_5);
                __m128i p6 = _mm_maddubs_epi16(q2_6, q8_6);
                __m128i p7 = _mm_maddubs_epi16(q2_7, q8_7);

                /* Shuffle scales: 0x0100 pattern, stepping by 0x0200 */
                __m128i shuffle = _mm_set1_epi16(0x0100);
                p0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p0);
                shuffle = _mm_add_epi16(shuffle, m2);
                p1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p1);
                shuffle = _mm_add_epi16(shuffle, m2);
                p2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p2);
                shuffle = _mm_add_epi16(shuffle, m2);
                p3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p3);
                shuffle = _mm_add_epi16(shuffle, m2);
                p4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p4);
                shuffle = _mm_add_epi16(shuffle, m2);
                p5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p5);
                shuffle = _mm_add_epi16(shuffle, m2);
                p6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p6);
                shuffle = _mm_add_epi16(shuffle, m2);
                p7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p7);

                p0 = _mm_add_epi32(p0, p1);
                p2 = _mm_add_epi32(p2, p3);
                p4 = _mm_add_epi32(p4, p5);
                p6 = _mm_add_epi32(p6, p7);

                sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p0, p2));
                sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p4, p6));
            }

            __m256i sumi = _mm256_set_m128i(sumi_1, sumi_0);
            acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dall),
                    _mm256_cvtepi32_ps(sumi)), acc);
        }

        sumf = hsum_avx(acc);
    }

#else
    /* Scalar fallback (mirrors llama.cpp generic path) */
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q2 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (int)(sc[j] >> 4);
        }

        const float d    = fp16_to_fp32_lookup(x[i].d)    * y[i].d;
        const float dmin = fp16_to_fp32_lookup(x[i].dmin) * y[i].d;

        int isum = 0;
        int is = 0;
        for (int chunk = 0; chunk < 2; ++chunk) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                int s = sc[is++] & 0xF;
                int isuml = 0;
                for (int l = 0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += s * isuml;

                s = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 0; l < 16; ++l) isuml += q8[l + 16] * ((q2[l + 16] >> shift) & 3);
                isum += s * isuml;

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }

        sumf += d * (float)isum - dmin * (float)summs;
    }
#endif
    return sumf;
}

/* ================================================================
 * vec_dot_q2_K_f32: Q2_K weights x float32 activations
 * Pre-quantizes activations to Q8_K and delegates to vec_dot_q2_K_q8_K.
 * ================================================================ */
float vec_dot_q2_K_f32(const void *src, const float *x, int n) {
    /* Pre-quantize x to Q8_K and delegate to vec_dot_q2_K_q8_K.
     * For small n, quantization overhead dominates; fall back to scalar. */
    if (n >= 256 && n % 256 == 0) {
        size_t nq8 = (size_t)(n / 256) * sizeof(block_q8_K);
        /* Use stack buffer for small sizes to avoid malloc overhead */
        block_q8_K qx_buf[2];
        block_q8_K *qx;
        int qx_owned = 0;

        if (nq8 <= sizeof(qx_buf)) {
            qx = qx_buf;
        } else {
            qx = (block_q8_K *)malloc(nq8);
            if (!qx) goto scalar_path;
            qx_owned = 1;
        }
        quantize_row_q8_K(x, qx, n);
        float result = vec_dot_q2_K_q8_K(src, qx, n);
        if (qx_owned) free(qx);
        return result;

    scalar_path:
        ;
    }

    /* Scalar fallback: dequantize Q2_K to float, then dot */
    const block_q2_K *blocks = (const block_q2_K *)src;
    int nb = n / 256;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d    = fp16_to_fp32_lookup(blocks[i].d);
        float dmin = fp16_to_fp32_lookup(blocks[i].dmin);
        const uint8_t *q = blocks[i].qs;
        const float *xp = x + i * 256;
        float block_sum = 0.0f;
        int is = 0;
        float dl, ml;
        for (int chunk = 0; chunk < 256; chunk += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = blocks[i].scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) block_sum += (dl * (int8_t)((q[l] >> shift) & 3) - ml) * xp[l];
                sc = blocks[i].scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) block_sum += (dl * (int8_t)((q[l+16] >> shift) & 3) - ml) * xp[l + 16];
                shift += 2;
                xp += 32;
            }
            q += 32;
        }
        sumf += block_sum;
    }
    return sumf;
}

/* ---- Generic dispatch ---- */

float vec_dot(const void *src, const float *x, int n, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_Q4_K: return vec_dot_q4_K_f32(src, x, n);
        case GGUF_TYPE_Q5_K: {
            /* Scalar fallback: dequantize to float, then dot */
#if defined(_WIN32)
            /* MSVC has no __thread; use plain static */
            static float q5_tmp[4096];
#elif defined(__APPLE__)
            /* __thread not supported on old Mac OS X; use static buffer */
            static float q5_tmp[4096];
#else
            static __thread float q5_tmp[4096];
#endif
            if (n > 4096) {
                float *tmp = (float *)malloc((size_t)n * sizeof(float));
                float r;
                dequantize_row_q5_K(src, tmp, n);
                r = vec_dot_f32_f32(tmp, x, n);
                free(tmp);
                return r;
            }
            dequantize_row_q5_K(src, q5_tmp, n);
            return vec_dot_f32_f32(q5_tmp, x, n);
        }
        case GGUF_TYPE_Q6_K: return vec_dot_q6_K_f32(src, x, n);
        case GGUF_TYPE_F32:  return vec_dot_f32_f32(src, x, n);
        case GGUF_TYPE_Q8_0: return vec_dot_q8_0_f32(src, x, n);
        case GGUF_TYPE_Q4_0: return vec_dot_q4_0_f32(src, x, n);
        case GGUF_TYPE_Q4_1: return vec_dot_q4_1_f32(src, x, n);
        case GGUF_TYPE_Q1_0: return vec_dot_q1_0_f32(src, x, n);
        case GGUF_TYPE_Q2_0: return vec_dot_q2_0_f32(src, x, n);
        case GGUF_TYPE_Q2_K:  return vec_dot_q2_K_f32(src, x, n);
        case GGUF_TYPE_Q4_0_4_4: return vec_dot_q4_0_4_4_f32(src, x, n);
        case GGUF_TYPE_Q4_0_4_8: return vec_dot_q4_0_4_8_f32(src, x, n);
        case GGUF_TYPE_Q4_0_8_8: return vec_dot_q4_0_8_8_f32(src, x, n);
        case GGUF_TYPE_F16:  return vec_dot_f16_f32(src, x, n);
        case GGUF_TYPE_BF16: return vec_dot_bf16_f32(src, x, n);
        default: {
            /* Fallback: dequantize to temp buffer, then dot */
            float tmp[8192];
            float *buf = (n <= 8192) ? tmp : (float *)malloc((size_t)n * sizeof(float));
            dequantize_row(src, buf, n, type);
            float sum = vec_dot_f32_f32(buf, x, n);
            if (buf != tmp) free(buf);
            return sum;
        }
    }
}

/* ---- repack_q4_0_to_q4_0x8: Standard Q4_0 -> Q4_0_8x8 interleaved (for AVX2) ----
 * Ported from ggml's make_block_q4_0x8() (ggml/src/ggml-cpu/repack.cpp), verified
 * against the real algorithm rather than reconstructed from memory: for chunk
 * index i in 0..15, src_id = i%8 selects which of the 8 source rows, src_offset
 * = (i/8)*8 selects the first or second half of that row's 16-byte qs, and the
 * resulting 8 bytes are copied to out.qs[i*8 .. i*8+8), XOR'd with 0x88.
 *
 * XOR 0x88 flips bit 3 of each nibble (v -> v^8), which combined with the
 * signextendlut used by the dot-product kernel (LUT[v] = v for v<8, v-16 for
 * v>=8) reproduces Q4_0's (nibble-8) dequant exactly: LUT[v^8] = v-8 for all
 * v in 0..15. Confirmed algebraically before use, not just copied.
 *
 * Precondition: nrows % 8 == 0, ncols % 32 == 0.
 * ================================================================ */
void repack_q4_0_to_q4_0x8(const void *src, void *dst, int nrows, int ncols) {
    const block_q4_0 *s = (const block_q4_0 *)src;
    block_q4_0x8 *d = (block_q4_0x8 *)dst;
    int nb = ncols / 32;  /* blocks per row */
    const uint64_t xor_mask = 0x8888888888888888ULL;

    for (int row8 = 0; row8 < nrows; row8 += 8) {
        for (int b = 0; b < nb; b++) {
            const block_q4_0 *in = s + row8 * nb + b; /* in[r] = row (row8+r)'s block b, stride nb */
            for (int r = 0; r < 8; r++) {
                d->d[r] = in[r * nb].d;
            }
            for (int i = 0; i < 16; i++) {
                int src_id = i % 8;
                int src_offset = (i / 8) * 8;
                uint64_t elems;
                memcpy(&elems, &in[src_id * nb].qs[src_offset], sizeof(uint64_t));
                elems ^= xor_mask;
                memcpy(&d->qs[i * 8], &elems, sizeof(uint64_t));
            }
            d++;
        }
    }
}

/* ---- repack_q4_0_to_q4_0x4: Standard Q4_0 -> Q4_0_4x4 interleaved ----
 * Same as 8x8 but with 4 rows and blocklen=4.
 * ================================================================ */
void repack_q4_0_to_q4_0x4(const void *src, void *dst, int nrows, int ncols) {
    const block_q4_0 *s = (const block_q4_0 *)src;
    block_q4_0x4 *d = (block_q4_0x4 *)dst;
    int nb = ncols / 32;

    for (int row4 = 0; row4 < nrows; row4 += 4) {
        for (int b = 0; b < nb; b++) {
            for (int r = 0; r < 4; r++) {
                d->d[r] = s[b + (row4 + r) * nb].d;
            }
            for (int k = 0; k < 4; k++) {
                for (int r = 0; r < 4; r++) {
                    const uint8_t *src_qs = ((const block_q4_0 *)(s + b + (row4 + r) * nb))->qs;
                    for (int j = 0; j < 4; j++) {
                        d->qs[k * 16 + r * 4 + j] = src_qs[k * 4 + j] ^ 0x88;
                    }
                }
            }
            d++;
        }
    }
}

/* ---- vec_dot_q4_0x8_q8_0_avx2: Q4_0_8x8 interleaved x Q8_0 (AVX2) ----
 * Processes 8 output rows simultaneously using 256-bit AVX2 registers.
 * Adapted from llama.cpp's gemv_q4_b32_8x8_q8_0_lut_avx.
 *
 * Uses a lookup table to convert 4-bit nibbles to signed 8-bit values via
 * _mm256_shuffle_epi8 (PSHUFB), avoiding manual bit manipulation.
 *
 * Precondition: nrows % 8 == 0. Remaining rows handled by scalar fallback.
 * ================================================================ */
void vec_dot_q4_0x8_q8_0_avx2(const void *vx, const void *wy, int n, float *out, int nrows) {
#ifdef PICOLM_AVX2
    const block_q4_0x8 *b_ptr_start = (const block_q4_0x8 *)vx;
    const block_q8_0 *a_ptr = (const block_q8_0 *)wy;
    int nb = n / 32;

    /* Lookup table: maps 4-bit nibble to signed byte [-8..7] */
    __m256i signextendlut = _mm256_castsi128_si256(
        _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

    /* Final permute to reorder output lanes to correct row order */
    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    const __m256i m4b = _mm256_set1_epi8(0x0F);

    /* Rearrange mask for loading 8 FP16 deltas in the correct order */
    const __m128i changemask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4,
                                             11, 10, 3, 2, 9, 8, 1, 0);

    /* Process groups of 8 output rows */
    for (int y = 0; y < nrows / 8; y++) {
        const block_q4_0x8 *b_ptr = b_ptr_start + y * nb;

        /* Accumulate all 8 rows simultaneously into one __m256 */
        __m256 acc = _mm256_setzero_ps();

        for (int b = 0; b < nb; b++) {
            /* Load 4x256-bit chunks of interleaved qs (8 rows x 32 nibbles) */
            const __m256i rhs0_0 = _mm256_loadu_si256((const __m256i *)b_ptr[b].qs);
            const __m256i rhs1_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
            const __m256i rhs0_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
            const __m256i rhs1_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

            /* Low nibbles via LUT (4-bit -> signed 8-bit) */
            const __m256i r00 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs0_0, m4b));
            const __m256i r10 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs1_0, m4b));
            const __m256i r01 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs0_1, m4b));
            const __m256i r11 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs1_1, m4b));

            /* High nibbles via right-shift + LUT */
            const __m256i r02 = _mm256_shuffle_epi8(signextendlut,
                _mm256_and_si256(_mm256_srli_epi16(rhs0_0, 4), m4b));
            const __m256i r12 = _mm256_shuffle_epi8(signextendlut,
                _mm256_and_si256(_mm256_srli_epi16(rhs1_0, 4), m4b));
            const __m256i r03 = _mm256_shuffle_epi8(signextendlut,
                _mm256_and_si256(_mm256_srli_epi16(rhs0_1, 4), m4b));
            const __m256i r13 = _mm256_shuffle_epi8(signextendlut,
                _mm256_and_si256(_mm256_srli_epi16(rhs1_1, 4), m4b));

            /* Load 8 column scales (FP16 -> FP32), rearranged for AVX2 lanes */
            __m256 col_scales = _mm256_cvtph_ps(
                _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)b_ptr[b].d), changemask));

            /* Load Q8_0 input: duplicate to fill 256-bit (a0=a[0..15], a1=a[16..31]) */
            __m256i a0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)a_ptr[b].qs));
            __m256i a1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16)));
            a0 = _mm256_permute2f128_si256(a0, a0, 0);
            a1 = _mm256_permute2f128_si256(a1, a1, 0);

            /* Row scale (Q8_0 delta) */
            __m256 row_scale = _mm256_set1_ps(fp16_to_fp32_lookup(a_ptr[b].d));

            /* Combine column and row scales */
            __m256 sd = _mm256_mul_ps(col_scales, row_scale);

            /* Int32 accumulator for 8 rows */
            __m256i iacc = _mm256_setzero_si256();

            /* Signed x signed int8 dot-product-and-widen, matching ggml's
             * mul_sum_i8_pairs_acc_int32x8: maddubs needs an unsigned first
             * operand, so take abs(weight-nibble) and copy weight's sign
             * onto the activation byte first, then reduce 16-bit pairs to
             * int32 via madd_epi16 with a vector of ones. (The previous
             * version of this kernel skipped the abs/sign step entirely --
             * passing signed nibbles straight into maddubs, which requires
             * an unsigned first operand -- and then tried to recover
             * correct int32 sums via an ad hoc hi/lo 16-bit shift-and-add
             * that doesn't correspond to any real reduction of maddubs'
             * actual output width. That's what produced all-zero results.) */
            #define Q4_0X8_MULSUM(bvec, avec) \
                _mm256_add_epi32(iacc, _mm256_madd_epi16(_mm256_set1_epi16(1), \
                    _mm256_maddubs_epi16(_mm256_sign_epi8(bvec, bvec), _mm256_sign_epi8(avec, bvec))))

            /* Low nibbles: interleave 8 rows' nibbles with a0[0..7] and a0[8..15] */
            {
                __m256i bl = _mm256_blend_epi32(r00, _mm256_shuffle_epi32(r10, 177), 170);
                __m256i bh = _mm256_blend_epi32(_mm256_shuffle_epi32(r00, 177), r10, 170);
                iacc = Q4_0X8_MULSUM(bl, _mm256_shuffle_epi32(a0, 0));
                iacc = Q4_0X8_MULSUM(bh, _mm256_shuffle_epi32(a0, 85));

                bl = _mm256_blend_epi32(r01, _mm256_shuffle_epi32(r11, 177), 170);
                bh = _mm256_blend_epi32(_mm256_shuffle_epi32(r01, 177), r11, 170);
                iacc = Q4_0X8_MULSUM(bl, _mm256_shuffle_epi32(a0, 170));
                iacc = Q4_0X8_MULSUM(bh, _mm256_shuffle_epi32(a0, 255));
            }
            /* High nibbles: interleave 8 rows' nibbles with a1[0..7] and a1[8..15] */
            {
                __m256i bl = _mm256_blend_epi32(r02, _mm256_shuffle_epi32(r12, 177), 170);
                __m256i bh = _mm256_blend_epi32(_mm256_shuffle_epi32(r02, 177), r12, 170);
                iacc = Q4_0X8_MULSUM(bl, _mm256_shuffle_epi32(a1, 0));
                iacc = Q4_0X8_MULSUM(bh, _mm256_shuffle_epi32(a1, 85));

                bl = _mm256_blend_epi32(r03, _mm256_shuffle_epi32(r13, 177), 170);
                bh = _mm256_blend_epi32(_mm256_shuffle_epi32(r03, 177), r13, 170);
                iacc = Q4_0X8_MULSUM(bl, _mm256_shuffle_epi32(a1, 170));
                iacc = Q4_0X8_MULSUM(bh, _mm256_shuffle_epi32(a1, 255));
            }
            #undef Q4_0X8_MULSUM

            /* Scale and accumulate */
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), sd, acc);
        }

        /* Permute to correct order and store */
        __m256 result = _mm256_permutevar8x32_ps(acc, finalpermutemask);
        _mm256_storeu_ps(out + y * 8, result);
    }

    /* Scalar fallback for remaining rows (nrows % 8 != 0) */
    {
        int aligned = (nrows / 8) * 8;
        for (int row = aligned; row < nrows; row++) {
            int group = row / 8;
            int r = row % 8;
            float sumf = 0.0f;
            for (int b = 0; b < nb; b++) {
                const block_q4_0x8 *bp = &((const block_q4_0x8 *)vx)[b + group * nb];
                float dd = fp16_to_fp32_lookup(bp->d[r]) * fp16_to_fp32_lookup(a_ptr[b].d);
                int sumi = 0;
                /* Row r's original 16-byte qs is split across this block's
                 * qs[r*8 .. r*8+8) (original bytes 0-7) and
                 * qs[64+r*8 .. 64+r*8+8) (original bytes 8-15) -- see
                 * repack_q4_0_to_q4_0x8. Standard Q4_0 packing: byte j's low
                 * nibble is value[j], high nibble is value[j+16]. So the
                 * first half's low/high nibbles are value[0..7]/value[16..23],
                 * and the second half's are value[8..15]/value[24..31] --
                 * NOT value[i]/value[i+8] as a naive reading might suggest. */
                for (int i = 0; i < 8; i++) {
                    uint8_t byte = bp->qs[r * 8 + i];
                    int v0 = (int8_t)(byte << 4) >> 4; /* low nibble, sign-extended: value[i] */
                    int v1 = (int8_t)(byte & 0xF0) >> 4; /* high nibble, sign-extended: value[i+16] */
                    sumi += v0 * a_ptr[b].qs[i] + v1 * a_ptr[b].qs[i + 16];
                }
                for (int i = 0; i < 8; i++) {
                    uint8_t byte = bp->qs[64 + r * 8 + i];
                    int v0 = (int8_t)(byte << 4) >> 4; /* value[8+i] */
                    int v1 = (int8_t)(byte & 0xF0) >> 4; /* value[24+i] */
                    sumi += v0 * a_ptr[b].qs[8 + i] + v1 * a_ptr[b].qs[24 + i];
                }
                sumf += (float)sumi * dd;
            }
            out[row] = sumf;
        }
    }
#else
    /* Non-AVX2: scalar fallback processing all rows from Q4_0x8 data.
     * See the AVX2 branch's remainder-tail comment for the nibble/activation
     * index pairing derivation (row's first-half byte -> value[i]/value[i+16],
     * second-half byte -> value[8+i]/value[24+i]). */
    {
        const block_q4_0x8 *b_ptr = (const block_q4_0x8 *)vx;
        const block_q8_0 *a_ptr = (const block_q8_0 *)wy;
        int nb = n / 32;
        for (int row = 0; row < nrows; row++) {
            int group = row / 8;
            int r = row % 8;
            float sumf = 0.0f;
            for (int b = 0; b < nb; b++) {
                float dd = fp16_to_fp32_lookup(b_ptr[group * nb + b].d[r]) *
                            fp16_to_fp32_lookup(a_ptr[b].d);
                int sumi = 0;
                for (int i = 0; i < 8; i++) {
                    uint8_t byte = b_ptr[group * nb + b].qs[r * 8 + i];
                    int v0 = (int8_t)(byte << 4) >> 4;   /* value[i] */
                    int v1 = (int8_t)(byte & 0xF0) >> 4; /* value[i+16] */
                    sumi += v0 * a_ptr[b].qs[i] + v1 * a_ptr[b].qs[i + 16];
                }
                for (int i = 0; i < 8; i++) {
                    uint8_t byte = b_ptr[group * nb + b].qs[64 + r * 8 + i];
                    int v0 = (int8_t)(byte << 4) >> 4;   /* value[8+i] */
                    int v1 = (int8_t)(byte & 0xF0) >> 4; /* value[24+i] */
                    sumi += v0 * a_ptr[b].qs[8 + i] + v1 * a_ptr[b].qs[24 + i];
                }
                sumf += (float)sumi * dd;
            }
            out[row] = sumf;
        }
    }
#endif
}

/* ---- quantize_row_q4_0: FP32 -> Q4_0 blocks ----
 * Clamps each quantized value to [0, 15] to prevent 4-bit overflow.
 * Matches llama.cpp's MIN(15, ...) clamping in quantize_row_q4_0_ref. */
void quantize_row_q4_0(const float *x, void *dst, int n) {
    block_q4_0 *blocks = (block_q4_0 *)dst;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        const float *b = x + i * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float v = b[j] < 0 ? -b[j] : b[j];
            if (v > amax) amax = v;
        }
        float d = amax / 8.0f;
        blocks[i].d = fp32_to_fp16(d);
        float id = (amax != 0.0f) ? 8.0f / amax : 0.0f;
        uint8_t *q = blocks[i].qs;
        /* GGUF Q4_0: qs[j] = {b[j] low nibble, b[j+16] high nibble} */
        for (int j = 0; j < 16; j++) {
            uint8_t v0 = (uint8_t)(b[j] * id + 8.5f);
            uint8_t v1 = (uint8_t)(b[j + 16] * id + 8.5f);
            if (v0 > 15) v0 = 15;
            if (v1 > 15) v1 = 15;
            q[j] = (uint8_t)(v0 | (v1 << 4));
        }
    }
}

/* ---- scale_add_q8_0_f32: dst[i] += scale * dequant(q8_0[i]) ----
 * dequant(q8_0[i]) = qs[i] * d (per-block)
 * So: dst[i] += scale * qs[i] * d = (scale*d) * qs[i] */

void scale_add_q8_0_f32(float *dst, float scale, const void *src, int n) {
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        const block_q8_0 *b = &blocks[i];
        float sd = fp16_to_fp32_lookup(b->d) * scale;
        const int8_t *qs = b->qs;
        float *dp = dst + i * 32;

#ifdef PICOLM_AVX
        __m256 sd_v = _mm256_set1_ps(sd);

        for (int j = 0; j < 32; j += 8) {
            __m128i q8 = _mm_loadl_epi64((const __m128i *)(qs + j));
            __m128i q16 = _mm_srai_epi16(_mm_unpacklo_epi8(_mm_setzero_si128(), q8), 8);
            __m128i q32lo = _mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(), q16), 16);
            __m128i q32hi = _mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(), q16), 16);
            __m256 qf = _mm256_cvtepi32_ps(_mm256_set_m128i(q32hi, q32lo));
            __m256 scaled = _mm256_mul_ps(qf, sd_v);
            __m256 acc = _mm256_loadu_ps(dp + j);
            _mm256_storeu_ps(dp + j, _mm256_add_ps(acc, scaled));
        }
#elif defined(PICOLM_SSE2)
        __m128 sd_v = _mm_set1_ps(sd);
        for (int j = 0; j < 32; j += 8) {
            __m128i q8 = _mm_loadl_epi64((const __m128i *)(qs + j));
            __m128i q16 = _mm_srai_epi16(_mm_unpacklo_epi8(_mm_setzero_si128(), q8), 8);
            __m128i q32lo = _mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(), q16), 16);
            __m128i q32hi = _mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(), q16), 16);
            __m128 qf = _mm_cvtepi32_ps(q32lo);
            __m128 scaled = _mm_mul_ps(qf, sd_v);
            __m128 acc = _mm_loadu_ps(dp + j);
            _mm_storeu_ps(dp + j, _mm_add_ps(acc, scaled));

            qf = _mm_cvtepi32_ps(q32hi);
            scaled = _mm_mul_ps(qf, sd_v);
            acc = _mm_loadu_ps(dp + j + 4);
            _mm_storeu_ps(dp + j + 4, _mm_add_ps(acc, scaled));
        }
#elif defined(PICOLM_NEON)
        float32x4_t sd_v = vdupq_n_f32(sd);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t q8 = vld1_s8(qs + j);
            int16x8_t q16 = vmovl_s8(q8);
            float32x4_t qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
            float32x4_t acc = vld1q_f32(dp + j);
            vst1q_f32(dp + j, vmlaq_f32(acc, qf0, sd_v));

            float32x4_t qf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
            acc = vld1q_f32(dp + j + 4);
            vst1q_f32(dp + j + 4, vmlaq_f32(acc, qf1, sd_v));
        }
#else
        for (int j = 0; j < 32; j++) {
            dp[j] += qs[j] * sd;
        }
#endif
    }
}

/* ---- scale_add_q4_0_f32: dst[i] += scale * dequant(q4_0[i]) ---- */
void scale_add_q4_0_f32(float *dst, float scale, const void *src, int n) {
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        const block_q4_0 *b = &blocks[i];
        float sd = fp16_to_fp32_lookup(b->d) * scale;
        const uint8_t *qs = b->qs;
        float *dp = dst + i * 32;

        for (int j = 0; j < 16; j++) {
            dp[j]      += ((float)((qs[j] & 0xF) - 8)) * sd;
            dp[j + 16] += ((float)((qs[j] >> 4) - 8)) * sd;
        }
    }
}

/* ---- fma_scale_q8_0_f32: dst[i] = dst[i] * correction + dequant(q8_0[i]) ---- */
void fma_scale_q8_0_f32(float *dst, float correction, const void *src, int n) {
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        const block_q8_0 *b = &blocks[i];
        float d = fp16_to_fp32_lookup(b->d);
        const int8_t *qs = b->qs;
        float *dptr = dst + i * 32;

#ifdef PICOLM_AVX
        __m256 corr = _mm256_set1_ps(correction);
        __m256 df = _mm256_set1_ps(d);

        for (int j = 0; j < 32; j += 8) {
            __m128i q8 = _mm_loadl_epi64((const __m128i *)(qs + j));
            __m128i q16 = _mm_srai_epi16(_mm_unpacklo_epi8(_mm_setzero_si128(), q8), 8);
            __m128i q32lo = _mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(), q16), 16);
            __m128i q32hi = _mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(), q16), 16);
            __m256 qf = _mm256_cvtepi32_ps(_mm256_set_m128i(q32hi, q32lo));
            __m256 scaled = _mm256_mul_ps(qf, df);
            __m256 acc = _mm256_loadu_ps(dptr + j);
#ifdef __FMA__
            _mm256_storeu_ps(dptr + j, _mm256_fmadd_ps(acc, corr, scaled));
#else
            _mm256_storeu_ps(dptr + j, _mm256_add_ps(_mm256_mul_ps(acc, corr), scaled));
#endif
        }
#elif defined(PICOLM_SSE2)
        __m128 corr = _mm_set1_ps(correction);
        __m128 df = _mm_set1_ps(d);
        for (int j = 0; j < 32; j += 8) {
            __m128i q8 = _mm_loadl_epi64((const __m128i *)(qs + j));
            __m128i q16 = _mm_srai_epi16(_mm_unpacklo_epi8(_mm_setzero_si128(), q8), 8);
            __m128i q32lo = _mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(), q16), 16);
            __m128i q32hi = _mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(), q16), 16);
            __m128 qf = _mm_cvtepi32_ps(q32lo);
            __m128 scaled = _mm_mul_ps(qf, df);
            __m128 acc = _mm_loadu_ps(dptr + j);
            _mm_storeu_ps(dptr + j, _mm_add_ps(_mm_mul_ps(acc, corr), scaled));

            qf = _mm_cvtepi32_ps(q32hi);
            scaled = _mm_mul_ps(qf, df);
            acc = _mm_loadu_ps(dptr + j + 4);
            _mm_storeu_ps(dptr + j + 4, _mm_add_ps(_mm_mul_ps(acc, corr), scaled));
        }
#elif defined(PICOLM_NEON)
        float32x4_t corr = vdupq_n_f32(correction);
        float32x4_t df = vdupq_n_f32(d);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t q8 = vld1_s8(qs + j);
            int16x8_t q16 = vmovl_s8(q8);
            float32x4_t qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
            float32x4_t scaled = vmulq_f32(qf0, df);
            float32x4_t acc = vld1q_f32(dptr + j);
            vst1q_f32(dptr + j, vaddq_f32(vmulq_f32(acc, corr), scaled));

            float32x4_t qf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
            scaled = vmulq_f32(qf1, df);
            acc = vld1q_f32(dptr + j + 4);
            vst1q_f32(dptr + j + 4, vaddq_f32(vmulq_f32(acc, corr), scaled));
        }
#else
        for (int j = 0; j < 32; j++) {
            dptr[j] = dptr[j] * correction + qs[j] * d;
        }
#endif
    }
}

/* ---- fma_scale_q4_0_f32: dst[i] = dst[i] * correction + dequant(q4_0[i]) ---- */
void fma_scale_q4_0_f32(float *dst, float correction, const void *src, int n) {
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        const block_q4_0 *b = &blocks[i];
        float d = fp16_to_fp32_lookup(b->d);
        const uint8_t *qs = b->qs;
        float *dptr = dst + i * 32;

        for (int j = 0; j < 16; j++) {
            dptr[j]      = dptr[j] * correction + ((float)((qs[j] & 0xF) - 8)) * d;
            dptr[j + 16] = dptr[j + 16] * correction + ((float)((qs[j] >> 4) - 8)) * d;
        }
    }
}

/* ==================================================================
 * Walsh-Hadamard Transform (for KV cache rotation)
 *
 * Applies an in-place orthonormal Walsh-Hadamard transform.
 * Self-inverting: fast_ht(x, n) applied twice returns x to its original
 * values (up to float precision). This property (H * H = I) enables
 * dot-product preservation: (H*q) . (H*k) = q . k.
 *
 * From ik_llama's fast_ht() reference implementation.
 *
 * 'signs' parameter: if non-NULL, apply deterministic sign flips before the
 * butterfly stages. This breaks positional correlation in structured outliers,
 * improving KV cache quantization accuracy. See MNN TurboQuant tq3_wht_forward_32.
 * If signs is NULL and 'signs' is true, generate golden-ratio signs inline.
 * ================================================================== */
void picolm_fast_ht(float *x, int n, const float *signs) {
    /* n must be a power of 2 */
    assert((n > 0) && ((n & (n - 1)) == 0));

    /* Step 0: Apply deterministic sign flips before butterfly stages.
     * This randomizes structured outliers so the Hadamard transform
     * distributes energy more uniformly. */
    if (signs) {
        for (int i = 0; i < n; i++) x[i] *= signs[i];
    }

    const float ksqrt2 = 0.707106781f;
    float scale = 1.0f;

    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                float a = x[j];
                float b = x[j + h];
                x[j + 0] = a + b;
                x[j + h] = a - b;
            }
        }
        scale *= ksqrt2;
    }
    for (int i = 0; i < n; ++i) x[i] *= scale;
}

/* Deterministic sign pattern for WHT randomization (golden ratio hash).
 * signs[i] = ((i * 0x9E3779B9) >> 31) ? -1.0f : 1.0f
 * From MNN TurboQuant: breaks positional correlation before Hadamard rotation. */
static const float picolm_wht_signs_32[32] = {
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
   -1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,
   -1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f,
   -1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f,
};

void picolm_hadamard_transform(float *x, int head_dim, int nrot) {
    /* Apply fast_ht to each nrot-sized block within head_dim */
    assert(head_dim > 0 && head_dim % nrot == 0);
    int nblocks = head_dim / nrot;
    for (int b = 0; b < nblocks; b++) {
        const float *signs = (nrot == 32) ? picolm_wht_signs_32 : NULL;
        picolm_fast_ht(x + b * nrot, nrot, signs);
    }
}

/* ================================================================
 * TurboQuant TQ3: 3-bit Lloyd-Max codebook KV cache quantization
 *
 * Based on MNN's TurboQuant.hpp (Alibaba, 2024)
 * Algorithm: RMS normalize -> WHT forward (sign-randomized) ->
 *            Lloyd-Max 3-bit scalar quantization -> 14 bytes/block
 *
 * Block layout: [2 bytes fp16 RMS scale] [12 bytes packed 3-bit indices]
 * 32 values -> 14 bytes = 3.5 bits/value (vs Q4_0: 4.25 bits/value)
 *
 * The WHT used here is block_size=32 with deterministic sign flips
 * (golden ratio hash). This is the same transform as picolm_fast_ht
 * with signs=picolm_wht_signs_32, but TQ3 hardcodes block=32.
 * ================================================================ */

const float tq3_codebook[8] = TQ3_CODEBOOK_8;
const float tq3_boundaries[7] = TQ3_BOUNDARIES_7;
const float tq3_signs[32] = TQ3_SIGNS_32;

/* Pack 8 3-bit indices into 3 bytes */
static inline void tq3_pack_3bit_8(uint8_t *dst, const uint8_t *idx) {
    dst[0] = (uint8_t)(idx[0] | (idx[1] << 3) | (idx[2] << 6));
    dst[1] = (uint8_t)((idx[2] >> 2) | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7));
    dst[2] = (uint8_t)((idx[5] >> 1) | (idx[6] << 2) | (idx[7] << 5));
}
/* tq3_unpack_3bit_8 is declared in quant.h as static inline for use in model.c */

/* Find nearest TQ3 codebook index (scalar) */
static inline uint8_t tq3_find_nearest(float val) {
    uint8_t idx = 0;
    for (int b = 0; b < 7; b++) {
        if (val > tq3_boundaries[b]) idx = (uint8_t)(b + 1);
    }
    return idx;
}

/* WHT forward for TQ3 block (32 elements, in-place)
 * Uses the same butterfly as picolm_fast_ht but with TQ3 signs hardcoded. */
static void tq3_wht_forward_32(float *out, const float *in) {
    /* Step 1: Apply sign flips */
    for (int i = 0; i < 32; i++) out[i] = in[i] * tq3_signs[i];

    /* Step 2: Butterfly stages (log2(32) = 5) */
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = out[j];
                float b = out[j + step];
                out[j] = a + b;
                out[j + step] = a - b;
            }
        }
    }

    /* Step 3: Normalize by 1/sqrt(32) */
    const float norm = 1.0f / sqrtf(32.0f);
    for (int i = 0; i < 32; i++) out[i] *= norm;
}

/* WHT inverse for TQ3 block (32 elements, out-of-place)
 * Applies butterfly + normalize + undo sign flips. */
static void tq3_wht_inverse_32(float *out, const float *in) {
    /* Step 1: Copy input */
    for (int i = 0; i < 32; i++) out[i] = in[i];

    /* Step 2: Butterfly stages (self-inverse up to scaling) */
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = out[j];
                float b = out[j + step];
                out[j] = a + b;
                out[j + step] = a - b;
            }
        }
    }

    /* Step 3: Normalize and undo sign flips */
    const float norm = 1.0f / sqrtf(32.0f);
    for (int i = 0; i < 32; i++) out[i] *= norm * tq3_signs[i];
}

/* ---- quantize_row_tq3: float32 -> TQ3 blocks ---- */
void quantize_row_tq3(const float *x, void *dst, int n) {
    assert(n > 0 && (n % TQ3_BLOCK_SIZE) == 0);
    block_tq3 *blocks = (block_tq3 *)dst;
    int nb = n / TQ3_BLOCK_SIZE;

    for (int bi = 0; bi < nb; bi++) {
        const float *src = x + bi * TQ3_BLOCK_SIZE;

        /* Step 1: Compute RMS scale */
        float sumSq = 0.0f;
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++)
            sumSq += src[i] * src[i];
        float rms = sqrtf(sumSq / TQ3_BLOCK_SIZE);
        if (rms < 1e-10f) rms = 1e-10f;

        blocks[bi].d = fp32_to_fp16(rms);

        /* Step 2: Normalize by RMS */
        float normalized[TQ3_BLOCK_SIZE];
        float invRms = 1.0f / rms;
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++)
            normalized[i] = src[i] * invRms;

        /* Step 3: WHT forward with sign pattern */
        float rotated[TQ3_BLOCK_SIZE];
        tq3_wht_forward_32(rotated, normalized);

        /* Step 4: Find nearest codebook index per element */
        uint8_t indices[TQ3_BLOCK_SIZE];
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++)
            indices[i] = tq3_find_nearest(rotated[i]);

        /* Step 5: Pack 3-bit indices (4 groups of 8 -> 12 bytes) */
        for (int g = 0; g < 4; g++)
            tq3_pack_3bit_8(blocks[bi].qs + g * 3, indices + g * 8);
    }
}

/* ---- vec_dot_tq3_f32: TQ3-encoded K dot float32 pre-rotated Q ----
 *
 * The Q input must already be WHT-forward rotated (block=32) with
 * the TQ3 sign pattern. This is the same rotation applied during
 * quantization, so the dot product simplifies to:
 *   dot = scale * sum(q_rotated[i] * codebook[idx[i]])
 * No inverse WHT needed. */
float vec_dot_tq3_f32(const void *src, const float *q_rotated, int n) {
    assert(n > 0 && (n % TQ3_BLOCK_SIZE) == 0);
    const block_tq3 *blocks = (const block_tq3 *)src;
    int nb = n / TQ3_BLOCK_SIZE;
    float sumf = 0.0f;

    for (int bi = 0; bi < nb; bi++) {
        float scale = fp16_to_fp32(blocks[bi].d);
        const uint8_t *packed = blocks[bi].qs;
        const float *qr = q_rotated + bi * TQ3_BLOCK_SIZE;
        float dot = 0.0f;

        for (int g = 0; g < 4; g++) {
            uint8_t indices[8];
            tq3_unpack_3bit_8(indices, packed + g * 3);
            for (int k = 0; k < 8; k++) {
                dot += qr[g * 8 + k] * tq3_codebook[indices[k]];
            }
        }
        sumf += dot * scale;
    }
    return sumf;
}

/* ---- scale_add_tq3_f32: dst[i] += scale * tq3_dequant(src[i]) ----
 *
 * Full dequantization path: unpack -> codebook lookup -> WHT inverse ->
 * scale multiply -> add to destination. The inverse WHT is applied per
 * TQ3 block (32 elements) before writing to dst. */
void scale_add_tq3_f32(float *dst, float scale, const void *src, int n) {
    assert(n > 0 && (n % TQ3_BLOCK_SIZE) == 0);
    const block_tq3 *blocks = (const block_tq3 *)src;
    int nb = n / TQ3_BLOCK_SIZE;

    for (int bi = 0; bi < nb; bi++) {
        float block_scale = scale * fp16_to_fp32(blocks[bi].d);
        const uint8_t *packed = blocks[bi].qs;

        /* Unpack indices and look up codebook values -> rotated domain */
        float rotated[TQ3_BLOCK_SIZE];
        for (int g = 0; g < 4; g++) {
            uint8_t indices[8];
            tq3_unpack_3bit_8(indices, packed + g * 3);
            for (int k = 0; k < 8; k++)
                rotated[g * 8 + k] = tq3_codebook[indices[k]];
        }

        /* WHT inverse -> original domain, then scale + add */
        float reconstructed[TQ3_BLOCK_SIZE];
        tq3_wht_inverse_32(reconstructed, rotated);
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++)
            dst[bi * TQ3_BLOCK_SIZE + i] += reconstructed[i] * block_scale;
    }
}

/* ---- fma_scale_tq3_f32: dst[i] = dst[i] * correction + tq3_dequant(src[i]) ----
 *
 * Used for the online softmax "new max" path in attention.
 * correction = exp(old_max - new_max), dst *= correction, then add dequant. */
void fma_scale_tq3_f32(float *dst, float correction, const void *src, int n) {
    assert(n > 0 && (n % TQ3_BLOCK_SIZE) == 0);
    const block_tq3 *blocks = (const block_tq3 *)src;
    int nb = n / TQ3_BLOCK_SIZE;

    for (int bi = 0; bi < nb; bi++) {
        float block_scale = fp16_to_fp32(blocks[bi].d);
        const uint8_t *packed = blocks[bi].qs;
        float *dptr = dst + bi * TQ3_BLOCK_SIZE;

        /* Unpack indices and look up codebook values */
        float rotated[TQ3_BLOCK_SIZE];
        for (int g = 0; g < 4; g++) {
            uint8_t indices[8];
            tq3_unpack_3bit_8(indices, packed + g * 3);
            for (int k = 0; k < 8; k++)
                rotated[g * 8 + k] = tq3_codebook[indices[k]];
        }

        /* WHT inverse -> original domain */
        float reconstructed[TQ3_BLOCK_SIZE];
        tq3_wht_inverse_32(reconstructed, rotated);

        /* FMA: dst[i] = dst[i] * correction + reconstructed[i] * scale */
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++)
            dptr[i] = dptr[i] * correction + reconstructed[i] * block_scale;
    }
}

/* ================================================================
 * TQ4: 4-bit TurboQuant (4.5 bits/value)
 *
 * Same WHT rotation as TQ3, but 16-entry Lloyd-Max codebook with
 * nibble (4-bit) packing. D_mse ~0.0095 (vs TQ3 ~0.032).
 * Block layout: 2 bytes fp16 RMS scale + 16 bytes packed indices = 18 bytes.
 * ================================================================ */

const float tq4_codebook[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f,
};

const float tq4_boundaries[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f, -0.7995f, -0.5224f, -0.2582f, 0.0000f,
     0.2582f,  0.5224f,  0.7995f,  1.0993f,  1.4371f,  1.8435f,  2.4008f,
};

/* Find nearest TQ4 codebook index for a value */
static uint8_t tq4_find_nearest(float val) {
    uint8_t idx = 0;
    for (int b = 0; b < 15; b++) {
        if (val > tq4_boundaries[b]) idx = (uint8_t)(b + 1);
    }
    return idx;
}

/* Pack 8 4-bit indices into 4 bytes (nibble packing) */
static void tq4_pack_4bit_8(uint8_t *dst, const uint8_t *idx) {
    dst[0] = (uint8_t)(idx[0] | (idx[1] << 4));
    dst[1] = (uint8_t)(idx[2] | (idx[3] << 4));
    dst[2] = (uint8_t)(idx[4] | (idx[5] << 4));
    dst[3] = (uint8_t)(idx[6] | (idx[7] << 4));
}

/* ---- quantize_row_tq4: float32 -> TQ4 blocks ---- */
void quantize_row_tq4(const float *x, void *dst, int n) {
    assert(n > 0 && (n % TQ4_BLOCK_SIZE) == 0);
    block_tq4 *blocks = (block_tq4 *)dst;

    for (int bi = 0; bi < n / TQ4_BLOCK_SIZE; bi++) {
        const float *src = x + bi * TQ4_BLOCK_SIZE;

        /* Step 1: RMS normalization */
        float rms = 0.0f;
        for (int i = 0; i < TQ4_BLOCK_SIZE; i++) rms += src[i] * src[i];
        rms = sqrtf(rms / TQ4_BLOCK_SIZE);
        if (rms < 1e-10f) rms = 1e-10f;
        blocks[bi].d = fp32_to_fp16(rms);

        /* Step 2: Normalize by RMS */
        float normalized[TQ4_BLOCK_SIZE];
        float invRms = 1.0f / rms;
        for (int i = 0; i < TQ4_BLOCK_SIZE; i++)
            normalized[i] = src[i] * invRms;

        /* Step 3: WHT forward with sign pattern (reuses TQ3's WHT) */
        float rotated[TQ4_BLOCK_SIZE];
        tq3_wht_forward_32(rotated, normalized);

        /* Step 4: Find nearest codebook index per element */
        uint8_t indices[TQ4_BLOCK_SIZE];
        for (int i = 0; i < TQ4_BLOCK_SIZE; i++)
            indices[i] = tq4_find_nearest(rotated[i]);

        /* Step 5: Pack 4-bit indices (4 groups of 8 -> 16 bytes) */
        for (int g = 0; g < 4; g++)
            tq4_pack_4bit_8(blocks[bi].qs + g * 4, indices + g * 8);
    }
}

/* ---- vec_dot_tq4_f32: TQ4-encoded K dot float32 pre-rotated Q ---- */
float vec_dot_tq4_f32(const void *src, const float *q_rotated, int n) {
    assert(n > 0 && (n % TQ4_BLOCK_SIZE) == 0);
    const block_tq4 *blocks = (const block_tq4 *)src;
    int nb = n / TQ4_BLOCK_SIZE;
    float sumf = 0.0f;

    for (int bi = 0; bi < nb; bi++) {
        float scale = fp16_to_fp32(blocks[bi].d);
        const uint8_t *packed = blocks[bi].qs;
        const float *qr = q_rotated + bi * TQ4_BLOCK_SIZE;
        float dot = 0.0f;

        for (int g = 0; g < 4; g++) {
            uint8_t indices[8];
            tq4_unpack_4bit_8(indices, packed + g * 4);
            for (int k = 0; k < 8; k++) {
                dot += qr[g * 8 + k] * tq4_codebook[indices[k]];
            }
        }
        sumf += dot * scale;
    }
    return sumf;
}

/* ---- scale_add_tq4_f32: dst[i] += scale * tq4_dequant(src[i]) ---- */
void scale_add_tq4_f32(float *dst, float scale, const void *src, int n) {
    assert(n > 0 && (n % TQ4_BLOCK_SIZE) == 0);
    const block_tq4 *blocks = (const block_tq4 *)src;
    int nb = n / TQ4_BLOCK_SIZE;

    for (int bi = 0; bi < nb; bi++) {
        float block_scale = scale * fp16_to_fp32(blocks[bi].d);
        const uint8_t *packed = blocks[bi].qs;

        /* Unpack indices and look up codebook values -> rotated domain */
        float rotated[TQ4_BLOCK_SIZE];
        for (int g = 0; g < 4; g++) {
            uint8_t indices[8];
            tq4_unpack_4bit_8(indices, packed + g * 4);
            for (int k = 0; k < 8; k++)
                rotated[g * 8 + k] = tq4_codebook[indices[k]];
        }

        /* WHT inverse -> original domain, then scale + add */
        float reconstructed[TQ4_BLOCK_SIZE];
        tq3_wht_inverse_32(reconstructed, rotated);
        for (int i = 0; i < TQ4_BLOCK_SIZE; i++)
            dst[bi * TQ4_BLOCK_SIZE + i] += reconstructed[i] * block_scale;
    }
}

/* ---- fma_scale_tq4_f32: dst[i] = dst[i] * correction + tq4_dequant(src[i]) ---- */
void fma_scale_tq4_f32(float *dst, float correction, const void *src, int n) {
    assert(n > 0 && (n % TQ4_BLOCK_SIZE) == 0);
    const block_tq4 *blocks = (const block_tq4 *)src;
    int nb = n / TQ4_BLOCK_SIZE;

    for (int bi = 0; bi < nb; bi++) {
        float block_scale = fp16_to_fp32(blocks[bi].d);
        const uint8_t *packed = blocks[bi].qs;
        float *dptr = dst + bi * TQ4_BLOCK_SIZE;

        /* Unpack indices and look up codebook values */
        float rotated[TQ4_BLOCK_SIZE];
        for (int g = 0; g < 4; g++) {
            uint8_t indices[8];
            tq4_unpack_4bit_8(indices, packed + g * 4);
            for (int k = 0; k < 8; k++)
                rotated[g * 8 + k] = tq4_codebook[indices[k]];
        }

        /* WHT inverse -> original domain */
        float reconstructed[TQ4_BLOCK_SIZE];
        tq3_wht_inverse_32(reconstructed, rotated);

        /* FMA: dst[i] = dst[i] * correction + reconstructed[i] * scale */
        for (int i = 0; i < TQ4_BLOCK_SIZE; i++)
            dptr[i] = dptr[i] * correction + reconstructed[i] * block_scale;
    }
}

/* ================================================================
 * GELU lookup table (mirrors llama.cpp approach)
 * ================================================================ */
uint16_t picolm_gelu_table[65536];

static float gelu_f32_scalar(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

void picolm_gelu_f32(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = gelu_f32_scalar(x[i]);
    }
}

void picolm_init_gelu_table(void) {
    for (int i = 0; i < 65536; i++) {
        float f = fp16_to_fp32_lookup(i);
        float g = gelu_f32_scalar(f);
        picolm_gelu_table[i] = fp32_to_fp16(g);
    }
}

void picolm_gelu_table_f32(float *x, int size) {
#ifdef PICOLM_NEON
    int i;
    for (i = 0; i + 3 < size; i += 4) {
        uint16_t t0 = fp32_to_fp16(x[i]);
        uint16_t t1 = fp32_to_fp16(x[i+1]);
        uint16_t t2 = fp32_to_fp16(x[i+2]);
        uint16_t t3 = fp32_to_fp16(x[i+3]);
        x[i]   = fp16_to_fp32_lookup(picolm_gelu_table[t0]);
        x[i+1] = fp16_to_fp32_lookup(picolm_gelu_table[t1]);
        x[i+2] = fp16_to_fp32_lookup(picolm_gelu_table[t2]);
        x[i+3] = fp16_to_fp32_lookup(picolm_gelu_table[t3]);
    }
    for (; i < size; i++) {
        uint16_t t = fp32_to_fp16(x[i]);
        x[i] = fp16_to_fp32_lookup(picolm_gelu_table[t]);
    }
#else
    for (int i = 0; i < size; i++) {
        uint16_t t = fp32_to_fp16(x[i]);
        x[i] = fp16_to_fp32_lookup(picolm_gelu_table[t]);
    }
#endif
}
