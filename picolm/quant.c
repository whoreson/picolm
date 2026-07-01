#include "quant.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
    return fp16_to_fp32_table[h];
}

/* ================================================================
 * FP16 <-> FP32 conversion (software, no hardware dependency)
 * ================================================================ */

/* fp16-fp32 dot product: sum of fp16_to_fp32(k[i]) * x[i] for i=0..n-1 */
float vec_dot_f16_f32(const void *src, const float *x, int n) {
    const uint16_t *k = (const uint16_t *)src;
#ifdef PICOLM_NEON
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        float32x4_t kf0 = fp16x4_to_fp32_inline(k + i);
        float32x4_t xf0 = vld1q_f32(x + i);
        acc0 = vmlaq_f32(acc0, kf0, xf0);
        float32x4_t kf1 = fp16x4_to_fp32_inline(k + i + 4);
        float32x4_t xf1 = vld1q_f32(x + i + 4);
        acc1 = vmlaq_f32(acc1, kf1, xf1);
    }
    float sumf = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sumf += fp16_to_fp32(k[i]) * x[i];
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
    for (; i < n; i++) sumf += fp16_to_fp32(k[i]) * x[i];
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
    for (; i < n; i++) sumf += fp16_to_fp32(k[i]) * x[i];
    return sumf;
#else
    float sumf = 0.0f;
    for (int i = 0; i < n; i++) sumf += fp16_to_fp32(k[i]) * x[i];
    return sumf;
#endif
}

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign; /* +/- zero */
        } else {
            /* subnormal: renormalize */
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13); /* inf / nan */
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

uint16_t fp32_to_fp16(float f) {
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
}

/* ---- Q4_K helpers ---- */

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >>  4) | ((q[j    ] >> 6) << 4);
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
        float d    = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);
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
    const block_q3_K *blocks = (const block_q3_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q3_K *b = &blocks[i];
        float d = fp16_to_fp32(b->d);

        int32_t scales[16];
        {
            for (int j = 0; j < 8; j++) {
                scales[j] = (int32_t)(b->scales[j] & 0xF);
            }
            for (int j = 0; j < 8; j++) {
                scales[8 + j] = (int32_t)(b->scales[j] >> 4);
            }
            for (int j = 0; j < 4; j++) {
                scales[2*j]     |= ((b->scales[8 + j]     ) & 3) << 4;
                scales[2*j + 1] |= ((b->scales[8 + j] >> 2) & 3) << 4;
                scales[2*j + 8] |= ((b->scales[8 + j] >> 4) & 3) << 4;
                scales[2*j + 9] |= ((b->scales[8 + j] >> 6) & 3) << 4;
            }
            for (int j = 0; j < 16; j++) {
                scales[j] -= 32;
            }
        }

        const uint8_t *qs    = b->qs;
        const uint8_t *hmask = b->hmask;
        int out_idx = i * 256;

        for (int j = 0; j < 256; j++) {
            int q2 = (qs[j / 4] >> (2 * (j % 4))) & 3;
            int hbit = (hmask[j / 8] >> (j % 8)) & 1;
            int q3 = q2 | (hbit << 2);
            int sb = j / 16;
            dst[out_idx + j] = d * (float)scales[sb] * ((float)q3 - 4.0f);
        }
    }
}

void dequantize_row_q2_K(const void *src, float *dst, int n) {
    const block_q2_K *blocks = (const block_q2_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q2_K *b = &blocks[i];
        float d    = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);

        const uint8_t *qs = b->qs;
        int out_idx = i * 256;

        for (int j = 0; j < 256; j++) {
            int q2 = (qs[j / 4] >> (2 * (j % 4))) & 3;
            int sb = j / 16;
            uint8_t sc = b->scales[sb] & 0xF;
            uint8_t mn = b->scales[sb] >> 4;
            dst[out_idx + j] = d * (float)sc * (float)q2 - dmin * (float)mn;
        }
    }
}

void dequantize_row_q6_K(const void *src, float *dst, int n) {
    const block_q6_K *blocks = (const block_q6_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
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
        float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            dst[i * 32 + j] = d * (float)blocks[i].qs[j];
        }
    }
}

void dequantize_row_q4_0(const void *src, float *dst, int n) {
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            uint8_t nibble;
            if (j < 16) {
                nibble = blocks[i].qs[j] & 0xF;
            } else {
                nibble = blocks[i].qs[j - 16] >> 4;
            }
            dst[i * 32 + j] = d * ((float)nibble - 8.0f);
        }
    }
}

void dequantize_row_f16(const void *src, float *dst, int n) {
    const uint16_t *fp16 = (const uint16_t *)src;
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32(fp16[i]);
    }
}

void dequantize_row_f32(const void *src, float *dst, int n) {
    memcpy(dst, src, n * sizeof(float));
}

void dequantize_row(const void *src, float *dst, int n, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   dequantize_row_f32(src, dst, n);  break;
        case GGUF_TYPE_F16:   dequantize_row_f16(src, dst, n);  break;
        case GGUF_TYPE_Q4_0:  dequantize_row_q4_0(src, dst, n); break;
        case GGUF_TYPE_Q8_0:  dequantize_row_q8_0(src, dst, n); break;
        case GGUF_TYPE_Q2_K:  dequantize_row_q2_K(src, dst, n); break;
        case GGUF_TYPE_Q3_K:  dequantize_row_q3_K(src, dst, n); break;
        case GGUF_TYPE_Q4_K:  dequantize_row_q4_K(src, dst, n); break;
        case GGUF_TYPE_Q6_K:  dequantize_row_q6_K(src, dst, n); break;
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
        default: return 0;
    }
}

size_t gguf_type_row_size(gguf_type_t type, int n) {
    int bs = gguf_type_block_size(type);
    int qs = gguf_type_quant_size(type);
    if (bs == 0 || qs == 0) return 0;
    return (size_t)(n / bs) * qs;
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
        float d    = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);
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
        float d = fp16_to_fp32(blocks[i].d);
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
 * quantize_row_q8_0: quantize float32 -> Q8_0 blocks
 *
 * Adapted from llama.cpp's AVX quantize_row_q8_0.
 * ================================================================ */

void quantize_row_q8_0(const float *x, void *dst, int n) {
    block_q8_0 *y = (block_q8_0 *)dst;
    int nb = n / 32;

#ifdef PICOLM_AVX2
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
            y[i].qs[4*j+0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j+1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j+2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j+3] = vgetq_lane_s32(vi, 3);
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
        for (int j = 0; j < 64; j += 8) {
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
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j * 16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        x += 256;
    }
}

/* ================================================================
 * vec_dot_q4_K_q8_K: int8 MAC for Q4_K weights * Q8_K input
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
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

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

        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales256,
                _mm256_setr_epi8(0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1, 2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3));
            const __m256i scale_h = _mm256_shuffle_epi8(scales256,
                _mm256_setr_epi8(4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5, 6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7));

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
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales128 = _mm_cvtepu8_epi16(utmps);
        const __m128i mins128 = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

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

#else
    /* Scalar fallback: nibble extraction + int8 MAC */
    /* Used on NEON (no SIMD Q4_K_q8_K yet) and x86 without AVX */
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

        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; l++) sums[l] += d * aux32[l];

        const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
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

#if defined(PICOLM_AVX2) || defined(PICOLM_AVX) || defined(PICOLM_SSE2)
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
                fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
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
            float d0 = fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d);
            float d1 = fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d);
            __m256 deltas = _mm256_set_m128(_mm_set1_ps(d1), _mm_set1_ps(d0));
            accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
        }
        sumf = hsum_avx(accum);
    }

#elif defined(PICOLM_SSE2)
    {
        const __m128i mask4 = _mm_set1_epi8(15);
        const __m128i off = _mm_set1_epi8(8);
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();
        __m128 acc2 = _mm_setzero_ps();
        __m128 acc3 = _mm_setzero_ps();

        for (; ib + 1 < nb; ib += 2) {
            const __m128 d_0_1 = _mm_set1_ps(
                fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
            const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i *)x[ib].qs);
            __m128i bx_0 = _mm_sub_epi8(_mm_and_si128(mask4, tmp_0_1), off);
            const __m128i i32_0 = mul_sum_i8_pairs_sse(bx_0, _mm_loadu_si128((const __m128i *)y[ib].qs));
            __m128i bx_1 = _mm_sub_epi8(_mm_and_si128(mask4, _mm_srli_epi64(tmp_0_1, 4)), off);
            const __m128i i32_1 = mul_sum_i8_pairs_sse(bx_1, _mm_loadu_si128((const __m128i *)(y[ib].qs + 16)));

            const __m128 d_2_3 = _mm_set1_ps(
                fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d));
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
        sumf += (float)(sumi0 + sumi1) * fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d);
    }

    return sumf;
}

float vec_dot_q8_0_q8_0(const void *qx, const void *qw, int n) {
    const block_q8_0 *x = (const block_q8_0 *)qx;
    const block_q8_0 *w = (const block_q8_0 *)qw;
    int nb = n / 32;
    float sumf = 0.0f;
    int i = 0;

#ifdef PICOLM_NEON
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
            const float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(w[i].d);
            sumv0 = vmlaq_n_f32(sumv0, s, d);
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
            const float d = fp16_to_fp32(x[i+1].d) * fp16_to_fp32(w[i+1].d);
            sumv1 = vmlaq_n_f32(sumv1, s, d);
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

        float d0 = fp16_to_fp32(x[i].d) * fp16_to_fp32(w[i].d);
        float d1 = fp16_to_fp32(x[i + 1].d) * fp16_to_fp32(w[i + 1].d);
        __m256 dd0 = _mm256_set1_ps(d0);
        __m256 dd1 = _mm256_set1_ps(d1);

        acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_mul_ps(f0, dd0), _mm256_mul_ps(f1, dd1)));
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

        float d0 = fp16_to_fp32(x[i].d) * fp16_to_fp32(w[i].d);
        float d1 = fp16_to_fp32(x[i + 1].d) * fp16_to_fp32(w[i + 1].d);
        __m256 deltas = _mm256_set_m128(_mm_set1_ps(d1), _mm_set1_ps(d0));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(deltas, sums));
    }
    sumf = hsum_avx(acc);

#elif defined(PICOLM_SSE2)
    /* SSE2: process 2 blocks per iteration with 2 accumulators */
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
        float d0 = fp16_to_fp32(x[i].d) * fp16_to_fp32(w[i].d);
        __m128 dd0 = _mm_set1_ps(d0);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_cvtepi32_ps(s0), dd0));

        /* Block i+1 */
        __m128i qx2 = _mm_loadu_si128((const __m128i *)x[i + 1].qs);
        __m128i qx3 = _mm_loadu_si128((const __m128i *)x[i + 1].qs + 1);
        __m128i qw2 = _mm_loadu_si128((const __m128i *)w[i + 1].qs);
        __m128i qw3 = _mm_loadu_si128((const __m128i *)w[i + 1].qs + 1);
        __m128i s1 = _mm_add_epi32(mul_sum_i8_pairs_sse(qx2, qw2),
                                    mul_sum_i8_pairs_sse(qx3, qw3));
        float d1 = fp16_to_fp32(x[i + 1].d) * fp16_to_fp32(w[i + 1].d);
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
        sumf += (float)sumi * fp16_to_fp32(x[i].d) * fp16_to_fp32(w[i].d);
    }
    return sumf;
}

/* ================================================================
 * vec_dot_q8_0_q8_0_deltas: int8 MAC with pre-converted x deltas
 *
 * Same as vec_dot_q8_0_q8_0 but takes pre-converted float32 deltas
 * for the quantized input x, avoiding per-block fp16_to_fp32 calls.
 * This saves ~nb/2 fp16_to_fp32 calls per vec_dot invocation.
 * ================================================================ */

float vec_dot_q8_0_q8_0_deltas(const void *qx, const float *qx_d, const void *qw, int n) {
    const block_q8_0 *x = (const block_q8_0 *)qx;
    const block_q8_0 *w = (const block_q8_0 *)qw;
    int nb = n / 32;
    float sumf = 0.0f;
    int i = 0;

#ifdef PICOLM_NEON
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
            const float d = qx_d[i] * fp16_to_fp32(w[i].d);
            sumv0 = vmlaq_n_f32(sumv0, s, d);
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
            const float d = qx_d[i+1] * fp16_to_fp32(w[i+1].d);
            sumv1 = vmlaq_n_f32(sumv1, s, d);
        }
    }
    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

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

        float d0 = qx_d[i] * fp16_to_fp32(w[i].d);
        float d1 = qx_d[i + 1] * fp16_to_fp32(w[i + 1].d);
        __m256 dd0 = _mm256_set1_ps(d0);
        __m256 dd1 = _mm256_set1_ps(d1);
        acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_mul_ps(f0, dd0), _mm256_mul_ps(f1, dd1)));
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

        float d0 = qx_d[i] * fp16_to_fp32(w[i].d);
        float d1 = qx_d[i + 1] * fp16_to_fp32(w[i + 1].d);
        __m256 deltas = _mm256_set_m128(_mm_set1_ps(d1), _mm_set1_ps(d0));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(deltas, sums));
    }
    sumf = hsum_avx(acc);

#elif defined(PICOLM_SSE2)
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();

    for (i = 0; i + 1 < nb; i += 2) {
        __m128i qx0 = _mm_loadu_si128((const __m128i *)x[i].qs);
        __m128i qx1 = _mm_loadu_si128((const __m128i *)x[i].qs + 1);
        __m128i qw0 = _mm_loadu_si128((const __m128i *)w[i].qs);
        __m128i qw1 = _mm_loadu_si128((const __m128i *)w[i].qs + 1);
        __m128i s0 = _mm_add_epi32(mul_sum_i8_pairs_sse(qx0, qw0),
                                    mul_sum_i8_pairs_sse(qx1, qw1));
        float d0 = qx_d[i] * fp16_to_fp32(w[i].d);
        __m128 dd0 = _mm_set1_ps(d0);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_cvtepi32_ps(s0), dd0));

        __m128i qx2 = _mm_loadu_si128((const __m128i *)x[i + 1].qs);
        __m128i qx3 = _mm_loadu_si128((const __m128i *)x[i + 1].qs + 1);
        __m128i qw2 = _mm_loadu_si128((const __m128i *)w[i + 1].qs);
        __m128i qw3 = _mm_loadu_si128((const __m128i *)w[i + 1].qs + 1);
        __m128i s1 = _mm_add_epi32(mul_sum_i8_pairs_sse(qx2, qw2),
                                    mul_sum_i8_pairs_sse(qx3, qw3));
        float d1 = qx_d[i + 1] * fp16_to_fp32(w[i + 1].d);
        __m128 dd1 = _mm_set1_ps(d1);
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_cvtepi32_ps(s1), dd1));
    }
    sumf = hsum_sse(_mm_add_ps(acc0, acc1));
#endif

    for (; i < nb; i++) {
        int sumi = 0;
        for (int j = 0; j < 32; j++) sumi += x[i].qs[j] * w[i].qs[j];
        sumf += (float)sumi * qx_d[i] * fp16_to_fp32(w[i].d);
    }
    return sumf;
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
        float d = fp16_to_fp32(b->d);
        const int8_t *qs = b->qs;
        const float *xp = x + i * 32;

#ifdef PICOLM_NEON
        float32x4_t acc = vdupq_n_f32(0);

        for (int j = 0; j < 32; j += 4) {
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
        for (int j = 0; j < 32; j++) {
            sumf += (float)qs[j] * xp[j];
        }
        sumf *= d;
#endif
    }
    return sumf;
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
        float d = fp16_to_fp32(blocks[i].d);
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

/* ---- Generic dispatch ---- */

float vec_dot(const void *src, const float *x, int n, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_Q4_K: return vec_dot_q4_K_f32(src, x, n);
        case GGUF_TYPE_Q6_K: return vec_dot_q6_K_f32(src, x, n);
        case GGUF_TYPE_F32:  return vec_dot_f32_f32(src, x, n);
        case GGUF_TYPE_Q8_0: return vec_dot_q8_0_f32(src, x, n);
        case GGUF_TYPE_Q4_0: return vec_dot_q4_0_f32(src, x, n);
        case GGUF_TYPE_F16:  return vec_dot_f16_f32(src, x, n);
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
