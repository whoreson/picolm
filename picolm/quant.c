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
        float32x4_t kf0 = fp16x4_to_fp32_inline(k + i);
        float32x4_t xf0 = vld1q_f32(x + i);
        acc0 = vmlaq_f32(acc0, kf0, xf0);
        float32x4_t kf1 = fp16x4_to_fp32_inline(k + i + 4);
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

    return result;
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
    const block_q3_K *blocks = (const block_q3_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q3_K *b = &blocks[i];
        float d = fp16_to_fp32_lookup(b->d);

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
        float d    = fp16_to_fp32_lookup(b->d);
        float dmin = fp16_to_fp32_lookup(b->dmin);

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
 * Only the first row of each 4-row group is dequantized. */
void dequantize_row_q4_0_4_4(const void *src, float *dst, int n) {
    const block_q4_0x4 *blocks = (const block_q4_0x4 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32_lookup(blocks[i].d[0]);
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 4; j++) {
                uint8_t byte = blocks[i].qs[k * 16 + j * 4];
                int v0 = (int8_t)(byte << 4) >> 4;
                int v1 = (int8_t)(byte & 0xF0) >> 4;
                dst[i * 32 + k * 8 + j * 2] = d * (float)v0;
                dst[i * 32 + k * 8 + j * 2 + 1] = d * (float)v1;
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
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32_lookup(fp16[i]);
    }
}

void dequantize_row_f32(const void *src, float *dst, int n) {
#if defined(__APPLE__) && defined(__ppc__)
    /* GGUF stores F32 as little-endian; swap on big-endian */
    const uint32_t *src32 = (const uint32_t *)src;
    uint32_t *dst32 = (uint32_t *)dst;
    for (int i = 0; i < n; i++) {
        dst32[i] = (src32[i] >> 24) | ((src32[i] >> 8) & 0xff00) | ((src32[i] << 8) & 0xff0000) | (src32[i] << 24);
    }
#else
    memcpy(dst, src, n * sizeof(float));
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
            for (int l = 0; l < 32; ++l) *dst++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *dst++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
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
            int v = (int)(x[j] * id);
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

        const float d = fp16_to_fp32_lookup(x[i].d) * y[i].d;
        for (int l = 0; l < 8; l++) sums[l] += d * aux32[l];

        const float dmin = fp16_to_fp32_lookup(x[i].dmin) * y[i].d;
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

#if defined(PICOLM_AVX512)
/* AVX-512 signed int8 x int8 dot product, 64 pairs per call (2 q8_0
 * blocks' worth). Note _mm512_sign_epi8 does not exist in AVX-512 (Intel
 * didn't extend the SSSE3 "sign" instructions to 512-bit) -- the usual
 * abs(x)/sign(y,x) trick needs a mask-based substitute: extract the sign
 * bit of each byte of x with _mm512_movepi8_mask, then blend y against
 * -y using that mask. When x's byte is 0 the "sign" is arbitrary, but
 * ax=abs(x) is also 0 there so the product is 0 regardless -- matches
 * _mm256_sign_epi8's behavior for that case.
 * With AVX512-VNNI, _mm512_dpbusd_epi32 does the unsigned x signed
 * multiply-and-reduce to int32 in one instruction (also sidesteps the
 * int16 intermediate _mm512_maddubs_epi16 produces, though for q8_0's
 * +-127 range that intermediate never overflows int16 anyway -- the
 * win here is purely fewer instructions). Falls back to
 * maddubs+madd (still 512-bit, still 2 blocks/call) without VNNI. */
static inline __m512i mul_sum_i8_pairs_avx512(const __m512i x, const __m512i y) {
    __m512i ax = _mm512_abs_epi8(x);
    __mmask64 neg_mask = _mm512_movepi8_mask(x);
    __m512i neg_y = _mm512_sub_epi8(_mm512_setzero_si512(), y);
    __m512i sy = _mm512_mask_blend_epi8(neg_mask, y, neg_y);
#if defined(__AVX512VNNI__)
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), ax, sy);
#else
    __m512i dot = _mm512_maddubs_epi16(ax, sy);
    __m512i ones = _mm512_set1_epi16(1);
    return _mm512_madd_epi16(ones, dot);
#endif
}
#endif

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

        acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_mul_ps(f0, dd0), _mm256_mul_ps(f1, dd1)));
    }
    sumf = hsum_avx(acc);

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

        float d0 = fp16_to_fp32_lookup(x[i].d) * fp16_to_fp32_lookup(w[i].d);
        float d1 = fp16_to_fp32_lookup(x[i + 1].d) * fp16_to_fp32_lookup(w[i + 1].d);
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
        const block_q4_0x4 *b_ptr = xb + (size_t)(row_group / 4) * nb;
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
        const block_q4_0x4 *group_base = xb + (size_t)group * nb;
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

        float d0 = qx_d[i] * fp16_to_fp32_lookup(w[i].d);
        float d1 = qx_d[i + 1] * fp16_to_fp32_lookup(w[i + 1].d);
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
        int nq8 = (n / 32) * sizeof(block_q8_0);
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
        int nb = n / 128;
        int nq8 = (n / 32) * sizeof(block_q8_0);
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
            sumi += d1 * sumi_block;
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
#else
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
            sumi += d1 * sumi_block;
        }
        sumf += d0 * sumi;
    }
#endif
    return sumf;
}

/* vec_dot for Q4_0_4_4: dequantize first row of interleaved block, then dot */
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
                uint8_t byte = blocks[i].qs[k * 16 + j * 4];
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

/* ---- Generic dispatch ---- */

float vec_dot(const void *src, const float *x, int n, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_Q4_K: return vec_dot_q4_K_f32(src, x, n);
        case GGUF_TYPE_Q5_K: {
            /* Scalar fallback: dequantize to float, then dot */
#if defined(__APPLE__) && defined(__ppc__)
            /* __thread not supported on Mac OS X PPC; use static buffer (not thread-safe but fine without OpenMP) */
            static float q5_tmp[4096];
#else
            static float __thread q5_tmp[4096];
#endif
            if (n > 4096) {
                float *tmp = (float *)malloc(n * sizeof(float));
                dequantize_row_q5_K(src, tmp, n);
                float r = vec_dot_f32_f32(tmp, x, n);
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
        case GGUF_TYPE_Q4_0_4_4: return vec_dot_q4_0_4_4_f32(src, x, n);
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

/* ---- quantize_row_q4_0: FP32 -> Q4_0 blocks ---- */
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
        for (int j = 0; j < 16; j++) {
            uint8_t v0 = (uint8_t)(b[j * 2] * id + 8.5f);
            uint8_t v1 = (uint8_t)(b[j * 2 + 1] * id + 8.5f);
            q[j] = v0 | (v1 << 4);
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
        for (int j = 0; j < 32; j += 4) {
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
        for (int j = 0; j < 32; j += 4) {
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
