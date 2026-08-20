#ifndef BACKEND_GPU_DEQUANT_CUH
#define BACKEND_GPU_DEQUANT_CUH

#include "backend_gpu_common.cuh"

__device__ static inline float dequant_q4_0(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    const uint8_t *qs = b + 2;
    int v;
    if (i < 16) {
        v = qs[i] & 0xF;           /* low nibble: values 0-15 */
    } else {
        v = qs[i - 16] >> 4;       /* high nibble: values 16-31 */
    }
    return (float)(v - 8) * d;
}

/* block_q8_0: 34 bytes = 2B scale(FP16) + 32B qs (32 values)
 * dequant(i) = qs[i] * d */
__device__ static inline float dequant_q8_0(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    return (float)((const int8_t *)(b + 2))[i] * d;
}

/* block_q1_0: 18 bytes = 2B scale(FP16) + 16B qs (128 sign bits)
 * dequant(i) = (bit[i] ? +d : -d) */
__device__ static inline float dequant_q1_0(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    int bit = (b[2 + (i >> 3)] >> (i & 7)) & 1;
    return bit ? d : -d;
}

/* block_q2_0: 34 bytes = 2B scale(FP16) + 32B qs (128 values * 2 bits each)
 * dequant(i) = (qs[i] - 1) * d, where qs[i] in {0,1,2,3} -> {-d, 0, +d, +2d} */
__device__ static inline float dequant_q2_0(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    int byte_idx = i >> 2;          /* 4 values per byte */
    int shift = (i & 3) << 1;       /* 2 bits per value */
    int v = (b[2 + byte_idx] >> shift) & 3;
    return (float)(v - 1) * d;
}

/* F16: raw array of uint16_t, each value is an individual FP16 element.
 * No block structure: dequant(i) = gpu_fp16_to_fp32(weights[i]) */
__device__ static inline float dequant_f16(const void *weights, int i) {
    const uint16_t *w = (const uint16_t *)weights;
    return gpu_fp16_to_fp32(w[i]);
}

__device__ static inline float dequant_bf16(const void *weights, int i) {
    const uint16_t *w = (const uint16_t *)weights;
    /* Zero-extend BF16 to F32: shift upper 16 bits, reinterpret as float.
     * __bfloat162float is BROKEN on CUDA 13.0 / sm_121 ARM64 (returns raw
     * uint16 value, e.g. 16256 instead of 1.0 for 0x3f80).
     * Use portable bit manipulation matching the CPU bf16_to_fp32(). */
    union { uint32_t u; float f; } o;
    o.u = ((uint32_t)w[i]) << 16;
    return o.f;
}

/* block_q4_K: 144 bytes for 256 values
 * Layout from quant.h:
 *   uint16_t d;          offset 0  super-block scale (FP16)
 *   uint16_t dmin;       offset 2  super-block min (FP16)
 *   uint8_t  scales[12]; offset 4  packed 6-bit scales+mins (8 pairs)
 *   uint8_t  qs[128];    offset 16 quantized values (256 nibbles)
 *
 * 256 values = 4 superblocks of 64 values each.
 * Each superblock has 2 sub-blocks of 32 values, each with its own scale+min.
 * dequant(i) = d * scale[j] + qs[i] - dmin * min[j]
 * where qs[i] is a raw nibble [0..15], NOT offset by -8.
 *
 * get_scale_min_k4(j, scales, &sc, &mn) extracts scale and min for sub-block j.
 * j=0..7 maps to 8 sub-blocks across 4 superblocks. */
__device__ static inline void gpu_get_scale_min_k4(int j, const uint8_t *scales, uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *mn = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

/* Q2_K dequant: 256 values in 84 bytes.
 * Layout: scales[16] (4-bit sc+mn each), qs[64] (2-bit quants), d (fp16), dmin (fp16).
 * Structure: 2 chunks of 128 values each.
 * Each chunk: 4 groups of 16 values. Each group uses a bit-shift (0,2,4,6) over 32 qs bytes.
 *   Group 0: (qs[0..15] >> 0) & 3, scale[0]
 *   Group 1: (qs[16..31] >> 0) & 3, scale[1]
 *   Group 2: (qs[0..15] >> 2) & 3, scale[2]
 *   Group 3: (qs[16..31] >> 2) & 3, scale[3]
 *   Group 4: (qs[0..15] >> 4) & 3, scale[4]
 *   Group 5: (qs[16..31] >> 4) & 3, scale[5]
 *   Group 6: (qs[0..15] >> 6) & 3, scale[6]
 *   Group 7: (qs[16..31] >> 6) & 3, scale[7]
 * Then same pattern for second chunk (qs[32..63], scales[8..15]).
 * Each scale byte: low nibble = scale, high nibble = min.
 * Result = d * sc * q - dmin * mn. */
__device__ static inline float dequant_q2_K(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    const uint8_t *scales = b;
    const uint8_t *qs = b + 16;
    uint16_t d_raw = b[80] | ((uint16_t)b[81] << 8);
    uint16_t dm_raw = b[82] | ((uint16_t)b[83] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    float dmin = gpu_fp16_to_fp32(dm_raw);

    /* i in 0..255 */
    int chunk = i / 128;              /* 0 or 1 */
    int i_in_chunk = i % 128;         /* 0..127 */
    int group = i_in_chunk / 16;      /* 0..7 */
    int l = i_in_chunk % 16;          /* 0..15 */

    int shift = (group % 4) * 2;      /* 0, 2, 4, 6 */
    int half = group / 4;             /* 0 or 1 (first or second half of qs bytes) */
    int scale_idx = chunk * 8 + group; /* 0..15 */

    uint8_t sc_byte = scales[scale_idx];
    uint8_t sc = sc_byte & 0xF;
    uint8_t mn = sc_byte >> 4;

    int qs_base = chunk * 32 + half * 16;
    int v = (qs[qs_base + l] >> shift) & 3;

    return d * (float)sc * (float)v - dmin * (float)mn;
}

__device__ static inline float dequant_q4_K(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    uint16_t dmin_raw = b[2] | ((uint16_t)b[3] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    float dmin = gpu_fp16_to_fp32(dmin_raw);
    const uint8_t *scales = b + 4;
    const uint8_t *qs = b + 16;

    int superblk = i / 64;
    int subblk = (i % 64) / 32;
    int offset_in_sub = i % 32;
    int scale_idx = superblk * 2 + subblk;

    uint8_t sc, mn;
    gpu_get_scale_min_k4(scale_idx, scales, &sc, &mn);

    int qs_byte_idx = superblk * 32 + offset_in_sub;
    int v = (qs_byte_idx < 128) ? (subblk == 0 ? (qs[qs_byte_idx] & 0xF) : (qs[qs_byte_idx] >> 4)) : 0;

    return d * (float)sc * (float)v - dmin * (float)mn;
}

/* Q6_K dequant: 256 values in 210 bytes.
 * Layout: ql[128] (4-bit low), qh[64] (2-bit high), scales[16] (int8), d (fp16).
 * Each element: 6-bit signed value (biased by 32: stored [0..63], actual [-32..31]).
 * 16 scales, one per 16-element sub-block.
 * Result = d * scale[scale_idx] * (quantized_value - 32). */
__device__ static inline float dequant_q6_K(const void *blk, int i) {
    const uint8_t *ql = (const uint8_t *)blk;
    const uint8_t *qh = ql + 128;
    const int8_t *scales = (const int8_t *)(qh + 64);
    uint16_t d_raw = ((const uint16_t *)(scales + 16))[0];
    float d = gpu_fp16_to_fp32(d_raw);

    /* Which 128-element half, and which element within it */
    int chunk = i / 128;            /* 0 or 1 */
    int l = i % 128;                /* 0..127 */

    const uint8_t *ql_c = ql + chunk * 64;
    const uint8_t *qh_c = qh + chunk * 32;

    /* Determine which of the 4 elements (q1/q2/q3/q4) this is */
    int sub_chunk = l / 32;         /* 0=q1, 1=q2, 2=q3, 3=q4 */
    int sub_l = l % 32;             /* 0..31 */
    int is = chunk * 8;             /* scale chunk base: 0 or 8 */
    int sc_idx = is + (sub_chunk * 2) + (sub_l / 16);  /* 0..15 */

    /* Extract 6-bit value: 4 bits from ql, 2 bits from qh */
    uint8_t ql_val, qh_val;
    if (sub_chunk == 0) {           /* q1: ql[sub_l]&0xF, qh[sub_l]>>0 */
        ql_val = ql_c[sub_l] & 0xF;
        qh_val = (qh_c[sub_l] >> 0) & 3;
    } else if (sub_chunk == 1) {    /* q2: ql[sub_l+32]&0xF, qh[sub_l]>>2 */
        ql_val = ql_c[sub_l + 32] & 0xF;
        qh_val = (qh_c[sub_l] >> 2) & 3;
    } else if (sub_chunk == 2) {    /* q3: ql[sub_l]>>4, qh[sub_l]>>4 */
        ql_val = ql_c[sub_l] >> 4;
        qh_val = (qh_c[sub_l] >> 4) & 3;
    } else {                        /* q4: ql[sub_l+32]>>4, qh[sub_l]>>6 */
        ql_val = ql_c[sub_l + 32] >> 4;
        qh_val = (qh_c[sub_l] >> 6) & 3;
    }

    int q = (ql_val | (qh_val << 4)) - 32;
    return d * (float)scales[sc_idx] * (float)q;
}

/* Q5_K dequant: 256 values in 176 bytes.
 * Layout: d (fp16), dm (fp16), scales[12] (packed 6-bit sc+mn via get_scale_min_k4),
 *         qh[32] (1 high bit per quant), qs[128] (4-bit low nibbles, 2 per byte).
 * 4 sub-blocks of 64, each with 2 scales+mins (for two groups of 32).
 * Result = d * sc * (qs_nibble + high_bit*16) - dm * mn. */
__device__ static inline float dequant_q5_K(const void *blk, int i) {
    const uint8_t *b = (const uint8_t *)blk;
    uint16_t d_raw = b[0] | ((uint16_t)b[1] << 8);
    uint16_t dm_raw = b[2] | ((uint16_t)b[3] << 8);
    float d = gpu_fp16_to_fp32(d_raw);
    float dm = gpu_fp16_to_fp32(dm_raw);
    const uint8_t *scales = b + 4;   /* 12 bytes packed */
    const uint8_t *qh = b + 16;     /* 32 bytes, 1 high bit per quant */
    const uint8_t *qs = b + 48;     /* 128 bytes, 2 nibbles per byte */

    /* i in 0..255. 4 groups of 64, each group has 2 sub-groups of 32.
     * group = i / 64 (0..3), pos = i % 64 (0..63).
     * Within group: sub = pos / 32 (0 or 1), l = pos % 32 (0..31).
     * scale_idx = group * 2 + sub (0..7 out of 12 packed entries, but only 8 used for 4*2).
     */
    int group = i / 64;
    int sub = (i % 64) / 32;
    int l = i % 32;
    int scale_idx = group * 2 + sub;

    uint8_t sc, mn;
    gpu_get_scale_min_k4(scale_idx, scales, &sc, &mn);

    /* qs layout: for each group of 64, 32 bytes hold the low nibbles.
     * sub=0 reads qs[group*32 + l] & 0xF, sub=1 reads qs[group*32 + l] >> 4.
     * qh layout: 32 bytes, one bit per quant. The bit position cycles.
     * For group g, sub s, position l:
     *   The CPU code shifts u1/u2 by 2 bits per group iteration.
     *   u1 starts at 1, u2 starts at 2, each multiplied by 4 per 64-element chunk.
     *   For the first 32 (sub=0): qh_bit = (qh[l] & u1) != 0
     *   For the next 32 (sub=1): qh_bit = (qh[l] & u2) != 0
     *   u1 = 1 << (2*group), u2 = 2 << (2*group)
     */
    int nibble = sub == 0 ? (qs[group * 32 + l] & 0xF) : (qs[group * 32 + l] >> 4);
    int u = (sub == 0) ? (1 << (2 * group)) : (2 << (2 * group));
    int high_bit = (qh[l] & u) ? 16 : 0;

    return d * (float)sc * (float)(nibble + high_bit) - dm * (float)mn;
}

/* ---- quant_matmul kernel ----
 *
 * Simple, correct GEMV kernel for all quant formats.
 * 1 thread per output element, shared-memory tree reduce.
 * Not bandwidth-optimal (dequantizes weights on-the-fly) but universally correct.
 *
 * Grid: [O, S], Block: 256 threads
 * Each thread computes y[s*O + o] = sum_i x[s*I + i] * W[o][i]
 */


#endif /* BACKEND_GPU_DEQUANT_CUH */
