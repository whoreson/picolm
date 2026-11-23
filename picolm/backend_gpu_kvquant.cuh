#ifndef BACKEND_GPU_KVQUANT_CUH
#define BACKEND_GPU_KVQUANT_CUH

/* tq3_signs_dev, gpu_fp16_to_fp32/gpu_fp32_to_fp16, gpuShflXorSync and
 * gpuShflSync all come from here. Included rather than duplicated so the
 * sign pattern used to WRITE a TQ block cannot differ from the one used to
 * READ it back. */
#include "backend_gpu_dequant.cuh"

/* Device-side TurboQuant KV quantization: TQ3 (3.5 bits/value) and TQ4
 * (4.5 bits/value), byte-compatible with quantize_row_tq3/_tq4 in quant.c.
 *
 * The CPU algorithm, per 32-value block:
 *   1. rms  = sqrt(sum(x^2) / 32)
 *   2. n[i] = x[i] / rms
 *   3. rot  = WHT32(n) with the golden-ratio sign pattern, then * 1/sqrt(32)
 *   4. idx  = nearest Lloyd-Max codebook entry (8 for TQ3, 16 for TQ4)
 *   5. pack: 4 groups of 8 indices, 3 bytes (TQ3) or 4 bytes (TQ4)
 *   6. store 2-byte fp16 rms + the packed bytes
 *
 * On the GPU one warp handles one block, one lane per value:
 *   - the rms reduction is a warp XOR butterfly (every lane ends up with the
 *     same sum, so no broadcast is needed)
 *   - the WHT is the same butterfly in the *distributed* form, with each
 *     lane's value replaced by the pair sum/difference. Stage k shuffles the
 *     PREVIOUS stage's value, because the CPU WHT reuses its buffer in place:
 *     stage k reads stage k-1 of both partners, not this lane's own update.
 *   - packing gathers the 8 indices of each group into that group's lane 0.
 *
 * All plain C plus the gpu*() shims, so HIP compiles it unchanged.
 */

/* 32-point WHT in distributed form. Lane l holds element l of `v` on entry
 * and the transformed element l on exit (normalisation NOT applied: the
 * caller multiplies once by 1/sqrt(32), exactly as the CPU does). */
__device__ static inline float tq_wht32_warp(float v, int lane) {
    float cur = v;
    for (int off = 1; off < 32; off <<= 1) {
        float prev = cur;                       /* must be the pre-stage value */
        float o = gpuShflXorSync(0xFFFFFFFFu, prev, off);
        cur = (lane & off) ? (o - prev) : (prev + o);
    }
    return cur;
}

/* Nearest TQ3 codebook index: scan the 7 boundaries of TQ3_BOUNDARIES_7 with
 * a strict '>' so an exact boundary lands on the lower index, as the CPU
 * tq3_find_nearest() does. */
__device__ static inline uint8_t tq3_nearest(float v) {
    uint8_t idx = 0;
    if (v > -1.7479f) idx = 1;
    if (v > -1.0500f) idx = 2;
    if (v > -0.5005f) idx = 3;
    if (v >  0.0000f) idx = 4;
    if (v >  0.5005f) idx = 5;
    if (v >  1.0500f) idx = 6;
    if (v >  1.7479f) idx = 7;
    return idx;
}

/* Nearest TQ4 codebook index (TQ4_BOUNDARIES_15), same convention. */
__device__ static inline uint8_t tq4_nearest(float v) {
    uint8_t idx = 0;
    if (v > -2.4008f) idx = 1;
    if (v > -1.8435f) idx = 2;
    if (v > -1.4371f) idx = 3;
    if (v > -1.0993f) idx = 4;
    if (v > -0.7995f) idx = 5;
    if (v > -0.5224f) idx = 6;
    if (v > -0.2582f) idx = 7;
    if (v >  0.0000f) idx = 8;
    if (v >  0.2582f) idx = 9;
    if (v >  0.5224f) idx = 10;
    if (v >  0.7995f) idx = 11;
    if (v >  1.0993f) idx = 12;
    if (v >  1.4371f) idx = 13;
    if (v >  1.8435f) idx = 14;
    if (v >  2.4008f) idx = 15;
    return idx;
}

/* ---- TQ block quantisation, one warp ----
 *
 * v_in    : this lane's value (element `lane` of the 32-value block)
 * is_tq4  : 0 = TQ3 (8-entry codebook, 12 packed bytes)
 *           1 = TQ4 (16-entry codebook, 16 packed bytes)
 * dst     : pointer to the block's 2 + (12|16) bytes; only the owning lanes
 *           write, and every byte is written exactly once.
 *
 * Byte layout, identical to block_tq3 / block_tq4:
 *   [0..1]  fp16 rms          (lane 0)
 *   [2..]   4 groups of 8 indices; group g written by lane g*8
 *           TQ3: 3 bytes  idx0..idx7 -> 24 bits, little-endian bit stream
 *           TQ4: 4 bytes  idx0..idx7 -> 8 nibbles, low nibble first
 */
__device__ static inline void tq_quant_block_warp(float v_in, int lane,
                                                  int is_tq4, uint8_t *dst) {
    /* 1-2. rms reduction across the warp. */
    float sq = v_in * v_in;
    float sumsq = sq;
    for (int off = 16; off > 0; off >>= 1)
        sumsq += gpuShflXorSync(0xFFFFFFFFu, sumsq, off);
    float rms = sqrtf(sumsq * (1.0f / 32.0f));
    if (rms < 1e-10f) rms = 1e-10f;

    if (lane == 0) {
        uint16_t dh = gpu_fp32_to_fp16(rms);
        dst[0] = (uint8_t)(dh & 0xFF);
        dst[1] = (uint8_t)(dh >> 8);
    }

    /* 3. normalise, sign-flip, WHT, scale.
     * tq3_signs_dev[lane] is the same golden-ratio pattern the CPU uses. */
    float n  = v_in * (1.0f / rms) * tq3_signs_dev[lane];
    float r  = tq_wht32_warp(n, lane) * 0.176776695f;   /* 1/sqrt(32) */

    /* 4. nearest codebook entry. */
    int idx = is_tq4 ? (int)tq4_nearest(r) : (int)tq3_nearest(r);

    /* 5-6. pack. Every lane has its own idx; the 8 lanes of a group each
     * gather the group with 8 shuffles, so no two lanes race on a byte. */
    int grp = lane >> 3;                    /* which group of 8 (0..3) */
    int k   = lane & 7;                     /* index within the group  */
    int j0 = (int)gpuShflSync(0xFFFFFFFFu, (float)idx, grp * 8 + 0);
    int j1 = (int)gpuShflSync(0xFFFFFFFFu, (float)idx, grp * 8 + 1);
    int j2 = (int)gpuShflSync(0xFFFFFFFFu, (float)idx, grp * 8 + 2);
    int j3 = (int)gpuShflSync(0xFFFFFFFFu, (float)idx, grp * 8 + 3);
    int j4 = (int)gpuShflSync(0xFFFFFFFFu, (float)idx, grp * 8 + 4);
    int j5 = (int)gpuShflSync(0xFFFFFFFFu, (float)idx, grp * 8 + 5);
    int j6 = (int)gpuShflSync(0xFFFFFFFFu, (float)idx, grp * 8 + 6);
    int j7 = (int)gpuShflSync(0xFFFFFFFFu, (float)idx, grp * 8 + 7);
    (void)k;

    if (k == 0) {
        if (is_tq4) {
            uint8_t *d = dst + 2 + grp * 4;
            d[0] = (uint8_t)(j0 | (j1 << 4));
            d[1] = (uint8_t)(j2 | (j3 << 4));
            d[2] = (uint8_t)(j4 | (j5 << 4));
            d[3] = (uint8_t)(j6 | (j7 << 4));
        } else {
            uint8_t *d = dst + 2 + grp * 3;
            d[0] = (uint8_t)(j0 | (j1 << 3) | (j2 << 6));
            d[1] = (uint8_t)((j2 >> 2) | (j3 << 1) | (j4 << 4) | (j5 << 7));
            d[2] = (uint8_t)((j5 >> 1) | (j6 << 2) | (j7 << 5));
        }
    }
}

#endif /* BACKEND_GPU_KVQUANT_CUH */
