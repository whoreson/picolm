#include "model.h"
#include "tensor.h"
#include "quant.h"
#include "model_internal.h"
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#include <assert.h>
#endif

#ifdef PICOLM_GPU
#include "backend_gpu.h"
#endif

#ifdef PICOLM_VIZ
#include "viz.h"
#endif


/* ================================================================
 * Forward pass with:
 *   - FP16 KV cache (halves memory bandwidth in attention)
 *   - Flash attention / online softmax (single pass, no score buffer)
 *   - Pre-computed RoPE tables (table lookup instead of trig)
 * ================================================================ */


/* Threaded attention: per-head online softmax K.dot.Q + weighted V.
 * Each head is fully independent -> parallelized via tensor_parallel_for.
 *
 * attn_core() is the shared math: given one query vector, one KV head,
 * and a causal limit `pos`, it scans t=0..pos in the KV cache and writes
 * the attention output for that (token, head) into xbh. Both the decode
 * path (model_forward, one token at a time) and the prefill path
 * (model_forward_prefill, many tokens at once) call this same core, so
 * there is exactly one implementation of the quant-aware SIMD attention
 * math to maintain and both paths get the same optimizations for free. */
void attn_core(
        float *xbh, const float *qh, int kv_h, int pos,
        const uint8_t *kcache, const uint8_t *vcache,
        int kv_type_k, int kv_type_v,
        size_t kv_row_size_k, size_t kv_row_size_v,
        size_t kv_head_stride_k, size_t kv_head_stride_v,
        int head_dim, float attn_scale, int n_swa) {
    float max_score = -1e30f, sum_exp = 0.0f;
    float acc[256];
    memset(acc, 0, (size_t)head_dim * sizeof(float));

    /* Sliding window: restrict the causal scan [0..pos] down to
     * [max(0, pos - n_swa + 1) .. pos] when this layer is SWA (n_swa > 0).
     * n_swa == 0 means full attention -- start stays at 0, identical to
     * the pre-SWA behavior, so this is a strict no-op for every other
     * model family. */
    int t_start = (n_swa > 0) ? (pos - n_swa + 1) : 0;
    if (t_start < 0) t_start = 0;

    for (int t = t_start; t <= pos; t++) {
        /* GQA layout: [pos] * kv_row_size + head * kv_head_stride */
        const uint8_t *kt = kcache + (size_t)t * kv_row_size_k + kv_h * kv_head_stride_k;
        float score;
        if (kv_type_k == KV_CACHE_Q8_0) score = vec_dot_q8_0_f32(kt, qh, head_dim);
        else if (kv_type_k == KV_CACHE_Q4_0) score = vec_dot_q4_0_f32(kt, qh, head_dim);
        else if (kv_type_k == KV_CACHE_TQ3) {
            /* TQ3 K: dequant to F32 then dot. The vec_dot_tq3_f32 codebook
             * approach loses too much accuracy for attention scoring when
             * used alone (TQ3 K + non-TQ3 V). Full dequant is needed. */
            float k_f32_local[256];
            memset(k_f32_local, 0, (size_t)head_dim * sizeof(float));
            scale_add_tq3_f32(k_f32_local, 1.0f, kt, head_dim);
            score = 0;
            for (int d = 0; d < head_dim; d++) score += qh[d] * k_f32_local[d];
        }
        else if (kv_type_k == KV_CACHE_TQ4) {
            /* TQ4 K: codebook-lookup dot product. Q must be pre-rotated
             * with WHT forward (block=32). The 16-entry codebook provides
             * sufficient accuracy (~2% dot product error) for attention. */
            score = vec_dot_tq4_f32(kt, qh, head_dim);
        }
        else score = vec_dot_f16_f32(kt, qh, head_dim);
        score *= attn_scale;
        const uint8_t *vt = vcache + (size_t)t * kv_row_size_v + kv_h * kv_head_stride_v;
        if (score > max_score) {
            float correction = expf(max_score - score);
            sum_exp = sum_exp * correction + 1.0f;
            if (kv_type_v == KV_CACHE_Q8_0) fma_scale_q8_0_f32(acc, correction, vt, head_dim);
            else if (kv_type_v == KV_CACHE_Q4_0) fma_scale_q4_0_f32(acc, correction, vt, head_dim);
            else if (kv_type_v == KV_CACHE_TQ3) fma_scale_tq3_f32(acc, correction, vt, head_dim);
            else if (kv_type_v == KV_CACHE_TQ4) fma_scale_tq4_f32(acc, correction, vt, head_dim);
            else {
                const uint16_t *vt16 = (const uint16_t *)vt;
#ifdef PICOLM_AVX512
                { __m512 cv = _mm512_set1_ps(correction); int d = 0;
                  for (; d + 15 < head_dim; d += 16) { __m512 vf = fp16x16_to_fp32_inline(vt16 + d); __m512 af = _mm512_loadu_ps(acc + d); _mm512_storeu_ps(acc + d, _mm512_fmadd_ps(af, cv, vf)); }
                  for (; d < head_dim; d++) acc[d] = fmaf(acc[d], correction, fp16_to_fp32(vt16[d])); }
#elif defined(PICOLM_AVX)
                { __m256 cv = _mm256_set1_ps(correction); int d = 0;
                  for (; d + 7 < head_dim; d += 8) { __m256 vf = fp16x8_to_fp32_inline(vt16 + d); __m256 af = _mm256_loadu_ps(acc + d); _mm256_storeu_ps(acc + d, _mm256_add_ps(_mm256_mul_ps(af, cv), vf)); }
                  for (; d < head_dim; d++) acc[d] = fmaf(acc[d], correction, fp16_to_fp32(vt16[d])); }
#elif defined(PICOLM_NEON_AARCH64)
                { int d = 0;
                  float32x4_t cv = vdupq_n_f32(correction);
                  for (; d + 3 < head_dim; d += 4) {
                      float16x4_t hf = vld1_f16((const float16_t *)(vt16 + d));
                      float32x4_t vf = vcvt_f32_f16(hf);
                      float32x4_t af = vld1q_f32(acc + d);
                      vst1q_f32(acc + d, vaddq_f32(vmulq_f32(af, cv), vf));
                  }
                  for (; d < head_dim; d++) acc[d] = fmaf(acc[d], correction, fp16_to_fp32(vt16[d]));
                }
#elif defined(PICOLM_NEON)
                { int d = 0;
                  float32x4_t cv = vdupq_n_f32(correction);
                  for (; d + 3 < head_dim; d += 4) {
                      float32x4_t vf = fp16x4_to_fp32_inline(vt16 + d);
                      float32x4_t af = vld1q_f32(acc + d);
                      vst1q_f32(acc + d, vaddq_f32(vmulq_f32(af, cv), vf));
                  }
                  for (; d < head_dim; d++) acc[d] = fmaf(acc[d], correction, fp16_to_fp32(vt16[d]));
                }
#else
                for (int d = 0; d < head_dim; d++) acc[d] = fmaf(acc[d], correction, fp16_to_fp32(vt16[d]));
#endif
            }
            max_score = score;
        } else {
            float w = expf(score - max_score);
            sum_exp += w;
            if (kv_type_v == KV_CACHE_Q8_0) scale_add_q8_0_f32(acc, w, vt, head_dim);
            else if (kv_type_v == KV_CACHE_Q4_0) scale_add_q4_0_f32(acc, w, vt, head_dim);
            else if (kv_type_v == KV_CACHE_TQ3) scale_add_tq3_f32(acc, w, vt, head_dim);
            else if (kv_type_v == KV_CACHE_TQ4) scale_add_tq4_f32(acc, w, vt, head_dim);
            else {
                const uint16_t *vt16 = (const uint16_t *)vt;
#ifdef PICOLM_AVX512
                { __m512 wv = _mm512_set1_ps(w); int d = 0;
                  for (; d + 15 < head_dim; d += 16) { __m512 vf = fp16x16_to_fp32_inline(vt16 + d); __m512 af = _mm512_loadu_ps(acc + d); _mm512_storeu_ps(acc + d, _mm512_fmadd_ps(vf, wv, af)); }
                  for (; d < head_dim; d++) acc[d] = fmaf(w, fp16_to_fp32(vt16[d]), acc[d]); }
#elif defined(PICOLM_AVX)
                { __m256 wv = _mm256_set1_ps(w); int d = 0;
                  for (; d + 7 < head_dim; d += 8) { __m256 vf = fp16x8_to_fp32_inline(vt16 + d); __m256 af = _mm256_loadu_ps(acc + d); _mm256_storeu_ps(acc + d, _mm256_add_ps(_mm256_mul_ps(vf, wv), af)); }
                  for (; d < head_dim; d++) acc[d] = fmaf(w, fp16_to_fp32(vt16[d]), acc[d]); }
#elif defined(PICOLM_NEON_AARCH64)
                { int d = 0;
                  float32x4_t wv = vdupq_n_f32(w);
                  for (; d + 3 < head_dim; d += 4) {
                      float16x4_t hf = vld1_f16((const float16_t *)(vt16 + d));
                      float32x4_t vf = vcvt_f32_f16(hf);
                      float32x4_t af = vld1q_f32(acc + d);
                      vst1q_f32(acc + d, vaddq_f32(af, vmulq_f32(vf, wv)));
                  }
                  for (; d < head_dim; d++) acc[d] = fmaf(w, fp16_to_fp32(vt16[d]), acc[d]);
                }
#elif defined(PICOLM_NEON)
                { int d = 0;
                  float32x4_t wv = vdupq_n_f32(w);
                  for (; d + 3 < head_dim; d += 4) {
                      float32x4_t vf = fp16x4_to_fp32_inline(vt16 + d);
                      float32x4_t af = vld1q_f32(acc + d);
                      vst1q_f32(acc + d, vaddq_f32(af, vmulq_f32(vf, wv)));
                  }
                  for (; d < head_dim; d++) acc[d] = fmaf(w, fp16_to_fp32(vt16[d]), acc[d]);
                }
#else
                for (int d = 0; d < head_dim; d++) acc[d] = fmaf(w, fp16_to_fp32(vt16[d]), acc[d]);
#endif
            }
        }
    }
    float inv_sum = 1.0f / sum_exp;
#ifdef PICOLM_AVX512
    { __m512 inv = _mm512_set1_ps(inv_sum); int d = 0;
      for (; d + 15 < head_dim; d += 16) { __m512 af = _mm512_loadu_ps(acc + d); _mm512_storeu_ps(xbh + d, _mm512_mul_ps(af, inv)); }
      for (; d < head_dim; d++) xbh[d] = acc[d] * inv_sum; }
#elif defined(PICOLM_AVX)
    { __m256 inv = _mm256_set1_ps(inv_sum); int d = 0;
      for (; d + 7 < head_dim; d += 8) { __m256 af = _mm256_loadu_ps(acc + d); _mm256_storeu_ps(xbh + d, _mm256_mul_ps(af, inv)); }
      for (; d < head_dim; d++) xbh[d] = acc[d] * inv_sum; }
#elif defined(PICOLM_NEON)
    { float32x4_t inv = vdupq_n_f32(inv_sum); int d = 0;
      for (; d + 3 < head_dim; d += 4) { float32x4_t af = vld1q_f32(acc + d); vst1q_f32(xbh + d, vmulq_f32(af, inv)); }
      for (; d < head_dim; d++) xbh[d] = acc[d] * inv_sum; }
#elif defined(PICOLM_SSE2)
    { __m128 inv = _mm_set1_ps(inv_sum); int d = 0;
      for (; d + 3 < head_dim; d += 4) { __m128 af = _mm_loadu_ps(acc + d); _mm_storeu_ps(xbh + d, _mm_mul_ps(af, inv)); }
      for (; d < head_dim; d++) xbh[d] = acc[d] * inv_sum; }
#else
    for (int d = 0; d < head_dim; d++) xbh[d] = acc[d] * inv_sum;
#endif
}

/* ================================================================
 * GQA-grouped attention: process all kv_mul Q heads sharing a KV
 * head in a single pass over the KV cache. This reduces KV cache
 * memory bandwidth by kv_mul x (4x for Fimbulvetr2-11B).
 *
 * For each position t in the KV cache, we load K[t] once and compute
 * kv_mul dot products against the kv_mul Q heads. Similarly V[t] is
 * loaded once and accumulated into kv_mul output vectors.
 * ================================================================ */

void attention_group(int kv_head_idx, void *ctx_ptr) {
    attn_group_ctx_t *ctx = (attn_group_ctx_t *)ctx_ptr;
    int kv_h = kv_head_idx;
    int kv_mul = ctx->kv_mul;
    int head_dim = ctx->head_dim;
    int pos = ctx->pos;
    int first_qh = kv_h * kv_mul;
    size_t kv_head_stride_k = ctx->kv_head_stride_k;
    size_t kv_head_stride_v = ctx->kv_head_stride_v;
    
    /* Per-Q-head softmax state (kv_mul up to 8) */
    assert(kv_mul <= 8 && head_dim <= 256 && "attention_group: stack arrays too small for this model");
    float max_score[8], sum_exp[8];
    for (int g = 0; g < kv_mul; g++) {
        max_score[g] = -1e30f;
        sum_exp[g] = 0.0f;
    }
    float acc[8][256];
    for (int g = 0; g < kv_mul; g++)
        for (int d = 0; d < head_dim; d++) acc[g][d] = 0.0f;

    /* Sliding window: see attn_core() above for the identical rationale. */
    int t_start = (ctx->n_swa > 0) ? (pos - ctx->n_swa + 1) : 0;
    if (t_start < 0) t_start = 0;

    for (int t = t_start; t <= pos; t++) {
        /* GQA layout: [pos] * kv_row_size_k + head * kv_head_stride_k */
        const uint8_t *kt = ctx->kcache + (size_t)t * ctx->kv_row_size_k + kv_h * kv_head_stride_k;
        const uint8_t *vt = ctx->vcache + (size_t)t * ctx->kv_row_size_v + kv_h * kv_head_stride_v;

#ifdef PICOLM_AVX512
        /* AVX-512 fast path: dequantize K and V to F32 vectors,
         * then use FMA-based attention math. Supports FP16, Q8_0, Q4_0.
         * FP16 uses SIMD fp16x16_to_fp32_inline for speed;
         * Q8_0/Q4_0 use scalar dequant (head_dim is small, overhead is OK). */
        float k_f32[256], v_f32[256];
        if (ctx->kv_type_k == KV_CACHE_Q8_0) {
            dequantize_row_q8_0(kt, k_f32, head_dim);
        } else if (ctx->kv_type_k == KV_CACHE_Q4_0) {
            dequantize_row_q4_0(kt, k_f32, head_dim);
        } else if (ctx->kv_type_k == KV_CACHE_TQ3) {
            /* TQ3 K: use vec_dot_tq3_f32 directly (Q is pre-rotated) */
            /* For AVX512 path, dequant to f32 for uniform dot product */
            float k_f32_tq3[256];
            for (int hkv_block = 0; hkv_block < head_dim; hkv_block += TQ3_BLOCK_SIZE) {
                const block_tq3 *tb = (const block_tq3 *)kt;
                /* Inline TQ3 dequant per block */
                const block_tq3 *blk = &tb[hkv_block / TQ3_BLOCK_SIZE];
                float sc = fp16_to_fp32(blk->d);
                float rot[32];
                for (int gg = 0; gg < 4; gg++) {
                    uint8_t idx8[8];
                    tq3_unpack_3bit_8(idx8, blk->qs + gg * 3);
                    for (int kk = 0; kk < 8; kk++)
                        rot[gg * 8 + kk] = tq3_codebook[idx8[kk]];
                }
                /* WHT inverse */
                {
                    float tmp2[32]; memcpy(tmp2, rot, sizeof(tmp2));
                    for (int step = 1; step < 32; step <<= 1)
                        for (int ii = 0; ii < 32; ii += step << 1)
                            for (int jj = ii; jj < ii + step; jj++) {
                                float aa = tmp2[jj], bb = tmp2[jj + step];
                                tmp2[jj] = aa + bb; tmp2[jj + step] = aa - bb;
                            }
                    const float norm = 1.0f / 5.656854f; /* 1/sqrt(32) */
                    for (int ii = 0; ii < 32; ii++)
                        k_f32_tq3[hkv_block + ii] = tmp2[ii] * norm * tq3_signs[ii] * sc;
                }
            }
            memcpy(k_f32, k_f32_tq3, head_dim * sizeof(float));
        } else if (ctx->kv_type_k == KV_CACHE_TQ4) {
            /* TQ4 K: Q is pre-rotated with WHT forward (block=32).
             * Keep K in the rotated domain for the dot product:
             * just scale the codebook values (no inverse WHT). */
            for (int hkv_block = 0; hkv_block < head_dim; hkv_block += TQ4_BLOCK_SIZE) {
                const block_tq4 *blk = (const block_tq4 *)((const uint8_t *)kt +
                    (hkv_block / TQ4_BLOCK_SIZE) * sizeof(block_tq4));
                float sc = fp16_to_fp32(blk->d);
                for (int gg = 0; gg < 4; gg++) {
                    uint8_t idx8[8];
                    tq4_unpack_4bit_8(idx8, blk->qs + gg * 4);
                    for (int kk = 0; kk < 8; kk++)
                        k_f32[hkv_block + gg * 8 + kk] = tq4_codebook[idx8[kk]] * sc;
                }
            }
        } else {
            /* FP16: SIMD-accelerated conversion */
            const uint16_t *k16 = (const uint16_t *)kt;
            int d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 kf = fp16x16_to_fp32_inline(k16 + d);
                _mm512_storeu_ps(k_f32 + d, kf);
            }
            for (; d < head_dim; d++) k_f32[d] = fp16_to_fp32(k16[d]);
        }

        if (ctx->kv_type_v == KV_CACHE_Q8_0) {
            dequantize_row_q8_0(vt, v_f32, head_dim);
        } else if (ctx->kv_type_v == KV_CACHE_Q4_0) {
            dequantize_row_q4_0(vt, v_f32, head_dim);
        } else if (ctx->kv_type_v == KV_CACHE_TQ3) {
            /* TQ3 V: dequant to f32 for uniform path */
            for (int hkv_block = 0; hkv_block < head_dim; hkv_block += TQ3_BLOCK_SIZE) {
                float rot[32];
                const block_tq3 *blk = (const block_tq3 *)((const uint8_t *)vt +
                    (hkv_block / TQ3_BLOCK_SIZE) * sizeof(block_tq3));
                float sc = fp16_to_fp32(blk->d);
                for (int gg = 0; gg < 4; gg++) {
                    uint8_t idx8[8];
                    tq3_unpack_3bit_8(idx8, blk->qs + gg * 3);
                    for (int kk = 0; kk < 8; kk++)
                        rot[gg * 8 + kk] = tq3_codebook[idx8[kk]];
                }
                {
                    float tmp2[32]; memcpy(tmp2, rot, sizeof(tmp2));
                    for (int step = 1; step < 32; step <<= 1)
                        for (int ii = 0; ii < 32; ii += step << 1)
                            for (int jj = ii; jj < ii + step; jj++) {
                                float aa = tmp2[jj], bb = tmp2[jj + step];
                                tmp2[jj] = aa + bb; tmp2[jj + step] = aa - bb;
                            }
                    const float norm = 1.0f / 5.656854f;
                    for (int ii = 0; ii < 32; ii++)
                        v_f32[hkv_block + ii] = tmp2[ii] * norm * tq3_signs[ii] * sc;
                }
            }
        } else if (ctx->kv_type_v == KV_CACHE_TQ4) {
            /* TQ4 V: dequant to f32 for AVX512 path */
            for (int hkv_block = 0; hkv_block < head_dim; hkv_block += TQ4_BLOCK_SIZE) {
                const block_tq4 *blk = (const block_tq4 *)((const uint8_t *)vt +
                    (hkv_block / TQ4_BLOCK_SIZE) * sizeof(block_tq4));
                float sc = fp16_to_fp32(blk->d);
                float rot[32];
                for (int gg = 0; gg < 4; gg++) {
                    uint8_t idx8[8];
                    tq4_unpack_4bit_8(idx8, blk->qs + gg * 4);
                    for (int kk = 0; kk < 8; kk++)
                        rot[gg * 8 + kk] = tq4_codebook[idx8[kk]];
                }
                {
                    float tmp2[32]; memcpy(tmp2, rot, sizeof(tmp2));
                    for (int step = 1; step < 32; step <<= 1)
                        for (int ii = 0; ii < 32; ii += step << 1)
                            for (int jj = ii; jj < ii + step; jj++) {
                                float aa = tmp2[jj], bb = tmp2[jj + step];
                                tmp2[jj] = aa + bb; tmp2[jj + step] = aa - bb;
                            }
                    const float norm = 1.0f / 5.656854f;
                    for (int ii = 0; ii < 32; ii++)
                        v_f32[hkv_block + ii] = tmp2[ii] * norm * tq3_signs[ii] * sc;
                }
            }
        } else {
            /* FP16: SIMD-accelerated conversion */
            const uint16_t *v16 = (const uint16_t *)vt;
            int d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 vf = fp16x16_to_fp32_inline(v16 + d);
                _mm512_storeu_ps(v_f32 + d, vf);
            }
            for (; d < head_dim; d++) v_f32[d] = fp16_to_fp32(v16[d]);
        }

        for (int g = 0; g < kv_mul; g++) {
            const float *qg = ctx->q + (first_qh + g) * head_dim;
            float score = 0.0f;

            /* K.dot.Q */
            {
                __m512 s = _mm512_setzero_ps();
                int d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 kf = _mm512_loadu_ps(k_f32 + d);
                    __m512 qf = _mm512_loadu_ps(qg + d);
                    s = _mm512_fmadd_ps(kf, qf, s);
                }
                score = _mm512_reduce_add_ps(s);
                for (; d < head_dim; d++)
                    score += k_f32[d] * qg[d];
            }
            score *= ctx->attn_scale;

            float *accg = acc[g];

            if (score > max_score[g]) {
                float correction = expf(max_score[g] - score);
                sum_exp[g] = sum_exp[g] * correction + 1.0f;
                __m512 cv = _mm512_set1_ps(correction);
                int d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 af = _mm512_loadu_ps(accg + d);
                    __m512 vf = _mm512_loadu_ps(v_f32 + d);
                    _mm512_storeu_ps(accg + d, _mm512_fmadd_ps(af, cv, vf));
                }
                for (; d < head_dim; d++)
                    accg[d] = fmaf(accg[d], correction, v_f32[d]);
                max_score[g] = score;
            } else {
                float w = expf(score - max_score[g]);
                sum_exp[g] += w;
                __m512 wv = _mm512_set1_ps(w);
                int d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 af = _mm512_loadu_ps(accg + d);
                    __m512 vf = _mm512_loadu_ps(v_f32 + d);
                    _mm512_storeu_ps(accg + d, _mm512_fmadd_ps(vf, wv, af));
                }
                for (; d < head_dim; d++)
                    accg[d] = fmaf(w, v_f32[d], accg[d]);
            }
        }
#else
        /* Fallback: per-Q-head processing (same as attn_core, no grouping benefit) */
        for (int g = 0; g < kv_mul; g++) {
            const float *qg = ctx->q + (first_qh + g) * head_dim;
            float score;
            if (ctx->kv_type_k == KV_CACHE_Q8_0) score = vec_dot_q8_0_f32(kt, qg, head_dim);
            else if (ctx->kv_type_k == KV_CACHE_Q4_0) score = vec_dot_q4_0_f32(kt, qg, head_dim);
            else if (ctx->kv_type_k == KV_CACHE_TQ3) {
                float k_f32_local[256];
                memset(k_f32_local, 0, (size_t)head_dim * sizeof(float));
                scale_add_tq3_f32(k_f32_local, 1.0f, kt, head_dim);
                score = 0;
                for (int d = 0; d < head_dim; d++) score += qg[d] * k_f32_local[d];
            }
            else if (ctx->kv_type_k == KV_CACHE_TQ4) {
                /* TQ4 K: use codebook-lookup dot product (Q is pre-rotated) */
                score = vec_dot_tq4_f32(kt, qg, head_dim);
            }
            else score = vec_dot_f16_f32(kt, qg, head_dim);
            score *= ctx->attn_scale;

            float *accg = acc[g];
            if (score > max_score[g]) {
                float correction = expf(max_score[g] - score);
                sum_exp[g] = sum_exp[g] * correction + 1.0f;
                if (ctx->kv_type_v == KV_CACHE_Q8_0) fma_scale_q8_0_f32(accg, correction, vt, head_dim);
                else if (ctx->kv_type_v == KV_CACHE_Q4_0) fma_scale_q4_0_f32(accg, correction, vt, head_dim);
                else if (ctx->kv_type_v == KV_CACHE_TQ3) fma_scale_tq3_f32(accg, correction, vt, head_dim);
                else if (ctx->kv_type_v == KV_CACHE_TQ4) fma_scale_tq4_f32(accg, correction, vt, head_dim);
                else {
                    const uint16_t *vt16 = (const uint16_t *)vt;
                    for (int d = 0; d < head_dim; d++) accg[d] = fmaf(accg[d], correction, fp16_to_fp32(vt16[d]));
                }
                max_score[g] = score;
            } else {
                float w = expf(score - max_score[g]);
                sum_exp[g] += w;
                if (ctx->kv_type_v == KV_CACHE_Q8_0) scale_add_q8_0_f32(accg, w, vt, head_dim);
                else if (ctx->kv_type_v == KV_CACHE_Q4_0) scale_add_q4_0_f32(accg, w, vt, head_dim);
                else if (ctx->kv_type_v == KV_CACHE_TQ3) scale_add_tq3_f32(accg, w, vt, head_dim);
                else if (ctx->kv_type_v == KV_CACHE_TQ4) scale_add_tq4_f32(accg, w, vt, head_dim);
                else {
                    const uint16_t *vt16 = (const uint16_t *)vt;
                    for (int d = 0; d < head_dim; d++) accg[d] = fmaf(w, fp16_to_fp32(vt16[d]), accg[d]);
                }
            }
        }
#endif
    }

    /* Normalize and write output */
    for (int g = 0; g < kv_mul; g++) {
        float inv_sum = 1.0f / sum_exp[g];
        float *accg = acc[g];
        float *xbhg = ctx->xb + (first_qh + g) * head_dim;
#ifdef PICOLM_AVX512
        { __m512 inv = _mm512_set1_ps(inv_sum); int d = 0;
          for (; d + 16 <= head_dim; d += 16) { __m512 af = _mm512_loadu_ps(accg + d); _mm512_storeu_ps(xbhg + d, _mm512_mul_ps(af, inv)); }
          for (; d < head_dim; d++) xbhg[d] = accg[d] * inv_sum; }
#else
        for (int d = 0; d < head_dim; d++) xbhg[d] = accg[d] * inv_sum;
#endif
    }
}

/* ---- MoE Forward Pass ---- */

static void prefill_attn_task(int flat_idx, void *ctx_ptr) {
    prefill_attn_ctx_t *ctx = (prefill_attn_ctx_t *)ctx_ptr;
    int bi = flat_idx / ctx->n_heads;
    int h  = flat_idx % ctx->n_heads;
    int pos = ctx->start_pos + bi;
    int kv_h = h / ctx->kv_mul;
    const float *qh = ctx->q_batch + (size_t)bi * ctx->n_heads * ctx->head_dim + h * ctx->head_dim;
    float *xbh = ctx->xb_batch + (size_t)bi * ctx->xb_stride + h * ctx->head_dim;

    attn_core(xbh, qh, kv_h, pos, ctx->kcache, ctx->vcache,
              ctx->kv_type_k, ctx->kv_type_v,
              ctx->kv_row_size_k, ctx->kv_row_size_v,
              ctx->kv_head_stride_k, ctx->kv_head_stride_v,
              ctx->head_dim, ctx->attn_scale, ctx->n_swa);
}

/* GQA-grouped prefill attention task: processes all kv_mul Q heads for one
 * (token, kv_head) pair. This reduces KV cache reads by kv_mul compared to
 * dispatching per-Q-head tasks (prefill_attn_task), since the KV cache scan
 * is done once per KV head instead of once per Q head.
 *
 * Task index = bi * n_kv_heads + kv_h (token-major for L2 locality of Q rows). */
static void prefill_attn_task_grouped(int flat_idx, void *ctx_ptr) {
    prefill_attn_ctx_t *ctx = (prefill_attn_ctx_t *)ctx_ptr;
    int bi = flat_idx / ctx->n_kv_heads;
    int kv_h = flat_idx % ctx->n_kv_heads;
    int pos = ctx->start_pos + bi;
    int kv_mul = ctx->kv_mul;
    int head_dim = ctx->head_dim;
    const float *q_base = ctx->q_batch + (size_t)bi * ctx->n_heads * head_dim
                        + kv_h * kv_mul * head_dim;
    float *xb_base = ctx->xb_batch + (size_t)bi * ctx->xb_stride
                   + kv_h * kv_mul * head_dim;

    /* Per-Q-head online-softmax state */
    assert(kv_mul <= 8 && head_dim <= 256);
    float max_score[8], sum_exp[8];
    for (int g = 0; g < kv_mul; g++) { max_score[g] = -1e30f; sum_exp[g] = 0.0f; }
    float acc[8][256];
    memset(acc, 0, sizeof(acc));

    /* Sliding window */
    int t_start = (ctx->n_swa > 0) ? (pos - ctx->n_swa + 1) : 0;
    if (t_start < 0) t_start = 0;

    for (int t = t_start; t <= pos; t++) {
        const uint8_t *kt = ctx->kcache + (size_t)t * ctx->kv_row_size_k
                          + kv_h * ctx->kv_head_stride_k;
        const uint8_t *vt = ctx->vcache + (size_t)t * ctx->kv_row_size_v
                          + kv_h * ctx->kv_head_stride_v;

        /* Compute scores for all kv_mul Q heads against this K position */
        float score[8];
        for (int g = 0; g < kv_mul; g++) {
            const float *qh = q_base + g * head_dim;
            if (ctx->kv_type_k == KV_CACHE_Q8_0)
                score[g] = vec_dot_q8_0_f32(kt, qh, head_dim);
            else if (ctx->kv_type_k == KV_CACHE_Q4_0)
                score[g] = vec_dot_q4_0_f32(kt, qh, head_dim);
            else
                score[g] = vec_dot_f16_f32(kt, qh, head_dim);
            score[g] *= ctx->attn_scale;
        }

        /* Online softmax update for all kv_mul heads */
        for (int g = 0; g < kv_mul; g++) {
            float *accg = acc[g];
            const float *qh = q_base + g * head_dim;
            if (score[g] > max_score[g]) {
                float correction = expf(max_score[g] - score[g]);
                sum_exp[g] = sum_exp[g] * correction + 1.0f;
                /* Rescale acc and add V */
                if (ctx->kv_type_v == KV_CACHE_F16) {
                    const uint16_t *vt16 = (const uint16_t *)vt;
#ifdef PICOLM_AVX512
                    { __m512 cv = _mm512_set1_ps(correction); int d = 0;
                      for (; d + 15 < head_dim; d += 16) {
                          __m512 vf = fp16x16_to_fp32_inline(vt16 + d);
                          __m512 af = _mm512_loadu_ps(accg + d);
                          _mm512_storeu_ps(accg + d, _mm512_fmadd_ps(af, cv, vf));
                      }
                      for (; d < head_dim; d++) accg[d] = fmaf(accg[d], correction, fp16_to_fp32(vt16[d])); }
#else
                    for (int d = 0; d < head_dim; d++) accg[d] = fmaf(accg[d], correction, fp16_to_fp32(((const uint16_t *)vt)[d]));
#endif
                } else {
                    /* Quantized V: use scale_add with correction */
                    for (int d = 0; d < head_dim; d++) accg[d] *= correction;
                    /* Add V contribution */
                    if (ctx->kv_type_v == KV_CACHE_Q8_0) {
                        fma_scale_q8_0_f32(accg, 1.0f, vt, head_dim);
                    } else if (ctx->kv_type_v == KV_CACHE_Q4_0) {
                        fma_scale_q4_0_f32(accg, 1.0f, vt, head_dim);
                    } else {
                        for (int d = 0; d < head_dim; d++) accg[d] += fp16_to_fp32(((const uint16_t *)vt)[d]);
                    }
                }
                max_score[g] = score[g];
            } else {
                float w = expf(score[g] - max_score[g]);
                sum_exp[g] += w;
                if (ctx->kv_type_v == KV_CACHE_F16) {
                    const uint16_t *vt16 = (const uint16_t *)vt;
#ifdef PICOLM_AVX512
                    { __m512 wv = _mm512_set1_ps(w); int d = 0;
                      for (; d + 15 < head_dim; d += 16) {
                          __m512 vf = fp16x16_to_fp32_inline(vt16 + d);
                          __m512 af = _mm512_loadu_ps(accg + d);
                          _mm512_storeu_ps(accg + d, _mm512_fmadd_ps(vf, wv, af));
                      }
                      for (; d < head_dim; d++) accg[d] = fmaf(w, fp16_to_fp32(vt16[d]), accg[d]); }
#else
                    for (int d = 0; d < head_dim; d++) accg[d] = fmaf(w, fp16_to_fp32(((const uint16_t *)vt)[d]), accg[d]);
#endif
                } else {
                    if (ctx->kv_type_v == KV_CACHE_Q8_0) {
                        scale_add_q8_0_f32(accg, w, vt, head_dim);
                    } else if (ctx->kv_type_v == KV_CACHE_Q4_0) {
                        scale_add_q4_0_f32(accg, w, vt, head_dim);
                    } else {
                        for (int d = 0; d < head_dim; d++) accg[d] += w * fp16_to_fp32(((const uint16_t *)vt)[d]);
                    }
                }
            }
        }
    }

    /* Normalize and write output for all kv_mul Q heads */
    for (int g = 0; g < kv_mul; g++) {
        float inv_sum = 1.0f / sum_exp[g];
        float *accg = acc[g];
        float *out = xb_base + g * head_dim;
#ifdef PICOLM_AVX512
        { __m512 inv = _mm512_set1_ps(inv_sum); int d = 0;
          for (; d + 15 < head_dim; d += 16) { __m512 af = _mm512_loadu_ps(accg + d); _mm512_storeu_ps(out + d, _mm512_mul_ps(af, inv)); }
          for (; d < head_dim; d++) out[d] = accg[d] * inv_sum; }
#else
        for (int d = 0; d < head_dim; d++) out[d] = accg[d] * inv_sum;
#endif
    }
}

/* Tiled attention: tile size in KV positions */
#define ATTN_TILE 64

/* The tiled path keeps scores in the log2 domain (Q is pre-scaled by
 * attn_scale * log2(e)), so softmax weights are 2^(s - M): this removes the
 * x*log2(e) multiply and the ln2 range-reduction step from the exp kernel.
 * M, S and acc are internally consistent in that domain; acc/S is unchanged. */
#define ATTN_LOG2E 1.4426950408889634f

/* Lazy online-softmax rescale threshold (log2 units, 2^8 = 256x headroom).
 * 0 = only skip when the running max did not change. */
#ifndef ATTN_LAZY_RESCALE_THRESH
#define ATTN_LAZY_RESCALE_THRESH 8.0f
#endif
typedef struct {
    int kv_h;
    int kv_mul;
    int n_kv_heads;
    int head_dim;
    int tile_size;
    int n_q_rows;
    int group_token_start;  /* first token index in this group */
    int kv_tile_start;      /* first KV position in this tile */
    int kv_tile_end;        /* one past last KV position */
    int is_diagonal;        /* 1 if this tile needs causal masking */

    gguf_type_t kv_gguf_k;  /* gguf_type for this kv cache type */
    size_t kv_row_size_k;   /* bytes per GQA row in cache */
    size_t kv_head_stride_k;/* bytes per head within GQA row */
    const uint8_t *kcache;  /* layer K cache base */
    const float *q_rows;    /* [n_q_rows x head_dim] query vectors */
    float *scores;          /* [n_q_rows x tile_size] score buffer */
    uint8_t *tile_k;        /* contiguous K-tile scratch [tile_size x head_dim] in kv format */
    float *tile_v_f32;      /* contiguous V-tile in F32 [tile_size x head_dim] */
    float *M;               /* [n_q_rows] running max */
    float *S;               /* [n_q_rows] running sum_exp */
    float *acc;             /* [n_q_rows x head_dim] running accumulator */
    float *out;             /* [n_q_rows x head_dim] final output (written only after all tiles) */
    int last_tile;          /* 1 if this is the last tile to process */
    float attn_scale;       /* attention score scale factor */
    int n_swa;               /* 0 = full attention; >0 = sliding window size */
} attn_tile_task_t;

/* Forward declaration for batch_attention_layer gating */
static void attn_process_tile(attn_tile_task_t *t);

/* Context for the parallel tiled attention: one instance shared by all
 * concurrent tasks. Each task gets its own scratch slice from scratch_pool. */
typedef struct {
    float *xb_batch;
    const float *q_batch;
    const uint8_t *kcache;
    const uint8_t *vcache;
    int n_tokens, start_pos;
    int n_heads, n_kv_heads, kv_mul, head_dim;
    int xb_stride;
    int kv_type_k, kv_type_v;
    size_t kv_row_size_k, kv_row_size_v;
    size_t kv_head_stride_k, kv_head_stride_v;
    float attn_scale;
    int n_swa;
    int tile, n_token_groups, n_kv_tiles;
    gguf_type_t gguf_k;
    gguf_type_t gguf_v;
    size_t k_rb_gguf;
    /* Scratch pool: pre-allocated, sliced per concurrent task */
    uint8_t *scratch_pool;
    size_t scratch_per_task, scores_per, ms_per, acc_per, qr_per, tk_per, tv_per;
    int max_concurrent;
} batch_tiled_ctx_t;

/* Worker for tensor_parallel_for: task_idx maps to (kv_head, token_group).
 * kv_head = task_idx / n_token_groups, token_group = task_idx % n_token_groups.
 * This grouping ensures that tasks sharing the same KV head (and thus the
 * same K/V cache locality) run on the same thread when possible. */
static void batch_tiled_task(int task_idx, void *ctx_ptr) {
    batch_tiled_ctx_t *bc = (batch_tiled_ctx_t *)ctx_ptr;

    /* Get our scratch slice. tensor_parallel_for runs at most
     * bc->max_concurrent tasks concurrently, but to be safe we use
     * a slot per task index modulo max_concurrent. Since
     * tensor_parallel_for dispatches tasks in order and waits for
     * completion, tasks with non-overlapping indices never overlap
     * in time IF max_concurrent >= n_threads. We use task_idx
     * directly when n_tasks <= max_concurrent, otherwise modulo. */
    int slot = task_idx % bc->max_concurrent;
    uint8_t *scratch = bc->scratch_pool + (size_t)slot * bc->scratch_per_task;

    /* Layout the scratch slice */
    float *scores = (float *)scratch;            scratch += bc->scores_per;
    float *M       = (float *)scratch;           scratch += bc->ms_per;
    float *S       = (float *)scratch;           scratch += bc->ms_per;
    float *acc     = (float *)scratch;           scratch += bc->acc_per;
    float *q_rows  = (float *)scratch;           scratch += bc->qr_per;
    uint8_t *tile_k = scratch;                   scratch += bc->tk_per;
    float *tile_v_f32 = (float *)scratch;

    int kv_h = task_idx / bc->n_token_groups;
    int tg = task_idx % bc->n_token_groups;
    int kv_mul = bc->kv_mul;
    int head_dim = bc->head_dim;
    int tile = bc->tile;
    int start_pos = bc->start_pos;
    int n_tokens = bc->n_tokens;
    int n_kv_tiles = bc->n_kv_tiles;

    int q_group_start = tg * tile;
    int q_group_end = q_group_start + tile;
    if (q_group_end > n_tokens) q_group_end = n_tokens;
    int n_q = q_group_end - q_group_start;
    int n_q_padded = n_q * kv_mul;

    /* Gather query rows for this (kv_head, token_group), pre-scaling by
     * attn_scale. This fuses the scale into the QK^T GEMM inputs,
     * eliminating the per-tile score-scaling loop (n_q*ts multiplies per
     * tile visit, ~884K total for a 3037-token prefill). The scale is
     * applied once per Q row (n_q*hd multiplies) instead of once per
     * score element (n_q*ts multiplies per tile). */
    const float q_scale_log2 = bc->attn_scale * ATTN_LOG2E;  /* scores come out in log2 units */
    for (int ti = 0; ti < n_q; ti++) {
        const float *q_tok = bc->q_batch + (size_t)(q_group_start + ti) * bc->n_heads * head_dim;
        for (int g = 0; g < kv_mul; g++) {
            const float *qh = q_tok + (kv_h * kv_mul + g) * head_dim;
            float *qr = q_rows + ((size_t)ti * kv_mul + g) * head_dim;
            for (int d = 0; d < head_dim; d++)
                qr[d] = qh[d] * q_scale_log2;
        }
    }

    /* Initialize M, S, acc */
    for (int i = 0; i < n_q_padded; i++) {
        M[i] = -1e30f;
        S[i] = 0.0f;
    }
    memset(acc, 0, (size_t)n_q_padded * head_dim * sizeof(float));

    /* Tile loop over KV positions */
    for (int tk = 0; tk < n_kv_tiles; tk++) {
        int kv_t0 = tk * tile;
        int kv_t1 = kv_t0 + tile;
        if (kv_t1 > start_pos + n_tokens) kv_t1 = start_pos + n_tokens;
        int first_pos = start_pos + q_group_start;
        /* Stop once the tile starts past the last row of this group.
         * (Do NOT also skip tiles past first_pos: when start_pos is not a
         * multiple of the tile size, the group's rows straddle KV tiles.) */
        if (kv_t0 >= start_pos + q_group_end) break;

        /* Skip tiles entirely BEFORE every row's SWA window */
        if (bc->n_swa > 0) {
            int win_start_first = first_pos - bc->n_swa + 1;
            if (win_start_first < 0) win_start_first = 0;
            if (kv_t1 <= win_start_first) continue;
        }

        int this_tile_size = kv_t1 - kv_t0;
        if (this_tile_size <= 0) continue;

        /* Causal masking is needed whenever the tile has a column beyond the
         * FIRST row's position. Equals the old diagonal test when start_pos is
         * tile-aligned, but also covers unaligned starts. */
        int is_diag = (kv_t1 - 1 > first_pos);

        /* Extract K-tile into contiguous scratch */
        {
            size_t rb = bc->kv_head_stride_k;
            size_t row_stride = bc->kv_row_size_k;
            for (int p = 0; p < this_tile_size; p++) {
                const uint8_t *src = bc->kcache + (size_t)(kv_t0 + p) * row_stride
                                   + kv_h * rb;
                uint8_t *dst = tile_k + (size_t)p * bc->k_rb_gguf;
                memcpy(dst, src, bc->k_rb_gguf);
            }
        }

        /* Extract V-tile and dequantize to F32 */
        {
            size_t rb = bc->kv_row_size_v;
            size_t v_head_stride = bc->kv_head_stride_v;
            for (int p = 0; p < this_tile_size; p++) {
                const uint8_t *src = bc->vcache + (size_t)(kv_t0 + p) * rb
                                   + kv_h * v_head_stride;
                dequantize_row(src, tile_v_f32 + (size_t)p * head_dim,
                              head_dim, bc->gguf_v);
            }
        }

        /* Build task context and process */
        attn_tile_task_t task;
        memset(&task, 0, sizeof(task));
        task.kv_h = kv_h;
        task.kv_mul = kv_mul;
        task.n_kv_heads = bc->n_kv_heads;
        task.head_dim = head_dim;
        task.tile_size = this_tile_size;
        task.n_q_rows = n_q_padded;
        task.group_token_start = first_pos;
        task.kv_tile_start = kv_t0;
        task.kv_tile_end = kv_t1;
        task.is_diagonal = is_diag;
        task.kv_gguf_k = bc->gguf_k;
        task.kv_row_size_k = bc->kv_row_size_k;
        task.kv_head_stride_k = bc->kv_head_stride_k;
        task.kcache = bc->kcache;
        task.q_rows = q_rows;
        task.scores = scores;
        task.tile_k = tile_k;
        task.tile_v_f32 = tile_v_f32;
        task.M = M;
        task.S = S;
        task.acc = acc;
        task.attn_scale = bc->attn_scale;
        task.n_swa = bc->n_swa;

        attn_process_tile(&task);
    }

    /* Normalize and write output */
    {
        static int _dbg_task = 0;
        const char *_dbg = getenv("PICOLM_ATTN_TASK_DBG");
        int _do_dbg = _dbg && _dbg_task < atoi(_dbg);
        if (_do_dbg) _dbg_task++;
        for (int ti = 0; ti < n_q; ti++) {
            for (int g = 0; g < kv_mul; g++) {
                int ri = ti * kv_mul + g;
                float inv_sum = 1.0f / S[ri];
                float *acc_row = acc + ri * head_dim;
                float *out = bc->xb_batch + (size_t)(q_group_start + ti) * bc->xb_stride
                           + (kv_h * kv_mul + g) * head_dim;
                for (int d = 0; d < head_dim; d++)
                    out[d] = acc_row[d] * inv_sum;
                if (_do_dbg && kv_h == 0 && ti < 3) {
                    fprintf(stderr, "[TASK kv_h=0 tg=%d ti=%d ri=%d] M=%g S=%g inv=%g out.sum=%g acc0=%g\n",
                            tg, ti, ri, M[ri], S[ri], inv_sum, out[0] + out[1] + out[2] + out[3], acc_row[0]);
                }
            }
        }
    }
}

static void batch_attention_tiled(
        float *xb_batch, const float *q_batch,
        const uint8_t *kcache, const uint8_t *vcache,
        int n_tokens, int start_pos,
        int n_heads, int n_kv_heads, int head_dim,
        int xb_stride,
        kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v,
        size_t kv_row_size_k, size_t kv_row_size_v,
        size_t kv_head_stride_k, size_t kv_head_stride_v,
        float attn_scale, int n_swa);

/* Forward declaration for callback to tensor_parallel_for */
static void prefill_attn_task(int flat_idx, void *ctx_ptr);

#include <time.h>

/* ---- Attention timing instrumentation (PICOLM_ATTN_TIMING=1) ---- */
static double attn_timing_total = 0.0;
static long long attn_timing_calls = 0;
static long long attn_timing_tokens = 0;
static long long attn_timing_ctx_sum = 0;   /* sum of start_pos*n_tokens */
static int attn_timing_enabled = -1;

static double attn_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void attn_timing_report(void) {
    if (attn_timing_calls == 0) return;
    fprintf(stderr, "[ATTN-TIMING] calls=%lld tokens=%lld avg_ctx=%lld total_ms=%.1f ms_per_token_ctx=%.6f\n",
            attn_timing_calls, attn_timing_tokens,
            attn_timing_tokens ? attn_timing_ctx_sum / attn_timing_tokens : 0,
            attn_timing_total,
            attn_timing_tokens ? attn_timing_total / attn_timing_tokens : 0);
}

static void attn_timing_register(void) __attribute__((constructor));
static void attn_timing_register(void) {
    atexit(attn_timing_report);
}

void batch_attention_layer(
        float *xb_batch, const float *q_batch,
        const uint8_t *kcache, const uint8_t *vcache,
        int n_tokens, int start_pos,
        int n_heads, int n_kv_heads, int head_dim,
        int xb_stride,
        int kv_type_k, int kv_type_v,
        size_t kv_row_size_k, size_t kv_row_size_v,
        size_t kv_head_stride_k, size_t kv_head_stride_v,
        float attn_scale, int n_swa)
{
    double _attn_t0 = 0;
    if (attn_timing_enabled < 0)
        attn_timing_enabled = getenv("PICOLM_ATTN_TIMING") ? 1 : 0;
    if (attn_timing_enabled) _attn_t0 = attn_now_ms();

    /* Build the prefill_attn_ctx for both the original path and the test */
    prefill_attn_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.n_heads = n_heads; ctx.n_kv_heads = n_kv_heads; ctx.kv_mul = n_heads / n_kv_heads;
    ctx.head_dim = head_dim; ctx.start_pos = start_pos;
    ctx.kv_type_k = kv_type_k; ctx.kv_type_v = kv_type_v;
    ctx.kv_row_size_k = kv_row_size_k; ctx.kv_row_size_v = kv_row_size_v;
    ctx.kv_head_stride_k = kv_head_stride_k; ctx.kv_head_stride_v = kv_head_stride_v;
    ctx.kcache = kcache; ctx.vcache = vcache;
    ctx.q_batch = q_batch; ctx.xb_batch = xb_batch; ctx.xb_stride = xb_stride;
    ctx.attn_scale = attn_scale;
    ctx.n_swa = n_swa;

    /* For large enough batches, use the tiled/batched attention path which
     * amortizes KV cache load/dequant across multiple query tokens.
     * For small batches, the per-(token,head) path has better parallelism
     * (n_tokens * n_heads tasks vs n_kv_heads * ceil(n_tokens/tile) tasks).
     *
     * The tiled path requires F16 KV cache (which is what the store loop
     * always writes). Q8_0/Q4_0 cache types are a planned enhancement.
     *
     * Verified against the grouped path by test_attn_diff (see
     * test_attn_diff_sweep.sh), including unaligned start_pos. */
    static int _layer_call = 0;
    int _layer_idx = _layer_call++;
    int _use_tiled = n_tokens >= 2 * ATTN_TILE &&
                     (int)kv_type_k == (int)KV_CACHE_F16 &&
                     (int)kv_type_v == (int)KV_CACHE_F16;
    {
        /* TEST HOOK: PICOLM_TEST_FORCE_TILED=0 -> grouped path,
         *            =1 -> tiled path regardless of size gate
         * PICOLM_ATTN_TILED_LAYERS=bitmask -> per-layer control (bit l) */
        const char *_ft = getenv("PICOLM_TEST_FORCE_TILED");
        if (_ft) _use_tiled = atoi(_ft);
        const char *_lm = getenv("PICOLM_ATTN_TILED_LAYERS");
        if (_lm) {
            int mask = (int)strtoul(_lm, NULL, 0);
            _use_tiled = (mask >> _layer_idx) & 1;
        }
    }
    {
        static int _dbg_once = 0;
        if (!_dbg_once) {
            _dbg_once = 1;
            fprintf(stderr, "[BATCH-ATTN] n_tokens=%d start_pos=%d n_heads=%d n_kv_heads=%d "
                    "head_dim=%d xb_stride=%d kv_type=%d/%d kv_row=%zu/%zu kv_hstride=%zu/%zu "
                    "scale=%g n_swa=%d tiled=%d\n",
                    n_tokens, start_pos, n_heads, n_kv_heads, head_dim, xb_stride,
                    kv_type_k, kv_type_v, kv_row_size_k, kv_row_size_v,
                    kv_head_stride_k, kv_head_stride_v, attn_scale, n_swa, _use_tiled);
        }
    }
    if (_use_tiled) {
        batch_attention_tiled(xb_batch, q_batch, kcache, vcache,
                              n_tokens, start_pos,
                              n_heads, n_kv_heads, head_dim,
                              xb_stride,
                              (kv_cache_type_t)kv_type_k, (kv_cache_type_t)kv_type_v,
                              kv_row_size_k, kv_row_size_v,
                              kv_head_stride_k, kv_head_stride_v,
                              attn_scale, n_swa);
    } else {
        /* One dispatch per layer for the whole batch: n_tokens * n_kv_heads
         * independent (token, kv_head) tasks with GQA grouping. Each task
         * scans the causal range once for all kv_mul Q heads sharing this
         * KV head, reducing KV cache reads by kv_mul vs per-Q-head dispatch.
         * Still provides n_tokens * n_kv_heads tasks for thread parallelism
         * (e.g., 13 tokens * 8 kv_heads = 104 tasks on 16 threads). */
        tensor_parallel_for(n_tokens * n_kv_heads, prefill_attn_task_grouped, &ctx);
    }

    if (attn_timing_enabled) {
        double _attn_t1 = attn_now_ms();
        attn_timing_total += _attn_t1 - _attn_t0;
        attn_timing_calls++;
        attn_timing_tokens += n_tokens;
        attn_timing_ctx_sum += (long long)start_pos * n_tokens;
    }
    {
        static int _dbg_call = 0;
        const char *_dump = getenv("PICOLM_ATTN_DUMP_CHECKSUM");
        if (_dump && _dbg_call < atoi(_dump)) {
            _dbg_call++;
            double qsum_in = 0, ksum_in = 0, vsum_in = 0;
            for (int i = 0; i < n_tokens * n_heads * head_dim; i++) qsum_in += q_batch[i];
            for (int p = 0; p < n_tokens + start_pos; p++) {
                const uint16_t *kp = (const uint16_t *)(kcache + (size_t)p * kv_row_size_k);
                const uint16_t *vp = (const uint16_t *)(vcache + (size_t)p * kv_row_size_v);
                for (int d = 0; d < 8; d++) {
                    ksum_in += fp16_to_fp32_lookup(kp[d]);
                    vsum_in += fp16_to_fp32_lookup(vp[d]);
                }
            }
            fprintf(stderr, "[IN %d] qsum=%.4f ksum=%.4f vsum=%.4f\n", _dbg_call, qsum_in, ksum_in, vsum_in);
            {
                const char *df = getenv("PICOLM_ATTN_DUMP_IN");
                if (df && _dbg_call == 2) {
                    FILE *f = fopen(df, "wb");
                    if (f) {
                        size_t w1 = fwrite(q_batch, sizeof(float), (size_t)n_tokens * n_heads * head_dim, f);
                        size_t w2 = fwrite(kcache, 1, (size_t)(n_tokens + start_pos) * kv_row_size_k, f);
                        size_t w3 = fwrite(vcache, 1, (size_t)(n_tokens + start_pos) * kv_row_size_v, f);
                        fprintf(stderr, "[DUMP] w1=%zu w2=%zu w3=%zu (expect %d, %zu, %zu)\n",
                                w1, w2, w3, (size_t)n_tokens * n_heads * head_dim,
                                (size_t)(n_tokens + start_pos) * kv_row_size_k,
                                (size_t)(n_tokens + start_pos) * kv_row_size_v);
                        fclose(f);
                    }
                }
            }
            double sum = 0, sum2 = 0; int nan_cnt = 0;
            double qsum = 0, ksum = 0, vsum = 0;
            for (int i = 0; i < n_tokens * n_heads * head_dim; i++) {
                float v = xb_batch[i];
                if (isnan(v)) nan_cnt++;
                sum += v; sum2 += (double)v * v;
            }
            /* checksum Q input and first 64 K/V rows of kv head 0 */
            for (int i = 0; i < n_tokens * n_heads * head_dim; i++) {
                float v = q_batch[i];
                qsum += v; 
            }
            for (int p = 0; p < n_tokens + start_pos; p++) {
                const uint16_t *kp = (const uint16_t *)(kcache + (size_t)p * kv_row_size_k);
                const uint16_t *vp = (const uint16_t *)(vcache + (size_t)p * kv_row_size_v);
                for (int d = 0; d < head_dim; d++) {
                    ksum += fp16_to_fp32_lookup(kp[d]);
                    vsum += fp16_to_fp32_lookup(vp[d]);
                }
            }
            {
                int ntok_dbg = n_tokens < 129 ? n_tokens : 129;
                fprintf(stderr, "[ATTN-CALL %d] nt=%d sp=%d out.sum=%.4f out.sum2=%.4f nan=%d q.sum=%.4f k0.sum=%.4f v0.sum=%.4f tiled=%d per_tok=[",
                    _dbg_call, n_tokens, start_pos, sum, sum2, nan_cnt, qsum, ksum, vsum, _use_tiled);
                for (int t = 0; t < ntok_dbg; t++) {
                    double ts_ = 0;
                    for (int h = 0; h < n_heads * head_dim; h++) ts_ += xb_batch[(size_t)t * xb_stride + h];
                    fprintf(stderr, "%s%.6f", t ? "," : "", ts_);
                }
                fprintf(stderr, "]\n");
                const char *dumpf = getenv("PICOLM_ATTN_DUMP_FILE");
                if (dumpf && (_dbg_call == 1 || _dbg_call == 2)) {
                    char p2[512];
                    snprintf(p2, sizeof(p2), "%s.call%d", dumpf, _dbg_call);
                    FILE *df = fopen(p2, "wb");
                    if (df) { fwrite(xb_batch, sizeof(float), (size_t)n_tokens * xb_stride, df); fclose(df); }
                }
            }
        }
    }
}

/* ================================================================
 * Tiled/blocked attention for prefill.
 *
 * Reframes attention as a sequence of small matmul_batch calls,
 * tiling over KV positions. K/V tiles are extracted from the
 * interleaved KV cache into contiguous scratch buffers, then
 * processed through the existing weight-stationary batched matmul
 * infrastructure (same Q8_0 activation-quantization fast paths).
 *
 * Key insight: for GQA models, all kv_mul query heads sharing a KV
 * head see the same K/V tile. So the activation batch for the QK^T
 * matmul is kv_mul * GROUP_SIZE rows, amortizing the K/V tile load
 * across all grouped heads.
 *
 * Online-softmax merge across tiles: standard flash-attention
 * recurrence with per-query running (M, S, acc) state.
 *
 * Causal masking via block-diagonal scheme: GROUP_SIZE == TILE.
 * Tokens are partitioned into groups aligned with KV tile boundaries.
 * For each group, KV tiles before the group's own block are fully
 * visible (dense matmul), the diagonal tile needs row-wise masking,
 * and tiles after are skipped.
 * ================================================================ */

/* Map kv_cache_type_t to gguf_type_t for matmul_batch */
static gguf_type_t kv_cache_to_gguf_type(kv_cache_type_t kv_type) {
    switch (kv_type) {
        case KV_CACHE_F16:  return GGUF_TYPE_F16;
        case KV_CACHE_Q8_0: return GGUF_TYPE_Q8_0;
        case KV_CACHE_Q4_0: return GGUF_TYPE_Q4_0;
        case KV_CACHE_TQ3:  return GGUF_TYPE_F16; /* TQ3 doesn't have a GGUF equivalent */
        case KV_CACHE_TQ4:  return GGUF_TYPE_F16; /* TQ4 doesn't have a GGUF equivalent */
    }
    return GGUF_TYPE_F16;
}

/* Tiled attention task: process one (kv_head, token_group) pair.
 * tile_k: contiguous K-tile [tile_size x head_dim] in kv_type_k format
 * tile_v_f32: contiguous V-tile [tile_size x head_dim] in F32
 * scores: scores for this tile [n_q_rows x tile_size] in F32
 * q_rows: query rows [n_q_rows x head_dim] in F32 (kv_mul * GROUP_SIZE)
 * n_q_rows: number of query rows in this group (kv_mul * tokens_in_group)
 * tile_size: actual KV positions in this tile (may be < ATTN_TILE for tail)
 * M, S, acc: running softmax state to update in-place
 * out: final output to write [n_q_rows x head_dim] (only after last tile) */

/* Specialized attention QK^T kernel for F16 KV cache tiles.
 * Computes scores[n_q][ts] = q_rows[n_q][hd] @ K[ts][hd]^T * scale.
 *
 * Unlike the generic sgemm_f16_f32, this kernel:
 *   - Converts K-tile from F16 to F32 ONCE (not per Q-row block)
 *   - Avoids the 1672-byte stack frame of the generic GEMM
 *   - Uses a simple 1 Q-row x 16 K-cols inner loop optimized for
 *     the small tile sizes used in attention (ts <= 64, n_q <= 256)
 *
 * Called once per (kv_head, token_group, kv_tile) visit.
 * Only compiled for AVX-512 (the generic path is used elsewhere). */
/* Process one tile within a (kv_head, token_group) task.
 * Called inline from the task loop. */
static void attn_process_tile(attn_tile_task_t *t) {
    int n_q = t->n_q_rows;
    int ts = t->tile_size;
    int hd = t->head_dim;
    int is_diag = t->is_diagonal;
    int kv_tile_start = t->kv_tile_start;
    int group_token_start = t->group_token_start;

    size_t k_rb_gguf = gguf_type_row_size(t->kv_gguf_k, hd);

    /* Extract K-tile from GQA KV cache into contiguous scratch.
     * KV cache layout: [pos][kv_row_size_gqa] with head offset = kv_h * kv_head_stride_k
     * For positions [kv_tile_start, kv_tile_start+ts), head kv_h: */
    {
        size_t rb = t->kv_head_stride_k;
        size_t row_stride = t->kv_row_size_k;
        for (int p = 0; p < ts; p++) {
            const uint8_t *src = t->kcache + (size_t)(kv_tile_start + p) * row_stride
                               + t->kv_h * rb;
            uint8_t *dst = (uint8_t *)t->tile_k + (size_t)p * k_rb_gguf;
            memcpy(dst, src, rb);
        }
    }

    /* QK^T: matmul_batch(scores, q_rows, n_q, tile_k, hd, ts, kv_gguf_k)
     * out layout: [n_q][ts], scores[b*ts + i] = row b, col i
     *
     * NOTE: This is called from inside tensor_parallel_for workers.
     * matmul_batch for F16/F32 dispatches to picolm_sgemm with nth=1
     * (serial), which does NOT use the thread pool. However, for
     * quantized KV types, matmul_batch may fall through to the threaded
     * worker path, which WOULD deadlock. Since the tiled path currently
     * only activates for F16 KV cache, this is safe in practice. If
     * quantized KV types are added, replace with a serial dot-product
     * loop or a non-pool dispatch mechanism. */
    matmul_batch(t->scores, t->q_rows, n_q, t->tile_k, hd, ts, t->kv_gguf_k);

    /* NOTE: attn_scale * log2(e) is pre-applied to q_rows during gather, so
     * scores are in the log2 domain and no per-tile scaling is needed here. */

    /* Causal masking for diagonal tile: for each query row i,
     * only positions [0, i_within_group] are valid.
     * Within the diagonal tile, query row i (0..n_q-1) corresponds to
     * absolute position (group_token_start + i/kv_mul) -- group_token_start
     * is set by the caller as start_pos + q_group_start, i.e. already
     * absolute (see batch_attention_tiled()).
     * The diagonal tile starts at kv_tile_start. The query's causal limit
     * is pos = group_token_start + i/kv_mul.
     * Within this tile, valid columns are [0, pos - kv_tile_start]. */

    if (is_diag) {
        for (int i = 0; i < n_q; i++) {
            int token_idx = i / t->kv_mul;
            int pos = group_token_start + token_idx;
            int valid_cols = pos - kv_tile_start + 1;
            if (valid_cols < 0) valid_cols = 0;
            if (valid_cols > ts) valid_cols = ts;
            float *row = t->scores + i * ts;
            for (int j = valid_cols; j < ts; j++)
                row[j] = -1e30f;
        }
    }

    /* Sliding-window masking: applies to EVERY tile, not just the
     * diagonal one -- a tile fully in the causal past can still contain
     * KV positions older than a row's SWA window and must be masked
     * there too. For row i (absolute position `pos`), valid columns are
     * those with absolute KV position >= max(0, pos - n_swa + 1); columns
     * before that (relative to this tile's start) are masked to -inf,
     * exactly like the causal mask above, just on the other side. */
    if (t->n_swa > 0) {
        for (int i = 0; i < n_q; i++) {
            int token_idx = i / t->kv_mul;
            int pos = group_token_start + token_idx;
            int win_start = pos - t->n_swa + 1;
            if (win_start < 0) win_start = 0;
            int mask_upto = win_start - kv_tile_start; /* columns [0, mask_upto) predate the window */
            if (mask_upto > 0) {
                if (mask_upto > ts) mask_upto = ts;
                float *row = t->scores + i * ts;
                for (int j = 0; j < mask_upto; j++)
                    row[j] = -1e30f;
            }
        }
    }

    /* Online softmax merge:
     * For each query row i:
     *   tile_max[i] = max(scores[i*ts .. (i+1)*ts - 1])
     *   new_M[i] = max(old_M[i], tile_max[i])
     *   corr[i] = exp(old_M[i] - new_M[i])
     *   tile_exp[i*j] = exp(scores[i*j] - new_M[i])
     *   tile_sum[i] = sum(tile_exp[i*0..ts-1])
     *   old_S[i] *= corr[i]
     *   old_acc[i*] *= corr[i]
     *   S[i] = old_S[i] + tile_sum[i]
     *   acc[i*] += tile_exp[i*] @ V_tile (row-major: tile_sum_i = sum_j tile_exp[i*j] * V[j*])
     *
     * For the acc update: tile_exp[n_q x ts] @ V_tile[ts x hd]
     * = matmul_batch(acc_add, tile_exp, n_q, V_tile_T, ts, hd, F32)
     * But V_tile_T would need to be quantized... Instead do it manually.
     *
     * Actually: we can do acc_add = tile_exp @ V_tile as a matmul_batch
     * where V_tile is in F32 format stored [ts][hd].
     * matmul_batch wants weight [d rows][n cols] = [hd rows][ts cols]
     * which is the TRANSPOSE of V_tile. So we need V_t[hd x ts].
     */

    /* Per-row tile_exp buffer (tile_size wide, tile_size <= ATTN_TILE = 64) */
    float tile_exp_buf[ATTN_TILE];

    /* Process each query row independently: compute max, exp, sum, and acc update */
    for (int i = 0; i < n_q; i++) {
        float *srow = t->scores + i * ts;

        /* Row max (vectorized with scalar tail) */
#if defined(PICOLM_AVX512)
        __m512 vmax = _mm512_set1_ps(srow[0]);
        int j = 1;
        for (; j + 15 < ts; j += 16)
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(srow + j));
        float rmax = _mm512_reduce_max_ps(vmax);
        for (; j < ts; j++) if (srow[j] > rmax) rmax = srow[j];
#else
        float rmax = srow[0];
        for (int j = 1; j < ts; j++) {
            if (srow[j] > rmax) rmax = srow[j];
        }
#endif

        /* Update running M. Rescale S/acc only when the max grows by more than
         * ATTN_LAZY_RESCALE_THRESH (log2 units). Otherwise keep the
         * stale M: the 2^x arguments may then be slightly positive (<= THRESH,
         * far from float32 overflow) but acc/S stay mutually consistent, so
         * the final acc/S is mathematically unchanged. THRESH = 0 gives the
         * exact "skip when the max did not move" behaviour. */
        float old_M = t->M[i];
        float new_M = old_M;
        if (rmax > old_M + ATTN_LAZY_RESCALE_THRESH) {
            new_M = rmax;
            t->M[i] = new_M;
            float corr = exp2f(old_M - new_M);

            /* Scale old S and acc by correction */
            t->S[i] *= corr;
            float *acc_row = t->acc + i * hd;
#if defined(PICOLM_AVX512)
            {
                __m512 vc = _mm512_set1_ps(corr);
                for (int d = 0; d + 15 < hd; d += 16)
                    _mm512_storeu_ps(acc_row + d, _mm512_mul_ps(_mm512_loadu_ps(acc_row + d), vc));
                for (int d = (hd / 16) * 16; d < hd; d++)
                    acc_row[d] *= corr;
            }
#else
            for (int d = 0; d < hd; d++)
                acc_row[d] *= corr;
#endif
        }
        float *acc_row = t->acc + i * hd;

        /* Compute 2^(s[j] - new_M) for this row and accumulate. Scores are in
         * the log2 domain (see ATTN_LOG2E). With the lazy rescale above the
         * argument can be slightly positive (<= ATTN_LAZY_RESCALE_THRESH). */
        float rsum = 0.0f;
#if defined(PICOLM_AVX512)
        {
            __m512 vm = _mm512_set1_ps(new_M);
            __m512 vsum = _mm512_setzero_ps();
            const __m512 vlo = _mm512_set1_ps(-125.0f);
            int j = 0;
            for (; j + 15 < ts; j += 16) {
                __m512 x = _mm512_sub_ps(_mm512_loadu_ps(srow + j), vm);
                /* Lanes below -125 (incl. masked -1e30) are forced to exactly 0.
                 * The clamp is REQUIRED, not cosmetic: vscalefps on a denormal
                 * result triggers a microcode assist (~29x slower measured), and
                 * 2^-125 * p(r) >= 2^-125.5 stays a normal float. */
                __mmask16 valid = _mm512_cmp_ps_mask(x, vlo, _CMP_GE_OQ);
                x = _mm512_max_ps(x, vlo);
                /* 2^x = 2^n * 2^r, n = round(x), r in [-0.5, 0.5] */
                __m512 rn = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                __m512 r  = _mm512_sub_ps(x, rn);
                /* Degree-6 minimax for 2^r on [-0.5, 0.5] with c0 == 1 exactly
                 * (so the row max, x == 0, gets weight exactly 1.0).
                 * Max rel error 7.9e-8 (1.3 ulp) with FMA Horner evaluation. */
                __m512 p = _mm512_set1_ps(1.535336196e-04f);
                p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.339887502e-03f));
                p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(9.618436918e-03f));
                p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.550332367e-02f));
                p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(2.402264774e-01f));
                p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(6.931471825e-01f));
                p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
                /* 2^rn via bit-trick, then multiply by polynomial.
                 * scalef_ps (vscalefps) is fast on SKL+ but slow on Cascade Lake
                 * (microcode assist). The bit-trick is fast everywhere. */
                __m512 e2 = _mm512_castsi512_ps(
                    _mm512_cvtps_epi32(_mm512_add_ps(
                        _mm512_mul_ps(rn, _mm512_set1_ps(8388608.0f)),
                        _mm512_set1_ps(1065353216.0f))));
                __m512 result = _mm512_maskz_mov_ps(valid, _mm512_mul_ps(p, e2));
                _mm512_storeu_ps(tile_exp_buf + j, result);
                vsum = _mm512_add_ps(vsum, result);
            }
            rsum = _mm512_reduce_add_ps(vsum);
            for (; j < ts; j++) {
                tile_exp_buf[j] = exp2f(srow[j] - new_M);
                rsum += tile_exp_buf[j];
            }
        }
#else
        for (int j = 0; j < ts; j++) {
            tile_exp_buf[j] = exp2f(srow[j] - new_M);
            rsum += tile_exp_buf[j];
        }
#endif
        t->S[i] += rsum;

        /* acc_row += sum_j tile_exp[j] * V[j, d]
         * Register-blocked: hoist accumulator into zmm registers across the
         * full j loop to eliminate redundant load/store of acc_row per j.
         * For AVX-512: hd=128 floats = 8 zmm accumulators, all kept in regs.
         * For each d-block: 1 load + ts FMAs + 1 store (vs ts loads + ts stores). */
        float *v_tile = t->tile_v_f32;
#if defined(PICOLM_AVX512)
        {
            /* Register-blocked: keep ALL hd/D_BLK accumulators in zmm regs
             * across the full j loop. This gives hd/16 independent FMA
             * chains (8 for hd=128), eliminating the 4-cycle loop-carried
             * FMA latency bottleneck. The CPU pipelines 8 FMAs per j
             * iteration at 2/cycle throughput instead of waiting 4 cycles
             * per sequential FMA. */
            /* Fully unrolled: 8 named accumulators stay in zmm registers
             * (no stack spills). hd=128 -> 8 blocks of 16 floats. */
            const float *vr = v_tile;
            if (hd == 128) {
                __m512 a0 = _mm512_loadu_ps(acc_row + 0);
                __m512 a1 = _mm512_loadu_ps(acc_row + 16);
                __m512 a2 = _mm512_loadu_ps(acc_row + 32);
                __m512 a3 = _mm512_loadu_ps(acc_row + 48);
                __m512 a4 = _mm512_loadu_ps(acc_row + 64);
                __m512 a5 = _mm512_loadu_ps(acc_row + 80);
                __m512 a6 = _mm512_loadu_ps(acc_row + 96);
                __m512 a7 = _mm512_loadu_ps(acc_row + 112);
                for (int j = 0; j < ts; j++) {
                    __m512 wv = _mm512_set1_ps(tile_exp_buf[j]);
                    const float *vj = v_tile + (size_t)j * hd;
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 0), wv, a0);
                    a1 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 16), wv, a1);
                    a2 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 32), wv, a2);
                    a3 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 48), wv, a3);
                    a4 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 64), wv, a4);
                    a5 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 80), wv, a5);
                    a6 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 96), wv, a6);
                    a7 = _mm512_fmadd_ps(_mm512_loadu_ps(vj + 112), wv, a7);
                }
                _mm512_storeu_ps(acc_row + 0, a0);
                _mm512_storeu_ps(acc_row + 16, a1);
                _mm512_storeu_ps(acc_row + 32, a2);
                _mm512_storeu_ps(acc_row + 48, a3);
                _mm512_storeu_ps(acc_row + 64, a4);
                _mm512_storeu_ps(acc_row + 80, a5);
                _mm512_storeu_ps(acc_row + 96, a6);
                _mm512_storeu_ps(acc_row + 112, a7);
            } else {
                /* Generic path for hd != 128 (e.g. hd=64, 96) */
                const int D_BLK = 16;
                int n_blks = hd / D_BLK;
                if (n_blks > 8) n_blks = 8;
                __m512 accv[8];
                for (int db = 0; db < n_blks; db++)
                    accv[db] = _mm512_loadu_ps(acc_row + db * D_BLK);
                for (int j = 0; j < ts; j++) {
                    __m512 wv = _mm512_set1_ps(tile_exp_buf[j]);
                    const float *vj = v_tile + (size_t)j * hd;
                    for (int db = 0; db < n_blks; db++)
                        accv[db] = _mm512_fmadd_ps(_mm512_loadu_ps(vj + db * D_BLK), wv, accv[db]);
                }
                for (int db = 0; db < n_blks; db++)
                    _mm512_storeu_ps(acc_row + db * D_BLK, accv[db]);
            }
            /* Scalar tail for hd % 16 != 0 */
            {
                int d = (hd / 16) * 16;
                for (; d < hd; d++) {
                    float a = acc_row[d];
                    for (int j = 0; j < ts; j++)
                        a += tile_exp_buf[j] * v_tile[(size_t)j * hd + d];
                    acc_row[d] = a;
                }
            }
        }
#elif defined(PICOLM_AVX)
        {
            const int D_BLK = 8; /* floats per ymm */
            int n_blks = hd / D_BLK;
            for (int db = 0; db < n_blks; db++) {
                int d = db * D_BLK;
                __m256 acc0 = _mm256_loadu_ps(acc_row + d);
                const float *vr = v_tile + d;
                for (int j = 0; j < ts; j++) {
                    __m256 wv = _mm256_set1_ps(tile_exp_buf[j]);
                    __m256 vv = _mm256_loadu_ps(vr + (size_t)j * hd);
                    acc0 = _mm256_fmadd_ps(vv, wv, acc0);
                }
                _mm256_storeu_ps(acc_row + d, acc0);
            }
            for (int d = n_blks * D_BLK; d < hd; d++) {
                float a = acc_row[d];
                for (int j = 0; j < ts; j++)
                    a += tile_exp_buf[j] * v_tile[(size_t)j * hd + d];
                acc_row[d] = a;
            }
        }
#elif defined(PICOLM_NEON_AARCH64)
        {
            const int D_BLK = 4;
            int n_blks = hd / D_BLK;
            for (int db = 0; db < n_blks; db++) {
                int d = db * D_BLK;
                float32x4_t acc0 = vld1q_f32(acc_row + d);
                const float *vr = v_tile + d;
                for (int j = 0; j < ts; j++) {
                    float32x4_t wv = vdupq_n_f32(tile_exp_buf[j]);
                    float32x4_t vv = vld1q_f32(vr + (size_t)j * hd);
                    acc0 = vfmaq_f32(acc0, vv, wv);
                }
                vst1q_f32(acc_row + d, acc0);
            }
            for (int d = n_blks * D_BLK; d < hd; d++) {
                float a = acc_row[d];
                for (int j = 0; j < ts; j++)
                    a += tile_exp_buf[j] * v_tile[(size_t)j * hd + d];
                acc_row[d] = a;
            }
        }
#else
        for (int d = 0; d < hd; d++) {
            float a = acc_row[d];
            for (int j = 0; j < ts; j++)
                a += tile_exp_buf[j] * v_tile[(size_t)j * hd + d];
            acc_row[d] = a;
        }
#endif
    }
}

static void batch_attention_tiled(
        float *xb_batch, const float *q_batch,
        const uint8_t *kcache, const uint8_t *vcache,
        int n_tokens, int start_pos,
        int n_heads, int n_kv_heads, int head_dim,
        int xb_stride,
        kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v,
        size_t kv_row_size_k, size_t kv_row_size_v,
        size_t kv_head_stride_k, size_t kv_head_stride_v,
        float attn_scale, int n_swa)
{
    /* kv_head_stride_v is used below in V-tile extraction */
    int kv_mul = n_heads / n_kv_heads;
    int tile = ATTN_TILE;

    /* Clamp tile to n_tokens for small batches */
    if (tile > n_tokens) tile = n_tokens;
    if (tile < 1) tile = 1;

    int n_token_groups = (n_tokens + tile - 1) / tile;
    int n_kv_tiles = (start_pos + n_tokens + tile - 1) / tile;

    gguf_type_t gguf_k = kv_cache_to_gguf_type((kv_cache_type_t)kv_type_k);
    gguf_type_t gguf_v = kv_cache_to_gguf_type((kv_cache_type_t)kv_type_v);
    size_t k_rb_gguf = gguf_type_row_size(gguf_k, head_dim);

    if (n_kv_heads < 1 || n_token_groups < 1) return;

    /* Run all (kv_head, token_group) tasks in parallel via tensor_parallel_for.
     * Each task processes one (kv_head, token_group) pair across ALL KV tiles,
     * with its own pre-allocated scratch buffers (no malloc inside the task). */
    batch_tiled_ctx_t bctx;
    memset(&bctx, 0, sizeof(bctx));
    bctx.xb_batch = xb_batch;
    bctx.q_batch = q_batch;
    bctx.kcache = kcache;
    bctx.vcache = vcache;
    bctx.n_tokens = n_tokens;
    bctx.start_pos = start_pos;
    bctx.n_heads = n_heads;
    bctx.n_kv_heads = n_kv_heads;
    bctx.kv_mul = kv_mul;
    bctx.head_dim = head_dim;
    bctx.xb_stride = xb_stride;
    bctx.kv_type_k = (int)kv_type_k;
    bctx.kv_type_v = (int)kv_type_v;
    bctx.kv_row_size_k = kv_row_size_k;
    bctx.kv_row_size_v = kv_row_size_v;
    bctx.kv_head_stride_k = kv_head_stride_k;
    bctx.kv_head_stride_v = kv_head_stride_v;
    bctx.attn_scale = attn_scale;
    bctx.n_swa = n_swa;
    bctx.tile = tile;
    bctx.n_token_groups = n_token_groups;
    bctx.n_kv_tiles = n_kv_tiles;
    bctx.gguf_k = gguf_k;
    bctx.gguf_v = gguf_v;
    bctx.k_rb_gguf = k_rb_gguf;

    /* Pre-allocate scratch for all tasks. Each task needs:
     *   scores: kv_mul*tile * tile floats
     *   M,S:    kv_mul*tile floats each
     *   acc:    kv_mul*tile * head_dim floats
     *   q_rows: kv_mul*tile * head_dim floats
     *   tile_k: tile * k_rb_gguf bytes
     *   tile_v: tile * head_dim floats
     * We allocate a pool of scratch spaces, one per possible concurrent task.
 * The maximum concurrent tasks equals the thread count. */
    int n_tasks = n_kv_heads * n_token_groups;
    /* Scratch slots must cover the maximum number of concurrently running
     * tasks. tensor_parallel_for runs up to pool_total_threads(n_threads)
     * tasks at once (one per pool thread), so cap at the THREAD COUNT, not
     * an arbitrary 16. The old cap of 16 made slot = task_idx % 16 alias
     * between concurrently running tasks whenever active > 16 (e.g. 24
     * threads: thread 0 on idx 0 and thread 8 on idx 112 both use slot 0),
     * silently corrupting attention scratch (scores/acc) for prompts long
     * enough to spawn > 16 tasks. Cost: ~370KB per slot x n_threads. */
    int max_concurrent;
    {
        int nth = tensor_get_n_threads();
        max_concurrent = n_tasks < nth ? n_tasks : nth;
    }
    size_t scores_per = (size_t)kv_mul * tile * tile * sizeof(float);
    size_t ms_per     = (size_t)kv_mul * tile * sizeof(float);
    size_t acc_per    = (size_t)kv_mul * tile * head_dim * sizeof(float);
    size_t qr_per     = (size_t)kv_mul * tile * head_dim * sizeof(float);
    size_t tk_per     = (size_t)tile * k_rb_gguf;
    size_t tv_per     = (size_t)tile * head_dim * sizeof(float);
    size_t scratch_per_task = scores_per + 2*ms_per + acc_per + qr_per + tk_per + tv_per;
    size_t total_scratch = (size_t)max_concurrent * scratch_per_task;

    /* Scratch pool: pre-allocated, sliced per concurrent task.
     * Uses mmap + guard page on UNIX to catch overflows; malloc on WIN32.
     * Guard page requires page-aligned boundary -- if mprotect fails we
     * fall back to malloc so cleanup can't mix free() and munmap(). */
    uint8_t *scratch_pool;
    int scratch_is_mmap = 0;
#ifndef _WIN32
    {
        size_t pg = (size_t)getpagesize();
        size_t aligned = (total_scratch + pg - 1) & ~(pg - 1);
        uint8_t *pool = mmap(NULL, aligned + pg,
                             PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (pool != MAP_FAILED &&
            mprotect(pool + aligned, pg, PROT_NONE) == 0) {
            scratch_pool = pool;
            scratch_is_mmap = 1;
        } else {
            scratch_pool = malloc(total_scratch);
        }
        (void)pg;
    }
#else
    scratch_pool = malloc(total_scratch);
#endif
    if (!scratch_pool) {
        /* OOM: fall back to the per-(token,head) non-tiled path */
        prefill_attn_ctx_t ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.n_heads = n_heads; ctx.n_kv_heads = n_kv_heads; ctx.kv_mul = kv_mul;
        ctx.head_dim = head_dim; ctx.start_pos = start_pos;
        ctx.kv_type_k = (int)kv_type_k; ctx.kv_type_v = (int)kv_type_v;
        ctx.kv_row_size_k = kv_row_size_k; ctx.kv_row_size_v = kv_row_size_v;
        ctx.kv_head_stride_k = kv_head_stride_k; ctx.kv_head_stride_v = kv_head_stride_v;
        ctx.kcache = kcache; ctx.vcache = vcache;
        ctx.q_batch = q_batch; ctx.xb_batch = xb_batch; ctx.xb_stride = xb_stride;
        ctx.attn_scale = attn_scale;
        ctx.n_swa = n_swa;
        tensor_parallel_for(n_tokens * n_kv_heads, prefill_attn_task_grouped, &ctx);
        return;
    }
    bctx.scratch_pool = scratch_pool;
    bctx.scratch_per_task = scratch_per_task;
    bctx.scores_per = scores_per;
    bctx.ms_per = ms_per;
    bctx.acc_per = acc_per;
    bctx.qr_per = qr_per;
    bctx.tk_per = tk_per;
    bctx.tv_per = tv_per;
    bctx.max_concurrent = max_concurrent;

    /* Strided dispatch: task index -> (kv_head, token_group) with work growing
     * linearly with the group index (causal: group g sees g+1 KV tiles).
     * Contiguous chunking hands the last thread ~4x the work of the first
     * (measured: 226ms avg wall per layer vs 57ms perfectly-balanced).
     * Striding interleaves heavy and light tasks across threads so every
     * thread's total converges to the mean. Scratch slots stay safe: with
     * stride = active, in-flight indices are t + w*active, so idx % active
     * is unique per in-flight task (given max_concurrent >= active). */
    tensor_parallel_for_strided(n_tasks, batch_tiled_task, &bctx);
    /* Release the pool */
#ifndef _WIN32
    if (scratch_is_mmap) {
        size_t pg = (size_t)getpagesize();
        size_t aligned = (total_scratch + pg - 1) & ~(pg - 1);
        munmap(scratch_pool, aligned + pg);
    } else
#endif
    {
        free(scratch_pool);
    }
}
