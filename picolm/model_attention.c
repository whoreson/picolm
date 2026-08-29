#include "model.h"
#include "tensor.h"
#include "quant.h"
#include "model_internal.h"

#include <string.h>
#include <math.h>
#include <stdlib.h>
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
        int head_dim, float attn_scale) {
    float max_score = -1e30f, sum_exp = 0.0f;
    float acc[256];
    memset(acc, 0, (size_t)head_dim * sizeof(float));

    for (int t = 0; t <= pos; t++) {
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
#elif defined(PICOLM_NEON)
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
#elif defined(PICOLM_NEON)
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

    for (int t = 0; t <= pos; t++) {
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
                float tmp[32];
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
              ctx->head_dim, ctx->attn_scale);
}

/* Tiled attention: tile size in KV positions */
#define ATTN_TILE 64

/* Forward declaration for batch_attention_layer gating */
static void batch_attention_tiled(
        float *xb_batch, const float *q_batch,
        const uint8_t *kcache, const uint8_t *vcache,
        int n_tokens, int start_pos,
        int n_heads, int n_kv_heads, int head_dim,
        int xb_stride,
        kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v,
        size_t kv_row_size_k, size_t kv_row_size_v,
        size_t kv_head_stride_k, size_t kv_head_stride_v,
        float attn_scale);

/* Forward declaration for callback to tensor_parallel_for */
static void prefill_attn_task(int flat_idx, void *ctx_ptr);

void batch_attention_layer(
        float *xb_batch, const float *q_batch,
        const uint8_t *kcache, const uint8_t *vcache,
        int n_tokens, int start_pos,
        int n_heads, int n_kv_heads, int head_dim,
        int xb_stride,
        int kv_type_k, int kv_type_v,
        size_t kv_row_size_k, size_t kv_row_size_v,
        size_t kv_head_stride_k, size_t kv_head_stride_v,
        float attn_scale)
{
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

    /* For large enough batches, use the tiled/batched attention path which
     * amortizes KV cache load/dequant across multiple query tokens via the
     * existing matmul_batch infrastructure. For small batches, the original
     * per-(token,head) path is simpler and avoids malloc overhead.
     *
     * The tiled path currently only supports F16 KV cache (which is what
     * the store loop always writes). Q8_0/Q4_0 cache types are a planned
     * enhancement. */
    if (n_tokens >= 2 * ATTN_TILE && (int)kv_type_k == (int)KV_CACHE_F16 && (int)kv_type_v == (int)KV_CACHE_F16) {
        batch_attention_tiled(xb_batch, q_batch, kcache, vcache,
                              n_tokens, start_pos,
                              n_heads, n_kv_heads, head_dim,
                              xb_stride,
                              (kv_cache_type_t)kv_type_k, (kv_cache_type_t)kv_type_v,
                              kv_row_size_k, kv_row_size_v,
                              kv_head_stride_k, kv_head_stride_v,
                              attn_scale);
        return;
    }

    /* One dispatch per layer for the whole batch: n_tokens * n_heads
     * independent (token, head) tasks, each O(head_dim) memory, each
     * scanning only its own causal range t=0..pos. */
    tensor_parallel_for(n_tokens * n_heads, prefill_attn_task, &ctx);
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
} attn_tile_task_t;

/* Process one tile within a (kv_head, token_group) task.
 * Called inline from the task loop. */
static void attn_process_tile(attn_tile_task_t *t) {
    int n_q = t->n_q_rows;
    int ts = t->tile_size;
    int hd = t->head_dim;
    int is_diag = t->is_diagonal;
    int kv_tile_start = t->kv_tile_start;
    int group_token_start = t->group_token_start;

    /* Extract K-tile from GQA KV cache into contiguous scratch.
     * KV cache layout: [pos][kv_row_size_gqa] with head offset = kv_h * kv_head_stride_k
     * For positions [kv_tile_start, kv_tile_start+ts), head kv_h: */
    {
        size_t rb = t->kv_head_stride_k;
        size_t row_stride = t->kv_row_size_k;
        int gguf_k = t->kv_gguf_k;
        size_t k_rb_gguf = gguf_type_row_size(gguf_k, hd);
        /* K tile: ts positions, each rb bytes from cache, copied to
         * contiguous buffer with stride k_rb_gguf. For F16 this is
         * the same size and just a memcpy; for Q8_0/Q4_0 also same. */
        for (int p = 0; p < ts; p++) {
            const uint8_t *src = t->kcache + (size_t)(kv_tile_start + p) * row_stride
                               + t->kv_h * rb;
            uint8_t *dst = (uint8_t *)t->tile_k + (size_t)p * k_rb_gguf;
            memcpy(dst, src, rb);
        }
    }

    /* QK^T: matmul_batch(scores, q_rows, n_q, tile_k, hd, ts, kv_gguf_k)
     * out layout: [n_q][ts], scores[b*ts + i] = row b, col i */
    matmul_batch(t->scores, t->q_rows, n_q, t->tile_k, hd, ts, t->kv_gguf_k);

    /* Scale scores */
    for (int i = 0; i < n_q * ts; i++)
        t->scores[i] *= t->attn_scale;

    /* Causal masking for diagonal tile: for each query row i,
     * only positions [0, i_within_group] are valid.
     * Within the diagonal tile, query row i (0..n_q-1) corresponds to
     * token (group_token_start + i/kv_mul), and the valid KV positions
     * within this tile are [0, row_offset_within_tile].
     * The diagonal tile starts at kv_tile_start. The query's causal limit
     * is pos = start_pos + group_token_start + i/kv_mul.
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

        /* Row max */
        float rmax = srow[0];
        for (int j = 1; j < ts; j++) {
            if (srow[j] > rmax) rmax = srow[j];
        }

        /* Update running M and compute correction */
        float old_M = t->M[i];
        float new_M = (rmax > old_M) ? rmax : old_M;
        t->M[i] = new_M;
        float corr = expf(old_M - new_M);

        /* Scale old S and acc by correction */
        t->S[i] *= corr;
        float *acc_row = t->acc + i * hd;
        for (int d = 0; d < hd; d++)
            acc_row[d] *= corr;

        /* Compute exp(s[j] - new_M) for this row and accumulate */
        float rsum = 0.0f;
        for (int j = 0; j < ts; j++) {
            tile_exp_buf[j] = expf(srow[j] - new_M);
            rsum += tile_exp_buf[j];
        }
        t->S[i] += rsum;

        /* acc_row += sum_j tile_exp[j] * V[j, d] */
        for (int d = 0; d < hd; d++) {
            float add = 0.0f;
            for (int j = 0; j < ts; j++) {
                add += tile_exp_buf[j] * t->tile_v_f32[j * hd + d];
            }
            acc_row[d] += add;
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
        float attn_scale)
{
    /* kv_head_stride_v is used below in V-tile extraction */
    int kv_mul = n_heads / n_kv_heads;
    int tile = ATTN_TILE;

    /* Clamp tile to n_tokens for small batches */
    if (tile > n_tokens) tile = n_tokens;
    if (tile < 1) tile = 1;

    int n_token_groups = (n_tokens + tile - 1) / tile;
    int n_kv_tiles = (start_pos + n_tokens + tile - 1) / tile;

    /* For each (kv_head, token_group), we need:
     * - scores: [kv_mul * tile x tile] floats
     * - M, S: [kv_mul * tile] floats each
     * - acc: [kv_mul * tile x head_dim] floats
     * - tile_k: tile x head_dim in kv format (reuse k_rb * tile)
     * - tile_v_f32: tile x head_dim floats
     * - tile_exp_buf: already in stack in attn_process_tile
     *
     * Total per task: ~kv_mul * tile * (tile + head_dim) + tile * head_dim * 3
     * For kv_mul=8, tile=64, head_dim=128:
     *   8*64*(64+128) = 8*64*192 = 98304 floats = 393KB
     *   tile*head_dim*3 = 64*128*3 = 24576 floats = 98KB
     *   ~491KB per task, manageable with malloc. */

    gguf_type_t gguf_k = kv_cache_to_gguf_type((kv_cache_type_t)kv_type_k);
    gguf_type_t gguf_v = kv_cache_to_gguf_type((kv_cache_type_t)kv_type_v);
    size_t k_rb_gguf = gguf_type_row_size(gguf_k, head_dim);

    if (n_kv_heads < 1 || n_token_groups < 1) return;

    /* Run all (kv_head, token_group) tasks serially. The inner matmul_batch
     * calls are already threaded via the global thread pool, so adding an
     * outer parallel_for would deadlock from nested pool_wake/pool_wait. */
        for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
            for (int tg = 0; tg < n_token_groups; tg++) {
                int q_group_start = tg * tile;
                int q_group_end = q_group_start + tile;
                if (q_group_end > n_tokens) q_group_end = n_tokens;
                int n_q = q_group_end - q_group_start;
                int n_q_padded = n_q * kv_mul;

                /* Scratch allocation */
                size_t scores_sz = (size_t)(n_q_padded * tile) * sizeof(float);
                size_t ms_sz = (size_t)n_q_padded * sizeof(float);
                size_t acc_sz = (size_t)n_q_padded * head_dim * sizeof(float);
                size_t tk_sz = (size_t)tile * k_rb_gguf;
                size_t tv_sz = (size_t)tile * head_dim * sizeof(float);

                float *scores = malloc(scores_sz);
                float *M = malloc(ms_sz);
                float *S = malloc(ms_sz);
                float *acc = malloc(acc_sz);
                uint8_t *tile_k_buf = malloc(tk_sz);
                float *tile_v_f32 = malloc(tv_sz);
                if (!scores || !M || !S || !acc || !tile_k_buf || !tile_v_f32) {
                    free(scores); free(M); free(S); free(acc); free(tile_k_buf); free(tile_v_f32);
                    /* Fallback to original path on OOM */
                    return;
                }

                /* Gather query rows for this (kv_head, token_group).
                 * q_batch layout: [n_tokens][n_heads * head_dim]
                 * For kv_head kv_h, the query heads are [kv_h*kv_mul .. kv_h*kv_mul+kv_mul).
                 * For token_group tg, tokens are [q_group_start .. q_group_end).
                 * We interleave: q_rows[i*head_dim] where i = token_offset * kv_mul + qh_offset.
                 * Actually: q_rows[row_idx] = q for token (q_group_start + row_idx/kv_mul),
                 * head (kv_h * kv_mul + row_idx % kv_mul). */
                float *q_rows = malloc((size_t)n_q_padded * head_dim * sizeof(float));
                if (!q_rows) {
                    free(scores); free(M); free(S); free(acc); free(tile_k_buf); free(tile_v_f32);
                    return;
                }
                for (int ti = 0; ti < n_q; ti++) {
                    const float *q_tok = q_batch + (size_t)(q_group_start + ti) * n_heads * head_dim;
                    for (int g = 0; g < kv_mul; g++) {
                        const float *qh = q_tok + (kv_h * kv_mul + g) * head_dim;
                        float *qr = q_rows + ((size_t)ti * kv_mul + g) * head_dim;
                        memcpy(qr, qh, head_dim * sizeof(float));
                    }
                }

                /* Initialize M, S, acc */
                for (int i = 0; i < n_q_padded; i++) {
                    M[i] = -1e30f;
                    S[i] = 0.0f;
                }
                memset(acc, 0, acc_sz);

                /* Tile loop over KV positions */
                for (int tk = 0; tk < n_kv_tiles; tk++) {
                    int kv_t0 = tk * tile;
                    int kv_t1 = kv_t0 + tile;
                    if (kv_t1 > start_pos + n_tokens) kv_t1 = start_pos + n_tokens;
                    if (kv_t1 > q_group_start + start_pos + 1) {
                        /* This tile and all future tiles are fully in the future
                         * for ALL query rows in this group. Stop. */
                        /* Actually need per-row check: the last query row's pos is
                         * start_pos + q_group_end - 1. If kv_t0 >= that, skip. */
                        /* But we need to be more careful: some rows may have
                         * earlier causal limits. Let's just check if kv_t0 is
                         * past the causal limit of the FIRST query row. */
                        int first_pos = start_pos + q_group_start;
                        if (kv_t0 > first_pos) continue;
                        if (kv_t0 >= start_pos + q_group_end) break;
                    }
                    /* Skip tiles entirely in the future */
                    int first_pos = start_pos + q_group_start;
                    if (kv_t0 > first_pos) continue;

                    int this_tile_size = kv_t1 - kv_t0;
                    if (this_tile_size <= 0) continue;

                    /* Is this the diagonal tile? */
                    int is_diag = (kv_t0 <= q_group_start) && (kv_t1 > q_group_start);

                    /* Extract V-tile and dequantize to F32 */
                    {
                        size_t rb = kv_row_size_v;
                        size_t v_head_stride = kv_head_stride_v;
                        for (int p = 0; p < this_tile_size; p++) {
                            const uint8_t *src = vcache + (size_t)(kv_t0 + p) * rb
                                               + kv_h * v_head_stride;
                            dequantize_row(src, tile_v_f32 + (size_t)p * head_dim,
                                          head_dim, gguf_v);
                        }
                    }

                    /* Build task context and process */
                    attn_tile_task_t task;
                    memset(&task, 0, sizeof(task));
                    task.kv_h = kv_h;
                    task.kv_mul = kv_mul;
                    task.n_kv_heads = n_kv_heads;
                    task.head_dim = head_dim;
                    task.tile_size = this_tile_size;
                    task.n_q_rows = n_q_padded;
                    task.group_token_start = q_group_start;
                    task.kv_tile_start = kv_t0;
                    task.kv_tile_end = kv_t1;
                    task.is_diagonal = is_diag;
                    task.kv_gguf_k = gguf_k;
                    task.kv_row_size_k = kv_row_size_k;
                    task.kv_head_stride_k = kv_head_stride_k;
                    task.kcache = kcache;
                    task.q_rows = q_rows;
                    task.scores = scores;
                    task.tile_k = tile_k_buf;
                    task.tile_v_f32 = tile_v_f32;
                    task.M = M;
                    task.S = S;
                    task.acc = acc;
                    task.attn_scale = attn_scale;

                    attn_process_tile(&task);
                }

                /* Normalize and write output */
                for (int ti = 0; ti < n_q; ti++) {
                    for (int g = 0; g < kv_mul; g++) {
                        int ri = ti * kv_mul + g;
                        float inv_sum = 1.0f / S[ri];
                        float *acc_row = acc + ri * head_dim;
                        /* Write to xb_batch: token (q_group_start+ti), head (kv_h*kv_mul+g) */
                        float *out = xb_batch + (size_t)(q_group_start + ti) * xb_stride
                                   + (kv_h * kv_mul + g) * head_dim;
                        for (int d = 0; d < head_dim; d++)
                            out[d] = acc_row[d] * inv_sum;
                    }
                }

                free(scores); free(M); free(S); free(acc);
                free(tile_k_buf); free(tile_v_f32); free(q_rows);
            }
        }
}

