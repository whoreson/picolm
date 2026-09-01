#include "model.h"
#include "tensor.h"
#include "quant.h"
#include "model_internal.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifdef PICOLM_GPU
#include "backend_gpu.h"
#endif

#ifdef PICOLM_VIZ
#include "viz.h"
#endif


/* ================================================================
 * Gemma-3n forward pass
 * ================================================================ */

/* Plumbing debug file handle */
static FILE *g_plumbing_fp = NULL;

/* Compute magnitude: sqrt(sum of squared elements) */
static float gemma3n_calc_magnitude(float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sqrtf(sum);
}

/* Normalize: divide by magnitude */
static void gemma3n_normalize(float *x, int n, float mag) {
    float inv = 1.0f / (mag + 1e-12f);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* Router: norm -> matmul -> tanh -> [n_altup] */
void gemma3n_router(float *out, float *inp, int n_embd, int n_altup,
                           const float *norm_w, const float *router_w_f32,
                           float rms_norm_eps, float *tmp_buf) {
    /* RMSNorm */
    rmsnorm(tmp_buf, inp, norm_w, n_embd, rms_norm_eps);
    /* Scale by 1/n_embd */
    { float sc = 1.0f / (float)n_embd;
      for (int i = 0; i < n_embd; i++) tmp_buf[i] *= sc; }
    /* Matmul: router_w_f32 [n_embd, n_altup] -> [n_altup] */
    for (int a = 0; a < n_altup; a++) {
        float sum = 0;
        const float *col = router_w_f32 + a * n_embd;
        for (int i = 0; i < n_embd; i++) sum += tmp_buf[i] * col[i];
        out[a] = tanhf(sum);
    }
}

float *model_forward_gemma3n(model_t *m, int token, int pos) {
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;

    int dim = c->n_embd;
    int n_ffn = c->n_ffn;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int kv_mul = n_heads / n_kv_heads;
    int seq_len = c->max_seq_len;
    int rope_dim = (c->rope_dim > 0) ? c->rope_dim : head_dim;
    int half_dim = rope_dim / 2;
    int n_altup = c->n_altup;
    int i_altup_act = c->i_altup_act;
    int n_embd_altup = c->n_embd_altup;
    int n_layer_kv = c->n_layer_kv_from_start;
    int laurel_rank = c->laurel_rank;
    float rms_norm_eps = c->rms_norm_eps;
    float sqrt_dim = sqrtf((float)dim);
    float sqrt_altup_dim = sqrtf((float)n_embd_altup);
    float sqrt_2 = sqrtf(2.0f);
    float sqrt_embd = sqrtf((float)dim);

    /* Per-layer RoPE table selection for SWA (done inside layer loop) */
    int swa_period = c->swa_period;

    /* 1. Token embedding lookup, scaled by sqrt(n_embd) */
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
        dequantize_row(embd_row, s->x, dim, w->type_token_embd);
        for (int i = 0; i < dim; i++) s->x[i] *= sqrt_dim;
    }
    /* Plumbing check: dump embedding for specific token/pos */
    { 
      if(!g_plumbing_fp){const char *p=getenv("PICOLM_PLUMBING");if(p)g_plumbing_fp=fopen(p,"w");}
      if(g_plumbing_fp && pos==18 && token==496){ /* pos=18, token=" a" (496) */
        fprintf(g_plumbing_fp,"EMB token=%d rms=%.6f first5=%.6f,%.6f,%.6f,%.6f,%.6f\n",token,
          0,s->x[0],s->x[1],s->x[2],s->x[3],s->x[4]);
        double sm=0; for(int i=0;i<dim;i++){double v=s->x[i];sm+=v*v;}
        fprintf(g_plumbing_fp,"EMB rms=%.6f\n",sqrt(sm/dim));
      }}
    if (pos == 0 && getenv("PICOLM_DBG")) {
        double em = 0; for(int i=0;i<dim;i++){double v=s->x[i]; em+=v*v;}
    }

    /* 2. Build per-layer inputs: [n_embd_altup, n_layer]
     * From per_layer_tok_embd[token] * sqrt(n_embd_altup)
     * Then project: per_layer_model_proj * inpL / sqrt(n_embd) + norm -> add per_layer inputs
     * Then scale by 1/sqrt(2)
     */
    {
        /* per_layer_tok_embd: [n_embd_altup * n_layer, n_vocab] -> get token column */
        int total_embd = n_embd_altup * c->n_layers;
        size_t embd_row_bytes = gguf_type_row_size(w->type_per_layer_tok_embd, total_embd);
        const void *embd_row = (const uint8_t *)w->per_layer_tok_embd + (size_t)token * embd_row_bytes;
        dequantize_row(embd_row, s->gemma3n_per_layer_inp, total_embd, w->type_per_layer_tok_embd);
        for (int i = 0; i < total_embd; i++) s->gemma3n_per_layer_inp[i] *= sqrt_altup_dim;

        /* Project: per_layer_model_proj [n_embd, total_embd] @ inpL [n_embd] -> [total_embd]
         * GGUF tensor [dim, total_embd]: ne[0]=dim (fastest), ne[1]=total_embd
         * matmul(out, x, W, n, d): out has d elements, each is dot(x[n], W_row[i][n])
         * So matmul(s->xb, s->x, W, dim, total_embd) gives total_embd outputs. ✓ */
        matmul(s->xb, s->x, w->per_layer_model_proj, dim, total_embd, w->type_per_layer_model_proj);
        float proj_scale = 1.0f / sqrt_embd;
        for (int i = 0; i < total_embd; i++) s->xb[i] *= proj_scale;

        /* RMSNorm (per_layer_proj_norm) - reshape to [n_embd_altup, n_layer], norm each layer */
        const float *proj_norm_raw = (const float *)w->per_layer_proj_norm;
        for (int l = 0; l < c->n_layers; l++) {
            float *layer_inp = s->xb + l * n_embd_altup;
            rmsnorm(s->hb, layer_inp, proj_norm_raw, n_embd_altup, rms_norm_eps);
            memcpy(layer_inp, s->hb, n_embd_altup * sizeof(float));
        }

        /* Add per-layer tok_embd */
        for (int i = 0; i < total_embd; i++) s->xb[i] += s->gemma3n_per_layer_inp[i];

        /* Scale by 1/sqrt(2) */
        float inp_scale = 1.0f / sqrt_2;
        for (int i = 0; i < total_embd; i++) s->xb[i] *= inp_scale;

        /* Store in gemma3n_per_layer_inp: [n_embd_altup, n_layer] */
        memcpy(s->gemma3n_per_layer_inp, s->xb, total_embd * sizeof(float));
    }
    if (pos == 0 && getenv("PICOLM_DBG")) {
        double pm = 0; for(int i=0;i<n_embd_altup;i++){double v=s->gemma3n_per_layer_inp[i]; pm+=v*v;}
        double pm2 = 0; for(int i=0;i<n_embd_altup;i++){double v=s->gemma3n_per_layer_inp[c->n_layers*n_embd_altup-i-1]; pm2+=v*v;}
    }

    /* 3. ALTUP expand: create n_altup copies
     * inp_repeated = repeat inpL (n_altup-1 times)
     * altup_added = altup_proj * inp_repeated, normalize to target magnitude
     * inpL = concat(inpL, altup_added) -> [n_embd, n_altup] stored in gemma3n_altup_state
     */
    {
        /* Copy active altup (first) */
        memcpy(s->gemma3n_altup_state + i_altup_act * dim, s->x, dim * sizeof(float));
        /* Copy to other altups first (will be overwritten) */
        float target_mag = gemma3n_calc_magnitude(s->x, dim);

        for (int a = 0; a < n_altup - 1; a++) {
            int a_dst = (a < i_altup_act) ? a : a + 1;
            /* altup_proj: [n_embd, n_embd, n_altup-1], take slice a -> [n_embd, n_embd]
             * GGUF layout: ne[0]=dim, ne[1]=dim, ne[2]=n_altup-1 (fastest to slowest)
             * Slice a starts at byte offset a * dim * dim * elem_size.
             * Use uint8_t pointer for correct byte arithmetic regardless of element type (F16=2B). */
            float *dst = s->gemma3n_altup_state + a_dst * dim;
            size_t proj_elem_size = gguf_type_quant_size(w->type_altup_proj) / gguf_type_block_size(w->type_altup_proj);
            const void *proj_a = (const uint8_t *)w->altup_proj + a * dim * dim * proj_elem_size;
            matmul(dst, s->x, proj_a, dim, dim, w->type_altup_proj);
            /* Normalize to target magnitude */
            float new_mag = gemma3n_calc_magnitude(dst, dim);
            gemma3n_normalize(dst, dim, new_mag);
            /* Scale to target magnitude (new_mag is now ~1.0) */
            for (int i = 0; i < dim; i++) dst[i] *= target_mag;
        }
        if (pos == 0 && getenv("PICOLM_DBG")) {
            for(int a=0;a<n_altup;a++){
                double m=0; for(int i=0;i<dim;i++){double v=s->gemma3n_altup_state[a*dim+i]; m+=v*v;}
                fprintf(stderr,"%.1f%s",sqrt(m/dim),a<n_altup-1?",":"");
            }
            fprintf(stderr, "}\n");
        }
    }

    /* Plumbing check: dump altup state after expand */
    { if(g_plumbing_fp && pos==18 && token==496){
        fprintf(g_plumbing_fp,"ALTUP_EXPAND rms=");
        for(int a=0;a<n_altup;a++){
          double sm=0;for(int i=0;i<dim;i++){double v=s->gemma3n_altup_state[a*dim+i];sm+=v*v;}
          fprintf(g_plumbing_fp,"%.6f",sqrt(sm/dim));if(a<n_altup-1)fputc(',',g_plumbing_fp);}
        fputc('\n',g_plumbing_fp);}
    }

    /* 4. Transformer layers */
    {
        int stop_after = -999;
        const char *stop_env = getenv("PICOLM_STOP_LAYER");
        if (stop_env) stop_after = atoi(stop_env);
        for (int l = 0; l < c->n_layers; l++) {
            if (pos == 0 && getenv("PICOLM_DBG")) {
                for(int a=0;a<n_altup;a++){
                    double m=0; for(int i=0;i<dim;i++){double v=s->gemma3n_altup_state[a*dim+i]; m+=v*v;}
                    fprintf(stderr,"%.2f%s",sqrt(m/dim),a<n_altup-1?",":"");
                }
                fprintf(stderr, "}\n");
            }
            if (l == stop_after + 1) {
                /* Stop after layer l-1, skip remaining layers */
                break;
            }
        layer_weights_t *lw = &w->layers[l];
        float *predictions;
        float *active_pred;

        /* ALTUP predict: router -> predict_coef -> linear combine altups */
        {
            /* Get active altup */
            float *active = s->gemma3n_altup_state + i_altup_act * dim;
            /* Router: [n_embd] -> [n_altup] */
            gemma3n_router(s->gemma3n_router_out, active, dim, n_altup,
                          s->altup_router_norm_w[l], s->altup_router_w[l],
                          rms_norm_eps, s->xb);
            /* predict_coef: GGUF shape {n_altup, n_altup*n_altup}, ne[0]=n_altup, ne[1]=n_altup*n_altup.
             * ggml_mul_mat(predict_coef, modalities) computes:
             *   out[c] = sum_r predict_coef[r + c * n_altup] * modalities[r]
             * (reduction over ne[0]=n_altup, output has ne[1]=n_altup*n_altup elements) */
            float *all_coefs = s->xb2;
            for (int a = 0; a < n_altup * n_altup; a++) {
                float sum = 0;
                for (int r = 0; r < n_altup; r++) sum += s->gemma3n_router_out[r] * ((const float *)lw->altup_predict_coef)[r + a * n_altup];
                all_coefs[a] = sum;
            }
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                for(int r=0;r<n_altup;r++) fprintf(stderr,"%.4f%s",s->gemma3n_router_out[r],r<n_altup-1?",":"");
                fprintf(stderr, "}\n");
                for(int a=0;a<n_altup*n_altup;a++) fprintf(stderr,"%.4f%s",all_coefs[a],a<n_altup*n_altup-1?",":"");
                fprintf(stderr, "}\n");
                for(int a=0;a<4;a++) fprintf(stderr,"%.6f%s",((const float *)lw->altup_predict_coef)[a],a<3?",":"");
                fprintf(stderr, "}\n");
                for(int a=0;a<4;a++) fprintf(stderr,"%.6f%s",((const float *)lw->altup_correct_coef)[a],a<3?",":"");
                fprintf(stderr, "}\n");
            }
            /* GGML: ggml_mul_mat(cur_permuted, all_coefs) where
             *   cur_permuted has ne=[n_altup, n_embd, n_tokens] (data: cur[d,t,a])
             *   all_coefs has ne=[n_altup, n_altup, n_tokens] (from reshape_3d of flat [n_altup^2, n_tokens])
             * Result: ne=[n_embd, n_altup, n_tokens], then permuted back to [n_embd, n_tokens, n_altup]
             *   result[d, a1, t] = sum_a2 cur[d, t, a2] * all_coefs_3d[a2, a1, t]
             *   all_coefs_3d[a2, a1, t] = all_coefs_flat[a2 + a1*n_altup + t*n_altup^2]
             * So the coefficient for cur[a2] contributing to predictions[a1] is all_coefs[a2 + a1*n_altup] */
            predictions = s->gemma3n_predictions;
            for (int a = 0; a < n_altup; a++) {
                float *pred = predictions + a * dim;
                float *cur_a = s->gemma3n_altup_state;
                for (int d = 0; d < dim; d++) pred[d] = all_coefs[0 + a * n_altup] * cur_a[d];
                for (int a2 = 1; a2 < n_altup; a2++) {
                    cur_a = s->gemma3n_altup_state + a2 * dim;
                    for (int d = 0; d < dim; d++) pred[d] += all_coefs[a2 + a * n_altup] * cur_a[d];
                }
            }

            /* Residual: predictions += cur */
            for (int d = 0; d < dim * n_altup; d++) predictions[d] += s->gemma3n_altup_state[d];

            /* Active prediction */
            active_pred = predictions + i_altup_act * dim;
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double apm=0; for(int i=0;i<dim;i++){double v=active_pred[i]; apm+=v*v;}
                for(int a=0;a<n_altup;a++){
                    double pm=0; for(int i=0;i<dim;i++){double v=predictions[a*dim+i]; pm+=v*v;}
                }
                for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",active_pred[i],i<4?",":"");
                fprintf(stderr, "}\n");
            }
            /* ATTN NORM */
            rmsnorm(s->xb, active_pred, s->attn_norm_w[l], dim, rms_norm_eps);

            /* LAUREL: norm(laurel_r @ (laurel_l @ x_normed)) + x_normed
             * x is the ATTN-NORMED active prediction (s->xb) */
            {
                int laurel_t = getenv("PICOLM_LAUREL_T") ? 1 : 0;
                /* s->xb holds the attn-normed active prediction */
                if (laurel_t) {
                    /* Transposed access: out[i] = sum_n W[n,i] * x[n] */
                    const uint16_t *wl = (const uint16_t *)lw->laurel_l;
                    for (int i = 0; i < laurel_rank; i++) {
                        float sum = 0.0f;
                        for (int n = 0; n < dim; n++)
                            sum += fp16_to_fp32(wl[n * laurel_rank + i]) * s->xb[n];
                        s->hb[i] = sum;
                    }
                    const uint16_t *wr = (const uint16_t *)lw->laurel_r;
                    for (int i = 0; i < dim; i++) {
                        float sum = 0.0f;
                        for (int n = 0; n < laurel_rank; n++)
                            sum += fp16_to_fp32(wr[n * dim + i]) * s->hb[n];
                        s->gemma3n_laurel_out[i] = sum;
                    }
                } else {
                    matmul(s->hb, s->xb, lw->laurel_l, dim, laurel_rank, lw->type_laurel_l);
                    matmul(s->gemma3n_laurel_out, s->hb, lw->laurel_r, laurel_rank, dim, lw->type_laurel_r);
                }
                rmsnorm(s->gemma3n_laurel_out, s->gemma3n_laurel_out, s->laurel_post_norm_w[l], dim, rms_norm_eps);
                if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                    double wm = 0; for(int i=0;i<dim;i++){double v=s->laurel_post_norm_w[l][i]; wm+=v*v;}
                    double am = 0; for(int i=0;i<dim;i++){double v=s->gemma3n_laurel_out[i]; am+=v*v;}
                }
                /* Residual: laurel_out + x_normed (not active_pred!) */
                for (int i = 0; i < dim; i++) s->gemma3n_laurel_out[i] += s->xb[i];
            }
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<dim;i++){double v=s->xb[i]; am+=v*v;}
                int skip_laurel = getenv("PICOLM_SKIP_LAUREL") ? 1 : 0;
                if (!skip_laurel) {
                    double lm = 0; for(int i=0;i<dim;i++){double v=s->gemma3n_laurel_out[i]; lm+=v*v;}
                }
            }
        }

        int kv_ordinal = (l < n_layer_kv) ? l : (n_layer_kv - 1);
        uint8_t *kcache_layer = s->key_cache + (size_t)kv_ordinal * seq_len * s->kv_row_size_k;
        uint8_t *vcache_layer = s->val_cache + (size_t)kv_ordinal * seq_len * s->kv_row_size_v;

        if (l < n_layer_kv) {
            /* KV-storing layer: compute Q, K, V, store K/V, attention */
            /* Q projection */
            matmul(s->q, s->xb, lw->attn_q, dim, q_dim, lw->type_attn_q);
            /* K projection */
            matmul(s->xb2, s->xb, lw->attn_k, dim, kv_dim, lw->type_attn_k);

            /* QK-norm */
            if (lw->attn_q_norm) {
                float *qnw = s->attn_q_norm_w[l];
                float *knw = s->attn_k_norm_w[l];
                for (int h = 0; h < n_heads; h++)
                    rmsnorm(s->q + h * head_dim, s->q + h * head_dim, qnw, head_dim, rms_norm_eps);
                for (int h = 0; h < n_kv_heads; h++)
                    rmsnorm(s->xb2 + h * head_dim, s->xb2 + h * head_dim, knw, head_dim, rms_norm_eps);
            }

            /* V projection + RMSNorm (unweighted in Gemma-3n) */
            matmul(s->hb, s->xb, lw->attn_v, dim, kv_dim, lw->type_attn_v);
            { /* V-norm: per-KV-head RMSNorm without learned weight (Gemma-3n) */
              for (int h = 0; h < n_kv_heads; h++) {
                  float ss = 0.0f;
                  float *v_head = s->hb + h * head_dim;
                  for (int i = 0; i < head_dim; i++) ss += v_head[i] * v_head[i];
                  ss = 1.0f / sqrtf(ss / (float)head_dim + rms_norm_eps);
                  for (int i = 0; i < head_dim; i++) v_head[i] *= ss;
              }
            }

            /* RoPE: select SWA or global table per layer
             * llama.cpp: is_swa(il) = il % swa_period < (swa_period - 1)
             * SWA layers use freq_base=10000, global uses freq_base=1000000 */
            {
                int is_swa = (swa_period > 0) && ((l % swa_period) < (swa_period - 1));
                float *lcos = (is_swa ? s->rope_cos_swa : s->rope_cos) + (size_t)pos * half_dim;
                float *lsin = (is_swa ? s->rope_sin_swa : s->rope_sin) + (size_t)pos * half_dim;
                rope(s->q, s->xb2, head_dim, n_heads, n_kv_heads, lcos, lsin, c->rope_type, half_dim);
            }

            /* Store K/V */
            {
                uint8_t *key_pos = kcache_layer + (size_t)pos * s->kv_row_size_k;
                if (s->kv_type_k == KV_CACHE_Q8_0) {
                    quantize_row_q8_0(s->xb2, key_pos, kv_dim);
                } else if (s->kv_type_k == KV_CACHE_Q4_0) {
                    quantize_row_q4_0(s->xb2, key_pos, kv_dim);
                } else {
                    for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                        float *k_head = s->xb2 + hkv * head_dim;
                        uint16_t *kf = (uint16_t *)(key_pos + hkv * s->kv_head_stride_k);
                        for (int d = 0; d < head_dim; d++) kf[d] = fp32_to_fp16(k_head[d]);
                    }
                }
            }
            {
                uint8_t *val_pos = vcache_layer + (size_t)pos * s->kv_row_size_v;
                if (s->kv_type_v == KV_CACHE_Q8_0) {
                    quantize_row_q8_0(s->hb, val_pos, kv_dim);
                } else if (s->kv_type_v == KV_CACHE_Q4_0) {
                    quantize_row_q4_0(s->hb, val_pos, kv_dim);
                } else {
                    for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                        float *v_head = s->hb + hkv * head_dim;
                        uint16_t *vf = (uint16_t *)(val_pos + hkv * s->kv_head_stride_v);
                        for (int d = 0; d < head_dim; d++) vf[d] = fp32_to_fp16(v_head[d]);
                    }
                }
            }

            /* Attention */
            {
                attn_group_ctx_t gctx;
                gctx.kv_mul = kv_mul; gctx.head_dim = head_dim; gctx.pos = pos;
                gctx.kv_type_k = s->kv_type_k; gctx.kv_type_v = s->kv_type_v;
                gctx.kv_row_size_k = s->kv_row_size_k; gctx.kv_row_size_v = s->kv_row_size_v;
                gctx.kv_head_stride_k = s->kv_head_stride_k; gctx.kv_head_stride_v = s->kv_head_stride_v;
                gctx.kcache = kcache_layer; gctx.vcache = vcache_layer;
                gctx.q = s->q; gctx.xb = s->xb;
                gctx.n_kv_heads = c->n_kv_heads;
                gctx.kv_hadamard_k = s->kv_hadamard_k;
                gctx.kv_hadamard_v = s->kv_hadamard_v;
                gctx.kv_hadamard_size = s->kv_hadamard_size;
                gctx.attn_scale = (c->f_attention_scale > 0) ? c->f_attention_scale : (1.0f / sqrtf((float)head_dim));
                tensor_parallel_for(c->n_kv_heads, attention_group, &gctx);
            }
        } else {
            /* Non-KV layer: only compute Q, reuse KV cache from last KV layer */
            /* Q projection */
            matmul(s->q, s->xb, lw->attn_q, dim, q_dim, lw->type_attn_q);

            /* Q-norm only (no K, V) */
            if (lw->attn_q_norm) {
                float *qnw = s->attn_q_norm_w[l];
                for (int h = 0; h < n_heads; h++)
                    rmsnorm(s->q + h * head_dim, s->q + h * head_dim, qnw, head_dim, rms_norm_eps);
            }

            /* RoPE on Q only (same SWA/global selection as KV layer above) */
            {
                int is_swa = (swa_period > 0) && ((l % swa_period) < (swa_period - 1));
                float *lcos = (is_swa ? s->rope_cos_swa : s->rope_cos) + (size_t)pos * half_dim;
                float *lsin = (is_swa ? s->rope_sin_swa : s->rope_sin) + (size_t)pos * half_dim;
                rope(s->q, s->xb2, head_dim, n_heads, n_kv_heads, lcos, lsin, c->rope_type, half_dim);
            }

            /* Attention (reuse KV from layer n_layer_kv-1) */
            {
                attn_group_ctx_t gctx;
                gctx.kv_mul = kv_mul; gctx.head_dim = head_dim; gctx.pos = pos;
                gctx.kv_type_k = s->kv_type_k; gctx.kv_type_v = s->kv_type_v;
                gctx.kv_row_size_k = s->kv_row_size_k; gctx.kv_row_size_v = s->kv_row_size_v;
                gctx.kv_head_stride_k = s->kv_head_stride_k; gctx.kv_head_stride_v = s->kv_head_stride_v;
                gctx.kcache = kcache_layer; gctx.vcache = vcache_layer;
                gctx.q = s->q; gctx.xb = s->xb;
                gctx.n_kv_heads = c->n_kv_heads;
                gctx.kv_hadamard_k = s->kv_hadamard_k;
                gctx.kv_hadamard_v = s->kv_hadamard_v;
                gctx.kv_hadamard_size = s->kv_hadamard_size;
                gctx.attn_scale = (c->f_attention_scale > 0) ? c->f_attention_scale : (1.0f / sqrtf((float)head_dim));
                tensor_parallel_for(c->n_kv_heads, attention_group, &gctx);
            }
        }

        /* Output projection: attn_output @ attn_result */
        matmul(s->xb2, s->xb, lw->attn_output, q_dim, dim, lw->type_attn_output);

        if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
            double am = 0; for(int i=0;i<dim;i++){double v=s->xb2[i]; am+=v*v;}
            for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->xb2[i],i<4?",":"");
            fprintf(stderr, "}\n");
        }
        /* attn_post_norm(attn_result) */
        if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
            double wm = 0; for(int i=0;i<dim;i++){double v=s->attn_post_norm_w[l][i]; wm+=v*v;}
        }
        rmsnorm(s->xb, s->xb2, s->attn_post_norm_w[l], dim, rms_norm_eps);
        if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
            double am = 0; for(int i=0;i<dim;i++){double v=s->xb[i]; am+=v*v;}
        }

        /* Add active prediction (residual) */
        for (int i = 0; i < dim; i++) s->xb[i] += predictions[i_altup_act * dim + i];

        /* Add laurel, scale by 1/sqrt(2) */
        {
            int skip_laurel = getenv("PICOLM_SKIP_LAUREL") ? 1 : 0;
            for (int i = 0; i < dim; i++) {
                s->x[i] = (s->xb[i] + (skip_laurel ? 0.0f : s->gemma3n_laurel_out[i])) / sqrt_2;
            }
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<dim;i++){double v=s->x[i]; am+=v*v;}
            }
        }

        /* FFN */
        {
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double wm = 0; for(int i=0;i<dim;i++){double v=s->post_attn_norm_w[l][i]; wm+=v*v;}
                for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->post_attn_norm_w[l][i],i<4?",":"");
                fprintf(stderr, "}\n");
            }
            rmsnorm(s->xb, s->x, s->post_attn_norm_w[l], dim, rms_norm_eps);
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<dim;i++){double v=s->xb[i]; am+=v*v;}
            }
            matmul(s->hb, s->xb, lw->ffn_gate, dim, n_ffn, lw->type_ffn_gate);
            /* Activation sparsity (gaussian_topk) for first n_layer_sparsity layers */
            if (c->is_gemma3n && l < c->n_layer_sparsity && c->n_layer_sparsity > 0) {
                float *gp = s->hb;
                /* mean = mean(gate_proj) */
                float mean = 0.0f;
                for (int i = 0; i < n_ffn; i++) mean += gp[i];
                mean /= (float)n_ffn;
                /* std = sqrt(sum((x - mean)^2) / n_ffn) -- population std, matching jnp.std(ddof=0) */
                float var = 0.0f;
                for (int i = 0; i < n_ffn; i++) { float d = gp[i] - mean; var += d * d; }
                var /= (float)n_ffn;
                float std = sqrtf(var);
                /* cutoff = mean + std * f_sparsity_std_mul */
                float cutoff = mean + std * c->f_sparsity_std_mul;
                /* relu(x - cutoff): zero out values below cutoff */
                for (int i = 0; i < n_ffn; i++) {
                    float v = gp[i] - cutoff;
                    gp[i] = (v > 0.0f) ? v : 0.0f;
                }
            }
            matmul(s->hb2, s->xb, lw->ffn_up, dim, n_ffn, lw->type_ffn_up);
            gelu(s->hb, n_ffn);  /* Gemma-3n uses GELU, not SiLU */
            elemwise_mul(s->hb, s->hb, s->hb2, n_ffn);
            matmul(s->xb, s->hb, lw->ffn_down, n_ffn, dim, lw->type_ffn_down);
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<dim;i++){double v=s->xb[i]; am+=v*v;}
            }
            /* post_ffw_norm */
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double wm = 0; for(int i=0;i<dim;i++){double v=s->post_ffw_norm_w[l][i]; wm+=v*v;}
                for(int i=0;i<3;i++) fprintf(stderr,"%.4f%s",s->post_ffw_norm_w[l][i],i<2?",":"");
                fprintf(stderr, "}\n");
            }
            rmsnorm(s->xb, s->xb, s->post_ffw_norm_w[l], dim, rms_norm_eps);
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<dim;i++){double v=s->xb[i]; am+=v*v;}
            }
            /* Add residual (attn_laurel combined) */
            for (int i = 0; i < dim; i++) s->xb[i] += s->x[i];
        }

        /* ALTUP Correct
         * activated = s->xb (output after FFN)
         * active_prediction = predictions[i_altup_act] (from predict step)
         * innovation = activated - active_prediction
         * correct_coefs = altup_correct_coef @ modalities + 1.0  [n_altup]
         * corrected = predictions + innovation * correct_coefs  [broadcast n_altup]
         */
        if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
            double am = 0; for(int i=0;i<dim;i++){double v=s->xb[i]; am+=v*v;}
        }
        {
            /* Router on activated state */
            gemma3n_router(s->gemma3n_router_out, s->xb, dim, n_altup,
                          s->altup_router_norm_w[l], s->altup_router_w[l],
                          rms_norm_eps, s->xb2);
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                for(int r=0;r<n_altup;r++) fprintf(stderr,"%.4f%s",s->gemma3n_router_out[r],r<n_altup-1?",":"");
                fprintf(stderr, "}\n");
            }

            /* correct_coefs = altup_correct_coef @ modalities + 1.0
             * GGUF shape {n_altup, n_altup}, ne[0]=n_altup, ne[1]=n_altup.
             * ggml_mul_mat: out[a] = sum_r correct_coef[r + a * n_altup] * modalities[r] */
            float *correct_coefs = s->xb2;
            const float *correct_coef = lw->altup_correct_coef;
            for (int a = 0; a < n_altup; a++) {
                float sum = 0;
                for (int r = 0; r < n_altup; r++)
                    sum += s->gemma3n_router_out[r] * correct_coef[r + a * n_altup];
                correct_coefs[a] = sum + 1.0f;
            }
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                for(int a=0;a<n_altup;a++) fprintf(stderr,"%.4f%s",correct_coefs[a],a<n_altup-1?",":"");
                fprintf(stderr, "}\n");
            }

            /* innovation = activated - predictions[i_altup_act] */
            float *active_prediction = predictions + i_altup_act * dim;
            for (int d = 0; d < dim; d++)
                s->hb[d] = s->xb[d] - active_prediction[d];

            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double im = 0; for(int i=0;i<dim;i++){double v=s->hb[i]; im+=v*v;}
            }

            /* corrected[a] = predictions[a] + innovation * correct_coefs[a] */
            for (int a = 0; a < n_altup; a++) {
                float coef = correct_coefs[a];
                float *pred_a = predictions + a * dim;
                float *cur_a = s->gemma3n_altup_state + a * dim;
                for (int d = 0; d < dim; d++)
                    cur_a[d] = pred_a[d] + s->hb[d] * coef;
            }

            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                for (int a = 0; a < n_altup; a++) {
                    double am = 0; for(int i=0;i<dim;i++){double v=s->gemma3n_altup_state[a*dim+i]; am+=v*v;}
                }
            }
        }

        /* Per-layer gating
         * first_prediction = corrected[i_altup_act] * altup_correct_scale
         * first_prediction = GELU(inp_gate @ first_prediction) * per_layer_inp[l]
         * first_prediction = rmsnorm(per_layer_proj @ first_prediction)
         * corrected[1:] += first_prediction
         */
        {
            float *first_pred = s->gemma3n_first_pred;
            /* Take active corrected altup */
            float *corrected_active = s->gemma3n_altup_state + i_altup_act * dim;
            /* Scale by altup_correct_scale */
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double sc_rms = 0; for(int i=0;i<dim;i++){double v=s->altup_correct_scale_w[l][i]; sc_rms+=v*v;}
                for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->altup_correct_scale_w[l][i],i<4?",":"");
                fprintf(stderr, "}\n");
            }
            for (int i = 0; i < dim; i++) first_pred[i] = corrected_active[i] * s->altup_correct_scale_w[l][i];

            /* inp_gate: [n_embd, n_embd_altup] */
            matmul(s->gemma3n_inp_gate_out, first_pred, lw->per_layer_inp_gate, dim, n_embd_altup, lw->type_per_layer_inp_gate);
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<n_embd_altup;i++){double v=s->gemma3n_inp_gate_out[i]; am+=v*v;}
                for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->gemma3n_inp_gate_out[i],i<4?",":"");
                fprintf(stderr, "}\n");
            }
            gelu(s->gemma3n_inp_gate_out, n_embd_altup);  /* Gemma-3n uses GELU */
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<n_embd_altup;i++){double v=s->gemma3n_inp_gate_out[i]; am+=v*v;}
                for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->gemma3n_inp_gate_out[i],i<4?",":"");
                fprintf(stderr, "}\n");
            }

            /* Multiply by per-layer input for this layer */
            float *layer_inp = s->gemma3n_per_layer_inp + l * n_embd_altup;
            for (int i = 0; i < n_embd_altup; i++) s->gemma3n_inp_gate_out[i] *= layer_inp[i];
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<n_embd_altup;i++){double v=s->gemma3n_inp_gate_out[i]; am+=v*v;}
            }

            /* per_layer_proj: [n_embd_altup, n_embd] */
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                /* Check per_layer_proj weight at row 1331 */
                for(int i=0;i<n_embd_altup;i++) {
                    /* per_layer_proj is [n_embd_altup, n_embd] = [256, 2048], row i is at offset i*n_embd */
                    /* matmul computes out[d] = sum_j x[j] * W[d * n_embd_altup + j] */
                    /* Wait, matmul(out, x, W, n, d) = out[i] = sum_j x[j] * W[i*n + j] */
                    /* So W is [d, n] = [2048, 256], row d is at offset d*256 */
                }
            }
            matmul(s->hb, s->gemma3n_inp_gate_out, lw->per_layer_proj, n_embd_altup, dim, lw->type_per_layer_proj);
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<dim;i++){double v=s->hb[i]; am+=v*v;}
                float maxx = 0, maxw = 0; int maxx_i = 0, maxw_i = 0;
                double x2w2 = 0;
                for(int i=0;i<dim;i++) {
                    if(fabsf(s->hb[i])>maxx){maxx=fabsf(s->hb[i]);maxx_i=i;}
                    if(fabsf(s->per_layer_post_norm_w[l][i])>maxw){maxw=fabsf(s->per_layer_post_norm_w[l][i]);maxw_i=i;}
                    x2w2 += (double)s->hb[i]*s->hb[i] * s->per_layer_post_norm_w[l][i]*s->per_layer_post_norm_w[l][i];
                }
            }
            rmsnorm(s->hb, s->hb, s->per_layer_post_norm_w[l], dim, rms_norm_eps);

            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<dim;i++){double v=s->hb[i]; am+=v*v;}
                double wn = 0; for(int i=0;i<dim;i++){double v=s->per_layer_post_norm_w[l][i]; wn+=v*v;}
            }
            /* Add to altup indices 1..n_altup-1 (reference: corrected[1:] += first_prediction)
             * Note: this is NOT the active altup. The reference always skips index 0,
             * regardless of which is the active altup. */
            for (int a = 1; a < n_altup; a++) {
                float *a_dst = s->gemma3n_altup_state + a * dim;
                for (int i = 0; i < dim; i++) a_dst[i] += s->hb[i];
            }
            /* Active altup already set by correct step */
        }

                /* cur = corrected predictions (all altups) -> next layer input is the concat */
#ifdef PICOLM_VIZ
        viz_push_layer(l, s->gemma3n_altup_state + i_altup_act * dim, dim);
#endif
    }
    } /* end of stop_layer scope */

    /* Plumbing check: dump final altup state before unembed */
    { if(g_plumbing_fp && pos==18 && token==496){
        fprintf(g_plumbing_fp,"POST_LAYERS rms=");
        for(int a=0;a<n_altup;a++){
          double sm=0;for(int i=0;i<dim;i++){double v=s->gemma3n_altup_state[a*dim+i];sm+=v*v;}
          fprintf(g_plumbing_fp,"%.6f",sqrt(sm/dim));if(a<n_altup-1)fputc(',',g_plumbing_fp);}
        fputc('\n',g_plumbing_fp);}
    }

    /* 5. ALTUP unembed: merge all altups back to single copy
     * llama.cpp always uses altup 0 as the base (not the active altup),
     * and unembeds altups 1..n_altup-1 through altup_unembd_proj[0..n_altup-2].
     * Target magnitude comes from the active altup. */
    {
        float *active = s->gemma3n_altup_state + i_altup_act * dim;
        float target_mag = gemma3n_calc_magnitude(active, dim);

        /* Start with altup 0 (always, not the active one) */
        memcpy(s->x, s->gemma3n_altup_state, dim * sizeof(float));

        size_t unembd_elem_size = gguf_type_quant_size(w->type_altup_unembd_proj) / gguf_type_block_size(w->type_altup_unembd_proj);
        for (int a = 0; a < n_altup - 1; a++) {
            /* altup a+1 is the source (skipping altup 0) */
            float *src = s->gemma3n_altup_state + (a + 1) * dim;
            /* unembed: altup_unembd_proj[a] * src */
            const void *unembd_a = (const uint8_t *)w->altup_unembd_proj + a * dim * dim * unembd_elem_size;
            matmul(s->xb, src, unembd_a, dim, dim, w->type_altup_unembd_proj);
            /* Normalize to target magnitude */
            float new_mag = gemma3n_calc_magnitude(s->xb, dim);
            gemma3n_normalize(s->xb, dim, new_mag);
            for (int i = 0; i < dim; i++) s->xb[i] *= target_mag;
            /* Add to accumulator */
            for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
        }

        /* Mean across all altups */
        float inv_n = 1.0f / (float)n_altup;
        for (int i = 0; i < dim; i++) s->x[i] *= inv_n;
    }

            /* Plumbing debug: dump hidden state at generation positions */
    if (pos >= 16 && pos <= 20) {
        double _sm = 0; for(int _i = 0; _i < dim; _i++) { double _v = s->x[_i]; _sm += _v*_v; }
        float _rms = sqrtf(_sm / dim);
        fprintf(stderr, "[PLUMB pos=%d rms=%.6f x0=%.6f]\n", pos, _rms, s->x[0]);
    }
    /* 6. Final RMSNorm */
    if (pos <= 1 && getenv("PICOLM_DBG")) {
        double om = 0; for(int i=0;i<dim;i++){double v=s->x[i]; om+=v*v;}
        for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->x[i],i<4?",":"");
        fprintf(stderr, "}\n");
    }
    /* Plumbing check: dump hidden state before and after final norm */
    { if(g_plumbing_fp && pos==18 && token==496){
        double sm=0;for(int i=0;i<dim;i++){double v=s->x[i];sm+=v*v;}
        fprintf(g_plumbing_fp,"PRE_NORM rms=%.6f first5=%.6f,%.6f,%.6f,%.6f,%.6f\n",sqrt(sm/dim),s->x[0],s->x[1],s->x[2],s->x[3],s->x[4]);}
    }
    rmsnorm(s->x, s->x, s->output_norm_w, dim, rms_norm_eps);
    { if(g_plumbing_fp && pos==18 && token==496){
        double sm=0;for(int i=0;i<dim;i++){double v=s->x[i];sm+=v*v;}
        fprintf(g_plumbing_fp,"POST_NORM rms=%.6f first5=%.6f,%.6f,%.6f,%.6f,%.6f\n",sqrt(sm/dim),s->x[0],s->x[1],s->x[2],s->x[3],s->x[4]);}
    }

    if (pos <= 1 && getenv("PICOLM_DBG")) {
        double om = 0; for(int i=0;i<dim;i++){double v=s->x[i]; om+=v*v;}
        for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->x[i],i<4?",":"");
        fprintf(stderr, "}\n");
    }
    /* 7. Output projection -> logits */
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);

    /* Plumbing check: dump logits before and after softcap */
    { if(g_plumbing_fp && pos==18 && token==496){
        fprintf(g_plumbing_fp,"RAW_LOGITS top5=");
        for(int rank=0;rank<5;rank++){
            int best=-1;float bestv=-1e30f;
            for(int i=0;i<c->vocab_size;i++) if(s->logits[i]>bestv){bestv=s->logits[i];best=i;}
            fprintf(g_plumbing_fp,"%d(%.4f) ",best,bestv);s->logits[best]=-1e30f;}
        fprintf(g_plumbing_fp,"\n");}
    }
    /* 8. Logit soft-capping: tanh(logits / softcap) * softcap */
    if (c->f_final_logit_softcapping > 0) {
        float inv_cap = 1.0f / c->f_final_logit_softcapping;
        for (int i = 0; i < c->vocab_size; i++) {
            s->logits[i] = tanhf(s->logits[i] * inv_cap) * c->f_final_logit_softcapping;
        }
    }
    { if(g_plumbing_fp && pos==18 && token==496){
        fprintf(g_plumbing_fp,"CAP_LOGITS top5=");
        for(int rank=0;rank<5;rank++){
            int best=-1;float bestv=-1e30f;
            for(int i=0;i<c->vocab_size;i++) if(s->logits[i]>bestv){bestv=s->logits[i];best=i;}
            fprintf(g_plumbing_fp,"%d(%.4f) ",best,bestv);s->logits[best]=-1e30f;}
        fprintf(g_plumbing_fp,"\n");fclose(g_plumbing_fp);g_plumbing_fp=NULL;}
    }

    if (pos <= 1 && getenv("PICOLM_DBG")) {
        int top_idx[5]; float top_val[5];
        for (int rank = 0; rank < 5; rank++) {
            int best = -1; float bestv = -1e30f;
            for (int i = 0; i < c->vocab_size; i++) {
                if (s->logits[i] > bestv) { bestv = s->logits[i]; best = i; }
            }
            top_idx[rank] = best; top_val[rank] = bestv;
            s->logits[best] = -1e30f;
            fprintf(stderr, "[%d:%.4f] ", best, bestv);
        }
        for (int rank = 0; rank < 5; rank++)
            s->logits[top_idx[rank]] = top_val[rank];
        fprintf(stderr, "\n");
    }

    return s->logits;
}

