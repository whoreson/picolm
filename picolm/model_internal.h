#ifndef MODEL_INTERNAL_H
#define MODEL_INTERNAL_H

/* Internal forward declarations shared between model_*.c files.
 * These are static in model.c but need cross-file visibility after the split. */

/* GGUF metadata value types */
enum {
    GGUF_META_UINT8   = 0,
    GGUF_META_INT8    = 1,
    GGUF_META_UINT16  = 2,
    GGUF_META_INT16   = 3,
    GGUF_META_UINT32  = 4,
    GGUF_META_INT32   = 5,
    GGUF_META_FLOAT32 = 6,
    GGUF_META_BOOL    = 7,
    GGUF_META_STRING  = 8,
    GGUF_META_ARRAY   = 9,
    GGUF_META_UINT64  = 10,
    GGUF_META_INT64   = 11,
    GGUF_META_FLOAT64 = 12,
};

/* --- SSM (Qwen3.5) --- */
extern void ssm_forward(model_t *m, run_state_t *s, float *x, float *residual,
                        layer_weights_t *lw, int il, int pos, void *gpu_lw);
extern void ssm_prefill_layer(model_t *m, run_state_t *s,
                              float *x_batch, float *xb_batch, float *xb2_batch,
                              float *hb_batch, float *hb2_batch,
                              layer_weights_t *lw, int l,
                              int n_tokens, int start_pos, int xb2_stride,
                              void **gpu_lw);
#ifdef PICOLM_GPU
extern int ssm_prefill_layer_gpu(model_t *m, run_state_t *s,
    float *bx, float *bxb, float *bq, float *battn_out, float *bffn_norm,
    float *bgate, float *bup, layer_weights_t *lw, int l,
    int n_tokens, int start_pos, int dev);
#endif

/* --- Attention --- */
extern void attn_core(
        float *xbh, const float *qh, int kv_h, int pos,
        const uint8_t *kcache, const uint8_t *vcache,
        int kv_type_k, int kv_type_v,
        size_t kv_row_size_k, size_t kv_row_size_v,
        size_t kv_head_stride_k, size_t kv_head_stride_v,
        int head_dim, float attn_scale);
extern void attention_group(int kv_head_idx, void *ctx_ptr);
extern void prefill_attn_task(int flat_idx, void *ctx_ptr);

/* Shared attention context struct */
typedef struct {
    int kv_h, kv_mul, n_kv_heads, head_dim, pos;
    int kv_type_k, kv_type_v;
    size_t kv_row_size_k, kv_row_size_v;
    size_t kv_head_stride_k, kv_head_stride_v;
    const uint8_t *kcache, *vcache;
    const float *q;
    float *xb;
    int kv_hadamard_k, kv_hadamard_v;
    int kv_hadamard_size;
    float attn_scale;
} attn_group_ctx_t;

/* Prefill attention context */
typedef struct {
    int n_heads, n_kv_heads, kv_mul, head_dim, start_pos;
    int kv_type_k, kv_type_v;
    size_t kv_row_size_k, kv_row_size_v;
    size_t kv_head_stride_k, kv_head_stride_v;
    const uint8_t *kcache, *vcache;
    const float *q_batch;
    float *xb_batch;
    int xb_stride;
    int kv_hadamard_k, kv_hadamard_v;
    int kv_hadamard_size;
    float attn_scale;
} prefill_attn_ctx_t;

/* --- MoE --- */
extern void moe_forward(model_t *m, run_state_t *s, const float *x, float *residual,
                        const layer_weights_t *lw);
extern void moe_forward_batch(model_t *m, run_state_t *s,
                              const float *x_batch, float *residual_batch,
                              int n_tokens, const layer_weights_t *lw);
extern void moe_expert_worker(int i, void *vctx);

/* --- Gemma-3n --- */
float *model_forward_gemma3n(model_t *m, int token, int pos);
extern void gemma3n_router(float *out, float *inp, int n_embd, int n_altup,
                           const float *norm_w, const float *router_w_f32,
                           float rms_norm_eps, float *tmp_buf);

/* --- KV Cache --- */
extern int _kv_flush(int fd, uint8_t *buf, size_t *buf_off);
extern void _kv_close(int fd, uint8_t *buf, size_t *buf_off);

/* --- GGUF Loader --- */
typedef struct { const uint8_t *data; size_t pos; size_t size; } reader_t;
extern int parse_gguf(model_t *m, int max_seq_len);
extern void prepare_mmap(const void *addr, size_t size);
extern void _do_prefault(const void *addr, size_t size);

#endif /* MODEL_INTERNAL_H */
