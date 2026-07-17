#ifndef MODEL_H
#define MODEL_H

#include "quant.h"
#include <stdint.h>
#include <stddef.h>

#define GGUF_MAGIC 0x46554747
#define MAX_LAYERS 64

/* Magic for KV cache files */
#define KVCACHE_MAGIC 0x4B564350  /* "KVCP" */

/* ---- Configuration ---- */

typedef struct {
    int n_embd;         /* embedding dimension (e.g. 2048) */
    int n_ffn;          /* feed-forward hidden size (e.g. 5632) */
    int n_heads;        /* number of attention heads (e.g. 32) */
    int n_kv_heads;     /* number of KV heads for GQA (e.g. 4) */
    int n_layers;       /* number of transformer layers (e.g. 22) */
    int vocab_size;     /* vocabulary size (e.g. 32000) */
    int is_qwen;        /* 1 if model architecture is qwen3/qwen35 */
    int max_seq_len;    /* maximum sequence length (e.g. 2048) */
    int head_dim;       /* = n_embd / n_heads */
    float rope_freq_base; /* RoPE theta base (e.g. 10000.0) */
    float rms_norm_eps;   /* RMS norm epsilon (e.g. 1e-5) */
    int rope_type;        /* 0=llama pairwise, 1=qwen2 interleaved */
    int rope_dim;         /* RoPE dimension (default=head_dim) */
    int alignment;      /* GGUF data alignment */
    gguf_type_t weight_type; /* default weight quantization type */
    /* SSM parameters (Qwen3.5) */
    int has_ssm;            /* 1 if model has SSM layers */
    int ssm_d_conv;         /* convolution kernel size */
    int ssm_d_state;        /* state size (= head_k_dim) */
    int ssm_n_group;        /* group count (= n_k_heads) */
    int ssm_dt_rank;        /* time step rank (= n_v_heads) */
    int ssm_d_inner;        /* inner size (= value_dim) */
    uint8_t layer_type[MAX_LAYERS]; /* 0=attention, 1=SSM */
    /* MTP (Multi-Token Prediction) - Qwen3.5 second release and later.
     * MTP layers sit at the end of the layer list (layers n_active..n_layers-1).
     * They have "nextn." tensors (e.g. blk.N.nextn.eh_proj, blk.N.nextn.enorm).
     * During generation, MTP layers are skipped in model_forward().
     * Full MTP support planned: run MTP layers on output embedding to produce
     * N candidate tokens, verify with fast forward pass (speculative decoding). */
    int has_mtp;              /* 1 if model has MTP layers */
    int n_mtp_layers;         /* number of MTP layers at the end of layer list */
} model_config_t;

/* ---- Per-layer weight pointers (into mmap) ---- */

typedef struct {
    const void *attn_norm;
    const void *attn_q;
    const void *attn_k;
    const void *attn_v;
    const void *attn_output;
    const void *attn_q_norm;   /* QK-norm (Qwen3): per-head RMSNorm weight [head_dim] */
    const void *attn_k_norm;   /* QK-norm (Qwen3): per-head RMSNorm weight [head_dim] */
    const void *post_attn_norm; /* post-attention norm (ffn_norm for older models) */
    const void *ffn_gate;
    const void *ffn_down;
    const void *ffn_up;
    /* SSM layer weights (Qwen3.5) */
    const void *attn_qkv;       /* SSM: [n_embd, conv_dim] */
    const void *attn_gate_ssm;  /* SSM: [n_embd, value_dim] */
    const void *ssm_a;          /* SSM: [dt_rank] F32 */
    const void *ssm_alpha;      /* SSM: [n_embd, dt_rank] F32/Q8_0 */
    const void *ssm_beta;       /* SSM: [n_embd, dt_rank] F32/Q8_0 */
    gguf_type_t type_ssm_alpha;
    gguf_type_t type_ssm_beta;
    const void *ssm_conv1d;     /* SSM: [d_conv, conv_dim] F32 */
    const void *ssm_dt;         /* SSM: [dt_rank] F32 bias */
    const void *ssm_norm;       /* SSM: [head_v_dim] F32 */
    const void *ssm_out;        /* SSM: [value_dim, n_embd] */
    /* Layer type */
    int is_attn_layer;          /* 1=full attention, 0=SSM */
    /* Per-tensor quantization types */
    gguf_type_t type_attn_norm;
    gguf_type_t type_attn_q;
    gguf_type_t type_attn_k;
    gguf_type_t type_attn_v;
    gguf_type_t type_attn_output;
    gguf_type_t type_attn_q_norm;
    gguf_type_t type_attn_k_norm;
    gguf_type_t type_post_attn_norm;
    gguf_type_t type_ffn_gate;
    gguf_type_t type_ffn_down;
    gguf_type_t type_ffn_up;
    /* SSM tensor types */
    gguf_type_t type_attn_qkv;
    gguf_type_t type_attn_gate_ssm;
    gguf_type_t type_ssm_out;
    gguf_type_t type_ssm_conv1d;
    gguf_type_t type_ssm_dt;
    gguf_type_t type_ssm_a;
    gguf_type_t type_ssm_norm;
} layer_weights_t;

typedef struct {
    const void *token_embd;
    gguf_type_t type_token_embd;
    const void *output_norm;
    gguf_type_t type_output_norm;
    const void *output;        /* final output projection (may alias token_embd) */
    gguf_type_t type_output;
    layer_weights_t layers[MAX_LAYERS];
} model_weights_t;

/* KV cache quantization type */
typedef enum {
    KV_CACHE_F16,   /* FP16 (default, 2 bytes per element) */
    KV_CACHE_Q8_0,  /* Q8_0 (1 byte + 2 byte scale per 32 elements) */
    KV_CACHE_Q4_0,  /* Q4_0 (0.5 bytes + 2 byte scale per 32 elements) */
} kv_cache_type_t;

/* ---- Runtime state (pre-allocated buffers) ---- */

typedef struct {
    float *x;            /* current activation [n_embd] */
    float *xb;           /* buffer after norm / attention output [n_embd] */
    float *xb2;          /* second buffer [n_embd] */
    float *q;            /* query vector [n_embd] */
    /* att buffer REMOVED - flash attention uses online softmax */
    float *hb;           /* FFN hidden buffer [n_ffn] */
    float *hb2;          /* FFN hidden buffer 2 [n_ffn] */
    float *logits;       /* output logits [vocab_size] */

    /* KV cache - can be FP16, Q8_0, or Q4_0 */
    uint8_t *key_cache;    /* quantized key cache */
    uint8_t *val_cache;    /* quantized val cache */

    /* KV cache metadata */
    kv_cache_type_t kv_type_k;
    kv_cache_type_t kv_type_v;
    size_t kv_row_size_k;  /* bytes per head row of K */
    size_t kv_row_size_v;  /* bytes per head row of V */

    float *dequant_scratch; /* scratch for matmul dequant [max(n_embd, n_ffn)] */

    /* Pre-computed RoPE cos/sin tables [max_seq_len * head_dim/2] */
    float *rope_cos;
    float *rope_sin;

    /* Pre-dequantized norm weights (small, keep in RAM) */
    float *norm_weights;
    float *attn_norm_w[MAX_LAYERS];
    float *post_attn_norm_w[MAX_LAYERS]; /* post-attention norm (was ffn_norm) */
    float *attn_q_norm_w[MAX_LAYERS];  /* QK-norm (Qwen3) */
    float *attn_k_norm_w[MAX_LAYERS];  /* QK-norm (Qwen3) */
    float *output_norm_w;
    /* SSM runtime state (Qwen3.5) */
    float *ssm_conv_state[MAX_LAYERS]; /* [(d_conv-1) * conv_dim] per SSM layer */
    float *ssm_state[MAX_LAYERS];      /* [ssm_d_state * ssm_d_inner] per SSM layer */
    float *ssm_tmp;                    /* scratch for ssm_forward */
    /* Pre-dequantized small SSM arrays */
    float *ssm_a_w[MAX_LAYERS];        /* [dt_rank] F32 */
    float *ssm_dt_w[MAX_LAYERS];       /* [dt_rank] F32 */
    float *ssm_norm_w[MAX_LAYERS];     /* [head_v_dim] F32 */
    float *ssm_conv1d_w[MAX_LAYERS];   /* [d_conv * conv_dim] F32 */

    /* Single allocation base */
    void *mem_block;
    size_t mem_size;

    /* Separate allocation for KV cache */
    void *kv_block;
    size_t kv_size;
} run_state_t;

/* ---- Model ---- */

typedef struct {
    model_config_t  config;
    model_weights_t weights;
    run_state_t     state;

    /* mmap bookkeeping */
    void  *mmap_addr;
    size_t mmap_size;
#ifdef _WIN32
    void  *file_handle;
    void  *map_handle;
#else
    int    fd;
#endif

    /* Tokenizer data offsets (filled by GGUF parser, used by tokenizer_load) */
    const void *tok_tokens_data;
    uint64_t    tok_n_tokens;
    const void *tok_scores_data;
    uint64_t    tok_n_scores;
    const void *tok_merges_data;
    uint64_t    tok_n_merges;
    const void *tok_token_type_data;
    uint64_t    tok_n_token_type;
    uint32_t    tok_bos_id;
    uint32_t    tok_eos_id;
    /* Pre-tokenizer type: 0=U+2581 (default), 1=U+0100 (smollm) */
    int         tok_space_marker;
    char        *tok_eos_str;

    /* Runtime repacked weight buffers (for AVX2 Q4_0_8x8 optimization) */
/* ri = 2 + layer * 9, max ri+6 for 64 layers = 577, so 580 is safe */
#define MAX_REPACK_SLOTS  580
    void       *repack_buffers[MAX_REPACK_SLOTS]; /* per-layer repacked data + output norms */
    int         repack_used[MAX_REPACK_SLOTS];    /* 1 if repacked, 0 if not */

    /* Weight pinning */
    int         locked_layers;   /* number of layers pinned in RAM (0=disabled) */
    /* Whether model was loaded from safetensors (norm weights need +1.0) */
    int         from_safetensors;
} model_t;

/* Load a GGUF model file. Returns 0 on success. */
int model_load(model_t *m, const char *path, int max_seq_len, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v);
int model_load_safetensors(model_t *m, const char *model_dir, int max_seq_len, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v);

/* Pin layer weights in RAM. Given a budget in bytes, locks the maximum
 * number of consecutive layers (starting from 0) plus global tensors.
 * Returns the number of layers locked, or 0 on failure. */
int model_lock_layers(model_t *m, size_t mem_bytes);

/* Enable prefaulting: touch every mmap page at load time to bring the
 * model into the page cache before inference begins. Call before model_load. */
void model_set_prefault(int v);
/* Unconditionally prefault model pages (for server mode, bypasses g_do_prefault). */
void model_prefault(model_t *m);

/* Unlock previously pinned weight layers. Returns 0 on success. */
int model_unlock_layers(model_t *m);

/* Run one forward pass. Returns pointer to logits[vocab_size]. */
int allocate_run_state(model_t *m, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v);
float *model_forward(model_t *m, int token, int pos);
float *model_forward_prefill(model_t *m, const int *tokens, int n_tokens, int start_pos);

/* Free all resources. */
void model_free(model_t *m);

/* ---- KV cache persistence ---- */

/* Save KV cache state for positions [0, n_pos) to a file.
 * Returns 0 on success. */
int kvcache_save(const model_t *m, const char *path, int n_pos);

/* Load KV cache state from a file. Returns the number of positions
 * loaded (0 on failure). Caller should start generation from this position. */
int kvcache_load(model_t *m, const char *path);

#endif /* MODEL_H */
