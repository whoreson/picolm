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
    int max_seq_len;    /* maximum sequence length (e.g. 2048) */
    int head_dim;       /* = n_embd / n_heads */
    float rope_freq_base; /* RoPE theta base (e.g. 10000.0) */
    int alignment;      /* GGUF data alignment */
    gguf_type_t weight_type; /* default weight quantization type */
} model_config_t;

/* ---- Per-layer weight pointers (into mmap) ---- */

typedef struct {
    const void *attn_norm;
    const void *attn_q;
    const void *attn_k;
    const void *attn_v;
    const void *attn_output;
    const void *ffn_norm;
    const void *ffn_gate;
    const void *ffn_down;
    const void *ffn_up;
    /* Per-tensor quantization types */
    gguf_type_t type_attn_norm;
    gguf_type_t type_attn_q;
    gguf_type_t type_attn_k;
    gguf_type_t type_attn_v;
    gguf_type_t type_attn_output;
    gguf_type_t type_ffn_norm;
    gguf_type_t type_ffn_gate;
    gguf_type_t type_ffn_down;
    gguf_type_t type_ffn_up;
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
    float *ffn_norm_w[MAX_LAYERS];
    float *output_norm_w;

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
    uint32_t    tok_bos_id;
    uint32_t    tok_eos_id;
    /* Pre-tokenizer type: 0=U+2581 (default), 1=U+0100 (smollm) */
    int         tok_space_marker;

    /* Runtime repacked weight buffers (for AVX2 Q4_0_8x8 optimization) */
    void       *repack_buffers[MAX_LAYERS + 4]; /* per-layer repacked data + output norms */
    int         repack_used[MAX_LAYERS + 4];    /* 1 if repacked, 0 if not */
} model_t;

/* Load a GGUF model file. Returns 0 on success. */
int model_load(model_t *m, const char *path, int max_seq_len, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v);

/* Run one forward pass. Returns pointer to logits[vocab_size]. */
float *model_forward(model_t *m, int token, int pos);

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
