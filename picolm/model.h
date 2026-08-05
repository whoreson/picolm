#ifndef MODEL_H
#define MODEL_H

#include "quant.h"
#include <stdint.h>
#include <stddef.h>

#define GGUF_MAGIC 0x46554747
#define MAX_LAYERS 128

/* GGUF format is always little-endian. On big-endian platforms, swap. */
#if defined(__APPLE__) && defined(__ppc__)
/* Mac OS X PPC: use libkern byte-order functions */
#include <libkern/OSByteOrder.h>
#define GGUF_LE32(v) OSSwapInt32(v)
#define GGUF_LE64(v) OSSwapInt64(v)
#define GGUF_LE16(v) OSSwapInt16(v)
#elif defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#include <byteswap.h>
#define GGUF_LE32(v) __builtin_bswap32(v)
#define GGUF_LE64(v) __builtin_bswap64(v)
#define GGUF_LE16(v) __builtin_bswap16(v)
#else
#define GGUF_LE32(v) (v)
#define GGUF_LE64(v) (v)
#define GGUF_LE16(v) (v)
#endif

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
    int is_gemma3n;     /* 1 if model architecture is gemma3n */
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
    /* MoE parameters (Qwen3.6-35B-A3B = qwen35moe architecture) */
    int has_moe;              /* 1 if model is MoE */
    int n_expert;             /* total number of experts (256) */
    int n_expert_used;        /* top-K experts per token (8) */
    int n_ff_exp;             /* per-expert FFN hidden dim (512) */
    int n_ff_shexp;           /* shared expert FFN hidden dim (512) */
    /* Gemma-3n specific */
    int n_altup;              /* number of altup copies (default 4) */
    int i_altup_act;          /* active altup index (default 0) */
    int laurel_rank;          /* LAUREL low-rank bottleneck dimension */
    int n_embd_altup;         /* per-layer input dimension (n_embd_altup = 256) */
    int n_layer_kv_from_start;/* number of layers with KV cache from start */
    float f_final_logit_softcapping; /* final logit soft-capping (0=disabled) */
} model_config_t;

/* ---- GPU-resident weight handles (compiled in only with PICOLM_GPU) ---- */
#ifdef PICOLM_GPU
/* Per-layer GPU tensor handles, parallel to layer_weights_t.
 * Each handle is an opaque picolm_gpu_tensor_t* cast to void*. */
typedef struct {
    void *attn_q;
    void *attn_k;
    void *attn_v;
    void *attn_output;
    void *ffn_gate;
    void *ffn_down;
    void *ffn_up;
    void *attn_qkv;            /* SSM */
    void *attn_gate_ssm;       /* SSM */
    void *ssm_out;             /* SSM */
} gpu_layer_weights_t;

typedef struct {
    void *output;
    gpu_layer_weights_t layers[MAX_LAYERS];
    int device;
    int active;    /* 1 if GPU backend is active and at least some tensors uploaded */
} gpu_weights_t;
#endif /* PICOLM_GPU */

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
    /* MoE tensors (3D expert weights) */
    const void *ffn_gate_exps;      /* [n_embd, n_ff_exp, n_expert] */
    const void *ffn_up_exps;        /* [n_embd, n_ff_exp, n_expert] */
    const void *ffn_down_exps;      /* [n_ff_exp, n_embd, n_expert] */
    const void *ffn_gate_inp;       /* [n_embd, n_expert] router */
    gguf_type_t type_ffn_gate_exps;
    gguf_type_t type_ffn_up_exps;
    gguf_type_t type_ffn_down_exps;
    gguf_type_t type_ffn_gate_inp;
    /* Shared expert tensors */
    const void *ffn_gate_inp_shexp; /* [n_embd, 1] */
    const void *ffn_gate_shexp;     /* [n_embd, n_ff_shexp] */
    const void *ffn_up_shexp;       /* [n_embd, n_ff_shexp] */
    const void *ffn_down_shexp;     /* [n_ff_shexp, n_embd] */
    gguf_type_t type_ffn_gate_shexp;
    gguf_type_t type_ffn_up_shexp;
    gguf_type_t type_ffn_down_shexp;
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
    /* Gemma-3n specific weights */
    const void *attn_post_norm;     /* post-attention norm [n_embd] */
    gguf_type_t type_attn_post_norm;
    const void *post_ffw_norm;      /* post-FFW norm [n_embd] */
    gguf_type_t type_post_ffw_norm;
    const void *per_layer_inp_gate;  /* [n_embd, n_embd_altup] */
    gguf_type_t type_per_layer_inp_gate;
    const void *per_layer_proj;      /* [n_embd_altup, n_embd] */
    gguf_type_t type_per_layer_proj;
    const void *per_layer_post_norm; /* [n_embd] */
    gguf_type_t type_per_layer_post_norm;
    const void *altup_correct_coef;  /* [n_altup, n_altup] F32 */
    const void *altup_correct_scale; /* [n_embd] F32 */
    const void *altup_predict_coef;  /* [n_altup, n_altup*n_altup] F32 */
    const void *altup_router;        /* [n_embd, n_altup] */
    gguf_type_t type_altup_router;
    const void *altup_router_norm;   /* [n_embd] F32 */
    const void *laurel_l;            /* [n_embd, laurel_rank] */
    gguf_type_t type_laurel_l;
    const void *laurel_r;            /* [laurel_rank, n_embd] */
    gguf_type_t type_laurel_r;
    const void *laurel_post_norm;    /* [n_embd] F32 */
} layer_weights_t;

typedef struct {
    const void *token_embd;
    gguf_type_t type_token_embd;
    const void *output_norm;
    gguf_type_t type_output_norm;
    const void *output;        /* final output projection (may alias token_embd) */
    gguf_type_t type_output;
    /* Gemma-3n global tensors */
    const void *altup_proj;            /* [n_embd, n_embd, n_altup-1] */
    gguf_type_t type_altup_proj;
    const void *altup_unembd_proj;     /* [n_embd, n_embd, n_altup-1] */
    gguf_type_t type_altup_unembd_proj;
    const void *per_layer_tok_embd;    /* [n_embd_altup * n_layer, n_vocab] */
    gguf_type_t type_per_layer_tok_embd;
    const void *per_layer_model_proj;  /* [n_embd, n_embd_altup * n_layer] */
    gguf_type_t type_per_layer_model_proj;
    const void *per_layer_proj_norm;   /* [n_embd_altup] */
    layer_weights_t layers[MAX_LAYERS];
} model_weights_t;

/* KV cache quantization type */
typedef enum {
    KV_CACHE_F16,   /* FP16 (default, 2 bytes per element) */
    KV_CACHE_Q8_0,  /* Q8_0 (1 byte + 2 byte scale per 32 elements) */
    KV_CACHE_Q4_0,  /* Q4_0 (0.5 bytes + 2 byte scale per 32 elements) */
    KV_CACHE_TQ3,   /* TurboQuant 3-bit: 3.5 bits/element (WHT + Lloyd-Max) */
    KV_CACHE_TQ4,   /* TurboQuant 4-bit: 4.5 bits/element (WHT + Lloyd-Max) */
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
    size_t kv_row_size_k;   /* bytes per GQA row of K (all KV heads concatenated) */
    size_t kv_row_size_v;   /* bytes per GQA row of V (all KV heads concatenated) */
    size_t kv_head_stride_k;/* bytes per head within a GQA row (for attention reads) */
    size_t kv_head_stride_v;/* bytes per head within a GQA row (for attention reads) */

    /* KV cache Hadamard rotation */
    int kv_hadamard_k;      /* apply Walsh-Hadamard rotation to K cache before quant */
    int kv_hadamard_v;      /* apply Walsh-Hadamard rotation to V cache before quant */
    int kv_hadamard_size;   /* block size for Hadamard (64, 128, etc.) */

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

    /* Gemma-3n runtime state */
    float *gemma3n_altup_state;        /* [n_altup * n_embd] for gen */
    float *gemma3n_per_layer_inp;      /* [n_embd_altup * n_layer] projected per-layer input */
    float *gemma3n_inp_gate_out;       /* [n_embd_altup] gated per-layer output */
    float *gemma3n_laurel_out;         /* [n_embd] laurel output */
    float *gemma3n_predictions;        /* [n_altup * n_embd] saved predictions for correct step */
    float *gemma3n_router_out;         /* [n_altup] router output */
    float *gemma3n_norm_w[MAX_LAYERS]; /* extra norm weights for gemma3n */
    float *attn_post_norm_w[MAX_LAYERS];
    float *post_ffw_norm_w[MAX_LAYERS];
    float *per_layer_post_norm_w[MAX_LAYERS];
    float *laurel_post_norm_w[MAX_LAYERS];
    float *altup_router_norm_w[MAX_LAYERS];
    float *altup_correct_scale_w[MAX_LAYERS];
    /* Pre-dequantized small Gemma-3n arrays */
    float *altup_correct_coef_w[MAX_LAYERS];  /* [n_altup * n_altup] */
    float *altup_predict_coef_w[MAX_LAYERS];  /* [n_altup * n_altup * n_altup] */
    float *laurel_l_w[MAX_LAYERS];
    float *laurel_r_w[MAX_LAYERS];
    float *altup_router_w[MAX_LAYERS];     /* [n_embd * n_altup] dequantized router weights */
    /* MoE runtime state */
    float *expert_logits;     /* router logits [n_expert] */
    int *expert_ids;          /* selected expert indices [n_expert_used] */
    float *expert_weights;    /* routing weights [n_expert_used] */
    float *moe_out;           /* MoE output accumulator [n_embd] */
    float *hb_exp;            /* per-expert hidden buffer [n_ff_exp] */
    float *hb2_exp;           /* second per-expert hidden buffer [n_ff_exp] */
    float *gate_batch;        /* batched gate outputs [n_ff_exp][n_expert_used] */
    float *up_batch;          /* batched up outputs [n_ff_exp][n_expert_used] */
    block_q8_0 *down_qx;      /* quantized SwiGLU for down projection */
    float *down_qx_d;         /* deltas */
    block_q8_0 *shared_qx;    /* quantized input for shared expert */
    float *shared_qx_d;       /* deltas */
    block_q8_0 *shared_down_qx; /* quantized SwiGLU for shared down */
    float *shared_down_qx_d;  /* deltas */

    /* Per-thread MoE scratch buffers for parallel dispatch (no malloc in workers).
     * Each thread gets: gate_buf[n_ff_exp], up_buf[n_ff_exp], xb2_buf[n_embd],
     * down_qx (Q8_0 row), down_qx_d (deltas), acc_buf[n_embd].
     * Allocated as [n_threads × per_thread_size] contiguous block. */
    void *moe_thread_scratch;  /* base pointer to per-thread scratch area */
    size_t moe_thread_stride;  /* bytes per thread */

    /* Pre-allocated batch buffers for weight-centric MoE path:
     * gate_batch_exp[n_ff_exp] — gate output for one expert (single token or batched)
     * up_batch_exp[n_ff_exp] — up output for one expert
     * down_qx_exp — Q8_0 quantized SwiGLU for down projection
     * down_qx_d_exp — deltas for down_qx_exp */
    float *gate_batch_exp;    /* [n_ff_exp] gate output buffer */
    float *up_batch_exp;      /* [n_ff_exp] up output buffer */
    block_q8_0 *down_qx_exp;  /* quantized SwiGLU for down projection */
    float *down_qx_d_exp;     /* deltas */

    /* Pre-allocated qx_all buffer: quantized Q8_0 inputs for all tokens.
     * Size: n_ctx × (q8_row_blocks + nb_deltas) floats. */
    float *moe_qx_all;
    int moe_qx_d_off;
    int moe_q8_buf_per_token;

    /* Shared expert batch buffers (gate+up for all tokens) */
    float *sh_gate;   /* [n_ff_shexp × n_ctx] */
    float *sh_up;     /* [n_ff_shexp × n_ctx] */

    /* Per-token accumulator for MoE: [n_embd × n_ctx].
     * Used by weight-centric path to accumulate expert outputs
     * in Top-K order (matching single-token moe_forward). */
    float *moe_acc_batch;

    /* mm_id buffers: per-token, per-slot expert outputs
     * mm_gate_out: [n_tokens * n_used * n_ff] — gate projections
     * mm_up_out:   [n_tokens * n_used * n_ff] — up projections
     * mm_down_out: [n_tokens * n_used * n_embd] — down projections (post-SwiGLU) */
    float *mm_gate_out;
    float *mm_up_out;
    float *mm_down_out;
    size_t mm_gateup_alloc;  /* allocated size of mm_gate_out/mm_up_out in bytes */
    size_t mm_down_alloc;    /* allocated size of mm_down_out in bytes */

    /* Scratch Q8_0 buffer for mm_id down projection (single-token) */
    block_q8_0 *mm_scratch_qx;
    float *mm_scratch_qx_d;

    /* Per-expert down quantization buffers for mm_id batching.
     * mm_down_qx_all: per-thread × per-token Q8_0 scratch for parallel expert dispatch
     * mm_down_qx_d_all: NULL (deltas embedded in Q8_0 buffer) */
    block_q8_0 *mm_down_qx_all;
    float *mm_down_qx_d_all;

    /* Precomputed routing map for mm_id dispatch.
     * expert_assignments: [n_expert × max_tokens] packed (token << 8 | slot)
     * expert_counts: [n_expert] number of assigned tokens per expert */
    int *expert_assignments;
    int *expert_counts;

    /* Single allocation base */
    void *mem_block;
    size_t mem_size;

    /* Separate allocation for KV cache */
    void *kv_block;
    size_t kv_size;

    /* KV cache layer mapping (for SSM/mixed models).
     * kv_layer_map[l] = 1 if layer l has KV cache (attention layer), 0 if SSM layer.
     * kv_layer_ordinal[l] = ordinal within KV layers (0..kv_layer_count-1), or -1 if SSM.
     * kv_layer_count = number of layers with KV cache data. */
    int kv_layer_count;
    uint8_t kv_layer_map[MAX_LAYERS];
    int kv_layer_ordinal[MAX_LAYERS];
} run_state_t;

/* ---- Split GGUF mmap entries ---- */

#define MAX_SPLIT_FILES 64

typedef struct {
    void  *mmap_addr;
    size_t mmap_size;
#ifdef _WIN32
    void  *file_handle;
    void  *map_handle;
#else
    int    fd;
#endif
} split_mmap_t;

/* ---- Model ---- */

typedef struct {
    model_config_t  config;
    model_weights_t weights;
    run_state_t     state;

    /* SSM batching: 0=per-token, 1=batched (default) */
    int             ssm_batched_prefill;
    /* SSM chunk size for batched prefill (default 64, 0=auto) */
    int             ssm_chunk_size;

    /* mmap bookkeeping - array for split files, legacy fields for compat */
    void     *mmap_addr;      /* kept for backward compat, points to splits[0].mmap_addr */
    size_t    mmap_size;
#ifdef _WIN32
    void     *file_handle;
    void     *map_handle;
#else
    int       fd;
#endif
    split_mmap_t splits[MAX_SPLIT_FILES]; /* one per split file */
    int        n_splits;                   /* number of split files (1 = non-split) */
    int        split_count;                /* split.count from metadata (0 if not split) */
    int        split_no;                   /* split.no of first split (0 if not split) */
    int        split_tensors_count;        /* split.tensors.count total (0 if not split) */
    size_t     tensor_data_base[MAX_SPLIT_FILES]; /* tensor data base per split */
    char       first_split_path[512];      /* path to first split file */

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
    /* Pre-tokenizer type: 0=U+2581 (default), 1=U+0100 (smollm), 2=literal space, 3=qwen35 */
    int         tok_space_marker;
    /* GGUF tokenizer flags */
    int         tok_add_bos;          /* tokenizer.ggml.add_bos_token (default: 1) */
    int         tok_add_space_prefix; /* tokenizer.ggml.add_space_prefix (default: 1) */
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
#ifdef PICOLM_GPU
    /* GPU-resident weight tensors */
    gpu_weights_t gpu;
#endif
} model_t;

/* Load a GGUF model file. Returns 0 on success. */
int model_load(model_t *m, const char *path, int max_seq_len, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v,
               int k_cache_hadamard, int v_cache_hadamard);
/* List all tensors in a GGUF file (name, dims, type) and exit. Returns 0 on success. */
int model_list_tensors(const char *path);
/* List all KV metadata entries in a GGUF file and exit. Returns 0 on success. */
int model_list_kv(const char *path);
int model_load_safetensors(model_t *m, const char *model_dir, int max_seq_len, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v,
                           int k_cache_hadamard, int v_cache_hadamard);

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

/* Benchmark layer callback: called at end of each layer during forward/prefill.
 * Set via model_set_bench_callback(). Parameters: layer index, 1=prefill 0=gen,
 * wall-clock ms elapsed in this layer, page fault counts (minflt + majflt). */
typedef void (*bench_layer_cb_t)(int layer, int is_prefill, double elapsed_ms, long minflt, long majflt, void *user_data);
void model_set_bench_callback(bench_layer_cb_t cb, void *user_data);
size_t layer_weight_size(model_t *m, int l);

/* Run one forward pass. Returns pointer to logits[vocab_size]. */
int allocate_run_state(model_t *m, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v,
                       int k_cache_hadamard, int v_cache_hadamard);
float *model_forward(model_t *m, int token, int pos);
float *model_forward_prefill(model_t *m, const int *tokens, int n_tokens, int start_pos, volatile int *interrupt);

/* Free all resources. */
void model_free(model_t *m);

/* ---- KV cache persistence ---- */

/* Save KV cache state for positions [0, n_pos) to a file.
 * tokens: pointer to token array (optional, NULL if not saving tokens).
 * Returns 0 on success. */
int kvcache_save(const model_t *m, const char *path, int n_pos, const int *tokens);

/* Load KV cache state from a file. Returns the number of positions
 * loaded (0 on failure). *tokens_out is set to a malloc'd token array
 * (free by caller), or NULL if no tokens were saved. */
int kvcache_load(model_t *m, const char *path, int **tokens_out);

/* ---- SSM state checkpointing (Qwen3.5/3.6) ---- */

/* Returns total bytes needed for a full SSM state snapshot (all SSM layers).
 * Returns 0 if model has no SSM layers. */
size_t model_ssm_snapshot_size(const model_t *m);

/* Save current SSM state into pre-allocated buffer. Returns bytes written.
 * buf must be >= model_ssm_snapshot_size(m). */
size_t model_ssm_state_save(const model_t *m, uint8_t *buf, size_t buf_size);

/* Restore SSM state from buffer. Returns bytes read.
 * buf must contain a valid snapshot (matching model config). */
size_t model_ssm_state_restore(model_t *m, const uint8_t *buf, size_t buf_size);

/* Reset all SSM state to zero (fresh start). No-op for non-SSM models. */
void model_ssm_state_reset(model_t *m);

#endif /* MODEL_H */
