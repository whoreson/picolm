#include "model.h"
#include "model_internal.h"
#include "tensor.h"
#include "quant.h"

/* SSM verification debug: guarded by compile-time PICOLM_SSM_VERIFY define.
 * When undefined, the macro expands to 0, eliminating all debug code at compile time.
 * When defined, falls back to _SSM_DBG for runtime toggle. */
#ifdef PICOLM_SSM_VERIFY
#define _SSM_DBG _SSM_DBG
#else
#define _SSM_DBG (0)
#endif
#include <inttypes.h>

#if defined(__APPLE__) && defined(__ppc__) && defined(__ALTIVEC__)
#include <altivec.h>
#undef bool
#undef pixel
#undef vec_add
#endif

/* Mac OS X uses MAP_ANON instead of MAP_ANONYMOUS */
#if defined(__APPLE__) && !defined(MAP_ANONYMOUS)
#define MAP_ANONYMOUS MAP_ANON
#endif

/* aligned_alloc is C11; provide fallback for C99 / old macOS */
#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L
/* valloc returns page-aligned memory (4096), which satisfies any
 * alignment up to a page. All current callers ask for 64-byte alignment.
 * valloc'd memory is free()-able like any other malloc'd memory. */
#define aligned_alloc(a, s) valloc((s) + (a) - 1)
#endif

#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef PICOLM_GPU
#include "backend_gpu.h"
#ifdef PICOLM_GPU
extern int cudaProfilerStart(void);
extern int cudaProfilerStop(void);
#endif
#endif
#ifdef PICOLM_VIZ
#include "viz.h"
#endif
#include <errno.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef PICOLM_NEON
#include <arm_neon.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <unistd.h>
#include <malloc.h>
#define SECURITY_WIN32
#include <security.h>
#include <accctrl.h>
#include <aclapi.h>
#else
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <pthread.h>
#endif

/* Forward declarations from backend_gpu.cu (avoids nvcc extern "C" overload
 * confusion when backend_gpu.h is included from the .cu compilation unit). */
extern int picolm_gpu_pipeline_batch_alloc(int dim, int q_dim, int kv_dim, int ffn_hidden,
                                            int xb_stride, int max_seq_len, int device);
extern int picolm_gpu_prealloc_q8(size_t max_xq_bytes, size_t max_xd_bytes, int device);
extern int picolm_gpu_ssm_head_permute_batch_dev(float *dst, const float *src,
                                                  const int *head_map,
                                                  int head_dim, int n_heads,
                                                  int n_tokens, int src_stride,
                                                  int dst_stride, int device);
#ifdef PICOLM_SSM_VERIFY
extern void picolm_gpu_debug_tensor(const char *name, void *tensor, int device, int layer, int dump_weights);
#endif

#if defined(__APPLE__) && defined(__ppc__) && defined(__ALTIVEC__)
/* Swap uint16 values in-place using Altivec vec_perm.
 * Uses static 64-byte aligned buffer to avoid alignment issues. */
static char _f16swap_buf[64] __attribute__((aligned(64)));
static unsigned char _f16swap_mask[16] __attribute__((aligned(64))) =
    {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14};
static vector unsigned char _f16swap_vmask;

static void swap_f16_block(uint16_t *dst, size_t n) {
    /* Lazily initialize the permute mask from the static array */
    static int initialized = 0;
    if (!initialized) {
        _f16swap_vmask = vec_ld(0, (const unsigned char *)_f16swap_mask);
        initialized = 1;
    }
    vector unsigned char vm = _f16swap_vmask;
    size_t i;
    for (i = 0; i + 8 <= n; i += 8) {
        memcpy(_f16swap_buf, dst + i, 16);
        vector unsigned char v = (vector unsigned char)vec_ld(0, (const unsigned char*)_f16swap_buf);
        vector unsigned char s = vec_perm(v, v, vm);
        vec_st(s, 0, (unsigned char*)_f16swap_buf);
        memcpy(dst + i, _f16swap_buf, 16);
    }
    for (; i < n; i++) dst[i] = GGUF_LE16(dst[i]);
}
#else
/* swap_f16_block not needed on LE; parse_gguf swap loop is PPC-only */
#endif

/* Runtime prefault: when set, prepare_mmap() touches every 4KB page of the
 * mmap region to bring the model into the page cache before inference. */
static int g_do_prefault = 0;

/* Benchmark layer callback */
static bench_layer_cb_t g_bench_cb = NULL;
static void *g_bench_user_data = NULL;

void model_set_bench_callback(bench_layer_cb_t cb, void *user_data) {
    g_bench_cb = cb;
    g_bench_user_data = user_data;
}

/* Get per-layer weight size in bytes for bandwidth calculation */
size_t layer_weight_size(model_t *m, int l) {
    model_weights_t *w = &m->weights;
    layer_weights_t *lw = &w->layers[l];
    size_t sz = 0;
    int dim = m->config.n_embd;
    int n_ffn = m->config.n_ffn;
    int head_dim = m->config.head_dim;
    int n_heads = m->config.n_heads;
    int n_kv_heads = m->config.n_kv_heads;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Attention or SSM attention path */
    if (lw->attn_qkv) {
        /* SSM layer: attn_qkv [dim, conv_dim], attn_gate_ssm [dim, ssm_d_inner],
         * ssm_out [ssm_d_inner, dim], ssm_conv1d [d_conv, conv_dim],
         * ssm_alpha [dim, dt_rank], ssm_beta [dim, dt_rank],
         * ssm_a [dt_rank], ssm_dt [dt_rank], ssm_norm [head_v_dim] */
        int conv_dim = 2 * m->config.ssm_d_state * m->config.ssm_n_group + m->config.ssm_d_inner;
        int ssm_d_inner = m->config.ssm_d_inner;
        int dt_rank = m->config.ssm_dt_rank;
        int head_v_dim = ssm_d_inner / dt_rank;

        sz += gguf_type_row_size(lw->type_attn_qkv, conv_dim) * dim;
        sz += gguf_type_row_size(lw->type_attn_gate_ssm, ssm_d_inner) * dim;
        sz += gguf_type_row_size(lw->type_ssm_out, dim) * ssm_d_inner;
        sz += gguf_type_row_size(lw->type_ssm_conv1d, conv_dim) * m->config.ssm_d_conv;
        sz += gguf_type_row_size(lw->type_ssm_alpha, dt_rank) * dim;
        sz += gguf_type_row_size(lw->type_ssm_beta, dt_rank) * dim;
        sz += gguf_type_row_size(lw->type_ssm_a, 1) * dt_rank;
        sz += gguf_type_row_size(lw->type_ssm_dt, 1) * dt_rank;
        sz += gguf_type_row_size(GGUF_TYPE_F32, 1) * head_v_dim; /* ssm_norm is always F32 */
    } else if (lw->attn_q) {
        /* Standard attention: attn_q [dim, q_full_dim], attn_k [dim, kv_dim],
         * attn_v [dim, kv_dim], attn_output [q_dim, dim] */
        int q_full_dim = (m->config.has_ssm && lw->is_attn_layer) ? (q_dim * 2) : q_dim;
        sz += gguf_type_row_size(lw->type_attn_q, q_full_dim) * dim;
        sz += gguf_type_row_size(lw->type_attn_k, kv_dim) * dim;
        sz += gguf_type_row_size(lw->type_attn_v, kv_dim) * dim;
        sz += gguf_type_row_size(lw->type_attn_output, dim) * q_dim;
    }

    /* FFN (shared by both SSM and attention layers) */
    if (lw->ffn_gate) {
        sz += gguf_type_row_size(lw->type_ffn_gate, n_ffn) * dim;
    }
    if (lw->ffn_up) {
        sz += gguf_type_row_size(lw->type_ffn_up, n_ffn) * dim;
    }
    if (lw->ffn_down) {
        sz += gguf_type_row_size(lw->type_ffn_down, dim) * n_ffn;
    }
    /* MoE tensors: 3D [dim0, dim1, n_expert], stored row-major (dim0 fastest) */
    if (lw->ffn_gate_exps) {
        int n_ff_exp = m->config.n_ff_exp;
        int n_expert = m->config.n_expert;
        /* [dim, n_ff_exp, n_expert]: n_expert * n_ff_exp rows of dim elements */
        sz += (size_t)n_expert * n_ff_exp * gguf_type_row_size(lw->type_ffn_gate_exps, dim);
    }
    if (lw->ffn_up_exps) {
        int n_ff_exp = m->config.n_ff_exp;
        int n_expert = m->config.n_expert;
        sz += (size_t)n_expert * n_ff_exp * gguf_type_row_size(lw->type_ffn_up_exps, dim);
    }
    if (lw->ffn_down_exps) {
        int n_ff_exp = m->config.n_ff_exp;
        int n_expert = m->config.n_expert;
        /* [n_ff_exp, dim, n_expert]: n_expert * dim rows of n_ff_exp elements */
        sz += (size_t)n_expert * dim * gguf_type_row_size(lw->type_ffn_down_exps, n_ff_exp);
    }
    if (lw->ffn_gate_inp) {
        int n_expert = m->config.n_expert;
        sz += gguf_type_row_size(lw->type_ffn_gate_inp, n_expert) * dim;
    }
    if (lw->ffn_gate_inp_shexp) {
        /* [dim] — quantization type varies (F32, Q4_K, etc.) */
        sz += gguf_type_row_size(lw->type_ffn_gate_inp_shexp, dim);
    }
    if (lw->ffn_gate_shexp) {
        int n_ff_shexp = m->config.n_ff_shexp;
        sz += gguf_type_row_size(lw->type_ffn_gate_shexp, n_ff_shexp) * dim;
    }
    if (lw->ffn_up_shexp) {
        int n_ff_shexp = m->config.n_ff_shexp;
        sz += gguf_type_row_size(lw->type_ffn_up_shexp, n_ff_shexp) * dim;
    }
    if (lw->ffn_down_shexp) {
        int n_ff_shexp = m->config.n_ff_shexp;
        sz += gguf_type_row_size(lw->type_ffn_down_shexp, dim) * n_ff_shexp;
    }
    return sz;
}

/* Invoke the benchmark callback for a completed layer */
static void bench_emit(int l, int is_prefill, double elapsed_ms, long minflt, long majflt) {
    if (!g_bench_cb) return;
    g_bench_cb(l, is_prefill, elapsed_ms, minflt, majflt, g_bench_user_data);
}

/* Inline-like macro for benchmarking a layer: captures rusage before, time before,
 * caller does work, then bench_emit_after does the rest. */
double get_time_ms(void);
#ifdef _WIN32
#define BENCH_LAYER_START() double _bench_t0 = get_time_ms()
#define BENCH_LAYER_END(_l, _pref) bench_emit(_l, _pref, get_time_ms() - (_bench_t0), 0, 0)
#else
#define BENCH_LAYER_START() double _bench_t0 = get_time_ms(); struct rusage _bench_ru0; getrusage(RUSAGE_SELF, &_bench_ru0)
#define BENCH_LAYER_END(_l, _pref) do { struct rusage _bench_ru1; getrusage(RUSAGE_SELF, &_bench_ru1); \
    bench_emit(_l, _pref, get_time_ms() - (_bench_t0), \
        (long)_bench_ru1.ru_minflt - (long)_bench_ru0.ru_minflt, \
        (long)_bench_ru1.ru_majflt - (long)_bench_ru0.ru_majflt); } while(0)
#endif

void model_set_prefault(int v) {
    g_do_prefault = v;
}

/* Check if AVX2 is available at runtime (for repack decision) */
static int cpu_has_avx2(void) {
#ifdef PICOLM_AVX2
    return 1;
#else
    return 0;
#endif
}

/* AVX2/AVX horizontal sum: reduce 8 FP32 lanes to one float */
#if defined(PICOLM_AVX2) || defined(PICOLM_AVX)
#endif /* PICOLM_AVX2 || PICOLM_AVX */

static void repack_model_weights_q4_0x8(model_t *m);

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef HAVE_WINDOWS_H
#include <windows.h>
#endif
#endif
/* Prefault: touch one byte per 4KB page of the mmap region to bring the
 * entire model into the page cache before inference begins.
 *
 * On Linux this triggers synchronous readahead through the page cache,
 * eliminating page-fault latency during prefill. A 10 GB model (~2.5M
 * pages) typically faults in within 1-2 seconds on an SSD.
 *
 * Uses volatile reads to prevent the compiler from optimizing the loop
 * away. The accesses are spread across the entire mapping so the OS
 * issues readahead for all pages, not just the first few. */
void _do_prefault(const void *addr, size_t size) {
    size_t pages = (size + 4095) / 4096;
    const volatile char *p = (const volatile char *)addr;
    for (size_t off = 0; off < size; off += 4096)
        (void)p[off];
    fprintf(stderr, "Prefaulted %zu pages (%.1f MB)\n", pages, (double)size / (1024.0 * 1024.0));
}

void prepare_mmap(const void *addr, size_t size) {
    if (!g_do_prefault) return;
    _do_prefault(addr, size);
}

void model_prefault(model_t *m) {
    int n = m->n_splits > 0 ? m->n_splits : 1;
    for (int i = 0; i < n; i++) {
        if (m->splits[i].mmap_addr) {
            _do_prefault(m->splits[i].mmap_addr, m->splits[i].mmap_size);
        }
    }
}


/* ---- Pre-compute RoPE cos/sin lookup tables ---- */

static void init_rope_tables(run_state_t *s, const model_config_t *c) {
    int rope_dim = (c->rope_dim > 0) ? c->rope_dim : c->head_dim;
    int half_dim = rope_dim / 2;
    for (int pos = 0; pos < c->max_seq_len; pos++) {
        float *cos_row = s->rope_cos + (size_t)pos * half_dim;
        float *sin_row = s->rope_sin + (size_t)pos * half_dim;
        for (int i = 0; i < half_dim; i++) {
            float theta = (float)pos / powf(c->rope_freq_base, (float)(2 * i) / (float)rope_dim);
            cos_row[i] = cosf(theta);
            sin_row[i] = sinf(theta);
        }
    }
}

/* ---- Buffer allocation ---- */

/* Compute row size in bytes for a given KV cache type and number of elements */
/* kv_row_size: bytes for a GQA row (n_kv_heads * head_dim elements)
 * For FP16: n * sizeof(uint16_t)
 * For quantized types: quantized block layout for the full GQA row */
static size_t kv_row_size_gqa(kv_cache_type_t kv_type, int n_elements) {
    switch (kv_type) {
        case KV_CACHE_F16:  return (size_t)n_elements * sizeof(uint16_t);
        case KV_CACHE_Q8_0: return ((size_t)(n_elements / 32)) * sizeof(block_q8_0);
        case KV_CACHE_Q4_0: return ((size_t)(n_elements / 32)) * sizeof(block_q4_0);
        case KV_CACHE_TQ3:  return ((size_t)(n_elements / TQ3_BLOCK_SIZE)) * sizeof(block_tq3);
        case KV_CACHE_TQ4:  return ((size_t)(n_elements / TQ4_BLOCK_SIZE)) * sizeof(block_tq4);
    }
    return 0;
}

/* kv_head_stride: byte offset between consecutive heads within a GQA row
 * For FP16: head_dim * sizeof(uint16_t)
 * For quantized: the quantized size of head_dim elements
 * (for head_dim=64: Q8_0=68, Q4_0=44, TQ3=28, TQ4=36) */
static size_t kv_head_stride(kv_cache_type_t kv_type, int head_dim) {
    switch (kv_type) {
        case KV_CACHE_F16:  return (size_t)head_dim * sizeof(uint16_t);
        case KV_CACHE_Q8_0: return ((size_t)(head_dim / 32)) * sizeof(block_q8_0);
        case KV_CACHE_Q4_0: return ((size_t)(head_dim / 32)) * sizeof(block_q4_0);
        case KV_CACHE_TQ3:  return ((size_t)(head_dim / TQ3_BLOCK_SIZE)) * sizeof(block_tq3);
        case KV_CACHE_TQ4:  return ((size_t)(head_dim / TQ4_BLOCK_SIZE)) * sizeof(block_tq4);
    }
    return 0;
}

int allocate_run_state(model_t *m, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v,
                       int k_cache_hadamard, int v_cache_hadamard, int n_threads) {
    model_config_t *c = &m->config;
    run_state_t *s = &m->state;

    int rope_dim = (c->rope_dim > 0) ? c->rope_dim : c->head_dim;
    int half_dim = rope_dim / 2;
    int q_dim = c->n_heads * c->head_dim;
    /* Qwen3.5 full attention: Q+gate joint = 2x q_dim */
    int q_full_dim = c->has_ssm ? (q_dim * 2) : q_dim;
    /* SSM conv_dim may be larger */
    int ssm_conv_dim = c->has_ssm ? (2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner) : 0;
    int max_proj_dim = q_full_dim;
    if (ssm_conv_dim > max_proj_dim) max_proj_dim = ssm_conv_dim;
    int max_dim = (max_proj_dim > c->n_embd) ? max_proj_dim : c->n_embd;

    /* Calculate sizes for float buffers */
    size_t sz_x      = (size_t)c->n_embd * sizeof(float);
    size_t sz_xb     = (size_t)max_proj_dim * sizeof(float);
    size_t sz_xb2    = (size_t)max_proj_dim * sizeof(float);
    size_t sz_q      = (size_t)max_proj_dim * sizeof(float);
    /* att buffer removed (flash attention) */
    size_t sz_hb     = (size_t)c->n_ffn * sizeof(float);
    size_t sz_hb2    = (size_t)c->n_ffn * sizeof(float);
    size_t sz_logits = (size_t)c->vocab_size * sizeof(float);

    int scratch_dim = max_dim;
    if (c->n_ffn > scratch_dim) scratch_dim = c->n_ffn;
    if (c->vocab_size > scratch_dim) scratch_dim = c->vocab_size;
    size_t sz_scratch = (size_t)scratch_dim * sizeof(float);

    /* RoPE tables: cos and sin for each (position, dim_pair) */
    size_t sz_rope = (size_t)c->max_seq_len * half_dim * sizeof(float) * 2;

    /* Norm weights: (n_layers * 2 + 1) * n_embd + n_layers * head_dim * 2 (QK-norm)
     * + n_embd for output_norm_w (the carve skips n_norm + n_embd) */
    size_t n_norm = (size_t)(c->n_layers * 2 + 1) * c->n_embd
                  + (size_t)c->n_layers * c->head_dim * 2 + c->n_embd;
    size_t sz_norm = n_norm * sizeof(float);

    /* SSM state buffers (Qwen3.5) */
    size_t sz_ssm_conv = 0, sz_ssm_state = 0, sz_ssm_small = 0;
    if (c->has_ssm) {
        int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
        int head_v_dim = c->ssm_d_inner / c->ssm_dt_rank;
        /* Allocate conservatively: all n_layers * (conv_state + small arrays + state)
         * Only SSM layers actually use the buffers, but this avoids under-allocation. */
        int ssm_per_layer = (c->ssm_d_conv - 1) * conv_dim  /* conv_state */
                          + c->ssm_dt_rank * 2              /* a + dt */
                          + head_v_dim                      /* norm */
                          + c->ssm_d_conv * conv_dim        /* conv1d weights */
                          + c->ssm_d_state * c->ssm_d_inner /* recurrent state */;
        sz_ssm_conv = sz_ssm_state = sz_ssm_small = 0;
        sz_ssm_small = (size_t)c->n_layers * ssm_per_layer * sizeof(float);
    }
    /* SSM scratch buffer (shared across SSM layers) */
    size_t sz_ssm_tmp = 0;
    if (c->has_ssm) {
        int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
        sz_ssm_tmp = ((size_t)conv_dim * 3 +
                      (size_t)c->ssm_d_state * c->ssm_n_group * 2 +
                      (size_t)c->ssm_d_inner * 3 +
                      (size_t)c->ssm_dt_rank * 4) * sizeof(float);
    }
    /* Gemma-3n buffers */
    size_t sz_gemma3n = 0;
    if (c->is_gemma3n) {
        int n_altup = c->n_altup;
        int n_embd = c->n_embd;
        int n_embd_altup = c->n_embd_altup;
        int laurel_rank = c->laurel_rank;
        sz_gemma3n += (size_t)n_altup * n_embd * sizeof(float);         /* altup_state (single-token) */
        sz_gemma3n += (size_t)n_altup * n_embd * sizeof(float);         /* predictions buffer */
        sz_gemma3n += (size_t)n_embd_altup * c->n_layers * sizeof(float); /* per_layer_inp */
        sz_gemma3n += (size_t)n_embd_altup * sizeof(float);              /* inp_gate_out */
        sz_gemma3n += (size_t)n_embd * sizeof(float);                    /* first_pred */
        sz_gemma3n += (size_t)n_embd * sizeof(float);                    /* laurel_out */
        sz_gemma3n += (size_t)n_altup * sizeof(float);                  /* router_out */
        /* Extra norm weights per layer */
        sz_gemma3n += (size_t)c->n_layers * n_embd * 5 * sizeof(float); /* attn_post, post_ffw, per_layer_post, laurel_post, router_norm */
        sz_gemma3n += (size_t)c->n_layers * n_embd * sizeof(float);     /* altup_correct_scale */
        /* Pre-dequantized small arrays per layer */
        sz_gemma3n += (size_t)c->n_layers * n_altup * n_altup * sizeof(float);        /* correct_coef [n_altup x n_altup] */
        sz_gemma3n += (size_t)c->n_layers * n_altup * n_altup * n_altup * sizeof(float); /* predict_coef [n_altup x n_altup*n_altup] */
        sz_gemma3n += (size_t)c->n_layers * (n_embd * laurel_rank + laurel_rank * n_embd) * sizeof(float); /* laurel_l + laurel_r */
        sz_gemma3n += (size_t)c->n_layers * n_embd * n_altup * sizeof(float); /* altup_router dequantized */
    }

    /* MoE buffers */
    size_t sz_moe = 0;
    if (c->has_moe) {
        sz_moe += (size_t)c->n_expert * c->max_seq_len * sizeof(float);       /* expert_logits (batched) */
        sz_moe += (size_t)c->n_expert_used * sizeof(int);    /* expert_ids */
        sz_moe += (size_t)c->n_expert_used * sizeof(float);  /* expert_weights */
        sz_moe += (size_t)c->n_embd * sizeof(float);         /* moe_out */
        sz_moe += (size_t)c->n_ff_exp * sizeof(float) * 2;   /* hb_exp, hb2_exp */
        /* Pre-allocated quantization buffers — avoid per-expert malloc */
        sz_moe += (size_t)c->n_ff_exp * (size_t)c->n_expert_used * sizeof(float); /* gate_batch */
        sz_moe += (size_t)c->n_ff_exp * (size_t)c->n_expert_used * sizeof(float); /* up_batch */
        sz_moe += gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_ff_exp); /* down_qx */
        sz_moe += (c->n_ff_exp / 32) * sizeof(float);         /* down_qx_d */
        sz_moe += gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_embd); /* shared_qx */
        sz_moe += (c->n_embd / 32) * sizeof(float);           /* shared_qx_d */
        sz_moe += gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_ff_shexp); /* shared_down_qx */
        sz_moe += (c->n_ff_shexp / 32) * sizeof(float);       /* shared_down_qx_d */

        /* Per-thread scratch for parallel MoE dispatch.
         * Each thread needs: gate_buf[n_ff_exp] + up_buf[n_ff_exp] + xb2_buf[n_embd]
         * + down_qx (Q8_0 row of n_ff_exp) + down_qx_d (n_ff_exp/32) + acc_buf[n_embd] */
        {
            size_t per_thread = 0;
            per_thread += (size_t)c->n_ff_exp * sizeof(float) * 2;   /* gate + up */
            per_thread += (size_t)c->n_embd * sizeof(float) * 2;     /* xb2 + acc */
            per_thread += gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_ff_exp); /* down_qx */
            per_thread += (c->n_ff_exp / 32) * sizeof(float);        /* down_qx_d */
            per_thread = (per_thread + 31) / 32 * 32;                /* align to 32 bytes */
            sz_moe += per_thread * (size_t)n_threads;
        }

        /* Per-expert scratch: gate_buf, up_buf, down_qx, down_qx_d.
         * gate_batch_exp and up_batch_exp are [n_ff_exp * n_ctx] where
         * n_ctx is the actual context size used for this batch.
         * We use a moderate size: n_ff_exp * 128 (covers typical batches). */
        {
            size_t batch_max = c->max_seq_len;
            if (batch_max > 256) batch_max = 256;
            sz_moe += (size_t)c->n_ff_exp * batch_max * sizeof(float) * 2; /* gate_batch_exp + up_batch_exp */
        }
        sz_moe += gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_ff_exp);   /* down_qx_exp */
        sz_moe += (c->n_ff_exp / 32) * sizeof(float);                /* down_qx_d_exp */

        /* Pre-allocated qx_all buffer for batched MoE.
         * Size: max_seq_len * (q8_row_blocks + nb_deltas) floats. */
        {
            size_t q8_rb = c->n_embd / 32;
            size_t q8_data_off = (q8_rb * sizeof(block_q8_0) + sizeof(float) - 1) / sizeof(float);
            size_t q8_buf_per_token = q8_data_off + q8_rb;
            sz_moe += (size_t)c->max_seq_len * q8_buf_per_token;
        }

        /* Shared expert batch buffers (gate+up for all tokens) */
        {
            size_t batch_max = c->max_seq_len;
            if (batch_max > 256) batch_max = 256;
            sz_moe += (size_t)c->n_ff_shexp * batch_max * sizeof(float) * 2; /* sh_gate + sh_up */
        }

        /* Per-token accumulator for MoE: [n_embd × n_expert_used × batch_max] */
        {
            size_t batch_max = c->max_seq_len;
            if (batch_max > 256) batch_max = 256;
            sz_moe += (size_t)c->n_embd * c->n_expert_used * batch_max * sizeof(float); /* moe_acc_batch */
        }

        /* mm_id buffers: allocated on-demand in moe_forward_batch (not at load time).
         * Sizes: [n_tokens * n_used * n_ff] for gate+up, [n_tokens * n_used * n_embd] for down.
         * Only needed during prefill (multi-token path); generation uses single-token moe_forward. */
        /* Scratch Q8_0 for mm_id down: n_ff_exp / 32 blocks + deltas */
        {
            size_t dnb = c->n_ff_exp / 32;
            sz_moe += gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_ff_exp) + dnb * sizeof(float);
        }

        /* Per-expert down quantization buffers for batched mm_id.
         * Size: n_expert_used tokens × q8_buf_per_token floats per entry */
        {
            size_t q8_per_token = gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_ff_exp) / sizeof(float)
                                + c->n_ff_exp / 32; /* blocks + deltas */
            sz_moe += c->n_expert_used * q8_per_token * sizeof(float) * 2; /* qx + qx_d, 2 = aligned padding */
        }

        /* Precomputed routing map: expert_assignments + expert_counts */
        sz_moe += (size_t)c->n_expert * c->max_seq_len * sizeof(int);
        sz_moe += (size_t)c->n_expert * sizeof(int);

        /* Per-thread mm_down_qx_all: n_threads × 256 × down_q8_per_token */
        {
            size_t down_q8_rb = c->n_ff_exp / 32;
            size_t down_q8_data_off = (down_q8_rb * sizeof(block_q8_0) + sizeof(float) - 1) / sizeof(float);
            size_t down_q8_per_token = down_q8_data_off + down_q8_rb;
            sz_moe += (size_t)n_threads * 256 * down_q8_per_token; /* n_threads × 256 max tokens per expert */
        }
    }

    size_t total = sz_x + sz_xb + sz_xb2 + sz_q +
                   sz_hb + sz_hb2 + sz_logits +
                   sz_scratch + sz_rope + sz_norm +
                   sz_ssm_conv + sz_ssm_state + sz_ssm_small + sz_ssm_tmp +
                   sz_gemma3n + sz_moe;

    /* Quantized KV cache: separate allocation (only for attention layers)
     * Full-row GQA layout: [layer][pos] -> GQA row of n_kv_heads*head_dim */

    /* Quantized KV cache requires head_dim to be a multiple of the quantization
     * block size (32 for Q8_0/Q4_0). This is because:
     *
     * 1. The K/V cache stores quantized blocks per-GQA-row, but attention reads
     *    access individual heads at head-stride offsets within that row.
     * 2. Each quantization block (32 elements) carries its own scale factor.
     *    If head_dim is not a multiple of 32, block boundaries fall in the
     *    middle of a head, so a single block's scale covers elements from two
     *    different heads. The per-head read stride cannot split blocks, so it
     *    either truncates the head (reading fewer than head_dim elements) or
     *    reads into the next head's data.
     *
     * Example: head_dim=48, Q8_0. Block 0 covers elements 0-31, Block 1 covers
     * elements 32-63. Head 0 owns elements 0-47, but Block 0 only provides
     * elements 0-31 with one scale, and Block 1 mixes elements 32-47 (head 0)
     * with 48-63 (head 1) under a shared scale. Reading head 0 at stride=34
     * bytes (1 block) yields only 32 of the 48 elements.
     *
     * llama.cpp enforces the same constraint in its flash attention path:
     *   "K cache type q8_0 with block size 32 does not divide n_embd_head_k=48"
     *
     * We check here and fall back to FP16 for any incompatible cache type.
     * This affects head_dim values like 48, 80, 112, ... (anything % 32 != 0). */
    {
        int blck_size = 32; /* Q8_0 and Q4_0 block size */
        if (kv_type_k != KV_CACHE_F16 && c->head_dim % blck_size != 0) {
            fprintf(stderr, "KV cache: head_dim=%d is not divisible by Q8_0/Q4_0 block size %d; "
                    "falling back K cache to f16\n", c->head_dim, blck_size);
            kv_type_k = KV_CACHE_F16;
        }
        if (kv_type_v != KV_CACHE_F16 && c->head_dim % blck_size != 0) {
            fprintf(stderr, "KV cache: head_dim=%d is not divisible by Q8_0/Q4_0 block size %d; "
                    "falling back V cache to f16\n", c->head_dim, blck_size);
            kv_type_v = KV_CACHE_F16;
        }
    }

    size_t sz_k_row = kv_row_size_gqa(kv_type_k, c->n_kv_heads * c->head_dim);
    size_t sz_v_row = kv_row_size_gqa(kv_type_v, c->n_kv_heads * c->head_dim);
    size_t sz_k_head = kv_head_stride(kv_type_k, c->head_dim);
    size_t sz_v_head = kv_head_stride(kv_type_v, c->head_dim);
    int attn_layer_count = 0;
    if (c->has_ssm) {
        for (int i = 0; i < c->n_layers; i++) {
            if (m->weights.layers[i].is_attn_layer) attn_layer_count++;
        }
    }
    int kv_layers = attn_layer_count > 0 ? attn_layer_count : c->n_layers;
    /* Gemma-3n: only first n_layer_kv_from_start layers have KV cache */
    if (c->is_gemma3n && c->n_layer_kv_from_start > 0 && c->n_layer_kv_from_start < kv_layers) {
        kv_layers = c->n_layer_kv_from_start;
    }
    /* GQA layout: each position stores one GQA row, not n_kv_heads rows */
    size_t sz_kv = (size_t)kv_layers * c->max_seq_len * (sz_k_row + sz_v_row);

    const char *kv_name_k = "f16";
    const char *kv_name_v = "f16";
    if (kv_type_k == KV_CACHE_Q8_0) kv_name_k = "q8_0";
    else if (kv_type_k == KV_CACHE_Q4_0) kv_name_k = "q4_0";
    else if (kv_type_k == KV_CACHE_TQ3) kv_name_k = "tq3";
    else if (kv_type_k == KV_CACHE_TQ4) kv_name_k = "tq4";
    if (kv_type_v == KV_CACHE_Q8_0) kv_name_v = "q8_0";
    else if (kv_type_v == KV_CACHE_Q4_0) kv_name_v = "q4_0";
    else if (kv_type_v == KV_CACHE_TQ3) kv_name_v = "tq3";
    else if (kv_type_v == KV_CACHE_TQ4) kv_name_v = "tq4";
    fprintf(stderr, "Allocating %.2f MB for runtime state (+ %.2f MB KV cache [%s/%s])\n",
            (double)total / (1024.0 * 1024.0),
            (double)sz_kv / (1024.0 * 1024.0), kv_name_k, kv_name_v);

    /* Use mmap on POSIX for large allocations to avoid address space
     * collisions with the GGUF file mmap. On Windows, use calloc. */
    {
        size_t total_alloc = total + sz_kv;
        void *base = NULL;
#ifdef _WIN32
        base = calloc(1, total_alloc);
        if (!base) {
            fprintf(stderr, "OOM: calloc failed for %zu bytes\n", total_alloc);
            return -1;
        }
#else
        if (m->mmap_addr) {
            /* Try to allocate just below the GGUF mmap */
            uintptr_t hint = (uintptr_t)m->mmap_addr - (total_alloc + 4096);
            base = mmap((void *)hint, total_alloc, PROT_READ|PROT_WRITE,
                        MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
            if (base == MAP_FAILED || base != (void *)hint) {
                /* mmap at hint failed or moved us; try anywhere */
                base = mmap(NULL, total_alloc, PROT_READ|PROT_WRITE,
                            MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
            }
        } else {
            base = mmap(NULL, total_alloc, PROT_READ|PROT_WRITE,
                        MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        }
        if (base == MAP_FAILED) {
            fprintf(stderr, "OOM: mmap failed for %zu bytes: %s\n", total_alloc, strerror(errno));
            return -1;
        }
#endif

        s->mem_block = base;
        s->mem_size = total + sz_kv;

        /* KV cache follows mem_block contiguously */
        s->kv_block = (char *)base + total;
        s->kv_size = sz_kv;
    }

    /* Populate KV layer mapping */
    {
        int kv_ord = 0;
        for (int l = 0; l < c->n_layers; l++) {
            if (c->has_ssm && !m->weights.layers[l].is_attn_layer) {
                s->kv_layer_map[l] = 0;
                s->kv_layer_ordinal[l] = -1;
            } else {
                s->kv_layer_map[l] = 1;
                s->kv_layer_ordinal[l] = kv_ord++;
            }
        }
        s->kv_layer_count = kv_ord;
    }

    s->kv_type_k = kv_type_k;
    s->kv_type_v = kv_type_v;
    s->kv_row_size_k = sz_k_row;
    s->kv_row_size_v = sz_v_row;
    s->kv_head_stride_k = sz_k_head;
    s->kv_head_stride_v = sz_v_head;

    /* Hadamard rotation for KV cache (disabled by default) */
    s->kv_hadamard_k = 0;
    s->kv_hadamard_v = 0;
    s->kv_hadamard_size = 0;
    if (k_cache_hadamard && kv_type_k != KV_CACHE_F16) {
        s->kv_hadamard_k = 1;
    }
    if (v_cache_hadamard && kv_type_v != KV_CACHE_F16) {
        s->kv_hadamard_v = 1;
    }
    if (s->kv_hadamard_k || s->kv_hadamard_v) {
        /* Determine block size: largest power of 2 that divides head_dim */
        int nrot = c->head_dim & ~(c->head_dim - 1); /* largest power-of-2 factor */
        if (nrot < 2) {
            fprintf(stderr, "KV Hadamard: head_dim=%d has no power-of-2 divisor >= 2, "
                    "disabling Hadamard rotation\n", c->head_dim);
            s->kv_hadamard_k = 0;
            s->kv_hadamard_v = 0;
        } else {
            s->kv_hadamard_size = nrot;
            fprintf(stderr, "KV Hadamard rotation enabled: K=%d V=%d (nrot=%d)\n",
                    s->kv_hadamard_k, s->kv_hadamard_v, nrot);
        }
    }

    /* Carve float pointers */
    float *p = (float *)s->mem_block;
    s->x      = p; p += c->n_embd;
    s->xb     = p; p += max_proj_dim;
    s->xb2    = p; p += max_proj_dim;
    s->q      = p; p += max_proj_dim;
    s->hb     = p; p += c->n_ffn;
    s->hb2    = p; p += c->n_ffn;
    s->logits = p; p += c->vocab_size;
    s->dequant_scratch = p; p += scratch_dim;

    /* RoPE tables */
    s->rope_cos = p; p += (size_t)c->max_seq_len * half_dim;
    s->rope_sin = p; p += (size_t)c->max_seq_len * half_dim;

    /* Norm weights */
    s->norm_weights = p;
    p += n_norm; /* skip norm weights area (includes output_norm in n_norm count) */

    /* KV cache pointers: K layers first, then V layers (all layers, SSM layers just don't use their slots)
     * GQA layout: [layer][pos] * kv_row_size, with head offset = h * kv_head_stride */
    size_t layer_stride_k = (size_t)c->max_seq_len * sz_k_row;
    uint8_t *kb = (uint8_t *)s->kv_block;
    s->key_cache = kb;
    s->val_cache = kb + (size_t)kv_layers * layer_stride_k;

    /* SSM state buffers (Qwen3.5) */
    if (c->has_ssm) {
        int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
        int head_v_dim = c->ssm_d_inner / c->ssm_dt_rank;
        float *ssm_p = p;
        for (int l = 0; l < c->n_layers; l++) {
            if (m->weights.layers[l].is_attn_layer) {
                s->ssm_conv_state[l] = NULL;
                s->ssm_state[l] = NULL;
                s->ssm_a_w[l] = NULL;
                s->ssm_dt_w[l] = NULL;
                s->ssm_norm_w[l] = NULL;
                s->ssm_conv1d_w[l] = NULL;
            } else {
                s->ssm_conv_state[l] = ssm_p; ssm_p += (c->ssm_d_conv - 1) * conv_dim;
                s->ssm_state[l] = NULL; /* allocated later */
                s->ssm_a_w[l] = ssm_p; ssm_p += c->ssm_dt_rank;
                s->ssm_dt_w[l] = ssm_p; ssm_p += c->ssm_dt_rank;
                s->ssm_norm_w[l] = ssm_p; ssm_p += head_v_dim;
                s->ssm_conv1d_w[l] = ssm_p; ssm_p += c->ssm_d_conv * conv_dim;
            }
        }
        /* Note: allocation accounts for all layers conservatively.
         * Only SSM layers actually use the state buffers. */
        for (int l = 0; l < c->n_layers; l++) {
            if (!m->weights.layers[l].is_attn_layer) {
                s->ssm_state[l] = ssm_p;
                memset(ssm_p, 0, c->ssm_d_state * c->ssm_d_inner * sizeof(float));
                ssm_p += c->ssm_d_state * c->ssm_d_inner;
            }
        }
        p = ssm_p;
    }
    /* SSM scratch buffer (shared across all SSM layers) */
    if (c->has_ssm) {
        int ssm_d_state = c->ssm_d_state;
        int ssm_n_group = c->ssm_n_group;
        int ssm_d_inner = c->ssm_d_inner;
        int ssm_dt_rank = c->ssm_dt_rank;
        int n_v_heads_ssm = c->ssm_dt_rank;
        /* ssm_tmp layout:
         * conv_output[conv_dim] + q_conv[qk] + k_conv[qk] + v_conv[d_inner]
         * + alpha[dt_rank] + gate[dt_rank] + beta[dt_rank] + gate_exp[dt_rank]
         * + q_rep[d_state*n_v] + k_rep[d_state*n_v]
         * + sk[d_state*n_v] + d_vals[d_state*n_v]
         * + ssm_output[d_state*n_v] + final_output[d_state*n_v (= d_inner)]
         */
        int ssm_tmp_size = ssm_conv_dim + ssm_d_state*ssm_n_group*2 + ssm_d_inner
            + ssm_dt_rank*4
            + ssm_d_state*n_v_heads_ssm*2  /* q_rep + k_rep */
            + ssm_d_state*n_v_heads_ssm*4  /* sk + d_vals + ssm_output + final_output */;
        s->ssm_tmp = p;
        p += ssm_tmp_size;
    }

    /* Gemma-3n buffer carving */
    if (c->is_gemma3n) {
        int n_altup = c->n_altup;
        int n_embd = c->n_embd;
        int n_embd_altup = c->n_embd_altup;
        int laurel_rank = c->laurel_rank;
        float *gp = p;

        s->gemma3n_altup_state = gp; gp += (size_t)n_altup * n_embd;
        s->gemma3n_predictions = gp; gp += (size_t)n_altup * n_embd;
        s->gemma3n_per_layer_inp = gp; gp += (size_t)n_embd_altup * c->n_layers;
        s->gemma3n_inp_gate_out = gp; gp += (size_t)n_embd_altup;
        s->gemma3n_first_pred = gp; gp += (size_t)n_embd;
        s->gemma3n_laurel_out = gp; gp += (size_t)n_embd;
        s->gemma3n_router_out = gp; gp += (size_t)n_altup;

        /* Extra norm weights per layer */
        for (int l = 0; l < c->n_layers; l++) {
            s->attn_post_norm_w[l] = gp; gp += n_embd;
            s->post_ffw_norm_w[l] = gp; gp += n_embd;
            s->per_layer_post_norm_w[l] = gp; gp += n_embd;
            s->laurel_post_norm_w[l] = gp; gp += n_embd;
            s->altup_router_norm_w[l] = gp; gp += n_embd;
        }
        /* altup_correct_scale per layer */
        for (int l = 0; l < c->n_layers; l++) {
            s->altup_correct_scale_w[l] = gp; gp += n_embd;
        }
        /* Pre-dequantized small arrays per layer */
        for (int l = 0; l < c->n_layers; l++) {
            s->altup_correct_coef_w[l] = gp; gp += (size_t)n_altup * n_altup;
            s->altup_predict_coef_w[l] = gp; gp += (size_t)n_altup * n_altup * n_altup;
        }
        for (int l = 0; l < c->n_layers; l++) {
            s->laurel_l_w[l] = gp; gp += (size_t)n_embd * laurel_rank;
            s->laurel_r_w[l] = gp; gp += (size_t)laurel_rank * n_embd;
            s->altup_router_w[l] = gp; gp += (size_t)n_embd * n_altup;
        }
        p = gp;
        /* Verify we didn't exceed the allocation */
        { size_t carved = (size_t)((char *)p - (char *)s->mem_block) / sizeof(float);
          (void)carved;
        }
    }

    /* MoE buffer carving */
    if (c->has_moe) {
        s->expert_logits = p; p += (size_t)c->n_expert * c->max_seq_len;
        /* Align int pointer to float boundary */
        s->expert_ids = (int *)((uint8_t *)p);
        p += (c->n_expert_used * sizeof(int) + sizeof(float) - 1) / sizeof(float);
        s->expert_weights = p; p += c->n_expert_used;
        s->moe_out = p; p += c->n_embd;
        s->hb_exp = p; p += c->n_ff_exp;
        s->hb2_exp = p; p += c->n_ff_exp;
        s->gate_batch = p; p += (size_t)c->n_ff_exp * c->n_expert_used;
        s->up_batch = p; p += (size_t)c->n_ff_exp * c->n_expert_used;
        {
            size_t q8_rb, q8_nb;
            q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_ff_exp);
            q8_nb = c->n_ff_exp / 32;
            s->down_qx = (block_q8_0 *)p; p += (q8_rb + sizeof(float) - 1) / sizeof(float);
            s->down_qx_d = p; p += q8_nb;
            q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_embd);
            q8_nb = c->n_embd / 32;
            s->shared_qx = (block_q8_0 *)p; p += (q8_rb + sizeof(float) - 1) / sizeof(float);
            s->shared_qx_d = p; p += q8_nb;
            q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_ff_shexp);
            q8_nb = c->n_ff_shexp / 32;
            s->shared_down_qx = (block_q8_0 *)p; p += (q8_rb + sizeof(float) - 1) / sizeof(float);
            s->shared_down_qx_d = p; p += q8_nb;

            /* Per-thread MoE scratch buffers */
            {
                size_t per_thread = 0;
                per_thread += (size_t)c->n_ff_exp * sizeof(float) * 2;   /* gate + up */
                per_thread += (size_t)c->n_embd * sizeof(float) * 2;     /* xb2 + acc */
                per_thread += gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_ff_exp); /* down_qx */
                per_thread += (c->n_ff_exp / 32) * sizeof(float);        /* down_qx_d */
                per_thread = (per_thread + 31) / 32 * 32;                /* align to 32 bytes */
                s->moe_thread_scratch = (void *)p;
                s->moe_thread_stride = per_thread;
                p += (per_thread / sizeof(float) + 1) * (size_t)n_threads;
            }

            /* Per-expert scratch: gate_buf, up_buf, down_qx, down_qx_d */
            {
                size_t batch_max = c->max_seq_len;
                if (batch_max > 256) batch_max = 256;
                s->gate_batch_exp = p; p += (size_t)c->n_ff_exp * batch_max;
                s->up_batch_exp = p; p += (size_t)c->n_ff_exp * batch_max;
            }
            {
                size_t q8_rb = gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_ff_exp);
                s->down_qx_exp = (block_q8_0 *)p; p += (q8_rb + sizeof(float) - 1) / sizeof(float);
                s->down_qx_d_exp = p; p += c->n_ff_exp / 32;
            }

            /* Pre-allocated qx_all buffer for batched MoE */
            {
                size_t q8_rb = c->n_embd / 32;
                size_t q8_data_off = (q8_rb * sizeof(block_q8_0) + sizeof(float) - 1) / sizeof(float);
                size_t q8_buf_per_token = q8_data_off + q8_rb;
                s->moe_qx_d_off = (int)q8_data_off;
                s->moe_q8_buf_per_token = (int)q8_buf_per_token;
                s->moe_qx_all = p;
                p += (size_t)c->max_seq_len * q8_buf_per_token;
            }

            /* Shared expert batch buffers (gate+up for all tokens) */
            {
                size_t batch_max = c->max_seq_len;
                if (batch_max > 256) batch_max = 256;
                s->sh_gate = p; p += (size_t)c->n_ff_shexp * batch_max;
                s->sh_up = p; p += (size_t)c->n_ff_shexp * batch_max;
            }

            /* mm_id scratch Q8_0 buffer (allocated at load time; gate/up/down buffers allocated on-demand) */
            {
                size_t dnb = c->n_ff_exp / 32;
                s->mm_scratch_qx = (block_q8_0 *)p;
                p = (float *)((char *)p + gguf_type_row_size(GGUF_TYPE_Q8_0, c->n_ff_exp));
                s->mm_scratch_qx_d = p; p += dnb;
            }

            /* Per-expert down quantization buffers for batched mm_id.
             * Layout: per-thread × per-token Q8_0 buffer with embedded deltas.
             * Each entry is for n_ff_exp (not n_embd) since we quantize SwiGLU output.
             * Total: n_threads × 256 (max tokens per expert) × down_q8_per_token floats. */
            {
                int n_threads_max = n_threads;
                size_t down_q8_rb = c->n_ff_exp / 32;
                size_t down_q8_data_off = (down_q8_rb * sizeof(block_q8_0) + sizeof(float) - 1) / sizeof(float);
                size_t down_q8_per_token = down_q8_data_off + down_q8_rb;
                size_t per_thread_buf = (size_t)n_threads_max * 256 * down_q8_per_token;
                s->mm_down_qx_all = (block_q8_0 *)p;
                p += per_thread_buf;
                s->mm_down_qx_d_all = NULL; /* not needed, deltas are embedded */
            }

            /* Precomputed routing map: expert_assignments[eid * n_tokens + a] = packed (t << 8 | slot)
             * expert_counts[eid] = number of assigned tokens. */
            {
                s->expert_assignments = (int *)p;
                p += (size_t)c->n_expert * c->max_seq_len * sizeof(int);
                s->expert_counts = (int *)p;
                p += (size_t)c->n_expert * sizeof(int);
            }
        }
    }

    /* Pre-dequantize norm weights */
    float *nw = s->norm_weights;
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &m->weights.layers[l];
        s->attn_norm_w[l] = nw;
        if (lw->attn_norm) {
            dequantize_row(lw->attn_norm, nw, c->n_embd, lw->type_attn_norm);
            if (m->from_safetensors) for (int _ni = 0; _ni < c->n_embd; _ni++) nw[_ni] += 1.0f;
        } else {
            for (int _ni = 0; _ni < c->n_embd; _ni++) nw[_ni] = 1.0f;
        }
        nw += c->n_embd;

        s->post_attn_norm_w[l] = nw;
        if (lw->post_attn_norm) {
            dequantize_row(lw->post_attn_norm, nw, c->n_embd, lw->type_post_attn_norm);
            if (m->from_safetensors) for (int _ni = 0; _ni < c->n_embd; _ni++) nw[_ni] += 1.0f;
        } else {
            for (int _ni = 0; _ni < c->n_embd; _ni++) nw[_ni] = 1.0f;
        }
        nw += c->n_embd;

        /* Qwen3 QK-norm weights (per-head, if present) */
        s->attn_q_norm_w[l] = nw;
        if (lw->attn_q_norm) {
            dequantize_row(lw->attn_q_norm, nw, c->head_dim,
                           lw->type_attn_q_norm);
            if (m->from_safetensors) for (int _ni = 0; _ni < c->head_dim; _ni++) nw[_ni] += 1.0f;
        } else {
            for (int _ni = 0; _ni < c->head_dim; _ni++) nw[_ni] = 1.0f;
        }
        nw += c->head_dim;

        s->attn_k_norm_w[l] = nw;
        if (lw->attn_k_norm) {
            dequantize_row(lw->attn_k_norm, nw, c->head_dim,
                           lw->type_attn_k_norm);
            if (m->from_safetensors) for (int _ni = 0; _ni < c->head_dim; _ni++) nw[_ni] += 1.0f;
        } else {
            for (int _ni = 0; _ni < c->head_dim; _ni++) nw[_ni] = 1.0f;
        }
        nw += c->head_dim;
    }
    s->output_norm_w = nw;
    dequantize_row(m->weights.output_norm, nw, c->n_embd,
                   m->weights.type_output_norm);
    if (m->from_safetensors) {
        for (int _ni = 0; _ni < c->n_embd; _ni++) nw[_ni] += 1.0f;
    }

    /* Dequantize SSM F32 weights (Qwen3.5) */
    if (c->has_ssm) {
        for (int l = 0; l < c->n_layers; l++) {
            layer_weights_t *lw = &m->weights.layers[l];
            if (lw->is_attn_layer) continue;
            /* ssm_a: [dt_rank]
             * GGUF: F32, already converted to A (not A_log)
             * safetensors: F32, stores A_log, need A = -exp(A_log)
             */
            if (lw->ssm_a && s->ssm_a_w[l]) {
                if (m->from_safetensors) {
                    /* safetensors: A_log is F32, convert to A = -exp(A_log) */
                    for (int i = 0; i < c->ssm_dt_rank; i++) {
                        s->ssm_a_w[l][i] = -expf(((const float *)lw->ssm_a)[i]);
                    }
                } else {
                    /* GGUF: already A. Inverse _reorder_v_heads: simple transpose */
                    const float *src = (const float *)lw->ssm_a;
                    if (c->ssm_n_group > 0 && c->ssm_n_group < c->ssm_dt_rank) {
                        int n_k = c->ssm_n_group;
                        int n_vpk = c->ssm_dt_rank / n_k;
                        for (int g = 0; g < c->ssm_dt_rank; g++) {
                            int v = g / n_k;
                            int k = g % n_k;
                            s->ssm_a_w[l][k * n_vpk + v] = src[g];
                        }
                    } else {
                        memcpy(s->ssm_a_w[l], src, c->ssm_dt_rank * sizeof(float));
                    }
                }
            }
            /* ssm_dt.bias: [dt_rank]
             * GGUF: F32, safetensors: BF16
             */
            if (lw->ssm_dt && s->ssm_dt_w[l]) {
                if (lw->type_ssm_dt == GGUF_TYPE_BF16 || lw->type_ssm_dt == GGUF_TYPE_F16) {
                    dequantize_row(lw->ssm_dt, s->ssm_dt_w[l], c->ssm_dt_rank, lw->type_ssm_dt);
                } else {
                    memcpy(s->ssm_dt_w[l], (const float *)lw->ssm_dt, c->ssm_dt_rank * sizeof(float));
                }
                /* Inverse reorder dt_bias: GGUF tiled -> grouped (simple transpose) */
                if (!m->from_safetensors && c->ssm_n_group > 0 && c->ssm_n_group < c->ssm_dt_rank) {
                    int n_k = c->ssm_n_group;
                    int n_vpk = c->ssm_dt_rank / n_k;
                    float *tmp = alloca(c->ssm_dt_rank * sizeof(float));
                    memcpy(tmp, s->ssm_dt_w[l], c->ssm_dt_rank * sizeof(float));
                    for (int g = 0; g < c->ssm_dt_rank; g++) {
                        int v = g / n_k;
                        int k = g % n_k;
                        s->ssm_dt_w[l][k * n_vpk + v] = tmp[g];
                    }
                                    }
            }
            /* ssm_norm.weight: [head_v_dim] F32 */
            if (lw->ssm_norm && s->ssm_norm_w[l]) {
                memcpy(s->ssm_norm_w[l], (const float *)lw->ssm_norm,
                       (c->ssm_d_inner / c->ssm_dt_rank) * sizeof(float));
            }
            /* ssm_conv1d.weight: [d_conv, conv_dim]
             * GGUF: F32, safetensors: BF16
             */
            if (lw->ssm_conv1d && s->ssm_conv1d_w[l]) {
                int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
                if (lw->type_ssm_conv1d == GGUF_TYPE_BF16 || lw->type_ssm_conv1d == GGUF_TYPE_F16) {
                    dequantize_row(lw->ssm_conv1d, s->ssm_conv1d_w[l],
                                   c->ssm_d_conv * conv_dim, lw->type_ssm_conv1d);
                } else {
                    memcpy(s->ssm_conv1d_w[l], (const float *)lw->ssm_conv1d,
                           c->ssm_d_conv * conv_dim * sizeof(float));
                }
            }
        }
    }

    /* Gemma-3n: dequantize extra norm weights and small arrays */
    if (c->is_gemma3n) {
        (void)c->laurel_rank; /* used in dequantization loop below */
        int n_altup = c->n_altup;
        int n_embd = c->n_embd;
        int laurel_rank = c->laurel_rank;
        for (int l = 0; l < c->n_layers; l++) {
            layer_weights_t *lw = &m->weights.layers[l];
            /* attn_post_norm */
            if (lw->attn_post_norm) {
                dequantize_row(lw->attn_post_norm, s->attn_post_norm_w[l], n_embd, lw->type_attn_post_norm);
                if (m->from_safetensors) { float *w = s->attn_post_norm_w[l]; for (int i = 0; i < n_embd; i++) w[i] += 1.0f; }
            } else { float *w = s->attn_post_norm_w[l]; for (int i = 0; i < n_embd; i++) w[i] = 1.0f; }
            /* post_ffw_norm */
            if (lw->post_ffw_norm) {
                dequantize_row(lw->post_ffw_norm, s->post_ffw_norm_w[l], n_embd, lw->type_post_ffw_norm);
                if (m->from_safetensors) { float *w = s->post_ffw_norm_w[l]; for (int i = 0; i < n_embd; i++) w[i] += 1.0f; }
            } else { float *w = s->post_ffw_norm_w[l]; for (int i = 0; i < n_embd; i++) w[i] = 1.0f; }
            /* per_layer_post_norm */
            if (lw->per_layer_post_norm) {
                dequantize_row(lw->per_layer_post_norm, s->per_layer_post_norm_w[l], n_embd, lw->type_per_layer_post_norm);
                if (m->from_safetensors) { float *w = s->per_layer_post_norm_w[l]; for (int i = 0; i < n_embd; i++) w[i] += 1.0f; }
            } else { float *w = s->per_layer_post_norm_w[l]; for (int i = 0; i < n_embd; i++) w[i] = 1.0f; }
            /* laurel_post_norm */
            if (lw->laurel_post_norm) {
                memcpy(s->laurel_post_norm_w[l], (const float *)lw->laurel_post_norm, n_embd * sizeof(float));
                if (m->from_safetensors) { float *w = s->laurel_post_norm_w[l]; for (int i = 0; i < n_embd; i++) w[i] += 1.0f; }
            } else { float *w = s->laurel_post_norm_w[l]; for (int i = 0; i < n_embd; i++) w[i] = 1.0f; }
            /* altup_router_norm */
            if (lw->altup_router_norm) {
                memcpy(s->altup_router_norm_w[l], (const float *)lw->altup_router_norm, n_embd * sizeof(float));
                if (m->from_safetensors) { float *w = s->altup_router_norm_w[l]; for (int i = 0; i < n_embd; i++) w[i] += 1.0f; }
            } else { float *w = s->altup_router_norm_w[l]; for (int i = 0; i < n_embd; i++) w[i] = 1.0f; }
            /* altup_correct_scale */
            if (lw->altup_correct_scale) {
                memcpy(s->altup_correct_scale_w[l], (const float *)lw->altup_correct_scale, n_embd * sizeof(float));
            } else { float *w = s->altup_correct_scale_w[l]; for (int i = 0; i < n_embd; i++) w[i] = 1.0f; }
            /* altup_correct_coef: [n_altup, n_altup] F32 */
            if (lw->altup_correct_coef) {
                memcpy(s->altup_correct_coef_w[l], (const float *)lw->altup_correct_coef,
                       (size_t)n_altup * n_altup * sizeof(float));
            }
            /* altup_predict_coef: [n_altup, n_altup*n_altup] F32 */
            if (lw->altup_predict_coef) {
                memcpy(s->altup_predict_coef_w[l], (const float *)lw->altup_predict_coef,
                       (size_t)n_altup * n_altup * n_altup * sizeof(float));
            }
            /* laurel_l: [n_embd, laurel_rank] */
            if (lw->laurel_l) {
                dequantize_row(lw->laurel_l, s->laurel_l_w[l], (int)((size_t)n_embd * laurel_rank), lw->type_laurel_l);
            }
            /* laurel_r: [laurel_rank, n_embd] */
            if (lw->laurel_r) {
                dequantize_row(lw->laurel_r, s->laurel_r_w[l], (int)((size_t)laurel_rank * n_embd), lw->type_laurel_r);
            }
            /* altup_router: [n_embd, n_altup] F16 -> F32 */
            if (lw->altup_router) {
                dequantize_row(lw->altup_router, s->altup_router_w[l],
                               (int)((size_t)n_embd * n_altup), lw->type_altup_router);
            }
        }
    }

    /* Gemma-3n dequantization complete */

    /* Init tensor scratch */
    tensor_init_scratch(s->dequant_scratch, scratch_dim);

    /* Pre-compute RoPE tables (eliminates powf/cosf/sinf from hot path) */
    init_rope_tables(s, c);

#ifdef PICOLM_GPU
    /* Initialize GPU backend if requested via environment variable */
    {
        const char *gpu_env = getenv("PICOLM_GPU");
        if (gpu_env && atoi(gpu_env)) {
            const char *dev_env = getenv("PICOLM_GPU_DEVICE");
            int device = dev_env ? atoi(dev_env) : 0;
            fprintf(stderr, "INFO: initializing GPU device %d\n", device);
            if (!picolm_gpu_init(&device, 1)) {
                fprintf(stderr, "WARN: GPU init failed, falling back to CPU\n");
            } else {
                fprintf(stderr, "INFO: GPU initialized\n");
            }
        }
    }
#endif

    return 0;
}

/* Forward declaration for SSM v-head remap function (used during GPU weight upload,
 * defined later in the file) */
static inline int qwen35_vhead_gguf(int h, int n_vpk, int n_k);
static inline int qwen35_vhead_natural(int g, int n_vpk, int n_k);

/* ---- Public API ---- */

int model_load(model_t *m, const char *path, int max_seq_len, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v,
               int k_cache_hadamard, int v_cache_hadamard, int n_threads) {
    memset(m, 0, sizeof(*m));

    if (mmap_file(m, path) != 0) return -1;
    if (parse_gguf(m, max_seq_len) != 0) return -1;

    if (m->config.n_layers > MAX_LAYERS) {
        fprintf(stderr, "ERROR: model has %d layers but MAX_LAYERS=%d\n", m->config.n_layers, MAX_LAYERS);
        return -1;
    }

    /* Validate that all required tensors are present */
    {
        layer_weights_t *lw = &m->weights.layers[0];
        if (m->config.has_moe) {
            /* MoE model: check MoE tensors */
            if (!lw->ffn_gate_exps || !lw->ffn_up_exps || !lw->ffn_down_exps ||
                !lw->ffn_gate_inp || !lw->ffn_gate_inp_shexp ||
                !lw->ffn_gate_shexp || !lw->ffn_up_shexp || !lw->ffn_down_shexp) {
                fprintf(stderr, "Missing MoE tensors\n");
                return -1;
            }
            /* MoE models also have SSM tensors (like qwen35) */
            if (m->config.has_ssm) {
                if (!lw->attn_qkv || !lw->attn_gate_ssm || !lw->ssm_a ||
                    !lw->ssm_alpha || !lw->ssm_beta || !lw->ssm_conv1d ||
                    !lw->ssm_dt || !lw->ssm_norm || !lw->ssm_out) {
                    fprintf(stderr, "Missing SSM tensors in MoE model\n");
                    return -1;
                }
            }
        } else if (m->config.has_ssm) {
            /* SSM model: check SSM tensors (MLP weights are optional per layer) */
            if (!lw->attn_qkv || !lw->attn_gate_ssm || !lw->ssm_a ||
                !lw->ssm_alpha || !lw->ssm_beta || !lw->ssm_conv1d ||
                !lw->ssm_dt || !lw->ssm_norm || !lw->ssm_out) {
                fprintf(stderr, "Missing SSM tensors\n");
                return -1;
            }
        } else {
            /* Standard transformer: check attention tensors */
            if (!lw->attn_q || !lw->attn_k || !lw->attn_v || !lw->attn_output ||
                !lw->ffn_gate || !lw->ffn_up || !lw->ffn_down) {
                fprintf(stderr, "Unsupported model architecture (missing standard transformer tensors)\n");
                return -1;
            }
        }
    }

    if (allocate_run_state(m, kv_type_k, kv_type_v, k_cache_hadamard, v_cache_hadamard, n_threads) != 0) return -1;

    /* Repack Q4_0 tensors to Q4_0x8 for AVX2 SIMD optimization.
     * Only repack tensors where nrows % 8 == 0 and ncols % 32 == 0. */
    if (0 && cpu_has_avx2()) {
        repack_model_weights_q4_0x8(m);
    }

    /* Upload weight tensors to GPU */
#ifdef PICOLM_GPU
    {
        const char *gpu_env = getenv("PICOLM_GPU");
        if (gpu_env && atoi(gpu_env) && picolm_gpu_device_count() > 0) {
            int device = picolm_gpu_device_at(0);
            fprintf(stderr, "INFO: uploading model weights to GPU device %d\n", device);

            model_config_t *c = &m->config;
            int q_dim = c->n_heads * c->head_dim;
            int kv_dim = c->n_kv_heads * c->head_dim;
            int uploaded = 0, attempted = 0;

            /* Output projection: [vocab_size, n_embd] */
            attempted++;
            if (picolm_gpu_tensor_upload(&m->gpu.output,
                    m->weights.output, m->weights.type_output,
                    c->n_embd, c->vocab_size, device)) uploaded++;

            for (int l = 0; l < c->n_layers; l++) {
                layer_weights_t *lw = &m->weights.layers[l];
                gpu_layer_weights_t *gl = &m->gpu.layers[l];

#define GPU_UPLOAD(name, I, O, type) do { attempted++; \
    if (picolm_gpu_tensor_upload(&gl->name, lw->name, lw->type ## _ ## name, (I), (O), device)) uploaded++; \
} while(0)

                /* Attention Q: [q_dim, n_embd] for dense, [q_full_dim, n_embd] for SSM attn */
                attempted++;
                { int qo = (c->has_ssm && lw->is_attn_layer) ? q_dim * 2 : q_dim;
                  if (picolm_gpu_tensor_upload(&gl->attn_q,
                          lw->attn_q, lw->type_attn_q, c->n_embd, qo, device)) uploaded++; }
                /* Attention K: [kv_dim, n_embd] */
                attempted++;
                if (picolm_gpu_tensor_upload(&gl->attn_k,
                        lw->attn_k, lw->type_attn_k, c->n_embd, kv_dim, device)) uploaded++;
                /* Attention V: [kv_dim, n_embd] */
                attempted++;
                if (picolm_gpu_tensor_upload(&gl->attn_v,
                        lw->attn_v, lw->type_attn_v, c->n_embd, kv_dim, device)) uploaded++;
                /* Attention O: [n_embd, q_dim] */
                attempted++;
                if (picolm_gpu_tensor_upload(&gl->attn_output,
                        lw->attn_output, lw->type_attn_output, q_dim, c->n_embd, device)) uploaded++;
                /* FFN gate: [n_ffn, n_embd] */
                attempted++;
                if (picolm_gpu_tensor_upload(&gl->ffn_gate,
                        lw->ffn_gate, lw->type_ffn_gate, c->n_embd, c->n_ffn, device)) uploaded++;
                /* FFN up: [n_ffn, n_embd] */
                attempted++;
                if (picolm_gpu_tensor_upload(&gl->ffn_up,
                        lw->ffn_up, lw->type_ffn_up, c->n_embd, c->n_ffn, device)) uploaded++;
                /* FFN down: [n_embd, n_ffn] */
                attempted++;
                if (picolm_gpu_tensor_upload(&gl->ffn_down,
                        lw->ffn_down, lw->type_ffn_down, c->n_ffn, c->n_embd, device)) uploaded++;
                /* SSM layer tensors (Qwen3.5) */
                if (!lw->is_attn_layer && c->has_ssm) {
                    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
                    attempted++;
                    if (picolm_gpu_tensor_upload(&gl->attn_qkv,
                            lw->attn_qkv, lw->type_attn_qkv, c->n_embd, conv_dim, device)) uploaded++;
                    attempted++;
                    if (picolm_gpu_tensor_upload(&gl->attn_gate_ssm,
                            lw->attn_gate_ssm, lw->type_attn_gate_ssm, c->n_embd, c->ssm_d_inner, device)) uploaded++;
                    /* SSM F32 weights: conv1d [d_conv x conv_dim], ssm_a [dt_rank], ssm_dt [dt_rank],
                     * ssm_norm [head_v_dim], ssm_alpha [dim x dt_rank], ssm_beta [dim x dt_rank] */
                    {
                        attempted++;
                        if (picolm_gpu_tensor_upload(&gl->ssm_conv1d,
                                lw->ssm_conv1d, lw->type_ssm_conv1d, c->ssm_d_conv, conv_dim, device)) uploaded++;
                        attempted++;
                        if (picolm_gpu_tensor_upload(&gl->ssm_alpha,
                                lw->ssm_alpha, lw->type_ssm_alpha, c->n_embd, c->ssm_dt_rank, device)) uploaded++;
                        attempted++;
                        if (picolm_gpu_tensor_upload(&gl->ssm_beta,
                                lw->ssm_beta, lw->type_ssm_beta, c->n_embd, c->ssm_dt_rank, device)) uploaded++;
                        /* ssm_out: [value_dim, n_embd] projects FROM ssm_d_inner (value_dim)
                         * TO dim (n_embd), matching matmul()'s (I, O) convention.
                         * Without this upload gl->ssm_out stays NULL and
                         * tensor_set_gpu_tensor(gl->ssm_out, ...) is a silent no-op. */
                        attempted++;
                        if (picolm_gpu_tensor_upload(&gl->ssm_out,
                                lw->ssm_out, lw->type_ssm_out, c->ssm_d_inner, c->n_embd, device)) uploaded++;
                    }
                }
            }

            /* Bridge per-layer GPU tensor pointers to the global SSM arrays.
             * The SSM branch in model_forward_gpu uses gw->ssm_alpha_dev[l] etc.,
             * but the per-layer tensor upload stores them in gl->ssm_alpha (a
             * picolm_gpu_tensor_t*). Extract the .weights device pointer. */
            {
                gpu_weights_t *gw = &m->gpu;
                int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
                for (int l = 0; l < c->n_layers; l++) {
                    gpu_layer_weights_t *gl = &m->gpu.layers[l];
                    if (gl->ssm_conv1d) {
                        /* Upload dequantized F32 conv1d weight for GPU.
                         * The raw tensor is Q8_0, but the conv1d kernel
                         * expects F32 weights (like the CPU path). */
                        void *f32_w = picolm_gpu_upload_f32(
                            m->state.ssm_conv1d_w[l],
                            (size_t)c->ssm_d_conv * conv_dim, device);
                        gw->ssm_conv1d_dev[l] = f32_w;
                    }
                    if (gl->ssm_alpha)
                        gw->ssm_alpha_dev[l] = (void *)picolm_gpu_tensor_weights((picolm_gpu_tensor_t *)gl->ssm_alpha);
                    if (gl->ssm_beta)
                        gw->ssm_beta_dev[l] = (void *)picolm_gpu_tensor_weights((picolm_gpu_tensor_t *)gl->ssm_beta);
                }
            }

            m->gpu.device = device;
            if (uploaded > 0) {
                m->gpu.active = 1;

                /* Phase 1+2: Allocate GPU KV cache and pipeline buffers for eligible models.
                 * Eligibility (non-SSM): standard MHA (no SSM), rope_type==0, no QK-norm, F16 KV.
                 * Eligibility (SSM models): Qwen3.6 DeltaNet shape accepted.
                 *   - attention-only layers: standard MHA/GQA
                 *   - hybrid layers: must have ssm_d_state > 0, ssm_dt_rank > 0, etc.
                 *   - rope_type==0 required
                 *   - F16 KV cache required
                 * QK-norm layers in attention-only layers are handled by GPU RMSNorm (no veto). */
                {
                    int eligible = 1;
                    /* rope_type 0 (Llama pairwise) and 1 (Qwen2 interleaved)
                     * are both supported now (b67b1df) -- this used to be
                     * a hard veto on anything but 0, stale since that fix. */
                    if (c->rope_type != 0 && c->rope_type != 1) eligible = 0;
                    /* F16 KV cache only */
                    if (kv_type_k != KV_CACHE_F16 || kv_type_v != KV_CACHE_F16) eligible = 0;
                    /* SSM models: enable GPU pipeline.
                     * Q+gate de-interleave race condition is fixed.
                     * Recurrence kernel matches NEON order. */
                    if (c->has_ssm) {
                        if (c->ssm_d_state <= 0 || c->ssm_d_state > 256) eligible = 0;
                        if (c->ssm_dt_rank <= 0) eligible = 0;
                        if (c->ssm_n_group <= 0) eligible = 0;
                        if (c->ssm_d_inner <= 0) eligible = 0;
                        if (c->ssm_d_conv <= 1) eligible = 0;
                        if (eligible && c->ssm_d_inner % c->ssm_dt_rank != 0) eligible = 0;
                    }

                    if (eligible) {
                        /* Compute KV cache sizes matching CPU allocation */
                        int kv_layers = 0;
                        for (int l = 0; l < c->n_layers; l++) {
                            if (m->weights.layers[l].is_attn_layer) kv_layers++;
                        }
                        if (kv_layers == 0) kv_layers = c->n_layers;

                        size_t kv_head_stride = c->n_kv_heads * c->head_dim * sizeof(uint16_t);
                        /* Per-layer: max_seq_len * n_kv_heads * head_dim * sizeof(uint16_t) */
                        size_t layer_bytes = (size_t)c->max_seq_len * kv_head_stride;
                        size_t total_k = (size_t)kv_layers * layer_bytes;
                        size_t total_v = (size_t)kv_layers * layer_bytes;

                        if (picolm_gpu_kv_alloc(total_k, total_v, device)) {
                            m->gpu.kv_k_dev = (void *)1; /* opaque marker: allocated */
                            m->gpu.kv_v_dev = (void *)1;
                            m->gpu.kv_k_cap = total_k;
                            m->gpu.kv_v_cap = total_v;
                            m->gpu.kv_active = 1;
                            fprintf(stderr, "INFO: GPU KV cache allocated (%zu MB K + %zu MB V)\n",
                                    total_k / (1024*1024), total_v / (1024*1024));

                            /* Phase 2: allocate pipeline buffers and upload norm/RoPE weights */
                            int q_dim = c->n_heads * c->head_dim;
                            int kv_dim = c->n_kv_heads * c->head_dim;
                            /* SSM models use q_full_dim = 2*q_dim for attention Q+gate projection.
                             * pipe_q and pipe_attn_out must be sized for the larger output. */
                            int q_pipeline_dim = c->has_ssm ? (q_dim * 2) : q_dim;
                            if (!picolm_gpu_pipeline_alloc(c->n_embd, q_pipeline_dim, kv_dim, c->n_ffn, device)) {
                                fprintf(stderr, "WARN: GPU pipeline buffer alloc failed\n");
                            }
                            /* Prefill batch buffers - xb_stride must accommodate both
                             * q_pipeline_dim (attention QKV) and ssm_conv_dim (SSM QKV) */
                            int ssm_conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
                            int xb_stride = q_pipeline_dim;
                            if (ssm_conv_dim > xb_stride) xb_stride = ssm_conv_dim;
                            if (c->n_embd > xb_stride) xb_stride = c->n_embd;
                            if (!picolm_gpu_pipeline_batch_alloc(c->n_embd, q_pipeline_dim, kv_dim,
                                                                 c->n_ffn, xb_stride, c->max_seq_len, device)) {
                                fprintf(stderr, "WARN: GPU prefill batch buffer alloc failed\n");
                            }

                            /* Pre-allocate Q8_0 scratch buffers to max size needed
                             * to avoid runtime reallocation races between GPU work
                             * and host-side cudaFree/cudaMalloc in reserve/reserve_i8.
                             * Max input dim = max(q_pipeline_dim, ssm_conv_dim, n_ffn).
                             * Max batch = c->max_seq_len. */
                            int max_q8_dim = q_pipeline_dim;
                            if (ssm_conv_dim > max_q8_dim) max_q8_dim = ssm_conv_dim;
                            if (c->n_ffn > max_q8_dim) max_q8_dim = c->n_ffn;
                            size_t max_xq = (size_t)c->max_seq_len * max_q8_dim;
                            size_t max_xd = (size_t)c->max_seq_len * (max_q8_dim + 31) / 32 * sizeof(float);
                            picolm_gpu_prealloc_q8(max_xq, max_xd, device);

                            /* Upload norm weights and RoPE tables to device */
                            run_state_t *s = &m->state;
                            /* RoPE cos/sin: [max_seq_len * rope_half] */
                            {
                                int rope_dim = (c->rope_dim > 0) ? c->rope_dim : c->head_dim;
                                int rope_half = rope_dim / 2;
                                size_t rope_n = (size_t)c->max_seq_len * rope_half;
                                m->gpu.rope_cos_dev = picolm_gpu_upload_f32(s->rope_cos, rope_n, device);
                                m->gpu.rope_sin_dev = picolm_gpu_upload_f32(s->rope_sin, rope_n, device);
                            }
                            /* Output norm */
                            m->gpu.output_norm_dev = picolm_gpu_upload_f32(s->output_norm_w, c->n_embd, device);
                            /* Per-layer norm weights */
                            for (int l = 0; l < c->n_layers; l++) {
                                m->gpu.attn_norm_dev[l] =
                                    picolm_gpu_upload_f32(s->attn_norm_w[l], c->n_embd, device);
                                m->gpu.post_attn_norm_dev[l] =
                                    picolm_gpu_upload_f32(s->post_attn_norm_w[l], c->n_embd, device);
                                /* QK-norm weights (Qwen3): per-head RMSNorm [head_dim]
                                 * Only upload if the model actually has QK-norm tensors
                                 * (w->layers[l].attn_q_norm non-NULL). The host buffer
                                 * is always allocated but initialized to 1.0 when absent.
                                 * Must match the CPU path which checks lw->attn_q_norm. */
                                if (m->weights.layers[l].attn_q_norm) {
                                    m->gpu.attn_qk_norm_q_dev[l] =
                                        picolm_gpu_upload_f32(s->attn_q_norm_w[l], c->head_dim, device);
                                    m->gpu.attn_qk_norm_k_dev[l] =
                                        picolm_gpu_upload_f32(s->attn_k_norm_w[l], c->head_dim, device);
                                }
                            }

                            /* SSM pipeline buffers and weight uploads (for hybrid layers) */
                            if (c->has_ssm) {
                                int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
                                if (!picolm_gpu_ssm_pipeline_alloc(conv_dim, c->ssm_d_inner,
                                                                    c->ssm_dt_rank, device)) {
                                    fprintf(stderr, "WARN: GPU SSM pipeline buffer alloc failed\n");
                                }
                                run_state_t *ss = &m->state;
                                int head_v_dim = c->ssm_d_inner / c->ssm_dt_rank;
                                for (int l = 0; l < c->n_layers; l++) {
                                    layer_weights_t *lw = &m->weights.layers[l];
                                    if (lw->is_attn_layer) continue;
                                    m->gpu.ssm_a_dev[l] = picolm_gpu_upload_f32(ss->ssm_a_w[l], c->ssm_dt_rank, device);
                                    m->gpu.ssm_dt_dev[l] = picolm_gpu_upload_f32(ss->ssm_dt_w[l], c->ssm_dt_rank, device);
                                    m->gpu.ssm_norm_dev[l] = picolm_gpu_upload_f32(ss->ssm_norm_w[l], head_v_dim, device);
                                    /* Allocate device-resident SSM state */
                                    if (ss->ssm_state[l]) {
                                        size_t ssm_st_bytes = (size_t)c->ssm_d_state *
                                             (size_t)c->ssm_d_state *
                                             (size_t)c->ssm_dt_rank * sizeof(float);
                                        m->gpu.ssm_state_dev[l] = picolm_gpu_alloc_device(ssm_st_bytes, device);
                                        if (m->gpu.ssm_state_dev[l]) {
                                            picolm_gpu_device_memset(m->gpu.ssm_state_dev[l], 0,
                                                ssm_st_bytes, device);
                                        }
                                    }
                                    /* Conv state: (d_conv-1) * conv_dim floats */
                                    size_t conv_st_bytes = (size_t)(c->ssm_d_conv - 1) *
                                        (size_t)conv_dim * sizeof(float);
                                    m->gpu.ssm_conv_state_dev[l] = picolm_gpu_alloc_device(conv_st_bytes, device);
                                    if (m->gpu.ssm_conv_state_dev[l]) {
                                        picolm_gpu_device_memset(m->gpu.ssm_conv_state_dev[l], 0,
                                            conv_st_bytes, device);
                                    }
                                }
                                /* Upload head_map for v-head remapping (if GGUF reorders v-heads) */
                                {
                                    int n_vh = c->ssm_dt_rank;
                                    int n_kh = c->ssm_n_group;
                                    int n_vpk = c->ssm_dt_rank / n_kh;
                                    int half_vpk = n_vpk / 2;
                                    int do_rm = !m->from_safetensors && n_kh > 0 &&
                                                n_kh < n_vh && half_vpk > 0;
                                    if (do_rm) {
                                        int *hmap = alloca(n_vh * sizeof(int));
                                        int *himap = alloca(n_vh * sizeof(int));
                                        for (int h = 0; h < n_vh; h++) {
                                            hmap[h] = qwen35_vhead_gguf(h, n_vpk, n_kh);
                                            himap[h] = qwen35_vhead_natural(h, n_vpk, n_kh);
                                        }
                                        m->gpu.ssm_head_map_dev = picolm_gpu_upload_int(hmap, n_vh, device);
                                        m->gpu.ssm_head_invmap_dev = picolm_gpu_upload_int(himap, n_vh, device);
                                    }
                                }
                            }
                        } else {
                            fprintf(stderr, "WARN: GPU KV cache allocation failed, falling back to CPU\n");
                        }
                    } else {
                        fprintf(stderr, "INFO: GPU KV cache disabled (model not eligible: SSM=%d rope=%d)\n",
                                c->has_ssm, c->rope_type);
                        /* SSM model: upload small SSM F32 weights to device for the GPU kernels
                         * that are already called from ssm_forward().
                         * These are: ssm_a [dt_rank], ssm_dt [dt_rank], ssm_norm [head_v_dim] per layer.
                         * The larger SSM weights (conv1d, alpha, beta) were already uploaded above
                         * as GPU tensor handles during the per-layer upload loop. */
                        if (c->has_ssm) {
                            /* SSM pipeline buffers (needed for the future hybrid layer branch) */
                            {
                                int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
                                if (!picolm_gpu_ssm_pipeline_alloc(conv_dim, c->ssm_d_inner,
                                                                    c->ssm_dt_rank, device)) {
                                    fprintf(stderr, "WARN: GPU SSM pipeline buffer alloc failed\n");
                                }
                            }
                            run_state_t *s = &m->state;
                            int head_v_dim = c->ssm_d_inner / c->ssm_dt_rank;
                            for (int l = 0; l < c->n_layers; l++) {
                                layer_weights_t *lw = &m->weights.layers[l];
                                if (lw->is_attn_layer) continue;
                                m->gpu.ssm_a_dev[l] = picolm_gpu_upload_f32(s->ssm_a_w[l], c->ssm_dt_rank, device);
                                m->gpu.ssm_dt_dev[l] = picolm_gpu_upload_f32(s->ssm_dt_w[l], c->ssm_dt_rank, device);
                                m->gpu.ssm_norm_dev[l] = picolm_gpu_upload_f32(s->ssm_norm_w[l], head_v_dim, device);
                                /* Allocate device-resident SSM state */
                                if (s->ssm_state[l]) {
                                    size_t st_bytes = (size_t)c->ssm_dt_rank * c->ssm_d_state * c->ssm_d_state * sizeof(float);
                                    m->gpu.ssm_state_dev[l] = picolm_gpu_alloc_device(st_bytes, device);
                                    /* Initialize to zero (fresh start) */
                                    if (m->gpu.ssm_state_dev[l])
                                        picolm_gpu_device_memset(m->gpu.ssm_state_dev[l], 0, st_bytes, device);
                                }
                                /* Allocate device-resident conv state: [(d_conv-1) x conv_dim] */
                                if (s->ssm_conv_state[l]) {
                                    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
                                    size_t cs_bytes = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
                                    m->gpu.ssm_conv_state_dev[l] = picolm_gpu_alloc_device(cs_bytes, device);
                                    if (m->gpu.ssm_conv_state_dev[l])
                                        picolm_gpu_device_memset(m->gpu.ssm_conv_state_dev[l], 0, cs_bytes, device);
                                }
                            }
                            /* Upload head_map for v-head remapping (if needed) */
                            {
                                int do_remap = (c->ssm_n_group > 0 && c->ssm_n_group < c->ssm_dt_rank);
                                if (do_remap) {
                                    int n_v_heads = c->ssm_dt_rank;
                                    int n_k_heads = c->ssm_n_group;
                                    int n_vpk = c->ssm_dt_rank / n_k_heads;
                                    int *hmap = alloca(n_v_heads * sizeof(int));
                                    int *himap = alloca(n_v_heads * sizeof(int));
                                    for (int h = 0; h < n_v_heads; h++) {
                                        hmap[h] = qwen35_vhead_gguf(h, n_vpk, n_k_heads);
                                        himap[h] = qwen35_vhead_natural(h, n_vpk, n_k_heads);
                                    }
                                    m->gpu.ssm_head_map_dev = picolm_gpu_upload_int(hmap, n_v_heads, device);
                                    m->gpu.ssm_head_invmap_dev = picolm_gpu_upload_int(himap, n_v_heads, device);
                                }
                            }
                        }
                    }
                }
                /* Log per-type upload stats */
                int tcounts[512] = {0};
                for (int l = 0; l < c->n_layers; l++) {
                    layer_weights_t *lw = &m->weights.layers[l];
                    int types[] = {lw->type_attn_q, lw->type_attn_k, lw->type_attn_v,
                                   lw->type_attn_output, lw->type_ffn_gate, lw->type_ffn_up, lw->type_ffn_down};
                    for (int j = 0; j < 7; j++) {
                        int t = types[j];
                        if (t >= 0 && t < 512) tcounts[t]++;
                    }
                }
                /* Count output tensor */
                int ot = m->weights.type_output;
                if (ot >= 0 && ot < 512) tcounts[ot]++;
                fprintf(stderr, "INFO: GPU weights uploaded (%d/%d tensors)\n", uploaded, attempted);
                for (int t = 0; t < 20; t++) {
                    if (tcounts[t]) fprintf(stderr, "  type %d (%s): %d tensors\n", t, gguf_type_name(t), tcounts[t]);
                }
            } else {
                fprintf(stderr, "WARN: GPU upload failed for all tensors, using CPU\n");
            }
        }
    }
#endif

    /* Log tensor type distribution */
    {
        int tcounts[512] = {0};
        for (int l = 0; l < m->config.n_layers; l++) {
            layer_weights_t *lw = &m->weights.layers[l];
            int types[] = {lw->type_attn_q, lw->type_attn_k, lw->type_attn_v,
                           lw->type_attn_output, lw->type_ffn_gate, lw->type_ffn_up, lw->type_ffn_down,
                           lw->type_ssm_out, lw->type_ssm_conv1d, lw->type_ssm_alpha, lw->type_ssm_beta,
                           lw->type_ssm_a, lw->type_ssm_dt, lw->type_attn_norm, lw->type_post_attn_norm};
            for (int j = 0; j < 15; j++) {
                if (types[j] >= 0 && types[j] < 512) tcounts[types[j]]++;
            }
        }
        if (m->weights.type_token_embd >= 0 && m->weights.type_token_embd < 512) tcounts[m->weights.type_token_embd]++;
        if (m->weights.type_output >= 0 && m->weights.type_output < 512) tcounts[m->weights.type_output]++;
        fprintf(stderr, "Tensor types:");
        for (int t = 0; t < 512; t++) {
            if (tcounts[t]) fprintf(stderr, " %s(%d)", gguf_type_name(t), tcounts[t]);
        }
        fprintf(stderr, "\n");
    }

    return 0;
}

/* Helper: repack a single Q4_0 tensor to Q4_0x8 if eligible.
 * Returns the allocated buffer or NULL. */
static void *try_repack_q4_0(model_t *m, const void *data, int nrows, int ncols, int buf_idx) {
    if (nrows % 8 != 0 || ncols % 32 != 0 || !data) return NULL;
    size_t size = gguf_type_row_size(GGUF_TYPE_Q4_0, ncols) * nrows;
    void *buf = malloc(size);
    if (!buf) return NULL;
    repack_q4_0_to_q4_0x8(data, buf, nrows, ncols);
    m->repack_buffers[buf_idx] = buf;
    m->repack_used[buf_idx] = 1;
    fprintf(stderr, "Repacked Q4_0 tensor (%dx%d) to Q4_0x8 for AVX2 [%zu bytes]\n",
            nrows, ncols, size);
    return buf;
}

/* Repack Q4_0 weight tensors to Q4_0x8 interleaved format for AVX2.
 * Allocates buffers only for eligible tensors (nrows%8==0, ncols%32==0). */
static void repack_model_weights_q4_0x8(model_t *m) {
    model_weights_t *w = &m->weights;
    model_config_t *c = &m->config;

    /* Token embedding: vocab_size x n_embd */
    try_repack_q4_0(m, w->token_embd, c->vocab_size, c->n_embd, 0);

    /* Output projection: n_embd x vocab_size (transpose: vocab_size x n_embd) */
    if (w->output && w->type_output == GGUF_TYPE_Q4_0) {
        try_repack_q4_0(m, w->output, c->vocab_size, c->n_embd, 1);
    }

    /* Per-layer weights */
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];
        int idx = 2 + l * 9; /* base index for this layer */
        int kv_dim = c->n_kv_heads * c->head_dim;

        if (lw->type_attn_q == GGUF_TYPE_Q4_0)
            try_repack_q4_0(m, lw->attn_q, c->n_embd, c->n_embd, idx);
        if (lw->type_attn_k == GGUF_TYPE_Q4_0)
            try_repack_q4_0(m, lw->attn_k, kv_dim, c->n_embd, idx + 1);
        if (lw->type_attn_v == GGUF_TYPE_Q4_0)
            try_repack_q4_0(m, lw->attn_v, kv_dim, c->n_embd, idx + 2);
        if (lw->type_attn_output == GGUF_TYPE_Q4_0)
            try_repack_q4_0(m, lw->attn_output, c->n_embd, c->n_embd, idx + 3);
        if (lw->type_ffn_gate == GGUF_TYPE_Q4_0)
            try_repack_q4_0(m, lw->ffn_gate, c->n_ffn, c->n_embd, idx + 4);
        if (lw->type_ffn_down == GGUF_TYPE_Q4_0)
            try_repack_q4_0(m, lw->ffn_down, c->n_embd, c->n_ffn, idx + 5);
        if (lw->type_ffn_up == GGUF_TYPE_Q4_0)
            try_repack_q4_0(m, lw->ffn_up, c->n_ffn, c->n_embd, idx + 6);
    }
}


/* ---- Gemma-3n forward pass (specialized) ---- */
float *model_forward_gemma3n(model_t *m, int token, int pos);

float *model_forward(model_t *m, int token, int pos) {
    /* Gemma-3n has a fundamentally different architecture */
    if (m->config.is_gemma3n) {
        return model_forward_gemma3n(m, token, pos);
    }
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;

    int dim    = c->n_embd;
    int n_ffn  = c->n_ffn;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int q_dim = n_heads * head_dim;
    /* Qwen3.5 full attention uses Q+gate joint projection: 2x q_dim */
#ifdef PICOLM_GPU
    int gpu_ok = m->gpu.active;
    int gpu_dev = m->gpu.device;
#endif
    int q_full_dim = c->has_ssm ? (n_heads * head_dim * 2) : q_dim;
    int kv_dim = n_kv_heads * head_dim;
    int kv_mul = n_heads / n_kv_heads;
    int seq_len = c->max_seq_len;
    int rope_dim = (c->rope_dim > 0) ? c->rope_dim : head_dim;
    int half_dim = rope_dim / 2;

    /* RoPE table pointers for this position */
    const float *cos_pos = s->rope_cos + (size_t)pos * half_dim;
    const float *sin_pos = s->rope_sin + (size_t)pos * half_dim;

    /* 1. Embedding lookup */
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        if (w->type_token_embd == GGUF_TYPE_Q4_0_8_8) {
            int nb = dim / 32;
            size_t group_bytes = (size_t)nb * sizeof(block_q4_0x8);
            int group = token / 8;
            int r = token % 8;
            const block_q4_0x8 *blocks = (const block_q4_0x8 *)((const uint8_t *)w->token_embd + (size_t)group * group_bytes);

            for (int i = 0; i < nb; i++) {
                float d = fp16_to_fp32(blocks[i].d[r]);
                for (int j = 0; j < 8; j++) {
                    uint8_t byte = blocks[i].qs[r * 8 + j];
                    int v0 = (int8_t)(byte << 4);
                    int v1 = (int8_t)(byte & 0xF0);
                    s->x[i * 32 + j * 2]     = d * (float)(v0 >> 4);
                    s->x[i * 32 + j * 2 + 1] = d * (float)(v1 >> 4);
                }
                for (int j = 0; j < 8; j++) {
                    uint8_t byte = blocks[i].qs[64 + r * 8 + j];
                    int v0 = (int8_t)(byte << 4);
                    int v1 = (int8_t)(byte & 0xF0);
                    s->x[i * 32 + 16 + j * 2]     = d * (float)(v0 >> 4);
                    s->x[i * 32 + 16 + j * 2 + 1] = d * (float)(v1 >> 4);
                }
            }
        } else if (w->type_token_embd == GGUF_TYPE_Q4_0_4_4) {
            /* Q4_0_4_4: 4 rows interleaved in block_q4_0x4 structs.
             * qs[k*16 + r*4 + j] = row_r.qs[k*4 + j] ^ 0x88
             * Each block_q4_0x4 covers 4 rows x 32 columns. */
            int nb = dim / 32;
            size_t group_bytes = (size_t)nb * sizeof(block_q4_0x4);
            int group = token / 4;
            int r = token % 4;
            const block_q4_0x4 *blocks = (const block_q4_0x4 *)((const uint8_t *)w->token_embd + (size_t)group * group_bytes);

            for (int i = 0; i < nb; i++) {
                float d = fp16_to_fp32_lookup(blocks[i].d[r]);
                for (int k = 0; k < 4; k++) {
                    for (int j = 0; j < 4; j++) {
                        uint8_t byte = blocks[i].qs[k * 16 + r * 4 + j];
                        int v0 = (int8_t)(byte << 4) >> 4;
                        int v1 = (int8_t)(byte & 0xF0) >> 4;
                        s->x[i * 32 + k * 8 + j * 2]     = d * (float)v0;
                        s->x[i * 32 + k * 8 + j * 2 + 1] = d * (float)v1;
                    }
                }
            }
        } else if (w->type_token_embd == GGUF_TYPE_Q4_0_4_8) {
            /* Q4_0_4_8: 4 rows interleaved in block_q4_0x4, blocklen=8.
             * qs[k*32 + r*8 + j] = row_r.qs[k*8 + j] ^ 0x88
             * Each block_q4_0x4 covers 4 rows x 32 columns, 2 groups of 8 bytes per row. */
            int nb = dim / 32;
            size_t group_bytes = (size_t)nb * sizeof(block_q4_0x4);
            int group = token / 4;
            int r = token % 4;
            const block_q4_0x4 *blocks = (const block_q4_0x4 *)((const uint8_t *)w->token_embd + (size_t)group * group_bytes);

            for (int i = 0; i < nb; i++) {
                float d = fp16_to_fp32_lookup(blocks[i].d[r]);
                for (int k = 0; k < 2; k++) {
                    for (int j = 0; j < 8; j++) {
                        uint8_t byte = blocks[i].qs[k * 32 + r * 8 + j];
                        int v0 = (int8_t)(byte << 4) >> 4;
                        int v1 = (int8_t)(byte & 0xF0) >> 4;
                        s->x[i * 32 + k * 16 + j * 2]     = d * (float)v0;
                        s->x[i * 32 + k * 16 + j * 2 + 1] = d * (float)v1;
                    }
                }
            }
        } else {
            const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
            dequantize_row(embd_row, s->x, dim, w->type_token_embd);
        }
    }

    /* 2. Transformer layers
     * Skip MTP (Multi-Token Prediction) layers at the end.
     * MTP layers have "nextn." tensors and are used for speculative
     * decoding. Full MTP support is planned: during generation, after
     * the main forward pass, run MTP layers on the output embedding
     * to produce N candidate tokens, then verify with a fast forward
     * pass. For now, skip them entirely. */
    int n_active_layers = c->n_layers - c->n_mtp_layers;
    int attn_ordinal = 0;
    for (int slot = 0; slot < n_active_layers; slot++) {
#ifdef PICOLM_VIZ
        int l = viz_layer_permute(slot);
#else
        int l = slot;
#endif
        layer_weights_t *lw = &w->layers[l];
        BENCH_LAYER_START();
#ifdef PICOLM_GPU
        gpu_layer_weights_t *gl = &m->gpu.layers[l];
#endif
        int ri = 2 + l * 9;

#ifdef PICOLM_VIZ
        /* Layer skip: toggled by VNC mouse click, zero overhead when unused */
        if (viz_layer_skip(l)) {
            /* Pass through: residual stays unchanged, no computation */
#ifdef PICOLM_VIZ
            viz_push_layer(l, s->x, dim);
#endif
            BENCH_LAYER_END(l, 0);
            continue;
        }
#endif

        if (c->has_ssm && !lw->is_attn_layer) {
            /* SSM layer (Qwen3.5) */
#ifdef PICOLM_GPU
            tensor_set_gpu_tensor(NULL, 0); /* clear stale handle from previous layer */
            float *ssm_residual = s->xb2;
            ssm_forward(m, s, s->x, ssm_residual, lw, l, pos, &m->gpu.layers[l]);
#else
            float *ssm_residual = s->xb2;
            ssm_forward(m, s, s->x, ssm_residual, lw, l, pos, NULL);
#endif
#ifdef PICOLM_VIZ
            viz_push_layer(l, s->x, dim);
#endif
            BENCH_LAYER_END(l, 0);
            continue;
        }
        /* ---- Attention ---- */
        rmsnorm(s->xb, s->x, s->attn_norm_w[l], dim, c->rms_norm_eps);

        /* Q projection (Q+gate joint for Qwen3.5 full attention) */
        tensor_set_repacked(m->repack_used[ri] ? m->repack_buffers[ri] : NULL);
#ifdef PICOLM_GPU
        if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->attn_q, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
        int this_q_dim = (c->has_ssm && lw->is_attn_layer) ? q_full_dim : q_dim;
        matmul(s->q, s->xb, lw->attn_q, dim, this_q_dim, lw->type_attn_q);
        tensor_set_repacked(NULL);

        /* For Qwen3.5: de-interleave per-head Q+gate into block layout
         * GGUF stores [Q_0, Gate_0, Q_1, Gate_1, ...] (per-head interleaved)
         * We need [Q_0, Q_1, ..., Q_15] in s->q, gate stored separately.
         * Gate stored in s->hb (FFN buffer) to survive K/V projection writes.
         */
        float *qwen35_attn_gate = NULL;
        if (c->has_ssm && lw->is_attn_layer) {
            float *qg_raw = s->q; /* [q_full_dim] = interleaved Q+gate */
            float *q_block = s->q; /* compact Q heads here (in-place) */
            float *gate_block = s->hb; /* gate survives K/V projection */
            for (int h = 0; h < n_heads; h++) {
                memmove(q_block + h * head_dim, qg_raw + h * 2 * head_dim,
                        head_dim * sizeof(float));
                memmove(gate_block + h * head_dim, qg_raw + h * 2 * head_dim + head_dim,
                        head_dim * sizeof(float));
            }
            qwen35_attn_gate = gate_block;
        }

        /* K projection */
        tensor_set_repacked(m->repack_used[ri+1] ? m->repack_buffers[ri+1] : NULL);
#ifdef PICOLM_GPU
        if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->attn_k, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
        float *k_tmp = s->xb2; /* reuse xb2 as temp for K (kv_dim <= dim) */
        matmul(k_tmp, s->xb, lw->attn_k, dim, kv_dim, lw->type_attn_k);
        tensor_set_repacked(NULL);

        int this_attn_ordinal = attn_ordinal++;
        /* GQA layout: kcache_layer points to this layer's K cache [seq_len][kv_row_size_gqa] */
        uint8_t *kcache_layer = s->key_cache + (size_t)this_attn_ordinal * seq_len * s->kv_row_size_k;
        uint8_t *vcache_layer = s->val_cache + (size_t)this_attn_ordinal * seq_len * s->kv_row_size_v;

        /* QK-norm (Qwen3): per-head RMSNorm applied before RoPE */
        if (lw->attn_q_norm) {
            float *qnw = s->attn_q_norm_w[l];
            float *knw = s->attn_k_norm_w[l];
            int qk_done = 0;
#ifdef PICOLM_GPU
            if (gpu_ok && m->gpu.active) {
                if (picolm_gpu_rmsnorm_batched(s->q, s->q, qnw, head_dim, c->rms_norm_eps, n_heads, 0, m->gpu.device) &&
                    picolm_gpu_rmsnorm_batched(k_tmp, k_tmp, knw, head_dim, c->rms_norm_eps, n_kv_heads, 0, m->gpu.device)) {
                    qk_done = 1;
                }
            }
#endif
            if (!qk_done) {
                for (int h = 0; h < n_heads; h++)
                    rmsnorm(s->q + h * head_dim, s->q + h * head_dim, qnw, head_dim, c->rms_norm_eps);
                for (int h = 0; h < n_kv_heads; h++)
                    rmsnorm(k_tmp + h * head_dim, k_tmp + h * head_dim, knw, head_dim, c->rms_norm_eps);
            }
        }

        /* Apply RoPE to Q and K */
        int rope_half = rope_dim / 2;
        rope(s->q, k_tmp, head_dim, n_heads, n_kv_heads, cos_pos, sin_pos, c->rope_type, rope_half);

        /* Store K: GQA row quantization
         * For quantized types, quantize the full GQA row (n_kv_heads*head_dim) at once.
         * For FP16, store each head's FP16 data at its offset within the row. */
        {
            uint8_t *key_pos = kcache_layer + (size_t)pos * s->kv_row_size_k;
            /* Hadamard rotation of K before quantization */
            if (s->kv_hadamard_k && s->kv_type_k != KV_CACHE_F16) {
                for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                    picolm_hadamard_transform(k_tmp + hkv * head_dim, head_dim, s->kv_hadamard_size);
                }
            }
            if (s->kv_type_k == KV_CACHE_Q8_0) {
                /* Quantize full GQA row at once */
                quantize_row_q8_0(k_tmp, key_pos, kv_dim);
            } else if (s->kv_type_k == KV_CACHE_Q4_0) {
                /* Quantize full GQA row at once */
                quantize_row_q4_0(k_tmp, key_pos, kv_dim);
            } else if (s->kv_type_k == KV_CACHE_TQ3) {
                /* TQ3: quantize per head (not full GQA row) since
                 * quantize_row_tq3 operates on contiguous float32.
                 * TQ3 handles WHT internally so skip Hadamard rotation. */
                for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                    quantize_row_tq3(k_tmp + hkv * head_dim,
                        key_pos + hkv * s->kv_head_stride_k, head_dim);
                }
            } else if (s->kv_type_k == KV_CACHE_TQ4) {
                /* TQ4: same per-head quantization as TQ3, 16-entry codebook */
                for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                    quantize_row_tq4(k_tmp + hkv * head_dim,
                        key_pos + hkv * s->kv_head_stride_k, head_dim);
                }
            } else {
                /* FP16: store each head at its offset within the GQA row */
                for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                    float *k_head = k_tmp + hkv * head_dim;
                    uint16_t *kf = (uint16_t *)(key_pos + hkv * s->kv_head_stride_k);
#ifdef PICOLM_FP16_HW
                    { int d = 0;
                      for (; d + 3 < head_dim; d += 4)
                          f32x4_to_fp16_hw(kf + d, vld1q_f32(k_head + d));
                      for (; d < head_dim; d++) kf[d] = fp32_to_fp16(k_head[d]);
                    }
#else
                    for (int d = 0; d < head_dim; d++) kf[d] = fp32_to_fp16(k_head[d]);
#endif
                }
                /* Phase 1.5: key_pos now holds the full contiguous GQA row
                 * (all kv_heads) -- one bulk async copy instead of one
                 * kernel launch per head. */
#ifdef PICOLM_GPU
                if (gpu_ok && m->gpu.kv_active) {
                    picolm_gpu_kv_store_rows(1, this_attn_ordinal, pos, 1,
                                             key_pos, s->kv_row_size_k,
                                             n_kv_heads, head_dim, seq_len, gpu_dev);
                }
#endif
            }
        }

        /* V projection -> store */
        tensor_set_repacked(m->repack_used[ri+2] ? m->repack_buffers[ri+2] : NULL);
#ifdef PICOLM_GPU
        if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->attn_v, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
        float *v_tmp = s->xb2;
        matmul(v_tmp, s->xb, lw->attn_v, dim, kv_dim, lw->type_attn_v);
        tensor_set_repacked(NULL);
        /* Store V: GQA row quantization */
        {
            uint8_t *val_pos = vcache_layer + (size_t)pos * s->kv_row_size_v;
            /* Hadamard rotation of V before quantization */
            if (s->kv_hadamard_v && s->kv_type_v != KV_CACHE_F16) {
                for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                    picolm_hadamard_transform(v_tmp + hkv * head_dim, head_dim, s->kv_hadamard_size);
                }
            }
            if (s->kv_type_v == KV_CACHE_Q8_0) {
                quantize_row_q8_0(v_tmp, val_pos, kv_dim);
            } else if (s->kv_type_v == KV_CACHE_Q4_0) {
                quantize_row_q4_0(v_tmp, val_pos, kv_dim);
            } else if (s->kv_type_v == KV_CACHE_TQ3) {
                /* TQ3: quantize per head */
                for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                    quantize_row_tq3(v_tmp + hkv * head_dim,
                        val_pos + hkv * s->kv_head_stride_v, head_dim);
                }
            } else if (s->kv_type_v == KV_CACHE_TQ4) {
                /* TQ4: quantize per head */
                for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                    quantize_row_tq4(v_tmp + hkv * head_dim,
                        val_pos + hkv * s->kv_head_stride_v, head_dim);
                }
            } else {
                for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                    float *v_head = v_tmp + hkv * head_dim;
                    uint16_t *vf = (uint16_t *)(val_pos + hkv * s->kv_head_stride_v);
#ifdef PICOLM_FP16_HW
                    { int d = 0;
                      for (; d + 3 < head_dim; d += 4)
                          f32x4_to_fp16_hw(vf + d, vld1q_f32(v_head + d));
                      for (; d < head_dim; d++) vf[d] = fp32_to_fp16(v_head[d]);
                    }
#else
                    for (int d = 0; d < head_dim; d++) vf[d] = fp32_to_fp16(v_head[d]);
#endif
                }
                #ifdef PICOLM_GPU
                if (gpu_ok && m->gpu.kv_active) {
                    picolm_gpu_kv_store_rows(0, this_attn_ordinal, pos, 1,
                                             val_pos, s->kv_row_size_v,
                                             n_kv_heads, head_dim, seq_len, gpu_dev);
                }
#endif
            }
        }

/* ---- Flash Attention (online softmax) ----
         *
         * Instead of materializing the full [n_heads * seq_len] score array,
         * compute attention in a single pass using the online softmax trick:
         *
         *   max_s = -inf, sum_exp = 0, acc[d] = 0
         *   for each cached position t:
         *     s = dot(Q_h, K_t) / sqrt(d)
         *     if s > max_s:
         *       correction = exp(max_s - s)
         *       acc *= correction, sum_exp *= correction
         *       sum_exp += 1, acc += V_t
         *       max_s = s
         *     else:
         *       w = exp(s - max_s)
         *       sum_exp += w, acc += w * V_t
         *   result = acc / sum_exp
         *
         * This saves memory (no att[] buffer) and is more cache-friendly.
         */
        /* Hadamard rotation of Q before attention (Point B) */
        /* For TQ3: Q must be WHT-forward rotated (block=32) to match K encoding.
         * This is done via the standard Hadamard path with nrot=32, which
         * uses the same sign pattern as TQ3's internal WHT. */
        if (s->kv_hadamard_k) {
            for (int h = 0; h < n_heads; h++) {
                picolm_hadamard_transform(s->q + h * head_dim, head_dim, s->kv_hadamard_size);
            }
        } else if (s->kv_type_k == KV_CACHE_TQ3) {
            /* TQ3 K: Q is NOT rotated. We dequant TQ3 to F32 in the
             * attention path, so Q stays in the original domain. */
        } else if (s->kv_type_k == KV_CACHE_TQ4) {
            /* TQ4 K: rotate Q with WHT forward (block=32) to match
             * the codebook-lookup dot product in vec_dot_tq4_f32.
             * Uses the same sign pattern as TQ3. */
            for (int h = 0; h < n_heads; h++) {
                picolm_hadamard_transform(s->q + h * head_dim, head_dim, TQ4_BLOCK_SIZE);
            }
        }

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
        gctx.attn_scale = 1.0f / sqrtf((float)head_dim);

#ifdef PICOLM_GPU
        /* Phase 1: GPU attention decode path */
        if (gpu_ok && m->gpu.kv_active && s->kv_type_k == KV_CACHE_F16 && s->kv_type_v == KV_CACHE_F16) {
            /* Clear GPU tensor before calling attention (no weight tensors involved) */
            tensor_set_gpu_tensor(NULL, 0);
            if (picolm_gpu_attention_decode(s->xb, s->q,
                                             this_attn_ordinal, pos,
                                             n_heads, n_kv_heads, head_dim,
                                             seq_len, gpu_dev)) {
                /* GPU attention succeeded */
            } else {
                tensor_parallel_for(c->n_kv_heads, attention_group, &gctx);
            }
        } else
#endif
        {
            tensor_parallel_for(c->n_kv_heads, attention_group, &gctx);
        }

#ifdef PICOLM_GPU
        /* PICOLM_DBG_ATTN: compare GPU vs CPU attention output */
        {
            const char *dbg_attn = getenv("PICOLM_DBG_ATTN");
            if (dbg_attn && atoi(dbg_attn) && gpu_ok && m->gpu.kv_active && s->kv_type_k == KV_CACHE_F16) {
                float xb_cpu[n_heads * 256];
                memcpy(xb_cpu, s->xb, n_heads * head_dim * sizeof(float));
                tensor_parallel_for(c->n_kv_heads, attention_group, &gctx);
                float max_diff = 0.0f;
                for (int i = 0; i < n_heads * head_dim; i++) {
                    float d = xb_cpu[i] - s->xb[i];
                    if (d < 0) d = -d;
                    if (d > max_diff) max_diff = d;
                }
                if (max_diff > 1e-3f || (pos < 5 && l == 0)) {
                    fprintf(stderr, "[ATTN_DBG decode] layer=%d pos=%d max_diff=%.6f\n", l, pos, max_diff);
                }
                memcpy(s->xb, xb_cpu, n_heads * head_dim * sizeof(float));
            }
        }
#endif


        /* Hadamard rotation of attention output back to original space (Point D) */
        if (s->kv_hadamard_v) {
            for (int h = 0; h < n_heads; h++) {
                picolm_hadamard_transform(s->xb + h * head_dim, head_dim, s->kv_hadamard_size);
            }
        } else if (s->kv_type_v == KV_CACHE_TQ3) {
            /* TQ3 V-path: the scale_add_tq3_f32 already applies inverse WHT
             * internally, so no post-rotation needed. */
        } else if (s->kv_type_v == KV_CACHE_TQ4) {
            /* TQ4 V-path: scale_add_tq4_f32 already applies inverse WHT
             * internally, so no post-rotation needed. */
        }

        /* Qwen3.5 full attention: apply gate sigmoid to attention output */
        if (qwen35_attn_gate) {
            for (int i = 0; i < q_dim; i++) {
                float g = 1.0f / (1.0f + expf(-qwen35_attn_gate[i]));
                s->xb[i] *= g;
            }
        }

        /* Output projection */
        tensor_set_repacked(m->repack_used[ri+3] ? m->repack_buffers[ri+3] : NULL);
#ifdef PICOLM_GPU
        if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->attn_output, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
        matmul(s->xb2, s->xb, lw->attn_output, q_dim, dim, lw->type_attn_output);
        tensor_set_repacked(NULL);
        vec_add(s->x, s->xb2, dim);

        /* ---- FFN (SwiGLU or MoE) - only if MLP weights exist for this layer ---- */
        if (c->has_moe) {
            /* MoE forward pass */
            rmsnorm(s->xb, s->x, s->post_attn_norm_w[l], dim, c->rms_norm_eps);
            moe_forward(m, s, s->xb, s->xb, lw);
            vec_add(s->x, s->xb, dim);
        } else if (lw->ffn_gate && lw->ffn_up && lw->ffn_down) {
            rmsnorm(s->xb, s->x, s->post_attn_norm_w[l], dim, c->rms_norm_eps);

#ifdef PICOLM_GPU
            /* Fused FFN on GPU: y = down(silu(gate(x)) * up(x)) in ONE command
             * buffer -> 3 dispatches collapse to 1 (each layer saves ~2x the
             * per-dispatch sync floor) and silu*mul runs on the GPU. x and y
             * alias s->xb safely: expert_mlp memcpy()s x into device scratch
             * first. On miss (GPU off / a tensor not uploaded / OOM) fall
             * through to the per-matmul path below. */
            if (gpu_ok && gl->ffn_gate && gl->ffn_up && gl->ffn_down &&
                picolm_gpu_expert_mlp((picolm_gpu_tensor_t *)gl->ffn_gate,
                                      (picolm_gpu_tensor_t *)gl->ffn_up,
                                      (picolm_gpu_tensor_t *)gl->ffn_down,
                                      s->xb, s->xb, 1)) {
                tensor_set_gpu_tensor(NULL, 0);
                goto ffn_done;
            }
#endif
            tensor_set_repacked(m->repack_used[ri+4] ? m->repack_buffers[ri+4] : NULL);
#ifdef PICOLM_GPU
            if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->ffn_gate, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
            matmul(s->hb,  s->xb, lw->ffn_gate, dim, n_ffn, lw->type_ffn_gate);
            tensor_set_repacked(NULL);

            tensor_set_repacked(m->repack_used[ri+6] ? m->repack_buffers[ri+6] : NULL);
#ifdef PICOLM_GPU
            if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->ffn_up, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
            matmul(s->hb2, s->xb, lw->ffn_up,   dim, n_ffn, lw->type_ffn_up);
            tensor_set_repacked(NULL);

            silu(s->hb, n_ffn);
            elemwise_mul(s->hb, s->hb, s->hb2, n_ffn);

            tensor_set_repacked(m->repack_used[ri+5] ? m->repack_buffers[ri+5] : NULL);
#ifdef PICOLM_GPU
            if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->ffn_down, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
            matmul(s->xb, s->hb, lw->ffn_down, n_ffn, dim, lw->type_ffn_down);
            tensor_set_repacked(NULL);
#ifdef PICOLM_GPU
ffn_done:
#endif
            vec_add(s->x, s->xb, dim);
        }
#ifdef PICOLM_VIZ
        viz_push_layer(l, s->x, dim);
#endif
        BENCH_LAYER_END(l, 0);
    }

    /* 3. Final RMSNorm */
    rmsnorm(s->x, s->x, s->output_norm_w, dim, c->rms_norm_eps);

    /* 4. Output projection -> logits */
    tensor_set_repacked(m->repack_used[1] ? m->repack_buffers[1] : NULL);
#ifdef PICOLM_GPU
    if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)m->gpu.output, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);
    tensor_set_repacked(NULL);

    return s->logits;
}

void model_free(model_t *m) {
#ifdef PICOLM_GPU
    /* Free GPU weight tensors */
    if (m->gpu.output) { picolm_gpu_tensor_free(m->gpu.output); m->gpu.output = NULL; }
    for (int l = 0; l < m->config.n_layers; l++) {
        gpu_layer_weights_t *gl = &m->gpu.layers[l];
        if (gl->attn_q) { picolm_gpu_tensor_free(gl->attn_q); gl->attn_q = NULL; }
        if (gl->attn_k) { picolm_gpu_tensor_free(gl->attn_k); gl->attn_k = NULL; }
        if (gl->attn_v) { picolm_gpu_tensor_free(gl->attn_v); gl->attn_v = NULL; }
        if (gl->attn_output) { picolm_gpu_tensor_free(gl->attn_output); gl->attn_output = NULL; }
        if (gl->ffn_gate) { picolm_gpu_tensor_free(gl->ffn_gate); gl->ffn_gate = NULL; }
        if (gl->ffn_up) { picolm_gpu_tensor_free(gl->ffn_up); gl->ffn_up = NULL; }
        if (gl->ffn_down) { picolm_gpu_tensor_free(gl->ffn_down); gl->ffn_down = NULL; }
    }
    picolm_gpu_shutdown();
#endif
    /* Free repacked weight buffers */
    for (int i = 0; i < MAX_LAYERS + 4; i++) {
        if (m->repack_buffers[i]) {
            free(m->repack_buffers[i]);
            m->repack_buffers[i] = NULL;
            m->repack_used[i] = 0;
        }
    }

    /* Free on-demand mm_id buffers */
#ifdef _WIN32
    if (m->state.mm_gate_out) { _aligned_free(m->state.mm_gate_out); m->state.mm_gate_out = NULL; }
    if (m->state.mm_up_out)   { _aligned_free(m->state.mm_up_out);   m->state.mm_up_out = NULL; }
    if (m->state.mm_down_out) { _aligned_free(m->state.mm_down_out); m->state.mm_down_out = NULL; }
#else
    if (m->state.mm_gate_out) { free(m->state.mm_gate_out); m->state.mm_gate_out = NULL; }
    if (m->state.mm_up_out)   { free(m->state.mm_up_out);   m->state.mm_up_out = NULL; }
    if (m->state.mm_down_out) { free(m->state.mm_down_out); m->state.mm_down_out = NULL; }
#endif
    m->state.mm_gateup_alloc = 0;
    m->state.mm_down_alloc = 0;

    if (m->state.mem_block) {
        /* mem_block and kv_block are contiguously allocated; release */
#ifdef _WIN32
        free(m->state.mem_block);
#else
        munmap(m->state.mem_block, m->state.mem_size);
#endif
        m->state.mem_block = NULL;
        m->state.kv_block = NULL;
    }
    /* Unmap all split files */
    for (int i = 0; i < m->n_splits; i++) {
        munmap_one_file(&m->splits[i]);
    }
    m->mmap_addr = NULL;
}

/* ================================================================
 * Weight pinning: mlock() a budget of layers so they stay in RAM.
 * On multi-turn conversations, locked layers are never re-streamed
 * from disk.
 * ================================================================ */

/* Compute the byte size of a weight tensor given nrows, ncols, type. */
static size_t weight_tensor_bytes(int nrows, int ncols, gguf_type_t type) {
    size_t row_bytes = gguf_type_row_size(type, ncols);
    return row_bytes * (size_t)nrows;
}

/* Compute the total byte size for one layer's weight tensors. */
static size_t layer_weight_bytes(const model_t *m, int layer) {
    const model_config_t *c = &m->config;
    const layer_weights_t *lw = &m->weights.layers[layer];
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    size_t total = 0;

    /* Norm tensors are 1D vectors of length n_embd (or head_dim for q/k norms).
     * weight_tensor_bytes(1, n, type) = gguf_type_row_size(type, n). */
    if (lw->attn_norm) total += weight_tensor_bytes(1, c->n_embd, lw->type_attn_norm);
    if (lw->attn_q)    total += weight_tensor_bytes(q_dim, c->n_embd, lw->type_attn_q);
    if (lw->attn_k)    total += weight_tensor_bytes(kv_dim, c->n_embd, lw->type_attn_k);
    if (lw->attn_v)    total += weight_tensor_bytes(kv_dim, c->n_embd, lw->type_attn_v);
    if (lw->attn_output) total += weight_tensor_bytes(c->n_embd, q_dim, lw->type_attn_output);
    if (lw->attn_q_norm) total += weight_tensor_bytes(1, c->head_dim, lw->type_attn_q_norm);
    if (lw->attn_k_norm) total += weight_tensor_bytes(1, c->head_dim, lw->type_attn_k_norm);
    if (lw->post_attn_norm) total += weight_tensor_bytes(1, c->n_embd, lw->type_post_attn_norm);
    if (lw->ffn_gate)  total += weight_tensor_bytes(c->n_ffn, c->n_embd, lw->type_ffn_gate);
    if (lw->ffn_up)    total += weight_tensor_bytes(c->n_ffn, c->n_embd, lw->type_ffn_up);
    if (lw->ffn_down)  total += weight_tensor_bytes(c->n_embd, c->n_ffn, lw->type_ffn_down);
    /* SSM weights */
    if (lw->attn_qkv)      total += weight_tensor_bytes(c->ssm_d_inner + 2*c->ssm_d_state*c->ssm_n_group, c->n_embd, lw->type_attn_qkv);
    if (lw->attn_gate_ssm) total += weight_tensor_bytes(c->ssm_d_inner, c->n_embd, lw->type_attn_gate_ssm);
    if (lw->ssm_out)       total += weight_tensor_bytes(c->n_embd, c->ssm_d_inner, lw->type_ssm_out);
    return total;
}

/* Compute the total byte size for global weight tensors. */
static size_t global_weight_bytes(const model_t *m) {
    const model_weights_t *w = &m->weights;
    size_t total = 0;
    if (w->token_embd)
        total += weight_tensor_bytes(m->config.vocab_size, m->config.n_embd, w->type_token_embd);
    if (w->output_norm)
        total += weight_tensor_bytes(1, m->config.n_embd, w->type_output_norm);
    if (w->output && w->output != w->token_embd)
        total += weight_tensor_bytes(m->config.n_embd, m->config.vocab_size, w->type_output);
    return total;
}

/* Round a pointer up to the next page boundary. */
static const uint8_t *page_align_up(const uint8_t *p) {
    uintptr_t addr = (uintptr_t)p;
    return (const uint8_t *)((addr + 4095) & ~(uintptr_t)4095);
}

/* Round a pointer down to the previous page boundary. */
static const uint8_t *page_align_down(const uint8_t *p) {
    uintptr_t addr = (uintptr_t)p;
    return (const uint8_t *)(addr & ~(uintptr_t)4095);
}

/* Lock layer weights in RAM using mlock().
 *
 * Given a budget in bytes, determines how many consecutive layers (starting
 * from layer 0) fit, plus the global tensors, then calls mlock() on the
 * page-aligned range covering all those tensors.
 *
 * Returns the number of layers locked, or 0 on failure. */
int model_lock_layers(model_t *m, size_t mem_bytes) {
    const model_config_t *c = &m->config;

    /* Cap budget to RLIMIT_MEMLOCK minus page alignment overhead.
     * On Windows, VirtualLock has no such limit, so skip this check. */
    size_t effective_budget = mem_bytes;
#ifndef _WIN32
    {
        struct rlimit rl;
        if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) {
            size_t page_overhead = (size_t)(c->n_layers + 1) * 4 * 4096;
            size_t rlimit_budget = rl.rlim_cur - page_overhead;
            if (rlimit_budget < effective_budget)
                effective_budget = rlimit_budget;
        }
    }
#endif

    size_t gbytes = global_weight_bytes(m);
    if (gbytes > effective_budget) {
        fprintf(stderr, "Lock: budget too small for global tensors (%.1f MB)\n",
                (double)gbytes / (1024.0 * 1024.0));
        return 0;
    }

    /* Compute per-layer sizes and find how many fit */
    size_t locked_bytes = gbytes;
    int layers_locked = 0;

    for (int l = 0; l < c->n_layers; l++) {
        size_t lbytes = layer_weight_bytes(m, l);
        if (locked_bytes + lbytes > effective_budget) break;
        locked_bytes += lbytes;
        layers_locked++;
    }

    if (layers_locked == 0) {
        fprintf(stderr, "Lock: budget too small for any layer (%.1f MB needed, %.1f MB available)\n",
                (double)layer_weight_bytes(m, 0) / (1024.0 * 1024.0) + (double)gbytes / (1024.0 * 1024.0),
                (double)mem_bytes / (1024.0 * 1024.0));
        return 0;
    }

    /* Collect all tensor pointers from global + locked layers, find min/max.
     * The GGUF may not store tensors contiguously, so we lock per-tensor
     * ranges individually to avoid locking gaps. */
    int n_ranges = 0;
    const uint8_t *ranges_start[200];
    const uint8_t *ranges_end[200];
    size_t ranges_size[200];
    #define NR_MAX 200

    /* Helper: add a tensor range */
    #define ADD_RANGE(ptr, sz) do { \
        size_t _sz = (sz); \
        if (ptr && _sz > 0 && n_ranges < NR_MAX) { \
            ranges_start[n_ranges] = page_align_down(ptr); \
            ranges_end[n_ranges] = page_align_up((const uint8_t *)ptr + _sz); \
            ranges_size[n_ranges] = (size_t)((const uint8_t *)ranges_end[n_ranges] - (const uint8_t *)ranges_start[n_ranges]); \
            n_ranges++; \
        } \
    } while(0)

    /* Global tensors */
    ADD_RANGE(m->weights.token_embd, weight_tensor_bytes(c->vocab_size, c->n_embd, m->weights.type_token_embd));
    ADD_RANGE(m->weights.output_norm, weight_tensor_bytes(1, c->n_embd, m->weights.type_output_norm));
    if (m->weights.output && m->weights.output != m->weights.token_embd)
        ADD_RANGE(m->weights.output, weight_tensor_bytes(c->n_embd, c->vocab_size, m->weights.type_output));

    /* Locked layers */
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    for (int l = 0; l < layers_locked; l++) {
        const layer_weights_t *lw = &m->weights.layers[l];
        ADD_RANGE(lw->attn_norm, lw->attn_norm ? weight_tensor_bytes(1, c->n_embd, lw->type_attn_norm) : 0);
        ADD_RANGE(lw->attn_q, lw->attn_q ? weight_tensor_bytes(q_dim, c->n_embd, lw->type_attn_q) : 0);
        ADD_RANGE(lw->attn_k, lw->attn_k ? weight_tensor_bytes(kv_dim, c->n_embd, lw->type_attn_k) : 0);
        ADD_RANGE(lw->attn_v, lw->attn_v ? weight_tensor_bytes(kv_dim, c->n_embd, lw->type_attn_v) : 0);
        ADD_RANGE(lw->attn_output, lw->attn_output ? weight_tensor_bytes(c->n_embd, q_dim, lw->type_attn_output) : 0);
        ADD_RANGE(lw->attn_q_norm, lw->attn_q_norm ? weight_tensor_bytes(1, c->head_dim, lw->type_attn_q_norm) : 0);
        ADD_RANGE(lw->attn_k_norm, lw->attn_k_norm ? weight_tensor_bytes(1, c->head_dim, lw->type_attn_k_norm) : 0);
        ADD_RANGE(lw->post_attn_norm, lw->post_attn_norm ? weight_tensor_bytes(1, c->n_embd, lw->type_post_attn_norm) : 0);
        ADD_RANGE(lw->ffn_gate, lw->ffn_gate ? weight_tensor_bytes(c->n_ffn, c->n_embd, lw->type_ffn_gate) : 0);
        ADD_RANGE(lw->ffn_up, lw->ffn_up ? weight_tensor_bytes(c->n_ffn, c->n_embd, lw->type_ffn_up) : 0);
        ADD_RANGE(lw->ffn_down, lw->ffn_down ? weight_tensor_bytes(c->n_embd, c->n_ffn, lw->type_ffn_down) : 0);
    }

    /* Sort ranges by start address (simple insertion sort, N is small) */
    for (int i = 1; i < n_ranges; i++) {
        const uint8_t *s = ranges_start[i];
        const uint8_t *e = ranges_end[i];
        size_t sz = ranges_size[i];
        int j = i - 1;
        while (j >= 0 && ranges_start[j] > s) {
            ranges_start[j+1] = ranges_start[j];
            ranges_end[j+1] = ranges_end[j];
            ranges_size[j+1] = ranges_size[j];
            j--;
        }
        ranges_start[j+1] = s;
        ranges_end[j+1] = e;
        ranges_size[j+1] = sz;
    }

    /* Merge overlapping/adjacent ranges */
    int merged = 1;
    for (int i = 1; i < n_ranges; i++) {
        if ((const uint8_t *)ranges_start[i] <= (const uint8_t *)ranges_end[merged-1]) {
            /* Overlapping or adjacent - extend */
            if ((const uint8_t *)ranges_end[i] > (const uint8_t *)ranges_end[merged-1])
                ranges_end[merged-1] = ranges_end[i];
        } else {
            /* Gap - start new range */
            ranges_start[merged] = ranges_start[i];
            ranges_end[merged] = ranges_end[i];
            merged++;
        }
    }

    /* Clamp range ends to the mmap boundary of the corresponding split.
     * page_align_up can push the end past the file size when the last
     * tensor in a split is near EOF. */
    for (int i = 0; i < merged; i++) {
        /* Find which split this range belongs to */
        for (int si = 0; si < m->n_splits; si++) {
            const uint8_t *s_start = (const uint8_t *)m->splits[si].mmap_addr;
            const uint8_t *s_end = s_start + m->splits[si].mmap_size;
            const uint8_t *s_end_aligned = (const uint8_t *)(((uintptr_t)s_end + 4095) & ~(uintptr_t)4095);
            if (ranges_start[i] >= s_start && ranges_start[i] < s_end) {
                if ((const uint8_t *)ranges_end[i] > s_end_aligned)
                    ranges_end[i] = s_end_aligned;
                break;
            }
        }
    }

    /* On Windows, try to acquire SE_LOCK_MEMORY_NAME privilege.
     * Without it, VirtualLock fails with ERROR_WORKING_SET_QUOTA (1453)
     * because the default working set quota is far too small. */
#ifdef _WIN32
    {
        BOOL got_lock_priv = FALSE;
        HANDLE hToken;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)) {
            TOKEN_PRIVILEGES tp;
            tp.PrivilegeCount = 1;
            if (LookupPrivilegeValueA(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid)) {
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(tp), NULL, NULL);
                if (GetLastError() == ERROR_SUCCESS)
                    got_lock_priv = TRUE;
            }
            CloseHandle(hToken);
        }
        /* If we got the privilege, VirtualLock will work. Otherwise, increase
         * the working set quota as a fallback so VirtualLock doesn't fail with
         * ERROR_WORKING_SET_QUOTA. SetProcessWorkingSetSize with 0x7FFFFFFF
         * means "no limit from Windows" - the OS still respects physical memory. */
        if (!got_lock_priv) {
            /* MAXIMUM_INTEGER32 (0x7FFFFFFF) is the sentinel meaning "no
             * upper limit" for the working set. This does NOT require
             * SE_INCREASE_QUOTA_NAME. Without it, VirtualLock fails with
             * ERROR_WORKING_SET_QUOTA (1453) because the default quota is
             * very small on non-admin accounts. */
            SetProcessWorkingSetSize(GetCurrentProcess(),
                                     (SIZE_T)0x7FFFFFFF, (SIZE_T)0x7FFFFFFF);
        }
    }
#endif

    /* Call mlock on each merged range */
    size_t total_locked = 0;
    for (int i = 0; i < merged; i++) {
        size_t sz = (size_t)((const uint8_t *)ranges_end[i] - (const uint8_t *)ranges_start[i]);
#ifdef _WIN32
        if (VirtualLock((void *)(const void *)ranges_start[i], sz) == 0) {
            fprintf(stderr, "Lock: VirtualLock failed (error %lu)\n", (unsigned long)GetLastError());
            return 0;
        }
#else
        if (mlock(ranges_start[i], sz) != 0) {
            int err = errno;
            if (err == EACCES)
                fprintf(stderr, "Lock: mlock failed - check RLIMIT_MEMLOCK (ulimit -l)\n");
            else
                fprintf(stderr, "Lock: mlock failed: %s\n", strerror(err));
            return 0;
        }
#endif
        total_locked += sz;
    }

    m->locked_layers = layers_locked;
    fprintf(stderr, "Lock: pinned %.1f MB (layers 0..%d of %d)\n",
            (double)total_locked / (1024.0 * 1024.0),
            layers_locked - 1, c->n_layers - 1);
    return layers_locked;
}
#undef ADD_RANGE
#undef NR_MAX

/* Unlock previously pinned weight layers. */
int model_unlock_layers(model_t *m) {
    if (m->locked_layers == 0) return 0;

    const model_config_t *c = &m->config;

    /* Re-compute the ranges (same logic as lock, then munmap) */
    int n_ranges = 0;
    const uint8_t *ranges_start[200];
    const uint8_t *ranges_end[200];
    #define NR_MAX 200
    #define ADD_RANGE(ptr, sz) do { \
        size_t _sz = (sz); \
        if (ptr && _sz > 0 && n_ranges < NR_MAX) { \
            ranges_start[n_ranges] = page_align_down(ptr); \
            ranges_end[n_ranges] = page_align_up((const uint8_t *)ptr + _sz); \
            n_ranges++; \
        } \
    } while(0)

    ADD_RANGE(m->weights.token_embd, weight_tensor_bytes(c->vocab_size, c->n_embd, m->weights.type_token_embd));
    ADD_RANGE(m->weights.output_norm, weight_tensor_bytes(1, c->n_embd, m->weights.type_output_norm));
    if (m->weights.output && m->weights.output != m->weights.token_embd)
        ADD_RANGE(m->weights.output, weight_tensor_bytes(c->n_embd, c->vocab_size, m->weights.type_output));

    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    for (int l = 0; l < m->locked_layers; l++) {
        const layer_weights_t *lw = &m->weights.layers[l];
        ADD_RANGE(lw->attn_norm, lw->attn_norm ? weight_tensor_bytes(1, c->n_embd, lw->type_attn_norm) : 0);
        ADD_RANGE(lw->attn_q, lw->attn_q ? weight_tensor_bytes(q_dim, c->n_embd, lw->type_attn_q) : 0);
        ADD_RANGE(lw->attn_k, lw->attn_k ? weight_tensor_bytes(kv_dim, c->n_embd, lw->type_attn_k) : 0);
        ADD_RANGE(lw->attn_v, lw->attn_v ? weight_tensor_bytes(kv_dim, c->n_embd, lw->type_attn_v) : 0);
        ADD_RANGE(lw->attn_output, lw->attn_output ? weight_tensor_bytes(c->n_embd, q_dim, lw->type_attn_output) : 0);
        ADD_RANGE(lw->attn_q_norm, lw->attn_q_norm ? weight_tensor_bytes(1, c->head_dim, lw->type_attn_q_norm) : 0);
        ADD_RANGE(lw->attn_k_norm, lw->attn_k_norm ? weight_tensor_bytes(1, c->head_dim, lw->type_attn_k_norm) : 0);
        ADD_RANGE(lw->post_attn_norm, lw->post_attn_norm ? weight_tensor_bytes(1, c->n_embd, lw->type_post_attn_norm) : 0);
        ADD_RANGE(lw->ffn_gate, lw->ffn_gate ? weight_tensor_bytes(c->n_ffn, c->n_embd, lw->type_ffn_gate) : 0);
        ADD_RANGE(lw->ffn_up, lw->ffn_up ? weight_tensor_bytes(c->n_ffn, c->n_embd, lw->type_ffn_up) : 0);
        ADD_RANGE(lw->ffn_down, lw->ffn_down ? weight_tensor_bytes(c->n_embd, c->n_ffn, lw->type_ffn_down) : 0);
    }

    /* Merge */
    for (int i = 1; i < n_ranges; i++) {
        const uint8_t *s = ranges_start[i], *e = ranges_end[i];
        int j = i - 1;
        while (j >= 0 && ranges_start[j] > s) {
            ranges_start[j+1] = ranges_start[j];
            ranges_end[j+1] = ranges_end[j];
            j--;
        }
        ranges_start[j+1] = s; ranges_end[j+1] = e;
    }
    int merged = 1;
    for (int i = 1; i < n_ranges; i++) {
        if ((const uint8_t *)ranges_start[i] <= (const uint8_t *)ranges_end[merged-1]) {
            if ((const uint8_t *)ranges_end[i] > (const uint8_t *)ranges_end[merged-1])
                ranges_end[merged-1] = ranges_end[i];
        } else {
            ranges_start[merged] = ranges_start[i];
            ranges_end[merged] = ranges_end[i];
            merged++;
        }
    }

    /* Clamp to mmap boundary (same as lock path) */
    const uint8_t *munlock_mmap_end = (const uint8_t *)m->mmap_addr + m->mmap_size;
    munlock_mmap_end = (const uint8_t *)(((uintptr_t)munlock_mmap_end + 4095) & ~(uintptr_t)4095);
    for (int i = 0; i < merged; i++) {
        if ((const uint8_t *)ranges_end[i] > munlock_mmap_end)
            ranges_end[i] = munlock_mmap_end;
    }

    size_t total_unlocked = 0;
    for (int i = 0; i < merged; i++) {
        size_t sz = (size_t)((const uint8_t *)ranges_end[i] - (const uint8_t *)ranges_start[i]);
#ifdef _WIN32
        VirtualUnlock((void *)(const void *)ranges_start[i], sz);
#else
        munlock(ranges_start[i], sz);
#endif
        total_unlocked += sz;
    }

    m->locked_layers = 0;
    fprintf(stderr, "Unlock: released %.1f MB\n", (double)total_unlocked / (1024.0 * 1024.0));
    return 0;
}

/* ================================================================
 * Batched attention for prefill: computes attention for all tokens
 * and all heads in a single batched operation per layer.
 *
 * Replaces the per-token attention_head() loop with:
 *   1. K@Q^T: batched score computation per head
 *   2. Causal mask + softmax per score row
 *   3. V@softmax: batched weighted V accumulation
 *
 * GQA: each Q head h maps to KV head h/kv_mul.
 * Batch prefill: all prompt tokens processed at once.
 * Projection matmuls batched (weights read once). Attention batched.
 * ================================================================ */

float *model_forward_prefill(model_t *m, const int *tokens, int n_tokens, int start_pos, volatile int *interrupt) {
    /* Gemma-3n: use per-token forward pass for now (batched prefill not yet implemented) */
    if (m->config.is_gemma3n) {
        for (int i = 0; i < n_tokens; i++) {
            if (interrupt && *interrupt) break;
            (void)model_forward(m, tokens[i], start_pos + i);
        }
        return m->state.logits;
    }

    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;
    int dim = c->n_embd, n_ffn = c->n_ffn;
    int bi;
    int n_heads = c->n_heads, n_kv_heads = c->n_kv_heads, head_dim = c->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int q_dim = n_heads * head_dim, seq_len = c->max_seq_len;
    /* Qwen3.5 full attention layers have interleaved Q+gate */
    int q_full_dim = (c->has_ssm) ? (q_dim * 2) : q_dim;
    int max_dim = (q_full_dim > dim) ? q_full_dim : dim;
    size_t bs = (size_t)n_tokens;

    int xb2_stride = dim;
    /* hb/hb2 buffers need to be large enough for FFN (n_ffn) OR gate storage (q_dim) on SSM+attn layers */
    int ffn_buf_size = n_ffn;
    if (q_dim > ffn_buf_size) ffn_buf_size = q_dim;
    size_t sz = bs * (dim + max_dim + dim + q_full_dim + 2 * kv_dim + 2 * ffn_buf_size);
    float *buf = (float *)malloc(sz * sizeof(float));
    if (!buf) { fprintf(stderr, "OOM: prefill batch\n"); exit(1); }
    float *p = buf;
    float *x_batch = p;  p += bs * dim;
    float *xb_batch = p; p += bs * max_dim; /* stride = max_dim (attention needs q_full_dim) */
    float *xb2_batch = p; p += bs * dim;
    float *q_batch = p;  p += bs * q_full_dim;
    float *k_batch = p;  p += bs * kv_dim;
    float *v_batch = p;  p += bs * kv_dim;
    float *hb_batch = p; p += bs * ffn_buf_size;
    float *hb2_batch = p; p += bs * ffn_buf_size;

    /* Embedding lookup */
    {
        if (w->type_token_embd == GGUF_TYPE_Q4_0_4_4) {
            /* Q4_0_4_4: rows are interleaved in groups of 4 */
            int nb = dim / 32;
            size_t group_bytes = (size_t)nb * sizeof(block_q4_0x4);
                        for (bi = 0; bi < n_tokens; bi++) {
                int group = tokens[bi] / 4;
                int r = tokens[bi] % 4;
                const block_q4_0x4 *blocks = (const block_q4_0x4 *)((const uint8_t *)w->token_embd + (size_t)group * group_bytes);
                float *dst = x_batch + bi * dim;
                for (int i = 0; i < nb; i++) {
                    float d = fp16_to_fp32_lookup(blocks[i].d[r]);
                    for (int k = 0; k < 4; k++) {
                        for (int j = 0; j < 4; j++) {
                            uint8_t byte = blocks[i].qs[k * 16 + r * 4 + j];
                            int v0 = (int8_t)(byte << 4) >> 4;
                            int v1 = (int8_t)(byte & 0xF0) >> 4;
                            dst[i * 32 + k * 8 + j * 2]     = d * (float)v0;
                            dst[i * 32 + k * 8 + j * 2 + 1] = d * (float)v1;
                        }
                    }
                }
            }
        } else if (w->type_token_embd == GGUF_TYPE_Q4_0_4_8) {
            /* Q4_0_4_8: rows interleaved in groups of 4, blocklen=8 */
            int nb = dim / 32;
            size_t group_bytes = (size_t)nb * sizeof(block_q4_0x4);
                        for (bi = 0; bi < n_tokens; bi++) {
                int group = tokens[bi] / 4;
                int r = tokens[bi] % 4;
                const block_q4_0x4 *blocks = (const block_q4_0x4 *)((const uint8_t *)w->token_embd + (size_t)group * group_bytes);
                float *dst = x_batch + bi * dim;
                for (int i = 0; i < nb; i++) {
                    float d = fp16_to_fp32_lookup(blocks[i].d[r]);
                    for (int k = 0; k < 2; k++) {
                        for (int j = 0; j < 8; j++) {
                            uint8_t byte = blocks[i].qs[k * 32 + r * 8 + j];
                            int v0 = (int8_t)(byte << 4) >> 4;
                            int v1 = (int8_t)(byte & 0xF0) >> 4;
                            dst[i * 32 + k * 16 + j * 2]     = d * (float)v0;
                            dst[i * 32 + k * 16 + j * 2 + 1] = d * (float)v1;
                        }
                    }
                }
            }
        } else {
            size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
                        for (bi = 0; bi < n_tokens; bi++) {
                const void *er = (const uint8_t *)w->token_embd + (size_t)tokens[bi] * row_bytes;
                dequantize_row(er, x_batch + bi * dim, dim, w->type_token_embd);
            }
        }
    }

    /* Profiling counters */
    int attn_ordinal = 0; /* KV cache ordinal for attention layers (SSM models) */
#ifdef PICOLM_GPU
    int gpu_ok = m->gpu.active;
    int gpu_dev = m->gpu.device;
#endif
    int n_active_layers = c->n_layers - c->n_mtp_layers;
    for (int slot = 0; slot < n_active_layers; slot++) {
        /* Check for client disconnect interrupt */
        if (interrupt && *interrupt) {
            free(buf);
            return NULL;
        }
#ifdef PICOLM_VIZ
        int l = viz_layer_permute(slot);
#else
        int l = slot;
#endif
        layer_weights_t *lw = &w->layers[l];
        BENCH_LAYER_START();

#ifdef PICOLM_VIZ
        /* Layer skip: toggled by VNC mouse click, zero overhead when unused */
        if (viz_layer_skip(l)) {
            /* Pass through: batch stays unchanged, no computation */
#ifdef PICOLM_VIZ
            viz_push_layer(l, x_batch + (n_tokens - 1) * dim, dim);
#endif
            BENCH_LAYER_END(l, 1);
            continue;
        }
#endif

        if (c->has_ssm && !lw->is_attn_layer) {
            if (m->ssm_batched_prefill) {
                /* Batched SSM layer prefill (--ssm-batched) */
#ifdef PICOLM_GPU
                ssm_prefill_layer(m, s, x_batch, xb_batch, xb2_batch,
                                  hb_batch, hb2_batch, lw, l,
                                  n_tokens, start_pos, xb2_stride,
                                  (void **)m->gpu.layers);
#else
                ssm_prefill_layer(m, s, x_batch, xb_batch, xb2_batch,
                                  hb_batch, hb2_batch, lw, l,
                                  n_tokens, start_pos, xb2_stride, NULL);
#endif
            } else {
                /* Per-token SSM fallback (default) - uses s->x/s->xb2 like original */
                for (bi = 0; bi < n_tokens; bi++) {
                    memcpy(s->x, x_batch + bi * dim, dim * sizeof(float));
                    float *ssm_residual = s->xb2;
#ifdef PICOLM_GPU
                    ssm_forward(m, s, s->x, ssm_residual, lw, l, start_pos + bi, &m->gpu.layers[l]);
#else
                    ssm_forward(m, s, s->x, ssm_residual, lw, l, start_pos + bi, NULL);
#endif
                    memcpy(x_batch + bi * dim, s->x, dim * sizeof(float));
                }
            }
#ifdef PICOLM_VIZ
            /* Push the last token in the batch for visualization */
            viz_push_layer(l, x_batch + (n_tokens - 1) * dim, dim);
#endif
            BENCH_LAYER_END(l, 1);
            continue;
        }

        /* RMSNorm */
        for (bi = 0; bi < n_tokens; bi++)
            rmsnorm(xb_batch + bi * dim, x_batch + bi * dim, s->attn_norm_w[l], dim, c->rms_norm_eps);

        /* Q projection (batched) */
        tensor_set_repacked(m->repack_used[2+l*9] ? m->repack_buffers[2+l*9] : NULL);
#ifdef PICOLM_GPU
        if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)m->gpu.layers[l].attn_q, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
        { int this_q_dim = (c->has_ssm && lw->is_attn_layer) ? q_full_dim : q_dim;
          matmul_batch(q_batch, xb_batch, n_tokens, lw->attn_q, dim, this_q_dim, lw->type_attn_q);
          if(_SSM_DBG && l==3){
              int lt=n_tokens-1; double qr=0;for(int _i=0;_i<q_dim;_i++)qr+=q_batch[lt*q_full_dim+_i]*q_batch[lt*q_full_dim+_i];
              fprintf(stderr,"[DBG CPU attn_Q l=%d] last_token_rms=%.6f\n",l,sqrt(qr/q_dim));}
        }
        tensor_set_repacked(NULL);

        /* For Qwen3.5: de-interleave per-head Q+gate into block layout.
         * GGUF stores [Q_0, Gate_0, Q_1, Gate_1, ...] (per-head interleaved).
         * Gate stored in hb2_batch to survive K/V projection writes. */
        { float *qwen35_gate_batch = NULL;
          if (c->has_ssm && lw->is_attn_layer) {
              qwen35_gate_batch = hb2_batch; /* reuse as gate storage */
                            for (bi = 0; bi < n_tokens; bi++) {
                  float *qg_raw = q_batch + bi * q_full_dim;
                  float *q_block = q_batch + bi * q_dim;
                  float *gate_block = qwen35_gate_batch + bi * q_dim;
                  for (int h = 0; h < n_heads; h++) {
                      memmove(q_block + h * head_dim, qg_raw + h * 2 * head_dim,
                              head_dim * sizeof(float));
                      memmove(gate_block + h * head_dim, qg_raw + h * 2 * head_dim + head_dim,
                              head_dim * sizeof(float));
                  }
              }
          }
        }

        /* K+V projection (batched dual) */
        tensor_set_repacked(m->repack_used[3+l*9] ? m->repack_buffers[3+l*9] : NULL);
        matmul_dual_batch(k_batch, v_batch, xb_batch, n_tokens,
                          lw->attn_k, lw->attn_v, dim, kv_dim,
                          lw->type_attn_k, lw->type_attn_v);
        tensor_set_repacked(NULL);

        /* Per-position: RoPE, KV store */
        {
            int this_attn_ord = c->has_ssm ? attn_ordinal++ : l;
            /* GQA layout: [layer][pos] * kv_row_size_gqa */
            uint8_t *kcl = s->key_cache + (size_t)this_attn_ord * seq_len * s->kv_row_size_k;
            uint8_t *vcl = s->val_cache + (size_t)this_attn_ord * seq_len * s->kv_row_size_v;
                        for (bi = 0; bi < n_tokens; bi++) {
                int pos = start_pos + bi;
                float *q_pos = q_batch + bi * q_dim;
                float *k_pos = k_batch + bi * kv_dim;
                float *v_pos = v_batch + bi * kv_dim;

                const float *cos_pos = s->rope_cos + (size_t)pos * (head_dim / 2);
                const float *sin_pos = s->rope_sin + (size_t)pos * (head_dim / 2);

                /* QK-norm (Qwen3) */
                if (lw->attn_q_norm) {
                    float *qnw = s->attn_q_norm_w[l];
                    float *knw = s->attn_k_norm_w[l];
                    int qk_done = 0;
#ifdef PICOLM_GPU
                    if (gpu_ok && m->gpu.active) {
                        if (picolm_gpu_rmsnorm_batched(q_pos, q_pos, qnw, head_dim, c->rms_norm_eps, n_heads, 0, m->gpu.device) &&
                            picolm_gpu_rmsnorm_batched(k_pos, k_pos, knw, head_dim, c->rms_norm_eps, n_kv_heads, 0, m->gpu.device)) {
                            qk_done = 1;
                        }
                    }
#endif
                    if (!qk_done) {
                        for (int h = 0; h < n_heads; h++)
                            rmsnorm(q_pos + h * head_dim, q_pos + h * head_dim, qnw, head_dim, c->rms_norm_eps);
                        for (int h = 0; h < n_kv_heads; h++)
                            rmsnorm(k_pos + h * head_dim, k_pos + h * head_dim, knw, head_dim, c->rms_norm_eps);
                    }
                }

                int rope_dim_pf = (c->rope_dim > 0) ? c->rope_dim : head_dim;
                int rope_half_pf = rope_dim_pf / 2;
                rope(q_pos, k_pos, head_dim, n_heads, n_kv_heads, cos_pos, sin_pos, c->rope_type, rope_half_pf);
                /* Debug: compute score for first Q token, KV pos 0, head 0 */
                if (l == 3 && pos == 0 && _SSM_DBG) {
                    float cpu_score = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                        cpu_score += q_pos[d] * k_pos[d];
                    cpu_score /= sqrtf((float)head_dim);
                    fprintf(stderr, "ATNDBG: CPU l=%d q0[:4]={%.6f,%.6f,%.6f,%.6f} k0[:4]={%.6f,%.6f,%.6f,%.6f} score=%.6f\n",
                            l, q_pos[0], q_pos[1], q_pos[2], q_pos[3],
                            k_pos[0], k_pos[1], k_pos[2], k_pos[3], cpu_score);
                }

                /* KV cache store: GQA row quantization */
                {
                    uint8_t *kp = kcl + (size_t)pos * s->kv_row_size_k;
                    uint8_t *vp = vcl + (size_t)pos * s->kv_row_size_v;
                    /* Hadamard rotation of K before quantization (prefill) */
                    if (s->kv_hadamard_k && s->kv_type_k != KV_CACHE_F16) {
                        for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                            picolm_hadamard_transform(k_pos + hkv * head_dim, head_dim, s->kv_hadamard_size);
                        }
                    }
                    if (s->kv_type_k == KV_CACHE_Q8_0) {
                        quantize_row_q8_0(k_pos, kp, kv_dim);
                    } else if (s->kv_type_k == KV_CACHE_Q4_0) {
                        quantize_row_q4_0(k_pos, kp, kv_dim);
                    } else if (s->kv_type_k == KV_CACHE_TQ3) {
                        for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                            quantize_row_tq3(k_pos + hkv * head_dim,
                                kp + hkv * s->kv_head_stride_k, head_dim);
                        }
                    } else if (s->kv_type_k == KV_CACHE_TQ4) {
                        for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                            quantize_row_tq4(k_pos + hkv * head_dim,
                                kp + hkv * s->kv_head_stride_k, head_dim);
                        }
                    } else {
                        /* FP16: store each head at its offset within the GQA row */
                        for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                            float *k_head = k_pos + hkv * head_dim;
                            uint16_t *kf = (uint16_t *)(kp + hkv * s->kv_head_stride_k);
                            for (int d2 = 0; d2 < head_dim; d2++)
                                kf[d2] = fp32_to_fp16(k_head[d2]);
                        }
                    }
                    /* Phase 1.5: GPU KV store for this chunk is done once,
                     * after the bi loop below, as a single bulk copy -
                     * see "GPU KV cache: bulk store" below. */
                    /* Hadamard rotation of V before quantization (prefill) */
                    if (s->kv_hadamard_v && s->kv_type_v != KV_CACHE_F16) {
                        for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                            picolm_hadamard_transform(v_pos + hkv * head_dim, head_dim, s->kv_hadamard_size);
                        }
                    }
                    if (s->kv_type_v == KV_CACHE_Q8_0) {
                        quantize_row_q8_0(v_pos, vp, kv_dim);
                    } else if (s->kv_type_v == KV_CACHE_Q4_0) {
                        quantize_row_q4_0(v_pos, vp, kv_dim);
                    } else if (s->kv_type_v == KV_CACHE_TQ3) {
                        for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                            quantize_row_tq3(v_pos + hkv * head_dim,
                                vp + hkv * s->kv_head_stride_v, head_dim);
                        }
                    } else if (s->kv_type_v == KV_CACHE_TQ4) {
                        for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                            quantize_row_tq4(v_pos + hkv * head_dim,
                                vp + hkv * s->kv_head_stride_v, head_dim);
                        }
                    } else {
                        for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                            float *v_head = v_pos + hkv * head_dim;
                            uint16_t *vf = (uint16_t *)(vp + hkv * s->kv_head_stride_v);
                            for (int d2 = 0; d2 < head_dim; d2++)
                                vf[d2] = fp32_to_fp16(v_head[d2]);
                        }
                    }
                    }
            }

            /* Phase 1.5: GPU KV cache bulk store. kcl/vcl already hold
             * n_tokens contiguous F16 GQA rows starting at start_pos
             * (identical layout to the device cache) -- one async copy
             * per K/V per chunk, replacing n_tokens*n_kv_heads launches. */
            #ifdef PICOLM_GPU
            if (gpu_ok && m->gpu.kv_active &&
                s->kv_type_k == KV_CACHE_F16 && s->kv_type_v == KV_CACHE_F16) {
                picolm_gpu_kv_store_rows(1, this_attn_ord, start_pos, n_tokens,
                                         kcl + (size_t)start_pos * s->kv_row_size_k,
                                         s->kv_row_size_k, n_kv_heads, head_dim,
                                         seq_len, gpu_dev);
                picolm_gpu_kv_store_rows(0, this_attn_ord, start_pos, n_tokens,
                                         vcl + (size_t)start_pos * s->kv_row_size_v,
                                         s->kv_row_size_v, n_kv_heads, head_dim,
                                         seq_len, gpu_dev);
                }
#endif

            /* Zero init xb_batch for attention accumulation */
            memset(xb_batch, 0, (size_t)n_tokens * max_dim * sizeof(float));

            /* Hadamard rotation of Q before attention (prefill) */
            if (s->kv_hadamard_k) {
                for (bi = 0; bi < n_tokens; bi++) {
                    float *q_pos = q_batch + bi * q_dim;
                    for (int h = 0; h < n_heads; h++) {
                        picolm_hadamard_transform(q_pos + h * head_dim, head_dim, s->kv_hadamard_size);
                    }
                }
            } else if (s->kv_type_k == KV_CACHE_TQ3) {
                /* TQ3 K: Q is NOT rotated (we dequant to F32 in attention path) */
            } else if (s->kv_type_k == KV_CACHE_TQ4) {
                /* TQ4 K: rotate Q with WHT forward (block=32) to match
                 * the codebook-lookup dot product in vec_dot_tq4_f32. */
                for (bi = 0; bi < n_tokens; bi++) {
                    float *q_pos = q_batch + bi * q_dim;
                    for (int h = 0; h < n_heads; h++) {
                        picolm_hadamard_transform(q_pos + h * head_dim, head_dim, TQ4_BLOCK_SIZE);
                    }
                }
            }

            /* Clear GPU tensor: batch_attention_layer's internal matmul_batch
             * calls use KV cache tiles (F16 on CPU), NOT GPU weight tensors.
             * Stale gpu_tensor from attn_q projection would cause the GPU
             * dispatch to use the wrong tensor dimensions and crash. */
#ifdef PICOLM_GPU
            tensor_set_gpu_tensor(NULL, 0);
#endif
            #ifdef PICOLM_GPU
            /* Phase 1: GPU attention prefill path */
            if (gpu_ok && m->gpu.kv_active && s->kv_type_k == KV_CACHE_F16 && s->kv_type_v == KV_CACHE_F16 &&
                s->kv_head_stride_k == s->kv_head_stride_v) {
                if (picolm_gpu_attention_prefill(xb_batch, q_batch,
                                                  this_attn_ord, start_pos, n_tokens,
                                                  n_heads, c->n_kv_heads, head_dim,
                                                  seq_len, gpu_dev)) {
                    /* GPU attention succeeded */
                } else {
                    batch_attention_layer(xb_batch, q_batch, kcl, vcl,
                                          n_tokens, start_pos,
                                          n_heads, c->n_kv_heads, head_dim,
                                          max_dim, (int)s->kv_type_k, (int)s->kv_type_v,
                                          s->kv_row_size_k, s->kv_row_size_v,
                                          s->kv_head_stride_k, s->kv_head_stride_v,
                                          1.0f / sqrtf((float)head_dim));
                }
            } else
#endif
            {
                /* Batched attention: all tokens, all heads, one thread dispatch */
                batch_attention_layer(xb_batch, q_batch, kcl, vcl,
                                      n_tokens, start_pos,
                                      n_heads, c->n_kv_heads, head_dim,
                                      max_dim, (int)s->kv_type_k, (int)s->kv_type_v,
                                      s->kv_row_size_k, s->kv_row_size_v,
                                      s->kv_head_stride_k, s->kv_head_stride_v,
                                      1.0f / sqrtf((float)head_dim));
            }
        }
        /* Dump attn_raw (pre-gate) for comparison */
        if (_SSM_DBG && l == 3) {
            int lt = n_tokens - 1;
            fprintf(stderr, "[DBG CPU attn_raw l=%d] last[:4]={%.6f,%.6f,%.6f,%.6f}\n", l,
                xb_batch[lt*max_dim], xb_batch[lt*max_dim+1], xb_batch[lt*max_dim+2], xb_batch[lt*max_dim+3]);
            /* Per-KV-head RMS */
            fprintf(stderr, "[DBG CPU attn_raw l=3] per-kv-head RMS:");
            for (int kh = 0; kh < c->n_kv_heads; kh++) {
                double r = 0;
                for (int qh = kh * (n_heads / c->n_kv_heads);
                     qh < (kh + 1) * (n_heads / c->n_kv_heads); qh++) {
                    for (int d = 0; d < head_dim; d++)
                        r += xb_batch[lt*max_dim + qh*head_dim + d] *
                             xb_batch[lt*max_dim + qh*head_dim + d];
                }
                r = sqrt(r / (head_dim * (n_heads / c->n_kv_heads)));
                fprintf(stderr, " h%d=%.4f", kh, (float)r);
            }
            fprintf(stderr, "\n");
        }

        /* Hadamard rotation of attention output back to original space (prefill) */
        if (s->kv_hadamard_v) {
            for (bi = 0; bi < n_tokens; bi++) {
                float *xb_pos = xb_batch + bi * max_dim;
                for (int h = 0; h < n_heads; h++) {
                    picolm_hadamard_transform(xb_pos + h * head_dim, head_dim, s->kv_hadamard_size);
                }
            }
        } else if (s->kv_type_v == KV_CACHE_TQ3) {
            /* TQ3 V-path: scale_add_tq3_f32 already applies inverse WHT internally */
        } else if (s->kv_type_v == KV_CACHE_TQ4) {
            /* TQ4 V-path: scale_add_tq4_f32 already applies inverse WHT internally */
        }

        /* Dump gate values for comparison */
        if (_SSM_DBG && l == 3) {
            int lt = n_tokens - 1;
            fprintf(stderr, "[DBG CPU attn_gate l=%d] last[:4]={%.6f,%.6f,%.6f,%.6f}\n", l, hb2_batch[lt*q_dim], hb2_batch[lt*q_dim+1], hb2_batch[lt*q_dim+2], hb2_batch[lt*q_dim+3]);
        }
        /* Dump attention output for comparison - BEFORE gate sigmoid */
        if (_SSM_DBG && l == 3) {
            int lt = n_tokens - 1;
            fprintf(stderr,"[DBG CPU attn_O l=3 per-head RMS (pre-gate): ");
            for(int h=0;h<n_heads;h++){double hr=0;for(int d=0;d<head_dim;d++)hr+=xb_batch[lt*max_dim+h*head_dim+d]*xb_batch[lt*max_dim+h*head_dim+d];fprintf(stderr,"h%d=%.4f ",h,sqrt(hr/head_dim));}
            fprintf(stderr,"\n");
        }
        /* Apply Qwen3.5 attention gate (sigmoid, element-wise multiply on attention output before proj) */
        if (c->has_ssm && lw->is_attn_layer) {
                        for (bi = 0; bi < n_tokens; bi++) {
                float *xb = xb_batch + bi * max_dim;
                float *gate = hb2_batch + bi * q_dim;
                for (int i = 0; i < q_dim; i++) {
                    float g = 1.0f / (1.0f + expf(-gate[i]));
                    xb[i] *= g;
                }
            }
        }
        /* After gate sigmoid */
        if (_SSM_DBG && l == 3) {
            int lt = n_tokens - 1;
            double cr2=0;for(int _i=0;_i<q_dim;_i++)cr2+=xb_batch[lt*max_dim+_i]*xb_batch[lt*max_dim+_i];
            fprintf(stderr, "[DBG CPU attn_gated l=%d] last[:4]={%.6f,%.6f,%.6f,%.6f} token_rms=%.6f\n", l, xb_batch[lt*max_dim], xb_batch[lt*max_dim+1], xb_batch[lt*max_dim+2], xb_batch[lt*max_dim+3], sqrt(cr2/q_dim));
        }
        /* Output projection (batched) */
        /* Note: xb_batch has stride max_dim, but attention output is q_dim wide.
         * matmul_batch requires stride == n, so we copy to compact buffer. */
        { float *attn_out_batch = NULL;
          if (max_dim > q_dim) {
              attn_out_batch = (float *)malloc((size_t)n_tokens * q_dim * sizeof(float));
              for (bi = 0; bi < n_tokens; bi++)
                  memcpy(attn_out_batch + bi * q_dim, xb_batch + bi * max_dim, q_dim * sizeof(float));
          }
          tensor_set_repacked(m->repack_used[6+l*9] ? m->repack_buffers[6+l*9] : NULL);
#ifdef PICOLM_GPU
          if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)m->gpu.layers[l].attn_output, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
          matmul_batch(xb2_batch, attn_out_batch ? attn_out_batch : xb_batch, n_tokens, lw->attn_output, q_dim, dim, lw->type_attn_output);
          if (attn_out_batch) free(attn_out_batch);
          if (_SSM_DBG && l == 3) {
              int lt = n_tokens - 1;
              fprintf(stderr, "[DBG CPU outproj l=%d] last[:4]={%.6f,%.6f,%.6f,%.6f}\n", l, xb2_batch[lt*dim], xb2_batch[lt*dim+1], xb2_batch[lt*dim+2], xb2_batch[lt*dim+3]);
          }
        }
                tensor_set_repacked(NULL);

        /* Residual: x += attn_out */
        for (bi = 0; bi < n_tokens; bi++) {
            float *a = x_batch + bi * dim, *b = xb2_batch + bi * dim;
            for (int d2 = 0; d2 < dim; d2++) a[d2] += b[d2];
        }

        /* FFN (MoE or dense) */
        if (c->has_moe) {
            for (bi = 0; bi < n_tokens; bi++) {
                rmsnorm(xb_batch + bi * dim, x_batch + bi * dim, s->post_attn_norm_w[l], dim, c->rms_norm_eps);
            }
            moe_forward_batch(m, s, xb_batch, xb2_batch, n_tokens, lw);
            for (bi = 0; bi < n_tokens; bi++) {
                float *a = x_batch + bi * dim, *b = xb2_batch + bi * dim;
                for (int d2 = 0; d2 < dim; d2++) a[d2] += b[d2];
            }
        } else {
            /* FFN RMSNorm */
            for (bi = 0; bi < n_tokens; bi++)
                rmsnorm(xb_batch + bi * dim, x_batch + bi * dim, s->post_attn_norm_w[l], dim, c->rms_norm_eps);

            /* FFN gate+up (batched dual) */
            tensor_set_repacked(m->repack_used[7+l*9] ? m->repack_buffers[7+l*9] : NULL);
            matmul_dual_batch(hb_batch, hb2_batch, xb_batch, n_tokens,
                              lw->ffn_gate, lw->ffn_up, dim, n_ffn,
                              lw->type_ffn_gate, lw->type_ffn_up);
            tensor_set_repacked(NULL);

            /* SiLU + mul */
            for (bi = 0; bi < n_tokens; bi++) {
                silu(hb_batch + bi * n_ffn, n_ffn);
                elemwise_mul(hb_batch + bi * n_ffn, hb_batch + bi * n_ffn, hb2_batch + bi * n_ffn, n_ffn);
            }

            /* FFN down (batched) */
            tensor_set_repacked(m->repack_used[8+l*9] ? m->repack_buffers[8+l*9] : NULL);
#ifdef PICOLM_GPU
            if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)m->gpu.layers[l].ffn_down, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
            matmul_batch(xb2_batch, hb_batch, n_tokens, lw->ffn_down, n_ffn, dim, lw->type_ffn_down);
            tensor_set_repacked(NULL);

            /* Residual: x += ffn_out */
            for (bi = 0; bi < n_tokens; bi++) {
                float *a = x_batch + bi * dim, *b = xb2_batch + bi * dim;
                for (int d2 = 0; d2 < dim; d2++) a[d2] += b[d2];
            }
        }
#ifdef PICOLM_VIZ
        viz_push_layer(l, x_batch + (n_tokens - 1) * dim, dim);
#endif
        if (_SSM_DBG && lw->is_attn_layer) {
            int lt = n_tokens - 1;
            fprintf(stderr, "[DBG CPU attn l=%d] bx_last[:4]={%.6f,%.6f,%.6f,%.6f}\n", l, x_batch[lt*dim], x_batch[lt*dim+1], x_batch[lt*dim+2], x_batch[lt*dim+3]);
        }
        BENCH_LAYER_END(l, 1);
    }

    /* Final norm + output (last token only) */
    float *last_x = x_batch + (n_tokens - 1) * dim;
        rmsnorm(s->x, last_x, s->output_norm_w, dim, c->rms_norm_eps);
    tensor_set_repacked(m->repack_used[1] ? m->repack_buffers[1] : NULL);
#ifdef PICOLM_GPU
    if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)m->gpu.output, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);
    tensor_set_repacked(NULL);

    /* Clear GPU tensor handle so generation path doesn't inherit stale handle */
#ifdef PICOLM_GPU
    tensor_set_gpu_tensor(NULL, 0);
#endif
    free(buf);
    return s->logits;
}

