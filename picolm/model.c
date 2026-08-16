#include "model.h"
#include "tensor.h"
#include "quant.h"
#include <inttypes.h>

#if defined(__APPLE__) && defined(__ppc__) && defined(__ALTIVEC__)
#include <altivec.h>
#undef bool
#undef pixel
#undef vec_add
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
extern void picolm_gpu_debug_tensor(const char *name, void *tensor, int device, int layer, int dump_weights);

#if defined(__APPLE__) && defined(__ppc__) && defined(__ALTIVEC__)
/* Swap uint16 values in-place using Altivec vec_perm.
 * Uses static 64-byte aligned buffer to avoid alignment issues. */
static char _f16swap_buf[64] __attribute__((aligned(64)));
static unsigned char _f16swap_mask[16] __attribute__((aligned(64))) =
    {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14};
static vector unsigned char _f16swap_vmask;

static void swap_f16_block(uint16_t *dst, size_t n) {
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
static inline float hreduce256_ps(__m256 a) {
    __m256 t1 = _mm256_permute2f128_ps(a, a, 1);
    __m256 t2 = _mm256_add_ps(a, t1);
    __m128 t3 = _mm256_castps256_ps128(t2);
    t3 = _mm_add_ss(t3, _mm_movehl_ps(t3, t3));
    t3 = _mm_add_ss(t3, _mm_shuffle_ps(t3, t3, 0x1));
    return _mm_cvtss_f32(t3);
}
#endif /* PICOLM_AVX2 || PICOLM_AVX */

static void repack_model_weights_q4_0x8(model_t *m);

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef HAVE_WINDOWS_H
#include <windows.h>
#endif
#endif

/* ---- GGUF metadata value types ---- */
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

/* ---- Helpers for reading GGUF binary format ---- */

typedef struct {
    const uint8_t *data;
    size_t pos;
    size_t size;
} reader_t;

static uint8_t read_u8(reader_t *r) {
    uint8_t v = r->data[r->pos];
    r->pos += 1;
    return v;
}

static uint16_t read_u16(reader_t *r) {
    uint16_t v;
    memcpy(&v, r->data + r->pos, 2);
    r->pos += 2;
    return v;
}

static uint32_t read_u32(reader_t *r) {
    uint32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return GGUF_LE32(v);
}

static int32_t read_i32(reader_t *r) {
    int32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return (int32_t)GGUF_LE32((uint32_t)v);
}

static uint64_t read_u64(reader_t *r) {
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return GGUF_LE64(v);
}

static float read_f32(reader_t *r) {
    uint32_t vi;
    memcpy(&vi, r->data + r->pos, 4);
    r->pos += 4;
    vi = GGUF_LE32(vi);
    float v;
    memcpy(&v, &vi, 4);
    return v;
}

typedef struct { const char *str; uint64_t len; } gguf_str_t;

static gguf_str_t read_gguf_string(reader_t *r) {
    gguf_str_t s;
    s.len = read_u64(r);
    s.str = (const char *)(r->data + r->pos);
    r->pos += s.len;
    return s;
}

static int str_eq(gguf_str_t s, const char *lit) {
    size_t n = strlen(lit);
    return s.len == n && memcmp(s.str, lit, n) == 0;
}

/* Forward declarations */
static uint64_t skip_meta_value(reader_t *r, uint32_t vtype, int *is_numeric);
static int gguf_format_value(char *buf, int buflen, reader_t *r, uint32_t vtype);

/* Format a single scalar GGUF metadata value into buf (up to buflen-1 chars).
 * Returns number of chars written (not counting null terminator). */
static int gguf_format_value(char *buf, int buflen, reader_t *r, uint32_t vtype) {
    switch (vtype) {
        case GGUF_META_UINT8:   return snprintf(buf, buflen, "%u", read_u8(r));
        case GGUF_META_INT8:    return snprintf(buf, buflen, "%d", (int8_t)read_u8(r));
        case GGUF_META_UINT16:  return snprintf(buf, buflen, "%u", read_u16(r));
        case GGUF_META_INT16:   return snprintf(buf, buflen, "%d", (int16_t)read_u16(r));
        case GGUF_META_UINT32:  return snprintf(buf, buflen, "%u", read_u32(r));
        case GGUF_META_INT32:   return snprintf(buf, buflen, "%d", read_i32(r));
        case GGUF_META_UINT64:  return snprintf(buf, buflen, "%" PRIu64, read_u64(r));
        case GGUF_META_INT64:   return snprintf(buf, buflen, "%" PRId64, (int64_t)read_u64(r));
        case GGUF_META_FLOAT32: return snprintf(buf, buflen, "%g", (double)read_f32(r));
        case GGUF_META_FLOAT64: { uint64_t raw = read_u64(r); double d; memcpy(&d, &raw, 8);
#if defined(__APPLE__) && defined(__ppc__)
            /* Big-endian: GGUF stores LE */
            { uint64_t sw; memcpy(&sw, &raw, 8); sw = GGUF_LE64(sw); memcpy(&d, &sw, 8); }
#endif
            return snprintf(buf, buflen, "%g", d); }
        case GGUF_META_BOOL:    return snprintf(buf, buflen, "%s", read_u8(r) ? "true" : "false");
        case GGUF_META_STRING: {
            gguf_str_t s = read_gguf_string(r);
            int n = (int)s.len < buflen - 1 ? (int)s.len : buflen - 1;
            memcpy(buf, s.str, (size_t)n);
            buf[n] = '\0';
            return n;
        }
        case GGUF_META_ARRAY: {
            uint32_t arr_type = read_u32(r);
            uint64_t arr_len  = read_u64(r);
            if (arr_len == 0) { buf[0] = '['; buf[1] = ']'; buf[2] = '\0'; return 2; }
            /* Sample first few elements, skip the rest */
            int written = snprintf(buf, buflen, "[");
            uint64_t show = arr_len < 4 ? arr_len : 3;
            for (uint64_t i = 0; i < show; i++) {
                char valbuf[64];
                gguf_format_value(valbuf, (int)sizeof(valbuf), r, arr_type);
                written += snprintf(buf + written, buflen - written, "%s%s",
                    i > 0 ? ", " : "", valbuf);
            }
            /* Skip remaining elements */
            { int dummy;
              for (uint64_t i = show; i < arr_len; i++) {
                  skip_meta_value(r, arr_type, &dummy);
              } }
            written += snprintf(buf + written, buflen - written,
                arr_len > 4 ? ", ..%lu..]" : "]", (unsigned long)(arr_len - show));
            return written;
        }
        default: return snprintf(buf, buflen, "<type %u>", vtype);
    }
}

static uint64_t skip_meta_value(reader_t *r, uint32_t vtype, int *is_numeric) {
    *is_numeric = 1;
    switch (vtype) {
        case GGUF_META_UINT8:   return read_u8(r);
        case GGUF_META_INT8:    return (uint64_t)(int64_t)(int8_t)read_u8(r);
        case GGUF_META_UINT16:  return read_u16(r);
        case GGUF_META_INT16:   return (uint64_t)(int64_t)(int16_t)read_u16(r);
        case GGUF_META_UINT32:  return read_u32(r);
        case GGUF_META_INT32:   return (uint64_t)(int64_t)read_i32(r);
        case GGUF_META_UINT64:  return read_u64(r);
        case GGUF_META_INT64:   return read_u64(r);
        case GGUF_META_FLOAT32: { read_f32(r); *is_numeric = 0; return 0; }
        case GGUF_META_FLOAT64: { r->pos += 8; *is_numeric = 0; return 0; }
        case GGUF_META_BOOL:    return read_u8(r);
        case GGUF_META_STRING:  { read_gguf_string(r); *is_numeric = 0; return 0; }
        case GGUF_META_ARRAY: {
            *is_numeric = 0;
            uint32_t arr_type = read_u32(r);
            uint64_t arr_len  = read_u64(r);
            int dummy;
            for (uint64_t i = 0; i < arr_len; i++) {
                skip_meta_value(r, arr_type, &dummy);
            }
            return 0;
        }
        default:
            fprintf(stderr, "Unknown GGUF metadata type: %u\n", vtype);
            exit(1);
    }
}

/* ---- mmap abstraction ---- */

/* Two modes for mmap'd model loading:

 * 1. Default (PICO mode): bare mmap, no hints. The OS pages in weights
 *    on demand during inference. Uses minimal RAM - can run 10B models
 *    on 256MB RAM. Prefill is slower due to page faults.
 *
 * 2. Prefault mode (-DPICOLM_PREFAULT): touch every page at load time
 *    to bring the entire model into the page cache. Uses model-size RAM
 *    but eliminates page-fault overhead during inference.
 *
 * The prefault loop touches one byte per 4KB page. A 1GB model has
 * ~250K pages and takes ~2-3 seconds to fault in at load time. */

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
static void _do_prefault(const void *addr, size_t size) {
    size_t pages = (size + 4095) / 4096;
    const volatile char *p = (const volatile char *)addr;
    for (size_t off = 0; off < size; off += 4096)
        (void)p[off];
    fprintf(stderr, "Prefaulted %zu pages (%.1f MB)\n", pages, (double)size / (1024.0 * 1024.0));
}

static void prepare_mmap(const void *addr, size_t size) {
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

/* Forward declarations for split mmap helpers */
static int mmap_one_file(split_mmap_t *s, const char *path);
static void munmap_one_file(split_mmap_t *s);
static int split_path_prefix(char *prefix, size_t maxlen, const char *split_path);
static void split_path_build(char *path, size_t maxlen, const char *prefix, int split_no, int split_count);

static int mmap_file(model_t *m, const char *path) {
    /* Store path for split derivation later */
    strncpy(m->first_split_path, path, sizeof(m->first_split_path) - 1);
    m->first_split_path[sizeof(m->first_split_path) - 1] = '\0';

    /* Mmap as split 0 */
    if (mmap_one_file(&m->splits[0], path) != 0) return -1;

    /* Backward compat: legacy fields point to split 0 */
    m->mmap_addr = m->splits[0].mmap_addr;
    m->mmap_size = m->splits[0].mmap_size;
#ifdef _WIN32
    m->file_handle = m->splits[0].file_handle;
    m->map_handle  = m->splits[0].map_handle;
#else
    m->fd = m->splits[0].fd;
#endif
    m->n_splits = 1;

    prepare_mmap(m->mmap_addr, m->mmap_size);
    return 0;
}


/* ================================================================
 * Split GGUF support
 * ================================================================
 *
 * GGUF split files use the naming convention:
 *   <prefix>-NNNNN-of-NNNNN.gguf  (1-based, zero-padded to 5 digits)
 *
 * Each split is a complete, valid GGUF file. Only the first split
 * (index 0) contains full metadata. Tensor offsets are per-file.
 */

#define MAX_SPLIT_FILES 64

/* Extract the prefix from a split file path.
 * Pattern: <prefix>-DDDDD-of-DDDDD.gguf (e.g. "/path/model-Q4_0-split-00001-of-00003.gguf")
 * Returns the prefix part (everything before "-DDDDD-of-").
 * Returns 1 on success, 0 on failure (path doesn't match split convention). */
static int split_path_prefix(char *prefix, size_t maxlen, const char *split_path) {
    size_t plen = strlen(split_path);
    /* Suffix is: -DDDDD-of-DDDDD.gguf = 20 chars. Minimum total path: 24 chars. */
    if (plen < 24) return 0;

    const char *suf = split_path + plen - 20;
    /* Must end with .gguf */
    if (strcmp(suf + 15, ".gguf") != 0) return 0;
    /* First char must be dash */
    if (suf[0] != '-') return 0;
    /* DDDDD-of-DDDDD: positions 1-5 are digits, 6-9 are "-of-", 10-14 are digits */
    for (int i = 1; i <= 5; i++) {
        if (suf[i] < '0' || suf[i] > '9') return 0;
    }
    if (suf[6] != '-' || suf[7] != 'o' || suf[8] != 'f' || suf[9] != '-') return 0;
    for (int i = 10; i <= 14; i++) {
        if (suf[i] < '0' || suf[i] > '9') return 0;
    }

    size_t plen2 = (size_t)(suf - split_path);
    if (plen2 >= maxlen) return 0;
    memcpy(prefix, split_path, plen2);
    prefix[plen2] = '\0';
    return 1;
}

/* Construct a split file path from prefix, split_no (0-based), and split_count. */
static void split_path_build(char *path, size_t maxlen, const char *prefix, int split_no, int split_count) {
    /* Suffix "-NNNNN-of-NNNNN.gguf" is exactly 18 chars */
    if (maxlen < 19) { path[0] = '\0'; return; }
    int plen = snprintf(path, maxlen, "%s", prefix);
    snprintf(path + plen, maxlen - (size_t)plen, "-%05d-of-%05d.gguf", split_no + 1, split_count);
}

/* Mmap a single split file into a split_mmap_t struct. */
static int mmap_one_file(split_mmap_t *s, const char *path) {
#ifdef _WIN32
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Cannot open split file: %s\n", path);
        return -1;
    }
    LARGE_INTEGER fsize;
    GetFileSizeEx(fh, &fsize);
    s->mmap_size = (size_t)fsize.QuadPart;
    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) {
        fprintf(stderr, "CreateFileMapping failed for: %s\n", path);
        CloseHandle(fh);
        return -1;
    }
    void *addr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        fprintf(stderr, "MapViewOfFile failed for: %s\n", path);
        CloseHandle(mh);
        CloseHandle(fh);
        return -1;
    }
    s->mmap_addr  = addr;
    s->file_handle = fh;
    s->map_handle  = mh;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open split file: %s\n", path);
        return -1;
    }
    struct stat st;
    fstat(fd, &st);
    s->mmap_size = (size_t)st.st_size;
    void *addr = mmap(NULL, s->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "mmap failed for split file %s: %s\n", path, strerror(errno));
        close(fd);
        return -1;
    }
    s->mmap_addr = addr;
    s->fd = fd;
#endif
    return 0;
}

/* Unmap a single split file. */
static void munmap_one_file(split_mmap_t *s) {
    if (!s->mmap_addr) return;
#ifdef _WIN32
    UnmapViewOfFile(s->mmap_addr);
    CloseHandle(s->map_handle);
    CloseHandle(s->file_handle);
#else
    munmap(s->mmap_addr, s->mmap_size);
    close(s->fd);
#endif
    s->mmap_addr = NULL;
}

/* ---- Tensor listing ---- */

static const char *gguf_type_name(uint32_t type) {
    switch (type) {
        case 0:  return "f32";
        case 1:  return "f16";
        case 2:  return "q4_0";
        case 3:  return "q4_1";
        case 6:  return "q5_0";
        case 7:  return "q5_1";
        case 8:  return "q8_0";
        case 9:  return "q8_1";
        case 10: return "q2_k";
        case 11: return "q3_k";
        case 12: return "q4_k";
        case 13: return "q5_k";
        case 14: return "q6_k";
        case 15: return "q8_k";
        case 16: return "iq2_xxs";
        case 17: return "iq2_xs";
        case 18: return "iq3_xxs";
        case 19: return "iq1_s";
        case 20: return "iq4_xxs";
        case 21: return "iq3_s";
        case 22: return "iq2_s";
        case 23: return "iq4_xs";
        case 24: return "iq1_m";
        case 25: return "iq6_xxs";
        case 26: return "iq4_nl";
        case 27: return "i8";
        case 28: return "u8";
        case 29: return "i4";
        case 30: return "bf16";
        case 31: return "q4_0_4_4";
        case 32: return "q4_0_4_8";
        case 33: return "q4_0_8_8";
        case 34: return "tq1_0";
        case 35: return "tq2_0";
        case 36: return "mxfp4";
        case 37: return "q8_4";
        case 38: return "q3_0";
        case 39: return "q4_3";
        case 40: return "nvfp4";
        case 41: return "q1_0";
        case 42: return "q2_0";
        /* ik_llama repacked types */
        case 202: return "q4_0_r8";
        case 208: return "q8_0_r8";
        case 212: return "q4_k_r4";
        case 213: return "q5_k_r4";
        case 214: return "q6_k_r4";
        default: return "unknown";
    }
}

int model_list_tensors(const char *path) {
    uint8_t *addr = NULL;
    size_t fsize = 0;
    char *buf = NULL;
    int rc = -1;
#ifdef _WIN32
    HANDLE fh = INVALID_HANDLE_VALUE;
    HANDLE mh = NULL;
#else
    int fd = -1;
#endif

#ifdef _WIN32
    fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        return -1;
    }

    { LARGE_INTEGER ls;
      GetFileSizeEx(fh, &ls);
      fsize = (size_t)ls.QuadPart; }

    mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) {
        fprintf(stderr, "CreateFileMapping failed\n");
        goto out;
    }

    addr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        fprintf(stderr, "MapViewOfFile failed\n");
        goto out;
    }
#else
    fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }

    { struct stat st;
      if (fstat(fd, &st) < 0) goto out;
      addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
      if (addr == MAP_FAILED) { addr = NULL; goto out; }
      fsize = (size_t)st.st_size; }
#endif

    reader_t r = { .data = addr, .pos = 0, .size = fsize };

    { uint32_t magic = read_u32(&r);
      if (magic != GGUF_MAGIC) {
          fprintf(stderr, "Invalid GGUF magic: 0x%08X\n", magic);
          goto out;
      } }

    uint32_t version = read_u32(&r);
    if (version < 2 || version > 3) {
        fprintf(stderr, "Unsupported GGUF version: %u (only v2/v3 supported)\n", version);
        goto out;
    }
    uint64_t n_tensors = read_u64(&r);
    uint64_t n_metadata = read_u64(&r);

    /* Skip metadata */
    for (uint64_t i = 0; i < n_metadata; i++) {
        (void)read_gguf_string(&r);
        uint32_t vtype = read_u32(&r);
        int dummy;
        skip_meta_value(&r, vtype, &dummy);
    }

    /*
     * Build the entire output (GGUF header + tensor table) into a buffer,
     * then write it in a single fwrite() to avoid the "bit-bang"
     * one-write-per-line problem when stdout is redirected to a slow
     * destination (e.g. network share).
     */
    { size_t buf_size = 128 + (n_tensors + 2) * 96;
      buf = malloc(buf_size);
      if (!buf) {
          fprintf(stderr, "error: out of memory\n");
          goto out;
      }

      size_t pos = 0;
      /* Safe append macro: clamps on truncation */
      #define APPEND(fmt, ...) do {                                          \
          int _r = snprintf(buf + pos, buf_size - pos, fmt, __VA_ARGS__);    \
          if (_r > 0) pos += (size_t)_r < (buf_size - pos) ? (size_t)_r : 0; \
      } while (0)

      APPEND("GGUF v%u: %" PRIu64 " metadata entries, %" PRIu64 " tensors\n\n",
             version, n_metadata, n_tensors);
      APPEND("%-52s %14s %-12s %s\n", "Name", "Shape", "Type", "Type ID");
      APPEND("%-52s %14s %-12s %s\n", "----", "----", "----", "-------");

      for (uint64_t i = 0; i < n_tensors; i++) {
          gguf_str_t name = read_gguf_string(&r);
          uint32_t n_dims = read_u32(&r);
          uint64_t dims[4] = {0};
          for (uint32_t d = 0; d < n_dims; d++) dims[d] = read_u64(&r);
          uint32_t type = read_u32(&r);
          (void)read_u64(&r); /* offset */

          char nbuf[56];
          size_t nlen = name.len < sizeof(nbuf) ? name.len : sizeof(nbuf);
          memcpy(nbuf, name.str, nlen);
          nbuf[nlen] = '\0';
          if (name.len >= sizeof(nbuf)) {
              nbuf[nlen-4] = '.'; nbuf[nlen-3] = '.'; nbuf[nlen-2] = '.';
          }

          /* Build shape string with pointer arithmetic instead of strncat */
          char dstr[16];
          char *dp = dstr;
          int drem = (int)(sizeof(dstr) - 1);
          int n = snprintf(dp, drem + 1, "[%" PRIu64, dims[0]);
          if (n > 0 && n < drem) { dp += n; drem -= n; }
          for (uint32_t d = 1; d < n_dims && drem > 1; d++) {
              n = snprintf(dp, drem + 1, ",%" PRIu64, dims[d]);
              if (n > 0 && n < drem) { dp += n; drem -= n; } else break;
          }
          if (drem > 0) { *dp = ']'; dp++; }
          *dp = '\0';

          APPEND("%-52s %14s %-12s %u\n", nbuf, dstr, gguf_type_name(type), type);
      }
      #undef APPEND

      fwrite(buf, 1, pos, stdout);
      fflush(stdout);
    }

    rc = 0;
out:
    free(buf);
#ifdef _WIN32
    if (addr) UnmapViewOfFile(addr);
    if (mh) CloseHandle(mh);
    if (fh != INVALID_HANDLE_VALUE) CloseHandle(fh);
#else
    if (addr) munmap(addr, fsize);
    if (fd >= 0) close(fd);
#endif
    return rc;
}

/* ---- GGUF KV List ---- */

int model_list_kv(const char *path) {
    uint8_t *addr = NULL;
    size_t fsize = 0;
    char *buf = NULL;
    int rc = -1;
#ifdef _WIN32
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Cannot open '%s'\n", path);
        return -1;
    }
    { LARGE_INTEGER sz;
      GetFileSizeEx(fh, &sz);
      fsize = (size_t)sz.QuadPart; }
    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mh) {
        addr = (uint8_t *)MapViewOfFile(mh, FILE_MAP_READ, 0, 0, fsize);
    }
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }
    { struct stat st;
      if (fstat(fd, &st) < 0) goto out;
      addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
      if (addr == MAP_FAILED) { addr = NULL; goto out; }
      fsize = (size_t)st.st_size; }
#endif

    if (!addr) {
        fprintf(stderr, "Failed to map '%s'\n", path);
        goto out;
    }

    reader_t r = { .data = addr, .pos = 0, .size = fsize };

    { uint32_t magic = read_u32(&r);
      if (magic != GGUF_MAGIC) {
          fprintf(stderr, "Invalid GGUF magic: 0x%08X\n", magic);
          goto out;
      } }

    uint32_t version = read_u32(&r);
    if (version < 2 || version > 3) {
        fprintf(stderr, "Unsupported GGUF version: %u (only v2/v3 supported)\n", version);
        goto out;
    }
    uint64_t n_tensors = read_u64(&r);
    uint64_t n_metadata = read_u64(&r);

    /* Buffer: key + value per entry, plus header */
    { size_t buf_size = 1024 + n_metadata * 128;
      buf = malloc(buf_size);
      if (!buf) {
          fprintf(stderr, "error: out of memory\n");
          goto out;
      }

      size_t pos = 0;
      #define APPEND(fmt, ...) do {                                          \
          int _r = snprintf(buf + pos, buf_size - pos, fmt, __VA_ARGS__);    \
          if (_r > 0) pos += (size_t)_r < (buf_size - pos) ? (size_t)_r : 0; \
      } while (0)

      APPEND("GGUF v%u: %" PRIu64 " metadata entries, %" PRIu64 " tensors\n\n",
             version, n_metadata, n_tensors);
      APPEND("%-50s %s\n", "Key", "Value");
      APPEND("%-50s %s\n", "---", "-----");

      for (uint64_t i = 0; i < n_metadata; i++) {
          gguf_str_t key = read_gguf_string(&r);
          uint32_t vtype = read_u32(&r);

          /* Truncate key for display */
          char keybuf[54];
          size_t klen = key.len < sizeof(keybuf) - 1 ? key.len : sizeof(keybuf) - 4;
          memcpy(keybuf, key.str, klen);
          keybuf[klen] = '\0';
          if (key.len >= sizeof(keybuf) - 1) {
              keybuf[klen-3] = '.'; keybuf[klen-2] = '.'; keybuf[klen-1] = '.';
          }

          /* Format the value */
          char valbuf[80];
          gguf_format_value(valbuf, (int)sizeof(valbuf), &r, vtype);

          APPEND("%-50s %s\n", keybuf, valbuf);
      }
      #undef APPEND

      fwrite(buf, 1, pos, stdout);
      fflush(stdout);
    }

    rc = 0;
out:
    free(buf);
#ifdef _WIN32
    if (addr) UnmapViewOfFile(addr);
    if (mh) CloseHandle(mh);
    if (fh != INVALID_HANDLE_VALUE) CloseHandle(fh);
#else
    if (addr) munmap(addr, fsize);
    if (fd >= 0) close(fd);
#endif
    return rc;
}

/* ---- GGUF Parser ---- */

static int parse_gguf(model_t *m, int max_seq_len) {
    reader_t r = { .data = (const uint8_t *)m->mmap_addr, .pos = 0, .size = m->mmap_size };
    model_config_t *cfg = &m->config;

    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: 0x%08X\n", magic);
        return -1;
    }

    uint32_t version = read_u32(&r);
    if (version < 2 || version > 3) {
        fprintf(stderr, "Unsupported GGUF version: %u\n", version);
        return -1;
    }

    uint64_t n_tensors  = read_u64(&r);
    uint64_t n_metadata = read_u64(&r);

    cfg->alignment = 32;
    cfg->rope_freq_base = 10000.0f;
    cfg->rms_norm_eps = 1e-5f;
    cfg->rope_type = 0;  /* llama pairwise */
    cfg->rope_dim = 0;   /* 0 = use head_dim (default) */
    cfg->max_seq_len = 2048;
    cfg->weight_type = GGUF_TYPE_F16;
    cfg->n_layer_sparsity = 0;
    cfg->f_sparsity_std_mul = 0.0f;
    cfg->n_swa = 0;
    cfg->swa_period = 0;
    cfg->rope_freq_base_swa = 10000.0f;
    cfg->f_attention_scale = 0.0f;  /* 0 = use 1/sqrt(head_dim) default */
    cfg->n_altup = 0;
    cfg->i_altup_act = -1;
    cfg->n_embd_altup = 0;
    cfg->n_layer_kv_from_start = -1;
    cfg->f_final_logit_softcapping = 0.0f;
    cfg->laurel_rank = 0;
    m->tok_bos_id = 1;
    m->tok_eos_id = 2;
    m->tok_add_bos = 1;
    m->tok_add_space_prefix = 1;

    for (uint64_t i = 0; i < n_metadata; i++) {
        gguf_str_t key = read_gguf_string(&r);
        uint32_t vtype = read_u32(&r);

        if (str_eq(key, "general.architecture")) {
            int dummy;
            gguf_str_t arch;
            if (vtype == GGUF_META_STRING) {
                arch = read_gguf_string(&r);
                /* Check if architecture value contains "qwen3" (qwen3 or qwen35) */
                for (uint64_t k = 0; k + 5 <= arch.len; k++) {
                    if (arch.str[k] == 'q' && arch.str[k+1] == 'w' && arch.str[k+2] == 'e' &&
                        arch.str[k+3] == 'n' && arch.str[k+4] == '3') {
                        cfg->is_qwen = 1; break;
                    }
                }
                /* Check for gemma3n */
                for (uint64_t k = 0; k + 7 <= arch.len; k++) {
                    if (arch.str[k] == 'g' && arch.str[k+1] == 'e' && arch.str[k+2] == 'm' &&
                        arch.str[k+3] == 'm' && arch.str[k+4] == 'a' && arch.str[k+5] == '3' &&
                        arch.str[k+6] == 'n') {
                        cfg->is_gemma3n = 1; break;
                    }
                }
            } else {
                skip_meta_value(&r, vtype, &dummy);
            }
        } else


        if (str_eq(key, "llama.embedding_length") || str_eq(key, "general.embedding_length")
            || str_eq(key, "qwen2.embedding_length") || str_eq(key, "qwen3.embedding_length") || str_eq(key, "qwen35.embedding_length") || str_eq(key, "qwen35moe.embedding_length")
            || str_eq(key, "gemma3n.embedding_length")) {
            int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
            /* NOTE: Qwen2 uses interleaved RoPE. Qwen3 and Qwen3.5 use pairwise RoPE
             * (same as Llama). Only set rope_type=1 for qwen2, not qwen3/qwen35.
             * Gemma-3n uses standard pairwise RoPE (rope_type=0). */
            if (key.str[0] == 'q' && key.len > 6 && key.str[5] == '2') cfg->rope_type = 1;
        } else if (str_eq(key, "llama.feed_forward_length") || str_eq(key, "general.feed_forward_length")
            || str_eq(key, "qwen2.feed_forward_length") || str_eq(key, "qwen3.feed_forward_length") || str_eq(key, "qwen35.feed_forward_length")
            || str_eq(key, "gemma3n.feed_forward_length")) {
            int dummy; cfg->n_ffn = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count")
            || str_eq(key, "qwen2.attention.head_count") || str_eq(key, "qwen3.attention.head_count") || str_eq(key, "qwen35.attention.head_count") || str_eq(key, "qwen35moe.attention.head_count")
            || str_eq(key, "gemma3n.attention.head_count")) {
            int dummy; cfg->n_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count_kv")
            || str_eq(key, "qwen2.attention.head_count_kv") || str_eq(key, "qwen3.attention.head_count_kv") || str_eq(key, "qwen35.attention.head_count_kv") || str_eq(key, "qwen35moe.attention.head_count_kv")
            || str_eq(key, "gemma3n.attention.head_count_kv")) {
            int dummy; cfg->n_kv_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "attention.key_length")
            || str_eq(key, "qwen2.attention.key_length")
            || str_eq(key, "qwen3.attention.key_length") || str_eq(key, "qwen35.attention.key_length")
            || str_eq(key, "qwen35moe.attention.key_length")
            || str_eq(key, "gemma3n.attention.key_length")) {
            /* Explicit head_dim (Qwen3/3.5/Gemma-3n may differ from n_embd/n_heads) */
            int dummy; cfg->head_dim = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.block_count")
            || str_eq(key, "qwen2.block_count") || str_eq(key, "qwen3.block_count") || str_eq(key, "qwen35.block_count") || str_eq(key, "qwen35moe.block_count")
            || str_eq(key, "gemma3n.block_count")) {
            int dummy; cfg->n_layers = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.context_length")
            || str_eq(key, "qwen2.context_length") || str_eq(key, "qwen3.context_length") || str_eq(key, "qwen35.context_length") || str_eq(key, "qwen35moe.context_length")
            || str_eq(key, "gemma3n.context_length")) {
            int dummy; cfg->max_seq_len = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.rope.freq_base")
            || str_eq(key, "qwen2.rope.freq_base") || str_eq(key, "qwen3.rope.freq_base") || str_eq(key, "qwen35.rope.freq_base") || str_eq(key, "qwen35moe.rope.freq_base")
            || str_eq(key, "gemma3n.rope.freq_base")) {
            if (vtype == GGUF_META_FLOAT32) {
                cfg->rope_freq_base = read_f32(&r);
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "qwen35.rope.dimension_count")
            || str_eq(key, "qwen35moe.rope.dimension_count")
            || str_eq(key, "llama.rope.dimension_count")) {
            int dummy; cfg->rope_dim = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "rope.dimension_sections")
            || str_eq(key, "qwen35.rope.dimension_sections")
            || str_eq(key, "qwen35moe.rope.dimension_sections")) {
            /* ARRAY of I32 (elem_type=5): [11, 11, 10, 0] for Qwen3.5 */
            if (vtype == GGUF_META_ARRAY) {
                uint32_t elem_type = read_u32(&r);
                uint64_t elem_count = read_u64(&r);
                cfg->rope_dim = 0;
                for (uint64_t ei = 0; ei < elem_count; ei++) {
                    if (elem_type == 4 || elem_type == 5) { /* U32 or I32 */
                        int32_t v = read_i32(&r);
                        cfg->rope_dim += v;
                    } else {
                        /* skip this element */
                        if (elem_type == 0 || elem_type == 1) r.pos += 1;
                        else if (elem_type == 2 || elem_type == 3) r.pos += 2;
                        else if (elem_type == 6 || elem_type == 7 || elem_type == 11 || elem_type == 12) r.pos += 8;
                        else r.pos += 4;
                    }
                }
                cfg->rope_dim *= 2; /* each section is a pair */
                /* NOTE: rope.dimension_sections is for MTP multi-rope, not for
                 * the main transformer. The main transformer applies RoPE to the
                 * full head_dim. We store rope_dim for future MTP support but
                 * use head_dim for RoPE in model_forward(). Reset to 0 so the
                 * (c->rope_dim > 0) ? c->rope_dim : head_dim fallback gives head_dim. */
                cfg->rope_dim = 0;
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "llama.attention.layer_norm_rms_epsilon")
            || str_eq(key, "qwen2.attention.layer_norm_rms_epsilon")
            || str_eq(key, "qwen3.attention.layer_norm_rms_epsilon") || str_eq(key, "qwen35.attention.layer_norm_rms_epsilon") || str_eq(key, "qwen35moe.attention.layer_norm_rms_epsilon")
            || str_eq(key, "gemma3n.attention.layer_norm_rms_epsilon")) {
            /* Read epsilon from GGUF (F32 type=6 or F64 type=11 in metadata) */
            if (vtype == GGUF_META_FLOAT32) { /* F32 */
                cfg->rms_norm_eps = read_f32(&r);
            } else if (vtype == 11) { /* F64 */
                uint64_t vi;
                memcpy(&vi, r.data + r.pos, 8); r.pos += 8;
                vi = GGUF_LE64(vi);
                double val; memcpy(&val, &vi, 8);
                cfg->rms_norm_eps = (float)val;
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "qwen35.ssm.conv_kernel") || str_eq(key, "qwen35moe.ssm.conv_kernel") || str_eq(key, "qwen3.ssm.conv_kernel")) {
            int dummy; cfg->ssm_d_conv = (int)skip_meta_value(&r, vtype, &dummy); cfg->has_ssm = 1;
        } else if (str_eq(key, "qwen35.ssm.state_size") || str_eq(key, "qwen35moe.ssm.state_size") || str_eq(key, "qwen3.ssm.state_size")) {
            int dummy; cfg->ssm_d_state = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.group_count") || str_eq(key, "qwen35moe.ssm.group_count") || str_eq(key, "qwen3.ssm.group_count")) {
            int dummy; cfg->ssm_n_group = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.time_step_rank") || str_eq(key, "qwen35moe.ssm.time_step_rank") || str_eq(key, "qwen3.ssm.time_step_rank")) {
            int dummy; cfg->ssm_dt_rank = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.inner_size") || str_eq(key, "qwen35moe.ssm.inner_size") || str_eq(key, "qwen3.ssm.inner_size")) {
            int dummy; cfg->ssm_d_inner = (int)skip_meta_value(&r, vtype, &dummy);
        /* Gemma-3n specific config */
        } else if (str_eq(key, "gemma3n.altup.num_inputs")) {
            int dummy; cfg->n_altup = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "gemma3n.altup.active_idx")) {
            int dummy; cfg->i_altup_act = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "gemma3n.embedding_length_per_layer_input")) {
            int dummy; cfg->n_embd_altup = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "gemma3n.attention.shared_kv_layers")) {
            /* shared_kv_layers is f32 in GGUF but represents an integer count.
             * Value is the number of layers from the END that share KV.
             * n_layer_kv_from_start = n_layers - shared_kv_layers */
            if (vtype == GGUF_META_FLOAT32) {
                cfg->n_layer_kv_from_start = cfg->n_layers - (int)read_f32(&r);
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "gemma3n.attention.sliding_window")) {
            int dummy; cfg->n_swa = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "gemma3n.attention.sliding_window_pattern")) {
            /* ARRAY of bool (elem_type=8): [true, true, true, true, false, ...]
             * We read the pattern period from the array length.
             * The pattern repeats with period = array_length / (count of true+false).
             * Actually, the pattern is just a per-layer boolean array.
             * We store it in a bitfield in layer_type or a separate array.
             * For now, just skip it - we'll derive swa_pattern from swa_period. */
            int dummy; skip_meta_value(&r, vtype, &dummy);
        }
        /* MoE specific config (qwen35moe) */
        else if (str_eq(key, "qwen35moe.expert_count")) {
            int dummy; cfg->n_expert = (int)skip_meta_value(&r, vtype, &dummy); cfg->has_moe = 1;
        } else if (str_eq(key, "qwen35moe.expert_used_count")) {
            int dummy; cfg->n_expert_used = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35moe.expert_feed_forward_length")) {
            int dummy; cfg->n_ff_exp = (int)skip_meta_value(&r, vtype, &dummy);
            cfg->n_ffn = cfg->n_ff_exp; /* reuse n_ffn for buffer sizing */
        } else if (str_eq(key, "qwen35moe.expert_shared_feed_forward_length")) {
            int dummy; cfg->n_ff_shexp = (int)skip_meta_value(&r, vtype, &dummy);
        }
        /* Gemma-3n: laurel_rank is derived from tensor shapes, not from KV */
        /* Gemma-3n: final_logit_softcapping default */
        else if (str_eq(key, "general.alignment")) {
            int dummy; cfg->alignment = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.vocab_size")
            || str_eq(key, "qwen2.vocab_size") || str_eq(key, "qwen3.vocab_size") || str_eq(key, "qwen35.vocab_size") || str_eq(key, "qwen35moe.vocab_size")
            || str_eq(key, "gemma3n.vocab_size")) {
            int dummy; cfg->vocab_size = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.bos_token_id")) {
            int dummy; m->tok_bos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.model")) {
            /* Set default BOS token based on tokenizer model type */
            if (vtype == GGUF_META_STRING) {
                gguf_str_t model_name = read_gguf_string(&r);
                if (model_name.len == 4 && strncmp(model_name.str, "gpt2", 4) == 0) {
                    /* gpt2 tokenizer: BOS=11, like llama.cpp */
                    if (m->tok_bos_id == 1) m->tok_bos_id = 11;
                } else if (model_name.len == 5 && strncmp(model_name.str, "llama", 5) == 0) {
                    /* llama tokenizer: BOS=1 (already default) */
                    ;
                }
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "tokenizer.ggml.eos_token_id")) {
            int dummy; m->tok_eos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.pre")) {
            /* Detect space marker type for tokenization.
             * GPT-2 BPE tokenizers use literal spaces, not U+2581. */
            if (vtype == GGUF_META_STRING) {
                gguf_str_t pre = read_gguf_string(&r);
                if (pre.len >= 6 && strncmp(pre.str, "smollm", 6) == 0) {
                    m->tok_space_marker = 1; /* U+0100 */
                } else if (pre.len >= 6 && strncmp(pre.str, "qwen35", 6) == 0) {
                    m->tok_space_marker = 3; /* qwen35: U+0100, no prefix on first token */
                } else if (pre.len >= 5 && strncmp(pre.str, "gpt-2", 5) == 0) {
                    m->tok_space_marker = 1; /* GPT-2 BPE: U+0100 space marker */
                } else {
                    m->tok_space_marker = 0; /* U+2581 (default, SPM models) */
                }
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "tokenizer.ggml.add_bos_token")) {
            int dummy; m->tok_add_bos = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.add_space_prefix")) {
            int dummy; m->tok_add_space_prefix = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.tokens")) {
            if (vtype != GGUF_META_ARRAY) {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            } else {
                uint32_t arr_type = read_u32(&r);
                uint64_t arr_len  = read_u64(&r);
                m->tok_tokens_data = r.data + r.pos;
                m->tok_n_tokens = arr_len;
                int dummy;
                for (uint64_t j = 0; j < arr_len; j++) {
                    skip_meta_value(&r, arr_type, &dummy);
                }
            }
        } else if (str_eq(key, "tokenizer.ggml.scores")) {
            if (vtype != GGUF_META_ARRAY) {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            } else {
                uint32_t arr_type = read_u32(&r);
                uint64_t arr_len  = read_u64(&r);
                (void)arr_type;
                m->tok_scores_data = r.data + r.pos;
                m->tok_n_scores = arr_len;
                r.pos += arr_len * 4;
            }
        } else if (str_eq(key, "tokenizer.ggml.merges")) {
            if (vtype != GGUF_META_ARRAY) {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            } else {
                uint32_t arr_type = read_u32(&r);
                uint64_t arr_len  = read_u64(&r);
                if (arr_type == GGUF_META_STRING) {
                    m->tok_merges_data = r.data + r.pos;
                    m->tok_n_merges = arr_len;
                }
                { const uint8_t *pp = r.data + r.pos;
                  uint64_t nn = arr_len;
                  for (uint64_t ii = 0; ii < nn; ii++) {
                      uint64_t sl; memcpy(&sl, pp, 8); sl = GGUF_LE64(sl); pp += 8 + sl; }
                  r.pos = (size_t)(pp - r.data); }
            }
        } else if (str_eq(key, "tokenizer.ggml.token_type")) {
            if (vtype != GGUF_META_ARRAY) {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            } else {
                uint32_t arr_type = read_u32(&r);
                uint64_t arr_len  = read_u64(&r);
                if (arr_type == GGUF_META_INT32 || arr_type == GGUF_META_UINT32) {
                    m->tok_token_type_data = r.data + r.pos;
                    m->tok_n_token_type = arr_len;
                    r.pos += arr_len * 4;
                } else {
                    int dummy;
                    for (uint64_t j = 0; j < arr_len; j++) {
                        skip_meta_value(&r, arr_type, &dummy);
                    }
                }
            }
        /* Split GGUF metadata */
        } else if (str_eq(key, "split.count")) {
            int dummy; m->split_count = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "split.no")) {
            int dummy; m->split_no = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "split.tensors.count")) {
            int dummy; m->split_tensors_count = (int)skip_meta_value(&r, vtype, &dummy);
        } else {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        }
    }

    /* Apply user-specified context length override (-c option) */
    if (max_seq_len > 0) {
        cfg->max_seq_len = max_seq_len;
    }

    /* ---- Split GGUF: detect and prepare ---- */
    {
        int total_splits = m->split_count > 1 ? m->split_count : 1;
        m->n_splits = total_splits;

        if (total_splits > 1) {
            /* Derive prefix from first split path */
            char prefix[512];
            if (!split_path_prefix(prefix, sizeof(prefix), m->first_split_path)) {
                fprintf(stderr, "Split model: could not derive prefix from path '%s'\n", m->first_split_path);
                return -1;
            }

            fprintf(stderr, "Split model: %d splits, total tensors=%d\n",
                    total_splits, m->split_tensors_count);

            /* Mmap remaining splits */
            for (int si = 1; si < total_splits; si++) {
                char split_path[512];
                split_path_build(split_path, sizeof(split_path), prefix, si, total_splits);

                if (mmap_one_file(&m->splits[si], split_path) != 0) {
                    /* Clean up already-mapped splits */
                    for (int sj = 1; sj <= si; sj++) munmap_one_file(&m->splits[sj]);
                    return -1;
                }
                prepare_mmap(m->splits[si].mmap_addr, m->splits[si].mmap_size);
            }
        }
    }

    /* Validate required config fields before using them */
    if (cfg->n_embd <= 0 || cfg->n_heads <= 0 || cfg->n_layers <= 0) {
        fprintf(stderr, "Model architecture not recognized (n_embd=%d n_heads=%d n_layers=%d)\n",
                cfg->n_embd, cfg->n_heads, cfg->n_layers);
        return -1;
    }

    /* head_dim: use GGUF's attention.key_length if set (Qwen3), else derive */
    if (cfg->head_dim <= 0) {
        cfg->head_dim = cfg->n_embd / cfg->n_heads;
    }

    /* Gemma-3n defaults */
    if (cfg->is_gemma3n) {
        if (cfg->n_altup <= 0) cfg->n_altup = 4;
        if (cfg->i_altup_act < 0) cfg->i_altup_act = 0;
        if (cfg->n_embd_altup <= 0) cfg->n_embd_altup = 256;
        if (cfg->n_layer_kv_from_start < 0) cfg->n_layer_kv_from_start = cfg->n_layers;
        if (cfg->f_final_logit_softcapping <= 0) cfg->f_final_logit_softcapping = 30.0f;
        /* Gemma-3n uses NeoX-style RoPE (split halves), not Llama pairwise */
        cfg->rope_type = 1;
        /* Attention scale: 1.0 (no 1/sqrt(head_dim) scaling, QK-norm handles it) */
        cfg->f_attention_scale = 1.0f;
        /* Activation sparsity: first 10 layers use gaussian_topk */
        cfg->n_layer_sparsity = 10;
        cfg->f_sparsity_std_mul = 1.6448533535003662f;
        /* SWA: period 5, window 512. SWA layers use freq_base=10000 */
        cfg->swa_period = 5;
        cfg->rope_freq_base_swa = 10000.0f;
    }

    /* ---- Parse tensor info entries (split-aware) ---- */
    /* tensor_info_t with split_idx */
    typedef struct {
        gguf_str_t name;
        uint32_t   n_dims;
        uint64_t   dims[4];
        uint32_t   type;
        uint64_t   offset;
        int        split_idx;
    } tensor_info_t;

    /* Total tensor count: use split.tensors.count if split, else header n_tensors */
    uint64_t total_tensor_count = (m->split_tensors_count > 0) ? (uint64_t)m->split_tensors_count : n_tensors;

    tensor_info_t *tinfos = (tensor_info_t *)malloc(total_tensor_count * sizeof(tensor_info_t));
    if (!tinfos) { fprintf(stderr, "OOM allocating tensor info\n"); return -1; }

    /* Parse tensor tables from each split */
    {
        uint64_t tidx = 0;
        for (int si = 0; si < m->n_splits; si++) {
            reader_t sr;
            uint64_t s_n_tensors;

            if (si == 0) {
                /* Split 0: we already parsed metadata, reader 'r' is positioned after it.
                 * We need the per-file n_tensors. Re-read header to get it, then reuse
                 * the existing reader position (after metadata) for tensor entries. */
                {
                    reader_t hr = (reader_t){ .data = (const uint8_t *)m->splits[0].mmap_addr,
                                              .pos = 0, .size = m->splits[0].mmap_size };
                    (void)read_u32(&hr); /* magic */
                    (void)read_u32(&hr); /* version */
                    s_n_tensors = read_u64(&hr);
                }
                /* Continue reading tensor entries from 'r' (positioned after metadata) */
                for (uint64_t ti = 0; ti < s_n_tensors; ti++) {
                    tinfos[tidx].name     = read_gguf_string(&r);
                    tinfos[tidx].n_dims   = read_u32(&r);
                    for (uint32_t d = 0; d < tinfos[tidx].n_dims; d++) {
                        tinfos[tidx].dims[d] = read_u64(&r);
                    }
                    tinfos[tidx].type     = read_u32(&r);
                    tinfos[tidx].offset   = read_u64(&r);
                    tinfos[tidx].split_idx = 0;
                    tidx++;
                }
                /* tensor_data_base for split 0: after all tensor entries in split 0 */
                m->tensor_data_base[0] = (r.pos + cfg->alignment - 1) & ~((size_t)cfg->alignment - 1);
            } else {
                /* Other splits: parse from scratch */
                sr = (reader_t){ .data = (const uint8_t *)m->splits[si].mmap_addr,
                                .pos = 0, .size = m->splits[si].mmap_size };
                (void)read_u32(&sr); /* magic */
                (void)read_u32(&sr); /* version */
                s_n_tensors = read_u64(&sr);
                uint64_t s_n_metadata = read_u64(&sr);
                for (uint64_t mi = 0; mi < s_n_metadata; mi++) {
                    (void)read_gguf_string(&sr);
                    uint32_t svtype = read_u32(&sr);
                    int dummy;
                    skip_meta_value(&sr, svtype, &dummy);
                }

                for (uint64_t ti = 0; ti < s_n_tensors; ti++) {
                    tinfos[tidx].name     = read_gguf_string(&sr);
                    tinfos[tidx].n_dims   = read_u32(&sr);
                    for (uint32_t d = 0; d < tinfos[tidx].n_dims; d++) {
                        tinfos[tidx].dims[d] = read_u64(&sr);
                    }
                    tinfos[tidx].type     = read_u32(&sr);
                    tinfos[tidx].offset   = read_u64(&sr);
                    tinfos[tidx].split_idx = si;
                    tidx++;
                }
                /* tensor_data_base: after tensor entries */
                m->tensor_data_base[si] = (sr.pos + cfg->alignment - 1) & ~((size_t)cfg->alignment - 1);
            }
        }
        /* Verify tensor count matches */
        if (tidx != total_tensor_count) {
            fprintf(stderr, "Split model: expected %lu tensors but got %lu\n",
                    (unsigned long)total_tensor_count, (unsigned long)tidx);
            free(tinfos);
            return -1;
        }
    }

    /* Detect MTP (Multi-Token Prediction) layers by scanning for "nextn" tensors */
    cfg->has_mtp = 0;
    cfg->n_mtp_layers = 0;
    for (uint64_t i = 0; i < total_tensor_count; i++) {
        if (strstr(tinfos[i].name.str, "nextn.") && tinfos[i].name.len > 0) {
            cfg->has_mtp = 1;
            break;
        }
    }

    model_weights_t *w = &m->weights;
    memset(w, 0, sizeof(*w));

    for (uint64_t i = 0; i < total_tensor_count; i++) {
        /* Resolve tensor pointer from the correct split's mmap region */
        int si = tinfos[i].split_idx;
        const void *ptr = (const uint8_t *)m->splits[si].mmap_addr + m->tensor_data_base[si] + tinfos[i].offset;
        gguf_type_t qtype = (gguf_type_t)tinfos[i].type;

        if (str_eq(tinfos[i].name, "token_embd.weight")) {
            w->token_embd = ptr; w->type_token_embd = qtype;
        } else if (str_eq(tinfos[i].name, "output_norm.weight")) {
            w->output_norm = ptr; w->type_output_norm = qtype;
        } else if (str_eq(tinfos[i].name, "output.weight")) {
            w->output = ptr; w->type_output = qtype;
        } else {
            int layer = -1;
            char suffix[64] = {0};

            if (tinfos[i].name.len > 4 && memcmp(tinfos[i].name.str, "blk.", 4) == 0) {
                const char *p = tinfos[i].name.str + 4;
                const char *end = tinfos[i].name.str + tinfos[i].name.len;
                layer = 0;
                while (p < end && *p >= '0' && *p <= '9') {
                    layer = layer * 10 + (*p - '0');
                    p++;
                }
                if (p < end && *p == '.') {
                    p++;
                    size_t slen = (size_t)(end - p);
                    if (slen < sizeof(suffix)) {
                        memcpy(suffix, p, slen);
                        suffix[slen] = '\0';
                    }
                }
            }

            if (layer >= 0 && layer < MAX_LAYERS) {
                layer_weights_t *lw = &w->layers[layer];
                if (strcmp(suffix, "attn_norm.weight") == 0) {
                    lw->attn_norm = ptr; lw->type_attn_norm = qtype; if (layer == 0) fprintf(stderr, "DBG attn_norm type=%d\n", qtype);
                } else if (strcmp(suffix, "attn_q.weight") == 0) {
                    lw->attn_q = ptr; lw->type_attn_q = qtype; if (layer == 0) fprintf(stderr, "DBG attn_q type=%d\n", qtype);
                } else if (strcmp(suffix, "attn_k.weight") == 0) {
                    lw->attn_k = ptr; lw->type_attn_k = qtype; if (layer == 0) fprintf(stderr, "DBG attn_k type=%d\n", qtype); lw->is_attn_layer = 1;
                } else if (strcmp(suffix, "attn_v.weight") == 0) {
                    lw->attn_v = ptr; lw->type_attn_v = qtype; if (layer == 0) fprintf(stderr, "DBG attn_v type=%d\n", qtype); lw->is_attn_layer = 1;
                } else if (strcmp(suffix, "attn_output.weight") == 0) {
                    lw->attn_output = ptr; lw->type_attn_output = qtype; if (layer == 0) fprintf(stderr, "DBG attn_output type=%d\n", qtype);
                } else if (strcmp(suffix, "attn_q_norm.weight") == 0) {
                    lw->attn_q_norm = ptr; lw->type_attn_q_norm = qtype;
                } else if (strcmp(suffix, "attn_k_norm.weight") == 0) {
                    lw->attn_k_norm = ptr; lw->type_attn_k_norm = qtype;
                } else if (strcmp(suffix, "ffn_norm.weight") == 0) {
                    lw->post_attn_norm = ptr; lw->type_post_attn_norm = qtype;
                } else if (strcmp(suffix, "post_attention_norm.weight") == 0) {
                    /* Standard models (Llama, Qwen2/3/3.5/3.6): this is the
                     * FFN pre-norm (alias for ffn_norm).
                     * Gemma-3n: separate attention post-norm (ffn_norm exists too). */
                    if (cfg->is_gemma3n) {
                        lw->attn_post_norm = ptr; lw->type_attn_post_norm = qtype;
                    } else {
                        lw->post_attn_norm = ptr; lw->type_post_attn_norm = qtype;
                    }
                } else if (strcmp(suffix, "ffn_gate.weight") == 0) {
                    lw->ffn_gate = ptr; lw->type_ffn_gate = qtype; if (layer == 0) fprintf(stderr, "DBG ffn_gate type=%d\n", qtype);
                } else if (strcmp(suffix, "ffn_down.weight") == 0) {
                    lw->ffn_down = ptr; lw->type_ffn_down = qtype; if (layer == 0) fprintf(stderr, "DBG ffn_down type=%d\n", qtype);
                } else if (strcmp(suffix, "ffn_up.weight") == 0) {
                    lw->ffn_up = ptr; lw->type_ffn_up = qtype; if (layer == 0) fprintf(stderr, "DBG ffn_up type=%d\n", qtype);
                }
                /* MoE tensors (qwen35moe) */
                else if (strcmp(suffix, "ffn_gate_exps.weight") == 0) {
                    lw->ffn_gate_exps = ptr; lw->type_ffn_gate_exps = qtype;
                } else if (strcmp(suffix, "ffn_up_exps.weight") == 0) {
                    lw->ffn_up_exps = ptr; lw->type_ffn_up_exps = qtype;
                } else if (strcmp(suffix, "ffn_down_exps.weight") == 0) {
                    lw->ffn_down_exps = ptr; lw->type_ffn_down_exps = qtype;
                } else if (strcmp(suffix, "ffn_gate_inp.weight") == 0) {
                    lw->ffn_gate_inp = ptr; lw->type_ffn_gate_inp = qtype;
                } else if (strcmp(suffix, "ffn_gate_inp_shexp.weight") == 0) {
                    lw->ffn_gate_inp_shexp = ptr; lw->type_ffn_gate_inp_shexp = qtype;
                } else if (strcmp(suffix, "ffn_gate_shexp.weight") == 0) {
                    lw->ffn_gate_shexp = ptr; lw->type_ffn_gate_shexp = qtype;
                } else if (strcmp(suffix, "ffn_up_shexp.weight") == 0) {
                    lw->ffn_up_shexp = ptr; lw->type_ffn_up_shexp = qtype;
                } else if (strcmp(suffix, "ffn_down_shexp.weight") == 0) {
                    lw->ffn_down_shexp = ptr; lw->type_ffn_down_shexp = qtype;
                }
                /* SSM tensors (Qwen3.5) */
                else if (strcmp(suffix, "attn_qkv.weight") == 0) {
                    lw->attn_qkv = ptr; lw->type_attn_qkv = qtype; lw->is_attn_layer = 0;
                } else if (strcmp(suffix, "attn_gate.weight") == 0) {
                    lw->attn_gate_ssm = ptr; lw->type_attn_gate_ssm = qtype;
                } else if (strcmp(suffix, "ssm_a") == 0) {
                    lw->ssm_a = ptr; lw->type_ssm_a = qtype;
                } else if (strcmp(suffix, "ssm_alpha.weight") == 0) {
                    lw->ssm_alpha = ptr; lw->type_ssm_alpha = qtype;
                } else if (strcmp(suffix, "ssm_beta.weight") == 0) {
                    lw->ssm_beta = ptr; lw->type_ssm_beta = qtype;
                } else if (strcmp(suffix, "ssm_conv1d.weight") == 0) {
                    lw->ssm_conv1d = ptr; lw->type_ssm_conv1d = qtype;
                } else if (strcmp(suffix, "ssm_dt.bias") == 0) {
                    lw->ssm_dt = ptr; lw->type_ssm_dt = qtype;
                } else if (strcmp(suffix, "ssm_norm.weight") == 0) {
                    lw->ssm_norm = ptr;
                } else if (strcmp(suffix, "ssm_out.weight") == 0) {
                    lw->ssm_out = ptr; lw->type_ssm_out = qtype;
                }
                /* Gemma-3n tensors */
                else if (strcmp(suffix, "attn_post_norm.weight") == 0) {
                    lw->attn_post_norm = ptr; lw->type_attn_post_norm = qtype;
                } else if (strcmp(suffix, "post_ffw_norm.weight") == 0) {
                    lw->post_ffw_norm = ptr; lw->type_post_ffw_norm = qtype;
                } else if (strcmp(suffix, "per_layer_inp_gate.weight") == 0
                        || strcmp(suffix, "inp_gate.weight") == 0) {
                    lw->per_layer_inp_gate = ptr; lw->type_per_layer_inp_gate = qtype;
                    if (layer == 0) fprintf(stderr, "DBG per_layer_inp_gate type=%d\n", qtype);
                } else if (strcmp(suffix, "per_layer_proj.weight") == 0
                        || strcmp(suffix, "proj.weight") == 0) {
                    lw->per_layer_proj = ptr; lw->type_per_layer_proj = qtype;
                    if (layer == 0) fprintf(stderr, "DBG per_layer_proj type=%d\n", qtype);
                } else if (strcmp(suffix, "per_layer_post_norm.weight") == 0
                        || strcmp(suffix, "post_norm.weight") == 0) {
                    lw->per_layer_post_norm = ptr; lw->type_per_layer_post_norm = qtype;
                } else if (strcmp(suffix, "altup_correct_coef.weight") == 0) {
                    lw->altup_correct_coef = ptr;
                } else if (strcmp(suffix, "altup_correct_scale.weight") == 0) {
                    lw->altup_correct_scale = ptr;
                    if (layer == 0) fprintf(stderr, "DBG altup_correct_scale type=%d\n", qtype);
                } else if (strcmp(suffix, "altup_predict_coef.weight") == 0) {
                    lw->altup_predict_coef = ptr;
                } else if (strcmp(suffix, "altup_router.weight") == 0) {
                    lw->altup_router = ptr; lw->type_altup_router = qtype;
                } else if (strcmp(suffix, "altup_router_norm.weight") == 0) {
                    lw->altup_router_norm = ptr;
                } else if (strcmp(suffix, "laurel_l.weight") == 0) {
                    lw->laurel_l = ptr; lw->type_laurel_l = qtype;
                    if (layer == 0) fprintf(stderr, "DBG laurel_l type=%d\n", qtype);
                    /* Derive laurel_rank from tensor shape [n_embd, laurel_rank] */
                    for (uint64_t ti = 0; ti < total_tensor_count; ti++) {
                        if (tinfos[ti].name.len > 4 && memcmp(tinfos[ti].name.str, "blk.", 4) == 0) {
                            const char *p = tinfos[ti].name.str + 4;
                            int bl = 0;
                            while (*p >= '0' && *p <= '9') { bl = bl * 10 + (*p - '0'); p++; }
                            if (*p != '.' || bl != layer) continue;
                            p++; /* skip dot */
                            if (strncmp(p, "laurel_l.weight", 15) == 0 && tinfos[ti].n_dims >= 2) {
                                cfg->laurel_rank = (int)tinfos[ti].dims[1];
                                break;
                            }
                        }
                    }
                } else if (strcmp(suffix, "laurel_r.weight") == 0) {
                    lw->laurel_r = ptr; lw->type_laurel_r = qtype;
                    if (layer == 0) fprintf(stderr, "DBG laurel_r type=%d\n", qtype);
                } else if (strcmp(suffix, "laurel_post_norm.weight") == 0) {
                    lw->laurel_post_norm = ptr;
                }
            } else if (cfg->is_gemma3n) {
                /* Gemma-3n global tensors (not under blk.) */
                if (str_eq(tinfos[i].name, "altup_proj.weight")) {
                    w->altup_proj = ptr; w->type_altup_proj = qtype;
                    fprintf(stderr, "DBG altup_proj type=%d\n", qtype);
                } else if (str_eq(tinfos[i].name, "altup_unembd_proj.weight")) {
                    w->altup_unembd_proj = ptr; w->type_altup_unembd_proj = qtype;
                    fprintf(stderr, "DBG altup_unembd_proj type=%d\n", qtype);
                } else if (str_eq(tinfos[i].name, "per_layer_token_embd.weight")) {
                    w->per_layer_tok_embd = ptr; w->type_per_layer_tok_embd = qtype;
                    fprintf(stderr, "DBG per_layer_tok_embd type=%d\n", qtype);
                } else if (str_eq(tinfos[i].name, "per_layer_model_proj.weight")) {
                    w->per_layer_model_proj = ptr; w->type_per_layer_model_proj = qtype;
                    fprintf(stderr, "DBG per_layer_model_proj type=%d\n", qtype);
                } else if (str_eq(tinfos[i].name, "per_layer_proj_norm.weight")) {
                    w->per_layer_proj_norm = ptr;
                    fprintf(stderr, "DBG per_layer_proj_norm type=%d\n", qtype);
                }
            }
        }
    }

    if (!w->output) {
        w->output = w->token_embd;
        w->type_output = w->type_token_embd;
    }

    if (cfg->vocab_size == 0) {
        for (uint64_t i = 0; i < total_tensor_count; i++) {
            if (str_eq(tinfos[i].name, "token_embd.weight")) {
                if (tinfos[i].n_dims >= 2) {
                    int d0 = (int)tinfos[i].dims[0];
                    int d1 = (int)tinfos[i].dims[1];
                    cfg->vocab_size = (d0 == cfg->n_embd) ? d1 : d0;
                }
                break;
            }
        }
    }
    if (cfg->vocab_size == 0 && m->tok_n_tokens > 0) {
        cfg->vocab_size = (int)m->tok_n_tokens;
    }

    // For SSM models, the first layer may not have attn_q
    if (cfg->has_ssm && w->layers[0].type_attn_q == 0) {
        cfg->weight_type = w->layers[0].type_attn_qkv;
    } else {
        cfg->weight_type = w->layers[0].type_attn_q;
    }

    /* Count MTP layers from the end: layers with "nextn" tensors */
    if (cfg->has_mtp) {
        for (int i = cfg->n_layers - 1; i >= 0; i--) {
            int has_nextn = 0;
            for (uint64_t ti = 0; ti < total_tensor_count; ti++) {
                if (strstr(tinfos[ti].name.str, "blk.") == NULL) continue;
                const char *p = tinfos[ti].name.str + 4;
                int bl = 0;
                while (*p >= '0' && *p <= '9') { bl = bl * 10 + (*p - '0'); p++; }
                if (*p != '.') continue;
                if (bl == i && strstr(tinfos[ti].name.str, "nextn.")) {
                    has_nextn = 1; break;
                }
            }
            if (has_nextn) {
                cfg->n_mtp_layers++;
            } else {
                break; /* MTP layers are contiguous at the end */
            }
        }
    }

    fprintf(stderr, "Model config:\n");
    fprintf(stderr, "  n_embd=%d, n_ffn=%d, n_heads=%d, n_kv_heads=%d\n",
            cfg->n_embd, cfg->n_ffn, cfg->n_heads, cfg->n_kv_heads);
        fprintf(stderr, "  n_layers=%d, vocab_size=%d, max_seq=%d\n",
            cfg->n_layers, cfg->vocab_size, cfg->max_seq_len);
    if (cfg->has_ssm) {
        int conv_dim = 2 * cfg->ssm_d_state * cfg->ssm_n_group + cfg->ssm_d_inner;
        fprintf(stderr, "  SSM: conv=%d state=%d groups=%d dt_rank=%d inner=%d conv_dim=%d\n",
                cfg->ssm_d_conv, cfg->ssm_d_state, cfg->ssm_n_group,
                cfg->ssm_dt_rank, cfg->ssm_d_inner, conv_dim);
        int attn_count = 0, ssm_count = 0;
        for (int i = 0; i < cfg->n_layers; i++) {
            if (w->layers[i].is_attn_layer) attn_count++; else ssm_count++;
        }
        fprintf(stderr, "  Layers: %d SSM + %d full attention\n", ssm_count, attn_count);
    }
    fprintf(stderr, "  head_dim=%d, rope_dim=%d, rope_base=%.1f\n", cfg->head_dim, cfg->rope_dim, (double)cfg->rope_freq_base);
    if (cfg->has_moe) {
        fprintf(stderr, "  MoE: experts=%d used=%d ff_exp=%d ff_shexp=%d\n",
                cfg->n_expert, cfg->n_expert_used, cfg->n_ff_exp, cfg->n_ff_shexp);
    }
    if (cfg->is_gemma3n) {
        fprintf(stderr, "  Gemma-3n: altup=%d active=%d laurel_rank=%d embd_altup=%d kv_layers=%d softcap=%.1f\n",
                cfg->n_altup, cfg->i_altup_act, cfg->laurel_rank, cfg->n_embd_altup,
                cfg->n_layer_kv_from_start, (double)cfg->f_final_logit_softcapping);
    }
    /* On big-endian, GGUF stores all multi-byte values as little-endian.
     * Swap F16 values in-place for all quantized block types that contain FP16 scales. */
#if defined(__APPLE__) && defined(__ppc__)
    { int be_start = clock();
    fprintf(stderr, "Big-endian: swapping F16 values...\n");
    for (uint64_t i = 0; i < total_tensor_count; i++) {
        gguf_type_t qt = (gguf_type_t)tinfos[i].type;
        int _si = tinfos[i].split_idx;
        void *ptr = (void *)((uint8_t *)m->splits[_si].mmap_addr + m->tensor_data_base[_si] + tinfos[i].offset);
        size_t nrows = tinfos[i].dims[0];
        for (uint64_t d = 1; d < tinfos[i].n_dims; d++) nrows *= tinfos[i].dims[d];

        if (qt == GGUF_TYPE_F16 || qt == GGUF_TYPE_BF16) {
            uint16_t *f16p = (uint16_t *)ptr;
            swap_f16_block(f16p, nrows);
        } else if (qt == GGUF_TYPE_Q8_0) {
            size_t nblocks = nrows / 32;
            for (size_t b = 0; b < nblocks; b++) {
                block_q8_0 *blk = (block_q8_0 *)((uint8_t *)ptr + b * sizeof(block_q8_0));
                blk->d = GGUF_LE16(blk->d);
            }
        } else if (qt == GGUF_TYPE_Q4_0 || qt == GGUF_TYPE_Q4_1 ||
                   qt == GGUF_TYPE_Q4_0_4_4 || qt == GGUF_TYPE_Q4_0_8_8) {
            size_t bs = (qt == GGUF_TYPE_Q4_0_4_4) ? sizeof(block_q4_0x4)
                       : (qt == GGUF_TYPE_Q4_0_8_8) ? sizeof(block_q4_0x8)
                       : (qt == GGUF_TYPE_Q4_1) ? sizeof(block_q4_1)
                       : sizeof(block_q4_0);
            size_t nblocks = nrows / 32;
            for (size_t b = 0; b < nblocks; b++) {
                block_q4_0 *blk = (block_q4_0 *)((uint8_t *)ptr + b * bs);
                blk->d = GGUF_LE16(blk->d);
            }
        } else if (qt == GGUF_TYPE_Q2_0) {
            size_t nblocks = nrows / 128;
            for (size_t b = 0; b < nblocks; b++) {
                block_q2_0 *blk = (block_q2_0 *)((uint8_t *)ptr + b * sizeof(block_q2_0));
                blk->d = GGUF_LE16(blk->d);
            }
        } else if (qt == GGUF_TYPE_Q1_0) {
            size_t nblocks = nrows / 128;
            for (size_t b = 0; b < nblocks; b++) {
                block_q1_0 *blk = (block_q1_0 *)((uint8_t *)ptr + b * sizeof(block_q1_0));
                blk->d = GGUF_LE16(blk->d);
            }
        } else if (qt == GGUF_TYPE_Q5_K) {
            size_t nblocks = nrows / 256;
            for (size_t b = 0; b < nblocks; b++) {
                block_q5_K *blk = (block_q5_K *)((uint8_t *)ptr + b * sizeof(block_q5_K));
                blk->d = GGUF_LE16(blk->d);
                blk->dm = GGUF_LE16(blk->dm);
            }
        } else if (qt == GGUF_TYPE_Q6_K) {
            size_t nblocks = nrows / 256;
            for (size_t b = 0; b < nblocks; b++) {
                block_q6_K *blk = (block_q6_K *)((uint8_t *)ptr + b * sizeof(block_q6_K));
                blk->d = GGUF_LE16(blk->d);
            }
        }
    }
    fprintf(stderr, "Big-endian: swap done (%.0fms)\n", (clock() - be_start) / (double)CLOCKS_PER_SEC * 1000);
}
#endif

    free(tinfos);
    return 0;
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
                if (l == 0) {
                    float maxw = 0; int maxw_i = 0;
                    for(int i=0;i<n_embd;i++) if(fabsf(s->per_layer_post_norm_w[l][i])>maxw){maxw=fabsf(s->per_layer_post_norm_w[l][i]);maxw_i=i;}
                    fprintf(stderr, "DBG per_layer_post_norm type=%d max|w|=%.4f@%d [1328:1335]={", lw->type_per_layer_post_norm, maxw, maxw_i);
                    for(int i=1328;i<1335;i++) fprintf(stderr,"%.4f%s",s->per_layer_post_norm_w[l][i],i<1334?",":"");
                    fprintf(stderr, "}\n");
                }
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
                            /* RoPE cos/sin: [max_seq_len * half_dim] */
                            {
                                int half_dim = c->head_dim / 2;
                                size_t rope_n = (size_t)c->max_seq_len * half_dim;
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

/* ================================================================
 * Forward pass with:
 *   - FP16 KV cache (halves memory bandwidth in attention)
 *   - Flash attention / online softmax (single pass, no score buffer)
 *   - Pre-computed RoPE tables (table lookup instead of trig)
 * ================================================================ */

/* Forward declarations for SSM helpers */
static void ssm_forward(model_t *m, run_state_t *s, float *x, float *residual,
                        layer_weights_t *lw, int il, int pos, void *gpu_lw);

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
static void attn_core(
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
                  for (; d < head_dim; d++) acc[d] = acc[d] * correction + fp16_to_fp32(vt16[d]); }
#elif defined(PICOLM_AVX)
                { __m256 cv = _mm256_set1_ps(correction); int d = 0;
                  for (; d + 7 < head_dim; d += 8) { __m256 vf = fp16x8_to_fp32_inline(vt16 + d); __m256 af = _mm256_loadu_ps(acc + d); _mm256_storeu_ps(acc + d, _mm256_add_ps(_mm256_mul_ps(af, cv), vf)); }
                  for (; d < head_dim; d++) acc[d] = acc[d] * correction + fp16_to_fp32(vt16[d]); }
#else
                for (int d = 0; d < head_dim; d++) acc[d] = acc[d] * correction + fp16_to_fp32(vt16[d]);
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
                  for (; d < head_dim; d++) acc[d] += w * fp16_to_fp32(vt16[d]); }
#elif defined(PICOLM_AVX)
                { __m256 wv = _mm256_set1_ps(w); int d = 0;
                  for (; d + 7 < head_dim; d += 8) { __m256 vf = fp16x8_to_fp32_inline(vt16 + d); __m256 af = _mm256_loadu_ps(acc + d); _mm256_storeu_ps(acc + d, _mm256_add_ps(_mm256_mul_ps(vf, wv), af)); }
                  for (; d < head_dim; d++) acc[d] += w * fp16_to_fp32(vt16[d]); }
#else
                for (int d = 0; d < head_dim; d++) acc[d] += w * fp16_to_fp32(vt16[d]);
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
typedef struct {
    int kv_h, kv_mul, n_kv_heads, head_dim, pos;
    int kv_type_k, kv_type_v;  /* kv_cache_type_t values: 0=F16, 1=Q8_0, 2=Q4_0 */
    size_t kv_row_size_k, kv_row_size_v;
    size_t kv_head_stride_k, kv_head_stride_v;
    const uint8_t *kcache, *vcache;
    const float *q;   /* [n_heads][head_dim] */
    float *xb;        /* [n_heads][head_dim] */
    int kv_hadamard_k, kv_hadamard_v;
    int kv_hadamard_size;
    float attn_scale;  /* scale factor for QK scores (default 1/sqrt(head_dim)) */
} attn_group_ctx_t;

static void attention_group(int kv_head_idx, void *ctx_ptr) {
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
                    accg[d] = accg[d] * correction + v_f32[d];
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
                    accg[d] += w * v_f32[d];
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
                    for (int d = 0; d < head_dim; d++) accg[d] = accg[d] * correction + fp16_to_fp32(vt16[d]);
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
                    for (int d = 0; d < head_dim; d++) accg[d] += w * fp16_to_fp32(vt16[d]);
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

/* Get pointer to expert e's sub-tensor within a 3D expert tensor.
 * For gate_exps [n_embd, n_ff_exp, n_expert]: each expert is [n_embd, n_ff_exp].
 * For down_exps [n_ff_exp, n_embd, n_expert]: each expert is [n_ff_exp, n_embd].
 * GGUF stores dims row-major: dims[0] varies fastest.
 * Expert e starts at: e * dim1 * gguf_type_row_size(type, dim0)
 * (each row has dim0 elements, and there are dim1 rows per expert) */
static const void *get_expert_slice(const void *base, int expert,
                                    int dim0, int dim1, gguf_type_t type) {
    size_t expert_stride = (size_t)dim1 * gguf_type_row_size(type, dim0);
    return (const void *)((const uint8_t *)base + expert * expert_stride);
}

/* Parallel expert worker for moe_forward: processes one expert using
 * per-thread scratch buffers. Uses matmul_q8_seq (not matmul_q8) to
 * avoid racing on the global n_threads variable and re-entering the
 * thread pool from inside a tensor_parallel_for worker. */
typedef struct {
    const block_q8_0 *qx;       /* pre-quantized input */
    const float *qx_d;          /* input Q8_0 deltas */
    int dim, n_ff;
    int *ids;                   /* expert ids [n_used] */
    float *weights;             /* expert weights [n_used] */
    gguf_type_t type_gate, type_up, type_down;
    const void *gate_exps, *up_exps, *down_exps;
    float *expert_out;          /* output [n_used * dim], each expert writes to slot i*dim */
    run_state_t *s;
} moe_expert_ctx;

static void moe_expert_worker(int i, void *vctx) {
    moe_expert_ctx *ctx = (moe_expert_ctx *)vctx;
    int dim = ctx->dim, n_ff = ctx->n_ff;

    int eid = ctx->ids[i];
    float w_i = ctx->weights[i];

    /* Get per-thread scratch buffer */
    unsigned tid = tensor_get_thread_id();
    int nt = tensor_get_n_threads();
    if (tid >= (unsigned)nt) tid = 0;
    char *scratch = (char *)ctx->s->moe_thread_scratch + tid * ctx->s->moe_thread_stride;

    float *gate_buf = (float *)scratch;
    float *up_buf = gate_buf + n_ff;
    /* xb2 + acc area follows (dim * 2 floats), then Q8 buffers */
    char *q8_ptr = (char *)(up_buf + n_ff) + (size_t)dim * 2 * sizeof(float);
    block_q8_0 *down_qx = (block_q8_0 *)q8_ptr;
    float *down_qx_d = (float *)((char *)down_qx + gguf_type_row_size(GGUF_TYPE_Q8_0, n_ff));

    const void *gate_w = get_expert_slice(ctx->gate_exps, eid, dim, n_ff, ctx->type_gate);
    const void *up_w = get_expert_slice(ctx->up_exps, eid, dim, n_ff, ctx->type_up);
    const void *down_w = get_expert_slice(ctx->down_exps, eid, n_ff, dim, ctx->type_down);

    /* Use matmul_q8_seq: sequential, no thread pool, safe inside tensor_parallel_for */
    matmul_q8_seq(gate_buf, ctx->qx, ctx->qx_d, gate_w, dim, n_ff, ctx->type_gate);
    matmul_q8_seq(up_buf, ctx->qx, ctx->qx_d, up_w, dim, n_ff, ctx->type_up);

    /* SwiGLU: silu(gate) * up */
    silu(gate_buf, n_ff);
    elemwise_mul(gate_buf, gate_buf, up_buf, n_ff);

    /* Quantize SwiGLU output for Q8xQ8 down projection */
    {
        size_t dnb = n_ff / 32;
        quantize_row_q8_0(gate_buf, down_qx, n_ff);
        for (size_t b = 0; b < dnb; b++) {
            down_qx_d[b] = fp16_to_fp32(down_qx[b].d);
        }
    }

    /* Down projection */
    float *out = ctx->expert_out + (size_t)i * dim;
    matmul_q8_seq(out, down_qx, down_qx_d, down_w, n_ff, dim, ctx->type_down);

    /* Scale by expert weight */
#ifdef PICOLM_AVX512
    {
        __m512 bw = _mm512_set1_ps(w_i);
        int di = 0;
        for (; di + 23 < dim; di += 16) {
            __m512 v0 = _mm512_loadu_ps(out + di);
            __m512 v1 = _mm512_loadu_ps(out + di + 8);
            _mm512_storeu_ps(out + di, _mm512_mul_ps(bw, v0));
            _mm512_storeu_ps(out + di + 8, _mm512_mul_ps(bw, v1));
        }
        for (; di + 15 < dim; di += 16) {
            __m512 v = _mm512_loadu_ps(out + di);
            _mm512_storeu_ps(out + di, _mm512_mul_ps(bw, v));
        }
        for (; di < dim; di++) out[di] *= w_i;
    }
#elif defined(PICOLM_AVX2)
    {
        __m256 bw = _mm256_set1_ps(w_i);
        int di = 0;
        for (; di + 19 < dim; di += 16) {
            __m256 v0 = _mm256_loadu_ps(out + di);
            __m256 v1 = _mm256_loadu_ps(out + di + 4);
            __m256 v2 = _mm256_loadu_ps(out + di + 8);
            __m256 v3 = _mm256_loadu_ps(out + di + 12);
            _mm256_storeu_ps(out + di, _mm256_mul_ps(bw, v0));
            _mm256_storeu_ps(out + di + 4, _mm256_mul_ps(bw, v1));
            _mm256_storeu_ps(out + di + 8, _mm256_mul_ps(bw, v2));
            _mm256_storeu_ps(out + di + 12, _mm256_mul_ps(bw, v3));
        }
        for (; di < dim; di++) out[di] *= w_i;
    }
#elif defined(PICOLM_AVX)
    {
        __m128 bw = _mm_set1_ps(w_i);
        int di = 0;
        for (; di + 7 < dim; di += 8) {
            __m128 v0 = _mm_loadu_ps(out + di);
            __m128 v1 = _mm_loadu_ps(out + di + 4);
            _mm_storeu_ps(out + di, _mm_mul_ps(bw, v0));
            _mm_storeu_ps(out + di + 4, _mm_mul_ps(bw, v1));
        }
        for (; di < dim; di++) out[di] *= w_i;
    }
#elif defined(PICOLM_SSE2)
    {
        __m128 bw = _mm_set1_ps(w_i);
        int di = 0;
        for (; di + 3 < dim; di += 4) {
            __m128 v = _mm_loadu_ps(out + di);
            _mm_storeu_ps(out + di, _mm_mul_ps(bw, v));
        }
        for (; di < dim; di++) out[di] *= w_i;
    }
#elif defined(PICOLM_NEON)
    {
        float32x4_t bw = vdupq_n_f32(w_i);
        int di = 0;
        for (; di + 3 < dim; di += 4) {
            float32x4_t v = vld1q_f32(out + di);
            vst1q_f32(out + di, vmulq_f32(bw, v));
        }
        for (; di < dim; di++) out[di] *= w_i;
    }
#else
    for (int d = 0; d < dim; d++) out[d] *= w_i;
#endif
}

/* MoE forward pass: router + top-K expert selection + SwiGLU per expert + shared expert.
 * Input: x[n_embd], Output: residual[n_embd] (additive to input) */
/* Optimized MoE forward: pre-quantize x, Q8xQ8 dot products for gate+up,
 * AVX-512 vectorized accumulation. Experts processed in parallel via
 * tensor_parallel_for for multi-threaded generation. */
static void moe_forward(model_t *m, run_state_t *s, const float *x, float *residual,
                        const layer_weights_t *lw) {
    model_config_t *c = &m->config;
    int dim = c->n_embd;
    int n_ff = c->n_ff_exp;
    int n_expert = c->n_expert;
    int n_used = c->n_expert_used;
    float *logits = s->expert_logits;
    int *ids = s->expert_ids;
    float *weights = s->expert_weights;
    float *moe_out = s->moe_out;
    float *expert_out = NULL;

    /* 1. Router: logits = x @ ffn_gate_inp  [n_embd, n_expert] -> [n_expert] */
    matmul(logits, (float *)x, lw->ffn_gate_inp, dim, n_expert, lw->type_ffn_gate_inp);

    /* 2. Softmax over logits */
    {
        float max_l = logits[0];
        for (int i = 1; i < n_expert; i++) {
            if (logits[i] > max_l) max_l = logits[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < n_expert; i++) {
            logits[i] = expf(logits[i] - max_l);
            sum += logits[i];
        }
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < n_expert; i++) {
            logits[i] *= inv_sum;
        }
    }

    /* 3. Find top-K experts (simple selection sort for small K) */
    {
        int idx[256];
        for (int i = 0; i < n_expert; i++) idx[i] = i;
        for (int i = 0; i < n_used; i++) {
            int best = i;
            for (int j = i + 1; j < n_expert; j++) {
                if (logits[idx[j]] > logits[idx[best]]) best = j;
            }
            { int t = idx[i]; idx[i] = idx[best]; idx[best] = t; }
            ids[i] = idx[i];
            weights[i] = logits[idx[i]];
        }
    }

    /* 3b. Normalize top-K weights by their sum */
    {
        float wsum = 0.0f;
        for (int i = 0; i < n_used; i++) wsum += weights[i];
        float inv_wsum = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;
        for (int i = 0; i < n_used; i++) weights[i] *= inv_wsum;
    }

    /* 5. Pre-quantize input x to Q8_0 ONCE (Phase A + D).
     * All expert gate+up projections reuse this quantized buffer via matmul_q8,
     * saving 16 redundant quantizations per MoE layer. */
    {
        size_t nb = dim / 32;
        block_q8_0 *qx = s->shared_qx;
        float *qx_d = s->shared_qx_d;
        quantize_row_q8_0(x, qx, dim);
        for (size_t bi = 0; bi < nb; bi++) {
            qx_d[bi] = fp16_to_fp32(qx[bi].d);
        }

        gguf_type_t type_gate = lw->type_ffn_gate_exps;
        gguf_type_t type_up = lw->type_ffn_up_exps;
        gguf_type_t type_down = lw->type_ffn_down_exps;

        /* Parallel expert dispatch: each expert runs in its own thread.
         * matmul_q8_seq is used (not matmul_q8) to avoid racing on the
         * global n_threads and re-entering the thread pool.
         * Use calloc instead of alloca to avoid Windows stack overflow. */
        expert_out = (float *)calloc((size_t)n_used * dim, sizeof(float));
        if (!expert_out) { fprintf(stderr, "OOM: expert_out\n"); return; }

        moe_expert_ctx ctx = {
            .qx = s->shared_qx,
            .qx_d = s->shared_qx_d,
            .dim = dim,
            .n_ff = n_ff,
            .ids = ids,
            .weights = weights,
            .type_gate = type_gate,
            .type_up = type_up,
            .type_down = type_down,
            .gate_exps = lw->ffn_gate_exps,
            .up_exps = lw->ffn_up_exps,
            .down_exps = lw->ffn_down_exps,
            .expert_out = expert_out,
            .s = s,
        };

        tensor_parallel_for(n_used, moe_expert_worker, &ctx);

        /* Reduce per-expert outputs into moe_out */
        memset(moe_out, 0, dim * sizeof(float));
        for (int i = 0; i < n_used; i++) {
            float *eo = expert_out + (size_t)i * dim;
#ifdef PICOLM_AVX512
            {
                int di = 0;
                for (; di + 23 < dim; di += 16) {
                    __m512 v0 = _mm512_loadu_ps(moe_out + di);
                    __m512 v1 = _mm512_loadu_ps(eo + di);
                    __m512 v2 = _mm512_loadu_ps(moe_out + di + 8);
                    __m512 v3 = _mm512_loadu_ps(eo + di + 8);
                    _mm512_storeu_ps(moe_out + di, _mm512_add_ps(v0, v1));
                    _mm512_storeu_ps(moe_out + di + 8, _mm512_add_ps(v2, v3));
                }
                for (; di < dim; di++) moe_out[di] += eo[di];
            }
#elif defined(PICOLM_AVX2)
            {
                int di = 0;
                for (; di + 20 < dim; di += 16) {
                    __m256 v0 = _mm256_loadu_ps(moe_out + di);
                    __m256 v1 = _mm256_loadu_ps(eo + di);
                    __m256 v2 = _mm256_loadu_ps(moe_out + di + 4);
                    __m256 v3 = _mm256_loadu_ps(eo + di + 4);
                    __m256 v4 = _mm256_loadu_ps(moe_out + di + 8);
                    __m256 v5 = _mm256_loadu_ps(eo + di + 8);
                    __m256 v6 = _mm256_loadu_ps(moe_out + di + 12);
                    __m256 v7 = _mm256_loadu_ps(eo + di + 12);
                    _mm256_storeu_ps(moe_out + di, _mm256_add_ps(v0, v1));
                    _mm256_storeu_ps(moe_out + di + 4, _mm256_add_ps(v2, v3));
                    _mm256_storeu_ps(moe_out + di + 8, _mm256_add_ps(v4, v5));
                    _mm256_storeu_ps(moe_out + di + 12, _mm256_add_ps(v6, v7));
                }
                for (; di < dim; di++) moe_out[di] += eo[di];
            }
#elif defined(PICOLM_AVX)
            {
                int di = 0;
                for (; di + 7 < dim; di += 8) {
                    __m128 v0 = _mm_loadu_ps(moe_out + di);
                    __m128 v1 = _mm_loadu_ps(eo + di);
                    __m128 v2 = _mm_loadu_ps(moe_out + di + 4);
                    __m128 v3 = _mm_loadu_ps(eo + di + 4);
                    _mm_storeu_ps(moe_out + di, _mm_add_ps(v0, v1));
                    _mm_storeu_ps(moe_out + di + 4, _mm_add_ps(v2, v3));
                }
                for (; di < dim; di++) moe_out[di] += eo[di];
            }
#elif defined(PICOLM_SSE2)
            {
                int di = 0;
                for (; di + 3 < dim; di += 4) {
                    __m128 v0 = _mm_loadu_ps(moe_out + di);
                    __m128 v1 = _mm_loadu_ps(eo + di);
                    _mm_storeu_ps(moe_out + di, _mm_add_ps(v0, v1));
                }
                for (; di < dim; di++) moe_out[di] += eo[di];
            }
#elif defined(PICOLM_NEON)
            {
                int di = 0;
                for (; di + 3 < dim; di += 4) {
                    float32x4_t v0 = vld1q_f32(moe_out + di);
                    float32x4_t v1 = vld1q_f32(eo + di);
                    vst1q_f32(moe_out + di, vaddq_f32(v0, v1));
                }
                for (; di < dim; di++) moe_out[di] += eo[di];
            }
#else
            for (int d = 0; d < dim; d++) moe_out[d] += eo[d];
#endif
        }
    }

    /* 6. Shared expert — Q8×Q8 with pre-allocated buffers */
    {
        int saved_threads = tensor_get_n_threads();
        tensor_set_n_threads(1);
        size_t nb = dim / 32;
        quantize_row_q8_0(x, s->shared_qx, dim);
        for (size_t bi = 0; bi < nb; bi++) {
            s->shared_qx_d[bi] = fp16_to_fp32(s->shared_qx[bi].d);
        }
        matmul_q8(s->hb, s->shared_qx, s->shared_qx_d, lw->ffn_gate_shexp, dim, c->n_ff_shexp, lw->type_ffn_gate_shexp);
        matmul_q8(s->hb2, s->shared_qx, s->shared_qx_d, lw->ffn_up_shexp, dim, c->n_ff_shexp, lw->type_ffn_up_shexp);
        silu(s->hb, c->n_ff_shexp);
        elemwise_mul(s->hb, s->hb, s->hb2, c->n_ff_shexp);
        {
            size_t dnb = c->n_ff_shexp / 32;
            quantize_row_q8_0(s->hb, s->shared_down_qx, c->n_ff_shexp);
            for (size_t bi = 0; bi < dnb; bi++) {
                s->shared_down_qx_d[bi] = fp16_to_fp32(s->shared_down_qx[bi].d);
            }
            matmul_q8(s->xb2, s->shared_down_qx, s->shared_down_qx_d, lw->ffn_down_shexp, c->n_ff_shexp, dim, lw->type_ffn_down_shexp);
        }
        tensor_set_n_threads(saved_threads);
    }

    /* Shared expert sigmoid gate: sigmoid(x @ ffn_gate_inp_shexp) */
    {
        float gate_val;
        matmul(&gate_val, (float *)x, lw->ffn_gate_inp_shexp, dim, 1, lw->type_ffn_gate_inp_shexp);
        gate_val = 1.0f / (1.0f + expf(-gate_val));
        for (int d = 0; d < dim; d++) s->xb2[d] *= gate_val;
    }

    /* 7. Combine: moe_out + shared */
    for (int d = 0; d < dim; d++) {
        residual[d] = moe_out[d] + s->xb2[d];
    }
    free(expert_out);
}


/* moe_forward_batch: mm_id-style batched MoE forward pass.
 *
 * Strategy: follow llama.cpp's ggml_mul_mat_id pattern. For each
 * projection (gate, up, down), sweep all 256 experts linearly.
 * For each expert, gather the tokens that selected it and compute
 * the projection. This ensures each expert's ~1.1MB weights are
 * streamed from RAM exactly once per layer.
 *
 * Output layout: [n_tokens * n_used * n_ff] for gate/up,
 *                [n_tokens * n_used * dim] for down.
 * Final accumulation is in Top-K order per token.
 *
 * x_batch:        input [n_tokens * dim], stride = dim
 * residual_batch: output [n_tokens * dim], stride = dim
 */
static void moe_forward_batch(model_t *m, run_state_t *s,
                              const float *x_batch, float *residual_batch,
                              int n_tokens, const layer_weights_t *lw) {
    model_config_t *c = &m->config;
    int dim = c->n_embd;
    int n_ff = c->n_ff_exp;
    int n_expert = c->n_expert;
    int n_used = c->n_expert_used;

    /* Single-token fast path: use moe_forward directly.
     * The batched path has overhead from routing map setup and tensor_parallel_for
     * dispatch across 256 experts that outweighs the benefit for a single token. */
    if (n_tokens == 1) {
        moe_forward(m, s, (const float *)x_batch, residual_batch, lw);
        return;
    }

    int saved_threads = tensor_get_n_threads();
    /* Phase 1-2: routing and quantization are lightweight, keep single-threaded */
    tensor_set_n_threads(1);

    /* Pre-allocated quantization buffer variables */
    int q8_buf_per_token = s->moe_q8_buf_per_token;
    int qx_d_off = s->moe_qx_d_off;
    float *qx_all = s->moe_qx_all;

    /* Down projection Q8_0 per-token buffer size (for n_ff, not n_embd) */
    size_t down_q8_rb = n_ff / 32;
    size_t down_q8_data_off = (down_q8_rb * sizeof(block_q8_0) + sizeof(float) - 1) / sizeof(float);
    int down_q8_per_token = (int)(down_q8_data_off + down_q8_rb);

    /* On-demand mm_id buffers (sized to actual batch size).
     * Reallocate if batch size has grown since last call. */
    size_t gateup_sz = (size_t)n_tokens * n_used * n_ff * sizeof(float);
    size_t down_sz = (size_t)n_tokens * n_used * dim * sizeof(float);
    float *mm_gate_out = s->mm_gate_out;
    float *mm_up_out = s->mm_up_out;
    float *mm_down_out = s->mm_down_out;

    /* Allocate or grow if needed */
    if (!mm_gate_out || s->mm_gateup_alloc < gateup_sz) {
#ifdef _WIN32
        _aligned_free(mm_gate_out);
        _aligned_free(mm_up_out);
        _aligned_free(mm_down_out);
#else
        free(mm_gate_out);
        free(mm_up_out);
        free(mm_down_out);
#endif
#ifdef _WIN32
        s->mm_gate_out = mm_gate_out = (float *)_aligned_malloc(gateup_sz + 4095, 64);
        s->mm_up_out = mm_up_out = (float *)_aligned_malloc(gateup_sz + 4095, 64);
        s->mm_down_out = mm_down_out = (float *)_aligned_malloc(down_sz + 4095, 64);
#else
        s->mm_gate_out = mm_gate_out = (float *)aligned_alloc(64, gateup_sz + 4095);
        s->mm_up_out = mm_up_out = (float *)aligned_alloc(64, gateup_sz + 4095);
        s->mm_down_out = mm_down_out = (float *)aligned_alloc(64, down_sz + 4095);
#endif
        s->mm_gateup_alloc = gateup_sz;
        s->mm_down_alloc = down_sz;
    } else if (s->mm_down_alloc < down_sz) {
#ifdef _WIN32
        _aligned_free(mm_down_out);
#else
        free(mm_down_out);
#endif
#ifdef _WIN32
        s->mm_down_out = mm_down_out = (float *)_aligned_malloc(down_sz + 4095, 64);
#else
        s->mm_down_out = mm_down_out = (float *)aligned_alloc(64, down_sz + 4095);
#endif
        s->mm_down_alloc = down_sz;
    }

    /* Per-token expert routing tables: [n_tokens][n_used] */
    int all_ids[n_tokens][8];       /* 8 = max n_expert_used */
    float all_weights[n_tokens][8];

    /* Per-token: number of experts assigned (for variable top-K) */
    int n_experts_per_token[n_tokens];

    /* ---- Phase 1: Route all tokens ---- */
    {
        int *idx = (int *)malloc(n_expert * sizeof(int));

        /* Batched router: all tokens through ffn_gate_inp in one matmul_batch */
        {
            float *logits_batch = s->expert_logits;
            matmul_batch(logits_batch, (float *)x_batch, n_tokens,
                lw->ffn_gate_inp, dim, n_expert, lw->type_ffn_gate_inp);

            for (int t = 0; t < n_tokens; t++) {
                float *logits = logits_batch + t * n_expert;
                softmax(logits, n_expert);

                /* Top-K selection */
                {
                    int *ids = all_ids[t];
                    float *weights = all_weights[t];
                for (int i = 0; i < n_expert; i++) idx[i] = i;
                for (int i = 0; i < n_used; i++) {
                    int best = i;
                    for (int j = i + 1; j < n_expert; j++) {
                        if (logits[idx[j]] > logits[idx[best]]) best = j;
                    }
                    { int tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp; }
                    ids[i] = idx[i];
                    weights[i] = logits[idx[i]];
                }

                /* Normalize weights by their sum */
                {
                    float wsum = 0.0f;
                    for (int i = 0; i < n_used; i++) wsum += weights[i];
                    float inv_wsum = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;
                    for (int i = 0; i < n_used; i++) weights[i] *= inv_wsum;
                }
                }
            n_experts_per_token[t] = n_used;
            }
        }
        free(idx);
    }

    /* ---- Phase 2: Quantize all token inputs to Q8_0 (pre-allocated) ---- */
    {
        size_t q8_row_blocks = dim / 32;
        for (int t = 0; t < n_tokens; t++) {
            float *tbuf = qx_all + t * q8_buf_per_token;
            block_q8_0 *qx = (block_q8_0 *)tbuf;
            float *qx_d = tbuf + qx_d_off;
            quantize_row_q8_0(x_batch + t * dim, qx, dim);
            for (size_t bi = 0; bi < q8_row_blocks; bi++) {
                qx_d[bi] = fp16_to_fp32(qx[bi].d);
            }
        }
    }

    /* ---- Phase 2b: Precompute routing map for mm_id dispatch ---- */
    /* expert_assignments[eid * n_tokens + a] = packed (token << 8 | slot) */
    {
        memset(s->expert_counts, 0, n_expert * sizeof(int));
        for (int t = 0; t < n_tokens; t++) {
            int n_tok_experts = n_experts_per_token[t];
            for (int sl = 0; sl < n_tok_experts; sl++) {
                int eid = all_ids[t][sl];
                int idx = s->expert_counts[eid];
                s->expert_assignments[eid * n_tokens + idx] = (t << 8) | sl;
                s->expert_counts[eid]++;
            }
        }
    }

    /* ---- Phase 3: mm_id gate+up projections ---- */
    {
        tensor_set_n_threads(saved_threads);
        gguf_type_t type = lw->type_ffn_gate_exps;
        matmul_mm_id_gate_up(mm_gate_out, mm_up_out,
            qx_all, qx_d_off, q8_buf_per_token,
            lw->ffn_gate_exps, lw->ffn_up_exps,
            s->expert_assignments, s->expert_counts,
            n_tokens, n_used, dim, n_ff, n_expert, type);
        tensor_set_n_threads(1);
    }

    /* ---- Phase 4: SwiGLU per (token, slot) ---- */
    {
        for (int t = 0; t < n_tokens; t++) {
            for (int sl = 0; sl < n_used; sl++) {
                float *g = mm_gate_out + (size_t)t * n_used * n_ff + (size_t)sl * n_ff;
                float *u = mm_up_out + (size_t)t * n_used * n_ff + (size_t)sl * n_ff;
                silu(g, n_ff);
                elemwise_mul(g, g, u, n_ff);
            }
        }
    }
    /* ---- Phase 5: mm_id down projections ---- */
    {
        tensor_set_n_threads(saved_threads);
        gguf_type_t type = lw->type_ffn_down_exps;
        matmul_mm_id_down(mm_down_out, mm_gate_out,
            lw->ffn_down_exps,
            s->expert_assignments, s->expert_counts,
            n_tokens, n_used, dim, n_ff, n_expert, type,
            s->mm_scratch_qx, s->mm_scratch_qx_d,
            s->mm_down_qx_all, s->mm_down_qx_d_all, down_q8_per_token);
        tensor_set_n_threads(1);
    }
    /* ---- Phase 6: Weighted accumulation in Top-K order ---- */
    {
        for (int t = 0; t < n_tokens; t++) {
            float *out = residual_batch + t * dim;
            memset(out, 0, dim * sizeof(float));

            for (int sl = 0; sl < n_used; sl++) {
                float w_i = all_weights[t][sl];
                float *expert_out = mm_down_out + (size_t)t * n_used * dim + (size_t)sl * dim;

#ifdef PICOLM_AVX512
                {
                    __m512 bw = _mm512_set1_ps(w_i);
                    int di = 0;
                    for (; di + 23 < dim; di += 16) {
                        __m512 v0 = _mm512_loadu_ps(out + di);
                        __m512 v1 = _mm512_loadu_ps(expert_out + di);
                        __m512 v2 = _mm512_loadu_ps(out + di + 8);
                        __m512 v3 = _mm512_loadu_ps(expert_out + di + 8);
                        _mm512_storeu_ps(out + di,
                            _mm512_add_ps(v0, _mm512_mul_ps(bw, v1)));
                        _mm512_storeu_ps(out + di + 8,
                            _mm512_add_ps(v2, _mm512_mul_ps(bw, v3)));
                    }
                    for (; di < dim; di++)
                        out[di] += w_i * expert_out[di];
                }
#elif defined(PICOLM_AVX2)
                {
                    __m256 bw = _mm256_set1_ps(w_i);
                    int di = 0;
                    for (; di + 20 < dim; di += 16) {
                        __m256 v0 = _mm256_loadu_ps(out + di);
                        __m256 v1 = _mm256_loadu_ps(expert_out + di);
                        __m256 v2 = _mm256_loadu_ps(out + di + 4);
                        __m256 v3 = _mm256_loadu_ps(expert_out + di + 4);
                        __m256 v4 = _mm256_loadu_ps(out + di + 8);
                        __m256 v5 = _mm256_loadu_ps(expert_out + di + 8);
                        __m256 v6 = _mm256_loadu_ps(out + di + 12);
                        __m256 v7 = _mm256_loadu_ps(expert_out + di + 12);
                        _mm256_storeu_ps(out + di,
                            _mm256_add_ps(v0, _mm256_mul_ps(bw, v1)));
                        _mm256_storeu_ps(out + di + 4,
                            _mm256_add_ps(v2, _mm256_mul_ps(bw, v3)));
                        _mm256_storeu_ps(out + di + 8,
                            _mm256_add_ps(v4, _mm256_mul_ps(bw, v5)));
                        _mm256_storeu_ps(out + di + 12,
                            _mm256_add_ps(v6, _mm256_mul_ps(bw, v7)));
                    }
                    for (; di < dim; di++)
                        out[di] += w_i * expert_out[di];
                }
#elif defined(PICOLM_NEON)
                {
                    float32x4_t bw = vdupq_n_f32(w_i);
                    int di = 0;
                    for (; di + 3 < dim; di += 4) {
                        float32x4_t v0 = vld1q_f32(out + di);
                        float32x4_t v1 = vld1q_f32(expert_out + di);
                        vst1q_f32(out + di, vaddq_f32(v0, vmulq_f32(bw, v1)));
                    }
                    for (; di < dim; di++)
                        out[di] += w_i * expert_out[di];
                }
#else
                for (int d = 0; d < dim; d++)
                    out[d] += w_i * expert_out[d];
#endif
            }
        }
    }

    /* ---- Phase 7: Shared expert (batched via matmul_q8_batch) ---- */
    {
        int sh_ff = c->n_ff_shexp;
        float *sh_gate = s->sh_gate;
        float *sh_up = s->sh_up;

        matmul_q8_batch(sh_gate, qx_all, qx_d_off, q8_buf_per_token,
                        lw->ffn_gate_shexp, dim, sh_ff, n_tokens, lw->type_ffn_gate_shexp);
        matmul_q8_batch(sh_up, qx_all, qx_d_off, q8_buf_per_token,
                        lw->ffn_up_shexp, dim, sh_ff, n_tokens, lw->type_ffn_up_shexp);

        for (int t = 0; t < n_tokens; t++) {
            float *g = sh_gate + t * sh_ff;
            float *u = sh_up + t * sh_ff;
            float *out = residual_batch + t * dim;

            silu(g, sh_ff);
            elemwise_mul(g, g, u, sh_ff);

            {
                size_t dnb = sh_ff / 32;
                quantize_row_q8_0(g, s->shared_down_qx, sh_ff);
                for (size_t bi = 0; bi < dnb; bi++) {
                    s->shared_down_qx_d[bi] = fp16_to_fp32(s->shared_down_qx[bi].d);
                }
                matmul_q8(s->xb2, s->shared_down_qx, s->shared_down_qx_d, lw->ffn_down_shexp, sh_ff, dim, lw->type_ffn_down_shexp);
                }

            {
                float gate_val;
                matmul(&gate_val, x_batch + t * dim, lw->ffn_gate_inp_shexp, dim, 1, lw->type_ffn_gate_inp_shexp);
                gate_val = 1.0f / (1.0f + expf(-gate_val));
                for (int d = 0; d < dim; d++) s->xb2[d] *= gate_val;
            }

            for (int d = 0; d < dim; d++) out[d] += s->xb2[d];
        }
    }

    tensor_set_n_threads(saved_threads);
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

/* ================================================================
 * Gemma-3n forward pass
 * ================================================================ */

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
static void gemma3n_router(float *out, float *inp, int n_embd, int n_altup,
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

    float *cos_pos = s->rope_cos + (size_t)pos * half_dim;
    float *sin_pos = s->rope_sin + (size_t)pos * half_dim;

    /* 1. Token embedding lookup, scaled by sqrt(n_embd) */
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
        dequantize_row(embd_row, s->x, dim, w->type_token_embd);
        for (int i = 0; i < dim; i++) s->x[i] *= sqrt_dim;
    }
    if (pos == 0 && getenv("PICOLM_DBG")) {
        fprintf(stderr, "DBG rms_norm_eps=%e\n", rms_norm_eps);
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
        fprintf(stderr, "DBG per_layer_inp[0]_rms=%.4f\n", sqrt(pm/n_embd_altup));
        double pm2 = 0; for(int i=0;i<n_embd_altup;i++){double v=s->gemma3n_per_layer_inp[c->n_layers*n_embd_altup-i-1]; pm2+=v*v;}
        fprintf(stderr, "DBG per_layer_inp[last]_rms=%.4f\n", sqrt(pm2/n_embd_altup));
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
            /* altup_proj: [n_embd, n_embd, n_altup-1], take slice a -> [n_embd, n_embd] */
            float *dst = s->gemma3n_altup_state + a_dst * dim;
            matmul(dst, s->x, (const float *)w->altup_proj + a * dim * dim,
                   dim, dim, w->type_altup_proj);
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

    /* 4. Transformer layers */
    {
        int stop_after = -999;
        const char *stop_env = getenv("PICOLM_STOP_LAYER");
        if (stop_env) stop_after = atoi(stop_env);
        for (int l = 0; l < c->n_layers; l++) {
            if (pos == 0 && getenv("PICOLM_DBG")) {
                fprintf(stderr, "DBG layer=%d altup_state_rms={", l);
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
                fprintf(stderr, "DBG L0 modalities={");
                for(int r=0;r<n_altup;r++) fprintf(stderr,"%.4f%s",s->gemma3n_router_out[r],r<n_altup-1?",":"");
                fprintf(stderr, "}\n");
                fprintf(stderr, "DBG L0 predict_coefs={");
                for(int a=0;a<n_altup*n_altup;a++) fprintf(stderr,"%.4f%s",all_coefs[a],a<n_altup*n_altup-1?",":"");
                fprintf(stderr, "}\n");
                fprintf(stderr, "DBG L0 raw_predict_coef[0:4]={");
                for(int a=0;a<4;a++) fprintf(stderr,"%.6f%s",((const float *)lw->altup_predict_coef)[a],a<3?",":"");
                fprintf(stderr, "}\n");
                fprintf(stderr, "DBG L0 raw_correct_coef[0:4]={");
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
                    fprintf(stderr,"DBG L0 pred[%d]_rms=%.4f\n", a, sqrt(pm/dim));
                }
                fprintf(stderr, "DBG L0 active_pred_rms=%.4f [0:5]={", sqrt(apm/dim));
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
                    fprintf(stderr, "DBG L0 laurel_post_norm_rms=%.4f laurel_normed_rms=%.4f\n", sqrt(wm/dim), sqrt(am/dim));
                }
                /* Residual: laurel_out + x_normed (not active_pred!) */
                for (int i = 0; i < dim; i++) s->gemma3n_laurel_out[i] += s->xb[i];
            }
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<dim;i++){double v=s->xb[i]; am+=v*v;}
                fprintf(stderr, "DBG L0 attn_norm_rms=%.4f\n", sqrt(am/dim));
                int skip_laurel = getenv("PICOLM_SKIP_LAUREL") ? 1 : 0;
                if (!skip_laurel) {
                    double lm = 0; for(int i=0;i<dim;i++){double v=s->gemma3n_laurel_out[i]; lm+=v*v;}
                    fprintf(stderr, "DBG L0 laurel_out_rms=%.4f\n", sqrt(lm/dim));
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

            /* RoPE */
            rope(s->q, s->xb2, head_dim, n_heads, n_kv_heads, cos_pos, sin_pos, c->rope_type, half_dim);

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

            /* RoPE on Q only */
            rope(s->q, s->xb2, head_dim, n_heads, n_kv_heads, cos_pos, sin_pos, c->rope_type, half_dim);

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
            fprintf(stderr, "DBG L0 attn_post_norm_weight_rms=%.2f\n", sqrt(wm/dim));
        }
        rmsnorm(s->xb, s->xb2, s->attn_post_norm_w[l], dim, rms_norm_eps);
        if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
            double am = 0; for(int i=0;i<dim;i++){double v=s->xb[i]; am+=v*v;}
            fprintf(stderr, "DBG L0 attn_post_norm_rms=%.2f\n", sqrt(am/dim));
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
                fprintf(stderr, "DBG L0 attn_laurel_rms=%.2f\n", sqrt(am/dim));
            }
        }

        /* FFN */
        {
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double wm = 0; for(int i=0;i<dim;i++){double v=s->post_attn_norm_w[l][i]; wm+=v*v;}
                fprintf(stderr, "DBG L0 ffn_norm_weight_rms=%.2f [0:5]={", sqrt(wm/dim));
                for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->post_attn_norm_w[l][i],i<4?",":"");
                fprintf(stderr, "}\n");
            }
            rmsnorm(s->xb, s->x, s->post_attn_norm_w[l], dim, rms_norm_eps);
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<dim;i++){double v=s->xb[i]; am+=v*v;}
                fprintf(stderr, "DBG L0 ffn_norm_rms=%.2f\n", sqrt(am/dim));
            }
            matmul(s->hb, s->xb, lw->ffn_gate, dim, n_ffn, lw->type_ffn_gate);
            /* Activation sparsity (gaussian_topk) for first n_layer_sparsity layers */
            if (c->is_gemma3n && l < c->n_layer_sparsity && c->n_layer_sparsity > 0) {
                float *gp = s->hb;
                /* mean = mean(gate_proj) */
                float mean = 0.0f;
                for (int i = 0; i < n_ffn; i++) mean += gp[i];
                mean /= (float)n_ffn;
                /* std = sqrt(sum((x - mean)^2) / (n_ffn - 1)) */
                float var = 0.0f;
                for (int i = 0; i < n_ffn; i++) { float d = gp[i] - mean; var += d * d; }
                var /= (float)(n_ffn - 1);
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
                fprintf(stderr, "DBG L0 ffn_down_rms=%.2f\n", sqrt(am/dim));
            }
            /* post_ffw_norm */
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double wm = 0; for(int i=0;i<dim;i++){double v=s->post_ffw_norm_w[l][i]; wm+=v*v;}
                fprintf(stderr, "DBG L0 post_ffw_norm_weight_rms=%.2f [0:3]={", sqrt(wm/dim));
                for(int i=0;i<3;i++) fprintf(stderr,"%.4f%s",s->post_ffw_norm_w[l][i],i<2?",":"");
                fprintf(stderr, "}\n");
            }
            rmsnorm(s->xb, s->xb, s->post_ffw_norm_w[l], dim, rms_norm_eps);
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<dim;i++){double v=s->xb[i]; am+=v*v;}
                fprintf(stderr, "DBG L0 ffn_post_norm_rms=%.2f\n", sqrt(am/dim));
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
            fprintf(stderr, "DBG L0 activated_rms=%.2f\n", sqrt(am/dim));
        }
        {
            /* Router on activated state */
            gemma3n_router(s->gemma3n_router_out, s->xb, dim, n_altup,
                          s->altup_router_norm_w[l], s->altup_router_w[l],
                          rms_norm_eps, s->xb2);
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                fprintf(stderr, "DBG L0 correct_modalities={");
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
                fprintf(stderr, "DBG L0 correct_coefs={");
                for(int a=0;a<n_altup;a++) fprintf(stderr,"%.4f%s",correct_coefs[a],a<n_altup-1?",":"");
                fprintf(stderr, "}\n");
            }

            /* innovation = activated - predictions[i_altup_act] */
            float *active_prediction = predictions + i_altup_act * dim;
            for (int d = 0; d < dim; d++)
                s->hb[d] = s->xb[d] - active_prediction[d];

            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double im = 0; for(int i=0;i<dim;i++){double v=s->hb[i]; im+=v*v;}
                fprintf(stderr, "DBG L0 innovation_rms=%.4f\n", sqrt(im/dim));
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
                    fprintf(stderr, "DBG L0 corrected[%d]_rms=%.4f\n", a, sqrt(am/dim));
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
                fprintf(stderr, "DBG L0 altup_correct_scale_rms=%.4f [0:5]={", sqrt(sc_rms/dim));
                for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->altup_correct_scale_w[l][i],i<4?",":"");
                fprintf(stderr, "}\n");
            }
            for (int i = 0; i < dim; i++) first_pred[i] = corrected_active[i] * s->altup_correct_scale_w[l][i];

            /* inp_gate: [n_embd, n_embd_altup] */
            matmul(s->gemma3n_inp_gate_out, first_pred, lw->per_layer_inp_gate, dim, n_embd_altup, lw->type_per_layer_inp_gate);
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<n_embd_altup;i++){double v=s->gemma3n_inp_gate_out[i]; am+=v*v;}
                fprintf(stderr, "DBG L0 inp_gate_out_rms=%.4f (before gelu) [0:5]={", sqrt(am/n_embd_altup));
                for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->gemma3n_inp_gate_out[i],i<4?",":"");
                fprintf(stderr, "}\n");
            }
            gelu(s->gemma3n_inp_gate_out, n_embd_altup);  /* Gemma-3n uses GELU */
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<n_embd_altup;i++){double v=s->gemma3n_inp_gate_out[i]; am+=v*v;}
                fprintf(stderr, "DBG L0 inp_gate_out_rms=%.4f (after gelu) [0:5]={", sqrt(am/n_embd_altup));
                for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->gemma3n_inp_gate_out[i],i<4?",":"");
                fprintf(stderr, "}\n");
            }

            /* Multiply by per-layer input for this layer */
            float *layer_inp = s->gemma3n_per_layer_inp + l * n_embd_altup;
            for (int i = 0; i < n_embd_altup; i++) s->gemma3n_inp_gate_out[i] *= layer_inp[i];
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<n_embd_altup;i++){double v=s->gemma3n_inp_gate_out[i]; am+=v*v;}
                fprintf(stderr, "DBG L0 inp_gate_out_rms=%.4f (after gelu*inp)\n", sqrt(am/n_embd_altup));
            }

            /* per_layer_proj: [n_embd_altup, n_embd] */
            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                /* Check per_layer_proj weight at row 1331 */
                float max_w = 0; int max_w_row = 0;
                for(int i=0;i<n_embd_altup;i++) {
                    /* per_layer_proj is [n_embd_altup, n_embd] = [256, 2048], row i is at offset i*n_embd */
                    /* matmul computes out[d] = sum_j x[j] * W[d * n_embd_altup + j] */
                    /* Wait, matmul(out, x, W, n, d) = out[i] = sum_j x[j] * W[i*n + j] */
                    /* So W is [d, n] = [2048, 256], row d is at offset d*256 */
                }
                /* Actually let's just print the weight type */
                fprintf(stderr, "DBG L0 per_layer_proj type=%d\n", lw->type_per_layer_proj);
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
                fprintf(stderr, "DBG L0 per_layer_proj_out_rms=%.4f max|x|=%.4f@%d max|w|=%.4f@%d x2w2_mean=%.4f\n",
                    sqrt(am/dim), maxx, maxx_i, maxw, maxw_i, x2w2/dim);
            }
            rmsnorm(s->hb, s->hb, s->per_layer_post_norm_w[l], dim, rms_norm_eps);

            if (l == 0 && pos == 0 && getenv("PICOLM_DBG")) {
                double am = 0; for(int i=0;i<dim;i++){double v=s->hb[i]; am+=v*v;}
                double wn = 0; for(int i=0;i<dim;i++){double v=s->per_layer_post_norm_w[l][i]; wn+=v*v;}
                fprintf(stderr, "DBG L0 gating_out_rms=%.2f post_norm_w_rms=%.4f\n", sqrt(am/dim), sqrt(wn/dim));
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
    }
    } /* end of stop_layer scope */

    /* 5. ALTUP unembed: merge all altups back to single copy */
    {
        float *active = s->gemma3n_altup_state + i_altup_act * dim;
        float target_mag = gemma3n_calc_magnitude(active, dim);

        /* Start with active altup */
        memcpy(s->x, active, dim * sizeof(float));

        for (int a = 0; a < n_altup - 1; a++) {
            int a_src = (a < i_altup_act) ? a : a + 1;
            float *src = s->gemma3n_altup_state + a_src * dim;
            /* unembed: altup_unembd_proj * src */
            matmul(s->xb, src, (const float *)w->altup_unembd_proj + a * dim * dim,
                   dim, dim, w->type_altup_unembd_proj);
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

    /* 6. Final RMSNorm */
    if (pos <= 1 && getenv("PICOLM_DBG")) {
        double om = 0; for(int i=0;i<dim;i++){double v=s->x[i]; om+=v*v;}
        fprintf(stderr, "DBG pos=%d unembed_rms=%.4f [0:5]={", pos, sqrt(om/dim));
        for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->x[i],i<4?",":"");
        fprintf(stderr, "}\n");
    }
    rmsnorm(s->x, s->x, s->output_norm_w, dim, rms_norm_eps);

    if (pos <= 1 && getenv("PICOLM_DBG")) {
        double om = 0; for(int i=0;i<dim;i++){double v=s->x[i]; om+=v*v;}
        fprintf(stderr, "DBG pos=%d output_norm_rms=%.4f [0:5]={", pos, sqrt(om/dim));
        for(int i=0;i<5;i++) fprintf(stderr,"%.4f%s",s->x[i],i<4?",":"");
        fprintf(stderr, "}\n");
    }
    /* 7. Output projection -> logits */
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);

    /* 8. Logit soft-capping: tanh(logits / softcap) * softcap */
    if (c->f_final_logit_softcapping > 0) {
        float inv_cap = 1.0f / c->f_final_logit_softcapping;
        for (int i = 0; i < c->vocab_size; i++) {
            s->logits[i] = tanhf(s->logits[i] * inv_cap) * c->f_final_logit_softcapping;
        }
    }

    if (pos <= 1 && getenv("PICOLM_DBG")) {
        fprintf(stderr, "DBG pos=%d top5 logits: ", pos);
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

/* ================================================================
 * SSM forward pass helpers (Qwen3.5)
 * ================================================================ */

#ifdef DEBUG_SSM
static void dbg_vec(const char *tag, float *v, int n, int max_print) {
    int p = n < max_print ? n : max_print;
    for (int i = 0; i < p; i++) fprintf(stderr, "%.6f ", v[i]);
    fprintf(stderr, "\n");
}
#endif

/* Qwen3.5 GGUF v-head reordering (all v-head-indexed tensors).
 *
 * The GGUF converter reorders via _reorder_v_heads: a simple transpose.
 *   Sequential: [k0v0, k0v1, k0v2, ..., k1v0, ...]
 *   GGUF:       [v0*k, v1*k, ..., vn_vpk-1*k] for each k group
 *   GGUF_index = v * n_k + k  where  k = h / n_vpk,  v = h % n_vpk
 *
 * This applies uniformly to ALL v-head-indexed tensors:
 *   attn_gate_ssm, attn_qkv V portion, ssm_conv1d V channels,
 *   ssm_alpha, ssm_beta, ssm_out columns, dt_bias, ssm_a.
 */

/* Simple transpose: sequential head h -> GGUF index */
static inline int qwen35_vhead_gguf(int h, int n_vpk, int n_k) {
    int k = h / n_vpk;
    int v = h % n_vpk;
    return v * n_k + k;
}

/* Inverse: given GGUF head g, recover natural head h.
 * gguf: g = v*n_k + k, where k=h/n_vpk, v=h%n_vpk
 * inverse: v = g/n_k, k = g%n_k, h = k*n_vpk + v */
static inline int qwen35_vhead_natural(int g, int n_vpk, int n_k) {
    int v = g / n_k;
    int k = g % n_k;
    return k * n_vpk + v;
}

/* ---- SSM per-head task (threaded state recurrence) ----
 * Each of the n_v_heads has its own independent [d_state x d_state]
 * state block.  tensor_parallel_for dispatches one head per task.
 *
 * Layout note: ssm_state is stored [n_v_heads][d_state][d_state]
 * (row = contracted dim, col = output dim) with the row index being
 * the dimension summed over in sk/output.  This means sk and output
 * scan column-wise against row-major storage (stride d_state per
 * element).  The expert03 insight was that flipping the loop order
 * to row-outer makes all four steps (decay/sk/update/output)
 * contiguous.  However, the state matrix is shared with the
 * batched prefill path which assumes the original layout, so we
 * keep the existing layout here and instead compute the recurrence
 * with row-major access: d1-outer loop, d2-inner, which is still
 * reasonably cache-friendly for d_state=128. */
typedef struct {
    float *state;               /* [n_v_heads][d_state][d_state] */
    const float *q_conv, *k_conv; /* [n_k_heads][d_state], head-major */
    const float *v_conv;         /* [n_v_heads][head_v_dim], head-major */
    const float *gate_exp, *beta; /* [n_v_heads] */
    float *ssm_output;           /* [d_state][n_v_heads], dim-major */
    int d_state, head_v_dim, n_v_heads, repeat;
} ssm_head_ctx_t;

static void ssm_head_task(int h, void *ctxp) {
    ssm_head_ctx_t *ctx = (ssm_head_ctx_t *)ctxp;
    int d_state = ctx->d_state;
    int n_v_heads = ctx->n_v_heads;
    assert(d_state <= 256 && "ssm_head_task: stack scratch too large");
    assert(ctx->head_v_dim == d_state &&
           "ssm_head_task assumes head_v_dim == d_state");

    float *st = ctx->state + (size_t)h * d_state * d_state;
    float ge = ctx->gate_exp[h];
    int kh = h / ctx->repeat;
    const float *qh = ctx->q_conv + (size_t)kh * d_state;
    const float *khv = ctx->k_conv + (size_t)kh * d_state;
    const float *vh = ctx->v_conv + (size_t)h * ctx->head_v_dim;
    float bh = ctx->beta[h];

#ifdef PICOLM_AVX512
    /* ---- AVX-512 vectorized recurrence ----
     * Process 4 rows of the d_state x d_state state matrix simultaneously.
     * Each __m512 holds 16 floats. For d_state=128: 8 vector lanes per row,
     * 4 rows processed at once = 32 vector registers for the matrix.
     *
     * Layout: st[h] is [d_state][d_state] row-major.
     * We process 4 consecutive rows (r, r+1, r+2, r+3) together. */
    {
        int nr = d_state / 4; /* number of 4-row groups */
        __m512 ge_v = _mm512_set1_ps(ge);
        int d16 = d_state / 16; /* number of 16-float vector lanes per row */

        for (int g = 0; g < nr; g++) {
            int base = g * 4; /* row base in this group */
            /* Load 4 rows x d16 vectors = 4*d128 floats for this group */

            /* Phase 1: Decay + sk computation */
            /* sk[4 rows][16 lanes] = sum over columns: st[4r][c] * k[c] */
            __m512 sk0 = _mm512_setzero_ps(), sk1 = _mm512_setzero_ps();
            __m512 sk2 = _mm512_setzero_ps(), sk3 = _mm512_setzero_ps();

            for (int v = 0; v < d16; v++) {
                int col_base = v * 16;
                /* Load k[col_base..col_base+15] broadcast to decay/multiply */
                __m512 kv = _mm512_loadu_ps(khv + col_base);

                /* Row 0: decay + accumulate sk */
                __m512 r0 = _mm512_loadu_ps(st + base * d_state + col_base);
                r0 = _mm512_mul_ps(r0, ge_v);
                _mm512_storeu_ps(st + base * d_state + col_base, r0);
                sk0 = _mm512_fmadd_ps(r0, kv, sk0);

                /* Row 1 */
                __m512 r1 = _mm512_loadu_ps(st + (base+1) * d_state + col_base);
                r1 = _mm512_mul_ps(r1, ge_v);
                _mm512_storeu_ps(st + (base+1) * d_state + col_base, r1);
                sk1 = _mm512_fmadd_ps(r1, kv, sk1);

                /* Row 2 */
                __m512 r2 = _mm512_loadu_ps(st + (base+2) * d_state + col_base);
                r2 = _mm512_mul_ps(r2, ge_v);
                _mm512_storeu_ps(st + (base+2) * d_state + col_base, r2);
                sk2 = _mm512_fmadd_ps(r2, kv, sk2);

                /* Row 3 */
                __m512 r3 = _mm512_loadu_ps(st + (base+3) * d_state + col_base);
                r3 = _mm512_mul_ps(r3, ge_v);
                _mm512_storeu_ps(st + (base+3) * d_state + col_base, r3);
                sk3 = _mm512_fmadd_ps(r3, kv, sk3);
            }

            /* Horizontal reduce sk (sum across 16 lanes per __m512) */
            float sk0s = _mm512_reduce_add_ps(sk0);
            float sk1s = _mm512_reduce_add_ps(sk1);
            float sk2s = _mm512_reduce_add_ps(sk2);
            float sk3s = _mm512_reduce_add_ps(sk3);

            /* Phase 2: Compute delta = (v - sk) * beta */
            float d0 = (vh[base + 0] - sk0s) * bh;
            float d1 = (vh[base + 1] - sk1s) * bh;
            float d2 = (vh[base + 2] - sk2s) * bh;
            float d3 = (vh[base + 3] - sk3s) * bh;

            __m512 d0v = _mm512_set1_ps(d0);
            __m512 d1v = _mm512_set1_ps(d1);
            __m512 d2v = _mm512_set1_ps(d2);
            __m512 d3v = _mm512_set1_ps(d3);

            /* Phase 3: State update + output computation
             * state[r][c] += k[c] * d[r]  (rank-1 update)
             * output[r] = sum_c state[r][c] * q[c] */
            __m512 out0 = _mm512_setzero_ps(), out1 = _mm512_setzero_ps();
            __m512 out2 = _mm512_setzero_ps(), out3 = _mm512_setzero_ps();

            for (int v = 0; v < d16; v++) {
                int col_base = v * 16;
                __m512 kv = _mm512_loadu_ps(khv + col_base);
                __m512 qv = _mm512_loadu_ps(qh + col_base);

                /* Row 0: update state + accumulate output */
                __m512 r0 = _mm512_loadu_ps(st + base * d_state + col_base);
                r0 = _mm512_fmadd_ps(kv, d0v, r0);
                _mm512_storeu_ps(st + base * d_state + col_base, r0);
                out0 = _mm512_fmadd_ps(r0, qv, out0);

                /* Row 1 */
                __m512 r1 = _mm512_loadu_ps(st + (base+1) * d_state + col_base);
                r1 = _mm512_fmadd_ps(kv, d1v, r1);
                _mm512_storeu_ps(st + (base+1) * d_state + col_base, r1);
                out1 = _mm512_fmadd_ps(r1, qv, out1);

                /* Row 2 */
                __m512 r2 = _mm512_loadu_ps(st + (base+2) * d_state + col_base);
                r2 = _mm512_fmadd_ps(kv, d2v, r2);
                _mm512_storeu_ps(st + (base+2) * d_state + col_base, r2);
                out2 = _mm512_fmadd_ps(r2, qv, out2);

                /* Row 3 */
                __m512 r3 = _mm512_loadu_ps(st + (base+3) * d_state + col_base);
                r3 = _mm512_fmadd_ps(kv, d3v, r3);
                _mm512_storeu_ps(st + (base+3) * d_state + col_base, r3);
                out3 = _mm512_fmadd_ps(r3, qv, out3);
            }

            /* Horizontal reduce output */
            ctx->ssm_output[(size_t)base * n_v_heads + h] = _mm512_reduce_add_ps(out0);
            ctx->ssm_output[(size_t)(base+1) * n_v_heads + h] = _mm512_reduce_add_ps(out1);
            ctx->ssm_output[(size_t)(base+2) * n_v_heads + h] = _mm512_reduce_add_ps(out2);
            ctx->ssm_output[(size_t)(base+3) * n_v_heads + h] = _mm512_reduce_add_ps(out3);
        }
    }
#elif defined(PICOLM_AVX2) || defined(PICOLM_AVX)
    /* ---- AVX2/AVX vectorized recurrence ----
     * Process 4 rows of the d_state x d_state state matrix simultaneously.
     * Each __m256 holds 8 floats. For d_state=128: 16 vector lanes per row,
     * 4 rows processed at once = 64 vector registers for the matrix.
     *
     * Layout: st[h] is [d_state][d_state] row-major.
     * We process 4 consecutive rows (r, r+1, r+2, r+3) together.
     *
     * FMA256 macro: uses _mm256_fmadd_ps when FMA3 is available,
     * falls back to mul+add for AVX-only CPUs without FMA (e.g. Sandy Bridge). */
#ifdef PICOLM_FMA
    #define FMA256(a,b,c) _mm256_fmadd_ps((a),(b),(c))
#else
    #define FMA256(a,b,c) _mm256_add_ps(_mm256_mul_ps((a),(b)),(c))
#endif

    {
        int nr = d_state / 4; /* number of 4-row groups */
        __m256 ge_v = _mm256_set1_ps(ge);
        int d8 = d_state / 8; /* number of 8-float vector lanes per row */

        for (int g = 0; g < nr; g++) {
            int base = g * 4; /* row base in this group */

            /* Phase 1: Decay + sk computation */
            __m256 sk0 = _mm256_setzero_ps(), sk1 = _mm256_setzero_ps();
            __m256 sk2 = _mm256_setzero_ps(), sk3 = _mm256_setzero_ps();

            for (int v = 0; v < d8; v++) {
                int col_base = v * 8;
                __m256 kv = _mm256_loadu_ps(khv + col_base);

                /* Row 0 */
                __m256 r0 = _mm256_loadu_ps(st + base * d_state + col_base);
                r0 = _mm256_mul_ps(r0, ge_v);
                _mm256_storeu_ps(st + base * d_state + col_base, r0);
                sk0 = FMA256(r0, kv, sk0);

                /* Row 1 */
                __m256 r1 = _mm256_loadu_ps(st + (base+1) * d_state + col_base);
                r1 = _mm256_mul_ps(r1, ge_v);
                _mm256_storeu_ps(st + (base+1) * d_state + col_base, r1);
                sk1 = FMA256(r1, kv, sk1);

                /* Row 2 */
                __m256 r2 = _mm256_loadu_ps(st + (base+2) * d_state + col_base);
                r2 = _mm256_mul_ps(r2, ge_v);
                _mm256_storeu_ps(st + (base+2) * d_state + col_base, r2);
                sk2 = FMA256(r2, kv, sk2);

                /* Row 3 */
                __m256 r3 = _mm256_loadu_ps(st + (base+3) * d_state + col_base);
                r3 = _mm256_mul_ps(r3, ge_v);
                _mm256_storeu_ps(st + (base+3) * d_state + col_base, r3);
                sk3 = FMA256(r3, kv, sk3);
            }

            float sk0s = hreduce256_ps(sk0);
            float sk1s = hreduce256_ps(sk1);
            float sk2s = hreduce256_ps(sk2);
            float sk3s = hreduce256_ps(sk3);

            /* Phase 2: Compute delta = (v - sk) * beta */
            float d0 = (vh[base + 0] - sk0s) * bh;
            float d1 = (vh[base + 1] - sk1s) * bh;
            float d2 = (vh[base + 2] - sk2s) * bh;
            float d3 = (vh[base + 3] - sk3s) * bh;

            __m256 d0v = _mm256_set1_ps(d0);
            __m256 d1v = _mm256_set1_ps(d1);
            __m256 d2v = _mm256_set1_ps(d2);
            __m256 d3v = _mm256_set1_ps(d3);

            /* Phase 3: State update + output computation */
            __m256 out0 = _mm256_setzero_ps(), out1 = _mm256_setzero_ps();
            __m256 out2 = _mm256_setzero_ps(), out3 = _mm256_setzero_ps();

            for (int v = 0; v < d8; v++) {
                int col_base = v * 8;
                __m256 kv = _mm256_loadu_ps(khv + col_base);
                __m256 qv = _mm256_loadu_ps(qh + col_base);

                /* Row 0 */
                __m256 r0 = _mm256_loadu_ps(st + base * d_state + col_base);
                r0 = FMA256(kv, d0v, r0);
                _mm256_storeu_ps(st + base * d_state + col_base, r0);
                out0 = FMA256(r0, qv, out0);

                /* Row 1 */
                __m256 r1 = _mm256_loadu_ps(st + (base+1) * d_state + col_base);
                r1 = FMA256(kv, d1v, r1);
                _mm256_storeu_ps(st + (base+1) * d_state + col_base, r1);
                out1 = FMA256(r1, qv, out1);

                /* Row 2 */
                __m256 r2 = _mm256_loadu_ps(st + (base+2) * d_state + col_base);
                r2 = FMA256(kv, d2v, r2);
                _mm256_storeu_ps(st + (base+2) * d_state + col_base, r2);
                out2 = FMA256(r2, qv, out2);

                /* Row 3 */
                __m256 r3 = _mm256_loadu_ps(st + (base+3) * d_state + col_base);
                r3 = FMA256(kv, d3v, r3);
                _mm256_storeu_ps(st + (base+3) * d_state + col_base, r3);
                out3 = FMA256(r3, qv, out3);
            }

            /* Horizontal reduce output */
            ctx->ssm_output[(size_t)base * n_v_heads + h] = hreduce256_ps(out0);
            ctx->ssm_output[(size_t)(base+1) * n_v_heads + h] = hreduce256_ps(out1);
            ctx->ssm_output[(size_t)(base+2) * n_v_heads + h] = hreduce256_ps(out2);
            ctx->ssm_output[(size_t)(base+3) * n_v_heads + h] = hreduce256_ps(out3);
        }
    }
#elif defined(PICOLM_NEON)
    /* ---- NEON vectorized recurrence ----
     * Process 4 rows of the d_state x d_state state matrix simultaneously.
     * Each float32x4_t holds 4 floats. For d_state=128: 32 vector lanes per row,
     * 4 rows processed at once = 128 vector registers for the matrix.
     *
     * Layout: st[h] is [d_state][d_state] row-major.
     * We process 4 consecutive rows (r, r+1, r+2, r+3) together.
     *
     * NEON always has FP32 FMA via vmlaq_f32. */
    {
        int nr = d_state / 4; /* number of 4-row groups */
        float32x4_t ge_v = vdupq_n_f32(ge);
        int d4 = d_state / 4; /* number of 4-float vector lanes per row */

        for (int g = 0; g < nr; g++) {
            int base = g * 4; /* row base in this group */

            /* Phase 1: Decay + sk computation */
            float32x4_t sk0 = vdupq_n_f32(0);
            float32x4_t sk1 = vdupq_n_f32(0);
            float32x4_t sk2 = vdupq_n_f32(0);
            float32x4_t sk3 = vdupq_n_f32(0);

            for (int v = 0; v < d4; v++) {
                int col_base = v * 4;
                float32x4_t kv = vld1q_f32(khv + col_base);

                /* Row 0 */
                float32x4_t r0 = vld1q_f32(st + base * d_state + col_base);
                r0 = vmulq_f32(r0, ge_v);
                vst1q_f32(st + base * d_state + col_base, r0);
                sk0 = vmlaq_f32(sk0, r0, kv);

                /* Row 1 */
                float32x4_t r1 = vld1q_f32(st + (base+1) * d_state + col_base);
                r1 = vmulq_f32(r1, ge_v);
                vst1q_f32(st + (base+1) * d_state + col_base, r1);
                sk1 = vmlaq_f32(sk1, r1, kv);

                /* Row 2 */
                float32x4_t r2 = vld1q_f32(st + (base+2) * d_state + col_base);
                r2 = vmulq_f32(r2, ge_v);
                vst1q_f32(st + (base+2) * d_state + col_base, r2);
                sk2 = vmlaq_f32(sk2, r2, kv);

                /* Row 3 */
                float32x4_t r3 = vld1q_f32(st + (base+3) * d_state + col_base);
                r3 = vmulq_f32(r3, ge_v);
                vst1q_f32(st + (base+3) * d_state + col_base, r3);
                sk3 = vmlaq_f32(sk3, r3, kv);
            }

            float sk0s = vaddvq_f32_compat(sk0);
            float sk1s = vaddvq_f32_compat(sk1);
            float sk2s = vaddvq_f32_compat(sk2);
            float sk3s = vaddvq_f32_compat(sk3);

            /* Phase 2: Compute delta = (v - sk) * beta */
            float d0 = (vh[base + 0] - sk0s) * bh;
            float d1 = (vh[base + 1] - sk1s) * bh;
            float d2 = (vh[base + 2] - sk2s) * bh;
            float d3 = (vh[base + 3] - sk3s) * bh;

            float32x4_t d0v = vdupq_n_f32(d0);
            float32x4_t d1v = vdupq_n_f32(d1);
            float32x4_t d2v = vdupq_n_f32(d2);
            float32x4_t d3v = vdupq_n_f32(d3);

            /* Phase 3: State update + output computation */
            float32x4_t out0 = vdupq_n_f32(0);
            float32x4_t out1 = vdupq_n_f32(0);
            float32x4_t out2 = vdupq_n_f32(0);
            float32x4_t out3 = vdupq_n_f32(0);

            for (int v = 0; v < d4; v++) {
                int col_base = v * 4;
                float32x4_t kv = vld1q_f32(khv + col_base);
                float32x4_t qv = vld1q_f32(qh + col_base);

                /* Row 0 */
                float32x4_t r0 = vld1q_f32(st + base * d_state + col_base);
                r0 = vmlaq_f32(r0, kv, d0v);
                vst1q_f32(st + base * d_state + col_base, r0);
                out0 = vmlaq_f32(out0, r0, qv);

                /* Row 1 */
                float32x4_t r1 = vld1q_f32(st + (base+1) * d_state + col_base);
                r1 = vmlaq_f32(r1, kv, d1v);
                vst1q_f32(st + (base+1) * d_state + col_base, r1);
                out1 = vmlaq_f32(out1, r1, qv);

                /* Row 2 */
                float32x4_t r2 = vld1q_f32(st + (base+2) * d_state + col_base);
                r2 = vmlaq_f32(r2, kv, d2v);
                vst1q_f32(st + (base+2) * d_state + col_base, r2);
                out2 = vmlaq_f32(out2, r2, qv);

                /* Row 3 */
                float32x4_t r3 = vld1q_f32(st + (base+3) * d_state + col_base);
                r3 = vmlaq_f32(r3, kv, d3v);
                vst1q_f32(st + (base+3) * d_state + col_base, r3);
                out3 = vmlaq_f32(out3, r3, qv);
            }

            /* Horizontal reduce output */
            ctx->ssm_output[(size_t)base * n_v_heads + h] = vaddvq_f32_compat(out0);
            ctx->ssm_output[(size_t)(base+1) * n_v_heads + h] = vaddvq_f32_compat(out1);
            ctx->ssm_output[(size_t)(base+2) * n_v_heads + h] = vaddvq_f32_compat(out2);
            ctx->ssm_output[(size_t)(base+3) * n_v_heads + h] = vaddvq_f32_compat(out3);
        }
    }
#else
    /* ---- Scalar fallback ---- */
    float sk_local[256];
    float d_local[256];

    /* Decay: elementwise */
    for (int i = 0; i < d_state * d_state; i++) st[i] *= ge;

    /* sk[row] = sum_col state[row][col] * k[col] -- row-major contiguous */
    for (int row = 0; row < d_state; row++) {
        const float *st_row = st + (size_t)row * d_state;
        float sum = 0.0f;
        for (int col = 0; col < d_state; col++) sum += st_row[col] * khv[col];
        sk_local[row] = sum;
    }
    for (int row = 0; row < d_state; row++)
        d_local[row] = (vh[row] - sk_local[row]) * bh;

    /* state[row][col] += k[col] * d[row] -- row-major contiguous */
    for (int row = 0; row < d_state; row++) {
        float dv = d_local[row];
        float *st_row = st + (size_t)row * d_state;
        for (int col = 0; col < d_state; col++) st_row[col] += khv[col] * dv;
    }

    /* output[row] = sum_col state[row][col] * q[col] -- row-major contiguous */
    for (int row = 0; row < d_state; row++) {
        const float *st_row = st + (size_t)row * d_state;
        float sum = 0.0f;
        for (int col = 0; col < d_state; col++) sum += st_row[col] * qh[col];
        ctx->ssm_output[(size_t)row * n_v_heads + h] = sum;
    }
#endif
}

/* ================================================================
 * AVX-512 micro-kernels for chunked DeltaNet recurrence.
 *
 * These replace the scalar C loops in ssm_chunk_head_task with
 * AVX-512 vectorized GEMM kernels. Each kernel is single-threaded,
 * fixed-shape (d=128, cs<=64), and operates entirely in registers/L1.
 *
 * d=128 = 8 x __m512 (each holds 16 floats).
 * cs=64 max = 4 x __m512.
 *
 * The parallelism comes from the outer tensor_parallel_for across
 * (head, chunk) -- each kernel is called once per v-head per chunk.
 * ================================================================ */

#ifdef PICOLM_AVX512
#include <immintrin.h>

/* Kernel 1: V_eff -- compute S_init @ K^T -> sk[cs][d]
 *
 * sk[i][r] = sum_{c=0}^{d-1} state[r][c] * k[i][c]
 *
 * This is a GEMM: sk = state @ K^T   [d x d] @ [d x cs] -> [d x cs]
 * Transposed view: sk[cs][d] where sk[i] is the i-th row.
 */
static void ssm_kernel_veff(
        float *sk,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *k,   /* [cs][d] */
        int d, int cs)
{
    int d16 = d / 16;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *ski = sk + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int v = 0; v < d16; v++) {
                int col = v * 16;
                __m512 kv = _mm512_loadu_ps(ki + col);
                __m512 s0 = _mm512_loadu_ps(state + base * d + col);
                __m512 s1 = _mm512_loadu_ps(state + (base+1) * d + col);
                __m512 s2 = _mm512_loadu_ps(state + (base+2) * d + col);
                __m512 s3 = _mm512_loadu_ps(state + (base+3) * d + col);
                acc0 = _mm512_fmadd_ps(s0, kv, acc0);
                acc1 = _mm512_fmadd_ps(s1, kv, acc1);
                acc2 = _mm512_fmadd_ps(s2, kv, acc2);
                acc3 = _mm512_fmadd_ps(s3, kv, acc3);
            }

            ski[base] = _mm512_reduce_add_ps(acc0);
            ski[base+1] = _mm512_reduce_add_ps(acc1);
            ski[base+2] = _mm512_reduce_add_ps(acc2);
            ski[base+3] = _mm512_reduce_add_ps(acc3);
        }
    }
}

/* Kernel 2: Interaction matrix -- K @ K^T -> M[cs][cs]
 *
 * M[i][j] = (k[i] . k[j]) * decay[i][j]
 *
 * Only lower triangle (j <= i) is computed; upper triangle is zeroed.
 * The decay mask is applied elementwise.
 */
static void ssm_kernel_interaction(
        float *M,          /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *decay, /* [cs][cs] decay mask */
        int d, int cs)
{
    int d16 = d / 16;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *Mi = M + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            /* Dot product ki . kj using AVX-512 */
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            /* Unroll 4 vectors at a time = 64 floats per iteration */
            int v;
            for (v = 0; v + 3 < d16; v += 4) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(ki + v * 16),
                                        _mm512_loadu_ps(kj + v * 16), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(ki + (v+1) * 16),
                                        _mm512_loadu_ps(kj + (v+1) * 16), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(ki + (v+2) * 16),
                                        _mm512_loadu_ps(kj + (v+2) * 16), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(ki + (v+3) * 16),
                                        _mm512_loadu_ps(kj + (v+3) * 16), acc3);
            }

            /* Handle remaining lanes */
            for (; v < d16; v++) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(ki + v * 16),
                                        _mm512_loadu_ps(kj + v * 16), acc0);
            }

            /* Horizontal reduce all accumulators */
            float dot = _mm512_reduce_add_ps(acc0) + _mm512_reduce_add_ps(acc1)
                      + _mm512_reduce_add_ps(acc2) + _mm512_reduce_add_ps(acc3);

            Mi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            Mi[j] = 0.0f;
        }
    }
}

/* Kernel 3: Output cross-products -- K @ Q^T -> kq[cs][cs]
 *
 * kq[i][j] = (k[j] . q[i]) * decay[i][j]
 *
 * Note: k is indexed by j, q by i. This is effectively K^T @ Q but
 * with indices swapped compared to a standard GEMM.
 */
static void ssm_kernel_output_cross(
        float *kq,         /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *q,    /* [cs][d] */
        const float *decay, /* [cs][cs] */
        int d, int cs)
{
    int d16 = d / 16;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *kqi = kq + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            /* Dot product kj . qi using AVX-512 */
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            int v;
            for (v = 0; v + 3 < d16; v += 4) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(kj + v * 16),
                                        _mm512_loadu_ps(qi + v * 16), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(kj + (v+1) * 16),
                                        _mm512_loadu_ps(qi + (v+1) * 16), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(kj + (v+2) * 16),
                                        _mm512_loadu_ps(qi + (v+2) * 16), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(kj + (v+3) * 16),
                                        _mm512_loadu_ps(qi + (v+3) * 16), acc3);
            }

            for (; v < d16; v++) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(kj + v * 16),
                                        _mm512_loadu_ps(qi + v * 16), acc0);
            }

            float dot = _mm512_reduce_add_ps(acc0) + _mm512_reduce_add_ps(acc1)
                      + _mm512_reduce_add_ps(acc2) + _mm512_reduce_add_ps(acc3);

            kqi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            kqi[j] = 0.0f;
        }
    }
}

/* Kernel 4: State update -- accumulate decay[j] * outer(V_hat[j], k[j]) -> S_update[d][d]
 *
 * S_update[r][c] = sum_{j=0}^{cs-1} decay_to_end[j] * v_hat[j][r] * k[j][c]
 *
 * This is a weighted sum of rank-1 outer products.
 * Process d x d output tiled in AVX-512 friendly blocks.
 */
static void ssm_kernel_state_update(
        float *state,           /* [d][d] in-place: state *= total_decay, then += update */
        const float *v_hat,     /* [cs][d] */
        const float *k,         /* [cs][d] */
        const float *decay_to_end, /* [cs] decay from each position to end of chunk */
        float total_decay,
        int d, int cs)
{
    int d16 = d / 16;

    /* First: decay the existing state by total_decay */
    __m512 td = _mm512_set1_ps(total_decay);
    for (int r = 0; r < d; r++) {
        float *sr = state + r * d;
        for (int v = 0; v < d16; v++) {
            __m512 sv = _mm512_loadu_ps(sr + v * 16);
            _mm512_storeu_ps(sr + v * 16, _mm512_mul_ps(sv, td));
        }
    }

    /* Second: accumulate weighted outer products.
     * For each j: decay_to_end[j] * outer(V_hat[j], k[j])
     * Process 4 rows at a time for better register utilization. */
    int nr = d / 4; /* number of 4-row groups */

    for (int j = 0; j < cs; j++) {
        const float *vj = v_hat + j * d;
        const float *kj = k + j * d;
        float dj = decay_to_end[j];

        if (dj == 0.0f) continue;

                for (int g = 0; g < nr; g++) {
            int base = g * 4;
            float vj0 = vj[base];
            float vj1 = vj[base + 1];
            float vj2 = vj[base + 2];
            float vj3 = vj[base + 3];

            __m512 s0 = _mm512_set1_ps(vj0 * dj);
            __m512 s1 = _mm512_set1_ps(vj1 * dj);
            __m512 s2 = _mm512_set1_ps(vj2 * dj);
            __m512 s3 = _mm512_set1_ps(vj3 * dj);

            for (int v = 0; v < d16; v++) {
                int col_base = v * 16;
                __m512 kv = _mm512_loadu_ps(kj + col_base);

                __m512 r0 = _mm512_loadu_ps(state + base * d + col_base);
                __m512 r1 = _mm512_loadu_ps(state + (base+1) * d + col_base);
                __m512 r2 = _mm512_loadu_ps(state + (base+2) * d + col_base);
                __m512 r3 = _mm512_loadu_ps(state + (base+3) * d + col_base);

                _mm512_storeu_ps(state + base * d + col_base, _mm512_fmadd_ps(kv, s0, r0));
                _mm512_storeu_ps(state + (base+1) * d + col_base, _mm512_fmadd_ps(kv, s1, r1));
                _mm512_storeu_ps(state + (base+2) * d + col_base, _mm512_fmadd_ps(kv, s2, r2));
                _mm512_storeu_ps(state + (base+3) * d + col_base, _mm512_fmadd_ps(kv, s3, r3));
            }
        }
    }
}

/* Kernel 1b: S_init @ q[i] for each of cs positions -> sq[cs][d]
 *
 * sq[i][r] = sum_{c=0}^{d-1} state[r][c] * q[i][c]
 *
 * Same structure as Kernel 1 but with q instead of k.
 */
static void ssm_kernel_sq(
        float *sq,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *q,   /* [cs][d] */
        int d, int cs)
{
    int d16 = d / 16;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *sqi = sq + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int v = 0; v < d16; v++) {
                int col = v * 16;
                __m512 qv = _mm512_loadu_ps(qi + col);
                __m512 s0 = _mm512_loadu_ps(state + base * d + col);
                __m512 s1 = _mm512_loadu_ps(state + (base+1) * d + col);
                __m512 s2 = _mm512_loadu_ps(state + (base+2) * d + col);
                __m512 s3 = _mm512_loadu_ps(state + (base+3) * d + col);
                acc0 = _mm512_fmadd_ps(s0, qv, acc0);
                acc1 = _mm512_fmadd_ps(s1, qv, acc1);
                acc2 = _mm512_fmadd_ps(s2, qv, acc2);
                acc3 = _mm512_fmadd_ps(s3, qv, acc3);
            }

            sqi[base] = _mm512_reduce_add_ps(acc0);
            sqi[base+1] = _mm512_reduce_add_ps(acc1);
            sqi[base+2] = _mm512_reduce_add_ps(acc2);
            sqi[base+3] = _mm512_reduce_add_ps(acc3);
        }
    }
}
#elif defined(PICOLM_AVX2) || defined(PICOLM_AVX)
#ifdef PICOLM_FMA
    #define FMA256(a,b,c) _mm256_fmadd_ps((a),(b),(c))
#else
    #define FMA256(a,b,c) _mm256_add_ps(_mm256_mul_ps((a),(b)),(c))
#endif

/* Kernel 1: V_eff -- compute S_init @ K^T -> sk[cs][d] */
static void ssm_kernel_veff(
        float *sk,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *k,   /* [cs][d] */
        int d, int cs)
{
    int d8 = d / 8;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *ski = sk + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int v = 0; v < d8; v++) {
                int col = v * 8;
                __m256 kv = _mm256_loadu_ps(ki + col);
                __m256 s0 = _mm256_loadu_ps(state + base * d + col);
                __m256 s1 = _mm256_loadu_ps(state + (base+1) * d + col);
                __m256 s2 = _mm256_loadu_ps(state + (base+2) * d + col);
                __m256 s3 = _mm256_loadu_ps(state + (base+3) * d + col);
                acc0 = FMA256(s0, kv, acc0);
                acc1 = FMA256(s1, kv, acc1);
                acc2 = FMA256(s2, kv, acc2);
                acc3 = FMA256(s3, kv, acc3);
            }

            ski[base] = hreduce256_ps(acc0);
            ski[base+1] = hreduce256_ps(acc1);
            ski[base+2] = hreduce256_ps(acc2);
            ski[base+3] = hreduce256_ps(acc3);
        }
    }
}

/* Kernel 2: Interaction matrix -- K @ K^T -> M[cs][cs] */
static void ssm_kernel_interaction(
        float *M,          /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *decay, /* [cs][cs] decay mask */
        int d, int cs)
{
    int d8 = d / 8;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *Mi = M + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            /* Dot product ki . kj using AVX2 */
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int v;
            for (v = 0; v + 3 < d8; v += 4) {
                acc0 = FMA256(_mm256_loadu_ps(ki + v * 8),
                                        _mm256_loadu_ps(kj + v * 8), acc0);
                acc1 = FMA256(_mm256_loadu_ps(ki + (v+1) * 8),
                                        _mm256_loadu_ps(kj + (v+1) * 8), acc1);
                acc2 = FMA256(_mm256_loadu_ps(ki + (v+2) * 8),
                                        _mm256_loadu_ps(kj + (v+2) * 8), acc2);
                acc3 = FMA256(_mm256_loadu_ps(ki + (v+3) * 8),
                                        _mm256_loadu_ps(kj + (v+3) * 8), acc3);
            }

            for (; v < d8; v++) {
                acc0 = FMA256(_mm256_loadu_ps(ki + v * 8),
                                        _mm256_loadu_ps(kj + v * 8), acc0);
            }

            float dot = hreduce256_ps(acc0) + hreduce256_ps(acc1)
                      + hreduce256_ps(acc2) + hreduce256_ps(acc3);

            Mi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            Mi[j] = 0.0f;
        }
    }
}

/* Kernel 3: Output cross-products -- K @ Q^T -> kq[cs][cs] */
static void ssm_kernel_output_cross(
        float *kq,         /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *q,    /* [cs][d] */
        const float *decay, /* [cs][cs] */
        int d, int cs)
{
    int d8 = d / 8;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *kqi = kq + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            /* Dot product kj . qi using AVX2 */
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int v;
            for (v = 0; v + 3 < d8; v += 4) {
                acc0 = FMA256(_mm256_loadu_ps(kj + v * 8),
                                        _mm256_loadu_ps(qi + v * 8), acc0);
                acc1 = FMA256(_mm256_loadu_ps(kj + (v+1) * 8),
                                        _mm256_loadu_ps(qi + (v+1) * 8), acc1);
                acc2 = FMA256(_mm256_loadu_ps(kj + (v+2) * 8),
                                        _mm256_loadu_ps(qi + (v+2) * 8), acc2);
                acc3 = FMA256(_mm256_loadu_ps(kj + (v+3) * 8),
                                        _mm256_loadu_ps(qi + (v+3) * 8), acc3);
            }

            for (; v < d8; v++) {
                acc0 = FMA256(_mm256_loadu_ps(kj + v * 8),
                                        _mm256_loadu_ps(qi + v * 8), acc0);
            }

            float dot = hreduce256_ps(acc0) + hreduce256_ps(acc1)
                      + hreduce256_ps(acc2) + hreduce256_ps(acc3);

            kqi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            kqi[j] = 0.0f;
        }
    }
}

/* Kernel 4: State update -- accumulate weighted outer products */
static void ssm_kernel_state_update(
        float *state,           /* [d][d] in-place */
        const float *v_hat,     /* [cs][d] */
        const float *k,         /* [cs][d] */
        const float *decay_to_end, /* [cs] */
        float total_decay,
        int d, int cs)
{
    int d8 = d / 8;

    /* First: decay the existing state by total_decay */
    __m256 td = _mm256_set1_ps(total_decay);
    for (int r = 0; r < d; r++) {
        float *sr = state + r * d;
        for (int v = 0; v < d8; v++) {
            __m256 sv = _mm256_loadu_ps(sr + v * 8);
            _mm256_storeu_ps(sr + v * 8, _mm256_mul_ps(sv, td));
        }
    }

    /* Second: accumulate weighted outer products.
     * Process 4 rows at a time for better register utilization. */
    int nr = d / 4;

    for (int j = 0; j < cs; j++) {
        const float *vj = v_hat + j * d;
        const float *kj = k + j * d;
        float dj = decay_to_end[j];

        if (dj == 0.0f) continue;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            float vj0 = vj[base];
            float vj1 = vj[base + 1];
            float vj2 = vj[base + 2];
            float vj3 = vj[base + 3];

            __m256 s0 = _mm256_set1_ps(vj0 * dj);
            __m256 s1 = _mm256_set1_ps(vj1 * dj);
            __m256 s2 = _mm256_set1_ps(vj2 * dj);
            __m256 s3 = _mm256_set1_ps(vj3 * dj);

            for (int v = 0; v < d8; v++) {
                int col_base = v * 8;
                __m256 kv = _mm256_loadu_ps(kj + col_base);

                __m256 r0 = _mm256_loadu_ps(state + base * d + col_base);
                __m256 r1 = _mm256_loadu_ps(state + (base+1) * d + col_base);
                __m256 r2 = _mm256_loadu_ps(state + (base+2) * d + col_base);
                __m256 r3 = _mm256_loadu_ps(state + (base+3) * d + col_base);

                _mm256_storeu_ps(state + base * d + col_base, FMA256(kv, s0, r0));
                _mm256_storeu_ps(state + (base+1) * d + col_base, FMA256(kv, s1, r1));
                _mm256_storeu_ps(state + (base+2) * d + col_base, FMA256(kv, s2, r2));
                _mm256_storeu_ps(state + (base+3) * d + col_base, FMA256(kv, s3, r3));
            }
        }
    }
}

/* Kernel 1b: S_init @ Q^T -> sq[cs][d] */
static void ssm_kernel_sq(
        float *sq,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *q,   /* [cs][d] */
        int d, int cs)
{
    int d8 = d / 8;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *sqi = sq + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int v = 0; v < d8; v++) {
                int col = v * 8;
                __m256 qv = _mm256_loadu_ps(qi + col);
                __m256 s0 = _mm256_loadu_ps(state + base * d + col);
                __m256 s1 = _mm256_loadu_ps(state + (base+1) * d + col);
                __m256 s2 = _mm256_loadu_ps(state + (base+2) * d + col);
                __m256 s3 = _mm256_loadu_ps(state + (base+3) * d + col);
                acc0 = FMA256(s0, qv, acc0);
                acc1 = FMA256(s1, qv, acc1);
                acc2 = FMA256(s2, qv, acc2);
                acc3 = FMA256(s3, qv, acc3);
            }

            sqi[base] = hreduce256_ps(acc0);
            sqi[base+1] = hreduce256_ps(acc1);
            sqi[base+2] = hreduce256_ps(acc2);
            sqi[base+3] = hreduce256_ps(acc3);
        }
    }
}
#undef FMA256
#endif /* PICOLM_AVX512 || PICOLM_AVX2 || PICOLM_AVX */

/* ================================================================
 * NEON micro-kernels for chunked DeltaNet recurrence.
 *
 * d=128 = 32 x float32x4_t (each holds 4 floats).
 * cs=64 max = 16 x float32x4_t.
 *
 * NEON always has FP32 FMA via vmlaq_f32.
 * ================================================================ */
#ifdef PICOLM_NEON

/* Kernel 1: V_eff -- compute S_init @ K^T -> sk[cs][d] */
static void ssm_kernel_veff(
        float *sk,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *k,   /* [cs][d] */
        int d, int cs)
{
    int d4 = d / 4;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *ski = sk + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0);
            float32x4_t acc3 = vdupq_n_f32(0);

            for (int v = 0; v < d4; v++) {
                int col = v * 4;
                float32x4_t kv = vld1q_f32(ki + col);
                float32x4_t s0 = vld1q_f32(state + base * d + col);
                float32x4_t s1 = vld1q_f32(state + (base+1) * d + col);
                float32x4_t s2 = vld1q_f32(state + (base+2) * d + col);
                float32x4_t s3 = vld1q_f32(state + (base+3) * d + col);
                acc0 = vmlaq_f32(acc0, s0, kv);
                acc1 = vmlaq_f32(acc1, s1, kv);
                acc2 = vmlaq_f32(acc2, s2, kv);
                acc3 = vmlaq_f32(acc3, s3, kv);
            }

            ski[base] = vaddvq_f32_compat(acc0);
            ski[base+1] = vaddvq_f32_compat(acc1);
            ski[base+2] = vaddvq_f32_compat(acc2);
            ski[base+3] = vaddvq_f32_compat(acc3);
        }
    }
}

/* Kernel 2: Interaction matrix -- K @ K^T -> M[cs][cs] */
static void ssm_kernel_interaction(
        float *M,          /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *decay, /* [cs][cs] decay mask */
        int d, int cs)
{
    int d4 = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        float *Mi = M + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            /* Dot product ki . kj using NEON - 4 accumulators */
            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0);
            float32x4_t acc3 = vdupq_n_f32(0);

            int v;
            for (v = 0; v + 3 < d4; v += 4) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(ki + v * 4), vld1q_f32(kj + v * 4));
                acc1 = vmlaq_f32(acc1, vld1q_f32(ki + (v+1) * 4), vld1q_f32(kj + (v+1) * 4));
                acc2 = vmlaq_f32(acc2, vld1q_f32(ki + (v+2) * 4), vld1q_f32(kj + (v+2) * 4));
                acc3 = vmlaq_f32(acc3, vld1q_f32(ki + (v+3) * 4), vld1q_f32(kj + (v+3) * 4));
            }

            for (; v < d4; v++) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(ki + v * 4), vld1q_f32(kj + v * 4));
            }

            /* Reduce 4 accumulators to one float */
            float32x4_t sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float dot = vaddvq_f32_compat(sum);

            Mi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            Mi[j] = 0.0f;
        }
    }
}

/* Kernel 3: Output cross-products -- K @ Q^T -> kq[cs][cs] */
static void ssm_kernel_output_cross(
        float *kq,         /* [cs][cs] output */
        const float *k,    /* [cs][d] */
        const float *q,    /* [cs][d] */
        const float *decay, /* [cs][cs] */
        int d, int cs)
{
    int d4 = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *kqi = kq + i * cs;
        const float *decay_i = decay + i * cs;

        for (int j = 0; j <= i; j++) {
            const float *kj = k + j * d;

            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0);
            float32x4_t acc3 = vdupq_n_f32(0);

            int v;
            for (v = 0; v + 3 < d4; v += 4) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(kj + v * 4), vld1q_f32(qi + v * 4));
                acc1 = vmlaq_f32(acc1, vld1q_f32(kj + (v+1) * 4), vld1q_f32(qi + (v+1) * 4));
                acc2 = vmlaq_f32(acc2, vld1q_f32(kj + (v+2) * 4), vld1q_f32(qi + (v+2) * 4));
                acc3 = vmlaq_f32(acc3, vld1q_f32(kj + (v+3) * 4), vld1q_f32(qi + (v+3) * 4));
            }

            for (; v < d4; v++) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(kj + v * 4), vld1q_f32(qi + v * 4));
            }

            float32x4_t sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float dot = vaddvq_f32_compat(sum);

            kqi[j] = dot * decay_i[j];
        }

        /* Zero upper triangle */
        for (int j = i + 1; j < cs; j++) {
            kqi[j] = 0.0f;
        }
    }
}

/* Kernel 4: State update -- accumulate weighted outer products */
static void ssm_kernel_state_update(
        float *state,           /* [d][d] in-place */
        const float *v_hat,     /* [cs][d] */
        const float *k,         /* [cs][d] */
        const float *decay_to_end, /* [cs] */
        float total_decay,
        int d, int cs)
{
    int d4 = d / 4;

    /* First: decay the existing state by total_decay */
    float32x4_t td = vdupq_n_f32(total_decay);
    for (int r = 0; r < d; r++) {
        float *sr = state + r * d;
        for (int v = 0; v < d4; v++) {
            float32x4_t sv = vld1q_f32(sr + v * 4);
            vst1q_f32(sr + v * 4, vmulq_f32(sv, td));
        }
    }

    /* Second: accumulate weighted outer products.
     * Process 4 rows at a time for better register utilization. */
    int nr = d / 4;

    for (int j = 0; j < cs; j++) {
        const float *vj = v_hat + j * d;
        const float *kj = k + j * d;
        float dj = decay_to_end[j];

        if (dj == 0.0f) continue;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            float vj0 = vj[base];
            float vj1 = vj[base + 1];
            float vj2 = vj[base + 2];
            float vj3 = vj[base + 3];

            float32x4_t s0 = vdupq_n_f32(vj0 * dj);
            float32x4_t s1 = vdupq_n_f32(vj1 * dj);
            float32x4_t s2 = vdupq_n_f32(vj2 * dj);
            float32x4_t s3 = vdupq_n_f32(vj3 * dj);

            for (int v = 0; v < d4; v++) {
                int col_base = v * 4;
                float32x4_t kv = vld1q_f32(kj + col_base);

                float32x4_t r0 = vld1q_f32(state + base * d + col_base);
                float32x4_t r1 = vld1q_f32(state + (base+1) * d + col_base);
                float32x4_t r2 = vld1q_f32(state + (base+2) * d + col_base);
                float32x4_t r3 = vld1q_f32(state + (base+3) * d + col_base);

                vst1q_f32(state + base * d + col_base, vmlaq_f32(r0, kv, s0));
                vst1q_f32(state + (base+1) * d + col_base, vmlaq_f32(r1, kv, s1));
                vst1q_f32(state + (base+2) * d + col_base, vmlaq_f32(r2, kv, s2));
                vst1q_f32(state + (base+3) * d + col_base, vmlaq_f32(r3, kv, s3));
            }
        }
    }
}

/* Kernel 1b: S_init @ Q^T -> sq[cs][d] */
static void ssm_kernel_sq(
        float *sq,        /* [cs][d] output */
        const float *state, /* [d][d] state */
        const float *q,   /* [cs][d] */
        int d, int cs)
{
    int d4 = d / 4;
    int nr = d / 4;

    for (int i = 0; i < cs; i++) {
        const float *qi = q + i * d;
        float *sqi = sq + i * d;

        for (int g = 0; g < nr; g++) {
            int base = g * 4;
            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0);
            float32x4_t acc3 = vdupq_n_f32(0);

            for (int v = 0; v < d4; v++) {
                int col = v * 4;
                float32x4_t qv = vld1q_f32(qi + col);
                float32x4_t s0 = vld1q_f32(state + base * d + col);
                float32x4_t s1 = vld1q_f32(state + (base+1) * d + col);
                float32x4_t s2 = vld1q_f32(state + (base+2) * d + col);
                float32x4_t s3 = vld1q_f32(state + (base+3) * d + col);
                acc0 = vmlaq_f32(acc0, s0, qv);
                acc1 = vmlaq_f32(acc1, s1, qv);
                acc2 = vmlaq_f32(acc2, s2, qv);
                acc3 = vmlaq_f32(acc3, s3, qv);
            }

            sqi[base] = vaddvq_f32_compat(acc0);
            sqi[base+1] = vaddvq_f32_compat(acc1);
            sqi[base+2] = vaddvq_f32_compat(acc2);
            sqi[base+3] = vaddvq_f32_compat(acc3);
        }
    }
}
#endif /* PICOLM_NEON */

/* ================================================================
 * Chunked DeltaNet recurrence.
 *
 * Replaces the sequential per-token recurrence with a chunked
 * algorithm that processes CS tokens at a time using triangular
 * matrix operations. This converts O(n_tokens * d_state^2)
 * sequential work into O(n_tokens * d_state^2) parallel work.
 *
 * Per v-head, the recurrence is:
 *   S *= ge[t]          (scalar decay)
 *   sk = S @ k[t]       (d_state vector)
 *   d = (v[t] - sk) * beta[t]  (d_state vector)
 *   S += outer(k[t], d) (rank-1 update)
 *   out = S @ q[t]      (d_state vector)
 *
 * Within a chunk, we unroll this into:
 *   1. Compute cumulative decay D[t] = prod(ge[0..t])
 *   2. Build CS x CS decay mask: decay[i][j] = D[i]/D[j]
 *   3. Compute interaction matrix kb[i][j] = k[i].k[j]*beta[j] * decay
 *   4. Forward-substitute to get V_hat (solved V values)
 *   5. Compute output from initial state + intra-chunk attention
 *   6. Update state for next chunk
 *
 * Each v-head is independent; parallelized via tensor_parallel_for.
 * ================================================================ */

typedef struct {
    int idx;               /* v-head index */
    int d_state;
    int cs;                /* actual chunk size (last chunk may be smaller) */
    int repeat;            /* n_v_heads / n_k_heads */

    /* Input data: Q, K are [CS][d_state] per k-head, V is [CS][d_state] per v-head */
    const float *q;        /* [cs][d_state] for this k-head */
    const float *k;        /* [cs][d_state] for this k-head */
    const float *v;        /* [cs][d_state] for this v-head */
    const float *gate_log; /* [cs] log(gate_exp) for this v-head */
    const float *beta;     /* [cs] beta for this v-head */

    /* State: [d_state][d_state] for this v-head (in/out) */
    float *state;

    /* Output: [cs][d_state] for this v-head */
    float *out;

    /* Scratch buffer pointer (allocated externally) */
    float *scratch;
} ssm_chunk_head_task_t;

/* Process one v-head's chunked recurrence.
 * Corrected formulation following the DeltaNet chunking derivation:
 *
 * Per-token recurrence:
 *   S *= ge              (scalar decay)
 *   sk[r] = S[r] . k     (d_state vector, one per row)
 *   delta[r] = beta * (v[r] - sk[r])
 *   S += outer(k, delta) (rank-1 update)
 *   out[r] = S[r] . q    (d_state vector output)
 *
 * Chunked formulation (per v-head):
 *   cum_g[t] = sum_{j=0}^{t} log(ge[j])
 *   decay[i][j] = exp(cum_g[i] - cum_g[j]) for i >= j
 *
 *   V_eff[i] = beta[i] * (v[i] - decay[i] * S_init . k[i])
 *   M[i][j] = (k[i] . k[j]) * decay[i][j]   (no beta on k[j])
 *   V_hat[i] = V_eff[i] - beta[i] * sum_{j<i} M[i][j] * V_hat[j]
 *
 *   out[i] = decay[i] * S_init . q[i]
 *          + sum_{j<=i} (k[j] . q[i]) * decay[i][j] * V_hat[j]
 *
 *   S_new = decay[last] * S_init
 *         + sum_j decay[last][j] * outer(V_hat[j], k[j])
 */
static void ssm_chunk_head_task(int h, void *ctxp) {
    ssm_chunk_head_task_t *tasks = (ssm_chunk_head_task_t *)ctxp;
    ssm_chunk_head_task_t *ctx = &tasks[h];
    int d = ctx->d_state;
    int cs = ctx->cs;
    int kh = h / ctx->repeat;

    /* Point to this head's data within the chunk */
    const float *q = ctx->q + (size_t)kh * cs * d;
    const float *k = ctx->k + (size_t)kh * cs * d;
    const float *v = ctx->v + (size_t)h * cs * d;
    const float *gate_log = ctx->gate_log + h * cs;
    const float *beta = ctx->beta + h * cs;

    float *state = ctx->state + (size_t)h * d * d;
    /* ctx->out already offset by h*cs*d_state; no additional offset needed */
    float *out = ctx->out;

    /* Allocate scratch from the pre-allocated buffer.
     * We need:
     *   cum_g[cs], q_decay[cs]
     *   decay_mask[cs*cs], M[cs*cs]  (interaction matrix, no beta)
     *   v_eff[cs*d], v_hat[cs*d]
     *   sk[cs*d] (for AVX-512 kernel 1: S_init @ K^T)
     *   sq[cs*d] (for AVX-512 kernel 1b: S_init @ Q^T)
     *   kq[cs*cs] (for AVX-512 kernel 3: K @ Q^T)
     *   decay_to_end[cs] (for kernel 4: state update)
     * Total: 2*cs + 2*cs*cs + 3*cs*d + cs*d + cs*cs + cs floats
     * For CS=64, d=128: ~38KB per head */
    float *sp = ctx->scratch;

    float *cum_g = sp; sp += cs;
    float *q_decay = sp; sp += cs;
    float *decay_mask = sp; sp += cs * cs;
    float *M_mat = sp; sp += cs * cs;
    float *v_eff = sp; sp += cs * d;
    float *v_hat = sp; sp += cs * d;

#ifdef PICOLM_AVX512
    float *sk = sp; sp += cs * d;        /* S_init @ K^T */
    float *sq = sp; sp += cs * d;        /* S_init @ Q^T */
    float *kq = sp; sp += cs * cs;       /* K @ Q^T */
    float *decay_to_end = sp; sp += cs;  /* decay from each j to end */
#elif defined(PICOLM_AVX2) || defined(PICOLM_AVX) || defined(PICOLM_NEON)
    float *sk = sp; sp += cs * d;        /* S_init @ K^T */
    float *sq = sp; sp += cs * d;        /* S_init @ Q^T */
    float *kq = sp; sp += cs * cs;       /* K @ Q^T */
    float *decay_to_end = sp; sp += cs;  /* decay from each j to end */
#else
    (void)sp; /* silence unused warning */
#endif

    /* Step 1: Compute cumulative log-decay and decay from start */
    {
        float cum = 0.0f;
        for (int t = 0; t < cs; t++) {
            cum += gate_log[t];
            cum_g[t] = cum;
            float ex = cum;
            if (ex > 50.0f) ex = 50.0f;
            if (ex < -50.0f) ex = -50.0f;
            q_decay[t] = expf(ex);
        }
    }

    /* Pre-compute decay from each position to end of chunk (for state update) */
#if defined(PICOLM_AVX512) || defined(PICOLM_AVX2) || defined(PICOLM_AVX) || defined(PICOLM_NEON)
    {
        float cum_last = cum_g[cs - 1];
        for (int j = 0; j < cs; j++) {
            float diff = cum_last - cum_g[j];
            if (diff > 50.0f) diff = 50.0f;
            if (diff < -50.0f) diff = -50.0f;
            decay_to_end[j] = expf(diff);
        }
    }
#endif

    /* Build decay mask: decay_mask[i][j] = exp(cum_g[i] - cum_g[j]) for j <= i */
    for (int i = 0; i < cs; i++) {
        for (int j = 0; j <= i; j++) {
            float dm;
            if (i == j) {
                dm = 1.0f;
            } else {
                float diff = cum_g[i] - cum_g[j];
                if (diff > 50.0f) diff = 50.0f;
                if (diff < -50.0f) diff = -50.0f;
                dm = expf(diff);
            }
            decay_mask[i * cs + j] = dm;
        }
        for (int j = i + 1; j < cs; j++) {
            decay_mask[i * cs + j] = 0.0f;
        }
    }

#if defined(PICOLM_AVX512) || defined(PICOLM_AVX2) || defined(PICOLM_AVX) || defined(PICOLM_NEON)
    /* === AVX-512 / AVX2 / NEON micro-kernel path === */

    /* Kernel 2: Interaction matrix M = K @ K^T with decay mask */
    ssm_kernel_interaction(M_mat, k, decay_mask, d, cs);

    /* Kernel 1: sk = S_init @ K^T (for V_eff computation) */
    ssm_kernel_veff(sk, state, k, d, cs);

    /* Step 3 (scalar post-processing): V_eff[i] = beta[i] * (v[i] - decay[i] * sk[i]) */
    for (int i = 0; i < cs; i++) {
        float decay_i = q_decay[i];
        float bt = beta[i];
        const float *vi = v + i * d;
        const float *ski = sk + i * d;
        float *veffi = v_eff + i * d;
        for (int r = 0; r < d; r++) {
            veffi[r] = bt * (vi[r] - decay_i * ski[r]);
        }
    }

    /* Step 4: Forward substitution (sequential, scalar, cheap) */
    for (int i = 0; i < cs; i++) {
        float bt = beta[i];
        for (int r = 0; r < d; r++) {
            float sum_mv = 0.0f;
            for (int j = 0; j < i; j++) {
                sum_mv += M_mat[i * cs + j] * v_hat[j * d + r];
            }
            v_hat[i * d + r] = v_eff[i * d + r] - bt * sum_mv;
        }
    }

    /* Kernel 1b: sq = S_init @ Q^T (for initial state contribution to output) */
    ssm_kernel_sq(sq, state, q, d, cs);

    /* Kernel 3: kq = K @ Q^T with decay mask */
    ssm_kernel_output_cross(kq, k, q, decay_mask, d, cs);

    /* Step 5 (scalar assembly): out[i][r] = sq[i][r] * decay[i] + sum_{j<=i} kq[i][j] * v_hat[j][r] */
    for (int i = 0; i < cs; i++) {
        float decay_i = q_decay[i];
        const float *sqi = sq + i * d;
        float *outi = out + i * d;
        const float *kqi = kq + i * cs;

        /* Initial state contribution (scaled by decay) */
        for (int r = 0; r < d; r++) {
            outi[r] = sqi[r] * decay_i;
        }

        /* Intra-chunk contribution */
        for (int j = 0; j <= i; j++) {
            float attn = kqi[j];
            if (attn == 0.0f) continue;
            const float *vht = v_hat + j * d;
            for (int r = 0; r < d; r++) {
                outi[r] += attn * vht[r];
            }
        }
    }

    /* Kernel 4: State update (in-place: decay + accumulate weighted outer products) */
    {
        float cum_last = cum_g[cs - 1];
        float total_decay;
        {
            float ex = cum_last;
            if (ex > 50.0f) ex = 50.0f;
            if (ex < -50.0f) ex = -50.0f;
            total_decay = expf(ex);
        }
        ssm_kernel_state_update(state, v_hat, k, decay_to_end, total_decay, d, cs);
    }
#else
    /* === Scalar path (reference) === */

    /* Step 2: Compute interaction matrix M[i][j] = k_i . k_j * decay (scalar) */
    for (int i = 0; i < cs; i++) {
        const float *ki = k + i * d;
        for (int j = 0; j <= i; j++) {
            float dot = 0.0f;
            const float *kj = k + j * d;
            for (int di = 0; di < d; di++) dot += ki[di] * kj[di];
            M_mat[i * cs + j] = dot * decay_mask[i * cs + j];
        }
        for (int j = i + 1; j < cs; j++) {
            M_mat[i * cs + j] = 0.0f;
        }
    }

    /* Step 3: Compute V_eff[i] = beta[i] * (v[i] - decay[i] * S_init . k[i]) */
    for (int i = 0; i < cs; i++) {
        float decay_i = q_decay[i];
        float bt = beta[i];
        const float *ki = k + i * d;
        for (int r = 0; r < d; r++) {
            float s_dot_k = 0.0f;
            const float *st_row = state + r * d;
            for (int c = 0; c < d; c++) s_dot_k += st_row[c] * ki[c];
            v_eff[i * d + r] = bt * (v[i * d + r] - decay_i * s_dot_k);
        }
    }

    /* Step 4: Forward substitution. */
    for (int i = 0; i < cs; i++) {
        float bt = beta[i];
        for (int r = 0; r < d; r++) {
            float sum_mv = 0.0f;
            for (int j = 0; j < i; j++) {
                sum_mv += M_mat[i * cs + j] * v_hat[j * d + r];
            }
            v_hat[i * d + r] = v_eff[i * d + r] - bt * sum_mv;
        }
    }

    /* Step 5: Compute output. */
    for (int i = 0; i < cs; i++) {
        float decay_i = q_decay[i];
        const float *qi = q + i * d;

        /* Initial state contribution */
        for (int r = 0; r < d; r++) {
            float s_dot_q = 0.0f;
            const float *st_row = state + r * d;
            for (int c = 0; c < d; c++) s_dot_q += st_row[c] * qi[c];
            out[i * d + r] = s_dot_q * decay_i;
        }

        /* Intra-chunk contribution (includes j == i) */
        for (int j = 0; j <= i; j++) {
            float dm = decay_mask[i * cs + j];
            float k_dot_q = 0.0f;
            const float *kj = k + j * d;
            for (int di = 0; di < d; di++) k_dot_q += kj[di] * qi[di];
            float attn = k_dot_q * dm;
            const float *vht = v_hat + j * d;
            float *outi = out + i * d;
            for (int r = 0; r < d; r++) outi[r] += attn * vht[r];
        }
    }

    /* Step 6: Update state for next chunk. */
    {
        float total_decay;
        {
            float cum = cum_g[cs - 1];
            if (cum > 50.0f) cum = 50.0f;
            if (cum < -50.0f) cum = -50.0f;
            total_decay = expf(cum);
        }

        for (int r = 0; r < d; r++) {
            for (int c = 0; c < d; c++) {
                float update = 0.0f;
                for (int j = 0; j < cs; j++) {
                    float diff = cum_g[cs - 1] - cum_g[j];
                    float decay_to_end;
                    if (diff > 50.0f) diff = 50.0f;
                    if (diff < -50.0f) diff = -50.0f;
                    decay_to_end = expf(diff);
                    update += v_hat[j * d + r] * k[j * d + c] * decay_to_end;
                }
                state[r * d + c] = state[r * d + c] * total_decay + update;
            }
        }
    }
#endif
}

/* Chunked SSM recurrence: replaces the sequential per-token loop.
 * Processes all n_tokens in chunks of CS, parallelized across v-heads.
 *
 * conv_batch layout: [n_tokens][conv_dim] where conv_dim = 2*qk_dim + value_dim
 *   Within each token: [Q[n_k_heads][d_state] | K[n_k_heads][d_state] | V[n_v_heads][head_v_dim]]
 * alpha_batch: [n_tokens][n_v_heads] gate_log values (log-space: softplus(alpha+dt_w)*a_w)
 * beta_batch:  [n_tokens][n_v_heads] sigmoid(beta) values
 * state:       [n_v_heads][d_state][d_state] recurrent state (updated in-place)
 * xb2_batch:   [n_tokens][value_dim] head-major output [h*head_v_dim + d]
 */
static void ssm_chunked_recurrence(
        const float *conv_batch,
        const float *alpha_batch,
        const float *beta_batch,
        float *state,
        float *xb2_batch,
        int n_tokens, int value_dim,
        int d_state, int n_k_heads, int n_v_heads, int head_v_dim, int repeat,
        int conv_dim, int cs)
{
    /* cs: chunk size (from model->ssm_chunk_size, default 64, 0=auto) */
    if (cs <= 0) cs = 64; /* default */
    if (cs > n_tokens) cs = n_tokens;
    int n_chunks = (n_tokens + cs - 1) / cs;
    if (n_chunks < 1) n_chunks = 1;

    int qk_dim = d_state * n_k_heads;

    /* Compute scratch size per head for ssm_chunk_head_task.
     * Base: cum_g[cs] + q_decay[cs] + decay_mask[cs*cs] + M_mat[cs*cs]
     *       + v_eff[cs*d] + v_hat[cs*d]
     * AVX-512 extras: sk[cs*d] + sq[cs*d] + kq[cs*cs] + decay_to_end[cs]
     * Total: 3*cs + 3*cs*cs + 4*cs*d (for AVX-512)
     * For cs=64, d=128: ~38KB per head */
    size_t scratch_per_head = 3UL * cs + 3UL * cs * cs + 4UL * cs * (size_t)d_state;
    scratch_per_head = (scratch_per_head + 15) & ~15UL;

    /* Allocate task contexts and per-head scratch separately to avoid stride issues */
    ssm_chunk_head_task_t *tasks = (ssm_chunk_head_task_t *)calloc(n_v_heads, sizeof(ssm_chunk_head_task_t));
    float *scratch_pool = (float *)calloc(n_v_heads * scratch_per_head, sizeof(float));
    if (!tasks || !scratch_pool) { fprintf(stderr, "OOM: chunk tasks\n"); exit(1); }
    for (int h = 0; h < n_v_heads; h++) {
        tasks[h].scratch = scratch_pool + (size_t)h * scratch_per_head;
    }

    /* Allocate gather buffers for contiguous chunk data.
     * Layout: [n_heads][cs][d] for Q/K/V, [n_v_heads][cs] for scalars.
     * This matches the access pattern in ssm_chunk_head_task:
     *   q[kh*cs*d + t*d + di], k[kh*cs*d + t*d + di], v[h*cs*d + t*d + di]
     *   gate_log[h*cs + t], beta[h*cs + t] */
    size_t cs_alloc = cs; /* runtime chunk size */
    float *chunk_q = (float *)malloc(cs_alloc * (size_t)n_k_heads * (size_t)d_state * sizeof(float));
    float *chunk_k = (float *)malloc(cs_alloc * (size_t)n_k_heads * (size_t)d_state * sizeof(float));
    float *chunk_v = (float *)malloc(cs_alloc * (size_t)n_v_heads * (size_t)head_v_dim * sizeof(float));
    float *chunk_beta = (float *)malloc(cs_alloc * (size_t)n_v_heads * sizeof(float));
    float *gate_log = (float *)malloc(cs_alloc * (size_t)n_v_heads * sizeof(float));
    float *chunk_out = (float *)malloc(cs_alloc * (size_t)n_v_heads * (size_t)d_state * sizeof(float));
    if (!chunk_q || !chunk_k || !chunk_v || !chunk_beta || !gate_log || !chunk_out) {
        fprintf(stderr, "OOM: chunk gather buffers\n"); exit(1);
    }

    /* Process chunk by chunk */
    for (int ci = 0; ci < n_chunks; ci++) {
        int cs_actual = (ci == n_chunks - 1) ? (n_tokens - ci * cs) : cs;
        if (cs_actual <= 0) break;
        int chunk_start = ci * cs;

        /* Gather Q, K, V, beta, gate_log for this chunk.
         * Layout: [n_heads][cs_actual][d] for Q/K/V, [n_v_heads][cs_actual] for scalars. */
        for (int h = 0; h < n_k_heads; h++) {
            for (int t = 0; t < cs_actual; t++) {
                const float *tok = conv_batch + (chunk_start + t) * conv_dim;
                memcpy(chunk_q + (size_t)h * cs_actual * d_state + t * d_state,
                       tok + h * d_state, d_state * sizeof(float));
                memcpy(chunk_k + (size_t)h * cs_actual * d_state + t * d_state,
                       tok + qk_dim + h * d_state, d_state * sizeof(float));
            }
        }
        for (int h = 0; h < n_v_heads; h++) {
            for (int t = 0; t < cs_actual; t++) {
                const float *tok = conv_batch + (chunk_start + t) * conv_dim;
                memcpy(chunk_v + (size_t)h * cs_actual * head_v_dim + t * head_v_dim,
                       tok + 2 * qk_dim + h * head_v_dim, head_v_dim * sizeof(float));
                chunk_beta[h * cs_actual + t] = beta_batch[(chunk_start + t) * n_v_heads + h];
                gate_log[h * cs_actual + t] = alpha_batch[(chunk_start + t) * n_v_heads + h];
            }
        }

        /* Initialize each task */
        for (int h = 0; h < n_v_heads; h++) {
            tasks[h].idx = h;
            tasks[h].d_state = d_state;
            tasks[h].cs = cs_actual;
            tasks[h].repeat = repeat;
            /* Now Q/K/V are contiguous [cs_actual][n_heads][d], so we can slice by head */
            tasks[h].q = chunk_q;
            tasks[h].k = chunk_k;
            tasks[h].v = chunk_v;
            tasks[h].gate_log = gate_log;
            tasks[h].beta = chunk_beta;
            tasks[h].state = state;
            /* Output: [cs_actual][d_state] per v-head */
            tasks[h].out = chunk_out + (size_t)h * cs_actual * d_state;
        }

        /* Process all v-heads in parallel */
        tensor_parallel_for(n_v_heads, ssm_chunk_head_task, tasks);

        /* Reshape output from [n_v_heads][cs_actual][d_state] to [cs_actual][value_dim] head-major.
         * chunk_out[h * cs_actual * d_state + t * d_state + r] -> xb2_batch[(chunk_start+t)*value_dim + h*head_v_dim + r] */
        for (int t = 0; t < cs_actual; t++) {
            float *out_tok = xb2_batch + (chunk_start + t) * value_dim;
            for (int h = 0; h < n_v_heads; h++) {
                const float *oh = chunk_out + (size_t)h * cs_actual * d_state + t * d_state;
                float *od = out_tok + h * head_v_dim;
                memcpy(od, oh, d_state * sizeof(float));
            }
        }

            }

    free(tasks); free(scratch_pool);
    free(chunk_q); free(chunk_k); free(chunk_v);
    free(chunk_beta); free(gate_log); free(chunk_out);
}

#ifdef PICOLM_GPU
/* Persistent device scratch buffers for ssm_forward()'s device-native
 * vecdot calls (picolm_gpu_ssm_vecdot_dev). The _dev variant needs
 * device-resident x/out pointers -- weights and head_map are already
 * device-resident from model load, but the activation (s->xb) lives
 * on the host and the small per-head output needs a device buffer
 * before D2H back. Sized once from dim/n_v_heads and reused. */
static void *g_ssm_vecdot_x_dev = NULL;
static size_t g_ssm_vecdot_x_bytes = 0;
static void *g_ssm_vecdot_out_dev = NULL;
static size_t g_ssm_vecdot_out_bytes = 0;
static int g_ssm_vecdot_device = -1;

static int ssm_vecdot_dev_scratch_ensure(int device, size_t x_bytes, size_t out_bytes) {
    if (g_ssm_vecdot_device != device) {
        g_ssm_vecdot_x_dev = NULL; g_ssm_vecdot_x_bytes = 0;
        g_ssm_vecdot_out_dev = NULL; g_ssm_vecdot_out_bytes = 0;
        g_ssm_vecdot_device = device;
    }
    if (!g_ssm_vecdot_x_dev || g_ssm_vecdot_x_bytes < x_bytes) {
        g_ssm_vecdot_x_dev = picolm_gpu_alloc_device(x_bytes, device);
        if (!g_ssm_vecdot_x_dev) return 0;
        g_ssm_vecdot_x_bytes = x_bytes;
    }
    if (!g_ssm_vecdot_out_dev || g_ssm_vecdot_out_bytes < out_bytes) {
        g_ssm_vecdot_out_dev = picolm_gpu_alloc_device(out_bytes, device);
        if (!g_ssm_vecdot_out_dev) return 0;
        g_ssm_vecdot_out_bytes = out_bytes;
    }
    return 1;
}

/* Device scratch buffers for batched matmul via picolm_gpu_matmul_dev.
 * Unlike the host-wrapper picolm_gpu_matmul() (called via matmul()/
 * tensor_set_gpu_tensor), which does TWO gpuDeviceSynchronize() calls
 * per Q8_0 matmul (one after quantize-x, one after the matmul kernel),
 * matmul_dev has zero internal syncs. Kernels on the same stream
 * execute in submission order, so two back-to-back matmul_dev calls
 * need only one sync total -- when host code actually reads results.
 * This is ~4x fewer syncs for the attn_qkv+attn_gate_ssm pair (1 vs 4). */
static void *g_ssm_mm_xb_dev = NULL;      static size_t g_ssm_mm_xb_bytes = 0;
static void *g_ssm_mm_qkv_dev = NULL;     static size_t g_ssm_mm_qkv_bytes = 0;
static void *g_ssm_mm_gate_dev = NULL;    static size_t g_ssm_mm_gate_bytes = 0;
static void *g_ssm_mm_outin_dev = NULL;   static size_t g_ssm_mm_outin_bytes = 0;
static void *g_ssm_mm_outres_dev = NULL;  static size_t g_ssm_mm_outres_bytes = 0;
static int g_ssm_mm_device = -1;

static int ssm_mm_alloc(void **buf, size_t *cap, size_t need, int device) {
    if (*buf && *cap >= need) return 1;
    *buf = picolm_gpu_alloc_device(need, device);
    if (!*buf) { *cap = 0; return 0; }
    *cap = need;
    return 1;
}

static int ssm_qkv_gate_dev_scratch_ensure(int device, size_t xb_bytes,
                                            size_t qkv_bytes, size_t gate_bytes) {
    if (g_ssm_mm_device != device) {
        g_ssm_mm_xb_dev = NULL; g_ssm_mm_xb_bytes = 0;
        g_ssm_mm_qkv_dev = NULL; g_ssm_mm_qkv_bytes = 0;
        g_ssm_mm_gate_dev = NULL; g_ssm_mm_gate_bytes = 0;
        g_ssm_mm_outin_dev = NULL; g_ssm_mm_outin_bytes = 0;
        g_ssm_mm_outres_dev = NULL; g_ssm_mm_outres_bytes = 0;
        g_ssm_mm_device = device;
    }
    return ssm_mm_alloc(&g_ssm_mm_xb_dev, &g_ssm_mm_xb_bytes, xb_bytes, device) &&
           ssm_mm_alloc(&g_ssm_mm_qkv_dev, &g_ssm_mm_qkv_bytes, qkv_bytes, device) &&
           ssm_mm_alloc(&g_ssm_mm_gate_dev, &g_ssm_mm_gate_bytes, gate_bytes, device);
}

static int ssm_out_dev_scratch_ensure(int device, size_t in_bytes, size_t out_bytes) {
    if (g_ssm_mm_device != device) {
        g_ssm_mm_xb_dev = NULL; g_ssm_mm_xb_bytes = 0;
        g_ssm_mm_qkv_dev = NULL; g_ssm_mm_qkv_bytes = 0;
        g_ssm_mm_gate_dev = NULL; g_ssm_mm_gate_bytes = 0;
        g_ssm_mm_outin_dev = NULL; g_ssm_mm_outin_bytes = 0;
        g_ssm_mm_outres_dev = NULL; g_ssm_mm_outres_bytes = 0;
        g_ssm_mm_device = device;
    }
    return ssm_mm_alloc(&g_ssm_mm_outin_dev, &g_ssm_mm_outin_bytes, in_bytes, device) &&
           ssm_mm_alloc(&g_ssm_mm_outres_dev, &g_ssm_mm_outres_bytes, out_bytes, device);
}
#endif

/* SSM layer forward pass (autoregressive, single token) */
static void ssm_forward(model_t *m, run_state_t *s, float *x, float *residual,
                        layer_weights_t *lw, int il, int pos, void *gpu_lw) {
#ifndef PICOLM_GPU
    (void)gpu_lw;
#endif
    model_config_t *c = &m->config;
    int dim = c->n_embd;
    int d_conv = c->ssm_d_conv;
    int d_state = c->ssm_d_state;
    int n_k_heads = c->ssm_n_group;
    int n_v_heads = c->ssm_dt_rank;
    int conv_dim = 2 * d_state * n_k_heads + c->ssm_d_inner;
    int head_v_dim = c->ssm_d_inner / n_v_heads;
    float eps = c->rms_norm_eps;
    
    /* Qwen3.5 GGUF v-head reorder parameters (used throughout this function) */
    int n_k = c->ssm_n_group;
    int n_vpk = n_v_heads / n_k;
    int half_vpk = n_vpk / 2;
    int do_remap = !m->from_safetensors && n_k > 0 && n_k < n_v_heads && half_vpk > 0;

    /* Scratch space: dedicated SSM buffer */
    float *tmp = s->ssm_tmp;

    /* 1. RMSNorm (attn_norm) */
    rmsnorm(s->xb, x, s->attn_norm_w[il], dim, eps);
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("xb[:8]", s->xb, 8, 8);
#endif
    /* 2. QKV projection: qkv_mixed = matmul(attn_qkv, xb) -> [conv_dim] */
    /* 3. Z gate: z = matmul(attn_gate_ssm, xb) -> [value_dim] */
    /* Batched: both read the same xb, so one H2D, two matmul_dev,
     * one sync, two D2H. Replaces two matmul() calls each doing
     * H2D + 2x sync + D2H (4 syncs total for Q8_0 -> 1 sync). */
    int ssm_qkv_gate_gpu_done = 0;
#ifdef PICOLM_GPU
    if (gpu_lw) {
        gpu_layer_weights_t *gl = (gpu_layer_weights_t *)gpu_lw;
        size_t xb_bytes = (size_t)dim * sizeof(float);
        size_t qkv_bytes = (size_t)conv_dim * sizeof(float);
        size_t gate_bytes = (size_t)c->ssm_d_inner * sizeof(float);
        if (gl->attn_qkv && gl->attn_gate_ssm &&
            ssm_qkv_gate_dev_scratch_ensure(m->gpu.device, xb_bytes, qkv_bytes, gate_bytes) &&
            picolm_gpu_memcpy(g_ssm_mm_xb_dev, s->xb, xb_bytes, 1, m->gpu.device) &&
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_qkv,
                                   g_ssm_mm_qkv_dev, g_ssm_mm_xb_dev, 1, m->gpu.device, 0, 0) &&
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_gate_ssm,
                                   g_ssm_mm_gate_dev, g_ssm_mm_xb_dev, 1, m->gpu.device, 0, 0) &&
            picolm_gpu_sync(m->gpu.device) &&
            picolm_gpu_memcpy(s->q, g_ssm_mm_qkv_dev, qkv_bytes, -1, m->gpu.device) &&
            picolm_gpu_memcpy(s->xb2, g_ssm_mm_gate_dev, gate_bytes, -1, m->gpu.device)) {
            ssm_qkv_gate_gpu_done = 1;
        }
    }
#endif
    if (!ssm_qkv_gate_gpu_done)
    {
        matmul(s->q, s->xb, lw->attn_qkv, dim, conv_dim, lw->type_attn_qkv);
        matmul(s->xb2, s->xb, lw->attn_gate_ssm, dim, c->ssm_d_inner, lw->type_attn_gate_ssm);
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("qkv[:8]", s->q, 8, 8);
#endif
    /* If GGUF reorders v-head rows, convert xb2 from GGUF order to sequential order */
    if (do_remap) {
        float *xb2_tmp = (float *)malloc(c->ssm_d_inner * sizeof(float));
        memcpy(xb2_tmp, s->xb2, c->ssm_d_inner * sizeof(float));
        for (int h = 0; h < n_v_heads; h++) {
            int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
            memcpy(s->xb2 + h * head_v_dim, xb2_tmp + gh * head_v_dim, head_v_dim * sizeof(float));
        }
        free(xb2_tmp);
        if (il == 0 || il == 8 || il == 16 || il == 32 || il == 48 || il == 60) {
            }
    }

    /* 4. Convolution: compute BEFORE shifting conv_state */
    float *conv_state = s->ssm_conv_state[il];
    int state_stride = conv_dim;
    int n_state_rows = d_conv - 1;
    float *conv_output = tmp; /* [conv_dim] */
    float *conv1d_w = s->ssm_conv1d_w[il];
    for (int co = 0; co < conv_dim; co++) {
        float sum = 0.0f;
        for (int d = 0; d < n_state_rows; d++) {
            sum += conv1d_w[d + co * d_conv] * conv_state[d * state_stride + co];
        }
        sum += conv1d_w[(d_conv - 1) + co * d_conv] * s->q[co];
        float v = sum;
        conv_output[co] = v * (1.0f / (1.0f + expf(-v))); /* silu */
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("conv_out[:8]", conv_output, 8, 8);
#endif

    /* Shift conv_state left and append new token */
    for (int r = 0; r < n_state_rows - 1; r++) {
        memcpy(conv_state + r * state_stride, conv_state + (r + 1) * state_stride, state_stride * sizeof(float));
    }
    memcpy(conv_state + (n_state_rows - 1) * state_stride, s->q, state_stride * sizeof(float));

    /* 6. Split into Q, K, V from conv_output (contiguous layout)
     * conv_output: [conv_dim] = [q_part + k_part + v_part]
     * Q: [head_k_dim, n_k_heads] stored head-major: [h*d_state + d]
     * K: [head_k_dim, n_k_heads] stored head-major
     * V: [head_v_dim, n_v_heads] stored head-major
     */
    int qk_dim = d_state * n_k_heads;
    float *q_conv = tmp + conv_dim; /* [qk_dim] */
    float *k_conv = tmp + conv_dim + qk_dim; /* [qk_dim] */
    float *v_conv = tmp + conv_dim + 2 * qk_dim; /* [c->ssm_d_inner] */

    memcpy(q_conv, conv_output, qk_dim * sizeof(float));
    memcpy(k_conv, conv_output + qk_dim, qk_dim * sizeof(float));
    memcpy(v_conv, conv_output + 2 * qk_dim, c->ssm_d_inner * sizeof(float));

    /* If GGUF reorders V channels, convert v_conv from GGUF order to sequential order */
    if (do_remap) {
        float *v_conv_tmp = (float *)malloc(c->ssm_d_inner * sizeof(float));
        memcpy(v_conv_tmp, v_conv, c->ssm_d_inner * sizeof(float));
        for (int h = 0; h < n_v_heads; h++) {
            int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
            memcpy(v_conv + h * head_v_dim, v_conv_tmp + gh * head_v_dim, head_v_dim * sizeof(float));
        }
        free(v_conv_tmp);
    }

    /* 7. L2 normalize Q and K per k_head */
    for (int h = 0; h < n_k_heads; h++) {
        float *qh = q_conv + h * d_state;
        float nrm = 0.0f;
        for (int d = 0; d < d_state; d++) nrm += qh[d] * qh[d];
        nrm = 1.0f / sqrtf(nrm + 1e-12f);
        for (int d = 0; d < d_state; d++) qh[d] *= nrm;
    }
    for (int h = 0; h < n_k_heads; h++) {
        float *kh = k_conv + h * d_state;
        float nrm = 0.0f;
        for (int d = 0; d < d_state; d++) nrm += kh[d] * kh[d];
        nrm = 1.0f / sqrtf(nrm + 1e-12f);
        for (int d = 0; d < d_state; d++) kh[d] *= nrm;
    }

    /* 8. Scale Q by 1/sqrt(d_state) */
    float q_scale = 1.0f / sqrtf((float)d_state);
    for (int i = 0; i < qk_dim; i++) q_conv[i] *= q_scale;
#ifdef DEBUG_SSM
    if (il == 0 || il == 8 || il == 16 || il == 32 || il == 48 || il == 60) {
        dbg_vec("q_conv_scaled[:8]", q_conv, 8, 8);
        dbg_vec("k_conv_scaled[:8]", k_conv, 8, 8);
    }
#endif

    /* Alpha/beta per-head projections share the same activation vector
     * (s->xb) across every one of the (up to dozens of) v-heads -- quantize
     * it to Q8_0 once here and reuse the fast int8 x int8 kernels
     * (vec_dot_q8_0_q8_0_deltas / vec_dot_q4_0_q8_0) for whichever of
     * alpha/beta uses a Q8_0 or Q4_0 weight, instead of the mixed
     * int8-weight x float32-activation kernel vec_dot()'s generic
     * dispatch falls back to for those types -- same fix already applied
     * to attn_core's K-dot product and to the FFN/projection matmuls. */
    uint8_t xb_q8_stack[8192 / 32 * 34];
    void *xb_q8 = (size_t)(dim / 32) * 34 <= sizeof(xb_q8_stack) ? (void *)xb_q8_stack : malloc((size_t)(dim / 32) * 34);
    float xb_q8_d_stack[8192 / 32];
    float *xb_q8_d = (dim / 32) <= (int)(sizeof(xb_q8_d_stack) / sizeof(float)) ? xb_q8_d_stack : (float *)malloc(sizeof(float) * (dim / 32));
    {
        int nb_xb = dim / 32;
        quantize_row_q8_0(s->xb, xb_q8, dim);
        const block_q8_0 *xqb = (const block_q8_0 *)xb_q8;
        for (int k = 0; k < nb_xb; k++) xb_q8_d[k] = fp16_to_fp32_lookup(xqb[k].d);
    }

    /* 9. Alpha: alpha = matmul(ssm_alpha, xb) + ssm_dt.bias -> [dt_rank] */
    /* GGUF stores [dim, n_v_heads] column-major: each head has dim contiguous elements */
    /* GGUF v-heads may be in tiled/interleaved order. Map sequential h -> GGUF head index. */
    /* Mapping: sequential [k0v0, k0v1, k0v2, ..., k0v7, k1v0, ...] */
    /*           GGUF     [k0v0, k0v2, k0v4, k0v6, k1v0, ..., k0v1, k0v3, ...] */
    /* Helper: map sequential head h -> GGUF head index gh */
    /* qwen35_vhead_gguf defined at file scope */
#ifdef PICOLM_GPU
    /* Upload xb once, shared by both alpha and beta device-native vecdot.
     * Weights (ssm_alpha_dev/ssm_beta_dev) and head_map are already
     * device-resident from model load. Only the activation needs per-call H2D
     * (~20KB) instead of picolm_gpu_ssm_vecdot()'s ~255KB weight re-upload. */
    int ssm_vecdot_gpu_ready = 0;
    if (gpu_lw && m->gpu.active) {
        size_t vx_bytes = (size_t)dim * sizeof(float);
        size_t vout_bytes = (size_t)n_v_heads * sizeof(float);
        if (ssm_vecdot_dev_scratch_ensure(m->gpu.device, vx_bytes, vout_bytes) &&
            picolm_gpu_memcpy(g_ssm_vecdot_x_dev, s->xb, vx_bytes, 1, m->gpu.device)) {
            ssm_vecdot_gpu_ready = 1;
        }
    }
    const int *ssm_vecdot_hmap_dev = do_remap ? (const int *)m->gpu.ssm_head_map_dev : NULL;
#endif
    float *alpha_out = tmp + conv_dim + 2 * qk_dim + c->ssm_d_inner; /* [dt_rank] */
    {
        gguf_type_t alpha_type = lw->type_ssm_alpha;
        size_t row_bytes = gguf_type_row_size(alpha_type, dim);
        int alpha_map[256];
        for (int h = 0; h < n_v_heads; h++) alpha_map[h] = do_remap ? qwen35_vhead_gguf(h, n_vpk, n_k) : h;
#ifdef PICOLM_GPU
        if (gpu_lw && ssm_vecdot_gpu_ready && m->gpu.ssm_alpha_dev[il] &&
            (!do_remap || ssm_vecdot_hmap_dev) &&
            (alpha_type == GGUF_TYPE_F32 || alpha_type == GGUF_TYPE_Q4_0 || alpha_type == GGUF_TYPE_Q8_0) &&
            picolm_gpu_ssm_vecdot_dev(g_ssm_vecdot_out_dev, g_ssm_vecdot_x_dev,
                                       m->gpu.ssm_alpha_dev[il], alpha_type, dim,
                                       n_v_heads, (int)row_bytes, ssm_vecdot_hmap_dev,
                                       m->gpu.device) &&
            picolm_gpu_sync(m->gpu.device) &&
            picolm_gpu_memcpy(alpha_out, g_ssm_vecdot_out_dev,
                               (size_t)n_v_heads * sizeof(float), -1, m->gpu.device)) {
            for (int h = 0; h < n_v_heads; h++) alpha_out[h] += s->ssm_dt_w[il][h];
        } else
#endif
        {
            for (int h = 0; h < n_v_heads; h++) {
                int gh = alpha_map[h];
                const uint8_t *head_data = (const uint8_t *)lw->ssm_alpha + (size_t)gh * row_bytes;
                float sum;
                if (alpha_type == GGUF_TYPE_Q8_0) sum = vec_dot_q8_0_q8_0_deltas(xb_q8, xb_q8_d, head_data, dim);
                else if (alpha_type == GGUF_TYPE_Q4_0) sum = vec_dot_q4_0_q8_0(head_data, xb_q8, dim);
                else sum = vec_dot(head_data, s->xb, dim, alpha_type);
                alpha_out[h] = sum + s->ssm_dt_w[il][h];
            }
        }
        /* alpha_map is stack-allocated */
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("alpha[:8]", alpha_out, n_v_heads, 8);
#endif

    /* gate = ssm_a * softplus(alpha) -> [dt_rank] */
    float *gate = alpha_out + n_v_heads; /* [dt_rank] */
    for (int h = 0; h < n_v_heads; h++) {
        float a = alpha_out[h];
        float sp = (a > 20.0f) ? a : (a < -20.0f) ? expf(a) : logf(1.0f + expf(a));
        gate[h] = sp * s->ssm_a_w[il][h];
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("gate[:8]", gate, n_v_heads, 8);
#endif

    /* 10. Beta: sigmoid(matmul(ssm_beta, xb)) -> [dt_rank] */
    /* GGUF stores [dim, n_v_heads] column-major, v-heads may be tiled/interleaved */
    float *beta = gate + n_v_heads; /* [dt_rank] */
    {
        gguf_type_t beta_type = lw->type_ssm_beta;
        size_t row_bytes = gguf_type_row_size(beta_type, dim);
        int beta_map[256];
        for (int h = 0; h < n_v_heads; h++) beta_map[h] = do_remap ? qwen35_vhead_gguf(h, n_vpk, n_k) : h;
#ifdef PICOLM_GPU
        if (gpu_lw && ssm_vecdot_gpu_ready && m->gpu.ssm_beta_dev[il] &&
            (!do_remap || ssm_vecdot_hmap_dev) &&
            (beta_type == GGUF_TYPE_F32 || beta_type == GGUF_TYPE_Q4_0 || beta_type == GGUF_TYPE_Q8_0) &&
            picolm_gpu_ssm_vecdot_dev(g_ssm_vecdot_out_dev, g_ssm_vecdot_x_dev,
                                       m->gpu.ssm_beta_dev[il], beta_type, dim,
                                       n_v_heads, (int)row_bytes, ssm_vecdot_hmap_dev,
                                       m->gpu.device) &&
            picolm_gpu_sync(m->gpu.device) &&
            picolm_gpu_memcpy(beta, g_ssm_vecdot_out_dev,
                               (size_t)n_v_heads * sizeof(float), -1, m->gpu.device)) {
            for (int h = 0; h < n_v_heads; h++) beta[h] = 1.0f / (1.0f + expf(-beta[h]));
        } else
#endif
        {
            for (int h = 0; h < n_v_heads; h++) {
                int gh = beta_map[h];
                const uint8_t *head_data = (const uint8_t *)lw->ssm_beta + (size_t)gh * row_bytes;
                float sum;
                if (beta_type == GGUF_TYPE_Q8_0) sum = vec_dot_q8_0_q8_0_deltas(xb_q8, xb_q8_d, head_data, dim);
                else if (beta_type == GGUF_TYPE_Q4_0) sum = vec_dot_q4_0_q8_0(head_data, xb_q8, dim);
                else sum = vec_dot(head_data, s->xb, dim, beta_type);
                beta[h] = 1.0f / (1.0f + expf(-sum));
            }
        }
        /* beta_map is stack-allocated */
    }
    if (xb_q8 != (void *)xb_q8_stack) free(xb_q8);
    if (xb_q8_d != xb_q8_d_stack) free(xb_q8_d);
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("beta[:8]", beta, n_v_heads, 8);
#endif

    /* 11. Gate expansion: exp(gate) -> [dt_rank] */
    float *gate_exp = beta + n_v_heads; /* [dt_rank] */
    for (int h = 0; h < n_v_heads; h++) {
        float g = gate[h];
        float ge = (g < -50.0f) ? 0.0f : expf(g);
        gate_exp[h] = ge;
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("gate_exp[:8]", gate_exp, n_v_heads, 8);
#endif

    /* 12-17. State recurrence, threaded across n_v_heads (each head's
     * [d_state x d_state] state block is fully independent -- no
     * cross-head data dependency within a single token, only token-to-
     * token via `state` persisting across calls). Previously this was
     * four separate fully-serial for-h loops (decay/sk/update/output)
     * with zero threading on what is the dominant per-token FLOP cost
     * for SSM layers (O(n_v_heads * d_state^2) each for sk/update/output). */
    float *state = s->ssm_state[il];
    int repeat = n_v_heads / n_k_heads;
    float *ssm_output = gate_exp + n_v_heads; /* [d_state * n_v_heads], dim-major */

    ssm_head_ctx_t ssm_ctx;
    ssm_ctx.state = state; ssm_ctx.d_state = d_state; ssm_ctx.head_v_dim = head_v_dim;
    ssm_ctx.n_v_heads = n_v_heads; ssm_ctx.repeat = repeat;
    ssm_ctx.q_conv = q_conv; ssm_ctx.k_conv = k_conv; ssm_ctx.v_conv = v_conv;
    ssm_ctx.gate_exp = gate_exp; ssm_ctx.beta = beta;
    ssm_ctx.ssm_output = ssm_output; /* shared, dim-major [d*n_v_heads+h] */
#ifdef PICOLM_GPU
    if (1) {
        int rec_done = 0;
        if (m->gpu.ssm_state_dev[il]) {
            /* m->gpu.ssm_state_dev[il] is the ONLY up-to-date copy of this
             * layer's recurrence state once GPU-native decode is running:
             * picolm_gpu_ssm_recurrence_dev() deliberately never D2H's the
             * state back to s->ssm_state[il] (that round-trip is exactly
             * what the persistent-state optimization exists to avoid, see
             * "State remains on device - no D2H needed" in backend_gpu.cu).
             * s->ssm_state[il] (== `state` below) is therefore stale/
             * divergent from the moment the first token after prefill goes
             * through this path, and only gets further out of date with
             * every subsequent token.
             *
             * Previously, a failed recurrence_dev call here silently fell
             * back to picolm_gpu_ssm_recurrence(state, ...) or the CPU
             * tensor_parallel_for path -- both of which read that same
             * stale `state`. That computes ssm_output for this token from
             * the wrong starting state (corrupting this token's logits,
             * which then feeds back into generation as the next input
             * token), while leaving the real, correct device state
             * un-updated for this token -- a silent, undetectable
             * correctness bug. A transient failure here (e.g. gpuMalloc
             * contention) becomes far more likely under added system load
             * (such as running under `perf record`), which is consistent
             * with the early-stop/garbage-output reports.
             *
             * Retry once (transient GPU errors are typically one-off),
             * then fail loudly rather than silently corrupting output. */
            if (picolm_gpu_ssm_recurrence_dev(m->gpu.ssm_state_dev[il],
                                               q_conv, k_conv, v_conv,
                                               gate_exp, beta, ssm_output,
                                               n_v_heads, d_state, repeat, m->gpu.device)) {
                rec_done = 1;
            } else if (picolm_gpu_ssm_recurrence_dev(m->gpu.ssm_state_dev[il],
                                               q_conv, k_conv, v_conv,
                                               gate_exp, beta, ssm_output,
                                               n_v_heads, d_state, repeat, m->gpu.device)) {
                fprintf(stderr, "WARN: GPU SSM recurrence retry succeeded "
                        "(layer %d, pos %d) after an initial failure -- "
                        "investigate transient GPU errors.\n", il, pos);
                rec_done = 1;
            } else {
                fprintf(stderr, "FATAL: GPU SSM recurrence failed twice for "
                        "layer %d, pos %d with device-resident state present. "
                        "Falling back to CPU/host state here would silently "
                        "corrupt this and all subsequent tokens (device state "
                        "is the sole up-to-date copy and is not host-synced "
                        "per-token). Aborting instead of producing wrong "
                        "output. Check GPU memory pressure/errors.\n", il, pos);
                abort();
            }
        }
        if (!rec_done) {
            /* No device-resident state exists for this layer (e.g. GPU
             * disabled, or state never allocated) -- `state` (host) is the
             * authoritative copy in this case, so these paths are safe. */
            if (!picolm_gpu_ssm_recurrence(state, q_conv, k_conv, v_conv,
                                            gate_exp, beta, ssm_output,
                                            n_v_heads, d_state, repeat, m->gpu.device)) {
                tensor_parallel_for(n_v_heads, ssm_head_task, &ssm_ctx);
            }
        }
    } else
#endif
    {
        tensor_parallel_for(n_v_heads, ssm_head_task, &ssm_ctx);
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("ssm_out_pre[:8]", ssm_output, head_v_dim, 8);
#endif

    /* 18. Gated normalization */
    /* ssm_output: [d * n_v_heads + h] (dim-major from delta_net output) */
    float *norm_w = s->ssm_norm_w[il]; /* [head_v_dim] */
    float *final_output = ssm_output + d_state * n_v_heads; /* [head_v_dim * n_v_heads] */
    {
        int gn_done = 0;
#ifdef PICOLM_GPU
        /* Try GPU gated norm if GPU is active.
         * Note: we pass head_map=NULL so the kernel writes in sequential order.
         * The GGUF v-head remap (do_remap) is handled separately in step 19. */
        if (gpu_lw && m->gpu.active) {
            if (picolm_gpu_ssm_gated_norm(final_output, ssm_output, s->xb2,
                                           norm_w, NULL, head_v_dim, n_v_heads, eps,
                                           m->gpu.device)) {
                gn_done = 1;
            }
        }
#endif
        if (!gn_done) {
            /* CPU fallback */
            for (int h = 0; h < n_v_heads; h++) {
                float nrm = 0.0f;
                for (int d = 0; d < head_v_dim; d++) {
                    float v = ssm_output[d * n_v_heads + h];
                    nrm += v * v;
                }
                nrm = 1.0f / sqrtf(nrm / (float)head_v_dim + eps);
                for (int d = 0; d < head_v_dim; d++) {
                    float v = ssm_output[d * n_v_heads + h];
                    float zv = s->xb2[h * head_v_dim + d];
                    float silu_z = zv * (1.0f / (1.0f + expf(-zv)));
                    final_output[h * head_v_dim + d] = v * nrm * norm_w[d] * silu_z;
                }
            }
        }
    }
#ifdef DEBUG_SSM
    if (il == 0 || il == 8 || il == 16 || il == 32 || il == 48 || il == 60) {
        dbg_vec("xb2[:8]", s->xb2, 8, 8);
        dbg_vec("final_out[:8]", final_output, 8, 8);
    }
#endif

    /* 19. Reshape to [value_dim] and output projection */
    /* final_output is [head_v_dim * n_v_heads] = [value_dim] = [4096] */
    /* ssm_out: [n_embd, value_dim] - GGUF columns may be reordered */
    float *fo_gguf = NULL;
    if (do_remap) {
        fo_gguf = alloca(c->ssm_d_inner * sizeof(float));
        for (int h = 0; h < n_v_heads; h++) {
            int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
            memcpy(fo_gguf + gh * head_v_dim, final_output + h * head_v_dim, head_v_dim * sizeof(float));
        }
    }
    const float *ssm_out_src = do_remap ? fo_gguf : final_output;
    int ssm_out_gpu_done = 0;
#ifdef PICOLM_GPU
    if (gpu_lw) {
        gpu_layer_weights_t *gl = (gpu_layer_weights_t *)gpu_lw;
        size_t in_bytes = (size_t)c->ssm_d_inner * sizeof(float);
        size_t out_bytes = (size_t)dim * sizeof(float);
        if (gl->ssm_out &&
            ssm_out_dev_scratch_ensure(m->gpu.device, in_bytes, out_bytes) &&
            picolm_gpu_memcpy(g_ssm_mm_outin_dev, ssm_out_src, in_bytes, 1, m->gpu.device) &&
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ssm_out,
                                   g_ssm_mm_outres_dev, g_ssm_mm_outin_dev, 1, m->gpu.device, 0, 0) &&
            picolm_gpu_sync(m->gpu.device) &&
            picolm_gpu_memcpy(residual, g_ssm_mm_outres_dev, out_bytes, -1, m->gpu.device)) {
            ssm_out_gpu_done = 1;
        }
    }
#endif
    if (!ssm_out_gpu_done)
    {
        matmul(residual, ssm_out_src, lw->ssm_out, c->ssm_d_inner, dim, lw->type_ssm_out);
    }
    #ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("residual[:8]", residual, 8, 8);
#endif
    /* 20. Residual add */
    vec_add(x, residual, dim);

    /* 21. Post-attention norm + FFN (only if MLP weights exist for this layer) */
    if (c->has_moe) {
        /* MoE forward pass */
        rmsnorm(s->xb, x, s->post_attn_norm_w[il], dim, eps);
        moe_forward(m, s, s->xb, s->xb, lw);
        vec_add(x, s->xb, dim);
    } else if (lw->ffn_gate && lw->ffn_up && lw->ffn_down) {
        rmsnorm(s->xb, x, s->post_attn_norm_w[il], dim, eps);
#ifdef PICOLM_GPU
        /* Fused FFN on GPU: y = down(silu(gate(x)) * up(x)) in one command
         * buffer (3 dispatches -> 1); s->xb aliases in/out safely. On miss
         * fall through to the per-matmul CPU path. Matches model_forward. */
        if (gpu_lw) {
            gpu_layer_weights_t *gl = (gpu_layer_weights_t *)gpu_lw;
            int mlp_ok = 0;
            if (gl->ffn_gate && gl->ffn_up && gl->ffn_down) {
                mlp_ok = picolm_gpu_expert_mlp((picolm_gpu_tensor_t *)gl->ffn_gate,
                                      (picolm_gpu_tensor_t *)gl->ffn_up,
                                      (picolm_gpu_tensor_t *)gl->ffn_down,
                                      s->xb, s->xb, 1);
            }
            if (mlp_ok)
                goto ssm_ffn_done;
        }
        /* Defensive: the qkv/gate/ssm_out matmuls above now go through
         * matmul_dev directly and never touch tensor_set_gpu_tensor,
         * but clear it before the CPU-fallback matmul() calls below,
         * which read the global gpu_tensor with no paired set/clear. */
        if (gpu_lw) tensor_set_gpu_tensor(NULL, 0);
#endif
        matmul(s->hb, s->xb, lw->ffn_gate, dim, c->n_ffn, lw->type_ffn_gate);
        matmul(s->hb2, s->xb, lw->ffn_up, dim, c->n_ffn, lw->type_ffn_up);
        silu(s->hb, c->n_ffn);
        elemwise_mul(s->hb, s->hb, s->hb2, c->n_ffn);
        matmul(s->xb, s->hb, lw->ffn_down, c->n_ffn, dim, lw->type_ffn_down);
#ifdef PICOLM_GPU
ssm_ffn_done:
#endif
        vec_add(x, s->xb, dim);
    }
}

/* GPU-native SSM forward pass. Operates entirely on device-resident
 * pipeline buffers (pipe_x, pipe_xb, ssm_qkv_raw, ssm_conv_out, etc.).
 * Zero H2D/D2H. The residual add writes back into pipe_x, and the FFN
 * block writes into pipe_x via the attention-layer FFN pipeline buffers.
 *
 * Returns 1 on success, 0 if any prerequisite is missing (caller should
 * fall back to D2H + ssm_forward + H2D).
 */
#ifdef PICOLM_GPU
static int
ssm_forward_gpu(model_t *m, run_state_t *s, float *x, float *residual,
                layer_weights_t *lw, int il, int pos, void *gpu_lw, int device) {
    (void)x; (void)residual;
    gpu_layer_weights_t *gl = (gpu_layer_weights_t *)gpu_lw;
    model_config_t *c = &m->config;
    gpu_weights_t *gw = &m->gpu;
    int dim = c->n_embd;
    int d_conv = c->ssm_d_conv;
    int d_state = c->ssm_d_state;
    int n_k_heads = c->ssm_n_group;
    int n_v_heads = c->ssm_dt_rank;
    int conv_dim = 2 * d_state * n_k_heads + c->ssm_d_inner;
    int qk_dim = d_state * n_k_heads;
    int head_v_dim = c->ssm_d_inner / n_v_heads;
    int repeat = n_v_heads / n_k_heads;
    float eps = c->rms_norm_eps;

    /* Bug 3 fix: this GPU-native path has no MoE FFN implementation --
     * step 16 below only wires up dense ffn_gate/up/down. On a MoE layer
     * gl->ffn_gate would be NULL and the block would silently no-op,
     * dropping the FFN entirely. Bail before any state mutation. */
    if (c->has_moe) return 0;

    /* Bug 2 fix: picolm_gpu_ssm_vecdot_kernel only implements F32/Q4_0/Q8_0
     * and silently leaves output at 0.0f for any other type. Check up front,
     * before conv1d touches persistent state. */
    if (lw->type_ssm_alpha != GGUF_TYPE_F32 && lw->type_ssm_alpha != GGUF_TYPE_Q4_0 &&
        lw->type_ssm_alpha != GGUF_TYPE_Q8_0) return 0;
    if (lw->type_ssm_beta != GGUF_TYPE_F32 && lw->type_ssm_beta != GGUF_TYPE_Q4_0 &&
        lw->type_ssm_beta != GGUF_TYPE_Q8_0) return 0;

    /* Prerequisites: all GPU tensors must be resident */
    if (!gl->attn_qkv || !gl->attn_gate_ssm || !gl->ssm_out || !gl->ssm_conv1d) return 0;
    if (!gw->ssm_alpha_dev[il] || !gw->ssm_beta_dev[il] || !gw->ssm_a_dev[il]) return 0;
    if (!gw->ssm_dt_dev[il] || !gw->ssm_norm_dev[il]) return 0;
    if (!gw->ssm_conv_state_dev[il] || !gw->ssm_state_dev[il]) return 0;

    /* Bug 3 fix (2026-08-11): this GPU-native path has no MoE FFN
     * implementation -- step 16 below only wires up dense ffn_gate/up/
     * down. On a MoE layer gl->ffn_gate would be NULL and that whole
     * block would silently no-op, dropping the FFN's contribution
     * entirely (not a numerical difference -- a missing computation).
     * Bail here, before anything below has mutated persistent state, so
     * the caller's CPU-hybrid ssm_forward() fallback (which does call
     * moe_forward()) is always safe to take. */
    if (c->has_moe) return 0;

    /* Bug 2 fix (2026-08-11): picolm_gpu_ssm_vecdot_kernel only
     * implements F32/Q4_0/Q8_0 and silently leaves output at 0.0f for
     * any other type (see steps 9-10 below). Check up front -- like
     * has_moe above, this must happen before conv1d touches persistent
     * state, not at the point of use, or a return-0 there would no
     * longer be safe for the caller to fall back from. */
    if (lw->type_ssm_alpha != GGUF_TYPE_F32 && lw->type_ssm_alpha != GGUF_TYPE_Q4_0 &&
        lw->type_ssm_alpha != GGUF_TYPE_Q8_0) return 0;
    if (lw->type_ssm_beta != GGUF_TYPE_F32 && lw->type_ssm_beta != GGUF_TYPE_Q4_0 &&
        lw->type_ssm_beta != GGUF_TYPE_Q8_0) return 0;

    /* Pipeline buffers */
    float *pipe_x = picolm_gpu_pipe_x(device);
    float *pipe_xb = picolm_gpu_pipe_xb(device);
    float *pipe_ffn_norm = picolm_gpu_pipe_ffn_norm(device);
    float *pipe_gate = picolm_gpu_pipe_gate(device);
    float *pipe_up = picolm_gpu_pipe_up(device);
    float *ssm_qkv_raw = picolm_gpu_ssm_qkv_raw(device);
    float *ssm_conv_out = picolm_gpu_ssm_conv_out(device);
    float *ssm_xb2 = picolm_gpu_ssm_xb2(device);
    float *ssm_xb2_remap = picolm_gpu_ssm_xb2_remap(device);
    float *ssm_v_remap = picolm_gpu_ssm_v_remap(device);
    float *ssm_alpha_raw = picolm_gpu_ssm_alpha_raw(device);
    float *ssm_beta_raw = picolm_gpu_ssm_beta_raw(device);
    float *ssm_gate_exp = picolm_gpu_ssm_gate_exp(device);
    float *ssm_beta_d = picolm_gpu_ssm_beta(device);
    float *ssm_output = picolm_gpu_ssm_output(device);
    float *ssm_final_output = picolm_gpu_ssm_final_output(device);
    if (!pipe_x || !pipe_xb || !ssm_qkv_raw || !ssm_conv_out ||
        !ssm_xb2 || !ssm_xb2_remap || !ssm_alpha_raw || !ssm_beta_raw ||
        !ssm_gate_exp || !ssm_beta_d || !ssm_output || !ssm_final_output) return 0;

    int n_k = c->ssm_n_group;
    int n_vpk = n_v_heads / n_k;
    int half_vpk = n_vpk / 2;
    int do_remap = !m->from_safetensors && n_k > 0 && n_k < n_v_heads && half_vpk > 0;
    const int *head_map_dev = do_remap ? (const int *)gw->ssm_head_map_dev : NULL;
    if (do_remap && !head_map_dev) return 0;

    /* 1. RMSNorm: pipe_xb = rmsnorm(pipe_x, attn_norm) */
    if (!picolm_gpu_rmsnorm_dev(pipe_xb, pipe_x,
            (float *)gw->attn_norm_dev[il], dim, eps, device)) return 0;

    /* 2. QKV projection: ssm_qkv_raw = attn_qkv @ pipe_xb */
    if (!picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_qkv,
            ssm_qkv_raw, pipe_xb, 1, device, 0, 0)) return 0;

    /* 3. Z gate: ssm_xb2 = attn_gate_ssm @ pipe_xb */
    if (!picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_gate_ssm,
            ssm_xb2, pipe_xb, 1, device, 0, 0)) return 0;

    /* 3b. Head permute xb2 (GGUF v-head reorder) */
    float *xb2_src = ssm_xb2;
    if (do_remap) {
        if (!picolm_gpu_ssm_head_permute_dev(ssm_xb2_remap, ssm_xb2,
                head_map_dev, head_v_dim, n_v_heads, device)) return 0;
        xb2_src = ssm_xb2_remap;
    }

    /* 4. Conv1d + silu + state shift: ssm_conv_out = conv1d(ssm_qkv_raw) */
    /* DEBUG: download pre-conv state for CPU reference comparison */
    float *dbg_pre_state = NULL;
    float *dbg_qkv_raw = NULL;
    float *dbg_conv_w = NULL;
    if (il == 0 && pos < 40 && getenv("PICOLM_SSM_DBG")) {
        dbg_pre_state = alloca((d_conv - 1) * conv_dim * sizeof(float));
        dbg_qkv_raw = alloca(conv_dim * sizeof(float));
        dbg_conv_w = alloca(d_conv * conv_dim * sizeof(float));
        picolm_gpu_sync(device);
        picolm_gpu_memcpy(dbg_pre_state, gw->ssm_conv_state_dev[il], (d_conv-1) * conv_dim * sizeof(float), -1, device);
        picolm_gpu_memcpy(dbg_qkv_raw, ssm_qkv_raw, conv_dim * sizeof(float), -1, device);
        picolm_gpu_memcpy(dbg_conv_w, gw->ssm_conv1d_dev[il], d_conv * conv_dim * sizeof(float), -1, device);
    }
    if (!picolm_gpu_ssm_conv1d_dev(ssm_conv_out,
            (float *)gw->ssm_conv_state_dev[il],
            ssm_qkv_raw,
            (float *)gw->ssm_conv1d_dev[il],
            conv_dim, d_conv, device)) {
        fprintf(stderr, "FATAL: GPU SSM conv1d failed for layer %d, pos %d "
                "-- aborting rather than risk a double conv-state shift via "
                "CPU fallback.\n", il, pos);
        abort();
    }

    /* DEBUG: compare GPU conv1d output with CPU reference */
    if (il == 0 && pos < 40 && getenv("PICOLM_SSM_DBG")) {
        picolm_gpu_sync(device);
        float *gpu_conv_out_buf = alloca(8 * sizeof(float));
        float *cpu_conv_out_buf = alloca(8 * sizeof(float));
        picolm_gpu_memcpy(gpu_conv_out_buf, ssm_conv_out, 8 * sizeof(float), -1, device);
        for (int co = 0; co < 8; co++) {
            float sum = 0.0f;
            for (int d = 0; d < d_conv - 1; d++)
                sum += dbg_conv_w[d + co * d_conv] * dbg_pre_state[d * conv_dim + co];
            sum += dbg_conv_w[(d_conv-1) + co * d_conv] * dbg_qkv_raw[co];
            cpu_conv_out_buf[co] = sum / (1.0f + expf(-sum));
        }
        fprintf(stderr, "SSMDBG l=%d p=%d conv1d GPU[:8]={", il, pos);
        for(int i=0;i<8;i++) fprintf(stderr,"%s%.6f",i?",":"",gpu_conv_out_buf[i]);
        fprintf(stderr, "} CPUref[:8]={");
        for(int i=0;i<8;i++) fprintf(stderr,"%s%.6f",i?",":"",cpu_conv_out_buf[i]);
        fprintf(stderr, "}\n");
        /* Dump state rows for channel 0 */
        fprintf(stderr, "SSMDBG state[:4][0]={");
        for(int d=0;d<d_conv-1;d++) fprintf(stderr,"%s%.6f",d?",":"",dbg_pre_state[d*conv_dim+0]);
        fprintf(stderr, "} qkv[0]=%.6f w[0][:4]={", dbg_qkv_raw[0]);
        for(int d=0;d<d_conv;d++) fprintf(stderr,"%s%.6f",d?",":"",dbg_conv_w[d]);
        fprintf(stderr, "}\n");
    }

    /* 5. Split Q/K/V by pointer offset (no copy) */
    float *q_dev = ssm_conv_out;
    float *k_dev = ssm_conv_out + qk_dim;
    float *v_dev = ssm_conv_out + 2 * qk_dim;

    /* 6. L2 normalize Q in-place, then scale by 1/sqrt(d_state).
     * 7. L2 normalize K in-place -- NOT scaled.
     * CORRECTED: the CPU reference's qk_dim is d_state * n_k_heads,
     * the size of ONE of Q/K (q_conv and k_conv are separate arrays).
     * Its step-8 loop `for (i < qk_dim) q_conv[i] *= q_scale` therefore
     * touches only q_conv -- k_conv is never scaled. The previous version
     * scaled k_dev by q_scale too, based on a misreading that qk_dim
     * covered both; that shrinks every K vector by 1/sqrt(d_state)
     * before the recurrence, on every token of every layer.
     * Fixed: K gets extra_scale=1.0 (identity, normalize only). */
    float q_scale = 1.0f / sqrtf((float)d_state);
    if (!picolm_gpu_ssm_l2norm_dev(q_dev, d_state, n_k_heads, 1e-12f, q_scale, device) ||
        !picolm_gpu_ssm_l2norm_dev(k_dev, d_state, n_k_heads, 1e-12f, 1.0f, device)) {
        fprintf(stderr, "FATAL: GPU SSM L2 norm failed for layer %d, pos %d.\n", il, pos);
        abort();
    }

    /* 8. Head permute V (if GGUF reorders) */
    if (do_remap) {
        if (!picolm_gpu_ssm_head_permute_dev(ssm_v_remap, v_dev,
                head_map_dev, head_v_dim, n_v_heads, device)) {
            fprintf(stderr, "FATAL: GPU SSM V head-permute failed for layer %d, pos %d.\n", il, pos);
            abort();
        }
        v_dev = ssm_v_remap;
    }

    /* 9. Alpha vecdot -- type already validated at top of function */
    {
        gguf_type_t alpha_type = lw->type_ssm_alpha;
        size_t row_bytes = gguf_type_row_size(alpha_type, dim);
        if (!picolm_gpu_ssm_vecdot_dev(ssm_alpha_raw, pipe_xb,
                gw->ssm_alpha_dev[il], alpha_type, dim,
                n_v_heads, (int)row_bytes, head_map_dev, device)) {
            fprintf(stderr, "FATAL: GPU SSM alpha vecdot failed for layer %d, pos %d.\n", il, pos);
            abort();
        }
    }

    /* 10. Beta vecdot -- type already validated at top of function */
    {
        gguf_type_t beta_type = lw->type_ssm_beta;
        size_t row_bytes = gguf_type_row_size(beta_type, dim);
        if (!picolm_gpu_ssm_vecdot_dev(ssm_beta_raw, pipe_xb,
                gw->ssm_beta_dev[il], beta_type, dim,
                n_v_heads, (int)row_bytes, head_map_dev, device)) {
            fprintf(stderr, "FATAL: GPU SSM beta vecdot failed for layer %d, pos %d.\n", il, pos);
            abort();
        }
    }

    /* 11. Softplus(alpha)+gate + sigmoid(beta) */
    if (!picolm_gpu_ssm_gate_beta_dev(ssm_gate_exp, ssm_beta_d,
            ssm_alpha_raw, ssm_beta_raw,
            (float *)gw->ssm_a_dev[il],
            (float *)gw->ssm_dt_dev[il],
            n_v_heads, device)) {
        fprintf(stderr, "FATAL: GPU SSM gate/beta activation failed for layer %d, pos %d.\n", il, pos);
        abort();
    }

    /* 12. SSM recurrence (thread-0 kernel by default -- bit-exact with
     * CPU NEON; warp-shuffle only if PICOLM_SSM_WARP_KERNEL_VALIDATED).
     * This mutates gw->ssm_state_dev[il], persistent per-token state. */
    if (!picolm_gpu_ssm_recurrence_pipeline_dev(gw->ssm_state_dev[il],
            q_dev, k_dev, v_dev,
            ssm_gate_exp, ssm_beta_d,
            ssm_output, n_v_heads, d_state, repeat, device)) {
        fprintf(stderr, "FATAL: GPU SSM recurrence failed for layer %d, pos %d "
                "with device-resident state already mutated -- "
                "aborting rather than risk corrupting generation.\n", il, pos);
        abort();
    }

    /* DEBUG: dump recurrence output (dim-major ssm_output) for comparison */
    if (il == 0 && pos < 40 && getenv("PICOLM_SSM_DBG")) {
        picolm_gpu_sync(device);
        float *dbg_ssm_out = alloca(8 * sizeof(float));
        picolm_gpu_memcpy(dbg_ssm_out, ssm_output, 8 * sizeof(float), -1, device);
        fprintf(stderr, "SSMDBG l=%d p=%d ssm_out_pre[:8]={", il, pos);
        for(int i=0;i<8;i++) fprintf(stderr,"%s%.6f",i?",":"",dbg_ssm_out[i]);
        fprintf(stderr, "}\n");
    }

    /* 13. Gated normalization (fuses RMSNorm of dim-major ssm_output
     *     with silu(xb2) gating, writes head-major with GGUF reorder) */
    if (!picolm_gpu_ssm_gated_norm_dev(ssm_final_output, ssm_output, xb2_src,
            (float *)gw->ssm_norm_dev[il],
            head_map_dev,
            head_v_dim, n_v_heads, eps, device)) {
        fprintf(stderr, "FATAL: GPU SSM gated norm failed for layer %d, pos %d.\n", il, pos);
        abort();
    }

    /* 14. SSM output projection: pipe_xb = ssm_out @ ssm_final_output */
    if (!picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ssm_out,
            pipe_xb, ssm_final_output, 1, device, 0, 0)) {
        fprintf(stderr, "FATAL: GPU SSM output projection failed for layer %d, pos %d.\n", il, pos);
        abort();
    }

    /* DEBUG: run CPU ssm_forward in parallel to compare intermediate values */
    if (il == 0 && pos < 38 && getenv("PICOLM_SSM_DBG")) {
        picolm_gpu_sync(device);
        /* Download GPU intermediates */
        float *gpu_final = alloca(c->ssm_d_inner * sizeof(float));
        float *gpu_residual = alloca(dim * sizeof(float));
        float *gpu_pipe_x = alloca(dim * sizeof(float));
        float *gpu_xb2 = alloca(c->ssm_d_inner * sizeof(float));
        picolm_gpu_memcpy(gpu_final, ssm_final_output, c->ssm_d_inner * sizeof(float), -1, device);
        picolm_gpu_memcpy(gpu_residual, pipe_xb, dim * sizeof(float), -1, device);
        picolm_gpu_memcpy(gpu_pipe_x, pipe_x, dim * sizeof(float), -1, device);
        picolm_gpu_memcpy(gpu_xb2, xb2_src, c->ssm_d_inner * sizeof(float), -1, device);

        fprintf(stderr, "SSMDBG l=%d p=%d GPU ssm_final[:4]={", il, pos);
        for(int i=0;i<4;i++) fprintf(stderr,"%s%.6f",i?",":"",gpu_final[i]);
        fprintf(stderr, "} GPU res[:4]={");
        for(int i=0;i<4;i++) fprintf(stderr,"%s%.6f",i?",":"",gpu_residual[i]);
        fprintf(stderr, "} GPU xb2[:4]={");
        for(int i=0;i<4;i++) fprintf(stderr,"%s%.6f",i?",":"",gpu_xb2[i]);
        fprintf(stderr, "}\n");

        /* Compare with CPU: run CPU ssm_forward on a copy of pipe_x */
        float *cpu_x = alloca(dim * sizeof(float));
        float *cpu_res = alloca(dim * sizeof(float));
        memcpy(cpu_x, gpu_pipe_x, dim * sizeof(float));
        /* We can't easily call ssm_forward here because it modifies shared state.
         * Instead, compare the ssm_out matmul input (ssm_final_output) manually. */
    }

    /* 15. Residual add: pipe_x += pipe_xb */
    if (!picolm_gpu_residual_add(pipe_x, pipe_x, pipe_xb, 1, dim, dim, device)) {
        fprintf(stderr, "FATAL: GPU SSM residual add failed for layer %d, pos %d.\n", il, pos);
        abort();
    }

    /* FFN handled by caller (model_forward_gpu) via the !did_cpu_ssm block,
     * which uses fully device-native picolm_gpu_matmul_dev calls.
     * picolm_gpu_expert_mlp expects host pointers and would corrupt data
     * if called with device-resident pipe_x. */
    return 1;
}
#endif

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
 * KV cache is in F16 format, interleaved by [pos][kv_head][head_dim].
 * ================================================================ */

/* Prefill attention: same math as decode's attn_core, applied to every
 * (token, head) pair in the batch. Flattened into one index space so a
 * single tensor_parallel_for() call threads the whole layer's attention
 * at once (previously this was a fresh, unthreaded, F32-dequanting
 * reimplementation that also scored the full [0, n_kv) range before
 * masking -- wasting ~2x the dot products vs. a causal-limited loop, and
 * allocating/freeing O(n_kv * n_kv_heads * head_dim) scratch per layer). */
typedef struct {
    int n_heads, n_kv_heads, kv_mul, head_dim, start_pos;
    int kv_type_k, kv_type_v;
    size_t kv_row_size_k, kv_row_size_v;
    size_t kv_head_stride_k, kv_head_stride_v;
    const uint8_t *kcache, *vcache;
    const float *q_batch;   /* [n_tokens][n_heads * head_dim] */
    float *xb_batch;        /* [n_tokens][xb_stride] */
    int xb_stride;
    int kv_hadamard_k, kv_hadamard_v;
    int kv_hadamard_size;
    float attn_scale;
} prefill_attn_ctx_t;

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

static void batch_attention_layer(
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

/* ================================================================
 * Device-native batched SSM prefill layer (zero H2D/D2H)
 * ================================================================ */
#define SSM_DBG_SYNC /* disabled */
static int ssm_prefill_layer_gpu(model_t *m, run_state_t *s,
    float *bx, float *bxb, float *bq, float *battn_out, float *bffn_norm,
    float *bgate, float *bup, layer_weights_t *lw, int l,
    int n_tokens, int start_pos, int dev) {
#ifdef PICOLM_GPU
    (void)start_pos;
    model_config_t *c = &m->config;
    gpu_weights_t *gw = &m->gpu;
    int dim=c->n_embd, d_state=c->ssm_d_state;
    int n_k=c->ssm_n_group, n_v=c->ssm_dt_rank;
    int conv_dim=2*d_state*n_k+c->ssm_d_inner;
    int value_dim=c->ssm_d_inner;
    int hvdim=value_dim/n_v;
    float eps=c->rms_norm_eps;
    int repeat=n_v/n_k, qk_dim=d_state*n_k;
    gpu_layer_weights_t *gl = &m->gpu.layers[l];
    float *cs_dev=(float*)gw->ssm_conv_state_dev[l];
    float *st_dev=(float*)gw->ssm_state_dev[l];
    if(!cs_dev||!st_dev||!gl->ssm_conv1d||!gl->attn_qkv||!gl->attn_gate_ssm||!gl->ssm_out) return 0;
    /* do_remap: must match CPU reference (ssm_prefill_layer, ssm_forward, ssm_forward_gpu) */
    int n_vpk = n_k > 0 ? n_v / n_k : 0, half_vpk = n_vpk / 2;
    int do_remap = !m->from_safetensors && n_k > 0 && n_k < n_v && half_vpk > 0;
    int ok=1;
    /* Pipe buffer stride: matches pipeline_batch_alloc */
    int xb2_stride = c->n_heads * c->head_dim * (c->has_ssm ? 2 : 1);
    /* bffn_norm temporarily holds the RMSNorm'd input until vecdot consumes it.
     * bxb becomes the Q/K/V-conv working buffer immediately after. */

    /* Per-step diagnostic: dump last token RMS to compare GPU vs CPU */
    #define _DBG_RMS(name, ptr, stride) do { \
        if (l == 0) { \
            float _rms=0; float _v[8]; \
            picolm_gpu_memcpy(_v, ptr + (size_t)(n_tokens-1) * stride, 32, -1, dev); \
            for (int _i=0;_i<8;_i++) _rms += _v[_i]*_v[_i]; \
            _rms = sqrtf(_rms/8); \
            fprintf(stderr, "[DBG l%d] %s rms_last=%.6f {%.6f %.6f %.6f %.6f}\n", l, name, _rms, _v[0],_v[1],_v[2],_v[3]); \
        } \
    } while(0)
    /* Canonical xb2_stride-aware per-token dump: reads 4 floats per token, reports RMS */
    #define _DBG_TOK(name, ptr) do { \
        if (l == 0) { \
            float _vt[40]; \
            for (int _ti=0;_ti<10;_ti++) \
                picolm_gpu_memcpy(_vt+_ti*4, ptr + (size_t)_ti * xb2_stride, 16, -1, dev); \
            float _trms=0; \
            for (int _ti=0;_ti<10;_ti++) { float _r=0; for(int _j=0;_j<4;_j++) _r+=_vt[_ti*4+_j]*_vt[_ti*4+_j]; _r=sqrtf(_r/4); _trms+=_r*_r; } \
            _trms=sqrtf(_trms/10); \
            fprintf(stderr, "[DBG l%d] %s tok0={%.6f %.6f %.6f %.6f} tok9={%.6f %.6f %.6f %.6f} mean_rms=%.6f\n", \
                l, name, _vt[0],_vt[1],_vt[2],_vt[3], _vt[36],_vt[37],_vt[38],_vt[39], _trms); \
        } \
    } while(0)

    ok&=picolm_gpu_rmsnorm_batched_dev(bffn_norm,bx,(float*)s->attn_norm_w[l],dim,eps,n_tokens,xb2_stride,dev);
    SSM_DBG_SYNC;
    _DBG_RMS("rmsnorm",bffn_norm,dim);
    /* QKV + Z-gate projections (read from RMSNorm'd input) */
    ok&=picolm_gpu_matmul_dev((picolm_gpu_tensor_t*)gl->attn_qkv,bxb,bffn_norm,n_tokens,dev,xb2_stride,0);
    SSM_DBG_SYNC;
    _DBG_RMS("qkv",bxb,xb2_stride);
    ok&=picolm_gpu_matmul_dev((picolm_gpu_tensor_t*)gl->attn_gate_ssm,bq,bffn_norm,n_tokens,dev,xb2_stride,0);
    SSM_DBG_SYNC;
    _DBG_RMS("gate",bq,xb2_stride);
    /* Dump conv1d weight for debugging */
    if (l == 0) {
        float cw[16], cs[12];
        picolm_gpu_memcpy(cw, gw->ssm_conv1d_dev[l], 64, -1, dev);
        fprintf(stderr, "[DBG l%d] conv1d_w[0][:4]={%.6f,%.6f,%.6f,%.6f} w[1][:4]={%.6f,%.6f,%.6f,%.6f}\n", l, cw[0],cw[1],cw[2],cw[3],cw[4],cw[5],cw[6],cw[7]);
        picolm_gpu_memcpy(cs, cs_dev, 48, -1, dev);
        fprintf(stderr, "[DBG l%d] conv_state[0][:4]={%.6f,%.6f,%.6f,%.6f}\n", l, cs[0],cs[1],cs[2],cs[3]);
    }
    /* Conv1d + silu (in-place on bxb which has QKV output) */
    ok&=picolm_gpu_ssm_conv1d_batch_dev(bxb,cs_dev,bxb,(float*)gw->ssm_conv1d_dev[l],conv_dim,c->ssm_d_conv,n_tokens,dev,xb2_stride);
    SSM_DBG_SYNC;
    _DBG_RMS("conv1d",bxb,xb2_stride);
    /* L2 norm Q/K */
    _DBG_RMS("pre_l2_q",bxb,xb2_stride);
    if(ok){float qs=1.0f/sqrtf((float)d_state);
      ok&=picolm_gpu_ssm_l2norm_batch_dev(bxb,d_state,n_k,n_tokens,xb2_stride,1e-12f,qs,dev);
    SSM_DBG_SYNC;
    _DBG_RMS("l2norm_q",bxb,xb2_stride);
      ok&=picolm_gpu_ssm_l2norm_batch_dev(bxb+qk_dim,d_state,n_k,n_tokens,xb2_stride,1e-12f,1.0f,dev);
    _DBG_RMS("l2norm_k",bxb+qk_dim,xb2_stride);
      _DBG_TOK("Q",bxb);}
    /* Head permute V. bxb+2*qk_dim can't be permuted into itself --
     * that races whenever head_map isn't the identity (concurrent
     * threadblocks, no ordering guarantee, one can overwrite a
     * position another hasn't read yet). Snapshot it out unaliased
     * first (identity copy, head_map=NULL), then permute from that
     * snapshot directly into its real strided destination -- both
     * steps are the same primitive, so this is "permute" used twice
     * (once as a plain copy) rather than a permute paired with an
     * unrelated generic copy kernel. */
    if(ok&&do_remap&&gw->ssm_head_map_dev){
      ok&=picolm_gpu_ssm_head_permute_batch_dev(battn_out,bxb+2*qk_dim,NULL,hvdim,n_v,n_tokens,xb2_stride,value_dim,dev);
    SSM_DBG_SYNC;
      ok&=picolm_gpu_ssm_head_permute_batch_dev(bxb+2*qk_dim,battn_out,(const int*)gw->ssm_head_map_dev,hvdim,n_v,n_tokens,value_dim,xb2_stride,dev);
      /* Z-gate (bq) permute: GGUF order -> natural order, same two-step trick */
    SSM_DBG_SYNC;
      ok&=picolm_gpu_ssm_head_permute_batch_dev(battn_out,bq,NULL,hvdim,n_v,n_tokens,xb2_stride,value_dim,dev);
    SSM_DBG_SYNC;
      ok&=picolm_gpu_ssm_head_permute_batch_dev(bq,battn_out,(const int*)gw->ssm_head_map_dev,hvdim,n_v,n_tokens,value_dim,xb2_stride,dev);}
    SSM_DBG_SYNC;
      _DBG_TOK("V",bxb+2*qk_dim);
    /* Alpha/beta vecdot (read from RMSNorm'd bffn_norm) */
    if(ok){gguf_type_t at=lw->type_ssm_alpha,bt=lw->type_ssm_beta;
      size_t ra=gguf_type_row_size(at,dim),rb=gguf_type_row_size(bt,dim);
      if(at!=0&&at!=2&&at!=8&&at!=30){static int w1=0;if(!w1){fprintf(stderr,"WARN: ssm_prefill_layer_gpu bail: alpha type=%d (need F32/Q4_0/Q8_0/BF16)\n",(int)at);w1=1;}return 0;}
      if(bt!=0&&bt!=2&&bt!=8&&bt!=30){static int w2=0;if(!w2){fprintf(stderr,"WARN: ssm_prefill_layer_gpu bail: beta type=%d (need F32/Q4_0/Q8_0/BF16)\n",(int)bt);w2=1;}return 0;}
      const int *hm=do_remap&&gw->ssm_head_map_dev?(const int*)gw->ssm_head_map_dev:NULL;
      ok&=picolm_gpu_ssm_vecdot_batch_dev(bgate,bffn_norm,(void*)picolm_gpu_tensor_weights((picolm_gpu_tensor_t*)gl->ssm_alpha),at,dim,n_v,n_tokens,(int)ra,hm,dev,xb2_stride);
    SSM_DBG_SYNC;
      ok&=picolm_gpu_ssm_vecdot_batch_dev(bup,bffn_norm,(void*)picolm_gpu_tensor_weights((picolm_gpu_tensor_t*)gl->ssm_beta),bt,dim,n_v,n_tokens,(int)rb,hm,dev,xb2_stride);}
    SSM_DBG_SYNC;
#ifdef PICOLM_SSM_VERIFY
    if(ok && l == 0) {
        picolm_gpu_sync(dev);
        float vtmp[8];
        picolm_gpu_memcpy(vtmp, bgate, sizeof(vtmp), -1, dev);
        fprintf(stderr, "SSM_VERIFY l=0 gpu alpha_raw[:4]={%.6f,%.6f,%.6f,%.6f}\n",
            vtmp[0],vtmp[1],vtmp[2],vtmp[3]);
        picolm_gpu_memcpy(vtmp, bup, sizeof(vtmp), -1, dev);
        fprintf(stderr, "SSM_VERIFY l=0 gpu beta_raw[:4]={%.6f,%.6f,%.6f,%.6f}\n",
            vtmp[0],vtmp[1],vtmp[2],vtmp[3]);
    }
#endif
    /* Gate/beta post-process */
    if(ok){float *dw=(float*)gw->ssm_dt_dev[l],*aw=(float*)gw->ssm_a_dev[l];
      if(!dw||!aw)return 0;
      ok&=picolm_gpu_ssm_gate_beta_batch_dev(bgate,bup,bgate,bup,dw,aw,n_v,n_tokens,dev,xb2_stride);}
    SSM_DBG_SYNC;
    /* Chunked recurrence */
    /* Step-by-step verify: D2H intermediates after each GPU kernel, compare with CPU */
    if(ok && getenv("PICOLM_SSM_VERIFY") && l <= 4) {
        picolm_gpu_sync(dev);
        int cs = 64; if(cs > n_tokens) cs = n_tokens;
        int d = d_state;
        size_t cb = (size_t)n_tokens * conv_dim * sizeof(float);
        size_t ab = (size_t)n_tokens * n_v * sizeof(float);
        float *ch = (float*)malloc(cb), *ah = (float*)malloc(ab), *bh = (float*)malloc(ab);
        float *sh = (float*)calloc(n_v * d * d, sizeof(float));
        /* D2H conv_batch (Q+K+V), alpha, beta */
        for (int _t = 0; _t < n_tokens; _t++) {
            picolm_gpu_memcpy(ch + _t * conv_dim, bxb + (size_t)_t * xb2_stride, conv_dim * sizeof(float), -1, dev);
            picolm_gpu_memcpy(ah + _t * n_v, bgate + (size_t)_t * xb2_stride, n_v * sizeof(float), -1, dev);
            picolm_gpu_memcpy(bh + _t * n_v, bup + (size_t)_t * xb2_stride, n_v * sizeof(float), -1, dev);
        }
        /* D2H state */
        float *st_h = (float*)calloc(n_v * d * d, sizeof(float));
        picolm_gpu_memcpy(st_h, st_dev, n_v * d * d * sizeof(float), -1, dev);
        /* Run CPU recurrence to get intermediates */
        /* We'll manually replicate the chunked steps on CPU and compare with GPU D2H dumps */
        int ca = cs; int co2 = 0;
        /* === DECAY KERNEL === */
        /* D2H d_gl (gate_log), compute CPU decay_mask, D2H GPU decay_mask */
        {
            float *d_gl_dev = (float*)malloc(n_v * ca * sizeof(float));
            float *d_gl_cpu = (float*)calloc(n_v * ca, sizeof(float));
            float *d_dm_dev = (float*)malloc(n_v * ca * ca * sizeof(float));
            float *d_dm_cpu = (float*)calloc(n_v * ca * ca, sizeof(float));
            /* gate_log is ah (alpha_batch) after gate/beta post-process */
            /* Copy gate_log from ah[co2..co2+ca] for all heads */
            for (int h = 0; h < n_v; h++)
                memcpy(d_gl_cpu + h * ca, ah + (co2 * n_v + h), ca * sizeof(float));
            /* Compute CPU decay: cum_g, q_decay, decay_mask */
            for (int h = 0; h < n_v; h++) {
                float cg = 0.0f;
                for (int t = 0; t < ca; t++) {
                    cg += d_gl_cpu[h * ca + t];
                    if (cg > 50) cg = 50;
                }
                float *dm = d_dm_cpu + h * ca * ca;
                for (int t1 = 0; t1 < ca; t1++) {
                    float g = 0.0f;
                    for (int t2 = 0; t2 <= t1; t2++) {
                        g += d_gl_cpu[h * ca + t2];
                        dm[t1 * ca + t2] = expf(-g);
                    }
                }
            }
            fprintf(stderr, "[STEP l0] decay_mask CPU dm[0][0..3]={%.6f,%.6f,%.6f,%.6f} dm[ca-1][0..3]={%.6f,%.6f,%.6f,%.6f}\n",
                d_dm_cpu[0], d_dm_cpu[1], d_dm_cpu[2], d_dm_cpu[3],
                d_dm_cpu[(ca-1)*ca], d_dm_cpu[(ca-1)*ca+1], d_dm_cpu[(ca-1)*ca+2], d_dm_cpu[(ca-1)*ca+3]);
            free(d_gl_dev); free(d_gl_cpu); free(d_dm_dev); free(d_dm_cpu);
        }
        free(ch); free(ah); free(bh); free(sh); free(st_h);
    }
    if(ok){
        ok&=picolm_gpu_ssm_chunked_recurrence_dev(bxb,bgate,bup,st_dev,battn_out,n_tokens,value_dim,xb2_stride,d_state,n_k,n_v,hvdim,repeat,conv_dim,64,dev);
    SSM_DBG_SYNC;
        _DBG_TOK("xb2",battn_out);
        if(ok && getenv("PICOLM_SSM_VERIFY") && l <= 4) {
            /* D2H GPU recurrence output and compare with CPU reference */
            size_t xb2_bytes=(size_t)n_tokens*value_dim*sizeof(float);
            float *xb2_gpu=(float*)malloc(xb2_bytes);
            float *xb2_cpu=(float*)calloc(n_tokens*value_dim,sizeof(float));
            if(xb2_gpu&&xb2_cpu){
                picolm_gpu_sync(dev);
                /* Copy xb2 with proper xb2_stride (battn_out is strided) */
                for (int _t = 0; _t < n_tokens; _t++)
                    picolm_gpu_memcpy(xb2_gpu + _t * value_dim, battn_out + (size_t)_t * xb2_stride, value_dim * sizeof(float), -1, dev);
                /* Run CPU recurrence on same inputs */
                size_t cb=(size_t)n_tokens*conv_dim*sizeof(float);
                size_t ab=(size_t)n_tokens*n_v*sizeof(float);
                float *ch=(float*)malloc(cb),*ah=(float*)malloc(ab),*bh=(float*)malloc(ab);
                if(ch&&ah&&bh){
                    /* Copy conv_batch with proper xb2_stride (not contiguous) */
                    for (int _t = 0; _t < n_tokens; _t++) {
                        picolm_gpu_memcpy(ch + _t * conv_dim, bxb + (size_t)_t * xb2_stride, conv_dim * sizeof(float), -1, dev);
                        picolm_gpu_memcpy(ah + _t * n_v, bgate + (size_t)_t * xb2_stride, n_v * sizeof(float), -1, dev);
                        picolm_gpu_memcpy(bh + _t * n_v, bup + (size_t)_t * xb2_stride, n_v * sizeof(float), -1, dev);
                    }
                    fprintf(stderr, "[DBG] l0 gate_log D2H h0: t0=%.6f t1=%.6f t5=%.6f t10=%.6f t20=%.6f t35=%.6f\n",
                        ah[0], ah[n_v], ah[5*n_v], ah[10*n_v], ah[20*n_v], ah[35*n_v]);
                    float *sh=(float*)calloc(n_v*d_state*d_state,sizeof(float));
                    ssm_chunked_recurrence(ch,ah,bh,sh,xb2_cpu,n_tokens,value_dim,d_state,n_k,n_v,hvdim,repeat,conv_dim,m->ssm_chunk_size);
                    free(sh);
                    fprintf(stderr, "[DBG] l0 CPU xb2 tok0={%.6f %.6f %.6f %.6f} tok9={%.6f %.6f %.6f %.6f}\n",
                        xb2_cpu[0],xb2_cpu[1],xb2_cpu[2],xb2_cpu[3],
                        xb2_cpu[9*value_dim],xb2_cpu[9*value_dim+1],xb2_cpu[9*value_dim+2],xb2_cpu[9*value_dim+3]);
                    /* Compare */
                    float maxd=0;
                    for(int i=0;i<n_tokens*value_dim;i++){float d=xb2_gpu[i]-xb2_cpu[i];if(d<0)d=-d;if(d>maxd)maxd=d;}
                    double gr=0,cr=0;
                    for(int i=0;i<n_tokens*value_dim;i++){gr+=xb2_gpu[i]*xb2_gpu[i];cr+=xb2_cpu[i]*xb2_cpu[i];}
                    fprintf(stderr,"[DBG] l=0 GPU vs CPU rec: max_diff=%.6e gpu_rms=%.6e cpu_rms=%.6e\n",
                        maxd, sqrt(gr/(n_tokens*value_dim)), sqrt(cr/(n_tokens*value_dim)));
                }
                free(ch);free(ah);free(bh);
            }
            free(xb2_gpu);free(xb2_cpu);
        }
    }
    /* COMPARISON: D2H last token of xb2 (recurrence output) with proper xb2_stride */
    if(ok && l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
        picolm_gpu_sync(dev);
        int lt = n_tokens - 1;
        float *xb2_last = (float *)malloc(value_dim * sizeof(float));
        picolm_gpu_memcpy(xb2_last, battn_out + (size_t)lt * xb2_stride, value_dim * sizeof(float), -1, dev);
        double xrms=0; for(int i=0;i<value_dim;i++) xrms+=xb2_last[i]*xb2_last[i]; xrms=sqrt(xrms/value_dim);
        fprintf(stderr,"[CMP l%d] xb2_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms=%.6f\n",
            l, xb2_last[0],xb2_last[1],xb2_last[2],xb2_last[3],xb2_last[4],xb2_last[5],xb2_last[6],xb2_last[7], (float)xrms);
        free(xb2_last);
    }
    if(l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
        /* xb2 per-head dump before gated_norm */
        { size_t lt2=(size_t)(n_tokens-1)*xb2_stride;
            float *a2=malloc(value_dim*sizeof(float));
            for(int hh=0;hh<n_v;hh++){
                float *seg=a2+hh*hvdim;
                picolm_gpu_memcpy(seg,battn_out+lt2+hh*hvdim,hvdim*sizeof(float),-1,dev);
                double h_rms=0; for(int dd=0;dd<hvdim;dd++) h_rms+=seg[dd]*seg[dd];
                if(hh<4||hh>=n_v-2||sqrt(h_rms/hvdim)>0.001){
                    fprintf(stderr,"[CMP l%d] xb2_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                        l, hh,(float)sqrt(h_rms/hvdim),seg[0],seg[1],seg[2],seg[3]);
                }
            }
            free(a2);
        }
    }
    if(ok){float *nw=(float*)gw->ssm_norm_dev[l];
      if(!nw)return 0;
      if(l<=4&&getenv("PICOLM_SSM_VERIFY")){
          float *nwf=(float*)gw->ssm_norm_dev[0];
          float *nwa=malloc(hvdim*sizeof(float));picolm_gpu_memcpy(nwa,nwf,(size_t)hvdim*sizeof(float),-1,dev);
          double nr=0;for(int i=0;i<hvdim;i++)nr+=nwa[i]*nwa[i];
          fprintf(stderr,"[CMP l%d] ssm_norm_w rms=%.6f first8={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f}\n",
              l, (float)sqrt(nr/hvdim),nwa[0],nwa[1],nwa[2],nwa[3],nwa[4],nwa[5],nwa[6],nwa[7]);
          free(nwa);
      }
      ok&=picolm_gpu_ssm_prefill_gated_norm_dev(battn_out,bq,nw,hvdim,n_v,n_tokens,eps,xb2_stride,xb2_stride,dev);}
    if(l<=4&&getenv("PICOLM_SSM_VERIFY")){
        /* Z-gate (bq) per-head dump */
        { size_t lt2=(size_t)(n_tokens-1)*xb2_stride;
            float *a2=malloc(value_dim*sizeof(float));
            for(int hh=0;hh<n_v;hh++){
                float *seg=a2+hh*hvdim;
                picolm_gpu_memcpy(seg,bq+lt2+hh*hvdim,hvdim*sizeof(float),-1,dev);
                double h_rms=0;float hmax=0; for(int dd=0;dd<hvdim;dd++){h_rms+=seg[dd]*seg[dd];float av=seg[dd];if(av<0)av=-av;if(av>hmax)hmax=av;}
                if(hh<4||hh>=n_v-2||sqrt(h_rms/hvdim)>0.01){
                    fprintf(stderr,"[CMP l%d] z_h%d rms=%.6f maxabs=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                        l, hh,(float)sqrt(h_rms/hvdim),hmax,seg[0],seg[1],seg[2],seg[3]);
                }
            }
            free(a2);
        }
    }
    if(ok) ok&=picolm_gpu_sync(dev);
    SSM_DBG_SYNC;
      _DBG_TOK("gn_out",battn_out);
    if(l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
        picolm_gpu_sync(dev);
        float gb[8];picolm_gpu_memcpy(gb,battn_out+(size_t)(n_tokens-1)*xb2_stride,32,-1,dev);
        double gr=0;for(int _i=0;_i<8;_i++)gr+=gb[_i]*gb[_i];
        fprintf(stderr,"[CMP l%d] gn_out_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,gb[0],gb[1],gb[2],gb[3],gb[4],gb[5],gb[6],gb[7],sqrt(gr/8));
    }
    /* Output projection: reuse bffn_norm as temp, then residual add into bx */
    if(ok){
      /* Head-permute battn_out to GGUF column order before ssm_out matmul.
       * CPU does: fo_gguf[gh] = xb2_natural[h]  (scatter, forward map)
       * GPU kernel does: dst[h] = src[map[h]]  (gather)
       * To match: need dst[h] = src[inv_gguf[h]] so that dst[gh] = src[h].
       * i.e. use the INVERSE map so the gather reproduces the scatter. */
      if(do_remap && gw->ssm_head_invmap_dev)
        ok&=picolm_gpu_ssm_head_permute_batch_dev(bxb,battn_out,(const int*)gw->ssm_head_invmap_dev,hvdim,n_v,n_tokens,xb2_stride,value_dim,dev);
      else if(do_remap && gw->ssm_head_map_dev)
        ok&=picolm_gpu_ssm_head_permute_batch_dev(bxb,battn_out,(const int*)gw->ssm_head_map_dev,hvdim,n_v,n_tokens,xb2_stride,value_dim,dev);
      else
        ok&=picolm_gpu_ssm_head_permute_batch_dev(bxb,battn_out,NULL,hvdim,n_v,n_tokens,xb2_stride,value_dim,dev);
      SSM_DBG_SYNC;
      if(!ok) return 0;
      /* ssm_out matmul: read bxb packed at value_dim, write bffn_norm packed at dim */
      if(l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
          picolm_gpu_sync(dev);
          float hb[8];picolm_gpu_memcpy(hb,bxb+(size_t)(n_tokens-1)*value_dim,32,-1,dev);
          double hr=0;for(int _i=0;_i<8;_i++)hr+=hb[_i]*hb[_i];
          fprintf(stderr,"[CMP l%d] hperm_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,hb[0],hb[1],hb[2],hb[3],hb[4],hb[5],hb[6],hb[7],sqrt(hr/8));
          /* Full bxb last-token dump: per-head RMS */
          { size_t lt_off=(size_t)(n_tokens-1)*value_dim;
            int n_vh=value_dim; // = n_v * hvdim = 48*128 = 6144
            float *hbuf=malloc(n_vh*sizeof(float));
            picolm_gpu_memcpy(hbuf,bxb+lt_off,(size_t)n_vh*sizeof(float),-1,dev);
            double frms=0; for(int fi=0;fi<n_vh;fi++) frms+=hbuf[fi]*hbuf[fi];
            fprintf(stderr,"[CMP l%d] bxb_full_rms=%.6f n_vh=%d\n",l,sqrt(frms/n_vh),n_vh);
            for(int hh=0;hh<n_v;hh++){
                double h_rms=0; for(int dd=0;dd<hvdim;dd++) h_rms+=hbuf[hh*hvdim+dd]*hbuf[hh*hvdim+dd];
                if(hh<4||hh>=n_v-2||sqrt(h_rms/hvdim)>0.01){
                    fprintf(stderr,"[CMP l%d] bxb_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                        l,hh,(float)sqrt(h_rms/hvdim),hbuf[hh*hvdim],hbuf[hh*hvdim+1],hbuf[hh*hvdim+2],hbuf[hh*hvdim+3]);
                }
            }
            free(hbuf);
          }
          /* battn_out per-head dump (same token, same stride) for comparison */
          { size_t lt2=(size_t)(n_tokens-1)*xb2_stride;
            float *a2=malloc(value_dim*sizeof(float));
            // Read head-by-head from battn_out at xb2_stride offsets
            for(int hh=0;hh<n_v;hh++){
                float *seg=a2+hh*hvdim;
                picolm_gpu_memcpy(seg,battn_out+lt2+hh*hvdim,hvdim*sizeof(float),-1,dev);
                double h_rms=0; for(int dd=0;dd<hvdim;dd++) h_rms+=seg[dd]*seg[dd];
                if(hh<4||hh>=n_v-2||sqrt(h_rms/hvdim)>0.01){
                    fprintf(stderr,"[CMP l%d] attn_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                        l,hh,(float)sqrt(h_rms/hvdim),seg[0],seg[1],seg[2],seg[3]);
                }
            }
            free(a2);
          }
      }
      if(l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
          /* Compare full bxb last token - find outlier elements */
          { size_t lt_off=(size_t)(n_tokens-1)*value_dim;
            float *hbuf=malloc(value_dim*sizeof(float));
            picolm_gpu_sync(dev);
            picolm_gpu_memcpy(hbuf,bxb+lt_off,(size_t)value_dim*sizeof(float),-1,dev);
            int maxidx=0; float maxab=0;
            for(int vi=0;vi<value_dim;vi++){float av=hbuf[vi];if(av<0)av=-av;if(av>maxab){maxab=av;maxidx=vi;}}
            fprintf(stderr,"[CMP l%d] bxb_maxabs=%.6f at idx=%d (head=%d off=%d)\n",l,maxab,maxidx,maxidx/hvdim,maxidx%hvdim);
            int s=maxidx-4;if(s<0)s=0;
            fprintf(stderr,"[CMP l%d] bxb[%d..%d]={",l,s,s+8);
            for(int vi=s;vi<s+8&&vi<value_dim;vi++)fprintf(stderr,"%s%.6f",vi>s?",":"",hbuf[vi]);
            fprintf(stderr,"}\n");
            /* Count elements above threshold */
            int nbig=0; for(int vi=0;vi<value_dim;vi++){float av=hbuf[vi];if(av<0)av=-av;if(av>0.01)nbig++;}
            fprintf(stderr,"[CMP l%d] bxb nbig(>0.01)=%d/%d\n",l,nbig,value_dim);
            /* Now dump battn_out at the SAME bxb index to see if the source had the outlier */
            { int ohead = maxidx / hvdim;
              int ooff = maxidx % hvdim;
              /* The bxb index maxidx came from head ohead in the permuted output.
               * The head_permute uses invmap: bxb[ohead] = battn_out[invmap[ohead]]
               * So the source head is invmap[ohead] */
              int invmap[48];
              int n_vpk = n_v / n_k;
              for(int hh=0;hh<n_v;hh++) invmap[hh] = qwen35_vhead_natural(hh, n_vpk, n_k);
              int src_head = invmap[ohead];
              size_t soff = (size_t)(n_tokens-1)*xb2_stride + (size_t)src_head*hvdim + ooff;
              float srcv; picolm_gpu_memcpy(&srcv, battn_out+soff, sizeof(float), 0, dev);
              fprintf(stderr,"[CMP l%d] battn_out src_head=%d off=%d val=%.6f (dst_head=%d val=%.6f)\n",
                  l, src_head, ooff, srcv, ohead, hbuf[maxidx]);
            }
            free(hbuf);
          }
      }
      ok&=picolm_gpu_matmul_dev((picolm_gpu_tensor_t*)gl->ssm_out,bffn_norm,bxb,n_tokens,dev,xb2_stride,0);
    SSM_DBG_SYNC;
      if(l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
          picolm_gpu_sync(dev);
          float so[8]; picolm_gpu_memcpy(so,bffn_norm+(size_t)(n_tokens-1)*xb2_stride,32,-1,dev);
          double srms=0;for(int _i=0;_i<8;_i++)srms+=so[_i]*so[_i];
          fprintf(stderr,"[CMP l%d] ssm_out_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,so[0],so[1],so[2],so[3],so[4],so[5],so[6],so[7],sqrt(srms/8));
      }
      ok&=picolm_gpu_residual_add(bx,bx,bffn_norm,n_tokens,dim,xb2_stride,dev);}
    SSM_DBG_SYNC;
    if(l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
        picolm_gpu_sync(dev);
        float rb[8];picolm_gpu_memcpy(rb,bx+(size_t)(n_tokens-1)*xb2_stride,32,-1,dev);
        double rr=0;for(int _i=0;_i<8;_i++)rr+=rb[_i]*rb[_i];
        fprintf(stderr,"[CMP l%d] resid_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,rb[0],rb[1],rb[2],rb[3],rb[4],rb[5],rb[6],rb[7],sqrt(rr/8));
    }
    /* FFN */
    if(ok&&lw->ffn_gate&&lw->ffn_up&&lw->ffn_down){
      ok&=picolm_gpu_rmsnorm_batched_dev(bffn_norm,bx,(float*)s->post_attn_norm_w[l],dim,eps,n_tokens,xb2_stride,dev);
    SSM_DBG_SYNC;
      ok&=picolm_gpu_expert_mlp_dev((picolm_gpu_tensor_t*)gl->ffn_gate,(picolm_gpu_tensor_t*)gl->ffn_up,(picolm_gpu_tensor_t*)gl->ffn_down,battn_out,bffn_norm,n_tokens,dim,xb2_stride,dev);
    SSM_DBG_SYNC;
      ok&=picolm_gpu_residual_add(bx,bx,battn_out,n_tokens,dim,xb2_stride,dev);}
    SSM_DBG_SYNC;
    if (l == 0 && ok && getenv("PICOLM_SSM_VERIFY")) {
        float fi[8];
        picolm_gpu_sync(dev);
        picolm_gpu_memcpy(fi, bffn_norm + (size_t)(n_tokens-1)*dim, sizeof(fi), -1, dev);
        double firms=0;for(int _i=0;_i<8;_i++)firms+=fi[_i]*fi[_i];
        fprintf(stderr,"[CMP l%d] ffn_in_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",
            l, fi[0],fi[1],fi[2],fi[3],fi[4],fi[5],fi[6],fi[7],sqrt(firms/8));
        float ff[8];
        picolm_gpu_sync(dev);
        picolm_gpu_memcpy(ff, battn_out + (size_t)(n_tokens-1)*xb2_stride, sizeof(ff), -1, dev);
        double frms=0;for(int _i=0;_i<8;_i++)frms+=ff[_i]*ff[_i];
        fprintf(stderr,"[CMP l%d] ffn_out_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",
            l, ff[0],ff[1],ff[2],ff[3],ff[4],ff[5],ff[6],ff[7],sqrt(frms/8));
    }
    /* Dump bx_last for bit-exact GPU vs CPU comparison */
    if (l == 0 && ok) {
        picolm_gpu_sync(dev);
        float bx_v[100];
        picolm_gpu_memcpy(bx_v, bx + (size_t)(n_tokens-1)*xb2_stride, 400, -1, dev);
        float bx_rms=0;
        for (int i=0;i<100;i++) bx_rms += bx_v[i]*bx_v[i];
        bx_rms = sqrtf(bx_rms/100);
        fprintf(stderr, "[GPU l%d] bx_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms100=%.6f\n",
            l, bx_v[0],bx_v[1],bx_v[2],bx_v[3],bx_v[4],bx_v[5],bx_v[6],bx_v[7], bx_rms);
    }
    return ok;
#else
    (void)m;(void)s;(void)bx;(void)bxb;(void)bq;(void)battn_out;(void)bffn_norm;
    (void)bgate;(void)bup;(void)lw;(void)l;(void)n_tokens;(void)start_pos;(void)dev;
    return 0;
#endif
}

/* ================================================================
 * Batched SSM prefill layer.

 *
 * All projection matmuls batched across tokens (weights read once).
 * Convolution, alpha/beta projections also batched.
 * State recurrence remains sequential per token but uses threaded
 * ssm_head_task across v-heads within each token.
 * ================================================================ */
static void ssm_prefill_layer(model_t *m, run_state_t *s,
                              float *x_batch, float *xb_batch, float *xb2_batch,
                              float *hb_batch, float *hb2_batch,
                              layer_weights_t *lw, int l,
                              int n_tokens, int start_pos, int xb2_stride,
                              void **gpu_lw) {
    int bi;
    (void)start_pos;
    (void)gpu_lw;
    (void)xb_batch; /* not used: SSM layer uses local ssm_xb buffer */
    model_config_t *c = &m->config;
    int dim = c->n_embd;
    int d_state = c->ssm_d_state;
    int n_k_heads = c->ssm_n_group;
    int n_v_heads = c->ssm_dt_rank;
    int conv_dim = 2 * d_state * n_k_heads + c->ssm_d_inner;
    int head_v_dim = c->ssm_d_inner / n_v_heads;
    float eps = c->rms_norm_eps;

    int n_k = c->ssm_n_group;
    int n_vpk = n_v_heads / n_k;
    int half_vpk = n_vpk / 2;
    int do_remap = !m->from_safetensors && n_k > 0 && n_k < n_v_heads && half_vpk > 0;
    int value_dim = c->ssm_d_inner;
    int qk_dim = d_state * n_k_heads;
    int repeat = n_v_heads / n_k_heads;

    float *conv_state = s->ssm_conv_state[l];
    float *state = s->ssm_state[l];
    float *conv1d_w = s->ssm_conv1d_w[l];
    int n_state_rows = c->ssm_d_conv - 1;
    int state_stride = conv_dim;
    int ri = 2 + l * 9;

    /* Allocate per-token scratch: xb (RMSNorm'd input) + qkv + z + conv_out */
    size_t per_tok = (size_t)(dim + conv_dim + value_dim + conv_dim);
    float *ssm_buf = (float *)malloc((size_t)n_tokens * per_tok * sizeof(float));
    if (!ssm_buf) { fprintf(stderr, "OOM: SSM prefill scratch\n"); exit(1); }
    float *ssm_xb = ssm_buf;              /* [n_tokens][dim] - RMSNorm'd input */
    float *qkv_batch = ssm_xb + (size_t)n_tokens * dim;
    float *z_batch = qkv_batch + (size_t)n_tokens * conv_dim;
    float *conv_batch = z_batch + (size_t)n_tokens * value_dim;

    /* 1. Batched RMSNorm (write to local ssm_xb with stride dim) */
    for (bi = 0; bi < n_tokens; bi++)
        rmsnorm(ssm_xb + bi * dim, x_batch + bi * dim, s->attn_norm_w[l], dim, eps);

    /* 2. Batched QKV projection (read from ssm_xb with stride dim) */
    tensor_set_repacked(m->repack_used[ri] ? m->repack_buffers[ri] : NULL);
#ifdef PICOLM_GPU
    /* gpu_lw is gpu_layer_weights_t[] (an array of STRUCTS, ~80B each), passed
     * as void**. Index with struct stride, NOT pointer stride -- the old
     * `gpu_lw[l]` read a field value (a heap handle) as a struct pointer and
     * crashed on gl->attn_qkv. m->gpu is zeroed in model_load, so any tensor
     * not uploaded for this layer is NULL and matmul_batch falls back to CPU. */
    if (gpu_lw) {
        gpu_layer_weights_t *gl = &((gpu_layer_weights_t *)gpu_lw)[l];
        tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->attn_qkv, m->gpu.device);
    }
#endif
    if (l <= 1 && n_tokens > 0) {
        float vd = vec_dot(lw->attn_qkv, ssm_xb, dim, lw->type_attn_qkv);
        /* Check activation values */
        float amax = 0, amin = 0;
        int isnan_cnt = 0;
        for (int j = 0; j < dim; j++) {
            float ax = ssm_xb[j];
            if (isnan(ax)) { isnan_cnt++; continue; }
            if (ax > amax) amax = ax;
            if (ax < amin) amin = ax;
        }
        }
    matmul_batch(qkv_batch, ssm_xb, n_tokens, lw->attn_qkv, dim, conv_dim, lw->type_attn_qkv);
    tensor_set_repacked(NULL);
    if (l == 0) {
        float qkv_rms=0; float qkv_v[4];
        for (int i=0;i<4;i++) qkv_v[i]=qkv_batch[(n_tokens-1)*conv_dim+i];
        for (int i=0;i<4;i++) qkv_rms += qkv_v[i]*qkv_v[i];
        qkv_rms = sqrtf(qkv_rms/4);
        fprintf(stderr, "[CPU l%d] qkv rms_last=%.6f {%.6f %.6f %.6f %.6f}\n", l, qkv_rms, qkv_v[0],qkv_v[1],qkv_v[2],qkv_v[3]);
        float *cw = s->ssm_conv1d_w[l];
        fprintf(stderr, "[CPU l%d] conv1d_w[0][:4]={%.6f,%.6f,%.6f,%.6f}\n", l, cw[0],cw[1],cw[2],cw[3]);
    }
#ifdef PICOLM_GPU
    if (gpu_lw) {
        gpu_layer_weights_t *gl = &((gpu_layer_weights_t *)gpu_lw)[l];
        tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->attn_gate_ssm, m->gpu.device);
    }
#endif

    /* 3. Batched Z gate projection */
    tensor_set_repacked(m->repack_used[ri+1] ? m->repack_buffers[ri+1] : NULL);
    matmul_batch(z_batch, ssm_xb, n_tokens, lw->attn_gate_ssm, dim, value_dim, lw->type_attn_gate_ssm);
    tensor_set_repacked(NULL);
#ifdef PICOLM_GPU
    if (gpu_lw) tensor_set_gpu_tensor(NULL, 0);
#endif
    if (do_remap) {
        /* Single temp buffer for reordering v-heads (reused each iteration).
         * Cannot use alloca inside the loop: that grows stack without shrinking,
         * causing stack overflow for large n_tokens (e.g. 613 * 24KB > 8MB). */
        float *zt = malloc(value_dim * sizeof(float));
        if (!zt) { fprintf(stderr, "OOM: zt remap buffer\n"); exit(1); }
        for (bi = 0; bi < n_tokens; bi++) {
            float *zb = z_batch + bi * value_dim;
            memcpy(zt, zb, value_dim * sizeof(float));
            for (int h = 0; h < n_v_heads; h++) {
                int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
                memcpy(zb + h * head_v_dim, zt + gh * head_v_dim, head_v_dim * sizeof(float));
            }
        }
        free(zt);
    }

    /* 4. Convolution + silu (sequential per token: conv_state is stateful)
     * Each token sees a different conv_state because we shift after each token. */
    int conv1d_gpu_done = 0;
#ifdef PICOLM_GPU
    if (gpu_lw) {
        conv1d_gpu_done = picolm_gpu_ssm_conv1d_batch(conv_batch, conv_state, qkv_batch,
                                                        conv1d_w, conv_dim, c->ssm_d_conv,
                                                        n_tokens, m->gpu.device);
    }
#endif
    if (!conv1d_gpu_done) {
        for (bi = 0; bi < n_tokens; bi++) {
            float *qkv = qkv_batch + bi * conv_dim;
            float *conv_out = conv_batch + bi * conv_dim;
            for (int co = 0; co < conv_dim; co++) {
                float sum = 0.0f;
                for (int d = 0; d < n_state_rows; d++)
                    sum += conv1d_w[d + co * c->ssm_d_conv] * conv_state[d * state_stride + co];
                sum += conv1d_w[(c->ssm_d_conv - 1) + co * c->ssm_d_conv] * qkv[co];
                float v = sum;
                conv_out[co] = v * (1.0f / (1.0f + expf(-v)));
            }
            /* Shift conv_state and append new token */
            for (int r = 0; r < n_state_rows - 1; r++)
                memcpy(conv_state + r * state_stride, conv_state + (r + 1) * state_stride, state_stride * sizeof(float));
            memcpy(conv_state + (n_state_rows - 1) * state_stride, qkv, state_stride * sizeof(float));
        }
    }

    /* 5. Split Q/K/V + L2 norm + Q scale (batched across tokens) */
    if (do_remap) {
        /* Single temp buffer for V reordering (reused each iteration). */
        float *vt = malloc(value_dim * sizeof(float));
        if (!vt) { fprintf(stderr, "OOM: vt remap buffer\n"); exit(1); }
        for (bi = 0; bi < n_tokens; bi++) {
            float *conv = conv_batch + bi * conv_dim;
            memcpy(vt, conv + 2*qk_dim, value_dim * sizeof(float));
            for (int h = 0; h < n_v_heads; h++) {
                int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
                memcpy(conv + 2*qk_dim + h * head_v_dim, vt + gh * head_v_dim, head_v_dim * sizeof(float));
            }
        }
        free(vt);
    }
    float q_scale = 1.0f / sqrtf((float)d_state);
    int l2norm_gpu_done = 0;
#ifdef PICOLM_GPU
    if (gpu_lw) {
        l2norm_gpu_done =
            picolm_gpu_ssm_l2norm_batch(conv_batch, d_state, n_k_heads, n_tokens,
                                         conv_dim, 1e-12f, q_scale, m->gpu.device) &&
            picolm_gpu_ssm_l2norm_batch(conv_batch + qk_dim, d_state, n_k_heads, n_tokens,
                                         conv_dim, 1e-12f, 1.0f, m->gpu.device);
    }
#endif
    if (!l2norm_gpu_done) {
        for (bi = 0; bi < n_tokens; bi++) {
            float *conv = conv_batch + bi * conv_dim;
            float *q = conv;
            float *k = conv + qk_dim;
            for (int h = 0; h < n_k_heads; h++) {
                float *qh = q + h * d_state;
                float nrm = 0.0f;
                for (int d = 0; d < d_state; d++) nrm += qh[d] * qh[d];
                nrm = 1.0f / sqrtf(nrm + 1e-12f) * q_scale;
                for (int d = 0; d < d_state; d++) qh[d] *= nrm;
            }
            for (int h = 0; h < n_k_heads; h++) {
                float *kh = k + h * d_state;
                float nrm = 0.0f;
                for (int d = 0; d < d_state; d++) nrm += kh[d] * kh[d];
                nrm = 1.0f / sqrtf(nrm + 1e-12f);
                for (int d = 0; d < d_state; d++) kh[d] *= nrm;
            }
        }
    }

    /* 6. Batched alpha + gate_exp + beta projections.
     * GGUF stores [dim, n_v_heads] column-major per head, with possible
     * v-head reordering. Each head is a vec_dot of dim elements.
     * We process all tokens and all heads in batched fashion. */
    /* Phase 1.3: alpha/beta stored in pooled ssm_buf instead of separate mallocs */
    float *alpha_batch = (float *)malloc((size_t)n_tokens * n_v_heads * sizeof(float));
    float *beta_batch = (float *)malloc((size_t)n_tokens * n_v_heads * sizeof(float));
    {
        gguf_type_t alpha_type = lw->type_ssm_alpha;
        gguf_type_t beta_type = lw->type_ssm_beta;
        size_t row_bytes_alpha = gguf_type_row_size(alpha_type, dim);
        size_t row_bytes_beta = gguf_type_row_size(beta_type, dim);

        /* Precompute head maps */
        int alpha_map[256];
        int beta_map[256];
        for (int h = 0; h < n_v_heads; h++) {
            alpha_map[h] = do_remap ? qwen35_vhead_gguf(h, n_vpk, n_k) : h;
            beta_map[h] = do_remap ? qwen35_vhead_gguf(h, n_vpk, n_k) : h;
        }

        int vecdot_gpu_done = 0;
#ifdef PICOLM_GPU
        /* Type-guarded exactly like ssm_forward_gpu's alpha/beta steps --
         * picolm_gpu_ssm_vecdot_batch only implements F32/Q4_0/Q8_0.
         * Decided jointly for alpha+beta (rather than independently) to
         * avoid splitting the shared Q8_0-quantize-once-per-token setup
         * the CPU fallback below uses for both. */
        if (gpu_lw &&
            (alpha_type == GGUF_TYPE_F32 || alpha_type == GGUF_TYPE_Q4_0 || alpha_type == GGUF_TYPE_Q8_0) &&
            (beta_type == GGUF_TYPE_F32 || beta_type == GGUF_TYPE_Q4_0 || beta_type == GGUF_TYPE_Q8_0)) {
            vecdot_gpu_done =
                picolm_gpu_ssm_vecdot_batch(alpha_batch, ssm_xb, lw->ssm_alpha, alpha_type,
                                             dim, n_v_heads, n_tokens, (int)row_bytes_alpha,
                                             do_remap ? alpha_map : NULL, m->gpu.device) &&
                picolm_gpu_ssm_vecdot_batch(beta_batch, ssm_xb, lw->ssm_beta, beta_type,
                                             dim, n_v_heads, n_tokens, (int)row_bytes_beta,
                                             do_remap ? beta_map : NULL, m->gpu.device);
            if (vecdot_gpu_done) {
                /* GPU vecdot doesn't fuse the dt_w bias (matches the
                 * un-fused contract of every other GPU vecdot call in
                 * this file) -- add it here, same as the CPU path's
                 * `al[h] = sum + s->ssm_dt_w[l][h]` below. */
                for (bi = 0; bi < n_tokens; bi++) {
                    float *al = alpha_batch + bi * n_v_heads;
                    for (int h = 0; h < n_v_heads; h++) al[h] += s->ssm_dt_w[l][h];
                }
            }
        }
#endif
        if (!vecdot_gpu_done) {
        /* Quantize all xb tokens to Q8_0 once for fast vec_dot */
        int nb_xb = dim / 32;
        uint8_t *xb_q8_batch = (uint8_t *)malloc((size_t)n_tokens * nb_xb * 34);
        float *xb_q8_d_batch = (float *)malloc((size_t)n_tokens * nb_xb * sizeof(float));
        for (bi = 0; bi < n_tokens; bi++) {
            quantize_row_q8_0(ssm_xb + bi * dim, xb_q8_batch + bi * nb_xb * 34, dim);
            const block_q8_0 *xqb = (const block_q8_0 *)(xb_q8_batch + bi * nb_xb * 34);
            for (int k = 0; k < nb_xb; k++) {
                xb_q8_d_batch[bi * nb_xb + k] = fp16_to_fp32_lookup(xqb[k].d);
            }
        }

        /* Alpha: per-token, per-head vec_dot with proper GGUF head indexing */
        for (bi = 0; bi < n_tokens; bi++) {
            const void *xb_q8 = (const void *)(xb_q8_batch + bi * nb_xb * 34);
            const float *xb_q8_d = xb_q8_d_batch + bi * nb_xb;
            float *al = alpha_batch + bi * n_v_heads;
            const uint8_t *alpha_w = (const uint8_t *)lw->ssm_alpha;
            for (int h = 0; h < n_v_heads; h++) {
                int gh = alpha_map[h];
                const uint8_t *head_data = alpha_w + (size_t)gh * row_bytes_alpha;
                float sum;
                if (alpha_type == GGUF_TYPE_Q8_0) sum = vec_dot_q8_0_q8_0_deltas(xb_q8, xb_q8_d, head_data, dim);
                else if (alpha_type == GGUF_TYPE_Q4_0) sum = vec_dot_q4_0_q8_0(head_data, xb_q8, dim);
                else sum = vec_dot(head_data, ssm_xb + bi * dim, dim, alpha_type);
                al[h] = sum + s->ssm_dt_w[l][h];
            }
            /* Beta */
            const uint8_t *beta_w = (const uint8_t *)lw->ssm_beta;
            float *bt = beta_batch + bi * n_v_heads;
            for (int h = 0; h < n_v_heads; h++) {
                int gh = beta_map[h];
                const uint8_t *head_data = beta_w + (size_t)gh * row_bytes_beta;
                float sum;
                if (beta_type == GGUF_TYPE_Q8_0) sum = vec_dot_q8_0_q8_0_deltas(xb_q8, xb_q8_d, head_data, dim);
                else if (beta_type == GGUF_TYPE_Q4_0) sum = vec_dot_q4_0_q8_0(head_data, xb_q8, dim);
                else sum = vec_dot(head_data, ssm_xb + bi * dim, dim, beta_type);
                bt[h] = sum;
            }
        }
        free(xb_q8_batch);
        free(xb_q8_d_batch);
        /* alpha_map and beta_map are stack-allocated */
        }

        /* Post-process: alpha -> softplus -> gate_log; beta -> sigmoid */
        /* Store gate_log = softplus(alpha + dt_w) * a_w directly in alpha_batch.
         * The chunked recurrence uses gate_log to build cum_g and decay_mask.
         * Previously we did expf(gate) here and logf(expf(gate)) in the recurrence --
         * a useless exp/log round-trip that compounds floating-point error. */
        for (bi = 0; bi < n_tokens; bi++) {
            float *al = alpha_batch + bi * n_v_heads;
            float *bt = beta_batch + bi * n_v_heads;
            for (int h = 0; h < n_v_heads; h++) {
                float a = al[h];
                float sp = (a > 20.0f) ? a : (a < -20.0f) ? expf(a) : logf(1.0f + expf(a));
                al[h] = sp * s->ssm_a_w[l][h];
                bt[h] = 1.0f / (1.0f + expf(-bt[h]));
            }
        }
    }

    /* 7. Chunked state recurrence (Phase 2).
     * Replaces sequential per-token recurrence with chunked DeltaNet.
     * Processes tokens in chunks of CS=64, using triangular matrix
    * operations within each chunk. Each v-head is fully independent
    * and parallelized via tensor_parallel_for.
    * For single-token inputs, falls back to the standard per-token path. */
    {
        if (n_tokens > 1) {
            int chunked_gpu_done = 0;
#ifdef PICOLM_SSM_VERIFY
            /* Save state+xb2 before GPU modifies them */
            size_t xb2_sz = (size_t)n_tokens * value_dim * sizeof(float);
            size_t st_sz = (size_t)n_v_heads * d_state * d_state * sizeof(float);
            float *xb2_pre = malloc(xb2_sz);
            float *st_pre = malloc(st_sz);
            if (xb2_pre) memcpy(xb2_pre, xb2_batch, xb2_sz);
            if (st_pre) memcpy(st_pre, state, st_sz);
#endif
#ifdef PICOLM_GPU
            if (gpu_lw) {
                chunked_gpu_done = picolm_gpu_ssm_chunked_recurrence(
                    conv_batch, alpha_batch, beta_batch,
                    state, xb2_batch,
                    n_tokens, value_dim,
                    d_state, n_k_heads, n_v_heads, head_v_dim, repeat,
                    conv_dim, m->ssm_chunk_size, m->gpu.device);
            }
#endif
            if (!chunked_gpu_done) {
                ssm_chunked_recurrence(
                    conv_batch, alpha_batch, beta_batch,
                    state, xb2_batch,
                    n_tokens, value_dim,
                    d_state, n_k_heads, n_v_heads, head_v_dim, repeat,
                    conv_dim, m->ssm_chunk_size);
            }
#ifdef PICOLM_SSM_VERIFY
            if (xb2_pre && st_pre && chunked_gpu_done) {
                /* GPU ran via host-facing wrapper. Restore state, run CPU, compare.
                 * Only verify first SSM layer to keep overhead low. */
                static int verify_cnt = 0;
                if (++verify_cnt <= 3) {
                    float *xb2_cpu = malloc(xb2_sz);
                    if (xb2_cpu) {
                        memcpy(state, st_pre, st_sz);
                        ssm_chunked_recurrence(
                            conv_batch, alpha_batch, beta_batch,
                            state, xb2_cpu,
                            n_tokens, value_dim,
                            d_state, n_k_heads, n_v_heads, head_v_dim, repeat,
                            conv_dim, m->ssm_chunk_size);
                        float max_diff = 0;
                        for (int i = 0; i < n_tokens * value_dim; i++) {
                            float d = fabsf(xb2_batch[i] - xb2_cpu[i]);
                            if (d > max_diff) max_diff = d;
                        }
                        fprintf(stderr, "SSM_VERIFY l=%d xb2_max_diff=%.6e\n", l, max_diff);
                        if (l == 0) {
                            fprintf(stderr, "SSM_VERIFY l=0 cpu_xb2[:8]={");
                            for (int i = 0; i < 8; i++) fprintf(stderr, "%s%.6f", i ? ", " : "", xb2_cpu[i]);
                            fprintf(stderr, "}\n");
                            fprintf(stderr, "SSM_VERIFY l=0 gpu_xb2[:8]={");
                            for (int i = 0; i < 8; i++) fprintf(stderr, "%s%.6f", i ? ", " : "", xb2_batch[i]);
                            fprintf(stderr, "}\n");
                            /* Dump post-ssm_out bx (residual target) from CPU path */
                            fprintf(stderr, "SSM_VERIFY l=0 cpu_post_out xb2_batch_last[:8]={");
                            int lt = n_tokens - 1;
                            for (int i = 0; i < 8; i++) fprintf(stderr, "%s%.6f", i ? ", " : "", xb2_batch[lt * xb2_stride + i]);
                            fprintf(stderr, "}\n");
                        }
                    }
                    free(xb2_cpu);
                }
            }
            free(xb2_pre);
            free(st_pre);
#endif
        } else {
            /* Single token: use the standard per-token path */
            float *ssm_output = (float *)malloc((size_t)d_state * n_v_heads * sizeof(float));
            ssm_head_ctx_t ssm_ctx;
            ssm_ctx.state = state;
            ssm_ctx.d_state = d_state;
            ssm_ctx.head_v_dim = head_v_dim;
            ssm_ctx.n_v_heads = n_v_heads;
            ssm_ctx.repeat = repeat;
            ssm_ctx.ssm_output = ssm_output;
            ssm_ctx.q_conv = conv_batch;
            ssm_ctx.k_conv = conv_batch + qk_dim;
            ssm_ctx.v_conv = conv_batch + 2 * qk_dim;
            ssm_ctx.gate_exp = alpha_batch;
            ssm_ctx.beta = beta_batch;
            tensor_parallel_for(n_v_heads, ssm_head_task, &ssm_ctx);
            for (int d = 0; d < d_state; d++)
                for (int h = 0; h < n_v_heads; h++)
                    xb2_batch[h * head_v_dim + d] = ssm_output[d * n_v_heads + h];
            free(ssm_output);
        }
    }

    free(alpha_batch);
    free(beta_batch);

    /* 8. Gated normalization (batched across tokens) */
    int gated_norm_gpu_done = 0;
#ifdef PICOLM_GPU
    if (gpu_lw) {
        gated_norm_gpu_done = picolm_gpu_ssm_prefill_gated_norm(
            xb2_batch, z_batch, s->ssm_norm_w[l], head_v_dim, n_v_heads, n_tokens, eps, m->gpu.device);
    }
#endif
    if (!gated_norm_gpu_done) {
        for (bi = 0; bi < n_tokens; bi++) {
            float *ssm_out = xb2_batch + bi * value_dim;
            float *z = z_batch + bi * value_dim;
            float *norm_w = s->ssm_norm_w[l];
            for (int h = 0; h < n_v_heads; h++) {
                float nrm = 0.0f;
                for (int d = 0; d < head_v_dim; d++) {
                    float v = ssm_out[h * head_v_dim + d];
                    nrm += v * v;
                }
                nrm = 1.0f / sqrtf(nrm / (float)head_v_dim + eps);
                for (int d = 0; d < head_v_dim; d++) {
                    float v = ssm_out[h * head_v_dim + d];
                    float zv = z[h * head_v_dim + d];
                    ssm_out[h * head_v_dim + d] = v * nrm * norm_w[d] *
                        zv * (1.0f / (1.0f + expf(-zv)));
                }
            }
        }
    }
    if (l == 0) {
        float gn_rms=0; float gn_v[4];
        for (int i=0;i<4;i++) { gn_v[i]=xb2_batch[(n_tokens-1)*value_dim+i]; gn_rms+=gn_v[i]*gn_v[i]; }
        gn_rms = sqrtf(gn_rms/4);
        fprintf(stderr, "[CPU l%d] gated_norm rms_last=%.6f {%.6f %.6f %.6f %.6f}\n", l, gn_rms, gn_v[0],gn_v[1],gn_v[2],gn_v[3]);
        /* Per-head dump after gated_norm */
        { int lt=n_tokens-1; const float *xb2l=xb2_batch+lt*value_dim;
          const float *nwm=s->ssm_norm_w[l];
          double nw_rms=0;for(int i=0;i<head_v_dim;i++)nw_rms+=nwm[i]*nwm[i];
          fprintf(stderr,"[CPU l0] ssm_norm_w rms=%.6f first8={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f}\n",
              sqrtf(nw_rms/head_v_dim),nwm[0],nwm[1],nwm[2],nwm[3],nwm[4],nwm[5],nwm[6],nwm[7]);
          for(int hh=0;hh<n_v_heads;hh++){
              float h_rms=0; for(int dd=0;dd<head_v_dim;dd++) h_rms+=xb2l[hh*head_v_dim+dd]*xb2l[hh*head_v_dim+dd];
              if(hh<4||hh>=n_v_heads-2||sqrtf(h_rms/head_v_dim)>0.01){
                  fprintf(stderr,"[CPU l0] gn_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                      hh,sqrtf(h_rms/head_v_dim),xb2l[hh*head_v_dim],xb2l[hh*head_v_dim+1],xb2l[hh*head_v_dim+2],xb2l[hh*head_v_dim+3]);
              }
          }
        }
    }

    /* Per-head gated_norm dump for CPU (always, after both GPU-assisted and pure CPU paths) */
    if (l == 0) {
        int lt=n_tokens-1; const float *xb2l=xb2_batch+lt*value_dim;
        const float *nwm=s->ssm_norm_w[l];
        double nw_rms=0;for(int i=0;i<head_v_dim;i++)nw_rms+=nwm[i]*nwm[i];
        fprintf(stderr,"[CPU l0] ssm_norm_w rms=%.6f first8={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f}\n",
            sqrtf(nw_rms/head_v_dim),nwm[0],nwm[1],nwm[2],nwm[3],nwm[4],nwm[5],nwm[6],nwm[7]);
        for(int hh=0;hh<n_v_heads;hh++){
            float h_rms=0; for(int dd=0;dd<head_v_dim;dd++) h_rms+=xb2l[hh*head_v_dim+dd]*xb2l[hh*head_v_dim+dd];
            if(hh<4||hh>=n_v_heads-2||sqrtf(h_rms/head_v_dim)>0.01){
                fprintf(stderr,"[CPU l0] gn_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                    hh,sqrtf(h_rms/head_v_dim),xb2l[hh*head_v_dim],xb2l[hh*head_v_dim+1],xb2l[hh*head_v_dim+2],xb2l[hh*head_v_dim+3]);
            }
        }
    }

    /* CPU xb2 per-head dump for layers 0-4 */
    if (l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
        { int lt2 = n_tokens - 1;
          for (int hh = 0; hh < n_v_heads; hh++) {
              float hr = 0; for (int dd = 0; dd < head_v_dim; dd++) { float v = xb2_batch[lt2 * value_dim + hh * head_v_dim + dd]; hr += v * v; }
              float hrms = sqrtf(hr / head_v_dim);
              if (hh < 4 || hh >= n_v_heads - 2 || hrms > 0.01) {
                  fprintf(stderr, "[CPU l%d] xb2_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                      l, hh, hrms, xb2_batch[lt2*value_dim+hh*head_v_dim],
                      xb2_batch[lt2*value_dim+hh*head_v_dim+1],
                      xb2_batch[lt2*value_dim+hh*head_v_dim+2],
                      xb2_batch[lt2*value_dim+hh*head_v_dim+3]);
              }
          }
          float mx = 0; for (int i = 0; i < value_dim; i++) { float a = xb2_batch[lt2 * value_dim + i]; if (a < 0) a = -a; if (a > mx) mx = a; }
          fprintf(stderr, "[CPU l%d] xb2_maxabs=%.6f\n", l, mx);
        }
    }

    /* 9. Output projection (batched) */
    if (do_remap) {
        /* Reorder ALL tokens to GGUF column order first (cheap CPU
         * pass), then ONE batched GPU matmul across all n_tokens --
         * NOT n_tokens separate single-token matmul() calls. The old
         * per-token loop here was, per profiling, the single largest
         * cost in prefill: each single-token GPU matmul call pays its
         * own full H2D+kernel+D2H (and, via matmul()'s own scratch
         * management, malloc) overhead, so a 36-token prompt paid that
         * fixed per-call overhead 36 times per layer instead of once. */
        float *fo_gguf_batch = (float *)malloc((size_t)n_tokens * value_dim * sizeof(float));
        for (bi = 0; bi < n_tokens; bi++) {
            float *fo = xb2_batch + bi * value_dim;
            float *dst = fo_gguf_batch + bi * value_dim;
            for (int h = 0; h < n_v_heads; h++) {
                int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
                memcpy(dst + gh * head_v_dim, fo + h * head_v_dim, head_v_dim * sizeof(float));
            }
        }
        tensor_set_repacked(m->repack_used[ri+5] ? m->repack_buffers[ri+5] : NULL);
#ifdef PICOLM_GPU
        if (gpu_lw) {
            gpu_layer_weights_t *gl = &((gpu_layer_weights_t *)gpu_lw)[l];
            tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->ssm_out, m->gpu.device);
        }
#endif
        if (l == 0) {
            int lt=n_tokens-1; const float *fb=fo_gguf_batch+lt*value_dim;
            double frms=0; for(int fi=0;fi<value_dim;fi++) frms+=fb[fi]*fb[fi];
            fprintf(stderr,"[CPU l0] fo_gguf_rms=%.6f n_vh=%d\n",sqrtf(frms/value_dim),value_dim);
            for(int hh=0;hh<n_v_heads;hh++){
                float h_rms=0; for(int dd=0;dd<head_v_dim;dd++) h_rms+=fb[hh*head_v_dim+dd]*fb[hh*head_v_dim+dd];
                if(hh<4||hh>=n_v_heads-2||sqrtf(h_rms/head_v_dim)>0.01){
                    fprintf(stderr,"[CPU l0] fo_h%d rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",
                        hh,sqrtf(h_rms/head_v_dim),fb[hh*head_v_dim],fb[hh*head_v_dim+1],fb[hh*head_v_dim+2],fb[hh*head_v_dim+3]);
                }
            }
        }
        if (l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
            /* Dequantize first 64 floats of ssm_out weight on CPU for comparison */
            { float *cw = malloc(64 * sizeof(float));
              if (lw->type_ssm_out == 2) {
                  const unsigned char *q = (const unsigned char *)lw->ssm_out;
                  float sc = *(const float *)(q + 32);
                  for (int i = 0; i < 32; i++) cw[i] = (q[i] - 128) * sc;
                  for (int i = 32; i < 64; i++) {
                      float sc2 = *(const float *)(q + 64 + 32);
                      cw[i] = (q[64 + i - 32] - 128) * sc2;
                  }
              } else if (lw->type_ssm_out == 0) {
                  const float *f = (const float *)lw->ssm_out;
                  for (int i = 0; i < 64; i++) cw[i] = f[i];
              }
              fprintf(stderr,"[CPU l%d] ssm_out_w[0][:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} type=%d I=%d O=%d\n",
                  l, cw[0],cw[1],cw[2],cw[3],cw[4],cw[5],cw[6],cw[7], (int)lw->type_ssm_out, value_dim, dim);
              double wr=0;for(int wi=0;wi<64;wi++)wr+=cw[wi]*cw[wi];
              fprintf(stderr,"[CPU l%d] ssm_out_w rms64=%.6f\n",l,sqrt(wr/64));
              free(cw);
            }
        }
        float *ssm_out_buf = (float *)malloc((size_t)n_tokens * dim * sizeof(float));
        matmul_batch(ssm_out_buf, fo_gguf_batch, n_tokens, lw->ssm_out, value_dim, dim, lw->type_ssm_out);
        for (bi = 0; bi < n_tokens; bi++)
            memcpy(xb2_batch + bi * xb2_stride, ssm_out_buf + bi * dim, dim * sizeof(float));
        free(ssm_out_buf);
        free(fo_gguf_batch);
        if (l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
            float so[8], sor=0; int lt=n_tokens-1;
            for(int i=0;i<8;i++){so[i]=xb2_batch[lt*xb2_stride+i];sor+=so[i]*so[i];}sor=sqrtf(sor/8);
            fprintf(stderr,"[CPU l%d] ssm_out_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,so[0],so[1],so[2],so[3],so[4],so[5],so[6],so[7],sor);
        }
        tensor_set_repacked(NULL);
#ifdef PICOLM_GPU
        if (gpu_lw) tensor_set_gpu_tensor(NULL, 0);
#endif
    } else {
        tensor_set_repacked(m->repack_used[ri+5] ? m->repack_buffers[ri+5] : NULL);
#ifdef PICOLM_GPU
        if (gpu_lw) {
            gpu_layer_weights_t *gl = &((gpu_layer_weights_t *)gpu_lw)[l];
            tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->ssm_out, m->gpu.device);
        }
#endif
        /* Cannot alias in/out when value_dim != dim (strides differ, cross-token corruption) */
        float *ssm_out_buf = (float *)malloc((size_t)n_tokens * dim * sizeof(float));
        matmul_batch(ssm_out_buf, xb2_batch, n_tokens, lw->ssm_out, value_dim, dim, lw->type_ssm_out);
        if (l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
            float so[8], sor=0; int lt=n_tokens-1;
            for(int i=0;i<8;i++){so[i]=ssm_out_buf[lt*dim+i];sor+=so[i]*so[i];}sor=sqrtf(sor/8);
            fprintf(stderr,"[CPU l%d] ssm_out_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,so[0],so[1],so[2],so[3],so[4],so[5],so[6],so[7],sor);
        }
        for (bi = 0; bi < n_tokens; bi++)
            memcpy(xb2_batch + bi * xb2_stride, ssm_out_buf + bi * dim, dim * sizeof(float));
        free(ssm_out_buf);
        tensor_set_repacked(NULL);
#ifdef PICOLM_GPU
        if (gpu_lw) tensor_set_gpu_tensor(NULL, 0);
#endif
    }

    /* 10. Residual add (batched) */
    for (bi = 0; bi < n_tokens; bi++) {
        float *a = x_batch + bi * dim, *b = xb2_batch + bi * xb2_stride;
        for (int d = 0; d < dim; d++) a[d] += b[d];
    }
    if (l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
        float rb[8], rbr=0; int lt=n_tokens-1;
        for(int i=0;i<8;i++){rb[i]=x_batch[lt*dim+i];rbr+=rb[i]*rb[i];}rbr=sqrtf(rbr/8);
        fprintf(stderr,"[CPU l%d] bx_after_resid[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",l,rb[0],rb[1],rb[2],rb[3],rb[4],rb[5],rb[6],rb[7],rbr);
    }

    /* 11. Batched FFN (if present) */
    if (c->has_moe) {
        for (bi = 0; bi < n_tokens; bi++) {
            rmsnorm(ssm_xb + bi * dim, x_batch + bi * dim, s->post_attn_norm_w[l], dim, eps);
        }
        moe_forward_batch(m, s, ssm_xb, xb2_batch, n_tokens, lw);
        for (bi = 0; bi < n_tokens; bi++) {
            float *a = x_batch + bi * dim, *b = xb2_batch + bi * dim;
            for (int d = 0; d < dim; d++) a[d] += b[d];
        }
    } else if (lw->ffn_gate && lw->ffn_up && lw->ffn_down) {
        for (bi = 0; bi < n_tokens; bi++)
            rmsnorm(ssm_xb + bi * dim, x_batch + bi * dim, s->post_attn_norm_w[l], dim, eps);

        /* Fused batched FFN on GPU: y = down(silu(gate(x)) * up(x)) for all
         * n_tokens in one call (one H2D, one D2H total). Previously this
         * branch always used matmul_dual_batch for gate+up, which has NO
         * GPU dispatch implemented at all (unlike matmul_batch, it never
         * checks the gpu_tensor global) -- so SSM-layer FFN during prefill
         * was 100% CPU no matter what. */
        int ffn_gpu_done = 0;
#ifdef PICOLM_GPU
        if (gpu_lw) {
            gpu_layer_weights_t *gl = &((gpu_layer_weights_t *)gpu_lw)[l];
            if (gl->ffn_gate && gl->ffn_up && gl->ffn_down) {
                ffn_gpu_done = picolm_gpu_expert_mlp(
                    (picolm_gpu_tensor_t *)gl->ffn_gate,
                    (picolm_gpu_tensor_t *)gl->ffn_up,
                    (picolm_gpu_tensor_t *)gl->ffn_down,
                    xb2_batch, ssm_xb, n_tokens);
            }
        }
#endif
        if (!ffn_gpu_done) {
            tensor_set_repacked(m->repack_used[ri+7] ? m->repack_buffers[ri+7] : NULL);
            matmul_dual_batch(hb_batch, hb2_batch, ssm_xb, n_tokens,
                              lw->ffn_gate, lw->ffn_up, dim, c->n_ffn,
                              lw->type_ffn_gate, lw->type_ffn_up);
            tensor_set_repacked(NULL);

            for (bi = 0; bi < n_tokens; bi++) {
                silu(hb_batch + bi * c->n_ffn, c->n_ffn);
                elemwise_mul(hb_batch + bi * c->n_ffn, hb_batch + bi * c->n_ffn,
                             hb2_batch + bi * c->n_ffn, c->n_ffn);
            }

            tensor_set_repacked(m->repack_used[ri+8] ? m->repack_buffers[ri+8] : NULL);
            matmul_batch(xb2_batch, hb_batch, n_tokens, lw->ffn_down, c->n_ffn, dim, lw->type_ffn_down);
            tensor_set_repacked(NULL);
        }

        for (bi = 0; bi < n_tokens; bi++) {
            float *a = x_batch + bi * dim, *b = xb2_batch + bi * dim;
            for (int d = 0; d < dim; d++) a[d] += b[d];
        }
    }
#ifdef PICOLM_SSM_VERIFY
    if (l <= 4 && getenv("PICOLM_SSM_VERIFY")) {
        int lt = n_tokens - 1;
        fprintf(stderr, "SSM_VERIFY l=0 cpu bx_last[:4]={%.6f,%.6f,%.6f,%.6f}\n",
            x_batch[lt*dim], x_batch[lt*dim+1], x_batch[lt*dim+2], x_batch[lt*dim+3]);
        /* Dump RMS of xb2 (last token) for input comparison */
        { double xrms = 0;
          for(int i = 0; i < value_dim; i++) xrms += xb2_batch[lt*xb2_stride+i]*xb2_batch[lt*xb2_stride+i];
          fprintf(stderr, "SSM_VERIFY l=0 cpu xb2_rms=%.6e\n", sqrt(xrms/value_dim)); }
        /* Dump ssm_out matmul output for last token */
        { float *ssm_out_buf = (float *)malloc((size_t)n_tokens * dim * sizeof(float));
          matmul_batch(ssm_out_buf, xb2_batch, n_tokens, lw->ssm_out, value_dim, dim, lw->type_ssm_out);
          double orm = 0; for(int i=0;i<dim;i++) orm+=ssm_out_buf[lt*dim+i]*ssm_out_buf[lt*dim+i];
          fprintf(stderr, "SSM_VERIFY l=0 cpu ssm_out_rms=%.6e\n", sqrt(orm/dim));
          fprintf(stderr, "SSM_VERIFY l=0 cpu ssm_out_last[:4]={%.6f,%.6f,%.6f,%.6f}\n",
              ssm_out_buf[lt*dim],ssm_out_buf[lt*dim+1],ssm_out_buf[lt*dim+2],ssm_out_buf[lt*dim+3]);
          free(ssm_out_buf); }
    }
#endif
    if (n_tokens > 0 && getenv("PICOLM_SSM_VERIFY")) {
        int lt = n_tokens - 1;
        fprintf(stderr, "[DBG CPU l=%d] bx_last[:4]={%.6f,%.6f,%.6f,%.6f}\n", l, x_batch[lt*dim], x_batch[lt*dim+1], x_batch[lt*dim+2], x_batch[lt*dim+3]);
    }
    free(ssm_buf);
}

/* ================================================================
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
          if(getenv("PICOLM_SSM_VERIFY") && l==3){
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
        if (getenv("PICOLM_SSM_VERIFY") && l == 3) {
            int lt = n_tokens - 1;
            fprintf(stderr, "[DBG CPU attn_gate l=%d] last[:4]={%.6f,%.6f,%.6f,%.6f}\n", l, hb2_batch[lt*q_dim], hb2_batch[lt*q_dim+1], hb2_batch[lt*q_dim+2], hb2_batch[lt*q_dim+3]);
        }
        /* Dump attention output for comparison - BEFORE gate sigmoid */
        if (getenv("PICOLM_SSM_VERIFY") && l == 3) {
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
        if (getenv("PICOLM_SSM_VERIFY") && l == 3) {
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
          if (getenv("PICOLM_SSM_VERIFY") && l == 3) {
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
        if (getenv("PICOLM_SSM_VERIFY") && lw->is_attn_layer) {
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

/* ================================================================
 * KV Cache Persistence: save/load KV state to skip prompt prefill
 *
 * File format v2 (legacy): 7 uint32_t header + K/V data (all layers)
 * File format v3: 9 uint32_t header + K/V data (all layers)
 * File format v4 (SSM-aware, buffered I/O):
 *   13 uint32_t header + layer bitmap + K/V (attn only) + SSM state
 * ================================================================ */

/* Flush buffered write to file descriptor. Returns 0 on success, -1 on error. */
static int _kv_flush(int fd, uint8_t *buf, size_t *buf_off) {
    if (*buf_off == 0) return 0;
    ssize_t w;
#ifdef _WIN32
    w = (ssize_t)_write(fd, (const char *)buf, (unsigned)*buf_off);
#else
    w = write(fd, buf, *buf_off);
#endif
    if (w != (ssize_t)*buf_off) {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return -1;
    }
    *buf_off = 0;
    return 0;
}

/* Ensure at least 'need' bytes available in read buffer. Returns 0 on success.
 * Uses lseek to seek to the correct file offset on refill, since the kernel's
 * file offset may be ahead of what the buffer has consumed. */
static int _kv_ensure(int fd, uint8_t *buf, size_t buf_size,
                      size_t need, size_t *buf_avail, size_t *buf_pos,
                      size_t *file_offset) {
    if (*buf_pos + need <= *buf_avail) return 0;
    /* Seek to where the buffer data started + bytes consumed before this buffer load */
    size_t consume_start = *file_offset;
    size_t consumed_in_buf = *buf_pos;
    size_t next_offset = consume_start + consumed_in_buf;
    if (lseek(fd, (off_t)next_offset, SEEK_SET) < 0) {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return -1;
    }
    ssize_t r;
#ifdef _WIN32
    r = (ssize_t)_read(fd, (char *)buf, (unsigned)buf_size);
#else
    r = read(fd, (void *)buf, buf_size);
#endif
    if (r <= 0) {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return -1;
    }
    *file_offset = next_offset;
    *buf_avail = (size_t)r;
    *buf_pos = 0;
    if (*buf_avail < need) {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return -1;
    }
    return 0;
}

/* Close file descriptor (cross-platform wrapper) */
static void _kv_close(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif
}

int kvcache_save(const model_t *m, const char *path, int n_pos, const int *tokens) {
    const model_config_t *c = &m->config;
    const run_state_t *s = &m->state;

    if (n_pos <= 0 || n_pos > c->max_seq_len) return -1;

#ifdef _WIN32
    int fd = _open(path, _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, 0644);
#else
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
    if (fd < 0) {
        fprintf(stderr, "kvcache_save: cannot open %s\n", path);
        return -1;
    }

    /* 128 KB intermediate buffer for batching I/O */
    size_t buf_size = 1 << 17;
    uint8_t *buf = alloca(buf_size);
    size_t buf_off = 0;

    size_t ssm_sz = model_ssm_snapshot_size(m);

    /* Version 4 header: 13 uint32_t entries */
    uint32_t header[13] = {
        KVCACHE_MAGIC | 0x00000004,
        (uint32_t)n_pos,
        (uint32_t)c->n_layers,
        (uint32_t)c->n_kv_heads,
        (uint32_t)c->head_dim,
        (uint32_t)s->kv_type_k,
        (uint32_t)s->kv_type_v,
        (uint32_t)s->kv_hadamard_k,
        (uint32_t)s->kv_hadamard_v,
        (uint32_t)c->has_ssm,
        (uint32_t)ssm_sz,
        (uint32_t)s->kv_layer_count,
        (uint32_t)n_pos,
    };
    memcpy(buf + buf_off, header, sizeof(header));
    buf_off += sizeof(header);

    /* Token sequence */
    if (tokens && n_pos > 0) {
        size_t tk_sz = (size_t)n_pos * sizeof(uint32_t);
        if (buf_off + tk_sz > buf_size) { if (_kv_flush(fd, buf, &buf_off)) return -1; }
        /* Convert int tokens to uint32_t in buffer */
        uint32_t *tk_buf = (uint32_t *)(buf + buf_off);
        for (int i = 0; i < n_pos; i++) tk_buf[i] = (uint32_t)tokens[i];
        buf_off += tk_sz;
    }

    /* Layer type bitmap */
    if (buf_off + (size_t)c->n_layers > buf_size) { if (_kv_flush(fd, buf, &buf_off)) return -1; }
    memcpy(buf + buf_off, s->kv_layer_map, (size_t)c->n_layers);
    buf_off += (size_t)c->n_layers;

    /* KV cache: only attention layers, using kv_ordinal for offsets */
    size_t pos_stride_k = s->kv_row_size_k;
    size_t pos_stride_v = s->kv_row_size_v;
    for (int l = 0; l < c->n_layers; l++) {
        if (!s->kv_layer_map[l]) continue;
        int ao = s->kv_layer_ordinal[l];
        const uint8_t *kcache_l = s->key_cache + (size_t)ao * c->max_seq_len * pos_stride_k;
        for (int p = 0; p < n_pos; p++) {
            if (buf_off + pos_stride_k > buf_size) { if (_kv_flush(fd, buf, &buf_off)) return -1; }
            memcpy(buf + buf_off, kcache_l + (size_t)p * pos_stride_k, pos_stride_k);
            buf_off += pos_stride_k;
        }
    }
    for (int l = 0; l < c->n_layers; l++) {
        if (!s->kv_layer_map[l]) continue;
        int ao = s->kv_layer_ordinal[l];
        const uint8_t *vcache_l = s->val_cache + (size_t)ao * c->max_seq_len * pos_stride_v;
        for (int p = 0; p < n_pos; p++) {
            if (buf_off + pos_stride_v > buf_size) { if (_kv_flush(fd, buf, &buf_off)) return -1; }
            memcpy(buf + buf_off, vcache_l + (size_t)p * pos_stride_v, pos_stride_v);
            buf_off += pos_stride_v;
        }
    }

    /* SSM state: stream directly to file in per-layer chunks to avoid
     * allocating a massive buffer for 27B+ models.
     * Layout: [conv_state_l0][conv_state_l1]...[state_l0][state_l1]... */
    if (c->has_ssm && ssm_sz > 0) {
        int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;

        /* First pass: save all conv_states */
        for (int l = 0; l < c->n_layers; l++) {
            if (m->weights.layers[l].is_attn_layer) continue;
            if (!s->ssm_conv_state[l]) continue;
            size_t sz = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
            /* Stream in buf_size chunks */
            const uint8_t *src = (const uint8_t *)s->ssm_conv_state[l];
            while (sz > 0) {
                size_t chunk = sz < buf_size ? sz : buf_size;
                if (buf_off + chunk > buf_size) { if (_kv_flush(fd, buf, &buf_off)) return -1; }
                memcpy(buf + buf_off, src, chunk);
                buf_off += chunk;
                src += chunk;
                sz -= chunk;
            }
        }
        /* Second pass: save all ssm_states */
        for (int l = 0; l < c->n_layers; l++) {
            if (m->weights.layers[l].is_attn_layer) continue;
            if (!s->ssm_state[l]) continue;
            size_t sz = (size_t)c->ssm_d_state * c->ssm_d_inner * sizeof(float);
            const uint8_t *src = (const uint8_t *)s->ssm_state[l];
            while (sz > 0) {
                size_t chunk = sz < buf_size ? sz : buf_size;
                if (buf_off + chunk > buf_size) { if (_kv_flush(fd, buf, &buf_off)) return -1; }
                memcpy(buf + buf_off, src, chunk);
                buf_off += chunk;
                src += chunk;
                sz -= chunk;
            }
        }
    }

    if (_kv_flush(fd, buf, &buf_off)) return -1;
    _kv_close(fd);
    fprintf(stderr, "\nKV cache saved: %d positions to %s\n", n_pos, path);
    return 0;
}

int kvcache_load(model_t *m, const char *path, int **tokens_out) {
    *tokens_out = NULL;
    const model_config_t *c = &m->config;
    run_state_t *s = &m->state;


#ifdef _WIN32
    int fd = _open(path, _O_RDONLY | _O_BINARY);
#else
    int fd = open(path, O_RDONLY);
#endif
    if (fd < 0) {
        return 0;
    }

    size_t buf_size = 1 << 17;
    uint8_t *buf = alloca(buf_size);
    size_t buf_avail = 0;
    size_t buf_pos = 0;
    size_t file_offset = 0;  /* tracks file offset of the start of current buffer load */

    uint32_t header[13];
    int version = 4;

    /* Try v4: read 13 entries */
    if (_kv_ensure(fd, buf, buf_size, sizeof(header), &buf_avail, &buf_pos, &file_offset)) {
        return 0;
    }
    memcpy(header, buf + buf_pos, sizeof(header));
    buf_pos += sizeof(header);

    if ((header[0] & 0xFFFFFFF0) != KVCACHE_MAGIC || (header[0] & 0xF) != 4) {
        /* Fall back to v3 (9 entries) */
        if (lseek(fd, 0, SEEK_SET) < 0) { _kv_close(fd); return 0; }
        buf_avail = 0; buf_pos = 0;
        {
            ssize_t r;
#ifdef _WIN32
            r = (ssize_t)_read(fd, (char *)buf, (unsigned)buf_size);
#else
            r = read(fd, (void *)buf, buf_size);
#endif
            if (r > (ssize_t)(sizeof(uint32_t) * 9)) {
                memcpy(header, buf, sizeof(uint32_t) * 9);
                memset(header + 9, 0, 4 * sizeof(uint32_t));
                version = 3;
                buf_pos = sizeof(uint32_t) * 9;
                buf_avail = (size_t)r;
            } else {
                /* Fall back to v2 (7 entries) */
                if (lseek(fd, 0, SEEK_SET) < 0) { _kv_close(fd); return 0; }
                buf_avail = 0; buf_pos = 0;
#ifdef _WIN32
                r = (ssize_t)_read(fd, (char *)buf, (unsigned)buf_size);
#else
                r = read(fd, (void *)buf, buf_size);
#endif
                if (r > (ssize_t)(sizeof(uint32_t) * 7)) {
                    memcpy(header, buf, sizeof(uint32_t) * 7);
                    memset(header + 7, 0, 6 * sizeof(uint32_t));
                    version = 2;
                    buf_pos = sizeof(uint32_t) * 7;
                    buf_avail = (size_t)r;
                } else {
                    _kv_close(fd); return 0;
                }
            }
        }
    }

    if ((header[0] & (uint32_t)~0xF) != KVCACHE_MAGIC) {
        fprintf(stderr, "kvcache_load: invalid magic\n");
        _kv_close(fd); return 0;
    }

    int n_pos = (int)header[1];
    int file_layers = (int)header[2];
    int file_n_kv_heads = (int)header[3];
    int file_head_dim = (int)header[4];
    int has_ssm = (version >= 4) ? (int)header[9] : 0;

    if (file_layers != c->n_layers || file_n_kv_heads != c->n_kv_heads ||
        file_head_dim != c->head_dim) {
        fprintf(stderr, "kvcache_load: model mismatch (layers=%d/%d, kv_heads=%d/%d, head_dim=%d/%d)\n",
                file_layers, c->n_layers, file_n_kv_heads, c->n_kv_heads,
                file_head_dim, c->head_dim);
        _kv_close(fd); return 0;
    }
    if (n_pos > c->max_seq_len) {
        fprintf(stderr, "kvcache_load: cached %d positions exceeds max_seq_len %d\n",
                n_pos, c->max_seq_len);
        _kv_close(fd); return 0;
    }
    if (header[5] != s->kv_type_k || header[6] != s->kv_type_v) {
        fprintf(stderr, "kvcache_load: KV type mismatch (file k=%d v=%d vs current k=%d v=%d)\n",
                header[5], header[6], s->kv_type_k, s->kv_type_v);
        _kv_close(fd); return 0;
    }
    if (header[7] != (uint32_t)s->kv_hadamard_k || header[8] != (uint32_t)s->kv_hadamard_v) {
        fprintf(stderr, "kvcache_load: Hadamard mismatch (file K=%d V=%d vs current K=%d V=%d)\n",
                header[7], header[8], s->kv_hadamard_k, s->kv_hadamard_v);
        _kv_close(fd); return 0;
    }

    /* Token sequence (v4 only) */
    if (version >= 4 && n_pos > 0) {
        size_t tk_sz = (size_t)n_pos * sizeof(uint32_t);
        /* Read tokens: may span multiple buffer reads */
        *tokens_out = malloc(tk_sz);
        if (*tokens_out) {
            uint32_t *tk = (uint32_t *)*tokens_out;
            size_t tk_off = 0;
            while (tk_off < tk_sz) {
                size_t need = tk_sz - tk_off;
                if (need > buf_size) need = buf_size;
                if (_kv_ensure(fd, buf, buf_size, need, &buf_avail, &buf_pos, &file_offset)) {
                    free(*tokens_out);
                    *tokens_out = NULL;
                    return 0;
                }
                size_t copy = need;
                if (copy > tk_sz - tk_off) copy = tk_sz - tk_off;
                memcpy((uint8_t *)tk + tk_off, buf + buf_pos, copy);
                tk_off += copy;
                buf_pos += copy;
            }
            /* Convert uint32_t to int array in place */
            /* Already uint32_t, cast to int* is fine */
        }
    }

    /* Layer type bitmap (v4 only; v2/v3: all layers are attention) */
    uint8_t layer_map[MAX_LAYERS];
    if (version >= 4) {
        if (_kv_ensure(fd, buf, buf_size, (size_t)c->n_layers, &buf_avail, &buf_pos, &file_offset)) {
            return 0;
        }
        memcpy(layer_map, buf + buf_pos, (size_t)c->n_layers);
        buf_pos += (size_t)c->n_layers;
    } else {
        memset(layer_map, 1, (size_t)c->n_layers);
    }

    /* Load KV cache */
    size_t pos_stride_k = s->kv_row_size_k;
    size_t pos_stride_v = s->kv_row_size_v;
    for (int l = 0; l < c->n_layers; l++) {
        int has_kv = (version >= 4) ? layer_map[l] : 1;
        if (!has_kv) continue;
        int ao = (version >= 4) ? s->kv_layer_ordinal[l] : l;
        uint8_t *kcache_l = s->key_cache + (size_t)ao * c->max_seq_len * pos_stride_k;
        for (int p = 0; p < n_pos; p++) {
            if (_kv_ensure(fd, buf, buf_size, pos_stride_k, &buf_avail, &buf_pos, &file_offset)) return 0;
            memcpy(kcache_l + (size_t)p * pos_stride_k, buf + buf_pos, pos_stride_k);
            buf_pos += pos_stride_k;
        }
    }
    for (int l = 0; l < c->n_layers; l++) {
        int has_kv = (version >= 4) ? layer_map[l] : 1;
        if (!has_kv) continue;
        int ao = (version >= 4) ? s->kv_layer_ordinal[l] : l;
        uint8_t *vcache_l = s->val_cache + (size_t)ao * c->max_seq_len * pos_stride_v;
        for (int p = 0; p < n_pos; p++) {
            if (_kv_ensure(fd, buf, buf_size, pos_stride_v, &buf_avail, &buf_pos, &file_offset)) return 0;
            memcpy(vcache_l + (size_t)p * pos_stride_v, buf + buf_pos, pos_stride_v);
            buf_pos += pos_stride_v;
        }
    }

    /* Load SSM state (v4 only): stream directly from file, per-layer */
    if (has_ssm && header[10] > 0) {
        size_t ssm_sz = header[10];
        int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
        size_t bytes_left = ssm_sz;

        /* First pass: restore all conv_states */
        for (int l = 0; l < c->n_layers && bytes_left > 0; l++) {
            if (m->weights.layers[l].is_attn_layer) continue;
            if (!s->ssm_conv_state[l]) continue;
            size_t sz = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
            if (sz > bytes_left) { free(*tokens_out); _kv_close(fd); return 0; }
            uint8_t *dst = (uint8_t *)s->ssm_conv_state[l];
            size_t remain = sz;
            while (remain > 0) {
                size_t need = remain < buf_size ? remain : buf_size;
                if (_kv_ensure(fd, buf, buf_size, need, &buf_avail, &buf_pos, &file_offset)) return 0;
                size_t copy = need < remain ? need : remain;
                memcpy(dst, buf + buf_pos, copy);
                dst += copy;
                buf_pos += copy;
                remain -= copy;
                bytes_left -= copy;
            }
        }
        /* Second pass: restore all ssm_states */
        for (int l = 0; l < c->n_layers && bytes_left > 0; l++) {
            if (m->weights.layers[l].is_attn_layer) continue;
            if (!s->ssm_state[l]) continue;
            size_t sz = (size_t)c->ssm_d_state * c->ssm_d_inner * sizeof(float);
            if (sz > bytes_left) { free(*tokens_out); _kv_close(fd); return 0; }
            uint8_t *dst = (uint8_t *)s->ssm_state[l];
            size_t remain = sz;
            while (remain > 0) {
                size_t need = remain < buf_size ? remain : buf_size;
                if (_kv_ensure(fd, buf, buf_size, need, &buf_avail, &buf_pos, &file_offset)) return 0;
                size_t copy = need < remain ? need : remain;
                memcpy(dst, buf + buf_pos, copy);
                dst += copy;
                buf_pos += copy;
                remain -= copy;
                bytes_left -= copy;
            }
        }
    }

    _kv_close(fd);
    fprintf(stderr, "\nKV cache loaded: %d positions from %s", n_pos, path);
    if (has_ssm && header[10] > 0) {
        fprintf(stderr, " + SSM state (%.1f MB)", (double)(size_t)header[10] / (1024.0 * 1024.0));
    }
    fprintf(stderr, "\n");
    return n_pos;
}


size_t model_ssm_snapshot_size(const model_t *m) {
    const model_config_t *c = &m->config;
    if (!c->has_ssm) return 0;

    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    size_t total = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        total += (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
        total += (size_t)c->ssm_d_state * c->ssm_d_inner * sizeof(float);
    }
    return total;
}

/* Save current SSM state into pre-allocated buffer.
 * Layout: [conv_state_l0][conv_state_l1]...[state_l0][state_l1]...
 * Returns bytes written (= model_ssm_snapshot_size on success). */
size_t model_ssm_state_save(const model_t *m, uint8_t *buf, size_t buf_size) {
    const model_config_t *c = &m->config;
    if (!c->has_ssm) return 0;

    size_t needed = model_ssm_snapshot_size(m);
    if (buf_size < needed) {
        fprintf(stderr, "SSM state save: buffer too small (%zu < %zu)\n", buf_size, needed);
        return 0;
    }

    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    size_t off = 0;
    /* First pass: save all conv_states */
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (!m->state.ssm_conv_state[l]) continue;
        size_t sz = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
        memcpy(buf + off, m->state.ssm_conv_state[l], sz);
        off += sz;
    }
    /* Second pass: save all ssm_states */
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (!m->state.ssm_state[l]) continue;
        size_t sz = (size_t)c->ssm_d_state * c->ssm_d_inner * sizeof(float);
        memcpy(buf + off, m->state.ssm_state[l], sz);
        off += sz;
    }
    return off;
}

/* Restore SSM state from buffer.
 * Returns bytes read (= model_ssm_snapshot_size on success). */
size_t model_ssm_state_restore(model_t *m, const uint8_t *buf, size_t buf_size) {
    const model_config_t *c = &m->config;
    if (!c->has_ssm) return 0;

    size_t needed = model_ssm_snapshot_size(m);
    if (buf_size < needed) {
        fprintf(stderr, "SSM state restore: buffer too small (%zu < %zu)\n", buf_size, needed);
        return 0;
    }

    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    size_t off = 0;
    /* First pass: restore all conv_states */
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (!m->state.ssm_conv_state[l]) continue;
        size_t sz = (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float);
        memcpy(m->state.ssm_conv_state[l], buf + off, sz);
        off += sz;
    }
    /* Second pass: restore all ssm_states */
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (!m->state.ssm_state[l]) continue;
        size_t sz = (size_t)c->ssm_d_state * c->ssm_d_inner * sizeof(float);
        memcpy(m->state.ssm_state[l], buf + off, sz);
        off += sz;
    }
    return off;
}

/* Reset all SSM state to zero (fresh start). */
void model_ssm_state_reset(model_t *m) {
    const model_config_t *c = &m->config;
    if (!c->has_ssm) return;

    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (m->state.ssm_conv_state[l])
            memset(m->state.ssm_conv_state[l], 0,
                   (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float));
        if (m->state.ssm_state[l])
            memset(m->state.ssm_state[l], 0,
                   (size_t)c->ssm_d_state * c->ssm_d_inner * sizeof(float));
    }
}

#ifdef PICOLM_GPU
/* Phase 2: GPU-pipelined decode (skeleton)
 * Keeps activations on-device across all layers.
 * Falls back to model_forward until fully implemented.
 * The elementwise kernels (gpu_rmsnorm, gpu_rope_apply,
 * gpu_residual_add) are implemented in backend_gpu.cu. */
float *model_forward_gpu(model_t *m, int token, int pos) {
    /* Phase 2: GPU-pipelined forward pass.
     * Keeps activations on-device across all layers.
     * Falls back to model_forward() if pipeline not ready. */
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    gpu_weights_t *gw = &m->gpu;
    run_state_t *s = &m->state;

    int gpu_dev = gw->device;
    int dim = c->n_embd;
    int n_ffn = c->n_ffn;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int seq_len = c->max_seq_len;
    int rope_half = head_dim / 2;
    int q_pipeline_dim = c->has_ssm ? (q_dim * 2) : q_dim;

    /* Verify pipeline is ready */
    if (!gw->kv_active) return model_forward(m, token, pos);
    if (!picolm_gpu_pipe_x(gpu_dev)) return model_forward(m, token, pos);
    if (!gw->rope_cos_dev || !gw->rope_sin_dev) return model_forward(m, token, pos);

    /* Pipeline buffer pointers */
    float *pipe_x = picolm_gpu_pipe_x(gpu_dev);
    float *pipe_xb = picolm_gpu_pipe_xb(gpu_dev);
    float *pipe_q = picolm_gpu_pipe_q(gpu_dev);
    float *pipe_k = picolm_gpu_pipe_k(gpu_dev);
    float *pipe_v = picolm_gpu_pipe_v(gpu_dev);
    float *pipe_attn_out = picolm_gpu_pipe_attn_out(gpu_dev);
    float *pipe_ffn_norm = picolm_gpu_pipe_ffn_norm(gpu_dev);
    float *pipe_gate = picolm_gpu_pipe_gate(gpu_dev);
    float *pipe_up = picolm_gpu_pipe_up(gpu_dev);

    /* GPU layer weight handles */
    gpu_layer_weights_t *gl;

    /* 1. Embedding lookup on CPU (same path as model_forward), then H2D */
    /* For Fimbulvetr (type=1 F16 embd), the generic dequantize_row works.
     * For interleaved Q4_0 formats, fall back to CPU path. */
    if (w->type_token_embd == GGUF_TYPE_Q4_0_8_8 ||
        w->type_token_embd == GGUF_TYPE_Q4_0_4_4 ||
        w->type_token_embd == GGUF_TYPE_Q4_0_4_8) {
        /* Interleaved token embedding formats need the full model_forward
         * dequant path. Fall back to CPU. */
        return model_forward(m, token, pos);
    }
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
        dequantize_row(embd_row, s->x, dim, w->type_token_embd);
        picolm_gpu_memcpy(pipe_x, s->x, dim * sizeof(float), 1, gpu_dev);
    }

    /* KV cache store is now fully device-native (picolm_gpu_kv_store_dev),
     * no host scratch buffer needed. */

    int this_attn_ordinal = 0;

    /* 2. Per-layer pipeline */
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];
        gl = &gw->layers[l];
        int did_cpu_ssm = 0;

        if (!c->has_ssm || lw->is_attn_layer) {
            /* A. RMSNorm: pipe_xb = rmsnorm(pipe_x, attn_norm_w[l]) */
            picolm_gpu_rmsnorm_dev(pipe_xb, pipe_x,
                                    (float *)gw->attn_norm_dev[l],
                                    dim, c->rms_norm_eps, gpu_dev);

            /* B. Q projection: pipe_q = attn_q @ pipe_xb
             * For SSM models this writes q_full_dim = 2*q_dim
             * (interleaved [Q0,Gate0,Q1,Gate1,...]). */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_q,
                                   pipe_q, pipe_xb, 1, gpu_dev, 0, 0);

            /* B1. For SSM models: de-interleave Q+gate from pipe_q.
             * After this, pipe_q holds compacted Q[q_dim], pipe_gate
             * holds gate[q_dim]. For non-SSM models, skip. */
            if (c->has_ssm) {
                /* De-interleave Q+gate. Cannot write Q in-place to pipe_q:
                 * thread h writes pipe_q[h*head_dim] which overlaps with
                 * thread h+1's read from pipe_q[(h+1)*2*head_dim].
                 * Use pipe_attn_out as temp scratch for raw Q+gate data.
                 * pipe_attn_out is sized for q_pipeline_dim (>= q_full_dim).
                 * pipe_ffn_norm is only dim-sized and would overflow.
                 *
                 * SYNC BEFORE D2D to ensure Q projection (on ctx->stream)
                 * completes before the D2D copy (on stream 0) reads pipe_q. */
                picolm_gpu_sync(gpu_dev);
                picolm_gpu_memcpy(pipe_attn_out, pipe_q,
                                   (size_t)n_heads * 2 * head_dim * sizeof(float),
                                   0, gpu_dev);
                /* D2D is synchronous on stream 0, but ctx->stream (cudaStreamNonBlocking)
                 * doesn't synchronize with stream 0. Sync again to ensure the de-interleave
                 * kernel on ctx->stream doesn't start reading pipe_attn_out before D2D writes it. */
                picolm_gpu_sync(gpu_dev);
                picolm_gpu_qg_deinterleave_dev(pipe_attn_out, pipe_q,
                                                pipe_gate, n_heads, head_dim,
                                                gpu_dev);
            }

            /* C. K projection: pipe_k = attn_k @ pipe_xb */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_k,
                                   pipe_k, pipe_xb, 1, gpu_dev, 0, 0);

            /* D. V projection: pipe_v = attn_v @ pipe_xb */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_v,
                                   pipe_v, pipe_xb, 1, gpu_dev, 0, 0);

            /* E. QK-norm (Qwen3): per-head RMSNorm on Q and K.
             * pipe_q is [n_heads][head_dim], weight is [head_dim].
             * pipe_k is [n_kv_heads][head_dim], same weight layout. */
            if (gw->attn_qk_norm_q_dev[l]) {
                picolm_gpu_rmsnorm_batched(pipe_q, pipe_q,
                                           (float *)gw->attn_qk_norm_q_dev[l],
                                           head_dim, c->rms_norm_eps, n_heads, 0, gpu_dev);
                picolm_gpu_rmsnorm_batched(pipe_k, pipe_k,
                                           (float *)gw->attn_qk_norm_k_dev[l],
                                           head_dim, c->rms_norm_eps, n_kv_heads, 0, gpu_dev);
            }

            /* F. RoPE on Q (in-place): rope(pipe_q, n_heads) */
            /* RoPE tables for this position on device */
            float *rope_cos_pos = (float *)gw->rope_cos_dev + (size_t)pos * rope_half;
            float *rope_sin_pos = (float *)gw->rope_sin_dev + (size_t)pos * rope_half;
            picolm_gpu_rope_apply(pipe_q, n_heads, head_dim,
                                   rope_cos_pos, rope_sin_pos, rope_half, c->rope_type, gpu_dev);

            /* G. RoPE on K (in-place): rope(pipe_k, n_kv_heads) */
            picolm_gpu_rope_apply(pipe_k, n_kv_heads, head_dim,
                                   rope_cos_pos, rope_sin_pos, rope_half, c->rope_type, gpu_dev);

            /* G. KV cache store: pack F32 -> F16 and write directly into the
             * device KV cache, entirely device-to-device. The previous
             * version of this step did a synchronous D2H of pipe_k/pipe_v,
             * a CPU F16 conversion, then an H2D via picolm_gpu_kv_store_rows
             * -- each layer, twice (K and V). picolm_gpu_memcpy is a
             * blocking gpuMemcpy, so that was also forcing a full device
             * sync twice per layer (64x per token for a 32-layer model),
             * which defeated most of the point of this pipeline. This
             * version never touches the host, so that costs nothing beyond
             * two tiny kernel launches on ctx->stream.
             *
             * Trade-off: s->key_cache/val_cache (the host-side KV mirror)
             * is NOT updated here anymore. That's fine for the current
             * eligibility gate (kv_active requires no SSM, decided once at
             * model load, never changes mid-generation), but if a mid-stream
             * CPU fallback is ever introduced, it needs a one-time bulk
             * device->host flush of the whole KV cache first -- not
             * per-token reconstruction. Not needed today; flagged here so
             * it isn't a silent trap later. */
            picolm_gpu_kv_store_dev(1, this_attn_ordinal, pos, pipe_k,
                                     n_kv_heads, head_dim, seq_len, gpu_dev);
            picolm_gpu_kv_store_dev(0, this_attn_ordinal, pos, pipe_v,
                                     n_kv_heads, head_dim, seq_len, gpu_dev);
            this_attn_ordinal++;

            /* H. Attention decode: pipe_attn_out = attn(pipe_q)
             * Uses this_attn_ordinal - 1 (already incremented above), the
             * same compacted index the KV cache was just written at. This
             * ordinal only advances for actual attention layers (above) --
             * SSM/hybrid layers below have no KV cache entry at all, they
             * index ssm_state_dev/ssm_conv_state_dev by the raw layer index
             * `l` instead. */
            picolm_gpu_attention_decode_dev(pipe_attn_out, pipe_q,
                                             this_attn_ordinal - 1, pos,
                                             n_heads, n_kv_heads, head_dim,
                                             seq_len, gpu_dev);

            /* I1. For SSM models: apply gate sigmoid to attention output.
             * pipe_attn_out *= sigmoid(pipe_gate)
             * This must happen before the output projection. */
            if (c->has_ssm) {
                picolm_gpu_sigmoid_mul_dev(pipe_attn_out, pipe_gate,
                                            q_dim, gpu_dev);
            }

            /* I2. Output projection: pipe_xb = attn_output @ pipe_attn_out */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_output,
                                   pipe_xb, pipe_attn_out, 1, gpu_dev, q_pipeline_dim, 0);

            /* J. Residual add: pipe_x += pipe_xb */
            picolm_gpu_residual_add(pipe_x, pipe_x, pipe_xb, 1, dim, q_pipeline_dim, gpu_dev);
        } else {
            /* SSM/hybrid layer: try GPU-native path first, fallback to CPU hybrid */
            if (!ssm_forward_gpu(m, s, s->x, s->xb2, lw, l, pos,
                                  &m->gpu.layers[l], gpu_dev)) {
                /* Fallback to CPU hybrid */
                picolm_gpu_sync(gpu_dev);
                picolm_gpu_memcpy(s->x, pipe_x, (size_t)dim * sizeof(float), -1, gpu_dev);

                float *ssm_residual = s->xb2;
                ssm_forward(m, s, s->x, ssm_residual, lw, l, pos, &m->gpu.layers[l]);

                picolm_gpu_memcpy(pipe_x, s->x, (size_t)dim * sizeof(float), 1, gpu_dev);
                did_cpu_ssm = 1;
            }
        }

        /* FFN block: only for attention layers (and any GPU-side SSM
         * layer if that path is ever re-enabled). ssm_forward() already
         * runs its own RMSNorm+FFN+residual internally when the layer
         * has one, so running this again for a CPU-hybrid SSM layer
         * would apply the FFN twice. */
        if (!did_cpu_ssm) {
            /* K. FFN: pipe_xb = rmsnorm(pipe_x, post_attn_norm_w[l]) */
            picolm_gpu_rmsnorm_dev(pipe_ffn_norm, pipe_x,
                                    (float *)gw->post_attn_norm_dev[l],
                                    dim, c->rms_norm_eps, gpu_dev);

            /* L. Gate: pipe_gate = ffn_gate @ pipe_ffn_norm */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_gate,
                                   pipe_gate, pipe_ffn_norm, 1, gpu_dev, 0, 0);

            /* M. Up: pipe_up = ffn_up @ pipe_ffn_norm */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_up,
                                   pipe_up, pipe_ffn_norm, 1, gpu_dev, 0, 0);

            /* N. SiLU-mul: pipe_gate = silu(pipe_gate) * pipe_up (in-place on gate) */
            picolm_gpu_silu_mul_dev(pipe_gate, pipe_up, n_ffn, gpu_dev);

            /* O. Down: pipe_xb = ffn_down @ pipe_gate */
            picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_down,
                                   pipe_xb, pipe_gate, 1, gpu_dev, q_pipeline_dim, 0);

            /* P. Residual add: pipe_x += pipe_xb */
            picolm_gpu_residual_add(pipe_x, pipe_x, pipe_xb, 1, dim, q_pipeline_dim, gpu_dev);
        } /* end if (!did_cpu_ssm) */
    }

    /* 3. Final RMSNorm: pipe_x = rmsnorm(pipe_x, output_norm_w) */
    picolm_gpu_rmsnorm_dev(pipe_x, pipe_x,
                            (float *)gw->output_norm_dev,
                            dim, c->rms_norm_eps, gpu_dev);

    /* 4. Sync once, then lm_head on host */
    picolm_gpu_sync(gpu_dev);

    /* Download pipe_x to host */
    picolm_gpu_memcpy(s->x, pipe_x, dim * sizeof(float), -1, gpu_dev);

    /* 5. Output projection -> logits (host-facing, needs D2H anyway for sampling) */
    tensor_set_repacked(m->repack_used[1] ? m->repack_buffers[1] : NULL);
    tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gw->output, gpu_dev);
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);
    tensor_set_repacked(NULL);
    tensor_set_gpu_tensor(NULL, 0);

    return s->logits;
}

/* Phase 2: GPU-pipelined prefill forward pass.
 * Keeps activations on-device across all layers with S=n_tokens batching.
 * Falls back to model_forward_prefill() if pipeline not ready. */
float *model_forward_prefill_gpu(model_t *m, const int *tokens, int n_tokens, int start_pos, volatile int *interrupt) {
#ifdef PICOLM_GPU
    /* Allow nsys to skip model upload: profile only this function */
    cudaProfilerStart();
#endif
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    gpu_weights_t *gw = &m->gpu;
    run_state_t *s = &m->state;

    int gpu_dev = gw->device;
    int dim = c->n_embd;
    int n_ffn = c->n_ffn;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int seq_len = c->max_seq_len;
    int rope_half = head_dim / 2;
    int q_dim = n_heads * head_dim;
    int xb_stride = c->has_ssm ? (q_dim * 2) : q_dim;
    if (dim > xb_stride) xb_stride = dim;
    int ssm_conv_dim = c->ssm_d_inner + 2 * c->ssm_d_state * c->ssm_n_group;
    if (ssm_conv_dim > xb_stride) xb_stride = ssm_conv_dim;

    /* Verify pipeline and batch buffers are ready */
    if (!gw->kv_active) return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);
    if (!picolm_gpu_pipe_x(gpu_dev)) return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);
    if (!picolm_gpu_pipe_x_b(gpu_dev)) return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);
    if (!gw->rope_cos_dev || !gw->rope_sin_dev) return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);

    /* Batch buffer pointers */
    float *bx = picolm_gpu_pipe_x_b(gpu_dev);
    float *bxb = picolm_gpu_pipe_xb_b(gpu_dev);
    float *bq = picolm_gpu_pipe_q_b(gpu_dev);
    float *bk = picolm_gpu_pipe_k_b(gpu_dev);
    float *bv = picolm_gpu_pipe_v_b(gpu_dev);
    float *battn_out = picolm_gpu_pipe_attn_out_b(gpu_dev);
    float *bffn_norm = picolm_gpu_pipe_ffn_norm_b(gpu_dev);
    float *bgate = picolm_gpu_pipe_gate_b(gpu_dev);
    float *bup = picolm_gpu_pipe_up_b(gpu_dev);

    gpu_layer_weights_t *gl;

    /* 1. Embedding lookup (CPU) into host buffer, then single H2D */
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        if (w->type_token_embd == GGUF_TYPE_Q4_0_8_8 ||
            w->type_token_embd == GGUF_TYPE_Q4_0_4_4 ||
            w->type_token_embd == GGUF_TYPE_Q4_0_4_8) {
            return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);
        }
        size_t embd_bytes = (size_t)n_tokens * dim * sizeof(float);
        float *host_embd = (float *)malloc(embd_bytes);
        if (!host_embd) {
            return model_forward_prefill(m, tokens, n_tokens, start_pos, interrupt);
        }
        for (int bi = 0; bi < n_tokens; bi++) {
            const void *embd_row = (const uint8_t *)w->token_embd + (size_t)tokens[bi] * row_bytes;
            float *dst = host_embd + (size_t)bi * dim;
            dequantize_row(embd_row, dst, dim, w->type_token_embd);
        }
        /* Strided H2D for entire batch (pipe_x_b has xb_stride, not dim) */
        for (int bi = 0; bi < n_tokens; bi++) {
            picolm_gpu_memcpy(bx + (size_t)bi * xb_stride, host_embd + (size_t)bi * dim, dim * sizeof(float), 1, gpu_dev);
        }
        free(host_embd);
    }

    int this_attn_ordinal = 0;

    /* 2. Per-layer pipeline (S=n_tokens batched) */
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];
        gl = &gw->layers[l];

        if (c->has_ssm && !lw->is_attn_layer) {
            /* SSM/hybrid layer: try GPU-native path first, fallback to CPU hybrid */
            if (ssm_prefill_layer_gpu(m, s, bx, bxb, bq, battn_out, bffn_norm, bgate, bup, lw, l, n_tokens, start_pos, gpu_dev)) {
                static int w1 = 0;
                if (!w1) { fprintf(stderr, "INFO: SSM prefill device-native GPU active\n"); w1 = 1; }
            } else {
                static int w2 = 0;
                if (!w2) { fprintf(stderr, "WARN: SSM prefill GPU->hybrid CPU\n"); w2 = 1; }
                size_t batch_bytes = (size_t)n_tokens * dim * sizeof(float);
                picolm_gpu_sync(gpu_dev);
                picolm_gpu_memcpy(s->x, bx, batch_bytes, -1, gpu_dev);
                { int xb2s=dim;if(c->ssm_d_inner>dim)xb2s=c->ssm_d_inner;
                  int fs=n_ffn,md=c->n_heads*2*c->head_dim;if(dim>md)md=dim;
                  int kd=c->n_kv_heads*c->head_dim,qf=c->n_heads*2*c->head_dim;
                  size_t sz=(size_t)n_tokens*(dim+md+xb2s+qf+2*kd+2*fs);
                  float *buf=(float*)malloc(sz*sizeof(float));
                  if(!buf)return model_forward_prefill(m,tokens,n_tokens,start_pos,interrupt);
                  float *p2=buf;float *xb=p2;p2+=n_tokens*dim;float *xbb=p2;p2+=n_tokens*md;
                  float *xb2=p2;p2+=n_tokens*xb2s;float *qb=p2;p2+=n_tokens*qf;(void)qb;
                  float *kb=p2;p2+=n_tokens*kd;(void)kb;float *vb=p2;p2+=n_tokens*kd;(void)vb;
                  float *hb=p2;p2+=n_tokens*fs;float *hb2=p2;p2+=n_tokens*fs;
                  memcpy(xb,s->x,batch_bytes);
                  ssm_prefill_layer(m,s,xb,xbb,xb2,hb,hb2,lw,l,n_tokens,start_pos,xb2s,(void**)m->gpu.layers);
                  memcpy(s->x,xb,batch_bytes);
                  /* Dump bx_last for bit-exact GPU vs CPU comparison */
                  if (l == 0) {
                      float bx_rms=0;
                      for (int i=0;i<100;i++) bx_rms += xb[(n_tokens-1)*dim+i]*xb[(n_tokens-1)*dim+i];
                      bx_rms = sqrtf(bx_rms/100);
                      fprintf(stderr, "[CPU l0] bx_last[:8]={%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f} rms100=%.6f\n",
                          xb[(n_tokens-1)*dim],xb[(n_tokens-1)*dim+1],xb[(n_tokens-1)*dim+2],xb[(n_tokens-1)*dim+3],
                          xb[(n_tokens-1)*dim+4],xb[(n_tokens-1)*dim+5],xb[(n_tokens-1)*dim+6],xb[(n_tokens-1)*dim+7], bx_rms);
                  }
                  for(int _bi=0;_bi<n_tokens;_bi++){
                      picolm_gpu_memcpy(bx+(size_t)_bi*xb_stride,s->x+(size_t)_bi*dim,dim*sizeof(float),1,gpu_dev);}
                  free(buf);}
            }
            if(getenv("PICOLM_SSM_VERIFY")){
                picolm_gpu_sync(gpu_dev);
                float lt8[8];picolm_gpu_memcpy(lt8,bx+(size_t)(n_tokens-1)*xb_stride,32,0,gpu_dev);
                fprintf(stderr,"[DBG GPU l=%d] bx_last[:4]={%.6f,%.6f,%.6f,%.6f}\n",l,lt8[0],lt8[1],lt8[2],lt8[3]);}
            continue;
        }

        /* A. RMSNorm batched: one launch for all n_tokens */
#ifdef PICOLM_SSM_VERIFY
        if (l == 0 || l == 47 || l == 59 || l == 64) {
            float tmp8[8];
            picolm_gpu_sync(gpu_dev);
            picolm_gpu_memcpy(tmp8, bx, sizeof(tmp8), -1, gpu_dev);
            int bn=0; for(int i=0;i<8;i++) if(isnanf(tmp8[i])) bn++;
            fprintf(stderr, "SSM_VERIFY attn_pre l=%d bx_first[:4]={%.4f,%.4f,%.4f,%.4f} nan=%d\n",
                l, tmp8[0],tmp8[1],tmp8[2],tmp8[3],bn);
            float tmpl[8];
            picolm_gpu_memcpy(tmpl, bx + (size_t)(n_tokens-1)*dim, sizeof(tmpl), -1, gpu_dev);
            fprintf(stderr, "SSM_VERIFY attn_pre l=%d bx_last[:4]={%.4f,%.4f,%.4f,%.4f}\n",
                l, tmpl[0],tmpl[1],tmpl[2],tmpl[3]);
        }
#endif
        picolm_gpu_rmsnorm_batched_dev(bxb, bx,
                                        (float *)gw->attn_norm_dev[l],
                                        dim, c->rms_norm_eps, n_tokens, xb_stride, gpu_dev);
#ifdef PICOLM_SSM_VERIFY
        if (l == 48 || l == 64) {
            float tmp8[8];
            picolm_gpu_sync(gpu_dev);
            picolm_gpu_memcpy(tmp8, bxb, sizeof(tmp8), -1, gpu_dev);
            int bn=0; for(int i=0;i<8;i++) if(isnanf(tmp8[i])) bn++;
            fprintf(stderr, "SSM_VERIFY attn_rms l=%d bxb[:4]={%.4f,%.4f,%.4f,%.4f} nan=%d\n",
                l, tmp8[0],tmp8[1],tmp8[2],tmp8[3],bn);
        }
#endif

        /* B. Q projection: bq = attn_q @ bxb (S=n_tokens)
         * For SSM models this writes q_full_dim = 2*q_dim per token. */
        picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_q,
                               bq, bxb, n_tokens, gpu_dev, 0, 0);

        if(getenv("PICOLM_SSM_VERIFY") && l==64){
            float bq_last[4]; picolm_gpu_sync(gpu_dev);
            picolm_gpu_memcpy(bq_last, bq+(size_t)(n_tokens-1)*n_heads*2*head_dim, 16, -1, gpu_dev);
            fprintf(stderr,"[DBG] bq_last l=64[:4]={%.6f,%.6f,%.6f,%.6f}\n",bq_last[0],bq_last[1],bq_last[2],bq_last[3]);}

        /* B1. For SSM models: de-interleave Q+gate from bq.
         * After this, bq holds compacted Q[q_dim] per token,
         * bgate holds gate[q_dim] per token. */
        if (c->has_ssm) {
            int q_dim = n_heads * head_dim;
            /* Copy raw Q+gate to battn_out (scratch) before de-interleaving.
             * Same in-place aliasing bug as decode path. */
            size_t qg_bytes = (size_t)n_tokens * n_heads * 2 * head_dim * sizeof(float);
            picolm_gpu_memcpy(battn_out, bq, qg_bytes, 0, gpu_dev);
            picolm_gpu_sync(gpu_dev); /* ensure D2D copy completes before kernel */
            picolm_gpu_qg_deinterleave_batched_dev(battn_out, bq, bgate,
                                                    n_heads, head_dim,
                                                    n_tokens, gpu_dev);
        }

        /* C. K projection: bk = attn_k @ bxb */
        int k_ok = picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_k, bk, bxb, n_tokens, gpu_dev, 0, 0);
        if(getenv("PICOLM_SSM_VERIFY") && l==3){
            float bk8[8]; picolm_gpu_sync(gpu_dev);
            picolm_gpu_memcpy(bk8, bk+(size_t)(n_tokens-1)*n_kv_heads*head_dim, sizeof(bk8), -1, gpu_dev);
            double br=0;for(int _i=0;_i<8;_i++)br+=bk8[_i]*bk8[_i];
            fprintf(stderr,"[DBG] attn_K l=3 last[:4]={%.6f,%.6f,%.6f,%.6f} rms8=%.6f\n",bk8[0],bk8[1],bk8[2],bk8[3],sqrt(br/8));}


        /* D. V projection: bv = attn_v @ bxb */
        picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_v,
                               bv, bxb, n_tokens, gpu_dev, 0, 0);


        /* D1. QK-norm (Qwen3): per-head RMSNorm on Q and K (batched over S).
         * bq is [S][n_heads][head_dim], weight is [head_dim].
         * bk is [S][n_kv_heads][head_dim], same weight layout.
         * Total heads for batched RMSNorm: S * n_heads. */
        if (gw->attn_qk_norm_q_dev[l]) {
            picolm_gpu_rmsnorm_batched(bq, bq,
                                       (float *)gw->attn_qk_norm_q_dev[l],
                                       head_dim, c->rms_norm_eps, n_heads * n_tokens, 0, gpu_dev);
            picolm_gpu_rmsnorm_batched(bk, bk,
                                       (float *)gw->attn_qk_norm_k_dev[l],
                                       head_dim, c->rms_norm_eps, n_kv_heads * n_tokens, 0, gpu_dev);
        }

        /* E. RoPE on Q (batched, one launch for whole chunk) */
        picolm_gpu_rope_apply_batched(bq, n_heads, head_dim,
                                       (float *)gw->rope_cos_dev,
                                       (float *)gw->rope_sin_dev,
                                       rope_half, start_pos, n_tokens, c->rope_type, gpu_dev);

        /* F. RoPE on K (batched) */
        picolm_gpu_rope_apply_batched(bk, n_kv_heads, head_dim,
                                       (float *)gw->rope_cos_dev,
                                       (float *)gw->rope_sin_dev,
                                       rope_half, start_pos, n_tokens, c->rope_type, gpu_dev);

        /* G. KV cache store: batched F32->F16 pack+store, 2 launches per layer */
        picolm_gpu_kv_store_dev_batched(1, this_attn_ordinal, start_pos, n_tokens,
                                         bk, n_kv_heads, head_dim, seq_len, gpu_dev);
        picolm_gpu_kv_store_dev_batched(0, this_attn_ordinal, start_pos, n_tokens,
                                         bv, n_kv_heads, head_dim, seq_len, gpu_dev);
        if(getenv("PICOLM_SSM_VERIFY") && l == 64){
            float bk_sample[4];
            picolm_gpu_sync(gpu_dev);
            picolm_gpu_memcpy(bk_sample, bk, 16, -1, gpu_dev);
            fprintf(stderr,"[DBG] K_f32 l=64 token0[:4]={%.6f,%.6f,%.6f,%.6f}\n",
                bk_sample[0],bk_sample[1],bk_sample[2],bk_sample[3]);
        }
        this_attn_ordinal++;

        /* H. Attention prefill: battn_out = attn_prefill(bq) */
        if(getenv("PICOLM_SSM_VERIFY") && l==3){
            /* Per-head RMS of attn output (pre-gate) */
            float *odmp = (float*)malloc(n_heads*head_dim*sizeof(float));
            picolm_gpu_attention_prefill_dev(battn_out, bq,
                                              this_attn_ordinal - 1, start_pos, n_tokens,
                                              n_heads, n_kv_heads, head_dim,
                                              seq_len, gpu_dev);
            picolm_gpu_sync(gpu_dev);
            picolm_gpu_memcpy(odmp, battn_out+(size_t)(n_tokens-1)*n_heads*head_dim, n_heads*head_dim*sizeof(float), -1, gpu_dev);
            fprintf(stderr,"[DBG] attn_O l=3 per-head RMS (pre-gate): ");
            for(int h=0;h<n_heads;h++){double hr=0;for(int d=0;d<head_dim;d++)hr+=odmp[(h*head_dim+d)]*odmp[(h*head_dim+d)];fprintf(stderr,"h%d=%.4f ",h,sqrt(hr/head_dim));}
            fprintf(stderr,"\n");
            free(odmp);
        } else {
            picolm_gpu_attention_prefill_dev(battn_out, bq,
                                              this_attn_ordinal - 1, start_pos, n_tokens,
                                              n_heads, n_kv_heads, head_dim,
                                              seq_len, gpu_dev);
        }
        if(getenv("PICOLM_SSM_VERIFY") && (l==64||l==3)){
            float aol[4]; picolm_gpu_sync(gpu_dev); picolm_gpu_memcpy(aol, battn_out+(size_t)(n_tokens-1)*n_heads*head_dim, 16, -1, gpu_dev);
            fprintf(stderr,"[DBG] attn_raw l=%d last[:4]={%.6f,%.6f,%.6f,%.6f}\n",l,aol[0],aol[1],aol[2],aol[3]);}

        /* H1. For SSM models: apply gate sigmoid to attention output. */
        if (c->has_ssm) {
            int q_dim = n_heads * head_dim;
            picolm_gpu_sigmoid_mul_batched_dev(battn_out, bgate,
                                                q_dim, n_tokens, gpu_dev);
        if(getenv("PICOLM_SSM_VERIFY") && l==3){
            float *gdmp = (float*)malloc(q_dim*sizeof(float));
            picolm_gpu_sync(gpu_dev);
            picolm_gpu_memcpy(gdmp, battn_out+(size_t)(n_tokens-1)*q_dim, q_dim*sizeof(float), -1, gpu_dev);
            double gr=0;for(int _i=0;_i<q_dim;_i++)gr+=gdmp[_i]*gdmp[_i];
            fprintf(stderr,"[DBG] attn_gated l=3 token_rms=%.6f first4={%.6f,%.6f,%.6f,%.6f}\n",sqrt(gr/q_dim),gdmp[0],gdmp[1],gdmp[2],gdmp[3]);
            free(gdmp);}
        }
        if(getenv("PICOLM_SSM_VERIFY") && l==64){
            float ao[4], aol[4]; picolm_gpu_sync(gpu_dev); picolm_gpu_memcpy(ao, battn_out, 16, -1, gpu_dev);
            picolm_gpu_memcpy(aol, battn_out+(size_t)(n_tokens-1)*n_heads*2*head_dim, 16, -1, gpu_dev);
            fprintf(stderr,"[DBG] attn_gate l=64 first[:4]={%.6f,%.6f,%.6f,%.6f} last[:4]={%.6f,%.6f,%.6f,%.6f}\n",
                ao[0],ao[1],ao[2],ao[3], aol[0],aol[1],aol[2],aol[3]);}

        /* I. Output projection: bxb = attn_output @ battn_out */
        picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->attn_output,
                               bxb, battn_out, n_tokens, gpu_dev, xb_stride, 0);

        if(getenv("PICOLM_SSM_VERIFY") && (l==64||l==3)){
            float oo[4]; picolm_gpu_sync(gpu_dev); picolm_gpu_memcpy(oo, bxb+(size_t)(n_tokens-1)*xb_stride, 16, -1, gpu_dev);
            fprintf(stderr,"[DBG] outproj l=%d last[:4]={%.6f,%.6f,%.6f,%.6f}\n",l,oo[0],oo[1],oo[2],oo[3]);}

        /* J. Residual add: bx += bxb (batched, single launch) */
        picolm_gpu_residual_add(bx, bx, bxb, n_tokens, dim, xb_stride, gpu_dev);
        if(getenv("PICOLM_SSM_VERIFY") && l==64){
            float rx[4]; picolm_gpu_sync(gpu_dev); picolm_gpu_memcpy(rx, bx+(size_t)(n_tokens-1)*xb_stride, 16, -1, gpu_dev);
            fprintf(stderr,"[DBG] resid l=64 bx_last[:4]={%.6f,%.6f,%.6f,%.6f}\n",rx[0],rx[1],rx[2],rx[3]);}

        /* K. FFN: rmsnorm (batched) */
        picolm_gpu_rmsnorm_batched_dev(bffn_norm, bx,
                                        (float *)gw->post_attn_norm_dev[l],
                                        dim, c->rms_norm_eps, n_tokens, xb_stride, gpu_dev);

        /* FFN gate */
        picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_gate,
                               bgate, bffn_norm, n_tokens, gpu_dev, 0, 0);

        /* FFN up */
        picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_up,
                               bup, bffn_norm, n_tokens, gpu_dev, 0, 0);

        /* FFN silu_mul: bgate = silu(bgate) * bup (batched, single launch) */
        picolm_gpu_silu_mul_dev(bgate, bup, n_tokens * n_ffn, gpu_dev);

        /* FFN down: bxb = down_proj @ bgate */
        picolm_gpu_matmul_dev((picolm_gpu_tensor_t *)gl->ffn_down,
                               bxb, bgate, n_tokens, gpu_dev, xb_stride, 0);

        /* FFN residual: bx += bxb (batched, single launch) */
        picolm_gpu_residual_add(bx, bx, bxb, n_tokens, dim, xb_stride, gpu_dev);
        if(getenv("PICOLM_SSM_VERIFY") && (l==3||l==48||l==56||l==64)){
            float tmp[4]; picolm_gpu_sync(gpu_dev);
            picolm_gpu_memcpy(tmp,bx+(size_t)(n_tokens-1)*xb_stride,16,0,gpu_dev);
            fprintf(stderr,"[DBG] attn_post l=%d bx_last[:4]={%.6f,%.6f,%.6f,%.6f}\n",l,tmp[0],tmp[1],tmp[2],tmp[3]);
        }
    }

    /* 3. Final rmsnorm + sync + D2H + host lm_head matmul */
    {
        float *last_x = bx + (size_t)(n_tokens - 1) * xb_stride;
        picolm_gpu_sync(gpu_dev);
        picolm_gpu_memcpy(s->x, last_x, dim * sizeof(float), -1, gpu_dev);
        if(getenv("PICOLM_SSM_VERIFY")){
            fprintf(stderr,"[DBG] prefill_last x[:4]={%.6f,%.6f,%.6f,%.6f}\n",
                s->x[0],s->x[1],s->x[2],s->x[3]);
            double r=0;for(int i=0;i<dim;i++)r+=s->x[i]*s->x[i];
            fprintf(stderr,"[DBG] prefill_last x_rms=%.6e\n",sqrt(r/dim));
        }
    }

    rmsnorm(s->x, s->x, s->output_norm_w, dim, c->rms_norm_eps);

    tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gw->output, gpu_dev);
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);
    tensor_set_repacked(NULL);
    tensor_set_gpu_tensor(NULL, 0);

    return s->logits;
}

void picolm_ssm_state_sync_to_device(model_t *m, int device) {
    model_config_t *c = &m->config;
    if (!c->has_ssm) return;
    int conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].is_attn_layer) continue;
        if (m->gpu.ssm_state_dev[l] && m->state.ssm_state[l]) {
            size_t sz = (size_t)c->ssm_dt_rank * c->ssm_d_state * c->ssm_d_state * sizeof(float);
            picolm_gpu_memcpy(m->gpu.ssm_state_dev[l], m->state.ssm_state[l], sz, 1, device);
        }
        if (m->gpu.ssm_conv_state_dev[l] && m->state.ssm_conv_state[l]) {
            size_t sz = (size_t)(c->ssm_d_conv - 1) * (size_t)conv_dim * sizeof(float);
            picolm_gpu_memcpy(m->gpu.ssm_conv_state_dev[l], m->state.ssm_conv_state[l], sz, 1, device);
        }
    }
#ifdef PICOLM_GPU
    cudaProfilerStop();
#endif
}

#endif /* PICOLM_GPU */
