#include "model.h"
#include "tensor.h"
#include "quant.h"

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
#endif
#include <errno.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <windows.h>
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
#endif

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
    if (m->mmap_addr) {
        _do_prefault(m->mmap_addr, m->mmap_size);
    }
}

static int mmap_file(model_t *m, const char *path) {
#ifdef _WIN32
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        return -1;
    }

    LARGE_INTEGER fsize;
    GetFileSizeEx(fh, &fsize);
    m->mmap_size = (size_t)fsize.QuadPart;

    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) {
        fprintf(stderr, "CreateFileMapping failed\n");
        CloseHandle(fh);
        return -1;
    }

    void *addr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        fprintf(stderr, "MapViewOfFile failed\n");
        CloseHandle(mh);
        CloseHandle(fh);
        return -1;
    }

    m->mmap_addr  = addr;
    m->file_handle = fh;
    m->map_handle  = mh;
    prepare_mmap(addr, m->mmap_size);
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        return -1;
    }

    struct stat st;
    fstat(fd, &st);
    m->mmap_size = (size_t)st.st_size;

    void *addr = mmap(NULL, m->mmap_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        close(fd);
        return -1;
    }

    m->mmap_addr = addr;
    m->fd = fd;
    prepare_mmap(addr, m->mmap_size);
#endif
    return 0;
}

static void munmap_file(model_t *m) {
    if (!m->mmap_addr) return;
#ifdef _WIN32
    UnmapViewOfFile(m->mmap_addr);
    CloseHandle(m->map_handle);
    CloseHandle(m->file_handle);
#else
    munmap(m->mmap_addr, m->mmap_size);
    close(m->fd);
#endif
    m->mmap_addr = NULL;
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
    m->tok_bos_id = 1;
    m->tok_eos_id = 2;

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
            } else {
                skip_meta_value(&r, vtype, &dummy);
            }
        } else


        if (str_eq(key, "llama.embedding_length") || str_eq(key, "general.embedding_length")
            || str_eq(key, "qwen2.embedding_length") || str_eq(key, "qwen3.embedding_length") || str_eq(key, "qwen35.embedding_length")) {
            int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
            /* NOTE: Qwen2 uses interleaved RoPE. Qwen3 and Qwen3.5 use pairwise RoPE
             * (same as Llama). Only set rope_type=1 for qwen2, not qwen3/qwen35. */
            if (key.str[0] == 'q' && key.len > 6 && key.str[5] == '2') cfg->rope_type = 1;
        } else if (str_eq(key, "llama.feed_forward_length") || str_eq(key, "general.feed_forward_length")
            || str_eq(key, "qwen2.feed_forward_length") || str_eq(key, "qwen3.feed_forward_length") || str_eq(key, "qwen35.feed_forward_length")) {
            int dummy; cfg->n_ffn = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count")
            || str_eq(key, "qwen2.attention.head_count") || str_eq(key, "qwen3.attention.head_count") || str_eq(key, "qwen35.attention.head_count")) {
            int dummy; cfg->n_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count_kv")
            || str_eq(key, "qwen2.attention.head_count_kv") || str_eq(key, "qwen3.attention.head_count_kv") || str_eq(key, "qwen35.attention.head_count_kv")) {
            int dummy; cfg->n_kv_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "attention.key_length")
            || str_eq(key, "qwen2.attention.key_length")
            || str_eq(key, "qwen3.attention.key_length") || str_eq(key, "qwen35.attention.key_length")) {
            /* Explicit head_dim (Qwen3/3.5 may differ from n_embd/n_heads) */
            int dummy; cfg->head_dim = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.block_count")
            || str_eq(key, "qwen2.block_count") || str_eq(key, "qwen3.block_count") || str_eq(key, "qwen35.block_count")) {
            int dummy; cfg->n_layers = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.context_length")
            || str_eq(key, "qwen2.context_length") || str_eq(key, "qwen3.context_length") || str_eq(key, "qwen35.context_length")) {
            int dummy; cfg->max_seq_len = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.rope.freq_base")
            || str_eq(key, "qwen2.rope.freq_base") || str_eq(key, "qwen3.rope.freq_base") || str_eq(key, "qwen35.rope.freq_base")) {
            if (vtype == GGUF_META_FLOAT32) {
                cfg->rope_freq_base = read_f32(&r);
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "qwen35.rope.dimension_count")
            || str_eq(key, "llama.rope.dimension_count")) {
            int dummy; cfg->rope_dim = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "rope.dimension_sections")
            || str_eq(key, "qwen35.rope.dimension_sections")) {
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
            || str_eq(key, "qwen3.attention.layer_norm_rms_epsilon") || str_eq(key, "qwen35.attention.layer_norm_rms_epsilon")) {
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
        } else if (str_eq(key, "qwen35.ssm.conv_kernel") || str_eq(key, "qwen3.ssm.conv_kernel")) {
            int dummy; cfg->ssm_d_conv = (int)skip_meta_value(&r, vtype, &dummy); cfg->has_ssm = 1;
        } else if (str_eq(key, "qwen35.ssm.state_size") || str_eq(key, "qwen3.ssm.state_size")) {
            int dummy; cfg->ssm_d_state = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.group_count") || str_eq(key, "qwen3.ssm.group_count")) {
            int dummy; cfg->ssm_n_group = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.time_step_rank") || str_eq(key, "qwen3.ssm.time_step_rank")) {
            int dummy; cfg->ssm_dt_rank = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.inner_size") || str_eq(key, "qwen3.ssm.inner_size")) {
            int dummy; cfg->ssm_d_inner = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "general.alignment")) {
            int dummy; cfg->alignment = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.vocab_size")
            || str_eq(key, "qwen2.vocab_size") || str_eq(key, "qwen3.vocab_size") || str_eq(key, "qwen35.vocab_size")) {
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
            /* Detect space marker type for tokenization */
            if (vtype == GGUF_META_STRING) {
                gguf_str_t pre = read_gguf_string(&r);
                if (pre.len >= 6 && strncmp(pre.str, "smollm", 6) == 0) {
                    m->tok_space_marker = 1; /* U+0100 */
                } else if (pre.len >= 6 && strncmp(pre.str, "qwen35", 6) == 0) {
                    m->tok_space_marker = 3; /* qwen35: U+0100, no prefix on first token */
                } else {
                    m->tok_space_marker = 0; /* U+2581 (default) */
                }
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
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
                  r.pos = pp - r.data; }
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
        } else {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        }
    }

    if (max_seq_len > 0 && max_seq_len < cfg->max_seq_len) {
        cfg->max_seq_len = max_seq_len;
    }
    /* head_dim: use GGUF's attention.key_length if set (Qwen3), else derive */
    if (cfg->head_dim <= 0) {
        cfg->head_dim = cfg->n_embd / cfg->n_heads;
    }

    /* Parse tensor info entries */
    typedef struct {
        gguf_str_t name;
        uint32_t   n_dims;
        uint64_t   dims[4];
        uint32_t   type;
        uint64_t   offset;
    } tensor_info_t;

    tensor_info_t *tinfos = (tensor_info_t *)malloc(n_tensors * sizeof(tensor_info_t));
    if (!tinfos) { fprintf(stderr, "OOM allocating tensor info\n"); return -1; }

    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfos[i].name   = read_gguf_string(&r);
        tinfos[i].n_dims = read_u32(&r);
        for (uint32_t d = 0; d < tinfos[i].n_dims; d++) {
            tinfos[i].dims[d] = read_u64(&r);
        }
        tinfos[i].type   = read_u32(&r);
        tinfos[i].offset = read_u64(&r);
            }

    /* Detect MTP (Multi-Token Prediction) layers by scanning for "nextn" tensors */
    cfg->has_mtp = 0;
    cfg->n_mtp_layers = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (strstr(tinfos[i].name.str, "nextn.") && tinfos[i].name.len > 0) {
            cfg->has_mtp = 1;
            break;
        }
    }

    size_t alignment = (size_t)cfg->alignment;
    size_t tensor_data_base = (r.pos + alignment - 1) & ~(alignment - 1);

    model_weights_t *w = &m->weights;
    memset(w, 0, sizeof(*w));

    for (uint64_t i = 0; i < n_tensors; i++) {
        const void *ptr = (const uint8_t *)m->mmap_addr + tensor_data_base + tinfos[i].offset;
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
                    lw->attn_norm = ptr; lw->type_attn_norm = qtype;
                } else if (strcmp(suffix, "attn_q.weight") == 0) {
                    lw->attn_q = ptr; lw->type_attn_q = qtype;
                } else if (strcmp(suffix, "attn_k.weight") == 0) {
                    lw->attn_k = ptr; lw->type_attn_k = qtype; lw->is_attn_layer = 1;
                } else if (strcmp(suffix, "attn_v.weight") == 0) {
                    lw->attn_v = ptr; lw->type_attn_v = qtype; lw->is_attn_layer = 1;
                } else if (strcmp(suffix, "attn_output.weight") == 0) {
                    lw->attn_output = ptr; lw->type_attn_output = qtype;
                } else if (strcmp(suffix, "attn_q_norm.weight") == 0) {
                    lw->attn_q_norm = ptr; lw->type_attn_q_norm = qtype;
                } else if (strcmp(suffix, "attn_k_norm.weight") == 0) {
                    lw->attn_k_norm = ptr; lw->type_attn_k_norm = qtype;
                } else if (strcmp(suffix, "ffn_norm.weight") == 0
                        || strcmp(suffix, "post_attention_norm.weight") == 0) {
                    lw->post_attn_norm = ptr; lw->type_post_attn_norm = qtype;
                } else if (strcmp(suffix, "ffn_gate.weight") == 0) {
                    lw->ffn_gate = ptr; lw->type_ffn_gate = qtype;
                } else if (strcmp(suffix, "ffn_down.weight") == 0) {
                    lw->ffn_down = ptr; lw->type_ffn_down = qtype;
                } else if (strcmp(suffix, "ffn_up.weight") == 0) {
                    lw->ffn_up = ptr; lw->type_ffn_up = qtype;
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
            }
        }
    }

    if (!w->output) {
        w->output = w->token_embd;
        w->type_output = w->type_token_embd;
    }

    if (cfg->vocab_size == 0) {
        for (uint64_t i = 0; i < n_tensors; i++) {
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
            for (uint64_t ti = 0; ti < n_tensors; ti++) {
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
    fprintf(stderr, "  head_dim=%d, rope_dim=%d, rope_base=%.1f\n", cfg->head_dim, cfg->rope_dim, cfg->rope_freq_base);
    /* On big-endian, GGUF stores all multi-byte values as little-endian.
     * Swap F16 values in-place for all quantized block types that contain FP16 scales. */
#if defined(__APPLE__) && defined(__ppc__)
    { int be_start = clock();
    fprintf(stderr, "Big-endian: swapping F16 values...\n");
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_type_t qt = (gguf_type_t)tinfos[i].type;
        void *ptr = (void *)((uint8_t *)m->mmap_addr + tensor_data_base + tinfos[i].offset);
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
static size_t kv_row_size(kv_cache_type_t kv_type, int n) {
    switch (kv_type) {
        case KV_CACHE_F16:  return (size_t)n * sizeof(uint16_t);
        case KV_CACHE_Q8_0: return ((size_t)(n / 32)) * sizeof(block_q8_0);
        case KV_CACHE_Q4_0: return ((size_t)(n / 32)) * sizeof(block_q4_0);
    }
    return 0;
}

int allocate_run_state(model_t *m, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v) {
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

    /* Norm weights: (n_layers * 2 + 1) * n_embd + n_layers * head_dim * 2 (QK-norm) */
    size_t n_norm = (size_t)(c->n_layers * 2 + 1) * c->n_embd
                  + (size_t)c->n_layers * c->head_dim * 2;
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
    size_t total = sz_x + sz_xb + sz_xb2 + sz_q +
                   sz_hb + sz_hb2 + sz_logits +
                   sz_scratch + sz_rope + sz_norm +
                   sz_ssm_conv + sz_ssm_state + sz_ssm_small + sz_ssm_tmp;

    /* Quantized KV cache: separate allocation (only for attention layers) */
    size_t sz_k_row = kv_row_size(kv_type_k, c->head_dim);
    size_t sz_v_row = kv_row_size(kv_type_v, c->head_dim);
    int attn_layer_count = 0;
    if (c->has_ssm) {
        for (int i = 0; i < c->n_layers; i++) {
            if (m->weights.layers[i].is_attn_layer) attn_layer_count++;
        }
    }
    int kv_layers = attn_layer_count > 0 ? attn_layer_count : c->n_layers;
    size_t sz_kv = (size_t)kv_layers * c->max_seq_len * c->n_kv_heads * (sz_k_row + sz_v_row);

    const char *kv_name_k = "f16";
    const char *kv_name_v = "f16";
    if (kv_type_k == KV_CACHE_Q8_0) kv_name_k = "q8_0";
    if (kv_type_k == KV_CACHE_Q4_0) kv_name_k = "q4_0";
    if (kv_type_v == KV_CACHE_Q8_0) kv_name_v = "q8_0";
    if (kv_type_v == KV_CACHE_Q4_0) kv_name_v = "q4_0";
    fprintf(stderr, "Allocating %.2f MB for runtime state (+ %.2f MB KV cache [%s/%s])\n",
            (double)total / (1024.0 * 1024.0),
            (double)sz_kv / (1024.0 * 1024.0), kv_name_k, kv_name_v);

    s->mem_block = calloc(1, total);
    if (!s->mem_block) {
        fprintf(stderr, "OOM: cannot allocate %zu bytes\n", total);
        return -1;
    }
    s->mem_size = total + sz_kv;

    /* Allocate KV cache separately */
    s->kv_block = calloc(1, sz_kv);
    if (!s->kv_block) {
        fprintf(stderr, "OOM: cannot allocate %zu bytes for KV cache\n", sz_kv);
        free(s->mem_block);
        return -1;
    }
    s->kv_size = sz_kv;

    s->kv_type_k = kv_type_k;
    s->kv_type_v = kv_type_v;
    s->kv_row_size_k = sz_k_row;
    s->kv_row_size_v = sz_v_row;

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
    p += n_norm + c->n_embd; /* skip norm weights area (dequantized separately via nw pointer) */

    /* KV cache pointers: K layers first, then V layers (all layers, SSM layers just don't use their slots) */
    size_t layer_stride_k = (size_t)c->max_seq_len * c->n_kv_heads * sz_k_row;
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
        int ssm_conv_dim = 2 * c->ssm_d_state * c->ssm_n_group + c->ssm_d_inner;
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

/* ---- Public API ---- */

int model_load(model_t *m, const char *path, int max_seq_len, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v) {
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
        if (m->config.has_ssm) {
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

    if (allocate_run_state(m, kv_type_k, kv_type_v) != 0) return -1;

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

            int ok = 1;
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
                }
            }

            m->gpu.device = device;
            if (uploaded > 0) {
                m->gpu.active = 1;
                /* Log per-type upload stats */
                int tcounts[20] = {0};
                for (int l = 0; l < c->n_layers; l++) {
                    gpu_layer_weights_t *gl = &m->gpu.layers[l];
                    layer_weights_t *lw = &m->weights.layers[l];
                    const void *ptrs[] = {gl->attn_q, gl->attn_k, gl->attn_v,
                                          gl->attn_output, gl->ffn_gate, gl->ffn_up, gl->ffn_down};
                    int types[] = {lw->type_attn_q, lw->type_attn_k, lw->type_attn_v,
                                   lw->type_attn_output, lw->type_ffn_gate, lw->type_ffn_up, lw->type_ffn_down};
                    for (int j = 0; j < 7; j++) {
                        int t = types[j];
                        if (t >= 0 && t < 20) tcounts[t]++;
                    }
                }
                /* Count output tensor */
                int ot = m->weights.type_output;
                if (ot >= 0 && ot < 20) tcounts[ot]++;
                fprintf(stderr, "INFO: GPU weights uploaded (%d/%d tensors)\n", uploaded, attempted);
                for (int t = 0; t < 20; t++) {
                    if (tcounts[t]) fprintf(stderr, "  type %d: %d tensors\n", t, tcounts[t]);
                }
            } else {
                fprintf(stderr, "WARN: GPU upload failed for all tensors, using CPU\n");
            }
        }
    }
#endif

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
        int n_kv_heads, int kv_type_k, int kv_type_v,
        size_t kv_row_size_k, size_t kv_row_size_v, int head_dim) {
    float max_score = -1e30f, sum_exp = 0.0f;
    float acc[256];
    memset(acc, 0, (size_t)head_dim * sizeof(float));

    for (int t = 0; t <= pos; t++) {
        const uint8_t *kt = kcache + (size_t)t * n_kv_heads * kv_row_size_k + kv_h * kv_row_size_k;
        float score;
        if (kv_type_k == KV_CACHE_Q8_0) score = vec_dot_q8_0_f32(kt, qh, head_dim);
        else if (kv_type_k == KV_CACHE_Q4_0) score = vec_dot_q4_0_f32(kt, qh, head_dim);
        else score = vec_dot_f16_f32(kt, qh, head_dim);
        score /= sqrtf((float)head_dim);
        const uint8_t *vt = vcache + (size_t)t * n_kv_heads * kv_row_size_v + kv_h * kv_row_size_v;
        if (score > max_score) {
            float correction = expf(max_score - score);
            sum_exp = sum_exp * correction + 1.0f;
            if (kv_type_v == KV_CACHE_Q8_0) fma_scale_q8_0_f32(acc, correction, vt, head_dim);
            else if (kv_type_v == KV_CACHE_Q4_0) fma_scale_q4_0_f32(acc, correction, vt, head_dim);
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
    const uint8_t *kcache, *vcache;
    const float *q;   /* [n_heads][head_dim] */
    float *xb;        /* [n_heads][head_dim] */
} attn_group_ctx_t;

static void attention_group(int kv_head_idx, void *ctx_ptr) {
    attn_group_ctx_t *ctx = (attn_group_ctx_t *)ctx_ptr;
    int kv_h = kv_head_idx;
    int kv_mul = ctx->kv_mul;
    int head_dim = ctx->head_dim;
    int pos = ctx->pos;
    int first_qh = kv_h * kv_mul;
    
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
        const uint8_t *kt = ctx->kcache + (size_t)t * ctx->n_kv_heads * ctx->kv_row_size_k + kv_h * ctx->kv_row_size_k;
        const uint8_t *vt = ctx->vcache + (size_t)t * ctx->n_kv_heads * ctx->kv_row_size_v + kv_h * ctx->kv_row_size_v;

#ifdef PICOLM_AVX512
        /* Load K once, convert to F32 (16 F16 per chunk via fp16x16_to_fp32_inline) */
        __m512 k_vec[16]; /* head_dim up to 256 = 16x __m512 */
        {
            const uint16_t *k16 = (const uint16_t *)kt;
            int d = 0;
            for (; d + 16 <= head_dim; d += 16)
                k_vec[d/16] = fp16x16_to_fp32_inline(k16 + d);
        }

        /* Load V once, convert to F32 */
        __m512 v_vec[16]; /* head_dim up to 256 = 16x __m512 */
        {
            const uint16_t *v16 = (const uint16_t *)vt;
            int d = 0;
            for (; d + 16 <= head_dim; d += 16)
                v_vec[d/16] = fp16x16_to_fp32_inline(v16 + d);
        }

        for (int g = 0; g < kv_mul; g++) {
            const float *qg = ctx->q + (first_qh + g) * head_dim;
            float score = 0.0f;

            /* K.dot.Q */
            {
                __m512 s = _mm512_setzero_ps();
                int d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 qf = _mm512_loadu_ps(qg + d);
                    s = _mm512_fmadd_ps(k_vec[d/16], qf, s);
                }
                score = _mm512_reduce_add_ps(s);
                for (; d < head_dim; d++)
                    score += fp16_to_fp32(((uint16_t*)kt)[d]) * qg[d];
            }
            score /= sqrtf((float)head_dim);

            float *accg = acc[g];

            if (score > max_score[g]) {
                float correction = expf(max_score[g] - score);
                sum_exp[g] = sum_exp[g] * correction + 1.0f;
                __m512 cv = _mm512_set1_ps(correction);
                int d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 af = _mm512_loadu_ps(accg + d);
                    _mm512_storeu_ps(accg + d, _mm512_fmadd_ps(af, cv, v_vec[d/16]));
                }
                for (; d < head_dim; d++)
                    accg[d] = accg[d] * correction + fp16_to_fp32(((uint16_t*)vt)[d]);
                max_score[g] = score;
            } else {
                float w = expf(score - max_score[g]);
                sum_exp[g] += w;
                __m512 wv = _mm512_set1_ps(w);
                int d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 af = _mm512_loadu_ps(accg + d);
                    _mm512_storeu_ps(accg + d, _mm512_fmadd_ps(v_vec[d/16], wv, af));
                }
                for (; d < head_dim; d++)
                    accg[d] += w * fp16_to_fp32(((uint16_t*)vt)[d]);
            }
        }
#else
        /* Fallback: per-Q-head processing (same as attn_core, no grouping benefit) */
        for (int g = 0; g < kv_mul; g++) {
            const float *qg = ctx->q + (first_qh + g) * head_dim;
            float score;
            if (ctx->kv_type_k == KV_CACHE_Q8_0) score = vec_dot_q8_0_f32(kt, qg, head_dim);
            else if (ctx->kv_type_k == KV_CACHE_Q4_0) score = vec_dot_q4_0_f32(kt, qg, head_dim);
            else score = vec_dot_f16_f32(kt, qg, head_dim);
            score /= sqrtf((float)head_dim);

            float *accg = acc[g];
            if (score > max_score[g]) {
                float correction = expf(max_score[g] - score);
                sum_exp[g] = sum_exp[g] * correction + 1.0f;
                if (ctx->kv_type_v == KV_CACHE_Q8_0) fma_scale_q8_0_f32(accg, correction, vt, head_dim);
                else if (ctx->kv_type_v == KV_CACHE_Q4_0) fma_scale_q4_0_f32(accg, correction, vt, head_dim);
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

float *model_forward(model_t *m, int token, int pos) {
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
    for (int l = 0; l < n_active_layers; l++) {
        layer_weights_t *lw = &w->layers[l];
#ifdef PICOLM_GPU
        gpu_layer_weights_t *gl = &m->gpu.layers[l];
#endif
        int ri = 2 + l * 9;

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
        uint8_t *kcache_layer = s->key_cache + (size_t)this_attn_ordinal * seq_len * c->n_kv_heads * s->kv_row_size_k;
        uint8_t *vcache_layer = s->val_cache + (size_t)this_attn_ordinal * seq_len * c->n_kv_heads * s->kv_row_size_v;

        /* QK-norm (Qwen3): per-head RMSNorm applied before RoPE */
        if (lw->attn_q_norm) {
            float *qnw = s->attn_q_norm_w[l];
            float *knw = s->attn_k_norm_w[l];
            for (int h = 0; h < n_heads; h++)
                rmsnorm(s->q + h * head_dim, s->q + h * head_dim, qnw, head_dim, c->rms_norm_eps);
            for (int h = 0; h < n_kv_heads; h++)
                rmsnorm(k_tmp + h * head_dim, k_tmp + h * head_dim, knw, head_dim, c->rms_norm_eps);
        }

        /* Apply RoPE to Q and K */
        int rope_dim = (c->rope_dim > 0) ? c->rope_dim : head_dim;
        int rope_half = rope_dim / 2;
        rope(s->q, k_tmp, head_dim, n_heads, n_kv_heads, cos_pos, sin_pos, c->rope_type, rope_half);

        /* Store K per head */
        for (int hkv = 0; hkv < n_kv_heads; hkv++) {
            float *k_head = k_tmp + hkv * head_dim;
            uint8_t *key_pos = kcache_layer + (size_t)pos * c->n_kv_heads * s->kv_row_size_k
                              + hkv * s->kv_row_size_k;
            if (s->kv_type_k == KV_CACHE_Q8_0) {
                quantize_row_q8_0(k_head, key_pos, head_dim);
            } else if (s->kv_type_k == KV_CACHE_Q4_0) {
                quantize_row_q4_0(k_head, key_pos, head_dim);
            } else {
                uint16_t *kf = (uint16_t *)key_pos;
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
        }

        /* V projection -> store per head */
        tensor_set_repacked(m->repack_used[ri+2] ? m->repack_buffers[ri+2] : NULL);
#ifdef PICOLM_GPU
        if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->attn_v, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
        float *v_tmp = s->xb2;
        matmul(v_tmp, s->xb, lw->attn_v, dim, kv_dim, lw->type_attn_v);
        tensor_set_repacked(NULL);
        for (int hkv = 0; hkv < n_kv_heads; hkv++) {
            float *v_head = v_tmp + hkv * head_dim;
            uint8_t *val_pos = vcache_layer + (size_t)pos * c->n_kv_heads * s->kv_row_size_v
                              + hkv * s->kv_row_size_v;
            if (s->kv_type_v == KV_CACHE_Q8_0) {
                quantize_row_q8_0(v_head, val_pos, head_dim);
            } else if (s->kv_type_v == KV_CACHE_Q4_0) {
                quantize_row_q4_0(v_head, val_pos, head_dim);
            } else {
                uint16_t *vf = (uint16_t *)val_pos;
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
        attn_group_ctx_t gctx;
        gctx.kv_mul = kv_mul; gctx.head_dim = head_dim; gctx.pos = pos;
        gctx.kv_type_k = s->kv_type_k; gctx.kv_type_v = s->kv_type_v;
        gctx.kv_row_size_k = s->kv_row_size_k; gctx.kv_row_size_v = s->kv_row_size_v;
        gctx.kcache = kcache_layer; gctx.vcache = vcache_layer;
        gctx.q = s->q; gctx.xb = s->xb;
        gctx.n_kv_heads = c->n_kv_heads;
        tensor_parallel_for(c->n_kv_heads, attention_group, &gctx);

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

        /* ---- FFN (SwiGLU) - only if MLP weights exist for this layer ---- */
        if (lw->ffn_gate && lw->ffn_up && lw->ffn_down) {
            rmsnorm(s->xb, s->x, s->post_attn_norm_w[l], dim, c->rms_norm_eps);

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
            vec_add(s->x, s->xb, dim);
        }
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
 * SSM forward pass helpers (Qwen3.5)
 * ================================================================ */

#ifdef DEBUG_SSM
static void dbg_vec(const char *tag, float *v, int n, int max_print) {
    int p = n < max_print ? n : max_print;
    fprintf(stderr, "DBG %s: ", tag);
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

    /* Decay: elementwise */
    for (int i = 0; i < d_state * d_state; i++) st[i] *= ge;

    int kh = h / ctx->repeat;
    const float *qh = ctx->q_conv + (size_t)kh * d_state;
    const float *khv = ctx->k_conv + (size_t)kh * d_state;
    const float *vh = ctx->v_conv + (size_t)h * ctx->head_v_dim;
    float bh = ctx->beta[h];

    float sk_local[256];
    float d_local[256];

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
}

/* SSM layer forward pass (autoregressive, single token) */
static void ssm_forward(model_t *m, run_state_t *s, float *x, float *residual,
                        layer_weights_t *lw, int il, int pos, void *gpu_lw) {
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
#ifdef PICOLM_GPU
    if (gpu_lw) {
        gpu_layer_weights_t *gl = (gpu_layer_weights_t *)gpu_lw;
        tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->attn_qkv, m->gpu.device);
    }
#endif
    matmul(s->q, s->xb, lw->attn_qkv, dim, conv_dim, lw->type_attn_qkv);
#ifdef PICOLM_GPU
    if (gpu_lw) {
        gpu_layer_weights_t *gl = (gpu_layer_weights_t *)gpu_lw;
        tensor_set_gpu_tensor((picolm_gpu_tensor_t *)gl->attn_gate_ssm, m->gpu.device);
    }
#endif
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("qkv[:8]", s->q, 8, 8);
#endif

    /* 3. Z gate: z = matmul(attn_gate_ssm, xb) -> [value_dim] */
    matmul(s->xb2, s->xb, lw->attn_gate_ssm, dim, c->ssm_d_inner, lw->type_attn_gate_ssm);
#ifdef PICOLM_GPU
    if (gpu_lw) tensor_set_gpu_tensor(NULL, 0); /* clear before next layer */
#endif
    /* If GGUF reorders v-head rows, convert xb2 from GGUF order to sequential order */
    if (do_remap) {
        float *xb2_tmp = alloca(c->ssm_d_inner * sizeof(float));
        memcpy(xb2_tmp, s->xb2, c->ssm_d_inner * sizeof(float));
        for (int h = 0; h < n_v_heads; h++) {
            int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
            memcpy(s->xb2 + h * head_v_dim, xb2_tmp + gh * head_v_dim, head_v_dim * sizeof(float));
        }
        if (il == 0 && pos == 0) {
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
        float *v_conv_tmp = alloca(c->ssm_d_inner * sizeof(float));
        memcpy(v_conv_tmp, v_conv, c->ssm_d_inner * sizeof(float));
        for (int h = 0; h < n_v_heads; h++) {
            int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
            memcpy(v_conv + h * head_v_dim, v_conv_tmp + gh * head_v_dim, head_v_dim * sizeof(float));
        }
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
    if (il == 0 && pos == 0) {
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
    float *alpha_out = tmp + conv_dim + 2 * qk_dim + c->ssm_d_inner; /* [dt_rank] */
    {
        gguf_type_t alpha_type = lw->type_ssm_alpha;
        size_t row_bytes = gguf_type_row_size(alpha_type, dim);
        int alpha_map[n_v_heads];
        for (int h = 0; h < n_v_heads; h++) alpha_map[h] = do_remap ? qwen35_vhead_gguf(h, n_vpk, n_k) : h;
#ifdef PICOLM_GPU
        if (gpu_lw && (alpha_type == GGUF_TYPE_Q4_0 || alpha_type == GGUF_TYPE_Q8_0) &&
            picolm_gpu_ssm_vecdot(alpha_out, s->xb, lw->ssm_alpha, alpha_type, dim,
                                  n_v_heads, (int)row_bytes, alpha_map, m->gpu.device)) {
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
        int beta_map[n_v_heads];
        for (int h = 0; h < n_v_heads; h++) beta_map[h] = do_remap ? qwen35_vhead_gguf(h, n_vpk, n_k) : h;
#ifdef PICOLM_GPU
        if (gpu_lw && (beta_type == GGUF_TYPE_Q4_0 || beta_type == GGUF_TYPE_Q8_0) &&
            picolm_gpu_ssm_vecdot(beta, s->xb, lw->ssm_beta, beta_type, dim,
                                  n_v_heads, (int)row_bytes, beta_map, m->gpu.device)) {
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
    if (gpu_lw) {
        /* GPU SSM recurrence kernel */
        if (!picolm_gpu_ssm_recurrence(state, q_conv, k_conv, v_conv,
                                        gate_exp, beta, ssm_output,
                                        n_v_heads, d_state, repeat, m->gpu.device)) {
            /* Fall back to CPU */
            tensor_parallel_for(n_v_heads, ssm_head_task, &ssm_ctx);
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
    for (int h = 0; h < n_v_heads; h++) {
        float nrm = 0.0f;
        for (int d = 0; d < head_v_dim; d++) {
            float v = ssm_output[d * n_v_heads + h];
            nrm += v * v;
        }
        nrm = 1.0f / sqrtf(nrm / head_v_dim + eps);
        for (int d = 0; d < head_v_dim; d++) {
            float v = ssm_output[d * n_v_heads + h];
            float zv = s->xb2[h * head_v_dim + d];
            float silu_z = zv * (1.0f / (1.0f + expf(-zv)));
            final_output[h * head_v_dim + d] = v * nrm * norm_w[d] * silu_z;
        }
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) {
        dbg_vec("xb2[:8]", s->xb2, 8, 8);
        dbg_vec("final_out[:8]", final_output, 8, 8);
    }
#endif

    /* 19. Reshape to [value_dim] and output projection */
    /* final_output is [head_v_dim * n_v_heads] = [value_dim] = [4096] */
    /* ssm_out: [n_embd, value_dim] - GGUF columns may be reordered */
    if (do_remap) {
        /* Reorder final_output to GGUF column order before matmul, then matmul,
         * then reorder result back. Actually easier: reorder final_output to GGUF
         * order, matmul, and the result is already correct because the weight
         * columns and input elements are matched. */
        float *fo_gguf = alloca(c->ssm_d_inner * sizeof(float));
        for (int h = 0; h < n_v_heads; h++) {
            int gh = qwen35_vhead_gguf(h, n_vpk, n_k);
            memcpy(fo_gguf + gh * head_v_dim, final_output + h * head_v_dim, head_v_dim * sizeof(float));
        }
        matmul(residual, fo_gguf, lw->ssm_out, c->ssm_d_inner, dim, lw->type_ssm_out);
    } else {
        matmul(residual, final_output, lw->ssm_out, c->ssm_d_inner, dim, lw->type_ssm_out);
    }
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("residual[:8]", residual, 8, 8);
#endif
    /* 20. Residual add */
    vec_add(x, residual, dim);

    /* 21. Post-attention norm + FFN (only if MLP weights exist for this layer) */
    if (lw->ffn_gate && lw->ffn_up && lw->ffn_down) {
        rmsnorm(s->xb, x, s->post_attn_norm_w[il], dim, eps);
        matmul(s->hb, s->xb, lw->ffn_gate, dim, c->n_ffn, lw->type_ffn_gate);
        matmul(s->hb2, s->xb, lw->ffn_up, dim, c->n_ffn, lw->type_ffn_up);
        silu(s->hb, c->n_ffn);
        elemwise_mul(s->hb, s->hb, s->hb2, c->n_ffn);
        matmul(s->xb, s->hb, lw->ffn_down, c->n_ffn, dim, lw->type_ffn_down);
        vec_add(x, s->xb, dim);
    }
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

    if (m->state.mem_block) {
        free(m->state.mem_block);
        m->state.mem_block = NULL;
    }
    if (m->state.kv_block) {
        free(m->state.kv_block);
        m->state.kv_block = NULL;
    }
    munmap_file(m);
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
                gbytes / (1024.0 * 1024.0));
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
                layer_weight_bytes(m, 0) / (1024.0 * 1024.0) + gbytes / (1024.0 * 1024.0),
                mem_bytes / (1024.0 * 1024.0));
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
            ranges_size[n_ranges] = (const uint8_t *)ranges_end[n_ranges] - (const uint8_t *)ranges_start[n_ranges]; \
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

    /* Clamp range ends to the mmap boundary. page_align_up can push the
     * end past the file size when the last tensor is near EOF. */
    const uint8_t *mmap_end = (const uint8_t *)m->mmap_addr + m->mmap_size;
    mmap_end = (const uint8_t *)(((uintptr_t)mmap_end + 4095) & ~(uintptr_t)4095);
    for (int i = 0; i < merged; i++) {
        if ((const uint8_t *)ranges_end[i] > mmap_end)
            ranges_end[i] = mmap_end;
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
        size_t sz = (const uint8_t *)ranges_end[i] - (const uint8_t *)ranges_start[i];
#ifdef _WIN32
        if (VirtualLock((void *)ranges_start[i], sz) == 0) {
            fprintf(stderr, "Lock: VirtualLock failed (error %lu)\n", (unsigned long)GetLastError());
            return 0;
        }
#else
        if (mlock((void *)ranges_start[i], sz) != 0) {
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
            total_locked / (1024.0 * 1024.0),
            layers_locked - 1, c->n_layers - 1);
    return layers_locked;
}
#undef ADD_RANGE
#undef NR_MAX

/* Unlock previously pinned weight layers. */
int model_unlock_layers(model_t *m) {
    if (m->locked_layers == 0) return 0;

    const model_config_t *c = &m->config;
    size_t gbytes = global_weight_bytes(m);
    size_t budget = gbytes;
    for (int l = 0; l < m->locked_layers; l++)
        budget += layer_weight_bytes(m, l);

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
        size_t sz = (const uint8_t *)ranges_end[i] - (const uint8_t *)ranges_start[i];
#ifdef _WIN32
        VirtualUnlock((void *)ranges_start[i], sz);
#else
        munlock((void *)ranges_start[i], sz);
#endif
        total_unlocked += sz;
    }

    m->locked_layers = 0;
    fprintf(stderr, "Unlock: released %.1f MB\n", total_unlocked / (1024.0 * 1024.0));
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
    const uint8_t *kcache, *vcache;
    const float *q_batch;   /* [n_tokens][n_heads * head_dim] */
    float *xb_batch;        /* [n_tokens][xb_stride] */
    int xb_stride;
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
              ctx->n_kv_heads, ctx->kv_type_k, ctx->kv_type_v,
              ctx->kv_row_size_k, ctx->kv_row_size_v, ctx->head_dim);
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
        size_t kv_row_size_k, size_t kv_row_size_v);

static void batch_attention_layer(
        float *xb_batch, const float *q_batch,
        const uint8_t *kcache, const uint8_t *vcache,
        int n_tokens, int start_pos,
        int n_heads, int n_kv_heads, int head_dim,
        int xb_stride,
        int kv_type_k, int kv_type_v,
        size_t kv_row_size_k, size_t kv_row_size_v)
{
    /* Build the prefill_attn_ctx for both the original path and the test */
    prefill_attn_ctx_t ctx;
    ctx.n_heads = n_heads; ctx.n_kv_heads = n_kv_heads; ctx.kv_mul = n_heads / n_kv_heads;
    ctx.head_dim = head_dim; ctx.start_pos = start_pos;
    ctx.kv_type_k = kv_type_k; ctx.kv_type_v = kv_type_v;
    ctx.kv_row_size_k = kv_row_size_k; ctx.kv_row_size_v = kv_row_size_v;
    ctx.kcache = kcache; ctx.vcache = vcache;
    ctx.q_batch = q_batch; ctx.xb_batch = xb_batch; ctx.xb_stride = xb_stride;

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
                              kv_row_size_k, kv_row_size_v);
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

/* Map kv_cache_type_t (0/1/2) to gguf_type_t (1/8/2) for matmul_batch */
static gguf_type_t kv_cache_to_gguf_type(kv_cache_type_t kv_type) {
    switch (kv_type) {
        case KV_CACHE_F16:  return GGUF_TYPE_F16;
        case KV_CACHE_Q8_0: return GGUF_TYPE_Q8_0;
        case KV_CACHE_Q4_0: return GGUF_TYPE_Q4_0;
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
    size_t kv_row_size_k;   /* bytes per KV head row in cache */
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

    /* Extract K-tile from interleaved KV cache into contiguous scratch.
     * KV cache layout: [pos][n_kv_heads][kv_row_size_k]
     * For positions [kv_tile_start, kv_tile_start+ts), head kv_h: */
    {
        size_t rb = t->kv_row_size_k;
        int gguf_k = t->kv_gguf_k;
        size_t k_rb_gguf = gguf_type_row_size(gguf_k, hd);
        /* K tile: ts positions, each rb bytes from cache, copied to
         * contiguous buffer with stride k_rb_gguf. For F16 this is
         * the same size and just a memcpy; for Q8_0/Q4_0 also same. */
        for (int p = 0; p < ts; p++) {
            const uint8_t *src = t->kcache + (size_t)(kv_tile_start + p) * t->n_kv_heads * rb
                               + t->kv_h * rb;
            uint8_t *dst = (uint8_t *)t->tile_k + (size_t)p * k_rb_gguf;
            memcpy(dst, src, rb);
        }
    }

    /* QK^T: matmul_batch(scores, q_rows, n_q, tile_k, hd, ts, kv_gguf_k)
     * out layout: [n_q][ts], scores[b*ts + i] = row b, col i */
    matmul_batch(t->scores, t->q_rows, n_q, t->tile_k, hd, ts, t->kv_gguf_k);

    /* Scale scores by 1/sqrt(head_dim) */
    float inv_sqrt_hd = 1.0f / sqrtf((float)hd);
    for (int i = 0; i < n_q * ts; i++)
        t->scores[i] *= inv_sqrt_hd;

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
        size_t kv_row_size_k, size_t kv_row_size_v)
{
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
                        for (int p = 0; p < this_tile_size; p++) {
                            const uint8_t *src = vcache + (size_t)(kv_t0 + p) * n_kv_heads * rb
                                               + kv_h * rb;
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
                    task.kcache = kcache;
                    task.q_rows = q_rows;
                    task.scores = scores;
                    task.tile_k = tile_k_buf;
                    task.tile_v_f32 = tile_v_f32;
                    task.M = M;
                    task.S = S;
                    task.acc = acc;

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
 * Batch prefill: all prompt tokens processed at once.
 * Projection matmuls batched (weights read once). Attention batched.
 * ================================================================ */

float *model_forward_prefill(model_t *m, const int *tokens, int n_tokens, int start_pos) {
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;
    int dim = c->n_embd, n_ffn = c->n_ffn;
    int n_heads = c->n_heads, n_kv_heads = c->n_kv_heads, head_dim = c->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int q_dim = n_heads * head_dim, seq_len = c->max_seq_len;
    /* Qwen3.5 full attention layers have interleaved Q+gate */
    int q_full_dim = (c->has_ssm) ? (q_dim * 2) : q_dim;
    int max_dim = (q_full_dim > dim) ? q_full_dim : dim;
    size_t bs = (size_t)n_tokens;

    size_t sz = bs * (2 * dim + max_dim + q_full_dim + 2 * kv_dim + 2 * n_ffn);
    float *buf = (float *)malloc(sz * sizeof(float));
    if (!buf) { fprintf(stderr, "OOM: prefill batch\n"); exit(1); }
    float *p = buf;
    float *x_batch = p;  p += bs * dim;
    float *xb_batch = p; p += bs * max_dim;
    float *xb2_batch = p; p += bs * dim;
    float *q_batch = p;  p += bs * q_full_dim;
    float *k_batch = p;  p += bs * kv_dim;
    float *v_batch = p;  p += bs * kv_dim;
    float *hb_batch = p; p += bs * n_ffn;
    float *hb2_batch = p;

    /* Embedding lookup */
    {
        if (w->type_token_embd == GGUF_TYPE_Q4_0_4_4) {
            /* Q4_0_4_4: rows are interleaved in groups of 4 */
            int nb = dim / 32;
            size_t group_bytes = (size_t)nb * sizeof(block_q4_0x4);
            #ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int bi = 0; bi < n_tokens; bi++) {
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
        } else {
            size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
            #ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int bi = 0; bi < n_tokens; bi++) {
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
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];

        if (c->has_ssm && !lw->is_attn_layer) {
            /* SSM layer: process tokens sequentially (stateful: conv_state + ssm_state) */
            for (int bi = 0; bi < n_tokens; bi++) {
                /* Copy batch token into s->x, call ssm_forward which does residual add */
                memcpy(s->x, x_batch + bi * dim, dim * sizeof(float));
                float *ssm_residual = s->xb2;
#ifdef PICOLM_GPU
                ssm_forward(m, s, s->x, ssm_residual, lw, l, start_pos + bi, &m->gpu.layers[l]);
#else
                ssm_forward(m, s, s->x, ssm_residual, lw, l, start_pos + bi, NULL);
#endif
                /* Copy result back to batch buffer */
                memcpy(x_batch + bi * dim, s->x, dim * sizeof(float));
            }
            continue;
        }

        /* RMSNorm */
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int bi = 0; bi < n_tokens; bi++)
            rmsnorm(xb_batch + bi * dim, x_batch + bi * dim, s->attn_norm_w[l], dim, c->rms_norm_eps);

        /* Q projection (batched) */
        tensor_set_repacked(m->repack_used[2+l*9] ? m->repack_buffers[2+l*9] : NULL);
#ifdef PICOLM_GPU
        if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)m->gpu.layers[l].attn_q, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
        { int this_q_dim = (c->has_ssm && lw->is_attn_layer) ? q_full_dim : q_dim;
          matmul_batch(q_batch, xb_batch, n_tokens, lw->attn_q, dim, this_q_dim, lw->type_attn_q);
        }
        tensor_set_repacked(NULL);

        /* For Qwen3.5: de-interleave per-head Q+gate into block layout.
         * GGUF stores [Q_0, Gate_0, Q_1, Gate_1, ...] (per-head interleaved).
         * Gate stored in hb2_batch to survive K/V projection writes. */
        { float *qwen35_gate_batch = NULL;
          if (c->has_ssm && lw->is_attn_layer) {
              qwen35_gate_batch = hb2_batch; /* reuse as gate storage */
              #ifdef _OPENMP
#pragma omp parallel for
#endif
              for (int bi = 0; bi < n_tokens; bi++) {
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
            uint8_t *kcl = s->key_cache + (size_t)this_attn_ord * seq_len * c->n_kv_heads * s->kv_row_size_k;
            uint8_t *vcl = s->val_cache + (size_t)this_attn_ord * seq_len * c->n_kv_heads * s->kv_row_size_v;
            #ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int bi = 0; bi < n_tokens; bi++) {
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
                    for (int h = 0; h < n_heads; h++)
                        rmsnorm(q_pos + h * head_dim, q_pos + h * head_dim, qnw, head_dim, c->rms_norm_eps);
                    for (int h = 0; h < n_kv_heads; h++)
                        rmsnorm(k_pos + h * head_dim, k_pos + h * head_dim, knw, head_dim, c->rms_norm_eps);
                }

                int rope_dim_pf = (c->rope_dim > 0) ? c->rope_dim : head_dim;
                int rope_half_pf = rope_dim_pf / 2;
                rope(q_pos, k_pos, head_dim, n_heads, n_kv_heads, cos_pos, sin_pos, c->rope_type, rope_half_pf);

                /* KV cache store */
                for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                    uint8_t *kp = kcl + (size_t)pos * c->n_kv_heads * s->kv_row_size_k + hkv * s->kv_row_size_k;
                    uint8_t *vp = vcl + (size_t)pos * c->n_kv_heads * s->kv_row_size_v + hkv * s->kv_row_size_v;
                    for (int d2 = 0; d2 < head_dim; d2++)
                        ((uint16_t*)kp)[d2] = fp32_to_fp16(k_pos[hkv * head_dim + d2]);
                    for (int d2 = 0; d2 < head_dim; d2++)
                        ((uint16_t*)vp)[d2] = fp32_to_fp16(v_pos[hkv * head_dim + d2]);
                }
            }

            /* Zero init xb_batch for attention accumulation */
            memset(xb_batch, 0, (size_t)n_tokens * max_dim * sizeof(float));
            /* Batched attention: all tokens, all heads, one thread dispatch */
            batch_attention_layer(xb_batch, q_batch, kcl, vcl,
                                  n_tokens, start_pos,
                                  n_heads, c->n_kv_heads, head_dim,
                                  max_dim, (int)s->kv_type_k, (int)s->kv_type_v,
                                  s->kv_row_size_k, s->kv_row_size_v);
        }

        /* Apply Qwen3.5 attention gate (sigmoid, element-wise multiply on attention output before proj) */
        if (c->has_ssm && lw->is_attn_layer) {
            #ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int bi = 0; bi < n_tokens; bi++) {
                float *xb = xb_batch + bi * max_dim;
                float *gate = hb2_batch + bi * q_dim;
                for (int i = 0; i < q_dim; i++) {
                    float g = 1.0f / (1.0f + expf(-gate[i]));
                    xb[i] *= g;
                }
            }
        }

        /* Output projection (batched) */
        tensor_set_repacked(m->repack_used[6+l*9] ? m->repack_buffers[6+l*9] : NULL);
#ifdef PICOLM_GPU
        if (gpu_ok) tensor_set_gpu_tensor((picolm_gpu_tensor_t *)m->gpu.layers[l].attn_output, gpu_dev); else tensor_set_gpu_tensor(NULL, 0);
#endif
        matmul_batch(xb2_batch, xb_batch, n_tokens, lw->attn_output, q_dim, dim, lw->type_attn_output);
        tensor_set_repacked(NULL);

        /* Residual: x += attn_out */
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int bi = 0; bi < n_tokens; bi++) {
            float *a = x_batch + bi * dim, *b = xb2_batch + bi * dim;
            for (int d2 = 0; d2 < dim; d2++) a[d2] += b[d2];
        }

        /* FFN RMSNorm */
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int bi = 0; bi < n_tokens; bi++)
            rmsnorm(xb_batch + bi * dim, x_batch + bi * dim, s->post_attn_norm_w[l], dim, c->rms_norm_eps);

        /* FFN gate+up (batched dual) */
        tensor_set_repacked(m->repack_used[7+l*9] ? m->repack_buffers[7+l*9] : NULL);
        matmul_dual_batch(hb_batch, hb2_batch, xb_batch, n_tokens,
                          lw->ffn_gate, lw->ffn_up, dim, n_ffn,
                          lw->type_ffn_gate, lw->type_ffn_up);
        tensor_set_repacked(NULL);

        /* SiLU + mul */
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int bi = 0; bi < n_tokens; bi++) {
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int bi = 0; bi < n_tokens; bi++) {
            float *a = x_batch + bi * dim, *b = xb2_batch + bi * dim;
            for (int d2 = 0; d2 < dim; d2++) a[d2] += b[d2];
        }
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
 * KV Cache Persistence — save/load KV state to skip prompt prefill
 *
 * File format:
 *   [4 bytes] magic: KVCACHE_MAGIC
 *   [4 bytes] n_pos: number of cached positions
 *   [4 bytes] n_layers
 *   [4 bytes] kv_dim (n_kv_heads * head_dim)
 *   [N bytes] key_cache FP16 data (n_layers * n_pos * kv_dim * sizeof(uint16_t))
 *   [N bytes] val_cache FP16 data (same size)
 * ================================================================ */

int kvcache_save(const model_t *m, const char *path, int n_pos) {
    const model_config_t *c = &m->config;
    const run_state_t *s = &m->state;
    int seq_len = c->max_seq_len;

    if (n_pos <= 0 || n_pos > seq_len) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "kvcache_save: cannot open %s\n", path);
        return -1;
    }

    /* Version 2 header */
    uint32_t header[7] = {
        KVCACHE_MAGIC | 0x00000002,
        (uint32_t)n_pos,
        (uint32_t)c->n_layers,
        (uint32_t)c->n_kv_heads,
        (uint32_t)c->head_dim,
        (uint32_t)s->kv_type_k,
        (uint32_t)s->kv_type_v
    };
    fwrite(header, sizeof(uint32_t), 7, f);

    size_t pos_stride_k = (size_t)c->n_kv_heads * s->kv_row_size_k;
    size_t pos_stride_v = (size_t)c->n_kv_heads * s->kv_row_size_v;
    for (int l = 0; l < c->n_layers; l++) {
        const uint8_t *kcache_l = s->key_cache + (size_t)l * seq_len * pos_stride_k;
        for (int p = 0; p < n_pos; p++) {
            fwrite(kcache_l + (size_t)p * pos_stride_k, 1, pos_stride_k, f);
        }
    }
    for (int l = 0; l < c->n_layers; l++) {
        const uint8_t *vcache_l = s->val_cache + (size_t)l * seq_len * pos_stride_v;
        for (int p = 0; p < n_pos; p++) {
            fwrite(vcache_l + (size_t)p * pos_stride_v, 1, pos_stride_v, f);
        }
    }

    fclose(f);
    fprintf(stderr, "KV cache saved: %d positions to %s\n", n_pos, path);
    return 0;
}

int kvcache_load(model_t *m, const char *path) {
    const model_config_t *c = &m->config;
    run_state_t *s = &m->state;
    int seq_len = c->max_seq_len;

    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    uint32_t header[7];
    if (fread(header, sizeof(uint32_t), 7, f) != 7) {
        fclose(f);
        return 0;
    }

    uint32_t magic_base = header[0] & ~0x00000003;
    if (magic_base != KVCACHE_MAGIC) {
        fprintf(stderr, "kvcache_load: invalid magic\n");
        fclose(f);
        return 0;
    }

    int n_pos = (int)header[1];
    int file_layers = (int)header[2];
    int file_n_kv_heads = (int)header[3];
    int file_head_dim = (int)header[4];

    if (file_layers != c->n_layers || file_n_kv_heads != c->n_kv_heads ||
        file_head_dim != c->head_dim) {
        fprintf(stderr, "kvcache_load: model mismatch (layers=%d/%d, kv_heads=%d/%d, head_dim=%d/%d)\n",
                file_layers, c->n_layers, file_n_kv_heads, c->n_kv_heads,
                file_head_dim, c->head_dim);
        fclose(f);
        return 0;
    }
    if (n_pos > seq_len) {
        fprintf(stderr, "kvcache_load: cached %d positions exceeds max_seq_len %d\n",
                n_pos, seq_len);
        fclose(f);
        return 0;
    }

    if (header[5] != s->kv_type_k || header[6] != s->kv_type_v) {
        fprintf(stderr, "kvcache_load: KV type mismatch (file k=%d v=%d vs current k=%d v=%d)\n",
                header[5], header[6], s->kv_type_k, s->kv_type_v);
        fclose(f);
        return 0;
    }

    size_t pos_stride_k = (size_t)c->n_kv_heads * s->kv_row_size_k;
    size_t pos_stride_v = (size_t)c->n_kv_heads * s->kv_row_size_v;
    for (int l = 0; l < c->n_layers; l++) {
        uint8_t *kcache_l = s->key_cache + (size_t)l * seq_len * pos_stride_k;
        for (int p = 0; p < n_pos; p++) {
            if (fread(kcache_l + (size_t)p * pos_stride_k, 1, pos_stride_k, f) != pos_stride_k) {
                fclose(f);
                return 0;
            }
        }
    }
    for (int l = 0; l < c->n_layers; l++) {
        uint8_t *vcache_l = s->val_cache + (size_t)l * seq_len * pos_stride_v;
        for (int p = 0; p < n_pos; p++) {
            if (fread(vcache_l + (size_t)p * pos_stride_v, 1, pos_stride_v, f) != pos_stride_v) {
                fclose(f);
                return 0;
            }
        }
    }

    fclose(f);
    fprintf(stderr, "KV cache loaded: %d positions from %s\n", n_pos, path);
    return n_pos;
}

/* ---- SSM State Checkpointing (Qwen3.5/3.6) ---- */

/* Compute total bytes needed to snapshot all SSM layer states.
 * Each SSM layer has:
 *   ssm_conv_state[l]: (d_conv-1) * conv_dim floats
 *   ssm_state[l]:      d_state * d_inner floats
 * Returns 0 if model has no SSM layers. */
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
