#include "model.h"
#include "tensor.h"
#include "quant.h"

#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#endif

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
    return v;
}

static int32_t read_i32(reader_t *r) {
    int32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(reader_t *r) {
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static float read_f32(reader_t *r) {
    float v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
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

static void prepare_mmap(const void *addr, size_t size) {
#ifdef PICOLM_PREFAULT
    const volatile char *p = (const volatile char *)addr;
    for (size_t off = 0; off < size; off += 4096)
        (void)p[off];
#endif
    (void)addr; (void)size;
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

    void *addr = mmap(NULL, m->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
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

        if (str_eq(key, "llama.embedding_length") || str_eq(key, "general.embedding_length")
            || str_eq(key, "qwen2.embedding_length") || str_eq(key, "qwen3.embedding_length") || str_eq(key, "qwen35.embedding_length")) {
            int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
            if (key.str[0] == 'q') cfg->rope_type = 1;  /* qwen2/qwen3/qwen35 interleaved */
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
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "llama.attention.layer_norm_rms_epsilon")
            || str_eq(key, "qwen2.attention.layer_norm_rms_epsilon")
            || str_eq(key, "qwen3.attention.layer_norm_rms_epsilon") || str_eq(key, "qwen35.attention.layer_norm_rms_epsilon")) {
            /* Read epsilon from GGUF (usually F64 type=3) */
            if (vtype == 3) { /* F64 */
                double val;
                memcpy(&val, r.data + r.pos, 8); r.pos += 8;
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
                    lw->ssm_a = ptr;
                } else if (strcmp(suffix, "ssm_alpha.weight") == 0) {
                    lw->ssm_alpha = ptr; lw->type_ssm_alpha = qtype;
                } else if (strcmp(suffix, "ssm_beta.weight") == 0) {
                    lw->ssm_beta = ptr; lw->type_ssm_beta = qtype;
                } else if (strcmp(suffix, "ssm_conv1d.weight") == 0) {
                    lw->ssm_conv1d = ptr;
                } else if (strcmp(suffix, "ssm_dt.bias") == 0) {
                    lw->ssm_dt = ptr;
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
        s->ssm_tmp = p;
        p += (ssm_conv_dim * 3 + c->ssm_d_state * c->ssm_n_group * 2 +
              c->ssm_d_inner * 3 + c->ssm_dt_rank * 4);
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
                    /* GGUF: already A */
                    memcpy(s->ssm_a_w[l], (const float *)lw->ssm_a, c->ssm_dt_rank * sizeof(float));
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

    return 0;
}

/* ---- Public API ---- */

int model_load(model_t *m, const char *path, int max_seq_len, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v) {
    memset(m, 0, sizeof(*m));

    if (mmap_file(m, path) != 0) return -1;
    if (parse_gguf(m, max_seq_len) != 0) return -1;

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
                        layer_weights_t *lw, int il, int pos);

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
        } else {
            const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
            dequantize_row(embd_row, s->x, dim, w->type_token_embd);

        }
    }

    /* 2. Transformer layers */
    int attn_ordinal = 0;
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];
        int ri = 2 + l * 9;

        if (c->has_ssm && !lw->is_attn_layer) {
            /* SSM layer (Qwen3.5) */
            float *ssm_residual = s->xb2; /* use xb2 as residual buffer */
            ssm_forward(m, s, s->x, ssm_residual, lw, l, pos);
            continue;
        }

        /* ---- Attention ---- */
        rmsnorm(s->xb, s->x, s->attn_norm_w[l], dim, c->rms_norm_eps);

        /* Q projection (Q+gate joint for Qwen3.5 full attention) */
        tensor_set_repacked(m->repack_used[ri] ? m->repack_buffers[ri] : NULL);
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
                memcpy(q_block + h * head_dim, qg_raw + h * 2 * head_dim,
                       head_dim * sizeof(float));
                memcpy(gate_block + h * head_dim, qg_raw + h * 2 * head_dim + head_dim,
                       head_dim * sizeof(float));
            }
            qwen35_attn_gate = gate_block;
        }

        /* K projection */
        tensor_set_repacked(m->repack_used[ri+1] ? m->repack_buffers[ri+1] : NULL);
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
        for (int h = 0; h < n_heads; h++) {
            float *qh = s->q + h * head_dim;
            int kv_h = h / kv_mul;
            float *xbh = s->xb + h * head_dim;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            /* Accumulator for weighted V values */
            float acc[256]; /* head_dim is typically 64-128 */
            memset(acc, 0, (size_t)head_dim * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                /* Compute score: dot(Q_h, K_t) / sqrt(head_dim) */
                const uint8_t *kt = kcache_layer + (size_t)t * c->n_kv_heads * s->kv_row_size_k
                                   + kv_h * s->kv_row_size_k;
                float score;
                if (s->kv_type_k == KV_CACHE_Q8_0) {
                    score = vec_dot_q8_0_f32(kt, qh, head_dim);
                } else if (s->kv_type_k == KV_CACHE_Q4_0) {
                    score = vec_dot_q4_0_f32(kt, qh, head_dim);
                } else {
                    score = vec_dot_f16_f32(kt, qh, head_dim);
                }
                score /= sqrtf((float)head_dim);

                /* Online softmax update */
                const uint8_t *vt = vcache_layer + (size_t)t * c->n_kv_heads * s->kv_row_size_v
                                   + kv_h * s->kv_row_size_v;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    /* FMA: acc = acc * correction + dequant(vt) */
                    if (s->kv_type_v == KV_CACHE_Q8_0) {
                        fma_scale_q8_0_f32(acc, correction, vt, head_dim);
                    } else if (s->kv_type_v == KV_CACHE_Q4_0) {
                        fma_scale_q4_0_f32(acc, correction, vt, head_dim);
                    } else {
                        const uint16_t *vt16 = (const uint16_t *)vt;
#ifdef PICOLM_FP16_HW
                        { float32x4_t cv = vdupq_n_f32(correction); int d = 0;
                          for (; d + 3 < head_dim; d += 4) {
                              float32x4_t a = vld1q_f32(acc + d);
                              vst1q_f32(acc + d, vmlaq_f32(fp16x4_to_f32_hw(vt16 + d), a, cv));
                          }
                          for (; d < head_dim; d++)
                              acc[d] = acc[d] * correction + fp16_to_fp32(vt16[d]);
                        }
#else
                        for (int d = 0; d < head_dim; d++) {
                            acc[d] = acc[d] * correction + fp16_to_fp32(vt16[d]);
                        }
#endif
                    }
                    max_score = score;
                } else {
                    float w = expf(score - max_score);
                    sum_exp += w;
                    /* acc += w * dequant(vt) */
                    if (s->kv_type_v == KV_CACHE_Q8_0) {
                        scale_add_q8_0_f32(acc, w, vt, head_dim);
                    } else if (s->kv_type_v == KV_CACHE_Q4_0) {
                        scale_add_q4_0_f32(acc, w, vt, head_dim);
                    } else {
                        const uint16_t *vt16 = (const uint16_t *)vt;
#ifdef PICOLM_FP16_HW
                        { float32x4_t wv = vdupq_n_f32(w); int d = 0;
                          for (; d + 3 < head_dim; d += 4) {
                              float32x4_t a = vld1q_f32(acc + d);
                              vst1q_f32(acc + d, vmlaq_f32(a, fp16x4_to_f32_hw(vt16 + d), wv));
                          }
                          for (; d < head_dim; d++)
                              acc[d] += w * fp16_to_fp32(vt16[d]);
                        }
#else
                        for (int d = 0; d < head_dim; d++) {
                            acc[d] += w * fp16_to_fp32(vt16[d]);
                        }
#endif
                    }
                }
            }



            /* Normalize */
            float inv_sum = 1.0f / sum_exp;
#ifdef PICOLM_NEON
            {
                float32x4_t inv = vdupq_n_f32(inv_sum);
                int d = 0;
                for (; d + 3 < head_dim; d += 4) {
                    float32x4_t af = vld1q_f32(acc + d);
                    vst1q_f32(xbh + d, vmulq_f32(af, inv));
                }
                for (; d < head_dim; d++) xbh[d] = acc[d] * inv_sum;
            }
#elif defined(PICOLM_AVX)
            {
                __m256 inv = _mm256_set1_ps(inv_sum);
                int d = 0;
                for (; d + 7 < head_dim; d += 8) {
                    __m256 af = _mm256_loadu_ps(acc + d);
                    _mm256_storeu_ps(xbh + d, _mm256_mul_ps(af, inv));
                }
                for (; d < head_dim; d++) xbh[d] = acc[d] * inv_sum;
            }
#elif defined(PICOLM_SSE2)
            {
                __m128 inv = _mm_set1_ps(inv_sum);
                int d = 0;
                for (; d + 3 < head_dim; d += 4) {
                    __m128 af = _mm_loadu_ps(acc + d);
                    _mm_storeu_ps(xbh + d, _mm_mul_ps(af, inv));
                }
                for (; d < head_dim; d++) xbh[d] = acc[d] * inv_sum;
            }
#else
            for (int d = 0; d < head_dim; d++) xbh[d] = acc[d] * inv_sum;
#endif
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
        matmul(s->xb2, s->xb, lw->attn_output, q_dim, dim, lw->type_attn_output);
        tensor_set_repacked(NULL);
        vec_add(s->x, s->xb2, dim);

        /* ---- FFN (SwiGLU) - only if MLP weights exist for this layer ---- */
        if (lw->ffn_gate && lw->ffn_up && lw->ffn_down) {
            rmsnorm(s->xb, s->x, s->post_attn_norm_w[l], dim, c->rms_norm_eps);

            tensor_set_repacked(m->repack_used[ri+4] ? m->repack_buffers[ri+4] : NULL);
            matmul(s->hb,  s->xb, lw->ffn_gate, dim, n_ffn, lw->type_ffn_gate);
            tensor_set_repacked(NULL);

            tensor_set_repacked(m->repack_used[ri+6] ? m->repack_buffers[ri+6] : NULL);
            matmul(s->hb2, s->xb, lw->ffn_up,   dim, n_ffn, lw->type_ffn_up);
            tensor_set_repacked(NULL);

            silu(s->hb, n_ffn);
            elemwise_mul(s->hb, s->hb, s->hb2, n_ffn);

            tensor_set_repacked(m->repack_used[ri+5] ? m->repack_buffers[ri+5] : NULL);
            matmul(s->xb, s->hb, lw->ffn_down, n_ffn, dim, lw->type_ffn_down);
            tensor_set_repacked(NULL);
            vec_add(s->x, s->xb, dim);
        }
    }

    /* 3. Final RMSNorm */
    rmsnorm(s->x, s->x, s->output_norm_w, dim, c->rms_norm_eps);

    /* 4. Output projection -> logits */
    tensor_set_repacked(m->repack_used[1] ? m->repack_buffers[1] : NULL);
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

/* SSM layer forward pass (autoregressive, single token) */
static void ssm_forward(model_t *m, run_state_t *s, float *x, float *residual,
                        layer_weights_t *lw, int il, int pos) {
    model_config_t *c = &m->config;
    int dim = c->n_embd;
    int d_conv = c->ssm_d_conv;
    int d_state = c->ssm_d_state;
    int n_k_heads = c->ssm_n_group;
    int n_v_heads = c->ssm_dt_rank;
    int conv_dim = 2 * d_state * n_k_heads + c->ssm_d_inner;
    int head_v_dim = c->ssm_d_inner / n_v_heads;
    float eps = c->rms_norm_eps;

    /* Scratch space: dedicated SSM buffer */
    float *tmp = s->ssm_tmp;

    /* 1. RMSNorm (attn_norm) */
    rmsnorm(s->xb, x, s->attn_norm_w[il], dim, eps);
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("xb[:8]", s->xb, 8, 8);
#endif

    /* 2. QKV projection: qkv_mixed = matmul(attn_qkv, xb) -> [conv_dim] */
    matmul(s->q, s->xb, lw->attn_qkv, dim, conv_dim, lw->type_attn_qkv);
#ifdef DEBUG_SSM
    if (il == 0 && pos == 0) dbg_vec("qkv[:8]", s->q, 8, 8);
#endif

    /* 3. Z gate: z = matmul(attn_gate_ssm, xb) -> [value_dim] */
    matmul(s->xb2, s->xb, lw->attn_gate_ssm, dim, c->ssm_d_inner, lw->type_attn_gate_ssm);

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

    /* 9. Alpha: alpha = matmul(ssm_alpha, xb) + ssm_dt.bias -> [dt_rank] */
    /* GGUF stores [dim, n_v_heads] column-major: each head has dim contiguous elements */
    float *alpha_out = tmp + conv_dim + 2 * qk_dim + c->ssm_d_inner; /* [dt_rank] */
    {
        gguf_type_t alpha_type = lw->type_ssm_alpha;
        size_t row_bytes = gguf_type_row_size(alpha_type, dim);
        if (alpha_type == GGUF_TYPE_F32) {
            for (int h = 0; h < n_v_heads; h++) {
                float vd = vec_dot((const uint8_t *)lw->ssm_alpha + (size_t)h * row_bytes,
                                   s->xb, dim, alpha_type);
                alpha_out[h] = vd + s->ssm_dt_w[il][h];
            }
        } else {
            for (int h = 0; h < n_v_heads; h++) {
                const uint8_t *head_data = (const uint8_t *)lw->ssm_alpha + (size_t)h * row_bytes;
                float sum = vec_dot(head_data, s->xb, dim, alpha_type);
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
    /* GGUF stores [dim, n_v_heads] column-major: each head has dim contiguous elements */
    float *beta = gate + n_v_heads; /* [dt_rank] */
    {
        gguf_type_t beta_type = lw->type_ssm_beta;
        size_t row_bytes = gguf_type_row_size(beta_type, dim);
        if (beta_type == GGUF_TYPE_F32) {
            for (int h = 0; h < n_v_heads; h++) {
                beta[h] = vec_dot((const uint8_t *)lw->ssm_beta + (size_t)h * row_bytes,
                                  s->xb, dim, beta_type);
                beta[h] = 1.0f / (1.0f + expf(-beta[h]));
            }
        } else {
            float bvals[dim];
            for (int h = 0; h < n_v_heads; h++) {
                dequantize_row((const uint8_t *)lw->ssm_beta + (size_t)h * row_bytes,
                               bvals, dim, beta_type);
                float sum = 0.0f;
                for (int d = 0; d < dim; d++) sum += bvals[d] * s->xb[d];
                beta[h] = 1.0f / (1.0f + expf(-sum));
            }
        }
    }
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

    /* 12. State: [d_state, d_state, n_v_heads] = [128, 128, 32] */
    float *state = s->ssm_state[il];

    /* Decay state: state[h] *= gate_exp[h] for all elements */
    for (int h = 0; h < n_v_heads; h++) {
        float ge = gate_exp[h];
        float *st = state + h * d_state * d_state;
        for (int i = 0; i < d_state * d_state; i++) st[i] *= ge;
    }

    /* 13. Repeat q/k to n_v_heads */
    int repeat = n_v_heads / n_k_heads;
    /* q_rep and k_rep: [d_state, n_v_heads] */
    float *q_rep = gate_exp + n_v_heads; /* [d_state * n_v_heads] */
    float *k_rep = q_rep + d_state * n_v_heads;
    for (int h = 0; h < n_v_heads; h++) {
        int kh = h / repeat;
        memcpy(q_rep + h * d_state, q_conv + kh * d_state, d_state * sizeof(float));
        memcpy(k_rep + h * d_state, k_conv + kh * d_state, d_state * sizeof(float));
    }

    /* 14. sk[d,h] = sum_{d1}(state[h][d1][d] * k[h][d1])
     * llama.cpp: sk = sum_rows(s * k), where s[d1][d2][h] * k[d1][h]
     * k is indexed by d1 (the summed dimension), not d2 */
    float *sk = k_rep + d_state * n_v_heads; /* [d_state * n_v_heads] */
    for (int h = 0; h < n_v_heads; h++) {
        float *st_h = state + h * d_state * d_state;
        float *kr_h = k_rep + h * d_state;
        for (int d2 = 0; d2 < d_state; d2++) {
            float sum = 0.0f;
            for (int d1 = 0; d1 < d_state; d1++) {
                sum += st_h[d1 * d_state + d2] * kr_h[d1];
            }
            sk[d2 * n_v_heads + h] = sum;
        }
    }

    /* 15. d = (v_conv - sk) * beta -> [d_state, n_v_heads] */
    /* v_conv is [head_v_dim, n_v_heads] = [d_state, n_v_heads] */
    float *d_vals = sk + d_state * n_v_heads; /* reuse sk space */
    for (int d = 0; d < d_state; d++) {
        for (int h = 0; h < n_v_heads; h++) {
            d_vals[d * n_v_heads + h] = (v_conv[h * head_v_dim + d] - sk[d * n_v_heads + h]) * beta[h];
        }
    }

    /* 16. state[h][d1][d2] += k[h][d1] * d[d2][h]
     * llama.cpp: kd[d1][d2][h] = k[d1][h] * d[d2][h], s += kd */
    for (int h = 0; h < n_v_heads; h++) {
        float *st_h = state + h * d_state * d_state;
        float *kr_h = k_rep + h * d_state;
        for (int d1 = 0; d1 < d_state; d1++) {
            float kfactor = kr_h[d1];
            float *st_row = st_h + d1 * d_state;
            for (int d2 = 0; d2 < d_state; d2++) {
                st_row[d2] += kfactor * d_vals[d2 * n_v_heads + h];
            }
        }
    }

    /* 17. output[d,h] = sum_{d1}(state[h][d1][d] * q[h][d1])
     * llama.cpp: o = sum_rows(s * q), where s[d1][d2][h] * q[d1][h]
     * q is indexed by d1 (the summed dimension), not d2 */
    float *ssm_output = d_vals + d_state * n_v_heads; /* [d_state * n_v_heads] */
    for (int h = 0; h < n_v_heads; h++) {
        float *st_h = state + h * d_state * d_state;
        float *qr_h = q_rep + h * d_state;
        for (int d2 = 0; d2 < d_state; d2++) {
            float sum = 0.0f;
            for (int d1 = 0; d1 < d_state; d1++) {
                sum += st_h[d1 * d_state + d2] * qr_h[d1];
            }
            ssm_output[d2 * n_v_heads + h] = sum;
        }
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
    if (il == 0 && pos == 0) dbg_vec("final_out[:8]", final_output, 8, 8);
#endif

    /* 19. Reshape to [value_dim] and output projection */
    /* final_output is [head_v_dim * n_v_heads] = [value_dim] = [4096] */
    /* ssm_out: [n_embd, value_dim] */
    matmul(residual, final_output, lw->ssm_out, c->ssm_d_inner, dim, lw->type_ssm_out);
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
 * Batch prefill: all prompt tokens processed at once.
 * Projection matmuls batched (weights read once). Attention per-token.
 * ================================================================ */

float *model_forward_prefill(model_t *m, const int *tokens, int n_tokens, int start_pos) {
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;
    int dim = c->n_embd, n_ffn = c->n_ffn;
    int n_heads = c->n_heads, n_kv_heads = c->n_kv_heads, head_dim = c->head_dim;
    int kv_dim = n_kv_heads * head_dim, kv_mul = n_heads / n_kv_heads;
    int q_dim = n_heads * head_dim, seq_len = c->max_seq_len;
    int max_dim = (q_dim > dim) ? q_dim : dim;
    size_t bs = (size_t)n_tokens;

    size_t sz = bs * (dim + 2 * max_dim + q_dim + 2 * kv_dim + 2 * n_ffn);
    float *buf = (float *)malloc(sz * sizeof(float));
    if (!buf) { fprintf(stderr, "OOM: prefill batch\n"); exit(1); }
    float *p = buf;
    float *x_batch = p;  p += bs * dim;
    float *xb_batch = p; p += bs * max_dim;
    float *xb2_batch = p; p += bs * dim;
    float *q_batch = p;  p += bs * q_dim;
    float *k_batch = p;  p += bs * kv_dim;
    float *v_batch = p;  p += bs * kv_dim;
    float *hb_batch = p; p += bs * n_ffn;
    float *hb2_batch = p;

    /* Embedding lookup */
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        for (int bi = 0; bi < n_tokens; bi++) {
            const void *er = (const uint8_t *)w->token_embd + (size_t)tokens[bi] * row_bytes;
            dequantize_row(er, x_batch + bi * dim, dim, w->type_token_embd);
        }
    }

    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];

        /* RMSNorm */
        for (int bi = 0; bi < n_tokens; bi++)
            rmsnorm(xb_batch + bi * dim, x_batch + bi * dim, s->attn_norm_w[l], dim, c->rms_norm_eps);

        /* Q projection (batched) */
        tensor_set_repacked(m->repack_used[2+l*9] ? m->repack_buffers[2+l*9] : NULL);
        matmul_batch(q_batch, xb_batch, n_tokens, lw->attn_q, dim, q_dim, lw->type_attn_q);
        tensor_set_repacked(NULL);

        /* K+V projection (batched dual) */
        tensor_set_repacked(m->repack_used[3+l*9] ? m->repack_buffers[3+l*9] : NULL);
        matmul_dual_batch(k_batch, v_batch, xb_batch, n_tokens,
                          lw->attn_k, lw->attn_v, dim, kv_dim,
                          lw->type_attn_k, lw->type_attn_v);
        tensor_set_repacked(NULL);

        /* Per-position: RoPE, KV store, attention */
        for (int bi = 0; bi < n_tokens; bi++) {
            int pos = start_pos + bi;
            float *q_pos = q_batch + bi * q_dim;
            float *k_pos = k_batch + bi * kv_dim;
            float *v_pos = v_batch + bi * kv_dim;
            float *xbi = xb_batch + bi * dim;

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
            uint8_t *kcl = s->key_cache + (size_t)l * seq_len * c->n_kv_heads * s->kv_row_size_k;
            uint8_t *vcl = s->val_cache + (size_t)l * seq_len * c->n_kv_heads * s->kv_row_size_v;
            for (int hkv = 0; hkv < n_kv_heads; hkv++) {
                uint8_t *kp = kcl + (size_t)pos * c->n_kv_heads * s->kv_row_size_k + hkv * s->kv_row_size_k;
                uint8_t *vp = vcl + (size_t)pos * c->n_kv_heads * s->kv_row_size_v + hkv * s->kv_row_size_v;
                for (int d2 = 0; d2 < head_dim; d2++)
                    ((uint16_t*)kp)[d2] = fp32_to_fp16(k_pos[hkv * head_dim + d2]);
                for (int d2 = 0; d2 < head_dim; d2++)
                    ((uint16_t*)vp)[d2] = fp32_to_fp16(v_pos[hkv * head_dim + d2]);
            }

            /* Attention (copied exactly from model_forward) */
            for (int h = 0; h < n_heads; h++) {
                float *qh = q_pos + h * head_dim;
                int kv_h = h / kv_mul;
                float *xbh = xbi + h * head_dim;
                float max_score = -1e30f, sum_exp = 0.0f;
                float acc[256];
                memset(acc, 0, (size_t)head_dim * sizeof(float));

                for (int t = 0; t <= pos; t++) {
                    const uint8_t *kt = kcl + (size_t)t * c->n_kv_heads * s->kv_row_size_k + kv_h * s->kv_row_size_k;
                    float score;
                    if (s->kv_type_k == KV_CACHE_Q8_0)
                        score = vec_dot_q8_0_f32(kt, qh, head_dim);
                    else if (s->kv_type_k == KV_CACHE_Q4_0)
                        score = vec_dot_q4_0_f32(kt, qh, head_dim);
                    else
                        score = vec_dot_f16_f32(kt, qh, head_dim);
                    score /= sqrtf((float)head_dim);

                    const uint8_t *vt = vcl + (size_t)t * c->n_kv_heads * s->kv_row_size_v + kv_h * s->kv_row_size_v;
                    if (score > max_score) {
                        float correction = expf(max_score - score);
                        sum_exp = sum_exp * correction + 1.0f;
                        if (s->kv_type_v == KV_CACHE_Q8_0)
                            fma_scale_q8_0_f32(acc, correction, vt, head_dim);
                        else if (s->kv_type_v == KV_CACHE_Q4_0)
                            fma_scale_q4_0_f32(acc, correction, vt, head_dim);
                        else {
                            const uint16_t *vt16 = (const uint16_t*)vt;
                            for (int d2 = 0; d2 < head_dim; d2++)
                                acc[d2] = acc[d2] * correction + fp16_to_fp32(vt16[d2]);
                        }
                        max_score = score;
                    } else {
                        float wt = expf(score - max_score);
                        sum_exp += wt;
                        if (s->kv_type_v == KV_CACHE_Q8_0)
                            scale_add_q8_0_f32(acc, wt, vt, head_dim);
                        else if (s->kv_type_v == KV_CACHE_Q4_0)
                            scale_add_q4_0_f32(acc, wt, vt, head_dim);
                        else {
                            const uint16_t *vt16 = (const uint16_t*)vt;
                            for (int d2 = 0; d2 < head_dim; d2++)
                                acc[d2] += wt * fp16_to_fp32(vt16[d2]);
                        }
                    }
                }

                float inv_sum = 1.0f / sum_exp;
                for (int d2 = 0; d2 < head_dim; d2++) xbh[d2] = acc[d2] * inv_sum;
            }
        }

        /* Output projection (batched) */
        tensor_set_repacked(m->repack_used[6+l*9] ? m->repack_buffers[6+l*9] : NULL);
        matmul_batch(xb2_batch, xb_batch, n_tokens, lw->attn_output, q_dim, dim, lw->type_attn_output);
        tensor_set_repacked(NULL);

        /* Residual: x += attn_out */
        for (int bi = 0; bi < n_tokens; bi++) {
            float *a = x_batch + bi * dim, *b = xb2_batch + bi * dim;
            for (int d2 = 0; d2 < dim; d2++) a[d2] += b[d2];
        }

        /* FFN RMSNorm */
        for (int bi = 0; bi < n_tokens; bi++)
            rmsnorm(xb_batch + bi * dim, x_batch + bi * dim, s->post_attn_norm_w[l], dim, c->rms_norm_eps);

        /* FFN gate+up (batched dual) */
        tensor_set_repacked(m->repack_used[7+l*9] ? m->repack_buffers[7+l*9] : NULL);
        matmul_dual_batch(hb_batch, hb2_batch, xb_batch, n_tokens,
                          lw->ffn_gate, lw->ffn_up, dim, n_ffn,
                          lw->type_ffn_gate, lw->type_ffn_up);
        tensor_set_repacked(NULL);

        /* SiLU + mul */
        for (int bi = 0; bi < n_tokens; bi++) {
            silu(hb_batch + bi * n_ffn, n_ffn);
            elemwise_mul(hb_batch + bi * n_ffn, hb_batch + bi * n_ffn, hb2_batch + bi * n_ffn, n_ffn);
        }

        /* FFN down (batched) */
        tensor_set_repacked(m->repack_used[8+l*9] ? m->repack_buffers[8+l*9] : NULL);
        matmul_batch(xb2_batch, hb_batch, n_tokens, lw->ffn_down, n_ffn, dim, lw->type_ffn_down);
        tensor_set_repacked(NULL);

        /* Residual: x += ffn_out */
        for (int bi = 0; bi < n_tokens; bi++) {
            float *a = x_batch + bi * dim, *b = xb2_batch + bi * dim;
            for (int d2 = 0; d2 < dim; d2++) a[d2] += b[d2];
        }
    }

    /* Final norm + output (last token only) */
    float *last_x = x_batch + (n_tokens - 1) * dim;
    rmsnorm(s->x, last_x, s->output_norm_w, dim, c->rms_norm_eps);
    tensor_set_repacked(m->repack_used[1] ? m->repack_buffers[1] : NULL);
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);
    tensor_set_repacked(NULL);

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
