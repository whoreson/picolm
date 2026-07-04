#include "model.h"
#include "tensor.h"
#include "quant.h"

#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
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
    cfg->max_seq_len = 2048;
    cfg->weight_type = GGUF_TYPE_F16;
    m->tok_bos_id = 1;
    m->tok_eos_id = 2;

    for (uint64_t i = 0; i < n_metadata; i++) {
        gguf_str_t key = read_gguf_string(&r);
        uint32_t vtype = read_u32(&r);

        if (str_eq(key, "llama.embedding_length") || str_eq(key, "general.embedding_length")
            || str_eq(key, "qwen2.embedding_length") || str_eq(key, "qwen3.embedding_length")) {
            int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
            if (key.str[0] == 'q') cfg->rope_type = 1;  /* qwen2/qwen3 interleaved */
        } else if (str_eq(key, "llama.feed_forward_length") || str_eq(key, "general.feed_forward_length")
            || str_eq(key, "qwen2.feed_forward_length") || str_eq(key, "qwen3.feed_forward_length")) {
            int dummy; cfg->n_ffn = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count")
            || str_eq(key, "qwen2.attention.head_count") || str_eq(key, "qwen3.attention.head_count")) {
            int dummy; cfg->n_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count_kv")
            || str_eq(key, "qwen2.attention.head_count_kv") || str_eq(key, "qwen3.attention.head_count_kv")) {
            int dummy; cfg->n_kv_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "attention.key_length")
            || str_eq(key, "qwen2.attention.key_length")
            || str_eq(key, "qwen3.attention.key_length")) {
            /* Explicit head_dim (Qwen3 may differ from n_embd/n_heads) */
            int dummy; cfg->head_dim = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.block_count")
            || str_eq(key, "qwen2.block_count") || str_eq(key, "qwen3.block_count")) {
            int dummy; cfg->n_layers = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.context_length")
            || str_eq(key, "qwen2.context_length") || str_eq(key, "qwen3.context_length")) {
            int dummy; cfg->max_seq_len = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.rope.freq_base")
            || str_eq(key, "qwen2.rope.freq_base") || str_eq(key, "qwen3.rope.freq_base")) {
            if (vtype == GGUF_META_FLOAT32) {
                cfg->rope_freq_base = read_f32(&r);
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "llama.attention.layer_norm_rms_epsilon")
            || str_eq(key, "qwen2.attention.layer_norm_rms_epsilon")
            || str_eq(key, "qwen3.attention.layer_norm_rms_epsilon")) {
            /* Read epsilon from GGUF (usually F64 type=3) */
            if (vtype == 3) { /* F64 */
                double val;
                memcpy(&val, r.data + r.pos, 8); r.pos += 8;
                cfg->rms_norm_eps = (float)val;
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "general.alignment")) {
            int dummy; cfg->alignment = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.vocab_size")
            || str_eq(key, "qwen2.vocab_size") || str_eq(key, "qwen3.vocab_size")) {
            int dummy; cfg->vocab_size = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.bos_token_id")) {
            int dummy; m->tok_bos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.eos_token_id")) {
            int dummy; m->tok_eos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.pre")) {
            /* Detect space marker type for tokenization */
            if (vtype == GGUF_META_STRING) {
                gguf_str_t pre = read_gguf_string(&r);
                /* SmolLM uses U+0100 (0xC4 0xA0) instead of U+2581 */
                if (pre.len >= 6 && strncmp(pre.str, "smollm", 6) == 0) {
                    m->tok_space_marker = 1; /* U+0100 */
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
        if (i < 5) {
            fprintf(stderr, "T%llu: %s type=%u offset=%" PRIu64 " abs=%zu\n",
                    (unsigned long long)i, tinfos[i].name.str, tinfos[i].type,
                    tinfos[i].offset, (size_t)((const uint8_t*)ptr - (const uint8_t*)m->mmap_addr));
        }

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
                    lw->attn_k = ptr; lw->type_attn_k = qtype;
                } else if (strcmp(suffix, "attn_v.weight") == 0) {
                    lw->attn_v = ptr; lw->type_attn_v = qtype;
                } else if (strcmp(suffix, "attn_output.weight") == 0) {
                    lw->attn_output = ptr; lw->type_attn_output = qtype;
                } else if (strcmp(suffix, "attn_q_norm.weight") == 0) {
                    lw->attn_q_norm = ptr; lw->type_attn_q_norm = qtype;
                } else if (strcmp(suffix, "attn_k_norm.weight") == 0) {
                    lw->attn_k_norm = ptr; lw->type_attn_k_norm = qtype;
                } else if (strcmp(suffix, "ffn_norm.weight") == 0) {
                    lw->ffn_norm = ptr; lw->type_ffn_norm = qtype;
                } else if (strcmp(suffix, "ffn_gate.weight") == 0) {
                    lw->ffn_gate = ptr; lw->type_ffn_gate = qtype;
                } else if (strcmp(suffix, "ffn_down.weight") == 0) {
                    lw->ffn_down = ptr; lw->type_ffn_down = qtype;
                } else if (strcmp(suffix, "ffn_up.weight") == 0) {
                    lw->ffn_up = ptr; lw->type_ffn_up = qtype;
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

    cfg->weight_type = w->layers[0].type_attn_q;

    fprintf(stderr, "Model config:\n");
    fprintf(stderr, "  n_embd=%d, n_ffn=%d, n_heads=%d, n_kv_heads=%d\n",
            cfg->n_embd, cfg->n_ffn, cfg->n_heads, cfg->n_kv_heads);
    fprintf(stderr, "  n_layers=%d, vocab_size=%d, max_seq=%d\n",
            cfg->n_layers, cfg->vocab_size, cfg->max_seq_len);
    fprintf(stderr, "  head_dim=%d, rope_base=%.1f\n", cfg->head_dim, cfg->rope_freq_base);
    free(tinfos);
    return 0;
}

/* ---- Pre-compute RoPE cos/sin lookup tables ---- */

static void init_rope_tables(run_state_t *s, const model_config_t *c) {
    int half_dim = c->head_dim / 2;
    for (int pos = 0; pos < c->max_seq_len; pos++) {
        float *cos_row = s->rope_cos + (size_t)pos * half_dim;
        float *sin_row = s->rope_sin + (size_t)pos * half_dim;
        for (int i = 0; i < half_dim; i++) {
            float theta = (float)pos / powf(c->rope_freq_base, (float)(2 * i) / (float)c->head_dim);
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

static int allocate_run_state(model_t *m, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v) {
    model_config_t *c = &m->config;
    run_state_t *s = &m->state;

    int half_dim = c->head_dim / 2;
    int q_dim = c->n_heads * c->head_dim;
    int max_dim = (q_dim > c->n_embd) ? q_dim : c->n_embd;

    /* Calculate sizes for float buffers */
    size_t sz_x      = (size_t)c->n_embd * sizeof(float);
    size_t sz_xb     = (size_t)q_dim * sizeof(float);
    size_t sz_xb2    = (size_t)q_dim * sizeof(float);
    size_t sz_q      = (size_t)q_dim * sizeof(float);
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

    size_t total = sz_x + sz_xb + sz_xb2 + sz_q +
                   sz_hb + sz_hb2 + sz_logits +
                   sz_scratch + sz_rope + sz_norm;

    /* Quantized KV cache: separate allocation */
    size_t sz_k_row = kv_row_size(kv_type_k, c->head_dim);
    size_t sz_v_row = kv_row_size(kv_type_v, c->head_dim);
    size_t sz_kv = (size_t)c->n_layers * c->max_seq_len * c->n_kv_heads * (sz_k_row + sz_v_row);

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
    s->xb     = p; p += c->n_embd;
    s->xb2    = p; p += c->n_embd;
    s->q      = p; p += c->n_embd;
    s->hb     = p; p += c->n_ffn;
    s->hb2    = p; p += c->n_ffn;
    s->logits = p; p += c->vocab_size;
    s->dequant_scratch = p; p += scratch_dim;

    /* RoPE tables */
    s->rope_cos = p; p += (size_t)c->max_seq_len * half_dim;
    s->rope_sin = p; p += (size_t)c->max_seq_len * half_dim;

    /* Norm weights */
    s->norm_weights = p;

    /* KV cache pointers: K layers first, then V layers */
    size_t layer_stride_k = (size_t)c->max_seq_len * c->n_kv_heads * sz_k_row;
    uint8_t *kb = (uint8_t *)s->kv_block;
    s->key_cache = kb;
    s->val_cache = kb + (size_t)c->n_layers * layer_stride_k;

    /* Pre-dequantize norm weights */
    float *nw = s->norm_weights;
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &m->weights.layers[l];
        s->attn_norm_w[l] = nw;
        dequantize_row(lw->attn_norm, nw, c->n_embd, lw->type_attn_norm);
        nw += c->n_embd;

        s->ffn_norm_w[l] = nw;
        dequantize_row(m->weights.layers[l].ffn_norm, nw, c->n_embd,
                       m->weights.layers[l].type_ffn_norm);
        nw += c->n_embd;

        /* Qwen3 QK-norm weights (per-head, if present) */
        s->attn_q_norm_w[l] = nw;
        if (lw->attn_q_norm)
            dequantize_row(lw->attn_q_norm, nw, c->head_dim,
                           lw->type_attn_q_norm);
        else
            memset(nw, 0, (size_t)c->head_dim * sizeof(float));
        nw += c->head_dim;

        s->attn_k_norm_w[l] = nw;
        if (lw->attn_k_norm)
            dequantize_row(lw->attn_k_norm, nw, c->head_dim,
                           lw->type_attn_k_norm);
        else
            memset(nw, 0, (size_t)c->head_dim * sizeof(float));
        nw += c->head_dim;
    }
    s->output_norm_w = nw;
    dequantize_row(m->weights.output_norm, nw, c->n_embd,
                   m->weights.type_output_norm);

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
    int kv_dim = n_kv_heads * head_dim;
    int kv_mul = n_heads / n_kv_heads;
    int seq_len = c->max_seq_len;
    int half_dim = head_dim / 2;

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
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];
        int ri = 2 + l * 9; /* repack buffer index base for this layer */

        /* ---- Attention ---- */
        rmsnorm(s->xb, s->x, s->attn_norm_w[l], dim, c->rms_norm_eps);

        /* Q projection */
        tensor_set_repacked(m->repack_used[ri] ? m->repack_buffers[ri] : NULL);
        matmul(s->q, s->xb, lw->attn_q, dim, q_dim, lw->type_attn_q);
        tensor_set_repacked(NULL);

        /* K projection */
        tensor_set_repacked(m->repack_used[ri+1] ? m->repack_buffers[ri+1] : NULL);
        float *k_tmp = s->xb2; /* reuse xb2 as temp for K (kv_dim <= dim) */
        matmul(k_tmp, s->xb, lw->attn_k, dim, kv_dim, lw->type_attn_k);
        tensor_set_repacked(NULL);

        /* KV cache layout: [layer][pos][head] with per-head quantized rows */
        uint8_t *kcache_layer = s->key_cache + (size_t)l * seq_len * c->n_kv_heads * s->kv_row_size_k;
        uint8_t *vcache_layer = s->val_cache + (size_t)l * seq_len * c->n_kv_heads * s->kv_row_size_v;

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
        rope(s->q, k_tmp, head_dim, n_heads, n_kv_heads, cos_pos, sin_pos, c->rope_type);

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
                for (int d = 0; d < head_dim; d++) kf[d] = fp32_to_fp16(k_head[d]);
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
                for (int d = 0; d < head_dim; d++) vf[d] = fp32_to_fp16(v_head[d]);
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
                        for (int d = 0; d < head_dim; d++) {
                            acc[d] = acc[d] * correction + fp16_to_fp32(vt16[d]);
                        }
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
                        for (int d = 0; d < head_dim; d++) {
                            acc[d] += w * fp16_to_fp32(vt16[d]);
                        }
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

        /* Output projection */
        tensor_set_repacked(m->repack_used[ri+3] ? m->repack_buffers[ri+3] : NULL);
        matmul(s->xb2, s->xb, lw->attn_output, q_dim, dim, lw->type_attn_output);
        tensor_set_repacked(NULL);
        vec_add(s->x, s->xb2, dim);

        /* ---- FFN (SwiGLU) ---- */
        rmsnorm(s->xb, s->x, s->ffn_norm_w[l], dim, c->rms_norm_eps);

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

    /* 3. Final RMSNorm */
    rmsnorm(s->x, s->x, s->output_norm_w, dim, c->rms_norm_eps);

    /* 4. Output projection -> logits */
    tensor_set_repacked(m->repack_used[1] ? m->repack_buffers[1] : NULL);
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);
    tensor_set_repacked(NULL);

    return s->logits;
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

            rope(q_pos, k_pos, head_dim, n_heads, n_kv_heads, cos_pos, sin_pos, c->rope_type);

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
            rmsnorm(xb_batch + bi * dim, x_batch + bi * dim, s->ffn_norm_w[l], dim, c->rms_norm_eps);

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
