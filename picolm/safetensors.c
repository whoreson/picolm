/* safetensors model loader for PicoLM - loads from HF safetensors format.
 * Uses csafetensors.c/h from qwen3.5-c for mmap + header parsing.
 * Uses PicoLM's cJSON for config.json and tokenizer.json parsing. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#include "model.h"
#include "quant.h"
#include "csafetensors.h"
#include "cJSON.h"

typedef struct {
    const char *filename;
    uint8_t *mmap_addr;
    size_t mmap_size;
    int fd;
} st_file_t;

#define MAX_ST_FILES 16

typedef struct {
    st_file_t files[MAX_ST_FILES];
    int n_files;
    csafetensors_t st_handles[MAX_ST_FILES];
} st_state_t;

static int json_get_int(cJSON *v, int default_val) {
    if (!v || !cJSON_IsNumber(v)) return default_val;
    return (int)cJSON_GetNumberValue(v);
}

static int load_config_safetensors(const char *model_dir, model_config_t *cfg) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json_str = (char *)malloc(size + 1);
    if (!json_str) { fclose(f); return -1; }
    if (fread(json_str, 1, size, f) != (size_t)size) { free(json_str); fclose(f); return -1; }
    json_str[size] = '\0';
    fclose(f);

    cJSON *root = cJSON_Parse(json_str);
    free(json_str);
    if (!root) return -1;

    cJSON *cfg_json = cJSON_GetObjectItem(root, "text_config");
    if (!cfg_json || !cJSON_IsObject(cfg_json)) cfg_json = root;

    memset(cfg, 0, sizeof(*cfg));
    cfg->n_embd         = json_get_int(cJSON_GetObjectItem(cfg_json, "hidden_size"), 0);
    cfg->n_ffn          = json_get_int(cJSON_GetObjectItem(cfg_json, "intermediate_size"), 0);
    cfg->n_heads        = json_get_int(cJSON_GetObjectItem(cfg_json, "num_attention_heads"), 0);
    cfg->n_kv_heads     = json_get_int(cJSON_GetObjectItem(cfg_json, "num_key_value_heads"), 0);
    cfg->n_layers       = json_get_int(cJSON_GetObjectItem(cfg_json, "num_hidden_layers"), 0);
    cfg->vocab_size     = json_get_int(cJSON_GetObjectItem(cfg_json, "vocab_size"), 0);
    cfg->head_dim       = json_get_int(cJSON_GetObjectItem(cfg_json, "head_dim"), 0);
    if (cfg->head_dim == 0) cfg->head_dim = cfg->n_embd / cfg->n_heads;
    cfg->rms_norm_eps   = (float)cJSON_GetNumberValue(cJSON_GetObjectItem(cfg_json, "rms_norm_eps"));
    if (cfg->rms_norm_eps == 0) cfg->rms_norm_eps = 1e-6f;
    #if 0
    /* cJSON-based rope parameter extraction (disabled: cJSON corrupts valuestring in large configs) */
    {
        cJSON *rt = cJSON_GetObjectItem(cfg_json, "rope_theta");
        if (rt && cJSON_IsNumber(rt) && !cJSON_IsNull(rt)) {
            cfg->rope_freq_base = (float)cJSON_GetNumberValue(rt);
        } else {
            cJSON *rp = cJSON_GetObjectItem(cfg_json, "rope_parameters");
            if (rp && cJSON_IsObject(rp)) {
                cJSON *rpt = cJSON_GetObjectItem(rp, "rope_theta");
                if (rpt && cJSON_IsNumber(rpt) && !cJSON_IsNull(rpt)) {
                    cfg->rope_freq_base = (float)cJSON_GetNumberValue(rpt);
                }
                cJSON *prf = cJSON_GetObjectItem(rp, "partial_rotary_factor");
                if (prf && cJSON_IsNumber(prf) && !cJSON_IsNull(prf)) {
                    float prf_val = (float)cJSON_GetNumberValue(prf);
                    cfg->rope_dim = (int)((float)cfg->head_dim * prf_val + 0.5f);
                }
            }
        }
    }
#endif
    cfg->rope_freq_base = (float)cJSON_GetNumberValue(cJSON_GetObjectItem(cfg_json, "rope_theta"));
    if (cfg->rope_freq_base <= 0.0f) cfg->rope_freq_base = 10000.0f;

    /* Qwen3.5 nests rope params under rope_parameters, not at top level.
     * Use raw strstr to avoid cJSON corruption of valuestring in large configs. */
    {
        char cfg_path[1024];
        snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", model_dir);
        FILE *cf = fopen(cfg_path, "r");
        if (cf) {
            fseek(cf, 0, SEEK_END);
            long csz = ftell(cf);
            fseek(cf, 0, SEEK_SET);
            char *cbuf = malloc(csz + 1);
            if (cbuf) {
                fread(cbuf, 1, csz, cf);
                cbuf[csz] = '\0';
                /* Try rope_parameters block first (Qwen3.5), then top-level */
                char *rp = strstr(cbuf, "\"rope_parameters\"");
                if (!rp) rp = cbuf;
                char *rt = strstr(rp, "\"rope_theta\"");
                if (rt) {
                    char *colon = strchr(rt + 12, ':');
                    if (colon) cfg->rope_freq_base = (float)strtod(colon + 1, NULL);
                }
                /* partial_rotary_factor is for MTP multi-rope, not for the main
                 * transformer RoPE. The main transformer applies RoPE to the
                 * full head_dim. Skip this to let rope_dim default to 0 (=head_dim). */
                free(cbuf);
            }
            fclose(cf);
        }
    }
    if (cfg->rope_freq_base <= 0.0f) cfg->rope_freq_base = 10000.0f;

    cfg->ssm_d_conv     = json_get_int(cJSON_GetObjectItem(cfg_json, "linear_conv_kernel_dim"), 4);
    cfg->ssm_d_state    = json_get_int(cJSON_GetObjectItem(cfg_json, "linear_key_head_dim"), 128);
    cfg->ssm_n_group    = json_get_int(cJSON_GetObjectItem(cfg_json, "linear_num_key_heads"), 0);
    cfg->ssm_dt_rank    = json_get_int(cJSON_GetObjectItem(cfg_json, "linear_num_value_heads"), 0);
    cfg->ssm_d_inner    = json_get_int(cJSON_GetObjectItem(cfg_json, "linear_value_head_dim"), 0);
    /* Compute value_dim = head_v_dim * n_v_heads */
    if (cfg->ssm_d_inner > 0 && cfg->ssm_dt_rank > 0)
        cfg->ssm_d_inner *= cfg->ssm_dt_rank;
    cfg->has_ssm = cfg->ssm_n_group > 0;

    /* Default: all attention layers. Qwen3.5 pattern: SSM SSM SSM ATTN */
    for (int i = 0; i < cfg->n_layers; i++)
        cfg->layer_type[i] = cfg->has_ssm ? (i >= 3 && (i - 3) % 4 == 0) ? 0 : 1 : 0;

    cJSON_Delete(root);
    return 0;
}

/* Detect naming convention: some models use "model.language_model.layers." (Qwen)
   others use "model.layers." (Llama/SmolLM). Try both. */
static const void *find_tensor(st_state_t *st, const char *name, int *out_type) {
    for (int f = 0; f < st->n_files; f++) {
        const csafetensors_tensor_t *tensor = csafetensors_get_tensor(st->st_handles + f, name);
        if (tensor) {
            const uint8_t *data = csafetensors_get_tensor_data(st->st_handles + f, tensor);
            if (tensor->dtype == CSAFETENSORS_DTYPE_BFLOAT16) *out_type = GGUF_TYPE_BF16;
            else if (tensor->dtype == CSAFETENSORS_DTYPE_FLOAT16) *out_type = GGUF_TYPE_F16;
            else if (tensor->dtype == CSAFETENSORS_DTYPE_FLOAT32) *out_type = GGUF_TYPE_F32;
            else return NULL;
            return data;
        }
    }
    /* Try alternative prefix: replace "model.language_model." with "model." */
    char alt[512];
    const char *src = name;
    char *dst = alt;
    while (*src) {
        if (strncmp(src, "model.language_model.", 21) == 0) {
            strcpy(dst, "model.");
            dst += 6;
            src += 21;
        } else {
            *dst++ = *src++;
        }
    }
    *dst = '\0';
    for (int f = 0; f < st->n_files; f++) {
        const csafetensors_tensor_t *tensor = csafetensors_get_tensor(st->st_handles + f, alt);
        if (tensor) {
            const uint8_t *data = csafetensors_get_tensor_data(st->st_handles + f, tensor);
            if (tensor->dtype == CSAFETENSORS_DTYPE_BFLOAT16) *out_type = GGUF_TYPE_BF16;
            else if (tensor->dtype == CSAFETENSORS_DTYPE_FLOAT16) *out_type = GGUF_TYPE_F16;
            else if (tensor->dtype == CSAFETENSORS_DTYPE_FLOAT32) *out_type = GGUF_TYPE_F32;
            else return NULL;
            return data;
        }
    }
    return NULL;
}

static int load_safetensors_files(const char *model_dir, st_state_t *st) {
    memset(st, 0, sizeof(*st));
    char path[4096];
    st->n_files = 0;

    /* Try to load model.safetensors-*.safetensors files by globbing. */
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    if (access(path, F_OK) == 0) {
        if (csafetensors_mmap_from_file(path, st->st_handles + st->n_files) == CSAFETENSORS_SUCCESS) {
            st->n_files++;
            return 0;
        }
    }

    /* Try sharded files */
    for (int i = 1; i <= 16 && st->n_files < MAX_ST_FILES; i++) {
        int loaded = 0;
        for (int n = 2; n <= 16; n++) {
            snprintf(path, sizeof(path), "%s/model.safetensors-%05d-of-%05d.safetensors",
                     model_dir, i, n);
            if (access(path, F_OK) == 0) {
                if (csafetensors_mmap_from_file(path, st->st_handles + st->n_files) == CSAFETENSORS_SUCCESS) {
                    st->n_files++;
                    loaded = 1;
                    break;
                }
            }
        }
        if (!loaded) break;
    }

    if (st->n_files == 0) {
        fprintf(stderr, "No safetensors files found in %s\n", model_dir);
        return -1;
    }
    return 0;
}

static int map_tensors_safetensors(st_state_t *st, model_t *m) {
    model_config_t *cfg = &m->config;
    model_weights_t *w = &m->weights;
    char name[512];
    int type;

    snprintf(name, sizeof(name), "model.language_model.embed_tokens.weight");
    w->token_embd = find_tensor(st, name, &type); w->type_token_embd = type;

    snprintf(name, sizeof(name), "model.language_model.norm.weight");
    w->output_norm = find_tensor(st, name, &type); w->type_output_norm = type;

    snprintf(name, sizeof(name), "lm_head.weight");
    w->output = find_tensor(st, name, &type);
    if (!w->output) { w->output = w->token_embd; w->type_output = w->type_token_embd; }
    else w->type_output = type;

    for (int l = 0; l < cfg->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];
        snprintf(name, sizeof(name), "model.language_model.layers.%d.input_layernorm.weight", l);
        lw->attn_norm = find_tensor(st, name, &type); lw->type_attn_norm = type;

        snprintf(name, sizeof(name), "model.language_model.layers.%d.post_attention_layernorm.weight", l);
        lw->post_attn_norm = find_tensor(st, name, &type); lw->type_post_attn_norm = type;

        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate_proj.weight", l);
        lw->ffn_gate = find_tensor(st, name, &type); lw->type_ffn_gate = type;
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.up_proj.weight", l);
        lw->ffn_up = find_tensor(st, name, &type); lw->type_ffn_up = type;
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.down_proj.weight", l);
        lw->ffn_down = find_tensor(st, name, &type); lw->type_ffn_down = type;

        if (cfg->layer_type[l] == 0) {
            lw->is_attn_layer = 1;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_proj.weight", l);
            lw->attn_q = find_tensor(st, name, &type); lw->type_attn_q = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_proj.weight", l);
            lw->attn_k = find_tensor(st, name, &type); lw->type_attn_k = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.v_proj.weight", l);
            lw->attn_v = find_tensor(st, name, &type); lw->type_attn_v = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.o_proj.weight", l);
            lw->attn_output = find_tensor(st, name, &type); lw->type_attn_output = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_norm.weight", l);
            lw->attn_q_norm = find_tensor(st, name, &type); lw->type_attn_q_norm = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_norm.weight", l);
            lw->attn_k_norm = find_tensor(st, name, &type); lw->type_attn_k_norm = type;
        } else {
            lw->is_attn_layer = 0;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", l);
            lw->attn_qkv = find_tensor(st, name, &type); lw->type_attn_qkv = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_z.weight", l);
            lw->attn_gate_ssm = find_tensor(st, name, &type); lw->type_attn_gate_ssm = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", l);
            lw->ssm_alpha = find_tensor(st, name, &type); lw->type_ssm_alpha = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", l);
            lw->ssm_beta = find_tensor(st, name, &type); lw->type_ssm_beta = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.conv1d.weight", l);
            lw->ssm_conv1d = find_tensor(st, name, &type); lw->type_ssm_conv1d = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.dt_bias", l);
            lw->ssm_dt = find_tensor(st, name, &type); lw->type_ssm_dt = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.A_log", l);
            lw->ssm_a = find_tensor(st, name, &type); lw->type_ssm_a = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.norm.weight", l);
            lw->ssm_norm = find_tensor(st, name, &type); lw->type_ssm_norm = type;
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.out_proj.weight", l);
            lw->ssm_out = find_tensor(st, name, &type); lw->type_ssm_out = type;
        }
    }
    return 0;
}

static int load_tokenizer_safetensors(const char *model_dir, model_t *m) {
    char path[512];
    m->tok_bos_id = 1;
    m->tok_eos_id = 2;
    m->tok_space_marker = 1;

    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    FILE *cf = fopen(path, "r");
    if (cf) {
        fseek(cf, 0, SEEK_END);
        long csz = ftell(cf);
        fseek(cf, 0, SEEK_SET);
        char *cb = malloc(csz + 1);
        if (cb) {
            fread(cb, 1, csz, cf);
            cb[csz] = '\0';
            cJSON *cr = cJSON_ParseWithLength(cb, (size_t)csz);
            if (cr) {
                cJSON *v = cJSON_GetObjectItem(cr, "bos_token_id");
                if (v && cJSON_IsNumber(v)) m->tok_bos_id = (int)cJSON_GetNumberValue(v);
                v = cJSON_GetObjectItem(cr, "eos_token_id");
                if (v && cJSON_IsNumber(v)) m->tok_eos_id = (int)cJSON_GetNumberValue(v);
                cJSON_Delete(cr);
            }
            /* Detect model_type from raw config bytes (cJSON can corrupt strings in large configs) */
            if (strstr(cb, "\"model_type\"") && strstr(cb, "qwen")) m->tok_bos_id = 11;
            free(cb);
        }
        fclose(cf);
    }

    snprintf(path, sizeof(path), "%s/tokenizer_config.json", model_dir);
    FILE *tf = fopen(path, "r");
    if (tf) {
        fseek(tf, 0, SEEK_END);
        long tsz = ftell(tf);
        fseek(tf, 0, SEEK_SET);
        char *tb = malloc(tsz + 1);
        if (tb) {
            fread(tb, 1, tsz, tf);
            tb[tsz] = '\0';
            char *p = strstr(tb, "\"model_type\"");
            if (p) {
                if (strstr(p, "qwen")) {
                    m->tok_space_marker = 3;
                    m->tok_bos_id = 11; /* Qwen3.5 always uses BOS=11 */
                } else if (strstr(p, "smollm")) m->tok_space_marker = 2;
            }
            free(tb);
        }
        fclose(tf);
    }

    /* Parse vocab.json - character by character to handle both
     * single-line (SmolLM2) and multi-line (Qwen3.5) formats */
    m->tok_n_tokens = m->config.vocab_size;
    snprintf(path, sizeof(path), "%s/vocab.json", model_dir);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long vsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *vbuf = malloc(vsz + 1);
    if (!vbuf) { fclose(f); return -1; }
    fread(vbuf, 1, vsz, f);
    vbuf[vsz] = '\0';
    fclose(f);

    m->tok_scores_data = malloc(m->tok_n_tokens * sizeof(float));
    if (!m->tok_scores_data) { free(vbuf); return -1; }
    memset(m->tok_scores_data, 0, m->tok_n_tokens * sizeof(float));

    struct { uint64_t len; char *str; } *tok_arr = calloc(m->tok_n_tokens, sizeof(*tok_arr));
    if (!tok_arr) { free(vbuf); return -1; }

    /* Walk the JSON: {"<key>": <id>, "<key>": <id>, ...} */
    const char *p = vbuf;
    while (*p) {
        /* Find opening quote of key */
        while (*p && *p != '"') p++;
        if (*p != '"') break;
        p++;

        /* Read key string into buffer */
        char key[256];
        int ki = 0;
        while (*p && *p != '"' && ki < 254) {
            if (*p == '\\' && *(p+1)) {
                p++;
                switch (*p) {
                    case '"': key[ki++] = '"'; break;
                    case '\\': key[ki++] = '\\'; break;
                    case 'n': key[ki++] = '\n'; break;
                    case 't': key[ki++] = '\t'; break;
                    case 'r': key[ki++] = '\r'; break;
                    case '/': key[ki++] = '/'; break;
                    case 'u':
                        if (*(p+1) && *(p+2) && *(p+3) && *(p+4)) {
                            int hex = 0;
                            for (int h = 0; h < 4; h++) {
                                char c = *(p+1+h);
                                hex = hex * 16 + (c >= '0' && c <= '9' ? c - '0' :
                                          c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                                          c >= 'A' && c <= 'F' ? c - 'A' + 10 : 0);
                            }
                            if (hex < 0x80) key[ki++] = hex;
                            else if (hex < 0x800) { key[ki++] = 0xC0 | (hex >> 6); key[ki++] = 0x80 | (hex & 0x3F); }
                            else { key[ki++] = 0xE0 | (hex >> 12); key[ki++] = 0x80 | ((hex >> 6) & 0x3F); key[ki++] = 0x80 | (hex & 0x3F); }
                            p += 4;
                        }
                        break;
                    default: key[ki++] = *p; break;
                }
            } else {
                key[ki++] = *p;
            }
            p++;
        }
        key[ki] = '\0';
        if (*p != '"') break;
        p++;

        /* Skip to colon */
        while (*p && *p != ':') p++;
        if (*p != ':') break;
        p++;

        /* Skip whitespace */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;

        /* Parse integer token ID */
        int token_id = 0;
        while (*p >= '0' && *p <= '9') {
            token_id = token_id * 10 + (*p - '0');
            p++;
        }

        /* Skip comma */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;

        if (token_id >= 0 && token_id < (int)m->tok_n_tokens) {
            tok_arr[token_id].len = ki;
            tok_arr[token_id].str = malloc(ki + 1);
            if (tok_arr[token_id].str) {
                memcpy(tok_arr[token_id].str, key, ki);
                tok_arr[token_id].str[ki] = '\0';
            }
        }
    }
    free(vbuf);

    size_t total_size = 0;
    for (int i = 0; i < m->tok_n_tokens; i++) {
        total_size += tok_arr[i].str ? 8 + tok_arr[i].len : 8;
    }
    char *tok_buf = malloc(total_size + 1);
    if (!tok_buf) { free(tok_arr); return -1; }

    char *wp = tok_buf;
    for (int i = 0; i < m->tok_n_tokens; i++) {
        uint64_t slen = tok_arr[i].len;
        memcpy(wp, &slen, 8);
        wp += 8;
        if (tok_arr[i].str) {
            memcpy(wp, tok_arr[i].str, slen);
            wp += slen;
            free(tok_arr[i].str);
        }
    }
    *wp = '\0';
    free(tok_arr);
    m->tok_tokens_data = tok_buf;
    return 0;
}

int model_load_safetensors(model_t *m, const char *model_dir, int max_seq_len, kv_cache_type_t kv_type_k, kv_cache_type_t kv_type_v) {
    memset(m, 0, sizeof(*m));
    st_state_t st;

    if (load_config_safetensors(model_dir, &m->config) != 0) return -1;
    m->config.max_seq_len = max_seq_len > 0 ? max_seq_len : 4096;

    if (load_safetensors_files(model_dir, &st) != 0) return -1;
    if (map_tensors_safetensors(&st, m) != 0) return -1;

    /* Validate */
    layer_weights_t *lw = &m->weights.layers[0];
    if (m->config.has_ssm) {
        if (!lw->attn_qkv || !lw->attn_gate_ssm || !lw->ssm_a || !lw->ssm_alpha ||
            !lw->ssm_beta || !lw->ssm_conv1d || !lw->ssm_dt || !lw->ssm_norm || !lw->ssm_out) {
            fprintf(stderr, "Missing SSM tensors\n");
            return -1;
        }
    } else {
        if (!lw->attn_q || !lw->attn_k || !lw->attn_v || !lw->attn_output) {
            fprintf(stderr, "Missing attention tensors\n");
            return -1;
        }
    }

    m->from_safetensors = 1;
    if (allocate_run_state(m, kv_type_k, kv_type_v) != 0) return -1;
    if (load_tokenizer_safetensors(model_dir, m) != 0) return -1;

    m->mmap_addr = malloc(sizeof(st_state_t));
    if (!m->mmap_addr) return -1;
    memcpy(m->mmap_addr, &st, sizeof(st_state_t));

    return 0;
}
