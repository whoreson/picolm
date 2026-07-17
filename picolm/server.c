/*
 * PicoLM OpenAI-Compatible HTTP Server
 *
 * Supports:
 *   POST /v1/chat/completions  - Chat completions (with streaming)
 *   POST /v1/completions        - Raw completions (with streaming)
 *   GET  /v1/models             - List available models
 *
 * Build: compile with -DPICOLM_SERVER
 * Usage: ./picolm --server <model.gguf> [--port PORT] [--host HOST]
 */

#include "model.h"
#include "tensor.h"
#include "tokenizer.h"
#include "sampler.h"
#include "grammar.h"
#include "cJSON.h"
#include "qwen_tokenize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <errno.h>
#pragma comment(lib, "ws2_32.lib")
typedef int socklen_t;
#define SHUT_RD SD_RECEIVE
#define SHUT_WR SD_SEND
#define SHUT_RDWR SD_BOTH
/* On Windows, INVALID_SOCKET is the standard way to check for error */
#define SOCKET_INVALID INVALID_SOCKET
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <poll.h>
#define closesocket close
typedef int SOCKET;
#define SOCKET_INVALID ((SOCKET)-1)
#endif

#include "picolm_server.h"
#include <time.h>

/* Declared in picolm.c */
extern double get_time_ms(void);

#ifndef PICO_SERVER_BACKLOG
#define PICO_SERVER_BACKLOG 8
#endif

/* ---- HTTP Helpers ---- */

/* Check if the client is still connected.
 * Returns 1 if alive, 0 if the client has disconnected.
 * Uses a non-blocking poll + peek to avoid blocking. */
static int client_alive(SOCKET sock) {
#ifdef _WIN32
    /* On Windows, use select() to check readability */
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(sock, &rfds);
    if (select((int)sock + 1, &rfds, NULL, NULL, &tv) > 0) {
        /* Socket is readable - check if it's EOF or just data */
        char buf[1];
        int n = recv(sock, buf, 1, MSG_PEEK);
        if (n <= 0) return 0; /* EOF or error */
    }
    return 1;
#else
    struct pollfd pfd;
    pfd.fd = (int)sock;
    pfd.events = POLLIN;
    pfd.revents = 0;
    if (poll(&pfd, 1, 0) > 0 && (pfd.revents & (POLLIN | POLLHUP | POLLERR))) {
        /* Socket is readable - check if it's EOF */
        char buf[1];
        ssize_t n = recv(sock, buf, 1, MSG_PEEK);
        if (n <= 0) return 0; /* EOF or error */
    }
    return 1;
#endif
}

/* Send an HTTP response and close the connection. */
/* Send data with HTTP chunked transfer encoding framing.
 * Each call sends: chunk_length_hex\r\n data chunk\r\n
 * Returns 0 on success, -1 if the client has disconnected. */
static int send_chunked(SOCKET sock, const char *data, size_t len) {
    /* Send HTTP chunked encoding frame: hex-length + CRLF + data + CRLF */
    char hdr[32];
    int n = snprintf(hdr, sizeof(hdr), "%zx\r\n", (size_t)len);
    if (send(sock, hdr, (size_t)n, 0) < 0) return -1;
    if (len > 0) {
        if (send(sock, data, len, 0) < 0) return -1;
        if (send(sock, "\r\n", 2, 0) < 0) return -1;
    } else {
        /* Zero-length chunk: signals end of stream */
        if (send(sock, "\r\n", 2, 0) < 0) return -1;
    }
    return 0;
}

static void http_send(SOCKET sock, int status, const char *content_type, const char *body) {
    const char *status_text = "OK";
    switch (status) {
        case 200: status_text = "OK"; break;
        case 201: status_text = "Created"; break;
        case 400: status_text = "Bad Request"; break;
        case 404: status_text = "Not Found"; break;
        case 500: status_text = "Internal Server Error"; break;
    }

    char header[512];
    int n = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n",
        status, status_text, content_type, strlen(body));

    send(sock, header, n, 0);
    if (body) send(sock, body, strlen(body), 0);
}

/* Send SSE data chunk. Returns 0 on success, -1 if client disconnected. */
static int sse_send(SOCKET sock, const char *data, const char *event) {
    /* Build SSE message: "data: {json}\r\n\r\n" and send as one chunk */
    /* data may contain newlines; each gets "data: " prefix */
    int total_len = 0;
    if (event) {
        total_len += (int)(7 + strlen(event) + 2); /* "event: {ev}\r\n" */
    }
    /* Count data lines */
    const char *p = data;
    while (*p) {
        const char *nl = strchr(p, '\n');
        total_len += 6; /* "data: " */
        if (nl) {
            total_len += (int)(nl - p) + 2; /* line + "\r\n" */
            p = nl + 1;
        } else {
            total_len += (int)strlen(p) + 2; /* line + "\r\n" */
            break;
        }
    }
    total_len += 2; /* final "\r\n" */
    total_len++;     /* extra safety byte to prevent snprintf truncation */

    char *buf = (char *)calloc((size_t)total_len + 1, 1);
    int off = 0;
    if (event) {
        off += snprintf(buf + off, (size_t)(total_len - off), "event: %s\r\n", event);
    }
    p = data;
    while (*p) {
        const char *nl = strchr(p, '\n');
        if (nl) {
            off += snprintf(buf + off, (size_t)(total_len - off), "data: %.*s\r\n", (int)(nl - p), p);
            p = nl + 1;
        } else {
            off += snprintf(buf + off, (size_t)(total_len - off), "data: %s\r\n", p);
            break;
        }
    }
    off += snprintf(buf + off, (size_t)(total_len - off), "\r\n");
    int rc = send_chunked(sock, buf, (size_t)off);
    free(buf);
    return rc;
}

/* Forward declaration of global server state */
static struct _server_state {
    SOCKET listen_sock;
    int port;
    char host[256];
    const char *model_path;
    int running;
    char recv_buf[65536];
    SOCKET sock_in_buf;

    /* Persistent model state (loaded once, reused across requests) */
    int model_loaded;
    model_t model;
    tokenizer_t tokenizer;
    int use_qwen_tok;
    qwen_enc_t qwen_enc;
    int kv_pos;                /* current KV cache position for prompt caching */
    int *last_prompt_tokens;   /* cached prompt tokens for reuse */
    int last_prompt_len;       /* number of cached prompt tokens */
    int max_last_prompt;       /* allocated size of last_prompt_tokens */
} srv;

/* Parse HTTP request line: "METHOD /path HTTP/1.1" */
static int http_parse_request(char *buf, int len, char *method, int mlen, char *path, int plen) {
    method[0] = path[0] = '\0';
    /* Read until \r\n\r\n (end of headers) */
    int have = 0;
    while (have < len - 1) {
        int n = recv(srv.sock_in_buf, buf + have, len - have - 1, 0);
        if (n <= 0) return -1;
        have += n;
        buf[have] = '\0';
        if (strstr(buf, "\r\n\r\n")) break;
    }
    /* Extract method and path */
    char *sp1 = strchr(buf, ' ');
    if (!sp1) return -1;
    int mlen_actual = (int)(sp1 - buf);
    if (mlen_actual >= mlen) mlen_actual = mlen - 1;
    memcpy(method, buf, mlen_actual);
    method[mlen_actual] = '\0';

    char *sp2 = strchr(sp1 + 1, ' ');
    if (!sp2) return -1;
    int plen_actual = (int)(sp2 - sp1 - 1);
    if (plen_actual >= plen) plen_actual = plen - 1;
    memcpy(path, sp1 + 1, plen_actual);
    path[plen_actual] = '\0';

    return have;
}

/* ---- Global server state ---- */

/* ---- Endpoint: GET /v1/models ---- */

/* ---- Server init / free (persistent model) ---- */

static int server_init(const char *model_path, int num_threads, int do_prefault) {
    fprintf(stderr, "[server] Loading model: %s\n", model_path);
    fp16_table_init();

    if (model_load(&srv.model, model_path, 0, KV_CACHE_F16, KV_CACHE_F16) != 0) {
        fprintf(stderr, "[server] Failed to load model\n");
        return -1;
    }

    /* Print SIMD capability */
    fprintf(stderr, "[server] SIMD: %s\n",
#if defined(PICOLM_AVX512)
        "AVX-512"
#elif defined(PICOLM_AVX2)
        "AVX2"
#elif defined(PICOLM_AVX)
        "AVX"
#elif defined(PICOLM_SSE3)
        "SSE3"
#elif defined(PICOLM_SSE2)
        "SSE2"
#elif defined(PICOLM_NEON)
        "NEON"
#else
        "scalar"
#endif
    );

    /* Print weight types */
    fprintf(stderr, "[server] Weight type: %d, Emb type: %d, Output type: %d\n",
        srv.model.config.weight_type, srv.model.weights.type_token_embd, srv.model.weights.type_output);
    if (srv.model.config.n_layers > 0) {
        layer_weights_t *lw0 = &srv.model.weights.layers[0];
        fprintf(stderr, "[server] Layer0: attn_q=%d k=%d v=%d gate=%d up=%d down=%d\n",
            lw0->type_attn_q, lw0->type_attn_k, lw0->type_attn_v,
            lw0->type_ffn_gate, lw0->type_ffn_up, lw0->type_ffn_down);
    }

    /* Prefault if requested */
    if (do_prefault) {
        extern void model_prefault(model_t *m);
        model_prefault(&srv.model);
    }

    tensor_set_threads(num_threads);
    tensor_threadpool_init(num_threads);

    srv.use_qwen_tok = qwen_tokenize_should_use(&srv.model);
    if (srv.use_qwen_tok) {
        if (qwen_tokenize_init(&srv.qwen_enc, &srv.model) != 0) {
            fprintf(stderr, "[server] Failed to init Qwen tokenizer\n");
            model_free(&srv.model);
            return -1;
        }
        srv.tokenizer.bos_id = srv.qwen_enc.bos_id;
        srv.tokenizer.eos_id = srv.qwen_enc.bos_id;
        srv.tokenizer.vocab_size = srv.qwen_enc.vocab_size;
        fprintf(stderr, "[server] Using native Qwen GPT-2 BPE tokenizer\n");
    } else {
        if (tokenizer_load(&srv.tokenizer, &srv.model) != 0) {
            fprintf(stderr, "[server] Failed to load tokenizer\n");
            model_free(&srv.model);
            return -1;
        }
    }

    srv.model_loaded = 1;
    srv.kv_pos = 0;
    srv.last_prompt_len = 0;
    srv.last_prompt_tokens = NULL;
    srv.max_last_prompt = 0;
    return 0;
}

static void server_free(void) {
    if (srv.model_loaded) {
        tokenizer_free(&srv.tokenizer);
        tensor_threadpool_free();
        model_free(&srv.model);
        srv.model_loaded = 0;
    }
    free(srv.last_prompt_tokens);
    srv.last_prompt_tokens = NULL;
    srv.max_last_prompt = 0;
}

/* ---- Endpoints ---- */

static void handle_list_models(SOCKET sock, const char *model_path) {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "object", "list");

    cJSON *arr = cJSON_CreateArray();
    cJSON *m = cJSON_CreateObject();
    cJSON_AddStringToObject(m, "id", "pico");
    cJSON_AddStringToObject(m, "object", "model");
    cJSON_AddStringToObject(m, "owned_by", "pico");
    if (model_path)
        cJSON_AddStringToObject(m, "path", model_path);
    cJSON_AddItemToArray(arr, m);
    cJSON_AddItemToObject(root, "data", arr);

    char *json = cJSON_PrintUnformatted(root);
    http_send(sock, 200, "application/json", json);
    free(json);
    cJSON_Delete(root);
}

/* ---- Endpoint: GET /props ---- */

static void handle_props(SOCKET sock) {
    if (!srv.model_loaded) {
        http_send(sock, 503, "application/json", "{\"error\":\"Model not loaded\"}");
        return;
    }
    model_t *model = &srv.model;
    tokenizer_t *tokenizer = &srv.tokenizer;

    const char *fname = srv.model_path;
    const char *slash = strrchr(srv.model_path, '/');
    if (!slash) slash = strrchr(srv.model_path, '\\');
    if (slash) fname = slash + 1;

    cJSON *root = cJSON_CreateObject();

    cJSON *dgs = cJSON_CreateObject();
    cJSON *params = cJSON_CreateObject();

    cJSON_AddNumberToObject(params, "seed", 4294967295);
    cJSON_AddNumberToObject(params, "temperature", 1.0);
    cJSON_AddNumberToObject(params, "dynatemp_range", 0.0);
    cJSON_AddNumberToObject(params, "dynatemp_exponent", 1.0);
    cJSON_AddNumberToObject(params, "top_k", 40);
    cJSON_AddNumberToObject(params, "top_p", 0.95);
    cJSON_AddNumberToObject(params, "min_p", 0.05);
    cJSON_AddNumberToObject(params, "top_n_sigma", -1.0);
    cJSON_AddNumberToObject(params, "xtc_probability", 0.0);
    cJSON_AddNumberToObject(params, "xtc_threshold", 0.1);
    cJSON_AddNumberToObject(params, "typical_p", 1.0);
    cJSON_AddNumberToObject(params, "repeat_last_n", 64);
    cJSON_AddNumberToObject(params, "repeat_penalty", 1.1);
    cJSON_AddNumberToObject(params, "presence_penalty", 0.0);
    cJSON_AddNumberToObject(params, "frequency_penalty", 0.0);
    cJSON_AddNumberToObject(params, "dry_multiplier", 0.0);
    cJSON_AddNumberToObject(params, "dry_base", 1.75);
    cJSON_AddNumberToObject(params, "dry_allowed_length", 2);
    cJSON_AddNumberToObject(params, "dry_penalty_last_n", -1);
    cJSON_AddNumberToObject(params, "mirostat", 0);
    cJSON_AddNumberToObject(params, "mirostat_tau", 5.0);
    cJSON_AddNumberToObject(params, "mirostat_eta", 0.1);
    cJSON_AddNumberToObject(params, "max_tokens", -1);
    cJSON_AddNumberToObject(params, "n_predict", -1);
    cJSON_AddNumberToObject(params, "n_keep", 0);
    cJSON_AddNumberToObject(params, "n_discard", 0);
    cJSON_AddBoolToObject(params, "ignore_eos", 0);
    cJSON_AddBoolToObject(params, "stream", 0);
    cJSON_AddNumberToObject(params, "n_probs", 0);
    cJSON_AddNumberToObject(params, "min_keep", 0);
    cJSON_AddStringToObject(params, "chat_format", "Content-only");
    cJSON_AddStringToObject(params, "reasoning_format", "none");
    cJSON_AddBoolToObject(params, "reasoning_in_content", 0);
    cJSON_AddStringToObject(params, "generation_prompt", "");
    cJSON_AddNumberToObject(params, "timings_per_token", 0);
    cJSON_AddNumberToObject(params, "post_sampling_probs", 0);
    cJSON_AddNumberToObject(params, "backend_sampling", 0);
    cJSON_AddNumberToObject(params, "lora", 0);
    cJSON_AddItemToObject(dgs, "params", params);

    /* n_ctx from model config */
    cJSON_AddNumberToObject(dgs, "n_ctx", model->config.max_seq_len);
    cJSON_AddItemToObject(root, "default_generation_settings", dgs);

    /* Model metadata */
    cJSON_AddNumberToObject(root, "total_slots", 1);
    cJSON_AddStringToObject(root, "model_alias", fname);
    cJSON_AddStringToObject(root, "model_path", srv.model_path);

    /* Modalities */
    cJSON *modalities = cJSON_CreateObject();
    cJSON_AddBoolToObject(modalities, "vision", 0);
    cJSON_AddBoolToObject(modalities, "audio", 0);
    cJSON_AddItemToObject(root, "modalities", modalities);

    /* Media marker (placeholder) */
    cJSON_AddStringToObject(root, "media_marker", "<__media__>");

    /* Endpoints */
    cJSON_AddBoolToObject(root, "endpoint_slots", 0);
    cJSON_AddBoolToObject(root, "endpoint_props", 1);
    cJSON_AddBoolToObject(root, "endpoint_metrics", 0);
    cJSON_AddBoolToObject(root, "webui", 0);
    cJSON_AddNumberToObject(root, "webui_settings", 0);

    /* Chat template - generic template based on model's tokenizer */
    /* We don't have a proper Jinja2 chat template, use a simple one */
    const char *chat_tpl = "{% if messages %}{% for message in messages %}{{ '### ' + message.role + '\\n' + message.content + '\\n\\n' }}{% endfor %}{% if add_generation_prompt %}### assistant\\n{% endif %}{% endif %}";
    cJSON_AddStringToObject(root, "chat_template", chat_tpl);

    /* chat_template_caps */
    cJSON *caps = cJSON_CreateObject();
    cJSON_AddBoolToObject(caps, "supports_object_arguments", 0);
    cJSON_AddBoolToObject(caps, "supports_parallel_tool_calls", 0);
    cJSON_AddBoolToObject(caps, "supports_preserve_reasoning", 0);
    cJSON_AddBoolToObject(caps, "supports_string_content", 1);
    cJSON_AddBoolToObject(caps, "supports_system_role", 1);
    cJSON_AddBoolToObject(caps, "supports_tool_calls", 0);
    cJSON_AddBoolToObject(caps, "supports_tools", 0);
    cJSON_AddBoolToObject(caps, "supports_typed_content", 0);
    cJSON_AddItemToObject(root, "chat_template_caps", caps);

    /* BOS/EOS tokens */
    {
        char bos_buf[32], eos_buf[32];
        snprintf(bos_buf, sizeof(bos_buf), "<bos_id=%d>", tokenizer->bos_id);
        cJSON_AddStringToObject(root, "bos_token", bos_buf);
        snprintf(eos_buf, sizeof(eos_buf), "<eos_id=%d>", tokenizer->eos_id);
        cJSON_AddStringToObject(root, "eos_token", eos_buf);
    }

    /* Build info */
    cJSON_AddStringToObject(root, "build_info", "picolm");
    cJSON_AddBoolToObject(root, "is_sleeping", 0);

    char *json = cJSON_PrintUnformatted(root);
    http_send(sock, 200, "application/json", json);
    free(json);
    cJSON_Delete(root);
}

/* ---- Build prompt from chat messages or raw text ---- */

/* Convert chat messages to a prompt string.
 * Simple template: each message becomes "### {role}\n{text}\n\n".
 * Returns a malloc'd string. */
static char *chat_to_prompt(cJSON *messages, const char *system_template) {
    int total_len = 0;
    int n_msgs = cJSON_GetArraySize(messages);

    /* First pass: calculate total length */
    for (int i = 0; i < n_msgs; i++) {
        cJSON *msg = cJSON_GetArrayItem(messages, i);
        cJSON *role = cJSON_GetObjectItem(msg, "role");
        cJSON *content = cJSON_GetObjectItem(msg, "content");
        if (!role || !content) continue;
        const char *rs = cJSON_GetStringValue(role);
        const char *cs = cJSON_GetStringValue(content);
        if (!rs || !cs) continue;
        total_len += strlen(rs) + strlen(cs) + 32; /* role header + newline + content + spacing */
    }

    char *prompt = (char *)malloc((size_t)total_len + 1);
    if (!prompt) return NULL;
    prompt[0] = '\0';

    /* Second pass: build the prompt */
    for (int i = 0; i < n_msgs; i++) {
        cJSON *msg = cJSON_GetArrayItem(messages, i);
        cJSON *role = cJSON_GetObjectItem(msg, "role");
        cJSON *content = cJSON_GetObjectItem(msg, "content");
        if (!role || !content) continue;
        const char *rs = cJSON_GetStringValue(role);
        const char *cs = cJSON_GetStringValue(content);
        if (!rs || !cs) continue;

        if (strcmp(rs, "system") == 0) {
            if (system_template && strlen(system_template) > 0) {
                snprintf(prompt + strlen(prompt), total_len - strlen(prompt),
                         system_template, cs);
            } else {
                strcat(prompt, "### system\n");
                strcat(prompt, cs);
                strcat(prompt, "\n\n");
            }
        } else {
            strcat(prompt, "### ");
            strcat(prompt, rs);
            strcat(prompt, "\n");
            strcat(prompt, cs);
            strcat(prompt, "\n\n");
        }
    }

    return prompt;
}

/* ---- Endpoint: POST /v1/completions and /v1/chat/completions ---- */

static void handle_completion(SOCKET sock, const char *request_body, int is_chat) {
    model_t *model = &srv.model;
    tokenizer_t *tokenizer = &srv.tokenizer;

    cJSON *req = cJSON_ParseWithLength(request_body, strlen(request_body));
    if (!req) {
        http_send(sock, 400, "application/json", "{\"error\":{\"message\":\"Invalid JSON\"}}");
        return;
    }

    const char *prompt = NULL;
    char *chat_prompt = NULL;
    char *raw_prompt_copy = NULL;
    int max_tokens = 256;
    float temperature = 0.8f;
    float top_p = 0.9f;
    uint64_t seed = 42;
    int n_choices = 1;
    int do_stream = 0;
    int timings_per_token = 0;
    const char *model_name = NULL;

    if (is_chat) {
        cJSON *messages = cJSON_GetObjectItem(req, "messages");
        if (!messages || !cJSON_IsArray(messages) || cJSON_GetArraySize(messages) == 0) {
            http_send(sock, 400, "application/json", "{\"error\":{\"message\":\"'messages' required\"}}");
            cJSON_Delete(req);
            return;
        }
        const char *sys_template = NULL;
        cJSON *sti = cJSON_GetObjectItem(req, "system_template");
        if (sti) sys_template = cJSON_GetStringValue(sti);

        chat_prompt = chat_to_prompt(messages, sys_template);
        if (!chat_prompt) {
            http_send(sock, 400, "application/json", "{\"error\":{\"message\":\"Failed to build prompt\"}}");
            cJSON_Delete(req);
            return;
        }
        prompt = chat_prompt;
    } else {
        cJSON *p = cJSON_GetObjectItem(req, "prompt");
        if (!p || !cJSON_IsString(p)) {
            http_send(sock, 400, "application/json", "{\"error\":{\"message\":\"'prompt' required\"}}");
            cJSON_Delete(req);
            return;
        }
        raw_prompt_copy = (char *)malloc(strlen(cJSON_GetStringValue(p)) + 1);
        strcpy(raw_prompt_copy, cJSON_GetStringValue(p));
        prompt = raw_prompt_copy;
    }

    cJSON *item;
    if ((item = cJSON_GetObjectItem(req, "max_tokens"))) max_tokens = (int)cJSON_GetNumberValue(item);
    int prompt_only = (max_tokens == 0);
    if (max_tokens <= 0 && !prompt_only) max_tokens = 256;
    if ((item = cJSON_GetObjectItem(req, "temperature"))) temperature = (float)cJSON_GetNumberValue(item);
    if ((item = cJSON_GetObjectItem(req, "top_p"))) top_p = (float)cJSON_GetNumberValue(item);
    if ((item = cJSON_GetObjectItem(req, "seed"))) seed = (uint64_t)cJSON_GetNumberValue(item);
    if ((item = cJSON_GetObjectItem(req, "stream")) && cJSON_IsTrue(item)) do_stream = 1;
    if ((item = cJSON_GetObjectItem(req, "timings_per_token")) && cJSON_IsTrue(item)) timings_per_token = 1;
    if ((item = cJSON_GetObjectItem(req, "n"))) n_choices = (int)cJSON_GetNumberValue(item);
    if (n_choices < 1) n_choices = 1;
    if (n_choices > 4) n_choices = 4;
    if ((item = cJSON_GetObjectItem(req, "model"))) model_name = cJSON_GetStringValue(item);
    cJSON_Delete(req);

    /* Encode prompt */
    int max_pt = (int)strlen(prompt) + 3;
    int *ptokens = (int *)malloc((size_t)max_pt * sizeof(int));
    int n_prompt;
    if (srv.use_qwen_tok) {
        n_prompt = qwen_tokenize_encode(&srv.qwen_enc, prompt, ptokens, max_pt);
    } else {
        n_prompt = tokenizer_encode(tokenizer, prompt, ptokens, max_pt, 1);
    }

    /* Prompt caching: find shared prefix with last request */
    int cache_start = 0;
    if (srv.last_prompt_tokens) {
        int min_len = n_prompt < srv.last_prompt_len ? n_prompt : srv.last_prompt_len;
        for (cache_start = 0; cache_start < min_len; cache_start++) {
            if (ptokens[cache_start] != srv.last_prompt_tokens[cache_start]) break;
        }
    }
    int start_pos = cache_start;
    if (start_pos == 0) {
        srv.kv_pos = 0;
    }

    /* Save prompt tokens for next request */
    if (n_prompt > srv.max_last_prompt) {
        srv.max_last_prompt = n_prompt * 2;
        srv.last_prompt_tokens = (int *)realloc(srv.last_prompt_tokens, (size_t)srv.max_last_prompt * sizeof(int));
    }
    memcpy(srv.last_prompt_tokens, ptokens, (size_t)n_prompt * sizeof(int));
    srv.last_prompt_len = n_prompt;

    /* Helper: send SSE chunk */
    int send_chunk(SOCKET sk, const char *content, const char *role, int is_done) {
        cJSON *chunk = cJSON_CreateObject();
        cJSON *choice = cJSON_CreateObject();
        cJSON *delta = cJSON_CreateObject();
        if (role && role[0]) cJSON_AddStringToObject(delta, "role", role);
        cJSON_AddStringToObject(delta, "content", content);
        cJSON_AddItemToObject(choice, "delta", delta);
        cJSON_AddNumberToObject(choice, "index", 0);
        cJSON_AddStringToObject(choice, "finish_reason", is_done ? "stop" : "");
        if (!is_chat) cJSON_AddStringToObject(choice, "text", content);
        cJSON *choices = cJSON_CreateArray();
        cJSON_AddItemToArray(choices, choice);
        cJSON_AddItemToObject(chunk, "choices", choices);
        cJSON_AddStringToObject(chunk, "id", is_chat ? "chatcmpl-pico" : "cmpl-pico");
        cJSON_AddStringToObject(chunk, "object", is_chat ? "chat.completion.chunk" : "text_completion");
        cJSON_AddNumberToObject(chunk, "created", (double)(time(NULL)));
        if (model_name) cJSON_AddStringToObject(chunk, "model", model_name);
        char *json = cJSON_PrintUnformatted(chunk);
        int rc = sse_send(sk, json, NULL);
        free(json);
        cJSON_Delete(chunk);
        return rc;
    }

    int total_prompt_tokens = n_prompt;
    int total_generation_tokens = 0;

    if (do_stream) {
        char hdr[256];
        snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Transfer-Encoding: chunked\r\n"
            
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n");
        send(sock, hdr, strlen(hdr), 0);

        sampler_t sampler;
        sampler_init(&sampler, temperature, top_p, seed);

        int gen_start_oai = start_pos;
        if (gen_start_oai >= n_prompt) gen_start_oai = n_prompt - 1;
        int token = ptokens[gen_start_oai];
        double t_start = get_time_ms();
        double t_prefill_end = 0;
        int gen_count = 0;

        for (int pos = gen_start_oai; pos < n_prompt + max_tokens; pos++) {
            if (!client_alive(sock)) {
                fprintf(stderr, "\n[server] OAI stream: client disconnected at pos %d\n", pos);
                break;
            }
            float *logits = model_forward(model, token, pos);

            if (pos < n_prompt - 1 || prompt_only) {
                token = ptokens[pos + 1];
                continue;
            }

            /* First generated token: mark prefill end */
            if (gen_count == 0) {
                t_prefill_end = get_time_ms();
            }

            int next = sampler_sample(&sampler, logits, model->config.vocab_size);
            static char qwen_buf[64];
            const char *piece;
            if (srv.use_qwen_tok) {
                qwen_tokenize_decode(&srv.qwen_enc, next, qwen_buf, sizeof(qwen_buf));
                piece = qwen_buf;
            } else {
                piece = tokenizer_decode(tokenizer, token, next);
            }
            if (!piece) piece = "";

            /* Build chunk with optional timing */
            cJSON *chunk = cJSON_CreateObject();
            cJSON *choice = cJSON_CreateObject();
            cJSON *delta = cJSON_CreateObject();
            if (gen_count == 0) cJSON_AddStringToObject(delta, "role", is_chat ? "assistant" : "");
            cJSON_AddStringToObject(delta, "content", piece);
            cJSON_AddItemToObject(choice, "delta", delta);
            cJSON_AddNumberToObject(choice, "index", 0);
            cJSON_AddStringToObject(choice, "finish_reason", "");
            if (!is_chat) cJSON_AddStringToObject(choice, "text", piece);
            cJSON *choices = cJSON_CreateArray();
            cJSON_AddItemToArray(choices, choice);
            cJSON_AddItemToObject(chunk, "choices", choices);
            cJSON_AddStringToObject(chunk, "id", is_chat ? "chatcmpl-pico" : "cmpl-pico");
            cJSON_AddStringToObject(chunk, "object", is_chat ? "chat.completion.chunk" : "text_completion");
            cJSON_AddNumberToObject(chunk, "created", (double)(time(NULL)));
            if (model_name) cJSON_AddStringToObject(chunk, "model", model_name);

            /* Per-token timing */
            if (timings_per_token && gen_count > 0) {
                double t_now = get_time_ms();
                double tok_ms = t_now - (gen_count == 1 ? t_prefill_end : t_now);
                /* Actually track per-token delta */
                cJSON *tt = cJSON_CreateObject();
                cJSON_AddNumberToObject(tt, "token_ms", tok_ms);
                cJSON_AddItemToObject(chunk, "timings", tt);
            }
            double t_tok_start = gen_count > 0 ? t_prefill_end : get_time_ms();
            (void)t_tok_start; /* used for per-token tracking below */

            char *cj = cJSON_PrintUnformatted(chunk);
            if (sse_send(sock, cj, NULL) < 0) {
                free(cj); cJSON_Delete(chunk); break;
            }
            free(cj);
            cJSON_Delete(chunk);
            total_generation_tokens++;
            gen_count++;
            printf("%s", piece);
            fflush(stdout);

            if (next == (int)tokenizer->eos_id) break;
            token = next;
        }

        double t_end = get_time_ms();
        double t_prefill_ms = t_prefill_end > 0 ? t_prefill_end - t_start : t_end - t_start;
        double t_gen_ms = t_prefill_end > 0 ? t_end - t_prefill_end : 0;
        int prompt_evaluated = n_prompt - start_pos;

        /* Send final chunk with timings */
        cJSON *done_chunk = cJSON_CreateObject();
        cJSON *done_choice = cJSON_CreateObject();
        cJSON *done_delta = cJSON_CreateObject();
        cJSON_AddStringToObject(done_delta, "content", "");
        cJSON_AddItemToObject(done_choice, "delta", done_delta);
        cJSON_AddNumberToObject(done_choice, "index", 0);
        cJSON_AddStringToObject(done_choice, "finish_reason", "stop");
        cJSON *done_choices = cJSON_CreateArray();
        cJSON_AddItemToArray(done_choices, done_choice);
        cJSON_AddItemToObject(done_chunk, "choices", done_choices);
        cJSON_AddStringToObject(done_chunk, "id", is_chat ? "chatcmpl-pico" : "cmpl-pico");
        cJSON_AddStringToObject(done_chunk, "object", is_chat ? "chat.completion.chunk" : "text_completion");
        cJSON_AddNumberToObject(done_chunk, "created", (double)(time(NULL)));

        /* Timings object */
        cJSON *timings = cJSON_CreateObject();
        cJSON_AddNumberToObject(timings, "cache_n", start_pos);
        cJSON_AddNumberToObject(timings, "prompt_n", prompt_evaluated);
        cJSON_AddNumberToObject(timings, "prompt_ms", t_prefill_ms);
        cJSON_AddNumberToObject(timings, "prompt_per_token_ms", prompt_evaluated > 0 ? t_prefill_ms / prompt_evaluated : 0);
        cJSON_AddNumberToObject(timings, "prompt_per_second", prompt_evaluated > 0 ? prompt_evaluated / (t_prefill_ms / 1000.0) : 0);
        cJSON_AddNumberToObject(timings, "predicted_n", total_generation_tokens);
        cJSON_AddNumberToObject(timings, "predicted_ms", t_gen_ms);
        cJSON_AddNumberToObject(timings, "predicted_per_token_ms", total_generation_tokens > 0 ? t_gen_ms / total_generation_tokens : 0);
        cJSON_AddNumberToObject(timings, "predicted_per_second", total_generation_tokens > 0 ? total_generation_tokens / (t_gen_ms / 1000.0) : 0);
        cJSON_AddItemToObject(done_chunk, "timings", timings);

        char *dcj = cJSON_PrintUnformatted(done_chunk);
        sse_send(sock, dcj, NULL);
        free(dcj);
        cJSON_Delete(done_chunk);
        send_chunked(sock, "data: [DONE]\r\n", 14);
        send_chunked(sock, "", 0);

        srv.kv_pos = n_prompt;

        fprintf(stderr, "\n[server] OAI stream: prompt %d new+%d cached in %.0fms (%.1f tok/s), gen %d in %.0fms (%.1f tok/s)\n",
                prompt_evaluated, start_pos, t_prefill_ms,
                prompt_evaluated > 0 ? prompt_evaluated / (t_prefill_ms / 1000.0) : 0,
                total_generation_tokens, t_gen_ms,
                total_generation_tokens > 0 ? total_generation_tokens / (t_gen_ms / 1000.0) : 0);
    } else {
        /* Non-streaming */
        cJSON *root = cJSON_CreateObject();
        cJSON *choices = cJSON_CreateArray();
        double t_start_ns = get_time_ms();
        double t_prefill_end_ns = 0;
        int gen_count_ns = 0;

        for (int c = 0; c < n_choices; c++) {
            sampler_t sampler;
            sampler_init(&sampler, temperature, top_p, seed + (uint64_t)c);

            int gen_start_oai2 = start_pos;
            if (gen_start_oai2 >= n_prompt) gen_start_oai2 = n_prompt - 1;
            int token = ptokens[gen_start_oai2];
            char *generated = (char *)malloc((size_t)max_tokens * 100 + 1);
            if (!generated) {
                free(chat_prompt); free(raw_prompt_copy); free(ptokens);
                http_send(sock, 500, "application/json", "{\"error\":{\"message\":\"OOM\"}}");
                return;
            }
            generated[0] = '\0';
            int gen_count = 0;
            const char *finish_reason = "stop";

            for (int pos = gen_start_oai2; pos < n_prompt + max_tokens; pos++) {
                if (!client_alive(sock)) {
                    free(generated); break;
                }
                float *logits = model_forward(model, token, pos);

                if (pos < n_prompt - 1 || prompt_only) {
                    token = ptokens[pos + 1];
                    continue;
                }
                if (gen_count_ns == 0) t_prefill_end_ns = get_time_ms();
                gen_count_ns++;

                int next = sampler_sample(&sampler, logits, model->config.vocab_size);
                static char qwen_buf2[64];
                const char *piece;
                if (srv.use_qwen_tok) {
                    qwen_tokenize_decode(&srv.qwen_enc, next, qwen_buf2, sizeof(qwen_buf2));
                    piece = qwen_buf2;
                } else {
                    piece = tokenizer_decode(tokenizer, token, next);
                }
                if (!piece) piece = "";
                strncat(generated, piece, max_tokens * 100 - strlen(generated) - 1);
                gen_count++;
                printf("%s", piece);
                fflush(stdout);

                if (next == (int)tokenizer->eos_id) { finish_reason = "stop"; break; }
                if (gen_count >= max_tokens) { finish_reason = "length"; break; }
                token = next;
            }
            total_generation_tokens += gen_count;

            cJSON *choice = cJSON_CreateObject();
            cJSON_AddNumberToObject(choice, "index", c);
            if (is_chat) {
                cJSON *msg = cJSON_CreateObject();
                cJSON_AddStringToObject(msg, "role", "assistant");
                cJSON_AddStringToObject(msg, "content", generated);
                cJSON_AddItemToObject(choice, "message", msg);
            } else {
                cJSON_AddStringToObject(choice, "text", generated);
            }
            cJSON_AddStringToObject(choice, "finish_reason", finish_reason);
            cJSON_AddItemToArray(choices, choice);
            free(generated);
        }

        cJSON_AddItemToObject(root, "choices", choices);
        cJSON_AddStringToObject(root, "id", is_chat ? "chatcmpl-pico" : "cmpl-pico");
        cJSON_AddStringToObject(root, "object", is_chat ? "chat.completion" : "text_completion");
        cJSON_AddNumberToObject(root, "created", (double)(time(NULL)));
        if (model_name) cJSON_AddStringToObject(root, "model", model_name);
        else cJSON_AddStringToObject(root, "model", "pico");

        cJSON *usage = cJSON_CreateObject();
        cJSON_AddNumberToObject(usage, "prompt_tokens", total_prompt_tokens);
        cJSON_AddNumberToObject(usage, "completion_tokens", total_generation_tokens);
        cJSON_AddNumberToObject(usage, "total_tokens", total_prompt_tokens + total_generation_tokens);
        cJSON_AddItemToObject(root, "usage", usage);

        /* Timings */
        double t_end_ns = get_time_ms();
        double t_prefill_ms_ns = t_prefill_end_ns > 0 ? t_prefill_end_ns - t_start_ns : t_end_ns - t_start_ns;
        double t_gen_ms_ns = t_prefill_end_ns > 0 ? t_end_ns - t_prefill_end_ns : 0;
        int prompt_evaluated_ns = n_prompt - start_pos;
        cJSON *timings_ns = cJSON_CreateObject();
        cJSON_AddNumberToObject(timings_ns, "cache_n", start_pos);
        cJSON_AddNumberToObject(timings_ns, "prompt_n", prompt_evaluated_ns);
        cJSON_AddNumberToObject(timings_ns, "prompt_ms", t_prefill_ms_ns);
        cJSON_AddNumberToObject(timings_ns, "prompt_per_token_ms", prompt_evaluated_ns > 0 ? t_prefill_ms_ns / prompt_evaluated_ns : 0);
        cJSON_AddNumberToObject(timings_ns, "prompt_per_second", prompt_evaluated_ns > 0 ? prompt_evaluated_ns / (t_prefill_ms_ns / 1000.0) : 0);
        cJSON_AddNumberToObject(timings_ns, "predicted_n", total_generation_tokens);
        cJSON_AddNumberToObject(timings_ns, "predicted_ms", t_gen_ms_ns);
        cJSON_AddNumberToObject(timings_ns, "predicted_per_token_ms", total_generation_tokens > 0 ? t_gen_ms_ns / total_generation_tokens : 0);
        cJSON_AddNumberToObject(timings_ns, "predicted_per_second", total_generation_tokens > 0 ? total_generation_tokens / (t_gen_ms_ns / 1000.0) : 0);
        cJSON_AddItemToObject(root, "timings", timings_ns);

        char *json = cJSON_PrintUnformatted(root);
        http_send(sock, 200, "application/json", json);
        free(json);
        cJSON_Delete(root);

        srv.kv_pos = n_prompt;

        fprintf(stderr, "\n[server] OAI: prompt %d new+%d cached in %.0fms (%.1f tok/s), gen %d in %.0fms (%.1f tok/s)\n",
                prompt_evaluated_ns, start_pos, t_prefill_ms_ns,
                prompt_evaluated_ns > 0 ? prompt_evaluated_ns / (t_prefill_ms_ns / 1000.0) : 0,
                total_generation_tokens, t_gen_ms_ns,
                total_generation_tokens > 0 ? total_generation_tokens / (t_gen_ms_ns / 1000.0) : 0);

    }

    printf("\n");
    fflush(stdout);
    free(raw_prompt_copy);
    free(chat_prompt);
    free(ptokens);
}

/* ---- Endpoint: POST /completion (llama.cpp-style) ---- */

/* Handle the non-OAI /completion endpoint.
 * This matches the llama.cpp server format with content, tokens, stop,
 * stop_type, generation_settings, model, timings, etc. */
static void handle_llama_completion(SOCKET sock, const char *request_body) {

    cJSON *req = cJSON_ParseWithLength(request_body, strlen(request_body));
    if (!req) {
        http_send(sock, 400, "application/json", "{\"error\":\"Invalid JSON\"}");
        return;
    }

    /* Extract prompt - supports string or array of tokens */
    const char *prompt = NULL;
    int *token_prompt = NULL;
    int n_token_prompt = 0;
    int prompt_is_tokens = 0;

    cJSON *p = cJSON_GetObjectItem(req, "prompt");
    if (!p) {
        cJSON_Delete(req);
        http_send(sock, 400, "application/json", "{\"error\":\"'prompt' is required\"}");
        return;
    }

    if (cJSON_IsString(p)) {
        prompt = cJSON_GetStringValue(p);
    } else if (cJSON_IsArray(p)) {
        /* Token array prompt */
        n_token_prompt = cJSON_GetArraySize(p);
        token_prompt = (int *)malloc((size_t)n_token_prompt * sizeof(int));
        for (int i = 0; i < n_token_prompt; i++) {
            cJSON *item = cJSON_GetArrayItem(p, i);
            if (cJSON_IsNumber(item))
                token_prompt[i] = (int)cJSON_GetNumberValue(item);
            else if (cJSON_IsString(item)) {
                /* Mixed array with strings - skip for now, require pure tokens */
            }
        }
        prompt_is_tokens = 1;
    } else {
        cJSON_Delete(req);
        http_send(sock, 400, "application/json", "{\"error\":\"'prompt' must be a string or token array\"}");
        return;
    }

    /* Parameters (llama.cpp naming) */
    int n_predict = -1; /* -1 = unlimited (use max_seq_len) */
    float temperature = 0.8f;
    float top_p = 0.95f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int repeat_last_n = 64;
    int seed = -1; /* -1 = random */
    int do_stream = 0;
    int return_tokens = 0;
    int ignore_eos = 0;
    const char *model_name = NULL;

    cJSON *item;
    if ((item = cJSON_GetObjectItem(req, "n_predict")))
        n_predict = (int)cJSON_GetNumberValue(item);
    if ((item = cJSON_GetObjectItem(req, "temperature")))
        temperature = (float)cJSON_GetNumberValue(item);
    if ((item = cJSON_GetObjectItem(req, "top_p")))
        top_p = (float)cJSON_GetNumberValue(item);
    if ((item = cJSON_GetObjectItem(req, "top_k")))
        top_k = (int)cJSON_GetNumberValue(item);
    if ((item = cJSON_GetObjectItem(req, "repeat_penalty")))
        repeat_penalty = (float)cJSON_GetNumberValue(item);
    if ((item = cJSON_GetObjectItem(req, "repeat_last_n")))
        repeat_last_n = (int)cJSON_GetNumberValue(item);
    if ((item = cJSON_GetObjectItem(req, "seed"))) {
        seed = (int)cJSON_GetNumberValue(item);
    }
    if ((item = cJSON_GetObjectItem(req, "stream")) && cJSON_IsTrue(item))
        do_stream = 1;
    if ((item = cJSON_GetObjectItem(req, "return_tokens")) && cJSON_IsTrue(item))
        return_tokens = 1;
    if ((item = cJSON_GetObjectItem(req, "ignore_eos")) && cJSON_IsTrue(item))
        ignore_eos = 1;
    if ((item = cJSON_GetObjectItem(req, "model")))
        model_name = cJSON_GetStringValue(item);

    /* Parse stop words - MUST copy before cJSON_Delete */
    char *stop_words[16];
    int n_stop_words = 0;
    cJSON *stop_arr = cJSON_GetObjectItem(req, "stop");
    if (stop_arr && cJSON_IsArray(stop_arr)) {
        int n = cJSON_GetArraySize(stop_arr);
        if (n > 16) n = 16;
        for (int i = 0; i < n; i++) {
            cJSON *sw = cJSON_GetArrayItem(stop_arr, i);
            if (cJSON_IsString(sw)) {
                const char *s = cJSON_GetStringValue(sw);
                if (s && strlen(s) > 0) {
                    stop_words[n_stop_words++] = strdup(s);
                }
            }
        }
    }

    char *prompt_copy = NULL;
    if (prompt) {
        prompt_copy = (char *)malloc(strlen(prompt) + 1);
        strcpy(prompt_copy, prompt);
        prompt = prompt_copy;
    }
    cJSON_Delete(req);

    model_t *model = &srv.model;
    tokenizer_t *tokenizer = &srv.tokenizer;

    /* Encode prompt */
    int *ptokens = NULL;
    int n_prompt = 0;
    if (prompt_is_tokens) {
        ptokens = token_prompt;
        n_prompt = n_token_prompt;
        token_prompt = NULL; /* transferred ownership */
    } else {
        int max_pt = (int)strlen(prompt) + 3;
        ptokens = (int *)malloc((size_t)max_pt * sizeof(int));
        if (srv.use_qwen_tok) {
            n_prompt = qwen_tokenize_encode(&srv.qwen_enc, prompt, ptokens, max_pt);
        } else {
            n_prompt = tokenizer_encode(tokenizer, prompt, ptokens, max_pt, 1);
        }
    }

    /* Prompt caching: find shared prefix */
    int cache_start = 0;
    if (srv.last_prompt_tokens) {
        int min_len = n_prompt < srv.last_prompt_len ? n_prompt : srv.last_prompt_len;
        for (cache_start = 0; cache_start < min_len; cache_start++) {
            if (ptokens[cache_start] != srv.last_prompt_tokens[cache_start]) break;
        }
    }
    /* Use shared prefix from KV cache. The attention loop iterates
     * from t=0 to t<=pos, so stale KV data at positions beyond n_prompt
     * is never read (it will be overwritten during generation). */
    int start_pos = cache_start;
    if (start_pos == 0) {
        srv.kv_pos = 0;
    }

    /* Save prompt tokens for next request */
    if (n_prompt > srv.max_last_prompt) {
        srv.max_last_prompt = n_prompt * 2;
        srv.last_prompt_tokens = (int *)realloc(srv.last_prompt_tokens, (size_t)srv.max_last_prompt * sizeof(int));
    }
    memcpy(srv.last_prompt_tokens, ptokens, (size_t)n_prompt * sizeof(int));
    srv.last_prompt_len = n_prompt;

    int prompt_only = (n_predict == 0);
    if (n_predict < 0) n_predict = model->config.max_seq_len - n_prompt;
    if (n_predict <= 0 && !prompt_only) n_predict = 32;

    /* Random seed */
    if (seed < 0) seed = (uint64_t)(time(NULL) ^ (uint64_t)((long)&seed));

    sampler_t sampler;
    sampler_init(&sampler, temperature, top_p, (uint64_t)seed);

    /* Build generation_settings JSON */
    cJSON *gen_settings = cJSON_CreateObject();
    cJSON_AddNumberToObject(gen_settings, "n_predict", n_predict);
    cJSON_AddNumberToObject(gen_settings, "temperature", temperature);
    cJSON_AddNumberToObject(gen_settings, "top_p", top_p);
    cJSON_AddNumberToObject(gen_settings, "top_k", top_k);
    cJSON_AddNumberToObject(gen_settings, "repeat_penalty", repeat_penalty);
    cJSON_AddNumberToObject(gen_settings, "repeat_last_n", repeat_last_n);
    cJSON_AddNumberToObject(gen_settings, "seed", seed);

    double t_start = get_time_ms();

    if (do_stream) {
        /* Streaming: SSE per token, HTTP chunked transfer encoding */
        char hdr[256];
        snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Transfer-Encoding: chunked\r\n"
            
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n");
        send(sock, hdr, strlen(hdr), 0);

        int gen_count = 0;
        const char *stop_type = "none";
        const char *stopping_word = "";
        char *generated_stream = (char *)calloc(1, 1024);
        int generated_stream_cap = 1024;

        /* If the entire prompt is cached (start_pos >= n_prompt), step back
         * to position n_prompt - 1 so model_forward produces correct logits
         * for sampling the first generated token. The KV cache at that
         * position is already valid from the previous request. */
        int gen_start = start_pos;
        if (gen_start >= n_prompt) {
            gen_start = n_prompt - 1;
        }
        /* The token for position gen_start is ptokens[gen_start] (the token
         * stored at that position), not ptokens[gen_start-1]. For the normal
         * case (gen_start < n_prompt), the loop condition pos < n_prompt - 1
         * will advance token to ptokens[pos+1]. For the step-back case,
         * gen_start == n_prompt - 1, so we need token = ptokens[n_prompt-1]. */
        int token = ptokens[gen_start];

        for (int pos = gen_start; pos < n_prompt + n_predict; pos++) {
            if (!client_alive(sock)) {
                fprintf(stderr, "\n[server] /completion stream: client disconnected at pos %d\n", pos);
                break;
            }
            float *logits = model_forward(model, token, pos);

            if (pos < n_prompt - 1 || prompt_only) {
                token = ptokens[pos + 1];
                continue;
            }

            int next = sampler_sample(&sampler, logits, model->config.vocab_size);
            static char qwen_buf3[64];
            const char *piece;
            if (srv.use_qwen_tok) {
                qwen_tokenize_decode(&srv.qwen_enc, next, qwen_buf3, sizeof(qwen_buf3));
                piece = qwen_buf3;
            } else {
                piece = tokenizer_decode(tokenizer, token, next);
            }
            if (!piece) piece = "";

            /* Append to cumulative output for stop word matching */
            int gen_len = (int)strlen(generated_stream);
            int piece_len = (int)strlen(piece);
            if (gen_len + piece_len + 1 > generated_stream_cap) {
                generated_stream_cap = (gen_len + piece_len + 1) * 2;
                generated_stream = (char *)realloc(generated_stream, generated_stream_cap);
            }
            memcpy(generated_stream + gen_len, piece, (size_t)(piece_len + 1));

            /* Check stop words against cumulative output (not individual token) */
            int stopped = 0;
            for (int i = 0; i < n_stop_words && !stopped; i++) {
                if (strstr(generated_stream, stop_words[i]) != NULL) {
                    stopped = 1;
                    stop_type = "word";
                    stopping_word = stop_words[i];
                    /* Trim output to before the stop word */
                    char *found = strstr(generated_stream, stop_words[i]);
                    if (found) *found = '\0';
                }
            }

            /* Build streaming response: {content, tokens, stop} */
            cJSON *resp = cJSON_CreateObject();
            cJSON_AddStringToObject(resp, "content", piece);

            if (return_tokens || do_stream) {
                cJSON *tok_arr = cJSON_CreateArray();
                cJSON_AddNumberToObject(tok_arr, "", next);
                cJSON_AddItemToObject(resp, "tokens", tok_arr);
            }

            if (stopped || next == (int)tokenizer->eos_id) {
                if (!stopped) stop_type = "eos";
                cJSON_AddBoolToObject(resp, "stop", 1);
                sse_send(sock, cJSON_PrintUnformatted(resp), NULL);
                free(cJSON_PrintUnformatted(resp));
                cJSON_Delete(resp);
                gen_count++;
                printf("%s", piece);
                fflush(stdout);
                break;
            }

            cJSON_AddBoolToObject(resp, "stop", 0);
            if (sse_send(sock, cJSON_PrintUnformatted(resp), NULL) < 0) {
                free(cJSON_PrintUnformatted(resp));
                cJSON_Delete(resp);
                fprintf(stderr, "\n[server] /completion stream: send failed\n");
                break;
            }
            free(cJSON_PrintUnformatted(resp));
            cJSON_Delete(resp);

            gen_count++;
            printf("%s", piece);
            fflush(stdout);

            if (!ignore_eos && next == (int)tokenizer->eos_id) {
                stop_type = "eos";
                break;
            }
            token = next;
        }

        free(generated_stream);
        double t_end = get_time_ms();
        double t_prefill_ms = t_end - t_start; /* includes everything for streaming */
        double gen_ms = t_end - t_start;

        /* Final timing SSE */
        cJSON *final = cJSON_CreateObject();
        cJSON_AddBoolToObject(final, "stop", 1);
        cJSON_AddStringToObject(final, "stop_type", stop_type);
        cJSON_AddStringToObject(final, "stopping_word", stopping_word);
        cJSON_AddNumberToObject(final, "tokens_evaluated", n_prompt);
        cJSON *timings_c = cJSON_CreateObject();
        cJSON_AddNumberToObject(timings_c, "cache_n", start_pos);
        cJSON_AddNumberToObject(timings_c, "prompt_n", n_prompt - start_pos);
        cJSON_AddNumberToObject(timings_c, "prompt_ms", t_prefill_ms);
        cJSON_AddNumberToObject(timings_c, "prompt_per_token_ms", (n_prompt - start_pos) > 0 ? t_prefill_ms / (n_prompt - start_pos) : 0);
        cJSON_AddNumberToObject(timings_c, "prompt_per_second", (n_prompt - start_pos) > 0 ? (n_prompt - start_pos) / (t_prefill_ms / 1000.0) : 0);
        cJSON_AddNumberToObject(timings_c, "predicted_n", gen_count);
        cJSON_AddNumberToObject(timings_c, "predicted_ms", gen_ms);
        cJSON_AddNumberToObject(timings_c, "predicted_per_token_ms", gen_count > 0 ? gen_ms / gen_count : 0);
        cJSON_AddNumberToObject(timings_c, "predicted_per_second", gen_count > 0 ? gen_count / (gen_ms / 1000.0) : 0);
        cJSON_AddItemToObject(final, "timings", timings_c);
        sse_send(sock, cJSON_PrintUnformatted(final), NULL);
        free(cJSON_PrintUnformatted(final));
        cJSON_Delete(final);
        send_chunked(sock, "", 0);

        fprintf(stderr, "\n[server] /completion: prompt %d new+%d cached in %.0fms, gen %d in %.0fms (%.1f tok/s)\n",
                n_prompt - start_pos, start_pos, t_prefill_ms,
                gen_count, gen_ms, gen_count > 0 ? gen_count / (gen_ms / 1000.0) : 0);
    } else {
        /* Non-streaming: collect all tokens, build one big response */
        int gen_start_ns = start_pos;
        if (gen_start_ns >= n_prompt) {
            gen_start_ns = n_prompt - 1;
        }
        int token = ptokens[gen_start_ns];
        int gen_count = 0;
        const char *stop_type = "none";
        const char *stopping_word = "";
        char *generated = (char *)malloc(n_predict * 100 + 1);
        generated[0] = '\0';
        int *gen_token_ids = (int *)malloc((size_t)n_predict * sizeof(int));

        for (int pos = gen_start_ns; pos < n_prompt + n_predict; pos++) {
            if (!client_alive(sock)) {
                fprintf(stderr, "\n[server] /completion non-stream: client disconnected at pos %d\n", pos);
                break;
            }
            float *logits = model_forward(model, token, pos);

            if (pos < n_prompt - 1 || prompt_only) {
                token = ptokens[pos + 1];
                continue;
            }

            int next = sampler_sample(&sampler, logits, model->config.vocab_size);
            static char qwen_buf4[64];
            const char *piece;
            if (srv.use_qwen_tok) {
                qwen_tokenize_decode(&srv.qwen_enc, next, qwen_buf4, sizeof(qwen_buf4));
                piece = qwen_buf4;
            } else {
                piece = tokenizer_decode(tokenizer, token, next);
            }
            if (!piece) piece = "";

            strncat(generated, piece, n_predict * 100 - strlen(generated) - 1);
            gen_token_ids[gen_count] = next;
            gen_count++;
            printf("%s", piece);
            fflush(stdout);

            /* Check stop words */
            for (int i = 0; i < n_stop_words; i++) {
                if (strstr(generated, stop_words[i]) != NULL) {
                    stop_type = "word";
                    stopping_word = stop_words[i];
                    /* Trim to before the stop word */
                    char *found = strstr(generated, stop_words[i]);
                    if (found) *found = '\0';
                    gen_count--;
                    break;
                }
            }
            if (strcmp(stop_type, "word") == 0) break;

            if (!ignore_eos && next == (int)tokenizer->eos_id) {
                stop_type = "eos";
                break;
            }
            if (gen_count >= n_predict) { stop_type = "limit"; break; }
            token = next;
        }

        double t_end = get_time_ms();
        double gen_ms = t_end - t_start;

        /* Build response */
        cJSON *resp = cJSON_CreateObject();
        cJSON_AddStringToObject(resp, "content", generated);

        if (return_tokens) {
            cJSON *tok_arr = cJSON_CreateArray();
            for (int i = 0; i < gen_count; i++)
                cJSON_AddNumberToObject(tok_arr, "", gen_token_ids[i]);
            cJSON_AddItemToObject(resp, "tokens", tok_arr);
        }

        cJSON_AddStringToObject(resp, "stop_type", stop_type);
        cJSON_AddStringToObject(resp, "stopping_word", stopping_word);
        cJSON_AddNumberToObject(resp, "tokens_evaluated", n_prompt);
        cJSON_AddNumberToObject(resp, "tokens_predicted", gen_count);

        if (model_name) cJSON_AddStringToObject(resp, "model", model_name);
        else cJSON_AddStringToObject(resp, "model", "pico");

        cJSON_AddItemToObject(resp, "generation_settings", gen_settings);

        /* Timings */
        cJSON *timings = cJSON_CreateObject();
        cJSON_AddNumberToObject(timings, "cache_n", start_pos);
        cJSON_AddNumberToObject(timings, "prompt_n", n_prompt - start_pos);
        cJSON_AddNumberToObject(timings, "prompt_ms", gen_ms);
        cJSON_AddNumberToObject(timings, "prompt_per_token_ms", (n_prompt - start_pos) > 0 ? gen_ms / (n_prompt - start_pos) : 0);
        cJSON_AddNumberToObject(timings, "prompt_per_second", (n_prompt - start_pos) > 0 ? (n_prompt - start_pos) / (gen_ms / 1000.0) : 0);
        cJSON_AddNumberToObject(timings, "predicted_n", gen_count);
        cJSON_AddNumberToObject(timings, "predicted_ms", gen_ms);
        cJSON_AddNumberToObject(timings, "predicted_per_token_ms", gen_count > 0 ? gen_ms / gen_count : 0);
        cJSON_AddNumberToObject(timings, "predicted_per_second", gen_count > 0 ? gen_count / (gen_ms / 1000.0) : 0);
        cJSON_AddItemToObject(resp, "timings", timings);

        char *json = cJSON_PrintUnformatted(resp);
        http_send(sock, 200, "application/json", json);
        free(json);
        cJSON_Delete(resp);

        srv.kv_pos = n_prompt;
        fprintf(stderr, "\n[server] /completion: prompt %d new+%d cached in %.0fms, gen %d in %.0fms (%.1f tok/s)\n",
                n_prompt - start_pos, start_pos, gen_ms,
                gen_count, gen_ms, gen_count > 0 ? gen_count / (gen_ms / 1000.0) : 0);

        free(generated);
        free(gen_token_ids);
    }

    printf("\n");
    fflush(stdout);

    free(prompt_copy);
    free(token_prompt);
    free(ptokens);
    for (int i = 0; i < n_stop_words; i++) free(stop_words[i]);
}

/* ---- Request Router ---- */

static void handle_request(SOCKET sock) {
    char method[16], path[512];
    srv.sock_in_buf = sock;
    int body_len = http_parse_request(srv.recv_buf, sizeof(srv.recv_buf), method, sizeof(method), path, sizeof(path));
    if (body_len < 0) {
        return;
    }

    char *body_start = strstr(srv.recv_buf, "\r\n\r\n");
    if (!body_start) body_start = srv.recv_buf + body_len;
    else body_start += 4;

    if (strcmp(method, "GET") == 0) {
        if (strcmp(path, "/v1/models") == 0) {
            handle_list_models(sock, srv.model_path);
        } else if (strcmp(path, "/props") == 0 || strcmp(path, "/v1/props") == 0) {
            handle_props(sock);
        } else if (strcmp(path, "/") == 0 || strcmp(path, "/health") == 0) {
            http_send(sock, 200, "text/plain", "PicoLM server running\n");
        } else {
            http_send(sock, 404, "application/json", "{\"error\":{\"message\":\"Not found\"}}");
        }
    } else if (strcmp(method, "POST") == 0) {
        if (!srv.model_loaded) {
            http_send(sock, 503, "application/json", "{\"error\":{\"message\":\"Model not loaded\"}}");
            return;
        }
        if (strcmp(path, "/v1/chat/completions") == 0) {
            handle_completion(sock, body_start, 1);
        } else if (strcmp(path, "/v1/completions") == 0) {
            handle_completion(sock, body_start, 0);
        } else if (strcmp(path, "/completion") == 0) {
            handle_llama_completion(sock, body_start);
        } else {
            http_send(sock, 404, "application/json", "{\"error\":{\"message\":\"Not found\"}}");
        }
    } else {
        http_send(sock, 405, "application/json", "{\"error\":{\"message\":\"Method not allowed\"}}");
    }
}

/* ---- Server Main Loop ---- */

int server_main(int port, const char *host, const char *model_path, int num_threads, int do_prefault) {
    srv.port = port;
    strncpy(srv.host, host, sizeof(srv.host) - 1);
    srv.model_path = model_path;
    srv.running = 1;

    /* Load model once at startup */
    if (server_init(model_path, num_threads, do_prefault) != 0) {
        return -1;
    }

#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    /* Create listening socket */
    srv.listen_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (srv.listen_sock == SOCKET_INVALID) {
        fprintf(stderr, "[server] Failed to create socket: %s\n",
#ifdef _WIN32
                "WSAGetLastError()"
#else
                strerror(errno)
#endif
        );
        return -1;
    }

    /* Set socket options */
    int opt = 1;
#ifdef _WIN32
    setsockopt(srv.listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));
#else
    setsockopt(srv.listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    /* Bind */
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (strcmp(host, "0.0.0.0") == 0 || strcmp(host, "::") == 0)
        addr.sin_addr.s_addr = INADDR_ANY;
    else
        addr.sin_addr.s_addr = inet_addr(host);

    if (bind(srv.listen_sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[server] Failed to bind to %s:%d: %s\n",
                host, port,
#ifdef _WIN32
                "WSAGetLastError()"
#else
                strerror(errno)
#endif
        );
        closesocket(srv.listen_sock);
        return -1;
    }

    if (listen(srv.listen_sock, PICO_SERVER_BACKLOG) < 0) {
        fprintf(stderr, "[server] Failed to listen");
#ifdef _WIN32
        fprintf(stderr, " (WSA error %lu)\n", (unsigned long)WSAGetLastError());
#else
        fprintf(stderr, ": %s\n", strerror(errno));
#endif
        closesocket(srv.listen_sock);
        return -1;
    }

    fprintf(stderr, "[server] Listening on %s:%d (model: %s)\n", host, port, model_path);
    fprintf(stderr, "[server] Endpoints:\n");
    fprintf(stderr, "  POST /v1/chat/completions\n");
    fprintf(stderr, "  POST /v1/completions\n");
    fprintf(stderr, "  POST /completion       (llama.cpp-style)\n");
    fprintf(stderr, "  GET  /v1/models\n");
    fprintf(stderr, "  GET  /props            (server properties)\n");
    fprintf(stderr, "[server] Press Ctrl+C to stop\n");

    /* Signal handler for graceful shutdown */
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); /* ignore broken pipe from closed clients */
#endif

    /* Main accept loop - single-threaded, one connection at a time */
    while (srv.running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        SOCKET client = accept(srv.listen_sock, (struct sockaddr *)&client_addr, &client_len);
        if (client == SOCKET_INVALID) {
            if (srv.running) {
#ifdef _WIN32
                fprintf(stderr, "[server] accept failed (WSA error %lu)\n", (unsigned long)WSAGetLastError());
#else
                fprintf(stderr, "[server] accept failed: %s\n", strerror(errno));
#endif
            }
            continue;
        }

        fprintf(stderr, "[server] Connection from %s:%d\n",
                inet_ntoa(client_addr.sin_addr),
                ntohs(client_addr.sin_port));

        /* Disable Nagle's algorithm for lower latency streaming */
#ifdef _WIN32
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, (const char *)&opt, (int)sizeof(opt));
#else
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#endif

        handle_request(client);
        closesocket(client);
    }

    closesocket(srv.listen_sock);
#ifdef _WIN32
    WSACleanup();
#endif
    server_free();
    return 0;
}
