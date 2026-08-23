#include "model.h"
#include "tensor.h"
#include "quant.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef PICOLM_DOS
#include <alloca.h>
#endif

#include <errno.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
/* POSIX type polyfills for MSVC (MinGW/MSYS2 provides these natively) */
#if defined(_MSC_VER)
#ifndef ssize_t
typedef SSIZE_T ssize_t;
#endif
#ifndef off_t
typedef long long off_t;
#endif
#endif
#elif !defined(PICOLM_DOS)
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

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
    #ifdef _WIN32
    if (_lseeki64(fd, (off_t)next_offset, SEEK_SET) < 0) {
#else
    if (lseek(fd, (off_t)next_offset, SEEK_SET) < 0) {
#endif
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
        #ifdef _WIN32
        if (_lseeki64(fd, 0, SEEK_SET) < 0) { _kv_close(fd); return 0; }
#else
        if (lseek(fd, 0, SEEK_SET) < 0) { _kv_close(fd); return 0; }
#endif
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
                #ifdef _WIN32
                if (_lseeki64(fd, 0, SEEK_SET) < 0) { _kv_close(fd); return 0; }
#else
                if (lseek(fd, 0, SEEK_SET) < 0) { _kv_close(fd); return 0; }
#endif
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


