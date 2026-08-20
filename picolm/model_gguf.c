#include "model.h"
#include "tensor.h"
#include "quant.h"
#include "model_internal.h"
#include <inttypes.h>
#include <stdio.h>
/* SSM verification debug (used in GGUF parsing) */
#ifdef PICOLM_SSM_VERIFY
#define _SSM_DBG _SSM_DBG
#else
#define _SSM_DBG (0)
#endif
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <unistd.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#endif

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
    /* PROT_READ only: PROT_WRITE causes COW fault storms over CIFS/NFS on
     * large models (see 9c1b3a5). PPC big-endian needs write access for the
     * in-place byte-swap loop; it uses mprotect() around the swap region. */
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

int parse_gguf(model_t *m, int max_seq_len) {
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
                    m->tok_unknown_model = 1; /* clear when pre=smollm etc is seen */
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
                    m->tok_unknown_model = 0; /* smollm pre-tokenizer works with SPM */
                } else if (pre.len >= 6 && strncmp(pre.str, "qwen35", 6) == 0) {
                    m->tok_space_marker = 3; /* qwen35: U+0100, no prefix on first token */
                    m->tok_unknown_model = 0;
                } else if (pre.len >= 10 && strncmp(pre.str, "llama-bpe", 9) == 0) {
                    /* llama-bpe: GPT-2 BPE with Llama preprocessing - unsupported, use U+0100 as best guess */
                    m->tok_space_marker = 1;
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
                    lw->ffn_gate = ptr; lw->type_ffn_gate = qtype;
                } else if (strcmp(suffix, "ffn_down.weight") == 0) {
                    lw->ffn_down = ptr; lw->type_ffn_down = qtype;
                } else if (strcmp(suffix, "ffn_up.weight") == 0) {
                    lw->ffn_up = ptr; lw->type_ffn_up = qtype;
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
                } else if (strcmp(suffix, "per_layer_proj.weight") == 0
                        || strcmp(suffix, "proj.weight") == 0) {
                    lw->per_layer_proj = ptr; lw->type_per_layer_proj = qtype;
                } else if (strcmp(suffix, "per_layer_post_norm.weight") == 0
                        || strcmp(suffix, "post_norm.weight") == 0) {
                    lw->per_layer_post_norm = ptr; lw->type_per_layer_post_norm = qtype;
                } else if (strcmp(suffix, "altup_correct_coef.weight") == 0) {
                    lw->altup_correct_coef = ptr;
                } else if (strcmp(suffix, "altup_correct_scale.weight") == 0) {
                    lw->altup_correct_scale = ptr;
                } else if (strcmp(suffix, "altup_predict_coef.weight") == 0) {
                    lw->altup_predict_coef = ptr;
                } else if (strcmp(suffix, "altup_router.weight") == 0) {
                    lw->altup_router = ptr; lw->type_altup_router = qtype;
                } else if (strcmp(suffix, "altup_router_norm.weight") == 0) {
                    lw->altup_router_norm = ptr;
                } else if (strcmp(suffix, "laurel_l.weight") == 0) {
                    lw->laurel_l = ptr; lw->type_laurel_l = qtype;
                    if (layer == 0 && _SSM_DBG) fprintf(stderr, "DBG laurel_l type=%d\n", qtype);
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
                    if (layer == 0 && _SSM_DBG) fprintf(stderr, "DBG laurel_r type=%d\n", qtype);
                } else if (strcmp(suffix, "laurel_post_norm.weight") == 0) {
                    lw->laurel_post_norm = ptr;
                }
            } else if (cfg->is_gemma3n) {
                /* Gemma-3n global tensors (not under blk.) */
                if (str_eq(tinfos[i].name, "altup_proj.weight")) {
                    w->altup_proj = ptr; w->type_altup_proj = qtype;
                } else if (str_eq(tinfos[i].name, "altup_unembd_proj.weight")) {
                    w->altup_unembd_proj = ptr; w->type_altup_unembd_proj = qtype;
                } else if (str_eq(tinfos[i].name, "per_layer_token_embd.weight")) {
                    w->per_layer_tok_embd = ptr; w->type_per_layer_tok_embd = qtype;
                } else if (str_eq(tinfos[i].name, "per_layer_model_proj.weight")) {
                    w->per_layer_model_proj = ptr; w->type_per_layer_model_proj = qtype;
                } else if (str_eq(tinfos[i].name, "per_layer_proj_norm.weight")) {
                    w->per_layer_proj_norm = ptr;
                    if (_SSM_DBG) fprintf(stderr, "DBG per_layer_proj_norm type=%d\n", qtype);
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
     * Swap F16 values in-place for all quantized block types that contain FP16 scales.
     *
     * The mmap is PROT_READ (COW mappings over CIFS fail with PROT_WRITE on
     * low-RAM machines; see 9c1b3a5). We temporarily mprotect each split to
     * PROT_READ|PROT_WRITE for the swap loop, then restore PROT_READ. */
#if defined(__APPLE__) && defined(__ppc__)
    { int be_start = clock();
    fprintf(stderr, "Big-endian: swapping F16 values...\n");

    /* Temporarily make all split mmap regions writable for in-place byte swap.
     * We must do this because GGUF stores all multi-byte values in LE, and the
     * quantized block scales (FP16) need swapping. Original commit 6ddcd2d used
     * PROT_READ|PROT_WRITE from the start; 9c1b3a5 changed to PROT_READ for
     * CIFS compatibility, so we use mprotect around the swap. */
    for (int _si = 0; _si < m->n_splits; _si++) {
        if (m->splits[_si].mmap_addr && m->splits[_si].mmap_size > 0) {
            mprotect(m->splits[_si].mmap_addr, m->splits[_si].mmap_size,
                     PROT_READ | PROT_WRITE);
        }
    }

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

    /* Restore mmap regions to read-only */
    for (int _si = 0; _si < m->n_splits; _si++) {
        if (m->splits[_si].mmap_addr && m->splits[_si].mmap_size > 0) {
            mprotect(m->splits[_si].mmap_addr, m->splits[_si].mmap_size, PROT_READ);
        }
    }
}
#endif

    free(tinfos);
    return 0;
}
