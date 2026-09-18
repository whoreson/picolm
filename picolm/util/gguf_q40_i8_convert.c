/* gguf_q40_i8_convert.c -- Convert Q4_0_8_8 to pre-dequantized int8 format.
 *
 * Reads a GGUF file with Q4_0_8_8 (type 33) tensors and converts them to
 * a new format: block_q4i_0x8 (custom type 34), where nibbles are
 * pre-dequantized to signed int8 and pre-arranged in dpbusd lane order.
 *
 * This eliminates the blend+permute and LUT dequant shuffle stages from
 * the runtime kernel, reducing shuffle instructions by ~48 per block pair.
 *
 * Output file uses GGUF type 34 (Q4I_0_8_8) for Q4_0_8_8 tensors.
 * All other tensors are copied unchanged.
 *
 * Block layout (272 bytes):
 *   d[8]     : 16 bytes (8 FP16 scales, unchanged)
 *   qs[256]  : 256 bytes (8 rows x 32 signed int8, dpbusd lane order)
 *
 * qs layout per block (8 chunks of 32 bytes):
 *   qs[  0.. 31] = even rows {0,1,4,5} vals 0-7    (chunk 0)
 *   qs[ 32.. 63] = even rows {0,1,4,5} vals 8-15   (chunk 1)
 *   qs[ 64.. 95] = even rows {0,1,4,5} vals 16-23  (chunk 2)
 *   qs[ 96..127] = even rows {0,1,4,5} vals 24-31  (chunk 3)
 *   qs[128..159] = odd  rows {2,3,6,7} vals 0-7    (chunk 0)
 *   qs[160..191] = odd  rows {2,3,6,7} vals 8-15   (chunk 1)
 *   qs[192..223] = odd  rows {2,3,6,7} vals 16-23  (chunk 2)
 *   qs[224..255] = odd  rows {2,3,6,7} vals 24-31  (chunk 3)
 *
 * Within each 32-byte chunk (4 rows x 8 values = 32 int8):
 *   [0..3]   = row_r0 vals 0..3  (int32 lane 0)
 *   [4..7]   = row_r0 vals 4..7  (int32 lane 1)
 *   [8..11]  = row_r1 vals 0..3  (int32 lane 2)
 *   [12..15] = row_r1 vals 4..7  (int32 lane 3)
 *   [16..19] = row_r2 vals 0..3  (int32 lane 4)
 *   [20..23] = row_r2 vals 4..7  (int32 lane 5)
 *   [24..27] = row_r3 vals 0..3  (int32 lane 6)
 *   [28..31] = row_r3 vals 4..7  (int32 lane 7)
 *
 * The nibble XOR with 0x88 is handled: each byte in the original qs
 * has its nibbles XOR'd with 0x88 (converting unsigned 0-15 to
 * sign-magnitude -8..7). The dequant step extracts each nibble and
 * applies the XOR + sign extension via the LUT:
 *   LUT[nibble ^ 0x88] = {-1,-2,-3,-4,-5,-6,-7,-8, 7,6,5,4,3,2,1,0}
 * which maps the XOR'd nibble to the correct signed int8 value.
 *
 * Usage: gguf_q40_i8_convert input.gguf [output.gguf]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static uint32_t r32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
static uint64_t r64(const uint8_t *p) {
    return (uint64_t)p[0] | ((uint64_t)p[1]<<8) | ((uint64_t)p[2]<<16) | ((uint64_t)p[3]<<24) |
           ((uint64_t)p[4]<<32) | ((uint64_t)p[5]<<40) | ((uint64_t)p[6]<<48) | ((uint64_t)p[7]<<56);
}

static void w32(uint8_t *p, uint32_t v) {
    p[0]=v&0xff; p[1]=(v>>8)&0xff; p[2]=(v>>16)&0xff; p[3]=(v>>24)&0xff;
}
static void w64(uint8_t *p, uint64_t v) {
    p[0]=v&0xff; p[1]=(v>>8)&0xff; p[2]=(v>>16)&0xff; p[3]=(v>>24)&0xff;
    p[4]=(v>>32)&0xff; p[5]=(v>>40)&0xff; p[6]=(v>>48)&0xff; p[7]=(v>>56)&0xff;
}

enum { G_Q4_0X8 = 33, G_Q4I_0X8 = 34 };
#define BS_Q4X8      144   /* block_q4_0x8: d[8](16) + qs[128](128) */
#define BS_Q4I_0X8   272   /* block_q4i_0x8: d[8](16) + qs[256](256) */
#define ROWS_PER_BLK 8
#define K_PER_BLK    32

#define MAX_TENSORS 8192

typedef struct {
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
    uint64_t size;        /* total byte size of this tensor's data in input file */
    uint64_t hdr_off_pos; /* position in header buffer where this tensor's offset is stored */
    uint64_t hdr_type_pos; /* position in header buffer where this tensor's type is stored */
    uint64_t orig_idx;    /* original index before sorting */
} ti_t;

static uint8_t *g_hdrbuf = NULL;
static size_t g_hdrbuf_cap = 0;
static uint8_t *hdr_ensure(size_t need) {
    if (need > g_hdrbuf_cap) {
        uint8_t *p = realloc(g_hdrbuf, need);
        if (!p) { fprintf(stderr, "OOM reading header (%zu bytes)\n", need); exit(1); }
        g_hdrbuf = p; g_hdrbuf_cap = need;
    }
    return g_hdrbuf;
}

static uint64_t skip_value(const uint8_t *d, uint64_t pos, uint32_t vt) {
    switch (vt) {
        case 0: case 1: case 7: return 1;
        case 2: case 3:         return 2;
        case 4: case 5: case 6: return 4;
        case 10:case 11:case 12: return 8;
        case 8: return 8 + r64(d + pos);
        case 9: {
            uint32_t at = r32(d + pos);
            uint64_t al = r64(d + pos + 4);
            uint64_t p = 12;
            size_t esz = 0;
            switch (at) {
                case 0: case 1: case 7: esz = 1; break;
                case 2: case 3:         esz = 2; break;
                case 4: case 5: case 6: esz = 4; break;
                case 10:case 11:case 12: esz = 8; break;
            }
            if (esz > 0) p += al * esz;
            else for (uint64_t i = 0; i < al; i++) {
                uint64_t sl = r64(d + pos + p); p += 8 + sl;
            }
            return p;
        }
        default: return 0;
    }
}

/* LUT for XOR'd nibble -> signed int8.
 * The nibble in the file is XOR'd with 0x88 per byte.
 * XOR with 0x88 flips bit 3 of each nibble, converting
 * unsigned [0..15] to sign-magnitude form where the LUT maps directly.
 *
 * If raw nibble = n (0..15), XOR'd = n ^ 0x8 (since we XOR per byte with 0x88,
 * each nibble's bit 3 is flipped):
 *   0^8=8, 1^8=9, ..., 7^8=15, 8^8=0, 9^8=1, ..., 15^8=7
 * LUT maps XOR'd value to signed int8:
 *   LUT[8]= -8, LUT[9]= -7, ..., LUT[15]= -1, LUT[0]= 0, ..., LUT[7]= 7
 * This is equivalent to: val = (raw_nibble - 8) for raw 0..15.
 */
static const int8_t DEQUANT_LUT[16] = {
    0,  1,  2,  3,  4,  5,  6,  7,  /* indices 0..7: XOR'd nibbles for raw 8..15 */
   -8, -7, -6, -5, -4, -3, -2, -1   /* indices 8..15: XOR'd nibbles for raw 0..7 */
};

/* Convert one block_q4_0x8 to block_q4i_0x8.
 *
 * Input: 144 bytes (d[8] + qs[128])
 * Output: 272 bytes (d[8] + qs[256])
 */
static void convert_block(const uint8_t *src, int8_t *dst_qs) {
    /* src layout: d[8] at src[0..15], qs[128] at src[16..143]
     * dst_qs: 256 bytes of pre-arranged int8 */

    const uint8_t *qs = src + 16;  /* skip d[8] */

    /* Original qs interleaving (from repack_q4_0_to_q4_0x8, blocklen=8):
     * i=0..7:  qs[i*8 .. i*8+7]   = row(i%8)'s first 8 bytes  (values 0-15)
     * i=8..15: qs[i*8 .. i*8+7]   = row(i%8)'s second 8 bytes (values 16-31)
     *
     * For row r, chunk k (0-3, each 8 values), value v (0-7 within chunk):
     *   half = k / 2  (0 = vals 0-15, 1 = vals 16-31)
     *   byte_in_half = (k % 2) * 4 + v / 2  (0-7)
     *   nibble = v & 1  (0=low, 1=high)
     *   byte = qs[half * 64 + row * 8 + byte_in_half]
     *   nibble_val = nibble ? (byte >> 4) : (byte & 0x0F)
     *   val = DEQUANT_LUT[nibble_val]
     */

    /* Target layout: 8 chunks of 32 bytes each
     * even chunks (0-3): rows {0,1,4,5} vals per chunk
     * odd chunks (4-7):  rows {2,3,6,7} vals per chunk
     *
     * Within each chunk, the 4 rows' 8 values are stored in dpbusd lane order
     * matching the ORIGINAL kernel's post-blend+permute+dequant layout:
     *
     * The 256-bit register has 8 int32 lanes:
     *   Lane 0: row_r0 vals 0-3 (4 int8)
     *   Lane 1: row_r1 vals 0-3
     *   Lane 2: row_r2 vals 0-3
     *   Lane 3: row_r3 vals 0-3
     *   Lane 4: row_r0 vals 4-7
     *   Lane 5: row_r1 vals 4-7
     *   Lane 6: row_r2 vals 4-7
     *   Lane 7: row_r3 vals 4-7
     *
     * = 8 x 4 = 32 bytes per chunk
     *
     * NOTE: This is NOT contiguous per row. The original kernel interleaves
     * by value range first (vals 0-3 for all 4 rows), then by value range
     * 4-7 for all 4 rows. This matches how the blend+permute step produces
     * the register layout.
     */

    /* Even row indices within block: 0, 1, 4, 5 */
    /* Odd row indices within block: 2, 3, 6, 7 */
    const int even_rows[4] = {0, 1, 4, 5};
    const int odd_rows[4] = {2, 3, 6, 7};

    /* Zero out destination (safety) */
    memset(dst_qs, 0, 256);

    /* Target layout (must match sgemm_q4i_0x8.c kernel loading):
     * The kernel loads 4 x 32-byte chunks for the even group (qs[0..127]) and
     * 4 x 32-byte chunks for the odd group (qs[128..255]).
     *
     * After merge to 512-bit (bp0 lanes 0-7, bp1 lanes 8-15) and shuffle(136)/
     * shuffle(221), the dpbusd pipeline expects:
     *   re0 (qs[0..31]):   rows {0,1,4,5} vals {0,2,4,6,8,10,12,14} (even vals 0-15)
     *   re1 (qs[32..63]):  rows {0,1,4,5} vals {16,18,20,22,24,26,28,30}
     *   re2 (qs[64..95]):  rows {0,1,4,5} vals {1,3,5,7,9,11,13,15} (odd vals 0-15)
     *   re3 (qs[96..127]): rows {0,1,4,5} vals {17,19,21,23,25,27,29,31}
     *
     * Within each 32-byte chunk, the 4 rows are stored contiguously:
     *   [ri*8 .. ri*8+7] = row even_rows[ri]'s 8 values for this chunk
     *
     * Same pattern for odd rows {2,3,6,7} at qs[128..255].
     */

    /* Value index sets for each chunk */
    static const int chunk_vals[4][8] = {
        {0, 2, 4, 6, 8, 10, 12, 14},       /* chunk 0: even vals 0-15 */
        {16, 18, 20, 22, 24, 26, 28, 30},   /* chunk 1: even vals 16-31 */
        {1, 3, 5, 7, 9, 11, 13, 15},        /* chunk 2: odd vals 0-15 */
        {17, 19, 21, 23, 25, 27, 29, 31}    /* chunk 3: odd vals 16-31 */
    };

    for (int k = 0; k < 4; k++) {
        int dst_even = k * 32;
        int dst_odd = 128 + k * 32;

        /* Even rows */
        for (int ri = 0; ri < 4; ri++) {
            int row = even_rows[ri];
            int base = dst_even + ri * 8;
            for (int v = 0; v < 8; v++) {
                int val_idx = k * 8 + v;  /* sequential: chunk k = vals k*8..k*8+7 */
                /* Q4_0 byte b holds value b (low nibble) and value b+16 (high nibble).
                 * For value val_idx: byte_index = val_idx % 16, is_high = val_idx / 16. */
                int byte_index = val_idx % 16;   /* Q4_0 byte 0-15 */
                int is_high = val_idx / 16;      /* 0 = low nibble (vals 0-15), 1 = high (vals 16-31) */
                int qs_off;
                if (byte_index < 8)
                    qs_off = row * 8 + byte_index;            /* first half: bytes 0-7 */
                else
                    qs_off = 64 + row * 8 + (byte_index - 8); /* second half: bytes 8-15 */
                uint8_t raw = qs[qs_off];
                uint8_t nibble_val = is_high ? (raw >> 4) : (raw & 0x0F);
                dst_qs[base + v] = DEQUANT_LUT[nibble_val];
            }
        }

        /* Odd rows */
        for (int ri = 0; ri < 4; ri++) {
            int row = odd_rows[ri];
            int base = dst_odd + ri * 8;
            for (int v = 0; v < 8; v++) {
                int val_idx = k * 8 + v;  /* sequential */
                int byte_index = val_idx % 16;
                int is_high = val_idx / 16;
                int qs_off;
                if (byte_index < 8)
                    qs_off = row * 8 + byte_index;
                else
                    qs_off = 64 + row * 8 + (byte_index - 8);
                uint8_t raw = qs[qs_off];
                uint8_t nibble_val = is_high ? (raw >> 4) : (raw & 0x0F);
                dst_qs[base + v] = DEQUANT_LUT[nibble_val];
            }
        }
    }
}

#define COPY_CHUNK (4 * 1024 * 1024)
static int copy_passthrough(int ifd, int ofd, uint64_t n) {
    static uint8_t *buf = NULL;
    if (!buf) buf = malloc(COPY_CHUNK);
    if (!buf) { fprintf(stderr, "OOM\n"); return -1; }
    while (n > 0) {
        size_t want = n < COPY_CHUNK ? (size_t)n : COPY_CHUNK;
        ssize_t got = read(ifd, buf, want);
        if (got <= 0) { perror("read"); return -1; }
        ssize_t off = 0;
        while (off < got) {
            ssize_t w = write(ofd, buf + off, got - off);
            if (w <= 0) { perror("write"); return -1; }
            off += w;
        }
        n -= (uint64_t)got;
    }
    return 0;
}

#define BATCH_STRUCTS 2048
static int convert_q4x8_tensor(int ifd, int ofd, uint64_t nrows, uint64_t n) {
    if (nrows % ROWS_PER_BLK != 0 || n % K_PER_BLK != 0) {
        fprintf(stderr, "  WARNING: dims not block-aligned, copying unchanged\n");
        return copy_passthrough(ifd, ofd, nrows / ROWS_PER_BLK * (n / K_PER_BLK) * BS_Q4X8);
    }
    uint64_t nb = n / K_PER_BLK;
    uint64_t total_blocks = (nrows / ROWS_PER_BLK) * nb;

    uint8_t *inbuf = malloc((size_t)BATCH_STRUCTS * BS_Q4X8);
    int8_t *out_qs = malloc((size_t)BATCH_STRUCTS * (BS_Q4I_0X8 - 16));  /* qs only */
    uint8_t *outbuf = malloc((size_t)BATCH_STRUCTS * BS_Q4I_0X8);
    if (!inbuf || !out_qs || !outbuf) {
        fprintf(stderr, "OOM\n");
        free(inbuf); free(out_qs); free(outbuf);
        return -1;
    }

    uint64_t done = 0;
    while (done < total_blocks) {
        uint64_t batch = total_blocks - done;
        if (batch > BATCH_STRUCTS) batch = BATCH_STRUCTS;
        size_t in_bytes = (size_t)batch * BS_Q4X8;
        size_t out_bytes = (size_t)batch * BS_Q4I_0X8;

        size_t got = 0;
        while (got < in_bytes) {
            ssize_t r = read(ifd, inbuf + got, in_bytes - got);
            if (r <= 0) { perror("read"); goto fail; }
            got += (size_t)r;
        }

        for (uint64_t i = 0; i < batch; i++) {
            const uint8_t *s = inbuf + i * BS_Q4X8;
            uint8_t *d = outbuf + i * BS_Q4I_0X8;
            memcpy(d, s, 16);  /* d[8] scales unchanged */
            convert_block(s, (int8_t*)(d + 16));  /* qs converted */
        }

        size_t off = 0;
        while (off < out_bytes) {
            ssize_t w = write(ofd, outbuf + off, out_bytes - off);
            if (w <= 0) { perror("write"); goto fail; }
            off += w;
        }
        done += batch;
    }
    free(inbuf); free(out_qs); free(outbuf);
    return 0;
fail:
    free(inbuf); free(out_qs); free(outbuf);
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s input.gguf [output.gguf]\n", argv[0]);
        return 1;
    }
    const char *inpath = argv[1];
    char outpath[4096];
    if (argc == 3) {
        strncpy(outpath, argv[2], sizeof(outpath) - 1);
        outpath[sizeof(outpath) - 1] = '\0';
    } else {
        size_t ilen = strlen(inpath);
        const char *dot = strrchr(inpath, '.');
        if (dot && strcmp(dot, ".gguf") == 0)
            snprintf(outpath, sizeof(outpath), "%.*s.q4i.gguf", (int)(dot - inpath), inpath);
        else
            snprintf(outpath, sizeof(outpath), "%s.q4i.gguf", inpath);
        (void)ilen;
    }
    fprintf(stderr, "Input:  %s\nOutput: %s\n", inpath, outpath);

    int ifd = open(inpath, O_RDONLY);
    if (ifd < 0) { perror("open input"); return 1; }

    uint8_t magic_hdr[24];
    if (read(ifd, magic_hdr, 24) != 24) { fprintf(stderr, "short read\n"); return 1; }
    if (memcmp(magic_hdr, "GGUF", 4) != 0) { fprintf(stderr, "not GGUF\n"); return 1; }
    uint32_t version = r32(magic_hdr + 4);
    uint64_t n_tensors = r64(magic_hdr + 8);
    uint64_t n_metadata = r64(magic_hdr + 16);
    fprintf(stderr, "GGUF v%u: %lu tensors, %lu metadata\n",
            version, (unsigned long)n_tensors, (unsigned long)n_metadata);
    if (n_tensors > MAX_TENSORS) { fprintf(stderr, "too many tensors\n"); return 1; }

    /* Read header incrementally, growing buffer as needed.
     * Tokenizer arrays (tokens, merges) can push the header past 1MB.
     * Use inline reading (not skip_value) for metadata to avoid needing
     * the entire value in the buffer before knowing its size. */
    size_t hcap = 1 << 20;
    uint8_t *hdr = hdr_ensure(hcap);
    memcpy(hdr, magic_hdr, 24);
    size_t hlen = 24;

    /* Helper: ensure buffer is large enough AND data is read up to 'need' */
    #define HDR_FILL(need) do { \
        hdr = hdr_ensure((need) + hcap / 4); hcap = g_hdrbuf_cap; \
        while ((need) > hlen) { \
            ssize_t _got = read(ifd, hdr + hlen, hcap - hlen); \
            if (_got <= 0) { fprintf(stderr, "short read header\n"); return 1; } \
            hlen += (size_t)_got; \
        } \
    } while(0)

    uint64_t pos = 24;
    uint64_t alignment = 32;
    for (uint64_t mi = 0; mi < n_metadata; mi++) {
        HDR_FILL(pos + 8);
        uint64_t klen = r64(hdr + pos);
        uint64_t key_start = pos + 8;
        HDR_FILL(pos + 8 + klen + 4);
        pos += 8 + klen;
        uint32_t vt = r32(hdr + pos);
        pos += 4;
        if (klen == 22 && pos <= hlen && memcmp(hdr + key_start, "tensor_data.alignment", 22) == 0) {
            if (vt == 10) alignment = r64(hdr + pos);
            else if (vt == 4) alignment = (uint64_t)r32(hdr + pos);
        }
        /* Inline skip_value with incremental buffer growth */
        switch (vt) {
            case 0: case 1: case 7: pos += 1; break;
            case 2: case 3:         pos += 2; break;
            case 4: case 5: case 6: pos += 4; break;
            case 10:case 11:case 12: pos += 8; break;
            case 8: { /* String */
                HDR_FILL(pos + 8);
                uint64_t sl = r64(hdr + pos);
                pos += 8 + sl;
                break;
            }
            case 9: { /* Array */
                HDR_FILL(pos + 12);
                uint32_t at = r32(hdr + pos);
                uint64_t al = r64(hdr + pos + 4);
                uint64_t val_start = pos + 12;
                pos = val_start;
                if (at == 0 || at == 1 || at == 7) {
                    pos += al; /* 1-byte elements */
                } else if (at == 2 || at == 3) {
                    pos += al * 2; /* 2-byte elements */
                } else if (at == 4 || at == 5 || at == 6) {
                    pos += al * 4; /* 4-byte elements */
                } else if (at >= 10 && at <= 12) {
                    pos += al * 8; /* 8-byte elements */
                } else {
                    /* String array: iterate */
                    for (uint64_t si = 0; si < al; si++) {
                        HDR_FILL(pos + 8);
                        uint64_t sl = r64(hdr + pos);
                        pos += 8 + sl;
                    }
                }
                break;
            }
            default: break; /* Unknown type, skip 0 bytes */
        }
    }
    #undef HDR_FILL

    ti_t *tis = calloc(n_tensors, sizeof(ti_t));
    if (!tis) { fprintf(stderr, "OOM\n"); return 1; }
    for (uint64_t ti = 0; ti < n_tensors; ti++) {
        uint64_t nlen = r64(hdr + pos); pos += 8 + nlen;
        tis[ti].n_dims = r32(hdr + pos); pos += 4;
        for (uint32_t d = 0; d < tis[ti].n_dims; d++) { tis[ti].dims[d] = r64(hdr + pos); pos += 8; }
        tis[ti].hdr_type_pos = pos;  /* type position in header */
        tis[ti].type = r32(hdr + pos); pos += 4;
        tis[ti].hdr_off_pos = pos;   /* offset position in header */
        tis[ti].offset = r64(hdr + pos);
        tis[ti].orig_idx = ti;
        pos += 8;
    }

    uint64_t tbase = (pos + alignment - 1) & ~(alignment - 1);
    fprintf(stderr, "Header: %lu bytes, alignment=%lu, data base=%lu\n",
            (unsigned long)pos, (unsigned long)alignment, (unsigned long)tbase);

    /* Update types in header buffer (before sorting, using original order) */
    for (uint64_t ti = 0; ti < n_tensors; ti++) {
        if (tis[ti].type == G_Q4_0X8) {
            w32(hdr + tis[ti].hdr_type_pos, G_Q4I_0X8);
        }
    }

    /* Sort tensors by offset for sequential processing */
    for (uint64_t a = 0; a < n_tensors; a++)
        for (uint64_t b = a + 1; b < n_tensors; b++)
            if (tis[b].offset < tis[a].offset) { ti_t t = tis[a]; tis[a] = tis[b]; tis[b] = t; }

    /* Compute tensor sizes.
     * For converted types: compute from dims.
     * For passthrough types: use gguf_type_row_size equivalent to compute actual data size. */
    for (uint64_t ti = 0; ti < n_tensors; ti++) {
        uint64_t nrows = 1;
        for (uint32_t d = 0; d + 1 < tis[ti].n_dims; d++) nrows *= tis[ti].dims[d];
        uint64_t n = tis[ti].n_dims > 0 ? tis[ti].dims[tis[ti].n_dims - 1] : 0;
        if (tis[ti].type == G_Q4_0X8 && tis[ti].n_dims >= 1) {
            tis[ti].size = (nrows / ROWS_PER_BLK) * (n / K_PER_BLK) * BS_Q4X8;
        } else {
            /* Compute actual data size from GGUF type info */
            int bs = 0, qs = 0;
            switch (tis[ti].type) {
                case 0: bs=1; qs=4; break;   /* F32 */
                case 1: bs=1; qs=2; break;   /* F16 */
                case 2: bs=32; qs=18; break; /* Q4_0 */
                case 6: bs=32; qs=34; break; /* Q8_0 */
                case 14: bs=256; qs=210; break; /* Q6_K */
                default:
                    /* Fallback: use gap to next tensor */
                    if (ti + 1 < n_tensors) {
                        uint64_t gap = tis[ti+1].offset - tis[ti].offset;
                        tis[ti].size = gap - (gap % alignment);
                    } else {
                        struct stat st;
                        if (fstat(ifd, &st) < 0) { perror("fstat"); return 1; }
                        tis[ti].size = (uint64_t)st.st_size - tbase - tis[ti].offset;
                    }
                    continue;
            }
            if (bs > 0 && qs > 0) {
                tis[ti].size = (uint64_t)qs * (uint64_t)n / (uint64_t)bs * nrows;
            }
        }
    }

    int ofd = open(outpath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0) { perror("open output"); return 1; }

    /* Write header + padding to tbase */
    if ((size_t)write(ofd, hdr, (size_t)tbase) != (size_t)tbase) { perror("write header"); return 1; }

    if (lseek(ifd, (off_t)tbase, SEEK_SET) != (off_t)tbase) { perror("lseek"); return 1; }

    int n_converted = 0, n_passthrough = 0;
    uint64_t out_pos = tbase;  /* output file position, aligned */

    for (uint64_t ti = 0; ti < n_tensors; ti++) {
        /* Seek input to this tensor's data */
        uint64_t abs_in = tbase + tis[ti].offset;
        if (lseek(ifd, (off_t)abs_in, SEEK_SET) < 0) { perror("lseek in"); return 1; }

        uint64_t nrows = 1;
        for (uint32_t d = 0; d + 1 < tis[ti].n_dims; d++) nrows *= tis[ti].dims[d];
        uint64_t n = tis[ti].n_dims > 0 ? tis[ti].dims[tis[ti].n_dims - 1] : 0;

        if (tis[ti].type == G_Q4_0X8 && tis[ti].n_dims >= 1) {
            fprintf(stderr, "  tensor %lu: Q4_0_8_8 %lu x %lu -> Q4I_0_8_8 (converting)\n",
                    (unsigned long)ti, (unsigned long)nrows, (unsigned long)n);
            if (convert_q4x8_tensor(ifd, ofd, nrows, n) != 0) return 1;
            /* Compute output size with alignment */
            uint64_t sz = (nrows / ROWS_PER_BLK) * (n / K_PER_BLK) * BS_Q4I_0X8;
            uint64_t padded = (sz + alignment - 1) & ~(alignment - 1);
            /* Write padding if needed */
            if (padded > sz) {
                uint8_t z = 0;
                for (uint64_t pp = sz; pp < padded; pp++)
                    if (write(ofd, &z, 1) != 1) return 1;
            }
            /* Update header offset */
            w64(hdr + tis[ti].hdr_off_pos, out_pos - tbase);
            out_pos += padded;
            n_converted++;
        } else {
            /* Copy passthrough with alignment */
            uint64_t sz = tis[ti].size;
            if (copy_passthrough(ifd, ofd, sz) != 0) return 1;
            uint64_t padded = (sz + alignment - 1) & ~(alignment - 1);
            if (padded > sz) {
                uint8_t z = 0;
                for (uint64_t pp = sz; pp < padded; pp++)
                    if (write(ofd, &z, 1) != 1) return 1;
            }
            w64(hdr + tis[ti].hdr_off_pos, out_pos - tbase);
            out_pos += padded;
            n_passthrough++;
        }
    }

    /* Rewrite header with all updated offsets */
    if (lseek(ofd, 0, SEEK_SET) < 0) { perror("lseek"); return 1; }
    if ((size_t)write(ofd, hdr, (size_t)tbase) != (size_t)tbase) { perror("rewrite header"); return 1; }
    if (ftruncate(ofd, (off_t)out_pos) < 0) { perror("ftruncate"); return 1; }

    /* Verify */
    { uint64_t v0 = r64(hdr + tis[0].hdr_off_pos);
      uint64_t vn = r64(hdr + tis[n_tensors-1].hdr_off_pos);
      fprintf(stderr, "Verify: first_off=%lu, last_off=%lu, out_pos=%lu\n",
              (unsigned long)v0, (unsigned long)vn, (unsigned long)out_pos); }

    fprintf(stderr, "Output file size: %lu bytes\n", (unsigned long)out_pos);
    fprintf(stderr, "Done: %d converted, %d copied.\n", n_converted, n_passthrough);
    fprintf(stderr, "Load with PICOLM_Q4I_0_8_8=1\n");

    close(ifd); close(ofd); free(tis); free(g_hdrbuf);
    return 0;
}
