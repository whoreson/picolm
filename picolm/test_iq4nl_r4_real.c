#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "quant.h"

extern void fp16_table_init(void);
extern void dequantize_row_iq4_nl_r4_single(const block_iq4_nl_r4 *x, float *dst, int n, int row);
extern void vec_dot_iq4_nl_r4_q8_0_batch4(const void *vx, const void *wy, int n, float *out);

typedef struct { const uint8_t *data; uint64_t pos; } reader_t;
static inline uint32_t read_u32(reader_t *r) { uint32_t v; memcpy(&v, r->data + r->pos, 4); r->pos += 4; return v; }
static inline uint64_t read_u64(reader_t *r) { uint64_t v; memcpy(&v, r->data + r->pos, 8); r->pos += 8; return v; }
static uint64_t skip_value(reader_t *r, uint32_t vtype) {
    uint64_t start = r->pos;
    switch (vtype) {
        case 0: /* GGUF_META_UINT8 */ r->pos += 1; break;
        case 1: /* GGUF_META_INT8 */ r->pos += 1; break;
        case 2: /* GGUF_META_UINT16 */ r->pos += 2; break;
        case 3: /* GGUF_META_INT16 */ r->pos += 2; break;
        case 4: /* GGUF_META_UINT32 */ r->pos += 4; break;
        case 5: /* GGUF_META_INT32 */ r->pos += 4; break;
        case 6: /* GGUF_META_FLOAT32 */ r->pos += 4; break;
        case 7: /* GGUF_META_BOOL */ r->pos += 1; break;
        case 8: /* GGUF_META_STRING */ { uint64_t slen = read_u64(r); r->pos += slen; break; }
        case 9: /* GGUF_META_ARRAY */ {
            uint32_t et = read_u32(r);
            uint64_t elen = read_u64(r);
            for (uint64_t i = 0; i < elen; i++) skip_value(r, et);
            break;
        }
        case 10: /* GGUF_META_UINT64 */ r->pos += 8; break;
        case 11: /* GGUF_META_INT64 */ r->pos += 8; break;
        case 12: /* GGUF_META_FLOAT64 */ r->pos += 8; break;
        default: break;
    }
    return r->pos - start;
}

int main(int argc, char **argv) {
    fp16_table_init();
    if (argc < 2) { fprintf(stderr, "Usage: %s <gguf>\n", argv[0]); return 1; }

    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    void *map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) return 1;

    reader_t r = { .data = (const uint8_t *)map + 4 }; /* skip magic */
    uint32_t version = read_u32(&r);
    uint64_t n_tensors = read_u64(&r);
    uint64_t n_kv = read_u64(&r);
    fprintf(stderr, "v=%u tensors=%lu kv=%lu pos=%lu\n", version, n_tensors, n_kv, r.pos);

    /* Skip KV entries: nk (8), key (nk), vtype (4), value (depends on vtype) */
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t nk = read_u64(&r);
        r.pos += nk; /* skip key */
        uint32_t vtype = read_u32(&r);
        uint64_t vs = skip_value(&r, vtype);
        r.pos += (8 - vs % 8) % 8; /* pad to 8 */
        if (i < 3) fprintf(stderr, "KV %lu: nk=%lu vtype=%u vs=%lu pos=%lu\n", i, nk, vtype, vs, r.pos);
    }
    fprintf(stderr, "After KV pos=%lu\n", r.pos);

    /* Find first IQ4_NL_R4 tensor */
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nk = read_u64(&r);
        uint64_t nd = read_u64(&r);
        const char *name = (const char *)(r.data + r.pos); r.pos += nk;
        r.pos += (8 - nk % 8) % 8; /* key name is padded to 8 */
        uint64_t dims[4] = {0};
        for (uint64_t j = 0; j < nd; j++) dims[j] = read_u64(&r);
        uint32_t dtype = read_u32(&r);
        uint64_t offset = read_u64(&r);

        if (dtype == 220) {
            printf("Tensor: %.*s dims=%lu,%lu offset=%lu\n", (int)nk, name, dims[0], dims[1], offset);
            const block_iq4_nl_r4 *block = (const block_iq4_nl_r4 *)((const uint8_t *)map + offset);
            printf("  d={0x%04x,0x%04x,0x%04x,0x%04x}\n", block->d[0],block->d[1],block->d[2],block->d[3]);
            printf("  qs[0:16]=");
            for (int j = 0; j < 16; j++) printf(" %02x", block->qs[j]);
            printf("\n");

            int n = 32;
            float deq[4][32];
            for (int row = 0; row < 4; row++) dequantize_row_iq4_nl_r4_single(block, deq[row], n, row);

            block_q8_0 yb; yb.d = 0x3C00;
            for (int j = 0; j < 32; j++) yb.qs[j] = (j % 2 == 0) ? 1 : -1;

            float scalar_out[4];
            for (int row = 0; row < 4; row++) {
                float s = 0;
                for (int j = 0; j < 32; j++) s += deq[row][j] * yb.qs[j];
                scalar_out[row] = s;
            }

            float avx2_out[4];
            vec_dot_iq4_nl_r4_q8_0_batch4(block, &yb, n, avx2_out);

            for (int row = 0; row < 4; row++) {
                float diff = scalar_out[row] - avx2_out[row];
                if (diff < 0) diff = -diff;
                printf("Row %d: scalar=%.6f avx2=%.6f diff=%.6f %s\n",
                       row, scalar_out[row], avx2_out[row], scalar_out[row] - avx2_out[row],
                       diff > 0.01f ? "MISMATCH" : "OK");
            }
            break;
        }
    }
    munmap(map, st.st_size); close(fd);
    return 0;
}
