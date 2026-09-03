/* gguf_be_swap.c -- Pre-process GGUF files for big-endian platforms (PPC).
 *
 * On big-endian systems, GGUF stores all multi-byte values as little-endian.
 * The runtime model_gguf.c does an in-place byte swap at load time (~5s for
 * small models). This tool does the swap as a one-time offline operation,
 * writing a .ffug file (GGUF backwards) that skips the runtime swap.
 *
 * Usage: gguf_be_swap input.gguf [output.ffug]
 *
 * Approach: copy entire file byte-for-byte, then mmap the output for
 * in-place F16 field swapping. Only the quant block scale fields (uint16_t)
 * need swapping -- F32 is handled at dequant time by quant.c.
 *
 * Types handled: F16, BF16, Q8_0, Q4_0, Q4_1, Q2_0, Q1_0,
 *   Q4_0_4_4, Q4_0_8_8, Q4_K, Q5_K, Q6_K, Q3_K, Q2_K, Q5_0, Q5_1
 *
 * Compile: cc -O2 -o gguf_be_swap gguf_be_swap.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

static uint16_t swap16(uint16_t v) { return (v >> 8) | (v << 8); }

static uint32_t r32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
static uint64_t r64(const uint8_t *p) {
    return (uint64_t)p[0] | ((uint64_t)p[1]<<8) | ((uint64_t)p[2]<<16) | ((uint64_t)p[3]<<24) |
           ((uint64_t)p[4]<<32) | ((uint64_t)p[5]<<40) | ((uint64_t)p[6]<<48) | ((uint64_t)p[7]<<56);
}

enum {
    G_F32=0, G_F16=1, G_Q4_0=2, G_Q4_1=3, G_Q8_0=8,
    G_Q2_K=10, G_Q3_K=11, G_Q4_K=12, G_Q5_K=13, G_Q6_K=14,
    G_BF16=15, G_Q5_0=6, G_Q5_1=7,
    G_Q4_0X4=31, G_Q4_0X8=33,
    G_Q1_0=41, G_Q2_0=42,
};

#define BS_Q8_0   34
#define BS_Q4_0   18
#define BS_Q4_1   20
#define BS_Q2_0   34
#define BS_Q1_0   18
#define BS_Q4K    144
#define BS_Q5K    176
#define BS_Q6K    210
#define BS_Q3K    110
#define BS_Q2K    84
#define BS_Q5_0   22
#define BS_Q5_1   24
#define BS_Q4X4   18
#define BS_Q4X8   144

#define MAX_TENSORS 8192

typedef struct {
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
} ti_t;

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
            if (esz > 0) { p += al * esz; }
            else {
                for (uint64_t i = 0; i < al; i++) {
                    uint64_t sl = r64(d + pos + p);
                    p += 8 + sl;
                }
            }
            return p;
        }
        default: return 0;
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s input.gguf [output.ffug]\n", argv[0]);
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
            snprintf(outpath, sizeof(outpath), "%.*s.ffug", (int)(dot - inpath), inpath);
        else
            snprintf(outpath, sizeof(outpath), "%s.ffug", inpath);
    }

    fprintf(stderr, "Input:  %s\nOutput: %s\n", inpath, outpath);

    int ifd = open(inpath, O_RDONLY);
    if (ifd < 0) { perror("open input"); return 1; }
    struct stat st;
    if (fstat(ifd, &st) < 0) { perror("fstat"); close(ifd); return 1; }
    size_t fsize = (size_t)st.st_size;

    const uint8_t *data = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, ifd, 0);
    if (data == MAP_FAILED) { perror("mmap"); close(ifd); return 1; }
    close(ifd);

    if (memcmp(data, "GGUF", 4) != 0) {
        fprintf(stderr, "ERROR: not a GGUF file\n"); return 1;
    }
    uint32_t version = r32(data + 4);
    uint64_t n_tensors = r64(data + 8);
    uint64_t n_metadata = r64(data + 16);

    fprintf(stderr, "GGUF v%u: %lu tensors, %lu metadata\n",
            version, (unsigned long)n_tensors, (unsigned long)n_metadata);

    if (n_tensors > MAX_TENSORS) {
        fprintf(stderr, "ERROR: too many tensors\n"); return 1;
    }

    /* Parse metadata */
    uint64_t pos = 24;
    uint64_t alignment = 32;
    for (uint64_t mi = 0; mi < n_metadata; mi++) {
        uint64_t klen = r64(data + pos);
        uint64_t key_start = pos + 8;
        pos += 8 + klen;
        uint32_t vt = r32(data + pos);
        pos += 4;
        if (klen == 22 && memcmp(data + key_start, "tensor_data.alignment", 22) == 0) {
            if (vt == 10) alignment = r64(data + pos);
            else if (vt == 4) alignment = (uint64_t)r32(data + pos);
        }
        pos += skip_value(data, pos, vt);
    }

    /* Parse tensor table */
    uint64_t ttable_start = pos;
    ti_t *tis = calloc(n_tensors, sizeof(ti_t));
    if (!tis) { fprintf(stderr, "OOM\n"); return 1; }

    for (uint64_t ti = 0; ti < n_tensors; ti++) {
        uint64_t nlen = r64(data + pos); pos += 8 + nlen;
        tis[ti].n_dims = r32(data + pos); pos += 4;
        for (uint32_t d = 0; d < tis[ti].n_dims; d++) {
            tis[ti].dims[d] = r64(data + pos); pos += 8;
        }
        tis[ti].type = r32(data + pos); pos += 4;
        tis[ti].offset = r64(data + pos); pos += 8;
    }

    uint64_t tbase = (pos + alignment - 1) & ~((uint64_t)alignment - 1);
    fprintf(stderr, "Tensor table: %lu-%lu, Data base: %lu\n",
            (unsigned long)ttable_start, (unsigned long)pos, (unsigned long)tbase);

    /* Copy entire file to output */
    int ofd = open(outpath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0) { perror("open output"); free(tis); return 1; }
    if (write(ofd, data, fsize) != (ssize_t)fsize) {
        perror("write"); free(tis); close(ofd); return 1; }
    close(ofd);
    munmap((void *)data, fsize);

    /* Mmap output for in-place F16 swapping */
    ofd = open(outpath, O_RDWR);
    if (ofd < 0) { perror("open output rdwr"); free(tis); return 1; }
    uint8_t *out = mmap(NULL, fsize, PROT_READ|PROT_WRITE, MAP_SHARED, ofd, 0);
    if (out == MAP_FAILED) { perror("mmap output"); close(ofd); free(tis); return 1; }

    int nswapped = 0, nskipped = 0;
    clock_t t0 = clock();

    for (uint64_t ti = 0; ti < n_tensors; ti++) {
        uint32_t qt = tis[ti].type;
        uint64_t toff = tis[ti].offset;
        /* nrows = product of all dims except last; n = last dim */
        uint64_t nrows = 1;
        for (uint32_t d = 0; d + 1 < tis[ti].n_dims; d++) nrows *= tis[ti].dims[d];
        uint64_t n = tis[ti].dims[tis[ti].n_dims - 1];

        uint8_t *ptr = out + tbase + toff;
        int needs_swap = 0;

        switch (qt) {
            case G_F32: case G_F16: case G_BF16: case G_Q8_0: case G_Q4_0: case G_Q4_1:
            case G_Q4_0X4: case G_Q4_0X8: case G_Q2_0: case G_Q1_0:
            case G_Q4_K: case G_Q5_K: case G_Q6_K: case G_Q3_K: case G_Q2_K:
            case G_Q5_0: case G_Q5_1:
                needs_swap = 1; break;
        }

        if (!needs_swap) { nskipped++; continue; }

        switch (qt) {
            case G_F32: {
                size_t total = (size_t)nrows * (size_t)n;
                size_t ii2;
                for (ii2 = 0; ii2 < total; ii2++) {
                    uint32_t v = ((uint32_t *)ptr)[ii2];
                    ((uint32_t *)ptr)[ii2] = (v >> 24) | ((v >> 8) & 0xff00) |
                                              ((v << 8) & 0xff0000) | (v << 24);
                }
                nswapped += (int)total;
                break;
            }
            case G_F16: case G_BF16: {
                for (size_t i = 0; i < nrows * n; i++)
                    ((uint16_t *)ptr)[i] = swap16(((uint16_t *)ptr)[i]);
                nswapped += (int)(nrows * n);
                break;
            }
            case G_Q8_0: {
                size_t nb = nrows * n / 32;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q8_0;
                    ((uint16_t *)bl)[0] = swap16(((uint16_t *)bl)[0]);
                }
                nswapped += (int)nb;
                break;
            }
            case G_Q4_0: {
                size_t nb = nrows * n / 32;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q4_0;
                    ((uint16_t *)bl)[0] = swap16(((uint16_t *)bl)[0]);
                }
                nswapped += (int)nb;
                break;
            }
            case G_Q4_1: {
                size_t nb = nrows * n / 32;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q4_1;
                    ((uint16_t *)bl)[0] = swap16(((uint16_t *)bl)[0]);
                }
                nswapped += (int)nb;
                break;
            }
            case G_Q4_0X4: {
                size_t nb = nrows * n / 32;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q4X4;
                    for (int i = 0; i < 2; i++)
                        ((uint16_t *)bl)[i] = swap16(((uint16_t *)bl)[i]);
                }
                nswapped += (int)(nb * 2);
                break;
            }
            case G_Q4_0X8: {
                size_t nb = nrows * n / 32;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q4X8;
                    for (int i = 0; i < 8; i++)
                        ((uint16_t *)bl)[i] = swap16(((uint16_t *)bl)[i]);
                }
                nswapped += (int)(nb * 8);
                break;
            }
            case G_Q2_0: {
                size_t nb = nrows * n / 128;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q2_0;
                    ((uint16_t *)bl)[0] = swap16(((uint16_t *)bl)[0]);
                }
                nswapped += (int)nb;
                break;
            }
            case G_Q1_0: {
                size_t nb = nrows * n / 128;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q1_0;
                    ((uint16_t *)bl)[0] = swap16(((uint16_t *)bl)[0]);
                }
                nswapped += (int)nb;
                break;
            }
            case G_Q4_K: {
                size_t nb = nrows * n / 256;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q4K;
                    ((uint16_t *)bl)[0] = swap16(((uint16_t *)bl)[0]);
                    ((uint16_t *)bl)[1] = swap16(((uint16_t *)bl)[1]);
                }
                nswapped += (int)(nb * 2);
                break;
            }
            case G_Q5_K: {
                size_t nb = nrows * n / 256;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q5K;
                    ((uint16_t *)bl)[0] = swap16(((uint16_t *)bl)[0]);
                    ((uint16_t *)bl)[1] = swap16(((uint16_t *)bl)[1]);
                }
                nswapped += (int)(nb * 2);
                break;
            }
            case G_Q6_K: {
                size_t nb = nrows * n / 256;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q6K;
                    ((uint16_t *)bl)[0] = swap16(((uint16_t *)bl)[0]);
                }
                nswapped += (int)nb;
                break;
            }
            case G_Q3_K: {
                size_t nb = nrows * n / 256;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q3K;
                    ((uint16_t *)(bl + 108))[0] = swap16(((uint16_t *)(bl + 108))[0]);
                }
                nswapped += (int)nb;
                break;
            }
            case G_Q2_K: {
                size_t nb = nrows * n / 256;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q2K;
                    ((uint16_t *)(bl + 80))[0] = swap16(((uint16_t *)(bl + 80))[0]);
                    ((uint16_t *)(bl + 82))[0] = swap16(((uint16_t *)(bl + 82))[0]);
                }
                nswapped += (int)(nb * 2);
                break;
            }
            case G_Q5_0: {
                size_t nb = nrows * n / 32;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q5_0;
                    ((uint16_t *)bl)[0] = swap16(((uint16_t *)bl)[0]);
                }
                nswapped += (int)nb;
                break;
            }
            case G_Q5_1: {
                size_t nb = nrows * n / 32;
                for (size_t b = 0; b < nb; b++) {
                    uint8_t *bl = ptr + b * BS_Q5_1;
                    ((uint16_t *)bl)[0] = swap16(((uint16_t *)bl)[0]);
                }
                nswapped += (int)nb;
                break;
            }
        }
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;
    fprintf(stderr, "Swapped %d F16 values, skipped %d tensors in %.0fms\n",
            nswapped, nskipped, elapsed);
    fprintf(stderr, "Done: %s\n", outpath);

    msync(out, fsize, MS_SYNC);
    munmap(out, fsize);
    close(ofd);
    free(tis);
    return 0;
}
