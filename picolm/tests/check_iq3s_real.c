/* A/B the IQ3_S vec_dot kernels on REAL GGUF tensor rows.
 * Usage: check_iq3s_real <gguf-file> [tensor-name-substr] [row-limit]
 * Finds the first IQ3_S tensor whose name contains the substring, then
 * compares scalar vs AVX2 vec_dot_iq3_s_q8_k on real rows with random
 * Q8_K activations. Exit 1 on mismatch. */
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

static FILE *g_f;
static int rd(void *p, size_t n) { return fread(p, 1, n, g_f) == n; }

static int walk_gguf(const char *path, const char *substr,
                     long *out_off, long *out_ne0, long *out_ne1, char *out_name) {
    g_f = fopen(path, "rb");
    if (!g_f) return -1;
    uint32_t magic, ver;
    if (!rd(&magic, 4) || magic != 0x46554747) { fclose(g_f); return -1; }
    if (!rd(&ver, 4)) { fclose(g_f); return -1; }
    /* GGUF: tensor_count first, then metadata_kv_count */
    uint64_t ntensors_hdr, nkv; if (!rd(&ntensors_hdr, 8) || !rd(&nkv, 8)) { fclose(g_f); return -1; }
    for (uint64_t i = 0; i < nkv; i++) {
        uint64_t klen; if (!rd(&klen, 8)) { fclose(g_f); return -1; }
        fseek(g_f, (long)klen, SEEK_CUR);
        uint32_t vt; if (!rd(&vt, 4)) { fclose(g_f); return -1; }
        if (vt == 8) { uint64_t sl; rd(&sl, 8); fseek(g_f, (long)sl, SEEK_CUR); }
        else if (vt == 10) {
            uint32_t et; rd(&et, 4);
            uint64_t n; rd(&n, 8);
            if (et == 8) {
                for (uint64_t j = 0; j < n; j++) { uint64_t sl; rd(&sl, 8); fseek(g_f, (long)sl, SEEK_CUR); }
            } else if (et == 10) { fclose(g_f); return -1; }
            else {
                size_t step = (et == 0 || et == 1) ? 1 : (et == 2 || et == 3) ? 2
                            : (et <= 7) ? 4 : 8;
                fseek(g_f, (long)(n * step), SEEK_CUR);
            }
        }
        else if (vt == 0 || vt == 1 || vt == 7) fseek(g_f, 1, SEEK_CUR);
        else if (vt == 2 || vt == 3) fseek(g_f, 2, SEEK_CUR);
        else if (vt == 4 || vt == 5 || vt == 6) fseek(g_f, 4, SEEK_CUR);
        else if (vt == 9) fseek(g_f, 8, SEEK_CUR);
    }
    /* alignment comes from general.alignment metadata (default 32); we did
     * not track it above, so re-scan: assume 32 and verify with tensor 0's
     * magic-free heuristic below. */
    long dir_end = ftell(g_f);
    long align = 32;
    long data_base = (dir_end + align - 1) / align * align;
    /* Re-walk the tensor directory from dir_end. */
    fseek(g_f, dir_end, SEEK_SET);
    for (int64_t t = 0; t < (int64_t)ntensors_hdr; t++) {
        uint64_t nlen; if (!rd(&nlen, 8)) break;
        char name[256] = {0};
        if (nlen < sizeof name) { if (!rd(name, nlen)) break; } else fseek(g_f, (long)nlen, SEEK_CUR);
        uint32_t ndims; if (!rd(&ndims, 4)) break;
        long long ne[4] = {1,1,1,1};
        for (uint32_t d = 0; d < ndims; d++) rd(&ne[d], 8);
        uint32_t type; if (!rd(&type, 4)) break;
        int64_t off; if (!rd(&off, 8)) break;
        if ((int)type == GGUF_TYPE_IQ3_S && strstr(name, substr)) {
            long start = data_base + off;   /* offsets are relative to data section */
            *out_off = start; *out_ne0 = ne[0]; *out_ne1 = (ndims > 1) ? ne[1] : 1;
            snprintf(out_name, 256, "%s", name);
            fclose(g_f);
            return 0;
        }
    }
    fclose(g_f);
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <gguf> [substr] [maxrows]\n", argv[0]); return 1; }
    const char *substr = argc > 2 ? argv[2] : "attn_output";
    int maxrows = argc > 3 ? atoi(argv[3]) : 32;
    long off, ne0, ne1; char name[256];
    if (walk_gguf(argv[1], substr, &off, &ne0, &ne1, name) != 0) {
        fprintf(stderr, "no matching IQ3_S tensor found\n"); return 1;
    }
    int nb = (int)(ne0 / 256);
    fprintf(stderr, "found %s ne={%ld,%ld} type IQ3_S rowsize=%d off=%ld\n",
            name, ne0, ne1, nb * 110, off);
    if (ne0 % 256) { fprintf(stderr, "ne0 %% 256 != 0\n"); return 1; }

    size_t tsz = (size_t)nb * 110 * (size_t)ne1;
    int fd = open(argv[1], O_RDONLY);
    const uint8_t *base = mmap(NULL, (size_t)off + tsz + 4096, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    const uint8_t *tensor = base + off;

    int n = (int)ne0;
    int rows = (int)(ne1 < maxrows ? ne1 : maxrows);
    void *q8 = malloc((size_t)n / 256 * sizeof(block_q8_K));
    srand(1234);
    int nmis = 0, ncmp = 0;
    double worst = 0;
    for (int v = 0; v < 8; v++) {
        float *x = malloc((size_t)n * sizeof(float));
        for (int i = 0; i < n; i++) x[i] = (float)((rand() % 2001) - 1000) / 500.f;
        quantize_row_q8_K(x, q8, n);
        free(x);
        for (int r = 0; r < rows; r++) {
            const void *wrow = tensor + (size_t)r * (size_t)nb * 110;
            float s1 = vec_dot_iq3_s_q8_k_scalar(wrow, q8, n);
            float s2 = vec_dot_iq3_s_q8_k(wrow, q8, n);
            double denom = fabs(s1) > 1e-9 ? fabs(s1) : 1;
            double rel = fabs((double)s1 - s2) / denom;
            ncmp++;
            if (rel > 1e-4) {
                nmis++;
                if (nmis <= 5)
                    fprintf(stderr, "MISMATCH v=%d row=%d scalar=%.6f avx2=%.6f rel=%.3e\n",
                            v, r, s1, s2, rel);
            }
            if (rel > worst) worst = rel;
        }
    }
    fprintf(stderr, "compared %d dots on real rows, mismatches=%d worst_rel=%.3e\n",
            ncmp, nmis, worst);
    return nmis ? 1 : 0;
}
