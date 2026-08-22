#include "model.h"
#include "tensor.h"
#include "tokenizer.h"
#include "sampler.h"
#include "grammar.h"
#include "qwen_tokenize.h"
#include "picolm_server.h"
#ifdef PICOLM_VIZ
#include "viz.h"
#endif
#ifdef PICOLM_GPU
#include "backend_gpu.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <math.h>
#ifdef PICOLM_DOS
#include <alloca.h>
#include <time.h>
#include <fcntl.h>
#endif
#ifdef PICOLM_GPU
#include <cuda_profiler_api.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <time.h>
#ifndef strdup
#define strdup _strdup
#endif
double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#ifndef PICOLM_DOS
#include <sys/time.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#endif
double get_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
#elif defined(PICOLM_DOS)
    return (double)clock() / (double)CLOCKS_PER_SEC * 1000.0;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
#endif
}
#endif

/* Unescape common C-style escape sequences in a prompt string.
 * Modifies the string in-place and returns it.
 * Handles: \n, \r, \t, \\, \", and \uXXXX (Unicode escapes via UTF-8). */
static char *unescape_prompt(char *s) {
    char *dst = s;
    const char *src = s;
    while (*src) {
        if (*src == '\\' && *(src + 1)) {
            src++;
            switch (*src) {
                case 'n':  *dst++ = '\n'; break;
                case 'r':  *dst++ = '\r'; break;
                case 't':  *dst++ = '\t'; break;
                case '\\': *dst++ = '\\'; break;
                case '"':  *dst++ = '"'; break;
                case '\'': *dst++ = '\''; break;
                case 'a':  *dst++ = '\a'; break;
                case 'b':  *dst++ = '\b'; break;
                case 'f':  *dst++ = '\f'; break;
                case 'v':  *dst++ = '\v'; break;
                case 'u': {
                    /* \uXXXX -> UTF-8 */
                    unsigned int cp = 0;
                    char hex[7];
                    strncpy(hex, src + 1, 4);
                    hex[4] = '\0';
                    sscanf(hex, "%4x", &cp);
                    src += 4;
                    if (cp < 0x80) {
                        *dst++ = (char)cp;
                    } else if (cp < 0x800) {
                        *dst++ = (char)(0xC0 | (cp >> 6));
                        *dst++ = (char)(0x80 | (cp & 0x3F));
                    } else {
                        *dst++ = (char)(0xE0 | (cp >> 12));
                        *dst++ = (char)(0x80 | ((cp >> 6) & 0x3F));
                        *dst++ = (char)(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default:
                    *dst++ = '\\';
                    *dst++ = *src;
                    break;
            }
        } else {
            *dst++ = *src;
        }
        src++;
    }
    *dst = '\0';
    return s;
}

static void usage(const char *prog) {
    fprintf(stderr, "PicoLLM - ultra-lightweight LLM inference engine\n\n");
    fprintf(stderr, "Usage: %s <model.gguf> [options]\n", prog);
    fprintf(stderr, "\nGeneration options:\n");
    fprintf(stderr, "  -p <prompt>    Input prompt (or pipe via stdin)\n");
    fprintf(stderr, "  -n <int>       Max tokens to generate (default: 256)\n");
    fprintf(stderr, "  -t <float>     Temperature (default: 0.8, 0=greedy)\n");
    fprintf(stderr, "  -k <float>     Top-p / nucleus sampling (default: 0.9)\n");
    fprintf(stderr, "  -s <int>       RNG seed (default: 42)\n");
    fprintf(stderr, "  -c <int>       Context length override\n");
    fprintf(stderr, "  -j <int>       Number of threads (default: auto-detect physical cores)\n");
    fprintf(stderr, "  --mem <MB>      Pin this many MB of layers in RAM (mlock)\n");
    fprintf(stderr, "  --prefault      Prefault all model pages into RAM at load time\n");
    fprintf(stderr, "\nServer options:\n");
    fprintf(stderr, "  --server <model> Start HTTP server (OpenAI-compatible)\n");
    fprintf(stderr, "  --port <int>     Server port (default: 8080)\n");
    fprintf(stderr, "  --host <addr>    Server bind address (default: 0.0.0.0)\n");
    fprintf(stderr, "\nAdvanced options:\n");
    fprintf(stderr, "  --json         Grammar-constrained JSON output mode\n");
    fprintf(stderr, "  --cache <file> KV cache file (saves/loads prompt state)\n");
    fprintf(stderr, "  -ctk <type>    Key cache type: f16, q8_0, q4_0, tq3, tq4 (default: f16)\n");
    fprintf(stderr, "  -ctv <type>    Val cache type: f16, q8_0, q4_0, tq3, tq4 (default: f16)\n");
    fprintf(stderr, "\nGPU debug options:\n");
    fprintf(stderr, "  --gpu-diff <S> I O  Diff GPU kernels (IMMA vs scalar) on random input\n");
    fprintf(stderr, "                  S=seq_len, I=input_dim, O=output_dim (must be multiples of 16/8/32)\n");
    fprintf(stderr, "                  Example: --gpu-diff 32 512 1024\n");
    fprintf(stderr, "  --gpu-attn-diff <n_tok> <n_heads> <n_kv_heads> <head_dim>  Diff attention (FA2 vs scalar)\n");
    fprintf(stderr, "                  Generates random Q/K/V, runs FA2 and scalar kernels, diffs output.\n");
    fprintf(stderr, "                  n_tok must be >= 64, head_dim must be multiple of 16.\n");
    fprintf(stderr, "  -khad          Apply Walsh-Hadamard rotation to K cache before quantization\n");
    fprintf(stderr, "  -vhad          Apply Walsh-Hadamard rotation to V cache before quantization\n");
    fprintf(stderr, "\nServer slot options:\n");
    fprintf(stderr, "  --slot-save-path <dir>  Directory for /slots save/restore files\n");
    fprintf(stderr, "\nSSM checkpoint options (Qwen3.5/3.6 only, no-op for other models):\n");
    fprintf(stderr, "  --checkpoint-max <N>        Max checkpoints to keep (default: 0=disabled)\n");
    fprintf(stderr, "  --checkpoint-every-nt <N>   Checkpoint every N tokens during prefill (default: 256)\n");
    fprintf(stderr, "  --checkpoint-every-nt-gen <N> Checkpoint every N tokens during generation (default: 64)\n");
    fprintf(stderr, "  --checkpoint-tail-offset <N> Checkpoint N tokens before end of prompt (default: 5)\n");
    fprintf(stderr, "  --ssm-serial      Use serial per-token SSM prefill (default is batched)\n");
    fprintf(stderr, "  --ssm-chunk-size <N> SSM batch chunk size (default 64; note: different sizes may\n");
    fprintf(stderr, "                       produce slightly different outputs due to float rounding order)\n");
    fprintf(stderr, "\nInfo options:\n");
    fprintf(stderr, "  --list-tensors   List all tensors (name, shape, type) and exit\n");
    fprintf(stderr, "  --list-kv        List all KV metadata entries and exit\n");
    fprintf(stderr, "  --benchmark      Continuous benchmark loop (Ctrl-C to stop)\n");
#ifdef PICOLM_VIZ
    fprintf(stderr, "\nVisualization options:\n");
    fprintf(stderr, "  --viz             Start VNC visualization server (requires PICOLM_VIZ)\n");
    fprintf(stderr, "  --viz-port <int>  VNC port (default: 5900)\n");
    fprintf(stderr, "  --viz-res WxH     VNC resolution (default: 800x480)\n");
#endif
}

static char *read_stdin(void) {
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;

    int ch;
    while ((ch = fgetc(stdin)) != EOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *tmp = (char *)realloc(buf, cap);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
        buf[len++] = (char)ch;
    }
    buf[len] = '\0';
    return buf;
}

/* ---- Benchmark mode ---- */

/* Per-iteration stats */
typedef struct {
    /* Accumulators */
    double prefill_total_ms;       /* total prefill time */
    double gen_total_ms;           /* total generation time */
    int    prefill_layer_count;    /* total layers processed in prefill */
    int    gen_layer_count;        /* total layers processed in gen */
    /* Per-layer ring buffer (last N layers) for ETA */
#define BENCH_LAYER_HISTORY 32
    double layer_times[BENCH_LAYER_HISTORY];  /* recent layer times in ms */
    int    layer_hist_idx;
    int    layer_hist_count;
    /* Running stats */
    double layer_avg_ms;
    double layer_sum_ms;
    int    layer_total_count;
    /* Model info (set once) */
    int    n_layers;
    size_t layer_weight_sizes[128];  /* per-layer weight size in bytes */
    int    n_active_layers;
    /* Current phase */
    int    is_prefill;
    int    n_prompt_tokens;
    /* Page fault counts */
    long total_minflt;
    long total_majflt;
    /* Iteration counter */
    int iteration;
    /* Tokens generated in this iteration */
    int gen_tokens;
} bench_ctx_t;

static void bench_callback(int layer, int is_prefill, double elapsed_ms, long minflt, long majflt, void *user_data) {
    bench_ctx_t *bc = (bench_ctx_t *)user_data;
    bc->total_minflt += minflt;
    bc->total_majflt += majflt;

    /* Add to ring buffer */
    int idx = bc->layer_hist_idx % BENCH_LAYER_HISTORY;
    bc->layer_times[idx] = elapsed_ms;
    bc->layer_hist_idx++;
    if (bc->layer_hist_count < BENCH_LAYER_HISTORY) bc->layer_hist_count++;

    /* Running average */
    bc->layer_sum_ms += elapsed_ms;
    bc->layer_total_count++;
    bc->layer_avg_ms = bc->layer_sum_ms / bc->layer_total_count;

    if (is_prefill) {
        bc->prefill_total_ms += elapsed_ms;
        bc->prefill_layer_count++;
    } else {
        bc->gen_total_ms += elapsed_ms;
        bc->gen_layer_count++;
    }

    /* layers_done is the count *after* this layer, used for ETA */
    int layers_done = is_prefill ? bc->prefill_layer_count : bc->gen_layer_count;

    /* Compute ETA using recent average (ring buffer), not the all-time average.
     * The all-time average can be skewed by slow prefill layers from earlier
     * in the iteration. The ring buffer of the last 32 layers gives a more
     * accurate picture of current performance. */
    double recent_avg_ms = 0;
    {
        double ring_sum = 0;
        int ring_count = 0;
        for (int i = 0; i < bc->layer_hist_count; i++) {
            ring_sum += bc->layer_times[i];
            ring_count++;
        }
        recent_avg_ms = ring_count > 0 ? ring_sum / ring_count : bc->layer_avg_ms;
    }

    int layers_remaining;
    if (is_prefill) {
        /* Prefill: ETA to complete the prefill phase */
        layers_remaining = bc->n_active_layers - layers_done;
    } else {
        /* Generation: ETA to the NEXT token (end of current forward pass).
         * layers_done wraps every n_active_layers. When layers_done is a
         * multiple of n_active_layers, the current token just completed. */
        int layers_in_token = layers_done % bc->n_active_layers;
        layers_remaining = (layers_in_token == 0) ? 0 : (bc->n_active_layers - layers_in_token);
    }
    double eta_ms = layers_remaining * recent_avg_ms;
    double eta_sec = eta_ms / 1000.0;

    /* Effective weight throughput: weight_bytes / elapsed_ms.
     * This captures the combined I/O + compute bottleneck - how fast the layer
     * "consumes" its own weights end-to-end. */
    size_t wsize = 0;
    if (layer >= 0 && layer < bc->n_layers) {
        wsize = bc->layer_weight_sizes[layer];
    }
        double wt_mbs = 0; /* MB/s effective weight throughput */
    if (elapsed_ms > 0 && wsize > 0) {
        wt_mbs = (double)wsize / (elapsed_ms / 1000.0) / (1024.0 * 1024.0);
    }

    /* Format ETA */
    char eta_str[32];
    if (eta_sec < 1.0) snprintf(eta_str, sizeof(eta_str), "%.0fms", eta_sec * 1000.0);
    else if (eta_sec < 60.0) snprintf(eta_str, sizeof(eta_str), "%.1fs", eta_sec);
    else snprintf(eta_str, sizeof(eta_str), "%.0fm%.0fs", eta_sec / 60.0, fmod(eta_sec, 60.0));

    /* Format I/O column: weight throughput + I/O fraction */
    char bw_str[64];
    if (wsize > 0 && wt_mbs > 0) {
        /* Compute I/O fraction: what % of the effective throughput is actual I/O */
        double io_bytes = (double)majflt * 4096;
        double io_mbs = elapsed_ms > 0 ? io_bytes / (elapsed_ms / 1000.0) / (1024.0 * 1024.0) : 0;
        double io_pct = (wt_mbs > 0) ? (io_mbs / wt_mbs) * 100.0 : 0;
        /* Also compute "cache I/O" fraction from minor faults */
        double cache_bytes = (double)minflt * 4096;
        double cache_mbs = elapsed_ms > 0 ? cache_bytes / (elapsed_ms / 1000.0) / (1024.0 * 1024.0) : 0;
        double cache_pct = (wt_mbs > 0) ? (cache_mbs / wt_mbs) * 100.0 : 0;

        if (wt_mbs >= 100.0)
            snprintf(bw_str, sizeof(bw_str), "%.0fMB/s io=%.0f%% cache=%.0f%%", wt_mbs, io_pct, cache_pct);
        else if (wt_mbs >= 1.0)
            snprintf(bw_str, sizeof(bw_str), "%.1fMB/s io=%.0f%% cache=%.0f%%", wt_mbs, io_pct, cache_pct);
        else
            snprintf(bw_str, sizeof(bw_str), "%.0fKB/s io=%.0f%% cache=%.0f%%", wt_mbs*1024.0, io_pct, cache_pct);
    } else {
        snprintf(bw_str, sizeof(bw_str), "?MB/s");
    }

    /* Format page faults */
    /* Print per-layer line */
    const char *phase = is_prefill ? "Prefill" : "Gen";
    fprintf(stderr, "%s [layer %d/%d]: %6.1fms  %-8s  %-14s",
            phase, layer, bc->n_active_layers, elapsed_ms, eta_str, bw_str);
    if (minflt > 0 || majflt > 0) {
        fprintf(stderr, " [pf=%lu+%lu]", (unsigned long)minflt, (unsigned long)majflt);
    }
    fprintf(stderr, "\n");
}

/* Print benchmark summary */
static void bench_summary(const bench_ctx_t *bc, int iteration, double wall_sec) {
    fprintf(stderr, "\n--- Iteration %d (%.2fs wall) ---\n", iteration, wall_sec);
    if (bc->prefill_layer_count > 0) {
        double prefill_tok_s = (bc->prefill_total_ms > 0) ?
            (double)bc->n_prompt_tokens / (bc->prefill_total_ms / 1000.0) : 0;
        fprintf(stderr, "  Prefill: %d tokens, %d layers, %.1fms (%.1f tok/s, %.1fms/layer)\n",
                bc->n_prompt_tokens, bc->prefill_layer_count,
                bc->prefill_total_ms, prefill_tok_s,
                bc->prefill_total_ms / bc->prefill_layer_count);
    }
    if (bc->gen_layer_count > 0) {
        double gen_tok_s = (bc->gen_total_ms > 0) ?
            (double)bc->gen_tokens / (bc->gen_total_ms / 1000.0) : 0;
        fprintf(stderr, "  Generation: %d tokens, %d layers, %.1fms (%.1f tok/s, %.1fms/token)\n",
                bc->gen_tokens, bc->gen_layer_count,
                bc->gen_total_ms, gen_tok_s,
                bc->gen_total_ms / bc->gen_tokens);
    }
    if (bc->total_minflt > 0 || bc->total_majflt > 0) {
        fprintf(stderr, "  Page faults: %lu minor, %lu major\n",
                (unsigned long)bc->total_minflt, (unsigned long)bc->total_majflt);
    }
    fprintf(stderr, "  Total wall time: %.2fs\n", wall_sec);
    fprintf(stderr, "----------------------------------\n\n");
}

#ifdef PICOLM_GPU
#include <cuda_runtime.h>

/* Full struct definition needed from C (forward decl in tensor.h is opaque) */
struct picolm_gpu_tensor {
    void *weights;
    int qtype;
    int I, O, device;
    size_t row_bytes;
    int block_size;
    int tracked;
    int zero_copy;
};

/* GPU kernel diff test: quantize random input to Q8_0, run IMMA and scalar
 * matmul on the same data, diff outputs. Used to validate kernel correctness. */
static void gpu_matmul_diff(int S, int I, int O) {
    if (I % 32 != 0) { fprintf(stderr, "I must be multiple of 32\n"); exit(1); }
    if (S % 16 != 0) { fprintf(stderr, "S must be multiple of 16\n"); exit(1); }
    if (O % 8 != 0) { fprintf(stderr, "O must be multiple of 8\n"); exit(1); }

    size_t ib = (size_t)S * I * sizeof(float);
    size_t ob = (size_t)S * O * sizeof(float);

    float *x = (float *)malloc(ib);
    float *y_imma = (float *)malloc(ob);
    float *y_scalar = (float *)malloc(ob);
    if (!x || !y_imma || !y_scalar) { fprintf(stderr, "OOM\n"); exit(1); }

    /* Generate random weights as Q8_0 blocks on host */
    int n_blocks = I / 32;
    size_t w_bytes = (size_t)O * (2 + 32 * n_blocks);
    uint8_t *w = (uint8_t *)malloc(w_bytes);
    if (!w) { fprintf(stderr, "OOM\n"); exit(1); }

    srand48(42);
    for (int i = 0; i < S * I; i++) x[i] = (float)(drand48() * 2.0 - 1.0);

    for (int o = 0; o < O; o++) {
        for (int b = 0; b < n_blocks; b++) {
            uint8_t *blk = w + (size_t)o * (2 + 32 * n_blocks) + b * 34;
            float d = 0.01f + (float)drand48() * 0.1f;
            /* FP16 conversion: use simple bit hack matching CPU path */
            union { uint32_t u; float f; } u32;
            u32.f = d;
            uint16_t d16 = (uint16_t)(u32.u >> 16); /* crude but works for positive floats */
            blk[0] = d16 & 0xFF;
            blk[1] = (d16 >> 8) & 0xFF;
            for (int j = 0; j < 32; j++)
                blk[2 + j] = (uint8_t)((int)(drand48() * 256) - 128);
        }
    }

    void *d_x, *d_y_imma, *d_y_scalar, *d_w;
    cudaError_t e;
    if ((e = cudaMalloc(&d_x, ib)) != cudaSuccess) { fprintf(stderr, "cudaMalloc x: %s\n", cudaGetErrorString(e)); exit(1); }
    if ((e = cudaMalloc(&d_y_imma, ob)) != cudaSuccess) { fprintf(stderr, "cudaMalloc y: %s\n", cudaGetErrorString(e)); exit(1); }
    if ((e = cudaMalloc(&d_y_scalar, ob)) != cudaSuccess) { fprintf(stderr, "cudaMalloc y: %s\n", cudaGetErrorString(e)); exit(1); }
    if ((e = cudaMalloc(&d_w, w_bytes)) != cudaSuccess) { fprintf(stderr, "cudaMalloc w: %s\n", cudaGetErrorString(e)); exit(1); }
    cudaMemcpy(d_x, x, ib, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, w_bytes, cudaMemcpyHostToDevice);

    struct picolm_gpu_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.I = I;
    tensor.O = O;
    tensor.qtype = 8; /* GGUF_TYPE_Q8_0 */
    tensor.row_bytes = 2 + 32 * n_blocks;
    tensor.weights = d_w;
    tensor.device = 0;

    /* Initialize GPU context */
    int devs[1] = {0};
    if (!picolm_gpu_init(devs, 1)) {
        fprintf(stderr, "GPU init failed\n"); exit(1);
    }

    if (!picolm_gpu_matmul(&tensor, y_imma, x, S, 0)) {
        fprintf(stderr, "IMMA matmul failed (I=%d O=%d S=%d, need I>=512 O>=256)\n", I, O, S); exit(1);
    }

    setenv("PICOLM_FORCE_F32_MATMUL", "1", 1);
    if (!picolm_gpu_matmul(&tensor, y_scalar, x, S, 0)) {
        fprintf(stderr, "Scalar matmul failed\n"); exit(1);
    }
    unsetenv("PICOLM_FORCE_F32_MATMUL");

    float max_abs_err = 0.0f, max_rel_err = 0.0f;
    int max_err_pos = 0;
    size_t n = (size_t)S * O;
    for (size_t i = 0; i < n; i++) {
        float diff = y_imma[i] - y_scalar[i];
        if (diff < 0) diff = -diff;
        float rel = (fabsf(y_scalar[i]) > 1e-8f) ? diff / fabsf(y_scalar[i]) : diff;
        if (diff > max_abs_err) {
            max_abs_err = diff; max_rel_err = rel; max_err_pos = (int)i;
        }
    }

    int sy = max_err_pos / O, oy = max_err_pos % O;
    fprintf(stderr, "GPU kernel diff: S=%d I=%d O=%d, %zu elements\n", S, I, O, n);
    fprintf(stderr, "  max_abs_err = %.6f (at s=%d, o=%d)\n", max_abs_err, sy, oy);
    fprintf(stderr, "  max_rel_err = %.6f\n", max_rel_err);
    fprintf(stderr, "  y_imma[%d][%d] = %f\n", sy, oy, y_imma[max_err_pos]);
    fprintf(stderr, "  y_scalar[%d][%d] = %f\n", sy, oy, y_scalar[max_err_pos]);

    if (max_rel_err > 0.01f) {
        fprintf(stderr, "  RESULT: FAIL (rel_err > 1%%)\n");
    } else {
        fprintf(stderr, "  RESULT: PASS (rel_err <= 1%%)\n");
    }

    cudaFree(d_x); cudaFree(d_y_imma); cudaFree(d_y_scalar); cudaFree(d_w);
    free(x); free(y_imma); free(y_scalar); free(w);
}

/* GPU attention diff test: generate random Q/K/V, run FA2 and scalar
 * attention kernels, diff the output. No model needed. */
static void gpu_attn_diff_test(int n_tokens, int n_heads, int n_kv_heads, int head_dim) {
    if (n_tokens < 64) { fprintf(stderr, "n_tokens must be >= 64\n"); exit(1); }
    if (head_dim % 16 != 0) { fprintf(stderr, "head_dim must be multiple of 16\n"); exit(1); }
    if (n_heads % n_kv_heads != 0) { fprintf(stderr, "n_heads must be multiple of n_kv_heads\n"); exit(1); }

    /* Initialize GPU */
    int devs[1] = {0};
    if (!picolm_gpu_init(devs, 1)) { fprintf(stderr, "GPU init failed\n"); exit(1); }

    size_t q_bytes = (size_t)n_tokens * n_heads * head_dim * sizeof(float);
    size_t kv_bytes = (size_t)n_tokens * n_kv_heads * head_dim * sizeof(uint16_t);
    size_t out_bytes = q_bytes;

    /* Allocate host buffers */
    float *q_h = (float *)malloc(q_bytes);
    uint16_t *k_h = (uint16_t *)malloc(kv_bytes);
    uint16_t *v_h = (uint16_t *)malloc(kv_bytes);
    float *y_fa2 = (float *)malloc(out_bytes);
    float *y_scalar = (float *)malloc(out_bytes);
    if (!q_h || !k_h || !v_h || !y_fa2 || !y_scalar) { fprintf(stderr, "OOM\n"); exit(1); }

    /* Generate random Q (FP32), K (FP16), V (FP16) */
    srand48(12345);
    for (int i = 0; i < n_tokens * n_heads * head_dim; i++) {
        q_h[i] = (float)(drand48() * 2.0 - 1.0) * 0.5f;
    }
    for (int i = 0; i < n_tokens * n_kv_heads * head_dim; i++) {
        float vf = (float)(drand48() * 2.0 - 1.0) * 0.5f;
        union { uint32_t u; float f; } u32;
        u32.f = vf;
        k_h[i] = u32.u >> 16;
        vf = (float)(drand48() * 2.0 - 1.0) * 0.5f;
        u32.f = vf;
        v_h[i] = u32.u >> 16;
    }

    /* Set up KV cache (required by attention kernels) */
    size_t layer_kv_bytes = (size_t)n_tokens * n_kv_heads * head_dim * sizeof(uint16_t);
    int max_seq_len = n_tokens;
    if (!picolm_gpu_kv_alloc(layer_kv_bytes, layer_kv_bytes, 0)) { fprintf(stderr, "KV alloc failed\n"); exit(1); }

    /* Store K and V into KV cache using host API */
    size_t row_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    /* k_h and v_h are flat [n_tokens][n_kv_heads][head_dim] uint16_t */
    for (int p = 0; p < n_tokens; p++) {
        uint16_t *row = k_h + (size_t)p * n_kv_heads * head_dim;
        picolm_gpu_kv_store_rows(1, 0, p, 1, row, row_bytes,
            n_kv_heads, head_dim, max_seq_len, 0);
    }
    for (int p = 0; p < n_tokens; p++) {
        uint16_t *row = v_h + (size_t)p * n_kv_heads * head_dim;
        picolm_gpu_kv_store_rows(0, 0, p, 1, row, row_bytes,
            n_kv_heads, head_dim, max_seq_len, 0);
    }

    /* Run FA2 and scalar kernels multiple times on the same stream to
     * stress-test for race conditions / stale shared memory / OOB writes */
    const int n_iters = 500;
    fprintf(stderr, "Running %d stress iterations...\n", n_iters);

    unsetenv("PICOLM_FORCE_SCALAR_ATTN");
    for (int iter = 0; iter < n_iters; iter++) {
        if (!picolm_gpu_attention_prefill(y_fa2, q_h, 0, 0, n_tokens,
                n_heads, n_kv_heads, head_dim, max_seq_len, 0)) {
            fprintf(stderr, "FA2 attention failed iter %d\n", iter); exit(1);
        }
        setenv("PICOLM_FORCE_SCALAR_ATTN", "1", 1);
        if (!picolm_gpu_attention_prefill(y_scalar, q_h, 0, 0, n_tokens,
                n_heads, n_kv_heads, head_dim, max_seq_len, 0)) {
            fprintf(stderr, "Scalar attention failed iter %d\n", iter); exit(1);
        }
        unsetenv("PICOLM_FORCE_SCALAR_ATTN");

        /* Check after each iteration */
        float max_abs = 0, max_rel = 0;
        int max_pos = 0;
        size_t n = (size_t)n_tokens * n_heads * head_dim;
        for (size_t i = 0; i < n; i++) {
            float diff = y_fa2[i] - y_scalar[i];
            if (diff < 0) diff = -diff;
            float rel = (fabsf(y_scalar[i]) > 1e-8f) ? diff / fabsf(y_scalar[i]) : diff;
            if (diff > max_abs) { max_abs = diff; max_rel = rel; max_pos = (int)i; }
        }
        if (max_rel > 0.01f) {
            int tk = max_pos / (n_heads * head_dim);
            int hh = (max_pos / head_dim) % n_heads;
            int dd = max_pos % head_dim;
            fprintf(stderr, "  max_abs_err = %.6f (tok=%d, head=%d, dim=%d) at iter %d\n",
                max_abs, tk, hh, dd, iter);
            fprintf(stderr, "  max_rel_err = %.6f\n", max_rel);
            fprintf(stderr, "  RESULT: DIFFER (rel_err > 1%%)\n");
            return;
        }
    }

    fprintf(stderr, "GPU attn diff: %d tok, %d heads, %d kv_heads, hd=%d, %d iters\n",
        n_tokens, n_heads, n_kv_heads, head_dim, n_iters);
    fprintf(stderr, "  RESULT: MATCH (all %d iterations)\n", n_iters);

    free(q_h); free(k_h); free(v_h); free(y_fa2); free(y_scalar);
}
#endif

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char *model_path = NULL;
    char *prompt_buf = NULL;   /* malloc'd buffer for -p prompt (may be modified by unescape) */
    const char *prompt = NULL;
    int    max_tokens = 256;
    float  temperature = 0.8f;
    float  top_p = 0.9f;
    int    top_k = 40;
    uint64_t seed = 42;
    int    context_override = 0;
    int    num_threads = 0; /* 0 = auto-detect from physical cores */
    int    json_mode = 0;
    int    benchmark_mode = 0;
    const char *cache_file = NULL;
    kv_cache_type_t kv_type_k = KV_CACHE_F16;
    kv_cache_type_t kv_type_v = KV_CACHE_F16;
    int    k_cache_hadamard = 0;  /* -khad: Walsh-Hadamard rotation for K cache */
    int    v_cache_hadamard = 0;  /* -vhad: Walsh-Hadamard rotation for V cache */
    int    mem_mb = 0;      /* --mem budget in megabytes (0=disabled) */
    int    do_prefault = 0; /* --prefault (touch all mmap pages at load time) */
    int    gpu_diff = 0;    /* --gpu-diff S I O */
    int    gpu_diff_S = 32, gpu_diff_I = 512, gpu_diff_O = 1024;
    int    do_attn_diff = 0;  /* --gpu-attn-diff n_tok n_heads n_kv_heads head_dim */
    int    attn_n_tok = 64, attn_n_heads = 40, attn_n_kv = 8, attn_head_dim = 128;
    #if !defined(_WIN32) && !defined(PICOLM_DOS)
    int    server_daemon = 0;
#endif
    int    server_mode = 0;
    int    server_port = 8080;
    char   server_host[256] = "0.0.0.0";
    /* SSM options */
    int    ssm_batched_prefill = 1;     /* 1=batched (default), 0=per-token */
    int    ssm_chunk_size = 0;          /* 0=default(64), 1=serial-equivalent */
    /* SSM checkpoint options */
    int    checkpoint_max = 0;          /* 0=disabled */
    int    checkpoint_interval = 256;
    int    checkpoint_interval_gen = 64;
    int    checkpoint_tail_offset = 1;  /* 1: ensures checkpoint at n_prompt-1 for stepback */
    char  *slot_save_path = NULL;  /* --slot-save-path */
#ifdef PICOLM_VIZ
    /* Visualization options */
    int    viz_mode = 0;
    int    viz_port = 5900;
    int    viz_width = 800;
    int    viz_height = 480;
#endif

    /* Parse arguments */
    int list_tensors = 0;
    int list_kv = 0;
    for (int i = 1; i < argc; i++) {
        /* Positional model path: first non-dash arg */
        if (argv[i][0] != '-') {
            if (!model_path) model_path = argv[i];
            continue;
        }
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
#ifdef PICOLM_DOS
            { size_t len = strlen(argv[i+1]) + 1;
              prompt_buf = malloc(len);
              if (!prompt_buf) { fprintf(stderr, "strdup failed\n"); return 1; }
              memcpy(prompt_buf, argv[++i], len); }
#else
            prompt_buf = strdup(argv[++i]);
            if (!prompt_buf) { fprintf(stderr, "strdup failed\n"); return 1; }
#endif
            unescape_prompt(prompt_buf);
            prompt = prompt_buf;
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            context_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        #if !defined(_WIN32) && !defined(PICOLM_DOS)
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--daemon") == 0) {
            server_daemon = 1;
#endif
        } else if (strcmp(argv[i], "--server") == 0 && i + 1 < argc) {
            server_mode = 1;
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            server_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            strncpy(server_host, argv[++i], sizeof(server_host) - 1);
            server_host[sizeof(server_host) - 1] = '\0';
        } else if (strcmp(argv[i], "--json") == 0) {
            json_mode = 1;
        } else if (strcmp(argv[i], "--cache") == 0 && i + 1 < argc) {
            cache_file = argv[++i];
        } else if (strcmp(argv[i], "--mem") == 0 && i + 1 < argc) {
            mem_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prefault") == 0) {
            do_prefault = 1;
        } else if ((strcmp(argv[i], "-ctk") == 0 || strcmp(argv[i], "-ctv") == 0) && i + 1 < argc) {
            const char *typestr = argv[++i];
            kv_cache_type_t *tgt = (strcmp(argv[i-1], "-ctk") == 0) ? &kv_type_k : &kv_type_v;
            if (strcmp(typestr, "q8_0") == 0) *tgt = KV_CACHE_Q8_0;
            else if (strcmp(typestr, "q4_0") == 0) *tgt = KV_CACHE_Q4_0;
            else if (strcmp(typestr, "tq3") == 0) *tgt = KV_CACHE_TQ3;
            else if (strcmp(typestr, "tq4") == 0) *tgt = KV_CACHE_TQ4;
            else if (strcmp(typestr, "f16") == 0) *tgt = KV_CACHE_F16;
            else { fprintf(stderr, "Unknown KV cache type: %s (use f16, q8_0, q4_0, tq3, tq4)\n", typestr); return 1; }
        } else if (strcmp(argv[i], "-khad") == 0) {
            k_cache_hadamard = 1;
        } else if (strcmp(argv[i], "-vhad") == 0) {
            v_cache_hadamard = 1;
        } else if (strcmp(argv[i], "--checkpoint-max") == 0 && i + 1 < argc) {
            checkpoint_max = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--checkpoint-every-nt") == 0 && i + 1 < argc) {
            checkpoint_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--checkpoint-every-nt-gen") == 0 && i + 1 < argc) {
            checkpoint_interval_gen = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--checkpoint-tail-offset") == 0 && i + 1 < argc) {
            checkpoint_tail_offset = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--slot-save-path") == 0 && i + 1 < argc) {
            slot_save_path = argv[++i];
        } else if (strcmp(argv[i], "--gpu-diff") == 0 && i + 3 < argc) {
            gpu_diff = 1;
            gpu_diff_S = atoi(argv[++i]);
            gpu_diff_I = atoi(argv[++i]);
            gpu_diff_O = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-attn-diff") == 0 && i + 4 < argc) {
            do_attn_diff = 1;
            attn_n_tok = atoi(argv[++i]);
            attn_n_heads = atoi(argv[++i]);
            attn_n_kv = atoi(argv[++i]);
            attn_head_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ssm-batched") == 0) {
            /* no-op: batched is now the default */
        } else if (strcmp(argv[i], "--ssm-serial") == 0) {
            ssm_batched_prefill = 0;
        } else if (strcmp(argv[i], "--ssm-chunk-size") == 0 && i + 1 < argc) {
            ssm_chunk_size = atoi(argv[++i]);
            if (ssm_chunk_size < 1) {
                fprintf(stderr, "ERROR: --ssm-chunk-size must be >= 1\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = 1;
#ifdef PICOLM_VIZ
        } else if (strcmp(argv[i], "--viz") == 0) {
            viz_mode = 1;
        } else if (strcmp(argv[i], "--viz-port") == 0 && i + 1 < argc) {
            viz_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--viz-res") == 0 && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &viz_width, &viz_height) < 2) {
                fprintf(stderr, "Invalid resolution: %s (use WxH, e.g. 800x480)\n", argv[i]);
                return 1;
            }
#endif
        } else if (strcmp(argv[i], "--list-tensors") == 0) {
            list_tensors = 1;
        } else if (strcmp(argv[i], "--list-kv") == 0) {
            list_kv = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* --gpu-attn-diff: attention kernel diff test (FA2 vs scalar) */
    if (do_attn_diff) {
        if (getenv("PICOLM_GPU") == NULL) {
            fprintf(stderr, "GPU not available (PICOLM_GPU not set)\n");
            return 1;
        }
        gpu_attn_diff_test(attn_n_tok, attn_n_heads, attn_n_kv, attn_head_dim);
        return 0;
    }

    /* --gpu-diff: standalone GPU kernel diff test, no model needed */
    if (gpu_diff) {
        if (getenv("PICOLM_GPU") == NULL) {
            fprintf(stderr, "GPU not available (PICOLM_GPU not set)\n");
            return 1;
        }
        gpu_matmul_diff(gpu_diff_S, gpu_diff_I, gpu_diff_O);
        return 0;
    }

    /* --list-tensors / --list-kv: now that model_path is known regardless of arg order */
    if (list_tensors || list_kv) {
        if (!model_path) {
            fprintf(stderr, "No model file specified\n");
            usage(argv[0]);
            return 1;
        }
        if (list_tensors) model_list_tensors(model_path);
        else model_list_kv(model_path);
        return 0;
    }

    /* Read prompt from stdin if not provided via -p */
    char *stdin_prompt = NULL;
    if (!prompt) {
#ifdef _WIN32
        HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
        DWORD mode;
        if (!GetConsoleMode(h, &mode)) {
            stdin_prompt = read_stdin();
            prompt = stdin_prompt;
        }
#elif !defined(PICOLM_DOS)
        if (!isatty(fileno(stdin))) {
            stdin_prompt = read_stdin();
            prompt = stdin_prompt;
        }
#endif /* _WIN32 */
    }

    /* Resolve thread count: if not specified, auto-detect physical cores */
    if (num_threads <= 0) {
        num_threads = tensor_default_threads();
    }

    /* Server mode: start HTTP server (no prompt needed) */
    if (server_mode) {
        /* Daemonize if requested (Unix-only) */
#if !defined(_WIN32) && !defined(PICOLM_DOS)
        if (server_daemon) {
            pid_t pid = fork();
            if (pid < 0) { perror("fork"); return 1; }
            if (pid > 0) {
                fprintf(stderr, "[server] Daemonized with PID %d\n", (int)pid);
                return 0;
            }
            /* Child: redirect stdio to /dev/null */
            FILE *null = fopen("/dev/null", "r");
            if (null) { dup2(fileno(null), STDIN_FILENO); fclose(null); }
            null = fopen("/dev/null", "w");
            if (null) dup2(fileno(null), STDOUT_FILENO);
            null = fopen("/tmp/picolm_stderr.log", "a");
            if (null) dup2(fileno(null), STDERR_FILENO);
            setsid(); /* create new session */
            setsid(); /* create new session */
        }
#endif
        fp16_table_init();
        /* When --viz is specified in server mode, pass the viz parameters through.
         * When --viz is not specified, viz_port=0 so server_init skips viz. */
        int srv_viz_port = 0, srv_viz_width = 800, srv_viz_height = 480;
#ifdef PICOLM_VIZ
        srv_viz_port = viz_mode ? viz_port : 0;
        srv_viz_width = viz_width;
        srv_viz_height = viz_height;
#endif
        server_config_t sc = {0};
        sc.port = server_port;
        memcpy(sc.host, server_host, sizeof(server_host)); sc.host[sizeof(sc.host) - 1] = '\0';
        sc.model_path = model_path;
        sc.num_threads = num_threads;
        sc.do_prefault = do_prefault;
        sc.context_override = context_override;
        sc.mem_mb = mem_mb;
        sc.checkpoint_max = checkpoint_max;
        sc.checkpoint_interval = checkpoint_interval;
        sc.checkpoint_interval_gen = checkpoint_interval_gen;
        sc.checkpoint_tail_offset = checkpoint_tail_offset;
        sc.slot_save_path = slot_save_path;
        sc.kv_type_k = kv_type_k;
        sc.kv_type_v = kv_type_v;
        sc.k_cache_hadamard = k_cache_hadamard;
        sc.v_cache_hadamard = v_cache_hadamard;
        sc.viz_port = srv_viz_port;
        sc.viz_width = srv_viz_width;
        sc.viz_height = srv_viz_height;
        return server_main(&sc);
    }

    if (!prompt) {
        fprintf(stderr, "No prompt provided. Use -p or pipe via stdin.\n");
        usage(argv[0]);
        return 1;
    }

    /* Load model */
    fprintf(stderr, "PicoLM v1.0-beta2\n");
    fprintf(stderr, "Loading model: %s\n", model_path);
    #if defined(PICOLM_AVX512)
    fprintf(stderr, "SIMD: AVX-512\n");
#elif defined(PICOLM_AVX2)
    fprintf(stderr, "SIMD: AVX2\n");
#elif defined(PICOLM_AVX)
    fprintf(stderr, "SIMD: AVX\n");
#elif defined(PICOLM_SSE3)
    fprintf(stderr, "SIMD: SSE3\n");
#elif defined(PICOLM_SSE2)
    fprintf(stderr, "SIMD: SSE2\n");
#elif defined(PICOLM_NEON)
#if defined(PICOLM_I8MM)
    fprintf(stderr, "SIMD: NEON +I8MM\n");
#else
    fprintf(stderr, "SIMD: NEON\n");
#endif
#elif defined(PICOLM_ALTIVEC)
    fprintf(stderr, "SIMD: Altivec\n");
#else
    fprintf(stderr, "SIMD: scalar\n");
#endif

    /* Runtime check: warn if CPU supports I8MM but compiler didn't enable it.
     * Linux-only: it reads the ELF auxiliary vector (getauxval/AT_HWCAP2),
     * which does not exist on macOS. Apple Silicon is __aarch64__ too, so it
     * must be excluded explicitly (this also broke the macOS/`make metal` and
     * `make native` builds with "sys/auxv.h not found"). */
#if defined(__aarch64__) && !defined(PICOLM_I8MM) && !defined(_WIN32) && !defined(__APPLE__)
    {
        #include <sys/auxv.h>
        #include <elf.h>
        unsigned long hwcap2 = getauxval(AT_HWCAP2);
        if (hwcap2 & (1UL << 13)) { /* HWCAP2_I8MM */
            fprintf(stderr, "NOTE: CPU supports I8MM integer dot product but it was not compiled in.\n");
            fprintf(stderr, "      Rebuild with GCC 15+ (-march=native) for ~2-3x faster Q4_K and Q8_0 inference.\n");
        }
    }
#endif

    model_t model;
    /* Initialize FP16->FP32 lookup table (64KB) for fast attention */
    fp16_table_init();

    /* Auto-detect: if path is a directory with safetensors files, use safetensors loader */
    /* Enable prefaulting before model load (affects mmap_file in model_load) */
    model_set_prefault(do_prefault);

    int is_safetensors = 0;
#if !defined(_WIN32) && !defined(PICOLM_DOS)
    {
        char idx_path[4096];
        snprintf(idx_path, sizeof(idx_path), "%s/model.safetensors.index.json", model_path);
        if (access(idx_path, F_OK) == 0) is_safetensors = 1;
        if (!is_safetensors) {
            snprintf(idx_path, sizeof(idx_path), "%s/model.safetensors", model_path);
            if (access(idx_path, F_OK) == 0) is_safetensors = 1;
        }
    }
#elif defined(_WIN32)
    {
        char idx_path[4096];
        snprintf(idx_path, sizeof(idx_path), "%s/model.safetensors.index.json", model_path);
        if (GetFileAttributesA(idx_path) != INVALID_FILE_ATTRIBUTES) is_safetensors = 1;
        if (!is_safetensors) {
            snprintf(idx_path, sizeof(idx_path), "%s/model.safetensors", model_path);
            if (GetFileAttributesA(idx_path) != INVALID_FILE_ATTRIBUTES) is_safetensors = 1;
        }
    }
#endif
    /* Guard: TQ3 is too lossy for mixed-mode. Require TQ3/TQ3 pairing. */
    if ((kv_type_k == KV_CACHE_TQ3 && kv_type_v != KV_CACHE_TQ3) ||
        (kv_type_v == KV_CACHE_TQ3 && kv_type_k != KV_CACHE_TQ3)) {
        fprintf(stderr, "ERROR: TQ3 KV cache must be used for both K and V (-ctk tq3 -ctv tq3).\n");
        fprintf(stderr, "       Mixed-mode (TQ3 K + non-TQ3 V or vice versa) produces corrupted output\n");
        fprintf(stderr, "       because 3-bit quantization error in attention scores is too large.\n");
        return 1;
    }
    int load_ok = 0;
    if (is_safetensors) {
        load_ok = (model_load_safetensors(&model, model_path, context_override, kv_type_k, kv_type_v, k_cache_hadamard, v_cache_hadamard, num_threads) == 0);
    } else {
        load_ok = (model_load(&model, model_path, context_override, kv_type_k, kv_type_v, k_cache_hadamard, v_cache_hadamard, num_threads) == 0);
    }
    if (!load_ok) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    model.ssm_batched_prefill = ssm_batched_prefill;
    model.ssm_chunk_size = ssm_chunk_size;

#ifdef PICOLM_VIZ
    /* Start visualization server if requested */
    if (viz_mode) {
        /* Build a short model name for the VNC title */
        char model_short[256];
        const char *base = model_path;
        const char *slash = strrchr(model_path, '/');
        if (slash) base = slash + 1;
        /* Strip extension */
        strncpy(model_short, base, sizeof(model_short) - 1);
        model_short[sizeof(model_short) - 1] = '\0';
        char *dot = strrchr(model_short, '.');
        if (dot) *dot = '\0';

        if (viz_init(viz_width, viz_height, viz_port) != 0) {
            fprintf(stderr, "Failed to start visualization server\n");
            /* Non-fatal: continue without viz */
        } else {
            viz_set_model_info(model.config.n_layers, model.config.n_embd, model_short);
        }
    }
#endif

    /* Pin layers in RAM if --mem was specified */
    if (mem_mb > 0) {
        size_t budget = (size_t)mem_mb * 1024 * 1024;
        model_lock_layers(&model, budget);
    }

    tensor_set_threads(num_threads);
    tensor_threadpool_init(num_threads);
#ifdef _OPENMP
    omp_set_num_threads(num_threads); /* match OpenMP to custom pool */
#endif

    /* Load tokenizer */
    clock_t t_tok_start = clock();
    tokenizer_t tokenizer;
    int use_qwen_tok = qwen_tokenize_should_use(&model);
    qwen_enc_t qwen_enc = {0};

    if (use_qwen_tok) {
        if (qwen_tokenize_init(&qwen_enc, &model) != 0) {
            fprintf(stderr, "Failed to init Qwen tokenizer\n");
            model_free(&model);
            return 1;
        }
        tokenizer.bos_id = qwen_enc.bos_id;
        tokenizer.eos_id = qwen_enc.eos_id;
        tokenizer.vocab_size = qwen_enc.vocab_size;
        fprintf(stderr, "Using native Qwen GPT-2 BPE tokenizer\n");
    } else {
        tokenizer_load(&tokenizer, &model);
    }
    fprintf(stderr, "%0.1fms: tokenizer loaded\n", (double)(clock() - t_tok_start) / CLOCKS_PER_SEC * 1000.0);

    /* Warn about unsupported tokenizer types */
    if (model.tok_unknown_model) {
        fprintf(stderr, "WARN: tokenizer model type not supported by PicoLM (using SPM fallback). Output may differ from reference.\n");
    }

    /* Init sampler */
    sampler_t sampler;
    sampler_init(&sampler, temperature, top_p, top_k, 0.05f, seed);

    /* Buffer for generated text (printed at the end for easy grepping) */
    char *gen_buf = malloc(max_tokens * 16); /* avg 16 bytes per token */
    int gen_buf_len = 0;

    /* Init grammar constraint */
    grammar_state_t grammar;
    grammar_init(&grammar, json_mode ? GRAMMAR_JSON : GRAMMAR_NONE, &tokenizer);
    if (json_mode) {
        fprintf(stderr, "JSON grammar mode enabled\n");
    }

    /* Try to load KV cache (skip prefill for cached prompt) */
    int cache_pos = 0;
    int *cache_tokens = NULL;
    if (cache_file) {
        cache_pos = kvcache_load(&model, cache_file, &cache_tokens);
    }

    /* Encode prompt */
    clock_t t_start_token = clock();
    int max_prompt_tokens = (int)strlen(prompt) + 16;
    int *prompt_tokens = (int *)malloc((size_t)max_prompt_tokens * sizeof(int));
    int n_prompt;
    if (use_qwen_tok) {
        /* Respect tok_add_bos from GGUF metadata. Qwen3.5 models typically have
           add_bos=false, Qwen3.6 GGUFs may set it to 0 or 1. The server path
           doesn't prepend BOS for Qwen models, so CLI should match. */
        if (model.tok_add_bos && tokenizer.bos_id != 11) {
            n_prompt = qwen_tokenize_encode(&qwen_enc, prompt, prompt_tokens + 1, max_prompt_tokens - 1);
            prompt_tokens[0] = (int)tokenizer.bos_id;
            n_prompt++;
        } else {
            n_prompt = qwen_tokenize_encode(&qwen_enc, prompt, prompt_tokens, max_prompt_tokens);
        }
    } else {
        n_prompt = tokenizer_encode(&tokenizer, prompt, prompt_tokens, max_prompt_tokens, model.tok_add_bos);
    }
    fprintf(stderr, "%0.1fms: tokenized %d tokens\n", (double)(clock() - t_start_token) / CLOCKS_PER_SEC * 1000.0, n_prompt);
    fprintf(stderr, "Prompt tokens (%d):", n_prompt);
    for (int i = 0; i < n_prompt; i++) fprintf(stderr, " %d", prompt_tokens[i]);
    fprintf(stderr, "\n");
    fflush(stderr);

    /* Exit here for tokenization benchmarking */
    if (getenv("PICOLM_EXIT_AFTER_TOKENIZE")) {
        model_free(&model);
        free(prompt_tokens);
        free(gen_buf);
        if (!use_qwen_tok) tokenizer_free(&tokenizer);
        qwen_tokenize_free(&qwen_enc);
        exit(0);
    }

    /* If cache covers part of the prompt, skip those positions */
    int start_pos = 0;
    if (cache_pos > 0 && cache_pos <= n_prompt) {
        start_pos = cache_pos;
        fprintf(stderr, "Skipping %d cached prompt tokens\n", start_pos);
    }

    fprintf(stderr, "Prompt: %d tokens, generating up to %d (temp=%.2f, top_p=%.2f, threads=%d)\n",
            n_prompt, max_tokens, (double)temperature, (double)top_p, num_threads);
    fprintf(stderr, "---\n");

#ifdef PICOLM_GPU
    /* PICOLM_GPU_PATH: 0=model_forward (CPU-hybrid), 1=model_forward_gpu (full GPU).
     * Controls decode path for both benchmark and interactive modes.
     * static: initialized once, shared across benchmark iterations. */
    static int gpu_path = -1;
    if (gpu_path < 0) {
        const char *gp = getenv("PICOLM_GPU_PATH");
        gpu_path = gp ? atoi(gp) : 1;
        if (gpu_path != 0 && gpu_path != 1) {
            fprintf(stderr, "WARNING: PICOLM_GPU_PATH=%d invalid, using 1\n", gpu_path);
            gpu_path = 1;
        }
    }
#endif

    /* ---- Benchmark mode ---- */
    if (benchmark_mode) {
        fprintf(stderr, "\nBenchmark mode (Ctrl-C to stop)\n");
        fprintf(stderr, "  Model: %.1f GB | SIMD: ", (double)model.mmap_size / (1024.0*1024.0*1024.0));
#if defined(PICOLM_AVX512)
        fprintf(stderr, "AVX-512");
#elif defined(PICOLM_AVX2)
        fprintf(stderr, "AVX2");
#elif defined(PICOLM_AVX)
        fprintf(stderr, "AVX");
#elif defined(PICOLM_SSE3)
        fprintf(stderr, "SSE3");
#elif defined(PICOLM_SSE2)
        fprintf(stderr, "SSE2");
#elif defined(PICOLM_NEON)
        fprintf(stderr, "NEON");
#elif defined(PICOLM_ALTIVEC)
        fprintf(stderr, "Altivec");
#else
        fprintf(stderr, "scalar");
#endif
        fprintf(stderr, " | Threads: %d\n", num_threads);
        fprintf(stderr, "  Context: %d | KV cache: %.0f MB\n",
                model.config.max_seq_len, (double)model.state.kv_size / (1024.0*1024.0));
        fprintf(stderr, "  Prompt: %d tokens | Generate: %d tokens\n\n", n_prompt, max_tokens);

        /* Set up benchmark context */
        bench_ctx_t bench;
        memset(&bench, 0, sizeof(bench));
        bench.n_layers = model.config.n_layers;
        bench.n_active_layers = model.config.n_layers - model.config.n_mtp_layers;
        bench.n_prompt_tokens = n_prompt;

        /* Pre-compute per-layer weight sizes */
        for (int l = 0; l < bench.n_layers; l++) {
            bench.layer_weight_sizes[l] = layer_weight_size(&model, l);
        }

        model_set_bench_callback(bench_callback, &bench);

        /* Benchmark runs until SIGINT/SIGTERM terminates the process.
         * On Linux, 'timeout' sends SIGTERM. Ctrl-C sends SIGINT. */

        int iteration = 0;
        int do_prefill = (n_prompt > 0);
        int do_generate = (max_tokens > 0);
        if (!do_prefill && !do_generate) {
            /* No prompt and no -n: just do prefill with BOS token */
            do_prefill = 1;
            n_prompt = 1;
            prompt_tokens[0] = (int)tokenizer.bos_id;
        }

        while (1) {
            iteration++;
            bench.iteration = iteration;
            bench.prefill_total_ms = 0;
            bench.gen_total_ms = 0;
            bench.prefill_layer_count = 0;
            bench.gen_layer_count = 0;
            bench.gen_tokens = 0;
            bench.layer_sum_ms = 0;
            bench.layer_total_count = 0;
            bench.layer_avg_ms = 0;
            bench.layer_hist_idx = 0;
            bench.layer_hist_count = 0;
            bench.total_minflt = 0;
            bench.total_majflt = 0;

            double t_iter_start = get_time_ms();

            /* Reset model state for fresh iteration */
            model_ssm_state_reset(&model);
#ifdef PICOLM_GPU
            if (model.gpu.kv_active) {
                model_ssm_state_reset_gpu(&model);
                picolm_gpu_kv_cache_clear(model.gpu.device);
            }
#endif
            /* Zero host KV cache (redundant when GPU active, but safe) */
            if (model.state.kv_block)
                memset(model.state.kv_block, 0, model.state.kv_size);

            int pos = 0;
            float *logits = NULL;

            /* Prefill phase */
            if (do_prefill) {
                bench.is_prefill = 1;
#ifdef PICOLM_GPU
                if (model.gpu.kv_active) {
                    logits = model_forward_prefill_gpu(&model, prompt_tokens, n_prompt, 0, NULL);
                    if (!logits) logits = model_forward_prefill(&model, prompt_tokens, n_prompt, 0, NULL);
                } else
#endif
                {
                    logits = model_forward_prefill(&model, prompt_tokens, n_prompt, 0, NULL);
                }
                pos = n_prompt - 1;
                /* Sync SSM state from host to device after CPU prefill */
#ifdef PICOLM_GPU
                if (model.gpu.kv_active && model.config.has_ssm) {
                    picolm_ssm_state_sync_to_device(&model, model.gpu.device);
                }
#endif
            } else {
                bench.is_prefill = 0;
                logits = model_forward(&model, tokenizer.bos_id, pos);
            }

            /* Generation phase */
            if (do_generate) {
                bench.is_prefill = 0;
                int token = 0, next = 0;
                for (int g = 0; g < max_tokens; g++) {
                    grammar_apply(&grammar, logits, model.config.vocab_size);
                    next = sampler_sample(&sampler, logits, model.config.vocab_size);
                    grammar_advance(&grammar, &tokenizer, next);
                    token = next;
                    bench.gen_tokens++;
                    pos++;
#ifdef PICOLM_GPU
                    if (model.gpu.kv_active && gpu_path == 1) {
                        logits = model_forward_gpu(&model, token, pos);
                        if (!logits) logits = model_forward(&model, token, pos);

                        /* PICOLM_DBG_PIPELINE: compare GPU pipeline vs CPU per token */
                        {
                            static int dbg_pipeline_active = 0;
                            static int dbg_pipeline_init = 0;
                            if (!dbg_pipeline_init++) {
                                const char *dbg = getenv("PICOLM_DBG_PIPELINE");
                                if (dbg && atoi(dbg)) {
                                    dbg_pipeline_active = 1;
                                    fprintf(stderr, "INFO: PICOLM_DBG_PIPELINE active\n");
                                }
                            }
                            if (dbg_pipeline_active) {
                                float *logits_cpu = model_forward(&model, token, pos);
                                if (logits_cpu) {
                                    float max_diff = 0.0f;
                                    for (int v = 0; v < model.config.vocab_size; v++) {
                                        float d = logits[v] - logits_cpu[v];
                                        if (d < 0) d = -d;
                                        if (d > max_diff) max_diff = d;
                                    }
                                    if (max_diff > 1e-2f) {
                                        fprintf(stderr, "[DBG_PIPELINE] pos=%d max_logit_diff=%.6f\n",
                                                pos, max_diff);
                                    }
                                }
                            }
                        }
                    } else
#endif
                    {
                        logits = model_forward(&model, token, pos);
                    }
                }
            }

            double t_iter_end = get_time_ms();
            double wall_sec = (t_iter_end - t_iter_start) / 1000.0;
            bench_summary(&bench, iteration, wall_sec);
        }

        /* Unreachable, but for compiler */
        model_set_bench_callback(NULL, NULL);
        free(prompt_tokens);
        free(gen_buf);
        free(prompt_buf);
        free(stdin_prompt);
        if (use_qwen_tok) qwen_tokenize_free(&qwen_enc);
        else tokenizer_free(&tokenizer);
        model_free(&model);
        tensor_threadpool_free();
        return 0;
    }

    /* Generation loop */
    int total_gen = 0;
    double t_start = get_time_ms();
    double t_first_token = 0;
    double t_end = 0;

    int pos = start_pos;
    int total_steps = n_prompt + max_tokens;
    if (total_steps > model.config.max_seq_len) {
        total_steps = model.config.max_seq_len;
    }

    /* Batch prefill: process all prompt tokens at once */
    float *logits = NULL;
    if (n_prompt > 0) {
        /* All models use model_forward_prefill.
         * SSM models: batched by default, use --ssm-serial for per-token path. */
        #ifdef PICOLM_GPU
        int gpu_prefill_used = 0;
#endif
#ifdef PICOLM_GPU
        if (model.gpu.kv_active) {
            logits = model_forward_prefill_gpu(&model, prompt_tokens, n_prompt, start_pos, NULL);
            if (logits) {
                gpu_prefill_used = 1;
            } else {
                logits = model_forward_prefill(&model, prompt_tokens, n_prompt, start_pos, NULL);
            }
        } else
#endif
        {
            logits = model_forward_prefill(&model, prompt_tokens, n_prompt, start_pos, NULL);
        }
        pos = start_pos + n_prompt - 1;
        /* Sync SSM state from host to device after CPU prefill only.
         * After GPU prefill, the device already has correct state and
         * syncing stale CPU state would overwrite it. */
#ifdef PICOLM_GPU
        if (model.gpu.kv_active && model.config.has_ssm && !gpu_prefill_used) {
            picolm_ssm_state_sync_to_device(&model, model.gpu.device);
        }
#endif
    } else {
        /* No prompt: just generate from BOS */
        logits = model_forward(&model, tokenizer.bos_id, pos);
    }

    int token = 0;
    int next = 0;
    for (; pos < total_steps; pos++) {
        /* Generation: apply grammar constraints, then sample */
        if (t_first_token == 0) {
            t_first_token = get_time_ms();
        }

                grammar_apply(&grammar, logits, model.config.vocab_size);
        next = sampler_sample(&sampler, logits, model.config.vocab_size);
        fprintf(stderr, "[GEN] pos=%d next=%d eos=%d\n", pos, next, (int)tokenizer.eos_id);

        /* Update grammar state with the generated token */
        grammar_advance(&grammar, &tokenizer, next);

        /* Decode and print */
        static char qwen_decode_buf[64];
        const char *decode_str;
        if (use_qwen_tok) {
            qwen_tokenize_decode(&qwen_enc, next, qwen_decode_buf, sizeof(qwen_decode_buf));
            decode_str = qwen_decode_buf;
        } else {
            decode_str = tokenizer_decode(&tokenizer, token, next);
        }
        printf("%s", decode_str);
        fflush(stdout);
        /* Also capture in buffer */
        { int plen = (int)strlen(decode_str);
          if (gen_buf_len + plen < max_tokens * 16 - 1) {
              memcpy(gen_buf + gen_buf_len, decode_str, plen);
              gen_buf_len += plen;
          }
        }

        total_gen++;

        /* Stop on EOS or grammar completion */
        if (next == (int)tokenizer.eos_id) break;
        if (grammar_is_complete(&grammar)) break;

        token = next;
#ifdef PICOLM_GPU
        static int dbg_pipeline_active = 0, dbg_pipeline_init = 0;
        if (!dbg_pipeline_init++) {
            const char *dbg = getenv("PICOLM_DBG_PIPELINE");
            if (dbg && atoi(dbg)) {
                dbg_pipeline_active = 1;
                fprintf(stderr, "INFO: PICOLM_DBG_PIPELINE active (default loop)\n");
            }
        }
        if (model.gpu.kv_active && gpu_path == 1) {
            logits = model_forward_gpu(&model, token, pos + 1);
            if (!logits) logits = model_forward(&model, token, pos + 1);
            /* PICOLM_DBG_PIPELINE: compare GPU pipeline vs CPU at first token only
             * (running CPU path for all tokens corrupts the KV cache shared with GPU) */
            if (dbg_pipeline_active && logits && pos == n_prompt) {
                float *logits_cpu = model_forward(&model, token, pos + 1);
                if (logits_cpu) {
                    float max_diff = 0.0f;
                    for (int v = 0; v < model.config.vocab_size; v++) {
                        float d = logits[v] - logits_cpu[v];
                        if (d < 0) d = -d;
                        if (d > max_diff) max_diff = d;
                    }
                    fprintf(stderr, "[DBG_PIPELINE] pos=%d max_logit_diff=%.6f\n",
                            pos + 1, max_diff);
                }
            }
        } else
#endif
        {
            logits = model_forward(&model, token, pos + 1);
        }
    }

    printf("\n");
    t_end = get_time_ms();

    /* Save KV cache if requested */
    if (cache_file && n_prompt > 0) {
        kvcache_save(&model, cache_file, n_prompt, prompt_tokens);
    }
    free(cache_tokens);

    double total_time = (t_end - t_start) / 1000.0;
    if (t_first_token == 0) t_first_token = t_end;
    double gen_time = (t_end - t_first_token) / 1000.0;
    double prefill_time = (t_first_token - t_start) / 1000.0;
    int actual_prefill = n_prompt - start_pos;
    if (actual_prefill < 0) actual_prefill = 0;

    fprintf(stderr, "---\n");
    fprintf(stderr, "Prefill: %d tokens in %.2fs (%.1f tok/s)%s\n",
            actual_prefill, prefill_time,
            prefill_time > 0 ? (double)actual_prefill / prefill_time : 0,
            start_pos > 0 ? " [partially cached]" : "");
    fprintf(stderr, "Generation: %d tokens in %.2fs (%.1f tok/s)\n",
            total_gen, gen_time,
            gen_time > 0 ? (double)total_gen / gen_time : 0);
    fprintf(stderr, "Total: %.2fs\n", total_time);
    gen_buf[gen_buf_len] = '\0';
    fprintf(stderr, "OUTPUT: %s\n", gen_buf);
    free(gen_buf);
    {
        const char *kname = "f16";
        if (model.state.kv_type_k == KV_CACHE_Q8_0) kname = "q8_0";
        else if (model.state.kv_type_k == KV_CACHE_Q4_0) kname = "q4_0";
        else if (model.state.kv_type_k == KV_CACHE_TQ3) kname = "tq3";
        else if (model.state.kv_type_k == KV_CACHE_TQ4) kname = "tq4";
        const char *vname = "f16";
        if (model.state.kv_type_v == KV_CACHE_Q8_0) vname = "q8_0";
        else if (model.state.kv_type_v == KV_CACHE_Q4_0) vname = "q4_0";
        else if (model.state.kv_type_v == KV_CACHE_TQ3) vname = "tq3";
        else if (model.state.kv_type_v == KV_CACHE_TQ4) vname = "tq4";
        fprintf(stderr, "Memory: %.2f MB runtime state (%s/%s KV cache)\n",
                (double)model.state.mem_size / (1024.0 * 1024.0), kname, vname);
    }

    /* Cleanup */
    if (model.locked_layers > 0)
        model_unlock_layers(&model);
    grammar_free(&grammar);
    free(prompt_tokens);
    free(prompt_buf);
    free(stdin_prompt);
    if (use_qwen_tok) qwen_tokenize_free(&qwen_enc);
    else tokenizer_free(&tokenizer);
#ifdef PICOLM_VIZ
    if (viz_mode) viz_free();
#endif
    model_free(&model);
    tensor_threadpool_free();

    return 0;
}
