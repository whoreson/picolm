#include "model.h"
#include "tensor.h"
#include "tokenizer.h"
#include "sampler.h"
#include "grammar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <sys/time.h>
#include <unistd.h>
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}
#endif

static void usage(const char *prog) {
    fprintf(stderr, "PicoLLM — ultra-lightweight LLM inference engine\n\n");
    fprintf(stderr, "Usage: %s <model.gguf> [options]\n", prog);
    fprintf(stderr, "\nGeneration options:\n");
    fprintf(stderr, "  -p <prompt>    Input prompt (or pipe via stdin)\n");
    fprintf(stderr, "  -n <int>       Max tokens to generate (default: 256)\n");
    fprintf(stderr, "  -t <float>     Temperature (default: 0.8, 0=greedy)\n");
    fprintf(stderr, "  -k <float>     Top-p / nucleus sampling (default: 0.9)\n");
    fprintf(stderr, "  -s <int>       RNG seed (default: 42)\n");
    fprintf(stderr, "  -c <int>       Context length override\n");
    fprintf(stderr, "  -j <int>       Number of threads (default: 4)\n");
    fprintf(stderr, "\nAdvanced options:\n");
    fprintf(stderr, "  --json         Grammar-constrained JSON output mode\n");
    fprintf(stderr, "  --cache <file> KV cache file (saves/loads prompt state)\n");
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

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = NULL;
    int    max_tokens = 256;
    float  temperature = 0.8f;
    float  top_p = 0.9f;
    uint64_t seed = 42;
    int    context_override = 0;
    int    num_threads = 4;
    int    json_mode = 0;
    const char *cache_file = NULL;

    /* Parse arguments */
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            context_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            json_mode = 1;
        } else if (strcmp(argv[i], "--cache") == 0 && i + 1 < argc) {
            cache_file = argv[++i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
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
#else
        if (!isatty(fileno(stdin))) {
            stdin_prompt = read_stdin();
            prompt = stdin_prompt;
        }
#endif
    }

    if (!prompt || !*prompt) {
        fprintf(stderr, "No prompt provided. Use -p or pipe via stdin.\n");
        usage(argv[0]);
        return 1;
    }

    /* Load model */
    fprintf(stderr, "Loading model: %s\n", model_path);
    fprintf(stderr, "SIMD: %s\n",
#if defined(PICOLM_AVX2)
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
    model_t model;
    if (model_load(&model, model_path, context_override) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    tensor_set_threads(num_threads);

    /* Load tokenizer */
    tokenizer_t tokenizer;
    if (tokenizer_load(&tokenizer, &model) != 0) {
        fprintf(stderr, "Failed to load tokenizer\n");
        model_free(&model);
        return 1;
    }

    /* Init sampler */
    sampler_t sampler;
    sampler_init(&sampler, temperature, top_p, seed);

    /* Init grammar constraint */
    grammar_state_t grammar;
    grammar_init(&grammar, json_mode ? GRAMMAR_JSON : GRAMMAR_NONE, &tokenizer);
    if (json_mode) {
        fprintf(stderr, "JSON grammar mode enabled\n");
    }

    /* Try to load KV cache (skip prefill for cached prompt) */
    int cache_pos = 0;
    if (cache_file) {
        cache_pos = kvcache_load(&model, cache_file);
    }

    /* Encode prompt */
    int max_prompt_tokens = (int)strlen(prompt) + 3;
    int *prompt_tokens = (int *)malloc((size_t)max_prompt_tokens * sizeof(int));
    int n_prompt = tokenizer_encode(&tokenizer, prompt, prompt_tokens, max_prompt_tokens, 1);

    /* If cache covers part of the prompt, skip those positions */
    int start_pos = 0;
    if (cache_pos > 0 && cache_pos <= n_prompt) {
        start_pos = cache_pos;
        fprintf(stderr, "Skipping %d cached prompt tokens\n", start_pos);
    }

    fprintf(stderr, "Prompt: %d tokens, generating up to %d (temp=%.2f, top_p=%.2f, threads=%d)\n",
            n_prompt, max_tokens, temperature, top_p, num_threads);
    fprintf(stderr, "---\n");

    /* Generation loop */
    int total_gen = 0;
    double t_start = get_time_ms();
    double t_first_token = 0;

    int token = prompt_tokens[start_pos > 0 ? start_pos - 1 : 0];
    int pos = start_pos > 0 ? start_pos - 1 : 0;
    int total_steps = n_prompt + max_tokens;
    if (total_steps > model.config.max_seq_len) {
        total_steps = model.config.max_seq_len;
    }

    for (; pos < total_steps; pos++) {
        /* Determine which token to feed */
        if (pos < start_pos) {
            /* This shouldn't happen given our start logic, but safety */
            token = prompt_tokens[pos];
            continue;
        }

        /* Forward pass */
        float *logits = model_forward(&model, token, pos);

        int next;
        if (pos < n_prompt - 1) {
            /* Prefill: use next prompt token */
            next = prompt_tokens[pos + 1];
        } else {
            /* Generation: apply grammar constraints, then sample */
            if (pos == n_prompt - 1) {
                t_first_token = get_time_ms();
            }

            grammar_apply(&grammar, logits, model.config.vocab_size);
            next = sampler_sample(&sampler, logits, model.config.vocab_size);

            /* Update grammar state with the generated token */
            grammar_advance(&grammar, &tokenizer, next);

            /* Decode and print */
            const char *piece = tokenizer_decode(&tokenizer, token, next);
            printf("%s", piece);
            fflush(stdout);

            total_gen++;

            /* Stop on EOS or grammar completion */
            if (next == (int)tokenizer.eos_id) break;
            if (grammar_is_complete(&grammar)) break;
        }

        token = next;
    }

    printf("\n");
    double t_end = get_time_ms();

    /* Save KV cache if requested (save the full prompt state) */
    if (cache_file && n_prompt > 0) {
        kvcache_save(&model, cache_file, n_prompt);
    }

    /* Stats */
    double total_time = (t_end - t_start) / 1000.0;
    if (t_first_token == 0) t_first_token = t_end; /* no generation happened */
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
    fprintf(stderr, "Memory: %.2f MB runtime state (FP16 KV cache)\n",
            (double)model.state.mem_size / (1024.0 * 1024.0));

    /* Cleanup */
    grammar_free(&grammar);
    free(prompt_tokens);
    free(stdin_prompt);
    tokenizer_free(&tokenizer);
    model_free(&model);

    return 0;
}
