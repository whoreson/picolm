#include "model.h"
#include "tensor.h"
#include "tokenizer.h"
#include "sampler.h"
#include "grammar.h"
#include "qwen_tokenize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}
#endif

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
    fprintf(stderr, "  -j <int>       Number of threads (default: 4)\n");
    fprintf(stderr, "  --mem <MB>      Pin this many MB of layers in RAM (mlock)\n");
    fprintf(stderr, "  --prefault      Prefault all model pages into RAM at load time\n");
    fprintf(stderr, "\nServer options:\n");
    fprintf(stderr, "  --server <model> Start HTTP server (OpenAI-compatible)\n");
    fprintf(stderr, "  --port <int>     Server port (default: 8080)\n");
    fprintf(stderr, "  --host <addr>    Server bind address (default: 0.0.0.0)\n");
    fprintf(stderr, "\nAdvanced options:\n");
    fprintf(stderr, "  --json         Grammar-constrained JSON output mode\n");
    fprintf(stderr, "  --cache <file> KV cache file (saves/loads prompt state)\n");
    fprintf(stderr, "  -ctk <type>    Key cache type: f16, q8_0, q4_0 (default: f16)\n");
    fprintf(stderr, "  -ctv <type>    Val cache type: f16, q8_0, q4_0 (default: f16)\n");
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
    kv_cache_type_t kv_type_k = KV_CACHE_F16;
    kv_cache_type_t kv_type_v = KV_CACHE_F16;
    int    mem_mb = 0;      /* --mem budget in megabytes (0=disabled) */
    int    do_prefault = 0; /* --prefault (touch all mmap pages at load time) */
    int    server_daemon = 0;
    int    server_mode = 0;
    int    server_port = 8080;
    char   server_host[256] = "0.0.0.0";

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        /* Skip positional model path (argv[1] when it doesn't start with -) */
        if (i == 1 && argv[1][0] != '-') continue;
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
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--daemon") == 0) {
            server_daemon = 1;
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
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            context_override = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-ctk") == 0 || strcmp(argv[i], "-ctv") == 0) && i + 1 < argc) {
            const char *typestr = argv[++i];
            kv_cache_type_t *tgt = (strcmp(argv[i-1], "-ctk") == 0) ? &kv_type_k : &kv_type_v;
            if (strcmp(typestr, "q8_0") == 0) *tgt = KV_CACHE_Q8_0;
            else if (strcmp(typestr, "q4_0") == 0) *tgt = KV_CACHE_Q4_0;
            else if (strcmp(typestr, "f16") == 0) *tgt = KV_CACHE_F16;
            else { fprintf(stderr, "Unknown KV cache type: %s (use f16, q8_0, q4_0)\n", typestr); return 1; }
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

    /* Server mode: start HTTP server (no prompt needed) */
    if (server_mode) {
        /* Daemonize if requested (Unix-only) */
#ifndef _WIN32
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
        extern int server_main(int port, const char *host, const char *model_path, int num_threads, int do_prefault, int context_override, int mem_mb);
        return server_main(server_port, server_host, model_path, num_threads, do_prefault, context_override, mem_mb);
    }

    if (!prompt) {
        fprintf(stderr, "No prompt provided. Use -p or pipe via stdin.\n");
        usage(argv[0]);
        return 1;
    }

    /* Load model */
    fprintf(stderr, "Loading model: %s\n", model_path);
    fprintf(stderr, "SIMD: %s\n",
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
    model_t model;
    /* Initialize FP16->FP32 lookup table (64KB) for fast attention */
    fp16_table_init();

    /* Auto-detect: if path is a directory with safetensors files, use safetensors loader */
    /* Enable prefaulting before model load (affects mmap_file in model_load) */
    model_set_prefault(do_prefault);

    int is_safetensors = 0;
#ifndef _WIN32
    {
        char idx_path[4096];
        snprintf(idx_path, sizeof(idx_path), "%s/model.safetensors.index.json", model_path);
        if (access(idx_path, F_OK) == 0) is_safetensors = 1;
        if (!is_safetensors) {
            snprintf(idx_path, sizeof(idx_path), "%s/model.safetensors", model_path);
            if (access(idx_path, F_OK) == 0) is_safetensors = 1;
        }
    }
#else
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
    int load_ok = 0;
    if (is_safetensors) {
        load_ok = (model_load_safetensors(&model, model_path, context_override, kv_type_k, kv_type_v) == 0);
    } else {
        load_ok = (model_load(&model, model_path, context_override, kv_type_k, kv_type_v) == 0);
    }
    if (!load_ok) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

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

    /* Debug: print weight types */
    fprintf(stderr, "Weight type: %d, Emb type: %d, Output type: %d\n",
        model.config.weight_type, model.weights.type_token_embd, model.weights.type_output);
    if (model.config.n_layers > 0) {
        layer_weights_t *lw0 = &model.weights.layers[0];
        fprintf(stderr, "Layer0: attn_q=%d k=%d v=%d gate=%d up=%d down=%d\n",
            lw0->type_attn_q, lw0->type_attn_k, lw0->type_attn_v,
            lw0->type_ffn_gate, lw0->type_ffn_up, lw0->type_ffn_down);
    }

    /* Load tokenizer */
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

    /* Init sampler */
    sampler_t sampler;
    sampler_init(&sampler, temperature, top_p, seed);

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
    if (cache_file) {
        cache_pos = kvcache_load(&model, cache_file);
    }

    /* Encode prompt */
    int max_prompt_tokens = (int)strlen(prompt) + 16;
    int *prompt_tokens = (int *)malloc((size_t)max_prompt_tokens * sizeof(int));
    int n_prompt;
    if (use_qwen_tok) {
        /* Qwen3.5 GGUF has no bos_token_id, so tok_bos_id=11 is just the gpt2 fallback.
           Don't prepend it - llama.cpp doesn't either (add_bos=false for qwen35 pretype).
           Qwen3.6 has bos_token_id=248044 from GGUF, so prepend that. */
        if (tokenizer.bos_id != 11) {
            n_prompt = qwen_tokenize_encode(&qwen_enc, prompt, prompt_tokens + 1, max_prompt_tokens - 1);
            prompt_tokens[0] = (int)tokenizer.bos_id;
            n_prompt++;
        } else {
            n_prompt = qwen_tokenize_encode(&qwen_enc, prompt, prompt_tokens, max_prompt_tokens);
        }
    } else {
        n_prompt = tokenizer_encode(&tokenizer, prompt, prompt_tokens, max_prompt_tokens, 1);
    }
    fprintf(stderr, "Prompt tokens (%d):", n_prompt);
    for (int i = 0; i < n_prompt; i++) fprintf(stderr, " %d", prompt_tokens[i]);
    fprintf(stderr, "\n");

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
    double t_end = 0;

    int pos = start_pos;
    int total_steps = n_prompt + max_tokens;
    if (total_steps > model.config.max_seq_len) {
        total_steps = model.config.max_seq_len;
    }

    /* Batch prefill: process all prompt tokens at once */
    float *logits = NULL;
    if (n_prompt > 0 && !model.config.has_ssm) {
        /* Attention-only models: full batched prefill */
        logits = model_forward_prefill(&model, prompt_tokens, n_prompt, start_pos);
        pos = start_pos + n_prompt - 1;
    } else if (n_prompt > 0) {
        /* SSM models (Qwen3.5/3.6): per-token prefill (SSM is stateful) */
        int token = start_pos > 0 ? prompt_tokens[start_pos - 1] : prompt_tokens[0];
        for (int p = start_pos; p < start_pos + n_prompt; p++) {
            logits = model_forward(&model, token, p);
            token = prompt_tokens[p + 1 - start_pos];
        }
        pos = start_pos + n_prompt;
    } else {
        /* No prompt: just generate from BOS */
        logits = model_forward(&model, tokenizer.bos_id, pos);
    }

    int token = 0;
    int next = 0;
    for (; pos < total_steps; pos++) {
        /* Generation: apply grammar constraints, then sample */
        if (pos == n_prompt - 1) {
            t_first_token = get_time_ms();
        }

        grammar_apply(&grammar, logits, model.config.vocab_size);
        next = sampler_sample(&sampler, logits, model.config.vocab_size);

        /* Update grammar state with the generated token */
        grammar_advance(&grammar, &tokenizer, next);

        /* Decode and print */
        static char qwen_decode_buf[64];
        char *decode_str;
        if (use_qwen_tok) {
            qwen_tokenize_decode(&qwen_enc, next, qwen_decode_buf, sizeof(qwen_decode_buf));
            decode_str = qwen_decode_buf;
        } else {
            decode_str = (char *)tokenizer_decode(&tokenizer, token, next);
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
        logits = model_forward(&model, token, pos + 1);
    }

    printf("\n");
    t_end = get_time_ms();

    /* Save KV cache if requested */
    if (cache_file && n_prompt > 0) {
        kvcache_save(&model, cache_file, n_prompt);
    }

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
        if (model.state.kv_type_k == KV_CACHE_Q4_0) kname = "q4_0";
        const char *vname = "f16";
        if (model.state.kv_type_v == KV_CACHE_Q8_0) vname = "q8_0";
        if (model.state.kv_type_v == KV_CACHE_Q4_0) vname = "q4_0";
        fprintf(stderr, "Memory: %.2f MB runtime state (%s/%s KV cache)\n",
                (double)model.state.mem_size / (1024.0 * 1024.0), kname, vname);
    }

    /* Cleanup */
    if (model.locked_layers > 0)
        model_unlock_layers(&model);
    grammar_free(&grammar);
    free(prompt_tokens);
    free(stdin_prompt);
    if (use_qwen_tok) qwen_tokenize_free(&qwen_enc);
    else tokenizer_free(&tokenizer);
    model_free(&model);
    tensor_threadpool_free();

    return 0;
}
