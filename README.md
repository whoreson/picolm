<p align="center">
  <img src="https://img.shields.io/badge/Language-C11-blue?style=flat-square" alt="C11">
  <img src="https://img.shields.io/badge/Binary_Size-~80KB-brightgreen?style=flat-square" alt="Binary Size">
  <img src="https://img.shields.io/badge/Runtime_RAM-45MB-orange?style=flat-square" alt="RAM">
  <img src="https://img.shields.io/badge/Dependencies-Zero-success?style=flat-square" alt="Zero Dependencies">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/badge/HTTP_Server-OpenAI_API-red?style=flat-square" alt="HTTP Server">
  <img src="https://img.shields.io/badge/Models-Qwen3%2CSmolLM%2CLlama-blue?style=flat-square" alt="Models">
</p>

<h1 align="center">PicoLM - But It's Actually Useful</h1>

## What's New

**v1.0-beta3** (upcoming)

- GPU architecture rewrite: device-native pipeline replacing the old D2H/H2D matmul crutch
- IMMA Tensor Core kernels for all quant types: Q8_0, Q6_K, Q5_K, Q4_K, Q4_0, Q4_1, Q3_K, Q2_K, Q1_0
- FlashAttention-2 tensor core prefill kernel, warp-group scalar attention (new default), shared-memory staged IMMA W16 (+17% small ctx)
- Split-K decode attention (flash-decoding pattern), GPU-resident output projection
- GPU-resident KV cache, device-native RMSNorm, RoPE, elementwise ops
- Device-native SSM batched kernels: conv1d, l2norm, vecdot, gated_norm, head_permute, expert_mlp
- GPU-pipelined prefill path (`model_forward_prefill_gpu`), fused RMSNorm+Quantize+QKV IMMA, two-tier batching
- FFN rewrite: Q8_0 tiled matmul, 2.6x faster on GPU
- GPT-2 architecture support with batched prefill and native tokenizer
- GGUF split-file loader
- Gemma-3n architecture support (WIP, doesn't work yet)
- TurboQuant TQ3/TQ4 KV cache
- Deterministic sign randomization for Walsh-Hadamard transform
- `sgemm` tiled GEMM engine for F16/F32 matmuls
- NEON acceleration for Q4_0, Q4_K, Q5_K, Q6_K, Q3_K, Q2_0, Q1_0 vec_dot paths
- AVX2/AVX/SSSE3 SIMD paths for Q2_K and Q4_K (Q4_K SSE3: 3.3x vs scalar)
- `--benchmark-ctx` context scaling benchmark, `--gpu-diff` kernel diff test
- `-pf` option (auto-apply chat template), Big.LITTLE awareness, `--list-kv`, `PICOLM_GPU_PATH` env var

## What all this is

The original PicoLM was a clever proof-of-concept: mmap a GGUF, stream layers through 45MB of RAM, call it a day. Noble effort, but it was missing basically everything you need to actually use an LLM. So [I](http://gabucino.hu) went in and added hundreds of commits across quantization acceleration, model support, HTTP server, GPU backends, and cross-platform fixes.

Since then, progress has been steady. The main goals are (beyond satisfying
curiosity on what Qwen-3.6-27B can and can't do):

- **First, portability.** The only good software is one that runs (and is tested) on everything from the last 50 years. PicoLM runs from 32-bit MS-DOS through Raspberry Pi, MIPS/OpenWRT, AMD ROCm (MI50 tested), Metal (if someone bothers), to RTX 4090 or DGX Spark.
- **Second, speed.** Compete with llama.cpp in as many areas as possible. 100% emphasis on CPU and GPU optimizations. Reaching ik_llama.cpp speeds is impossible, of course, courtesy of 40k lines of matmul engines. Work needs to be done on prefill (prompt processing). As far as token generation is concerned, it's already there, and in some cases outpaces llama.cpp.
- **Third: model support.** Slopped and benchmaxxed 2023+ models aren't priorities, except Qwen 3.6-27B which is a first-tier model that PicoLM supports (both CPU and GPU, between Q1_0 and f16). Exceptions are made for small models like Gemma-3/4n and sorts, fitting the project profile. Fimbulvetr (SOLAR-10.7b finetune) is also tier 1, and Miqu-70B is known to work. Original Mistral Nemo is planned (needs tokenizer support), and DeepSeek V3-0325 is also on TODO, later versions not at all. MoE and SSM architecture in place, tested with Qwen 3.6-31B-A3B (which is not a good model btw but whatever).

PicoLM doesn't preload weights in RAM, therefore doesn't have a hard RAM requirement like llama.cpp. This has the disadvantage of not being able to reorder tensors to col-major, but this is the niche we're in. Support for transposing tensors on-disk (breaking file compatibility) is on the table. The `--prefault` option preloads all weights for faster first reply, when enough RAM is available.

In low-memory environments, to avoid weights needlessly being rotated round-robin to/from the RAM by the OS, PicoLM has a `--mem` option which does a `mlock()`/`VirtualLock()` on the most important weights (explicit MoE support too), keeping them in the RAM at all times, while only the rest is being streamed. Very measurable speed increase.

Per-layer activation heatmap, viewable over the built-in VNC server, with no speed loss. Looks cool. Also can enable/disable/reorder layers with the pointer, though the Basilisk probably won't be very approving of this. Think about the future, Eckhardt! More visualizations planned, even console ANSI.

**What was added since the upstream baseline:**

| Category | Details |
|----------|---------|
| **Quantization** | Q8_0, Q4_K, Q4_0, Q4_1, Q2_K, Q3_K, Q5_K, Q6_K, Q8_K, Q4_0_8_8, Q4_0_4_8, Q4_0_4_4, Q1_0, Q2_0, F16, BF16, F32 |
| **SIMD: x86** | AVX-512/VNNI/AVX2/AVX/SSSE3/SSE (fp16x16, vec_dot, rmsnorm, RoPE, attention, all quants, SSM, sgemm) |
| **SIMD: ARM** | NEON (RoPE, FP16 HW convert, Q8_0/Q4_0/Q4_K/Q5_K/Q6_K/Q3_K/Q2_0/Q1_0 vec_dot, attention, SSM kernels), DotProd, i8mm (Q4_0_4_4, Q4_0_4_8, ARMv8.2+ SDOT) |
| **GPU** | CUDA/HIP IMMA Tensor Core GEMM (m16n8k32, W16 shared-memory staged), FA2 prefill, warp-group scalar attention, Split-K decode, device-native SSM/attention/RMSNorm/RoPE, all quants Q1_0..Q8_0, Metal SSM kernels |
| **Models** | Llama 1/2/3, GPT-2, Qwen2, Qwen3 (non-uniform head_dim), Qwen3.5/3.6 (SSM/Mamba hybrid), Gemma-3n (AltUp/Laurel) |
| **Scale** | Tested up to 70B parameters (miqu-70b) |
| **HTTP server** | OpenAI API (`/v1/completions`, `/v1/chat/completions`, `/v1/models`), llama.cpp-compatible (`/completion`, `/props`), `/tokenize`, `/detokenize`, `/health`, streaming, persistent model/KV cache |
| **Threading** | Persistent thread pool (gen-counter barrier), physical core auto-detect, big.LITTLE awareness, GQA grouped attention, tiled/blocked attention for prefill, batched GEMV |
| **Platform** | Linux (AVX-512/VNNI, ARM NEON, RISC-V), Windows 7-11 (SRWLOCK, MinGW, VirtualLock, `--mem`), Mac OS/X 10.4-10.6, MS-DOS, PPC/Altivec, Android, FreeBSD |
| **KV cache** | F16, Q8_0, Q4_0, TQ4, TQ3, Walsh-Hadamard rotation (`-khad`/`-vhad`), persistent with prefix matching |
| **Misc** | 64KB FP16 lookup table, `--mem` mlock, `--prefault`, `--daemon`, `-pf` chat template, `--json` grammar, VNC visualization server, GGUF split-file loader, `sgemm` tiled GEMM |

**What it will never have**

Support for fucking Jinja chat templates. Raw text-completion only, write your own prompts or fuck the fuck off. Maybe an external Python proxy/shim.

---

```
                    +--------------------------------------------------+
   What goes        |         45 MB Runtime RAM                        |
   in RAM           |  +-----------+ +------------+ +---------------+  |
                    |  | Buffers   | | FP16 KV    | | Tokenizer     |  |
                    |  |  1.2 MB   | | Cache      | |   4.5 MB      |  |
                    |  |           | |  ~40 MB    | |               |  |
                    |  +-----------+ +------------+ +---------------+  |
                    +--------------------------------------------------+

                    +--------------------------------------------------+
   What stays       |             Model on Disk                        |
   on disk          |       (mmap - OS pages in layers                 |
   (via mmap)       |        as needed, ~1 at a time)                  |
                    +--------------------------------------------------+
```

## Past Releases

### v1.0-beta2

    KV cache v4 with token sequences and streamed SSM I/O, Q2_K/Q6_K/Q3_K
    SIMD acceleration, Metal SSM kernels, Walsh-Hadamard KV cache rotation,
    and embedded VNC visualization server.

    132 commits since beta1.

    KV Cache v4: Persistence with Token Sequences and SSM
      Slot files now embed the token sequence alongside KV cache data,
        enabling full state save/restore without re-tokenizing the prompt
      Streamed SSM I/O: SSM state checkpoints saved/restored within slot files
      Server-side slot management with per-request isolation
      KV layer mapping for mixed SSM/attention models (ordinal tracking)
      kvcache_load() returns malloc'd token array for caller to free

    Quantization: Q2_K, Q6_K, Q3_K SIMD Fast Paths
      Q2_K: vec_dot_q2_K_q8_K with NEON SIMD (vmull_s8 + vpaddlq_s16)
        RPi 4: 2-5x speedup on cached layers, 200ms -> 109-170ms/layer
      Q6_K: vec_dot_q6_K_q8_K with AVX2/AVX/AVX-512 SIMD paths
        27B gen: 2.3 tok/s (was 0.3, now ~90% of Q8_0)
      Q3_K: vec_dot_q3_K_q8_K ported from llama.cpp (AVX2/AVX/scalar)
        Q3_K_S 27B gen: 2.7 tok/s (was 0.6, 4.5x speedup)
      Q5_K: vec_dot_q5_K_q8_K added, scale field bug fixed (dm vs d)
      Q4_0: GGUF nibble layout corrected (first/second-half pairing)
      Q4_0_8_8: parallelized batched matmul, closing 2x prefill gap on AVX2

    KV Cache: Walsh-Hadamard Rotation and Layout Refactor
      Walsh-Hadamard transform before quantization (-khad/-vhad flags)
      Reduces outlier impact on Q4_0 block scales, dot products preserved
      Q8_0 and Q4_0 KV cache quantization fixed (-ctk/-ctv options)
      GQA full-row KV cache layout for quantized types

    SSM: Correctness, Speed, GPU Support
      Batched SSM prefill is now the default (was --ssm-batched)
      --ssm-chunk-size N: configurable chunk size (was compile-time CS=64)
      AVX2, AVX1/FMA, and NEON SSM kernels for per-token + chunked recurrence
      Critical fix: veff/sq transpose bug in all SIMD SSM kernels
      Critical fix: dot product double-count in SSM kernels
      Qwen tokenizer fix: symbol runs kept together (backticks)

    Metal Backend: SSM Kernels and Performance
      SSM vecdot (alpha/beta) + DeltaNet recurrence kernels on GPU
      Multi-output GEMV for Q4_K/Q6_K matmuls
      FFN fused into single GPU command buffer
      GPU profiling (PICOLM_GPU_PROFILE=1)

    CUDA/HIP: Infrastructure and Fixes
      Q8_0xQ8_0 GPU matmul matching CPU path exactly
      F16/BF16 GPU support, Q4_K device dequant
      ROCm: auto-detect path/arch, guard WMMA for < 6.4
      Windows CUDA build target (make hunger)

    Server: Features and Critical Bug Fixes
      Stop word support on /v1/completions (two-phase: full match + partial prefix)
      Prefill cancellation on client disconnect (all endpoints, all layer types)
      KV cache corruption when prompt fully cached (2nd identical request)
      SSE [DONE] unterminated message, dangling model_name pointer (UAF)
      Final timing chunk missing content field (llama.vim crash)
      max_tokens accepted as alias for n_predict on /completion

    Build & Platform
      Auto-detect SIMD level, enable AVX2/AVX512 by default
      Makefile SIMD detection fixed for mingw
      -mf16c conditional on actual CPU F16C support
      SSSE3, AVX1 compatibility fixes (sumf declaration, fmadd fusion)

    VNC Visualization Server
      Live activation heatmap via embedded RFB 3.3 VNC protocol
      X axis = token index (time), Y axis = layer index
      Per-layer bucket-RMS of residual stream, normalized with adaptive color
      Dynamic layer skipping via mouse click, drag-and-drop layer permutation
      Wired into server mode; zero-cost when no client connected

    Tokenizer & Correctness
      Unescape all whitespace markers (U+2581, U+0100) in tokenizer_decode()
      V-cache OOB in tiled attention (GQA ratio 8, 513+ prompt tokens)
      fp16 SIMD tail overflow in tiled attention (ASan-catchable)
      Q2_K NEON double-consumption segfault fix

    Developer Tooling
      --benchmark mode: per-layer timing, ETA, and I/O stats
      --list-tensors works before model path, Q1_0/Q2_0 type names fixed

    Incoming Soon
      GBNF grammar support: full PEG parser replacing the JSON-only grammar.
        Arbitrary .gbnf files via --grammar flag, Unicode character classes,
        server-side grammar constraints on /v1/completions. ~1000-1500 lines.
      RoPE scaling: linear, YaRN, and longrope methods. Pre-computed cos/sin
        tables modified per the GGUF rope.scaling metadata (freq_scale,
        ext_factor, attn_factor, beta_fast/slow). Enables extended context.
      Gemma 3 Next (gemma3n): AltUp (alternate embeddings) and Laurel
        (low-rank attention enrichment) architecture. Per-layer token
        embeddings, sliding window attention patterns, Q/K norm, FFN post-norm.
      Refactored GPU acceleration

### v1.0-beta1

    SSM state checkpointing for Mamba-based models, Qwen3.6 special token
    handling, and the /slots monitoring endpoint.

    Ten commits since alpha2, adding support for state checkpointing in hybrid
    SSM/Transformer models, fixing Qwen3.6 tokenizer special token visibility,
    and porting to big-endian and legacy platforms.

    SSM State Checkpointing (Qwen3.5/3.6, Mamba models)
      Model-level API: model_ssm_snapshot_size(), model_ssm_state_save(),
        model_ssm_state_restore(), model_ssm_state_reset() in model.h/model.c.
        Captures and restores the full SSM state vector at any KV cache position.
      Server-side checkpoint management:
        Interval-based checkpoints during prefill (--checkpoint-every-nt, default 256)
        Interval-based checkpoints during generation (--checkpoint-every-nt-gen, default 64)
        Tail checkpoint (--checkpoint-tail-offset, default 5)
        End-of-prompt checkpoint (always, when enabled)
        Variance-based eviction when at capacity (minimizes gap variance)
      Integrated into prefix-match logic: on partial KV cache reuse, restores
        the closest SSM checkpoint and reprocesses the delta tokens. Without
        this, SSM models produce garbled output after cache reuse.
      CLI options: --checkpoint-max, --checkpoint-every-nt, --checkpoint-every-nt-gen,
        --checkpoint-tail-offset. Max defaults to 0 (disabled).
      Snapshot size: ~50 MB for Qwen3.5-4B, proportional to hidden_dim * n_ssm_states.

    Tokenizer: Special Token Visibility
      qwen_tokenize_decode2(): add add_special parameter. By default, BOS and
        EOS tokens produce empty output (hidden), but all other control and
        user-defined tokens (thinking tags, tool_call tags, FIM tokens, push/pop
        markers, etc.) are now printed with their raw vocab bytes. This allows
        client harnesses to detect and parse model-generated tag structure.
      Stop word matching fix: streaming /completion responses no longer leak
        the stop token's decoded content in the final SSE chunk when a stop
        word is matched.

    Server Endpoints
      GET /slots and GET /v1/slots: llama.cpp-compatible slot state endpoint.
        Returns a JSON array with one slot object (id=0) containing n_ctx,
        is_processing, params (defaults), and next_token state.

    Debugging
      PICOLM_DBG_TOKEN=1 env var: prints token ID and decoded text for every
        generated token to stderr. Useful for diagnosing tokenizer issues.

    Porting & Build
      Big-endian PPC support for GGUF model loading (byte-swap headers,
        tensor shapes, and GGUF scalar values)
      Altivec SIMD: PPC G4 support for rmsnorm and F16/F32 byte swap
      Mac OS X 10.6 Snow Leopard (Core 2 Duo, GCC 4.2.1) port
      GCC 15: eliminate all compilation warnings (AVX-512, fread, misc)

### v1.0-alpha2

    Q1_0 and Q2_0 extreme quantization support with full SIMD acceleration.

    Nine commits since alpha1, focusing on making the lowest-precision quant types
    (1-bit and 2-bit per weight) correct and fast across all supported architectures.

    Quantization: Q1_0 and Q2_0 (GGUF types 41/42)
      Q1_0 (128 values per 18-byte block, 1-bit sign + FP16 scale):
        AVX-512: VNNI via Q8_0 delegation
        AVX2: bytes_from_bits_32 bit expansion + maddubs_epi16 int8 MAC
        AVX/SSSE3: 128-bit pshufb bit expansion + maddubs
        NEON: table-based sign mask expansion + vpaddlq_s8
        Scalar: sign-flip accumulation
      Q2_0 (128 values per 34-byte block, 2-bit values + FP16 scale):
        AVX-512-VNNI: dpbusd unpack (2x gen speedup vs scalar)
        AVX2/AVX/SSSE3: scalar fallback (VNNI required for SIMD)
        NEON: vqtbl1q_u8 unpack + vmull_s8 wide multiply
        Scalar: shift-and-mask 2-bit unpack
      matmul() and matmul_dual_batch(): Q8_0 activation pre-quantization,
        matching the approach already used for Q4_0/Q4_K/Q8_0. This enables the
        int8 MAC kernels instead of scalar dequantize-per-element.
      vec_dot_q*_f32: inline Q8_0 delegation allows SSM alpha/beta paths to
        benefit from SIMD (AVX2 for Q1_0, VNNI for Q2_0).
      Benchmark (Qwen3.6-27B, AVX-512 host):
        Q2_0: 1.1 -> 2.2 tok/s gen (2x), 0.5 -> 1.2 tok/s prefill
        Q1_0: 2.0 -> 2.5 tok/s prefill, 1.8 -> 2.2 tok/s gen
        Memory: 230 MB (Q2_0) vs 3.1 GB (Q8_0)

    Bugfixes
      Q1_0 bit expansion: fixed AVX and SSSE3 paths that incorrectly used
        _mm_shufflelo_epi16 to extract bytes 2-3 of a 32-bit sign value
        (word indices 2-3 were always zero, corrupting sign masks for bits 16-31).

    CLI & Build
      Tokenization timing prints around tokenizer load and encode stages.
      PICOLM_EXIT_AFTER_TOKENIZE env var for benchmarking without forward pass.
      Windows: include <time.h> in _WIN32 block (fixes missing clock_t),
        add -lws2_32 to Makefile LDFLAGS (fixes server mode with MinGW/GCC).

### v1.0-alpha1

    From proof-of-concept to production inference engine. 143 commits, zero new
    dependencies.

    The original PicoLM was a clever demonstration: mmap a GGUF, stream one layer
    at a time, ~45MB RAM. It was missing everything needed for real-world use.
    This release adds the missing pieces across 143 commits, transforming it from
    a demo into a serious, multi-architecture inference engine that supports models
    up to 70B parameters, multiple model families (Llama, Qwen2/3, Qwen3.5/3.6 SSM),
    a full HTTP server, and SIMD acceleration on x86 and ARM.

    Quantization Support (was scalar-only, now fully accelerated)
      Q8_0:      AVX-512 (VNNI), AVX2, SSE4.1, NEON - pre-converted delta path, int8 MAC
      Q4_K:      AVX2, AVX, SSE2, NEON - multi-block scales, AVX-512 VNNI
      Q4_0:      AVX2, AVX, SSE2, NEON - activation quantization + int8 MAC
      Q4_1:      Scalar - full dequantize + vec_dot support
      Q4_0_8_8:  AVX2 - interleaved 8-row batched format
      Q4_0_4_4:  ARM DotProd - interleaved 4-row batched format (ARMv8.2+ SDOT)
      Q2_K-Q8_K: Scalar - K-quants via generic dequantize + vec_dot
      Q1_0/Q2_0: AVX2 (Q1_0), Scalar (Q2_0) - extreme quantization (1-2 bits/weight)
      F16/BF16:  AVX-512/AVX/NEON - full precision paths
      64KB FP16 lookup table replaces arithmetic fp16->fp32, ~4x faster on non-F16C

    Model Architecture Support
      Llama 1/2/3 (including all Qwen2 derivatives)
      Qwen3 (non-uniform head_dim, pairwise RoPE)
      Qwen3.5/3.6 (SSM/Mamba hybrid, interleaved Q+gate, v-head reordering,
        Gemma-style RMSNorm, rope_dim, MTP detection)
      Up to 70B parameters tested (miq-70b, Mistral-derived Llama architecture)
      MAX_LAYERS increased from 64 to 128

    SIMD & Performance
      AVX-512: VNNI integer MAC, fp16x16>fp32 conversion, vec_dot, rmsnorm, RoPE,
        attention Q*K and V-accumulation
      AVX2: Full Q8_0/Q4_K/Q4_0/Q4_0_8_8/Q1_0 int8 MAC paths
      AVX/SSE2: Q8_0/Q4_K paths, attention FMA
      ARM NEON: RoPE, FP16 HW conversion, Q8_0 int8 MAC, attention Q*K and V-accumulation
      ARM DotProd: Q4_0_4_4 interleaved format (vdotq_laneq_s32, ARMv8.2+)
      Persistent thread pool (generation-counter barrier, zero thread churn)
      Physical core count auto-detection (avoids hyperthreading overhead)
      Tiled/blocked attention for prefill (online softmax, F16 KV cache)
      Batched GEMV for prefill (activation quantization, per-token parallelization)
      GQA grouped attention (kv_mul heads share one KV scan)

    HTTP Server (OpenAI API Compatible)
      /v1/completions, /v1/chat/completions (OpenAI-compatible)
      /completion (llama.cpp-compatible, stop words, grammar, timings)
      /v1/models, /props, /health
      /tokenize, /detokenize
      Streaming responses with chunked transfer encoding
      Persistent model loading (model stays in memory across requests)
      KV cache reuse with prompt prefix matching (skip prefill on overlapping prompts)
        - SSM models: KV cache reuse supported but rewinding not yet (requires checkpointing)
      Client disconnect detection (abort prefill/generation on disconnect)
      --daemon mode, --prefault for deterministic prefill timing
      --mem option to mlock() layer weights in RAM

    Platform & Compatibility
      Linux: Full support, AVX-512/VNNI, ARM NEON, RISC-V
      Windows: SRWLOCK+CONDITION_VARIABLE thread pool, VirtualLock privilege acquisition,
        MinGW/ucrt64 support, --mem with working set quota
      Quantized KV cache: F16 (default), Q8_0 (53% of F16 memory), Q4_0 (34% of F16 memory)

    What's Not Here Yet
      GPU backend (HIP/CUDA infrastructure exists, not functional)
      SSM KV cache rewinding (checkpointing for Qwen3.5/3.6 mid-prompt truncation)
      Full batched GEMM kernels for AVX2 (currently uses GEMV)

## Inference Supported By

Hunger and A'rpi/MPlayer

## Patrons

a Lajos

## License

MIT License. See [LICENSE](LICENSE) for details.
