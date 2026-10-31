# TODO: cross-family model validation matrix

**Why this exists.** The Qwen3.5/3.6 divergence (fixed in 7dc3bf4) survived for a
long time because every model family was validated in isolation, and the Qwen
check was "does it produce fluent English". It does — the model was rotating the
wrong 64 lanes of every head and still writing perfect prose. It only failed as
near-ties flipped, which needs long context and specific prompts.

The whole test suite was green the entire time. `test_attn_diff_sweep` (267
configs) and `test_glm5_ops` (16680 cases) are good differential harnesses, but
they exercise *kernels*, not *architectural decisions*. A wrong `rope_type`,
a wrong `n_k_heads`, a wrong norm placement, or a wrong head-pairing rule is
invisible to them — every kernel behaves exactly as specified while the model
computes the wrong thing.

## What should be added

A script that, for every GGUF in a configured directory, does the following and
compares against a reference implementation (ik_llama's `llama-cli`, or the
`llama-qwen-oracle` built in `~/ik_old/examples/glm5-dump`):

1. Tokenize a fixed prompt with a *known* answer and check the greedy
   continuation. Fluent output is not a pass criterion; exact match (or
   top-k overlap) is.
2. Dump per-position logits from both and assert cosine similarity above a
   family-specific threshold plus agreement of the top-1 token. The howto in
   `notes/picolm/howto_numerical_comparison.md` has the tooling.
3. Assert a *second* prompt that is long enough (>= 32 tokens) for near-ties to
   matter. Short prompts hide the entire class of bug this is meant to catch —
   the 27B matched ik to 0.9999 on 5 tokens while being wrong, and only failed
   at 37.
4. Record the resolved architectural knobs (rope_type, rope_dim, norm placement,
   head counts, partial-rotary, layer-type pattern) and diff them against what
   the reference reports. This is the check that would have caught the bug in
   one step, before any numerics.

## Families worth covering

At minimum, one model per family that PicoLM claims to support, because the
failure mode is silent and per-family:

- Llama / Llama-3 (pairwise RoPE, `LLAMA_ROPE_TYPE_NORM`)
- Qwen2 / Qwen3 / Qwen3.5 / Qwen3.6 (half-split; QWEN35 is IMROPE in ik)
- Gemma-3n (own norm placement, own RoPE branch, batched prefill falls back)
- GPTNeoX / StableLM / Clio (parallel residual, split-halves RoPE, LayerNorm)
- GLM-5.3-Flash (MLA + KDA, `general.architecture` normalisation)
- A non-English or code prompt for each, since tokenizer bugs also pass the
  "fluent English" test.

## Note on thresholds

Do not set a single global cosine threshold. A real quantisation difference
between PicoLM and ik is expected and family-dependent — measured ~1.17e-02
relative on Qwen3.6-27B purely from int8 activation quantisation (see the
`TODO(activation-quant-precision)` note in `type_dispatch.c`). The check that
actually catches architectural bugs is **top-1 token agreement at >= 32
tokens**, not the absolute cosine. Prefer that as the pass criterion and keep
cosine as a diagnostic.
