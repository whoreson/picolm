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

## Status (what actually exists now)

Three layers of coverage, cheapest first. Only the first two are in CI.

### 1. `make check-rowinterleave` -- interleaved (R4/R8) layout, model-free
`tests/test_rowinterleave.c`. Runs in well under a second, no GGUF needed.

Asserts, for row-interleaved types:
- EVERY logical row of a block group dequantizes to an independent scalar
  reference build from the layout comment (not from the code under test);
- `vec_dot_row()` equals dequantize-then-dot for all 4/8 rows, and that rows
  1..N are not silently aliased to row 0;
- `gguf_type_row_ptr()` / `gguf_type_row_in_block()` / `gguf_type_rows_per_block()`
  address groups and rows correctly, and leave plain types alone;
- the type's GEMM equals a scalar reference at batch sizes straddling the R8
  minimum-batch threshold (1, 3, 15, 16, 17, 33).

Why these specific assertions: both shipped IQ4_NL_R4 bugs were layout errors --
"read only row 0 of a group" and "decoded nibbles with the wrong interleave" --
and neither was reachable by any prompt on any model in the test set. This test
fails against both, verified by re-introducing each bug.

Note it sets `PICOLM_R8_MIN_BLOCKS=0`: the decode-once R8 prefill path is gated
by a matrix-size heuristic (16384 blocks) that a CI-sized model can never reach.
That gate, not just the batch threshold, is part of why the bug shipped.

### 2. `make check-oracle` -- dummy oracle model, end to end
`tools/mk_qwen35_tiny_r4.py` generates a ~5 MB Qwen3.5/3.6-shaped hybrid model
(6 SSM + 2 attention layers) whose weights are type 220, including
`ssm_alpha`/`ssm_beta` at `n_v_heads` rows -- the exact per-head row walk that
broke. `tools/oracle_check.sh` then runs it and compares **hashes**:

- same run, 3 times              -> must be identical
- fast path vs its own fallback  -> must be identical
  (`PICOLM_Q4_R8`, `PICOLM_Q8_VNNI`, `PICOLM_Q8_AVX2T`)
- SIMD build vs scalar build     -> must be identical

The model is random-weight, so its output is gibberish. That is fine and
intended: the test is *"is the garbage the same garbage on every path?"* — not
"is it fluent". Do not add a fluency assertion here.

The scalar comparison is the one that catches the original "scalar works, SIMD
doesn't" signature; `make build-scalar-ref` produces `./picolm_scalar` for it.

Requires `pip install gguf numpy`; the target skips with a note if absent.

### 3. Real-model matrix -- NOT in CI
Everything in "What should be added" above. Needs multi-GB weights and minutes
per run, so it belongs in a nightly/manual job on a machine that has them.
Keep the pass criterion as **top-1 token agreement at >= 32 tokens** against
ik's `llama-cli`, not cosine similarity and definitely not fluency.

One thing worth adding to that job, since it is cheap once the models are
loaded: assert that the model's tensor-type histogram covers the quant types
under active development. A "large prefill" smoke test on a model with no
R4/R8 tensors proves the prefill loop runs, not that any quant kernel in it is
correct -- which is exactly how the IQ4_NL_R4 bugs passed CI.
