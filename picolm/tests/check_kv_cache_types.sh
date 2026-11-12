#!/bin/sh
# check_kv_cache_types.sh -- assert every KV cache type produces coherent text.
#
# Why this exists
# ---------------
# TQ4 silently regressed for months. Commit 5811f5d ("tiled prefill attention
# for quantized KV caches") made the tiled path dequantize TQ K tiles with the
# *inverse* WHT (back to head space, so scores are plain dot products), but left
# the pre-existing "rotate Q with WHT forward" code in the batched prefill path
# for TQ4. The attention then computed dot(rotated_Q, unrotated_K).
#
# Nothing crashed and no kernel was wrong: every kernel did exactly what it was
# told, and TQ3 (which never rotated Q) stayed perfect. The symptom was a
# collapsed softmax -- TQ4's top logit dropped from 12.5 to 3.0 against a -13.9
# uniform baseline -- so the model emitted a single repeated token. Greedy
# decoding made it deterministic and easy to spot; with temperature it could
# look like mere incoherence.
#
# This is the cheap guard: for each cache type, generate greedily and require the
# output is not a single repeated character (the TQ4 failure signature -- its
# softmax collapsed so every step picked the same token). It does NOT validate
# quality: TQ3 at 3.5 bits/value is lossy by construction (see
# notes/picolm/tq3_session_20251218.md), and every low-bit cache legitimately
# continues a prompt differently from F16. The point is only to catch "the
# attention math is wired to the wrong domain", which is catastrophic, not subtle.
#
# USAGE
#   tests/check_kv_cache_types.sh <model.gguf> [picolm] [prompt] [tokens]
# Exit: 0 all cache types coherent, 1 a type failed, 2 setup problem.
#
# Prompt length matters. The tiled prefill path only engages above
# ATTN_TILED_MIN_TOKENS (see the size gate in batch_attention_layer), and the Q
# domain bug ONLY existed in the batched prefill. A 2-token prompt takes the
# single-token path, which was correct all along and would have passed while the
# bug was live. Keep the prompt at >= 32 tokens' worth of text.

set -eu

MODEL="${1:?usage: check_kv_cache_types.sh <model.gguf> [picolm] [prompt] [tokens]}"
BIN="${2:-./picolm}"
# Two prompts on purpose. The shot prompt reproduces the historical TQ4 failure
# through the *grouped* path (which is what a real short prompt hits), and the
# long one crosses ATTN_TILED_MIN_TOKENS into the tiled path, which is where the
# Q-domain bug was actually introduced. Either path alone can hide the other.
PROMPT_SHORT="${3:-The world has}"
PROMPT_LONG="The world has changed a lot in the last few decades. We have made incredible strides in technology, medicine, and our understanding of the world. The world has changed a lot in the last few decades."
NTOK="${4:-24}"

[ -x "$BIN" ] || { echo "check_kv_cache_types: $BIN is not executable" >&2; exit 2; }
[ -r "$MODEL" ] || { echo "check_kv_cache_types: cannot read $MODEL" >&2; exit 2; }

TMP="${TMPDIR:-/tmp}/kvcheck.$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT INT TERM

# mode -> -ctk/-ctv arguments. F16 first: it is the reference.
MODES="f16 q8_0 q4_0 tq3 tq4"

echo "model: $MODEL"
echo

fail=0
for mode in $MODES; do
    case "$mode" in
        f16)  args="-ctk f16" ;;
        *)    args="-ctk $mode -ctv $mode" ;;   # TQ3 is refused in mixed mode
    esac

    for which in short long; do
        case "$which" in
            short) prompt="$PROMPT_SHORT" ;;
            long)  prompt="$PROMPT_LONG" ;;
        esac

        # The CLI prints "OUTPUT: ..." on stderr (progress diagnostics share it),
        # so capture both streams and pick the line out of the log.
        if ! "$BIN" "$MODEL" -p "$prompt" -c 512 $args -t 0 -n "$NTOK" \
                >"$TMP/$mode-$which.log" 2>&1; then
            echo "FAIL $mode/$which: run failed"; sed -n '1,5p' "$TMP/$mode-$which.log"
            fail=1; continue
        fi
        out=$(sed -n 's/^OUTPUT: //p' "$TMP/$mode-$which.log" | tail -1)

        # Not a degenerate single repeated character (the TQ4 failure signature)
        distinct=$(printf '%s' "$out" | tr -d ' \\' | fold -w1 | sort -u | wc -l | tr -d ' ')
        nchar=$(printf '%s' "$out" | tr -d ' \\' | wc -c | tr -d ' ')

        # (a) is deliberately NOT a text match against the F16 baseline. Low-bit
        #     caches legitimately pick different (but equally sensible) words,
        #     e.g. SmolLM2 gives "know so far" on F16 and "have so far" on TQ4
        #     for the same prompt; both are valid continuations. The TQ4
        #     regression signature is not subtle -- it is one repeated token --
        #     so degeneracy is the discriminator.
        if [ "$nchar" -gt 4 ] && [ "$distinct" -le 1 ]; then
            echo "FAIL $mode/$which: degenerate output (one repeated character): '$out'"
            fail=1
        elif [ "$nchar" -lt 4 ]; then
            echo "FAIL $mode/$which: too little output ('$out')"
            fail=1
        else
            echo "ok   $mode/$which: $out"
        fi
    done
done

echo
if [ "$fail" -ne 0 ]; then
    echo "RESULT: FAIL"
    exit 1
fi
echo "RESULT: ok (all cache types coherent)"
