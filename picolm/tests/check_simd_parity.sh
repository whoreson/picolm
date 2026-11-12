#!/bin/sh
# check_simd_parity.sh -- assert two PicoLM builds agree bit-for-bit.
#
# Why this exists
# ---------------
# The tiled prefill attention path selects a V-tile layout at run time (v_sm:
# row-major vs slice-major) and then picks a consumer kernel per SIMD tier. Two
# whole classes of bug live in that seam, and BOTH are silent -- no crash, no
# NaN, no out-of-bounds access, just wrong numbers:
#
#   1. A tier sets v_sm=1 (asking attn_v_prep_run() for slice-major) but has no
#      slice-major consumer. Commit 8eca9e4 did exactly this on AVX2: the
#      extractor learned the 8-wide layout, the consumer was still row-major, so
#      every V value came from the wrong KV position. Fimbulvetr2 IQ6_K
#      prefill degenerated into "ٹیم ٹیم ٹیم ..." while AVX-512 was fine.
#
#   2. A scalar writer lays the tile out with the wrong lane width. The F16
#      writers were correctly per-arch (16 on AVX-512, 8 on AVX2) but the TQ3/
#      TQ4 and generic-dequant writers hardcoded 16, so those caches were
#      transposed on AVX2 only.
#
# Both are invisible to a single-build smoke test: whichever build you happen to
# run looks either perfect or like a model that "quantised badly". Comparing two
# builds on the SAME greedy prompt makes them a one-line failure.
#
# Note this is a cross-*architecture* parity check, not a quality check. Greedy
# decoding is used so the comparison is deterministic; the reference SIMD tier
# (AVX-512) is assumed correct. It compares a PREFIX rather than whole strings --
# see the comment at the comparison for why exact equality is the wrong test.
#
# USAGE
#   tests/check_simd_parity.sh <model.gguf> <bin-a> <bin-b> [prompt] [tokens]
# Exit: 0 the two builds agree on every case, 1 a mismatch, 2 setup problem.
#
# CASE MATRIX deliberately covers the tiled gate: the prompts are long enough
# for ATTN_TILED_MIN_TOKENS (see batch_attention_layer), because the row-major
# fallback shared by both builds is what hid these bugs.

set -eu

MODEL="${1:?usage: check_simd_parity.sh <model.gguf> <bin-a> <bin-b> [prompt] [tokens]}"
BIN_A="${2:?missing first binary}"
BIN_B="${3:?missing second binary}"
PROMPT="${4:-The world has changed a lot in the last few decades. We have made incredible strides in technology, medicine, and our understanding of the world. The world has changed a lot in the last few decades.}"
NTOK="${5:-24}"

[ -x "$BIN_A" ] || { echo "check_simd_parity: $BIN_A is not executable" >&2; exit 2; }
[ -x "$BIN_B" ] || { echo "check_simd_parity: $BIN_B is not executable" >&2; exit 2; }
[ -r "$MODEL" ] || { echo "check_simd_parity: cannot read $MODEL" >&2; exit 2; }

TMP="${TMPDIR:-/tmp}/simdpar.$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT INT TERM

# KV cache cases. F16 is the tiled path's "fast" lane; q8_0/q4_0 exercise the
# int8/block lane; TQ3/TQ4 exercise the scalar-dequant slice-major writers that
# were hardcoding the wrong lane width.
CASES="f16 q8_0 q4_0 tq3 tq4"

echo "model:  $MODEL"
echo "bin A:  $BIN_A"
echo "bin B:  $BIN_B"
echo

fail=0
for mode in $CASES; do
    case "$mode" in
        f16) args="-ctk f16" ;;
        *)   args="-ctk $mode -ctv $mode" ;;   # TQ3 is refused in mixed mode
    esac

    # -t 0: greedy, so both builds must emit the identical token sequence.
    for bin in A B; do
        eval b=\$BIN_$bin
        if ! "$b" "$MODEL" -p "$PROMPT" -c 512 $args -t 0 -n "$NTOK" \
                >"$TMP/$mode-$bin.log" 2>&1; then
            echo "FAIL $mode: $bin did not run"
            sed -n '1,5p' "$TMP/$mode-$bin.log"
            fail=1
            continue 2
        fi
        sed -n 's/^OUTPUT: //p' "$TMP/$mode-$bin.log" | tail -1 >"$TMP/$mode-$bin.out"
    done

    # Compare a PREFIX, not the whole string. Different SIMD widths legitimately
    # accumulate in a different order (8- vs 16-wide FMA chains), which flips
    # genuine near-ties a few tokens in -- on a healthy pair above, AVX-512 and
    # AVX2 agree for 8 tokens then pick different (equally sensible) words. A
    # layout/gating bug instead diverges IMMEDIATELY, because it feeds the
    # attention transposed or not at all: the observed Fimbulvetr2 IQ6_K TQ4
    # failure emitted a repeated non-Latin token from the very first step.
    prefix_a=$(cut -c1-24 "$TMP/$mode-A.out")
    prefix_b=$(cut -c1-24 "$TMP/$mode-B.out")
    if [ "$prefix_a" = "$prefix_b" ] && [ -n "$prefix_a" ]; then
        echo "ok   $mode: same first 24 chars"
    else
        echo "FAIL $mode: builds disagree from the start -- a run-time layout/tier"
        echo "      decision is wrong (not a benign FMA-order near-tie)"
        echo "  A: $(cat "$TMP/$mode-A.out")"
        echo "  B: $(cat "$TMP/$mode-B.out")"
        fail=1
    fi
done

echo
if [ "$fail" -ne 0 ]; then
    echo "RESULT: FAIL"
    exit 1
fi
echo "RESULT: ok (SIMD tiers agree on every prefix)"
