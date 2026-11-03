#!/bin/sh
# oracle_check.sh -- differential test for quant/layout bugs using a tiny
# dummy-oracle model (see tools/mk_qwen35_tiny_r4.py).
#
# IDEA
# ----
# A random-weight model produces gibberish, but the gibberish must be the SAME
# gibberish on every code path.  That is the whole test:
#
#   * same inputs, different runs                -> identical bytes
#   * fast path vs its own fallback              -> identical bytes
#   * SIMD build vs scalar build                 -> identical bytes
#
# "Is the output known garbage?"  Yes -- and compared by hash, not by eye.
# Do NOT judge this model on fluency; judge it on AGREEMENT.
#
# This is what the shipped IQ4_NL_R4 bugs needed:
#   BUG A  vec_dot on an interleaved type read only row 0 of a block group
#   BUG B  R8 prefill repack used the wrong nibble layout
# Both produce plausible-looking numbers on their own; only a differential run
# exposes them.
#
# USAGE
# -----
#   tools/oracle_check.sh [-m model.gguf] [-n tokens] [-p prompt] [-j threads]
#
# Environment:
#   PICOLM   path to the picolm binary (default: ./picolm)
#   SCALAR   path to a scalar build, to also diff SIMD vs scalar (optional)
#
# Exit: 0 all comparisons agreed, 1 a mismatch, 2 setup problem.
#
# Why PICOLM_R8_MIN_BLOCKS=0: the decode-once R8 prefill GEMM is gated by a
# matrix-size heuristic (default 16384 blocks) that a CI-sized model can never
# reach.  Opening the gate is REQUIRED for this test to cover that path -- it is
# precisely the gate that let BUG B ship unnoticed.

set -u

MODEL=""
NTOK=16
PROMPT="Once upon a"
THREADS=1
while getopts "m:n:p:j:h" opt; do
    case "$opt" in
        m) MODEL="$OPTARG" ;;
        n) NTOK="$OPTARG" ;;
        p) PROMPT="$OPTARG" ;;
        j) THREADS="$OPTARG" ;;
        h) sed -n '2,40p' "$0"; exit 0 ;;
        *) exit 2 ;;
    esac
done

BIN="${PICOLM:-./picolm}"
[ -x "$BIN" ] || { echo "error: picolm binary not found: $BIN" >&2; exit 2; }

if [ -z "$MODEL" ]; then
    echo "error: no model given (-m). Generate one with:" >&2
    echo "  python3 tools/mk_qwen35_tiny_r4.py -o /tmp/qwen35_tiny_r4.gguf" >&2
    exit 2
fi
[ -f "$MODEL" ] || { echo "error: model not found: $MODEL" >&2; exit 2; }

# A prompt long enough to push prefill past the R8 minimum batch.
case "$PROMPT" in
    *"Once upon a"*) ;;   # caller-provided, leave alone
esac
LONG_PROMPT="$(printf '%025d\n%s' 0 "$PROMPT")"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

run() { # run <outfile> <env-assignments...> ; remaining args via $RUN_ARGS
    # shellcheck disable=SC2086
    "$@" > "$1" 2>/dev/null
}

hash_run() { # hash_run <prompt> <binary> <envstring>
    _prompt="$1"; _bin="$2"; _env="$3"
    # shellcheck disable=SC2086
    env $_env PICOLM_R8_MIN_BLOCKS=0 "$_bin" "$MODEL" \
        -p "$_prompt" -n "$NTOK" -t 0 -c 256 -j "$THREADS" 2>/dev/null |
        md5sum | cut -d' ' -f1
}

fail=0
say() { printf '%-46s %s\n' "$1" "$2"; }

# ---- 1. determinism: the same run must produce the same bytes -------------
h1=$(hash_run "$PROMPT" "$BIN" "")
h2=$(hash_run "$PROMPT" "$BIN" "")
h3=$(hash_run "$PROMPT" "$BIN" "")
if [ "$h1" = "$h2" ] && [ "$h2" = "$h3" ]; then
    say "determinism (3 runs)" "ok"
else
    say "determinism (3 runs)" "FAIL  $h1 / $h2 / $h3"
    echo "  -> the model itself is nondeterministic. That is a bug: check for"
    echo "     uninitialized buffers or a dimension mismatch in the generator"
    echo "     (e.g. head_v_dim = inner / n_v_heads truncating)."
    fail=1
fi

# ---- 2. fast path vs fallback ---------------------------------------------
# These are the pairs that have actually been wrong. Each compares the
# optimised kernel against the simpler path it replaces; they must agree.
compare() { # compare <label> <prompt> <envA> <envB>
    a=$(hash_run "$2" "$BIN" "$3")
    b=$(hash_run "$2" "$BIN" "$4")
    if [ "$a" = "$b" ]; then
        say "$1" "ok"
    else
        say "$1" "FAIL  $a != $b"
        fail=1
    fi
}

compare "R8 decode-once GEMM vs fallback"      "$LONG_PROMPT" "" "PICOLM_Q4_R8=0"
compare "R8 VNNI kernel vs AVX2 kernel"        "$LONG_PROMPT" "" "PICOLM_Q8_VNNI=0"
compare "Q8 panel AVX2T vs tiled"              "$LONG_PROMPT" "" "PICOLM_Q8_AVX2T=0"

# ---- 3. SIMD build vs scalar build (optional) -----------------------------
if [ -n "${SCALAR:-}" ] && [ -x "${SCALAR}" ]; then
    a=$(hash_run "$LONG_PROMPT" "$BIN" "")
    b=$(hash_run "$LONG_PROMPT" "$SCALAR" "")
    if [ "$a" = "$b" ]; then
        say "SIMD build vs scalar build" "ok"
    else
        say "SIMD build vs scalar build" "FAIL  $a != $b"
        echo "  -> a SIMD kernel disagrees with the scalar reference. This is"
        echo "     the 'scalar works, SIMD does not' signature."
        fail=1
    fi
else
    say "SIMD build vs scalar build" "skipped (set SCALAR=/path/to/scalar/picolm)"
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "=== oracle_check: all comparisons agreed ==="
    exit 0
else
    echo "=== oracle_check: MISMATCH (see above) ==="
    exit 1
fi
