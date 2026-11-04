#!/bin/sh
# simd_diff.sh -- differential test: SIMD build vs scalar build on a real model.
#
# WHY THIS EXISTS
# ---------------
# This is the check that found the zeroed-R8-kernels bug, and it is the check
# that would have caught it years earlier. The failure mode it targets is
# specific and nasty: a kernel that exists only for one ISA has a non-functional
# stub elsewhere, so one build silently computes different numbers. Nothing
# crashes, nothing warns, and the output still looks like text -- it is simply
# a different (wrong) string.
#
# Concretely it found that vec_dot_q4_0_r8_q8_2_avx2 and
# vec_dot_q8_0_r8_q8_0_avx2 returned zeros when AVX2 was absent, so any model
# carrying Q4_0_R8 or Q8_0_R8 decoded to garbage on a scalar build:
#
#   SIMD  : "there was a little girl named Lily"
#   scalar: "there was a big, there was a big, funnower"
#
# A hash comparison catches that in one line; reading the output and thinking
# "hmm, that looks a bit off" does not.
#
# WHAT IT DOES
# ------------
# Runs the same greedy generation on both builds and requires byte-identical
# output. Two prompt lengths, because short and long prefill exercise different
# dispatch paths (small batches take the GEMV/worker path, larger ones take the
# tiled GEMM), and the R8 decode-once path is additionally gated on batch size.
#
# PICOLM_R8_MIN_BLOCKS=0 opens that size gate, so the decode-once kernels are
# compared even on a model too small to reach them naturally.
#
# USAGE
#   tools/simd_diff.sh [-m model.gguf] [-n tokens]
# Environment:
#   PICOLM   SIMD binary   (default ./picolm)
#   SCALAR   scalar binary (default ./picolm_scalar)
# Exit: 0 agree, 1 differ, 2 setup problem.

set -u

MODEL="../models/stories260K-stuffed.gguf"
NTOK=24
while getopts "m:n:h" opt; do
    case "$opt" in
        m) MODEL="$OPTARG" ;;
        n) NTOK="$OPTARG" ;;
        h) sed -n '2,40p' "$0"; exit 0 ;;
        *) exit 2 ;;
    esac
done

SIMD="${PICOLM:-./picolm}"
SCALAR_BIN="${SCALAR:-./picolm_scalar}"

[ -x "$SIMD" ] || { echo "error: no SIMD binary at $SIMD" >&2; exit 2; }
[ -x "$SCALAR_BIN" ] || {
    echo "error: no scalar binary at $SCALAR_BIN."
    echo "       Run 'make build-scalar-ref' first -- without a scalar build this"
    echo "       test cannot run, and skipping it is how the R8 stub bug lived."
    exit 2
}
[ -f "$MODEL" ] || { echo "error: no model at $MODEL" >&2; exit 2; }

fail=0
run() { # run <binary> <prompt> <ntok>
    PICOLM_R8_MIN_BLOCKS=0 "$1" "$MODEL" -p "$2" -n "$3" -t 0 -c 256 2>/dev/null |
        md5sum | cut -d' ' -f1
}

compare() { # compare <label> <prompt> <ntok>
    a=$(run "$SIMD" "$2" "$3")
    b=$(run "$SCALAR_BIN" "$2" "$3")
    if [ "$a" = "$b" ]; then
        printf '%-42s ok\n' "$1"
    else
        printf '%-42s DIFFER  %s != %s\n' "$1" "$a" "$b"
        echo "  SIMD  : $(PICOLM_R8_MIN_BLOCKS=0 "$SIMD" "$MODEL" -p "$2" -n "$3" -t 0 -c 256 2>/dev/null | head -c 90)"
        echo "  scalar: $(PICOLM_R8_MIN_BLOCKS=0 "$SCALAR_BIN" "$MODEL" -p "$2" -n "$3" -t 0 -c 256 2>/dev/null | head -c 90)"
        echo "  -> a kernel is producing different numbers on one build. If the"
        echo "     scalar output is garbage rather than merely different, suspect"
        echo "     a zero-returning non-SIMD stub for the type that model uses."
        fail=1
    fi
}

SHORT="Once upon a time"
LONG="000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
Once upon a time"

echo "SIMD   : $SIMD"
echo "scalar : $SCALAR_BIN"
echo "model  : $MODEL"
echo
compare "short prompt ($NTOK tok)"            "$SHORT" "$NTOK"
compare "long prompt, small batch (8 tok)"    "$LONG" 8
compare "long prefill ($NTOK tok)"            "$LONG" "$NTOK"
# Long prefill again, past the tiled-attention size gate. The gate needs
# >= 2*ATTN_TILE (128) prefill tokens, so the prompts above (137 tokens with
# a 24-token budget) always took the TILED path -- but 24 generated tokens is
# enough for a GENERATION-time divergence to be masked. It is the
# prefill that decides the first token, but a longer budget is cheaper than
# debating it -- the AVX-512 head_dim<16 accumulator overrun (see
# attn_process_tile) kept diverging well past the first token.
compare "long prefill + long decode (96 tok)"  "$LONG" 96

echo
if [ "$fail" -eq 0 ]; then
    echo "=== simd_diff: SIMD and scalar builds agree ==="
    exit 0
fi
echo "=== simd_diff: MISMATCH ==="
exit 1
