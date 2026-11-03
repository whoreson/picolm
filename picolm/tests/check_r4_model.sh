#!/bin/sh
# check_r4_model.sh -- sanity-check a GGUF that is supposed to contain
# row-interleaved (R4) tensors AND still produce coherent text.
#
# Two independent things must hold, and neither is implied by the other:
#
#   A. The file really contains interleaved tensors. A model with none of them
#      silently falls back to whatever types it does have, so a "coherent
#      output" check alone would pass while testing nothing (this is exactly
#      how the IQ4_NL_R4 bugs passed CI: the smoke model was f32/f16/q4_0/q8_0
#      only and no prompt could reach an R4 kernel).
#
#   B. The model still generates the expected text. Quantising stories260K is
#      delicate: several perfectly reasonable type mixes decode to <unk>
#      forever on a model this small (see tools/mk_stories_r4.sh). So assert
#      the known-good continuation rather than just "it produced tokens".
#
# Both bugs in the R4 code were found by differential runs, not by reading
# output; this script is the cheap "is the fixture still valid" guard. For
# actual differential coverage see tools/oracle_check.sh and
# tests/test_rowinterleave.c.
#
# USAGE
#   tests/check_r4_model.sh <model.gguf> [picolm-binary]
# Exit: 0 both properties hold, 1 either fails, 2 setup problem.

set -eu

MODEL="${1:?usage: check_r4_model.sh <model.gguf> [picolm]}"
BIN="${2:-./picolm}"
NTOK="${NTOK:-32}"
# Continuation that stories260K reliably produces at temp 0 for this prompt.
EXPECT="${EXPECT:-little girl named Lily}"

[ -f "$MODEL" ] || { echo "error: no such model: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "error: no such binary: $BIN" >&2; exit 2; }

fail=0

# ---- A. interleaved tensors present --------------------------------------
n_r4=$("$BIN" --list-tensors "$MODEL" 2>/dev/null |
       awk '$NF == 220 || $NF == 225' | wc -l | tr -d ' ')
# Report the full histogram once; it is the thing that was missing from CI.
echo "tensor type histogram:"
"$BIN" --list-tensors "$MODEL" 2>/dev/null |
    grep -E "^(blk\.|token_embd|output\.|output_norm)" |
    awk '{print $NF}' | sort -n | uniq -c | sed 's/^/  /'

if [ "$n_r4" -gt 0 ]; then
    echo "PASS: $n_r4 interleaved (R4) tensor(s) present"
else
    echo "FAIL: no interleaved (type 220/225) tensors -- this model cannot"
    echo "      exercise the R4 code paths at all. Build one with"
    echo "      tools/mk_stories_r4.sh"
    fail=1
fi

# ---- B. still coherent ---------------------------------------------------
# The generated text goes to STDOUT; PicoLM's "OUTPUT:" banner is on STDERR.
out=$("$BIN" "$MODEL" -p "Once upon a time" -n "$NTOK" -t 0 2>/dev/null | tr -d '\0')
case "$out" in
    *"$EXPECT"*)
        echo "PASS: expected continuation present"
        ;;
    *)
        echo "FAIL: expected '$EXPECT' in output"
        echo "      got: $out"
        echo "      (a model this small decodes to <unk> for several type mixes;"
        echo "       see tools/mk_stories_r4.sh for the mix that works)"
        fail=1
        ;;
esac

# ---- B2. report the tensor-type mix --------------------------------------
# Printed because it is the thing CI was previously blind to: a model can be
# perfectly coherent and still contain no interleaved tensors. Q5_0 is called
# out separately since it historically had no Q8_0-activation dot at all and so
# was unreadable on a scalar-only build; vec_dot_q5_0_q8_0() now covers that,
# and a build which regresses it will abort in matmul_worker_f instead of
# emitting junk.
for t in 6 220 225; do
    n=$("$BIN" --list-tensors "$MODEL" 2>/dev/null | awk -v t="$t" '$NF == t' | wc -l | tr -d ' ')
    [ "$n" -gt 0 ] && echo "type $t: $n tensor(s)"
done

# ---- C. deterministic (cheap, and catches uninitialised buffers) ---------
h1=$("$BIN" "$MODEL" -p "Once upon a time" -n 8 -t 0 2>/dev/null | md5sum)
h2=$("$BIN" "$MODEL" -p "Once upon a time" -n 8 -t 0 2>/dev/null | md5sum)
if [ "$h1" = "$h2" ]; then
    echo "PASS: deterministic across runs"
else
    echo "FAIL: nondeterministic output -- suspect uninitialised buffers"
    fail=1
fi

[ "$fail" -eq 0 ] && echo "=== check_r4_model: ok ===" && exit 0
echo "=== check_r4_model: FAILED ==="
exit 1
