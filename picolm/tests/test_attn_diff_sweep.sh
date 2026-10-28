#!/bin/bash
# Regression sweep: tiled prefill attention vs grouped reference.
# Covers aligned/unaligned start_pos, tile-boundary token counts, SWA, threads.
# Usage: make check-attn   (or ./test_attn_diff_sweep.sh [path/to/test_attn_diff])
T=${1:-./test_attn_diff}
[ -x "$T" ] || { echo "$T not found; run 'make test_attn_diff' (or 'make check-attn')"; exit 2; }
fail=0; n=0
run() { # heads kv hd tokens start swa threads
  out=$("$T" $1 $2 $3 $4 $5 $6 1 $7 2>&1 | grep -oE 'MISMATCH|MATCH' | tail -1)
  n=$((n+1))
  if [ "$out" != "MATCH" ]; then echo "FAIL heads=$1 kv=$2 hd=$3 tokens=$4 start=$5 swa=$6 thr=$7 -> ${out:-no result}"; fail=$((fail+1)); fi
}
for sp in 0 1 30 63 64 65 100 127 128 200; do
  for nt in 128 129 191 192 200 333; do
    run 8 4 64 $nt $sp 0 1
  done
done
for sp in 0 30 100; do run 8 4 64 300 $sp 96 1; run 8 4 64 300 $sp 96 4; run 8 4 128 200 $sp 0 4; run 16 2 128 200 $sp 0 1; done
# Small batches (below the 2*ATTN_TILE size gate): the tiled path is selected
# for these when threads allow, so it must be correct for every n_tokens.
for sp in 0 30 100; do
  for nt in 1 2 8 16 17 31 32 33 63 64 65 96 116 127; do run 8 4 64 $nt $sp 0 1; done
  run 8 4 128 116 $sp 0 4; run 32 8 128 40 $sp 0 1; run 8 4 64 100 $sp 48 1
done
# Stress: large/peaky scores and a K ramp so the running softmax max keeps
# growing across tiles (exercises online-softmax rescaling, incl. lazy rescale).
for q in 20 60 150; do for ramp in 0 4; do for sp in 0 30; do
  export PICOLM_TEST_QSCALE=$q PICOLM_TEST_KRAMP=$ramp
  run 8 4 64 333 $sp 0 1; run 8 4 128 200 $sp 0 4; run 8 4 64 300 $sp 96 1
done; done; done
unset PICOLM_TEST_QSCALE PICOLM_TEST_KRAMP
echo "$((n-fail))/$n passed"; exit $fail
