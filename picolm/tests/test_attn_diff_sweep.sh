#!/bin/bash
# Regression sweep: tiled prefill attention vs grouped reference.
# Covers aligned/unaligned start_pos, tile-boundary token counts, SWA, threads.
# Usage: ./test_attn_diff_sweep.sh [path/to/test_attn_diff]
T=${1:-/tmp/test_attn_diff}
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
echo "$((n-fail))/$n passed"; exit $fail
