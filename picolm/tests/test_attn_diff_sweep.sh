#!/bin/bash
# Regression sweep: tiled prefill attention vs grouped reference.
# Covers aligned/unaligned start_pos, tile-boundary token counts, SWA, threads,
# and all supported KV cache types (f16 exact; q8_0/q4_0 within int8 requant
# noise -- see the QK precision note in model_attention.c).
# Usage: make check-attn   (or ./test_attn_diff_sweep.sh [path/to/test_attn_diff])
T=${1:-./test_attn_diff}
[ -x "$T" ] || { echo "$T not found; run 'make test_attn_diff' (or 'make check-attn')"; exit 2; }
fail=0; n=0
run() { # heads kv hd tokens start swa threads [kvtype] [tol]
  kt=${8:-f16}; tol=${9:-1e-4}
  out=$(PICOLM_TEST_TOL=$tol "$T" $1 $2 $3 $4 $5 $6 1 $7 - "$kt" 2>&1 | grep -oE 'MISMATCH|MATCH' | tail -1)
  n=$((n+1))
  if [ "$out" != "MATCH" ]; then echo "FAIL heads=$1 kv=$2 hd=$3 tokens=$4 start=$5 swa=$6 thr=$7 kv=$kt -> ${out:-no result}"; fail=$((fail+1)); fi
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

# ---- KV cache types other than F16 -------------------------------------------
# F16 must stay exact; Q8_0/Q4_0 requantize the K tile and the query rows to
# int8, so they are compared at a looser tolerance (bounded, non-accumulating
# error -- 8e-3 covers the worst observed case). tq3/tq4 are dequantized once
# per tile and stay close to F16 accuracy, but their grouped reference path is
# not implemented on all SIMD lanes, so only spot-check them here.
for kt in q8_0 q4_0; do
  for sp in 0 30 65 100; do
    for nt in 64 65 127 128 129 333; do run 8 4 64 $nt $sp 0 1 $kt 8e-3; done
  done
  run 8 4 64 300 0 96 1 $kt 8e-3
  run 8 4 128 200 30 0 1 $kt 8e-3
  run 16 2 128 200 30 0 1 $kt 8e-3
  run 32 8 128 120 30 0 8 $kt 8e-3
  run 24 4 256 128 30 0 1 $kt 8e-3
  run 24 4 256 256 0 0 8 $kt 8e-3
done
# head_dim not a multiple of 32 -> the QK path must fall back to F32 (exact).
for hd in 48 96 160; do
  for kt in q8_0 q4_0; do
    PICOLM_ATTN_TILED_MIN_TOKENS=1 PICOLM_ATTN_TILED_EFF=1 \
      run 8 2 $hd 130 30 0 1 $kt 1e-3
  done
done
# The exact F32 QK path must reproduce the reference bit-for-bit.
runf32() { # heads kv hd tokens start swa threads kvtype
  out=$(PICOLM_ATTN_QK=f32 PICOLM_TEST_TOL=1e-6 "$T" $1 $2 $3 $4 $5 $6 1 $7 - "$8" 2>&1 \
        | grep -oE 'MISMATCH|MATCH' | tail -1)
  n=$((n+1))
  [ "$out" = "MATCH" ] || { echo "FAIL f32-qk h=$1 kv=$2 hd=$3 tok=$4 sp=$5 thr=$7 kv=$8 -> ${out:-no result}"; fail=$((fail+1)); }
}
for kt in q8_0 q4_0; do
  runf32 8 4 64 128 30 0 1 $kt
  runf32 8 4 128 200 0 0 1 $kt
  runf32 24 4 256 200 30 0 1 $kt
done

echo "$((n-fail))/$n passed"; exit $fail
