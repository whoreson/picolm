#!/bin/bash
# Regression sweep: decode attention (blocked softmax + split-KV) vs the
# original per-position reference. Usage: make check-attn-decode
T=${1:-./test_attn_decode}
[ -x "$T" ] || { echo "$T not found; run 'make check-attn-decode'"; exit 2; }
fail=0; n=0
run() { # heads kv hd pos kvtype swa threads [vtype]
  out=$("$T" $1 $2 $3 $4 $5 $6 $7 0 1 ${8:-$5} 2>&1 | grep -oE 'MISMATCH|MATCH' | tail -1)
  n=$((n+1))
  if [ "$out" != "MATCH" ]; then echo "FAIL heads=$1 kv=$2 hd=$3 pos=$4 ktype=$5 swa=$6 thr=$7 vtype=${8:-$5} -> ${out:-no result}"; fail=$((fail+1)); fi
}
# every KV type, partial blocks (ctx not a multiple of 32), 1 and 4 threads
for kt in 0 1 2 3 4; do
  for pos in 0 1 30 31 32 33 63 64 100 255 1000 2047; do
    run 8 4 64 $pos $kt 0 1; run 32 8 128 $pos $kt 0 4
  done
done
# mixed K/V cache types (-ctk != -ctv), including with split-KV
for kt in 0 1 2 4; do for vt in 0 1 2 3 4; do
  [ $kt = $vt ] && continue
  run 8 2 128 333 $kt 0 1 $vt; run 8 2 128 1500 $kt 0 4 $vt
done; done
# small / odd head dims (8-wide AVX2 only supports multiples of 8; others use the reference)
for hd in 8 16 24 40 48 72; do run 8 4 $hd 300 0 0 1; run 8 4 $hd 70 0 0 4; done
# GQA ratios (specialised 1/2/4/8 and runtime 3/5/6/7), head dims incl. 80/96/256
for kvm in 1 2 3 4 5 6 7 8; do
  for hd in 64 80 96 128 256; do run $((kvm*2)) 2 $hd 700 0 0 1; run $((kvm*2)) 2 $hd 700 1 0 1; done
done
# sliding window (incl. window larger than ctx, and tiny windows) and split-KV
for swa in 1 5 33 64 500 5000; do for kt in 0 1; do run 8 2 128 1500 $kt $swa 1; run 8 2 128 1500 $kt $swa 8; done; done
# split-KV: more threads than KV heads, various context lengths / chunk edges
for thr in 2 3 5 8 16 24; do for pos in 255 511 512 1023 4095 9999; do run 32 8 128 $pos 0 0 $thr; run 8 2 128 $pos 1 0 $thr; run 4 1 64 $pos 0 0 $thr; done; done
# peaky softmax / growing K scale: running max keeps rising across blocks
for q in 20 60 150; do for ramp in 0 4; do
  export PICOLM_TEST_QSCALE=$q PICOLM_TEST_KRAMP=$ramp
  run 8 2 128 2000 0 0 1; run 8 2 128 2000 0 0 8; run 8 2 64 777 1 0 1
done; done
unset PICOLM_TEST_QSCALE PICOLM_TEST_KRAMP
echo "$((n-fail))/$n passed"; exit $fail
