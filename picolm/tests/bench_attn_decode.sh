#!/bin/bash
# Decode-attention benchmark: reference vs blocked kernel vs blocked + split-KV.
# Usage: tests/bench_attn_decode.sh [threads] [heads] [kv_heads] [head_dim] [kvtype] [ctx list...]
#   kvtype: 0=F16 1=Q8_0 2=Q4_0 3=TQ3 4=TQ4.   Needs: make test_attn_decode
#   e.g.  tests/bench_attn_decode.sh 24 32 8 128 0 8192 32768 131072
# Prints median ms per call (one layer, all KV heads) and speedup vs the reference.
T=${T:-./test_attn_decode}
THR=${1:-$(nproc)}; H=${2:-32}; KV=${3:-8}; HD=${4:-128}; KT=${5:-0}
shift 5 2>/dev/null
CTXS=${@:-2048 8192 16384 32768 65536 131072}
[ -x "$T" ] || { echo "$T not found; run 'make test_attn_decode'"; exit 2; }
ms() { grep "TIME $1" | sed -E 's/.*median +([0-9.]+) ms.*/\1/'; }
echo "threads=$THR heads=$H kv_heads=$KV head_dim=$HD kvtype=$KT"
printf "%8s | %10s | %10s %7s | %10s %7s | %s\n" ctx "ref ms" "blocked ms" "x" "+split ms" "x" "split tasks"
for ctx in $CTXS; do
  pos=$((ctx-1))
  reps=$(( ctx >= 65536 ? 8 : 20 ))
  a=$("$T" $H $KV $HD $pos $KT 0 $THR $reps)         # ref + new (split-KV on)
  b=$(PICOLM_ATTN_SPLIT=0 "$T" $H $KV $HD $pos $KT 0 $THR $reps)   # new, one task per KV head
  echo "$a" | grep -q MISMATCH && echo "  !! MISMATCH at ctx=$ctx"
  r=$(echo "$a" | ms ref); n=$(echo "$a" | ms new); s=$(echo "$b" | ms new)
  tasks=$(( THR / KV )); [ $tasks -lt 1 ] && tasks=1
  printf "%8d | %10.3f | %10.3f %6.2fx | %10.3f %6.2fx | %dx%d\n" $ctx $r $s $(echo "$r/$s" | bc -l) $n $(echo "$r/$n" | bc -l) $KV $tasks
done
