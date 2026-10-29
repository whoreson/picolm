#!/bin/bash
# Thread-pool stress matrix (thread counts x spin budgets). Hangs are caught by timeout.
# Usage: make check-pool          (set TSAN=1 to also run the ThreadSanitizer build)
T=${1:-./test_pool_stress}
[ -x "$T" ] || { echo "$T not found; run 'make check-pool'"; exit 2; }
fail=0; n=0
for nt in 2 3 4 8 24; do for sp in 0 1 5 50 500; do
  n=$((n+1)); r=$(PICOLM_POOL_SPIN_US=$sp timeout 120 "$T" $nt 3000 $nt$sp 2>&1 | tail -1)
  case "$r" in OK*) ;; *) echo "FAIL nt=$nt spin=${sp}us: ${r:-hang/timeout}"; fail=$((fail+1));; esac
done; done
echo "$((n-fail))/$n passed"; exit $fail
