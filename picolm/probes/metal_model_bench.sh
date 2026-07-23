#!/bin/bash
# probes/metal_model_bench.sh
#
# End-to-end timing of the PicoLM Metal backend on real GGUF models.
# Runs ./picolm on each model, parses prefill/generation tok/s, and prints a
# table. Use it to measure optimization gains (run before/after with the same
# models and compare).
#
# Run from picolm/picolm (where the `picolm` binary lives). Build first with:
#     make metal
# then:
#     make metal-model-bench            # GPU only
#     make metal-model-bench CPU=1      # GPU + CPU baseline (CPU can be slow!)
#     LABEL=after make metal-model-bench # tag the run
#
# Override models / prompt / length with env vars or args:
#     MODELS="../../a.gguf ../../b.gguf" make metal-model-bench
#     ./probes/metal_model_bench.sh /abs/path/model.gguf
set -u

BIN="${BIN:-./picolm}"
PROMPT="${PROMPT:-The capital of France is}"
N="${N:-60}"
LABEL="${LABEL:-gpu}"
DO_CPU="${CPU:-0}"
MODELS_DIR="${MODELS_DIR:-../..}"

# --- resolve model list: args > $MODELS > autodiscover in $MODELS_DIR ---
if [ "$#" -gt 0 ]; then
    MODELS="$*"
else
    MODELS="${MODELS:-}"
    if [ -z "$MODELS" ]; then
        discovered=""
        shopt -s nocaseglob
        for pat in "*tinyllama*q4*" "*thinkingcap*qwen*27b*q4*" "*qwen3*27b*q4*" "*qwen*-27b*q4*"; do
            for f in "$MODELS_DIR"/$pat; do
                [ -f "$f" ] && discovered="$discovered $f"
            done
        done
        shopt -u nocaseglob
        MODELS="$(echo $discovered | tr ' ' '\n' | awk '!seen[$0]++' | tr '\n' ' ')"
    fi
fi

# --- checks ---
if [ ! -x "$BIN" ]; then
    echo "ERROR: binary '$BIN' not found. Build it first:  make metal" >&2
    exit 1
fi
if [ -z "$MODELS" ]; then
    echo "ERROR: no models found in '$MODELS_DIR'. Pass paths as args:" >&2
    echo "       ./probes/metal_model_bench.sh /path/to/model.gguf" >&2
    exit 1
fi

echo "metal-model-bench  bin=$BIN  label=$LABEL  prompt=\"$PROMPT\"  n=$N  cpu=$DO_CPU"
printf "%-46s %-5s %12s %12s %10s   %s\n" "model" "mode" "prefill(t/s)" "gen(t/s)" "total(s)" "sample output"
printf '%0.s-' $(seq 1 118); echo

run_one() {
    local model="$1" mode="$2"
    local gpu_flag=""
    [ "$mode" = "gpu" ] && gpu_flag="PICOLM_GPU=1"
    local out
    out=$(env $gpu_flag "$BIN" "$model" -p "$PROMPT" -n "$N" -t 0 2>&1)
    # parse: "Prefill: 6 tokens in 0.31s (19.1 tok/s)"  /  "Generation: ..."  /  "Total: 4.66s"
    local pft=$(echo "$out" | grep -i 'Prefill:'    | sed -n 's/.*(\([0-9.]*\) tok\/s).*/\1/p')
    local gnt=$(echo "$out" | grep -i 'Generation:' | sed -n 's/.*(\([0-9.]*\) tok\/s).*/\1/p')
    local tot=$(echo "$out" | grep -i 'Total:'      | sed -n 's/Total: *\([0-9.]*\)s.*/\1/p')
    local sample=$(echo "$out" | sed -n 's/^OUTPUT: *//p' | tr '\n' ' ' | cut -c1-30)
    [ -z "$pft" ] && pft="-"
    [ -z "$gnt" ] && gnt="-"
    [ -z "$tot" ] && tot="-"
    [ -z "$sample" ] && sample="(no OUTPUT line)"
    local base; base=$(basename "$model")
    printf "%-46s %-5s %12s %12s %10s   %s\n" "$base" "$mode" "$pft" "$gnt" "$tot" "$sample"
}

for m in $MODELS; do
    if [ ! -f "$m" ]; then echo "skip (missing): $m" >&2; continue; fi
    run_one "$m" "gpu"
    if [ "$DO_CPU" = "1" ]; then run_one "$m" "cpu"; fi
done
echo
