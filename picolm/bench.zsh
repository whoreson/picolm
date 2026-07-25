#!/usr/bin/env zsh
#
# bench.zsh — minimal, dependency-free benchmark harness for PicoLM.
#
# Measures prefill + decode throughput on real GGUF models with greedy,
# deterministic settings (-t 0, -s 42), so runs are directly comparable
# across code changes. Standard tools only: grep, sed, awk, sort, cksum.
#
# It parses picolm's own stderr summary (model-load time excluded, so the
# numbers reflect the inference engine, not disk I/O):
#     "Prefill:    <n> tokens in <s>s (<x> tok/s)"
#     "Generation: <n> tokens in <s>s (<x> tok/s)"
#     "Total: <s>s"
# and fingerprints the generated stdout for repeatability checks.
#
# QUICK START
#   make metal && ./bench.zsh                         # GPU, tinyllama, 3 iters
#   ./bench.zsh --modes cpu,gpu --model tinyllama    # A/B CPU vs GPU
#   ./bench.zsh --model big -n 96 -i 5               # 27B, longer runs
#   ./bench.zsh --out logs/after
#   ./bench.zsh --compare logs/before/summary.tsv logs/after/summary.tsv
#
# TYPICAL WORKFLOW FOR TESTING A CHANGE
#   1. ./bench.zsh --out logs/before                 # baseline (current binary)
#   2. <edit backend_metal.mm / model.c, rebuild>
#   3. ./bench.zsh --out logs/after
#   4. ./bench.zsh --compare logs/before/summary.tsv logs/after/summary.tsv

# Force the C locale for all numeric text processing. The harness parses
# picolm's period-decimal output ("Prefill/Generation ... tok/s") and computes
# medians/min/max with sort/awk/printf. Under a comma-decimal locale (de_DE,
# hu_HU, ...) BSD `sort -n` and float `printf` mis-handle the period and
# collapse "59.3" -> "59", silently corrupting median/min/max. picolm is a C
# program that never calls setlocale(), so it always emits period decimals
# regardless of this setting.
export LC_ALL=C

emulate -L zsh
setopt extended_glob

# ============================ defaults ======================================
local BIN=./picolm
local GEN=64                 # tokens to generate (decode)
local THREADS=0              # 0 = let picolm auto-detect physical cores
local ITERS=3
local WARMUP=1               # discard one short run first (warms disk + GPU)
local SEED=42
local GEN_WARM=8             # warmup generation length
local OUTDIR=
local COMPARE=0
local -a CMP_FILES=()
local -a MODELS=()           # resolved later if empty
local -a MODES=(gpu)
local LLMS_DIR=
local PROMPT="Explain how a transformer neural network works, step by step, in plain language. Cover self-attention, the feed-forward network, residuals, and layer normalization, and say why each part is needed."

# ============================ helpers =======================================
usage() {
  cat <<'EOF'
bench.zsh — PicoLM benchmark harness

  -m, --model NAME|PATH   model to run (PATH used as-is; NAME resolved in LLMs dir)
                          presets: tinyllama|tiny, big|27b|qwen  (repeatable; default: tinyllama)
      --modes LIST        comma list of cpu and/or gpu (default: gpu)
  -n N                    tokens to generate per run (default: 64)
  -j N                    threads for CPU path (default: auto)
  -i, --iters N           measured runs per model/mode (default: 3)
      --no-warmup         skip the discarded warmup run
  -s, --seed N            RNG seed (default: 42; greedy ignores it)
  -p, --prompt TEXT       prompt string
      --llms-dir PATH     directory holding .gguf files (default: auto-detect)
  -b, --bin PATH          picolm binary (default: ./picolm)
  -o, --out DIR           write logs + summary.tsv here (default: bench_logs)
      --compare A B       diff two summary.tsv files, then exit
  -h, --help              this help
EOF
}

human_size() {
  awk -v b="$1" 'BEGIN{
    split("B KB MB GB TB",u); i=1
    while(b>=1024 && i<5){b/=1024;i++}
    printf "%.0f%s", b, u[i]
  }'
}

# numeric median over stdin (filters non-numbers; e.g. failed runs -> "nan")
median() {
  grep -E '^-?[0-9]+\.?[0-9]*$' | sort -n | awk '
    { a[NR]=$1 }
    END {
      if (NR==0) { print "nan"; exit }
      if (NR%2==1) { printf "%.3f\n", a[(NR+1)/2] }
      else         { printf "%.3f\n", (a[NR/2]+a[NR/2+1])/2 }
    }'
}
min_of() { grep -E '^-?[0-9]+\.?[0-9]*$' | sort -n | head -1; }
max_of() { grep -E '^-?[0-9]+\.?[0-9]*$' | sort -n | tail -1; }

# run picolm once: stdout -> $stem.out, stderr -> $stem.err
run_picolm() {
  local mode="$1" model="$2" stem="$3" ng="${4:-$GEN}"
  local env_pre
  [[ "$mode" == "gpu" ]] && env_pre=(PICOLM_GPU=1) || env_pre=(PICOLM_GPU=0)
  local -a args=("$model" -p "$PROMPT" -n "$ng" -t 0 -s "$SEED")
  (( THREADS > 0 )) && args+=(-j "$THREADS")
  env $env_pre "$BIN" "${args[@]}" >"$stem.out" 2>"$stem.err"
}

# parse a "Prefill:"/"Generation:" line -> sets P_NTok P_SEC P_TPS
parse_line() {
  local errfile="$1" label="$2"
  P_NTok= P_SEC= P_TPS=
  local line; line=$(grep -m1 "^${label}: " "$errfile") || return 1
  [[ $line =~ "${label}: ([0-9]+) tokens in ([0-9.]+)s \(([0-9.]+) tok/s)" ]] || return 1
  P_NTok=$match[1] P_SEC=$match[2] P_TPS=$match[3]
}

# parse "Total: Xs" -> prints X
parse_total() {
  local line; line=$(grep -m1 "^Total: " "$1") || return 1
  [[ $line =~ "Total: ([0-9.]+)s" ]] || return 1
  print -- "$match[1]"
}

resolve_model() {
  local name="$1"
  [[ -f "$name" ]] && { print -- "${name:A}"; return 0 }
  [[ -n "$LLMS_DIR" ]] || return 1
  local pat
  case "$name:l" in
    tinyllama|tiny|small)              pat='(#i)*tiny*llama*' ;;
    big|27b|qwen|qwen27|thinkingcap)    pat='(#i)*27b*' ;;
    *)                                  pat="(#i)*${name}*" ;;
  esac
  local m
  for m in "$LLMS_DIR"/$~pat(N.); do print -- "${m:A}"; return 0; done
  return 1
}

# diff two summary.tsv (columns: model mode runs prefill_med decode_med ...)
# prints % throughput change of candidate (B) vs baseline (A); + is faster.
compare_summaries() {
  local A="$1" B="$2"
  print "=== compare ==="
  printf "  baseline : %s\n  candidate: %s\n\n" "$A" "$B"
  printf "%-34s %-4s  %10s %10s %6s   %10s %10s %6s\n" \
    model mode prefillA prefillB "d%" decodeA decodeB "d%"
  awk -F'\t' '
    FNR==NR { if (FNR>1) { key=$1"|"$2; pa[key]=$4; da[key]=$5 } next }
             { if (FNR>1) { key=$1"|"$2; pb[key]=$4; db[key]=$5 } }
    END {
      for (k in pb) {
        n=split(k, x, "|"); model=x[1]; mode=x[2]
        pa_v=(k in pa)?pa[k]:"-"; pb_v=pb[k]
        da_v=(k in da)?da[k]:"-"; db_v=db[k]
        pp = (pa_v!="-" && pa_v+0>0) ? sprintf("%+.0f%%", (pb_v/pa_v-1)*100) : ""
        dp = (da_v!="-" && da_v+0>0) ? sprintf("%+.0f%%", (db_v/da_v-1)*100) : ""
        printf "%-34s %-4s  %10s %10s %6s   %10s %10s %6s\n", \
          model, mode, pa_v, pb_v, pp, da_v, db_v, dp
      }
    }' "$A" "$B"
  print "\n(pref%/dec% = candidate vs baseline; + = faster)"
}

# ============================ arg parsing ===================================
while (( $# )); do
  case "$1" in
    -h|--help)    usage; exit 0 ;;
    -m|--model)   MODELS+=("$2"); shift 2 ;;
    --model=*)    MODELS+=("${1#*=}"); shift ;;
    --modes)      __ma="$2";          MODES=("${(@s:,:)__ma}"); shift 2 ;;
    --modes=*)    __ma="${1#*=}";     MODES=("${(@s:,:)__ma}"); shift ;;
    -n)           GEN="$2"; shift 2 ;;
    -j)           THREADS="$2"; shift 2 ;;
    -i|--iters)   ITERS="$2"; shift 2 ;;
    --no-warmup)  WARMUP=0; shift ;;
    -s|--seed)    SEED="$2"; shift 2 ;;
    -p|--prompt)  PROMPT="$2"; shift 2 ;;
    --prompt=*)   PROMPT="${1#*=}"; shift ;;
    --llms-dir)   LLMS_DIR="$2"; shift 2 ;;
    -b|--bin)     BIN="$2"; shift 2 ;;
    --bin=*)      BIN="${1#*=}"; shift ;;
    -o|--out)     OUTDIR="$2"; shift 2 ;;
    --out=*)      OUTDIR="${1#*=}"; shift ;;
    --compare)    COMPARE=1; CMP_FILES=("$2" "$3"); shift 3 ;;
    *) print -u2 "bench: unknown argument: $1"; usage; exit 2 ;;
  esac
done

# ---- compare mode (no benchmarking) ----------------------------------------
if (( COMPARE )); then
  (( ${#CMP_FILES[@]} == 2 )) || { print -u2 "bench: --compare needs two files"; exit 2; }
  for f in "${CMP_FILES[@]}"; do
    [[ -f "$f" ]] || { print -u2 "bench: not found: $f"; exit 2; }
  done
  compare_summaries "${CMP_FILES[1]}" "${CMP_FILES[2]}"
  exit 0
fi

# ============================ validate ======================================
[[ -x "$BIN" ]] || { print -u2 "bench: picolm binary not found/executable: $BIN"; exit 2; }
BIN="${BIN:A}"

if [[ -z "$LLMS_DIR" ]]; then
  local -a cands=(
    "$PWD/LLMs" "${BIN:h}/LLMs" "${BIN:h}/../LLMs" "${BIN:h}/../../LLMs"
    "$HOME/LLMs" "$HOME/Models" "$HOME/Downloads/LLMs"
  )
  local d
  for d in "${cands[@]}"; do [[ -d "$d" ]] && { LLMS_DIR="${d:A}"; break; }; done
fi

(( ${#MODELS[@]} )) || MODELS=(tinyllama)

local -a RESOLVED=()
local mm
for mm in "${MODELS[@]}"; do
  local r=$(resolve_model "$mm")
  if [[ -z "$r" ]]; then
    print -u2 "bench: could not resolve model '$mm'$([[ -n "$LLMS_DIR" ]] && echo " (looked in $LLMS_DIR)")"
    if [[ -n "$LLMS_DIR" && -d "$LLMS_DIR" ]]; then
      print -u2 "  available:"; ls -1 "$LLMS_DIR" | sed 's/^/    /'
    fi
    exit 2
  fi
  RESOLVED+=("$r")
done

local mo
for mo in "${MODES[@]}"; do
  case "$mo" in
    cpu|gpu) ;;
    *) print -u2 "bench: --modes entries must be cpu or gpu (got '$mo')"; exit 2 ;;
  esac
done

[[ -n "$OUTDIR" ]] || OUTDIR="$PWD/bench_logs"
[[ -d "$OUTDIR/raw" ]] || mkdir -p "$OUTDIR/raw"

# ============================ header ========================================
local bintime
if [[ "$(uname)" == Darwin ]]; then
  bintime=$(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$BIN")
else
  bintime=$(stat -c '%y' "$BIN")
fi
local warmstr=""; (( WARMUP )) && warmstr=" (+warmup)"
print "=== PicoLM bench ==="
print "bin      : $BIN  (built $bintime)"
print "llms dir : ${LLMS_DIR:-<none>}"
print "host     : $(uname -srm)"
print "modes    : ${MODES[*]}"
print "gen/prompt/iters: ${GEN} tok / ${#PROMPT} chars / ${ITERS} run(s)${warmstr}"
print "logs     : $OUTDIR"
print

# summary.tsv: one row per (model,mode)
print "model\tmode\truns\tprefill_tps_med\tdecode_tps_med\tdecode_tps_min\tdecode_tps_max\ttotal_s_med\tout_fp\tstatus" >"$OUTDIR/summary.tsv"

# ============================ benchmark loop ================================
run_block() {
  local model="$1" mode="$2"
  local mname="${model:t}"
  local msize; msize=$(human_size "$(stat -f%z "$model" 2>/dev/null || stat -c%s "$model" 2>/dev/null)")
  print "[$mname]  ${msize}  — mode: ${mode}"

  local safe="${mname//[^A-Za-z0-9._-]/_}_${mode}"
  local -a pre_tps=() dec_tps=() tot_s=()
  local fp_last="-" rstatus="ok"

  if (( WARMUP )); then
    run_picolm "$mode" "$model" "$OUTDIR/raw/${safe}_warm" "$GEN_WARM" >/dev/null 2>&1
  fi

  local r stem rc pt dt tot fp peek
  for (( r=1; r<=ITERS; r++ )); do
    stem="$OUTDIR/raw/${safe}_run${r}"
    run_picolm "$mode" "$model" "$stem"
    rc=$?

    if (( rc != 0 )); then
      rstatus="FAIL(rc=$rc)"
      printf "    run %d  %s\n" "$r" "$rstatus"
      tail -n 3 "$stem.err" 2>/dev/null | sed 's/^/        | /'
      pre_tps+=("nan"); dec_tps+=("nan"); tot_s+=("nan")
      continue
    fi

    parse_line "$stem.err" Prefill;    pt=$P_TPS
    parse_line "$stem.err" Generation; dt=$P_TPS
    tot=$(parse_total "$stem.err"); [[ -z "$tot" ]] && tot="nan"
    fp=$(cksum <"$stem.out" | awk '{print $1}')
    fp_last="$fp"
    pre_tps+=("${pt:-nan}"); dec_tps+=("${dt:-nan}"); tot_s+=("$tot")

    peek=$(head -c 54 "$stem.out" | tr '\n' ' ')
    printf "    run %d  pre %6.1f tok/s  dec %6.1f tok/s  tot %5.2fs  fp=%-8s  \"%s…\"\n" \
      "$r" "${pt:-0}" "${dt:-0}" "$tot" "$fp" "$peek"
  done

  local pmed dmed dmin dmax wmed
  pmed=$(printf '%s\n' "${pre_tps[@]}" | median)
  dmed=$(printf '%s\n' "${dec_tps[@]}" | median)
  dmin=$(printf '%s\n' "${dec_tps[@]}" | min_of)
  dmax=$(printf '%s\n' "${dec_tps[@]}" | max_of)
  wmed=$(printf '%s\n' "${tot_s[@]}" | median)
  printf "    -->   pre %6.1f tok/s  dec %6.1f tok/s  (dec %s..%s)  tot %ss\n\n" \
    "$pmed" "$dmed" "$dmin" "$dmax" "$wmed"

  printf "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$mname" "$mode" "$ITERS" "$pmed" "$dmed" "$dmin" "$dmax" "$wmed" "$fp_last" "$rstatus" \
    >>"$OUTDIR/summary.tsv"
}

for model in "${RESOLVED[@]}"; do
  for mode in "${MODES[@]}"; do
    run_block "$model" "$mode"
  done
done

# ============================ footer ========================================
print "=== summary (median throughput) ==="
print "wrote: $OUTDIR/summary.tsv  (raw logs in $OUTDIR/raw/)"
print "compare two builds:"
print "  ./bench.zsh --compare logs/before/summary.tsv logs/after/summary.tsv"
