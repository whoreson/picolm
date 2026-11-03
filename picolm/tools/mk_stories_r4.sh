#!/bin/sh
# mk_stories_r4.sh -- build a stories260K GGUF that contains row-interleaved
# (IQ4_NL_R4, GGUF type 220) tensors AND still produces coherent English.
#
# WHY THIS IS FIDDLY
# ------------------
# stories260K is tiny: n_embd=64, n_ffn=172, 4 layers, vocab 512. Two separate
# problems show up, neither of which is PicoLM's fault:
#
#  1. n_ffn=172 is not a multiple of QK4_NL (32), so ffn_down (n=172) cannot be
#     quantized to any 4-bit block type. ggml asserts
#     "n_per_row % QK4_NL == 0" and aborts. Those tensors must be pinned to
#     something that tolerates the size -- F16 works (and is what the shipped
#     stories260K-Q2_K uses for ffn_down anyway).
#
#  2. token_embd cannot be row-interleaved at all ("token_embd.weight tensors
#     cannot be quantized with row-interleaved quants"), so it stays Q4_0.
#
# And a third, non-obvious one: NOT every working type mix yields readable
# text. A plain "Q2_K" target on a model this small does not create Q2_K at
# all -- llama-quantize silently falls back to a mixture -- and the resulting
# mix (IQ2_S-family, type 154) makes PicoLM emit <unk> forever. Plain Q4_0,
# Q5_0, Q8_0 and IQ4_NL *also* all decode to <unk> here. Only the Q4_0 +
# Q8_0-output + F16-ffn_down shape below produces English.
#
# So this script does not "pick a quant level"; it reproduces one specific,
# verified-working mix and then swaps the attention projections to R4.
#
# WHAT YOU GET
# ------------
#   attn_q / attn_k / attn_v / attn_output : IQ4_NL_R4 (type 220)  <- the point
#   ffn_gate                               : Q6_0
#   ffn_up                                 : Q5_0
#   ffn_down                               : F16   (n=172, not 32-aligned)
#   token_embd                             : Q4_1  (cannot be interleaved)
#   output                                 : Q8_0
#   *_norm                                 : F32
#
# Seven distinct tensor types in a 300 KB model. The mix is deliberately wider
# than the minimum: every additional type exercises a different kernel, and
# Q6_0 / Q5_0 / Q4_1 are cheap to include once the delicate tensors are pinned.
# ffn_down is the fragile one -- quantising it to anything 32-value-block based
# fails outright (n=172), and Q8_0 there silently degrades output to <unk>.
#
# Output is ~277 KB (same size as models/stories260K-Q2_K.gguf), loads in
# milliseconds, generates deterministic English, and exercises
# sgemm_iq4_nl_r4 / the R8 prefill repack for real.
#
# COVERAGE NOTE
# -------------
# This model catches BUG B (wrong R4 nibble layout in the R8 repack) but NOT
# BUG A (per-head row addressing), because stories260K is a llama-architecture
# model with no SSM layers and therefore no ssm_alpha/ssm_beta. For BUG A use
# the generated model from tools/mk_qwen35_tiny_r4.py via tools/oracle_check.sh.
#
# USAGE
#   tools/mk_stories_r4.sh                      # -> ./stories260K-r4.gguf
#   tools/mk_stories_r4.sh -o out.gguf
#   F32=/path/to/stories260K-f32.gguf \
#   QUANTIZE=/path/to/llama-quantize tools/mk_stories_r4.sh
#
# llama-quantize must be the one matching its libs. If it needs a newer
# libstdc++ (very common: builds with gcc 15, box has gcc 11 runtime), set:
#   QUANTIZE_LIBS=$HOME/gcc-15/lib64:$HOME/ik_llama.cpp/build/src:$HOME/ik_llama.cpp/build/ggml/src

set -eu

OUT="./stories260K-r4.gguf"
F32="${F32:-$HOME/llm/stories260K-f32.gguf}"
QUANTIZE="${QUANTIZE:-$HOME/iklcpp-2026-10-28/llama-quantize}"
TYPE_R4="${TYPE_R4:-IQ4_NL_R4}"

while getopts "o:f:q:t:h" opt; do
    case "$opt" in
        o) OUT="$OPTARG" ;;
        f) F32="$OPTARG" ;;
        q) QUANTIZE="$OPTARG" ;;
        t) TYPE_R4="$OPTARG" ;;
        h) sed -n '2,60p' "$0"; exit 0 ;;
        *) exit 2 ;;
    esac
done

[ -f "$F32" ]      || { echo "error: f32 source not found: $F32" >&2; exit 2; }
[ -x "$QUANTIZE" ] || { echo "error: llama-quantize not found: $QUANTIZE" >&2; exit 2; }

# llama-quantize often needs a newer libstdc++ than the system one (it is built
# with a modern gcc but the box has an older runtime). Allow the caller to point
# at it; this is the single most common reason the tool "does not work".
if [ -n "${QUANTIZE_LIBS:-}" ]; then
    LD_LIBRARY_PATH="$QUANTIZE_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
fi

echo "source : $F32"
echo "quant  : $QUANTIZE"
[ -n "${QUANTIZE_LIBS:-}" ] && echo "libs   : $QUANTIZE_LIBS"
echo "out    : $OUT"
echo "R4 type: $TYPE_R4  (attention projections only)"

# The exact mix that both (a) decodes as English and (b) admits R4 on the
# 32-aligned tensors. See the header comment for why each line is what it is.
exec "$QUANTIZE" \
    --custom-q "attn_q=${TYPE_R4},attn_k=${TYPE_R4},attn_v=${TYPE_R4},attn_output=${TYPE_R4},ffn_gate=Q6_0,ffn_up=Q5_0,ffn_down=F16" \
    --token-embedding-type Q4_1 \
    --output-tensor-type Q8_0 \
    "$F32" "$OUT" Q4_0 4
