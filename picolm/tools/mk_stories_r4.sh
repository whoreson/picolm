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
# stories260K-stuffed.gguf: 12 distinct tensor types in ~290 KB, verified
# coherent by BOTH PicoLM and ik_llama's llama-cli.
#
# The 24 n=64 slots (attn_q/k/v/output, ffn_gate/up across 4 layers) cycle
# through ten types, giving each 2-3 tensors:
#
#     q4_0 (2)  q4_1 (3)  q5_0 (3)  q5_1 (2)  q6_0 (2)  q8_0 (3)
#     iq4_nl (3)  q4_0_r8 (2)  q8_0_r8 (2)  iq4_nl_r4 (2)
#
# plus ffn_down=F16, token_embd=Q4_0, output=Q8_0, *_norm=F32.
#
# ffn_down is F16 and must stay F16: its n is 172, which is not a multiple of
# 32, so ANY block quantizer covers only 160 of the 172 input channels and
# drops 12. Measured: ffn_down=q4_0/q4_1/q6_0 all produce garbage in ik AND
# PicoLM, while F16 produces correct text. This is a shape property of
# stories260K, not a property of any particular quant type -- it cost a while
# to realise, because the symptom (garbage) looked type-specific.
#
# Each type needs several tensors to be worth testing. An earlier 11-type build
# had q4_1 on a SINGLE tensor, and deleting Q4_1's kernel from the dispatch
# table did not change the output by one byte. At 2-3 tensors per type,
# removing Q4_1, Q4_0_R8 or Q8_0_R8 each changes the output and is caught.
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
# Ten types over the 24 equally-shaped slots, three per layer rotating so no
# slot is always the same type.
exec "$QUANTIZE" \
    --custom-q "blk\.0\.attn_q=q4_0,blk\.0\.attn_k=q4_1,blk\.0\.attn_v=q5_0,blk\.0\.attn_output=q5_1,blk\.0\.ffn_gate=q6_0,blk\.0\.ffn_up=q8_0,blk\.1\.attn_q=iq4_nl,blk\.1\.attn_k=q4_0_r8,blk\.1\.attn_v=q8_0_r8,blk\.1\.attn_output=iq4_nl_r4,blk\.1\.ffn_gate=q4_0,blk\.1\.ffn_up=q4_1,blk\.2\.attn_q=q5_0,blk\.2\.attn_v=q5_1,blk\.2\.attn_output=q6_0,blk\.2\.ffn_gate=q8_0,blk\.2\.ffn_up=iq4_nl,blk\.2\.attn_k=q4_0_r8,blk\.3\.attn_q=q8_0_r8,blk\.3\.attn_k=iq4_nl_r4,blk\.3\.attn_v=q4_0,blk\.3\.attn_output=q4_1,blk\.3\.ffn_gate=q5_0,blk\.3\.ffn_up=q5_1,ffn_down=F16" \
    --token-embedding-type Q4_0 \
    --output-tensor-type Q8_0 \
    "$F32" "$OUT" Q4_0 4
