#!/usr/bin/env python3
"""Generate a tiny DeepSeek V3 (deepseek2) GGUF for PicoLM correctness work.

Mirrors the real model's layout with scaled-down dimensions:
  - MLA with low-rank Q (attn_q_a -> norm -> attn_q_b), COMBINED attn_kv_b
    ([kv_lora, NH*(nope+value)]) -- exercises PicoLM's ds2_build_kv_b
    decomposition at load time (ik's llm_prepare_mla does the same).
  - partial RoPE (rope.dimension_count < key_length), pairwise (NORM type)
  - dense lead layers, then MoE (sigmoid routing, selection bias, weight
    norm + scale 2.5, one shared expert) -- identical to glm5_moe_ffn
  - NO rope scaling (plain RoPE): keeps the ik/PicoLM comparison clean of
    YaRN conventions; the real-model YaRN path is verified separately.

All weights F32 and random; byte-level GPT-2 vocab so arbitrary text
tokenizes identically in ik and PicoLM (one token per byte).

Run ik with:  llama-cli -m deepseek2_tiny.gguf -p "..." -n 8  (CPU)
Run PicoLM with the same -p / --tokens and compare logits/text, or use
PICOLM_DS2_DUMP_POS / PICOLM_DUMP_LOGITS for numeric comparison.
"""
import argparse, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'jani', 'ik_old', 'gguf-py'))
from gguf import GGUFWriter

p = argparse.ArgumentParser()
p.add_argument('-o', '--output', default='deepseek2_tiny.gguf')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--n-head', type=int, default=4, help='attention heads (real V3: 128)')
p.add_argument('--kv-lora', type=int, default=128, help='MLA latent width (real: 512)')
p.add_argument('--nope', type=int, default=64, help='non-RoPE head dims (real: 128)')
p.add_argument('--rope-dims', type=int, default=32, help='RoPE head dims (real: 64)')
p.add_argument('--q-lora', type=int, default=64, help='q_lora_rank (real: 1536)')
p.add_argument('--n-embd', type=int, default=256, help='embedding length (real: 7168)')
p.add_argument('--layers', type=int, default=9, help='block_count (real: 61)')
p.add_argument('--dense-lead', type=int, default=3, help='leading dense blocks (real: 3)')
p.add_argument('--n-exp', type=int, default=16, help='expert_count (real: 256)')
p.add_argument('--n-exp-used', type=int, default=4, help='expert_used_count (real: 8)')
p.add_argument('--n-ff-exp', type=int, default=128, help='expert FFN width (real: 2048)')
p.add_argument('--n-ff-dense', type=int, default=0, help='dense FFN width (0 = 2*n_embd)')
args = p.parse_args()
rng = np.random.default_rng(args.seed)

N_EMBD   = args.n_embd
N_HEAD   = args.n_head
KV_LORA  = args.kv_lora
NOPE     = args.nope
ROPE     = args.rope_dims
KHD      = NOPE + ROPE          # attention.key_length (q head width)
VHD      = NOPE                 # attention.value_length (real: == nope)
Q_LORA   = args.q_lora
N_EXP    = args.n_exp
N_USED   = args.n_exp_used
FF_EXP   = args.n_ff_exp
FF_DENSE = args.n_ff_dense or 2 * N_EMBD
N_LAYERS = args.layers
DENSE    = args.dense_lead
VOCAB    = 257
CTX      = 4096

def rq(*ne, scale=0.05):
    """Random F32 weight; args are GGUF ne[] dims (ne0 = input/row length).
    Call convention: rq(input_dim, output_dim)."""
    shape = tuple(reversed(ne))
    return (rng.uniform(-1, 1, shape) * scale).astype(np.float32)

def ones(*ne):
    return np.ones(ne if len(ne) > 1 else ne[0], dtype=np.float32)

def zeros(*ne):
    return np.zeros(ne if len(ne) > 1 else ne[0], dtype=np.float32)

w = GGUFWriter(args.output, 'deepseek2')
w.add_uint32('deepseek2.block_count', N_LAYERS)
w.add_uint32('deepseek2.context_length', CTX)
w.add_uint32('deepseek2.embedding_length', N_EMBD)
w.add_uint32('deepseek2.feed_forward_length', FF_DENSE)
w.add_uint32('deepseek2.attention.head_count', N_HEAD)
w.add_uint32('deepseek2.attention.head_count_kv', N_HEAD)  # like real V3 GGUF:
# head_count_kv == head_count makes ik load deepseek2 natively (its
# mainline-GGUF fixup only fires for head_count_kv == 1).
w.add_uint32('deepseek2.attention.key_length', KHD)      # NOPE + ROPE
w.add_uint32('deepseek2.attention.value_length', VHD)
w.add_uint32('deepseek2.attention.q_lora_rank', Q_LORA)
w.add_uint32('deepseek2.attention.kv_lora_rank', KV_LORA)
w.add_uint32('deepseek2.rope.dimension_count', ROPE)
w.add_float32('deepseek2.rope.freq_base', 10000.0)
# deliberately NO rope.scaling.* keys: plain RoPE, freq_scale=1
w.add_float32('deepseek2.attention.layer_norm_rms_epsilon', 1e-5)
w.add_uint32('deepseek2.expert_count', N_EXP)
w.add_uint32('deepseek2.expert_used_count', N_USED)
w.add_uint32('deepseek2.expert_shared_count', 1)
w.add_uint32('deepseek2.expert_feed_forward_length', FF_EXP)
w.add_float32('deepseek2.expert_weights_scale', 2.5)
w.add_bool('deepseek2.expert_weights_norm', True)
w.add_uint32('deepseek2.expert_gating_func', 2)          # sigmoid
w.add_uint32('deepseek2.leading_dense_block_count', DENSE)
w.add_uint32('deepseek2.vocab_size', VOCAB)
# tokenizer: byte-level GPT-2 vocab (same scheme as mk_glm5_tiny.py) so text
# tokenizes to one token per byte in both engines with zero merge ambiguity.
def _gpt2_byte_chars():
    keep = list(range(ord('!'), ord('~') + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    out = [None] * 256
    for b in keep:
        out[b] = chr(b)
    n = 0
    for b in range(256):
        if out[b] is None:
            out[b] = chr(256 + n)
            n += 1
    return out
_BYTES = _gpt2_byte_chars()
w.add_string('tokenizer.ggml.model', 'gpt2')
w.add_string('tokenizer.ggml.pre', 'deepseek-v3')
w.add_uint32('tokenizer.ggml.eos_token_id', VOCAB - 1)
w.add_uint32('tokenizer.ggml.padding_token_id', VOCAB - 1)
w.add_uint32('tokenizer.ggml.bos_token_id', 0)
w.add_array('tokenizer.ggml.tokens', _BYTES + ['</s>'])
w.add_array('tokenizer.ggml.token_type', [1] * 256 + [3])
w.add_array('tokenizer.ggml.merges', [f'{_BYTES[0]} {_BYTES[1]}'])

def T(name, a):
    w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32))

T('token_embd.weight', rq(N_EMBD, VOCAB))
T('output_norm.weight', ones(N_EMBD))
T('output.weight', rq(N_EMBD, VOCAB))

PER_HEAD = NOPE + VHD                    # attn_kv_b column stride per head
for il in range(N_LAYERS):
    p_ = f'blk.{il}.'
    moe = il >= DENSE

    T(p_ + 'attn_norm.weight', ones(N_EMBD))
    T(p_ + 'ffn_norm.weight', ones(N_EMBD))

    # MLA with low-rank Q
    T(p_ + 'attn_q_a.weight', rq(N_EMBD, Q_LORA))
    T(p_ + 'attn_q_a_norm.weight', ones(Q_LORA))
    T(p_ + 'attn_q_b.weight', rq(Q_LORA, N_HEAD * KHD))
    T(p_ + 'attn_kv_a_mqa.weight', rq(N_EMBD, KV_LORA + ROPE))
    T(p_ + 'attn_kv_a_norm.weight', ones(KV_LORA))
    # COMBINED attn_kv_b: ne = {KV_LORA, NH*(NOPE+VHD)} -- the layout the real
    # V3 GGUF ships and the one ds2_build_kv_b must decompose at load time.
    T(p_ + 'attn_kv_b.weight', rq(KV_LORA, N_HEAD * PER_HEAD))
    T(p_ + 'attn_output.weight', rq(N_HEAD * VHD, N_EMBD))

    if not moe:
        T(p_ + 'ffn_gate.weight', rq(N_EMBD, FF_DENSE))
        T(p_ + 'ffn_up.weight', rq(N_EMBD, FF_DENSE))
        T(p_ + 'ffn_down.weight', rq(FF_DENSE, N_EMBD))
    else:
        T(p_ + 'ffn_gate_inp.weight', rq(N_EMBD, N_EXP) * 0.01)
        T(p_ + 'exp_probs_b.bias', zeros(N_EXP))
        T(p_ + 'ffn_gate_exps.weight', rq(N_EMBD, FF_EXP, N_EXP))
        T(p_ + 'ffn_up_exps.weight', rq(N_EMBD, FF_EXP, N_EXP))
        T(p_ + 'ffn_down_exps.weight', rq(FF_EXP, N_EMBD, N_EXP))
        T(p_ + 'ffn_gate_shexp.weight', rq(N_EMBD, FF_EXP))
        T(p_ + 'ffn_up_shexp.weight', rq(N_EMBD, FF_EXP))
        T(p_ + 'ffn_down_shexp.weight', rq(FF_EXP, N_EMBD))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f'wrote {args.output}: {N_LAYERS} layers ({DENSE} dense + '
      f'{N_LAYERS - DENSE} MoE), embd={N_EMBD} heads={N_HEAD} '
      f'kv_lora={KV_LORA} nope={NOPE} rope={ROPE} experts={N_EXP}(top{N_USED})')
