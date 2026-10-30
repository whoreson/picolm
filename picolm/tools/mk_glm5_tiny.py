#!/usr/bin/env python3
"""Generate a tiny GLM-5.3-Flash (glm5next) GGUF for PicoLM development.

Mirrors the real model's tensor layout with scaled-down dimensions.
All shapes given in GGUF ne[] order (ne0 = rows = output dim for 2D weights);
numpy arrays are allocated with the reversed shape since gguf-py reverses.
"""
import argparse, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'jani', 'ik_old', 'gguf-py'))
from gguf import GGUFWriter

p = argparse.ArgumentParser()
p.add_argument('-o', '--output', default='glm5_tiny.gguf')
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()
rng = np.random.default_rng(args.seed)

# ---- dimensions ----
N_EMBD      = 256
N_HEAD      = 4          # MLA q heads
KDA_DIM     = 128        # kda.head_dim
N_KDA_HEADS = N_HEAD     # ik derives ssm_dt_rank = n_head, d_inner = kda_dim*N_HEAD
D_INNER     = N_KDA_HEADS * KDA_DIM   # 512 = n_head * kda_dim
Q_LORA      = 64
KV_LORA     = 128
MLA_HD      = 64
IDX_NH      = 4
IDX_DIM     = 32
IDX_TOPK    = 8
KPOOL       = 4
HC          = 4
SINKHORN    = 20
HC_EPS      = 1e-6
N_EXP       = 16
N_EXP_USED  = 4
N_FF_EXP    = 256
N_FF_DENSE  = 512
N_LAYERS    = 9          # 8 trunk + 1 NextN
N_DENSE_LEAD= 3
VOCAB       = 257   # 256 raw byte tokens + 1 </s>
CONV_K      = 4
LOWER_BOUND = -5.0

def is_mla(il):
    """MLA at il%4==3 for trunk layers; the NextN layer (last) is always MLA."""
    return il % 4 == 3 or il == N_LAYERS - 1
def is_moe(il): return il >= N_DENSE_LEAD

def rq(*ne, scale=0.05):
    """Random F32 weight. Args are GGUF ne[] dims (ne0=innermost/row_len/input,
    ne1=rows/output). numpy shape is reversed(ne) since gguf-py reverses numpy
    shape back to GGUF ne order. So rq(4096, 8192) -> attn_q [8192,4096] display.
    Call convention: rq(input_dim, output_dim) for 2D weights."""
    shape = tuple(reversed(ne))
    return (rng.uniform(-1, 1, shape) * scale).astype(np.float32)

def ones(*ne):
    """1-D or uniform constant tensor; args are the element counts per dim."""
    return np.ones(ne if len(ne) > 1 else ne[0], dtype=np.float32)

def zeros(*ne):
    return np.zeros(ne if len(ne) > 1 else ne[0], dtype=np.float32)

w = GGUFWriter(args.output, 'glm5next')
w.add_uint32('glm5next.block_count', N_LAYERS)
w.add_uint32('glm5next.context_length', 4096)
w.add_uint32('glm5next.embedding_length', N_EMBD)
w.add_uint32('glm5next.feed_forward_length', N_FF_DENSE)
w.add_uint32('glm5next.attention.head_count', N_HEAD)
w.add_float32('glm5next.attention.layer_norm_rms_epsilon', 1e-5)
w.add_uint32('glm5next.expert_count', N_EXP)
w.add_uint32('glm5next.expert_used_count', N_EXP_USED)
w.add_uint32('glm5next.expert_gating_func', 2)
w.add_uint32('glm5next.vocab_size', VOCAB)
w.add_array('glm5next.attention.head_count_kv', [1 if is_mla(i) else 0 for i in range(N_LAYERS)])
# head_count_kv: 0 = KDA, 1 = MLA (per-layer array); NextN (last) is MLA
w.add_uint32('glm5next.attention.q_lora_rank', Q_LORA)
w.add_uint32('glm5next.attention.kv_lora_rank', KV_LORA)
w.add_uint32('glm5next.rope.dimension_count', 0)
w.add_uint32('glm5next.attention.key_length', D_INNER)
w.add_uint32('glm5next.attention.value_length', D_INNER)
w.add_uint32('glm5next.attention.key_length_mla', MLA_HD)
w.add_uint32('glm5next.attention.value_length_mla', MLA_HD)
w.add_float32('glm5next.attention.layer_norm_epsilon', 1e-6)
w.add_uint32('glm5next.ssm.conv_kernel', CONV_K)
w.add_uint32('glm5next.kda.head_dim', KDA_DIM)
w.add_float32('glm5next.kda.gate_lower_bound', LOWER_BOUND)
w.add_uint32('glm5next.attention.indexer.head_count', IDX_NH)
w.add_uint32('glm5next.attention.indexer.key_length', IDX_DIM)
w.add_uint32('glm5next.attention.indexer.top_k', IDX_TOPK)
w.add_uint32('glm5next.attention.indexer.kpool', KPOOL)
w.add_uint32('glm5next.hyper_connection.count', HC)
w.add_uint32('glm5next.hyper_connection.sinkhorn_iterations', SINKHORN)
w.add_float32('glm5next.hyper_connection.epsilon', HC_EPS)
w.add_uint32('glm5next.expert_feed_forward_length', N_FF_EXP)
w.add_uint32('glm5next.expert_shared_feed_forward_length', N_FF_EXP)
w.add_uint32('glm5next.expert_shared_count', 1)
w.add_uint32('glm5next.leading_dense_block_count', N_DENSE_LEAD)
w.add_float32('glm5next.expert_weights_scale', 2.5)
w.add_bool('glm5next.expert_weights_norm', True)
w.add_array('glm5next.swiglu_clamp_exp', [10.0]*N_LAYERS)
w.add_array('glm5next.swiglu_clamp_shexp', [10.0]*N_LAYERS)
w.add_uint32('glm5next.nextn_predict_layers', 1)
# tokenizer
w.add_string('tokenizer.ggml.model', 'gpt2')
w.add_string('tokenizer.ggml.pre', 'glm4')
w.add_uint32('tokenizer.ggml.eos_token_id', VOCAB-1)
w.add_uint32('tokenizer.ggml.padding_token_id', VOCAB-1)
w.add_uint32('tokenizer.ggml.bos_token_id', 0)
# GPT-2 byte-level vocab: every single byte maps to one printable codepoint, so
# arbitrary text tokenizes (one token per byte) without any merge rules. This is
# what makes the tiny model usable as an oracle for both ik and PicoLM.
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
w.add_array('tokenizer.ggml.tokens', _BYTES + ['</s>'])
w.add_array('tokenizer.ggml.token_type', [1] * 256 + [3])
# ik/llama.cpp refuses to load a gpt2-vocab model with an empty merges array
# ("cannot find tokenizer merges in model file"), so emit one harmless merge of
# two byte tokens. It is never selected for normal text.
w.add_array('tokenizer.ggml.merges', [f'{_BYTES[0]} {_BYTES[1]}'])

def T(name, a):
    w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32))

T('token_embd.weight', rq(N_EMBD, VOCAB))
T('output_norm.weight', ones(N_EMBD))
T('output.weight', rq(N_EMBD, VOCAB))

for il in range(N_LAYERS):
    p_ = f'blk.{il}.'
    mla = is_mla(il)
    nextn = (il == N_LAYERS - 1)
    moe = il >= N_DENSE_LEAD
    HC_MIX = (2 + HC) * HC

    if not nextn:
        T(p_+'hc_attn_fn.weight', rq(N_EMBD*HC, HC_MIX))
        T(p_+'hc_attn_base.weight', zeros(HC_MIX))
        T(p_+'hc_attn_scale.weight', np.array([1.0, 1.0, 1.0], dtype=np.float32))
        T(p_+'hc_ffn_fn.weight', rq(N_EMBD*HC, HC_MIX))
        T(p_+'hc_ffn_base.weight', zeros(HC_MIX))
        T(p_+'hc_ffn_scale.weight', np.array([1.0, 1.0, 1.0], dtype=np.float32))
    T(p_+'attn_norm.weight', ones(N_EMBD))
    T(p_+'ffn_norm.weight', ones(N_EMBD))

    if mla:
        T(p_+'attn_q_a.weight', rq(N_EMBD, Q_LORA))
        T(p_+'attn_q_a_norm.weight', ones(Q_LORA))
        T(p_+'attn_q_b.weight', rq(Q_LORA, N_HEAD*MLA_HD))
        T(p_+'attn_kv_a_mqa.weight', rq(N_EMBD, KV_LORA))
        T(p_+'attn_kv_a_norm.weight', ones(KV_LORA))
        T(p_+'attn_k_b.weight', rq(MLA_HD, KV_LORA, N_HEAD))
        T(p_+'attn_v_b.weight', rq(KV_LORA, MLA_HD, N_HEAD))
        T(p_+'attn_output.weight', rq(N_EMBD, N_HEAD*MLA_HD))
        T(p_+'indexer.k_norm.weight', ones(IDX_DIM))
        T(p_+'indexer.k_norm.bias', zeros(IDX_DIM))
        T(p_+'indexer.proj.weight', rq(N_EMBD, IDX_NH))
        T(p_+'indexer.attn_k.weight', rq(N_EMBD, IDX_DIM))
        T(p_+'indexer.attn_q_b.weight', rq(Q_LORA, IDX_NH*IDX_DIM))
        T(p_+'indexer_compressor_gate.weight', rq(N_EMBD, IDX_DIM))
        T(p_+'indexer_compressor_ape.weight', zeros(KPOOL, IDX_DIM))
    else:
        T(p_+'attn_q.weight', rq(N_EMBD, D_INNER))
        T(p_+'attn_k.weight', rq(N_EMBD, D_INNER))
        T(p_+'attn_v.weight', rq(N_EMBD, D_INNER))
        T(p_+'ssm_conv1d_q.weight', rq(CONV_K, 1, D_INNER)*0.1)
        T(p_+'ssm_conv1d_k.weight', rq(CONV_K, 1, D_INNER)*0.1)
        T(p_+'ssm_conv1d_v.weight', rq(CONV_K, 1, D_INNER)*0.1)
        T(p_+'ssm_f_a.weight', rq(N_EMBD, KDA_DIM))
        T(p_+'ssm_f_b.weight', rq(KDA_DIM, D_INNER))
        T(p_+'ssm_g_a.weight', rq(N_EMBD, KDA_DIM))
        T(p_+'ssm_g_b.weight', rq(KDA_DIM, D_INNER))
        T(p_+'ssm_beta.weight', rq(N_EMBD, N_KDA_HEADS))
        T(p_+'ssm_a', rq(N_KDA_HEADS, scale=1.0) - 1.0)  # negative: -exp(A_log)
        T(p_+'ssm_dt.bias', zeros(D_INNER))
        T(p_+'ssm_norm.weight', ones(KDA_DIM))
        T(p_+'attn_output.weight', rq(D_INNER, N_EMBD))

    if nextn:
        T(p_+'nextn.eh_proj.weight', rq(2*N_EMBD, N_EMBD))
        T(p_+'nextn.enorm.weight', ones(N_EMBD))
        T(p_+'nextn.hnorm.weight', ones(N_EMBD))
        T(p_+'nextn.shared_head_norm.weight', ones(N_EMBD))

    if not moe:
        T(p_+'ffn_gate.weight', rq(N_EMBD, N_FF_DENSE))
        T(p_+'ffn_up.weight', rq(N_EMBD, N_FF_DENSE))
        T(p_+'ffn_down.weight', rq(N_FF_DENSE, N_EMBD))
    else:
        T(p_+'ffn_gate_inp.weight', rq(N_EMBD, N_EXP)*0.01)
        T(p_+'exp_probs_b.bias', zeros(N_EXP))
        T(p_+'ffn_gate_exps.weight', rq(N_EMBD, N_FF_EXP, N_EXP))
        T(p_+'ffn_up_exps.weight', rq(N_EMBD, N_FF_EXP, N_EXP))
        T(p_+'ffn_down_exps.weight', rq(N_FF_EXP, N_EMBD, N_EXP))
        T(p_+'ffn_gate_shexp.weight', rq(N_FF_EXP, N_EMBD))
        T(p_+'ffn_up_shexp.weight', rq(N_FF_EXP, N_EMBD))
        T(p_+'ffn_down_shexp.weight', rq(N_FF_EXP, N_EMBD))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f'wrote {args.output}: {N_LAYERS} layers ({N_DENSE_LEAD} dense KDA + '
      f'{N_LAYERS-N_DENSE_LEAD-1} MoE + 1 NextN)')
