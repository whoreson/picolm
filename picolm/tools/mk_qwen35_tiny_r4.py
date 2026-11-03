#!/usr/bin/env python3
"""Generate a tiny Qwen3.5/3.6-style hybrid (SSM + full-attention) GGUF with
row-interleaved (type 220, "IQ4_NL_R4") weights -- a *dummy oracle model*.

WHY THIS EXISTS
---------------
The CI smoke model (stories260K-Q2_K) contains only f32/f16/q4_0/q8_0 tensors,
so no CI prompt can ever exercise an R4/R8/SSM kernel. Two real bugs shipped
through that gap (see notes/picolm/qwen3.6_iq4_nl_r4_garbage_DEBUG_JOURNAL.txt):

  BUG A  vec_dot() on an interleaved type only read row 0 of a block group, and
         SSM alpha/beta walked per-head rows by a single row stride -> wrong
         row for 3 of every 4 heads -> NaN at layer 8 -> all-zero logits.
  BUG B  the R8 prefill repack decoded IQ4_NL_R4 nibbles with the Q4_0_4_4
         byte-contiguous layout instead of type 220's four-window spread.

Both are *layout* bugs, and both are invisible to "does it produce fluent
English". This generator produces a model small enough to run in seconds whose
tensors have exactly the shapes that triggered them, with the R4 blocks written
byte-by-byte from the format spec so the expected layout is reviewable as code.

WHAT IT GIVES YOU
-----------------
  * ssm_alpha / ssm_beta as type 220 with n_v_heads rows -> the per-head row
    walk that BUG A broke (vec_dot_row must pick row gh % 4).
  * ffn_gate/up/down and attn_qkv as type 220 at sizes large enough for the
    decode-once R8 prefill GEMM (ncols >= PICOLM_Q4_R8_MIN_N, nrows % 8 == 0)
    -> BUG B.
  * A real SSM recurrence (conv1d + gated norm), so NaN propagation is visible
    exactly as it was in the 27B.

Because the weights are random, the model does not produce English and must
NOT be judged on fluency. Judge it by comparing two builds against each other
(different quant type, or SIMD vs scalar) on the same fixed prompt: the
diagnostic is *agreement*, not readability.

USAGE
-----
  python3 tools/mk_qwen35_tiny_r4.py -o /tmp/qwen35_tiny_r4.gguf
  python3 tools/mk_qwen35_tiny_r4.py -o /tmp/qwen35_tiny_r4.gguf --seed 7

  # then, with two builds (or two quant types):
  ./picolm /tmp/qwen35_tiny_r4.gguf -p "Once upon a" -n 16 -t 0
  PICOLM_Q4_R8=0 ./picolm /tmp/qwen35_tiny_r4.gguf -p "Once upon a" -n 16 -t 0
  # both must print the SAME tokens. A disagreement (or NaN) is a bug.
"""
import argparse
import os
import sys

import numpy as np

try:
    from gguf import GGUFWriter
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
except ImportError:
    sys.exit("error: gguf-py not found.  pip install gguf  (or set PYTHONPATH "
             "to a llama.cpp gguf-py checkout)")

# gguf-py has no entry for type 220 (IQ4_NL_R4 is a PicoLM/ik repack format;
# upstream GGMLQuantizationType is a closed enum we cannot extend). Register a
# minimal stand-in that carries the numeric type id and enough of the interface
# for GGUFWriter: comparison, hashing, and a GGML_QUANT_SIZES entry so it can
# compute element/byte counts. 220 = 32 values per 72-byte block.
class _RawType:
    def __init__(self, value: int, name: str):
        self.value = value
        self.name = name
    def __int__(self):        return self.value
    def __index__(self):      return self.value
    def __eq__(self, o):      return int(self) == int(o) if isinstance(o, (int, _RawType)) else NotImplemented
    def __hash__(self):       return hash(self.value)
    def __repr__(self):       return f"<raw quant type {self.name}:{self.value}>"

IQ4_NL_R4 = _RawType(220, "IQ4_NL_R4")
GGML_QUANT_SIZES[IQ4_NL_R4] = (32, 72)

# ---------------------------------------------------------------- dimensions
# Everything is small enough to load and run in well under a second, while
# keeping the *shapes* that matter: n_v_heads not a multiple of 4-only, ffn
# rows divisible by 8, and at least one attention layer among the SSM ones.
N_EMBD     = 256
N_FFN      = 512
N_HEAD     = 4
N_KV_HEAD  = 2
HEAD_DIM   = 64
N_LAYERS   = 8
N_SSM      = 6                      # layers 0..5 SSM, 6..7 full attention
SSM_D_CONV = 4
SSM_D_STATE = 16
SSM_N_GROUP = 2
SSM_DT_RANK = 16                    # n_v_heads. Must divide SSM_D_INNER -- the
                                    # code does head_v_dim = inner / n_v_heads,
                                    # which TRUNCATES silently if it does not
                                    # divide. 16/256 -> head_v_dim 16, and 16
                                    # rows is 4 R4 groups so 12 of the 16 heads
                                    # sit at a row offset that is not 0 mod 4.
SSM_D_INNER = N_HEAD * HEAD_DIM     # 256
VOCAB      = 257                    # 256 byte tokens + </s>
ROPE_BASE  = 10000.0

assert SSM_DT_RANK % 4 == 0, "needs whole R4 groups"
assert SSM_D_INNER % SSM_DT_RANK == 0, \
    "head_v_dim = inner / n_v_heads truncates unless this divides"
assert N_FFN % 8 == 0, "ffn rows must be a multiple of 8 for the R8 GEMM"

QK = 32                             # IQ4_NL values per row per R4 block
IVQ = 16                            # kvalues_iq4nl
KVALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                     1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)

rng = np.random.default_rng(42)


# ------------------------------------------------------------------ quantize
def iq4nl_indices(vals, amax):
    """Map 32 float values to IQ4_NL LUT indices (0..15), given the block amax.

    kvalues_iq4nl is a non-linear 16-entry table whose first half is negative
    and second half positive, so the nearest entry already encodes the sign.
    Index 0..7 -> magnitudes 127..10 (negative), 8..15 -> 1..113 (positive).
    """
    ref = KVALUES.astype(np.int32)
    scaled = (vals / amax) * np.abs(ref).max() if amax else np.zeros_like(vals)
    # compare against |LUT| and then pick the signed entry on that magnitude
    mags = np.abs(ref)
    k = np.argmin(np.abs(np.abs(scaled)[:, None] - mags[None, :]), axis=1)
    return np.where(scaled < 0, k, k + 8).astype(np.uint8)


def fp16(x):
    return np.float16(x)


# ------------------------------------------------- IQ4_NL_R4 (GGUF type 220)
# block: uint16 d[4] then 64 qs bytes.  Four logical rows per block.
#
# Row r's 32 values come from FOUR 4-byte windows spread across the whole 64-byte
# qs[], with a permuted output order.  This is the layout that
# dequantize_row_iq4_nl_r4_single() implements and that r8a_repack_r4_iq4nl()
# got wrong (it used the Q4_0_4_4 byte-contiguous layout instead):
#
#   qs[4*r + i +  0] -> value[i+ 0] (low nibble), value[i+ 8] (high nibble)
#   qs[4*r + i + 16] -> value[i+16] (low),        value[i+24] (high)
#   qs[4*r + i + 32] -> value[i+ 4] (low),        value[i+12] (high)
#   qs[4*r + i + 48] -> value[i+20] (low),        value[i+28] (high)
#
# Written out explicitly here so the expected layout is reviewable as code.
def r4_block_from_rows(rows_idx, rows_d):
    """rows_idx: (4, 32) uint8 IQ4_NL indices. rows_d: (4,) fp16 scales."""
    assert rows_idx.shape == (4, QK)
    qs = np.zeros(64, dtype=np.uint8)
    for r in range(4):
        v = rows_idx[r]
        for i in range(4):
            qs[4 * r + i +  0] = (v[i +  0] & 0xF) | ((v[i +  8] & 0xF) << 4)
            qs[4 * r + i + 16] = (v[i + 16] & 0xF) | ((v[i + 24] & 0xF) << 4)
            qs[4 * r + i + 32] = (v[i +  4] & 0xF) | ((v[i + 12] & 0xF) << 4)
            qs[4 * r + i + 48] = (v[i + 20] & 0xF) | ((v[i + 28] & 0xF) << 4)
    d = np.asarray(rows_d, dtype=np.float16)
    return d.tobytes() + qs.tobytes()


def make_r4_tensor(n_rows, k, scale=0.05, seed=None):
    """Random IQ4_NL_R4 weights: n_rows logical rows of k values.

    Returns bytes laid out as ceil(n_rows/4) groups of (k/32) 72-byte blocks,
    which is what gguf_type_row_size(220, k) and gguf_type_row_ptr() expect.
    """
    assert n_rows % 4 == 0 and k % QK == 0
    r = np.random.default_rng(seed) if seed is not None else rng
    nb = k // QK
    out = bytearray()
    for g in range(n_rows // 4):
        for ib in range(nb):
            rows_idx = np.zeros((4, QK), dtype=np.uint8)
            rows_d = np.zeros(4, dtype=np.float16)
            for rr in range(4):
                vals = (r.uniform(-1, 1, QK) * scale).astype(np.float32)
                amax = float(np.max(np.abs(vals))) or scale
                rows_idx[rr] = iq4nl_indices(vals, amax)
                rows_d[rr] = fp16(amax / 113.0)
            out += r4_block_from_rows(rows_idx, rows_d)
    return bytes(out)


def make_iq4_nl_plain(n_rows, k, scale=0.05):
    """Plain IQ4_NL (type 20) rows: 32 values per 18-byte block (fp16 d + 16 qs)."""
    assert k % QK == 0
    nb = k // QK
    out = bytearray()
    for _ in range(n_rows):
        for _ in range(nb):
            vals = (rng.uniform(-1, 1, QK) * scale).astype(np.float32)
            amax = float(np.max(np.abs(vals))) or scale
            idx = iq4nl_indices(vals, amax)
            d = fp16(amax / 113.0)
            lo, hi = idx[:16], idx[16:]
            out += d.tobytes() + bytes((lo | (hi << 4)).astype(np.uint8).tobytes())
    return bytes(out)


def make_f32(*ne, scale=0.05):
    return (rng.uniform(-1, 1, tuple(reversed(ne))) * scale).astype(np.float32)


# ------------------------------------------------------------------- writer
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="qwen35_tiny_r4.gguf")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    global rng
    rng = np.random.default_rng(args.seed)

    w = GGUFWriter(args.output, "qwen35")
    A = "qwen35."
    w.add_uint32(A + "block_count", N_LAYERS)
    w.add_uint32(A + "context_length", 512)
    w.add_uint32(A + "embedding_length", N_EMBD)
    w.add_uint32(A + "feed_forward_length", N_FFN)
    w.add_uint32(A + "attention.head_count", N_HEAD)
    w.add_uint32(A + "attention.head_count_kv", N_KV_HEAD)
    w.add_uint32(A + "attention.key_length", HEAD_DIM)
    w.add_uint32(A + "attention.value_length", HEAD_DIM)
    w.add_float32(A + "attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32(A + "rope.freq_base", ROPE_BASE)
    w.add_uint32(A + "rope.dimension_count", HEAD_DIM)
    w.add_uint32(A + "vocab_size", VOCAB)
    # SSM knobs -- these set has_ssm and the conv / recurrence shapes.
    w.add_uint32(A + "ssm.conv_kernel", SSM_D_CONV)
    w.add_uint32(A + "ssm.state_size", SSM_D_STATE)
    w.add_uint32(A + "ssm.group_count", SSM_N_GROUP)
    w.add_uint32(A + "ssm.time_step_rank", SSM_DT_RANK)
    w.add_uint32(A + "ssm.inner_size", SSM_D_INNER)

    # Byte-level GPT-2 vocab so arbitrary text tokenizes one token per byte.
    def gpt2_byte_chars():
        keep = list(range(ord('!'), ord('~') + 1)) + list(range(0xA1, 0xAD)) + \
               list(range(0xAE, 0x100))
        out = [None] * 256
        for b in keep:
            out[b] = chr(b)
        n = 0
        for b in range(256):
            if out[b] is None:
                out[b] = chr(256 + n)
                n += 1
        return out

    B = gpt2_byte_chars()
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "qwen2")
    w.add_uint32("tokenizer.ggml.eos_token_id", VOCAB - 1)
    w.add_uint32("tokenizer.ggml.bos_token_id", 0)
    w.add_array("tokenizer.ggml.tokens", B + ["</s>"])
    w.add_array("tokenizer.ggml.token_type", [1] * 256 + [3])
    w.add_array("tokenizer.ggml.merges", [f"{B[0]} {B[1]}"])

    # ---- tensor helpers -------------------------------------------------
    def T(name, arr):
        """Plain F32 tensor."""
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))

    def T_q(name, raw, dtype, k, n_rows):
        """Pre-packed quant tensor.

        Two things must both be right:
          - the BYTE BUFFER (layout is spelled out in make_r4_tensor above);
          - the LOGICAL SHAPE. gguf-py needs ne[] so PicoLM can derive this
            tensor's n and d. Passing the bytes without a shape makes them a
            rank-1 tensor, the loader then has no n/d, and the model silently
            misbehaves -- so assert the size matches what the shape implies.
        """
        if dtype == IQ4_NL_R4:
            # one 72-byte block holds 4 logical rows x 32 values
            expect = (n_rows // 4) * (k // 32) * 72
        else:
            blk_elems, blk_bytes = GGML_QUANT_SIZES[dtype]
            expect = (n_rows * k // blk_elems) * blk_bytes
        assert len(raw) == expect, (name, len(raw), expect, dtype)
        # int8 (not uint8) so gguf-py does NOT try to reinterpret the byte
        # buffer as a quant shape; raw_shape carries the logical shape instead.
        # raw_shape is in NUMPY order and gguf-py reverses it to get GGUF ne[]:
        # we want ne = {k, n_rows} (ne[0] = row length -> matmul n=k,
        # ne[1] = row count -> matmul d=n_rows), so pass (n_rows, k) here.
        w.add_tensor(name, np.frombuffer(raw, dtype=np.int8),
                     raw_dtype=dtype, raw_shape=(n_rows, k))

    def T_r4(name, n_rows, k, seed=None):
        T_q(name, make_r4_tensor(n_rows, k, seed=seed), IQ4_NL_R4, k, n_rows)

    q_dim    = N_HEAD * HEAD_DIM
    kv_dim   = N_KV_HEAD * HEAD_DIM
    conv_dim = 2 * SSM_D_STATE * SSM_N_GROUP + SSM_D_INNER
    value_dim = SSM_D_INNER

    T_q("token_embd.weight", make_iq4_nl_plain(VOCAB, N_EMBD),
        GGMLQuantizationType.IQ4_NL, N_EMBD, VOCAB)
    T("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    T("output.weight", make_f32(N_EMBD, VOCAB))

    for l in range(N_LAYERS):
        p = f"blk.{l}."
        is_attn = l >= N_LAYERS - N_ATTN
        T(p + "attn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        T(p + "post_attention_norm.weight", np.ones(N_EMBD, dtype=np.float32))

        if is_attn:
            T(p + "attn_q.weight", make_f32(N_EMBD, 2 * q_dim))   # Q + gate
            T(p + "attn_k.weight", make_f32(N_EMBD, kv_dim))
            T(p + "attn_v.weight", make_f32(N_EMBD, kv_dim))
            T(p + "attn_output.weight", make_f32(q_dim, N_EMBD))
            T(p + "attn_q_norm.weight", np.ones(HEAD_DIM, dtype=np.float32))
            T(p + "attn_k_norm.weight", np.ones(HEAD_DIM, dtype=np.float32))
        else:
            # QKV projection as type 220, sized to engage the decode-once R8
            # prefill GEMM (nrows % 8 == 0).
            T_r4(p + "attn_qkv.weight", conv_dim, N_EMBD, seed=args.seed + l)
            T_r4(p + "attn_gate.weight", value_dim, N_EMBD, seed=args.seed + l + 50)
            T(p + "ssm_conv1d.weight", make_f32(SSM_D_CONV, conv_dim, scale=0.3))
            T(p + "ssm_a", rng.uniform(-2.0, -0.5, SSM_DT_RANK).astype(np.float32))
            T(p + "ssm_dt.bias", rng.uniform(-0.5, 0.5, SSM_DT_RANK).astype(np.float32))
            # THE BUG A SHAPE: per-head alpha/beta rows, n_v_heads = SSM_DT_RANK,
            # stored as type 220. 12 rows = 3 R4 groups, so 9 of the 12 heads sit
            # at a row offset that is NOT 0 mod 4 -- exactly the case a
            # vec_dot that always reads row 0 gets wrong.
            T_r4(p + "ssm_alpha.weight", SSM_DT_RANK, N_EMBD, seed=args.seed + 100 + l)
            T_r4(p + "ssm_beta.weight",  SSM_DT_RANK, N_EMBD, seed=args.seed + 200 + l)
            T(p + "ssm_norm.weight", np.ones(SSM_D_INNER // SSM_DT_RANK, dtype=np.float32))
            T_r4(p + "ssm_out.weight", N_EMBD, SSM_D_INNER, seed=args.seed + 250 + l)

        T_r4(p + "ffn_gate.weight", N_FFN, N_EMBD, seed=args.seed + 300 + l)
        T_r4(p + "ffn_up.weight",   N_FFN, N_EMBD, seed=args.seed + 400 + l)
        T_r4(p + "ffn_down.weight", N_EMBD, N_FFN, seed=args.seed + 500 + l)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    sz = os.path.getsize(args.output)
    print(f"wrote {args.output} ({sz/1024:.1f} KiB): {N_LAYERS} layers "
          f"({N_LAYERS-N_ATTN} SSM + {N_ATTN} attn); type 220 for "
          f"ssm_alpha/ssm_beta/ssm_out/attn_qkv/attn_gate/ffn_*")


N_ATTN = 2


if __name__ == "__main__":
    main()
