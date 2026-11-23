#ifndef BACKEND_GPU_ATTN_FA3_CUH
#define BACKEND_GPU_ATTN_FA3_CUH
/* ---------------------------------------------------------------------------
 * FA3: flash-attention prefill with BOTH matmuls on tensor cores and the O
 * accumulator in registers. Replaces the earlier FA3 scaffold, whose P@V was a
 * scalar loop.
 *
 * Shape handled: head_dim in {64,128,256}; KV cache F16 / Q8_0 / Q4_0, K and V
 * independently; causal, optional sliding window (n_swa). NVIDIA sm_80+.
 *
 * Structure (the standard FlashAttention-2 layout, specialised for hd256):
 *   block  = one query head x 64 query rows, 4 warps, 16 rows per warp
 *   S      = Q K^T      mma.m16n8k16, A=Q from smem via ldmatrix,
 *                       B=K from smem via ldmatrix (K is stored [key][dim], which
 *                       is exactly the "col" B layout mma wants -> no .trans)
 *   P      = softmax(S) in registers. The S accumulator fragment layout is
 *                       IDENTICAL to the A-operand layout of the next mma, so P
 *                       goes straight from the S registers into P@V with no
 *                       shared-memory round trip.
 *   O     += P V        B=V via ldmatrix.TRANS (V is [key][dim] = row-major B)
 *   O      lives in mma accumulators: 16 rows x HD / 32 lanes = HD/2 floats per
 *          lane (128 at hd256). That is what lets hd256 fit: no smem accumulator.
 *
 * Shared memory tiles are 16-byte-chunk XOR-swizzled (chunk ^= row & 7). Without
 * it every ldmatrix reads 8 rows at a 512-byte stride = the same 8 banks, an
 * 8-way conflict on every load.
 *
 * Pipeline: K/V tiles are fetched with cp.async one tile ahead. F16 KV lands
 * directly in the (double-buffered) swizzled tile. Quantised KV lands RAW in a
 * small double-buffered staging area and is then dequantised smem->smem into one
 * fp16 tile. The dequant uses packed half2 arithmetic and is BIT-IDENTICAL to the
 * scalar kv_row_dequant() path: (n-8)*d and int8*d are each one exactly-rounded
 * fp16 multiply of an exactly-representable integer by the fp16 scale, which is
 * what (float)(v)*d followed by round-to-fp16 produces.
 *
 * NOT bit-identical to the CPU reference (P is rounded to fp16, summation order
 * differs). Validate against llama.cpp / the scalar warpgrp kernel.
 *
 * The same source is compiled by tests/fa3_emu.cpp with the PTX primitives
 * replaced by C++ emulations, so the real kernel logic can be executed and
 * checked without a GPU (-DPICOLM_FA3_EMU).
 * ------------------------------------------------------------------------- */

#if defined(PICOLM_FA3_EMU)
#  define FA3_ENABLED 1
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
#  define FA3_ENABLED 1
#else
#  define FA3_ENABLED 0
#endif

#define FA3_THREADS 128      /* 4 warps */
#define FA3_BM      64       /* query rows per block (16 per warp) */

#ifndef PICOLM_FA3_EMU
#  define FA3_HD  __host__ __device__
#else
#  define FA3_HD
#endif

/* bytes of one head row of raw quantised KV: (hd/32) blocks of 34 (q8_0) / 18 (q4_0) */
/* Bytes of one quantised head row. t is kv_cache_type_t numbering:
 *   1 = Q8_0 (34 B/block), 2 = Q4_0 (18), 3 = TQ3 (14), 4 = TQ4 (18).
 * 0 (F16) returns 0 on purpose -- F16 rows are not "raw", the kernel copies
 * them straight into the swizzled tile. */
static inline FA3_HD int fa3_raw_row_bytes(int hd, int t) {
    switch (t) {
        case 1:  return (hd / 32) * 34;   /* Q8_0 */
        case 2:  return (hd / 32) * 18;   /* Q4_0 */
        case 3:  return (hd / 32) * 14;   /* TQ3  */
        case 4:  return (hd / 32) * 18;   /* TQ4  */
        default: return 0;
    }
}
/* Bytes per 32-value block, same numbering. */
static inline FA3_HD int fa3_block_bytes(int t) {
    switch (t) {
        case 1:  return 34;
        case 2:  return 18;
        case 3:  return 14;
        case 4:  return 18;
        default: return 0;
    }
}
static inline FA3_HD size_t fa3_up16(size_t x) { return (x + 15) & ~(size_t)15; }

/* Dynamic shared memory layout, in bytes. Identical on host and device. */
struct fa3_layout { size_t q, kt, kr, vt, vr, total; };
static inline FA3_HD fa3_layout fa3_make_layout(int hd, int bn, int kt, int vt) {
    fa3_layout L; size_t o = 0;
    L.q  = o; o += fa3_up16((size_t)FA3_BM * hd * 2);
    L.kt = o; o += fa3_up16((size_t)(kt == 0 ? 2 : 1) * bn * hd * 2);
    L.kr = o; o += kt == 0 ? 0 : fa3_up16((size_t)2 * bn * fa3_raw_row_bytes(hd, kt));
    L.vt = o; o += fa3_up16((size_t)(vt == 0 ? 2 : 1) * bn * hd * 2);
    L.vr = o; o += vt == 0 ? 0 : fa3_up16((size_t)2 * bn * fa3_raw_row_bytes(hd, vt));
    L.total = o;
    return L;
}

#if FA3_ENABLED && !defined(PICOLM_FA3_EMU)
/* ---- PTX primitives (sm_80+) --------------------------------------------- */
__device__ __forceinline__ uint32_t fa3_smem_u32(const void *p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void fa3_ldsm_x4(uint32_t r[4], const void *lane_ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(fa3_smem_u32(lane_ptr)) : "memory");
}
__device__ __forceinline__ void fa3_ldsm_x4_trans(uint32_t r[4], const void *lane_ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(fa3_smem_u32(lane_ptr)) : "memory");
}
/* d += A(16x16,row) * B(16x8,col), fp16 inputs, fp32 accumulate */
__device__ __forceinline__ void fa3_mma(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void fa3_cp16(void *dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(fa3_smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void fa3_cp4(void *dst, const void *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"(fa3_smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void fa3_cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void fa3_cp_wait1()  { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
/* pack two fp32 into one fp16x2 register: lo -> bits 0..15, hi -> bits 16..31 */
__device__ __forceinline__ uint32_t fa3_pack2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ uint32_t fa3_h2sub(uint32_t a, uint32_t b) {
    uint32_t r; asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b)); return r;
}
__device__ __forceinline__ uint32_t fa3_h2mul(uint32_t a, uint32_t b) {
    uint32_t r; asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b)); return r;
}
__device__ __forceinline__ float fa3_shfl_xor(float v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }
__device__ __forceinline__ int   fa3_any(int p) { return __any_sync(0xffffffffu, p); }
#  define FA3_DYN_SMEM_PTR(name) \
     extern __shared__ __align__(16) uint8_t name##_raw[]; uint8_t *name = name##_raw
#endif /* FA3_ENABLED && !EMU */

#if FA3_ENABLED
/* 16-byte chunk index of (row, chunk) in a [rows][CH] swizzled tile */
#define FA3_SW(row, chunk, CH) ((row) * (CH) + ((chunk) ^ ((row) & 7)))

__device__ static inline uint32_t fa3_ld16(const uint8_t *p) { return *(const uint16_t *)p; }

/* fp16 bit pattern -> fp32. Only the TQ dequant needs this (Q8_0/Q4_0 use the
 * packed half2 trick). backend_gpu_common.cuh already provides it for both
 * host and device passes; tests/fa3_emu.cpp supplies its own. */
#ifndef PICOLM_FA3_EMU
static inline FA3_HD float fa3_h2f(uint16_t h) { return gpu_fp16_to_fp32(h); }
#endif

/* ---- TurboQuant tables -------------------------------------------------
 * Duplicated from backend_gpu_dequant.cuh rather than included: this header is
 * also compiled standalone by tests/fa3_emu.cpp, which supplies its own
 * emulated primitives. They MUST match quant.c's tq3_codebook / tq4_codebook /
 * tq3_signs bit for bit, since the same bytes have to decode identically
 * through this path, the scalar kv_row_dequant() path and the CPU. */
/* ---- TQ3/TQ4 dequant ----
 * The codebook and sign tables are NOT duplicated here. backend_gpu_dequant.cuh
 * already defines tq3_codebook_dev / tq4_codebook_dev / tq3_signs_dev, and for
 * the nvcc build this header is included from backend_gpu_kernels.cu after
 * backend_gpu_kernels.cuh (which pulls that in), so they are already in scope.
 *
 * Only the emulator, which compiles this header standalone, needs its own
 * copies -- and it gets them for the host pass by aliasing the device names. */
#ifdef PICOLM_FA3_EMU
static const float tq3_codebook_dev[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f
};
static const float tq4_codebook_dev[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};
static const float tq3_signs_dev[32] = {
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
   -1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,
   -1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f,
   -1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f
};
#  define FA3_TQ_FN static inline
#else
#  define FA3_TQ_FN static __device__ inline
#endif

/* Unpack 8 3-bit indices (3 bytes) / 8 4-bit indices (4 bytes). */
static inline FA3_HD void fa3_tq_unpack3(uint8_t *dst, const uint8_t *src) {
    dst[0] = src[0] & 7;
    dst[1] = (src[0] >> 3) & 7;
    dst[2] = (uint8_t)(((src[0] >> 6) | (src[1] << 2)) & 7);
    dst[3] = (src[1] >> 1) & 7;
    dst[4] = (src[1] >> 4) & 7;
    dst[5] = (uint8_t)(((src[1] >> 7) | (src[2] << 1)) & 7);
    dst[6] = (src[2] >> 2) & 7;
    dst[7] = (src[2] >> 5) & 7;
}
static inline FA3_HD void fa3_tq_unpack4(uint8_t *dst, const uint8_t *src) {
    dst[0] = src[0] & 0xF;
    dst[1] = src[0] >> 4;
    dst[2] = src[1] & 0xF;
    dst[3] = src[1] >> 4;
    dst[4] = src[2] & 0xF;
    dst[5] = src[2] >> 4;
    dst[6] = src[3] & 0xF;
    dst[7] = src[3] >> 4;
}

/* TQ3/TQ4 block -> 16 fp16x2 words: codebook lookup in the rotated domain,
 * 32-point inverse WHT, /sqrt(32), undo the sign flips, times the block RMS.
 * Same order as tq3_dequant_block() in backend_gpu_dequant.cuh. */
FA3_TQ_FN void fa3_dequant_tq(const uint8_t *blk, int t, uint32_t out[16]) {
    float rot[32];
    if (t == 3) {
        for (int g = 0; g < 4; g++) {
            uint8_t idx[8];
            fa3_tq_unpack3(idx, blk + 2 + g * 3);
            for (int k = 0; k < 8; k++) rot[g * 8 + k] = tq3_codebook_dev[idx[k]];
        }
    } else {
        for (int g = 0; g < 4; g++) {
            uint8_t idx[8];
            fa3_tq_unpack4(idx, blk + 2 + g * 4);
            for (int k = 0; k < 8; k++) rot[g * 8 + k] = tq4_codebook_dev[idx[k]];
        }
    }
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                const float a = rot[j], b = rot[j + step];
                rot[j] = a + b;
                rot[j + step] = a - b;
            }
        }
    }
    const float norm = 0.176776695f;                 /* 1/sqrt(32) */
    const float scale = fa3_h2f((uint16_t)(blk[0] | ((uint16_t)blk[1] << 8)));
    for (int k = 0; k < 16; k++) {
        const float lo = rot[2 * k] * norm * tq3_signs_dev[2 * k] * scale;
        const float hi = rot[2 * k + 1] * norm * tq3_signs_dev[2 * k + 1] * scale;
        out[k] = fa3_pack2(lo, hi);
    }
}

/* Dequantise one 32-element block (raw bytes at `blk`) into 16 fp16x2 words
 * (elements 2k, 2k+1 in word k). Bit-identical to kv_row_dequant()+fp32->fp16. */
__device__ static inline void fa3_dequant_block(const uint8_t *blk, int t, uint32_t out[16]) {
    if (t == 3 || t == 4) { fa3_dequant_tq(blk, t, out); return; }   /* TQ3 / TQ4 */
    const uint32_t d  = fa3_ld16(blk);
    const uint32_t d2 = d | (d << 16);
    if (t == 2) {                                   /* Q4_0: 16 bytes of nibbles */
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t w  = fa3_ld16(blk + 2 + 2 * j);        /* qs[2j], qs[2j+1] */
            /* 0x6400 | n  ==  fp16(1024 + n) exactly for n < 1024 */
            const uint32_t lo = ((w & 0x000Fu) | ((w & 0x0F00u) << 8)) | 0x64006400u;
            const uint32_t hi = (((w >> 4) & 0x000Fu) | (((w >> 12) & 0x000Fu) << 16)) | 0x64006400u;
            out[j]     = fa3_h2mul(fa3_h2sub(lo, 0x64086408u), d2);   /* 1032 = 1024 + 8 */
            out[8 + j] = fa3_h2mul(fa3_h2sub(hi, 0x64086408u), d2);
        }
    } else {                                        /* Q8_0: 32 int8 */
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const uint32_t w = fa3_ld16(blk + 2 + 2 * j) ^ 0x8080u;   /* signed -> biased unsigned */
            const uint32_t v = ((w & 0x00FFu) | ((w & 0xFF00u) << 8)) | 0x64006400u; /* 1024 + b + 128 */
            out[j] = fa3_h2mul(fa3_h2sub(v, 0x64806480u), d2);        /* 1152 = 1024 + 128 */
        }
    }
}




template <int HD, int BN>
__global__ void __launch_bounds__(FA3_THREADS, 1)
picolm_gpu_attention_prefill_fa3_kernel(
        float *__restrict__ out, const float *__restrict__ q_dev,
        const uint8_t *__restrict__ kv_k, const uint8_t *__restrict__ kv_v,
        int layer_ordinal, int start_pos, int n_tokens,
        int n_heads, int n_kv_heads, int max_seq_len,
        size_t k_pos_stride, size_t k_head_stride,
        size_t v_pos_stride, size_t v_head_stride,
        int n_swa, int kt, int vt)
{
    static_assert(HD == 64 || HD == 128 || HD == 256, "head_dim");
    static_assert(BN % 16 == 0, "BN must be a multiple of 16");
    constexpr int CH   = HD / 8;      /* 16-byte chunks per fp16 row */
    constexpr int NT_S = BN / 8;      /* n8 tiles of S per warp */
    constexpr int NT_O = HD / 8;      /* n8 tiles of O per warp */

    FA3_DYN_SMEM_PTR(sm);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tg = lane & 3;
    const int q_h = blockIdx.x, qtile = blockIdx.y;
    const int kv_h = q_h / (n_heads / n_kv_heads);

    const fa3_layout L = fa3_make_layout(HD, BN, kt, vt);
    uint4 *Q4  = (uint4 *)(sm + L.q);
    uint4 *K4  = (uint4 *)(sm + L.kt);
    uint4 *V4  = (uint4 *)(sm + L.vt);
    uint8_t *Kr = sm + L.kr;
    uint8_t *Vr = sm + L.vr;
    const int krow = fa3_raw_row_bytes(HD, kt), vrow = fa3_raw_row_bytes(HD, vt);
    const int kbsz = fa3_block_bytes(kt), vbsz = fa3_block_bytes(vt);

    /* ---- block geometry: ALL of this is block-uniform ------------------- */
    const int kv_total = start_pos + n_tokens;
    int kv_end = start_pos + (qtile + 1) * FA3_BM;
    if (kv_end > kv_total) kv_end = kv_total;
    const int q_blk0 = qtile * FA3_BM;
    int k_lo = 0;
    if (n_swa > 0) {
        k_lo = start_pos + q_blk0 - n_swa + 1;
        if (k_lo < 0) k_lo = 0;
        k_lo = (k_lo / BN) * BN;
    }
    const int n_tiles = (kv_end - k_lo + BN - 1) / BN;

    const uint8_t *kbase = kv_k + (size_t)layer_ordinal * max_seq_len * k_pos_stride + (size_t)kv_h * k_head_stride;
    const uint8_t *vbase = kv_v + (size_t)layer_ordinal * max_seq_len * v_pos_stride + (size_t)kv_h * v_head_stride;

    /* ---- async tile issue ------------------------------------------------ */
    auto issue = [&](int jt) {
        const int t0 = k_lo + jt * BN, buf = jt & 1;
        if (kt == 0) {
            for (int u = tid; u < BN * CH; u += FA3_THREADS) {
                const int r = u / CH, c = u % CH, key = t0 + r;
                if (key < kv_end)
                    fa3_cp16(&K4[buf * BN * CH + FA3_SW(r, c, CH)], kbase + (size_t)key * k_pos_stride + c * 16);
            }
        } else {
            const int wpr = krow / 4;
            for (int u = tid; u < BN * wpr; u += FA3_THREADS) {
                const int r = u / wpr, w = u % wpr, key = t0 + r;
                if (key < kv_end)
                    fa3_cp4(Kr + buf * BN * krow + r * krow + w * 4, kbase + (size_t)key * k_pos_stride + w * 4);
            }
        }
        if (vt == 0) {
            for (int u = tid; u < BN * CH; u += FA3_THREADS) {
                const int r = u / CH, c = u % CH, key = t0 + r;
                if (key < kv_end)
                    fa3_cp16(&V4[buf * BN * CH + FA3_SW(r, c, CH)], vbase + (size_t)key * v_pos_stride + c * 16);
            }
        } else {
            const int wpr = vrow / 4;
            for (int u = tid; u < BN * wpr; u += FA3_THREADS) {
                const int r = u / wpr, w = u % wpr, key = t0 + r;
                if (key < kv_end)
                    fa3_cp4(Vr + buf * BN * vrow + r * vrow + w * 4, vbase + (size_t)key * v_pos_stride + w * 4);
            }
        }
    };

    /* ---- prologue: start tile 0 loading, then convert Q while it is in flight */
    issue(0);
    fa3_cp_commit();

    for (int u = tid; u < FA3_BM * CH; u += FA3_THREADS) {
        const int r = u / CH, c = u % CH, tq = q_blk0 + r;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (tq < n_tokens) {
            const float4 *s = (const float4 *)(q_dev + ((size_t)tq * n_heads + q_h) * HD + c * 8);
            const float4 a = s[0], b = s[1];
            v.x = fa3_pack2(a.x, a.y); v.y = fa3_pack2(a.z, a.w);
            v.z = fa3_pack2(b.x, b.y); v.w = fa3_pack2(b.z, b.w);
        }
        Q4[FA3_SW(r, c, CH)] = v;
    }

    /* ---- per-warp state --------------------------------------------------- */
    const int qb_w = q_blk0 + warp * 16;          /* first token of this warp */
    const bool warp_alive = qb_w < n_tokens;
    const int pos_w0 = start_pos + qb_w;          /* position of this warp's first row */
    const int pos0 = pos_w0 + g, pos1 = pos0 + 8; /* positions of the 2 rows this lane holds */
    const float scale2 = (1.0f / sqrtf((float)HD)) * 1.4426950408889634f;   /* log2(e)/sqrt(d) */

    float O[NT_O][4];
#pragma unroll
    for (int i = 0; i < NT_O; i++) { O[i][0] = O[i][1] = O[i][2] = O[i][3] = 0.f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    /* ldmatrix lane addressing (constant across tiles) */
    const int lj = lane >> 3, lr = lane & 7;
    const int q_row   = warp * 16 + (lj & 1) * 8 + lr;      /* A-frag of Q: matrix j -> rows (j&1)*8, chunk +(j>>1) */
    const int k_row_o = (lj >> 1) * 8 + lr;                  /* B-frag of K: matrix j -> ntile +(j>>1), chunk +(j&1) */
    const int v_row_o = (lj & 1) * 8 + lr;                   /* B-frag of V: matrix j -> keys +(j&1)*8, chunk +(j>>1) */

    for (int jt = 0; jt < n_tiles; jt++) {
        const int t0 = k_lo + jt * BN, buf = jt & 1;
        if (jt + 1 < n_tiles) issue(jt + 1);
        fa3_cp_commit();                 /* always commit so wait_group counting stays uniform */
        fa3_cp_wait1();                  /* this thread's copies for tile jt have landed */

        /* rows past the end of the cache (last tile only): zero them. F16 tiles
         * are zeroed here; quantised tiles are zeroed by the dequant pass. A NaN
         * pattern in unused smem would otherwise reach O through 0 * NaN. */
        if (kt == 0) {
            for (int u = tid; u < BN * CH; u += FA3_THREADS) {
                const int r = u / CH, c = u % CH;
                if (t0 + r >= kv_end) K4[buf * BN * CH + FA3_SW(r, c, CH)] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        if (vt == 0) {
            for (int u = tid; u < BN * CH; u += FA3_THREADS) {
                const int r = u / CH, c = u % CH;
                if (t0 + r >= kv_end) V4[buf * BN * CH + FA3_SW(r, c, CH)] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();                 /* every thread's copies (and zero fills) are visible */

        if (kt != 0 || vt != 0) {        /* quantised operand(s): dequant raw -> fp16 tile */
            constexpr int NB = HD / 32;
            for (int bi = tid; bi < BN * NB; bi += FA3_THREADS) {
                const int r = bi / NB, b = bi % NB, key = t0 + r;
                if (kt != 0) {
                    uint32_t o[16];
                    if (key < kv_end) fa3_dequant_block(Kr + buf * BN * krow + r * krow + b * kbsz, kt, o);
                    else { for (int i = 0; i < 16; i++) o[i] = 0u; }
#pragma unroll
                    for (int u = 0; u < 4; u++)
                        K4[FA3_SW(r, b * 4 + u, CH)] = make_uint4(o[4*u], o[4*u+1], o[4*u+2], o[4*u+3]);
                }
                if (vt != 0) {
                    uint32_t o[16];
                    if (key < kv_end) fa3_dequant_block(Vr + buf * BN * vrow + r * vrow + b * vbsz, vt, o);
                    else { for (int i = 0; i < 16; i++) o[i] = 0u; }
#pragma unroll
                    for (int u = 0; u < 4; u++)
                        V4[FA3_SW(r, b * 4 + u, CH)] = make_uint4(o[4*u], o[4*u+1], o[4*u+2], o[4*u+3]);
                }
            }
            __syncthreads();
        }

        /* this warp has work in this tile only if some key is visible to some of its rows */
        const bool live = warp_alive && (t0 <= pos_w0 + 15)
                       && (n_swa <= 0 || t0 + BN - 1 >= pos_w0 - n_swa + 1);
        if (live) {
            const uint4 *Kt = K4 + (kt == 0 ? buf * BN * CH : 0);
            const uint4 *Vt = V4 + (vt == 0 ? buf * BN * CH : 0);

            /* ---- S = Q K^T ---------------------------------------------- */
            float S[NT_S][4];
#pragma unroll
            for (int i = 0; i < NT_S; i++) { S[i][0] = S[i][1] = S[i][2] = S[i][3] = 0.f; }
#pragma unroll
            for (int ks = 0; ks < HD / 16; ks++) {
                uint32_t qa[4];
                fa3_ldsm_x4(qa, &Q4[q_row * CH + ((2 * ks + (lj >> 1)) ^ lr)]);
#pragma unroll
                for (int np = 0; np < NT_S / 2; np++) {
                    uint32_t kb[4];
                    const int kr = np * 16 + k_row_o;
                    fa3_ldsm_x4(kb, &Kt[kr * CH + ((2 * ks + (lj & 1)) ^ lr)]);
                    fa3_mma(S[2 * np],     qa, kb[0], kb[1]);
                    fa3_mma(S[2 * np + 1], qa, kb[2], kb[3]);
                }
            }

            /* ---- online softmax (scores in log2 domain) ------------------- */
            const bool need_mask = (t0 + BN - 1 > pos_w0)
                                || (n_swa > 0 && t0 < pos_w0 + 15 - n_swa + 1);
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < NT_S; nt++) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int kv = t0 + nt * 8 + tg * 2 + e;
                    float s0 = S[nt][e] * scale2, s1 = S[nt][2 + e] * scale2;
                    if (need_mask) {
                        if (kv > pos0 || (n_swa > 0 && kv < pos0 - n_swa + 1)) s0 = -INFINITY;
                        if (kv > pos1 || (n_swa > 0 && kv < pos1 - n_swa + 1)) s1 = -INFINITY;
                    }
                    S[nt][e] = s0; S[nt][2 + e] = s1;
                    mx0 = fmaxf(mx0, s0); mx1 = fmaxf(mx1, s1);
                }
            }
            mx0 = fmaxf(mx0, fa3_shfl_xor(mx0, 1)); mx0 = fmaxf(mx0, fa3_shfl_xor(mx0, 2));
            mx1 = fmaxf(mx1, fa3_shfl_xor(mx1, 1)); mx1 = fmaxf(mx1, fa3_shfl_xor(mx1, 2));
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
            /* a fully masked row keeps m = -inf; use 0 as the shift so exp2 yields 0, not NaN */
            const float mu0 = (mn0 == -INFINITY) ? 0.f : mn0;
            const float mu1 = (mn1 == -INFINITY) ? 0.f : mn1;
            const float al0 = exp2f(m0 - mu0), al1 = exp2f(m1 - mu1);
            if (fa3_any((mn0 != m0) || (mn1 != m1))) {      /* skip 128 FMULs when no max moved */
#pragma unroll
                for (int i = 0; i < NT_O; i++) {
                    O[i][0] *= al0; O[i][1] *= al0; O[i][2] *= al1; O[i][3] *= al1;
                }
            }
            m0 = mn0; m1 = mn1;
            float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
            for (int nt = 0; nt < NT_S; nt++) {
                S[nt][0] = exp2f(S[nt][0] - mu0); S[nt][1] = exp2f(S[nt][1] - mu0);
                S[nt][2] = exp2f(S[nt][2] - mu1); S[nt][3] = exp2f(S[nt][3] - mu1);
                rs0 += S[nt][0] + S[nt][1];
                rs1 += S[nt][2] + S[nt][3];
            }
            l0 = l0 * al0 + rs0;          /* per-thread partial; reduced across the quad at the end */
            l1 = l1 * al1 + rs1;

            /* ---- O += P V ---------------------------------------------------
             * S[2*ks2][0..1] / [2..3] and S[2*ks2+1][0..1] / [2..3] are exactly
             * the a0 / a1 / a2 / a3 registers of the m16n8k16 A operand. */
#pragma unroll
            for (int ks2 = 0; ks2 < BN / 16; ks2++) {
                uint32_t pa[4];
                pa[0] = fa3_pack2(S[2 * ks2][0],     S[2 * ks2][1]);
                pa[1] = fa3_pack2(S[2 * ks2][2],     S[2 * ks2][3]);
                pa[2] = fa3_pack2(S[2 * ks2 + 1][0], S[2 * ks2 + 1][1]);
                pa[3] = fa3_pack2(S[2 * ks2 + 1][2], S[2 * ks2 + 1][3]);
                const int vr = ks2 * 16 + v_row_o;
#pragma unroll
                for (int dp = 0; dp < NT_O / 2; dp++) {
                    uint32_t vb[4];
                    fa3_ldsm_x4_trans(vb, &Vt[vr * CH + ((2 * dp + (lj >> 1)) ^ lr)]);
                    fa3_mma(O[2 * dp],     pa, vb[0], vb[1]);
                    fa3_mma(O[2 * dp + 1], pa, vb[2], vb[3]);
                }
            }
        }
        __syncthreads();      /* all warps done with this tile before it is overwritten */
    }

    /* ---- epilogue: reduce the row sums across the quad, normalise, store --- */
    l0 += fa3_shfl_xor(l0, 1); l0 += fa3_shfl_xor(l0, 2);
    l1 += fa3_shfl_xor(l1, 1); l1 += fa3_shfl_xor(l1, 2);
    const float inv0 = l0 > 0.f ? 1.0f / l0 : 0.f, inv1 = l1 > 0.f ? 1.0f / l1 : 0.f;
    const int t_a = qb_w + g, t_b = t_a + 8;
    if (t_a < n_tokens) {
        float *o = out + ((size_t)t_a * n_heads + q_h) * HD + tg * 2;
#pragma unroll
        for (int i = 0; i < NT_O; i++)
            *(float2 *)(o + i * 8) = make_float2(O[i][0] * inv0, O[i][1] * inv0);
    }
    if (t_b < n_tokens) {
        float *o = out + ((size_t)t_b * n_heads + q_h) * HD + tg * 2;
#pragma unroll
        for (int i = 0; i < NT_O; i++)
            *(float2 *)(o + i * 8) = make_float2(O[i][2] * inv1, O[i][3] * inv1);
    }
}
#else  /* !FA3_ENABLED: host pass of nvcc, or a pre-sm_80 device pass. Empty stub so the
        * launcher in backend_gpu_kernels.cu still links; it is never launched there
        * (the dispatcher requires has_imma, i.e. sm_80+). */
template <int HD, int BN>
__global__ void picolm_gpu_attention_prefill_fa3_kernel(
        float *__restrict__, const float *__restrict__,
        const uint8_t *__restrict__, const uint8_t *__restrict__,
        int, int, int, int, int, int, size_t, size_t, size_t, size_t, int, int, int) {}
#endif /* FA3_ENABLED */

#endif /* BACKEND_GPU_ATTN_FA3_CUH */
