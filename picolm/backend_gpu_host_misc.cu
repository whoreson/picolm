// backend_gpu_host_misc.cu - KV cache, pipeline, attention, rmsnorm, rope, residual, misc
#include "backend_gpu_kernels.cuh"

extern "C" {
__global__ void picolm_gpu_kv_pack_store_kernel(uint16_t *dst_row, const float *src, int n);
}

extern "C" int
picolm_gpu_kv_store_dev_batched_strided(int is_k, int layer_ordinal, int start_pos, int n_positions,
                                         const float *src_dev, int n_kv_heads, int head_dim,
                                         int max_seq_len, int device, int src_stride);


/* Print once per kernel type when first dispatched */
static void gpu_dispatch_print(const char *name) {
    static char seen_names[64][64] = {0};
    for (int i = 0; i < 64; i++) {
        if (seen_names[i][0] == '\0') {
            snprintf(seen_names[i], sizeof(seen_names[i]), "%s", name);
            fprintf(stderr, "[GPU] kernel: %s\n", name);
            return;
        }
        if (strcmp(seen_names[i], name) == 0) return;
    }
}

/* Static globals (were file-static in monolithic backend_gpu.cu) */
static uint16_t *g_kv_k_dev[PICOLM_GPU_MAX_DEVICES];
static uint16_t *g_kv_v_dev[PICOLM_GPU_MAX_DEVICES];
static size_t g_kv_k_cap[PICOLM_GPU_MAX_DEVICES];
static size_t g_kv_v_cap[PICOLM_GPU_MAX_DEVICES];
static float *g_verify_buf_dev[PICOLM_GPU_MAX_DEVICES];
extern "C" int
picolm_gpu_kv_alloc(size_t kv_k_bytes, size_t kv_v_bytes, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    /* Align to 256 bytes for coalesced access */
    kv_k_bytes = (kv_k_bytes + 255) & ~(size_t)255;
    kv_v_bytes = (kv_v_bytes + 255) & ~(size_t)255;

    if (g_kv_k_cap[device] < kv_k_bytes) {
        if (g_kv_k_dev[device]) gpuFree(g_kv_k_dev[device]);
        if (!gpu_ok(gpuMalloc(&g_kv_k_dev[device], kv_k_bytes), "KV K allocation")) return 0;
        /* Zero-initialize */
        gpuMemset(g_kv_k_dev[device], 0, kv_k_bytes);
        g_kv_k_cap[device] = kv_k_bytes;
    }
    if (g_kv_v_cap[device] < kv_v_bytes) {
        if (g_kv_v_dev[device]) gpuFree(g_kv_v_dev[device]);
        if (!gpu_ok(gpuMalloc(&g_kv_v_dev[device], kv_v_bytes), "KV V allocation")) return 0;
        gpuMemset(g_kv_v_dev[device], 0, kv_v_bytes);
        g_kv_v_cap[device] = kv_v_bytes;
    }

    return 1;
}

extern "C" void
picolm_gpu_kv_cache_clear(int device) {
    if (device < 0 || device >= PICOLM_GPU_MAX_DEVICES) return;
    if (g_kv_k_dev[device] && g_kv_k_cap[device])
        gpuMemset(g_kv_k_dev[device], 0, g_kv_k_cap[device]);
    if (g_kv_v_dev[device] && g_kv_v_cap[device])
        gpuMemset(g_kv_v_dev[device], 0, g_kv_v_cap[device]);
}

extern "C" void
picolm_gpu_kv_free(void) {
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        if (g_kv_k_dev[i]) { gpuFree(g_kv_k_dev[i]); g_kv_k_dev[i] = NULL; g_kv_k_cap[i] = 0; }
        if (g_kv_v_dev[i]) { gpuFree(g_kv_v_dev[i]); g_kv_v_dev[i] = NULL; g_kv_v_cap[i] = 0; }
    }
}

/* Phase 2: allocate the fixed-size device-resident pipeline buffers for
 * model_forward_gpu() decode (S=1 only). Called once at model load,
 * right after picolm_gpu_kv_alloc(), with sizes derived from model
 * config. Idempotent: safe to call again with the same sizes (no-op if
 * ctx->pipe_ready already set for this device -- these buffers never
 * need to grow, unlike reserve()-based scratch, since decode is always
 * S=1). Returns 1 on success. */
extern "C" int
picolm_gpu_pipeline_alloc(int dim, int q_dim, int kv_dim, int ffn_hidden, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (ctx->pipe_ready) return 1;

    size_t db = (size_t)dim * sizeof(float);
    size_t qb = (size_t)q_dim * sizeof(float);
    size_t kvb = (size_t)kv_dim * sizeof(float);
    size_t fb = (size_t)ffn_hidden * sizeof(float);

    int ok = 1;
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_x, db), "pipe_x alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_xb, db), "pipe_xb alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_q, qb), "pipe_q alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_k, kvb), "pipe_k alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_v, kvb), "pipe_v alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_attn_out, qb), "pipe_attn_out alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_ffn_norm, db), "pipe_ffn_norm alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_gate, fb), "pipe_gate alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_up, fb), "pipe_up alloc");
    if (!ok) return 0;

    ctx->pipe_ready = 1;
    return 1;
}

/* Allocate SSM pipeline buffers for hybrid SSM+attention layers.
 * Called from model_load after SSM eligibility passes. */
extern "C" int
picolm_gpu_ssm_pipeline_alloc(int conv_dim, int ssm_d_inner, int n_v_heads, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (ctx->ssm_ready) return 1;

    size_t conv_b = (size_t)conv_dim * sizeof(float);
    size_t inner_b = (size_t)ssm_d_inner * sizeof(float);
    size_t heads_b = (size_t)n_v_heads * sizeof(float);

    int ok = 1;
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_qkv_raw, conv_b), "ssm_qkv_raw alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_conv_out, conv_b), "ssm_conv_out alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_xb2, inner_b), "ssm_xb2 alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_xb2_remap, inner_b), "ssm_xb2_remap alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_v_remap, inner_b), "ssm_v_remap alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_alpha_raw, heads_b), "ssm_alpha_raw alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_beta_raw, heads_b), "ssm_beta_raw alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_gate_exp, heads_b), "ssm_gate_exp alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_beta, heads_b), "ssm_beta alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_output, inner_b), "ssm_output alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->ssm_final_output, inner_b), "ssm_final_output alloc");
    if (!ok) return 0;

    ctx->ssm_ready = 1;
    return 1;
}

/* Allocate prefill batch buffers: [max_seq_len][max_stride] for S>1 pipeline
 * xb_stride = max(q_dim, conv_dim, dim) to accommodate all uses:
 * - attention: RMSNorm output + QKV projections (q_dim stride)
 * - SSM prefill: attn_qkv output (conv_dim = 2*d_state*n_k + ssm_d_inner)
 * - residual add: dim stride */
extern "C" int
picolm_gpu_pipeline_batch_alloc(int dim, int q_dim, int kv_dim, int ffn_hidden,
                                 int xb_stride, int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (ctx->pipe_b_ready) return 1;

    size_t bsz = (size_t)max_seq_len;
    size_t db = bsz * dim * sizeof(float);
    size_t qb = bsz * q_dim * sizeof(float);
    size_t xb = bsz * xb_stride * sizeof(float);
    size_t kvb = bsz * kv_dim * sizeof(float);
    size_t fb = bsz * ffn_hidden * sizeof(float);

    int ok = 1;
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_x_b, xb), "pipe_x_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_xb_b, xb), "pipe_xb_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_q_b, qb), "pipe_q_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_k_b, kvb), "pipe_k_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_v_b, kvb), "pipe_v_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_attn_out_b, qb), "pipe_attn_out_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_ffn_norm_b, xb), "pipe_ffn_norm_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_gate_b, fb), "pipe_gate_b alloc");
    ok &= gpu_ok(gpuMalloc(&ctx->pipe_up_b, fb), "pipe_up_b alloc");
    ok &= gpu_ok(gpuMalloc(&g_verify_buf_dev[ctx->device], 64 * sizeof(float)), "verify_buf alloc");
    if (!ok) return 0;

    ctx->pipe_b_ready = 1;
    return 1;
}

extern "C" void
picolm_gpu_pipeline_free(void) {
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!ctx->pipe_ready) continue;
        if (ctx->pipe_x) gpuFree(ctx->pipe_x);
        if (ctx->pipe_xb) gpuFree(ctx->pipe_xb);
        if (ctx->pipe_q) gpuFree(ctx->pipe_q);
        if (ctx->pipe_k) gpuFree(ctx->pipe_k);
        if (ctx->pipe_v) gpuFree(ctx->pipe_v);
        if (ctx->pipe_attn_out) gpuFree(ctx->pipe_attn_out);
        if (ctx->pipe_ffn_norm) gpuFree(ctx->pipe_ffn_norm);
        if (ctx->pipe_gate) gpuFree(ctx->pipe_gate);
        if (ctx->pipe_up) gpuFree(ctx->pipe_up);
        ctx->pipe_x = ctx->pipe_xb = ctx->pipe_q = ctx->pipe_k = ctx->pipe_v =
            ctx->pipe_attn_out = ctx->pipe_ffn_norm = ctx->pipe_gate = ctx->pipe_up = NULL;
        ctx->pipe_ready = 0;
    }
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!ctx->pipe_b_ready) continue;
        if (ctx->pipe_x_b) gpuFree(ctx->pipe_x_b);
        if (ctx->pipe_xb_b) gpuFree(ctx->pipe_xb_b);
        if (ctx->pipe_q_b) gpuFree(ctx->pipe_q_b);
        if (ctx->pipe_k_b) gpuFree(ctx->pipe_k_b);
        if (ctx->pipe_v_b) gpuFree(ctx->pipe_v_b);
        if (ctx->pipe_attn_out_b) gpuFree(ctx->pipe_attn_out_b);
        if (ctx->pipe_ffn_norm_b) gpuFree(ctx->pipe_ffn_norm_b);
        if (ctx->pipe_gate_b) gpuFree(ctx->pipe_gate_b);
        if (ctx->pipe_up_b) gpuFree(ctx->pipe_up_b);
        if (g_verify_buf_dev[ctx->device]) { gpuFree(g_verify_buf_dev[ctx->device]); g_verify_buf_dev[ctx->device] = NULL; }
        ctx->pipe_x_b = ctx->pipe_xb_b = ctx->pipe_q_b = ctx->pipe_k_b = ctx->pipe_v_b =
            ctx->pipe_attn_out_b = ctx->pipe_ffn_norm_b = ctx->pipe_gate_b = ctx->pipe_up_b = NULL;
        ctx->pipe_b_ready = 0;
    }
    /* SSM pipeline buffers */
    for (int i = 0; i < PICOLM_GPU_MAX_DEVICES; i++) {
        gpu_device_ctx_t *ctx = &g_gpu_ctx[i];
        if (!ctx->ssm_ready) continue;
        if (ctx->ssm_qkv_raw) gpuFree(ctx->ssm_qkv_raw);
        if (ctx->ssm_conv_out) gpuFree(ctx->ssm_conv_out);
        if (ctx->ssm_xb2) gpuFree(ctx->ssm_xb2);
        if (ctx->ssm_xb2_remap) gpuFree(ctx->ssm_xb2_remap);
        if (ctx->ssm_v_remap) gpuFree(ctx->ssm_v_remap);
        if (ctx->ssm_alpha_raw) gpuFree(ctx->ssm_alpha_raw);
        if (ctx->ssm_beta_raw) gpuFree(ctx->ssm_beta_raw);
        if (ctx->ssm_gate_exp) gpuFree(ctx->ssm_gate_exp);
        if (ctx->ssm_beta) gpuFree(ctx->ssm_beta);
        if (ctx->ssm_output) gpuFree(ctx->ssm_output);
        if (ctx->ssm_final_output) gpuFree(ctx->ssm_final_output);
        ctx->ssm_qkv_raw = ctx->ssm_conv_out = ctx->ssm_xb2 = ctx->ssm_xb2_remap =
            ctx->ssm_v_remap = ctx->ssm_alpha_raw = ctx->ssm_beta_raw = ctx->ssm_gate_exp =
            ctx->ssm_beta = ctx->ssm_output = ctx->ssm_final_output = NULL;
        ctx->ssm_ready = 0;
    }
}

/* Pre-allocate Q8_0 scratch buffers to a fixed maximum size.
 * This eliminates runtime reallocation races where cudaFree of a
 * scratch buffer races with a kernel still reading from it. */
extern "C" int
picolm_gpu_prealloc_q8(size_t max_xq_bytes, size_t max_xd_bytes, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!ctx->q8_xq) {
        if (!gpu_ok(gpuMalloc(&ctx->q8_xq, max_xq_bytes), "prealloc q8_xq")) return 0;
        ctx->q8_xq_cap = max_xq_bytes;
    }
    if (!ctx->q8_xd) {
        if (!gpu_ok(gpuMalloc(&ctx->q8_xd, max_xd_bytes), "prealloc q8_xd")) return 0;
        ctx->q8_xd_cap = max_xd_bytes;
    }
    return 1;
}

/* Device pointer accessors, so model.c doesn't need gpu_device_ctx_t's
 * internal layout (it's file-static to backend_gpu.cu). */
extern "C" float *picolm_gpu_pipe_x(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_x : NULL; }
extern "C" float *picolm_gpu_pipe_xb(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_xb : NULL; }
extern "C" float *picolm_gpu_pipe_q(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_q : NULL; }
extern "C" float *picolm_gpu_pipe_k(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_k : NULL; }
extern "C" float *picolm_gpu_pipe_v(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_v : NULL; }
extern "C" float *picolm_gpu_pipe_attn_out(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_attn_out : NULL; }
extern "C" float *picolm_gpu_pipe_ffn_norm(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_ffn_norm : NULL; }
extern "C" float *picolm_gpu_pipe_gate(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_gate : NULL; }
extern "C" float *picolm_gpu_pipe_up(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_up : NULL; }

extern "C" float *picolm_gpu_pipe_x_b(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_x_b : NULL; }
extern "C" float *picolm_gpu_pipe_xb_b(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_xb_b : NULL; }
extern "C" float *picolm_gpu_pipe_q_b(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_q_b : NULL; }
extern "C" float *picolm_gpu_pipe_k_b(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_k_b : NULL; }
extern "C" float *picolm_gpu_pipe_v_b(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_v_b : NULL; }
extern "C" float *picolm_gpu_pipe_attn_out_b(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_attn_out_b : NULL; }
extern "C" float *picolm_gpu_pipe_ffn_norm_b(int device)   { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_ffn_norm_b : NULL; }
extern "C" float *picolm_gpu_pipe_gate_b(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_gate_b : NULL; }
extern "C" float *picolm_gpu_pipe_up_b(int device)         { gpu_device_ctx_t *c = find_ctx(device); return c ? c->pipe_up_b : NULL; }

/* Pinned host staging buffer accessor. Returns a page-locked host pointer
 * that is safe for async H2D copies via gpuMemcpyAsync. Auto-grows to
 * accommodate the requested byte count. Returns NULL on OOM. */
extern "C" float *picolm_gpu_staging_host(int device, size_t bytes) {
    gpu_device_ctx_t *c = find_ctx(device);
    if (!c) return NULL;
    if (!reserve_pinned(&c->host_x, &c->host_x_cap, bytes)) return NULL;
    return c->host_x;
}

extern "C" float *picolm_gpu_verify_buf(int device)        { return g_verify_buf_dev[device]; }

/* SSM pipeline buffer accessors */
extern "C" float *picolm_gpu_ssm_qkv_raw(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_qkv_raw : NULL; }
extern "C" float *picolm_gpu_ssm_conv_out(int device)      { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_conv_out : NULL; }
extern "C" float *picolm_gpu_ssm_xb2(int device)           { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_xb2 : NULL; }
extern "C" float *picolm_gpu_ssm_xb2_remap(int device)     { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_xb2_remap : NULL; }
extern "C" float *picolm_gpu_ssm_v_remap(int device)       { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_v_remap : NULL; }
extern "C" float *picolm_gpu_ssm_alpha_raw(int device)     { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_alpha_raw : NULL; }
extern "C" float *picolm_gpu_ssm_beta_raw(int device)      { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_beta_raw : NULL; }
extern "C" float *picolm_gpu_ssm_gate_exp(int device)      { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_gate_exp : NULL; }
extern "C" float *picolm_gpu_ssm_beta(int device)          { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_beta : NULL; }
extern "C" float *picolm_gpu_ssm_output(int device)        { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_output : NULL; }
extern "C" float *picolm_gpu_ssm_final_output(int device)  { gpu_device_ctx_t *c = find_ctx(device); return c ? c->ssm_final_output : NULL; }
extern "C" uint16_t *picolm_gpu_kv_k_dev(int device)       { return g_kv_k_dev[device]; }
extern "C" uint16_t *picolm_gpu_kv_v_dev(int device)       { return g_kv_v_dev[device]; }

/* Bit-exact device port of the host fp32_to_fp16() in quant.c -- NOT
 * CUDA's __float2half (different rounding/tie behavior in edge cases),
 * so that PICOLM_DBG_PIPELINE logit comparisons against the CPU path
 * stay meaningful all the way through the KV cache, not just up to
 * whatever tolerance a different rounding rule would introduce. */
__device__ __forceinline__ uint16_t
picolm_gpu_fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (((bits >> 23) & 0xFF) == 0) return (uint16_t)sign;
    if (((bits >> 23) & 0xFF) == 0xFF)
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t round_bit = 1U << (shift - 1);
        mant = (mant + round_bit) >> shift;
        return (uint16_t)(sign | mant);
    }
    mant += 0x00001000;
    if (mant & 0x00800000) {
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Strided KV pack+store kernel: src has tokens packed at src_stride (not kv_dim).
 * Used by GPU prefill ubatch loop where bk/bv have xb_stride layout. */
__global__ void
picolm_gpu_kv_pack_store_strided_kernel(uint16_t *dst_row, const float *src,
    int kv_dim, int n_tokens, int src_stride) {
    for (int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
         i < n_tokens * kv_dim;
         i += (int)gridDim.x * gpuBlockDim_x) {
        int t = i / kv_dim;
        int d = i % kv_dim;
        dst_row[i] = picolm_gpu_fp32_to_fp16(src[(size_t)t * src_stride + d]);
    }
}

/* Pack a device-resident F32 [n_kv_heads*head_dim] vector to F16 and
 * write it directly into the device KV cache row for (layer_ordinal,
 * pos). One thread per element, grid-stride, no shared memory. This
 * replaces the D2H -> CPU convert -> H2D round trip that
 * model_forward_gpu() otherwise needs every layer of every token: with
 * this kernel the KV store never touches the host on the hot path. */
__global__ void
picolm_gpu_kv_pack_store_kernel(uint16_t *dst_row, const float *src, int n) {
    for (int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x; i < n;
         i += (int)gridDim.x * gpuBlockDim_x) {
        dst_row[i] = picolm_gpu_fp32_to_fp16(src[i]);
    }
}

/* Device-native KV store: src_dev is pipe_k/pipe_v (F32, already on
 * device, kv_dim = n_kv_heads*head_dim elements). No transfer, no sync --
 * same ctx->stream ordering argument as the other _dev primitives.
 * Caller must call this (K then V) before picolm_gpu_attention_decode_dev
 * for the same (layer_ordinal, pos), same as picolm_gpu_kv_store_rows. */
extern "C" int
picolm_gpu_kv_store_dev(int is_k, int layer_ordinal, int pos,
                         const float *src_dev, int n_kv_heads, int head_dim,
                         int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;

    int kv_dim = n_kv_heads * head_dim;
    size_t row_bytes = (size_t)kv_dim * sizeof(uint16_t);
    uint16_t *dst_base = is_k ? g_kv_k_dev[device] : g_kv_v_dev[device];
    uint16_t *dst_row = dst_base
        + ((size_t)layer_ordinal * max_seq_len * row_bytes
           + (size_t)pos * row_bytes) / sizeof(uint16_t);

    int n_threads = min(kv_dim, 256);
    int n_blocks = (kv_dim + n_threads - 1) / n_threads;
    picolm_gpu_kv_pack_store_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        dst_row, src_dev, kv_dim);
    if (!gpu_ok(gpuGetLastError(), "kv pack+store (dev)")) return 0;
    return 1;
}

/* Batched variant: src_dev is [S][kv_dim] contiguous F32, positions
 * start_pos..start_pos+S-1. Reuses picolm_gpu_kv_pack_store_kernel --
 * since the device KV cache has no per-row padding, S contiguous source
 * rows map to S contiguous destination rows. One launch for the whole
 * prefill chunk. */
extern "C" int
picolm_gpu_kv_store_dev_batched(int is_k, int layer_ordinal, int start_pos, int n_positions,
                                 const float *src_dev, int n_kv_heads, int head_dim,
                                 int max_seq_len, int device) {
    return picolm_gpu_kv_store_dev_batched_strided(is_k, layer_ordinal, start_pos,
        n_positions, src_dev, n_kv_heads, head_dim, max_seq_len, device, 0);
}

extern "C" int
picolm_gpu_kv_store_dev_batched_strided(int is_k, int layer_ordinal, int start_pos, int n_positions,
                                         const float *src_dev, int n_kv_heads, int head_dim,
                                         int max_seq_len, int device, int src_stride) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;

    int kv_dim = n_kv_heads * head_dim;
    size_t row_bytes = (size_t)kv_dim * sizeof(uint16_t);
    uint16_t *dst_base = is_k ? g_kv_k_dev[device] : g_kv_v_dev[device];
    uint16_t *dst_row = dst_base
        + ((size_t)layer_ordinal * max_seq_len * row_bytes
           + (size_t)start_pos * row_bytes) / sizeof(uint16_t);

    int total = n_positions * kv_dim;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 4096);
    if (src_stride > 0 && src_stride != kv_dim) {
        picolm_gpu_kv_pack_store_strided_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
            dst_row, src_dev, kv_dim, n_positions, src_stride);
    } else {
        picolm_gpu_kv_pack_store_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
            dst_row, src_dev, total);
    }
    if (!gpu_ok(gpuGetLastError(), "kv pack+store batched (dev)")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_kv_debug_dump(int is_k, int layer_ordinal, int pos,
                          uint16_t *dst, int n_elements, int n_kv_heads,
                          int head_dim, int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device]) return 0;
    uint16_t *src_base = is_k ? g_kv_k_dev[device] : g_kv_v_dev[device];
    int kv_dim = n_kv_heads * head_dim;
    uint16_t *src_row = src_base + (size_t)layer_ordinal * max_seq_len * kv_dim + (size_t)pos * kv_dim;
    if (!gpu_ok(gpuMemcpy(dst, src_row, n_elements * sizeof(uint16_t), gpuMemcpyDeviceToHost),
                "kv_debug_dump D2H"))
        return 0;
    return 1;
}

extern "C" int
picolm_gpu_kv_store_rows(int is_k, int layer_ordinal, int start_pos, int n_positions,
                          const void *host_rows, size_t row_bytes,
                          int n_kv_heads, int head_dim, int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;

    /* Sanity: row_bytes must match the F16 GQA row size this cache was
     * sized for. If a caller passes a quantized row_bytes here, refuse
     * rather than silently corrupting the cache. */
    size_t expect_row_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    if (row_bytes != expect_row_bytes) return 0;

    uint16_t *dst_base = is_k ? g_kv_k_dev[device] : g_kv_v_dev[device];
    /* Layout: [layer_ordinal][pos][kv_head][head_dim], contiguous rows.
     * n_positions contiguous rows starting at start_pos == one memcpy. */
    size_t layer_off_bytes = (size_t)layer_ordinal * max_seq_len * row_bytes;
    size_t pos_off_bytes = (size_t)start_pos * row_bytes;
    uint8_t *dst = (uint8_t *)dst_base + layer_off_bytes + pos_off_bytes;

    size_t copy_bytes = (size_t)n_positions * row_bytes;
    if (!gpu_ok(gpuMemcpyAsync(dst, host_rows, copy_bytes,
                               gpuMemcpyHostToDevice, ctx->stream),
                "kv_store_rows async copy"))
        return 0;
    return 1;
}

/* Bulk upload: copy all n_positions for one layer from host F16 cache to device.
 * host_rows: [n_positions][n_kv_heads][head_dim] contiguous uint16_t. */
extern "C" int
picolm_gpu_kv_upload_layer(int is_k, int layer_ordinal, int n_positions,
                            const uint16_t *host_rows, int n_kv_heads,
                            int head_dim, int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;

    size_t row_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    uint16_t *dst_base = is_k ? g_kv_k_dev[device] : g_kv_v_dev[device];
    size_t layer_off_bytes = (size_t)layer_ordinal * max_seq_len * row_bytes;
    uint8_t *dst = (uint8_t *)dst_base + layer_off_bytes;

    size_t copy_bytes = (size_t)n_positions * row_bytes;
    if (!gpu_ok(gpuMemcpyAsync(dst, host_rows, copy_bytes,
                               gpuMemcpyHostToDevice, ctx->stream),
                "kv_upload_layer async copy"))
        return 0;
    return 1;
}

/* Shared dispatch for both picolm_gpu_attention_decode and _dev: chooses
 * single-pass (grid=n_kv_heads, matching the pre-split-K behavior
 * exactly -- no regression risk for short contexts) vs split-K (grid=
 * n_kv_heads*n_splits + a merge pass) based on how many KV positions
 * there are to walk. xb_dev/q_dev must already be device pointers. */
static int
attn_decode_dispatch(float *xb_dev, const float *q_dev,
                      int layer_ordinal, int pos,
                      int n_heads, int n_kv_heads, int head_dim, int max_seq_len,
                      gpu_device_ctx_t *ctx, int device) {
    int kv_mul = n_heads / n_kv_heads;
    size_t kv_pos_stride_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    size_t kv_head_stride_bytes = head_dim * sizeof(uint16_t);

    int total_kv = pos + 1;
    int n_splits = (total_kv + ATTN_DECODE_MIN_CHUNK - 1) / ATTN_DECODE_MIN_CHUNK;
    if (n_splits > ATTN_DECODE_MAX_SPLITS) n_splits = ATTN_DECODE_MAX_SPLITS;
    if (n_splits < 1) n_splits = 1;

    size_t shared_bytes = 2 * head_dim * sizeof(uint16_t) + 256 * sizeof(float)
                         + (size_t)kv_mul * head_dim * sizeof(float)
                         + (size_t)kv_mul * sizeof(float)
                         + (size_t)kv_mul * sizeof(float)
                         + 2 * sizeof(float); /* rescale_sh + weight_sh */

    if (n_splits <= 1) {
        gpu_dispatch_print("attn_decode");
        dim3 grid((unsigned)n_kv_heads, 1, 1);
        picolm_gpu_attention_decode_kernel<<<grid, 256, (unsigned)shared_bytes, ctx->stream>>>(
            xb_dev, q_dev, g_kv_k_dev[device], g_kv_v_dev[device],
            layer_ordinal, pos, n_heads, n_kv_heads, head_dim, max_seq_len,
            kv_pos_stride_bytes, kv_head_stride_bytes);
        if (!gpu_ok(gpuGetLastError(), "attn decode kernel")) return 0;
        return 1;
    }

    int chunk_size = (total_kv + n_splits - 1) / n_splits;
    size_t need = (size_t)n_heads * n_splits * (head_dim + 2) * sizeof(float);
    if (!reserve(&ctx->attn_partial, &ctx->attn_partial_cap, need)) return 0;
    float *partial_max = ctx->attn_partial;
    float *partial_sum = partial_max + (size_t)n_heads * n_splits;
    float *partial_acc = partial_sum + (size_t)n_heads * n_splits;

    gpu_dispatch_print("attn_decode_split");
    dim3 grid_split((unsigned)n_kv_heads, (unsigned)n_splits, 1);
    picolm_gpu_attention_decode_split_kernel<<<grid_split, 256, (unsigned)shared_bytes, ctx->stream>>>(
        partial_max, partial_sum, partial_acc,
        q_dev, g_kv_k_dev[device], g_kv_v_dev[device],
        layer_ordinal, pos, n_heads, n_kv_heads, head_dim, max_seq_len,
        kv_pos_stride_bytes, kv_head_stride_bytes, n_splits, chunk_size);
    if (!gpu_ok(gpuGetLastError(), "attn decode split kernel")) return 0;

    size_t merge_shared = (size_t)kv_mul * 2 * sizeof(float);
    gpu_dispatch_print("attn_decode_merge");
    dim3 grid_merge((unsigned)n_kv_heads, 1, 1);
    picolm_gpu_attention_decode_merge_kernel<<<grid_merge, 128, (unsigned)merge_shared, ctx->stream>>>(
        xb_dev, partial_max, partial_sum, partial_acc,
        n_heads, n_kv_heads, head_dim, n_splits);
    if (!gpu_ok(gpuGetLastError(), "attn decode merge kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_attention_decode(float *xb_out, const float *q_host,
                             int layer_ordinal, int pos,
                             int n_heads, int n_kv_heads, int head_dim,
                             int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0;

    int kv_mul = n_heads / n_kv_heads;
    if (kv_mul < 1 || kv_mul > 8) return 0;

    size_t x_bytes = (size_t)n_heads * head_dim * sizeof(float);
    size_t y_bytes = x_bytes;
    if (!reserve(&ctx->x, &ctx->x_cap, x_bytes) ||
        !reserve(&ctx->y, &ctx->y_cap, y_bytes)) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, q_host, x_bytes, gpuMemcpyHostToDevice),
                "attn decode Q upload")) return 0;

    if (!attn_decode_dispatch(ctx->y, ctx->x, layer_ordinal, pos,
                              n_heads, n_kv_heads, head_dim, max_seq_len,
                              ctx, device)) return 0;

    if (!gpu_ok(gpuDeviceSynchronize(), "attn decode sync")) return 0;

    if (!gpu_ok(gpuMemcpy(xb_out, ctx->y, y_bytes, gpuMemcpyDeviceToHost),
                "attn decode output download")) return 0;
    return 1;
}

/* Device-native decode attention: q_dev and xb_out_dev are already
 * device-resident (Phase 2 pipeline buffers, e.g. the output of the Q
 * projection matmul_dev and rope_apply, both in place on the same
 * buffer). No H2D, no D2H, no gpuDeviceSynchronize() -- same ordering
 * argument as picolm_gpu_matmul_dev: this kernel and picolm_gpu_kv_store_rows
 * (the KV write for this position) are both launched on ctx->stream, so
 * in-stream ordering guarantees the K/V write for `pos` is visible before
 * this kernel reads it, PROVIDED the K/V store call for `pos` happens
 * first in program order on the same device/stream -- caller must store
 * K/V before calling this. q_dev/xb_out_dev must not alias ctx->x/ctx->y. */
extern "C" int
picolm_gpu_attention_decode_dev(float *xb_out_dev, const float *q_dev,
                                 int layer_ordinal, int pos,
                                 int n_heads, int n_kv_heads, int head_dim,
                                 int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0; /* stack array limit */

    int kv_mul = n_heads / n_kv_heads;
    if (kv_mul < 1 || kv_mul > 8) return 0;

    return attn_decode_dispatch(xb_out_dev, q_dev, layer_ordinal, pos,
                                 n_heads, n_kv_heads, head_dim, max_seq_len,
                                 ctx, device);
}

/* Total dynamic shared memory picolm_gpu_attention_prefill_kernel needs:
 * K tile + V tile (u16) + reduce (256 float) + acc[tile_q][head_dim]
 * (float) + max_score[tile_q] + sum_exp[tile_q] (float).
 * tile_q is chosen by the caller based on available shared memory.
 * For head_dim=128 with tile_q=32 this is ~50KB -- over the default
 * 48KB/block limit, so the caller must opt in via gpuFuncSetAttribute. */
static size_t
attn_prefill_shared_bytes(int head_dim, int tile_q) {
    /* Dynamic shared memory (extern __shared__): K tile + V tile + reduce +
     * acc + max_score + sum_exp. Plus 8 bytes of static __shared__ for
     * rescale_sh/weight_sh. Total must be set via gpuFuncSetAttribute. */
    return 2 * (size_t)ATTN_TILE_K * head_dim * sizeof(uint16_t)
         + 256 * sizeof(float)
         + (size_t)tile_q * head_dim * sizeof(float)
         + (size_t)tile_q * sizeof(float)
         + (size_t)tile_q * sizeof(float)
         + 2 * sizeof(float); /* rescale_sh + weight_sh (static __shared__) */
}

/* Opts the kernel into a larger dynamic shared memory limit if needed.
 * Idempotent: tracks the largest size already configured so repeated
 * calls (every prefill call, every layer) are a cheap no-op after the
 * first. Returns 1 if `bytes` is safe to launch with, 0 if the opt-in
 * itself failed (e.g. device doesn't support this much shared memory). */
static int
ensure_attn_prefill_shared_mem(size_t bytes) {
    static size_t configured = 49152; /* default limit, no opt-in needed under this */
    if (bytes <= configured) return 1;
    if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_gpu_attention_prefill_kernel,
                                     gpuFuncAttributeMaxDynamicSharedMemorySize, (int)bytes),
                "attn prefill shared mem opt-in"))
        return 0;
    configured = bytes;
    return 1;
}

/* Query device for maximum shared memory per block (in bytes).
 * Returns 0 on failure. */
static size_t
get_device_shared_mem_per_block(int device) {
    static size_t cached[PICOLM_GPU_MAX_DEVICES] = {0};
    if (device < 0 || device >= PICOLM_GPU_MAX_DEVICES) return 0;
    if (cached[device]) return cached[device];
    int value = 0;
    if (!gpu_ok(gpuDeviceGetAttribute(&value, gpuDeviceAttributeMaxSharedMemoryPerBlockOptin, device),
                "get shared mem per block")) {
        /* Fall back to device total shared memory per multiprocessor / max blocks per SM.
         * Some older drivers don't support the optin attribute. */
        int smem_mp = 0, blocks_sm = 1;
        gpuDeviceGetAttribute(&smem_mp, gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor, device);
        gpuDeviceGetAttribute(&blocks_sm, gpuDeviceAttributeMaxBlocksPerMultiprocessor, device);
        if (smem_mp > 0 && blocks_sm > 0) {
            cached[device] = (size_t)smem_mp / blocks_sm;
            return cached[device];
        }
        cached[device] = 49152; /* conservative default */
        return cached[device];
    }
    cached[device] = (size_t)value;
    return cached[device];
}

static int
ensure_attn_fa2_shared_mem(size_t bytes) {
    static size_t configured = 49152;
    if (bytes <= configured) return 1;
    if (!gpu_ok(gpuFuncSetAttribute((const void *)picolm_gpu_attention_prefill_fa2_kernel,
                                     gpuFuncAttributeMaxDynamicSharedMemorySize, (int)bytes),
                "attn fa2 shared mem opt-in"))
        return 0;
    configured = bytes;
    return 1;
}

extern "C" int
picolm_gpu_attention_prefill(float *xb_out, const float *q_host,
                              int layer_ordinal, int start_pos, int n_tokens,
                              int n_heads, int n_kv_heads, int head_dim,
                              int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0;

    size_t x_bytes = (size_t)n_tokens * n_heads * head_dim * sizeof(float);
    size_t y_bytes = x_bytes;
    if (!reserve(&ctx->x, &ctx->x_cap, x_bytes) ||
        !reserve(&ctx->y, &ctx->y_cap, y_bytes)) return 0;

    if (!gpu_ok(gpuMemcpyAsync(ctx->x, q_host, x_bytes, gpuMemcpyHostToDevice, ctx->stream),
                "attn prefill Q upload")) return 0;

    size_t kv_pos_stride_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    size_t kv_head_stride_bytes = head_dim * sizeof(uint16_t);

    /* Try FA2 (tensor core) kernel when available (NVIDIA IMMA only) */
    int use_fa2 = ctx->has_imma && n_tokens >= 64 && !getenv("PICOLM_FORCE_SCALAR_ATTN");
    if (use_fa2) {
        int n_q_tiles = (n_tokens + 63) / 64;
        /* Shared memory per block (one query head, 4 warps * 16 Q rows):
         * K(FA2_TILE_K*hd) + V(FA2_TILE_K*hd) + Q(64*hd) uint16_t
         * + score(64*FA2_TILE_K) + acc(64*hd) + max(64) + sum(64) float
         * For hd=128, TILE_K=32: 72.5KB. For hd=256, TILE_K=32: 137KB (exceeds limit, scalar fallback). */
        size_t fa2_shared = (size_t)FA2_TILE_K * head_dim * 2 * sizeof(uint16_t)
            + 64 * head_dim * sizeof(uint16_t)
            + 64 * FA2_TILE_K * sizeof(float)
            + 64 * head_dim * sizeof(float)
            + 128 * sizeof(float);
        if (fa2_shared <= 98304 && ensure_attn_fa2_shared_mem(fa2_shared)) {
            gpu_dispatch_print("attn_prefill_fa2_host");
            dim3 grid((unsigned)n_heads, (unsigned)n_q_tiles, 1);
            picolm_gpu_attention_prefill_fa2_kernel<<<grid, 128, (unsigned)fa2_shared, ctx->stream>>>(
                ctx->y, ctx->x,
                g_kv_k_dev[device], g_kv_v_dev[device],
                layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
                kv_pos_stride_bytes, kv_head_stride_bytes);
            if (!gpu_ok(gpuGetLastError(), "attn prefill fa2 (host)")) return 0;
        } else {
            use_fa2 = 0;
        }
    }

    if (!use_fa2) {
        gpu_dispatch_print("attn_prefill_scalar_host");
        /* Choose tile_q that fits device shared memory */
        int tile_q = ATTN_TILE_Q;
        size_t smem_max = get_device_shared_mem_per_block(device);
        while (tile_q > 8 && attn_prefill_shared_bytes(head_dim, tile_q) > smem_max) {
            tile_q /= 2;
        }
        int n_tiles_q = (n_tokens + tile_q - 1) / tile_q;
        size_t shared_bytes = attn_prefill_shared_bytes(head_dim, tile_q);
        if (!ensure_attn_prefill_shared_mem(shared_bytes)) return 0;
        int block_threads = 128;

        dim3 grid((unsigned)n_heads, (unsigned)n_tiles_q, 1);
        picolm_gpu_attention_prefill_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
            ctx->y, ctx->x,
            g_kv_k_dev[device], g_kv_v_dev[device],
            layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
            kv_pos_stride_bytes, kv_head_stride_bytes, tile_q);
    }

    if (!gpu_ok(gpuGetLastError(), "attn prefill kernel")) return 0;

    /* Stream ordering ensures kernel completes before D2H starts.
     * Async D2H implicitly blocks CPU on completion, no explicit sync needed. */
    if (!gpu_ok(gpuMemcpyAsync(xb_out, ctx->y, y_bytes, gpuMemcpyDeviceToHost, ctx->stream),
                "attn prefill output download")) return 0;
    return 1;
}

/* Device-native prefill attention: xb_out_dev and q_dev are already
 * device-resident. No H2D, no D2H, no gpuDeviceSynchronize(). */
extern "C" int
picolm_gpu_attention_prefill_dev(float *xb_out_dev, const float *q_dev,
                                  int layer_ordinal, int start_pos, int n_tokens,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  int max_seq_len, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (!g_kv_k_dev[device] || !g_kv_v_dev[device]) return 0;
    if (head_dim > 256) return 0;
    if (head_dim % 16 != 0) return 0;

    size_t kv_pos_stride_bytes = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
    size_t kv_head_stride_bytes = head_dim * sizeof(uint16_t);

    /* Use FA2 (tensor core) kernel only on NVIDIA with IMMA.
     * HIP scalar fallback is too slow (10-15s for 200 tokens). */
    if (ctx->has_imma && n_tokens >= 64 && !getenv("PICOLM_FORCE_SCALAR_ATTN")) {
        int n_q_tiles = (n_tokens + 63) / 64;  /* ceil, kernel handles OOB */
        /* Shared memory per block (one query head):
         * K(FA2_TILE_K*hd) + V(FA2_TILE_K*hd) + Q(64*hd) uint16_t
         * + score(64*FA2_TILE_K) + acc(64*hd) + max/sum float
         * For hd=128, TILE_K=32: 72.5KB. For hd=256, TILE_K=32: 137KB (exceeds 96KB, scalar fallback). */
        size_t fa2_shared = (size_t)FA2_TILE_K * head_dim * 2 * sizeof(uint16_t)
            + 64 * head_dim * sizeof(uint16_t)
            + 64 * FA2_TILE_K * sizeof(float)
            + 64 * head_dim * sizeof(float)
            + 128 * sizeof(float);
        if (fa2_shared > 98304) {
            /* Too large, fall through to scalar */
        } else {
            if (!ensure_attn_fa2_shared_mem(fa2_shared)) return 0;
            gpu_dispatch_print("attn_prefill_fa2_dev");
            dim3 grid((unsigned)n_heads, (unsigned)n_q_tiles, 1);
            picolm_gpu_attention_prefill_fa2_kernel<<<grid, 128, (unsigned)fa2_shared, ctx->stream>>>(
                xb_out_dev, q_dev,
                g_kv_k_dev[device], g_kv_v_dev[device],
                layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
                kv_pos_stride_bytes, kv_head_stride_bytes);
            if (!gpu_ok(gpuGetLastError(), "attn prefill fa2 (dev)")) return 0;
            return 1;
        }
    }

    gpu_dispatch_print("attn_prefill_scalar_dev");
    /* Scalar fallback: choose tile_q that fits device shared memory.
     * Start with 32, halve until it fits. Minimum is 8. */
    int tile_q = ATTN_TILE_Q;
    size_t smem_max = get_device_shared_mem_per_block(device);
    while (tile_q > 8 && attn_prefill_shared_bytes(head_dim, tile_q) > smem_max) {
        tile_q /= 2;
    }
    int n_tiles_q = (n_tokens + tile_q - 1) / tile_q;
    size_t shared_bytes = attn_prefill_shared_bytes(head_dim, tile_q);
    if (!ensure_attn_prefill_shared_mem(shared_bytes)) return 0;
    int block_threads = 128;

    dim3 grid((unsigned)n_heads, (unsigned)n_tiles_q, 1);
    picolm_gpu_attention_prefill_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
        xb_out_dev, q_dev,
        g_kv_k_dev[device], g_kv_v_dev[device],
        layer_ordinal, start_pos, n_tokens, n_heads, n_kv_heads, head_dim, max_seq_len,
        kv_pos_stride_bytes, kv_head_stride_bytes, tile_q);
    if (!gpu_ok(gpuGetLastError(), "attn prefill (dev)")) return 0;
    return 1;
}

/* Device-native prefill attention with FP32 K/V (no KV cache read).
 * Used for SSM-hybrid models where FP16 KV round-trip causes too much
 * numerical drift. K and V are passed as FP32 projection buffers
 * with the same layout as the KV cache: [pos][kv_head][head_dim]. */
extern "C" int
picolm_gpu_attention_prefill_f32kv(float *xb_out_dev, const float *q_dev,
                                    const float *k_dev, const float *v_dev,
                                    int start_pos, int n_tokens,
                                    int n_heads, int n_kv_heads, int head_dim,
                                    int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (head_dim > 256) return 0;

    /* FP32 K/V tiles are 4x larger than FP16.
     * shared_bytes(tile_q) = 2*ATTN_TILE_K*hd*4 + 256*4 + tile_q*hd*4 + 2*tile_q*4 + 8
     * For hd=256, tile_q=32: 65536+1024+32768+256+8 = 99592 (~97KB) -- too big for gfx906.
     * Choose tile_q adaptively. */
    int tile_q = ATTN_TILE_Q;
    size_t smem_max = get_device_shared_mem_per_block(device);
    auto f32kv_smem = [](int hd, int tq) -> size_t {
        return 2 * (size_t)ATTN_TILE_K * hd * sizeof(float)
             + 256 * sizeof(float)
             + (size_t)tq * hd * sizeof(float)
             + 2 * (size_t)tq * sizeof(float)
             + 2 * sizeof(float);
    };
    while (tile_q > 8 && f32kv_smem(head_dim, tile_q) > smem_max) {
        tile_q /= 2;
    }
    int n_tiles_q = (n_tokens + tile_q - 1) / tile_q;
    size_t shared_bytes = f32kv_smem(head_dim, tile_q);
    /* Opt-in FP32 kernel to larger shared memory */
    if (shared_bytes > 49152) {
        gpuError_t e = gpuFuncSetAttribute((const void *)picolm_gpu_attention_prefill_f32kv_kernel,
            gpuFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);
        if (e != gpuSuccess) {
            fprintf(stderr, "[GPU] f32kv opt-in fail: err=%d (%s) bytes=%zu\n",
                (int)e, gpuGetErrorString(e), shared_bytes);
        }
    }
    gpu_dispatch_print("attn_prefill_f32kv_dev");
    int block_threads = 128;
    dim3 grid((unsigned)n_heads, (unsigned)n_tiles_q, 1);
    picolm_gpu_attention_prefill_f32kv_kernel<<<grid, block_threads, (unsigned)shared_bytes, ctx->stream>>>(
        xb_out_dev, q_dev, k_dev, v_dev,
        start_pos, n_tokens, n_heads, n_kv_heads, head_dim, tile_q);
    if (!gpu_ok(gpuGetLastError(), "attn prefill f32kv (dev)")) return 0;
    return 1;
}

/* ================================================================
 * Phase 2: Device-resident elementwise kernels (stubs)
 * ================================================================ */

/* RMSNorm kernel: out[d] = x[d] * rsqrt(mean(x^2) + eps) * weight[d].
 * Single block, grid-stride loop -- correct for any dim (an earlier
 * version returned early before __syncthreads() for threads with
 * d >= dim, which deadlocks/UB whenever dim isn't an exact multiple of
 * blockDim). Launched with grid=1 by the host wrapper below. */
__global__ void
picolm_gpu_rmsnorm_kernel(float *out, const float *x, const float *weight,
                           int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = gpuThreadIdx_x; i < dim; i += gpuBlockDim_x) {
        sum_sq += x[i] * x[i];
    }
    __shared__ float ssum[256];
    ssum[gpuThreadIdx_x] = sum_sq;
    gpuSyncthreads();
    for (int s = gpuBlockDim_x / 2; s > 0; s >>= 1) {
        if (gpuThreadIdx_x < s) ssum[gpuThreadIdx_x] += ssum[gpuThreadIdx_x + s];
        gpuSyncthreads();
    }
    float rms = sqrtf(ssum[0] / dim + eps);
    float inv_rms = 1.0f / rms;
    for (int d = gpuThreadIdx_x; d < dim; d += gpuBlockDim_x) {
        out[d] = x[d] * inv_rms * weight[d];
    }
}

/* Batched rmsnorm: one block per row (S rows), same reduction per block.
 * weight is shared across all rows. Mirrors how matmul_dev batches over S. */
__global__ void
picolm_gpu_rmsnorm_batched_kernel(float *out, const float *x, const float *weight,
                                   int dim, float eps, int x_stride) {
    int row = (int)gpuBlockIdx_x;
    int stride = (x_stride > 0) ? x_stride : dim;
    const float *xr = x + (size_t)row * stride;
    float *outr = out + (size_t)row * stride;

    float sum_sq = 0.0f;
    for (int i = gpuThreadIdx_x; i < dim; i += gpuBlockDim_x) {
        sum_sq += xr[i] * xr[i];
    }
    __shared__ float ssum[256];
    ssum[gpuThreadIdx_x] = sum_sq;
    gpuSyncthreads();
    for (int s = gpuBlockDim_x / 2; s > 0; s >>= 1) {
        if (gpuThreadIdx_x < s) ssum[gpuThreadIdx_x] += ssum[gpuThreadIdx_x + s];
        gpuSyncthreads();
    }
    float rms = sqrtf(ssum[0] / dim + eps);
    float inv_rms = 1.0f / rms;
    for (int d = gpuThreadIdx_x; d < dim; d += gpuBlockDim_x) {
        outr[d] = xr[d] * inv_rms * weight[d];
    }
}

/* Batched RoPE: each element computes its own absolute position's table
 * row. x is [S][n_heads][head_dim] contiguous, positions start_pos..start_pos+S-1.
 * One launch for the whole prefill chunk. */
__global__ void
picolm_gpu_rope_batched_kernel(float *x, int n_heads, int head_dim,
                                const float *cos_tbl, const float *sin_tbl,
                                int half_dim, int start_pos, int S,
                                int rope_type) {
    int idx = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int per_row = n_heads * head_dim;
    int total = S * per_row;
    for (int i = idx; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        int row = i / per_row;
        int rem = i % per_row;
        int h = rem / head_dim;
        int d = rem % head_dim;
        float *xr = x + (size_t)row * per_row;
        int pos = start_pos + row;

        if (rope_type) {
            /* Qwen2 interleaved: (x[d], x[d+half_dim]) rotated */
            if (d >= half_dim) continue;
            int h1 = d, h2 = d + half_dim;
            float x1 = xr[h * head_dim + h1];
            float x2 = xr[h * head_dim + h2];
            float c = cos_tbl[(size_t)pos * half_dim + h1];
            float s = sin_tbl[(size_t)pos * half_dim + h1];
            xr[h * head_dim + h1] = x1 * c - x2 * s;
            xr[h * head_dim + h2] = x1 * s + x2 * c;
        } else {
            /* Llama pairwise: (x[2i], x[2i+1]) rotated by cos[i], sin[i] */
            if (d & 1) continue; /* odd index: handled by even partner */
            if (d + 1 >= head_dim) continue; /* safety: beyond head */
            int idx_pair = d >> 1; /* 0,1 -> pair 0; 2,3 -> pair 1; etc. */
            if (idx_pair >= half_dim) continue; /* beyond rope range */
            float x0 = xr[h * head_dim + d];
            float x1v = xr[h * head_dim + d + 1];
            float c = cos_tbl[(size_t)pos * half_dim + idx_pair];
            float s = sin_tbl[(size_t)pos * half_dim + idx_pair];
            xr[h * head_dim + d]     = x0 * c - x1v * s;
            xr[h * head_dim + d + 1] = x0 * s + x1v * c;
        }
    }
}

/* RoPE kernel: applies pairwise rotary position embedding */
__global__ void
picolm_gpu_rope_kernel(float *x, int n_heads, int head_dim,
                        const float *cos_tbl, const float *sin_tbl,
                        int half_dim, int rope_type) {
    int idx = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int total = n_heads * head_dim;
    for (int i = idx; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        int h = i / head_dim;
        int d = i % head_dim;

        if (rope_type) {
            /* Qwen2 interleaved: (x[d], x[d+half_dim]) rotated */
            if (d >= half_dim) continue;
            int h1 = d, h2 = d + half_dim;
            float x1 = x[h * head_dim + h1];
            float x2 = x[h * head_dim + h2];
            float c = cos_tbl[h1];
            float s = sin_tbl[h1];
            x[h * head_dim + h1] = x1 * c - x2 * s;
            x[h * head_dim + h2] = x1 * s + x2 * c;
        } else {
            /* Llama pairwise: (x[2i], x[2i+1]) rotated by cos[i], sin[i] */
            if (d & 1) continue; /* odd: handled by even partner */
            if (d + 1 >= head_dim) continue;
            int idx_pair = d >> 1;
            if (idx_pair >= half_dim) continue;
            float x0 = x[h * head_dim + d];
            float x1v = x[h * head_dim + d + 1];
            float c = cos_tbl[idx_pair];
            float s = sin_tbl[idx_pair];
            x[h * head_dim + d]     = x0 * c - x1v * s;
            x[h * head_dim + d + 1] = x0 * s + x1v * c;
        }
    }
}

/* Residual add kernel: out[i] = a[i] + b[i] */
__global__ void
picolm_gpu_residual_add_kernel(float *out, const float *a, const float *b, int n, int dim, int stride) {
    int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int total = n * dim;
    for (; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        int tok = i / dim;
        int off = i % dim;
        out[tok * stride + off] = a[tok * stride + off] + b[tok * stride + off];
    }
}

/* Q+gate de-interleave kernel for SSM attention layers.
 * GGUF stores [Q_0, Gate_0, Q_1, Gate_1, ...] per head, each head_dim floats.
 * Reads raw[head * 2 * head_dim + ...], writes Q to out_q[head * head_dim + ...]
 * and Gate to out_g[head * head_dim + ...].
 * One thread per head, thread-0 of each block does the memmove-like copy. */
__global__ void
picolm_gpu_qg_deinterleave_kernel(const float *raw, float *out_q, float *out_g,
                                   int n_heads, int head_dim) {
    int h = gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (h >= n_heads) return;
    const float *src = raw + (size_t)h * 2 * head_dim;
    float *dst_q = out_q + (size_t)h * head_dim;
    float *dst_g = out_g + (size_t)h * head_dim;
    for (int d = 0; d < head_dim; d++) {
        dst_q[d] = src[d];           /* Q portion */
        dst_g[d] = src[head_dim + d]; /* Gate portion */
    }
}

/* Elementwise sigmoid-multiply kernel: out[i] = a[i] * sigmoid(g[i])
 * For SSM attention: pipe_attn_out *= sigmoid(gate) before output projection. */
__global__ void
picolm_gpu_sigmoid_mul_kernel(float *out, const float *gate, int n) {
    int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    for (; i < n; i += (int)gridDim.x * gpuBlockDim_x) {
        float g = gate[i];
        float sg = (g > 20.0f) ? 1.0f : (g < -20.0f) ? 0.0f : 1.0f / (1.0f + expf(-g));
        out[i] *= sg;
    }
}

/* Batched Q+gate de-interleave: processes S sequences, each with n_heads heads.
 * raw is [S][n_heads * 2 * head_dim], out_q is [S][n_heads * head_dim],
 * out_g is [S][n_heads * head_dim]. */
__global__ void
picolm_gpu_qg_deinterleave_batched_kernel(const float *raw, float *out_q,
                                           float *out_g, int n_heads,
                                           int head_dim, int S) {
    int idx = gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    int total = n_heads * S;
    if (idx >= total) return;
    int s = idx / n_heads;
    int h = idx % n_heads;
    size_t row_stride = (size_t)n_heads * 2 * head_dim;
    size_t q_stride = (size_t)n_heads * head_dim;
    const float *src = raw + (size_t)s * row_stride + (size_t)h * 2 * head_dim;
    float *dst_q = out_q + (size_t)s * q_stride + (size_t)h * head_dim;
    float *dst_g = out_g + (size_t)s * q_stride + (size_t)h * head_dim;
    for (int d = 0; d < head_dim; d++) {
        dst_q[d] = src[d];
        dst_g[d] = src[head_dim + d];
    }
}

/* Batched sigmoid-multiply: processes S sequences, each n elements. */
__global__ void
picolm_gpu_sigmoid_mul_batched_kernel(float *out, const float *gate,
                                       int n, int S) {
    int i = gpuThreadIdx_x + (int)gpuBlockIdx_x * gpuBlockDim_x;
    int total = n * S;
    for (; i < total; i += (int)gridDim.x * gpuBlockDim_x) {
        float g = gate[i];
        float sg = (g > 20.0f) ? 1.0f : (g < -20.0f) ? 0.0f : 1.0f / (1.0f + expf(-g));
        out[i] *= sg;
    }
}

/* Phase 2 host API */
/* Device-native rmsnorm: all pointers are device-resident, no H2D/D2H, no sync.
 * Used from model_forward_gpu() pipeline path. */
extern "C" int
picolm_gpu_rmsnorm_dev(float *out, const float *x, const float *weight,
                        int dim, float eps, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_kernel<<<1, n_threads, 0, ctx->stream>>>(
        out, x, weight, dim, eps);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm dev kernel")) return 0;
    return 1;
}

/* Host-side rmsnorm: takes host pointers, does H2D/D2H/sync.
 * Used from ssm_forward() QK-norm path and other host-side callers. */
extern "C" int
picolm_gpu_rmsnorm(float *out, const float *x, const float *weight,
                    int dim, float eps, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    size_t xb = dim * sizeof(float);
    size_t wb = dim * sizeof(float);
    void *dx, *dw, *do_;
    if (!gpu_ok(gpuMalloc(&dx, xb), "rmsnorm x") ||
        !gpu_ok(gpuMalloc(&dw, wb), "rmsnorm w") ||
        !gpu_ok(gpuMalloc(&do_, xb), "rmsnorm o")) return 0;
    if (!gpu_ok(gpuMemcpy(dx, x, xb, gpuMemcpyHostToDevice), "rmsnorm x h2d") ||
        !gpu_ok(gpuMemcpy(dw, weight, wb, gpuMemcpyHostToDevice), "rmsnorm w h2d")) {
        gpuFree(dx); gpuFree(dw); gpuFree(do_); return 0;
    }
    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_kernel<<<1, n_threads, 0, ctx->stream>>>(
        (float *)do_, (const float *)dx, (const float *)dw, dim, eps);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm kernel") ||
        !gpu_ok(gpuDeviceSynchronize(), "rmsnorm sync")) {
        gpuFree(dx); gpuFree(dw); gpuFree(do_); return 0;
    }
    gpuMemcpy(out, do_, xb, gpuMemcpyDeviceToHost);
    gpuFree(dx); gpuFree(dw); gpuFree(do_);
    return 1;
}

/* Device-native batched rmsnorm: all pointers device-resident, no H2D/D2H/sync.
 * Used from model_forward_gpu() / model_forward_prefill_gpu() pipeline paths. */
extern "C" int
picolm_gpu_rmsnorm_batched_dev(float *out, const float *x, const float *weight,
                                int dim, float eps, int S, int x_stride, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;
    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_batched_kernel<<<S, n_threads, 0, ctx->stream>>>(
        out, x, weight, dim, eps, x_stride);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm batched dev kernel")) return 0;
    return 1;
}

/* Host-side batched rmsnorm: takes host pointers, does H2D/D2H/sync.
 /* Used from ssm_forward() QK-norm path and prefill attention QK-norm.
 * Launches the kernel directly on ctx->stream.
 * x and out must be device-accessible.
 * weight: if it's a device pointer (from gpu_upload_f32), used directly.
 *   If it's a host pointer (from dequantize_row), uploaded once and cached. */
extern "C" int
picolm_gpu_rmsnorm_batched(float *out, const float *x, const float *weight,
                            int dim, float eps, int S, int x_stride, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;

    const void *w_dev = weight;
    gpu_mutex_lock();
    int cached = 0;
    for (int i = 0; i < ctx->rmsnorm_w_n; i++) {
        if (ctx->rmsnorm_w_keys[i] == weight) { w_dev = ctx->rmsnorm_w_dev[i]; cached = 1; break; }
    }
    if (!cached) {
        if ((uintptr_t)weight < 1024ull * 1024 * 1024) {
            /* Host pointer: upload to device and cache */
            size_t wb = dim * sizeof(float);
            if (ctx->rmsnorm_w_n >= 64) { gpu_mutex_unlock(); return 0; }
            void *dw = NULL;
            if (!gpu_ok(gpuMalloc(&dw, wb), "rmsnorm_b w")) { gpu_mutex_unlock(); return 0; }
            gpuDeviceSynchronize();
            if (!gpu_ok(gpuMemcpy(dw, weight, wb, gpuMemcpyHostToDevice), "rmsnorm_b w h2d")) {
                gpuFree(dw); gpu_mutex_unlock(); return 0;
            }
            ctx->rmsnorm_w_keys[ctx->rmsnorm_w_n] = weight;
            ctx->rmsnorm_w_dev[ctx->rmsnorm_w_n] = dw;
            ctx->rmsnorm_w_n++;
            w_dev = dw;
        }
    }
    gpu_mutex_unlock();

    int n_threads = min(dim, 256);
    picolm_gpu_rmsnorm_batched_kernel<<<S, n_threads, 0, ctx->stream>>>(
        out, x, (const float *)w_dev, dim, eps, x_stride);
    if (!gpu_ok(gpuGetLastError(), "rmsnorm batched kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_rope_apply(float *x, int n_heads, int head_dim,
                       const float *cos_tbl, const float *sin_tbl,
                       int half_dim, int rope_type, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    int total = n_heads * head_dim;
    int n_threads = 128;
    int n_blocks = min((total + n_threads - 1) / n_threads, 128);
    picolm_gpu_rope_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        x, n_heads, head_dim, cos_tbl, sin_tbl, half_dim, rope_type);
    if (!gpu_ok(gpuGetLastError(), "rope kernel")) return 0;
    return 1;
}

/* Batched RoPE: cos_tbl_base/sin_tbl_base are the UNOFFSET [max_seq_len][half_dim]
 * base pointers. Each row computes its own position as start_pos + row. */
extern "C" int
picolm_gpu_rope_apply_batched(float *x, int n_heads, int head_dim,
                               const float *cos_tbl_base, const float *sin_tbl_base,
                               int half_dim, int start_pos, int S,
                               int rope_type, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (S < 1) return 0;

    int total = S * n_heads * head_dim;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 4096);
    picolm_gpu_rope_batched_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        x, n_heads, head_dim, cos_tbl_base, sin_tbl_base, half_dim, start_pos, S, rope_type);
    if (!gpu_ok(gpuGetLastError(), "rope batched kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_residual_add(float *out, const float *a, const float *b,
                         int n, int dim, int stride, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    int n_threads = 256;
    int n_blocks = min((n * dim + n_threads - 1) / n_threads, 256);
    picolm_gpu_residual_add_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        out, a, b, n, dim, stride);
    if (!gpu_ok(gpuGetLastError(), "residual_add kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_qg_deinterleave_dev(const float *raw_dev, float *out_q_dev,
                                float *out_g_dev, int n_heads, int head_dim,
                                int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int n_threads = 256;
    int n_blocks = min((n_heads + n_threads - 1) / n_threads, 256);
    picolm_gpu_qg_deinterleave_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        raw_dev, out_q_dev, out_g_dev, n_heads, head_dim);
    if (!gpu_ok(gpuGetLastError(), "qg_deinterleave kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_sigmoid_mul_dev(float *out_dev, const float *gate_dev,
                            int n, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int n_threads = 256;
    int n_blocks = min((n + n_threads - 1) / n_threads, 256);
    picolm_gpu_sigmoid_mul_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        out_dev, gate_dev, n);
    if (!gpu_ok(gpuGetLastError(), "sigmoid_mul kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_qg_deinterleave_batched_dev(const float *raw_dev, float *out_q_dev,
                                        float *out_g_dev, int n_heads,
                                        int head_dim, int S, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int total = n_heads * S;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 256);
    picolm_gpu_qg_deinterleave_batched_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        raw_dev, out_q_dev, out_g_dev, n_heads, head_dim, S);
    if (!gpu_ok(gpuGetLastError(), "qg_deinterleave_batched kernel")) return 0;
    return 1;
}

extern "C" int
picolm_gpu_sigmoid_mul_batched_dev(float *out_dev, const float *gate_dev,
                                   int n, int S, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    int total = n * S;
    int n_threads = 256;
    int n_blocks = min((total + n_threads - 1) / n_threads, 256);
    picolm_gpu_sigmoid_mul_batched_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        out_dev, gate_dev, n, S);
    if (!gpu_ok(gpuGetLastError(), "sigmoid_mul_batched kernel")) return 0;
    return 1;
}

/* Standalone device-native host wrapper. picolm_silu_mul itself was
 * previously only ever launched inline inside picolm_gpu_expert_mlp()
 * (which does its own H2D/D2H); the Phase 2 pipeline needs to call it
 * on already-device-resident gate/up buffers between matmul_dev calls,
 * with no transfer and no per-call sync (same stream-ordering argument
 * as picolm_gpu_matmul_dev / picolm_gpu_attention_decode_dev). Result is
 * written in place into gate_dev. */
extern "C" int
picolm_gpu_silu_mul_dev(float *gate_dev, const float *up_dev, size_t n, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_silu_mul<<<(unsigned)((n + 255) / 256), 256, 0, ctx->stream>>>(
        gate_dev, up_dev, n);
    if (!gpu_ok(gpuGetLastError(), "silu_mul (dev)")) return 0;
    return 1;
}

/* The single sync point for the whole model_forward_gpu() pass: call
 * this exactly once, after the last device-native op (typically the
 * final rmsnorm), before reading anything back via D2H. Every _dev
 * primitive above is launched on ctx->stream with no internal sync, so
 * this is what actually guarantees all of it has completed. */
extern "C" int
picolm_gpu_sync(int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    return gpu_ok(gpuDeviceSynchronize(), "pipeline sync");
}

/* Synchronous memcpy wrapper for model.c (C file, no CUDA types).
 * dir: 1 = H2D, -1 = D2H, 0 = D2D. Returns 1 on success.
 * Uses ctx->stream for the actual transfer to avoid multi-engine
 * reordering with default stream on sm_121 (GB10/CUDA 13). */
extern "C" int
picolm_gpu_memcpy(void *dst, const void *src, size_t bytes, int dir, int device) {
    if (!dst || !src || bytes < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    gpuMemcpyKind kind = (dir > 0) ? gpuMemcpyHostToDevice :
                          (dir < 0) ? gpuMemcpyDeviceToHost :
                                      gpuMemcpyDeviceToDevice;

    /* Use async copy on ctx->stream + sync to avoid default-stream
     * interleaving on GB10 multi-engine scheduler. */
    gpuError_t err = gpuMemcpyAsync(dst, src, bytes, kind, ctx->stream);
    if (!gpu_ok(err, "pipeline memcpy async")) return 0;
    return gpu_ok(gpuDeviceSynchronize(), "pipeline memcpy sync");
}

/* Async variant: copies on ctx->stream without synchronizing.
 * Caller is responsible for ensuring ordering via subsequent
 * kernel launches or explicit syncs. Only safe for D2D copies
 * and for cases where the caller knows the stream ordering is sufficient. */
extern "C" int
picolm_gpu_memcpy_async(void *dst, const void *src, size_t bytes, int dir, int device) {
    if (!dst || !src || bytes < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    gpuMemcpyKind kind = (dir > 0) ? gpuMemcpyHostToDevice :
                          (dir < 0) ? gpuMemcpyDeviceToHost :
                                      gpuMemcpyDeviceToDevice;
    return gpu_ok(gpuMemcpyAsync(dst, src, bytes, kind, ctx->stream), "pipeline memcpy async");
}
extern "C"
int picolm_gpu_ssm_conv1d_batch_dev(float *od, float *sd, const float *id, const float *wd,
    int cd, int dc, int nt, int dev, int stride) {
    if(cd<1||dc<1||nt<1) return 0;
    if(dc>PICOLM_SSM_CONV_MAX_D_CONV) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    int nb=(cd+255)/256;
    picolm_gpu_ssm_conv1d_batch_kernel<<<nb,256,0,ctx->stream>>>(od,sd,id,wd,cd,dc,nt,stride);
    return gpu_ok(gpuGetLastError(),"conv1d batch dev");
}

extern "C"
int picolm_gpu_ssm_l2norm_batch_dev(float *xd, int hd, int nh, int nt, int ts, float eps, float es, int dev) {
    if(hd<1||nh<1||hd>256||nt<1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    picolm_gpu_ssm_l2norm_batch_kernel<<<dim3((unsigned)nh,(unsigned)nt),min(hd,256),0,ctx->stream>>>(xd,hd,nh,nt,ts,eps,es);
    return gpu_ok(gpuGetLastError(),"l2norm batch dev");
}

extern "C"
/* Forward declaration: defined later in this file */
__global__ void picolm_gpu_ssm_head_permute_batch_kernel(float *dst, const float *src,
    const int *head_map, int head_dim, int n_heads, int n_tokens,
    int src_stride, int dst_stride);

/* IMMA-accelerated vecdot: x[n_tokens][dim] @ weights[n_heads][dim]^T -> y[n_tokens][n_heads].
 * Mathematically a matmul with S=n_tokens, I=dim, O=n_heads.
 * For Q8_0 weights: quantize x to Q8_0 on-device, dispatch IMMA/tiled/scalar kernel.
 * For F32/BF16 weights: pre-quantize to Q8_0 on first use, cache in static buffer,
 *   then dispatch IMMA/tiled/scalar kernel.
 * For Q4_0: fall back to the serial vecdot kernel.
 * When head_map != NULL, post-permute output from GGUF head order to natural head order. */
int picolm_gpu_ssm_vecdot_batch_dev(float *od, const float *xd, const void *wd, gguf_type_t qt,
    int dim, int nvh, int nt, int rb, const int *hm, int dev, int in_stride, int out_stride) {
    if(nvh<=0||dim<=0||nt<=0) return 0;
    if(dim>PICOLM_SSM_VECDOT_MAX_DIM) return 0;
    if(!wd) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;

    /* IMMA fast path for Q8_0, F32, BF16 weights */
    if (qt == GGUF_TYPE_Q8_0 && !getenv("PICOLM_FORCE_F32_MATMUL") && !getenv("PICOLM_NO_VECDOT_IMMA")) {
        int n_blocks = dim / 32;
        if (n_blocks >= 1 && dim % 32 == 0) {
            void *wq8 = (void *)wd; /* Q8_0: use directly, others: use pre-quantized copy */
            int wq8_rb = rb; /* Q8_0: use original row_bytes */

            /* For F32/BF16: pre-quantize weights to Q8_0 once per (wd, qt, dim, nvh) */
            if (qt != GGUF_TYPE_Q8_0) {
                /* Static cache: one pre-quantized Q8_0 copy per device context.
                 * The weights don't change between calls, so this is a one-time cost. */
                static float *vd_wq8_cache = NULL;
                static const void *vd_wq8_key = NULL;
                static int vd_wq8_qt = -1;
                static size_t vd_wq8_cap = 0;
                size_t q8_rb = (size_t)n_blocks * GPU_BLOCK_Q8_0_SIZE;
                size_t q8_bytes = (size_t)nvh * q8_rb;

                if (vd_wq8_key != wd || vd_wq8_qt != (int)qt || q8_bytes > vd_wq8_cap) {
                    if (!reserve(&vd_wq8_cache, &vd_wq8_cap, q8_bytes)) goto vecdot_serial;
                    vd_wq8_key = wd;
                    vd_wq8_qt = (int)qt;
                    /* Quantize F32/BF16 weights to Q8_0 on-device */
                    int src_stride = (qt == 30) ? rb / (int)sizeof(uint16_t) : rb / (int)sizeof(float);
                    picolm_quantize_weights_to_q8_0<<<(unsigned)nvh, 32, 0, ctx->stream>>>(
                        (void *)vd_wq8_cache, (const float *)wd, qt, dim, nvh, src_stride, q8_rb);
                    if (!gpu_ok(gpuGetLastError(), "vecdot wq8 quantize")) goto vecdot_serial;
                }
                wq8 = (void *)vd_wq8_cache;
                wq8_rb = (int)q8_rb;
            }

            /* Quantize strided input to Q8_0 */
            int nt_padded = (nt + 15) & ~15;
            size_t xq_bytes = (size_t)nt_padded * dim;
            size_t xd_bytes = (size_t)nt_padded * n_blocks * sizeof(float);
            if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_bytes) ||
                !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_bytes)) goto vecdot_serial;

            dim3 q_grid((unsigned)n_blocks, (unsigned)nt);
            if (in_stride > 0 && in_stride != dim) {
                picolm_quantize_q8_0_strided<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
                    ctx->q8_xq, ctx->q8_xd, xd, dim, nt, in_stride);
            } else {
                picolm_quantize_q8_0<<<q_grid, 32, 32 * sizeof(float), ctx->stream>>>(
                    ctx->q8_xq, ctx->q8_xd, xd, dim, nt);
            }
            if (!gpu_ok(gpuGetLastError(), "vecdot q8 quantize")) return 0;

            int ys = out_stride > 0 ? out_stride : nvh;
            float *y_target = od;
            int y_tmp_needed = 0;
            if (hm) {
                size_t tmp_bytes = (size_t)nt * nvh * sizeof(float);
                if (reserve(&ctx->gate, &ctx->gate_cap, tmp_bytes)) {
                    y_target = ctx->gate;
                    ys = nvh;
                    y_tmp_needed = 1;
                }
            }

            /* Dispatch: IMMA (NVIDIA Turing+), else tiled, else scalar */
            if (ctx->has_imma && nt >= 16 && nvh >= 8) {
                gpu_dispatch_print("vecdot_imma_dev");
                dim3 grid((unsigned)((nvh + 7) / 8), (unsigned)((nt + 15) / 16));
                picolm_q8_q8_matmul_imma<<<grid, 32, 0, ctx->stream>>>(
                    y_target, ctx->q8_xq, ctx->q8_xd, wq8, nt, dim, nvh, wq8_rb, ys);
            } else if (nt >= Q8_TILE_S && wq8_rb + 2048 <= 49152) {
                gpu_dispatch_print("vecdot_tiled_dev");
                dim3 grid((unsigned)nvh, (unsigned)((nt + Q8_TILE_S - 1) / Q8_TILE_S));
                picolm_q8_q8_matmul_tiled<<<grid, 256, (unsigned)wq8_rb, ctx->stream>>>(
                    y_target, ctx->q8_xq, ctx->q8_xd, wq8, nt, dim, nvh, wq8_rb, ys);
            } else {
                gpu_dispatch_print("vecdot_scalar_dev");
                dim3 grid((unsigned)nvh, (unsigned)nt);
                picolm_q8_q8_matmul<<<grid, 256, 0, ctx->stream>>>(
                    y_target, ctx->q8_xq, ctx->q8_xd, wq8, nt, dim, nvh, wq8_rb, ys);
            }
            if (!gpu_ok(gpuGetLastError(), "vecdot q8 matmul")) return 0;

            if (y_tmp_needed) {
                picolm_gpu_ssm_head_permute_batch_kernel<<<
                    dim3((unsigned)nvh, (unsigned)nt), 1, 0, ctx->stream>>>(
                    od, y_target, hm, 1, nvh, nt, nvh, ys);
                if (!gpu_ok(gpuGetLastError(), "vecdot head permute")) return 0;
            }
            return 1;
        }
    }

vecdot_serial:
    /* Serial vecdot fallback for Q4_0 weights, misaligned dims, or OOM */
    gpu_dispatch_print("vecdot_serial_dev");
    picolm_ssm_vecdot_batch_kernel<<<dim3((unsigned)nvh,(unsigned)nt),256,0,ctx->stream>>>(
        od,xd,wd,qt,dim,nvh,nt,rb,hm,in_stride,out_stride);
    return gpu_ok(gpuGetLastError(),"vecdot batch dev");
}

extern "C"
int picolm_gpu_ssm_prefill_gated_norm_dev(float *od, const float *zd, const float *nd,
    int hd, int nh, int nt, float eps, int so_stride, int z_stride, int dev) {
    if(hd<1||nh<1||nt<1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    picolm_gpu_ssm_prefill_gated_norm_kernel<<<dim3((unsigned)nh,(unsigned)nt),min(hd,256),0,ctx->stream>>>(od,zd,nd,hd,nh,nt,eps,so_stride,z_stride);
    return gpu_ok(gpuGetLastError(),"gated norm dev");
}

__global__ void
picolm_gpu_ssm_head_permute_batch_kernel(float *dst, const float *src,
                                          const int *head_map,
                                          int head_dim, int n_heads, int n_tokens,
                                          int src_stride, int dst_stride) {
    int h = blockIdx.x; int t = blockIdx.y;
    if (h >= n_heads || t >= n_tokens) return;
    int gh = head_map ? head_map[h] : h;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        dst[(size_t)t * dst_stride + h * head_dim + d] = src[(size_t)t * src_stride + gh * head_dim + d];
}

extern "C"
int picolm_gpu_ssm_head_permute_batch_dev(float *dd, const float *sd, const int *hm,
    int hd, int nh, int nt, int ss, int ds, int dev) {
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_gpu_ssm_head_permute_batch_kernel<<<dim3((unsigned)nh,(unsigned)nt),128,0,ctx->stream>>>(dd,sd,hm,hd,nh,nt,ss,ds);
    return gpu_ok(gpuGetLastError(),"head permute batch dev");
}

__global__ void
/* Gate/beta post-process: outputs gate_log (log-space gate value) directly.
 * Previously this kernel did expf(gate) and the recurrence did logf(expf(gate))
 * -- a useless exp/log round-trip that compounded floating-point error.
 * Now it outputs gate_log = softplus(alpha + dt_w) * a_w, matching the CPU path. */
picolm_ssm_gate_beta_batch_kernel(float *ge, float *be,
    const float *ai, const float *bi, const float *dw, const float *aw,
    int nvh, int nt, int stride) {
    int h = blockIdx.x, t = blockIdx.y;
    if (h >= nvh || t >= nt) return;
    if (threadIdx.x != 0) return;
    int s = stride > 0 ? stride : nvh;
    float a = ai[t*s+h] + dw[h];
    float sp = (a>20.0f)?a:(a<-20.0f)?expf(a):logf(1.0f+expf(a));
    ge[t*s+h] = sp * aw[h];
    be[t*s+h] = 1.0f/(1.0f+expf(-bi[t*s+h]));
}

extern "C"
int picolm_gpu_ssm_gate_beta_batch_dev(float *ge, float *be, const float *ai, const float *bi,
    const float *dw, const float *aw, int nvh, int nt, int dev, int stride) {
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if (!ctx || !select_ctx(ctx)) return 0;
    picolm_ssm_gate_beta_batch_kernel<<<dim3((unsigned)nvh,(unsigned)nt),1,0,ctx->stream>>>(ge,be,ai,bi,dw,aw,nvh,nt,stride);
    return gpu_ok(gpuGetLastError(),"gate beta batch dev");
}

extern "C"
int picolm_gpu_expert_mlp_dev(picolm_gpu_tensor_t *g, picolm_gpu_tensor_t *u, picolm_gpu_tensor_t *d,
    float *yd, const float *xd, int S, int x_stride, int y_stride, int dev) {
    if(!g||!u||!d||!xd||!yd||S<1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(dev);
    if(!ctx||!select_ctx(ctx)) return 0;
    int D=g->I, I=g->O;
    if(!reserve(&ctx->gate,&ctx->gate_cap,(size_t)S*I*sizeof(float))||
       !reserve(&ctx->up,&ctx->up_cap,(size_t)S*I*sizeof(float))) return 0;
    /* F32 fallback: always use picolm_gpu_matmul_dev for non-Q8_0 weights
     * (BF16, F32, etc.). For Q8_0 weights, the int8-MAC path is used below. */
    if (g->qtype != GGUF_TYPE_Q8_0 || u->qtype != GGUF_TYPE_Q8_0 || d->qtype != GGUF_TYPE_Q8_0 ||
        getenv("PICOLM_FORCE_F32_MLP") || getenv("PICOLM_FORCE_F32_MATMUL")) {
        if (!picolm_gpu_matmul_dev(g, ctx->gate, xd, S, dev, I, x_stride)) return 0;
        if (!picolm_gpu_matmul_dev(u, ctx->up, xd, S, dev, I, x_stride)) return 0;
        picolm_silu_mul<<<(unsigned)((S*I+255)/256),256,0,ctx->stream>>>(ctx->gate, ctx->up, S*I);
        if (!gpu_ok(gpuGetLastError(), "expert silu")) return 0;
        int ys = y_stride > 0 ? y_stride : D;
        if (!picolm_gpu_matmul_dev(d, yd, ctx->gate, S, dev, ys, I)) return 0;
        return 1;
    }
    /* Q8_0 fast path: quantize F32 input to Q8_0, then int8 MAC */
    int d_blocks = D / 32;
    int i_blocks = I / 32;
    if (d_blocks < 1 || i_blocks < 1) return 0;

    int S_padded = (S + 15) & ~15;
    size_t xq_d = (size_t)S_padded * D;
    size_t xd_d = (size_t)S_padded * d_blocks * sizeof(float);
    size_t xq_i = (size_t)S_padded * I;
    size_t xd_i = (size_t)S_padded * i_blocks * sizeof(float);
    if (!reserve_i8(&ctx->q8_xq, &ctx->q8_xq_cap, xq_d > xq_i ? xq_d : xq_i) ||
        !reserve(&ctx->q8_xd, &ctx->q8_xd_cap, xd_d > xd_i ? xd_d : xd_i)) return 0;

    /* Quantize F32 input [D*S] to Q8_0 */
    if (x_stride > 0 && x_stride != D) {
        picolm_quantize_q8_0_strided<<<dim3((unsigned)d_blocks,(unsigned)S),32,32*sizeof(float),ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, xd, D, S, x_stride);
    } else {
        picolm_quantize_q8_0<<<dim3((unsigned)d_blocks,(unsigned)S),32,32*sizeof(float),ctx->stream>>>(
            ctx->q8_xq, ctx->q8_xd, xd, D, S);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 quantize input (dev)")) return 0;

    /* gate = q8_q8_tiled(input_q8, gate_weights) */
    /* gate = q8_q8 matmul(input_q8, gate_weights) */
    if (ctx->has_imma && S >= 16 && I >= 8) {
        dim3 grid((unsigned)((I + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma<<<grid,32,0,ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, g->weights, S, D, I, (int)g->row_bytes, I);
    } else if (S > 1 && g->row_bytes + 2048 <= 49152) {
        dim3 grid((unsigned)I, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid,256,(unsigned)g->row_bytes,ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, g->weights, S, D, I, (int)g->row_bytes, I);
    } else {
        picolm_q8_q8_matmul<<<dim3((unsigned)I,(unsigned)S),256,0,ctx->stream>>>(
            ctx->gate, ctx->q8_xq, ctx->q8_xd, g->weights, S, D, I, (int)g->row_bytes, I);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 gate (dev)")) return 0;

    /* up = q8_q8 matmul(input_q8, up_weights) */
    if (ctx->has_imma && S >= 16 && I >= 8) {
        dim3 grid((unsigned)((I + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma<<<grid,32,0,ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, u->weights, S, D, I, (int)u->row_bytes, I);
    } else if (S > 1 && u->row_bytes + 2048 <= 49152) {
        dim3 grid((unsigned)I, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid,256,(unsigned)u->row_bytes,ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, u->weights, S, D, I, (int)u->row_bytes, I);
    } else {
        picolm_q8_q8_matmul<<<dim3((unsigned)I,(unsigned)S),256,0,ctx->stream>>>(
            ctx->up, ctx->q8_xq, ctx->q8_xd, u->weights, S, D, I, (int)u->row_bytes, I);
    }
    if (!gpu_ok(gpuGetLastError(), "expert q8 up (dev)")) return 0;

    /* silu(gate) * up -> ctx->gate */
    picolm_silu_mul<<<(unsigned)((S*I+255)/256),256,0,ctx->stream>>>(ctx->gate, ctx->up, S*I);
    if (!gpu_ok(gpuGetLastError(), "expert silu (dev)")) return 0;

    /* Quantize F32 hidden [I*S] to Q8_0 for down */
    picolm_quantize_q8_0<<<dim3((unsigned)i_blocks,(unsigned)S),32,32*sizeof(float),ctx->stream>>>(
        ctx->q8_xq, ctx->q8_xd, ctx->gate, I, S);
    if (!gpu_ok(gpuGetLastError(), "expert q8 quantize hidden (dev)")) return 0;

    /* y = q8_q8 matmul(hidden_q8, down_weights) */
    int ys = y_stride > 0 ? y_stride : D;
    if (ctx->has_imma && S >= 16 && D >= 8) {
        dim3 grid((unsigned)((D + 7) / 8), (unsigned)((S + 15) / 16));
        picolm_q8_q8_matmul_imma<<<grid,32,0,ctx->stream>>>(
            yd, ctx->q8_xq, ctx->q8_xd, d->weights, S, I, D, (int)d->row_bytes, ys);
    } else if (S > 1 && d->row_bytes + 2048 <= 49152) {
        dim3 grid((unsigned)D, (unsigned)((S + Q8_TILE_S - 1) / Q8_TILE_S));
        picolm_q8_q8_matmul_tiled<<<grid,256,(unsigned)d->row_bytes,ctx->stream>>>(
            yd, ctx->q8_xq, ctx->q8_xd, d->weights, S, I, D, (int)d->row_bytes, ys);
    } else {
        picolm_q8_q8_matmul<<<dim3((unsigned)D,(unsigned)S),256,0,ctx->stream>>>(
            yd, ctx->q8_xq, ctx->q8_xd, d->weights, S, I, D, (int)d->row_bytes, ys);
    }
    return gpu_ok(gpuGetLastError(), "expert MLP dev");
}

extern "C"
int picolm_gpu_ssm_chunked_recurrence_dev(const float *conv_dev, const float *alpha_dev,
    const float *beta_dev, float *state_dev, float *xb2_dev,
    int n_tokens, int value_dim, int xb2_stride,
    int d_state, int n_k_heads, int n_v_heads,
    int head_v_dim, int repeat, int conv_dim, int cs, int device) {
    (void)conv_dim; /* conv_dim is used for offset calculations, xb2_stride for striding */
#ifndef PICOLM_SSM_CHUNKED_GPU_VALIDATED
    (void)conv_dev;(void)alpha_dev;(void)beta_dev;(void)state_dev;(void)xb2_dev;
    (void)n_tokens;(void)value_dim;(void)xb2_stride;(void)d_state;(void)n_k_heads;
    (void)n_v_heads;(void)head_v_dim;(void)repeat;(void)cs;(void)device;
    return 0;
#else
    if(d_state!=head_v_dim) return 0;
    if(cs<=0) cs=64; if(cs>n_tokens) cs=n_tokens;
    int nc=(n_tokens+cs-1)/cs; if(nc<1) nc=1;
    int d=d_state, qk_dim=d_state*n_k_heads;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if(!ctx||!select_ctx(ctx)) return 0;
    int n_threads=256;
    /* Persistent scratch buffers: must be static so they survive across
     * calls (one per SSM layer per layer loop). ssm_batch_scratch_ensure
     * is a grow-only allocator that checks *cap >= need before allocating.
     * If these were local variables, every call would re-allocate and leak. */
    static float *d_cq=NULL,*d_ck=NULL,*d_cv=NULL,*d_cb=NULL;
    static float *d_gl=NULL,*d_cg=NULL,*d_qd=NULL,*d_dm=NULL;
    static float *d_M=NULL,*d_kq=NULL,*d_ve=NULL,*d_vh=NULL;
    static float *d_sk=NULL,*d_sq=NULL,*d_co=NULL;
    static size_t caps[15]={0};
    size_t szs[15]={
        (size_t)n_k_heads*cs*d*sizeof(float),  /* cq */
        (size_t)n_k_heads*cs*d*sizeof(float),  /* ck */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* cv */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* cb */
        (size_t)n_v_heads*cs*sizeof(float),    /* gl */
        (size_t)n_v_heads*cs*sizeof(float),    /* cg */
        (size_t)n_v_heads*cs*sizeof(float),    /* qd */
        (size_t)n_v_heads*cs*cs*sizeof(float), /* dm */
        (size_t)n_v_heads*cs*cs*sizeof(float), /* M */
        (size_t)n_v_heads*cs*cs*sizeof(float), /* kq */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* ve */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* vh */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* sk */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* sq */
        (size_t)n_v_heads*cs*d*sizeof(float),  /* co */
    };
    float **ptrs[15]={&d_cq,&d_ck,&d_cv,&d_cb,&d_gl,&d_cg,&d_qd,&d_dm,
        &d_M,&d_kq,&d_ve,&d_vh,&d_sk,&d_sq,&d_co};
    int ok=1;
    for(int i=0;i<15&&ok;i++){
        ok&=ssm_batch_scratch_ensure((void**)ptrs[i],&caps[i],szs[i]);
        if(ok) ok&=gpu_ok(gpuMemsetAsync(*(void**)ptrs[i],0,szs[i],ctx->stream),"scratch zero");
    }
    if(!ok) return 0;
    for(int ci=0;ci<nc&&ok;ci++){
      int ca=(ci==nc-1)?(n_tokens-ci*cs):cs; if(ca<=0)break;
      int co2=ci*cs;
      ssm_chunk_gather_qk_kernel<<<n_k_heads,n_threads,0,ctx->stream>>>((float*)d_cq,(float*)d_ck,conv_dev,co2,ca,xb2_stride,qk_dim,d_state,n_k_heads);
      if(!gpu_ok(gpuGetLastError(),"gather_qk"))return 0;
      ssm_chunk_gather_v_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_cv,(float*)d_cb,(float*)d_gl,conv_dev,alpha_dev,beta_dev,co2,ca,xb2_stride,qk_dim,head_v_dim,n_v_heads,xb2_stride);
      if(!gpu_ok(gpuGetLastError(),"gather_v"))return 0;
      ssm_chunk_decay_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_cg,(float*)d_qd,(float*)d_dm,(float*)d_gl,n_v_heads,ca,ca);
      if(!gpu_ok(gpuGetLastError(),"decay"))return 0;
#ifdef PICOLM_GPU
      if (getenv("PICOLM_SSM_STEP_VERIFY") && ci == 0) {
          /* D2H decay_mask and gate_log scratch */
          float _dm[16], _gl[40], _cg[40];
          gpuDeviceSynchronize();
          gpuMemcpy(_dm, d_dm, 64, gpuMemcpyDeviceToHost);
          gpuMemcpy(_gl, d_gl, 160, gpuMemcpyDeviceToHost);
          gpuMemcpy(_cg, d_cg, 160, gpuMemcpyDeviceToHost);
          float _dm_full[1300];
          gpuMemcpy(_dm_full, d_dm, 5200, gpuMemcpyDeviceToHost);
          fprintf(stderr, "[STEP l0] dm row0=[%.6f %.6f %.6f %.6f] row1=[%.6f %.6f %.6f %.6f] dm35_0=%.6f dm35_35=%.6f cg0=%.6f cg35=%.6f\n",
              _dm_full[0],_dm_full[1],_dm_full[2],_dm_full[3],
              _dm_full[36],_dm_full[37],_dm_full[38],_dm_full[39],
              _dm_full[35*ca+0], _dm_full[35*ca+35], _cg[0], _cg[35]);
      }
#endif
      ssm_chunk_masked_gemm_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_M,(const float*)d_ck,(const float*)d_ck,(const float*)d_dm,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"M_gemm"))return 0;
      ssm_chunk_matvec_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_sk,state_dev,(const float*)d_ck,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"sk_matvec"))return 0;
      ssm_chunk_veff_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_ve,(const float*)d_cv,(const float*)d_sk,(const float*)d_qd,(const float*)d_cb,n_v_heads,ca,d);
      if(!gpu_ok(gpuGetLastError(),"veff"))return 0;
      {int tr=d<1024?d:1024; ssm_chunk_trisolve_kernel<<<n_v_heads,tr,0,ctx->stream>>>((float*)d_vh,(const float*)d_ve,(const float*)d_M,(const float*)d_cb,n_v_heads,ca,d);}
      if(!gpu_ok(gpuGetLastError(),"trisolve"))return 0;
      ssm_chunk_matvec_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_sq,state_dev,(const float*)d_cq,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"sq_matvec"))return 0;
      ssm_chunk_masked_gemm_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_kq,(const float*)d_cq,(const float*)d_ck,(const float*)d_dm,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"kq_gemm"))return 0;
      ssm_chunk_output_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>((float*)d_co,(const float*)d_sq,(const float*)d_qd,(const float*)d_kq,(const float*)d_vh,n_v_heads,ca,d);
      if(!gpu_ok(gpuGetLastError(),"output"))return 0;
#ifdef PICOLM_GPU
      if (getenv("PICOLM_SSM_STEP_VERIFY") && ci == 0) {
          /* D2H v_eff[0][0..3] and v_hat[0][0..3] for head 0 */
          float _ve[4], _vh[4], _co[4];
          gpuDeviceSynchronize();
          gpuMemcpy(_ve, d_ve, 16, gpuMemcpyDeviceToHost);
          gpuMemcpy(_vh, d_vh, 16, gpuMemcpyDeviceToHost);
          gpuMemcpy(_co, d_co, 16, gpuMemcpyDeviceToHost);
          fprintf(stderr, "[STEP l0] GPU ve[0][0..3]={%.6f,%.6f,%.6f,%.6f} vh[0][0..3]={%.6f,%.6f,%.6f,%.6f} co[0][0..3]={%.6f,%.6f,%.6f,%.6f}\n",
              _ve[0],_ve[1],_ve[2],_ve[3], _vh[0],_vh[1],_vh[2],_vh[3], _co[0],_co[1],_co[2],_co[3]);
      }
#endif
      ssm_chunk_scatter_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>(xb2_dev,(const float*)d_co,co2,ca,xb2_stride,head_v_dim,n_v_heads);
      if(!gpu_ok(gpuGetLastError(),"scatter"))return 0;
      ssm_chunk_state_update_kernel<<<n_v_heads,n_threads,0,ctx->stream>>>(state_dev,(const float*)d_vh,(const float*)d_ck,(const float*)d_cg,n_v_heads,repeat,ca,d);
      if(!gpu_ok(gpuGetLastError(),"state_up"))return 0;
    }
    return ok;
#endif
}
