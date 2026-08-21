// backend_gpu_host_ssm.cu - SSM GPU functions (recurrence, conv1d, gated_norm, vecdot, etc.)
#include "backend_gpu_kernels.cuh"

extern "C" int
picolm_gpu_ssm_recurrence(float *state,
                           const float *q_conv,
                           const float *k_conv,
                           const float *v_conv,
                           const float *gate_exp,
                           const float *beta,
                           float *ssm_output,
                           int n_v_heads, int d_state,
                           int repeat, int device) {
    if (n_v_heads <= 0 || d_state <= 0) return 0;
    if (d_state > 256) return 0;

    if (!gpu_ok(gpuSetDevice(device), "ssm device")) return 0;
    gpu_device_ctx_t *ctx = NULL;
    for (int i = 0; i < g_nctx; i++) {
        if (g_gpu_ctx[i].device == device) { ctx = &g_gpu_ctx[i]; break; }
    }
    if (!ctx) return 0;

    /* Upload all CPU data to device memory.
     * On AMD iGPU, CPU malloc'd memory is NOT directly GPU-accessible. */
    size_t state_bytes = (size_t)n_v_heads * d_state * d_state * sizeof(float);
    size_t q_bytes = (size_t)(n_v_heads / repeat) * d_state * sizeof(float);
    size_t k_bytes = q_bytes;
    size_t v_bytes = (size_t)n_v_heads * d_state * sizeof(float);
    size_t scalar_bytes = (size_t)n_v_heads * sizeof(float);
    size_t out_bytes = (size_t)d_state * n_v_heads * sizeof(float);

    void *ds, *dq, *dk, *dv, *dg, *db, *do_;
    if (!gpu_ok(gpuMalloc(&ds, state_bytes), "ssm st") ||
        !gpu_ok(gpuMalloc(&dq, q_bytes), "ssm q") ||
        !gpu_ok(gpuMalloc(&dk, k_bytes), "ssm k") ||
        !gpu_ok(gpuMalloc(&dv, v_bytes), "ssm v") ||
        !gpu_ok(gpuMalloc(&dg, scalar_bytes), "ssm g") ||
        !gpu_ok(gpuMalloc(&db, scalar_bytes), "ssm b") ||
        !gpu_ok(gpuMalloc(&do_, out_bytes), "ssm o")) return 0;

    if (!gpu_ok(gpuMemcpy(ds, state, state_bytes, gpuMemcpyHostToDevice), "ssm st h2d") ||
        !gpu_ok(gpuMemcpy(dq, q_conv, q_bytes, gpuMemcpyHostToDevice), "ssm q h2d") ||
        !gpu_ok(gpuMemcpy(dk, k_conv, k_bytes, gpuMemcpyHostToDevice), "ssm k h2d") ||
        !gpu_ok(gpuMemcpy(dv, v_conv, v_bytes, gpuMemcpyHostToDevice), "ssm v h2d") ||
        !gpu_ok(gpuMemcpy(dg, gate_exp, scalar_bytes, gpuMemcpyHostToDevice), "ssm g h2d") ||
        !gpu_ok(gpuMemcpy(db, beta, scalar_bytes, gpuMemcpyHostToDevice), "ssm b h2d")) {
        gpuFree(ds); gpuFree(dq); gpuFree(dk); gpuFree(dv);
        gpuFree(dg); gpuFree(db); gpuFree(do_); return 0;
    }

    /* Warp-shuffle kernel for common d_state values. Falls back to
     * thread-0 kernel for unsupported sizes.
     * Gated behind PICOLM_SSM_WARP_KERNEL_VALIDATED (defined by default
     * for CUDA builds). The warp kernel has been validated on real
     * hardware. The thread-0 kernel is also bit-exact with CPU NEON
     * (previously verified -- see ssm_gpu_session_findings.md). */
    int warp_launched = 0;
#ifdef PICOLM_SSM_WARP_KERNEL_VALIDATED
    if (d_state == 128) {
        constexpr int S_v = 128;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ds, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 64) {
        constexpr int S_v = 64;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ds, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 32) {
        constexpr int S_v = 32;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ds, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 16) {
        constexpr int S_v = 16;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ds, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    }
#endif /* PICOLM_SSM_WARP_KERNEL_VALIDATED */
    if (!warp_launched) {
        dim3 grid((unsigned)n_v_heads, 1, 1);
        picolm_ssm_recurrence_kernel<<<grid, 256, 0, ctx->stream>>>(
            (float *)ds, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, d_state, repeat);
    }

    if (!gpu_ok(gpuGetLastError(), "ssm recurrence") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm sync")) {
        gpuFree(ds); gpuFree(dq); gpuFree(dk); gpuFree(dv);
        gpuFree(dg); gpuFree(db); gpuFree(do_); return 0;
    }
    gpuMemcpy(ssm_output, do_, out_bytes, gpuMemcpyDeviceToHost);
    gpuMemcpy(state, ds, state_bytes, gpuMemcpyDeviceToHost);
    gpuFree(ds); gpuFree(dq); gpuFree(dk); gpuFree(dv);
    gpuFree(dg); gpuFree(db); gpuFree(do_);
    return 1;
}

/* Device-native SSM recurrence: takes device-resident state, no per-call
 * malloc/H2D/D2H for state. q/k/v/gate_exp/beta/ssm_output still CPU-side,
 * uploaded/downloaded per call. The key saving is eliminating the persistent
 * state H2D/D2H round-trip per token. */
extern "C" int
picolm_gpu_ssm_recurrence_dev(void *ssm_state_dev,  /* in/out, device [n_v_heads][d_state][d_state] */
                               const float *q_conv,  /* host [n_k_heads][d_state] */
                               const float *k_conv,  /* host [n_k_heads][d_state] */
                               const float *v_conv,  /* host [n_v_heads][head_v_dim==d_state] */
                               const float *gate_exp, /* host [n_v_heads] */
                               const float *beta,    /* host [n_v_heads] */
                               float *ssm_output,    /* out, host [d_state * n_v_heads] */
                               int n_v_heads, int d_state,
                               int repeat, int device) {
    if (n_v_heads <= 0 || d_state <= 0) return 0;
    if (d_state > 256) return 0;
    if (!ssm_state_dev) return 0;

    if (!gpu_ok(gpuSetDevice(device), "ssm device")) return 0;
    gpu_device_ctx_t *ctx = NULL;
    for (int i = 0; i < g_nctx; i++) {
        if (g_gpu_ctx[i].device == device) { ctx = &g_gpu_ctx[i]; break; }
    }
    if (!ctx) return 0;

    size_t q_bytes = (size_t)(n_v_heads / repeat) * d_state * sizeof(float);
    size_t k_bytes = q_bytes;
    size_t v_bytes = (size_t)n_v_heads * d_state * sizeof(float);
    size_t scalar_bytes = (size_t)n_v_heads * sizeof(float);
    size_t out_bytes = (size_t)d_state * n_v_heads * sizeof(float);

    void *dq, *dk, *dv, *dg, *db, *do_;
    if (!gpu_ok(gpuMalloc(&dq, q_bytes), "ssm q") ||
        !gpu_ok(gpuMalloc(&dk, k_bytes), "ssm k") ||
        !gpu_ok(gpuMalloc(&dv, v_bytes), "ssm v") ||
        !gpu_ok(gpuMalloc(&dg, scalar_bytes), "ssm g") ||
        !gpu_ok(gpuMalloc(&db, scalar_bytes), "ssm b") ||
        !gpu_ok(gpuMalloc(&do_, out_bytes), "ssm o")) return 0;

    if (!gpu_ok(gpuMemcpy(dq, q_conv, q_bytes, gpuMemcpyHostToDevice), "ssm q h2d") ||
        !gpu_ok(gpuMemcpy(dk, k_conv, k_bytes, gpuMemcpyHostToDevice), "ssm k h2d") ||
        !gpu_ok(gpuMemcpy(dv, v_conv, v_bytes, gpuMemcpyHostToDevice), "ssm v h2d") ||
        !gpu_ok(gpuMemcpy(dg, gate_exp, scalar_bytes, gpuMemcpyHostToDevice), "ssm g h2d") ||
        !gpu_ok(gpuMemcpy(db, beta, scalar_bytes, gpuMemcpyHostToDevice), "ssm b h2d")) {
        gpuFree(dq); gpuFree(dk); gpuFree(dv);
        gpuFree(dg); gpuFree(db); gpuFree(do_); return 0;
    }

    /* Warp-shuffle kernel for common d_state values.
     * Gated behind PICOLM_SSM_WARP_KERNEL_VALIDATED -- see the longer
     * comment at the first dispatch site in picolm_gpu_ssm_recurrence()
     * above. Enabled by default for CUDA builds. */
    int warp_launched = 0;
#ifdef PICOLM_SSM_WARP_KERNEL_VALIDATED
    if (d_state == 128) {
        constexpr int S_v = 128;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 64) {
        constexpr int S_v = 64;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 32) {
        constexpr int S_v = 32;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 16) {
        constexpr int S_v = 16;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, repeat);
        warp_launched = 1;
    }
#endif /* PICOLM_SSM_WARP_KERNEL_VALIDATED */
    if (!warp_launched) {
        dim3 grid((unsigned)n_v_heads, 1, 1);
        picolm_ssm_recurrence_kernel<<<grid, 256, 0, ctx->stream>>>(
            (float *)ssm_state_dev, (const float *)dq, (const float *)dk,
            (const float *)dv, (const float *)dg, (const float *)db,
            (float *)do_, n_v_heads, d_state, repeat);
    }

    if (!gpu_ok(gpuGetLastError(), "ssm recurrence") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm sync")) {
        gpuFree(dq); gpuFree(dk); gpuFree(dv);
        gpuFree(dg); gpuFree(db); gpuFree(do_); return 0;
    }
    gpuMemcpy(ssm_output, do_, out_bytes, gpuMemcpyDeviceToHost);
    /* State remains on device - no D2H needed */
    gpuFree(dq); gpuFree(dk); gpuFree(dv);
    gpuFree(dg); gpuFree(db); gpuFree(do_);
    return 1;
}

/* Fully device-native: ALL of q_conv/k_conv/v_conv/gate_exp/beta/
 * ssm_output are already device-resident pipeline buffers -- no malloc,
 * no H2D/D2H, no sync at all, unlike picolm_gpu_ssm_recurrence_dev above
 * (which still round-trips q/k/v/gate/beta/output through host memory
 * every call, only state is persistent there). This is the one to use
 * from model_forward_gpu's SSM layer branch. */
extern "C" int
picolm_gpu_ssm_recurrence_pipeline_dev(void *ssm_state_dev,
                                        const float *q_conv_dev,
                                        const float *k_conv_dev,
                                        const float *v_conv_dev,
                                        const float *gate_exp_dev,
                                        const float *beta_dev,
                                        float *ssm_output_dev,
                                        int n_v_heads, int d_state,
                                        int repeat, int device) {
    if (n_v_heads <= 0 || d_state <= 0) return 0;
    if (d_state > 256) return 0;
    if (!ssm_state_dev) return 0;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    /* Warp-shuffle kernel for common d_state values.
     * Gated behind PICOLM_SSM_WARP_KERNEL_VALIDATED -- see the longer
     * comment at the first dispatch site in picolm_gpu_ssm_recurrence()
     * above. Enabled by default for CUDA builds. */
    int warp_launched = 0;
#ifdef PICOLM_SSM_WARP_KERNEL_VALIDATED
    if (d_state == 128) {
        constexpr int S_v = 128;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, q_conv_dev, k_conv_dev,
            v_conv_dev, gate_exp_dev, beta_dev,
            ssm_output_dev, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 64) {
        constexpr int S_v = 64;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, q_conv_dev, k_conv_dev,
            v_conv_dev, gate_exp_dev, beta_dev,
            ssm_output_dev, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 32) {
        constexpr int S_v = 32;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, q_conv_dev, k_conv_dev,
            v_conv_dev, gate_exp_dev, beta_dev,
            ssm_output_dev, n_v_heads, repeat);
        warp_launched = 1;
    } else if (d_state == 16) {
        constexpr int S_v = 16;
        constexpr int num_warps = 4;
        dim3 g((unsigned)n_v_heads, 1,
               (unsigned)((S_v + num_warps * (GPU_WARP_SIZE / 4) - 1) /
                          (num_warps * (GPU_WARP_SIZE / 4))));
        dim3 b(GPU_WARP_SIZE, num_warps, 1);
        picolm_ssm_recurrence_warp_kernel<S_v><<<g, b, 0, ctx->stream>>>(
            (float *)ssm_state_dev, q_conv_dev, k_conv_dev,
            v_conv_dev, gate_exp_dev, beta_dev,
            ssm_output_dev, n_v_heads, repeat);
        warp_launched = 1;
    }
#endif /* PICOLM_SSM_WARP_KERNEL_VALIDATED */
    if (!warp_launched) {
        dim3 grid((unsigned)n_v_heads, 1, 1);
        picolm_ssm_recurrence_kernel<<<grid, 256, 0, ctx->stream>>>(
            (float *)ssm_state_dev, q_conv_dev, k_conv_dev,
            v_conv_dev, gate_exp_dev, beta_dev,
            ssm_output_dev, n_v_heads, d_state, repeat);
    }
    if (!gpu_ok(gpuGetLastError(), "ssm recurrence (pipeline dev)")) return 0;
    return 1;
}

/* ---- SSM gate/beta activation (Finding 6) ----
 * Direct port of the CPU reference (ssm_forward, steps 9-11):
 *   alpha[h]    = alpha_raw[h] + ssm_dt_w[h]   (bias add, easy to miss --
 *                 caught on a second read-through of the CPU reference;
 *                 the vecdot output alone is NOT what softplus takes)
 *   gate[h]     = softplus(alpha[h]) * ssm_a_w[h]
 *   gate_exp[h] = (gate[h] < -50) ? 0 : exp(gate[h])
 *   beta_out[h] = sigmoid(beta_raw[h])  (no bias -- confirmed against
 *                 the CPU reference, beta's vecdot output goes straight
 *                 into sigmoid)
 * Tiny (n_v_heads elements, e.g. 48) and embarrassingly parallel -- one
 * thread per head, no shared memory or sync needed. */
__global__ void
picolm_gpu_ssm_gate_beta_kernel(float *gate_exp_out, float *beta_out,
                                 const float *alpha_in, const float *beta_raw_in,
                                 const float *ssm_a_w, const float *ssm_dt_w,
                                 int n_v_heads) {
    int h = (int)gpuBlockIdx_x * gpuBlockDim_x + gpuThreadIdx_x;
    if (h >= n_v_heads) return;

    float a = alpha_in[h] + ssm_dt_w[h];
    float sp = (a > 20.0f) ? a : (a < -20.0f) ? expf(a) : logf(1.0f + expf(a));
    float gate = sp * ssm_a_w[h];
    gate_exp_out[h] = (gate < -50.0f) ? 0.0f : expf(gate);

    float braw = beta_raw_in[h];
    beta_out[h] = 1.0f / (1.0f + expf(-braw));
}

/* Device-native: all pointers device-resident, no H2D/D2H, no sync. */
extern "C" int
picolm_gpu_ssm_gate_beta_dev(float *gate_exp_out_dev, float *beta_out_dev,
                              const float *alpha_in_dev, const float *beta_raw_in_dev,
                              const float *ssm_a_w_dev, const float *ssm_dt_w_dev,
                              int n_v_heads, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (n_v_heads < 1) return 0;

    int n_threads = min(n_v_heads, 256);
    int n_blocks = (n_v_heads + n_threads - 1) / n_threads;
    picolm_gpu_ssm_gate_beta_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        gate_exp_out_dev, beta_out_dev, alpha_in_dev, beta_raw_in_dev,
        ssm_a_w_dev, ssm_dt_w_dev, n_v_heads);
    if (!gpu_ok(gpuGetLastError(), "ssm gate/beta (dev)")) return 0;
    return 1;
}

/* ---- SSM L2 normalization (Q/K, per k_head) ----
 * Direct port of the CPU reference (ssm_forward, step 7-8): per-head L2
 * norm (NOT RMS -- no division by head_dim, no learned weight), with an
 * optional fused extra_scale applied after normalizing. The CPU
 * reference does Q's 1/sqrt(d_state) scale as a separate pass (decode
 * path) or fused into the norm (prefill path) -- this kernel always
 * fuses it (extra_scale=q_scale for Q, extra_scale=1.0 for K), matching
 * the prefill reference exactly and reducing to the same result as the
 * decode reference's two-pass version (multiplication is associative
 * here, single scalar factor either way).
 * In-place: x is normalized in place. Grid = n_heads. */
__global__ void
picolm_gpu_ssm_l2norm_kernel(float *x, int head_dim, int n_heads,
                              float eps, float extra_scale) {
    int h = (int)gpuBlockIdx_x;
    if (h >= n_heads) return;
    if (gpuThreadIdx_x != 0) return;
    float *xh = x + (size_t)h * head_dim;

    /* Thread-0 sequential: matches the CPU scalar loop's left-to-right
     * sum exactly (a parallel tree reduction sums in a different order
     * and gives a different, non-bit-identical float result). */
    float nrm = 0.0f;
    for (int d = 0; d < head_dim; d++) nrm += xh[d] * xh[d];
    nrm = (1.0f / sqrtf(nrm + eps)) * extra_scale;
    for (int d = 0; d < head_dim; d++) xh[d] *= nrm;
}

/* Batched-over-tokens version: identical per-(token,head) computation,
 * grid.y = n_tokens instead of a host-side loop over tokens.
 * token_stride is the element stride between consecutive tokens' head
 * groups in x -- pass n_heads*head_dim for a tightly-packed
 * [n_tokens][n_heads][head_dim] buffer, or something larger (e.g.
 * conv_dim) if the head group this call operates on (Q or K) is
 * embedded inside a bigger per-token block alongside other data (as in
 * ssm_prefill_layer's conv_batch, where each token's block is
 * [Q][K][V] and Q/K individually don't sit at a tight n_heads*head_dim
 * stride from one token to the next). In-place. */
__global__ void
picolm_gpu_ssm_l2norm_batch_kernel(float *x, int head_dim, int n_heads,
                                    int n_tokens, int token_stride,
                                    float eps, float extra_scale) {
    int h = (int)gpuBlockIdx_x;
    int t = (int)gpuBlockIdx_y;
    if (h >= n_heads || t >= n_tokens) return;
    if (gpuThreadIdx_x != 0) return;
    float *xh = x + (size_t)t * token_stride + (size_t)h * head_dim;

    float nrm = 0.0f;
    for (int d = 0; d < head_dim; d++) nrm += xh[d] * xh[d];
    float nrm_inv = (1.0f / sqrtf(nrm + eps)) * extra_scale;
    for (int d = 0; d < head_dim; d++) xh[d] *= nrm_inv;
}

/* Device-native, in-place. eps must match the CPU reference (1e-12).
 * extra_scale: pass 1/sqrtf(d_state) for Q, 1.0f for K. */
extern "C" int
picolm_gpu_ssm_l2norm_dev(float *x_dev, int head_dim, int n_heads,
                           float eps, float extra_scale, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (head_dim < 1 || n_heads < 1 || head_dim > 256) return 0;

    int n_threads = min(head_dim, 256);
    dim3 grid((unsigned)n_heads, 1, 1);
    picolm_gpu_ssm_l2norm_kernel<<<grid, n_threads, 0, ctx->stream>>>(
        x_dev, head_dim, n_heads, eps, extra_scale);
    if (!gpu_ok(gpuGetLastError(), "ssm l2norm (dev)")) return 0;
    return 1;
}

/* Host-facing, batched-over-tokens L2-norm: takes a CPU pointer to the
 * start of the first token's head group, does one H2D/D2H round trip
 * for the whole batch (in place). See the kernel comment above for
 * token_stride semantics. Same eps/extra_scale contract as the
 * per-token _dev version above. */
extern "C" int
picolm_gpu_ssm_l2norm_batch(float *x_host, int head_dim, int n_heads,
                             int n_tokens, int token_stride,
                             float eps, float extra_scale,
                             int device) {
    if (head_dim < 1 || n_heads < 1 || head_dim > 256 || n_tokens < 1) return 0;
    if (token_stride < n_heads * head_dim) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    /* Copy the full strided span [0, (n_tokens-1)*token_stride + n_heads*head_dim)
     * in one shot -- includes bytes belonging to other data (K/V) sitting
     * between consecutive tokens' head groups when token_stride is larger
     * than n_heads*head_dim, but that's still one H2D/D2H instead of
     * n_tokens of them, and avoids needing a strided-memcpy helper. */
    size_t span = (size_t)(n_tokens - 1) * token_stride + (size_t)n_heads * head_dim;
    size_t bytes = span * sizeof(float);
    static void *d_x = NULL; static size_t d_x_cap = 0;
    if (!ssm_batch_scratch_ensure(&d_x, &d_x_cap, bytes)) return 0;
    if (!gpu_ok(gpuMemcpy(d_x, x_host, bytes, gpuMemcpyHostToDevice), "l2norm batch h2d")) return 0;

    int n_threads = min(head_dim, 256);
    dim3 grid((unsigned)n_heads, (unsigned)n_tokens, 1);
    picolm_gpu_ssm_l2norm_batch_kernel<<<grid, n_threads, 0, ctx->stream>>>(
        (float *)d_x, head_dim, n_heads, n_tokens, token_stride, eps, extra_scale);

    if (!gpu_ok(gpuGetLastError(), "ssm l2norm batch") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm l2norm batch sync")) return 0;
    return gpu_ok(gpuMemcpy(x_host, d_x, bytes, gpuMemcpyDeviceToHost), "l2norm batch d2h");
}

/* ---- SSM head permute (GGUF v-head remap) ----
 * Generic per-head gather: dst[h] = src[head_map[h]], head_dim elements
 * each. Used for BOTH the xb2 (z-gate) remap and the v_conv remap --
 * same head_map (qwen35_vhead_gguf), same head_dim (head_v_dim), same
 * n_heads (n_v_heads) in both cases. dst and src must NOT alias (the
 * CPU reference uses a temp buffer for exactly this reason -- a
 * permutation isn't safe to do purely in place). */
__global__ void
picolm_gpu_ssm_head_permute_kernel(float *dst, const float *src,
                                    const int *head_map,
                                    int head_dim, int n_heads) {
    int h = (int)gpuBlockIdx_x;
    if (h >= n_heads) return;
    int gh = head_map[h];
    int tid = gpuThreadIdx_x, nt = gpuBlockDim_x;
    const float *srch = src + (size_t)gh * head_dim;
    float *dsth = dst + (size_t)h * head_dim;
    for (int d = tid; d < head_dim; d += nt) dsth[d] = srch[d];
}

extern "C" int
picolm_gpu_ssm_head_permute_dev(float *dst_dev, const float *src_dev,
                                 const int *head_map_dev,
                                 int head_dim, int n_heads, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (head_dim < 1 || n_heads < 1) return 0;

    int n_threads = min(head_dim, 256);
    dim3 grid((unsigned)n_heads, 1, 1);
    picolm_gpu_ssm_head_permute_kernel<<<grid, n_threads, 0, ctx->stream>>>(
        dst_dev, src_dev, head_map_dev, head_dim, n_heads);
    if (!gpu_ok(gpuGetLastError(), "ssm head permute (dev)")) return 0;
    return 1;
}

/* ---- SSM gated normalization ----
 * Direct port of the CPU reference ("18. Gated normalization" in
 * ssm_forward): per-head RMSNorm of the SSM output, scaled by a
 * learned per-dim weight (shared across heads) and gated by
 * silu(xb2) (the z-gate computed earlier in the layer).
 *
 * Two different layouts meet here, exactly as in the CPU code -- get
 * this wrong and every hybrid layer's output is silently corrupted:
 *   ssm_output: dim-major  [head_v_dim][n_v_heads], index d*n_v_heads+h
 *   xb2 (gate): head-major [n_v_heads][head_v_dim], index h*head_v_dim+d
 *   output:     head-major [n_v_heads][head_v_dim], index h*head_v_dim+d
 *     (or, if head_map is non-NULL, written at head_map[h] instead of h
 *      -- this fuses the GGUF v-head remap into the output write,
 *      avoiding the CPU reference's separate fo_gguf permute-copy pass)
 *
 * Grid = n_v_heads blocks; each block does one head's RMS reduction
 * then writes its head_v_dim outputs, both parallelized across all
 * threads (no per-thread-array/serial-thread-0 pattern -- learned that
 * lesson three times already this session). */
__global__ void
picolm_gpu_ssm_gated_norm_kernel(float *final_output,
                                  const float *ssm_output,
                                  const float *xb2,
                                  const float *norm_w,
                                  const int *head_map,
                                  int head_v_dim, int n_v_heads, float eps) {
    int h = (int)gpuBlockIdx_x;
    if (h >= n_v_heads) return;
    if (gpuThreadIdx_x != 0) return;

    /* Thread-0 sequential: matches the CPU scalar loop's left-to-right
     * sum exactly (a parallel tree reduction sums in a different order
     * and gives a different, non-bit-identical float result). */
    float nrm = 0.0f;
    for (int d = 0; d < head_v_dim; d++) {
        float v = ssm_output[(size_t)d * n_v_heads + h];
        nrm += v * v;
    }
    nrm = 1.0f / sqrtf(nrm / head_v_dim + eps);

    int gh = head_map ? head_map[h] : h;
    float *out_h = final_output + (size_t)gh * head_v_dim;
    const float *xb2_h = xb2 + (size_t)h * head_v_dim;

    for (int d = 0; d < head_v_dim; d++) {
        float v = ssm_output[(size_t)d * n_v_heads + h];
        float zv = xb2_h[d];
        float silu_z = zv / (1.0f + expf(-zv));
        out_h[d] = v * nrm * norm_w[d] * silu_z;
    }
}

/* Batched-over-tokens version: identical per-(token,head) computation.
 * ssm_output is [n_tokens][head_v_dim][n_v_heads] (dim-major per
 * token), xb2 and final_output are [n_tokens][n_v_heads][head_v_dim]
 * (head-major per token, final_output's head slot remapped through
 * head_map exactly as in the per-token kernel). */
__global__ void
picolm_gpu_ssm_gated_norm_batch_kernel(float *final_output,
                                        const float *ssm_output,
                                        const float *xb2,
                                        const float *norm_w,
                                        const int *head_map,
                                        int head_v_dim, int n_v_heads,
                                        int n_tokens, float eps) {
    int h = (int)gpuBlockIdx_x;
    int t = (int)gpuBlockIdx_y;
    if (h >= n_v_heads || t >= n_tokens) return;
    if (gpuThreadIdx_x != 0) return;

    const float *so_t = ssm_output + (size_t)t * head_v_dim * n_v_heads;

    float nrm = 0.0f;
    for (int d = 0; d < head_v_dim; d++) {
        float v = so_t[(size_t)d * n_v_heads + h];
        nrm += v * v;
    }
    nrm = 1.0f / sqrtf(nrm / head_v_dim + eps);

    int gh = head_map ? head_map[h] : h;
    float *out_h = final_output + (size_t)t * n_v_heads * head_v_dim + (size_t)gh * head_v_dim;
    const float *xb2_h = xb2 + (size_t)t * n_v_heads * head_v_dim + (size_t)h * head_v_dim;

    for (int d = 0; d < head_v_dim; d++) {
        float v = so_t[(size_t)d * n_v_heads + h];
        float zv = xb2_h[d];
        float silu_z = zv / (1.0f + expf(-zv));
        out_h[d] = v * nrm * norm_w[d] * silu_z;
    }
}

/* Device-native: all pointers device-resident, no H2D/D2H, no sync.
 * head_map_dev may be NULL (identity, no remap). eps matches the CPU
 * reference's default (typically 1e-6 / 1e-5 -- confirm against
 * s->ssm_norm_w's actual eps at the call site, don't hardcode here). */
extern "C" int
picolm_gpu_ssm_gated_norm_dev(float *final_output_dev,
                               const float *ssm_output_dev,
                               const float *xb2_dev,
                               const float *norm_w_dev,
                               const int *head_map_dev,
                               int head_v_dim, int n_v_heads, float eps,
                               int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (head_v_dim < 1 || n_v_heads < 1 || head_v_dim > 256) return 0;

    int n_threads = min(head_v_dim, 256);
    dim3 grid((unsigned)n_v_heads, 1, 1);
    picolm_gpu_ssm_gated_norm_kernel<<<grid, n_threads, 0, ctx->stream>>>(
        final_output_dev, ssm_output_dev, xb2_dev, norm_w_dev, head_map_dev,
        head_v_dim, n_v_heads, eps);
    if (!gpu_ok(gpuGetLastError(), "ssm gated norm (dev)")) return 0;
    return 1;
}

/* Host wrapper for ssm_conv1d (moved here to be after helper functions) */
extern "C" int
picolm_gpu_ssm_conv1d_dev(float *conv_output_dev, float *conv_state_dev,
                           const float *new_input_dev, const float *conv1d_w_dev,
                           int conv_dim, int d_conv, int device) {
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;
    if (conv_dim < 1 || d_conv < 1) return 0;

    int n_threads = 256;
    int n_blocks = (conv_dim + n_threads - 1) / n_threads;
    picolm_gpu_ssm_conv1d_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        conv_output_dev, conv_state_dev, new_input_dev, conv1d_w_dev, conv_dim, d_conv);
    if (!gpu_ok(gpuGetLastError(), "ssm conv1d (dev)")) return 0;
    return 1;
}

/* Host-facing, batched-over-tokens conv1d: takes CPU pointers, does its
 * own H2D/D2H/sync, all in one round trip for the whole n_tokens batch
 * (not one per token). conv_state_host is read once and overwritten
 * with the final window once -- same host buffer ssm_prefill_layer and
 * the CPU-hybrid ssm_forward() decode path both already use as the
 * single source of truth (this does NOT touch the separate persistent
 * gw->ssm_conv_state_dev[il] device buffer that only the currently-
 * disabled ssm_forward_gpu() reads/writes).
 * Returns 0 (caller should fall back to the CPU path -- safe, nothing
 * has been mutated yet) if d_conv exceeds PICOLM_SSM_CONV_MAX_D_CONV. */
extern "C" int
picolm_gpu_ssm_conv1d_batch(float *conv_output_host,      /* out [n_tokens][conv_dim] */
                             float *conv_state_host,       /* in/out [d_conv-1][conv_dim] */
                             const float *new_input_host,  /* in [n_tokens][conv_dim] */
                             const float *conv1d_w_host,   /* in [conv_dim][d_conv] */
                             int conv_dim, int d_conv, int n_tokens, int device) {
    if (conv_dim < 1 || d_conv < 1 || n_tokens < 1) return 0;
    if (d_conv > PICOLM_SSM_CONV_MAX_D_CONV) return 0;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    size_t state_bytes = (size_t)(d_conv - 1) * conv_dim * sizeof(float);
    size_t input_bytes = (size_t)n_tokens * conv_dim * sizeof(float);
    size_t w_bytes = (size_t)conv_dim * d_conv * sizeof(float);
    size_t out_bytes = input_bytes;

    static void *d_out = NULL; static size_t d_out_cap = 0;
    static void *d_state = NULL; static size_t d_state_cap = 0;
    static void *d_in = NULL; static size_t d_in_cap = 0;
    static void *d_w = NULL; static size_t d_w_cap = 0;

    if (!ssm_batch_scratch_ensure(&d_out, &d_out_cap, out_bytes) ||
        !ssm_batch_scratch_ensure(&d_state, &d_state_cap, state_bytes > 0 ? state_bytes : 1) ||
        !ssm_batch_scratch_ensure(&d_in, &d_in_cap, input_bytes) ||
        !ssm_batch_scratch_ensure(&d_w, &d_w_cap, w_bytes)) return 0;

    int ok = 1;
    if (state_bytes > 0)
        ok = ok && gpu_ok(gpuMemcpy(d_state, conv_state_host, state_bytes, gpuMemcpyHostToDevice), "conv1d batch state h2d");
    ok = ok && gpu_ok(gpuMemcpy(d_in, new_input_host, input_bytes, gpuMemcpyHostToDevice), "conv1d batch in h2d");
    ok = ok && gpu_ok(gpuMemcpy(d_w, conv1d_w_host, w_bytes, gpuMemcpyHostToDevice), "conv1d batch w h2d");
    if (!ok) return 0;

    int n_threads = 256;
    int n_blocks = (conv_dim + n_threads - 1) / n_threads;
    int stride = conv_dim;
    picolm_gpu_ssm_conv1d_batch_kernel<<<n_blocks, n_threads, 0, ctx->stream>>>(
        (float *)d_out, (float *)d_state, (const float *)d_in, (const float *)d_w,
        conv_dim, d_conv, n_tokens, stride);

    if (!gpu_ok(gpuGetLastError(), "ssm conv1d batch") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm conv1d batch sync")) return 0;
    ok = gpu_ok(gpuMemcpy(conv_output_host, d_out, out_bytes, gpuMemcpyDeviceToHost), "conv1d batch out d2h");
    if (ok && state_bytes > 0)
        ok = gpu_ok(gpuMemcpy(conv_state_host, d_state, state_bytes, gpuMemcpyDeviceToHost), "conv1d batch state d2h");
    return ok;
}

/* Host-side gated norm wrapper: takes CPU pointers, does its own H2D/D2H/sync.
 * For use from ssm_forward() when the pipeline is not active. */
extern "C" int
picolm_gpu_ssm_gated_norm(float *final_output,  /* out, host [head_v_dim * n_v_heads] */
                           const float *ssm_output, /* in, host [d_state * n_v_heads] */
                           const float *xb2,    /* in, host [head_v_dim * n_v_heads] */
                           const float *norm_w, /* in, host [head_v_dim] */
                           const int *head_map, /* in, host [n_v_heads] or NULL */
                           int head_v_dim, int n_v_heads, float eps,
                           int device) {
    if (head_v_dim < 1 || n_v_heads < 1) return 0;
    if (!gpu_ok(gpuSetDevice(device), "ssm gn")) return 0;
    gpu_device_ctx_t *ctx = NULL;
    for (int i = 0; i < g_nctx; i++) {
        if (g_gpu_ctx[i].device == device) { ctx = &g_gpu_ctx[i]; break; }
    }
    if (!ctx) return 0;

    size_t out_bytes = (size_t)n_v_heads * head_v_dim * sizeof(float);
    size_t so_bytes = (size_t)n_v_heads * head_v_dim * sizeof(float); /* ssm_output: d_state==head_v_dim */
    size_t xb2_bytes = (size_t)n_v_heads * head_v_dim * sizeof(float);
    size_t nw_bytes = head_v_dim * sizeof(float);
    size_t hm_bytes = n_v_heads * sizeof(int);

    void *d_final, *d_so, *d_xb2, *d_nw, *d_hm;
    if (!gpu_ok(gpuMalloc(&d_final, out_bytes), "ssm gn final") ||
        !gpu_ok(gpuMalloc(&d_so, so_bytes), "ssm gn so") ||
        !gpu_ok(gpuMalloc(&d_xb2, xb2_bytes), "ssm gn xb2") ||
        !gpu_ok(gpuMalloc(&d_nw, nw_bytes), "ssm gn nw")) return 0;
    d_hm = NULL;
    if (head_map) {
        if (!gpu_ok(gpuMalloc(&d_hm, hm_bytes), "ssm gn hmap")) {
            gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw); return 0;
        }
    }
    if (!gpu_ok(gpuMemcpy(d_so, ssm_output, so_bytes, gpuMemcpyHostToDevice), "ssm gn so h2d") ||
        !gpu_ok(gpuMemcpy(d_xb2, xb2, xb2_bytes, gpuMemcpyHostToDevice), "ssm gn xb2 h2d") ||
        !gpu_ok(gpuMemcpy(d_nw, norm_w, nw_bytes, gpuMemcpyHostToDevice), "ssm gn nw h2d")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
        if (d_hm) gpuFree(d_hm); return 0;
    }
    if (d_hm && !gpu_ok(gpuMemcpy(d_hm, head_map, hm_bytes, gpuMemcpyHostToDevice), "ssm gn hm h2d")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw); gpuFree(d_hm); return 0;
    }

    picolm_gpu_ssm_gated_norm_kernel<<<(unsigned)n_v_heads, min(head_v_dim, 256), 0, ctx->stream>>>(
        (float *)d_final, (const float *)d_so, (const float *)d_xb2,
        (const float *)d_nw, (const int *)d_hm, head_v_dim, n_v_heads, eps);

    if (!gpu_ok(gpuGetLastError(), "ssm gated norm") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm gn sync")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
        if (d_hm) gpuFree(d_hm); return 0;
    }
    gpuMemcpy(final_output, d_final, out_bytes, gpuMemcpyDeviceToHost);
    gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
    if (d_hm) gpuFree(d_hm);
    return 1;
}

/* Batched-over-tokens version of picolm_gpu_ssm_gated_norm above: one
 * H2D/D2H round trip for the whole n_tokens batch instead of one per
 * token. Layouts match the per-token version with an added leading
 * token dimension -- see picolm_gpu_ssm_gated_norm_batch_kernel's
 * comment for exact strides. */
extern "C" int
picolm_gpu_ssm_gated_norm_batch(float *final_output,  /* out, host [n_tokens][n_v_heads][head_v_dim] */
                                 const float *ssm_output, /* in, host [n_tokens][head_v_dim][n_v_heads] */
                                 const float *xb2,     /* in, host [n_tokens][n_v_heads][head_v_dim] */
                                 const float *norm_w,  /* in, host [head_v_dim] */
                                 const int *head_map,  /* in, host [n_v_heads] or NULL */
                                 int head_v_dim, int n_v_heads, int n_tokens, float eps,
                                 int device) {
    if (head_v_dim < 1 || n_v_heads < 1 || n_tokens < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    size_t out_bytes = (size_t)n_tokens * n_v_heads * head_v_dim * sizeof(float);
    size_t so_bytes = out_bytes;
    size_t xb2_bytes = out_bytes;
    size_t nw_bytes = (size_t)head_v_dim * sizeof(float);
    size_t hm_bytes = (size_t)n_v_heads * sizeof(int);

    void *d_final, *d_so, *d_xb2, *d_nw, *d_hm = NULL;
    if (!gpu_ok(gpuMalloc(&d_final, out_bytes), "ssm gn batch final") ||
        !gpu_ok(gpuMalloc(&d_so, so_bytes), "ssm gn batch so") ||
        !gpu_ok(gpuMalloc(&d_xb2, xb2_bytes), "ssm gn batch xb2") ||
        !gpu_ok(gpuMalloc(&d_nw, nw_bytes), "ssm gn batch nw")) return 0;
    if (head_map && !gpu_ok(gpuMalloc(&d_hm, hm_bytes), "ssm gn batch hmap")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw); return 0;
    }
    if (!gpu_ok(gpuMemcpy(d_so, ssm_output, so_bytes, gpuMemcpyHostToDevice), "ssm gn batch so h2d") ||
        !gpu_ok(gpuMemcpy(d_xb2, xb2, xb2_bytes, gpuMemcpyHostToDevice), "ssm gn batch xb2 h2d") ||
        !gpu_ok(gpuMemcpy(d_nw, norm_w, nw_bytes, gpuMemcpyHostToDevice), "ssm gn batch nw h2d")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
        if (d_hm) gpuFree(d_hm); return 0;
    }
    if (d_hm && !gpu_ok(gpuMemcpy(d_hm, head_map, hm_bytes, gpuMemcpyHostToDevice), "ssm gn batch hm h2d")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw); gpuFree(d_hm); return 0;
    }

    dim3 grid((unsigned)n_v_heads, (unsigned)n_tokens, 1);
    picolm_gpu_ssm_gated_norm_batch_kernel<<<grid, min(head_v_dim, 256), 0, ctx->stream>>>(
        (float *)d_final, (const float *)d_so, (const float *)d_xb2,
        (const float *)d_nw, (const int *)d_hm, head_v_dim, n_v_heads, n_tokens, eps);

    if (!gpu_ok(gpuGetLastError(), "ssm gated norm batch") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm gn batch sync")) {
        gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
        if (d_hm) gpuFree(d_hm); return 0;
    }
    int ok = gpu_ok(gpuMemcpy(final_output, d_final, out_bytes, gpuMemcpyDeviceToHost), "ssm gn batch out d2h");
    gpuFree(d_final); gpuFree(d_so); gpuFree(d_xb2); gpuFree(d_nw);
    if (d_hm) gpuFree(d_hm);
    return ok;
}

/* ---- SSM prefill gated normalization ----
 * Direct port of ssm_prefill_layer's "8. Gated normalization" CPU loop
 * (model.c). Deliberately NOT shared with picolm_gpu_ssm_gated_norm(_dev)
 * above: those are direct ports of ssm_forward()'s per-token step 18,
 * which takes dim-major ssm_output and fuses the GGUF head_map remap
 * into the output write. Prefill's chunked recurrence
 * (ssm_chunked_recurrence) already writes its output head-major
 * (matching xb2/z's layout) and applies no remap at this stage -- the
 * remap only happens later, in ssm_prefill_layer's do_remap output-
 * projection branch. Forcing this through the decode-path kernel would
 * mean transposing dim-major<->head-major for no reason and getting the
 * remap timing wrong; a second, simpler kernel matching this layout
 * exactly is safer than reusing one built for a different layout.
 *
 * In-place on ssm_out. Same thread-0-per-(token,head) sequential
 * accumulation as every other norm kernel in this file, to bit-match
 * the CPU scalar reference's left-to-right sum. */
__global__ void
picolm_gpu_ssm_prefill_gated_norm_kernel(float *ssm_out,   /* in/out [n_tokens][n_v_heads][head_v_dim] */
                                          const float *z,   /* in [n_tokens][n_v_heads][head_v_dim] */
                                          const float *norm_w, /* in [head_v_dim] */
                                          int head_v_dim, int n_v_heads,
                                          int n_tokens, float eps,
                                          int so_stride, int z_stride) {
    int h = (int)gpuBlockIdx_x;
    int t = (int)gpuBlockIdx_y;
    if (h >= n_v_heads || t >= n_tokens) return;
    if (gpuThreadIdx_x != 0) return;

    float *out_h = ssm_out + (size_t)t * so_stride + (size_t)h * head_v_dim;
    const float *z_h = z + (size_t)t * z_stride + (size_t)h * head_v_dim;

    float nrm = 0.0f;
    for (int d = 0; d < head_v_dim; d++) {
        float v = out_h[d];
        nrm += v * v;
    }
    nrm = 1.0f / sqrtf(nrm / (float)head_v_dim + eps);

    for (int d = 0; d < head_v_dim; d++) {
        float v = out_h[d];
        float zv = z_h[d];
        float silu_z = zv / (1.0f + expf(-zv));
        out_h[d] = v * nrm * norm_w[d] * silu_z;
    }
}

extern "C" int
picolm_gpu_ssm_prefill_gated_norm(float *ssm_out_host,   /* in/out [n_tokens][n_v_heads][head_v_dim] */
                                   const float *z_host,   /* in [n_tokens][n_v_heads][head_v_dim] */
                                   const float *norm_w_host, /* in [head_v_dim] */
                                   int head_v_dim, int n_v_heads, int n_tokens, float eps,
                                   int device) {
    if (head_v_dim < 1 || n_v_heads < 1 || n_tokens < 1) return 0;
    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    size_t so_bytes = (size_t)n_tokens * n_v_heads * head_v_dim * sizeof(float);
    size_t nw_bytes = (size_t)head_v_dim * sizeof(float);

    static void *d_so = NULL; static size_t d_so_cap = 0;
    static void *d_z = NULL; static size_t d_z_cap = 0;
    static void *d_nw = NULL; static size_t d_nw_cap = 0;

    if (!ssm_batch_scratch_ensure(&d_so, &d_so_cap, so_bytes) ||
        !ssm_batch_scratch_ensure(&d_z, &d_z_cap, so_bytes) ||
        !ssm_batch_scratch_ensure(&d_nw, &d_nw_cap, nw_bytes)) return 0;

    if (!gpu_ok(gpuMemcpy(d_so, ssm_out_host, so_bytes, gpuMemcpyHostToDevice), "prefill gn so h2d") ||
        !gpu_ok(gpuMemcpy(d_z, z_host, so_bytes, gpuMemcpyHostToDevice), "prefill gn z h2d") ||
        !gpu_ok(gpuMemcpy(d_nw, norm_w_host, nw_bytes, gpuMemcpyHostToDevice), "prefill gn nw h2d")) return 0;

    int so_stride = n_v_heads * head_v_dim, z_stride = n_v_heads * head_v_dim;
    dim3 grid((unsigned)n_v_heads, (unsigned)n_tokens, 1);
    picolm_gpu_ssm_prefill_gated_norm_kernel<<<grid, min(head_v_dim, 256), 0, ctx->stream>>>(
        (float *)d_so, (const float *)d_z, (const float *)d_nw, head_v_dim, n_v_heads, n_tokens, eps, so_stride, z_stride);

    if (!gpu_ok(gpuGetLastError(), "ssm prefill gated norm") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm prefill gn sync")) return 0;
    return gpu_ok(gpuMemcpy(ssm_out_host, d_so, so_bytes, gpuMemcpyDeviceToHost), "prefill gn out d2h");
}
extern "C" int
picolm_gpu_ssm_vecdot(float *out_host,
                       const float *x_host,
                       const void *weights_host,
                       gguf_type_t qtype,
                       int dim, int n_v_heads,
                       int row_bytes,
                       const int *head_map,
                       int device) {
    if (n_v_heads <= 0 || dim <= 0) return 0;
    if (qtype != 0 && qtype != 2 && qtype != 8) return 0;
    if (dim > PICOLM_SSM_VECDOT_MAX_DIM) return 0;

    if (!gpu_ok(gpuSetDevice(device), "ssm vecdot device")) return 0;
    gpu_device_ctx_t *ctx = NULL;
    for (int i = 0; i < g_nctx; i++) {
        if (g_gpu_ctx[i].device == device) { ctx = &g_gpu_ctx[i]; break; }
    }
    if (!ctx) return 0;

    size_t x_bytes = (size_t)dim * sizeof(float);
    size_t w_bytes = (size_t)n_v_heads * row_bytes;
    size_t out_bytes = (size_t)n_v_heads * sizeof(float);

    if (!reserve(&ctx->x, &ctx->x_cap, x_bytes) ||
        !reserve(&ctx->y, &ctx->y_cap, out_bytes)) return 0;
    void *w_dev = NULL;
    if (!gpu_ok(gpuMalloc(&w_dev, w_bytes), "ssm vecdot w malloc")) return 0;

    if (!gpu_ok(gpuMemcpy(ctx->x, x_host, x_bytes, gpuMemcpyHostToDevice), "ssm x h2d") ||
        !gpu_ok(gpuMemcpy(w_dev, weights_host, w_bytes, gpuMemcpyHostToDevice), "ssm w h2d")) {
        gpuFree(w_dev);
        return 0;
    }

    dim3 grid((unsigned)n_v_heads, 1, 1);
    void *hm_dev = NULL;
    if (head_map) {
        gpuMalloc(&hm_dev, (size_t)n_v_heads * sizeof(int));
        gpuMemcpy(hm_dev, head_map, (size_t)n_v_heads * sizeof(int), gpuMemcpyHostToDevice);
    }
    picolm_ssm_vecdot_kernel<<<grid, 256, 0, ctx->stream>>>(
        ctx->y, ctx->x, w_dev, qtype, dim, n_v_heads, row_bytes,
        hm_dev ? (const int *)hm_dev : NULL);

    if (!gpu_ok(gpuDeviceSynchronize(), "ssm vecdot sync") ||
        !gpu_ok(gpuMemcpy(out_host, ctx->y, out_bytes, gpuMemcpyDeviceToHost), "ssm out d2h")) {
        gpuFree(w_dev);
        if (hm_dev) gpuFree(hm_dev);
        return 0;
    }
    gpuFree(w_dev);
    if (hm_dev) gpuFree(hm_dev);
    return 1;
}

/* Batched-over-tokens version of picolm_gpu_ssm_vecdot above: uploads
 * the weight matrix and head_map ONCE for the whole n_tokens batch
 * (instead of once per token, which is what calling picolm_gpu_ssm_vecdot
 * n_tokens times in a loop would do), and does a single H2D for all
 * n_tokens' worth of x and a single D2H for all outputs. */
extern "C" int
picolm_gpu_ssm_vecdot_batch(float *out_host,       /* out [n_tokens][n_v_heads] */
                             const float *x_host,   /* in [n_tokens][dim] */
                             const void *weights_host,
                             gguf_type_t qtype,
                             int dim, int n_v_heads, int n_tokens,
                             int row_bytes,
                             const int *head_map,
                             int device) {
    if (n_v_heads <= 0 || dim <= 0 || n_tokens <= 0) return 0;
    if (qtype != 0 && qtype != 2 && qtype != 8) return 0;
    if (dim > PICOLM_SSM_VECDOT_MAX_DIM) return 0;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    size_t x_bytes = (size_t)n_tokens * dim * sizeof(float);
    size_t w_bytes = (size_t)n_v_heads * row_bytes;
    size_t out_bytes = (size_t)n_tokens * n_v_heads * sizeof(float);
    size_t hm_bytes = (size_t)n_v_heads * sizeof(int);

    static void *x_dev = NULL; static size_t x_dev_cap = 0;
    static void *w_dev = NULL; static size_t w_dev_cap = 0;
    static void *out_dev = NULL; static size_t out_dev_cap = 0;
    static void *hm_dev = NULL; static size_t hm_dev_cap = 0;

    if (!ssm_batch_scratch_ensure(&x_dev, &x_dev_cap, x_bytes) ||
        !ssm_batch_scratch_ensure(&w_dev, &w_dev_cap, w_bytes) ||
        !ssm_batch_scratch_ensure(&out_dev, &out_dev_cap, out_bytes)) return 0;
    if (head_map && !ssm_batch_scratch_ensure(&hm_dev, &hm_dev_cap, hm_bytes)) return 0;

    int ok = gpu_ok(gpuMemcpy(x_dev, x_host, x_bytes, gpuMemcpyHostToDevice), "vecdot batch x h2d") &&
             gpu_ok(gpuMemcpy(w_dev, weights_host, w_bytes, gpuMemcpyHostToDevice), "vecdot batch w h2d");
    if (ok && head_map)
        ok = gpu_ok(gpuMemcpy(hm_dev, head_map, hm_bytes, gpuMemcpyHostToDevice), "vecdot batch hmap h2d");
    if (!ok) return 0;

    dim3 grid((unsigned)n_v_heads, (unsigned)n_tokens, 1);
    picolm_ssm_vecdot_batch_kernel<<<grid, 256, 0, ctx->stream>>>(
        (float *)out_dev, (const float *)x_dev, w_dev, qtype, dim, n_v_heads, n_tokens,
        row_bytes, head_map ? (const int *)hm_dev : NULL, dim, 0);

    if (!gpu_ok(gpuGetLastError(), "ssm vecdot batch") ||
        !gpu_ok(gpuDeviceSynchronize(), "ssm vecdot batch sync")) return 0;
    return gpu_ok(gpuMemcpy(out_host, out_dev, out_bytes, gpuMemcpyDeviceToHost), "vecdot batch out d2h");
}

/* Fully device-native SSM vecdot: weights_dev and head_map_dev must
 * already be device-resident, uploaded ONCE at model load (via
 * picolm_gpu_tensor_upload for quantized weights or picolm_gpu_upload_f32
 * for F32 ssm_alpha/ssm_beta, and a one-time int array upload for the
 * head_map) -- NOT re-uploaded every call like picolm_gpu_ssm_vecdot()
 * above does. x_dev/out_dev are pipeline buffers. No malloc, no H2D/D2H,
 * no internal sync -- same ctx->stream ordering argument as every other
 * _dev primitive this session. head_map_dev may be NULL (identity). */
extern "C" int
picolm_gpu_ssm_vecdot_dev(float *out_dev,
                           const float *x_dev,
                           const void *weights_dev,
                           gguf_type_t qtype,
                           int dim, int n_v_heads,
                           int row_bytes,
                           const int *head_map_dev,
                           int device) {
    if (n_v_heads <= 0 || dim <= 0) return 0;
    if (qtype != 0 && qtype != 2 && qtype != 8 && qtype != 30) return 0;
    if (dim > PICOLM_SSM_VECDOT_MAX_DIM) return 0;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    dim3 grid((unsigned)n_v_heads, 1, 1);
    picolm_ssm_vecdot_kernel<<<grid, 256, 0, ctx->stream>>>(
        out_dev, x_dev, weights_dev, qtype, dim, n_v_heads, row_bytes, head_map_dev);
    if (!gpu_ok(gpuGetLastError(), "ssm vecdot (dev)")) return 0;
    return 1;
}

/* ================================================================
 * Chunked DeltaNet SSM recurrence -- GPU port (prefill).
 *
 * VALIDATED. Gated behind PICOLM_SSM_CHUNKED_GPU_VALIDATED (defined
 * by default for CUDA builds in the Makefile; undefined for HIP builds,
 * where this returns 0 and the caller falls back to the CPU path).
 *
 * Direct, deliberately unoptimized (one-thread-per-output-element, no
 * tiling/shared-memory GEMM) port of the scalar reference path in
 * ssm_chunk_head_task() / ssm_chunked_recurrence() (model.c) -- ported
 * from the #else scalar branch there, not the AVX/NEON micro-kernel
 * branches (multiple SIMD variants exist; the scalar branch is the one
 * unambiguous reference all of them are optimizing). See
 * chunked_ssm_gpu_design.md in the project notes for:
 *   - why bit-exactness with the CPU path is NOT the goal (unlike the
 *     single-token recurrence kernel, these are GEMM-shaped reductions
 *     over d_state~128 elements -- forcing a specific summation order
 *     there would mean giving up tiling, i.e. giving up most of the
 *     performance point)
 *   - the recommended validation plan (per-chunk numerical comparison
 *     against ssm_chunk_head_task before trusting end-to-end output)
 *   - why every kernel below batches per-v-head as one threadblock,
 *     matching the CPU's tensor_parallel_for(n_v_heads, ...) head
 *     parallelization, with chunks processed sequentially on the host
 *     side (state carries chunk-to-chunk within a head)
 *
 * Buffer layout (matches ssm_chunk_head_task's scratch pool exactly,
 * so this can be checked line-by-line against it):
 *   chunk_q, chunk_k:  [n_k_heads][cs][d]   (gathered once per chunk)
 *   chunk_v:           [n_v_heads][cs][d]
 *   chunk_beta, gate_log, cum_g, q_decay:  [n_v_heads][cs]
 *   decay_mask, M_mat, kq:                 [n_v_heads][cs][cs]
 *   v_eff, v_hat, sk, sq, chunk_out:        [n_v_heads][cs][d]
 *   state:                                  [n_v_heads][d][d]
 *
 * `d` below is shared between K/Q (d_state) and V/out/state
 * (head_v_dim) -- this whole algorithm relies on d_state == head_v_dim
 * (state is square, V/out are d_state-wide per head), which
 * ssm_chunk_head_task() on the CPU side already relies on too (it
 * indexes chunk_v, gathered at head_v_dim stride, using ctx->d_state).
 * The host driver checks this invariant explicitly and refuses to
 * dispatch (safe fallback to CPU) if it doesn't hold, rather than
 * assume it silently.
 * ================================================================ */

/* ---- Gather: gate_log/beta/Q/K/V for one chunk, from the token-major
 * conv_batch/alpha_batch/beta_batch buffers, into the head-major
 * layout the rest of this pipeline uses. Direct port of the CPU gather
 * loops in ssm_chunked_recurrence(). Two kernels (Q/K vs V+scalars)
 * since they're indexed by different head counts (n_k_heads vs
 * n_v_heads). ---- */
__global__ void
ssm_chunk_gather_qk_kernel(float *chunk_q, float *chunk_k,
                            const float *conv_batch,
                            int chunk_start, int cs_actual, int conv_stride,
                            int qk_dim, int d_state, int n_k_heads) {
    int h = blockIdx.x;
    if (h >= n_k_heads) return;
    float *cq_h = chunk_q + (size_t)h * cs_actual * d_state;
    float *ck_h = chunk_k + (size_t)h * cs_actual * d_state;
    int n_elem = cs_actual * d_state;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int t = idx / d_state, di = idx % d_state;
        const float *tok = conv_batch + (size_t)(chunk_start + t) * conv_stride;
        cq_h[idx] = tok[h * d_state + di];
        ck_h[idx] = tok[qk_dim + h * d_state + di];
    }
}

__global__ void
ssm_chunk_gather_v_kernel(float *chunk_v, float *chunk_beta, float *gate_log,
                           const float *conv_batch, const float *alpha_batch,
                           const float *beta_batch,
                           int chunk_start, int cs_actual, int conv_stride,
                           int qk_dim, int head_v_dim, int n_v_heads,
                           int ab_stride) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    float *cv_h = chunk_v + (size_t)h * cs_actual * head_v_dim;
    float *cb_h = chunk_beta + (size_t)h * cs_actual;
    float *gl_h = gate_log + (size_t)h * cs_actual;

    int n_elem = cs_actual * head_v_dim;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int t = idx / head_v_dim, di = idx % head_v_dim;
        const float *tok = conv_batch + (size_t)(chunk_start + t) * conv_stride;
        cv_h[idx] = tok[2 * qk_dim + h * head_v_dim + di];
    }
    int as = ab_stride > 0 ? ab_stride : n_v_heads;
    for (int t = threadIdx.x; t < cs_actual; t += blockDim.x) {
        cb_h[t] = beta_batch[(size_t)(chunk_start + t) * as + h];
        /* alpha_batch now contains gate_log directly (log-space), not expf(gate).
         * No logf needed -- direct copy, matching the CPU path. */
        gl_h[t] = alpha_batch[(size_t)(chunk_start + t) * as + h];
    }
}

/* ---- Step 1: cumulative log-decay + decay mask. Direct port of the
 * CPU's "Step 1" + "Build decay mask" blocks. Thread 0 does the
 * inherently-sequential prefix sum (matches CPU's ascending-t loop
 * exactly, same clamping, same expf calls); all threads then fill the
 * cs x cs mask in parallel once cum_g is visible via __syncthreads(). ---- */
__global__ void
ssm_chunk_decay_kernel(float *cum_g, float *q_decay, float *decay_mask,
                        const float *gate_log, int n_v_heads, int cs, int gl_stride) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    const float *gl = gate_log + (size_t)h * cs;
    (void)gl_stride; /* scratch buffer is contiguous head-major */
    float *cg = cum_g + (size_t)h * cs;
    float *qd = q_decay + (size_t)h * cs;
    float *dm = decay_mask + (size_t)h * cs * cs;

    if (threadIdx.x == 0) {
        float cum = 0.0f;
        for (int t = 0; t < cs; t++) {
            cum += gl[t];
            cg[t] = cum;
            float ex = cum;
            if (ex > 50.0f) ex = 50.0f;
            if (ex < -50.0f) ex = -50.0f;
            qd[t] = expf(ex);
        }
    }
    __syncthreads();

    int n_elem = cs * cs;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int i = idx / cs, j = idx % cs;
        float v;
        if (j > i) v = 0.0f;
        else if (i == j) v = 1.0f;
        else {
            float diff = cg[i] - cg[j];
            if (diff > 50.0f) diff = 50.0f;
            if (diff < -50.0f) diff = -50.0f;
            v = expf(diff);
        }
        dm[idx] = v;
    }
}

/* ---- Masked cs x cs GEMM: out[h][i][j] = dot(A[kh][i], B[kh][j]) *
 * decay_mask[h][i][j] for j<=i, else 0. Reused for both M (A=B=chunk_k,
 * "Step 2" in the CPU scalar path) and kq (A=chunk_q, B=chunk_k,
 * part of CPU "Step 5"). One threadblock per v-head, threads stride
 * over the cs*cs output elements; each does a length-d dot product --
 * no tiling, deliberately simple for a first correctness pass (see
 * design doc). ---- */
__global__ void
ssm_chunk_masked_gemm_kernel(float *out_cs_cs, const float *A, const float *B,
                              const float *decay_mask,
                              int n_v_heads, int repeat, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    int kh = h / repeat;
    const float *Ah = A + (size_t)kh * cs * d;
    const float *Bh = B + (size_t)kh * cs * d;
    const float *dm = decay_mask + (size_t)h * cs * cs;
    float *out = out_cs_cs + (size_t)h * cs * cs;

    int n_elem = cs * cs;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int i = idx / cs, j = idx % cs;
        if (j > i) { out[idx] = 0.0f; continue; }
        const float *ai = Ah + (size_t)i * d;
        const float *bj = Bh + (size_t)j * d;
        float dot = 0.0f;
        for (int c = 0; c < d; c++) dot += ai[c] * bj[c];
        out[idx] = dot * dm[idx];
    }
}

/* ---- cs x d matvec-GEMM: out[h][i][r] = sum_c state[h][r][c] * X[kh][i][c],
 * i.e. out = X @ State^T. Reused for sk (X=chunk_k) and sq (X=chunk_q)
 * -- CPU "Kernel 1"/"Kernel 1b" in the SIMD path, inlined into "Step 3"
 * /"Step 5" in the scalar path. ---- */
__global__ void
ssm_chunk_matvec_kernel(float *out, const float *state, const float *chunk_x,
                         int n_v_heads, int repeat, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    int kh = h / repeat;
    const float *x_h = chunk_x + (size_t)kh * cs * d;
    const float *st = state + (size_t)h * d * d;
    float *o = out + (size_t)h * cs * d;

    int n_elem = cs * d;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int i = idx / d, r = idx % d;
        const float *xi = x_h + (size_t)i * d;
        const float *st_row = st + (size_t)r * d;
        float sum = 0.0f;
        for (int c = 0; c < d; c++) sum += st_row[c] * xi[c];
        o[idx] = sum;
    }
}

/* ---- V_eff assembly: direct port of CPU "Step 3". ---- */
__global__ void
ssm_chunk_veff_kernel(float *v_eff, const float *chunk_v, const float *sk,
                       const float *q_decay, const float *chunk_beta,
                       int n_v_heads, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    const float *v_h = chunk_v + (size_t)h * cs * d;
    const float *sk_h = sk + (size_t)h * cs * d;
    const float *qd_h = q_decay + (size_t)h * cs;
    const float *bt_h = chunk_beta + (size_t)h * cs;
    float *ve_h = v_eff + (size_t)h * cs * d;

    int n_elem = cs * d;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int i = idx / d;
        ve_h[idx] = bt_h[i] * (v_h[idx] - qd_h[i] * sk_h[idx]);
    }
}

/* ---- Forward substitution / triangular solve for V_hat: the one
 * inherently sequential part of this whole algorithm. Direct port of
 * CPU "Step 4". One threadblock per v-head; blockDim.x should cover d
 * (loops if not). cs sequential steps over i, ascending, with
 * __syncthreads() between each: v_hat[i] must be fully written by all
 * r-threads before any thread reads it for i+1 (every later i' > i
 * reads all of v_hat[0..i'-1]). This is the piece flagged in the
 * design doc as needing the most dedicated validation (small-cs and
 * boundary-chunk cases especially). ---- */
__global__ void
ssm_chunk_trisolve_kernel(float *v_hat, const float *v_eff, const float *M_mat,
                           const float *chunk_beta,
                           int n_v_heads, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    const float *ve_h = v_eff + (size_t)h * cs * d;
    const float *M_h = M_mat + (size_t)h * cs * cs;
    const float *bt_h = chunk_beta + (size_t)h * cs;
    float *vh_h = v_hat + (size_t)h * cs * d;

    for (int i = 0; i < cs; i++) {
        float bt = bt_h[i];
        const float *Mi = M_h + (size_t)i * cs;
        for (int r = threadIdx.x; r < d; r += blockDim.x) {
            float sum_mv = 0.0f;
            for (int j = 0; j < i; j++) sum_mv += Mi[j] * vh_h[(size_t)j * d + r];
            vh_h[(size_t)i * d + r] = ve_h[(size_t)i * d + r] - bt * sum_mv;
        }
        __syncthreads();
    }
}

/* ---- Output assembly: direct port of CPU "Step 5". kq already has
 * the decay_mask factor folded in (from ssm_chunk_masked_gemm_kernel),
 * matching kq[i][j] == CPU's `attn` exactly -- no separate mask lookup
 * needed here, same as the CPU scalar path's `attn = k_dot_q * dm`
 * being precomputed rather than looked up in the AVX/NEON paths. ---- */
__global__ void
ssm_chunk_output_kernel(float *chunk_out, const float *sq, const float *q_decay,
                         const float *kq, const float *v_hat,
                         int n_v_heads, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    const float *sq_h = sq + (size_t)h * cs * d;
    const float *qd_h = q_decay + (size_t)h * cs;
    const float *kq_h = kq + (size_t)h * cs * cs;
    const float *vh_h = v_hat + (size_t)h * cs * d;
    float *out_h = chunk_out + (size_t)h * cs * d;

    int n_elem = cs * d;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int i = idx / d, r = idx % d;
        float acc = sq_h[idx] * qd_h[i];
        const float *kqi = kq_h + (size_t)i * cs;
        for (int j = 0; j <= i; j++)
            acc += kqi[j] * vh_h[(size_t)j * d + r];
        out_h[idx] = acc;
    }
}

/* ---- State update: direct port of CPU "Step 6" (scalar path),
 * including its exact redundant per-(r,c,j) recomputation of
 * decay_to_end rather than precomputing it once per j -- kept for
 * fidelity to the reference rather than "optimized" into a precompute
 * pass, since this is a correctness-first port. ---- */
__global__ void
ssm_chunk_state_update_kernel(float *state, const float *v_hat, const float *chunk_k,
                               const float *cum_g,
                               int n_v_heads, int repeat, int cs, int d) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    int kh = h / repeat;
    const float *vh_h = v_hat + (size_t)h * cs * d;
    const float *k_h = chunk_k + (size_t)kh * cs * d;
    const float *cg_h = cum_g + (size_t)h * cs;
    float *st_h = state + (size_t)h * d * d;

    float cum_last = cg_h[cs - 1];
    float ex_total = cum_last;
    if (ex_total > 50.0f) ex_total = 50.0f;
    if (ex_total < -50.0f) ex_total = -50.0f;
    float total_decay = expf(ex_total);

    int n_elem = d * d;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int r = idx / d, c = idx % d;
        float update = 0.0f;
        for (int j = 0; j < cs; j++) {
            float diff = cum_last - cg_h[j];
            if (diff > 50.0f) diff = 50.0f;
            if (diff < -50.0f) diff = -50.0f;
            float decay_to_end = expf(diff);
            update += vh_h[(size_t)j * d + r] * k_h[(size_t)j * d + c] * decay_to_end;
        }
        st_h[idx] = st_h[idx] * total_decay + update;
    }
}

/* ---- Scatter: chunk_out [n_v_heads][cs][d] -> xb2_batch
 * [n_tokens][value_dim] head-major, matching ssm_chunked_recurrence()'s
 * CPU reshape loop exactly. ---- */
__global__ void
ssm_chunk_scatter_kernel(float *xb2_batch, const float *chunk_out,
                          int chunk_start, int cs_actual, int xb2_stride,
                          int head_v_dim, int n_v_heads) {
    int h = blockIdx.x;
    if (h >= n_v_heads) return;
    const float *co_h = chunk_out + (size_t)h * cs_actual * head_v_dim;
    int n_elem = cs_actual * head_v_dim;
    for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x) {
        int t = idx / head_v_dim, r = idx % head_v_dim;
        xb2_batch[(size_t)(chunk_start + t) * xb2_stride + h * head_v_dim + r] = co_h[idx];
    }
}

/* ---- Host driver ----
 * Mirrors ssm_chunked_recurrence()'s structure: allocate scratch sized
 * for the max chunk size once, then loop over chunks sequentially
 * (state carries forward within a head across chunks -- chunks
 * themselves cannot be parallelized). Everything (conv_batch, alpha,
 * beta, state, xb2_batch) is uploaded/downloaded ONCE for the whole
 * call, not once per chunk: state and xb2_batch stay device-resident
 * across the entire chunk loop, only leaving the GPU once at the end
 * on success.
 *
 * Safe to fall back to the CPU path on ANY failure, at ANY point in
 * the chunk loop: state_host and xb2_batch_host are only overwritten
 * by the final D2H copies after every chunk has succeeded and the
 * whole thing has been synced -- unlike the decode-path _dev
 * functions, there's no in-place mutation of the sole copy of truth
 * here (same reasoning as picolm_gpu_ssm_conv1d_batch above). On
 * failure, the device-side partial work is simply discarded and
 * state_host/xb2_batch_host are exactly as they were on entry, so the
 * caller's ssm_chunked_recurrence() CPU fallback is always correct.
 *
 * Returns 0 unconditionally unless PICOLM_SSM_CHUNKED_GPU_VALIDATED is
 * defined by the build -- see the section-header comment above. */
extern "C" int
picolm_gpu_ssm_chunked_recurrence(const float *conv_batch_host,
                                   const float *alpha_batch_host,
                                   const float *beta_batch_host,
                                   float *state_host,
                                   float *xb2_batch_host,
                                   int n_tokens, int value_dim,
                                   int d_state, int n_k_heads, int n_v_heads,
                                   int head_v_dim, int repeat,
                                   int conv_dim, int cs, int device) {
#ifndef PICOLM_SSM_CHUNKED_GPU_VALIDATED
    (void)conv_batch_host; (void)alpha_batch_host; (void)beta_batch_host;
    (void)state_host; (void)xb2_batch_host; (void)n_tokens; (void)value_dim;
    (void)d_state; (void)n_k_heads; (void)n_v_heads; (void)head_v_dim;
    (void)repeat; (void)conv_dim; (void)cs; (void)device;
    return 0;
#else
    if (d_state != head_v_dim) return 0; /* architectural invariant this whole port relies on */
    if (cs <= 0) cs = 64;
    if (cs > n_tokens) cs = n_tokens;
    int n_chunks = (n_tokens + cs - 1) / cs;
    if (n_chunks < 1) n_chunks = 1;
    int qk_dim = d_state * n_k_heads;
    int d = d_state;

    gpu_device_ctx_t *ctx = find_ctx(device);
    if (!ctx || !select_ctx(ctx)) return 0;

    /* Per-call, uploaded/downloaded once */
    size_t conv_bytes = (size_t)n_tokens * conv_dim * sizeof(float);
    size_t alpha_bytes = (size_t)n_tokens * n_v_heads * sizeof(float);
    size_t state_bytes = (size_t)n_v_heads * d * d * sizeof(float);
    size_t xb2_bytes = (size_t)n_tokens * value_dim * sizeof(float);

    /* Per-chunk scratch, sized for the max chunk size, reused every chunk */
    size_t qk_sc_bytes = (size_t)n_k_heads * cs * d * sizeof(float);
    size_t v_sc_bytes = (size_t)n_v_heads * cs * d * sizeof(float);
    size_t scalar_sc_bytes = (size_t)n_v_heads * cs * sizeof(float);
    size_t sq_sc_bytes = (size_t)n_v_heads * cs * cs * sizeof(float);

    /* All buffers persistent/grow-only across calls (see
     * ssm_batch_scratch_ensure's comment): this function alone was
     * ~20 gpuMalloc/gpuFree pairs EVERY call, and it's called once per
     * layer -- for a 48-layer model that's ~1000 alloc/free pairs per
     * prompt just from this one function. Grown to the largest chunk
     * size seen so far and never freed until the device changes. */
    static void *d_conv = NULL; static size_t d_conv_cap = 0;
    static void *d_alpha = NULL; static size_t d_alpha_cap = 0;
    static void *d_beta = NULL; static size_t d_beta_cap = 0;
    static void *d_state_buf = NULL; static size_t d_state_buf_cap = 0;
    static void *d_xb2 = NULL; static size_t d_xb2_cap = 0;
    static void *d_chunk_q = NULL; static size_t d_chunk_q_cap = 0;
    static void *d_chunk_k = NULL; static size_t d_chunk_k_cap = 0;
    static void *d_chunk_v = NULL; static size_t d_chunk_v_cap = 0;
    static void *d_chunk_beta = NULL; static size_t d_chunk_beta_cap = 0;
    static void *d_gate_log = NULL; static size_t d_gate_log_cap = 0;
    static void *d_cum_g = NULL; static size_t d_cum_g_cap = 0;
    static void *d_q_decay = NULL; static size_t d_q_decay_cap = 0;
    static void *d_decay_mask = NULL; static size_t d_decay_mask_cap = 0;
    static void *d_M = NULL; static size_t d_M_cap = 0;
    static void *d_kq = NULL; static size_t d_kq_cap = 0;
    static void *d_v_eff = NULL; static size_t d_v_eff_cap = 0;
    static void *d_v_hat = NULL; static size_t d_v_hat_cap = 0;
    static void *d_sk = NULL; static size_t d_sk_cap = 0;
    static void *d_sq = NULL; static size_t d_sq_cap = 0;
    static void *d_chunk_out = NULL; static size_t d_chunk_out_cap = 0;

    int ok = 1;
    ok = ok && ssm_batch_scratch_ensure(&d_conv, &d_conv_cap, conv_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_alpha, &d_alpha_cap, alpha_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_beta, &d_beta_cap, alpha_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_state_buf, &d_state_buf_cap, state_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_xb2, &d_xb2_cap, xb2_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_chunk_q, &d_chunk_q_cap, qk_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_chunk_k, &d_chunk_k_cap, qk_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_chunk_v, &d_chunk_v_cap, v_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_chunk_beta, &d_chunk_beta_cap, scalar_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_gate_log, &d_gate_log_cap, scalar_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_cum_g, &d_cum_g_cap, scalar_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_q_decay, &d_q_decay_cap, scalar_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_decay_mask, &d_decay_mask_cap, sq_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_M, &d_M_cap, sq_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_kq, &d_kq_cap, sq_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_v_eff, &d_v_eff_cap, v_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_v_hat, &d_v_hat_cap, v_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_sk, &d_sk_cap, v_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_sq, &d_sq_cap, v_sc_bytes);
    ok = ok && ssm_batch_scratch_ensure(&d_chunk_out, &d_chunk_out_cap, v_sc_bytes);

    if (ok) {
        ok = ok && gpu_ok(gpuMemcpy(d_conv, conv_batch_host, conv_bytes, gpuMemcpyHostToDevice), "chunk conv h2d");
        ok = ok && gpu_ok(gpuMemcpy(d_alpha, alpha_batch_host, alpha_bytes, gpuMemcpyHostToDevice), "chunk alpha h2d");
        ok = ok && gpu_ok(gpuMemcpy(d_beta, beta_batch_host, alpha_bytes, gpuMemcpyHostToDevice), "chunk beta h2d");
        ok = ok && gpu_ok(gpuMemcpy(d_state_buf, state_host, state_bytes, gpuMemcpyHostToDevice), "chunk state h2d");
    }

    if (ok) {
        int n_threads = 256;
        for (int ci = 0; ci < n_chunks && ok; ci++) {
            int cs_actual = (ci == n_chunks - 1) ? (n_tokens - ci * cs) : cs;
            if (cs_actual <= 0) break;
            int chunk_start = ci * cs;

            ssm_chunk_gather_qk_kernel<<<n_k_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_chunk_q, (float *)d_chunk_k, (const float *)d_conv,
                chunk_start, cs_actual, conv_dim, qk_dim, d_state, n_k_heads);
            ssm_chunk_gather_v_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_chunk_v, (float *)d_chunk_beta, (float *)d_gate_log,
                (const float *)d_conv, (const float *)d_alpha, (const float *)d_beta,
                chunk_start, cs_actual, conv_dim, qk_dim, head_v_dim, n_v_heads, 0);

            ssm_chunk_decay_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_cum_g, (float *)d_q_decay, (float *)d_decay_mask,
                (const float *)d_gate_log, n_v_heads, cs_actual, cs_actual);

            ssm_chunk_masked_gemm_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_M, (const float *)d_chunk_k, (const float *)d_chunk_k,
                (const float *)d_decay_mask, n_v_heads, repeat, cs_actual, d);

            ssm_chunk_matvec_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_sk, (const float *)d_state_buf, (const float *)d_chunk_k,
                n_v_heads, repeat, cs_actual, d);

            ssm_chunk_veff_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_v_eff, (const float *)d_chunk_v, (const float *)d_sk,
                (const float *)d_q_decay, (const float *)d_chunk_beta, n_v_heads, cs_actual, d);

            {
                int tri_threads = d < 1024 ? d : 1024;
                ssm_chunk_trisolve_kernel<<<n_v_heads, tri_threads, 0, ctx->stream>>>(
                    (float *)d_v_hat, (const float *)d_v_eff, (const float *)d_M,
                    (const float *)d_chunk_beta, n_v_heads, cs_actual, d);
            }

            ssm_chunk_matvec_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_sq, (const float *)d_state_buf, (const float *)d_chunk_q,
                n_v_heads, repeat, cs_actual, d);

            ssm_chunk_masked_gemm_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_kq, (const float *)d_chunk_q, (const float *)d_chunk_k,
                (const float *)d_decay_mask, n_v_heads, repeat, cs_actual, d);

            ssm_chunk_output_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_chunk_out, (const float *)d_sq, (const float *)d_q_decay,
                (const float *)d_kq, (const float *)d_v_hat, n_v_heads, cs_actual, d);

            ssm_chunk_scatter_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_xb2, (const float *)d_chunk_out,
                chunk_start, cs_actual, value_dim, head_v_dim, n_v_heads);

            /* State update must come after gather/matvec/output above
             * have all read the PRE-update state for this chunk --
             * stream ordering guarantees that since everything above
             * is submitted first on the same stream. */
            ssm_chunk_state_update_kernel<<<n_v_heads, n_threads, 0, ctx->stream>>>(
                (float *)d_state_buf, (const float *)d_v_hat, (const float *)d_chunk_k,
                (const float *)d_cum_g, n_v_heads, repeat, cs_actual, d);

            ok = gpu_ok(gpuGetLastError(), "ssm chunked recurrence chunk");
        }
    }

    ok = ok && gpu_ok(gpuDeviceSynchronize(), "ssm chunked recurrence sync");
    if (ok) {
        ok = gpu_ok(gpuMemcpy(xb2_batch_host, d_xb2, xb2_bytes, gpuMemcpyDeviceToHost), "chunk xb2 d2h");
        ok = ok && gpu_ok(gpuMemcpy(state_host, d_state_buf, state_bytes, gpuMemcpyDeviceToHost), "chunk state d2h");
    }
    return ok;
#endif
}

/* Per-device KV cache pointers */


static uint16_t *g_kv_k_dev[PICOLM_GPU_MAX_DEVICES];
static uint16_t *g_kv_v_dev[PICOLM_GPU_MAX_DEVICES];
static size_t g_kv_k_cap[PICOLM_GPU_MAX_DEVICES];
static size_t g_kv_v_cap[PICOLM_GPU_MAX_DEVICES];
static float *g_verify_buf_dev[PICOLM_GPU_MAX_DEVICES];  /* device verification buffer */

