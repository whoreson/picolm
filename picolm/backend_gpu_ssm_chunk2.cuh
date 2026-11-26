#ifndef BACKEND_GPU_SSM_CHUNK2_CUH
#define BACKEND_GPU_SSM_CHUNK2_CUH
/* ---------------------------------------------------------------------------
 * Chunked gated delta rule (Qwen3.5/3.6 SSM layers), prefill.
 *
 * The existing pipeline (ssm_chunk_*_kernel, driven by
 * picolm_gpu_ssm_chunked_recurrence_dev) runs, for EVERY chunk of 64 tokens, twelve
 * small kernels (gather x2, decay, masked gemm x2, matvec x2, veff, trisolve, output,
 * scatter, state update) with grids of 16-48 blocks on a 128-SM GPU, serialised through
 * the carried state: ~3,800 launches per 512-token ubatch over 48 layers, plus 15
 * memsets of scratch per layer. Its profile showed ~100 ms of 275 ms "unattributed",
 * which is launch gaps and tail latency, not arithmetic (the whole layer is ~6 MFLOP
 * per token; the matmuls are ~1000x that).
 *
 * The algebra lets almost all of it run in parallel. Per v-head, with S[r][c] the
 * state (r = value dim, c = key dim), cg the cumulative log-gate, qd = exp(cg),
 * dm[i][j] = exp(cg_i - cg_j) (j <= i), B = diag(beta), M = (K K^T) .* dm and
 * T = (I + B M_strict)^-1, the existing kernels compute
 *
 *     vh  = T B (V - qd .* (K S^T))        (veff + trisolve, needs S)
 *     out = qd .* (Q S^T) + kq vh          (kq = (Q K^T) .* dm)
 *     S  <- decay*S + (vh .* dte)^T K      (state update)
 *
 * and vh is LINEAR in S:   vh = u - Wk S^T   with   u = T B V,  Wk = T B diag(qd) K,
 * neither of which depends on S. So:
 *
 *   Phase A  ssm2_chunk_pre_kernel   grid (n_v_heads, chunks)   fully parallel
 *            cum-gate, decay mask, K K^T, Q K^T, and u / Wk by forward substitution.
 *            The substitution is parallel across the 256 right-hand-side columns
 *            (u: 128, Wk: 128): every column has its own registers, there is no
 *            barrier per step (the old kernel did 64 __syncthreads per chunk).
 *   Phase B  ssm2_chunk_scan_kernel  grid (n_v_heads, D/JT)     sequential over chunks
 *            Each ROW of S evolves independently, so a block owns JT=16 rows of S
 *            (held in shared memory) and walks the chunks in order:
 *            vh = u - Wk S^T ; out = qd (Q S^T) + kq vh ; S = decay S + (vh dte)^T K.
 *
 * Same clamps (+-50 on every exponent), same conventions as the old kernels, fp32
 * with explicit FMAs throughout. NOT bit-identical to the old pipeline (different
 * summation order, and no CPU AVX-512 reduction tree to imitate); validated against
 * a double-precision token-sequential reference (tests/ssm_chunk2_emu.cpp).
 *
 * The same source is executed on the CPU by tests/ssm_chunk2_emu.cpp
 * (-DPICOLM_FA3_EMU) so the real kernel logic is verified without a GPU.
 * ------------------------------------------------------------------------- */

#define SSM2_CS 64      /* tokens per chunk */
#define SSM2_D  128     /* d_state == head_v_dim (the only size supported) */
#define SSM2_JT 16      /* rows of S per scan block */
#define SSM2_THREADS 256

#ifdef PICOLM_FA3_EMU
#  define SSM2_SMEM_F32(name) float *name = (float *)g_ctx->smem
#else
#  define SSM2_SMEM_F32(name) extern __shared__ __align__(16) float name##_raw[]; float *name = name##_raw
#endif

/* shared-memory sizes in floats (host and device agree through these) */
#define SSM2_PRE_M_STRIDE   65
#define SSM2_PRE_T_STRIDE   68
#define SSM2_PRE_FLOATS     (SSM2_CS * SSM2_PRE_M_STRIDE + 2 * 32 * SSM2_PRE_T_STRIDE + 3 * SSM2_CS)
#define SSM2_SCAN_ST        129          /* row stride of the S slice and of the 64 x 128 tiles */
#define SSM2_SCAN_FLOATS    (SSM2_JT * SSM2_SCAN_ST + SSM2_CS * SSM2_SCAN_ST + SSM2_CS * 65 + SSM2_CS * 17 + 2 * SSM2_CS)

static inline __device__ float ssm2_expc(float x) {       /* exp with the pipeline's +-50 clamp */
    if (x > 50.0f) x = 50.0f;
    if (x < -50.0f) x = -50.0f;
    return expf(x);
}

/* ---- Phase A: everything that does not depend on the carried state ---------------
 * grid (n_v_heads, nchunks_in_group), 256 threads. Outputs per (chunk, head):
 *   u_out [64][128], wk_out [64][128], kq_out [64][64], cg_out [64]. Rows past the end of
 * a partial chunk are written as zeros so the scan can always read full 64-row tiles. */
__global__ void __launch_bounds__(SSM2_THREADS)
ssm2_chunk_pre_kernel(float *__restrict__ u_out, float *__restrict__ wk_out,
                      float *__restrict__ kq_out, float *__restrict__ cg_out,
                      const float *__restrict__ conv, const float *__restrict__ alpha,
                      const float *__restrict__ beta,
                      int chunk0, int n_tokens, int conv_stride, int ab_stride,
                      int qk_dim, int n_v_heads, int repeat)
{
    SSM2_SMEM_F32(sm);
    float *sM    = sm;                                        /* [64][65]  strict-lower M */
    float *sK    = sM + SSM2_CS * SSM2_PRE_M_STRIDE;          /* [32][68]  K tile, transposed */
    float *sQ    = sK + 32 * SSM2_PRE_T_STRIDE;               /* [32][68]  Q tile, transposed */
    float *sCg   = sQ + 32 * SSM2_PRE_T_STRIDE;               /* [64] */
    float *sQd   = sCg + SSM2_CS;                             /* [64] */
    float *sBeta = sQd + SSM2_CS;                             /* [64] */

    const int tid = threadIdx.x;
    const int h = blockIdx.x, cidx = blockIdx.y, kh = h / repeat;
    const int t0 = (chunk0 + cidx) * SSM2_CS;
    int ca = n_tokens - t0; if (ca > SSM2_CS) ca = SSM2_CS;   /* tokens in this chunk (>= 1) */
    const size_t blk = (size_t)cidx * n_v_heads + h;

    /* per-token scalars; padded tokens get beta = 0 and gate_log = 0 */
    if (tid < SSM2_CS) {
        const bool ok = tid < ca;
        sBeta[tid] = ok ? beta [(size_t)(t0 + tid) * ab_stride + h] : 0.0f;
        sCg[tid]   = ok ? alpha[(size_t)(t0 + tid) * ab_stride + h] : 0.0f;   /* gate_log, becomes cum below */
    }
    __syncthreads();
    if (tid == 0) {                       /* inherently sequential, 64 adds: same order as the old kernel */
        float cum = 0.0f;
        for (int t = 0; t < SSM2_CS; t++) {
            cum += sCg[t];
            sCg[t] = cum;
            sQd[t] = ssm2_expc(cum);
        }
    }
    __syncthreads();
    if (tid < SSM2_CS) cg_out[blk * SSM2_CS + tid] = sCg[tid];

    /* ---- M = (K K^T) .* dm  and  kq = (Q K^T) .* dm, 4x4 register tile per thread ---- */
    const int ti = (tid >> 4) * 4, tj = (tid & 15) * 4;
    float am[4][4], aq[4][4];
    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) { am[a][b] = 0.0f; aq[a][b] = 0.0f; }

    for (int dk = 0; dk < SSM2_D; dk += 32) {
        for (int idx = tid; idx < 32 * SSM2_CS; idx += SSM2_THREADS) {
            const int kk = idx & 31, row = idx >> 5;           /* consecutive threads = consecutive d: coalesced */
            float kv = 0.0f, qv = 0.0f;
            if (row < ca) {
                const float *tok = conv + (size_t)(t0 + row) * conv_stride;
                qv = tok[kh * SSM2_D + dk + kk];
                kv = tok[qk_dim + kh * SSM2_D + dk + kk];
            }
            sK[kk * SSM2_PRE_T_STRIDE + row] = kv;
            sQ[kk * SSM2_PRE_T_STRIDE + row] = qv;
        }
        __syncthreads();
        for (int kk = 0; kk < 32; kk++) {
            const float4 ki = *(const float4 *)&sK[kk * SSM2_PRE_T_STRIDE + ti];
            const float4 kj = *(const float4 *)&sK[kk * SSM2_PRE_T_STRIDE + tj];
            const float4 qi = *(const float4 *)&sQ[kk * SSM2_PRE_T_STRIDE + ti];
            const float kia[4] = {ki.x, ki.y, ki.z, ki.w}, kja[4] = {kj.x, kj.y, kj.z, kj.w},
                        qia[4] = {qi.x, qi.y, qi.z, qi.w};
            for (int a = 0; a < 4; a++)
                for (int b = 0; b < 4; b++) {
                    am[a][b] = fmaf(kia[a], kja[b], am[a][b]);
                    aq[a][b] = fmaf(qia[a], kja[b], aq[a][b]);
                }
        }
        __syncthreads();
    }
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++) {
            const int i = ti + a, j = tj + b;
            float dmv = 0.0f;
            if (j == i) dmv = 1.0f;
            else if (j < i) dmv = ssm2_expc(sCg[i] - sCg[j]);
            sM[i * SSM2_PRE_M_STRIDE + j] = (j < i) ? am[a][b] * dmv : 0.0f;          /* strict lower */
            kq_out[blk * SSM2_CS * SSM2_CS + (size_t)i * SSM2_CS + j] = (j <= i) ? aq[a][b] * dmv : 0.0f;
        }
    __syncthreads();

    /* ---- forward substitution for the 256 right-hand-side columns ----------------------
     * col < 128 : u  column  (rhs_i = beta_i * v[i][col])
     * col >= 128: Wk column  (rhs_i = beta_i * qd_i * k[i][col-128])
     * x_i = rhs_i - beta_i * sum_{j<i} M[i][j] x_j. Each thread owns one column in
     * registers; M is read from shared memory as a broadcast. No barriers inside. */
    {
        const int col = tid;
        const bool is_u = col < SSM2_D;
        const int cc = is_u ? col : col - SSM2_D;
        const float *src = conv + (is_u ? (size_t)(2 * qk_dim + h * SSM2_D + cc) : (size_t)(qk_dim + kh * SSM2_D + cc));
        float x[SSM2_CS];
#pragma unroll
        for (int i = 0; i < SSM2_CS; i++) {
            const float bt = sBeta[i];
            float rhs = 0.0f;
            if (i < ca) {
                const float val = src[(size_t)(t0 + i) * conv_stride];
                rhs = is_u ? bt * val : bt * sQd[i] * val;
            }
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < i; j++) sum = fmaf(sM[i * SSM2_PRE_M_STRIDE + j], x[j], sum);
            x[i] = rhs - bt * sum;
        }
        float *dst = is_u ? u_out : wk_out;
#pragma unroll
        for (int i = 0; i < SSM2_CS; i++) dst[(blk * SSM2_CS + i) * SSM2_D + cc] = x[i];
    }
}

/* ---- Phase B: sequential scan over the chunks of one group ------------------------------
 * grid (n_v_heads, D/JT), 256 threads. A block owns JT=16 rows of S (r0 .. r0+15) in shared
 * memory. Per chunk:  vh = u - Wk S^T ; out = qd (Q S^T) + kq vh ; S = decay S + (vh dte)^T K.
 * thread mapping for the dot products: rl = tid & 15 (row of the slice), ig = tid >> 4,
 * handling tokens ig, ig+16, ig+32, ig+48. */
__global__ void __launch_bounds__(SSM2_THREADS)
ssm2_chunk_scan_kernel(float *__restrict__ state, float *__restrict__ xb2,
                       const float *__restrict__ u_in, const float *__restrict__ wk_in,
                       const float *__restrict__ kq_in, const float *__restrict__ cg_in,
                       const float *__restrict__ conv,
                       int chunk0, int nchunks, int n_tokens, int conv_stride, int xb2_stride,
                       int qk_dim, int n_v_heads, int repeat)
{
    SSM2_SMEM_F32(sm);
    float *sS   = sm;                                        /* [16][129] slice of the state */
    float *sW   = sS + SSM2_JT * SSM2_SCAN_ST;               /* [64][129] Wk, then Q, then K */
    float *sKQ  = sW + SSM2_CS * SSM2_SCAN_ST;               /* [64][65] */
    float *sVh  = sKQ + SSM2_CS * 65;                        /* [64][17] vh for this slice */
    float *sQd  = sVh + SSM2_CS * 17;                        /* [64] */
    float *sDte = sQd + SSM2_CS;                             /* [64] */

    const int tid = threadIdx.x;
    const int h = blockIdx.x, rb = blockIdx.y, kh = h / repeat, r0 = rb * SSM2_JT;
    const int rl = tid & 15, ig = tid >> 4;
    float *st_h = state + (size_t)h * SSM2_D * SSM2_D;

    for (int idx = tid; idx < SSM2_JT * SSM2_D; idx += SSM2_THREADS) {
        const int r = idx >> 7, c = idx & 127;
        sS[r * SSM2_SCAN_ST + c] = st_h[(size_t)(r0 + r) * SSM2_D + c];
    }
    __syncthreads();

    for (int ci = 0; ci < nchunks; ci++) {
        const int t0 = (chunk0 + ci) * SSM2_CS;
        int ca = n_tokens - t0; if (ca > SSM2_CS) ca = SSM2_CS;
        const size_t blk = (size_t)ci * n_v_heads + h;
        const float cum_last = cg_in[blk * SSM2_CS + (SSM2_CS - 1)];   /* padded gates are 0, so [63] is the last real one */
        const float total_decay = ssm2_expc(cum_last);

        /* (1) Wk tile, kq tile, per-token decays */
        for (int idx = tid; idx < SSM2_CS * SSM2_D; idx += SSM2_THREADS) {
            const int i = idx >> 7, c = idx & 127;
            sW[i * SSM2_SCAN_ST + c] = wk_in[(blk * SSM2_CS + i) * SSM2_D + c];
        }
        for (int idx = tid; idx < SSM2_CS * SSM2_CS; idx += SSM2_THREADS) {
            const int i = idx >> 6, j = idx & 63;
            sKQ[i * 65 + j] = kq_in[blk * SSM2_CS * SSM2_CS + idx];
        }
        if (tid < SSM2_CS) {
            const float cgi = cg_in[blk * SSM2_CS + tid];
            sQd[tid]  = ssm2_expc(cgi);
            sDte[tid] = ssm2_expc(cum_last - cgi);
        }
        __syncthreads();

        /* (2) vh[i][r] = u[i][r] - dot(Wk[i][:], S[r][:]) */
        for (int m = 0; m < 4; m++) {
            const int i = ig + 16 * m;
            float acc = 0.0f;
            for (int c = 0; c < SSM2_D; c++) acc = fmaf(sW[i * SSM2_SCAN_ST + c], sS[rl * SSM2_SCAN_ST + c], acc);
            sVh[i * 17 + rl] = u_in[(blk * SSM2_CS + i) * SSM2_D + r0 + rl] - acc;
        }
        __syncthreads();

        /* (3) Q tile over the Wk tile */
        for (int idx = tid; idx < SSM2_CS * SSM2_D; idx += SSM2_THREADS) {
            const int i = idx >> 7, c = idx & 127;
            sW[i * SSM2_SCAN_ST + c] = (i < ca) ? conv[(size_t)(t0 + i) * conv_stride + kh * SSM2_D + c] : 0.0f;
        }
        __syncthreads();

        /* (4) out[i][r] = qd[i] * dot(Q[i][:], S[r][:]) + sum_{j<=i} kq[i][j] vh[j][r] */
        for (int m = 0; m < 4; m++) {
            const int i = ig + 16 * m;
            float sq = 0.0f;
            for (int c = 0; c < SSM2_D; c++) sq = fmaf(sW[i * SSM2_SCAN_ST + c], sS[rl * SSM2_SCAN_ST + c], sq);
            float acc = sq * sQd[i];
            for (int j = 0; j <= i; j++) acc = fmaf(sKQ[i * 65 + j], sVh[j * 17 + rl], acc);
            if (i < ca) xb2[(size_t)(t0 + i) * xb2_stride + h * SSM2_D + r0 + rl] = acc;
        }
        __syncthreads();

        /* (5) K tile, then S = decay*S + sum_j (vh[j][r]*dte[j]) * K[j][:] */
        for (int idx = tid; idx < SSM2_CS * SSM2_D; idx += SSM2_THREADS) {
            const int i = idx >> 7, c = idx & 127;
            sW[i * SSM2_SCAN_ST + c] = (i < ca) ? conv[(size_t)(t0 + i) * conv_stride + qk_dim + kh * SSM2_D + c] : 0.0f;
        }
        __syncthreads();
        {
            const int r = tid >> 4, c16 = tid & 15;        /* element c = c16 + 16*e: conflict-free */
            float s[8];
            for (int e = 0; e < 8; e++) s[e] = sS[r * SSM2_SCAN_ST + c16 + 16 * e] * total_decay;
            for (int j = 0; j < SSM2_CS; j++) {
                const float vd = sVh[j * 17 + r] * sDte[j];
                for (int e = 0; e < 8; e++) s[e] = fmaf(sW[j * SSM2_SCAN_ST + c16 + 16 * e], vd, s[e]);
            }
            for (int e = 0; e < 8; e++) sS[r * SSM2_SCAN_ST + c16 + 16 * e] = s[e];
        }
        __syncthreads();
    }

    for (int idx = tid; idx < SSM2_JT * SSM2_D; idx += SSM2_THREADS) {
        const int r = idx >> 7, c = idx & 127;
        st_h[(size_t)(r0 + r) * SSM2_D + c] = sS[r * SSM2_SCAN_ST + c];
    }
}

#endif /* BACKEND_GPU_SSM_CHUNK2_CUH */
