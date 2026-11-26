/* ssm_chunk2_emu: executes the REAL backend_gpu_ssm_chunk2.cuh kernels on the CPU (one OS
 * thread per CUDA thread, real barriers) and checks them against an INDEPENDENT reference:
 * the token-sequential gated delta rule in double precision,
 *     S <- exp(g_t) S;  pred = S k_t;  delta = beta_t (v_t - pred);  S <- S + delta (x) k_t;
 *     out_t = S q_t
 * which is what the chunked algorithm is an algebraic rewrite of. It is NOT a port of the
 * kernels under test.
 *
 * Covers: a partial tail chunk (n % 64 != 0), the state carried across TWO separate launch
 * groups (chunk0 offsets, scratch indexing), GQA-style sharing of q/k heads (repeat = 3),
 * strided conv / gate buffers, a strongly gated case (cumulative log-gate far below -50, so
 * the +-50 clamps are exercised) and beta close to 1.
 *
 *   g++ -std=c++11 -O2 -pthread -DPICOLM_FA3_EMU -Wno-unknown-pragmas -I. -Itests \
 *       tests/ssm_chunk2_emu.cpp -o ssm_chunk2_emu && ./ssm_chunk2_emu [quick] */
#include "emu_prims.h"
#define PICOLM_FA3_EMU 1
#include "backend_gpu_ssm_chunk2.cuh"

static uint32_t rs = 99;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float frand() { return (rnd() % 20001) / 10000.0f - 1.0f; }       /* [-1, 1] */
static float urand() { return (rnd() % 10001) / 10000.0f; }              /* [0, 1]  */

template <class F>
static void run_grid(unsigned gx, unsigned gy, size_t smem_floats, F kern) {
    std::vector<float> smem(smem_floats + 8, 7.0e30f);       /* poison: any smem the kernel fails to define is obvious */
    for (unsigned by = 0; by < gy; by++)
        for (unsigned bx = 0; bx < gx; bx++) {
            Ctx ctx; ctx.blk = new EmuBarrier(SSM2_THREADS);
            for (int k = 0; k < 8; k++) ctx.wbar[k] = new EmuBarrier(32);
            ctx.smem = (uint8_t *)(((uintptr_t)smem.data() + 15) & ~(uintptr_t)15);
            g_ctx = &ctx;
            std::vector<std::thread> th;
            for (int t = 0; t < SSM2_THREADS; t++)
                th.push_back(std::thread([&, t] {
                    threadIdx.x = (unsigned)t; threadIdx.y = 0; threadIdx.z = 0;
                    blockIdx.x = bx; blockIdx.y = by; blockIdx.z = 0;
                    kern();
                }));
            for (size_t i = 0; i < th.size(); i++) th[i].join();
            delete ctx.blk; for (int k = 0; k < 8; k++) delete ctx.wbar[k];
        }
}

struct Cfg { int n_tokens, n_k, n_v; float gate_lo, beta_hi; bool split; const char *name; };

static int run(const Cfg &cf) {
    const int D = SSM2_D, n_k = cf.n_k, n_v = cf.n_v, repeat = n_v / n_k, T = cf.n_tokens;
    const int qk_dim = n_k * D;
    const int conv_stride = 2 * qk_dim + n_v * D + 24;       /* padded rows */
    const int ab_stride = n_v * D + 40;                      /* like the real call: ab_stride == xb2_stride */
    const int xb2_stride = n_v * D;
    std::vector<float> conv((size_t)T * conv_stride, 0.f), alpha((size_t)T * ab_stride, 0.f), beta((size_t)T * ab_stride, 0.f);
    for (int t = 0; t < T; t++) {
        float *row = &conv[(size_t)t * conv_stride];
        for (int kh = 0; kh < n_k; kh++) {                   /* q, k L2-normalised per head (as the model does) */
            double nq = 0, nk = 0;
            for (int c = 0; c < D; c++) { row[kh * D + c] = frand(); row[qk_dim + kh * D + c] = frand(); nq += row[kh*D+c]*row[kh*D+c]; nk += row[qk_dim+kh*D+c]*row[qk_dim+kh*D+c]; }
            for (int c = 0; c < D; c++) { row[kh * D + c] /= (float)sqrt(nq); row[qk_dim + kh * D + c] /= (float)sqrt(nk); }
        }
        for (int i = 0; i < n_v * D; i++) row[2 * qk_dim + i] = frand();
        for (int h = 0; h < n_v; h++) {
            alpha[(size_t)t * ab_stride + h] = -urand() * (-cf.gate_lo);              /* gate_log in [gate_lo, 0] */
            beta [(size_t)t * ab_stride + h] = 0.05f + urand() * (cf.beta_hi - 0.05f);
        }
    }
    std::vector<float> state((size_t)n_v * D * D);
    for (auto &x : state) x = 0.1f * frand();
    std::vector<float> state0 = state;

    /* ---------------- reference: token-sequential, double ---------------- */
    std::vector<double> ref_out((size_t)T * n_v * D, 0.0), ref_state((size_t)n_v * D * D);
    for (int h = 0; h < n_v; h++) {
        const int kh = h / repeat;
        std::vector<double> S(D * D);
        for (int i = 0; i < D * D; i++) S[i] = state0[(size_t)h * D * D + i];
        for (int t = 0; t < T; t++) {
            const float *row = &conv[(size_t)t * conv_stride];
            const double g = alpha[(size_t)t * ab_stride + h], bt = beta[(size_t)t * ab_stride + h], dec = exp(g);
            for (int i = 0; i < D * D; i++) S[i] *= dec;
            for (int r = 0; r < D; r++) {
                double pred = 0; for (int c = 0; c < D; c++) pred += S[r * D + c] * (double)row[qk_dim + kh * D + c];
                const double delta = bt * ((double)row[2 * qk_dim + h * D + r] - pred);
                for (int c = 0; c < D; c++) S[r * D + c] += delta * (double)row[qk_dim + kh * D + c];
            }
            for (int r = 0; r < D; r++) {
                double o = 0; for (int c = 0; c < D; c++) o += S[r * D + c] * (double)row[kh * D + c];
                ref_out[((size_t)t * n_v + h) * D + r] = o;
            }
        }
        for (int i = 0; i < D * D; i++) ref_state[(size_t)h * D * D + i] = S[i];
    }

    /* ---------------- the kernels under test ---------------- */
    const int nc = (T + SSM2_CS - 1) / SSM2_CS;
    std::vector<float> xb2((size_t)T * xb2_stride, NAN);     /* NaN: an unwritten output fails */
    const int groups = cf.split && nc >= 2 ? 2 : 1;
    int c_begin = 0;
    for (int g = 0; g < groups; g++) {
        const int ng = groups == 1 ? nc : (g == 0 ? nc - 1 : 1);
        const size_t nb = (size_t)ng * n_v;
        std::vector<float> u(nb * SSM2_CS * D, NAN), wk(nb * SSM2_CS * D, NAN), kq(nb * SSM2_CS * SSM2_CS, NAN), cg(nb * SSM2_CS, NAN);
        const int chunk0 = c_begin;
        run_grid((unsigned)n_v, (unsigned)ng, SSM2_PRE_FLOATS, [&] {
            ssm2_chunk_pre_kernel(u.data(), wk.data(), kq.data(), cg.data(), conv.data(), alpha.data(), beta.data(),
                                  chunk0, T, conv_stride, ab_stride, qk_dim, n_v, repeat);
        });
        run_grid((unsigned)n_v, (unsigned)(D / SSM2_JT), SSM2_SCAN_FLOATS, [&] {
            ssm2_chunk_scan_kernel(state.data(), xb2.data(), u.data(), wk.data(), kq.data(), cg.data(), conv.data(),
                                   chunk0, ng, T, conv_stride, xb2_stride, qk_dim, n_v, repeat);
        });
        c_begin += ng;
    }

    /* ---------------- compare ---------------- */
    double max_out = 0, max_state = 0, err_out = 0, err_state = 0; size_t bad = 0;
    for (int t = 0; t < T; t++)
        for (int h = 0; h < n_v; h++)
            for (int r = 0; r < D; r++) {
                const float got = xb2[(size_t)t * xb2_stride + h * D + r];
                const double rv = ref_out[((size_t)t * n_v + h) * D + r];
                if (!std::isfinite(got)) { bad++; continue; }
                err_out = std::max(err_out, fabs((double)got - rv)); max_out = std::max(max_out, fabs(rv));
            }
    for (size_t i = 0; i < state.size(); i++) {
        if (!std::isfinite(state[i])) { bad++; continue; }
        err_state = std::max(err_state, fabs((double)state[i] - ref_state[i])); max_state = std::max(max_state, fabs(ref_state[i]));
    }
    const double rel_out = err_out / (max_out > 0 ? max_out : 1), rel_state = err_state / (max_state > 0 ? max_state : 1);
    const bool fail = bad || !(rel_out < 2e-4) || !(rel_state < 2e-4);
    printf("%s %-24s T=%-4d heads=%d/%d chunks=%d%s  out: max|ref| %.3f rel_err %.1e   state: max|ref| %.3f rel_err %.1e  nonfinite=%zu\n",
           fail ? "FAIL" : "ok  ", cf.name, T, n_k, n_v, nc, groups == 2 ? " (2 groups)" : "", max_out, rel_out, max_state, rel_state, bad);
    return fail;
}

int main(int argc, char **argv) {
    const bool quick = argc > 1 && !strcmp(argv[1], "quick");
    int f = 0;
    f |= run({64,  2, 6, -0.4f, 0.9f, false, "one full chunk"});
    f |= run({150, 2, 6, -0.4f, 0.9f, false, "partial tail (150)"});
    f |= run({150, 2, 6, -0.4f, 0.9f, true,  "state across 2 groups"});
    f |= run({40,  1, 3, -0.4f, 0.9f, false, "single partial chunk"});
    if (!quick) {
        f |= run({200, 2, 6, -4.0f, 0.99f, false, "strong gating + clamps"});
        f |= run({129, 2, 6, -0.05f, 0.99f, true, "weak gating, beta~1"});
        f |= run({65,  4, 12, -0.4f, 0.9f, false, "chunk+1 token, repeat 3"});
    }
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
