/* test_ssm_chunk2_gpu: the restructured chunked gated-delta-rule kernels
 * (backend_gpu_ssm_chunk2.cuh) on real hardware, at the real Qwen3.5/3.6 sizes
 * (48 v-heads, 16 k-heads, d = 128), against the token-sequential double-precision
 * reference. A subset of heads is checked (the sequential reference is O(T * d^2) per head).
 *
 * Pass = relative error of outputs and final state <= 1e-4 and every output finite
 * (xb2 is pre-filled with NaN, so an unwritten element fails).
 *
 * It also times the two launches per 512-token group and projects the cost over 48 SSM layers,
 * for comparison with the old pipeline's ssm_chunk_rec profile slot (about 0.65 ms/token).
 * The launch loop below mirrors ssm2_chunked_dev() in backend_gpu_host_misc.cu.
 *
 *   make test-ssm-chunk2-gpu && ./test_ssm_chunk2_gpu */
#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include "backend_gpu_ssm_chunk2.cuh"

static uint32_t rs = 4711;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float frand() { return (rnd() % 20001) / 10000.0f - 1.0f; }
static float urand() { return (rnd() % 10001) / 10000.0f; }

static int run(int T, float gate_lo, const char *name, int bench) {
    const int D = SSM2_D, n_k = 16, n_v = 48, repeat = n_v / n_k, qk_dim = n_k * D;
    const int xb2_stride = n_v * D + 64;                 /* conv, gate and beta rows share this stride, as in the model */
    const int conv_stride = xb2_stride > 2 * qk_dim + n_v * D ? xb2_stride : 2 * qk_dim + n_v * D;
    std::vector<float> conv((size_t)T * conv_stride, 0.f), alpha((size_t)T * conv_stride, 0.f), beta((size_t)T * conv_stride, 0.f);
    for (int t = 0; t < T; t++) {
        float *row = &conv[(size_t)t * conv_stride];
        for (int kh = 0; kh < n_k; kh++) {
            double nq = 0, nk = 0;
            for (int c = 0; c < D; c++) { row[kh*D+c] = frand(); row[qk_dim+kh*D+c] = frand(); nq += row[kh*D+c]*row[kh*D+c]; nk += row[qk_dim+kh*D+c]*row[qk_dim+kh*D+c]; }
            for (int c = 0; c < D; c++) { row[kh*D+c] /= (float)sqrt(nq); row[qk_dim+kh*D+c] /= (float)sqrt(nk); }
        }
        for (int i = 0; i < n_v * D; i++) row[2 * qk_dim + i] = frand();
        for (int h = 0; h < n_v; h++) { alpha[(size_t)t*conv_stride+h] = -urand() * (-gate_lo); beta[(size_t)t*conv_stride+h] = 0.05f + urand() * 0.9f; }
    }
    std::vector<float> state((size_t)n_v * D * D); for (auto &x : state) x = 0.1f * frand();
    std::vector<float> state0 = state;

    float *dconv, *dalpha, *dbeta, *dstate, *dxb2, *du, *dwk, *dkq, *dcg;
    const int nc = (T + SSM2_CS - 1) / SSM2_CS, grp = nc < 8 ? nc : 8;
    const size_t nb = (size_t)grp * n_v;
    cudaMalloc(&dconv, conv.size()*4); cudaMalloc(&dalpha, alpha.size()*4); cudaMalloc(&dbeta, beta.size()*4);
    cudaMalloc(&dstate, state.size()*4); cudaMalloc(&dxb2, (size_t)T*xb2_stride*4);
    cudaMalloc(&du, nb*SSM2_CS*D*4); cudaMalloc(&dwk, nb*SSM2_CS*D*4); cudaMalloc(&dkq, nb*SSM2_CS*SSM2_CS*4); cudaMalloc(&dcg, nb*SSM2_CS*4);
    cudaMemcpy(dconv, conv.data(), conv.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, alpha.data(), alpha.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, beta.data(), beta.size()*4, cudaMemcpyHostToDevice);
    const size_t smem_pre = (size_t)SSM2_PRE_FLOATS * 4, smem_scan = (size_t)SSM2_SCAN_FLOATS * 4;
    cudaFuncSetAttribute((const void *)ssm2_chunk_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_scan);

    auto launch_all = [&]() {
        for (int g0 = 0; g0 < nc; g0 += 8) {
            const int ng = (nc - g0) < 8 ? (nc - g0) : 8;
            ssm2_chunk_pre_kernel<<<dim3(n_v, ng), SSM2_THREADS, smem_pre>>>(du, dwk, dkq, dcg, dconv, dalpha, dbeta, g0, T, xb2_stride, xb2_stride, qk_dim, n_v, repeat);
            ssm2_chunk_scan_kernel<<<dim3(n_v, D / SSM2_JT), SSM2_THREADS, smem_scan>>>(dstate, dxb2, du, dwk, dkq, dcg, dconv, g0, ng, T, xb2_stride, xb2_stride, qk_dim, n_v, repeat);
        }
    };
    cudaMemcpy(dstate, state0.data(), state0.size()*4, cudaMemcpyHostToDevice);
    cudaMemset(dxb2, 0xFF, (size_t)T * xb2_stride * 4);
    launch_all(); cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    std::vector<float> xb2((size_t)T * xb2_stride); cudaMemcpy(xb2.data(), dxb2, xb2.size()*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.data(), dstate, state.size()*4, cudaMemcpyDeviceToHost);

    /* reference on a subset of heads */
    const int heads[4] = {0, 17, 31, 47};
    double max_out = 0, err_out = 0, max_st = 0, err_st = 0; size_t bad = 0;
    for (size_t i = 0; i < xb2.size(); i++) { /* every written location must be finite */ }
    for (int t = 0; t < T; t++) for (int h = 0; h < n_v; h++) for (int r = 0; r < D; r++) if (!std::isfinite(xb2[(size_t)t*xb2_stride + h*D + r])) bad++;
    for (int hi = 0; hi < 4; hi++) {
        const int h = heads[hi], kh = h / repeat;
        std::vector<double> S(D * D);
        for (int i = 0; i < D * D; i++) S[i] = state0[(size_t)h * D * D + i];
        for (int t = 0; t < T; t++) {
            const float *row = &conv[(size_t)t * conv_stride];
            const double bt = beta[(size_t)t*conv_stride+h], dec = exp((double)alpha[(size_t)t*conv_stride+h]);
            for (int i = 0; i < D * D; i++) S[i] *= dec;
            for (int r = 0; r < D; r++) {
                double pred = 0; for (int c = 0; c < D; c++) pred += S[r*D+c] * (double)row[qk_dim + kh*D + c];
                const double delta = bt * ((double)row[2*qk_dim + h*D + r] - pred);
                for (int c = 0; c < D; c++) S[r*D+c] += delta * (double)row[qk_dim + kh*D + c];
            }
            for (int r = 0; r < D; r++) {
                double o = 0; for (int c = 0; c < D; c++) o += S[r*D+c] * (double)row[kh*D + c];
                const double got = xb2[(size_t)t*xb2_stride + h*D + r];
                if (std::isfinite(got)) { err_out = std::max(err_out, fabs(got - o)); max_out = std::max(max_out, fabs(o)); }
            }
        }
        for (int i = 0; i < D * D; i++) { err_st = std::max(err_st, fabs((double)state[(size_t)h*D*D+i] - S[i])); max_st = std::max(max_st, fabs(S[i])); }
    }
    const double rel_o = err_out / (max_out > 0 ? max_out : 1), rel_s = err_st / (max_st > 0 ? max_st : 1);
    const int fail = err != cudaSuccess || bad || !(rel_o <= 1e-4) || !(rel_s <= 1e-4);
    printf("%s %-22s T=%-5d out rel_err %.1e  state rel_err %.1e  nonfinite=%zu%s\n", fail ? "FAIL" : "ok  ", name, T, rel_o, rel_s, bad,
           err != cudaSuccess ? cudaGetErrorString(err) : "");
    if (bench) {
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        launch_all(); cudaDeviceSynchronize();
        const int iters = 20; cudaEventRecord(a); for (int i = 0; i < iters; i++) launch_all(); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
        printf("      timing: %.3f ms per layer for %d tokens -> x48 layers = %.2f ms = %.4f ms/token (the old pipeline profiled ~0.65 ms/token)\n", ms, T, ms * 48, ms * 48 / T);
    }
    cudaFree(dconv); cudaFree(dalpha); cudaFree(dbeta); cudaFree(dstate); cudaFree(dxb2); cudaFree(du); cudaFree(dwk); cudaFree(dkq); cudaFree(dcg);
    return fail;
}

int main(void) {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (sm_%d%d, %d SMs), max opt-in smem/block %zu B\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.sharedMemPerBlockOptin);
    int f = 0;
    f |= run(64,   -0.4f, "one chunk", 0);
    f |= run(150,  -0.4f, "partial tail", 0);
    f |= run(512,  -0.4f, "ubatch of 512", 1);
    f |= run(1000, -0.4f, "2 groups + tail", 0);
    f |= run(512,  -4.0f, "strong gating", 0);
    f |= run(86,   -0.4f, "bench chunk (86)", 1);
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
