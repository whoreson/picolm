/* gemm_emu: executes the REAL backend_gpu_gemm_f16.cuh kernel on the CPU (ldmatrix,
 * mma.m16n8k16, cp.async, cvt.f16x2, shuffles and barriers emulated to their PTX
 * semantics, one OS thread per CUDA thread) and checks it against references built
 * from the project's own CPU dequantisers in quant.c.
 *
 * Two checks per configuration:
 *   identity  X = I (S = I), so Y[s][o] = fp16(W[o][s]) with a single product per
 *             output and fp32 accumulation of zeros: the result must EQUAL the
 *             fp16-rounded CPU dequant of every weight, bit for bit. This verifies
 *             the decode, the swizzle and the ldmatrix/mma mapping for all of W.
 *   random    ragged S and O, x_stride > I, y_stride > O: compared with a double
 *             precision reference (fp16-rounded X and W, exact sums).
 *
 *   gcc -c quant.c -I. -o quant.o
 *   g++ -std=c++11 -O2 -pthread -DPICOLM_FA3_EMU -Wno-unknown-pragmas -I. -Itests \
 *       tests/gemm_emu.cpp quant.o -o gemm_emu && ./gemm_emu
 * Build with -fsanitize=address to also catch out-of-bounds reads of the weights
 * (the weight buffer is allocated with NO slack, so e.g. a Q6_K window that read 2
 * bytes past the tensor would be reported). */
#include "emu_prims.h"
#define PICOLM_FA3_EMU 1
extern "C" {
#include "quant.h"
}
#include "backend_gpu_gemm_f16.cuh"

static uint32_t rs = 777;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float frand() { return (rnd() % 20001) / 10000.0f - 1.0f; }

static int fmt_ggml(int fmt) { return fmt; }
static size_t row_bytes_of(int fmt, int I) {
    switch (fmt) {
    case 5: return (size_t)(I / 256) * 176;   /* Q5_K */
    case 6: return (size_t)(I / 256) * 210;   /* Q6_K */
    case 7: return (size_t)(I / 256) * 212;   /* IQ6_K */
    default: return (size_t)(I / 32) * 34;    /* Q8_0 */
    }
}

static void make_weights(std::vector<uint8_t> &w, int fmt, int I, int O) {
    const size_t rb = row_bytes_of(fmt, I);
    w.assign((size_t)O * rb, 0);
    for (size_t i = 0; i < w.size(); i++) w[i] = (uint8_t)(rnd() >> 11);
    for (int o = 0; o < O; o++)                       /* sane fp16 scales so values stay O(1) */
        for (size_t b = 0; b < rb; ) {
            uint8_t *blk = &w[(size_t)o * rb + b];
            const uint16_t d = f2h(0.002f + 0.01f * (frand() * 0.5f + 0.5f));
            const uint16_t dm = f2h(0.002f + 0.01f * (frand() * 0.5f + 0.5f));
            if (fmt == 5) { memcpy(blk, &d, 2); memcpy(blk + 2, &dm, 2); b += 176; }
            else if (fmt == 6) { memcpy(blk + 208, &d, 2); for (int k = 0; k < 16; k++) blk[192 + k] = (uint8_t)(int8_t)((int)(rnd() % 41) - 20); b += 210; }
            else if (fmt == 7) { /* IQ6_K: d, extra, int8 scales[16]; qs/qh stay random.
                                  * extra is a full random uint16 (all 16 LUT-select bits valid),
                                  * scales in [-20,20) keep dl*lut within fp16 range like Q6_K. */
                memcpy(blk, &d, 2);                       /* d at byte 0 */
                for (int k = 0; k < 16; k++) blk[4 + k] = (uint8_t)(int8_t)((int)(rnd() % 41) - 20);
                b += 212; }
            else { memcpy(blk, &d, 2); b += 34; }
        }
}
static void ref_dequant(const std::vector<uint8_t> &w, int fmt, int I, int O, std::vector<float> &out) {
    out.assign((size_t)O * I, 0.f);
    const size_t rb = row_bytes_of(fmt, I);
    for (int o = 0; o < O; o++) {
        const uint8_t *r = &w[(size_t)o * rb]; float *d = &out[(size_t)o * I];
        if (fmt == 5) dequantize_row_q5_K(r, d, I); else if (fmt == 6) dequantize_row_q6_K(r, d, I);
        else if (fmt == 7) dequantize_row_iq6_k(r, d, I); else dequantize_row_q8_0(r, d, I);
    }
}

template <int FMT, int BM, int BN>
static void run_kernel(std::vector<float> &y, const std::vector<float> &x, const std::vector<uint8_t> &w,
                       int S, int I, int O, int xstride, int ystride) {
    const size_t rb = row_bytes_of(FMT, I);
    const size_t smem_bytes = gemmf_smem_bytes(FMT, BM, BN);
    std::vector<uint8_t> smem(smem_bytes + 64, 0xFF);   /* NaN fp16 patterns in any smem the kernel fails to define */
    const unsigned gx = (S + BN - 1) / BN, gy = (O + BM - 1) / BM;
    for (unsigned by = 0; by < gy; by++)
        for (unsigned bx = 0; bx < gx; bx++) {
            Ctx ctx; ctx.blk = new EmuBarrier(GEMMF_THREADS);
            for (int k = 0; k < 8; k++) ctx.wbar[k] = new EmuBarrier(32);
            ctx.smem = (uint8_t *)(((uintptr_t)smem.data() + 15) & ~(uintptr_t)15);
            g_ctx = &ctx;
            std::vector<std::thread> th;
            for (int t = 0; t < GEMMF_THREADS; t++)
                th.push_back(std::thread([&, t] {
                    threadIdx.x = (unsigned)t; threadIdx.y = 0; threadIdx.z = 0;
                    blockIdx.x = bx; blockIdx.y = by; blockIdx.z = 0;
                    picolm_gpu_gemm_f16_kernel<FMT, BM, BN>(y.data(), x.data(), w.data(), S, I, O, rb, xstride, ystride);
                }));
            for (size_t i = 0; i < th.size(); i++) th[i].join();
            delete ctx.blk; for (int k = 0; k < 8; k++) delete ctx.wbar[k];
        }
}

template <int FMT, int BM, int BN>
static int run_case(int S, int I, int O, bool identity) {
    std::vector<uint8_t> w; make_weights(w, FMT, I, O);
    std::vector<float> W; ref_dequant(w, FMT, I, O, W);
    const int xs = identity ? I : I + 8, ys = identity ? O : O + 2;
    std::vector<float> x((size_t)S * xs, 0.f), y((size_t)S * ys, NAN);
    if (identity) { for (int s = 0; s < S && s < I; s++) x[(size_t)s * xs + s] = 1.0f; }
    else for (int s = 0; s < S; s++) for (int k = 0; k < I; k++) x[(size_t)s * xs + k] = frand() * 2.0f;
    if (!identity && S > 5 && I > 70) {            /* activation outliers: the kernel must saturate to +-65504, never inf */
        x[(size_t)1 * xs + 3] = 1.0e6f; x[(size_t)(S - 1) * xs + I - 2] = -3.0e7f; x[(size_t)4 * xs + 69] = 70000.f;
    }
    run_kernel<FMT, BM, BN>(y, x, w, S, I, O, xs, ys);

    double maxe = 0, maxr = 0, worst_rel = 0; size_t bad = 0, mism = 0;
    std::vector<double> ref((size_t)S * O);
    std::vector<double> rowmax(S, 0.0);
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            double r = 0;
            if (identity) r = h2f(f2h(W[(size_t)o * I + s]));
            else for (int k = 0; k < I; k++)
                r += (double)h2f(f2h(std::min(65504.f, std::max(-65504.f, x[(size_t)s * xs + k])))) * (double)h2f(f2h(W[(size_t)o * I + k]));
            ref[(size_t)s * O + o] = r; rowmax[s] = std::max(rowmax[s], fabs(r)); maxr = std::max(maxr, fabs(r));
        }
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            const float got = y[(size_t)s * ys + o];
            if (!std::isfinite(got)) { bad++; continue; }
            const double r = ref[(size_t)s * O + o], e = fabs((double)got - r);
            if (identity && got != (float)r) mism++;
            maxe = std::max(maxe, e);
            worst_rel = std::max(worst_rel, e / (rowmax[s] > 0 ? rowmax[s] : 1.0));   /* error relative to the token row's own scale */
        }
    const bool fail = bad || mism || !(worst_rel <= (identity ? 0.0 : 2e-6));
    printf("%s fmt=%d BM=%-3d BN=%-3d %-8s S=%-4d I=%-4d O=%-4d  max_err %.2e (|y|max %.3f, worst row-relative %.1e) nonfinite=%zu inexact=%zu\n",
           fail ? "FAIL" : "ok  ", FMT, BM, BN, identity ? "identity" : "random", S, I, O, maxe, maxr, worst_rel, bad, mism);
    return fail;
}

template <int FMT> static int run_fmt(bool quick) {
    int f = 0;
    f |= run_case<FMT, 128, 128>(256, 256, 72, true);          /* every weight of a 256-deep tensor, exactly */
    f |= run_case<FMT, 64, 64>(256, 256, 72, true);
    f |= run_case<FMT, 128, 128>(70, 512, 136, false);         /* ragged S and O, strided x and y */
    f |= run_case<FMT, 64, 32>(50, 512, 72, false);
    if (!quick) {
        f |= run_case<FMT, 128, 64>(512, 512, 136, true);      /* identity over a deeper K and 2 stages */
        f |= run_case<FMT, 128, 64>(130, 768, 200, false);
        f |= run_case<FMT, 64, 128>(40, 256, 24, false);
    }
    return f;
}

int main(int argc, char **argv) {
    const bool quick = argc > 1 && !strcmp(argv[1], "quick");
    fp16_table_init();     /* quant.c's Q6_K / Q8_0 dequantisers read d through a lookup table */
    int f = 0;
    f |= run_fmt<GEMMF_Q8_0>(quick);
    f |= run_fmt<GEMMF_Q5_K>(quick);
    f |= run_fmt<GEMMF_Q6_K>(quick);
    f |= run_fmt<GEMMF_IQ6_K>(quick);
    printf(f ? "FAILED\n" : "all ok\n");
    return f;
}
