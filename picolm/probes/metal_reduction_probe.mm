// probes/metal_reduction_probe.mm
//
// Self-contained smoke test for the PicoLM Metal backend's kernel mechanism
// (raw Metal framework, no MLX-C/CMake). Run with: make metal-probe-run
//
// It exercises, in one small GEMV, everything the backend kernels depend on:
//   1. compiling a Metal Shading Language kernel from a source string at runtime
//   2. reading raw GGUF bytes via `device const uint8_t*`
//   3. dequantizing Q4_0 blocks on the fly (fp16 scale via as_type<half>)
//   4. cooperating across a threadgroup with shared memory + tree reduction
//   5. shared-storage (unified-memory) buffers for host<->device data
//
// Computes y[O] = W[O,I] @ x[I] on the GPU and diffs against a CPU reference;
// prints PASS/FAIL. Catches toolchain/MSL regressions independently of any
// model file.
//
// Computes y[O] = W[O,I] @ x[I], W in Q4_0 (18B per 32 values),
// O=4, I=64, TPB=32 (each thread handles 2 blocks, strided). Compares to a
// CPU reference; prints PASS/FAIL.
//
// BUILD:  make metal-probe       (or see Makefile)
// RUN:    ./probes/metal_reduction_probe

#include <metal/Metal.h>
#include <Foundation/Foundation.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define O 4
#define I 64
#define TPB 32
#define BLOCKS_PER_ROW (I / 32)         // 2
#define Q4_0_BLOCK_BYTES 18             // uint16 d + uint8 qs[16]
#define ROW_BYTES (BLOCKS_PER_ROW * Q4_0_BLOCK_BYTES)

// ---- host-side fp16 <-> fp32 (matches PicoLM's quant.c software path) ----
static float fp16_to_fp32_sw(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant =  h        & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; }
              mant &= 0x3ff;
              f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13); }
    } else { f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13); }
    float out; memcpy(&out, &f, 4); return out;
}
static uint16_t fp32_to_fp16_sw(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant |= 0x800000;
        uint32_t sh = (uint32_t)(14 - exp);
        uint32_t mo = mant >> sh;
        if ((mant >> (sh - 1)) & 1) mo++;
        return sign | mo;
    } else if (exp >= 0x1f) { return sign | 0x7c00; }
    uint32_t out = sign | (exp << 10) | (mant >> 13);
    if ((mant >> 12) & 1) out++;
    return out;
}

// ---- Metal Shading Language kernel (compiled at runtime from this string) ----
//   buffer(0): w   device const uint8_t*  GGUF byte blob [O*ROW_BYTES]
//   buffer(1): x   device const float*    activations [I]
//   buffer(2): y   device float*          output [O]
//   buffer(3): I   constant int           (via setBytes)
//   buffer(4): TPB constant int           (via setBytes)
static NSString* KERNEL_SRC = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
kernel void gemv_q4_0(device const uint8_t* w [[buffer(0)]],\n\
                      device const float*   x [[buffer(1)]],\n\
                      device float*         y [[buffer(2)]],\n\
                      constant int& I       [[buffer(3)]],\n\
                      constant int& TPB     [[buffer(4)]],\n\
                      uint tid [[thread_index_in_threadgroup]],\n\
                      uint gp  [[threadgroup_position_in_grid]]) {\n\
    int o = (int)gp;\n\
    const int BLOCKS = I / 32;\n\
    const device uint8_t* row = w + (unsigned long)o * (unsigned long)(BLOCKS * 18);\n\
    float sum = 0.0f;\n\
    for (int bi = (int)tid; bi < BLOCKS; bi += (int)TPB) {\n\
        const device uint8_t* blk = row + (unsigned long)(bi * 18);\n\
        uint16_t d_raw = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);\n\
        float d = (float)as_type<half>(d_raw);\n\
        for (int j = 0; j < 32; j++) {\n\
            int nib = (blk[2 + (j >> 1)] >> ((j & 1) * 4)) & 0xF;\n\
            sum += x[bi * 32 + j] * (float)(nib - 8) * d;\n\
        }\n\
    }\n\
    threadgroup float shared[256];\n\
    shared[tid] = sum;\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    for (int n = TPB >> 1; n > 0; n >>= 1) {\n\
        if (tid < (uint)n) shared[tid] += shared[tid + n];\n\
        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    }\n\
    if (tid == 0) y[o] = shared[0];\n\
}\n";

int main(void) {
    @autoreleasepool {
        // ---- 0. device ----
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "FAIL: no Metal device\n"); return 1; }
        NSLog(@"[probe] device: %@", device.name);

        // ---- 1. build host data ----
        uint8_t w_host[O * ROW_BYTES];
        for (int o = 0; o < O; o++) {
            for (int bi = 0; bi < BLOCKS_PER_ROW; bi++) {
                uint8_t *blk = w_host + o * ROW_BYTES + bi * Q4_0_BLOCK_BYTES;
                float scale = 0.5f * (float)(bi + 1) + 0.1f * (float)o;
                uint16_t d = fp32_to_fp16_sw(scale);
                blk[0] = d & 0xFF; blk[1] = (d >> 8) & 0xFF;
                for (int b = 0; b < 16; b++) {
                    uint8_t lo = (uint8_t)((o * 7 + bi * 3 + b * 5) & 0xF);
                    uint8_t hi = (uint8_t)((o * 11 + bi * 13 + b * 2) & 0xF);
                    blk[2 + b] = lo | (hi << 4);
                }
            }
        }
        float x_host[I];
        for (int i = 0; i < I; i++) x_host[i] = (float)(i - I / 2) * 0.01f;

        // ---- 2. CPU reference ----
        float y_ref[O];
        for (int o = 0; o < O; o++) {
            double acc = 0.0;
            for (int bi = 0; bi < BLOCKS_PER_ROW; bi++) {
                const uint8_t *blk = w_host + o * ROW_BYTES + bi * Q4_0_BLOCK_BYTES;
                uint16_t d_raw = blk[0] | ((uint16_t)blk[1] << 8);
                float d = fp16_to_fp32_sw(d_raw);
                for (int j = 0; j < 32; j++) {
                    int nib = (blk[2 + (j >> 1)] >> ((j & 1) * 4)) & 0xF;
                    acc += x_host[bi * 32 + j] * (double)(nib - 8) * (double)d;
                }
            }
            y_ref[o] = (float)acc;
        }

        // ---- 3. compile kernel ----
        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:KERNEL_SRC
                                                  options:nil
                                                    error:&err];
        if (!lib) { fprintf(stderr, "FAIL: compile library: %s\n",
                            [[err localizedDescription] UTF8String]); return 1; }
        id<MTLFunction> func = [lib newFunctionWithName:@"gemv_q4_0"];
        id<MTLComputePipelineState> ps =
            [device newComputePipelineStateWithFunction:func error:&err];
        if (!ps) { fprintf(stderr, "FAIL: pipeline: %s\n",
                           [[err localizedDescription] UTF8String]); return 1; }

        // ---- 4. buffers (shared storage = unified memory, CPU+GPU visible) ----
        id<MTLBuffer> wbuf = [device newBufferWithBytes:w_host
                                                 length:sizeof(w_host)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> xbuf = [device newBufferWithBytes:x_host
                                                 length:sizeof(x_host)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> ybuf = [device newBufferWithLength:O * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        if (!wbuf || !xbuf || !ybuf) { fprintf(stderr, "FAIL: buffer alloc\n"); return 1; }

        // ---- 5. encode + dispatch ----
        int32_t I_const = I, TPB_const = TPB;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:wbuf offset:0 atIndex:0];
        [enc setBuffer:xbuf offset:0 atIndex:1];
        [enc setBuffer:ybuf offset:0 atIndex:2];
        [enc setBytes:&I_const   length:sizeof(I_const)   atIndex:3];
        [enc setBytes:&TPB_const length:sizeof(TPB_const) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(O, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(TPB, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // ---- 6. read back + compare ----
        const float *y_gpu = (const float *)[ybuf contents];
        int pass = 1; float max_abs = 0.0f;
        printf("[probe] O=%d I=%d TPB=%d  per-output results:\n", O, I, TPB);
        for (int o = 0; o < O; o++) {
            float diff = fabsf(y_gpu[o] - y_ref[o]);
            if (diff > max_abs) max_abs = diff;
            int ok = diff < 1e-4f;
            printf("   y[%d]  gpu=%+.6f  cpu=%+.6f  diff=%.2e  %s\n",
                   o, y_gpu[o], y_ref[o], diff, ok ? "ok" : "MISMATCH");
            if (!ok) pass = 0;
        }
        printf("[probe] max abs diff = %.3e  ->  %s\n", max_abs, pass ? "PASS" : "FAIL");
        return pass ? 0 : 2;
    }
}
