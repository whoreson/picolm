#ifndef BACKEND_GPU_MMA_CUH
#define BACKEND_GPU_MMA_CUH
/* Shared tensor-core primitives for the hand-written mma kernels (FA3 attention,
 * fp16 GEMM): ldmatrix, mma.m16n8k16 (fp16 in, fp32 accumulate), cp.async, packed
 * fp16 conversion/arithmetic, warp shuffles, and the XOR swizzle that makes
 * ldmatrix on 128/256/512-byte-stride tiles bank-conflict free.
 *
 * Every primitive has a CPU emulation in tests/emu_prims.h (selected with
 * -DPICOLM_FA3_EMU) so the kernels that use them can be executed and verified
 * without a GPU. The fa3_ prefix is historical: this was extracted from the FA3
 * header unchanged so the verified attention kernel keeps working. */
#if defined(PICOLM_FA3_EMU)
#  define FA3_ENABLED 1
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__)
#  define FA3_ENABLED 1
#else
#  define FA3_ENABLED 0
#endif

#ifndef PICOLM_FA3_EMU
#  define FA3_HD  __host__ __device__
#else
#  define FA3_HD
#endif

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
__device__ __forceinline__ float fa3_f16_to_f32(uint32_t bits) {
    float f; asm("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"((unsigned short)bits)); return f;
}
__device__ __forceinline__ float fa3_shfl_xor(float v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }
__device__ __forceinline__ int   fa3_any(int p) { return __any_sync(0xffffffffu, p); }
#  define FA3_DYN_SMEM_PTR(name) \
     extern __shared__ __align__(16) uint8_t name##_raw[]; uint8_t *name = name##_raw
#endif /* FA3_ENABLED && !EMU */

#if FA3_ENABLED
/* 16-byte chunk index of (row, chunk) in a [rows][CH] swizzled tile */
#define FA3_SW(row, chunk, CH) ((row) * (CH) + ((chunk) ^ ((row) & 7)))
#endif

#endif /* BACKEND_GPU_MMA_CUH */
