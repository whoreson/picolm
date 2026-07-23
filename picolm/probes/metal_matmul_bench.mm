// probes/metal_matmul_bench.mm
//
// Benchmark + correctness harness for the PicoLM Metal matmul kernels.
// Build:  make metal-bench      Run:  ./probes/metal_matmul_bench
//
// What it does (all in one binary, no model file needed):
//   1. Compiles the SAME MSL kernels the backend uses (copied verbatim from
//      backend_metal.mm KERNEL_SOURCE) + a new coalesced Q6_K variant
//      (mm_q6_K_c) under test.
//   2. CORRECTNESS: for small dims (1-2 blocks) diffs GPU output vs a CPU
//      reference (ported from quant.c dequantize_row_q4_K/q6_K). Catches any
//      dequant-index regression BEFORE it reaches the model.
//   3. THROUGHPUT: times each kernel at TinyLlama-relevant dims (decode S=1 and
//      prefill S=32/128) and reports ms/iter and effective weight bandwidth.
//   4. DISPATCH OVERHEAD: empty-kernel commit+waitUntilCompleted round-trip,
//      so we can tell whether kernels or host launch/sync is the bottleneck.
//
// NOTE: the kernel block below MUST mirror picolm/backend_metal.mm KERNEL_SOURCE.
// If you change the kernels there, paste the same change here (search MIRROR).

#include <metal/Metal.h>
#include <Foundation/Foundation.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>

#define TPB 256
#define NW  (TPB/32)

enum { T_F32 = 0, T_F16, T_Q4_0, T_Q8_0, T_Q4K, T_Q5K, T_Q6K };

// ---------------- host fp16 <-> fp32 (matches quant.c software path) ----------------
static float fp16tof32(uint16_t h) {
    uint32_t sign=(h>>15)&1, exp=(h>>10)&0x1f, mant=h&0x3ff, f;
    if (exp==0) { if(mant==0){f=sign<<31;} else {exp=1; while(!(mant&0x400)){mant<<=1;exp--;} mant&=0x3ff; f=(sign<<31)|((exp+127-15)<<23)|(mant<<13);} }
    else { f=(sign<<31)|((exp+127-15)<<23)|(mant<<13); }
    float o; memcpy(&o,&f,4); return o;
}
static uint16_t fp32tofp16(float f) {
    uint32_t x; memcpy(&x,&f,4);
    uint32_t sign=(x>>16)&0x8000; int32_t exp=(int32_t)((x>>23)&0xff)-127+15; uint32_t mant=x&0x7fffff;
    if (exp<=0){ if(exp<-10) return sign; mant|=0x800000; uint32_t sh=(uint32_t)(14-exp); uint32_t mo=mant>>sh; if((mant>>(sh-1))&1)mo++; return sign|mo; }
    else if (exp>=0x1f) return sign|0x7c00;
    uint32_t o=sign|((uint32_t)exp<<10)|(mant>>13); if((mant>>12)&1)o++; return o;
}
static void put_fp16(uint8_t* p, float v){ uint16_t h=fp32tofp16(v); p[0]=h&0xFF; p[1]=(h>>8)&0xFF; }

// ---------------- CPU dequant references (ported from quant.c) ----------------
static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* sc, uint8_t* mn) {
    if (j<4){ *sc=q[j]&63; *mn=q[j+4]&63; }
    else { *sc=(q[j+4]&0xF)|((q[j-4]>>6)<<4); *mn=(q[j+4]>>4)|((q[j]>>6)<<4); }
}
static void deq_q4_K(const uint8_t* src, float* dst, int n) {
    int nb=n/256;
    for (int i=0;i<nb;i++){ const uint8_t* b=src+i*144;
        float d=fp16tof32(b[0]|(b[1]<<8)), dmin=fp16tof32(b[2]|(b[3]<<8));
        const uint8_t* sc=b+4; const uint8_t* q=b+16; float* y=dst+i*256; int is=0;
        for (int j=0;j<4;j++){ uint8_t s,m; get_scale_min_k4(is,sc,&s,&m); float d1=d*s,m1=dmin*m;
            get_scale_min_k4(is+1,sc,&s,&m); float d2=d*s,m2=dmin*m;
            for(int l=0;l<32;l++) y[l]    =d1*(float)(q[l]&0xF)-m1;
            for(int l=0;l<32;l++) y[l+32] =d2*(float)(q[l]>>4)-m2;
            y+=64; q+=32; is+=2; }
    }
}
static void deq_q6_K(const uint8_t* src, float* dst, int n) {
    int nb=n/256;
    for (int i=0;i<nb;i++){ const uint8_t* blk=src+i*210;
        float d=fp16tof32(blk[208]|(blk[209]<<8));
        const uint8_t* ql=blk; const uint8_t* qh=blk+128; const int8_t* sc=(const int8_t*)(blk+192);
        float* y=dst+i*256;
        for (int chunk=0;chunk<256;chunk+=128){ int is=chunk/16;
            for (int l=0;l<32;l++){
                int q1=(int)((ql[l]&0xF)|(((qh[l]>>0)&3)<<4))-32;
                int q2=(int)((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
                int q3=(int)((ql[l]>>4)|(((qh[l]>>4)&3)<<4))-32;
                int q4=(int)((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32;
                int is_l=is+(l/16);
                y[l]   =d*(float)sc[is_l+0]*(float)q1; y[l+32]=d*(float)sc[is_l+2]*(float)q2;
                y[l+64]=d*(float)sc[is_l+4]*(float)q3; y[l+96]=d*(float)sc[is_l+6]*(float)q4;
            }
            y+=128; ql+=64; qh+=32;
        }
    }
}
// dequant one weight row, dot with xrow -> scalar
static float cpu_dot(const uint8_t* wrow, const float* xrow, int I, int qtype, float* tmp) {
    if (qtype==T_Q4K) deq_q4_K(wrow,tmp,I); else deq_q6_K(wrow,tmp,I);
    double acc=0; for (int i=0;i<I;i++) acc+=(double)tmp[i]*(double)xrow[i]; return (float)acc;
}

// ---------------- random quantized weight block generation ----------------
static uint8_t rb(void){ return (uint8_t)(rand()&0xFF); }
// Fill O rows of 'row_bytes' each with random-but-valid blocks (controlled fp16 scales).
static void gen_weights(uint8_t* w, int O, size_t row_bytes, int qtype) {
    int bsz = (qtype==T_Q4K)?144:210;   // bytes per 256-value block
    for (int o=0;o<O;o++){
        uint8_t* row = w + (size_t)o*row_bytes;
        int nblocks = (int)(row_bytes/bsz);
        for (int bi=0;bi<nblocks;bi++){
            uint8_t* blk = row + bi*bsz;
            for (int i=0;i<bsz;i++) blk[i]=rb();
            if (qtype==T_Q4K){ put_fp16(blk+0, 0.02f+0.01f*(bi&3)); put_fp16(blk+2, 0.01f*(bi&1)); }
            else             { put_fp16(blk+208, 0.02f+0.01f*(bi&3)); }
        }
    }
}

// ================= MSL KERNEL SOURCE (MIRROR of backend_metal.mm) =================
static NSString* KS = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
inline float f16tof32(uint16_t h){ return (float)as_type<half>(h); }\n\
#define TPB 256\n\
#define NW (TPB/32)\n\
inline float reduce_all(float acc, threadgroup float* wb, uint tid){\n\
    uint warp=tid>>5, lane=tid&31; acc=simd_sum(acc);\n\
    if(lane==0) wb[warp]=acc; threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    float v=(warp==0&&lane<NW)?wb[lane]:0.0f; return simd_sum(v);\n\
}\n\
inline void k4_unpack(device const uint8_t* s,int is,thread float& dsc,thread float& dmn,float d,float dmin){\n\
    uint8_t sc,mn; if(is<4){sc=s[is]&63;mn=s[is+4]&63;}else{sc=(s[is+4]&0xF)|((s[is-4]>>6)<<4);mn=(s[is+4]>>4)|((s[is]>>6)<<4);}\n\
    dsc=d*(float)sc; dmn=dmin*(float)mn;\n\
}\n\
/* ---- Q4_K (current backend kernel, verbatim) ---- */\n\
kernel void mm_q4_K(device const uint8_t* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]],\n\
   constant int& I [[buffer(3)]],constant int& S [[buffer(4)]],constant int& O [[buffer(5)]],\n\
   uint tid [[thread_index_in_threadgroup]],uint2 gp [[threadgroup_position_in_grid]]){\n\
    int o=(int)gp.x,s=(int)gp.y; int nblocks=I/256;\n\
    device const uint8_t* row=w+(unsigned long)o*(unsigned long)(nblocks*144);\n\
    uint warp=tid>>5,lane=tid&31,g=lane/8,within=(lane&7)*4; float acc=0.0f;\n\
    for(int bi=(int)warp;bi<nblocks;bi+=NW){ device const uint8_t* blk=row+(unsigned long)(bi*144);\n\
        float d=f16tof32((uint16_t)blk[0]|((uint16_t)blk[1]<<8)); float dmin=f16tof32((uint16_t)blk[2]|((uint16_t)blk[3]<<8));\n\
        device const uint8_t* scales=blk+4; device const uint8_t* qs=blk+16;\n\
        float d_lo,m_lo,d_hi,m_hi; k4_unpack(scales,2*(int)g,d_lo,m_lo,d,dmin); k4_unpack(scales,2*(int)g+1,d_hi,m_hi,d,dmin);\n\
        uint qw=*(device const uint*)(qs+lane*4); uint base=(uint)(bi*256)+g*64+within;\n\
        for(int k=0;k<4;k++){ uint bv=(qw>>(k*8))&0xFF; float lo=(float)(bv&0xF),hi=(float)(bv>>4);\n\
            acc+=x[(unsigned long)s*I+base+k]*(d_lo*lo-m_lo); acc+=x[(unsigned long)s*I+base+k+32]*(d_hi*hi-m_hi); }\n\
    }\n\
    threadgroup float wb[NW]; float total=reduce_all(acc,wb,tid); if(tid==0) y[(unsigned long)s*O+o]=total;\n\
}\n\
/* ---- Q6_K (current backend kernel, verbatim) ---- */\n\
kernel void mm_q6_K(device const uint8_t* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]],\n\
   constant int& I [[buffer(3)]],constant int& S [[buffer(4)]],constant int& O [[buffer(5)]],\n\
   uint tid [[thread_index_in_threadgroup]],uint2 gp [[threadgroup_position_in_grid]]){\n\
    int o=(int)gp.x,s=(int)gp.y; int nblocks=I/256;\n\
    device const uint8_t* row=w+(unsigned long)o*(unsigned long)(nblocks*210);\n\
    float acc=0.0f;\n\
    for(int bi=(int)tid;bi<nblocks;bi+=TPB){ device const uint8_t* blk=row+(unsigned long)(bi*210);\n\
        float d=f16tof32((uint16_t)blk[208]|((uint16_t)blk[209]<<8));\n\
        device const uint8_t* ql=blk; device const uint8_t* qh=blk+128; device const int8_t* sc=(device const int8_t*)(blk+192);\n\
        for(int j=0;j<256;j++){ int chunk=j/128,within=j%128,sub=within/32,l=within%32;\n\
            int ql_idx=(sub==1||sub==3)?(l+32):l; uint8_t qlb=ql[chunk*64+ql_idx]; uint8_t qhb=qh[chunk*32+l];\n\
            int qh_shift=(sub==0)?0:(sub==1)?2:(sub==2)?4:6;\n\
            int qraw=((sub<2)?(int)(qlb&0xF):(int)(qlb>>4))|(int)(((qhb>>qh_shift)&3)<<4);\n\
            int sc_idx=chunk*8+l/16+2*sub; acc+=x[(unsigned long)s*I+bi*256+j]*(d*(float)sc[sc_idx]*(float)(qraw-32));\n\
        }\n\
    }\n\
    threadgroup float wb[NW]; float total=reduce_all(acc,wb,tid); if(tid==0) y[(unsigned long)s*O+o]=total;\n\
}\n\
/* ---- Q6_K COALESCED variant under test (mm_q6_K_c) ----\n\
 * One simdgroup (32 lanes) per 256-value block; each lane owns 8 values.\n\
 * ql[128] read in 4 coalesced 32B passes (lane t owns bytes t,32+t,64+t,96+t);\n\
 * qh[64] in 2 coalesced passes; scales read direct (cached). Indexing derived\n\
 * from quant.c dequantize_row_q6_K and hand-traced for values j=0,32,96,128. */\n\
kernel void mm_q6_K_c(device const uint8_t* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]],\n\
   constant int& I [[buffer(3)]],constant int& S [[buffer(4)]],constant int& O [[buffer(5)]],\n\
   uint tid [[thread_index_in_threadgroup]],uint2 gp [[threadgroup_position_in_grid]]){\n\
    int o=(int)gp.x,s=(int)gp.y; int nblocks=I/256;\n\
    device const uint8_t* row=w+(unsigned long)o*(unsigned long)(nblocks*210);\n\
    uint warp=tid>>5,lane=tid&31; float acc=0.0f;\n\
    for(int bi=(int)warp;bi<nblocks;bi+=NW){ device const uint8_t* blk=row+(unsigned long)(bi*210);\n\
        float d=f16tof32((uint16_t)blk[208]|((uint16_t)blk[209]<<8));\n\
        device const uint8_t* ql=blk; device const uint8_t* qh=blk+128; device const int8_t* sc=(device const int8_t*)(blk+192);\n\
        for(int k=0;k<4;k++){ uint b=(uint)(k*32)+lane; uint chk=b>>6,grp=(b&63)>>5,l=b&31;\n\
            uint qlb=ql[b]; uint qhb=qh[chk*32+l];\n\
            int qr0=(int)(qlb&0xF)|(int)(((qhb>>(2u*grp))&3u)<<4);   int si0=(int)(chk*8+l/16+2u*grp);     uint j0=chk*128+grp*32+l;\n\
            acc+=x[(unsigned long)s*I+(unsigned long)bi*256+j0]*(d*(float)sc[si0]*(float)(qr0-32));\n\
            int qr1=(int)(qlb>>4)|(int)(((qhb>>(2u*(grp+2u)))&3u)<<4); int si1=(int)(chk*8+l/16+2u*(grp+2u)); uint j1=chk*128+(grp+2u)*32+l;\n\
            acc+=x[(unsigned long)s*I+(unsigned long)bi*256+j1]*(d*(float)sc[si1]*(float)(qr1-32));\n\
        }\n\
    }\n\
    threadgroup float wb[NW]; float total=reduce_all(acc,wb,tid); if(tid==0) y[(unsigned long)s*O+o]=total;\n\
}\n\
/* ---- MULTI-OUTPUT decode GEMV (1 warp per output; no cross-warp reduce) ----\n\
 * grid=(O/8,S,1), TG=256 (8 warps). warp w computes output o=gp.x*8+w over ALL\n\
 * blocks; simd_sum within the warp only (no threadgroup barrier / shared mem).\n\
 * Cuts the per-output reduction and packs 8 outputs per threadgroup -> better\n\
 * effective bandwidth, esp. on small-I rows (lm_head). */\n\
kernel void mm_q4_K_mo(device const uint8_t* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]],\n\
   constant int& I [[buffer(3)]],constant int& S [[buffer(4)]],constant int& O [[buffer(5)]],\n\
   uint tid [[thread_index_in_threadgroup]],uint2 gp [[threadgroup_position_in_grid]]){\n\
    int s=(int)gp.y; int o=(int)gp.x*8+(int)(tid>>5); if(o>=O) return;\n\
    int nblocks=I/256; device const uint8_t* row=w+(unsigned long)o*(unsigned long)(nblocks*144);\n\
    uint lane=tid&31,g=lane/8,within=(lane&7)*4; float acc=0.0f;\n\
    for(int bi=0;bi<nblocks;bi++){ device const uint8_t* blk=row+(unsigned long)(bi*144);\n\
        float d=f16tof32((uint16_t)blk[0]|((uint16_t)blk[1]<<8)); float dmin=f16tof32((uint16_t)blk[2]|((uint16_t)blk[3]<<8));\n\
        device const uint8_t* scales=blk+4; device const uint8_t* qs=blk+16;\n\
        float d_lo,m_lo,d_hi,m_hi; k4_unpack(scales,2*(int)g,d_lo,m_lo,d,dmin); k4_unpack(scales,2*(int)g+1,d_hi,m_hi,d,dmin);\n\
        uint qw=*(device const uint*)(qs+lane*4); uint base=(uint)(bi*256)+g*64+within;\n\
        for(int k=0;k<4;k++){ uint bv=(qw>>(k*8))&0xFF; float lo=(float)(bv&0xF),hi=(float)(bv>>4);\n\
            acc+=x[(unsigned long)s*I+base+k]*(d_lo*lo-m_lo); acc+=x[(unsigned long)s*I+base+k+32]*(d_hi*hi-m_hi); }\n\
    }\n\
    float total=simd_sum(acc); if(lane==0) y[(unsigned long)s*O+o]=total;\n\
}\n\
kernel void mm_q6_K_mo(device const uint8_t* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]],\n\
   constant int& I [[buffer(3)]],constant int& S [[buffer(4)]],constant int& O [[buffer(5)]],\n\
   uint tid [[thread_index_in_threadgroup]],uint2 gp [[threadgroup_position_in_grid]]){\n\
    int s=(int)gp.y; int o=(int)gp.x*8+(int)(tid>>5); if(o>=O) return;\n\
    int nblocks=I/256; device const uint8_t* row=w+(unsigned long)o*(unsigned long)(nblocks*210);\n\
    uint lane=tid&31; float acc=0.0f;\n\
    for(int bi=0;bi<nblocks;bi++){ device const uint8_t* blk=row+(unsigned long)(bi*210);\n\
        float d=f16tof32((uint16_t)blk[208]|((uint16_t)blk[209]<<8));\n\
        device const uint8_t* ql=blk; device const uint8_t* qh=blk+128; device const int8_t* sc=(device const int8_t*)(blk+192);\n\
        for(int k=0;k<4;k++){ uint b=(uint)(k*32)+lane; uint chk=b>>6,grp=(b&63)>>5,l=b&31;\n\
            uint qlb=ql[b]; uint qhb=qh[chk*32+l];\n\
            int qr0=(int)(qlb&0xF)|(int)(((qhb>>(2u*grp))&3u)<<4); int si0=(int)(chk*8+l/16+2u*grp); uint j0=chk*128+grp*32+l;\n\
            acc+=x[(unsigned long)s*I+(unsigned long)bi*256+j0]*(d*(float)sc[si0]*(float)(qr0-32));\n\
            int qr1=(int)(qlb>>4)|(int)(((qhb>>(2u*(grp+2u)))&3u)<<4); int si1=(int)(chk*8+l/16+2u*(grp+2u)); uint j1=chk*128+(grp+2u)*32+l;\n\
            acc+=x[(unsigned long)s*I+(unsigned long)bi*256+j1]*(d*(float)sc[si1]*(float)(qr1-32));\n\
        }\n\
    }\n\
    float total=simd_sum(acc); if(lane==0) y[(unsigned long)s*O+o]=total;\n\
}\n\
/* trivial kernel for dispatch-overhead probe */\n\
kernel void noop_k(device float* y [[buffer(0)]],uint gid [[thread_position_in_threadgroup]]){ if(gid==0u) y[0]=0.0f; }\n\
";

// ---------------- timing ----------------
static double g_tick2ns=0;
static double now_s(void){ if(!g_tick2ns){ mach_timebase_info_data_t tb; mach_timebase_info(&tb); g_tick2ns=(double)tb.numer/(double)tb.denom; } return (double)mach_absolute_time()*g_tick2ns/1e9; }

static id<MTLCommandBuffer> enc_matmul(id<MTLCommandQueue> q, id<MTLComputePipelineState> ps,
        id<MTLBuffer> w, id<MTLBuffer> x, id<MTLBuffer> y, int I,int S,int O,int mo){
    NSUInteger gx = (mo>1) ? (NSUInteger)((O+mo-1)/mo) : (NSUInteger)O;
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:ps];
    [e setBuffer:w offset:0 atIndex:0]; [e setBuffer:x offset:0 atIndex:1]; [e setBuffer:y offset:0 atIndex:2];
    int32_t cI=I,cS=S,cO=O; [e setBytes:&cI length:4 atIndex:3]; [e setBytes:&cS length:4 atIndex:4]; [e setBytes:&cO length:4 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(gx,(NSUInteger)S,1) threadsPerThreadgroup:MTLSizeMake(TPB,1,1)];
    [e endEncoding]; return cb;
}
static double bench(id<MTLCommandQueue> q, id<MTLComputePipelineState> ps,
        id<MTLBuffer> w,id<MTLBuffer> x,id<MTLBuffer> y,int I,int S,int O,int mo,int iters){
    for(int i=0;i<5;i++){ @autoreleasepool{ id<MTLCommandBuffer> cb=enc_matmul(q,ps,w,x,y,I,S,O,mo); [cb commit]; [cb waitUntilCompleted]; } }
    double t0=now_s();
    for(int i=0;i<iters;i++){ @autoreleasepool{ id<MTLCommandBuffer> cb=enc_matmul(q,ps,w,x,y,I,S,O,mo); [cb commit]; [cb waitUntilCompleted]; } }
    return (now_s()-t0)/(double)iters;
}
static double bench_noop(id<MTLCommandQueue> q, id<MTLComputePipelineState> ps, id<MTLBuffer> y, int iters){
    for(int i=0;i<10;i++){ @autoreleasepool{ id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:ps]; [e setBuffer:y offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; } }
    double t0=now_s();
    for(int i=0;i<iters;i++){ @autoreleasepool{ id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:ps]; [e setBuffer:y offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; } }
    return (now_s()-t0)/(double)iters;
}

static size_t row_bytes_for(int qtype,int I){
    if(qtype==T_Q4K) return (size_t)(I/256)*144;
    if(qtype==T_Q6K) return (size_t)(I/256)*210;
    return 0;
}

int main(void){
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
        if(!dev){ fprintf(stderr,"FAIL: no Metal device\n"); return 1; }
        id<MTLCommandQueue> q=[dev newCommandQueue];
        NSError* err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithSource:KS options:nil error:&err];
        if(!lib){ fprintf(stderr,"FAIL compile: %s\n",[[err localizedDescription] UTF8String]); return 1; }
        id<MTLComputePipelineState> ps_q4K  =[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_q4_K"]    error:&err];
        id<MTLComputePipelineState> ps_q6K  =[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_q6_K"]    error:nil];
        id<MTLComputePipelineState> ps_q6Kc =[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_q6_K_c"]  error:nil];
        id<MTLComputePipelineState> ps_q4Kmo=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_q4_K_mo"] error:nil];
        id<MTLComputePipelineState> ps_q6Kmo=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_q6_K_mo"] error:nil];
        id<MTLComputePipelineState> ps_noop =[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"noop_k"]     error:nil];
        if(!ps_q4K||!ps_q6K||!ps_q6Kc||!ps_q4Kmo||!ps_q6Kmo||!ps_noop){ fprintf(stderr,"FAIL: pipeline state\n"); return 1; }

        NSLog(@"[bench] device: %@", dev.name);

        // ---------- 1. correctness (small dims) ----------
        printf("\n=== CORRECTNESS (GPU vs CPU; max abs diff) ===\n");
        printf("%-12s type  I    O    S   maxdiff   verdict\n","kernel");
        struct { const char* name; id<MTLComputePipelineState> ps; int qtype; int mo; } cks[]={
            {"mm_q4_K",   ps_q4K,   T_Q4K, 1}, {"mm_q6_K",   ps_q6K,   T_Q6K, 1}, {"mm_q6_K_c", ps_q6Kc,  T_Q6K, 1},
            {"mm_q4_K_mo",ps_q4Kmo, T_Q4K, 8}, {"mm_q6_K_mo",ps_q6Kmo, T_Q6K, 8},
        };
        int cdims[][3]={{256,32,1},{512,48,4}}; // (I,O,S)
        float* tmp=(float*)malloc(4096*sizeof(float));
        for(int c=0;c<3;c++){
            for(int d=0;d<2;d++){
                int I=cdims[d][0],O=cdims[d][1],S=cdims[d][2]; int qt=cks[c].qtype;
                size_t rb=row_bytes_for(qt,I); size_t wb=O*rb, xb=(size_t)S*I*4, yb=(size_t)S*O*4;
                uint8_t* wh=(uint8_t*)malloc(wb); gen_weights(wh,O,rb,qt);
                float* xh=(float*)malloc(xb); for(size_t i=0;i<xb/4;i++) xh[i]=((float)(rand()%2001)-1000.0f)*1e-3f;
                float* yref=(float*)malloc(yb);
                for(int s=0;s<S;s++) for(int o=0;o<O;o++)
                    yref[(size_t)s*O+o]=cpu_dot(wh+(size_t)o*rb, xh+(size_t)s*I, I, qt, tmp);
                id<MTLBuffer> wbuf=[dev newBufferWithBytes:wh length:wb options:MTLResourceStorageModeShared];
                id<MTLBuffer> xbuf=[dev newBufferWithBytes:xh length:xb options:MTLResourceStorageModeShared];
                id<MTLBuffer> ybuf=[dev newBufferWithLength:yb options:MTLResourceStorageModeShared];
                @autoreleasepool{ id<MTLCommandBuffer> cb=enc_matmul(q,cks[c].ps,wbuf,xbuf,ybuf,I,S,O,cks[c].mo); [cb commit]; [cb waitUntilCompleted]; }
                const float* yg=(const float*)[ybuf contents]; float mx=0;
                for(size_t i=0;i<yb/4;i++){ float df=fabsf(yg[i]-yref[i]); if(df>mx)mx=df; }
                printf("%-12s Q%d_K  %-4d %-4d %-3d %.3e  %s\n", cks[c].name, qt==T_Q4K?4:6, I,O,S, mx, mx<1e-3f?"OK":"MISMATCH");
                free(wh); free(xh); free(yref);
            }
        }
        free(tmp);

        // ---------- 2. dispatch overhead ----------
        printf("\n=== DISPATCH OVERHEAD (noop kernel: encode+commit+waitUntilCompleted) ===\n");
        id<MTLBuffer> y0=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        double us=bench_noop(q,ps_noop,y0,2000)*1e6;
        printf("   per dispatch: %.2f us   (decode does ~3 GPU dispatches/layer x 22 = ~66/token)\n", us);
        printf("   => overhead floor @ 66 dispatches/token: %.2f ms/token  (~%.0f tok/s ceiling)\n", us*66/1000.0, 1000.0/(us*66/1000.0));

        // ---------- 3. throughput ----------
        printf("\n=== THROUGHPUT (ms/iter and effective weight bandwidth) ===\n");
        printf("%-10s type  I     O      S    wMB    ms/iter   GB/s\n","kernel");
        struct { const char* name; id<MTLComputePipelineState> ps; int qtype; int mo; } tks[]={
            {"mm_q4_K",   ps_q4K,   T_Q4K, 1},
            {"mm_q6_K_c", ps_q6Kc,  T_Q6K, 1},
            {"mm_q4_K_mo",ps_q4Kmo, T_Q4K, 8},
            {"mm_q6_K_mo",ps_q6Kmo, T_Q6K, 8},
        };
        int tdims[][3]={{2048,2048,1},{2048,5632,1},{5632,2048,1},{2048,32000,1},
                        {2048,2048,32},{2048,5632,32},{2048,2048,128}};
        for(int t=0;t<3;t++){
            int qt=tks[t].qtype;
            for(int d=0;d<7;d++){
                int I=tdims[d][0],O=tdims[d][1],S=tdims[d][2];
                size_t rb=row_bytes_for(qt,I); size_t wb=O*rb, xb=(size_t)S*I*4, yb=(size_t)S*O*4;
                uint8_t* wh=(uint8_t*)malloc(wb); gen_weights(wh,O,rb,qt);
                float* xh=(float*)malloc(xb); for(size_t i=0;i<xb/4;i++) xh[i]=((float)(rand()%2001)-1000.0f)*1e-3f;
                id<MTLBuffer> wbuf=[dev newBufferWithBytesNoCopy:wh length:((wb+4095)&~(size_t)4095)
                                options:MTLResourceStorageModeShared deallocator:^(void* p,NSUInteger l){(void)p;(void)l;}];
                if(!wbuf) wbuf=[dev newBufferWithBytes:wh length:wb options:MTLResourceStorageModeShared];
                id<MTLBuffer> xbuf=[dev newBufferWithBytes:xh length:xb options:MTLResourceStorageModeShared];
                id<MTLBuffer> ybuf=[dev newBufferWithLength:yb options:MTLResourceStorageModeShared];
                int iters = (S>=128)?30 : (S>=32?100:300);
                double sec=bench(q,tks[t].ps,wbuf,xbuf,ybuf,I,S,O,tks[t].mo,iters);
                double wmb=(double)wb/1e6, gbs=(double)wb/sec/1e9;
                printf("%-10s Q%d_K  %-5d %-6d %-3d  %5.1f  %.4f   %6.1f\n", tks[t].name, qt==T_Q4K?4:6, I,O,S, wmb, sec*1e3, gbs);
                free(wh); free(xh);
            }
        }
        printf("\nApple M2 Pro ~200 GB/s unified-memory bandwidth is the roof for decode (S=1).\n");
        return 0;
    }
}
