/* K-quant baseline: per-row dot (what the table-less types use for every prefill column) and, where it
 * exists, the dedicated GEMM.   ./bench_kq [m n k reps]   (k multiple of 256)
 * Weights are random bytes with sane fp16 scales, activations are quantize_row_q8_K of random floats. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <time.h>
#include "quant.h"
#include "sgemm.h"
#include "type_dispatch.h"
#include <math.h>
void fp16_table_init(void);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static unsigned rs=777; static unsigned rnd(void){rs=rs*1664525u+1013904223u;return rs>>8;}
typedef struct { const char *name; size_t bsz; size_t d_off, dmin_off; float (*dot)(const void*,const void*,int); } kt_t;
static const kt_t KT[] = {
 {"q2_K", sizeof(block_q2_K), offsetof(block_q2_K,d), offsetof(block_q2_K,dmin), vec_dot_q2_K_q8_K},
 {"q3_K", sizeof(block_q3_K), offsetof(block_q3_K,d), (size_t)-1,               vec_dot_q3_K_q8_K},
 {"q4_K", sizeof(block_q4_K), offsetof(block_q4_K,d), offsetof(block_q4_K,dmin), vec_dot_q4_K_q8_K},
 {"q5_K", sizeof(block_q5_K), offsetof(block_q5_K,d), offsetof(block_q5_K,dm),   vec_dot_q5_K_q8_K},
 {"q6_K", sizeof(block_q6_K), offsetof(block_q6_K,d), (size_t)-1,               vec_dot_q6_K_q8_K},
 {"iq3_k",sizeof(block_iq3_k),offsetof(block_iq3_k,d),(size_t)-1,               vec_dot_iq3_k_q8_k},
 {"iq4_k",sizeof(block_iq4_k),offsetof(block_iq4_k,d),(size_t)-1,               vec_dot_iq4_k_q8_k},
};

static int check_shape(const kt_t*q,int gt,int m,int n,int k,int nth){
    int nb=k/256; size_t wrow=(size_t)nb*q->bsz; unsigned char*W=malloc((size_t)m*wrow); for(size_t i=0;i<(size_t)m*wrow;i++) W[i]=(unsigned char)rnd();
    for(size_t i=0;i<(size_t)m*nb;i++){ unsigned char*b=W+i*q->bsz; uint16_t d=fp32_to_fp16(0.001f+(rnd()&255)*1e-5f); memcpy(b+q->d_off,&d,2); if(q->dmin_off!=(size_t)-1){ uint16_t dm=fp32_to_fp16(0.0005f+(rnd()&255)*1e-5f); memcpy(b+q->dmin_off,&dm,2);} }
    block_q8_K *B=malloc((size_t)n*nb*sizeof(block_q8_K)); float*tmp=malloc(sizeof(float)*k);
    for(int c=0;c<n;c++){ for(int i=0;i<k;i++) tmp[i]=(rnd()&0xFFFF)/32768.0f-1.0f; quantize_row_q8_K(tmp,B+(size_t)c*nb,k); }
    float*C=malloc(sizeof(float)*(size_t)n*m); for(size_t i=0;i<(size_t)n*m;i++) C[i]=1e30f;   /* poison: every element must be written */
    if(gt<0){ for(int t=0;t<nth;t++) picolm_sgemm_d_q4k(m,n,nb,W,nb,B,nb,C,m,t,nth); }
    else { type_gemm_fn gf=typeinfo((gguf_type_t)gt)->fn_gemm; for(int t=0;t<nth;t++) gf(m,n,k,W,B,C,(size_t)m,t,nth); }
    double md=0,rm=0; int bad=0;
    for(int c=0;c<n;c++)for(int i=0;i<m;i++){ double ref=q->dot(W+(size_t)i*wrow,B+(size_t)c*nb,k); double dd=fabs(ref-C[(size_t)c*m+i]); if(!(dd==dd)||C[(size_t)c*m+i]>1e29f) bad=1; if(dd>md)md=dd; rm+=ref*ref; }
    rm=sqrt(rm/((double)m*n)+1e-30); if(md>1e-3*rm+1e-3) bad=1;
    if(bad) printf("FAIL %s m=%d n=%d k=%d nth=%d maxdiff=%.3g rms=%.3g\n",q->name,m,n,k,nth,md,rm);
    free(W);free(B);free(tmp);free(C); return bad; }
int main(int argc,char**argv){
    setenv("PICOLM_R8_MIN_BLOCKS","0",0);
    if(argc>1&&!strcmp(argv[1],"check")){ int bad=0,cnt=0; fp16_table_init();
        int ms[]={8,24,72,100,136}, ns[]={4,15,16,17,33,130}, ks[]={256,512,1280}, nts[]={1,3,5};
        { int idx[]={4,1,3,0,2}; int gts[]={GGUF_TYPE_Q6_K,GGUF_TYPE_Q3_K,GGUF_TYPE_Q5_K,GGUF_TYPE_Q2_K,-1};
          for(int ty=0;ty<5;ty++){ const kt_t*q=&KT[idx[ty]]; int gt=gts[ty]; if(gt>=0 && !typeinfo((gguf_type_t)gt)->fn_gemm) continue;
            for(unsigned a=0;a<5;a++)for(unsigned bb=0;bb<6;bb++)for(unsigned c=0;c<3;c++)for(unsigned d=0;d<3;d++){ bad|=check_shape(q,gt,ms[a],ns[bb],ks[c],nts[d]); cnt++; } } }
        printf("check: %d shapes, %s\n",cnt,bad?"FAIL":"all OK"); return bad; }
    int m=argc>1?atoi(argv[1]):2048, n=argc>2?atoi(argv[2]):256, k=argc>3?atoi(argv[3]):4096, reps=argc>4?atoi(argv[4]):2; int nb=k/256;
    fp16_table_init();
    block_q8_K *B=malloc((size_t)n*nb*sizeof(block_q8_K)); float *tmp=malloc(sizeof(float)*k);
    for(int c=0;c<n;c++){ for(int i=0;i<k;i++) tmp[i]=(rnd()&0xFFFF)/32768.0f-1.0f; quantize_row_q8_K(tmp,B+(size_t)c*nb,k); }
    for(unsigned t=0;t<sizeof KT/sizeof*KT;t++){ const kt_t*q=&KT[t]; size_t wrow=(size_t)nb*q->bsz;
        unsigned char*W=malloc((size_t)m*wrow); for(size_t i=0;i<(size_t)m*wrow;i++) W[i]=(unsigned char)rnd();
        for(size_t i=0;i<(size_t)m*nb;i++){ unsigned char*b=W+i*q->bsz; uint16_t d=fp32_to_fp16(0.001f+(rnd()&255)*1e-5f); memcpy(b+q->d_off,&d,2); if(q->dmin_off!=(size_t)-1){ uint16_t dm=fp32_to_fp16(0.0005f+(rnd()&255)*1e-5f); memcpy(b+q->dmin_off,&dm,2);} }
        double best=1e9; float sink=0;
        for(int r=0;r<reps;r++){ double t0=now(); for(int i=0;i<m;i++) sink+=q->dot(W+(size_t)i*wrow,B,k); double dt=now()-t0; if(dt<best)best=dt; }
        double ns=best*1e9/m; printf("%-6s dot: %7.1f ns/row (k=%d) = %6.1f GOPS prefill-per-column-equivalent, %5.2f GB/s weights  [%g]\n",q->name,ns,k,2.0*k/ns,(double)wrow/ns,sink);
        if(!strcmp(q->name,"q4_K")){ float*C=calloc((size_t)n*m,sizeof(float)); picolm_sgemm_d_q4k(m,n,nb,W,nb,B,nb,C,m,0,1); double b2=1e9; for(int r=0;r<reps;r++){ double t0=now(); picolm_sgemm_d_q4k(m,n,nb,W,nb,B,nb,C,m,0,1); double dt=now()-t0; if(dt<b2)b2=dt; }
            printf("%-6s GEMM (picolm_sgemm_d_q4k): %6.1f GOPS (1 thread, n=%d)\n",q->name,2.0*m*n*k/b2/1e9,n); free(C); }
        /* table GEMM (types that have one): speed + correctness against the per-column dot */
        { int gt = !strcmp(q->name,"q6_K") ? GGUF_TYPE_Q6_K : !strcmp(q->name,"q3_K") ? GGUF_TYPE_Q3_K : !strcmp(q->name,"q5_K") ? GGUF_TYPE_Q5_K : !strcmp(q->name,"q2_K") ? GGUF_TYPE_Q2_K : -1;
          if(gt>=0 && typeinfo((gguf_type_t)gt)->fn_gemm){ type_gemm_fn gf=typeinfo((gguf_type_t)gt)->fn_gemm; float*C=calloc((size_t)n*m,sizeof(float));
            gf(m,n,k,W,B,C,(size_t)m,0,1); double b2=1e9; for(int r=0;r<reps;r++){ double t0=now(); gf(m,n,k,W,B,C,(size_t)m,0,1); double dt=now()-t0; if(dt<b2)b2=dt; }
            double md=0,rm=0; for(int c=0;c<n&&c<6;c++)for(int i=0;i<m&&i<48;i++){ double ref=q->dot(W+(size_t)i*wrow,B+(size_t)c*nb,k); double dd=fabs(ref-C[(size_t)c*m+i]); if(dd>md)md=dd; rm+=ref*ref; }
            rm=sqrt(rm/(6.0*48)); printf("%-6s table GEMM: %6.1f GOPS (1 thread, n=%d)   max|gemm-dot|=%.3g (rms %.3g) %s\n",q->name,2.0*m*n*k/b2/1e9,n,md,rm,md>1e-3*rm+1e-3?"FAIL":"ok"); free(C); } }
        free(W); }
    return 0; }
