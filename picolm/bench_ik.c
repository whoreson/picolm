/* Timing of table GEMMs for the ik-style / R4 K-quant types (random weights with sane scales).
 *   ./bench_ik [m n k reps]      (k multiple of 256, m multiple of 8)  */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <time.h>
#include "quant.h"
#include "type_dispatch.h"
#include <math.h>
void fp16_table_init(void);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static unsigned rs=4242; static unsigned rnd(void){rs=rs*1664525u+1013904223u;return rs>>8;}
typedef struct { const char *name; int gt; size_t bsz; int rows; int nd; size_t d_off; } it_t;   /* rows = rows per block, nd = number of fp16 scales to set (from d_off) */
static const it_t IT[] = {
 {"iq2_k",    GGUF_TYPE_IQ2_K,    sizeof(block_iq2_k),    1, 1, offsetof(block_iq2_k,d)},
 {"iq3_k",    GGUF_TYPE_IQ3_K,    sizeof(block_iq3_k),    1, 1, offsetof(block_iq3_k,d)},
 {"iq4_k",    GGUF_TYPE_IQ4_K,    sizeof(block_iq4_k),    1, 1, offsetof(block_iq4_k,d)},
 {"iq2_k_r4", GGUF_TYPE_IQ2_K_R4, sizeof(block_iq2_k_r4), 4, 4, offsetof(block_iq2_k_r4,d)},
 {"iq3_k_r4", GGUF_TYPE_IQ3_K_R4, sizeof(block_iq3_k_r4), 4, 4, offsetof(block_iq3_k_r4,d)},
 {"iq4_k_r4", GGUF_TYPE_IQ4_K_R4, sizeof(block_iq4_k_r4), 4, 4, offsetof(block_iq4_k_r4,d)},
 {"q4_k_r4",  GGUF_TYPE_Q4_K_R4,  sizeof(block_q4_k_r4),  4, 8, offsetof(block_q4_k_r4,d)},
 {"q6_K_r4",  GGUF_TYPE_Q6_K_R4,  sizeof(block_q6_K_R4),  4, 4, offsetof(block_q6_K_R4,d)},
};

/* correctness: table GEMM vs (float dequant of the repo's reference) . (Q8_K activations), poisoned output */
typedef void (*deq_fn)(const void *blocks, float *dst, int n, int row);
static void deq_q4_k_r4(const void *x, float *dst, int n, int row){ dequantize_row_q4_k_r4_single((const block_q4_k_r4*)x,dst,n,row); }
static void deq_q6_k_r4(const void *x, float *dst, int n, int row){ float *tmp=malloc(sizeof(float)*4*n); dequantize_row_q6_K_R4(x,tmp,n); memcpy(dst,tmp+(size_t)row*n,sizeof(float)*n); free(tmp); }
static void deq_iq4_k(const void *x, float *dst, int n, int row){ (void)row; dequantize_row_iq4_k(x,dst,n); }
static void deq_iq4_k_r4(const void *x, float *dst, int n, int row){ dequantize_row_iq4_k_r4_single((const block_iq4_k_r4*)x,dst,n,row); }
static int check_one(const it_t*q, deq_fn deq, int m, int n, int k, int nth){
    int nb=k/256; size_t groups=(size_t)m/q->rows, wbytes=groups*nb*q->bsz; unsigned char*W=malloc(wbytes);
    for(size_t i=0;i<wbytes;i++) W[i]=(unsigned char)rnd();
    for(size_t i=0;i<groups*nb;i++){ unsigned char*b=W+i*q->bsz+q->d_off; for(int j=0;j<q->nd;j++){ uint16_t d=fp32_to_fp16(0.001f+(rnd()&255)*1e-5f); memcpy(b+2*j,&d,2);} }
    block_q8_K *B=malloc((size_t)n*nb*sizeof(block_q8_K)); float*tmp=malloc(sizeof(float)*k);
    for(int c=0;c<n;c++){ for(int i=0;i<k;i++) tmp[i]=(rnd()&0xFFFF)/32768.0f-1.0f; quantize_row_q8_K(tmp,B+(size_t)c*nb,k); }
    float*C=malloc(sizeof(float)*(size_t)n*m); for(size_t i=0;i<(size_t)n*m;i++) C[i]=1e30f;
    type_gemm_fn gf=typeinfo((gguf_type_t)q->gt)->fn_gemm; for(int t=0;t<nth;t++) gf(m,n,k,W,B,C,(size_t)m,t,nth);
    float*row=malloc(sizeof(float)*k); double md=0,rm=0; int bad=0;
    for(int r=0;r<m;r++){ deq(W+(size_t)(r/q->rows)*nb*q->bsz,row,k,r%q->rows);
        for(int c=0;c<n;c++){ double ref=0; for(int q8=0;q8<nb;q8++){ const block_q8_K*y=B+(size_t)c*nb+q8; double s=0; for(int i=0;i<256;i++) s+=(double)row[q8*256+i]*y->qs[i]; ref+=s*y->d; }
            double got=C[(size_t)c*m+r]; double dd=fabs(ref-got); if(!(dd==dd)||got>1e29f) bad=1; if(dd>md)md=dd; rm+=ref*ref; } }
    rm=sqrt(rm/((double)m*n)+1e-30); if(md>2e-3*rm+1e-3) bad=1;
    if(bad) printf("FAIL %s m=%d n=%d k=%d nth=%d maxdiff=%.3g rms=%.3g\n",q->name,m,n,k,nth,md,rm);
    free(W);free(B);free(tmp);free(C);free(row); return bad; }
int main(int argc,char**argv){
    setenv("PICOLM_R8_MIN_BLOCKS","0",0);
    if(argc>1&&!strcmp(argv[1],"check")){ int bad=0,cnt=0; fp16_table_init();
        int ms[]={8,16,24,72,136,12,20}, ns[]={4,15,16,17,33,130}, ks[]={256,512,1280}, nts[]={1,3,5};
        struct { const char*name; deq_fn deq; } T[]={{"q4_k_r4",deq_q4_k_r4},{"q6_K_r4",deq_q6_k_r4},{"iq4_k",deq_iq4_k},{"iq4_k_r4",deq_iq4_k_r4}};
        for(unsigned ty=0;ty<sizeof T/sizeof*T;ty++){ const it_t*q=NULL; for(unsigned i=0;i<sizeof IT/sizeof*IT;i++) if(!strcmp(IT[i].name,T[ty].name)) q=&IT[i];
            for(unsigned a=0;a<7;a++)for(unsigned bb=0;bb<6;bb++)for(unsigned c=0;c<3;c++)for(unsigned d=0;d<3;d++){ bad|=check_one(q,T[ty].deq,ms[a],ns[bb],ks[c],nts[d]); cnt++; } }
        printf("check: %d shapes, %s\n",cnt,bad?"FAIL":"all OK"); return bad; }
    int m=argc>1?atoi(argv[1]):1024, n=argc>2?atoi(argv[2]):256, k=argc>3?atoi(argv[3]):4096, reps=argc>4?atoi(argv[4]):2; int nb=k/256;
    fp16_table_init();
    block_q8_K *B=malloc((size_t)n*nb*sizeof(block_q8_K)); float *tmp=malloc(sizeof(float)*k);
    for(int c=0;c<n;c++){ for(int i=0;i<k;i++) tmp[i]=(rnd()&0xFFFF)/32768.0f-1.0f; quantize_row_q8_K(tmp,B+(size_t)c*nb,k); }
    for(unsigned t=0;t<sizeof IT/sizeof*IT;t++){ const it_t*q=&IT[t]; type_gemm_fn gf=typeinfo((gguf_type_t)q->gt)->fn_gemm; if(!gf){ printf("%-9s no table GEMM\n",q->name); continue; }
        size_t groups=(size_t)m/q->rows, wbytes=groups*nb*q->bsz; unsigned char*W=malloc(wbytes);
        for(size_t i=0;i<wbytes;i++) W[i]=(unsigned char)rnd();
        for(size_t i=0;i<groups*nb;i++){ unsigned char*b=W+i*q->bsz+q->d_off; for(int j=0;j<q->nd;j++){ uint16_t d=fp32_to_fp16(0.001f+(rnd()&255)*1e-5f); memcpy(b+2*j,&d,2);} }
        float*C=calloc((size_t)n*m,sizeof(float)); gf(m,n,k,W,B,C,(size_t)m,0,1); double best=1e9;
        for(int r=0;r<reps;r++){ double t0=now(); gf(m,n,k,W,B,C,(size_t)m,0,1); double dt=now()-t0; if(dt<best)best=dt; }
        printf("%-9s table GEMM: %6.1f GOPS (1 thread, m=%d n=%d k=%d)\n",q->name,2.0*m*n*k/best/1e9,m,n,k); free(W); free(C); }
    return 0; }
