/* IQ6_K prefill GEMM: decode-once R8 kernel (sgemm_iq6_k_q8_k_r8t) vs the 1x2 tile kernel.
 *   ./bench_iq6k_gemm check                      edge-shape correctness vs the tile kernel
 *   ./bench_iq6k_gemm [m n k reps threads]       throughput (GOPS), e.g. 14336 512 4096 3 24
 * Weights are random IQ6_K blocks (any bit pattern is a valid block; d is set to a sane fp16). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quant.h"
#include "tensor.h"
void fp16_table_init(void);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static unsigned rs=5; static unsigned rnd(void){rs=rs*1664525u+1013904223u;return rs>>8;}
typedef struct{int which,m,n,k,nth;const void*W,*Y;float*C;}ctx_t;
static void task(int idx,void*p){ctx_t*c=p;
    if(c->which==0) sgemm_iq6_k_q8_k_avx2(c->m,c->n,c->k,c->W,c->Y,c->C,c->m,idx,c->nth);
    else sgemm_iq6_k_q8_k_r8t(c->m,c->n,c->k,c->W,c->Y,c->C,c->m,idx,c->nth);}
static block_iq6_k *mkW(int m,int nb){block_iq6_k*W=malloc((size_t)m*nb*sizeof(block_iq6_k));
    for(size_t i=0;i<(size_t)m*nb;i++){uint8_t*b=(uint8_t*)&W[i];for(size_t j=0;j<sizeof(block_iq6_k);j++)b[j]=rnd();W[i].d=0x2C00+(rnd()&0xFF);
        for(int g=0;g<16;g++)W[i].scales[g]=(int8_t)((int)(rnd()%64)-32);}   /* realistic scale range */
    return W;}
static block_q8_K *mkY(int n,int k){size_t nb=k/256,rb=nb*sizeof(block_q8_K);block_q8_K*Y=malloc((size_t)n*rb);float*t=malloc(4*k);
    for(int c=0;c<n;c++){for(int i=0;i<k;i++)t[i]=((rnd()&0xFFFF)/32768.0f-1.0f)*((c%5)+1);quantize_row_q8_K(t,(char*)Y+c*rb,k);}free(t);return Y;}
static int check(void){
    int ms[]={8,16,24,40,200}, ns[]={1,2,3,5,7,8,9,11,12,13,25,95,96,97,200}, ks[]={256,512,1280,4096}, ths[]={1,2,3,7}; int bad=0,total=0;
    for(int a=0;a<5;a++)for(int b=0;b<15;b++)for(int c=0;c<4;c++)for(int t=0;t<4;t++){
        int m=ms[a],n=ns[b],k=ks[c],nth=ths[t],nb=k/256; block_iq6_k*W=mkW(m,nb); block_q8_K*Y=mkY(n,k);
        float*C1=calloc((size_t)n*m,4),*C2=calloc((size_t)n*m,4);
        sgemm_iq6_k_q8_k_avx2(m,n,k,W,Y,C1,m,0,1);
        int ok=1; for(int th=0;th<nth;th++) ok&=(sgemm_iq6_k_q8_k_r8t(m,n,k,W,Y,C2,m,th,nth)!=0);
        double md=0,rm=0;for(size_t i=0;i<(size_t)n*m;i++){double d=fabs(C1[i]-C2[i]);if(d>md)md=d;rm+=(double)C1[i]*C1[i];}rm=sqrt(rm/((size_t)n*m)+1e-30);
        total++; if(!ok||md>2e-5*rm+1e-3){bad++;printf("FAIL m=%d n=%d k=%d nth=%d ok=%d maxdiff=%.3g rms=%.3g\n",m,n,k,nth,ok,md,rm);}
        free(W);free(Y);free(C1);free(C2);}
    printf("check: %d/%d shapes OK\n",total-bad,total);return bad;}
int main(int argc,char**argv){
    fp16_table_init();
    if(argc>1&&!strcmp(argv[1],"check")) return check()?1:0;
    int m=argc>1?atoi(argv[1]):4096,n=argc>2?atoi(argv[2]):512,k=argc>3?atoi(argv[3]):4096,reps=argc>4?atoi(argv[4]):2,nth=argc>5?atoi(argv[5]):1;
    if(nth>1){tensor_set_n_threads(nth);tensor_threadpool_init(nth);}
    int nb=k/256; block_iq6_k*W=mkW(m,nb); block_q8_K*Y=mkY(n,k); float*C0=calloc((size_t)n*m,4),*C1=calloc((size_t)n*m,4); double ops=2.0*m*n*k,gt[2]={0,0};
    for(int which=0;which<2;which++){ float*C=which?C1:C0; ctx_t cx={which,m,n,k,nth,W,Y,C}; double best=1e9;
        for(int r=-1;r<reps;r++){double t=now(); if(nth>1)tensor_parallel_for(nth,task,&cx); else task(0,&cx); t=now()-t; if(r>=0&&t<best)best=t;}
        gt[which]=ops/best/1e9; printf("  %-34s: %8.1f GOPS total = %6.1f per thread  (%d thread%s)\n",which?"decode-once R8 (new)":"1x2 tile kernel (old)",gt[which],gt[which]/nth,nth,nth>1?"s":""); }
    double md=0,rm=0;for(size_t i=0;i<(size_t)n*m;i++){double d=fabs(C0[i]-C1[i]);if(d>md)md=d;rm+=(double)C0[i]*C0[i];}rm=sqrt(rm/((size_t)n*m));
    printf("  speedup %.2fx   max|old-new|=%.2e (rms %.3g)\n",gt[1]/gt[0],md,rm); return 0;}
