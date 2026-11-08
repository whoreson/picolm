/* Decode (GEMV) cost per row for every quant type that has a table fn_gemv: what a generated token pays.
 *   ./bench_gemv [rows k reps]       (defaults 512 4096 5; weights are random with sane scales; rows is a multiple of 4)
 * Reports ns/row and GB/s of weights streamed per thread (in-cache-ish, so it shows the COMPUTE ceiling). */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <time.h>
#include <math.h>
#include "quant.h"
#include "type_dispatch.h"
void fp16_table_init(void);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static unsigned rs=99; static unsigned rnd(void){rs=rs*1664525u+1013904223u;return rs>>8;}
/* name, type, block bytes, rows per block, #fp16 scales at d_off (+ optional second scale region), act */
typedef struct { const char *name; int gt; size_t bsz; int rows; size_t d_off; int nd; size_t d2_off; int nd2; int kblk; /* values per block along k */ int q8k; } g_t;
#define L(nm,gt,T,rows,doff,nd,k,q8k) {nm,gt,sizeof(T),rows,doff,nd,(size_t)-1,0,k,q8k}
static const g_t G[] = {
 L("q8_0",GGUF_TYPE_Q8_0,block_q8_0,1,0,1,32,0), L("q4_0",GGUF_TYPE_Q4_0,block_q4_0,1,0,1,32,0), L("iq4_nl",GGUF_TYPE_IQ4_NL,block_iq4_nl,1,0,1,32,0),
 L("iq4_nl_r4",GGUF_TYPE_IQ4_NL_R4,block_iq4_nl_r4,4,0,4,32,0),
 L("q2_K",GGUF_TYPE_Q2_K,block_q2_K,1,offsetof(block_q2_K,d),2,256,1), L("q3_K",GGUF_TYPE_Q3_K,block_q3_K,1,offsetof(block_q3_K,d),1,256,1),
 L("q4_K",GGUF_TYPE_Q4_K,block_q4_K,1,offsetof(block_q4_K,d),2,256,1), L("q5_K",GGUF_TYPE_Q5_K,block_q5_K,1,offsetof(block_q5_K,d),2,256,1),
 L("q6_K",GGUF_TYPE_Q6_K,block_q6_K,1,offsetof(block_q6_K,d),1,256,1), L("iq4_xs",GGUF_TYPE_IQ4_XS,block_iq4_xs,1,0,1,256,1),
 L("iq2_k",GGUF_TYPE_IQ2_K,block_iq2_k,1,0,1,256,1), L("iq3_k",GGUF_TYPE_IQ3_K,block_iq3_k,1,0,1,256,1), L("iq4_k",GGUF_TYPE_IQ4_K,block_iq4_k,1,0,1,256,1),
 L("iq6_k",GGUF_TYPE_IQ6_K,block_iq6_k,1,0,1,256,1),
 L("iq2_k_r4",GGUF_TYPE_IQ2_K_R4,block_iq2_k_r4,4,0,4,256,1), L("iq3_k_r4",GGUF_TYPE_IQ3_K_R4,block_iq3_k_r4,4,0,4,256,1),
 L("iq4_k_r4",GGUF_TYPE_IQ4_K_R4,block_iq4_k_r4,4,0,4,256,1), L("q4_k_r4",GGUF_TYPE_Q4_K_R4,block_q4_k_r4,4,0,8,256,1),
 L("q6_K_r4",GGUF_TYPE_Q6_K_R4,block_q6_K_R4,4,0,4,256,1),
};

/* ---- correctness of decode kernels against the repo's own float dequantizers ---- */
typedef void (*row_deq_fn)(const void *grp, float *dst, int k, int row);
static void d_iq4_k(const void*g,float*d,int k,int r){(void)r;dequantize_row_iq4_k(g,d,k);}
static void d_q4_k_r4(const void*g,float*d,int k,int r){dequantize_row_q4_k_r4_single((const block_q4_k_r4*)g,d,k,r);}
static void d_iq4_k_r4(const void*g,float*d,int k,int r){dequantize_row_iq4_k_r4_single((const block_iq4_k_r4*)g,d,k,r);}
static int check_type(const g_t*q,row_deq_fn deq,int rows,int k){
    const type_info_t*ti=typeinfo((gguf_type_t)q->gt); int nbk=k/q->kblk; size_t groups=(size_t)rows/q->rows, gbytes=(size_t)nbk*q->bsz; unsigned char*W=malloc(groups*gbytes);
    for(size_t i=0;i<groups*gbytes;i++) W[i]=(unsigned char)rnd();
    for(size_t i=0;i<groups*nbk;i++){ unsigned char*b=W+i*q->bsz+q->d_off; for(int j=0;j<q->nd;j++){ uint16_t d=fp32_to_fp16(0.001f+(rnd()&255)*1e-5f); memcpy(b+2*j,&d,2);} }
    float*tmpv=malloc(sizeof(float)*k); for(int i=0;i<k;i++) tmpv[i]=(rnd()&0xFFFF)/32768.0f-1.0f;
    block_q8_K*y=malloc((size_t)(k/256)*sizeof(block_q8_K)); quantize_row_q8_K(tmpv,y,k);
    float*out=malloc(sizeof(float)*(size_t)rows); for(int i=0;i<rows;i++) out[i]=1e30f;
    if(q->rows==1) for(int i=0;i<rows;i++) ti->fn_gemv(W+(size_t)i*gbytes,y,k,out+i,1);
    else for(size_t g=0;g<groups;g++) ti->fn_gemv(W+g*gbytes,y,k,out+g*q->rows,q->rows);
    float*row=malloc(sizeof(float)*k); double md=0,rm=0; int bad=0;
    for(int r=0;r<rows;r++){ deq(W+(size_t)(r/q->rows)*gbytes,row,k,r%q->rows); double ref=0;
        for(int b2=0;b2<k/256;b2++){ double s=0; for(int i=0;i<256;i++) s+=(double)row[b2*256+i]*y[b2].qs[i]; ref+=s*y[b2].d; }
        double dd=fabs(ref-out[r]); if(!(dd==dd)||out[r]>1e29f) bad=1; if(dd>md)md=dd; rm+=ref*ref; }
    rm=sqrt(rm/rows+1e-30); if(md>2e-4*rm+1e-4) bad=1;
    if(bad) printf("FAIL %s rows=%d k=%d maxdiff=%.3g rms=%.3g\n",q->name,rows,k,md,rm);
    free(W);free(tmpv);free(y);free(out);free(row); return bad; }
int main(int argc,char**argv){
    if(argc>1&&!strcmp(argv[1],"check")){ fp16_table_init(); int bad=0,cnt=0; int rs_[]={4,8,12,64}, ks[]={256,512,1280,4096};
        struct { const char*n; row_deq_fn f; } T[]={{"iq4_k",d_iq4_k},{"q4_k_r4",d_q4_k_r4},{"iq4_k_r4",d_iq4_k_r4}};
        for(unsigned t=0;t<3;t++){ const g_t*q=NULL; for(unsigned i=0;i<sizeof G/sizeof*G;i++) if(!strcmp(G[i].name,T[t].n)) q=&G[i];
            for(int a=0;a<4;a++)for(int c=0;c<4;c++)for(int rep=0;rep<12;rep++){ bad|=check_type(q,T[t].f,rs_[a],ks[c]); cnt++; } }
        printf("check: %d cases, %s\n",cnt,bad?"FAIL":"all OK"); return bad; }
    int rows=argc>1?atoi(argv[1]):512, k=argc>2?atoi(argv[2]):4096, reps=argc>3?atoi(argv[3]):5; fp16_table_init();
    float *tmp=malloc(sizeof(float)*k); for(int i=0;i<k;i++) tmp[i]=(rnd()&0xFFFF)/32768.0f-1.0f;
    char *a8=malloc((size_t)(k/32+1)*sizeof(block_q8_0)), *aK=malloc((size_t)(k/256+1)*sizeof(block_q8_K));
    quantize_row_q8_0(tmp,a8,k); quantize_row_q8_K(tmp,aK,k);
    printf("%-10s %9s %9s   (k=%d, %d rows, 1 thread, in-cache-ish => compute ceiling)\n","type","ns/row","GB/s",k,rows);
    for(unsigned t=0;t<sizeof G/sizeof*G;t++){ const g_t*q=&G[t]; const type_info_t*ti=typeinfo((gguf_type_t)q->gt);
        if(!ti||!ti->fn_gemv){ printf("%-10s no fn_gemv\n",q->name); continue; }
        int nbk=k/q->kblk; size_t groups=(size_t)rows/q->rows, gbytes=(size_t)nbk*q->bsz, wbytes=groups*gbytes; unsigned char*W=malloc(wbytes);
        for(size_t i=0;i<wbytes;i++) W[i]=(unsigned char)rnd();
        for(size_t i=0;i<groups*nbk;i++){ unsigned char*b=W+i*q->bsz+q->d_off; for(int j=0;j<q->nd;j++){ uint16_t d=fp32_to_fp16(0.001f+(rnd()&255)*1e-5f); memcpy(b+2*j,&d,2);} }
        float*out=malloc(sizeof(float)*(size_t)rows); const void*act=q->q8k?(void*)aK:(void*)a8; double best=1e9;
        for(int r=0;r<reps;r++){ double t0=now();
            if(q->rows==1){ for(int i=0;i<rows;i++) ti->fn_gemv(W+(size_t)i*gbytes,act,k,out+i,1); }   /* plain types: one row per call */
            else { for(size_t g=0;g<groups;g++) ti->fn_gemv(W+g*gbytes,act,k,out+g*q->rows,q->rows); }
            double dt=now()-t0; if(dt<best)best=dt; }
        { uint64_t h=1469598103934665603ull; for(int i=0;i<rows;i++){ uint32_t u; memcpy(&u,&out[i],4); h=(h^u)*1099511628211ull; }
          printf("%-10s %9.1f %9.2f   out-hash %016llx\n",q->name,best*1e9/rows,(double)wbytes/best/1e9,(unsigned long long)h); }
        free(W);free(out); }
    return 0; }
