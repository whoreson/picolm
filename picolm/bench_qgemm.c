/* Generic legacy-quant bench: prefill GEMM (picolm_sgemm_d) GOPS and decode dot (vec_dot) GB/s.
 *   ./bench_qgemm <q8_0|q4_0|iq4_nl|q5_0> [m n k reps nth]        (check: ./bench_qgemm check)
 * Weights are random bytes (any 4-bit pattern is a valid IQ4_NL/Q4_0 block), so no quantizer needed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quant.h"
#include "sgemm.h"
#include "tensor.h"
void fp16_table_init(void);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static unsigned rs=12345; static unsigned rnd(void){rs=rs*1664525u+1013904223u;return rs>>8;}
static float fr(void){return (rnd()&0xFFFF)/32768.0f-1.0f;}
typedef struct { const char *name; int type; size_t bsz; } qt_t;
static const qt_t QT[] = {{"q8_0",GGUF_TYPE_Q8_0,sizeof(block_q8_0)},{"q4_0",GGUF_TYPE_Q4_0,sizeof(block_q4_0)},
                          {"iq4_nl",GGUF_TYPE_IQ4_NL,sizeof(block_iq4_nl)},{"q5_0",GGUF_TYPE_Q5_0,sizeof(block_q5_0)}};
static void fill_w(unsigned char *p, size_t n, size_t bsz){          /* random payload + sane fp16 scale in first 2 bytes */
    for(size_t i=0;i<n;i+=bsz){ for(size_t j=0;j<bsz;j++) p[i+j]=(unsigned char)rnd(); uint16_t d=fp32_to_fp16(0.01f+(rnd()&255)*1e-4f); memcpy(p+i,&d,2); }
}
/* scalar reference value of one weight (int) for block b, index i */
static int wval(int type,const unsigned char*b,int i){
    if(type==GGUF_TYPE_Q8_0) return ((const int8_t*)(b+2))[i];
    if(type==GGUF_TYPE_Q4_0){ unsigned char q=b[2+(i&15)]; return (i<16?(q&15):(q>>4))-8; }
    if(type==GGUF_TYPE_IQ4_NL){ unsigned char q=b[2+(i&15)]; return kvalues_iq4nl[i<16?(q&15):(q>>4)]; }
    const block_q5_0*q5=(const block_q5_0*)b; uint32_t qh; memcpy(&qh,q5->qh,4);
    int lo=(i<16)?(q5->qs[i]&15):(q5->qs[i-16]>>4); int hi=(qh>>i)&1; return (lo|(hi<<4))-16;
}
static int run(const qt_t*t,int m,int n,int k,int reps,int nth,int check){
    int nb=k/32; size_t wrow=(size_t)nb*t->bsz, qrow=(size_t)nb*sizeof(block_q8_0);
    unsigned char *W=malloc((size_t)m*wrow); fill_w(W,(size_t)m*wrow,t->bsz);
    block_q8_0 *B=malloc((size_t)n*qrow); float *Bd=malloc(sizeof(float)*(size_t)n*nb), *tmp=malloc(sizeof(float)*k);
    for(int c=0;c<n;c++){for(int i=0;i<k;i++)tmp[i]=fr();quantize_row_q8_0(tmp,(char*)B+c*qrow,k);
        const block_q8_0*bl=(const block_q8_0*)((char*)B+c*qrow);for(int b=0;b<nb;b++)Bd[(size_t)c*nb+b]=fp16_to_fp32(bl[b].d);}
    float *C=calloc((size_t)n*m,sizeof(float));
    int ok=1;
    if(nth<=1) ok=picolm_sgemm_d(m,n,nb,W,nb,B,nb,Bd,nb,C,m,t->type,0,1);
    else { for(int i=0;i<nth;i++) ok&=picolm_sgemm_d(m,n,nb,W,nb,B,nb,Bd,nb,C,m,t->type,i,nth); }  /* serial emulation of split */
    double gops=0; 
    if(!ok){ printf("%-7s prefill: picolm_sgemm_d unsupported for this type/build (falls to per-column vec_dot)\n",t->name); }
    else {
        double best=1e9; for(int r=0;r<reps;r++){double t0=now();picolm_sgemm_d(m,n,nb,W,nb,B,nb,Bd,nb,C,m,t->type,0,1);double dt=now()-t0;if(dt<best)best=dt;}
        gops=2.0*m*n*k/best/1e9; printf("%-7s prefill m=%d n=%d k=%d : %7.1f GOPS (1 thread)\n",t->name,m,n,k,gops);
    }
    if(check){
        double maxd=0,rms=0; int bad=0;
        for(int c=0;c<n&&c<8;c++)for(int r=0;r<m&&r<64;r++){ double acc=0;
            for(int q=0;q<nb;q++){ const unsigned char*wb=W+(size_t)r*wrow+(size_t)q*t->bsz; int is=0;
                for(int i=0;i<32;i++) is+=wval(t->type,wb,i)*((const block_q8_0*)((char*)B+c*qrow))[q].qs[i];
                uint16_t d; memcpy(&d,wb,2); acc+=(double)is*fp16_to_fp32(d)*Bd[(size_t)c*nb+q]; }
            double dd=fabs(acc-C[(size_t)c*m+r]); if(dd>maxd)maxd=dd; rms+=acc*acc; }
        rms=sqrt(rms/(8.0*64)); if(maxd>1e-3*rms+1e-3){bad=1;} printf("%-7s check: maxdiff=%.3g (rms %.3g) %s\n",t->name,maxd,rms,bad?"FAIL":"ok"); if(bad) return 1;
    }
    /* decode: dot of every weight row against one activation, weights >> cache (no exported Q5_0 dot) */
    if(t->type!=GGUF_TYPE_Q5_0){ size_t rows=(size_t)m; double best=1e9; float sink=0; const block_q8_0*y=B;
      for(int r=0;r<reps;r++){ double t0=now();
        for(size_t i=0;i<rows;i++){ const void*w=W+i*wrow;
            sink+= t->type==GGUF_TYPE_Q8_0?vec_dot_q8_0_q8_0(w,y,k): t->type==GGUF_TYPE_Q4_0?vec_dot_q4_0_q8_0(w,y,k): t->type==GGUF_TYPE_IQ4_NL?vec_dot_iq4_nl_q8_0(w,y,k): 0.0f; }
        double dt=now()-t0; if(dt<best)best=dt; }
      printf("%-7s decode dot: %6.1f ns/row (k=%d)  %5.2f GB/s weights  [%g]\n",t->name,best*1e9/rows,k,(double)rows*wrow/best/1e9,sink); }
    free(W);free(B);free(Bd);free(tmp);free(C); return 0;
}
int main(int argc,char**argv){
    fp16_table_init();
    if(argc>1&&!strcmp(argv[1],"check")){ int bad=0; int shapes[][3]={{8,5,64},{16,33,128},{24,64,256},{64,7,96},{40,128,512}};
        for(unsigned t=0;t<sizeof QT/sizeof*QT;t++){for(unsigned s=0;s<sizeof shapes/sizeof*shapes;s++) bad|=run(&QT[t],shapes[s][0],shapes[s][1],shapes[s][2],1,1,1);
            bad|=run(&QT[t],72,70,320,1,3,1); bad|=run(&QT[t],64,129,224,1,5,1); } return bad; }
    const char*tn=argc>1?argv[1]:"iq4_nl"; int m=argc>2?atoi(argv[2]):4096,n=argc>3?atoi(argv[3]):512,k=argc>4?atoi(argv[4]):4096,reps=argc>5?atoi(argv[5]):3,nth=argc>6?atoi(argv[6]):1;
    for(unsigned t=0;t<sizeof QT/sizeof*QT;t++) if(!strcmp(QT[t].name,tn)||!strcmp(tn,"all")) run(&QT[t],m,n,k,reps,nth,0);
    return 0; }
