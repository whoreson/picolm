/* Single-thread Q8_0 prefill GEMM comparison:
 *   A) picolm_sgemm_d (current Q8_0 prefill path)   B) sgemm_q8_0_r8_q8_0_avx2   C) raw VNNI peak */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include "quant.h"
#include "sgemm.h"
#include "tensor.h"
void fp16_table_init(void);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static unsigned rs=1; static float fr(void){rs=rs*1664525u+1013904223u;return ((rs>>8)&0xFFFF)/32768.0f-1.0f;}

static double peak_vnni512(void){
    __m512i a0=_mm512_set1_epi32(1),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0,a8=a0,a9=a0,a10=a0,a11=a0;
    __m512i x=_mm512_set1_epi8(3), y=_mm512_set1_epi8(5);
    long N=20000000; double t0=now();
    for(long i=0;i<N;i++){
        a0=_mm512_dpbusd_epi32(a0,x,y);a1=_mm512_dpbusd_epi32(a1,x,y);a2=_mm512_dpbusd_epi32(a2,x,y);a3=_mm512_dpbusd_epi32(a3,x,y);
        a4=_mm512_dpbusd_epi32(a4,x,y);a5=_mm512_dpbusd_epi32(a5,x,y);a6=_mm512_dpbusd_epi32(a6,x,y);a7=_mm512_dpbusd_epi32(a7,x,y);
        a8=_mm512_dpbusd_epi32(a8,x,y);a9=_mm512_dpbusd_epi32(a9,x,y);a10=_mm512_dpbusd_epi32(a10,x,y);a11=_mm512_dpbusd_epi32(a11,x,y);
        __asm__ volatile("" : "+v"(a0),"+v"(a1),"+v"(a2),"+v"(a3),"+v"(a4),"+v"(a5),"+v"(a6),"+v"(a7),"+v"(a8),"+v"(a9),"+v"(a10),"+v"(a11));
    }
    double dt=now()-t0; return (double)N*12*128/dt/1e9;   /* 64 MACs = 128 ops per instr */
}
static double peak_avx2_madd(void){
    __m256i a0=_mm256_set1_epi32(1),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0; __m256i x=_mm256_set1_epi8(3),y=_mm256_set1_epi8(5),m1=_mm256_set1_epi16(1);
    long N=20000000; double t0=now();
    for(long i=0;i<N;i++){
        __asm__ volatile("" : "+x"(x),"+x"(y));      /* defeat loop-invariant hoisting of maddubs/madd */
        a0=_mm256_add_epi32(a0,_mm256_madd_epi16(m1,_mm256_maddubs_epi16(x,y)));a1=_mm256_add_epi32(a1,_mm256_madd_epi16(m1,_mm256_maddubs_epi16(x,y)));
        a2=_mm256_add_epi32(a2,_mm256_madd_epi16(m1,_mm256_maddubs_epi16(x,y)));a3=_mm256_add_epi32(a3,_mm256_madd_epi16(m1,_mm256_maddubs_epi16(x,y)));
        __asm__ volatile("" : "+x"(x),"+x"(y));
        a4=_mm256_add_epi32(a4,_mm256_madd_epi16(m1,_mm256_maddubs_epi16(x,y)));a5=_mm256_add_epi32(a5,_mm256_madd_epi16(m1,_mm256_maddubs_epi16(x,y)));
        a6=_mm256_add_epi32(a6,_mm256_madd_epi16(m1,_mm256_maddubs_epi16(x,y)));a7=_mm256_add_epi32(a7,_mm256_madd_epi16(m1,_mm256_maddubs_epi16(x,y)));
        __asm__ volatile("" : "+x"(a0),"+x"(a1),"+x"(a2),"+x"(a3),"+x"(a4),"+x"(a5),"+x"(a6),"+x"(a7));
    }
    double dt=now()-t0; return (double)N*8*64/dt/1e9;
}
static double peak_vnni256(void){
    __m256i a0=_mm256_set1_epi32(1),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0,a8=a0,a9=a0,a10=a0,a11=a0;
    __m256i x=_mm256_set1_epi8(3), y=_mm256_set1_epi8(5);
    long N=40000000; double t0=now();
    for(long i=0;i<N;i++){
        a0=_mm256_dpbusd_epi32(a0,x,y);a1=_mm256_dpbusd_epi32(a1,x,y);a2=_mm256_dpbusd_epi32(a2,x,y);a3=_mm256_dpbusd_epi32(a3,x,y);
        a4=_mm256_dpbusd_epi32(a4,x,y);a5=_mm256_dpbusd_epi32(a5,x,y);a6=_mm256_dpbusd_epi32(a6,x,y);a7=_mm256_dpbusd_epi32(a7,x,y);
        a8=_mm256_dpbusd_epi32(a8,x,y);a9=_mm256_dpbusd_epi32(a9,x,y);a10=_mm256_dpbusd_epi32(a10,x,y);a11=_mm256_dpbusd_epi32(a11,x,y);
        __asm__ volatile("" : "+v"(a0),"+v"(a1),"+v"(a2),"+v"(a3),"+v"(a4),"+v"(a5),"+v"(a6),"+v"(a7),"+v"(a8),"+v"(a9),"+v"(a10),"+v"(a11));
    }
    double dt=now()-t0; return (double)N*12*64/dt/1e9;   /* 32 MACs = 64 ops per ymm dpbusd */
}
typedef struct {int which,m,n,k,nb,nth; const void *W,*R,*B; const float *Bd; float *C;} gctx_t;
static void gtask(int idx, void *p){
    gctx_t *g=(gctx_t*)p;
    if(g->which==0) picolm_sgemm_d(g->m,g->n,g->nb,g->W,g->nb,(const block_q8_0*)g->B,g->nb,g->Bd,g->nb,g->C,g->m,GGUF_TYPE_Q8_0,idx,g->nth);
    else if(g->which==1) sgemm_q8_0_r8_q8_0_avx2(g->m,g->n,g->k,g->R,g->B,g->C,g->m,idx,g->nth);
    else if(g->which==2) sgemm_q8_0_r8_vnni_q8_0(g->m,g->n,g->k,g->R,g->B,g->C,g->m,idx,g->nth);
    else picolm_sgemm_q8_0_vnni(g->m,g->n,g->nb,g->W,g->nb,g->B,g->nb,g->C,g->m,idx,g->nth);
}
static int check_mode(void){
    /* edge shapes: remainder column tiles (n%12), odd group counts per thread, tiny k, strided ld */
    int ms[]={8,16,24,40,200}, ns[]={1,2,3,5,7,11,12,13,25,95,96,97,200}, ks[]={32,64,1056}, ths[]={1,2,3,7};
    int bad=0,total=0; unsigned s=7;
    for(int a=0;a<5;a++)for(int bn=0;bn<13;bn++)for(int c=0;c<3;c++)for(int t=0;t<4;t++){
        int m=ms[a],n=ns[bn],k=ks[c],nth=ths[t],nb=k/32,lda=nb+(c==1?2:0),ldb=nb+(bn%2?1:0); /* padded strides */
        block_q8_0 *W=calloc((size_t)m*lda,sizeof(block_q8_0)),*B=calloc((size_t)n*ldb,sizeof(block_q8_0)); float *Bd=calloc((size_t)n*ldb,sizeof(float));
        float *tmp=malloc(sizeof(float)*k);
        for(int r=0;r<m;r++){for(int i=0;i<k;i++){s=s*1664525u+1013904223u;tmp[i]=((s>>8)&0xFFFF)/32768.0f-1.0f;}quantize_row_q8_0(tmp,W+(size_t)r*lda,k);}
        for(int cc=0;cc<n;cc++){for(int i=0;i<k;i++){s=s*1664525u+1013904223u;tmp[i]=((s>>8)&0xFFFF)/32768.0f-1.0f;}quantize_row_q8_0(tmp,B+(size_t)cc*ldb,k);
            for(int q=0;q<nb;q++)Bd[(size_t)cc*ldb+q]=fp16_to_fp32(B[(size_t)cc*ldb+q].d);}
        float *C1=calloc((size_t)n*m,sizeof(float)),*C2=calloc((size_t)n*m,sizeof(float));
        if(n>=2) picolm_sgemm_d(m,n,nb,W,lda,B,ldb,Bd,ldb,C1,m,GGUF_TYPE_Q8_0,0,1);
        else { /* sgemm_d needs n>=2: build reference by scalar math */
            for(int cc=0;cc<n;cc++)for(int r=0;r<m;r++){double acc=0;for(int q=0;q<nb;q++){int is=0;for(int i=0;i<32;i++)is+=W[(size_t)r*lda+q].qs[i]*B[(size_t)cc*ldb+q].qs[i];acc+=(double)is*fp16_to_fp32(W[(size_t)r*lda+q].d)*Bd[(size_t)cc*ldb+q];}C1[(size_t)cc*m+r]=(float)acc;} }
        int ok=1; for(int th=0;th<nth;th++) ok&=picolm_sgemm_q8_0_vnni(m,n,nb,W,lda,B,ldb,C2,m,th,nth);
        double md=0,rm=0; for(size_t i=0;i<(size_t)n*m;i++){double d=fabs(C1[i]-C2[i]);if(d>md)md=d;rm+=C1[i]*C1[i];} rm=sqrt(rm/((size_t)n*m)+1e-30);
        total++; if(!ok||md>1e-4*rm+1e-3){bad++;printf("FAIL m=%d n=%d k=%d nth=%d ok=%d maxdiff=%.3g rms=%.3g\n",m,n,k,nth,ok,md,rm);}
        free(W);free(B);free(Bd);free(tmp);free(C1);free(C2);
    }
    printf("check: %d/%d shapes OK\n",total-bad,total); return bad;
}
int main(int argc,char**argv){
    if(argc>1&&!strcmp(argv[1],"check")){fp16_table_init();return check_mode()?1:0;}
    int m=argc>1?atoi(argv[1]):2048, n=argc>2?atoi(argv[2]):256, k=argc>3?atoi(argv[3]):4096, reps=argc>4?atoi(argv[4]):3, nth=argc>5?atoi(argv[5]):1;
    fp16_table_init();
    if(nth>1){tensor_set_n_threads(nth);tensor_threadpool_init(nth);}
    int nb=k/32; size_t rb=(size_t)nb*sizeof(block_q8_0);
    float *tmp=malloc(sizeof(float)*k);
    block_q8_0 *W=malloc((size_t)m*rb), *B=malloc((size_t)n*rb); float *Bd=malloc(sizeof(float)*n*nb);
    for(int r=0;r<m;r++){for(int i=0;i<k;i++)tmp[i]=fr();quantize_row_q8_0(tmp,(char*)W+r*rb,k);}
    for(int c=0;c<n;c++){for(int i=0;i<k;i++)tmp[i]=fr();quantize_row_q8_0(tmp,(char*)B+c*rb,k);
        const block_q8_0*bl=(const block_q8_0*)((char*)B+c*rb);for(int b=0;b<nb;b++)Bd[(size_t)c*nb+b]=fp16_to_fp32(bl[b].d);}
    /* R8 weights: repack the SAME quantized bytes (layout per quant.c quantize_row_q8_0_r8) */
    block_q8_0_r8 *R=malloc((size_t)(m/8)*nb*sizeof(block_q8_0_r8));
    for(int g=0;g<m/8;g++)for(int ib=0;ib<nb;ib++){
        block_q8_0_r8*b=&R[(size_t)g*nb+ib];
        for(int r=0;r<8;r++){const block_q8_0*s=(const block_q8_0*)((char*)W+(size_t)(g*8+r)*rb)+ib;b->d[r]=s->d;
            for(int l=0;l<4;l++)for(int i=0;i<4;i++){b->qs[32*l+4*r+i]=s->qs[4*l+i];b->qs[32*l+4*r+i+128]=s->qs[16+4*l+i];}}
    }
    float *CA=calloc((size_t)n*m,sizeof(float)),*CB=calloc((size_t)n*m,sizeof(float));
    double ops=2.0*m*n*k, best;
    int okA=picolm_sgemm_d(m,n,nb,W,nb,B,nb,Bd,nb,CA,m,GGUF_TYPE_Q8_0,0,1);
    best=1e9;for(int r=0;r<reps;r++){double t=now();picolm_sgemm_d(m,n,nb,W,nb,B,nb,Bd,nb,CA,m,GGUF_TYPE_Q8_0,0,1);t=now()-t;if(t<best)best=t;}
    double gA=ops/best/1e9;
    sgemm_q8_0_r8_q8_0_avx2(m,n,k,R,B,CB,m,0,1);
    best=1e9;for(int r=0;r<reps;r++){double t=now();sgemm_q8_0_r8_q8_0_avx2(m,n,k,R,B,CB,m,0,1);t=now()-t;if(t<best)best=t;}
    double gB=ops/best/1e9;
    float *CV=calloc((size_t)n*m,sizeof(float));
    int okV=sgemm_q8_0_r8_vnni_q8_0(m,n,k,R,B,CV,m,0,1);
    best=1e9;for(int r=0;r<reps&&okV;r++){double t=now();sgemm_q8_0_r8_vnni_q8_0(m,n,k,R,B,CV,m,0,1);t=now()-t;if(t<best)best=t;}
    double gV=okV?ops/best/1e9:0, mdV=0; if(okV)for(size_t i=0;i<(size_t)n*m;i++){double d=fabs(CA[i]-CV[i]);if(d>mdV)mdV=d;}
    float *CP=calloc((size_t)n*m,sizeof(float));
    int okP=picolm_sgemm_q8_0_vnni(m,n,nb,W,nb,B,nb,CP,m,0,1);
    best=1e9;for(int r=0;r<reps&&okP;r++){double t=now();picolm_sgemm_q8_0_vnni(m,n,nb,W,nb,B,nb,CP,m,0,1);t=now()-t;if(t<best)best=t;}
    double gP=okP?ops/best/1e9:0, mdP=0; if(okP)for(size_t i=0;i<(size_t)n*m;i++){double d=fabs(CA[i]-CP[i]);if(d>mdP)mdP=d;}
    double md=0,rms=0;for(size_t i=0;i<(size_t)n*m;i++){double d=fabs(CA[i]-CB[i]);if(d>md)md=d;rms+=CA[i]*CA[i];}rms=sqrt(rms/((size_t)n*m));
    double pv=peak_vnni512(), pv2=peak_vnni256(), pa=peak_avx2_madd();
    printf("shape m=%d n=%d k=%d  (1 thread)\n",m,n,k);
    printf("  sgemm_d Q8_0 (current prefill)   : %7.1f GOPS   (ok=%d)\n",gA,okA);
    printf("  sgemm_q8_0_r8 AVX2 (R8 kernel)   : %7.1f GOPS   max|A-B|=%.2e (rms %.3g)\n",gB,md,rms);
    printf("  sgemm_q8_0_r8_vnni (new, 12-col)  : %7.1f GOPS   max|A-V|=%.2e%s\n",gV,mdV,okV?"":"   (unavailable: needs AVX-512 VNNI+VL build)");
    printf("  picolm_sgemm_q8_0_vnni (plain Q8_0 weights, on-the-fly repack): %7.1f GOPS   max|A-P|=%.2e   speedup vs current: %.2fx%s\n",gP,mdP,gA>0?gP/gA:0,okP?"":"   (unavailable)");
    printf("  per-core peaks: VNNI zmm %.1f | VNNI ymm %.1f | AVX2 maddubs chain %.1f GOPS\n",pv,pv2,pa);
    printf("  efficiency vs ymm-VNNI peak: current %.1f%%  r8-avx2 %.1f%%  r8-vnni %.1f%%   (speedup of r8-vnni over current: %.2fx)\n",gA/pv2*100,gB/pv2*100,gV/pv2*100,gA>0?gV/gA:0);
    if(nth>1){
        float *CT=calloc((size_t)n*m,sizeof(float));
        for(int which=0;which<4;which++){ if((which==2&&!okV)||(which==3&&!okP))continue;
            gctx_t g={which,m,n,k,nb,nth,W,R,B,Bd,CT};
            tensor_parallel_for(nth,gtask,&g);   /* warm */
            double bt=1e9;for(int r=0;r<reps;r++){double t=now();tensor_parallel_for(nth,gtask,&g);t=now()-t;if(t<bt)bt=t;}
            double gt=ops/bt/1e9, g1=which==0?gA:which==1?gB:which==2?gV:gP; double mdd=0;
            const float*ref=which==0?CA:which==1?CB:which==2?CV:CP;for(size_t i=0;i<(size_t)n*m;i++){double d=fabs(CT[i]-ref[i]);if(d>mdd)mdd=d;}
            printf("  %2d threads %-10s: %8.1f GOPS total = %6.1f per thread  -> scaling %.1f%% of ideal (%.1fx of 1T)   max|diff vs 1T|=%.1e\n",
                   nth,which==0?"sgemm_d":which==1?"r8-avx2":which==2?"r8-vnni":"q8-vnni",gt,gt/nth,gt/(g1*nth)*100,gt/g1,mdd);
        }
    }
    return 0;
}
