// Port of llama.cpp llamafile/sgemm.cpp tinyBLAS tiled GEMM to C.
// MIT License (Copyright 2024 Mozilla Foundation).
// Tiled GEMM: C[n x m] = A^T[m x k] * B[k x n]
#include "sgemm.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__AVX__)
#include <immintrin.h>
#endif
#if defined(__ALTIVEC__)
#include <altivec.h>
#undef bool
#undef true
#undef false
#endif
#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

static inline int64_t bloc_pos(int64_t ib, int64_t ibN, int64_t bloc_size) {
    return ib < ibN ? ib * bloc_size : ibN * bloc_size + (ib - ibN) * (bloc_size - 1);
}

#define JSTART(nb,ith,nth) ((nb)*(ith)/(nth))
#define JEND(nb,ith,nth)   ((nb)*((ith)+1)/(nth))

/* ============================================================
 * F32 x F32 tiled GEMM
 * ============================================================ */

#if defined(__AVX512F__)

#define F32_KN 16
#define F32_RM 4
#define F32_RN 6
#define F32_BN 12
#define F32_V __m512
#define F32_ZERO _mm512_setzero_ps()
#define F32_LDA(p) _mm512_loadu_ps(p)
#define F32_LDB(p) _mm512_loadu_ps(p)
#define F32_MADD(a,b,c) _mm512_fmadd_ps(a,b,c)
#define F32_HSUM(x) _mm512_reduce_add_ps(x)

static void sgemm_f32_f32(int m, int n, int k, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
    const int64_t KN=F32_KN, RM=F32_RM, RN=F32_RN, BN=F32_BN;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                F32_V Cv[F32_RN][F32_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=F32_ZERO;
                for (int64_t l=0;l<k;l+=KN) {
                    F32_V Av[F32_RM];
                    for(int c=0;c<RM;c++) Av[c]=F32_LDA(A+lda*(ii+c)+l);
                    for(int r=0;r<RN;r++) { F32_V Bv=F32_LDB(B+ldb*(jj+r)+l);
                        for(int c=0;c<RM;c++) Cv[r][c]=F32_MADD(Av[c],Bv,Cv[r][c]); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) C[ldc*(jj+r)+(ii+c)]=F32_HSUM(Cv[r][c]);
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                F32_V Cv[F32_RM];
                for(int c=0;c<RM;c++) Cv[c]=F32_ZERO;
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) Cv[c]=F32_MADD(F32_LDA(A+lda*(ii+c)+l),F32_LDB(B+ldb*jj+l),Cv[c]);
                for(int c=0;c<RM;c++) C[ldc*jj+(ii+c)]=F32_HSUM(Cv[c]);
            }
        }
    }
}
#undef F32_KN
#undef F32_RM
#undef F32_RN
#undef F32_BN
#undef F32_V
#undef F32_ZERO
#undef F32_LDA
#undef F32_LDB
#undef F32_MADD
#undef F32_HSUM

#elif defined(__AVX__) || defined(__AVX2__)

#define F32_KN 8
#define F32_RM 4
#define F32_RN 3
#define F32_BN 24
#define F32_V __m256
#define F32_ZERO _mm256_setzero_ps()
#define F32_LDA(p) _mm256_loadu_ps(p)
#define F32_LDB(p) _mm256_loadu_ps(p)
#ifdef __FMA__
#define F32_MADD(a,b,c) _mm256_fmadd_ps(a,b,c)
#else
#define F32_MADD(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
#endif

static float f32_hsum(F32_V x) {
    __m128 s=_mm_add_ps(_mm256_castps256_ps128(x),_mm256_extractf128_ps(x,1));
    s=_mm_add_ps(s,_mm_movehl_ps(s,s)); s=_mm_add_ss(s,_mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
#define F32_HSUM(x) f32_hsum(x)

static void sgemm_f32_f32(int m, int n, int k, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
    const int64_t KN=F32_KN, RM=F32_RM, RN=F32_RN, BN=F32_BN;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                F32_V Cv[F32_RN][F32_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=F32_ZERO;
                for (int64_t l=0;l<k;l+=KN) {
                    F32_V Av[F32_RM];
                    for(int c=0;c<RM;c++) Av[c]=F32_LDA(A+lda*(ii+c)+l);
                    for(int r=0;r<RN;r++) { F32_V Bv=F32_LDB(B+ldb*(jj+r)+l);
                        for(int c=0;c<RM;c++) Cv[r][c]=F32_MADD(Av[c],Bv,Cv[r][c]); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) C[ldc*(jj+r)+(ii+c)]=F32_HSUM(Cv[r][c]);
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                F32_V Cv[F32_RM];
                for(int c=0;c<RM;c++) Cv[c]=F32_ZERO;
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) Cv[c]=F32_MADD(F32_LDA(A+lda*(ii+c)+l),F32_LDB(B+ldb*jj+l),Cv[c]);
                for(int c=0;c<RM;c++) C[ldc*jj+(ii+c)]=F32_HSUM(Cv[c]);
            }
        }
    }
}
#undef F32_HSUM
#undef f32_hsum

#undef F32_KN
#undef F32_RM
#undef F32_RN
#undef F32_BN
#undef F32_V
#undef F32_ZERO
#undef F32_LDA
#undef F32_LDB
#undef F32_MADD

#elif defined(__SSE__)

#define F32_KN 4
#define F32_RM 4
#define F32_RN 3
#define F32_BN 24
#define F32_V __m128
#define F32_ZERO _mm_setzero_ps()
#define F32_LDA(p) _mm_loadu_ps(p)
#define F32_LDB(p) _mm_loadu_ps(p)
#define F32_MADD(a,b,c) _mm_add_ps(_mm_mul_ps(a,b),c)

static float f32_hsum(F32_V x) {
    __m128 t=_mm_shuffle_ps(x,x,_MM_SHUFFLE(2,3,0,1));
    x=_mm_add_ps(x,t); t=_mm_movehl_ps(t,x); x=_mm_add_ss(x,t);
    return _mm_cvtss_f32(x);
}
#define F32_HSUM(x) f32_hsum(x)

static void sgemm_f32_f32(int m, int n, int k, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
    const int64_t KN=F32_KN, RM=F32_RM, RN=F32_RN, BN=F32_BN;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                F32_V Cv[F32_RN][F32_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=F32_ZERO;
                for (int64_t l=0;l<k;l+=KN) {
                    F32_V Av[F32_RM];
                    for(int c=0;c<RM;c++) Av[c]=F32_LDA(A+lda*(ii+c)+l);
                    for(int r=0;r<RN;r++) { F32_V Bv=F32_LDB(B+ldb*(jj+r)+l);
                        for(int c=0;c<RM;c++) Cv[r][c]=F32_MADD(Av[c],Bv,Cv[r][c]); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) C[ldc*(jj+r)+(ii+c)]=F32_HSUM(Cv[r][c]);
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                F32_V Cv[F32_RM];
                for(int c=0;c<RM;c++) Cv[c]=F32_ZERO;
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) Cv[c]=F32_MADD(F32_LDA(A+lda*(ii+c)+l),F32_LDB(B+ldb*jj+l),Cv[c]);
                for(int c=0;c<RM;c++) C[ldc*jj+(ii+c)]=F32_HSUM(Cv[c]);
            }
        }
    }
}
#undef F32_KN
#undef F32_RM
#undef F32_RN
#undef F32_BN
#undef F32_V
#undef F32_ZERO
#undef F32_LDA
#undef F32_LDB
#undef F32_MADD
#undef F32_HSUM
#undef f32_hsum

#elif defined(__ARM_NEON)

#define F32_KN 4
#define F32_RM 4
#define F32_RN 6
#define F32_BN 12
#define F32_V float32x4_t
#define F32_ZERO vdupq_n_f32(0)
#define F32_LDA(p) vld1q_f32(p)
#define F32_LDB(p) vld1q_f32(p)
#define F32_MADD(c,a,b) vfmaq_f32(c,a,b)
#define F32_HSUM(x) vaddvq_f32(x)

static void sgemm_f32_f32(int m, int n, int k, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
    const int64_t KN=F32_KN, RM=F32_RM, RN=F32_RN, BN=F32_BN;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                F32_V Cv[F32_RN][F32_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=F32_ZERO;
                for (int64_t l=0;l<k;l+=KN) {
                    F32_V Av[F32_RM];
                    for(int c=0;c<RM;c++) Av[c]=F32_LDA(A+lda*(ii+c)+l);
                    for(int r=0;r<RN;r++) { F32_V Bv=F32_LDB(B+ldb*(jj+r)+l);
                        for(int c=0;c<RM;c++) Cv[r][c]=F32_MADD(Cv[r][c],Av[c],Bv); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) C[ldc*(jj+r)+(ii+c)]=F32_HSUM(Cv[r][c]);
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                F32_V Cv[F32_RM];
                for(int c=0;c<RM;c++) Cv[c]=F32_ZERO;
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) Cv[c]=F32_MADD(Cv[c],F32_LDA(A+lda*(ii+c)+l),F32_LDB(B+ldb*jj+l));
                for(int c=0;c<RM;c++) C[ldc*jj+(ii+c)]=F32_HSUM(Cv[c]);
            }
        }
    }
}
#undef F32_KN
#undef F32_RM
#undef F32_RN
#undef F32_BN
#undef F32_V
#undef F32_ZERO
#undef F32_LDA
#undef F32_LDB
#undef F32_MADD
#undef F32_HSUM

#elif defined(__ALTIVEC__)

typedef vector float v4sf;
#define F32_KN 4
#define F32_RM 4
#define F32_RN 3
#define F32_BN 24
#define F32_V v4sf
#define F32_LDA(p) vec_ld(0,p)
#define F32_LDB(p) vec_ld(0,p)
/* GCC 4.0.1 on Mac OS X 10.4 PPC lacks altivec.h vec_sld/vec_extract prototypes.
   Use C vector operators for madd; horizontal sum via store+scalar fallback. */
#define F32_MADD(a,b,c) ((a)*(b) + (c))

/* Horizontal sum of 4 floats in a vector: store to scratch, accumulate scalar */
static float sgemm_altivec_hsum(v4sf v) {
    float tmp[4];
    vec_st(v, 0, tmp);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

static void sgemm_f32_f32(int m, int n, int k, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
    const int64_t KN=F32_KN, RM=F32_RM, RN=F32_RN, BN=F32_BN;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    static char hsbuf[128];
    unsigned long hba=((unsigned long)hsbuf+63)/64*64;
    ((float *)hsbuf)[0]=1.0f;
    v4sf z=vec_splat(vec_ld(0-hba,(float *)hba),0);
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                v4sf Cv[F32_RN][F32_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=z;
                for (int64_t l=0;l<k;l+=KN) {
                    v4sf Av[F32_RM];
                    for(int c=0;c<RM;c++) Av[c]=vec_ld(0,A+lda*(ii+c)+l);
                    for(int r=0;r<RN;r++) { v4sf Bv=vec_ld(0,B+ldb*(jj+r)+l);
                        for(int c=0;c<RM;c++) Cv[r][c]=F32_MADD(Av[c],Bv,Cv[r][c]); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) {
                    C[ldc*(jj+r)+(ii+c)] = sgemm_altivec_hsum(Cv[r][c]); }
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                v4sf Cv[F32_RM];
                for(int c=0;c<RM;c++) Cv[c]=z;
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) Cv[c]=F32_MADD(vec_ld(0,A+lda*(ii+c)+l),vec_ld(0,B+ldb*jj+l),Cv[c]);
                for(int c=0;c<RM;c++) {
                    C[ldc*jj+(ii+c)] = sgemm_altivec_hsum(Cv[c]); }
            }
        }
    }
}
#undef F32_KN
#undef F32_RM
#undef F32_RN
#undef F32_BN
#undef F32_V
#undef F32_LDA
#undef F32_LDB
#undef F32_MADD

#elif defined(__riscv_v_intrinsic)

#define F32_RM 4
#define F32_RN 3
#define F32_BN 24

static void sgemm_f32_f32(int m, int n, int k, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
    const int64_t RM=F32_RM, RN=F32_RN, BN=F32_BN;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    const vlen_t vlmax=__riscv_vsetvlmax_e32m4();
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                vfloat32m4_t Cv[F32_RN][F32_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=__riscv_vfmv_v_f_f32m4(0,vlmax);
                for (int64_t l=0;l<k;l+=vlmax) {
                    vlen_t vl=__riscv_vsetvl_e32m4(vlmax);
                    vfloat32m4_t Av[F32_RM];
                    for(int c=0;c<RM;c++) Av[c]=__riscv_vle32_v_f32m4(A+lda*(ii+c)+l,vl);
                    for(int r=0;r<RN;r++) {
                        vfloat32m4_t Bv=__riscv_vle32_v_f32m4(B+ldb*(jj+r)+l,vl);
                        for(int c=0;c<RM;c++) Cv[r][c]=__riscv_vfmacc_vv_f32m4(Cv[r][c],Av[c],Bv,vl);
                    }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++)
                    C[ldc*(jj+r)+(ii+c)]=__riscv_vfmv_f_s_f32(__riscv_vfredusum_vs_f32m4_f32(Cv[r][c],__riscv_vfmv_v_f_f32(0,vlmax),vlmax));
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                vfloat32m4_t Cv[F32_RM];
                for(int c=0;c<RM;c++) Cv[c]=__riscv_vfmv_v_f_f32m4(0,vlmax);
                for (int64_t l=0;l<k;l+=vlmax) {
                    vlen_t vl=__riscv_vsetvl_e32m4(vlmax);
                    for(int c=0;c<RM;c++)
                        Cv[c]=__riscv_vfmacc_vv_f32m4(Cv[c],__riscv_vle32_v_f32m4(A+lda*(ii+c)+l,vl),__riscv_vle32_v_f32m4(B+ldb*jj+l,vl),vl);
                }
                for(int c=0;c<RM;c++)
                    C[ldc*jj+(ii+c)]=__riscv_vfmv_f_s_f32(__riscv_vfredusum_vs_f32m4_f32(Cv[c],__riscv_vfmv_v_f_f32(0,vlmax),vlmax));
            }
        }
    }
}
#undef F32_RM
#undef F32_RN
#undef F32_BN

#endif /* F32xF32 arch chain */

/* ============================================================
 * F16 x F32 kernels (F16 weights, F32 activations)
 * ============================================================ */

#if defined(__AVX512F__)

static void sgemm_f16_f32(int m, int n, int k, const uint16_t *A, int lda,
                           const float *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
#define SGEMM_KN 16
#define SGEMM_RM 4
#define SGEMM_RN 6
#define SGEMM_BN 12
    const int64_t KN=16, RM=4, RN=6, BN=12;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                __m512 Cv[SGEMM_RN][SGEMM_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=_mm512_setzero_ps();
                for (int64_t l=0;l<k;l+=KN) {
                    __m512 Av[SGEMM_RM];
                    for(int c=0;c<RM;c++) Av[c]=_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A+lda*(ii+c)+l)));
                    for(int r=0;r<RN;r++) { __m512 Bv=_mm512_loadu_ps(B+ldb*(jj+r)+l);
                        for(int c=0;c<RM;c++) Cv[r][c]=_mm512_fmadd_ps(Av[c],Bv,Cv[r][c]); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) C[ldc*(jj+r)+(ii+c)]=_mm512_reduce_add_ps(Cv[r][c]);
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                __m512 Cv[SGEMM_RM];
                for(int c=0;c<RM;c++) Cv[c]=_mm512_setzero_ps();
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) Cv[c]=_mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A+lda*(ii+c)+l))),_mm512_loadu_ps(B+ldb*jj+l),Cv[c]);
                for(int c=0;c<RM;c++) C[ldc*jj+(ii+c)]=_mm512_reduce_add_ps(Cv[c]);
            }
        }
    }
}

#elif defined(__AVX__) || defined(__AVX2__)
#if defined(__F16C__)
#if defined(__FMA__)
#define SGEMM_FMA(a,b,c) _mm256_fmadd_ps(a,b,c)
#else
#define SGEMM_FMA(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
#endif
static void sgemm_f16_f32(int m, int n, int k, const uint16_t *A, int lda,
                           const float *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
#define SGEMM_KN 8
#define SGEMM_RM 4
#define SGEMM_RN 3
#define SGEMM_BN 24
    const int64_t KN=8, RM=4, RN=3, BN=24;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                __m256 Cv[SGEMM_RN][SGEMM_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=_mm256_setzero_ps();
                for (int64_t l=0;l<k;l+=KN) {
                    __m256 Av[SGEMM_RM];
                    for(int c=0;c<RM;c++) Av[c]=_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(A+lda*(ii+c)+l)));
                    for(int r=0;r<RN;r++) { __m256 Bv=_mm256_loadu_ps(B+ldb*(jj+r)+l);
                        for(int c=0;c<RM;c++) Cv[r][c]=SGEMM_FMA(Av[c],Bv,Cv[r][c]); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) {
                    __m256 s=Cv[r][c]; __m128 lo=_mm256_castps256_ps128(s);
                    lo=_mm_add_ps(lo,_mm256_extractf128_ps(s,1));
                    lo=_mm_add_ps(lo,_mm_movehl_ps(lo,lo));
                    lo=_mm_add_ss(lo,_mm_movehdup_ps(lo));
                    C[ldc*(jj+r)+(ii+c)]=_mm_cvtss_f32(lo); }
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                __m256 Cv[SGEMM_RM];
                for(int c=0;c<RM;c++) Cv[c]=_mm256_setzero_ps();
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) Cv[c]=SGEMM_FMA(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(A+lda*(ii+c)+l))),_mm256_loadu_ps(B+ldb*jj+l),Cv[c]);
                for(int c=0;c<RM;c++) {
                    __m256 s=Cv[c]; __m128 lo=_mm256_castps256_ps128(s);
                    lo=_mm_add_ps(lo,_mm256_extractf128_ps(s,1));
                    lo=_mm_add_ps(lo,_mm_movehl_ps(lo,lo));
                    lo=_mm_add_ss(lo,_mm_movehdup_ps(lo));
                    C[ldc*jj+(ii+c)]=_mm_cvtss_f32(lo); }
            }
        }
    }
}
#undef SGEMM_FMA
#endif
#elif defined(__ARM_NEON)
static void sgemm_f16_f32(int m, int n, int k, const uint16_t *A, int lda,
                           const float *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
#define SGEMM_KN 4
#define SGEMM_RM 4
#define SGEMM_RN 6
#define SGEMM_BN 12
    const int64_t KN=4, RM=4, RN=6, BN=12;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                float32x4_t Cv[SGEMM_RN][SGEMM_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=vdupq_n_f32(0);
                for (int64_t l=0;l<k;l+=KN) {
                    float32x4_t Av[SGEMM_RM];
                    for(int c=0;c<RM;c++) Av[c]=vcvt_f32_f16(vld1_f16((const float16_t*)(A+lda*(ii+c)+l)));
                    for(int r=0;r<RN;r++) { float32x4_t Bv=vld1q_f32(B+ldb*(jj+r)+l);
                        for(int c=0;c<RM;c++) Cv[r][c]=vfmaq_f32(Cv[r][c],Av[c],Bv); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) C[ldc*(jj+r)+(ii+c)]=vaddvq_f32(Cv[r][c]);
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                float32x4_t Cv[SGEMM_RM];
                for(int c=0;c<RM;c++) Cv[c]=vdupq_n_f32(0);
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) Cv[c]=vfmaq_f32(Cv[c],vcvt_f32_f16(vld1_f16((const float16_t*)(A+lda*(ii+c)+l))),vld1q_f32(B+ldb*jj+l));
                for(int c=0;c<RM;c++) C[ldc*jj+(ii+c)]=vaddvq_f32(Cv[c]);
            }
        }
    }
}
#endif /* F16xF32 arch chain */

/* ============================================================
 * F16 x F16 kernels (F16 weights, F16 activations)
 * ============================================================ */

#if defined(__AVX512F__)
static void sgemm_f16_f16(int m, int n, int k, const uint16_t *A, int lda,
                           const uint16_t *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
#define SGEMM_KN 16
#define SGEMM_RM 4
#define SGEMM_RN 6
#define SGEMM_BN 12
    const int64_t KN=16, RM=4, RN=6, BN=12;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                __m512 Cv[SGEMM_RN][SGEMM_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=_mm512_setzero_ps();
                for (int64_t l=0;l<k;l+=KN) {
                    __m512 Av[SGEMM_RM];
                    for(int c=0;c<RM;c++) Av[c]=_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A+lda*(ii+c)+l)));
                    for(int r=0;r<RN;r++) { __m512 Bv=_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B+ldb*(jj+r)+l)));
                        for(int c=0;c<RM;c++) Cv[r][c]=_mm512_fmadd_ps(Av[c],Bv,Cv[r][c]); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) C[ldc*(jj+r)+(ii+c)]=_mm512_reduce_add_ps(Cv[r][c]);
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                __m512 Cv[SGEMM_RM];
                for(int c=0;c<RM;c++) Cv[c]=_mm512_setzero_ps();
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) Cv[c]=_mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A+lda*(ii+c)+l))),_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B+ldb*jj+l))),Cv[c]);
                for(int c=0;c<RM;c++) C[ldc*jj+(ii+c)]=_mm512_reduce_add_ps(Cv[c]);
            }
        }
    }
}
#elif defined(__AVX__) || defined(__AVX2__)
#if defined(__F16C__)
#if defined(__FMA__)
#define SGEMM_FMA(a,b,c) _mm256_fmadd_ps(a,b,c)
#else
#define SGEMM_FMA(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
#endif
static void sgemm_f16_f16(int m, int n, int k, const uint16_t *A, int lda,
                           const uint16_t *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
#define SGEMM_KN 8
#define SGEMM_RM 4
#define SGEMM_RN 3
#define SGEMM_BN 24
    const int64_t KN=8, RM=4, RN=3, BN=24;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                __m256 Cv[SGEMM_RN][SGEMM_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=_mm256_setzero_ps();
                for (int64_t l=0;l<k;l+=KN) {
                    __m256 Av[SGEMM_RM];
                    for(int c=0;c<RM;c++) Av[c]=_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(A+lda*(ii+c)+l)));
                    for(int r=0;r<RN;r++) { __m256 Bv=_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(B+ldb*(jj+r)+l)));
                        for(int c=0;c<RM;c++) Cv[r][c]=SGEMM_FMA(Av[c],Bv,Cv[r][c]); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) {
                    __m256 s=Cv[r][c]; __m128 lo=_mm256_castps256_ps128(s);
                    lo=_mm_add_ps(lo,_mm256_extractf128_ps(s,1));
                    lo=_mm_add_ps(lo,_mm_movehl_ps(lo,lo));
                    lo=_mm_add_ss(lo,_mm_movehdup_ps(lo));
                    C[ldc*(jj+r)+(ii+c)]=_mm_cvtss_f32(lo); }
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                __m256 Cv[SGEMM_RM];
                for(int c=0;c<RM;c++) Cv[c]=_mm256_setzero_ps();
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) Cv[c]=SGEMM_FMA(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(A+lda*(ii+c)+l))),_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(B+ldb*jj+l))),Cv[c]);
                for(int c=0;c<RM;c++) {
                    __m256 s=Cv[c]; __m128 lo=_mm256_castps256_ps128(s);
                    lo=_mm_add_ps(lo,_mm256_extractf128_ps(s,1));
                    lo=_mm_add_ps(lo,_mm_movehl_ps(lo,lo));
                    lo=_mm_add_ss(lo,_mm_movehdup_ps(lo));
                    C[ldc*jj+(ii+c)]=_mm_cvtss_f32(lo); }
            }
        }
    }
}
#undef SGEMM_FMA
#endif
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static void sgemm_f16_f16(int m, int n, int k, const uint16_t *A, int lda,
                           const uint16_t *B, int ldb, float *C, int ldc,
                           int ith, int nth) {
#define SGEMM_KN 8
#define SGEMM_RM 4
#define SGEMM_RN 6
#define SGEMM_BN 12
    const int64_t KN=8, RM=4, RN=6, BN=12;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1, jB=nB-(nB*sB-xt);
    int64_t nj=yt*nB, js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    for (int64_t j=js;j<je;j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0;bi<BM*RM;bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0;jj<jj1;jj+=RN) {
                float32x4_t Cv[SGEMM_RN][SGEMM_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=vdupq_n_f32(0);
                for (int64_t l=0;l<k;l+=KN) {
                    float16x8_t Av[SGEMM_RM];
                    for(int c=0;c<RM;c++) Av[c]=vld1q_f16((const float16_t*)(A+lda*(ii+c)+l));
                    for(int r=0;r<RN;r++) {
                        float16x8_t Bv=vld1q_f16((const float16_t*)(B+ldb*(jj+r)+l));
                        for(int c=0;c<RM;c++) {
                            float16x8_t prod=vmulq_f16(Av[c],Bv);
                            Cv[r][c]=vaddq_f32(Cv[r][c],vcvt_f32_f16(vget_low_f16(prod)));
                            Cv[r][c]=vaddq_f32(Cv[r][c],vcvt_f32_f16(vget_high_f16(prod)));
                        }
                    }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) C[ldc*(jj+r)+(ii+c)]=vaddvq_f32(Cv[r][c]);
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                float32x4_t Cv[SGEMM_RM];
                for(int c=0;c<RM;c++) Cv[c]=vdupq_n_f32(0);
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) {
                        float16x8_t prod=vmulq_f16(vld1q_f16((const float16_t*)(A+lda*(ii+c)+l)),vld1q_f16((const float16_t*)(B+ldb*jj+l)));
                        Cv[c]=vaddq_f32(Cv[c],vcvt_f32_f16(vget_low_f16(prod)));
                        Cv[c]=vaddq_f32(Cv[c],vcvt_f32_f16(vget_high_f16(prod)));
                    }
                for(int c=0;c<RM;c++) C[ldc*jj+(ii+c)]=vaddvq_f32(Cv[c]);
            }
        }
    }
}
#endif /* F16xF16 arch chain */

/* ============================================================
 * Dispatch function
 * ============================================================ */

int picolm_sgemm(int m, int n, int k,
                 const void *A, int lda,
                 const void *B, int ldb,
                 float *C, int ldc,
                 int Atype, int Btype,
                 int ith, int nth) {
    if (m < 4 || n < 2 || k < 1)
        return 0;

    if (Atype == GGUF_TYPE_F32 && Btype == GGUF_TYPE_F32) {
#if defined(__AVX512F__)
        if (k % 16 == 0 && m % 4 == 0) { sgemm_f32_f32(m,n,k,(const float*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#elif defined(__AVX__) || defined(__AVX2__)
        if (k % 8 == 0 && m % 4 == 0) { sgemm_f32_f32(m,n,k,(const float*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#elif defined(__SSE__)
        if (k % 4 == 0 && m % 4 == 0) { sgemm_f32_f32(m,n,k,(const float*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#elif defined(__ARM_NEON)
        if (n < 4) return 0;
        if (k % 4 == 0 && m % 4 == 0) { sgemm_f32_f32(m,n,k,(const float*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#elif defined(__ALTIVEC__)
        if (k % 4 == 0 && m % 4 == 0) { sgemm_f32_f32(m,n,k,(const float*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#elif defined(__riscv_v_intrinsic)
        if (m % 4 == 0) { sgemm_f32_f32(m,n,k,(const float*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#endif
    }
    if (Atype == GGUF_TYPE_F16 && Btype == GGUF_TYPE_F32) {
#if defined(__AVX512F__)
        if (k % 16 == 0 && m % 4 == 0) { sgemm_f16_f32(m,n,k,(const uint16_t*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#elif defined(__AVX__) || defined(__AVX2__)
#if defined(__F16C__)
        if (k % 8 == 0 && m % 4 == 0) { sgemm_f16_f32(m,n,k,(const uint16_t*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#endif
#elif defined(__ARM_NEON)
        if (n < 4) return 0;
        if (k % 4 == 0 && m % 4 == 0) { sgemm_f16_f32(m,n,k,(const uint16_t*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#endif
    }
    if (Atype == GGUF_TYPE_F16 && Btype == GGUF_TYPE_F16) {
#if defined(__AVX512F__)
        if (k % 16 == 0 && m % 4 == 0) { sgemm_f16_f16(m,n,k,(const uint16_t*)A,lda,(const uint16_t*)B,ldb,C,ldc,ith,nth); return 1; }
#elif defined(__AVX__) || defined(__AVX2__)
#if defined(__F16C__)
        if (k % 8 == 0 && m % 4 == 0) { sgemm_f16_f16(m,n,k,(const uint16_t*)A,lda,(const uint16_t*)B,ldb,C,ldc,ith,nth); return 1; }
#endif
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        if (n >= 8 && m % 4 == 0) { sgemm_f16_f16(m,n,k,(const uint16_t*)A,lda,(const uint16_t*)B,ldb,C,ldc,ith,nth); return 1; }
#endif
    }
    return 0;
}
