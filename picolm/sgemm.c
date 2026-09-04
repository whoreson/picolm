// Port of llama.cpp llamafile/sgemm.cpp tinyBLAS tiled GEMM to C.
// MIT License (Copyright 2024 Mozilla Foundation).
// Tiled GEMM: C[n x m] = A^T[m x k] * B[k x n]
#ifndef PICOLM_NOSGEMM
#define PICOLM_NOSGEMM 0
#endif
#include "sgemm.h"
#include "quant.h"
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
/* Keep F32_MADD for F16 kernels below (same vector type v4sf) */

/* ============================================================
 * F16 x F32 kernel for Altivec (PPC G4, no hardware F16)
 * Convert F16->F32 via lookup table, then use F32 vector math.
 * ============================================================ */
static void sgemm_f16_f32_altivec(int m, int n, int k, const uint16_t *A, int lda,
                                   const float *B, int ldb, float *C, int ldc,
                                   int ith, int nth) {
#define F16F32_KN 4
#define F16F32_RM 4
#define F16F32_RN 3
#define F16F32_BN 24
    const int64_t KN=F16F32_KN, RM=F16F32_RM, RN=F16F32_RN, BN=F16F32_BN;
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
                v4sf Cv[F16F32_RN][F16F32_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=z;
                for (int64_t l=0;l<k;l+=KN) {
                    v4sf Av[F16F32_RM];
                    for(int c=0;c<RM;c++) {
                        /* Convert 4 F16 -> F32 via lookup table */
                        float tmp[4];
                        const uint16_t *ap = A+lda*(ii+c)+l;
                        tmp[0]=fp16_to_fp32_lookup(ap[0]);
                        tmp[1]=fp16_to_fp32_lookup(ap[1]);
                        tmp[2]=fp16_to_fp32_lookup(ap[2]);
                        tmp[3]=fp16_to_fp32_lookup(ap[3]);
                        Av[c]=vec_ld(0,tmp);
                    }
                    for(int r=0;r<RN;r++) { v4sf Bv=vec_ld(0,B+ldb*(jj+r)+l);
                        for(int c=0;c<RM;c++) Cv[r][c]=F32_MADD(Av[c],Bv,Cv[r][c]); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) {
                    C[ldc*(jj+r)+(ii+c)] = sgemm_altivec_hsum(Cv[r][c]); }
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                v4sf Cv[F16F32_RM];
                for(int c=0;c<RM;c++) Cv[c]=z;
                for (int64_t l=0;l<k;l+=KN) {
                    for(int c=0;c<RM;c++) {
                        float tmp[4];
                        const uint16_t *ap = A+lda*(ii+c)+l;
                        tmp[0]=fp16_to_fp32_lookup(ap[0]);
                        tmp[1]=fp16_to_fp32_lookup(ap[1]);
                        tmp[2]=fp16_to_fp32_lookup(ap[2]);
                        tmp[3]=fp16_to_fp32_lookup(ap[3]);
                        Cv[c]=F32_MADD(vec_ld(0,tmp),vec_ld(0,B+ldb*jj+l),Cv[c]);
                    }
                }
                for(int c=0;c<RM;c++) {
                    C[ldc*jj+(ii+c)] = sgemm_altivec_hsum(Cv[c]); }
            }
        }
    }
}
#undef F16F32_KN
#undef F16F32_RM
#undef F16F32_RN
#undef F16F32_BN

/* ============================================================
 * F16 x F16 kernel for Altivec
 * Both A and B converted F16->F32, then F32 vector math.
 * ============================================================ */
static void sgemm_f16_f16_altivec(int m, int n, int k, const uint16_t *A, int lda,
                                   const uint16_t *B, int ldb, float *C, int ldc,
                                   int ith, int nth) {
#define F16F16_KN 4
#define F16F16_RM 4
#define F16F16_RN 3
#define F16F16_BN 24
    const int64_t KN=F16F16_KN, RM=F16F16_RM, RN=F16F16_RN, BN=F16F16_BN;
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
                v4sf Cv[F16F16_RN][F16F16_RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=z;
                for (int64_t l=0;l<k;l+=KN) {
                    v4sf Av[F16F16_RM];
                    for(int c=0;c<RM;c++) {
                        float tmp[4];
                        const uint16_t *ap = A+lda*(ii+c)+l;
                        tmp[0]=fp16_to_fp32_lookup(ap[0]);
                        tmp[1]=fp16_to_fp32_lookup(ap[1]);
                        tmp[2]=fp16_to_fp32_lookup(ap[2]);
                        tmp[3]=fp16_to_fp32_lookup(ap[3]);
                        Av[c]=vec_ld(0,tmp);
                    }
                    for(int r=0;r<RN;r++) {
                        float tmp[4];
                        const uint16_t *bp = B+ldb*(jj+r)+l;
                        tmp[0]=fp16_to_fp32_lookup(bp[0]);
                        tmp[1]=fp16_to_fp32_lookup(bp[1]);
                        tmp[2]=fp16_to_fp32_lookup(bp[2]);
                        tmp[3]=fp16_to_fp32_lookup(bp[3]);
                        v4sf Bv=vec_ld(0,tmp);
                        for(int c=0;c<RM;c++) Cv[r][c]=F32_MADD(Av[c],Bv,Cv[r][c]);
                    }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) {
                    C[ldc*(jj+r)+(ii+c)] = sgemm_altivec_hsum(Cv[r][c]); }
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                v4sf Cv[F16F16_RM];
                for(int c=0;c<RM;c++) Cv[c]=z;
                for (int64_t l=0;l<k;l+=KN) {
                    /* B is shared across RM columns for this jj row */
                    float tmpb[4];
                    const uint16_t *bp = B+ldb*jj+l;
                    tmpb[0]=fp16_to_fp32_lookup(bp[0]);
                    tmpb[1]=fp16_to_fp32_lookup(bp[1]);
                    tmpb[2]=fp16_to_fp32_lookup(bp[2]);
                    tmpb[3]=fp16_to_fp32_lookup(bp[3]);
                    v4sf Bv=vec_ld(0,tmpb);
                    for(int c=0;c<RM;c++) {
                        float tmpa[4];
                        const uint16_t *ap = A+lda*(ii+c)+l;
                        tmpa[0]=fp16_to_fp32_lookup(ap[0]);
                        tmpa[1]=fp16_to_fp32_lookup(ap[1]);
                        tmpa[2]=fp16_to_fp32_lookup(ap[2]);
                        tmpa[3]=fp16_to_fp32_lookup(ap[3]);
                        Cv[c]=F32_MADD(vec_ld(0,tmpa),Bv,Cv[c]);
                    }
                }
                for(int c=0;c<RM;c++) {
                    C[ldc*jj+(ii+c)] = sgemm_altivec_hsum(Cv[c]); }
            }
        }
    }
}
#undef F16F16_KN
#undef F16F16_RM
#undef F16F16_RN
#undef F16F16_BN

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
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
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
 * BF16 x F32 / BF16 x BF16 kernels (BF16 weights)
 * BF16 dequant: zero-extend 16-bit to 32-bit, shift left 16, reinterpret as float.
 * AVX-512: _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(...)))
 *   converts 8 bf16 -> 8 f32 (cvtepu16: 8x16->8x32 zero-extend)
 * AVX2:    _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(...)))
 *   converts 8 bf16 -> 8 f32 (pmovzxwd is AVX2)
 * ============================================================ */

#if defined(__AVX512F__)
static void sgemm_bf16_f32(int m, int n, int k, const uint16_t *A, int lda,
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
                    for(int c=0;c<RM;c++) {
                        __m256i raw = _mm256_loadu_si256((__m256i*)(A+lda*(ii+c)+l));
                        Av[c] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
                    }
                    for(int r=0;r<RN;r++) { __m512 Bv=_mm512_loadu_ps(B+ldb*(jj+r)+l);
                        for(int c=0;c<RM;c++) Cv[r][c]=_mm512_fmadd_ps(Av[c],Bv,Cv[r][c]); }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) C[ldc*(jj+r)+(ii+c)]=_mm512_reduce_add_ps(Cv[r][c]);
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                __m512 Cv[SGEMM_RM];
                for(int c=0;c<RM;c++) Cv[c]=_mm512_setzero_ps();
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) {
                        __m256i raw = _mm256_loadu_si256((__m256i*)(A+lda*(ii+c)+l));
                        __m512 Av = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
                        Cv[c]=_mm512_fmadd_ps(Av,_mm512_loadu_ps(B+ldb*jj+l),Cv[c]);
                    }
                for(int c=0;c<RM;c++) C[ldc*jj+(ii+c)]=_mm512_reduce_add_ps(Cv[c]);
            }
        }
    }
}
#undef SGEMM_KN; #undef SGEMM_RM; #undef SGEMM_RN; #undef SGEMM_BN

static void sgemm_bf16_bf16(int m, int n, int k, const uint16_t *A, int lda,
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
                    for(int c=0;c<RM;c++) {
                        __m256i raw = _mm256_loadu_si256((__m256i*)(A+lda*(ii+c)+l));
                        Av[c] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
                    }
                    for(int r=0;r<RN;r++) {
                        __m256i raw = _mm256_loadu_si256((__m256i*)(B+ldb*(jj+r)+l));
                        __m512 Bv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
                        for(int c=0;c<RM;c++) Cv[r][c]=_mm512_fmadd_ps(Av[c],Bv,Cv[r][c]);
                    }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) C[ldc*(jj+r)+(ii+c)]=_mm512_reduce_add_ps(Cv[r][c]);
            }
            for (int64_t jj=jj1;jj<jj2;jj++) {
                __m512 Cv[SGEMM_RM];
                for(int c=0;c<RM;c++) Cv[c]=_mm512_setzero_ps();
                for (int64_t l=0;l<k;l+=KN)
                    for(int c=0;c<RM;c++) {
                        __m256i raw_a = _mm256_loadu_si256((__m256i*)(A+lda*(ii+c)+l));
                        __m512 Av = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw_a), 16));
                        __m256i raw_b = _mm256_loadu_si256((__m256i*)(B+ldb*jj+l));
                        __m512 Bv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw_b), 16));
                        Cv[c]=_mm512_fmadd_ps(Av,Bv,Cv[c]);
                    }
                for(int c=0;c<RM;c++) C[ldc*jj+(ii+c)]=_mm512_reduce_add_ps(Cv[c]);
            }
        }
    }
}
#undef SGEMM_KN; #undef SGEMM_RM; #undef SGEMM_RN; #undef SGEMM_BN

#elif defined(__AVX__) || defined(__AVX2__)
#if defined(__FMA__)
#define SGEMM_FMA(a,b,c) _mm256_fmadd_ps(a,b,c)
#else
#define SGEMM_FMA(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
#endif
static void sgemm_bf16_f32(int m, int n, int k, const uint16_t *A, int lda,
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
                    for(int c=0;c<RM;c++) {
                        __m128i raw = _mm_loadu_si128((__m128i*)(A+lda*(ii+c)+l));
                        Av[c] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw)));
                    }
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
                    for(int c=0;c<RM;c++) {
                        __m128i raw = _mm_loadu_si128((__m128i*)(A+lda*(ii+c)+l));
                        __m256 Av = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw)));
                        Cv[c]=SGEMM_FMA(Av,_mm256_loadu_ps(B+ldb*jj+l),Cv[c]);
                    }
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
#undef SGEMM_KN; #undef SGEMM_RM; #undef SGEMM_RN; #undef SGEMM_BN

static void sgemm_bf16_bf16(int m, int n, int k, const uint16_t *A, int lda,
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
                    for(int c=0;c<RM;c++) {
                        __m128i raw = _mm_loadu_si128((__m128i*)(A+lda*(ii+c)+l));
                        Av[c] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw)));
                    }
                    for(int r=0;r<RN;r++) {
                        __m128i raw = _mm_loadu_si128((__m128i*)(B+ldb*(jj+r)+l));
                        __m256 Bv = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw)));
                        for(int c=0;c<RM;c++) Cv[r][c]=SGEMM_FMA(Av[c],Bv,Cv[r][c]);
                    }
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
                    for(int c=0;c<RM;c++) {
                        __m128i raw_a = _mm_loadu_si128((__m128i*)(A+lda*(ii+c)+l));
                        __m256 Av = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_a)));
                        __m128i raw_b = _mm_loadu_si128((__m128i*)(B+ldb*jj+l));
                        __m256 Bv = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_b)));
                        Cv[c]=SGEMM_FMA(Av,Bv,Cv[c]);
                    }
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
#undef SGEMM_KN; #undef SGEMM_RM; #undef SGEMM_RN; #undef SGEMM_BN
#endif /* AVX2 F16C chain */

/* ============================================================
 * Quantized GEMM kernels (Q8_0 x Q8_0, Q4_0 x Q8_0, Q5_0 x Q8_0)
 * Port of llamafile tinyBLAS_Q0 to C.
 *
 * Semantics: C[n x m] = A^T[m x k_blocks] * B[k_blocks x n]
 * A is weights in quantized blocks (block_q8_0, block_q4_0, etc.)
 * B is pre-quantized activations in block_q8_0 format.
 * k_blocks = k / 32 (each block has 32 values).
 *
 * The sign trick: _mm256_sign_epi8(a,a) *_mm256_sign_epi8(b,a)
 * computes |a|*|b| with correct sign for signed int8 multiply
 * using unsigned multiply hardware (maddubs or dpbusd).
 * ============================================================ */

#if defined(__AVX2__) || defined(__AVX__)

/* Horizontal sum of __m256 -> float */
static float q8_hsum_f32(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    lo = _mm_add_ps(lo, _mm256_extractf128_ps(x, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

/* updot: unsigned int8 dot product -> 4x int32 -> 4x float */
static __m256 q8_updot_avx(__m256i u, __m256i s) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    __m256i res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#else
    __m256i res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#endif
    return _mm256_cvtepi32_ps(res);
}

/* Load 32 int8 values from block_q8_0 */
static __m256i q8_load_qs(const block_q8_0 *b) {
    return _mm256_loadu_si256((const __m256i *)b->qs);
}

/* Load 32 signed int8 values from block_q4_0 (nibble dequant, subtract 8) */
static __m256i q4_load_qs(const block_q4_0 *b) {
    __m128i x = _mm_loadu_si128((const __m128i *)b->qs);
    __m128i lo128 = _mm_and_si128(_mm_set1_epi8(15), x);
    __m128i hi128 = _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4));
    __m256i lo = _mm256_castsi128_si256(lo128);
    __m256i hi = _mm256_insertf128_si256(lo, hi128, 1);
    __m256i eight = _mm256_set1_epi8(8);
    return _mm256_sub_epi8(hi, eight);
}

/* Load 32 signed int8 values from block_q5_0 */
static __m256i q5_load_qs(const block_q5_0 *b) {
    __m128i x = _mm_loadu_si128((const __m128i *)b->qs);
    uint32_t qh32;
    memcpy(&qh32, b->qh, sizeof(uint32_t));
    __m128i hi_nib = _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4));
    __m128i lo_nib = _mm_and_si128(_mm_set1_epi8(15), x);
    __m256i lo = _mm256_castsi128_si256(lo_nib);
    __m256i xs = _mm256_insertf128_si256(lo, hi_nib, 1);
    /* Extract 5th bits from qh */
    __m256i qh256 = _mm256_set1_epi32(qh32);
    __m256i idx0 = _mm256_set_epi64x(0x0101010101010101ULL,0x0000000000000000ULL,
                                      0x0101010101010101ULL,0x0000000000000000ULL);
    __m256i idx1 = _mm256_set_epi64x(0x0303030303030303ULL,0x0202020202020202ULL,
                                      0x0303030303030303ULL,0x0202020202020202ULL);
    __m256i pat = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeULL);
    __m256i r0 = _mm256_andnot_si256(
        _mm256_cmpeq_epi8(pat, _mm256_or_si256(pat, _mm256_shuffle_epi8(qh256, idx0))),
        _mm256_set1_epi8((char)0xF0));
    __m256i r1 = _mm256_andnot_si256(
        _mm256_cmpeq_epi8(pat, _mm256_or_si256(pat, _mm256_shuffle_epi8(qh256, idx1))),
        _mm256_set1_epi8((char)0xF0));
    __m128i bits0 = _mm256_castsi256_si128(r0);
    __m128i bits1 = _mm256_extracti128_si256(r1, 1);
    __m256i qhbits = _mm256_insertf128_si256(
        _mm256_castsi128_si256(bits0), bits1, 1);
    return _mm256_or_si256(xs, qhbits);
}

/* Generic quantized GEMM for AVX2/AVX. RM=4 weight rows, RN=2 act rows. */
#define QGEMM_DEFINE(load_fn, blk_t)                                            \
static void sgemm_##load_fn(int m, int n, int k_blocks,                         \
                            const blk_t *A, int lda,                            \
                            const block_q8_0 *B, int ldb,                       \
                            float *C, int ldc,                                  \
                            int ith, int nth) {                                 \
    const int64_t RM=4, RN=2, BN=12;                                            \
    int64_t BM = (m >= RM*4*(int64_t)nth) ? 4 : (m%8==0) ? 2 : 1;              \
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);                     \
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1;               \
    int64_t jB=nB-(nB*sB-xt), nj=yt*nB;                                        \
    int64_t js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);                        \
    for (int64_t j=js; j<je; j++) {                                            \
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;                                     \
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);             \
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);             \
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;                                       \
        for (int64_t bi=0; bi<BM*RM; bi+=RM) {                                 \
            int64_t ii=iib+bi;                                                 \
            for (int64_t jj=jj0; jj<jj1; jj+=RN) {                             \
                __m256 Cv[RN][RM] = {};                                        \
                for (int64_t l=0; l<k_blocks; ++l) {                           \
                    __m256i a0=load_fn(A+lda*(ii+0)+l);                        \
                    __m256i a1=load_fn(A+lda*(ii+1)+l);                        \
                    __m256i a2=load_fn(A+lda*(ii+2)+l);                        \
                    __m256i a3=load_fn(A+lda*(ii+3)+l);                        \
                    float da0=fp16_to_fp32(A[lda*(ii+0)+l].d);                 \
                    float da1=fp16_to_fp32(A[lda*(ii+1)+l].d);                 \
                    float da2=fp16_to_fp32(A[lda*(ii+2)+l].d);                 \
                    float da3=fp16_to_fp32(A[lda*(ii+3)+l].d);                 \
                    for (int64_t jr=0; jr<RN; ++jr) {                          \
                        __m256i b=q8_load_qs(B+ldb*(jj+jr)+l);                 \
                        float db=fp16_to_fp32(B[ldb*(jj+jr)+l].d);             \
                        Cv[jr][0]=_mm256_add_ps(Cv[jr][0],                     \
                            _mm256_mul_ps(q8_updot_avx(                         \
                                _mm256_sign_epi8(a0,a0),                       \
                                _mm256_sign_epi8(b,a0)),                       \
                                _mm256_set1_ps(da0*db)));                      \
                        Cv[jr][1]=_mm256_add_ps(Cv[jr][1],                     \
                            _mm256_mul_ps(q8_updot_avx(                         \
                                _mm256_sign_epi8(a1,a1),                       \
                                _mm256_sign_epi8(b,a1)),                       \
                                _mm256_set1_ps(da1*db)));                      \
                        Cv[jr][2]=_mm256_add_ps(Cv[jr][2],                     \
                            _mm256_mul_ps(q8_updot_avx(                         \
                                _mm256_sign_epi8(a2,a2),                       \
                                _mm256_sign_epi8(b,a2)),                       \
                                _mm256_set1_ps(da2*db)));                      \
                        Cv[jr][3]=_mm256_add_ps(Cv[jr][3],                     \
                            _mm256_mul_ps(q8_updot_avx(                         \
                                _mm256_sign_epi8(a3,a3),                       \
                                _mm256_sign_epi8(b,a3)),                       \
                                _mm256_set1_ps(da3*db)));                      \
                    }                                                           \
                }                                                               \
                for (int64_t jr=0;jr<RN;++jr)                                  \
                    for (int64_t ir=0;ir<RM;++ir)                               \
                        C[ldc*(jj+jr)+(ii+ir)]=q8_hsum_f32(Cv[jr][ir]);        \
            }                                                                   \
            for (int64_t jj=jj1; jj<jj2; ++jj) {                               \
                __m256 Cv[RM] = {};                                            \
                for (int64_t l=0; l<k_blocks; ++l) {                           \
                    __m256i a0=load_fn(A+lda*(ii+0)+l);                        \
                    __m256i a1=load_fn(A+lda*(ii+1)+l);                        \
                    __m256i a2=load_fn(A+lda*(ii+2)+l);                        \
                    __m256i a3=load_fn(A+lda*(ii+3)+l);                        \
                    float da0=fp16_to_fp32(A[lda*(ii+0)+l].d);                 \
                    float da1=fp16_to_fp32(A[lda*(ii+1)+l].d);                 \
                    float da2=fp16_to_fp32(A[lda*(ii+2)+l].d);                 \
                    float da3=fp16_to_fp32(A[lda*(ii+3)+l].d);                 \
                    __m256i b=q8_load_qs(B+ldb*jj+l);                          \
                    float db=fp16_to_fp32(B[ldb*jj+l].d);                      \
                    Cv[0]=_mm256_add_ps(Cv[0],                                 \
                        _mm256_mul_ps(q8_updot_avx(                             \
                            _mm256_sign_epi8(a0,a0),                           \
                            _mm256_sign_epi8(b,a0)),                           \
                            _mm256_set1_ps(da0*db)));                          \
                    Cv[1]=_mm256_add_ps(Cv[1],                                 \
                        _mm256_mul_ps(q8_updot_avx(                             \
                            _mm256_sign_epi8(a1,a1),                           \
                            _mm256_sign_epi8(b,a1)),                           \
                            _mm256_set1_ps(da1*db)));                          \
                    Cv[2]=_mm256_add_ps(Cv[2],                                 \
                        _mm256_mul_ps(q8_updot_avx(                             \
                            _mm256_sign_epi8(a2,a2),                           \
                            _mm256_sign_epi8(b,a2)),                           \
                            _mm256_set1_ps(da2*db)));                          \
                    Cv[3]=_mm256_add_ps(Cv[3],                                 \
                        _mm256_mul_ps(q8_updot_avx(                             \
                            _mm256_sign_epi8(a3,a3),                           \
                            _mm256_sign_epi8(b,a3)),                           \
                            _mm256_set1_ps(da3*db)));                          \
                }                                                               \
                for (int64_t ir=0;ir<RM;++ir)                                   \
                    C[ldc*jj+(ii+ir)]=q8_hsum_f32(Cv[ir]);                     \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}

QGEMM_DEFINE(q8_load_qs, block_q8_0)
QGEMM_DEFINE(q4_load_qs, block_q4_0)
QGEMM_DEFINE(q5_load_qs, block_q5_0)

#endif /* AVX2/AVX */

/* ARM NEON / I8MM / DOTPROD path */
#if defined(__ARM_NEON)
#include <arm_neon.h>

static void sgemm_q8_q8_neon(int m, int n, int k_blocks,
                              const block_q8_0 *A, int lda,
                              const block_q8_0 *B, int ldb,
                              float *C, int ldc,
                              int ith, int nth) {
    const int64_t RM=4, RN=3, BN=12;
    int64_t BM=(m>=RM*4*(int64_t)nth)?4:(m%8==0)?2:1;
    int64_t yt=m/(RM*BM), xt=(n+RN-1)/RN, jR=xt-(xt*RN-n);
    int64_t nB=xt<BN?1:(xt+BN/2)/BN, sB=xt%nB==0?xt/nB:xt/nB+1;
    int64_t jB=nB-(nB*sB-xt), nj=yt*nB;
    int64_t js=JSTART(nj,ith,nth), je=JEND(nj,ith,nth);
    for (int64_t j=js; j<je; j++) {
        int64_t iib=(j%yt)*RM*BM, jb=j/yt;
        int64_t jr0=bloc_pos(jb,jB,sB), jrN=bloc_pos(jb+1,jB,sB);
        int64_t jj0=bloc_pos(jr0,jR,RN), jj2=bloc_pos(jrN,jR,RN);
        int64_t jj1=jj2<jR*RN?jj2:jR*RN;
        for (int64_t bi=0; bi<BM*RM; bi+=RM) {
            int64_t ii=iib+bi;
            for (int64_t jj=jj0; jj<jj1; jj+=RN) {
                float32x4_t Cv[RN][RM];
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++) Cv[r][c]=vdupq_n_f32(0);
                for (int64_t l=0; l<k_blocks; ++l) {
                    for (int64_t jr=0; jr<RN; ++jr) {
                        const block_q8_0 *br=B+ldb*(jj+jr)+l;
                        float32x4_t dr=vdupq_n_f32(fp16_to_fp32(br->d));
                        int8x16_t blo=vld1q_s8(br->qs);
                        int8x16_t bhi=vld1q_s8(br->qs+16);
                        for (int64_t ir=0; ir<RM; ++ir) {
                            const block_q8_0 *ar=A+lda*(ii+ir)+l;
                            float32x4_t scale=vmulq_n_f32(dr,fp16_to_fp32(ar->d));
                            int8x16_t alo=vld1q_s8(ar->qs);
                            int8x16_t ahi=vld1q_s8(ar->qs+16);
#if defined(__ARM_FEATURE_MATMUL_INT8)
                            int32x4_t acc=vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0),alo,blo),ahi,bhi);
                            float32x4_t s=vmulq_f32(vcvtq_f32_s32(acc),scale);
#elif defined(__ARM_FEATURE_DOTPROD)
                            int32x4_t acc=vdotq_s32(vdotq_s32(vdupq_n_s32(0),alo,blo),ahi,bhi);
                            float32x4_t s=vmulq_f32(vcvtq_f32_s32(acc),scale);
#else
                            /* Scalar fallback: plain NEON without DOTPROD/I8MM */
                            int32_t acc_s = 0;
                            for (int v = 0; v < 16; v++) acc_s += (int32_t)vgetq_lane_s8(alo,v) * vgetq_lane_s8(blo,v);
                            for (int v = 0; v < 16; v++) acc_s += (int32_t)vgetq_lane_s8(ahi,v) * vgetq_lane_s8(bhi,v);
                            float32x4_t s = vmulq_n_f32(scale, (float)acc_s);
#endif
                            Cv[jr][ir]=vaddq_f32(Cv[jr][ir],s);
                        }
                    }
                }
                for(int r=0;r<RN;r++) for(int c=0;c<RM;c++)
                    C[ldc*(jj+r)+(ii+c)]=vaddvq_f32(Cv[r][c]);
            }
            for (int64_t jj=jj1; jj<jj2; ++jj) {
                float32x4_t Cv[RM];
                for(int c=0;c<RM;c++) Cv[c]=vdupq_n_f32(0);
                for (int64_t l=0; l<k_blocks; ++l) {
                    const block_q8_0 *br=B+ldb*jj+l;
                    float32x4_t dr=vdupq_n_f32(fp16_to_fp32(br->d));
                    int8x16_t blo=vld1q_s8(br->qs);
                    int8x16_t bhi=vld1q_s8(br->qs+16);
                    for (int64_t ir=0; ir<RM; ++ir) {
                        const block_q8_0 *ar=A+lda*(ii+ir)+l;
                        float32x4_t scale=vmulq_n_f32(dr,fp16_to_fp32(ar->d));
                        int8x16_t alo=vld1q_s8(ar->qs);
                        int8x16_t ahi=vld1q_s8(ar->qs+16);
#if defined(__ARM_FEATURE_MATMUL_INT8)
                        int32x4_t acc=vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0),alo,blo),ahi,bhi);
                        float32x4_t s=vmulq_f32(vcvtq_f32_s32(acc),scale);
#elif defined(__ARM_FEATURE_DOTPROD)
                        int32x4_t acc=vdotq_s32(vdotq_s32(vdupq_n_s32(0),alo,blo),ahi,bhi);
                        float32x4_t s=vmulq_f32(vcvtq_f32_s32(acc),scale);
#else
                        /* Scalar fallback: plain NEON without DOTPROD/I8MM */
                        int32_t acc_s = 0;
                        for (int v = 0; v < 16; v++) acc_s += (int32_t)vgetq_lane_s8(alo,v) * vgetq_lane_s8(blo,v);
                        for (int v = 0; v < 16; v++) acc_s += (int32_t)vgetq_lane_s8(ahi,v) * vgetq_lane_s8(bhi,v);
                        float32x4_t s = vmulq_n_f32(scale, (float)acc_s);
#endif
                        Cv[ir]=vaddq_f32(Cv[ir],s);
                    }
                }
                for(int c=0;c<RM;c++) C[ldc*jj+(ii+c)]=vaddvq_f32(Cv[c]);
            }
        }
    }
}
#endif /* ARM_NEON */

/* ============================================================
 * Optimized quantized GEMM using _mm_cvtph_ps for delta conversion.
 * Based on llamafile tinyBLAS_Q0_AVX::gemm4xN.
 *
 * Key: pack 4x fp16 weight deltas into uint64_t, convert all 4 to fp32
 * with single _mm_cvtph_ps. Activation deltas pre-converted to float32
 * by tensor.c (B_d). No table lookups, no VLA, no scalar fp16->fp32.
 * RM=4 weight rows, RN=2 activation rows per tile.
 * ============================================================ */
#if defined(__AVX2__) && defined(__F16C__)

/* updot: sign-trick int8 dot product -> 4x int32 -> 4x float */
static __m256 qg_updot(__m256i u, __m256i s) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s));
#else
    return _mm256_cvtepi32_ps(
        _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s)));
#endif
}

/* Horizontal sum __m256 -> float (reuse q8_hsum_f32) */
/* q8_hsum_f32 already defined above */

/* Pack 4x uint16 deltas into uint64 for _mm_cvtph_ps (little-endian) */
static inline uint64_t qg_pack4_d(uint16_t d0, uint16_t d1, uint16_t d2, uint16_t d3) {
    return (uint64_t)d3 << 48 | (uint64_t)d2 << 32 | (uint64_t)d1 << 16 | (uint64_t)d0;
}

/* Load Q8 qs */
static __m256i qg_q8_qs(const block_q8_0 *b) {
    return _mm256_loadu_si256((const __m256i *)b->qs);
}

/* Load Q4_0 qs (nibble dequant, subtract 8) */
static __m256i qg_q4_qs(const block_q4_0 *b) {
    __m128i x = _mm_loadu_si128((const __m128i *)b->qs);
    __m128i lo = _mm_and_si128(_mm_set1_epi8(15), x);
    __m128i hi = _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4));
    return _mm256_sub_epi8(
        _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1),
        _mm256_set1_epi8(8));
}

/* Load Q5_0 qs (nibble + 5th bit extraction) */
static __m256i qg_q5_qs(const block_q5_0 *b) {
    __m128i x = _mm_loadu_si128((const __m128i *)b->qs);
    uint32_t qh32; memcpy(&qh32, b->qh, sizeof(uint32_t));
    __m128i lo = _mm_and_si128(_mm_set1_epi8(15), x);
    __m128i hi = _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4));
    __m256i xs = _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
    __m256i qh256 = _mm256_set1_epi32(qh32);
    __m256i idx0 = _mm256_set_epi64x(0x0101010101010101ULL,0x0000000000000000ULL,
                                      0x0101010101010101ULL,0x0000000000000000ULL);
    __m256i idx1 = _mm256_set_epi64x(0x0303030303030303ULL,0x0202020202020202ULL,
                                      0x0303030303030303ULL,0x0202020202020202ULL);
    __m256i pat = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeULL);
    __m256i r0 = _mm256_andnot_si256(
        _mm256_cmpeq_epi8(pat, _mm256_or_si256(pat, _mm256_shuffle_epi8(qh256, idx0))),
        _mm256_set1_epi8((char)0xF0));
    __m256i r1 = _mm256_andnot_si256(
        _mm256_cmpeq_epi8(pat, _mm256_or_si256(pat, _mm256_shuffle_epi8(qh256, idx1))),
        _mm256_set1_epi8((char)0xF0));
    __m256i qhbits = _mm256_insertf128_si256(
        _mm256_castsi128_si256(_mm256_castsi256_si128(r0)),
        _mm256_extracti128_si256(r1, 1), 1);
    return _mm256_or_si256(xs, qhbits);
}

/* Helper to get fp16 delta pointer from any block type */
#define QG_D_PTR(b) ((const uint16_t*)&(b)->d)

/* Compute one 4(weight rows) x 2(activation rows) tile at (ii,jj).
 * Shared by the main tiled loop and both tail loops below so there is
 * exactly one place that implements the actual math -- tiling logic
 * changes can't silently diverge from it again. */
#define QGEMM_TILE_4x2(load_qs_fn, blk_t, A, lda, B, ldb, B_d, ldb_d, C, ldc, ii, jj, k_blocks) do { \
    __m256 Cv[2][4];                                                            \
    for (int r=0;r<2;r++) for (int c=0;c<4;c++) Cv[r][c]=_mm256_setzero_ps();    \
    for (int64_t l = 0; l < (k_blocks); ++l) {                                  \
        uint64_t ad = qg_pack4_d(*QG_D_PTR((A)+(lda)*((ii)+0)+l),                \
                                 *QG_D_PTR((A)+(lda)*((ii)+1)+l),               \
                                 *QG_D_PTR((A)+(lda)*((ii)+2)+l),               \
                                 *QG_D_PTR((A)+(lda)*((ii)+3)+l));              \
        __m128 da = _mm_cvtph_ps(_mm_set_epi64x(0, ad));                        \
        __m256i a0 = load_qs_fn((A)+(lda)*((ii)+0)+l);                          \
        __m256i a1 = load_qs_fn((A)+(lda)*((ii)+1)+l);                          \
        __m256i a2 = load_qs_fn((A)+(lda)*((ii)+2)+l);                          \
        __m256i a3 = load_qs_fn((A)+(lda)*((ii)+3)+l);                          \
        __m256i b0 = qg_q8_qs((B) + (ldb)*((jj)+0) + l);                        \
        __m128 db0 = _mm_set1_ps((B_d)[(ldb_d)*((jj)+0) + l]);                  \
        __m256i b1 = qg_q8_qs((B) + (ldb)*((jj)+1) + l);                        \
        __m128 db1 = _mm_set1_ps((B_d)[(ldb_d)*((jj)+1) + l]);                  \
        __m128 _da_db0 = _mm_mul_ps(da, db0);                                    \
        __m128 _da_db1 = _mm_mul_ps(da, db1);                                    \
        __m256 dv0 = _mm256_broadcast_ps(&_da_db0);                             \
        __m256 dv1 = _mm256_broadcast_ps(&_da_db1);                             \
        Cv[0][0]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv0,dv0,0x00),               \
            qg_updot(_mm256_sign_epi8(a0,a0),_mm256_sign_epi8(b0,a0)),Cv[0][0]); \
        Cv[0][1]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv0,dv0,0x55),               \
            qg_updot(_mm256_sign_epi8(a1,a1),_mm256_sign_epi8(b0,a1)),Cv[0][1]); \
        Cv[0][2]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv0,dv0,0xAA),               \
            qg_updot(_mm256_sign_epi8(a2,a2),_mm256_sign_epi8(b0,a2)),Cv[0][2]); \
        Cv[0][3]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv0,dv0,0xFF),               \
            qg_updot(_mm256_sign_epi8(a3,a3),_mm256_sign_epi8(b0,a3)),Cv[0][3]); \
        Cv[1][0]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv1,dv1,0x00),               \
            qg_updot(_mm256_sign_epi8(a0,a0),_mm256_sign_epi8(b1,a0)),Cv[1][0]); \
        Cv[1][1]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv1,dv1,0x55),               \
            qg_updot(_mm256_sign_epi8(a1,a1),_mm256_sign_epi8(b1,a1)),Cv[1][1]); \
        Cv[1][2]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv1,dv1,0xAA),               \
            qg_updot(_mm256_sign_epi8(a2,a2),_mm256_sign_epi8(b1,a2)),Cv[1][2]); \
        Cv[1][3]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv1,dv1,0xFF),               \
            qg_updot(_mm256_sign_epi8(a3,a3),_mm256_sign_epi8(b1,a3)),Cv[1][3]); \
    }                                                                            \
    for (int64_t jr=0;jr<2;++jr)                                                \
        for (int64_t ir=0;ir<4;++ir)                                            \
            (C)[(ldc)*((jj)+jr)+((ii)+ir)]=q8_hsum_f32(Cv[jr][ir]);              \
} while (0)

/* 4x4 tile: 4 weight rows x 4 activation rows. Matches llama.cpp gemm4xN<4>. */
#define QGEMM_TILE_4x4(load_qs_fn, blk_t, A, lda, B, ldb, B_d, ldb_d, C, ldc, ii, jj, k_blocks) do { \
    __m256 Cv[4][4];                                                           \
    for (int r=0;r<4;r++) for (int c=0;c<4;c++) Cv[r][c]=_mm256_setzero_ps();   \
    for (int64_t l = 0; l < (k_blocks); ++l) {                                 \
        uint64_t ad = qg_pack4_d(*QG_D_PTR((A)+(lda)*((ii)+0)+l),               \
                                 *QG_D_PTR((A)+(lda)*((ii)+1)+l),              \
                                 *QG_D_PTR((A)+(lda)*((ii)+2)+l),              \
                                 *QG_D_PTR((A)+(lda)*((ii)+3)+l));             \
        __m128 da = _mm_cvtph_ps(_mm_set_epi64x(0, ad));                       \
        __m256i a0 = load_qs_fn((A)+(lda)*((ii)+0)+l);                         \
        __m256i a1 = load_qs_fn((A)+(lda)*((ii)+1)+l);                         \
        __m256i a2 = load_qs_fn((A)+(lda)*((ii)+2)+l);                         \
        __m256i a3 = load_qs_fn((A)+(lda)*((ii)+3)+l);                         \
        for (int64_t jr=0; jr<4; ++jr) {                                       \
            __m256i b = qg_q8_qs((B) + (ldb)*((jj)+jr) + l);                   \
            __m128 db = _mm_set1_ps((B_d)[(ldb_d)*((jj)+jr) + l]);             \
            __m128 _da_db = _mm_mul_ps(da, db);                                 \
            __m256 dv = _mm256_broadcast_ps(&_da_db);                           \
            Cv[jr][0]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv,dv,0x00),            \
                qg_updot(_mm256_sign_epi8(a0,a0),_mm256_sign_epi8(b,a0)),Cv[jr][0]); \
            Cv[jr][1]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv,dv,0x55),            \
                qg_updot(_mm256_sign_epi8(a1,a1),_mm256_sign_epi8(b,a1)),Cv[jr][1]); \
            Cv[jr][2]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv,dv,0xAA),            \
                qg_updot(_mm256_sign_epi8(a2,a2),_mm256_sign_epi8(b,a2)),Cv[jr][2]); \
            Cv[jr][3]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv,dv,0xFF),            \
                qg_updot(_mm256_sign_epi8(a3,a3),_mm256_sign_epi8(b,a3)),Cv[jr][3]); \
        }                                                                        \
    }                                                                              \
    for (int64_t jr=0;jr<4;++jr)                                                  \
        for (int64_t ir=0;ir<4;++ir)                                              \
            (C)[(ldc)*((jj)+jr)+((ii)+ir)]=q8_hsum_f32(Cv[jr][ir]);                \
} while (0)

/* Scalar-ish fallback for a single weight row x single activation column,
 * used only in the rare m%4 tail (kept simple and correct, not fast --
 * it runs on at most 3 rows out of m). */
#define QGEMM_CELL_1x1(load_qs_fn, blk_t, A, lda, B, ldb, B_d, ldb_d, C, ldc, ii, jj, k_blocks) do { \
    __m256 acc = _mm256_setzero_ps();                                           \
    for (int64_t l = 0; l < (k_blocks); ++l) {                                  \
        float da = fp16_to_fp32(*QG_D_PTR((A)+(lda)*(ii)+l));                    \
        float db = (B_d)[(ldb_d)*(jj) + l];                                     \
        __m256i a = load_qs_fn((A)+(lda)*(ii)+l);                               \
        __m256i b = qg_q8_qs((B) + (ldb)*(jj) + l);                             \
        __m256 dot = qg_updot(_mm256_sign_epi8(a,a), _mm256_sign_epi8(b,a));     \
        acc = _mm256_fmadd_ps(_mm256_set1_ps(da*db), dot, acc);                 \
    }                                                                            \
    (C)[(ldc)*(jj)+(ii)] = q8_hsum_f32(acc);                                    \
} while (0)

/* 4(weight rows) x N(activation rows, N=1..3) tail tile.
 * Vectorized over weight rows (4 at a time via CV accumulators),
 * but handles variable activation row count for the n%4 tail.
 * This is ~100x faster than QGEMM_CELL_1x1 for the n-tail case
 * because it reuses weight data across all leftover activation rows. */
#define QGEMM_TILE_4xN(load_qs_fn, blk_t, A, lda, B, ldb, B_d, ldb_d, C, ldc, ii, jj, n_tail, k_blocks) do { \
    __m256 Cv[4][4]; /* Cv[jr][ir], jr=0..n_tail-1, ir=0..3 */                \
    for (int jr=0;jr<(n_tail);jr++) for (int c=0;c<4;c++) Cv[jr][c]=_mm256_setzero_ps(); \
    for (int64_t l = 0; l < (k_blocks); ++l) {                                 \
        uint64_t ad = qg_pack4_d(*QG_D_PTR((A)+(lda)*((ii)+0)+l),               \
                                 *QG_D_PTR((A)+(lda)*((ii)+1)+l),              \
                                 *QG_D_PTR((A)+(lda)*((ii)+2)+l),              \
                                 *QG_D_PTR((A)+(lda)*((ii)+3)+l));             \
        __m128 da = _mm_cvtph_ps(_mm_set_epi64x(0, ad));                       \
        __m256i a0 = load_qs_fn((A)+(lda)*((ii)+0)+l);                         \
        __m256i a1 = load_qs_fn((A)+(lda)*((ii)+1)+l);                         \
        __m256i a2 = load_qs_fn((A)+(lda)*((ii)+2)+l);                         \
        __m256i a3 = load_qs_fn((A)+(lda)*((ii)+3)+l);                         \
        for (int64_t jr=0; jr<(n_tail); ++jr) {                                \
            __m256i b = qg_q8_qs((B) + (ldb)*((jj)+jr) + l);                   \
            __m128 db = _mm_set1_ps((B_d)[(ldb_d)*((jj)+jr) + l]);             \
            __m128 _da_db = _mm_mul_ps(da, db);                                 \
            __m256 dv = _mm256_broadcast_ps(&_da_db);                           \
            Cv[jr][0]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv,dv,0x00),            \
                qg_updot(_mm256_sign_epi8(a0,a0),_mm256_sign_epi8(b,a0)),Cv[jr][0]); \
            Cv[jr][1]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv,dv,0x55),            \
                qg_updot(_mm256_sign_epi8(a1,a1),_mm256_sign_epi8(b,a1)),Cv[jr][1]); \
            Cv[jr][2]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv,dv,0xAA),            \
                qg_updot(_mm256_sign_epi8(a2,a2),_mm256_sign_epi8(b,a2)),Cv[jr][2]); \
            Cv[jr][3]=_mm256_fmadd_ps(_mm256_shuffle_ps(dv,dv,0xFF),            \
                qg_updot(_mm256_sign_epi8(a3,a3),_mm256_sign_epi8(b,a3)),Cv[jr][3]); \
        }                                                                        \
    }                                                                              \
    for (int64_t jr=0;jr<(n_tail);++jr)                                           \
        for (int64_t ir=0;ir<4;++ir)                                              \
            (C)[(ldc)*((jj)+jr)+((ii)+ir)]=q8_hsum_f32(Cv[jr][ir]);                \
} while (0)

/* mnpack-style tiling: main body is a flat grid of ytiles x xtiles 4x2
 * tiles, split evenly across [0,nth) by flat tile index -- no bloc_pos,
 * no uneven remainder distribution, no per-tile lookup overhead. Every
 * tile is exactly the same shape and cost, so the split is exact.
 * m%4 and n%2 tails (m, n need not be multiples of 4/2) are handled by
 * ith==0 alone after the main body's barrier-free completion; they are
 * a tiny fraction of total work and not worth splitting further. */
#define QGEMM_D_IMPL(load_qs_fn, blk_t)                                         \
static void sgemm_##load_qs_fn##_d(int m, int n, int k_blocks,                   \
                                   const blk_t *A, int lda,                     \
                                   const block_q8_0 *B, int ldb,                \
                                   const float *B_d, int ldb_d,                 \
                                   float *C, int ldc,                           \
                                   int ith, int nth) {                          \
    int64_t ytiles = m / 4;                                                     \
    int64_t xtiles = n / 2;                                                     \
    int64_t n_tail = n - xtiles * 2;                                            \
    /* Merge n-tail into the tile grid so it gets parallelized. */              \
    { int64_t xtiles_ext = xtiles + (n_tail > 0 ? 1 : 0);                       \
      int64_t tiles = ytiles * xtiles_ext;                                      \
      if (tiles > 0) {                                                          \
          int64_t duty = (tiles + nth - 1) / nth;                               \
          int64_t start = duty * ith;                                           \
          int64_t end = start + duty;                                           \
          if (end > tiles) end = tiles;                                         \
          for (int64_t job = start; job < end; ++job) {                         \
              int64_t ii = (job / xtiles_ext) * 4;  /* weight rows */           \
              int64_t xt = job % xtiles_ext;                                    \
              if (xt < xtiles) {                                                \
                  QGEMM_TILE_4x2(load_qs_fn, blk_t, A, lda, B, ldb, B_d, ldb_d, \
                                 C, ldc, ii, xt * 2, k_blocks);                  \
              } else {                                                          \
                  QGEMM_TILE_4xN(load_qs_fn, blk_t, A, lda, B, ldb, B_d, ldb_d, \
                                 C, ldc, ii, xtiles * 2, n_tail, k_blocks);      \
              }                                                                  \
          }                                                                      \
      }                                                                          \
    }                                                                            \
    if (ith == 0) {                                                             \
        /* m-tail: leftover weight rows when m%4 != 0, all n columns. */        \
        for (int64_t ii = ytiles * 4; ii < m; ++ii) {                           \
            for (int64_t jj = 0; jj < n; ++jj) {                                \
                QGEMM_CELL_1x1(load_qs_fn, blk_t, A, lda, B, ldb, B_d, ldb_d,    \
                               C, ldc, ii, jj, k_blocks);                        \
            }                                                                    \
        }                                                                        \
    }                                                                            \
}

QGEMM_D_IMPL(qg_q8_qs, block_q8_0)
QGEMM_D_IMPL(qg_q4_qs, block_q4_0)
QGEMM_D_IMPL(qg_q5_qs, block_q5_0)

/* RN=4 variant: 4x4 tiles. Uses x-major traversal (llama.cpp style):
 * iterate over xtiles (activation columns) first within each job,
 * then ytiles (weight rows). This keeps weight data in cache across
 * consecutive tiles, which is optimal when m >> n (typical prefill shape). */
#define QGEMM_D4_IMPL(load_qs_fn, blk_t)                                       \
static void sgemm_##load_qs_fn##_d4(int m, int n, int k_blocks,                 \
                                    const blk_t *A, int lda,                    \
                                    const block_q8_0 *B, int ldb,               \
                                    const float *B_d, int ldb_d,                \
                                    float *C, int ldc,                          \
                                    int ith, int nth) {                         \
    int64_t ytiles = m / 4;                                                     \
    int64_t xtiles = n / 4;                                                     \
    int64_t n_tail = n - xtiles * 4;                                            \
    /* Merge n-tail into the tile grid so it gets parallelized.                  \
     * Each ytile row has (xtiles + 1) tiles when n_tail > 0.                   \
     * The last tile in each ytile row uses QGEMM_TILE_4xN. */                 \
    { int64_t xtiles_ext = xtiles + (n_tail > 0 ? 1 : 0);                       \
      int64_t tiles = ytiles * xtiles_ext;                                      \
      if (tiles > 0) {                                                          \
          int64_t duty = (tiles + nth - 1) / nth;                               \
          int64_t start = duty * ith;                                           \
          int64_t end = start + duty;                                           \
          if (end > tiles) end = tiles;                                         \
          for (int64_t job = start; job < end; ++job) {                         \
              int64_t ii = (job / xtiles_ext) * 4;  /* weight rows */           \
              int64_t xt = job % xtiles_ext;                                    \
              if (xt < xtiles) {                                                \
                  QGEMM_TILE_4x4(load_qs_fn, blk_t, A, lda, B, ldb, B_d, ldb_d, \
                                 C, ldc, ii, xt * 4, k_blocks);                  \
              } else {                                                          \
                  QGEMM_TILE_4xN(load_qs_fn, blk_t, A, lda, B, ldb, B_d, ldb_d, \
                                 C, ldc, ii, xtiles * 4, n_tail, k_blocks);      \
              }                                                                  \
          }                                                                      \
      }                                                                          \
    }                                                                            \
    if (ith == 0) {                                                             \
        /* m-tail: leftover weight rows when m%4 != 0, all n columns. */        \
        for (int64_t ii = ytiles * 4; ii < m; ++ii) {                           \
            for (int64_t jj = 0; jj < n; ++jj) {                                \
                QGEMM_CELL_1x1(load_qs_fn, blk_t, A, lda, B, ldb, B_d, ldb_d,    \
                               C, ldc, ii, jj, k_blocks);                        \
            }                                                                    \
        }                                                                        \
    }                                                                            \
}

QGEMM_D4_IMPL(qg_q8_qs, block_q8_0)
QGEMM_D4_IMPL(qg_q4_qs, block_q4_0)
QGEMM_D4_IMPL(qg_q5_qs, block_q5_0)

#endif /* AVX2+F16C */

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
        if (k % 4 == 0 && m % 4 == 0 && !PICOLM_NOSGEMM) { sgemm_f32_f32(m,n,k,(const float*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
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
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        if (n < 4) return 0;
        if (k % 4 == 0 && m % 4 == 0) { sgemm_f16_f32(m,n,k,(const uint16_t*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#elif defined(__ALTIVEC__)
        if (k % 4 == 0 && m % 4 == 0 && !PICOLM_NOSGEMM) { sgemm_f16_f32_altivec(m,n,k,(const uint16_t*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
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
#elif defined(__ALTIVEC__)
        if (k % 4 == 0 && m % 4 == 0 && !PICOLM_NOSGEMM) { sgemm_f16_f16_altivec(m,n,k,(const uint16_t*)A,lda,(const uint16_t*)B,ldb,C,ldc,ith,nth); return 1; }
#endif
    }

    /* BF16 weights: A=BF16 (uint16), B=F32 or BF16 */
    if (Atype == GGUF_TYPE_BF16) {
        if (Btype == GGUF_TYPE_F32) {
#if defined(__AVX512F__)
            if (k % 16 == 0 && m % 4 == 0) { sgemm_bf16_f32(m,n,k,(const uint16_t*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#elif defined(__AVX__) || defined(__AVX2__)
            if (k % 8 == 0 && m % 4 == 0) { sgemm_bf16_f32(m,n,k,(const uint16_t*)A,lda,(const float*)B,ldb,C,ldc,ith,nth); return 1; }
#endif
        }
        if (Btype == GGUF_TYPE_BF16) {
#if defined(__AVX512F__)
            if (k % 16 == 0 && m % 4 == 0) { sgemm_bf16_bf16(m,n,k,(const uint16_t*)A,lda,(const uint16_t*)B,ldb,C,ldc,ith,nth); return 1; }
#elif defined(__AVX__) || defined(__AVX2__)
            if (k % 8 == 0 && m % 4 == 0) { sgemm_bf16_bf16(m,n,k,(const uint16_t*)A,lda,(const uint16_t*)B,ldb,C,ldc,ith,nth); return 1; }
#endif
        }
    }

    /* Quantized GEMM: A=quantized weights, B=Q8_0 pre-quantized activations
     * k is number of blocks (k_blocks), each block has 32 values. */
    /* TODO: Add IQ4_NL GEMM path (Atype==GGUF_TYPE_IQ4_NL, Btype==GGUF_TYPE_Q8_0).
     * IQ4_NL uses a 16-entry LUT (kvalues_iq4nl) for dequant instead of linear
     * (qs-8)*d scaling. Requires a new kernel with LUT-based dequant inside the
     * GEMM tile. PicoLM currently lacks IQ4_NL vec_dot/quant support, so this
     * is blocked on adding IQ4_NL to quant.c first.
     * llama.cpp reference: tinyBLAS_Q0_AVX<block_iq4_nl, block_q8_0, float> in
     * ggml/src/ggml-cpu/llamafile/sgemm.cpp case GGML_TYPE_IQ4_NL. */

    /* Quantized GEMM: A=quantized weights, B=Q8_0 pre-quantized activations
     * k is number of blocks (k_blocks), each block has 32 values. */
    if (Btype == GGUF_TYPE_Q8_0 && m % 4 == 0 && n >= 2 && k >= 1) {
        if (Atype == GGUF_TYPE_Q8_0) {
#if defined(__AVX2__) || defined(__AVX__)
            sgemm_q8_load_qs(m, n, k, (const block_q8_0*)A, lda, (const block_q8_0*)B, ldb, C, ldc, ith, nth);
            return 1;
#elif defined(__ARM_NEON)
            sgemm_q8_q8_neon(m, n, k, (const block_q8_0*)A, lda, (const block_q8_0*)B, ldb, C, ldc, ith, nth);
            return 1;
#endif
        } else if (Atype == GGUF_TYPE_Q4_0) {
#if defined(__AVX2__) || defined(__AVX__)
            sgemm_q4_load_qs(m, n, k, (const block_q4_0*)A, lda, (const block_q8_0*)B, ldb, C, ldc, ith, nth);
            return 1;
#endif
        } else if (Atype == GGUF_TYPE_Q5_0) {
#if defined(__AVX2__) || defined(__AVX__)
            sgemm_q5_load_qs(m, n, k, (const block_q5_0*)A, lda, (const block_q8_0*)B, ldb, C, ldc, ith, nth);
            return 1;
#endif
        }
    }

    return 0;
}

/* Dispatch with pre-converted activation deltas (B_d).
 * ldb_d = stride in floats for B_d (typically k_blocks).
 * Uses _mm_cvtph_ps for weight delta conversion when F16C available. */
int picolm_sgemm_d(int m, int n, int k_blocks,
                   const void *A, int lda,
                   const block_q8_0 *B, int ldb,
                   const float *B_d, int ldb_d,
                   float *C, int ldc,
                   int Atype,
                   int ith, int nth) {
    if (m < 4 || n < 2 || k_blocks < 1)
        return 0;
#if defined(__AVX2__) && defined(__F16C__)
    /* Prefer 4x4 tiles (RN=4) when n >= 4: better activation cache reuse */
    if (Atype == GGUF_TYPE_Q8_0) {
        if (n >= 4) { sgemm_qg_q8_qs_d4(m, n, k_blocks, (const block_q8_0*)A, lda, B, ldb, B_d, ldb_d, C, ldc, ith, nth); return 1; }
        sgemm_qg_q8_qs_d(m, n, k_blocks, (const block_q8_0*)A, lda, B, ldb, B_d, ldb_d, C, ldc, ith, nth);
        return 1;
    } else if (Atype == GGUF_TYPE_Q4_0) {
        if (n >= 4) { sgemm_qg_q4_qs_d4(m, n, k_blocks, (const block_q4_0*)A, lda, B, ldb, B_d, ldb_d, C, ldc, ith, nth); return 1; }
        sgemm_qg_q4_qs_d(m, n, k_blocks, (const block_q4_0*)A, lda, B, ldb, B_d, ldb_d, C, ldc, ith, nth);
        return 1;
    } else if (Atype == GGUF_TYPE_Q5_0) {
        if (n >= 4) { sgemm_qg_q5_qs_d4(m, n, k_blocks, (const block_q5_0*)A, lda, B, ldb, B_d, ldb_d, C, ldc, ith, nth); return 1; }
        sgemm_qg_q5_qs_d(m, n, k_blocks, (const block_q5_0*)A, lda, B, ldb, B_d, ldb_d, C, ldc, ith, nth);
        return 1;
    }
#endif
    return 0;
}
