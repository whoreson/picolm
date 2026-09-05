#pragma once
#ifndef PICOLM_SGEMM_H
#define PICOLM_SGEMM_H

/*
 * Port of llama.cpp llamafile/sgemm.cpp tinyBLAS tiled GEMM to C.
 * MIT License (Copyright 2024 Mozilla Foundation).
 *
 * Tiled GEMM: C[n x m] = A^T[m x k] * B[k x n]
 * Column-major semantics.
 * Returns 1 if handled, 0 if caller should fall back to gemv.
 *
 * Quantized GEMM support:
 *   Atype=GGUF_TYPE_Q8_0, Btype=GGUF_TYPE_Q8_0 -> AVX2 sign-trick + maddubs/dpbusd
 *   Atype=GGUF_TYPE_Q4_0, Btype=GGUF_TYPE_Q8_0 -> AVX2 nibble dequant + maddubs
 *   Atype=GGUF_TYPE_Q5_0, Btype=GGUF_TYPE_Q8_0 -> AVX2 Q5 dequant + maddubs
 *   k is number of blocks (each block = 32 values).
 */

#include "quant.h"

#ifdef __cplusplus
extern "C" {
#endif

int picolm_sgemm(int m, int n, int k,
                 const void *A, int lda,
                 const void *B, int ldb,
                 float *C, int ldc,
                 int Atype, int Btype,
                 int ith, int nth);

/* Quantized GEMM with pre-converted activation deltas.
 * B_d is float32 delta array, ldb_d stride in floats per activation row. */
int picolm_sgemm_d(int m, int n, int k_blocks,
                   const void *A, int lda,
                   const block_q8_0 *B, int ldb,
                   const float *B_d, int ldb_d,
                   float *C, int ldc,
                   int Atype,
                   int ith, int nth);

/* Interleaved Q4_0_8_8 GEMM: C[nr][nc] = A[nr][k] @ B[nc][k]^T
 * nr = activation rows (n_batch), nc = weight rows (d), k = shared dim
 * Returns number of rows computed (aligned to 16). */
int sgemm_q4_0x8_q8_0x4(int nr, int nc, int k,
        const void *vx, const void *vy, float *s, size_t bs);

#ifdef __cplusplus
}
#endif

#endif
