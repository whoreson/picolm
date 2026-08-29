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

#ifdef __cplusplus
}
#endif

#endif
