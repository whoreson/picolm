/*
 * type_dispatch.h - Unified type descriptor table for matmul dispatch.
 *
 * Replaces the scattered if-else chains across matmul(), matmul_batch(),
 * matmul_dual_batch(), and matmul_worker_f() with a single table.
 *
 * Adding a new quant type: add one entry to type_info[] and implement
 * the required kernels in quant.c/sgemm_*.c. No tensor.c dispatch edits.
 */
#ifndef PICOLM_TYPE_DISPATCH_H
#define PICOLM_TYPE_DISPATCH_H

#include "quant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum rows per interleaved block (R8 formats) */
#define TYPE_INFO_MAX_RPB 8

/* GEMV kernel: dot product of one weight row/block with one activation row.
 * For plain types (rows_per_block==1): writes 1 float to out.
 * For interleaved types (rows_per_block>1): writes rows_per_block floats.
 *
 * Signature: (weight_ptr, activation_ptr, n_values, out_ptr, nrows)
 *   weight_ptr     - start of weight data (one block or one row)
 *   activation_ptr - start of quantized activation data
 *   n_values       - number of values per row (shared dimension)
 *   out_ptr        - output floats (nrows entries)
 *   nrows          - number of rows being processed (rows_per_block for
 *                    interleaved, 1 for plain)
 */
typedef void (*type_gemv_fn)(const void *w, const void *a, int n, float *out, int nrows);

/* GEMM kernel: batched matrix multiply with tiled cache blocking.
 * Signature: (nrows, ncols, k, weight, activation, out, out_stride, ith, nth)
 *   nrows    - weight rows (output dimension d)
 *   ncols    - activation columns (batch size)
 *   k        - shared dimension (n)
 *   Returns rows processed, or 0 if not supported.
 */
typedef int (*type_gemm_fn)(int nrows, int ncols, int k,
                            const void *w, const void *a,
                            float *out, size_t bs, int ith, int nth);

/* Dequantize a single row from an interleaved block (for scalar fallback).
 * Only needed for rows_per_block > 1.
 * Signature: (block_ptr, dst, n_values, row_in_block)
 */
typedef void (*type_dequant_single_fn)(const void *wblock, float *dst, int n, int row_in_block);

typedef struct {
    /* Activation format: determines quantization function */
    act_fmt_t act_fmt;

    /* Rows per block: 1 = plain, 4 = R4, 8 = R8 */
    int rows_per_block;

    /* GEMV kernel (single activation, used by matmul decode + worker) */
    type_gemv_fn fn_gemv;

    /* GEMM kernel (batched, tiled). NULL if no GEMM for this type. */
    type_gemm_fn fn_gemm;

    /* Whether picolm_sgemm_d() supports this type (standard Q8_0 GEMM) */
    int has_qgemm_d;

    /* Dequantize single row from interleaved block. NULL for plain types. */
    type_dequant_single_fn fn_dequant_single;

    /* Whether activation must be pre-quantized (all except F32/F16) */
    int needs_qx;

    /* Whether this type uses the wptr_repacked global (Q4_0_8_8 only) */
    int use_repacked;

    /* Whether the old dual-batch scalar_vec_dot had an explicit handler for this
     * type (quantizing activations + calling type-specific vec_dot). If 0, the
     * old code fell through to vec_dot() with F32 activations. */
    int has_dual_scalar_path;
} type_info_t;

/* ---- Unified activation quantization ---- */

typedef struct {
    void *qbuf;       /* quantized activations [n_batch x qstride] */
    float *dbuf;      /* per-block FP32 deltas (Q8_0 only, appended after blocks) */
    size_t qstride;   /* bytes per activation row (excluding deltas) */
    size_t total_stride; /* bytes per activation row including deltas */
    act_fmt_t fmt;
    int n_batch;
    int n;
    int owned;        /* whether qbuf was malloc'd (vs scratch_buf) */
} quant_buf_t;

/* Quantize activations for a given weight type.
 * x: float32 activations [n_batch x n]
 * n_batch: number of activation rows
 * n: values per row
 * scratch: optional scratch buffer (if big enough, avoids malloc)
 * scratch_size: size of scratch buffer in bytes
 * Returns quant_buf_t with qbuf!=NULL on success, qbuf==NULL on failure.
 */
quant_buf_t quant_activations(gguf_type_t wtype, const float *x, int n_batch, int n,
                               void *scratch, size_t scratch_size);
/* use_amax_deltas: 1 = compute deltas from amax/127 (decode path),
 *                  0 = use fp16 round-trip from block d field (batched path) */
quant_buf_t quant_activations_ex(gguf_type_t wtype, const float *x, int n_batch, int n,
                                  void *scratch, size_t scratch_size, int use_amax_deltas);
void free_quant_buf(quant_buf_t *qb, void *scratch);

/* The table. Populated in type_dispatch.c. */
extern const type_info_t type_info_table[];

/* Lookup helper. Returns NULL for unknown/unsupported types. */
static inline const type_info_t *typeinfo(gguf_type_t t) {
    if (t < 0 || t >= GGUF_TYPE_COUNT) return NULL;
    const type_info_t *ti = &type_info_table[t];
    /* Check if this entry was populated (non-zero act_fmt is valid for F32=0,
     * so check fn_gemv or fn_gemm as a sentinel) */
    if (!ti->fn_gemv && !ti->fn_gemm && ti->act_fmt == ACT_FMT_F32 && ti->rows_per_block == 0)
        return NULL;
    return ti;
}

#ifdef __cplusplus
}
#endif

#endif /* PICOLM_TYPE_DISPATCH_H */
