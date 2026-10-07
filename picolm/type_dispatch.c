/*
 * type_dispatch.c - Unified type descriptor table population.
 *
 * One entry per GGUF quantization type. The table is indexed by type ID
 * (sparse, up to GGUF_TYPE_COUNT=400).
 *
 * Wrapper functions normalize the diverse vec_dot signatures in quant.c
 * to the uniform type_gemv_fn signature: (w, a, n, out, nrows).
 */
#include "type_dispatch.h"
#include "sgemm.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ---- GEMV wrappers for float-returning vec_dot functions ---- */

/* Q8_0 with FP32 deltas: qx_d is appended after the Q8_0 blocks.
 * The activation pointer a points to block_q8_0 array, deltas follow at
 * offset (n/32)*sizeof(block_q8_0). */
static void gemv_q8_0_deltas(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows; /* always 1 for plain Q8_0 */
    int nb = n / 32;
    const float *qx_d = (const float *)((const char *)a + (size_t)nb * sizeof(block_q8_0));
    *out = vec_dot_q8_0_q8_0_deltas(a, qx_d, w, n);
}

static void gemv_q4_0(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_q4_0_q8_0(w, a, n);
}

static void gemv_iq4_nl(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_iq4_nl_q8_0(w, a, n);
}

static void gemv_q5_1(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_q5_1_q8_0(w, a, n);
}

static void gemv_q4_K(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_q4_K_q8_K(w, a, n);
}

static void gemv_q5_K(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_q5_K_q8_K(w, a, n);
}

static void gemv_q6_K(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_q6_K_q8_K(w, a, n);
}

static void gemv_q3_K(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_q3_K_q8_K(w, a, n);
}

static void gemv_q2_K(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_q2_K_q8_K(w, a, n);
}

static void gemv_q1_0(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_q1_0_q8_0(w, a, n);
}

static void gemv_q2_0(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_q2_0_q8_0(w, a, n);
}

static void gemv_q6_0(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_q6_0_q8_0(w, a, n);
}

static void gemv_iq2_k(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_iq2_k_q8_k(w, a, n);
}

static void gemv_iq3_k(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_iq3_k_q8_k(w, a, n);
}

static void gemv_iq4_k(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_iq4_k_q8_k(w, a, n);
}

static void gemv_iq6_k(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_iq6_k_q8_k(w, a, n);
}

static void gemv_iq4_xs(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    *out = vec_dot_iq4_xs_q8_k(w, a, n);
}

/* ---- R4 wrappers: batch4 functions have matching signature ---- */

/* R4 batch4 functions already have (vx, vy, n, out) writing 4 floats.
 * We need to add the nrows parameter. */

static void gemv_iq2_k_r4(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    vec_dot_iq2_k_r4_q8_k_batch4(w, a, n, out);
}

static void gemv_iq3_k_r4(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    vec_dot_iq3_k_r4_q8_k_batch4(w, a, n, out);
}

static void gemv_iq4_k_r4(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    vec_dot_iq4_k_r4_q8_k_batch4(w, a, n, out);
}

static void gemv_iq4_nl_r4(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    vec_dot_iq4_nl_r4_q8_0_batch4(w, a, n, out);
}

static void gemv_q4_k_r4(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    vec_dot_q4_k_r4_q8_k_batch4(w, a, n, out);
}

static void gemv_q6_k_r4(const void *w, const void *a, int n, float *out, int nrows) {
    (void)nrows;
    vec_dot_q6_K_R4_q8_K(w, a, n, out);
}

/* ---- R8 wrappers: already have matching (vx, wy, n, out, nrows) signature ---- */
/* vec_dot_q4_0_r8_q8_2_avx2, vec_dot_q8_k_r8_q8_k_avx2, vec_dot_q4_0x8_q8_0_avx2
 * all match type_gemv_fn directly. */

/* ---- Q4_0_4_4 / Q4_0_4_8 wrappers (4 rows, (vx, wy, n, out, nrows)) ---- */
/* vec_dot_q4_0x4_q8_0 and vec_dot_q4_0x4_4x8_q8_0 already match type_gemv_fn. */

/* ---- Q4I_0_8_8 ---- */
/* vec_dot_q4i_0x8_q8_0 has (vx, wy, n, out, nrows) signature. */

/* ---- F32/F16: no GEMV (use generic vec_dot fallback) ---- */

/* ---- The table ---- */

const type_info_t type_info_table[] = {
    /* Q8_0: plain, Q8_0 activations with FP32 deltas */
    [GGUF_TYPE_Q8_0] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 1,
        .fn_gemv = gemv_q8_0_deltas,
        .fn_gemm = NULL, /* uses picolm_sgemm_d */
        .has_qgemm_d = 1,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 1,
    },
    /* Q4_0: plain, Q8_0 activations */
    [GGUF_TYPE_Q4_0] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 1,
        .fn_gemv = gemv_q4_0,
        .fn_gemm = NULL, /* uses picolm_sgemm_d */
        .has_qgemm_d = 1,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 1,
    },
    /* IQ4_NL: plain, Q8_0 activations. Uses threaded_worker with fn_gemv. */
    [GGUF_TYPE_IQ4_NL] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 1,
        .fn_gemv = gemv_iq4_nl,
        .fn_gemm = NULL,
        .has_qgemm_d = 1,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* Q5_1: plain, Q8_0 activations */
    [GGUF_TYPE_Q5_1] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 1,
        .fn_gemv = gemv_q5_1,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* Q1_0: plain, Q8_0 activations */
    [GGUF_TYPE_Q1_0] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 1,
        .fn_gemv = gemv_q1_0,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 1,
    },
    /* Q2_0: plain, Q8_0 activations */
    [GGUF_TYPE_Q2_0] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 1,
        .fn_gemv = gemv_q2_0,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 1,
    },
    /* Q6_0: plain, Q8_0 activations */
    [GGUF_TYPE_Q6_0] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 1,
        .fn_gemv = gemv_q6_0,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* Q4_K: plain, Q8_K activations. GEMM uses picolm_sgemm_d_q4k (different sig). */
    [GGUF_TYPE_Q4_K] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 1,
        .fn_gemv = gemv_q4_K,
        .fn_gemm = NULL, /* handled in matmul_batch via picolm_sgemm_d_q4k */
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* Q5_K: plain, Q8_K activations */
    [GGUF_TYPE_Q5_K] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 1,
        .fn_gemv = gemv_q5_K,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 1,
    },
    /* Q6_K: plain, Q8_K activations */
    [GGUF_TYPE_Q6_K] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 1,
        .fn_gemv = gemv_q6_K,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 1,
    },
    /* Q3_K: plain, Q8_K activations */
    [GGUF_TYPE_Q3_K] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 1,
        .fn_gemv = gemv_q3_K,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 1,
    },
    /* Q2_K: plain, Q8_K activations */
    [GGUF_TYPE_Q2_K] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 1,
        .fn_gemv = gemv_q2_K,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 1,
    },
    /* IQ4_XS: plain, Q8_K activations */
    [GGUF_TYPE_IQ4_XS] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 1,
        .fn_gemv = gemv_iq4_xs,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* IQ2_K: plain, Q8_K activations */
    [GGUF_TYPE_IQ2_K] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 1,
        .fn_gemv = gemv_iq2_k,
        .fn_gemm = sgemm_iq2_k_q8_k_avx2,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* IQ3_K: plain, Q8_K activations */
    [GGUF_TYPE_IQ3_K] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 1,
        .fn_gemv = gemv_iq3_k,
        .fn_gemm = sgemm_iq3_k_q8_k_avx2,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* IQ4_K: plain, Q8_K activations */
    [GGUF_TYPE_IQ4_K] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 1,
        .fn_gemv = gemv_iq4_k,
        .fn_gemm = sgemm_iq4_k_q8_k_avx2,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* IQ6_K: plain, Q8_K activations */
    [GGUF_TYPE_IQ6_K] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 1,
        .fn_gemv = gemv_iq6_k,
        .fn_gemm = sgemm_iq6_k_q8_k_avx2,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* ---- R4 interleaved types ---- */
    [GGUF_TYPE_IQ2_K_R4] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 4,
        .fn_gemv = gemv_iq2_k_r4,
        .fn_gemm = sgemm_iq2_k_r4_q8_k_avx2,
        .has_qgemm_d = 0,
        .fn_dequant_single = (type_dequant_single_fn)dequantize_row_iq2_k_r4_single,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    [GGUF_TYPE_IQ3_K_R4] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 4,
        .fn_gemv = gemv_iq3_k_r4,
        .fn_gemm = sgemm_iq3_k_r4_q8_k_avx2,
        .has_qgemm_d = 0,
        .fn_dequant_single = (type_dequant_single_fn)dequantize_row_iq3_k_r4_single,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    [GGUF_TYPE_IQ4_K_R4] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 4,
        .fn_gemv = gemv_iq4_k_r4,
        .fn_gemm = sgemm_iq4_k_r4_q8_k_avx2,
        .has_qgemm_d = 0,
        .fn_dequant_single = (type_dequant_single_fn)dequantize_row_iq4_k_r4_single,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    [GGUF_TYPE_IQ4_NL_R4] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 4,
        .fn_gemv = gemv_iq4_nl_r4,
        .fn_gemm = sgemm_iq4_nl_r4_q8_0_avx2,
        .has_qgemm_d = 0,
        .fn_dequant_single = (type_dequant_single_fn)dequantize_row_iq4_nl_r4_single,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    [GGUF_TYPE_Q4_K_R4] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 4,
        .fn_gemv = gemv_q4_k_r4,
        .fn_gemm = sgemm_q4_k_r4_q8_k_avx2,
        .has_qgemm_d = 0,
        .fn_dequant_single = (type_dequant_single_fn)dequantize_row_q4_k_r4_single,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    [GGUF_TYPE_Q6_K_R4] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 4,
        .fn_gemv = gemv_q6_k_r4,
        .fn_gemm = sgemm_q6_k_r4_q8_k,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL, /* no _single variant; use full-block dequant */
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* ---- R8 interleaved types ---- */
    [GGUF_TYPE_Q4_0_R8] = {
        .act_fmt = ACT_FMT_Q8_2,
        .rows_per_block = 8,
        .fn_gemv = vec_dot_q4_0_r8_q8_2_avx2,
        .fn_gemm = sgemm_q4_0_r8_q8_2_avx2,
        .has_qgemm_d = 0,
        .fn_dequant_single = dequantize_row_q4_0_r8_single,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    [GGUF_TYPE_Q8_K_R8] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 8,
        .fn_gemv = vec_dot_q8_k_r8_q8_k_avx2,
        .fn_gemm = sgemm_q8_k_r8_q8_k_avx2,
        .has_qgemm_d = 0,
        .fn_dequant_single = dequantize_row_q8_k_r8_single,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* ---- Q4_0_4_4 / Q4_0_4_8: 4-row interleaved (DOTPROD/AVX2) ---- */
    [GGUF_TYPE_Q4_0_4_4] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 4,
        .fn_gemv = vec_dot_q4_0x4_q8_0,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL, /* handled in worker */
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    [GGUF_TYPE_Q4_0_4_8] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 4,
        .fn_gemv = vec_dot_q4_0x4_4x8_q8_0,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL, /* handled in worker */
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* Q4_0_8_8: 8-row interleaved. GEMM uses special block_q8_0x4 activation format
     * (not block_q8_0), so it cannot use the unified quant_activations_ex() path.
     * Falls back to old matmul_batch for batched path. */
    [GGUF_TYPE_Q4_0_8_8] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 8,
        .fn_gemv = vec_dot_q4_0x8_q8_0_avx2,
        .fn_gemm = NULL, /* block_q8_0x4 activation, not unified */
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 1,
        .has_dual_scalar_path = 0,
    },
    /* Q4I_0_8_8: 8-row pre-dequantized int8. Same GEMM issue as Q4_0_8_8. */
    [GGUF_TYPE_Q4I_0_8_8] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 8,
        .fn_gemv = vec_dot_q4i_0x8_q8_0,
        .fn_gemm = NULL, /* block_q8_0x4 activation, not unified */
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 1,
    },
    /* ---- F32/F16: no quantization, no GEMV wrapper (use generic fallback) ---- */
    [GGUF_TYPE_F32] = {
        .act_fmt = ACT_FMT_F32,
        .rows_per_block = 1,
        .fn_gemv = NULL, /* generic vec_dot */
        .fn_gemm = NULL, /* uses picolm_sgemm */
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 0,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    [GGUF_TYPE_F16] = {
        .act_fmt = ACT_FMT_F32,
        .rows_per_block = 1,
        .fn_gemv = NULL, /* generic vec_dot */
        .fn_gemm = NULL, /* uses picolm_sgemm */
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 0,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    [GGUF_TYPE_BF16] = {
        .act_fmt = ACT_FMT_F32,
        .rows_per_block = 1,
        .fn_gemv = NULL,
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 0,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* Q5_0: plain, Q8_0 activations (generic fallback in matmul, sgemm in batch) */
    [GGUF_TYPE_Q5_0] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 1,
        .fn_gemv = NULL, /* no dedicated AVX2 GEMV in tensor.c, uses generic */
        .fn_gemm = NULL, /* picolm_sgemm handles Q5_0 */
        .has_qgemm_d = 1, /* picolm_sgemm_d supports Q5_0 */
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* Q8_K: plain, Q8_K activations (generic fallback in matmul) */
    [GGUF_TYPE_Q8_K] = {
        .act_fmt = ACT_FMT_Q8_K,
        .rows_per_block = 1,
        .fn_gemv = NULL, /* no dedicated GEMV wrapper in tensor.c */
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
    /* Q8_0_R8: not yet implemented in tensor.c, but table entry for completeness */
    [GGUF_TYPE_Q8_0_R8] = {
        .act_fmt = ACT_FMT_Q8_0,
        .rows_per_block = 8,
        .fn_gemv = NULL, /* TODO */
        .fn_gemm = NULL,
        .has_qgemm_d = 0,
        .fn_dequant_single = NULL,
        .needs_qx = 1,
        .use_repacked = 0,
        .has_dual_scalar_path = 0,
    },
};

/* ---- quant_activations implementation ---- */

/* Compute FP32 deltas from the quantized Q8_0 blocks (FP16 delta round-trip).
 * This matches the old matmul_batch behavior: dbuf[i] = fp16_to_fp32(blk[i].d)
 * The deltas are appended after the Q8_0 blocks in the buffer. */
static void compute_q8_0_deltas(const void *qblocks, int n, float *deltas) {
    int nb = n / 32;
    const block_q8_0 *blk = (const block_q8_0 *)qblocks;
    for (int bi = 0; bi < nb; bi++) {
        deltas[bi] = fp16_to_fp32(blk[bi].d);
    }
}

quant_buf_t quant_activations(gguf_type_t wtype, const float *x, int n_batch, int n,
                               void *scratch, size_t scratch_size) {
    /* Default: batched mode uses fp16 round-trip deltas */
    return quant_activations_ex(wtype, x, n_batch, n, scratch, scratch_size, 0);
}

quant_buf_t quant_activations_ex(gguf_type_t wtype, const float *x, int n_batch, int n,
                                  void *scratch, size_t scratch_size, int use_amax_deltas) {
    quant_buf_t qb = {0};
    const type_info_t *ti = typeinfo(wtype);
    if (!ti || !ti->needs_qx) {
        qb.fmt = ACT_FMT_F32;
        return qb;
    }

    qb.fmt = ti->act_fmt;
    qb.n_batch = n_batch;
    qb.n = n;

    switch (qb.fmt) {
    case ACT_FMT_Q8_0: {
        size_t blocks_size = (size_t)n_batch * (n / 32) * sizeof(block_q8_0);
        size_t deltas_size = (size_t)n_batch * (n / 32) * sizeof(float);
        size_t total = blocks_size + deltas_size;
        qb.qstride = (n / 32) * sizeof(block_q8_0);
        qb.total_stride = qb.qstride + (n / 32) * sizeof(float);

        if (scratch && total <= scratch_size) {
            qb.qbuf = scratch;
            qb.owned = 0;
        } else {
            qb.qbuf = malloc(total);
            qb.owned = 1;
        }
        if (!qb.qbuf) return qb;

        qb.dbuf = (float *)((char *)qb.qbuf + blocks_size);
        for (int b = 0; b < n_batch; b++) {
            const float *xb = x + (size_t)b * n;
            block_q8_0 *dst = (block_q8_0 *)((char *)qb.qbuf + (size_t)b * qb.qstride);
            quantize_row_q8_0(xb, dst, n);
            float *deltas = qb.dbuf + (size_t)b * (n / 32);
            if (use_amax_deltas) {
                /* Decode path: compute from amax (matches old matmul() behavior) */
                int nb = n / 32;
                for (int bi = 0; bi < nb; bi++) {
                    float amax = 0.0f;
                    for (int j = 0; j < 32; j++) {
                        float v = xb[bi * 32 + j];
                        if (v < 0) v = -v;
                        if (v > amax) amax = v;
                    }
                    deltas[bi] = amax / 127.0f;
                }
            } else {
                /* Batched path: use fp16 round-trip (matches old matmul_batch behavior) */
                compute_q8_0_deltas(dst, n, deltas);
            }
        }
        break;
    }
    case ACT_FMT_Q8_K: {
        /* Q8_K: blocks only, no deltas */
        qb.qstride = (n / 256) * sizeof(block_q8_K);
        qb.total_stride = qb.qstride;
        size_t total = (size_t)n_batch * qb.qstride;

        if (scratch && total <= scratch_size) {
            qb.qbuf = scratch;
            qb.owned = 0;
        } else {
            qb.qbuf = malloc(total);
            qb.owned = 1;
        }
        if (!qb.qbuf) return qb;

        for (int b = 0; b < n_batch; b++) {
            const float *xb = x + (size_t)b * n;
            block_q8_K *dst = (block_q8_K *)((char *)qb.qbuf + (size_t)b * qb.qstride);
            quantize_row_q8_K(xb, dst, n);
        }
        break;
    }
    case ACT_FMT_Q8_2: {
        /* Q8_2: blocks only (contains FP16 delta + int16 sum) */
        qb.qstride = (n / 32) * sizeof(block_q8_2);
        qb.total_stride = qb.qstride;
        size_t total = (size_t)n_batch * qb.qstride;

        if (scratch && total <= scratch_size) {
            qb.qbuf = scratch;
            qb.owned = 0;
        } else {
            qb.qbuf = malloc(total);
            qb.owned = 1;
        }
        if (!qb.qbuf) return qb;

        for (int b = 0; b < n_batch; b++) {
            const float *xb = x + (size_t)b * n;
            block_q8_2 *dst = (block_q8_2 *)((char *)qb.qbuf + (size_t)b * qb.qstride);
            quantize_row_q8_2(xb, dst, n);
        }
        break;
    }
    case ACT_FMT_F32:
    default:
        /* No quantization */
        qb.fmt = ACT_FMT_F32;
        break;
    }

    return qb;
}

void free_quant_buf(quant_buf_t *qb, void *scratch) {
    if (qb->qbuf && qb->owned && qb->qbuf != scratch) {
        free(qb->qbuf);
    }
    qb->qbuf = NULL;
    qb->dbuf = NULL;
    qb->owned = 0;
}
