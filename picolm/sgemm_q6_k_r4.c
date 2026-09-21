/* ================================================================
 * Q6_K_R4 x Q8_K GEMM/GEMV kernel
 * ================================================================
 * Port of llama.cpp ik branch iqk_gemm_kquants.cpp:
 *   mul_mat_q6_k_r4_q8_k
 *
 * Weights: block_q6_K_R4 (4-row interleaved Q6_K, 840 bytes/group)
 * Activations: block_q8_K (plain, one per activation row)
 *
 * The actual per-group dot product (vec_dot_q6_K_R4_q8_K, AVX2 +
 * portable scalar fallback) lives in quant.c next to the rest of the
 * Q6_K family of kernels; this file just tiles it over row groups and
 * activation columns, following the same structure as
 * sgemm_q4_0_r8.c's sgemm_q4_0_r8_q8_0_avx2.
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "quant.h"

int sgemm_q6_k_r4_q8_k(int nrows, int ncols, int k,
                        const void *vx, const void *vy,
                        float *out, size_t bs,
                        int ith, int nth) {
    if (nrows < 4 || ncols < 1 || k <= 0 || k % 256 != 0 || nrows % 4 != 0) {
        return 0;
    }

    /* Per-row-equivalent byte stride: gguf_type_row_size(GGUF_TYPE_Q6_K_R4, k)
     * == sizeof(block_q6_K) * (k/256). Multiplying by 4 gives the real byte
     * size of one interleaved block_q6_K_R4 row group (== sizeof(block_q6_K_R4)
     * * (k/256), since sizeof(block_q6_K_R4) == 4*sizeof(block_q6_K)). */
    const size_t row_bytes = gguf_type_row_size(GGUF_TYPE_Q6_K_R4, k);
    const size_t group_bytes = row_bytes * 4;
    const size_t q8k_row_bytes = gguf_type_row_size(GGUF_TYPE_Q8_K, k);

    int start_col = (ncols * ith) / nth;
    int end_col = (ncols * (ith + 1)) / nth;
    if (end_col <= start_col) return 0;

    for (int w4 = 0; w4 < nrows / 4; w4++) {
        const char *group_base = (const char *)vx + (size_t)w4 * group_bytes;
        for (int c = start_col; c < end_col; c++) {
            const char *qy = (const char *)vy + (size_t)c * q8k_row_bytes;
            float *out_base = out + w4 * 4 + c * bs;
            vec_dot_q6_K_R4_q8_K(group_base, qy, k, out_base);
        }
    }
    return (nrows / 4) * 4;
}
