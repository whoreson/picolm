/* test_quant.c -- Cross-platform quantization unit tests
 *
 * Catches SIMD implementation bugs by comparing platform dequantize against
 * a pure-C scalar reference. Each test constructs known quantized blocks
 * and verifies the SIMD dequantize produces identical float results.
 *
 * The canonical failure: Q5_K NEON qh bit interleaving bug (commit e3e378c).
 * NEON used qh_bit/qh_bit<<4 for low/high nibbles; correct is interleaved
 * qh_bit/qh_bit<<1 (shared per-block, advancing by 2 each sub-iteration).
 *
 * Compile: cc -O2 -o test_quant test_quant.c quant.c sgemm.c -lm
 * Run:     ./test_quant [-v]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include "quant.h"

static int verbose = 0;
static int tests_run, tests_pass, tests_fail;
/* Floor on how many assertions must run. Bump when adding suites; never lower
 * it. Guards against a test_* function being orphaned from main(). */
#define MIN_EXPECTED_TESTS 8
static void pass_t(const char *n) { tests_pass++; if (verbose) printf("  PASS %s\n", n); }
static void fail_t(const char *n) { tests_fail++; fprintf(stderr, "  FAIL %s\n", n); }

/* Scalar FP16->FP32 (IEEE 754) */
static float ref_fp16(uint16_t h) {
    union { uint32_t u; float f; } r;
    uint32_t s = (h >> 15) & 1; int e = (h >> 10) & 0x1F; int m = h & 0x3FF;
    if (e == 0) {
        if (m == 0) { r.u = s << 31; return r.f; }
        r.u = (s << 31) | (0x70-14) << 23; m <<= 13;
        while ((m & (1<<23)) == 0) { m <<= 1; r.u -= 1<<23; }
        r.u |= m & 0x7FFFFF; return r.f;
    }
    if (e == 31) { r.u = (s<<31) | 0x7F800000 | (m<<13); return r.f; }
    r.u = (s<<31) | (e + 112) << 23 | m << 13; return r.f;
}

/* Inline get_scale_min_k4 from quant.c (static there, not in header) */
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    if (j < 4) { *sc = q[j] & 63; *mn = q[j+4] & 63; }
    else { *sc = (uint8_t)((q[j+4] & 0xF) | ((q[j-4]>>6)<<4));
            *mn = (uint8_t)((q[j+4]>>4) | ((q[j]>>6)<<4)); }
}

/* ================================================================
 * Reference Q5_K dequant (matches GGUF spec, scalar)
 * ================================================================
 * Q5_K layout per block (176 bytes, 256 values):
 *   d(FP16), dm(FP16), qs[128], qh[32], scales[12]
 *
 * Processing: 4 groups of 64 values each. Within each group:
 *   First 32: low nibbles of qs[l], 5th bit from qh[l]&u1
 *   Next 32: high nibbles of qs[l], 5th bit from qh[l]&u2
 *   u1 starts at 1, u2 at 2. After each 64-value group: u1<<=2, u2<<=2.
 *   The SAME qh[0..31] bytes are used for all 4 groups.
 *   Val = (5bit * d1/d2) - m1/m2 per sub-block.
 */
static void ref_dequant_q5_K(const block_q5_K *b, float *dst) {
    float d = ref_fp16(b->d), dm = ref_fp16(b->dm);
    const uint8_t *ql = b->qs, *qh = b->qh;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(j/32+0, b->scales, &sc, &m);
        float d1 = d*sc, m1 = dm*m;
        get_scale_min_k4(j/32+1, b->scales, &sc, &m);
        float d2 = d*sc, m2 = dm*m;
        for (int l = 0; l < 32; l++) *dst++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) *dst++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32; u1 <<= 2; u2 <<= 2;
    }
}

/* Build Q5_K block with known values (0..255 linear).
 * Uses d=1.0, dm=0.0, all scales=1, all mins=0.
 * Correctly handles the qh byte sharing across groups. */
static void make_q5_K_block(block_q5_K *b, const int vals[256]) {
    memset(b, 0, sizeof(*b));
    b->d = 0x3C00; /* 1.0 */
    /* scales: all=1, mins: all=0 */
    b->scales[0]=1; b->scales[1]=1; b->scales[2]=1; b->scales[3]=1;
    b->scales[8]=1; b->scales[9]=1;
    /* Build qs and qh.
     * The dequant reads: for each group j (0,64,128,192):
     *   low: qs[ql_off..ql_off+31] low nibbles, qh[0..31] with u1
     *   high: qs[ql_off..ql_off+31] high nibbles, qh[0..31] with u2
     *   ql_off = 0,32,64,96 for the 4 groups
     *   u1 = 1,4,16,64; u2 = 2,8,32,128
     *
     * So qh[l] byte holds 8 bits total: 2 per group, one for low and one for high nibble.
     * For group g (0..3), sub-block sb (0=low,1=high):
     *   qh_bit_position = g*2 + sb
     *   qh[l] bit at that position = 5th bit of vals[group*64 + sb*32 + l]
     */
    for (int g = 0; g < 4; g++) {
        for (int l = 0; l < 32; l++) {
            int v_lo = vals[g * 64 + l] & 0x1F;
            int v_hi = vals[g * 64 + 32 + l] & 0x1F;
            int qs_idx = g * 32 + l;
            b->qs[qs_idx] = (v_lo & 0xF) | ((v_hi & 0xF) << 4);
            if (v_lo & 0x10) b->qh[l] |= (1 << (g * 2));
            if (v_hi & 0x10) b->qh[l] |= (1 << (g * 2 + 1));
        }
    }
}

/* ================================================================
 * Reference Q3_K dequant
 * ================================================================
 * hmask bit cycling: bits 0..7 for chunk0, 8..15 for chunk1 within each byte.
 * Each byte of hmask covers 32 values, with 8 bits = 4 groups * 2 chunks.
 */
static void ref_dequant_q3_K(const block_q3_K *b, float *dst) {
    /* Written from the Q3_K layout, independently of dequantize_row_q3_K().
     *
     * Critical detail that the previous version got wrong: block_q3_K has
     * qs[64] but hmask[32]. The 2-bit quants advance by 32 bytes per 128-value
     * chunk, while hmask and its bit cursor m run ACROSS all 8 sub-blocks
     * (m = 1,2,4,...,128), i.e. hmask is shared by both chunks. Within a chunk
     * each sub-block is 16+16 values (q[l] and q[l+16], same shift). A CLEAR
     * hmask bit means "subtract 4". */
    float d = ref_fp16(b->d);
    uint32_t aux[4];
    memcpy(aux, b->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
    aux[0] = (aux[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4);
    aux[1] = (aux[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4);
    const int8_t *scales = (const int8_t *)aux;

    int is = 0;
    uint8_t m = 1;
    float *y = dst;
    for (int chunk = 0; chunk < 2; chunk++) {
        const uint8_t *q  = b->qs + chunk * 32;
        const uint8_t *hm = b->hmask;          /* NOT advanced per chunk */
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            const float dl0 = d * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++)
                *y++ = dl0 * (float)((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            const float dl1 = d * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++)
                *y++ = dl1 * (float)((int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m <<= 1;
        }
    }
}

/* Build Q3_K block */
static void make_q3_K_block(block_q3_K *b, const int8_t vals[256]) {
    /* Encoder for Q3_K matching the layout dequantize_row_q3_K() consumes:
     *   - qs advances 32 bytes per 128-value chunk (qs[64])
     *   - hmask is SHARED across both chunks (hmask[32]); its bit cursor m runs
     *     1,2,4,...,128 over the 8 sub-blocks in order
     *   - within a chunk, sub-block 2j is q[l] and 2j+1 is q[l+16], both at
     *     2-bit shift j*2
     *   - hmask bit set => stored value is 0..3; bit clear => value is
     *     (stored - 4)
     */
    memset(b, 0, sizeof(*b));
    b->d = 0x3C00;
    for (int i = 0; i < 8; i++)  b->scales[i] = 0x11;
    for (int i = 8; i < 12; i++) b->scales[i] = 0xAA;

    int sb = 0;                     /* sub-block index 0..15 */
    for (int chunk = 0; chunk < 2; chunk++) {
        uint8_t *q = b->qs + chunk * 32;
        for (int j = 0; j < 4; j++) {
            const int shift = j * 2;
            for (int half = 0; half < 2; half++, sb++) {
                const int off = half * 16;
                const uint8_t m = (uint8_t)(1u << sb);
                for (int l = 0; l < 16; l++) {
                    int v = vals[sb * 16 + l];
                    uint8_t stored;
                    if (v < 0) { stored = (uint8_t)((v + 4) & 3); b->hmask[l + off] &= (uint8_t)~m; }
                    else       { stored = (uint8_t)(v & 3);      b->hmask[l + off] |= m; }
                    q[l + off] |= (uint8_t)(stored << shift);
                }
            }
        }
    }
}

/* Reference Q4_K dequant */
static void ref_dequant_q4_K(const block_q4_K *b, float *dst) {
    float d = ref_fp16(b->d), dm = ref_fp16(b->dmin);
    const uint8_t *q = b->qs; uint8_t sc, m;
    int is = 0;
    for (int j = 0; j < 4; j++) {
        get_scale_min_k4(is, b->scales, &sc, &m);
        float d1 = d*sc, m1 = dm*m;
        get_scale_min_k4(is+1, b->scales, &sc, &m);
        float d2 = d*sc, m2 = dm*m;
        for (int l = 0; l < 32; l++) *dst++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *dst++ = d2 * (q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}

static void make_q4_K_block(block_q4_K *b) {
    memset(b, 0, sizeof(*b));
    b->d = 0x3C00;
    b->scales[0]=1; b->scales[1]=1; b->scales[2]=1; b->scales[3]=1;
    b->scales[8]=1; b->scales[9]=1;
    for (int i = 0; i < 128; i++) b->qs[i] = (i*7+3) & 0xFF;
}

/* Reference Q6_K dequant */
static void ref_dequant_q6_K(const block_q6_K *b, float *dst) {
    /* Written from the Q6_K layout, independently of dequantize_row_q6_K().
     * Each 256-value super-block is two 128-value halves. Within a half, the
     * 32-byte qh window packs FOUR 2-bit high parts per byte (bits 0-1, 2-3,
     * 4-5, 6-7) for ql[l], ql[l+32], and their high nibbles. The previous
     * version of this reference indexed ql[j] and qh[j] at the same j for
     * j in [0,128), which cannot describe that layout -- it never matched the
     * kernel, so the test was wrong, not the code under test. */
    float d = ref_fp16(b->d);
    for (int chunk = 0; chunk < 256; chunk += 128) {
        const uint8_t *ql = b->ql + chunk / 2;
        const uint8_t *qh = b->qh + chunk / 4;
        const int is = chunk / 16;
        for (int l = 0; l < 32; l++) {
            int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
            const int is_l = is + (l / 16);
            dst[chunk + l]      = d * (float)b->scales[is_l + 0] * (float)q1;
            dst[chunk + l + 32] = d * (float)b->scales[is_l + 2] * (float)q2;
            dst[chunk + l + 64] = d * (float)b->scales[is_l + 4] * (float)q3;
            dst[chunk + l + 96] = d * (float)b->scales[is_l + 6] * (float)q4;
        }
    }
}

static void make_q6_K_block(block_q6_K *b) {
    memset(b, 0, sizeof(*b));
    b->d = 0x3C00;
    for (int i = 0; i < 16; i++) b->scales[i] = 1;
    for (int i = 0; i < 128; i++) b->ql[i] = (i*13+5) & 0xFF;
    for (int i = 0; i < 32; i++) b->qh[i] = (i*7+3) & 0xFF;
}

/* ================================================================
 * Tests
 * ================================================================ */

static void test_fp16(void) {
    tests_run++;
    fp16_table_init();
    /* Expected values must be exactly representable in fp16. The previous
     * table claimed 0x3E00==2.0 (it is 1.5), 0x4000==4.0 (it is 2.0) and
     * 0xBB80==-1.0 (it is -0.9375), so this test could never pass. */
    struct { uint16_t h; float e; } c[] = {
        {0x0000, 0.0f}, {0x8000, -0.0f}, {0x3C00, 1.0f}, {0xBC00, -1.0f},
        {0x4000, 2.0f}, {0xC000, -2.0f}, {0x3E00, 1.5f}, {0xBB80, -0.9375f},
        {0x3555, 0.333251953125f}, {0x7BFF, 65504.0f}, {0x0001, 5.960464477539063e-08f},
    };
    int errs = 0;
    for (int i = 0; i < (int)(sizeof c / sizeof c[0]); i++) {
        if (fabsf(fp16_to_fp32_lookup(c[i].h) - c[i].e) > 1e-6f) errs++;
        if (fabsf(ref_fp16(c[i].h) - c[i].e) > 1e-6f) errs++;
    }
    if (errs == 0) pass_t("FP16->FP32"); else fail_t("FP16->FP32");
}

static void test_q8_0(void) {
    tests_run++;
    block_q8_0 a, b;
    a.d = 0x3C00; b.d = 0x3C00;
    for (int i = 0; i < 32; i++) { a.qs[i] = (i*13+5)%256-128; b.qs[i] = (i*7+3)%256-128; }
    float rs = vec_dot_q8_0_q8_0(&a, &b, 32);
    float fa[32], fb[32];
    dequantize_row_q8_0(&a, fa, 32); dequantize_row_q8_0(&b, fb, 32);
    float rr = 0; for (int i = 0; i < 32; i++) rr += fa[i]*fb[i];
    if (fabsf(rs - rr) < 1e-3f) pass_t("Q8_0 vec_dot");
    else { fail_t("Q8_0 vec_dot"); fprintf(stderr, "    SIMD=%.2f REF=%.2f\n", rs, rr); }
}

static void test_q5_K_dequant(void) {
    tests_run++;
    /* The canonical test: catches Q5_K NEON qh bit interleaving bug.
     * Values cycle 0..31, exercising every qh bit position. */
    block_q5_K blk;
    int vals[256];
    for (int i = 0; i < 256; i++) vals[i] = i % 32;
    make_q5_K_block(&blk, vals);

    float ds[256], dr[256];
    dequantize_row_q5_K(&blk, ds, 256);
    ref_dequant_q5_K(&blk, dr);

    float maxd = 0; int fi = -1;
    for (int i = 0; i < 256; i++) {
        float d = fabsf(ds[i] - dr[i]);
        if (d > maxd) maxd = d;
        if (d > 1e-5f && fi < 0) fi = i;
    }
    if (maxd >= 1e-4f && fi >= 0)
        fprintf(stderr, "    max diff=%.4f first@%d: SIMD=%.2f REF=%.2f val=%d\n",
                maxd, fi, ds[fi], dr[fi], vals[fi]);
    if (maxd < 1e-4f) pass_t("Q5_K dequant (qh bit interleaving)");
    else fail_t("Q5_K dequant (qh bit interleaving)");
}

static void test_q5_K_dot(void) {
    tests_run++;
    block_q5_K wb; int wv[256];
    for (int i = 0; i < 256; i++) wv[i] = (i*7+3) % 32;
    make_q5_K_block(&wb, wv);

    block_q8_K ab; ab.d = 1.0f;
    memset(ab.bsums, 0, sizeof(ab.bsums));
    for (int i = 0; i < 256; i++) ab.qs[i] = (i*13+5)%256-128;

    float rs = vec_dot_q5_K_q8_K(&wb, &ab, 256);
    float wf[256], af[256];
    ref_dequant_q5_K(&wb, wf);
    /* ab is a block_q8_K: 256 int8 values sharing ONE float scale (PicoLM's
     * simplified Q8_K, see quant.h). The old code called
     * dequantize_row_q8_0() on it, which walks it as 32-value Q8_0 blocks with
     * fp16 scales -- nonsense activations, so the test failed against a
     * perfectly good kernel. */
    for (int i = 0; i < 256; i++) af[i] = ab.d * (float)ab.qs[i];
    float rr = 0; for (int i = 0; i < 256; i++) rr += wf[i]*af[i];
    float tol = 0.01f * fabsf(rr) + 1.0f;
    if (fabsf(rs - rr) < tol) pass_t("Q5_K vec_dot_q5_K_q8_K");
    else { fail_t("Q5_K vec_dot"); fprintf(stderr, "    SIMD=%.2f REF=%.2f\n", rs, rr); }
}

/* Q5_0 x Q8_0. Q5_0 was the only legacy type without a fused Q8_0-activation
 * dot, so a scalar-only build could not consume a Q5_0 tensor at all (it fell
 * through to the F32 path with a quantized buffer and produced NaN). This
 * covers the portable replacement across several block counts, since a
 * one-block test would not catch a per-block scale or stride error. */
static void test_q5_0_q8_0(void) {
    tests_run++;
    int bad = 0;
    for (int nb = 1; nb <= 5; nb++) {
        block_q5_0 w[5]; block_q8_0 q[5];
        float x[160], wf[160], qf[160];
        for (int k = 0; k < nb; k++) {
            w[k].d = fp32_to_fp16(0.02f + 0.01f * k);
            for (int i = 0; i < 16; i++) w[k].qs[i] = (uint8_t)((k * 53 + i * 11 + 7) & 0xFF);
            uint32_t qh = 0;
            for (int i = 0; i < 32; i++) if (((i * 13 + k * 7) & 16)) qh |= 1u << i;
            memcpy(w[k].qh, &qh, 4);
        }
        const int n = nb * 32;
        for (int i = 0; i < n; i++) x[i] = ((i * 17) % 41 - 20) * 0.01f;
        for (int k = 0; k < nb; k++) quantize_row_q8_0(x + k * 32, &q[k], 32);
        for (int k = 0; k < nb; k++) {
            dequantize_row_q5_0(&w[k], wf + k * 32, 32);
            dequantize_row_q8_0(&q[k], qf + k * 32, 32);
        }
        double ref = 0;
        for (int i = 0; i < n; i++) ref += (double)wf[i] * qf[i];
        float got = vec_dot_q5_0_q8_0(w, q, n);
        if (fabs((double)got - ref) > 1e-4 * fabs(ref) + 1e-5) bad++;
    }
    if (!bad) pass_t("Q5_0 x Q8_0 vec_dot (nb=1..5)");
    else fail_t("Q5_0 x Q8_0 vec_dot (nb=1..5)");
}

static void test_q4_K(void) {
    tests_run++;
    block_q4_K blk; make_q4_K_block(&blk);
    float ds[256], dr[256];
    dequantize_row_q4_K(&blk, ds, 256);
    ref_dequant_q4_K(&blk, dr);
    float maxd = 0;
    for (int i = 0; i < 256; i++) { float d = fabsf(ds[i]-dr[i]); if (d > maxd) maxd = d; }
    if (maxd < 1e-4f) pass_t("Q4_K dequant");
    else {
        fail_t("Q4_K dequant");
        int fi = -1;
        for (int i = 0; i < 256; i++) {
            float d = fabsf(ds[i]-dr[i]);
            if (d > 1e-5f && fi < 0) fi = i;
        }
        fprintf(stderr, "    max diff=%.4f first@%d: SIMD=%.2f REF=%.2f\n", maxd, fi, ds[fi], dr[fi]);
    }
}

static void test_q3_K(void) {
    tests_run++;
    block_q3_K blk;
    int8_t vals[256];
    for (int i = 0; i < 256; i++) vals[i] = (i%8) - 4;
    make_q3_K_block(&blk, vals);
    float ds[256], dr[256];
    dequantize_row_q3_K(&blk, ds, 256);
    ref_dequant_q3_K(&blk, dr);
    float maxd = 0; int fi = -1;
    for (int i = 0; i < 256; i++) {
        float d = fabsf(ds[i]-dr[i]);
        if (d > maxd) maxd = d;
        if (d > 1e-5f && fi < 0) fi = i;
    }
    if (maxd >= 1e-4f && fi >= 0)
        fprintf(stderr, "    max diff=%.4f first@%d: SIMD=%.2f REF=%.2f\n", maxd, fi, ds[fi], dr[fi]);
    if (maxd < 1e-4f) pass_t("Q3_K dequant (hmask bit cycling)");
    else fail_t("Q3_K dequant (hmask bit cycling)");
}

static void test_q6_K(void) {
    tests_run++;
    block_q6_K blk; make_q6_K_block(&blk);
    float ds[256], dr[256];
    dequantize_row_q6_K(&blk, ds, 256);
    ref_dequant_q6_K(&blk, dr);
    float maxd = 0;
    for (int i = 0; i < 256; i++) { float d = fabsf(ds[i]-dr[i]); if (d > maxd) maxd = d; }
    if (maxd < 1e-4f) pass_t("Q6_K dequant");
    else { fail_t("Q6_K dequant"); fprintf(stderr, "    max diff=%.4f\n", maxd); }
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "-v") == 0) verbose = 1;
    printf("PicoLM Quant Unit Tests\n");
    printf("Platform:");
#if defined(PICOLM_AVX512)
    printf(" AVX-512");
#elif defined(PICOLM_AVX2)
    printf(" AVX2");
#elif defined(PICOLM_NEON)
    printf(" NEON");
#elif defined(__ALTIVEC__)
    printf(" Altivec");
#else
    printf(" Scalar");
#endif
    printf("\n");
    printf("Blocks: Q5_K=%zu Q8_0=%zu\n", sizeof(block_q5_K), sizeof(block_q8_0));
    /* Every test_* defined above must be called here. Historically test_fp16,
     * test_q4_K, test_q3_K and test_q6_K were defined but never invoked, so
     * this program printed "2 tests: 2 passed" and CI went green while four
     * suites silently did nothing. The minimum-count assertion below turns
     * that class of rot into a hard failure instead of a quiet pass. */
    test_fp16();
    test_q8_0();
    test_q5_K_dequant();
    test_q5_K_dot();
    test_q4_K();
    test_q3_K();
    test_q6_K();
    test_q5_0_q8_0();
    printf("\n%d tests: %d passed, %d failed\n", tests_run, tests_pass, tests_fail);
    /* "tests_run too low" means a suite was skipped or deleted without updating
     * this file -- that is a test bug, not a pass. */
    if (tests_run < MIN_EXPECTED_TESTS) {
        fprintf(stderr,
                "FAIL: only %d tests ran, expected at least %d -- a suite is not "
                "being called from main() (or was removed).\n",
                tests_run, MIN_EXPECTED_TESTS);
        return 1;
    }
    return tests_fail > 0 ? 1 : 0;
}
