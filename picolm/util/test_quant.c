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
    float d = ref_fp16(b->d);
    const uint8_t *q3 = b->qs, *hm = b->hmask;
    uint32_t aux[4];
    memcpy(aux, b->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0]>>4)&0x0F0F0F0F) | (((tmp>>4)&0x03030303)<<4);
    aux[3] = ((aux[1]>>4)&0x0F0F0F0F) | (((tmp>>6)&0x03030303)<<4);
    aux[0] = (aux[0]&0x0F0F0F0F) | (((tmp>>0)&0x03030303)<<4);
    aux[1] = (aux[1]&0x0F0F0F0F) | (((tmp>>2)&0x03030303)<<4);
    const int8_t *sc = (const int8_t *)aux;
    for (int chunk = 0; chunk < 2; chunk++) {
        for (int j = 0; j < 4; j++) {
            int g = chunk*4+j, scale = sc[g]-32;
            for (int l = 0; l < 32; l++) {
                int q3l = (q3[l] >> (j*2)) & 3;
                int hbit = (hm[l] >> (chunk*4+j)) & 1;
                *dst++ = d * (hbit ? (q3l-4) : q3l) * scale;
            }
        }
        q3 += 32;
    }
}

/* Build Q3_K block */
static void make_q3_K_block(block_q3_K *b, const int8_t vals[256]) {
    memset(b, 0, sizeof(*b));
    b->d = 0x3C00;
    /* scales: all sub-blocks = 33 (scale-32=1) */
    for (int i = 0; i < 4; i++) { b->scales[i] = 0x21; b->scales[i+4] = 0x21; }
    for (int i = 0; i < 4; i++) { b->scales[i+8] = 0x21; }
    b->scales[2]=0x22; b->scales[3]=0x22; b->scales[6]=0x22;
    b->scales[7]=0x22; b->scales[10]=0x22; b->scales[11]=0x22;
    /* qs: 2 bits per value, 16 values per byte.
     * hmask: 1 bit per value, 32 values per byte.
     * Layout: 2 chunks of 4 groups of 32 values.
     * qs: chunk0 uses qs[0..31], chunk1 uses qs[32..63].
     * Within each chunk, group j reads qs[l] bits j*2..j*2+1.
     * hmask: hm[l] bit (chunk*4+j) for value l in group j of chunk. */
    for (int chunk = 0; chunk < 2; chunk++) {
        for (int j = 0; j < 4; j++) {
            for (int l = 0; l < 32; l++) {
                int idx = chunk*128 + j*32 + l;
                int8_t v = vals[idx];
                int byte_idx = chunk*32 + l;
                int bit_shift = j * 2;
                int hbit_pos = chunk*4+j;
                if (v < 0) { b->hmask[l] |= (1 << hbit_pos); v = (v+4) & 3; }
                b->qs[byte_idx] |= ((v & 3) << bit_shift);
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
    float d = ref_fp16(b->d);
    const uint8_t *ql = b->ql, *qh = b->qh;
    const int8_t *sc = b->scales;
    for (int i = 0; i < 256; i += 128) {
        for (int j = i; j < i+64; j++)
            dst[j] = d * ((int8_t)((ql[j] + ((qh[j]&0x0F)<<4)) >> 4)) * sc[j/32];
        for (int j = i+64; j < i+128; j++)
            dst[j] = d * ((int8_t)((ql[j] + ((qh[j]&0xF0)<<0)) >> 4)) * sc[j/32];
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
    struct { uint16_t h; float e; } c[] = {
        {0x0000, 0.0f}, {0x3C00, 1.0f}, {0x3E00, 2.0f},
        {0x4000, 4.0f}, {0xBB80, -1.0f},
    };
    int errs = 0;
    for (int i = 0; i < 5; i++) {
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
    dequantize_row_q8_0(&ab, af, 256);
    float rr = 0; for (int i = 0; i < 256; i++) rr += wf[i]*af[i];
    float tol = 0.01f * fabsf(rr) + 1.0f;
    if (fabsf(rs - rr) < tol) pass_t("Q5_K vec_dot_q5_K_q8_K");
    else { fail_t("Q5_K vec_dot"); fprintf(stderr, "    SIMD=%.2f REF=%.2f\n", rs, rr); }
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
    test_q8_0();
    test_q5_K_dequant();
    printf("\n%d tests: %d passed, %d failed\n", tests_run, tests_pass, tests_fail);
    return tests_fail > 0 ? 1 : 0;
}
