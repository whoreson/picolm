#include "steer.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { int layer, neuron; float dose; float *col; float col_norm; } steer_entry_t;

static steer_entry_t g_e[STEER_MAX];
static int    g_ne = 0;
static float *g_vec[MAX_LAYERS];            /* final per-layer vector, NULL = layer not steered */
static float  g_norm_override[MAX_LAYERS];  /* >0: use instead of calibrating */
static float *g_norms[MAX_LAYERS];          /* calibration samples */
static int    g_nn[MAX_LAYERS], g_cap[MAX_LAYERS];
static int    g_calibrating = 0, g_dim = 0;

static const char CALIB_TEXT[] =
    "The history of the lighthouse at the edge of the harbor goes back more than two centuries. "
    "Originally built from rough granite blocks hauled across the bay by flat-bottomed boats, it "
    "guided fishing crews home through fog and storms. Over the years the keepers changed, the lamp "
    "was replaced twice, and a small museum was added at its base. Today visitors climb the narrow "
    "spiral staircase to look out over the water, while local historians argue about which family "
    "first tended the flame. In a different part of town, a bakery opens before dawn, and the smell "
    "of fresh bread drifts down the cobbled street. Programmers sometimes joke that a good function "
    "should do one thing well, but a good loaf of bread takes patience, flour, water, salt and time. "
    "When the afternoon rain finally arrives, the market stalls fold their awnings and everyone "
    "waits for the clouds to pass, talking about the weather, the harvest and the coming holidays.";

const char *steer_calib_text(void) { return CALIB_TEXT; }

int steer_add(const char *spec)
{
    int l, n; float d;
    if (g_ne >= STEER_MAX || sscanf(spec, "%d:%d:%f", &l, &n, &d) != 3) return -1;
    if (l < 0 || l >= MAX_LAYERS || n < 0) return -1;
    g_e[g_ne].layer = l; g_e[g_ne].neuron = n; g_e[g_ne].dose = d; g_e[g_ne].col = NULL;
    g_ne++;
    return 0;
}

int steer_set_norm(const char *spec)
{
    int l; float v;
    if (sscanf(spec, "%d:%f", &l, &v) != 2 || l < 0 || l >= MAX_LAYERS || v <= 0.0f) return -1;
    g_norm_override[l] = v;
    return 0;
}

int steer_active(void) { return g_ne > 0; }

static int type_is_interleaved(gguf_type_t t)
{
    switch (t) {
    case GGUF_TYPE_IQ2_K_R4: case GGUF_TYPE_IQ3_K_R4: case GGUF_TYPE_IQ4_K_R4:
    case GGUF_TYPE_IQ4_NL_R4: case GGUF_TYPE_Q4_K_R4: case GGUF_TYPE_Q6_K_R4:
    case GGUF_TYPE_Q4_0_R8: case GGUF_TYPE_Q8_0_R8: case GGUF_TYPE_Q8_K_R8:
    case GGUF_TYPE_Q4_0_4_4: case GGUF_TYPE_Q4_0_4_8: case GGUF_TYPE_Q4_0_8_8:
    case GGUF_TYPE_Q4I_0_8_8:
        return 1;
    default: return 0;
    }
}

/* ffn_down is [n_embd rows][n_ffn cols] in memory, so W_down[:,n] is element n of every
 * row. Quant blocks run along a row: dequantize just the block containing n, per row. */
int steer_init(model_t *m)
{
    const model_config_t *c = &m->config;
    const int dim = c->n_embd, nffn = c->n_ffn;
    g_dim = dim;
    for (int k = 0; k < g_ne; k++) {
        steer_entry_t *e = &g_e[k];
        if (e->layer >= c->n_layers || e->neuron >= nffn) {
            fprintf(stderr, "steer: %d:%d out of range (n_layers=%d, n_ffn=%d)\n",
                    e->layer, e->neuron, c->n_layers, nffn);
            return -1;
        }
        const layer_weights_t *lw = &m->weights.layers[e->layer];
        if (!lw->ffn_down) { fprintf(stderr, "steer: layer %d has no dense ffn_down (MoE?)\n", e->layer); return -1; }
        gguf_type_t t = lw->type_ffn_down;
        int blk = gguf_type_block_size(t), qsz = gguf_type_quant_size(t);
        if (type_is_interleaved(t) || blk <= 0 || qsz <= 0 || blk > 1024 || nffn % blk) {
            fprintf(stderr, "steer: unsupported ffn_down type %d on layer %d\n", (int)t, e->layer);
            return -1;
        }
        size_t row_bytes = gguf_type_row_size(t, nffn);
        float tmp[1024];
        e->col = (float *)malloc(sizeof(float) * (size_t)dim);
        if (!e->col) return -1;
        double ss = 0;
        for (int i = 0; i < dim; i++) {
            const uint8_t *src = (const uint8_t *)lw->ffn_down + (size_t)i * row_bytes
                               + (size_t)(e->neuron / blk) * (size_t)qsz;
            dequantize_row(src, tmp, blk, t);
            e->col[i] = tmp[e->neuron % blk];
            ss += (double)e->col[i] * e->col[i];
        }
        e->col_norm = (float)sqrt(ss);
        if (!(e->col_norm > 0.0f)) { fprintf(stderr, "steer: zero column %d:%d\n", e->layer, e->neuron); return -1; }
    }
    return 0;
}

static int cmpf(const void *a, const void *b)
{ float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

static float median_of(float *v, int n)
{
    qsort(v, (size_t)n, sizeof(float), cmpf);
    return (n & 1) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

static int steer_build(void)
{
    for (int l = 0; l < MAX_LAYERS; l++) {
        int used = 0;
        for (int k = 0; k < g_ne; k++) used |= (g_e[k].layer == l);
        if (!used) continue;
        float med = g_norm_override[l];
        if (!(med > 0.0f)) {
            if (g_nn[l] < 1) { fprintf(stderr, "steer: no calibration samples for layer %d\n", l); return -1; }
            med = median_of(g_norms[l], g_nn[l]);
        }
        fprintf(stderr, "steer: layer %d median resid norm (tokens 1:) = %.3f\n", l, med);
        g_vec[l] = (float *)calloc((size_t)g_dim, sizeof(float));
        if (!g_vec[l]) return -1;
        for (int k = 0; k < g_ne; k++) {
            if (g_e[k].layer != l) continue;
            float amt = g_e[k].dose * med / g_e[k].col_norm;   /* delta on down_proj input N */
            for (int i = 0; i < g_dim; i++) g_vec[l][i] += amt * g_e[k].col[i];
            fprintf(stderr, "steer:   L%d N%d dose %+g -> activation delta %+.5f (||W_down[:,N]||=%.4f)\n",
                    l, g_e[k].neuron, g_e[k].dose, amt, g_e[k].col_norm);
        }
    }
    return 0;
}

/* Unsteered CPU decode pass over `toks`, sampling ||x|| after each steered layer. */
int steer_calibrate(model_t *m, const int *toks, int n)
{
    int need = 0;
    for (int k = 0; k < g_ne; k++) if (!(g_norm_override[g_e[k].layer] > 0.0f)) need = 1;
    if (need) {
        if (n < 8) { fprintf(stderr, "steer: calibration text too short (%d tokens)\n", n); return -1; }
        if (n > m->config.max_seq_len) n = m->config.max_seq_len;
        g_calibrating = 1;
        for (int i = 0; i < n; i++) model_forward(m, toks[i], i);
        g_calibrating = 0;
        model_ssm_state_reset(m);       /* drop recurrent state from the calibration run */
    }
    return steer_build();
}

void steer_hook(int layer, float *x, int n_tokens, int dim, int pos0)
{
    if (g_calibrating) {
        int hit = 0;
        for (int k = 0; k < g_ne; k++) hit |= (g_e[k].layer == layer);
        if (!hit) return;
        for (int t = 0; t < n_tokens; t++) {
            if (pos0 + t == 0) continue;                       /* tokens 1: (sink token excluded) */
            double ss = 0; const float *r = x + (size_t)t * dim;
            for (int i = 0; i < dim; i++) ss += (double)r[i] * r[i];
            if (g_nn[layer] == g_cap[layer]) {
                g_cap[layer] = g_cap[layer] ? g_cap[layer] * 2 : 256;
                g_norms[layer] = (float *)realloc(g_norms[layer], sizeof(float) * (size_t)g_cap[layer]);
            }
            g_norms[layer][g_nn[layer]++] = (float)sqrt(ss);
        }
        return;
    }
    const float *v = g_vec[layer];
    if (!v) return;
    for (int t = 0; t < n_tokens; t++) {                        /* every token, including pos 0 */
        float *r = x + (size_t)t * dim;
        for (int i = 0; i < dim; i++) r[i] += v[i];
    }
}
