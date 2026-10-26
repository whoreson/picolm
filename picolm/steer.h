/* steer.h - neuron steering through ffn_down columns.
 *
 * For each steered neuron L:N with dose d, the down_proj INPUT activation N is
 * shifted by   d * median(||resid leaving layer L||, tokens 1:) / ||W_down[:,N]||
 * at every token. down_proj is linear, so this is identical to adding
 *   d * median_L * unit(W_down[:,N])
 * to the residual stream right after layer L's FFN. We precompute one vector per
 * layer and add it there (steer_hook), so no matmul/hb surgery is needed.
 *
 * CPU path only (the GPU forward passes keep the residual on the device).
 */
#ifndef PICOLM_STEER_H
#define PICOLM_STEER_H

#include "model.h"

#define STEER_MAX 16

int  steer_add(const char *spec);        /* "L:N:DOSE"  e.g. "14:8949:+1"; 0 ok */
int  steer_set_norm(const char *spec);   /* "L:NORM"    skip calibration for layer L; 0 ok */
int  steer_active(void);                 /* any --steer given? */
const char *steer_calib_text(void);

int  steer_init(model_t *m);             /* extract W_down columns (after model_load) */
int  steer_calibrate(model_t *m, const int *toks, int n); /* clean CPU pass -> medians -> vectors */

/* Call after layer `layer` finished (FFN residual already added).
 * x = n_tokens rows of `dim` floats; first row has absolute position pos0. */
void steer_hook(int layer, float *x, int n_tokens, int dim, int pos0);

#endif
