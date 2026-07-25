#ifndef PICOLM_VIZ_H
#define PICOLM_VIZ_H

#include <stdint.h>
#include <stdatomic.h>

typedef struct viz_thread_handle_t viz_thread_handle_t;

/* ------------------------------------------------------------------ */
/*  PicoLM Visualization Subsystem                                     */
/*                                                                     */
/*  Embeds a tiny VNC server (RFB 3.3) on its own thread.              */
/*  During inference, viz_push_layer() pushes per-layer activation     */
/*  statistics into a lock-free ring buffer.  The VNC thread renders   */
/*  a scrolling heatmap and serves it to connected clients.            */
/*                                                                     */
/*  Enabled at compile time with: -DPICOLM_VIZ=1                       */
/*  Zero cost when disabled (empty static inlines).                    */
/* ------------------------------------------------------------------ */

/* Number of bucket-RMS values computed per layer.
 * n_embd is divided into VIZ_N_BUCKETS equal chunks; each chunk's RMS
 * is one "row" in the heatmap.  32 is a good default: it fits nicely
 * on a VNC screen and provides enough resolution to see structure.   */
#ifndef VIZ_N_BUCKETS
#define VIZ_N_BUCKETS 32
#endif

/* Ring buffer depth (number of layers buffered before dropping).
 * Must be a power of two for the mask-based indexing.                */
#ifndef VIZ_RING_DEPTH
#define VIZ_RING_DEPTH 8192  /* must be >= VIZ_TOKEN_RING * VIZ_MAX_LAYERS (64*128=8192) */
#endif

/* Maximum layers we can visualize.                                    */
#ifndef VIZ_MAX_LAYERS
#define VIZ_MAX_LAYERS 128
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the visualization subsystem.
 * Starts the VNC server thread listening on `port` (default 5900).
 * `width` x `height` is the framebuffer size (default 800x480).
 * Returns 0 on success, -1 on failure.
 * Call once after model loading.                                      */
int  viz_init(int width, int height, int port);

/* Push per-layer activation data into the ring buffer.
 * Called from the inference thread, once per layer per token.
 * `layer` is the layer index (0-based).
 * `data` points to `n_embd` floats (the residual stream after the layer).
 * `n_embd` is the embedding dimension.
 * This function is lock-free and never blocks.                        */
void viz_push_layer(int layer, const float *data, int n_embd);

/* Mark the start of a new token's forward pass.
 * Signals the renderer that the previous token's columns are complete
 * and a new group begins.                                             */
void viz_new_token(void);

/* Set global metadata (called once during init).                      */
void viz_set_model_info(int n_layers, int n_embd, const char *name);

/* Shut down the VNC server and free resources.                        */
void viz_free(void);

/* Check if any viewer is currently connected (optional, for debugging).*/
int  viz_has_viewers(void);

/* Check if a layer should be skipped (toggled by VNC mouse click).
 * Returns 1 if layer `layer` should be skipped, 0 otherwise.
 * Safe to call from inference thread (atomic load).                   */
int  viz_layer_skip(int layer);

/* Get the execution order permutation for layers.
 * Returns the logical layer index that should execute at position `slot`.
 * When no permutation is active, returns `slot` (identity).
 * Safe to call from inference thread (atomic load).                   */
int  viz_layer_permute(int slot);

/* Reset layer permutation to identity. */
void viz_layer_permute_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* PICOLM_VIZ_H */

