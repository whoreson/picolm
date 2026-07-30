/*
 * PicoLM Server API
 *
 * Build with -DPICOLM_SERVER to include the HTTP server.
 * Usage: ./picolm --server <model.gguf> [--port PORT] [--host HOST]
 */

#ifndef PICOLM_SERVER_H
#define PICOLM_SERVER_H

#include "model.h"

/* Server configuration. Add new options here in ONE place. */
typedef struct {
    int port;
    char host[256];
    const char *model_path;
    int num_threads;
    int do_prefault;
    int context_override;
    int mem_mb;
    int checkpoint_max;
    int checkpoint_interval;
    int checkpoint_interval_gen;
    int checkpoint_tail_offset;
    const char *slot_save_path;
    kv_cache_type_t kv_type_k;
    kv_cache_type_t kv_type_v;
    int k_cache_hadamard;
    int v_cache_hadamard;
    int viz_port;
    int viz_width;
    int viz_height;
} server_config_t;

/* Start the HTTP server. Blocks until interrupted.
 * Returns 0 on success, -1 on failure. */
int server_main(server_config_t *cfg);

/* Get current time in milliseconds (declared in picolm.c) */
double get_time_ms(void);

#endif /* PICOLM_SERVER_H */
