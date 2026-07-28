/*
 * PicoLM Server API
 *
 * Build with -DPICOLM_SERVER to include the HTTP server.
 * Usage: ./picolm --server <model.gguf> [--port PORT] [--host HOST]
 */

#ifndef PICOLM_SERVER_H
#define PICOLM_SERVER_H

/* Start the HTTP server. Blocks until interrupted.
 * Returns 0 on success, -1 on failure.
 * viz_port=0 means skip viz; otherwise start VNC server on that port. */
int server_main(int port, const char *host, const char *model_path, int num_threads, int do_prefault, int context_override, int mem_mb,
                int checkpoint_max, int checkpoint_interval, int checkpoint_interval_gen, int checkpoint_tail_offset,
                int viz_port, int viz_width, int viz_height);

/* Get current time in milliseconds (declared in picolm.c) */
double get_time_ms(void);

#endif /* PICOLM_SERVER_H */
