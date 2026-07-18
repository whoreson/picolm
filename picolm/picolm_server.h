/*
 * PicoLM Server API
 *
 * Build with -DPICOLM_SERVER to include the HTTP server.
 * Usage: ./picolm --server <model.gguf> [--port PORT] [--host HOST]
 */

#ifndef PICOLM_SERVER_H
#define PICOLM_SERVER_H

/* Start the HTTP server. Blocks until interrupted.
 * Returns 0 on success, -1 on failure. */
int server_main(int port, const char *host, const char *model_path, int num_threads, int do_prefault, int context_override, int mem_mb);

/* Get current time in milliseconds (declared in picolm.c) */
double get_time_ms(void);

#endif /* PICOLM_SERVER_H */
