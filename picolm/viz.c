/*
 * PicoLM Visualization Subsystem
 * ===============================
 *
 * A self-contained VNC server + activation heatmap renderer.
 * Embeds into PicoLM via a single #include "viz.h" + one call per layer.
 *
 * License: Same as PicoLM (MIT).
 * The VNC protocol implementation is adapted from the natpos VNC server
 * (Antivirial License), rewritten to be self-contained with no external
 * natpos dependencies.
 *
 * Build: add -DPICOLM_VIZ=1 to CFLAGS, link viz.o.
 */

#include "viz.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <errno.h>
#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")
#endif
#ifndef socklen_t
typedef int socklen_t;
#endif
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

/* viz socket close: closesocket on Windows, close on POSIX */
#ifdef _WIN32
#define _viz_close closesocket
#else
#define _viz_close close
#endif

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define VIZ_FRAME_BUF_SIZE  (VIZ_MAX_LAYERS * VIZ_N_BUCKETS)
#define VIZ_TOKEN_RING       128  /* positions of history kept (= visible columns) */
#define VIZ_MAX_CLIENTS      1
#define VNC_BUF_SIZE         65536

/* ------------------------------------------------------------------ */
/*  Color palette: viridis-inspired, mapped to BGRA framebuffer        */
/* ------------------------------------------------------------------ */

/* Returns a BGRA pixel for a normalized value in [0, 1].
 * Uses a plasma-style colormap: dark blue -> purple -> pink -> yellow -> white */
static inline unsigned int viz_color(float t) {
    /* Clamp to [0,1] */
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    /* Multi-segment colormap (plasma-like):
     * 0.0   = deep blue
     * 0.25  = purple
     * 0.5   = magenta/pink
     * 0.75  = orange
     * 1.0   = bright yellow-white */
    unsigned char r, g, b;

    if (t < 0.25f) {
        float s = t / 0.25f;
        r = (unsigned char)(15.0f  + s * 100.0f);
        g = (unsigned char)(10.0f  + s * 20.0f);
        b = (unsigned char)(150.0f + s * 80.0f);
    } else if (t < 0.5f) {
        float s = (t - 0.25f) / 0.25f;
        r = (unsigned char)(115.0f + s * 120.0f);
        g = (unsigned char)(30.0f  + s * 30.0f);
        b = (unsigned char)(230.0f - s * 120.0f);
    } else if (t < 0.75f) {
        float s = (t - 0.5f) / 0.25f;
        r = (unsigned char)(235.0f + s * 20.0f);
        g = (unsigned char)(60.0f  + s * 120.0f);
        b = (unsigned char)(110.0f - s * 80.0f);
    } else {
        float s = (t - 0.75f) / 0.25f;
        r = (unsigned char)(255.0f);
        g = (unsigned char)(180.0f + s * 75.0f);
        b = (unsigned char)(30.0f  + s * 30.0f);
    }

    /* All framebuffer writers take low byte = B, so pack as 0x00RRGGBB. */
    return ((unsigned int)r << 16) | ((unsigned int)g << 8) | (unsigned int)b;
}

/* A simpler grayscale colormap for debug */
static inline unsigned int viz_color_gray(float t) {
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    unsigned char v = (unsigned char)(t * 255.0f);
    return (unsigned int)v | ((unsigned int)v << 8) | ((unsigned int)v << 16);
}

/* ------------------------------------------------------------------ */
/*  Activation grid, indexed by ABSOLUTE token position                */
/* ------------------------------------------------------------------ */
/*
 * The grid is cells[pos % VIZ_TOKEN_RING][layer].  Every cell remembers the
 * absolute position it holds, so
 *   - arrival order does not matter: decode pushes (pos, layer 0..L-1) one
 *     token at a time, batched prefill pushes (layer 0: pos p..p+n-1), then
 *     layer 1 for all n tokens, ... (layer-major, because the GEMMs advance
 *     the whole batch one layer at a time).  Both land in the right cell.
 *   - stale data is detectable (cell.pos != expected pos).
 *   - a KV-cache rewind (pos goes backwards) invalidates the newer cells.
 * Cells are protected by a per-cell seqlock (single writer = inference
 * thread, reader = VNC/render thread).
 */
typedef struct {
    _Atomic unsigned seq;        /* seqlock: odd = write in progress */
    _Atomic int      pos;        /* absolute position held, -1 = empty */
    float            buckets[VIZ_N_BUCKETS];
} viz_cell_t;

typedef struct {
    viz_cell_t cells[VIZ_TOKEN_RING][VIZ_MAX_LAYERS];
    _Atomic int head;            /* newest position pushed, -1 = none yet */
    _Atomic int active;          /* 1 once viz_init succeeded (gates all work) */
    _Atomic unsigned gen;        /* bumped on every data change: lets the renderer idle */
    int n_layers;                /* set by viz_set_model_info */
} viz_ring_t;

static viz_ring_t viz_ring;

static void viz_ring_reset(void) {
    for (int c = 0; c < VIZ_TOKEN_RING; c++)
        for (int l = 0; l < VIZ_MAX_LAYERS; l++) {
            atomic_store_explicit(&viz_ring.cells[c][l].seq, 0, memory_order_relaxed);
            atomic_store_explicit(&viz_ring.cells[c][l].pos, -1, memory_order_relaxed);
        }
    atomic_store(&viz_ring.head, -1);
}

/* ------------------------------------------------------------------ */
/*  Shared state                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Framebuffer (BGRA, Y-down). */
    unsigned char *framebuffer;
    int width;
    int height;

    /* Model metadata. */
    int n_layers;
    int n_embd;
    char model_name[128];

    /* VNC server state. */
    int listen_fd;
    int client_fd;
    int port;
    _Atomic int client_connected;
    _Atomic int running;
    _Atomic int update_requested;
    _Atomic int exited; /* set to 1 when VNC thread has fully exited */

    /* VNC protocol state. */
    int state; /* 0=disconnected, 1=protocol, 2=client_init, 3=connected */

    /* Mutex for framebuffer writes during rendering. */
    pthread_mutex_t fb_mutex;

    /* Thread handle. */
    pthread_t thread;

    /* Scroll offset: how many tokens back from the latest. */
    int scroll_offset;

    /* Display mode: 0=heatmap, 1=norm-bars, 2=both */
    int display_mode;

    /* Global min/max for normalization (updated periodically). */
    float global_min;
    float global_max;
    int   norm_count;

    /* Color scale.  Colors depend ONLY on a cell's own value and on a scale
     * that is frozen once calibrated, so scrolling never recolors a column. */
    _Atomic int norm_mode;       /* 0 = one global log scale, 1 = per-layer log scale */
    _Atomic int bucket_mode;     /* 0 = one cell per (layer,pos), 1 = bucket sub-rows */
    _Atomic int recal_request;   /* 'r' key: re-derive the scale on the next frame */
    int   cal_valid;
    int   fixed_scale;           /* PICOLM_VIZ_LO/HI given: never auto-calibrate */
    float glo, ghi;              /* log10 range, global mode */
    float lo[VIZ_MAX_LAYERS], hi[VIZ_MAX_LAYERS]; /* log10 range, per-layer mode */
    unsigned dump_frames;        /* PICOLM_VIZ_DUMP frame counter */

    /* VNC transport state (VNC thread only, except where noted) */
    unsigned char *shadow;       /* BGRA copy of what the client currently shows */
    unsigned char *dirty;        /* per-tile dirty flags (scratch) */
    unsigned char *txbuf;        /* one FramebufferUpdate message, sent in one go */
    size_t txcap;
    int    fbu_pending;          /* client has an unanswered FramebufferUpdateRequest */
    int    fbu_full;             /* non-incremental request: resend the region unconditionally */
    int    req_x, req_y, req_w, req_h;
    int    enc_hextile;          /* client advertised Hextile */
    unsigned rendered_gen;       /* viz_ring.gen the framebuffer was last rendered at */
    unsigned long long net_bytes;
    unsigned net_updates, net_rects;
    double net_t0;

    /* FPS counter */
    double last_frame_time;
    int fps;
    int frame_count;
    double fps_timer;

} viz_state_t;

static viz_state_t viz;

/* ------------------------------------------------------------------ */
/*  Layer skip array (toggled by VNC mouse click)                      */
/*  Shared between VNC thread (writer) and inference thread (reader).  */
/* ------------------------------------------------------------------ */
static _Atomic int viz_skip_layer[VIZ_MAX_LAYERS];

/* Layer permutation: viz_layer_permute(slot) returns which logical layer
 * should execute at position `slot` in the forward pass. Identity by
 * default. Modified by VNC drag-and-drop. */
static _Atomic int viz_layer_perm[VIZ_MAX_LAYERS];

/* Build an inverse map: perm_inv[layer] = slot where layer executes.
 * Used for rendering (knowing which row to draw which layer's data). */
static int viz_perm_inv[VIZ_MAX_LAYERS];

static void viz_perm_rebuild_inv(void) {
    for (int i = 0; i < VIZ_MAX_LAYERS; i++) {
        int layer = atomic_load_explicit(&viz_layer_perm[i], memory_order_relaxed);
        if (layer >= 0 && layer < VIZ_MAX_LAYERS)
            viz_perm_inv[layer] = i;
    }
}

/* Check if a layer should be skipped.
 * Called from the inference thread. Atomic load, no blocking. */
int viz_layer_skip(int layer) {
    if (layer >= 0 && layer < VIZ_MAX_LAYERS)
        return atomic_load_explicit(&viz_skip_layer[layer], memory_order_relaxed);
    return 0;
}

/* Get the execution order permutation for layers.
 * Returns the logical layer index that should execute at position `slot`. */
int viz_layer_permute(int slot) {
    if (slot >= 0 && slot < VIZ_MAX_LAYERS)
        return atomic_load_explicit(&viz_layer_perm[slot], memory_order_relaxed);
    return slot;
}

/* Reset layer permutation to identity. */
void viz_layer_permute_reset(void) {
    for (int i = 0; i < VIZ_MAX_LAYERS; i++)
        atomic_store_explicit(&viz_layer_perm[i], i, memory_order_relaxed);
    viz_perm_rebuild_inv();
}

/* Ensure identity permutation at program startup, before main() runs.
 * This is critical: when PICOLM_VIZ is defined but --viz is not passed,
 * viz_init() is never called, leaving viz_layer_perm[] zero-initialized
 * (every slot maps to layer 0). The constructor guarantees correct identity
 * mapping regardless of whether the viz server is started. */
#ifdef _MSC_VER
#pragma section(".CRT$XCU", read)
static void viz_permute_startup_init(void);
__declspec(allocate(".CRT$XCU")) void (*viz_permute_startup_init_ptr)(void) = viz_permute_startup_init;
static void viz_permute_startup_init(void)
#else
__attribute__((constructor))
static void viz_permute_startup_init(void)
#endif
{
    viz_layer_permute_reset();
}

/* ------------------------------------------------------------------ */
/*  VNC Protocol Helpers (big-endian network byte order)               */
/* ------------------------------------------------------------------ */

static void vnc_write_u8(unsigned char *buf, uint8_t val) {
    buf[0] = val;
}

static void vnc_write_u16_be(unsigned char *buf, uint16_t val) {
    buf[0] = (unsigned char)((val >> 8) & 0xFF);
    buf[1] = (unsigned char)(val & 0xFF);
}

static void vnc_write_u32_be(unsigned char *buf, uint32_t val) {
    buf[0] = (unsigned char)((val >> 24) & 0xFF);
    buf[1] = (unsigned char)((val >> 16) & 0xFF);
    buf[2] = (unsigned char)((val >> 8) & 0xFF);
    buf[3] = (unsigned char)(val & 0xFF);
}

static uint8_t  __attribute__((unused)) vnc_read_u8(const unsigned char *buf) { return buf[0]; }

static uint16_t vnc_read_u16_be(const unsigned char *buf) {
    return ((uint16_t)buf[0] << 8) | (uint16_t)buf[1];
}

static uint32_t vnc_read_u32_be(const unsigned char *buf) {
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  | (uint32_t)buf[3];
}

/* ------------------------------------------------------------------ */
/*  Socket Helpers                                                     */
/* ------------------------------------------------------------------ */

static int viz_set_nonblocking(int fd) {
#ifdef _WIN32
    u_long mode = 1;
    return ioctlsocket(fd, FIONBIO, &mode);
#else
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
#endif
}

static int viz_recv_full(int fd, void *buf, size_t len) {
    size_t received = 0;
    unsigned char *ptr = (unsigned char *)buf;
    while (received < len) {
        int r = recv(fd, (char *)(ptr + received), (int)(len - received), 0);
        if (r > 0) {
            received += (size_t)r;
        } else if (r == 0) {
            return -1;
        } else {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err != WSAEWOULDBLOCK) return -1;
            Sleep(1);
#else
            if (errno != EAGAIN && errno != EWOULDBLOCK) return -1;
            usleep(1000); /* 1ms */
#endif
        }
    }
    return 0;
}

static int viz_send_full(int fd, const void *buf, size_t len) {
    size_t sent = 0;
    const unsigned char *ptr = (const unsigned char *)buf;
    while (sent < len) {
        int s = send(fd, (const char *)(ptr + sent), (int)(len - sent), 0);
        if (s > 0) {
            sent += (size_t)s;
        } else if (s == 0) {
            return -1;
        } else {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err != WSAEWOULDBLOCK) return -1;
            Sleep(1);
#else
            if (errno != EAGAIN && errno != EWOULDBLOCK) return -1;
            usleep(1000);
#endif
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  usleep portability                                                 */
/* ------------------------------------------------------------------ */

static void viz_sleep(double seconds) {
    if (seconds > 0) {
#ifdef _WIN32
        Sleep((DWORD)(seconds * 1000.0));
#else
        struct timespec ts;
        ts.tv_sec = (time_t)floor(seconds);
        ts.tv_nsec = (long)round((seconds - (double)ts.tv_sec) * 1000000000.0);
        nanosleep(&ts, NULL);
#endif
    }
}

/* Portable error string for socket operations (only used on Windows) */
#ifdef _WIN32
static const char *viz_socket_strerror(int err) {
    /* WSAGetLastError codes: WSAEINVAL=10022, WSAEMFILE=10024, etc. */
    switch (err) {
        case WSAENOTSOCK: return "socket operation on non-socket";
        case WSAEADDRINUSE: return "address already in use";
        case WSAEACCES: return "permission denied";
        case WSAEINVAL: return "invalid argument";
        default: return "socket error";
    }
}
#endif

/* Monotonic time in seconds */
static double viz_mono_time(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
#endif
}

/* ------------------------------------------------------------------ */
/*  Compute bucket-RMS from activation vector                          */
/* ------------------------------------------------------------------ */

static void viz_compute_buckets(float *buckets, const float *data, int n_embd, int n_buckets) {
    int bucket_size = n_embd / n_buckets;
    if (bucket_size < 1) bucket_size = 1;

    for (int b = 0; b < n_buckets; b++) {
        double sum = 0.0;
        int start = b * bucket_size;
        /* The last bucket absorbs the remainder (n_embd % n_buckets) so the
         * trailing dimensions of the embedding are never silently dropped
         * from the visualization. Previously `end = start + bucket_size`
         * for the final bucket, which left the last few dims of any
         * n_embd not evenly divisible by n_buckets completely unread. */
        int end = (b == n_buckets - 1) ? n_embd : start + bucket_size;
        if (end > n_embd) end = n_embd;
        if (start > n_embd) start = n_embd;
        for (int i = start; i < end; i++) {
            sum += (double)data[i] * (double)data[i];
        }
        int count = end - start;
        /* Guard against count==0 (happens whenever n_embd < n_buckets,
         * e.g. a tiny model), which previously produced sqrtf(0.0/0.0) ==
         * NaN that then silently poisoned min/max tracking and coloring
         * downstream. */
        buckets[b] = (count > 0) ? (float)sqrtf(sum / (double)count) : 0.0f;
    }
}

/* ------------------------------------------------------------------ */
/*  Producer side (inference thread)                                   */
/* ------------------------------------------------------------------ */

static void viz_cell_write(viz_cell_t *c, int pos, const float *data, int n_embd) {
    unsigned q = atomic_load_explicit(&c->seq, memory_order_relaxed);
    atomic_store_explicit(&c->seq, q + 1, memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    if (data) viz_compute_buckets(c->buckets, data, n_embd, VIZ_N_BUCKETS);
    atomic_store_explicit(&c->pos, pos, memory_order_relaxed);
    atomic_store_explicit(&c->seq, q + 2, memory_order_release);
}

/* Consistent snapshot of one cell. Returns 1 and fills out/pos on success. */
static int viz_cell_read(const viz_cell_t *c, int *pos, float *out) {
    for (int tries = 0; tries < 8; tries++) {
        unsigned q1 = atomic_load_explicit(&c->seq, memory_order_acquire);
        if (q1 & 1u) continue;
        *pos = atomic_load_explicit(&c->pos, memory_order_relaxed);
        memcpy(out, c->buckets, sizeof(float) * VIZ_N_BUCKETS);
        atomic_thread_fence(memory_order_acquire);
        if (atomic_load_explicit(&c->seq, memory_order_relaxed) == q1) return 1;
    }
    return 0;
}

/* Positions >= `from` are about to be recomputed (KV rewind / new prompt on a
 * reused slot): drop everything newer so stale columns don't linger. */
static void viz_rewind(int from, int head) {
    int n = head - from + 1;
    if (n > VIZ_TOKEN_RING) { from = head - VIZ_TOKEN_RING + 1; }
    for (int p = from; p <= head; p++) {
        if (p < 0) continue;
        for (int l = 0; l < VIZ_MAX_LAYERS; l++) {
            viz_cell_t *c = &viz_ring.cells[p % VIZ_TOKEN_RING][l];
            if (atomic_load_explicit(&c->pos, memory_order_relaxed) == p)
                viz_cell_write(c, -1, NULL, 0);
        }
    }
}

static void viz_dump_maybe(int pos);

static void viz_push_one(int layer, int pos, const float *data, int n_embd) {
    viz_cell_t *c = &viz_ring.cells[pos % VIZ_TOKEN_RING][layer];
    int head = atomic_load_explicit(&viz_ring.head, memory_order_relaxed);
    int rewound = 0;
    /* Same (pos,layer) already present => this position is being recomputed. */
    if (pos <= head && atomic_load_explicit(&c->pos, memory_order_relaxed) == pos) {
        viz_rewind(pos, head);
        rewound = 1;
    }
    viz_cell_write(c, pos, data, n_embd);
    if (pos > head || rewound)
        atomic_store_explicit(&viz_ring.head, pos, memory_order_release);
    atomic_fetch_add_explicit(&viz_ring.gen, 1u, memory_order_release);
}

void viz_push_row(int layer, int pos, const float *data, int n_embd) {
    if (!atomic_load_explicit(&viz_ring.active, memory_order_relaxed)) return;
    if (layer < 0 || layer >= VIZ_MAX_LAYERS || pos < 0 || !data) return;
    viz_push_one(layer, pos, data, n_embd);
    viz_dump_maybe(pos);
}

/* Batched prefill: x_batch is [n_tokens][n_embd] row-major (row i = absolute
 * position start_pos+i), the residual stream AFTER `layer` for the whole
 * batch.  Only the newest VIZ_TOKEN_RING rows can ever be displayed, so older
 * rows are skipped (bounds the cost for very long prompts). */
void viz_push_batch(int layer, int start_pos, const float *x_batch, int n_tokens, int n_embd) {
    if (!atomic_load_explicit(&viz_ring.active, memory_order_relaxed)) return;
    if (layer < 0 || layer >= VIZ_MAX_LAYERS || start_pos < 0 || !x_batch || n_tokens < 1) return;
    int first = n_tokens > VIZ_TOKEN_RING ? n_tokens - VIZ_TOKEN_RING : 0;
    for (int i = first; i < n_tokens; i++)
        viz_push_one(layer, start_pos + i, x_batch + (size_t)i * (size_t)n_embd, n_embd);
    viz_dump_maybe(start_pos + n_tokens - 1);
}

/* Deprecated: positions are now explicit.  Kept so old callers still link. */
void viz_new_token(void) {}

/* Debug/test helper: RMS over the whole vector for (layer,pos), or -1. */
float viz_peek(int layer, int pos) {
    if (layer < 0 || layer >= VIZ_MAX_LAYERS || pos < 0) return -1.0f;
    float b[VIZ_N_BUCKETS]; int p;
    if (!viz_cell_read(&viz_ring.cells[pos % VIZ_TOKEN_RING][layer], &p, b) || p != pos) return -1.0f;
    double sq = 0; for (int i = 0; i < VIZ_N_BUCKETS; i++) sq += (double)b[i] * b[i];
    return (float)sqrt(sq / VIZ_N_BUCKETS);
}

/* ------------------------------------------------------------------ */
/*  Rendering: draw the heatmap to the framebuffer                     */
/* ------------------------------------------------------------------ */

/* Draw text as pixel characters (simple 5x7 font, monochrome).
 * Very basic - just for labels. */
static const unsigned char viz_font[96][5] = {
    /* Printable ASCII 32-126, 5 pixels wide, 7 tall (top 3 bits unused,
     * each row is a byte where bottom 5 bits are pixels) */
    [ ' ' - 32]  = {0x00,0x00,0x00,0x00,0x00}, /* space */
    [ '!' - 32]  = {0x04,0x04,0x04,0x00,0x04},
    [ '"' - 32]  = {0x1A,0x1A,0x00,0x00,0x00},
    [ '#' - 32]  = {0x1A,0x7E,0x1A,0x7E,0x1A},
    [ '$' - 32]  = {0x0A,0x1C,0x10,0x1C,0x0A},
    [ '%' - 32]  = {0x06,0x3C,0x18,0x0C,0x06},
    [ '&' - 32]  = {0x0C,0x1A,0x0C,0x32,0x22},
    [ '\'' - 32] = {0x04,0x04,0x08,0x00,0x00},
    [ '(' - 32]  = {0x04,0x08,0x10,0x10,0x08},
    [ ')' - 32]  = {0x10,0x08,0x04,0x04,0x08},
    [ '*' - 32]  = {0x00,0x1A,0x00,0x7E,0x00},
    [ '+' - 32]  = {0x00,0x10,0x3E,0x10,0x00},
    [ ',' - 32]  = {0x00,0x00,0x00,0x04,0x04},
    [ '-' - 32]  = {0x00,0x00,0x1C,0x00,0x00},
    [ '.' - 32]  = {0x00,0x00,0x00,0x00,0x04},
    [ '/' - 32]  = {0x02,0x04,0x08,0x10,0x02},
    [ '0' - 32]  = {0x1C,0x22,0x2A,0x32,0x1C},
    [ '1' - 32]  = {0x04,0x0C,0x04,0x04,0x0E},
    [ '2' - 32]  = {0x1C,0x22,0x04,0x08,0x1E},
    [ '3' - 32]  = {0x1C,0x02,0x08,0x20,0x1C},
    [ '4' - 32]  = {0x04,0x14,0x24,0x3E,0x04},
    [ '5' - 32]  = {0x3E,0x20,0x1C,0x02,0x1C},
    [ '6' - 32]  = {0x1C,0x20,0x3C,0x22,0x1C},
    [ '7' - 32]  = {0x3E,0x02,0x04,0x08,0x08},
    [ '8' - 32]  = {0x1C,0x2A,0x1C,0x2A,0x1C},
    [ '9' - 32]  = {0x1C,0x22,0x1E,0x02,0x1C},
    [ ':' - 32]  = {0x00,0x04,0x00,0x04,0x00},
    [ ';' - 32]  = {0x00,0x04,0x00,0x04,0x04},
    [ '<' - 32]  = {0x08,0x10,0x20,0x10,0x08},
    [ '=' - 32]  = {0x00,0x1C,0x00,0x1C,0x00},
    [ '>' - 32]  = {0x02,0x04,0x08,0x04,0x02},
    [ '?' - 32]  = {0x1C,0x22,0x04,0x00,0x04},
    [ '@' - 32]  = {0x1C,0x22,0x3A,0x2A,0x14},
    [ 'A' - 32]  = {0x14,0x22,0x3E,0x22,0x22},
    [ 'B' - 32]  = {0x3C,0x22,0x3C,0x22,0x3C},
    [ 'C' - 32]  = {0x1C,0x22,0x20,0x20,0x1C},
    [ 'D' - 32]  = {0x38,0x24,0x22,0x24,0x38},
    [ 'E' - 32]  = {0x3E,0x20,0x3C,0x20,0x3E},
    [ 'F' - 32]  = {0x3E,0x20,0x3C,0x20,0x20},
    [ 'G' - 32]  = {0x1C,0x22,0x2A,0x2A,0x16},
    [ 'H' - 32]  = {0x22,0x22,0x3E,0x22,0x22},
    [ 'I' - 32]  = {0x1C,0x04,0x04,0x04,0x1C},
    [ 'J' - 32]  = {0x1E,0x02,0x02,0x02,0x1C},
    [ 'K' - 32]  = {0x22,0x2A,0x32,0x2A,0x22},
    [ 'L' - 32]  = {0x20,0x20,0x20,0x20,0x3E},
    [ 'M' - 32]  = {0x22,0x3A,0x3A,0x2A,0x22},
    [ 'N' - 32]  = {0x22,0x32,0x2A,0x26,0x22},
    [ 'O' - 32]  = {0x1C,0x22,0x22,0x22,0x1C},
    [ 'P' - 32]  = {0x3C,0x22,0x3C,0x20,0x20},
    [ 'Q' - 32]  = {0x1C,0x22,0x22,0x24,0x1A},
    [ 'R' - 32]  = {0x3C,0x22,0x3C,0x2A,0x22},
    [ 'S' - 32]  = {0x1C,0x20,0x1C,0x02,0x1C},
    [ 'T' - 32]  = {0x3E,0x04,0x04,0x04,0x04},
    [ 'U' - 32]  = {0x22,0x22,0x22,0x22,0x1C},
    [ 'V' - 32]  = {0x22,0x22,0x22,0x14,0x08},
    [ 'W' - 32]  = {0x22,0x22,0x2A,0x3A,0x22},
    [ 'X' - 32]  = {0x22,0x14,0x08,0x14,0x22},
    [ 'Y' - 32]  = {0x22,0x14,0x08,0x08,0x08},
    [ 'Z' - 32]  = {0x3E,0x04,0x08,0x10,0x3E},
    [ '[' - 32]  = {0x14,0x10,0x10,0x10,0x14},
    [ ']' - 32]  = {0x1C,0x10,0x10,0x10,0x1C},
    [ '_' - 32]  = {0x00,0x00,0x00,0x00,0x3E},
    /* Lowercase letters (5x5, same "byte = pattern << 1" convention as the
     * uppercase/digit glyphs above -- viz_draw_char() applies ">> 1" to
     * every glyph uniformly, so all entries in this table must be encoded
     * the same way). The previous values here did not actually decode to
     * their named letters (e.g. 'u' and 'w' rendered as unrelated
     * checkerboard blobs, 'a' and 'd' were nearly identical); this set was
     * regenerated from explicit 5x5 bitmaps and round-trip verified against
     * viz_draw_char's decode path. */
    [ 'a' - 32]  = {0x00,0x1C,0x20,0x22,0x1C},
    [ 'b' - 32]  = {0x20,0x20,0x38,0x22,0x38},
    [ 'c' - 32]  = {0x00,0x1C,0x20,0x20,0x1C},
    [ 'd' - 32]  = {0x02,0x02,0x1C,0x22,0x1C},
    [ 'e' - 32]  = {0x00,0x1C,0x3E,0x20,0x1C},
    [ 'f' - 32]  = {0x0C,0x10,0x38,0x10,0x10},
    [ 'g' - 32]  = {0x00,0x1C,0x22,0x1E,0x04},
    [ 'h' - 32]  = {0x20,0x20,0x38,0x22,0x22},
    [ 'i' - 32]  = {0x10,0x00,0x10,0x10,0x10},
    [ 'j' - 32]  = {0x08,0x00,0x08,0x08,0x30},
    [ 'k' - 32]  = {0x20,0x24,0x38,0x24,0x24},
    [ 'l' - 32]  = {0x30,0x10,0x10,0x10,0x38},
    [ 'm' - 32]  = {0x00,0x2A,0x2A,0x2A,0x2A},
    [ 'n' - 32]  = {0x00,0x38,0x24,0x24,0x24},
    [ 'o' - 32]  = {0x00,0x1C,0x22,0x22,0x1C},
    [ 'p' - 32]  = {0x00,0x38,0x24,0x38,0x20},
    [ 'q' - 32]  = {0x00,0x1C,0x24,0x1C,0x04},
    [ 'r' - 32]  = {0x00,0x2C,0x30,0x20,0x20},
    [ 's' - 32]  = {0x00,0x1C,0x30,0x06,0x38},
    [ 't' - 32]  = {0x10,0x38,0x10,0x10,0x0C},
    [ 'u' - 32]  = {0x00,0x24,0x24,0x24,0x1C},
    [ 'v' - 32]  = {0x00,0x22,0x22,0x14,0x08},
    [ 'w' - 32]  = {0x00,0x22,0x2A,0x2A,0x14},
    [ 'x' - 32]  = {0x00,0x22,0x14,0x14,0x22},
    [ 'y' - 32]  = {0x00,0x22,0x14,0x08,0x10},
    [ 'z' - 32]  = {0x00,0x3E,0x04,0x10,0x3E},
    /* Extra punctuation */
    [ '{' - 32]  = {0x10,0x08,0x20,0x08,0x10},
    [ '}' - 32]  = {0x20,0x10,0x08,0x10,0x20},
    [ '|' - 32]  = {0x08,0x08,0x08,0x08,0x08},
    [ '\\' - 32] = {0x10,0x08,0x04,0x02,0x10},
    [ '^' - 32]  = {0x08,0x1C,0x08,0x00,0x00},
    [ '`' - 32]  = {0x10,0x08,0x00,0x00,0x00},
    [ '~' - 32]  = {0x00,0x1A,0x08,0x00,0x00},
};

/* Draw a character at (x, y) in color `col` on BGRA framebuffer.
 * Font is 5 wide x 5 tall. */
#define VIZ_FONT_H 5
static void viz_draw_char(unsigned char *fb, int fb_w, int fb_h, int fx, int fy,
                          char ch, unsigned int col) {
    if (ch < ' ' || ch > '~') return;
    const unsigned char *glyph = viz_font[ch - ' '];
    /* glyph[0] is the TOP row of the character and glyph[4] is the bottom;
     * this matches the framebuffer's top-to-bottom row order directly, so
     * no reversal is needed (or performed) here. */
    for (int y = 0; y < VIZ_FONT_H; y++) {
        unsigned char row = glyph[y] >> 1; /* shift bits 1-5 to 0-4 */
        for (int x = 0; x < 5; x++) {
            if (row & (1 << (4 - x))) { /* bit 4 = leftmost pixel */
                int px = fx + x;
                int py = fy + y;
                if (px >= 0 && px < fb_w && py >= 0 && py < fb_h) {
                    int idx = (py * fb_w + px) * 4;
                    fb[idx]     = (unsigned char)(col & 0xFF);         /* B */
                    fb[idx + 1] = (unsigned char)((col >> 8) & 0xFF);  /* G */
                    fb[idx + 2] = (unsigned char)((col >> 16) & 0xFF); /* R */
                    fb[idx + 3] = 255;                                  /* A */
                }
            }
        }
    }
}

/* Draw a null-terminated string. Returns the x position after the last char. */
static int viz_draw_string(unsigned char *fb, int fb_w, int fb_h, int sx, int sy,
                           const char *str, unsigned int col) {
    int x = sx;
    while (*str) {
        viz_draw_char(fb, fb_w, fb_h, x, sy, *str, col);
        x += 6; /* 5px glyph + 1px spacing */
        str++;
    }
    return x;
}

/* Draw a filled rectangle in BGRA framebuffer. */
static void __attribute__((unused)) viz_draw_rect(unsigned char *fb, int fb_w, int x, int y,
                          int w, int h, unsigned int col) {
    for (int dy = 0; dy < h; dy++) {
        for (int dx = 0; dx < w; dx++) {
            int px = x + dx;
            int py = y + dy;
            if (px >= 0 && px < fb_w && py >= 0) {
                int idx = (py * fb_w + px) * 4;
                fb[idx]     = (unsigned char)(col & 0xFF);
                fb[idx + 1] = (unsigned char)((col >> 8) & 0xFF);
                fb[idx + 2] = (unsigned char)((col >> 16) & 0xFF);
                fb[idx + 3] = 255;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Render the activation heatmap                                      */
/* ------------------------------------------------------------------ */

/* Layout constants shared with viz_y_to_slot() (mouse -> layer mapping). */
#define VIZ_TOP_BAR_H 20

static int viz_cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static float viz_pctl(float *v, int n, float q) {   /* v sorted ascending */
    if (n <= 0) return 0.0f;
    int i = (int)(q * (float)(n - 1) + 0.5f);
    return v[i < 0 ? 0 : (i >= n ? n - 1 : i)];
}

/* RMS of buckets [b0,b1) */
static float viz_rms_range(const float *b, int b0, int b1) {
    double sq = 0.0;
    for (int i = b0; i < b1; i++) sq += (double)b[i] * (double)b[i];
    return (b1 > b0) ? (float)sqrt(sq / (double)(b1 - b0)) : 0.0f;
}

/* log10(value) -> [0,1] under the current (frozen) scale. */
static float viz_norm(int layer, float v) {
    float lv = log10f(v > 1e-12f ? v : 1e-12f);
    float lo = atomic_load_explicit(&viz.norm_mode, memory_order_relaxed) ? viz.lo[layer] : viz.glo;
    float hi = atomic_load_explicit(&viz.norm_mode, memory_order_relaxed) ? viz.hi[layer] : viz.ghi;
    return (hi > lo) ? (lv - lo) / (hi - lo) : 0.5f;
}

static float viz_rms_snap[VIZ_TOKEN_RING][VIZ_MAX_LAYERS]; /* guarded by fb_mutex */
static float viz_scratch_a[VIZ_TOKEN_RING * VIZ_MAX_LAYERS];
static float viz_scratch_b[VIZ_TOKEN_RING];

/* Derive the color scale from what is on screen right now, but only once a
 * FULL column (all layers) exists, then freeze it.  Uses 1st/99th percentiles
 * so a single massive-activation outlier (e.g. an attention-sink token)
 * saturates instead of crushing everything else into one color. */
static void viz_calibrate(int ncols, int n_layers, int first_pos) {
    int n = 0;
    for (int c = 0; c < ncols; c++)
        for (int l = 0; l < n_layers; l++)
            if (viz_rms_snap[c][l] >= 0.0f) viz_scratch_a[n++] = log10f(fmaxf(viz_rms_snap[c][l], 1e-12f));
    if (n < 1) return;
    qsort(viz_scratch_a, (size_t)n, sizeof(float), viz_cmp_float);
    float lo = viz_pctl(viz_scratch_a, n, 0.02f) - 0.05f;
    float hi = viz_pctl(viz_scratch_a, n, 0.98f) + 0.05f;
    if (hi - lo < 1.0f) { float m = 0.5f * (hi + lo); lo = m - 0.5f; hi = m + 0.5f; }
    viz.glo = lo; viz.ghi = hi;

    int n_out = 0, out_l0 = 0, out_l1 = 0, out_pos = 0; float out_v = 0, out_hi = 0;
    for (int l = 0; l < n_layers; l++) {
        int m = 0, amax = 0; float vmax = -1.0f;
        for (int c = 0; c < ncols; c++) {
            float v = viz_rms_snap[c][l];
            if (v < 0.0f) continue;
            if (v > vmax) { vmax = v; amax = c; }
            viz_scratch_b[m++] = log10f(fmaxf(v, 1e-12f));
        }
        if (m < 1) { viz.lo[l] = lo; viz.hi[l] = hi; continue; }
        qsort(viz_scratch_b, (size_t)m, sizeof(float), viz_cmp_float);
        /* Robust per-layer range: 5th..95th percentile. Outlier tokens saturate
         * instead of squashing every other token into one color. */
        float l0 = viz_pctl(viz_scratch_b, m, 0.05f);
        float h0 = viz_pctl(viz_scratch_b, m, 0.95f);
        if (h0 - l0 < 0.02f) { float mid = 0.5f * (h0 + l0); l0 = mid - 0.01f; h0 = mid + 0.01f; }
        viz.lo[l] = l0; viz.hi[l] = h0;
        if (log10f(fmaxf(vmax, 1e-12f)) > h0 + 0.7f) {  /* >5x above p95 */
            if (n_out == 0) { out_l0 = l; out_pos = first_pos + amax; out_v = vmax; out_hi = powf(10.0f, h0); }
            out_l1 = l; n_out++;
        }
    }
    if (n_out)
        fprintf(stderr, "[VIZ] outlier token (saturated in color): pos=%d in %d layers (%d..%d); "
                "e.g. layer %d rms=%.4g vs p95=%.4g\n", out_pos, n_out, out_l0, out_l1, out_l0, out_v, out_hi);
}

static void viz_render_heatmap(void) {
    int fb_w = viz.width, fb_h = viz.height;
    unsigned char *fb = viz.framebuffer;
    memset(fb, 0, (size_t)fb_w * fb_h * 4);

    /* Geometry (must match viz_y_to_slot): layers on Y, absolute position on X,
     * oldest left / newest right.  Until the window is full, position p sits at
     * a fixed x; after that the window scrolls one column per new position. */
    int heatmap_y = VIZ_TOP_BAR_H + 2;
    int heatmap_h = fb_h - heatmap_y - 2;
    int n_layers = viz.n_layers < 1 ? 1 : viz.n_layers;
    if (n_layers > VIZ_MAX_LAYERS) n_layers = VIZ_MAX_LAYERS;
    int layer_h = heatmap_h / n_layers; if (layer_h < 1) layer_h = 1;

    int avail = fb_w - 4 - 20;                 /* leave room for the color bar */
    int col_w = avail / VIZ_TOKEN_RING; if (col_w < 1) col_w = 1;
    int W = avail / col_w; if (W > VIZ_TOKEN_RING) W = VIZ_TOKEN_RING; if (W < 1) W = 1;

    int head = atomic_load_explicit(&viz_ring.head, memory_order_acquire);
    int first = (head >= W) ? head - W + 1 : 0;
    int ncols = head >= 0 ? head - first + 1 : 0;

    /* Snapshot pass: RMS per (column,layer), -1 if absent. */
    int full_cols = 0, head_filled = 0;
    for (int c = 0; c < ncols; c++) {
        int p = first + c, filled = 0;
        for (int l = 0; l < n_layers; l++) {
            float b[VIZ_N_BUCKETS]; int cp;
            viz_rms_snap[c][l] = -1.0f;
            if (viz_cell_read(&viz_ring.cells[p % VIZ_TOKEN_RING][l], &cp, b) && cp == p) {
                float r = viz_rms_range(b, 0, VIZ_N_BUCKETS);
                viz_rms_snap[c][l] = isfinite(r) ? r : -2.0f;   /* -2 = NaN/Inf */
                filled++;
            }
        }
        if (filled == n_layers) full_cols++;
        if (p == head) head_filled = filled;
    }

    {   /* Report non-finite activations once per layer (they are a model/quant problem). */
        static unsigned char reported[VIZ_MAX_LAYERS];
        for (int c = 0; c < ncols; c++)
            for (int l = 0; l < n_layers; l++)
                if (viz_rms_snap[c][l] < -1.5f && !reported[l]) {
                    reported[l] = 1;
                    fprintf(stderr, "[VIZ] WARNING: non-finite (NaN/Inf) activations at layer %d, pos %d "
                                    "(drawn white)\n", l, first + c);
                }
    }

    /* Scale: fixed by env, or calibrated once and frozen. */
    if (atomic_exchange_explicit(&viz.recal_request, 0, memory_order_relaxed) && !viz.fixed_scale)
        viz.cal_valid = 0;
    if (!viz.fixed_scale && !viz.cal_valid && ncols > 0) {
        viz_calibrate(ncols, n_layers, first);
        if (full_cols > 0) viz.cal_valid = 1;   /* provisional until a full column exists */
    }

    unsigned int txt = 0xFFFFFF, dim = 0xA0A0B0;
    char buf[256];
    snprintf(buf, sizeof(buf), "PicoLM VIZ: %s [%dL, %dd] pos=%d lay=%d/%d %s",
             viz.model_name, viz.n_layers, viz.n_embd, head, head_filled, n_layers,
             atomic_load_explicit(&viz.client_connected, memory_order_relaxed) ? "VNC" : "");
    viz_draw_string(fb, fb_w, fb_h, 4, 2, buf, txt);
    snprintf(buf, sizeof(buf), "Y=layer X=pos  RMS log-scale %s%s [r]ecal [n]orm [b]uckets",
             atomic_load_explicit(&viz.norm_mode, memory_order_relaxed) ? "per-layer" : "global",
             atomic_load_explicit(&viz.bucket_mode, memory_order_relaxed) ? "+buckets" : "");
    viz_draw_string(fb, fb_w, fb_h, 4, 9, buf, dim);

    /* Position ticks (absolute token index) every 16 columns. */
    for (int c = 0; c < ncols; c++) {
        int p = first + c;
        if (p % 16) continue;
        int x = 2 + c * col_w;
        snprintf(buf, sizeof(buf), "%d", p);
        viz_draw_string(fb, fb_w, fb_h, x, VIZ_TOP_BAR_H - 6, buf, dim);
    }

    int bucket_mode = atomic_load_explicit(&viz.bucket_mode, memory_order_relaxed);
    int ch = layer_h >= 3 ? layer_h - 1 : layer_h;   /* 1px gap between layers */
    int cw = col_w   >= 3 ? col_w   - 1 : col_w;     /* 1px gap between columns */

    for (int c = 0; c < ncols; c++) {
        int p = first + c, x0 = 2 + c * col_w;
        for (int l = 0; l < n_layers; l++) {
            if (viz_rms_snap[c][l] == -1.0f) continue;    /* not computed (yet) */
            int y0 = heatmap_y + l * layer_h;
            if (viz_rms_snap[c][l] < -1.5f) {             /* NaN/Inf: pure white, never a palette color */
                viz_draw_rect(fb, fb_w, x0, y0, cw, ch, 0xFFFFFF);
                continue;
            }
            if (!bucket_mode || ch < 2) {
                viz_draw_rect(fb, fb_w, x0, y0, cw, ch, viz_color(viz_norm(l, viz_rms_snap[c][l])));
            } else {
                float b[VIZ_N_BUCKETS]; int cp;
                if (!viz_cell_read(&viz_ring.cells[p % VIZ_TOKEN_RING][l], &cp, b) || cp != p) continue;
                int sr = 1;                                   /* sub-rows: pow2 <= min(N, ch) */
                while (sr * 2 <= VIZ_N_BUCKETS && sr * 2 <= ch) sr *= 2;
                int sh = ch / sr;
                for (int sIdx = 0; sIdx < sr; sIdx++) {
                    int b0 = sIdx * VIZ_N_BUCKETS / sr, b1 = (sIdx + 1) * VIZ_N_BUCKETS / sr;
                    viz_draw_rect(fb, fb_w, x0, y0 + sIdx * sh, cw, sh,
                                  viz_color(viz_norm(l, viz_rms_range(b, b0, b1))));
                }
            }
        }
    }

    /* Color bar + its labels (real values of the frozen scale). */
    int bar_x = fb_w - 16, bar_w = 12;
    if (bar_x > 4 && heatmap_h > 0) {
        for (int y = 0; y < heatmap_h; y++)
            viz_draw_rect(fb, fb_w, bar_x, heatmap_y + y, bar_w, 1,
                          viz_color(1.0f - (float)y / (float)heatmap_h));
        if (atomic_load_explicit(&viz.norm_mode, memory_order_relaxed)) {
            viz_draw_string(fb, fb_w, fb_h, bar_x - 14, heatmap_y, "hi", dim);
            viz_draw_string(fb, fb_w, fb_h, bar_x - 14, heatmap_y + heatmap_h - VIZ_FONT_H, "lo", dim);
        } else {
            snprintf(buf, sizeof(buf), "%.3g", powf(10.0f, viz.ghi));
            viz_draw_string(fb, fb_w, fb_h, bar_x - 6 * (int)strlen(buf) - 2, heatmap_y, buf, dim);
            snprintf(buf, sizeof(buf), "%.3g", powf(10.0f, viz.glo));
            viz_draw_string(fb, fb_w, fb_h, bar_x - 6 * (int)strlen(buf) - 2,
                            heatmap_y + heatmap_h - VIZ_FONT_H, buf, dim);
        }
    }

    /* Layer labels on the left. */
    if (layer_h >= 10 && n_layers <= 64) {
        for (int l = 0; l < n_layers; l++) {
            snprintf(buf, sizeof(buf), "%d", l);
            viz_draw_string(fb, fb_w, fb_h, 1, heatmap_y + l * layer_h + 1, buf, dim);
        }
    }

    /* Permutation indicators ("L@S") for layers not in their natural slot. */
    {
        int label_x = fb_w - 52; if (label_x < 20) label_x = fb_w / 2;
        for (int l = 0; l < n_layers; l++) {
            int exec_slot = viz_perm_inv[l];
            if (exec_slot == l) continue;
            snprintf(buf, sizeof(buf), "%d@%d", l, exec_slot);
            viz_draw_string(fb, fb_w, fb_h, label_x, heatmap_y + l * layer_h + 1, buf, 0xFFFFAA);
        }
    }

    /* Dim skipped layer rows and mark them. */
    for (int l = 0; l < n_layers; l++) {
        if (!atomic_load_explicit(&viz_skip_layer[l], memory_order_relaxed)) continue;
        int row_start = heatmap_y + l * layer_h;
        for (int py = row_start; py < row_start + layer_h && py < fb_h; py++)
            for (int px = 12; px < bar_x && px < fb_w; px++) {
                size_t idx = ((size_t)py * fb_w + px) * 4;
                fb[idx] = (unsigned char)(fb[idx] * 3 / 10);
                fb[idx+1] = (unsigned char)(fb[idx+1] * 3 / 10);
                fb[idx+2] = (unsigned char)(fb[idx+2] * 3 / 10);
            }
        int s_y = row_start + (layer_h - 8) / 2; if (s_y < heatmap_y) s_y = row_start + 1;
        viz_draw_string(fb, fb_w, fb_h, fb_w / 2, s_y, "SKIP", 0xFF3030);
    }
}

/* Write the current frame as a binary PPM (also usable as a test hook). */
int viz_snapshot_ppm(const char *path) {
    if (!viz.framebuffer) return -1;
    pthread_mutex_lock(&viz.fb_mutex);
    viz_render_heatmap();
    FILE *f = fopen(path, "wb");
    int rc = -1;
    if (f) {
        fprintf(f, "P6\n%d %d\n255\n", viz.width, viz.height);
        for (int i = 0; i < viz.width * viz.height; i++) {
            const unsigned char *px = viz.framebuffer + (size_t)i * 4;   /* B G R A */
            fputc(px[2], f); fputc(px[1], f); fputc(px[0], f);
        }
        fclose(f); rc = 0;
    }
    pthread_mutex_unlock(&viz.fb_mutex);
    return rc;
}

/* PICOLM_VIZ_DUMP=prefix: write prefix_NNNN.ppm each time the newest column
 * becomes complete, i.e. once per forward pass (decode token or prefill batch). */
static void viz_dump_maybe(int pos) {
    static const char *pfx; static int init;
    if (!init) { pfx = getenv("PICOLM_VIZ_DUMP"); init = 1; }
    if (!pfx || viz.n_layers < 1 || pos != atomic_load_explicit(&viz_ring.head, memory_order_relaxed)) return;
    int n = 0;
    for (int l = 0; l < viz.n_layers && l < VIZ_MAX_LAYERS; l++)
        if (atomic_load_explicit(&viz_ring.cells[pos % VIZ_TOKEN_RING][l].pos, memory_order_relaxed) == pos) n++;
    if (n != viz.n_layers) return;
    char fn[512];
    snprintf(fn, sizeof(fn), "%s_%04u.ppm", pfx, viz.dump_frames++);
    viz_snapshot_ppm(fn);
}

/* ------------------------------------------------------------------ */
/*  VNC Server Implementation                                          */
/* ------------------------------------------------------------------ */

static void viz_vnc_disconnect(void) {
    if (viz.client_fd > 0) {
        fprintf(stderr, "[VIZ] VNC: Client disconnected\n");
        _viz_close(viz.client_fd);
    }
    viz.client_fd = -1;
    viz.state = 0;
    viz.fbu_pending = 0; viz.fbu_full = 1; viz.enc_hextile = 0;
    atomic_store_explicit(&viz.client_connected, 0, memory_order_release);
}

static int viz_vnc_handle_handshake(void) {
    unsigned char buf[256];
    int r;

    switch (viz.state) {
        case 0: /* disconnected -> send protocol version */
            if (viz_send_full(viz.client_fd, "RFB 003.003\n", 12) < 0) {
                return -1;
            }
            fprintf(stderr, "[VIZ] VNC: Sent protocol version RFB 003.003\n");
            viz.state = 1;
            return 0;

        case 1: /* protocol -> receive client version */
            r = viz_recv_full(viz.client_fd, buf, 12);
            if (r < 0) return -1;
            buf[11] = '\0';
            fprintf(stderr, "[VIZ] VNC: Client version: %.11s\n", buf);

            /* Send security type: None */
            vnc_write_u32_be(buf, 1);
            if (viz_send_full(viz.client_fd, buf, 4) < 0) return -1;
            viz.state = 2;
            return 0;

        case 2: /* client_init -> receive ClientInit, send ServerInit */
            r = viz_recv_full(viz.client_fd, buf, 1);
            if (r < 0) return -1;

            fprintf(stderr, "[VIZ] VNC: ClientInit received (shared=%d)\n", buf[0]);

            /* Send ServerInit: 20 bytes header + desktop name */
            unsigned char sinit[24];
            vnc_write_u16_be(sinit + 0, (uint16_t)viz.width);
            vnc_write_u16_be(sinit + 2, (uint16_t)viz.height);

            /* Pixel format: 32bpp, little-endian, true-color, RGB */
            vnc_write_u8(sinit + 4, 32);    /* bits-per-pixel */
            vnc_write_u8(sinit + 5, 24);    /* depth */
            vnc_write_u8(sinit + 6, 0);     /* little-endian */
            vnc_write_u8(sinit + 7, 1);     /* true-color */
            vnc_write_u16_be(sinit + 8, 255);   /* red-max */
            vnc_write_u16_be(sinit + 10, 255);  /* green-max */
            vnc_write_u16_be(sinit + 12, 255);  /* blue-max */
            vnc_write_u8(sinit + 14, 16);   /* red-shift */
            vnc_write_u8(sinit + 15, 8);    /* green-shift */
            vnc_write_u8(sinit + 16, 0);    /* blue-shift */
            vnc_write_u8(sinit + 17, 0);
            vnc_write_u8(sinit + 18, 0);
            vnc_write_u8(sinit + 19, 0);

            {
                uint32_t name_len = (uint32_t)strlen(viz.model_name);
                vnc_write_u32_be(sinit + 20, name_len);

                if (viz_send_full(viz.client_fd, sinit, 24) < 0) return -1;
                if (viz_send_full(viz.client_fd, viz.model_name, name_len) < 0) return -1;
            }

            fprintf(stderr, "[VIZ] VNC: Handshake complete\n");
            viz.state = 3;
            atomic_store_explicit(&viz.client_connected, 1, memory_order_release);
            atomic_store_explicit(&viz.update_requested, 1, memory_order_release);
            return 0;

        default:
            return -1;
    }
}

/* Drag state for layer permutation (persisted across calls) */
static int viz_drag_slot = -1;    /* slot where button was pressed */
static int viz_drag_prev_y = -1;  /* y of last event, for debouncing */

/* Map a y-coordinate to a layer slot, or -1 if outside heatmap */
static int viz_y_to_slot(uint16_t y) {
    int heatmap_y = VIZ_TOP_BAR_H + 2;
    int heatmap_h = viz.height - heatmap_y - 2;
    int n_layers = viz.n_layers;
    if (n_layers > 0 && y >= (uint16_t)heatmap_y && y < (uint16_t)(heatmap_y + heatmap_h)) {
        int layer_h = heatmap_h / n_layers;
        if (layer_h > 0) {
            int s = (y - heatmap_y) / layer_h;
            if (s >= 0 && s < n_layers) return s;
        }
    }
    return -1;
}

static void viz_vnc_process_messages(void) {
    unsigned char msg_type;
    unsigned char buf[64];
    int r;

    r = recv(viz.client_fd, (char *)&msg_type, 1,
#ifdef _WIN32
             0
#else
             MSG_DONTWAIT
#endif
    );
    if (r == 0) {
        viz_vnc_disconnect();
        return;
    }
    if (r < 0) {
#ifdef _WIN32
        if (WSAGetLastError() == WSAEWOULDBLOCK) return;
#else
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
#endif
        viz_vnc_disconnect();
        return;
    }

    switch (msg_type) {
        case 0: /* SetPixelFormat */
            if (viz_recv_full(viz.client_fd, buf, 19) < 0) {
                viz_vnc_disconnect();
            }
            break;

        case 2: /* SetEncodings */
            if (viz_recv_full(viz.client_fd, buf, 3) < 0) {
                viz_vnc_disconnect();
                return;
            }
            {
                uint16_t num_enc = vnc_read_u16_be(buf + 1);
                int hextile = 0;
                for (uint16_t i = 0; i < num_enc; i++) {
                    if (viz_recv_full(viz.client_fd, buf, 4) < 0) {
                        viz_vnc_disconnect();
                        return;
                    }
                    if (vnc_read_u32_be(buf) == 5u) hextile = 1;
                }
                viz.enc_hextile = hextile;
            }
            break;

        case 3: /* FramebufferUpdateRequest */
            if (viz_recv_full(viz.client_fd, buf, 9) < 0) {
                viz_vnc_disconnect();
                return;
            }
            {
                viz.req_x = vnc_read_u16_be(buf + 1); viz.req_y = vnc_read_u16_be(buf + 3);
                viz.req_w = vnc_read_u16_be(buf + 5); viz.req_h = vnc_read_u16_be(buf + 7);
                viz.fbu_pending = 1;
                if (!buf[0]) {            /* non-incremental: client lost its picture, resend region */
                    viz.fbu_full = 1;
                    atomic_store_explicit(&viz.update_requested, 1, memory_order_release);
                }
            }
            break;

        case 4: /* KeyEvent: r = recalibrate scale, n = global/per-layer, b = bucket rows */
            if (viz_recv_full(viz.client_fd, buf, 7) < 0) {
                viz_vnc_disconnect();
                break;
            }
            if (buf[0]) {
                uint32_t key = vnc_read_u32_be(buf + 3);
                if (key == 'r') atomic_store(&viz.recal_request, 1);
                else if (key == 'n') atomic_store(&viz.norm_mode, !atomic_load(&viz.norm_mode));
                else if (key == 'b') atomic_store(&viz.bucket_mode, !atomic_load(&viz.bucket_mode));
                atomic_store_explicit(&viz.update_requested, 1, memory_order_release);
            }
            break;

        case 5: /* PointerEvent - click toggles skip, drag permutes layers */
            if (viz_recv_full(viz.client_fd, buf, 5) < 0) {
                viz_vnc_disconnect();
            }
            {
                uint8_t button_mask = buf[0];
                (void)vnc_read_u16_be(buf + 1); /* mouse_x */
                uint16_t mouse_y = vnc_read_u16_be(buf + 3);

                int slot = viz_y_to_slot(mouse_y);

                if (button_mask & 0x01) {
                    /* Button held: track start slot */
                    if (viz_drag_slot < 0 && slot >= 0) {
                        viz_drag_slot = slot;
                        viz_drag_prev_y = mouse_y;
                    }
                } else {
                    /* Button released: decide click vs drag */
                    if (viz_drag_slot >= 0) {
                        int release_slot = viz_y_to_slot(mouse_y);
                        if (release_slot >= 0 && release_slot != viz_drag_slot) {
                            /* Drag: swap original press slot with release slot */
                            int a = viz_drag_slot, b = release_slot;
                            int tmp = atomic_load_explicit(&viz_layer_perm[a], memory_order_relaxed);
                            atomic_store_explicit(&viz_layer_perm[a],
                                atomic_load_explicit(&viz_layer_perm[b], memory_order_relaxed),
                                memory_order_release);
                            atomic_store_explicit(&viz_layer_perm[b], tmp, memory_order_release);
                            viz_perm_rebuild_inv();
                            fprintf(stderr, "[VIZ] Swapped exec slots %d<->%d (layers %d<->%d)\n",
                                    a, b,
                                    atomic_load_explicit(&viz_layer_perm[a], memory_order_relaxed),
                                    atomic_load_explicit(&viz_layer_perm[b], memory_order_relaxed));
                            atomic_store_explicit(&viz.update_requested, 1, memory_order_release);
                        } else if (release_slot == viz_drag_slot) {
                            /* Click: released on same slot -> toggle skip */
                            int old = atomic_load_explicit(&viz_skip_layer[viz_drag_slot],
                                                            memory_order_relaxed);
                            atomic_store_explicit(&viz_skip_layer[viz_drag_slot], !old,
                                                   memory_order_release);
                            fprintf(stderr, "[VIZ] Slot %d (layer %d): %s\n",
                                    viz_drag_slot,
                                    atomic_load_explicit(&viz_layer_perm[viz_drag_slot], memory_order_relaxed),
                                    !old ? "SKIPPED" : "ACTIVE");
                            atomic_store_explicit(&viz.update_requested, 1,
                                                   memory_order_release);
                        }
                        viz_drag_slot = -1;
                        viz_drag_prev_y = -1;
                    }
                }
            }
            break;

        case 6: /* ClientCutText */
            if (viz_recv_full(viz.client_fd, buf, 7) < 0) {
                viz_vnc_disconnect();
                return;
            }
            {
                uint32_t text_len = vnc_read_u32_be(buf + 3);
                if (text_len > 65536) text_len = 65536; /* cap */
                unsigned char *tmp = (unsigned char *)malloc(text_len);
                if (tmp) {
                    viz_recv_full(viz.client_fd, tmp, text_len);
                    free(tmp);
                }
            }
            break;

        default:
            viz_vnc_disconnect();
            break;
    }
}

/* ------------------------------------------------------------------ */
/*  Framebuffer transport: dirty tiles -> Hextile (or Raw)             */
/* ------------------------------------------------------------------ */
/*
 * Per update we (1) diff the freshly rendered framebuffer against `shadow`
 * (what the client already has) in 16x16 tiles, (2) merge dirty tiles into
 * rectangles, (3) encode them into ONE buffer, (4) send it outside the
 * framebuffer mutex.  Nothing changed => nothing is sent and the client's
 * request simply stays outstanding (legal RFB).
 * Pixel format on the wire: 32bpp little-endian, bytes B,G,R,0.
 */
#define VIZ_TILE 16
#define VIZ_MAX_RECTS 2048

static inline uint32_t viz_px(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v & 0x00FFFFFFu;   /* ignore alpha byte */
}

static inline unsigned char *viz_put_px(unsigned char *o, uint32_t v) {
    o[0] = (unsigned char)v; o[1] = (unsigned char)(v >> 8); o[2] = (unsigned char)(v >> 16); o[3] = 0;
    return o + 4;
}

static unsigned char *viz_enc_raw(unsigned char *o, const unsigned char *fb, int fbw,
                                  int rx, int ry, int rw, int rh) {
    for (int y = 0; y < rh; y++) {
        const unsigned char *src = fb + ((size_t)(ry + y) * fbw + rx) * 4;
        for (int x = 0; x < rw; x++) o = viz_put_px(o, viz_px(src + x * 4));
    }
    return o;
}

/* Hextile (RFB encoding 5).  We never rely on background/foreground carry-over
 * between tiles: every non-raw tile states its own background and uses
 * coloured subrects, so each tile decodes independently. */
static unsigned char *viz_enc_hextile(unsigned char *o, const unsigned char *fb, int fbw,
                                      int rx, int ry, int rw, int rh) {
    uint32_t t[VIZ_TILE * VIZ_TILE];
    unsigned char skip[VIZ_TILE * VIZ_TILE];
    for (int ty = 0; ty < rh; ty += VIZ_TILE) {
        int th = rh - ty < VIZ_TILE ? rh - ty : VIZ_TILE;
        for (int tx = 0; tx < rw; tx += VIZ_TILE) {
            int tw = rw - tx < VIZ_TILE ? rw - tx : VIZ_TILE;
            for (int y = 0; y < th; y++) {
                const unsigned char *src = fb + ((size_t)(ry + ty + y) * fbw + rx + tx) * 4;
                for (int x = 0; x < tw; x++) t[y * tw + x] = viz_px(src + x * 4);
            }
            int n = tw * th;

            /* most frequent colour = background; bail to raw if too noisy */
            uint32_t dc[24]; int cnt[24]; int nd = 0, noisy = 0;
            for (int i = 0; i < n && !noisy; i++) {
                int k = 0;
                while (k < nd && dc[k] != t[i]) k++;
                if (k == nd) { if (nd == 24) { noisy = 1; break; } dc[nd] = t[i]; cnt[nd] = 0; nd++; }
                cnt[k]++;
            }
            size_t raw_sz = 1 + (size_t)n * 4;
            if (!noisy) {
                int bi = 0;
                for (int k = 1; k < nd; k++) if (cnt[k] > cnt[bi]) bi = k;
                uint32_t bg = dc[bi];
                if (nd == 1) {                       /* solid tile: flag + bg */
                    *o++ = 2; o = viz_put_px(o, bg);
                    continue;
                }
                unsigned char sub[255 * 6]; int ns = 0, fits = 1;
                for (int i = 0; i < n; i++) skip[i] = (t[i] == bg);
                for (int y = 0; y < th && fits; y++)
                    for (int x = 0; x < tw; x++) {
                        if (skip[y * tw + x]) continue;
                        uint32_t col = t[y * tw + x];
                        int w = 1, h = 1;
                        while (x + w < tw && !skip[y * tw + x + w] && t[y * tw + x + w] == col) w++;
                        for (; y + h < th; h++) {
                            int ok = 1;
                            for (int i = 0; i < w; i++)
                                if (skip[(y + h) * tw + x + i] || t[(y + h) * tw + x + i] != col) { ok = 0; break; }
                            if (!ok) break;
                        }
                        for (int yy = 0; yy < h; yy++) for (int xx = 0; xx < w; xx++) skip[(y + yy) * tw + x + xx] = 1;
                        if (ns >= 255) { fits = 0; break; }
                        unsigned char *p = sub + ns * 6; viz_put_px(p, col);
                        p[4] = (unsigned char)((x << 4) | y);
                        p[5] = (unsigned char)(((w - 1) << 4) | (h - 1));
                        ns++;
                    }
                size_t enc_sz = 1 + 4 + 1 + (size_t)ns * 6;
                if (fits && enc_sz < raw_sz) {
                    *o++ = 2 | 8 | 16;               /* BgSpecified | AnySubrects | SubrectsColoured */
                    o = viz_put_px(o, bg);
                    *o++ = (unsigned char)ns;
                    memcpy(o, sub, (size_t)ns * 6); o += (size_t)ns * 6;
                    continue;
                }
            }
            *o++ = 1;                                /* Raw tile */
            for (int i = 0; i < n; i++) o = viz_put_px(o, t[i]);
        }
    }
    return o;
}

/* Build and send one FramebufferUpdate.  Returns 1 if something was sent. */
static int viz_vnc_send_update(void) {
    if (viz.client_fd <= 0 || viz.state != 3 || !viz.fbu_pending) return 0;
    int W = viz.width, H = viz.height;
    int rx0 = viz.req_x < 0 ? 0 : viz.req_x, ry0 = viz.req_y < 0 ? 0 : viz.req_y;
    int rx1 = viz.req_x + viz.req_w, ry1 = viz.req_y + viz.req_h;
    if (rx1 > W) rx1 = W;
    if (ry1 > H) ry1 = H;
    if (rx1 <= rx0 || ry1 <= ry0) { viz.fbu_pending = 0; return 0; }

    int tcx = (W + VIZ_TILE - 1) / VIZ_TILE, tcy = (H + VIZ_TILE - 1) / VIZ_TILE;
    static int rect_x[VIZ_MAX_RECTS], rect_y[VIZ_MAX_RECTS], rect_w[VIZ_MAX_RECTS], rect_h[VIZ_MAX_RECTS];
    int nr = 0, ndirty = 0, prev_lo = 0, prev_hi = 0;
    size_t len = 0;

    pthread_mutex_lock(&viz.fb_mutex);

    /* 1. dirty tiles inside the requested region */
    for (int ty = 0; ty < tcy; ty++)
        for (int tx = 0; tx < tcx; tx++) {
            int x0 = tx * VIZ_TILE, y0 = ty * VIZ_TILE;
            int x1 = x0 + VIZ_TILE < W ? x0 + VIZ_TILE : W, y1 = y0 + VIZ_TILE < H ? y0 + VIZ_TILE : H;
            unsigned char d = 0;
            if (x1 > rx0 && x0 < rx1 && y1 > ry0 && y0 < ry1) {
                if (viz.fbu_full) d = 1;
                else
                    for (int y = y0; y < y1 && !d; y++) {
                        const unsigned char *a = viz.framebuffer + ((size_t)y * W + x0) * 4;
                        const unsigned char *b = viz.shadow      + ((size_t)y * W + x0) * 4;
                        for (int x = 0; x < x1 - x0; x++)
                            if (viz_px(a + x * 4) != viz_px(b + x * 4)) { d = 1; break; }
                    }
            }
            viz.dirty[ty * tcx + tx] = d;
            ndirty += d;
        }
    if (!ndirty) { pthread_mutex_unlock(&viz.fb_mutex); return 0; }   /* idle: send nothing */

    /* 2. merge: horizontal runs per tile row, then extend vertically when the
     *    previous row had a run with identical x-extent. */
    for (int ty = 0; ty < tcy; ty++) {
        int lo = nr;
        for (int tx = 0; tx < tcx; ) {
            if (!viz.dirty[ty * tcx + tx]) { tx++; continue; }
            int tx2 = tx; while (tx2 < tcx && viz.dirty[ty * tcx + tx2]) tx2++;
            int x0 = tx * VIZ_TILE, x1 = tx2 * VIZ_TILE < W ? tx2 * VIZ_TILE : W;
            int y0 = ty * VIZ_TILE, y1 = y0 + VIZ_TILE < H ? y0 + VIZ_TILE : H;
            if (x0 < rx0) x0 = rx0;
            if (x1 > rx1) x1 = rx1;
            if (y0 < ry0) y0 = ry0;
            if (y1 > ry1) y1 = ry1;
            int merged = 0;
            for (int k = prev_lo; k < prev_hi; k++)
                if (rect_x[k] == x0 && rect_w[k] == x1 - x0 && rect_y[k] + rect_h[k] == y0) {
                    rect_h[k] += y1 - y0; merged = 1; break;
                }
            if (!merged && nr < VIZ_MAX_RECTS) {
                rect_x[nr] = x0; rect_y[nr] = y0; rect_w[nr] = x1 - x0; rect_h[nr] = y1 - y0; nr++;
            } else if (!merged) {            /* out of slots: fall back to one bounding rect */
                nr = 1; rect_x[0] = rx0; rect_y[0] = ry0; rect_w[0] = rx1 - rx0; rect_h[0] = ry1 - ry0;
                ty = tcy; break;
            }
            tx = tx2;
        }
        prev_lo = lo; prev_hi = nr;
    }

    /* 3. encode everything into the single tx buffer, update shadow */
    unsigned char *o = viz.txbuf;
    *o++ = 0; *o++ = 0; vnc_write_u16_be(o, (uint16_t)nr); o += 2;
    for (int i = 0; i < nr; i++) {
        vnc_write_u16_be(o + 0, (uint16_t)rect_x[i]); vnc_write_u16_be(o + 2, (uint16_t)rect_y[i]);
        vnc_write_u16_be(o + 4, (uint16_t)rect_w[i]); vnc_write_u16_be(o + 6, (uint16_t)rect_h[i]);
        vnc_write_u32_be(o + 8, viz.enc_hextile ? 5u : 0u);
        o += 12;
        o = viz.enc_hextile ? viz_enc_hextile(o, viz.framebuffer, W, rect_x[i], rect_y[i], rect_w[i], rect_h[i])
                            : viz_enc_raw    (o, viz.framebuffer, W, rect_x[i], rect_y[i], rect_w[i], rect_h[i]);
        for (int y = rect_y[i]; y < rect_y[i] + rect_h[i]; y++)
            memcpy(viz.shadow + ((size_t)y * W + rect_x[i]) * 4,
                   viz.framebuffer + ((size_t)y * W + rect_x[i]) * 4, (size_t)rect_w[i] * 4);
    }
    len = (size_t)(o - viz.txbuf);
    pthread_mutex_unlock(&viz.fb_mutex);        /* never block on the socket while holding it */

    viz.fbu_pending = 0; viz.fbu_full = 0;
    if (viz_send_full(viz.client_fd, viz.txbuf, len) < 0) { viz_vnc_disconnect(); return 0; }
    viz.net_bytes += len; viz.net_updates++; viz.net_rects += (unsigned)nr;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  VNC Thread Entry Point                                             */
/* ------------------------------------------------------------------ */

static void *viz_vnc_thread_func(void *arg) {
    (void)arg;

    fprintf(stderr, "[VIZ] VNC thread started, listening on port %d\n", viz.port);

    /* Set socket non-blocking */
    viz_set_nonblocking(viz.listen_fd);

    double last_update = 0.0;
    const double update_interval = 0.1; /* 10 Hz max; idle => nothing is sent at all */
    const int viz_netstat_enabled = getenv("PICOLM_VIZ_NETSTAT") != NULL;

    last_update = viz_mono_time();

    while (atomic_load_explicit(&viz.running, memory_order_relaxed)) {
        /* Accept new connections */
        if (viz.client_fd < 0 && viz.state < 1) {
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            int new_fd = accept(viz.listen_fd, (struct sockaddr *)&client_addr, &addr_len);

#ifdef _WIN32
            if (new_fd != (int)INVALID_SOCKET) {
#else
            if (new_fd > 0) {
#endif
                fprintf(stderr, "[VIZ] VNC: Connection from %s\n",
                        inet_ntoa(client_addr.sin_addr));
                int flag = 1;
                setsockopt(new_fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&flag, sizeof(flag));
                viz_set_nonblocking(new_fd);
                viz.client_fd = new_fd;
                viz.state = 0;
            }
        }

        /* Handle client */
        if (viz.client_fd > 0) {
            if (viz.state < 3) {
                if (viz_vnc_handle_handshake() < 0) {
                    viz_vnc_disconnect();
                }
            } else {
                viz_vnc_process_messages();
            }
        }

        /* Render only when data changed; send only when the client asked AND
         * something is different.  Idle => no render, no bytes. */
        double current = viz_mono_time();

        if (current - last_update >= update_interval) {
            last_update = current;
            int has_client = (viz.client_fd > 0 && viz.state == 3);

            if (has_client) {
                unsigned g = atomic_load_explicit(&viz_ring.gen, memory_order_acquire);
                if (g != viz.rendered_gen ||
                    atomic_exchange_explicit(&viz.update_requested, 0, memory_order_acq_rel)) {
                    pthread_mutex_lock(&viz.fb_mutex);
                    viz_render_heatmap();
                    pthread_mutex_unlock(&viz.fb_mutex);
                    viz.rendered_gen = g;
                }
                viz_vnc_send_update();

                if (viz_netstat_enabled && current - viz.net_t0 >= 2.0) {
                    fprintf(stderr, "[VIZ] net: %.1f KB/s  (%u updates, %u rects, %s)\n",
                            (double)viz.net_bytes / 1024.0 / (current - viz.net_t0),
                            viz.net_updates, viz.net_rects, viz.enc_hextile ? "hextile" : "raw");
                    viz.net_bytes = 0; viz.net_updates = viz.net_rects = 0; viz.net_t0 = current;
                }
            }
        }

        /* Sleep briefly to avoid busy-waiting */
        viz_sleep(0.016); /* ~60Hz poll rate, but frames at 5Hz */
    }

    /* Cleanup */
    if (viz.client_fd > 0) {
        _viz_close(viz.client_fd);
        viz.client_fd = -1;
    }
    if (viz.listen_fd >= 0) {
        _viz_close(viz.listen_fd);
        viz.listen_fd = -1;
    }

    fprintf(stderr, "[VIZ] VNC thread exiting\n");
    atomic_store_explicit(&viz.exited, 1, memory_order_release);
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

int viz_init(int width, int height, int port) {
    if (width < 64) width = 800;
    if (height < 32) height = 480;
    if (port < 1) port = 5900;

    /* Zero out state */
    memset(&viz, 0, sizeof(viz));
    memset(&viz_ring, 0, sizeof(viz_ring));

#ifdef _WIN32
    {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            fprintf(stderr, "[VIZ] WSAStartup failed\n");
            return -1;
        }
    }
#endif

    viz.width = width;
    viz.height = height;
    viz.model_name[0] = '\0';
    viz.n_layers = 0;
    viz.n_embd = 0;
    viz.global_min = 0.0f;
    viz.global_max = 1.0f;
    viz.display_mode = 0;
    viz.scroll_offset = 0;
    atomic_store(&viz.norm_mode, 1);   /* per-layer contrast by default */

    /* Initialize skip flags to 0 (not skipped) */
    for (int i = 0; i < VIZ_MAX_LAYERS; i++)
        atomic_store_explicit(&viz_skip_layer[i], 0, memory_order_relaxed);

    /* Initialize layer permutation to identity */
    viz_layer_permute_reset();

    /* Allocate framebuffer */
    viz.framebuffer = (unsigned char *)malloc((size_t)width * height * 4);
    if (!viz.framebuffer) {
        fprintf(stderr, "[VIZ] Failed to allocate framebuffer %dx%d\n", width, height);
        return -1;
    }
    memset(viz.framebuffer, 0, (size_t)width * height * 4);
    viz.shadow = (unsigned char *)calloc((size_t)width * height, 4);
    viz.dirty  = (unsigned char *)malloc((size_t)((width + 15) / 16) * (size_t)((height + 15) / 16));
    viz.txcap  = (size_t)width * height * 5 + 262144;   /* raw worst case + headers; hextile <= raw+1.5% */
    viz.txbuf  = (unsigned char *)malloc(viz.txcap);
    viz.fbu_full = 1;
    if (!viz.shadow || !viz.dirty || !viz.txbuf) {
        fprintf(stderr, "[VIZ] Failed to allocate transport buffers\n");
        return -1;
    }

    /* Initialize ring buffer */
    viz_ring_reset();
    viz_ring.n_layers = 0;
    {   /* optional fixed color scale (disables auto-calibration) */
        const char *lo = getenv("PICOLM_VIZ_LO"), *hi = getenv("PICOLM_VIZ_HI");
        if (lo && hi && atof(lo) > 0 && atof(hi) > atof(lo)) {
            viz.glo = log10f((float)atof(lo)); viz.ghi = log10f((float)atof(hi));
            for (int l = 0; l < VIZ_MAX_LAYERS; l++) { viz.lo[l] = viz.glo; viz.hi[l] = viz.ghi; }
            viz.fixed_scale = 1; viz.cal_valid = 1;
        }
    }

    /* Initialize mutex */
    pthread_mutex_init(&viz.fb_mutex, NULL);

    /* Create listening socket */
    viz.listen_fd = socket(AF_INET, SOCK_STREAM, 0);
#ifdef _WIN32
    if (viz.listen_fd == (int)INVALID_SOCKET) {
        fprintf(stderr, "[VIZ] Failed to create socket: %s\n",
                viz_socket_strerror(WSAGetLastError()));
#else
    if (viz.listen_fd < 0) {
        fprintf(stderr, "[VIZ] Failed to create socket: %s\n", strerror(errno));
#endif
        free(viz.framebuffer);
        viz.framebuffer = NULL;
        return -1;
    }

    int opt = 1;
    setsockopt(viz.listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

#ifdef _WIN32
    if (bind(viz.listen_fd, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "[VIZ] Failed to bind to port %d: %s\n", port,
                viz_socket_strerror(WSAGetLastError()));
#else
    if (bind(viz.listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[VIZ] Failed to bind to port %d: %s\n", port, strerror(errno));
#endif
        _viz_close(viz.listen_fd);
        free(viz.framebuffer);
        viz.framebuffer = NULL;
        return -1;
    }

#ifdef _WIN32
    if (listen(viz.listen_fd, VIZ_MAX_CLIENTS) == SOCKET_ERROR) {
        fprintf(stderr, "[VIZ] Failed to listen: %s\n",
                viz_socket_strerror(WSAGetLastError()));
#else
    if (listen(viz.listen_fd, VIZ_MAX_CLIENTS) < 0) {
        fprintf(stderr, "[VIZ] Failed to listen: %s\n", strerror(errno));
#endif
        _viz_close(viz.listen_fd);
        free(viz.framebuffer);
        viz.framebuffer = NULL;
        return -1;
    }

    viz.port = port;
    viz.client_fd = -1;
    viz.state = 0;
    atomic_store(&viz.client_connected, 0);
    atomic_store(&viz.update_requested, 0);
    atomic_store(&viz.running, 1);
    atomic_store(&viz.exited, 0);

    /* Start VNC thread */
    if (pthread_create(&viz.thread, NULL, viz_vnc_thread_func, NULL) != 0) {
        fprintf(stderr, "[VIZ] Failed to create VNC thread: %s\n", strerror(errno));
        _viz_close(viz.listen_fd);
        free(viz.framebuffer);
        viz.framebuffer = NULL;
        return -1;
    }


    atomic_store(&viz_ring.active, 1);
    fprintf(stderr, "[VIZ] Visualization initialized: %dx%d on port %d\n",
            width, height, port);
    fprintf(stderr, "[VIZ] Connect with: vncviewer %s:%d\n",
            "localhost", port);

    return 0;
}

void viz_set_model_info(int n_layers, int n_embd, const char *name) {
    viz.n_layers = n_layers;
    viz.n_embd = n_embd;
    viz_ring.n_layers = n_layers;
    if (name) {
        snprintf(viz.model_name, sizeof(viz.model_name), "%s", name);
    } else {
        snprintf(viz.model_name, sizeof(viz.model_name), "model_%dL_%dd", n_layers, n_embd);
    }
}

void viz_free(void) {
    /* 1. Signal the VNC thread to stop. */
    atomic_store_explicit(&viz.running, 0, memory_order_release);
    atomic_store(&viz_ring.active, 0);

    /* 2. Wake the thread if it's blocked in select() by writing a spurious byte
     * to the listen socket. The thread checks `running` each loop iteration. */
    if (viz.listen_fd >= 0) {
        char dummy = 0;
        sendto(viz.listen_fd, &dummy, 1, 0, NULL, 0);
    }

    /* 3. Wait for the thread to fully exit before freeing shared resources.
     * This prevents use-after-free on framebuffer and mutex destruction. */
    if (viz.thread != 0) {
        pthread_join(viz.thread, NULL);
    }

    /* 4. Now safe to free shared resources. */
    if (viz.framebuffer) {
        free(viz.framebuffer);
        viz.framebuffer = NULL;
    }
    free(viz.shadow); viz.shadow = NULL;
    free(viz.dirty);  viz.dirty  = NULL;
    free(viz.txbuf);  viz.txbuf  = NULL;
    pthread_mutex_destroy(&viz.fb_mutex);

    if (viz.listen_fd >= 0) {
        _viz_close(viz.listen_fd);
        viz.listen_fd = -1;
    }

    fprintf(stderr, "[VIZ] Freed.\n");

#ifdef _WIN32
    WSACleanup();
#endif
}

int viz_has_viewers(void) {
    return atomic_load_explicit(&viz.client_connected, memory_order_acquire);
}
