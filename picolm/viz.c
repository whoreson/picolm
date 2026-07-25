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
#define VIZ_TOKEN_RING       64   /* keep this many tokens of history */
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

    return ((unsigned int)b << 16) | ((unsigned int)g << 8) | (unsigned int)r;
}

/* A simpler grayscale colormap for debug */
static inline unsigned int viz_color_gray(float t) {
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    unsigned char v = (unsigned char)(t * 255.0f);
    return (unsigned int)v | ((unsigned int)v << 8) | ((unsigned int)v << 16);
}

/* ------------------------------------------------------------------ */
/*  Ring buffer for layer data                                         */
/* ------------------------------------------------------------------ */

/* One entry in the ring buffer: one layer's bucket-RMS data for one token. */
typedef struct {
    float buckets[VIZ_N_BUCKETS];
    unsigned int flags; /* bit 0: valid */
} viz_layer_entry_t;

/* The ring buffer: circular, lock-free single-writer multi-reader.
 * The inference thread writes (single producer), the VNC thread reads. */
typedef struct {
    /* Ring of per-layer entries for the current token.
     * Indexed by (write_pos % VIZ_RING_DEPTH).
     * Each "frame" is VIZ_MAX_LAYERS entries. */
    viz_layer_entry_t entries[VIZ_RING_DEPTH];

    /* Write position (only written by inference thread).
     * The VNC thread reads this with atomic load. */
    _Atomic unsigned int write_pos;

    /* Last successfully rendered position (written by VNC thread). */
    _Atomic unsigned int render_pos;

    /* Number of layers in the current model. */
    int n_layers;

    /* Total tokens generated (for column offset). */
    _Atomic unsigned int token_count;

    /* Number of token boundaries pushed. */
    _Atomic unsigned int token_boundaries;

} viz_ring_t;

static viz_ring_t viz_ring;

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

/* Check if a layer should be skipped.
 * Called from the inference thread. Atomic load, no blocking. */
int viz_layer_skip(int layer) {
    if (layer >= 0 && layer < VIZ_MAX_LAYERS)
        return atomic_load_explicit(&viz_skip_layer[layer], memory_order_relaxed);
    return 0;
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
/*  Ring buffer operations (lock-free, single writer)                  */
/* ------------------------------------------------------------------ */

void viz_push_layer(int layer, const float *data, int n_embd) {
    if (layer < 0 || layer >= VIZ_MAX_LAYERS) return;
    if (!data) return;

    unsigned int pos = atomic_load_explicit(&viz_ring.write_pos, memory_order_relaxed);
    viz_layer_entry_t *entry = &viz_ring.entries[pos % VIZ_RING_DEPTH];

    viz_compute_buckets(entry->buckets, data, n_embd, VIZ_N_BUCKETS);
    entry->flags = 1; /* valid */

    atomic_store_explicit(&viz_ring.write_pos, pos + 1, memory_order_release);
}

void viz_new_token(void) {
    atomic_fetch_add_explicit(&viz_ring.token_count, 1, memory_order_release);
    atomic_fetch_add_explicit(&viz_ring.token_boundaries, 1, memory_order_release);
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

static void viz_render_heatmap(void) {
    int fb_w = viz.width;
    int fb_h = viz.height;
    unsigned char *fb = viz.framebuffer;

    /* Clear framebuffer to dark background. */
    memset(fb, 0, (size_t)fb_w * fb_h * 4);

    /* Layout:
     * Top bar (0-20): title, stats
     * Heatmap area (20-H): layers on Y axis, tokens on X axis
     * Each cell = 1 layer x 1 token, colored by bucket-RMS magnitude
     *
     * The heatmap shows the LAST N tokens that fit on screen.
     * Y axis: layers (0 at top, n_layers-1 at bottom).
     * X axis: tokens (oldest on left, newest on right). */

    int top_bar_h = 20;
    int heatmap_y = top_bar_h + 2;
    int heatmap_h = fb_h - heatmap_y - 2;
    int heatmap_w = fb_w - 4;

    /* Title bar: model name, layer/token stats */
    unsigned int white = viz_color(1.0f);
    unsigned int dim = viz_color(0.5f);

    char title_buf[256];
    unsigned int wpos = atomic_load_explicit(&viz_ring.write_pos, memory_order_acquire);
    int connected = atomic_load_explicit(&viz.client_connected, memory_order_relaxed);
    int n_layers_tmp = viz.n_layers;
    if (n_layers_tmp < 1) n_layers_tmp = 1;
    int ntok = (int)(wpos / (unsigned int)n_layers_tmp);

    snprintf(title_buf, sizeof(title_buf), "PicoLM VIZ: %s [%dL, %dd] tok=%d lay=%d %s",
             viz.model_name, viz.n_layers, viz.n_embd, ntok,
             (int)(wpos % (unsigned int)n_layers_tmp),
             connected ? "VNC" : "  ");
    viz_draw_string(fb, fb_w, fb_h, 4, 2, title_buf, white);

    /* Subtitle explaining the axes */
    char subtitle[128];
    snprintf(subtitle, sizeof(subtitle), "Y=layer X=token per-layer RMS normalized");
    viz_draw_string(fb, fb_w, fb_h, 4, 9, subtitle, dim);

    /* Compute how many tokens we can display.
     * Each token = 1 column. We want at least 2px per column. */
    int max_tokens = heatmap_w / 2;
    if (max_tokens < 1) max_tokens = 1;
    if (max_tokens > VIZ_TOKEN_RING) max_tokens = VIZ_TOKEN_RING;

    /* Get the range of tokens to display. */
    int start_token = ntok - max_tokens;
    if (start_token < 0) start_token = 0;
    int num_tokens = ntok - start_token;

    /* Determine the cell height per layer. */
    int n_layers = viz.n_layers;
    if (n_layers < 1) n_layers = 1;
    int layer_h = heatmap_h / n_layers;
    if (layer_h < 1) layer_h = 1;

    /* Per-layer min/max for normalization.
     * Each layer has its own activation scale; normalizing per-layer
     * gives the best visual contrast.
     *
     * Only count COMPLETE tokens (all n_layers entries written).
     * Partial tokens (mid-forward-pass) are skipped to avoid stale data. */
    float layer_min[VIZ_MAX_LAYERS];
    float layer_max[VIZ_MAX_LAYERS];
    for (int l = 0; l < n_layers; l++) {
        layer_min[l] = 1e10f;
        layer_max[l] = -1e10f;
    }

    int valid_count = 0;
    /* Only use complete tokens: the last token might be mid-forward-pass */
    int complete_tokens = num_tokens;
    if (complete_tokens > 0) {
        /* Check if the last token is complete */
        unsigned int last_tok = (unsigned int)(start_token + complete_tokens - 1);
        unsigned int last_entry = (last_tok + 1) * (unsigned int)n_layers;
        if (last_entry > wpos) {
            complete_tokens--; /* skip incomplete last token */
        }
    }

    for (int t = 0; t < complete_tokens; t++) {
        int tok = start_token + t;
        unsigned int base_pos = (unsigned int)tok * (unsigned int)n_layers;

        for (int l = 0; l < n_layers; l++) {
            unsigned int entry_pos = base_pos + (unsigned int)l;
            if (entry_pos >= wpos) break;

            viz_layer_entry_t *entry = &viz_ring.entries[entry_pos % VIZ_RING_DEPTH];
            if (!(entry->flags & 1)) continue;

            float avg = 0.0f;
            for (int b = 0; b < VIZ_N_BUCKETS; b++) {
                avg += entry->buckets[b];
            }
            avg /= (float)VIZ_N_BUCKETS;

            if (avg < layer_min[l]) layer_min[l] = avg;
            if (avg > layer_max[l]) layer_max[l] = avg;
            valid_count++;
        }
    }

    /* Also track global min/max for the color bar label. */
    float gmin2 = 1e10f, gmax2 = -1e10f;
    for (int l = 0; l < n_layers; l++) {
        if (layer_min[l] < gmin2) gmin2 = layer_min[l];
        if (layer_max[l] > gmax2) gmax2 = layer_max[l];
    }
    if (gmax2 <= gmin2) { gmin2 = 0.0f; gmax2 = 1.0f; }

    /* If no valid data yet, just draw the title bar and return. */
    if (complete_tokens < 1 || valid_count < 1) {
        return;
    }

    /* Second pass: render the heatmap with per-layer normalization. */
    int col_w = heatmap_w / complete_tokens;
    if (col_w < 1) col_w = 1;

    for (int t = 0; t < complete_tokens; t++) {
        int tok = start_token + t;
        int cx = 2 + t * col_w;

        for (int l = 0; l < n_layers; l++) {
            unsigned int entry_pos = (unsigned int)tok * (unsigned int)n_layers + (unsigned int)l;
            if (entry_pos >= wpos) break;

            viz_layer_entry_t *entry = &viz_ring.entries[entry_pos % VIZ_RING_DEPTH];
            if (!(entry->flags & 1)) continue;

            /* Compute average bucket-RMS for this layer/token. */
            float sum = 0.0f;
            for (int b = 0; b < VIZ_N_BUCKETS; b++) {
                sum += entry->buckets[b];
            }
            float avg = sum / (float)VIZ_N_BUCKETS;

            /* Normalize within this layer's range.
             * Enforce a minimum range relative to global to prevent
             * wild color swings when a layer has near-uniform activations. */
            float lrange = layer_max[l] - layer_min[l];
            float global_range = gmax2 - gmin2;
            float min_range = global_range * 0.01f; /* 1% of global range */
            if (min_range < 1e-6f) min_range = 1e-6f;

            float norm;
            if (lrange < min_range) {
                /* Layer has very little variation; use global normalization
                 * so it still shows some color rather than being uniform. */
                if (global_range < 1e-6f) {
                    norm = 0.5f;
                } else {
                    norm = (avg - gmin2) / global_range;
                }
            } else {
                norm = (avg - layer_min[l]) / lrange;
            }

            /* Gamma correction: boost mid-range contrast for better visibility.
             * norm = pow(norm, 0.8) expands the middle, compresses extremes. */
            if (norm > 0.0f && norm < 1.0f) {
                norm = (float)pow((double)norm, 0.8);
            }

            unsigned int col = viz_color(norm);

            int cy = heatmap_y + l * layer_h;

            /* Draw the cell. If layer_h >= 2 and col_w >= 2, draw interior + border.
             * Otherwise just fill. */
            if (layer_h >= 2 && col_w >= 2) {
                /* Fill interior with color */
                for (int dy = 0; dy < layer_h - 1; dy++) {
                    for (int dx = 0; dx < col_w - 1; dx++) {
                        int px = cx + dx;
                        int py = cy + dy;
                        int idx = (py * fb_w + px) * 4;
                        fb[idx]     = (unsigned char)(col & 0xFF);
                        fb[idx + 1] = (unsigned char)((col >> 8) & 0xFF);
                        fb[idx + 2] = (unsigned char)((col >> 16) & 0xFF);
                        fb[idx + 3] = 255;
                    }
                }
                /* Border - dark gray */
                for (int dx = 0; dx < col_w; dx++) {
                    int idx = (cy * fb_w + cx + dx) * 4;
                    fb[idx] = 0x20; fb[idx+1] = 0x20; fb[idx+2] = 0x20;
                    idx = ((cy + layer_h - 1) * fb_w + cx + dx) * 4;
                    fb[idx] = 0x20; fb[idx+1] = 0x20; fb[idx+2] = 0x20;
                }
                for (int dy = 0; dy < layer_h; dy++) {
                    int idx = ((cy + dy) * fb_w + cx) * 4;
                    fb[idx] = 0x20; fb[idx+1] = 0x20; fb[idx+2] = 0x20;
                    idx = ((cy + dy) * fb_w + cx + col_w - 1) * 4;
                    fb[idx] = 0x20; fb[idx+1] = 0x20; fb[idx+2] = 0x20;
                }
            } else {
                /* Single pixel per cell */
                int px = cx;
                int py = cy;
                if (py < fb_h && px < fb_w) {
                    int idx = (py * fb_w + px) * 4;
                    fb[idx]     = (unsigned char)(col & 0xFF);
                    fb[idx + 1] = (unsigned char)((col >> 8) & 0xFF);
                    fb[idx + 2] = (unsigned char)((col >> 16) & 0xFF);
                    fb[idx + 3] = 255;
                }
            }
        }
    }

    /* Color bar on the right edge. */
    int bar_x = fb_w - 16;
    int bar_w = 12;
    int bar_h = heatmap_h;
    if (bar_x > 4 && bar_h > 0) {
        for (int y = 0; y < bar_h; y++) {
            float t = 1.0f - (float)y / (float)bar_h;
            unsigned int col = viz_color(t);
            for (int dx = 0; dx < bar_w; dx++) {
                int px = bar_x + dx;
                int py = heatmap_y + y;
                if (px < fb_w && py < fb_h) {
                    int idx = (py * fb_w + px) * 4;
                    fb[idx]     = (unsigned char)(col & 0xFF);
                    fb[idx + 1] = (unsigned char)((col >> 8) & 0xFF);
                    fb[idx + 2] = (unsigned char)((col >> 16) & 0xFF);
                    fb[idx + 3] = 255;
                }
            }
        }

        /* Labels on the color bar: max at top, min at bottom.
         * These show the global range; each layer is normalized independently. */
        char num_buf[16];
        snprintf(num_buf, sizeof(num_buf), "%.3f", gmax2);
        viz_draw_string(fb, fb_w, fb_h, bar_x + 1, heatmap_y, num_buf, dim);
        snprintf(num_buf, sizeof(num_buf), "%.3f", gmin2);
        viz_draw_string(fb, fb_w, fb_h, bar_x + 1, heatmap_y + bar_h - VIZ_FONT_H, num_buf, dim);
    }

    /* Layer labels + skip indicators on the left. */
    if (layer_h >= 10 && n_layers <= 64) {
        char lbl[16];
        for (int l = 0; l < n_layers; l++) {
            int cy = heatmap_y + l * layer_h + 1;
            snprintf(lbl, sizeof(lbl), "%d", l);
            viz_draw_char(fb, fb_w, fb_h, 1, cy, lbl[0], dim);
            if (l >= 9) {
                viz_draw_char(fb, fb_w, fb_h, 7, cy, lbl[1], dim);
            }
            /* Draw "S" indicator if layer is skipped */
            if (atomic_load_explicit(&viz_skip_layer[l], memory_order_relaxed)) {
                /* Bright red "S" at right edge before color bar */
                unsigned int red = 0xFFFFFF00; /* BGRA: bright red */
                int skip_lbl_x = fb_w - 30;
                if (skip_lbl_x < 20) skip_lbl_x = fb_w / 2;
                viz_draw_char(fb, fb_w, fb_h, skip_lbl_x, cy, 'S', red);
                /* Dim the entire row by drawing semi-transparent overlay */
                int row_start = heatmap_y + l * layer_h;
                int row_h = layer_h;
                for (int py = row_start; py < row_start + row_h && py < fb_h; py++) {
                    size_t row_off = (size_t)py * fb_w * 4;
                    for (int px = 0; px < fb_w; px++) {
                        size_t idx = row_off + px * 4;
                        /* Reduce all channels to 30% */
                        fb[idx]     = (unsigned char)(fb[idx] * 3 / 10);
                        fb[idx + 1] = (unsigned char)(fb[idx + 1] * 3 / 10);
                        fb[idx + 2] = (unsigned char)(fb[idx + 2] * 3 / 10);
                        /* Alpha stays 255 */
                    }
                }
            }
        }
    }

    /* Y-axis label: "LAYER" */
    /* Draw vertically... skip for now, too complex for this font. */
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
                for (uint16_t i = 0; i < num_enc; i++) {
                    if (viz_recv_full(viz.client_fd, buf, 4) < 0) {
                        viz_vnc_disconnect();
                        return;
                    }
                }
            }
            break;

        case 3: /* FramebufferUpdateRequest */
            if (viz_recv_full(viz.client_fd, buf, 9) < 0) {
                viz_vnc_disconnect();
                return;
            }
            {
                uint8_t incremental = buf[0];
                if (!incremental) {
                    atomic_store_explicit(&viz.update_requested, 1, memory_order_release);
                }
            }
            break;

        case 4: /* KeyEvent - we ignore but consume */
            if (viz_recv_full(viz.client_fd, buf, 7) < 0) {
                viz_vnc_disconnect();
            }
            break;

        case 5: /* PointerEvent - mouse click toggles layer skip */
            if (viz_recv_full(viz.client_fd, buf, 5) < 0) {
                viz_vnc_disconnect();
            }
            {
                uint8_t button_mask = buf[0];
                (void)vnc_read_u16_be(buf + 1); /* mouse_x, not used for layer skip */
                uint16_t mouse_y = vnc_read_u16_be(buf + 3);

                /* Left button (mask & 0x01) click on heatmap toggles layer skip */
                if (button_mask & 0x01) {
                    /* Layout: top bar = 20px, then 2px gap, then heatmap rows.
                     * Each layer gets (height - 22) / n_layers pixels. */
                    int top_bar_h = 20;
                    int heatmap_y = top_bar_h + 2;
                    int heatmap_h = viz.height - heatmap_y - 2;
                    int n_layers = viz.n_layers;
                    if (n_layers > 0 && mouse_y >= heatmap_y && mouse_y < (uint16_t)(heatmap_y + heatmap_h)) {
                        int layer_h = heatmap_h / n_layers;
                        if (layer_h > 0) {
                            int clicked_layer = (mouse_y - heatmap_y) / layer_h;
                            if (clicked_layer >= 0 && clicked_layer < n_layers) {
                                int old = atomic_load_explicit(&viz_skip_layer[clicked_layer],
                                                                memory_order_relaxed);
                                atomic_store_explicit(&viz_skip_layer[clicked_layer], !old,
                                                       memory_order_release);
                                fprintf(stderr, "[VIZ] Layer %d: %s\n",
                                        clicked_layer, !old ? "SKIPPED" : "ACTIVE");
                                atomic_store_explicit(&viz.update_requested, 1,
                                                       memory_order_release);
                            }
                        }
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

/* Send the current framebuffer to the VNC client as a raw encoding.
 * Converts BGRA (our format) to RGBX (VNC little-endian 32bpp format). */
static void viz_vnc_send_framebuffer(void) {
    if (viz.client_fd <= 0 || viz.state != 3) return;

    pthread_mutex_lock(&viz.fb_mutex);

    unsigned char head[4];
    unsigned char rect[12];

    /* FramebufferUpdate header */
    vnc_write_u8(head + 0, 0);
    vnc_write_u8(head + 1, 0);
    vnc_write_u16_be(head + 2, 1); /* one rectangle */

    if (viz_send_full(viz.client_fd, head, 4) < 0) {
        pthread_mutex_unlock(&viz.fb_mutex);
        viz_vnc_disconnect();
        return;
    }

    /* Rectangle header */
    vnc_write_u16_be(rect + 0, 0);
    vnc_write_u16_be(rect + 2, 0);
    vnc_write_u16_be(rect + 4, (uint16_t)viz.width);
    vnc_write_u16_be(rect + 6, (uint16_t)viz.height);
    vnc_write_u32_be(rect + 8, 0); /* Raw encoding */

    if (viz_send_full(viz.client_fd, rect, 12) < 0) {
        pthread_mutex_unlock(&viz.fb_mutex);
        viz_vnc_disconnect();
        return;
    }

    /* Send row by row, converting BGRA -> RGBX.
     * Our framebuffer: BGRA (B at lowest addr, A at highest).
     * VNC little-endian 32bpp: BGRX (B at lowest addr).
     * So we need to swap R and B, and zero the alpha. */
    unsigned char *row_buf = (unsigned char *)malloc((size_t)viz.width * 4);
    if (!row_buf) {
        pthread_mutex_unlock(&viz.fb_mutex);
        return;
    }

    int ok = 1;
    for (int y = 0; y < viz.height && ok; y++) {
        unsigned char *src = viz.framebuffer + (size_t)y * viz.width * 4;
        for (int x = 0; x < viz.width; x++) {
            /* Source: B G R A -> Dest: B G R X (but VNC expects RGB in LE = BGRX)
             * Our BGRA: src[x*4+0]=B, src[x*4+1]=G, src[x*4+2]=R, src[x*4+3]=A
             * VNC RGBX LE: buf[x*4+0]=B, buf[x*4+1]=G, buf[x*4+2]=R, buf[x*4+3]=0
             * So it's already in the right order! Just zero the alpha. */
            row_buf[x * 4 + 0] = src[x * 4 + 0]; /* B */
            row_buf[x * 4 + 1] = src[x * 4 + 1]; /* G */
            row_buf[x * 4 + 2] = src[x * 4 + 2]; /* R */
            row_buf[x * 4 + 3] = 0;              /* X */
        }

        if (viz_send_full(viz.client_fd, row_buf, (size_t)viz.width * 4) < 0) {
            ok = 0;
        }
    }

    free(row_buf);
    pthread_mutex_unlock(&viz.fb_mutex);

    if (!ok) {
        viz_vnc_disconnect();
        return;
    }

    atomic_store_explicit(&viz.update_requested, 0, memory_order_release);
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
    const double update_interval = 0.2; /* 5 Hz max update rate */

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

        /* Render and send frame at controlled rate */
        double current = viz_mono_time();

        if (current - last_update >= update_interval) {
            last_update = current;

            /* Always re-render the framebuffer (even without client, cheap enough) */
            /* Actually, only render if we have a connected client or need to update */
            int has_client = (viz.client_fd > 0 && viz.state == 3);

            if (has_client) {
                /* Render into framebuffer under mutex */
                pthread_mutex_lock(&viz.fb_mutex);
                viz_render_heatmap();
                pthread_mutex_unlock(&viz.fb_mutex);

                /* Send to client */
                viz_vnc_send_framebuffer();
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

    /* Allocate framebuffer */
    viz.framebuffer = (unsigned char *)malloc((size_t)width * height * 4);
    if (!viz.framebuffer) {
        fprintf(stderr, "[VIZ] Failed to allocate framebuffer %dx%d\n", width, height);
        return -1;
    }
    memset(viz.framebuffer, 0, (size_t)width * height * 4);

    /* Initialize ring buffer */
    atomic_store(&viz_ring.write_pos, 0);
    atomic_store(&viz_ring.render_pos, 0);
    atomic_store(&viz_ring.token_count, 0);
    atomic_store(&viz_ring.token_boundaries, 0);
    viz_ring.n_layers = 0;

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