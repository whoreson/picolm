#define _POSIX_C_SOURCE 199309L
// backend_vulkan.c
//
// Vulkan compute backend for PicoLM. Implements the backend_gpu.h C ABI using
// the Vulkan loader library (no SDK needed at runtime; just the driver-shipped
// vulkan-1.dll / libvulkan.so).
//
// Build:  make vulkan
//   Requires: vulkan loader headers, glslc (shaderc) for shader compilation.
//   Link:     -lvulkan
//
// The matmul kernels are hand-written GLSL compute shaders (compiled to SPIR-V
// with glslc). The device-side dequant matches PicoLM's CPU reference in quant.c.
//
// Memory: HOST_VISIBLE|HOST_COHERENT mapped buffers. On discrete GPUs all data
// crosses PCIe. On APUs with unified memory this can be zero-copy.
//
// Supported quants (Phase 1): F32, F16, Q4_0, Q8_0.
// All other backend_gpu.h functions return 0 (fall back to CPU).

#include "backend_gpu.h"  // includes quant.h (gguf_type_t enum); extern "C" ABI

#include <vulkan/vulkan.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
static double _vk_now(void) {
    static double freq = 0;
    if (!freq) {
        LARGE_INTEGER f; QueryPerformanceFrequency(&f); freq = (double)f.QuadPart;
    }
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart / freq * 1000.0;
}
#define vk_now() _vk_now()
#else
static double vk_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
#endif

#define VKCHECK(x, what) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[VK] %s:%d %s failed: %s (%d)\n", __FILE__, __LINE__, what, vk_result_str(_r), (int)_r); \
    return 0; } } while (0)

// Forward decls for shutdown cleanup
static VkBuffer unwrap_buf_offset(const void *p, VkDeviceSize *out_off);
static void free_dev_buf(void *p);
static uint16_t f32tof16_cpu(float v);

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

typedef struct {
    VkBuffer buf; VkDeviceMemory mem; void *ptr; size_t cap;
} Scratch;

struct picolm_gpu_tensor {
    VkBuffer wbuf; VkDeviceMemory wmem;
    gguf_type_t qtype;
    int I, O, device;
    size_t row_bytes;   // bytes per weight row on CPU side
    size_t row_words;   // uint32 words per weight row on GPU side (padded)
    size_t wbytes;      // total GPU weight buffer size
};

typedef struct VkWArena {
    VkDeviceMemory mem; uint8_t *base;
    size_t cap, off;
    struct VkWArena *next;
} VkWArena;

#define VK_WARENA_BLOCK ((size_t)256 << 20)

// Human-readable Vulkan result string
static const char *vk_result_str(VkResult r) {
    switch ((int)r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
        case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
        default: return "(unknown)";
    }
}

static struct {
    int ready;
    int devices[PICOLM_GPU_MAX_DEVICES];
    int device_count;

    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    uint32_t memtype;
    uint32_t memtype_cached;
    uint32_t memtype_local;    // DEVICE_LOCAL memory type for pipeline buffers
    uint32_t memtype_staging;  // HOST_VISIBLE for H2D staging

    // Unified pipeline layout (shared by ALL shaders to fix RADV push constant bug)
    VkDescriptorSetLayout dsl_unified;
    VkPipelineLayout plyt_unified;

    // Matmul pipeline
    VkPipeline pipe;
    VkShaderModule shader;
    VkDescriptorPool dpool;
    VkDescriptorSet dset[256];
    uint32_t dset_idx;  // ring buffer index for descriptor sets

    // RMSNorm pipeline
    VkShaderModule shader_nrm;
    VkPipeline pipe_nrm;
    VkDescriptorPool dpool_nrm;
    VkDescriptorSet dset_nrm;
    VkDescriptorPool dpool_nrm_ring;
    VkDescriptorSet dset_nrm_ring[256];  // ring buffer for batched prefill
    // Push constant staging buffer (legacy, not used with unified layout)
    VkBuffer buf_pc_nrm;
    VkDeviceMemory mem_pc_nrm;

    // Elementwise pipeline (residual_add, silu_mul, rope, f16pack)
    VkShaderModule shader_elem;
    VkPipeline pipe_elem;
    VkDescriptorPool dpool_elem;
    VkDescriptorSet dset_elem;
    VkDescriptorPool dpool_elem_ring;
    VkDescriptorSet dset_elem_ring[256];  // ring buffer for batched prefill

    // Attention decode pipeline
    VkShaderModule shader_attn_dec;
    VkPipeline pipe_attn_dec;
    VkDescriptorPool dpool_attn_dec;
    VkDescriptorSet dset_attn_dec;

    // Attention prefill pipeline
    VkShaderModule shader_attn_prefill;
    VkPipeline pipe_attn_prefill;
    VkDescriptorPool dpool_attn_prefill;
    VkDescriptorSet dset_attn_prefill;

    // Q8_0 quantize pipeline (F32->Q8_0 on device)
    VkShaderModule shader_quantize;
    VkPipeline pipe_quantize;
    VkDescriptorPool dpool_quantize;
    VkDescriptorSet dset_quantize;
    VkDescriptorSet dset_quantize_ring[256];

    // Q8_0 x Q8_0 matmul pipeline (pre-quantized activations)
    VkShaderModule shader_q8q8;
    VkPipeline pipe_q8q8;
    VkDescriptorPool dpool_q8q8;
    VkDescriptorSet dset_q8q8;
    VkDescriptorSet dset_q8q8_ring[256];

    // F16 pack: shares elementwise resources (shader, pipeline, dpool, dset)
    VkShaderModule shader_f16pack;
    VkPipeline pipe_f16pack;
    VkDescriptorPool dpool_f16pack;
    VkDescriptorSet dset_f16pack;

    // KV cache store (device-side F32->F16 pack)
    VkShaderModule shader_kv_store;
    VkPipeline pipe_kv_store;
    VkDescriptorPool dpool_kv_store;
    VkDescriptorSet dset_kv_store_ring[256];

    VkCommandPool cpool;
    VkCommandBuffer cmd;       // Phase 1: single-command sync
    VkCommandBuffer cmd_nrm;   // RMSNorm host-facing

    // Phase 2: device-native pipeline command buffers
    VkCommandBuffer cmd_dev;   // Accumulates device-native work
    VkCommandBuffer cmd_xfer;  // Transfer commands (staging H2D)
    VkFence fence;             // Phase 1 sync fence
    VkFence fence_dev;         // Phase 2 sync fence (for picolm_gpu_sync)
    VkFence fence_xfer;        // Transfer fence

    Scratch x_buf, y_buf;      // Phase 1 scratch (HOST_VISIBLE)

    // Phase 2: device-local pipeline buffers
    int pipe_ready;
    VkBuffer pipe_x, pipe_xb, pipe_q, pipe_k, pipe_v,
             pipe_attn_out, pipe_ffn_norm, pipe_gate, pipe_up;
    VkDeviceMemory pipe_x_m, pipe_xb_m, pipe_q_m, pipe_k_m, pipe_v_m,
                   pipe_attn_out_m, pipe_ffn_norm_m, pipe_gate_m, pipe_up_m;
    size_t pipe_dim, pipe_q_dim, pipe_kv_dim, pipe_ffn_hidden;
    /* Cached vk_dev_buf_t* wrappers for pipe accessors (avoids realloc + list growth) */
    void *pipe_x_d, *pipe_xb_d, *pipe_q_d, *pipe_k_d, *pipe_v_d,
          *pipe_attn_out_d, *pipe_ffn_norm_d, *pipe_gate_d, *pipe_up_d;

    // Phase 2: batch pipeline buffers (prefill)
    int pipe_b_ready;
    VkBuffer pipe_x_b, pipe_xb_b, pipe_q_b, pipe_k_b, pipe_v_b,
             pipe_attn_out_b, pipe_ffn_norm_b, pipe_gate_b, pipe_up_b;
    VkDeviceMemory pipe_x_b_m, pipe_xb_b_m, pipe_q_b_m, pipe_k_b_m, pipe_v_b_m,
                   pipe_attn_out_b_m, pipe_ffn_norm_b_m, pipe_gate_b_m, pipe_up_b_m;
    VkBuffer pipe_logits_buf; VkDeviceMemory pipe_logits_m;
    size_t pipe_b_max_seq_len, pipe_b_xb_stride;
    /* Cached wrappers for batch pipes */
    void *pipe_x_b_d, *pipe_xb_b_d, *pipe_q_b_d, *pipe_k_b_d, *pipe_v_b_d,
          *pipe_attn_out_b_d, *pipe_ffn_norm_b_d, *pipe_gate_b_d, *pipe_up_b_d, *pipe_logits_d;

    // Phase 2: KV cache on device
    VkBuffer kv_k_buf, kv_v_buf;
    VkDeviceMemory kv_k_mem, kv_v_mem;
    size_t kv_k_bytes, kv_v_bytes;
    void *kv_k_d, *kv_v_d;  /* cached wrappers */
    void *kv_k_mapped, *kv_v_mapped;  /* CPU-mapped for readback */
    // Mapped pipe buffers (for direct CPU readback on llvmpipe/RADV)
    void *pipe_k_b_mapped, *pipe_v_b_mapped, *pipe_x_b_mapped;
    // KV staging buffer (device-local, for KAVERI shader read workaround)
    VkBuffer kv_staging_buf;
    VkDeviceMemory kv_staging_mem;
    size_t kv_staging_cap;

    // Phase 2: staging buffer for H2D transfers
    VkBuffer staging_buf;
    VkDeviceMemory staging_mem;
    void *staging_ptr;
    size_t staging_cap;

    // Phase 2: quantize scratch buffers (for matmul_dev Q8 path)
    VkBuffer q8_xq_buf, q8_xd_buf;
    VkDeviceMemory q8_xq_mem, q8_xd_mem;
    size_t q8_xq_cap, q8_xd_cap;
    /* Cached wrappers for q8 scratch */
    void *q8_xq_d, *q8_xd_d;

    // Command caching for matmul
    picolm_gpu_tensor_t *bound_tensor;
    int bound_S, bound_I, bound_O;
    VkBuffer bound_xbuf, bound_ybuf;
    int cmd_ready;

    // Command caching for rmsnorm
    int nrm_cmd_ready;
    int nrm_bound_S, nrm_bound_D;

    VkWArena *warena;
    size_t used_bytes;

    int in_batch;   // Phase 1 refactor: non-zero when recording into a single cmd buffer

    char device_name[256];
    VkPhysicalDeviceMemoryProperties mem_props;
    VkPhysicalDeviceProperties dev_props;
    int subgroup_size;
} G;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static long g_vk_spin_us = -1;

static VkResult vk_fence_wait(VkDevice dev, VkFence f) {
    if (g_vk_spin_us < 0) {
        const char *e = getenv("PICOLM_VK_SPIN_US");
        g_vk_spin_us = e ? atol(e) : 300;
        if (g_vk_spin_us < 0) g_vk_spin_us = 0;
    }
    if (g_vk_spin_us > 0) {
        double t0 = vk_now();
        do {
            VkResult r = vkGetFenceStatus(dev, f);
            if (r != VK_NOT_READY) return r;
        } while ((vk_now() - t0) * 1000.0 < (double)g_vk_spin_us);
    }
    return vkWaitForFences(dev, 1, &f, VK_TRUE, 10000000000ULL);
}

static VkResult vk_fence_wait_timeout(VkDevice dev, VkFence f, uint64_t timeout_ns) {
    if (g_vk_spin_us > 0) {
        double t0 = vk_now();
        do {
            VkResult r = vkGetFenceStatus(dev, f);
            if (r != VK_NOT_READY) return r;
        } while ((vk_now() - t0) * 1000.0 < (double)g_vk_spin_us);
    }
    return vkWaitForFences(dev, 1, &f, VK_TRUE, timeout_ns);
}

static int pick_memtype(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    int best = -1;
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) return (int)i;
            if (best < 0) best = (int)i;
        }
    }
    return best;
}

static int pick_memtype_cached(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) return (int)i;
    }
    return pick_memtype(phys);
}

static int pick_memtype_local(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    // Find the largest device-local heap index
    uint32_t best_heap = (uint32_t)-1;
    size_t best_heap_size = 0;
    for (uint32_t i = 0; i < m.memoryHeapCount; i++) {
        if (m.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (m.memoryHeaps[i].size > best_heap_size) {
                best_heap_size = m.memoryHeaps[i].size;
                best_heap = i;
            }
        }
    }
    if (best_heap == (uint32_t)-1) return -1;
    // Prefer DEVICE_LOCAL + HOST_VISIBLE types (for direct CPU readback)
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            m.memoryTypes[i].heapIndex == best_heap) {
            return (int)i;
        }
    }
    // Pick the first DEVICE_LOCAL memory type backed by the largest heap
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            m.memoryTypes[i].heapIndex == best_heap) {
            return (int)i;
        }
    }
    // Fallback: any device-local type
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        if (m.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            return (int)i;
    }
    return -1;
}

static int alloc_device_local(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer(dev)");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = G.memtype_local};
    VkResult r = vkAllocateMemory(G.dev, &ai, NULL, mem);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[VK] alloc_device_local FAILED: bytes=%zu req.size=%zu memtype=%u error=%d\n",
                bytes, (size_t)req.size, G.memtype_local, (int)r);
        vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE;
        return 0;
    }
    VKCHECK(vkBindBufferMemory(G.dev, *buf, *mem, 0), "vkBindBufferMemory(dev)");
    return 1;
}

static int alloc_hostvis_mt(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem,
                            void **ptr, uint32_t memtype) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = memtype};
    VKCHECK(vkAllocateMemory(G.dev, &ai, NULL, mem), "vkAllocateMemory");
    VKCHECK(vkBindBufferMemory(G.dev, *buf, *mem, 0), "vkBindBufferMemory");
    if (ptr) {
        VKCHECK(vkMapMemory(G.dev, *mem, 0, bytes, 0, ptr), "vkMapMemory");
            }
    return 1;
}

static int alloc_hostvis_mt2(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem,
                            void **ptr, uint32_t memtype, VkBufferUsageFlags usage) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer(hv2)");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = memtype};
    VKCHECK(vkAllocateMemory(G.dev, &ai, NULL, mem), "vkAllocateMemory(hv2)");
    VKCHECK(vkBindBufferMemory(G.dev, *buf, *mem, 0), "vkBindBufferMemory(hv2)");
    if (ptr) {
        VKCHECK(vkMapMemory(G.dev, *mem, 0, bytes, 0, ptr), "vkMapMemory(hv2)");
            }
    return 1;
}

static int alloc_hostvis(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr) {
    return alloc_hostvis_mt(bytes, buf, mem, ptr, G.memtype);
}

static int scratch_reserve_mt(Scratch *s, size_t bytes, uint32_t memtype) {
    if (s->cap >= bytes) return 1;
    if (s->buf) {
        if (s->ptr) vkUnmapMemory(G.dev, s->mem);
        vkDestroyBuffer(G.dev, s->buf, NULL);
        vkFreeMemory(G.dev, s->mem, NULL);
    }
    s->buf = VK_NULL_HANDLE; s->cap = 0; s->ptr = NULL;
    if (!alloc_hostvis_mt(bytes, &s->buf, &s->mem, &s->ptr, memtype)) return 0;
    s->cap = bytes;
    return 1;
}

static int scratch_reserve(Scratch *s, size_t bytes) {
    return scratch_reserve_mt(s, bytes, G.memtype);
}

// ---------------------------------------------------------------------------
// GGUF row size helpers
// ---------------------------------------------------------------------------

static size_t gguf_row_bytes(gguf_type_t q, int I) {
    switch (q) {
        case GGUF_TYPE_F32:    return (size_t)I * 4;
        case GGUF_TYPE_F16:    return (size_t)I * 2;
        case GGUF_TYPE_Q8_0:   return (size_t)((I + 31) / 32) * 34;
        case GGUF_TYPE_Q4_0:   return (size_t)((I + 31) / 32) * 18;
        case GGUF_TYPE_Q4_K:   return (size_t)((I + 255) / 256) * 144;
        case GGUF_TYPE_Q5_K:   return (size_t)((I + 255) / 256) * 176;
        case GGUF_TYPE_Q6_K:   return (size_t)((I + 255) / 256) * 210;
        case GGUF_TYPE_BF16:   return (size_t)I * 2;
        case GGUF_TYPE_Q4_0_4_4: return (size_t)((I + 31) / 32) * 18;
        case GGUF_TYPE_Q4_0_4_8: return (size_t)((I + 31) / 32) * 18;
        case GGUF_TYPE_Q1_0:   return (size_t)((I + 127) / 128) * 18;
        case GGUF_TYPE_Q2_0:   return (size_t)((I + 127) / 128) * 34;
        case GGUF_TYPE_Q2_K:   return (size_t)((I + 255) / 256) * 84;
        case GGUF_TYPE_Q3_K:   return (size_t)((I + 255) / 256) * 110;
        default: return 0;
    }
}

static int gguf_row_words(gguf_type_t q, int I) {
    size_t rb = gguf_row_bytes(q, I);
    return (int)((rb + 3) / 4);
}

// ---------------------------------------------------------------------------
// Weight arena suballocator
// ---------------------------------------------------------------------------

static int arena_suballoc(size_t bytes, VkBuffer *buf, void **ptr) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    if (!(req.memoryTypeBits & (1u << G.memtype))) {
        vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0;
    }
    VkDeviceMemory mem;
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = G.memtype};
    VKCHECK(vkAllocateMemory(G.dev, &ai, NULL, &mem), "vkAllocateMemory");
    VKCHECK(vkBindBufferMemory(G.dev, *buf, mem, 0), "vkBindBufferMemory");
    VkWArena *a = calloc(1, sizeof(*a));
    if (!a) {
        vkFreeMemory(G.dev, mem, NULL);
        vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0;
    }
    a->mem = mem;
    a->cap = req.size;
    if (ptr) {
        VKCHECK(vkMapMemory(G.dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&a->base), "vkMapMemory");
        *ptr = a->base;
    }
    a->next = G.warena; G.warena = a;
    return 1;
}

// ---------------------------------------------------------------------------
// SPIR-V shader loading
// ---------------------------------------------------------------------------

static VkShaderModule load_spv(VkDevice dev, const char *basename) {
    // Search paths: PICOLM_VK_SHADER_DIR, ./shaders/, shaders/, basename
    static char path[512];
    const char *sd = getenv("PICOLM_VK_SHADER_DIR");
    if (sd) {
        snprintf(path, sizeof(path), "%s/%s", sd, basename);
        FILE *f = fopen(path, "rb");
        if (f) { fclose(f); goto try_load; }
    }
    snprintf(path, sizeof(path), "shaders/%s", basename);
    { FILE *f = fopen(path, "rb"); if (f) { fclose(f); goto try_load; } }
    snprintf(path, sizeof(path), "./shaders/%s", basename);
    { FILE *f = fopen(path, "rb"); if (f) { fclose(f); goto try_load; } }
    snprintf(path, sizeof(path), "%s", basename);

try_load:
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[VK] cannot open %s\n", basename); return VK_NULL_HANDLE; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n % 4 != 0) {
        fprintf(stderr, "[VK] bad SPIR-V size %ld in %s\n", n, path);
        fclose(f); return VK_NULL_HANDLE;
    }
    uint32_t *code = malloc((size_t)n);
    if (!code) { fclose(f); return VK_NULL_HANDLE; }
    if (fread(code, 1, (size_t)n, f) != (size_t)n) {
        fclose(f); free(code); return VK_NULL_HANDLE;
    }
    fclose(f);
    VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)n, .pCode = code};
    VkShaderModule m;
    VkResult r = vkCreateShaderModule(dev, &si, NULL, &m);
    free(code);
    return r == VK_SUCCESS ? m : VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Pipeline builder (nbind storage buffers, pc_size push constants)
// ---------------------------------------------------------------------------

// Build pipeline using a SHARED descriptor set layout + pipeline layout.
// All shaders share G.dsl_unified (4 bindings) and G.plyt_unified (48-byte PC range).
// Each shader gets its own VkPipeline and VkDescriptorSet.
static int build_pipeline_unified(VkDevice dev, VkShaderModule shader,
                                  VkPipeline *pipe, VkDescriptorPool *dpool,
                                  VkDescriptorSet *dset, int nsets) {
    // Create compute pipeline using the unified layout
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = G.plyt_unified};
    VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, pipe), "pipeline");
    // Descriptor pool + set(s) for this shader
    // Allocate one at a time to avoid RADV KAVERI pool fragmentation bug
    int ndesc = nsets * 4;  // 4 bindings per set
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = (uint32_t)ndesc};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = (uint32_t)nsets, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(dev, &dpi, NULL, dpool), "descPool");
    for (int si = 0; si < nsets; si++) {
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = *dpool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl_unified};
        VkResult aret = vkAllocateDescriptorSets(dev, &dsa, &dset[si]);
        if (aret != VK_SUCCESS) {
            fprintf(stderr, "[VK] allocDescSet[%d](%d sets, %d desc) failed: %d\n",
                    si, nsets, ndesc, (int)aret);
            VKCHECK(aret, "allocDescSet");
        }
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
static int staging_ensure(size_t bytes);
static VkBuffer unwrap_buf_offset(const void *ptr, VkDeviceSize *out_off);
static VkDescriptorBufferInfo desc_buf_info(const void *ptr);

// ---------------------------------------------------------------------------
// Descriptor update
// ---------------------------------------------------------------------------

static void wr_desc(VkDescriptorSet set, int n, const VkDescriptorBufferInfo *bi) {
    VkWriteDescriptorSet w[8];
    for (int i = 0; i < n; i++) w[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
        .dstBinding = (uint32_t)i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(G.dev, (uint32_t)n, w, 0, NULL);
}

// Ring-buffer descriptor set update for matmul path.
// Each call advances G.dset_idx, ensuring the GPU sees a fresh set.
static void wr_desc_ring(int n, const VkDescriptorBufferInfo *bi) {
    VkDescriptorSet set = G.dset[G.dset_idx % 256];
    G.dset_idx++;
    wr_desc(set, n, bi);
}

// ---------------------------------------------------------------------------
// INIT / SHUTDOWN
// ---------------------------------------------------------------------------

int picolm_gpu_init(const int *devices, int count) {
    if (G.ready) return 1;
    (void)devices; (void)count;

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "PicoLM", .applicationVersion = VK_MAKE_VERSION(1,0,0),
        .pEngineName = "PicoLM", .engineVersion = VK_MAKE_VERSION(1,0,0),
        .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app};
    VKCHECK(vkCreateInstance(&ici, NULL, &G.inst), "vkCreateInstance");

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(G.inst, &nd, NULL);
    if (!nd) { fprintf(stderr, "[VK] no physical devices\n"); return 0; }
    VkPhysicalDevice devs[16]; if (nd > 16) nd = 16;
    vkEnumeratePhysicalDevices(G.inst, &nd, devs);

    G.phys = devs[0];
    int bestrank = -1;
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        int rank = (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)   ? 4 :
                   (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 3 :
                   (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)    ? 2 :
                   (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_OTHER)          ? 1 : 0;
        if (rank > bestrank) { bestrank = rank; G.phys = devs[i]; }
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &nq, NULL);
    VkQueueFamilyProperties qf[32]; if (nq > 32) nq = 32;
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &nq, qf);
    G.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { G.qfam = i; break; }
    if (G.qfam == UINT32_MAX) { fprintf(stderr, "[VK] no compute queue\n"); return 0; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = G.qfam, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
    VKCHECK(vkCreateDevice(G.phys, &di, NULL, &G.dev), "vkCreateDevice");
    vkGetDeviceQueue(G.dev, G.qfam, 0, &G.queue);

    // Print device limits for debugging
    VkPhysicalDeviceProperties pdev;
    vkGetPhysicalDeviceProperties(G.phys, &pdev);
    fprintf(stderr, "[VK] limits: maxSharedMem=%u maxWGSize=%u maxWGCount=%u maxStorageBuf=%u\n",
            (unsigned)pdev.limits.maxComputeSharedMemorySize,
            (unsigned)pdev.limits.maxComputeWorkGroupSize[0],
            (unsigned)pdev.limits.maxComputeWorkGroupCount[0],
            (unsigned)pdev.limits.maxStorageBufferRange);

    int mt = pick_memtype(G.phys);
    if (mt < 0) { fprintf(stderr, "[VK] no host-visible memory\n"); return 0; }
    G.memtype = (uint32_t)mt;
    G.memtype_cached = (uint32_t)pick_memtype_cached(G.phys);
    vkGetPhysicalDeviceMemoryProperties(G.phys, &G.mem_props);

    // Device-local memory for pipeline buffers
    int mtl = pick_memtype_local(G.phys);
    if (mtl < 0) {
        fprintf(stderr, "[VK] no device-local memory, falling back to host-visible\n");
        G.memtype_local = G.memtype;
    } else {
        G.memtype_local = (uint32_t)mtl;
    }

    // WA for RADV+amdgpu (Mesa 22.3.6) on VEGA20: D2D copy from
    // host-visible staging (type[3]) to device-local (type[0]) produces
    // garbage data. Force pipeline buffers to use host-visible memory
    // so H2D is a direct memcpy instead of D2D copy.
    // Detect by checking if device-local heap != host-visible heap
    // and the driver is RADV with old Mesa.
    {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(G.phys, &mp);
        // If type[0] (device-local) and type[3] (host-vis) are on
        // different heaps, and we have RADV, use host-visible for
        // pipeline buffers to avoid the D2D copy bug.
        uint32_t local_heap = mp.memoryTypes[G.memtype_local].heapIndex;
        uint32_t hvis_heap = mp.memoryTypes[G.memtype].heapIndex;
        if (local_heap != hvis_heap) {
            // Check device name for VEGA/raven/raven2/renoir
            VkPhysicalDeviceProperties pdp;
            vkGetPhysicalDeviceProperties(G.phys, &pdp);
            char name[256];
            strncpy(name, pdp.deviceName, 255);
            name[255] = 0;
            if (strstr(name, "VEGA") || strstr(name, "raven") ||
                strstr(name, "Raven") || strstr(name, "renoir") ||
                strstr(name, " Renoir")) {
                fprintf(stderr, "[VK] WA: VEGA/raven detected, using host-visible pipeline buffers (D2D copy bug)\n");
                G.memtype_local = G.memtype;
            }
        }
    }
    // Print heap info for debugging
    fprintf(stderr, "[VK] memory heaps: %u types, %u heaps\n",
            (unsigned)G.mem_props.memoryTypeCount, (unsigned)G.mem_props.memoryHeapCount);
    for (uint32_t i = 0; i < G.mem_props.memoryHeapCount; i++) {
        fprintf(stderr, "[VK]   heap[%u]: %zu MB, flags=%u (device_local=%d)\n",
            i, G.mem_props.memoryHeaps[i].size / (1024*1024),
            (unsigned)G.mem_props.memoryHeaps[i].flags,
            !!(G.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT));
    }
    for (uint32_t i = 0; i < G.mem_props.memoryTypeCount; i++) {
        fprintf(stderr, "[VK]   type[%u]: heap=%u flags=%u (dev_local=%d host_vis=%d coherent=%d)\n",
            i, G.mem_props.memoryTypes[i].heapIndex,
            (unsigned)G.mem_props.memoryTypes[i].propertyFlags,
            !!(G.mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            !!(G.mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
            !!(G.mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    }
    G.memtype_staging = G.memtype;

    vkGetPhysicalDeviceProperties(G.phys, &G.dev_props);
    // Subgroup size: AMD GCN = 64, NVIDIA = 32. Use vendor ID to detect.
    if (G.dev_props.vendorID == 0x1002) G.subgroup_size = 64;  // AMD
    else if (G.dev_props.vendorID == 0x10DE) G.subgroup_size = 32;  // NVIDIA
    else G.subgroup_size = 32;  // default

    // ============================================================
// UNIFIED PIPELINE LAYOUT (Fixes RADV push constant bug)
// All shaders share one DSL (4 bindings) and one pipeline layout (48-byte PC).
// This avoids the RADV bug where independent pipeline layouts corrupt
// push constants beyond byte 7 on KAVERI (GCN1).
// ============================================================
    {
        VkDescriptorSetLayoutBinding b[4];
        for (int i = 0; i < 4; i++) b[i] = (VkDescriptorSetLayoutBinding){
            .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
        VkDescriptorSetLayoutCreateInfo dsli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4, .pBindings = b};
        VKCHECK(vkCreateDescriptorSetLayout(G.dev, &dsli, NULL, &G.dsl_unified), "unifiedDSL");
        VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 48};
        VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &G.dsl_unified,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
        VKCHECK(vkCreatePipelineLayout(G.dev, &pli, NULL, &G.plyt_unified), "unifiedPlyt");
    }

    // Load matmul shader
    G.shader = load_spv(G.dev, "qmatmul_vk.spv");
    if (!G.shader) { fprintf(stderr, "[VK] failed to load qmatmul_vk.spv\n"); return 0; }
    // Matmul: 4 sets for ring buffer
    if (!build_pipeline_unified(G.dev, G.shader, &G.pipe, &G.dpool, G.dset, 256)) return 0;

    // RMSNorm shader (push: int S, int D, float eps, int x_stride = 16 bytes)
    G.shader_nrm = load_spv(G.dev, "rmsnorm_vk.spv");
    if (!G.shader_nrm) {
        fprintf(stderr, "FATAL: [VK] missing rmsnorm_vk.spv -- GPU backend unusable\n");
        goto init_fail;
    }
    if (!build_pipeline_unified(G.dev, G.shader_nrm, &G.pipe_nrm, &G.dpool_nrm, &G.dset_nrm, 1)) {
        G.shader_nrm = VK_NULL_HANDLE;
        fprintf(stderr, "FATAL: [VK] failed to build rmsnorm pipeline\n");
        goto init_fail;
    }
    // Build ring buffer for RMSNorm (avoids singleton descriptor aliasing)
    { VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 256*4};
      VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 256, .poolSizeCount = 1, .pPoolSizes = &ps};
      VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.dpool_nrm_ring), "nrm_ring_pool");
      for (int _si = 0; _si < 256; _si++) {
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
          .descriptorPool = G.dpool_nrm_ring, .descriptorSetCount = 1, .pSetLayouts = &G.dsl_unified};
        vkAllocateDescriptorSets(G.dev, &dsa, &G.dset_nrm_ring[_si]);
      }
    }

    // Command pool + buffer
    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = G.qfam};
    VKCHECK(vkCreateCommandPool(G.dev, &cpci, NULL, &G.cpool), "cmdPool");
    VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = G.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi, &G.cmd), "cmdBuf");
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi, &G.cmd_nrm), "cmdBufNrm");

    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT};
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.fence), "fence");
    // Phase 2 fences
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.fence_dev), "fenceDev");
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.fence_xfer), "fenceXfer");

    // Phase 2: device-native command buffer
    VkCommandBufferAllocateInfo cbi2 = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = G.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi2, &G.cmd_dev), "cmdBufDev");
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi, &G.cmd_xfer), "cmdBufXfer");

    // Phase 2: staging buffer (HOST_VISIBLE, growable)
    G.staging_cap = 0;

    // Load elementwise shader (residual_add, silu_mul, rope, f16pack)
    G.shader_elem = load_spv(G.dev, "elementwise_vk.spv");
    if (!G.shader_elem) {
        fprintf(stderr, "FATAL: [VK] missing elementwise_vk.spv -- GPU backend unusable\n");
        goto init_fail;
    }
    if (!build_pipeline_unified(G.dev, G.shader_elem, &G.pipe_elem, &G.dpool_elem, &G.dset_elem, 1)) {
        G.shader_elem = VK_NULL_HANDLE;
        fprintf(stderr, "FATAL: [VK] failed to build elementwise pipeline\n");
        goto init_fail;
    }
    // Build ring buffer for elementwise (avoids singleton descriptor aliasing)
    { VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 256*4};
      VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 256, .poolSizeCount = 1, .pPoolSizes = &ps};
      VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.dpool_elem_ring), "elem_ring_pool");
      for (int _si = 0; _si < 256; _si++) {
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
          .descriptorPool = G.dpool_elem_ring, .descriptorSetCount = 1, .pSetLayouts = &G.dsl_unified};
        VkResult ar = vkAllocateDescriptorSets(G.dev, &dsa, &G.dset_elem_ring[_si]);
        if (ar != VK_SUCCESS) {
          fprintf(stderr, "[VK] elem_ring alloc[%d] failed: %s (%d)\n", _si, vk_result_str(ar), (int)ar);
          break;
        }
      }
    }

    // Load attention decode shader
    G.shader_attn_dec = load_spv(G.dev, "attn_decode_vk.spv");
    if (G.shader_attn_dec) {
        if (!build_pipeline_unified(G.dev, G.shader_attn_dec, &G.pipe_attn_dec, &G.dpool_attn_dec, &G.dset_attn_dec, 1)) {
            G.shader_attn_dec = VK_NULL_HANDLE;
        }
    }

    // Load attention prefill shader
    G.shader_attn_prefill = load_spv(G.dev, "attn_prefill_vk.spv");
    if (!G.shader_attn_prefill) {
        fprintf(stderr, "FATAL: [VK] missing attn_prefill_vk.spv -- GPU backend unusable\n");
        goto init_fail;
    }
    if (!build_pipeline_unified(G.dev, G.shader_attn_prefill, &G.pipe_attn_prefill, &G.dpool_attn_prefill, &G.dset_attn_prefill, 1)) {
        G.shader_attn_prefill = VK_NULL_HANDLE;
        fprintf(stderr, "FATAL: [VK] failed to build attn_prefill pipeline\n");
        goto init_fail;
    }

    // Load Q8_0 quantize shader (F32->Q8_0 on device)
    G.shader_quantize = load_spv(G.dev, "quantize_q8_vk.spv");
    if (G.shader_quantize) {
        if (!build_pipeline_unified(G.dev, G.shader_quantize, &G.pipe_quantize, &G.dpool_quantize, G.dset_quantize_ring, 256)) {
            G.shader_quantize = VK_NULL_HANDLE;
        }
        }

    // Load Q8_0 x Q8_0 matmul shader
    G.shader_q8q8 = load_spv(G.dev, "q8_q8_matmul_vk.spv");
    if (G.shader_q8q8) {
        if (!build_pipeline_unified(G.dev, G.shader_q8q8, &G.pipe_q8q8, &G.dpool_q8q8, G.dset_q8q8_ring, 256)) {
            G.shader_q8q8 = VK_NULL_HANDLE;
        }
        }

    // F16 pack: reuse elementwise shader + pipeline (op=3), no separate pipeline needed
    G.shader_f16pack = G.shader_elem;
    G.pipe_f16pack = G.pipe_elem;
    G.dpool_f16pack = G.dpool_elem;
    G.dset_f16pack = G.dset_elem;

    // KV cache store shader (device-side F32->F16)
    G.shader_kv_store = load_spv(G.dev, "kv_store_vk.spv");
    if (G.shader_kv_store) {
        if (!build_pipeline_unified(G.dev, G.shader_kv_store, &G.pipe_kv_store, &G.dpool_kv_store, G.dset_kv_store_ring, 256)) {
            G.shader_kv_store = VK_NULL_HANDLE;
        }
    }

    G.ready = 1;

    init_fail:
    if (!G.ready) {
        picolm_gpu_shutdown();
        return 0;
    }
    G.device_count = 1;
    G.devices[0] = 0;

    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(G.phys, &p);
    strncpy(G.device_name, p.deviceName, sizeof(G.device_name) - 1);
    G.device_name[sizeof(G.device_name) - 1] = '\0';

    fprintf(stderr, "[VK] ready: %s, compute qfam %u, memtype %u%s, local=%u, sg=%d\n",
            G.device_name, G.qfam, G.memtype,
            G.shader_nrm ? ", rmsnorm" : "",
            G.memtype_local, G.subgroup_size);
    return 1;
}

void picolm_gpu_shutdown(void) {
    if (!G.ready) return;
    vkDeviceWaitIdle(G.dev);

    if (G.x_buf.ptr) vkUnmapMemory(G.dev, G.x_buf.mem);
    if (G.y_buf.ptr) vkUnmapMemory(G.dev, G.y_buf.mem);
    if (G.x_buf.buf) vkDestroyBuffer(G.dev, G.x_buf.buf, NULL);
    if (G.x_buf.mem) vkFreeMemory(G.dev, G.x_buf.mem, NULL);
    if (G.y_buf.buf) vkDestroyBuffer(G.dev, G.y_buf.buf, NULL);
    if (G.y_buf.mem) vkFreeMemory(G.dev, G.y_buf.mem, NULL);

    if (G.dset)  vkFreeDescriptorSets(G.dev, G.dpool, 1, &G.dset);
    if (G.dpool) vkDestroyDescriptorPool(G.dev, G.dpool, NULL);
    if (G.pipe)  vkDestroyPipeline(G.dev, G.pipe, NULL);
    if (G.shader) vkDestroyShaderModule(G.dev, G.shader, NULL);

    if (G.dset_nrm)  vkFreeDescriptorSets(G.dev, G.dpool_nrm, 1, &G.dset_nrm);
    for (int _i = 0; _i < 256 && G.dset_nrm_ring[_i]; _i++)
        vkFreeDescriptorSets(G.dev, G.dpool_nrm_ring, 1, &G.dset_nrm_ring[_i]);
    if (G.dpool_nrm) vkDestroyDescriptorPool(G.dev, G.dpool_nrm, NULL);
    if (G.dpool_nrm_ring) vkDestroyDescriptorPool(G.dev, G.dpool_nrm_ring, NULL);
    if (G.pipe_nrm)  vkDestroyPipeline(G.dev, G.pipe_nrm, NULL);
    if (G.shader_nrm) vkDestroyShaderModule(G.dev, G.shader_nrm, NULL);
    if (G.buf_pc_nrm) vkDestroyBuffer(G.dev, G.buf_pc_nrm, NULL);
    if (G.mem_pc_nrm) vkFreeMemory(G.dev, G.mem_pc_nrm, NULL);

    if (G.dset_elem)  vkFreeDescriptorSets(G.dev, G.dpool_elem, 1, &G.dset_elem);
    for (int _i = 0; _i < 256 && G.dset_elem_ring[_i]; _i++)
        vkFreeDescriptorSets(G.dev, G.dpool_elem_ring, 1, &G.dset_elem_ring[_i]);
    if (G.dpool_elem) vkDestroyDescriptorPool(G.dev, G.dpool_elem, NULL);
    if (G.dpool_elem_ring) vkDestroyDescriptorPool(G.dev, G.dpool_elem_ring, NULL);
    if (G.pipe_elem)  vkDestroyPipeline(G.dev, G.pipe_elem, NULL);
    if (G.shader_elem) vkDestroyShaderModule(G.dev, G.shader_elem, NULL);

    if (G.dset_attn_dec)  vkFreeDescriptorSets(G.dev, G.dpool_attn_dec, 1, &G.dset_attn_dec);
    if (G.dpool_attn_dec) vkDestroyDescriptorPool(G.dev, G.dpool_attn_dec, NULL);
    if (G.pipe_attn_dec)  vkDestroyPipeline(G.dev, G.pipe_attn_dec, NULL);
    if (G.shader_attn_dec) vkDestroyShaderModule(G.dev, G.shader_attn_dec, NULL);

    if (G.dset_attn_prefill)  vkFreeDescriptorSets(G.dev, G.dpool_attn_prefill, 1, &G.dset_attn_prefill);
    if (G.dpool_attn_prefill) vkDestroyDescriptorPool(G.dev, G.dpool_attn_prefill, NULL);
    if (G.pipe_attn_prefill)  vkDestroyPipeline(G.dev, G.pipe_attn_prefill, NULL);
    if (G.shader_attn_prefill) vkDestroyShaderModule(G.dev, G.shader_attn_prefill, NULL);

    if (G.dpool_quantize) {
        // Free all quantize ring descriptor sets (however many were allocated)
        for (int _i = 0; _i < 256 && G.dset_quantize_ring[_i]; _i++)
            vkFreeDescriptorSets(G.dev, G.dpool_quantize, 1, &G.dset_quantize_ring[_i]);
        vkDestroyDescriptorPool(G.dev, G.dpool_quantize, NULL);
    }
    if (G.pipe_quantize)  vkDestroyPipeline(G.dev, G.pipe_quantize, NULL);
    if (G.shader_quantize) vkDestroyShaderModule(G.dev, G.shader_quantize, NULL);

    if (G.dpool_q8q8) {
        for (int _i = 0; _i < 256 && G.dset_q8q8_ring[_i]; _i++)
            vkFreeDescriptorSets(G.dev, G.dpool_q8q8, 1, &G.dset_q8q8_ring[_i]);
        vkDestroyDescriptorPool(G.dev, G.dpool_q8q8, NULL);
    }
    if (G.pipe_q8q8)  vkDestroyPipeline(G.dev, G.pipe_q8q8, NULL);
    if (G.shader_q8q8) vkDestroyShaderModule(G.dev, G.shader_q8q8, NULL);

    // f16pack shares elementwise resources - skip
    // KV store cleanup
    if (G.dpool_kv_store) {
        for (int _i = 0; _i < 256 && G.dset_kv_store_ring[_i]; _i++)
            vkFreeDescriptorSets(G.dev, G.dpool_kv_store, 1, &G.dset_kv_store_ring[_i]);
        vkDestroyDescriptorPool(G.dev, G.dpool_kv_store, NULL);
    }
    if (G.pipe_kv_store) vkDestroyPipeline(G.dev, G.pipe_kv_store, NULL);
    if (G.shader_kv_store) vkDestroyShaderModule(G.dev, G.shader_kv_store, NULL);

    // Destroy unified layout (shared by all pipelines)
    if (G.plyt_unified) vkDestroyPipelineLayout(G.dev, G.plyt_unified, NULL);
    if (G.dsl_unified)  vkDestroyDescriptorSetLayout(G.dev, G.dsl_unified, NULL);

    if (G.cmd)   vkFreeCommandBuffers(G.dev, G.cpool, 1, &G.cmd);
    if (G.cmd_nrm) vkFreeCommandBuffers(G.dev, G.cpool, 1, &G.cmd_nrm);
    if (G.cmd_dev) vkFreeCommandBuffers(G.dev, G.cpool, 1, &G.cmd_dev);
    if (G.cmd_xfer) vkFreeCommandBuffers(G.dev, G.cpool, 1, &G.cmd_xfer);
    if (G.cpool) vkDestroyCommandPool(G.dev, G.cpool, NULL);
    if (G.fence) vkDestroyFence(G.dev, G.fence, NULL);
    if (G.fence_dev) vkDestroyFence(G.dev, G.fence_dev, NULL);
    if (G.fence_xfer) vkDestroyFence(G.dev, G.fence_xfer, NULL);

    // Free staging buffer
    if (G.staging_ptr) vkUnmapMemory(G.dev, G.staging_mem);
    if (G.staging_buf) vkDestroyBuffer(G.dev, G.staging_buf, NULL);
    if (G.staging_mem) vkFreeMemory(G.dev, G.staging_mem, NULL);

    // Free quantize scratch buffers
    if (G.q8_xq_buf) vkDestroyBuffer(G.dev, G.q8_xq_buf, NULL);
    if (G.q8_xq_mem) vkFreeMemory(G.dev, G.q8_xq_mem, NULL);
    if (G.q8_xd_buf) vkDestroyBuffer(G.dev, G.q8_xd_buf, NULL);
    if (G.q8_xd_mem) vkFreeMemory(G.dev, G.q8_xd_mem, NULL);

    // Free pipeline buffers
    #define FREE_PIPE_BUF(b, m) do { if (G.b) { vkDestroyBuffer(G.dev, G.b, NULL); G.b = VK_NULL_HANDLE; } \
        if (G.m) { vkFreeMemory(G.dev, G.m, NULL); G.m = VK_NULL_HANDLE; } } while(0)
    if (G.pipe_ready) {
        FREE_PIPE_BUF(pipe_x, pipe_x_m); FREE_PIPE_BUF(pipe_xb, pipe_xb_m);
        FREE_PIPE_BUF(pipe_q, pipe_q_m); FREE_PIPE_BUF(pipe_k, pipe_k_m);
        FREE_PIPE_BUF(pipe_v, pipe_v_m); FREE_PIPE_BUF(pipe_attn_out, pipe_attn_out_m);
        FREE_PIPE_BUF(pipe_ffn_norm, pipe_ffn_norm_m);
        FREE_PIPE_BUF(pipe_gate, pipe_gate_m); FREE_PIPE_BUF(pipe_up, pipe_up_m);
    }
    if (G.pipe_b_ready) {
        FREE_PIPE_BUF(pipe_x_b, pipe_x_b_m); FREE_PIPE_BUF(pipe_xb_b, pipe_xb_b_m);
        FREE_PIPE_BUF(pipe_q_b, pipe_q_b_m); FREE_PIPE_BUF(pipe_k_b, pipe_k_b_m);
        FREE_PIPE_BUF(pipe_v_b, pipe_v_b_m); FREE_PIPE_BUF(pipe_attn_out_b, pipe_attn_out_b_m);
        FREE_PIPE_BUF(pipe_ffn_norm_b, pipe_ffn_norm_b_m);
        FREE_PIPE_BUF(pipe_gate_b, pipe_gate_b_m); FREE_PIPE_BUF(pipe_up_b, pipe_up_b_m);
        if (G.pipe_logits_d) { free_dev_buf(G.pipe_logits_d); G.pipe_logits_d = NULL; }
        if (G.pipe_logits_buf) { vkDestroyBuffer(G.dev, G.pipe_logits_buf, NULL); vkFreeMemory(G.dev, G.pipe_logits_m, NULL); G.pipe_logits_buf = VK_NULL_HANDLE; }
    }
    #undef FREE_PIPE_BUF

    // Free KV cache
    if (G.kv_k_buf) vkDestroyBuffer(G.dev, G.kv_k_buf, NULL);
    if (G.kv_k_mem) vkFreeMemory(G.dev, G.kv_k_mem, NULL);
    if (G.kv_v_buf) vkDestroyBuffer(G.dev, G.kv_v_buf, NULL);
    if (G.kv_v_mem) vkFreeMemory(G.dev, G.kv_v_mem, NULL);

    VkWArena *a = G.warena;
    while (a) {
        VkWArena *nx = a->next;
        if (a->base) vkUnmapMemory(G.dev, a->mem);
        vkFreeMemory(G.dev, a->mem, NULL);
        free(a);
        a = nx;
    }

    if (G.dev)  vkDestroyDevice(G.dev, NULL);
    if (G.inst) vkDestroyInstance(G.inst, NULL);

    memset(&G, 0, sizeof(G));
}

int picolm_gpu_device_count(void) { return G.ready ? G.device_count : 0; }

int picolm_gpu_device_at(int index) {
    if (index < 0 || index >= G.device_count) return -1;
    return G.devices[index];
}

int picolm_gpu_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    if (!G.ready || device != 0) return 0;
    (void)free_bytes;
    if (total_bytes) {
        size_t total = 0;
        for (uint32_t i = 0; i < G.mem_props.memoryHeapCount; i++) {
            if (G.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                total += (size_t)G.mem_props.memoryHeaps[i].size;
        }
        *total_bytes = total;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// TENSOR UPLOAD
// ---------------------------------------------------------------------------

int picolm_gpu_tensor_upload(void **tensor, const void *weights,
                              gguf_type_t qtype, int I, int O, int device) {
    if (!G.ready || !weights || !tensor || device != 0) return 0;

    picolm_gpu_tensor_t **slot = (picolm_gpu_tensor_t **)tensor;
    if (*slot) {
        picolm_gpu_tensor_t *t = *slot;
        return (t->qtype == qtype && t->I == I && t->O == O);
    }

    size_t rb = gguf_row_bytes(qtype, I);
    if (rb == 0) { fprintf(stderr, "[VK] upload: rb=0 for qtype=%d I=%d\n", qtype, I); return 0; }

    int rw = gguf_row_words(qtype, I);
    size_t gpu_stride = (size_t)rw * 4;
    size_t total = gpu_stride * (size_t)O;
    
    void *wptr;
    VkBuffer wbuf;
    if (!arena_suballoc(total, &wbuf, &wptr)) {
                return 0;
    }
    // For K-quant formats (Q2_K, Q3_K), dequantize to F32 on CPU and upload as F32.
    // The shader doesn't handle K-quant dequant correctly on weak GPUs.
    gguf_type_t upload_qtype = qtype;
    if (qtype == 10 || qtype == 11 || qtype == 14) {
        // Convert to F32: each row is I floats = I*4 bytes
        int f32_rw = (I * 4 + 3) / 4;
        size_t f32_stride = (size_t)f32_rw * 4;
        size_t f32_total = f32_stride * (size_t)O;
        
        // Reallocate if needed (F32 might be larger than quantized)
        if (f32_total > total) {
            vkDestroyBuffer(G.dev, wbuf, NULL);
            if (!arena_suballoc(f32_total, &wbuf, &wptr)) return 0;
            total = f32_total;
        }
        
        // Dequantize each row to F32
        float *f32_buf = malloc(I * sizeof(float));
        if (!f32_buf) { vkDestroyBuffer(G.dev, wbuf, NULL); return 0; }
        for (int o = 0; o < O; o++) {
            dequantize_row((const uint8_t *)weights + (size_t)o * rb, f32_buf, I, qtype);
            memcpy((uint8_t *)wptr + (size_t)o * f32_stride, f32_buf, I * sizeof(float));
        }
        free(f32_buf);
        upload_qtype = 0; // F32
        
        picolm_gpu_tensor_t *t = calloc(1, sizeof(*t));
        if (!t) { vkDestroyBuffer(G.dev, wbuf, NULL); return 0; }
        t->wbuf = wbuf; t->wmem = VK_NULL_HANDLE;
        t->qtype = upload_qtype; t->I = I; t->O = O; t->device = device;
        t->row_bytes = I * 4; t->row_words = f32_rw; t->wbytes = total;
        G.used_bytes += total;
        *slot = t;
        return 1;
    }
    
    for (int o = 0; o < O; o++) {
        memcpy((uint8_t *)wptr + (size_t)o * gpu_stride,
               (const uint8_t *)weights + (size_t)o * rb, rb);
    }

    picolm_gpu_tensor_t *t = calloc(1, sizeof(*t));
    if (!t) { vkDestroyBuffer(G.dev, wbuf, NULL); return 0; }
    t->wbuf = wbuf; t->wmem = VK_NULL_HANDLE;
    t->qtype = qtype; t->I = I; t->O = O; t->device = device;
    t->row_bytes = rb; t->row_words = (size_t)rw; t->wbytes = total;

    G.used_bytes += total;
    *slot = t;
    return 1;
}

void picolm_gpu_tensor_free(picolm_gpu_tensor_t *tensor) {
    if (!tensor) return;
    if (tensor->wbuf) vkDestroyBuffer(G.dev, tensor->wbuf, NULL);
    if (tensor->wmem) vkFreeMemory(G.dev, tensor->wmem, NULL);
    G.used_bytes -= tensor->wbytes;
    free(tensor);
}

size_t picolm_gpu_tensor_bytes(const picolm_gpu_tensor_t *tensor) {
    return tensor ? tensor->wbytes : 0;
}

int picolm_gpu_tensor_device(const picolm_gpu_tensor_t *tensor) {
    return tensor ? tensor->device : -1;
}

const void *picolm_gpu_tensor_weights(const picolm_gpu_tensor_t *tensor) {
    (void)tensor; return NULL;
}

// ---------------------------------------------------------------------------
// MATMUL: host-facing (H2D, D2H, sync)
// ---------------------------------------------------------------------------

// Push constant struct for qmatmul shader:
//   int fmt, int S, int I, int O, int rowWords, int x_stride, int y_stride
// Must match shader push_constant layout (28 bytes, within 48-byte PC range)
typedef struct {
    int fmt;       // GGUF_TYPE enum value
    int S, I, O;
    int rowWords;  // uint32 words per weight row
    int x_stride;  // activation stride (0 = I)
    int y_stride;  // output stride (0 = O)
} PC_Matmul;

int picolm_gpu_matmul(picolm_gpu_tensor_t *t, float *y, const float *x,
                       int S, int device) {
    if (!G.ready || !t || device != 0 || S < 1) return 0;

    size_t xb = (size_t)S * t->I * sizeof(float);
    size_t yb = (size_t)S * t->O * sizeof(float);

    if (!scratch_reserve(&G.x_buf, xb)) { fprintf(stderr, "[VK] matmul: x scratch failed %zu\n", xb); return 0; }
    if (!scratch_reserve_mt(&G.y_buf, yb, G.memtype)) { fprintf(stderr, "[VK] matmul: y scratch failed %zu\n", yb); return 0; }

    memcpy(G.x_buf.ptr, x, xb);

    int rebind = G.bound_tensor != t || G.x_buf.buf != G.bound_xbuf ||
                 G.y_buf.buf != G.bound_ybuf;
    if (rebind) {
        VkDescriptorBufferInfo bi[3] = {
            {G.x_buf.buf, 0, VK_WHOLE_SIZE},
            {t->wbuf, 0, VK_WHOLE_SIZE},
            {G.y_buf.buf, 0, VK_WHOLE_SIZE}
        };
        wr_desc_ring(3, bi);
        G.bound_tensor = t; G.bound_xbuf = G.x_buf.buf; G.bound_ybuf = G.y_buf.buf;
    }

    // Always re-record command buffer: host-visible scratch buffer contents change
    // every call, and some drivers (RADV on older AMD) don't properly see updated
    // HOST_COHERENT memory on command buffer replay without explicit barriers.
    {
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
        VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
        VkDescriptorSet cur_dset = G.dset[(G.dset_idx - 1) % 256];
        vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                G.plyt_unified, 0, 1, &cur_dset, 0, NULL);
        PC_Matmul pc = {t->qtype, S, t->I, t->O, (int)t->row_words};
        vkCmdPushConstants(G.cmd, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        // One workgroup per output row per sequence
        vkCmdDispatch(G.cmd, (uint32_t)((t->O + 15) / 16), (uint32_t)S, 1);
        VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");
        G.cmd_ready = 1; G.bound_S = S; G.bound_I = t->I; G.bound_O = t->O;
    }

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    VkResult wr = vk_fence_wait(G.dev, G.fence);
    if (wr != VK_SUCCESS) {
        fprintf(stderr, "[VK] fence wait failed: %d -- disabling GPU offload\n", (int)wr);
        G.ready = 0;
        return 0;
    }
    memcpy(y, G.y_buf.ptr, yb);
    return 1;
}

// ---------------------------------------------------------------------------
// Device buffer wrapper (used by host-facing functions above)
// ---------------------------------------------------------------------------

typedef struct vk_dev_buf {
    VkBuffer buf;
    VkDeviceMemory mem;
    size_t off, sz;
    int idx;
    struct vk_dev_buf *next;  /* linked list for offset lookup */
} vk_dev_buf_t;

static vk_dev_buf_t *g_dev_buf_list = NULL;
static int g_dev_buf_count = 1; /* Start at 1 so idx==0 doesn't produce NULL pointer */

static VkDescriptorBufferInfo desc_buf_info(const void *p);

// ---------------------------------------------------------------------------
// RMSNorm: host-facing
// ---------------------------------------------------------------------------

// PC_RmsNorm replaced by inline struct in dispatch functions

int picolm_gpu_rmsnorm(float *out, const float *x, const float *weight,
                        int dim, float eps, int device) {
    return picolm_gpu_rmsnorm_batched(out, x, weight, dim, eps, 1, dim, device);
}

int picolm_gpu_rmsnorm_batched(float *out, const float *x, const float *weight,
                                int dim, float eps, int S, int x_stride, int device) {
    if (!G.ready || device != 0 || S < 1) return 0;
    /* Detect device pointers: if unwrap_buf_offset finds a known buffer,
     * this is a device-native call (from model_forward_gpu). Dispatch to _dev. */
    VkDeviceSize off;
    if (unwrap_buf_offset(x, &off) || unwrap_buf_offset(weight, &off)) {
        return picolm_gpu_rmsnorm_batched_dev(out, x, weight, dim, eps, S, x_stride, device);
    }
    /* Host pointer path: allocate temp buffers, copy, dispatch */
    if (!G.shader_nrm) return 0;
    size_t xb = (size_t)S * x_stride * sizeof(float);
    size_t wb = (size_t)dim * sizeof(float);
    size_t yb = (size_t)S * x_stride * sizeof(float);
    VkBuffer xbuf, wbuf, ybuf;
    VkDeviceMemory xmem, wmem, ymem;
    void *xptr = NULL, *wptr = NULL, *yptr = NULL;
    if (!alloc_hostvis(xb, &xbuf, &xmem, &xptr)) return 0;
    if (!alloc_hostvis(wb, &wbuf, &wmem, &wptr)) return 0;
    if (!alloc_hostvis_mt(yb, &ybuf, &ymem, &yptr, G.memtype_cached)) return 0;
    memcpy(xptr, x, xb);
    memcpy(wptr, weight, wb);
    VkDescriptorBufferInfo bi[3] = {
        {xbuf, 0, VK_WHOLE_SIZE}, {wbuf, 0, VK_WHOLE_SIZE}, {ybuf, 0, VK_WHOLE_SIZE}
    };
    wr_desc(G.dset_nrm, 3, bi);

    VKCHECK(vkResetCommandBuffer(G.cmd_nrm, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd_nrm, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd_nrm, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_nrm);
    vkCmdBindDescriptorSets(G.cmd_nrm, VK_PIPELINE_BIND_POINT_COMPUTE,
                            G.plyt_unified, 0, 1, &G.dset_nrm, 0, NULL);
    PC_Matmul pc_nrm = {0, S, dim, 0, x_stride};
    memcpy(&pc_nrm.O, &eps, sizeof(float)); // bitcast eps into O field
    vkCmdPushConstants(G.cmd_nrm, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc_nrm), &pc_nrm);
    vkCmdDispatch(G.cmd_nrm, (uint32_t)S, 1, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd_nrm), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd_nrm};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) {
        vkDestroyBuffer(G.dev, xbuf, NULL); vkFreeMemory(G.dev, xmem, NULL);
        vkDestroyBuffer(G.dev, wbuf, NULL); vkFreeMemory(G.dev, wmem, NULL);
        vkDestroyBuffer(G.dev, ybuf, NULL); vkFreeMemory(G.dev, ymem, NULL);
        return 0;
    }
    memcpy(out, yptr, yb);

    vkUnmapMemory(G.dev, xmem);
    vkUnmapMemory(G.dev, wmem);
    vkUnmapMemory(G.dev, ymem);
    vkDestroyBuffer(G.dev, xbuf, NULL); vkFreeMemory(G.dev, xmem, NULL);
    vkDestroyBuffer(G.dev, wbuf, NULL); vkFreeMemory(G.dev, wmem, NULL);
    vkDestroyBuffer(G.dev, ybuf, NULL); vkFreeMemory(G.dev, ymem, NULL);
    return 1;
}

// ---------------------------------------------------------------------------
// UPLOAD F32 (for norm weights, rope tables, etc.)
// ---------------------------------------------------------------------------

float *picolm_gpu_upload_f32(const float *host, size_t n, int device) {
    if (!G.ready || device != 0 || n == 0) return NULL;
    size_t bytes = n * sizeof(float);
    VkBuffer wbuf; void *wptr;
    if (!arena_suballoc(bytes, &wbuf, &wptr)) {
        // Fallback: device-local + staging copy
        void *dst = picolm_gpu_alloc_device(bytes, device);
        if (!dst) return NULL;
        if (!picolm_gpu_memcpy_async(dst, host, bytes, 1, device)) {
            vk_dev_buf_t *d = (vk_dev_buf_t *)dst;
            vkDestroyBuffer(G.dev, d->buf, NULL);
            vkFreeMemory(G.dev, d->mem, NULL);
            free(d);
            return NULL;
        }
        picolm_gpu_sync(device);
        return (float *)dst;
    }
    // Direct write to host-visible memory
    memcpy(wptr, host, bytes);
    /* Wrap as device pointer */
    vk_dev_buf_t *d = malloc(sizeof(*d));
    if (!d) { vkDestroyBuffer(G.dev, wbuf, NULL); return NULL; }
    d->buf = wbuf; d->mem = VK_NULL_HANDLE; d->off = 0; d->sz = bytes;
    d->next = g_dev_buf_list; d->idx = g_dev_buf_count;
    g_dev_buf_list = d; g_dev_buf_count++;
    void *dst = (void*)((uintptr_t)d->idx << 48);
    return (float *)dst;
}

// ---------------------------------------------------------------------------
// DEVICE MEMORY
// ---------------------------------------------------------------------------

/* Synthetic device pointers: encode buffer index in bits 48-63,
 * byte offset in bits 0-47. On 64-bit systems, user-space addresses
 * use only 48 bits, so bits 48-63 are always zero for real pointers.
 * This makes our synthetic pointers distinguishable and immune to
 * pointer arithmetic corruption. */

#define DEV_PTR_MASK ((1ULL << 48) - 1)

static void *wrap_buf(VkBuffer b, VkDeviceMemory m, size_t off, size_t sz) {
    vk_dev_buf_t *d = malloc(sizeof(*d));
    if (!d) return NULL;
    d->buf = b; d->mem = m; d->off = off; d->sz = sz;
    d->next = g_dev_buf_list;
    d->idx = g_dev_buf_count;
    g_dev_buf_list = d;
    g_dev_buf_count++;
    return (void*)((uintptr_t)d->idx << 48);
}

/* Remove a vk_dev_buf_t from the global list and free it.
 * Used when a temporary device buffer is no longer needed. */
static void free_dev_buf(void *p) {
    if (!p) return;
    int idx = (int)((uintptr_t)p >> 48);
    vk_dev_buf_t **pp = &g_dev_buf_list;
    while (*pp) {
        if ((*pp)->idx == idx) {
            vk_dev_buf_t *d = *pp;
            *pp = d->next;
            g_dev_buf_count--;
            free(d);
            return;
        }
        pp = &(*pp)->next;
    }
}

/* Unwrap a device pointer to get (VkBuffer, offset).
 * Handles offset pointers from CUDA-style pointer arithmetic:
 * (float*)base + N produces an address = base + N*4 bytes.
 * We find the base vk_dev_buf_t by searching our list for an entry
 * whose address is <= the given pointer, then compute the byte offset. */
static VkBuffer unwrap_buf_offset(const void *p, VkDeviceSize *out_off) {
    uintptr_t addr = (uintptr_t)p;
    if (!addr) { *out_off = 0; return VK_NULL_HANDLE; }
    int idx = (int)(addr >> 48);
    VkDeviceSize byte_offset = addr & DEV_PTR_MASK;
    for (vk_dev_buf_t *d = g_dev_buf_list; d; d = d->next) {
        if (d->idx == idx) {
            *out_off = d->off + byte_offset;
            return d->buf;
        }
    }
    *out_off = 0;
    return VK_NULL_HANDLE;
}

static VkBuffer unwrap_buf(const void *p) {
    VkDeviceSize off;
    return unwrap_buf_offset(p, &off);
}

/* Create a VkDescriptorBufferInfo with proper offset handling */
static VkDescriptorBufferInfo desc_buf_info(const void *p) {
    VkDeviceSize off = 0;
    VkBuffer b = unwrap_buf_offset(p, &off);
    return (VkDescriptorBufferInfo){b, off, VK_WHOLE_SIZE};
}

void *picolm_gpu_alloc_device(size_t bytes, int device) {
    if (!G.ready || device != 0 || bytes < 1) return NULL;
    VkBuffer buf; VkDeviceMemory mem;
    if (!alloc_device_local(bytes, &buf, &mem)) return NULL;
    return wrap_buf(buf, mem, 0, bytes);
}

int picolm_gpu_device_memset(void *dev_ptr, int value, size_t bytes, int device) {
    if (!G.ready || !dev_ptr || bytes < 1 || device != 0) return 0;
    VkBuffer buf = unwrap_buf(dev_ptr);
    if (!buf) return 0;
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkResetCommandBuffer(G.cmd_xfer, 0);
    vkBeginCommandBuffer(G.cmd_xfer, &begin);
    vkCmdFillBuffer(G.cmd_xfer, buf, 0, VK_WHOLE_SIZE, (uint32_t)value);
    vkEndCommandBuffer(G.cmd_xfer);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
    vkResetFences(G.dev, 1, &G.fence_xfer);
    vkQueueSubmit(G.queue, 1, &si, G.fence_xfer);
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
    return 1;
}

void *picolm_gpu_upload_int(const int *host, size_t n, int device) {
    if (!G.ready || !host || n == 0 || device != 0) return NULL;
    size_t bytes = n * sizeof(int);
    void *dst = picolm_gpu_alloc_device(bytes, device);
    if (!dst) return NULL;
    if (!picolm_gpu_memcpy_async(dst, host, bytes, 1, device)) {
        vk_dev_buf_t *d = (vk_dev_buf_t *)dst;
        vkDestroyBuffer(G.dev, d->buf, NULL);
        vkFreeMemory(G.dev, d->mem, NULL);
        free(d);
        return NULL;
    }
    picolm_gpu_sync(device);
    return dst;
}

// ---------------------------------------------------------------------------
// STAGING BUFFER + MEMCPY
// ---------------------------------------------------------------------------

static int staging_ensure(size_t bytes) {
    if (G.staging_cap >= bytes) return 1;
    if (G.staging_ptr) vkUnmapMemory(G.dev, G.staging_mem);
    if (G.staging_buf) vkDestroyBuffer(G.dev, G.staging_buf, NULL);
    if (G.staging_mem) vkFreeMemory(G.dev, G.staging_mem, NULL);
    G.staging_buf = VK_NULL_HANDLE; G.staging_mem = VK_NULL_HANDLE;
    G.staging_ptr = NULL; G.staging_cap = 0;

    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(G.dev, &bi, NULL, &G.staging_buf) != VK_SUCCESS) return 0;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, G.staging_buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = G.memtype_staging};
    if (vkAllocateMemory(G.dev, &ai, NULL, &G.staging_mem) != VK_SUCCESS) {
        vkDestroyBuffer(G.dev, G.staging_buf, NULL); return 0;
    }
    vkBindBufferMemory(G.dev, G.staging_buf, G.staging_mem, 0);
    vkMapMemory(G.dev, G.staging_mem, 0, VK_WHOLE_SIZE, 0, &G.staging_ptr);
    G.staging_cap = bytes;
    return 1;
}

int picolm_gpu_memcpy(void *dst, const void *src, size_t bytes, int dir, int device) {
    if (!G.ready || !dst || !src || bytes < 1 || device != 0) return 0;
    if (!picolm_gpu_memcpy_async(dst, src, bytes, dir, device)) return 0;
    return picolm_gpu_sync(device);
}

int picolm_gpu_memcpy_async(void *dst, const void *src, size_t bytes, int dir, int device) {
    if (!G.ready || !dst || !src || bytes < 1 || device != 0) return 0;
    if (dir == 1) { /* H2D via staging */
        if (!staging_ensure(bytes)) return 0;
        memcpy(G.staging_ptr, src, bytes);
        /* Flush CPU write to ensure GPU sees the data */
        VkMappedMemoryRange flush = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = G.staging_mem, .offset = 0, .size = VK_WHOLE_SIZE};
        vkFlushMappedMemoryRanges(G.dev, 1, &flush);
        VkDeviceSize dst_off = 0;
        VkBuffer dst_buf = unwrap_buf_offset(dst, &dst_off);
        if (!dst_buf) return 0;
        /* Ensure prior xfer is done before reusing cmd_xfer */
        vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkResetCommandBuffer(G.cmd_xfer, 0);
        vkBeginCommandBuffer(G.cmd_xfer, &begin);
        VkBufferCopy bc = {0, dst_off, bytes};
        vkCmdCopyBuffer(G.cmd_xfer, G.staging_buf, dst_buf, 1, &bc);
        vkEndCommandBuffer(G.cmd_xfer);
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
        vkResetFences(G.dev, 1, &G.fence_xfer);
        vkQueueSubmit(G.queue, 1, &si, G.fence_xfer);
        return 1;
    }
    if (dir == -1) { /* D2H via staging */
        if (!staging_ensure(bytes)) return 0;
        VkDeviceSize src_off = 0;
        VkBuffer src_buf = unwrap_buf_offset(src, &src_off);
        if (!src_buf) return 0;
        /* If in a batch, flush it first so GPU has executed the dispatches */
        if (G.in_batch) picolm_gpu_batch_end(0);
        /* Now wait on both fences: fence_dev for compute completion, fence_xfer for xfer completion */
        vk_fence_wait_timeout(G.dev, G.fence_dev, 10ULL*1000*1000*1000);
        vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkResetCommandBuffer(G.cmd_xfer, 0);
        vkBeginCommandBuffer(G.cmd_xfer, &begin);
        VkBufferCopy bc = {src_off, 0, bytes};
        vkCmdCopyBuffer(G.cmd_xfer, src_buf, G.staging_buf, 1, &bc);
        vkEndCommandBuffer(G.cmd_xfer);
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
        vkResetFences(G.dev, 1, &G.fence_xfer);
        vkQueueSubmit(G.queue, 1, &si, G.fence_xfer);
        vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
        memcpy(dst, G.staging_ptr, bytes);
        return 1;
    }
    return 0;
}

float *picolm_gpu_staging_host(int device, size_t bytes) {
    if (!G.ready || device != 0) return NULL;
    if (!staging_ensure(bytes)) return NULL;
    return (float *)G.staging_ptr;
}

int picolm_gpu_sync(int device) {
    if (!G.ready || device != 0) return 0;
    if (G.in_batch) picolm_gpu_batch_end(device);
    vk_fence_wait_timeout(G.dev, G.fence_dev, 5ULL*1000*1000*1000);
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 5ULL*1000*1000*1000);
    return 1;
}

// ---------------------------------------------------------------------------
// PHASE 1: SINGLE COMMAND BUFFER (batch begin/end)
// ---------------------------------------------------------------------------
// All dispatch functions within a batch record into a single command buffer.
// One submit + one fence wait at the end. Eliminates ~470 CPU<->GPU round-trips.

// Helper: if NOT in batch, do fence_wait + reset + begin on cmd_dev
#define VK_BATCH_DISPATCH_PRE() do { \
    if (!G.in_batch) { \
        vk_fence_wait_timeout(G.dev, G.fence_dev, 10ULL*1000*1000*1000); \
        VkCommandBufferBeginInfo _vb = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; \
        vkResetCommandBuffer(G.cmd_dev, 0); \
        vkBeginCommandBuffer(G.cmd_dev, &_vb); \
    } \
} while(0)

// Helper: if NOT in batch, do end + submit + fence_wait on cmd_dev
#define VK_BATCH_DISPATCH_POST() do { \
    if (!G.in_batch) { \
        vkEndCommandBuffer(G.cmd_dev); \
        VkSubmitInfo _vsi = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, \
            .commandBufferCount = 1, .pCommandBuffers = &G.cmd_dev}; \
        vkResetFences(G.dev, 1, &G.fence_dev); \
        vkQueueSubmit(G.queue, 1, &_vsi, G.fence_dev); \
        vk_fence_wait_timeout(G.dev, G.fence_dev, 10ULL*1000*1000*1000); \
    } \
} while(0)

// Helper: barrier between dispatches (only needed if NOT in batch, or always for safety)
// In a batch, Vulkan guarantees in-order compute shader execution on same queue.
// But we still need a memory barrier for shader_write->shader_read between
// different dispatches that share buffers.
#define VK_BATCH_BARRIER() do { \
    VkMemoryBarrier _mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, \
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, \
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT}; \
    vkCmdPipelineBarrier(G.cmd_dev, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, \
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &_mb, 0, NULL, 0, NULL); \
} while(0)

int picolm_gpu_batch_begin(int device) {
    if (!G.ready || device != 0) return 0;
    if (G.in_batch) return 0;  // already in a batch
    // Wait for any prior dev/xfer work to complete before starting
    vk_fence_wait_timeout(G.dev, G.fence_dev, 10ULL*1000*1000*1000);
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkResetCommandBuffer(G.cmd_dev, 0);
    vkBeginCommandBuffer(G.cmd_dev, &begin);
    G.in_batch = 1;
    return 1;
}

int picolm_gpu_batch_end(int device) {
    if (!G.ready || device != 0 || !G.in_batch) return 0;
    VkResult r = vkEndCommandBuffer(G.cmd_dev);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[VK] batch_end: vkEndCommandBuffer failed: %d\n", (int)r);
        G.in_batch = 0;
        return 0;
    }
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd_dev};
    vkResetFences(G.dev, 1, &G.fence_dev);
    vkQueueSubmit(G.queue, 1, &si, G.fence_dev);
    vk_fence_wait_timeout(G.dev, G.fence_dev, 10ULL*1000*1000*1000);
    G.in_batch = 0;
    return 1;
}
// ---------------------------------------------------------------------------
// PIPELINE ALLOCATION
// ---------------------------------------------------------------------------

int picolm_gpu_pipeline_alloc(int dim, int q_dim, int kv_dim,
                               int ffn_hidden, int vocab_size, int device) {
    if (!G.ready || device != 0) return 0;
    if (G.pipe_ready) return 1;
    size_t db = (size_t)dim * sizeof(float);
    size_t qb = (size_t)q_dim * sizeof(float);
    size_t kvb = (size_t)kv_dim * sizeof(float);
    size_t fb = (size_t)ffn_hidden * sizeof(float);
    int ok = 1;
    size_t total = 2*db + 2*qb + 2*kvb + 2*fb;
    fprintf(stderr, "[VK] pipeline_alloc: dim=%d q=%d kv=%d ffn=%d total=%zu KB\n",
            dim, q_dim, kv_dim, ffn_hidden, total/1024);
    #define AP(name, sz) do { if (!alloc_device_local((sz), &G.pipe_##name, &G.pipe_##name##_m)) ok = 0; } while(0)
    AP(x, db); AP(xb, db); AP(q, qb); AP(k, kvb); AP(v, kvb);
    AP(attn_out, qb); AP(ffn_norm, db); AP(gate, fb); AP(up, fb);
    #undef AP
    if (!ok) { fprintf(stderr, "[VK] pipeline_alloc: FAILED\n"); picolm_gpu_pipeline_free(); return 0; }
    G.pipe_ready = 1;
    G.pipe_dim = dim; G.pipe_q_dim = q_dim;
    G.pipe_kv_dim = kv_dim; G.pipe_ffn_hidden = ffn_hidden;
    /* Create cached vk_dev_buf_t* wrappers for all pipe accessors */
    G.pipe_x_d = wrap_buf(G.pipe_x, G.pipe_x_m, 0, db);
    G.pipe_xb_d = wrap_buf(G.pipe_xb, G.pipe_xb_m, 0, db);
    G.pipe_q_d = wrap_buf(G.pipe_q, G.pipe_q_m, 0, qb);
    G.pipe_k_d = wrap_buf(G.pipe_k, G.pipe_k_m, 0, kvb);
    G.pipe_v_d = wrap_buf(G.pipe_v, G.pipe_v_m, 0, kvb);
    G.pipe_attn_out_d = wrap_buf(G.pipe_attn_out, G.pipe_attn_out_m, 0, qb);
    G.pipe_ffn_norm_d = wrap_buf(G.pipe_ffn_norm, G.pipe_ffn_norm_m, 0, db);
    G.pipe_gate_d = wrap_buf(G.pipe_gate, G.pipe_gate_m, 0, fb);
    G.pipe_up_d = wrap_buf(G.pipe_up, G.pipe_up_m, 0, fb);
    fprintf(stderr, "[VK] pipeline_alloc: OK\n");
    return 1;
}

int picolm_gpu_pipeline_batch_alloc(int dim, int q_dim, int kv_dim,
                                     int ffn_hidden, int xb_stride,
                                     int max_seq_len, int device) {
    if (!G.ready || device != 0) return 0;
    if (G.pipe_b_ready) return 1;
    size_t bsz = (size_t)max_seq_len;
    (void)dim;
    size_t qb = bsz * q_dim * sizeof(float);
    size_t xb = bsz * xb_stride * sizeof(float);
    size_t kvb = bsz * kv_dim * sizeof(float);
    size_t fb = bsz * ffn_hidden * sizeof(float);
    size_t total = 2*xb + 2*qb + 2*kvb + 2*fb;
    fprintf(stderr, "[VK] pipeline_batch_alloc: seq=%d stride=%d total=%zu MB\n",
            max_seq_len, xb_stride, total/(1024*1024));
    int ok = 1;
    #define APB(name, sz) do { if (!alloc_device_local((sz), &G.pipe_##name##_b, &G.pipe_##name##_b_m)) ok = 0; } while(0)
    APB(x, xb); APB(xb, xb); APB(q, qb); APB(k, kvb); APB(v, kvb);
    APB(attn_out, qb); APB(ffn_norm, xb); APB(gate, fb); APB(up, fb);
    /* Logits buffer: vocab_size floats, allocated per-model in pipeline_alloc */
    #undef APB
    if (!ok) { fprintf(stderr, "[VK] pipeline_batch_alloc: FAILED\n"); return 0; }
    G.pipe_b_ready = 1;
    G.pipe_b_max_seq_len = max_seq_len;
    G.pipe_b_xb_stride = xb_stride;
    /* Cached wrappers for batch pipes */
    G.pipe_x_b_d = wrap_buf(G.pipe_x_b, G.pipe_x_b_m, 0, xb);
    G.pipe_xb_b_d = wrap_buf(G.pipe_xb_b, G.pipe_xb_b_m, 0, xb);
    G.pipe_q_b_d = wrap_buf(G.pipe_q_b, G.pipe_q_b_m, 0, qb);
    G.pipe_k_b_d = wrap_buf(G.pipe_k_b, G.pipe_k_b_m, 0, kvb);
    G.pipe_v_b_d = wrap_buf(G.pipe_v_b, G.pipe_v_b_m, 0, kvb);
    // Map pipe buffers for direct CPU readback (works on llvmpipe/RADV where memtype is host-visible)
    G.pipe_k_b_mapped = NULL; G.pipe_v_b_mapped = NULL;
    VkResult mr;
    mr = vkMapMemory(G.dev, G.pipe_k_b_m, 0, kvb, 0, &G.pipe_k_b_mapped);
    if (mr != VK_SUCCESS) G.pipe_k_b_mapped = NULL;
    mr = vkMapMemory(G.dev, G.pipe_v_b_m, 0, kvb, 0, &G.pipe_v_b_mapped);
    if (mr != VK_SUCCESS) G.pipe_v_b_mapped = NULL;
    mr = vkMapMemory(G.dev, G.pipe_x_b_m, 0, xb, 0, &G.pipe_x_b_mapped);
    if (mr != VK_SUCCESS) { G.pipe_x_b_mapped = NULL; }
    G.pipe_attn_out_b_d = wrap_buf(G.pipe_attn_out_b, G.pipe_attn_out_b_m, 0, qb);
    G.pipe_ffn_norm_b_d = wrap_buf(G.pipe_ffn_norm_b, G.pipe_ffn_norm_b_m, 0, xb);
    G.pipe_gate_b_d = wrap_buf(G.pipe_gate_b, G.pipe_gate_b_m, 0, fb);
    G.pipe_up_b_d = wrap_buf(G.pipe_up_b, G.pipe_up_b_m, 0, fb);
    G.pipe_logits_d = NULL; /* allocated lazily */
    fprintf(stderr, "[VK] pipeline_batch_alloc: OK\n");
    return 1;
}

void picolm_gpu_pipeline_free(void) {
    if (!G.ready) return;
    /* Free cached wrappers (don't remove from list to keep offset lookup valid) */
    if (G.pipe_ready) {
        free_dev_buf(G.pipe_x_d); free_dev_buf(G.pipe_xb_d); free_dev_buf(G.pipe_q_d); free_dev_buf(G.pipe_k_d); free_dev_buf(G.pipe_v_d);
        free_dev_buf(G.pipe_attn_out_d); free_dev_buf(G.pipe_ffn_norm_d); free_dev_buf(G.pipe_gate_d); free_dev_buf(G.pipe_up_d);
        G.pipe_x_d = G.pipe_xb_d = G.pipe_q_d = G.pipe_k_d = G.pipe_v_d = NULL;
        G.pipe_attn_out_d = G.pipe_ffn_norm_d = G.pipe_gate_d = G.pipe_up_d = NULL;
    }
    if (G.pipe_b_ready) {
        free_dev_buf(G.pipe_x_b_d); free_dev_buf(G.pipe_xb_b_d); free_dev_buf(G.pipe_q_b_d); free_dev_buf(G.pipe_k_b_d); free_dev_buf(G.pipe_v_b_d);
        free_dev_buf(G.pipe_attn_out_b_d); free_dev_buf(G.pipe_ffn_norm_b_d); free_dev_buf(G.pipe_gate_b_d); free_dev_buf(G.pipe_up_b_d);
        G.pipe_x_b_d = G.pipe_xb_b_d = G.pipe_q_b_d = G.pipe_k_b_d = G.pipe_v_b_d = NULL;
        G.pipe_attn_out_b_d = G.pipe_ffn_norm_b_d = G.pipe_gate_b_d = G.pipe_up_b_d = NULL;
        if (G.pipe_logits_d) { free_dev_buf(G.pipe_logits_d); G.pipe_logits_d = NULL; }
        if (G.pipe_logits_buf) { vkDestroyBuffer(G.dev, G.pipe_logits_buf, NULL); vkFreeMemory(G.dev, G.pipe_logits_m, NULL); G.pipe_logits_buf = VK_NULL_HANDLE; }
    }
    #define FB(n) do { if (G.pipe_##n) { vkDestroyBuffer(G.dev, G.pipe_##n, NULL); G.pipe_##n = VK_NULL_HANDLE; } \
        if (G.pipe_##n##_m) { vkFreeMemory(G.dev, G.pipe_##n##_m, NULL); G.pipe_##n##_m = VK_NULL_HANDLE; } } while(0)
    #define FB2(n) do { if (G.pipe_##n##_b) { vkDestroyBuffer(G.dev, G.pipe_##n##_b, NULL); G.pipe_##n##_b = VK_NULL_HANDLE; } \
        if (G.pipe_##n##_b_m) { vkFreeMemory(G.dev, G.pipe_##n##_b_m, NULL); G.pipe_##n##_b_m = VK_NULL_HANDLE; } } while(0)
    if (G.pipe_ready) { FB(x); FB(xb); FB(q); FB(k); FB(v); FB(attn_out); FB(ffn_norm); FB(gate); FB(up); }
    if (G.pipe_b_ready) { FB2(x); FB2(xb); FB2(q); FB2(k); FB2(v); FB2(attn_out); FB2(ffn_norm); FB2(gate); FB2(up); }
    G.pipe_ready = 0; G.pipe_b_ready = 0;
    #undef FB
    #undef FB2
}

int picolm_gpu_ssm_pipeline_alloc(int cd, int ssi, int nv, int device) {
    (void)cd; (void)ssi; (void)nv; (void)device; return 0;
}

int picolm_gpu_pipeline_logits_alloc(size_t bytes, int device) {
    if (!G.ready || device != 0 || bytes < 1) return 0;
    if (G.pipe_logits_d) return 1; /* already allocated */
    VkBuffer buf; VkDeviceMemory mem;
    if (!alloc_device_local(bytes, &buf, &mem)) return 0;
    G.pipe_logits_buf = buf; G.pipe_logits_m = mem;
    G.pipe_logits_d = wrap_buf(buf, mem, 0, bytes);
    return G.pipe_logits_d ? 1 : 0;
}

int picolm_gpu_prealloc_q8(size_t mxq, size_t mxd, int device) {
    if (!G.ready || device != 0) return 0;
    if (mxq > G.q8_xq_cap) {
        if (G.q8_xq_buf) { vkDestroyBuffer(G.dev, G.q8_xq_buf, NULL); G.q8_xq_buf = VK_NULL_HANDLE; }
        if (G.q8_xq_mem) { vkFreeMemory(G.dev, G.q8_xq_mem, NULL); G.q8_xq_mem = VK_NULL_HANDLE; }
        if (G.q8_xq_d) { free_dev_buf(G.q8_xq_d); G.q8_xq_d = NULL; }
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = mxq, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        if (vkCreateBuffer(G.dev, &bi, NULL, &G.q8_xq_buf) != VK_SUCCESS) return 0;
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(G.dev, G.q8_xq_buf, &req);
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = req.size, .memoryTypeIndex = G.memtype_local};
        if (vkAllocateMemory(G.dev, &ai, NULL, &G.q8_xq_mem) != VK_SUCCESS) return 0;
        vkBindBufferMemory(G.dev, G.q8_xq_buf, G.q8_xq_mem, 0);
        G.q8_xq_cap = mxq;
        G.q8_xq_d = wrap_buf(G.q8_xq_buf, G.q8_xq_mem, 0, mxq);
    }
    if (mxd > G.q8_xd_cap) {
        if (G.q8_xd_buf) { vkDestroyBuffer(G.dev, G.q8_xd_buf, NULL); G.q8_xd_buf = VK_NULL_HANDLE; }
        if (G.q8_xd_mem) { vkFreeMemory(G.dev, G.q8_xd_mem, NULL); G.q8_xd_mem = VK_NULL_HANDLE; }
        if (G.q8_xd_d) { free_dev_buf(G.q8_xd_d); G.q8_xd_d = NULL; }
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = mxd, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        if (vkCreateBuffer(G.dev, &bi, NULL, &G.q8_xd_buf) != VK_SUCCESS) return 0;
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(G.dev, G.q8_xd_buf, &req);
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = req.size, .memoryTypeIndex = G.memtype_local};
        if (vkAllocateMemory(G.dev, &ai, NULL, &G.q8_xd_mem) != VK_SUCCESS) return 0;
        vkBindBufferMemory(G.dev, G.q8_xd_buf, G.q8_xd_mem, 0);
        G.q8_xd_cap = mxd;
        G.q8_xd_d = wrap_buf(G.q8_xd_buf, G.q8_xd_mem, 0, mxd);
    }
    return 1;
}
// ---------------------------------------------------------------------------
// PIPELINE BUFFER ACCESSORS
// ---------------------------------------------------------------------------

float *picolm_gpu_pipe_x(int d) { return (d==0&&G.pipe_ready)?(float*)G.pipe_x_d:NULL; }
float *picolm_gpu_pipe_xb(int d) { return (d==0&&G.pipe_ready)?(float*)G.pipe_xb_d:NULL; }
float *picolm_gpu_pipe_q(int d) { return (d==0&&G.pipe_ready)?(float*)G.pipe_q_d:NULL; }
float *picolm_gpu_pipe_k(int d) { return (d==0&&G.pipe_ready)?(float*)G.pipe_k_d:NULL; }
float *picolm_gpu_pipe_v(int d) { return (d==0&&G.pipe_ready)?(float*)G.pipe_v_d:NULL; }
float *picolm_gpu_pipe_attn_out(int d) { return (d==0&&G.pipe_ready)?(float*)G.pipe_attn_out_d:NULL; }
float *picolm_gpu_pipe_ffn_norm(int d) { return (d==0&&G.pipe_ready)?(float*)G.pipe_ffn_norm_d:NULL; }
float *picolm_gpu_pipe_gate(int d) { return (d==0&&G.pipe_ready)?(float*)G.pipe_gate_d:NULL; }
float *picolm_gpu_pipe_up(int d) { return (d==0&&G.pipe_ready)?(float*)G.pipe_up_d:NULL; }
float *picolm_gpu_pipe_x_b(int d) { return (d==0&&G.pipe_b_ready)?(float*)G.pipe_x_b_d:NULL; }
float *picolm_gpu_pipe_xb_b(int d) { return (d==0&&G.pipe_b_ready)?(float*)G.pipe_xb_b_d:NULL; }
float *picolm_gpu_pipe_q_b(int d) { return (d==0&&G.pipe_b_ready)?(float*)G.pipe_q_b_d:NULL; }
float *picolm_gpu_pipe_k_b(int d) { return (d==0&&G.pipe_b_ready)?(float*)G.pipe_k_b_d:NULL; }
float *picolm_gpu_pipe_v_b(int d) { return (d==0&&G.pipe_b_ready)?(float*)G.pipe_v_b_d:NULL; }
float *picolm_gpu_pipe_attn_out_b(int d) { return (d==0&&G.pipe_b_ready)?(float*)G.pipe_attn_out_b_d:NULL; }
void *picolm_gpu_pipe_x_b_mapped(int d) { return (d==0&&G.pipe_b_ready)?G.pipe_x_b_mapped:NULL; }

int picolm_gpu_d2h_via_mapped(void *dst, const void *src, size_t bytes, int device) {
    /* D2H via mapped memory for pipe buffers that are host-visible */
    if (!G.ready || !dst || !src || device != 0) return 0;
    VkDeviceSize off;
    VkBuffer buf = unwrap_buf_offset(src, &off);
    if (!buf) return 0;
    /* Check if this buffer has a mapped pointer */
    void *mapped = NULL;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (buf == G.pipe_x_b) { mapped = G.pipe_x_b_mapped; mem = G.pipe_x_b_m; }
    static int _d2h_dbg = 0;
    if (!_d2h_dbg++) fprintf(stderr, "[D2H_MAPPED] buf=%p mapped=%p mem=%p off=%zu\n", (void*)buf, mapped, (void*)mem, off);
    if (!mapped || !mem) return 0;
    /* Invalidate for CPU read */
    VkMappedMemoryRange inv = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = mem, .offset = 0, .size = VK_WHOLE_SIZE};
    vkInvalidateMappedMemoryRanges(G.dev, 1, &inv);
    memcpy(dst, (const char*)mapped + off, bytes);
    { float rms=0; for(int _i=0;_i<bytes/sizeof(float);_i++) rms+=((float*)dst)[_i]*((float*)dst)[_i];
      rms=sqrtf(rms/(bytes/sizeof(float)));
      fprintf(stderr, "[D2H_MAPPED] dst_rms=%.4f\n", rms); fflush(stderr); }
    return 1;
}
float *picolm_gpu_pipe_ffn_norm_b(int d) { return (d==0&&G.pipe_b_ready)?(float*)G.pipe_ffn_norm_b_d:NULL; }
float *picolm_gpu_pipe_gate_b(int d) { return (d==0&&G.pipe_b_ready)?(float*)G.pipe_gate_b_d:NULL; }
float *picolm_gpu_pipe_up_b(int d) { return (d==0&&G.pipe_b_ready)?(float*)G.pipe_up_b_d:NULL; }
float *picolm_gpu_ssm_qkv_raw(int d) { (void)d; return NULL; }
float *picolm_gpu_ssm_conv_out(int d) { (void)d; return NULL; }
float *picolm_gpu_ssm_xb2(int d) { (void)d; return NULL; }
float *picolm_gpu_ssm_xb2_remap(int d) { (void)d; return NULL; }
float *picolm_gpu_ssm_v_remap(int d) { (void)d; return NULL; }
float *picolm_gpu_ssm_alpha_raw(int d) { (void)d; return NULL; }
float *picolm_gpu_ssm_beta_raw(int d) { (void)d; return NULL; }
float *picolm_gpu_ssm_gate_exp(int d) { (void)d; return NULL; }
float *picolm_gpu_ssm_beta(int d) { (void)d; return NULL; }
float *picolm_gpu_ssm_output(int d) { (void)d; return NULL; }
float *picolm_gpu_ssm_final_output(int d) { (void)d; return NULL; }
uint16_t *picolm_gpu_kv_k_dev(int d) { return (d==0&&G.kv_k_d)?(uint16_t*)G.kv_k_d:NULL; }
uint16_t *picolm_gpu_kv_v_dev(int d) { return (d==0&&G.kv_v_d)?(uint16_t*)G.kv_v_d:NULL; }

// ---------------------------------------------------------------------------
// MATMUL DEV - device-native
// ---------------------------------------------------------------------------

static int _matmul_dev_skip_cnt = 0;

/* Device-native quantize F32->Q8_0. Dispatches quantize kernel on cmd_dev. */
static int _quantize_dev(const float *x_dev, int I, int S) {
    if (!G.shader_quantize) return 0;
    int n_blocks = (I + 31) / 32;
    int S_padded = (S + 15) & ~15;
    size_t xq_bytes = (size_t)S_padded * I;     // int8 packed as uint32: S*I bytes
    size_t xd_bytes = (size_t)S_padded * n_blocks * sizeof(float);
    if (!picolm_gpu_prealloc_q8(xq_bytes, xd_bytes, 0)) return 0;

    VkDescriptorBufferInfo bi[3] = {
        desc_buf_info(x_dev),
        desc_buf_info(G.q8_xq_d),
        desc_buf_info(G.q8_xd_d)
    };
    VkDescriptorSet quant_set = G.dset_quantize_ring[G.dset_idx % 256];
    G.dset_idx++;
    wr_desc(quant_set, 3, bi);
    typedef struct { int I, S, pad; } PC_Q;
    PC_Q pc = {I, S, 0};
    vkCmdBindPipeline(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_quantize);
    vkCmdBindDescriptorSets(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE,
                            G.plyt_unified, 0, 1, &quant_set, 0, NULL);
    vkCmdPushConstants(G.cmd_dev, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd_dev, (uint32_t)n_blocks, (uint32_t)S, 1);

    return 1;
}

/* Device-native Q8xQ8 matmul. Dispatches q8_q8 kernel on cmd_dev. */
static int _q8q8_matmul_dev(picolm_gpu_tensor_t *t, float *y_dev, int S) {
    if (!G.shader_q8q8) return 0;
    int n_blocks = t->I / 32;
    if (n_blocks < 1 || t->I % 32 != 0) return 0;
    uint32_t total = (uint32_t)t->O * (uint32_t)S;
    VkDescriptorBufferInfo bi[4] = {
        desc_buf_info(G.q8_xq_d),   // xq (uint32-packed int8)
        desc_buf_info(G.q8_xd_d),   // xd (float deltas)
        {t->wbuf, 0, VK_WHOLE_SIZE}, // weights
        desc_buf_info(y_dev)        // output
    };
    VkDescriptorSet q8q8_set = G.dset_q8q8_ring[G.dset_idx % 256];
    G.dset_idx++;
    wr_desc(q8q8_set, 4, bi);
    typedef struct { int I, S, O, rowWords; } PC_Q8;
    PC_Q8 pc = {t->I, S, t->O, (int)t->row_words};
    // Memory barrier: quantize writes to q8_xq/q8_xd, q8q8 reads them
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(G.cmd_dev, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_q8q8);
    vkCmdBindDescriptorSets(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE,
                            G.plyt_unified, 0, 1, &q8q8_set, 0, NULL);
    vkCmdPushConstants(G.cmd_dev, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd_dev, (total + 255u) / 256u, 1, 1);
    return 1;
}

int picolm_gpu_matmul_dev(picolm_gpu_tensor_t *t, float *y_dev, const float *x_dev,
                           int S, int device, int y_stride, int x_stride) {
    if (!G.ready || !t || device != 0 || S < 1) return 0;
    if (t->I < 64 || t->O < 32) {
        if (_matmul_dev_skip_cnt++ < 3) fprintf(stderr, "[VK] matmul_dev skip: I=%d O=%d (need >=64/32)\n", t->I, t->O);
        return 0;
    }
    // x_stride and y_stride are passed through PC_Matmul push constants
    VkDescriptorBufferInfo xbi = desc_buf_info(x_dev), ybi = desc_buf_info(y_dev);
    if (!xbi.buffer || !ybi.buffer) return 0;

    VK_BATCH_DISPATCH_PRE();
    VK_BATCH_BARRIER();

    int ok = 0;  // Assume fallback needed
    // Q8_0 path: quantize F32->Q8 on device, then Q8xQ8 matmul
    // Q8_0 path: quantize F32->Q8 on device, then Q8xQ8 matmul
    if (t->qtype == GGUF_TYPE_Q8_0 && G.shader_quantize && G.shader_q8q8) {
        if (getenv("PICOLM_DISPATCH")) {
            static int _q8d=0; if(!_q8d){_q8d=1;fprintf(stderr,"DISPATCH matmul_dev: Q8_0 I=%d O=%d S=%d\n",t->I,t->O,S);}
        }
        if (_quantize_dev(x_dev, t->I, S)) {
            // Memory barrier: quantize wrote to q8_xq/q8_xd, q8q8 reads from them
            VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(G.cmd_dev, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
            ok = _q8q8_matmul_dev(t, y_dev, S);
        }
    }
    // Fallback: scalar dequant matmul (existing shader)
    if (!ok) {
        if (getenv("PICOLM_DISPATCH")) {
            static int _fd=0; if(!_fd){_fd=1;fprintf(stderr,"DISPATCH matmul_dev: fallback scalar I=%d O=%d S=%d qtype=%d\n",t->I,t->O,S,t->qtype);}
        }
        VkDescriptorBufferInfo bi[3] = {xbi, {t->wbuf,0,VK_WHOLE_SIZE}, ybi};
        wr_desc_ring(3, bi);
        VkDescriptorSet cur_dset = G.dset[(G.dset_idx - 1) % 256];
        vkCmdBindPipeline(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
        vkCmdBindDescriptorSets(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE,
                                G.plyt_unified, 0, 1, &cur_dset, 0, NULL);
        PC_Matmul pc = {t->qtype, S, t->I, t->O, (int)t->row_words, x_stride, y_stride};
        vkCmdPushConstants(G.cmd_dev, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        // Phase 2: 2D dispatch [O, S, 1], local_size_x=16 -> ceil(O/16) actual groups
        vkCmdDispatch(G.cmd_dev, (uint32_t)((t->O + 15) / 16), (uint32_t)S, 1);
        ok = 1;
    }

    VK_BATCH_DISPATCH_POST();

    return ok;
}

int picolm_gpu_matmul_dev_strided(picolm_gpu_tensor_t *t, float *y_dev,
                                   const float *x_dev, int S, int device,
                                   int x_stride, int y_stride) {
    return picolm_gpu_matmul_dev(t, y_dev, x_dev, S, device, y_stride, x_stride);
}

int picolm_gpu_matmul_dev_qkv(picolm_gpu_tensor_t *tq, picolm_gpu_tensor_t *tk,
                               picolm_gpu_tensor_t *tv, float *bq, float *bk, float *bv,
                               const float *x_dev, int S, int device,
                               int ysq, int ysk, int xs) {
    (void)ysq; (void)ysk; (void)xs;
    if (!picolm_gpu_matmul_dev(tq, bq, x_dev, S, device, 0, 0)) return 0;
    if (!picolm_gpu_matmul_dev(tk, bk, x_dev, S, device, 0, 0)) return 0;
    if (!picolm_gpu_matmul_dev(tv, bv, x_dev, S, device, 0, 0)) return 0;
    return 1;
}

int picolm_gpu_matmul_dev_gu(picolm_gpu_tensor_t *tg, picolm_gpu_tensor_t *tu,
                              float *bgate, float *bup,
                              const float *x_dev, int S, int device,
                              int ys, int xs) {
    (void)ys; (void)xs;
    if (!picolm_gpu_matmul_dev(tg, bgate, x_dev, S, device, 0, 0)) return 0;
    if (!picolm_gpu_matmul_dev(tu, bup, x_dev, S, device, 0, 0)) return 0;
    return 1;
}

// ---------------------------------------------------------------------------
// ELEMENTWISE (device-native)
// ---------------------------------------------------------------------------

typedef struct { int op; int n; float fp; int pad; } PC_Elem;

int picolm_gpu_rmsnorm_dev(float *out, const float *x, const float *weight,
                            int dim, float eps, int device) {
    return picolm_gpu_rmsnorm_batched_dev(out, x, weight, dim, eps, 1, dim, device);
}

int picolm_gpu_rmsnorm_batched_dev(float *out, const float *x, const float *weight,
                                    int dim, float eps, int S, int xs, int device) {
    if (!G.shader_nrm) { fprintf(stderr, "FATAL: [VK] rmsnorm dispatch with no shader\n"); abort(); }
    if (device != 0) return 0;
    VkDescriptorBufferInfo bi[3] = {desc_buf_info(x), desc_buf_info(weight), desc_buf_info(out)};
    if (!bi[0].buffer || !bi[1].buffer || !bi[2].buffer) { 
        fprintf(stderr, "[RN_DESC] FAIL x_buf=%p w_buf=%p y_buf=%p weight=%p x=%p out=%p\n", 
                (void*)bi[0].buffer, (void*)bi[1].buffer, (void*)bi[2].buffer, (void*)weight, (void*)x, (void*)out); 
        return 0; 
    }
    VkDescriptorSet nrm_set = G.dset_nrm_ring[G.dset_idx % 256];
    G.dset_idx++;
    wr_desc(nrm_set, 3, bi);
    VK_BATCH_DISPATCH_PRE();
    VK_BATCH_BARRIER();
    vkCmdBindPipeline(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_nrm);
    // Update descriptor set: bindings 0=X, 1=W, 2=Y, 3=PC buffer
    if (!G.buf_pc_nrm) {
        // Allocate device-local PC buffer (H2D via staging, not host-mapped)
        VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = 64, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VKCHECK(vkCreateBuffer(G.dev, &bci, NULL, &G.buf_pc_nrm), "bufPcNrm");
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(G.dev, G.buf_pc_nrm, &mr);
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mr.size, .memoryTypeIndex = G.memtype_local};
        VKCHECK(vkAllocateMemory(G.dev, &ai, NULL, &G.mem_pc_nrm), "memPcNrm");
        VKCHECK(vkBindBufferMemory(G.dev, G.buf_pc_nrm, G.mem_pc_nrm, 0), "bindPcNrm");
    }
    // Write params via H2D transfer through staging buffer
    float params[4] = { (float)S, (float)dim, eps, (float)xs };
    memcpy(G.staging_ptr, params, sizeof(params));
    /* Ensure CPU write is visible before GPU reads from staging */
    VkMappedMemoryRange flush = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = G.staging_mem, .offset = 0, .size = sizeof(params)};
    vkFlushMappedMemoryRanges(G.dev, 1, &flush);
    VkDescriptorBufferInfo bi_dev[4] = {
        desc_buf_info(x), desc_buf_info(weight), desc_buf_info(out),
        {G.buf_pc_nrm, 0, 64}
    };
    vkCmdCopyBuffer(G.cmd_dev, G.staging_buf, G.buf_pc_nrm, 1, &(VkBufferCopy){0, 0, 64});
    // Barrier: ensure transfer completes before compute reads
    VkMemoryBarrier mb_pc = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(G.cmd_dev, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_pc, 0, NULL, 0, NULL);
    // RMSNorm: use ring buffer descriptor set (avoids aliasing in batched prefill)
    VkDescriptorBufferInfo rn_bi[3] = {desc_buf_info(x), desc_buf_info(weight), desc_buf_info(out)};
    wr_desc(nrm_set, 3, rn_bi);
    vkCmdBindDescriptorSets(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_unified, 0, 1, &nrm_set, 0, NULL);
    // Push constants via PC_Matmul format: fmt=0, S, I=dim, O=eps(bitcast), rowWords=xs
    PC_Matmul pc_nrm = {0, S, dim, 0, xs};
    memcpy(&pc_nrm.O, &eps, sizeof(float));
    vkCmdPushConstants(G.cmd_dev, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_nrm), &pc_nrm);
    vkCmdDispatch(G.cmd_dev, (uint32_t)S, 1, 1);
    VK_BATCH_DISPATCH_POST();

    return 1;
}

int picolm_gpu_residual_add(float *out, const float *a, const float *b,
                             int n, int dim, int stride, int device) {
    if (!G.ready || !G.shader_elem || device != 0 || n < 1) return 0;
    (void)dim; (void)stride;
    VkDescriptorBufferInfo bi[4] = {desc_buf_info(a), desc_buf_info(b), desc_buf_info(out), {VK_NULL_HANDLE,0,0}};
    if (!bi[0].buffer || !bi[1].buffer || !bi[2].buffer) { fprintf(stderr, "[RN_DESC] FAIL x_buf=%p w_buf=%p y_buf=%p\n", (void*)bi[0].buffer, (void*)bi[1].buffer, (void*)bi[2].buffer); return 0; }
    VkDescriptorSet elem_set = G.dset_elem_ring[G.dset_idx % 256];
    G.dset_idx++;
    wr_desc(elem_set, 4, bi);
    VK_BATCH_DISPATCH_PRE();
    VK_BATCH_BARRIER();
    vkCmdBindPipeline(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_elem);
    vkCmdBindDescriptorSets(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_unified, 0, 1, &elem_set, 0, NULL);
    int total = n * dim;
    PC_Elem pc = {0, total, 0.0f, 0};
    vkCmdPushConstants(G.cmd_dev, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd_dev, (uint32_t)((total + 255) / 256), 1, 1);
    VK_BATCH_DISPATCH_POST();
    return 1;
}

int picolm_gpu_silu_mul_dev(float *g, const float *u, size_t n, int device) {
    if (!G.ready || !G.shader_elem || device != 0 || n < 1) return 0;
    VkDescriptorBufferInfo gbi = desc_buf_info(g), ubi = desc_buf_info(u);
    if (!gbi.buffer || !ubi.buffer) return 0;
    VkDescriptorBufferInfo bi[4] = {gbi, ubi, gbi, {VK_NULL_HANDLE,0,0}};
    VkDescriptorSet elem_set = G.dset_elem_ring[G.dset_idx % 256];
    G.dset_idx++;
    wr_desc(elem_set, 4, bi);
    VK_BATCH_DISPATCH_PRE();
    VK_BATCH_BARRIER();
    vkCmdBindPipeline(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_elem);
    vkCmdBindDescriptorSets(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_unified, 0, 1, &elem_set, 0, NULL);
    PC_Elem pc = {1, (int)n, 0.0f, 0};
    vkCmdPushConstants(G.cmd_dev, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd_dev, (uint32_t)((n + 255) / 256), 1, 1);
    VK_BATCH_DISPATCH_POST();
    return 1;
}

// GELU multiply: gate = gelu(gate) * up
int picolm_gpu_gelu_mul_dev(float *g, const float *u, size_t n, int device) {
    if (!G.ready || !G.shader_elem || device != 0 || n < 1) return 0;
    VkDescriptorBufferInfo gbi = desc_buf_info(g), ubi = desc_buf_info(u);
    if (!gbi.buffer || !ubi.buffer) return 0;
    VkDescriptorBufferInfo bi[4] = {gbi, ubi, gbi, {VK_NULL_HANDLE,0,0}};
    VkDescriptorSet elem_set = G.dset_elem_ring[G.dset_idx % 256];
    G.dset_idx++;
    wr_desc(elem_set, 4, bi);
    VK_BATCH_DISPATCH_PRE();
    VK_BATCH_BARRIER();
    vkCmdBindPipeline(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_elem);
    vkCmdBindDescriptorSets(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_unified, 0, 1, &elem_set, 0, NULL);
    PC_Elem pc = {5, (int)n, 0.0f, 0};
    vkCmdPushConstants(G.cmd_dev, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd_dev, (uint32_t)((n + 255) / 256), 1, 1);
    VK_BATCH_DISPATCH_POST();
    return 1;
}

// ---------------------------------------------------------------------------
// ROPE APPLY (device-native)
// ---------------------------------------------------------------------------

int picolm_gpu_rope_apply(float *x, int n_heads, int head_dim,
                           const float *cos_tbl, const float *sin_tbl,
                           int half_dim, int rope_type, int device) {
    if (!G.ready || !G.shader_elem || device != 0 || half_dim < 1) return 0;
    VkBuffer xb = unwrap_buf(x);
    VkDeviceSize cos_off = 0, sin_off = 0;
    VkBuffer cb = unwrap_buf_offset(cos_tbl, &cos_off), sb = unwrap_buf_offset(sin_tbl, &sin_off);
    if (!xb || !cb || !sb) return 0;
    VkDescriptorBufferInfo bi[4] = {{xb,0,VK_WHOLE_SIZE},{cb,cos_off,VK_WHOLE_SIZE},{sb,sin_off,VK_WHOLE_SIZE},{VK_NULL_HANDLE,0,0}};
    VkDescriptorSet elem_set = G.dset_elem_ring[G.dset_idx % 256];
    G.dset_idx++;
    wr_desc(elem_set, 4, bi);
    VK_BATCH_DISPATCH_PRE();
    vkCmdBindPipeline(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_elem);
    vkCmdBindDescriptorSets(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_unified, 0, 1, &elem_set, 0, NULL);
    PC_Elem pc = {2, half_dim, (float)n_heads, rope_type};
    vkCmdPushConstants(G.cmd_dev, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd_dev, (uint32_t)((half_dim + 255) / 256), 1, 1);
    VK_BATCH_DISPATCH_POST();
    return 1;
}

int picolm_gpu_rope_apply_batched(float *x, int n_heads, int head_dim,
                                   const float *cos_tbl_base, const float *sin_tbl_base,
                                   int half_dim, int start_pos, int S,
                                   int rope_type, int device) {
    if (!G.ready || !G.shader_elem || device != 0 || half_dim < 1 || S < 1) return 0;

    if (!G.in_batch) {
        vk_fence_wait_timeout(G.dev, G.fence_dev, 10ULL*1000*1000*1000);
        VkCommandBufferBeginInfo _vbb = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkResetCommandBuffer(G.cmd_dev, 0);
        vkBeginCommandBuffer(G.cmd_dev, &_vbb);
    } else {
        // Memory barrier: Q/K matmul writes must be visible before RoPE reads
        VkMemoryBarrier _mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(G.cmd_dev, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &_mb, 0, NULL, 0, NULL);
    }

    // Batched rope: process each token sequentially, each with its own cos/sin offset
    // cos/sin tables are [max_seq_len][half_dim], offset by start_pos+token_idx
    for (int si = 0; si < S; si++) {
        // Token si's x data at offset si * n_heads * head_dim
        float *x_tok = (float *)((uintptr_t)x + (uintptr_t)(size_t)si * n_heads * head_dim * sizeof(float));
        const float *cos_tok = (const float *)((uintptr_t)cos_tbl_base + (uintptr_t)(size_t)(start_pos + si) * half_dim * sizeof(float));
        const float *sin_tok = (const float *)((uintptr_t)sin_tbl_base + (uintptr_t)(size_t)(start_pos + si) * half_dim * sizeof(float));

        VkDeviceSize xb_off = 0;
        VkBuffer xb = unwrap_buf_offset(x_tok, &xb_off);
        VkDeviceSize cos_off = 0, sin_off = 0;
        VkBuffer cb = unwrap_buf_offset(cos_tok, &cos_off);
        VkBuffer sb = unwrap_buf_offset(sin_tok, &sin_off);
        if (!xb || !cb || !sb) return 0;

        size_t token_bytes = (size_t)n_heads * head_dim * sizeof(float);

        VkDescriptorBufferInfo bi[3] = {
            {xb, xb_off, token_bytes},
            {cb, cos_off, (size_t)half_dim * sizeof(float)},
            {sb, sin_off, (size_t)half_dim * sizeof(float)},
        };
        VkDescriptorSet elem_set = G.dset_elem_ring[G.dset_idx % 256];
        G.dset_idx++;
        wr_desc(elem_set, 3, bi);

        vkCmdBindPipeline(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_elem);
        vkCmdBindDescriptorSets(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE,
                                G.plyt_unified, 0, 1, &elem_set, 0, NULL);
        typedef struct { int op, n; float fp; int pad; } PC_Elem;
        PC_Elem pc = {2, half_dim, (float)n_heads, rope_type};
        vkCmdPushConstants(G.cmd_dev, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(G.cmd_dev, (uint32_t)((half_dim + 255) / 256), 1, 1);
    }

    if (!G.in_batch) {
        vkEndCommandBuffer(G.cmd_dev);
        VkSubmitInfo si_info = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_dev};
        vkResetFences(G.dev, 1, &G.fence_dev);
        vkQueueSubmit(G.queue, 1, &si_info, G.fence_dev);
        vk_fence_wait_timeout(G.dev, G.fence_dev, 10ULL*1000*1000*1000);
    }
    return 1;
}

// ---------------------------------------------------------------------------
// KV CACHE
// ---------------------------------------------------------------------------

int picolm_gpu_kv_alloc(size_t kv_k_bytes, size_t kv_v_bytes, int device) {
    if (!G.ready || device != 0) return 0;
    picolm_gpu_kv_free();
    fprintf(stderr, "[VK] kv_alloc: k=%zu MB, v=%zu MB, total=%zu MB\n",
            kv_k_bytes/(1024*1024), kv_v_bytes/(1024*1024),
            (kv_k_bytes+kv_v_bytes)/(1024*1024));
    int ok = 1;
    // Use host-visible memory for KV cache (KAVERI shader reads from device-local 10MB buffers are broken)
    VkBufferUsageFlags kv_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (kv_k_bytes) ok &= alloc_hostvis_mt2(kv_k_bytes, &G.kv_k_buf, &G.kv_k_mem, NULL, G.memtype, kv_usage);
    if (kv_v_bytes) ok &= alloc_hostvis_mt2(kv_v_bytes, &G.kv_v_buf, &G.kv_v_mem, NULL, G.memtype, kv_usage);
    if (!ok) { picolm_gpu_kv_free(); return 0; }
    G.kv_k_bytes = kv_k_bytes;
    G.kv_v_bytes = kv_v_bytes;
    G.kv_k_d = kv_k_bytes ? wrap_buf(G.kv_k_buf, G.kv_k_mem, 0, kv_k_bytes) : NULL;
    G.kv_v_d = kv_v_bytes ? wrap_buf(G.kv_v_buf, G.kv_v_mem, 0, kv_v_bytes) : NULL;
    // Map KV cache for CPU readback (needed for CPU decode)
    G.kv_k_mapped = NULL; G.kv_v_mapped = NULL;
    if (kv_k_bytes) VKCHECK(vkMapMemory(G.dev, G.kv_k_mem, 0, kv_k_bytes, 0, &G.kv_k_mapped), "mapKvK");
    if (kv_v_bytes) VKCHECK(vkMapMemory(G.dev, G.kv_v_mem, 0, kv_v_bytes, 0, &G.kv_v_mapped), "mapKvV");
    fprintf(stderr, "[VK] kv_alloc: OK (mapped for CPU readback)\n");
    return 1;
}

int picolm_gpu_kv_store_rows(int is_k, int lo, int sp, int np,
                              const void *hr, size_t rb,
                              int nkh, int hd, int msl, int device) {
    if (!G.ready || !hr || device != 0) return 0;
    VkBuffer dst_buf = is_k ? G.kv_k_buf : G.kv_v_buf;
    if (!dst_buf) return 0;
    size_t row_bytes = (size_t)nkh * hd * sizeof(uint16_t);
    size_t off = (size_t)lo * msl * row_bytes + (size_t)sp * row_bytes;
    size_t bytes = (size_t)np * row_bytes;
    /* Ensure any prior xfer is done before reusing cmd_xfer/fence_xfer */
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
    if (!staging_ensure(bytes)) return 0;
    memcpy(G.staging_ptr, hr, bytes);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkResetCommandBuffer(G.cmd_xfer, 0);
    vkBeginCommandBuffer(G.cmd_xfer, &begin);
    VkBufferCopy bc = {(VkDeviceSize)0, (VkDeviceSize)off, bytes};
    vkCmdCopyBuffer(G.cmd_xfer, G.staging_buf, dst_buf, 1, &bc);
    vkEndCommandBuffer(G.cmd_xfer);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
    vkResetFences(G.dev, 1, &G.fence_xfer);
    vkQueueSubmit(G.queue, 1, &si, G.fence_xfer);
    return 1;
}

int picolm_gpu_kv_store_dev(int is_k, int lo, int pos,
                             const float *sd, int nkh, int hd,
                             int msl, int device) {
    /* Decode path: single F32 row -> F16 -> KV cache buffer */
    if (!G.ready || !sd || device != 0) return 0;
    VkBuffer dst_buf = is_k ? G.kv_k_buf : G.kv_v_buf;
    if (!dst_buf) return 0;

    int kv_dim = nkh * hd;
    size_t row_bytes_f32 = (size_t)kv_dim * sizeof(float);
    size_t row_bytes_f16 = (size_t)kv_dim * sizeof(uint16_t);

    vk_fence_wait_timeout(G.dev, G.fence_dev, 5ULL*1000*1000*1000);
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 5ULL*1000*1000*1000);

    VkDeviceSize src_off;
    VkBuffer src_buf = unwrap_buf_offset(sd, &src_off);
    if (!src_buf) return 0;

    if (!staging_ensure(row_bytes_f32)) return 0;

    /* D2H */
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkResetCommandBuffer(G.cmd_xfer, 0);
    vkBeginCommandBuffer(G.cmd_xfer, &begin);
    VkBufferCopy bc = {src_off, (VkDeviceSize)0, row_bytes_f32};
    vkCmdCopyBuffer(G.cmd_xfer, src_buf, G.staging_buf, 1, &bc);
    vkEndCommandBuffer(G.cmd_xfer);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
    vkResetFences(G.dev, 1, &G.fence_xfer);
    vkQueueSubmit(G.queue, 1, &si, G.fence_xfer);
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);

    /* F32->F16 pack using tested fp32_to_fp16() from quant.c */
    uint16_t *h_pack = malloc(row_bytes_f16);
    if (!h_pack) return 0;
    const float *src = (const float *)G.staging_ptr;
    for (int j = 0; j < kv_dim; j++) {
        h_pack[j] = fp32_to_fp16(src[j]);
    }

    /* H2D to KV cache */
    size_t dst_off = (size_t)lo * msl * row_bytes_f16 + (size_t)pos * row_bytes_f16;
    memcpy(G.staging_ptr, h_pack, row_bytes_f16);
    free(h_pack);

    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
    vkResetCommandBuffer(G.cmd_xfer, 0);
    vkBeginCommandBuffer(G.cmd_xfer, &begin);
    VkBufferCopy bc2 = {(VkDeviceSize)0, (VkDeviceSize)dst_off, row_bytes_f16};
    vkCmdCopyBuffer(G.cmd_xfer, G.staging_buf, dst_buf, 1, &bc2);
    vkEndCommandBuffer(G.cmd_xfer);
    VkSubmitInfo si2 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
    vkResetFences(G.dev, 1, &G.fence_xfer);
    vkQueueSubmit(G.queue, 1, &si2, G.fence_xfer);
    /* Wait for H2D to complete before returning. */
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
    return 1;
}

/* CPU-side F32->F16 conversion (same algorithm as kv_store_vk.comp) */
static uint16_t f32tof16_cpu(float v) {
    union { float f; uint32_t u; } u = {.f = v};
    uint32_t sign = (u.u >> 16) & 0x8000u;
    uint32_t exp = (u.u >> 23) & 0xffu;
    uint32_t frac = u.u & 0x7fffffu;
    if (exp == 0) return (uint16_t)sign;
    if (exp == 255) return (uint16_t)(sign | 0x7c00u);
    int e = (int)exp - 127 + 15;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (uint32_t)e << 10 | (frac >> 13));
}

int picolm_gpu_kv_store_dev_batched(int is_k, int lo, int sp, int np,
                                     const float *sd, int nkh, int hd,
                                     int msl, int device) {
    if (!G.ready || !sd || device != 0) return 0;
    VkBuffer dst_buf = is_k ? G.kv_k_buf : G.kv_v_buf;
    if (!dst_buf) return 0;

    int kv_dim = nkh * hd;
    size_t total_f32 = (size_t)np * kv_dim;
    size_t total_f16 = total_f32 * sizeof(uint16_t);

    size_t f16_per_pos_bytes = (size_t)nkh * hd * sizeof(uint16_t);
    size_t dst_byte_off = (size_t)lo * msl * f16_per_pos_bytes + (size_t)sp * f16_per_pos_bytes;

    /* D2H + CPU F32->F16 + H2D path.
     * Must end compute batch first so the K/V matmul writes are visible. */
    int was_in_batch = G.in_batch;
    if (G.in_batch) picolm_gpu_batch_end(device);
    // Now fence_dev is signaled for the batch that contained the K/V matmul
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);

    // D2H
    if (!staging_ensure(total_f32 * sizeof(float))) return 0;
    VkDescriptorBufferInfo src_info = desc_buf_info(sd);
    VkCommandBufferBeginInfo _bgi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkResetCommandBuffer(G.cmd_xfer, 0); vkBeginCommandBuffer(G.cmd_xfer, &_bgi);
    VkBufferCopy _bc_d2h = {src_info.offset, 0, total_f32 * sizeof(float)};
    vkCmdCopyBuffer(G.cmd_xfer, src_info.buffer, G.staging_buf, 1, &_bc_d2h);
    vkEndCommandBuffer(G.cmd_xfer);
    VkSubmitInfo _si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
    vkResetFences(G.dev, 1, &G.fence_xfer);
    vkQueueSubmit(G.queue, 1, &_si, G.fence_xfer);
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);

    // CPU F32->F16
    { float *sf = (float*)G.staging_ptr;
      uint16_t *pk = (uint16_t*)malloc(total_f16);
      for(size_t i=0;i<total_f32;i++) pk[i] = f32tof16_cpu(sf[i]);
      memcpy(G.staging_ptr, pk, total_f16); free(pk); }

    // H2D
    vkResetCommandBuffer(G.cmd_xfer, 0); vkBeginCommandBuffer(G.cmd_xfer, &_bgi);
    VkBufferCopy _bc_h2d = {0, dst_byte_off, total_f16};
    vkCmdCopyBuffer(G.cmd_xfer, G.staging_buf, dst_buf, 1, &_bc_h2d);
    vkEndCommandBuffer(G.cmd_xfer);
    vkResetFences(G.dev, 1, &G.fence_xfer);
    vkQueueSubmit(G.queue, 1, &_si, G.fence_xfer);
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);

    // Restore batch state for subsequent dispatches
    if (was_in_batch && !G.in_batch) picolm_gpu_batch_begin(device);

    return 1;
}

int picolm_gpu_kv_store_dev_batched_strided(int is_k, int lo, int sp, int np,
                                             const float *sd, int nkh, int hd,
                                             int msl, int device, int src_stride) {
    if (!G.ready || !sd || device != 0) return 0;
    VkBuffer dst_buf = is_k ? G.kv_k_buf : G.kv_v_buf;
    if (!dst_buf) return 0;

    /* This function does synchronous D2H+CPU+H2D. Flush batch if active. */
    int was_in_batch = G.in_batch;
    if (G.in_batch) picolm_gpu_batch_end(device);
    picolm_gpu_sync(device);  // Same RADV/llvmpipe fence sync fix

    /* D2H copy from device, then F32->F16 + strided reorder, then H2D */
    int kv_dim = nkh * hd;
    size_t row_bytes_f32 = (size_t)kv_dim * sizeof(float);
    size_t row_bytes_f16 = (size_t)kv_dim * sizeof(uint16_t);
        /* Wait for both prior dev and xfer fences */
    vk_fence_wait_timeout(G.dev, G.fence_dev, 5ULL*1000*1000*1000);
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 5ULL*1000*1000*1000);

    /* Unwrap the device pointer */
    VkDeviceSize src_off;
    VkBuffer src_buf = unwrap_buf_offset(sd, &src_off);
    if (!src_buf) return 0;

    /* Allocate temporary host buffer for packed F16 output */
    size_t total_f16 = (size_t)np * row_bytes_f16;
    uint16_t *h_pack = malloc(total_f16);
    if (!h_pack) return 0;

    /* D2H row-by-row: each row is src_stride floats apart in device memory,
     * but we read kv_dim floats and pack to F16. */
    /* Staging must hold the larger of: one F32 row (D2H read) or all np F16 rows (H2D write) */
    { size_t staging_needed = row_bytes_f32;
      size_t dst_bytes_all = (size_t)np * row_bytes_f16;
      if (dst_bytes_all > staging_needed) staging_needed = dst_bytes_all;
      if (!staging_ensure(staging_needed)) { free(h_pack); return 0; }
    }
    for (int p = 0; p < np; p++) {
        /* D2H: copy this row from device */
        VkDeviceSize src_row_off = src_off + (VkDeviceSize)p * src_stride * sizeof(float);
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkResetCommandBuffer(G.cmd_xfer, 0);
        vkBeginCommandBuffer(G.cmd_xfer, &begin);
        VkBufferCopy bc = {src_row_off, (VkDeviceSize)0, row_bytes_f32};
        vkCmdCopyBuffer(G.cmd_xfer, src_buf, G.staging_buf, 1, &bc);
        vkEndCommandBuffer(G.cmd_xfer);
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
        vkResetFences(G.dev, 1, &G.fence_xfer);
        vkQueueSubmit(G.queue, 1, &si, G.fence_xfer);
        vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);

        /* F32->F16 pack for this row */
        const float *src = (const float *)G.staging_ptr;
        uint16_t *dst_row = h_pack + (size_t)p * kv_dim;
        for (int j = 0; j < kv_dim; j++) {
            float v = src[j];
            uint32_t sign_f; memcpy(&sign_f, &v, 4); uint32_t sign = (sign_f >> 16) & 0x8000;
            uint32_t exp_f; memcpy(&exp_f, &v, 4); uint32_t exp = (exp_f >> 23) & 0xFF;
            uint32_t frac_f; memcpy(&frac_f, &v, 4); uint32_t frac = frac_f & 0x7FFFFF;
            if (exp == 0) dst_row[j] = (uint16_t)sign;
            else if (exp == 0xFF) dst_row[j] = (uint16_t)(sign | 0x7C00);
            else {
                int e = (int)exp - 127 + 15;
                if (e <= 0) dst_row[j] = (uint16_t)sign;
                else if (e >= 31) dst_row[j] = (uint16_t)(sign | 0x7C00);
                else dst_row[j] = (uint16_t)(sign | (e << 10) | (frac >> 13));
            }
        }
    }

    /* H2D: copy packed F16 to KV cache buffer */
    size_t dst_off = (size_t)lo * msl * row_bytes_f16 + (size_t)sp * row_bytes_f16;
    size_t dst_bytes = (size_t)np * row_bytes_f16;
    memcpy(G.staging_ptr, h_pack, dst_bytes);

    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
    VkCommandBufferBeginInfo begin2 = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkResetCommandBuffer(G.cmd_xfer, 0);
    vkBeginCommandBuffer(G.cmd_xfer, &begin2);
    VkBufferCopy bc2 = {(VkDeviceSize)0, (VkDeviceSize)dst_off, dst_bytes};
    vkCmdCopyBuffer(G.cmd_xfer, G.staging_buf, dst_buf, 1, &bc2);
    vkEndCommandBuffer(G.cmd_xfer);
    VkSubmitInfo si2 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
    vkResetFences(G.dev, 1, &G.fence_xfer);
    vkQueueSubmit(G.queue, 1, &si2, G.fence_xfer);

    free(h_pack);

    /* If we were in a batch, restart it */
    if (was_in_batch) picolm_gpu_batch_begin(device);

    return 1;
}

int picolm_gpu_kv_upload_layer(int is_k, int lo, int np,
                                const uint16_t *hr, int nkh,
                                int hd, int msl, int device) {
    if (!hr || np == 0) return 0;
    return picolm_gpu_kv_store_rows(is_k, lo, 0, np, hr,
                                     (size_t)nkh * hd * sizeof(uint16_t),
                                     nkh, hd, msl, device);
}

int picolm_gpu_kv_debug_dump(int is_k, int lo, int pos,
                              uint16_t *dst, int ne, int nkh,
                              int hd, int msl, int device) {
    (void)is_k; (void)lo; (void)pos; (void)dst; (void)ne;
    (void)nkh; (void)hd; (void)msl; (void)device; return 0;
}

void picolm_gpu_kv_cache_clear(int device) {
    (void)device;
    if (G.ready) {
        if (G.kv_k_mapped) { vkUnmapMemory(G.dev, G.kv_k_mem); G.kv_k_mapped = NULL; }
        if (G.kv_v_mapped) { vkUnmapMemory(G.dev, G.kv_v_mem); G.kv_v_mapped = NULL; }
        if (G.pipe_k_b_mapped) { vkUnmapMemory(G.dev, G.pipe_k_b_m); G.pipe_k_b_mapped = NULL; }
        if (G.pipe_v_b_mapped) { vkUnmapMemory(G.dev, G.pipe_v_b_m); G.pipe_v_b_mapped = NULL; }
        if (G.kv_k_d) { free(G.kv_k_d); G.kv_k_d = NULL; }
        if (G.kv_v_d) { free(G.kv_v_d); G.kv_v_d = NULL; }
        if (G.kv_k_mem) { vkFreeMemory(G.dev, G.kv_k_mem, NULL); G.kv_k_mem = VK_NULL_HANDLE; }
        if (G.kv_k_buf) { vkDestroyBuffer(G.dev, G.kv_k_buf, NULL); G.kv_k_buf = VK_NULL_HANDLE; }
        if (G.kv_v_mem) { vkFreeMemory(G.dev, G.kv_v_mem, NULL); G.kv_v_mem = VK_NULL_HANDLE; }
        if (G.kv_v_buf) { vkDestroyBuffer(G.dev, G.kv_v_buf, NULL); G.kv_v_buf = VK_NULL_HANDLE; }
        G.kv_k_bytes = G.kv_v_bytes = 0;
    }
}

void picolm_gpu_kv_free(void) { picolm_gpu_kv_cache_clear(0); }

// ---------------------------------------------------------------------------
// KV CACHE FLUSH (GPU -> CPU)
// ---------------------------------------------------------------------------
int picolm_gpu_kv_flush_to_cpu(uint8_t *cpu_k, uint8_t *cpu_v,
                                int n_attn_ord, int max_seq,
                                int n_pos, size_t row_sz_k, size_t row_sz_v,
                                int device) {
    if (!G.ready || !cpu_k || !cpu_v || device != 0) return 0;
    if (!G.kv_k_mapped || !G.kv_v_mapped) {
        fprintf(stderr, "[VK] kv_flush: not mapped\n");
        return 0;
    }
    // Wait for GPU writes to complete
    vk_fence_wait_timeout(G.dev, G.fence_dev, 10ULL*1000*1000*1000);
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
    // Invalidate mapped memory for CPU read
    VkMappedMemoryRange inv = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
    inv.memory = G.kv_k_mem; inv.offset = 0; inv.size = G.kv_k_bytes;
    vkInvalidateMappedMemoryRanges(G.dev, 1, &inv);
    inv.memory = G.kv_v_mem; inv.size = G.kv_v_bytes;
    vkInvalidateMappedMemoryRanges(G.dev, 1, &inv);
    // Copy row-by-row
    size_t elems_per_row_k = row_sz_k / sizeof(uint16_t);
    size_t elems_per_row_v = row_sz_v / sizeof(uint16_t);
    for (int lo = 0; lo < n_attn_ord; lo++) {
        uint16_t *gk = (uint16_t*)G.kv_k_mapped + (size_t)lo * max_seq * elems_per_row_k;
        uint16_t *gv = (uint16_t*)G.kv_v_mapped + (size_t)lo * max_seq * elems_per_row_v;
        uint8_t *ck = cpu_k + (size_t)lo * max_seq * row_sz_k;
        uint8_t *cv = cpu_v + (size_t)lo * max_seq * row_sz_v;
        for (int p = 0; p < n_pos; p++) {
            memcpy(ck + (size_t)p * row_sz_k, gk + (size_t)p * elems_per_row_k, row_sz_k);
            memcpy(cv + (size_t)p * row_sz_v, gv + (size_t)p * elems_per_row_v, row_sz_v);
        }
    }
    // Debug: dump K cache layers 0-3 (check for non-zero entries)
    return 1;
}
// ---------------------------------------------------------------------------
// EXPERT MLP + W4A16 (stubs)
// ---------------------------------------------------------------------------

int picolm_gpu_expert_mlp(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                           picolm_gpu_tensor_t *down, float *y, const float *x, int S) {
    (void)gate; (void)up; (void)down; (void)y; (void)x; (void)S; return 0;
}
int picolm_gpu_expert_mlp_dev(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                               picolm_gpu_tensor_t *down, float *y_dev,
                               const float *x_dev, int S, int xs, int ys, int dev) {
    (void)gate; (void)up; (void)down; (void)y_dev; (void)x_dev;
    (void)S; (void)xs; (void)ys; (void)dev; return 0;
}
int picolm_gpu_w4a16_mlp(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                          picolm_gpu_tensor_t *down, float *y, const float *x, int S) {
    (void)gate; (void)up; (void)down; (void)y; (void)x; (void)S; return 0;
}
int picolm_gpu_w4a16_matmul(picolm_gpu_tensor_t *t, float *y, const float *x,
                             int S, int device) {
    (void)t; (void)y; (void)x; (void)S; (void)device; return 0;
}
// ---------------------------------------------------------------------------
// ATTENTION (stubs - shaders not yet written)
// ---------------------------------------------------------------------------

int picolm_gpu_attention_decode(float *xb_out, const float *q,
                                 int lo, int pos, int nh, int nkh,
                                 int hd, int msl, int device) {
    (void)xb_out; (void)q; (void)lo; (void)pos; (void)nh; (void)nkh;
    (void)hd; (void)msl; (void)device; return 0;
}
int picolm_gpu_attention_decode_dev(float *xb_out_dev, const float *q_dev,
                                     int lo, int pos, int nh, int nkh,
                                     int hd, int msl, int device) {
    if (1) return 0; // DISABLE GPU DECODE ATTENTION

    // KAVERI workaround: copy KV cache data (F16) from large buffer
    // to small pipeline buffers (F32) to avoid shader read bug.
    // Copy all prior tokens' K/V data for this layer.
    int kv_row = nkh * hd;  // bytes per token per K/V (F32)
    int copy_n = pos + 1;   // number of tokens to copy
    size_t copy_bytes = (size_t)copy_n * kv_row * sizeof(float);

    // Wait for prior KV store (fence_xfer) to complete
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
    vk_fence_wait_timeout(G.dev, G.fence_dev, 10ULL*1000*1000*1000);

    // D2H: read F16 K data from KV cache via staging
    staging_ensure(copy_bytes * 2);  // F16 size (2 bytes per float)
    uint16_t *k_f16 = (uint16_t*)G.staging_ptr;

    VkBufferCopy kc = {
        .srcOffset = (size_t)lo * msl * nkh * hd * sizeof(uint16_t),
        .dstOffset = 0,
        .size = (size_t)copy_n * kv_row * sizeof(uint16_t)
    };
    vk_fence_wait_timeout(G.dev, G.fence_xfer, 10ULL*1000*1000*1000);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkResetCommandBuffer(G.cmd_xfer, 0);
    vkBeginCommandBuffer(G.cmd_xfer, &begin);
    vkCmdCopyBuffer(G.cmd_xfer, G.kv_k_buf, G.staging_buf, 1, &kc);
    vkEndCommandBuffer(G.cmd_xfer);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
    vkResetFences(G.dev, 1, &G.fence_xfer);
    vkQueueSubmit(G.queue, 1, &si, G.fence_xfer);
    vkWaitForFences(G.dev, 1, &G.fence_xfer, VK_TRUE, UINT64_MAX);

    // CPU: F16->F32 conversion for K
    float *k_f32 = picolm_gpu_staging_host(device, copy_bytes);
    for (int i = 0; i < (int)(copy_n * kv_row); i++) {
        k_f32[i] = fp16_to_fp32(k_f16[i]);
    }

    // H2D: write F32 K to pipeline buffer
    staging_ensure(copy_bytes);
    memcpy(G.staging_ptr, k_f32, copy_bytes);
    VkMappedMemoryRange mrk = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = G.staging_mem,
        .offset = 0,
        .size = copy_bytes
    };
    vkFlushMappedMemoryRanges(G.dev, 1, &mrk);

    VkBufferCopy kcu = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = copy_bytes
    };
    vkResetCommandBuffer(G.cmd_xfer, 0);
    vkBeginCommandBuffer(G.cmd_xfer, &begin);
    vkCmdCopyBuffer(G.cmd_xfer, G.staging_buf, unwrap_buf(G.pipe_k_b_d), 1, &kcu);
    vkEndCommandBuffer(G.cmd_xfer);
    si = (VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
    vkResetFences(G.dev, 1, &G.fence_xfer);
    vkQueueSubmit(G.queue, 1, &si, G.fence_xfer);
    vkWaitForFences(G.dev, 1, &G.fence_xfer, VK_TRUE, UINT64_MAX);

    // Same for V
    VkBufferCopy vc = {
        .srcOffset = (size_t)lo * msl * nkh * hd * sizeof(uint16_t),
        .dstOffset = 0,
        .size = (size_t)copy_n * kv_row * sizeof(uint16_t)
    };
    vkResetCommandBuffer(G.cmd_xfer, 0);
    vkBeginCommandBuffer(G.cmd_xfer, &begin);
    vkCmdCopyBuffer(G.cmd_xfer, G.kv_v_buf, G.staging_buf, 1, &vc);
    vkEndCommandBuffer(G.cmd_xfer);
    si = (VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
    vkResetFences(G.dev, 1, &G.fence_xfer);
    vkQueueSubmit(G.queue, 1, &si, G.fence_xfer);
    vkWaitForFences(G.dev, 1, &G.fence_xfer, VK_TRUE, UINT64_MAX);

    float *v_f32 = picolm_gpu_staging_host(device, copy_bytes);
    for (int i = 0; i < (int)(copy_n * kv_row); i++) {
        v_f32[i] = fp16_to_fp32(((uint16_t*)G.staging_ptr)[i]);
    }

    staging_ensure(copy_bytes);
    memcpy(G.staging_ptr, v_f32, copy_bytes);
    VkMappedMemoryRange mrv = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = G.staging_mem,
        .offset = 0,
        .size = copy_bytes
    };
    vkFlushMappedMemoryRanges(G.dev, 1, &mrv);

    VkBufferCopy vcu = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = copy_bytes
    };
    vkResetCommandBuffer(G.cmd_xfer, 0);
    vkBeginCommandBuffer(G.cmd_xfer, &begin);
    vkCmdCopyBuffer(G.cmd_xfer, G.staging_buf, unwrap_buf(G.pipe_v_b_d), 1, &vcu);
    vkEndCommandBuffer(G.cmd_xfer);
    si = (VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_xfer};
    vkResetFences(G.dev, 1, &G.fence_xfer);
    vkQueueSubmit(G.queue, 1, &si, G.fence_xfer);
    vkWaitForFences(G.dev, 1, &G.fence_xfer, VK_TRUE, UINT64_MAX);

    // Now bind the pipeline buffers (F32 K/V) instead of KV cache
    VkDescriptorBufferInfo bi[4] = {
        desc_buf_info(q_dev),
        desc_buf_info((float*)G.pipe_k_b_d),
        desc_buf_info((float*)G.pipe_v_b_d),
        desc_buf_info(xb_out_dev)
    };
    if (!bi[0].buffer || !bi[3].buffer) return 0;
    wr_desc(G.dset_attn_dec, 4, bi);

    vkResetCommandBuffer(G.cmd_dev, 0);
    vkBeginCommandBuffer(G.cmd_dev, &begin);
    vkCmdBindPipeline(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_attn_dec);
    vkCmdBindDescriptorSets(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE,
                            G.plyt_unified, 0, 1, &G.dset_attn_dec, 0, NULL);

    typedef struct {
        int layer_ordinal, pos;
        int n_heads, n_kv_heads, head_dim;
        float inv_sqrt_hd;
        int pad;
    } PC_AttnD;
    PC_AttnD pc = {lo, pos, nh, nkh, hd, 1.0f / sqrtf(hd), 0};
    vkCmdPushConstants(G.cmd_dev, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd_dev, (uint32_t)((nh + 255) / 256), 1, 1);
    vkEndCommandBuffer(G.cmd_dev);
    VkSubmitInfo si2 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd_dev};
    vkResetFences(G.dev, 1, &G.fence_dev);
    vkQueueSubmit(G.queue, 1, &si2, G.fence_dev);
    return 1;
}
int picolm_gpu_attention_prefill(float *xb_out, const float *q,
                                  int lo, int sp, int nt,
                                  int nh, int nkh, int hd,
                                  int msl, int device) {
    (void)xb_out; (void)q; (void)lo; (void)sp; (void)nt;
    (void)nh; (void)nkh; (void)hd; (void)msl; (void)device; return 0;
}
int picolm_gpu_attention_prefill_dev(float *xb_out_dev, const float *q_dev,
                                      int lo, int sp, int nt,
                                      int nh, int nkh, int hd,
                                      int msl, int device) {
    if (!G.shader_attn_prefill) { fprintf(stderr, "FATAL: [VK] attn_prefill dispatch with no shader\n"); abort(); }
    if (device != 0 || nt < 1) return 0;

    // KAVERI workaround: use BK/BV pipeline buffers (F32) instead of
    // KV cache (F16) for prefill attention. KAVERI/RADV can't read
    // F16 data from large storage buffers in compute shaders.
    const float *bk_dev = (float*)G.pipe_k_b_d;
    const float *bv_dev = (float*)G.pipe_v_b_d;

    VkDescriptorBufferInfo qbi = desc_buf_info(q_dev);
    VkDescriptorBufferInfo obbi = desc_buf_info(xb_out_dev);
    VkDescriptorBufferInfo kbi = desc_buf_info(bk_dev);
    VkDescriptorBufferInfo vbi = desc_buf_info(bv_dev);
    if (!qbi.buffer || !obbi.buffer || !kbi.buffer || !vbi.buffer) return 0;
    qbi.range = (VkDeviceSize)nt * nh * hd * sizeof(float);
    obbi.range = (VkDeviceSize)nt * nh * hd * sizeof(float);
    kbi.range = (VkDeviceSize)nt * nkh * hd * sizeof(float);
    vbi.range = (VkDeviceSize)nt * nkh * hd * sizeof(float);

    VkDescriptorBufferInfo bi[4] = {
        qbi, kbi, vbi, obbi
    };
    wr_desc(G.dset_attn_prefill, 4, bi);

    uint32_t total = (uint32_t)nh * (uint32_t)nt;
    VK_BATCH_DISPATCH_PRE();

    // Barrier: BK/BV/Q writes (SHADER_WRITE) -> shader reads
    VkBufferMemoryBarrier barriers[3] = {
        { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = qbi.buffer, .offset = qbi.offset, .size = qbi.range },
        { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = kbi.buffer, .offset = kbi.offset, .size = kbi.range },
        { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = vbi.buffer, .offset = vbi.offset, .size = vbi.range }
    };
    // KAVERI workaround: use ALL_COMMANDS stages for barrier
    vkCmdPipelineBarrier(G.cmd_dev, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, NULL, 3, barriers, 0, NULL);

    vkCmdBindPipeline(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_attn_prefill);
    vkCmdBindDescriptorSets(G.cmd_dev, VK_PIPELINE_BIND_POINT_COMPUTE,
                            G.plyt_unified, 0, 1, &G.dset_attn_prefill, 0, NULL);

    typedef struct {
        int layer_ordinal, start_pos, n_tokens;
        int n_heads, n_kv_heads, head_dim;
        float inv_sqrt_hd;  // 1/sqrt(head_dim)
        int pad;
    } PC_Attn;
    PC_Attn pc = {lo, sp, nt, nh, nkh, hd, 1.0f / sqrtf(hd), 0};
    vkCmdPushConstants(G.cmd_dev, G.plyt_unified, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd_dev, (uint32_t)((total + 255) / 256), 1, 1);
    VK_BATCH_DISPATCH_POST();
    return 1;
}
int picolm_gpu_attention_prefill_f32kv(float *xb_out_dev, const float *q_dev,
                                        const float *k_dev, const float *v_dev,
                                        int sp, int nt, int nh, int nkh,
                                        int hd, int device) {
    (void)xb_out_dev; (void)q_dev; (void)k_dev; (void)v_dev;
    (void)sp; (void)nt; (void)nh; (void)nkh; (void)hd; (void)device; return 0;
}

// ---------------------------------------------------------------------------
// SSM FUNCTIONS (all stubs - SSM models not yet supported on Vulkan)
// ---------------------------------------------------------------------------

int picolm_gpu_ssm_vecdot(float *o, const float *x, const void *w,
                           gguf_type_t q, int d, int nv, int rb,
                           const int *hm, int device) {
    (void)o;(void)x;(void)w;(void)q;(void)d;(void)nv;(void)rb;(void)hm;(void)device; return 0;
}
int picolm_gpu_ssm_vecdot_batch(float *o, const float *x, const void *w,
                                 gguf_type_t q, int d, int nv, int nt,
                                 int rb, const int *hm, int device) {
    (void)o;(void)x;(void)w;(void)q;(void)d;(void)nv;(void)nt;(void)rb;(void)hm;(void)device; return 0;
}
int picolm_gpu_ssm_vecdot_dev(float *o, const float *x, const void *w,
                               gguf_type_t q, int d, int nv, int rb,
                               const int *hm, int device) {
    (void)o;(void)x;(void)w;(void)q;(void)d;(void)nv;(void)rb;(void)hm;(void)device; return 0;
}
int picolm_gpu_ssm_vecdot_batch_dev(float *o, const float *x, const void *w,
                                     gguf_type_t q, int d, int nv, int nt,
                                     int rb, const int *hm, int device,
                                     int is, int os) {
    (void)o;(void)x;(void)w;(void)q;(void)d;(void)nv;(void)nt;(void)rb;(void)hm;(void)device;(void)is;(void)os; return 0;
}
int picolm_gpu_ssm_gate_beta_dev(float *ge, float *bo, const float *ai, const float *bi,
                                  const float *aw, const float *dw, int nv, int device) {
    (void)ge;(void)bo;(void)ai;(void)bi;(void)aw;(void)dw;(void)nv;(void)device; return 0;
}
int picolm_gpu_ssm_gate_beta_batch_dev(float *ge, float *bo, const float *ai, const float *bi,
                                        const float *dw, const float *aw, int nv, int nt,
                                        int device, int stride) {
    (void)ge;(void)bo;(void)ai;(void)bi;(void)dw;(void)aw;(void)nv;(void)nt;(void)device;(void)stride; return 0;
}
int picolm_gpu_ssm_l2norm_dev(float *x, int hd, int nh, float e, float es, int device) {
    (void)x;(void)hd;(void)nh;(void)e;(void)es;(void)device; return 0;
}
int picolm_gpu_ssm_l2norm_batch(float *x, int hd, int nh, int nt, int ts,
                                 float e, float es, int device) {
    (void)x;(void)hd;(void)nh;(void)nt;(void)ts;(void)e;(void)es;(void)device; return 0;
}
int picolm_gpu_ssm_l2norm_batch_dev(float *x, int hd, int nh, int nt, int ts,
                                     float e, float es, int device) {
    (void)x;(void)hd;(void)nh;(void)nt;(void)ts;(void)e;(void)es;(void)device; return 0;
}
int picolm_gpu_ssm_head_permute_dev(float *d, const float *s, const int *h,
                                     int hd, int nh, int device) {
    (void)d;(void)s;(void)h;(void)hd;(void)nh;(void)device; return 0;
}
int picolm_gpu_ssm_head_permute_batch_dev(float *d, const float *s, const int *h,
                                           int hd, int nh, int nt, int ss, int ds, int device) {
    (void)d;(void)s;(void)h;(void)hd;(void)nh;(void)nt;(void)ss;(void)ds;(void)device; return 0;
}
int picolm_gpu_ssm_recurrence(float *st, const float *q, const float *k, const float *v,
                               const float *ge, const float *b, float *so,
                               int nv, int ds, int rep, int device) {
    (void)st;(void)q;(void)k;(void)v;(void)ge;(void)b;(void)so;(void)nv;(void)ds;(void)rep;(void)device; return 0;
}
int picolm_gpu_ssm_recurrence_dev(void *st, const float *q, const float *k, const float *v,
                                   const float *ge, const float *b, float *so,
                                   int nv, int ds, int rep, int device) {
    (void)st;(void)q;(void)k;(void)v;(void)ge;(void)b;(void)so;(void)nv;(void)ds;(void)rep;(void)device; return 0;
}
int picolm_gpu_ssm_recurrence_pipeline_dev(void *st, const float *q, const float *k,
                                            const float *v, const float *ge, const float *b,
                                            float *so, int nv, int ds, int rep, int device) {
    (void)st;(void)q;(void)k;(void)v;(void)ge;(void)b;(void)so;(void)nv;(void)ds;(void)rep;(void)device; return 0;
}
int picolm_gpu_ssm_chunked_recurrence_dev(const float *c, const float *a, const float *b,
                                           float *st, float *x, int nt, int vd, int xs,
                                           int ds, int nk, int nv, int hv, int rep,
                                           int cd, int cs, int device) {
    (void)c;(void)a;(void)b;(void)st;(void)x;(void)nt;(void)vd;(void)xs;(void)ds;
    (void)nk;(void)nv;(void)hv;(void)rep;(void)cd;(void)cs;(void)device; return 0;
}
int picolm_gpu_ssm_chunked_recurrence(const float *c, const float *a, const float *b,
                                       float *st, float *x, int nt, int vd, int ds,
                                       int nk, int nv, int hv, int rep, int cd,
                                       int cs, int device) {
    (void)c;(void)a;(void)b;(void)st;(void)x;(void)nt;(void)vd;(void)ds;(void)nk;
    (void)nv;(void)hv;(void)rep;(void)cd;(void)cs;(void)device; return 0;
}
int picolm_gpu_ssm_conv1d_dev(float *co, float *cs, const float *ni,
                               const float *w, int cd, int dc, int device) {
    (void)co;(void)cs;(void)ni;(void)w;(void)cd;(void)dc;(void)device; return 0;
}
int picolm_gpu_ssm_conv1d_batch(float *co, float *cs, const float *ni,
                                 const float *w, int cd, int dc, int nt, int device) {
    (void)co;(void)cs;(void)ni;(void)w;(void)cd;(void)dc;(void)nt;(void)device; return 0;
}
int picolm_gpu_ssm_conv1d_batch_dev(float *o, float *s, const float *i,
                                     const float *w, int cd, int dc, int nt,
                                     int device, int stride) {
    (void)o;(void)s;(void)i;(void)w;(void)cd;(void)dc;(void)nt;(void)device;(void)stride; return 0;
}
int picolm_gpu_ssm_gated_norm_dev(float *fo, const float *so, const float *xb,
                                   const float *nw, const int *hm, int hvd, int nv,
                                   float eps, int device) {
    (void)fo;(void)so;(void)xb;(void)nw;(void)hm;(void)hvd;(void)nv;(void)eps;(void)device; return 0;
}
int picolm_gpu_ssm_gated_norm(float *fo, const float *so, const float *xb,
                               const float *nw, const int *hm, int hvd, int nv,
                               float eps, int device) {
    (void)fo;(void)so;(void)xb;(void)nw;(void)hm;(void)hvd;(void)nv;(void)eps;(void)device; return 0;
}
int picolm_gpu_ssm_gated_norm_batch(float *fo, const float *so, const float *xb,
                                     const float *nw, const int *hm, int hvd, int nv,
                                     int nt, float eps, int device) {
    (void)fo;(void)so;(void)xb;(void)nw;(void)hm;(void)hvd;(void)nv;(void)nt;(void)eps;(void)device; return 0;
}
int picolm_gpu_ssm_prefill_gated_norm(float *o, const float *z, const float *nw,
                                       int hvd, int nv, int nt, float eps, int device) {
    (void)o;(void)z;(void)nw;(void)hvd;(void)nv;(void)nt;(void)eps;(void)device; return 0;
}
int picolm_gpu_ssm_prefill_gated_norm_dev(float *o, const float *z, const float *nw,
                                           int hvd, int nv, int nt, float eps,
                                           int device, int os, int zs) {
    (void)o;(void)z;(void)nw;(void)hvd;(void)nv;(void)nt;(void)eps;(void)device;(void)os;(void)zs; return 0;
}

// ---------------------------------------------------------------------------
// QG DEINTERLEAVE + SIGMOID MUL (stubs)
// ---------------------------------------------------------------------------

int picolm_gpu_qg_deinterleave_dev(const float *r, float *q, float *g,
                                    int nh, int hd, int device) {
    (void)r;(void)q;(void)g;(void)nh;(void)hd;(void)device; return 0;
}
int picolm_gpu_sigmoid_mul_dev(float *o, const float *g, int n, int device) {
    (void)o;(void)g;(void)n;(void)device; return 0;
}
int picolm_gpu_qg_deinterleave_batched_dev(const float *r, float *q, float *g,
                                            int nh, int hd, int S, int device) {
    (void)r;(void)q;(void)g;(void)nh;(void)hd;(void)S;(void)device; return 0;
}
int picolm_gpu_sigmoid_mul_batched_dev(float *o, const float *g,
                                        int n, int S, int device) {
    (void)o;(void)g;(void)n;(void)S;(void)device; return 0;
}

// ---------------------------------------------------------------------------
// EXTERNAL STUBS (declared in model_core.c / model_ssm.c)
// ---------------------------------------------------------------------------

int picolm_gpu_rmsnorm_matmul_dev_qkv(picolm_gpu_tensor_t *tq,
                                       picolm_gpu_tensor_t *tk,
                                       picolm_gpu_tensor_t *tv,
                                       float *bq, float *bk, float *bv,
                                       const float *bx, const float *rw,
                                       int dim, float eps, int S,
                                       int xs, int device,
                                       int ysq, int ysk) {
    (void)tq;(void)tk;(void)tv;(void)bq;(void)bk;(void)bv;
    (void)bx;(void)rw;(void)dim;(void)eps;(void)S;
    (void)xs;(void)device;(void)ysq;(void)ysk; return 0;
}
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// MISSING STUBS (functions added in commits after 79c9912)
// ---------------------------------------------------------------------------

float *picolm_gpu_pipe_logits(int device) {
    if (device != 0 || !G.pipe_logits_d) return NULL;
    return (float*)G.pipe_logits_d;
}

int picolm_gpu_matmul_logits(picolm_gpu_tensor_t *t, float *logits_dev,
                              const float *x_dev, int device) {
    if (!G.ready || !t || !logits_dev || !x_dev || device != 0) return 0;
    if (!G.pipe_logits_d) return 0; /* should be pre-allocated */
    if (logits_dev != (float*)G.pipe_logits_d) return 0;
    return picolm_gpu_matmul_dev(t, logits_dev, x_dev, 1, device, 0, 0);
}
