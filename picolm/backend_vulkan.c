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
    fprintf(stderr, "[VK] %s:%d %s failed: %d\n", __FILE__, __LINE__, what, (int)_r); \
    return 0; } } while (0)

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

    VkDescriptorSetLayout dsl;
    VkPipelineLayout plyt;
    VkPipeline pipe;
    VkShaderModule shader;
    VkDescriptorPool dpool;
    VkDescriptorSet dset;

    // RMSNorm pipeline
    VkShaderModule shader_nrm;
    VkDescriptorSetLayout dsl_nrm;
    VkPipelineLayout plyt_nrm;
    VkPipeline pipe_nrm;
    VkDescriptorPool dpool_nrm;
    VkDescriptorSet dset_nrm;

    VkCommandPool cpool;
    VkCommandBuffer cmd;
    VkCommandBuffer cmd_nrm;
    VkFence fence;

    Scratch x_buf, y_buf;

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

    char device_name[256];
    VkPhysicalDeviceMemoryProperties mem_props;
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
        fprintf(stderr, "[VK] alloc_hostvis_mt: mapped %zu bytes at %p (memtype=%u)\n", bytes, *ptr, memtype);
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
    fprintf(stderr, "[VK] arena_suballoc bytes=%zu memtype=%u\n", bytes, G.memtype);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer");
    fprintf(stderr, "[VK]   buffer created ok\n");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    fprintf(stderr, "[VK]   req.size=%zu req.align=%zu memTypeBits=%u\n",
            (size_t)req.size, (size_t)req.alignment, (unsigned)req.memoryTypeBits);
    if (!(req.memoryTypeBits & (1u << G.memtype))) {
        fprintf(stderr, "[VK]   ERROR: memtype %u not supported\n", G.memtype);
        vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0;
    }
    VkDeviceMemory mem;
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = G.memtype};
    VKCHECK(vkAllocateMemory(G.dev, &ai, NULL, &mem), "vkAllocateMemory");
    fprintf(stderr, "[VK]   memory allocated ok\n");
    VKCHECK(vkBindBufferMemory(G.dev, *buf, mem, 0), "vkBindBufferMemory");
    fprintf(stderr, "[VK]   buffer bound ok\n");
    // Track this allocation in a simple list so shutdown can free it
    VkWArena *a = calloc(1, sizeof(*a));
    if (!a) {
        vkFreeMemory(G.dev, mem, NULL);
        vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0;
    }
    a->mem = mem;
    a->cap = req.size;
    if (ptr) {
        VKCHECK(vkMapMemory(G.dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&a->base), "vkMapMemory");
        fprintf(stderr, "[VK]   memory mapped: ptr=%p\n", a->base);
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

static int build_pipeline(VkDevice dev, int nbind, size_t pc_size, VkShaderModule shader,
                          VkDescriptorSetLayout *dsl, VkPipelineLayout *plyt, VkPipeline *pipe,
                          VkDescriptorPool *dpool, VkDescriptorSet *dset) {
    VkDescriptorSetLayoutBinding b[8];
    for (int i = 0; i < nbind; i++) b[i] = (VkDescriptorSetLayoutBinding){
        .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dsli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = (uint32_t)nbind, .pBindings = b};
    VKCHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, dsl), "descSetLayout");
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = (uint32_t)pc_size};
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VKCHECK(vkCreatePipelineLayout(dev, &pli, NULL, plyt), "pipelineLayout");
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = *plyt};
    VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, pipe), "pipeline");
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = (uint32_t)nbind};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(dev, &dpi, NULL, dpool), "descPool");
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = *dpool, .descriptorSetCount = 1, .pSetLayouts = dsl};
    VKCHECK(vkAllocateDescriptorSets(dev, &dsa, dset), "allocDescSet");
    return 1;
}

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
    fprintf(stderr, "[VK] limits: maxSharedMem=%u maxWorkGroupSize=%u maxComputeWorkGroupCount[0]=%u\n",
            (unsigned)pdev.limits.maxComputeSharedMemorySize,
            (unsigned)pdev.limits.maxComputeWorkGroupSize,
            (unsigned)pdev.limits.maxComputeWorkGroupCount[0]);

    int mt = pick_memtype(G.phys);
    if (mt < 0) { fprintf(stderr, "[VK] no host-visible memory\n"); return 0; }
    G.memtype = (uint32_t)mt;
    G.memtype_cached = (uint32_t)pick_memtype_cached(G.phys);
    vkGetPhysicalDeviceMemoryProperties(G.phys, &G.mem_props);

    // Load matmul shader
    G.shader = load_spv(G.dev, "qmatmul_vk.spv");
    if (!G.shader) { fprintf(stderr, "[VK] failed to load qmatmul_vk.spv\n"); return 0; }
    // Push constants: int fmt, int S, int I, int O, int rowWords = 5*4 = 20 bytes
    if (!build_pipeline(G.dev, 3, 20, G.shader, &G.dsl, &G.plyt, &G.pipe,
                        &G.dpool, &G.dset)) return 0;

    // Optional: RMSNorm shader
    G.shader_nrm = load_spv(G.dev, "rmsnorm_vk.spv");
    if (G.shader_nrm) {
        if (!build_pipeline(G.dev, 3, 20, G.shader_nrm, &G.dsl_nrm, &G.plyt_nrm,
                            &G.pipe_nrm, &G.dpool_nrm, &G.dset_nrm)) {
            G.shader_nrm = VK_NULL_HANDLE;
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

    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.fence), "fence");

    G.ready = 1;
    G.device_count = 1;
    G.devices[0] = 0;

    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(G.phys, &p);
    strncpy(G.device_name, p.deviceName, sizeof(G.device_name) - 1);
    G.device_name[sizeof(G.device_name) - 1] = '\0';

    fprintf(stderr, "[VK] ready: %s, compute qfam %u, memtype %u%s\n",
            G.device_name, G.qfam, G.memtype,
            G.shader_nrm ? ", rmsnorm" : "");
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
    if (G.dsl)   vkDestroyDescriptorSetLayout(G.dev, G.dsl, NULL);
    if (G.pipe)  vkDestroyPipeline(G.dev, G.pipe, NULL);
    if (G.plyt)  vkDestroyPipelineLayout(G.dev, G.plyt, NULL);
    if (G.shader) vkDestroyShaderModule(G.dev, G.shader, NULL);

    if (G.dset_nrm)  vkFreeDescriptorSets(G.dev, G.dpool_nrm, 1, &G.dset_nrm);
    if (G.dpool_nrm) vkDestroyDescriptorPool(G.dev, G.dpool_nrm, NULL);
    if (G.dsl_nrm)   vkDestroyDescriptorSetLayout(G.dev, G.dsl_nrm, NULL);
    if (G.pipe_nrm)  vkDestroyPipeline(G.dev, G.pipe_nrm, NULL);
    if (G.plyt_nrm)  vkDestroyPipelineLayout(G.dev, G.plyt_nrm, NULL);
    if (G.shader_nrm) vkDestroyShaderModule(G.dev, G.shader_nrm, NULL);

    if (G.cmd)   vkFreeCommandBuffers(G.dev, G.cpool, 1, &G.cmd);
    if (G.cmd_nrm) vkFreeCommandBuffers(G.dev, G.cpool, 1, &G.cmd_nrm);
    if (G.cpool) vkDestroyCommandPool(G.dev, G.cpool, NULL);
    if (G.fence) vkDestroyFence(G.dev, G.fence, NULL);

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
    fprintf(stderr, "[VK] upload: qtype=%d I=%d O=%d rb=%zu gpu_stride=%zu total=%zu weights=%p\n",
            qtype, I, O, rb, gpu_stride, total, (const void *)weights);

    void *wptr;
    VkBuffer wbuf;
    if (!arena_suballoc(total, &wbuf, &wptr)) {
        fprintf(stderr, "[VK] upload: arena_suballoc FAILED\n");
        return 0;
    }
    fprintf(stderr, "[VK] upload: writing %d rows...\n", O);

    for (int o = 0; o < O; o++) {
        memcpy((uint8_t *)wptr + (size_t)o * gpu_stride,
               (const uint8_t *)weights + (size_t)o * rb, rb);
    }
    fprintf(stderr, "[VK] upload: memcpy done\n");

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
//   int fmt, int S, int I, int O, int rowWords
// Must match shader push_constant layout (20 bytes)
typedef struct {
    int fmt;       // GGUF_TYPE enum value
    int S, I, O;
    int rowWords;  // uint32 words per weight row
} PC_Matmul;

int picolm_gpu_matmul(picolm_gpu_tensor_t *t, float *y, const float *x,
                       int S, int device) {
    if (!G.ready || !t || device != 0 || S < 1) {
        if (G.ready && t && S >= 1) fprintf(stderr, "[VK] matmul skip: dev=%d\n", device);
        return 0;
    }

    size_t xb = (size_t)S * t->I * sizeof(float);
    size_t yb = (size_t)S * t->O * sizeof(float);

    if (!scratch_reserve(&G.x_buf, xb)) { fprintf(stderr, "[VK] matmul: x scratch failed %zu\n", xb); return 0; }
    if (!scratch_reserve_mt(&G.y_buf, yb, G.memtype_cached)) { fprintf(stderr, "[VK] matmul: y scratch failed %zu\n", yb); return 0; }

    static int _vk_xdbg = 0;
    if (_vk_xdbg++ < 2) {
        fprintf(stderr, "[VK] matmul xdbg: before memcpy x[0..2]=");
        for (int i = 0; i < 3 && i < t->I; i++) fprintf(stderr, " %f", x[i]);
        fprintf(stderr, "\n");
    }
    memcpy(G.x_buf.ptr, x, xb);

    int rebind = G.bound_tensor != t || G.x_buf.buf != G.bound_xbuf ||
                 G.y_buf.buf != G.bound_ybuf;
    if (rebind) {
        VkDescriptorBufferInfo bi[3] = {
            {G.x_buf.buf, 0, VK_WHOLE_SIZE},
            {t->wbuf, 0, VK_WHOLE_SIZE},
            {G.y_buf.buf, 0, VK_WHOLE_SIZE}
        };
        wr_desc(G.dset, 3, bi);
        G.bound_tensor = t; G.bound_xbuf = G.x_buf.buf; G.bound_ybuf = G.y_buf.buf;
    }

    int reshape = rebind || !G.cmd_ready || G.bound_S != S ||
                  G.bound_I != t->I || G.bound_O != t->O;
    if (reshape) {
        VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
        vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                G.plyt, 0, 1, &G.dset, 0, NULL);
        PC_Matmul pc = {t->qtype, S, t->I, t->O, (int)t->row_words};
        vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        // One workgroup per output row per sequence
        vkCmdDispatch(G.cmd, (uint32_t)t->O, (uint32_t)S, 1);
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
// RMSNorm: host-facing
// ---------------------------------------------------------------------------

typedef struct {
    int S, D;
    float eps;
    int pad; // alignment padding to 20 bytes
} PC_RmsNorm;

int picolm_gpu_rmsnorm(float *out, const float *x, const float *weight,
                        int dim, float eps, int device) {
    return picolm_gpu_rmsnorm_batched(out, x, weight, dim, eps, 1, dim, device);
}

int picolm_gpu_rmsnorm_batched(float *out, const float *x, const float *weight,
                                int dim, float eps, int S, int x_stride, int device) {
    if (!G.ready || !G.shader_nrm || device != 0 || S < 1) return 0;

    size_t xb = (size_t)S * x_stride * sizeof(float);
    size_t wb = (size_t)dim * sizeof(float);
    size_t yb = (size_t)S * x_stride * sizeof(float);

    // For simplicity, use temporary allocations (could be scratch-ified)
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
                            G.plyt_nrm, 0, 1, &G.dset_nrm, 0, NULL);
    PC_RmsNorm pc = {S, dim, eps, 0};
    vkCmdPushConstants(G.cmd_nrm, G.plyt_nrm, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
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
    VkBuffer buf; VkDeviceMemory mem;
    size_t bytes = n * sizeof(float);
    if (!alloc_hostvis(bytes, &buf, &mem, NULL)) return NULL;
    // We need a mapped ptr to copy; re-map
    void *ptr;
    if (vkMapMemory(G.dev, mem, 0, bytes, 0, &ptr) != VK_SUCCESS) {
        vkDestroyBuffer(G.dev, buf, NULL); vkFreeMemory(G.dev, mem, NULL);
        return NULL;
    }
    memcpy(ptr, host, bytes);
    // We don't unmap -- it stays mapped for the shader to read
    // Return a fake opaque pointer that we can't actually use from C
    // The caller uses this as a device pointer; for Vulkan we store the buffer handle
    // For simplicity, allocate a small struct and return it cast to float*
    typedef struct { VkBuffer b; VkDeviceMemory m; } f32_buf_t;
    f32_buf_t *fb = malloc(sizeof(*fb));
    if (!fb) {
        vkDestroyBuffer(G.dev, buf, NULL); vkFreeMemory(G.dev, mem, NULL);
        return NULL;
    }
    fb->b = buf; fb->m = mem;
    return (float *)fb;
}

// ---------------------------------------------------------------------------
// DEVICE MEMORY (stub)
// ---------------------------------------------------------------------------

void *picolm_gpu_alloc_device(size_t bytes, int device) {
    (void)bytes; (void)device;
    return NULL; // Phase 1: no device-native pipeline
}

int picolm_gpu_device_memset(void *dev_ptr, int value, size_t bytes, int device) {
    (void)dev_ptr; (void)value; (void)bytes; (void)device;
    return 0;
}

void *picolm_gpu_upload_int(const int *host, size_t n, int device) {
    (void)host; (void)n; (void)device;
    return NULL;
}

// ---------------------------------------------------------------------------
// STUBS: all functions not yet implemented return 0 (CPU fallback)
// ---------------------------------------------------------------------------

int picolm_gpu_matmul_dev(picolm_gpu_tensor_t *t, float *y_dev, const float *x_dev,
                           int S, int device, int y_stride, int x_stride) {
    (void)t; (void)y_dev; (void)x_dev; (void)S; (void)device;
    (void)y_stride; (void)x_stride; return 0;
}

int picolm_gpu_matmul_dev_strided(picolm_gpu_tensor_t *t, float *y_dev,
                                   const float *x_dev, int S, int device,
                                   int x_stride, int y_stride) {
    (void)t; (void)y_dev; (void)x_dev; (void)S; (void)device;
    (void)x_stride; (void)y_stride; return 0;
}

int picolm_gpu_matmul_dev_qkv(picolm_gpu_tensor_t *tq, picolm_gpu_tensor_t *tk,
                               picolm_gpu_tensor_t *tv, float *bq, float *bk, float *bv,
                               const float *x_dev, int S, int device,
                               int y_stride_q, int y_stride_kv, int x_stride) {
    (void)tq; (void)tk; (void)tv; (void)bq; (void)bk; (void)bv;
    (void)x_dev; (void)S; (void)device;
    (void)y_stride_q; (void)y_stride_kv; (void)x_stride; return 0;
}

int picolm_gpu_matmul_dev_gu(picolm_gpu_tensor_t *tg, picolm_gpu_tensor_t *tu,
                              float *bgate, float *bup,
                              const float *x_dev, int S, int device,
                              int y_stride, int x_stride) {
    (void)tg; (void)tu; (void)bgate; (void)bup;
    (void)x_dev; (void)S; (void)device;
    (void)y_stride; (void)x_stride; return 0;
}

int picolm_gpu_expert_mlp(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                           picolm_gpu_tensor_t *down, float *y, const float *x, int S) {
    (void)gate; (void)up; (void)down; (void)y; (void)x; (void)S; return 0;
}

int picolm_gpu_expert_mlp_dev(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                               picolm_gpu_tensor_t *down, float *y_dev,
                               const float *x_dev, int S, int x_stride,
                               int y_stride, int device) {
    (void)gate; (void)up; (void)down; (void)y_dev; (void)x_dev;
    (void)S; (void)x_stride; (void)y_stride; (void)device; return 0;
}

int picolm_gpu_w4a16_mlp(picolm_gpu_tensor_t *gate, picolm_gpu_tensor_t *up,
                          picolm_gpu_tensor_t *down, float *y, const float *x, int S) {
    (void)gate; (void)up; (void)down; (void)y; (void)x; (void)S; return 0;
}

int picolm_gpu_w4a16_matmul(picolm_gpu_tensor_t *t, float *y, const float *x,
                             int S, int device) {
    (void)t; (void)y; (void)x; (void)S; (void)device; return 0;
}

// SSM stubs
int picolm_gpu_ssm_vecdot(float *out, const float *x, const void *weights,
                           gguf_type_t qtype, int dim, int n_v_heads,
                           int row_bytes, const int *head_map, int device) {
    (void)out; (void)x; (void)weights; (void)qtype; (void)dim;
    (void)n_v_heads; (void)row_bytes; (void)head_map; (void)device; return 0;
}

int picolm_gpu_ssm_vecdot_batch(float *out, const float *x, const void *weights,
                                 gguf_type_t qtype, int dim, int n_v_heads,
                                 int n_tokens, int row_bytes,
                                 const int *head_map, int device) {
    (void)out; (void)x; (void)weights; (void)qtype; (void)dim;
    (void)n_v_heads; (void)n_tokens; (void)row_bytes;
    (void)head_map; (void)device; return 0;
}

int picolm_gpu_ssm_vecdot_dev(float *out_dev, const float *x_dev,
                               const void *weights_dev, gguf_type_t qtype,
                               int dim, int n_v_heads, int row_bytes,
                               const int *head_map_dev, int device) {
    (void)out_dev; (void)x_dev; (void)weights_dev; (void)qtype;
    (void)dim; (void)n_v_heads; (void)row_bytes;
    (void)head_map_dev; (void)device; return 0;
}

int picolm_gpu_ssm_vecdot_batch_dev(float *out, const float *x,
                                     const void *weights, gguf_type_t qtype,
                                     int dim, int n_v_heads, int n_tokens,
                                     int row_bytes, const int *head_map,
                                     int device, int in_stride, int out_stride) {
    (void)out; (void)x; (void)weights; (void)qtype; (void)dim;
    (void)n_v_heads; (void)n_tokens; (void)row_bytes;
    (void)head_map; (void)device; (void)in_stride; (void)out_stride; return 0;
}

int picolm_gpu_ssm_gate_beta_dev(float *gate_exp_out_dev, float *beta_out_dev,
                                  const float *alpha_in_dev, const float *beta_raw_in_dev,
                                  const float *ssm_a_w_dev, const float *ssm_dt_w_dev,
                                  int n_v_heads, int device) {
    (void)gate_exp_out_dev; (void)beta_out_dev; (void)alpha_in_dev;
    (void)beta_raw_in_dev; (void)ssm_a_w_dev; (void)ssm_dt_w_dev;
    (void)n_v_heads; (void)device; return 0;
}

int picolm_gpu_ssm_gate_beta_batch_dev(float *ge, float *bo,
                                        const float *ai, const float *bi,
                                        const float *dt_w, const float *a_w,
                                        int n_v_heads, int n_tokens,
                                        int device, int stride) {
    (void)ge; (void)bo; (void)ai; (void)bi; (void)dt_w; (void)a_w;
    (void)n_v_heads; (void)n_tokens; (void)device; (void)stride; return 0;
}

int picolm_gpu_ssm_l2norm_dev(float *x_dev, int head_dim, int n_heads,
                               float eps, float extra_scale, int device) {
    (void)x_dev; (void)head_dim; (void)n_heads;
    (void)eps; (void)extra_scale; (void)device; return 0;
}

int picolm_gpu_ssm_l2norm_batch(float *x_host, int head_dim, int n_heads,
                                 int n_tokens, int token_stride,
                                 float eps, float extra_scale, int device) {
    (void)x_host; (void)head_dim; (void)n_heads; (void)n_tokens;
    (void)token_stride; (void)eps; (void)extra_scale; (void)device; return 0;
}

int picolm_gpu_ssm_l2norm_batch_dev(float *x, int head_dim, int n_heads,
                                     int n_tokens, int token_stride,
                                     float eps, float extra_scale, int device) {
    (void)x; (void)head_dim; (void)n_heads; (void)n_tokens;
    (void)token_stride; (void)eps; (void)extra_scale; (void)device; return 0;
}

int picolm_gpu_ssm_head_permute_dev(float *dst_dev, const float *src_dev,
                                     const int *head_map_dev,
                                     int head_dim, int n_heads, int device) {
    (void)dst_dev; (void)src_dev; (void)head_map_dev;
    (void)head_dim; (void)n_heads; (void)device; return 0;
}

int picolm_gpu_ssm_head_permute_batch_dev(float *dst, const float *src,
                                           const int *head_map,
                                           int head_dim, int n_heads,
                                           int n_tokens, int src_stride,
                                           int dst_stride, int device) {
    (void)dst; (void)src; (void)head_map; (void)head_dim;
    (void)n_heads; (void)n_tokens; (void)src_stride;
    (void)dst_stride; (void)device; return 0;
}

int picolm_gpu_ssm_recurrence(float *state, const float *q_conv,
                               const float *k_conv, const float *v_conv,
                               const float *gate_exp, const float *beta,
                               float *ssm_output, int n_v_heads, int d_state,
                               int repeat, int device) {
    (void)state; (void)q_conv; (void)k_conv; (void)v_conv;
    (void)gate_exp; (void)beta; (void)ssm_output;
    (void)n_v_heads; (void)d_state; (void)repeat; (void)device; return 0;
}

int picolm_gpu_ssm_recurrence_dev(void *ssm_state_dev, const float *q_conv,
                                   const float *k_conv, const float *v_conv,
                                   const float *gate_exp, const float *beta,
                                   float *ssm_output, int n_v_heads,
                                   int d_state, int repeat, int device) {
    (void)ssm_state_dev; (void)q_conv; (void)k_conv; (void)v_conv;
    (void)gate_exp; (void)beta; (void)ssm_output;
    (void)n_v_heads; (void)d_state; (void)repeat; (void)device; return 0;
}

int picolm_gpu_ssm_recurrence_pipeline_dev(void *ssm_state_dev,
                                            const float *q_conv_dev,
                                            const float *k_conv_dev,
                                            const float *v_conv_dev,
                                            const float *gate_exp_dev,
                                            const float *beta_dev,
                                            float *ssm_output_dev,
                                            int n_v_heads, int d_state,
                                            int repeat, int device) {
    (void)ssm_state_dev; (void)q_conv_dev; (void)k_conv_dev;
    (void)v_conv_dev; (void)gate_exp_dev; (void)beta_dev;
    (void)ssm_output_dev; (void)n_v_heads; (void)d_state;
    (void)repeat; (void)device; return 0;
}

int picolm_gpu_ssm_chunked_recurrence_dev(const float *conv, const float *alpha,
                                           const float *beta, float *state,
                                           float *xb2, int n_tokens,
                                           int value_dim, int xb2_stride,
                                           int d_state, int n_k_heads,
                                           int n_v_heads, int head_v_dim,
                                           int repeat, int conv_dim,
                                           int cs, int device) {
    (void)conv; (void)alpha; (void)beta; (void)state; (void)xb2;
    (void)n_tokens; (void)value_dim; (void)xb2_stride;
    (void)d_state; (void)n_k_heads; (void)n_v_heads;
    (void)head_v_dim; (void)repeat; (void)conv_dim;
    (void)cs; (void)device; return 0;
}

int picolm_gpu_ssm_chunked_recurrence(const float *conv_batch_host,
                                       const float *alpha_batch_host,
                                       const float *beta_batch_host,
                                       float *state_host, float *xb2_batch_host,
                                       int n_tokens, int value_dim,
                                       int d_state, int n_k_heads,
                                       int n_v_heads, int head_v_dim,
                                       int repeat, int conv_dim,
                                       int cs, int device) {
    (void)conv_batch_host; (void)alpha_batch_host; (void)beta_batch_host;
    (void)state_host; (void)xb2_batch_host; (void)n_tokens;
    (void)value_dim; (void)d_state; (void)n_k_heads;
    (void)n_v_heads; (void)head_v_dim; (void)repeat;
    (void)conv_dim; (void)cs; (void)device; return 0;
}

int picolm_gpu_ssm_conv1d_dev(float *co, float *cs, const float *ni,
                               const float *w, int cd, int dc, int device) {
    (void)co; (void)cs; (void)ni; (void)w; (void)cd; (void)dc; (void)device; return 0;
}

int picolm_gpu_ssm_conv1d_batch(float *co, float *cs, const float *ni,
                                 const float *w, int cd, int dc,
                                 int nt, int device) {
    (void)co; (void)cs; (void)ni; (void)w; (void)cd; (void)dc; (void)nt; (void)device; return 0;
}

int picolm_gpu_ssm_conv1d_batch_dev(float *out, float *state, const float *inp,
                                     const float *w, int cd, int dc,
                                     int nt, int device, int stride) {
    (void)out; (void)state; (void)inp; (void)w; (void)cd; (void)dc;
    (void)nt; (void)device; (void)stride; return 0;
}

int picolm_gpu_ssm_gated_norm_dev(float *fo, const float *so,
                                   const float *xb2, const float *nw,
                                   const int *hm, int hvd, int nv,
                                   float eps, int device) {
    (void)fo; (void)so; (void)xb2; (void)nw; (void)hm;
    (void)hvd; (void)nv; (void)eps; (void)device; return 0;
}

int picolm_gpu_ssm_gated_norm(float *fo, const float *so,
                               const float *xb2, const float *nw,
                               const int *hm, int hvd, int nv,
                               float eps, int device) {
    (void)fo; (void)so; (void)xb2; (void)nw; (void)hm;
    (void)hvd; (void)nv; (void)eps; (void)device; return 0;
}

int picolm_gpu_ssm_gated_norm_batch(float *fo, const float *so,
                                     const float *xb2, const float *nw,
                                     const int *hm, int hvd, int nv,
                                     int nt, float eps, int device) {
    (void)fo; (void)so; (void)xb2; (void)nw; (void)hm;
    (void)hvd; (void)nv; (void)nt; (void)eps; (void)device; return 0;
}

int picolm_gpu_ssm_prefill_gated_norm(float *ssm_out_host, const float *z_host,
                                       const float *norm_w_host,
                                       int hvd, int nv, int nt,
                                       float eps, int device) {
    (void)ssm_out_host; (void)z_host; (void)norm_w_host;
    (void)hvd; (void)nv; (void)nt; (void)eps; (void)device; return 0;
}

int picolm_gpu_ssm_prefill_gated_norm_dev(float *ssm_out, const float *z,
                                           const float *norm_w,
                                           int hvd, int nv, int nt,
                                           float eps, int device,
                                           int so_stride, int z_stride) {
    (void)ssm_out; (void)z; (void)norm_w; (void)hvd;
    (void)nv; (void)nt; (void)eps; (void)device;
    (void)so_stride; (void)z_stride; return 0;
}

// KV cache stubs
int picolm_gpu_kv_alloc(size_t kv_k_bytes, size_t kv_v_bytes, int device) {
    (void)kv_k_bytes; (void)kv_v_bytes; (void)device; return 0;
}

int picolm_gpu_kv_store_rows(int is_k, int lo, int sp, int np,
                              const void *hr, size_t rb,
                              int nkh, int hd, int msl, int device) {
    (void)is_k; (void)lo; (void)sp; (void)np; (void)hr;
    (void)rb; (void)nkh; (void)hd; (void)msl; (void)device; return 0;
}

int picolm_gpu_kv_store_dev(int is_k, int lo, int pos,
                             const float *sd, int nkh, int hd,
                             int msl, int device) {
    (void)is_k; (void)lo; (void)pos; (void)sd;
    (void)nkh; (void)hd; (void)msl; (void)device; return 0;
}

int picolm_gpu_kv_store_dev_batched(int is_k, int lo, int sp, int np,
                                     const float *sd, int nkh, int hd,
                                     int msl, int device) {
    (void)is_k; (void)lo; (void)sp; (void)np; (void)sd;
    (void)nkh; (void)hd; (void)msl; (void)device; return 0;
}

int picolm_gpu_kv_store_dev_batched_strided(int is_k, int lo, int sp, int np,
                                             const float *sd, int nkh,
                                             int hd, int msl, int device,
                                             int src_stride) {
    (void)is_k; (void)lo; (void)sp; (void)np; (void)sd;
    (void)nkh; (void)hd; (void)msl; (void)device; (void)src_stride; return 0;
}

int picolm_gpu_kv_upload_layer(int is_k, int lo, int np,
                                const uint16_t *hr, int nkh,
                                int hd, int msl, int device) {
    (void)is_k; (void)lo; (void)np; (void)hr;
    (void)nkh; (void)hd; (void)msl; (void)device; return 0;
}

int picolm_gpu_kv_debug_dump(int is_k, int lo, int pos,
                              uint16_t *dst, int ne, int nkh,
                              int hd, int msl, int device) {
    (void)is_k; (void)lo; (void)pos; (void)dst; (void)ne;
    (void)nkh; (void)hd; (void)msl; (void)device; return 0;
}

void picolm_gpu_kv_cache_clear(int device) { (void)device; }
void picolm_gpu_kv_free(void) { }

// Attention stubs
int picolm_gpu_attention_decode(float *xb_out, const float *q,
                                 int lo, int pos, int nh, int nkh,
                                 int hd, int msl, int device) {
    (void)xb_out; (void)q; (void)lo; (void)pos;
    (void)nh; (void)nkh; (void)hd; (void)msl; (void)device; return 0;
}

int picolm_gpu_attention_decode_dev(float *xb_out_dev, const float *q_dev,
                                     int lo, int pos, int nh, int nkh,
                                     int hd, int msl, int device) {
    (void)xb_out_dev; (void)q_dev; (void)lo; (void)pos;
    (void)nh; (void)nkh; (void)hd; (void)msl; (void)device; return 0;
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
    (void)xb_out_dev; (void)q_dev; (void)lo; (void)sp; (void)nt;
    (void)nh; (void)nkh; (void)hd; (void)msl; (void)device; return 0;
}

int picolm_gpu_attention_prefill_f32kv(float *xb_out_dev, const float *q_dev,
                                        const float *k_dev, const float *v_dev,
                                        int sp, int nt, int nh, int nkh,
                                        int hd, int device) {
    (void)xb_out_dev; (void)q_dev; (void)k_dev; (void)v_dev;
    (void)sp; (void)nt; (void)nh; (void)nkh; (void)hd; (void)device; return 0;
}

// Pipeline stubs
int picolm_gpu_pipeline_alloc(int dim, int q_dim, int kv_dim,
                               int ffn_hidden, int device) {
    (void)dim; (void)q_dim; (void)kv_dim; (void)ffn_hidden; (void)device; return 0;
}

int picolm_gpu_ssm_pipeline_alloc(int cd, int ssi, int nv, int device) {
    (void)cd; (void)ssi; (void)nv; (void)device; return 0;
}

void picolm_gpu_pipeline_free(void) { }

int picolm_gpu_prealloc_q8(size_t mxq, size_t mxd, int device) {
    (void)mxq; (void)mxd; (void)device; return 0;
}

// Pipeline buffer accessors -- all return NULL for Vulkan Phase 1
float *picolm_gpu_pipe_x(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_xb(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_q(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_k(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_v(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_attn_out(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_ffn_norm(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_gate(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_up(int d) { (void)d; return NULL; }

float *picolm_gpu_pipe_x_b(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_xb_b(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_q_b(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_k_b(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_v_b(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_attn_out_b(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_ffn_norm_b(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_gate_b(int d) { (void)d; return NULL; }
float *picolm_gpu_pipe_up_b(int d) { (void)d; return NULL; }

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

uint16_t *picolm_gpu_kv_k_dev(int d) { (void)d; return NULL; }
uint16_t *picolm_gpu_kv_v_dev(int d) { (void)d; return NULL; }

// Elementwise stubs (dev variants)
int picolm_gpu_rmsnorm_dev(float *o, const float *x, const float *w,
                            int d, float e, int dev) {
    (void)o; (void)x; (void)w; (void)d; (void)e; (void)dev; return 0;
}

int picolm_gpu_rmsnorm_batched_dev(float *o, const float *x, const float *w,
                                    int d, float e, int S, int xs, int dev) {
    (void)o; (void)x; (void)w; (void)d; (void)e; (void)S;
    (void)xs; (void)dev; return 0;
}

int picolm_gpu_rope_apply(float *x, int nh, int hd,
                           const float *cos_tbl, const float *sin_tbl,
                           int hd2, int rt, int dev) {
    (void)x; (void)nh; (void)hd; (void)cos_tbl; (void)sin_tbl;
    (void)hd2; (void)rt; (void)dev; return 0;
}

int picolm_gpu_rope_apply_batched(float *x, int nh, int hd,
                                   const float *ct, const float *st,
                                   int hd2, int sp, int S,
                                   int rt, int dev) {
    (void)x; (void)nh; (void)hd; (void)ct; (void)st;
    (void)hd2; (void)sp; (void)S; (void)rt; (void)dev; return 0;
}

int picolm_gpu_residual_add(float *out, const float *a, const float *b,
                             int n, int dim, int stride, int device) {
    (void)out; (void)a; (void)b; (void)n; (void)dim;
    (void)stride; (void)device; return 0;
}

int picolm_gpu_silu_mul_dev(float *g, const float *u, size_t n, int dev) {
    (void)g; (void)u; (void)n; (void)dev; return 0;
}

int picolm_gpu_sync(int device) {
    if (!G.ready || device != 0) return 0;
    return vkDeviceWaitIdle(G.dev) == VK_SUCCESS;
}

int picolm_gpu_memcpy(void *dst, const void *src, size_t bytes,
                       int dir, int device) {
    (void)dst; (void)src; (void)bytes; (void)dir; (void)device; return 0;
}

int picolm_gpu_memcpy_async(void *dst, const void *src, size_t bytes,
                             int dir, int device) {
    (void)dst; (void)src; (void)bytes; (void)dir; (void)device; return 0;
}

float *picolm_gpu_staging_host(int device, size_t bytes) {
    (void)device; (void)bytes; return NULL;
}

int picolm_gpu_qg_deinterleave_dev(const float *raw, float *oq, float *og,
                                    int nh, int hd, int dev) {
    (void)raw; (void)oq; (void)og; (void)nh; (void)hd; (void)dev; return 0;
}

int picolm_gpu_sigmoid_mul_dev(float *o, const float *g,
                                int n, int dev) {
    (void)o; (void)g; (void)n; (void)dev; return 0;
}

int picolm_gpu_qg_deinterleave_batched_dev(const float *raw, float *oq,
                                            float *og, int nh, int hd,
                                            int S, int dev) {
    (void)raw; (void)oq; (void)og; (void)nh; (void)hd; (void)S; (void)dev; return 0;
}

int picolm_gpu_sigmoid_mul_batched_dev(float *o, const float *g,
                                        int n, int S, int dev) {
    (void)o; (void)g; (void)n; (void)S; (void)dev; return 0;
}

// Stubs for functions declared extern in model_core.c / model_ssm.c
// (not in backend_gpu.h, defined in backend_gpu.cu)

int picolm_gpu_pipeline_batch_alloc(int dim, int q_dim, int kv_dim,
                                     int ffn_hidden, int xb_stride,
                                     int max_seq_len, int device) {
    (void)dim; (void)q_dim; (void)kv_dim; (void)ffn_hidden;
    (void)xb_stride; (void)max_seq_len; (void)device; return 0;
}

int picolm_gpu_rmsnorm_matmul_dev_qkv(picolm_gpu_tensor_t *tq,
                                       picolm_gpu_tensor_t *tk,
                                       picolm_gpu_tensor_t *tv,
                                       float *bq, float *bk, float *bv,
                                       const float *bx, const float *rmsnorm_w,
                                       int dim, float eps, int S,
                                       int xb_stride, int device,
                                       int y_stride_q, int y_stride_kv) {
    (void)tq; (void)tk; (void)tv; (void)bq; (void)bk; (void)bv;
    (void)bx; (void)rmsnorm_w; (void)dim; (void)eps; (void)S;
    (void)xb_stride; (void)device; (void)y_stride_q; (void)y_stride_kv;
    return 0;
}
