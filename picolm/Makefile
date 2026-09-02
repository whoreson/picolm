CC      ?= gcc

# ---- Compiler detection ----
# -Wduplicated-cond, -Wduplicated-branches and -Wformat-truncation are
# GCC-only. Clang (incl. Apple Clang, and the `gcc` shim on macOS which is
# really clang) doesn't recognize them and emits a -Wunknown-warning-option
# note per file per flag -- 39 spurious lines across the 13 C TUs. Gate them so
# the warning stream stays trustworthy under both compilers.
_is_clang = $(shell echo | $(CC) -E -dM - 2>/dev/null | grep -c '__clang__')
ifeq ($(call _is_clang),1)
WARN_GCC_ONLY =
else
WARN_GCC_ONLY = -Wduplicated-cond -Wduplicated-branches -Wformat-truncation
endif

# ---- Strict warning flags (opt-in, not default) ----
# These catch real bugs but are too noisy for routine builds (600+ warnings
# across 13 TUs, mostly -Wsign-conversion and -Wdouble-promotion on benign
# int-to-size_t / float-to-double conversions). Enable with make WARN=1 or
# for ad-hoc audits: make native WARN=1
CFLAGS_STRICT = -Wshadow -Wdouble-promotion -Wimplicit-fallthrough -Wconversion -Wmisleading-indentation $(WARN_GCC_ONLY) -Wfloat-conversion -Wsign-conversion -Wcast-qual -Wformat-nonliteral

CFLAGS  ?= -O3 -g -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wpedantic -fno-common -fopenmp
LDFLAGS = -lm -lpthread -fopenmp

# Windows needs ws2_32 for server mode
ifeq ($(OS),Windows_NT)
LDFLAGS += -lws2_32
# Do NOT set SHELL := cmd.exe. On MSYS/MinGW, $(shell ...) pipe chains
# (echo | gcc -E -dM - | grep -c '__AVX2__') fail when SHELL is cmd.exe
# because MinGW gcc.exe cannot write stdout through a cmd.exe pipe to an
# MSYS binary like grep. MSYS sh.exe handles these pipes correctly.
# rm -f from MSYS works fine on Windows for cleaning .o files.
endif

RM = rm -f

# FreeBSD 8.4: system ld (2.15) doesn't support -plugin; use BFD
ifeq ($(shell uname -s 2>/dev/null),FreeBSD)
LDFLAGS += -fuse-ld=bfd
endif

# Source files and corresponding .o files
# model.c has been split into modular files
MODEL_SRCS = model_core.c model_attention.c model_moe.c model_gemma3n.c model_ssm.c model_kv_cache.c model_gguf.c
SRCS    = picolm.c $(MODEL_SRCS) tensor.c quant.c sgemm.c tokenizer.c qwen_tokenize.c sampler.c grammar.c cJSON.c server.c csafetensors.c json.c safetensors.c
OBJS    = $(SRCS:.c=.o)
TARGET  = picolm

# VIZ support (adds viz.c when PICOLM_VIZ is defined)
VIZ_SRCS = viz.c
VIZ_OBJS = $(VIZ_SRCS:.c=.o)

# GPU backend
HIPFLAGS ?= -O3 -std=c++11 -fPIC -fPIE
HIP_LIBS = -lamdhip64
GPU_OBJS =

# GPU host .cu sources (backend_gpu.cu has been split into modular files)
GPU_SRCS = backend_gpu_kernels.cu backend_gpu_host_core.cu backend_gpu_host_ssm.cu backend_gpu_host_misc.cu
GPU_CUH  = backend_gpu_common.cuh backend_gpu_dequant.cuh backend_gpu_kernels.cuh

PREFIX  ?= /usr/local
MODEL_DIR ?= /opt/picolm/models

# ---- Auto-detect SIMD level from compiler flags ----
# These define PICOLM_* macros that gate the optimized code paths.
# Detection cascades: AVX-512 > AVX2 > AVX > SSE3 > SSE2 (first match wins).
# When -march=native is used, the compiler defines __AVX512F__, __AVX2__, etc.

_have_avx512 = $(shell echo | $(CC) -E -dM $(1) - 2>/dev/null | grep -c '__AVX512F__')
_have_avx2   = $(shell echo | $(CC) -E -dM $(1) - 2>/dev/null | grep -c '__AVX2__')
_have_avx    = $(shell echo | $(CC) -E -dM $(1) - 2>/dev/null | grep -c '__AVX__')
_have_sse3   = $(shell echo | $(CC) -E -dM $(1) - 2>/dev/null | grep -c '__SSSE3__')
_have_sse2   = $(shell echo | $(CC) -E -dM $(1) - 2>/dev/null | grep -c '__SSE2__')
_have_fma    = $(shell echo | $(CC) -E -dM $(1) - 2>/dev/null | grep -c '__FMA__')
_have_f16c   = $(shell echo | $(CC) -E -dM $(1) - 2>/dev/null | grep -c '__F16C__')
_have_neon   = $(shell echo | $(CC) -E -dM $(1) - 2>/dev/null | grep -c 'define __ARM_NEON ')
_have_i8mm   = $(shell echo | $(CC) -E -dM $(1) - 2>/dev/null | grep -c '__ARM_FEATURE_MATMUL_INT8')

# Pick the highest SIMD level and emit the right -DPICOLM_* preprocessor defines.
# Cascading if/elif/else: only the first (highest) matching level is emitted.
simd_defines_for = $(strip \
  $(if $(filter 1,$(call _have_avx512,$(1))),\
    -DPICOLM_AVX512 -DPICOLM_AVX2 -DPICOLM_AVX -DPICOLM_SSE3 -DPICOLM_SSE2 -DPICOLM_FMA,\
    $(if $(filter 1,$(call _have_avx2,$(1))),\
      -DPICOLM_AVX2 -DPICOLM_AVX -DPICOLM_SSE3 -DPICOLM_SSE2 -DPICOLM_FMA,\
      $(if $(filter 1,$(call _have_avx,$(1))),\
        -DPICOLM_AVX -DPICOLM_SSE3 -DPICOLM_SSE2,\
        $(if $(filter 1,$(call _have_sse3,$(1))),\
          -DPICOLM_SSE3 -DPICOLM_SSE2,\
          $(if $(filter 1,$(call _have_sse2,$(1))),\
            -DPICOLM_SSE2,\
            ))))))

# Emit the right -m* compiler flags for the detected SIMD level.
# Used alongside -march=native to ensure inline intrinsics compile correctly
# (mingw sometimes needs explicit feature flags).
# Note: F16C is NOT implied by AVX. Bulldozer/Piledriver have AVX without F16C.
# We detect F16C separately and only add -mf16c when the CPU actually supports it.
# The $(1) argument is -march=native, so _have_f16c checks the actual CPU.
f16c_flag = $(if $(filter 1,$(call _have_f16c,$(1))),-mf16c,)
simd_flags_for = $(strip \
  $(if $(filter 1,$(call _have_avx512,$(1))),\
    -mavx512f -mavx512vl -mavx512vnni $(call f16c_flag,$(1)) -mbmi2 -mfma,\
    $(if $(filter 1,$(call _have_avx2,$(1))),\
      -mavx2 -mfma $(call f16c_flag,$(1)) -mbmi2,\
      $(if $(filter 1,$(call _have_avx,$(1))),\
        -mavx $(call f16c_flag,$(1)),\
        $(if $(filter 1,$(call _have_sse3,$(1))),\
          -mssse3 -mpopcnt,\
          )))))

# --- Default target: build with -march=native + auto-detected SIMD ---
.PHONY: all
all:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -march=native $(call simd_flags_for,-march=native) $(call simd_defines_for,-march=native) $(if $(filter 1,$(call _have_neon,-march=native)),-DPICOLM_NEON,) $(if $(filter 1,$(call _have_i8mm,-march=native)),-DPICOLM_I8MM,)"

# --- Native build: -march=native with auto-detected SIMD defines ---
native:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -march=native $(call simd_flags_for,-march=native) $(call simd_defines_for,-march=native) $(if $(filter 1,$(call _have_neon,-march=native)),-DPICOLM_NEON,) $(if $(filter 1,$(call _have_i8mm,-march=native)),-DPICOLM_I8MM,)"

# --- VIZ build: native with embedded VNC visualization server ---
# Remove affected .o files so the sub-make recompiles with -DPICOLM_VIZ.
# After model.c was split into modular files, we must also clean the model
# modules that contain #ifdef PICOLM_VIZ blocks (viz_push_layer calls).
# Uses $(RM) (Make built-in, defaults to `rm -f`) for cross-platform safety.
.PHONY: viz-clean-objs
viz-clean-objs:
	-$(RM) model.o picolm.o server.o model_core.o model_ssm.o model_attention.o model_gemma3n.o

viz: viz-clean-objs
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -march=native $(call simd_flags_for,-march=native) $(call simd_defines_for,-march=native) $(if $(filter 1,$(call _have_neon,-march=native)),-DPICOLM_NEON,) $(if $(filter 1,$(call _have_i8mm,-march=native)),-DPICOLM_I8MM,) -DPICOLM_VIZ=1 -DPICOLM_SERVER" OBJS="$(OBJS) viz.o"

# --- x86-64 default: -march=native + GPU disabled, safe fallback ---
x86: sse2

# --- No SIMD (scalar fallback, portable to any architecture) ---
scalar: CFLAGS += -mno-sse2 -mno-avx
scalar: $(TARGET)

# --- x86-64 with SSE2 only ---
sse2:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -msse2 -DPICOLM_SSE2"

# --- x86-64 with SSE2+SSE3+SSSE3 (covers AMD Phenom/Athlon and similar without AVX) ---
sse3:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -msse2 -msse3 -mssse3 -mpopcnt -DPICOLM_SSE3 -DPICOLM_SSSE3 -DPICOLM_SSE2"

# --- x86-64 with AVX (Sandy Bridge and newer Intel; Bulldozer and newer AMD) ---
avx:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -mavx -mf16c -mpopcnt -DPICOLM_AVX -DPICOLM_SSE2"

# --- x86-64 with AVX2 (Haswell and newer Intel; Excavator and newer AMD) ---
avx2:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -mavx2 -mfma -mpopcnt -mf16c -DPICOLM_AVX2 -DPICOLM_FMA -DPICOLM_AVX -DPICOLM_SSE2"

# --- Persistent thread pool (WIP, for development) ---
threadpool:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -march=native $(call simd_defines_for,-march=native) -DPICOLM_USE_PER_MATMUL_THREAD=0"

# --- Link ---
$(TARGET): $(OBJS) $(GPU_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(GPU_LIBS) -lstdc++

# --- Compile each .c to .o independently ---
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Header dependencies
picolm.o: picolm.c quant.h tensor.h model.h tokenizer.h sampler.h grammar.h
viz.o:  viz.c  viz.h
model_core.o:   model_core.c  model.h quant.h tensor.h model_internal.h
model_gguf.o:   model_gguf.c  model.h quant.h tensor.h model_internal.h
model_attention.o: model_attention.c  model.h quant.h tensor.h model_internal.h
model_moe.o:    model_moe.c  model.h quant.h tensor.h model_internal.h
model_gemma3n.o: model_gemma3n.c  model.h quant.h tensor.h model_internal.h
model_ssm.o:    model_ssm.c  model.h quant.h tensor.h model_internal.h
model_kv_cache.o: model_kv_cache.c  model.h quant.h tensor.h model_internal.h
tensor.o: tensor.c tensor.h quant.h model.h
quant.o:  quant.c  quant.h
sgemm.o:  sgemm.c  sgemm.h quant.h
tokenizer.o: tokenizer.c tokenizer.h
sampler.o: sampler.c sampler.h
grammar.o: grammar.c grammar.h

# --- Static build for single-binary deployment ---
# Uses SSE2 (not -march=native) so the binary runs on any x86-64, not just the build machine.
static:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -msse2 -DPICOLM_SSE2" LDFLAGS="$(LDFLAGS) -static"

# --- Pi 3/4/5 (64-bit ARM with NEON SIMD) ---
pi:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -march=armv8-a+simd -DPICOLM_NEON"

# --- Pi Zero / Pi 1 (32-bit ARM) ---
pi-arm32: CC = arm-linux-gnueabihf-gcc
pi-arm32:
	$(MAKE) $(TARGET) CC=arm-linux-gnueabihf-gcc CFLAGS="$(CFLAGS) -march=armv6 -mfpu=vfp"

# --- Cross-compile for Pi64 from platforms (e.g. x86) ---
cross-pi: CC = aarch64-linux-gnu-gcc
cross-pi:
	$(MAKE) $(TARGET) CC=aarch64-linux-gnu-gcc CFLAGS="$(CFLAGS) -march=armv8-a+simd -DPICOLM_NEON" LDFLAGS="$(LDFLAGS) -static"

# --- RISC-V (Sipeed LicheeRV, etc.) ---
riscv: CC = riscv64-linux-gnu-gcc
riscv: $(TARGET)

# --- Cross-compile for RISC-V from platforms (e.g. x86) ---
cross-riscv: CC = riscv64-linux-gnu-gcc
cross-riscv: $(TARGET)

# --- Mac OS X 10.4 Tiger (PowerPC G4, GCC 4.0.1, Altivec) ---
# Requires: gcc -maltivec, OSSwapInt* from libkern/OSByteOrder.h
# No OpenMP, no C11, no static linking, no libstdc++
ppc:
	$(MAKE) $(TARGET) CFLAGS="-O2 -g -std=c99 -D_DARWIN_C_SOURCE -D_GNU_SOURCE -Wall -maltivec -DPICOLM_ALTIVEC" LDFLAGS="-lm"

# --- Mac OS X 10.6 Snow Leopard (Core 2 Duo, GCC 4.2.1, SSE4.1) ---
# GCC 4.2: no -march=native, no -std=c11, no -Wpedantic, no <immintrin.h>
# SSE3 intrinsics are broken in GCC 4.2's <emmintrin.h>; SSSE3+SSE4.1 work
# via individual headers. __thread not supported.
mac106:
	$(MAKE) $(TARGET) CFLAGS="-O2 -g -std=c99 -D_GNU_SOURCE -Wall -msse4.1 -mssse3 -msse2 -DPICOLM_SSE3 -DPICOLM_SSE2" LDFLAGS="-lm -lpthread -fopenmp"

# --- Debug build ---
debug:
	$(MAKE) $(TARGET) CFLAGS="-std=c11 -D_GNU_SOURCE -Wall -Wextra -Wpedantic -g -O0"

# --- Install ---
install: $(TARGET)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/
	@echo "Installed $(TARGET) to $(DESTDIR)$(PREFIX)/bin/"

# --- Download default model ---
model:
	@mkdir -p $(MODEL_DIR)
	@if [ ! -f "$(MODEL_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]; then \
	 echo "Downloading TinyLlama 1.1B Q4_K_M (638 MB)..."; \
	 curl -L -o "$(MODEL_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
	 "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"; \
	 echo "Done."; \
	else \
	 echo "Model already exists at $(MODEL_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"; \
	fi

clean:
	-$(RM) $(TARGET) $(TARGET).exe $(OBJS) $(VIZ_OBJS) backend_gpu.o backend_gpu.o.hip backend_gpu.o.cuda backend_metal.o backend_vulkan.o
	-$(RM) $(GPU_SRCS:%.cu=%-hip.o) $(GPU_SRCS:%.cu=%-cuda.o) backend_gpu.dlink.o
	-$(RM) shaders/*.spv

.PHONY: asan
asan:
	$(MAKE) $(TARGET) CFLAGS="-fsanitize=address -g -O2 -march=native $(call simd_defines_for,-march=native) -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wpedantic -fopenmp" LDFLAGS="-lm -lpthread -fopenmp -fsanitize=address"

# --- Force recompile of GPU-dependent C objects ---
# These objects contain #ifdef PICOLM_GPU code; when building GPU targets
# alongside a prior native build, Make won't recompile them because
# CFLAGS is a variable, not a prerequisite. Remove them so the sub-make
# with -DPICOLM_GPU=1 rebuilds them from scratch.
.PHONY: gpu-clean-objs
gpu-clean-objs:
	-$(RM) model.o tensor.o quant.o picolm.o
	-$(RM) model_core.o model_attention.o model_moe.o model_gemma3n.o model_ssm.o model_kv_cache.o model_gguf.o

# --- ROCm/HIP GPU backend ---

# Auto-detect ROCm root. Strategy:
#   1. If rocminfo is in PATH, derive its parent/../ (e.g. /usr/bin/rocminfo -> /usr, /opt/rocm/bin/rocminfo -> /opt/rocm).
#   2. Scan well-known locations for a working rocminfo.
# Override with ROCM_ROOT=/path/to/rocm if needed.
ROCM_ROOT := $(shell \
  r=$$(command -v rocminfo 2>/dev/null); \
  if [ -n "$$r" ]; then dirname "$$(dirname "$$r")"; \
  else \
    for d in /opt/rocm /opt/rocm-6.3.3 /opt/rocm-6.2.4 /opt/rocm-6.2 /opt/rocm-6.1 /opt/rocm-5.7 /usr/lib/rocm /usr; do \
      if [ -x "$$d/bin/rocminfo" ]; then echo "$$d"; break; fi; \
    done; \
  fi)
ifeq ($(ROCM_ROOT),)
ROCM_ROOT := /opt/rocm
endif

ROCM_LIB ?= $(ROCM_ROOT)/lib
HIPCC    ?= $(ROCM_ROOT)/bin/hipcc

# Auto-detect GPU architecture via rocminfo, fall back to gfx906.
# rocminfo outputs lines like "Name:                    gfx906" for each GPU.
HIP_ARCH := $(shell $(ROCM_ROOT)/bin/rocminfo 2>/dev/null | grep 'Name:' | grep 'gfx' | head -1 | awk '{print $$2}')
ifeq ($(HIP_ARCH),)
HIP_ARCH := gfx906
endif

# HIP target architecture.
#   Auto-detected via rocminfo when possible. Override with HIP_ARCH=gfx940 etc.
#   Options: gfx906/908 (RDNA2/3, no WMMA), gfx940/941/942 (CDNA2+, WMMA)

# HIP: compile each GPU host .cu independently
HIP_OBJS := $(GPU_SRCS:%.cu=%-hip.o)

.PHONY: hip
hip: $(HIP_OBJS) gpu-clean-objs
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -march=native $(call simd_defines_for,-march=native) -DPICOLM_GPU=1 -DPICOLM_HIP=1" GPU_OBJS="$(HIP_OBJS)" GPU_LIBS="-L$(ROCM_LIB) -lamdhip64 -lstdc++ -Wl,-rpath,$(ROCM_LIB)"

$(HIP_OBJS): %-hip.o: %.cu $(GPU_CUH) backend_gpu.h quant.h
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(HIP_ARCH) -DPICOLM_GPU=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -c -o $@ $< -I.

# --- CUDA (NVIDIA) GPU backend ---
NVCC      ?= nvcc
CUDA_INC  ?= /usr/local/cuda/include
CUDA_LIB  ?= /usr/local/cuda/lib64

# CUDA target architecture(s).
#   Single arch (thin binary):   CUDA_ARCH=sm_121
#   Multiple archs (fat binary): CUDA_ARCHS="sm_70 sm_80 sm_121"
#   Default: sm_70 (Volta+; sufficient for WMMA Tensor Cores)
#   GPU compute capabilities:
#     sm_70 Volta, sm_75 Turing, sm_80 Ampere, sm_86/89 Ampere/Ada,
#     sm_90 Hopper, sm_121 Blackwell (GB10)
CUDA_ARCH ?= sm_70
CUDA_ARCHS = $(CUDA_ARCH)

# Build -gencode flags from CUDA_ARCHS
CUDA_GENCODE := $(foreach arch,$(CUDA_ARCHS),-gencode arch=compute_$(subst sm_,,$(arch)),code=$(arch))

CUDAFLAGS   ?= -O3 --fmad=false -std=c++11 -Xcompiler="-fPIC"

# CUDA: compile each GPU host .cu independently
CUDA_OBJS := $(GPU_SRCS:%.cu=%-cuda.o)

.PHONY: cuda
cuda: $(CUDA_OBJS) gpu-clean-objs
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -march=native $(call simd_defines_for,-march=native) -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -I$(CUDA_INC)" GPU_OBJS="$(CUDA_OBJS)" GPU_LIBS="-L$(CUDA_LIB) -lcudart -lstdc++ -Wl,-rpath,$(CUDA_LIB)"

$(CUDA_OBJS): %-cuda.o: %.cu $(GPU_CUH) backend_gpu.h quant.h
	$(NVCC) $(CUDAFLAGS) $(CUDA_GENCODE) -DPICOLM_GPU=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -c -o $@ $< -I.

# --- Metal (Apple Silicon) backend -- RAW Metal framework, no MLX-C ---
# macOS only. No external dependencies beyond the Metal framework that ships
# with macOS + the Command Line Tools (clang). Compile the whole backend with:
#     make metal
# Then run with: PICOLM_GPU=1 ./picolm model.gguf ...
METAL_CXX      ?= clang++
# Strict warning set mirroring the C CFLAGS (clang-compatible subset; the
# GCC-only flags are intentionally absent since METAL_CXX is clang++).
# backend_metal.mm is the one TU the metal build is supposed to validate --
# previously it compiled with no warning flags at all.
METAL_CXXFLAGS ?= -std=c++20 -O3 -fobjc-arc -fPIC -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wdouble-promotion -Wfloat-conversion -Wcast-qual -Wmisleading-indentation -Wimplicit-fallthrough -Wformat-nonliteral
METAL_FRAMEWORKS = -framework Metal -framework Foundation
METAL_LIBS       = $(METAL_FRAMEWORKS)
# (the link rule appends -lstdc++ for the ObjC++ runtime)
.PHONY: metal
metal: backend_metal.o gpu-clean-objs
	$(MAKE) $(TARGET) \
	  CFLAGS="$(filter-out -fopenmp,$(CFLAGS)) -DPICOLM_GPU=1" \
	  LDFLAGS="$(filter-out -fopenmp,$(LDFLAGS))" \
	  GPU_OBJS="backend_metal.o" GPU_LIBS="$(METAL_LIBS)"

# .mm = Objective-C++ so we can use <metal/Metal.h> directly.
backend_metal.o: backend_metal.mm backend_gpu.h quant.h
	$(METAL_CXX) $(METAL_CXXFLAGS) -DPICOLM_GPU=1 -c -o $@ $< -I.

# --- Metal kernel probe (smoke test for the raw-Metal kernel mechanism) ---
# Builds and runs probes/metal_reduction_probe.mm against the Metal framework.
# Needs only clang++ + the macOS SDK (no CMake, no MLX-C).
.PHONY: metal-probe metal-probe-run
metal-probe: probes/metal_reduction_probe
probes/metal_reduction_probe: probes/metal_reduction_probe.mm
	$(METAL_CXX) $(METAL_CXXFLAGS) $< $(METAL_FRAMEWORKS) -o $@
metal-probe-run: metal-probe
	./probes/metal_reduction_probe

# --- Metal matmul benchmark + correctness harness (no model file needed) ---
# Measures each kernel at decode/prefill dims, diffs vs a CPU reference, and
# reports effective weight bandwidth + per-dispatch overhead. Use it to tell
# whether to optimize the kernels or the host launch/sync path.
.PHONY: metal-bench metal-bench-run
metal-bench: probes/metal_matmul_bench
probes/metal_matmul_bench: probes/metal_matmul_bench.mm
	$(METAL_CXX) $(METAL_CXXFLAGS) $< $(METAL_FRAMEWORKS) -o $@
metal-bench-run: metal-bench
	./probes/metal_matmul_bench

# --- End-to-end model timing (needs `make metal` first + GGUF files) ---
# Runs ./picolm on real models, parses prefill/generation tok/s.
#   make metal-model-bench            # GPU
#   make metal-model-bench CPU=1      # also CPU baseline
#   LABEL=after make metal-model-bench # tag the run
# Override models with MODELS="a.gguf b.gguf" or pass them as script args.
.PHONY: metal-model-bench
metal-model-bench:
	./probes/metal_model_bench.sh
.PHONY: hip cuda metal vulkan hunger

# --- Vulkan compute backend ---
# Works on any Vulkan 1.2 GPU (AMD, NVIDIA, Intel, Windows drivers).
# Requires: vulkan loader headers (vulkan/vulkan.h), glslc (shaderc).
#
#   make vulkan                               # auto-detect SDK or use PATH
#   make vulkan VULKAN_SDK=/path/to/sdk       # explicit SDK root
#     On Windows: VULKAN_SDK=C:/data/util/vulkansdk/1.4.309.0
#     On Linux:    VULKAN_SDK=/opt/vulkan-sdk
#   make vulkan GLSLC=/path/to/glslc          # custom shader compiler
#
# Run with: PICOLM_GPU=1 ./picolm model.gguf -p "hello"
# On Windows MinGW: -lvulkan links against vulkan-1.dll (shipped by drivers).
# Shader .spv files are built from .comp sources; alternatively pre-compiled
# .spv files can be placed in ./shaders/ and set PICOLM_VK_SHADER_DIR.

# Vulkan SDK auto-detection: check VULKAN_SDK, then common paths.
# Provides VK_INC (headers), VK_LIB (loader library), VK_BIN (glslc).
ifneq ($(VULKAN_SDK),)
VK_INC      := $(VULKAN_SDK)/include
VK_LIB      := $(VULKAN_SDK)/lib
VK_BIN      := $(VULKAN_SDK)/bin
else
# Common SDK locations (Linux + Windows via MSYS path translation)
VK_INC      := $(shell for d in /opt/vulkan-sdk/include /usr/include; do \
               if [ -d "$$d" ]; then echo "$$d"; break; fi; done)
VK_LIB      := $(shell for d in /opt/vulkan-sdk/lib /usr/lib; do \
               if [ -d "$$d" ]; then echo "$$d"; break; fi; done)
VK_BIN      := $(shell for d in /opt/vulkan-sdk/bin /usr/bin; do \
               if [ -d "$$d" ]; then echo "$$d"; break; fi; done)
endif

# glslc: from SDK bin dir first, then PATH
ifneq ($(VK_BIN),)
GLSLC       ?= $(VK_BIN)/glslc
else
GLSLC       ?= glslc
endif

VK_CC       ?= gcc
VK_CXX      ?= gcc
# -I$(VK_INC) only added when VK_INC is non-empty and exists
VK_EXTRA_INC = $(if $(VK_INC),$(shell if [ -d "$(VK_INC)" ]; then echo "-I$(VK_INC)"; fi))
# On Windows SDK: MinGW ld can link against MSVC vulkan-1.lib (COFF format).
# On Linux SDK: lib/libvulkan.a or libvulkan.so.
# Fallback: -lvulkan from system paths.
# Note: pass full path to .lib to avoid -L/path libname split issues.
VK_EXTRA_LIB = $(if $(VULKAN_SDK),$(shell \
               if [ -f "$(VULKAN_SDK)/Lib/vulkan-1.lib" ]; then echo "$(VULKAN_SDK)/Lib/vulkan-1.lib"; \
               elif [ -f "$(VK_LIB)/libvulkan.a" ] || [ -f "$(VK_LIB)/libvulkan.so" ]; then echo "-L$(VK_LIB) -lvulkan"; \
               else echo "-lvulkan"; fi))
VK_CFLAGS   ?= -O3 -std=c11 -fPIC -Wall $(VK_EXTRA_INC)
VK_LDFLAGS  ?= $(VK_EXTRA_LIB)

shaders/%.spv: shaders/%.comp
	$(GLSLC) --target-env=vulkan1.2 $< -o $@

.PHONY: vulkan
vulkan: backend_vulkan.o shaders/qmatmul_vk.spv shaders/elementwise_vk.spv
	$(MAKE) $(TARGET) \
	  CFLAGS="$(filter-out -fopenmp,$(CFLAGS)) -march=native -DPICOLM_GPU=1 $(call simd_flags_for,-march=native) $(call simd_defines_for,-march=native) $(VK_EXTRA_INC)" \
	  LDFLAGS="$(filter-out -fopenmp,$(LDFLAGS)) $(VK_EXTRA_LIB)" \
	  GPU_OBJS="backend_vulkan.o"

backend_vulkan.o: backend_vulkan.c backend_gpu.h quant.h
	$(VK_CC) $(VK_CFLAGS) -DPICOLM_GPU=1 -c -o $@ $< -I.

.PHONY: vulkan-shaders
vulkan-shaders: shaders/qmatmul_vk.spv shaders/rmsnorm_vk.spv shaders/elementwise_vk.spv

# --- hunger: CUDA build for Windows 11 (MSYS2 + MSVC + CUDA 12.4) ---
# Target box: Windows 11, MSYS2 make, VS2022 Community on D:, CUDA on C:
# Paths gleaned from /data4/build-lcpp (VCP environment from that box).
#
#   make hunger
#
# Produces: picolm.exe with CUDA GPU support.
# Run with: set PICOLM_GPU=1 & picolm.exe <model.gguf> -p "hello"

HUNGER_CUDA_ROOT  ?= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4
HUNGER_CUDA_ARCH  ?= sm_89
HUNGER_VS_ROOT    ?= D:/Program Files/Microsoft Visual Studio/2022/Community
HUNGER_MSVC_VER   ?= 14.44.35207

# Compile as C. The /c flag is implicit when no /Fe or /LD is given.
# All .obj files go to the linker, never to cl.exe.
HUNGER_CL_DIR = $(HUNGER_VS_ROOT)/VC/Tools/MSVC/$(HUNGER_MSVC_VER)/bin/Hostx64/x64
HUNGER_CL = "$(HUNGER_CL_DIR)/cl.exe"
HUNGER_LNK = "$(HUNGER_CL_DIR)/link.exe"
HUNGER_CFILES = $(SRCS)
HUNGER_COBJS = $(HUNGER_CFILES:.c=.obj)
HUNGER_CUOBS = backend_gpu_kernels_win.obj backend_gpu_host_core_win.obj backend_gpu_host_ssm_win.obj backend_gpu_host_misc_win.obj
# cl.exe flags - all Make vars, NO batch %VAR% references
# MSVC /arch:AVX512 enables the codegen but the C code relies on PICOLM_*
# preprocessor macros for SIMD dispatch. These are not auto-derived by
# quant.h on MSVC (detection chain is GCC/Clang-centric), so we define
# them explicitly here. Cascade: AVX512 implies AVX2 implies AVX etc.
HUNGER_SIMD = /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1
HUNGER_WD = /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477
HUNGER_CF = /O2 /W3 $(HUNGER_WD) /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"$(HUNGER_CUDA_ROOT)/include" /openmp /c $(HUNGER_SIMD)
# hunger-nogpu: same as hunger but without GPU backend (pure MSVC, no nvcc, no cudart)
HUNGER_CF_NOGPU = /O2 /W3 $(HUNGER_WD) /arch:AVX512 /D _CRT_SECURE_NO_WARNINGS /I. /openmp /c $(HUNGER_SIMD)

hunger:
	echo @echo off > .hunger_build.bat
	echo del /q picolm.exe $(HUNGER_CUOBS) $(HUNGER_COBJS) 2^>nul >> .hunger_build.bat
	echo call "$(HUNGER_VS_ROOT)/VC/Auxiliary/Build/vcvars64.bat" >> .hunger_build.bat
	echo "$(HUNGER_CUDA_ROOT)/bin/nvcc.exe" -O3 --fmad=false -std=c++14 -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -ccbin $(HUNGER_CL) -Xcompiler "/EHsc /O2 /W3 /std:c++14 /arch:AVX512 /wd4244 /wd4267 /wd4101 /wd4102 /DPICOLM_AVX512=1 /DPICOLM_AVX2=1 /DPICOLM_AVX=1 /DPICOLM_SSE3=1 /DPICOLM_SSE2=1 /DPICOLM_FMA=1" -gencode arch=compute_89,code=$(HUNGER_CUDA_ARCH) -c -o backend_gpu_kernels_win.obj backend_gpu_kernels.cu -I. >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo "$(HUNGER_CUDA_ROOT)/bin/nvcc.exe" -O3 --fmad=false -std=c++14 -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -ccbin $(HUNGER_CL) -Xcompiler "/EHsc /O2 /W3 /std:c++14 /arch:AVX512 /wd4244 /wd4267 /wd4101 /wd4102 /DPICOLM_AVX512=1 /DPICOLM_AVX2=1 /DPICOLM_AVX=1 /DPICOLM_SSE3=1 /DPICOLM_SSE2=1 /DPICOLM_FMA=1" -gencode arch=compute_89,code=$(HUNGER_CUDA_ARCH) -c -o backend_gpu_host_core_win.obj backend_gpu_host_core.cu -I. >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo "$(HUNGER_CUDA_ROOT)/bin/nvcc.exe" -O3 --fmad=false -std=c++14 -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -ccbin $(HUNGER_CL) -Xcompiler "/EHsc /O2 /W3 /std:c++14 /arch:AVX512 /wd4244 /wd4267 /wd4101 /wd4102 /DPICOLM_AVX512=1 /DPICOLM_AVX2=1 /DPICOLM_AVX=1 /DPICOLM_SSE3=1 /DPICOLM_SSE2=1 /DPICOLM_FMA=1" -gencode arch=compute_89,code=$(HUNGER_CUDA_ARCH) -c -o backend_gpu_host_ssm_win.obj backend_gpu_host_ssm.cu -I. >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo "$(HUNGER_CUDA_ROOT)/bin/nvcc.exe" -O3 --fmad=false -std=c++14 -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -ccbin $(HUNGER_CL) -Xcompiler "/EHsc /O2 /W3 /std:c++14 /arch:AVX512 /wd4244 /wd4267 /wd4101 /wd4102 /DPICOLM_AVX512=1 /DPICOLM_AVX2=1 /DPICOLM_AVX=1 /DPICOLM_SSE3=1 /DPICOLM_SSE2=1 /DPICOLM_FMA=1" -gencode arch=compute_89,code=$(HUNGER_CUDA_ARCH) -c -o backend_gpu_host_misc_win.obj backend_gpu_host_misc.cu -I. >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) picolm.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) model_core.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) model_attention.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) model_moe.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) model_gemma3n.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) model_ssm.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) model_kv_cache.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) model_gguf.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) tensor.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) quant.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) sgemm.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) tokenizer.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) qwen_tokenize.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) sampler.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) grammar.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) cJSON.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) server.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) csafetensors.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) json.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) safetensors.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_LNK) /OUT:picolm.exe $(HUNGER_COBJS) $(HUNGER_CUOBS) "$(HUNGER_CUDA_ROOT)/lib/x64/cudart.lib" ws2_32.lib advapi32.lib >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo del /q $(HUNGER_CUOBS) 2^>nul >> .hunger_build.bat
	echo del .hunger_build.bat >> .hunger_build.bat
	cmd /c .hunger_build.bat

# --- hunger-nogpu: pure MSVC build, no CUDA, no GPU backend ---
# Useful for isolating CPU-only bugs on Windows (same compiler as hunger,
# but without PICOLM_GPU so all GPU code is excluded at compile time).
# --- hunger-msys-direct: compile directly from MSYS2 bash (no .bat, no cmd.exe) ---
# Requires env vars set BEFORE make:
#   MSYSTEM=UCRT64 MINGW_PREFIX=/ucrt64 MINGW_CHOST=x86_64-w64-mingw32
#   MSYSTEM_CARCH=x86_64 MSYSTEM_CHOST=x86_64-w64-mingw32 MSYSTEM_PREFIX=/ucrt64
#   PATH must include: CUDA bin + MSVC cl.exe bin + MSYS2 bin (/ucrt64/bin:/usr/bin:/bin)
#   INCLUDE: MSVC include + Windows SDK (ucrt, um, shared) with backslash paths
#   LIB: MSVC lib + Windows SDK lib (ucrt, um) with backslash paths
#   MSYS2_ARG_CONV_EXCL="*" (required for cl.exe /flags)
# See /data4/work/notes/picolm/msys2-build-notes.md for full setup.
HUNGER_CL_WIN = D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/$(HUNGER_MSVC_VER)/bin/Hostx64/x64/cl.exe
HUNGER_LINK_WIN = D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/$(HUNGER_MSVC_VER)/bin/Hostx64/x64/link.exe
HUNGER_CUDA_WIN = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4
HUNGER_NVCC_FLAGS = -O3 --fmad=false -std=c++14 -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -ccbin "$(HUNGER_CL_WIN)" -Xcompiler "/EHsc /O2 /W3 /std:c++14 /arch:AVX2 /wd4244 /wd4267 /wd4101 /wd4102 /DPICOLM_AVX2=1 /DPICOLM_AVX=1 /DPICOLM_SSE3=1 /DPICOLM_SSE2=1 /DPICOLM_FMA=1" -gencode arch=compute_89,code=$(HUNGER_CUDA_ARCH) -c -o
HUNGER_CL_FLAGS = /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX2 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /D _CRT_AMD64 /I. /openmp /c /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1

.PHONY: hunger-msys-direct
hunger-msys-direct: gpu-clean-objs
	TEMP=/tmp TMP=/tmp $(NVCC) $(HUNGER_NVCC_FLAGS) backend_gpu_kernels_win.obj backend_gpu_kernels.cu -I.
	TEMP=/tmp TMP=/tmp $(NVCC) $(HUNGER_NVCC_FLAGS) backend_gpu_host_core_win.obj backend_gpu_host_core.cu -I.
	TEMP=/tmp TMP=/tmp $(NVCC) $(HUNGER_NVCC_FLAGS) backend_gpu_host_ssm_win.obj backend_gpu_host_ssm.cu -I.
	TEMP=/tmp TMP=/tmp $(NVCC) $(HUNGER_NVCC_FLAGS) backend_gpu_host_misc_win.obj backend_gpu_host_misc.cu -I.
	@export MSYS2_ARG_CONV_EXCL="*" && export INCLUDE="$(INCLUDE);$(HUNGER_CUDA_WIN)/include" && TEMP=/tmp TMP=/tmp "$(HUNGER_CL_WIN)" $(HUNGER_CL_FLAGS) $(HUNGER_CFILES)
	@export MSYS2_ARG_CONV_EXCL="*" && export LIB="$(LIB);$(HUNGER_CUDA_WIN)/lib/x64" && TEMP=/tmp TMP=/tmp "$(HUNGER_LINK_WIN)" /OUT:picolm.exe $(HUNGER_COBJS) $(HUNGER_CUOBS) "$(HUNGER_CUDA_WIN)/lib/x64/cudart.lib" ws2_32.lib advapi32.lib
	-$(RM) $(HUNGER_CUOBS)

# --- hunger-nogpu: pure MSVC build, no CUDA, no GPU backend ---
# Usage: make hunger-nogpu
HUNGER_NOGPU_OBJS = $(HUNGER_CFILES:.c=.obj)

hunger-nogpu:
	echo @echo off > .hunger_nogpu_build.bat
	echo del /q picolm.exe $(HUNGER_NOGPU_OBJS) 2^>nul >> .hunger_nogpu_build.bat
	echo call "$(HUNGER_VS_ROOT)/VC/Auxiliary/Build/vcvars64.bat" >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) picolm.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) model_core.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) model_attention.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) model_moe.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) model_gemma3n.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) model_ssm.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) model_kv_cache.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) model_gguf.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) tensor.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) quant.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) sgemm.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) tokenizer.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) qwen_tokenize.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) sampler.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) grammar.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) cJSON.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) server.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) csafetensors.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) json.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) safetensors.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_LNK) /OUT:picolm.exe $(HUNGER_NOGPU_OBJS) ws2_32.lib advapi32.lib >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo del /q $(HUNGER_NOGPU_OBJS) 2^>nul >> .hunger_nogpu_build.bat
	echo del .hunger_nogpu_build.bat >> .hunger_nogpu_build.bat
	cmd /c .hunger_nogpu_build.bat

.PHONY: hunger-nogpu

# ---- DJGPP (MS-DOS) target ----
# Build host: Linux with cross-compiler at /data4/work/djgpp-build/djgpp-toolchain/bin/
# Run on DOS/DOSEMU/DOSBox. No GPU, no SIMD, no threading, no server/viz.
DJGPP_CC   = /data4/work/djgpp-build/djgpp-toolchain/bin/i586-pc-msdosdjgpp-gcc
DJGPP_INCLUDES = -nostdinc -isystem /data4/work/djgpp-build/djgpp-toolchain/lib/gcc/i586-pc-msdosdjgpp/7.5.0/include -isystem /data4/work/djgpp-build/djgpp-runtime/include
DJGPP_CFLAGS = -O2 -mmmx -march=k6 -std=gnu11 -DPICOLM_DOS=1 -Wall -fomit-frame-pointer -funroll-loops -fno-stack-protector -falign-functions=4 -falign-jumps=4 -falign-loops=4 -ffast-math $(DJGPP_INCLUDES)
DJGPP_SRCS = picolm.c $(MODEL_SRCS) tensor.c quant.c tokenizer.c qwen_tokenize.c \
	sampler.c grammar.c cJSON.c csafetensors.c json.c safetensors.c sgemm.c
DJGPP_OBJS = $(DJGPP_SRCS:.c=.o.dos)
DJGPP_RUNTIME = /data4/work/djgpp-build/djgpp-runtime

%.o.dos: %.c model.h quant.h tensor.h model_internal.h
	$(DJGPP_CC) $(DJGPP_CFLAGS) -c -o $@ $<

dos_stubs.o.dos: dos_stubs.c
	$(DJGPP_CC) $(DJGPP_CFLAGS) -c -o $@ $<

dos: dos_stubs.o.dos $(DJGPP_OBJS)
	$(DJGPP_CC) -nostartfiles -o picolm.exe $< $(DJGPP_RUNTIME)/lib/crt0.o \
	 $(DJGPP_OBJS) -L$(DJGPP_RUNTIME)/lib -lc -lm -lgcc

dos-clean:
	-rm -f $(DJGPP_OBJS) dos_stubs.o.dos picolm.exe

test_quant: util/test_quant.c quant.c sgemm.c
	$(CC) $(CFLAGS) -o $@ util/test_quant.c quant.c sgemm.c -lm

gguf_be_swap: util/gguf_be_swap.c
	$(CC) -O2 -std=c99 -o $@ util/gguf_be_swap.c -lm
