CC      ?= gcc
CFLAGS  ?= -O3 -g -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wpedantic -fopenmp
LDFLAGS = -lm -lpthread -fopenmp

# Windows needs ws2_32 for server mode
ifeq ($(OS),Windows_NT)
LDFLAGS += -lws2_32
endif

# Source files and corresponding .o files
SRCS    = picolm.c model.c tensor.c quant.c tokenizer.c qwen_tokenize.c sampler.c grammar.c cJSON.c server.c csafetensors.c json.c safetensors.c
OBJS    = $(SRCS:.c=.o)
TARGET  = picolm

# VIZ support (adds viz.c when PICOLM_VIZ is defined)
VIZ_SRCS = viz.c
VIZ_OBJS = $(VIZ_SRCS:.c=.o)

# GPU backend
HIPFLAGS ?= -O3 -std=c++11 -fPIC -fPIE
HIP_LIBS = -lamdhip64
GPU_OBJS =

PREFIX  ?= /usr/local
MODEL_DIR ?= /opt/picolm/models

# --- Default: native build with per-matmul threading ---
native: CFLAGS += -march=native
native: $(TARGET)

# --- VIZ build: native with embedded VNC visualization server ---
viz:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -march=native -DPICOLM_VIZ=1" OBJS="$(OBJS) viz.o"

# --- x86-64 default (SSE2 only, safe for all x86-64) ---
x86: sse2

# --- No SIMD (scalar fallback, portable to any architecture) ---
scalar: CFLAGS += -mno-sse2 -mno-avx
scalar: $(TARGET)

# --- x86-64 with SSE2 only ---
sse2: CFLAGS += -msse2
sse2: $(TARGET)

# --- x86-64 with SSE2+SSE3+SSSE3 (covers AMD Phenom/Athlon and similar without AVX) ---
sse3: CFLAGS += -msse2 -msse3 -mssse3 -mpopcnt
sse3: $(TARGET)

# --- x86-64 with AVX (Sandy Bridge and newer Intel; Bulldozer and newer AMD) ---
avx: CFLAGS += -mavx -mf16c -mpopcnt
avx: $(TARGET)

# --- x86-64 with AVX2 (Haswell and newer Intel; Excavator and newer AMD) ---
avx2: CFLAGS += -mavx2 -mfma -mpopcnt -mf16c
avx2: $(TARGET)

# --- Persistent thread pool (WIP, for development) ---
threadpool: CFLAGS += -march=native -DPICOLM_USE_PER_MATMUL_THREAD=0
threadpool: $(TARGET)

# --- Link ---
$(TARGET): $(OBJS) $(GPU_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(GPU_LIBS) -lstdc++

# --- Compile each .c to .o independently ---
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Header dependencies
picolm.o: picolm.c quant.h tensor.h model.h tokenizer.h sampler.h grammar.h
viz.o:  viz.c  viz.h
model.o:  model.c  model.h quant.h tensor.h
tensor.o: tensor.c tensor.h quant.h model.h
quant.o:  quant.c  quant.h
tokenizer.o: tokenizer.c tokenizer.h
sampler.o: sampler.c sampler.h
grammar.o: grammar.c grammar.h

# --- Static build for single-binary deployment ---
# Uses SSE2 (not -march=native) so the binary runs on any x86-64, not just the build machine.
static: CFLAGS += -msse2
static: LDFLAGS += -static
static: $(TARGET)

# --- Pi 3/4/5 (64-bit ARM with NEON SIMD) ---
pi: CFLAGS += -march=armv8-a+simd
pi: $(TARGET)

# --- Pi Zero / Pi 1 (32-bit ARM) ---
pi-arm32: CC = arm-linux-gnueabihf-gcc
pi-arm32: CFLAGS += -march=armv6 -mfpu=vfp
pi-arm32: $(TARGET)

# --- Cross-compile for Pi64 from platforms (e.g. x86) ---
cross-pi: CC = aarch64-linux-gnu-gcc
cross-pi: CFLAGS += -march=armv8-a+simd
cross-pi: LDFLAGS += -static
cross-pi: $(TARGET)

# --- RISC-V (Sipeed LicheeRV, etc.) ---
riscv: CC = riscv64-linux-gnu-gcc
riscv: CFLAGS += -march=rv64gc -mabi=lp64d
riscv: $(TARGET)

# --- Cross-compile for RISC-V from platforms (e.g. x86) ---
cross-riscv: CC = riscv64-linux-gnu-gcc
cross-riscv: CFLAGS += -march=rv64gc -mabi=lp64d
cross-riscv: LDFLAGS += -static
cross-riscv: $(TARGET)

# --- Mac OS X 10.4 Tiger (PowerPC G4, GCC 4.0.1, Altivec) ---
# Requires: gcc -maltivec, OSSwapInt* from libkern/OSByteOrder.h
# No OpenMP, no C11, no static linking, no libstdc++
ppc: CFLAGS = -O2 -g -std=c99 -D_DARWIN_C_SOURCE -D_GNU_SOURCE -Wall -maltivec
ppc: LDFLAGS = -lm
ppc: $(TARGET)

# --- Mac OS X 10.6 Snow Leopard (Core 2 Duo, GCC 4.2.1, SSE4.1) ---
# GCC 4.2: no -march=native, no -std=c11, no -Wpedantic, no <immintrin.h>
# SSE3 intrinsics are broken in GCC 4.2's <emmintrin.h>; SSSE3+SSE4.1 work
# via individual headers. __thread not supported.
mac106: CFLAGS = -O2 -g -std=c99 -D_GNU_SOURCE -Wall -msse4.1 -mssse3 -msse2
mac106: LDFLAGS = -lm -lpthread -fopenmp
mac106: $(TARGET)

# --- Debug build ---
debug: CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -g -O0
debug: $(TARGET)

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
	rm -f $(TARGET) $(TARGET).exe $(OBJS) $(VIZ_OBJS) backend_gpu.o backend_gpu.o.hip backend_gpu.o.cuda backend_metal.o probes/metal_reduction_probe probes/metal_matmul_bench


.PHONY: asan
asan: CFLAGS += -fsanitize=address -g -O2 -march=native
asan: LDFLAGS += -fsanitize=address
asan: picolm

# --- Force recompile of GPU-dependent C objects ---
# These objects contain #ifdef PICOLM_GPU code; when building GPU targets
# alongside a prior native build, Make won't recompile them because
# CFLAGS is a variable, not a prerequisite. Remove them so the sub-make
# with -DPICOLM_GPU=1 rebuilds them from scratch.
.PHONY: gpu-clean-objs
gpu-clean-objs:
	rm -f model.o tensor.o quant.o picolm.o

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

.PHONY: hip
hip: backend_gpu.o.hip gpu-clean-objs
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -DPICOLM_GPU=1" GPU_OBJS="backend_gpu.o.hip" GPU_LIBS="-L$(ROCM_LIB) -lamdhip64 -lstdc++ -Wl,-rpath,$(ROCM_LIB)"

backend_gpu.o.hip: backend_gpu.cu backend_gpu.h quant.h
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(HIP_ARCH) -DPICOLM_GPU=1 -c -o $@ $< -I.

# --- CUDA (NVIDIA) GPU backend ---
NVCC      ?= nvcc
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

CUDAFLAGS   ?= -O3 -std=c++11 -Xcompiler="-fPIC"

.PHONY: cuda
cuda: backend_gpu.o.cuda gpu-clean-objs
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -DPICOLM_GPU=1" GPU_OBJS="backend_gpu.o.cuda" GPU_LIBS="-L$(CUDA_LIB) -lcudart -lstdc++ -Wl,-rpath,$(CUDA_LIB)"

backend_gpu.o.cuda: backend_gpu.cu backend_gpu.h quant.h
	$(NVCC) $(CUDAFLAGS) $(CUDA_GENCODE) -DPICOLM_GPU=1 -c -o $@ $< -I.

# --- Metal (Apple Silicon) backend -- RAW Metal framework, no MLX-C ---
# macOS only. No external dependencies beyond the Metal framework that ships
# with macOS + the Command Line Tools (clang). Compile the whole backend with:
#     make metal
# Then run with: PICOLM_GPU=1 ./picolm model.gguf ...
METAL_CXX      ?= clang++
METAL_CXXFLAGS ?= -std=c++20 -O3 -fobjc-arc -fPIC
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
.PHONY: hip cuda metal hunger

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
HUNGER_CFILES = picolm.c model.c tensor.c quant.c tokenizer.c qwen_tokenize.c sampler.c grammar.c cJSON.c server.c csafetensors.c json.c safetensors.c
HUNGER_COBJS = $(HUNGER_CFILES:.c=.obj)
# cl.exe flags - all Make vars, NO batch %VAR% references
HUNGER_CF = /O2 /W3 /wd4996 /arch:AVX512 /D PICOLM_GPU=1 /D _CRT_SECURE_NO_WARNINGS /I. /openmp /c

hunger:
	echo @echo off > .hunger_build.bat
	echo del /q picolm.exe backend_gpu_cuda_win.obj $(HUNGER_COBJS) 2^>nul >> .hunger_build.bat
	echo call "$(HUNGER_VS_ROOT)/VC/Auxiliary/Build/vcvars64.bat" >> .hunger_build.bat
	echo "$(HUNGER_CUDA_ROOT)/bin/nvcc.exe" -O3 -std=c++14 -DPICOLM_GPU=1 -ccbin $(HUNGER_CL) -Xcompiler "/EHsc /O2 /W3 /std:c++14 /arch:AVX512" -gencode arch=compute_89,code=$(HUNGER_CUDA_ARCH) -c -o backend_gpu_cuda_win.obj backend_gpu.cu -I. >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) picolm.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) model.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) tensor.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) quant.c >> .hunger_build.bat
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
	echo $(HUNGER_LNK) /OUT:picolm.exe $(HUNGER_COBJS) backend_gpu_cuda_win.obj "$(HUNGER_CUDA_ROOT)/lib/x64/cudart.lib" ws2_32.lib advapi32.lib >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo del /q backend_gpu_cuda_win.obj 2^>nul >> .hunger_build.bat
	echo del .hunger_build.bat >> .hunger_build.bat
	cmd /c .hunger_build.bat
