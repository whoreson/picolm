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

# Apple Clang doesn't support -fopenmp. Detect Darwin and strip it.
# Also: _GNU_SOURCE alone doesn't enable POSIX symbols on macOS;
# _DARWIN_C_SOURCE is needed (e.g. for signal(), fork(), etc.).
ifeq ($(shell uname -s 2>/dev/null),Darwin)
OPENMP_FLAGS =
DARWIN_FLAGS = -D_DARWIN_C_SOURCE
else
OPENMP_FLAGS = -fopenmp
DARWIN_FLAGS =
endif

CFLAGS  ?= -O3 -g -std=c11 -D_GNU_SOURCE $(DARWIN_FLAGS) -Wall -Wextra -Wpedantic -fno-common $(OPENMP_FLAGS) -DPICOLM_SERVER
LDFLAGS = -lm -lpthread $(OPENMP_FLAGS)

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
MODEL_SRCS = model_core.c model_attention.c model_moe.c model_gemma3n.c model_glm5.c model_ssm.c model_kv_cache.c model_gguf.c
SGEMM_SRCS = sgemm.c sgemm_q0_neon_fast.c sgemm_q2_k_r4_neon.c sgemm_iq4nl_neon_fast.c sgemm_k_q8k_neon_fast.c sgemm_q2_k_neon.c sgemm_k_q8k_neon.c sgemm_q4_0x8.c sgemm_q4i_0x8.c sgemm_q4_0_r8.c sgemm_q8_0_r8.c sgemm_q8_k_r8.c sgemm_q8_k_r8_neon.c sgemm_iq2_k.c sgemm_iq2_k_r4.c sgemm_iq3_k.c sgemm_iq3_k_r4.c sgemm_iq4_k.c sgemm_iq4_nl_r4.c sgemm_q4_k_r4.c sgemm_q6_k_r4.c sgemm_iq6_k.c sgemm_iq4_xs.c
SRCS    = picolm.c $(MODEL_SRCS) steer.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) tokenizer.c qwen_tokenize.c sampler.c grammar.c cJSON.c server.c csafetensors.c json.c safetensors.c
OBJS    = $(SRCS:.c=.o)
TARGET  = picolm

# VIZ support (adds viz.c when PICOLM_VIZ is defined)
VIZ_SRCS = viz.c
VIZ_OBJS = $(VIZ_SRCS:.c=.o)

# GPU backend
HIPFLAGS ?= -O3 -std=c++11 -fPIC -fPIE
HIP_LIBS = -lamdhip64
GPU_OBJS =

# GPU host .cu sources (split into modular files)
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

# --- AArch64 native-CPU workaround -----------------------------------------
# Termux/Android clang 21 maps -march=native for these ARMv9.2 phone SoCs to
# "generic +v9.2a", which enables SVE/SVE2 even though Cortex-A520/A720 do not
# implement SVE.  The resulting binary SIGILLs on its very first vector loop.
# If the compiler claims SVE but the host CPU does not advertise it, fall back
# to the NEON/DOTPROD/I8MM feature set that PicoLM's ARM kernels actually use.
# Keep this Linux-only: the probe relies on /proc/cpuinfo.
_host_os     := $(shell uname -s 2>/dev/null)
_native_sve  := $(shell echo | $(CC) -E -dM -march=native - 2>/dev/null | grep -c '__ARM_FEATURE_SVE ')
ifeq ($(_host_os),Linux)
  _host_sve    := $(shell grep -qwi sve /proc/cpuinfo 2>/dev/null && echo 1)
  _arm_i8mm    := $(shell grep -qwi i8mm /proc/cpuinfo 2>/dev/null && echo 1)
  _arm_bf16    := $(shell grep -qwi bf16 /proc/cpuinfo 2>/dev/null && echo 1)
  _arm_dotprod := $(shell grep -Eqwi 'dotprod|asimddp' /proc/cpuinfo 2>/dev/null && echo 1)
else
  _host_sve    := native
endif
ifeq ($(filter 1,$(_native_sve)),1)
  ifeq ($(_host_sve),)
    _arm_base := $(if $(_arm_i8mm),armv8.6-a,$(if $(_arm_dotprod),armv8.2-a,armv8-a))
    NATIVE_MARCH := -march=$(_arm_base)$(if $(_arm_dotprod),+dotprod,)$(if $(_arm_i8mm),+i8mm,)$(if $(_arm_bf16),+bf16,)
    $(info NOTE: -march=native requested SVE on a CPU without SVE; using $(NATIVE_MARCH))
  else
    NATIVE_MARCH := -march=native
  endif
else
  NATIVE_MARCH := -march=native
endif

# Pick the highest SIMD level and emit the right -DPICOLM_* preprocessor defines.
# Cascading if/elif/else: only the first (highest) matching level is emitted.
simd_defines_for = $(strip \
  $(if $(filter 1,$(call _have_avx512,$(1))),\
    -DPICOLM_AVX512 -DPICOLM_AVX2 -DPICOLM_AVX -DPICOLM_SSSE3 -DPICOLM_SSE3 -DPICOLM_SSE2 -DPICOLM_FMA,\
    $(if $(filter 1,$(call _have_avx2,$(1))),\
      -DPICOLM_AVX2 -DPICOLM_AVX -DPICOLM_SSSE3 -DPICOLM_SSE3 -DPICOLM_SSE2 -DPICOLM_FMA,\
      $(if $(filter 1,$(call _have_avx,$(1))),\
        -DPICOLM_AVX -DPICOLM_SSSE3 -DPICOLM_SSE3 -DPICOLM_SSE2,\
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

# --- Default target: build with NATIVE_MARCH + auto-detected SIMD ---
.DEFAULT_GOAL := all
.PHONY: all
all:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) $(NATIVE_MARCH) $(call simd_flags_for,$(NATIVE_MARCH)) $(call simd_defines_for,$(NATIVE_MARCH)) $(if $(filter 1,$(call _have_neon,$(NATIVE_MARCH))),-DPICOLM_NEON,) $(if $(filter 1,$(call _have_i8mm,$(NATIVE_MARCH))),-DPICOLM_I8MM,)"

# --- Native build: NATIVE_MARCH with auto-detected SIMD defines ---
native:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) $(NATIVE_MARCH) $(call simd_flags_for,$(NATIVE_MARCH)) $(call simd_defines_for,$(NATIVE_MARCH)) $(if $(filter 1,$(call _have_neon,$(NATIVE_MARCH))),-DPICOLM_NEON,) $(if $(filter 1,$(call _have_i8mm,$(NATIVE_MARCH))),-DPICOLM_I8MM,)"

# --- VIZ build: native with embedded VNC visualization server ---
# Remove affected .o files so the sub-make recompiles with -DPICOLM_VIZ.
# After model.c was split into modular files, we must also clean the model
# modules that contain #ifdef PICOLM_VIZ blocks (viz_push_layer calls).
# Uses $(RM) (Make built-in, defaults to `rm -f`) for cross-platform safety.
.PHONY: viz-clean-objs
viz-clean-objs:
	-$(RM) model.o picolm.o server.o model_core.o model_ssm.o model_attention.o model_gemma3n.o

viz: viz-clean-objs
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) $(NATIVE_MARCH) $(call simd_flags_for,$(NATIVE_MARCH)) $(call simd_defines_for,$(NATIVE_MARCH)) $(if $(filter 1,$(call _have_neon,$(NATIVE_MARCH))),-DPICOLM_NEON,) $(if $(filter 1,$(call _have_i8mm,$(NATIVE_MARCH))),-DPICOLM_I8MM,) -DPICOLM_VIZ=1 -DPICOLM_SERVER" OBJS="$(OBJS) viz.o"

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
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -mavx -mf16c -mpopcnt -DPICOLM_AVX -DPICOLM_SSE2 -DPICOLM_SSE3 -DPICOLM_SSSE3"

# --- x86-64 with AVX2 (Haswell and newer Intel; Excavator and newer AMD) ---
avx2:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) -mavx2 -mfma -mpopcnt -mf16c -DPICOLM_AVX2 -DPICOLM_FMA -DPICOLM_AVX -DPICOLM_SSSE3 -DPICOLM_SSE3 -DPICOLM_SSE2"

# --- Persistent thread pool (WIP, for development) ---
threadpool:
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) $(NATIVE_MARCH) $(call simd_defines_for,$(NATIVE_MARCH)) -DPICOLM_USE_PER_MATMUL_THREAD=0"

# --- Link ---
$(TARGET): $(OBJS) $(GPU_OBJS)
ifeq ($(OSF1_LINK),1)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(GPU_LIBS)
else
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(GPU_LIBS) -lstdc++
endif

# --- Compile each .c to .o independently ---
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Header dependencies
picolm.o: picolm.c quant.h tensor.h model.h tokenizer.h sampler.h grammar.h steer.h
steer.o: steer.c steer.h model.h quant.h
viz.o:  viz.c  viz.h
model_core.o:   model_core.c  model.h quant.h tensor.h model_internal.h steer.h
model_gguf.o:   model_gguf.c  model.h quant.h tensor.h model_internal.h
model_attention.o: model_attention.c  model.h quant.h tensor.h model_internal.h
model_moe.o:    model_moe.c  model.h quant.h tensor.h model_internal.h
model_gemma3n.o: model_gemma3n.c  model.h quant.h tensor.h model_internal.h
model_glm5.o: model_glm5.c  model.h quant.h tensor.h model_internal.h
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

# --- Pi 3/4/5 (ARM with NEON SIMD) ---
# -mfpu=neon-vfpv4 needed on 32-bit ARM (Pi 3). 64-bit ARM ignores it, so
# we detect the target arch at build time.
pi:
	@pi_arch=$$($(CC) -dumpmachine); \
	neon_flags="-march=armv8-a+simd -DPICOLM_NEON"; \
	case "$$pi_arch" in *armv7*) neon_flags="$$neon_flags -mfpu=neon-vfpv4";; esac; \
	echo "  PI_ARCH $$pi_arch"; \
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) $$neon_flags"

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

iphoneos1:
	$(MAKE) $(TARGET) CC=/usr/local/arm-apple-darwin-ios1/bin/arm-apple-darwin-gcc CFLAGS="-O2 -g -std=c99 -D_DARWIN_C_SOURCE -D_GNU_SOURCE -Wall -arch armv6" LDFLAGS="-lm"

# --- OSF/1 Tru64 UNIX (DEC Alpha, GCC 3.3.6) ---
# Scalar only. compat/ has portable snprintf + fmaf/atoll/roundf/memlk stubs.
osf: compat/snprintf_portable.o compat/snprintf_osf1_supplement.o
	$(MAKE) $(TARGET) CFLAGS="-O2 -g -std=c99 -D_DARWIN_C_SOURCE -D_GNU_SOURCE -Wall -mieee" LDFLAGS="-lm -lc_r -lpthread" OBJS="$(OBJS) compat/snprintf_portable.o compat/snprintf_osf1_supplement.o" OSF1_LINK=1

compat/snprintf_portable.o: compat/snprintf_portable_src.c compat/osf1_compat.h
	$(CC) -O2 -g -std=c99 -D_DARWIN_C_SOURCE -D_GNU_SOURCE -Wall -DUSE_SNPRINTF -DHAVE_SNPRINTF -DPREFER_PORTABLE_SNPRINTF -c -o $@ $<

compat/snprintf_osf1_supplement.o: compat/snprintf_osf1_supplement.c compat/osf1_compat.h
	$(CC) -O2 -g -std=c99 -D_DARWIN_C_SOURCE -D_GNU_SOURCE -Wall -c -o $@ $<

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
	-$(RM) $(TARGET) $(TARGET).exe $(OBJS) $(VIZ_OBJS) backend_metal.o backend_vulkan.o
	-$(RM) $(GPU_SRCS:%.cu=%-hip.o) $(GPU_SRCS:%.cu=%-cuda.o) backend_gpu.dlink.o
	-$(RM) shaders/*.spv
	-$(RM) test_swa test_swa.exe test_quant test_attn_tiled test_attn_tiled.exe test_attn_diff test_attn_diff.exe
	-$(RM) tests/test_glm5_ops tests/test_glm5_ops.exe test_glm5_dsa test_glm5_snapshot

# --- GLM-5 operator unit tests (standalone, no model file needed) ---
# Re-derives hc_pre / KDA / conv1d / MoE-router with plain scalar code and
# checks the algebraic invariants. Catches formula and indexing bugs without
# needing the oracle.
.PHONY: test-glm5-ops
test-glm5-ops: tests/test_glm5_ops
	./tests/test_glm5_ops

tests/test_glm5_ops: tests/test_glm5_ops.c
	$(CC) -std=c11 -O2 -g -D_GNU_SOURCE -Wall -Wextra -o $@ $< -lm

.PHONY: asan
asan:
	$(MAKE) $(TARGET) CFLAGS="-fsanitize=address -g -O2 $(NATIVE_MARCH) $(call simd_defines_for,$(NATIVE_MARCH)) -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wpedantic -fopenmp" LDFLAGS="-lm -lpthread -fopenmp -fsanitize=address"

# check-all under ASan+UBSan. The bug classes seen most often in this codebase
# are out-of-bounds writes into scratch buffers carved from one malloc and
# signed-overflow / misaligned-store UB in the quantisation paths -- note that a
# plain heap overflow IS caught here but an overrun into a *sibling* region of
# the same allocation is not (see TODO(stride-invariants) in model_core.c); this
# target still catches the first kind and all the UB.
# ASAN_OPTIONS/UBSAN_OPTIONS are exported so the sub-make's children inherit
# them. detect_leaks=0 because the daemon paths (server, mmap'd model) leak by
# design at exit and would mask a real failure.
SAN_FLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1
.PHONY: check-all-san
check-all-san:
	$(MAKE) clean
	ASAN_OPTIONS=detect_leaks=0 \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	$(MAKE) check-all -j1 CC=$(CC) \
	  TEST_NATIVE_CFLAGS="$(CFLAGS) $(NATIVE_MARCH) $(call simd_flags_for,$(NATIVE_MARCH)) $(call simd_defines_for,$(NATIVE_MARCH)) -std=c11 -D_GNU_SOURCE -fopenmp $(SAN_FLAGS)"
	@echo ""
	@echo "=== check-all-san: clean under ASan+UBSan (leaks not checked) ===" 

# --- Force recompile of GPU-dependent C objects ---
# These objects contain #ifdef PICOLM_GPU code; when building GPU targets
# alongside a prior native build, Make won't recompile them because
# CFLAGS is a variable, not a prerequisite. Remove them so the sub-make
# with -DPICOLM_GPU=1 rebuilds them from scratch.
.PHONY: gpu-clean-objs
gpu-clean-objs:
	-$(RM) model.o tensor.o quant.o picolm.o
	-$(RM) model_core.o model_attention.o model_moe.o model_gemma3n.o model_glm5.o model_ssm.o model_kv_cache.o model_gguf.o

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
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) $(NATIVE_MARCH) $(call simd_defines_for,$(NATIVE_MARCH)) -DPICOLM_GPU=1 -DPICOLM_HIP=1" GPU_OBJS="$(HIP_OBJS)" GPU_LIBS="-L$(ROCM_LIB) -lamdhip64 -lstdc++ -Wl,-rpath,$(ROCM_LIB)"

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
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS) $(NATIVE_MARCH) $(call simd_defines_for,$(NATIVE_MARCH)) -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -I$(CUDA_INC)" GPU_OBJS="$(CUDA_OBJS)" GPU_LIBS="-L$(CUDA_LIB) -lcudart -lstdc++ -Wl,-rpath,$(CUDA_LIB)"

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
VK_LIB      := $(shell for d in /opt/vulkan-sdk/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/vulkan /usr/lib; do \
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
               else echo "-lvulkan"; fi),$(shell \
               if [ -f "$(VK_LIB)/libvulkan.a" ] || [ -f "$(VK_LIB)/libvulkan.so" ]; then echo "-L$(VK_LIB) -lvulkan"; \
               else echo "-lvulkan"; fi))
VK_CFLAGS   ?= -O3 -std=c11 -fPIC -Wall $(VK_EXTRA_INC)
VK_LDFLAGS  ?= $(VK_EXTRA_LIB)

shaders/%.spv: shaders/%.comp
	$(GLSLC) --target-env=vulkan1.2 $< -o $@

.PHONY: vulkan
vulkan: backend_vulkan.o shaders/qmatmul_vk.spv shaders/elementwise_vk.spv shaders/rmsnorm_vk.spv shaders/layernorm_vk.spv shaders/quantize_q8_vk.spv shaders/q8_q8_matmul_vk.spv shaders/attn_prefill_vk.spv shaders/attn_prefill_f16vk.spv shaders/attn_decode_vk.spv shaders/kv_store_vk.spv shaders/q8_q8_qkv3_vk.spv shaders/q8_q8_gu3_vk.spv shaders/q2k_matmul_vk.spv shaders/q3k_matmul_vk.spv shaders/q4k_matmul_vk.spv shaders/q6k_matmul_vk.spv shaders/iq4nl_matmul_vk.spv
	$(MAKE) $(TARGET) \
	  CFLAGS="$(filter-out -fopenmp,$(CFLAGS)) $(NATIVE_MARCH) -DPICOLM_GPU=1 -DPICOLM_VULKAN=1 $(call simd_flags_for,$(NATIVE_MARCH)) $(call simd_defines_for,$(NATIVE_MARCH)) $(VK_EXTRA_INC)" \
	  LDFLAGS="$(LDFLAGS) $(VK_EXTRA_LIB)" \
	  GPU_OBJS="backend_vulkan.o"

backend_vulkan.o: backend_vulkan.c backend_gpu.h quant.h
	$(VK_CC) $(VK_CFLAGS) -DPICOLM_GPU=1 -DPICOLM_VULKAN=1 -c -o $@ $< -I.

.PHONY: vulkan-shaders
vulkan-shaders: shaders/qmatmul_vk.spv shaders/elementwise_vk.spv shaders/rmsnorm_vk.spv shaders/quantize_q8_vk.spv shaders/q8_q8_matmul_vk.spv shaders/attn_prefill_vk.spv shaders/attn_prefill_f16vk.spv shaders/attn_decode_vk.spv shaders/kv_store_vk.spv shaders/q8_q8_qkv3_vk.spv shaders/q8_q8_gu3_vk.spv shaders/q2k_matmul_vk.spv shaders/q3k_matmul_vk.spv shaders/q4k_matmul_vk.spv shaders/q6k_matmul_vk.spv shaders/iq4nl_matmul_vk.spv

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
HUNGER_CF = /O2 /W3 $(HUNGER_WD) /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D PICOLM_SERVER=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"$(HUNGER_CUDA_ROOT)/include" /openmp /c $(HUNGER_SIMD)
# hunger-nogpu: same as hunger but without GPU backend (pure MSVC, no nvcc, no cudart)
HUNGER_CF_NOGPU = /O2 /W3 $(HUNGER_WD) /arch:AVX512 /D PICOLM_SERVER=1 /D _CRT_SECURE_NO_WARNINGS /I. /openmp /c $(HUNGER_SIMD)

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
	echo $(HUNGER_CL) $(HUNGER_CF) model_glm5.c >> .hunger_build.bat
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
	echo $(HUNGER_CL) $(HUNGER_CF) sgemm_q4_0x8.c >> .hunger_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF) sgemm_q4i_0x8.c >> .hunger_build.bat
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
HUNGER_CL_FLAGS = /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX2 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D PICOLM_SERVER=1 /D _CRT_SECURE_NO_WARNINGS /D _CRT_AMD64 /I. /openmp /c /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1

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
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) sgemm_q4_0x8.c >> .hunger_nogpu_build.bat
	echo if errorlevel 1 exit /b 1 >> .hunger_nogpu_build.bat
	echo $(HUNGER_CL) $(HUNGER_CF_NOGPU) sgemm_q4i_0x8.c >> .hunger_nogpu_build.bat
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
# Derive from SRCS, excluding server.c (no networking on DOS)
DJGPP_SRCS = $(filter-out server.c,$(SRCS))
DJGPP_OBJS = $(DJGPP_SRCS:.c=.o.dos)
DJGPP_RUNTIME = /data4/work/djgpp-build/djgpp-runtime

%.o.dos: %.c model.h quant.h tensor.h model_internal.h
	$(DJGPP_CC) $(DJGPP_CFLAGS) -c -o $@ $<

dos_stubs.o.dos: dos_stubs.c
	$(DJGPP_CC) $(DJGPP_CFLAGS) -c -o $@ $<

dos: dos_stubs.o.dos $(DJGPP_OBJS)
	$(DJGPP_CC) -nostartfiles -o picolm.exe $< $(DJGPP_RUNTIME)/lib/crt0.o \
	 $(DJGPP_OBJS) -L$(DJGPP_RUNTIME)/lib -lc -lm -lgcc

# DOS test for SWA correctness. Reuses the already-built .o.dos objects from 'make dos'.
# Single-threaded on DOS (no OpenMP), so batch_attention_layer takes the
# tensor_parallel_for -> n_threads<=1 scalar loop path. Still exercises both
# attn_core (decode) and batch_attention_tiled (prefill, forced by n_tokens>=128).
TEST_SWA_OBJS_DOS = model_attention.o.dos tensor.o.dos quant.o.dos sgemm.o.dos sgemm_q4_0x8.o.dos sgemm_q4i_0x8.o.dos
test_swa_dos: tests/test_swa.c $(TEST_SWA_OBJS_DOS) dos_stubs.o.dos
	$(DJGPP_CC) $(DJGPP_CFLAGS) -I. -c -o tests/test_swa.o.dos tests/test_swa.c
	$(DJGPP_CC) -nostartfiles -o test_swa.exe dos_stubs.o.dos tests/test_swa.o.dos $(DJGPP_RUNTIME)/lib/crt0.o \
	 $(TEST_SWA_OBJS_DOS) -L$(DJGPP_RUNTIME)/lib -lc -lm -lgcc

dos-clean:
	-rm -f $(DJGPP_OBJS) dos_stubs.o.dos picolm.exe tests/test_swa.o.dos test_swa.exe

# test_quant lives in util/ now but still includes "quant.h" etc. from the repo
# root, so it needs -I. (the implicit search of the source file's own directory
# no longer covers it).
test_quant: util/test_quant.c quant.c sgemm.c
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -o $@ util/test_quant.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp

test_q6_k_r4: tests/test_q6_k_r4.c quant.c $(SGEMM_SRCS)
	$(CC) $(TEST_NATIVE_CFLAGS) -o $@ tests/test_q6_k_r4.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp

# Validate every ARM K-quant x Q8_K GEMM against its scalar vec_dot reference.
# The integer accumulation is exact, so any deviation is a real bug; this is
# the gate for changes to sgemm_k_q8k_neon_fast.c / sgemm_k_q8k_neon.c.
test_kq_gemm: tests/kq_gemm_check.c quant.c $(SGEMM_SRCS) type_dispatch.c tensor.c tokenizer.c
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -o $@ tests/kq_gemm_check.c quant.c $(SGEMM_SRCS) \
	    type_dispatch.c tensor.c tokenizer.c -lm -lpthread -fopenmp

.PHONY: check-kq-gemm
check-kq-gemm: test_kq_gemm
	./test_kq_gemm

# R8 SIMD vs dequantize reference: Q4_0_R8 and Q8_0_R8 GEMV+GEMM.
test_r8_simd: tests/test_r8_simd.c quant.c $(SGEMM_SRCS)
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -o $@ tests/test_r8_simd.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp

# BF16 SIMD vs scalar reference: vec_dot_bf16_f32 and dequantize_row_bf16.
test_bf16_simd: tests/test_bf16_simd.c quant.c $(SGEMM_SRCS)
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -o $@ tests/test_bf16_simd.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp

.PHONY: check-r8-simd
check-r8-simd: test_r8_simd
	./test_r8_simd

# SIMD flags for the standalone test binaries. 'make avx2' etc. re-invoke make
# with the SIMD flags in CFLAGS, but a later bare 'make check-all' does not
# inherit them -- so the tests would be compiled scalar while the binary under
# test is AVX2. Derive the flags from the target the user asked for, falling
# back to whatever CFLAGS already carries.
ifeq ($(filter scalar,$(MAKECMDGOALS)),scalar)
TEST_SIMD_CFLAGS =
else ifeq ($(filter sse3,$(MAKECMDGOALS)),sse3)
TEST_SIMD_CFLAGS = -msse2 -msse3 -mssse3 -mpopcnt -DPICOLM_SSE3 -DPICOLM_SSSE3 -DPICOLM_SSE2
else ifeq ($(filter avx,$(MAKECMDGOALS)),avx)
TEST_SIMD_CFLAGS = -mavx -mf16c -mpopcnt -DPICOLM_AVX -DPICOLM_SSE2 -DPICOLM_SSE3 -DPICOLM_SSSE3
else ifeq ($(filter avx2,$(MAKECMDGOALS)),avx2)
TEST_SIMD_CFLAGS = -mavx2 -mfma -mpopcnt -mf16c -DPICOLM_AVX2 -DPICOLM_FMA -DPICOLM_AVX -DPICOLM_SSSE3 -DPICOLM_SSE3 -DPICOLM_SSE2
else ifneq ($(NATIVE_MARCH),)
TEST_SIMD_CFLAGS = $(NATIVE_MARCH) $(call simd_flags_for,$(NATIVE_MARCH)) $(call simd_defines_for,$(NATIVE_MARCH))
else
TEST_SIMD_CFLAGS =
endif

# Row-interleaved (R4/R8) layout regression. Model-free, sub-second. Catches the
# class of bug where code assumes a row-major layout for an interleaved type:
# reads only row 0 of a block group, walks rows by a single row stride, or
# decodes nibbles with the wrong interleave. See the DEBUG_JOURNAL in
# notes/picolm/ for the two shipped instances (IQ4_NL_R4 ssm_alpha/beta NaN, and
# the R8 prefill repack). Without SIMD flags the R4 GEMM does not exist and the
# GEMM half degrades to "skipped", so TEST_SIMD_CFLAGS above matters.
test_rowinterleave: tests/test_rowinterleave.c quant.c $(SGEMM_SRCS)
	$(CC) $(CFLAGS) $(TEST_SIMD_CFLAGS) -I. -o $@ tests/test_rowinterleave.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp

# Isolates picolm_sgemm_r8dec_avx2t against a dequantize-and-dot reference, so a
# disagreement can be attributed to a specific atype/repack path without a model,
# the SSM code, or the oracle's text hashing in the way.
test_r8dec_ref: tests/test_r8dec_ref.c quant.c $(SGEMM_SRCS)
	$(CC) $(CFLAGS) $(TEST_SIMD_CFLAGS) -I. -o $@ tests/test_r8dec_ref.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp

test_swa: tests/test_swa.c model_attention.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS)
	$(CC) $(CFLAGS) -I. -O2 -o $@ tests/test_swa.c model_attention.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

# Tiled-attention acc-update tail, for every head_dim (model-free, all ISAs).
# Pins the contract that attn_process_tile()'s acc += exp@V update touches
# exactly [acc_row, acc_row+head_dim) and nothing else. The AVX-512 branch
# used ceil(hd/16) blocks, so head_dim=8 wrote 16 floats per 8-float row and
# clobbered the next query row's accumulator -- silently, and only on the
# default `make` build. CI builds avx2, so this must not depend on AVX-512
# being present: the sentinel check fails on any ISA that overruns.
test_attn_tiled_tail: tests/test_attn_tiled_tail.c
	$(CC) $(CFLAGS) $(TEST_NATIVE_CFLAGS) -I. -O2 -o $@ $< -lm

.PHONY: check-attn-tail
check-attn-tail: test_attn_tiled_tail
	./test_attn_tiled_tail

test_attn_tiled: tests/test_attn_tiled.c model_attention.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS)
	$(CC) $(CFLAGS) -I. -O2 -o $@ tests/test_attn_tiled.c model_attention.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

# Same flag construction as 'make native', so the test exercises the production
# SIMD kernels (AVX-512/AVX2/NEON) instead of the scalar fallback.
TEST_NATIVE_CFLAGS = $(CFLAGS) $(NATIVE_MARCH) $(call simd_flags_for,$(NATIVE_MARCH)) $(call simd_defines_for,$(NATIVE_MARCH)) $(if $(filter 1,$(call _have_neon,$(NATIVE_MARCH))),-DPICOLM_NEON,) $(if $(filter 1,$(call _have_i8mm,$(NATIVE_MARCH))),-DPICOLM_I8MM,)

# Depends on the real sources (like test_swa) so it can never be stale.
test_attn_diff: test_attn_diff.c model_attention.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) model.h model_internal.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O2 -o $@ test_attn_diff.c model_attention.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

# GLM-5 hybrid checkpoint round-trip (MLA latent caches + KDA states).
test_glm5_snapshot: tests/test_glm5_snapshot.c model_glm5.c model_moe.c model_kv_cache.c model_ssm.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) model.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O2 -o $@ tests/test_glm5_snapshot.c model_glm5.c model_moe.c model_kv_cache.c model_ssm.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

# DSA k-pool selector invariants (causal, bounded, sorted, deduped).
test_glm5_dsa: tests/test_glm5_dsa.c model_glm5.c model_moe.c model_ssm.c model_kv_cache.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) model.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O2 -o $@ tests/test_glm5_dsa.c model_glm5.c model_moe.c model_ssm.c model_kv_cache.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

# kvcache_save() token-writer chunking (standalone; no model needed).
# Built with ASan+UBSan because the bug it guards was an overrun of a
# fixed alloca buffer and a misaligned store.
test_kvcache_tokwrite: tests/test_kvcache_tokwrite.c
	$(CC) -O2 -g -fsanitize=address,undefined -o $@ tests/test_kvcache_tokwrite.c

.PHONY: check-kvcache-tokwrite
check-kvcache-tokwrite: test_kvcache_tokwrite
	./test_kvcache_tokwrite

# Horizontal SIMD reduction regression. Catches the hreduce256_ps bug that made
# AVX2 builds compute 0.75x SSM dot products (lanes 3 and 7 dropped, lane 0
# double-counted). Built for AVX2 specifically -- the AVX-512 build never used
# this helper.
test_hreduce: tests/test_hreduce.c model.h model_internal.h
	$(CC) $(TEST_NATIVE_CFLAGS) -O2 -I. -o $@ $< -lm

.PHONY: check-hreduce
check-hreduce:
ifeq ($(shell uname -m),x86_64)
	$(MAKE) test_hreduce
	./test_hreduce
else
	@echo "SKIP check-hreduce (x86_64 only)"
endif

.PHONY: check-glm5-dsa
check-glm5-dsa: test_glm5_dsa
	./test_glm5_dsa

.PHONY: check-glm5-snapshot
check-glm5-snapshot: test_glm5_snapshot
	./test_glm5_snapshot

# Tiled-vs-grouped attention regression sweep (rebuilds the test first).
.PHONY: check-attn
check-attn: test_attn_diff
	./tests/test_attn_diff_sweep.sh ./test_attn_diff

# Decode attention (blocked softmax + split-KV) vs reference: correctness + timing.
test_attn_decode: test_attn_decode.c model_attention.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) model.h model_internal.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O3 -o $@ test_attn_decode.c model_attention.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

.PHONY: check-attn-decode
check-attn-decode: test_attn_decode
	./tests/test_attn_decode_sweep.sh ./test_attn_decode

# Thread-pool stress test (spin-then-sleep barrier): thread counts x spin budgets,
# verifies every dispatch; TSAN=1 additionally builds/runs it under ThreadSanitizer.
test_pool_stress: test_pool_stress.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) tensor.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O2 -o $@ test_pool_stress.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

.PHONY: check-pool
check-pool: test_pool_stress
	./tests/test_pool_stress_sweep.sh ./test_pool_stress
ifdef TSAN
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O2 -fsanitize=thread -o test_pool_stress_tsan test_pool_stress.c tensor.c type_dispatch.c quant.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++
	for c in "3 5" "4 0" "8 50" "24 500"; do set -- $$c; PICOLM_POOL_SPIN_US=$$2 ./test_pool_stress_tsan $$1 600 7 2>&1 | grep -q "WARNING: ThreadSanitizer" && { echo "TSAN warning nt=$$1 spin=$$2"; exit 1; }; done; echo "tsan clean"
endif

# ---------------------------------------------------------------------------
# check-all: every self-contained regression suite that needs no model file.
#
# This is the list CI runs (see .github/workflows/ci.yml), so local and CI stay
# in sync -- add new suites here rather than in the workflow.
#
# Deliberately excluded:
#   - anything needing a GGUF (model-dependent; see tests/README-model-matrix.md)
#   - GPU backends (need nvcc / Vulkan SDK / Apple toolchain; see TODO(backend-ci))
# ---------------------------------------------------------------------------
.PHONY: check-q6-k-r4
check-q6-k-r4: test_q6_k_r4
	./test_q6_k_r4

.PHONY: check-quant
check-quant: test_quant
	./test_quant

.PHONY: check-rowinterleave
check-rowinterleave: test_rowinterleave
	./test_rowinterleave

.PHONY: check-all
# check-simd-diff is deliberately NOT in check-all: it builds a second
# (scalar) picolm, which would clash with check-all-san's sanitizer build and
# roughly double check-all's cost. CI runs it as its own step.
check-all: check-hreduce check-kvcache-tokwrite check-quant check-q6-k-r4 \
           check-attn-tail \
           check-rowinterleave check-r8-simd \
           check-glm5-dsa check-glm5-snapshot check-attn check-attn-decode check-pool
	@echo ""
	@echo "=== check-all: all self-contained suites passed ==="

# ---------------------------------------------------------------------------
# check-simd-diff: the SIMD build and a scalar build must agree, byte for byte.
#
# This is the check that found the zeroed-R8-kernels bug: vec_dot_q4_0_r8_q8_2_avx2
# and vec_dot_q8_0_r8_q8_0_avx2 returned zeros when AVX2 was absent, so models
# carrying Q4_0_R8 / Q8_0_R8 decoded to garbage on a scalar build while the SIMD
# build was fine. Nothing crashed and nothing warned.
#
# Unlike check-oracle, the scalar build is REQUIRED here, not optional -- an
# optional comparison is how that bug survived. Build it with `make
# build-scalar-ref` (also done automatically by this target).
# ---------------------------------------------------------------------------
.PHONY: check-simd-diff
check-simd-diff:
	@test -x ./picolm || { echo "error: build ./picolm first"; exit 2; }
	@# Compare against a freshly built scalar reference. A STALE picolm_scalar is
	@# the dangerous case, not a missing one: it would test yesterday's code and
	@# pass while today's is broken. So always rebuild -- it is a single cheap
	@# target, and correctness here is worth more than the seconds.
	@$(MAKE) --no-print-directory build-scalar-ref
	@PICOLM=./picolm SCALAR=./picolm_scalar tools/simd_diff.sh

# ---------------------------------------------------------------------------
# check-oracle: differential test on a generated dummy-oracle model.
#
# A random-weight model produces gibberish, but the gibberish must be the SAME
# gibberish on every code path. Compares the fast paths against their own
# fallbacks, and the SIMD build against a scalar build. Needs gguf-py
# (pip install gguf) to generate the model; skipped with a note if absent.
#
# This is the only test that covers the full model pipeline end to end with
# interleaved (type 220) weights, including the SSM alpha/beta per-head row
# walk and the R8 prefill GEMM that the two shipped IQ4_NL_R4 bugs lived in.
# ---------------------------------------------------------------------------
ORACLE_MODEL ?= /tmp/qwen35_tiny_r4.gguf

.PHONY: oracle-model
oracle-model:
	@python3 -c 'import gguf' 2>/dev/null || { \
		echo "SKIP: gguf-py not installed (pip install gguf)"; exit 1; }
	python3 tools/mk_qwen35_tiny_r4.py -o $(ORACLE_MODEL)

.PHONY: check-oracle
check-oracle:
	@python3 -c 'import gguf' 2>/dev/null || { \
		echo "SKIP check-oracle: gguf-py not installed (pip install gguf)"; \
		exit 0; }
	@test -f $(ORACLE_MODEL) || $(MAKE) oracle-model
	@test -x ./picolm || { echo "error: build ./picolm first"; exit 2; }
	@if [ -x ./picolm_scalar ]; then \
		PICOLM=./picolm SCALAR=./picolm_scalar tools/oracle_check.sh -m $(ORACLE_MODEL); \
	else \
		echo "NOTE: ./picolm_scalar not built -- SIMD-vs-scalar comparison skipped."; \
		echo "      Run 'make build-scalar-ref' to cover the interleaved-row path."; \
		PICOLM=./picolm tools/oracle_check.sh -m $(ORACLE_MODEL); \
	fi

# Scalar reference binary used by check-oracle for the SIMD-vs-scalar diff.
# Built in its own object directory: the main .o files are compiled with SIMD
# flags and reusing them would silently produce another SIMD binary.
.PHONY: build-scalar-ref
build-scalar-ref:
	@echo "building scalar reference in build-scalar/ ..."
	@rm -rf build-scalar && mkdir -p build-scalar
	@cp $(SRCS) build-scalar/
	@cp *.h build-scalar/ 2>/dev/null || true
	@cp -r compat build-scalar/ 2>/dev/null || true
	@cp Makefile build-scalar/
	@$(MAKE) -C build-scalar TARGET=picolm_scalar \
		CFLAGS="$(CFLAGS) -mno-sse2 -mno-avx" picolm_scalar
	@cp build-scalar/picolm_scalar ./picolm_scalar
	@echo "built ./picolm_scalar (scalar reference)"

# ---------------------------------------------------------------------------
# check-r4-model: real-text models that exercise the quantization kernels.
#
# Two COMMITTED models, both ~300 KB and both verified coherent against ik's
# llama-cli so they are a real differential fixture and not just a
# self-consistency check:
#
#   models/stories260K-r4.gguf       7 distinct types, 16 of them type 220.
#                                    The R4/R8 prefill GEMM gets real coverage.
#   models/stories260K-stuffed.gguf 11 distinct types spread across the 32
#                                    quantized slots (q4_1, q5_0, q5_1, q6_0,
#                                    iq4_nl, q8_0, q4_0, q4_0_r8, iq4_nl_r4,
#                                    F16, F32).
#
# They are checked in on purpose: they cannot be rebuilt in CI, because that
# needs stories260K-f32.gguf plus a working llama-quantize, and reproducing the
# exact type mix requires ik_llama's `--custom-q`. Committing 300 KB each is far
# cheaper than that dependency.
#
# `make stories-r4-model` regenerates them (byte-identical) when ik_llama and
# the f32 source are available; see tools/mk_stories_r4.sh for why the mixes are
# what they are.
#
# MIN_TYPES is enforced, not advisory. A narrowing fixture is a silent coverage
# loss: an earlier 11-type build had q4_1 on a single tensor, and deleting
# Q4_1's kernel from the dispatch table did not change the output by one byte.
#
# Complements check-oracle (generated, random weights, gibberish is fine) with
# models that must actually say plausible English *and* carry the expected mix.
# ---------------------------------------------------------------------------
STORIES_F32 ?= $(HOME)/llm/stories260K-f32.gguf
QUANTIZE    ?= $(HOME)/iklcpp-2026-10-28/llama-quantize
# llama-quantize is routinely built against a newer libstdc++ than the system
# runtime; without this it fails with GLIBCXX/CXXABI version errors.
QUANTIZE_LIBS ?= $(HOME)/gcc-15/lib64:$(HOME)/ik_llama.cpp/build/src:$(HOME)/ik_llama.cpp/build/ggml/src
R4_MODEL    ?= ../models/stories260K-r4.gguf
STUFFED_MODEL ?= ../models/stories260K-stuffed.gguf
# Distinct tensor-type floor for each fixture. Bump deliberately; never lower.
R4_MIN_TYPES     ?= 7
STUFFED_MIN_TYPES ?= 12

.PHONY: check-r4-model
check-r4-model:
	@test -x ./picolm || { echo "error: build ./picolm first"; exit 2; }
	@test -f "$(R4_MODEL)" || { \
		echo "error: $(R4_MODEL) missing. It is committed; restore it with"; \
		echo "       git checkout -- models/stories260K-r4.gguf"; exit 2; }
	@echo "--- $(R4_MODEL) (expect >= $(R4_MIN_TYPES) distinct types)"
	@MIN_TYPES=$(R4_MIN_TYPES) tests/check_r4_model.sh "$(R4_MODEL)" ./picolm
	@test -f "$(STUFFED_MODEL)" || { \
		echo "error: $(STUFFED_MODEL) missing. It is committed; restore it with"; \
		echo "       git checkout -- models/stories260K-stuffed.gguf"; exit 2; }
	@echo "--- $(STUFFED_MODEL) (expect >= $(STUFFED_MIN_TYPES) distinct types)"
	@MIN_TYPES=$(STUFFED_MIN_TYPES) tests/check_r4_model.sh "$(STUFFED_MODEL)" ./picolm

# Regenerate the committed model from source. Needs ik_llama's llama-quantize
# and stories260K-f32.gguf; skips with a note if either is absent.
.PHONY: stories-r4-model
stories-r4-model:
	@test -f "$(STORIES_F32)" || { \
		echo "SKIP: $(STORIES_F32) not found (set STORIES_F32=...)"; exit 1; }
	@test -x "$(QUANTIZE)" || { \
		echo "SKIP: $(QUANTIZE) not found (set QUANTIZE=...)"; exit 1; }
	F32="$(STORIES_F32)" QUANTIZE="$(QUANTIZE)" \
		QUANTIZE_LIBS="$(QUANTIZE_LIBS)" \
		tools/mk_stories_r4.sh -o "$(R4_MODEL)"

gguf_be_swap: util/gguf_be_swap.c
	$(CC) -O2 -std=c99 -o $@ util/gguf_be_swap.c -lm

gguf_q40_i8_convert: util/gguf_q40_i8_convert.c
	$(CC) -O2 -std=c99 -o $@ util/gguf_q40_i8_convert.c -lm

# Q8_0 prefill GEMM benchmark: current picolm_sgemm_d vs the Q8_0_R8 AVX2 kernel vs raw VNNI peak.
# usage: ./bench_q8_gemm [m n k reps threads]   e.g.  ./bench_q8_gemm 14336 512 4096 3 24
bench_q8_gemm: bench_q8_gemm.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) quant.h sgemm.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O3 -o $@ bench_q8_gemm.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

# Legacy-quant (Q8_0/Q4_0/IQ4_NL/Q5_0) prefill GEMM + decode dot bench and check.
# usage: ./bench_qgemm check ;  ./bench_qgemm iq4_nl 4096 512 4096 3 1 ;  ./bench_qgemm all
bench_qgemm: bench_qgemm.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) quant.h sgemm.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O3 -o $@ bench_qgemm.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

# IQ6_K prefill GEMM: decode-once R8 kernel vs the 1x2 tile kernel (correctness + GOPS).
# usage: ./bench_iq6k_gemm check ;  ./bench_iq6k_gemm 14336 512 4096 3 24
bench_iq6k_gemm: bench_iq6k_gemm.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) quant.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O3 -o $@ bench_iq6k_gemm.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

# K-quant baseline and Q6_K/Q3_K table-GEMM check: ./bench_kq [m n k reps] ;  ./bench_kq check
bench_kq: bench_kq.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) quant.h sgemm.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O3 -o $@ bench_kq.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

# Prefill GEMM THREAD-SCALING probe (no model file needed). Runs one picolm_sgemm_d
# through the real thread pool at several -j values and reports per-thread GOPS,
# parallel efficiency and the cost of an empty barrier. Use it to tell a slow
# kernel apart from a thread-scaling problem (the Spark-X2.5 case).
# usage: ./probes/gemm_thread_bench ;  ./probes/gemm_thread_bench -m 6656 -n 103 -k 2048 -t q8_0 -j 24
#        ./probes/gemm_thread_bench -c          (correctness across thread counts)
.PHONY: gemm_thread_bench
probes/gemm_thread_bench: probes/gemm_thread_bench.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) quant.h sgemm.h tensor.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O3 -o $@ probes/gemm_thread_bench.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++
gemm_thread_bench: probes/gemm_thread_bench

# ik-style / R4 K-quant table GEMM timing: ./bench_ik [m n k reps]
bench_ik: bench_ik.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) quant.h sgemm.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O3 -o $@ bench_ik.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++

# decode (GEMV) cost per row for every quant type with a table fn_gemv: ./bench_gemv [rows k reps]
bench_gemv: bench_gemv.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) quant.h sgemm.h
	$(CC) $(TEST_NATIVE_CFLAGS) -I. -O3 -o $@ bench_gemv.c quant.c type_dispatch.c tensor.c $(SGEMM_SRCS) -lm -lpthread -fopenmp -lstdc++
