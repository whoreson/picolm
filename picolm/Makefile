CC      = gcc
CFLAGS  = -O3 -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wpedantic
LDFLAGS = -lm -lpthread
SRCS    = picolm.c model.c tensor.c quant.c tokenizer.c sampler.c grammar.c
TARGET  = picolm

PREFIX  ?= /usr/local
MODEL_DIR ?= /opt/picolm/models

# --- Native build (auto-detect CPU, enables SSE2/AVX on x86) ---
native: CFLAGS += -march=native
native: $(TARGET)

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
avx: CFLAGS += -mavx -mfma -mpopcnt
avx: $(TARGET)

# --- x86-64 with AVX2 (Haswell and newer Intel; Excavator and newer AMD) ---
avx2: CFLAGS += -mavx2 -mfma -mpopcnt
avx2: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

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
	rm -f $(TARGET) $(TARGET).exe *.obj *.o

.PHONY: native x86 scalar sse2 sse3 avx avx2 static pi pi-arm32 cross-pi riscv cross-riscv debug install model clean
