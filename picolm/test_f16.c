#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

float fp16_to_fp32_sw(uint16_t h) {
    unsigned s = (h >> 15) & 1;
    unsigned e = (h >> 10) & 31;
    unsigned m = h & 1023;
    float f;
    if (e == 0) {
        if (m == 0) { f = s ? -0.0f : 0.0f; }
        else { f = (float)m * 5.960464477539063e-8f; if (s) f = -f; }
    } else if (e == 31) {
        f = s ? -1.0f/0.0f : 1.0f/0.0f;
    } else {
        unsigned bits = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
        memcpy(&f, &bits, 4);
    }
    return f;
}

// Vulkan shader's f16tof32 (copied exactly)
float f16tof32_glsl(uint16_t h) {
    uint16_t s = (h >> 15) & 1;
    uint16_t e = (h >> 10) & 31;
    uint16_t m = h & 1023;
    float d;
    if (e == 0u) {
        if (m == 0u) { d = s != 0u ? -0.0f : 0.0f; }
        else { d = (float)m * 5.960464477539063e-8f; if (s) d = -d; }
    } else if (e == 31u) {
        d = s != 0u ? -1.0f/0.0f : 1.0f/0.0f;
    } else {
        uint32_t bits = (s << 31) | ((e - 15u + 127u) << 23) | (m << 13);
        memcpy(&d, &bits, 4);
    }
    return d;
}

int main(void) {
    int mismatches = 0;
    // Test specific Q8_0 scale values
    uint16_t test_vals[] = {
        0x3C00, // 1.0
        0x3800, // 0.5
        0x4000, // 2.0
        0xBC00, // -1.0
        0x3555, // ~0.333
        0x36CC, // ~0.4
        0x0000, // 0.0
        0x8000, // -0.0
        0x7C00, // inf
        0x3A00, // 0.25
    };
    int ntest = sizeof(test_vals) / sizeof(test_vals[0]);

    for (int i = 0; i < ntest; i++) {
        float cpu = fp16_to_fp32_sw(test_vals[i]);
        float gpu = f16tof32_glsl(test_vals[i]);
        printf("h=0x%04x: cpu=%.8f gpu=%.8f diff=%.2e\n",
               test_vals[i], cpu, gpu, (cpu>gpu?cpu-gpu:gpu-cpu));
        if (cpu != gpu) {
            if (!(cpu == 0 && gpu == 0)) mismatches++;
        }
    }

    // Also scan a range of values that are common for Q8_0 scales
    printf("\nScanning 0x0000-0x7FFF for mismatches...\n");
    int scan_mismatches = 0;
    for (uint16_t h = 0; h < 0x8000; h++) {
        float cpu = fp16_to_fp32_sw(h);
        float gpu = f16tof32_glsl(h);
        if (cpu != gpu) {
            if (cpu == 0 && gpu == 0) continue;
            if (scan_mismatches < 5)
                printf("  MISMATCH h=0x%04x: cpu=%f gpu=%f\n", h, cpu, gpu);
            scan_mismatches++;
        }
    }
    printf("Total scan mismatches: %d\n", scan_mismatches);

    return mismatches + scan_mismatches;
}

