// Verify Vulkan f16tof32 matches CPU fp16_to_fp32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "quant.h"

void fp16_table_init(void);

// This is the EXACT logic from the Vulkan shader
float f16tof32_shader(uint16_t h) {
    uint16_t s = (h >> 15) & 1;
    uint16_t e = (h >> 10) & 31;
    uint16_t m = h & 1023;
    float d;
    if (e == 0) {
        if (m == 0) { d = s ? -0.0f : 0.0f; }
        else { d = (float)m * 5.960464477539063e-8f; if (s) d = -d; }
    } else if (e == 31) {
        d = s ? -1.0f/0.0f : 1.0f/0.0f;
    } else {
        uint32_t bits = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
        memcpy(&d, &bits, 4);
    }
    return d;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    fp16_table_init();
    
    int mismatches = 0;
    for (uint16_t h = 0; h < (1 << 16); h++) {
        float cpu = fp16_to_fp32_lookup(h);
        float gpu = f16tof32_shader(h);
        
        if (cpu != gpu) {
            // Allow -0.0 == 0.0
            if (cpu == 0.0f && gpu == 0.0f) continue;
            // Allow inf == inf, nan == nan
            if ((cpu != cpu) && (gpu != gpu)) continue; // both NaN
            if ((cpu == 1e30f || cpu == -1e30f) && (gpu == 1e30f || gpu == -1e30f)) continue; // both inf
            
            if (mismatches < 20) {
                printf("MISMATCH h=0x%04x: cpu=%f, gpu=%f\n", h, cpu, gpu);
            }
            mismatches++;
        }
    }
    
    if (mismatches == 0) {
        printf("All 65536 FP16 values match between shader and CPU.\n");
    } else {
        printf("Total mismatches: %d\n", mismatches);
    }
    
    return mismatches > 0 ? 1 : 0;
}

