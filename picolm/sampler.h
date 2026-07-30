#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>

typedef struct {
    float    temperature;
    float    top_p;
    int      top_k;
    float    min_p;
    uint64_t rng_state;   /* xorshift64 state */
} sampler_t;

void sampler_init(sampler_t *s, float temperature, float top_p, int top_k,
                  float min_p, uint64_t seed);
int  sampler_sample(sampler_t *s, float *logits, int vocab_size);

#endif
