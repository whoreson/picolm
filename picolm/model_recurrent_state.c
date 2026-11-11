/* Generic recurrent-state checkpointing.
 *
 * One interface over the two recurrent families in this tree:
 *   - Qwen3.5/3.6 delta-net SSM        (model_ssm_state_*)
 *   - GLM-5.3-Flash KDA                (model_glm5_kda_state_*)
 * Pure-attention models (Llama/MHA, GPT-2, ...) have no recurrent state and
 * report size 0, which makes every checkpoint call a cheap no-op.
 *
 * Only RECURRENT state is snapshotted. The KV cache -- including GLM-5's MLA
 * latent cache -- is positional: rewinding to position N leaves rows 0..N-1
 * untouched, exactly like ordinary KV, so it does not need a snapshot. This
 * keeps the payload size independent of context length.
 *
 * See model.h for the full contract.
 */
#include "model.h"
#include <string.h>

int model_has_recurrent_state(const model_t *m) {
    const model_config_t *c = &m->config;
    if (c->has_ssm) return 1;
    if (c->is_glm5 && c->glm5_n_kda > 0) return 1;
    return 0;
}

size_t model_recurrent_state_size(const model_t *m) {
    const model_config_t *c = &m->config;
    if (c->has_ssm) return model_ssm_snapshot_size(m);
    if (c->is_glm5)  return model_glm5_kda_state_size(m);
    return 0;
}

size_t model_recurrent_state_save(const model_t *m, uint8_t *buf, size_t buf_size) {
    const model_config_t *c = &m->config;
    if (c->has_ssm) return model_ssm_state_save(m, buf, buf_size);
    if (c->is_glm5)  return model_glm5_kda_state_save(m, buf, buf_size);
    return 0;
}

size_t model_recurrent_state_restore(model_t *m, const uint8_t *buf, size_t buf_size) {
    const model_config_t *c = &m->config;
    if (c->has_ssm) return model_ssm_state_restore(m, buf, buf_size);
    if (c->is_glm5)  return model_glm5_kda_state_restore(m, buf, buf_size);
    return 0;
}

void model_recurrent_state_reset(model_t *m) {
    const model_config_t *c = &m->config;
    if (c->has_ssm) { model_ssm_state_reset(m); return; }
    if (c->is_glm5) {
        /* Zero every KDA buffer. Equivalent to restoring a zeroed snapshot,
         * but without allocating one. */
        size_t sz = model_glm5_kda_state_size(m);
        if (sz == 0) return;
        const int conv_dim = 3 * c->glm5_n_kda_heads * c->glm5_kda_dim;
        const int kda_dim  = c->glm5_kda_dim;
        for (int l = 0; l < c->n_layers; l++) {
            if (m->state.glm5.kda_conv_state[l])
                memset(m->state.glm5.kda_conv_state[l], 0,
                       (size_t)(c->ssm_d_conv - 1) * conv_dim * sizeof(float));
            if (m->state.glm5.kda_state[l] && !c->glm5_layer_is_mla[l])
                memset(m->state.glm5.kda_state[l], 0,
                       (size_t)c->glm5_n_kda_heads * kda_dim * kda_dim * sizeof(float));
        }
    }
}
