#ifndef SYNAPSE_CUDA_MINILM_H
#define SYNAPSE_CUDA_MINILM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SynapseCudaEncoderLayerParams {
    const float *query_weight;
    const float *query_bias;
    const float *key_weight;
    const float *key_bias;
    const float *value_weight;
    const float *value_bias;
    const float *attention_output_weight;
    const float *attention_output_bias;
    const float *attention_ln_weight;
    const float *attention_ln_bias;
    const float *intermediate_weight;
    const float *intermediate_bias;
    const float *output_weight;
    const float *output_bias;
    const float *output_ln_weight;
    const float *output_ln_bias;
} SynapseCudaEncoderLayerParams;

void *synapse_cuda_context_new(int32_t graphs_enabled);
void synapse_cuda_context_free(void *context);
int32_t synapse_cuda_encoder_forward(
    void *context,
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float layer_norm_eps,
    const uint16_t *input,
    const uint8_t *attention_mask,
    float *output,
    const SynapseCudaEncoderLayerParams *layers
);
const char *synapse_cuda_last_error(void);
uint64_t synapse_cuda_cublaslt_version(void);

#ifdef __cplusplus
}
#endif

#endif
