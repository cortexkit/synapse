#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char metal_step_error[1024];

typedef struct Qwen3MetalStepLayerParams {
    const void *input_norm;
    const void *post_attention_norm;
    const void *q_weight;
    const void *q_weight_q8;
    const void *q_norm;
    const void *k_weight;
    const void *k_weight_q8;
    const void *k_norm;
    const void *v_weight;
    const void *v_weight_q8;
    const void *o_weight;
    const void *o_weight_q8;
    const void *gate_weight;
    const void *gate_weight_q8;
    const void *up_weight;
    const void *up_weight_q8;
    const void *down_weight;
    const void *down_weight_q8;
} Qwen3MetalStepLayerParams;

typedef struct Qwen3MetalStepTimings {
    double feed_wall_s;
    double execute_wall_s;
    double logits_readback_wall_s;
    double kv_update_wall_s;
    uint64_t step_calls;
} Qwen3MetalStepTimings;

typedef struct StepWeight {
    id<MTLBuffer> fp16;
    id<MTLBuffer> q8;
} StepWeight;

typedef struct StepLayerBuffers {
    id<MTLBuffer> input_norm;
    id<MTLBuffer> post_attention_norm;
    StepWeight q_weight;
    id<MTLBuffer> q_norm;
    StepWeight k_weight;
    id<MTLBuffer> k_norm;
    StepWeight v_weight;
    StepWeight o_weight;
    StepWeight gate_weight;
    StepWeight up_weight;
    StepWeight down_weight;
    id<MTLBuffer> key_cache;
    id<MTLBuffer> value_cache;
} StepLayerBuffers;

typedef struct Qwen3MetalStepContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> rmsnorm;
    id<MTLComputePipelineState> qkv_matvec;
    id<MTLComputePipelineState> qk_norm_rope;
    id<MTLComputePipelineState> attention;
    id<MTLComputePipelineState> matvec_residual;
    id<MTLComputePipelineState> residual_rmsnorm;
    id<MTLComputePipelineState> gate_up_swiglu;
    id<MTLComputePipelineState> lm_head;
    StepLayerBuffers *layers;
    uint64_t layer_count;
    uint64_t bucket;
    uint64_t hidden;
    uint64_t query_heads;
    uint64_t kv_heads;
    uint64_t head_dim;
    uint64_t intermediate;
    uint64_t vocab;
    float epsilon;
    uint32_t quantized;
    id<MTLBuffer> x_a;
    id<MTLBuffer> x_b;
    id<MTLBuffer> normalized;
    id<MTLBuffer> query;
    id<MTLBuffer> key;
    id<MTLBuffer> context;
    id<MTLBuffer> mlp;
    id<MTLBuffer> final_norm;
    id<MTLBuffer> logits;
    id<MTLBuffer> final_norm_weight;
    StepWeight lm_head_weight;
    Qwen3MetalStepTimings timings;
} Qwen3MetalStepContext;

static void set_error(NSString *message) {
    snprintf(metal_step_error, sizeof(metal_step_error), "%s", message.UTF8String ?: "unknown Metal step error");
}

const char *synapse_qwen3_metal_step_last_error(void) {
    return metal_step_error;
}

static id<MTLBuffer> new_buffer(id<MTLDevice> device, const void *bytes, NSUInteger length, MTLResourceOptions options) {
    if (bytes == NULL || length == 0) return nil;
    return [device newBufferWithBytes:bytes length:length options:options];
}

static id<MTLBuffer> new_zero_buffer(id<MTLDevice> device, NSUInteger length, MTLResourceOptions options) {
    return [device newBufferWithLength:length options:options];
}

static StepWeight new_weight(id<MTLDevice> device, const void *fp16, const void *q8, NSUInteger elements) {
    StepWeight weight = { nil, nil };
    if (q8 != NULL) {
        weight.q8 = new_buffer(device, q8, elements / 32 * 34, MTLResourceStorageModeShared);
    } else {
        weight.fp16 = new_buffer(device, fp16, elements * sizeof(uint16_t), MTLResourceStorageModeShared);
    }
    return weight;
}

static void release_weight(StepWeight *weight) {
    [weight->q8 release];
    [weight->fp16 release];
    weight->q8 = nil;
    weight->fp16 = nil;
}

static void release_layer(StepLayerBuffers *layer) {
    [layer->input_norm release];
    [layer->post_attention_norm release];
    [layer->q_norm release];
    [layer->k_norm release];
    [layer->key_cache release];
    [layer->value_cache release];
    release_weight(&layer->q_weight);
    release_weight(&layer->k_weight);
    release_weight(&layer->v_weight);
    release_weight(&layer->o_weight);
    release_weight(&layer->gate_weight);
    release_weight(&layer->up_weight);
    release_weight(&layer->down_weight);
    memset(layer, 0, sizeof(*layer));
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) return nil;
    NSError *error = nil;
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (result == nil) {
        set_error(error.localizedDescription ?: [NSString stringWithFormat:@"failed to compile Metal kernel %@", name]);
    }
    return result;
}

void *synapse_qwen3_metal_step_context_new(
    uint64_t bucket,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t vocab,
    float epsilon,
    const char *metallib_path
) {
    @autoreleasepool {
        if (bucket == 0 || hidden == 0 || query_heads == 0 || kv_heads == 0 || head_dim == 0 ||
            intermediate == 0 || vocab == 0 || query_heads % kv_heads != 0 || head_dim % 2 != 0 ||
            metallib_path == NULL) {
            set_error(@"invalid Metal step dimensions or metallib path");
            return NULL;
        }
        Qwen3MetalStepContext *context = calloc(1, sizeof(*context));
        if (context == NULL) {
            set_error(@"failed to allocate Metal step context");
            return NULL;
        }
        context->device = MTLCreateSystemDefaultDevice();
        if (context->device == nil) {
            set_error(@"no Metal device for Metal step");
            free(context);
            return NULL;
        }
        context->queue = [context->device newCommandQueue];
        NSError *error = nil;
        NSURL *library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
        context->library = [context->device newLibraryWithURL:library_url error:&error];
        if (context->queue == nil || context->library == nil) {
            set_error(error.localizedDescription ?: @"failed to load Metal step metallib");
            [context->queue release];
            [context->library release];
            [context->device release];
            free(context);
            return NULL;
        }
        context->rmsnorm = pipeline(context->device, context->library, @"metal_step_rmsnorm");
        context->qkv_matvec = pipeline(context->device, context->library, @"metal_step_qkv_matvec");
        context->qk_norm_rope = pipeline(context->device, context->library, @"metal_step_qk_norm_rope");
        context->attention = pipeline(context->device, context->library, @"metal_step_attention");
        context->matvec_residual = pipeline(context->device, context->library, @"metal_step_matvec_residual");
        context->residual_rmsnorm = pipeline(context->device, context->library, @"metal_step_residual_rmsnorm");
        context->gate_up_swiglu = pipeline(context->device, context->library, @"metal_step_gate_up_swiglu");
        context->lm_head = pipeline(context->device, context->library, @"metal_step_lm_head");
        if (context->rmsnorm == nil || context->qkv_matvec == nil || context->qk_norm_rope == nil ||
            context->attention == nil || context->matvec_residual == nil || context->residual_rmsnorm == nil ||
            context->gate_up_swiglu == nil ||
            context->lm_head == nil) {
            [context->library release];
            [context->queue release];
            [context->device release];
            free(context);
            return NULL;
        }
        context->bucket = bucket;
        context->hidden = hidden;
        context->query_heads = query_heads;
        context->kv_heads = kv_heads;
        context->head_dim = head_dim;
        context->intermediate = intermediate;
        context->vocab = vocab;
        context->epsilon = epsilon;
        return context;
    }
}

int32_t synapse_qwen3_metal_step_prepare(
    void *raw,
    uint64_t layer_count,
    uint32_t quantized,
    const Qwen3MetalStepLayerParams *params,
    const void *final_norm_weight,
    const void *lm_head_weight,
    const void *lm_head_q8
) {
    @autoreleasepool {
        @try {
            Qwen3MetalStepContext *context = raw;
            if (context == NULL || layer_count == 0 || params == NULL || final_norm_weight == NULL ||
                lm_head_weight == NULL) {
                set_error(@"invalid Metal step preparation arguments");
                return -1;
            }
            uint64_t query_width = context->query_heads * context->head_dim;
            uint64_t kv_width = context->kv_heads * context->head_dim;
            NSUInteger cache_elements = (NSUInteger)(context->kv_heads * context->bucket * context->head_dim);
            context->layers = calloc((size_t)layer_count, sizeof(*context->layers));
            if (context->layers == NULL) {
                set_error(@"failed to allocate Metal step layer table");
                return -2;
            }
            context->layer_count = layer_count;
            context->quantized = quantized;
            for (uint64_t i = 0; i < layer_count; ++i) {
                const Qwen3MetalStepLayerParams *source = &params[i];
                StepLayerBuffers *target = &context->layers[i];
                if (quantized && (source->q_weight_q8 == NULL || source->k_weight_q8 == NULL ||
                                  source->v_weight_q8 == NULL || source->o_weight_q8 == NULL ||
                                  source->gate_weight_q8 == NULL || source->up_weight_q8 == NULL ||
                                  source->down_weight_q8 == NULL)) {
                    set_error(@"quantized Metal step is missing a Q8_0 weight buffer");
                    return -3;
                }
                target->input_norm = new_buffer(context->device, source->input_norm,
                                                context->hidden * sizeof(uint16_t), MTLResourceStorageModeShared);
                target->post_attention_norm = new_buffer(context->device, source->post_attention_norm,
                                                         context->hidden * sizeof(uint16_t), MTLResourceStorageModeShared);
                target->q_norm = new_buffer(context->device, source->q_norm,
                                            context->head_dim * sizeof(uint16_t), MTLResourceStorageModeShared);
                target->k_norm = new_buffer(context->device, source->k_norm,
                                            context->head_dim * sizeof(uint16_t), MTLResourceStorageModeShared);
                target->q_weight = new_weight(context->device, source->q_weight, source->q_weight_q8,
                                              (NSUInteger)(query_width * context->hidden));
                target->k_weight = new_weight(context->device, source->k_weight, source->k_weight_q8,
                                              (NSUInteger)(kv_width * context->hidden));
                target->v_weight = new_weight(context->device, source->v_weight, source->v_weight_q8,
                                              (NSUInteger)(kv_width * context->hidden));
                target->o_weight = new_weight(context->device, source->o_weight, source->o_weight_q8,
                                              (NSUInteger)(context->hidden * query_width));
                target->gate_weight = new_weight(context->device, source->gate_weight, source->gate_weight_q8,
                                                 (NSUInteger)(context->intermediate * context->hidden));
                target->up_weight = new_weight(context->device, source->up_weight, source->up_weight_q8,
                                               (NSUInteger)(context->intermediate * context->hidden));
                target->down_weight = new_weight(context->device, source->down_weight, source->down_weight_q8,
                                                 (NSUInteger)(context->hidden * context->intermediate));
                target->key_cache = new_zero_buffer(context->device, cache_elements * sizeof(uint16_t), MTLResourceStorageModePrivate);
                target->value_cache = new_zero_buffer(context->device, cache_elements * sizeof(uint16_t), MTLResourceStorageModePrivate);
                if (target->input_norm == nil || target->post_attention_norm == nil || target->q_norm == nil ||
                    target->k_norm == nil || target->key_cache == nil || target->value_cache == nil ||
                    (quantized ? (target->q_weight.q8 == nil || target->k_weight.q8 == nil || target->v_weight.q8 == nil ||
                                  target->o_weight.q8 == nil || target->gate_weight.q8 == nil || target->up_weight.q8 == nil ||
                                  target->down_weight.q8 == nil)
                                : (target->q_weight.fp16 == nil || target->k_weight.fp16 == nil || target->v_weight.fp16 == nil ||
                                   target->o_weight.fp16 == nil || target->gate_weight.fp16 == nil || target->up_weight.fp16 == nil ||
                                   target->down_weight.fp16 == nil))) {
                    set_error(@"failed to allocate Metal step weights or KV cache");
                    return -4;
                }
            }
            context->final_norm_weight = new_buffer(context->device, final_norm_weight,
                                                    context->hidden * sizeof(uint16_t), MTLResourceStorageModeShared);
            context->lm_head_weight = new_weight(context->device, lm_head_weight, lm_head_q8,
                                                 (NSUInteger)(context->vocab * context->hidden));
            NSUInteger hidden_bytes = (NSUInteger)context->hidden * sizeof(uint16_t);
            NSUInteger query_bytes = (NSUInteger)query_width * sizeof(uint16_t);
            NSUInteger intermediate_bytes = (NSUInteger)context->intermediate * sizeof(uint16_t);
            context->x_a = new_zero_buffer(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            context->x_b = new_zero_buffer(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            context->normalized = new_zero_buffer(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            context->query = new_zero_buffer(context->device, query_bytes, MTLResourceStorageModePrivate);
            context->key = new_zero_buffer(context->device, (NSUInteger)kv_width * sizeof(uint16_t), MTLResourceStorageModePrivate);
            context->context = new_zero_buffer(context->device, query_bytes, MTLResourceStorageModePrivate);
            context->mlp = new_zero_buffer(context->device, intermediate_bytes, MTLResourceStorageModePrivate);
            context->final_norm = new_zero_buffer(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            context->logits = new_zero_buffer(context->device, (NSUInteger)context->vocab * sizeof(float), MTLResourceStorageModeShared);
            if (context->final_norm_weight == nil ||
                (context->lm_head_weight.fp16 == nil && context->lm_head_weight.q8 == nil) ||
                context->x_a == nil || context->x_b == nil || context->normalized == nil || context->query == nil ||
                context->key == nil || context->context == nil || context->mlp == nil || context->final_norm == nil ||
                context->logits == nil) {
                set_error(@"failed to allocate Metal step activation buffers");
                return -5;
            }
            return 0;
        } @catch (NSException *exception) {
            set_error(exception.reason);
            return -100;
        }
    }
}

static MTLSize grid_size(NSUInteger count) {
    return MTLSizeMake(count, 1, 1);
}

static MTLSize group_size(NSUInteger count) {
    return MTLSizeMake(MIN((NSUInteger)256, MAX((NSUInteger)1, count)), 1, 1);
}

static MTLSize attention_grid(Qwen3MetalStepContext *context) {
    return grid_size((NSUInteger)context->query_heads * 32);
}

static void set_weight(id<MTLComputeCommandEncoder> encoder, StepWeight *weight, NSUInteger fp16_index, NSUInteger q8_index) {
    [encoder setBuffer:weight->fp16 offset:0 atIndex:fp16_index];
    [encoder setBuffer:weight->q8 offset:0 atIndex:q8_index];
}

static void encode_rmsnorm(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> input,
    id<MTLBuffer> output,
    id<MTLBuffer> weight,
    uint32_t width,
    float epsilon
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->rmsnorm];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:weight offset:0 atIndex:2];
    struct { uint32_t width; float epsilon; } config = { width, epsilon };
    [encoder setBytes:&config length:sizeof(config) atIndex:3];
    [encoder dispatchThreads:grid_size(1) threadsPerThreadgroup:group_size(1)];
    [encoder endEncoding];
}

static void encode_qkv(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    StepLayerBuffers *layer,
    id<MTLBuffer> input,
    uint32_t position
) {
    uint32_t query_width = (uint32_t)(context->query_heads * context->head_dim);
    uint32_t kv_width = (uint32_t)(context->kv_heads * context->head_dim);
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->qkv_matvec];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:context->query offset:0 atIndex:1];
    [encoder setBuffer:context->key offset:0 atIndex:2];
    [encoder setBuffer:layer->value_cache offset:0 atIndex:3];
    set_weight(encoder, &layer->q_weight, 4, 5);
    set_weight(encoder, &layer->k_weight, 6, 7);
    set_weight(encoder, &layer->v_weight, 8, 9);
    struct { uint32_t input_width; uint32_t query_width; uint32_t kv_width; uint32_t head_dim; uint32_t capacity; uint32_t position; uint32_t quantized; } config = {
        (uint32_t)context->hidden, query_width, kv_width, (uint32_t)context->head_dim,
        (uint32_t)context->bucket, position, context->quantized
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:10];
    [encoder dispatchThreads:grid_size(MAX(query_width, kv_width)) threadsPerThreadgroup:group_size(MAX(query_width, kv_width))];
    [encoder endEncoding];
}

static void encode_qk_norm_rope(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    StepLayerBuffers *layer,
    id<MTLBuffer> rope_cos,
    id<MTLBuffer> rope_sin
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->qk_norm_rope];
    [encoder setBuffer:context->query offset:0 atIndex:0];
    [encoder setBuffer:context->key offset:0 atIndex:1];
    [encoder setBuffer:layer->q_norm offset:0 atIndex:2];
    [encoder setBuffer:layer->k_norm offset:0 atIndex:3];
    [encoder setBuffer:rope_cos offset:0 atIndex:4];
    [encoder setBuffer:rope_sin offset:0 atIndex:5];
    struct { uint32_t query_heads; uint32_t kv_heads; uint32_t head_dim; float epsilon; } config = {
        (uint32_t)context->query_heads, (uint32_t)context->kv_heads, (uint32_t)context->head_dim, context->epsilon
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:6];
    [encoder dispatchThreads:grid_size(MAX(context->query_heads, context->kv_heads)) threadsPerThreadgroup:group_size(MAX(context->query_heads, context->kv_heads))];
    [encoder endEncoding];
}

static void encode_attention(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    StepLayerBuffers *layer,
    uint32_t position
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->attention];
    [encoder setBuffer:context->query offset:0 atIndex:0];
    [encoder setBuffer:layer->key_cache offset:0 atIndex:1];
    [encoder setBuffer:layer->value_cache offset:0 atIndex:2];
    [encoder setBuffer:context->context offset:0 atIndex:3];
    struct { uint32_t query_heads; uint32_t kv_heads; uint32_t head_dim; uint32_t capacity; uint32_t position; } config = {
        (uint32_t)context->query_heads, (uint32_t)context->kv_heads, (uint32_t)context->head_dim,
        (uint32_t)context->bucket, position
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:4];
    [encoder dispatchThreads:attention_grid(context) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
}

static void encode_matvec_residual(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> input,
    id<MTLBuffer> residual,
    id<MTLBuffer> output,
    StepWeight *weight,
    uint32_t input_width,
    uint32_t output_width
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->matvec_residual];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:residual offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    set_weight(encoder, weight, 3, 4);
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; } config = {
        input_width, output_width, context->quantized
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:5];
    [encoder dispatchThreads:grid_size(output_width) threadsPerThreadgroup:group_size(output_width)];
    [encoder endEncoding];
}

static void encode_residual_rmsnorm(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> projection,
    id<MTLBuffer> residual,
    id<MTLBuffer> normalized,
    id<MTLBuffer> weight
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->residual_rmsnorm];
    [encoder setBuffer:projection offset:0 atIndex:0];
    [encoder setBuffer:residual offset:0 atIndex:1];
    [encoder setBuffer:normalized offset:0 atIndex:2];
    [encoder setBuffer:weight offset:0 atIndex:3];
    struct { uint32_t width; float epsilon; } config = {
        (uint32_t)context->hidden, context->epsilon
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:4];
    [encoder dispatchThreads:grid_size(1) threadsPerThreadgroup:group_size(1)];
    [encoder endEncoding];
}

static void encode_gate_up(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    StepLayerBuffers *layer,
    id<MTLBuffer> input
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->gate_up_swiglu];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:context->mlp offset:0 atIndex:1];
    set_weight(encoder, &layer->gate_weight, 2, 3);
    set_weight(encoder, &layer->up_weight, 4, 5);
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; } config = {
        (uint32_t)context->hidden, (uint32_t)context->intermediate, context->quantized
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:6];
    [encoder dispatchThreads:grid_size(context->intermediate) threadsPerThreadgroup:group_size(context->intermediate)];
    [encoder endEncoding];
}

static void encode_lm_head(Qwen3MetalStepContext *context, id<MTLCommandBuffer> command_buffer) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->lm_head];
    [encoder setBuffer:context->final_norm offset:0 atIndex:0];
    [encoder setBuffer:context->logits offset:0 atIndex:1];
    set_weight(encoder, &context->lm_head_weight, 2, 3);
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; } config = {
        (uint32_t)context->hidden, (uint32_t)context->vocab, context->quantized
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:4];
    [encoder dispatchThreads:grid_size(context->vocab) threadsPerThreadgroup:group_size(context->vocab)];
    [encoder endEncoding];
}

int32_t synapse_qwen3_metal_step_import_caches(
    void *raw,
    const uint16_t *cache_data,
    uint64_t cache_data_elements
) {
    @autoreleasepool {
        Qwen3MetalStepContext *context = raw;
        NSUInteger one_cache = (NSUInteger)(context == NULL ? 0 : context->kv_heads * context->bucket * context->head_dim);
        if (context == NULL || cache_data == NULL || cache_data_elements != context->layer_count * one_cache * 2) {
            set_error(@"invalid Metal step cache handoff arguments");
            return -1;
        }
        id<MTLBuffer> source = [context->device newBufferWithBytes:cache_data
                                                               length:(NSUInteger)cache_data_elements * sizeof(uint16_t)
                                                              options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
        if (source == nil || command_buffer == nil || blit == nil) {
            [source release];
            set_error(@"failed to create Metal step cache handoff command");
            return -2;
        }
        NSUInteger bytes = one_cache * sizeof(uint16_t);
        for (uint64_t layer = 0; layer < context->layer_count; ++layer) {
            NSUInteger base = (NSUInteger)layer * bytes * 2;
            [blit copyFromBuffer:source sourceOffset:base
                          toBuffer:context->layers[layer].key_cache destinationOffset:0 size:bytes];
            [blit copyFromBuffer:source sourceOffset:base + bytes
                          toBuffer:context->layers[layer].value_cache destinationOffset:0 size:bytes];
        }
        [blit endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        BOOL ok = command_buffer.status != MTLCommandBufferStatusError;
        if (!ok) set_error(command_buffer.error.localizedDescription ?: @"Metal step cache handoff failed");
        [source release];
        context->timings.kv_update_wall_s += 0.0;
        return ok ? 0 : -3;
    }
}

int32_t synapse_qwen3_metal_step(
    void *raw,
    uint64_t position,
    const uint16_t *input,
    const uint16_t *rope_cos,
    const uint16_t *rope_sin,
    float *logits,
    float epsilon
) {
    @autoreleasepool {
        @try {
            metal_step_error[0] = '\0';
            double feed_started = [NSDate timeIntervalSinceReferenceDate];
            Qwen3MetalStepContext *context = raw;
            if (context == NULL || input == NULL || rope_cos == NULL || rope_sin == NULL || logits == NULL ||
                position >= context->bucket || context->layers == NULL) {
                set_error(@"invalid Metal step arguments");
                return -1;
            }
            NSUInteger hidden_bytes = (NSUInteger)context->hidden * sizeof(uint16_t);
            id<MTLBuffer> input_buffer = [context->device newBufferWithBytes:input length:hidden_bytes options:MTLResourceStorageModeShared];
            // RoPE is indexed by head dimension, not hidden dimension.
            id<MTLBuffer> cosine_buffer = [context->device newBufferWithBytes:rope_cos length:(NSUInteger)context->head_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> sine_buffer = [context->device newBufferWithBytes:rope_sin length:(NSUInteger)context->head_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
            if (input_buffer == nil || cosine_buffer == nil || sine_buffer == nil || command_buffer == nil) {
                [input_buffer release];
                [cosine_buffer release];
                [sine_buffer release];
                set_error(@"failed to allocate Metal step input buffers");
                return -2;
            }
            id<MTLBuffer> current = input_buffer;
            id<MTLBuffer> next = context->x_a;
            for (uint64_t index = 0; index < context->layer_count; ++index) {
                StepLayerBuffers *layer = &context->layers[index];
                encode_rmsnorm(context, command_buffer, current, context->normalized, layer->input_norm,
                               (uint32_t)context->hidden, epsilon);
                encode_qkv(context, command_buffer, layer, context->normalized, (uint32_t)position);
                encode_qk_norm_rope(context, command_buffer, layer, cosine_buffer, sine_buffer);
                encode_attention(context, command_buffer, layer, (uint32_t)position);
                encode_matvec_residual(context, command_buffer, context->context, current, next,
                                       &layer->o_weight, (uint32_t)context->query_heads * (uint32_t)context->head_dim,
                                       (uint32_t)context->hidden);
                encode_residual_rmsnorm(
                    context,
                    command_buffer,
                    next,
                    current,
                    context->normalized,
                    layer->post_attention_norm
                );
                encode_gate_up(context, command_buffer, layer, context->normalized);
                encode_matvec_residual(context, command_buffer, context->mlp, next, current,
                                       &layer->down_weight, (uint32_t)context->intermediate, (uint32_t)context->hidden);
                id<MTLBuffer> old_next = next;
                current = old_next;
                next = old_next == context->x_a ? context->x_b : context->x_a;
            }
            encode_rmsnorm(context, command_buffer, current, context->final_norm, context->final_norm_weight,
                           (uint32_t)context->hidden, epsilon);
            encode_lm_head(context, command_buffer);
            context->timings.feed_wall_s += [NSDate timeIntervalSinceReferenceDate] - feed_started;
            double started = [NSDate timeIntervalSinceReferenceDate];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            context->timings.execute_wall_s += [NSDate timeIntervalSinceReferenceDate] - started;
            BOOL ok = command_buffer.status != MTLCommandBufferStatusError;
            if (!ok) {
                set_error(command_buffer.error.localizedDescription ?: @"Metal step command buffer failed");
            } else {
                double readback_started = [NSDate timeIntervalSinceReferenceDate];
                memcpy(logits, context->logits.contents, (NSUInteger)context->vocab * sizeof(float));
                context->timings.logits_readback_wall_s += [NSDate timeIntervalSinceReferenceDate] - readback_started;
                context->timings.step_calls += 1;
            }
            [input_buffer release];
            [cosine_buffer release];
            [sine_buffer release];
            return ok ? 0 : -3;
        } @catch (NSException *exception) {
            set_error(exception.reason);
            return -100;
        }
    }
}

void synapse_qwen3_metal_step_timings(void *raw, Qwen3MetalStepTimings *timings) {
    if (raw == NULL || timings == NULL) return;
    *timings = ((Qwen3MetalStepContext *)raw)->timings;
}

int32_t synapse_qwen3_metal_step_cache_copy(void *raw, uint64_t layer, uint16_t *output, uint64_t elements) {
    @autoreleasepool {
        Qwen3MetalStepContext *context = raw;
        NSUInteger one_cache = (NSUInteger)(context == NULL ? 0 : context->kv_heads * context->bucket * context->head_dim);
        if (context == NULL || output == NULL || layer >= context->layer_count || elements != one_cache * 2) {
            set_error(@"invalid Metal step cache inspection arguments");
            return -1;
        }
        id<MTLBuffer> staging = new_zero_buffer(context->device, one_cache * sizeof(uint16_t) * 2, MTLResourceStorageModeShared);
        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
        if (staging == nil || command_buffer == nil || blit == nil) {
            [staging release];
            set_error(@"failed to stage Metal step cache inspection");
            return -2;
        }
        NSUInteger bytes = one_cache * sizeof(uint16_t);
        [blit copyFromBuffer:context->layers[layer].key_cache sourceOffset:0 toBuffer:staging destinationOffset:0 size:bytes];
        [blit copyFromBuffer:context->layers[layer].value_cache sourceOffset:0 toBuffer:staging destinationOffset:bytes size:bytes];
        [blit endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            set_error(command_buffer.error.localizedDescription ?: @"Metal step cache inspection failed");
            [staging release];
            return -3;
        }
        memcpy(output, staging.contents, bytes * 2);
        [staging release];
        return 0;
    }
}

void synapse_qwen3_metal_step_context_free(void *raw) {
    if (raw == NULL) return;
    Qwen3MetalStepContext *context = raw;
    if (context->layers != NULL) {
        for (uint64_t index = 0; index < context->layer_count; ++index) release_layer(&context->layers[index]);
        free(context->layers);
    }
    [context->lm_head_weight.q8 release];
    [context->lm_head_weight.fp16 release];
    [context->final_norm_weight release];
    [context->x_a release];
    [context->x_b release];
    [context->normalized release];
    [context->query release];
    [context->key release];
    [context->context release];
    [context->mlp release];
    [context->final_norm release];
    [context->logits release];
    [context->lm_head release];
    [context->gate_up_swiglu release];
    [context->matvec_residual release];
    [context->residual_rmsnorm release];
    [context->attention release];
    [context->qk_norm_rope release];
    [context->qkv_matvec release];
    [context->rmsnorm release];
    [context->library release];
    [context->queue release];
    [context->device release];
    free(context);
}
