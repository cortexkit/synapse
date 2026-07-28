#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char metal_step_error[1024];

// Maximum number of draft tokens one batched verification forward can carry.
// The mat-mat kernels size their per-column accumulator arrays to this bound
// (MAX_BATCH_K in the Metal source) and the lazy batch buffers are allocated
// for exactly this many columns. Keep the two in sync.
#define METAL_STEP_MAX_BATCH_K 16

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
    double kernel_rmsnorm_s;
    double kernel_qkv_matvec_s;
    double kernel_qk_norm_rope_s;
    double kernel_attention_s;
    double kernel_o_proj_s;
    double kernel_residual_rmsnorm_s;
    double kernel_down_proj_s;
    double kernel_gate_up_swiglu_s;
    double kernel_lm_head_s;
    uint64_t kernel_samples;
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
    id<MTLComputePipelineState> argmax_partial;
    id<MTLComputePipelineState> argmax_final;
    id<MTLComputePipelineState> embedding_gather;
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
    id<MTLBuffer> attention_scores;
    id<MTLBuffer> mlp;
    id<MTLBuffer> final_norm;
    id<MTLBuffer> logits;
    id<MTLBuffer> final_norm_weight;
    StepWeight lm_head_weight;
    // Chained-decode resources. The tied f16 embedding table lives resident so a
    // chained step can gather its input row device-side from the previous step's
    // argmax output, and the argmax scratch buffers turn the vocab logits into a
    // single token id without a host round trip. These are only encoded on the
    // chained path; the per-token path never touches them.
    id<MTLBuffer> embeddings;
    uint64_t argmax_partials;
    id<MTLBuffer> argmax_partial_keys;
    id<MTLBuffer> argmax_partial_ids;
    id<MTLBuffer> chain_token_ids;
    id<MTLBuffer> chain_input;
    // Batched-verification resources. These are allocated lazily on the first
    // verify_batch call (for METAL_STEP_MAX_BATCH_K columns) so the per-token
    // and chained paths never pay for them and their behavior is unchanged.
    // The batched path reuses the single-token norm/RoPE/attention/argmax/
    // gather kernels via per-column buffer offsets, and adds four mat-mat
    // projection pipelines that stream each weight row once across all columns.
    id<MTLComputePipelineState> qkv_matvec_batch;
    id<MTLComputePipelineState> matvec_residual_batch;
    id<MTLComputePipelineState> gate_up_swiglu_batch;
    id<MTLComputePipelineState> lm_head_batch;
    uint64_t batch_capacity;
    id<MTLBuffer> batch_input;
    id<MTLBuffer> batch_x_b;
    id<MTLBuffer> batch_normalized;
    id<MTLBuffer> batch_query;
    id<MTLBuffer> batch_key;
    id<MTLBuffer> batch_context;
    id<MTLBuffer> batch_attention_scores;
    id<MTLBuffer> batch_mlp;
    id<MTLBuffer> batch_final_norm;
    id<MTLBuffer> batch_logits;
    id<MTLBuffer> batch_argmax_partial_keys;
    id<MTLBuffer> batch_argmax_partial_ids;
    BOOL profile_kernels;
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

static id<MTLBuffer> new_private_buffer(
    id<MTLDevice> device,
    id<MTLBlitCommandEncoder> blit,
    const void *bytes,
    NSUInteger length
) {
    if (bytes == NULL || length == 0 || blit == nil) return nil;
    id<MTLBuffer> source = new_buffer(device, bytes, length, MTLResourceStorageModeShared);
    id<MTLBuffer> destination = new_zero_buffer(device, length, MTLResourceStorageModePrivate);
    if (source == nil || destination == nil) {
        [source release];
        [destination release];
        return nil;
    }
    [blit copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:length];
    [source release];
    return destination;
}

static StepWeight new_weight(
    id<MTLDevice> device,
    id<MTLBlitCommandEncoder> blit,
    const void *fp16,
    const void *q8,
    NSUInteger elements
) {
    StepWeight weight = { nil, nil };
    if (q8 != NULL) {
        weight.q8 = new_private_buffer(device, blit, q8, elements / 32 * 34);
    } else {
        weight.fp16 = new_private_buffer(device, blit, fp16, elements * sizeof(uint16_t));
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
        context->argmax_partial = pipeline(context->device, context->library, @"metal_step_argmax_partial");
        context->argmax_final = pipeline(context->device, context->library, @"metal_step_argmax_final");
        context->embedding_gather = pipeline(context->device, context->library, @"metal_step_embedding_gather");
        if (context->rmsnorm == nil || context->qkv_matvec == nil || context->qk_norm_rope == nil ||
            context->attention == nil || context->matvec_residual == nil || context->residual_rmsnorm == nil ||
            context->gate_up_swiglu == nil ||
            context->lm_head == nil || context->argmax_partial == nil || context->argmax_final == nil ||
            context->embedding_gather == nil) {
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
        // Profiling uses one command buffer per kernel invocation so each GPU
        // start/end pair identifies a single kernel class. It is opt-in because
        // the synchronization required for attribution is intentionally slow.
        const char *profile = getenv("SYNAPSE_METAL_STEP_PROFILE");
        context->profile_kernels = profile != NULL && profile[0] != '\0' && strcmp(profile, "0") != 0;
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
    const void *lm_head_q8,
    const void *embeddings
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
            id<MTLCommandBuffer> upload_command = [context->queue commandBuffer];
            id<MTLBlitCommandEncoder> upload_blit = [upload_command blitCommandEncoder];
            if (upload_command == nil || upload_blit == nil) {
                set_error(@"failed to create private weight upload command");
                return -3;
            }
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
                target->input_norm = new_private_buffer(context->device, upload_blit, source->input_norm,
                                                        context->hidden * sizeof(uint16_t));
                target->post_attention_norm = new_private_buffer(context->device, upload_blit, source->post_attention_norm,
                                                                 context->hidden * sizeof(uint16_t));
                target->q_norm = new_private_buffer(context->device, upload_blit, source->q_norm,
                                                    context->head_dim * sizeof(uint16_t));
                target->k_norm = new_private_buffer(context->device, upload_blit, source->k_norm,
                                                    context->head_dim * sizeof(uint16_t));
                target->q_weight = new_weight(context->device, upload_blit, source->q_weight, source->q_weight_q8,
                                              (NSUInteger)(query_width * context->hidden));
                target->k_weight = new_weight(context->device, upload_blit, source->k_weight, source->k_weight_q8,
                                              (NSUInteger)(kv_width * context->hidden));
                target->v_weight = new_weight(context->device, upload_blit, source->v_weight, source->v_weight_q8,
                                              (NSUInteger)(kv_width * context->hidden));
                target->o_weight = new_weight(context->device, upload_blit, source->o_weight, source->o_weight_q8,
                                              (NSUInteger)(context->hidden * query_width));
                target->gate_weight = new_weight(context->device, upload_blit, source->gate_weight, source->gate_weight_q8,
                                                 (NSUInteger)(context->intermediate * context->hidden));
                target->up_weight = new_weight(context->device, upload_blit, source->up_weight, source->up_weight_q8,
                                               (NSUInteger)(context->intermediate * context->hidden));
                target->down_weight = new_weight(context->device, upload_blit, source->down_weight, source->down_weight_q8,
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
            context->final_norm_weight = new_private_buffer(context->device, upload_blit, final_norm_weight,
                                                            context->hidden * sizeof(uint16_t));
            context->lm_head_weight = new_weight(context->device, upload_blit, lm_head_weight, lm_head_q8,
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
            context->attention_scores = new_zero_buffer(
                context->device,
                (NSUInteger)context->query_heads * context->bucket * sizeof(float),
                MTLResourceStorageModePrivate
            );
            context->mlp = new_zero_buffer(context->device, intermediate_bytes, MTLResourceStorageModePrivate);
            context->final_norm = new_zero_buffer(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            context->logits = new_zero_buffer(context->device, (NSUInteger)context->vocab * sizeof(float), MTLResourceStorageModeShared);
            // Chained-decode residents. Embeddings are uploaded once into a
            // private buffer so the gather kernel reads them without host-visible
            // access. One threadgroup argmax partial per 4,096 vocab entries keeps
            // the two-stage reduction's final fold within a single simdgroup.
            if (embeddings != NULL) {
                context->embeddings = new_private_buffer(context->device, upload_blit, embeddings,
                                                         (NSUInteger)(context->vocab * context->hidden) * sizeof(uint16_t));
                context->argmax_partials = (context->vocab + 4095) / 4096;
                if (context->argmax_partials == 0) context->argmax_partials = 1;
                context->argmax_partial_keys = new_zero_buffer(context->device,
                                                               (NSUInteger)context->argmax_partials * sizeof(int32_t),
                                                               MTLResourceStorageModePrivate);
                context->argmax_partial_ids = new_zero_buffer(context->device,
                                                              (NSUInteger)context->argmax_partials * sizeof(uint32_t),
                                                              MTLResourceStorageModePrivate);
                // Token-id scratch is host-visible so the host reads back the k
                // ids after a chain and seeds the first step of the next chain.
                context->chain_token_ids = new_zero_buffer(context->device, sizeof(uint32_t), MTLResourceStorageModeShared);
                context->chain_input = new_zero_buffer(context->device, hidden_bytes, MTLResourceStorageModePrivate);
            }
            [upload_blit endEncoding];
            [upload_command commit];
            [upload_command waitUntilCompleted];
            if (upload_command.status == MTLCommandBufferStatusError) {
                set_error(upload_command.error.localizedDescription ?: @"private weight upload failed");
                return -6;
            }
            if (context->final_norm_weight == nil ||
                (context->lm_head_weight.fp16 == nil && context->lm_head_weight.q8 == nil) ||
                context->x_a == nil || context->x_b == nil || context->normalized == nil || context->query == nil ||
                context->key == nil || context->context == nil || context->attention_scores == nil ||
                context->mlp == nil || context->final_norm == nil ||
                context->logits == nil) {
                set_error(@"failed to allocate Metal step activation buffers");
                return -5;
            }
            if (embeddings != NULL &&
                (context->embeddings == nil || context->argmax_partial_keys == nil ||
                 context->argmax_partial_ids == nil || context->chain_token_ids == nil ||
                 context->chain_input == nil)) {
                set_error(@"failed to allocate Metal step chained-decode buffers");
                return -7;
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

static BOOL finish_profiled_command(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    double *gpu_seconds
) {
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
        set_error(command_buffer.error.localizedDescription ?: @"Metal step profiled command buffer failed");
        return NO;
    }
    // GPUStartTime and GPUEndTime are populated after completion and use the
    // same host-clock domain, so their difference is the encoder's GPU span.
    double start = command_buffer.GPUStartTime;
    double end = command_buffer.GPUEndTime;
    if (start > 0.0 && end >= start) {
        *gpu_seconds += end - start;
        context->timings.kernel_samples += 1;
    }
    return YES;
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
    // A full simdgroup hides the fixed launch cost of the 1,024-element norm.
    [encoder dispatchThreads:grid_size(32) threadsPerThreadgroup:group_size(32)];
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
    NSUInteger qkv_rows = context->quantized
        ? (NSUInteger)query_width + (NSUInteger)kv_width * 2
        : (NSUInteger)MAX(query_width, kv_width);
    NSUInteger qkv_threads = context->quantized ? qkv_rows * 32 : qkv_rows;
    [encoder dispatchThreads:grid_size(qkv_threads) threadsPerThreadgroup:group_size(context->quantized ? 256 : qkv_rows)];
    [encoder endEncoding];
}

// Rope-offset-aware form. The per-token wrapper below passes offset 0, so its
// generated command stream is byte-identical to the previous single-block code;
// the chain path passes a per-step head_dim block offset into a shared buffer.
static void encode_qk_norm_rope_offset(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    StepLayerBuffers *layer,
    id<MTLBuffer> rope_cos,
    id<MTLBuffer> rope_sin,
    NSUInteger rope_offset,
    uint32_t position
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->qk_norm_rope];
    [encoder setBuffer:context->query offset:0 atIndex:0];
    [encoder setBuffer:context->key offset:0 atIndex:1];
    [encoder setBuffer:layer->q_norm offset:0 atIndex:2];
    [encoder setBuffer:layer->k_norm offset:0 atIndex:3];
    [encoder setBuffer:rope_cos offset:rope_offset atIndex:4];
    [encoder setBuffer:rope_sin offset:rope_offset atIndex:5];
    [encoder setBuffer:layer->key_cache offset:0 atIndex:6];
    struct { uint32_t query_heads; uint32_t kv_heads; uint32_t head_dim; float epsilon; uint32_t capacity; uint32_t position; } config = {
        (uint32_t)context->query_heads, (uint32_t)context->kv_heads, (uint32_t)context->head_dim, context->epsilon,
        (uint32_t)context->bucket, (uint32_t)position
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:7];
    // One simdgroup owns each independent head; lane zero preserves the exact
    // ascending norm reduction while all lanes split independent output pairs.
    [encoder dispatchThreads:grid_size(MAX(context->query_heads, context->kv_heads) * 32) threadsPerThreadgroup:group_size(256)];
    [encoder endEncoding];
}

// Per-token wrapper: rope tables are a single head_dim block, so offset is 0.
static void encode_qk_norm_rope(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    StepLayerBuffers *layer,
    id<MTLBuffer> rope_cos,
    id<MTLBuffer> rope_sin,
    uint32_t position
) {
    encode_qk_norm_rope_offset(context, command_buffer, layer, rope_cos, rope_sin, 0, position);
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
    [encoder setBuffer:context->attention_scores offset:0 atIndex:4];
    struct { uint32_t query_heads; uint32_t kv_heads; uint32_t head_dim; uint32_t capacity; uint32_t position; } config = {
        (uint32_t)context->query_heads, (uint32_t)context->kv_heads, (uint32_t)context->head_dim,
        (uint32_t)context->bucket, position
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:5];
    // One simdgroup owns a query head. Its 32 lanes split independent KV
    // positions for the serial-order QK dots, then split value dimensions.
    [encoder dispatchThreads:grid_size(context->query_heads * 32) threadsPerThreadgroup:group_size(256)];
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
    uint32_t output_width,
    BOOL add_residual
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->matvec_residual];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:residual offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    set_weight(encoder, weight, 3, 4);
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; uint32_t add_residual; } config = {
        input_width, output_width, context->quantized, (uint32_t)add_residual
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:5];
    // F16 uses one lane per independent output row; the row dot itself stays
    // serial. Q8 packs four independent rows into each 32-lane simdgroup,
    // eight sub-lanes per row, each row still reduced by its own simd_sum.
    NSUInteger matvec_threads =
        context->quantized ? (NSUInteger)((output_width + 3) / 4) * 32 : output_width;
    [encoder dispatchThreads:grid_size(matvec_threads) threadsPerThreadgroup:group_size(context->quantized ? 256 : output_width)];
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
    // One simdgroup: lane zero keeps the exact serial norm reduction while the
    // remaining lanes split the independent residual and normalize writes.
    [encoder dispatchThreads:grid_size(32) threadsPerThreadgroup:group_size(32)];
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
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; uint32_t add_residual; } config = {
        (uint32_t)context->hidden, (uint32_t)context->intermediate, context->quantized, 0
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:6];
    NSUInteger gate_threads =
        context->quantized ? ((context->intermediate + 3) / 4) * 32 : context->intermediate;
    [encoder dispatchThreads:grid_size(gate_threads) threadsPerThreadgroup:group_size(context->quantized ? 256 : context->intermediate)];
    [encoder endEncoding];
}

static void encode_lm_head(Qwen3MetalStepContext *context, id<MTLCommandBuffer> command_buffer) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->lm_head];
    [encoder setBuffer:context->final_norm offset:0 atIndex:0];
    [encoder setBuffer:context->logits offset:0 atIndex:1];
    set_weight(encoder, &context->lm_head_weight, 2, 3);
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; uint32_t add_residual; } config = {
        (uint32_t)context->hidden, (uint32_t)context->vocab, context->quantized, 0
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:4];
    // The large f16 vocabulary projection is row-parallel: each physical
    // lane owns one full serial dot, avoiding any accumulation reordering.
    // Q8 packs four vocabulary rows per simdgroup, eight sub-lanes each.
    NSUInteger lm_head_threads =
        context->quantized ? ((context->vocab + 3) / 4) * 32 : context->vocab;
    [encoder dispatchThreads:grid_size(lm_head_threads) threadsPerThreadgroup:group_size(context->quantized ? 256 : context->vocab)];
    [encoder endEncoding];
}

// Argmax the resident logits into `token_out[out_offset]`, matching the host
// greedy sampler's highest-logit / lowest-id rule (see the Metal kernels).
static void encode_argmax_offset(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> token_out,
    NSUInteger out_offset
) {
    struct { uint32_t vocab; uint32_t partials; } config = {
        (uint32_t)context->vocab, (uint32_t)context->argmax_partials
    };
    id<MTLComputeCommandEncoder> partial = [command_buffer computeCommandEncoder];
    [partial setComputePipelineState:context->argmax_partial];
    [partial setBuffer:context->logits offset:0 atIndex:0];
    [partial setBuffer:context->argmax_partial_keys offset:0 atIndex:1];
    [partial setBuffer:context->argmax_partial_ids offset:0 atIndex:2];
    [partial setBytes:&config length:sizeof(config) atIndex:3];
    // One thread per partial: each threadgroup owns exactly one vocab slice and
    // scans it serially. The grid is one thread per partial with a single-thread
    // group so threadgroup_position_in_grid indexes the slice.
    [partial dispatchThreadgroups:MTLSizeMake((NSUInteger)context->argmax_partials, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [partial endEncoding];

    id<MTLComputeCommandEncoder> final = [command_buffer computeCommandEncoder];
    [final setComputePipelineState:context->argmax_final];
    [final setBuffer:context->argmax_partial_keys offset:0 atIndex:0];
    [final setBuffer:context->argmax_partial_ids offset:0 atIndex:1];
    [final setBuffer:token_out offset:out_offset atIndex:2];
    [final setBytes:&config length:sizeof(config) atIndex:3];
    [final dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [final endEncoding];
}

// Gather the embedding row named by `token_in[in_offset]` into `input_out`.
static void encode_embedding_gather_offset(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> token_in,
    NSUInteger in_offset,
    id<MTLBuffer> input_out
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->embedding_gather];
    [encoder setBuffer:context->embeddings offset:0 atIndex:0];
    [encoder setBuffer:token_in offset:in_offset atIndex:1];
    [encoder setBuffer:input_out offset:0 atIndex:2];
    struct { uint32_t hidden; } config = { (uint32_t)context->hidden };
    [encoder setBytes:&config length:sizeof(config) atIndex:3];
    [encoder dispatchThreads:grid_size(context->hidden) threadsPerThreadgroup:group_size(context->hidden)];
    [encoder endEncoding];
}

// Lazily create the four mat-mat pipelines and the column-sized scratch buffers
// used by batched verification. Called once on the first verify_batch; the
// per-token and chained paths never invoke it, so their behavior and resident
// memory are unchanged. Buffers are sized for METAL_STEP_MAX_BATCH_K columns and
// reused for any smaller batch.
static BOOL ensure_batch_resources(Qwen3MetalStepContext *context) {
    if (context->batch_capacity >= METAL_STEP_MAX_BATCH_K) return YES;
    if (context->qkv_matvec_batch == nil) {
        context->qkv_matvec_batch = pipeline(context->device, context->library, @"metal_step_qkv_matvec_batch");
        context->matvec_residual_batch = pipeline(context->device, context->library, @"metal_step_matvec_residual_batch");
        context->gate_up_swiglu_batch = pipeline(context->device, context->library, @"metal_step_gate_up_swiglu_batch");
        context->lm_head_batch = pipeline(context->device, context->library, @"metal_step_lm_head_batch");
        if (context->qkv_matvec_batch == nil || context->matvec_residual_batch == nil ||
            context->gate_up_swiglu_batch == nil || context->lm_head_batch == nil) {
            set_error(@"failed to compile batched Metal step verification pipelines");
            return NO;
        }
    }
    uint64_t query_width = context->query_heads * context->head_dim;
    uint64_t kv_width = context->kv_heads * context->head_dim;
    NSUInteger k = METAL_STEP_MAX_BATCH_K;
    NSUInteger hidden_bytes = (NSUInteger)context->hidden * sizeof(uint16_t);
    NSUInteger query_bytes = (NSUInteger)query_width * sizeof(uint16_t);
    NSUInteger kv_bytes = (NSUInteger)kv_width * sizeof(uint16_t);
    NSUInteger intermediate_bytes = (NSUInteger)context->intermediate * sizeof(uint16_t);
    context->batch_input = new_zero_buffer(context->device, k * hidden_bytes, MTLResourceStorageModePrivate);
    context->batch_x_b = new_zero_buffer(context->device, k * hidden_bytes, MTLResourceStorageModePrivate);
    context->batch_normalized = new_zero_buffer(context->device, k * hidden_bytes, MTLResourceStorageModePrivate);
    context->batch_final_norm = new_zero_buffer(context->device, k * hidden_bytes, MTLResourceStorageModePrivate);
    context->batch_query = new_zero_buffer(context->device, k * query_bytes, MTLResourceStorageModePrivate);
    context->batch_key = new_zero_buffer(context->device, k * kv_bytes, MTLResourceStorageModePrivate);
    context->batch_context = new_zero_buffer(context->device, k * query_bytes, MTLResourceStorageModePrivate);
    context->batch_mlp = new_zero_buffer(context->device, k * intermediate_bytes, MTLResourceStorageModePrivate);
    context->batch_attention_scores = new_zero_buffer(
        context->device,
        k * (NSUInteger)context->query_heads * (NSUInteger)context->bucket * sizeof(float),
        MTLResourceStorageModePrivate
    );
    // Shared so the host can read back every column's logits for the byte-exact
    // verification gate without a per-column blit.
    context->batch_logits = new_zero_buffer(
        context->device,
        k * (NSUInteger)context->vocab * sizeof(float),
        MTLResourceStorageModeShared
    );
    context->batch_argmax_partial_keys = new_zero_buffer(
        context->device, k * (NSUInteger)context->argmax_partials * sizeof(int32_t), MTLResourceStorageModePrivate);
    context->batch_argmax_partial_ids = new_zero_buffer(
        context->device, k * (NSUInteger)context->argmax_partials * sizeof(uint32_t), MTLResourceStorageModePrivate);
    if (context->batch_input == nil || context->batch_x_b == nil || context->batch_normalized == nil ||
        context->batch_final_norm == nil || context->batch_query == nil || context->batch_key == nil ||
        context->batch_context == nil || context->batch_mlp == nil || context->batch_attention_scores == nil ||
        context->batch_logits == nil || context->batch_argmax_partial_keys == nil ||
        context->batch_argmax_partial_ids == nil) {
        set_error(@"failed to allocate batched Metal step verification buffers");
        return NO;
    }
    context->batch_capacity = METAL_STEP_MAX_BATCH_K;
    return YES;
}

// Offset-addressed wrappers around the single-token kernels. They dispatch the
// exact same pipelines as the per-token path, pointing each at one batch column
// through a buffer offset, so every column's norm/RoPE/attention/argmax/gather
// is bit-identical to a standalone single-token step. Only the heavy projections
// use the new mat-mat kernels below.
static void encode_rmsnorm_to(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> input,
    NSUInteger input_offset,
    id<MTLBuffer> output,
    NSUInteger output_offset,
    id<MTLBuffer> weight,
    uint32_t width,
    float epsilon
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->rmsnorm];
    [encoder setBuffer:input offset:input_offset atIndex:0];
    [encoder setBuffer:output offset:output_offset atIndex:1];
    [encoder setBuffer:weight offset:0 atIndex:2];
    struct { uint32_t width; float epsilon; } config = { width, epsilon };
    [encoder setBytes:&config length:sizeof(config) atIndex:3];
    [encoder dispatchThreads:grid_size(32) threadsPerThreadgroup:group_size(32)];
    [encoder endEncoding];
}

static void encode_qk_norm_rope_to(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> query,
    NSUInteger query_offset,
    id<MTLBuffer> key,
    NSUInteger key_offset,
    StepLayerBuffers *layer,
    id<MTLBuffer> rope_cos,
    id<MTLBuffer> rope_sin,
    NSUInteger rope_offset,
    uint32_t position
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->qk_norm_rope];
    [encoder setBuffer:query offset:query_offset atIndex:0];
    [encoder setBuffer:key offset:key_offset atIndex:1];
    [encoder setBuffer:layer->q_norm offset:0 atIndex:2];
    [encoder setBuffer:layer->k_norm offset:0 atIndex:3];
    [encoder setBuffer:rope_cos offset:rope_offset atIndex:4];
    [encoder setBuffer:rope_sin offset:rope_offset atIndex:5];
    [encoder setBuffer:layer->key_cache offset:0 atIndex:6];
    struct { uint32_t query_heads; uint32_t kv_heads; uint32_t head_dim; float epsilon; uint32_t capacity; uint32_t position; } config = {
        (uint32_t)context->query_heads, (uint32_t)context->kv_heads, (uint32_t)context->head_dim, context->epsilon,
        (uint32_t)context->bucket, position
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:7];
    [encoder dispatchThreads:grid_size(MAX(context->query_heads, context->kv_heads) * 32) threadsPerThreadgroup:group_size(256)];
    [encoder endEncoding];
}

static void encode_attention_to(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> query,
    NSUInteger query_offset,
    StepLayerBuffers *layer,
    id<MTLBuffer> output,
    NSUInteger output_offset,
    id<MTLBuffer> scores,
    NSUInteger scores_offset,
    uint32_t position
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->attention];
    [encoder setBuffer:query offset:query_offset atIndex:0];
    [encoder setBuffer:layer->key_cache offset:0 atIndex:1];
    [encoder setBuffer:layer->value_cache offset:0 atIndex:2];
    [encoder setBuffer:output offset:output_offset atIndex:3];
    [encoder setBuffer:scores offset:scores_offset atIndex:4];
    struct { uint32_t query_heads; uint32_t kv_heads; uint32_t head_dim; uint32_t capacity; uint32_t position; } config = {
        (uint32_t)context->query_heads, (uint32_t)context->kv_heads, (uint32_t)context->head_dim,
        (uint32_t)context->bucket, position
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:5];
    [encoder dispatchThreads:grid_size(context->query_heads * 32) threadsPerThreadgroup:group_size(256)];
    [encoder endEncoding];
}

static void encode_residual_rmsnorm_to(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> projection,
    NSUInteger projection_offset,
    id<MTLBuffer> residual,
    NSUInteger residual_offset,
    id<MTLBuffer> normalized,
    NSUInteger normalized_offset,
    id<MTLBuffer> weight
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->residual_rmsnorm];
    [encoder setBuffer:projection offset:projection_offset atIndex:0];
    [encoder setBuffer:residual offset:residual_offset atIndex:1];
    [encoder setBuffer:normalized offset:normalized_offset atIndex:2];
    [encoder setBuffer:weight offset:0 atIndex:3];
    struct { uint32_t width; float epsilon; } config = { (uint32_t)context->hidden, context->epsilon };
    [encoder setBytes:&config length:sizeof(config) atIndex:4];
    [encoder dispatchThreads:grid_size(32) threadsPerThreadgroup:group_size(32)];
    [encoder endEncoding];
}

static void encode_embedding_gather_to(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> token_in,
    NSUInteger in_offset,
    id<MTLBuffer> input_out,
    NSUInteger out_offset
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->embedding_gather];
    [encoder setBuffer:context->embeddings offset:0 atIndex:0];
    [encoder setBuffer:token_in offset:in_offset atIndex:1];
    [encoder setBuffer:input_out offset:out_offset atIndex:2];
    struct { uint32_t hidden; } config = { (uint32_t)context->hidden };
    [encoder setBytes:&config length:sizeof(config) atIndex:3];
    [encoder dispatchThreads:grid_size(context->hidden) threadsPerThreadgroup:group_size(context->hidden)];
    [encoder endEncoding];
}

static void encode_argmax_to(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> logits,
    NSUInteger logits_offset,
    id<MTLBuffer> partial_keys,
    NSUInteger keys_offset,
    id<MTLBuffer> partial_ids,
    NSUInteger ids_offset,
    id<MTLBuffer> token_out,
    NSUInteger out_offset
) {
    struct { uint32_t vocab; uint32_t partials; } config = {
        (uint32_t)context->vocab, (uint32_t)context->argmax_partials
    };
    id<MTLComputeCommandEncoder> partial = [command_buffer computeCommandEncoder];
    [partial setComputePipelineState:context->argmax_partial];
    [partial setBuffer:logits offset:logits_offset atIndex:0];
    [partial setBuffer:partial_keys offset:keys_offset atIndex:1];
    [partial setBuffer:partial_ids offset:ids_offset atIndex:2];
    [partial setBytes:&config length:sizeof(config) atIndex:3];
    [partial dispatchThreadgroups:MTLSizeMake((NSUInteger)context->argmax_partials, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [partial endEncoding];

    id<MTLComputeCommandEncoder> final = [command_buffer computeCommandEncoder];
    [final setComputePipelineState:context->argmax_final];
    [final setBuffer:partial_keys offset:keys_offset atIndex:0];
    [final setBuffer:partial_ids offset:ids_offset atIndex:1];
    [final setBuffer:token_out offset:out_offset atIndex:2];
    [final setBytes:&config length:sizeof(config) atIndex:3];
    [final dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [final endEncoding];
}

// Batched projection encoders: one dispatch runs all `batch` columns through a
// weight, streaming each weight row once across the columns.
static void encode_qkv_batch(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    StepLayerBuffers *layer,
    id<MTLBuffer> input,
    uint32_t base_position,
    uint32_t batch
) {
    uint32_t query_width = (uint32_t)(context->query_heads * context->head_dim);
    uint32_t kv_width = (uint32_t)(context->kv_heads * context->head_dim);
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->qkv_matvec_batch];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:context->batch_query offset:0 atIndex:1];
    [encoder setBuffer:context->batch_key offset:0 atIndex:2];
    [encoder setBuffer:layer->value_cache offset:0 atIndex:3];
    set_weight(encoder, &layer->q_weight, 4, 5);
    set_weight(encoder, &layer->k_weight, 6, 7);
    set_weight(encoder, &layer->v_weight, 8, 9);
    struct { uint32_t input_width; uint32_t query_width; uint32_t kv_width; uint32_t head_dim; uint32_t capacity; uint32_t position; uint32_t quantized; uint32_t batch; } config = {
        (uint32_t)context->hidden, query_width, kv_width, (uint32_t)context->head_dim,
        (uint32_t)context->bucket, base_position, context->quantized, batch
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:10];
    NSUInteger qkv_rows = (NSUInteger)query_width + (NSUInteger)kv_width * 2;
    NSUInteger qkv_threads = context->quantized ? qkv_rows * 32 : (NSUInteger)MAX(query_width, kv_width);
    [encoder dispatchThreads:grid_size(qkv_threads) threadsPerThreadgroup:group_size(context->quantized ? 256 : qkv_threads)];
    [encoder endEncoding];
}

static void encode_matvec_residual_batch(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> input,
    id<MTLBuffer> residual,
    id<MTLBuffer> output,
    StepWeight *weight,
    uint32_t input_width,
    uint32_t output_width,
    BOOL add_residual,
    uint32_t batch
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->matvec_residual_batch];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:residual offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    set_weight(encoder, weight, 3, 4);
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; uint32_t add_residual; uint32_t batch; } config = {
        input_width, output_width, context->quantized, (uint32_t)add_residual, batch
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:5];
    NSUInteger threads = context->quantized ? (NSUInteger)((output_width + 3) / 4) * 32 : output_width;
    [encoder dispatchThreads:grid_size(threads) threadsPerThreadgroup:group_size(context->quantized ? 256 : output_width)];
    [encoder endEncoding];
}

static void encode_gate_up_batch(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    StepLayerBuffers *layer,
    id<MTLBuffer> input,
    uint32_t batch
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->gate_up_swiglu_batch];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:context->batch_mlp offset:0 atIndex:1];
    set_weight(encoder, &layer->gate_weight, 2, 3);
    set_weight(encoder, &layer->up_weight, 4, 5);
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; uint32_t add_residual; uint32_t batch; } config = {
        (uint32_t)context->hidden, (uint32_t)context->intermediate, context->quantized, 0, batch
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:6];
    NSUInteger threads = context->quantized ? ((context->intermediate + 3) / 4) * 32 : context->intermediate;
    [encoder dispatchThreads:grid_size(threads) threadsPerThreadgroup:group_size(context->quantized ? 256 : context->intermediate)];
    [encoder endEncoding];
}

static void encode_lm_head_batch(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    uint32_t batch
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->lm_head_batch];
    [encoder setBuffer:context->batch_final_norm offset:0 atIndex:0];
    [encoder setBuffer:context->batch_logits offset:0 atIndex:1];
    set_weight(encoder, &context->lm_head_weight, 2, 3);
    struct { uint32_t input_width; uint32_t output_width; uint32_t quantized; uint32_t add_residual; uint32_t batch; } config = {
        (uint32_t)context->hidden, (uint32_t)context->vocab, context->quantized, 0, batch
    };
    [encoder setBytes:&config length:sizeof(config) atIndex:4];
    NSUInteger threads = context->quantized ? ((context->vocab + 3) / 4) * 32 : context->vocab;
    [encoder dispatchThreads:grid_size(threads) threadsPerThreadgroup:group_size(context->quantized ? 256 : context->vocab)];
    [encoder endEncoding];
}

// One batched forward pass for `batch` draft columns starting at `base_position`.
// Column k feeds batch_input row k (already gathered from the proposal tokens)
// and produces logits row k plus an argmax in argmax_out[k]. Weights stream once
// per layer across all columns; per-column norm/RoPE/attention reuse the
// single-token kernels through buffer offsets. KV slots base_position..+batch-1
// are written before any column's attention runs, so column k's causal prefix
// (positions <= base_position+k) is fully resident and identical to the
// sequential path.
static void encode_forward_batch(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> rope_cos,
    id<MTLBuffer> rope_sin,
    uint32_t base_position,
    uint32_t batch,
    float epsilon,
    id<MTLBuffer> argmax_out
) {
    id<MTLBuffer> current = context->batch_input;
    id<MTLBuffer> next = context->batch_x_b;
    NSUInteger hidden_bytes = (NSUInteger)context->hidden * sizeof(uint16_t);
    NSUInteger query_bytes = (NSUInteger)(context->query_heads * context->head_dim) * sizeof(uint16_t);
    NSUInteger kv_bytes = (NSUInteger)(context->kv_heads * context->head_dim) * sizeof(uint16_t);
    NSUInteger head_dim_bytes = (NSUInteger)context->head_dim * sizeof(uint16_t);
    NSUInteger score_stride = (NSUInteger)context->query_heads * (NSUInteger)context->bucket * sizeof(float);
    NSUInteger partial_stride = (NSUInteger)context->argmax_partials * sizeof(uint32_t);
    NSUInteger vocab_bytes = (NSUInteger)context->vocab * sizeof(float);
    uint32_t query_width = (uint32_t)(context->query_heads * context->head_dim);
    for (uint64_t index = 0; index < context->layer_count; ++index) {
        StepLayerBuffers *layer = &context->layers[index];
        for (uint32_t k = 0; k < batch; ++k) {
            encode_rmsnorm_to(context, command_buffer, current, k * hidden_bytes,
                              context->batch_normalized, k * hidden_bytes, layer->input_norm,
                              (uint32_t)context->hidden, epsilon);
        }
        encode_qkv_batch(context, command_buffer, layer, context->batch_normalized, base_position, batch);
        for (uint32_t k = 0; k < batch; ++k) {
            encode_qk_norm_rope_to(context, command_buffer, context->batch_query, k * query_bytes,
                                   context->batch_key, k * kv_bytes, layer, rope_cos, rope_sin,
                                   k * head_dim_bytes, base_position + k);
        }
        for (uint32_t k = 0; k < batch; ++k) {
            encode_attention_to(context, command_buffer, context->batch_query, k * query_bytes, layer,
                                context->batch_context, k * query_bytes, context->batch_attention_scores,
                                k * score_stride, base_position + k);
        }
        encode_matvec_residual_batch(context, command_buffer, context->batch_context, current, next,
                                     &layer->o_weight, query_width, (uint32_t)context->hidden, NO, batch);
        for (uint32_t k = 0; k < batch; ++k) {
            encode_residual_rmsnorm_to(context, command_buffer, next, k * hidden_bytes, current,
                                       k * hidden_bytes, context->batch_normalized, k * hidden_bytes,
                                       layer->post_attention_norm);
        }
        encode_gate_up_batch(context, command_buffer, layer, context->batch_normalized, batch);
        encode_matvec_residual_batch(context, command_buffer, context->batch_mlp, next, current,
                                     &layer->down_weight, (uint32_t)context->intermediate,
                                     (uint32_t)context->hidden, YES, batch);
    }
    for (uint32_t k = 0; k < batch; ++k) {
        encode_rmsnorm_to(context, command_buffer, current, k * hidden_bytes,
                          context->batch_final_norm, k * hidden_bytes, context->final_norm_weight,
                          (uint32_t)context->hidden, epsilon);
    }
    encode_lm_head_batch(context, command_buffer, batch);
    for (uint32_t k = 0; k < batch; ++k) {
        encode_argmax_to(context, command_buffer, context->batch_logits, k * vocab_bytes,
                         context->batch_argmax_partial_keys, k * partial_stride,
                         context->batch_argmax_partial_ids, k * partial_stride,
                         argmax_out, k * sizeof(uint32_t));
    }
}

// Batched speculative verification: run `steps` draft tokens through the
// transformer in ONE forward pass (weights streamed once per layer) instead of
// `steps` dependent single-token steps. Outputs are the greedy id after each
// supplied token (argmaxes_out) and, when logits_out is non-null, the full
// per-column f32 logits for the byte-exact gate. Produces results identical to
// `steps` sequential single-token steps at the same positions.
int32_t synapse_qwen3_metal_step_verify_batch(
    void *raw,
    uint64_t position,
    const uint32_t *token_ids,
    uint32_t steps,
    const uint16_t *rope_cos,
    const uint16_t *rope_sin,
    uint32_t *argmaxes_out,
    float *logits_out,
    float epsilon
) {
    @autoreleasepool {
        @try {
            metal_step_error[0] = '\0';
            double feed_started = [NSDate timeIntervalSinceReferenceDate];
            Qwen3MetalStepContext *context = raw;
            if (context == NULL || token_ids == NULL || rope_cos == NULL || rope_sin == NULL ||
                argmaxes_out == NULL || context->layers == NULL || context->embeddings == nil ||
                steps == 0 || steps > METAL_STEP_MAX_BATCH_K || position + steps > context->bucket) {
                set_error(@"invalid Metal step batched verification arguments");
                return -1;
            }
            if (!ensure_batch_resources(context)) {
                return -2;
            }
            NSUInteger rope_span = (NSUInteger)steps * (NSUInteger)context->head_dim;
            id<MTLBuffer> cosine_buffer = [context->device newBufferWithBytes:rope_cos
                length:rope_span * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> sine_buffer = [context->device newBufferWithBytes:rope_sin
                length:rope_span * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> proposal_buffer = [context->device newBufferWithBytes:token_ids
                length:(NSUInteger)steps * sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> argmax_buffer = [context->device newBufferWithLength:(NSUInteger)steps * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
            if (cosine_buffer == nil || sine_buffer == nil || proposal_buffer == nil ||
                argmax_buffer == nil || command_buffer == nil) {
                [cosine_buffer release];
                [sine_buffer release];
                [proposal_buffer release];
                [argmax_buffer release];
                set_error(@"failed to allocate Metal step batched verification buffers");
                return -2;
            }
            NSUInteger hidden_bytes = (NSUInteger)context->hidden * sizeof(uint16_t);
            for (uint32_t step = 0; step < steps; ++step) {
                encode_embedding_gather_to(context, command_buffer, proposal_buffer,
                                           (NSUInteger)step * sizeof(uint32_t),
                                           context->batch_input, (NSUInteger)step * hidden_bytes);
            }
            encode_forward_batch(context, command_buffer, cosine_buffer, sine_buffer,
                                 (uint32_t)position, steps, epsilon, argmax_buffer);
            context->timings.feed_wall_s += [NSDate timeIntervalSinceReferenceDate] - feed_started;
            double started = [NSDate timeIntervalSinceReferenceDate];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            context->timings.execute_wall_s += [NSDate timeIntervalSinceReferenceDate] - started;
            BOOL ok = command_buffer.status != MTLCommandBufferStatusError;
            if (!ok) {
                set_error(command_buffer.error.localizedDescription ?: @"Metal step batched verification command buffer failed");
            } else {
                double readback_started = [NSDate timeIntervalSinceReferenceDate];
                memcpy(argmaxes_out, argmax_buffer.contents, (NSUInteger)steps * sizeof(uint32_t));
                if (logits_out != NULL) {
                    memcpy(logits_out, context->batch_logits.contents,
                           (NSUInteger)steps * (NSUInteger)context->vocab * sizeof(float));
                }
                context->timings.logits_readback_wall_s += [NSDate timeIntervalSinceReferenceDate] - readback_started;
                context->timings.step_calls += steps;
            }
            [cosine_buffer release];
            [sine_buffer release];
            [proposal_buffer release];
            [argmax_buffer release];
            return ok ? 0 : -3;
        } @catch (NSException *exception) {
            set_error(exception.reason);
            return -100;
        }
    }
}

// Encode the full transformer forward pass for one chained step, reading `input`
// and advancing every layer's KV cache at `position`. This mirrors the per-token
// encode order exactly; the only difference is that `position`, `input`, and the
// per-step rope block (selected by `rope_offset` bytes) vary within the chain
// instead of across command buffers.
static void encode_forward(
    Qwen3MetalStepContext *context,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> input,
    id<MTLBuffer> rope_cos,
    id<MTLBuffer> rope_sin,
    NSUInteger rope_offset,
    uint32_t position,
    float epsilon
) {
    id<MTLBuffer> current = input;
    id<MTLBuffer> next = context->x_a;
    for (uint64_t index = 0; index < context->layer_count; ++index) {
        StepLayerBuffers *layer = &context->layers[index];
        encode_rmsnorm(context, command_buffer, current, context->normalized, layer->input_norm,
                       (uint32_t)context->hidden, epsilon);
        encode_qkv(context, command_buffer, layer, context->normalized, position);
        encode_qk_norm_rope_offset(context, command_buffer, layer, rope_cos, rope_sin, rope_offset, position);
        encode_attention(context, command_buffer, layer, position);
        encode_matvec_residual(context, command_buffer, context->context, current, next,
                               &layer->o_weight, (uint32_t)context->query_heads * (uint32_t)context->head_dim,
                               (uint32_t)context->hidden, NO);
        encode_residual_rmsnorm(context, command_buffer, next, current, context->normalized,
                                layer->post_attention_norm);
        encode_gate_up(context, command_buffer, layer, context->normalized);
        encode_matvec_residual(context, command_buffer, context->mlp, next, current,
                               &layer->down_weight, (uint32_t)context->intermediate, (uint32_t)context->hidden, YES);
    }
    encode_rmsnorm(context, command_buffer, current, context->final_norm, context->final_norm_weight,
                   (uint32_t)context->hidden, epsilon);
    encode_lm_head(context, command_buffer);
}

// Verification is the proposal-fed mode of the existing chained-step encoder.
// The host supplies every draft id, but each full forward pass, rope selection,
// in-slot KV write, and on-GPU argmax is shared with the greedy chain below.
// Outputs are the greedy ids after each supplied token; the Rust session aligns
// them with the pending logits that predict the first supplied token.
int32_t synapse_qwen3_metal_step_verify(
    void *raw,
    uint64_t position,
    const uint32_t *token_ids,
    uint32_t steps,
    const uint16_t *rope_cos,
    const uint16_t *rope_sin,
    uint32_t *argmaxes_out,
    float epsilon
) {
    @autoreleasepool {
        @try {
            metal_step_error[0] = '\0';
            double feed_started = [NSDate timeIntervalSinceReferenceDate];
            Qwen3MetalStepContext *context = raw;
            if (context == NULL || token_ids == NULL || rope_cos == NULL || rope_sin == NULL ||
                argmaxes_out == NULL || context->layers == NULL || context->embeddings == nil ||
                steps == 0 || position + steps > context->bucket) {
                set_error(@"invalid Metal step verification arguments");
                return -1;
            }
            NSUInteger rope_span = (NSUInteger)steps * (NSUInteger)context->head_dim;
            id<MTLBuffer> cosine_buffer = [context->device newBufferWithBytes:rope_cos
                length:rope_span * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> sine_buffer = [context->device newBufferWithBytes:rope_sin
                length:rope_span * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> proposal_buffer = [context->device newBufferWithBytes:token_ids
                length:(NSUInteger)steps * sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> argmax_buffer = [context->device newBufferWithLength:(NSUInteger)steps * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
            if (cosine_buffer == nil || sine_buffer == nil || proposal_buffer == nil ||
                argmax_buffer == nil || command_buffer == nil) {
                [cosine_buffer release];
                [sine_buffer release];
                [proposal_buffer release];
                [argmax_buffer release];
                set_error(@"failed to allocate Metal step verification buffers");
                return -2;
            }
            NSUInteger head_dim_bytes = (NSUInteger)context->head_dim * sizeof(uint16_t);
            for (uint32_t step = 0; step < steps; ++step) {
                encode_embedding_gather_offset(
                    context,
                    command_buffer,
                    proposal_buffer,
                    (NSUInteger)step * sizeof(uint32_t),
                    context->chain_input
                );
                encode_forward(context, command_buffer, context->chain_input,
                               cosine_buffer, sine_buffer, (NSUInteger)step * head_dim_bytes,
                               (uint32_t)position + step, epsilon);
                encode_argmax_offset(context, command_buffer, argmax_buffer,
                                     (NSUInteger)step * sizeof(uint32_t));
            }
            context->timings.feed_wall_s += [NSDate timeIntervalSinceReferenceDate] - feed_started;
            double started = [NSDate timeIntervalSinceReferenceDate];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            context->timings.execute_wall_s += [NSDate timeIntervalSinceReferenceDate] - started;
            BOOL ok = command_buffer.status != MTLCommandBufferStatusError;
            if (!ok) {
                set_error(command_buffer.error.localizedDescription ?: @"Metal step verification command buffer failed");
            } else {
                double readback_started = [NSDate timeIntervalSinceReferenceDate];
                memcpy(argmaxes_out, argmax_buffer.contents, (NSUInteger)steps * sizeof(uint32_t));
                context->timings.logits_readback_wall_s += [NSDate timeIntervalSinceReferenceDate] - readback_started;
                context->timings.step_calls += steps;
            }
            [cosine_buffer release];
            [sine_buffer release];
            [proposal_buffer release];
            [argmax_buffer release];
            return ok ? 0 : -3;
        } @catch (NSException *exception) {
            set_error(exception.reason);
            return -100;
        }
    }
}

// Chained multi-token decode: encode `steps` full forward passes plus an
// on-GPU argmax into a single command buffer, gathering each step's input token
// from the previous step's device-side argmax output. Position advances per
// step; rope tables are supplied host-side for the whole span (one head_dim
// block per step). The first step's input token is seeded by the host in
// token_in_first; subsequent steps read the id the argmax wrote. After the
// chain completes the host reads the `steps` token ids back at once (4*steps
// bytes) instead of a 604KB logits readback per token. This is only correct
// because the embedding gather and argmax are byte-exact with the per-token
// host path, so the produced token stream is identical to k=1.
int32_t synapse_qwen3_metal_step_chain(
    void *raw,
    uint64_t position,
    uint32_t steps,
    uint32_t token_in_first,
    const uint16_t *rope_cos,
    const uint16_t *rope_sin,
    uint32_t *token_ids_out,
    float epsilon
) {
    @autoreleasepool {
        @try {
            metal_step_error[0] = '\0';
            double feed_started = [NSDate timeIntervalSinceReferenceDate];
            Qwen3MetalStepContext *context = raw;
            if (context == NULL || rope_cos == NULL || rope_sin == NULL || token_ids_out == NULL ||
                context->layers == NULL || context->embeddings == nil || steps == 0 ||
                position + steps > context->bucket) {
                set_error(@"invalid Metal step chain arguments");
                return -1;
            }
            NSUInteger rope_span = (NSUInteger)steps * (NSUInteger)context->head_dim;
            id<MTLBuffer> cosine_buffer = [context->device newBufferWithBytes:rope_cos
                length:rope_span * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> sine_buffer = [context->device newBufferWithBytes:rope_sin
                length:rope_span * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            // A host-visible id buffer holds the seed token and each step's argmax
            // output; the host reads all `steps` ids back after completion.
            id<MTLBuffer> ids_buffer = [context->device newBufferWithLength:(NSUInteger)steps * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
            if (cosine_buffer == nil || sine_buffer == nil || ids_buffer == nil || command_buffer == nil) {
                [cosine_buffer release];
                [sine_buffer release];
                [ids_buffer release];
                set_error(@"failed to allocate Metal step chain buffers");
                return -2;
            }
            // Seed step 0's token; steps > 0 read the id the prior argmax wrote.
            *(uint32_t *)context->chain_token_ids.contents = token_in_first;
            NSUInteger head_dim_bytes = (NSUInteger)context->head_dim * sizeof(uint16_t);
            for (uint32_t step = 0; step < steps; ++step) {
                id<MTLBuffer> step_ids = (step == 0) ? context->chain_token_ids : ids_buffer;
                NSUInteger id_offset = (step == 0) ? 0 : (NSUInteger)(step - 1) * sizeof(uint32_t);
                // Gather this step's input from the token produced upstream: the
                // seed for step 0, or step-1's argmax output otherwise.
                encode_embedding_gather_offset(context, command_buffer, step_ids, id_offset, context->chain_input);
                // Select this step's head_dim rope block from the shared buffers.
                encode_forward(context, command_buffer, context->chain_input,
                               cosine_buffer, sine_buffer, (NSUInteger)step * head_dim_bytes,
                               (uint32_t)position + step, epsilon);
                // Argmax this step's logits into slot `step` of the id buffer.
                encode_argmax_offset(context, command_buffer, ids_buffer, (NSUInteger)step * sizeof(uint32_t));
            }
            context->timings.feed_wall_s += [NSDate timeIntervalSinceReferenceDate] - feed_started;
            double started = [NSDate timeIntervalSinceReferenceDate];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            context->timings.execute_wall_s += [NSDate timeIntervalSinceReferenceDate] - started;
            BOOL ok = command_buffer.status != MTLCommandBufferStatusError;
            if (!ok) {
                set_error(command_buffer.error.localizedDescription ?: @"Metal step chain command buffer failed");
            } else {
                double readback_started = [NSDate timeIntervalSinceReferenceDate];
                memcpy(token_ids_out, ids_buffer.contents, (NSUInteger)steps * sizeof(uint32_t));
                context->timings.logits_readback_wall_s += [NSDate timeIntervalSinceReferenceDate] - readback_started;
                context->timings.step_calls += steps;
            }
            [cosine_buffer release];
            [sine_buffer release];
            [ids_buffer release];
            return ok ? 0 : -3;
        } @catch (NSException *exception) {
            set_error(exception.reason);
            return -100;
        }
    }
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
            id<MTLCommandBuffer> command_buffer = context->profile_kernels ? nil : [context->queue commandBuffer];
            if (input_buffer == nil || cosine_buffer == nil || sine_buffer == nil ||
                (!context->profile_kernels && command_buffer == nil)) {
                [input_buffer release];
                [cosine_buffer release];
                [sine_buffer release];
                set_error(@"failed to allocate Metal step input buffers");
                return -2;
            }
            id<MTLBuffer> current = input_buffer;
            id<MTLBuffer> next = context->x_a;
            BOOL ok = YES;
            double profile_started = context->profile_kernels ? [NSDate timeIntervalSinceReferenceDate] : 0.0;
            if (context->profile_kernels) {
                // A profiled invocation deliberately serializes each kernel
                // class behind its predecessor. That makes the command-buffer
                // GPU span attributable without perturbing normal execution.
#define PROFILE_KERNEL(FIELD, ...) do { \
                    if (ok) { \
                        id<MTLCommandBuffer> profiled_command = [context->queue commandBuffer]; \
                        if (profiled_command == nil) { \
                            set_error(@"failed to allocate profiled Metal step command buffer"); \
                            ok = NO; \
                        } else { \
                            __VA_ARGS__; \
                            ok = finish_profiled_command(context, profiled_command, &context->timings.FIELD); \
                        } \
                    } \
                } while (0)
                for (uint64_t index = 0; index < context->layer_count; ++index) {
                    StepLayerBuffers *layer = &context->layers[index];
                    PROFILE_KERNEL(kernel_rmsnorm_s, encode_rmsnorm(context, profiled_command, current, context->normalized,
                                                                    layer->input_norm, (uint32_t)context->hidden, epsilon));
                    PROFILE_KERNEL(kernel_qkv_matvec_s, encode_qkv(context, profiled_command, layer, context->normalized,
                                                                   (uint32_t)position));
                    PROFILE_KERNEL(kernel_qk_norm_rope_s, encode_qk_norm_rope(context, profiled_command, layer,
                                                                               cosine_buffer, sine_buffer, (uint32_t)position));
                    PROFILE_KERNEL(kernel_attention_s, encode_attention(context, profiled_command, layer,
                                                                        (uint32_t)position));
                    PROFILE_KERNEL(kernel_o_proj_s, encode_matvec_residual(
                        context, profiled_command, context->context, current, next, &layer->o_weight,
                        (uint32_t)context->query_heads * (uint32_t)context->head_dim,
                        (uint32_t)context->hidden, NO));
                    PROFILE_KERNEL(kernel_residual_rmsnorm_s, encode_residual_rmsnorm(
                        context, profiled_command, next, current, context->normalized, layer->post_attention_norm));
                    PROFILE_KERNEL(kernel_gate_up_swiglu_s, encode_gate_up(context, profiled_command, layer,
                                                                           context->normalized));
                    PROFILE_KERNEL(kernel_down_proj_s, encode_matvec_residual(
                        context, profiled_command, context->mlp, next, current, &layer->down_weight,
                        (uint32_t)context->intermediate, (uint32_t)context->hidden, YES));
                }
                PROFILE_KERNEL(kernel_rmsnorm_s, encode_rmsnorm(context, profiled_command, current, context->final_norm,
                                                                context->final_norm_weight, (uint32_t)context->hidden,
                                                                epsilon));
                PROFILE_KERNEL(kernel_lm_head_s, encode_lm_head(context, profiled_command));
#undef PROFILE_KERNEL
            } else {
                for (uint64_t index = 0; index < context->layer_count; ++index) {
                    StepLayerBuffers *layer = &context->layers[index];
                    encode_rmsnorm(context, command_buffer, current, context->normalized, layer->input_norm,
                                   (uint32_t)context->hidden, epsilon);
                    encode_qkv(context, command_buffer, layer, context->normalized, (uint32_t)position);
                    encode_qk_norm_rope(context, command_buffer, layer, cosine_buffer, sine_buffer, (uint32_t)position);
                    encode_attention(context, command_buffer, layer, (uint32_t)position);
                    encode_matvec_residual(context, command_buffer, context->context, current, next,
                                           &layer->o_weight, (uint32_t)context->query_heads * (uint32_t)context->head_dim,
                                           (uint32_t)context->hidden, NO);
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
                                           &layer->down_weight, (uint32_t)context->intermediate, (uint32_t)context->hidden, YES);
                }
                encode_rmsnorm(context, command_buffer, current, context->final_norm, context->final_norm_weight,
                               (uint32_t)context->hidden, epsilon);
                encode_lm_head(context, command_buffer);
            }
            context->timings.feed_wall_s += [NSDate timeIntervalSinceReferenceDate] - feed_started;
            if (context->profile_kernels) {
                context->timings.execute_wall_s += [NSDate timeIntervalSinceReferenceDate] - profile_started;
            } else {
                double started = [NSDate timeIntervalSinceReferenceDate];
                [command_buffer commit];
                [command_buffer waitUntilCompleted];
                context->timings.execute_wall_s += [NSDate timeIntervalSinceReferenceDate] - started;
                ok = command_buffer.status != MTLCommandBufferStatusError;
                if (!ok) {
                    set_error(command_buffer.error.localizedDescription ?: @"Metal step command buffer failed");
                }
            }
            if (ok) {
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
    [context->attention_scores release];
    [context->mlp release];
    [context->final_norm release];
    [context->logits release];
    [context->embeddings release];
    [context->argmax_partial_keys release];
    [context->argmax_partial_ids release];
    [context->chain_token_ids release];
    [context->chain_input release];
    [context->batch_input release];
    [context->batch_x_b release];
    [context->batch_normalized release];
    [context->batch_final_norm release];
    [context->batch_query release];
    [context->batch_key release];
    [context->batch_context release];
    [context->batch_attention_scores release];
    [context->batch_mlp release];
    [context->batch_logits release];
    [context->batch_argmax_partial_keys release];
    [context->batch_argmax_partial_ids release];
    [context->qkv_matvec_batch release];
    [context->matvec_residual_batch release];
    [context->gate_up_swiglu_batch release];
    [context->lm_head_batch release];
    [context->embedding_gather release];
    [context->argmax_final release];
    [context->argmax_partial release];
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
