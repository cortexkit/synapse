#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MPSGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphActivationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphArithmeticOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphExecutable.h>
#import <MetalPerformanceShadersGraph/MPSGraphMatrixMultiplicationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMemoryOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphNormalizationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorShapeOps.h>

#include "mpsgraph_runtime.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char decode_error[1024];

typedef struct Qwen3DecodeLayerParams {
    const void *input_norm;
    const void *post_attention_norm;
    const void *q_weight;
    const void *q_norm;
    const void *k_weight;
    const void *k_norm;
    const void *v_weight;
    const void *o_weight;
    const void *gate_weight;
    const void *up_weight;
    const void *down_weight;
} Qwen3DecodeLayerParams;

typedef struct Qwen3DecodeLayerTensors {
    MPSGraphTensor *input_norm;
    MPSGraphTensor *post_attention_norm;
    MPSGraphTensor *q_weight;
    MPSGraphTensor *q_norm;
    MPSGraphTensor *k_weight;
    MPSGraphTensor *k_norm;
    MPSGraphTensor *v_weight;
    MPSGraphTensor *o_weight;
    MPSGraphTensor *gate_weight;
    MPSGraphTensor *up_weight;
    MPSGraphTensor *down_weight;
    MPSGraphTensor *key_cache;
    MPSGraphTensor *value_cache;
    MPSGraphTensor *new_key;
    MPSGraphTensor *new_value;
} Qwen3DecodeLayerTensors;

typedef struct Qwen3DecodePlan {
    MPSGraph *graph;
    BOOL step;
    uint64_t bucket;
    uint64_t layer_count;
    MPSShape *input_shape;
    MPSShape *mask_shape;
    MPSShape *rope_shape;
    MPSShape *selector_shape;
    MPSShape *hidden_vector_shape;
    MPSShape *head_vector_shape;
    MPSShape *q_weight_shape;
    MPSShape *kv_weight_shape;
    MPSShape *o_weight_shape;
    MPSShape *mlp_weight_shape;
    MPSShape *down_weight_shape;
    MPSShape *cache_shape;
    MPSGraphTensor *input;
    MPSGraphTensor *mask;
    MPSGraphTensor *rope_cos;
    MPSGraphTensor *rope_sin;
    MPSGraphTensor *selector;
    MPSGraphTensor *final_norm;
    MPSGraphTensor *lm_head;
    MPSGraphTensor *logits;
    Qwen3DecodeLayerTensors *layers;
    NSArray<MPSGraphTensor *> *targets;
    MPSGraphExecutable *executable;
    NSArray<MPSGraphTensor *> *executable_feed_tensors;
} Qwen3DecodePlan;

typedef struct Qwen3DecodeStageTimings {
    double graph_prepare_wall_s;
    double feed_wall_s;
    double execute_wall_s;
    double logits_readback_wall_s;
    double kv_update_wall_s;
    uint64_t prefill_calls;
    uint64_t step_calls;
} Qwen3DecodeStageTimings;

typedef struct Qwen3DecodeContext {
    SynapseMpsRuntimeContext runtime;
    uint64_t bucket;
    uint64_t layer_count;
    uint64_t kv_heads;
    uint64_t head_dim;
    NSMutableArray<id<MTLBuffer>> *key_caches;
    NSMutableArray<id<MTLBuffer>> *value_caches;
    NSMutableArray<id<MTLBuffer>> *key_updates;
    NSMutableArray<id<MTLBuffer>> *value_updates;
    BOOL legacy_cpu_readback;
    BOOL optimization_level_one;
    Qwen3DecodeStageTimings timings;
} Qwen3DecodeContext;

static double wall_time(void) {
    return [NSDate timeIntervalSinceReferenceDate];
}

static void set_error(NSString *message) {
    snprintf(decode_error, sizeof(decode_error), "%s", message.UTF8String ?: "unknown Qwen3 decode error");
}

const char *synapse_qwen3_decode_last_error(void) {
    return decode_error;
}

static MPSGraphTensor *placeholder(MPSGraph *graph, MPSShape *shape, MPSDataType type, NSString *name) {
    return [[graph placeholderWithShape:shape dataType:type name:name] retain];
}

static MPSGraphTensor *cast_tensor(MPSGraph *graph, MPSGraphTensor *tensor, MPSDataType type) {
    return tensor.dataType == type ? tensor : [graph castTensor:tensor toType:type name:nil];
}

static MPSGraphTensor *linear(MPSGraph *graph, MPSGraphTensor *input, MPSGraphTensor *weight) {
    MPSGraphTensor *transposed = [graph transposeTensor:weight dimension:0 withDimension:1 name:nil];
    return [graph matrixMultiplicationWithPrimaryTensor:input secondaryTensor:transposed name:nil];
}

static MPSGraphTensor *rms_norm(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *weight,
    NSInteger axis,
    MPSShape *reduced_shape,
    float epsilon
) {
    MPSGraphTensor *input32 = cast_tensor(graph, input, MPSDataTypeFloat32);
    MPSGraphTensor *weight32 = cast_tensor(graph, weight, MPSDataTypeFloat32);
    MPSGraphTensor *square = [graph multiplicationWithPrimaryTensor:input32 secondaryTensor:input32 name:nil];
    MPSGraphTensor *mean = [graph meanOfTensor:square axes:@[ @(axis) ] name:nil];
    mean = [graph reshapeTensor:mean withShape:reduced_shape name:nil];
    MPSGraphTensor *eps = [graph constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
    MPSGraphTensor *denominator = [graph squareRootWithTensor:[graph additionWithPrimaryTensor:mean secondaryTensor:eps name:nil] name:nil];
    MPSGraphTensor *normalized = [graph divisionWithPrimaryTensor:input32 secondaryTensor:denominator name:nil];
    return cast_tensor(
        graph,
        [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:weight32 name:nil],
        MPSDataTypeFloat16
    );
}

static MPSGraphTensor *rope(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *cosine,
    MPSGraphTensor *sine,
    uint64_t head_dim
) {
    NSUInteger half = (NSUInteger)(head_dim / 2);
    MPSGraphTensor *first = [graph sliceTensor:input dimension:3 start:0 length:half name:nil];
    MPSGraphTensor *second = [graph sliceTensor:input dimension:3 start:half length:half name:nil];
    MPSGraphTensor *rotated = [graph concatTensors:@[ [graph negativeWithTensor:second name:nil], first ] dimension:3 name:nil];
    return [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:input secondaryTensor:cosine name:nil]
                               secondaryTensor:[graph multiplicationWithPrimaryTensor:rotated secondaryTensor:sine name:nil]
                                          name:nil];
}

static MPSGraphTensor *repeat_kv(
    MPSGraph *graph,
    MPSGraphTensor *input,
    uint64_t kv_heads,
    uint64_t groups,
    uint64_t sequence,
    uint64_t head_dim
) {
    MPSGraphTensor *grouped = [graph reshapeTensor:input
                                         withShape:@[ @1, @(kv_heads), @1, @(sequence), @(head_dim) ]
                                              name:nil];
    MPSGraphTensor *broadcast = [graph broadcastTensor:grouped
                                               toShape:@[ @1, @(kv_heads), @(groups), @(sequence), @(head_dim) ]
                                                  name:nil];
    return [graph reshapeTensor:broadcast
                      withShape:@[ @1, @(kv_heads * groups), @(sequence), @(head_dim) ]
                           name:nil];
}

static void release_layer(Qwen3DecodeLayerTensors *layer) {
    [layer->new_value release];
    [layer->new_key release];
    [layer->value_cache release];
    [layer->key_cache release];
    [layer->down_weight release];
    [layer->up_weight release];
    [layer->gate_weight release];
    [layer->o_weight release];
    [layer->v_weight release];
    [layer->k_norm release];
    [layer->k_weight release];
    [layer->q_norm release];
    [layer->q_weight release];
    [layer->post_attention_norm release];
    [layer->input_norm release];
}

static void free_plan(Qwen3DecodePlan *plan) {
    if (plan == NULL) return;
    [plan->executable_feed_tensors release];
    [plan->executable release];
    [plan->targets release];
    if (plan->layers != NULL) {
        for (uint64_t i = 0; i < plan->layer_count; ++i) release_layer(&plan->layers[i]);
        free(plan->layers);
    }
    [plan->logits release];
    [plan->lm_head release];
    [plan->final_norm release];
    [plan->selector release];
    [plan->rope_sin release];
    [plan->rope_cos release];
    [plan->mask release];
    [plan->input release];
    [plan->cache_shape release];
    [plan->down_weight_shape release];
    [plan->mlp_weight_shape release];
    [plan->o_weight_shape release];
    [plan->kv_weight_shape release];
    [plan->q_weight_shape release];
    [plan->head_vector_shape release];
    [plan->hidden_vector_shape release];
    [plan->selector_shape release];
    [plan->rope_shape release];
    [plan->mask_shape release];
    [plan->input_shape release];
    [plan->graph release];
    free(plan);
}

static Qwen3DecodePlan *new_plan(
    BOOL step,
    uint64_t bucket,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    uint64_t vocab,
    float epsilon
) {
    Qwen3DecodePlan *plan = calloc(1, sizeof(Qwen3DecodePlan));
    if (plan == NULL) {
        set_error(@"failed to allocate Qwen3 decode plan");
        return NULL;
    }
    uint64_t sequence = step ? 1 : bucket;
    uint64_t keys = step ? bucket + 1 : bucket;
    uint64_t q_width = query_heads * head_dim;
    uint64_t kv_width = kv_heads * head_dim;
    uint64_t groups = query_heads / kv_heads;
    plan->step = step;
    plan->bucket = bucket;
    plan->layer_count = layer_count;
    plan->input_shape = [@[ @1, @(sequence), @(hidden) ] retain];
    plan->mask_shape = [@[ @1, @1, @(sequence), @(keys) ] retain];
    plan->rope_shape = [@[ @1, @1, @(sequence), @(head_dim) ] retain];
    plan->selector_shape = [@[ @(sequence), @1 ] retain];
    plan->hidden_vector_shape = [@[ @(hidden) ] retain];
    plan->head_vector_shape = [@[ @(head_dim) ] retain];
    plan->q_weight_shape = [@[ @(q_width), @(hidden) ] retain];
    plan->kv_weight_shape = [@[ @(kv_width), @(hidden) ] retain];
    plan->o_weight_shape = [@[ @(hidden), @(q_width) ] retain];
    plan->mlp_weight_shape = [@[ @(intermediate), @(hidden) ] retain];
    plan->down_weight_shape = [@[ @(hidden), @(intermediate) ] retain];
    plan->cache_shape = [@[ @1, @(kv_heads), @(bucket), @(head_dim) ] retain];
    plan->layers = calloc((size_t)layer_count, sizeof(Qwen3DecodeLayerTensors));
    plan->graph = [[MPSGraph alloc] init];
    if (plan->layers == NULL || plan->graph == nil) {
        free_plan(plan);
        set_error(@"failed to allocate Qwen3 decode graph objects");
        return NULL;
    }
    plan->graph.options = MPSGraphOptionsNone;
    plan->input = placeholder(plan->graph, plan->input_shape, MPSDataTypeFloat16, @"decode_input");
    plan->mask = placeholder(plan->graph, plan->mask_shape, MPSDataTypeFloat32, @"decode_mask");
    plan->rope_cos = placeholder(plan->graph, plan->rope_shape, MPSDataTypeFloat16, @"decode_rope_cos");
    plan->rope_sin = placeholder(plan->graph, plan->rope_shape, MPSDataTypeFloat16, @"decode_rope_sin");
    plan->selector = placeholder(plan->graph, plan->selector_shape, MPSDataTypeFloat16, @"decode_selector");
    plan->final_norm = placeholder(plan->graph, plan->hidden_vector_shape, MPSDataTypeFloat16, @"decode_final_norm");
    plan->lm_head = placeholder(plan->graph, @[ @(vocab), @(hidden) ], MPSDataTypeFloat16, @"decode_lm_head");
    MPSGraphTensor *x = [plan->graph reshapeTensor:plan->input withShape:@[ @(sequence), @(hidden) ] name:nil];
    NSMutableArray<MPSGraphTensor *> *targets = [NSMutableArray arrayWithCapacity:(NSUInteger)(1 + layer_count * 2)];

    for (uint64_t index = 0; index < layer_count; ++index) {
        Qwen3DecodeLayerTensors *layer = &plan->layers[index];
        NSString *prefix = [NSString stringWithFormat:@"decode_layer_%llu", (unsigned long long)index];
#define PH_WEIGHT(field, shape) layer->field = placeholder(plan->graph, shape, MPSDataTypeFloat16, [prefix stringByAppendingFormat:@"_%s", #field])
        PH_WEIGHT(input_norm, plan->hidden_vector_shape);
        PH_WEIGHT(post_attention_norm, plan->hidden_vector_shape);
        PH_WEIGHT(q_weight, plan->q_weight_shape);
        PH_WEIGHT(q_norm, plan->head_vector_shape);
        PH_WEIGHT(k_weight, plan->kv_weight_shape);
        PH_WEIGHT(k_norm, plan->head_vector_shape);
        PH_WEIGHT(v_weight, plan->kv_weight_shape);
        PH_WEIGHT(o_weight, plan->o_weight_shape);
        PH_WEIGHT(gate_weight, plan->mlp_weight_shape);
        PH_WEIGHT(up_weight, plan->mlp_weight_shape);
        PH_WEIGHT(down_weight, plan->down_weight_shape);
        if (step) {
            layer->key_cache = placeholder(plan->graph, plan->cache_shape, MPSDataTypeFloat16,
                                           [prefix stringByAppendingString:@"_key_cache"]);
            layer->value_cache = placeholder(plan->graph, plan->cache_shape, MPSDataTypeFloat16,
                                             [prefix stringByAppendingString:@"_value_cache"]);
        }
#undef PH_WEIGHT
        MPSGraphTensor *attention_residual = x;
        MPSGraphTensor *normalized = rms_norm(plan->graph, x, layer->input_norm, 1, @[ @(sequence), @1 ], epsilon);
        MPSGraphTensor *q = linear(plan->graph, normalized, layer->q_weight);
        MPSGraphTensor *k = linear(plan->graph, normalized, layer->k_weight);
        MPSGraphTensor *v = linear(plan->graph, normalized, layer->v_weight);
        q = [plan->graph reshapeTensor:q withShape:@[ @1, @(sequence), @(query_heads), @(head_dim) ] name:nil];
        k = [plan->graph reshapeTensor:k withShape:@[ @1, @(sequence), @(kv_heads), @(head_dim) ] name:nil];
        v = [plan->graph reshapeTensor:v withShape:@[ @1, @(sequence), @(kv_heads), @(head_dim) ] name:nil];
        q = rms_norm(plan->graph, q, layer->q_norm, 3, @[ @1, @(sequence), @(query_heads), @1 ], epsilon);
        k = rms_norm(plan->graph, k, layer->k_norm, 3, @[ @1, @(sequence), @(kv_heads), @1 ], epsilon);
        q = [plan->graph transposeTensor:q permutation:@[ @0, @2, @1, @3 ] name:nil];
        k = [plan->graph transposeTensor:k permutation:@[ @0, @2, @1, @3 ] name:nil];
        v = [plan->graph transposeTensor:v permutation:@[ @0, @2, @1, @3 ] name:nil];
        q = rope(plan->graph, q, plan->rope_cos, plan->rope_sin, head_dim);
        k = rope(plan->graph, k, plan->rope_cos, plan->rope_sin, head_dim);
        layer->new_key = [k retain];
        layer->new_value = [v retain];
        [targets addObject:k];
        [targets addObject:v];
        MPSGraphTensor *attention_k = step
            ? [plan->graph concatTensors:@[ layer->key_cache, k ] dimension:2 name:nil]
            : k;
        MPSGraphTensor *attention_v = step
            ? [plan->graph concatTensors:@[ layer->value_cache, v ] dimension:2 name:nil]
            : v;
        attention_k = repeat_kv(plan->graph, attention_k, kv_heads, groups, keys, head_dim);
        attention_v = repeat_kv(plan->graph, attention_v, kv_heads, groups, keys, head_dim);
        MPSGraphTensor *scores16 = [plan->graph matrixMultiplicationWithPrimaryTensor:q
                                                                    secondaryTensor:[plan->graph transposeTensor:attention_k dimension:2 withDimension:3 name:nil]
                                                                               name:nil];
        MPSGraphTensor *scores = cast_tensor(plan->graph, scores16, MPSDataTypeFloat32);
        MPSGraphTensor *scale = [plan->graph constantWithScalar:1.0 / sqrt((double)head_dim) dataType:MPSDataTypeFloat32];
        scores = [plan->graph multiplicationWithPrimaryTensor:scores secondaryTensor:scale name:nil];
        scores = [plan->graph additionWithPrimaryTensor:scores secondaryTensor:plan->mask name:nil];
        scores = [plan->graph softMaxWithTensor:scores axis:3 name:nil];
        MPSGraphTensor *probs16 = cast_tensor(plan->graph, scores, MPSDataTypeFloat16);
        MPSGraphTensor *context = [plan->graph matrixMultiplicationWithPrimaryTensor:probs16
                                                                     secondaryTensor:attention_v
                                                                                name:nil];
        context = cast_tensor(plan->graph, context, MPSDataTypeFloat16);
        context = [plan->graph transposeTensor:context permutation:@[ @0, @2, @1, @3 ] name:nil];
        context = [plan->graph reshapeTensor:context withShape:@[ @(sequence), @(q_width) ] name:nil];
        x = [plan->graph additionWithPrimaryTensor:attention_residual
                                   secondaryTensor:linear(plan->graph, context, layer->o_weight)
                                              name:nil];
        MPSGraphTensor *mlp_residual = x;
        normalized = rms_norm(plan->graph, x, layer->post_attention_norm, 1, @[ @(sequence), @1 ], epsilon);
        MPSGraphTensor *gate = linear(plan->graph, normalized, layer->gate_weight);
        gate = [plan->graph multiplicationWithPrimaryTensor:gate
                                            secondaryTensor:[plan->graph sigmoidWithTensor:gate name:nil]
                                                       name:nil];
        MPSGraphTensor *up = linear(plan->graph, normalized, layer->up_weight);
        MPSGraphTensor *gated = [plan->graph multiplicationWithPrimaryTensor:gate secondaryTensor:up name:nil];
        x = [plan->graph additionWithPrimaryTensor:mlp_residual
                                   secondaryTensor:linear(plan->graph, gated, layer->down_weight)
                                              name:nil];
    }
    x = rms_norm(plan->graph, x, plan->final_norm, 1, @[ @(sequence), @1 ], epsilon);
    MPSGraphTensor *selected = [plan->graph matrixMultiplicationWithPrimaryTensor:[plan->graph transposeTensor:x dimension:0 withDimension:1 name:nil]
                                                                  secondaryTensor:plan->selector
                                                                             name:nil];
    selected = [plan->graph transposeTensor:selected dimension:0 withDimension:1 name:nil];
    MPSGraphTensor *logits16 = linear(plan->graph, selected, plan->lm_head);
    plan->logits = [[plan->graph castTensor:logits16 toType:MPSDataTypeFloat32 name:@"decode_logits_fp32"] retain];
    [targets insertObject:plan->logits atIndex:0];
    plan->targets = [targets copy];
    if (plan->input == nil || plan->logits == nil || plan->targets.count != 1 + layer_count * 2) {
        free_plan(plan);
        set_error(@"failed to construct Qwen3 decode graph");
        return NULL;
    }
    return plan;
}

static BOOL add_shaped_type(NSMutableDictionary *feeds, MPSGraphTensor *tensor, MPSShape *shape, MPSDataType type) {
    MPSGraphShapedType *shaped = [[MPSGraphShapedType alloc] initWithShape:shape dataType:type];
    if (shaped == nil) return NO;
    [feeds setObject:shaped forKey:tensor];
    [shaped release];
    return YES;
}

static NSDictionary<MPSGraphTensor *, MPSGraphShapedType *> *shaped_feeds(Qwen3DecodePlan *plan, uint64_t vocab) {
    NSMutableDictionary *feeds = [[NSMutableDictionary alloc] initWithCapacity:(NSUInteger)(7 + plan->layer_count * 13)];
    BOOL ok = add_shaped_type(feeds, plan->input, plan->input_shape, MPSDataTypeFloat16) &&
        add_shaped_type(feeds, plan->mask, plan->mask_shape, MPSDataTypeFloat32) &&
        add_shaped_type(feeds, plan->rope_cos, plan->rope_shape, MPSDataTypeFloat16) &&
        add_shaped_type(feeds, plan->rope_sin, plan->rope_shape, MPSDataTypeFloat16) &&
        add_shaped_type(feeds, plan->selector, plan->selector_shape, MPSDataTypeFloat16) &&
        add_shaped_type(feeds, plan->final_norm, plan->hidden_vector_shape, MPSDataTypeFloat16) &&
        add_shaped_type(feeds, plan->lm_head, @[ @(vocab), plan->hidden_vector_shape[0] ], MPSDataTypeFloat16);
    for (uint64_t i = 0; ok && i < plan->layer_count; ++i) {
        Qwen3DecodeLayerTensors *layer = &plan->layers[i];
        ok = add_shaped_type(feeds, layer->input_norm, plan->hidden_vector_shape, MPSDataTypeFloat16) &&
            add_shaped_type(feeds, layer->post_attention_norm, plan->hidden_vector_shape, MPSDataTypeFloat16) &&
            add_shaped_type(feeds, layer->q_weight, plan->q_weight_shape, MPSDataTypeFloat16) &&
            add_shaped_type(feeds, layer->q_norm, plan->head_vector_shape, MPSDataTypeFloat16) &&
            add_shaped_type(feeds, layer->k_weight, plan->kv_weight_shape, MPSDataTypeFloat16) &&
            add_shaped_type(feeds, layer->k_norm, plan->head_vector_shape, MPSDataTypeFloat16) &&
            add_shaped_type(feeds, layer->v_weight, plan->kv_weight_shape, MPSDataTypeFloat16) &&
            add_shaped_type(feeds, layer->o_weight, plan->o_weight_shape, MPSDataTypeFloat16) &&
            add_shaped_type(feeds, layer->gate_weight, plan->mlp_weight_shape, MPSDataTypeFloat16) &&
            add_shaped_type(feeds, layer->up_weight, plan->mlp_weight_shape, MPSDataTypeFloat16) &&
            add_shaped_type(feeds, layer->down_weight, plan->down_weight_shape, MPSDataTypeFloat16);
        if (ok && plan->step) {
            ok = add_shaped_type(feeds, layer->key_cache, plan->cache_shape, MPSDataTypeFloat16) &&
                add_shaped_type(feeds, layer->value_cache, plan->cache_shape, MPSDataTypeFloat16);
        }
    }
    if (!ok) {
        [feeds release];
        return nil;
    }
    return [feeds autorelease];
}

static MPSGraphExecutable *prepare_executable(Qwen3DecodePlan *plan, id<MTLDevice> device, uint64_t vocab,
                                               BOOL optimization_level_one, const char *package_path) {
    MPSGraphCompilationDescriptor *descriptor = [[MPSGraphCompilationDescriptor alloc] init];
    descriptor.optimizationLevel = optimization_level_one
        ? MPSGraphOptimizationLevel1
        : MPSGraphOptimizationLevel0;
    descriptor.waitForCompilationCompletion = YES;
    MPSGraphExecutable *executable = nil;
    NSString *path = package_path == NULL ? nil : [NSString stringWithUTF8String:package_path];
    BOOL load = path != nil && [[NSFileManager defaultManager] fileExistsAtPath:path];
    if (load) {
        executable = [[MPSGraphExecutable alloc]
            initWithMPSGraphPackageAtURL:[NSURL fileURLWithPath:path]
            compilationDescriptor:descriptor];
    } else {
        NSDictionary *feeds = shaped_feeds(plan, vocab);
        MPSGraphDevice *graph_device = [MPSGraphDevice deviceWithMTLDevice:device];
        executable = [[plan->graph compileWithDevice:graph_device
                                               feeds:feeds
                                       targetTensors:plan->targets
                                    targetOperations:nil
                               compilationDescriptor:descriptor] retain];
        if (executable != nil) {
            NSMutableArray<MPSGraphType *> *input_types = [NSMutableArray arrayWithCapacity:executable.feedTensors.count];
            for (MPSGraphTensor *tensor in executable.feedTensors) {
                MPSGraphShapedType *type = [feeds objectForKey:tensor];
                if (type == nil) {
                    for (MPSGraphTensor *placeholder_tensor in feeds) {
                        if ([placeholder_tensor.operation.name isEqualToString:tensor.operation.name]) {
                            type = [feeds objectForKey:placeholder_tensor];
                            break;
                        }
                    }
                }
                if (type == nil) {
                    [executable release];
                    executable = nil;
                    break;
                }
                [input_types addObject:type];
            }
            if (executable != nil) {
                [executable specializeWithDevice:graph_device inputTypes:input_types compilationDescriptor:descriptor];
            }
        }
        if (executable != nil && path != nil) {
            MPSGraphExecutableSerializationDescriptor *serialization = [[MPSGraphExecutableSerializationDescriptor alloc] init];
            serialization.append = NO;
            [executable serializeToMPSGraphPackageAtURL:[NSURL fileURLWithPath:path] descriptor:serialization];
            [serialization release];
        }
    }
    [descriptor release];
    if (executable != nil) {
        executable.options = MPSGraphOptionsNone;
        NSArray<MPSGraphTensor *> *inputs = executable.feedTensors;
        plan->executable_feed_tensors = [(inputs.count > 0 ? inputs : plan->graph.placeholderTensors) retain];
    }
    return executable;
}

static BOOL add_feed(NSMutableDictionary *feeds, MPSGraphTensor *tensor, MPSShape *shape, id<MTLBuffer> buffer, MPSDataType type) {
    return synapse_mps_add_feed(feeds, tensor, shape, buffer, type);
}

static BOOL add_weight(Qwen3DecodeContext *context, NSMutableDictionary *feeds, MPSGraphTensor *tensor,
                       MPSShape *shape, const void *values, NSUInteger elements) {
    id<MTLBuffer> buffer = synapse_mps_cached_static_buffer(&context->runtime, values, elements * sizeof(uint16_t));
    return add_feed(feeds, tensor, shape, buffer, MPSDataTypeFloat16);
}

static BOOL add_layer_weights(
    Qwen3DecodeContext *context,
    Qwen3DecodePlan *plan,
    NSMutableDictionary *feeds,
    const Qwen3DecodeLayerParams *params,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate
) {
    NSUInteger q_count = (NSUInteger)(query_heads * head_dim * hidden);
    NSUInteger kv_count = (NSUInteger)(kv_heads * head_dim * hidden);
    NSUInteger o_count = (NSUInteger)(hidden * query_heads * head_dim);
    NSUInteger mlp_count = (NSUInteger)(intermediate * hidden);
    for (uint64_t i = 0; i < plan->layer_count; ++i) {
        Qwen3DecodeLayerTensors *t = &plan->layers[i];
        const Qwen3DecodeLayerParams *p = &params[i];
        if (!add_weight(context, feeds, t->input_norm, plan->hidden_vector_shape, p->input_norm, hidden) ||
            !add_weight(context, feeds, t->post_attention_norm, plan->hidden_vector_shape, p->post_attention_norm, hidden) ||
            !add_weight(context, feeds, t->q_weight, plan->q_weight_shape, p->q_weight, q_count) ||
            !add_weight(context, feeds, t->q_norm, plan->head_vector_shape, p->q_norm, head_dim) ||
            !add_weight(context, feeds, t->k_weight, plan->kv_weight_shape, p->k_weight, kv_count) ||
            !add_weight(context, feeds, t->k_norm, plan->head_vector_shape, p->k_norm, head_dim) ||
            !add_weight(context, feeds, t->v_weight, plan->kv_weight_shape, p->v_weight, kv_count) ||
            !add_weight(context, feeds, t->o_weight, plan->o_weight_shape, p->o_weight, o_count) ||
            !add_weight(context, feeds, t->gate_weight, plan->mlp_weight_shape, p->gate_weight, mlp_count) ||
            !add_weight(context, feeds, t->up_weight, plan->mlp_weight_shape, p->up_weight, mlp_count) ||
            !add_weight(context, feeds, t->down_weight, plan->down_weight_shape, p->down_weight, mlp_count)) return NO;
    }
    return YES;
}



static NSArray<MPSGraphTensorData *> *run_plan(Qwen3DecodeContext *context, Qwen3DecodePlan *plan, NSDictionary *feeds) {
    NSArray<MPSGraphTensorData *> *inputs = synapse_mps_executable_inputs(plan->executable_feed_tensors, feeds);
    if (inputs == nil) return nil;
    return [plan->executable runWithMTLCommandQueue:context->runtime.queue
                                        inputsArray:inputs
                                      resultsArray:nil
                               executionDescriptor:nil];
}

static BOOL export_prefill_cache(Qwen3DecodeContext *context, NSArray<MPSGraphTensorData *> *results) {
    id<MTLCommandBuffer> command_buffer = [context->runtime.queue commandBuffer];
    if (command_buffer == nil) return NO;
    for (uint64_t i = 0; i < context->layer_count; ++i) {
        [[[results objectAtIndex:(NSUInteger)(1 + i * 2)] mpsndarray]
            exportDataWithCommandBuffer:command_buffer
                                toBuffer:[context->key_caches objectAtIndex:(NSUInteger)i]
                     destinationDataType:MPSDataTypeFloat16
                                  offset:0
                              rowStrides:NULL];
        [[[results objectAtIndex:(NSUInteger)(2 + i * 2)] mpsndarray]
            exportDataWithCommandBuffer:command_buffer
                                toBuffer:[context->value_caches objectAtIndex:(NSUInteger)i]
                     destinationDataType:MPSDataTypeFloat16
                                  offset:0
                              rowStrides:NULL];
    }
    [command_buffer commit];
    return YES;
}

static BOOL export_step_cache(Qwen3DecodeContext *context, NSArray<MPSGraphTensorData *> *results,
                              uint64_t position) {
    id<MTLCommandBuffer> command_buffer = [context->runtime.queue commandBuffer];
    if (command_buffer == nil) return NO;
    for (uint64_t i = 0; i < context->layer_count; ++i) {
        [[[results objectAtIndex:(NSUInteger)(1 + i * 2)] mpsndarray]
            exportDataWithCommandBuffer:command_buffer
                                toBuffer:[context->key_updates objectAtIndex:(NSUInteger)i]
                     destinationDataType:MPSDataTypeFloat16
                                  offset:0
                              rowStrides:NULL];
        [[[results objectAtIndex:(NSUInteger)(2 + i * 2)] mpsndarray]
            exportDataWithCommandBuffer:command_buffer
                                toBuffer:[context->value_updates objectAtIndex:(NSUInteger)i]
                     destinationDataType:MPSDataTypeFloat16
                                  offset:0
                              rowStrides:NULL];
    }

    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    if (blit == nil) return NO;
    NSUInteger head_bytes = (NSUInteger)(context->head_dim * sizeof(uint16_t));
    for (uint64_t i = 0; i < context->layer_count; ++i) {
        id<MTLBuffer> key_source = [context->key_updates objectAtIndex:(NSUInteger)i];
        id<MTLBuffer> value_source = [context->value_updates objectAtIndex:(NSUInteger)i];
        id<MTLBuffer> key_destination = [context->key_caches objectAtIndex:(NSUInteger)i];
        id<MTLBuffer> value_destination = [context->value_caches objectAtIndex:(NSUInteger)i];
        for (uint64_t head = 0; head < context->kv_heads; ++head) {
            NSUInteger source_offset = (NSUInteger)head * head_bytes;
            NSUInteger destination_offset = (NSUInteger)(head * context->bucket + position) * head_bytes;
            [blit copyFromBuffer:key_source sourceOffset:source_offset
                      toBuffer:key_destination destinationOffset:destination_offset size:head_bytes];
            [blit copyFromBuffer:value_source sourceOffset:source_offset
                      toBuffer:value_destination destinationOffset:destination_offset size:head_bytes];
        }
    }
    [blit endEncoding];
    [command_buffer commit];
    return YES;
}

static void free_plan_erased(void *plan) {
    free_plan((Qwen3DecodePlan *)plan);
}

void synapse_qwen3_decode_context_free(void *raw);

void *synapse_qwen3_decode_context_new(uint64_t bucket, uint64_t layer_count, uint64_t kv_heads, uint64_t head_dim) {
    @autoreleasepool {
        if (bucket == 0 || layer_count == 0 || kv_heads == 0 || head_dim == 0) {
            set_error(@"invalid Qwen3 decode cache dimensions");
            return NULL;
        }
        Qwen3DecodeContext *context = calloc(1, sizeof(Qwen3DecodeContext));
        if (context == NULL || !synapse_mps_runtime_init(&context->runtime)) {
            free(context);
            set_error(@"no Metal device or command queue for Qwen3 decode");
            return NULL;
        }
        context->bucket = bucket;
        context->layer_count = layer_count;
        context->kv_heads = kv_heads;
        context->head_dim = head_dim;
        const char *legacy = getenv("SYNAPSE_QWEN3_DECODE_LEGACY_READBACK");
        context->legacy_cpu_readback = legacy != NULL && strcmp(legacy, "1") == 0;
        const char *optimization_level = getenv("SYNAPSE_QWEN3_DECODE_OPT_LEVEL");
        context->optimization_level_one = optimization_level == NULL || strcmp(optimization_level, "0") != 0;
        context->key_caches = [[NSMutableArray alloc] initWithCapacity:(NSUInteger)layer_count];
        context->value_caches = [[NSMutableArray alloc] initWithCapacity:(NSUInteger)layer_count];
        context->key_updates = [[NSMutableArray alloc] initWithCapacity:(NSUInteger)layer_count];
        context->value_updates = [[NSMutableArray alloc] initWithCapacity:(NSUInteger)layer_count];
        NSUInteger bytes = (NSUInteger)(kv_heads * bucket * head_dim * sizeof(uint16_t));
        for (uint64_t i = 0; i < layer_count; ++i) {
            MTLResourceOptions options = context->legacy_cpu_readback
                ? MTLResourceStorageModeShared
                : MTLResourceStorageModePrivate;
            id<MTLBuffer> key = [context->runtime.device newBufferWithLength:bytes options:options];
            id<MTLBuffer> value = [context->runtime.device newBufferWithLength:bytes options:options];
            NSUInteger update_bytes = (NSUInteger)(kv_heads * head_dim * sizeof(uint16_t));
            id<MTLBuffer> key_update = [context->runtime.device newBufferWithLength:update_bytes
                                                                           options:MTLResourceStorageModePrivate];
            id<MTLBuffer> value_update = [context->runtime.device newBufferWithLength:update_bytes
                                                                             options:MTLResourceStorageModePrivate];
            if (key == nil || value == nil || key_update == nil || value_update == nil) {
                [key release];
                [value release];
                [key_update release];
                [value_update release];
                synapse_qwen3_decode_context_free(context);
                set_error(@"failed to allocate Qwen3 decode KV buffers");
                return NULL;
            }
            if (context->legacy_cpu_readback) {
                memset(key.contents, 0, bytes);
                memset(value.contents, 0, bytes);
            }
            [context->key_caches addObject:key];
            [context->value_caches addObject:value];
            [context->key_updates addObject:key_update];
            [context->value_updates addObject:value_update];
            [key release];
            [value release];
            [key_update release];
            [value_update release];
        }
        return context;
    }
}

void synapse_qwen3_decode_context_free(void *raw) {
    if (raw == NULL) return;
    Qwen3DecodeContext *context = raw;
    [context->value_updates release];
    [context->key_updates release];
    [context->value_caches release];
    [context->key_caches release];
    synapse_mps_runtime_release(&context->runtime, free_plan_erased);
    free(context);
}

static Qwen3DecodePlan *cached_decode_plan(
    Qwen3DecodeContext *context,
    BOOL step,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    uint64_t vocab,
    float epsilon,
    const char *package_path
) {
    NSString *key = [NSString stringWithFormat:@"decode:%d:%d:%llu:%llu:%llu:%llu:%llu:%llu:%llu:%.9g",
        step, context->optimization_level_one, (unsigned long long)context->bucket, (unsigned long long)hidden,
        (unsigned long long)query_heads, (unsigned long long)kv_heads,
        (unsigned long long)head_dim, (unsigned long long)intermediate,
        (unsigned long long)vocab, (double)epsilon];
    Qwen3DecodePlan *plan = synapse_mps_cached_plan(&context->runtime, key);
    if (plan == NULL) {
        double prepare_started = wall_time();
        plan = new_plan(step, context->bucket, hidden, query_heads, kv_heads, head_dim,
                        intermediate, layer_count, vocab, epsilon);
        if (plan == NULL) return NULL;
        plan->graph.options = context->legacy_cpu_readback
            ? MPSGraphOptionsSynchronizeResults
            : MPSGraphOptionsNone;
        plan->executable = prepare_executable(plan, context->runtime.device, vocab,
                                              context->optimization_level_one, package_path);
        if (plan->executable != nil) {
            plan->executable.options = context->legacy_cpu_readback
                ? MPSGraphOptionsSynchronizeResults
                : MPSGraphOptionsNone;
        }
        if (plan->executable == nil) {
            free_plan(plan);
            set_error(@"failed to prepare Qwen3 decode executable");
            return NULL;
        }
        synapse_mps_cache_plan(&context->runtime, key, plan);
        context->timings.graph_prepare_wall_s += wall_time() - prepare_started;
    }
    return plan;
}

int32_t synapse_qwen3_decode_prepare(
    void *raw,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    uint64_t vocab,
    float epsilon,
    const char *prefill_package_path,
    const char *step_package_path
) {
    @autoreleasepool {
        @try {
            Qwen3DecodeContext *context = raw;
            if (context == NULL || layer_count != context->layer_count || kv_heads != context->kv_heads ||
                head_dim != context->head_dim || query_heads % kv_heads != 0 || head_dim % 2 != 0) {
                set_error(@"invalid Qwen3 decode preparation arguments");
                return -1;
            }
            Qwen3DecodePlan *prefill = cached_decode_plan(context, NO, hidden, query_heads, kv_heads, head_dim,
                                                          intermediate, layer_count, vocab, epsilon, prefill_package_path);
            Qwen3DecodePlan *step = cached_decode_plan(context, YES, hidden, query_heads, kv_heads, head_dim,
                                                       intermediate, layer_count, vocab, epsilon, step_package_path);
            return prefill != NULL && step != NULL ? 0 : -2;
        } @catch (NSException *exception) {
            set_error(exception.reason);
            return -100;
        }
    }
}

int32_t synapse_qwen3_decode_prefill(
    void *raw,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    uint64_t vocab,
    float epsilon,
    const char *package_path,
    const void *input,
    const float *mask,
    const void *rope_cos,
    const void *rope_sin,
    const void *selector,
    const Qwen3DecodeLayerParams *params,
    const void *final_norm,
    const void *lm_head,
    float *logits
) {
    @autoreleasepool {
        @try {
            decode_error[0] = '\0';
            Qwen3DecodeContext *context = raw;
            if (context == NULL || input == NULL || mask == NULL || rope_cos == NULL || rope_sin == NULL ||
                selector == NULL || params == NULL || final_norm == NULL || lm_head == NULL || logits == NULL ||
                layer_count != context->layer_count || kv_heads != context->kv_heads || head_dim != context->head_dim ||
                query_heads % kv_heads != 0 || head_dim % 2 != 0) {
                set_error(@"invalid Qwen3 decode prefill arguments");
                return -1;
            }
            Qwen3DecodePlan *plan = cached_decode_plan(context, NO, hidden, query_heads, kv_heads, head_dim,
                                                        intermediate, layer_count, vocab, epsilon, package_path);
            if (plan == NULL) return -2;
            context->timings.prefill_calls += 1;
            NSUInteger sequence = (NSUInteger)context->bucket;
            double feed_started = wall_time();
            id<MTLBuffer> input_buffer = [context->runtime.device newBufferWithBytes:input length:sequence * hidden * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> mask_buffer = [context->runtime.device newBufferWithBytes:mask length:sequence * sequence * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> cos_buffer = [context->runtime.device newBufferWithBytes:rope_cos length:sequence * head_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> sin_buffer = [context->runtime.device newBufferWithBytes:rope_sin length:sequence * head_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> selector_buffer = [context->runtime.device newBufferWithBytes:selector length:sequence * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            NSMutableDictionary *feeds = [[NSMutableDictionary alloc] initWithCapacity:(NSUInteger)(7 + layer_count * 11)];
            BOOL ok = add_feed(feeds, plan->input, plan->input_shape, input_buffer, MPSDataTypeFloat16) &&
                add_feed(feeds, plan->mask, plan->mask_shape, mask_buffer, MPSDataTypeFloat32) &&
                add_feed(feeds, plan->rope_cos, plan->rope_shape, cos_buffer, MPSDataTypeFloat16) &&
                add_feed(feeds, plan->rope_sin, plan->rope_shape, sin_buffer, MPSDataTypeFloat16) &&
                add_feed(feeds, plan->selector, plan->selector_shape, selector_buffer, MPSDataTypeFloat16) &&
                add_weight(context, feeds, plan->final_norm, plan->hidden_vector_shape, final_norm, hidden) &&
                add_weight(context, feeds, plan->lm_head, @[ @(vocab), @(hidden) ], lm_head, vocab * hidden) &&
                add_layer_weights(context, plan, feeds, params, hidden, query_heads, kv_heads, head_dim, intermediate);
            context->timings.feed_wall_s += wall_time() - feed_started;
            double execute_started = wall_time();
            NSArray<MPSGraphTensorData *> *results = ok ? run_plan(context, plan, feeds) : nil;
            context->timings.execute_wall_s += wall_time() - execute_started;
            if (results.count != plan->targets.count) {
                set_error(@"Qwen3 prefill executable returned incomplete outputs");
                ok = NO;
            }
            if (ok) {
                double logits_started = wall_time();
                [[[results objectAtIndex:0] mpsndarray] readBytes:logits strideBytes:NULL];
                context->timings.logits_readback_wall_s += wall_time() - logits_started;
                double kv_started = wall_time();
                if (context->legacy_cpu_readback) {
                    for (uint64_t i = 0; i < layer_count; ++i) {
                        MPSNDArray *key = [[results objectAtIndex:(NSUInteger)(1 + i * 2)] mpsndarray];
                        MPSNDArray *value = [[results objectAtIndex:(NSUInteger)(2 + i * 2)] mpsndarray];
                        [key readBytes:[context->key_caches[(NSUInteger)i] contents] strideBytes:NULL];
                        [value readBytes:[context->value_caches[(NSUInteger)i] contents] strideBytes:NULL];
                    }
                } else if (!export_prefill_cache(context, results)) {
                    set_error(@"failed to encode Qwen3 device-resident prefill cache export");
                    ok = NO;
                }
                context->timings.kv_update_wall_s += wall_time() - kv_started;
            }
            [feeds release];
            [selector_buffer release];
            [sin_buffer release];
            [cos_buffer release];
            [mask_buffer release];
            [input_buffer release];
            return ok ? 0 : -3;
        } @catch (NSException *exception) {
            set_error(exception.reason);
            return -100;
        }
    }
}

int32_t synapse_qwen3_decode_step(
    void *raw,
    uint64_t position,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    uint64_t vocab,
    float epsilon,
    const char *package_path,
    const void *input,
    const float *mask,
    const void *rope_cos,
    const void *rope_sin,
    const Qwen3DecodeLayerParams *params,
    const void *final_norm,
    const void *lm_head,
    float *logits
) {
    @autoreleasepool {
        @try {
            decode_error[0] = '\0';
            Qwen3DecodeContext *context = raw;
            if (context == NULL || position >= context->bucket || input == NULL || mask == NULL ||
                rope_cos == NULL || rope_sin == NULL || params == NULL || final_norm == NULL || lm_head == NULL ||
                logits == NULL || layer_count != context->layer_count || kv_heads != context->kv_heads ||
                head_dim != context->head_dim || query_heads % kv_heads != 0 || head_dim % 2 != 0) {
                set_error(@"invalid Qwen3 decode step arguments");
                return -1;
            }
            Qwen3DecodePlan *plan = cached_decode_plan(context, YES, hidden, query_heads, kv_heads, head_dim,
                                                        intermediate, layer_count, vocab, epsilon, package_path);
            if (plan == NULL) return -2;
            context->timings.step_calls += 1;
            NSUInteger keys = (NSUInteger)(context->bucket + 1);
            double feed_started = wall_time();
            id<MTLBuffer> input_buffer = [context->runtime.device newBufferWithBytes:input length:hidden * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> mask_buffer = [context->runtime.device newBufferWithBytes:mask length:keys * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> cos_buffer = [context->runtime.device newBufferWithBytes:rope_cos length:head_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> sin_buffer = [context->runtime.device newBufferWithBytes:rope_sin length:head_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            uint16_t selector_value = 0x3c00;
            id<MTLBuffer> selector_buffer = [context->runtime.device newBufferWithBytes:&selector_value length:sizeof(uint16_t) options:MTLResourceStorageModeShared];
            NSMutableDictionary *feeds = [[NSMutableDictionary alloc] initWithCapacity:(NSUInteger)(7 + layer_count * 13)];
            BOOL ok = add_feed(feeds, plan->input, plan->input_shape, input_buffer, MPSDataTypeFloat16) &&
                add_feed(feeds, plan->mask, plan->mask_shape, mask_buffer, MPSDataTypeFloat32) &&
                add_feed(feeds, plan->rope_cos, plan->rope_shape, cos_buffer, MPSDataTypeFloat16) &&
                add_feed(feeds, plan->rope_sin, plan->rope_shape, sin_buffer, MPSDataTypeFloat16) &&
                add_feed(feeds, plan->selector, plan->selector_shape, selector_buffer, MPSDataTypeFloat16) &&
                add_weight(context, feeds, plan->final_norm, plan->hidden_vector_shape, final_norm, hidden) &&
                add_weight(context, feeds, plan->lm_head, @[ @(vocab), @(hidden) ], lm_head, vocab * hidden) &&
                add_layer_weights(context, plan, feeds, params, hidden, query_heads, kv_heads, head_dim, intermediate);
            for (uint64_t i = 0; ok && i < layer_count; ++i) {
                ok = add_feed(feeds, plan->layers[i].key_cache, plan->cache_shape, context->key_caches[(NSUInteger)i], MPSDataTypeFloat16) &&
                    add_feed(feeds, plan->layers[i].value_cache, plan->cache_shape, context->value_caches[(NSUInteger)i], MPSDataTypeFloat16);
            }
            context->timings.feed_wall_s += wall_time() - feed_started;
            double execute_started = wall_time();
            NSArray<MPSGraphTensorData *> *results = ok ? run_plan(context, plan, feeds) : nil;
            context->timings.execute_wall_s += wall_time() - execute_started;
            if (results.count != plan->targets.count) {
                set_error(@"Qwen3 decode executable returned incomplete outputs");
                ok = NO;
            }
            if (ok) {
                double logits_started = wall_time();
                [[[results objectAtIndex:0] mpsndarray] readBytes:logits strideBytes:NULL];
                context->timings.logits_readback_wall_s += wall_time() - logits_started;
                double kv_started = wall_time();
                if (context->legacy_cpu_readback) {
                    NSUInteger head_bytes = (NSUInteger)(head_dim * sizeof(uint16_t));
                    NSUInteger current_elements = (NSUInteger)(kv_heads * head_dim);
                    uint16_t *temporary = malloc(current_elements * sizeof(uint16_t));
                    if (temporary == NULL) {
                        set_error(@"failed to allocate Qwen3 cache update staging");
                        ok = NO;
                    } else {
                        for (uint64_t i = 0; i < layer_count; ++i) {
                            [[[results objectAtIndex:(NSUInteger)(1 + i * 2)] mpsndarray]
                                readBytes:temporary strideBytes:NULL];
                            uint8_t *key_destination = [context->key_caches[(NSUInteger)i] contents];
                            for (uint64_t head = 0; head < kv_heads; ++head) {
                                memcpy(key_destination + (head * context->bucket + position) * head_bytes,
                                       temporary + head * head_dim, head_bytes);
                            }
                            [[[results objectAtIndex:(NSUInteger)(2 + i * 2)] mpsndarray]
                                readBytes:temporary strideBytes:NULL];
                            uint8_t *value_destination = [context->value_caches[(NSUInteger)i] contents];
                            for (uint64_t head = 0; head < kv_heads; ++head) {
                                memcpy(value_destination + (head * context->bucket + position) * head_bytes,
                                       temporary + head * head_dim, head_bytes);
                            }
                        }
                        free(temporary);
                    }
                } else if (!export_step_cache(context, results, position)) {
                    set_error(@"failed to encode Qwen3 device-resident cache export");
                    ok = NO;
                }
                context->timings.kv_update_wall_s += wall_time() - kv_started;
            }
            [feeds release];
            [selector_buffer release];
            [sin_buffer release];
            [cos_buffer release];
            [mask_buffer release];
            [input_buffer release];
            return ok ? 0 : -3;
        } @catch (NSException *exception) {
            set_error(exception.reason);
            return -100;
        }
    }
}

void synapse_qwen3_decode_stage_timings(void *raw, Qwen3DecodeStageTimings *timings) {
    if (raw == NULL || timings == NULL) return;
    *timings = ((Qwen3DecodeContext *)raw)->timings;
}

int32_t synapse_qwen3_decode_cache_copy(void *raw, uint64_t layer, uint16_t *output, uint64_t elements) {
    @autoreleasepool {
        Qwen3DecodeContext *context = raw;
        uint64_t one_cache = context == NULL ? 0 : context->kv_heads * context->bucket * context->head_dim;
        if (context == NULL || output == NULL || layer >= context->layer_count || elements != one_cache * 2) {
            set_error(@"invalid Qwen3 decode cache inspection arguments");
            return -1;
        }
        NSUInteger one_cache_bytes = (NSUInteger)one_cache * sizeof(uint16_t);
        id<MTLBuffer> staging = [context->runtime.device newBufferWithLength:one_cache_bytes * 2
                                                                     options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command_buffer = [context->runtime.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
        if (staging == nil || command_buffer == nil || blit == nil) {
            [staging release];
            set_error(@"failed to stage Qwen3 cache inspection");
            return -2;
        }
        [blit copyFromBuffer:[context->key_caches objectAtIndex:(NSUInteger)layer]
                sourceOffset:0 toBuffer:staging destinationOffset:0 size:one_cache_bytes];
        [blit copyFromBuffer:[context->value_caches objectAtIndex:(NSUInteger)layer]
                sourceOffset:0 toBuffer:staging destinationOffset:one_cache_bytes size:one_cache_bytes];
        [blit endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            set_error(command_buffer.error.localizedDescription ?: @"Qwen3 cache inspection blit failed");
            [staging release];
            return -3;
        }
        memcpy(output, staging.contents, one_cache_bytes * 2);
        [staging release];
        return 0;
    }
}
