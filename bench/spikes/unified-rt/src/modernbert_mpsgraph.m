#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MPSGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphArithmeticOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMatrixMultiplicationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphNormalizationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorShapeOps.h>

#include "mpsgraph_executable.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static char modernbert_mps_error[1024];

typedef struct ModernBertLayerParams {
    const void *qkv_weight;
    const void *attention_output_weight;
    const void *attention_norm_weight;
    const void *mlp_input_weight;
    const void *mlp_output_weight;
    const void *mlp_norm_weight;
    int32_t attention_type;
} ModernBertLayerParams;

typedef struct ModernBertLayerTensors {
    MPSGraphTensor *qkv_weight;
    MPSGraphTensor *attention_output_weight;
    MPSGraphTensor *attention_norm_weight;
    MPSGraphTensor *mlp_input_weight;
    MPSGraphTensor *mlp_output_weight;
    MPSGraphTensor *mlp_norm_weight;
} ModernBertLayerTensors;

typedef struct ModernBertPlan {
    MPSGraph *graph;
    MPSGraphTensor *input_tensor;
    MPSGraphTensor *full_mask_tensor;
    MPSGraphTensor *local_mask_tensor;
    MPSGraphTensor *global_cos_tensor;
    MPSGraphTensor *global_sin_tensor;
    MPSGraphTensor *local_cos_tensor;
    MPSGraphTensor *local_sin_tensor;
    MPSGraphTensor *final_norm_tensor;
    MPSGraphTensor *output_tensor;
    MPSShape *hidden_shape;
    MPSShape *hidden_2d_shape;
    MPSShape *mask_shape;
    MPSShape *rope_shape;
    MPSShape *hidden_vector_shape;
    MPSShape *qkv_weight_shape;
    MPSShape *hidden_weight_shape;
    MPSShape *mlp_input_weight_shape;
    MPSShape *mlp_output_weight_shape;
    ModernBertLayerTensors *layers;
    uint64_t layer_count;
    MPSGraphExecutable *executable;
    NSArray<MPSGraphTensor *> *executable_feed_tensors;
    NSArray<MPSGraphTensor *> *layer_outputs;
    NSArray<MPSGraphTensor *> *debug_outputs;
    NSArray<NSString *> *debug_names;
} ModernBertPlan;

typedef struct ModernBertContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString *, NSValue *> *plans;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *static_buffers;
} ModernBertContext;

static void modernbert_clear_error(void) {
    modernbert_mps_error[0] = '\0';
}

static void modernbert_set_error(const char *message) {
    snprintf(modernbert_mps_error, sizeof(modernbert_mps_error), "%s", message == NULL ? "unknown MPSGraph error" : message);
}

static void modernbert_set_ns_error(NSString *message) {
    modernbert_set_error(message.UTF8String);
}

const char *synapse_modernbert_mps_last_error(void) {
    return modernbert_mps_error;
}

static MPSGraphTensor *modernbert_placeholder(
    MPSGraph *graph,
    MPSShape *shape,
    MPSDataType data_type,
    NSString *name
) {
    return [[graph placeholderWithShape:shape dataType:data_type name:name] retain];
}

static MPSGraphTensor *modernbert_cast(MPSGraph *graph, MPSGraphTensor *tensor, MPSDataType data_type) {
    return tensor.dataType == data_type ? tensor : [graph castTensor:tensor toType:data_type name:nil];
}

static MPSGraphTensor *modernbert_matmul(
    MPSGraph *graph,
    MPSGraphTensor *primary,
    MPSGraphTensor *secondary,
    MPSDataType data_type
) {
    primary = modernbert_cast(graph, primary, MPSDataTypeFloat32);
    secondary = modernbert_cast(graph, secondary, MPSDataTypeFloat32);
    MPSGraphTensor *product = [graph matrixMultiplicationWithPrimaryTensor:primary secondaryTensor:secondary name:nil];
    return modernbert_cast(graph, product, data_type);
}

static MPSGraphTensor *modernbert_linear(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *weight,
    MPSDataType data_type
) {
    MPSGraphTensor *transposed = [graph transposeTensor:weight dimension:0 withDimension:1 name:nil];
    return modernbert_matmul(graph, input, transposed, data_type);
}

static MPSGraphTensor *modernbert_layer_norm(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *weight,
    uint64_t rows,
    float epsilon,
    MPSDataType data_type
) {
    input = modernbert_cast(graph, input, MPSDataTypeFloat32);
    weight = modernbert_cast(graph, weight, MPSDataTypeFloat32);
    NSArray<NSNumber *> *axes = @[ @1 ];
    MPSShape *reduced_shape = @[ @(rows), @1 ];
    MPSGraphTensor *mean = [graph meanOfTensor:input axes:axes name:nil];
    mean = [graph reshapeTensor:mean withShape:reduced_shape name:nil];
    MPSGraphTensor *centered = [graph subtractionWithPrimaryTensor:input secondaryTensor:mean name:nil];
    MPSGraphTensor *square = [graph multiplicationWithPrimaryTensor:centered secondaryTensor:centered name:nil];
    MPSGraphTensor *variance = [graph meanOfTensor:square axes:axes name:nil];
    variance = [graph reshapeTensor:variance withShape:reduced_shape name:nil];
    MPSGraphTensor *epsilon_tensor = [graph constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
    variance = [graph additionWithPrimaryTensor:variance secondaryTensor:epsilon_tensor name:nil];
    MPSGraphTensor *standard_deviation = [graph squareRootWithTensor:variance name:nil];
    MPSGraphTensor *normalized = [graph divisionWithPrimaryTensor:centered secondaryTensor:standard_deviation name:nil];
    MPSGraphTensor *output = [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:weight name:nil];
    return modernbert_cast(graph, output, data_type);
}

static MPSGraphTensor *modernbert_gelu(MPSGraph *graph, MPSGraphTensor *input, MPSDataType data_type) {
    input = modernbert_cast(graph, input, MPSDataTypeFloat32);
    MPSGraphTensor *inverse_sqrt_two = [graph constantWithScalar:0.70710678118654752440 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *one = [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *half = [graph constantWithScalar:0.5 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *scaled = [graph multiplicationWithPrimaryTensor:input secondaryTensor:inverse_sqrt_two name:nil];
    MPSGraphTensor *erf = [graph erfWithTensor:scaled name:nil];
    MPSGraphTensor *factor = [graph additionWithPrimaryTensor:one secondaryTensor:erf name:nil];
    MPSGraphTensor *weighted = [graph multiplicationWithPrimaryTensor:input secondaryTensor:factor name:nil];
    return modernbert_cast(
        graph,
        [graph multiplicationWithPrimaryTensor:weighted secondaryTensor:half name:nil],
        data_type
    );
}

static MPSGraphTensor *modernbert_rope(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *cosine,
    MPSGraphTensor *sine,
    MPSDataType data_type
) {
    input = modernbert_cast(graph, input, MPSDataTypeFloat32);
    NSArray<MPSGraphTensor *> *halves = [graph splitTensor:input numSplits:2 axis:3 name:nil];
    MPSGraphTensor *negative_second = [graph negativeWithTensor:[halves objectAtIndex:1] name:nil];
    MPSGraphTensor *rotated = [graph concatTensors:@[ negative_second, [halves objectAtIndex:0] ] dimension:3 name:nil];
    MPSGraphTensor *cosine_part = [graph multiplicationWithPrimaryTensor:input secondaryTensor:cosine name:nil];
    MPSGraphTensor *sine_part = [graph multiplicationWithPrimaryTensor:rotated secondaryTensor:sine name:nil];
    return modernbert_cast(
        graph,
        [graph additionWithPrimaryTensor:cosine_part secondaryTensor:sine_part name:nil],
        data_type
    );
}

static void modernbert_plan_free(ModernBertPlan *plan) {
    if (plan == NULL) {
        return;
    }
    [plan->debug_names release];
    [plan->debug_outputs release];
    [plan->layer_outputs release];
    [plan->executable_feed_tensors release];
    [plan->executable release];
    if (plan->layers != NULL) {
        for (uint64_t index = 0; index < plan->layer_count; index++) {
            [plan->layers[index].qkv_weight release];
            [plan->layers[index].attention_output_weight release];
            [plan->layers[index].attention_norm_weight release];
            [plan->layers[index].mlp_input_weight release];
            [plan->layers[index].mlp_output_weight release];
            [plan->layers[index].mlp_norm_weight release];
        }
        free(plan->layers);
    }
    [plan->output_tensor release];
    [plan->final_norm_tensor release];
    [plan->local_sin_tensor release];
    [plan->local_cos_tensor release];
    [plan->global_sin_tensor release];
    [plan->global_cos_tensor release];
    [plan->local_mask_tensor release];
    [plan->full_mask_tensor release];
    [plan->input_tensor release];
    [plan->mlp_output_weight_shape release];
    [plan->mlp_input_weight_shape release];
    [plan->hidden_weight_shape release];
    [plan->qkv_weight_shape release];
    [plan->hidden_vector_shape release];
    [plan->rope_shape release];
    [plan->mask_shape release];
    [plan->hidden_2d_shape release];
    [plan->hidden_shape release];
    [plan->graph release];
    free(plan);
}

static uint64_t modernbert_attention_pattern(const ModernBertLayerParams *layers, uint64_t layer_count) {
    uint64_t pattern = 0;
    for (uint64_t index = 0; index < layer_count; index++) {
        pattern = pattern * 131 + (uint64_t)(layers[index].attention_type + 1);
    }
    return pattern;
}

static NSString *modernbert_plan_key(
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float epsilon,
    uint64_t pattern,
    int32_t dtype
) {
    return [NSString stringWithFormat:@"modernbert:%llu:%llu:%llu:%llu:%llu:%llu:%.9g:%llu:%d",
                                      (unsigned long long)batch,
                                      (unsigned long long)seq,
                                      (unsigned long long)hidden,
                                      (unsigned long long)heads,
                                      (unsigned long long)intermediate,
                                      (unsigned long long)layer_count,
                                      (double)epsilon,
                                      (unsigned long long)pattern,
                                      dtype];
}

static ModernBertPlan *modernbert_plan_new(
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float epsilon,
    const ModernBertLayerParams *params,
    int32_t dtype
) {
    ModernBertPlan *plan = (ModernBertPlan *)calloc(1, sizeof(ModernBertPlan));
    if (plan == NULL) {
        modernbert_set_error("failed to allocate ModernBERT MPSGraph plan");
        return NULL;
    }
    uint64_t rows = batch * seq;
    uint64_t head_dim = hidden / heads;
    MPSDataType data_type = dtype == 1 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
    plan->layer_count = layer_count;
    plan->hidden_shape = [@[ @(batch), @(seq), @(hidden) ] retain];
    plan->hidden_2d_shape = [@[ @(rows), @(hidden) ] retain];
    plan->mask_shape = [@[ @(batch), @1, @(seq), @(seq) ] retain];
    plan->rope_shape = [@[ @1, @1, @(seq), @(head_dim) ] retain];
    plan->hidden_vector_shape = [@[ @(hidden) ] retain];
    plan->qkv_weight_shape = [@[ @(3 * hidden), @(hidden) ] retain];
    plan->hidden_weight_shape = [@[ @(hidden), @(hidden) ] retain];
    plan->mlp_input_weight_shape = [@[ @(2 * intermediate), @(hidden) ] retain];
    plan->mlp_output_weight_shape = [@[ @(hidden), @(intermediate) ] retain];
    plan->layers = (ModernBertLayerTensors *)calloc((size_t)layer_count, sizeof(ModernBertLayerTensors));
    plan->graph = [[MPSGraph alloc] init];
    if (plan->graph == nil || plan->layers == NULL) {
        modernbert_plan_free(plan);
        modernbert_set_error("failed to allocate ModernBERT MPSGraph objects");
        return NULL;
    }
    plan->graph.options = MPSGraphOptionsSynchronizeResults;
    plan->input_tensor = modernbert_placeholder(plan->graph, plan->hidden_shape, data_type, @"hidden_input");
    plan->full_mask_tensor = modernbert_placeholder(plan->graph, plan->mask_shape, MPSDataTypeFloat32, @"full_mask");
    plan->local_mask_tensor = modernbert_placeholder(plan->graph, plan->mask_shape, MPSDataTypeFloat32, @"local_mask");
    plan->global_cos_tensor = modernbert_placeholder(plan->graph, plan->rope_shape, MPSDataTypeFloat32, @"global_cos");
    plan->global_sin_tensor = modernbert_placeholder(plan->graph, plan->rope_shape, MPSDataTypeFloat32, @"global_sin");
    plan->local_cos_tensor = modernbert_placeholder(plan->graph, plan->rope_shape, MPSDataTypeFloat32, @"local_cos");
    plan->local_sin_tensor = modernbert_placeholder(plan->graph, plan->rope_shape, MPSDataTypeFloat32, @"local_sin");
    plan->final_norm_tensor = modernbert_placeholder(plan->graph, plan->hidden_vector_shape, MPSDataTypeFloat32, @"final_norm");

    MPSGraphTensor *x = [plan->graph reshapeTensor:plan->input_tensor withShape:plan->hidden_2d_shape name:nil];
    MPSShape *qkv_shape = @[ @(batch), @(seq), @3, @(heads), @(head_dim) ];
    MPSShape *head_shape = @[ @(batch), @(seq), @(heads), @(head_dim) ];
    double attention_scale = 1.0 / sqrt((double)head_dim);
    MPSGraphTensor *scale = [plan->graph constantWithScalar:attention_scale dataType:MPSDataTypeFloat32];
    NSMutableArray<MPSGraphTensor *> *layer_outputs = [NSMutableArray arrayWithCapacity:(NSUInteger)layer_count];
    NSMutableArray<MPSGraphTensor *> *debug_outputs = [NSMutableArray array];
    NSMutableArray<NSString *> *debug_names = [NSMutableArray array];

    for (uint64_t index = 0; index < layer_count; index++) {
        ModernBertLayerTensors *layer = &plan->layers[index];
        NSString *prefix = [NSString stringWithFormat:@"layer_%llu", (unsigned long long)index];
        layer->qkv_weight = modernbert_placeholder(plan->graph, plan->qkv_weight_shape, data_type, [prefix stringByAppendingString:@"_qkv"]);
        layer->attention_output_weight = modernbert_placeholder(plan->graph, plan->hidden_weight_shape, data_type, [prefix stringByAppendingString:@"_attention_output"]);
        if (index > 0) {
            layer->attention_norm_weight = modernbert_placeholder(plan->graph, plan->hidden_vector_shape, MPSDataTypeFloat32, [prefix stringByAppendingString:@"_attention_norm"]);
        }
        layer->mlp_input_weight = modernbert_placeholder(plan->graph, plan->mlp_input_weight_shape, data_type, [prefix stringByAppendingString:@"_mlp_input"]);
        layer->mlp_output_weight = modernbert_placeholder(plan->graph, plan->mlp_output_weight_shape, data_type, [prefix stringByAppendingString:@"_mlp_output"]);
        layer->mlp_norm_weight = modernbert_placeholder(plan->graph, plan->hidden_vector_shape, MPSDataTypeFloat32, [prefix stringByAppendingString:@"_mlp_norm"]);

        MPSGraphTensor *attention_input = x;
        if (index > 0) {
            attention_input = modernbert_layer_norm(plan->graph, x, layer->attention_norm_weight, rows, epsilon, data_type);
        }
        MPSGraphTensor *qkv = modernbert_linear(plan->graph, attention_input, layer->qkv_weight, data_type);
        if (index == 0) { [debug_outputs addObject:qkv]; [debug_names addObject:@"qkv"]; }
        qkv = [plan->graph reshapeTensor:qkv withShape:qkv_shape name:nil];
        NSArray<MPSGraphTensor *> *parts = [plan->graph splitTensor:qkv numSplits:3 axis:2 name:nil];
        MPSGraphTensor *query = [plan->graph reshapeTensor:[parts objectAtIndex:0] withShape:head_shape name:nil];
        MPSGraphTensor *key = [plan->graph reshapeTensor:[parts objectAtIndex:1] withShape:head_shape name:nil];
        MPSGraphTensor *value = [plan->graph reshapeTensor:[parts objectAtIndex:2] withShape:head_shape name:nil];
        query = [plan->graph transposeTensor:query permutation:@[ @0, @2, @1, @3 ] name:nil];
        key = [plan->graph transposeTensor:key permutation:@[ @0, @2, @1, @3 ] name:nil];
        value = [plan->graph transposeTensor:value permutation:@[ @0, @2, @1, @3 ] name:nil];
        BOOL sliding = params[index].attention_type != 0;
        MPSGraphTensor *cosine = sliding ? plan->local_cos_tensor : plan->global_cos_tensor;
        MPSGraphTensor *sine = sliding ? plan->local_sin_tensor : plan->global_sin_tensor;
        query = modernbert_rope(plan->graph, query, cosine, sine, data_type);
        key = modernbert_rope(plan->graph, key, cosine, sine, data_type);
        if (index == 0) { [debug_outputs addObject:query]; [debug_names addObject:@"query-rope"]; }
        key = [plan->graph transposeTensor:key dimension:2 withDimension:3 name:nil];
        MPSGraphTensor *scores = modernbert_matmul(plan->graph, query, key, data_type);
        scores = modernbert_cast(plan->graph, scores, MPSDataTypeFloat32);
        scores = [plan->graph multiplicationWithPrimaryTensor:scores secondaryTensor:scale name:nil];
        MPSGraphTensor *mask = sliding ? plan->local_mask_tensor : plan->full_mask_tensor;
        scores = [plan->graph additionWithPrimaryTensor:scores secondaryTensor:mask name:nil];
        scores = [plan->graph softMaxWithTensor:scores axis:3 name:nil];
        if (index == 0) { [debug_outputs addObject:scores]; [debug_names addObject:@"softmax"]; }
        scores = modernbert_cast(plan->graph, scores, data_type);
        MPSGraphTensor *context = modernbert_matmul(plan->graph, scores, value, data_type);
        context = [plan->graph transposeTensor:context permutation:@[ @0, @2, @1, @3 ] name:nil];
        context = [plan->graph reshapeTensor:context withShape:plan->hidden_2d_shape name:nil];
        if (index == 0) { [debug_outputs addObject:context]; [debug_names addObject:@"context"]; }
        MPSGraphTensor *attention_output = modernbert_linear(plan->graph, context, layer->attention_output_weight, data_type);
        x = [plan->graph additionWithPrimaryTensor:x secondaryTensor:attention_output name:nil];
        if (index == 0) { [debug_outputs addObject:x]; [debug_names addObject:@"attention-residual"]; }

        MPSGraphTensor *mlp_input = modernbert_layer_norm(plan->graph, x, layer->mlp_norm_weight, rows, epsilon, data_type);
        if (index == 0) { [debug_outputs addObject:mlp_input]; [debug_names addObject:@"mlp-norm"]; }
        MPSGraphTensor *projected = modernbert_linear(plan->graph, mlp_input, layer->mlp_input_weight, data_type);
        if (index == 0) { [debug_outputs addObject:projected]; [debug_names addObject:@"mlp-projected"]; }
        NSArray<MPSGraphTensor *> *mlp_parts = [plan->graph splitTensor:projected numSplits:2 axis:1 name:nil];
        MPSGraphTensor *activated = modernbert_gelu(plan->graph, [mlp_parts objectAtIndex:0], data_type);
        activated = [plan->graph multiplicationWithPrimaryTensor:activated secondaryTensor:[mlp_parts objectAtIndex:1] name:nil];
        MPSGraphTensor *mlp_output = modernbert_linear(plan->graph, activated, layer->mlp_output_weight, data_type);
        if (index == 0) { [debug_outputs addObject:mlp_output]; [debug_names addObject:@"mlp-output"]; }
        x = [plan->graph additionWithPrimaryTensor:x secondaryTensor:mlp_output name:nil];
        [layer_outputs addObject:x];
    }
    plan->layer_outputs = [layer_outputs copy];
    plan->debug_outputs = [debug_outputs copy];
    plan->debug_names = [debug_names copy];
    x = modernbert_layer_norm(plan->graph, x, plan->final_norm_tensor, rows, epsilon, data_type);
    plan->output_tensor = [[plan->graph reshapeTensor:x withShape:plan->hidden_shape name:@"hidden_output"] retain];
    if (plan->input_tensor == nil || plan->full_mask_tensor == nil || plan->local_mask_tensor == nil ||
        plan->global_cos_tensor == nil || plan->global_sin_tensor == nil || plan->local_cos_tensor == nil ||
        plan->local_sin_tensor == nil || plan->final_norm_tensor == nil || plan->output_tensor == nil) {
        modernbert_plan_free(plan);
        modernbert_set_error("failed to construct ModernBERT MPSGraph plan");
        return NULL;
    }
    return plan;
}

static ModernBertPlan *modernbert_get_plan(
    ModernBertContext *context,
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float epsilon,
    const ModernBertLayerParams *params,
    int32_t dtype
) {
    uint64_t pattern = modernbert_attention_pattern(params, layer_count);
    NSString *key = modernbert_plan_key(batch, seq, hidden, heads, intermediate, layer_count, epsilon, pattern, dtype);
    NSValue *cached = [context->plans objectForKey:key];
    if (cached != nil) {
        return (ModernBertPlan *)cached.pointerValue;
    }
    ModernBertPlan *plan = modernbert_plan_new(batch, seq, hidden, heads, intermediate, layer_count, epsilon, params, dtype);
    if (plan != NULL) {
        [context->plans setObject:[NSValue valueWithPointer:plan] forKey:key];
    }
    return plan;
}

// Pointer identity is a cache key only for model-owned allocations that outlive this context.
// Per-call activations and masks use uncached buffers; RoPE tables are host-cached by sequence
// bucket but also use per-call buffers so no temporary pointer can alias a static tensor.
static id<MTLBuffer> modernbert_static_buffer(ModernBertContext *context, const void *values, NSUInteger bytes) {
    if (values == NULL || bytes == 0) {
        modernbert_set_error("ModernBERT static tensor is null or empty");
        return nil;
    }
    NSString *key = [NSString stringWithFormat:@"%p:%llu", values, (unsigned long long)bytes];
    id<MTLBuffer> buffer = [context->static_buffers objectForKey:key];
    if (buffer == nil) {
        buffer = [context->device newBufferWithBytes:values length:bytes options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            modernbert_set_error("failed to allocate ModernBERT static Metal buffer");
            return nil;
        }
        [context->static_buffers setObject:buffer forKey:key];
        [buffer release];
        buffer = [context->static_buffers objectForKey:key];
    }
    return buffer;
}

static BOOL modernbert_add_feed(
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    MPSGraphTensor *tensor,
    MPSShape *shape,
    id<MTLBuffer> buffer,
    MPSDataType data_type
) {
    if (tensor == nil || shape == nil || buffer == nil) {
        modernbert_set_error("ModernBERT MPSGraph feed is incomplete");
        return NO;
    }
    MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shape dataType:data_type];
    if (data == nil) {
        modernbert_set_error("failed to wrap ModernBERT Metal buffer");
        return NO;
    }
    [feeds setObject:data forKey:tensor];
    [data release];
    return YES;
}

static BOOL modernbert_add_static_feed(
    ModernBertContext *context,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    MPSGraphTensor *tensor,
    MPSShape *shape,
    const void *values,
    NSUInteger elements,
    MPSDataType data_type
) {
    NSUInteger element_size = data_type == MPSDataTypeFloat16 ? sizeof(uint16_t) : sizeof(float);
    return modernbert_add_feed(
        feeds, tensor, shape,
        modernbert_static_buffer(context, values, elements * element_size), data_type
    );
}

void *synapse_modernbert_mps_context_new(void) {
    @autoreleasepool {
        @try {
            modernbert_clear_error();
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (device == nil || queue == nil) {
                [queue release];
                [device release];
                modernbert_set_error("no Metal device or command queue is available");
                return NULL;
            }
            ModernBertContext *context = (ModernBertContext *)calloc(1, sizeof(ModernBertContext));
            if (context == NULL) {
                [queue release];
                [device release];
                modernbert_set_error("failed to allocate ModernBERT Metal context");
                return NULL;
            }
            context->device = device;
            context->queue = queue;
            context->plans = [[NSMutableDictionary alloc] init];
            context->static_buffers = [[NSMutableDictionary alloc] init];
            return context;
        } @catch (NSException *exception) {
            modernbert_set_ns_error(exception.reason);
            return NULL;
        }
    }
}

void synapse_modernbert_mps_context_free(void *raw_context) {
    if (raw_context == NULL) {
        return;
    }
    ModernBertContext *context = (ModernBertContext *)raw_context;
    for (NSValue *value in [context->plans allValues]) {
        modernbert_plan_free((ModernBertPlan *)value.pointerValue);
    }
    [context->static_buffers release];
    [context->plans release];
    [context->queue release];
    [context->device release];
    free(context);
}

int32_t synapse_modernbert_mps_forward(
    void *raw_context,
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float epsilon,
    int32_t dtype,
    int32_t explicit_execution,
    const char *package_path,
    const void *input,
    const float *full_mask,
    const float *local_mask,
    const float *global_cos,
    const float *global_sin,
    const float *local_cos,
    const float *local_sin,
    const float *final_norm,
    const ModernBertLayerParams *params,
    void *output
) {
    @autoreleasepool {
        @try {
            modernbert_clear_error();
            ModernBertContext *context = (ModernBertContext *)raw_context;
            if (context == NULL || input == NULL || full_mask == NULL || local_mask == NULL ||
                global_cos == NULL || global_sin == NULL || local_cos == NULL || local_sin == NULL ||
                final_norm == NULL || params == NULL || output == NULL) {
                modernbert_set_error("ModernBERT forward received a null pointer");
                return -1;
            }
            if (batch == 0 || seq == 0 || hidden == 0 || heads == 0 || intermediate == 0 || layer_count == 0 || hidden % heads != 0) {
                modernbert_set_error("ModernBERT forward received invalid dimensions");
                return -2;
            }
            ModernBertPlan *plan = modernbert_get_plan(context, batch, seq, hidden, heads, intermediate, layer_count, epsilon, params, dtype);
            if (plan == NULL) {
                return -3;
            }
            NSUInteger rows = (NSUInteger)(batch * seq);
            NSUInteger hidden_count = rows * (NSUInteger)hidden;
            MPSDataType data_type = dtype == 1 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
            NSUInteger element_size = dtype == 1 ? sizeof(uint16_t) : sizeof(float);
            if (explicit_execution && plan->executable == nil) {
                plan->executable = synapse_explicit_executable(
                    plan->graph, context->device, plan->output_tensor, package_path,
                    &plan->executable_feed_tensors
                );
                if (plan->executable == nil) {
                    modernbert_set_error("failed to compile or load ModernBERT executable");
                    return -7;
                }
            }
            NSUInteger mask_count = (NSUInteger)(batch * seq * seq);
            NSUInteger rope_count = (NSUInteger)(seq * (hidden / heads));
            id<MTLBuffer> input_buffer = [context->device newBufferWithBytes:input length:hidden_count * element_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> full_mask_buffer = [context->device newBufferWithBytes:full_mask length:mask_count * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> local_mask_buffer = [context->device newBufferWithBytes:local_mask length:mask_count * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> global_cos_buffer = [context->device newBufferWithBytes:global_cos length:rope_count * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> global_sin_buffer = [context->device newBufferWithBytes:global_sin length:rope_count * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> local_cos_buffer = [context->device newBufferWithBytes:local_cos length:rope_count * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> local_sin_buffer = [context->device newBufferWithBytes:local_sin length:rope_count * sizeof(float) options:MTLResourceStorageModeShared];
            if (input_buffer == nil || full_mask_buffer == nil || local_mask_buffer == nil || global_cos_buffer == nil ||
                global_sin_buffer == nil || local_cos_buffer == nil || local_sin_buffer == nil) {
                modernbert_set_error("failed to allocate ModernBERT dynamic Metal buffers");
                return -4;
            }
            NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = [[NSMutableDictionary alloc] initWithCapacity:(NSUInteger)(8 + layer_count * 6)];
            BOOL feeds_ok =
                modernbert_add_feed(feeds, plan->input_tensor, plan->hidden_shape, input_buffer, data_type) &&
                modernbert_add_feed(feeds, plan->full_mask_tensor, plan->mask_shape, full_mask_buffer, MPSDataTypeFloat32) &&
                modernbert_add_feed(feeds, plan->local_mask_tensor, plan->mask_shape, local_mask_buffer, MPSDataTypeFloat32) &&
                modernbert_add_feed(feeds, plan->global_cos_tensor, plan->rope_shape, global_cos_buffer, MPSDataTypeFloat32) &&
                modernbert_add_feed(feeds, plan->global_sin_tensor, plan->rope_shape, global_sin_buffer, MPSDataTypeFloat32) &&
                modernbert_add_feed(feeds, plan->local_cos_tensor, plan->rope_shape, local_cos_buffer, MPSDataTypeFloat32) &&
                modernbert_add_feed(feeds, plan->local_sin_tensor, plan->rope_shape, local_sin_buffer, MPSDataTypeFloat32) &&
                modernbert_add_static_feed(context, feeds, plan->final_norm_tensor, plan->hidden_vector_shape, final_norm, (NSUInteger)hidden, MPSDataTypeFloat32);
            for (uint64_t index = 0; feeds_ok && index < layer_count; index++) {
                ModernBertLayerTensors *tensors = &plan->layers[index];
                const ModernBertLayerParams *layer = &params[index];
                feeds_ok =
                    modernbert_add_static_feed(context, feeds, tensors->qkv_weight, plan->qkv_weight_shape, layer->qkv_weight, (NSUInteger)(3 * hidden * hidden), data_type) &&
                    modernbert_add_static_feed(context, feeds, tensors->attention_output_weight, plan->hidden_weight_shape, layer->attention_output_weight, (NSUInteger)(hidden * hidden), data_type) &&
                    (index == 0 || modernbert_add_static_feed(context, feeds, tensors->attention_norm_weight, plan->hidden_vector_shape, layer->attention_norm_weight, (NSUInteger)hidden, MPSDataTypeFloat32)) &&
                    modernbert_add_static_feed(context, feeds, tensors->mlp_input_weight, plan->mlp_input_weight_shape, layer->mlp_input_weight, (NSUInteger)(2 * intermediate * hidden), data_type) &&
                    modernbert_add_static_feed(context, feeds, tensors->mlp_output_weight, plan->mlp_output_weight_shape, layer->mlp_output_weight, (NSUInteger)(hidden * intermediate), data_type) &&
                    modernbert_add_static_feed(context, feeds, tensors->mlp_norm_weight, plan->hidden_vector_shape, layer->mlp_norm_weight, (NSUInteger)hidden, MPSDataTypeFloat32);
            }
            int32_t status = 0;
            if (!feeds_ok) {
                status = -5;
            } else {
                MPSGraphTensorData *result = nil;
                if (plan->executable != nil) {
                    NSArray<MPSGraphTensorData *> *inputs = synapse_executable_inputs(plan->executable_feed_tensors, feeds);
                    result = [[plan->executable runWithMTLCommandQueue:context->queue inputsArray:inputs resultsArray:nil executionDescriptor:nil] firstObject];
                } else {
                    NSString *dump_dir = [[[NSProcessInfo processInfo] environment] objectForKey:@"SYNAPSE_MODERNBERT_DUMP_DIR"];
                    NSArray<MPSGraphTensor *> *targets = @[ plan->output_tensor ];
                    if (dump_dir != nil) {
                        targets = [[plan->debug_outputs arrayByAddingObjectsFromArray:plan->layer_outputs] arrayByAddingObject:plan->output_tensor];
                    }
                    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
                        [plan->graph runWithMTLCommandQueue:context->queue feeds:feeds targetTensors:targets targetOperations:nil];
                    result = [results objectForKey:plan->output_tensor];
                    if (dump_dir != nil) {
                        [[NSFileManager defaultManager] createDirectoryAtPath:dump_dir withIntermediateDirectories:YES attributes:nil error:nil];
                        for (NSUInteger index = 0; index < plan->debug_outputs.count; index++) {
                            MPSGraphTensor *tensor = [plan->debug_outputs objectAtIndex:index];
                            NSUInteger count = 1;
                            for (NSNumber *dimension in tensor.shape) count *= dimension.unsignedIntegerValue;
                            NSUInteger stage_size = tensor.dataType == MPSDataTypeFloat16 ? sizeof(uint16_t) : sizeof(float);
                            void *stage_bytes = malloc(count * stage_size);
                            MPSGraphTensorData *stage_data = [results objectForKey:tensor];
                            [[stage_data mpsndarray] readBytes:stage_bytes strideBytes:NULL];
                            NSData *stage = [NSData dataWithBytes:stage_bytes length:count * stage_size];
                            NSString *suffix = tensor.dataType == MPSDataTypeFloat16 ? @"f16" : @"f32";
                            [stage writeToFile:[dump_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"stage-%@.%@", [plan->debug_names objectAtIndex:index], suffix]] atomically:YES];
                            free(stage_bytes);
                        }
                        void *dump_bytes = malloc(hidden_count * element_size);
                        for (NSUInteger index = 0; index < plan->layer_outputs.count; index++) {
                            MPSGraphTensorData *layer_data = [results objectForKey:[plan->layer_outputs objectAtIndex:index]];
                            [[layer_data mpsndarray] readBytes:dump_bytes strideBytes:NULL];
                            NSData *data = [NSData dataWithBytes:dump_bytes length:hidden_count * element_size];
                            [data writeToFile:[dump_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"layer-%02lu.raw", (unsigned long)index]] atomically:YES];
                        }
                        free(dump_bytes);
                    }
                }
                MPSNDArray *array = [result mpsndarray];
                if (array == nil) {
                    modernbert_set_error("MPSGraph did not return ModernBERT output");
                    status = -6;
                } else {
                    [array readBytes:output strideBytes:NULL];
                }
            }
            [feeds release];
            [input_buffer release];
            [full_mask_buffer release];
            [local_mask_buffer release];
            [global_cos_buffer release];
            [global_sin_buffer release];
            [local_cos_buffer release];
            [local_sin_buffer release];
            return status;
        } @catch (NSException *exception) {
            modernbert_set_ns_error(exception.reason);
            return -100;
        }
    }
}
