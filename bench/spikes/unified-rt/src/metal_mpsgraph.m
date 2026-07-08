#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MPSGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphActivationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphArithmeticOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMatrixMultiplicationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMemoryOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphNormalizationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorShapeOps.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static char synapse_mps_error[1024];

typedef struct SynapseMpsPlan {
    MPSGraph *graph;
    MPSShape *a_shape;
    MPSShape *b_shape;
    MPSGraphTensor *a_tensor;
    MPSGraphTensor *b_tensor;
    MPSGraphTensor *product_tensor;
} SynapseMpsPlan;

typedef struct SynapseMpsEncoderLayerParams {
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
} SynapseMpsEncoderLayerParams;

typedef struct SynapseMpsEncoderLayerTensors {
    MPSGraphTensor *query_weight;
    MPSGraphTensor *query_bias;
    MPSGraphTensor *key_weight;
    MPSGraphTensor *key_bias;
    MPSGraphTensor *value_weight;
    MPSGraphTensor *value_bias;
    MPSGraphTensor *attention_output_weight;
    MPSGraphTensor *attention_output_bias;
    MPSGraphTensor *attention_ln_weight;
    MPSGraphTensor *attention_ln_bias;
    MPSGraphTensor *intermediate_weight;
    MPSGraphTensor *intermediate_bias;
    MPSGraphTensor *output_weight;
    MPSGraphTensor *output_bias;
    MPSGraphTensor *output_ln_weight;
    MPSGraphTensor *output_ln_bias;
} SynapseMpsEncoderLayerTensors;

typedef struct SynapseMpsEncoderPlan {
    MPSGraph *graph;
    MPSShape *hidden_shape;
    MPSShape *hidden_2d_shape;
    MPSShape *mask_shape;
    MPSShape *hidden_hidden_weight_shape;
    MPSShape *hidden_bias_shape;
    MPSShape *intermediate_hidden_weight_shape;
    MPSShape *intermediate_bias_shape;
    MPSShape *hidden_intermediate_weight_shape;
    MPSGraphTensor *input_tensor;
    MPSGraphTensor *mask_tensor;
    MPSGraphTensor *output_tensor;
    SynapseMpsEncoderLayerTensors *layers;
    uint64_t layer_count;
} SynapseMpsEncoderPlan;

typedef struct SynapseMpsContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString *, NSValue *> *plans;
    NSMutableDictionary<NSString *, NSValue *> *encoder_plans;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *rhs_buffers;
} SynapseMpsContext;

static void synapse_mps_clear_error(void) {
    synapse_mps_error[0] = '\0';
}

static void synapse_mps_set_c_error(const char *message) {
    if (message == NULL) {
        message = "unknown MPSGraph error";
    }
    snprintf(synapse_mps_error, sizeof(synapse_mps_error), "%s", message);
}

static void synapse_mps_set_ns_error(NSString *message) {
    synapse_mps_set_c_error(message.UTF8String);
}

const char *synapse_mps_last_error(void) {
    return synapse_mps_error;
}

static NSString *synapse_mps_plan_key(uint64_t m, uint64_t n, uint64_t k, int32_t b_is_row_major_nk) {
    return [NSString stringWithFormat:@"%llu:%llu:%llu:%d",
                                      (unsigned long long)m,
                                      (unsigned long long)n,
                                      (unsigned long long)k,
                                      b_is_row_major_nk];
}

static NSString *synapse_mps_encoder_plan_key(
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float layer_norm_eps
) {
    return [NSString stringWithFormat:@"encoder:%llu:%llu:%llu:%llu:%llu:%llu:%.9g",
                                      (unsigned long long)batch,
                                      (unsigned long long)seq,
                                      (unsigned long long)hidden,
                                      (unsigned long long)heads,
                                      (unsigned long long)intermediate,
                                      (unsigned long long)layer_count,
                                      (double)layer_norm_eps];
}

static NSString *synapse_mps_rhs_key(const float *b, NSUInteger byte_count) {
    return [NSString stringWithFormat:@"%p:%llu", b, (unsigned long long)byte_count];
}

static void synapse_mps_plan_free(SynapseMpsPlan *plan) {
    if (plan == NULL) {
        return;
    }
    [plan->product_tensor release];
    [plan->b_tensor release];
    [plan->a_tensor release];
    [plan->b_shape release];
    [plan->a_shape release];
    [plan->graph release];
    free(plan);
}

static void synapse_mps_encoder_layer_tensors_release(SynapseMpsEncoderLayerTensors *layer) {
    [layer->output_ln_bias release];
    [layer->output_ln_weight release];
    [layer->output_bias release];
    [layer->output_weight release];
    [layer->intermediate_bias release];
    [layer->intermediate_weight release];
    [layer->attention_ln_bias release];
    [layer->attention_ln_weight release];
    [layer->attention_output_bias release];
    [layer->attention_output_weight release];
    [layer->value_bias release];
    [layer->value_weight release];
    [layer->key_bias release];
    [layer->key_weight release];
    [layer->query_bias release];
    [layer->query_weight release];
}

static void synapse_mps_encoder_plan_free(SynapseMpsEncoderPlan *plan) {
    if (plan == NULL) {
        return;
    }
    if (plan->layers != NULL) {
        for (uint64_t i = 0; i < plan->layer_count; i++) {
            synapse_mps_encoder_layer_tensors_release(&plan->layers[i]);
        }
        free(plan->layers);
    }
    [plan->output_tensor release];
    [plan->mask_tensor release];
    [plan->input_tensor release];
    [plan->hidden_intermediate_weight_shape release];
    [plan->intermediate_bias_shape release];
    [plan->intermediate_hidden_weight_shape release];
    [plan->hidden_bias_shape release];
    [plan->hidden_hidden_weight_shape release];
    [plan->mask_shape release];
    [plan->hidden_2d_shape release];
    [plan->hidden_shape release];
    [plan->graph release];
    free(plan);
}

static MPSGraphTensor *synapse_mps_placeholder(MPSGraph *graph, MPSShape *shape, NSString *name) {
    return [[graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:name] retain];
}

static MPSGraphTensor *synapse_mps_linear(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *weight,
    MPSGraphTensor *bias
) {
    MPSGraphTensor *weight_t = [graph transposeTensor:weight dimension:0 withDimension:1 name:nil];
    MPSGraphTensor *product = [graph matrixMultiplicationWithPrimaryTensor:input
                                                            secondaryTensor:weight_t
                                                                       name:nil];
    return [graph additionWithPrimaryTensor:product secondaryTensor:bias name:nil];
}

static MPSGraphTensor *synapse_mps_layer_norm(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *weight,
    MPSGraphTensor *bias,
    uint64_t rows,
    float epsilon
) {
    NSArray<NSNumber *> *axes = @[ @1 ];
    MPSShape *mean_shape = @[ @(rows), @1 ];
    MPSGraphTensor *mean = [graph meanOfTensor:input axes:axes name:nil];
    MPSGraphTensor *variance = [graph varianceOfTensor:input meanTensor:mean axes:axes name:nil];
    mean = [graph reshapeTensor:mean withShape:mean_shape name:nil];
    variance = [graph reshapeTensor:variance withShape:mean_shape name:nil];
    MPSGraphTensor *centered = [graph subtractionWithPrimaryTensor:input secondaryTensor:mean name:nil];
    MPSGraphTensor *eps = [graph constantWithScalar:(double)epsilon dataType:MPSDataTypeFloat32];
    MPSGraphTensor *variance_eps = [graph additionWithPrimaryTensor:variance secondaryTensor:eps name:nil];
    MPSGraphTensor *std = [graph squareRootWithTensor:variance_eps name:nil];
    MPSGraphTensor *normalized = [graph divisionWithPrimaryTensor:centered secondaryTensor:std name:nil];
    MPSGraphTensor *scaled = [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:weight name:nil];
    return [graph additionWithPrimaryTensor:scaled secondaryTensor:bias name:nil];
}

static MPSGraphTensor *synapse_mps_gelu(MPSGraph *graph, MPSGraphTensor *input) {
    MPSGraphTensor *inv_sqrt2 = [graph constantWithScalar:0.70710678118654752440 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *one = [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *half = [graph constantWithScalar:0.5 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *scaled = [graph multiplicationWithPrimaryTensor:input secondaryTensor:inv_sqrt2 name:nil];
    MPSGraphTensor *erf = [graph erfWithTensor:scaled name:nil];
    MPSGraphTensor *one_plus_erf = [graph additionWithPrimaryTensor:one secondaryTensor:erf name:nil];
    MPSGraphTensor *weighted = [graph multiplicationWithPrimaryTensor:input secondaryTensor:one_plus_erf name:nil];
    return [graph multiplicationWithPrimaryTensor:weighted secondaryTensor:half name:nil];
}

static BOOL synapse_mps_encoder_layer_tensors_valid(SynapseMpsEncoderLayerTensors *layer) {
    return layer->query_weight != nil && layer->query_bias != nil &&
           layer->key_weight != nil && layer->key_bias != nil &&
           layer->value_weight != nil && layer->value_bias != nil &&
           layer->attention_output_weight != nil && layer->attention_output_bias != nil &&
           layer->attention_ln_weight != nil && layer->attention_ln_bias != nil &&
           layer->intermediate_weight != nil && layer->intermediate_bias != nil &&
           layer->output_weight != nil && layer->output_bias != nil &&
           layer->output_ln_weight != nil && layer->output_ln_bias != nil;
}

static SynapseMpsPlan *synapse_mps_plan_new(
    uint64_t m,
    uint64_t n,
    uint64_t k,
    int32_t b_is_row_major_nk
) {
    SynapseMpsPlan *plan = (SynapseMpsPlan *)calloc(1, sizeof(SynapseMpsPlan));
    if (plan == NULL) {
        synapse_mps_set_c_error("failed to allocate MPSGraph matmul plan");
        return NULL;
    }

    const NSUInteger rows = (NSUInteger)m;
    const NSUInteger cols = (NSUInteger)n;
    const NSUInteger inner = (NSUInteger)k;
    plan->a_shape = [@[ @(rows), @(inner) ] retain];
    plan->b_shape = [b_is_row_major_nk ? @[ @(cols), @(inner) ] : @[ @(inner), @(cols) ] retain];
    plan->graph = [[MPSGraph alloc] init];
    plan->graph.options = MPSGraphOptionsSynchronizeResults;
    plan->a_tensor = [[plan->graph placeholderWithShape:plan->a_shape
                                               dataType:MPSDataTypeFloat32
                                                   name:@"a"] retain];
    plan->b_tensor = [[plan->graph placeholderWithShape:plan->b_shape
                                               dataType:MPSDataTypeFloat32
                                                   name:@"b"] retain];
    MPSGraphTensor *rhs_tensor = plan->b_tensor;
    if (b_is_row_major_nk) {
        rhs_tensor = [plan->graph transposeTensor:plan->b_tensor
                                        dimension:0
                                    withDimension:1
                                             name:@"b_transposed"];
    }
    plan->product_tensor = [[plan->graph matrixMultiplicationWithPrimaryTensor:plan->a_tensor
                                                               secondaryTensor:rhs_tensor
                                                                          name:@"product"] retain];
    if (plan->a_shape == nil || plan->b_shape == nil || plan->graph == nil ||
        plan->a_tensor == nil || plan->b_tensor == nil || plan->product_tensor == nil) {
        synapse_mps_plan_free(plan);
        synapse_mps_set_c_error("failed to create MPSGraph matmul plan");
        return NULL;
    }
    return plan;
}

static SynapseMpsEncoderPlan *synapse_mps_encoder_plan_new(
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float layer_norm_eps
) {
    SynapseMpsEncoderPlan *plan = (SynapseMpsEncoderPlan *)calloc(1, sizeof(SynapseMpsEncoderPlan));
    if (plan == NULL) {
        synapse_mps_set_c_error("failed to allocate MPSGraph encoder plan");
        return NULL;
    }

    const uint64_t rows = batch * seq;
    const uint64_t head_dim = hidden / heads;
    plan->layer_count = layer_count;
    plan->hidden_shape = [@[ @(batch), @(seq), @(hidden) ] retain];
    plan->hidden_2d_shape = [@[ @(rows), @(hidden) ] retain];
    plan->mask_shape = [@[ @(batch), @1, @1, @(seq) ] retain];
    plan->hidden_hidden_weight_shape = [@[ @(hidden), @(hidden) ] retain];
    plan->hidden_bias_shape = [@[ @(hidden) ] retain];
    plan->intermediate_hidden_weight_shape = [@[ @(intermediate), @(hidden) ] retain];
    plan->intermediate_bias_shape = [@[ @(intermediate) ] retain];
    plan->hidden_intermediate_weight_shape = [@[ @(hidden), @(intermediate) ] retain];
    plan->layers = (SynapseMpsEncoderLayerTensors *)calloc((size_t)layer_count, sizeof(SynapseMpsEncoderLayerTensors));
    plan->graph = [[MPSGraph alloc] init];
    if (plan->hidden_shape == nil || plan->hidden_2d_shape == nil || plan->mask_shape == nil ||
        plan->hidden_hidden_weight_shape == nil || plan->hidden_bias_shape == nil ||
        plan->intermediate_hidden_weight_shape == nil || plan->intermediate_bias_shape == nil ||
        plan->hidden_intermediate_weight_shape == nil || plan->layers == NULL || plan->graph == nil) {
        synapse_mps_encoder_plan_free(plan);
        synapse_mps_set_c_error("failed to allocate MPSGraph encoder plan objects");
        return NULL;
    }
    plan->graph.options = MPSGraphOptionsSynchronizeResults;

    plan->input_tensor = synapse_mps_placeholder(plan->graph, plan->hidden_shape, @"encoder_input");
    plan->mask_tensor = synapse_mps_placeholder(plan->graph, plan->mask_shape, @"attention_mask_additive");
    MPSGraphTensor *x = [plan->graph reshapeTensor:plan->input_tensor withShape:plan->hidden_2d_shape name:nil];
    MPSShape *hidden_4d_shape = @[ @(batch), @(seq), @(heads), @(head_dim) ];

    for (uint64_t layer_index = 0; layer_index < layer_count; layer_index++) {
        SynapseMpsEncoderLayerTensors *layer = &plan->layers[layer_index];
        NSString *prefix = [NSString stringWithFormat:@"layer_%llu", (unsigned long long)layer_index];
        layer->query_weight = synapse_mps_placeholder(plan->graph, plan->hidden_hidden_weight_shape, [prefix stringByAppendingString:@"_query_weight"]);
        layer->query_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, [prefix stringByAppendingString:@"_query_bias"]);
        layer->key_weight = synapse_mps_placeholder(plan->graph, plan->hidden_hidden_weight_shape, [prefix stringByAppendingString:@"_key_weight"]);
        layer->key_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, [prefix stringByAppendingString:@"_key_bias"]);
        layer->value_weight = synapse_mps_placeholder(plan->graph, plan->hidden_hidden_weight_shape, [prefix stringByAppendingString:@"_value_weight"]);
        layer->value_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, [prefix stringByAppendingString:@"_value_bias"]);
        layer->attention_output_weight = synapse_mps_placeholder(plan->graph, plan->hidden_hidden_weight_shape, [prefix stringByAppendingString:@"_attention_output_weight"]);
        layer->attention_output_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, [prefix stringByAppendingString:@"_attention_output_bias"]);
        layer->attention_ln_weight = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, [prefix stringByAppendingString:@"_attention_ln_weight"]);
        layer->attention_ln_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, [prefix stringByAppendingString:@"_attention_ln_bias"]);
        layer->intermediate_weight = synapse_mps_placeholder(plan->graph, plan->intermediate_hidden_weight_shape, [prefix stringByAppendingString:@"_intermediate_weight"]);
        layer->intermediate_bias = synapse_mps_placeholder(plan->graph, plan->intermediate_bias_shape, [prefix stringByAppendingString:@"_intermediate_bias"]);
        layer->output_weight = synapse_mps_placeholder(plan->graph, plan->hidden_intermediate_weight_shape, [prefix stringByAppendingString:@"_output_weight"]);
        layer->output_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, [prefix stringByAppendingString:@"_output_bias"]);
        layer->output_ln_weight = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, [prefix stringByAppendingString:@"_output_ln_weight"]);
        layer->output_ln_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, [prefix stringByAppendingString:@"_output_ln_bias"]);
        if (!synapse_mps_encoder_layer_tensors_valid(layer)) {
            synapse_mps_encoder_plan_free(plan);
            synapse_mps_set_c_error("failed to create MPSGraph encoder placeholders");
            return NULL;
        }

        MPSGraphTensor *attention_residual = x;
        MPSGraphTensor *q = synapse_mps_linear(plan->graph, x, layer->query_weight, layer->query_bias);
        MPSGraphTensor *k = synapse_mps_linear(plan->graph, x, layer->key_weight, layer->key_bias);
        MPSGraphTensor *v = synapse_mps_linear(plan->graph, x, layer->value_weight, layer->value_bias);
        q = [plan->graph reshapeTensor:q withShape:hidden_4d_shape name:nil];
        k = [plan->graph reshapeTensor:k withShape:hidden_4d_shape name:nil];
        v = [plan->graph reshapeTensor:v withShape:hidden_4d_shape name:nil];
        q = [plan->graph transposeTensor:q permutation:@[ @0, @2, @1, @3 ] name:nil];
        k = [plan->graph transposeTensor:k permutation:@[ @0, @2, @1, @3 ] name:nil];
        v = [plan->graph transposeTensor:v permutation:@[ @0, @2, @1, @3 ] name:nil];
        k = [plan->graph transposeTensor:k dimension:2 withDimension:3 name:nil];
        MPSGraphTensor *scores = [plan->graph matrixMultiplicationWithPrimaryTensor:q secondaryTensor:k name:nil];
        MPSGraphTensor *scale = [plan->graph constantWithScalar:(1.0 / sqrt((double)head_dim)) dataType:MPSDataTypeFloat32];
        scores = [plan->graph multiplicationWithPrimaryTensor:scores secondaryTensor:scale name:nil];
        scores = [plan->graph additionWithPrimaryTensor:scores secondaryTensor:plan->mask_tensor name:nil];
        scores = [plan->graph softMaxWithTensor:scores axis:3 name:nil];
        MPSGraphTensor *context = [plan->graph matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v name:nil];
        context = [plan->graph transposeTensor:context permutation:@[ @0, @2, @1, @3 ] name:nil];
        context = [plan->graph reshapeTensor:context withShape:plan->hidden_2d_shape name:nil];

        MPSGraphTensor *attention_out = synapse_mps_linear(plan->graph, context, layer->attention_output_weight, layer->attention_output_bias);
        attention_out = [plan->graph additionWithPrimaryTensor:attention_out secondaryTensor:attention_residual name:nil];
        x = synapse_mps_layer_norm(plan->graph, attention_out, layer->attention_ln_weight, layer->attention_ln_bias, rows, layer_norm_eps);

        MPSGraphTensor *ffn_residual = x;
        MPSGraphTensor *intermediate_out = synapse_mps_linear(plan->graph, x, layer->intermediate_weight, layer->intermediate_bias);
        intermediate_out = synapse_mps_gelu(plan->graph, intermediate_out);
        MPSGraphTensor *output = synapse_mps_linear(plan->graph, intermediate_out, layer->output_weight, layer->output_bias);
        output = [plan->graph additionWithPrimaryTensor:output secondaryTensor:ffn_residual name:nil];
        x = synapse_mps_layer_norm(plan->graph, output, layer->output_ln_weight, layer->output_ln_bias, rows, layer_norm_eps);
    }

    plan->output_tensor = [[plan->graph reshapeTensor:x withShape:plan->hidden_shape name:@"encoder_output"] retain];
    if (plan->input_tensor == nil || plan->mask_tensor == nil || plan->output_tensor == nil) {
        synapse_mps_encoder_plan_free(plan);
        synapse_mps_set_c_error("failed to create MPSGraph encoder output");
        return NULL;
    }
    return plan;
}

static SynapseMpsPlan *synapse_mps_get_plan(
    SynapseMpsContext *context,
    uint64_t m,
    uint64_t n,
    uint64_t k,
    int32_t b_is_row_major_nk
) {
    NSString *key = synapse_mps_plan_key(m, n, k, b_is_row_major_nk);
    NSValue *cached = [context->plans objectForKey:key];
    if (cached != nil) {
        return (SynapseMpsPlan *)cached.pointerValue;
    }

    SynapseMpsPlan *plan = synapse_mps_plan_new(m, n, k, b_is_row_major_nk);
    if (plan == NULL) {
        return NULL;
    }
    [context->plans setObject:[NSValue valueWithPointer:plan] forKey:key];
    return plan;
}

static SynapseMpsEncoderPlan *synapse_mps_get_encoder_plan(
    SynapseMpsContext *context,
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float layer_norm_eps
) {
    NSString *key = synapse_mps_encoder_plan_key(batch, seq, hidden, heads, intermediate, layer_count, layer_norm_eps);
    NSValue *cached = [context->encoder_plans objectForKey:key];
    if (cached != nil) {
        return (SynapseMpsEncoderPlan *)cached.pointerValue;
    }

    SynapseMpsEncoderPlan *plan = synapse_mps_encoder_plan_new(batch, seq, hidden, heads, intermediate, layer_count, layer_norm_eps);
    if (plan == NULL) {
        return NULL;
    }
    [context->encoder_plans setObject:[NSValue valueWithPointer:plan] forKey:key];
    return plan;
}

static id<MTLBuffer> synapse_mps_get_cached_buffer(
    SynapseMpsContext *context,
    const float *values,
    NSUInteger byte_count
) {
    if (values == NULL || byte_count == 0) {
        synapse_mps_set_c_error("encoder feed received a null or empty static tensor");
        return nil;
    }
    NSString *key = synapse_mps_rhs_key(values, byte_count);
    id<MTLBuffer> buffer = [context->rhs_buffers objectForKey:key];
    if (buffer != nil) {
        return buffer;
    }
    buffer = [context->device newBufferWithBytes:values
                                         length:byte_count
                                        options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        synapse_mps_set_c_error("failed to allocate Metal static encoder buffer");
        return nil;
    }
    [context->rhs_buffers setObject:buffer forKey:key];
    [buffer release];
    return [context->rhs_buffers objectForKey:key];
}

static BOOL synapse_mps_add_feed(
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    MPSGraphTensor *tensor,
    MPSShape *shape,
    id<MTLBuffer> buffer
) {
    if (tensor == nil || shape == nil || buffer == nil) {
        synapse_mps_set_c_error("encoder feed is missing a tensor, shape, or Metal buffer");
        return NO;
    }
    MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                       shape:shape
                                                                    dataType:MPSDataTypeFloat32];
    if (data == nil) {
        synapse_mps_set_c_error("failed to wrap Metal buffer for MPSGraph encoder feed");
        return NO;
    }
    [feeds setObject:data forKey:tensor];
    [data release];
    return YES;
}

static BOOL synapse_mps_add_cached_feed(
    SynapseMpsContext *context,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    MPSGraphTensor *tensor,
    MPSShape *shape,
    const float *values,
    NSUInteger element_count
) {
    id<MTLBuffer> buffer = synapse_mps_get_cached_buffer(context, values, element_count * sizeof(float));
    if (buffer == nil) {
        return NO;
    }
    return synapse_mps_add_feed(feeds, tensor, shape, buffer);
}

void *synapse_mps_context_new(void) {
    @autoreleasepool {
        @try {
            synapse_mps_clear_error();
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                synapse_mps_set_c_error("no Metal device is available");
                return NULL;
            }
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (queue == nil) {
                [device release];
                synapse_mps_set_c_error("failed to create Metal command queue");
                return NULL;
            }

            SynapseMpsContext *context = (SynapseMpsContext *)calloc(1, sizeof(SynapseMpsContext));
            if (context == NULL) {
                [queue release];
                [device release];
                synapse_mps_set_c_error("failed to allocate Metal context");
                return NULL;
            }
            context->device = device;
            context->queue = queue;
            context->plans = [[NSMutableDictionary alloc] init];
            context->encoder_plans = [[NSMutableDictionary alloc] init];
            context->rhs_buffers = [[NSMutableDictionary alloc] init];
            if (context->plans == nil || context->encoder_plans == nil || context->rhs_buffers == nil) {
                [context->rhs_buffers release];
                [context->encoder_plans release];
                [context->plans release];
                [queue release];
                [device release];
                free(context);
                synapse_mps_set_c_error("failed to allocate Metal cache dictionaries");
                return NULL;
            }
            return context;
        } @catch (NSException *exception) {
            synapse_mps_set_ns_error(exception.reason);
            return NULL;
        }
    }
}

void synapse_mps_context_free(void *raw_context) {
    if (raw_context == NULL) {
        return;
    }
    SynapseMpsContext *context = (SynapseMpsContext *)raw_context;
    for (NSValue *value in [context->plans allValues]) {
        synapse_mps_plan_free((SynapseMpsPlan *)value.pointerValue);
    }
    for (NSValue *value in [context->encoder_plans allValues]) {
        synapse_mps_encoder_plan_free((SynapseMpsEncoderPlan *)value.pointerValue);
    }
    [context->rhs_buffers release];
    [context->encoder_plans release];
    [context->plans release];
    [context->queue release];
    [context->device release];
    free(context);
}

int32_t synapse_mps_matmul(
    void *raw_context,
    uint64_t m,
    uint64_t n,
    uint64_t k,
    const float *a,
    const float *b,
    int32_t b_is_row_major_nk,
    float *c,
    int32_t cache_rhs
) {
    @autoreleasepool {
        @try {
            synapse_mps_clear_error();
            SynapseMpsContext *context = (SynapseMpsContext *)raw_context;
            if (context == NULL || context->device == nil || context->queue == nil) {
                synapse_mps_set_c_error("Metal context is not initialized");
                return -1;
            }
            if (a == NULL || b == NULL || c == NULL) {
                synapse_mps_set_c_error("matmul received a null data pointer");
                return -2;
            }
            if (m == 0 || n == 0 || k == 0) {
                synapse_mps_set_c_error("matmul dimensions must be non-zero");
                return -3;
            }
            if (m > NSUIntegerMax || n > NSUIntegerMax || k > NSUIntegerMax) {
                synapse_mps_set_c_error("matmul dimensions exceed NSUIntegerMax");
                return -4;
            }

            const NSUInteger rows = (NSUInteger)m;
            const NSUInteger cols = (NSUInteger)n;
            const NSUInteger inner = (NSUInteger)k;
            const NSUInteger a_count = rows * inner;
            const NSUInteger b_count = b_is_row_major_nk ? cols * inner : inner * cols;
            const NSUInteger a_bytes = a_count * sizeof(float);
            const NSUInteger b_bytes = b_count * sizeof(float);

            SynapseMpsPlan *plan = synapse_mps_get_plan(context, m, n, k, b_is_row_major_nk);
            if (plan == NULL) {
                return -5;
            }

            id<MTLBuffer> a_buffer = [context->device newBufferWithBytes:a
                                                                   length:a_bytes
                                                                  options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_buffer = nil;
            BOOL release_b_buffer = NO;
            if (cache_rhs) {
                b_buffer = synapse_mps_get_cached_buffer(context, b, b_bytes);
            } else {
                b_buffer = [context->device newBufferWithBytes:b
                                                        length:b_bytes
                                                       options:MTLResourceStorageModeShared];
                release_b_buffer = YES;
            }
            if (a_buffer == nil || b_buffer == nil) {
                [a_buffer release];
                if (release_b_buffer) {
                    [b_buffer release];
                }
                synapse_mps_set_c_error("failed to allocate Metal input buffers");
                return -6;
            }

            MPSGraphTensorData *a_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:a_buffer
                                                                                 shape:plan->a_shape
                                                                              dataType:MPSDataTypeFloat32];
            MPSGraphTensorData *b_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:b_buffer
                                                                                 shape:plan->b_shape
                                                                              dataType:MPSDataTypeFloat32];
            if (a_data == nil || b_data == nil) {
                [a_data release];
                [b_data release];
                [a_buffer release];
                if (release_b_buffer) {
                    [b_buffer release];
                }
                synapse_mps_set_c_error("failed to wrap Metal input buffers for MPSGraph");
                return -7;
            }

            NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
                plan->a_tensor: a_data,
                plan->b_tensor: b_data,
            };
            NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
                [plan->graph runWithMTLCommandQueue:context->queue
                                              feeds:feeds
                                      targetTensors:@[ plan->product_tensor ]
                                   targetOperations:nil];
            MPSGraphTensorData *product_data = [results objectForKey:plan->product_tensor];
            if (product_data == nil) {
                [a_data release];
                [b_data release];
                [a_buffer release];
                if (release_b_buffer) {
                    [b_buffer release];
                }
                synapse_mps_set_c_error("MPSGraph did not return matmul output");
                return -8;
            }
            MPSNDArray *product_array = [product_data mpsndarray];
            if (product_array == nil) {
                [a_data release];
                [b_data release];
                [a_buffer release];
                if (release_b_buffer) {
                    [b_buffer release];
                }
                synapse_mps_set_c_error("MPSGraph output could not be read as an MPSNDArray");
                return -9;
            }
            [product_array readBytes:c strideBytes:NULL];

            [a_data release];
            [b_data release];
            [a_buffer release];
            if (release_b_buffer) {
                [b_buffer release];
            }
            return 0;
        } @catch (NSException *exception) {
            synapse_mps_set_ns_error(exception.reason);
            return -100;
        }
    }
}

int32_t synapse_mps_encoder_forward(
    void *raw_context,
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float layer_norm_eps,
    const float *input,
    const float *additive_mask,
    float *output,
    const SynapseMpsEncoderLayerParams *layers
) {
    @autoreleasepool {
        @try {
            synapse_mps_clear_error();
            SynapseMpsContext *context = (SynapseMpsContext *)raw_context;
            if (context == NULL || context->device == nil || context->queue == nil) {
                synapse_mps_set_c_error("Metal context is not initialized");
                return -1;
            }
            if (input == NULL || additive_mask == NULL || output == NULL || layers == NULL) {
                synapse_mps_set_c_error("encoder forward received a null data pointer");
                return -2;
            }
            if (batch == 0 || seq == 0 || hidden == 0 || heads == 0 || intermediate == 0 || layer_count == 0) {
                synapse_mps_set_c_error("encoder dimensions must be non-zero");
                return -3;
            }
            if (hidden % heads != 0) {
                synapse_mps_set_c_error("encoder hidden size must divide attention heads");
                return -4;
            }
            if (batch > NSUIntegerMax || seq > NSUIntegerMax || hidden > NSUIntegerMax ||
                heads > NSUIntegerMax || intermediate > NSUIntegerMax || layer_count > NSUIntegerMax) {
                synapse_mps_set_c_error("encoder dimensions exceed NSUIntegerMax");
                return -5;
            }

            const NSUInteger rows = (NSUInteger)(batch * seq);
            const NSUInteger hidden_count = rows * (NSUInteger)hidden;
            const NSUInteger mask_count = (NSUInteger)(batch * seq);
            const NSUInteger hidden_hidden_count = (NSUInteger)(hidden * hidden);
            const NSUInteger intermediate_hidden_count = (NSUInteger)(intermediate * hidden);
            const NSUInteger hidden_intermediate_count = (NSUInteger)(hidden * intermediate);
            const NSUInteger input_bytes = hidden_count * sizeof(float);
            const NSUInteger mask_bytes = mask_count * sizeof(float);

            SynapseMpsEncoderPlan *plan = synapse_mps_get_encoder_plan(
                context,
                batch,
                seq,
                hidden,
                heads,
                intermediate,
                layer_count,
                layer_norm_eps
            );
            if (plan == NULL) {
                return -6;
            }

            id<MTLBuffer> input_buffer = [context->device newBufferWithBytes:input
                                                                       length:input_bytes
                                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> mask_buffer = [context->device newBufferWithBytes:additive_mask
                                                                      length:mask_bytes
                                                                     options:MTLResourceStorageModeShared];
            if (input_buffer == nil || mask_buffer == nil) {
                [input_buffer release];
                [mask_buffer release];
                synapse_mps_set_c_error("failed to allocate Metal encoder input buffers");
                return -7;
            }

            NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
                [[NSMutableDictionary alloc] initWithCapacity:(NSUInteger)(2 + layer_count * 16)];
            if (feeds == nil) {
                [input_buffer release];
                [mask_buffer release];
                synapse_mps_set_c_error("failed to allocate MPSGraph encoder feeds");
                return -8;
            }
            if (!synapse_mps_add_feed(feeds, plan->input_tensor, plan->hidden_shape, input_buffer) ||
                !synapse_mps_add_feed(feeds, plan->mask_tensor, plan->mask_shape, mask_buffer)) {
                [feeds release];
                [input_buffer release];
                [mask_buffer release];
                return -9;
            }

            for (uint64_t layer_index = 0; layer_index < layer_count; layer_index++) {
                const SynapseMpsEncoderLayerParams *params = &layers[layer_index];
                SynapseMpsEncoderLayerTensors *tensors = &plan->layers[layer_index];
                if (!synapse_mps_add_cached_feed(context, feeds, tensors->query_weight, plan->hidden_hidden_weight_shape, params->query_weight, hidden_hidden_count) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->query_bias, plan->hidden_bias_shape, params->query_bias, (NSUInteger)hidden) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->key_weight, plan->hidden_hidden_weight_shape, params->key_weight, hidden_hidden_count) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->key_bias, plan->hidden_bias_shape, params->key_bias, (NSUInteger)hidden) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->value_weight, plan->hidden_hidden_weight_shape, params->value_weight, hidden_hidden_count) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->value_bias, plan->hidden_bias_shape, params->value_bias, (NSUInteger)hidden) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->attention_output_weight, plan->hidden_hidden_weight_shape, params->attention_output_weight, hidden_hidden_count) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->attention_output_bias, plan->hidden_bias_shape, params->attention_output_bias, (NSUInteger)hidden) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->attention_ln_weight, plan->hidden_bias_shape, params->attention_ln_weight, (NSUInteger)hidden) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->attention_ln_bias, plan->hidden_bias_shape, params->attention_ln_bias, (NSUInteger)hidden) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->intermediate_weight, plan->intermediate_hidden_weight_shape, params->intermediate_weight, intermediate_hidden_count) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->intermediate_bias, plan->intermediate_bias_shape, params->intermediate_bias, (NSUInteger)intermediate) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->output_weight, plan->hidden_intermediate_weight_shape, params->output_weight, hidden_intermediate_count) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->output_bias, plan->hidden_bias_shape, params->output_bias, (NSUInteger)hidden) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->output_ln_weight, plan->hidden_bias_shape, params->output_ln_weight, (NSUInteger)hidden) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->output_ln_bias, plan->hidden_bias_shape, params->output_ln_bias, (NSUInteger)hidden)) {
                    [feeds release];
                    [input_buffer release];
                    [mask_buffer release];
                    return -10;
                }
            }

            NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
                [plan->graph runWithMTLCommandQueue:context->queue
                                              feeds:feeds
                                      targetTensors:@[ plan->output_tensor ]
                                   targetOperations:nil];
            MPSGraphTensorData *output_data = [results objectForKey:plan->output_tensor];
            if (output_data == nil) {
                [feeds release];
                [input_buffer release];
                [mask_buffer release];
                synapse_mps_set_c_error("MPSGraph did not return encoder output");
                return -11;
            }
            MPSNDArray *output_array = [output_data mpsndarray];
            if (output_array == nil) {
                [feeds release];
                [input_buffer release];
                [mask_buffer release];
                synapse_mps_set_c_error("MPSGraph encoder output could not be read as an MPSNDArray");
                return -12;
            }
            [output_array readBytes:output strideBytes:NULL];

            [feeds release];
            [input_buffer release];
            [mask_buffer release];
            return 0;
        } @catch (NSException *exception) {
            synapse_mps_set_ns_error(exception.reason);
            return -100;
        }
    }
}
