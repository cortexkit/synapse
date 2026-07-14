#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MPSGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphActivationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphArithmeticOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphExecutable.h>
#import <MetalPerformanceShadersGraph/MPSGraphMatrixMultiplicationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMemoryOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphNormalizationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphReductionOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorShapeOps.h>

#include "mpsgraph_runtime.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static char synapse_mps_error[1024];

typedef NS_ENUM(int32_t, SynapseMpsDType) {
    SynapseMpsDTypeFloat32 = 0,
    SynapseMpsDTypeFloat16 = 1,
};

typedef NS_ENUM(int32_t, SynapseEvidenceVariant) {
    SynapseEvidenceFp32 = 0,
    SynapseEvidencePureF16 = 1,
    SynapseEvidenceOnlyMatmulF16 = 2,
    SynapseEvidenceOnlyLayernormF16 = 3,
    SynapseEvidenceOnlyGeluF16 = 4,
    SynapseEvidenceOnlyAttentionF16 = 5,
    SynapseEvidenceOnlyIoF16 = 6,
    SynapseEvidenceExceptMatmulFp32 = 7,
    SynapseEvidenceExceptLayernormFp32 = 8,
    SynapseEvidenceExceptGeluFp32 = 9,
    SynapseEvidenceExceptAttentionFp32 = 10,
    SynapseEvidenceExceptIoFp32 = 11,
    SynapseEvidenceWeightsF16ComputeF32 = 12,
    SynapseEvidenceWeightsF32ActivationsF16 = 13,
    SynapseEvidenceF16Fp32MatmulIslands = 14,
};

typedef struct SynapsePrecisionPolicy {
    MPSDataType io_type;
    MPSDataType weight_type;
    MPSDataType base_type;
    MPSDataType linear_type;
    MPSDataType layernorm_type;
    MPSDataType gelu_type;
    MPSDataType attention_type;
    MPSDataType softmax_type;
} SynapsePrecisionPolicy;

typedef struct SynapseMpsPlan {
    MPSGraph *graph;
    MPSShape *a_shape;
    MPSShape *b_shape;
    MPSGraphTensor *a_tensor;
    MPSGraphTensor *b_tensor;
    MPSGraphTensor *product_tensor;
} SynapseMpsPlan;

typedef struct SynapseMpsEncoderLayerParams {
    const void *query_weight;
    const void *query_bias;
    const void *key_weight;
    const void *key_bias;
    const void *value_weight;
    const void *value_bias;
    const void *attention_output_weight;
    const void *attention_output_bias;
    const void *attention_ln_weight;
    const void *attention_ln_bias;
    const void *intermediate_weight;
    const void *intermediate_bias;
    const void *output_weight;
    const void *output_bias;
    const void *output_ln_weight;
    const void *output_ln_bias;
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
    MPSGraphExecutable *executable;
    NSArray<MPSGraphTensor *> *executable_feed_tensors;
    SynapseMpsEncoderLayerTensors *layers;
    uint64_t layer_count;
} SynapseMpsEncoderPlan;

typedef struct SynapseMpsContext {
    SynapseMpsRuntimeContext runtime;
    NSMutableDictionary<NSString *, NSValue *> *encoder_plans;
    NSString *capture_path;
    NSString *graph_dump_path;
    BOOL capture_pending;
    BOOL graph_dump_pending;
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

static MPSDataType synapse_mps_data_type(int32_t dtype) {
    switch (dtype) {
        case SynapseMpsDTypeFloat16:
            return MPSDataTypeFloat16;
        case SynapseMpsDTypeFloat32:
        default:
            return MPSDataTypeFloat32;
    }
}

static NSUInteger synapse_mps_dtype_size(int32_t dtype) {
    switch (dtype) {
        case SynapseMpsDTypeFloat16:
            return sizeof(uint16_t);
        case SynapseMpsDTypeFloat32:
        default:
            return sizeof(float);
    }
}

static NSUInteger synapse_mps_type_size(MPSDataType data_type) {
    return data_type == MPSDataTypeFloat16 ? sizeof(uint16_t) : sizeof(float);
}

static SynapsePrecisionPolicy synapse_mps_precision_policy(int32_t variant) {
    const MPSDataType f32 = MPSDataTypeFloat32;
    const MPSDataType f16 = MPSDataTypeFloat16;
    SynapsePrecisionPolicy policy = {f32, f32, f32, f32, f32, f32, f32, f32};
    switch (variant) {
        case SynapseEvidencePureF16:
            policy = (SynapsePrecisionPolicy){f16, f16, f16, f16, f32, f32, f16, f32};
            break;
        case SynapseEvidenceOnlyMatmulF16:
            policy.linear_type = f16;
            policy.weight_type = f16;
            break;
        case SynapseEvidenceOnlyLayernormF16:
            policy.layernorm_type = f16;
            break;
        case SynapseEvidenceOnlyGeluF16:
            policy.gelu_type = f16;
            break;
        case SynapseEvidenceOnlyAttentionF16:
            policy.attention_type = f16;
            policy.softmax_type = f16;
            break;
        case SynapseEvidenceOnlyIoF16:
            policy.io_type = f16;
            break;
        case SynapseEvidenceExceptMatmulFp32:
            policy = (SynapsePrecisionPolicy){f16, f32, f16, f32, f16, f16, f16, f16};
            break;
        case SynapseEvidenceExceptLayernormFp32:
            policy = (SynapsePrecisionPolicy){f16, f16, f16, f16, f32, f16, f16, f16};
            break;
        case SynapseEvidenceExceptGeluFp32:
            policy = (SynapsePrecisionPolicy){f16, f16, f16, f16, f16, f32, f16, f16};
            break;
        case SynapseEvidenceExceptAttentionFp32:
            policy = (SynapsePrecisionPolicy){f16, f16, f16, f16, f16, f16, f32, f32};
            break;
        case SynapseEvidenceExceptIoFp32:
            policy = (SynapsePrecisionPolicy){f32, f16, f16, f16, f16, f16, f16, f16};
            break;
        case SynapseEvidenceWeightsF16ComputeF32:
            policy.weight_type = f16;
            break;
        case SynapseEvidenceWeightsF32ActivationsF16:
            policy = (SynapsePrecisionPolicy){f32, f32, f16, f32, f32, f32, f32, f32};
            break;
        case SynapseEvidenceF16Fp32MatmulIslands:
            policy = (SynapsePrecisionPolicy){f16, f16, f16, f32, f32, f32, f32, f32};
            break;
        case SynapseEvidenceFp32:
        default:
            break;
    }
    return policy;
}

static NSString *synapse_mps_plan_key(uint64_t m, uint64_t n, uint64_t k, int32_t dtype, int32_t b_is_row_major_nk) {
    return [NSString stringWithFormat:@"%llu:%llu:%llu:%d:%d",
                                      (unsigned long long)m,
                                      (unsigned long long)n,
                                      (unsigned long long)k,
                                      dtype,
                                      b_is_row_major_nk];
}

static NSString *synapse_mps_depthwise_plan_key(
    uint64_t rows,
    uint64_t channels,
    uint64_t kernel,
    int32_t dtype
) {
    return [NSString stringWithFormat:@"depthwise:%llu:%llu:%llu:%d",
                                      (unsigned long long)rows,
                                      (unsigned long long)channels,
                                      (unsigned long long)kernel,
                                      dtype];
}

static NSString *synapse_mps_encoder_plan_key(
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float layer_norm_eps,
    int32_t variant
) {
    return [NSString stringWithFormat:@"encoder:%llu:%llu:%llu:%llu:%llu:%llu:%.9g:%d",
                                      (unsigned long long)batch,
                                      (unsigned long long)seq,
                                      (unsigned long long)hidden,
                                      (unsigned long long)heads,
                                      (unsigned long long)intermediate,
                                      (unsigned long long)layer_count,
                                      (double)layer_norm_eps,
                                      variant];
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
    [plan->executable_feed_tensors release];
    [plan->executable release];
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

static MPSGraphTensor *synapse_mps_placeholder(MPSGraph *graph, MPSShape *shape, MPSDataType data_type, NSString *name) {
    return [[graph placeholderWithShape:shape dataType:data_type name:name] retain];
}

static MPSGraphTensor *synapse_mps_scalar(MPSGraph *graph, double value, MPSDataType data_type) {
    return [graph constantWithScalar:value dataType:data_type];
}

static MPSGraphTensor *synapse_mps_cast(MPSGraph *graph, MPSGraphTensor *tensor, MPSDataType data_type) {
    return tensor.dataType == data_type ? tensor : [graph castTensor:tensor toType:data_type name:nil];
}

static MPSGraphTensor *synapse_mps_linear(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *weight,
    MPSGraphTensor *bias,
    MPSDataType compute_type,
    MPSDataType output_type
) {
    input = synapse_mps_cast(graph, input, compute_type);
    weight = synapse_mps_cast(graph, weight, compute_type);
    bias = synapse_mps_cast(graph, bias, compute_type);
    MPSGraphTensor *weight_t = [graph transposeTensor:weight dimension:0 withDimension:1 name:nil];
    MPSGraphTensor *product = [graph matrixMultiplicationWithPrimaryTensor:input
                                                            secondaryTensor:weight_t
                                                                       name:nil];
    MPSGraphTensor *output = [graph additionWithPrimaryTensor:product secondaryTensor:bias name:nil];
    return synapse_mps_cast(graph, output, output_type);
}

static MPSGraphTensor *synapse_mps_layer_norm(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *weight,
    MPSGraphTensor *bias,
    uint64_t rows,
    float epsilon,
    MPSDataType compute_type,
    MPSDataType output_type
) {
    NSArray<NSNumber *> *axes = @[ @1 ];
    MPSShape *mean_shape = @[ @(rows), @1 ];
    MPSGraphTensor *norm_input = synapse_mps_cast(graph, input, compute_type);
    MPSGraphTensor *norm_weight = synapse_mps_cast(graph, weight, compute_type);
    MPSGraphTensor *norm_bias = synapse_mps_cast(graph, bias, compute_type);
    MPSGraphTensor *mean = [graph meanOfTensor:norm_input axes:axes name:nil];
    MPSGraphTensor *variance = [graph varianceOfTensor:norm_input meanTensor:mean axes:axes name:nil];
    mean = [graph reshapeTensor:mean withShape:mean_shape name:nil];
    variance = [graph reshapeTensor:variance withShape:mean_shape name:nil];
    MPSGraphTensor *centered = [graph subtractionWithPrimaryTensor:norm_input secondaryTensor:mean name:nil];
    MPSGraphTensor *eps = synapse_mps_scalar(graph, (double)epsilon, compute_type);
    MPSGraphTensor *variance_eps = [graph additionWithPrimaryTensor:variance secondaryTensor:eps name:nil];
    MPSGraphTensor *std = [graph squareRootWithTensor:variance_eps name:nil];
    MPSGraphTensor *normalized = [graph divisionWithPrimaryTensor:centered secondaryTensor:std name:nil];
    MPSGraphTensor *scaled = [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:norm_weight name:nil];
    MPSGraphTensor *output = [graph additionWithPrimaryTensor:scaled secondaryTensor:norm_bias name:nil];
    return synapse_mps_cast(graph, output, output_type);
}

static MPSGraphTensor *synapse_mps_gelu(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSDataType compute_type,
    MPSDataType output_type
) {
    MPSGraphTensor *gelu_input = synapse_mps_cast(graph, input, compute_type);
    MPSGraphTensor *inv_sqrt2 = synapse_mps_scalar(graph, 0.70710678118654752440, compute_type);
    MPSGraphTensor *one = synapse_mps_scalar(graph, 1.0, compute_type);
    MPSGraphTensor *half = synapse_mps_scalar(graph, 0.5, compute_type);
    MPSGraphTensor *scaled = [graph multiplicationWithPrimaryTensor:gelu_input secondaryTensor:inv_sqrt2 name:nil];
    MPSGraphTensor *erf = [graph erfWithTensor:scaled name:nil];
    MPSGraphTensor *one_plus_erf = [graph additionWithPrimaryTensor:one secondaryTensor:erf name:nil];
    MPSGraphTensor *weighted = [graph multiplicationWithPrimaryTensor:gelu_input secondaryTensor:one_plus_erf name:nil];
    MPSGraphTensor *output = [graph multiplicationWithPrimaryTensor:weighted secondaryTensor:half name:nil];
    return synapse_mps_cast(graph, output, output_type);
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
    int32_t dtype,
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
    MPSDataType graph_data_type = synapse_mps_data_type(dtype);
    plan->a_shape = [@[ @(rows), @(inner) ] retain];
    plan->b_shape = [b_is_row_major_nk ? @[ @(cols), @(inner) ] : @[ @(inner), @(cols) ] retain];
    plan->graph = [[MPSGraph alloc] init];
    plan->graph.options = MPSGraphOptionsSynchronizeResults;
    plan->a_tensor = synapse_mps_placeholder(plan->graph, plan->a_shape, graph_data_type, @"a");
    plan->b_tensor = synapse_mps_placeholder(plan->graph, plan->b_shape, graph_data_type, @"b");
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

static SynapseMpsPlan *synapse_mps_depthwise_plan_new(
    uint64_t rows,
    uint64_t channels,
    uint64_t kernel,
    int32_t dtype
) {
    SynapseMpsPlan *plan = (SynapseMpsPlan *)calloc(1, sizeof(SynapseMpsPlan));
    if (plan == NULL) {
        synapse_mps_set_c_error("failed to allocate MPSGraph depthwise convolution plan");
        return NULL;
    }

    MPSDataType graph_data_type = synapse_mps_data_type(dtype);
    plan->a_shape = [@[ @(rows), @(channels), @(kernel) ] retain];
    plan->b_shape = [@[ @1, @(channels), @(kernel) ] retain];
    plan->graph = [[MPSGraph alloc] init];
    plan->graph.options = MPSGraphOptionsSynchronizeResults;
    plan->a_tensor = synapse_mps_placeholder(plan->graph, plan->a_shape, graph_data_type, @"windows");
    plan->b_tensor = synapse_mps_placeholder(plan->graph, plan->b_shape, graph_data_type, @"weights");
    MPSGraphTensor *products = [plan->graph multiplicationWithPrimaryTensor:plan->a_tensor
                                                            secondaryTensor:plan->b_tensor
                                                                       name:@"depthwise_products"];
    plan->product_tensor = [[plan->graph reductionSumWithTensor:products
                                                          axes:@[ @2 ]
                                                          name:@"depthwise_output"] retain];
    if (plan->a_shape == nil || plan->b_shape == nil || plan->graph == nil ||
        plan->a_tensor == nil || plan->b_tensor == nil || plan->product_tensor == nil) {
        synapse_mps_plan_free(plan);
        synapse_mps_set_c_error("failed to create MPSGraph depthwise convolution plan");
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
    float layer_norm_eps,
    int32_t variant
) {
    SynapseMpsEncoderPlan *plan = (SynapseMpsEncoderPlan *)calloc(1, sizeof(SynapseMpsEncoderPlan));
    if (plan == NULL) {
        synapse_mps_set_c_error("failed to allocate MPSGraph encoder plan");
        return NULL;
    }

    const uint64_t rows = batch * seq;
    const uint64_t head_dim = hidden / heads;
    SynapsePrecisionPolicy policy = synapse_mps_precision_policy(variant);
    // MPSGraph exposes operand dtypes but no independent accumulation control.
    // Explicit casts make each evidence variant's arithmetic islands inspectable.
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

    plan->input_tensor = synapse_mps_placeholder(plan->graph, plan->hidden_shape, policy.io_type, @"encoder_input");
    plan->mask_tensor = synapse_mps_placeholder(plan->graph, plan->mask_shape, MPSDataTypeFloat32, @"attention_mask_additive");
    MPSGraphTensor *x = [plan->graph reshapeTensor:plan->input_tensor withShape:plan->hidden_2d_shape name:nil];
    x = synapse_mps_cast(plan->graph, x, policy.base_type);
    MPSShape *hidden_4d_shape = @[ @(batch), @(seq), @(heads), @(head_dim) ];

    for (uint64_t layer_index = 0; layer_index < layer_count; layer_index++) {
        SynapseMpsEncoderLayerTensors *layer = &plan->layers[layer_index];
        NSString *prefix = [NSString stringWithFormat:@"layer_%llu", (unsigned long long)layer_index];
        layer->query_weight = synapse_mps_placeholder(plan->graph, plan->hidden_hidden_weight_shape, policy.weight_type, [prefix stringByAppendingString:@"_query_weight"]);
        layer->query_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, policy.weight_type, [prefix stringByAppendingString:@"_query_bias"]);
        layer->key_weight = synapse_mps_placeholder(plan->graph, plan->hidden_hidden_weight_shape, policy.weight_type, [prefix stringByAppendingString:@"_key_weight"]);
        layer->key_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, policy.weight_type, [prefix stringByAppendingString:@"_key_bias"]);
        layer->value_weight = synapse_mps_placeholder(plan->graph, plan->hidden_hidden_weight_shape, policy.weight_type, [prefix stringByAppendingString:@"_value_weight"]);
        layer->value_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, policy.weight_type, [prefix stringByAppendingString:@"_value_bias"]);
        layer->attention_output_weight = synapse_mps_placeholder(plan->graph, plan->hidden_hidden_weight_shape, policy.weight_type, [prefix stringByAppendingString:@"_attention_output_weight"]);
        layer->attention_output_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, policy.weight_type, [prefix stringByAppendingString:@"_attention_output_bias"]);
        layer->attention_ln_weight = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, policy.weight_type, [prefix stringByAppendingString:@"_attention_ln_weight"]);
        layer->attention_ln_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, policy.weight_type, [prefix stringByAppendingString:@"_attention_ln_bias"]);
        layer->intermediate_weight = synapse_mps_placeholder(plan->graph, plan->intermediate_hidden_weight_shape, policy.weight_type, [prefix stringByAppendingString:@"_intermediate_weight"]);
        layer->intermediate_bias = synapse_mps_placeholder(plan->graph, plan->intermediate_bias_shape, policy.weight_type, [prefix stringByAppendingString:@"_intermediate_bias"]);
        layer->output_weight = synapse_mps_placeholder(plan->graph, plan->hidden_intermediate_weight_shape, policy.weight_type, [prefix stringByAppendingString:@"_output_weight"]);
        layer->output_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, policy.weight_type, [prefix stringByAppendingString:@"_output_bias"]);
        layer->output_ln_weight = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, policy.weight_type, [prefix stringByAppendingString:@"_output_ln_weight"]);
        layer->output_ln_bias = synapse_mps_placeholder(plan->graph, plan->hidden_bias_shape, policy.weight_type, [prefix stringByAppendingString:@"_output_ln_bias"]);
        if (!synapse_mps_encoder_layer_tensors_valid(layer)) {
            synapse_mps_encoder_plan_free(plan);
            synapse_mps_set_c_error("failed to create MPSGraph encoder placeholders");
            return NULL;
        }

        MPSGraphTensor *attention_residual = x;
        MPSGraphTensor *q = synapse_mps_linear(plan->graph, x, layer->query_weight, layer->query_bias, policy.linear_type, policy.base_type);
        MPSGraphTensor *k = synapse_mps_linear(plan->graph, x, layer->key_weight, layer->key_bias, policy.linear_type, policy.base_type);
        MPSGraphTensor *v = synapse_mps_linear(plan->graph, x, layer->value_weight, layer->value_bias, policy.linear_type, policy.base_type);
        q = [plan->graph reshapeTensor:q withShape:hidden_4d_shape name:nil];
        k = [plan->graph reshapeTensor:k withShape:hidden_4d_shape name:nil];
        v = [plan->graph reshapeTensor:v withShape:hidden_4d_shape name:nil];
        q = [plan->graph transposeTensor:q permutation:@[ @0, @2, @1, @3 ] name:nil];
        k = [plan->graph transposeTensor:k permutation:@[ @0, @2, @1, @3 ] name:nil];
        v = [plan->graph transposeTensor:v permutation:@[ @0, @2, @1, @3 ] name:nil];
        q = synapse_mps_cast(plan->graph, q, policy.attention_type);
        k = synapse_mps_cast(plan->graph, k, policy.attention_type);
        v = synapse_mps_cast(plan->graph, v, policy.attention_type);
        k = [plan->graph transposeTensor:k dimension:2 withDimension:3 name:nil];
        MPSGraphTensor *scores = [plan->graph matrixMultiplicationWithPrimaryTensor:q secondaryTensor:k name:nil];
        scores = synapse_mps_cast(plan->graph, scores, policy.softmax_type);
        MPSGraphTensor *scale = synapse_mps_scalar(plan->graph, 1.0 / sqrt((double)head_dim), policy.softmax_type);
        MPSGraphTensor *mask = synapse_mps_cast(plan->graph, plan->mask_tensor, policy.softmax_type);
        scores = [plan->graph multiplicationWithPrimaryTensor:scores secondaryTensor:scale name:nil];
        scores = [plan->graph additionWithPrimaryTensor:scores secondaryTensor:mask name:nil];
        scores = [plan->graph softMaxWithTensor:scores axis:3 name:nil];
        scores = synapse_mps_cast(plan->graph, scores, policy.attention_type);
        MPSGraphTensor *context = [plan->graph matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v name:nil];
        context = synapse_mps_cast(plan->graph, context, policy.base_type);
        context = [plan->graph transposeTensor:context permutation:@[ @0, @2, @1, @3 ] name:nil];
        context = [plan->graph reshapeTensor:context withShape:plan->hidden_2d_shape name:nil];

        MPSGraphTensor *attention_out = synapse_mps_linear(plan->graph, context, layer->attention_output_weight, layer->attention_output_bias, policy.linear_type, policy.base_type);
        attention_out = [plan->graph additionWithPrimaryTensor:attention_out secondaryTensor:attention_residual name:nil];
        x = synapse_mps_layer_norm(plan->graph, attention_out, layer->attention_ln_weight, layer->attention_ln_bias, rows, layer_norm_eps, policy.layernorm_type, policy.base_type);

        MPSGraphTensor *ffn_residual = x;
        MPSGraphTensor *intermediate_out = synapse_mps_linear(plan->graph, x, layer->intermediate_weight, layer->intermediate_bias, policy.linear_type, policy.base_type);
        intermediate_out = synapse_mps_gelu(plan->graph, intermediate_out, policy.gelu_type, policy.base_type);
        MPSGraphTensor *output = synapse_mps_linear(plan->graph, intermediate_out, layer->output_weight, layer->output_bias, policy.linear_type, policy.base_type);
        output = [plan->graph additionWithPrimaryTensor:output secondaryTensor:ffn_residual name:nil];
        x = synapse_mps_layer_norm(plan->graph, output, layer->output_ln_weight, layer->output_ln_bias, rows, layer_norm_eps, policy.layernorm_type, policy.base_type);
    }

    x = synapse_mps_cast(plan->graph, x, policy.io_type);
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
    int32_t dtype,
    int32_t b_is_row_major_nk
) {
    NSString *key = synapse_mps_plan_key(m, n, k, dtype, b_is_row_major_nk);
    SynapseMpsPlan *cached = synapse_mps_cached_plan(&context->runtime, key);
    if (cached != NULL) return cached;

    SynapseMpsPlan *plan = synapse_mps_plan_new(m, n, k, dtype, b_is_row_major_nk);
    if (plan == NULL) {
        return NULL;
    }
    synapse_mps_cache_plan(&context->runtime, key, plan);
    return plan;
}

static SynapseMpsPlan *synapse_mps_get_depthwise_plan(
    SynapseMpsContext *context,
    uint64_t rows,
    uint64_t channels,
    uint64_t kernel,
    int32_t dtype
) {
    NSString *key = synapse_mps_depthwise_plan_key(rows, channels, kernel, dtype);
    SynapseMpsPlan *cached = synapse_mps_cached_plan(&context->runtime, key);
    if (cached != NULL) return cached;

    SynapseMpsPlan *plan = synapse_mps_depthwise_plan_new(rows, channels, kernel, dtype);
    if (plan == NULL) {
        return NULL;
    }
    synapse_mps_cache_plan(&context->runtime, key, plan);
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
    float layer_norm_eps,
    int32_t variant
) {
    NSString *key = synapse_mps_encoder_plan_key(batch, seq, hidden, heads, intermediate, layer_count, layer_norm_eps, variant);
    NSValue *cached = [context->encoder_plans objectForKey:key];
    if (cached != nil) {
        return (SynapseMpsEncoderPlan *)cached.pointerValue;
    }

    SynapseMpsEncoderPlan *plan = synapse_mps_encoder_plan_new(batch, seq, hidden, heads, intermediate, layer_count, layer_norm_eps, variant);
    if (plan == NULL) {
        return NULL;
    }
    [context->encoder_plans setObject:[NSValue valueWithPointer:plan] forKey:key];
    return plan;
}

// This pointer-keyed cache accepts only model-owned parameters and their persistent f16 mirrors.
// Dynamic activations and masks are copied into uncached buffers for each encoder invocation.
static id<MTLBuffer> synapse_mps_get_cached_buffer(
    SynapseMpsContext *context,
    const void *values,
    NSUInteger byte_count
) {
    id<MTLBuffer> buffer = synapse_mps_cached_static_buffer(&context->runtime, values, byte_count);
    if (buffer == nil) synapse_mps_set_c_error("encoder feed received a null, empty, or unallocatable static tensor");
    return buffer;
}

static BOOL synapse_minilm_add_feed(
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    MPSGraphTensor *tensor,
    MPSShape *shape,
    id<MTLBuffer> buffer,
    MPSDataType data_type
) {
    if (!synapse_mps_add_feed(feeds, tensor, shape, buffer, data_type)) {
        synapse_mps_set_c_error("encoder feed is missing a tensor, shape, or Metal buffer");
        return NO;
    }
    return YES;
}

static BOOL synapse_mps_add_cached_feed(
    SynapseMpsContext *context,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    MPSGraphTensor *tensor,
    MPSShape *shape,
    const void *values,
    NSUInteger element_count,
    MPSDataType data_type
) {
    id<MTLBuffer> buffer = synapse_mps_get_cached_buffer(context, values, element_count * synapse_mps_type_size(data_type));
    if (buffer == nil) {
        return NO;
    }
    return synapse_minilm_add_feed(feeds, tensor, shape, buffer, data_type);
}

static BOOL synapse_mps_add_shaped_type(
    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds,
    MPSGraphTensor *tensor,
    MPSShape *shape,
    MPSDataType data_type
) {
    MPSGraphShapedType *shaped_type = [[MPSGraphShapedType alloc] initWithShape:shape dataType:data_type];
    if (shaped_type == nil) {
        synapse_mps_set_c_error("failed to create MPSGraph shaped feed type");
        return NO;
    }
    [feeds setObject:shaped_type forKey:tensor];
    [shaped_type release];
    return YES;
}

static NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *synapse_mps_encoder_shaped_feeds(
    SynapseMpsEncoderPlan *plan,
    SynapsePrecisionPolicy policy
) {
    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds =
        [[NSMutableDictionary alloc] initWithCapacity:(NSUInteger)(2 + plan->layer_count * 16)];
    if (feeds == nil ||
        !synapse_mps_add_shaped_type(feeds, plan->input_tensor, plan->hidden_shape, policy.io_type) ||
        !synapse_mps_add_shaped_type(feeds, plan->mask_tensor, plan->mask_shape, MPSDataTypeFloat32)) {
        [feeds release];
        return nil;
    }

    for (uint64_t layer_index = 0; layer_index < plan->layer_count; layer_index++) {
        SynapseMpsEncoderLayerTensors *layer = &plan->layers[layer_index];
        if (!synapse_mps_add_shaped_type(feeds, layer->query_weight, plan->hidden_hidden_weight_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->query_bias, plan->hidden_bias_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->key_weight, plan->hidden_hidden_weight_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->key_bias, plan->hidden_bias_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->value_weight, plan->hidden_hidden_weight_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->value_bias, plan->hidden_bias_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->attention_output_weight, plan->hidden_hidden_weight_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->attention_output_bias, plan->hidden_bias_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->attention_ln_weight, plan->hidden_bias_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->attention_ln_bias, plan->hidden_bias_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->intermediate_weight, plan->intermediate_hidden_weight_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->intermediate_bias, plan->intermediate_bias_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->output_weight, plan->hidden_intermediate_weight_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->output_bias, plan->hidden_bias_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->output_ln_weight, plan->hidden_bias_shape, policy.weight_type) ||
            !synapse_mps_add_shaped_type(feeds, layer->output_ln_bias, plan->hidden_bias_shape, policy.weight_type)) {
            [feeds release];
            return nil;
        }
    }
    return feeds;
}

static NSArray<MPSGraphTensorData *> *synapse_minilm_executable_inputs(
    SynapseMpsEncoderPlan *plan,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds
) {
    NSMutableArray<MPSGraphTensorData *> *inputs =
        [NSMutableArray arrayWithCapacity:plan->executable_feed_tensors.count];
    for (MPSGraphTensor *tensor in plan->executable_feed_tensors) {
        MPSGraphTensorData *data = [feeds objectForKey:tensor];
        if (data == nil) {
            for (MPSGraphTensor *feed_tensor in feeds) {
                if ([feed_tensor.operation.name isEqualToString:tensor.operation.name]) {
                    data = [feeds objectForKey:feed_tensor];
                    break;
                }
            }
        }
        if (data == nil) {
            synapse_mps_set_ns_error([NSString stringWithFormat:@"no encoder feed for executable input %@", tensor.operation.name]);
            return nil;
        }
        [inputs addObject:data];
    }
    return inputs;
}

int32_t synapse_mps_prepare_encoder(
    void *raw_context,
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t heads,
    uint64_t intermediate,
    uint64_t layer_count,
    float layer_norm_eps,
    int32_t variant,
    int32_t optimization_level,
    const char *package_path,
    int32_t load_package,
    int32_t append_package,
    double *prepare_wall_s,
    double *specialize_wall_s,
    double *serialize_wall_s
) {
    @autoreleasepool {
        @try {
            synapse_mps_clear_error();
            SynapseMpsContext *context = (SynapseMpsContext *)raw_context;
            if (context == NULL || context->runtime.device == nil || context->runtime.queue == nil) {
                synapse_mps_set_c_error("Metal context is not initialized");
                return -1;
            }
            if (batch == 0 || seq == 0 || hidden == 0 || heads == 0 || intermediate == 0 || layer_count == 0 ||
                prepare_wall_s == NULL || specialize_wall_s == NULL || serialize_wall_s == NULL) {
                synapse_mps_set_c_error("encoder preparation received invalid dimensions or timing pointers");
                return -2;
            }
            *prepare_wall_s = 0.0;
            *specialize_wall_s = 0.0;
            *serialize_wall_s = 0.0;

            SynapseMpsEncoderPlan *plan = synapse_mps_get_encoder_plan(
                context, batch, seq, hidden, heads, intermediate, layer_count, layer_norm_eps, variant
            );
            if (plan == NULL) {
                return -3;
            }
            if (plan->executable != nil) {
                return 0;
            }

            if (load_package && package_path == NULL) {
                synapse_mps_set_c_error("package load requires a package path");
                return -4;
            }
            NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds = nil;
            if (!load_package) {
                SynapsePrecisionPolicy policy = synapse_mps_precision_policy(variant);
                feeds = synapse_mps_encoder_shaped_feeds(plan, policy);
                if (feeds == nil) return -5;
            }
            plan->executable = synapse_mps_prepare_executable(
                plan->graph,
                context->runtime.device,
                plan->output_tensor,
                feeds,
                optimization_level,
                package_path,
                load_package != 0,
                append_package != 0,
                prepare_wall_s,
                specialize_wall_s,
                serialize_wall_s,
                &plan->executable_feed_tensors
            );
            [feeds release];
            if (plan->executable == nil) {
                synapse_mps_set_c_error(load_package ? "MPSGraph package returned no executable" : "MPSGraph compile returned no executable");
                return -6;
            }
            if (plan->executable_feed_tensors.count != (NSUInteger)(2 + layer_count * 16)) {
                synapse_mps_set_ns_error([NSString stringWithFormat:@"unexpected executable feed count %llu, expected %llu",
                                                                    (unsigned long long)plan->executable_feed_tensors.count,
                                                                    (unsigned long long)(2 + layer_count * 16)]);
                return -7;
            }
            return 0;
        } @catch (NSException *exception) {
            synapse_mps_set_ns_error(exception.reason);
            return -100;
        }
    }
}

void *synapse_mps_context_new(void) {
    @autoreleasepool {
        @try {
            synapse_mps_clear_error();
            SynapseMpsContext *context = (SynapseMpsContext *)calloc(1, sizeof(SynapseMpsContext));
            if (context == NULL || !synapse_mps_runtime_init(&context->runtime)) {
                free(context);
                synapse_mps_set_c_error("failed to initialize Metal context");
                return NULL;
            }
            context->encoder_plans = [[NSMutableDictionary alloc] init];
            if (context->encoder_plans == nil) {
                synapse_mps_runtime_release(&context->runtime, NULL);
                free(context);
                synapse_mps_set_c_error("failed to allocate Metal encoder plan cache");
                return NULL;
            }
            return context;
        } @catch (NSException *exception) {
            synapse_mps_set_ns_error(exception.reason);
            return NULL;
        }
    }
}

int32_t synapse_mps_configure_capture(void *raw_context, const char *path) {
    SynapseMpsContext *context = (SynapseMpsContext *)raw_context;
    if (context == NULL || path == NULL) {
        synapse_mps_set_c_error("capture configuration received a null context or path");
        return -1;
    }
    [context->capture_path release];
    context->capture_path = [[NSString alloc] initWithUTF8String:path];
    context->capture_pending = context->capture_path != nil;
    return context->capture_pending ? 0 : -2;
}

int32_t synapse_mps_configure_graph_dump(void *raw_context, const char *path) {
    SynapseMpsContext *context = (SynapseMpsContext *)raw_context;
    if (context == NULL || path == NULL) {
        synapse_mps_set_c_error("graph dump configuration received a null context or path");
        return -1;
    }
    [context->graph_dump_path release];
    context->graph_dump_path = [[NSString alloc] initWithUTF8String:path];
    context->graph_dump_pending = context->graph_dump_path != nil;
    return context->graph_dump_pending ? 0 : -2;
}

static void synapse_mps_plan_free_erased(void *plan) {
    synapse_mps_plan_free((SynapseMpsPlan *)plan);
}

void synapse_mps_context_free(void *raw_context) {
    if (raw_context == NULL) return;
    SynapseMpsContext *context = (SynapseMpsContext *)raw_context;
    for (NSValue *value in [context->encoder_plans allValues]) {
        synapse_mps_encoder_plan_free((SynapseMpsEncoderPlan *)value.pointerValue);
    }
    [context->capture_path release];
    [context->graph_dump_path release];
    [context->encoder_plans release];
    synapse_mps_runtime_release(&context->runtime, synapse_mps_plan_free_erased);
    free(context);
}

int32_t synapse_mps_matmul(
    void *raw_context,
    uint64_t m,
    uint64_t n,
    uint64_t k,
    const void *a,
    const void *b,
    int32_t dtype,
    int32_t b_is_row_major_nk,
    void *c,
    int32_t cache_rhs
) {
    @autoreleasepool {
        @try {
            synapse_mps_clear_error();
            SynapseMpsContext *context = (SynapseMpsContext *)raw_context;
            if (context == NULL || context->runtime.device == nil || context->runtime.queue == nil) {
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
            const NSUInteger element_size = synapse_mps_dtype_size(dtype);
            MPSDataType graph_data_type = synapse_mps_data_type(dtype);
            const NSUInteger a_count = rows * inner;
            const NSUInteger b_count = b_is_row_major_nk ? cols * inner : inner * cols;
            const NSUInteger a_bytes = a_count * element_size;
            const NSUInteger b_bytes = b_count * element_size;

            SynapseMpsPlan *plan = synapse_mps_get_plan(context, m, n, k, dtype, b_is_row_major_nk);
            if (plan == NULL) {
                return -5;
            }

            id<MTLBuffer> a_buffer = [context->runtime.device newBufferWithBytes:a
                                                                   length:a_bytes
                                                                  options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_buffer = nil;
            BOOL release_b_buffer = NO;
            if (cache_rhs) {
                b_buffer = synapse_mps_get_cached_buffer(context, b, b_bytes);
            } else {
                b_buffer = [context->runtime.device newBufferWithBytes:b
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
                                                                              dataType:graph_data_type];
            MPSGraphTensorData *b_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:b_buffer
                                                                                 shape:plan->b_shape
                                                                              dataType:graph_data_type];
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
                [plan->graph runWithMTLCommandQueue:context->runtime.queue
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

int32_t synapse_mps_depthwise_conv1d(
    void *raw_context,
    uint64_t rows,
    uint64_t channels,
    uint64_t kernel,
    const void *windows,
    const void *weights,
    int32_t dtype,
    void *output
) {
    @autoreleasepool {
        @try {
            synapse_mps_clear_error();
            SynapseMpsContext *context = (SynapseMpsContext *)raw_context;
            if (context == NULL || context->runtime.device == nil || context->runtime.queue == nil) {
                synapse_mps_set_c_error("Metal context is not initialized");
                return -1;
            }
            if (windows == NULL || weights == NULL || output == NULL) {
                synapse_mps_set_c_error("depthwise convolution received a null data pointer");
                return -2;
            }
            if (rows == 0 || channels == 0 || kernel == 0) {
                synapse_mps_set_c_error("depthwise convolution dimensions must be non-zero");
                return -3;
            }
            if (rows > NSUIntegerMax || channels > NSUIntegerMax || kernel > NSUIntegerMax) {
                synapse_mps_set_c_error("depthwise convolution dimensions exceed NSUIntegerMax");
                return -4;
            }

            const NSUInteger element_size = synapse_mps_dtype_size(dtype);
            const NSUInteger windows_bytes = (NSUInteger)rows * (NSUInteger)channels * (NSUInteger)kernel * element_size;
            const NSUInteger weights_bytes = (NSUInteger)channels * (NSUInteger)kernel * element_size;
            MPSDataType graph_data_type = synapse_mps_data_type(dtype);
            SynapseMpsPlan *plan = synapse_mps_get_depthwise_plan(context, rows, channels, kernel, dtype);
            if (plan == NULL) {
                return -5;
            }

            id<MTLBuffer> windows_buffer = synapse_mps_uncached_buffer(&context->runtime, windows, windows_bytes);
            id<MTLBuffer> weights_buffer = synapse_mps_cached_static_buffer(&context->runtime, weights, weights_bytes);
            if (windows_buffer == nil || weights_buffer == nil) {
                [windows_buffer release];
                synapse_mps_set_c_error("failed to allocate depthwise convolution input buffers");
                return -6;
            }
            MPSGraphTensorData *windows_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:windows_buffer
                                                                                        shape:plan->a_shape
                                                                                     dataType:graph_data_type];
            MPSGraphTensorData *weights_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:weights_buffer
                                                                                        shape:plan->b_shape
                                                                                     dataType:graph_data_type];
            if (windows_data == nil || weights_data == nil) {
                [windows_data release];
                [weights_data release];
                [windows_buffer release];
                synapse_mps_set_c_error("failed to wrap depthwise convolution input buffers");
                return -7;
            }

            NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
                [plan->graph runWithMTLCommandQueue:context->runtime.queue
                                             feeds:@{ plan->a_tensor: windows_data, plan->b_tensor: weights_data }
                                     targetTensors:@[ plan->product_tensor ]
                                  targetOperations:nil];
            MPSGraphTensorData *output_data = [results objectForKey:plan->product_tensor];
            MPSNDArray *output_array = [output_data mpsndarray];
            if (output_array == nil) {
                [windows_data release];
                [weights_data release];
                [windows_buffer release];
                synapse_mps_set_c_error("MPSGraph did not return depthwise convolution output");
                return -8;
            }
            [output_array readBytes:output strideBytes:NULL];
            [windows_data release];
            [weights_data release];
            [windows_buffer release];
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
    int32_t variant,
    const void *input,
    const float *additive_mask,
    void *output,
    const SynapseMpsEncoderLayerParams *layers
) {
    @autoreleasepool {
        @try {
            synapse_mps_clear_error();
            SynapseMpsContext *context = (SynapseMpsContext *)raw_context;
            if (context == NULL || context->runtime.device == nil || context->runtime.queue == nil) {
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
            SynapsePrecisionPolicy policy = synapse_mps_precision_policy(variant);
            const NSUInteger input_bytes = hidden_count * synapse_mps_type_size(policy.io_type);
            const NSUInteger mask_bytes = mask_count * sizeof(float);

            SynapseMpsEncoderPlan *plan = synapse_mps_get_encoder_plan(
                context,
                batch,
                seq,
                hidden,
                heads,
                intermediate,
                layer_count,
                layer_norm_eps,
                variant
            );
            if (plan == NULL) {
                return -6;
            }
            if (context->graph_dump_pending) {
                NSError *dump_error = nil;
                BOOL wrote = [plan->graph.description writeToFile:context->graph_dump_path
                                                      atomically:YES
                                                        encoding:NSUTF8StringEncoding
                                                           error:&dump_error];
                if (!wrote) {
                    synapse_mps_set_ns_error([NSString stringWithFormat:@"graph dump failed: %@", dump_error]);
                    return -13;
                }
                context->graph_dump_pending = NO;
            }

            id<MTLBuffer> input_buffer = [context->runtime.device newBufferWithBytes:input
                                                                       length:input_bytes
                                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> mask_buffer = [context->runtime.device newBufferWithBytes:additive_mask
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
            if (!synapse_mps_add_feed(feeds, plan->input_tensor, plan->hidden_shape, input_buffer, policy.io_type) ||
                !synapse_mps_add_feed(feeds, plan->mask_tensor, plan->mask_shape, mask_buffer, MPSDataTypeFloat32)) {
                [feeds release];
                [input_buffer release];
                [mask_buffer release];
                return -9;
            }

            for (uint64_t layer_index = 0; layer_index < layer_count; layer_index++) {
                const SynapseMpsEncoderLayerParams *params = &layers[layer_index];
                SynapseMpsEncoderLayerTensors *tensors = &plan->layers[layer_index];
                if (!synapse_mps_add_cached_feed(context, feeds, tensors->query_weight, plan->hidden_hidden_weight_shape, params->query_weight, hidden_hidden_count, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->query_bias, plan->hidden_bias_shape, params->query_bias, (NSUInteger)hidden, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->key_weight, plan->hidden_hidden_weight_shape, params->key_weight, hidden_hidden_count, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->key_bias, plan->hidden_bias_shape, params->key_bias, (NSUInteger)hidden, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->value_weight, plan->hidden_hidden_weight_shape, params->value_weight, hidden_hidden_count, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->value_bias, plan->hidden_bias_shape, params->value_bias, (NSUInteger)hidden, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->attention_output_weight, plan->hidden_hidden_weight_shape, params->attention_output_weight, hidden_hidden_count, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->attention_output_bias, plan->hidden_bias_shape, params->attention_output_bias, (NSUInteger)hidden, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->attention_ln_weight, plan->hidden_bias_shape, params->attention_ln_weight, (NSUInteger)hidden, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->attention_ln_bias, plan->hidden_bias_shape, params->attention_ln_bias, (NSUInteger)hidden, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->intermediate_weight, plan->intermediate_hidden_weight_shape, params->intermediate_weight, intermediate_hidden_count, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->intermediate_bias, plan->intermediate_bias_shape, params->intermediate_bias, (NSUInteger)intermediate, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->output_weight, plan->hidden_intermediate_weight_shape, params->output_weight, hidden_intermediate_count, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->output_bias, plan->hidden_bias_shape, params->output_bias, (NSUInteger)hidden, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->output_ln_weight, plan->hidden_bias_shape, params->output_ln_weight, (NSUInteger)hidden, policy.weight_type) ||
                    !synapse_mps_add_cached_feed(context, feeds, tensors->output_ln_bias, plan->hidden_bias_shape, params->output_ln_bias, (NSUInteger)hidden, policy.weight_type)) {
                    [feeds release];
                    [input_buffer release];
                    [mask_buffer release];
                    return -10;
                }
            }

            BOOL capturing = NO;
            if (context->capture_pending) {
                MTLCaptureManager *manager = [MTLCaptureManager sharedCaptureManager];
                MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
                descriptor.captureObject = context->runtime.queue;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:context->capture_path];
                [[NSFileManager defaultManager] removeItemAtURL:descriptor.outputURL error:nil];
                NSError *capture_error = nil;
                capturing = [manager startCaptureWithDescriptor:descriptor error:&capture_error];
                [descriptor release];
                if (!capturing) {
                    synapse_mps_set_ns_error([NSString stringWithFormat:@"Metal capture start failed: %@", capture_error]);
                    [feeds release];
                    [input_buffer release];
                    [mask_buffer release];
                    return -14;
                }
                context->capture_pending = NO;
            }
            MPSGraphTensorData *output_data = nil;
            @try {
                if (plan->executable != nil) {
                    NSArray<MPSGraphTensorData *> *inputs = synapse_minilm_executable_inputs(plan, feeds);
                    if (inputs == nil) {
                        [feeds release];
                        [input_buffer release];
                        [mask_buffer release];
                        return -15;
                    }
                    NSArray<MPSGraphTensorData *> *results =
                        [plan->executable runWithMTLCommandQueue:context->runtime.queue
                                                    inputsArray:inputs
                                                   resultsArray:nil
                                            executionDescriptor:nil];
                    output_data = results.firstObject;
                } else {
                    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
                        [plan->graph runWithMTLCommandQueue:context->runtime.queue
                                                     feeds:feeds
                                             targetTensors:@[ plan->output_tensor ]
                                          targetOperations:nil];
                    output_data = [results objectForKey:plan->output_tensor];
                }
            } @finally {
                if (capturing) {
                    [[MTLCaptureManager sharedCaptureManager] stopCapture];
                }
            }
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
