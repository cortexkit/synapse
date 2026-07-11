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

static char qwen3_error[1024];

typedef struct Qwen3LayerParams {
    const float *input_norm;
    const float *post_attention_norm;
    const float *q_weight;
    const float *q_norm;
    const float *k_weight;
    const float *k_norm;
    const float *v_weight;
    const float *o_weight;
    const float *gate_weight;
    const float *up_weight;
    const float *down_weight;
} Qwen3LayerParams;

typedef struct Qwen3LayerTensors {
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
} Qwen3LayerTensors;

typedef struct Qwen3Plan {
    MPSGraph *graph;
    MPSShape *hidden_shape;
    MPSShape *mask_shape;
    MPSShape *rope_shape;
    MPSShape *hidden_vector_shape;
    MPSShape *head_vector_shape;
    MPSShape *q_weight_shape;
    MPSShape *kv_weight_shape;
    MPSShape *o_weight_shape;
    MPSShape *mlp_weight_shape;
    MPSShape *down_weight_shape;
    MPSGraphTensor *input;
    MPSGraphTensor *mask;
    MPSGraphTensor *rope_cos;
    MPSGraphTensor *rope_sin;
    MPSGraphTensor *final_norm;
    MPSGraphTensor *output;
    Qwen3LayerTensors *layers;
    uint64_t layer_count;
} Qwen3Plan;

typedef struct Qwen3Context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString *, NSValue *> *plans;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *weights;
} Qwen3Context;

static void set_error(NSString *message) {
    snprintf(qwen3_error, sizeof(qwen3_error), "%s", message.UTF8String ?: "unknown Qwen3 MPSGraph error");
}

const char *synapse_qwen3_last_error(void) {
    return qwen3_error;
}

static MPSGraphTensor *placeholder(MPSGraph *graph, MPSShape *shape, NSString *name) {
    return [[graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:name] retain];
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
    MPSGraphTensor *square = [graph multiplicationWithPrimaryTensor:input secondaryTensor:input name:nil];
    MPSGraphTensor *mean = [graph meanOfTensor:square axes:@[ @(axis) ] name:nil];
    mean = [graph reshapeTensor:mean withShape:reduced_shape name:nil];
    MPSGraphTensor *eps = [graph constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
    MPSGraphTensor *denominator = [graph squareRootWithTensor:[graph additionWithPrimaryTensor:mean secondaryTensor:eps name:nil] name:nil];
    MPSGraphTensor *normalized = [graph divisionWithPrimaryTensor:input secondaryTensor:denominator name:nil];
    return [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:weight name:nil];
}

static MPSGraphTensor *rope(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *cosine,
    MPSGraphTensor *sine,
    uint64_t head_dim
) {
    const NSUInteger half = (NSUInteger)(head_dim / 2);
    MPSGraphTensor *first = [graph sliceTensor:input dimension:3 start:0 length:half name:nil];
    MPSGraphTensor *second = [graph sliceTensor:input dimension:3 start:half length:half name:nil];
    MPSGraphTensor *negative_second = [graph negativeWithTensor:second name:nil];
    MPSGraphTensor *rotated = [graph concatTensors:@[ negative_second, first ] dimension:3 name:nil];
    MPSGraphTensor *scaled = [graph multiplicationWithPrimaryTensor:input secondaryTensor:cosine name:nil];
    MPSGraphTensor *rotated_scaled = [graph multiplicationWithPrimaryTensor:rotated secondaryTensor:sine name:nil];
    return [graph additionWithPrimaryTensor:scaled secondaryTensor:rotated_scaled name:nil];
}

static MPSGraphTensor *repeat_kv(
    MPSGraph *graph,
    MPSGraphTensor *input,
    uint64_t batch,
    uint64_t kv_heads,
    uint64_t groups,
    uint64_t seq,
    uint64_t head_dim
) {
    MPSGraphTensor *grouped = [graph reshapeTensor:input
                                        withShape:@[ @(batch), @(kv_heads), @1, @(seq), @(head_dim) ]
                                             name:nil];
    MPSGraphTensor *broadcast = [graph broadcastTensor:grouped
                                              toShape:@[ @(batch), @(kv_heads), @(groups), @(seq), @(head_dim) ]
                                                 name:nil];
    return [graph reshapeTensor:broadcast
                      withShape:@[ @(batch), @(kv_heads * groups), @(seq), @(head_dim) ]
                           name:nil];
}

static void release_layer(Qwen3LayerTensors *layer) {
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

static void free_plan(Qwen3Plan *plan) {
    if (plan == NULL) return;
    if (plan->layers != NULL) {
        for (uint64_t i = 0; i < plan->layer_count; ++i) release_layer(&plan->layers[i]);
        free(plan->layers);
    }
    [plan->output release];
    [plan->final_norm release];
    [plan->rope_sin release];
    [plan->rope_cos release];
    [plan->mask release];
    [plan->input release];
    [plan->down_weight_shape release];
    [plan->mlp_weight_shape release];
    [plan->o_weight_shape release];
    [plan->kv_weight_shape release];
    [plan->q_weight_shape release];
    [plan->head_vector_shape release];
    [plan->hidden_vector_shape release];
    [plan->rope_shape release];
    [plan->mask_shape release];
    [plan->hidden_shape release];
    [plan->graph release];
    free(plan);
}

static Qwen3Plan *new_plan(
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    float epsilon
) {
    Qwen3Plan *plan = calloc(1, sizeof(Qwen3Plan));
    if (plan == NULL) {
        set_error(@"failed to allocate Qwen3 graph plan");
        return NULL;
    }
    const uint64_t rows = batch * seq;
    const uint64_t q_width = query_heads * head_dim;
    const uint64_t kv_width = kv_heads * head_dim;
    const uint64_t groups = query_heads / kv_heads;
    plan->layer_count = layer_count;
    plan->hidden_shape = [@[ @(batch), @(seq), @(hidden) ] retain];
    plan->mask_shape = [@[ @(batch), @1, @(seq), @(seq) ] retain];
    plan->rope_shape = [@[ @1, @1, @(seq), @(head_dim) ] retain];
    plan->hidden_vector_shape = [@[ @(hidden) ] retain];
    plan->head_vector_shape = [@[ @(head_dim) ] retain];
    plan->q_weight_shape = [@[ @(q_width), @(hidden) ] retain];
    plan->kv_weight_shape = [@[ @(kv_width), @(hidden) ] retain];
    plan->o_weight_shape = [@[ @(hidden), @(q_width) ] retain];
    plan->mlp_weight_shape = [@[ @(intermediate), @(hidden) ] retain];
    plan->down_weight_shape = [@[ @(hidden), @(intermediate) ] retain];
    plan->layers = calloc((size_t)layer_count, sizeof(Qwen3LayerTensors));
    plan->graph = [[MPSGraph alloc] init];
    if (plan->layers == NULL || plan->graph == nil) {
        free_plan(plan);
        set_error(@"failed to allocate Qwen3 graph objects");
        return NULL;
    }
    plan->graph.options = MPSGraphOptionsSynchronizeResults;
    plan->input = placeholder(plan->graph, plan->hidden_shape, @"qwen3_input");
    plan->mask = placeholder(plan->graph, plan->mask_shape, @"qwen3_causal_padding_mask");
    plan->rope_cos = placeholder(plan->graph, plan->rope_shape, @"qwen3_rope_cos");
    plan->rope_sin = placeholder(plan->graph, plan->rope_shape, @"qwen3_rope_sin");
    plan->final_norm = placeholder(plan->graph, plan->hidden_vector_shape, @"qwen3_final_norm");
    MPSGraphTensor *x = [plan->graph reshapeTensor:plan->input withShape:@[ @(rows), @(hidden) ] name:nil];

    for (uint64_t index = 0; index < layer_count; ++index) {
        Qwen3LayerTensors *layer = &plan->layers[index];
        NSString *prefix = [NSString stringWithFormat:@"qwen3_layer_%llu", (unsigned long long)index];
#define PH(field, shape) layer->field = placeholder(plan->graph, shape, [prefix stringByAppendingFormat:@"_%s", #field])
        PH(input_norm, plan->hidden_vector_shape);
        PH(post_attention_norm, plan->hidden_vector_shape);
        PH(q_weight, plan->q_weight_shape);
        PH(q_norm, plan->head_vector_shape);
        PH(k_weight, plan->kv_weight_shape);
        PH(k_norm, plan->head_vector_shape);
        PH(v_weight, plan->kv_weight_shape);
        PH(o_weight, plan->o_weight_shape);
        PH(gate_weight, plan->mlp_weight_shape);
        PH(up_weight, plan->mlp_weight_shape);
        PH(down_weight, plan->down_weight_shape);
#undef PH
        MPSGraphTensor *attention_residual = x;
        MPSGraphTensor *normalized = rms_norm(plan->graph, x, layer->input_norm, 1, @[ @(rows), @1 ], epsilon);
        MPSGraphTensor *q = linear(plan->graph, normalized, layer->q_weight);
        MPSGraphTensor *k = linear(plan->graph, normalized, layer->k_weight);
        MPSGraphTensor *v = linear(plan->graph, normalized, layer->v_weight);
        q = [plan->graph reshapeTensor:q withShape:@[ @(batch), @(seq), @(query_heads), @(head_dim) ] name:nil];
        k = [plan->graph reshapeTensor:k withShape:@[ @(batch), @(seq), @(kv_heads), @(head_dim) ] name:nil];
        v = [plan->graph reshapeTensor:v withShape:@[ @(batch), @(seq), @(kv_heads), @(head_dim) ] name:nil];
        q = rms_norm(plan->graph, q, layer->q_norm, 3, @[ @(batch), @(seq), @(query_heads), @1 ], epsilon);
        k = rms_norm(plan->graph, k, layer->k_norm, 3, @[ @(batch), @(seq), @(kv_heads), @1 ], epsilon);
        q = [plan->graph transposeTensor:q permutation:@[ @0, @2, @1, @3 ] name:nil];
        k = [plan->graph transposeTensor:k permutation:@[ @0, @2, @1, @3 ] name:nil];
        v = [plan->graph transposeTensor:v permutation:@[ @0, @2, @1, @3 ] name:nil];
        q = rope(plan->graph, q, plan->rope_cos, plan->rope_sin, head_dim);
        k = rope(plan->graph, k, plan->rope_cos, plan->rope_sin, head_dim);
        k = repeat_kv(plan->graph, k, batch, kv_heads, groups, seq, head_dim);
        v = repeat_kv(plan->graph, v, batch, kv_heads, groups, seq, head_dim);
        MPSGraphTensor *k_transposed = [plan->graph transposeTensor:k dimension:2 withDimension:3 name:nil];
        MPSGraphTensor *scores = [plan->graph matrixMultiplicationWithPrimaryTensor:q secondaryTensor:k_transposed name:nil];
        MPSGraphTensor *scale = [plan->graph constantWithScalar:1.0 / sqrt((double)head_dim) dataType:MPSDataTypeFloat32];
        scores = [plan->graph multiplicationWithPrimaryTensor:scores secondaryTensor:scale name:nil];
        scores = [plan->graph additionWithPrimaryTensor:scores secondaryTensor:plan->mask name:nil];
        scores = [plan->graph softMaxWithTensor:scores axis:3 name:nil];
        MPSGraphTensor *context = [plan->graph matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v name:nil];
        context = [plan->graph transposeTensor:context permutation:@[ @0, @2, @1, @3 ] name:nil];
        context = [plan->graph reshapeTensor:context withShape:@[ @(rows), @(q_width) ] name:nil];
        x = [plan->graph additionWithPrimaryTensor:attention_residual
                                   secondaryTensor:linear(plan->graph, context, layer->o_weight)
                                              name:nil];
        MPSGraphTensor *mlp_residual = x;
        normalized = rms_norm(plan->graph, x, layer->post_attention_norm, 1, @[ @(rows), @1 ], epsilon);
        MPSGraphTensor *gate = linear(plan->graph, normalized, layer->gate_weight);
        MPSGraphTensor *sigmoid = [plan->graph sigmoidWithTensor:gate name:nil];
        gate = [plan->graph multiplicationWithPrimaryTensor:gate secondaryTensor:sigmoid name:nil];
        MPSGraphTensor *up = linear(plan->graph, normalized, layer->up_weight);
        MPSGraphTensor *gated = [plan->graph multiplicationWithPrimaryTensor:gate secondaryTensor:up name:nil];
        x = [plan->graph additionWithPrimaryTensor:mlp_residual
                                   secondaryTensor:linear(plan->graph, gated, layer->down_weight)
                                              name:nil];
    }
    x = rms_norm(plan->graph, x, plan->final_norm, 1, @[ @(rows), @1 ], epsilon);
    plan->output = [[plan->graph reshapeTensor:x withShape:plan->hidden_shape name:@"qwen3_output"] retain];
    if (plan->input == nil || plan->mask == nil || plan->rope_cos == nil || plan->rope_sin == nil ||
        plan->final_norm == nil || plan->output == nil) {
        free_plan(plan);
        set_error(@"failed to construct Qwen3 MPSGraph");
        return NULL;
    }
    return plan;
}

static NSString *plan_key(uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t qh, uint64_t kvh,
                          uint64_t head_dim, uint64_t intermediate, uint64_t layers, float eps) {
    return [NSString stringWithFormat:@"%llu:%llu:%llu:%llu:%llu:%llu:%llu:%llu:%.9g",
            batch, seq, hidden, qh, kvh, head_dim, intermediate, layers, (double)eps];
}

static id<MTLBuffer> cached_buffer(Qwen3Context *context, const float *values, NSUInteger count) {
    NSString *key = [NSString stringWithFormat:@"%p:%llu", values, (unsigned long long)count];
    id<MTLBuffer> buffer = [context->weights objectForKey:key];
    if (buffer == nil) {
        buffer = [context->device newBufferWithBytes:values length:count * sizeof(float) options:MTLResourceStorageModeShared];
        if (buffer != nil) {
            [context->weights setObject:buffer forKey:key];
            [buffer release];
            buffer = [context->weights objectForKey:key];
        }
    }
    return buffer;
}

static BOOL add_feed(NSMutableDictionary *feeds, MPSGraphTensor *tensor, MPSShape *shape,
                     id<MTLBuffer> buffer) {
    if (tensor == nil || shape == nil || buffer == nil) return NO;
    MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                      shape:shape
                                                                   dataType:MPSDataTypeFloat32];
    if (data == nil) return NO;
    [feeds setObject:data forKey:tensor];
    [data release];
    return YES;
}

static BOOL add_weight(Qwen3Context *context, NSMutableDictionary *feeds, MPSGraphTensor *tensor,
                       MPSShape *shape, const float *values, NSUInteger count) {
    return add_feed(feeds, tensor, shape, cached_buffer(context, values, count));
}

void *synapse_qwen3_context_new(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (device == nil || queue == nil) {
            [queue release];
            [device release];
            set_error(@"no Metal device or command queue for Qwen3");
            return NULL;
        }
        Qwen3Context *context = calloc(1, sizeof(Qwen3Context));
        context->device = device;
        context->queue = queue;
        context->plans = [[NSMutableDictionary alloc] init];
        context->weights = [[NSMutableDictionary alloc] init];
        return context;
    }
}

void synapse_qwen3_context_free(void *raw) {
    if (raw == NULL) return;
    Qwen3Context *context = raw;
    for (NSValue *value in context->plans.allValues) free_plan(value.pointerValue);
    [context->weights release];
    [context->plans release];
    [context->queue release];
    [context->device release];
    free(context);
}

int32_t synapse_qwen3_forward(
    void *raw,
    uint64_t batch,
    uint64_t seq,
    uint64_t hidden,
    uint64_t query_heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    uint64_t intermediate,
    uint64_t layer_count,
    float epsilon,
    const float *input,
    const float *mask,
    const float *rope_cos,
    const float *rope_sin,
    const Qwen3LayerParams *params,
    const float *final_norm,
    float *output
) {
    @autoreleasepool {
        @try {
            qwen3_error[0] = '\0';
            Qwen3Context *context = raw;
            if (context == NULL || input == NULL || mask == NULL || params == NULL || output == NULL ||
                batch == 0 || seq == 0 || hidden == 0 || query_heads == 0 || kv_heads == 0 ||
                query_heads % kv_heads != 0 || head_dim % 2 != 0 || layer_count == 0) {
                set_error(@"invalid Qwen3 MPSGraph forward arguments");
                return -1;
            }
            NSString *key = plan_key(batch, seq, hidden, query_heads, kv_heads, head_dim, intermediate, layer_count, epsilon);
            Qwen3Plan *plan = [[context->plans objectForKey:key] pointerValue];
            if (plan == NULL) {
                plan = new_plan(batch, seq, hidden, query_heads, kv_heads, head_dim, intermediate, layer_count, epsilon);
                if (plan == NULL) return -2;
                [context->plans setObject:[NSValue valueWithPointer:plan] forKey:key];
            }
            const NSUInteger rows = (NSUInteger)(batch * seq);
            id<MTLBuffer> input_buffer = [context->device newBufferWithBytes:input length:rows * hidden * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> mask_buffer = [context->device newBufferWithBytes:mask length:batch * seq * seq * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> cos_buffer = [context->device newBufferWithBytes:rope_cos length:seq * head_dim * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> sin_buffer = [context->device newBufferWithBytes:rope_sin length:seq * head_dim * sizeof(float) options:MTLResourceStorageModeShared];
            NSMutableDictionary *feeds = [[NSMutableDictionary alloc] initWithCapacity:(NSUInteger)(5 + layer_count * 11)];
            if (!add_feed(feeds, plan->input, plan->hidden_shape, input_buffer) ||
                !add_feed(feeds, plan->mask, plan->mask_shape, mask_buffer) ||
                !add_feed(feeds, plan->rope_cos, plan->rope_shape, cos_buffer) ||
                !add_feed(feeds, plan->rope_sin, plan->rope_shape, sin_buffer) ||
                !add_weight(context, feeds, plan->final_norm, plan->hidden_vector_shape, final_norm, hidden)) {
                set_error(@"failed to feed Qwen3 graph inputs");
                [feeds release]; [input_buffer release]; [mask_buffer release]; [cos_buffer release]; [sin_buffer release];
                return -3;
            }
            const NSUInteger q_count = query_heads * head_dim * hidden;
            const NSUInteger kv_count = kv_heads * head_dim * hidden;
            const NSUInteger o_count = hidden * query_heads * head_dim;
            const NSUInteger mlp_count = intermediate * hidden;
            for (uint64_t i = 0; i < layer_count; ++i) {
                Qwen3LayerTensors *t = &plan->layers[i];
                const Qwen3LayerParams *p = &params[i];
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
                    !add_weight(context, feeds, t->down_weight, plan->down_weight_shape, p->down_weight, mlp_count)) {
                    set_error(@"failed to feed Qwen3 static weights");
                    [feeds release]; [input_buffer release]; [mask_buffer release]; [cos_buffer release]; [sin_buffer release];
                    return -4;
                }
            }
            NSDictionary *results = [plan->graph runWithMTLCommandQueue:context->queue
                                                                  feeds:feeds
                                                          targetTensors:@[ plan->output ]
                                                       targetOperations:nil];
            MPSNDArray *array = [[results objectForKey:plan->output] mpsndarray];
            if (array == nil) {
                set_error(@"Qwen3 MPSGraph returned no output");
                [feeds release]; [input_buffer release]; [mask_buffer release]; [cos_buffer release]; [sin_buffer release];
                return -5;
            }
            [array readBytes:output strideBytes:NULL];
            [feeds release]; [input_buffer release]; [mask_buffer release]; [cos_buffer release]; [sin_buffer release];
            return 0;
        } @catch (NSException *exception) {
            set_error(exception.reason);
            return -100;
        }
    }
}
