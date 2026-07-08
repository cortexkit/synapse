#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MPSGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphMatrixMultiplicationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMemoryOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorShapeOps.h>

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

typedef struct SynapseMpsContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString *, NSValue *> *plans;
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
            context->rhs_buffers = [[NSMutableDictionary alloc] init];
            if (context->plans == nil || context->rhs_buffers == nil) {
                [context->rhs_buffers release];
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
    [context->rhs_buffers release];
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
                NSString *rhs_key = synapse_mps_rhs_key(b, b_bytes);
                b_buffer = [context->rhs_buffers objectForKey:rhs_key];
                if (b_buffer == nil) {
                    b_buffer = [context->device newBufferWithBytes:b
                                                            length:b_bytes
                                                           options:MTLResourceStorageModeShared];
                    if (b_buffer != nil) {
                        [context->rhs_buffers setObject:b_buffer forKey:rhs_key];
                        [b_buffer release];
                        b_buffer = [context->rhs_buffers objectForKey:rhs_key];
                    }
                }
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
