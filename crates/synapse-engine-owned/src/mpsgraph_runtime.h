#ifndef SYNAPSE_MPSGRAPH_RUNTIME_H
#define SYNAPSE_MPSGRAPH_RUNTIME_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MPSGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphExecutable.h>

typedef struct SynapseMpsRuntimeContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString *, NSValue *> *plans;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *static_buffers;
} SynapseMpsRuntimeContext;

typedef void (*SynapseMpsPlanFree)(void *plan);

BOOL synapse_mps_runtime_init(SynapseMpsRuntimeContext *context);
void synapse_mps_runtime_release(SynapseMpsRuntimeContext *context, SynapseMpsPlanFree free_plan);
void *synapse_mps_cached_plan(SynapseMpsRuntimeContext *context, NSString *key);
void synapse_mps_cache_plan(SynapseMpsRuntimeContext *context, NSString *key, void *plan);

id<MTLBuffer> synapse_mps_uncached_buffer(
    SynapseMpsRuntimeContext *context,
    const void *values,
    NSUInteger byte_count
);
id<MTLBuffer> synapse_mps_cached_static_buffer(
    SynapseMpsRuntimeContext *context,
    const void *values,
    NSUInteger byte_count
);
BOOL synapse_mps_add_feed(
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    MPSGraphTensor *tensor,
    MPSShape *shape,
    id<MTLBuffer> buffer,
    MPSDataType data_type
);

MPSGraphExecutable *synapse_mps_prepare_executable(
    MPSGraph *graph,
    id<MTLDevice> device,
    MPSGraphTensor *output,
    NSDictionary<MPSGraphTensor *, MPSGraphShapedType *> *shaped_feeds,
    int32_t optimization_level,
    const char *package_path,
    BOOL load_package,
    BOOL append_package,
    double *prepare_wall_s,
    double *specialize_wall_s,
    double *serialize_wall_s,
    NSArray<MPSGraphTensor *> **feed_tensors
);
MPSGraphExecutable *synapse_mps_explicit_executable(
    MPSGraph *graph,
    id<MTLDevice> device,
    MPSGraphTensor *output,
    int32_t optimization_level,
    const char *package_path,
    NSArray<MPSGraphTensor *> **feed_tensors
);
NSArray<MPSGraphTensorData *> *synapse_mps_executable_inputs(
    NSArray<MPSGraphTensor *> *feed_tensors,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds
);

#endif
