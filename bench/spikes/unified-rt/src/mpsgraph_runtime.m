#import "mpsgraph_runtime.h"

BOOL synapse_mps_runtime_init(SynapseMpsRuntimeContext *context) {
    if (context == NULL) return NO;
    context->device = MTLCreateSystemDefaultDevice();
    context->queue = [context->device newCommandQueue];
    context->plans = [[NSMutableDictionary alloc] init];
    context->static_buffers = [[NSMutableDictionary alloc] init];
    if (context->device == nil || context->queue == nil ||
        context->plans == nil || context->static_buffers == nil) {
        synapse_mps_runtime_release(context, NULL);
        return NO;
    }
    return YES;
}

void synapse_mps_runtime_release(SynapseMpsRuntimeContext *context, SynapseMpsPlanFree free_plan) {
    if (context == NULL) return;
    if (free_plan != NULL) {
        for (NSValue *value in context->plans.allValues) free_plan(value.pointerValue);
    }
    [context->static_buffers release];
    [context->plans release];
    [context->queue release];
    [context->device release];
    context->static_buffers = nil;
    context->plans = nil;
    context->queue = nil;
    context->device = nil;
}

void *synapse_mps_cached_plan(SynapseMpsRuntimeContext *context, NSString *key) {
    return [[context->plans objectForKey:key] pointerValue];
}

void synapse_mps_cache_plan(SynapseMpsRuntimeContext *context, NSString *key, void *plan) {
    [context->plans setObject:[NSValue valueWithPointer:plan] forKey:key];
}

id<MTLBuffer> synapse_mps_uncached_buffer(
    SynapseMpsRuntimeContext *context,
    const void *values,
    NSUInteger byte_count
) {
    if (context == NULL || values == NULL || byte_count == 0) return nil;
    return [context->device newBufferWithBytes:values
                                       length:byte_count
                                      options:MTLResourceStorageModeShared];
}

id<MTLBuffer> synapse_mps_cached_static_buffer(
    SynapseMpsRuntimeContext *context,
    const void *values,
    NSUInteger byte_count
) {
    if (context == NULL || values == NULL || byte_count == 0) return nil;
    NSString *key = [NSString stringWithFormat:@"%p:%llu", values, (unsigned long long)byte_count];
    id<MTLBuffer> buffer = [context->static_buffers objectForKey:key];
    if (buffer == nil) {
        buffer = synapse_mps_uncached_buffer(context, values, byte_count);
        if (buffer != nil) {
            [context->static_buffers setObject:buffer forKey:key];
            [buffer release];
            buffer = [context->static_buffers objectForKey:key];
        }
    }
    return buffer;
}

BOOL synapse_mps_add_feed(
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    MPSGraphTensor *tensor,
    MPSShape *shape,
    id<MTLBuffer> buffer,
    MPSDataType data_type
) {
    if (feeds == nil || tensor == nil || shape == nil || buffer == nil) return NO;
    MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                      shape:shape
                                                                   dataType:data_type];
    if (data == nil) return NO;
    [feeds setObject:data forKey:tensor];
    [data release];
    return YES;
}

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
) {
    if (graph == nil || device == nil || output == nil || feed_tensors == NULL ||
        (load_package && package_path == NULL) || (!load_package && shaped_feeds == nil)) return nil;
    double ignored_prepare = 0.0;
    double ignored_specialize = 0.0;
    double ignored_serialize = 0.0;
    prepare_wall_s = prepare_wall_s ?: &ignored_prepare;
    specialize_wall_s = specialize_wall_s ?: &ignored_specialize;
    serialize_wall_s = serialize_wall_s ?: &ignored_serialize;
    *prepare_wall_s = 0.0;
    *specialize_wall_s = 0.0;
    *serialize_wall_s = 0.0;

    MPSGraphCompilationDescriptor *descriptor = [[MPSGraphCompilationDescriptor alloc] init];
    descriptor.optimizationLevel = optimization_level == 0
        ? MPSGraphOptimizationLevel0
        : MPSGraphOptimizationLevel1;
    descriptor.waitForCompilationCompletion = YES;
    NSTimeInterval started = [NSDate timeIntervalSinceReferenceDate];
    MPSGraphExecutable *executable = nil;
    NSString *path = package_path == NULL ? nil : [NSString stringWithUTF8String:package_path];
    if (load_package) {
        executable = [[MPSGraphExecutable alloc]
            initWithMPSGraphPackageAtURL:[NSURL fileURLWithPath:path]
            compilationDescriptor:descriptor];
    } else {
        MPSGraphDevice *graph_device = [MPSGraphDevice deviceWithMTLDevice:device];
        executable = [[graph compileWithDevice:graph_device
                                         feeds:shaped_feeds
                                 targetTensors:@[ output ]
                              targetOperations:nil
                         compilationDescriptor:descriptor] retain];
        if (executable != nil) {
            NSMutableArray<MPSGraphType *> *input_types =
                [NSMutableArray arrayWithCapacity:executable.feedTensors.count];
            for (MPSGraphTensor *tensor in executable.feedTensors) {
                MPSGraphShapedType *type = [shaped_feeds objectForKey:tensor];
                if (type == nil) {
                    for (MPSGraphTensor *placeholder in shaped_feeds) {
                        if ([placeholder.operation.name isEqualToString:tensor.operation.name]) {
                            type = [shaped_feeds objectForKey:placeholder];
                            break;
                        }
                    }
                }
                if (type == nil) {
                    [descriptor release];
                    [executable release];
                    return nil;
                }
                [input_types addObject:type];
            }
            NSTimeInterval specialize_started = [NSDate timeIntervalSinceReferenceDate];
            [executable specializeWithDevice:graph_device
                                  inputTypes:input_types
                       compilationDescriptor:descriptor];
            *specialize_wall_s = [NSDate timeIntervalSinceReferenceDate] - specialize_started;
        }
    }
    *prepare_wall_s = [NSDate timeIntervalSinceReferenceDate] - started;
    [descriptor release];
    if (executable == nil) return nil;
    executable.options = MPSGraphOptionsSynchronizeResults;
    NSArray<MPSGraphTensor *> *inputs = executable.feedTensors;
    *feed_tensors = [(inputs.count > 0 ? inputs : graph.placeholderTensors) retain];
    if (!load_package && path != nil) {
        MPSGraphExecutableSerializationDescriptor *serialization =
            [[MPSGraphExecutableSerializationDescriptor alloc] init];
        serialization.append = append_package;
        started = [NSDate timeIntervalSinceReferenceDate];
        [executable serializeToMPSGraphPackageAtURL:[NSURL fileURLWithPath:path]
                                         descriptor:serialization];
        *serialize_wall_s = [NSDate timeIntervalSinceReferenceDate] - started;
        [serialization release];
    }
    return executable;
}

MPSGraphExecutable *synapse_mps_explicit_executable(
    MPSGraph *graph,
    id<MTLDevice> device,
    MPSGraphTensor *output,
    int32_t optimization_level,
    const char *package_path,
    NSArray<MPSGraphTensor *> **feed_tensors
) {
    MPSGraphCompilationDescriptor *descriptor = [[MPSGraphCompilationDescriptor alloc] init];
    descriptor.optimizationLevel = optimization_level == 1
        ? MPSGraphOptimizationLevel1
        : MPSGraphOptimizationLevel0;
    descriptor.waitForCompilationCompletion = YES;
    MPSGraphExecutable *executable = nil;
    NSString *path = package_path == NULL ? nil : [NSString stringWithUTF8String:package_path];
    if (path != nil && [[NSFileManager defaultManager] fileExistsAtPath:path]) {
        executable = [[MPSGraphExecutable alloc]
            initWithMPSGraphPackageAtURL:[NSURL fileURLWithPath:path]
            compilationDescriptor:descriptor];
    } else {
        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds =
            [[NSMutableDictionary alloc] initWithCapacity:graph.placeholderTensors.count];
        for (MPSGraphTensor *tensor in graph.placeholderTensors) {
            MPSGraphShapedType *type = [[MPSGraphShapedType alloc]
                initWithShape:tensor.shape
                dataType:tensor.dataType];
            [feeds setObject:type forKey:tensor];
            [type release];
        }
        MPSGraphDevice *graph_device = [MPSGraphDevice deviceWithMTLDevice:device];
        executable = [[graph compileWithDevice:graph_device
                                         feeds:feeds
                                 targetTensors:@[ output ]
                              targetOperations:nil
                         compilationDescriptor:descriptor] retain];
        NSMutableArray<MPSGraphType *> *input_types =
            [NSMutableArray arrayWithCapacity:executable.feedTensors.count];
        for (MPSGraphTensor *tensor in executable.feedTensors) {
            MPSGraphShapedType *type = [feeds objectForKey:tensor];
            if (type == nil) {
                for (MPSGraphTensor *placeholder in feeds) {
                    if ([placeholder.operation.name isEqualToString:tensor.operation.name]) {
                        type = [feeds objectForKey:placeholder];
                        break;
                    }
                }
            }
            if (type == nil) {
                [feeds release];
                [descriptor release];
                [executable release];
                return nil;
            }
            [input_types addObject:type];
        }
        [executable specializeWithDevice:graph_device
                              inputTypes:input_types
                   compilationDescriptor:descriptor];
        [feeds release];
        if (executable != nil && path != nil) {
            MPSGraphExecutableSerializationDescriptor *serialization =
                [[MPSGraphExecutableSerializationDescriptor alloc] init];
            serialization.append = NO;
            [executable serializeToMPSGraphPackageAtURL:[NSURL fileURLWithPath:path]
                                             descriptor:serialization];
            [serialization release];
        }
    }
    [descriptor release];
    if (executable != nil) {
        executable.options = MPSGraphOptionsSynchronizeResults;
        NSArray<MPSGraphTensor *> *inputs = executable.feedTensors;
        *feed_tensors = [(inputs.count > 0 ? inputs : graph.placeholderTensors) retain];
    }
    return executable;
}

NSArray<MPSGraphTensorData *> *synapse_mps_executable_inputs(
    NSArray<MPSGraphTensor *> *feed_tensors,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds
) {
    NSMutableArray<MPSGraphTensorData *> *inputs =
        [NSMutableArray arrayWithCapacity:feed_tensors.count];
    for (MPSGraphTensor *tensor in feed_tensors) {
        MPSGraphTensorData *data = [feeds objectForKey:tensor];
        if (data == nil) {
            for (MPSGraphTensor *placeholder in feeds) {
                if ([placeholder.operation.name isEqualToString:tensor.operation.name]) {
                    data = [feeds objectForKey:placeholder];
                    break;
                }
            }
        }
        if (data == nil) return nil;
        [inputs addObject:data];
    }
    return inputs;
}
