// Native Metal harness for the LFM2 decode step kernels.
//
// This file exposes a small C ABI over the lfm2_decode_metal_step.metal kernels
// so the Rust driver (lfm2_decode_metal_step.rs) can drive them. It is additive:
// it shares no mutable state with the Qwen3 step harness and never touches the
// Qwen3 kernels, so the Qwen3 byte-identity fixtures are unaffected.
//
// The harness models the LFM2 conv-cache model directly: each convolution layer
// owns a device-resident rolling cache buffer (kernel_size rows of `hidden`
// channels). A decode step advances that cache in place on the GPU and reads
// back only the gated convolution output for the newest position, exactly like
// the KV-cache attention step reads back one context row. The cache buffers
// persist for the life of the context so a sequence of steps rolls the window
// forward without host round trips; cache_read/cache_write exist for tests and
// to leave room for a future rewind/rollback without changing the ABI.
//
// Memory management follows the rest of this directory: manual retain/release
// (the build does not enable ARC) and MTLResourceStorageModeShared buffers so
// the CPU can feed activations and read results through buffer.contents once a
// command buffer has completed.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static char lfm2_step_error[1024];

static void set_error(NSString *message) {
    if (message == nil) {
        message = @"unknown LFM2 Metal step error";
    }
    const char *utf8 = [message UTF8String];
    if (utf8 == NULL) {
        lfm2_step_error[0] = '\0';
        return;
    }
    strncpy(lfm2_step_error, utf8, sizeof(lfm2_step_error) - 1);
    lfm2_step_error[sizeof(lfm2_step_error) - 1] = '\0';
}

const char *synapse_lfm2_metal_step_last_error(void) {
    return lfm2_step_error;
}

// Mirrors Lfm2ConvStepParams in the Metal source. Two uint32 fields keep the
// struct layout identical on both sides of the FFI boundary (8 bytes, 4-byte
// aligned), which is what setBytes:index: binds into the kernel's constant
// address space.
typedef struct Lfm2ConvStepParams {
    uint32_t hidden;
    uint32_t kernel_size;
} Lfm2ConvStepParams;

typedef struct Lfm2ConvLayerBuffers {
    id<MTLBuffer> cache;       // [kernel_size * hidden] f32, zeroed, rolling window.
    id<MTLBuffer> conv_weight; // [hidden * kernel_size] f32, static depthwise taps.
} Lfm2ConvLayerBuffers;

typedef struct Lfm2MetalStepContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> conv_step;
    uint64_t hidden;
    uint64_t kernel_size;
    Lfm2ConvLayerBuffers *layers;
    uint64_t layer_count;
    // Reusable per-step scratch sized to `hidden` so a step performs no
    // allocation: the host feeds product/gate into these and reads `out`.
    id<MTLBuffer> product;
    id<MTLBuffer> gate;
    id<MTLBuffer> out;
} Lfm2MetalStepContext;

static id<MTLBuffer> shared_buffer(id<MTLDevice> device, const void *bytes, NSUInteger length) {
    if (bytes == NULL) {
        return [device newBufferWithLength:length
                                   options:MTLResourceStorageModeShared];
    }
    return [device newBufferWithBytes:bytes
                               length:length
                              options:MTLResourceStorageModeShared];
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (result == nil) {
        set_error(error.localizedDescription
                      ?: [NSString stringWithFormat:@"failed to compile Metal kernel %@", name]);
    }
    return result;
}

void *synapse_lfm2_metal_step_context_new(
    uint64_t hidden,
    uint64_t kernel_size,
    const char *metallib_path
) {
    @autoreleasepool {
        if (hidden == 0 || kernel_size == 0 || hidden > UINT32_MAX || kernel_size > UINT32_MAX ||
            metallib_path == NULL) {
            set_error(@"invalid LFM2 Metal step dimensions or metallib path");
            return NULL;
        }
        Lfm2MetalStepContext *context = calloc(1, sizeof(*context));
        if (context == NULL) {
            set_error(@"failed to allocate LFM2 Metal step context");
            return NULL;
        }
        context->device = MTLCreateSystemDefaultDevice();
        if (context->device == nil) {
            set_error(@"no Metal device for LFM2 Metal step");
            free(context);
            return NULL;
        }
        context->queue = [context->device newCommandQueue];
        NSError *error = nil;
        NSURL *library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
        context->library = [context->device newLibraryWithURL:library_url error:&error];
        if (context->queue == nil || context->library == nil) {
            set_error(error.localizedDescription ?: @"failed to load LFM2 Metal step metallib");
            [context->queue release];
            [context->library release];
            [context->device release];
            free(context);
            return NULL;
        }
        context->conv_step = pipeline(context->device, context->library, @"lfm2_conv_step");
        if (context->conv_step == nil) {
            [context->library release];
            [context->queue release];
            [context->device release];
            free(context);
            return NULL;
        }
        context->hidden = hidden;
        context->kernel_size = kernel_size;
        return context;
    }
}

int32_t synapse_lfm2_metal_step_prepare(
    void *raw,
    uint64_t conv_layer_count,
    const float *const *conv_weights
) {
    @autoreleasepool {
        @try {
            Lfm2MetalStepContext *context = raw;
            if (context == NULL || conv_layer_count == 0 || conv_weights == NULL) {
                set_error(@"invalid LFM2 Metal step preparation arguments");
                return -1;
            }
            if (context->layers != NULL) {
                set_error(@"LFM2 Metal step context already prepared");
                return -2;
            }
            const NSUInteger hidden = (NSUInteger)context->hidden;
            const NSUInteger kernel_size = (NSUInteger)context->kernel_size;
            const NSUInteger cache_bytes = (NSUInteger)(kernel_size * hidden * sizeof(float));
            const NSUInteger weight_bytes = (NSUInteger)(hidden * kernel_size * sizeof(float));

            context->layers = calloc((size_t)conv_layer_count, sizeof(*context->layers));
            if (context->layers == NULL) {
                set_error(@"failed to allocate LFM2 conv layer table");
                return -3;
            }
            context->layer_count = conv_layer_count;
            for (uint64_t layer = 0; layer < conv_layer_count; ++layer) {
                if (conv_weights[layer] == NULL) {
                    set_error([NSString stringWithFormat:@"LFM2 conv layer %llu weight is null", layer]);
                    return -4;
                }
                // Zero-initialised rolling cache: matches empty_decode_cache, which
                // starts every conv state at all zeros.
                context->layers[layer].cache = shared_buffer(context->device, NULL, cache_bytes);
                context->layers[layer].conv_weight =
                    shared_buffer(context->device, conv_weights[layer], weight_bytes);
                if (context->layers[layer].cache == nil || context->layers[layer].conv_weight == nil) {
                    set_error(@"failed to allocate LFM2 conv layer buffers");
                    return -5;
                }
            }
            const NSUInteger scratch_bytes = (NSUInteger)(hidden * sizeof(float));
            context->product = shared_buffer(context->device, NULL, scratch_bytes);
            context->gate = shared_buffer(context->device, NULL, scratch_bytes);
            context->out = shared_buffer(context->device, NULL, scratch_bytes);
            if (context->product == nil || context->gate == nil || context->out == nil) {
                set_error(@"failed to allocate LFM2 conv step scratch buffers");
                return -6;
            }
            return 0;
        } @catch (NSException *exception) {
            set_error(exception.reason ?: @"LFM2 Metal step preparation raised");
            return -7;
        }
    }
}

// Run one conv decode step for `layer`: feed product/gate, advance the layer's
// device-resident cache in place, and read back out[hidden] = gate * conv(newest
// position). The cache stays on device between calls, so a sequence of steps
// rolls the window forward exactly as the CPU decode_conv advances its Vec state.
int32_t synapse_lfm2_conv_step(
    void *raw,
    uint64_t layer,
    const float *product,
    const float *gate,
    float *out
) {
    @autoreleasepool {
        @try {
            Lfm2MetalStepContext *context = raw;
            if (context == NULL || context->layers == NULL || layer >= context->layer_count ||
                product == NULL || gate == NULL || out == NULL) {
                set_error(@"invalid LFM2 conv step arguments");
                return -1;
            }
            const NSUInteger scratch_bytes = (NSUInteger)(context->hidden * sizeof(float));
            memcpy(context->product.contents, product, scratch_bytes);
            memcpy(context->gate.contents, gate, scratch_bytes);

            id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:context->conv_step];
            [encoder setBuffer:context->layers[layer].cache offset:0 atIndex:0];
            [encoder setBuffer:context->product offset:0 atIndex:1];
            [encoder setBuffer:context->gate offset:0 atIndex:2];
            [encoder setBuffer:context->layers[layer].conv_weight offset:0 atIndex:3];
            [encoder setBuffer:context->out offset:0 atIndex:4];
            Lfm2ConvStepParams params = {
                (uint32_t)context->hidden,
                (uint32_t)context->kernel_size,
            };
            [encoder setBytes:&params length:sizeof(params) atIndex:5];
            NSUInteger threads_per_group = context->conv_step.maxTotalThreadsPerThreadgroup;
            if (threads_per_group > context->hidden) {
                threads_per_group = (NSUInteger)context->hidden;
            }
            if (threads_per_group == 0) {
                threads_per_group = 1;
            }
            [encoder dispatchThreads:MTLSizeMake(context->hidden, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                set_error(command_buffer.error.localizedDescription ?: @"LFM2 conv step command failed");
                return -2;
            }
            memcpy(out, context->out.contents, scratch_bytes);
            return 0;
        } @catch (NSException *exception) {
            set_error(exception.reason ?: @"LFM2 conv step raised");
            return -3;
        }
    }
}

// Read the layer's current device-resident cache back to the host. Used by the
// exactness gate to confirm the rolling window matches the CPU state, and a
// building block for any future rewind/rollback.
int32_t synapse_lfm2_conv_cache_read(void *raw, uint64_t layer, float *host) {
    @autoreleasepool {
        Lfm2MetalStepContext *context = raw;
        if (context == NULL || context->layers == NULL || layer >= context->layer_count || host == NULL) {
            set_error(@"invalid LFM2 conv cache read arguments");
            return -1;
        }
        const NSUInteger cache_bytes =
            (NSUInteger)(context->kernel_size * context->hidden * sizeof(float));
        memcpy(host, context->layers[layer].cache.contents, cache_bytes);
        return 0;
    }
}

// Overwrite the layer's device-resident cache from the host. Lets tests seed a
// starting window and leaves a clean hook for future rewind without an ABI break.
int32_t synapse_lfm2_conv_cache_write(void *raw, uint64_t layer, const float *host) {
    @autoreleasepool {
        Lfm2MetalStepContext *context = raw;
        if (context == NULL || context->layers == NULL || layer >= context->layer_count || host == NULL) {
            set_error(@"invalid LFM2 conv cache write arguments");
            return -1;
        }
        const NSUInteger cache_bytes =
            (NSUInteger)(context->kernel_size * context->hidden * sizeof(float));
        memcpy(context->layers[layer].cache.contents, host, cache_bytes);
        return 0;
    }
}

void synapse_lfm2_metal_step_context_free(void *raw) {
    @autoreleasepool {
        Lfm2MetalStepContext *context = raw;
        if (context == NULL) {
            return;
        }
        if (context->layers != NULL) {
            for (uint64_t layer = 0; layer < context->layer_count; ++layer) {
                [context->layers[layer].cache release];
                [context->layers[layer].conv_weight release];
            }
            free(context->layers);
        }
        [context->product release];
        [context->gate release];
        [context->out release];
        [context->conv_step release];
        [context->library release];
        [context->queue release];
        [context->device release];
        free(context);
    }
}
