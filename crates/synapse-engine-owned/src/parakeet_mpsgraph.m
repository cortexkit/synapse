#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <math.h>
#include <stdint.h>
#include <mach/mach.h>
#include <sys/resource.h>

uint64_t synapse_parakeet_resident_bytes(void) {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    return task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS ? info.resident_size : UINT64_MAX;
}
uint64_t synapse_parakeet_peak_resident_bytes(void) {
    struct rusage info;
    return getrusage(RUSAGE_SELF, &info) == 0 ? (uint64_t)info.ru_maxrss : UINT64_MAX;
}

// This bridge uses ARC; each model owns its weights, plans and in-flight tensors.
@interface PKPlan : NSObject
@property MPSGraph *graph;
@property MPSGraphTensor *output;
@property NSMutableDictionary<NSString *, MPSGraphTensor *> *inputs;
@property MPSGraphExecutable *executable;
@end
@implementation PKPlan
@end

@interface PKContext : NSObject
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property NSMutableDictionary<NSString *, MPSGraphTensorData *> *weights;
@property NSMutableDictionary<NSString *, PKPlan *> *plans;
@property MPSGraphTensorData *hidden;
@property MPSGraphTensorData *rowMask;
@property MPSGraphTensorData *pairMask;
@property MPSGraphTensorData *positions;
@property NSUInteger rung;
@property NSUInteger validMel;
@end
@implementation PKContext
@end

static _Thread_local char pkError[1024];
const char *synapse_parakeet_error(void) { return pkError; }
static int failure(NSException *e) {
    snprintf(pkError, sizeof(pkError), "%s", e.reason.UTF8String ?: "Parakeet graph failed");
    return -1;
}

void *synapse_parakeet_new(void) {
    @autoreleasepool {
        PKContext *c = [PKContext new];
        c.device = MTLCreateSystemDefaultDevice();
        c.queue = [c.device newCommandQueue];
        c.weights = [NSMutableDictionary new]; c.plans = [NSMutableDictionary new];
        return c.queue ? (__bridge_retained void *)c : NULL;
    }
}
void synapse_parakeet_free(void *raw) {
    if (raw) { PKContext *c = (__bridge_transfer PKContext *)raw; (void)c; }
}
static MPSGraphTensorData *data(PKContext *c, const float *values, MPSShape *shape) {
    NSUInteger count = 1;
    for (NSNumber *n in shape) count *= n.unsignedIntegerValue;
    id<MTLBuffer> buffer = [c.device newBufferWithBytes:values length:count * sizeof(float) options:MTLResourceStorageModeShared];
    if (!buffer) [NSException raise:@"Parakeet" format:@"Metal buffer allocation failed"];
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shape dataType:MPSDataTypeFloat32];
}
int32_t synapse_parakeet_weight(void *raw, const char *name, const float *values, const uint64_t *dims, uint64_t rank) {
    @autoreleasepool { @try {
        PKContext *c = (__bridge PKContext *)raw;
        NSMutableArray *shape = [NSMutableArray new];
        for (uint64_t i = 0; i < rank; i++) [shape addObject:@(dims[i])];
        c.weights[@(name)] = data(c, values, shape);
        return 0;
    } @catch(NSException *e) { return failure(e); } }
}

static MPSGraphTensor *ph(PKPlan *p, NSString *name, MPSShape *shape) {
    MPSGraphTensor *t = [p.graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:name];
    p.inputs[name] = t; return t;
}
static MPSGraphTensor *w(PKContext *c, PKPlan *p, NSString *name, MPSShape *shape) {
    NSString *actual = [name stringByReplacingOccurrencesOfString:@"layer." withString:@"encoder.layers.0."];
    if (![c.weights[actual].shape isEqualToArray:shape])
        [NSException raise:@"Parakeet" format:@"missing or invalid weight %@ (expected %@)", actual, shape];
    return ph(p, name, shape);
}
static MPSGraphTensor *add(MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) { return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil]; }
static MPSGraphTensor *mul(MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) { return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil]; }
static MPSGraphTensor *scale(MPSGraph *g, MPSGraphTensor *a, double s) { return mul(g, a, [g constantWithScalar:s dataType:MPSDataTypeFloat32]); }
static MPSGraphTensor *mm(MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) { return [g matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:nil]; }
static MPSGraphTensor *reshape(MPSGraph *g, MPSGraphTensor *x, MPSShape *shape) { return [g reshapeTensor:x withShape:shape name:nil]; }
static MPSGraphTensor *silu(MPSGraph *g, MPSGraphTensor *x) { return mul(g,x,[g sigmoidWithTensor:x name:nil]); }
static MPSGraphTensor *linear(PKContext *c, PKPlan *p, MPSGraphTensor *x, NSString *prefix, NSUInteger in, NSUInteger out, BOOL bias) {
    MPSGraphTensor *weight = w(c,p,[prefix stringByAppendingString:@".weight"],@[ @(out), @(in) ]);
    weight = [p.graph transposeTensor:weight dimension:0 withDimension:1 name:nil];
    x = mm(p.graph,x,weight);
    if (bias) x = add(p.graph,x,w(c,p,[prefix stringByAppendingString:@".bias"],@[ @(out) ]));
    return x;
}
static MPSGraphTensor *norm(PKContext *c, PKPlan *p, MPSGraphTensor *x, NSString *prefix) {
    MPSGraph *g = p.graph;
    MPSGraphTensor *mean = [g meanOfTensor:x axes:@[@1] name:nil];
    MPSGraphTensor *center = [g subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
    MPSGraphTensor *var = [g meanOfTensor:mul(g,center,center) axes:@[@1] name:nil];
    MPSGraphTensor *den = [g squareRootWithTensor:add(g,var,[g constantWithScalar:1e-5 dataType:MPSDataTypeFloat32]) name:nil];
    x = [g divisionWithPrimaryTensor:center secondaryTensor:den name:nil];
    return add(g,mul(g,x,w(c,p,[prefix stringByAppendingString:@".weight"],@[@1024])),w(c,p,[prefix stringByAppendingString:@".bias"],@[@1024]));
}
static MPSGraphTensor *conv2(PKContext *c, PKPlan *p, MPSGraphTensor *x, NSString *prefix, NSUInteger in, NSUInteger out, NSUInteger kernel, NSUInteger stride, NSUInteger groups, BOOL bias) {
    MPSGraphConvolution2DOpDescriptor *d = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride strideInY:stride dilationRateInX:1 dilationRateInY:1 groups:groups paddingLeft:kernel/2 paddingRight:kernel/2 paddingTop:kernel/2 paddingBottom:kernel/2 paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    MPSGraphTensor *weight = w(c,p,[prefix stringByAppendingString:@".weight"],@[ @(out), @(in/groups), @(kernel), @(kernel) ]);
    x = [p.graph convolution2DWithSourceTensor:x weightsTensor:weight descriptor:d name:nil];
    if (bias) x = add(p.graph,x,reshape(p.graph,w(c,p,[prefix stringByAppendingString:@".bias"],@[ @(out) ]),@[@1,@(out),@1,@1]));
    return x;
}
static PKPlan *plan(void) {
    PKPlan *p = [PKPlan new]; p.graph = [MPSGraph new];
    p.graph.options = MPSGraphOptionsSynchronizeResults;
    p.inputs = [NSMutableDictionary new]; return p;
}
static MPSGraphTensorData *run(PKContext *c, PKPlan *p, NSDictionary<NSString *, MPSGraphTensorData *> *dynamic, NSUInteger layer) {
    NSMutableDictionary *feeds = [NSMutableDictionary new];
    for (NSString *name in p.inputs) {
        NSString *actual = [name stringByReplacingOccurrencesOfString:@"layer." withString:[NSString stringWithFormat:@"encoder.layers.%lu.", (unsigned long)layer]];
        MPSGraphTensorData *value = dynamic[name] ?: c.weights[actual];
        if (!value || ![value.shape isEqualToArray:p.inputs[name].shape])
            [NSException raise:@"Parakeet" format:@"invalid feed %@", actual];
        feeds[p.inputs[name]] = value;
    }
    if (!p.executable) {
        NSMutableDictionary *types = [NSMutableDictionary new];
        for (MPSGraphTensor *t in feeds) types[t] = [[MPSGraphShapedType alloc] initWithShape:t.shape dataType:t.dataType];
        MPSGraphCompilationDescriptor *d = [MPSGraphCompilationDescriptor new];
        d.optimizationLevel = MPSGraphOptimizationLevel0; d.waitForCompilationCompletion = YES;
        MPSGraphDevice *device = [MPSGraphDevice deviceWithMTLDevice:c.device];
        p.executable = [p.graph compileWithDevice:device feeds:types targetTensors:@[p.output] targetOperations:nil compilationDescriptor:d];
        NSMutableArray *inputTypes = [NSMutableArray new];
        for (MPSGraphTensor *t in p.executable.feedTensors) {
            MPSGraphShapedType *type = types[t];
            if (!type) for (MPSGraphTensor *original in types) if ([original.operation.name isEqualToString:t.operation.name]) { type = types[original]; break; }
            if (!type) [NSException raise:@"Parakeet" format:@"missing executable input type"];
            [inputTypes addObject:type];
        }
        [p.executable specializeWithDevice:device inputTypes:inputTypes compilationDescriptor:d];
        p.executable.options = MPSGraphOptionsSynchronizeResults;
    }
    NSMutableArray *inputs = [NSMutableArray new];
    for (MPSGraphTensor *t in p.executable.feedTensors) {
        MPSGraphTensorData *value = feeds[t];
        if (!value) for (MPSGraphTensor *original in feeds) if ([original.operation.name isEqualToString:t.operation.name]) { value = feeds[original]; break; }
        if (!value) [NSException raise:@"Parakeet" format:@"missing executable feed"];
        [inputs addObject:value];
    }
    MPSGraphTensorData *result = [[p.executable runWithMTLCommandQueue:c.queue inputsArray:inputs resultsArray:nil executionDescriptor:nil] firstObject];
    if (!result) [NSException raise:@"Parakeet" format:@"GPU returned no result"];
    return result;
}

int32_t synapse_parakeet_begin(void *raw, const float *mel, uint64_t rung, uint64_t valid) {
    @autoreleasepool { @try {
        PKContext *c = (__bridge PKContext *)raw;
        if (c.rung != rung) { [c.plans removeAllObjects]; c.rung = rung; }
        c.validMel = valid;
        NSUInteger t = rung, m = rung * 8;
        NSMutableDictionary *dynamic = [NSMutableDictionary new];
        dynamic[@"mel"] = data(c,mel,@[@1,@1,@(m),@128]);
        for (NSUInteger stage = 0, length = m, n = valid; stage < 3; stage++) {
            length = (length+1)/2; n = (n+1)/2;
            NSMutableData *mask = [NSMutableData dataWithLength:length*sizeof(float)];
            float *v = mask.mutableBytes;
            for (NSUInteger i=0; i<n; i++) v[i]=1;
            dynamic[[NSString stringWithFormat:@"sub_mask_%lu",stage]] = data(c,v,@[@1,@1,@(length),@1]);
        }
        PKPlan *p = c.plans[@"sub"];
        if (!p) {
            p = plan(); MPSGraph *g = p.graph;
            MPSGraphTensor *x = ph(p,@"mel",@[@1,@1,@(m),@128]);
            x = conv2(c,p,x,@"encoder.subsampling.layers.0",1,256,3,2,1,YES);
            x = [g reLUWithTensor:mul(g,x,ph(p,@"sub_mask_0",@[@1,@1,@(m/2),@1])) name:nil];
            for (NSUInteger stage=1; stage<3; stage++) {
                NSUInteger i = 1+(stage-1)*3;
                NSString *prefix = [NSString stringWithFormat:@"encoder.subsampling.layers.%lu",i+1];
                x = conv2(c,p,x,prefix,256,256,3,2,256,YES);
                MPSGraphTensor *mask = ph(p,[NSString stringWithFormat:@"sub_mask_%lu",stage],@[@1,@1,@(m/(1<<(stage+1))),@1]);
                x = mul(g,x,mask);
                x = conv2(c,p,x,[NSString stringWithFormat:@"encoder.subsampling.layers.%lu",i+2],256,256,1,1,1,YES);
                // The reference masks both depthwise and pointwise convolutions.
                x = [g reLUWithTensor:mul(g,x,mask) name:nil];
            }
            x = [g transposeTensor:x permutation:@[@0,@2,@1,@3] name:nil];
            p.output = linear(c,p,reshape(g,x,@[@(t),@4096]),@"encoder.subsampling.linear",4096,1024,YES);
            c.plans[@"sub"] = p;
        }
        c.hidden = run(c,p,dynamic,0);
        NSUInteger n = (valid+7)/8;
        NSMutableData *row = [NSMutableData dataWithLength:t*sizeof(float)];
        float *rv = row.mutableBytes; for (NSUInteger i=0;i<n;i++) rv[i]=1;
        c.rowMask = data(c,rv,@[@(t),@1]);
        NSMutableData *pair = [NSMutableData dataWithLength:t*t*sizeof(float)];
        float *pv = pair.mutableBytes;
        for (NSUInteger i=0;i<n;i++) for (NSUInteger j=0;j<n;j++) pv[i*t+j]=1;
        c.pairMask = data(c,pv,@[@1,@(t),@(t)]);
        NSMutableData *positions = [NSMutableData dataWithLength:(2*t-1)*1024*sizeof(float)];
        float *pos = positions.mutableBytes;
        for (NSUInteger i=0;i<2*t-1;i++) for (NSUInteger j=0;j<512;j++) {
            float inv = 1.0f/powf(10000.0f,(float)(j*2)/1024.0f);
            float angle = (float)((int64_t)t-1-(int64_t)i)*inv;
            pos[i*1024+j*2] = sinf(angle); pos[i*1024+j*2+1] = cosf(angle);
        }
        c.positions = data(c,pos,@[@(2*t-1),@1024]);
        return 0;
    } @catch(NSException *e) { return failure(e); } }
}

static MPSGraphTensor *heads(MPSGraph *g, MPSGraphTensor *x, NSUInteger t) {
    return [g transposeTensor:reshape(g,x,@[@(t),@8,@128]) permutation:@[@1,@0,@2] name:nil];
}
static MPSGraphTensor *ff(PKContext *c, PKPlan *p, MPSGraphTensor *x, NSString *prefix) {
    x = linear(c,p,x,[prefix stringByAppendingString:@".linear1"],1024,4096,NO);
    return linear(c,p,silu(p.graph,x),[prefix stringByAppendingString:@".linear2"],4096,1024,NO);
}
static PKPlan *layerPlan(PKContext *c) {
    PKPlan *p = plan(); MPSGraph *g = p.graph; NSUInteger t = c.rung;
    MPSGraphTensor *x = ph(p,@"hidden",@[@(t),@1024]);
    MPSGraphTensor *row = ph(p,@"row",@[@(t),@1]);
    MPSGraphTensor *pair = ph(p,@"pair",@[@1,@(t),@(t)]);
    MPSGraphTensor *positions = ph(p,@"positions",@[@(2*t-1),@1024]);
    x = add(g,x,scale(g,ff(c,p,norm(c,p,x,@"layer.norm_feed_forward1"),@"layer.feed_forward1"),0.5));
    MPSGraphTensor *normalized = norm(c,p,x,@"layer.norm_self_att");
    MPSGraphTensor *q = heads(g,linear(c,p,normalized,@"layer.self_attn.q_proj",1024,1024,NO),t);
    MPSGraphTensor *k = heads(g,linear(c,p,normalized,@"layer.self_attn.k_proj",1024,1024,NO),t);
    MPSGraphTensor *v = heads(g,linear(c,p,normalized,@"layer.self_attn.v_proj",1024,1024,NO),t);
    MPSGraphTensor *u = reshape(g,w(c,p,@"layer.self_attn.bias_u",@[@8,@128]),@[@8,@1,@128]);
    MPSGraphTensor *vb = reshape(g,w(c,p,@"layer.self_attn.bias_v",@[@8,@128]),@[@8,@1,@128]);
    MPSGraphTensor *relative = heads(g,linear(c,p,positions,@"layer.self_attn.relative_k_proj",1024,1024,NO),2*t-1);
    MPSGraphTensor *bd = mm(g,add(g,q,vb),[g transposeTensor:relative dimension:1 withDimension:2 name:nil]);
    // Match the reference's pad/view/slice relative shift, including its sign.
    MPSGraphTensor *zero = [g constantWithScalar:0 shape:@[@8,@(t),@1] dataType:MPSDataTypeFloat32];
    bd = [g concatTensors:@[zero,bd] dimension:2 name:nil];
    bd = reshape(g,bd,@[@8,@(2*t),@(t)]);
    bd = [g sliceTensor:bd dimension:1 start:1 length:2*t-1 name:nil];
    bd = reshape(g,bd,@[@8,@(t),@(2*t-1)]);
    bd = [g sliceTensor:bd dimension:2 start:0 length:t name:nil];
    MPSGraphTensor *scores = add(g,scale(g,mm(g,add(g,q,u),[g transposeTensor:k dimension:1 withDimension:2 name:nil]),1.0/sqrt(128.0)),scale(g,bd,1.0/sqrt(128.0)));
    MPSGraphTensor *condition = [g castTensor:pair toType:MPSDataTypeBool name:nil];
    // Fully masked query rows use the sealed finite sentinel and then zero
    // probabilities, rather than propagating NaNs from softmax(-infinity).
    scores = [g selectWithPredicateTensor:condition truePredicateTensor:scores falsePredicateTensor:[g constantWithScalar:-10000 dataType:MPSDataTypeFloat32] name:nil];
    MPSGraphTensor *probs = mul(g,[g softMaxWithTensor:scores axis:2 name:nil],pair);
    MPSGraphTensor *attention = mm(g,probs,v);
    attention = reshape(g,[g transposeTensor:attention permutation:@[@1,@0,@2] name:nil],@[@(t),@1024]);
    x = add(g,x,linear(c,p,attention,@"layer.self_attn.o_proj",1024,1024,NO));
    normalized = norm(c,p,x,@"layer.norm_conv");
    // Conv1d pointwise weights have a singleton kernel dimension.
    MPSGraphTensor *pw = reshape(g,w(c,p,@"layer.conv.pointwise_conv1.weight",@[@2048,@1024,@1]),@[@2048,@1024]);
    MPSGraphTensor *glu = mm(g,normalized,[g transposeTensor:pw dimension:0 withDimension:1 name:nil]);
    MPSGraphTensor *a = [g sliceTensor:glu dimension:1 start:0 length:1024 name:nil];
    MPSGraphTensor *b = [g sliceTensor:glu dimension:1 start:1024 length:1024 name:nil];
    glu = mul(g,mul(g,a,[g sigmoidWithTensor:b name:nil]),row);
    MPSGraphTensor *dw = reshape(g,w(c,p,@"layer.conv.depthwise_conv.weight",@[@1024,@1,@9]),@[@1024,@1,@9,@1]);
    MPSGraphConvolution2DOpDescriptor *d = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1 strideInY:1 dilationRateInX:1 dilationRateInY:1 groups:1024 paddingLeft:0 paddingRight:0 paddingTop:4 paddingBottom:4 paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    glu = reshape(g,[g transposeTensor:glu dimension:0 withDimension:1 name:nil],@[@1,@1024,@(t),@1]);
    glu = [g convolution2DWithSourceTensor:glu weightsTensor:dw descriptor:d name:nil];
    glu = [g transposeTensor:reshape(g,glu,@[@1024,@(t)]) dimension:0 withDimension:1 name:nil];
    MPSGraphTensor *mean = w(c,p,@"layer.conv.norm.running_mean",@[@1024]);
    MPSGraphTensor *variance = w(c,p,@"layer.conv.norm.running_var",@[@1024]);
    glu = [g normalizationWithTensor:glu meanTensor:mean varianceTensor:variance gammaTensor:w(c,p,@"layer.conv.norm.weight",@[@1024]) betaTensor:w(c,p,@"layer.conv.norm.bias",@[@1024]) epsilon:1e-5 name:nil];
    glu = silu(g,glu);
    pw = reshape(g,w(c,p,@"layer.conv.pointwise_conv2.weight",@[@1024,@1024,@1]),@[@1024,@1024]);
    x = add(g,x,mm(g,glu,[g transposeTensor:pw dimension:0 withDimension:1 name:nil]));
    x = add(g,x,scale(g,ff(c,p,norm(c,p,x,@"layer.norm_feed_forward2"),@"layer.feed_forward2"),0.5));
    p.output = norm(c,p,x,@"layer.norm_out");
    return p;
}
int32_t synapse_parakeet_layer(void *raw, uint64_t layer) {
    @autoreleasepool { @try {
        PKContext *c = (__bridge PKContext *)raw;
        PKPlan *p = c.plans[@"layer"];
        if (!p) { p = layerPlan(c); c.plans[@"layer"] = p; }
        c.hidden = run(c,p,@{@"hidden":c.hidden,@"row":c.rowMask,@"pair":c.pairMask,@"positions":c.positions},layer);
        return 0;
    } @catch(NSException *e) { return failure(e); } }
}
int32_t synapse_parakeet_finish(void *raw, float *encoder, float *projected) {
    @autoreleasepool { @try {
        PKContext *c = (__bridge PKContext *)raw;
        PKPlan *p = c.plans[@"project"];
        if (!p) {
            p = plan();
            p.output = linear(c,p,ph(p,@"hidden",@[@(c.rung),@1024]),@"encoder_projector",1024,640,YES);
            c.plans[@"project"] = p;
        }
        MPSGraphTensorData *result = run(c,p,@{@"hidden":c.hidden},0);
        [c.hidden.mpsndarray readBytes:encoder strideBytes:NULL];
        [result.mpsndarray readBytes:projected strideBytes:NULL];
        c.hidden = nil;
        return 0;
    } @catch(NSException *e) { return failure(e); } }
}
