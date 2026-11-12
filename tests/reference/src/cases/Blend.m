// Blending and multisampling cases: blend factors and operations, dual
// source blending, write masks, alpha to coverage, per-sample coverage at
// the default sample positions, sample masks, sample shading, colour
// resolve, depth and stencil resolve filters, and invalid use.

#import "WMRefCase.h"

// Helpers in RenderSupport.m.
typedef struct { float position[4]; uint32_t data[4]; float extra[4]; } RVertex;
RVertex RVertexAt(double px, double py, double z, NSUInteger width, NSUInteger height, uint32_t id);
id<MTLBuffer> RVertexBuffer(WMCase *c, const RVertex *vertices, NSUInteger count);
id<MTLTexture> RTarget(WMCase *c, MTLPixelFormat format, NSUInteger width, NSUInteger height);
id<MTLTexture> RTargetSampled(WMCase *c, MTLPixelFormat format, NSUInteger width, NSUInteger height,
                              NSUInteger samples);
MTLRenderPassDescriptor *RPassColor(id<MTLTexture> color, MTLClearColor clear);
MTLRenderPipelineDescriptor *RPipelineDescriptor(WMCase *c, NSString *vertex, NSString *fragment,
                                                 MTLPixelFormat color);
NSData *RRead(WMCase *c, id<MTLTexture> texture, NSUInteger bytesPerPixel);
NSData *RRecordTexture(WMCase *c, NSString *name, id<MTLTexture> texture, WMElem elem,
                       NSUInteger bytesPerPixel, WMRule *rule);

#pragma mark Blending

// Inputs: every value has at most three significant bits, so every
// product and sum in the blend equations is exact in binary16 and the
// documented equations (MTLBlendFactor and MTLBlendOperation reference)
// determine each result exactly.
static const double kSource[4] = {0.75, 0.5, 0.25, 0.5};
static const double kSource1[4] = {0.125, 0.375, 0.625, 0.875};
static const double kDestination[4] = {0.25, 0.5, 0.75, 0.25};
static const double kBlendColor[4] = {0.5, 0.25, 0.125, 0.75};

static double BFactor(MTLBlendFactor f, int i) {
    const double *s = kSource, *d = kDestination, *b = kBlendColor, *s1 = kSource1;
    switch (f) {
    case MTLBlendFactorZero: return 0;
    case MTLBlendFactorOne: return 1;
    case MTLBlendFactorSourceColor: return s[i];
    case MTLBlendFactorOneMinusSourceColor: return 1 - s[i];
    case MTLBlendFactorSourceAlpha: return s[3];
    case MTLBlendFactorOneMinusSourceAlpha: return 1 - s[3];
    case MTLBlendFactorDestinationColor: return d[i];
    case MTLBlendFactorOneMinusDestinationColor: return 1 - d[i];
    case MTLBlendFactorDestinationAlpha: return d[3];
    case MTLBlendFactorOneMinusDestinationAlpha: return 1 - d[3];
    case MTLBlendFactorSourceAlphaSaturated: return i < 3 ? fmin(s[3], 1 - d[3]) : 1;
    case MTLBlendFactorBlendColor: return b[i];
    case MTLBlendFactorOneMinusBlendColor: return 1 - b[i];
    case MTLBlendFactorBlendAlpha: return b[3];
    case MTLBlendFactorOneMinusBlendAlpha: return 1 - b[3];
    case MTLBlendFactorSource1Color: return s1[i];
    case MTLBlendFactorOneMinusSource1Color: return 1 - s1[i];
    case MTLBlendFactorSource1Alpha: return s1[3];
    case MTLBlendFactorOneMinusSource1Alpha: return 1 - s1[3];
    default: return NAN;
    }
}

typedef struct {
    MTLBlendFactor srcRGB, dstRGB, srcA, dstA;
    MTLBlendOperation opRGB, opA;
} BState;

static double BEquation(MTLBlendOperation op, double s, double fs, double d, double fd) {
    switch (op) {
    case MTLBlendOperationAdd: return s * fs + d * fd;
    case MTLBlendOperationSubtract: return s * fs - d * fd;
    case MTLBlendOperationReverseSubtract: return d * fd - s * fs;
    case MTLBlendOperationMin: return fmin(s, d);
    case MTLBlendOperationMax: return fmax(s, d);
    default: return NAN;
    }
}

// Draws one point per state into its own pixel of a 16x4 RGBA16Float
// target cleared to kDestination; returns the target's bytes.
static NSData *BBlendStates(WMCase *c, const BState *states, NSUInteger count, BOOL dualSource) {
    id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA16Float, 16, 4);
    MTLRenderPassDescriptor *pass = RPassColor(target, MTLClearColorMake(kDestination[0], kDestination[1],
                                                                         kDestination[2], kDestination[3]));
    NSMutableArray *pipelines = [NSMutableArray array];
    for (NSUInteger k = 0; k < count; k++) {
        MTLRenderPipelineDescriptor *d = RPipelineDescriptor(
            c, @"rnd_vertex", dualSource ? @"rnd_fragment_dual_source" : @"rnd_fragment_float_point",
            MTLPixelFormatRGBA16Float);
        MTLRenderPipelineColorAttachmentDescriptor *a = d.colorAttachments[0];
        a.blendingEnabled = YES;
        a.sourceRGBBlendFactor = states[k].srcRGB;
        a.destinationRGBBlendFactor = states[k].dstRGB;
        a.sourceAlphaBlendFactor = states[k].srcA;
        a.destinationAlphaBlendFactor = states[k].dstA;
        a.rgbBlendOperation = states[k].opRGB;
        a.alphaBlendOperation = states[k].opA;
        [pipelines addObject:[c renderPipeline:d]];
    }
    RVertex *points = calloc(count, sizeof(RVertex));
    for (NSUInteger k = 0; k < count; k++) {
        points[k] = RVertexAt(k % 16 + 0.5, k / 16 + 0.5, 0, 16, 4, 0);
        float source[4] = {(float)kSource[0], (float)kSource[1], (float)kSource[2], (float)kSource[3]};
        memcpy(points[k].data, source, sizeof source);
    }
    id<MTLBuffer> buffer = RVertexBuffer(c, points, count);
    free(points);
    const float second[4] = {(float)kSource1[0], (float)kSource1[1], (float)kSource1[2], (float)kSource1[3]};
    const float *secondPointer = second;
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setBlendColorRed:(float)kBlendColor[0] green:(float)kBlendColor[1] blue:(float)kBlendColor[2]
                      alpha:(float)kBlendColor[3]];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        if (dualSource) [e setFragmentBytes:secondPointer length:sizeof(float) * 4 atIndex:0];
        for (NSUInteger k = 0; k < count; k++) {
            [e setRenderPipelineState:pipelines[k]];
            [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:k vertexCount:1];
        }
    }];
    return RRead(c, target, 8);
}

static void BRecordStates(WMCase *c, NSString *name, const BState *states, NSUInteger count, BOOL dualSource) {
    NSData *image = BBlendStates(c, states, count, dualSource);
    const uint16_t *halves = image.bytes;
    BOOL equal = YES;
    for (NSUInteger k = 0; k < count; k++)
        for (int i = 0; i < 4; i++) {
            BOOL rgb = i < 3;
            double expected = BEquation(rgb ? states[k].opRGB : states[k].opA, kSource[i],
                                        BFactor(rgb ? states[k].srcRGB : states[k].srcA, i), kDestination[i],
                                        BFactor(rgb ? states[k].dstRGB : states[k].dstA, i));
            equal &= WMHalfToDouble(halves[k * 4 + i]) == expected;
        }
    NSDictionary *layout = @{@"format": @"RGBA16Float", @"width": @16, @"height": @4, @"bytes_per_row": @128};
    [c data:name elem:WMElemF16 bytes:image.bytes length:count * 8 layout:layout rule:WMExact()];
    [c value:[name stringByAppendingString:@".equals-documented-equations"] boolean:equal rule:WMExact()];
}

WMREF_CASE(blendFactorsOperations, "blend.factors-operations", "blend.equations",
           "Every blend factor as source and destination factor, every blend operation, separate alpha state")
{
    BState states[64];
    NSUInteger n = 0;
    for (MTLBlendFactor f = MTLBlendFactorZero; f <= MTLBlendFactorOneMinusBlendAlpha; f++) {
        states[n++] = (BState){f, MTLBlendFactorZero, f, MTLBlendFactorZero, MTLBlendOperationAdd, MTLBlendOperationAdd};
        states[n++] = (BState){MTLBlendFactorZero, f, MTLBlendFactorZero, f, MTLBlendOperationAdd, MTLBlendOperationAdd};
    }
    for (MTLBlendOperation op = MTLBlendOperationAdd; op <= MTLBlendOperationMax; op++)
        states[n++] = (BState){MTLBlendFactorSourceAlpha, MTLBlendFactorOneMinusSourceAlpha, MTLBlendFactorOne,
                               MTLBlendFactorOne, op, op};
    // Separate RGB and alpha factors and operations.
    states[n++] = (BState){MTLBlendFactorSourceAlpha, MTLBlendFactorOneMinusSourceAlpha, MTLBlendFactorOne,
                           MTLBlendFactorZero, MTLBlendOperationAdd, MTLBlendOperationAdd};
    states[n++] = (BState){MTLBlendFactorDestinationColor, MTLBlendFactorBlendColor, MTLBlendFactorBlendAlpha,
                           MTLBlendFactorDestinationAlpha, MTLBlendOperationReverseSubtract, MTLBlendOperationMax};
    BRecordStates(c, @"results", states, n, NO);
}

WMREF_CASE(blendDualSource, "blend.dual-source", "blend.equations",
           "Dual-source blending with the Source1 blend factors")
{
    BState states[8];
    NSUInteger n = 0;
    for (MTLBlendFactor f = MTLBlendFactorSource1Color; f <= MTLBlendFactorOneMinusSource1Alpha; f++) {
        states[n++] = (BState){f, MTLBlendFactorZero, f, MTLBlendFactorZero, MTLBlendOperationAdd, MTLBlendOperationAdd};
        states[n++] = (BState){MTLBlendFactorOne, f, MTLBlendFactorOne, f, MTLBlendOperationAdd, MTLBlendOperationAdd};
    }
    BRecordStates(c, @"results", states, n, YES);
}

WMREF_CASE(blendWriteMask, "blend.write-mask", "blend.write-mask",
           "Colour write masks with and without blending")
{
    const MTLColorWriteMask masks[4] = {MTLColorWriteMaskNone, MTLColorWriteMaskRed | MTLColorWriteMaskAlpha,
                                        MTLColorWriteMaskGreen | MTLColorWriteMaskBlue, MTLColorWriteMaskAll};
    id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA16Float, 8, 1);
    MTLRenderPassDescriptor *pass = RPassColor(target, MTLClearColorMake(kDestination[0], kDestination[1],
                                                                         kDestination[2], kDestination[3]));
    NSMutableArray *pipelines = [NSMutableArray array];
    for (int blend = 0; blend < 2; blend++)
        for (int m = 0; m < 4; m++) {
            MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex", @"rnd_fragment_float_point",
                                                                 MTLPixelFormatRGBA16Float);
            d.colorAttachments[0].writeMask = masks[m];
            d.colorAttachments[0].blendingEnabled = blend;
            d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
            d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
            d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
            [pipelines addObject:[c renderPipeline:d]];
        }
    RVertex points[8];
    for (int k = 0; k < 8; k++) {
        points[k] = RVertexAt(k + 0.5, 0.5, 0, 8, 1, 0);
        float source[4] = {(float)kSource[0], (float)kSource[1], (float)kSource[2], (float)kSource[3]};
        memcpy(points[k].data, source, sizeof source);
    }
    id<MTLBuffer> buffer = RVertexBuffer(c, points, 8);
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        for (NSUInteger k = 0; k < 8; k++) {
            [e setRenderPipelineState:pipelines[k]];
            [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:k vertexCount:1];
        }
    }];
    RRecordTexture(c, @"results", target, WMElemF16, 8, WMExact());
}

WMREF_CASE(blendIntegerTarget, "blend.error.integer-target", "blend.equations",
           "Render pipeline with blending enabled on an R32Uint attachment (invalid use)")
{
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id",
                                                         MTLPixelFormatR32Uint);
    d.colorAttachments[0].blendingEnabled = YES;
    [c step:@"create-pipeline"];
    NSError *error = nil;
    id<MTLRenderPipelineState> pipeline = [c.device newRenderPipelineStateWithDescriptor:d error:&error];
    [c value:@"pipeline-is-nil" boolean:pipeline == nil rule:WMExact()];
    [c error:@"error" value:error rule:WMExact()];
}

#pragma mark Multisampling

// Reads every sample of a multisample texture with one of the
// rnd_read_ms_* kernels; returns width * height * samples elements.
static NSData *BReadSamples(WMCase *c, id<MTLTexture> texture, NSString *kernel, NSUInteger elementSize) {
    id<MTLComputePipelineState> pipeline = [c computePipeline:kernel];
    id<MTLBuffer> out = [c buffer:texture.width * texture.height * texture.sampleCount * elementSize];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setTexture:texture atIndex:0];
        [e setBuffer:out offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(texture.width, texture.height, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }];
    return [NSData dataWithBytes:out.contents length:out.length];
}

enum { BMW = 8, BMH = 8 };
static const uint32_t kAmbiguous = 0xffffffffu;

// The triangle used by the coverage cases, on a 1/16 pixel grid.
static const double kMSTriangle[6] = {1.0625, 0.8125, 7.4375, 2.3125, 2.1875, 7.6875};

static double BEdge(const double *a, const double *b, double x, double y) {
    double dx = b[0] - a[0], dy = b[1] - a[1];
    return ((x - a[0]) * dy - (y - a[1]) * dx) / sqrt(dx * dx + dy * dy);
}

// Classifies each sample of each pixel against kMSTriangle: 1 inside, 0
// outside, kAmbiguous within 1/16 pixel of an edge.
static void BClassify(const MTLSamplePosition *positions, NSUInteger samples, uint32_t *out) {
    const double a[2] = {kMSTriangle[0], kMSTriangle[1]}, b[2] = {kMSTriangle[2], kMSTriangle[3]},
                 cc[2] = {kMSTriangle[4], kMSTriangle[5]};
    const double margin = 1.0 / 16;
    for (NSUInteger y = 0; y < BMH; y++)
        for (NSUInteger x = 0; x < BMW; x++)
            for (NSUInteger s = 0; s < samples; s++) {
                double px = x + positions[s].x, py = y + positions[s].y;
                double e[3] = {BEdge(a, b, px, py), BEdge(b, cc, px, py), BEdge(cc, a, px, py)};
                BOOL strongOut = (e[0] < -margin || e[1] < -margin || e[2] < -margin) &&
                                 (e[0] > margin || e[1] > margin || e[2] > margin);
                BOOL near = fabs(e[0]) < margin || fabs(e[1]) < margin || fabs(e[2]) < margin;
                uint32_t value = (e[0] > 0 && e[1] > 0 && e[2] > 0) || (e[0] < 0 && e[1] < 0 && e[2] < 0);
                out[(y * BMW + x) * samples + s] = near && !strongOut ? kAmbiguous : value;
            }
}

// Copies `measured`, replacing samples of pixels that hold an ambiguous
// sample by kAmbiguous when `wholePixel` is set, else only the ambiguous
// samples themselves.
static NSData *BMask(NSData *measured, const uint32_t *classes, NSUInteger samples, BOOL wholePixel) {
    NSMutableData *masked = [measured mutableCopy];
    uint32_t *m = masked.mutableBytes;
    for (NSUInteger p = 0; p < BMW * BMH; p++) {
        BOOL any = NO;
        for (NSUInteger s = 0; s < samples; s++) any |= classes[p * samples + s] == kAmbiguous;
        for (NSUInteger s = 0; s < samples; s++)
            if (wholePixel ? any : classes[p * samples + s] == kAmbiguous) m[p * samples + s] = kAmbiguous;
    }
    return masked;
}

static id<MTLRenderPipelineState> BMSPipeline(WMCase *c, NSString *fragment, MTLPixelFormat format,
                                              NSUInteger samples) {
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", fragment, format);
    d.rasterSampleCount = samples;
    return [c renderPipeline:d];
}

static NSData *BDrawMS(WMCase *c, id<MTLRenderPipelineState> pipeline, NSUInteger samples, uint32_t sampleMask) {
    RVertex v[3];
    for (int k = 0; k < 3; k++) {
        v[k] = RVertexAt(kMSTriangle[2 * k], kMSTriangle[2 * k + 1], 0, BMW, BMH, 1);
        v[k].data[1] = sampleMask;
    }
    id<MTLBuffer> buffer = RVertexBuffer(c, v, 3);
    id<MTLTexture> target = RTargetSampled(c, MTLPixelFormatR32Uint, BMW, BMH, samples);
    MTLRenderPassDescriptor *pass = RPassColor(target, MTLClearColorMake(0, 0, 0, 0));
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    return BReadSamples(c, target, @"rnd_read_ms_uint", 4);
}

static void BCoverage(WMCase *c, NSUInteger samples) {
    [c require:[NSString stringWithFormat:@"device.texture-sample-count-%lu", (unsigned long)samples]
         value:[c.device supportsTextureSampleCount:samples]];
    MTLSamplePosition positions[16];
    [c.device getDefaultSamplePositions:positions count:samples];
    float raw[32];
    for (NSUInteger s = 0; s < samples; s++) {
        raw[2 * s] = positions[s].x;
        raw[2 * s + 1] = positions[s].y;
    }
    [c data:@"default-sample-positions" elem:WMElemF32 bytes:raw length:samples * 8 rule:WMDevice()];
    uint32_t classes[BMW * BMH * 16];
    BClassify(positions, samples, classes);
    [c data:@"classification" elem:WMElemU32 bytes:classes length:BMW * BMH * samples * 4 rule:WMDevice()];
    NSString *edgeRule = @"samples within 1/16 pixel of an edge depend on the rasterizer's edge rule and subpixel "
                         @"precision, which Metal's documentation does not state";

    // Coverage: each covered sample stores the triangle id (1).
    NSData *coverage = BDrawMS(c, BMSPipeline(c, @"rnd_fragment_id", MTLPixelFormatR32Uint, samples), samples, 0);
    [c data:@"coverage" elem:WMElemU32 bytes:coverage.bytes length:coverage.length rule:WMNone(edgeRule)];
    NSData *masked = BMask(coverage, classes, samples, NO);
    [c data:@"coverage.unambiguous" elem:WMElemU32 bytes:masked.bytes length:masked.length rule:WMExact()];
    NSData *expected = BMask([NSData dataWithBytes:classes length:BMW * BMH * samples * 4], classes, samples, NO);
    [c value:@"coverage.unambiguous-equals-sample-containment" boolean:[masked isEqualToData:expected]
        rule:WMExact()];

    // [[sample_mask]] input: the fragment of a pixel stores 0x100 + mask in
    // every covered sample; pixels with an ambiguous sample are masked out.
    NSData *maskIn = BDrawMS(c, BMSPipeline(c, @"rnd_fragment_mask_in", MTLPixelFormatR32Uint, samples), samples, 0);
    [c data:@"sample-mask-input" elem:WMElemU32 bytes:maskIn.bytes length:maskIn.length rule:WMNone(edgeRule)];
    NSData *maskInMasked = BMask(maskIn, classes, samples, YES);
    [c data:@"sample-mask-input.unambiguous-pixels" elem:WMElemU32 bytes:maskInMasked.bytes
     length:maskInMasked.length rule:WMExact()];

    // [[sample_id]]: per-sample shading stores 16 * id + sample index.
    NSData *sampleId = BDrawMS(c, BMSPipeline(c, @"rnd_fragment_sample_id", MTLPixelFormatR32Uint, samples), samples, 0);
    NSData *sampleIdMasked = BMask(sampleId, classes, samples, NO);
    [c data:@"sample-id.unambiguous" elem:WMElemU32 bytes:sampleIdMasked.bytes length:sampleIdMasked.length
       rule:WMExact()];

    // [[sample_mask]] output 0b0101: only samples 0 and 2 of covered pixels
    // are written.
    NSData *maskOut = BDrawMS(c, BMSPipeline(c, @"rnd_fragment_mask_out", MTLPixelFormatR32Uint, samples), samples, 0x5);
    NSData *maskOutMasked = BMask(maskOut, classes, samples, NO);
    [c data:@"sample-mask-output-0x5.unambiguous" elem:WMElemU32 bytes:maskOutMasked.bytes
     length:maskOutMasked.length rule:WMExact()];
}

WMREF_CASE(msaaCoverage4, "msaa.coverage-4x", "msaa.coverage",
           "4x multisample coverage, [[sample_mask]] input and output and [[sample_id]] at default positions")
{
    BCoverage(c, 4);
}

WMREF_CASE(msaaCoverage2, "msaa.coverage-2x", "msaa.coverage",
           "2x multisample coverage, [[sample_mask]] input and output and [[sample_id]] at default positions")
{
    BCoverage(c, 2);
}

WMREF_CASE(msaaSamplePositions, "msaa.default-sample-positions", "msaa.sample-positions",
           "getDefaultSamplePositions:count: for every supported sample count")
{
    [c value:@"programmable-sample-positions-supported" boolean:c.device.areProgrammableSamplePositionsSupported
        rule:WMDevice()];
    for (NSUInteger n = 1; n <= 8; n++) {
        BOOL supported = [c.device supportsTextureSampleCount:n];
        [c value:[NSString stringWithFormat:@"supports-%lu", (unsigned long)n] boolean:supported rule:WMDevice()];
        if (!supported) continue;
        MTLSamplePosition positions[8];
        [c.device getDefaultSamplePositions:positions count:n];
        float raw[16];
        for (NSUInteger s = 0; s < n; s++) {
            raw[2 * s] = positions[s].x;
            raw[2 * s + 1] = positions[s].y;
        }
        [c data:[NSString stringWithFormat:@"positions-%lu", (unsigned long)n] elem:WMElemF32 bytes:raw
         length:n * 8 rule:WMDevice()];
    }
}

WMREF_CASE(msaaColorResolve, "msaa.color-resolve", "msaa.resolve",
           "storeAndMultisampleResolve of a 4x RGBA16Float attachment: samples and resolved pixels")
{
    [c require:@"device.texture-sample-count-4" value:[c.device supportsTextureSampleCount:4]];
    MTLSamplePosition positions[4];
    [c.device getDefaultSamplePositions:positions count:4];
    uint32_t classes[BMW * BMH * 4];
    BClassify(positions, 4, classes);
    RVertex v[3];
    const float color[4] = {1.0f, 0.5f, 0.25f, 1.0f};
    for (int k = 0; k < 3; k++) {
        v[k] = RVertexAt(kMSTriangle[2 * k], kMSTriangle[2 * k + 1], 0, BMW, BMH, 0);
        memcpy(v[k].data, color, sizeof color);
    }
    id<MTLBuffer> buffer = RVertexBuffer(c, v, 3);
    id<MTLRenderPipelineState> pipeline = BMSPipeline(c, @"rnd_fragment_float", MTLPixelFormatRGBA16Float, 4);
    id<MTLTexture> target = RTargetSampled(c, MTLPixelFormatRGBA16Float, BMW, BMH, 4);
    id<MTLTexture> resolved = RTarget(c, MTLPixelFormatRGBA16Float, BMW, BMH);
    MTLRenderPassDescriptor *pass = RPassColor(target, MTLClearColorMake(0, 0, 0, 0));
    pass.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
    pass.colorAttachments[0].resolveTexture = resolved;
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    // Samples as float4: inside samples hold the fragment colour, outside
    // samples the clear colour; ambiguous samples are replaced by all-ones.
    NSData *samples = BReadSamples(c, target, @"rnd_read_ms_float", 16);
    NSMutableData *masked = [samples mutableCopy];
    uint32_t *words = masked.mutableBytes;
    for (NSUInteger i = 0; i < BMW * BMH * 4; i++)
        if (classes[i] == kAmbiguous)
            for (int k = 0; k < 4; k++) words[i * 4 + k] = kAmbiguous;
    [c data:@"samples.unambiguous" elem:WMElemU32 bytes:masked.bytes length:masked.length rule:WMExact()];
    RRecordTexture(c, @"resolved", resolved, WMElemF16, 8,
                   WMNone(@"Metal leaves the multisample resolve filter to the implementation"));
}

// Per-sample shading writes depths {-, 0.125, 0.5, 0.375} to samples 1..3
// (and stencil 9 with `stencil`); sample 0 is discarded and keeps the clear
// values. Each depth resolve filter is applied, and with `stencil` the
// stencil resolve filters sample0 and depthResolvedSample.
static void BDepthResolve(WMCase *c, BOOL stencil) {
    // Every pixel is covered; the clear values are depth 1.0 and stencil 3.
    RVertex v[3] = {RVertexAt(-1, -1, 0.5, 4, 4, 1), RVertexAt(12, -1, 0.5, 4, 4, 1), RVertexAt(-1, 12, 0.5, 4, 4, 1)};
    id<MTLBuffer> buffer = RVertexBuffer(c, v, 3);
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_sample_depth",
                                                         MTLPixelFormatR32Uint);
    d.rasterSampleCount = 4;
    MTLPixelFormat format = stencil ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatDepth32Float;
    d.depthAttachmentPixelFormat = format;
    if (stencil) d.stencilAttachmentPixelFormat = format;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
    ds.depthCompareFunction = MTLCompareFunctionAlways;
    ds.depthWriteEnabled = YES;
    MTLStencilDescriptor *s = [MTLStencilDescriptor new];
    s.stencilCompareFunction = MTLCompareFunctionAlways;
    s.depthStencilPassOperation = MTLStencilOperationReplace;
    if (stencil) {
        ds.frontFaceStencil = s;
        ds.backFaceStencil = s;
    }
    id<MTLDepthStencilState> state = [c depthStencilState:ds];
    const struct { MTLMultisampleDepthResolveFilter depth; MTLMultisampleStencilResolveFilter stencil;
                   __unsafe_unretained NSString *name; } filters[3] = {
        {MTLMultisampleDepthResolveFilterSample0, MTLMultisampleStencilResolveFilterSample0, @"sample0"},
        {MTLMultisampleDepthResolveFilterMin, MTLMultisampleStencilResolveFilterDepthResolvedSample, @"min"},
        {MTLMultisampleDepthResolveFilterMax, MTLMultisampleStencilResolveFilterDepthResolvedSample, @"max"},
    };
    for (int f = 0; f < 3; f++) {
        id<MTLTexture> color = RTargetSampled(c, MTLPixelFormatR32Uint, 4, 4, 4);
        id<MTLTexture> depthStencil = RTargetSampled(c, format, 4, 4, 4);
        id<MTLTexture> resolved = RTarget(c, format, 4, 4);
        MTLRenderPassDescriptor *pass = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
        pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
        pass.depthAttachment.texture = depthStencil;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.clearDepth = 1.0;
        pass.depthAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
        pass.depthAttachment.resolveTexture = resolved;
        pass.depthAttachment.depthResolveFilter = filters[f].depth;
        if (stencil) {
            pass.stencilAttachment.texture = depthStencil;
            pass.stencilAttachment.loadAction = MTLLoadActionClear;
            pass.stencilAttachment.clearStencil = 3;
            pass.stencilAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
            pass.stencilAttachment.resolveTexture = resolved;
            pass.stencilAttachment.stencilResolveFilter = filters[f].stencil;
        }
        [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:pipeline];
            [e setDepthStencilState:state];
            [e setStencilReferenceValue:9];
            [e setVertexBuffer:buffer offset:0 atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }];
        if (f == 0) {
            NSData *samples = BReadSamples(c, depthStencil, @"rnd_read_ms_depth", 4);
            [c data:@"depth-samples" elem:WMElemF32 bytes:samples.bytes length:samples.length rule:WMExact()];
        }
        id<MTLBuffer> out = [c buffer:16 * 4 + 16];
        [c blit:^(id<MTLBlitCommandEncoder> e) {
            [e copyFromTexture:resolved sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(4, 4, 1) toBuffer:out destinationOffset:0 destinationBytesPerRow:16
         destinationBytesPerImage:64 options:stencil ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone];
            if (stencil)
                [e copyFromTexture:resolved sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                        sourceSize:MTLSizeMake(4, 4, 1) toBuffer:out destinationOffset:64 destinationBytesPerRow:4
             destinationBytesPerImage:16 options:MTLBlitOptionStencilFromDepthStencil];
        }];
        [c data:[filters[f].name stringByAppendingString:@".resolved-depth"] elem:WMElemF32 bytes:out.contents
         length:64 rule:WMExact()];
        if (stencil)
            [c data:[filters[f].name stringByAppendingString:@".resolved-stencil"] elem:WMElemU8
              bytes:(const uint8_t *)out.contents + 64 length:16 rule:WMExact()];
    }
}

WMREF_CASE(msaaDepthResolve, "msaa.depth-resolve", "msaa.resolve",
           "Depth resolve filters sample0, min and max on a 4-sample Depth32Float attachment")
{
    // "MSAA depth resolve": Apple3 and later (Metal Feature Set Tables, 2026-10-23).
    [c require:@"family.apple3" value:[c.device supportsFamily:MTLGPUFamilyApple3]];
    [c require:@"device.texture-sample-count-4" value:[c.device supportsTextureSampleCount:4]];
    BDepthResolve(c, NO);
}

WMREF_CASE(msaaDepthStencilResolve, "msaa.depth-stencil-resolve", "msaa.resolve",
           "Depth resolve filters sample0, min and max with stencil resolve sample0 and depthResolvedSample")
{
    // "Stencil resolve": Apple5 and later (Metal Feature Set Tables, 2026-10-23).
    [c require:@"family.apple5" value:[c.device supportsFamily:MTLGPUFamilyApple5]];
    [c require:@"device.texture-sample-count-4" value:[c.device supportsTextureSampleCount:4]];
    BDepthResolve(c, YES);
}

WMREF_CASE(msaaAlphaToCoverage, "msaa.alpha-to-coverage", "msaa.alpha-to-coverage",
           "Alpha to coverage on a 4x attachment for alpha 0, 0.25, 0.5, 0.75 and 1")
{
    [c require:@"device.texture-sample-count-4" value:[c.device supportsTextureSampleCount:4]];
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_alpha",
                                                         MTLPixelFormatRGBA16Float);
    d.rasterSampleCount = 4;
    d.alphaToCoverageEnabled = YES;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    id<MTLTexture> target = RTargetSampled(c, MTLPixelFormatRGBA16Float, 10, 2, 4);
    MTLRenderPassDescriptor *pass = RPassColor(target, MTLClearColorMake(0, 0, 0, 0));
    NSMutableArray *buffers = [NSMutableArray array];
    for (int k = 0; k < 5; k++) {
        float alpha = k * 0.25f;
        RVertex v[3] = {RVertexAt(-1, -1, 0, 10, 2, 0), RVertexAt(40, -1, 0, 10, 2, 0), RVertexAt(-1, 40, 0, 10, 2, 0)};
        for (int i = 0; i < 3; i++) memcpy(&v[i].data[0], &alpha, 4);
        [buffers addObject:RVertexBuffer(c, v, 3)];
    }
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        for (NSUInteger k = 0; k < 5; k++) {
            [e setScissorRect:(MTLScissorRect){k * 2, 0, 2, 2}];
            [e setVertexBuffer:buffers[k] offset:0 atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
    }];
    NSData *samples = BReadSamples(c, target, @"rnd_read_ms_float", 16);
    [c data:@"samples" elem:WMElemF32 bytes:samples.bytes length:samples.length
       rule:WMNone(@"Metal does not define how alpha maps to a coverage mask")];
}

WMREF_CASE(msaaSampleCountMismatch, "msaa.error.sample-count-mismatch", "msaa.coverage",
           "Pipeline with rasterSampleCount 1 used with a 4x attachment (invalid use)")
{
    [c require:@"device.texture-sample-count-4" value:[c.device supportsTextureSampleCount:4]];
    id<MTLRenderPipelineState> pipeline = BMSPipeline(c, @"rnd_fragment_id", MTLPixelFormatR32Uint, 1);
    RVertex v[3] = {RVertexAt(-1, -1, 0, 4, 4, 1), RVertexAt(12, -1, 0, 4, 4, 1), RVertexAt(-1, 12, 0, 4, 4, 1)};
    id<MTLBuffer> buffer = RVertexBuffer(c, v, 3);
    id<MTLTexture> target = RTargetSampled(c, MTLPixelFormatR32Uint, 4, 4, 4);
    [c step:@"draw"];
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    [c step:@"completed"];
    NSData *samples = BReadSamples(c, target, @"rnd_read_ms_uint", 4);
    [c data:@"samples" elem:WMElemU32 bytes:samples.bytes length:samples.length
       rule:WMNone(@"the result of a sample count mismatch is not defined")];
}
