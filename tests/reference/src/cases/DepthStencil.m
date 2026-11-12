// Depth and stencil cases: compare functions, depth write mask, stencil
// operations, masks and reference values (including separate front and
// back values), fragment depth output, depth bias, and invalid use.
//
// Every draw covers whole pixel rectangles selected by a scissor rectangle
// around a triangle that covers the target, so coverage never depends on an
// edge rule. Depth values are dyadic and exact in binary32.

#import "WMRefCase.h"

// Helpers in RenderSupport.m.
typedef struct { float position[4]; uint32_t data[4]; float extra[4]; } RVertex;
RVertex RVertexAt(double px, double py, double z, NSUInteger width, NSUInteger height, uint32_t id);
id<MTLBuffer> RVertexBuffer(WMCase *c, const RVertex *vertices, NSUInteger count);
id<MTLTexture> RTarget(WMCase *c, MTLPixelFormat format, NSUInteger width, NSUInteger height);
MTLRenderPassDescriptor *RPassColor(id<MTLTexture> color, MTLClearColor clear);
MTLRenderPipelineDescriptor *RPipelineDescriptor(WMCase *c, NSString *vertex, NSString *fragment,
                                                 MTLPixelFormat color);
NSData *RRecordTexture(WMCase *c, NSString *name, id<MTLTexture> texture, WMElem elem,
                       NSUInteger bytesPerPixel, WMRule *rule);

enum { DSW = 16, DSH = 4 };

// A triangle that covers the whole DSW x DSH target at depth z.
static id<MTLBuffer> DSCover(WMCase *c, double z, uint32_t id) {
    RVertex v[3] = {RVertexAt(-1, -1, z, DSW, DSH, id), RVertexAt(3 * DSW, -1, z, DSW, DSH, id),
                    RVertexAt(-1, 3 * DSH, z, DSW, DSH, id)};
    return RVertexBuffer(c, v, 3);
}

// Reads one aspect of a Depth32Float_Stencil8 texture through a blit with
// the corresponding MTLBlitOption.
static NSData *DSReadAspect(WMCase *c, id<MTLTexture> texture, MTLBlitOption option, NSUInteger bytesPerPixel) {
    id<MTLBuffer> buffer = [c buffer:texture.width * texture.height * bytesPerPixel];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(texture.width, texture.height, 1) toBuffer:buffer destinationOffset:0
       destinationBytesPerRow:texture.width * bytesPerPixel
     destinationBytesPerImage:texture.width * texture.height * bytesPerPixel options:option];
    }];
    return [NSData dataWithBytes:buffer.contents length:buffer.length];
}

static void DSRecordAspect(WMCase *c, NSString *name, id<MTLTexture> texture, MTLBlitOption option) {
    BOOL stencil = option == MTLBlitOptionStencilFromDepthStencil;
    NSData *data = DSReadAspect(c, texture, option, stencil ? 1 : 4);
    NSDictionary *layout = @{@"format": stencil ? @"Depth32Float_Stencil8.stencil" : @"Depth32Float_Stencil8.depth",
                             @"width": @(texture.width), @"height": @(texture.height),
                             @"bytes_per_row": @(texture.width * (stencil ? 1 : 4))};
    [c data:name elem:stencil ? WMElemU8 : WMElemF32 bytes:data.bytes length:data.length layout:layout rule:WMExact()];
}

static MTLStencilDescriptor *DSStencil(MTLCompareFunction compare, MTLStencilOperation fail,
                                       MTLStencilOperation depthFail, MTLStencilOperation pass,
                                       uint32_t readMask, uint32_t writeMask) {
    MTLStencilDescriptor *s = [MTLStencilDescriptor new];
    s.stencilCompareFunction = compare;
    s.stencilFailureOperation = fail;
    s.depthFailureOperation = depthFail;
    s.depthStencilPassOperation = pass;
    s.readMask = readMask;
    s.writeMask = writeMask;
    return s;
}

WMREF_CASE(dsCompareFunctions, "depth-stencil.compare-functions", "depth-stencil.depth-test",
           "Every MTLCompareFunction for the depth test, with depth writes on and off")
{
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id",
                                                         MTLPixelFormatR32Uint);
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    // Three columns drawn at depths below, equal to and above the clear depth.
    const double depths[3] = {0.25, 0.5, 0.75};
    NSArray<id<MTLBuffer>> *covers = @[DSCover(c, depths[0], 1), DSCover(c, depths[1], 2), DSCover(c, depths[2], 3)];
    NSString *const names[8] = {@"never", @"less", @"equal", @"less-equal", @"greater", @"not-equal",
                                @"greater-equal", @"always"};
    for (int write = 1; write >= 0; write--)
        for (int f = 0; f < 8; f++) {
            MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
            ds.depthCompareFunction = (MTLCompareFunction)f;
            ds.depthWriteEnabled = write;
            id<MTLDepthStencilState> state = [c depthStencilState:ds];
            id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, DSW, DSH);
            id<MTLTexture> depth = RTarget(c, MTLPixelFormatDepth32Float, DSW, DSH);
            MTLRenderPassDescriptor *pass = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
            pass.depthAttachment.texture = depth;
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.storeAction = MTLStoreActionStore;
            pass.depthAttachment.clearDepth = 0.5;
            [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
                [e setRenderPipelineState:pipeline];
                [e setDepthStencilState:state];
                for (int k = 0; k < 3; k++) {
                    [e setScissorRect:(MTLScissorRect){(NSUInteger)k * 5, 0, k == 2 ? 6 : 5, DSH}];
                    [e setVertexBuffer:covers[k] offset:0 atIndex:0];
                    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                }
            }];
            NSString *prefix = [NSString stringWithFormat:@"%@.write-%@", names[f], write ? @"on" : @"off"];
            RRecordTexture(c, [prefix stringByAppendingString:@".color"], color, WMElemU32, 4, WMExact());
            RRecordTexture(c, [prefix stringByAppendingString:@".depth"], depth, WMElemF32, 4, WMExact());
        }
}

WMREF_CASE(dsStencilOperations, "depth-stencil.stencil-operations", "depth-stencil.stencil",
           "Every MTLStencilOperation on the stencil-fail, depth-fail and pass paths, with a write mask")
{
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id",
                                                         MTLPixelFormatR32Uint);
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    d.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    id<MTLBuffer> cover = DSCover(c, 0.5, 1);
    NSString *const names[8] = {@"keep", @"zero", @"replace", @"increment-clamp", @"decrement-clamp",
                                @"invert", @"increment-wrap", @"decrement-wrap"};
    const uint8_t clears[2] = {0xfe, 0x01};
    for (int clearIndex = 0; clearIndex < 2; clearIndex++)
        for (int op = 0; op < 8; op++) {
            MTLStencilOperation operation = (MTLStencilOperation)op;
            // Column 0: stencil test fails. Column 1: stencil passes, depth
            // fails. Column 2: both pass. Write mask 0xf7 keeps bit 3.
            MTLDepthStencilDescriptor *paths[3];
            for (int k = 0; k < 3; k++) {
                paths[k] = [MTLDepthStencilDescriptor new];
                paths[k].depthWriteEnabled = NO;
                paths[k].depthCompareFunction = k == 1 ? MTLCompareFunctionNever : MTLCompareFunctionAlways;
                MTLStencilDescriptor *s = DSStencil(k == 0 ? MTLCompareFunctionNever : MTLCompareFunctionAlways,
                                                    k == 0 ? operation : MTLStencilOperationKeep,
                                                    k == 1 ? operation : MTLStencilOperationKeep,
                                                    k == 2 ? operation : MTLStencilOperationKeep, 0xff, 0xf7);
                paths[k].frontFaceStencil = s;
                paths[k].backFaceStencil = s;
            }
            NSArray<id<MTLDepthStencilState>> *states = @[[c depthStencilState:paths[0]],
                                                          [c depthStencilState:paths[1]],
                                                          [c depthStencilState:paths[2]]];
            id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, DSW, DSH);
            id<MTLTexture> depthStencil = RTarget(c, MTLPixelFormatDepth32Float_Stencil8, DSW, DSH);
            MTLRenderPassDescriptor *pass = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
            pass.depthAttachment.texture = depthStencil;
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.storeAction = MTLStoreActionStore;
            pass.depthAttachment.clearDepth = 0.5;
            pass.stencilAttachment.texture = depthStencil;
            pass.stencilAttachment.loadAction = MTLLoadActionClear;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
            pass.stencilAttachment.clearStencil = clears[clearIndex];
            [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
                [e setRenderPipelineState:pipeline];
                [e setVertexBuffer:cover offset:0 atIndex:0];
                [e setStencilReferenceValue:0x3c];
                for (int k = 0; k < 3; k++) {
                    [e setDepthStencilState:states[k]];
                    [e setScissorRect:(MTLScissorRect){(NSUInteger)k * 5, 0, k == 2 ? 6 : 5, DSH}];
                    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                }
            }];
            NSString *prefix = [NSString stringWithFormat:@"%@.clear-%02x", names[op], clears[clearIndex]];
            DSRecordAspect(c, [prefix stringByAppendingString:@".stencil"], depthStencil,
                           MTLBlitOptionStencilFromDepthStencil);
            RRecordTexture(c, [prefix stringByAppendingString:@".color"], color, WMElemU32, 4, WMExact());
        }
}

WMREF_CASE(dsStencilReadMask, "depth-stencil.stencil-read-mask", "depth-stencil.stencil",
           "Stencil compare with read masks: (reference & readMask) against (stored & readMask)")
{
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id",
                                                         MTLPixelFormatR32Uint);
    d.stencilAttachmentPixelFormat = MTLPixelFormatStencil8;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    id<MTLBuffer> cover = DSCover(c, 0.5, 1);
    // Stored 0x5a, reference 0x4a: equal under read mask 0xef, different
    // under 0xff; 0x5a > 0x4a under 0xff for the greater test.
    const struct { MTLCompareFunction f; uint32_t mask; } tests[4] = {
        {MTLCompareFunctionEqual, 0xef}, {MTLCompareFunctionEqual, 0xff},
        {MTLCompareFunctionLess, 0xff}, {MTLCompareFunctionGreater, 0xff},
    };
    NSMutableArray<id<MTLDepthStencilState>> *states = [NSMutableArray array];
    for (int k = 0; k < 4; k++) {
        MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
        MTLStencilDescriptor *s = DSStencil(tests[k].f, MTLStencilOperationKeep, MTLStencilOperationKeep,
                                            MTLStencilOperationKeep, tests[k].mask, 0xff);
        ds.frontFaceStencil = s;
        ds.backFaceStencil = s;
        [states addObject:[c depthStencilState:ds]];
    }
    id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, DSW, DSH);
    id<MTLTexture> stencil = RTarget(c, MTLPixelFormatStencil8, DSW, DSH);
    MTLRenderPassDescriptor *pass = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
    pass.stencilAttachment.texture = stencil;
    pass.stencilAttachment.loadAction = MTLLoadActionClear;
    pass.stencilAttachment.storeAction = MTLStoreActionStore;
    pass.stencilAttachment.clearStencil = 0x5a;
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:cover offset:0 atIndex:0];
        [e setStencilReferenceValue:0x4a];
        for (int k = 0; k < 4; k++) {
            [e setDepthStencilState:states[k]];
            [e setScissorRect:(MTLScissorRect){(NSUInteger)k * 4, 0, 4, DSH}];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
    }];
    RRecordTexture(c, @"color", color, WMElemU32, 4, WMExact());
    RRecordTexture(c, @"stencil", stencil, WMElemU8, 1, WMExact());
}

WMREF_CASE(dsSeparateFrontBack, "depth-stencil.separate-front-back", "depth-stencil.stencil",
           "Separate front and back stencil reference values and stencil descriptors")
{
    enum { W = 16, H = 8 };
    // Left triangle clockwise, right triangle counter-clockwise on screen;
    // every pixel centre is at least 1/8 pixel from their edges.
    const double tris[2][6] = {{0.25, 0.25, 7.5, 0.25, 0.25, 7.5}, {8.25, 0.25, 8.25, 7.5, 15.5, 0.25}};
    RVertex v[6];
    for (int t = 0; t < 2; t++)
        for (int k = 0; k < 3; k++) v[t * 3 + k] = RVertexAt(tris[t][2 * k], tris[t][2 * k + 1], 0.5, W, H, (uint32_t)t + 1);
    id<MTLBuffer> buffer = RVertexBuffer(c, v, 6);
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_facing",
                                                         MTLPixelFormatR32Uint);
    d.stencilAttachmentPixelFormat = MTLPixelFormatStencil8;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];

    // Pass 1: compare equal against separate references 1 (front) and 2
    // (back) on a stencil cleared to 1: only front-facing fragments pass.
    // Pass 2: the same on a stencil cleared to 2: only back-facing pass.
    // Pass 3: always pass with front operation replace and back operation
    // invert, references 0x21 and 0x42.
    MTLDepthStencilDescriptor *equal = [MTLDepthStencilDescriptor new];
    equal.frontFaceStencil = DSStencil(MTLCompareFunctionEqual, MTLStencilOperationKeep, MTLStencilOperationKeep,
                                       MTLStencilOperationKeep, 0xff, 0xff);
    equal.backFaceStencil = equal.frontFaceStencil;
    MTLDepthStencilDescriptor *split = [MTLDepthStencilDescriptor new];
    split.frontFaceStencil = DSStencil(MTLCompareFunctionAlways, MTLStencilOperationKeep, MTLStencilOperationKeep,
                                       MTLStencilOperationReplace, 0xff, 0xff);
    split.backFaceStencil = DSStencil(MTLCompareFunctionAlways, MTLStencilOperationKeep, MTLStencilOperationKeep,
                                      MTLStencilOperationInvert, 0xff, 0xff);
    const struct { __unsafe_unretained MTLDepthStencilDescriptor *state; uint8_t clear; uint32_t front, back;
                   __unsafe_unretained NSString *name; } passes[3] = {
        {equal, 1, 1, 2, @"equal-clear-1"}, {equal, 2, 1, 2, @"equal-clear-2"},
        {split, 0x0f, 0x21, 0x42, @"replace-front-invert-back"},
    };
    for (int p = 0; p < 3; p++) {
        id<MTLDepthStencilState> state = [c depthStencilState:passes[p].state];
        id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, W, H);
        id<MTLTexture> stencil = RTarget(c, MTLPixelFormatStencil8, W, H);
        MTLRenderPassDescriptor *pass = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
        pass.stencilAttachment.texture = stencil;
        pass.stencilAttachment.loadAction = MTLLoadActionClear;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
        pass.stencilAttachment.clearStencil = passes[p].clear;
        uint32_t front = passes[p].front, back = passes[p].back;
        [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:pipeline];
            [e setDepthStencilState:state];
            [e setFrontFacingWinding:MTLWindingClockwise];
            [e setStencilFrontReferenceValue:front backReferenceValue:back];
            [e setVertexBuffer:buffer offset:0 atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
        }];
        RRecordTexture(c, [passes[p].name stringByAppendingString:@".color"], color, WMElemU32, 4, WMExact());
        RRecordTexture(c, [passes[p].name stringByAppendingString:@".stencil"], stencil, WMElemU8, 1, WMExact());
    }
}

WMREF_CASE(dsDepthOutput, "depth-stencil.fragment-depth", "depth-stencil.depth-output",
           "Fragment [[depth(any)]], [[depth(greater)]] and [[depth(less)]] outputs written to Depth32Float")
{
    // The written depths are k/256 or z +/- x/64, exact in binary32; a
    // Depth32Float attachment stores the value written (MSL 4.1 section
    // 8.7.3: values other than NaN and denormals are preserved).
    NSString *const fragments[3] = {@"rnd_fragment_depth_any", @"rnd_fragment_depth_greater", @"rnd_fragment_depth_less"};
    const double zs[3] = {0.5, 0.25, 0.75};
    NSString *const names[3] = {@"any", @"greater", @"less"};
    MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
    ds.depthCompareFunction = MTLCompareFunctionAlways;
    ds.depthWriteEnabled = YES;
    id<MTLDepthStencilState> state = [c depthStencilState:ds];
    for (int k = 0; k < 3; k++) {
        MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", fragments[k],
                                                             MTLPixelFormatR32Uint);
        d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
        id<MTLBuffer> cover = DSCover(c, zs[k], 1);
        id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, DSW, DSH);
        id<MTLTexture> depth = RTarget(c, MTLPixelFormatDepth32Float, DSW, DSH);
        MTLRenderPassDescriptor *pass = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
        pass.depthAttachment.texture = depth;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        pass.depthAttachment.clearDepth = 1.0;
        [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:pipeline];
            [e setDepthStencilState:state];
            [e setVertexBuffer:cover offset:0 atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }];
        RRecordTexture(c, [names[k] stringByAppendingString:@".depth"], depth, WMElemF32, 4, WMExact());
    }
}

WMREF_CASE(dsDepthBias, "depth-stencil.depth-bias", "depth-stencil.depth-bias",
           "setDepthBias:slopeScale:clamp: on Depth32Float and Depth16Unorm, flat and sloped triangles")
{
    enum { W = 16, H = 16 };
    MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
    ds.depthCompareFunction = MTLCompareFunctionAlways;
    ds.depthWriteEnabled = YES;
    id<MTLDepthStencilState> state = [c depthStencilState:ds];
    // Flat: z = 0.5 everywhere. Sloped: z from 0.25 at x = 0 to 0.75 at x = 16.
    RVertex flat[3] = {RVertexAt(-1, -1, 0.5, W, H, 1), RVertexAt(48, -1, 0.5, W, H, 1), RVertexAt(-1, 48, 0.5, W, H, 1)};
    RVertex sloped[3] = {RVertexAt(0, -16, 0.25, W, H, 2), RVertexAt(48, -16, 1.75, W, H, 2),
                         RVertexAt(0, 48, 0.25, W, H, 2)};
    id<MTLBuffer> flatBuffer = RVertexBuffer(c, flat, 3), slopedBuffer = RVertexBuffer(c, sloped, 3);
    const MTLPixelFormat formats[2] = {MTLPixelFormatDepth32Float, MTLPixelFormatDepth16Unorm};
    NSString *const formatNames[2] = {@"depth32float", @"depth16unorm"};
    const struct { float bias, slope, clamp; __unsafe_unretained NSString *name; } settings[4] = {
        {0, 0, 0, @"none"}, {64, 0, 0, @"constant-64"}, {0, 2, 0, @"slope-2"}, {1000, 0, 0.001f, @"clamped"},
    };
    NSString *reason = @"Metal's documentation does not define the depth bias formula or its units per format";
    for (int f = 0; f < 2; f++) {
        MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id",
                                                             MTLPixelFormatR32Uint);
        d.depthAttachmentPixelFormat = formats[f];
        id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
        for (int s = 0; s < 4; s++)
            for (int g = 0; g < 2; g++) {
                id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, W, H);
                id<MTLTexture> depth = RTarget(c, formats[f], W, H);
                MTLRenderPassDescriptor *pass = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
                pass.depthAttachment.texture = depth;
                pass.depthAttachment.loadAction = MTLLoadActionClear;
                pass.depthAttachment.storeAction = MTLStoreActionStore;
                pass.depthAttachment.clearDepth = 1.0;
                float bias = settings[s].bias, slope = settings[s].slope, clamp = settings[s].clamp;
                id<MTLBuffer> geometry = g ? slopedBuffer : flatBuffer;
                [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
                    [e setRenderPipelineState:pipeline];
                    [e setDepthStencilState:state];
                    [e setDepthBias:bias slopeScale:slope clamp:clamp];
                    [e setVertexBuffer:geometry offset:0 atIndex:0];
                    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                }];
                NSString *name = [NSString stringWithFormat:@"%@.%@.%@", formatNames[f], g ? @"sloped" : @"flat",
                                  settings[s].name];
                // Without bias, a constant depth of 0.5 stored in Depth32Float
                // is exact; Depth16Unorm conversion and interpolated depth are
                // not specified to a precision.
                BOOL exact = f == 0 && g == 0 && s == 0;
                RRecordTexture(c, name, depth, f ? WMElemU16 : WMElemF32, f ? 2 : 4,
                               exact ? WMExact()
                                     : WMNone(s == 0 ? @"depth interpolation and Depth16Unorm conversion precision "
                                                       @"are not stated by Metal"
                                                     : reason));
            }
    }
}

WMREF_CASE(dsNoDepthAttachment, "depth-stencil.error.test-without-attachment", "depth-stencil.depth-test",
           "Depth-stencil state with a depth test bound in a pass without a depth attachment (invalid use)")
{
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id",
                                                         MTLPixelFormatR32Uint);
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
    ds.depthCompareFunction = MTLCompareFunctionLess;
    ds.depthWriteEnabled = YES;
    id<MTLDepthStencilState> state = [c depthStencilState:ds];
    id<MTLBuffer> cover = DSCover(c, 0.5, 1);
    id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, DSW, DSH);
    [c step:@"draw"];
    [c render:RPassColor(color, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setDepthStencilState:state];
        [e setVertexBuffer:cover offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    [c step:@"completed"];
    RRecordTexture(c, @"color", color, WMElemU32, 4,
                   WMNone(@"the result of a depth test without a depth attachment is not defined"));
}
