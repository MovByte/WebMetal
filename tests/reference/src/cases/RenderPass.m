// Render pass cases: load and store actions, clear values, deferred store
// actions, layered rendering, parallel render encoders, memoryless
// attachments, visibility results, tessellation, and invalid use.

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

static id<MTLBuffer> PCover(WMCase *c, NSUInteger width, NSUInteger height, double z, uint32_t id) {
    RVertex v[3] = {RVertexAt(-1, -1, z, width, height, id), RVertexAt(3.0 * width, -1, z, width, height, id),
                    RVertexAt(-1, 3.0 * height, z, width, height, id)};
    return RVertexBuffer(c, v, 3);
}

static id<MTLRenderPipelineState> PIdPipeline(WMCase *c) {
    return [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id", MTLPixelFormatR32Uint)];
}

WMREF_CASE(passLoadActions, "pass.load-store-actions", "pass.load-store",
           "Load actions load, clear and dontCare; store actions store and dontCare")
{
    id<MTLRenderPipelineState> pipeline = PIdPipeline(c);
    id<MTLBuffer> cover = PCover(c, 8, 8, 0, 2);
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, 8, 8);
    // Pass 1 clears to 5 and stores. Pass 2 loads and draws id 2 into a
    // 3x2 scissor rectangle: the rest keeps 5.
    [c render:RPassColor(target, MTLClearColorMake(5, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {}];
    MTLRenderPassDescriptor *load = RPassColor(target, MTLClearColorMake(0, 0, 0, 0));
    load.colorAttachments[0].loadAction = MTLLoadActionLoad;
    [c render:load body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setScissorRect:(MTLScissorRect){2, 3, 3, 2}];
        [e setVertexBuffer:cover offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    RRecordTexture(c, @"load", target, WMElemU32, 4, WMExact());

    // dontCare load: pixels outside the drawn rectangle are undefined at the
    // start of the pass (Apple, "Setting load and store actions").
    MTLRenderPassDescriptor *dontCare = RPassColor(target, MTLClearColorMake(0, 0, 0, 0));
    dontCare.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    [c render:dontCare body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setScissorRect:(MTLScissorRect){0, 0, 4, 4}];
        [e setVertexBuffer:cover offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    NSData *image = RRead(c, target, 4);
    uint32_t drawn[16];
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) drawn[y * 4 + x] = ((const uint32_t *)image.bytes)[y * 8 + x];
    [c data:@"dont-care-load.drawn-rectangle" elem:WMElemU32 bytes:drawn length:sizeof drawn rule:WMExact()];
    [c data:@"dont-care-load.all" elem:WMElemU32 bytes:image.bytes length:image.length
       rule:WMNone(@"pixels not drawn after a dontCare load action are undefined")];

    // dontCare store: the attachment contents after the pass are undefined.
    MTLRenderPassDescriptor *discard = RPassColor(target, MTLClearColorMake(7, 0, 0, 0));
    discard.colorAttachments[0].storeAction = MTLStoreActionDontCare;
    [c render:discard body:^(id<MTLRenderCommandEncoder> e) {}];
    RRecordTexture(c, @"dont-care-store", target, WMElemU32, 4,
                   WMNone(@"contents after a dontCare store action are undefined"));
}

WMREF_CASE(passClearValues, "pass.clear-values", "pass.load-store",
           "Clear colours, depth and stencil values that are exactly representable in each attachment format")
{
    // Every clear value is representable in its format, so the stored value
    // is the clear value.
    const struct { MTLPixelFormat format; MTLClearColor clear; NSUInteger bpp; WMElem elem;
                   __unsafe_unretained NSString *name; } colors[] = {
        {MTLPixelFormatRGBA8Unorm, {0, 1, 1, 0}, 4, WMElemU8, @"rgba8unorm"},
        {MTLPixelFormatRGBA16Float, {0.375, -2.5, 65504, 0.0009765625}, 8, WMElemF16, @"rgba16float"},
        {MTLPixelFormatRGBA32Float, {0.375, -2.5, 1048576.25, 1}, 16, WMElemF32, @"rgba32float"},
        {MTLPixelFormatRGBA32Uint, {0, 1, 4294967295.0, 123456789}, 16, WMElemU32, @"rgba32uint"},
        {MTLPixelFormatRGBA16Sint, {-32768, 32767, -1, 0}, 8, WMElemI16, @"rgba16sint"},
    };
    for (size_t i = 0; i < sizeof colors / sizeof colors[0]; i++) {
        id<MTLTexture> target = RTarget(c, colors[i].format, 4, 4);
        [c render:RPassColor(target, colors[i].clear) body:^(id<MTLRenderCommandEncoder> e) {}];
        RRecordTexture(c, colors[i].name, target, colors[i].elem, colors[i].bpp, WMExact());
    }
    id<MTLTexture> depthStencil = RTarget(c, MTLPixelFormatDepth32Float_Stencil8, 4, 4);
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.depthAttachment.texture = depthStencil;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
    pass.depthAttachment.clearDepth = 0.375;
    pass.stencilAttachment.texture = depthStencil;
    pass.stencilAttachment.loadAction = MTLLoadActionClear;
    pass.stencilAttachment.storeAction = MTLStoreActionStore;
    pass.stencilAttachment.clearStencil = 0xa5;
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {}];
    id<MTLBuffer> out = [c buffer:64 + 16];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromTexture:depthStencil sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(4, 4, 1) toBuffer:out destinationOffset:0 destinationBytesPerRow:16
     destinationBytesPerImage:64 options:MTLBlitOptionDepthFromDepthStencil];
        [e copyFromTexture:depthStencil sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(4, 4, 1) toBuffer:out destinationOffset:64 destinationBytesPerRow:4
     destinationBytesPerImage:16 options:MTLBlitOptionStencilFromDepthStencil];
    }];
    [c data:@"depth" elem:WMElemF32 bytes:out.contents length:64 rule:WMExact()];
    [c data:@"stencil" elem:WMElemU8 bytes:(const uint8_t *)out.contents + 64 length:16 rule:WMExact()];
}

WMREF_CASE(passStoreActionUnknown, "pass.store-action-deferred", "pass.load-store",
           "MTLStoreActionUnknown finalized with setColorStoreAction:atIndex: before endEncoding")
{
    id<MTLRenderPipelineState> pipeline = PIdPipeline(c);
    id<MTLBuffer> cover = PCover(c, 4, 4, 0, 3);
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, 4, 4);
    MTLRenderPassDescriptor *pass = RPassColor(target, MTLClearColorMake(0, 0, 0, 0));
    pass.colorAttachments[0].storeAction = MTLStoreActionUnknown;
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:cover offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e setColorStoreAction:MTLStoreActionStore atIndex:0];
    }];
    RRecordTexture(c, @"result", target, WMElemU32, 4, WMExact());
}

WMREF_CASE(passStoreActionUnset, "pass.error.store-action-unknown-unset", "pass.load-store",
           "MTLStoreActionUnknown never finalized before endEncoding (invalid use)")
{
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, 4, 4);
    MTLRenderPassDescriptor *pass = RPassColor(target, MTLClearColorMake(6, 0, 0, 0));
    pass.colorAttachments[0].storeAction = MTLStoreActionUnknown;
    [c step:@"encode"];
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {}];
    [c step:@"completed"];
    RRecordTexture(c, @"result", target, WMElemU32, 4,
                   WMNone(@"the store action applied when MTLStoreActionUnknown is never finalized is not defined"));
}

WMREF_CASE(passLayered, "pass.layered-rendering", "pass.layered",
           "renderTargetArrayLength with [[render_target_array_index]], including an out-of-bounds index")
{
    // "Layered rendering": Apple5 and later (Metal Feature Set Tables, 2026-10-23).
    [c require:@"family.apple5" value:[c.device supportsFamily:MTLGPUFamilyApple5]];
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_layered", @"rnd_fragment_layered",
                                                         MTLPixelFormatR32Uint);
    d.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                                  width:4 height:4 mipmapped:NO];
    td.textureType = MTLTextureType2DArray;
    td.arrayLength = 3;
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> target = [c texture:td];
    MTLRenderPassDescriptor *pass = RPassColor(target, MTLClearColorMake(0, 0, 0, 0));
    pass.renderTargetArrayLength = 3;
    id<MTLBuffer> cover = PCover(c, 4, 4, 0, 0);
    // Instances 0..2 render ids 1..3 to layers 0..2; instance 3 selects
    // layer 7, which MSL 4.1 section 5.2.3.3 states "the hardware interprets
    // as 0", and renders id 4 there after instance 0.
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:cover offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:4];
    }];
    for (NSUInteger slice = 0; slice < 3; slice++) {
        NSData *data = [c readTexture:target level:0 slice:slice bytesPerRow:16];
        [c data:[NSString stringWithFormat:@"layer-%lu", (unsigned long)slice] elem:WMElemU32 bytes:data.bytes
         length:data.length rule:WMExact()];
    }
}

WMREF_CASE(passParallelEncoder, "pass.parallel-encoder-order", "pass.parallel-encoder",
           "Parallel render encoder: sub-encoders execute in creation order, not encoding or ending order")
{
    // Apple's documentation of -[MTLParallelRenderCommandEncoder
    // renderCommandEncoder]: "The rendering commands ... are executed in the
    // order in which the MTLRenderCommandEncoder objects are created, not in
    // the order they are ended."
    id<MTLRenderPipelineState> pipeline = PIdPipeline(c);
    id<MTLBuffer> first = PCover(c, 8, 8, 0, 1), second = PCover(c, 8, 8, 0, 2);
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, 8, 8);
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLParallelRenderCommandEncoder> parallel = [commandBuffer parallelRenderCommandEncoderWithDescriptor:
        RPassColor(target, MTLClearColorMake(0, 0, 0, 0))];
    id<MTLRenderCommandEncoder> a = [parallel renderCommandEncoder];
    id<MTLRenderCommandEncoder> b = [parallel renderCommandEncoder];
    // Encode b first: id 2 into the left half. Then a: id 1 everywhere.
    [b setRenderPipelineState:pipeline];
    [b setScissorRect:(MTLScissorRect){0, 0, 4, 8}];
    [b setVertexBuffer:second offset:0 atIndex:0];
    [b drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [b endEncoding];
    [a setRenderPipelineState:pipeline];
    [a setVertexBuffer:first offset:0 atIndex:0];
    [a drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [a endEncoding];
    [parallel endEncoding];
    [c commitAndWait:commandBuffer];
    // a runs before b: left half 2, right half 1.
    RRecordTexture(c, @"result", target, WMElemU32, 4, WMExact());
}

static void PMemorylessDraw(WMCase *c, MTLStoreAction depthStore) {
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id",
                                                         MTLPixelFormatR32Uint);
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
    ds.depthCompareFunction = MTLCompareFunctionLess;
    ds.depthWriteEnabled = YES;
    id<MTLDepthStencilState> state = [c depthStencilState:ds];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                  width:8 height:8 mipmapped:NO];
    td.storageMode = MTLStorageModeMemoryless;
    td.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> depth = [c texture:td];
    id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, 8, 8);
    MTLRenderPassDescriptor *pass = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
    pass.depthAttachment.texture = depth;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.clearDepth = 1.0;
    pass.depthAttachment.storeAction = depthStore;
    // Near quad (id 1, z 0.25) on the left 5 columns, then a far triangle
    // (id 2, z 0.5) everywhere: the depth test keeps id 1 on the left.
    id<MTLBuffer> near = PCover(c, 8, 8, 0.25, 1), far = PCover(c, 8, 8, 0.5, 2);
    [c step:@"draw"];
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setDepthStencilState:state];
        [e setScissorRect:(MTLScissorRect){0, 0, 5, 8}];
        [e setVertexBuffer:near offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e setScissorRect:(MTLScissorRect){0, 0, 8, 8}];
        [e setVertexBuffer:far offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    [c step:@"completed"];
    RRecordTexture(c, @"color", color, WMElemU32, 4,
                   depthStore == MTLStoreActionDontCare
                       ? WMExact()
                       : WMNone(@"a storing action on a memoryless attachment is invalid; its result is not defined"));
}

// Apple's documentation of MTLStorageMode.memoryless: "only available on
// Apple family GPUs"; memoryless targets "cannot be accessed with
// MTLLoadAction.load or MTLStoreAction.store".
WMREF_CASE(passMemorylessDepth, "pass.memoryless-depth", "pass.memoryless",
           "Memoryless depth attachment cleared and discarded within one pass")
{
    // "Memoryless render targets": Apple2 and later (Metal Feature Set Tables, 2026-10-23).
    [c require:@"family.apple2" value:[c.device supportsFamily:MTLGPUFamilyApple2]];
    PMemorylessDraw(c, MTLStoreActionDontCare);
}

WMREF_CASE(passMemorylessStore, "pass.error.memoryless-store", "pass.memoryless",
           "Memoryless depth attachment with MTLStoreActionStore (invalid use)")
{
    // "Memoryless render targets": Apple2 and later (Metal Feature Set Tables, 2026-10-23).
    [c require:@"family.apple2" value:[c.device supportsFamily:MTLGPUFamilyApple2]];
    PMemorylessDraw(c, MTLStoreActionStore);
}

WMREF_CASE(passVisibility, "pass.visibility-results", "pass.visibility",
           "Visibility result modes boolean and counting at 1x and 4x, results at buffer offsets")
{
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id",
                                                         MTLPixelFormatR32Uint);
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    MTLDepthStencilDescriptor *pass = [MTLDepthStencilDescriptor new];
    pass.depthCompareFunction = MTLCompareFunctionAlways;
    MTLDepthStencilDescriptor *fail = [MTLDepthStencilDescriptor new];
    fail.depthCompareFunction = MTLCompareFunctionNever;
    id<MTLDepthStencilState> passing = [c depthStencilState:pass], failing = [c depthStencilState:fail];
    id<MTLBuffer> cover = PCover(c, 8, 8, 0.5, 1);
    id<MTLBuffer> results = [c buffer:6 * 8];
    id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, 8, 8);
    id<MTLTexture> depth = RTarget(c, MTLPixelFormatDepth32Float, 8, 8);
    MTLRenderPassDescriptor *descriptor = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
    descriptor.depthAttachment.texture = depth;
    descriptor.depthAttachment.loadAction = MTLLoadActionClear;
    descriptor.depthAttachment.storeAction = MTLStoreActionDontCare;
    descriptor.visibilityResultBuffer = results;
    [c render:descriptor body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:cover offset:0 atIndex:0];
        // Offset 0: boolean, 12 pixels pass. Offset 8: boolean, none pass.
        // Offset 16: counting, 12 pixels pass. Offset 24: counting, none.
        // Offset 32: counting, 12 pixels pass, then offset 40: counting,
        // 20 pixels pass. (The debug layer rejects reusing an offset within
        // one encoder.)
        const struct { MTLVisibilityResultMode mode; NSUInteger offset; BOOL passes; MTLScissorRect rect; } q[] = {
            {MTLVisibilityResultModeBoolean, 0, YES, {0, 0, 4, 3}},
            {MTLVisibilityResultModeBoolean, 8, NO, {0, 0, 8, 8}},
            {MTLVisibilityResultModeCounting, 16, YES, {1, 1, 3, 4}},
            {MTLVisibilityResultModeCounting, 24, NO, {0, 0, 8, 8}},
            {MTLVisibilityResultModeCounting, 32, YES, {0, 0, 6, 2}},
            {MTLVisibilityResultModeCounting, 40, YES, {2, 3, 5, 4}},
        };
        for (size_t i = 0; i < sizeof q / sizeof q[0]; i++) {
            [e setVisibilityResultMode:q[i].mode offset:q[i].offset];
            [e setDepthStencilState:q[i].passes ? passing : failing];
            [e setScissorRect:q[i].rect];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        [e setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    }];
    const uint64_t *r = results.contents;
    // MTLVisibilityResultMode.boolean: "a nonzero value if at least one
    // fragment passed depth and stencil tests, and zero otherwise".
    [c value:@"boolean.passing.value" u64:r[0] rule:WMNone(@"Metal documents only that the value is nonzero")];
    [c value:@"boolean.passing.nonzero" boolean:r[0] != 0 rule:WMExact()];
    [c value:@"boolean.failing" u64:r[1] rule:WMExact()];
    // MTLVisibilityResultMode.counting: "how many samples passed".
    [c value:@"counting.passing" u64:r[2] rule:WMExact()];
    [c value:@"counting.failing" u64:r[3] rule:WMExact()];
    [c value:@"counting.offset-32" u64:r[4] rule:WMExact()];
    [c value:@"counting.offset-40" u64:r[5] rule:WMExact()];

    // Counting at 4x: every sample of 6 fully covered pixels passes.
    [c require:@"device.texture-sample-count-4" value:[c.device supportsTextureSampleCount:4]];
    MTLRenderPipelineDescriptor *d4 = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id",
                                                          MTLPixelFormatR32Uint);
    d4.rasterSampleCount = 4;
    id<MTLRenderPipelineState> pipeline4 = [c renderPipeline:d4];
    id<MTLBuffer> results4 = [c buffer:8];
    id<MTLTexture> color4 = RTargetSampled(c, MTLPixelFormatR32Uint, 8, 8, 4);
    MTLRenderPassDescriptor *descriptor4 = RPassColor(color4, MTLClearColorMake(0, 0, 0, 0));
    descriptor4.colorAttachments[0].storeAction = MTLStoreActionDontCare;
    descriptor4.visibilityResultBuffer = results4;
    [c render:descriptor4 body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline4];
        [e setVertexBuffer:cover offset:0 atIndex:0];
        [e setVisibilityResultMode:MTLVisibilityResultModeCounting offset:0];
        [e setScissorRect:(MTLScissorRect){2, 2, 3, 2}];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    }];
    [c value:@"counting.4x" u64:*(const uint64_t *)results4.contents rule:WMExact()];
}

WMREF_CASE(passVisibilityUnaligned, "pass.error.visibility-offset-unaligned", "pass.visibility",
           "setVisibilityResultMode:offset: with an offset that is not a multiple of 8 (invalid use)")
{
    id<MTLRenderPipelineState> pipeline = PIdPipeline(c);
    id<MTLBuffer> cover = PCover(c, 4, 4, 0, 1);
    id<MTLBuffer> results = [c buffer:32];
    id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, 4, 4);
    MTLRenderPassDescriptor *descriptor = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
    descriptor.visibilityResultBuffer = results;
    [c step:@"encode"];
    [c render:descriptor body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:cover offset:0 atIndex:0];
        [e setVisibilityResultMode:MTLVisibilityResultModeCounting offset:4];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    [c step:@"completed"];
    [c data:@"results" elem:WMElemU8 bytes:results.contents length:results.length
       rule:WMNone(@"the result of an unaligned visibility offset is not defined")];
}

WMREF_CASE(passVisibilityReused, "pass.error.visibility-offset-reused", "pass.visibility",
           "Counting visibility results at the same offset for two draws in one encoder (invalid use)")
{
    id<MTLRenderPipelineState> pipeline = PIdPipeline(c);
    id<MTLBuffer> cover = PCover(c, 8, 8, 0, 1);
    id<MTLBuffer> results = [c buffer:16];
    id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, 8, 8);
    MTLRenderPassDescriptor *descriptor = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
    descriptor.visibilityResultBuffer = results;
    [c step:@"encode"];
    [c render:descriptor body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:cover offset:0 atIndex:0];
        [e setVisibilityResultMode:MTLVisibilityResultModeCounting offset:0];
        [e setScissorRect:(MTLScissorRect){0, 0, 6, 2}];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e setVisibilityResultMode:MTLVisibilityResultModeCounting offset:8];
        [e setScissorRect:(MTLScissorRect){0, 2, 8, 1}];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e setVisibilityResultMode:MTLVisibilityResultModeCounting offset:0];
        [e setScissorRect:(MTLScissorRect){2, 3, 5, 4}];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    }];
    [c step:@"completed"];
    const uint64_t *r = results.contents;
    [c value:@"offset-0" u64:r[0] rule:WMNone(@"reusing a visibility offset within an encoder is invalid")];
    [c value:@"offset-8" u64:r[1] rule:WMNone(@"reusing a visibility offset within an encoder is invalid")];
}

#pragma mark Tessellation

static void PTessellate(WMCase *c, BOOL quad, MTLTessellationPartitionMode mode, const float *factors,
                        NSString *name) {
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [c function:quad ? @"rnd_tess_quad" : @"rnd_tess_triangle"];
    d.rasterizationEnabled = NO;
    d.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
    d.tessellationPartitionMode = mode;
    d.maxTessellationFactor = 16;
    d.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
    d.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionConstant;
    d.tessellationOutputWindingOrder = MTLWindingClockwise;
    d.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeNone;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    // MTLTriangleTessellationFactorsHalf: edge[3], inside; MTLQuadTessellationFactorsHalf:
    // edge[4], inside[2].
    uint16_t halves[6];
    NSUInteger count = quad ? 6 : 4;
    for (NSUInteger i = 0; i < count; i++) halves[i] = WMHalfFromFloat(factors[i]);
    id<MTLBuffer> factorBuffer = [c bufferWithBytes:halves length:sizeof halves];
    id<MTLBuffer> counter = [c buffer:4];
    id<MTLBuffer> points = [c buffer:1024 * 16];
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, 4, 4);
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setTessellationFactorBuffer:factorBuffer offset:0 instanceStride:0];
        [e setVertexBuffer:counter offset:0 atIndex:0];
        [e setVertexBuffer:points offset:0 atIndex:1];
        [e drawPatches:quad ? 4 : 3 patchStart:0 patchCount:1 patchIndexBuffer:nil patchIndexBufferOffset:0
         instanceCount:1 baseInstance:0];
    }];
    uint32_t invocations = *(const uint32_t *)counter.contents;
    if (invocations > 1024) [c fail:@"%u post-tessellation invocations exceed the capture buffer", invocations];
    // The set of distinct domain positions, sorted by bit pattern. The number
    // of invocations includes vertex reuse, which is implementation-defined.
    NSMutableArray<NSData *> *unique = [NSMutableArray array];
    NSMutableSet<NSData *> *seen = [NSMutableSet set];
    const float *p = points.contents;
    for (uint32_t i = 0; i < invocations; i++) {
        NSData *key = [NSData dataWithBytes:p + 4 * i length:12];
        if ([seen containsObject:key]) continue;
        [seen addObject:key];
        [unique addObject:key];
    }
    [unique sortUsingComparator:^NSComparisonResult(NSData *x, NSData *y) {
        int r = memcmp(x.bytes, y.bytes, 12);
        return r < 0 ? NSOrderedAscending : r > 0 ? NSOrderedDescending : NSOrderedSame;
    }];
    NSMutableData *sorted = [NSMutableData data];
    for (NSData *key in unique) [sorted appendData:key];
    NSString *reason = @"Metal's tessellation documentation describes the partitioning but not the binary32 values "
                       @"of the generated positions";
    [c value:[name stringByAppendingString:@".invocations"] u64:invocations
        rule:WMNone(@"post-tessellation vertex reuse is implementation-defined")];
    [c value:[name stringByAppendingString:@".distinct-positions"] u64:unique.count rule:WMNone(reason)];
    [c data:[name stringByAppendingString:@".positions"] elem:WMElemF32 bytes:sorted.bytes length:sorted.length
       rule:WMNone(reason)];
}

WMREF_CASE(passTessellation, "pass.tessellation", "pass.tessellation",
           "drawPatches for triangle and quad patches in every partition mode: generated domain positions")
{
    // "Tessellation": Apple3 and later, Metal 3 programming model (Metal Feature Set Tables,
    // 2026-10-23).
    [c require:@"family.apple3" value:[c.device supportsFamily:MTLGPUFamilyApple3]];
    const MTLTessellationPartitionMode modes[4] = {
        MTLTessellationPartitionModePow2, MTLTessellationPartitionModeInteger,
        MTLTessellationPartitionModeFractionalOdd, MTLTessellationPartitionModeFractionalEven};
    NSString *const modeNames[4] = {@"pow2", @"integer", @"fractional-odd", @"fractional-even"};
    const float triangleFactors[3][4] = {{1, 1, 1, 1}, {3, 3, 3, 3}, {2, 3, 4, 2.5f}};
    const float quadFactors[2][6] = {{2, 2, 2, 2, 2, 2}, {1, 2, 3, 4, 3.5f, 2}};
    for (int m = 0; m < 4; m++) {
        for (int f = 0; f < 3; f++)
            PTessellate(c, NO, modes[m], triangleFactors[f],
                        [NSString stringWithFormat:@"triangle.%@.factors-%d", modeNames[m], f]);
        for (int f = 0; f < 2; f++)
            PTessellate(c, YES, modes[m], quadFactors[f],
                        [NSString stringWithFormat:@"quad.%@.factors-%d", modeNames[m], f]);
    }
}
