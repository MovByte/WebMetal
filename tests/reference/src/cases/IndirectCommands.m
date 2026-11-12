// Indirect command buffer cases: compute and render commands encoded on
// the CPU and executed by an encoder, reset, and the blit operations that
// copy, optimize and reset indirect command ranges.

#import "WMRefCase.h"
#import "LibrariesSupport.h"

#pragma mark - Compute

static id<MTLComputePipelineState> WMICBComputePipeline(WMCase *c) {
    MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = [c function:@"icb_fill"];
    descriptor.supportIndirectCommandBuffers = YES;
    NSError *error = nil;
    id<MTLComputePipelineState> state =
        [c.device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil
                                                  error:&error];
    if (!state) [c fail:@"ICB compute pipeline: %@", error];
    return state;
}

// Three dispatches, each filling 16 words of `out` at its own offset with
// value + index.
static id<MTLIndirectCommandBuffer> WMICBCompute(WMCase *c, id<MTLComputePipelineState> pipeline, id<MTLBuffer> out,
                                                 id<MTLBuffer> values) {
    MTLIndirectCommandBufferDescriptor *descriptor = [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    descriptor.inheritBuffers = NO;
    descriptor.inheritPipelineState = NO;
    descriptor.maxKernelBufferBindCount = 2;
    id<MTLIndirectCommandBuffer> icb = [c.device newIndirectCommandBufferWithDescriptor:descriptor maxCommandCount:3
                                                                                options:MTLResourceStorageModeShared];
    [c require:@"compute-icb-created" value:icb != nil];
    for (NSUInteger i = 0; i < 3; i++) {
        id<MTLIndirectComputeCommand> command = [icb indirectComputeCommandAtIndex:i];
        [command setComputePipelineState:pipeline];
        [command setKernelBuffer:out offset:i * 16 * sizeof(uint32_t) atIndex:0];
        [command setKernelBuffer:values offset:i * sizeof(uint32_t) atIndex:1];
        [command concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
    }
    return icb;
}

static void WMRunComputeICB(WMCase *c, id<MTLIndirectCommandBuffer> icb, NSRange range, id<MTLBuffer> out,
                            id<MTLBuffer> values) {
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e useResource:out usage:MTLResourceUsageWrite];
        [e useResource:values usage:MTLResourceUsageRead];
        [e executeCommandsInBuffer:icb withRange:range];
    }];
}

WMREF_CASE(icbCompute, "icb.compute", "icb.compute",
           "Compute indirect command buffer encoded on the CPU, executed whole and in part, then reset")
{
    id<MTLComputePipelineState> pipeline = WMICBComputePipeline(c);
    const uint32_t values[3] = {0x100, 0x200, 0x300};
    id<MTLBuffer> valueBuffer = [c bufferWithBytes:values length:sizeof values];
    id<MTLBuffer> out = [c buffer:48 * sizeof(uint32_t)];
    id<MTLIndirectCommandBuffer> icb = WMICBCompute(c, pipeline, out, valueBuffer);
    [c value:@"size" u64:icb.size rule:WMExact()];
    WMRunComputeICB(c, icb, NSMakeRange(0, 3), out, valueBuffer);
    [c data:@"all-commands" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];

    memset(out.contents, 0, out.length);
    WMRunComputeICB(c, icb, NSMakeRange(1, 1), out, valueBuffer);
    [c data:@"command-1" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];

    // A reset command does nothing when executed.
    memset(out.contents, 0, out.length);
    [icb resetWithRange:NSMakeRange(0, 2)];
    WMRunComputeICB(c, icb, NSMakeRange(0, 3), out, valueBuffer);
    [c data:@"after-reset-0-1" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

#pragma mark - Render

static id<MTLRenderPipelineState> WMICBRenderPipeline(WMCase *c) {
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [c function:@"icb_vertex"];
    descriptor.fragmentFunction = [c function:@"icb_fragment"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.supportIndirectCommandBuffers = YES;
    return [c renderPipeline:descriptor];
}

typedef struct {
    id<MTLBuffer> positions;
    id<MTLBuffer> colors;
} WMICBRenderData;

// Two triangles per quadrant command: command 0 covers the left half,
// command 1 the top-right quarter, command 2 the bottom-right quarter of an
// 8x8 target. Edges lie on pixel boundaries, away from pixel centers.
static WMICBRenderData WMMakeICBRenderData(WMCase *c) {
    const float quads[3][4] = {{-1, -1, 0, 1}, {0, 0, 1, 1}, {0, -1, 1, 0}};
    float positions[3][12];
    for (int q = 0; q < 3; q++) {
        float x0 = quads[q][0], y0 = quads[q][1], x1 = quads[q][2], y1 = quads[q][3];
        const float v[12] = {x0, y0, x1, y0, x0, y1, x1, y0, x1, y1, x0, y1};
        memcpy(positions[q], v, sizeof v);
    }
    // Colors with components 0 and 1, exactly representable in RGBA8Unorm.
    const float colors[3][4] = {{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}};
    WMICBRenderData data = {[c bufferWithBytes:positions length:sizeof positions],
                            [c bufferWithBytes:colors length:sizeof colors]};
    return data;
}

static id<MTLIndirectCommandBuffer> WMICBRender(WMCase *c, id<MTLRenderPipelineState> pipeline, WMICBRenderData d,
                                                NSUInteger count) {
    MTLIndirectCommandBufferDescriptor *descriptor = [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeDraw;
    descriptor.inheritBuffers = NO;
    descriptor.inheritPipelineState = NO;
    descriptor.maxVertexBufferBindCount = 1;
    descriptor.maxFragmentBufferBindCount = 1;
    id<MTLIndirectCommandBuffer> icb = [c.device newIndirectCommandBufferWithDescriptor:descriptor
                                                                       maxCommandCount:count
                                                                               options:MTLResourceStorageModeShared];
    [c require:[NSString stringWithFormat:@"render-icb-created.%lu", (unsigned long)count] value:icb != nil];
    return icb;
}

static void WMEncodeICBRender(id<MTLIndirectCommandBuffer> icb, id<MTLRenderPipelineState> pipeline,
                              WMICBRenderData d) {
    for (NSUInteger i = 0; i < 3; i++) {
        id<MTLIndirectRenderCommand> command = [icb indirectRenderCommandAtIndex:i];
        [command setRenderPipelineState:pipeline];
        [command setVertexBuffer:d.positions offset:i * 12 * sizeof(float) atIndex:0];
        [command setFragmentBuffer:d.colors offset:i * 4 * sizeof(float) atIndex:0];
        [command drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6 instanceCount:1 baseInstance:0];
    }
}

static NSData *WMDrawICB(WMCase *c, id<MTLIndirectCommandBuffer> icb, NSRange range, WMICBRenderData d) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                          width:8 height:8
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageRenderTarget;
    descriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> target = [c texture:descriptor];
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e useResource:d.positions usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
        [e useResource:d.colors usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
        [e executeCommandsInBuffer:icb withRange:range];
    }];
    return [c readTexture:target level:0 slice:0 bytesPerRow:32];
}

static NSDictionary *WMICBLayout(void) {
    return @{@"format": @"RGBA8Unorm", @"width": @8, @"height": @8, @"depth": @1, @"bytes_per_row": @32};
}

WMREF_CASE(icbRender, "icb.render", "icb.render",
           "Render indirect command buffer encoded on the CPU and executed in a render pass, then reset")
{
    id<MTLRenderPipelineState> pipeline = WMICBRenderPipeline(c);
    WMICBRenderData d = WMMakeICBRenderData(c);
    id<MTLIndirectCommandBuffer> icb = WMICBRender(c, pipeline, d, 3);
    WMEncodeICBRender(icb, pipeline, d);
    NSData *pixels = WMDrawICB(c, icb, NSMakeRange(0, 3), d);
    [c data:@"all-commands" elem:WMElemU8 bytes:pixels.bytes length:pixels.length layout:WMICBLayout()
       rule:WMExact()];
    pixels = WMDrawICB(c, icb, NSMakeRange(2, 1), d);
    [c data:@"command-2" elem:WMElemU8 bytes:pixels.bytes length:pixels.length layout:WMICBLayout()
       rule:WMExact()];
    [icb resetWithRange:NSMakeRange(1, 1)];
    pixels = WMDrawICB(c, icb, NSMakeRange(0, 3), d);
    [c data:@"after-reset-1" elem:WMElemU8 bytes:pixels.bytes length:pixels.length layout:WMICBLayout()
       rule:WMExact()];
}

WMREF_CASE(icbBlitOperations, "icb.blit-copy-optimize-reset", "icb.blit",
           "copyIndirectCommandBuffer, optimizeIndirectCommandBuffer and resetCommandsInBuffer in a blit encoder")
{
    id<MTLRenderPipelineState> pipeline = WMICBRenderPipeline(c);
    WMICBRenderData d = WMMakeICBRenderData(c);
    id<MTLIndirectCommandBuffer> source = WMICBRender(c, pipeline, d, 3);
    WMEncodeICBRender(source, pipeline, d);
    id<MTLIndirectCommandBuffer> destination = WMICBRender(c, pipeline, d, 4);

    // Commands 1 and 2 of the source become commands 2 and 3 of the
    // destination; commands 0 and 1 of the destination stay empty.
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyIndirectCommandBuffer:source sourceRange:NSMakeRange(1, 2) destination:destination
                    destinationIndex:2];
    }];
    NSData *pixels = WMDrawICB(c, destination, NSMakeRange(0, 4), d);
    [c data:@"copied" elem:WMElemU8 bytes:pixels.bytes length:pixels.length layout:WMICBLayout() rule:WMExact()];

    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e optimizeIndirectCommandBuffer:destination withRange:NSMakeRange(0, 4)];
    }];
    pixels = WMDrawICB(c, destination, NSMakeRange(0, 4), d);
    [c data:@"optimized" elem:WMElemU8 bytes:pixels.bytes length:pixels.length layout:WMICBLayout()
       rule:WMExact()];

    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e resetCommandsInBuffer:source withRange:NSMakeRange(0, 2)];
    }];
    pixels = WMDrawICB(c, source, NSMakeRange(0, 3), d);
    [c data:@"source-after-blit-reset-0-1" elem:WMElemU8 bytes:pixels.bytes length:pixels.length
     layout:WMICBLayout() rule:WMExact()];
}
