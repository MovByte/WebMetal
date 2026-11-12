// Fragment-stage cases: discard, interpolation qualifiers and flat
// varyings, framebuffer fetch, raster order groups and memory barriers
// between draws.

#import "WMRefCase.h"

// Helpers in RenderSupport.m.
typedef struct { float position[4]; uint32_t data[4]; float extra[4]; } RVertex;
RVertex RVertexAt(double px, double py, double z, NSUInteger width, NSUInteger height, uint32_t id);
id<MTLBuffer> RVertexBuffer(WMCase *c, const RVertex *vertices, NSUInteger count);
id<MTLTexture> RTarget(WMCase *c, MTLPixelFormat format, NSUInteger width, NSUInteger height);
MTLRenderPassDescriptor *RPassColor(id<MTLTexture> color, MTLClearColor clear);
MTLRenderPipelineDescriptor *RPipelineDescriptor(WMCase *c, NSString *vertex, NSString *fragment,
                                                 MTLPixelFormat color);
NSData *RRead(WMCase *c, id<MTLTexture> texture, NSUInteger bytesPerPixel);
NSData *RRecordTexture(WMCase *c, NSString *name, id<MTLTexture> texture, WMElem elem,
                       NSUInteger bytesPerPixel, WMRule *rule);
void RDraw(WMCase *c, MTLRenderPassDescriptor *pass, id<MTLRenderPipelineState> pipeline,
           MTLPrimitiveType type, const RVertex *vertices, NSUInteger count);

// A triangle covering a whole width x height target; its edges are outside.
static id<MTLBuffer> FCover(WMCase *c, NSUInteger width, NSUInteger height, uint32_t id) {
    RVertex v[3] = {RVertexAt(-1, -1, 0, width, height, id), RVertexAt(3.0 * width, -1, 0, width, height, id),
                    RVertexAt(-1, 3.0 * height, 0, width, height, id)};
    return RVertexBuffer(c, v, 3);
}

WMREF_CASE(fragmentDiscard, "render.discard", "render.fragment",
           "discard_fragment for pixels with (x + y) % 3 == 0")
{
    id<MTLRenderPipelineState> pipeline =
        [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_discard", MTLPixelFormatR32Uint)];
    id<MTLBuffer> cover = FCover(c, 8, 8, 4);
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, 8, 8);
    [c render:RPassColor(target, MTLClearColorMake(9, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:cover offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    RRecordTexture(c, @"result", target, WMElemU32, 4, WMExact());
}

WMREF_CASE(fragmentInterpolation, "render.interpolation", "render.interpolation",
           "center_perspective, center_no_perspective, centroid_perspective and flat varyings")
{
    enum { W = 8, H = 8 };
    // A triangle whose vertices have different w, so perspective-correct
    // and linear interpolation differ.
    RVertex v[3] = {RVertexAt(-1, -1, 0, W, H, 1), RVertexAt(24, -1, 0, W, H, 2), RVertexAt(-1, 24, 0, W, H, 3)};
    const float values[3] = {0.0f, 1.0f, 0.5f};
    const float ws[3] = {1.0f, 2.0f, 4.0f};
    for (int k = 0; k < 3; k++) {
        v[k].extra[1] = values[k];
        for (int i = 0; i < 4; i++) v[k].position[i] *= ws[k];
    }
    id<MTLRenderPipelineState> pipeline =
        [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_interpolated", @"rnd_fragment_interpolated",
                                              MTLPixelFormatRGBA32Uint)];
    id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA32Uint, W, H);
    RDraw(c, RPassColor(target, MTLClearColorMake(0, 0, 0, 0)), pipeline, MTLPrimitiveTypeTriangle, v, 3);
    // Words per pixel: perspective, no_perspective, centroid (float bits),
    // flat id. The interpolation precision is not stated by Metal; the flat
    // value comes from the provoking vertex, which Metal's documentation does
    // not name.
    RRecordTexture(c, @"distinct-vertex-values", target, WMElemU32, 16,
                   WMNone(@"interpolation precision and the provoking vertex are not stated by Metal's documentation"));
    // With the same flat value and the same smooth value at every vertex,
    // every qualifier yields that value exactly.
    for (int k = 0; k < 3; k++) {
        v[k].extra[1] = 0.75f;
        v[k].data[0] = 6;
    }
    id<MTLTexture> target2 = RTarget(c, MTLPixelFormatRGBA32Uint, W, H);
    RDraw(c, RPassColor(target2, MTLClearColorMake(0, 0, 0, 0)), pipeline, MTLPrimitiveTypeTriangle, v, 3);
    NSData *uniform = RRead(c, target2, 16);
    const uint32_t *p = uniform.bytes;
    uint32_t threeQuarters;
    float tq = 0.75f;
    memcpy(&threeQuarters, &tq, 4);
    uint32_t flatWords[W * H];
    BOOL constant = YES;
    for (int i = 0; i < W * H; i++) {
        flatWords[i] = p[i * 4 + 3];
        constant &= p[i * 4] == threeQuarters && p[i * 4 + 1] == threeQuarters && p[i * 4 + 2] == threeQuarters;
    }
    [c data:@"equal-vertex-values.flat" elem:WMElemU32 bytes:flatWords length:sizeof flatWords rule:WMExact()];
    [c data:@"equal-vertex-values.smooth" elem:WMElemU32 bytes:uniform.bytes length:uniform.length
       rule:WMNone(@"interpolation of a constant is not stated to be exact by Metal")];
    [c value:@"equal-vertex-values.smooth-all-0.75" boolean:constant
        rule:WMNone(@"interpolation of a constant is not stated to be exact by Metal")];
}

// Programmable blending reads colour attachments in Apple-family GPUs
// (MSL 4.1 section 5.2.3.4, [[color(m)]] fragment input).
WMREF_CASE(fragmentFramebufferFetch, "render.framebuffer-fetch", "render.framebuffer-fetch",
           "[[color(0)]] input with three overlapping primitives: order-dependent result on R32Uint")
{
    // "Programmable blending": Apple2 and later (Metal Feature Set Tables, 2026-10-23).
    [c require:@"family.apple2" value:[c.device supportsFamily:MTLGPUFamilyApple2]];
    id<MTLRenderPipelineState> pipeline =
        [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_fetch", MTLPixelFormatR32Uint)];
    // Three triangles covering the whole target, drawn with ids 1, 2, 3 in
    // one draw and then 4 in a second draw: each fragment computes
    // previous * 3 + id, so the result depends on primitive order:
    // ((((0 * 3 + 1) * 3 + 2) * 3 + 3) * 3 + 4) = 58.
    RVertex v[12];
    for (int t = 0; t < 4; t++) {
        v[t * 3 + 0] = RVertexAt(-1, -1, 0, 8, 8, (uint32_t)t + 1);
        v[t * 3 + 1] = RVertexAt(24, -1, 0, 8, 8, (uint32_t)t + 1);
        v[t * 3 + 2] = RVertexAt(-1, 24, 0, 8, 8, (uint32_t)t + 1);
    }
    id<MTLBuffer> buffer = RVertexBuffer(c, v, 12);
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, 8, 8);
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:9];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:9 vertexCount:3];
    }];
    RRecordTexture(c, @"result", target, WMElemU32, 4, WMExact());
}

WMREF_CASE(fragmentRasterOrderGroups, "render.raster-order-groups", "render.raster-order-groups",
           "Read-modify-write of a read_write texture in raster order group 0 by overlapping primitives")
{
    [c require:@"device.raster-order-groups" value:c.device.rasterOrderGroupsSupported];
    [c require:@"device.read-write-texture-tier-1" value:c.device.readWriteTextureSupport >= MTLReadWriteTextureTier1];
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_raster_order",
                                                         MTLPixelFormatR32Uint);
    d.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                                  width:8 height:8 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [c texture:td];
    uint32_t zeros[64] = {0};
    [texture replaceRegion:MTLRegionMake2D(0, 0, 8, 8) mipmapLevel:0 withBytes:zeros bytesPerRow:32];
    RVertex v[9];
    for (int t = 0; t < 3; t++) {
        v[t * 3 + 0] = RVertexAt(-1, -1, 0, 8, 8, (uint32_t)t + 1);
        v[t * 3 + 1] = RVertexAt(24, -1, 0, 8, 8, (uint32_t)t + 1);
        v[t * 3 + 2] = RVertexAt(-1, 24, 0, 8, 8, (uint32_t)t + 1);
    }
    id<MTLBuffer> buffer = RVertexBuffer(c, v, 9);
    id<MTLTexture> dummy = RTarget(c, MTLPixelFormatR32Uint, 8, 8);
    [c render:RPassColor(dummy, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        [e setFragmentTexture:texture atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:9];
    }];
    // ((0 * 3 + 1) * 3 + 2) * 3 + 3 = 18 at every pixel.
    RRecordTexture(c, @"result", texture, WMElemU32, 4, WMExact());
}

WMREF_CASE(fragmentMemoryBarrier, "render.memory-barrier", "render.barriers",
           "memoryBarrierWithScope:afterStages:vertex beforeStages:fragment between a vertex stage that stores to a "
           "buffer and a fragment stage that reads it")
{
    if (@available(macOS 10.14, iOS 16.0, *)) {
        MTLRenderPipelineDescriptor *storeDescriptor = [MTLRenderPipelineDescriptor new];
        storeDescriptor.vertexFunction = [c function:@"rnd_vertex_store"];
        storeDescriptor.rasterizationEnabled = NO;
        storeDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
        id<MTLRenderPipelineState> store = [c renderPipeline:storeDescriptor];
        id<MTLRenderPipelineState> load =
            [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_load", MTLPixelFormatR32Uint)];
        id<MTLBuffer> cover = FCover(c, 8, 8, 5);
        id<MTLBuffer> shared = [c buffer:64 * sizeof(uint32_t)];
        id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, 8, 8);
        [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
            // rnd_vertex_store writes (instance << 16) | (vertex * 3 + 1) at
            // instance * 8 + vertex.
            [e setRenderPipelineState:store];
            [e setVertexBuffer:shared offset:0 atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:8 instanceCount:8];
            [e memoryBarrierWithScope:MTLBarrierScopeBuffers afterStages:MTLRenderStageVertex
                         beforeStages:MTLRenderStageFragment];
            [e setRenderPipelineState:load];
            [e setVertexBuffer:cover offset:0 atIndex:0];
            [e setFragmentBuffer:shared offset:0 atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }];
        [c data:@"buffer" elem:WMElemU32 bytes:shared.contents length:shared.length rule:WMExact()];
        RRecordTexture(c, @"result", target, WMElemU32, 4, WMExact());
    } else {
        [c require:@"os.render-memory-barrier" value:NO];
    }
}

WMREF_CASE(fragmentBarrierAfterFragment, "render.error.memory-barrier-after-fragment", "render.barriers",
           "memoryBarrierWithScope: with afterStages containing the fragment stage (rejected on Apple-family GPUs)")
{
    if (@available(macOS 10.14, iOS 16.0, *)) {
        id<MTLRenderPipelineState> store =
            [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_store", MTLPixelFormatR32Uint)];
        id<MTLRenderPipelineState> load =
            [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_load", MTLPixelFormatR32Uint)];
        id<MTLBuffer> cover = FCover(c, 8, 8, 5);
        id<MTLBuffer> shared = [c buffer:64 * sizeof(uint32_t)];
        id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, 8, 8);
        [c step:@"encode"];
        [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
            [e setVertexBuffer:cover offset:0 atIndex:0];
            [e setFragmentBuffer:shared offset:0 atIndex:0];
            [e setRenderPipelineState:store];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [c step:@"barrier"];
            [e memoryBarrierWithScope:MTLBarrierScopeBuffers afterStages:MTLRenderStageFragment
                         beforeStages:MTLRenderStageFragment];
            [e setRenderPipelineState:load];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }];
        [c step:@"completed"];
        RRecordTexture(c, @"result", target, WMElemU32, 4,
                       WMNone(@"the barrier is invalid on this device family; ordering of the two draws' buffer "
                              @"accesses is then not defined"));
    } else {
        [c require:@"os.render-memory-barrier" value:NO];
    }
}
