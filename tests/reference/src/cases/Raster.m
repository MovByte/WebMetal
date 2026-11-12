// Rasterization cases: triangle coverage, edge ties, points, lines, fill
// mode, culling and winding, viewport, scissor and depth clipping.
//
// Coverage at one sample per pixel is evaluated at the pixel centre: Apple's
// documentation of -[MTLDevice getDefaultSamplePositions:count:] states that
// "for GPUs capable of sampling once, the default position is at the center
// of the pixel". A pixel whose centre lies strictly inside or outside a
// primitive, by more than the subpixel precision, therefore has a defined
// coverage; ties on an edge are not defined by Metal's documentation.

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
void RRecordRegion(WMCase *c, NSString *name, NSData *image, NSUInteger imageWidth, NSUInteger bytesPerPixel,
                   NSString *format, NSUInteger x, NSUInteger y, NSUInteger width, NSUInteger height,
                   WMElem elem, WMRule *rule);
void RDraw(WMCase *c, MTLRenderPassDescriptor *pass, id<MTLRenderPipelineState> pipeline,
           MTLPrimitiveType type, const RVertex *vertices, NSUInteger count);
BOOL RPaintTriangles(const double (*triangles)[6], const uint32_t *ids, NSUInteger count, NSUInteger width,
                     NSUInteger height, double margin, uint32_t *map);

// Pixel centres closer than this to an edge are treated as ambiguous by
// the cases that claim exact coverage (1/16 pixel, well above the subpixel
// grid of any rasterizer that snaps to 1/256 pixel or finer).
static const double kMargin = 1.0 / 16.0;

static void RTrianglesToVertices(const double (*triangles)[6], const uint32_t *ids, NSUInteger count,
                                 NSUInteger width, NSUInteger height, RVertex *out) {
    for (NSUInteger t = 0; t < count; t++)
        for (int k = 0; k < 3; k++)
            out[t * 3 + k] = RVertexAt(triangles[t][2 * k], triangles[t][2 * k + 1], 0, width, height, ids[t]);
}

static id<MTLRenderPipelineState> RIdPipeline(WMCase *c) {
    return [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id",
                                                 MTLPixelFormatR32Uint)];
}

WMREF_CASE(rasterTriangleCoverage, "raster.triangle-coverage", "raster.coverage",
           "Triangle coverage by pixel-centre containment, including clipped, thin and degenerate triangles")
{
    enum { W = 16, H = 16, T = 6 };
    // Vertices in pixel coordinates on a 1/16 pixel grid (exact in binary32
    // and on any subpixel grid of 1/16 pixel or finer); every pixel centre
    // is at least 1/16 pixel from every edge (checked below).
    const double triangles[T][6] = {
        {1.125, 1.5625, 9.375, 2.0625, 3.4375, 10.9375},     // clockwise on screen
        {14.875, 1.125, 8.75, 13.3125, 15.375, 13.0},        // counter-clockwise on screen
        {1.25, 13.125, 32.0, 13.4375, 1.25, 13.9375},        // thin, one pixel row, tip outside
        {10.0625, 5.4375, 12.25, 7.5625, 11.15625, 6.5},     // zero area (third vertex is the midpoint)
        {6.25, 5.9375, 6.875, 6.625, 6.25, 6.8125},          // smaller than a pixel, one centre
        {12.5625, -2.8125, 20.125, 4.3125, 11.5, 6.375},     // partly outside the target
    };
    const uint32_t ids[T] = {1, 2, 3, 4, 5, 6};
    uint32_t expected[W * H] = {0};
    if (!RPaintTriangles(triangles, ids, T, W, H, kMargin, expected))
        [c fail:@"a pixel centre lies within the margin of an edge"];
    RVertex vertices[T * 3];
    RTrianglesToVertices(triangles, ids, T, W, H, vertices);
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, W, H);
    RDraw(c, RPassColor(target, MTLClearColorMake(0, 0, 0, 0)), RIdPipeline(c), MTLPrimitiveTypeTriangle,
          vertices, T * 3);
    NSData *ids2 = RRecordTexture(c, @"coverage", target, WMElemU32, 4, WMExact());
    [c value:@"coverage-equals-centre-containment" boolean:memcmp(ids2.bytes, expected, sizeof expected) == 0
        rule:WMExact()];
}

WMREF_CASE(rasterEdgeTies, "raster.edge-ties", "raster.coverage",
           "Coverage of pixel centres that lie exactly on triangle edges (shared diagonal, left/top/right/bottom)")
{
    enum { W = 16, H = 16 };
    // A square from 4.5 to 11.5 split along its diagonal: every edge passes
    // through pixel centres.
    const double triangles[2][6] = {
        {4.5, 4.5, 11.5, 4.5, 11.5, 11.5},
        {4.5, 4.5, 11.5, 11.5, 4.5, 11.5},
    };
    id<MTLRenderPipelineState> pipeline = RIdPipeline(c);
    NSString *reason = @"Metal's documentation does not state how a sample point on a primitive edge is assigned";
    NSData *maps[2];
    for (int t = 0; t < 2; t++) {
        RVertex vertices[3];
        const uint32_t triangleId = (uint32_t)t + 1;
        RTrianglesToVertices(&triangles[t], &triangleId, 1, W, H, vertices);
        id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, W, H);
        RDraw(c, RPassColor(target, MTLClearColorMake(0, 0, 0, 0)), pipeline, MTLPrimitiveTypeTriangle, vertices, 3);
        maps[t] = RRecordTexture(c, t ? @"lower-left" : @"upper-right", target, WMElemU32, 4, WMNone(reason));
    }
    // Pixel centres of the square covered by both triangles and by neither.
    uint64_t both = 0, neither = 0;
    const uint32_t *a = maps[0].bytes, *b = maps[1].bytes;
    for (int y = 4; y < 12; y++)
        for (int x = 4; x < 12; x++) {
            BOOL inSquare = x + 0.5 >= 4.5 && x + 0.5 <= 11.5 && y + 0.5 >= 4.5 && y + 0.5 <= 11.5;
            if (!inSquare) continue;
            both += a[y * W + x] && b[y * W + x];
            neither += !a[y * W + x] && !b[y * W + x];
        }
    [c value:@"square-centres-covered-twice" u64:both rule:WMNone(reason)];
    [c value:@"square-centres-covered-by-neither" u64:neither rule:WMNone(reason)];
}

WMREF_CASE(rasterPoints, "raster.points", "raster.points",
           "Point primitives with [[point_size]] at pixel centres and corners")
{
    enum { W = 32, H = 16 };
    // Each point sits in its own 8x8 box. Sizes and positions whose pixel
    // coverage is the same whether the point is a square or a disc of
    // diameter point_size, with no pixel centre on the boundary, are exact;
    // the others depend on the point shape, which the MSL specification
    // ("Size of a point primitive", section 5.2.3.3) does not define.
    const struct { double x, y, size; BOOL exact; } points[] = {
        {4.5, 4.5, 1.0, YES},   // one centre
        {12.5, 4.5, 3.0, YES},  // 3x3 centres for both shapes
        {20.0, 4.0, 2.0, YES},  // 2x2 centres around a corner for both shapes
        {28.0, 4.0, 4.0, NO},   // square 4x4, disc excludes the corners
        {4.5, 12.5, 2.0, NO},   // neighbour centres on the boundary
        {12.5, 12.5, 5.0, NO},  // square 5x5, disc excludes the corners
        {20.5, 12.5, 1.5, YES}, // one centre
    };
    enum { N = sizeof points / sizeof points[0] };
    RVertex vertices[N];
    for (NSUInteger i = 0; i < N; i++) {
        vertices[i] = RVertexAt(points[i].x, points[i].y, 0, W, H, (uint32_t)i + 1);
        vertices[i].extra[0] = (float)points[i].size;
    }
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, W, H);
    id<MTLRenderPipelineState> pipeline =
        [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex", @"rnd_fragment_point_id", MTLPixelFormatR32Uint)];
    RDraw(c, RPassColor(target, MTLClearColorMake(0, 0, 0, 0)), pipeline, MTLPrimitiveTypePoint, vertices, N);
    NSData *image = RRead(c, target, 4);
    for (NSUInteger i = 0; i < N; i++) {
        NSUInteger bx = (NSUInteger)(points[i].x / 8) * 8, by = (NSUInteger)(points[i].y / 8) * 8;
        NSString *name = [NSString stringWithFormat:@"point-%lu-size-%g", (unsigned long)i + 1, points[i].size];
        RRecordRegion(c, name, image, W, 4, @"R32Uint", bx, by, 8, 8, WMElemU32,
                      points[i].exact ? WMExact()
                                      : WMNone(@"coverage depends on the point shape, which Metal does not define"));
    }
}

WMREF_CASE(rasterLines, "raster.lines", "raster.lines",
           "Line and line-strip coverage")
{
    enum { W = 16, H = 16 };
    NSString *reason = @"Metal's documentation does not define line rasterization (pixel selection, width, end points)";
    id<MTLRenderPipelineState> pipeline = RIdPipeline(c);
    RVertex lines[] = {
        RVertexAt(1.2, 1.7, 0, W, H, 1), RVertexAt(13.8, 4.3, 0, W, H, 1),
        RVertexAt(2.3, 14.1, 0, W, H, 2), RVertexAt(5.1, 6.2, 0, W, H, 2),
        RVertexAt(8.5, 8.5, 0, W, H, 3), RVertexAt(14.5, 8.5, 0, W, H, 3),
    };
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, W, H);
    RDraw(c, RPassColor(target, MTLClearColorMake(0, 0, 0, 0)), pipeline, MTLPrimitiveTypeLine, lines, 6);
    RRecordTexture(c, @"line-list", target, WMElemU32, 4, WMNone(reason));
    RVertex strip[] = {
        RVertexAt(1.5, 1.5, 0, W, H, 4), RVertexAt(14.2, 2.6, 0, W, H, 4),
        RVertexAt(3.7, 12.9, 0, W, H, 4), RVertexAt(13.1, 14.4, 0, W, H, 4),
    };
    id<MTLTexture> target2 = RTarget(c, MTLPixelFormatR32Uint, W, H);
    RDraw(c, RPassColor(target2, MTLClearColorMake(0, 0, 0, 0)), pipeline, MTLPrimitiveTypeLineStrip, strip, 4);
    RRecordTexture(c, @"line-strip", target2, WMElemU32, 4, WMNone(reason));
}

WMREF_CASE(rasterFillModeLines, "raster.fill-mode-lines", "raster.fill-mode",
           "setTriangleFillMode:MTLTriangleFillModeLines")
{
    enum { W = 16, H = 16 };
    RVertex vertices[] = {
        RVertexAt(1.3, 1.2, 0, W, H, 7), RVertexAt(14.2, 3.1, 0, W, H, 7), RVertexAt(4.4, 13.6, 0, W, H, 7),
    };
    id<MTLBuffer> buffer = RVertexBuffer(c, vertices, 3);
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, W, H);
    id<MTLRenderPipelineState> pipeline = RIdPipeline(c);
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setTriangleFillMode:MTLTriangleFillModeLines];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    RRecordTexture(c, @"outline", target, WMElemU32, 4,
                   WMNone(@"line rasterization of triangle edges is not defined by Metal's documentation"));
}

WMREF_CASE(rasterCullWinding, "raster.cull-winding", "raster.cull",
           "Cull mode and front-facing winding with [[front_facing]] for both triangle orientations")
{
    enum { W = 16, H = 16, T = 4 };
    // Two clockwise and two counter-clockwise triangles (screen space, y
    // down), one per quadrant.
    const double triangles[T][6] = {
        {1.1875, 1.25, 6.4375, 1.375, 1.0, 6.5},
        {9.0625, 1.375, 9.125, 6.75, 14.875, 0.875},
        {0.8125, 9.3125, 6.6875, 8.75, 1.3125, 14.375},
        {9.3125, 9.375, 9.25, 14.375, 14.5, 8.8125},
    };
    const uint32_t ids[T] = {1, 2, 3, 4};
    uint32_t check[W * H] = {0};
    if (!RPaintTriangles(triangles, ids, T, W, H, kMargin, check))
        [c fail:@"a pixel centre lies within the margin of an edge"];
    RVertex vertices[T * 3];
    RTrianglesToVertices(triangles, ids, T, W, H, vertices);
    id<MTLBuffer> buffer = RVertexBuffer(c, vertices, T * 3);
    id<MTLRenderPipelineState> pipeline =
        [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_facing", MTLPixelFormatR32Uint)];
    const MTLCullMode culls[] = {MTLCullModeNone, MTLCullModeFront, MTLCullModeBack};
    const MTLWinding windings[] = {MTLWindingClockwise, MTLWindingCounterClockwise};
    NSString *const cullNames[] = {@"none", @"front", @"back"};
    NSString *const windingNames[] = {@"clockwise", @"counter-clockwise"};
    for (int w = 0; w < 2; w++)
        for (int k = 0; k < 3; k++) {
            id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, W, H);
            MTLWinding winding = windings[w];
            MTLCullMode cull = culls[k];
            [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
                [e setRenderPipelineState:pipeline];
                [e setFrontFacingWinding:winding];
                [e setCullMode:cull];
                [e setVertexBuffer:buffer offset:0 atIndex:0];
                [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:T * 3];
            }];
            // Value per pixel: 0x100 if front facing, plus the triangle id.
            RRecordTexture(c, [NSString stringWithFormat:@"front-%@.cull-%@", windingNames[w], cullNames[k]], target,
                           WMElemU32, 4, WMExact());
        }
}

WMREF_CASE(rasterViewport, "raster.viewport", "raster.viewport",
           "Viewport with a negative origin, larger than the attachment, with a depth range; [[position]] values")
{
    enum { W = 16, H = 16 };
    // One triangle whose clipped part covers clip space at z = 0.5 (Metal
    // clips primitives to the normalized device coordinate box, MSL 4.1
    // section 1.7). With viewport origin (-8, -8), size 32x32 and depth
    // range [0.25, 0.75], window x = (x_ndc + 1) * 16 - 8 and
    // z = 0.25 + 0.5 * 0.5 = 0.5; all values are exact in binary32.
    RVertex quad[3];
    const double corners[3][2] = {{-1, -1}, {3, -1}, {-1, 3}};
    for (int i = 0; i < 3; i++) {
        RVertex v = {{(float)corners[i][0], (float)corners[i][1], 0.5f, 1.0f}, {9, 0, 0, 0}, {1, 0, 0, 0}};
        quad[i] = v;
    }
    id<MTLBuffer> buffer = RVertexBuffer(c, quad, 3);
    id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA32Uint, W, H);
    id<MTLRenderPipelineState> pipeline =
        [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_position_w",
                                              MTLPixelFormatRGBA32Uint)];
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setViewport:(MTLViewport){-8, -8, 32, 32, 0.25, 0.75}];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    // [[position]] in the fragment stage: window-relative (x, y, z, 1/w)
    // (MSL 4.1 section 5.2.3.4) sampled at the pixel centre.
    NSData *image = RRecordTexture(c, @"position-xyzw-bits", target, WMElemU32, 16, WMExact());
    BOOL centres = YES;
    const uint32_t *p = image.bytes;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float fx, fy, fz, fw;
            memcpy(&fx, &p[(y * W + x) * 4 + 0], 4);
            memcpy(&fy, &p[(y * W + x) * 4 + 1], 4);
            memcpy(&fz, &p[(y * W + x) * 4 + 2], 4);
            memcpy(&fw, &p[(y * W + x) * 4 + 3], 4);
            centres &= fx == x + 0.5f && fy == y + 0.5f && fz == 0.5f && fw == 1.0f;
        }
    [c value:@"positions-are-centres-z-half-w-one" boolean:centres rule:WMExact()];

    // A smaller viewport inside the target: the clipped triangle covers
    // exactly the viewport rectangle (3, 5)-(11, 9), whose edges lie
    // between pixel centres.
    id<MTLTexture> target2 = RTarget(c, MTLPixelFormatRGBA32Uint, W, H);
    [c render:RPassColor(target2, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setViewport:(MTLViewport){3, 5, 8, 4, 0, 1}];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    RRecordTexture(c, @"inner-viewport-position-xyzw-bits", target2, WMElemU32, 16, WMExact());
}

WMREF_CASE(rasterScissor, "raster.scissor", "raster.scissor",
           "Scissor rectangle inside the attachment")
{
    enum { W = 16, H = 16 };
    // One triangle covering the whole target; its long edge is outside.
    const double triangles[1][6] = {{-1, -1, 40, -1, -1, 40}};
    const uint32_t ids[1] = {5};
    RVertex vertices[3];
    RTrianglesToVertices(triangles, ids, 1, W, H, vertices);
    id<MTLBuffer> buffer = RVertexBuffer(c, vertices, 3);
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, W, H);
    id<MTLRenderPipelineState> pipeline = RIdPipeline(c);
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setScissorRect:(MTLScissorRect){3, 5, 7, 4}];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    RRecordTexture(c, @"coverage", target, WMElemU32, 4, WMExact());
}

WMREF_CASE(rasterDepthClipClamp, "raster.depth-clip-clamp", "raster.depth-clip",
           "Depth clip mode clip versus clamp for primitives beyond the near and far planes")
{
    enum { W = 8, H = 8 };
    // Two quads, left half at z_ndc = 1.5 (beyond far), right half at
    // z_ndc = -0.5 (before near). Clipped: nothing drawn. Clamped: drawn
    // with depth clamped to the viewport depth range [0, 1] (MTLDepthClipModeClamp).
    RVertex quads[12];
    const double xs[2][2] = {{-1, 0}, {0, 1}};
    const float zs[2] = {1.5f, -0.5f};
    for (int q = 0; q < 2; q++) {
        const double corners[6][2] = {{xs[q][0], -1}, {xs[q][1], -1}, {xs[q][1], 1},
                                      {xs[q][0], -1}, {xs[q][1], 1}, {xs[q][0], 1}};
        for (int i = 0; i < 6; i++) {
            RVertex v = {{(float)corners[i][0], (float)corners[i][1], zs[q], 1.0f}, {(uint32_t)q + 1, 0, 0, 0},
                         {1, 0, 0, 0}};
            quads[q * 6 + i] = v;
        }
    }
    id<MTLBuffer> buffer = RVertexBuffer(c, quads, 12);
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id",
                                                         MTLPixelFormatR32Uint);
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
    ds.depthCompareFunction = MTLCompareFunctionAlways;
    ds.depthWriteEnabled = YES;
    id<MTLDepthStencilState> depthState = [c depthStencilState:ds];
    NSString *const names[2] = {@"clip", @"clamp"};
    const MTLDepthClipMode modes[2] = {MTLDepthClipModeClip, MTLDepthClipModeClamp};
    for (int m = 0; m < 2; m++) {
        id<MTLTexture> color = RTarget(c, MTLPixelFormatR32Uint, W, H);
        id<MTLTexture> depth = RTarget(c, MTLPixelFormatDepth32Float, W, H);
        MTLRenderPassDescriptor *pass = RPassColor(color, MTLClearColorMake(0, 0, 0, 0));
        pass.depthAttachment.texture = depth;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        pass.depthAttachment.clearDepth = 0.5;
        MTLDepthClipMode mode = modes[m];
        [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:pipeline];
            [e setDepthStencilState:depthState];
            [e setDepthClipMode:mode];
            [e setVertexBuffer:buffer offset:0 atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:12];
        }];
        RRecordTexture(c, [names[m] stringByAppendingString:@".color"], color, WMElemU32, 4, WMExact());
        RRecordTexture(c, [names[m] stringByAppendingString:@".depth"], depth, WMElemF32, 4, WMExact());
    }
}

WMREF_CASE(rasterScissorOutside, "raster.error.scissor-outside", "raster.scissor",
           "Scissor rectangle extending past the attachment (invalid use)")
{
    enum { W = 16, H = 16 };
    const double triangles[1][6] = {{-1, -1, 40, -1, -1, 40}};
    const uint32_t ids[1] = {3};
    RVertex vertices[3];
    RTrianglesToVertices(triangles, ids, 1, W, H, vertices);
    id<MTLBuffer> buffer = RVertexBuffer(c, vertices, 3);
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, W, H);
    id<MTLRenderPipelineState> pipeline = RIdPipeline(c);
    [c step:@"encode"];
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [c step:@"set-scissor"];
        [e setScissorRect:(MTLScissorRect){10, 10, 10, 10}];
        [c step:@"draw"];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    [c step:@"completed"];
    RRecordTexture(c, @"coverage", target, WMElemU32, 4,
                   WMNone(@"the result of a scissor rectangle outside the attachment is not defined"));
}
