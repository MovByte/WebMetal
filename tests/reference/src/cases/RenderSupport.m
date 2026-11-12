// Helpers shared by the render cases. The prototypes are repeated at the
// top of each render case file.

#import "WMRefCase.h"

// Mirrors struct RVertex in shaders/render.metal.
typedef struct {
    float position[4];
    uint32_t data[4];
    float extra[4];
} RVertex;

// A vertex at pixel coordinate (px, py) of a width x height target (x to the
// right, y down, as in Metal viewport coordinates), depth z, carrying id.
// For power-of-two target sizes the mapping to clip space is exact in
// binary32.
RVertex RVertexAt(double px, double py, double z, NSUInteger width, NSUInteger height, uint32_t id) {
    RVertex v = {{0}, {0}, {0}};
    v.position[0] = (float)(px / (double)width * 2.0 - 1.0);
    v.position[1] = (float)(1.0 - py / (double)height * 2.0);
    v.position[2] = (float)z;
    v.position[3] = 1.0f;
    v.data[0] = id;
    v.extra[0] = 1.0f;
    return v;
}

id<MTLBuffer> RVertexBuffer(WMCase *c, const RVertex *vertices, NSUInteger count) {
    return [c bufferWithBytes:vertices length:count * sizeof(RVertex)];
}

id<MTLTexture> RTargetSampled(WMCase *c, MTLPixelFormat format, NSUInteger width, NSUInteger height,
                              NSUInteger samples) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width
                                                                                height:height mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    if (samples > 1) {
        d.textureType = MTLTextureType2DMultisample;
        d.sampleCount = samples;
    }
    return [c texture:d];
}

id<MTLTexture> RTarget(WMCase *c, MTLPixelFormat format, NSUInteger width, NSUInteger height) {
    return RTargetSampled(c, format, width, height, 1);
}

MTLRenderPassDescriptor *RPassColor(id<MTLTexture> color, MTLClearColor clear) {
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = color;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = clear;
    return pass;
}

MTLRenderPipelineDescriptor *RPipelineDescriptor(WMCase *c, NSString *vertex, NSString *fragment,
                                                 MTLPixelFormat color) {
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [c function:vertex];
    if (fragment) d.fragmentFunction = [c function:fragment];
    d.colorAttachments[0].pixelFormat = color;
    return d;
}

NSDictionary *RLayout(id<MTLTexture> texture, NSUInteger bytesPerRow) {
    return @{@"format": WMPixelFormatName(texture.pixelFormat) ?: @"unknown", @"width": @(texture.width),
             @"height": @(texture.height), @"bytes_per_row": @(bytesPerRow)};
}

NSData *RRead(WMCase *c, id<MTLTexture> texture, NSUInteger bytesPerPixel) {
    return [c readTexture:texture level:0 slice:0 bytesPerRow:texture.width * bytesPerPixel];
}

// Records level 0, slice 0 of a texture.
NSData *RRecordTexture(WMCase *c, NSString *name, id<MTLTexture> texture, WMElem elem,
                       NSUInteger bytesPerPixel, WMRule *rule) {
    NSData *data = RRead(c, texture, bytesPerPixel);
    [c data:name elem:elem bytes:data.bytes length:data.length
     layout:RLayout(texture, texture.width * bytesPerPixel) rule:rule];
    return data;
}

// Records a rectangle of tightly packed image data.
void RRecordRegion(WMCase *c, NSString *name, NSData *image, NSUInteger imageWidth, NSUInteger bytesPerPixel,
                   NSString *format, NSUInteger x, NSUInteger y, NSUInteger width, NSUInteger height,
                   WMElem elem, WMRule *rule) {
    NSMutableData *region = [NSMutableData dataWithLength:width * height * bytesPerPixel];
    for (NSUInteger row = 0; row < height; row++)
        memcpy((uint8_t *)region.mutableBytes + row * width * bytesPerPixel,
               (const uint8_t *)image.bytes + ((y + row) * imageWidth + x) * bytesPerPixel, width * bytesPerPixel);
    NSDictionary *layout = @{@"format": format, @"x": @(x), @"y": @(y), @"width": @(width), @"height": @(height),
                             @"bytes_per_row": @(width * bytesPerPixel)};
    [c data:name elem:elem bytes:region.bytes length:region.length layout:layout rule:rule];
}

// One render pass with one pipeline that draws `count` vertices of `type`
// from `vertices` (bound at vertex buffer 0).
void RDraw(WMCase *c, MTLRenderPassDescriptor *pass, id<MTLRenderPipelineState> pipeline,
           MTLPrimitiveType type, const RVertex *vertices, NSUInteger count) {
    id<MTLBuffer> buffer = RVertexBuffer(c, vertices, count);
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:buffer offset:0 atIndex:0];
        [e drawPrimitives:type vertexStart:0 vertexCount:count];
    }];
}

// Signed distance (in pixels) from point (x, y) to the line through a and b,
// positive on the left of a->b in a y-down frame.
static double REdge(const double *a, const double *b, double x, double y) {
    double dx = b[0] - a[0], dy = b[1] - a[1];
    return ((x - a[0]) * dy - (y - a[1]) * dx) / sqrt(dx * dx + dy * dy);
}

// Paints triangles (pixel coordinates, three vertices of two doubles each)
// in order into a width x height map of ids, by containment of pixel
// centres. Returns NO if a pixel centre lies within `margin` pixels of an
// edge of a triangle whose bounding box contains it.
BOOL RPaintTriangles(const double (*triangles)[6], const uint32_t *ids, NSUInteger count, NSUInteger width,
                     NSUInteger height, double margin, uint32_t *map) {
    for (NSUInteger t = 0; t < count; t++) {
        const double *v = triangles[t];
        double a[2] = {v[0], v[1]}, b[2] = {v[2], v[3]}, cc[2] = {v[4], v[5]};
        for (NSUInteger y = 0; y < height; y++)
            for (NSUInteger x = 0; x < width; x++) {
                double px = x + 0.5, py = y + 0.5;
                double minX = fmin(a[0], fmin(b[0], cc[0])), maxX = fmax(a[0], fmax(b[0], cc[0]));
                double minY = fmin(a[1], fmin(b[1], cc[1])), maxY = fmax(a[1], fmax(b[1], cc[1]));
                if (px < minX - margin || px > maxX + margin || py < minY - margin || py > maxY + margin) continue;
                double e0 = REdge(a, b, px, py), e1 = REdge(b, cc, px, py), e2 = REdge(cc, a, px, py);
                if (fabs(e0) < margin || fabs(e1) < margin || fabs(e2) < margin) {
                    // A centre near the extension of an edge but outside
                    // the triangle is not ambiguous.
                    BOOL outside = (e0 < -margin || e1 < -margin || e2 < -margin) &&
                                   (e0 > margin || e1 > margin || e2 > margin);
                    if (!outside) return NO;
                    continue;
                }
                BOOL inside = (e0 > 0 && e1 > 0 && e2 > 0) || (e0 < 0 && e1 < 0 && e2 < 0);
                if (inside) map[y * width + x] = ids[t];
            }
    }
    return YES;
}
