// Draw-call cases: vertex and instance ids, indexed draws with base vertex
// and base instance, primitive restart, indirect draws, vertex descriptor
// formats and step functions, inline vertex bytes, buffer offsets, dynamic
// attribute strides and pipelines with rasterization disabled.
//
// Points of size 1 placed on pixel centres of a 16x16 R32/RGBA32 integer
// target carry the values under test; a point of size 1 centred on a pixel
// centre covers that pixel only, whatever the point shape.

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
BOOL RPaintTriangles(const double (*triangles)[6], const uint32_t *ids, NSUInteger count, NSUInteger width,
                     NSUInteger height, double margin, uint32_t *map);

// "Base vertex/instance drawing" and "Indirect draw and dispatch arguments":
// Apple3 and later (Metal Feature Set Tables, 2026-10-23).
static void DRequireBaseAndIndirect(WMCase *c) {
    [c require:@"family.apple3" value:[c.device supportsFamily:MTLGPUFamilyApple3]];
}

static id<MTLRenderPipelineState> DIdsPipeline(WMCase *c) {
    return [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_ids", @"rnd_fragment_id_data",
                                                 MTLPixelFormatRGBA32Uint)];
}

WMREF_CASE(drawVertexInstanceIds, "draw.vertex-instance-ids", "draw.instanced",
           "vertex_id and instance_id for drawPrimitives with vertexStart, instanceCount and baseInstance")
{
    DRequireBaseAndIndirect(c);
    id<MTLRenderPipelineState> pipeline = DIdsPipeline(c);
    id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA32Uint, 16, 16);
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        // vertex_id runs from vertexStart through vertexStart + vertexCount - 1;
        // instance_id from baseInstance through baseInstance + instanceCount - 1.
        [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:4 instanceCount:2];
        [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:5 vertexCount:6 instanceCount:3 baseInstance:9];
    }];
    // Each written pixel holds (vertex_id + 1, instance_id + 1, 0xa5, 0).
    RRecordTexture(c, @"ids", target, WMElemU32, 16, WMExact());
}

WMREF_CASE(drawIndexed, "draw.indexed", "draw.indexed",
           "drawIndexedPrimitives with uint16 and uint32 indices, index buffer offsets, base vertex and base instance")
{
    DRequireBaseAndIndirect(c);
    id<MTLRenderPipelineState> pipeline = DIdsPipeline(c);
    // uint16 indices start at byte 6; uint32 indices start at byte 8.
    const uint16_t indices16[] = {0xdead, 0xbeef, 0xf00d, 0, 1, 2, 5, 9, 3};
    const uint32_t indices32[] = {0xdeadbeef, 0xfeedface, 10, 12, 14, 7};
    id<MTLBuffer> buffer16 = [c bufferWithBytes:indices16 length:sizeof indices16];
    id<MTLBuffer> buffer32 = [c bufferWithBytes:indices32 length:sizeof indices32];
    id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA32Uint, 16, 16);
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        // vertex_id = index + baseVertex.
        [e drawIndexedPrimitives:MTLPrimitiveTypePoint indexCount:6 indexType:MTLIndexTypeUInt16
                     indexBuffer:buffer16 indexBufferOffset:6 instanceCount:2 baseVertex:4 baseInstance:7];
        [e drawIndexedPrimitives:MTLPrimitiveTypePoint indexCount:4 indexType:MTLIndexTypeUInt32
                     indexBuffer:buffer32 indexBufferOffset:8 instanceCount:1 baseVertex:-3 baseInstance:12];
        [e drawIndexedPrimitives:MTLPrimitiveTypePoint indexCount:3 indexType:MTLIndexTypeUInt16
                     indexBuffer:buffer16 indexBufferOffset:8];
    }];
    RRecordTexture(c, @"ids", target, WMElemU32, 16, WMExact());
}

// A strip of two separate quads joined by a restart index. Corners lie on
// integer pixel coordinates and every pixel centre is at least 1/16 pixel
// from the quads' diagonals (checked in the case).
static void DStripQuads(RVertex *v, NSUInteger width, NSUInteger height) {
    const double corners[8][2] = {{1, 1}, {1, 5}, {7, 1}, {7, 5}, {9, 8}, {9, 14}, {14, 8}, {14, 14}};
    for (int i = 0; i < 8; i++) v[i] = RVertexAt(corners[i][0], corners[i][1], 0, width, height, (uint32_t)(i / 4) + 1);
}

WMREF_CASE(drawPrimitiveRestart, "draw.primitive-restart", "draw.primitive-restart",
           "Primitive restart in indexed triangle and line strips with 0xFFFF and 0xFFFFFFFF")
{
    enum { W = 16, H = 16 };
    // Apple's documentation of drawIndexedPrimitives: "You can complete a
    // primitive and start a new one by passing a sentinel index value that's
    // the largest unsigned integer possible for indexType".
    RVertex vertices[8];
    DStripQuads(vertices, W, H);
    id<MTLBuffer> vertexBuffer = RVertexBuffer(c, vertices, 8);
    const uint16_t strip16[] = {0, 1, 2, 3, 0xffff, 4, 5, 6, 7};
    const uint32_t strip32[] = {0, 1, 2, 3, 0xffffffff, 4, 5, 6, 7};
    id<MTLBuffer> buffer16 = [c bufferWithBytes:strip16 length:sizeof strip16];
    id<MTLBuffer> buffer32 = [c bufferWithBytes:strip32 length:sizeof strip32];

    // Expected triangle-strip coverage: triangles (0,1,2), (1,2,3), (4,5,6), (5,6,7).
    const double triangles[4][6] = {
        {1, 1, 1, 5, 7, 1}, {1, 5, 7, 1, 7, 5}, {9, 8, 9, 14, 14, 8}, {9, 14, 14, 8, 14, 14},
    };
    const uint32_t ids[4] = {1, 1, 2, 2};
    uint32_t expected[W * H] = {0};
    if (!RPaintTriangles(triangles, ids, 4, W, H, 1.0 / 16, expected))
        [c fail:@"a pixel centre lies within the margin of an edge"];

    id<MTLRenderPipelineState> pipeline =
        [c renderPipeline:RPipelineDescriptor(c, @"rnd_vertex_triangles", @"rnd_fragment_id", MTLPixelFormatR32Uint)];
    const struct { MTLPrimitiveType type; MTLIndexType indexType; __unsafe_unretained NSString *name; } draws[] = {
        {MTLPrimitiveTypeTriangleStrip, MTLIndexTypeUInt16, @"triangle-strip.uint16"},
        {MTLPrimitiveTypeTriangleStrip, MTLIndexTypeUInt32, @"triangle-strip.uint32"},
        {MTLPrimitiveTypeLineStrip, MTLIndexTypeUInt16, @"line-strip.uint16"},
        {MTLPrimitiveTypeLineStrip, MTLIndexTypeUInt32, @"line-strip.uint32"},
    };
    for (int d = 0; d < 4; d++) {
        id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, W, H);
        MTLPrimitiveType type = draws[d].type;
        MTLIndexType indexType = draws[d].indexType;
        id<MTLBuffer> indices = indexType == MTLIndexTypeUInt16 ? buffer16 : buffer32;
        [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:pipeline];
            [e setVertexBuffer:vertexBuffer offset:0 atIndex:0];
            [e drawIndexedPrimitives:type indexCount:9 indexType:indexType indexBuffer:indices indexBufferOffset:0];
        }];
        if (type == MTLPrimitiveTypeTriangleStrip) {
            NSData *image = RRecordTexture(c, draws[d].name, target, WMElemU32, 4, WMExact());
            [c value:[draws[d].name stringByAppendingString:@".equals-two-separate-quads"]
                boolean:memcmp(image.bytes, expected, sizeof expected) == 0 rule:WMExact()];
        } else {
            RRecordTexture(c, draws[d].name, target, WMElemU32, 4,
                           WMNone(@"Metal's documentation does not define line rasterization"));
        }
    }
}

WMREF_CASE(drawIndirect, "draw.indirect", "draw.indirect",
           "drawPrimitives and drawIndexedPrimitives reading indirect arguments at buffer offsets")
{
    DRequireBaseAndIndirect(c);
    id<MTLRenderPipelineState> pipeline = DIdsPipeline(c);
    // MTLDrawPrimitivesIndirectArguments {vertexCount, instanceCount,
    // vertexStart, baseInstance} at byte 16; MTLDrawIndexedPrimitivesIndirectArguments
    // {indexCount, instanceCount, indexStart, baseVertex, baseInstance} at byte 48.
    uint32_t arguments[24] = {0};
    for (int i = 0; i < 24; i++) arguments[i] = 0xcccccccc;
    const uint32_t direct[4] = {3, 2, 4, 6};
    memcpy(&arguments[4], direct, sizeof direct);
    const int32_t indexed[5] = {4, 2, 1, 5, 11};
    memcpy(&arguments[12], indexed, sizeof indexed);
    id<MTLBuffer> indirect = [c bufferWithBytes:arguments length:sizeof arguments];
    const uint16_t indices[] = {0xffff, 0, 2, 4, 6, 8};
    id<MTLBuffer> indexBuffer = [c bufferWithBytes:indices length:sizeof indices];
    id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA32Uint, 16, 16);
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e drawPrimitives:MTLPrimitiveTypePoint indirectBuffer:indirect indirectBufferOffset:16];
        [e drawIndexedPrimitives:MTLPrimitiveTypePoint indexType:MTLIndexTypeUInt16 indexBuffer:indexBuffer
               indexBufferOffset:0 indirectBuffer:indirect indirectBufferOffset:48];
    }];
    [c data:@"arguments" elem:WMElemU32 bytes:arguments length:sizeof arguments rule:WMExact()];
    RRecordTexture(c, @"ids", target, WMElemU32, 16, WMExact());
}

#pragma mark Vertex formats

typedef NS_ENUM(int, DFetch) { DFetchUint, DFetchInt, DFetchFloat };

typedef struct {
    MTLVertexFormat format;
    const char *name;
    int size;        // bytes per element
    int components;  // components present in the format
    int kind;        // 0 unsigned integer, 1 signed integer, 2 float/half, 3 normalized, 4 packed float
} DFormat;

static const DFormat kFormats[] = {
    {MTLVertexFormatUChar, "UChar", 1, 1, 0}, {MTLVertexFormatUChar2, "UChar2", 2, 2, 0},
    {MTLVertexFormatUChar3, "UChar3", 3, 3, 0}, {MTLVertexFormatUChar4, "UChar4", 4, 4, 0},
    {MTLVertexFormatChar, "Char", 1, 1, 1}, {MTLVertexFormatChar2, "Char2", 2, 2, 1},
    {MTLVertexFormatChar3, "Char3", 3, 3, 1}, {MTLVertexFormatChar4, "Char4", 4, 4, 1},
    {MTLVertexFormatUShort, "UShort", 2, 1, 0}, {MTLVertexFormatUShort2, "UShort2", 4, 2, 0},
    {MTLVertexFormatUShort3, "UShort3", 6, 3, 0}, {MTLVertexFormatUShort4, "UShort4", 8, 4, 0},
    {MTLVertexFormatShort, "Short", 2, 1, 1}, {MTLVertexFormatShort2, "Short2", 4, 2, 1},
    {MTLVertexFormatShort3, "Short3", 6, 3, 1}, {MTLVertexFormatShort4, "Short4", 8, 4, 1},
    {MTLVertexFormatUInt, "UInt", 4, 1, 0}, {MTLVertexFormatUInt2, "UInt2", 8, 2, 0},
    {MTLVertexFormatUInt3, "UInt3", 12, 3, 0}, {MTLVertexFormatUInt4, "UInt4", 16, 4, 0},
    {MTLVertexFormatInt, "Int", 4, 1, 1}, {MTLVertexFormatInt2, "Int2", 8, 2, 1},
    {MTLVertexFormatInt3, "Int3", 12, 3, 1}, {MTLVertexFormatInt4, "Int4", 16, 4, 1},
    {MTLVertexFormatHalf, "Half", 2, 1, 2}, {MTLVertexFormatHalf2, "Half2", 4, 2, 2},
    {MTLVertexFormatHalf3, "Half3", 6, 3, 2}, {MTLVertexFormatHalf4, "Half4", 8, 4, 2},
    {MTLVertexFormatFloat, "Float", 4, 1, 2}, {MTLVertexFormatFloat2, "Float2", 8, 2, 2},
    {MTLVertexFormatFloat3, "Float3", 12, 3, 2}, {MTLVertexFormatFloat4, "Float4", 16, 4, 2},
    {MTLVertexFormatUCharNormalized, "UCharNormalized", 1, 1, 3},
    {MTLVertexFormatUChar2Normalized, "UChar2Normalized", 2, 2, 3},
    {MTLVertexFormatUChar3Normalized, "UChar3Normalized", 3, 3, 3},
    {MTLVertexFormatUChar4Normalized, "UChar4Normalized", 4, 4, 3},
    {MTLVertexFormatCharNormalized, "CharNormalized", 1, 1, 3},
    {MTLVertexFormatChar2Normalized, "Char2Normalized", 2, 2, 3},
    {MTLVertexFormatChar3Normalized, "Char3Normalized", 3, 3, 3},
    {MTLVertexFormatChar4Normalized, "Char4Normalized", 4, 4, 3},
    {MTLVertexFormatUShortNormalized, "UShortNormalized", 2, 1, 3},
    {MTLVertexFormatUShort2Normalized, "UShort2Normalized", 4, 2, 3},
    {MTLVertexFormatUShort3Normalized, "UShort3Normalized", 6, 3, 3},
    {MTLVertexFormatUShort4Normalized, "UShort4Normalized", 8, 4, 3},
    {MTLVertexFormatShortNormalized, "ShortNormalized", 2, 1, 3},
    {MTLVertexFormatShort2Normalized, "Short2Normalized", 4, 2, 3},
    {MTLVertexFormatShort3Normalized, "Short3Normalized", 6, 3, 3},
    {MTLVertexFormatShort4Normalized, "Short4Normalized", 8, 4, 3},
    {MTLVertexFormatUInt1010102Normalized, "UInt1010102Normalized", 4, 4, 3},
    {MTLVertexFormatInt1010102Normalized, "Int1010102Normalized", 4, 4, 3},
    {MTLVertexFormatUChar4Normalized_BGRA, "UChar4Normalized_BGRA", 4, 4, 3},
    {MTLVertexFormatFloatRG11B10, "FloatRG11B10", 4, 3, 4},
    {MTLVertexFormatFloatRGB9E5, "FloatRGB9E5", 4, 3, 4},
};

// Makes every float, half and packed-float element finite (and binary32
// elements normal), so the fetched values are defined bit patterns.
static void DSanitize(const DFormat *f, uint8_t *element) {
    if (f->kind == 2 && f->size / f->components == 4) {
        for (int k = 0; k < f->components; k++) {
            uint32_t bits;
            memcpy(&bits, element + 4 * k, 4);
            bits = (bits & 0x807fffff) | ((0x70u + (bits >> 23) % 0x1f) << 23);
            memcpy(element + 4 * k, &bits, 4);
        }
    } else if (f->kind == 2) {
        for (int k = 0; k < f->components; k++) {
            uint16_t bits;
            memcpy(&bits, element + 2 * k, 2);
            if (((bits >> 10) & 0x1f) == 0x1f) bits &= ~(uint16_t)0x4000;
            memcpy(element + 2 * k, &bits, 2);
        }
    } else if (f->format == MTLVertexFormatFloatRG11B10) {
        uint32_t bits;
        memcpy(&bits, element, 4);
        // R: bits 0-10, G: 11-21, B: 22-31; exponent fields 31 are NaN or
        // infinity (MSL 4.1 section 8.7.4).
        if (((bits >> 6) & 0x1f) == 0x1f) bits &= ~(1u << 10);
        if (((bits >> 17) & 0x1f) == 0x1f) bits &= ~(1u << 21);
        if (((bits >> 27) & 0x1f) == 0x1f) bits &= ~(1u << 31);
        memcpy(element, &bits, 4);
    }
}

static NSString *DFetchName(DFetch fetch) {
    return fetch == DFetchUint ? @"uint4" : fetch == DFetchInt ? @"int4" : @"float4";
}

// Draws 16 points whose attribute 0 is fetched from a buffer of 16
// elements of `format` (stride 16) as `fetch`, and returns the fetched
// values (four 32-bit words per point).
static NSData *DFetchFormat(WMCase *c, const DFormat *f, DFetch fetch, const uint8_t *elements) {
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(
        c, fetch == DFetchUint ? @"rnd_vertex_fetch_uint" : fetch == DFetchInt ? @"rnd_vertex_fetch_int"
                                                                              : @"rnd_vertex_fetch_float",
        @"rnd_fragment_id_data", MTLPixelFormatRGBA32Uint);
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    vd.attributes[0].format = f->format;
    vd.attributes[0].offset = 0;
    vd.attributes[0].bufferIndex = 1;
    vd.layouts[1].stride = 16;
    vd.layouts[1].stepFunction = MTLVertexStepFunctionPerVertex;
    d.vertexDescriptor = vd;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    id<MTLBuffer> buffer = [c bufferWithBytes:elements length:16 * 16];
    id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA32Uint, 16, 16);
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:buffer offset:0 atIndex:1];
        [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:16];
    }];
    NSData *image = RRead(c, target, 16);
    return [image subdataWithRange:NSMakeRange(0, 16 * 16)];
}

static BOOL DFormatIsPackedFloat(const DFormat *f) {
    return f->format == MTLVertexFormatFloatRG11B10 || f->format == MTLVertexFormatFloatRGB9E5;
}

static void DRecordFormat(WMCase *c, const DFormat *f);

WMREF_CASE(drawVertexFormats, "vertex.formats", "vertex.formats",
           "Every MTLVertexFormat fetched through a vertex descriptor into uint4, int4 or float4 attributes")
{
    // Conversion rules from the documentation of
    // MTLVertexAttributeDescriptor.format: integer formats convert to integer
    // arguments of the same sign and to float; components missing from the
    // format are filled from (0, 0, 0, 1). Integer, float and half values are
    // represented exactly (half to float is lossless, MSL 4.1 section 8.6;
    // 8- and 16-bit integers are exact in binary32). The MSL specification
    // states no precision for normalized vertex formats, so those values are
    // reference data only.
    for (size_t i = 0; i < sizeof kFormats / sizeof kFormats[0]; i++) {
        const DFormat *f = &kFormats[i];
        if (!DFormatIsPackedFloat(f)) DRecordFormat(c, f);
    }
}

WMREF_CASE(drawVertexFormatsPackedFloat, "vertex.formats-packed-float", "vertex.formats",
           "MTLVertexFormatFloatRG11B10 and FloatRGB9E5 fetched into float4 attributes")
{
    // "MTLAttributeFormat.floatRGB9E5 and .floatRG11B10": Apple5 and later
    // (Metal Feature Set Tables, 2026-10-23).
    [c require:@"family.apple5" value:[c.device supportsFamily:MTLGPUFamilyApple5]];
    for (size_t i = 0; i < sizeof kFormats / sizeof kFormats[0]; i++) {
        const DFormat *f = &kFormats[i];
        if (DFormatIsPackedFloat(f)) DRecordFormat(c, f);
    }
}

static void DRecordFormat(WMCase *c, const DFormat *f) {
    uint8_t elements[16 * 16];
    WMPattern(elements, sizeof elements, 0x500 + (uint32_t)f->format);
    for (int v = 0; v < 16; v++) DSanitize(f, elements + 16 * v);
    NSString *name = @(f->name);
    [c data:[name stringByAppendingString:@".input"] elem:WMElemU8 bytes:elements length:sizeof elements
       rule:WMExact()];
    DFetch fetches[2];
    int fetchCount = 0;
    if (f->kind == 0) fetches[fetchCount++] = DFetchUint;
    if (f->kind == 1) fetches[fetchCount++] = DFetchInt;
    if ((f->kind == 0 || f->kind == 1) && f->size / f->components <= 2) fetches[fetchCount++] = DFetchFloat;
    if (f->kind >= 2) fetches[fetchCount++] = DFetchFloat;
    for (int k = 0; k < fetchCount; k++) {
        NSData *fetched = DFetchFormat(c, f, fetches[k], elements);
        NSString *record = [NSString stringWithFormat:@"%@.as-%@", name, DFetchName(fetches[k])];
        WMRule *rule = f->kind == 3
            ? WMNone(@"the MSL specification states no precision for normalized vertex attribute conversion")
            : WMExact();
        [c data:record elem:fetches[k] == DFetchFloat ? WMElemF32 : WMElemU32 bytes:fetched.bytes
         length:fetched.length rule:rule];
    }
}

WMREF_CASE(drawStepFunctions, "vertex.step-functions", "vertex.step-functions",
           "Per-vertex, per-instance (step rate 2) and constant step functions")
{
    uint32_t perVertex[16 * 2], perInstance[16 * 2];
    for (uint32_t i = 0; i < 16; i++) {
        perVertex[2 * i] = 100 + i;
        perVertex[2 * i + 1] = 200 + i;
        perInstance[2 * i] = 300 + i;
        perInstance[2 * i + 1] = 400 + i;
    }
    id<MTLBuffer> vertexData = [c bufferWithBytes:perVertex length:sizeof perVertex];
    id<MTLBuffer> instanceData = [c bufferWithBytes:perInstance length:sizeof perInstance];
    const MTLVertexStepFunction functions[2] = {MTLVertexStepFunctionPerInstance, MTLVertexStepFunctionConstant};
    NSString *const names[2] = {@"per-instance-rate-2", @"constant"};
    for (int s = 0; s < 2; s++) {
        MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_fetch_two", @"rnd_fragment_id_data",
                                                             MTLPixelFormatRGBA32Uint);
        MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
        vd.attributes[0].format = MTLVertexFormatUInt2;
        vd.attributes[0].bufferIndex = 1;
        vd.attributes[1].format = MTLVertexFormatUInt2;
        vd.attributes[1].bufferIndex = 2;
        vd.layouts[1].stride = 8;
        vd.layouts[1].stepFunction = MTLVertexStepFunctionPerVertex;
        vd.layouts[2].stride = 8;
        vd.layouts[2].stepFunction = functions[s];
        // A constant step function requires a step rate of 0.
        vd.layouts[2].stepRate = s == 0 ? 2 : 0;
        d.vertexDescriptor = vd;
        id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
        id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA32Uint, 16, 16);
        [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:pipeline];
            [e setVertexBuffer:vertexData offset:0 atIndex:1];
            [e setVertexBuffer:instanceData offset:0 atIndex:2];
            [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:4 instanceCount:5];
        }];
        NSData *image = RRead(c, target, 16);
        RRecordRegion(c, names[s], image, 16, 16, @"RGBA32Uint", 0, 0, 4, 5, WMElemU32, WMExact());
    }
}

WMREF_CASE(drawStepBaseInstance, "vertex.step-base-instance", "vertex.step-functions",
           "Per-instance attribute fetch index with a nonzero baseInstance")
{
    DRequireBaseAndIndirect(c);
    uint32_t perVertex[4 * 2], perInstance[16 * 2];
    for (uint32_t i = 0; i < 4; i++) perVertex[2 * i] = perVertex[2 * i + 1] = i;
    for (uint32_t i = 0; i < 16; i++) perInstance[2 * i] = perInstance[2 * i + 1] = 300 + i;
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_fetch_two", @"rnd_fragment_id_data",
                                                         MTLPixelFormatRGBA32Uint);
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    vd.attributes[0].format = MTLVertexFormatUInt2;
    vd.attributes[0].bufferIndex = 1;
    vd.attributes[1].format = MTLVertexFormatUInt2;
    vd.attributes[1].bufferIndex = 2;
    vd.layouts[1].stride = 8;
    vd.layouts[2].stride = 8;
    vd.layouts[2].stepFunction = MTLVertexStepFunctionPerInstance;
    vd.layouts[2].stepRate = 2;
    d.vertexDescriptor = vd;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    id<MTLBuffer> vertexData = [c bufferWithBytes:perVertex length:sizeof perVertex];
    id<MTLBuffer> instanceData = [c bufferWithBytes:perInstance length:sizeof perInstance];
    id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA32Uint, 16, 16);
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:vertexData offset:0 atIndex:1];
        [e setVertexBuffer:instanceData offset:0 atIndex:2];
        [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:2 instanceCount:4 baseInstance:3];
    }];
    NSData *image = RRead(c, target, 16);
    // Rows 3..6 hold instances 3..6; component z is the per-instance value.
    RRecordRegion(c, @"fetched", image, 16, 16, @"RGBA32Uint", 0, 3, 2, 4, WMElemU32,
                  WMNone(@"Apple's documentation of MTLVertexStepFunctionPerInstance does not state how "
                         @"baseInstance enters the fetch index"));
}

WMREF_CASE(drawVertexBytesAndOffset, "vertex.bytes-and-buffer-offset", "vertex.bytes",
           "setVertexBytes and setVertexBufferOffset between draws")
{
    DRequireBaseAndIndirect(c);
    MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_fetch_uint", @"rnd_fragment_id_data",
                                                         MTLPixelFormatRGBA32Uint);
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    vd.attributes[0].format = MTLVertexFormatUInt2;
    vd.attributes[0].bufferIndex = 1;
    vd.layouts[1].stride = 8;
    d.vertexDescriptor = vd;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    uint32_t bytes[8], buffered[32];
    for (uint32_t i = 0; i < 8; i++) bytes[i] = 0x1000 + i;
    for (uint32_t i = 0; i < 32; i++) buffered[i] = 0x2000 + i;
    id<MTLBuffer> buffer = [c bufferWithBytes:buffered length:sizeof buffered];
    const uint32_t *bytesPointer = bytes;
    id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA32Uint, 16, 16);
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        // Row 0: inline bytes.
        [e setVertexBytes:bytesPointer length:8 * sizeof(uint32_t) atIndex:1];
        [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:4];
        // Row 1: buffer at offset 0, then rows 2 and 3 at offsets 40 and 64.
        [e setVertexBuffer:buffer offset:0 atIndex:1];
        [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:4 instanceCount:1 baseInstance:1];
        [e setVertexBufferOffset:40 atIndex:1];
        [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:4 instanceCount:1 baseInstance:2];
        [e setVertexBufferOffset:64 atIndex:1];
        [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:4 instanceCount:1 baseInstance:3];
    }];
    NSData *image = RRead(c, target, 16);
    RRecordRegion(c, @"fetched", image, 16, 16, @"RGBA32Uint", 0, 0, 4, 4, WMElemU32, WMExact());
}

WMREF_CASE(drawAttributeStride, "vertex.attribute-stride", "vertex.attribute-stride",
           "MTLBufferLayoutStrideDynamic with setVertexBuffer:offset:attributeStride:atIndex:")
{
    DRequireBaseAndIndirect(c);
    // "Dynamic stride for attribute buffers": Apple4 and later
    // (Metal Feature Set Tables, 2026-10-23).
    [c require:@"family.apple4" value:[c.device supportsFamily:MTLGPUFamilyApple4]];
    if (@available(macOS 14.0, iOS 17.0, *)) {
        MTLRenderPipelineDescriptor *d = RPipelineDescriptor(c, @"rnd_vertex_fetch_uint", @"rnd_fragment_id_data",
                                                             MTLPixelFormatRGBA32Uint);
        MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
        vd.attributes[0].format = MTLVertexFormatUInt2;
        vd.attributes[0].bufferIndex = 1;
        vd.layouts[1].stride = MTLBufferLayoutStrideDynamic;
        d.vertexDescriptor = vd;
        id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
        uint32_t words[64];
        for (uint32_t i = 0; i < 64; i++) words[i] = 0x3000 + i;
        id<MTLBuffer> buffer = [c bufferWithBytes:words length:sizeof words];
        id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA32Uint, 16, 16);
        [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:pipeline];
            [e setVertexBuffer:buffer offset:0 attributeStride:8 atIndex:1];
            [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:4];
            [e setVertexBufferOffset:4 attributeStride:16 atIndex:1];
            [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:4 instanceCount:1 baseInstance:1];
            [e setVertexBuffer:buffer offset:8 attributeStride:12 atIndex:1];
            [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:4 instanceCount:1 baseInstance:2];
        }];
        NSData *image = RRead(c, target, 16);
        RRecordRegion(c, @"fetched", image, 16, 16, @"RGBA32Uint", 0, 0, 4, 3, WMElemU32, WMExact());
    } else {
        [c require:@"os.attribute-stride" value:NO];
    }
}

WMREF_CASE(drawRasterizationDisabled, "draw.rasterization-disabled", "draw.rasterization-disabled",
           "Pipeline with rasterizationEnabled = NO whose vertex function writes a buffer")
{
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [c function:@"rnd_vertex_store"];
    d.rasterizationEnabled = NO;
    d.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:d];
    id<MTLBuffer> out = [c buffer:3 * 8 * sizeof(uint32_t)];
    id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, 8, 8);
    [c render:RPassColor(target, MTLClearColorMake(7, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setVertexBuffer:out offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:8 instanceCount:3];
    }];
    [c data:@"written" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
    // No fragments are generated: the attachment keeps its clear value.
    RRecordTexture(c, @"attachment", target, WMElemU32, 4, WMExact());
}

WMREF_CASE(drawUnalignedIndexOffset, "draw.error.index-offset-unaligned", "draw.indexed",
           "drawIndexedPrimitives with a uint32 index buffer offset that is not a multiple of 4 (invalid use)")
{
    id<MTLRenderPipelineState> pipeline = DIdsPipeline(c);
    const uint8_t bytes[16] = {0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0};
    id<MTLBuffer> indices = [c bufferWithBytes:bytes length:sizeof bytes];
    id<MTLTexture> target = RTarget(c, MTLPixelFormatRGBA32Uint, 16, 16);
    [c step:@"draw"];
    [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e drawIndexedPrimitives:MTLPrimitiveTypePoint indexCount:3 indexType:MTLIndexTypeUInt32
                     indexBuffer:indices indexBufferOffset:2];
    }];
    [c step:@"completed"];
    RRecordTexture(c, @"ids", target, WMElemU32, 16,
                   WMNone(@"the result of an unaligned index buffer offset is not defined"));
}
