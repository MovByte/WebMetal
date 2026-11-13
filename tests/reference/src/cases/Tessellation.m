// Tessellation cases: the domain points and triangles the tessellator
// generates for each partition mode over a sweep of factors, in the order the
// triangles are rasterized, and how a patch draw finds each patch's factors,
// patch identifier and instance.
//
// The order is read back through rasterization (shaders/tessellation.metal):
// each generated triangle is drawn at pseudo-random window positions chosen per
// point and per draw, and every fragment appends its triangle to a list per
// pixel in primitive order. Two triangles drawn one after the other in the
// draw appear one after the other in the list of every pixel both cover. Order
// draws are repeated with new positions until each triangle is followed
// directly by the next one in some pixel's list; the order is then the only
// one the lists allow. A triangle with no area (two of its points at one
// position, which small fractional factors produce) covers no pixel and is
// not recorded.

#import "WMRefCase.h"

// RenderSupport.m.
id<MTLTexture> RTarget(WMCase *c, MTLPixelFormat format, NSUInteger width, NSUInteger height);
MTLRenderPassDescriptor *RPassColor(id<MTLTexture> color, MTLClearColor clear);
NSData *RRecordTexture(WMCase *c, NSString *name, id<MTLTexture> texture, WMElem elem, NSUInteger bytesPerPixel,
                       WMRule *rule);

typedef struct {
    uint32_t instance, patch, u, v, w;
} TPoint;

typedef struct {
    uint32_t p[3];
} TTri;

typedef struct {
    uint32_t count, seed, width, capacity;
} TOrderArgs;

// One patch draw.
typedef struct {
    BOOL quad;
    MTLTessellationPartitionMode mode;
    NSUInteger maxFactor;
    MTLWinding winding;
    BOOL scaleEnabled;
    float scale;
    BOOL scaleUnset; // scaling enabled and setTessellationFactorScale: never called; recorded as a scale of 0
    MTLTessellationFactorStepFunction step;
    const uint16_t *factors; // binary16 bit patterns
    NSUInteger factorHalves;
    NSUInteger offset, instanceStride;
    NSUInteger patchStart, patchCount, instanceCount, baseInstance;
    const uint32_t *patchIndices;
    NSUInteger patchIndexCount;
} TDraw;

enum { TWidth = 32, TListCapacity = 2048, TPointCapacity = 1 << 18, TMaxOrderDraws = 96 };

static NSString *const TReason =
    @"Metal's tessellation documentation leaves the subdivision implementation-dependent; the tessellator of the "
    @"WebMetal executor is compared with these values";

static int TPointCompare(const void *x, const void *y) {
    const uint32_t *a = x, *b = y;
    for (int i = 0; i < 5; i++)
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}

static int TTriCompare(const void *x, const void *y) { return memcmp(x, y, sizeof(TTri)); }

static id<MTLRenderPipelineState> TPipeline(WMCase *c, const TDraw *d, BOOL order) {
    MTLRenderPipelineDescriptor *p = [MTLRenderPipelineDescriptor new];
    NSString *suffix = d->quad ? @"quad" : @"triangle";
    p.vertexFunction = [c function:[(order ? @"tess_order_" : @"tess_enum_") stringByAppendingString:suffix]];
    if (order) p.fragmentFunction = [c function:@"tess_order_fragment"];
    p.rasterizationEnabled = order;
    p.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
    p.tessellationPartitionMode = d->mode;
    p.maxTessellationFactor = d->maxFactor;
    p.tessellationFactorScaleEnabled = d->scaleEnabled;
    p.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
    p.tessellationFactorStepFunction = d->step;
    p.tessellationOutputWindingOrder = d->winding;
    p.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeNone;
    return [c renderPipeline:p];
}

static void TDrawPatches(const TDraw *d, id<MTLRenderCommandEncoder> e, id<MTLBuffer> factors,
                         id<MTLBuffer> patchIndices) {
    [e setTessellationFactorBuffer:factors offset:d->offset instanceStride:d->instanceStride];
    if (d->scaleEnabled && !d->scaleUnset) [e setTessellationFactorScale:d->scale];
    [e drawPatches:d->quad ? 4 : 3
            patchStart:d->patchStart
            patchCount:d->patchCount
      patchIndexBuffer:d->patchIndexCount ? patchIndices : nil
    patchIndexBufferOffset:0
         instanceCount:d->instanceCount
          baseInstance:d->baseInstance];
}

// Records the draw (`.input`, see below), its points, sorted by instance, patch and position (`.points`: u, v and w as binary32 bit
// patterns, w 0 on a quad; `.point-patches`: each point's instance and patch identifier, for draws of more than one
// patch), and its triangles in rasterization order (`.triangles`: each as its points' indices in the order of the
// triangle's vertices).
static void TRecordDraw(WMCase *c, NSString *name, const TDraw *d);

static void TRecord(WMCase *c, NSString *name, const TDraw *d) {
    @autoreleasepool {
        TRecordDraw(c, name, d);
    }
}

static void TRecordDraw(WMCase *c, NSString *name, const TDraw *d) {
    [c step:name];
    // The draw itself, for the programs that replay it: the pipeline's tessellation state, the encoder's factor
    // buffer state and the draw's arguments, then the factor buffer's binary16 values and the patch index buffer.
    NSMutableData *input = [NSMutableData data];
    float scale = d->scale;
    uint32_t scaleBits;
    memcpy(&scaleBits, &scale, 4);
    const uint32_t head[] = {d->quad, (uint32_t)d->mode, (uint32_t)d->maxFactor, (uint32_t)d->winding,
                             d->scaleEnabled, scaleBits, (uint32_t)d->step, (uint32_t)d->offset,
                             (uint32_t)d->instanceStride, (uint32_t)d->patchStart, (uint32_t)d->patchCount,
                             (uint32_t)d->instanceCount, (uint32_t)d->baseInstance, (uint32_t)d->factorHalves,
                             (uint32_t)d->patchIndexCount};
    [input appendBytes:head length:sizeof head];
    for (NSUInteger i = 0; i < d->factorHalves; i++) {
        uint32_t h = d->factors[i];
        [input appendBytes:&h length:4];
    }
    if (d->patchIndexCount) [input appendBytes:d->patchIndices length:4 * d->patchIndexCount];
    [c data:[name stringByAppendingString:@".input"] elem:WMElemU32 bytes:input.bytes length:input.length
       rule:WMExact()];
    id<MTLBuffer> factors = [c bufferWithBytes:d->factors length:MAX(2 * d->factorHalves, 4)];
    uint32_t noIndex = 0;
    id<MTLBuffer> patchIndices = d->patchIndexCount
                                     ? [c bufferWithBytes:d->patchIndices length:4 * d->patchIndexCount]
                                     : [c bufferWithBytes:&noIndex length:4];
    MTLTextureDescriptor *td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:TWidth height:TWidth
                                                       mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> target = [c texture:td];
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;

    // Enumeration.
    id<MTLRenderPipelineState> enumerate = TPipeline(c, d, NO);
    id<MTLBuffer> count = [c buffer:4];
    id<MTLBuffer> log = [c buffer:sizeof(TPoint) * TPointCapacity];
    uint32_t capacity = TPointCapacity;
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:enumerate];
        [e setVertexBuffer:count offset:0 atIndex:0];
        [e setVertexBuffer:log offset:0 atIndex:1];
        [e setVertexBytes:&capacity length:4 atIndex:2];
        TDrawPatches(d, e, factors, patchIndices);
    }];
    uint32_t invocations = *(const uint32_t *)count.contents;
    if (invocations > TPointCapacity) [c fail:@"%@: %u invocations exceed the capture", name, invocations];
    NSMutableData *pointData = [NSMutableData dataWithBytes:log.contents length:sizeof(TPoint) * invocations];
    TPoint *points = pointData.mutableBytes;
    qsort(points, invocations, sizeof(TPoint), TPointCompare);
    uint32_t n = 0;
    for (uint32_t i = 0; i < invocations; i++)
        if (n == 0 || TPointCompare(&points[n - 1], &points[i])) points[n++] = points[i];

    // Order draws.
    NSMutableData *triangleData = [NSMutableData data];
    NSMutableData *edgeData = [NSMutableData data]; // (from, to) pairs of triangles, as TTri pairs
    if (n > 0) {
        if (n > (1u << 16)) [c fail:@"%@: %u points exceed the 16-bit point index", name, n];
        id<MTLRenderPipelineState> order = TPipeline(c, d, YES);
        id<MTLBuffer> table = [c bufferWithBytes:points length:sizeof(TPoint) * n];
        id<MTLBuffer> lists = [c buffer:(NSUInteger)TWidth * TWidth * TListCapacity * 16];
        memset(lists.contents, 0xff, lists.length);
        for (uint32_t seed = 0; seed < TMaxOrderDraws; seed++) {
            TOrderArgs args = {n, seed, TWidth, TListCapacity};
            [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
                [e setRenderPipelineState:order];
                [e setVertexBuffer:table offset:0 atIndex:0];
                [e setVertexBytes:&args length:sizeof args atIndex:1];
                [e setFragmentBuffer:lists offset:0 atIndex:0];
                [e setFragmentBytes:&args length:sizeof args atIndex:1];
                TDrawPatches(d, e, factors, patchIndices);
            }];
            uint32_t *entries = lists.contents;
            for (uint32_t pixel = 0; pixel < TWidth * TWidth; pixel++) {
                TTri previous;
                BOOL hasPrevious = NO;
                for (uint32_t k = 0; k < TListCapacity; k++) {
                    uint32_t *r = entries + ((size_t)pixel * TListCapacity + k) * 4;
                    if (r[3] == 0xffffffffu) break;
                    if (k == TListCapacity - 1) [c fail:@"%@: a pixel's list overflowed", name];
                    if (r[3] == 1) {
                        TTri t = {{r[0], r[1], r[2]}};
                        if (t.p[0] >= n || t.p[1] >= n || t.p[2] >= n)
                            [c fail:@"%@: a fragment decoded a point outside the table", name];
                        [triangleData appendBytes:&t length:sizeof t];
                        if (hasPrevious && memcmp(&previous, &t, sizeof t)) {
                            [edgeData appendBytes:&previous length:sizeof previous];
                            [edgeData appendBytes:&t length:sizeof t];
                        }
                        previous = t;
                        hasPrevious = YES;
                    }
                    r[3] = 0xffffffffu;
                }
            }
            // Distinct triangles so far, and whether the edges order them uniquely.
            TTri *tris = triangleData.mutableBytes;
            size_t m = 0, all = triangleData.length / sizeof(TTri);
            qsort(tris, all, sizeof(TTri), TTriCompare);
            for (size_t i = 0; i < all; i++)
                if (m == 0 || memcmp(&tris[m - 1], &tris[i], sizeof(TTri))) tris[m++] = tris[i];
            triangleData.length = m * sizeof(TTri);
            if (seed + 1 < 16) continue;
            // Topological order with every consecutive pair joined by an edge.
            size_t ne = edgeData.length / (2 * sizeof(TTri));
            const TTri *pairs = edgeData.bytes;
            int *from = malloc(sizeof(int) * (ne + 1)), *to = malloc(sizeof(int) * (ne + 1));
            int *indegree = calloc(m + 1, sizeof(int)), *start = calloc(m + 2, sizeof(int));
            for (size_t i = 0; i < ne; i++) {
                from[i] = (int)((TTri *)bsearch(&pairs[2 * i], tris, m, sizeof(TTri), TTriCompare) - tris);
                to[i] = (int)((TTri *)bsearch(&pairs[2 * i + 1], tris, m, sizeof(TTri), TTriCompare) - tris);
                start[from[i] + 1]++;
            }
            for (size_t i = 0; i < m; i++) start[i + 1] += start[i];
            int *adjacent = malloc(sizeof(int) * (ne + 1)), *fill = calloc(m + 1, sizeof(int));
            for (size_t i = 0; i < ne; i++) {
                adjacent[start[from[i]] + fill[from[i]]++] = to[i];
                indegree[to[i]]++;
            }
            int *ordered = malloc(sizeof(int) * (m + 1)), *ready = malloc(sizeof(int) * (m + 1));
            size_t no = 0, nr = 0;
            BOOL unique = YES;
            for (size_t i = 0; i < m; i++)
                if (!indegree[i]) ready[nr++] = (int)i;
            while (nr) {
                if (nr > 1) unique = NO;
                int v = ready[--nr];
                ordered[no++] = v;
                for (int k = start[v]; k < start[v + 1]; k++)
                    if (!--indegree[adjacent[k]]) ready[nr++] = adjacent[k];
            }
            if (no != m) [c fail:@"%@: the per-pixel lists order the triangles inconsistently", name];
            for (size_t i = 0; unique && i + 1 < no; i++) {
                BOOL direct = NO;
                for (int k = start[ordered[i]]; k < start[ordered[i] + 1] && !direct; k++)
                    direct = adjacent[k] == ordered[i + 1];
                unique = direct;
            }
            if (unique) {
                NSMutableData *sorted = [NSMutableData dataWithLength:m * sizeof(TTri)];
                TTri *out = sorted.mutableBytes;
                for (size_t i = 0; i < m; i++) out[i] = tris[ordered[i]];
                triangleData = sorted;
            }
            free(from), free(to), free(indegree), free(start), free(adjacent), free(fill), free(ordered), free(ready);
            if (unique) break;
            if (seed + 1 == TMaxOrderDraws) [c fail:@"%@: the order draws did not order every triangle", name];
        }
    }
    WMRule *rule = WMNone(TReason);
    NSMutableData *positions = [NSMutableData dataWithLength:12 * n], *patches = [NSMutableData dataWithLength:8 * n];
    for (uint32_t i = 0; i < n; i++) {
        uint32_t *p = (uint32_t *)positions.mutableBytes + 3 * i, *q = (uint32_t *)patches.mutableBytes + 2 * i;
        p[0] = points[i].u, p[1] = points[i].v, p[2] = points[i].w;
        q[0] = points[i].instance, q[1] = points[i].patch;
    }
    size_t m = triangleData.length / sizeof(TTri);
    NSMutableData *triangles = [NSMutableData dataWithLength:6 * m];
    for (size_t i = 0; i < 3 * m; i++)
        ((uint16_t *)triangles.mutableBytes)[i] = (uint16_t)((const uint32_t *)triangleData.bytes)[i];
    [c data:[name stringByAppendingString:@".points"] elem:WMElemU32 bytes:positions.bytes length:positions.length
       rule:rule];
    if (d->patchCount * d->instanceCount > 1)
        [c data:[name stringByAppendingString:@".point-patches"] elem:WMElemU32 bytes:patches.bytes
         length:patches.length rule:rule];
    [c data:[name stringByAppendingString:@".triangles"] elem:WMElemU16 bytes:triangles.bytes length:triangles.length
       rule:rule];
}

static void TRequire(WMCase *c) {
    // "Tessellation": Apple3 and later; "Barycentric coordinates": Apple7 and later, and Mac2 (Metal Feature Set
    // Tables, 2026-10-23). Programmable blending: every Apple family.
    [c require:@"family.apple3" value:[c.device supportsFamily:MTLGPUFamilyApple3]];
    [c require:@"supportsShaderBarycentricCoordinates" value:c.device.supportsShaderBarycentricCoordinates];
}

static TDraw TSingle(BOOL quad, MTLTessellationPartitionMode mode, NSUInteger maxFactor, const uint16_t *factors) {
    TDraw d = {0};
    d.quad = quad;
    d.mode = mode;
    d.maxFactor = maxFactor;
    d.winding = MTLWindingClockwise;
    d.step = MTLTessellationFactorStepFunctionConstant;
    d.factors = factors;
    d.factorHalves = quad ? 6 : 4;
    d.patchCount = 1;
    d.instanceCount = 1;
    return d;
}

static NSString *const TModeNames[4] = {@"pow2", @"integer", @"fractional-odd", @"fractional-even"};

// Every partition mode over a sweep: every factor equal, then factors drawn from the generated-input pattern.
static void TSweep(WMCase *c, BOOL quad) {
    TRequire(c);
    const float uniform[] = {1, 1.5f, 2, 2.5f, 3, 3.75f, 4, 5, 6.5f, 7, 8, 9.25f, 11, 12, 13, 15.5f, 16, 17};
    NSUInteger n = quad ? 6 : 4;
    for (int m = 0; m < 4; m++) {
        for (size_t f = 0; f < sizeof uniform / sizeof uniform[0]; f++) {
            uint16_t h[6];
            for (NSUInteger k = 0; k < n; k++) h[k] = WMHalfFromFloat(uniform[f]);
            TDraw d = TSingle(quad, (MTLTessellationPartitionMode)m, 16, h);
            TRecord(c, [NSString stringWithFormat:@"%@.uniform-%zu", TModeNames[m], f], &d);
        }
        // Mixed factors in [0.5, 17): edges and insides independent.
        uint8_t bytes[12 * 6];
        WMPattern(bytes, sizeof bytes, 0x7e55 + m * 31 + (quad ? 7 : 0));
        for (int f = 0; f < 12; f++) {
            uint16_t h[6];
            for (NSUInteger k = 0; k < n; k++) h[k] = WMHalfFromFloat(0.5f + bytes[f * 6 + k] * (16.5f / 256));
            TDraw d = TSingle(quad, (MTLTessellationPartitionMode)m, 16, h);
            TRecord(c, [NSString stringWithFormat:@"%@.mixed-%d", TModeNames[m], f], &d);
        }
    }
}

WMREF_CASE(tessQuad, "tessellation.quad", "pass.tessellation",
           "Quad patches in every partition mode over a factor sweep: domain points and triangle order")
{
    TSweep(c, YES);
}

WMREF_CASE(tessTriangle, "tessellation.triangle", "pass.tessellation",
           "Triangle patches in every partition mode over a factor sweep: domain points and triangle order")
{
    TSweep(c, NO);
}

WMREF_CASE(tessFactors, "tessellation.factors", "pass.tessellation",
           "Tessellation factor processing: culling, clamping, rounding, scaling, the maximum factor and winding")
{
    TRequire(c);
    const float inf = INFINITY, nan = NAN;
    struct {
        const char *name;
        BOOL quad;
        int mode;
        NSUInteger maxFactor;
        float scale; // 0: scaling disabled
        MTLWinding winding;
        float f[6];
    } cases[] = {
        {"cull-edge-zero", YES, 1, 16, 0, 0, {0, 2, 2, 2, 3, 3}},
        {"cull-edge-negative", NO, 2, 16, 0, 0, {-1, 3, 3, 3}},
        {"cull-edge-nan", NO, 1, 16, 0, 0, {nan, 3, 3, 3}},
        {"edge-subnormal", YES, 1, 16, 0, 0, {0.00001f, 2, 2, 2, 3, 3}},
        {"edge-infinite", YES, 1, 16, 0, 0, {inf, 2, 2, 2, 5, 5}},
        {"inside-zero", YES, 1, 16, 0, 0, {2, 2, 2, 2, 0, 0}},
        {"inside-negative", NO, 3, 16, 0, 0, {3, 3, 3, -1}},
        {"inside-nan", NO, 2, 16, 0, 0, {3, 3, 3, nan}},
        {"inside-nan-quad", YES, 2, 16, 0, 0, {2, 2, 2, 2, nan, 3}},
        {"inside-infinite", YES, 1, 16, 0, 0, {2, 2, 2, 2, inf, inf}},
        {"below-one", YES, 1, 16, 0, 0, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}},
        {"minimum-quad", YES, 0, 16, 0, 0, {1, 1, 1, 1, 1, 1}},
        {"minimum-triangle", NO, 2, 16, 0, 0, {1, 1, 1, 1}},
        {"even-minimum", YES, 3, 16, 0, 0, {1, 1, 1, 1, 1, 1}},
        {"pow2-rounds-up", YES, 0, 16, 0, 0, {3, 5, 9, 1.5f, 6, 3}},
        {"pow2-clamp", YES, 0, 16, 0, 0, {17, 17, 17, 17, 17, 17}},
        {"max-8-pow2", YES, 0, 8, 0, 0, {5, 5, 5, 5, 7, 7}},
        {"max-8-odd", YES, 2, 8, 0, 0, {8, 8, 8, 8, 8, 8}},
        {"max-16-odd", NO, 2, 16, 0, 0, {16, 16, 16, 16}},
        {"max-12-even", NO, 3, 12, 0, 0, {13, 11.5f, 3, 12.5f}},
        {"max-10-integer", YES, 1, 10, 0, 0, {10.5f, 3, 11, 2, 9.5f, 12}},
        {"scale-1.5", YES, 2, 16, 1.5f, 0, {1.0009765625f, 1.0009765625f, 3.333f, 5.1f, 2.71f, 3.9f}},
        {"scale-0.7", YES, 3, 16, 0.7f, 0, {2.001f, 3.001f, 4.9f, 1.37f, 2.2f, 6.6f}},
        {"scale-1.333", NO, 2, 16, 1.333f, 0, {1.37f, 2.71f, 3.14f, 1.99f}},
        {"scale-2.71", YES, 2, 16, 2.71f, 0, {1.0009765625f, 1.0009765625f, 3.333f, 5.1f, 2.71f, 3.9f}},
        {"scale-pi", NO, 2, 16, 3.1416f, 0, {1.37f, 2.71f, 3.14f, 1.99f}},
        {"scale-zero", YES, 1, 16, 0.00000001f, 0, {3, 4, 5, 6, 2, 3}},
        {"scale-large", YES, 1, 16, 70000, 0, {3, 4, 5, 6, 2, 3}},
        {"winding-ccw-quad", YES, 1, 16, 0, MTLWindingCounterClockwise, {3, 5, 7, 2, 4, 6}},
        {"winding-ccw-quad-odd", YES, 2, 16, 0, MTLWindingCounterClockwise, {3.3f, 5.1f, 7.7f, 2.2f, 4.9f, 6.1f}},
        {"winding-ccw-triangle", NO, 1, 16, 0, MTLWindingCounterClockwise, {6, 6, 6, 6}},
        {"winding-ccw-triangle-odd", NO, 2, 16, 0, MTLWindingCounterClockwise, {5, 3, 7, 5}},
        {"winding-ccw-minimum", NO, 1, 16, 0, MTLWindingCounterClockwise, {1, 1, 1, 1}},
        {"max-64-thin", YES, 1, 64, 0, 0, {64, 40, 2, 33, 5, 64}},
        {"max-64-triangle", NO, 3, 64, 0, 0, {64, 63.5f, 21, 9}},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        uint16_t h[6];
        NSUInteger n = cases[i].quad ? 6 : 4;
        for (NSUInteger k = 0; k < n; k++) h[k] = WMHalfFromFloat(cases[i].f[k]);
        TDraw d = TSingle(cases[i].quad, cases[i].mode, cases[i].maxFactor, h);
        d.winding = cases[i].winding;
        d.scaleEnabled = cases[i].scale != 0;
        d.scale = cases[i].scale;
        TRecord(c, @(cases[i].name), &d);
    }
    // Scaling enabled on an encoder that never sets a scale: every patch is culled on the M4, as with a scale of 0,
    // which the draw's input records.
    uint16_t h[6];
    const float f[6] = {3.3f, 4.1f, 5.7f, 6.2f, 2.9f, 3.4f};
    for (int k = 0; k < 6; k++) h[k] = WMHalfFromFloat(f[k]);
    TDraw d = TSingle(YES, MTLTessellationPartitionModeFractionalOdd, 16, h);
    d.scaleEnabled = YES;
    d.scaleUnset = YES;
    TRecord(c, @"scale-unset", &d);
}

WMREF_CASE(tessDraws, "tessellation.draws", "pass.tessellation",
           "Patch draws: factor step functions, offsets and instance strides, patch start, patch index buffers, "
           "instances and base instance, and the order of patches and instances")
{
    TRequire(c);
    // 16 factor slots of 12 bytes (MTLQuadTessellationFactorsHalf); slot j has every factor (j % 6) + 1, slot 4
    // fractional.
    uint16_t slots[16 * 6];
    for (int j = 0; j < 16; j++)
        for (int k = 0; k < 6; k++) slots[j * 6 + k] = WMHalfFromFloat(j == 4 ? 2.5f + k : (float)(j % 6 + 1));
    const uint32_t indices[] = {9, 4, 7, 1, 12, 3, 0};
    struct {
        const char *name;
        MTLTessellationFactorStepFunction step;
        NSUInteger offset, stride, start, count, instances, base;
        BOOL indexed;
    } draws[] = {
        {"constant", MTLTessellationFactorStepFunctionConstant, 24, 0, 2, 3, 2, 3, NO},
        {"per-patch", MTLTessellationFactorStepFunctionPerPatch, 0, 0, 2, 4, 1, 0, NO},
        {"per-patch-offset", MTLTessellationFactorStepFunctionPerPatch, 12, 0, 0, 3, 2, 0, NO},
        {"per-patch-index-buffer", MTLTessellationFactorStepFunctionPerPatch, 0, 0, 1, 4, 1, 0, YES},
        {"per-instance", MTLTessellationFactorStepFunctionPerInstance, 0, 24, 0, 2, 3, 2, NO},
        {"per-patch-and-instance", MTLTessellationFactorStepFunctionPerPatchAndPerInstance, 12, 48, 1, 3, 2, 1, NO},
        {"per-patch-and-instance-index-buffer", MTLTessellationFactorStepFunctionPerPatchAndPerInstance, 0, 60, 2, 3, 2,
         0, YES},
    };
    for (size_t i = 0; i < sizeof draws / sizeof draws[0]; i++) {
        TDraw d = TSingle(YES, MTLTessellationPartitionModeInteger, 16, slots);
        d.factorHalves = sizeof slots / 2;
        d.step = draws[i].step;
        d.offset = draws[i].offset;
        d.instanceStride = draws[i].stride;
        d.patchStart = draws[i].start;
        d.patchCount = draws[i].count;
        d.instanceCount = draws[i].instances;
        d.baseInstance = draws[i].base;
        if (draws[i].indexed) {
            d.patchIndices = indices;
            d.patchIndexCount = sizeof indices / sizeof indices[0];
        }
        TRecord(c, @(draws[i].name), &d);
    }
    // A patch draw's state at each patch depends on the patches before it on the M4 (its batches of inside
    // triangles): large patches in sequence.
    uint16_t large[12];
    const float lf[12] = {2, 2, 2, 2, 24, 24, 5, 9, 3, 12, 40, 31};
    for (int k = 0; k < 12; k++) large[k] = WMHalfFromFloat(lf[k]);
    TDraw d = TSingle(YES, MTLTessellationPartitionModeInteger, 64, large);
    d.factorHalves = 12;
    d.step = MTLTessellationFactorStepFunctionPerPatch;
    d.patchCount = 2;
    TRecord(c, @"large-patches", &d);
}

WMREF_CASE(tessMisuse, "tessellation.misuse", "pass.tessellation",
           "Patch draws the API validation layer rejects: a factor scale that is not a positive normal value, and an "
           "instance stride with a step function that has no instances")
{
    TRequire(c);
    // The validation layer: "scale(-1) must be a normal floating point value (not denormal, infinite, nor NaN) and
    // must be > 0.0f", and "for MTLTessellationFactorStepFunctionConstant and
    // MTLTessellationFactorStepFunctionPerPatch tessellationFactorBufferInstanceStride(36) must be zero" (measured).
    // Without it, these record what the M4 does.
    uint16_t h[6];
    const float f[6] = {3, 4, 5, 6, 2, 3};
    for (int k = 0; k < 6; k++) h[k] = WMHalfFromFloat(f[k]);
    const float scales[] = {-1, NAN, 0};
    NSString *const names[] = {@"scale-negative", @"scale-nan", @"scale-zero"};
    for (int i = 0; i < 3; i++) {
        TDraw d = TSingle(YES, MTLTessellationPartitionModeFractionalOdd, 16, h);
        d.scaleEnabled = YES;
        d.scale = scales[i];
        TRecord(c, names[i], &d);
    }
    uint16_t slots[4 * 6];
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 6; k++) slots[j * 6 + k] = WMHalfFromFloat((float)(j + 1));
    TDraw d = TSingle(YES, MTLTessellationPartitionModeInteger, 16, slots);
    d.factorHalves = 24;
    d.step = MTLTessellationFactorStepFunctionPerPatch;
    d.instanceStride = 36;
    d.patchCount = 2;
    d.instanceCount = 2;
    TRecord(c, @"per-patch-instance-stride", &d);
}

#pragma mark Rendered draws

// tessellation.render: patch draws through post-tessellation vertex functions with control points and per-patch
// attributes (shaders/tessellation.metal), rasterized into R32Uint targets.

enum { TRenderSize = 32, TRenderPoints = 64, TRenderTags = 16, TRenderSlots = 24 };

typedef struct {
    const char *name;
    BOOL quad;
    MTLTessellationPartitionMode mode;
    MTLWinding winding;
    MTLTessellationFactorStepFunction step;
    MTLTessellationControlPointIndexType indexType;
    float scale; // 0: scaling disabled
    NSUInteger offset, stride, start, count, instances, base;
    BOOL patchIndices;
    NSUInteger indexOffset; // controlPointIndexBufferOffset
    BOOL texture;           // the fragment function reads the swizzled texture at index 0
} TRender;

WMREF_CASE(tessRender, "tessellation.render", "pass.tessellation",
           "Rendered patch draws: control points and per-patch attributes through the vertex descriptor, indexed "
           "control points, patch index buffers, factor step functions, culled patches, a fragment texture read and "
           "the triangles' order")
{
    [c require:@"family.apple3" value:[c.device supportsFamily:MTLGPUFamilyApple3]];
    // Buffer 1: a float4 per control point, its x and y on the 1/64 grid. Buffer 2: a uint per patch.
    float points[TRenderPoints * 4];
    for (int i = 0; i < TRenderPoints; i++) {
        points[4 * i] = (float)(i % 7 - 3) / 64.0f;
        points[4 * i + 1] = (float)(i % 5 - 2) / 64.0f;
        points[4 * i + 2] = 0;
        points[4 * i + 3] = (float)(i + 1) * 1.25f;
    }
    uint32_t tags[TRenderTags];
    for (uint32_t j = 0; j < TRenderTags; j++) tags[j] = (j + 1) * 0x9e3779b9u;
    // 24 slots of MTLQuadTessellationFactorsHalf (a triangle patch's factors are 8 bytes): factors from 1 to 6; quad
    // slot 10 has a zero edge (the patch is culled); quad slot 23 has inside factors of 16, whose inside the M4 emits
    // in tiles.
    uint16_t factors[TRenderSlots * 6];
    for (int i = 0; i < TRenderSlots * 6; i++) factors[i] = WMHalfFromFloat(1.0f + (float)((i * 7) % 11) * 0.5f);
    factors[10 * 6 + 1] = WMHalfFromFloat(0);
    const float large[6] = {4, 4, 4, 4, 16, 16};
    for (int k = 0; k < 6; k++) factors[23 * 6 + k] = WMHalfFromFloat(large[k]);
    const uint32_t patchIndices[] = {5, 2, 9, 0};
    uint16_t indices16[64];
    uint32_t indices32[64];
    for (uint32_t i = 0; i < 64; i++) {
        indices16[i] = (uint16_t)((i * 7 + 3) % TRenderPoints);
        indices32[i] = (i * 11 + 5) % TRenderPoints;
    }
    id<MTLBuffer> pointBuffer = [c bufferWithBytes:points length:sizeof points];
    id<MTLBuffer> tagBuffer = [c bufferWithBytes:tags length:sizeof tags];
    id<MTLBuffer> factorBuffer = [c bufferWithBytes:factors length:sizeof factors];
    id<MTLBuffer> patchIndexBuffer = [c bufferWithBytes:patchIndices length:sizeof patchIndices];
    id<MTLBuffer> index16Buffer = [c bufferWithBytes:indices16 length:sizeof indices16];
    id<MTLBuffer> index32Buffer = [c bufferWithBytes:indices32 length:sizeof indices32];
    // RGBA8Uint, 4 x 4, with the descriptor swizzle (alpha, red, green, blue) ("Texture swizzle": Apple2 and later,
    // Metal Feature Set Tables, 2026-10-23).
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Uint
                                                                                  width:4
                                                                                 height:4
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead;
    td.swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleAlpha, MTLTextureSwizzleRed, MTLTextureSwizzleGreen,
                                               MTLTextureSwizzleBlue);
    id<MTLTexture> texels = [c texture:td];
    uint8_t texelBytes[64];
    for (int i = 0; i < 64; i++) texelBytes[i] = (uint8_t)(i * 37 + 11);
    id<MTLBuffer> texelBuffer = [c bufferWithBytes:texelBytes length:sizeof texelBytes];
    [c blit:^(id<MTLBlitCommandEncoder> b) {
        [b copyFromBuffer:texelBuffer
                   sourceOffset:0
              sourceBytesPerRow:16
            sourceBytesPerImage:64
                     sourceSize:MTLSizeMake(4, 4, 1)
                      toTexture:texels
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(0, 0, 0)];
    }];

    const MTLTessellationFactorStepFunction perPatch = MTLTessellationFactorStepFunctionPerPatch;
    const MTLTessellationControlPointIndexType none = MTLTessellationControlPointIndexTypeNone;
    const TRender draws[] = {
        {"quad-integer", YES, MTLTessellationPartitionModeInteger, MTLWindingClockwise, perPatch, none, 0, 0, 0, 0, 3,
         1, 0, NO, 0, NO},
        {"quad-fractional-odd-instances", YES, MTLTessellationPartitionModeFractionalOdd, MTLWindingClockwise,
         perPatch, none, 0, 48, 0, 1, 2, 2, 1, NO, 0, NO},
        {"quad-culled-patch", YES, MTLTessellationPartitionModeInteger, MTLWindingClockwise, perPatch, none, 0, 0, 0,
         9, 3, 1, 0, NO, 0, NO},
        {"triangle-fractional-even", NO, MTLTessellationPartitionModeFractionalEven, MTLWindingClockwise, perPatch,
         none, 0, 0, 0, 0, 3, 1, 0, NO, 0, NO},
        {"triangle-pow2-counter-clockwise", NO, MTLTessellationPartitionModePow2, MTLWindingCounterClockwise,
         perPatch, none, 0, 0, 0, 1, 2, 1, 0, NO, 0, NO},
        {"quad-constant", YES, MTLTessellationPartitionModeFractionalEven, MTLWindingClockwise,
         MTLTessellationFactorStepFunctionConstant, none, 0, 24, 0, 0, 3, 2, 3, NO, 0, NO},
        {"quad-per-instance", YES, MTLTessellationPartitionModeInteger, MTLWindingClockwise,
         MTLTessellationFactorStepFunctionPerInstance, none, 0, 0, 24, 0, 2, 3, 2, NO, 0, NO},
        {"triangle-per-patch-and-instance", NO, MTLTessellationPartitionModeFractionalOdd, MTLWindingClockwise,
         MTLTessellationFactorStepFunctionPerPatchAndPerInstance, none, 0, 8, 40, 1, 2, 2, 1, NO, 0, NO},
        {"quad-patch-indices", YES, MTLTessellationPartitionModeInteger, MTLWindingClockwise, perPatch, none, 0, 0, 0,
         1, 3, 1, 0, YES, 0, NO},
        {"quad-indexed-uint16", YES, MTLTessellationPartitionModeFractionalEven, MTLWindingClockwise, perPatch,
         MTLTessellationControlPointIndexTypeUInt16, 0, 0, 0, 1, 2, 2, 0, NO, 6, NO},
        {"triangle-indexed-uint32-patch-indices", NO, MTLTessellationPartitionModeInteger, MTLWindingClockwise,
         perPatch, MTLTessellationControlPointIndexTypeUInt32, 0, 0, 0, 0, 4, 1, 0, YES, 8, NO},
        {"quad-scale", YES, MTLTessellationPartitionModeFractionalOdd, MTLWindingClockwise, perPatch, none, 1.5f, 0, 0,
         0, 2, 1, 0, NO, 0, NO},
        {"quad-tiled-inside", YES, MTLTessellationPartitionModeInteger, MTLWindingClockwise, perPatch, none, 0, 276, 0,
         0, 1, 1, 0, NO, 0, NO},
        {"quad-fragment-texture", YES, MTLTessellationPartitionModeInteger, MTLWindingClockwise, perPatch, none, 0, 0,
         0, 0, 2, 1, 0, NO, 0, YES},
    };
    for (size_t i = 0; i < sizeof draws / sizeof draws[0]; i++) {
        const TRender *d = &draws[i];
        [c step:@(d->name)];
        MTLRenderPipelineDescriptor *p = [MTLRenderPipelineDescriptor new];
        p.vertexFunction = [c function:d->quad ? @"tess_draw_quad" : @"tess_draw_triangle"];
        p.fragmentFunction = [c function:d->texture ? @"tess_draw_fragment_texture" : @"tess_draw_fragment"];
        p.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
        p.tessellationPartitionMode = d->mode;
        p.maxTessellationFactor = 16;
        p.tessellationFactorScaleEnabled = d->scale != 0;
        p.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
        p.tessellationFactorStepFunction = d->step;
        p.tessellationOutputWindingOrder = d->winding;
        p.tessellationControlPointIndexType = d->indexType;
        MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
        vd.attributes[0].format = MTLVertexFormatFloat4;
        vd.attributes[0].bufferIndex = 1;
        vd.layouts[1].stride = 16;
        vd.layouts[1].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
        vd.attributes[1].format = MTLVertexFormatUInt;
        vd.attributes[1].bufferIndex = 2;
        vd.layouts[2].stride = 4;
        vd.layouts[2].stepFunction = MTLVertexStepFunctionPerPatch;
        p.vertexDescriptor = vd;
        id<MTLRenderPipelineState> pipeline = [c renderPipeline:p];
        id<MTLTexture> target = RTarget(c, MTLPixelFormatR32Uint, TRenderSize, TRenderSize);
        NSUInteger controlPoints = d->quad ? 4 : 3;
        [c render:RPassColor(target, MTLClearColorMake(0, 0, 0, 0)) body:^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:pipeline];
            [e setVertexBuffer:pointBuffer offset:0 atIndex:1];
            [e setVertexBuffer:tagBuffer offset:0 atIndex:2];
            if (d->texture) [e setFragmentTexture:texels atIndex:0];
            [e setTessellationFactorBuffer:factorBuffer offset:d->offset instanceStride:d->stride];
            if (d->scale != 0) [e setTessellationFactorScale:d->scale];
            id<MTLBuffer> patches = d->patchIndices ? patchIndexBuffer : nil;
            if (d->indexType == none)
                [e drawPatches:controlPoints
                          patchStart:d->start
                          patchCount:d->count
                    patchIndexBuffer:patches
              patchIndexBufferOffset:0
                       instanceCount:d->instances
                        baseInstance:d->base];
            else
                [e drawIndexedPatches:controlPoints
                                patchStart:d->start
                                patchCount:d->count
                          patchIndexBuffer:patches
                    patchIndexBufferOffset:0
                   controlPointIndexBuffer:d->indexType == MTLTessellationControlPointIndexTypeUInt16 ? index16Buffer
                                                                                                     : index32Buffer
             controlPointIndexBufferOffset:d->indexOffset
                             instanceCount:d->instances
                              baseInstance:d->base];
        }];
        RRecordTexture(c, @(d->name), target, WMElemU32, 4, WMNone(TReason));
    }
}
