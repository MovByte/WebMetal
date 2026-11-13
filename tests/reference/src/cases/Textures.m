// Texture cases: every texture type with per-level and per-slice data,
// shader reads (including out-of-bounds reads and writes), texture views
// (format reinterpretation, level and slice ranges, type changes, swizzles),
// mipmap generation, texture properties and invalid descriptors.
//
// Texel values in R32Uint textures follow TexValue(level, slice, y, x), so
// every texel of every level, slice, face and depth image is distinct.
// Shader reads of R32Uint return (value, 0, 0, 1): components a format does
// not have read as 0, and alpha as 1 (MSL 4.1 section 6.13).

#import "WMRefCase.h"

static const WMElem kTexElem = WMElemU32;

static uint32_t TexValue(NSUInteger level, NSUInteger slice, NSUInteger y, NSUInteger x) {
    return 0xa0000000u | (uint32_t)(level << 24) | (uint32_t)(slice << 16) | (uint32_t)(y << 8) | (uint32_t)x;
}

// One read or write query: two uint4 (see shaders/textures.metal).
typedef struct {
    uint32_t c[8];
} TexQuery;

static TexQuery TexQ(uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t lod) {
    return (TexQuery){{x, y, z, w, lod, 0, 0, 0}};
}

typedef struct {
    uint32_t v[4];
} TexTexel;

static TexTexel TexDefault(void) {
    // Out-of-bounds read of a one-component format (MSL 4.1 section 6.13).
    return (TexTexel){{0, 0, 0, 1}};
}

static TexTexel TexR(uint32_t value) {
    return (TexTexel){{value, 0, 0, 1}};
}

// Runs `kernel` once per query (one thread per threadgroup, so the grid is
// uniform on every device) and returns the uint4 results.
static NSData *TexRunReads(WMCase *c, NSString *kernel, id<MTLTexture> texture, const TexQuery *queries,
                           NSUInteger count) {
    id<MTLComputePipelineState> pipeline = [c computePipeline:kernel];
    id<MTLBuffer> q = [c bufferWithBytes:queries length:count * sizeof(TexQuery)];
    id<MTLBuffer> out = [c buffer:count * 16];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setTexture:texture atIndex:0];
        [e setBuffer:q offset:0 atIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }];
    return [NSData dataWithBytes:out.contents length:out.length];
}

static void TexRunWrites(WMCase *c, NSString *kernel, id<MTLTexture> texture, const TexQuery *queries,
                         const TexTexel *values, NSUInteger count) {
    id<MTLComputePipelineState> pipeline = [c computePipeline:kernel];
    id<MTLBuffer> q = [c bufferWithBytes:queries length:count * sizeof(TexQuery)];
    id<MTLBuffer> v = [c bufferWithBytes:values length:count * sizeof(TexTexel)];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setTexture:texture atIndex:0];
        [e setBuffer:q offset:0 atIndex:0];
        [e setBuffer:v offset:0 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }];
}

static NSData *TexRunQuery(WMCase *c, NSString *kernel, id<MTLTexture> texture, NSUInteger words) {
    id<MTLComputePipelineState> pipeline = [c computePipeline:kernel];
    id<MTLBuffer> out = [c buffer:words * sizeof(uint32_t)];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setTexture:texture atIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }];
    return [NSData dataWithBytes:out.contents length:out.length];
}

static NSUInteger TexMip(NSUInteger size, NSUInteger level) {
    return MAX((NSUInteger)1, size >> level);
}

static BOOL TexIs3D(id<MTLTexture> t) { return t.textureType == MTLTextureType3D; }

// Number of blit slices: array elements times six for cube types.
static NSUInteger TexSlices(id<MTLTexture> t) {
    BOOL cube = t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray;
    return t.arrayLength * (cube ? 6 : 1);
}

// Fills every level and slice of an R32Uint texture through a blit. For 3D
// textures the z coordinate takes the place of the slice in TexValue.
static void TexUploadAll(WMCase *c, id<MTLTexture> t) {
    for (NSUInteger level = 0; level < t.mipmapLevelCount; level++) {
        NSUInteger w = TexMip(t.width, level), h = TexMip(t.height, level);
        NSUInteger d = TexIs3D(t) ? TexMip(t.depth, level) : 1;
        for (NSUInteger slice = 0; slice < TexSlices(t); slice++) {
            NSMutableData *bytes = [NSMutableData dataWithLength:w * h * d * 4];
            uint32_t *p = bytes.mutableBytes;
            for (NSUInteger z = 0; z < d; z++)
                for (NSUInteger y = 0; y < h; y++)
                    for (NSUInteger x = 0; x < w; x++)
                        p[(z * h + y) * w + x] = TexValue(level, TexIs3D(t) ? z : slice, y, x);
            id<MTLBuffer> source = [c bufferWithBytes:bytes.bytes length:bytes.length];
            [c blit:^(id<MTLBlitCommandEncoder> e) {
                [e copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:w * 4
                    sourceBytesPerImage:TexIs3D(t) ? w * h * 4 : 0 sourceSize:MTLSizeMake(w, h, d)
                              toTexture:t destinationSlice:slice destinationLevel:level
                      destinationOrigin:MTLOriginMake(0, 0, 0)];
            }];
        }
    }
}

// Records the blit readback of every level and slice and whether it equals
// the uploaded values.
static void TexRecordReadback(WMCase *c, id<MTLTexture> t) {
    BOOL equal = YES;
    for (NSUInteger level = 0; level < t.mipmapLevelCount; level++) {
        NSUInteger w = TexMip(t.width, level), h = TexMip(t.height, level);
        NSUInteger d = TexIs3D(t) ? TexMip(t.depth, level) : 1;
        for (NSUInteger slice = 0; slice < TexSlices(t); slice++) {
            NSData *data = [c readTexture:t level:level slice:slice bytesPerRow:w * 4];
            const uint32_t *p = data.bytes;
            for (NSUInteger z = 0; z < d; z++)
                for (NSUInteger y = 0; y < h; y++)
                    for (NSUInteger x = 0; x < w; x++)
                        equal &= p[(z * h + y) * w + x] == TexValue(level, TexIs3D(t) ? z : slice, y, x);
            NSDictionary *layout = @{@"format": WMPixelFormatName(t.pixelFormat), @"width": @(w), @"height": @(h),
                                     @"depth": @(d), @"bytes_per_row": @(w * 4), @"level": @(level),
                                     @"slice": @(slice)};
            [c data:[NSString stringWithFormat:@"readback.level%lu.slice%lu", (unsigned long)level,
                     (unsigned long)slice]
                elem:kTexElem bytes:data.bytes length:data.length layout:layout rule:WMExact()];
        }
    }
    [c value:@"readback-equals-upload" boolean:equal rule:WMExact()];
}

// Records the query list, the shader reads, and whether each read equals the
// value the MSL 4.1 section 6.13 rules give (`expected`).
static void TexRecordReads(WMCase *c, NSString *name, const TexQuery *queries, NSUInteger count, NSData *reads,
                           const TexTexel *expected) {
    [c data:[name stringByAppendingString:@".queries"] elem:WMElemU32 bytes:queries
     length:count * sizeof(TexQuery) rule:WMExact()];
    [c data:name elem:WMElemU32 bytes:reads.bytes length:reads.length rule:WMExact()];
    [c value:[name stringByAppendingString:@".match-specification"]
     boolean:memcmp(reads.bytes, expected, count * sizeof(TexTexel)) == 0 rule:WMExact()];
}

static MTLTextureDescriptor *TexDescriptor(MTLTextureType type, NSUInteger w, NSUInteger h, NSUInteger d,
                                           NSUInteger levels, NSUInteger arrayLength) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = type;
    desc.pixelFormat = MTLPixelFormatR32Uint;
    desc.width = w;
    desc.height = h;
    desc.depth = d;
    desc.mipmapLevelCount = levels;
    desc.arrayLength = arrayLength;
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    return desc;
}

#pragma mark - Texture types

WMREF_CASE(texType1D, "texture.type-1d", "texture.types",
           "1D texture: blit upload and readback, shader reads including out of bounds, size queries")
{
    // 1D textures have no mipmaps (MSL 4.1 section 6.13.1).
    id<MTLTexture> t = [c texture:TexDescriptor(MTLTextureType1D, 16, 1, 1, 1, 1)];
    TexUploadAll(c, t);
    TexRecordReadback(c, t);
    TexQuery q[19];
    TexTexel expected[19];
    NSUInteger n = 0;
    for (uint32_t x = 0; x < 16; x++, n++) {
        q[n] = TexQ(x, 0, 0, 0, 0);
        expected[n] = TexR(TexValue(0, 0, 0, x));
    }
    const uint32_t outside[] = {16, 17, 1000};
    for (int i = 0; i < 3; i++, n++) {
        q[n] = TexQ(outside[i], 0, 0, 0, 0);
        expected[n] = TexDefault();
    }
    TexRecordReads(c, @"reads", q, n, TexRunReads(c, @"tex_read_1d", t, q, n), expected);
    NSData *sizes = TexRunQuery(c, @"tex_query_1d", t, 2);
    const uint32_t *words = sizes.bytes;
    [c value:@"get_width" u64:words[0] rule:WMExact()];
    // MSL 4.1 section 6.13.1 states that get_num_mip_levels() of a 1D texture
    // always returns 0; the texture has one level, which is the other value
    // a device can return.
    [c value:@"get_num_mip_levels" u64:words[1]
        rule:WMOneOf(@[@0, @1], @"MSL 4.1 section 6.13.1 states 0 for 1D textures; the texture has 1 level")];
}

WMREF_CASE(texType1DArray, "texture.type-1d-array", "texture.types",
           "1D texture array: per-slice data, reads with out-of-bounds coordinate and slice")
{
    id<MTLTexture> t = [c texture:TexDescriptor(MTLTextureType1DArray, 8, 1, 1, 1, 3)];
    TexUploadAll(c, t);
    TexRecordReadback(c, t);
    TexQuery q[32];
    TexTexel expected[32];
    NSUInteger n = 0;
    for (uint32_t s = 0; s < 3; s++)
        for (uint32_t x = 0; x < 8; x++, n++) {
            q[n] = TexQ(x, s, 0, 0, 0);
            expected[n] = TexR(TexValue(0, s, 0, x));
        }
    const uint32_t outside[][2] = {{8, 0}, {0, 3}, {8, 3}, {3, 100}};
    for (int i = 0; i < 4; i++, n++) {
        q[n] = TexQ(outside[i][0], outside[i][1], 0, 0, 0);
        expected[n] = TexDefault();
    }
    TexRecordReads(c, @"reads", q, n, TexRunReads(c, @"tex_read_1d_array", t, q, n), expected);
    NSData *sizes = TexRunQuery(c, @"tex_query_1d_array", t, 2);
    [c data:@"get_width.get_array_size" elem:WMElemU32 bytes:sizes.bytes length:sizes.length rule:WMExact()];
}

WMREF_CASE(texType2D, "texture.type-2d", "texture.types",
           "Mipmapped 2D texture: every level uploaded, read back and read with explicit LOD")
{
    id<MTLTexture> t = [c texture:TexDescriptor(MTLTextureType2D, 8, 4, 1, 4, 1)];
    TexUploadAll(c, t);
    TexRecordReadback(c, t);
    NSMutableData *qd = [NSMutableData data], *ed = [NSMutableData data];
    for (uint32_t l = 0; l < 4; l++)
        for (uint32_t y = 0; y < TexMip(4, l); y++)
            for (uint32_t x = 0; x < TexMip(8, l); x++) {
                TexQuery q = TexQ(x, y, 0, 0, l);
                TexTexel e = TexR(TexValue(l, 0, y, x));
                [qd appendBytes:&q length:sizeof q];
                [ed appendBytes:&e length:sizeof e];
            }
    // Coordinates past each level's size, and a level past the last one.
    const uint32_t outside[][3] = {{8, 0, 0}, {0, 4, 0}, {4, 0, 1}, {0, 2, 1}, {1, 0, 3}, {0, 0, 4}, {0, 0, 9}};
    for (int i = 0; i < 7; i++) {
        TexQuery q = TexQ(outside[i][0], outside[i][1], 0, 0, outside[i][2]);
        TexTexel e = TexDefault();
        [qd appendBytes:&q length:sizeof q];
        [ed appendBytes:&e length:sizeof e];
    }
    NSUInteger n = qd.length / sizeof(TexQuery);
    TexRecordReads(c, @"reads", qd.bytes, n, TexRunReads(c, @"tex_read_2d", t, qd.bytes, n), ed.bytes);
    NSData *sizes = TexRunQuery(c, @"tex_query_2d", t, 1 + 2 * 4);
    [c data:@"get_num_mip_levels.sizes" elem:WMElemU32 bytes:sizes.bytes length:sizes.length rule:WMExact()];
}

WMREF_CASE(texType2DArray, "texture.type-2d-array", "texture.types",
           "Mipmapped 2D texture array: per-slice, per-level data and out-of-bounds slice and level")
{
    id<MTLTexture> t = [c texture:TexDescriptor(MTLTextureType2DArray, 4, 4, 1, 2, 3)];
    TexUploadAll(c, t);
    TexRecordReadback(c, t);
    NSMutableData *qd = [NSMutableData data], *ed = [NSMutableData data];
    for (uint32_t s = 0; s < 3; s++)
        for (uint32_t l = 0; l < 2; l++)
            for (uint32_t y = 0; y < TexMip(4, l); y++)
                for (uint32_t x = 0; x < TexMip(4, l); x++) {
                    TexQuery q = TexQ(x, y, s, 0, l);
                    TexTexel e = TexR(TexValue(l, s, y, x));
                    [qd appendBytes:&q length:sizeof q];
                    [ed appendBytes:&e length:sizeof e];
                }
    const uint32_t outside[][4] = {{4, 0, 0, 0}, {0, 4, 1, 0}, {0, 0, 3, 0}, {0, 0, 0, 2}, {2, 0, 2, 1}};
    for (int i = 0; i < 5; i++) {
        TexQuery q = TexQ(outside[i][0], outside[i][1], outside[i][2], 0, outside[i][3]);
        TexTexel e = TexDefault();
        [qd appendBytes:&q length:sizeof q];
        [ed appendBytes:&e length:sizeof e];
    }
    NSUInteger n = qd.length / sizeof(TexQuery);
    TexRecordReads(c, @"reads", qd.bytes, n, TexRunReads(c, @"tex_read_2d_array", t, qd.bytes, n), ed.bytes);
    NSData *sizes = TexRunQuery(c, @"tex_query_2d_array", t, 2 + 2 * 2);
    [c data:@"get_num_mip_levels.get_array_size.sizes" elem:WMElemU32 bytes:sizes.bytes length:sizes.length
       rule:WMExact()];
}

WMREF_CASE(texType3D, "texture.type-3d", "texture.types",
           "Mipmapped 3D texture: every depth image of every level, out-of-bounds z and level")
{
    id<MTLTexture> t = [c texture:TexDescriptor(MTLTextureType3D, 4, 4, 4, 3, 1)];
    TexUploadAll(c, t);
    TexRecordReadback(c, t);
    NSMutableData *qd = [NSMutableData data], *ed = [NSMutableData data];
    for (uint32_t l = 0; l < 3; l++)
        for (uint32_t z = 0; z < TexMip(4, l); z++)
            for (uint32_t y = 0; y < TexMip(4, l); y++)
                for (uint32_t x = 0; x < TexMip(4, l); x++) {
                    TexQuery q = TexQ(x, y, z, 0, l);
                    TexTexel e = TexR(TexValue(l, z, y, x));
                    [qd appendBytes:&q length:sizeof q];
                    [ed appendBytes:&e length:sizeof e];
                }
    const uint32_t outside[][4] = {{4, 0, 0, 0}, {0, 0, 4, 0}, {0, 0, 2, 1}, {0, 0, 1, 2}, {0, 0, 0, 3}};
    for (int i = 0; i < 5; i++) {
        TexQuery q = TexQ(outside[i][0], outside[i][1], outside[i][2], 0, outside[i][3]);
        TexTexel e = TexDefault();
        [qd appendBytes:&q length:sizeof q];
        [ed appendBytes:&e length:sizeof e];
    }
    NSUInteger n = qd.length / sizeof(TexQuery);
    TexRecordReads(c, @"reads", qd.bytes, n, TexRunReads(c, @"tex_read_3d", t, qd.bytes, n), ed.bytes);
    NSData *sizes = TexRunQuery(c, @"tex_query_3d", t, 1 + 3 * 3);
    [c data:@"get_num_mip_levels.sizes" elem:WMElemU32 bytes:sizes.bytes length:sizes.length rule:WMExact()];
}

WMREF_CASE(texTypeCube, "texture.type-cube", "texture.types",
           "Mipmapped cube texture: six faces per level, reads by face (table 6.23) and out-of-bounds face")
{
    MTLTextureDescriptor *desc = TexDescriptor(MTLTextureTypeCube, 4, 4, 1, 2, 1);
    id<MTLTexture> t = [c texture:desc];
    TexUploadAll(c, t);
    TexRecordReadback(c, t);
    NSMutableData *qd = [NSMutableData data], *ed = [NSMutableData data];
    for (uint32_t f = 0; f < 6; f++)
        for (uint32_t l = 0; l < 2; l++)
            for (uint32_t y = 0; y < TexMip(4, l); y++)
                for (uint32_t x = 0; x < TexMip(4, l); x++) {
                    TexQuery q = TexQ(x, y, f, 0, l);
                    TexTexel e = TexR(TexValue(l, f, y, x));
                    [qd appendBytes:&q length:sizeof q];
                    [ed appendBytes:&e length:sizeof e];
                }
    const uint32_t outside[][4] = {{4, 0, 0, 0}, {0, 0, 6, 0}, {0, 0, 7, 1}, {2, 0, 5, 1}, {0, 0, 0, 2}};
    for (int i = 0; i < 5; i++) {
        TexQuery q = TexQ(outside[i][0], outside[i][1], outside[i][2], 0, outside[i][3]);
        TexTexel e = TexDefault();
        [qd appendBytes:&q length:sizeof q];
        [ed appendBytes:&e length:sizeof e];
    }
    NSUInteger n = qd.length / sizeof(TexQuery);
    TexRecordReads(c, @"reads", qd.bytes, n, TexRunReads(c, @"tex_read_cube", t, qd.bytes, n), ed.bytes);
    NSData *sizes = TexRunQuery(c, @"tex_query_cube", t, 1 + 2 * 2);
    [c data:@"get_num_mip_levels.sizes" elem:WMElemU32 bytes:sizes.bytes length:sizes.length rule:WMExact()];
}

WMREF_CASE(texCubeSampling, "texture.cube-sample-axes", "texture.types",
           "Cube sampling along the six axes selects the face of the major axis (table 6.23)")
{
    // 3x3 faces: a direction along an axis lands on the centre texel of the
    // face whatever the face's (s, t) orientation, so the result depends
    // only on the face selection. Texel value = 100 * face + 3 * y + x.
    MTLTextureDescriptor *desc = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                       size:3 mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [c texture:desc];
    for (NSUInteger face = 0; face < 6; face++) {
        float texels[9];
        for (int i = 0; i < 9; i++) texels[i] = (float)(100 * face + i);
        id<MTLBuffer> source = [c bufferWithBytes:texels length:sizeof texels];
        [c blit:^(id<MTLBlitCommandEncoder> e) {
            [e copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:12 sourceBytesPerImage:0
                   sourceSize:MTLSizeMake(3, 3, 1) toTexture:t destinationSlice:face destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        }];
    }
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    id<MTLSamplerState> sampler = [c sampler:sd];
    const float directions[][4] = {
        {1, 0, 0, 0}, {-1, 0, 0, 0}, {0, 1, 0, 0}, {0, -1, 0, 0}, {0, 0, 1, 0}, {0, 0, -1, 0},
        {5, 0.5f, -0.25f, 0}, {-0.25f, -3, 1, 0},
        // Off-axis directions whose result also depends on each face's (s, t)
        // orientation.
        {1, 0.9f, 0.1f, 0}, {1, 0.1f, 0.9f, 0}, {0.1f, 1, 0.9f, 0}, {0.9f, 0.1f, -1, 0},
    };
    NSUInteger n = sizeof directions / sizeof directions[0];
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"tex_sample_cube"];
    id<MTLBuffer> in = [c bufferWithBytes:directions length:sizeof directions];
    id<MTLBuffer> out = [c buffer:n * 16];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setTexture:t atIndex:0];
        [e setSamplerState:sampler atIndex:0];
        [e setBuffer:in offset:0 atIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }];
    [c data:@"directions" elem:WMElemF32 bytes:directions length:sizeof directions rule:WMExact()];
    // The first eight directions select the centre texel of the face of
    // their major axis: exact.
    [c data:@"major-axis-samples" elem:WMElemF32 bytes:out.contents length:8 * 16 rule:WMExact()];
    [c data:@"off-axis-samples" elem:WMElemF32 bytes:(const uint8_t *)out.contents + 8 * 16 length:(n - 8) * 16
       rule:WMNone(@"MSL 4.1 lists the cube faces (table 6.23) but not the (s, t) orientation of each face")];
}

WMREF_CASE(texTypeCubeArray, "texture.type-cube-array", "texture.types",
           "Cube texture array: faces of every element, reads by face and element, out of bounds")
{
    // Metal feature set tables: cube map texture arrays from Apple4.
    [c require:@"family.apple4" value:[c.device supportsFamily:MTLGPUFamilyApple4]];
    id<MTLTexture> t = [c texture:TexDescriptor(MTLTextureTypeCubeArray, 2, 2, 1, 2, 2)];
    TexUploadAll(c, t);
    TexRecordReadback(c, t);
    NSMutableData *qd = [NSMutableData data], *ed = [NSMutableData data];
    for (uint32_t a = 0; a < 2; a++)
        for (uint32_t f = 0; f < 6; f++)
            for (uint32_t l = 0; l < 2; l++)
                for (uint32_t y = 0; y < TexMip(2, l); y++)
                    for (uint32_t x = 0; x < TexMip(2, l); x++) {
                        TexQuery q = TexQ(x, y, f, a, l);
                        // Blit slice of element a, face f is 6a + f.
                        TexTexel e = TexR(TexValue(l, 6 * a + f, y, x));
                        [qd appendBytes:&q length:sizeof q];
                        [ed appendBytes:&e length:sizeof e];
                    }
    const uint32_t outside[][5] = {{2, 0, 0, 0, 0}, {0, 0, 6, 0, 0}, {0, 0, 0, 2, 0}, {0, 0, 0, 0, 2}, {1, 0, 0, 1, 1}};
    for (int i = 0; i < 5; i++) {
        TexQuery q = TexQ(outside[i][0], outside[i][1], outside[i][2], outside[i][3], outside[i][4]);
        TexTexel e = TexDefault();
        [qd appendBytes:&q length:sizeof q];
        [ed appendBytes:&e length:sizeof e];
    }
    NSUInteger n = qd.length / sizeof(TexQuery);
    TexRecordReads(c, @"reads", qd.bytes, n, TexRunReads(c, @"tex_read_cube_array", t, qd.bytes, n), ed.bytes);
    NSData *sizes = TexRunQuery(c, @"tex_query_cube_array", t, 2 + 2 * 2);
    [c data:@"get_num_mip_levels.get_array_size.sizes" elem:WMElemU32 bytes:sizes.bytes length:sizes.length
       rule:WMExact()];
}

// Value the multisample fragment functions write, and the clear value.
static const uint32_t kTexClear = 0x01020304;

static uint32_t TexSampleValue(uint32_t layer, uint32_t sample, uint32_t y, uint32_t x) {
    return 0xab000000u | (layer << 20) | (sample << 16) | (y << 8) | x;
}

// Renders the rectangle over pixel columns [0, 2) into a 4x4, 4-sample
// R32Uint target cleared to kTexClear, with per-sample shading.
static void TexRenderMultisample(WMCase *c, id<MTLTexture> t, BOOL layered) {
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [c function:layered ? @"tex_layered_vertex" : @"tex_rect_vertex"];
    pd.fragmentFunction = [c function:layered ? @"tex_ms_layered_fragment" : @"tex_ms_fragment"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
    pd.rasterSampleCount = 4;
    if (layered) pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:pd];
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = t;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(kTexClear, 0, 0, 0);
    if (layered) pass.renderTargetArrayLength = t.arrayLength;
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6 instanceCount:layered ? t.arrayLength : 1];
    }];
}

WMREF_CASE(texType2DMultisample, "texture.type-2d-multisample", "texture.types",
           "4-sample 2D texture filled by per-sample shading, read per sample, out-of-bounds sample")
{
    [c require:@"sample-count-4" value:[c.device supportsTextureSampleCount:4]];
    MTLTextureDescriptor *desc = TexDescriptor(MTLTextureType2DMultisample, 4, 4, 1, 1, 1);
    desc.sampleCount = 4;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    id<MTLTexture> t = [c texture:desc];
    TexRenderMultisample(c, t, NO);
    NSMutableData *qd = [NSMutableData data], *ed = [NSMutableData data];
    for (uint32_t y = 0; y < 4; y++)
        for (uint32_t x = 0; x < 4; x++)
            for (uint32_t s = 0; s < 4; s++) {
                TexQuery q = TexQ(x, y, s, 0, 0);
                // Pixels in columns 0 and 1 are covered for every sample.
                TexTexel e = TexR(x < 2 ? TexSampleValue(0, s, y, x) : kTexClear);
                [qd appendBytes:&q length:sizeof q];
                [ed appendBytes:&e length:sizeof e];
            }
    const uint32_t outside[][3] = {{0, 0, 4}, {4, 0, 0}, {0, 4, 1}, {1, 1, 31}};
    for (int i = 0; i < 4; i++) {
        TexQuery q = TexQ(outside[i][0], outside[i][1], outside[i][2], 0, 0);
        TexTexel e = TexDefault();
        [qd appendBytes:&q length:sizeof q];
        [ed appendBytes:&e length:sizeof e];
    }
    NSUInteger n = qd.length / sizeof(TexQuery);
    TexRecordReads(c, @"reads", qd.bytes, n, TexRunReads(c, @"tex_read_2d_ms", t, qd.bytes, n), ed.bytes);
    NSData *sizes = TexRunQuery(c, @"tex_query_2d_ms", t, 3);
    [c data:@"get_width.get_height.get_num_samples" elem:WMElemU32 bytes:sizes.bytes length:sizes.length
       rule:WMExact()];
}

WMREF_CASE(texType2DMultisampleArray, "texture.type-2d-multisample-array", "texture.types",
           "4-sample 2D texture array filled by layered rendering, read per slice and sample")
{
    // Filled through layered rendering into a multisample texture, which the
    // Metal feature set tables list from Apple7.
    [c require:@"family.apple7" value:[c.device supportsFamily:MTLGPUFamilyApple7]];
    [c require:@"sample-count-4" value:[c.device supportsTextureSampleCount:4]];
    MTLTextureDescriptor *desc = TexDescriptor(MTLTextureType2DMultisampleArray, 4, 4, 1, 1, 2);
    desc.sampleCount = 4;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    id<MTLTexture> t = [c texture:desc];
    TexRenderMultisample(c, t, YES);
    NSMutableData *qd = [NSMutableData data], *ed = [NSMutableData data];
    for (uint32_t a = 0; a < 2; a++)
        for (uint32_t y = 0; y < 4; y++)
            for (uint32_t x = 0; x < 4; x++)
                for (uint32_t s = 0; s < 4; s++) {
                    TexQuery q = TexQ(x, y, a, s, 0);
                    TexTexel e = TexR(x < 2 ? TexSampleValue(a, s, y, x) : kTexClear);
                    [qd appendBytes:&q length:sizeof q];
                    [ed appendBytes:&e length:sizeof e];
                }
    const uint32_t outside[][4] = {{0, 0, 2, 0}, {0, 0, 0, 4}, {4, 0, 1, 0}};
    for (int i = 0; i < 3; i++) {
        TexQuery q = TexQ(outside[i][0], outside[i][1], outside[i][2], outside[i][3], 0);
        TexTexel e = TexDefault();
        [qd appendBytes:&q length:sizeof q];
        [ed appendBytes:&e length:sizeof e];
    }
    NSUInteger n = qd.length / sizeof(TexQuery);
    TexRecordReads(c, @"reads", qd.bytes, n, TexRunReads(c, @"tex_read_2d_ms_array", t, qd.bytes, n), ed.bytes);
    NSData *sizes = TexRunQuery(c, @"tex_query_2d_ms_array", t, 4);
    [c data:@"get_width.get_height.get_num_samples.get_array_size" elem:WMElemU32 bytes:sizes.bytes
     length:sizes.length rule:WMExact()];
}

WMREF_CASE(texTypeBufferBacked, "texture.type-texture-buffer", "texture.types",
           "Texture buffer over an MTLBuffer: reads, writes visible in the buffer, out-of-bounds access")
{
    NSUInteger alignment = [c.device minimumTextureBufferAlignmentForPixelFormat:MTLPixelFormatR32Uint];
    [c value:@"minimum-texture-buffer-alignment.R32Uint" u64:alignment rule:WMDevice()];
    if (alignment > 1024) [c fail:@"texture buffer alignment %lu exceeds the case's buffer layout", (unsigned long)alignment];
    const NSUInteger width = 32, offset = 1024;
    id<MTLBuffer> buffer = [c buffer:offset + width * 4 + 256];
    uint32_t *words = buffer.contents;
    for (NSUInteger i = 0; i < buffer.length / 4; i++) words[i] = 0x5a000000u | (uint32_t)i;
    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        textureBufferDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:width
                               resourceOptions:buffer.resourceOptions
                                         usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
    id<MTLTexture> t = [buffer newTextureWithDescriptor:desc offset:offset bytesPerRow:width * 4];
    if (!t) [c fail:@"newTextureWithDescriptor:offset:bytesPerRow: returned nil"];
    [c value:@"texture-type" u64:t.textureType rule:WMExact()];
    [c value:@"width" u64:t.width rule:WMExact()];
    [c value:@"buffer-is-source" boolean:t.buffer == buffer rule:WMExact()];
    [c value:@"buffer-offset" u64:t.bufferOffset rule:WMExact()];
    [c value:@"buffer-bytes-per-row" u64:t.bufferBytesPerRow rule:WMExact()];

    TexQuery q[36];
    TexTexel expected[36];
    NSUInteger n = 0;
    for (uint32_t x = 0; x < width; x++, n++) {
        q[n] = TexQ(x, 0, 0, 0, 0);
        expected[n] = TexR(words[offset / 4 + x]);
    }
    const uint32_t outside[] = {32, 33, 64, 100000};
    for (int i = 0; i < 4; i++, n++) {
        q[n] = TexQ(outside[i], 0, 0, 0, 0);
        expected[n] = TexDefault();
    }
    TexRecordReads(c, @"reads", q, n, TexRunReads(c, @"tex_read_buffer", t, q, n), expected);
    NSData *sizes = TexRunQuery(c, @"tex_query_buffer", t, 1);
    [c data:@"get_width" elem:WMElemU32 bytes:sizes.bytes length:sizes.length rule:WMExact()];

    // Writes at elements 0, 5 and 31 land in the buffer at the texture's
    // offset; writes at 32 and 40 are out of bounds and ignored.
    TexQuery wq[5] = {TexQ(0, 0, 0, 0, 0), TexQ(5, 0, 0, 0, 0), TexQ(31, 0, 0, 0, 0), TexQ(32, 0, 0, 0, 0),
                      TexQ(40, 0, 0, 0, 0)};
    TexTexel wv[5] = {{{0x11111111, 0, 0, 0}}, {{0x22222222, 0, 0, 0}}, {{0x33333333, 0, 0, 0}},
                      {{0x44444444, 0, 0, 0}}, {{0x55555555, 0, 0, 0}}};
    TexRunWrites(c, @"tex_write_buffer", t, wq, wv, 5);
    [c data:@"buffer-after-writes" elem:WMElemU32 bytes:buffer.contents length:buffer.length rule:WMExact()];
}

WMREF_CASE(texTypeBufferDevice, "texture.type-texture-buffer-device", "texture.types",
           "Texture buffer created by the device: shader writes then shader reads")
{
    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        textureBufferDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:24
                               resourceOptions:MTLResourceStorageModePrivate
                                         usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
    id<MTLTexture> t = [c texture:desc];
    [c value:@"texture-type" u64:t.textureType rule:WMExact()];
    TexQuery wq[24];
    TexTexel wv[24];
    for (uint32_t x = 0; x < 24; x++) {
        wq[x] = TexQ(x, 0, 0, 0, 0);
        wv[x] = (TexTexel){{TexValue(0, 0, 0, x), 7, 7, 7}};
    }
    TexRunWrites(c, @"tex_write_buffer", t, wq, wv, 24);
    TexQuery q[26];
    TexTexel expected[26];
    for (uint32_t x = 0; x < 24; x++) {
        q[x] = wq[x];
        expected[x] = TexR(TexValue(0, 0, 0, x));
    }
    q[24] = TexQ(24, 0, 0, 0, 0);
    q[25] = TexQ(1 << 20, 0, 0, 0, 0);
    expected[24] = expected[25] = TexDefault();
    TexRecordReads(c, @"reads", q, 26, TexRunReads(c, @"tex_read_buffer", t, q, 26), expected);
}

#pragma mark - Out-of-bounds writes

WMREF_CASE(texWriteOutOfBounds, "texture.write-out-of-bounds", "texture.access",
           "Shader writes at in-bounds and out-of-bounds coordinates, levels, slices, faces and depths")
{
    // Every texture starts with TexValue contents; in-bounds writes replace
    // one texel, out-of-bounds writes are ignored (MSL 4.1 section 6.13).
    struct {
        NSString *name, *kernel;
        MTLTextureDescriptor *desc;
    } targets[] = {
        {@"1d", @"tex_write_1d", TexDescriptor(MTLTextureType1D, 8, 1, 1, 1, 1)},
        {@"2d", @"tex_write_2d", TexDescriptor(MTLTextureType2D, 4, 4, 1, 2, 1)},
        {@"2d-array", @"tex_write_2d_array", TexDescriptor(MTLTextureType2DArray, 2, 2, 1, 1, 2)},
        {@"3d", @"tex_write_3d", TexDescriptor(MTLTextureType3D, 2, 2, 2, 1, 1)},
        {@"cube", @"tex_write_cube", TexDescriptor(MTLTextureTypeCube, 2, 2, 1, 1, 1)},
    };
    const TexQuery queries[][6] = {
        // 1D: x = 3 in bounds; x = 8, 9 out of bounds.
        {TexQ(3, 0, 0, 0, 0), TexQ(8, 0, 0, 0, 0), TexQ(9, 0, 0, 0, 0)},
        // 2D: (1, 1) level 0 and (1, 0) level 1 in bounds; x = 4, y = 4,
        // (2, 0) at level 1 and level 2 out of bounds.
        {TexQ(1, 1, 0, 0, 0), TexQ(1, 0, 0, 0, 1), TexQ(4, 0, 0, 0, 0), TexQ(0, 4, 0, 0, 0), TexQ(2, 0, 0, 0, 1),
         TexQ(0, 0, 0, 0, 2)},
        // 2D array: (1, 0) slice 1 in bounds; slice 2 out of bounds.
        {TexQ(1, 0, 1, 0, 0), TexQ(0, 0, 2, 0, 0), TexQ(2, 1, 0, 0, 0)},
        // 3D: (0, 1, 1) in bounds; z = 2 out of bounds.
        {TexQ(0, 1, 1, 0, 0), TexQ(0, 0, 2, 0, 0), TexQ(1, 2, 0, 0, 0)},
        // Cube: (1, 1) face 4 in bounds; face 6 out of bounds.
        {TexQ(1, 1, 4, 0, 0), TexQ(0, 0, 6, 0, 0), TexQ(0, 2, 5, 0, 0)},
    };
    const NSUInteger counts[] = {3, 6, 3, 3, 3};
    for (int i = 0; i < 5; i++) {
        id<MTLTexture> t = [c texture:targets[i].desc];
        TexUploadAll(c, t);
        TexTexel values[6];
        for (NSUInteger j = 0; j < counts[i]; j++) values[j] = (TexTexel){{0xc0de0000u | (uint32_t)(i << 8) | (uint32_t)j, 0, 0, 0}};
        TexRunWrites(c, targets[i].kernel, t, queries[i], values, counts[i]);
        NSMutableData *all = [NSMutableData data];
        for (NSUInteger level = 0; level < t.mipmapLevelCount; level++)
            for (NSUInteger slice = 0; slice < TexSlices(t); slice++)
                [all appendData:[c readTexture:t level:level slice:slice bytesPerRow:TexMip(t.width, level) * 4]];
        [c data:[targets[i].name stringByAppendingString:@".contents"] elem:WMElemU32 bytes:all.bytes
         length:all.length rule:WMExact()];
    }
}

// Every level and slice of an R32Uint texture as TexUploadAll fills them, in
// the order TexReadAll reads them back: levels, then slices (a 3D level is
// one slice of all its depth images).
static NSMutableData *TexInitialContents(id<MTLTexture> t) {
    NSMutableData *all = [NSMutableData data];
    for (NSUInteger level = 0; level < t.mipmapLevelCount; level++) {
        NSUInteger w = TexMip(t.width, level), h = TexMip(t.height, level);
        NSUInteger d = TexIs3D(t) ? TexMip(t.depth, level) : 1;
        for (NSUInteger slice = 0; slice < TexSlices(t); slice++)
            for (NSUInteger z = 0; z < d; z++)
                for (NSUInteger y = 0; y < h; y++)
                    for (NSUInteger x = 0; x < w; x++) {
                        uint32_t v = TexValue(level, TexIs3D(t) ? z : slice, y, x);
                        [all appendBytes:&v length:sizeof v];
                    }
    }
    return all;
}

static NSData *TexReadAll(WMCase *c, id<MTLTexture> t) {
    NSMutableData *all = [NSMutableData data];
    for (NSUInteger level = 0; level < t.mipmapLevelCount; level++)
        for (NSUInteger slice = 0; slice < TexSlices(t); slice++)
            [all appendData:[c readTexture:t level:level slice:slice bytesPerRow:TexMip(t.width, level) * 4]];
    return all;
}

// Applies to `contents` (TexInitialContents of `root`) a write of `value`
// through `t`, which is `root` or a view of it whose level 0 and slice 0 are
// the root's `baseLevel` and `baseSlice`, when the write is in bounds of `t`
// itself: its level, coordinates, array slice or face, and depth (MSL 4.1
// section 6.13). An out-of-bounds write is ignored.
static void TexApplyWrite(NSMutableData *contents, id<MTLTexture> root, id<MTLTexture> t, NSUInteger baseLevel,
                          NSUInteger baseSlice, TexQuery q, uint32_t value) {
    uint32_t x = q.c[0], y = q.c[1], z = q.c[2], lod = q.c[4];
    if (lod >= t.mipmapLevelCount) return;
    NSUInteger w = TexMip(t.width, lod), h = TexMip(t.height, lod);
    NSUInteger slice = 0, depth = 0;
    switch (t.textureType) {
    case MTLTextureType2D:
        break;
    case MTLTextureType2DArray:
        if (z >= t.arrayLength) return;
        slice = z;
        break;
    case MTLTextureTypeCube:
        if (z >= 6) return;
        slice = z;
        break;
    case MTLTextureType3D:
        if (z >= TexMip(t.depth, lod)) return;
        depth = z;
        break;
    default:
        return;
    }
    if (x >= w || y >= h) return;
    NSUInteger level = baseLevel + lod, offset = 0;
    for (NSUInteger l = 0; l < level; l++)
        offset += TexMip(root.width, l) * TexMip(root.height, l) * (TexIs3D(root) ? TexMip(root.depth, l) : 1) *
                  TexSlices(root);
    NSUInteger rw = TexMip(root.width, level), rh = TexMip(root.height, level);
    NSUInteger rd = TexIs3D(root) ? TexMip(root.depth, level) : 1;
    offset += (baseSlice + slice) * rw * rh * rd + (depth * rh + y) * rw + x;
    ((uint32_t *)contents.mutableBytes)[offset] = value;
}

WMREF_CASE(texWriteLevels, "texture.write-levels", "texture.access",
           "Shader writes at every level and at levels out of range of textures and of level-range views")
{
    // A write is out of bounds when its level is (MSL 4.1 section 6.13): not
    // below the mipmap level count of the texture the function writes, which
    // for a view is the view's own count from its first level. Out-of-bounds
    // writes are ignored. Each write runs in a command buffer of its own, in
    // order, so the contents do not depend on how a device schedules writes
    // that land on one texel.
    MTLTextureDescriptor *viewable2D = TexDescriptor(MTLTextureType2D, 8, 8, 1, 4, 1);
    viewable2D.usage |= MTLTextureUsagePixelFormatView;
    MTLTextureDescriptor *viewableArray = TexDescriptor(MTLTextureType2DArray, 8, 8, 1, 3, 3);
    viewableArray.usage |= MTLTextureUsagePixelFormatView;
    // Writes Apple silicon does not ignore although the specification puts
    // them out of bounds (measured on the Apple M4: the level is taken modulo
    // 2^16, and a write past a view's last level lands in its parent's level
    // when the parent has it; reads at the same levels are out of bounds).
    WMRule *deviation = WMNone(@"MSL 4.1 section 6.13 ignores a write at a level out of range of the texture written "
                               @"(a view's own levels); Apple silicon writes some of these levels, so the result is "
                               @"recorded for reference only");
    struct {
        NSString *name, *kernel;
        MTLTextureDescriptor *desc;
        // The view the writes go through, of levels and slices from these
        // bases and of these counts, or none (count 0).
        MTLTextureType viewType;
        NSUInteger baseLevel, levels, baseSlice, slices;
        // The writes' results follow the specification on every device the
        // case was measured on.
        BOOL exact;
    } targets[] = {
        {@"2d", @"tex_write_2d", TexDescriptor(MTLTextureType2D, 8, 8, 1, 4, 1), MTLTextureType2D, 0, 0, 0, 0, YES},
        {@"level-view", @"tex_write_2d", viewable2D, MTLTextureType2D, 1, 1, 0, 1, YES},
        {@"array-view", @"tex_write_2d_array", viewableArray, MTLTextureType2DArray, 1, 1, 1, 1, YES},
        {@"3d", @"tex_write_3d", TexDescriptor(MTLTextureType3D, 4, 4, 4, 3, 1), MTLTextureType3D, 0, 0, 0, 0, YES},
        {@"cube", @"tex_write_cube", TexDescriptor(MTLTextureTypeCube, 4, 4, 1, 3, 1), MTLTextureTypeCube, 0, 0, 0, 0,
         YES},
        {@"2d.wrapped-levels", @"tex_write_2d", TexDescriptor(MTLTextureType2D, 8, 8, 1, 4, 1), MTLTextureType2D, 0, 0,
         0, 0, NO},
        {@"level-view.past-view", @"tex_write_2d", viewable2D, MTLTextureType2D, 1, 1, 0, 1, NO},
        {@"array-view.past-view", @"tex_write_2d_array", viewableArray, MTLTextureType2DArray, 1, 1, 1, 1, NO},
    };
    const TexQuery queries[][11] = {
        // 2D, 4 levels: one write at each level; levels 4, 5, 31, 32,
        // 2^16 - 1, 2^31 - 1 and 2^32 - 1 are out of range.
        {TexQ(1, 1, 0, 0, 0), TexQ(2, 1, 0, 0, 1), TexQ(1, 0, 0, 0, 2), TexQ(0, 0, 0, 0, 3), TexQ(3, 3, 0, 0, 4),
         TexQ(0, 1, 0, 0, 5), TexQ(1, 1, 0, 0, 31), TexQ(2, 2, 0, 0, 32), TexQ(0, 0, 0, 0, 0xffffu),
         TexQ(0, 0, 0, 0, 0x7fffffffu), TexQ(0, 0, 0, 0, 0xffffffffu)},
        // A view of level 1 of a 4-level 2D texture: its level 0 is in
        // range; its level 3 would be the parent's level 4.
        {TexQ(3, 2, 0, 0, 0), TexQ(2, 0, 0, 0, 0), TexQ(0, 0, 0, 0, 3), TexQ(2, 2, 0, 0, 0xffffffffu)},
        // A view of level 1 and slice 1 of a 3-level, 3-slice 2D array: its
        // slice 1 (the parent's slice 2) is outside it.
        {TexQ(1, 0, 0, 0, 0), TexQ(3, 3, 0, 0, 0), TexQ(2, 2, 1, 0, 0), TexQ(0, 0, 0, 0, 3)},
        // 3D, 3 levels: one write at each level; z = 1 at level 2 (1 x 1 x 1),
        // level 3 and level 2^32 - 1 are out of bounds.
        {TexQ(1, 2, 3, 0, 0), TexQ(1, 0, 1, 0, 1), TexQ(0, 0, 0, 0, 2), TexQ(0, 0, 1, 0, 2), TexQ(1, 1, 1, 0, 3),
         TexQ(0, 0, 0, 0, 0xffffffffu)},
        // Cube, 3 levels: one write at each level; level 3, face 6 and level
        // 2^32 - 1 are out of bounds.
        {TexQ(3, 0, 0, 0, 0), TexQ(1, 1, 2, 0, 1), TexQ(0, 0, 5, 0, 2), TexQ(0, 0, 1, 0, 3), TexQ(0, 0, 6, 0, 1),
         TexQ(1, 1, 3, 0, 0xffffffffu)},
        // 2D, 4 levels: levels 2^16, 2^16 + 3, 2^31 and 2^31 + 1 are out of
        // range; their low 16 bits are levels the texture has.
        {TexQ(1, 1, 0, 0, 0x10000u), TexQ(0, 0, 0, 0, 0x10003u), TexQ(2, 3, 0, 0, 0x80000000u),
         TexQ(1, 0, 0, 0, 0x80000001u)},
        // The view of level 1: its levels 1 and 2 are the parent's levels 2
        // and 3, which the parent has.
        {TexQ(1, 1, 0, 0, 1), TexQ(0, 0, 0, 0, 2)},
        // The view of level 1 and slice 1: its level 1 is the parent's level
        // 2, which the parent has.
        {TexQ(0, 1, 0, 0, 1), TexQ(1, 1, 0, 0, 1)},
    };
    const NSUInteger counts[] = {11, 4, 4, 6, 6, 4, 2, 2};
    for (int i = 0; i < 8; i++) {
        id<MTLTexture> root = [c texture:targets[i].desc];
        TexUploadAll(c, root);
        id<MTLTexture> t = root;
        if (targets[i].levels) {
            t = [root newTextureViewWithPixelFormat:MTLPixelFormatR32Uint textureType:targets[i].viewType
                                             levels:NSMakeRange(targets[i].baseLevel, targets[i].levels)
                                             slices:NSMakeRange(targets[i].baseSlice, targets[i].slices)];
            if (!t) [c fail:@"%@: the view returned nil", targets[i].name];
        }
        if (i == 1) {
            // Reads through the view at its levels 1 and 2, which the parent
            // has, are out of bounds (MSL 4.1 section 6.13).
            TexQuery rq[4] = {TexQ(1, 0, 0, 0, 0), TexQ(1, 0, 0, 0, 1), TexQ(0, 0, 0, 0, 2),
                              TexQ(0, 0, 0, 0, 0xffffffffu)};
            TexTexel re[4] = {TexR(TexValue(1, 0, 0, 1)), TexDefault(), TexDefault(), TexDefault()};
            TexRecordReads(c, @"level-view.reads", rq, 4, TexRunReads(c, @"tex_read_2d", t, rq, 4), re);
        }
        NSMutableData *expected = TexInitialContents(root);
        for (NSUInteger j = 0; j < counts[i]; j++) {
            uint32_t value = 0xc0de0000u | (uint32_t)(i << 8) | (uint32_t)j;
            TexTexel v = {{value, 0, 0, 0}};
            TexRunWrites(c, targets[i].kernel, t, &queries[i][j], &v, 1);
            TexApplyWrite(expected, root, t, targets[i].baseLevel, targets[i].baseSlice, queries[i][j], value);
        }
        [c data:[targets[i].name stringByAppendingString:@".writes"] elem:WMElemU32 bytes:queries[i]
         length:counts[i] * sizeof(TexQuery) rule:WMExact()];
        NSData *contents = TexReadAll(c, root);
        WMRule *rule = targets[i].exact ? WMExact() : deviation;
        [c data:[targets[i].name stringByAppendingString:@".contents"] elem:WMElemU32 bytes:contents.bytes
         length:contents.length rule:rule];
        [c value:[targets[i].name stringByAppendingString:@".contents.match-specification"]
         boolean:[contents isEqualToData:expected] rule:rule];
    }
}

#pragma mark - Views

static void TexUploadBytes(WMCase *c, id<MTLTexture> t, const void *bytes, NSUInteger bytesPerRow) {
    id<MTLBuffer> source = [c bufferWithBytes:bytes length:bytesPerRow * t.height];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:bytesPerRow sourceBytesPerImage:0
               sourceSize:MTLSizeMake(t.width, t.height, 1) toTexture:t destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    }];
}

static NSUInteger TexAllTexels2D(id<MTLTexture> t, TexQuery *q) {
    NSUInteger n = 0;
    for (uint32_t y = 0; y < t.height; y++)
        for (uint32_t x = 0; x < t.width; x++) q[n++] = TexQ(x, y, 0, 0, 0);
    return n;
}

// RGBA8 test data with the corner codes 0 and 255 in texel 0.
static void TexRGBA8Pattern(uint8_t bytes[64]) {
    WMPattern(bytes, 64, 71);
    bytes[0] = 0;
    bytes[1] = 255;
    bytes[2] = 0;
    bytes[3] = 255;
    bytes[4] = 255;
    bytes[5] = 0;
}

// Normalized reads of 8-bit codes: within 1.5 ulp of c / 255, and exactly
// 0.0 and 1.0 for the codes 0 and 255 (MSL 4.1 section 8.7.1.1, table 8.6).
static void TexRecordUnormReads(WMCase *c, NSString *name, const uint8_t *codes, NSData *reads, NSUInteger count) {
    const float *values = reads.bytes;
    double *ideal = calloc(count, sizeof(double));
    NSMutableData *corners = [NSMutableData data];
    for (NSUInteger i = 0; i < count; i++) {
        ideal[i] = codes[i] / 255.0;
        if (codes[i] == 0 || codes[i] == 255) {
            uint32_t pair[2] = {(uint32_t)i, 0};
            memcpy(&pair[1], &values[i], 4);
            [corners appendBytes:pair length:sizeof pair];
        }
    }
    [c bounded:name elem:WMElemF32 bytes:values count:count ideal:ideal model:@"ulp" bound:@"1.5"
     reference:@"MSL 4.1 section 8.7.1.1, table 8.6 (8-bit normalized unsigned integer)"];
    // Pairs of (element index, float bits) for the corner codes.
    [c data:[name stringByAppendingString:@".corner-codes"] elem:WMElemU32 bytes:corners.bytes
     length:corners.length rule:WMExact()];
    free(ideal);
}

WMREF_CASE(texViewFormat, "texture.view-format-reinterpret", "texture.views",
           "Pixel-format views: RGBA8Unorm as RGBA8Uint, R32Float as R32Uint, sRGB and linear views")
{
    MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
    uint8_t pattern[64];
    TexRGBA8Pattern(pattern);
    TexQuery q[16];

    // RGBA8Unorm viewed as RGBA8Uint.
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                   width:4 height:4 mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = usage;
    id<MTLTexture> unorm = [c texture:desc];
    TexUploadBytes(c, unorm, pattern, 16);
    id<MTLTexture> uintView = [unorm newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Uint];
    if (!uintView) [c fail:@"RGBA8Uint view returned nil"];
    [c value:@"uint-view.pixel-format" u64:uintView.pixelFormat rule:WMExact()];
    [c value:@"uint-view.parent-is-texture" boolean:uintView.parentTexture == unorm rule:WMExact()];
    NSUInteger n = TexAllTexels2D(unorm, q);
    NSData *uintReads = TexRunReads(c, @"tex_read_2d", uintView, q, n);
    TexTexel expected[16];
    for (NSUInteger i = 0; i < 16; i++)
        expected[i] = (TexTexel){{pattern[4 * i], pattern[4 * i + 1], pattern[4 * i + 2], pattern[4 * i + 3]}};
    TexRecordReads(c, @"uint-view.reads", q, n, uintReads, expected);
    NSData *floatReads = TexRunReads(c, @"tex_read_2d_float", unorm, q, n);
    TexRecordUnormReads(c, @"unorm.reads", pattern, floatReads, 64);

    // Writes through the uint view saturate to 8 bits (MSL 4.1 section
    // 8.7.6, table 8.8) and are visible through the parent.
    TexQuery wq[2] = {TexQ(1, 0, 0, 0, 0), TexQ(2, 3, 0, 0, 0)};
    TexTexel wv[2] = {{{1, 128, 255, 256}}, {{1000, 0, 77, 0xffffffff}}};
    TexRunWrites(c, @"tex_write_2d", uintView, wq, wv, 2);
    NSData *after = [c readTexture:unorm level:0 slice:0 bytesPerRow:16];
    [c data:@"unorm.after-uint-writes" elem:WMElemU8 bytes:after.bytes length:after.length rule:WMExact()];

    // R32Float viewed as R32Uint: bit patterns written through the uint view
    // are stored unchanged; float reads preserve normal values, and may
    // flush denormals and change NaNs (MSL 4.1 section 8.7.3).
    MTLTextureDescriptor *fd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                 width:4 height:2 mipmapped:NO];
    fd.storageMode = MTLStorageModePrivate;
    fd.usage = usage;
    id<MTLTexture> floats = [c texture:fd];
    id<MTLTexture> bitsView = [floats newTextureViewWithPixelFormat:MTLPixelFormatR32Uint];
    if (!bitsView) [c fail:@"R32Uint view returned nil"];
    const uint32_t bits[8] = {0x3f800000, 0xc2f6e979, 0x00000000, 0x80000000, 0x7f7fffff,
                              0x7f800000, // normal values and signed zeros, infinity
                              0x00000001, 0x7fc00001}; // a denormal and a NaN
    TexQuery fq[8];
    TexTexel fv[8];
    for (uint32_t i = 0; i < 8; i++) {
        fq[i] = TexQ(i % 4, i / 4, 0, 0, 0);
        fv[i] = (TexTexel){{bits[i], 0, 0, 0}};
    }
    TexRunWrites(c, @"tex_write_2d", bitsView, fq, fv, 8);
    NSData *stored = [c readTexture:floats level:0 slice:0 bytesPerRow:16];
    [c data:@"r32float.stored-bits" elem:WMElemU32 bytes:stored.bytes length:stored.length rule:WMExact()];
    NSData *floatBits = TexRunReads(c, @"tex_read_2d_float", floats, fq, 8);
    const uint32_t *fb = floatBits.bytes;
    uint32_t preserved[6], other[2];
    for (int i = 0; i < 6; i++) preserved[i] = fb[4 * i];
    for (int i = 0; i < 2; i++) other[i] = fb[4 * (6 + i)];
    [c data:@"r32float.reads.preserved" elem:WMElemU32 bytes:preserved length:sizeof preserved rule:WMExact()];
    [c data:@"r32float.reads.denormal-and-nan" elem:WMElemU32 bytes:other length:sizeof other
       rule:WMNone(@"MSL 4.1 section 8.7.3: denormals may be flushed to zero and NaNs may be converted")];

    // sRGB texture with linear and uint views.
    MTLTextureDescriptor *sd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB
                                                                                 width:4 height:4 mipmapped:NO];
    sd.storageMode = MTLStorageModePrivate;
    sd.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    id<MTLTexture> srgb = [c texture:sd];
    TexUploadBytes(c, srgb, pattern, 16);
    id<MTLTexture> linear = [srgb newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
    id<MTLTexture> raw = [srgb newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Uint];
    if (!linear || !raw) [c fail:@"views of the sRGB texture returned nil"];
    NSData *decoded = TexRunReads(c, @"tex_read_2d_float", srgb, q, n);
    float rgb[48], alpha[16];
    double rgbCodes[48], alphaIdeal[16];
    const float *dv = decoded.bytes;
    for (int i = 0; i < 16; i++) {
        for (int k = 0; k < 3; k++) {
            rgb[3 * i + k] = dv[4 * i + k];
            rgbCodes[3 * i + k] = pattern[4 * i + k];
        }
        alpha[i] = dv[4 * i + 3];
        alphaIdeal[i] = pattern[4 * i + 3] / 255.0;
    }
    // sRGB decode: converting the result back to an unrounded sRGB code is
    // within 0.5 of the stored code; alpha is stored linearly (MSL 4.1
    // section 8.7.7).
    [c bounded:@"srgb.reads.rgb" elem:WMElemF32 bytes:rgb count:48 ideal:rgbCodes model:@"srgb8-decode" bound:@"0.5"
     reference:@"MSL 4.1 section 8.7.7"];
    [c bounded:@"srgb.reads.alpha" elem:WMElemF32 bytes:alpha count:16 ideal:alphaIdeal model:@"ulp" bound:@"1.5"
     reference:@"MSL 4.1 sections 8.7.7 and 8.7.1.1"];
    TexRecordUnormReads(c, @"srgb.linear-view.reads", pattern, TexRunReads(c, @"tex_read_2d_float", linear, q, n), 64);
    TexRecordReads(c, @"srgb.uint-view.reads", q, n, TexRunReads(c, @"tex_read_2d", raw, q, n), expected);
}

WMREF_CASE(texViewLevelsSlices, "texture.view-levels-slices", "texture.views",
           "Level and slice range views, type-changing views (array slice as 2D, cube as array), views of views")
{
    MTLTextureDescriptor *desc = TexDescriptor(MTLTextureType2DArray, 8, 8, 1, 3, 4);
    desc.usage |= MTLTextureUsagePixelFormatView;
    id<MTLTexture> parent = [c texture:desc];
    TexUploadAll(c, parent);

    // Levels [1, 3) and slices [1, 3) as a 2D array.
    id<MTLTexture> range = [parent newTextureViewWithPixelFormat:MTLPixelFormatR32Uint textureType:MTLTextureType2DArray
                                                          levels:NSMakeRange(1, 2) slices:NSMakeRange(1, 2)];
    if (!range) [c fail:@"level and slice range view returned nil"];
    NSArray *properties = @[@(range.textureType), @(range.width), @(range.height), @(range.mipmapLevelCount),
                            @(range.arrayLength), @(range.parentRelativeLevel), @(range.parentRelativeSlice)];
    for (NSUInteger i = 0; i < properties.count; i++) {
        NSString *names[] = {@"type", @"width", @"height", @"mipmap-level-count", @"array-length",
                             @"parent-relative-level", @"parent-relative-slice"};
        [c value:[@"range-view." stringByAppendingString:names[i]] u64:[properties[i] unsignedLongLongValue]
            rule:WMExact()];
    }
    [c value:@"range-view.parent-is-texture" boolean:range.parentTexture == parent rule:WMExact()];
    NSMutableData *qd = [NSMutableData data], *ed = [NSMutableData data];
    for (uint32_t s = 0; s < 2; s++)
        for (uint32_t l = 0; l < 2; l++)
            for (uint32_t y = 0; y < TexMip(4, l); y++)
                for (uint32_t x = 0; x < TexMip(4, l); x++) {
                    TexQuery q = TexQ(x, y, s, 0, l);
                    TexTexel e = TexR(TexValue(1 + l, 1 + s, y, x));
                    [qd appendBytes:&q length:sizeof q];
                    [ed appendBytes:&e length:sizeof e];
                }
    // Slice 2 and level 2 of the view are outside it.
    const uint32_t outside[][4] = {{0, 0, 2, 0}, {0, 0, 0, 2}, {4, 0, 0, 0}};
    for (int i = 0; i < 3; i++) {
        TexQuery q = TexQ(outside[i][0], outside[i][1], outside[i][2], 0, outside[i][3]);
        TexTexel e = TexDefault();
        [qd appendBytes:&q length:sizeof q];
        [ed appendBytes:&e length:sizeof e];
    }
    NSUInteger n = qd.length / sizeof(TexQuery);
    TexRecordReads(c, @"range-view.reads", qd.bytes, n, TexRunReads(c, @"tex_read_2d_array", range, qd.bytes, n),
                   ed.bytes);

    // A write through the view lands in parent level 1, slice 2.
    TexQuery wq = TexQ(3, 2, 1, 0, 0);
    TexTexel wv = {{0xfeedf00d, 0, 0, 0}};
    TexRunWrites(c, @"tex_write_2d_array", range, &wq, &wv, 1);
    NSData *written = [c readTexture:parent level:1 slice:2 bytesPerRow:16];
    [c data:@"parent.level1.slice2.after-view-write" elem:WMElemU32 bytes:written.bytes length:written.length
       rule:WMExact()];

    // Slice 3 as a 2D texture with all levels.
    id<MTLTexture> slice2D = [parent newTextureViewWithPixelFormat:MTLPixelFormatR32Uint textureType:MTLTextureType2D
                                                            levels:NSMakeRange(0, 3) slices:NSMakeRange(3, 1)];
    if (!slice2D) [c fail:@"2D view of an array slice returned nil"];
    [qd setLength:0];
    [ed setLength:0];
    for (uint32_t l = 0; l < 3; l++) {
        TexQuery q = TexQ(1, 1, 0, 0, l);
        TexTexel e = TexR(TexValue(l, 3, 1, 1));
        [qd appendBytes:&q length:sizeof q];
        [ed appendBytes:&e length:sizeof e];
    }
    TexRecordReads(c, @"slice-view.reads", qd.bytes, 3, TexRunReads(c, @"tex_read_2d", slice2D, qd.bytes, 3), ed.bytes);

    // A view of the range view: its level 1 and slice 1.
    id<MTLTexture> nested = [range newTextureViewWithPixelFormat:MTLPixelFormatR32Uint textureType:MTLTextureType2D
                                                          levels:NSMakeRange(1, 1) slices:NSMakeRange(1, 1)];
    if (!nested) [c fail:@"view of a view returned nil"];
    [c value:@"nested-view.parent-is-range-view" boolean:nested.parentTexture == range rule:WMExact()];
    [c value:@"nested-view.parent-is-root" boolean:nested.parentTexture == parent rule:WMExact()];
    [c value:@"nested-view.parent-relative-level" u64:nested.parentRelativeLevel rule:WMExact()];
    [c value:@"nested-view.parent-relative-slice" u64:nested.parentRelativeSlice rule:WMExact()];
    [c value:@"nested-view.width" u64:nested.width rule:WMExact()];
    TexQuery nq = TexQ(1, 0, 0, 0, 0);
    TexTexel ne = TexR(TexValue(2, 2, 0, 1));
    TexRecordReads(c, @"nested-view.reads", &nq, 1, TexRunReads(c, @"tex_read_2d", nested, &nq, 1), &ne);

    // A cube as a six-slice 2D array, one face as 2D, and a six-slice 2D
    // array as a cube.
    MTLTextureDescriptor *cd = TexDescriptor(MTLTextureTypeCube, 4, 4, 1, 2, 1);
    cd.usage |= MTLTextureUsagePixelFormatView;
    id<MTLTexture> cube = [c texture:cd];
    TexUploadAll(c, cube);
    id<MTLTexture> faces = [cube newTextureViewWithPixelFormat:MTLPixelFormatR32Uint
                                                   textureType:MTLTextureType2DArray
                                                        levels:NSMakeRange(0, 2) slices:NSMakeRange(0, 6)];
    id<MTLTexture> face = [cube newTextureViewWithPixelFormat:MTLPixelFormatR32Uint textureType:MTLTextureType2D
                                                       levels:NSMakeRange(0, 1) slices:NSMakeRange(3, 1)];
    if (!faces || !face) [c fail:@"views of the cube returned nil"];
    [c value:@"cube-as-array.array-length" u64:faces.arrayLength rule:WMExact()];
    [qd setLength:0];
    [ed setLength:0];
    for (uint32_t f = 0; f < 6; f++)
        for (uint32_t l = 0; l < 2; l++) {
            TexQuery q = TexQ(1, 0, f, 0, l);
            TexTexel e = TexR(TexValue(l, f, 0, 1));
            [qd appendBytes:&q length:sizeof q];
            [ed appendBytes:&e length:sizeof e];
        }
    TexRecordReads(c, @"cube-as-array.reads", qd.bytes, 12, TexRunReads(c, @"tex_read_2d_array", faces, qd.bytes, 12),
                   ed.bytes);
    TexQuery fq = TexQ(2, 3, 0, 0, 0);
    TexTexel fe = TexR(TexValue(0, 3, 3, 2));
    TexRecordReads(c, @"face-as-2d.reads", &fq, 1, TexRunReads(c, @"tex_read_2d", face, &fq, 1), &fe);

    MTLTextureDescriptor *ad = TexDescriptor(MTLTextureType2DArray, 2, 2, 1, 1, 6);
    ad.usage |= MTLTextureUsagePixelFormatView;
    id<MTLTexture> array = [c texture:ad];
    TexUploadAll(c, array);
    id<MTLTexture> asCube = [array newTextureViewWithPixelFormat:MTLPixelFormatR32Uint textureType:MTLTextureTypeCube
                                                          levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 6)];
    if (!asCube) [c fail:@"cube view of a 2D array returned nil"];
    [qd setLength:0];
    [ed setLength:0];
    for (uint32_t f = 0; f < 6; f++) {
        TexQuery q = TexQ(f % 2, f / 4, f, 0, 0);
        TexTexel e = TexR(TexValue(0, f, f / 4, f % 2));
        [qd appendBytes:&q length:sizeof q];
        [ed appendBytes:&e length:sizeof e];
    }
    TexRecordReads(c, @"array-as-cube.reads", qd.bytes, 6, TexRunReads(c, @"tex_read_cube", asCube, qd.bytes, 6),
                   ed.bytes);
}

static TexTexel TexSwizzled(const uint8_t *rgba, MTLTextureSwizzleChannels s, BOOL hasGBA) {
    uint32_t channel[6];
    channel[MTLTextureSwizzleZero] = 0;
    channel[MTLTextureSwizzleOne] = 1;
    channel[MTLTextureSwizzleRed] = rgba[0];
    // A one-component format reads its missing components as 0 and alpha
    // as 1 (MSL 4.1 section 6.13).
    channel[MTLTextureSwizzleGreen] = hasGBA ? rgba[1] : 0;
    channel[MTLTextureSwizzleBlue] = hasGBA ? rgba[2] : 0;
    channel[MTLTextureSwizzleAlpha] = hasGBA ? rgba[3] : 1;
    return (TexTexel){{channel[s.red], channel[s.green], channel[s.blue], channel[s.alpha]}};
}

WMREF_CASE(texViewSwizzle, "texture.view-swizzle", "texture.views",
           "Swizzled views and descriptor swizzles on RGBA8Uint and R8Uint for read and sample")
{
    // Metal feature set tables: texture swizzle from Apple2.
    [c require:@"family.apple2" value:[c.device supportsFamily:MTLGPUFamilyApple2]];
    uint8_t pattern[64];
    TexRGBA8Pattern(pattern);
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Uint
                                                                                   width:4 height:4 mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    id<MTLTexture> base = [c texture:desc];
    TexUploadBytes(c, base, pattern, 16);
    TexQuery q[16];
    NSUInteger n = TexAllTexels2D(base, q);

    MTLTextureSwizzleChannels viewSwizzle =
        MTLTextureSwizzleChannelsMake(MTLTextureSwizzleBlue, MTLTextureSwizzleZero, MTLTextureSwizzleOne,
                                      MTLTextureSwizzleRed);
    id<MTLTexture> view = [base newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Uint textureType:MTLTextureType2D
                                                       levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 1)
                                                      swizzle:viewSwizzle];
    if (!view) [c fail:@"swizzled view returned nil"];
    MTLTextureSwizzleChannels reported = view.swizzle;
    [c data:@"view.swizzle-property" elem:WMElemU8 bytes:&reported length:4 rule:WMExact()];
    TexTexel expected[16];
    for (NSUInteger i = 0; i < 16; i++) expected[i] = TexSwizzled(pattern + 4 * i, viewSwizzle, YES);
    TexRecordReads(c, @"view.reads", q, n, TexRunReads(c, @"tex_read_2d", view, q, n), expected);

    // Descriptor swizzle, read and nearest-sampled at texel centres.
    MTLTextureSwizzleChannels descSwizzle =
        MTLTextureSwizzleChannelsMake(MTLTextureSwizzleAlpha, MTLTextureSwizzleRed, MTLTextureSwizzleGreen,
                                      MTLTextureSwizzleBlue);
    desc.swizzle = descSwizzle;
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> swizzled = [c texture:desc];
    TexUploadBytes(c, swizzled, pattern, 16);
    for (NSUInteger i = 0; i < 16; i++) expected[i] = TexSwizzled(pattern + 4 * i, descSwizzle, YES);
    TexRecordReads(c, @"descriptor.reads", q, n, TexRunReads(c, @"tex_read_2d", swizzled, q, n), expected);
    float centres[32];
    for (int i = 0; i < 16; i++) {
        centres[2 * i] = ((i % 4) + 0.5f) / 4;
        centres[2 * i + 1] = ((i / 4) + 0.5f) / 4;
    }
    id<MTLSamplerState> sampler = [c sampler:[MTLSamplerDescriptor new]];
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"tex_sample_2d_uint"];
    id<MTLBuffer> coords = [c bufferWithBytes:centres length:sizeof centres];
    id<MTLBuffer> out = [c buffer:16 * 16];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setTexture:swizzled atIndex:0];
        [e setSamplerState:sampler atIndex:0];
        [e setBuffer:coords offset:0 atIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }];
    [c data:@"descriptor.samples" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
    [c value:@"descriptor.samples.match-specification" boolean:memcmp(out.contents, expected, sizeof expected) == 0
        rule:WMExact()];

    // R8Uint with swizzles that select missing components.
    MTLTextureDescriptor *rd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Uint
                                                                                 width:4 height:4 mipmapped:NO];
    rd.storageMode = MTLStorageModePrivate;
    rd.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    id<MTLTexture> single = [c texture:rd];
    uint8_t reds[16];
    WMPattern(reds, sizeof reds, 72);
    TexUploadBytes(c, single, reds, 4);
    MTLTextureSwizzleChannels redSwizzle =
        MTLTextureSwizzleChannelsMake(MTLTextureSwizzleAlpha, MTLTextureSwizzleGreen, MTLTextureSwizzleRed,
                                      MTLTextureSwizzleBlue);
    id<MTLTexture> redView = [single newTextureViewWithPixelFormat:MTLPixelFormatR8Uint textureType:MTLTextureType2D
                                                            levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 1)
                                                           swizzle:redSwizzle];
    if (!redView) [c fail:@"swizzled R8Uint view returned nil"];
    for (NSUInteger i = 0; i < 16; i++) {
        uint8_t rgba[4] = {reds[i], 0, 0, 0};
        expected[i] = TexSwizzled(rgba, redSwizzle, NO);
    }
    TexRecordReads(c, @"r8uint-view.reads", q, n, TexRunReads(c, @"tex_read_2d", redView, q, n), expected);
}

#pragma mark - Mipmaps and properties

WMREF_CASE(texGenerateMipmaps, "texture.generate-mipmaps", "texture.mipmaps",
           "generateMipmapsForTexture: on an 8x8 RGBA8Unorm texture with four levels")
{
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                   width:8 height:8 mipmapped:YES];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    id<MTLTexture> t = [c texture:desc];
    [c value:@"mipmap-level-count" u64:t.mipmapLevelCount rule:WMExact()];
    uint8_t base[256];
    WMPattern(base, sizeof base, 73);
    TexUploadBytes(c, t, base, 32);
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e generateMipmapsForTexture:t];
    }];
    for (NSUInteger level = 0; level < t.mipmapLevelCount; level++) {
        NSUInteger w = TexMip(8, level);
        NSData *data = [c readTexture:t level:level slice:0 bytesPerRow:w * 4];
        NSDictionary *layout = @{@"format": @"RGBA8Unorm", @"width": @(w), @"height": @(w), @"depth": @1,
                                 @"bytes_per_row": @(w * 4), @"level": @(level)};
        // Level 0 is the source and is unchanged; MTLBlitCommandEncoder.h
        // states only that the other levels are generated from it.
        WMRule *rule = level == 0 ? WMExact()
                                  : WMNone(@"Metal does not specify the filter generateMipmapsForTexture: uses");
        [c data:[NSString stringWithFormat:@"level%lu", (unsigned long)level] elem:WMElemU8 bytes:data.bytes
         length:data.length layout:layout rule:rule];
    }
}

static void TexRecordProperties(WMCase *c, NSString *prefix, id<MTLTexture> t) {
    NSString *(^name)(NSString *) = ^(NSString *suffix) {
        return [NSString stringWithFormat:@"%@.%@", prefix, suffix];
    };
    [c value:name(@"type") u64:t.textureType rule:WMExact()];
    [c value:name(@"pixel-format") u64:t.pixelFormat rule:WMExact()];
    [c value:name(@"width") u64:t.width rule:WMExact()];
    // MTLTexture.h: height is 1 for 1D textures; depth is 1 for non-3D
    // textures; sampleCount is 1 except for multisample types; arrayLength
    // is 1 for non-array types.
    [c value:name(@"height") u64:t.height rule:WMExact()];
    [c value:name(@"depth") u64:t.depth rule:WMExact()];
    [c value:name(@"mipmap-level-count") u64:t.mipmapLevelCount rule:WMExact()];
    [c value:name(@"sample-count") u64:t.sampleCount rule:WMExact()];
    [c value:name(@"array-length") u64:t.arrayLength rule:WMExact()];
    [c value:name(@"usage") u64:t.usage rule:WMExact()];
    [c value:name(@"storage-mode") u64:t.storageMode rule:WMExact()];
    [c value:name(@"cpu-cache-mode") u64:t.cpuCacheMode rule:WMExact()];
    // MTLTexture.h: hazardTrackingMode Default is treated as Tracked for
    // resources created from the device.
    [c value:name(@"hazard-tracking-mode") u64:t.hazardTrackingMode rule:WMExact()];
    [c value:name(@"allow-gpu-optimized-contents") boolean:t.allowGPUOptimizedContents rule:WMExact()];
    // MTLTexture.h: textures created directly by the application have no
    // framebufferOnly restriction.
    [c value:name(@"framebuffer-only") boolean:t.framebufferOnly rule:WMExact()];
    [c value:name(@"shareable") boolean:t.shareable rule:WMExact()];
    // MTLTexture.h: nil and 0 when the texture is not a view.
    [c value:name(@"parent-texture-is-nil") boolean:t.parentTexture == nil rule:WMExact()];
    [c value:name(@"buffer-is-nil") boolean:t.buffer == nil rule:WMExact()];
    [c value:name(@"parent-relative-level") u64:t.parentRelativeLevel rule:WMExact()];
    [c value:name(@"parent-relative-slice") u64:t.parentRelativeSlice rule:WMExact()];
    MTLTextureSwizzleChannels swizzle = t.swizzle;
    [c data:name(@"swizzle") elem:WMElemU8 bytes:&swizzle length:4 rule:WMExact()];
}

WMREF_CASE(texProperties, "texture.properties", "texture.properties",
           "Texture descriptor defaults and the properties of created textures of several types")
{
    MTLTextureDescriptor *defaults = [MTLTextureDescriptor new];
    // Defaults stated in MTLTexture.h.
    [c value:@"descriptor-default.type" u64:defaults.textureType rule:WMExact()];
    [c value:@"descriptor-default.pixel-format" u64:defaults.pixelFormat rule:WMExact()];
    [c value:@"descriptor-default.width" u64:defaults.width rule:WMExact()];
    [c value:@"descriptor-default.height" u64:defaults.height rule:WMExact()];
    [c value:@"descriptor-default.depth" u64:defaults.depth rule:WMExact()];
    [c value:@"descriptor-default.mipmap-level-count" u64:defaults.mipmapLevelCount rule:WMExact()];
    [c value:@"descriptor-default.sample-count" u64:defaults.sampleCount rule:WMExact()];
    [c value:@"descriptor-default.array-length" u64:defaults.arrayLength rule:WMExact()];
    [c value:@"descriptor-default.hazard-tracking-mode" u64:defaults.hazardTrackingMode rule:WMExact()];
    [c value:@"descriptor-default.allow-gpu-optimized-contents" boolean:defaults.allowGPUOptimizedContents
        rule:WMExact()];
    [c value:@"descriptor-default.compression-type" i64:defaults.compressionType rule:WMExact()];
    MTLTextureSwizzleChannels swizzle = defaults.swizzle;
    [c data:@"descriptor-default.swizzle" elem:WMElemU8 bytes:&swizzle length:4 rule:WMExact()];
    // Defaults MTLTexture.h does not state; they belong to the framework of
    // the platform under test.
    [c value:@"descriptor-default.usage" u64:defaults.usage rule:WMDevice()];
    [c value:@"descriptor-default.storage-mode" u64:defaults.storageMode rule:WMDevice()];
    [c value:@"descriptor-default.cpu-cache-mode" u64:defaults.cpuCacheMode rule:WMDevice()];
    [c value:@"descriptor-default.resource-options" u64:defaults.resourceOptions rule:WMDevice()];

    // texture2DDescriptorWithPixelFormat:...mipmapped:YES describes the full
    // chain down to 1x1: 1 + floor(log2(max(13, 7))) = 4 levels.
    MTLTextureDescriptor *chain = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                    width:13 height:7 mipmapped:YES];
    [c value:@"mipmapped-13x7.mipmap-level-count" u64:chain.mipmapLevelCount rule:WMExact()];
    MTLTextureDescriptor *cubeChain = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                            size:5 mipmapped:YES];
    [c value:@"cube-5.type" u64:cubeChain.textureType rule:WMExact()];
    [c value:@"cube-5.mipmap-level-count" u64:cubeChain.mipmapLevelCount rule:WMExact()];

    MTLTextureDescriptor *d1 = TexDescriptor(MTLTextureType1D, 16, 1, 1, 1, 1);
    TexRecordProperties(c, @"1d", [c texture:d1]);
    MTLTextureDescriptor *d2 = TexDescriptor(MTLTextureType2DArray, 8, 4, 1, 2, 3);
    d2.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    d2.allowGPUOptimizedContents = NO;
    TexRecordProperties(c, @"2d-array", [c texture:d2]);
    MTLTextureDescriptor *d3 = TexDescriptor(MTLTextureType3D, 4, 4, 2, 1, 1);
    d3.storageMode = MTLStorageModeShared;
    d3.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    TexRecordProperties(c, @"3d-shared-untracked", [c texture:d3]);
    MTLTextureDescriptor *d4 = TexDescriptor(MTLTextureTypeCube, 4, 4, 1, 3, 1);
    TexRecordProperties(c, @"cube", [c texture:d4]);
    if ([c.device supportsTextureSampleCount:4]) {
        MTLTextureDescriptor *d5 = TexDescriptor(MTLTextureType2DMultisample, 4, 4, 1, 1, 1);
        d5.sampleCount = 4;
        d5.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        TexRecordProperties(c, @"2d-multisample", [c texture:d5]);
    }
}

#pragma mark - Invalid descriptors and views

// Creating a texture or view from an invalid description: what Metal does
// without its debug layer is not defined, so whether an object came back is
// reference data; with the debug layer the case's outcome records the
// validation failure.
static void TexRecordCreation(WMCase *c, id (^create)(void)) {
    __block id object = nil;
    [c call:@"create" rule:WMExact() block:^{
        object = create();
    }];
    [c value:@"returned-nil" boolean:object == nil
        rule:WMNone(@"Metal does not define the result of an invalid creation call without its debug layer")];
}

WMREF_CASE(texError1DMipmapped, "texture.error-1d-mipmapped", "texture.errors",
           "1D texture descriptor with two mipmap levels (MSL: 1D textures have no mipmaps)")
{
    MTLTextureDescriptor *desc = TexDescriptor(MTLTextureType1D, 16, 1, 1, 2, 1);
    [c step:@"newTextureWithDescriptor"];
    TexRecordCreation(c, ^id { return [c.device newTextureWithDescriptor:desc]; });
}

WMREF_CASE(texErrorTooManyLevels, "texture.error-too-many-levels", "texture.errors",
           "8x8 2D texture descriptor with five mipmap levels (the full chain has four)")
{
    MTLTextureDescriptor *desc = TexDescriptor(MTLTextureType2D, 8, 8, 1, 5, 1);
    [c step:@"newTextureWithDescriptor"];
    TexRecordCreation(c, ^id { return [c.device newTextureWithDescriptor:desc]; });
}

WMREF_CASE(texErrorMultisampleMipmapped, "texture.error-multisample-mipmapped", "texture.errors",
           "Multisample texture descriptor with two mipmap levels (MTLTexture.h: must be 1)")
{
    MTLTextureDescriptor *desc = TexDescriptor(MTLTextureType2DMultisample, 4, 4, 1, 2, 1);
    desc.sampleCount = 4;
    desc.usage = MTLTextureUsageRenderTarget;
    [c step:@"newTextureWithDescriptor"];
    TexRecordCreation(c, ^id { return [c.device newTextureWithDescriptor:desc]; });
}

WMREF_CASE(texErrorViewSize, "texture.error-view-size-mismatch", "texture.errors",
           "RGBA8Unorm texture viewed as R8Unorm (different texel size)")
{
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                   width:4 height:4 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [c texture:desc];
    [c step:@"newTextureViewWithPixelFormat"];
    TexRecordCreation(c, ^id { return [t newTextureViewWithPixelFormat:MTLPixelFormatR8Unorm]; });
}

WMREF_CASE(texErrorViewUsage, "texture.error-view-without-usage", "texture.errors",
           "RGBA8Unorm texture without MTLTextureUsagePixelFormatView viewed as RGBA8Uint")
{
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                   width:4 height:4 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [c texture:desc];
    [c step:@"newTextureViewWithPixelFormat"];
    TexRecordCreation(c, ^id { return [t newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Uint]; });
}

WMREF_CASE(texErrorViewSRGBWithoutUsage, "texture.view-srgb-without-usage", "texture.views",
           "RGBA8Unorm texture without MTLTextureUsagePixelFormatView viewed as RGBA8Unorm_sRGB")
{
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                   width:4 height:4 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [c texture:desc];
    [c step:@"newTextureViewWithPixelFormat"];
    TexRecordCreation(c, ^id { return [t newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB]; });
}
