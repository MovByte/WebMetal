// Sampler cases: address modes and border colors with nearest filtering at
// texel centres, unnormalized coordinates, LOD selection and clamps
// (sampler, shader option and per-binding), compare functions, gather,
// samplers declared in MSL, linear and anisotropic filtering, sampler
// properties and invalid sampler descriptors.
//
// Nearest-filtered samples at texel centres of RGBA32Float and R32Float
// textures involve no format conversion and no filter weights, so the
// results depend only on texel selection by the address modes and are
// compared exactly.

#import "WMRefCase.h"

enum { kSmpSize = 8 };

// 8x8 RGBA32Float texture, texel (x, y) = (x + 1, y + 1, 100 + x + 8y, 0.5).
static id<MTLTexture> SmpRGBA32Texture(WMCase *c) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                   width:kSmpSize height:kSmpSize
                                                                               mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [c texture:desc];
    float texels[kSmpSize * kSmpSize * 4];
    for (NSUInteger y = 0; y < kSmpSize; y++)
        for (NSUInteger x = 0; x < kSmpSize; x++) {
            float *p = texels + (y * kSmpSize + x) * 4;
            p[0] = x + 1;
            p[1] = y + 1;
            p[2] = 100 + x + 8 * y;
            p[3] = 0.5f;
        }
    id<MTLBuffer> source = [c bufferWithBytes:texels length:sizeof texels];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:kSmpSize * 16 sourceBytesPerImage:0
               sourceSize:MTLSizeMake(kSmpSize, kSmpSize, 1) toTexture:t destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    }];
    return t;
}

static void SmpTexel(NSInteger x, NSInteger y, float out[4]) {
    out[0] = x + 1;
    out[1] = y + 1;
    out[2] = 100 + x + 8 * y;
    out[3] = 0.5f;
}

// Texel index selected by an address mode for the texel-centre coordinate
// (k + 0.5) / n, from the descriptions in MTLSampler.h; -1 when the
// coordinate is outside the texture for the clamp-to-zero and border modes.
static NSInteger SmpAddress(MTLSamplerAddressMode mode, NSInteger k, NSInteger n) {
    switch (mode) {
    case MTLSamplerAddressModeClampToEdge:
        return MIN(MAX(k, 0), n - 1);
    case MTLSamplerAddressModeRepeat:
        return ((k % n) + n) % n;
    case MTLSamplerAddressModeMirrorRepeat: {
        NSInteger m = ((k % (2 * n)) + 2 * n) % (2 * n);
        return m < n ? m : 2 * n - 1 - m;
    }
    case MTLSamplerAddressModeMirrorClampToEdge:
        return MIN(MAX(k >= 0 ? k : -1 - k, 0), n - 1);
    case MTLSamplerAddressModeClampToZero:
    case MTLSamplerAddressModeClampToBorderColor:
        return k >= 0 && k < n ? k : -1;
    }
    return -1;
}

static NSData *SmpRun(WMCase *c, NSString *kernel, id<MTLTexture> t, id<MTLSamplerState> s, const void *coords,
                      size_t coordSize, NSUInteger count, NSUInteger outPerCoord, size_t outSize) {
    id<MTLComputePipelineState> pipeline = [c computePipeline:kernel];
    id<MTLBuffer> in = [c bufferWithBytes:coords length:coordSize * count];
    id<MTLBuffer> out = [c buffer:count * outPerCoord * outSize];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setTexture:t atIndex:0];
        if (s) [e setSamplerState:s atIndex:0];
        [e setBuffer:in offset:0 atIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }];
    return [NSData dataWithBytes:out.contents length:out.length];
}

static MTLSamplerDescriptor *SmpNearest(MTLSamplerAddressMode s, MTLSamplerAddressMode t) {
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.minFilter = MTLSamplerMinMagFilterNearest;
    d.magFilter = MTLSamplerMinMagFilterNearest;
    d.sAddressMode = s;
    d.tAddressMode = t;
    d.rAddressMode = s;
    return d;
}

// Texel-centre offsets k of the coordinates (k + 0.5) / 8: inside, at the
// edges, one and two periods outside on both sides.
static const NSInteger kSmpU[] = {-17, -9, -8, -1, 0, 3, 7, 8, 12, 15, 16, 23};
static const NSInteger kSmpV[] = {-6, 2, 9};
#define SMP_COUNT (sizeof kSmpU / sizeof kSmpU[0] * (sizeof kSmpV / sizeof kSmpV[0]))

static void SmpCoordinates(float *coords) {
    NSUInteger n = 0;
    for (size_t j = 0; j < sizeof kSmpV / sizeof kSmpV[0]; j++)
        for (size_t i = 0; i < sizeof kSmpU / sizeof kSmpU[0]; i++, n++) {
            coords[2 * n] = (kSmpU[i] + 0.5f) / kSmpSize;
            coords[2 * n + 1] = (kSmpV[j] + 0.5f) / kSmpSize;
        }
}

// Expected sample for address modes (s, t) under the MTLSampler.h
// descriptions; `outside` is the value for clamp-to-zero or border modes.
static void SmpExpected(MTLSamplerAddressMode s, MTLSamplerAddressMode t, const float outside[4], float *expected) {
    NSUInteger n = 0;
    for (size_t j = 0; j < sizeof kSmpV / sizeof kSmpV[0]; j++)
        for (size_t i = 0; i < sizeof kSmpU / sizeof kSmpU[0]; i++, n++) {
            NSInteger x = SmpAddress(s, kSmpU[i], kSmpSize), y = SmpAddress(t, kSmpV[j], kSmpSize);
            if (x < 0 || y < 0) memcpy(expected + 4 * n, outside, 16);
            else SmpTexel(x, y, expected + 4 * n);
        }
}

static void SmpRecordModes(WMCase *c, NSString *name, id<MTLTexture> texture, MTLSamplerDescriptor *d,
                           const float outside[4]) {
    float coords[SMP_COUNT * 2], expected[SMP_COUNT * 4];
    SmpCoordinates(coords);
    SmpExpected(d.sAddressMode, d.tAddressMode, outside, expected);
    NSData *samples = SmpRun(c, @"smp_sample_2d", texture, [c sampler:d], coords, 8, SMP_COUNT, 1, 16);
    [c data:name elem:WMElemF32 bytes:samples.bytes length:samples.length rule:WMExact()];
    [c value:[name stringByAppendingString:@".match-header-description"]
     boolean:memcmp(samples.bytes, expected, sizeof expected) == 0 rule:WMExact()];
}

WMREF_CASE(smpAddressModes, "sampler.address-modes", "sampler.address-modes",
           "clampToEdge, repeat, mirrorRepeat and clampToZero with nearest filtering at texel centres")
{
    id<MTLTexture> t = SmpRGBA32Texture(c);
    float coords[SMP_COUNT * 2];
    SmpCoordinates(coords);
    [c data:@"coordinates" elem:WMElemF32 bytes:coords length:sizeof coords rule:WMExact()];
    // MTLSampler.h: clampToZero returns (0, 0, 0, 0) for formats with alpha.
    const float zero[4] = {0, 0, 0, 0};
    const MTLSamplerAddressMode modes[] = {MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeRepeat,
                                           MTLSamplerAddressModeMirrorRepeat, MTLSamplerAddressModeClampToZero};
    NSString *names[] = {@"clamp-to-edge", @"repeat", @"mirror-repeat", @"clamp-to-zero"};
    for (int i = 0; i < 4; i++) SmpRecordModes(c, names[i], t, SmpNearest(modes[i], modes[i]), zero);
    // Different modes per axis.
    SmpRecordModes(c, @"repeat.mirror-repeat", t,
                   SmpNearest(MTLSamplerAddressModeRepeat, MTLSamplerAddressModeMirrorRepeat), zero);
    SmpRecordModes(c, @"clamp-to-edge.repeat", t,
                   SmpNearest(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeRepeat), zero);
    SmpRecordModes(c, @"mirror-repeat.clamp-to-edge", t,
                   SmpNearest(MTLSamplerAddressModeMirrorRepeat, MTLSamplerAddressModeClampToEdge), zero);
}

WMREF_CASE(smpMirrorClampToEdge, "sampler.address-mode-mirror-clamp-to-edge", "sampler.address-modes",
           "mirrorClampToEdge with nearest filtering at texel centres")
{
    id<MTLTexture> t = SmpRGBA32Texture(c);
    const float zero[4] = {0, 0, 0, 0};
    SmpRecordModes(c, @"mirror-clamp-to-edge", t,
                   SmpNearest(MTLSamplerAddressModeMirrorClampToEdge, MTLSamplerAddressModeMirrorClampToEdge), zero);
}

WMREF_CASE(smpAddressMode3D, "sampler.address-mode-r", "sampler.address-modes",
           "rAddressMode on a 3D texture: repeat and clampToEdge along depth")
{
    // 4x4x4 RGBA32Float, texel = (x, y, z, 10 * z + 1).
    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = MTLTextureType3D;
    desc.pixelFormat = MTLPixelFormatRGBA32Float;
    desc.width = desc.height = desc.depth = 4;
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [c texture:desc];
    float texels[4 * 4 * 4 * 4];
    for (int z = 0; z < 4; z++)
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                float *p = texels + ((z * 4 + y) * 4 + x) * 4;
                p[0] = x;
                p[1] = y;
                p[2] = z;
                p[3] = 10 * z + 1;
            }
    id<MTLBuffer> source = [c bufferWithBytes:texels length:sizeof texels];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:64 sourceBytesPerImage:256
               sourceSize:MTLSizeMake(4, 4, 4) toTexture:t destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    }];
    const NSInteger ks[] = {-5, -1, 0, 3, 4, 6, 9};
    float coords[7 * 4];
    for (int i = 0; i < 7; i++) {
        coords[4 * i] = 1.5f / 4;
        coords[4 * i + 1] = 2.5f / 4;
        coords[4 * i + 2] = (ks[i] + 0.5f) / 4;
        coords[4 * i + 3] = 0;
    }
    [c data:@"coordinates" elem:WMElemF32 bytes:coords length:sizeof coords rule:WMExact()];
    const MTLSamplerAddressMode modes[] = {MTLSamplerAddressModeRepeat, MTLSamplerAddressModeClampToEdge};
    NSString *names[] = {@"r-repeat", @"r-clamp-to-edge"};
    for (int m = 0; m < 2; m++) {
        MTLSamplerDescriptor *d = SmpNearest(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
        d.rAddressMode = modes[m];
        NSData *samples = SmpRun(c, @"smp_sample_3d", t, [c sampler:d], coords, 16, 7, 1, 16);
        float expected[7 * 4];
        for (int i = 0; i < 7; i++) {
            NSInteger z = SmpAddress(modes[m], ks[i], 4);
            expected[4 * i] = 1;
            expected[4 * i + 1] = 2;
            expected[4 * i + 2] = z;
            expected[4 * i + 3] = 10 * z + 1;
        }
        [c data:names[m] elem:WMElemF32 bytes:samples.bytes length:samples.length rule:WMExact()];
        [c value:[names[m] stringByAppendingString:@".match-header-description"]
         boolean:memcmp(samples.bytes, expected, sizeof expected) == 0 rule:WMExact()];
    }
}

WMREF_CASE(smpBorderColors, "sampler.border-colors", "sampler.address-modes",
           "clampToBorderColor with each border color on formats with and without alpha, API and MSL samplers")
{
    // Metal feature set tables: border color from Apple7.
    [c require:@"family.apple7" value:[c.device supportsFamily:MTLGPUFamilyApple7]];
    id<MTLTexture> rgba = SmpRGBA32Texture(c);
    // MTLSampler.h: transparent black {0,0,0,0}, opaque black {0,0,0,1},
    // opaque white {1,1,1,1}.
    const MTLSamplerBorderColor colors[] = {MTLSamplerBorderColorTransparentBlack, MTLSamplerBorderColorOpaqueBlack,
                                            MTLSamplerBorderColorOpaqueWhite};
    const float values[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 1}};
    NSString *names[] = {@"transparent-black", @"opaque-black", @"opaque-white"};
    for (int i = 0; i < 3; i++) {
        MTLSamplerDescriptor *d = SmpNearest(MTLSamplerAddressModeClampToBorderColor,
                                             MTLSamplerAddressModeClampToBorderColor);
        d.borderColor = colors[i];
        SmpRecordModes(c, [@"rgba32float." stringByAppendingString:names[i]], rgba, d, values[i]);
    }
    // R32Float has only a red component. Outside the texture the red
    // component takes the border color (MSL 4.1 section 2.10); the other
    // components read as the defaults for components a format does not have,
    // 0 and alpha 1 (MSL 4.1 section 6.13).
    MTLTextureDescriptor *rd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                 width:4 height:4 mipmapped:NO];
    rd.storageMode = MTLStorageModePrivate;
    rd.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> red = [c texture:rd];
    float reds[16];
    for (int i = 0; i < 16; i++) reds[i] = 2 + i;
    id<MTLBuffer> source = [c bufferWithBytes:reds length:sizeof reds];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:16 sourceBytesPerImage:0
               sourceSize:MTLSizeMake(4, 4, 1) toTexture:red destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    }];
    // Inside, then outside on each side.
    const float coords[] = {0.375f, 0.625f, -0.125f, 0.375f, 1.125f, 0.375f, 0.375f, -0.375f, 0.375f, 1.875f};
    for (int i = 0; i < 3; i++) {
        MTLSamplerDescriptor *d = SmpNearest(MTLSamplerAddressModeClampToBorderColor,
                                             MTLSamplerAddressModeClampToBorderColor);
        d.borderColor = colors[i];
        NSData *samples = SmpRun(c, @"smp_sample_2d", red, [c sampler:d], coords, 8, 5, 1, 16);
        [c data:[@"r32float." stringByAppendingString:names[i]] elem:WMElemF32 bytes:samples.bytes
         length:samples.length rule:WMExact()];
    }
    // The same three border colors declared in MSL.
    float grid[SMP_COUNT * 2];
    SmpCoordinates(grid);
    NSData *constexprSamples = SmpRun(c, @"smp_constexpr_border", rgba, nil, grid, 8, SMP_COUNT, 3, 16);
    [c data:@"rgba32float.constexpr-border" elem:WMElemF32 bytes:constexprSamples.bytes
     length:constexprSamples.length rule:WMExact()];
}

WMREF_CASE(smpUnnormalized, "sampler.unnormalized-coordinates", "sampler.coordinates",
           "normalizedCoordinates = NO with clampToEdge, and MSL coord::pixel samplers")
{
    id<MTLTexture> t = SmpRGBA32Texture(c);
    // MTLSampler.h: unnormalized coordinates are for 1D and 2D textures with
    // clampToEdge; MSL 4.1 section 2.10 also requires equal min and mag
    // filters and no mip filter.
    MTLSamplerDescriptor *d = SmpNearest(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
    d.normalizedCoordinates = NO;
    d.mipFilter = MTLSamplerMipFilterNotMipmapped;
    float coords[SMP_COUNT * 2], expected[SMP_COUNT * 4];
    NSUInteger n = 0;
    for (size_t j = 0; j < sizeof kSmpV / sizeof kSmpV[0]; j++)
        for (size_t i = 0; i < sizeof kSmpU / sizeof kSmpU[0]; i++, n++) {
            coords[2 * n] = kSmpU[i] + 0.5f;
            coords[2 * n + 1] = kSmpV[j] + 0.5f;
            SmpTexel(SmpAddress(MTLSamplerAddressModeClampToEdge, kSmpU[i], kSmpSize),
                     SmpAddress(MTLSamplerAddressModeClampToEdge, kSmpV[j], kSmpSize), expected + 4 * n);
        }
    [c data:@"pixel-coordinates" elem:WMElemF32 bytes:coords length:sizeof coords rule:WMExact()];
    NSData *samples = SmpRun(c, @"smp_sample_2d", t, [c sampler:d], coords, 8, SMP_COUNT, 1, 16);
    [c data:@"api-sampler" elem:WMElemF32 bytes:samples.bytes length:samples.length rule:WMExact()];
    [c value:@"api-sampler.match-header-description" boolean:memcmp(samples.bytes, expected, sizeof expected) == 0
        rule:WMExact()];
    // coord::pixel with clamp_to_edge, then clamp_to_zero.
    NSData *pixel = SmpRun(c, @"smp_constexpr_pixel", t, nil, coords, 8, SMP_COUNT, 2, 16);
    [c data:@"constexpr-pixel" elem:WMElemF32 bytes:pixel.bytes length:pixel.length rule:WMExact()];
}

// 8x8 R32Float texture with four levels; every texel of level L holds
// 10 * (L + 1).
static id<MTLTexture> SmpLevelTexture(WMCase *c, MTLTextureUsage extraUsage) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                   width:8 height:8 mipmapped:YES];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead | extraUsage;
    id<MTLTexture> t = [c texture:desc];
    for (NSUInteger level = 0; level < t.mipmapLevelCount; level++) {
        NSUInteger w = MAX((NSUInteger)1, (NSUInteger)8 >> level);
        float texels[64];
        for (int i = 0; i < 64; i++) texels[i] = 10.0f * (level + 1);
        id<MTLBuffer> source = [c bufferWithBytes:texels length:w * w * 4];
        [c blit:^(id<MTLBlitCommandEncoder> e) {
            [e copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:w * 4 sourceBytesPerImage:0
                   sourceSize:MTLSizeMake(w, w, 1) toTexture:t destinationSlice:0 destinationLevel:level
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        }];
    }
    return t;
}

// Red channel of each float4 result.
static NSData *SmpRed(NSData *samples) {
    NSUInteger n = samples.length / 16;
    NSMutableData *red = [NSMutableData dataWithLength:n * 4];
    for (NSUInteger i = 0; i < n; i++) memcpy((uint8_t *)red.mutableBytes + 4 * i, (const uint8_t *)samples.bytes + 16 * i, 4);
    return red;
}

WMREF_CASE(smpLodSelection, "sampler.lod-selection", "sampler.lod",
           "Mip level selection: level(), bias(), gradients, sampler LOD clamps and mip filters")
{
    id<MTLTexture> t = SmpLevelTexture(c, 0);
    [c value:@"mipmap-level-count" u64:t.mipmapLevelCount rule:WMExact()];
    MTLSamplerDescriptor *nearest = SmpNearest(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
    nearest.mipFilter = MTLSamplerMipFilterNearest;

    // Explicit levels avoid the halfway points, where "the nearest mipmap
    // level" (MTLSampler.h) is not unique; values beyond the chain select
    // the last level, and negative ones level 0.
    const float levels[] = {0, 0.25f, 0.75f, 1, 1.25f, 1.75f, 2, 2.25f, 3, 3.4f, 7, -2};
    NSUInteger nl = sizeof levels / sizeof levels[0];
    float lq[12 * 4];
    for (NSUInteger i = 0; i < nl; i++) {
        lq[4 * i] = 0.3f;
        lq[4 * i + 1] = 0.6f;
        lq[4 * i + 2] = levels[i];
        lq[4 * i + 3] = 0;
    }
    [c data:@"explicit-levels" elem:WMElemF32 bytes:levels length:sizeof levels rule:WMExact()];
    NSData *r = SmpRun(c, @"smp_sample_2d_level", t, [c sampler:nearest], lq, 16, nl, 1, 16);
    NSData *red = SmpRed(r);
    [c data:@"level.mip-nearest" elem:WMElemF32 bytes:red.bytes length:red.length rule:WMExact()];

    // MTLSampler.h: a sampler that is not mipmapped reads level 0 only.
    MTLSamplerDescriptor *flat = SmpNearest(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
    red = SmpRed(SmpRun(c, @"smp_sample_2d_level", t, [c sampler:flat], lq, 16, nl, 1, 16));
    [c data:@"level.not-mipmapped" elem:WMElemF32 bytes:red.bytes length:red.length rule:WMExact()];

    // Sampler LOD clamps with explicit levels. MTLSampler.h says clamps are
    // ignored by sample variants with an explicit level of detail; MSL 4.1
    // section 6.13 says the mipmap selection is clamped to the sampler's LOD
    // range. Real Metal clamps, as MSL states, so the results are compared
    // exactly. The range [2, 2.25] has no halfway point between levels.
    MTLSamplerDescriptor *clamped = [nearest copy];
    clamped.lodMinClamp = 2;
    clamped.lodMaxClamp = 2.25f;
    red = SmpRed(SmpRun(c, @"smp_sample_2d_level", t, [c sampler:clamped], lq, 16, nl, 1, 16));
    [c data:@"level.sampler-clamp-2-2.25" elem:WMElemF32 bytes:red.bytes length:red.length rule:WMExact()];

    // Without an explicit level a compute function samples at LOD 0 (MSL 4.1
    // section 6.13); bias adds to it; the sampler clamps then apply.
    const float biases[] = {0, 0.25f, 0.75f, 1, 2, 2.75f, 9, -3};
    NSUInteger nb = sizeof biases / sizeof biases[0];
    float bq[8 * 4];
    for (NSUInteger i = 0; i < nb; i++) {
        bq[4 * i] = 0.3f;
        bq[4 * i + 1] = 0.6f;
        bq[4 * i + 2] = biases[i];
        bq[4 * i + 3] = 0;
    }
    [c data:@"biases" elem:WMElemF32 bytes:biases length:sizeof biases rule:WMExact()];
    red = SmpRed(SmpRun(c, @"smp_sample_2d_bias", t, [c sampler:nearest], bq, 16, nb, 1, 16));
    [c data:@"bias.mip-nearest" elem:WMElemF32 bytes:red.bytes length:red.length rule:WMExact()];
    MTLSamplerDescriptor *maxClamp = [nearest copy];
    maxClamp.lodMaxClamp = 1;
    red = SmpRed(SmpRun(c, @"smp_sample_2d_bias", t, [c sampler:maxClamp], bq, 16, nb, 1, 16));
    [c data:@"bias.sampler-max-clamp-1" elem:WMElemF32 bytes:red.bytes length:red.length rule:WMExact()];
    MTLSamplerDescriptor *minClamp = [nearest copy];
    minClamp.lodMinClamp = 2;
    red = SmpRed(SmpRun(c, @"smp_sample_2d_bias", t, [c sampler:minClamp], bq, 16, nb, 1, 16));
    [c data:@"bias.sampler-min-clamp-2" elem:WMElemF32 bytes:red.bytes length:red.length rule:WMExact()];

    // Explicit gradients: the LOD computed from a footprint is not specified
    // to a precision, so these results are reference data.
    const float gradients[] = {0.5f, 0.5f, 0, 0, 1.0f / 8, 0, 0, 1.0f / 8,
                               0.5f, 0.5f, 0, 0, 2.0f / 8, 0, 0, 2.0f / 8,
                               0.5f, 0.5f, 0, 0, 8.0f / 8, 0, 0, 8.0f / 8};
    red = SmpRed(SmpRun(c, @"smp_sample_2d_gradient", t, [c sampler:nearest], gradients, 32, 3, 1, 16));
    [c data:@"gradient.mip-nearest" elem:WMElemF32 bytes:red.bytes length:red.length
       rule:WMNone(@"Metal does not state the precision of the LOD computed from explicit gradients")];
}

WMREF_CASE(smpBindingLodClamps, "sampler.binding-lod-clamps", "sampler.lod",
           "LOD clamps given when binding a sampler to compute and fragment functions")
{
    id<MTLTexture> t = SmpLevelTexture(c, 0);
    MTLSamplerDescriptor *d = SmpNearest(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
    d.mipFilter = MTLSamplerMipFilterNearest;
    id<MTLSamplerState> s = [c sampler:d];
    const float coord[4] = {0.3f, 0.6f, 0, 0};
    const float clamps[][2] = {{0, FLT_MAX}, {1, FLT_MAX}, {2.25f, 3}, {0, 0}};
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"smp_sample_2d_bias"];
    for (int i = 0; i < 4; i++) {
        float minClamp = clamps[i][0], maxClamp = clamps[i][1];
        // Compute: LOD 0 plus a bias of 0 (first query) or 3 (second).
        float q[8];
        memcpy(q, coord, 16);
        memcpy(q + 4, coord, 16);
        q[6] = 3;
        id<MTLBuffer> in = [c bufferWithBytes:q length:sizeof q];
        id<MTLBuffer> out = [c buffer:32];
        [c compute:^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pipeline];
            [e setTexture:t atIndex:0];
            [e setSamplerState:s lodMinClamp:minClamp lodMaxClamp:maxClamp atIndex:0];
            [e setBuffer:in offset:0 atIndex:0];
            [e setBuffer:out offset:0 atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(2, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }];
        float red[2] = {((float *)out.contents)[0], ((float *)out.contents)[4]};
        [c data:[NSString stringWithFormat:@"compute.clamp-%g-%g", clamps[i][0], clamps[i][1]] elem:WMElemF32
          bytes:red length:sizeof red rule:WMExact()];
    }

    // Fragment: one texel per pixel at level 0 gives an implicit LOD of 0, so
    // level 0 unless a clamp moves it; with a bias of 3, level 3 unless the
    // maximum clamp lowers it.
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                 width:8 height:8 mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> target = [c texture:td];
    NSString *fragments[] = {@"smp_implicit_fragment", @"smp_bias_fragment"};
    for (int f = 0; f < 2; f++) {
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [c function:@"smp_fullscreen_vertex"];
        pd.fragmentFunction = [c function:fragments[f]];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatR32Float;
        id<MTLRenderPipelineState> rp = [c renderPipeline:pd];
        for (int i = 0; i < 4; i++) {
            float minClamp = clamps[i][0], maxClamp = clamps[i][1];
            MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture = target;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
                [e setRenderPipelineState:rp];
                [e setFragmentTexture:t atIndex:0];
                [e setFragmentSamplerState:s lodMinClamp:minClamp lodMaxClamp:maxClamp atIndex:0];
                [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            }];
            NSData *pixels = [c readTexture:target level:0 slice:0 bytesPerRow:32];
            [c data:[NSString stringWithFormat:@"fragment.%@.clamp-%g-%g", f ? @"bias-3" : @"implicit", clamps[i][0],
                     clamps[i][1]]
               elem:WMElemF32 bytes:pixels.bytes length:pixels.length rule:WMExact()];
        }
    }
}

// 4x1 Depth32Float texture holding 0.25, 0.5, 0.75, 1.0.
static id<MTLTexture> SmpDepthTexture(WMCase *c) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                   width:4 height:1 mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [c texture:desc];
    const float depths[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    id<MTLBuffer> source = [c bufferWithBytes:depths length:sizeof depths];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:16 sourceBytesPerImage:0
               sourceSize:MTLSizeMake(4, 1, 1) toTexture:t destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    }];
    return t;
}

// Every texel centre against references below, equal to, between and
// above the stored depths.
static NSUInteger SmpCompareQueries(float *q) {
    const float references[] = {0, 0.25f, 0.5f, 0.6f, 1.0f};
    NSUInteger n = 0;
    for (int r = 0; r < 5; r++)
        for (int x = 0; x < 4; x++, n++) {
            q[4 * n] = (x + 0.5f) / 4;
            q[4 * n + 1] = 0.5f;
            q[4 * n + 2] = references[r];
            q[4 * n + 3] = 0;
        }
    return n;
}

WMREF_CASE(smpCompareFunctions, "sampler.compare-functions", "sampler.compare",
           "sample_compare and gather_compare on Depth32Float with each MTLCompareFunction")
{
    // Metal feature set tables: MTLSamplerState comparison functions from
    // Apple3.
    [c require:@"family.apple3" value:[c.device supportsFamily:MTLGPUFamilyApple3]];
    id<MTLTexture> t = SmpDepthTexture(c);
    float q[20 * 4];
    NSUInteger n = SmpCompareQueries(q);
    [c data:@"queries" elem:WMElemF32 bytes:q length:sizeof q rule:WMExact()];
    NSString *names[] = {@"never", @"less", @"equal", @"less-equal", @"greater", @"not-equal", @"greater-equal",
                         @"always"};
    for (NSUInteger f = MTLCompareFunctionNever; f <= MTLCompareFunctionAlways; f++) {
        MTLSamplerDescriptor *d = SmpNearest(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
        d.compareFunction = (MTLCompareFunction)f;
        id<MTLSamplerState> s = [c sampler:d];
        // With nearest filtering each result is the 0 or 1 of one comparison.
        NSData *r = SmpRun(c, @"smp_compare", t, s, q, 16, n, 1, 4);
        [c data:[@"sample-compare." stringByAppendingString:names[f]] elem:WMElemF32 bytes:r.bytes length:r.length
           rule:WMExact()];
        // gather_compare at the corner between texels 1 and 2: four
        // comparisons, ordered as for gather (MSL 4.1 section 6.13).
        float gq[5 * 4];
        for (int i = 0; i < 5; i++) {
            gq[4 * i] = 0.5f;
            gq[4 * i + 1] = 0.5f;
            gq[4 * i + 2] = q[4 * (4 * i) + 2];
            gq[4 * i + 3] = 0;
        }
        NSData *g = SmpRun(c, @"smp_gather_compare", t, s, gq, 16, 5, 1, 16);
        [c data:[@"gather-compare." stringByAppendingString:names[f]] elem:WMElemF32 bytes:g.bytes length:g.length
           rule:WMExact()];
    }
}

WMREF_CASE(smpGather, "sampler.gather", "sampler.gather",
           "gather of each component at texel corners, with and without a texel offset")
{
    // 4x4 RGBA16Float, component c of texel (x, y) = x + 4y + 16c, exact in
    // binary16. At a texel corner the four samples gather selects are the
    // nearest samples at half-texel offsets (-,+), (+,+), (+,-), (-,-)
    // (MSL 4.1 section 6.13), so each component is one stored value.
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                   width:4 height:4 mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [c texture:desc];
    uint16_t texels[64];
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            for (int k = 0; k < 4; k++) texels[(y * 4 + x) * 4 + k] = WMHalfFromFloat(x + 4 * y + 16 * k);
    id<MTLBuffer> source = [c bufferWithBytes:texels length:sizeof texels];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:32 sourceBytesPerImage:0
               sourceSize:MTLSizeMake(4, 4, 1) toTexture:t destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    }];
    float coords[25 * 2];
    NSUInteger n = 0;
    for (int j = 0; j <= 4; j++)
        for (int i = 0; i <= 4; i++, n++) {
            coords[2 * n] = i / 4.0f;
            coords[2 * n + 1] = j / 4.0f;
        }
    [c data:@"coordinates" elem:WMElemF32 bytes:coords length:sizeof coords rule:WMExact()];
    NSString *names[] = {@"clamp-to-edge", @"repeat"};
    const MTLSamplerAddressMode modes[] = {MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeRepeat};
    for (int m = 0; m < 2; m++) {
        MTLSamplerDescriptor *d = SmpNearest(modes[m], modes[m]);
        NSData *g = SmpRun(c, @"smp_gather", t, [c sampler:d], coords, 8, n, 5, 16);
        float expected[25 * 5 * 4];
        for (NSUInteger p = 0; p < n; p++) {
            NSInteger i = (NSInteger)(coords[2 * p] * 4), j = (NSInteger)(coords[2 * p + 1] * 4);
            // (-,+), (+,+), (+,-), (-,-) in texel offsets from the corner.
            const NSInteger dx[4] = {-1, 0, 0, -1}, dy[4] = {0, 0, -1, -1};
            for (int k = 0; k < 5; k++) {
                int component = k < 4 ? k : 0;
                NSInteger offset = k < 4 ? 0 : 1;
                for (int s = 0; s < 4; s++) {
                    NSInteger x = SmpAddress(modes[m], i + dx[s] + offset, 4);
                    NSInteger y = SmpAddress(modes[m], j + dy[s], 4);
                    expected[(p * 5 + k) * 4 + s] = x + 4 * y + 16 * component;
                }
            }
        }
        [c data:names[m] elem:WMElemF32 bytes:g.bytes length:g.length rule:WMExact()];
        [c value:[names[m] stringByAppendingString:@".match-specification"]
         boolean:memcmp(g.bytes, expected, sizeof expected) == 0 rule:WMExact()];
    }
}

WMREF_CASE(smpConstexpr, "sampler.constexpr", "sampler.msl",
           "Samplers declared in MSL: defaults, repeat, mirrored_repeat, clamp_to_zero, per-axis, compare")
{
    id<MTLTexture> t = SmpRGBA32Texture(c);
    float coords[SMP_COUNT * 2];
    SmpCoordinates(coords);
    NSData *samples = SmpRun(c, @"smp_constexpr", t, nil, coords, 8, SMP_COUNT, 5, 16);
    // Expected values: MSL 4.1 section 2.10 defaults are clamp_to_edge,
    // nearest and normalized coordinates.
    const float zero[4] = {0, 0, 0, 0};
    float e[5][SMP_COUNT * 4];
    SmpExpected(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge, zero, e[0]);
    SmpExpected(MTLSamplerAddressModeRepeat, MTLSamplerAddressModeRepeat, zero, e[1]);
    SmpExpected(MTLSamplerAddressModeMirrorRepeat, MTLSamplerAddressModeMirrorRepeat, zero, e[2]);
    SmpExpected(MTLSamplerAddressModeClampToZero, MTLSamplerAddressModeClampToZero, zero, e[3]);
    SmpExpected(MTLSamplerAddressModeRepeat, MTLSamplerAddressModeMirrorRepeat, zero, e[4]);
    float expected[SMP_COUNT * 5 * 4];
    for (NSUInteger i = 0; i < SMP_COUNT; i++)
        for (int k = 0; k < 5; k++) memcpy(expected + (i * 5 + k) * 4, e[k] + i * 4, 16);
    [c data:@"samples" elem:WMElemF32 bytes:samples.bytes length:samples.length rule:WMExact()];
    [c value:@"samples.match-specification" boolean:memcmp(samples.bytes, expected, sizeof expected) == 0
        rule:WMExact()];

    id<MTLTexture> depth = SmpDepthTexture(c);
    float q[20 * 4];
    NSUInteger n = SmpCompareQueries(q);
    NSData *compared = SmpRun(c, @"smp_constexpr_compare", depth, nil, q, 16, n, 2, 4);
    [c data:@"compare.less.greater-equal" elem:WMElemF32 bytes:compared.bytes length:compared.length rule:WMExact()];
}

WMREF_CASE(smpFiltering, "sampler.filtering", "sampler.filtering",
           "Linear, linear-mip and anisotropic filtering results (reference data)")
{
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                   width:4 height:4 mipmapped:YES];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [c texture:desc];
    for (NSUInteger level = 0; level < t.mipmapLevelCount; level++) {
        NSUInteger w = MAX((NSUInteger)1, (NSUInteger)4 >> level);
        uint16_t texels[64];
        for (NSUInteger i = 0; i < w * w; i++)
            for (int k = 0; k < 4; k++) texels[i * 4 + k] = WMHalfFromFloat((float)(i + 16 * k + 64 * level));
        id<MTLBuffer> source = [c bufferWithBytes:texels length:w * w * 8];
        [c blit:^(id<MTLBlitCommandEncoder> e) {
            [e copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:w * 8 sourceBytesPerImage:0
                   sourceSize:MTLSizeMake(w, w, 1) toTexture:t destinationSlice:0 destinationLevel:level
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        }];
    }
    NSString *reason = @"Metal states no precision for filter weights or the anisotropic footprint";
    MTLSamplerDescriptor *linear = SmpNearest(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
    linear.minFilter = linear.magFilter = MTLSamplerMinMagFilterLinear;
    linear.mipFilter = MTLSamplerMipFilterLinear;
    // Texel centres, half-texel and quarter-texel positions.
    const float coords[] = {0.125f, 0.125f, 0.25f, 0.125f, 0.25f, 0.25f, 0.1875f, 0.3125f, 0.5f, 0.5f,
                            0.9f, 0.05f, 1.0f, 1.0f};
    NSData *bilinear = SmpRun(c, @"smp_sample_2d", t, [c sampler:linear], coords, 8, 7, 1, 16);
    [c data:@"bilinear" elem:WMElemF32 bytes:bilinear.bytes length:bilinear.length rule:WMNone(reason)];
    const float lq[] = {0.3f, 0.3f, 0.5f, 0, 0.3f, 0.3f, 1.25f, 0, 0.6f, 0.7f, 1.5f, 0};
    NSData *trilinear = SmpRun(c, @"smp_sample_2d_level", t, [c sampler:linear], lq, 16, 3, 1, 16);
    [c data:@"linear-mip" elem:WMElemF32 bytes:trilinear.bytes length:trilinear.length rule:WMNone(reason)];
    MTLSamplerDescriptor *aniso = [linear copy];
    aniso.maxAnisotropy = 16;
    const float gq[] = {0.5f, 0.5f, 0, 0, 1.0f, 0, 0, 1.0f / 16, 0.5f, 0.5f, 0, 0, 0.25f, 0.25f, -0.0625f, 0.0625f};
    NSData *anisotropic = SmpRun(c, @"smp_sample_2d_gradient", t, [c sampler:aniso], gq, 32, 2, 1, 16);
    [c data:@"anisotropic-16" elem:WMElemF32 bytes:anisotropic.bytes length:anisotropic.length rule:WMNone(reason)];
}

WMREF_CASE(smpProperties, "sampler.properties", "sampler.properties",
           "MTLSamplerDescriptor defaults, label and argument-buffer support on created samplers")
{
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    // Defaults stated in MTLSampler.h.
    [c value:@"default.min-filter" u64:d.minFilter rule:WMExact()];
    [c value:@"default.mag-filter" u64:d.magFilter rule:WMExact()];
    [c value:@"default.mip-filter" u64:d.mipFilter rule:WMExact()];
    [c value:@"default.max-anisotropy" u64:d.maxAnisotropy rule:WMExact()];
    [c value:@"default.s-address-mode" u64:d.sAddressMode rule:WMExact()];
    [c value:@"default.t-address-mode" u64:d.tAddressMode rule:WMExact()];
    [c value:@"default.r-address-mode" u64:d.rAddressMode rule:WMExact()];
    [c value:@"default.normalized-coordinates" boolean:d.normalizedCoordinates rule:WMExact()];
    [c value:@"default.lod-min-clamp" f32:d.lodMinClamp rule:WMExact()];
    [c value:@"default.lod-max-clamp" f32:d.lodMaxClamp rule:WMExact()];
    [c value:@"default.compare-function" u64:d.compareFunction rule:WMExact()];
    [c value:@"default.lod-average" boolean:d.lodAverage rule:WMExact()];
    [c value:@"default.label-is-nil" boolean:d.label == nil rule:WMExact()];
    // Defaults MTLSampler.h does not state.
    [c value:@"default.border-color" u64:d.borderColor rule:WMDevice()];
    [c value:@"default.support-argument-buffers" boolean:d.supportArgumentBuffers rule:WMDevice()];

    d.label = @"reference sampler";
    id<MTLSamplerState> labelled = [c sampler:d];
    [c value:@"state.label" string:labelled.label rule:WMExact()];
    [c value:@"state.device-is-device" boolean:labelled.device == c.device rule:WMExact()];
    MTLSamplerDescriptor *argument = [MTLSamplerDescriptor new];
    argument.supportArgumentBuffers = YES;
    id<MTLSamplerState> forArguments = [c.device newSamplerStateWithDescriptor:argument];
    [c value:@"support-argument-buffers.created" boolean:forArguments != nil rule:WMExact()];
    // The descriptor is copied at creation: later changes do not affect the
    // created sampler's label.
    d.label = @"changed";
    [c value:@"state.label-after-descriptor-change" string:labelled.label rule:WMExact()];
}

WMREF_CASE(smpMinLodClampFragment, "sampler.min-lod-clamp", "sampler.lod",
           "min_lod_clamp() in a fragment function over an implicit LOD of 0")
{
    // One texel per pixel at level 0 gives an implicit LOD of 0; the shader
    // clamp raises it to the given minimum, and nearest mip filtering
    // selects the nearest level (MSL 4.1 section 6.13, MTLSampler.h).
    id<MTLTexture> t = SmpLevelTexture(c, 0);
    MTLSamplerDescriptor *d = SmpNearest(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
    d.mipFilter = MTLSamplerMipFilterNearest;
    id<MTLSamplerState> s = [c sampler:d];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                 width:8 height:8 mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> target = [c texture:td];
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [c function:@"smp_fullscreen_vertex"];
    pd.fragmentFunction = [c function:@"smp_min_lod_clamp_fragment"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatR32Float;
    id<MTLRenderPipelineState> rp = [c renderPipeline:pd];
    const float minimums[] = {0, 0.75f, 1, 2.25f, 3, 6};
    for (int i = 0; i < 6; i++) {
        float minimum = minimums[i];
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:rp];
            [e setFragmentTexture:t atIndex:0];
            [e setFragmentSamplerState:s atIndex:0];
            [e setFragmentBytes:&minimum length:sizeof minimum atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }];
        NSData *pixels = [c readTexture:target level:0 slice:0 bytesPerRow:32];
        [c data:[NSString stringWithFormat:@"minimum-%g", minimum] elem:WMElemF32 bytes:pixels.bytes
         length:pixels.length rule:WMExact()];
    }
}

#pragma mark - Invalid sampler descriptors

// Without the debug layer Metal does not define the result of an invalid
// sampler descriptor; with it, the case outcome records the validation
// failure.
static void SmpRecordCreation(WMCase *c, MTLSamplerDescriptor *d) {
    __block id<MTLSamplerState> state = nil;
    [c step:@"newSamplerStateWithDescriptor"];
    [c call:@"create" rule:WMExact() block:^{
        state = [c.device newSamplerStateWithDescriptor:d];
    }];
    [c value:@"returned-nil" boolean:state == nil
        rule:WMNone(@"Metal does not define the result of an invalid creation call without its debug layer")];
}

WMREF_CASE(smpErrorUnnormalizedRepeat, "sampler.error-unnormalized-repeat", "sampler.errors",
           "Unnormalized coordinates with the repeat address mode (MTLSampler.h: results undefined)")
{
    MTLSamplerDescriptor *d = SmpNearest(MTLSamplerAddressModeRepeat, MTLSamplerAddressModeRepeat);
    d.normalizedCoordinates = NO;
    SmpRecordCreation(c, d);
}

WMREF_CASE(smpErrorUnnormalizedMip, "sampler.error-unnormalized-mip-filter", "sampler.errors",
           "Unnormalized coordinates with a mip filter (MSL 4.1 section 2.10: mip_filter must be none)")
{
    MTLSamplerDescriptor *d = SmpNearest(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
    d.normalizedCoordinates = NO;
    d.mipFilter = MTLSamplerMipFilterNearest;
    SmpRecordCreation(c, d);
}

WMREF_CASE(smpErrorUnnormalizedFilters, "sampler.error-unnormalized-different-filters", "sampler.errors",
           "Unnormalized coordinates with different min and mag filters (MSL 4.1 section 2.10)")
{
    MTLSamplerDescriptor *d = SmpNearest(MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
    d.normalizedCoordinates = NO;
    d.minFilter = MTLSamplerMinMagFilterLinear;
    SmpRecordCreation(c, d);
}

WMREF_CASE(smpErrorAnisotropy, "sampler.error-anisotropy-zero", "sampler.errors",
           "maxAnisotropy of 0 (the default and minimum are 1)")
{
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.maxAnisotropy = 0;
    SmpRecordCreation(c, d);
}

WMREF_CASE(smpErrorLodRange, "sampler.error-lod-min-above-max", "sampler.errors",
           "lodMinClamp greater than lodMaxClamp")
{
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.mipFilter = MTLSamplerMipFilterNearest;
    d.lodMinClamp = 3;
    d.lodMaxClamp = 1;
    SmpRecordCreation(c, d);
}
