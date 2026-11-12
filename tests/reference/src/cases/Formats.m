// Pixel-format cases: uncompressed color formats (upload and readback,
// shader read conversion, shader write conversion, render-target output),
// depth and stencil formats, the 422 formats and the three-channel
// texture-buffer formats.
//
// Conversion rules cite MSL 4.1 (2026-06-04) section 8.7. Per-family
// capabilities (Write, Color) come from the Metal Feature Set Tables
// (2026-10-23), "Texture capabilities by pixel format", whose note states
// that all graphics and compute kernels can read or sample a texture with
// any pixel format.

#import "WMRefCase.h"

#include <math.h>
#include <float.h>

#pragma mark - Format descriptions

typedef NS_ENUM(uint8_t, FmtType) {
    FmtUnorm,
    FmtSnorm,
    FmtUint,
    FmtSint,
    FmtHalf,
    FmtFloat,
    FmtUF11,  // 6-bit mantissa, 5-bit exponent, no sign (section 8.7.4)
    FmtUF10,  // 5-bit mantissa, 5-bit exponent, no sign (section 8.7.4)
    FmtSRGB,  // 8-bit sRGB-encoded color channel (section 8.7.7)
    FmtXR,    // extended-range 10-bit channel; encoding not stated in the SDK or MSL
    FmtE5,    // 9-bit mantissa of RGB9E5Float; exponent in bits 27-31 (section 8.7.5)
};

typedef struct {
    uint8_t component; // 0 red, 1 green, 2 blue, 3 alpha
    uint8_t offset;    // bit offset in the little-endian texel
    uint8_t bits;
    FmtType type;
} FmtChannel;

typedef struct {
    MTLPixelFormat format;
    uint8_t bytes;
    uint8_t count;
    FmtChannel channels[4];
    // Lowest AppleN family whose Feature Set Tables column lists the format
    // (available), the Write capability and the Color capability; 0 when no
    // family lists the capability.
    uint8_t available, write, color;
} FmtInfo;

enum { R = 0, G = 1, B = 2, A = 3 };

// Bit positions of the packed formats were confirmed on the device by
// reading textures with one bit set per texel; they agree with the format
// names read from the least significant bit.
static const FmtInfo kFormats[] = {
    {MTLPixelFormatA8Unorm, 1, 1, {{A, 0, 8, FmtUnorm}}, 2, 3, 3},
    {MTLPixelFormatR8Unorm, 1, 1, {{R, 0, 8, FmtUnorm}}, 2, 2, 2},
    {MTLPixelFormatR8Unorm_sRGB, 1, 1, {{R, 0, 8, FmtSRGB}}, 2, 2, 2},
    {MTLPixelFormatR8Snorm, 1, 1, {{R, 0, 8, FmtSnorm}}, 2, 2, 2},
    {MTLPixelFormatR8Uint, 1, 1, {{R, 0, 8, FmtUint}}, 2, 2, 2},
    {MTLPixelFormatR8Sint, 1, 1, {{R, 0, 8, FmtSint}}, 2, 2, 2},
    {MTLPixelFormatR16Unorm, 2, 1, {{R, 0, 16, FmtUnorm}}, 2, 2, 2},
    {MTLPixelFormatR16Snorm, 2, 1, {{R, 0, 16, FmtSnorm}}, 2, 2, 2},
    {MTLPixelFormatR16Uint, 2, 1, {{R, 0, 16, FmtUint}}, 2, 2, 2},
    {MTLPixelFormatR16Sint, 2, 1, {{R, 0, 16, FmtSint}}, 2, 2, 2},
    {MTLPixelFormatR16Float, 2, 1, {{R, 0, 16, FmtHalf}}, 2, 2, 2},
    {MTLPixelFormatRG8Unorm, 2, 2, {{R, 0, 8, FmtUnorm}, {G, 8, 8, FmtUnorm}}, 2, 2, 2},
    {MTLPixelFormatRG8Unorm_sRGB, 2, 2, {{R, 0, 8, FmtSRGB}, {G, 8, 8, FmtSRGB}}, 2, 2, 2},
    {MTLPixelFormatRG8Snorm, 2, 2, {{R, 0, 8, FmtSnorm}, {G, 8, 8, FmtSnorm}}, 2, 2, 2},
    {MTLPixelFormatRG8Uint, 2, 2, {{R, 0, 8, FmtUint}, {G, 8, 8, FmtUint}}, 2, 2, 2},
    {MTLPixelFormatRG8Sint, 2, 2, {{R, 0, 8, FmtSint}, {G, 8, 8, FmtSint}}, 2, 2, 2},
    {MTLPixelFormatB5G6R5Unorm, 2, 3, {{B, 0, 5, FmtUnorm}, {G, 5, 6, FmtUnorm}, {R, 11, 5, FmtUnorm}}, 2, 0, 2},
    {MTLPixelFormatA1BGR5Unorm, 2, 4,
     {{A, 0, 1, FmtUnorm}, {B, 1, 5, FmtUnorm}, {G, 6, 5, FmtUnorm}, {R, 11, 5, FmtUnorm}}, 2, 0, 2},
    {MTLPixelFormatABGR4Unorm, 2, 4,
     {{A, 0, 4, FmtUnorm}, {B, 4, 4, FmtUnorm}, {G, 8, 4, FmtUnorm}, {R, 12, 4, FmtUnorm}}, 2, 0, 2},
    {MTLPixelFormatBGR5A1Unorm, 2, 4,
     {{B, 0, 5, FmtUnorm}, {G, 5, 5, FmtUnorm}, {R, 10, 5, FmtUnorm}, {A, 15, 1, FmtUnorm}}, 2, 0, 2},
    {MTLPixelFormatR32Uint, 4, 1, {{R, 0, 32, FmtUint}}, 2, 2, 2},
    {MTLPixelFormatR32Sint, 4, 1, {{R, 0, 32, FmtSint}}, 2, 2, 2},
    {MTLPixelFormatR32Float, 4, 1, {{R, 0, 32, FmtFloat}}, 2, 2, 2},
    {MTLPixelFormatRG16Unorm, 4, 2, {{R, 0, 16, FmtUnorm}, {G, 16, 16, FmtUnorm}}, 2, 2, 2},
    {MTLPixelFormatRG16Snorm, 4, 2, {{R, 0, 16, FmtSnorm}, {G, 16, 16, FmtSnorm}}, 2, 2, 2},
    {MTLPixelFormatRG16Uint, 4, 2, {{R, 0, 16, FmtUint}, {G, 16, 16, FmtUint}}, 2, 2, 2},
    {MTLPixelFormatRG16Sint, 4, 2, {{R, 0, 16, FmtSint}, {G, 16, 16, FmtSint}}, 2, 2, 2},
    {MTLPixelFormatRG16Float, 4, 2, {{R, 0, 16, FmtHalf}, {G, 16, 16, FmtHalf}}, 2, 2, 2},
    {MTLPixelFormatRGBA8Unorm, 4, 4,
     {{R, 0, 8, FmtUnorm}, {G, 8, 8, FmtUnorm}, {B, 16, 8, FmtUnorm}, {A, 24, 8, FmtUnorm}}, 2, 2, 2},
    {MTLPixelFormatRGBA8Unorm_sRGB, 4, 4,
     {{R, 0, 8, FmtSRGB}, {G, 8, 8, FmtSRGB}, {B, 16, 8, FmtSRGB}, {A, 24, 8, FmtUnorm}}, 2, 2, 2},
    {MTLPixelFormatRGBA8Snorm, 4, 4,
     {{R, 0, 8, FmtSnorm}, {G, 8, 8, FmtSnorm}, {B, 16, 8, FmtSnorm}, {A, 24, 8, FmtSnorm}}, 2, 2, 2},
    {MTLPixelFormatRGBA8Uint, 4, 4,
     {{R, 0, 8, FmtUint}, {G, 8, 8, FmtUint}, {B, 16, 8, FmtUint}, {A, 24, 8, FmtUint}}, 2, 2, 2},
    {MTLPixelFormatRGBA8Sint, 4, 4,
     {{R, 0, 8, FmtSint}, {G, 8, 8, FmtSint}, {B, 16, 8, FmtSint}, {A, 24, 8, FmtSint}}, 2, 2, 2},
    {MTLPixelFormatBGRA8Unorm, 4, 4,
     {{B, 0, 8, FmtUnorm}, {G, 8, 8, FmtUnorm}, {R, 16, 8, FmtUnorm}, {A, 24, 8, FmtUnorm}}, 2, 2, 2},
    {MTLPixelFormatBGRA8Unorm_sRGB, 4, 4,
     {{B, 0, 8, FmtSRGB}, {G, 8, 8, FmtSRGB}, {R, 16, 8, FmtSRGB}, {A, 24, 8, FmtUnorm}}, 2, 2, 2},
    {MTLPixelFormatRGB10A2Unorm, 4, 4,
     {{R, 0, 10, FmtUnorm}, {G, 10, 10, FmtUnorm}, {B, 20, 10, FmtUnorm}, {A, 30, 2, FmtUnorm}}, 2, 3, 2},
    {MTLPixelFormatRGB10A2Uint, 4, 4,
     {{R, 0, 10, FmtUint}, {G, 10, 10, FmtUint}, {B, 20, 10, FmtUint}, {A, 30, 2, FmtUint}}, 2, 3, 2},
    {MTLPixelFormatRG11B10Float, 4, 3, {{R, 0, 11, FmtUF11}, {G, 11, 11, FmtUF11}, {B, 22, 10, FmtUF10}}, 2, 3, 2},
    {MTLPixelFormatRGB9E5Float, 4, 3, {{R, 0, 9, FmtE5}, {G, 9, 9, FmtE5}, {B, 18, 9, FmtE5}}, 2, 3, 2},
    {MTLPixelFormatBGR10A2Unorm, 4, 4,
     {{B, 0, 10, FmtUnorm}, {G, 10, 10, FmtUnorm}, {R, 20, 10, FmtUnorm}, {A, 30, 2, FmtUnorm}}, 2, 2, 2},
    {MTLPixelFormatBGR10_XR, 4, 3, {{B, 0, 10, FmtXR}, {G, 10, 10, FmtXR}, {R, 20, 10, FmtXR}}, 3, 3, 3},
    {MTLPixelFormatBGR10_XR_sRGB, 4, 3, {{B, 0, 10, FmtXR}, {G, 10, 10, FmtXR}, {R, 20, 10, FmtXR}}, 3, 3, 3},
    {MTLPixelFormatRG32Uint, 8, 2, {{R, 0, 32, FmtUint}, {G, 32, 32, FmtUint}}, 2, 2, 2},
    {MTLPixelFormatRG32Sint, 8, 2, {{R, 0, 32, FmtSint}, {G, 32, 32, FmtSint}}, 2, 2, 2},
    {MTLPixelFormatRG32Float, 8, 2, {{R, 0, 32, FmtFloat}, {G, 32, 32, FmtFloat}}, 2, 2, 2},
    {MTLPixelFormatRGBA16Unorm, 8, 4,
     {{R, 0, 16, FmtUnorm}, {G, 16, 16, FmtUnorm}, {B, 32, 16, FmtUnorm}, {A, 48, 16, FmtUnorm}}, 2, 2, 2},
    {MTLPixelFormatRGBA16Snorm, 8, 4,
     {{R, 0, 16, FmtSnorm}, {G, 16, 16, FmtSnorm}, {B, 32, 16, FmtSnorm}, {A, 48, 16, FmtSnorm}}, 2, 2, 2},
    {MTLPixelFormatRGBA16Uint, 8, 4,
     {{R, 0, 16, FmtUint}, {G, 16, 16, FmtUint}, {B, 32, 16, FmtUint}, {A, 48, 16, FmtUint}}, 2, 2, 2},
    {MTLPixelFormatRGBA16Sint, 8, 4,
     {{R, 0, 16, FmtSint}, {G, 16, 16, FmtSint}, {B, 32, 16, FmtSint}, {A, 48, 16, FmtSint}}, 2, 2, 2},
    {MTLPixelFormatRGBA16Float, 8, 4,
     {{R, 0, 16, FmtHalf}, {G, 16, 16, FmtHalf}, {B, 32, 16, FmtHalf}, {A, 48, 16, FmtHalf}}, 2, 2, 2},
    // The 10-bit codes of the 64-bit XR formats occupy the high 10 bits of
    // each 16-bit word (measured on the device; the low 6 bits do not
    // change the decoded value).
    {MTLPixelFormatBGRA10_XR, 8, 4,
     {{B, 6, 10, FmtXR}, {G, 22, 10, FmtXR}, {R, 38, 10, FmtXR}, {A, 54, 10, FmtXR}}, 3, 3, 3},
    {MTLPixelFormatBGRA10_XR_sRGB, 8, 4,
     {{B, 6, 10, FmtXR}, {G, 22, 10, FmtXR}, {R, 38, 10, FmtXR}, {A, 54, 10, FmtXR}}, 3, 3, 3},
    {MTLPixelFormatRGBA32Uint, 16, 4,
     {{R, 0, 32, FmtUint}, {G, 32, 32, FmtUint}, {B, 64, 32, FmtUint}, {A, 96, 32, FmtUint}}, 2, 2, 2},
    {MTLPixelFormatRGBA32Sint, 16, 4,
     {{R, 0, 32, FmtSint}, {G, 32, 32, FmtSint}, {B, 64, 32, FmtSint}, {A, 96, 32, FmtSint}}, 2, 2, 2},
    {MTLPixelFormatRGBA32Float, 16, 4,
     {{R, 0, 32, FmtFloat}, {G, 32, 32, FmtFloat}, {B, 64, 32, FmtFloat}, {A, 96, 32, FmtFloat}}, 2, 2, 2},
};

static const FmtInfo *FmtLookup(WMCase *c, MTLPixelFormat format) {
    const FmtInfo *found = NULL;
    for (size_t i = 0; i < sizeof kFormats / sizeof kFormats[0]; i++)
        if (kFormats[i].format == format) found = &kFormats[i];
    if (!found) [c fail:@"no description for pixel format %lu", (unsigned long)format];
    return found;
}

typedef NS_ENUM(uint8_t, FmtShader) { FmtShaderFloat, FmtShaderUint, FmtShaderInt };

static FmtShader FmtShaderType(const FmtInfo *info) {
    if (info->channels[0].type == FmtUint) return FmtShaderUint;
    if (info->channels[0].type == FmtSint) return FmtShaderInt;
    return FmtShaderFloat;
}

static NSString *FmtSuffix(FmtShader shader) {
    return shader == FmtShaderUint ? @"uint" : shader == FmtShaderInt ? @"int" : @"float";
}

static WMElem FmtElem(FmtShader shader) {
    return shader == FmtShaderUint ? WMElemU32 : shader == FmtShaderInt ? WMElemI32 : WMElemF32;
}

// Highest AppleN family the device reports, or 0.
static int FmtAppleFamily(id<MTLDevice> device) {
    for (int n = 11; n >= 1; n--)
        if ([device supportsFamily:(MTLGPUFamily)(1000 + n)]) return n;
    return 0;
}

static BOOL FmtTableAllows(id<MTLDevice> device, uint8_t family) {
    return family != 0 && FmtAppleFamily(device) >= family;
}

#pragma mark - Bit and number helpers

static uint64_t FmtGetBits(const uint8_t *texel, unsigned offset, unsigned bits) {
    uint64_t value = 0;
    for (unsigned i = 0; i < bits; i++) {
        unsigned bit = offset + i;
        value |= (uint64_t)((texel[bit / 8] >> (bit % 8)) & 1) << i;
    }
    return value;
}

static void FmtSetBits(uint8_t *texel, unsigned offset, unsigned bits, uint64_t value) {
    for (unsigned i = 0; i < bits; i++) {
        unsigned bit = offset + i;
        texel[bit / 8] = (uint8_t)((texel[bit / 8] & ~(1u << (bit % 8))) | (((value >> i) & 1) << (bit % 8)));
    }
}

static int64_t FmtSignExtend(uint64_t code, unsigned bits) {
    if (bits < 64 && (code >> (bits - 1)) & 1) return (int64_t)(code | (~0ull << bits));
    return (int64_t)code;
}

// Section 8.7.4: 10- and 11-bit floats.
static double FmtDecodeSmallFloat(uint64_t code, unsigned mantissaBits) {
    uint64_t mantissa = code & ((1u << mantissaBits) - 1), exponent = code >> mantissaBits;
    double n = (double)(1u << mantissaBits);
    if (exponent == 0) return ldexp(mantissa / n, -14);
    if (exponent == 31) return mantissa ? NAN : INFINITY;
    return ldexp(1.0 + mantissa / n, (int)exponent - 15);
}

// ulp(x) of binary32 as defined in MSL 4.1 section 8.4.
static double FmtUlp32(double x) {
    x = fabs(x);
    if (x == 0) return ldexp(1.0, -149);
    float f = (float)x;
    if ((double)f == x) {
        double below = x - (double)nextafterf(f, 0.0f);
        double above = (double)nextafterf(f, INFINITY) - x;
        return MIN(below, above);
    }
    float lo = (double)f < x ? f : nextafterf(f, 0.0f);
    return (double)nextafterf(lo, INFINITY) - (double)lo;
}

// The inverse of the section 8.7.7 sRGB decode, without rounding.
static double FmtSRGBEncodeExact(double linear) {
    if (linear <= 0.04045 / 12.92) return linear * 12.92 * 255.0;
    return (1.055 * pow(linear, 1.0 / 2.4) - 0.055) * 255.0;
}

// The section 8.7.7 linear-to-sRGB rule for an 8-bit code, before the
// "+ 0.5 and drop the fraction" step: the reference result of the bound.
static double FmtSRGBEncodeReference(float value) {
    double c = value;
    if (isnan(value)) c = 0.0;
    if (c > 1.0) c = 1.0;
    else if (c < 0.0) c = 0.0;
    else if (c < 0.0031308) c = 12.92 * c;
    else c = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    return c * 255.0;
}

#pragma mark - Device helpers

static id<MTLBuffer> FmtRunRead(WMCase *c, id<MTLTexture> texture, NSString *kernel, size_t elementSize) {
    id<MTLComputePipelineState> pipeline = [c computePipeline:kernel];
    NSUInteger w = texture.width, h = texture.height;
    id<MTLBuffer> out = [c buffer:w * h * elementSize];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setTexture:texture atIndex:0];
        [e setBuffer:out offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    }];
    return out;
}

static NSDictionary *FmtLayout(MTLPixelFormat format, NSUInteger width, NSUInteger height, NSUInteger bytesPerRow) {
    return @{@"format": WMPixelFormatName(format), @"width": @(width), @"height": @(height),
             @"bytes_per_row": @(bytesPerRow)};
}

enum { W = 8, H = 4, N = W * H };

#pragma mark - Upload, readback and shader read

// Codes placed in the first texels of every channel: the corner cases of
// table 8.6 for normalized channels, the range ends for integer channels,
// and signed zeros, infinities and the smallest subnormal for floats.
static NSUInteger FmtCorners(const FmtChannel *ch, uint64_t *codes) {
    uint64_t mask = ch->bits == 64 ? ~0ull : (1ull << ch->bits) - 1;
    uint64_t signBit = 1ull << (ch->bits - 1);
    switch (ch->type) {
    case FmtUnorm: case FmtUint: case FmtSRGB: case FmtXR:
        codes[0] = 0; codes[1] = mask; return 2;
    case FmtSnorm: case FmtSint:
        codes[0] = signBit; codes[1] = signBit + 1; codes[2] = 0; codes[3] = mask >> 1; return 4;
    case FmtHalf:
        codes[0] = 0x0000; codes[1] = 0x8000; codes[2] = 0x7c00; codes[3] = 0xfc00; codes[4] = 0x0001;
        codes[5] = 0x3c00; return 6;
    case FmtFloat:
        codes[0] = 0x00000000; codes[1] = 0x80000000; codes[2] = 0x7f800000; codes[3] = 0xff800000;
        codes[4] = 0x3f800000; codes[5] = 0x7f7fffff; return 6;
    case FmtUF11: case FmtUF10: {
        unsigned m = ch->type == FmtUF11 ? 6 : 5;
        codes[0] = 0; codes[1] = (30ull << m) | ((1ull << m) - 1); codes[2] = 1; codes[3] = 31ull << m; return 4;
    }
    case FmtE5:
        codes[0] = 0; codes[1] = 511; codes[2] = 1; return 3;
    }
    return 0;
}

static void FmtFillInput(const FmtInfo *info, uint8_t *texels, uint32_t seed) {
    WMPattern(texels, (size_t)N * info->bytes, seed);
    uint8_t clean[16];
    for (NSUInteger t = 0; t < N; t++) {
        uint8_t *texel = texels + t * info->bytes;
        memset(clean, 0, sizeof clean);
        for (unsigned k = 0; k < info->count; k++) {
            const FmtChannel *ch = &info->channels[k];
            uint64_t code = FmtGetBits(texel, ch->offset, ch->bits);
            uint64_t corners[8];
            NSUInteger cornerCount = FmtCorners(ch, corners);
            if (t < 6 && cornerCount) code = corners[t % cornerCount];
            else if (ch->type == FmtHalf && ((code >> 10) & 31) == 31) code &= ~(1ull << 10);
            else if (ch->type == FmtFloat) {
                uint64_t exponent = (code >> 23) & 0xff;
                // Section 8.7.3 lets reads flush subnormals and alter NaNs;
                // random texels avoid both and the infinities.
                if (exponent == 0xff) code &= ~(1ull << 23);
                if (exponent == 0 && (code & 0x7fffff)) code |= 1ull << 23;
            } else if ((ch->type == FmtUF11 || ch->type == FmtUF10) &&
                       (code >> (ch->type == FmtUF11 ? 6 : 5)) == 31) {
                code &= ~(1ull << (ch->type == FmtUF11 ? 6 : 5));
            }
            FmtSetBits(clean, ch->offset, ch->bits, code);
        }
        if (info->channels[0].type == FmtE5) {
            uint64_t exponent = FmtGetBits(texel, 27, 5);
            if (t < 6) exponent = t == 1 ? 31 : t == 2 ? 0 : t == 0 ? 0 : exponent;
            FmtSetBits(clean, 27, 5, exponent);
        }
        // Bits that belong to no channel are zero.
        memcpy(texel, clean, info->bytes);
    }
}

typedef struct {
    NSMutableData *exact, *normalized, *normalizedIdeal, *srgb, *srgbIdeal, *xr;
    BOOL within;
} FmtReadSplit;

static void FmtAppend(NSMutableData *data, const void *bytes, size_t length) { [data appendBytes:bytes length:length]; }

// Classifies every component that the read kernel returned and checks it
// against the section 8.7 rule that applies.
static FmtReadSplit FmtSplitRead(const FmtInfo *info, const uint8_t *texels, const void *results) {
    FmtReadSplit s = {[NSMutableData data], [NSMutableData data], [NSMutableData data], [NSMutableData data],
                      [NSMutableData data], [NSMutableData data], YES};
    FmtShader shader = FmtShaderType(info);
    for (NSUInteger t = 0; t < N; t++) {
        const uint8_t *texel = texels + t * info->bytes;
        for (unsigned comp = 0; comp < 4; comp++) {
            const uint32_t *word = (const uint32_t *)results + t * 4 + comp;
            float measured;
            memcpy(&measured, word, 4);
            const FmtChannel *ch = NULL;
            for (unsigned k = 0; k < info->count; k++)
                if (info->channels[k].component == comp) ch = &info->channels[k];
            if (!ch) {
                // Unspecified components read as 0, and alpha as 1 (section
                // 6.13 texture function rules; integer formats use the integral
                // value 1).
                if (shader == FmtShaderFloat) {
                    float ideal = comp == 3 ? 1.0f : 0.0f;
                    s.within &= measured == ideal && signbit(measured) == 0;
                } else {
                    s.within &= *word == (comp == 3 ? 1u : 0u);
                }
                FmtAppend(s.exact, word, 4);
                continue;
            }
            uint64_t code = FmtGetBits(texel, ch->offset, ch->bits);
            uint64_t max = (1ull << ch->bits) - 1;
            switch (ch->type) {
            case FmtUnorm: case FmtSnorm: {
                // Table 8.6 conversions; precision <= 1.5 ulp except the
                // listed corner cases, which are exact.
                double ideal;
                BOOL corner;
                if (ch->type == FmtUnorm) {
                    ideal = (double)code / (double)max;
                    corner = code == 0 || code == max;
                } else {
                    int64_t v = FmtSignExtend(code, ch->bits);
                    double scale = (double)((1ll << (ch->bits - 1)) - 1);
                    ideal = MAX(-1.0, v / scale);
                    corner = v == 0 || v == (int64_t)scale || v <= -(int64_t)scale;
                }
                FmtAppend(s.normalized, &measured, 4);
                FmtAppend(s.normalizedIdeal, &ideal, 8);
                s.within &= fabs(measured - ideal) <= 1.5 * FmtUlp32(ideal);
                if (corner) {
                    FmtAppend(s.exact, &measured, 4);
                    s.within &= (double)measured == ideal;
                }
                break;
            }
            case FmtUint:
                // Section 8.7.6: integer reads return the stored value.
                FmtAppend(s.exact, word, 4);
                s.within &= *word == (uint32_t)code;
                break;
            case FmtSint:
                FmtAppend(s.exact, word, 4);
                s.within &= (int32_t)*word == (int32_t)FmtSignExtend(code, ch->bits);
                break;
            case FmtHalf: {
                // Section 8.7.2: half to float is lossless.
                double ideal = WMHalfToDouble((uint16_t)code);
                FmtAppend(s.exact, &measured, 4);
                s.within &= (double)measured == ideal && signbit(measured) == signbit(ideal);
                break;
            }
            case FmtFloat: {
                // Section 8.7.3: values other than NaN and subnormals are
                // preserved; the inputs contain neither.
                FmtAppend(s.exact, &measured, 4);
                s.within &= *word == (uint32_t)code;
                break;
            }
            case FmtUF11: case FmtUF10: {
                // Section 8.7.4: lossless conversion, subnormals preserved.
                double ideal = FmtDecodeSmallFloat(code, ch->type == FmtUF11 ? 6 : 5);
                FmtAppend(s.exact, &measured, 4);
                s.within &= (double)measured == ideal;
                break;
            }
            case FmtE5: {
                // Section 8.7.5: lossless, 2^(exponent - 15) * (mantissa / 512).
                double ideal = ldexp(code / 512.0, (int)FmtGetBits(texel, 27, 5) - 15);
                FmtAppend(s.exact, &measured, 4);
                s.within &= (double)measured == ideal;
                break;
            }
            case FmtSRGB: {
                // Section 8.7.7: re-encoding the result without rounding
                // lands within 0.5 of the stored code.
                double ideal = (double)code;
                FmtAppend(s.srgb, &measured, 4);
                FmtAppend(s.srgbIdeal, &ideal, 8);
                s.within &= fabs(FmtSRGBEncodeExact(measured) - ideal) <= 0.5;
                break;
            }
            case FmtXR:
                FmtAppend(s.xr, &measured, 4);
                break;
            }
        }
    }
    return s;
}

static void FmtRecordRead(WMCase *c, const FmtInfo *info, const uint8_t *texels, id<MTLBuffer> results) {
    FmtShader shader = FmtShaderType(info);
    FmtReadSplit s = FmtSplitRead(info, texels, results.contents);
    [c data:@"read.exact" elem:FmtElem(shader) bytes:s.exact.bytes length:s.exact.length rule:WMExact()];
    if (s.normalized.length)
        [c bounded:@"read.normalized" elem:WMElemF32 bytes:s.normalized.bytes count:s.normalized.length / 4
             ideal:s.normalizedIdeal.bytes model:@"ulp" bound:@"1.5"
         reference:@"MSL 4.1 section 8.7.1.1, table 8.6"];
    if (s.srgb.length)
        [c bounded:@"read.srgb" elem:WMElemF32 bytes:s.srgb.bytes count:s.srgb.length / 4
             ideal:s.srgbIdeal.bytes model:@"srgb8-decode" bound:@"0.5" reference:@"MSL 4.1 section 8.7.7"];
    if (s.xr.length)
        [c data:@"read.xr" elem:WMElemF32 bytes:s.xr.bytes length:s.xr.length
           rule:WMNone(@"The decoding of extended-range (XR) channels is not stated in MTLPixelFormat.h or the "
                       @"MSL specification")];
    [c value:@"read.within-specification" boolean:s.within rule:WMExact()];
}

static void FmtUploadRead(WMCase *c, MTLPixelFormat format) {
    const FmtInfo *info = FmtLookup(c, format);
    [c require:@"feature-table.available" value:FmtTableAllows(c.device, info->available)];
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:W height:H
                                                                             mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [c texture:d];
    uint8_t texels[N * 16];
    FmtFillInput(info, texels, 0x1000u + (uint32_t)format);
    NSUInteger bytesPerRow = W * info->bytes;
    [c step:@"replaceRegion"];
    [texture replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:texels bytesPerRow:bytesPerRow];
    NSDictionary *layout = FmtLayout(format, W, H, bytesPerRow);
    [c data:@"input" elem:WMElemU8 bytes:texels length:N * info->bytes layout:layout rule:WMExact()];

    // Copies are byte copies; uploads and readbacks reproduce the bytes.
    NSData *readback = [c readTexture:texture level:0 slice:0 bytesPerRow:bytesPerRow];
    [c data:@"readback" elem:WMElemU8 bytes:readback.bytes length:readback.length layout:layout rule:WMExact()];
    uint8_t cpu[N * 16];
    [texture getBytes:cpu bytesPerRow:bytesPerRow fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
    [c value:@"getbytes-equals-readback" boolean:memcmp(cpu, readback.bytes, N * info->bytes) == 0 rule:WMExact()];

    FmtShader shader = FmtShaderType(info);
    id<MTLBuffer> results = FmtRunRead(c, texture, [@"fmt_read_" stringByAppendingString:FmtSuffix(shader)], 16);
    FmtRecordRead(c, info, texels, results);
}

#pragma mark - Shader write

static uint64_t FmtRTNE(double x) {
    double r = nearbyint(x); // default rounding mode: to nearest, ties to even
    return (uint64_t)(int64_t)r;
}

// Section 8.7.1.2, table 8.7, with the product computed exactly. Returns
// NO when the binary32 product f * scale would round to a different code
// than the exact product, so the result would depend on how the
// multiplication is carried out.
static BOOL FmtNormalizedCode(const FmtChannel *ch, float f, uint64_t *code) {
    double scale = ch->type == FmtUnorm ? (double)((1ull << ch->bits) - 1) : (double)((1ull << (ch->bits - 1)) - 1);
    double lo = ch->type == FmtUnorm ? 0.0 : -scale;
    if (isnan(f)) {
        *code = 0;
        return YES;
    }
    double exact = MIN(MAX((double)f * scale, lo), scale);
    double single = MIN(MAX((double)(f * (float)scale), lo), scale);
    if (nearbyint(exact) != nearbyint(single)) return NO;
    *code = FmtRTNE(exact) & ((1ull << ch->bits) - 1);
    return YES;
}

typedef NS_ENUM(uint8_t, FmtRound) { FmtRoundNearestEven, FmtRoundTowardZero };

// Section 8.7.4 encoding of a value as a 10- or 11-bit float under a
// rounding mode. Values below zero clamp to zero. Returns NO for NaN and for
// values above the largest finite value, which the section leaves open;
// *exact reports whether the value is representable without rounding.
static BOOL FmtSmallFloatCode(double v, unsigned mantissaBits, FmtRound mode, uint64_t *code, BOOL *exact) {
    if (isnan(v)) return NO;
    if (!(v > 0)) {
        *code = 0;
        *exact = YES;
        return YES;
    }
    double maxFinite = ldexp(2.0 - ldexp(1.0, -(int)mantissaBits), 15);
    if (v > maxFinite) return NO;
    int e;
    frexp(v, &e);
    e -= 1; // v = m * 2^e with m in [1, 2)
    double scaled;
    int field;
    if (e < -14) {
        field = 0;
        scaled = ldexp(v, 14 + (int)mantissaBits);
    } else {
        field = e + 15;
        scaled = (ldexp(v, -e) - 1.0) * (1u << mantissaBits);
    }
    double rounded = mode == FmtRoundNearestEven ? nearbyint(scaled) : floor(scaled);
    *exact = rounded == scaled;
    uint64_t m = (uint64_t)rounded;
    if (m == (1u << mantissaBits)) {
        m = 0;
        field += 1;
    }
    if (field >= 31) return NO;
    *code = ((uint64_t)field << mantissaBits) | m;
    return YES;
}

static uint16_t FmtHalfTowardZero(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
    uint32_t exponent = (bits >> 23) & 0xff, mantissa = bits & 0x7fffff;
    if (exponent == 0xff) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
    int e = (int)exponent - 127 + 15;
    if (e >= 31) return sign | 0x7bff;
    if (e <= 0) {
        if (e < -10) return sign;
        return sign | (uint16_t)((mantissa | 0x800000) >> (14 - e));
    }
    return sign | (uint16_t)(((uint32_t)e << 10) | (mantissa >> 13));
}

// Section 8.7.5 conversion to RGB9E5 (N = 9, B = 15, Emax = 31), with the
// preliminary exponent max(-B - 1, floor(log2(maxc))) + 1 + B. The
// round-toward-zero variant drops the + 0.5 terms.
static uint32_t FmtRGB9E5(const float *rgb, FmtRound mode) {
    const double N9 = 9, B15 = 15;
    double half = mode == FmtRoundNearestEven ? 0.5 : 0.0;
    double sharedMax = (511.0 / 512.0) * ldexp(1.0, 31 - 15);
    double c[3];
    for (int i = 0; i < 3; i++) c[i] = isnan(rgb[i]) ? 0.0 : MAX(0.0, MIN(sharedMax, (double)rgb[i]));
    double maxc = MAX(c[0], MAX(c[1], c[2]));
    double floorLog = maxc > 0 ? floor(log2(maxc)) : -INFINITY;
    double expp = MAX(-B15 - 1, floorLog) + 1 + B15;
    double maxs = floor(maxc / ldexp(1.0, (int)(expp - B15 - N9)) + half);
    double exps = maxs == 512 ? expp + 1 : expp;
    uint32_t out = (uint32_t)exps << 27;
    for (int i = 0; i < 3; i++)
        out |= (uint32_t)floor(c[i] / ldexp(1.0, (int)(exps - B15 - N9)) + half) << (9 * i);
    return out;
}

static BOOL FmtRGB9E5Exact(const float *rgb, uint32_t packed) {
    double sharedMax = (511.0 / 512.0) * ldexp(1.0, 31 - 15);
    int exponent = (int)(packed >> 27);
    for (int i = 0; i < 3; i++) {
        double clamped = isnan(rgb[i]) ? 0.0 : MAX(0.0, MIN(sharedMax, (double)rgb[i]));
        if (ldexp(((packed >> (9 * i)) & 511) / 512.0, exponent - 15) != clamped) return NO;
    }
    return YES;
}

static float FmtRandomFloat(uint32_t bits, float lo, float hi) {
    return lo + (hi - lo) * (float)((bits >> 8) & 0xffffff) / 16777216.0f;
}

// Values written by the shader-write case for one component.
static float FmtWriteFloat(const FmtChannel *ch, FmtType ct, NSUInteger t, unsigned comp, uint32_t r) {
    float v;
    if (ct == FmtUnorm || ct == FmtSnorm || ct == FmtSRGB || ct == FmtXR) {
        static const float special[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, NAN, 0.5f, -0.5f};
        if (t < 8) {
            v = special[(t + comp) % 8];
        } else if (t < 16 && ch) {
            // Near the midpoint between two codes.
            double scale = ct == FmtSnorm ? (double)((1ull << (ch->bits - 1)) - 1) : (double)((1ull << ch->bits) - 1);
            v = (float)(((r % (uint32_t)scale) + 0.5) / scale);
        } else {
            v = FmtRandomFloat(r, -1.25f, 1.25f);
        }
        // Move off values whose binary32 product would round to another
        // code than the exact product (FmtNormalizedCode).
        if (ch && (ct == FmtUnorm || ct == FmtSnorm)) {
            uint64_t probe;
            while (!FmtNormalizedCode(ch, v, &probe)) v = nextafterf(v, INFINITY);
        }
    } else if (ct == FmtHalf) {
        // Ties (1 + 2^-11, 1 + 3 * 2^-11), the largest finite value, a value
        // that rounds to it, the smallest normal value and the infinities.
        static const float special[] = {0.0f, -0.0f, 1.0f, 65504.0f, 65519.0f, 0x1p-14f, INFINITY,
                                        -INFINITY, 1.0f + 0x1p-11f, 1.0f + 3 * 0x1p-11f};
        if (t < 10) {
            v = special[(t + comp) % 10];
        } else if (t >= N - 4) {
            // Subnormal half values, representable without rounding.
            v = comp == 3 ? 1.0f : ldexpf(1.0f + (r & 7), -22 + (int)(r % 3));
        } else {
            uint32_t bits = (r & 0x807fffff) | ((uint32_t)(113 + r % 29) << 23);
            memcpy(&v, &bits, 4);
        }
    } else if (ct == FmtFloat) {
        static const float special[] = {0.0f, -0.0f, INFINITY, -INFINITY, FLT_MAX, -FLT_MIN};
        if (t < 6) {
            v = special[(t + comp) % 6];
        } else {
            uint32_t bits = (r & 0x807fffff) | ((uint32_t)(1 + r % 253) << 23);
            memcpy(&v, &bits, 4);
        }
    } else if (ct == FmtUF11 || ct == FmtUF10 || ct == FmtE5) {
        static const float special[] = {0.0f, -1.0f, -0.0f, 0x1p-16f, 1.0f, 60000.0f, NAN, 0.375f};
        if (t < 8) v = special[(t + comp) % 8];
        else v = ldexpf(FmtRandomFloat(r, 1.0f, 2.0f), (int)(r % 30) - 18);
    } else {
        v = 0;
    }
    return v;
}

typedef NS_ENUM(uint8_t, FmtGroup) { FmtGroupDefined, FmtGroupSubnormal, FmtGroupRounded, FmtGroupUndefined };

static void FmtShaderWrite(WMCase *c, MTLPixelFormat format) {
    const FmtInfo *info = FmtLookup(c, format);
    [c require:@"feature-table.write" value:FmtTableAllows(c.device, info->write)];
    FmtShader shader = FmtShaderType(info);
    FmtType type = info->channels[0].type;
    BOOL hasSRGB = NO;
    for (unsigned k = 0; k < info->count; k++) hasSRGB |= info->channels[k].type == FmtSRGB;

    uint32_t random[N * 4];
    WMPattern(random, sizeof random, 0x2000u + (uint32_t)format);
    float candidate[N][4];
    uint32_t candidateWords[N][4];
    uint8_t nearest[N][16], towardZero[N][16];
    FmtGroup group[N];

    for (NSUInteger t = 0; t < N; t++) {
        FmtGroup g = FmtGroupDefined;
        memset(nearest[t], 0, 16);
        memset(towardZero[t], 0, 16);
        for (unsigned comp = 0; comp < 4; comp++) {
            uint32_t r = random[t * 4 + comp];
            const FmtChannel *ch = NULL;
            for (unsigned k = 0; k < info->count; k++)
                if (info->channels[k].component == comp) ch = &info->channels[k];
            if (shader == FmtShaderFloat) {
                float v = FmtWriteFloat(ch, ch ? ch->type : type, t, comp, r);
                candidate[t][comp] = v;
                candidateWords[t][comp] = 0;
                if (!ch) continue;
                uint64_t ne = 0, z = 0;
                switch (ch->type) {
                case FmtUnorm: case FmtSnorm:
                    FmtNormalizedCode(ch, v, &ne);
                    z = ne;
                    break;
                case FmtHalf:
                    if (isnan(v)) {
                        g = FmtGroupUndefined;
                        break;
                    }
                    ne = WMHalfFromFloat(v);
                    z = FmtHalfTowardZero(v);
                    if (WMHalfToDouble((uint16_t)ne) != (double)v) g = MAX(g, FmtGroupRounded);
                    else if (((ne >> 10) & 31) == 0 && (ne & 0x3ff)) g = MAX(g, FmtGroupSubnormal);
                    break;
                case FmtFloat:
                    memcpy(&ne, &v, 4);
                    ne &= 0xffffffff;
                    z = ne;
                    break;
                case FmtUF11: case FmtUF10: {
                    unsigned m = ch->type == FmtUF11 ? 6 : 5;
                    BOOL exact = NO, ignored = NO;
                    if (!FmtSmallFloatCode(v, m, FmtRoundNearestEven, &ne, &exact) ||
                        !FmtSmallFloatCode(v, m, FmtRoundTowardZero, &z, &ignored)) {
                        g = FmtGroupUndefined;
                        break;
                    }
                    if (!exact) g = MAX(g, FmtGroupRounded);
                    break;
                }
                case FmtSRGB: case FmtXR: case FmtE5: default:
                    break;
                }
                FmtSetBits(nearest[t], ch->offset, ch->bits, ne);
                FmtSetBits(towardZero[t], ch->offset, ch->bits, z);
            } else {
                // Integer formats: values around and beyond the channel range.
                uint32_t w = r;
                uint64_t code = 0;
                if (ch) {
                    uint64_t max = ch->bits == 32 ? 0xffffffffull : (1ull << ch->bits) - 1;
                    static const int64_t signedSpecial[] = {0, -1, 127, 128, -128, -129, 32767, 32768, -32768,
                                                            -32769, INT32_MAX, INT32_MIN};
                    static const uint64_t unsignedSpecial[] = {0, 1, 255, 256, 65535, 65536, 1023, 1024,
                                                               0xffffffffull, 0x80000000ull};
                    if (ch->type == FmtSint) {
                        int32_t s = t < 12 ? (int32_t)signedSpecial[(t + comp) % 12] : (int32_t)r;
                        int64_t lo = -(int64_t)(1ull << (ch->bits - 1)), hi = (int64_t)(1ull << (ch->bits - 1)) - 1;
                        w = (uint32_t)s;
                        code = (uint64_t)MIN(MAX((int64_t)s, lo), hi) & max;
                    } else {
                        w = t < 10 ? (uint32_t)unsignedSpecial[(t + comp) % 10] : r;
                        // Section 8.7.6 defines saturation for 8- and 16-bit
                        // channels; out-of-range values for the 10- and
                        // 2-bit channels of RGB10A2Uint are not defined.
                        if (ch->bits != 8 && ch->bits != 16 && ch->bits != 32 && w > max) g = FmtGroupUndefined;
                        code = MIN((uint64_t)w, max);
                    }
                    FmtSetBits(nearest[t], ch->offset, ch->bits, code);
                    FmtSetBits(towardZero[t], ch->offset, ch->bits, code);
                }
                candidateWords[t][comp] = w;
                candidate[t][comp] = 0;
            }
        }
        if (shader == FmtShaderFloat && type == FmtE5) {
            uint32_t ne = FmtRGB9E5(candidate[t], FmtRoundNearestEven);
            uint32_t z = FmtRGB9E5(candidate[t], FmtRoundTowardZero);
            memcpy(nearest[t], &ne, 4);
            memcpy(towardZero[t], &z, 4);
            if (!FmtRGB9E5Exact(candidate[t], ne)) g = FmtGroupRounded;
        }
        group[t] = g;
    }

    // Texels in group order: defined, subnormal, rounded, undefined.
    NSUInteger order[N], counts[4] = {0, 0, 0, 0}, k = 0;
    for (int g = FmtGroupDefined; g <= FmtGroupUndefined; g++)
        for (NSUInteger t = 0; t < N; t++)
            if (group[t] == g) {
                order[k++] = t;
                counts[g]++;
            }
    float floats[N * 4];
    uint32_t words[N * 4];
    uint8_t expectedNearest[N * 16], expectedTowardZero[N * 16];
    for (NSUInteger i = 0; i < N; i++) {
        NSUInteger t = order[i];
        memcpy(&floats[i * 4], candidate[t], 16);
        memcpy(&words[i * 4], candidateWords[t], 16);
        memcpy(expectedNearest + i * info->bytes, nearest[t], info->bytes);
        memcpy(expectedTowardZero + i * info->bytes, towardZero[t], info->bytes);
    }

    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:W height:H
                                                                             mipmapped:NO];
    d.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> texture = [c texture:d];
    const void *values = shader == FmtShaderFloat ? (const void *)floats : (const void *)words;
    id<MTLBuffer> input = [c bufferWithBytes:values length:N * 16];
    id<MTLComputePipelineState> pipeline = [c computePipeline:[@"fmt_write_" stringByAppendingString:FmtSuffix(shader)]];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setTexture:texture atIndex:0];
        [e setBuffer:input offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    }];
    NSData *stored = [c readTexture:texture level:0 slice:0 bytesPerRow:W * info->bytes];
    const uint8_t *bytes = stored.bytes;
    [c data:@"write.input" elem:FmtElem(shader) bytes:values length:N * 16 rule:WMExact()];

    if (hasSRGB) {
        // Section 8.7.7 bounds each encoded color code by |reference -
        // integer| < 1.0, where the reference is the unrounded result of its
        // linear-to-sRGB rule; the linear alpha channel is compared exactly.
        NSMutableData *codes = [NSMutableData data], *references = [NSMutableData data];
        for (NSUInteger t = 0; t < N; t++)
            for (unsigned j = 0; j < info->count; j++) {
                const FmtChannel *ch = &info->channels[j];
                if (ch->type != FmtSRGB) continue;
                uint8_t code = (uint8_t)FmtGetBits(bytes + t * info->bytes, ch->offset, ch->bits);
                double reference = FmtSRGBEncodeReference(floats[t * 4 + ch->component]);
                [codes appendBytes:&code length:1];
                [references appendBytes:&reference length:sizeof reference];
            }
        [c bounded:@"write.srgb" elem:WMElemU8 bytes:codes.bytes count:codes.length ideal:references.bytes
             model:@"srgb8-encode" bound:@"1" reference:@"MSL 4.1 section 8.7.7"];
        const FmtChannel *alpha = NULL;
        for (unsigned j = 0; j < info->count; j++)
            if (info->channels[j].type == FmtUnorm) alpha = &info->channels[j];
        if (alpha) {
            uint8_t measured[N], expected[N];
            for (NSUInteger t = 0; t < N; t++) {
                measured[t] = (uint8_t)FmtGetBits(bytes + t * info->bytes, alpha->offset, alpha->bits);
                expected[t] = (uint8_t)FmtGetBits(expectedNearest + t * info->bytes, alpha->offset, alpha->bits);
            }
            [c data:@"write.alpha" elem:WMElemU8 bytes:measured length:N rule:WMExact()];
            [c value:@"write.alpha-equals-specification" boolean:memcmp(measured, expected, N) == 0 rule:WMExact()];
        }
        return;
    }
    if (type == FmtXR) {
        [c data:@"write.xr-texels" elem:WMElemU8 bytes:bytes length:N * info->bytes
           rule:WMNone(@"The encoding of extended-range (XR) channels is not stated in MTLPixelFormat.h or the "
                       @"MSL specification")];
        return;
    }

    // Defined texels follow MSL 4.1 section 8.7.1.2 (table 8.7) for
    // normalized channels, 8.7.6 for integer channels and 8.7.3 for float
    // channels; for half, 10- and 11-bit float and RGB9E5 channels they are
    // the values representable without rounding (8.7.2, 8.7.4, 8.7.5).
    size_t offset = 0, length = counts[FmtGroupDefined] * info->bytes;
    [c value:@"write.defined-texels" u64:counts[FmtGroupDefined] rule:WMExact()];
    [c data:@"write.defined" elem:WMElemU8 bytes:bytes length:length rule:WMExact()];
    [c value:@"write.defined-equals-specification" boolean:memcmp(bytes, expectedNearest, length) == 0
        rule:WMExact()];
    offset += length;
    if (counts[FmtGroupSubnormal]) {
        // Section 8.7.2: "Denormalized numbers for the half data type which
        // may be generated when converting a float to a half may not be
        // flushed to zero." The values are representable subnormals.
        length = counts[FmtGroupSubnormal] * info->bytes;
        [c data:@"write.half-subnormal" elem:WMElemU8 bytes:bytes + offset length:length rule:WMExact()];
        [c value:@"write.half-subnormal-preserved"
          boolean:memcmp(bytes + offset, expectedNearest + offset, length) == 0 rule:WMExact()];
        offset += length;
    }
    if (counts[FmtGroupRounded]) {
        // Section 1.6.7: texture writes to floating-point pixel formats use
        // the hardware's native rounding unless the shader is compiled with
        // -ftexture-write-rounding-mode; the harness library uses the
        // default. Whether the stored values match the section 8.7 conversion
        // (round to nearest; the 8.7.5 algorithm for RGB9E5) or truncation of
        // the mantissa is a property of the device.
        length = counts[FmtGroupRounded] * info->bytes;
        [c value:@"write.rounded-texels" u64:counts[FmtGroupRounded] rule:WMExact()];
        [c data:@"write.rounded" elem:WMElemU8 bytes:bytes + offset length:length
           rule:WMNone(@"MSL 4.1 section 1.6.7: texture writes to floating-point pixel formats use the "
                       @"hardware's native rounding by default")];
        [c value:@"write.rounded-matches-section-8.7-conversion"
          boolean:memcmp(bytes + offset, expectedNearest + offset, length) == 0 rule:WMDevice()];
        [c value:@"write.rounded-matches-truncation"
          boolean:memcmp(bytes + offset, expectedTowardZero + offset, length) == 0 rule:WMDevice()];
        offset += length;
    }
    if (counts[FmtGroupUndefined]) {
        NSString *why;
        if (shader != FmtShaderFloat)
            why = @"MSL 4.1 section 8.7.6 defines saturation only for 8- and 16-bit integer channels";
        else if (type == FmtHalf)
            why = @"MSL 4.1 section 8.7.2 allows a NaN to be stored as a NaN or as zero";
        else
            why = @"MSL 4.1 section 8.7.4 does not define NaN or overflow for 10- and 11-bit floats";
        [c data:@"write.undefined" elem:WMElemU8 bytes:bytes + offset
         length:counts[FmtGroupUndefined] * info->bytes rule:WMNone(why)];
    }
}

#pragma mark - Render target output

static void FmtRender(WMCase *c, MTLPixelFormat format) {
    const FmtInfo *info = FmtLookup(c, format);
    [c require:@"feature-table.color" value:FmtTableAllows(c.device, info->color)];
    FmtShader shader = FmtShaderType(info);
    uint32_t random[N * 4];
    WMPattern(random, sizeof random, 0x3000u + (uint32_t)format);
    float floats[N * 4];
    uint32_t words[N * 4];
    uint8_t expected[N * 16];
    memset(expected, 0, sizeof expected);
    BOOL representableKnown = YES;
    // Texels 0-15 (rows 0 and 1) hold values the format represents exactly;
    // texels 16-31 (rows 2 and 3) hold values that need rounding or range
    // conversion.
    for (NSUInteger t = 0; t < N; t++) {
        BOOL exactRow = t < N / 2;
        uint8_t *texel = expected + t * info->bytes;
        float e5[3] = {0, 0, 0};
        for (unsigned comp = 0; comp < 4; comp++) {
            uint32_t r = random[t * 4 + comp];
            const FmtChannel *ch = NULL;
            for (unsigned k = 0; k < info->count; k++)
                if (info->channels[k].component == comp) ch = &info->channels[k];
            FmtType ct = ch ? ch->type : info->channels[0].type;
            unsigned bits = ch ? ch->bits : 8;
            if (shader == FmtShaderFloat) {
                float v = 0;
                uint64_t code = 0;
                switch (ct) {
                case FmtUnorm: case FmtSRGB: case FmtXR:
                    v = exactRow ? (float)(r & 1) : FmtRandomFloat(r, -0.2f, 1.2f);
                    code = (r & 1) ? (1ull << bits) - 1 : 0;
                    break;
                case FmtSnorm: {
                    int choice = (int)(r % 3) - 1;
                    v = exactRow ? (float)choice : FmtRandomFloat(r, -1.2f, 1.2f);
                    code = (uint64_t)(int64_t)(choice * ((1ll << (bits - 1)) - 1)) & ((1ull << bits) - 1);
                    break;
                }
                case FmtHalf: {
                    uint16_t h = (uint16_t)((r & 0x83ff) | ((1 + (r >> 16) % 30) << 10));
                    v = exactRow ? (float)WMHalfToDouble(h) : FmtRandomFloat(r, -3.0f, 3.0f) + 0x1p-20f;
                    code = h;
                    break;
                }
                case FmtFloat: {
                    uint32_t b = (r & 0x807fffff) | ((uint32_t)(1 + r % 253) << 23);
                    if (!exactRow) b = (t & 1) ? 0x7fc00000 : (r & 0x807fffff); // NaN and subnormals
                    memcpy(&v, &b, 4);
                    code = b;
                    break;
                }
                case FmtUF11: case FmtUF10: {
                    unsigned m = ct == FmtUF11 ? 6 : 5;
                    code = ((uint64_t)(r % 31) << m) | ((r >> 8) & ((1u << m) - 1));
                    v = exactRow ? (float)FmtDecodeSmallFloat(code, m) : FmtRandomFloat(r, -1.0f, 4.0f);
                    break;
                }
                case FmtE5:
                    v = exactRow ? (float)ldexp((r % 512) / 512.0, (int)((random[t * 4] >> 20) % 32) - 15)
                                 : FmtRandomFloat(r, -1.0f, 4.0f);
                    break;
                default:
                    break;
                }
                if (!ch) v = (float)(r & 1);
                floats[t * 4 + comp] = v;
                if (ch && comp < 3) e5[comp] = v;
                if (ch && ct != FmtE5) FmtSetBits(texel, ch->offset, ch->bits, code);
                if (ct == FmtXR) representableKnown = NO;
            } else {
                uint64_t max = bits == 32 ? 0xffffffffull : (1ull << bits) - 1;
                uint32_t w;
                uint64_t code;
                if (ct == FmtSint) {
                    int64_t lo = -(int64_t)(1ull << (bits - 1)), hi = (int64_t)(1ull << (bits - 1)) - 1;
                    int64_t v = exactRow ? lo + (int64_t)(r % (uint64_t)(hi - lo + 1)) : (int64_t)(int32_t)r;
                    if (!exactRow && bits < 32 && v >= lo && v <= hi) v = (v >= 0) ? hi + 1 + (r & 7) : lo - 1 - (r & 7);
                    if (!exactRow && bits == 32) v = (int64_t)(int32_t)r;
                    w = (uint32_t)(int32_t)v;
                    code = (uint64_t)v & max;
                } else {
                    uint64_t v = exactRow ? (uint64_t)r % (max + 1) : (uint64_t)r | (max + 1);
                    if (bits == 32) v = r;
                    w = (uint32_t)v;
                    code = v & max;
                }
                words[t * 4 + comp] = w;
                if (ch) FmtSetBits(texel, ch->offset, ch->bits, code);
            }
        }
        if (shader == FmtShaderFloat && info->channels[0].type == FmtE5) {
            uint32_t packed = FmtRGB9E5(e5, FmtRoundNearestEven);
            memcpy(texel, &packed, 4);
        }
    }

    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [c function:@"fmt_fullscreen_vertex"];
    pd.fragmentFunction = [c function:[@"fmt_fragment_" stringByAppendingString:FmtSuffix(shader)]];
    pd.colorAttachments[0].pixelFormat = format;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:pd];
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:W height:H
                                                                             mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget;
    d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> target = [c texture:d];
    const void *values = shader == FmtShaderFloat ? (const void *)floats : (const void *)words;
    id<MTLBuffer> input = [c bufferWithBytes:values length:N * 16];
    const uint32_t width = W;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setFragmentBuffer:input offset:0 atIndex:0];
        [e setFragmentBytes:&width length:sizeof width atIndex:1];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    NSUInteger bytesPerRow = W * info->bytes;
    NSData *stored = [c readTexture:target level:0 slice:0 bytesPerRow:bytesPerRow];
    size_t half = N / 2 * info->bytes;
    [c data:@"render.input" elem:FmtElem(shader) bytes:values length:N * 16 rule:WMExact()];
    if (representableKnown) {
        // A value the format represents exactly is stored as that value.
        [c data:@"render.representable" elem:WMElemU8 bytes:stored.bytes length:half rule:WMExact()];
        [c value:@"render.representable-equals-specification"
          boolean:memcmp(stored.bytes, expected, half) == 0 rule:WMExact()];
    } else {
        [c data:@"render.representable" elem:WMElemU8 bytes:stored.bytes length:half
           rule:WMNone(@"The encoding of extended-range (XR) channels is not stated in MTLPixelFormat.h or the "
                       @"MSL specification")];
    }
    [c data:@"render.converted" elem:WMElemU8 bytes:(const uint8_t *)stored.bytes + half length:half
       rule:WMNone(@"MSL 4.1 section 8.7 states conversion rules for the texture write functions; the "
                   @"rounding and range conversion of fragment outputs stored to color attachments is not "
                   @"stated")];
}

#pragma mark - Color format cases

#define FMT_CASES(fn, name, enumName)                                                                   \
    WMREF_CASE(fn##UploadRead, "format." name ".upload-read", "format.color",                           \
               "Upload, readback and shader read conversion of " name)                                  \
    {                                                                                                   \
        FmtUploadRead(c, enumName);                                                                     \
    }                                                                                                   \
    WMREF_CASE(fn##Render, "format." name ".render", "format.color",                                    \
               "Fragment output stored to a " name " color attachment")                                 \
    {                                                                                                   \
        FmtRender(c, enumName);                                                                         \
    }

#define FMT_WRITE_CASE(fn, name, enumName)                                                              \
    WMREF_CASE(fn##ShaderWrite, "format." name ".shader-write", "format.color",                         \
               "Shader texture write conversion to " name)                                              \
    {                                                                                                   \
        FmtShaderWrite(c, enumName);                                                                    \
    }

#define FMT_ALL(fn, name, enumName) FMT_CASES(fn, name, enumName) FMT_WRITE_CASE(fn, name, enumName)

FMT_ALL(A8Unorm, "A8Unorm", MTLPixelFormatA8Unorm)
FMT_ALL(R8Unorm, "R8Unorm", MTLPixelFormatR8Unorm)
FMT_ALL(R8UnormSRGB, "R8Unorm_sRGB", MTLPixelFormatR8Unorm_sRGB)
FMT_ALL(R8Snorm, "R8Snorm", MTLPixelFormatR8Snorm)
FMT_ALL(R8Uint, "R8Uint", MTLPixelFormatR8Uint)
FMT_ALL(R8Sint, "R8Sint", MTLPixelFormatR8Sint)
FMT_ALL(R16Unorm, "R16Unorm", MTLPixelFormatR16Unorm)
FMT_ALL(R16Snorm, "R16Snorm", MTLPixelFormatR16Snorm)
FMT_ALL(R16Uint, "R16Uint", MTLPixelFormatR16Uint)
FMT_ALL(R16Sint, "R16Sint", MTLPixelFormatR16Sint)
FMT_ALL(R16Float, "R16Float", MTLPixelFormatR16Float)
FMT_ALL(RG8Unorm, "RG8Unorm", MTLPixelFormatRG8Unorm)
FMT_ALL(RG8UnormSRGB, "RG8Unorm_sRGB", MTLPixelFormatRG8Unorm_sRGB)
FMT_ALL(RG8Snorm, "RG8Snorm", MTLPixelFormatRG8Snorm)
FMT_ALL(RG8Uint, "RG8Uint", MTLPixelFormatRG8Uint)
FMT_ALL(RG8Sint, "RG8Sint", MTLPixelFormatRG8Sint)
// The packed 16-bit formats have no Write capability in any GPU family of
// the Metal Feature Set Tables, so they have no shader-write case.
FMT_CASES(B5G6R5Unorm, "B5G6R5Unorm", MTLPixelFormatB5G6R5Unorm)
FMT_CASES(A1BGR5Unorm, "A1BGR5Unorm", MTLPixelFormatA1BGR5Unorm)
FMT_CASES(ABGR4Unorm, "ABGR4Unorm", MTLPixelFormatABGR4Unorm)
FMT_CASES(BGR5A1Unorm, "BGR5A1Unorm", MTLPixelFormatBGR5A1Unorm)
FMT_ALL(R32Uint, "R32Uint", MTLPixelFormatR32Uint)
FMT_ALL(R32Sint, "R32Sint", MTLPixelFormatR32Sint)
FMT_ALL(R32Float, "R32Float", MTLPixelFormatR32Float)
FMT_ALL(RG16Unorm, "RG16Unorm", MTLPixelFormatRG16Unorm)
FMT_ALL(RG16Snorm, "RG16Snorm", MTLPixelFormatRG16Snorm)
FMT_ALL(RG16Uint, "RG16Uint", MTLPixelFormatRG16Uint)
FMT_ALL(RG16Sint, "RG16Sint", MTLPixelFormatRG16Sint)
FMT_ALL(RG16Float, "RG16Float", MTLPixelFormatRG16Float)
FMT_ALL(RGBA8Unorm, "RGBA8Unorm", MTLPixelFormatRGBA8Unorm)
FMT_ALL(RGBA8UnormSRGB, "RGBA8Unorm_sRGB", MTLPixelFormatRGBA8Unorm_sRGB)
FMT_ALL(RGBA8Snorm, "RGBA8Snorm", MTLPixelFormatRGBA8Snorm)
FMT_ALL(RGBA8Uint, "RGBA8Uint", MTLPixelFormatRGBA8Uint)
FMT_ALL(RGBA8Sint, "RGBA8Sint", MTLPixelFormatRGBA8Sint)
FMT_ALL(BGRA8Unorm, "BGRA8Unorm", MTLPixelFormatBGRA8Unorm)
FMT_ALL(BGRA8UnormSRGB, "BGRA8Unorm_sRGB", MTLPixelFormatBGRA8Unorm_sRGB)
FMT_ALL(RGB10A2Unorm, "RGB10A2Unorm", MTLPixelFormatRGB10A2Unorm)
FMT_ALL(RGB10A2Uint, "RGB10A2Uint", MTLPixelFormatRGB10A2Uint)
FMT_ALL(RG11B10Float, "RG11B10Float", MTLPixelFormatRG11B10Float)
FMT_ALL(RGB9E5Float, "RGB9E5Float", MTLPixelFormatRGB9E5Float)
FMT_ALL(BGR10A2Unorm, "BGR10A2Unorm", MTLPixelFormatBGR10A2Unorm)
FMT_ALL(BGR10XR, "BGR10_XR", MTLPixelFormatBGR10_XR)
FMT_ALL(BGR10XRSRGB, "BGR10_XR_sRGB", MTLPixelFormatBGR10_XR_sRGB)
FMT_ALL(RG32Uint, "RG32Uint", MTLPixelFormatRG32Uint)
FMT_ALL(RG32Sint, "RG32Sint", MTLPixelFormatRG32Sint)
FMT_ALL(RG32Float, "RG32Float", MTLPixelFormatRG32Float)
FMT_ALL(RGBA16Unorm, "RGBA16Unorm", MTLPixelFormatRGBA16Unorm)
FMT_ALL(RGBA16Snorm, "RGBA16Snorm", MTLPixelFormatRGBA16Snorm)
FMT_ALL(RGBA16Uint, "RGBA16Uint", MTLPixelFormatRGBA16Uint)
FMT_ALL(RGBA16Sint, "RGBA16Sint", MTLPixelFormatRGBA16Sint)
FMT_ALL(RGBA16Float, "RGBA16Float", MTLPixelFormatRGBA16Float)
FMT_ALL(BGRA10XR, "BGRA10_XR", MTLPixelFormatBGRA10_XR)
FMT_ALL(BGRA10XRSRGB, "BGRA10_XR_sRGB", MTLPixelFormatBGRA10_XR_sRGB)
FMT_ALL(RGBA32Uint, "RGBA32Uint", MTLPixelFormatRGBA32Uint)
FMT_ALL(RGBA32Sint, "RGBA32Sint", MTLPixelFormatRGBA32Sint)
FMT_ALL(RGBA32Float, "RGBA32Float", MTLPixelFormatRGBA32Float)

#pragma mark - Depth and stencil formats

// Uploads `bytes` (bytesPerPixel per texel, W x H) into one aspect of a
// depth or stencil texture through a blit.
static void FmtBlitUpload(WMCase *c, id<MTLTexture> texture, const void *bytes, NSUInteger bytesPerPixel,
                          MTLBlitOption option) {
    id<MTLBuffer> source = [c bufferWithBytes:bytes length:N * bytesPerPixel];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:W * bytesPerPixel
          sourceBytesPerImage:N * bytesPerPixel sourceSize:MTLSizeMake(W, H, 1) toTexture:texture
             destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0) options:option];
    }];
}

static NSData *FmtBlitReadback(WMCase *c, id<MTLTexture> texture, NSUInteger bytesPerPixel, MTLBlitOption option) {
    id<MTLBuffer> destination = [c buffer:N * bytesPerPixel];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(W, H, 1) toBuffer:destination destinationOffset:0
    destinationBytesPerRow:W * bytesPerPixel destinationBytesPerImage:N * bytesPerPixel options:option];
    }];
    return [NSData dataWithBytes:destination.contents length:destination.length];
}

static id<MTLTexture> FmtDepthTexture(WMCase *c, MTLPixelFormat format, MTLTextureUsage usage) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:W height:H
                                                                             mipmapped:NO];
    d.usage = usage;
    d.storageMode = MTLStorageModePrivate;
    return [c texture:d];
}

// Random depth values in [0, 1] with the corners 0 and 1 first.
static void FmtDepthValues(float *values, uint32_t seed) {
    uint32_t random[N];
    WMPattern(random, sizeof random, seed);
    for (NSUInteger t = 0; t < N; t++)
        values[t] = t == 0 ? 0.0f : t == 1 ? 1.0f : t == 2 ? 0.5f : (float)((random[t] >> 8) & 0xffffff) / 16777216.0f;
}

// The stencil value is the first component of a read; the other
// components of a stencil read are not defined by MSL 4.1 (section 6.13).
static void FmtRecordStencilRead(WMCase *c, NSString *name, id<MTLTexture> texture, const uint8_t *stencil) {
    id<MTLBuffer> results = FmtRunRead(c, texture, @"fmt_read_uint", 16);
    uint32_t first[N];
    BOOL equal = YES;
    for (NSUInteger t = 0; t < N; t++) {
        first[t] = ((const uint32_t *)results.contents)[t * 4];
        equal &= first[t] == stencil[t];
    }
    [c data:name elem:WMElemU32 bytes:first length:sizeof first rule:WMExact()];
    [c value:[name stringByAppendingString:@"-equals-stored"] boolean:equal rule:WMExact()];
}

WMREF_CASE(depth16UploadRead, "format.Depth16Unorm.upload-read", "format.depth-stencil",
           "Depth16Unorm upload and readback through blits and depth read conversion")
{
    [c require:@"feature-table.available" value:FmtTableAllows(c.device, 2)];
    uint16_t codes[N];
    WMPattern(codes, sizeof codes, 0x4001);
    codes[0] = 0;
    codes[1] = 0xffff;
    id<MTLTexture> texture = FmtDepthTexture(c, MTLPixelFormatDepth16Unorm, MTLTextureUsageShaderRead);
    FmtBlitUpload(c, texture, codes, 2, MTLBlitOptionNone);
    NSDictionary *layout = FmtLayout(MTLPixelFormatDepth16Unorm, W, H, W * 2);
    [c data:@"input" elem:WMElemU16 bytes:codes length:sizeof codes layout:layout rule:WMExact()];
    NSData *readback = FmtBlitReadback(c, texture, 2, MTLBlitOptionNone);
    [c data:@"readback" elem:WMElemU16 bytes:readback.bytes length:readback.length layout:layout rule:WMExact()];
    id<MTLBuffer> results = FmtRunRead(c, texture, @"fmt_read_depth", 4);
    const float *measured = results.contents;
    double ideal[N];
    BOOL within = YES;
    for (NSUInteger t = 0; t < N; t++) {
        ideal[t] = codes[t] / 65535.0;
        within &= fabs(measured[t] - ideal[t]) <= 1.5 * FmtUlp32(ideal[t]);
    }
    // Table 8.6: 16-bit normalized unsigned values, <= 1.5 ulp; 0 and 65535
    // convert exactly to 0.0 and 1.0.
    [c bounded:@"read.normalized" elem:WMElemF32 bytes:measured count:N ideal:ideal model:@"ulp" bound:@"1.5"
     reference:@"MSL 4.1 section 8.7.1.1, table 8.6"];
    [c data:@"read.exact" elem:WMElemF32 bytes:measured length:2 * sizeof(float) rule:WMExact()];
    within &= measured[0] == 0.0f && measured[1] == 1.0f;
    [c value:@"read.within-specification" boolean:within rule:WMExact()];
}

WMREF_CASE(depth32UploadRead, "format.Depth32Float.upload-read", "format.depth-stencil",
           "Depth32Float upload and readback through blits and depth read")
{
    [c require:@"feature-table.available" value:FmtTableAllows(c.device, 2)];
    float values[N];
    FmtDepthValues(values, 0x4002);
    id<MTLTexture> texture = FmtDepthTexture(c, MTLPixelFormatDepth32Float, MTLTextureUsageShaderRead);
    FmtBlitUpload(c, texture, values, 4, MTLBlitOptionNone);
    NSDictionary *layout = FmtLayout(MTLPixelFormatDepth32Float, W, H, W * 4);
    [c data:@"input" elem:WMElemF32 bytes:values length:sizeof values layout:layout rule:WMExact()];
    NSData *readback = FmtBlitReadback(c, texture, 4, MTLBlitOptionNone);
    [c data:@"readback" elem:WMElemF32 bytes:readback.bytes length:readback.length layout:layout rule:WMExact()];
    // Section 8.7.3: values other than NaN and subnormals are preserved.
    id<MTLBuffer> results = FmtRunRead(c, texture, @"fmt_read_depth", 4);
    [c data:@"read" elem:WMElemF32 bytes:results.contents length:sizeof values rule:WMExact()];
    [c value:@"read-equals-stored" boolean:memcmp(results.contents, values, sizeof values) == 0 rule:WMExact()];
}

WMREF_CASE(stencil8UploadRead, "format.Stencil8.upload-read", "format.depth-stencil",
           "Stencil8 upload and readback through blits and stencil read")
{
    [c require:@"feature-table.available" value:FmtTableAllows(c.device, 2)];
    uint8_t stencil[N];
    WMPattern(stencil, sizeof stencil, 0x4003);
    stencil[0] = 0;
    stencil[1] = 255;
    id<MTLTexture> texture = FmtDepthTexture(c, MTLPixelFormatStencil8, MTLTextureUsageShaderRead);
    FmtBlitUpload(c, texture, stencil, 1, MTLBlitOptionNone);
    NSDictionary *layout = FmtLayout(MTLPixelFormatStencil8, W, H, W);
    [c data:@"input" elem:WMElemU8 bytes:stencil length:sizeof stencil layout:layout rule:WMExact()];
    NSData *readback = FmtBlitReadback(c, texture, 1, MTLBlitOptionNone);
    [c data:@"readback" elem:WMElemU8 bytes:readback.bytes length:readback.length layout:layout rule:WMExact()];
    FmtRecordStencilRead(c, @"read.stencil", texture, stencil);
}

WMREF_CASE(depth32Stencil8UploadRead, "format.Depth32Float_Stencil8.upload-read", "format.depth-stencil",
           "Depth32Float_Stencil8 per-aspect blits, depth read and an X32_Stencil8 view")
{
    [c require:@"feature-table.available" value:FmtTableAllows(c.device, 2)];
    float depth[N];
    uint8_t stencil[N];
    FmtDepthValues(depth, 0x4004);
    WMPattern(stencil, sizeof stencil, 0x4005);
    id<MTLTexture> texture = FmtDepthTexture(c, MTLPixelFormatDepth32Float_Stencil8,
                                             MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView);
    // Blits of a combined format select one aspect with an option
    // (MTLBlitCommandEncoder.h: MTLBlitOptionDepthFromDepthStencil,
    // MTLBlitOptionStencilFromDepthStencil).
    FmtBlitUpload(c, texture, depth, 4, MTLBlitOptionDepthFromDepthStencil);
    FmtBlitUpload(c, texture, stencil, 1, MTLBlitOptionStencilFromDepthStencil);
    [c data:@"input.depth" elem:WMElemF32 bytes:depth length:sizeof depth rule:WMExact()];
    [c data:@"input.stencil" elem:WMElemU8 bytes:stencil length:sizeof stencil rule:WMExact()];
    NSData *depthBack = FmtBlitReadback(c, texture, 4, MTLBlitOptionDepthFromDepthStencil);
    NSData *stencilBack = FmtBlitReadback(c, texture, 1, MTLBlitOptionStencilFromDepthStencil);
    [c data:@"readback.depth" elem:WMElemF32 bytes:depthBack.bytes length:depthBack.length rule:WMExact()];
    [c data:@"readback.stencil" elem:WMElemU8 bytes:stencilBack.bytes length:stencilBack.length rule:WMExact()];
    id<MTLBuffer> results = FmtRunRead(c, texture, @"fmt_read_depth", 4);
    [c data:@"read.depth" elem:WMElemF32 bytes:results.contents length:sizeof depth rule:WMExact()];
    [c value:@"read.depth-equals-stored" boolean:memcmp(results.contents, depth, sizeof depth) == 0 rule:WMExact()];
    id<MTLTexture> view = [texture newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    if (!view) [c fail:@"newTextureViewWithPixelFormat:X32_Stencil8 returned nil"];
    [c value:@"view.pixel-format" u64:view.pixelFormat rule:WMExact()];
    FmtRecordStencilRead(c, @"read.x32-stencil8-view", view, stencil);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static BOOL FmtDepth24Stencil8Supported(id<MTLDevice> device) {
#if TARGET_OS_OSX
    return device.depth24Stencil8PixelFormatSupported;
#else
    // MTLDevice.h marks the property and the format unavailable on iOS.
    return NO;
#endif
}
#pragma clang diagnostic pop

WMREF_CASE(depth24Stencil8, "format.Depth24Unorm_Stencil8.upload-read", "format.depth-stencil",
           "Depth24Unorm_Stencil8 and X24_Stencil8, gated by depth24Stencil8PixelFormatSupported")
{
    [c require:@"depth24-stencil8-pixel-format-supported" value:FmtDepth24Stencil8Supported(c.device)];
    // No device that reports support was available to measure the blit
    // layout of the 24-bit depth aspect, so the contents are not covered.
    [c fail:@"Depth24Unorm_Stencil8 contents are not covered by this case"];
}

#pragma mark - 422 formats

static void FmtYUV422(WMCase *c, MTLPixelFormat format, BOOL lumaFirst) {
    // Feature Set Tables: GBGR422 and BGRG422 list Filter from Apple2.
    [c require:@"feature-table.available" value:FmtTableAllows(c.device, 2)];
    // Two pixels share 32 bits. GBGR422 stores Y0, Cb, Y1, Cr from the lowest
    // byte; BGRG422 stores Cb, Y0, Cr, Y1 (MTLPixelFormat.h).
    uint8_t texels[N * 2];
    WMPattern(texels, sizeof texels, 0x5000u + (uint32_t)format);
    unsigned y0 = lumaFirst ? 0 : 1, y1 = lumaFirst ? 2 : 3;
    texels[y0] = 0;
    texels[y1] = 255;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:W height:H
                                                                             mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [c texture:d];
    NSUInteger bytesPerRow = W * 2;
    [texture replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:texels bytesPerRow:bytesPerRow];
    NSDictionary *layout = FmtLayout(format, W, H, bytesPerRow);
    [c data:@"input" elem:WMElemU8 bytes:texels length:sizeof texels layout:layout rule:WMExact()];
    NSData *readback = [c readTexture:texture level:0 slice:0 bytesPerRow:bytesPerRow];
    [c data:@"readback" elem:WMElemU8 bytes:readback.bytes length:readback.length layout:layout rule:WMExact()];

    // MTLPixelFormat.h: "the shader will receive (Cr, Y, Cb, 1)".
    id<MTLBuffer> results = FmtRunRead(c, texture, @"fmt_read_float", 16);
    const float *v = results.contents;
    float luma[N], chroma[N * 2], alpha[N], corners[2];
    double ideal[N];
    BOOL within = YES;
    for (NSUInteger t = 0; t < N; t++) {
        NSUInteger pair = t / 2;
        uint8_t code = texels[pair * 4 + (t % 2 ? y1 : y0)];
        luma[t] = v[t * 4 + 1];
        ideal[t] = code / 255.0;
        within &= fabs(luma[t] - ideal[t]) <= 1.5 * FmtUlp32(ideal[t]);
        chroma[t * 2] = v[t * 4];
        chroma[t * 2 + 1] = v[t * 4 + 2];
        alpha[t] = v[t * 4 + 3];
        within &= alpha[t] == 1.0f;
    }
    corners[0] = luma[0];
    corners[1] = luma[1];
    within &= corners[0] == 0.0f && corners[1] == 1.0f;
    // Luma is an 8-bit normalized unsigned value (table 8.6, with 0 and
    // 255 exact).
    [c bounded:@"read.luma" elem:WMElemF32 bytes:luma count:N ideal:ideal model:@"ulp" bound:@"1.5"
     reference:@"MSL 4.1 section 8.7.1.1, table 8.6"];
    [c data:@"read.luma-corners" elem:WMElemF32 bytes:corners length:sizeof corners rule:WMExact()];
    [c data:@"read.alpha" elem:WMElemF32 bytes:alpha length:sizeof alpha rule:WMExact()];
    // Reads at a pixel combine the chroma of neighbouring pairs on this
    // device; the reconstruction is not stated in MTLPixelFormat.h.
    [c data:@"read.chroma" elem:WMElemF32 bytes:chroma length:sizeof chroma
       rule:WMNone(@"MTLPixelFormat.h states that two pixels share Cb and Cr but not how a read at one pixel "
                   @"reconstructs them")];
    [c value:@"read.within-specification" boolean:within rule:WMExact()];
}

WMREF_CASE(gbgr422, "format.GBGR422.upload-read", "format.yuv422", "GBGR422 upload, readback and read")
{
    FmtYUV422(c, MTLPixelFormatGBGR422, YES);
}

WMREF_CASE(bgrg422, "format.BGRG422.upload-read", "format.yuv422", "BGRG422 upload, readback and read")
{
    FmtYUV422(c, MTLPixelFormatBGRG422, NO);
}

#pragma mark - Three-channel texture-buffer formats

// MTLPixelFormat.h (SDK 27.2): RGB8, RGB16 and RGB32 formats, available from
// macOS 27.0 and iOS 27.0, only for MTLTextureTypeTextureBuffer textures
// without MTLTextureUsageShaderWrite. Raw values are used because the
// deployment target predates the declarations.
static const FmtInfo kTextureBufferFormats[] = {
    {(MTLPixelFormat)45, 3, 3, {{R, 0, 8, FmtUnorm}, {G, 8, 8, FmtUnorm}, {B, 16, 8, FmtUnorm}}, 2, 0, 0},
    {(MTLPixelFormat)46, 3, 3, {{R, 0, 8, FmtSnorm}, {G, 8, 8, FmtSnorm}, {B, 16, 8, FmtSnorm}}, 2, 0, 0},
    {(MTLPixelFormat)47, 3, 3, {{R, 0, 8, FmtUint}, {G, 8, 8, FmtUint}, {B, 16, 8, FmtUint}}, 2, 0, 0},
    {(MTLPixelFormat)48, 3, 3, {{R, 0, 8, FmtSint}, {G, 8, 8, FmtSint}, {B, 16, 8, FmtSint}}, 2, 0, 0},
    {(MTLPixelFormat)95, 6, 3, {{R, 0, 16, FmtUnorm}, {G, 16, 16, FmtUnorm}, {B, 32, 16, FmtUnorm}}, 2, 0, 0},
    {(MTLPixelFormat)96, 6, 3, {{R, 0, 16, FmtSnorm}, {G, 16, 16, FmtSnorm}, {B, 32, 16, FmtSnorm}}, 2, 0, 0},
    {(MTLPixelFormat)97, 6, 3, {{R, 0, 16, FmtUint}, {G, 16, 16, FmtUint}, {B, 32, 16, FmtUint}}, 2, 0, 0},
    {(MTLPixelFormat)98, 6, 3, {{R, 0, 16, FmtSint}, {G, 16, 16, FmtSint}, {B, 32, 16, FmtSint}}, 2, 0, 0},
    {(MTLPixelFormat)99, 6, 3, {{R, 0, 16, FmtHalf}, {G, 16, 16, FmtHalf}, {B, 32, 16, FmtHalf}}, 2, 0, 0},
    {(MTLPixelFormat)120, 12, 3, {{R, 0, 32, FmtUint}, {G, 32, 32, FmtUint}, {B, 64, 32, FmtUint}}, 2, 0, 0},
    {(MTLPixelFormat)121, 12, 3, {{R, 0, 32, FmtSint}, {G, 32, 32, FmtSint}, {B, 64, 32, FmtSint}}, 2, 0, 0},
    {(MTLPixelFormat)122, 12, 3, {{R, 0, 32, FmtFloat}, {G, 32, 32, FmtFloat}, {B, 64, 32, FmtFloat}}, 2, 0, 0},
};

static void FmtTextureBuffer(WMCase *c, NSUInteger raw) {
    const FmtInfo *info = NULL;
    for (size_t i = 0; i < sizeof kTextureBufferFormats / sizeof kTextureBufferFormats[0]; i++)
        if (kTextureBufferFormats[i].format == raw) info = &kTextureBufferFormats[i];
    if (!info) [c fail:@"no description for pixel format %lu", (unsigned long)raw];
    BOOL declared = NO;
    if (@available(macOS 27.0, iOS 27.0, *)) declared = YES;
    [c require:@"api.available" value:declared];
    // "RGB pixel formats": Apple11 and later, for texture buffers only
    // (Metal Feature Set Tables, 2026-10-23, feature availability and its
    // footnote 10). MTLGPUFamilyApple11 is 1011.
    [c require:@"family.apple11" value:[c.device supportsFamily:(MTLGPUFamily)1011]];
    uint8_t texels[N * 16];
    FmtFillInput(info, texels, 0x6000u + (uint32_t)raw);
    NSUInteger length = N * info->bytes;
    id<MTLBuffer> storage = [c bufferWithBytes:texels length:length];
    [c step:@"textureBufferDescriptorWithPixelFormat"];
    MTLTextureDescriptor *d = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:info->format width:N
                                                                            resourceOptions:MTLResourceStorageModeShared
                                                                                      usage:MTLTextureUsageShaderRead];
    [c step:@"newTextureWithDescriptor:offset:bytesPerRow:"];
    id<MTLTexture> texture = [storage newTextureWithDescriptor:d offset:0 bytesPerRow:length];
    if (!texture) [c fail:@"newTextureWithDescriptor:offset:bytesPerRow: returned nil"];
    [c data:@"input" elem:WMElemU8 bytes:texels length:length rule:WMExact()];
    FmtShader shader = FmtShaderType(info);
    id<MTLComputePipelineState> pipeline =
        [c computePipeline:[@"fmt_read_buffer_" stringByAppendingString:FmtSuffix(shader)]];
    id<MTLBuffer> results = [c buffer:N * 16];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setTexture:texture atIndex:0];
        [e setBuffer:results offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(N, 1, 1)];
    }];
    FmtRecordRead(c, info, texels, results);
}

#define FMT_TEXTURE_BUFFER_CASE(fn, name, raw)                                                          \
    WMREF_CASE(fn, "format." name ".texture-buffer-read", "format.texture-buffer",                      \
               "Texture-buffer read conversion of " name)                                               \
    {                                                                                                   \
        FmtTextureBuffer(c, raw);                                                                       \
    }

FMT_TEXTURE_BUFFER_CASE(rgb8Unorm, "RGB8Unorm", 45)
FMT_TEXTURE_BUFFER_CASE(rgb8Snorm, "RGB8Snorm", 46)
FMT_TEXTURE_BUFFER_CASE(rgb8Uint, "RGB8Uint", 47)
FMT_TEXTURE_BUFFER_CASE(rgb8Sint, "RGB8Sint", 48)
FMT_TEXTURE_BUFFER_CASE(rgb16Unorm, "RGB16Unorm", 95)
FMT_TEXTURE_BUFFER_CASE(rgb16Snorm, "RGB16Snorm", 96)
FMT_TEXTURE_BUFFER_CASE(rgb16Uint, "RGB16Uint", 97)
FMT_TEXTURE_BUFFER_CASE(rgb16Sint, "RGB16Sint", 98)
FMT_TEXTURE_BUFFER_CASE(rgb16Float, "RGB16Float", 99)
FMT_TEXTURE_BUFFER_CASE(rgb32Uint, "RGB32Uint", 120)
FMT_TEXTURE_BUFFER_CASE(rgb32Sint, "RGB32Sint", 121)
FMT_TEXTURE_BUFFER_CASE(rgb32Float, "RGB32Float", 122)
