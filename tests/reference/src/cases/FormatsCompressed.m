// Compressed pixel-format cases: block upload and readback, copies between
// compressed textures, and the decoded values a shader read returns.
//
// Availability per GPU family comes from the Metal Feature Set Tables
// (2026-10-23), "Texture capabilities by pixel format": PVRTC, EAC/ETC and
// ASTC (LDR) list Filter from Apple2, HDR ASTC from Apple6, and BC formats
// depend on MTLDevice.supportsBCTextureCompression for Apple7 and Apple8
// and are listed from Apple9.

#import "WMRefCase.h"

typedef NS_ENUM(uint8_t, CmpFamily) { CmpETC, CmpASTC, CmpASTCHDR, CmpBC, CmpPVRTC };

typedef struct {
    NSUInteger format; // raw MTLPixelFormat value; the PVRTC names are deprecated in the SDK
    uint8_t blockWidth, blockHeight, blockBytes;
    CmpFamily family;
} CmpInfo;

static const CmpInfo kCompressed[] = {
    {MTLPixelFormatEAC_R11Unorm, 4, 4, 8, CmpETC},     {MTLPixelFormatEAC_R11Snorm, 4, 4, 8, CmpETC},
    {MTLPixelFormatEAC_RG11Unorm, 4, 4, 16, CmpETC},   {MTLPixelFormatEAC_RG11Snorm, 4, 4, 16, CmpETC},
    {MTLPixelFormatEAC_RGBA8, 4, 4, 16, CmpETC},       {MTLPixelFormatEAC_RGBA8_sRGB, 4, 4, 16, CmpETC},
    {MTLPixelFormatETC2_RGB8, 4, 4, 8, CmpETC},        {MTLPixelFormatETC2_RGB8_sRGB, 4, 4, 8, CmpETC},
    {MTLPixelFormatETC2_RGB8A1, 4, 4, 8, CmpETC},      {MTLPixelFormatETC2_RGB8A1_sRGB, 4, 4, 8, CmpETC},
    {MTLPixelFormatASTC_4x4_LDR, 4, 4, 16, CmpASTC},   {MTLPixelFormatASTC_5x4_LDR, 5, 4, 16, CmpASTC},
    {MTLPixelFormatASTC_5x5_LDR, 5, 5, 16, CmpASTC},   {MTLPixelFormatASTC_6x5_LDR, 6, 5, 16, CmpASTC},
    {MTLPixelFormatASTC_6x6_LDR, 6, 6, 16, CmpASTC},   {MTLPixelFormatASTC_8x5_LDR, 8, 5, 16, CmpASTC},
    {MTLPixelFormatASTC_8x6_LDR, 8, 6, 16, CmpASTC},   {MTLPixelFormatASTC_8x8_LDR, 8, 8, 16, CmpASTC},
    {MTLPixelFormatASTC_10x5_LDR, 10, 5, 16, CmpASTC}, {MTLPixelFormatASTC_10x6_LDR, 10, 6, 16, CmpASTC},
    {MTLPixelFormatASTC_10x8_LDR, 10, 8, 16, CmpASTC}, {MTLPixelFormatASTC_10x10_LDR, 10, 10, 16, CmpASTC},
    {MTLPixelFormatASTC_12x10_LDR, 12, 10, 16, CmpASTC}, {MTLPixelFormatASTC_12x12_LDR, 12, 12, 16, CmpASTC},
    {MTLPixelFormatASTC_4x4_sRGB, 4, 4, 16, CmpASTC},  {MTLPixelFormatASTC_5x4_sRGB, 5, 4, 16, CmpASTC},
    {MTLPixelFormatASTC_5x5_sRGB, 5, 5, 16, CmpASTC},  {MTLPixelFormatASTC_6x5_sRGB, 6, 5, 16, CmpASTC},
    {MTLPixelFormatASTC_6x6_sRGB, 6, 6, 16, CmpASTC},  {MTLPixelFormatASTC_8x5_sRGB, 8, 5, 16, CmpASTC},
    {MTLPixelFormatASTC_8x6_sRGB, 8, 6, 16, CmpASTC},  {MTLPixelFormatASTC_8x8_sRGB, 8, 8, 16, CmpASTC},
    {MTLPixelFormatASTC_10x5_sRGB, 10, 5, 16, CmpASTC}, {MTLPixelFormatASTC_10x6_sRGB, 10, 6, 16, CmpASTC},
    {MTLPixelFormatASTC_10x8_sRGB, 10, 8, 16, CmpASTC}, {MTLPixelFormatASTC_10x10_sRGB, 10, 10, 16, CmpASTC},
    {MTLPixelFormatASTC_12x10_sRGB, 12, 10, 16, CmpASTC}, {MTLPixelFormatASTC_12x12_sRGB, 12, 12, 16, CmpASTC},
    {MTLPixelFormatASTC_4x4_HDR, 4, 4, 16, CmpASTCHDR}, {MTLPixelFormatASTC_5x4_HDR, 5, 4, 16, CmpASTCHDR},
    {MTLPixelFormatASTC_5x5_HDR, 5, 5, 16, CmpASTCHDR}, {MTLPixelFormatASTC_6x5_HDR, 6, 5, 16, CmpASTCHDR},
    {MTLPixelFormatASTC_6x6_HDR, 6, 6, 16, CmpASTCHDR}, {MTLPixelFormatASTC_8x5_HDR, 8, 5, 16, CmpASTCHDR},
    {MTLPixelFormatASTC_8x6_HDR, 8, 6, 16, CmpASTCHDR}, {MTLPixelFormatASTC_8x8_HDR, 8, 8, 16, CmpASTCHDR},
    {MTLPixelFormatASTC_10x5_HDR, 10, 5, 16, CmpASTCHDR}, {MTLPixelFormatASTC_10x6_HDR, 10, 6, 16, CmpASTCHDR},
    {MTLPixelFormatASTC_10x8_HDR, 10, 8, 16, CmpASTCHDR}, {MTLPixelFormatASTC_10x10_HDR, 10, 10, 16, CmpASTCHDR},
    {MTLPixelFormatASTC_12x10_HDR, 12, 10, 16, CmpASTCHDR}, {MTLPixelFormatASTC_12x12_HDR, 12, 12, 16, CmpASTCHDR},
    {MTLPixelFormatBC1_RGBA, 4, 4, 8, CmpBC},          {MTLPixelFormatBC1_RGBA_sRGB, 4, 4, 8, CmpBC},
    {MTLPixelFormatBC2_RGBA, 4, 4, 16, CmpBC},         {MTLPixelFormatBC2_RGBA_sRGB, 4, 4, 16, CmpBC},
    {MTLPixelFormatBC3_RGBA, 4, 4, 16, CmpBC},         {MTLPixelFormatBC3_RGBA_sRGB, 4, 4, 16, CmpBC},
    {MTLPixelFormatBC4_RUnorm, 4, 4, 8, CmpBC},        {MTLPixelFormatBC4_RSnorm, 4, 4, 8, CmpBC},
    {MTLPixelFormatBC5_RGUnorm, 4, 4, 16, CmpBC},      {MTLPixelFormatBC5_RGSnorm, 4, 4, 16, CmpBC},
    {MTLPixelFormatBC6H_RGBFloat, 4, 4, 16, CmpBC},    {MTLPixelFormatBC6H_RGBUfloat, 4, 4, 16, CmpBC},
    {MTLPixelFormatBC7_RGBAUnorm, 4, 4, 16, CmpBC},    {MTLPixelFormatBC7_RGBAUnorm_sRGB, 4, 4, 16, CmpBC},
    // PVRTC: 2 bpp formats use 8x4 blocks and 4 bpp formats 4x4 blocks, 8
    // bytes each.
    {160, 8, 4, 8, CmpPVRTC}, {161, 8, 4, 8, CmpPVRTC}, {162, 4, 4, 8, CmpPVRTC}, {163, 4, 4, 8, CmpPVRTC},
    {164, 8, 4, 8, CmpPVRTC}, {165, 8, 4, 8, CmpPVRTC}, {166, 4, 4, 8, CmpPVRTC}, {167, 4, 4, 8, CmpPVRTC},
};

static int CmpAppleFamily(id<MTLDevice> device) {
    for (int n = 11; n >= 1; n--)
        if ([device supportsFamily:(MTLGPUFamily)(1000 + n)]) return n;
    return 0;
}

static void CmpRequire(WMCase *c, const CmpInfo *info) {
    int family = CmpAppleFamily(c.device);
    switch (info->family) {
    case CmpETC: case CmpASTC: case CmpPVRTC:
        [c require:@"feature-table.available" value:family >= 2];
        break;
    case CmpASTCHDR:
        [c require:@"feature-table.available" value:family >= 6];
        break;
    case CmpBC:
        [c require:@"supports-bc-texture-compression" value:c.device.supportsBCTextureCompression];
        break;
    }
}

static id<MTLBuffer> CmpRead(WMCase *c, id<MTLTexture> texture) {
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"fmt_read_float"];
    NSUInteger w = texture.width, h = texture.height;
    id<MTLBuffer> out = [c buffer:w * h * 16];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setTexture:texture atIndex:0];
        [e setBuffer:out offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    }];
    return out;
}

static WMRule *CmpDecodedRule(void) {
    return WMNone(@"Metal does not state the output precision of compressed-format decoding for shader reads");
}

static void CmpBlocks(WMCase *c, NSUInteger raw) {
    const CmpInfo *info = NULL;
    for (size_t i = 0; i < sizeof kCompressed / sizeof kCompressed[0]; i++)
        if (kCompressed[i].format == raw) info = &kCompressed[i];
    if (!info) [c fail:@"no description for pixel format %lu", (unsigned long)raw];
    CmpRequire(c, info);
    MTLPixelFormat format = (MTLPixelFormat)raw;
    BOOL pvrtc = info->family == CmpPVRTC;
    // Two blocks in each direction when that stays within 16 texels, else
    // one; PVRTC textures are 16 x 16.
    NSUInteger blocksWide = pvrtc ? 16 / info->blockWidth : (2 * info->blockWidth <= 16 ? 2 : 1);
    NSUInteger blocksHigh = pvrtc ? 16 / info->blockHeight : (2 * info->blockHeight <= 16 ? 2 : 1);
    NSUInteger width = blocksWide * info->blockWidth, height = blocksHigh * info->blockHeight;
    NSUInteger bytesPerRow = blocksWide * info->blockBytes, length = bytesPerRow * blocksHigh;
    uint8_t blocks[512];
    WMPattern(blocks, length, 0x7000u + (uint32_t)raw);

    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width
                                                                                height:height mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [c texture:d];
    // MTLTexture replaceRegion and getBytes: bytesPerRow is the distance
    // between rows of blocks, and 0 for PVRTC.
    [c step:@"replaceRegion"];
    [texture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 slice:0 withBytes:blocks
               bytesPerRow:pvrtc ? 0 : bytesPerRow bytesPerImage:0];
    NSDictionary *layout = @{@"format": WMPixelFormatName(format), @"width": @(width), @"height": @(height),
                             @"bytes_per_row": @(pvrtc ? 0 : bytesPerRow), @"block_width": @(info->blockWidth),
                             @"block_height": @(info->blockHeight), @"block_bytes": @(info->blockBytes)};
    [c data:@"input" elem:WMElemU8 bytes:blocks length:length layout:layout rule:WMExact()];

    uint8_t cpu[512];
    [c step:@"getBytes"];
    [texture getBytes:cpu bytesPerRow:pvrtc ? 0 : bytesPerRow bytesPerImage:0
           fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 slice:0];
    [c data:@"getbytes" elem:WMElemU8 bytes:cpu length:length layout:layout rule:WMExact()];

    // Blit copies of compressed blocks. PVRTC copies between a texture and
    // a buffer require MTLBlitOptionRowLinearPVRTC (the debug layer rejects
    // them otherwise) and store the blocks in row-linear order.
    id<MTLBuffer> readback = [c buffer:length];
    [c step:@"copyFromTexture:toBuffer:"];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(width, height, 1) toBuffer:readback destinationOffset:0
    destinationBytesPerRow:bytesPerRow destinationBytesPerImage:length
                   options:pvrtc ? MTLBlitOptionRowLinearPVRTC : MTLBlitOptionNone];
    }];
    [c data:pvrtc ? @"readback-row-linear" : @"readback" elem:WMElemU8 bytes:readback.contents length:length
       rule:WMExact()];

    MTLTextureDescriptor *copyDescriptor = [d copy];
    copyDescriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> copy = [c texture:copyDescriptor];
    id<MTLBuffer> copyReadback = [c buffer:length];
    [c step:@"copyFromTexture:toTexture:"];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromTexture:texture toTexture:copy];
        [e copyFromTexture:copy sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(width, height, 1) toBuffer:copyReadback destinationOffset:0
    destinationBytesPerRow:bytesPerRow destinationBytesPerImage:length
                   options:pvrtc ? MTLBlitOptionRowLinearPVRTC : MTLBlitOptionNone];
    }];
    [c value:@"copy-equals-source" boolean:memcmp(copyReadback.contents, readback.contents, length) == 0
        rule:WMExact()];

    [c step:@"read"];
    id<MTLBuffer> decoded = CmpRead(c, texture);
    [c data:@"read.decoded" elem:WMElemF32 bytes:decoded.contents length:decoded.length
     layout:@{@"width": @(width), @"height": @(height), @"components": @4} rule:CmpDecodedRule()];
}

#define CMP_CASE(fn, name, raw)                                                                          \
    WMREF_CASE(fn, "format." name ".blocks", "format.compressed",                                       \
               "Block upload, readback, copy and decoded read of " name)                                \
    {                                                                                                   \
        CmpBlocks(c, raw);                                                                              \
    }

CMP_CASE(eacR11Unorm, "EAC_R11Unorm", MTLPixelFormatEAC_R11Unorm)
CMP_CASE(eacR11Snorm, "EAC_R11Snorm", MTLPixelFormatEAC_R11Snorm)
CMP_CASE(eacRG11Unorm, "EAC_RG11Unorm", MTLPixelFormatEAC_RG11Unorm)
CMP_CASE(eacRG11Snorm, "EAC_RG11Snorm", MTLPixelFormatEAC_RG11Snorm)
CMP_CASE(eacRGBA8, "EAC_RGBA8", MTLPixelFormatEAC_RGBA8)
CMP_CASE(eacRGBA8sRGB, "EAC_RGBA8_sRGB", MTLPixelFormatEAC_RGBA8_sRGB)
CMP_CASE(etc2RGB8, "ETC2_RGB8", MTLPixelFormatETC2_RGB8)
CMP_CASE(etc2RGB8sRGB, "ETC2_RGB8_sRGB", MTLPixelFormatETC2_RGB8_sRGB)
CMP_CASE(etc2RGB8A1, "ETC2_RGB8A1", MTLPixelFormatETC2_RGB8A1)
CMP_CASE(etc2RGB8A1sRGB, "ETC2_RGB8A1_sRGB", MTLPixelFormatETC2_RGB8A1_sRGB)
CMP_CASE(astc4x4LDR, "ASTC_4x4_LDR", MTLPixelFormatASTC_4x4_LDR)
CMP_CASE(astc5x4LDR, "ASTC_5x4_LDR", MTLPixelFormatASTC_5x4_LDR)
CMP_CASE(astc5x5LDR, "ASTC_5x5_LDR", MTLPixelFormatASTC_5x5_LDR)
CMP_CASE(astc6x5LDR, "ASTC_6x5_LDR", MTLPixelFormatASTC_6x5_LDR)
CMP_CASE(astc6x6LDR, "ASTC_6x6_LDR", MTLPixelFormatASTC_6x6_LDR)
CMP_CASE(astc8x5LDR, "ASTC_8x5_LDR", MTLPixelFormatASTC_8x5_LDR)
CMP_CASE(astc8x6LDR, "ASTC_8x6_LDR", MTLPixelFormatASTC_8x6_LDR)
CMP_CASE(astc8x8LDR, "ASTC_8x8_LDR", MTLPixelFormatASTC_8x8_LDR)
CMP_CASE(astc10x5LDR, "ASTC_10x5_LDR", MTLPixelFormatASTC_10x5_LDR)
CMP_CASE(astc10x6LDR, "ASTC_10x6_LDR", MTLPixelFormatASTC_10x6_LDR)
CMP_CASE(astc10x8LDR, "ASTC_10x8_LDR", MTLPixelFormatASTC_10x8_LDR)
CMP_CASE(astc10x10LDR, "ASTC_10x10_LDR", MTLPixelFormatASTC_10x10_LDR)
CMP_CASE(astc12x10LDR, "ASTC_12x10_LDR", MTLPixelFormatASTC_12x10_LDR)
CMP_CASE(astc12x12LDR, "ASTC_12x12_LDR", MTLPixelFormatASTC_12x12_LDR)
CMP_CASE(astc4x4sRGB, "ASTC_4x4_sRGB", MTLPixelFormatASTC_4x4_sRGB)
CMP_CASE(astc5x4sRGB, "ASTC_5x4_sRGB", MTLPixelFormatASTC_5x4_sRGB)
CMP_CASE(astc5x5sRGB, "ASTC_5x5_sRGB", MTLPixelFormatASTC_5x5_sRGB)
CMP_CASE(astc6x5sRGB, "ASTC_6x5_sRGB", MTLPixelFormatASTC_6x5_sRGB)
CMP_CASE(astc6x6sRGB, "ASTC_6x6_sRGB", MTLPixelFormatASTC_6x6_sRGB)
CMP_CASE(astc8x5sRGB, "ASTC_8x5_sRGB", MTLPixelFormatASTC_8x5_sRGB)
CMP_CASE(astc8x6sRGB, "ASTC_8x6_sRGB", MTLPixelFormatASTC_8x6_sRGB)
CMP_CASE(astc8x8sRGB, "ASTC_8x8_sRGB", MTLPixelFormatASTC_8x8_sRGB)
CMP_CASE(astc10x5sRGB, "ASTC_10x5_sRGB", MTLPixelFormatASTC_10x5_sRGB)
CMP_CASE(astc10x6sRGB, "ASTC_10x6_sRGB", MTLPixelFormatASTC_10x6_sRGB)
CMP_CASE(astc10x8sRGB, "ASTC_10x8_sRGB", MTLPixelFormatASTC_10x8_sRGB)
CMP_CASE(astc10x10sRGB, "ASTC_10x10_sRGB", MTLPixelFormatASTC_10x10_sRGB)
CMP_CASE(astc12x10sRGB, "ASTC_12x10_sRGB", MTLPixelFormatASTC_12x10_sRGB)
CMP_CASE(astc12x12sRGB, "ASTC_12x12_sRGB", MTLPixelFormatASTC_12x12_sRGB)
CMP_CASE(astc4x4HDR, "ASTC_4x4_HDR", MTLPixelFormatASTC_4x4_HDR)
CMP_CASE(astc5x4HDR, "ASTC_5x4_HDR", MTLPixelFormatASTC_5x4_HDR)
CMP_CASE(astc5x5HDR, "ASTC_5x5_HDR", MTLPixelFormatASTC_5x5_HDR)
CMP_CASE(astc6x5HDR, "ASTC_6x5_HDR", MTLPixelFormatASTC_6x5_HDR)
CMP_CASE(astc6x6HDR, "ASTC_6x6_HDR", MTLPixelFormatASTC_6x6_HDR)
CMP_CASE(astc8x5HDR, "ASTC_8x5_HDR", MTLPixelFormatASTC_8x5_HDR)
CMP_CASE(astc8x6HDR, "ASTC_8x6_HDR", MTLPixelFormatASTC_8x6_HDR)
CMP_CASE(astc8x8HDR, "ASTC_8x8_HDR", MTLPixelFormatASTC_8x8_HDR)
CMP_CASE(astc10x5HDR, "ASTC_10x5_HDR", MTLPixelFormatASTC_10x5_HDR)
CMP_CASE(astc10x6HDR, "ASTC_10x6_HDR", MTLPixelFormatASTC_10x6_HDR)
CMP_CASE(astc10x8HDR, "ASTC_10x8_HDR", MTLPixelFormatASTC_10x8_HDR)
CMP_CASE(astc10x10HDR, "ASTC_10x10_HDR", MTLPixelFormatASTC_10x10_HDR)
CMP_CASE(astc12x10HDR, "ASTC_12x10_HDR", MTLPixelFormatASTC_12x10_HDR)
CMP_CASE(astc12x12HDR, "ASTC_12x12_HDR", MTLPixelFormatASTC_12x12_HDR)
CMP_CASE(bc1RGBA, "BC1_RGBA", MTLPixelFormatBC1_RGBA)
CMP_CASE(bc1RGBAsRGB, "BC1_RGBA_sRGB", MTLPixelFormatBC1_RGBA_sRGB)
CMP_CASE(bc2RGBA, "BC2_RGBA", MTLPixelFormatBC2_RGBA)
CMP_CASE(bc2RGBAsRGB, "BC2_RGBA_sRGB", MTLPixelFormatBC2_RGBA_sRGB)
CMP_CASE(bc3RGBA, "BC3_RGBA", MTLPixelFormatBC3_RGBA)
CMP_CASE(bc3RGBAsRGB, "BC3_RGBA_sRGB", MTLPixelFormatBC3_RGBA_sRGB)
CMP_CASE(bc4RUnorm, "BC4_RUnorm", MTLPixelFormatBC4_RUnorm)
CMP_CASE(bc4RSnorm, "BC4_RSnorm", MTLPixelFormatBC4_RSnorm)
CMP_CASE(bc5RGUnorm, "BC5_RGUnorm", MTLPixelFormatBC5_RGUnorm)
CMP_CASE(bc5RGSnorm, "BC5_RGSnorm", MTLPixelFormatBC5_RGSnorm)
CMP_CASE(bc6hRGBFloat, "BC6H_RGBFloat", MTLPixelFormatBC6H_RGBFloat)
CMP_CASE(bc6hRGBUfloat, "BC6H_RGBUfloat", MTLPixelFormatBC6H_RGBUfloat)
CMP_CASE(bc7RGBAUnorm, "BC7_RGBAUnorm", MTLPixelFormatBC7_RGBAUnorm)
CMP_CASE(bc7RGBAUnormsRGB, "BC7_RGBAUnorm_sRGB", MTLPixelFormatBC7_RGBAUnorm_sRGB)
CMP_CASE(pvrtcRGB2, "PVRTC_RGB_2BPP", 160)
CMP_CASE(pvrtcRGB2sRGB, "PVRTC_RGB_2BPP_sRGB", 161)
CMP_CASE(pvrtcRGB4, "PVRTC_RGB_4BPP", 162)
CMP_CASE(pvrtcRGB4sRGB, "PVRTC_RGB_4BPP_sRGB", 163)
CMP_CASE(pvrtcRGBA2, "PVRTC_RGBA_2BPP", 164)
CMP_CASE(pvrtcRGBA2sRGB, "PVRTC_RGBA_2BPP_sRGB", 165)
CMP_CASE(pvrtcRGBA4, "PVRTC_RGBA_4BPP", 166)
CMP_CASE(pvrtcRGBA4sRGB, "PVRTC_RGBA_4BPP_sRGB", 167)
