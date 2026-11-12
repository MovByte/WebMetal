// Device cases: identity, GPU family and feature set answers, public
// capability getters and limits, alignment and size queries, default sample
// positions and counter sets. Every value here is a property of the device
// and carries the device rule.

#import "WMRefCase.h"

#include <TargetConditionals.h>

WMREF_CASE(deviceIdentity, "device.identity", "device.identity",
           "Device name, architecture and memory model")
{
    [c value:@"name" string:c.device.name rule:WMDevice()];
    if (@available(macOS 14.0, iOS 17.0, *))
        [c value:@"architecture" string:c.device.architecture.name rule:WMDevice()];
    [c value:@"has-unified-memory" boolean:c.device.hasUnifiedMemory rule:WMDevice()];
    // registryID and currentAllocatedSize change between boots and runs; they
    // are not recorded.
#if TARGET_OS_OSX
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [c value:@"low-power" boolean:c.device.lowPower rule:WMDevice()];
    [c value:@"headless" boolean:c.device.headless rule:WMDevice()];
    [c value:@"removable" boolean:c.device.removable rule:WMDevice()];
    [c value:@"location" u64:(uint64_t)c.device.location rule:WMDevice()];
    [c value:@"location-number" u64:c.device.locationNumber rule:WMDevice()];
    [c value:@"max-transfer-rate" u64:c.device.maxTransferRate rule:WMDevice()];
    [c value:@"peer-group-id" u64:c.device.peerGroupID rule:WMDevice()];
    [c value:@"peer-index" u64:c.device.peerIndex rule:WMDevice()];
    [c value:@"peer-count" u64:c.device.peerCount rule:WMDevice()];
#pragma clang diagnostic pop
#endif
}

WMREF_CASE(deviceFamilies, "device.families", "device.families",
           "supportsFamily: for every MTLGPUFamily and supportsFeatureSet: for every feature set of the platform")
{
    // MTLGPUFamily values from MTLDevice.h (SDK 27.2).
    static const struct { NSInteger value; const char *name; } families[] = {
        {1001, "apple1"}, {1002, "apple2"}, {1003, "apple3"}, {1004, "apple4"}, {1005, "apple5"},
        {1006, "apple6"}, {1007, "apple7"}, {1008, "apple8"}, {1009, "apple9"}, {1010, "apple10"},
        {1011, "apple11"}, {2001, "mac1"}, {2002, "mac2"}, {3001, "common1"}, {3002, "common2"},
        {3003, "common3"}, {4001, "mac-catalyst1"}, {4002, "mac-catalyst2"}, {5001, "metal3"},
    };
    for (size_t i = 0; i < sizeof families / sizeof families[0]; i++)
        [c value:[NSString stringWithFormat:@"family.%s", families[i].name]
         boolean:[c.device supportsFamily:(MTLGPUFamily)families[i].value] rule:WMDevice()];
    if (@available(macOS 26.0, iOS 26.0, *))
        [c value:@"family.metal4" boolean:[c.device supportsFamily:MTLGPUFamilyMetal4] rule:WMDevice()];

    // MTLFeatureSet values from MTLDevice.h; the enumeration is deprecated
    // and its values are platform specific.
#if TARGET_OS_OSX
    static const struct { NSUInteger value; const char *name; } sets[] = {
        {10000, "macOS_GPUFamily1_v1"}, {10001, "macOS_GPUFamily1_v2"}, {10002, "macOS_ReadWriteTextureTier2"},
        {10003, "macOS_GPUFamily1_v3"}, {10004, "macOS_GPUFamily1_v4"}, {10005, "macOS_GPUFamily2_v1"},
    };
#else
    static const struct { NSUInteger value; const char *name; } sets[] = {
        {0, "iOS_GPUFamily1_v1"}, {1, "iOS_GPUFamily2_v1"}, {2, "iOS_GPUFamily1_v2"}, {3, "iOS_GPUFamily2_v2"},
        {4, "iOS_GPUFamily3_v1"}, {5, "iOS_GPUFamily1_v3"}, {6, "iOS_GPUFamily2_v3"}, {7, "iOS_GPUFamily3_v2"},
        {8, "iOS_GPUFamily1_v4"}, {9, "iOS_GPUFamily2_v4"}, {10, "iOS_GPUFamily3_v3"}, {11, "iOS_GPUFamily4_v1"},
        {12, "iOS_GPUFamily1_v5"}, {13, "iOS_GPUFamily2_v5"}, {14, "iOS_GPUFamily3_v4"}, {15, "iOS_GPUFamily4_v2"},
        {16, "iOS_GPUFamily5_v1"},
    };
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    for (size_t i = 0; i < sizeof sets / sizeof sets[0]; i++)
        [c value:[NSString stringWithFormat:@"feature-set.%s", sets[i].name]
         boolean:[c.device supportsFeatureSet:(MTLFeatureSet)sets[i].value] rule:WMDevice()];
#pragma clang diagnostic pop
}

WMREF_CASE(deviceLimits, "device.limits", "device.limits",
           "Public limits and capability getters of MTLDevice")
{
    id<MTLDevice> d = c.device;
    MTLSize threads = d.maxThreadsPerThreadgroup;
    [c value:@"max-threads-per-threadgroup.width" u64:threads.width rule:WMDevice()];
    [c value:@"max-threads-per-threadgroup.height" u64:threads.height rule:WMDevice()];
    [c value:@"max-threads-per-threadgroup.depth" u64:threads.depth rule:WMDevice()];
    [c value:@"max-threadgroup-memory-length" u64:d.maxThreadgroupMemoryLength rule:WMDevice()];
    [c value:@"max-argument-buffer-sampler-count" u64:d.maxArgumentBufferSamplerCount rule:WMDevice()];
    [c value:@"max-buffer-length" u64:d.maxBufferLength rule:WMDevice()];
    [c value:@"recommended-max-working-set-size" u64:d.recommendedMaxWorkingSetSize rule:WMDevice()];
    [c value:@"argument-buffers-support" u64:(uint64_t)d.argumentBuffersSupport rule:WMDevice()];
    [c value:@"read-write-texture-support" u64:(uint64_t)d.readWriteTextureSupport rule:WMDevice()];
    [c value:@"raster-order-groups-supported" boolean:d.rasterOrderGroupsSupported rule:WMDevice()];
    [c value:@"supports-32bit-float-filtering" boolean:d.supports32BitFloatFiltering rule:WMDevice()];
    [c value:@"supports-32bit-msaa" boolean:d.supports32BitMSAA rule:WMDevice()];
    [c value:@"supports-query-texture-lod" boolean:d.supportsQueryTextureLOD rule:WMDevice()];
    [c value:@"supports-bc-texture-compression" boolean:d.supportsBCTextureCompression rule:WMDevice()];
    [c value:@"supports-pull-model-interpolation" boolean:d.supportsPullModelInterpolation rule:WMDevice()];
    [c value:@"supports-shader-barycentric-coordinates" boolean:d.supportsShaderBarycentricCoordinates
        rule:WMDevice()];
    [c value:@"programmable-sample-positions-supported" boolean:d.programmableSamplePositionsSupported
        rule:WMDevice()];
    [c value:@"supports-dynamic-libraries" boolean:d.supportsDynamicLibraries rule:WMDevice()];
    [c value:@"supports-render-dynamic-libraries" boolean:d.supportsRenderDynamicLibraries rule:WMDevice()];
    [c value:@"supports-raytracing" boolean:d.supportsRaytracing rule:WMDevice()];
    [c value:@"supports-raytracing-from-render" boolean:d.supportsRaytracingFromRender rule:WMDevice()];
    [c value:@"supports-primitive-motion-blur" boolean:d.supportsPrimitiveMotionBlur rule:WMDevice()];
    [c value:@"supports-function-pointers" boolean:d.supportsFunctionPointers rule:WMDevice()];
    [c value:@"supports-function-pointers-from-render" boolean:d.supportsFunctionPointersFromRender
        rule:WMDevice()];
    [c value:@"sparse-tile-size-in-bytes" u64:d.sparseTileSizeInBytes rule:WMDevice()];
    if (@available(macOS 26.4, iOS 26.4, *))
        [c value:@"supports-placement-sparse" boolean:d.supportsPlacementSparse rule:WMDevice()];
    for (NSUInteger layers = 1; layers <= 2; layers++)
        [c value:[NSString stringWithFormat:@"supports-rasterization-rate-map.layers-%lu", (unsigned long)layers]
         boolean:[d supportsRasterizationRateMapWithLayerCount:layers] rule:WMDevice()];
    for (NSUInteger count = 1; count <= 8; count++)
        [c value:[NSString stringWithFormat:@"supports-vertex-amplification-count.%lu", (unsigned long)count]
         boolean:[d supportsVertexAmplificationCount:count] rule:WMDevice()];
#if TARGET_OS_OSX
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [c value:@"depth24-stencil8-pixel-format-supported" boolean:d.depth24Stencil8PixelFormatSupported
        rule:WMDevice()];
#pragma clang diagnostic pop
    if (@available(macOS 13.3, *))
        [c value:@"maximum-concurrent-compilation-task-count" u64:d.maximumConcurrentCompilationTaskCount
            rule:WMDevice()];
#endif
}

WMREF_CASE(deviceSampleCounts, "device.texture-sample-counts", "device.limits",
           "supportsTextureSampleCount: for 1 through 8 and getDefaultSamplePositions:count:")
{
    for (NSUInteger count = 1; count <= 8; count++) {
        BOOL supported = [c.device supportsTextureSampleCount:count];
        [c value:[NSString stringWithFormat:@"supports.%lu", (unsigned long)count] boolean:supported
            rule:WMDevice()];
        if (!supported) continue;
        MTLSamplePosition positions[8];
        [c.device getDefaultSamplePositions:positions count:count];
        float coordinates[16];
        for (NSUInteger i = 0; i < count; i++) {
            coordinates[2 * i] = positions[i].x;
            coordinates[2 * i + 1] = positions[i].y;
        }
        [c data:[NSString stringWithFormat:@"default-sample-positions.%lu", (unsigned long)count]
           elem:WMElemF32 bytes:coordinates length:count * 2 * sizeof(float) rule:WMDevice()];
    }
}

static const MTLPixelFormat WMAlignmentFormats[] = {
    MTLPixelFormatR8Unorm, MTLPixelFormatRG8Unorm, MTLPixelFormatR16Float, MTLPixelFormatRGBA8Unorm,
    MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGB10A2Unorm, MTLPixelFormatR32Float, MTLPixelFormatRG32Uint,
    MTLPixelFormatRGBA16Float, MTLPixelFormatRGBA32Float,
};

WMREF_CASE(deviceAlignments, "device.alignments", "device.limits",
           "minimumLinearTextureAlignmentForPixelFormat: and minimumTextureBufferAlignmentForPixelFormat:")
{
    for (size_t i = 0; i < sizeof WMAlignmentFormats / sizeof WMAlignmentFormats[0]; i++) {
        MTLPixelFormat format = WMAlignmentFormats[i];
        NSString *name = WMPixelFormatName(format);
        [c value:[NSString stringWithFormat:@"linear-texture.%@", name]
             u64:[c.device minimumLinearTextureAlignmentForPixelFormat:format] rule:WMDevice()];
        [c value:[NSString stringWithFormat:@"texture-buffer.%@", name]
             u64:[c.device minimumTextureBufferAlignmentForPixelFormat:format] rule:WMDevice()];
    }
}

WMREF_CASE(deviceHeapSizes, "device.heap-size-align", "device.heap-sizes",
           "heapBufferSizeAndAlignWithLength:options: and heapTextureSizeAndAlignWithDescriptor:")
{
    const NSUInteger lengths[] = {1, 256, 4096, 65536, 65537, 1 << 20};
    const MTLResourceOptions options[] = {MTLResourceStorageModeShared, MTLResourceStorageModePrivate};
    const char *optionNames[] = {"shared", "private"};
    for (size_t o = 0; o < 2; o++)
        for (size_t i = 0; i < sizeof lengths / sizeof lengths[0]; i++) {
            MTLSizeAndAlign sa = [c.device heapBufferSizeAndAlignWithLength:lengths[i] options:options[o]];
            NSString *base = [NSString stringWithFormat:@"buffer.%s.%lu", optionNames[o], (unsigned long)lengths[i]];
            [c value:[base stringByAppendingString:@".size"] u64:sa.size rule:WMDevice()];
            [c value:[base stringByAppendingString:@".align"] u64:sa.align rule:WMDevice()];
        }
    struct { MTLTextureType type; MTLPixelFormat format; NSUInteger w, h, d, levels, samples; const char *name; }
    textures[] = {
        {MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 64, 64, 1, 1, 1, "2d-rgba8-64x64"},
        {MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 64, 64, 1, 7, 1, "2d-rgba8-64x64-mips"},
        {MTLTextureType2D, MTLPixelFormatR32Float, 100, 30, 1, 1, 1, "2d-r32f-100x30"},
        {MTLTextureType3D, MTLPixelFormatRGBA16Float, 16, 16, 16, 1, 1, "3d-rgba16f-16"},
        {MTLTextureType2DMultisample, MTLPixelFormatRGBA8Unorm, 32, 32, 1, 1, 4, "2dms4-rgba8-32x32"},
        {MTLTextureType2D, MTLPixelFormatDepth32Float, 64, 64, 1, 1, 1, "2d-depth32f-64x64"},
    };
    for (size_t i = 0; i < sizeof textures / sizeof textures[0]; i++) {
        MTLTextureDescriptor *descriptor = [MTLTextureDescriptor new];
        descriptor.textureType = textures[i].type;
        descriptor.pixelFormat = textures[i].format;
        descriptor.width = textures[i].w;
        descriptor.height = textures[i].h;
        descriptor.depth = textures[i].d;
        descriptor.mipmapLevelCount = textures[i].levels;
        descriptor.sampleCount = textures[i].samples;
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = textures[i].samples > 1 || textures[i].format == MTLPixelFormatDepth32Float
                               ? MTLTextureUsageRenderTarget
                               : MTLTextureUsageShaderRead;
        MTLSizeAndAlign sa = [c.device heapTextureSizeAndAlignWithDescriptor:descriptor];
        NSString *base = [NSString stringWithFormat:@"texture.%s", textures[i].name];
        [c value:[base stringByAppendingString:@".size"] u64:sa.size rule:WMDevice()];
        [c value:[base stringByAppendingString:@".align"] u64:sa.align rule:WMDevice()];
    }
}

WMREF_CASE(deviceCounters, "device.counter-sets", "device.counters",
           "counterSets with their counter names and supportsCounterSampling: for every sampling point")
{
    NSMutableArray<NSString *> *sets = [NSMutableArray array];
    for (id<MTLCounterSet> set in c.device.counterSets) {
        NSMutableArray<NSString *> *names = [NSMutableArray array];
        for (id<MTLCounter> counter in set.counters) [names addObject:counter.name];
        [names sortUsingSelector:@selector(compare:)];
        [sets addObject:[NSString stringWithFormat:@"%@: %@", set.name, [names componentsJoinedByString:@","]]];
    }
    [sets sortUsingSelector:@selector(compare:)];
    [c value:@"counter-sets" strings:sets rule:WMDevice()];
    static const struct { MTLCounterSamplingPoint point; const char *name; } points[] = {
        {MTLCounterSamplingPointAtStageBoundary, "stage"},
        {MTLCounterSamplingPointAtDrawBoundary, "draw"},
        {MTLCounterSamplingPointAtDispatchBoundary, "dispatch"},
        {MTLCounterSamplingPointAtTileDispatchBoundary, "tile-dispatch"},
        {MTLCounterSamplingPointAtBlitBoundary, "blit"},
    };
    for (size_t i = 0; i < sizeof points / sizeof points[0]; i++)
        [c value:[NSString stringWithFormat:@"supports-counter-sampling.%s", points[i].name]
         boolean:[c.device supportsCounterSampling:points[i].point] rule:WMDevice()];
}

WMREF_CASE(deviceSparse, "device.sparse-page-sizes", "device.limits",
           "sparseTileSizeInBytesForSparsePageSize: for each MTLSparsePageSize")
{
    // "Sparse textures": Apple6 and later (Metal Feature Set Tables, 2026-10-23).
    // sparseTileSizeInBytes is also recorded as a requirement because the getter is the device's
    // own answer.
    [c require:@"family.apple6" value:[c.device supportsFamily:MTLGPUFamilyApple6]];
    [c require:@"sparse-tile-size-nonzero" value:c.device.sparseTileSizeInBytes > 0];
    static const struct { MTLSparsePageSize size; const char *name; } sizes[] = {
        {MTLSparsePageSize16, "16"}, {MTLSparsePageSize64, "64"}, {MTLSparsePageSize256, "256"},
    };
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++)
        [c value:[NSString stringWithFormat:@"tile-size-in-bytes.%s", sizes[i].name]
             u64:[c.device sparseTileSizeInBytesForSparsePageSize:sizes[i].size] rule:WMDevice()];
    MTLSize tile = [c.device sparseTileSizeWithTextureType:MTLTextureType2D pixelFormat:MTLPixelFormatRGBA8Unorm
                                               sampleCount:1];
    [c value:@"tile.2d-rgba8.width" u64:tile.width rule:WMDevice()];
    [c value:@"tile.2d-rgba8.height" u64:tile.height rule:WMDevice()];
    [c value:@"tile.2d-rgba8.depth" u64:tile.depth rule:WMDevice()];
}
