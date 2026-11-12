// Heap and resource cases: automatic and placement heaps, heap size and
// alignment queries, aliasing state, purgeable-state transitions and memory
// accounting.

#import "WMRefCase.h"

// Uploads `length` bytes into a private buffer, increments every word with
// `sto_add_one` and downloads the result. Each step waits for completion:
// heap resources are untracked by default (MTLResource.h), so the CPU wait
// is what orders the passes.
static NSData *HeapsRoundTrip(WMCase *c, id<MTLBuffer> target, NSUInteger length, uint32_t seed) {
    id<MTLBuffer> upload = [c buffer:length], download = [c buffer:length];
    WMPattern(upload.contents, length, seed);
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:upload sourceOffset:0 toBuffer:target destinationOffset:0 size:length];
    }];
    id<MTLComputePipelineState> add = [c computePipeline:@"sto_add_one"];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:add];
        [e setBuffer:target offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(length / 4 / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:target sourceOffset:0 toBuffer:download destinationOffset:0 size:length];
    }];
    return [NSData dataWithBytes:download.contents length:length];
}

static MTLTextureDescriptor *HeapsTextureDescriptor(void) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Uint
                                                                                         width:16 height:16
                                                                                     mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead;
    return descriptor;
}

static NSData *HeapsTextureRoundTrip(WMCase *c, id<MTLTexture> texture, uint32_t seed) {
    NSUInteger length = 16 * 16 * 4;
    id<MTLBuffer> upload = [c buffer:length];
    WMPattern(upload.contents, length, seed);
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:upload sourceOffset:0 sourceBytesPerRow:64 sourceBytesPerImage:length
               sourceSize:MTLSizeMake(16, 16, 1) toTexture:texture destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    }];
    return [c readTexture:texture level:0 slice:0 bytesPerRow:64];
}

// MTLHeap.h: type "The default value is MTLHeapTypeAutomatic";
// hazardTrackingMode "The default value is MTLHazardTrackingModeDefault ...
// treated as MTLHazardTrackingModeUntracked"; newBufferWithLength:options:
// "@return The buffer or nil if heap is full."
WMREF_CASE(heapAutomatic, "heap.automatic", "heap.automatic",
           "Automatic heap: descriptor defaults, buffer and texture sub-allocations in use, accounting")
{
    MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
    [c value:@"descriptor-default.type" i64:descriptor.type rule:WMExact()];
    [c value:@"descriptor-default.hazard-tracking-mode" u64:descriptor.hazardTrackingMode rule:WMExact()];
    [c value:@"descriptor-default.cpu-cache-mode" u64:descriptor.cpuCacheMode rule:WMExact()];
    [c value:@"descriptor-default.storage-mode" u64:descriptor.storageMode rule:WMDevice()];
    descriptor.size = 256 * 1024;
    descriptor.storageMode = MTLStorageModePrivate;
    id<MTLHeap> heap = [c.device newHeapWithDescriptor:descriptor];
    if (!heap) [c fail:@"newHeapWithDescriptor returned nil"];
    [c value:@"heap.type" i64:heap.type rule:WMExact()];
    [c value:@"heap.storage-mode" u64:heap.storageMode rule:WMExact()];
    [c value:@"heap.hazard-tracking-mode" u64:heap.hazardTrackingMode rule:WMExact()];
    [c value:@"heap.size" u64:heap.size rule:WMDevice()];
    [c value:@"heap.size-at-least-requested" boolean:heap.size >= descriptor.size rule:WMExact()];
    [c value:@"heap.used-size.empty" u64:heap.usedSize rule:WMDevice()];
    [c value:@"heap.current-allocated-size" u64:heap.currentAllocatedSize rule:WMDevice()];
    [c value:@"heap.max-available.align-256" u64:[heap maxAvailableSizeWithAlignment:256] rule:WMDevice()];

    id<MTLBuffer> buffer = [heap newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
    if (!buffer) [c fail:@"heap buffer returned nil"];
    [c value:@"buffer.heap-is-heap" boolean:buffer.heap == heap rule:WMExact()];
    [c value:@"buffer.hazard-tracking-mode" u64:buffer.hazardTrackingMode rule:WMExact()];
    [c value:@"buffer.allocated-size" u64:buffer.allocatedSize rule:WMDevice()];
    [c value:@"heap.used-size.after-buffer" u64:heap.usedSize rule:WMDevice()];
    id<MTLTexture> texture = [heap newTextureWithDescriptor:HeapsTextureDescriptor()];
    if (!texture) [c fail:@"heap texture returned nil"];
    [c value:@"texture.heap-is-heap" boolean:texture.heap == heap rule:WMExact()];
    [c value:@"heap.used-size.after-texture" u64:heap.usedSize rule:WMDevice()];
    [c value:@"heap.max-available.after-texture" u64:[heap maxAvailableSizeWithAlignment:256] rule:WMDevice()];

    NSData *bufferResult = HeapsRoundTrip(c, buffer, 4096, 151);
    [c data:@"buffer.round-trip" elem:WMElemU32 bytes:bufferResult.bytes length:bufferResult.length rule:WMExact()];
    NSData *textureResult = HeapsTextureRoundTrip(c, texture, 152);
    [c data:@"texture.round-trip" elem:WMElemU8 bytes:textureResult.bytes length:textureResult.length rule:WMExact()];

    // Fill the heap until it reports full.
    NSMutableArray *fill = [NSMutableArray array];
    BOOL becameFull = NO;
    for (int i = 0; i < 64; i++) {
        id<MTLBuffer> more = [heap newBufferWithLength:16 * 1024 options:MTLResourceStorageModePrivate];
        if (!more) {
            becameFull = YES;
            break;
        }
        [fill addObject:more];
    }
    [c value:@"fill.returned-nil-when-full" boolean:becameFull rule:WMExact()];
    [c value:@"fill.buffers-before-full" u64:fill.count rule:WMDevice()];
}

// MTLHeap.h, MTLHeapTypePlacement: resources are placed at offsets the
// caller chooses (newBufferWithLength:options:offset:).
WMREF_CASE(heapPlacement, "heap.placement", "heap.placement",
           "Placement heap: buffers at chosen offsets in use, heapOffset read back")
{
    MTLSizeAndAlign sizeAndAlign = [c.device heapBufferSizeAndAlignWithLength:4096 options:MTLResourceStorageModePrivate];
    [c value:@"buffer-4096.size" u64:sizeAndAlign.size rule:WMDevice()];
    [c value:@"buffer-4096.align" u64:sizeAndAlign.align rule:WMDevice()];
    MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
    descriptor.type = MTLHeapTypePlacement;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.size = 64 * 1024;
    id<MTLHeap> heap = [c.device newHeapWithDescriptor:descriptor];
    if (!heap) [c fail:@"placement heap returned nil"];
    [c value:@"heap.type" i64:heap.type rule:WMExact()];
    NSUInteger second = (sizeAndAlign.size + sizeAndAlign.align - 1) / sizeAndAlign.align * sizeAndAlign.align;
    id<MTLBuffer> a = [heap newBufferWithLength:4096 options:MTLResourceStorageModePrivate offset:0];
    id<MTLBuffer> b = [heap newBufferWithLength:4096 options:MTLResourceStorageModePrivate offset:second];
    if (!a || !b) [c fail:@"placement buffer returned nil"];
    [c value:@"a.heap-offset" u64:a.heapOffset rule:WMExact()];
    [c value:@"b.heap-offset-is-requested" boolean:b.heapOffset == second rule:WMExact()];
    NSData *resultA = HeapsRoundTrip(c, a, 4096, 161);
    NSData *resultB = HeapsRoundTrip(c, b, 4096, 162);
    [c data:@"a.round-trip" elem:WMElemU32 bytes:resultA.bytes length:resultA.length rule:WMExact()];
    [c data:@"b.round-trip" elem:WMElemU32 bytes:resultB.bytes length:resultB.length rule:WMExact()];
    MTLSizeAndAlign textureSize = [c.device heapTextureSizeAndAlignWithDescriptor:HeapsTextureDescriptor()];
    NSUInteger third = (second + 4096 + textureSize.align - 1) / textureSize.align * textureSize.align;
    id<MTLTexture> texture = [heap newTextureWithDescriptor:HeapsTextureDescriptor() offset:third];
    if (!texture) [c fail:@"placement texture returned nil"];
    [c value:@"texture.heap-offset-is-requested" boolean:texture.heapOffset == third rule:WMExact()];
    NSData *textureResult = HeapsTextureRoundTrip(c, texture, 163);
    [c data:@"texture.round-trip" elem:WMElemU8 bytes:textureResult.bytes length:textureResult.length rule:WMExact()];
}

// MTLDevice.h, heapBufferSizeAndAlignWithLength:options: and
// heapTextureSizeAndAlignWithDescriptor: "Determine the byte size of
// buffers/textures when sub-allocated from a heap." The numbers belong to
// the device.
WMREF_CASE(heapSizeAndAlign, "heap.size-and-align", "heap.queries",
           "Heap size and alignment answers for buffers and textures")
{
    const NSUInteger lengths[] = {1, 256, 4096, 4097, 65536, 1 << 20};
    for (size_t i = 0; i < sizeof lengths / sizeof lengths[0]; i++) {
        for (int private = 0; private < 2; private++) {
            MTLResourceOptions options = private ? MTLResourceStorageModePrivate : MTLResourceStorageModeShared;
            MTLSizeAndAlign answer = [c.device heapBufferSizeAndAlignWithLength:lengths[i] options:options];
            NSString *name = [NSString stringWithFormat:@"buffer.%lu.%@", (unsigned long)lengths[i],
                              private ? @"private" : @"shared"];
            [c value:[name stringByAppendingString:@".size"] u64:answer.size rule:WMDevice()];
            [c value:[name stringByAppendingString:@".align"] u64:answer.align rule:WMDevice()];
        }
    }
    struct { const char *name; MTLPixelFormat format; NSUInteger width, height; BOOL mipmapped; } textures[] = {
        {"rgba8-64x64", MTLPixelFormatRGBA8Unorm, 64, 64, NO},
        {"rgba8-64x64-mipmapped", MTLPixelFormatRGBA8Unorm, 64, 64, YES},
        {"r32f-100x30", MTLPixelFormatR32Float, 100, 30, NO},
        {"depth32f-128x128", MTLPixelFormatDepth32Float, 128, 128, NO},
        {"rgba16f-1x1", MTLPixelFormatRGBA16Float, 1, 1, NO},
    };
    for (size_t i = 0; i < sizeof textures / sizeof textures[0]; i++) {
        MTLTextureDescriptor *descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:textures[i].format width:textures[i].width
                                                              height:textures[i].height
                                                           mipmapped:textures[i].mipmapped];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = textures[i].format == MTLPixelFormatDepth32Float ? MTLTextureUsageRenderTarget
                                                                            : MTLTextureUsageShaderRead;
        MTLSizeAndAlign answer = [c.device heapTextureSizeAndAlignWithDescriptor:descriptor];
        NSString *name = [NSString stringWithFormat:@"texture.%s", textures[i].name];
        [c value:[name stringByAppendingString:@".size"] u64:answer.size rule:WMDevice()];
        [c value:[name stringByAppendingString:@".align"] u64:answer.align rule:WMDevice()];
    }
}

// MTLResource.h, isAliasable: "YES if makeAliasable was previously
// successfully called on this resource. NO otherwise."
WMREF_CASE(heapAliasable, "heap.make-aliasable", "heap.aliasing",
           "isAliasable before and after makeAliasable on heap buffers and textures")
{
    MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
    descriptor.size = 256 * 1024;
    descriptor.storageMode = MTLStorageModePrivate;
    id<MTLHeap> heap = [c.device newHeapWithDescriptor:descriptor];
    if (!heap) [c fail:@"newHeapWithDescriptor returned nil"];
    id<MTLBuffer> buffer = [heap newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
    id<MTLTexture> texture = [heap newTextureWithDescriptor:HeapsTextureDescriptor()];
    if (!buffer || !texture) [c fail:@"heap resource returned nil"];
    [c value:@"buffer.before" boolean:buffer.isAliasable rule:WMExact()];
    [c value:@"texture.before" boolean:texture.isAliasable rule:WMExact()];
    NSUInteger usedBefore = heap.usedSize;
    [buffer makeAliasable];
    [texture makeAliasable];
    [c value:@"buffer.after" boolean:buffer.isAliasable rule:WMExact()];
    [c value:@"texture.after" boolean:texture.isAliasable rule:WMExact()];
    [c value:@"used-size.before" u64:usedBefore rule:WMDevice()];
    [c value:@"used-size.after" u64:heap.usedSize rule:WMDevice()];
}

// MTLResource.h, makeAliasable: "It is illegal to call this method on a
// non heap-based resource ... The debug layer will raise an exception."
WMREF_CASE(aliasableNonHeap, "resource.make-aliasable-non-heap", "heap.errors",
           "Error path: makeAliasable on a buffer created from the device")
{
    id<MTLBuffer> buffer = [c buffer:4096];
    [c step:@"make-aliasable"];
    [c call:@"make-aliasable" rule:WMExact() block:^{
        [buffer makeAliasable];
    }];
    [c value:@"is-aliasable-after" boolean:buffer.isAliasable
        rule:WMNone(@"MTLResource.h calls the use illegal and does not define its effect")];
}

// MTLResource.h, setPurgeableState: "Synchronously set the purgeability
// state of a resource and return what the prior (or current) state is."
// Volatile contents "may be discarded", so after a volatile state the
// answer is volatile or empty, depending on whether the system discarded
// the contents in the meantime; that answer is recorded as the property
// "volatile or empty", which holds on every run.
static BOOL HeapsVolatileOrEmpty(MTLPurgeableState state) {
    return state == MTLPurgeableStateVolatile || state == MTLPurgeableStateEmpty;
}

static void HeapsPurgeableSequence(WMCase *c, NSString *prefix, id<MTLResource> resource) {
    struct { const char *name; MTLPurgeableState set; BOOL afterVolatile; } steps[] = {
        {"query-initial", MTLPurgeableStateKeepCurrent, NO},
        {"set-volatile", MTLPurgeableStateVolatile, NO},
        {"query-volatile", MTLPurgeableStateKeepCurrent, YES},
        {"set-nonvolatile", MTLPurgeableStateNonVolatile, YES},
        {"query-nonvolatile", MTLPurgeableStateKeepCurrent, NO},
        {"set-empty", MTLPurgeableStateEmpty, NO},
        {"query-empty", MTLPurgeableStateKeepCurrent, NO},
        {"set-nonvolatile-after-empty", MTLPurgeableStateNonVolatile, NO},
        {"query-final", MTLPurgeableStateKeepCurrent, NO},
    };
    for (size_t i = 0; i < sizeof steps / sizeof steps[0]; i++) {
        MTLPurgeableState answer = [resource setPurgeableState:steps[i].set];
        NSString *name = [NSString stringWithFormat:@"%@.%s", prefix, steps[i].name];
        if (steps[i].afterVolatile)
            [c value:[name stringByAppendingString:@".volatile-or-empty"] boolean:HeapsVolatileOrEmpty(answer)
                rule:WMExact()];
        else
            [c value:name u64:answer rule:WMExact()];
    }
}

WMREF_CASE(purgeableState, "resource.purgeable-state", "resource.purgeable",
           "setPurgeableState: answers through volatile, nonvolatile and empty for a buffer, texture and heap")
{
    HeapsPurgeableSequence(c, @"shared-buffer", [c buffer:64 * 1024]);
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                         width:64 height:64
                                                                                     mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    HeapsPurgeableSequence(c, @"private-texture", [c texture:descriptor]);
    MTLHeapDescriptor *heapDescriptor = [MTLHeapDescriptor new];
    heapDescriptor.size = 64 * 1024;
    heapDescriptor.storageMode = MTLStorageModePrivate;
    id<MTLHeap> heap = [c.device newHeapWithDescriptor:heapDescriptor];
    if (!heap) [c fail:@"newHeapWithDescriptor returned nil"];
    [c value:@"heap.query-initial" u64:[heap setPurgeableState:MTLPurgeableStateKeepCurrent] rule:WMExact()];
    [c value:@"heap.set-volatile" u64:[heap setPurgeableState:MTLPurgeableStateVolatile] rule:WMExact()];
    [c value:@"heap.set-nonvolatile.volatile-or-empty"
        boolean:HeapsVolatileOrEmpty([heap setPurgeableState:MTLPurgeableStateNonVolatile]) rule:WMExact()];
    [c value:@"heap.query-final" u64:[heap setPurgeableState:MTLPurgeableStateKeepCurrent] rule:WMExact()];
}

// MTLResource.h, allocatedSize: "The size in bytes occupied by this
// resource"; MTLDevice.h, currentAllocatedSize: "The current size in bytes
// of all resources allocated by this device". Sizes are device values; the
// growth of the device total by at least a live allocation follows from the
// definition.
WMREF_CASE(allocatedSize, "resource.allocated-size", "resource.accounting",
           "allocatedSize of buffers and textures; currentAllocatedSize growth for a live allocation")
{
    NSUInteger before = c.device.currentAllocatedSize;
    id<MTLBuffer> large = [c.device newBufferWithLength:4 << 20 options:MTLResourceStorageModePrivate];
    if (!large) [c fail:@"4 MiB buffer returned nil"];
    NSUInteger after = c.device.currentAllocatedSize;
    [c value:@"device.current-allocated-size-grew-by-length" boolean:after >= before + large.length rule:WMExact()];
    const NSUInteger lengths[] = {1, 100, 4096, 4097, 65536, 1 << 20};
    for (size_t i = 0; i < sizeof lengths / sizeof lengths[0]; i++) {
        id<MTLBuffer> buffer = [c.device newBufferWithLength:lengths[i] options:MTLResourceStorageModeShared];
        if (!buffer) [c fail:@"buffer returned nil"];
        [c value:[NSString stringWithFormat:@"buffer.%lu.length", (unsigned long)lengths[i]] u64:buffer.length
            rule:WMExact()];
        [c value:[NSString stringWithFormat:@"buffer.%lu.allocated-size", (unsigned long)lengths[i]]
             u64:buffer.allocatedSize rule:WMDevice()];
    }
    for (int mipmapped = 0; mipmapped < 2; mipmapped++) {
        MTLTextureDescriptor *descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:64 height:64
                                                           mipmapped:mipmapped];
        descriptor.storageMode = MTLStorageModePrivate;
        id<MTLTexture> texture = [c texture:descriptor];
        [c value:[NSString stringWithFormat:@"texture.rgba8-64x64%@.allocated-size", mipmapped ? @"-mipmapped" : @""]
             u64:texture.allocatedSize rule:WMDevice()];
    }
    (void)large;
}

// MTLDevice.h memory properties; all are answers of the device.
WMREF_CASE(deviceMemory, "resource.device-memory", "resource.accounting",
           "recommendedMaxWorkingSetSize, hasUnifiedMemory, maxBufferLength and residency set support")
{
    [c value:@"recommended-max-working-set-size" u64:c.device.recommendedMaxWorkingSetSize rule:WMDevice()];
    [c value:@"has-unified-memory" boolean:c.device.hasUnifiedMemory rule:WMDevice()];
    [c value:@"max-buffer-length" u64:c.device.maxBufferLength rule:WMDevice()];
    BOOL residencySet = NO;
    if (@available(macOS 15.0, iOS 18.0, *)) {
        MTLResidencySetDescriptor *descriptor = [MTLResidencySetDescriptor new];
        NSError *error = nil;
        residencySet = [c.device newResidencySetWithDescriptor:descriptor error:&error] != nil;
    }
    [c value:@"residency-set-created" boolean:residencySet rule:WMDevice()];
}
