// Host checks of the plugin's system-client SPI (guest/SPI) on this Mac, through the macOS test shim.
//
//     WebMetalSPICheck <shim.dylib>
//
// Each check calls a driver selector the plugin implements on the shim's device or on an object of the plugin class
// that answers it, and compares the answer with what 21H461's driver answers (the address of the driver method is
// given with each check); LOUD stubs that guest/SPI/notes.toml annotates must report their note. This Mac's Metal
// device is no reference for these answers: they are the A10X driver's. Prints one line per failure and PASS or FAIL.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// The driver selectors the checks send; the plugin's classes answer them.
@protocol SPIDevice
- (NSString *)vendorName;
- (NSString *)familyName;
- (NSString *)productName;
- (BOOL)disableComputeEncoderCoalescing;
- (void)setDisableComputeEncoderCoalescing:(BOOL)disable;
- (id)newVisibleFunctionTableWithDescriptor:(id)descriptor;
- (id)newIndirectComputeCommandEncoderWithBuffer:(id)buffer;
- (id<MTLTexture>)newTextureWithBytesNoCopy:(void *)bytes
                                     length:(NSUInteger)length
                                 descriptor:(MTLTextureDescriptor *)descriptor
                                deallocator:(void (^)(void *pointer, NSUInteger length))deallocator;
- (NSUInteger)gpuCoreCount;
- (id)newTextureLayoutWithDescriptor:(MTLTextureDescriptor *)descriptor isHeapOrBufferBacked:(BOOL)heapOrBufferBacked;
@end

@protocol SPITextureLayout
- (NSUInteger)size;
- (NSUInteger)alignment;
- (void)finalizeTextureMemory:(void *)memory;
- (void)copyFromLinearBytes:(const void *)bytes
               linearOffset:(NSUInteger)offset
          linearBytesPerRow:(NSUInteger)bytesPerRow
        linearBytesPerImage:(NSUInteger)bytesPerImage
            toTextureMemory:(void *)memory
               textureSlice:(NSUInteger)slice
               textureLevel:(NSUInteger)level
              textureRegion:(MTLRegion)region;
@end

@protocol SPIQueue
- (BOOL)setGPUPriority:(NSUInteger)priority;
- (BOOL)setGPUPriority:(NSUInteger)priority offset:(uint16_t)offset;
- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority;
- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority offset:(uint16_t)offset;
- (BOOL)_setGPUPriority:(NSUInteger)priority backgroundPriority:(NSUInteger)background;
- (NSUInteger)getGPUPriority;
- (NSUInteger)getBackgroundGPUPriority;
- (NSUInteger)resolveAndUpdateQueuePriority:(NSUInteger)priority;
- (NSArray *)availableCounters;
- (NSDictionary *)counterInfo;
- (NSArray *)getRequestedCounters;
- (NSArray *)subdivideCounterList:(NSArray *)counters;
- (int)requestCounters:(NSArray *)counters;
- (void)setStatEnabled:(BOOL)enabled;
@end

@protocol SPICommandBuffer
- (void)commitEncoder;
- (void)encodeCacheHintTag:(NSUInteger)tag resourceGroups:(const id __unsafe_unretained *)groups count:(NSUInteger)count;
- (void)encodeCacheHintFinalize:(NSUInteger)hint
                 resourceGroups:(const id __unsafe_unretained *)groups
                          count:(NSUInteger)count;
- (void)setResourceGroups:(const id __unsafe_unretained *)groups count:(NSUInteger)count;
- (void)encodeDashboardTagForResourceGroup:(id)group;
@end

@protocol SPIResources
- (NSUInteger)resourceSize;
- (BOOL)isComplete;
- (void)waitUntilComplete;
@end

@protocol SPIConstants
- (NSUInteger)imageblockSampleLength;
- (NSUInteger)imageBlockSampleLength;
- (NSUInteger)imageblockMemoryLengthForDimensions:(MTLSize)dimensions;
- (NSUInteger)imageBlockMemoryLengthForDimensions:(MTLSize)dimensions;
- (NSUInteger)staticThreadgroupMemoryLength;
- (NSUInteger)objectThreadExecutionWidth;
- (NSUInteger)meshThreadExecutionWidth;
- (NSUInteger)resourceIndex;
- (uint32_t)commandBatchIdOffset;
- (NSUInteger)nextVirtualSubstream;
- (BOOL)parallelExecution;
- (BOOL)setMTLCounterSampleLocationWithBuffer:(id)buffer startIndex:(uint32_t)start endIndex:(uint32_t)end;
- (NSInteger)compressionFeedback;
- (NSUInteger)firstMipmapInTail;
- (NSUInteger)tailSize;
- (NSUInteger)tailSizeInBytes;
- (NSUInteger)sparseSurfaceDefaultValue;
@end

namespace {

unsigned gFailures, gChecks;

void check(bool ok, const std::string &what)
{
    gChecks++;
    if (!ok) {
        gFailures++;
        printf("FAIL %s\n", what.c_str());
    }
}

template <typename T> T symbol(void *handle, const char *name)
{
    void *address = dlsym(handle, name);
    if (!address) {
        fprintf(stderr, "the shim has no %s\n", name);
        exit(2);
    }
    return (T)address;
}

// An instance of a plugin class without its initializer: the constant answers read no state.
id bare(const char *name)
{
    Class cls = objc_getClass(name);
    if (!cls) {
        fprintf(stderr, "the shim has no class %s\n", name);
        exit(2);
    }
    return class_createInstance(cls, 0);
}

// Runs `body` and answers the reason of the WebMetalNotImplemented exception it raised, or nil.
NSString *loudReason(void (^body)(void))
{
    @try {
        body();
    } @catch (NSException *exception) {
        if ([exception.name isEqualToString:@"WebMetalNotImplemented"])
            return exception.reason ?: @"";
        @throw;
    }
    return nil;
}

void checkDevice(id<MTLDevice> device)
{
    id<SPIDevice> spi = (id<SPIDevice>)device;
    // -[AGXA10FamilyDevice vendorName] 0x21c8d4d24, -[AGXA10XDevice familyName] 0x21c9c4598,
    // -[AGXA10FamilyDevice productName] 0x21c8d4d3c.
    check([[spi vendorName] isEqualToString:@"Apple"], "vendorName is \"Apple\"");
    check([[spi familyName] isEqualToString:@"A10X"], "familyName is \"A10X\"");
    check([[spi productName] isEqualToString:@""], "productName is the empty string");
    // -[AGXA10FamilyDevice counterSets] 0x21c8db3d8.
    NSArray<id<MTLCounterSet>> *sets = device.counterSets;
    check(sets.count == 1 && [sets[0].name isEqualToString:MTLCommonCounterSetTimestamp] &&
              sets[0].counters.count == 1 && [sets[0].counters[0].name isEqualToString:MTLCommonCounterTimestamp],
          "counterSets is one timestamp set of one timestamp counter");
    check(device.counterSets != sets, "counterSets makes a new array on every call");
    // The descriptions are 21H461's MTLCounterSetDescriptionTimestamp and MTLCounterDescriptionTimestamp.
    check(sets.count == 1 && [[(id)sets[0] description] isEqualToString:@"A counter set containing only a GPU timestamp."] &&
              [[(id)sets[0].counters[0] description] isEqualToString:@"A timestamp in nanoseconds on the GPU."],
          "the counter set and counter carry Metal's timestamp descriptions");
    // -[AGXA10FamilyDevice disableComputeEncoderCoalescing] and its setter, 0x21c8dc43c and 0x21c8dc450.
    check(![spi disableComputeEncoderCoalescing], "disableComputeEncoderCoalescing starts NO");
    [spi setDisableComputeEncoderCoalescing:YES];
    check([spi disableComputeEncoderCoalescing], "disableComputeEncoderCoalescing reads back YES");
    ((void (*)(id, SEL, unsigned char))objc_msgSend)(spi, @selector(setDisableComputeEncoderCoalescing:), 2);
    check(![spi disableComputeEncoderCoalescing], "disableComputeEncoderCoalescing reads back the stored byte's low bit");
    [spi setDisableComputeEncoderCoalescing:NO];
    // -[AGXA10FamilyDevice newRenderPipelineStateWithTileDescriptor:options:reflection:error:] 0x21c8d8c64.
    MTLTileRenderPipelineDescriptor *tile = [MTLTileRenderPipelineDescriptor new];
    NSError *error = [NSError errorWithDomain:@"untouched" code:7 userInfo:nil];
    NSError *before = error;
    MTLAutoreleasedRenderPipelineReflection reflection = (MTLRenderPipelineReflection *)@"untouched";
    id pipeline = [device newRenderPipelineStateWithTileDescriptor:tile
                                                           options:MTLPipelineOptionNone
                                                        reflection:&reflection
                                                             error:&error];
    check(pipeline == nil && error == before && [(id)reflection isEqual:@"untouched"],
          "a tile pipeline is nil and leaves the reflection and the error alone");
    // The completion-handler forms (0x21c8d8c80, 0x21c8d8c60) return without calling the handler.
    __block int called = 0;
    [device newRenderPipelineStateWithTileDescriptor:tile
                                             options:MTLPipelineOptionNone
                                   completionHandler:^(id<MTLRenderPipelineState>, MTLRenderPipelineReflection *, NSError *) {
                                       called++;
                                   }];
    ((void (*)(id, SEL, id, id))objc_msgSend)(device, @selector(newRenderPipelineStateWithTileDescriptor:completionHandler:),
                                              tile, ^(id<MTLRenderPipelineState>, MTLRenderPipelineReflection *, NSError *) {
                                                  called++;
                                              });
    [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.2]];
    check(called == 0, "the tile pipeline completion handlers do not run");
    // -[AGXA10FamilyDevice newVisibleFunctionTableWithDescriptor:] 0x21c8db4d8 and
    // -newIndirectComputeCommandEncoderWithBuffer: 0x21c8d63f0 answer nil.
    check([spi newVisibleFunctionTableWithDescriptor:[MTLVisibleFunctionTableDescriptor new]] == nil,
          "newVisibleFunctionTableWithDescriptor: is nil");
    check([spi newIndirectComputeCommandEncoderWithBuffer:nil] == nil, "newIndirectComputeCommandEncoderWithBuffer: is nil");
    // A LOUD stub reports the note guest/SPI/notes.toml gives it.
    NSString *reason = loudReason(^{
        (void)[spi gpuCoreCount];
    });
    check([reason containsString:@"It needs a measurement on a real A10X (decision D-4)"],
          "gpuCoreCount is LOUD with its note (" + std::string(reason ? reason.UTF8String : "no report") + ")");
}

void checkQueue(id<MTLDevice> device)
{
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<SPIQueue> spi = (id<SPIQueue>)queue;
    // -[IOGPUMetalCommandQueue initWithDevice:descriptor:] 0x1fcb26578.
    check([spi getGPUPriority] == 1 && [spi getBackgroundGPUPriority] == 3, "a new queue has priorities 1 and 3");
    // IOGPUFamily's queue setter (kernel 0xfffffff005bd7d44) applies 0, 1, 2 and 4 as given.
    for (NSUInteger priority : {0, 2, 4, 1}) {
        check([spi setGPUPriority:priority] && [spi getGPUPriority] == priority,
              "setGPUPriority:" + std::to_string(priority) + " applies it");
        check([spi getBackgroundGPUPriority] == 3, "setGPUPriority: keeps the background priority");
    }
    check([spi setGPUPriority:2 offset:9] && [spi getGPUPriority] == 2, "setGPUPriority:offset: applies the priority");
    // Priority 5 needs com.apple.private.agx.rt-priority-spi, which this program does not have: 0 is applied.
    check([spi setGPUPriority:5] && [spi getGPUPriority] == 0, "setGPUPriority:5 without the entitlement applies 0");
    // The kernel refuses priority 3 and every priority above 5 (and IOGPU keeps its values); IOGPU's own validation
    // reports these first, so the kernel's answer is checked through the method that asks it.
    check(![spi _setGPUPriority:3 backgroundPriority:3] && [spi getGPUPriority] == 0,
          "the kernel refuses priority 3");
    check(![spi _setGPUPriority:6 backgroundPriority:3] && [spi getGPUPriority] == 0,
          "the kernel refuses priority 6");
    check([spi _setGPUPriority:0x100000001ull backgroundPriority:3] && [spi getGPUPriority] == 1,
          "the kernel sees the low 32 bits of the priority");
    // Background priorities 2 and 3 apply; others are refused.
    check([spi setBackgroundGPUPriority:2] && [spi getBackgroundGPUPriority] == 2 && [spi getGPUPriority] == 1,
          "setBackgroundGPUPriority:2 applies it and keeps the priority");
    check(![spi setBackgroundGPUPriority:4] && [spi getBackgroundGPUPriority] == 2, "background priority 4 is refused");
    check(![spi setBackgroundGPUPriority:0] && [spi getBackgroundGPUPriority] == 2, "background priority 0 is refused");
    check([spi setBackgroundGPUPriority:3 offset:1] && [spi getBackgroundGPUPriority] == 3,
          "setBackgroundGPUPriority:offset: applies the priority");
    check([spi resolveAndUpdateQueuePriority:4] == 4, "a customer build resolves a priority to itself");
    // The A10 family's queues have no statistics counters (0x21c8ca6e8 to 0x21c8ca710).
    check([spi availableCounters] == nil && [spi counterInfo] == nil && [spi getRequestedCounters] == nil &&
              [spi subdivideCounterList:@[ @"a" ]] == nil && [spi requestCounters:@[ @"a" ]] == -1,
          "the queue's counter queries answer nil and -1");
    [spi setStatEnabled:YES];

    // -[AGXA10FamilyCommandBuffer encodeCacheHintTag:resourceGroups:count:] 0x21c8c91a8 returns at once;
    // -encodeCacheHintFinalize:resourceGroups:count: 0x21c8c91ac skips nil groups. The command buffer still completes.
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<SPICommandBuffer> spiBuffer = (id<SPICommandBuffer>)buffer;
    id __unsafe_unretained none[2] = {nil, nil};
    id __unsafe_unretained object[1] = {@"not a resource group"};
    [spiBuffer encodeCacheHintTag:7 resourceGroups:object count:1];
    [spiBuffer encodeCacheHintFinalize:1 resourceGroups:none count:2];
    [spiBuffer encodeCacheHintFinalize:0 resourceGroups:object count:1];
    [spiBuffer setResourceGroups:object count:1];
    [spiBuffer encodeDashboardTagForResourceGroup:object[0]];
    [spiBuffer commitEncoder];
    id __unsafe_unretained const *named = object;
    NSString *reason = loudReason(^{
        [spiBuffer encodeCacheHintFinalize:1 resourceGroups:named count:1];
    });
    check(reason != nil, "a cache-hint finalize naming an object that is no resource group is LOUD");
    [buffer commit];
    [buffer waitUntilCompleted];
    check(buffer.status == MTLCommandBufferStatusCompleted && buffer.error == nil,
          "a command buffer with cache hints completes");
}

void checkResources(id<MTLDevice> device)
{
    // -[IOGPUMetalResource resourceSize] (0x1fcb2e758): a heap resource, a view and a texture over a buffer answer the
    // kernel resource they take from the heap, the parent or the buffer (-[IOGPUMetalResource initWithResource:]).
    auto size = [](id resource) { return [(id<SPIResources>)resource resourceSize]; };
    MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
    hd.size = 65536;
    hd.storageMode = MTLStorageModeShared;
    id<MTLHeap> heap = [device newHeapWithDescriptor:hd];
    id<MTLBuffer> heapBuffer = [heap newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:64 height:64 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device newTextureWithDescriptor:td];
    id<MTLTexture> view = [texture newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB];
    id<MTLTexture> heapTexture = [heap newTextureWithDescriptor:td];
    id<MTLBuffer> buffer = [device newBufferWithLength:65536 options:MTLResourceStorageModeShared];
    MTLTextureDescriptor *linear = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                      width:16 height:16 mipmapped:NO];
    linear.storageMode = MTLStorageModeShared;
    id<MTLTexture> bufferTexture = [buffer newTextureWithDescriptor:linear offset:0 bytesPerRow:256];
    check(heap && heapBuffer && texture && view && heapTexture && buffer && bufferTexture, "the resources are made");
    check(size(heapBuffer) == heap.currentAllocatedSize && size(heapTexture) == heap.currentAllocatedSize,
          "heap resources answer the heap's kernel resource size");
    check(size(texture) == texture.allocatedSize && size(view) == size(texture),
          "a texture answers its kernel resource size and its view the same");
    check(size(bufferTexture) == size(buffer) && size(buffer) == buffer.allocatedSize,
          "a texture over a buffer answers the buffer's kernel resource size");
    // A sub-allocated buffer or texture has a kernel resource of its own whose resident size is the primary buffer's,
    // 0x20000 (IOGPUResource::newResourceWithParentResourceID, kernelcache 0xfffffff005bc42d8); its allocated size is
    // its length (-[IOGPUMetalBuffer allocatedSize] 0x1fcb31bfc, -[IOGPUMetalTexture allocatedSize] 0x1fcb3a7bc). A view
    // shares its parent's kernel resource and answers its size as its allocated size.
    id<MTLBuffer> small = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    MTLTextureDescriptor *tiny = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                    width:4 height:4 mipmapped:NO];
    tiny.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    tiny.storageMode = MTLStorageModeShared;
    id<MTLTexture> tinyTexture = [device newTextureWithDescriptor:tiny];
    id<MTLTexture> tinyView = [tinyTexture newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB];
    check(small && tinyTexture && tinyView && tinyTexture.allocatedSize < 0x4000,
          "a small buffer and a small texture are made and the texture is sub-allocated");
    check(size(small) == 0x20000 && small.allocatedSize == 256, "a sub-allocated buffer answers its primary's size");
    check(size(tinyTexture) == 0x20000 && size(tinyView) == 0x20000 && tinyView.allocatedSize == 0x20000,
          "a sub-allocated texture and its view answer the primary's size");
    // -[IOGPUMetalResource isComplete] and -waitUntilComplete: the submissions of the A10X driver never stamp event 0 of
    // an untracked kernel resource (IOGPUResource's stamp, kernelcache 0xfffffff005bc4e68), so it stays clear; a
    // heap's resource is untracked unless its descriptor says otherwise.
    auto complete = [](id resource) {
        __block BOOL done = NO;
        NSString *reason = loudReason(^{
            done = [(id<SPIResources>)resource isComplete];
            [(id<SPIResources>)resource waitUntilComplete];
        });
        return !reason && done;
    };
    id<MTLBuffer> untracked = [device newBufferWithLength:65536
                                                  options:MTLResourceStorageModeShared |
                                                          MTLResourceHazardTrackingModeUntracked];
    id<MTLBuffer> untrackedSmall = [device newBufferWithLength:256
                                                       options:MTLResourceStorageModeShared |
                                                               MTLResourceHazardTrackingModeUntracked];
    td.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    id<MTLTexture> untrackedTexture = [device newTextureWithDescriptor:td];
    id<MTLTexture> untrackedView = [untrackedTexture newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB];
    id<MTLTexture> overUntracked = [untracked newTextureWithDescriptor:linear offset:0 bytesPerRow:256];
    check(complete(untracked) && complete(untrackedSmall) && complete(heapBuffer) && complete(heapTexture) &&
              complete(untrackedTexture) && complete(untrackedView) && complete(overUntracked),
          "untracked kernel resources are complete");
    // A tracked kernel resource no committed command buffer uses has an event 0 the kernel has not stamped.
    check(complete(buffer) && complete(bufferTexture) && complete(texture) && complete(view),
          "tracked kernel resources no command buffer uses are complete");
    // A texture over memory a texture layout filled (-[AGXA10FamilyDevice newTextureWithBytesNoCopy:length:descriptor:
    // deallocator:]): its texels are the memory's, its deallocator receives the memory, and the size of its kernel
    // resource is LOUD.
    td.hazardTrackingMode = MTLHazardTrackingModeDefault;
    id<SPITextureLayout> layout = [(id<SPIDevice>)device newTextureLayoutWithDescriptor:td isHeapOrBufferBacked:NO];
    void *bytes = nullptr;
    check(layout && posix_memalign(&bytes, layout.alignment, layout.size) == 0, "a texture layout and its memory");
    std::vector<uint32_t> texels(64 * 64);
    for (size_t i = 0; i < texels.size(); i++)
        texels[i] = uint32_t(i * 2654435761u);
    [layout copyFromLinearBytes:texels.data() linearOffset:0 linearBytesPerRow:256 linearBytesPerImage:256 * 64
                toTextureMemory:bytes textureSlice:0 textureLevel:0 textureRegion:MTLRegionMake2D(0, 0, 64, 64)];
    [layout finalizeTextureMemory:bytes];
    __block void *released = nullptr;
    __block NSUInteger releasedLength = 0;
    @autoreleasepool {
        id<MTLTexture> noCopy = [(id<SPIDevice>)device newTextureWithBytesNoCopy:bytes length:layout.size descriptor:td
                                                                     deallocator:^(void *pointer, NSUInteger length) {
            released = pointer;
            releasedLength = length;
        }];
        check(noCopy != nil, "newTextureWithBytesNoCopy: makes a texture over the layout's memory");
        std::vector<uint32_t> read(64 * 64);
        [noCopy getBytes:read.data() bytesPerRow:256 fromRegion:MTLRegionMake2D(0, 0, 64, 64) mipmapLevel:0];
        check(read == texels, "getBytes of a texture over caller memory reads the memory's texels");
        NSString *reason = loudReason(^{
            (void)noCopy.allocatedSize;
        });
        check([reason containsString:@"over caller memory"], "the kernel resource size of a no-copy texture is LOUD");
    }
    check(released == bytes && releasedLength == layout.size, "the deallocator receives the memory and its length");
    free(bytes);
}

// -[IOGPUMetalResource isComplete] tests event 0, which the A10X kernel stamps when it processes an encoder that lists
// the resource (render -useResource:usage: lists it with flags 5). The kernel stays at a wait on a shared event until
// the event reaches the value (IOGPUCommandQueue::processKernelCommand, kernelcache 0xfffffff005bd8d54 to
// 0xfffffff005bd8e18), so an encoder after the wait has stamped nothing until then.
void checkStamps(id<MTLDevice> device)
{
    id<MTLBuffer> late = [device newBufferWithLength:256
                                             options:MTLResourceStorageModeShared |
                                                     MTLResourceHazardTrackingModeTracked];
    id<MTLSharedEvent> event = [device newSharedEvent];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:8 height:8 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = [device newTextureWithDescriptor:td];
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    check(late && event && queue && pass.colorAttachments[0].texture, "the stamp check's objects are made");
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    [buffer encodeWaitForEvent:event value:1];
    id<MTLRenderCommandEncoder> after = [buffer renderCommandEncoderWithDescriptor:pass];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [after useResource:late usage:MTLResourceUsageRead];
#pragma clang diagnostic pop
    [after endEncoding];
    [buffer commit];
    usleep(100000);
    __block BOOL complete = NO;
    NSString *reason = loudReason(^{
        complete = [(id<SPIResources>)late isComplete];
    });
    check(!reason && complete, "a buffer only an encoder after an unpassed shared-event wait lists is complete");
    event.signaledValue = 1;
    [buffer waitUntilScheduled];
    complete = NO;
    reason = loudReason(^{
        [(id<SPIResources>)late waitUntilComplete];
        complete = [(id<SPIResources>)late isComplete];
    });
    check(!reason && complete, "once the wait is passed the buffer completes with the encoder after it");
    [buffer waitUntilCompleted];
    check(buffer.status == MTLCommandBufferStatusCompleted && buffer.error == nil,
          "the command buffer with the wait completes");
}

void checkConstants()
{
    // -[AGXA10FamilyRenderPipeline ...] 0x21c977a54 to 0x21c977c20.
    id<SPIConstants> render = bare("WebMetalRenderPipelineState");
    check([render imageblockSampleLength] == 0 && [render imageBlockSampleLength] == 0 &&
              [render imageblockMemoryLengthForDimensions:MTLSizeMake(8, 8, 1)] == 0 &&
              [render imageBlockMemoryLengthForDimensions:MTLSizeMake(8, 8, 1)] == 0 &&
              [render staticThreadgroupMemoryLength] == 0 && [render objectThreadExecutionWidth] == 32 &&
              [render meshThreadExecutionWidth] == 32 && [render resourceIndex] == 0,
          "the render pipeline's constants are the A10 family's");
    // -[AGXA10FamilyComputePipeline ...] 0x21c8cc3c8, 0x21c8cc3d0, 0x21c8cc7c8.
    id<SPIConstants> compute = bare("WebMetalComputePipelineState");
    check([compute imageblockMemoryLengthForDimensions:MTLSizeMake(4, 4, 1)] == 0 &&
              [compute imageBlockMemoryLengthForDimensions:MTLSizeMake(4, 4, 1)] == 0 && [compute resourceIndex] == 0,
          "the compute pipeline's constants are the A10 family's");
    // -[AGXA10FamilyBlitContext commandBatchIdOffset] 0x21c8baf38 and AGXA10FamilyComputeContext's 0x21c90c414 to
    // 0x21c90c4d4.
    id<SPIConstants> blit = bare("WebMetalBlitCommandEncoder");
    id<SPIConstants> computeEncoder = bare("WebMetalComputeCommandEncoder");
    check([blit commandBatchIdOffset] == 0 && [computeEncoder commandBatchIdOffset] == 0 &&
              [computeEncoder nextVirtualSubstream] == ~NSUInteger(0) && ![computeEncoder parallelExecution],
          "the encoders' constants are the A10 family's");
    // -[AGXA10FamilyThreadedRenderPass setMTLCounterSampleLocationWithBuffer:startIndex:endIndex:] 0x21c9c24a0.
    id<SPIConstants> parallel = bare("WebMetalParallelRenderCommandEncoder");
    check(![parallel setMTLCounterSampleLocationWithBuffer:nil startIndex:0 endIndex:1],
          "the parallel encoder's counter sample location is NO");
    // -[AGXA10FamilyTexture ...] 0x21c9822e0 to 0x21c9823c4.
    id<SPIConstants> texture = bare("WebMetalTexture");
    check([texture compressionFeedback] == 4 && [texture firstMipmapInTail] == 0 && [texture tailSize] == 0 &&
              [texture tailSizeInBytes] == 0 && [texture sparseSurfaceDefaultValue] == 0,
          "the texture's constants are the A10 family's");
}

}  // namespace

int main(int argc, char **argv)
{
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s shim.dylib\n", argv[0]);
            return 2;
        }
        void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            fprintf(stderr, "dlopen %s: %s\n", argv[1], dlerror());
            return 2;
        }
        id<MTLDevice> device = symbol<id<MTLDevice> (*)(void)>(handle, "WMRefProvideDevice")();
        if (!device) {
            printf("FAIL the shim created no device\n");
            return 1;
        }
        checkDevice(device);
        checkQueue(device);
        checkResources(device);
        checkStamps(device);
        checkConstants();
        printf("%s: %u of %u SPI checks failed\n", gFailures ? "FAIL" : "PASS", gFailures, gChecks);
        return gFailures ? 1 : 0;
    }
}
