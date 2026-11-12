// Storage and coherence cases: shared, private, write-combined, managed and
// no-copy buffers, the CPU/GPU coherence rule for shared memory, texture
// CPU access, buffer-backed and IOSurface-backed textures, and memoryless
// render targets.

#import "WMRefCase.h"

#include <IOSurface/IOSurfaceRef.h>
#include <TargetConditionals.h>
#include <stdlib.h>
#include <unistd.h>

static BOOL StorageWait(dispatch_semaphore_t semaphore, double seconds) {
    return dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(seconds * NSEC_PER_SEC))) == 0;
}

// Encodes `sto_add_one` over `count` words of `buffer` starting at word
// `first`; `count` is a multiple of 64.
static void StorageEncodeAddOne(WMCase *c, id<MTLCommandBuffer> commandBuffer, id<MTLBuffer> buffer,
                                NSUInteger first, NSUInteger count) {
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"sto_add_one"];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:first * sizeof(uint32_t) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(count / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
}

static void StorageAddOne(WMCase *c, id<MTLBuffer> buffer, NSUInteger count) {
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    StorageEncodeAddOne(c, commandBuffer, buffer, 0, count);
    [c commitAndWait:commandBuffer];
}

WMREF_CASE(sharedRoundTrip, "storage.shared-roundtrip", "storage.shared",
           "CPU writes, GPU increments, CPU reads a shared buffer, twice")
{
    // MTLResource.h, MTLResourceStorageModeShared: "coherency is only
    // guaranteed at command buffer boundaries".
    enum { N = 512 };
    id<MTLBuffer> buffer = [c buffer:N * 4];
    WMPattern(buffer.contents, N * 4, 51);
    StorageAddOne(c, buffer, N);
    [c data:@"after-first" elem:WMElemU32 bytes:buffer.contents length:N * 4 rule:WMExact()];
    WMPattern(buffer.contents, N * 4, 52);
    StorageAddOne(c, buffer, N);
    [c data:@"after-second" elem:WMElemU32 bytes:buffer.contents length:N * 4 rule:WMExact()];
}

// A deterministic program of CPU writes, GPU increments and fills, commits
// and waits on one shared buffer. CPU writes only touch blocks that no
// committed and unfinished command buffer accesses, which is the use
// Metal's coherence rule (coherency at command buffer boundaries) defines.
// Blocks are 256 bytes so that CPU and GPU never write within one cache
// line at the same time. A CPU model of the program gives the expected
// contents at every wait.
WMREF_CASE(coherenceInterleaving, "storage.coherence-interleaving", "storage.coherence",
           "Generated interleavings of CPU writes to disjoint blocks with in-flight GPU writes")
{
    enum { Blocks = 8, BlockWords = 64, Words = Blocks * BlockWords, Steps = 24 };
    const uint32_t seeds[2] = {101, 202};
    for (int program = 0; program < 2; program++) {
        id<MTLBuffer> buffer = [c buffer:Words * 4];
        uint32_t *words = buffer.contents;
        uint32_t model[Words] = {0};
        NSMutableArray<id<MTLCommandBuffer>> *inFlight = [NSMutableArray array];
        unsigned busy = 0; // blocks touched by in-flight command buffers
        uint32_t state = seeds[program];
        int snapshot = 0;
        NSMutableArray<NSString *> *trace = [NSMutableArray array];
        void (^waitAll)(void) = ^{
            for (id<MTLCommandBuffer> commandBuffer in inFlight) {
                [commandBuffer waitUntilCompleted];
                if (commandBuffer.status != MTLCommandBufferStatusCompleted)
                    [c fail:@"command buffer ended with status %lu", (unsigned long)commandBuffer.status];
            }
            [inFlight removeAllObjects];
        };
        for (int step = 0; step < Steps; step++) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            uint32_t r = state;
            unsigned op = r % 4;
            unsigned first = (r >> 4) % Blocks;
            unsigned length = 1 + (r >> 8) % 3;
            if (first + length > Blocks) length = Blocks - first;
            unsigned mask = ((1u << length) - 1) << first;
            if (op == 0 && (mask & busy)) op = 3;
            if (op == 0) {
                for (unsigned i = first * BlockWords; i < (first + length) * BlockWords; i++) {
                    uint32_t value = ((uint32_t)step << 24) | ((uint32_t)program << 16) | i;
                    words[i] = value;
                    model[i] = value;
                }
                [trace addObject:[NSString stringWithFormat:@"cpu-write %u+%u", first, length]];
            } else if (op == 1) {
                id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
                StorageEncodeAddOne(c, commandBuffer, buffer, first * BlockWords, length * BlockWords);
                [commandBuffer commit];
                [inFlight addObject:commandBuffer];
                busy |= mask;
                for (unsigned i = first * BlockWords; i < (first + length) * BlockWords; i++) model[i] += 1;
                [trace addObject:[NSString stringWithFormat:@"gpu-add %u+%u", first, length]];
            } else if (op == 2) {
                uint8_t value = (uint8_t)(r >> 16);
                id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
                id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
                [blit fillBuffer:buffer range:NSMakeRange(first * BlockWords * 4, length * BlockWords * 4) value:value];
                [blit endEncoding];
                [commandBuffer commit];
                [inFlight addObject:commandBuffer];
                busy |= mask;
                for (unsigned i = first * BlockWords; i < (first + length) * BlockWords; i++)
                    model[i] = value * 0x01010101u;
                [trace addObject:[NSString stringWithFormat:@"gpu-fill %u+%u 0x%02x", first, length, value]];
            } else {
                waitAll();
                busy = 0;
                [trace addObject:@"wait"];
                [c data:[NSString stringWithFormat:@"program-%d.snapshot-%d", program, snapshot++] elem:WMElemU32
                  bytes:words length:Words * 4 rule:WMExact()];
            }
        }
        waitAll();
        [trace addObject:@"wait"];
        [c value:[NSString stringWithFormat:@"program-%d.trace", program] strings:trace rule:WMExact()];
        [c data:[NSString stringWithFormat:@"program-%d.final", program] elem:WMElemU32 bytes:words
          length:Words * 4 rule:WMExact()];
        [c value:[NSString stringWithFormat:@"program-%d.matches-model", program]
            boolean:memcmp(model, words, sizeof model) == 0 rule:WMExact()];
    }
}

// MTLDevice.h, newBufferWithBytes:length:options: "Create a buffer by
// allocating new memory and specifing the initial contents to be copied
// into it."
WMREF_CASE(newBufferCopies, "storage.new-buffer-with-bytes-copies", "storage.shared",
           "newBufferWithBytes copies its source; later writes to either side stay separate")
{
    uint8_t source[256];
    WMPattern(source, sizeof source, 61);
    uint8_t original[256];
    memcpy(original, source, sizeof source);
    id<MTLBuffer> buffer = [c bufferWithBytes:source length:sizeof source];
    memset(source, 0x11, sizeof source);
    [c value:@"buffer-keeps-initial-bytes" boolean:memcmp(buffer.contents, original, 256) == 0 rule:WMExact()];
    memset(buffer.contents, 0x22, 256);
    BOOL sourceUnchanged = YES;
    for (int i = 0; i < 256; i++) sourceUnchanged &= source[i] == 0x11;
    [c value:@"source-unaffected-by-buffer-write" boolean:sourceUnchanged rule:WMExact()];
    [c value:@"length" u64:buffer.length rule:WMExact()];
}

// MTLDevice.h, newBufferWithBytesNoCopy:length:options:deallocator:
// "Create a buffer by wrapping an existing part of the address space."
WMREF_CASE(noCopyDeallocator, "storage.no-copy-deallocator", "storage.no-copy",
           "A no-copy buffer over page-aligned memory: GPU writes land in the memory; deallocator timing")
{
    size_t page = (size_t)getpagesize();
    size_t length = page;
    void *memory = NULL;
    if (posix_memalign(&memory, page, length) != 0) [c fail:@"posix_memalign failed"];
    uint32_t *words = memory;
    for (size_t i = 0; i < length / 4; i++) words[i] = (uint32_t)i * 3;
    dispatch_semaphore_t released = dispatch_semaphore_create(0), completed = dispatch_semaphore_create(0);
    __block BOOL pointerMatches = NO, lengthMatches = NO, writtenAtRelease = YES;
    @autoreleasepool {
        id<MTLBuffer> buffer = [c.device newBufferWithBytesNoCopy:memory length:length
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:^(void *pointer, NSUInteger size) {
            const uint32_t *seen = pointer;
            for (size_t i = 0; i < size / 4; i++) writtenAtRelease &= seen[i] == (uint32_t)i * 3 + 1;
            pointerMatches = pointer == memory;
            lengthMatches = size == length;
            free(pointer);
            dispatch_semaphore_signal(released);
        }];
        if (!buffer) [c fail:@"newBufferWithBytesNoCopy returned nil for page-aligned memory"];
        [c value:@"contents-is-wrapped-memory" boolean:buffer.contents == memory rule:WMExact()];
        [c value:@"length" u64:buffer.length rule:WMExact()];
        id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
        StorageEncodeAddOne(c, commandBuffer, buffer, 0, length / 4);
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> b) {
            dispatch_semaphore_signal(completed);
        }];
        [commandBuffer commit];
        buffer = nil;
        commandBuffer = nil;
    }
    if (!StorageWait(completed, 10)) [c fail:@"completed handler did not run within 10 s"];
    if (!StorageWait(released, 10)) [c fail:@"deallocator did not run within 10 s"];
    [c value:@"deallocator.pointer-matches" boolean:pointerMatches rule:WMExact()];
    [c value:@"deallocator.length-matches" boolean:lengthMatches rule:WMExact()];
    [c value:@"deallocator.sees-gpu-writes" boolean:writtenAtRelease rule:WMExact()];
    // The deallocator runs once the last reference is released; the command
    // buffer holds one until the GPU work completes, which the deallocator
    // observes as the incremented words. The order of the completed handler
    // and the release is not documented and is not recorded.
}

WMREF_CASE(noCopyUnaligned, "storage.no-copy-unaligned", "storage.errors",
           "Error path: newBufferWithBytesNoCopy on memory that is not page aligned")
{
    size_t page = (size_t)getpagesize();
    void *memory = NULL;
    if (posix_memalign(&memory, page, 2 * page) != 0) [c fail:@"posix_memalign failed"];
    __block BOOL isNil = NO;
    [c step:@"create"];
    [c call:@"create" rule:WMExact() block:^{
        id<MTLBuffer> buffer = [c.device newBufferWithBytesNoCopy:(uint8_t *)memory + 16 length:page
                                                          options:MTLResourceStorageModeShared deallocator:nil];
        isNil = buffer == nil;
    }];
    [c value:@"is-nil" boolean:isNil
        rule:WMNone(@"the documentation requires page-aligned memory and does not define other input")];
    free(memory);
}

WMREF_CASE(zeroLength, "storage.buffer-length-zero", "storage.errors",
           "Error path: newBufferWithLength:0")
{
    __block BOOL isNil = NO;
    [c step:@"create"];
    [c call:@"create" rule:WMExact() block:^{
        isNil = [c.device newBufferWithLength:0 options:MTLResourceStorageModeShared] == nil;
    }];
    [c value:@"is-nil" boolean:isNil rule:WMNone(@"MTLDevice.h does not define a zero length")];
}

// MTLBuffer documentation, contents: "A pointer to the shared copy of the
// buffer data, or NULL for buffers allocated with a private resource
// storage mode." The debug layer rejects the call; without it the answer is
// recorded as observed, since it is the documented value only in the text.
WMREF_CASE(privateContents, "storage.private-buffer-contents", "storage.errors",
           "Error path: contents of a private buffer")
{
    id<MTLBuffer> private = [c.device newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
    if (!private) [c fail:@"private buffer creation returned nil"];
    __block BOOL isNull = NO;
    [c step:@"contents"];
    [c call:@"contents" rule:WMExact() block:^{
        isNull = private.contents == NULL;
    }];
    [c value:@"contents-is-null" boolean:isNull
        rule:WMNone(@"the documentation states NULL while the debug layer treats the call as invalid use")];
}

WMREF_CASE(privateBuffer, "storage.private-buffer", "storage.private",
           "Private buffer filled, incremented and read back through blits")
{
    enum { N = 1024 };
    id<MTLBuffer> private = [c.device newBufferWithLength:N * 4 options:MTLResourceStorageModePrivate];
    if (!private) [c fail:@"private buffer creation returned nil"];
    [c value:@"storage-mode" u64:private.storageMode rule:WMExact()];
    id<MTLBuffer> upload = [c buffer:N * 4], download = [c buffer:N * 4];
    WMPattern(upload.contents, N * 4, 71);
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit copyFromBuffer:upload sourceOffset:0 toBuffer:private destinationOffset:0 size:N * 4];
    [blit endEncoding];
    StorageEncodeAddOne(c, commandBuffer, private, 0, N);
    blit = [commandBuffer blitCommandEncoder];
    [blit copyFromBuffer:private sourceOffset:0 toBuffer:download destinationOffset:0 size:N * 4];
    [blit endEncoding];
    [c commitAndWait:commandBuffer];
    [c data:@"download" elem:WMElemU32 bytes:download.contents length:N * 4 rule:WMExact()];
}

// MTLDevice.h states no initial contents for newBufferWithLength:options:.
WMREF_CASE(initialContents, "storage.new-buffer-initial-contents", "storage.initial-contents",
           "Initial contents of new shared and private buffers")
{
    WMRule *unstated = WMNone(@"MTLDevice.h, newBufferWithLength:options: does not state the initial contents");
    enum { N = 1 << 16 };
    id<MTLBuffer> shared = [c.device newBufferWithLength:N options:MTLResourceStorageModeShared];
    id<MTLBuffer> private = [c.device newBufferWithLength:N options:MTLResourceStorageModePrivate];
    if (!shared || !private) [c fail:@"buffer creation returned nil"];
    const uint8_t *bytes = shared.contents;
    BOOL zero = YES;
    for (size_t i = 0; i < N; i++) zero &= bytes[i] == 0;
    [c value:@"shared.all-zero" boolean:zero rule:unstated];
    id<MTLBuffer> readback = [c buffer:N];
    memset(readback.contents, 0xff, N);
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:private sourceOffset:0 toBuffer:readback destinationOffset:0 size:N];
    }];
    bytes = readback.contents;
    zero = YES;
    for (size_t i = 0; i < N; i++) zero &= bytes[i] == 0;
    [c value:@"private.all-zero" boolean:zero rule:unstated];
}

// MTLResource.h: resourceOptions is "A packed tuple of the storageMode,
// cpuCacheMode and hazardTrackingMode properties"; resources created from
// the device are tracked by default.
WMREF_CASE(resourceOptionsReadback, "storage.resource-options", "storage.options",
           "storageMode, cpuCacheMode, hazardTrackingMode and resourceOptions of buffers and textures")
{
    struct { const char *name; MTLResourceOptions options; } cases[] = {
        {"default", 0},
        {"write-combined", MTLResourceCPUCacheModeWriteCombined},
        {"private", MTLResourceStorageModePrivate},
        {"untracked", MTLResourceHazardTrackingModeUntracked},
        {"tracked", MTLResourceHazardTrackingModeTracked},
        {"shared-write-combined-untracked", MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined |
                                                MTLResourceHazardTrackingModeUntracked},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        id<MTLBuffer> buffer = [c.device newBufferWithLength:256 options:cases[i].options];
        if (!buffer) [c fail:@"buffer %s returned nil", cases[i].name];
        NSString *prefix = [NSString stringWithFormat:@"buffer.%s", cases[i].name];
        [c value:[prefix stringByAppendingString:@".storage-mode"] u64:buffer.storageMode rule:WMExact()];
        [c value:[prefix stringByAppendingString:@".cpu-cache-mode"] u64:buffer.cpuCacheMode rule:WMExact()];
        [c value:[prefix stringByAppendingString:@".hazard-tracking-mode"] u64:buffer.hazardTrackingMode rule:WMExact()];
        [c value:[prefix stringByAppendingString:@".resource-options"] u64:buffer.resourceOptions rule:WMExact()];
    }
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                         width:4 height:4
                                                                                     mipmapped:NO];
    // The default texture storage mode differs by platform (MTLResource.h:
    // shared "is the default storage mode for iOS Textures").
    [c value:@"texture-descriptor.default-storage-mode" u64:descriptor.storageMode rule:WMDevice()];
    [c value:@"texture-descriptor.default-cpu-cache-mode" u64:descriptor.cpuCacheMode rule:WMExact()];
    [c value:@"texture-descriptor.default-hazard-tracking-mode" u64:descriptor.hazardTrackingMode rule:WMExact()];
    descriptor.resourceOptions = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
    id<MTLTexture> texture = [c texture:descriptor];
    [c value:@"texture.storage-mode" u64:texture.storageMode rule:WMExact()];
    [c value:@"texture.cpu-cache-mode" u64:texture.cpuCacheMode rule:WMExact()];
    [c value:@"texture.hazard-tracking-mode" u64:texture.hazardTrackingMode rule:WMExact()];
    [c value:@"texture.resource-options" u64:texture.resourceOptions rule:WMExact()];
}

// MTLResource.h, MTLCPUCacheModeWriteCombined: "optimized for resources
// that the CPU will write into, but never read ... reads perform very
// poorly." Reads are slow, not undefined.
WMREF_CASE(writeCombined, "storage.write-combined", "storage.options",
           "A write-combined shared buffer written by the CPU, incremented by the GPU, read by the CPU")
{
    enum { N = 256 };
    id<MTLBuffer> buffer = [c.device newBufferWithLength:N * 4
                                                 options:MTLResourceStorageModeShared |
                                                         MTLResourceCPUCacheModeWriteCombined];
    if (!buffer) [c fail:@"write-combined buffer returned nil"];
    [c value:@"cpu-cache-mode" u64:buffer.cpuCacheMode rule:WMExact()];
    uint32_t pattern[N];
    WMPattern(pattern, sizeof pattern, 81);
    memcpy(buffer.contents, pattern, sizeof pattern);
    StorageAddOne(c, buffer, N);
    [c data:@"result" elem:WMElemU32 bytes:buffer.contents length:N * 4 rule:WMExact()];
}

// MTLResource.h, MTLResourceStorageModeManaged: macOS only; on Apple
// silicon "Managed storage has no effect", but didModifyRange: and
// synchronizeResource: remain the defined way to move data.
WMREF_CASE(managedSynchronize, "storage.managed-synchronize", "storage.managed",
           "A managed buffer: didModifyRange:, GPU increment, synchronizeResource:, CPU read")
{
#if TARGET_OS_OSX
    [c require:@"storage-mode.managed" value:YES];
    enum { N = 256 };
    id<MTLBuffer> buffer = [c.device newBufferWithLength:N * 4 options:MTLResourceStorageModeManaged];
    if (!buffer) [c fail:@"managed buffer returned nil"];
    [c value:@"storage-mode" u64:buffer.storageMode rule:WMExact()];
    WMPattern(buffer.contents, N * 4, 91);
    [buffer didModifyRange:NSMakeRange(0, N * 4)];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    StorageEncodeAddOne(c, commandBuffer, buffer, 0, N);
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit synchronizeResource:buffer];
    [blit endEncoding];
    [c commitAndWait:commandBuffer];
    [c data:@"result" elem:WMElemU32 bytes:buffer.contents length:N * 4 rule:WMExact()];
#else
    [c require:@"storage-mode.managed" value:NO];
#endif
}

static id<MTLTexture> StorageSharedTexture(WMCase *c, MTLPixelFormat format, NSUInteger width, NSUInteger height,
                                           MTLTextureUsage usage) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width
                                                                                         height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = usage;
    return [c texture:descriptor];
}

// MTLTexture documentation, replaceRegion: "This method runs on the CPU and
// immediately copies the pixel data into the texture." A copy encoded
// earlier in a command buffer that is not yet committed has not run, so it
// reads the bytes of the later replaceRegion.
WMREF_CASE(replaceRegionOrdering, "storage.texture-replace-region-ordering", "storage.texture-cpu-access",
           "replaceRegion after a copy is encoded but before its command buffer is committed")
{
    // Shared-storage textures: every device the harness runs on is an
    // Apple-family GPU (see README.md, "Devices").
    enum { W = 8, H = 8 };
    id<MTLTexture> texture = StorageSharedTexture(c, MTLPixelFormatRGBA8Uint, W, H, MTLTextureUsageShaderRead);
    uint8_t first[W * H * 4], second[W * H * 4];
    WMPattern(first, sizeof first, 101);
    WMPattern(second, sizeof second, 102);
    [texture replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:first bytesPerRow:W * 4];
    id<MTLBuffer> out = [c buffer:sizeof first];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(W, H, 1) toBuffer:out destinationOffset:0 destinationBytesPerRow:W * 4
  destinationBytesPerImage:W * H * 4];
    [blit endEncoding];
    [texture replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:second bytesPerRow:W * 4];
    [c commitAndWait:commandBuffer];
    [c value:@"copy-sees-later-bytes" boolean:memcmp(out.contents, second, sizeof second) == 0 rule:WMExact()];
    [c data:@"copied" elem:WMElemU8 bytes:out.contents length:sizeof second rule:WMExact()];
}

// MTLTexture documentation, getBytes: reading after the GPU work that
// writes the texture has completed returns the written texels.
WMREF_CASE(getBytesAfterWrite, "storage.texture-get-bytes-after-gpu-write", "storage.texture-cpu-access",
           "getBytes of a shared texture after a compute pass wrote it")
{
    enum { W = 8, H = 4 };
    id<MTLTexture> texture = StorageSharedTexture(c, MTLPixelFormatRGBA32Uint, W, H,
                                                  MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    uint32_t values[W * H * 4];
    WMPattern(values, sizeof values, 111);
    id<MTLBuffer> input = [c bufferWithBytes:values length:sizeof values];
    id<MTLComputePipelineState> write = [c computePipeline:@"sto_write_texture"];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:write];
        [e setTexture:texture atIndex:0];
        [e setBuffer:input offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(W, H, 1)];
    }];
    uint32_t read[W * H * 4];
    [texture getBytes:read bytesPerRow:W * 16 fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
    [c value:@"equals-written" boolean:memcmp(read, values, sizeof read) == 0 rule:WMExact()];
    [c data:@"texels" elem:WMElemU32 bytes:read length:sizeof read rule:WMExact()];
}

WMREF_CASE(privateReplaceRegion, "storage.private-texture-replace-region", "storage.errors",
           "Error path: replaceRegion on a private texture")
{
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Uint
                                                                                         width:4 height:4
                                                                                     mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> texture = [c texture:descriptor];
    uint8_t bytes[64];
    WMPattern(bytes, sizeof bytes, 121);
    const uint8_t *pointer = bytes;
    [c step:@"replace-region"];
    [c call:@"replace-region" rule:WMExact() block:^{
        [texture replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:pointer bytesPerRow:16];
    }];
}

// MTLBuffer.h, newTextureWithDescriptor:offset:bytesPerRow: creates a
// texture that shares the buffer's storage; MTLDevice.h,
// minimumLinearTextureAlignmentForPixelFormat: gives the offset and row
// alignment.
WMREF_CASE(bufferBackedTexture, "storage.buffer-backed-texture", "storage.linear-texture",
           "A texture over a buffer: CPU bytes read by a shader, shader writes visible in the buffer")
{
    enum { W = 8, H = 4 };
    MTLPixelFormat format = MTLPixelFormatRGBA8Uint;
    NSUInteger alignment = [c.device minimumLinearTextureAlignmentForPixelFormat:format];
    [c value:@"minimum-linear-texture-alignment" u64:alignment rule:WMDevice()];
    [c value:@"minimum-texture-buffer-alignment" u64:[c.device minimumTextureBufferAlignmentForPixelFormat:format]
        rule:WMDevice()];
    if (alignment == 0) [c fail:@"zero linear texture alignment"];
    NSUInteger bytesPerRow = (W * 4 + alignment - 1) / alignment * alignment;
    NSUInteger offset = alignment;
    id<MTLBuffer> buffer = [c buffer:offset + bytesPerRow * H];
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:W
                                                                                         height:H mipmapped:NO];
    descriptor.storageMode = buffer.storageMode;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> texture = [buffer newTextureWithDescriptor:descriptor offset:offset bytesPerRow:bytesPerRow];
    if (!texture) [c fail:@"newTextureWithDescriptor:offset:bytesPerRow: returned nil"];
    [c value:@"texture.buffer-is-source" boolean:texture.buffer == buffer rule:WMExact()];
    [c value:@"texture.buffer-offset" u64:texture.bufferOffset - offset rule:WMExact()];
    [c value:@"texture.buffer-bytes-per-row-matches" boolean:texture.bufferBytesPerRow == bytesPerRow rule:WMExact()];

    uint8_t texels[W * H * 4];
    WMPattern(texels, sizeof texels, 131);
    for (int y = 0; y < H; y++) memcpy((uint8_t *)buffer.contents + offset + y * bytesPerRow, texels + y * W * 4, W * 4);
    id<MTLBuffer> out = [c buffer:W * H * 16];
    id<MTLComputePipelineState> read = [c computePipeline:@"sto_read_texture"];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:read];
        [e setTexture:texture atIndex:0];
        [e setBuffer:out offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(W, H, 1)];
    }];
    uint32_t expected[W * H * 4];
    for (int i = 0; i < W * H * 4; i++) expected[i] = texels[i];
    [c value:@"shader-reads-buffer-bytes" boolean:memcmp(out.contents, expected, sizeof expected) == 0 rule:WMExact()];
    [c data:@"read" elem:WMElemU32 bytes:out.contents length:sizeof expected rule:WMExact()];

    uint32_t values[W * H * 4];
    for (int i = 0; i < W * H * 4; i++) values[i] = (uint32_t)(i * 7 + 3) & 0xff;
    id<MTLBuffer> input = [c bufferWithBytes:values length:sizeof values];
    id<MTLComputePipelineState> write = [c computePipeline:@"sto_write_texture"];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:write];
        [e setTexture:texture atIndex:0];
        [e setBuffer:input offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(W, H, 1)];
    }];
    uint8_t rows[W * H * 4];
    BOOL matches = YES;
    for (int y = 0; y < H; y++) {
        memcpy(rows + y * W * 4, (uint8_t *)buffer.contents + offset + y * bytesPerRow, W * 4);
        for (int i = 0; i < W * 4; i++) matches &= rows[y * W * 4 + i] == values[y * W * 4 + i];
    }
    [c value:@"buffer-holds-shader-writes" boolean:matches rule:WMExact()];
    [c data:@"written-rows" elem:WMElemU8 bytes:rows length:sizeof rows rule:WMExact()];
}

// MTLDevice.h, newTextureWithDescriptor:iosurface:plane: creates a texture
// that shares the IOSurface's memory; the CPU accesses it between
// IOSurfaceLock and IOSurfaceUnlock (IOSurfaceRef.h).
WMREF_CASE(iosurfaceTexture, "storage.iosurface-texture", "storage.iosurface",
           "An IOSurface-backed texture: CPU writes under lock read by a blit, blit writes read under lock")
{
    enum { W = 8, H = 4 };
    NSDictionary *properties = @{
        (__bridge NSString *)kIOSurfaceWidth: @(W),
        (__bridge NSString *)kIOSurfaceHeight: @(H),
        (__bridge NSString *)kIOSurfaceBytesPerElement: @4,
        (__bridge NSString *)kIOSurfacePixelFormat: @((uint32_t)'BGRA'),
    };
    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
    if (!surface) [c fail:@"IOSurfaceCreate returned NULL"];
    size_t bytesPerRow = IOSurfaceGetBytesPerRow(surface);
    [c value:@"surface.bytes-per-row" u64:bytesPerRow rule:WMDevice()];
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                         width:W height:H
                                                                                     mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [c.device newTextureWithDescriptor:descriptor iosurface:surface plane:0];
    if (!texture) {
        CFRelease(surface);
        [c fail:@"newTextureWithDescriptor:iosurface:plane: returned nil"];
    }
    [c value:@"texture.iosurface-is-surface" boolean:texture.iosurface == surface rule:WMExact()];
    [c value:@"texture.iosurface-plane" u64:texture.iosurfacePlane rule:WMExact()];

    uint8_t first[W * H * 4], second[W * H * 4];
    WMPattern(first, sizeof first, 141);
    WMPattern(second, sizeof second, 142);
    IOSurfaceLock(surface, 0, NULL);
    uint8_t *base = IOSurfaceGetBaseAddress(surface);
    for (int y = 0; y < H; y++) memcpy(base + y * bytesPerRow, first + y * W * 4, W * 4);
    IOSurfaceUnlock(surface, 0, NULL);

    id<MTLBuffer> out = [c buffer:sizeof first];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(W, H, 1) toBuffer:out destinationOffset:0 destinationBytesPerRow:W * 4
   destinationBytesPerImage:W * H * 4];
    }];
    [c value:@"gpu-reads-cpu-bytes" boolean:memcmp(out.contents, first, sizeof first) == 0 rule:WMExact()];

    id<MTLBuffer> input = [c bufferWithBytes:second length:sizeof second];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:input sourceOffset:0 sourceBytesPerRow:W * 4 sourceBytesPerImage:W * H * 4
               sourceSize:MTLSizeMake(W, H, 1) toTexture:texture destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    }];
    uint8_t seen[W * H * 4];
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    base = IOSurfaceGetBaseAddress(surface);
    for (int y = 0; y < H; y++) memcpy(seen + y * W * 4, base + y * bytesPerRow, W * 4);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    [c value:@"cpu-reads-gpu-bytes" boolean:memcmp(seen, second, sizeof second) == 0 rule:WMExact()];
    [c data:@"cpu-view-after-gpu-write" elem:WMElemU8 bytes:seen length:sizeof seen rule:WMExact()];
    texture = nil;
    CFRelease(surface);
}

static id<MTLRenderPipelineState> StorageDepthPipeline(WMCase *c) {
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [c function:@"sto_fullscreen_vertex"];
    descriptor.fragmentFunction = [c function:@"sto_color_fragment"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Uint;
    descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    return [c renderPipeline:descriptor];
}

static id<MTLTexture> StorageAttachment(WMCase *c, MTLPixelFormat format, MTLStorageMode mode) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:8
                                                                                         height:8 mipmapped:NO];
    descriptor.storageMode = mode;
    descriptor.usage = MTLTextureUsageRenderTarget;
    return [c texture:descriptor];
}

// MTLResource.h, MTLStorageModeMemoryless: contents exist only during the
// render pass. A depth attachment that is cleared, tested and discarded
// within one pass gives a fully defined color result.
WMREF_CASE(memorylessDepth, "storage.memoryless-depth", "storage.memoryless",
           "A memoryless depth attachment cleared, tested and discarded within one render pass")
{
    // "Memoryless render targets": Apple2 and later (Metal Feature Set Tables, 2026-10-23).
    [c require:@"family.apple2" value:[c.device supportsFamily:MTLGPUFamilyApple2]];
    id<MTLTexture> color = StorageAttachment(c, MTLPixelFormatRGBA8Uint, MTLStorageModePrivate);
    id<MTLTexture> depth = StorageAttachment(c, MTLPixelFormatDepth32Float, MTLStorageModeMemoryless);
    [c value:@"depth.storage-mode" u64:depth.storageMode rule:WMExact()];
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = color;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.depthAttachment.texture = depth;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.clearDepth = 0.5;
    pass.depthAttachment.storeAction = MTLStoreActionDontCare;
    id<MTLRenderPipelineState> pipeline = StorageDepthPipeline(c);
    MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
    dd.depthCompareFunction = MTLCompareFunctionLess;
    dd.depthWriteEnabled = YES;
    id<MTLDepthStencilState> depthState = [c depthStencilState:dd];
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:pipeline];
        [e setDepthStencilState:depthState];
        const float depths[3] = {0.25f, 0.75f, 0.125f};
        for (uint32_t i = 0; i < 3; i++) {
            const uint32_t colorValue[4] = {i + 1, 10 * (i + 1), 100 * (i + 1), 255};
            [e setVertexBytes:&depths[i] length:sizeof(float) atIndex:0];
            [e setFragmentBytes:colorValue length:sizeof colorValue atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
    }];
    // Draw 1 passes (0.25 < 0.5), draw 2 fails (0.75), draw 3 passes (0.125).
    NSData *pixels = [c readTexture:color level:0 slice:0 bytesPerRow:8 * 4];
    [c data:@"color" elem:WMElemU8 bytes:pixels.bytes length:pixels.length
     layout:@{@"format": @"RGBA8Uint", @"width": @8, @"height": @8, @"bytes_per_row": @32} rule:WMExact()];
}

WMREF_CASE(memorylessStore, "storage.memoryless-store-action", "storage.errors",
           "Error path: a memoryless depth attachment with store action store")
{
    // "Memoryless render targets": Apple2 and later (Metal Feature Set Tables, 2026-10-23).
    [c require:@"family.apple2" value:[c.device supportsFamily:MTLGPUFamilyApple2]];
    id<MTLTexture> color = StorageAttachment(c, MTLPixelFormatRGBA8Uint, MTLStorageModePrivate);
    id<MTLTexture> depth = StorageAttachment(c, MTLPixelFormatDepth32Float, MTLStorageModeMemoryless);
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = color;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.depthAttachment.texture = depth;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    __block BOOL isNil = NO;
    __block id<MTLRenderCommandEncoder> encoder = nil;
    [c step:@"create-encoder"];
    [c call:@"create-encoder" rule:WMExact() block:^{
        encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
        isNil = encoder == nil;
    }];
    [c value:@"encoder-is-nil" boolean:isNil rule:WMNone(@"storing a memoryless attachment is outside the documented use")];
    if (encoder) [encoder endEncoding];
    [c step:@"commit"];
    [commandBuffer commit];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        [commandBuffer waitUntilCompleted];
        dispatch_semaphore_signal(done);
    });
    BOOL finished = StorageWait(done, 10);
    WMRule *undefined = WMNone(@"storing a memoryless attachment is outside the documented use");
    [c value:@"completed-within-10s" boolean:finished rule:undefined];
    if (finished) {
        [c value:@"status" u64:commandBuffer.status rule:undefined];
        [c error:@"error" value:commandBuffer.error rule:undefined];
    }
}
