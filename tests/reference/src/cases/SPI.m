// Driver SPI the system's frameworks use (QuartzCore, CoreImage, Metal
// Performance Shaders, AppleMetalGLRenderer, WebKit): device names, command
// queue GPU priorities, command buffer cache hints and the pipeline answers of
// devices without imageblocks. The SDK headers do not declare these
// selectors; each case declares the ones it sends and records a requirement
// that the device answers them. The answers come from the GPU's driver, so
// values carry the device rule; what Metal does with the command buffer is
// compared exactly.

#import "WMRefCase.h"

@protocol WMRefDeviceNames
- (NSString *)vendorName;
- (NSString *)familyName;
- (NSString *)productName;
@end

@protocol WMRefQueuePriority
- (BOOL)setGPUPriority:(NSUInteger)priority;
- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority;
- (NSUInteger)getGPUPriority;
- (NSUInteger)getBackgroundGPUPriority;
@end

@protocol WMRefCacheHints
- (void)encodeCacheHintTag:(NSUInteger)tag resourceGroups:(const id __unsafe_unretained *)groups count:(NSUInteger)count;
- (void)encodeCacheHintFinalize:(NSUInteger)hint
                 resourceGroups:(const id __unsafe_unretained *)groups
                          count:(NSUInteger)count;
@end

@protocol WMRefResourceGroups
- (id)newResourceGroupFromResources:(const id<MTLResource> __unsafe_unretained *)resources count:(NSUInteger)count;
@end

@protocol WMRefRenderPipelineSPI
- (NSUInteger)staticThreadgroupMemoryLength;
@end

static BOOL SPIAnswers(id object, SEL selector) {
    return [object respondsToSelector:selector];
}

WMREF_CASE(spiDeviceNames, "spi.device-names", "spi.device",
           "vendorName, familyName and productName of the device")
{
    id<WMRefDeviceNames> device = (id<WMRefDeviceNames>)c.device;
    [c require:@"spi.device-names" value:SPIAnswers(device, @selector(vendorName)) &&
        SPIAnswers(device, @selector(familyName)) && SPIAnswers(device, @selector(productName))];
    [c value:@"vendor-name" string:device.vendorName rule:WMDevice()];
    [c value:@"family-name" string:device.familyName rule:WMDevice()];
    [c value:@"product-name" string:device.productName rule:WMDevice()];
}

// The priorities and background priorities the queue reports after each
// request it accepts. Requests the driver validates as misuse (priority 3
// in the foreground, priorities above 5) are not made.
WMREF_CASE(spiQueuePriority, "spi.queue-priority", "spi.queue",
           "GPU priority and background GPU priority of a command queue: initial values and accepted changes")
{
    id<WMRefQueuePriority> queue = (id<WMRefQueuePriority>)[c.device newCommandQueue];
    [c require:@"spi.queue-priority" value:SPIAnswers(queue, @selector(setGPUPriority:)) &&
        SPIAnswers(queue, @selector(setBackgroundGPUPriority:)) && SPIAnswers(queue, @selector(getGPUPriority)) &&
        SPIAnswers(queue, @selector(getBackgroundGPUPriority))];
    [c value:@"initial.priority" u64:queue.getGPUPriority rule:WMDevice()];
    [c value:@"initial.background-priority" u64:queue.getBackgroundGPUPriority rule:WMDevice()];
    for (NSUInteger priority = 0; priority <= 4; priority++) {
        if (priority == 3) continue;
        NSString *name = [NSString stringWithFormat:@"set-%lu", (unsigned long)priority];
        [c value:[name stringByAppendingString:@".accepted"] boolean:[queue setGPUPriority:priority] rule:WMDevice()];
        [c value:[name stringByAppendingString:@".priority"] u64:queue.getGPUPriority rule:WMDevice()];
        [c value:[name stringByAppendingString:@".background-priority"] u64:queue.getBackgroundGPUPriority
            rule:WMDevice()];
    }
    for (NSUInteger priority = 0; priority <= 5; priority++) {
        NSString *name = [NSString stringWithFormat:@"set-background-%lu", (unsigned long)priority];
        [c value:[name stringByAppendingString:@".accepted"] boolean:[queue setBackgroundGPUPriority:priority]
            rule:WMDevice()];
        [c value:[name stringByAppendingString:@".priority"] u64:queue.getGPUPriority rule:WMDevice()];
        [c value:[name stringByAppendingString:@".background-priority"] u64:queue.getBackgroundGPUPriority
            rule:WMDevice()];
    }
}

// CoreAnimation's render server tags a command buffer's resources with cache
// hints and finalizes them; the groups come from
// newResourceGroupFromResources:count:, which may answer nil. The command
// buffer then runs its work as usual.
WMREF_CASE(spiCacheHints, "spi.cache-hints", "spi.command-buffer",
           "A command buffer with cache hint tags and finalizes over the device's resource groups")
{
    id<MTLBuffer> data = [c buffer:256 * sizeof(uint32_t)];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<WMRefCacheHints> hints = (id<WMRefCacheHints>)commandBuffer;
    id<WMRefResourceGroups> device = (id<WMRefResourceGroups>)c.device;
    [c require:@"spi.cache-hints" value:SPIAnswers(hints, @selector(encodeCacheHintTag:resourceGroups:count:)) &&
        SPIAnswers(hints, @selector(encodeCacheHintFinalize:resourceGroups:count:)) &&
        SPIAnswers(device, @selector(newResourceGroupFromResources:count:))];
    id<MTLResource> __unsafe_unretained resources[1] = {data};
    id group = [device newResourceGroupFromResources:resources count:1];
    [c value:@"resource-group-is-nil" boolean:group == nil rule:WMDevice()];
    id __unsafe_unretained groups[1] = {group};
    [hints encodeCacheHintTag:1 resourceGroups:groups count:1];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:[c computePipeline:@"cmd_add_value"]];
    [encoder setBuffer:data offset:0 atIndex:0];
    const uint32_t value = 5;
    [encoder setBytes:&value length:sizeof value atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    [hints encodeCacheHintFinalize:1 resourceGroups:groups count:1];
    [c commitAndWait:commandBuffer];
    [c value:@"status" u64:commandBuffer.status rule:WMExact()];
    [c data:@"buffer" elem:WMElemU32 bytes:data.contents length:data.length rule:WMExact()];
}

// MTLRenderPipelineState.h: imageblockSampleLength is the imageblock's size
// per sample; a pipeline without a tile function or imageblock has none.
WMREF_CASE(spiRenderPipelineAnswers, "spi.render-pipeline-answers", "spi.pipelines",
           "imageblockSampleLength and staticThreadgroupMemoryLength of a render pipeline without tile functions")
{
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [c function:@"pip_simple_vertex"];
    descriptor.fragmentFunction = [c function:@"pip_simple_fragment"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:descriptor];
    [c require:@"spi.render-pipeline-answers" value:SPIAnswers(pipeline, @selector(staticThreadgroupMemoryLength))];
    [c value:@"imageblock-sample-length" u64:pipeline.imageblockSampleLength rule:WMDevice()];
    [c value:@"static-threadgroup-memory-length"
         u64:[(id<WMRefRenderPipelineSPI>)pipeline staticThreadgroupMemoryLength] rule:WMDevice()];
}

@protocol WMRefCommitAndWait
- (void)commitAndWaitUntilSubmitted;
@end

static void SPIEncodeAdd(WMCase *c, id<MTLCommandBuffer> commandBuffer, id<MTLBuffer> buffer, uint32_t value) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:[c computePipeline:@"cmd_add_value"]];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&value length:sizeof value atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
}

// commitAndWaitUntilSubmitted commits the command buffer and submits it to
// its queue on the calling thread before returning. The kernel times are
// set when a command buffer is scheduled: positive, ordered within a buffer,
// and in submission order across the buffers of one queue.
WMREF_CASE(spiCommitAndWaitUntilSubmitted, "spi.commit-and-wait-until-submitted", "spi.command-buffer",
           "commitAndWaitUntilSubmitted followed by an ordinary commit on the same queue, and their kernel times")
{
    id<MTLBuffer> data = [c buffer:256 * sizeof(uint32_t)];
    id<MTLCommandBuffer> first = [c commandBuffer];
    [c require:@"spi.commit-and-wait-until-submitted"
         value:SPIAnswers(first, @selector(commitAndWaitUntilSubmitted))];
    __block CFTimeInterval scheduledKernelStart = 0;
    [first addScheduledHandler:^(id<MTLCommandBuffer> buffer) {
        scheduledKernelStart = buffer.kernelStartTime;
    }];
    SPIEncodeAdd(c, first, data, 1);
    [(id<WMRefCommitAndWait>)first commitAndWaitUntilSubmitted];
    [c value:@"first.status-after-return-is-committed-or-later" boolean:first.status >= MTLCommandBufferStatusCommitted
        rule:WMExact()];
    id<MTLCommandBuffer> second = [c commandBuffer];
    SPIEncodeAdd(c, second, data, 2);
    [second commit];
    [first waitUntilCompleted];
    [second waitUntilCompleted];
    [c value:@"first.status" u64:first.status rule:WMExact()];
    [c value:@"second.status" u64:second.status rule:WMExact()];
    [c data:@"buffer" elem:WMElemU32 bytes:data.contents length:data.length rule:WMExact()];
    [c value:@"first.kernel-start-positive" boolean:first.kernelStartTime > 0 rule:WMExact()];
    [c value:@"first.kernel-end-not-before-start" boolean:first.kernelEndTime >= first.kernelStartTime rule:WMExact()];
    [c value:@"first.scheduled-handler-sees-kernel-start" boolean:scheduledKernelStart > 0 &&
        scheduledKernelStart == first.kernelStartTime rule:WMExact()];
    [c value:@"second.kernel-start-not-before-first" boolean:second.kernelStartTime >= first.kernelStartTime
        rule:WMExact()];
}

@protocol WMRefFeatureProbes
- (BOOL)supportsMemorylessRenderTargets;
@end

static MTLRenderPipelineDescriptor *SPICoverDescriptor(WMCase *c, NSString *fragment, BOOL depth) {
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [c function:@"spi_cover_vertex"];
    d.fragmentFunction = [c function:fragment];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    if (depth) d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    return d;
}

static NSData *SPIDrawCover(WMCase *c, id<MTLRenderPipelineState> pipeline, id<MTLTexture> depthTarget) {
    MTLTextureDescriptor *cd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                   width:8 height:8 mipmapped:NO];
    cd.usage = MTLTextureUsageRenderTarget;
    cd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> color = [c texture:cd];
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = color;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLDepthStencilState> depthState = nil;
    if (depthTarget) {
        pass.depthAttachment.texture = depthTarget;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.clearDepth = 1.0;
        pass.depthAttachment.storeAction = MTLStoreActionDontCare;
        MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
        dd.depthCompareFunction = MTLCompareFunctionLess;
        dd.depthWriteEnabled = YES;
        depthState = [c depthStencilState:dd];
    }
    [c render:pass body:^(id<MTLRenderCommandEncoder> encoder) {
        [encoder setRenderPipelineState:pipeline];
        if (depthState) [encoder setDepthStencilState:depthState];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    return [c readTexture:color level:0 slice:0 bytesPerRow:8 * 4];
}

// QuartzCore tests supportsMemorylessRenderTargets before it renders with a
// memoryless depth attachment, whose contents exist only during the pass
// (MTLResource.h, MTLStorageModeMemoryless): cleared at load, not stored.
// The second draw fails the depth test against the first, so the target
// holds the fragment's color once.
WMREF_CASE(spiMemorylessTarget, "spi.memoryless-render-target", "spi.feature-probes",
           "A render pass with a memoryless depth attachment, on a device that reports memoryless render targets")
{
    id<WMRefFeatureProbes> device = (id<WMRefFeatureProbes>)c.device;
    [c require:@"spi.memoryless-render-target" value:SPIAnswers(device, @selector(supportsMemorylessRenderTargets)) &&
        device.supportsMemorylessRenderTargets];
    MTLTextureDescriptor *dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                   width:8 height:8 mipmapped:NO];
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModeMemoryless;
    id<MTLTexture> depth = [c texture:dd];
    [c value:@"depth.storage-mode" u64:depth.storageMode rule:WMExact()];
    id<MTLRenderPipelineState> pipeline = [c renderPipeline:SPICoverDescriptor(c, @"spi_color_fragment", YES)];
    NSData *pixels = SPIDrawCover(c, pipeline, depth);
    [c data:@"color" elem:WMElemU8 bytes:pixels.bytes length:pixels.length rule:WMExact()];
}

// QuartzCore tests supportsRasterOrderGroups before it uses raster order
// groups. A device that reports them makes the pipeline and orders the
// fragments; what a device that does not report them does with such a
// pipeline is the driver's.
WMREF_CASE(spiRasterOrderGroups, "spi.raster-order-groups", "spi.feature-probes",
           "A render pipeline whose fragment color output is in a raster order group, and a draw with it")
{
    [c value:@"raster-order-groups-supported" boolean:c.device.rasterOrderGroupsSupported rule:WMDevice()];
    NSError *error = nil;
    id<MTLRenderPipelineState> pipeline =
        [c.device newRenderPipelineStateWithDescriptor:SPICoverDescriptor(c, @"spi_ordered_fragment", NO) error:&error];
    [c value:@"pipeline-made" boolean:pipeline != nil rule:WMDevice()];
    [c error:@"pipeline-error" value:error rule:WMDevice()];
    if (!pipeline) return;
    NSData *pixels = SPIDrawCover(c, pipeline, nil);
    [c data:@"color" elem:WMElemU8 bytes:pixels.bytes length:pixels.length rule:WMExact()];
}
