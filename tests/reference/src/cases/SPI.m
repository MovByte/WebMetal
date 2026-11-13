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
