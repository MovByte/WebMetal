// Encoder SPI (AGXA10FamilyBlitContext, ComputeContext, RenderContext and ThreadedRenderPass on 21H461) whose A10
// implementations do nothing or answer a constant.

#import "WebMetalSPI.h"

// The class adopts the driver's protocols; these categories implement the driver selectors they declare.
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

@implementation WebMetalBlitCommandEncoder (WebMetalSPI)

// -[AGXA10FamilyBlitContext commandBatchIdOffset] (AGXMetalA10 0x21c8baf38): the constant 0.
- (uint32_t)commandBatchIdOffset
{
    return 0;
}

@end

@implementation WebMetalComputeCommandEncoder (WebMetalSPI)

// -[AGXA10FamilyComputeContext commandBatchIdOffset] (AGXMetalA10 0x21c90c414): the constant 0.
- (uint32_t)commandBatchIdOffset
{
    return 0;
}

// -[AGXA10FamilyComputeContext nextVirtualSubstream] (AGXMetalA10 0x21c90c4d4): the constant ~0.
- (NSUInteger)nextVirtualSubstream
{
    return ~NSUInteger(0);
}

// -[AGXA10FamilyComputeContext parallelExecution] (AGXMetalA10 0x21c90bedc): the constant NO; -setParallelExecution:
// (0x21c90bed8) returns at once.
- (BOOL)parallelExecution
{
    return NO;
}

- (void)setParallelExecution:(BOOL)parallel
{
}

// -[AGXA10FamilyComputeContext dispatchWaitFlush], -dispatchFlushInvalidate, -dispatchFlushOnly,
// -dispatchInvalidateOnly and -dispatchFenceOnly (AGXMetalA10 0x21c90bec4 to 0x21c90bed4) return at once.
- (void)dispatchWaitFlush
{
}

- (void)dispatchFlushInvalidate
{
}

- (void)dispatchFlushOnly
{
}

- (void)dispatchInvalidateOnly
{
}

- (void)dispatchFenceOnly
{
}

@end

@implementation WebMetalRenderCommandEncoder (WebMetalSPI)

// -[AGXA10FamilyRenderContext useComputePipelineState:] and -useComputePipelineStates:count: (AGXMetalA10 0x21c9612b4,
// 0x21c9612b8) return at once.
- (void)useComputePipelineState:(id<MTLComputePipelineState>)pipeline
{
}

- (void)useComputePipelineStates:(const id<MTLComputePipelineState> __unsafe_unretained *)pipelines
                           count:(NSUInteger)count
{
}

@end

@implementation WebMetalParallelRenderCommandEncoder (WebMetalSPI)

// -[AGXA10FamilyThreadedRenderPass filterCounterRangeWithFirstBatch:lastBatch:filterIndex:] (AGXMetalA10 0x21c9c249c)
// returns at once.
- (void)filterCounterRangeWithFirstBatch:(uint32_t)first lastBatch:(uint32_t)last filterIndex:(uint32_t)index
{
}

// -[AGXA10FamilyThreadedRenderPass setMTLCounterSampleLocationWithBuffer:startIndex:endIndex:] (AGXMetalA10
// 0x21c9c24a0): the constant NO.
- (BOOL)setMTLCounterSampleLocationWithBuffer:(id)buffer startIndex:(uint32_t)start endIndex:(uint32_t)end
{
    return NO;
}

@end
