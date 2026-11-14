// Encoder SPI (AGXA10FamilyBlitContext, ComputeContext, RenderContext and ThreadedRenderPass on 21H461) whose A10
// implementations do nothing or answer a constant.

#import "WebMetalSPI.h"
#import "../Commands/WebMetalCommandState.h"
#import "../Resources/WebMetalCounterSampleBuffer.h"

// The class adopts the driver's protocols; these categories implement the driver selectors they declare.
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

namespace {

// The encoders' -setMTLCounterSampleLocationWithBuffer:... with a buffer: the driver keeps the indices, checks each in
// order against the buffer's sampleCount (validateSampleIndex, 0x21c9ced9c, which logs one past the samples and makes it
// MTLCounterDontSample) and, when one is a sample, attaches the buffer's storage to the encoder (setCounterSampleBuffer,
// 0x21c9ceaf4), answering whether it did; the GPU then writes timestamps into the storage, which the encoder's
// endEncoding refuses.
BOOL setSampleLocation(webmetal::EncoderState &encoder, id buffer, std::initializer_list<uint32_t> values)
{
    bool sampled = false;
    for (uint32_t value : values) {
        uint32_t index = value;
        sampled |= webmetal::ValidateSampleIndex(buffer, &index);
    }
    return sampled && webmetal::AttachSampleStorage(&encoder.sampleStorage, buffer);
}

}  // namespace

@implementation AGXA10FamilyBlitContext (WebMetalSPI)

// -[AGXA10FamilyBlitContext setMTLCounterSampleLocationWithBuffer:startIndex:endIndex:] (AGXMetalA10 0x21c8baf5c):
// NO for a nil buffer, before anything is kept.
- (BOOL)setMTLCounterSampleLocationWithBuffer:(id)buffer startIndex:(uint32_t)start endIndex:(uint32_t)end
{
    if (!buffer)
        return NO;
    return setSampleLocation(*_encoder, buffer, {start, end});
}

// -[AGXA10FamilyBlitContext commandBatchIdOffset] (AGXMetalA10 0x21c8baf38): the constant 0.
- (uint32_t)commandBatchIdOffset
{
    return 0;
}

@end

@implementation AGXA10FamilyComputeContext (WebMetalSPI)

// -[AGXA10FamilyComputeContext setMTLCounterSampleLocationWithBuffer:startIndex:endIndex:] (AGXMetalA10 0x21c90c438):
// NO for a nil buffer, before anything is kept.
- (BOOL)setMTLCounterSampleLocationWithBuffer:(id)buffer startIndex:(uint32_t)start endIndex:(uint32_t)end
{
    if (!buffer)
        return NO;
    return setSampleLocation(*_encoder, buffer, {start, end});
}

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

@implementation AGXA10FamilyRenderContext (WebMetalSPI)

// -[AGXA10FamilyRenderContext setMTLCounterSampleLocationWithBuffer:startIndex:endIndex:dm:] (AGXMetalA10 0x21c9652a8):
// the A10 family's render encoders take the vertex and fragment form; the driver logs that (agxa_render_objc.mm line
// 1268) and raises through -doesNotRecognizeSelector:.
- (BOOL)setMTLCounterSampleLocationWithBuffer:(id)buffer
                                   startIndex:(uint32_t)start
                                     endIndex:(uint32_t)end
                                           dm:(uint32_t)dm
{
    WM_AGX_LOG("agxa_render_objc.mm", 1268,
               "-[AGXA10FamilyRenderContext setMTLCounterSampleLocationWithBuffer:startIndex:endIndex:dm:]",
               "Please use setMTLCounterSampleLocationWithBuffer:vertexStartIndex:vertexEndIndex:fragmentStartIndex:"
               "fragmentEndIndex:");
    [self doesNotRecognizeSelector:_cmd];
    return NO;
}

// -[AGXA10FamilyRenderContext setMTLCounterSampleLocationWithBuffer:vertexStartIndex:vertexEndIndex:fragmentStartIndex:
// fragmentEndIndex:] (AGXMetalA10 0x21c965420): NO for a nil buffer, before anything is kept.
- (BOOL)setMTLCounterSampleLocationWithBuffer:(id)buffer
                             vertexStartIndex:(uint32_t)vertexStart
                               vertexEndIndex:(uint32_t)vertexEnd
                           fragmentStartIndex:(uint32_t)fragmentStart
                             fragmentEndIndex:(uint32_t)fragmentEnd
{
    if (!buffer)
        return NO;
    return setSampleLocation(*_encoder, buffer, {vertexStart, vertexEnd, fragmentStart, fragmentEnd});
}

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

@implementation AGXA10FamilyThreadedRenderPass (WebMetalSPI)

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
