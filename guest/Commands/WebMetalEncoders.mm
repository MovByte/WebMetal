// The command encoders (AGXA10FamilyRenderContext, ComputeContext, BlitContext and ThreadedRenderPass on 21H461;
// WebMetalCommands.h) and the command-buffer methods that create them.
//
// _MTLCommandEncoder's initializer makes the encoder its command buffer's current encoder and its -endEncoding clears
// it; its -dealloc asserts that the encoder ended. Each encoder writes a Begin* record when it is created, one record
// per command (guest/Commands/WebMetalEncoderRecords.mm) and EncoderEndEncoding when it ends. A parallel
// render encoder's sub-encoders each live in a command buffer of their own that Metal.framework creates
// (-[_MTLParallelRenderCommandEncoder _renderCommandEncoderCommon]); they record into their own streams, which the
// parallel encoder splices into its command buffer's stream, in creation order, when it ends.

#import "WebMetalCommandState.h"
#import "../Plugin/WebMetalReport.h"

#import <objc/runtime.h>

#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

// Getters of 21H461's pass descriptor classes that the SDK headers do not declare (the wire schema's
// RenderPassDescriptor, ComputePassDescriptor and BlitPassDescriptor list the state they carry).
@protocol WebMetalRenderPassDescriptorState
- (NSUInteger)defaultColorSampleCount;
- (BOOL)ditherEnabled;
- (BOOL)fineGrainedBackgroundVisibilityEnabled;
- (BOOL)openGLModeEnabled;
- (BOOL)pointCoordYFlipEnabled;
- (BOOL)skipEmptyTilesOnClearEnabled;
- (NSArray *)binaryArchives;
@end

@protocol WebMetalAttachmentState
- (BOOL)yInvert;
@end

@protocol WebMetalComputePassDescriptorState
- (BOOL)allowCommandEncoderCoalescing;
- (uint32_t)substreamCount;
@end

@protocol WebMetalDeviceLimits
- (NSUInteger)maxColorAttachments;
@end

namespace webmetal {

namespace {

// Entries of the render, compute and blit pass descriptors' sampleBufferAttachments arrays: 21H461's
// -[MTL{Render,Compute,Blit}PassSampleBufferAttachmentDescriptorArrayInternal objectAtIndexedSubscript:] accept indices
// below 4 (cmp x2, #0x4 at 0x187b9b914, 0x187b5c834, 0x187bb2914).
constexpr NSUInteger kPassSampleBuffers = 4;

// _MTLCommandEncoder._globalTraceObjectID, which IOGPU fills lazily from the device's trace ID ranges.
uint64_t encoderTraceID(id encoder)
{
    static std::atomic<uint64_t> next{1};
    static const ptrdiff_t offset =
        ivar_getOffset(class_getInstanceVariable(objc_getClass("_MTLCommandEncoder"), "_globalTraceObjectID"));
    uint64_t *slot = (uint64_t *)((uint8_t *)(__bridge void *)encoder + offset);
    if (!*slot)
        *slot = next.fetch_add(1);
    return *slot;
}

void checkNoOpenEncoder(WebMetalCommandBuffer *buffer, const char *function)
{
    if (buffer->_currentEncoder)
        MTLReportFailure(0, function, __LINE__, @"A command encoder is already encoding to this command buffer");
}

void endEncoder(id self, SEL cmd, EncoderState &encoder)
{
    EmitEncoderRecord(self, cmd, encoder, wire::EncoderEndEncoding{});
    encoder.ended = true;
    encoder.stream->endEncoder();
}

void setEncoderLabel(id self, SEL cmd, EncoderState &encoder, NSString *label)
{
    wire::EncoderSetLabel record;
    if (!wire::strFromNSString(label, record.label))
        WMLoudMethod(self, cmd, "the label has no UTF-8 form");
    EmitEncoderRecord(self, cmd, encoder, record);
}

void pushEncoderDebugGroup(id self, SEL cmd, EncoderState &encoder, NSString *string)
{
    wire::EncoderPushDebugGroup record;
    if (!wire::strFromNSString(string, record.string))
        WMLoudMethod(self, cmd, "the string has no UTF-8 form");
    EmitEncoderRecord(self, cmd, encoder, record);
}

// Each render segment lists the pass's attachment textures with flags 12, except memoryless ones, and a resolve texture
// with 12 when its attachment has a texture: AGX::Framebuffer's listing (0x21c919170, called with mask 0x3f from the
// segment's begin at 0x21c96e300) skips the attachments its exclusion mask fb+3100 holds, which are the memoryless ones
// (0x21c914848 to 0x21c914860, 0x21c915150, 0x21c915240), and lists colour resolve textures by fb+2440, which holds the
// attachments with both a texture and a resolve texture (0x21c91490c). Whether a depth or stencil resolve texture
// without its attachment's texture is listed is not read.
template <class Attachment> void attachmentCommon(id self, SEL cmd, Stream &stream, Attachment &out,
                                                  MTLRenderPassAttachmentDescriptor *in)
{
    const bool memoryless = in.texture && in.texture.storageMode == MTLStorageModeMemoryless;
    out.texture = Reference(stream, self, cmd, in.texture, memoryless ? Listing::Unlisted : Listing::Stamped);
    out.level = in.level;
    out.slice = in.slice;
    out.depthPlane = in.depthPlane;
    out.resolveTexture =
        Reference(stream, self, cmd, in.resolveTexture, in.texture ? Listing::Stamped : Listing::Untraced);
    out.resolveLevel = in.resolveLevel;
    out.resolveSlice = in.resolveSlice;
    out.resolveDepthPlane = in.resolveDepthPlane;
    out.loadAction = in.loadAction;
    out.storeAction = in.storeAction;
    out.storeActionOptions = in.storeActionOptions;
    out.yInvert = [(id<WebMetalAttachmentState>)in yInvert];
}

// The wire form of a render pass descriptor and the vectors its spans point into.
struct RenderPass {
    wire::RenderPassDescriptor descriptor;
    std::vector<wire::RenderPassColorAttachment> colors;
    std::vector<wire::RenderPassSampleBufferAttachment> sampleBuffers;
    std::vector<wire::SamplePosition> positions;
    std::vector<uint64_t> archives;
};

void renderPass(id self, SEL cmd, Stream &stream, id<MTLDevice> device, MTLRenderPassDescriptor *in, RenderPass &out)
{
    id<WebMetalRenderPassDescriptorState> state = (id<WebMetalRenderPassDescriptorState>)in;
    wire::RenderPassDescriptor &d = out.descriptor;
    const NSUInteger colorCount = [(id<WebMetalDeviceLimits>)device maxColorAttachments];
    out.colors.resize(colorCount);
    for (NSUInteger i = 0; i < colorCount; i++) {
        MTLRenderPassColorAttachmentDescriptor *color = in.colorAttachments[i];
        attachmentCommon(self, cmd, stream, out.colors[i], color);
        const MTLClearColor c = color.clearColor;
        out.colors[i].clearColor = {c.red, c.green, c.blue, c.alpha};
    }
    d.colorAttachments = {out.colors.data(), out.colors.size()};
    if (MTLRenderPassDepthAttachmentDescriptor *depth = in.depthAttachment) {
        d.depthAttachment.present = true;
        attachmentCommon(self, cmd, stream, d.depthAttachment.value, depth);
        d.depthAttachment.value.clearDepth = depth.clearDepth;
        d.depthAttachment.value.depthResolveFilter = depth.depthResolveFilter;
    }
    if (MTLRenderPassStencilAttachmentDescriptor *stencil = in.stencilAttachment) {
        d.stencilAttachment.present = true;
        attachmentCommon(self, cmd, stream, d.stencilAttachment.value, stencil);
        d.stencilAttachment.value.clearStencil = stencil.clearStencil;
        d.stencilAttachment.value.stencilResolveFilter = stencil.stencilResolveFilter;
    }
    // Listed only once the encoder sets a visibility result mode (NoteRecordListing).
    d.visibilityResultBuffer = Reference(stream, self, cmd, in.visibilityResultBuffer, Listing::Unlisted);
    d.renderTargetArrayLength = in.renderTargetArrayLength;
    d.imageblockSampleLength = in.imageblockSampleLength;
    d.threadgroupMemoryLength = in.threadgroupMemoryLength;
    d.tileWidth = in.tileWidth;
    d.tileHeight = in.tileHeight;
    d.defaultRasterSampleCount = in.defaultRasterSampleCount;
    d.renderTargetWidth = in.renderTargetWidth;
    d.renderTargetHeight = in.renderTargetHeight;
    d.rasterizationRateMap = Reference(stream, self, cmd, in.rasterizationRateMap);
    d.defaultColorSampleCount = [state defaultColorSampleCount];
    d.ditherEnabled = [state ditherEnabled];
    d.fineGrainedBackgroundVisibilityEnabled = [state fineGrainedBackgroundVisibilityEnabled];
    d.openGlModeEnabled = [state openGLModeEnabled];
    d.pointCoordYFlipEnabled = [state pointCoordYFlipEnabled];
    d.skipEmptyTilesOnClearEnabled = [state skipEmptyTilesOnClearEnabled];
    // 21H461's descriptor class has binaryArchives; a framework without the property cannot have been given any.
    if ([(id)in respondsToSelector:@selector(binaryArchives)])
        for (id archive in [state binaryArchives])
            out.archives.push_back(Reference(stream, self, cmd, archive));
    d.binaryArchives = {out.archives.data(), out.archives.size()};
    for (NSUInteger i = 0; i < kPassSampleBuffers; i++) {
        MTLRenderPassSampleBufferAttachmentDescriptor *attachment = in.sampleBufferAttachments[i];
        wire::RenderPassSampleBufferAttachment s;
        s.sampleBuffer = Reference(stream, self, cmd, attachment.sampleBuffer);
        s.startOfVertexSampleIndex = attachment.startOfVertexSampleIndex;
        s.endOfVertexSampleIndex = attachment.endOfVertexSampleIndex;
        s.startOfFragmentSampleIndex = attachment.startOfFragmentSampleIndex;
        s.endOfFragmentSampleIndex = attachment.endOfFragmentSampleIndex;
        out.sampleBuffers.push_back(s);
    }
    d.sampleBufferAttachments = {out.sampleBuffers.data(), out.sampleBuffers.size()};
    const NSUInteger positions = [in getSamplePositions:nullptr count:0];
    if (positions) {
        std::vector<MTLSamplePosition> raw(positions);
        [in getSamplePositions:raw.data() count:positions];
        for (const MTLSamplePosition &p : raw)
            out.positions.push_back({p.x, p.y});
    }
    d.samplePositions = {out.positions.data(), out.positions.size()};
}

}  // namespace

}  // namespace webmetal

namespace webmetal {

namespace {

const char kNoAccessFlags[] =
    "the A10X driver lists the resource with no access flags, which the kernel's parse of the segment's resource list "
    "refuses (AGXAllocationList2, kernelcache 0xfffffff005bf1af8 to 0xfffffff005bf1b04); what the kernel does with the "
    "command buffer then is not implemented";

bool anyNamed(const wire::Span<uint64_t> &handles)
{
    for (size_t i = 0; i < handles.size; i++)
        if (handles.data[i])
            return true;
    return false;
}

// -[AGXA10FamilyRenderContext useResource:usage:stages:] (0x21c960d70 to 0x21c960dc0) and -useResources:count:usage:
// stages: (0x21c960e28 to 0x21c960e84) list with read and write bits for the vertex (0, 1) and fragment or tile (2, 3)
// stages, which are all clear unless the usage has a read, write or sample bit and the stages a vertex, fragment or
// tile bit.
bool renderUseListed(uint64_t usage, uint64_t stages)
{
    return (usage & 7) && (stages & 7);
}

}  // namespace

void NoteRecordListing(id self, SEL cmd, EncoderState &, const wire::RenderUseResourceStages &record)
{
    if (record.resource && !renderUseListed(record.usage, record.stages))
        WMLoudMethod(self, cmd, kNoAccessFlags);
}

void NoteRecordListing(id self, SEL cmd, EncoderState &, const wire::RenderUseResourcesStages &record)
{
    if (anyNamed(record.resources) && !renderUseListed(record.usage, record.stages))
        WMLoudMethod(self, cmd, kNoAccessFlags);
}

// -[AGXA10FamilyRenderContext useHeap:stages:] (0x21c960f58 to 0x21c960f68) and -useHeaps:count:stages: (0x21c96106c to
// 0x21c96107c): read bits for the vertex (0) and fragment or tile (2) stages.
void NoteRecordListing(id self, SEL cmd, EncoderState &, const wire::RenderUseHeapStages &record)
{
    if (record.heap && !(record.stages & 7))
        WMLoudMethod(self, cmd, kNoAccessFlags);
}

void NoteRecordListing(id self, SEL cmd, EncoderState &, const wire::RenderUseHeapsStages &record)
{
    if (anyNamed(record.heaps) && !(record.stages & 7))
        WMLoudMethod(self, cmd, kNoAccessFlags);
}

// -[AGXA10FamilyComputeContext useResource:usage:] (0x21c90c0b8 to 0x21c90c0cc) and -useResources:count:usage:
// (0x21c90c134 to 0x21c90c140): (usage & 2) | ((usage & 7) != 0).
void NoteRecordListing(id self, SEL cmd, EncoderState &, const wire::ComputeUseResource &record)
{
    if (record.resource && !(record.usage & 7))
        WMLoudMethod(self, cmd, kNoAccessFlags);
}

void NoteRecordListing(id self, SEL cmd, EncoderState &, const wire::ComputeUseResources &record)
{
    if (anyNamed(record.resources) && !(record.usage & 7))
        WMLoudMethod(self, cmd, kNoAccessFlags);
}

// -[AGXA10FamilyRenderContext setVisibilityResultMode:offset:] lists the pass's visibility result buffer with flags 15
// (0x21c962c88) once a mode other than Disabled has been set in the encoder (ctx+2407, 0x21c962c34 to 0x21c962c44), in
// each segment that has the buffer (0x21c96e074 to 0x21c96e088, 0x21c96cc68); nothing else lists it.
void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::RenderSetVisibilityResultMode &record)
{
    if (record.mode != MTLVisibilityResultModeDisabled && encoder.visibilityResultKey)
        encoder.stream->noteUse(encoder.visibilityResultKey, Stream::UseKind::Stamped);
}

}  // namespace webmetal

using namespace webmetal;

@implementation WebMetalCommandBuffer (WebMetalEncoders)

- (id<MTLBlitCommandEncoder>)blitCommandEncoder
{
    checkNoOpenEncoder(self, "-[WebMetalCommandBuffer blitCommandEncoder]");
    WebMetalBlitCommandEncoder *encoder = [[WebMetalBlitCommandEncoder alloc] initWithCommandBuffer:self descriptor:nil];
    wire::BeginBlit record;
    record.encoder = encoder->_encoder->handle;
    EmitEncoderRecord(self, _cmd, *encoder->_encoder, record);
    return encoder;
}

- (id<MTLBlitCommandEncoder>)blitCommandEncoderWithDescriptor:(MTLBlitPassDescriptor *)descriptor
{
    checkNoOpenEncoder(self, "-[WebMetalCommandBuffer blitCommandEncoderWithDescriptor:]");
    WebMetalBlitCommandEncoder *encoder = [[WebMetalBlitCommandEncoder alloc] initWithCommandBuffer:self
                                                                                        descriptor:descriptor];
    EncoderState &state = *encoder->_encoder;
    wire::BeginBlitWithDescriptor record;
    record.encoder = state.handle;
    std::vector<wire::BlitPassSampleBufferAttachment> attachments;
    for (NSUInteger i = 0; i < kPassSampleBuffers; i++) {
        MTLBlitPassSampleBufferAttachmentDescriptor *in = descriptor.sampleBufferAttachments[i];
        wire::BlitPassSampleBufferAttachment out;
        out.sampleBuffer = Reference(*state.stream, self, _cmd, in.sampleBuffer);
        out.startOfEncoderSampleIndex = in.startOfEncoderSampleIndex;
        out.endOfEncoderSampleIndex = in.endOfEncoderSampleIndex;
        attachments.push_back(out);
    }
    record.blitPassDescriptor.sampleBufferAttachments = {attachments.data(), attachments.size()};
    EmitEncoderRecord(self, _cmd, state, record);
    return encoder;
}

- (id<MTLComputeCommandEncoder>)computeCommandEncoder
{
    checkNoOpenEncoder(self, "-[WebMetalCommandBuffer computeCommandEncoder]");
    WebMetalComputeCommandEncoder *encoder = [[WebMetalComputeCommandEncoder alloc] initWithCommandBuffer:self];
    wire::BeginCompute record;
    record.encoder = encoder->_encoder->handle;
    EmitEncoderRecord(self, _cmd, *encoder->_encoder, record);
    return encoder;
}

- (id<MTLComputeCommandEncoder>)computeCommandEncoderWithDispatchType:(MTLDispatchType)dispatchType
{
    checkNoOpenEncoder(self, "-[WebMetalCommandBuffer computeCommandEncoderWithDispatchType:]");
    WebMetalComputeCommandEncoder *encoder = [[WebMetalComputeCommandEncoder alloc] initWithCommandBuffer:self];
    encoder->_dispatchType = dispatchType;
    wire::BeginComputeWithDispatchType record;
    record.encoder = encoder->_encoder->handle;
    record.dispatchType = dispatchType;
    EmitEncoderRecord(self, _cmd, *encoder->_encoder, record);
    return encoder;
}

- (id<MTLComputeCommandEncoder>)computeCommandEncoderWithDispatchType:(MTLDispatchType)dispatchType
                                                       substreamCount:(uint32_t)substreamCount
{
    checkNoOpenEncoder(self, "-[WebMetalCommandBuffer computeCommandEncoderWithDispatchType:substreamCount:]");
    WebMetalComputeCommandEncoder *encoder = [[WebMetalComputeCommandEncoder alloc] initWithCommandBuffer:self];
    encoder->_dispatchType = dispatchType;
    wire::BeginComputeWithDispatchTypeSubstreams record;
    record.encoder = encoder->_encoder->handle;
    record.dispatchType = dispatchType;
    record.substreamCount = substreamCount;
    EmitEncoderRecord(self, _cmd, *encoder->_encoder, record);
    return encoder;
}

- (id<MTLComputeCommandEncoder>)computeCommandEncoderWithDescriptor:(MTLComputePassDescriptor *)descriptor
{
    checkNoOpenEncoder(self, "-[WebMetalCommandBuffer computeCommandEncoderWithDescriptor:]");
    WebMetalComputeCommandEncoder *encoder = [[WebMetalComputeCommandEncoder alloc] initWithCommandBuffer:self];
    EncoderState &state = *encoder->_encoder;
    encoder->_dispatchType = descriptor.dispatchType;
    wire::BeginComputeWithDescriptor record;
    record.encoder = state.handle;
    id<WebMetalComputePassDescriptorState> extra = (id<WebMetalComputePassDescriptorState>)descriptor;
    record.computePassDescriptor.dispatchType = descriptor.dispatchType;
    record.computePassDescriptor.allowCommandEncoderCoalescing = [extra allowCommandEncoderCoalescing];
    record.computePassDescriptor.substreamCount = [extra substreamCount];
    std::vector<wire::ComputePassSampleBufferAttachment> attachments;
    for (NSUInteger i = 0; i < kPassSampleBuffers; i++) {
        MTLComputePassSampleBufferAttachmentDescriptor *in = descriptor.sampleBufferAttachments[i];
        wire::ComputePassSampleBufferAttachment out;
        out.sampleBuffer = Reference(*state.stream, self, _cmd, in.sampleBuffer);
        out.startOfEncoderSampleIndex = in.startOfEncoderSampleIndex;
        out.endOfEncoderSampleIndex = in.endOfEncoderSampleIndex;
        attachments.push_back(out);
    }
    record.computePassDescriptor.sampleBufferAttachments = {attachments.data(), attachments.size()};
    EmitEncoderRecord(self, _cmd, state, record);
    return encoder;
}

- (id<MTLRenderCommandEncoder>)renderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)descriptor
{
    checkNoOpenEncoder(self, "-[WebMetalCommandBuffer renderCommandEncoderWithDescriptor:]");
    WebMetalRenderCommandEncoder *encoder = [[WebMetalRenderCommandEncoder alloc] initWithCommandBuffer:self
                                                                                            descriptor:descriptor];
    EncoderState &state = *encoder->_encoder;
    RenderPass pass;
    renderPass(self, _cmd, *state.stream, self.device, descriptor, pass);
    state.visibilityResultKey = KernelResourceKey(descriptor.visibilityResultBuffer);
    wire::BeginRenderPass record;
    record.encoder = state.handle;
    record.renderPassDescriptor = pass.descriptor;
    EmitEncoderRecord(self, _cmd, state, record);
    return encoder;
}

- (id<MTLParallelRenderCommandEncoder>)parallelRenderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)descriptor
{
    checkNoOpenEncoder(self, "-[WebMetalCommandBuffer parallelRenderCommandEncoderWithDescriptor:]");
    WebMetalParallelRenderCommandEncoder *encoder =
        [[WebMetalParallelRenderCommandEncoder alloc] initWithCommandBuffer:self renderPassDescriptor:descriptor];
    EncoderState &state = *encoder->_encoder;
    RenderPass pass;
    renderPass(self, _cmd, *state.stream, self.device, descriptor, pass);
    wire::BeginParallelRenderPass record;
    record.encoder = state.handle;
    record.renderPassDescriptor = pass.descriptor;
    EmitEncoderRecord(self, _cmd, state, record);
    return encoder;
}

// -[AGXA10FamilyCommandBuffer subRenderCommandEncoderWithDescriptor:subEncoderIndex:framebuffer:]: the render encoder
// of one sub-encoder of a parallel render encoder, created in the command buffer Metal.framework made for it. The
// framebuffer argument is the parallel encoder's shared state, here the parallel encoder itself.
- (id<MTLRenderCommandEncoder>)subRenderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)descriptor
                                                     subEncoderIndex:(NSUInteger)index
                                                         framebuffer:(void *)framebuffer
{
    checkNoOpenEncoder(self, "-[WebMetalCommandBuffer subRenderCommandEncoderWithDescriptor:subEncoderIndex:framebuffer:]");
    return [[WebMetalRenderCommandEncoder alloc] initWithCommandBuffer:self
                                                           descriptor:descriptor
                                                      subEncoderIndex:index
                                                          framebuffer:framebuffer];
}

@end

// Blit -----------------------------------------------------------------------------------------------------------------

@implementation WebMetalBlitCommandEncoder

// -[AGXA10FamilyBlitContext initWithCommandBuffer:descriptor:]: the framework encoder, then the driver's blit state.
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer descriptor:(MTLBlitPassDescriptor *)descriptor
{
    self = [super initWithCommandBuffer:commandBuffer];
    if (!self)
        return nil;
    _encoder = NewEncoderState((WebMetalCommandBuffer *)commandBuffer, nullptr);
    return self;
}

- (void)dealloc
{
    if (_encoder && _encoder->releases)
        _encoder->releases->dead(_encoder->handle);
}

- (NSUInteger)getType
{
    return 2;  // -[IOGPUMetalBlitCommandEncoder getType]
}

- (uint64_t)globalTraceObjectID
{
    return encoderTraceID(self);
}

- (void)endEncoding
{
    endEncoder(self, _cmd, *_encoder);
    [super endEncoding];
}

- (void)setLabel:(NSString *)label
{
    [super setLabel:label];
    setEncoderLabel(self, _cmd, *_encoder, label);
}

- (void)pushDebugGroup:(NSString *)string
{
    pushEncoderDebugGroup(self, _cmd, *_encoder, string);
    [super pushDebugGroup:string];
}

- (void)popDebugGroup
{
    EmitEncoderRecord(self, _cmd, *_encoder, wire::EncoderPopDebugGroup{});
    [super popDebugGroup];
}

// -[AGXA10FamilyBlitContext sampleCountersInBuffer:atSampleIndex:withBarrier:]: the A10X samples no counters at
// blit boundaries, so the driver reports the call (MTLReportFailure) and records nothing.
- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)sampleBuffer
                 atSampleIndex:(NSUInteger)sampleIndex
                   withBarrier:(BOOL)barrier
{
    MTLReportFailure(0, "-[AGXA10FamilyBlitContext sampleCountersInBuffer:atSampleIndex:withBarrier:]", 404,
                     @"MTLBlitCommandEncoder:sampleCountersInBuffer:atSampleIndex:withBarrier not supported on this "
                     @"device");
}

@end

// Compute --------------------------------------------------------------------------------------------------------------

@implementation WebMetalComputeCommandEncoder

// -[IOGPUMetalCommandEncoder initWithCommandBuffer:] under the driver's compute context; the dispatch type is serial
// unless the creating method sets another.
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
    self = [super initWithCommandBuffer:commandBuffer];
    if (!self)
        return nil;
    _encoder = NewEncoderState((WebMetalCommandBuffer *)commandBuffer, nullptr);
    _dispatchType = MTLDispatchTypeSerial;
    if ([(WebMetalCommandBuffer *)commandBuffer getSupportedAndEnabledErrorOptions] &
        MTLCommandBufferErrorOptionEncoderExecutionStatus)
        _markerID = encoderTraceID(self);
    return self;
}

- (void)dealloc
{
    if (_encoder && _encoder->releases)
        _encoder->releases->dead(_encoder->handle);
}

- (NSUInteger)getType
{
    return 1;  // -[IOGPUMetalComputeCommandEncoder getType]
}

- (uint64_t)globalTraceObjectID
{
    return encoderTraceID(self);
}

// -[AGXA10FamilyComputeContext getDriverUniqueID]: the encoder's marker, which -initWithCommandBuffer:config: sets to
// the encoder's global trace ID when its command buffer's encoder execution status is enabled, and leaves 0 otherwise.
- (uint64_t)getDriverUniqueID
{
    return _markerID;
}

// -[AGXA10FamilyComputeContext dispatchType]: 1 for a concurrent encoder, 0 otherwise.
- (MTLDispatchType)dispatchType
{
    return _dispatchType == MTLDispatchTypeConcurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
}

- (void)endEncoding
{
    endEncoder(self, _cmd, *_encoder);
    [super endEncoding];
}

- (void)setLabel:(NSString *)label
{
    [super setLabel:label];
    setEncoderLabel(self, _cmd, *_encoder, label);
}

- (void)pushDebugGroup:(NSString *)string
{
    pushEncoderDebugGroup(self, _cmd, *_encoder, string);
    [super pushDebugGroup:string];
}

- (void)popDebugGroup
{
    EmitEncoderRecord(self, _cmd, *_encoder, wire::EncoderPopDebugGroup{});
    [super popDebugGroup];
}

// -[AGXA10FamilyComputeContext sampleCountersInBuffer:atSampleIndex:withBarrier:]: the A10X samples no counters at
// dispatch boundaries, so the driver reports the call (MTLReportFailure) and records nothing.
- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)sampleBuffer
                 atSampleIndex:(NSUInteger)sampleIndex
                   withBarrier:(BOOL)barrier
{
    MTLReportFailure(0, "-[AGXA10FamilyComputeContext sampleCountersInBuffer:atSampleIndex:withBarrier:]", 993,
                     @"MTLComputeCommandEncoder:sampleCountersInBuffer:atSampleIndex:withBarrier not supported on this "
                     @"device");
}

@end

// Render ---------------------------------------------------------------------------------------------------------------

@implementation WebMetalRenderCommandEncoder

// -[AGXA10FamilyRenderContext initWithCommandBuffer:descriptor:]: the framework encoder, then the pass state (sent as
// BeginRenderPass by the creating method).
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer descriptor:(MTLRenderPassDescriptor *)descriptor
{
    self = [super initWithCommandBuffer:commandBuffer];
    if (!self)
        return nil;
    _encoder = NewEncoderState((WebMetalCommandBuffer *)commandBuffer, nullptr);
    return self;
}

// A sub-encoder of the parallel render encoder `framebuffer`: its records go to a stream of its own, which the
// parallel encoder splices into its command buffer's stream when it ends.
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           descriptor:(MTLRenderPassDescriptor *)descriptor
                      subEncoderIndex:(NSUInteger)index
                          framebuffer:(void *)framebuffer
{
    WebMetalParallelRenderCommandEncoder *parent = (__bridge WebMetalParallelRenderCommandEncoder *)framebuffer;
    self = [super initWithCommandBuffer:commandBuffer];
    if (!self)
        return nil;
    auto stream = std::make_shared<Stream>(commandBuffer.retainedReferences);
    {
        std::lock_guard<std::mutex> guard(parent->_subStreamsLock);
        if (index != parent->_subStreams.size())
            WMLoudMethod(self, _cmd, "sub-encoders were not created in index order");
        parent->_subStreams.push_back(stream);
    }
    _encoder = NewEncoderState((WebMetalCommandBuffer *)commandBuffer, stream.get());
    _encoder->own = stream;
    _encoder->visibilityResultKey = KernelResourceKey(descriptor.visibilityResultBuffer);
    wire::BeginParallelSubEncoder record;
    record.parent = parent->_encoder->handle;
    record.encoder = _encoder->handle;
    EmitEncoderRecord(self, _cmd, *_encoder, record);
    return self;
}

- (void)dealloc
{
    if (_encoder && _encoder->releases)
        _encoder->releases->dead(_encoder->handle);
}

- (NSUInteger)getType
{
    return 0;  // -[IOGPUMetalRenderCommandEncoder getType]
}

- (uint64_t)globalTraceObjectID
{
    return encoderTraceID(self);
}

- (void)endEncoding
{
    endEncoder(self, _cmd, *_encoder);
    [super endEncoding];
}

- (void)setLabel:(NSString *)label
{
    [super setLabel:label];
    setEncoderLabel(self, _cmd, *_encoder, label);
}

- (void)pushDebugGroup:(NSString *)string
{
    pushEncoderDebugGroup(self, _cmd, *_encoder, string);
    [super pushDebugGroup:string];
}

- (void)popDebugGroup
{
    EmitEncoderRecord(self, _cmd, *_encoder, wire::EncoderPopDebugGroup{});
    [super popDebugGroup];
}

// -[AGXA10FamilyRenderContext sampleCountersInBuffer:atSampleIndex:withBarrier:]: the A10X samples no counters at
// draw boundaries, so the driver reports the call (MTLReportFailure) and records nothing.
- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)sampleBuffer
                 atSampleIndex:(NSUInteger)sampleIndex
                   withBarrier:(BOOL)barrier
{
    MTLReportFailure(0, "-[AGXA10FamilyRenderContext sampleCountersInBuffer:atSampleIndex:withBarrier:]", 1262,
                     @"MTLRenderCommandEncoder:sampleCountersInBuffer:atSampleIndex:withBarrier not supported on this "
                     @"device");
}

@end

// Parallel render ------------------------------------------------------------------------------------------------------

@implementation WebMetalParallelRenderCommandEncoder

- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                 renderPassDescriptor:(MTLRenderPassDescriptor *)descriptor
{
    self = [super initWithCommandBuffer:commandBuffer renderPassDescriptor:descriptor];
    if (!self)
        return nil;
    _encoder = NewEncoderState((WebMetalCommandBuffer *)commandBuffer, nullptr);
    _descriptor = [descriptor copy];
    return self;
}

- (void)dealloc
{
    if (_encoder && _encoder->releases)
        _encoder->releases->dead(_encoder->handle);
}

- (NSUInteger)getType
{
    return 3;  // -[IOGPUMetalParallelRenderCommandEncoder getType]
}

// -[AGXA10FamilyThreadedRenderPass renderCommandEncoder]: a command buffer from the framework, then a sub-encoder in it.
- (id<MTLRenderCommandEncoder>)renderCommandEncoder
{
    WebMetalCommandBuffer *buffer = (WebMetalCommandBuffer *)[self _renderCommandEncoderCommon];
    NSUInteger index;
    {
        std::lock_guard<std::mutex> guard(_subStreamsLock);
        index = _subStreams.size();
    }
    return [buffer subRenderCommandEncoderWithDescriptor:_descriptor
                                         subEncoderIndex:index
                                             framebuffer:(__bridge void *)self];
}

// -[AGXA10FamilyThreadedRenderPass endEncoding]: the sub-encoders' commands run in their creation order inside the
// parallel encoder's pass.
- (void)endEncoding
{
    {
        std::lock_guard<std::mutex> guard(_subStreamsLock);
        for (const auto &stream : _subStreams)
            _encoder->stream->splice(*stream, _encoder->handle);
        _subStreams.clear();
    }
    endEncoder(self, _cmd, *_encoder);
    [super endEncoding];
}

- (void)setLabel:(NSString *)label
{
    [super setLabel:label];
    setEncoderLabel(self, _cmd, *_encoder, label);
}

- (void)pushDebugGroup:(NSString *)string
{
    pushEncoderDebugGroup(self, _cmd, *_encoder, string);
    [super pushDebugGroup:string];
}

- (void)popDebugGroup
{
    EmitEncoderRecord(self, _cmd, *_encoder, wire::EncoderPopDebugGroup{});
    [super popDebugGroup];
}

@end
