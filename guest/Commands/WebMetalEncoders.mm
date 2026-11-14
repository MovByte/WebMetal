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
#import "../Resources/WebMetalCounterSampleBuffer.h"

#import <objc/runtime.h>

#include <algorithm>
#include <cassert>
#include <type_traits>
#include <vector>

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

// The pass descriptors' sample buffer attachment arrays (MTL{Render,Compute,Blit}PassSampleBufferAttachmentDescriptor
// ArrayInternal): the attachment at an index, or nil where none was made (unlike -objectAtIndexedSubscript:, which
// makes one).
@protocol WebMetalSampleBufferAttachments
- (id)_descriptorAtIndex:(NSUInteger)index;
@end

namespace webmetal {

namespace {

// Entries of the render, compute and blit pass descriptors' sampleBufferAttachments arrays: 21H461's
// -[MTL{Render,Compute,Blit}PassSampleBufferAttachmentDescriptorArrayInternal objectAtIndexedSubscript:] accept indices
// below 4 (cmp x2, #0x4 at 0x187b9b914, 0x187b5c834, 0x187bb2914).
constexpr NSUInteger kPassSampleBuffers = 4;

// A pass descriptor's sample buffer attachments, as the driver reads them when it makes the encoder (blit and compute:
// AGXMetalA10 0x21c9cef60 and 0x21c9cf168; render: -[AGXA10FamilyRenderContext initWithCommandBuffer:descriptor:
// subEncoderIndex:framebuffer:] from 0x21c95ed40): each of the four that exists (_descriptorAtIndex:) and names a buffer
// has its indices checked in order (validateSampleIndex, logging one past the buffer's samples), as 32-bit values, and
// the buffer's storage attached to the encoder when one of them is a sample (setCounterSampleBuffer).
template <class Attachment> void attachPassSamples(EncoderState &encoder, id attachments)
{
    for (NSUInteger i = 0; i < kPassSampleBuffers; i++) {
        Attachment *a = [(id<WebMetalSampleBufferAttachments>)attachments _descriptorAtIndex:i];
        if (!a || !a.sampleBuffer)
            continue;
        std::vector<uint32_t> indices;
        if constexpr (std::is_same_v<Attachment, MTLRenderPassSampleBufferAttachmentDescriptor>)
            indices = {uint32_t(a.startOfVertexSampleIndex), uint32_t(a.endOfVertexSampleIndex),
                       uint32_t(a.startOfFragmentSampleIndex), uint32_t(a.endOfFragmentSampleIndex)};
        else
            indices = {uint32_t(a.startOfEncoderSampleIndex), uint32_t(a.endOfEncoderSampleIndex)};
        bool sampled = false;
        for (uint32_t &index : indices)
            sampled |= ValidateSampleIndex(a.sampleBuffer, &index);
        if (sampled)
            AttachSampleStorage(&encoder.sampleStorage, a.sampleBuffer);
    }
}

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

void checkNoOpenEncoder(AGXA10FamilyCommandBuffer *buffer, const char *function)
{
    if (buffer->_currentEncoder)
        MTLReportFailure(0, function, __LINE__, @"A command encoder is already encoding to this command buffer");
}

// An encoder with counter sample storage attached has the GPU write timestamps into the storage at its boundaries, at
// the sample locations the AGX kernel extension resolves to the storage's 8-byte samples (0xfffffff005c31cb8).
const char kGPUTimestamps[] =
    "the encoder samples the GPU's timestamps into a counter sample buffer: the A10X writes them into the buffer's "
    "storage at the encoder's boundaries, in mach_absolute_time's time base (-sampleTimestamps:gpuTimestamp:, "
    "0x21c8db364), and the host writes no GPU timestamps: WebGPU's timestamp queries are an optional feature and count "
    "in a time base of their own";

void endEncoder(id self, SEL cmd, EncoderState &encoder)
{
    if (encoder.sampleStorage)
        WMLoudMethod(self, cmd, kGPUTimestamps);
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

// Each render segment lists the pass's attachment textures with flags 12, except memoryless ones, and their resolve
// textures with 12: AGX::Framebuffer's listing (0x21c919170, called with mask 0x3f from the segment's begin at
// 0x21c96e300) skips the attachments its exclusion mask fb+3100 holds, which are the memoryless ones (0x21c914848 to
// 0x21c914860, 0x21c915150, 0x21c915240), and lists colour resolve textures by fb+2440, which holds the attachments with
// both a texture and a resolve texture (0x21c91490c): the framebuffer configuration skips a colour attachment without a
// texture, its resolve texture with it (0x21c9147d0). It keeps a depth or stencil resolve texture whether or not the
// attachment has a texture (0x21c914d54, 0x21c914eec), and the framebuffer lists it from fb+3248 or fb+3256 (set at
// 0x21c918670 to 0x21c9186d0; listing bits 3 and 5). What the listing names after the depth and stencil textures, fb+8
// and fb+128, are buffers the framebuffer makes for those attachments (0x21c916d1c, 0x21c916f48), not resolve textures.
template <class Attachment> void attachmentCommon(id self, SEL cmd, Stream &stream, Attachment &out,
                                                  MTLRenderPassAttachmentDescriptor *in, bool depthOrStencil)
{
    const bool memoryless = in.texture && in.texture.storageMode == MTLStorageModeMemoryless;
    out.texture = Reference(stream, self, cmd, in.texture, memoryless ? Listing::Unlisted : Listing::Stamped);
    out.level = in.level;
    out.slice = in.slice;
    out.depthPlane = in.depthPlane;
    out.resolveTexture = Reference(stream, self, cmd, in.resolveTexture,
                                   in.texture || depthOrStencil ? Listing::Stamped : Listing::Unlisted);
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
        attachmentCommon(self, cmd, stream, out.colors[i], color, false);
        const MTLClearColor c = color.clearColor;
        out.colors[i].clearColor = {c.red, c.green, c.blue, c.alpha};
    }
    d.colorAttachments = {out.colors.data(), out.colors.size()};
    if (MTLRenderPassDepthAttachmentDescriptor *depth = in.depthAttachment) {
        d.depthAttachment.present = true;
        attachmentCommon(self, cmd, stream, d.depthAttachment.value, depth, true);
        d.depthAttachment.value.clearDepth = depth.clearDepth;
        d.depthAttachment.value.depthResolveFilter = depth.depthResolveFilter;
    }
    if (MTLRenderPassStencilAttachmentDescriptor *stencil = in.stencilAttachment) {
        d.stencilAttachment.present = true;
        attachmentCommon(self, cmd, stream, d.stencilAttachment.value, stencil, true);
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
    // The sample buffers stay in the guest (attachPassSamples).
    for (NSUInteger i = 0; i < kPassSampleBuffers; i++) {
        MTLRenderPassSampleBufferAttachmentDescriptor *attachment =
            [(id<WebMetalSampleBufferAttachments>)in.sampleBufferAttachments _descriptorAtIndex:i];
        wire::RenderPassSampleBufferAttachment s;
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
    "refuses (AGXAllocationList2, kernelcache 0xfffffff005bf1af8 to 0xfffffff005bf1b04). IOGPUCommandQueue::"
    "processCommandBuffer then sets the queue's error to 9 after the command buffer's earlier segments went to the GPU "
    "(0xfffffff005bd851c, 0xfffffff005bd8728), and the command buffer and every later one of the queue complete with "
    "MTLCommandBufferErrorInvalidResource (\"Invalid Resource (00000009:kIOGPUCommandBufferCallbackErrorInvalidResource)"
    "\", -[NSError initWithIOGPUError:], IOGPU 0x1fcb3661c); running part of a command buffer and failing its queue is "
    "not implemented";

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

@implementation AGXA10FamilyCommandBuffer (WebMetalEncoders)

- (id<MTLBlitCommandEncoder>)blitCommandEncoder
{
    checkNoOpenEncoder(self, "-[AGXA10FamilyCommandBuffer blitCommandEncoder]");
    AGXA10FamilyBlitContext *encoder = [[AGXA10FamilyBlitContext alloc] initWithCommandBuffer:self descriptor:nil];
    wire::BeginBlit record;
    record.encoder = encoder->_encoder->handle;
    EmitEncoderRecord(self, _cmd, *encoder->_encoder, record);
    return encoder;
}

- (id<MTLBlitCommandEncoder>)blitCommandEncoderWithDescriptor:(MTLBlitPassDescriptor *)descriptor
{
    checkNoOpenEncoder(self, "-[AGXA10FamilyCommandBuffer blitCommandEncoderWithDescriptor:]");
    AGXA10FamilyBlitContext *encoder = [[AGXA10FamilyBlitContext alloc] initWithCommandBuffer:self
                                                                                  descriptor:descriptor];
    EncoderState &state = *encoder->_encoder;
    wire::BeginBlitWithDescriptor record;
    record.encoder = state.handle;
    std::vector<wire::BlitPassSampleBufferAttachment> attachments;
    // The sample buffers stay in the guest (attachPassSamples, in the encoder's initializer).
    for (NSUInteger i = 0; i < kPassSampleBuffers; i++) {
        MTLBlitPassSampleBufferAttachmentDescriptor *in =
            [(id<WebMetalSampleBufferAttachments>)descriptor.sampleBufferAttachments _descriptorAtIndex:i];
        wire::BlitPassSampleBufferAttachment out;
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
    checkNoOpenEncoder(self, "-[AGXA10FamilyCommandBuffer computeCommandEncoder]");
    AGXA10FamilyComputeContext *encoder = [[AGXA10FamilyComputeContext alloc] initWithCommandBuffer:self];
    wire::BeginCompute record;
    record.encoder = encoder->_encoder->handle;
    EmitEncoderRecord(self, _cmd, *encoder->_encoder, record);
    return encoder;
}

- (id<MTLComputeCommandEncoder>)computeCommandEncoderWithDispatchType:(MTLDispatchType)dispatchType
{
    checkNoOpenEncoder(self, "-[AGXA10FamilyCommandBuffer computeCommandEncoderWithDispatchType:]");
    AGXA10FamilyComputeContext *encoder = [[AGXA10FamilyComputeContext alloc] initWithCommandBuffer:self];
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
    checkNoOpenEncoder(self, "-[AGXA10FamilyCommandBuffer computeCommandEncoderWithDispatchType:substreamCount:]");
    AGXA10FamilyComputeContext *encoder = [[AGXA10FamilyComputeContext alloc] initWithCommandBuffer:self];
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
    checkNoOpenEncoder(self, "-[AGXA10FamilyCommandBuffer computeCommandEncoderWithDescriptor:]");
    AGXA10FamilyComputeContext *encoder = [[AGXA10FamilyComputeContext alloc] initWithCommandBuffer:self];
    EncoderState &state = *encoder->_encoder;
    encoder->_dispatchType = descriptor.dispatchType;
    wire::BeginComputeWithDescriptor record;
    record.encoder = state.handle;
    id<WebMetalComputePassDescriptorState> extra = (id<WebMetalComputePassDescriptorState>)descriptor;
    record.computePassDescriptor.dispatchType = descriptor.dispatchType;
    record.computePassDescriptor.allowCommandEncoderCoalescing = [extra allowCommandEncoderCoalescing];
    record.computePassDescriptor.substreamCount = [extra substreamCount];
    std::vector<wire::ComputePassSampleBufferAttachment> attachments;
    // The sample buffers stay in the guest.
    attachPassSamples<MTLComputePassSampleBufferAttachmentDescriptor>(state, descriptor.sampleBufferAttachments);
    for (NSUInteger i = 0; i < kPassSampleBuffers; i++) {
        MTLComputePassSampleBufferAttachmentDescriptor *in =
            [(id<WebMetalSampleBufferAttachments>)descriptor.sampleBufferAttachments _descriptorAtIndex:i];
        wire::ComputePassSampleBufferAttachment out;
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
    checkNoOpenEncoder(self, "-[AGXA10FamilyCommandBuffer renderCommandEncoderWithDescriptor:]");
    AGXA10FamilyRenderContext *encoder = [[AGXA10FamilyRenderContext alloc] initWithCommandBuffer:self
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
    checkNoOpenEncoder(self, "-[AGXA10FamilyCommandBuffer parallelRenderCommandEncoderWithDescriptor:]");
    AGXA10FamilyThreadedRenderPass *encoder =
        [[AGXA10FamilyThreadedRenderPass alloc] initWithCommandBuffer:self renderPassDescriptor:descriptor];
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
    checkNoOpenEncoder(self, "-[AGXA10FamilyCommandBuffer subRenderCommandEncoderWithDescriptor:subEncoderIndex:"
                             "framebuffer:]");
    return [[AGXA10FamilyRenderContext alloc] initWithCommandBuffer:self
                                                        descriptor:descriptor
                                                   subEncoderIndex:index
                                                       framebuffer:framebuffer];
}

@end

// Blit -----------------------------------------------------------------------------------------------------------------

@implementation AGXA10FamilyBlitContext

// -[AGXA10FamilyBlitContext initWithCommandBuffer:descriptor:]: the framework encoder, then the driver's blit state.
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer descriptor:(MTLBlitPassDescriptor *)descriptor
{
    self = [super initWithCommandBuffer:commandBuffer];
    if (!self)
        return nil;
    _encoder = NewEncoderState((AGXA10FamilyCommandBuffer *)commandBuffer, nullptr);
    attachPassSamples<MTLBlitPassSampleBufferAttachmentDescriptor>(*_encoder, descriptor.sampleBufferAttachments);
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

// -[IOGPUMetalBlitCommandEncoder copyFromTexture:toTexture:] (IOGPU 0x1fcb317cc): MTLPickLargestMip (MTLTexture_Private.h)
// halves the larger texture's width, height and depth, without a floor of 1, until none exceeds the other's: the
// source's when one of its dimensions exceeds the destination's, else the destination's. The levels must then agree in
// every dimension (each at least 1), or the process asserts. The copy then takes the levels both have from those
// (mipmapLevelCount minus the level, the smaller) and the slices both have (arrayLength, six per cube for the cube
// and cube-array types), through the slices-and-levels form.
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture toTexture:(id<MTLTexture>)destinationTexture
{
    const NSUInteger sw = sourceTexture.width, sh = sourceTexture.height, sd = sourceTexture.depth;
    const NSUInteger dw = destinationTexture.width, dh = destinationTexture.height, dd = destinationTexture.depth;
    NSUInteger sourceMip = 0, destinationMip = 0;
    const bool sourceLarger = sw > dw || sh > dh || sd > dd;
    NSUInteger w = sourceLarger ? sw : dw, h = sourceLarger ? sh : dh, d = sourceLarger ? sd : dd;
    const NSUInteger ow = sourceLarger ? dw : sw, oh = sourceLarger ? dh : sh, od = sourceLarger ? dd : sd;
    NSUInteger mip = 0;
    while (w > ow || h > oh || d > od) {
        w >>= 1;
        h >>= 1;
        d >>= 1;
        mip++;
    }
    (sourceLarger ? sourceMip : destinationMip) = mip;
    auto at = [](NSUInteger v, NSUInteger level) { return std::max<NSUInteger>(v >> level, 1); };
    if (at(sw, sourceMip) != at(dw, destinationMip) || at(sh, sourceMip) != at(dh, destinationMip) ||
        at(sd, sourceMip) != at(dd, destinationMip) || (sourceMip && destinationMip))
        __assert_rtn("MTLPickLargestMip", "MTLTexture_Private.h", 474,
                     "MAX(srcSize->width >> *srcMip, 1UL) == MAX(dstSize->width >> *dstMip, 1UL) && "
                     "MAX(srcSize->height >> *srcMip, 1UL) == MAX(dstSize->height >> *dstMip, 1UL) && "
                     "MAX(srcSize->depth >> *srcMip, 1UL) == MAX(dstSize->depth >> *dstMip, 1UL) && "
                     "(*srcMip == 0 || *dstMip == 0) && \"Postcondition does not hold\"");
    const NSUInteger levels = std::min(sourceTexture.mipmapLevelCount - sourceMip,
                                       destinationTexture.mipmapLevelCount - destinationMip);
    auto slices = [](id<MTLTexture> texture) {
        const NSUInteger type = texture.textureType;
        return texture.arrayLength * (type - MTLTextureTypeCube < 2 ? 6 : 1);
    };
    [self copyFromTexture:sourceTexture
              sourceSlice:0
              sourceLevel:sourceMip
                toTexture:destinationTexture
         destinationSlice:0
         destinationLevel:destinationMip
               sliceCount:std::min(slices(sourceTexture), slices(destinationTexture))
               levelCount:levels];
}

// -[AGXA10FamilyBlitContext resolveCounters:inRange:destinationBuffer:destinationOffset:] (AGXMetalA10 0x21c8bafdc)
// does nothing for an empty range, and logs a range past the buffer's sampleCount (location + length in 64 bits; 0
// for nil) (agxa_blit_template.hpp line 3355) and encodes nothing. Otherwise it copies length * sampleSize bytes from
// location * sampleSize of the buffer's sample storage to the destination as a buffer copy (0x21c8bb174 to
// 0x21c8bb208), listing the storage with flags 0x103 while it does (ctx+0x730), so the kernel stamps the storage only
// for the parts the vertex copy moves, which are not traced here; the destination it stamps as any copy's. Before the
// copy it ends the encoder's segment when the encoder samples into the range (0x21c8bb100 to 0x21c8bb148), which only
// an encoder with attached storage does, and endEncoding refuses that.
- (void)resolveCounters:(id<MTLCounterSampleBuffer>)sampleBuffer
                inRange:(NSRange)range
      destinationBuffer:(id<MTLBuffer>)destinationBuffer
      destinationOffset:(NSUInteger)destinationOffset
{
    if (!range.length)
        return;
    if (range.location + range.length > [sampleBuffer sampleCount]) {
        WM_AGX_LOG("agxa_blit_template.hpp", 3355, "resolveCounters",
                   "Cannot resolve MTLCounter data: out of bounds access");
        return;
    }
    AGXBuffer *storage = [(AGXMTLCounterSampleBuffer *)sampleBuffer buffer];
    const NSUInteger size = [(AGXMTLCounterSampleBuffer *)sampleBuffer sampleSize];
    // The buffer copy begins no kick when it would copy the storage onto itself (0x21c8bb2ac).
    const bool inPlace = destinationBuffer == (id)storage && destinationOffset == size * range.location;
    wire::BlitCopyBufferToBuffer record;
    record.sourceBuffer =
        Reference(*_encoder->stream, self, _cmd, storage, inPlace ? Listing::Unlisted : Listing::Untraced);
    record.sourceOffset = size * range.location;
    record.destinationBuffer = Reference(*_encoder->stream, self, _cmd, destinationBuffer,
                                         inPlace ? Listing::Unlisted : Listing::Stamped);
    record.destinationOffset = destinationOffset;
    record.size = size * range.length;
    EmitEncoderRecord(self, _cmd, *_encoder, record);
}

@end

// Compute --------------------------------------------------------------------------------------------------------------

@implementation AGXA10FamilyComputeContext

// -[IOGPUMetalCommandEncoder initWithCommandBuffer:] under the driver's compute context; the dispatch type is serial
// unless the creating method sets another.
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
    self = [super initWithCommandBuffer:commandBuffer];
    if (!self)
        return nil;
    _encoder = NewEncoderState((AGXA10FamilyCommandBuffer *)commandBuffer, nullptr);
    _dispatchType = MTLDispatchTypeSerial;
    if ([(AGXA10FamilyCommandBuffer *)commandBuffer getSupportedAndEnabledErrorOptions] &
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

@implementation AGXA10FamilyRenderContext

// -[AGXA10FamilyRenderContext initWithCommandBuffer:descriptor:]: the framework encoder, then the pass state (sent as
// BeginRenderPass by the creating method).
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer descriptor:(MTLRenderPassDescriptor *)descriptor
{
    self = [super initWithCommandBuffer:commandBuffer];
    if (!self)
        return nil;
    _encoder = NewEncoderState((AGXA10FamilyCommandBuffer *)commandBuffer, nullptr);
    attachPassSamples<MTLRenderPassSampleBufferAttachmentDescriptor>(*_encoder, descriptor.sampleBufferAttachments);
    return self;
}

// A sub-encoder of the parallel render encoder `framebuffer`: its records go to a stream of its own, which the
// parallel encoder splices into its command buffer's stream when it ends.
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           descriptor:(MTLRenderPassDescriptor *)descriptor
                      subEncoderIndex:(NSUInteger)index
                          framebuffer:(void *)framebuffer
{
    AGXA10FamilyThreadedRenderPass *parent = (__bridge AGXA10FamilyThreadedRenderPass *)framebuffer;
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
    _encoder = NewEncoderState((AGXA10FamilyCommandBuffer *)commandBuffer, stream.get());
    _encoder->own = stream;
    _encoder->visibilityResultKey = KernelResourceKey(descriptor.visibilityResultBuffer);
    attachPassSamples<MTLRenderPassSampleBufferAttachmentDescriptor>(*_encoder, descriptor.sampleBufferAttachments);
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

@implementation AGXA10FamilyThreadedRenderPass

- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                 renderPassDescriptor:(MTLRenderPassDescriptor *)descriptor
{
    self = [super initWithCommandBuffer:commandBuffer renderPassDescriptor:descriptor];
    if (!self)
        return nil;
    _encoder = NewEncoderState((AGXA10FamilyCommandBuffer *)commandBuffer, nullptr);
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
    AGXA10FamilyCommandBuffer *buffer = (AGXA10FamilyCommandBuffer *)[self _renderCommandEncoderCommon];
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
