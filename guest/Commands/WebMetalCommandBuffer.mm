// The command buffer (AGXA10FamilyCommandBuffer on 21H461; WebMetalCommands.h).
//
// _MTLCommandBuffer runs the status machine, the scheduled and completed handler lists, the waits and the timing
// fields; its initializer blocks while the queue's maxCommandBufferCount buffers are outstanding, and -commit enqueues
// and commits through the queue. The driver layers add what is checked here: IOGPU's -validate assertions before
// commit, the open-encoder assertions of encoder creation (AGX) and of event encoding (IOGPU), and the records of the
// command buffer's own commands.

#import "WebMetalCommandState.h"
#import "../Plugin/WebMetalReport.h"

#import <objc/runtime.h>

#include <algorithm>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>

#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

namespace webmetal {

void Stream::use(id object, uint64_t handle)
{
    if (named_.insert(handle).second)
        resources_.push_back(handle);
    if (retain_) {
        if (!objects_)
            objects_ = [NSMutableArray array];
        [objects_ addObject:object];
    }
}

void Stream::beginEncoder(uint64_t handle, bool topLevel)
{
    current_ = handle;
    if (topLevel)
        segments_.push_back(handle);
}

void Stream::splice(Stream &other, uint64_t parent)
{
    writer_.out.insert(writer_.out.end(), other.writer_.out.begin(), other.writer_.out.end());
    // No wait can be encoded while the parallel encoder is open: its sub-encoders follow the waits recorded before it.
    for (Use use : other.uses_) {
        use.encoder = parent;
        use.waits = waits_;
        uses_.push_back(use);
    }
    for (uint64_t handle : other.resources_)
        if (named_.insert(handle).second)
            resources_.push_back(handle);
    encoders_.insert(encoders_.end(), other.encoders_.begin(), other.encoders_.end());
    if (other.objects_) {
        if (!objects_)
            objects_ = [NSMutableArray array];
        [objects_ addObjectsFromArray:other.objects_];
    }
}

void PendingRelease::dead(uint64_t handle)
{
    Connection *connection = nullptr;
    {
        std::lock_guard<std::mutex> guard(lock_);
        if (bound_.erase(handle))
            connection = connection_;
        else
            dead_.insert(handle);
    }
    if (connection)
        ReleaseHandle(connection, handle);
}

void PendingRelease::bound(const std::vector<uint64_t> &handles, Connection *connection)
{
    std::vector<uint64_t> now;
    {
        std::lock_guard<std::mutex> guard(lock_);
        connection_ = connection;
        for (uint64_t handle : handles) {
            if (dead_.erase(handle))
                now.push_back(handle);
            else
                bound_.insert(handle);
        }
    }
    for (uint64_t handle : now)
        ReleaseHandle(connection, handle);
}

// How the A10X driver lists the resource that `cmd` of encoder `self` names in the IOGPU resource list of the encoder's
// segment (IOGPUResourceListAddResource, which ORs the flags of a resource's listings), and so whether the segment's
// kick stamps the kernel resource's event 0. The kernel's parse of the list (AGXAllocationList2, kernelcache
// 0xfffffff005bf1914) drops the read bit of a read and write pair (0x3, 0xc, 0x30) and puts the resource in a set per
// remaining bit; the render kick stamps sets 0, 1, 9, 10 and 11 with its vertex work and 2 and 3 with its fragment work
// (0xfffffff005bffce0 to 0xfffffff005c00518), the compute kick sets 0, 1, 10 and 11 (0xfffffff005bfc728 to
// 0xfffffff005bfca58) and the blit kick sets 2 and 3 (0xfffffff005bfda5c, 0xfffffff005bfdb5c); flag 0x80 and bits 4 to
// 8 are stamped by none.
//
// Render (AGXA10FamilyRenderContext): -useResource:usage: (0x21c960e94), -useResources:count:usage: (0x21c960ef0),
// -useHeap: (0x21c961088) and -useHeaps:count: (0x21c961190) list with 5; the forms with stages (0x21c960d60,
// 0x21c960e24, 0x21c960f30, 0x21c961058) with flags from the usage and the stages that are not 0 when the usage has a
// read, write or sample bit and the stages a vertex, fragment or tile bit (NoteRecordListing refuses the rest); the
// indexed draws list the index buffer with 1 (0x21c963d78, 0x21c964048, 0x21c964330, 0x21c9647b4) and the indirect
// draws the indirect buffer with 0x41 (0x21c964570, 0x21c9647f0). Compute (AGXA10FamilyComputeContext):
// -useResource:usage: (0x21c90c0a4) and -useResources:count:usage: (0x21c90c120) list with (usage & 2) | ((usage & 7)
// != 0), -useHeap: (0x21c90c16c) and -useHeaps:count: (0x21c90c274) with 1,
// -setStageInRegionWithIndirectBuffer:indirectBufferOffset: with 1 (0x21c90b7c8). Blit (AGXA10FamilyBlitContext):
// -invalidateCompressedTexture:, -optimizeContentsForGPUAccess:, -optimizeContentsForCPUAccess: and their slice and
// level forms return at once (0x21c8b9954 to 0x21c8b9968). Every other use is untraced: the binds the driver lists at
// draw and dispatch time by masks of the bound pipeline's compiled program (render 0x21c96ad3c, compute 0x21c90f2c4),
// which come from its compiler (packages G4 and G5); the indirect dispatch's buffer, listed only while the threadgroup
// fits the pipeline (0x21c90b8a0 to 0x21c90b8ac); and the blit operations, which the driver runs as blit or render
// kicks over the arguments by paths not read yet.
Listing listingOf(id self, SEL cmd)
{
    static const std::vector<SEL> render = {
        @selector(useResource:usage:),
        @selector(useResources:count:usage:),
        @selector(useHeap:),
        @selector(useHeaps:count:),
        @selector(useResource:usage:stages:),
        @selector(useResources:count:usage:stages:),
        @selector(useHeap:stages:),
        @selector(useHeaps:count:stages:),
        @selector(drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:),
        @selector(drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:),
        @selector(drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:
                      baseInstance:),
        @selector(drawPrimitives:indirectBuffer:indirectBufferOffset:),
        @selector(drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:),
    };
    static const std::vector<SEL> compute = {
        @selector(useResource:usage:),
        @selector(useResources:count:usage:),
        @selector(useHeap:),
        @selector(useHeaps:count:),
        @selector(setStageInRegionWithIndirectBuffer:indirectBufferOffset:),
    };
    static const std::vector<SEL> blitUnlisted = {
        @selector(invalidateCompressedTexture:),
        @selector(invalidateCompressedTexture:slice:level:),
        @selector(optimizeContentsForGPUAccess:),
        @selector(optimizeContentsForGPUAccess:slice:level:),
        @selector(optimizeContentsForCPUAccess:),
        @selector(optimizeContentsForCPUAccess:slice:level:),
    };
    auto in = [cmd](const std::vector<SEL> &list) { return std::find(list.begin(), list.end(), cmd) != list.end(); };
    if ([self isKindOfClass:[WebMetalRenderCommandEncoder class]] && in(render))
        return Listing::Stamped;
    if ([self isKindOfClass:[WebMetalComputeCommandEncoder class]] && in(compute))
        return Listing::Stamped;
    if ([self isKindOfClass:[WebMetalBlitCommandEncoder class]] && in(blitUnlisted))
        return Listing::Unlisted;
    return Listing::Untraced;
}

uint64_t Reference(Stream &stream, id self, SEL cmd, id object, Listing listing)
{
    if (!object)
        return 0;
    const uint64_t handle = HandleOf(object);
    if (!handle) {
        const std::string detail = std::string("an argument of class ") + object_getClassName(object) +
                                   " has no WebMetal handle, so the host cannot be told about it";
        WMLoudMethod(self, cmd, detail.c_str());
    }
    stream.use(object, handle);
    const uint64_t key = KernelResourceKey(object);
    if (key && listing != Listing::Unlisted)
        stream.noteUse(key, listing == Listing::Stamped ? Stream::UseKind::Stamped : Stream::UseKind::Untraced);
    return handle;
}

uint64_t Reference(Stream &stream, id self, SEL cmd, id object)
{
    return Reference(stream, self, cmd, object, listingOf(self, cmd));
}

// The stamps committed command buffers have pending, by kernel resource ----------------------------------------------

namespace {

std::mutex gStampsLock;
std::condition_variable gStampsChanged;
uint64_t gNextStamp = 1;

struct StampEntry {
    uint64_t id;
    std::shared_ptr<StampCompletion> commandBuffer;
    // Index into the command buffer's segments; SIZE_MAX for a use outside any encoder.
    size_t encoder;
    // The command buffer's event waits before the use.
    uint32_t waits;
    Stream::UseKind kind;
};
std::unordered_map<uint64_t, std::vector<StampEntry>> gStamps;

bool pending(const StampEntry &entry)
{
    const StampCompletion &c = *entry.commandBuffer;
    return !c.done && (entry.encoder == SIZE_MAX || entry.encoder >= c.completed);
}

// Whether the kernel has processed the encoder of the use, and so stamped what it lists. The A10X kernel processes a
// queue's kernel commands in order on submission and stays at an event wait until it is passed: a wait on a shared
// event sleeps in IOGPUCommandQueue::processKernelCommand until the event reaches the value (kernelcache
// 0xfffffff005bd8d54 to 0xfffffff005bd8e18, IOSurface's wait 0xfffffff005b97400), one on an MTLEvent until a signal of
// the value has been processed (0xfffffff005bdce38); the kick that stamps an encoder's resources is a later kernel
// command of the same loop (AGX kernel command 0x10000, 0xfffffff005bf7a60). The host passes waits the same way and
// reports each (EventWaitPassed), and processes a queue's command buffers in order.
bool processed(const StampEntry &entry)
{
    const StampCompletion &c = *entry.commandBuffer;
    if (c.passed < entry.waits)
        return false;
    for (const auto &earlier : c.behind)
        if (!earlier->scheduled && !earlier->done)
            return false;
    return true;
}

// Called with gStampsLock held: what `key` has pending, dropping the entries that have completed. `stamps` receives
// the ids of the pending traced stamps.
PendingStamps pendingLocked(uint64_t key, std::vector<uint64_t> *stamps = nullptr)
{
    PendingStamps out;
    auto found = gStamps.find(key);
    if (found == gStamps.end())
        return out;
    std::vector<StampEntry> &entries = found->second;
    entries.erase(std::remove_if(entries.begin(), entries.end(), [](const StampEntry &e) { return !pending(e); }),
                  entries.end());
    for (const StampEntry &e : entries) {
        if (!processed(e))
            continue;
        switch (e.kind) {
        case Stream::UseKind::Stamped:
            out.stamped = true;
            if (stamps)
                stamps->push_back(e.id);
            break;
        case Stream::UseKind::Untraced:
            out.untraced = true;
            break;
        }
    }
    if (entries.empty())
        gStamps.erase(found);
    return out;
}

}  // namespace

std::shared_ptr<StampCompletion> RegisterStamps(const Stream &stream,
                                                std::vector<std::shared_ptr<StampCompletion>> behind)
{
    auto completion = std::make_shared<StampCompletion>();
    completion->segments = stream.segments();
    completion->waits = stream.waits();
    completion->behind = std::move(behind);
    std::lock_guard<std::mutex> guard(gStampsLock);
    std::unordered_set<uint64_t> keys;
    for (const Stream::Use &use : stream.uses()) {
        size_t index = SIZE_MAX;
        if (use.encoder) {
            auto at = std::find(completion->segments.begin(), completion->segments.end(), use.encoder);
            if (at != completion->segments.end())
                index = static_cast<size_t>(at - completion->segments.begin());
        }
        gStamps[use.key].push_back(StampEntry{gNextStamp++, completion, index, use.waits, use.kind});
        if (keys.insert(use.key).second)
            completion->keys.push_back(use.key);
    }
    return completion;
}

bool StampsHoldQueue(const std::shared_ptr<StampCompletion> &completion)
{
    std::lock_guard<std::mutex> guard(gStampsLock);
    return completion && completion->waits && !completion->scheduled && !completion->done;
}

void EncoderStampsCompleted(const std::shared_ptr<StampCompletion> &completion, uint64_t encoder)
{
    if (!completion)
        return;
    auto at = std::find(completion->segments.begin(), completion->segments.end(), encoder);
    if (at == completion->segments.end())
        return;
    {
        std::lock_guard<std::mutex> guard(gStampsLock);
        completion->completed = std::max(completion->completed, static_cast<size_t>(at - completion->segments.begin()) + 1);
    }
    gStampsChanged.notify_all();
}

void CommandBufferStampsWaitPassed(const std::shared_ptr<StampCompletion> &completion, uint32_t waits)
{
    if (!completion)
        return;
    {
        std::lock_guard<std::mutex> guard(gStampsLock);
        completion->passed = std::max(completion->passed, waits);
    }
    gStampsChanged.notify_all();
}

void CommandBufferStampsScheduled(const std::shared_ptr<StampCompletion> &completion)
{
    if (!completion)
        return;
    {
        std::lock_guard<std::mutex> guard(gStampsLock);
        completion->scheduled = true;
        // Its own processing came after that of the earlier command buffers of its queue.
        completion->behind.clear();
    }
    gStampsChanged.notify_all();
}

void CommandBufferStampsCompleted(const std::shared_ptr<StampCompletion> &completion)
{
    if (!completion)
        return;
    {
        std::lock_guard<std::mutex> guard(gStampsLock);
        completion->done = true;
        completion->behind.clear();
        for (uint64_t key : completion->keys) {
            auto found = gStamps.find(key);
            if (found == gStamps.end())
                continue;
            std::vector<StampEntry> &entries = found->second;
            entries.erase(std::remove_if(entries.begin(), entries.end(),
                                         [&](const StampEntry &e) { return e.commandBuffer == completion; }),
                          entries.end());
            if (entries.empty())
                gStamps.erase(found);
        }
    }
    gStampsChanged.notify_all();
}

PendingStamps QueryStamps(uint64_t key)
{
    std::lock_guard<std::mutex> guard(gStampsLock);
    return pendingLocked(key);
}

// IOGPUResourceFinishEvent enters the kernel while the test fails; the kernel's finish (IOGPUEventMachine, kernelcache
// 0xfffffff005bbddc0) takes the event's channel mask once and waits for each channel's stamp in turn, reading that
// channel's stamp when it reaches it (waitForStamp 0xfffffff005bbd830). It returns once the stamps it took have passed;
// whether it waits for a stamp the kernel adds meanwhile depends on the GPU channel of that work and the order of the
// channels, which are not modeled, so that case is `racing`.
PendingStamps WaitStamps(uint64_t key)
{
    std::unique_lock<std::mutex> guard(gStampsLock);
    std::vector<uint64_t> started;
    PendingStamps first = pendingLocked(key, &started);
    if (!first.stamped)
        return first;
    for (;;) {
        gStampsChanged.wait(guard);
        std::vector<uint64_t> now;
        PendingStamps p = pendingLocked(key, &now);
        bool waiting = false, added = false;
        for (uint64_t id : now)
            (std::find(started.begin(), started.end(), id) != started.end() ? waiting : added) = true;
        if (!waiting) {
            p.stamped = false;
            return p;
        }
        if (added) {
            p.racing = true;
            return p;
        }
    }
}

template <class Record> void EmitCommandBufferRecord(WebMetalCommandBuffer *buffer, SEL cmd, const Record &record)
{
    if (!wire::encode(buffer->_state->stream.writer(), record))
        WMLoudMethod(buffer, cmd, "the arguments break a rule of the wire format");
}

namespace {

// Process-wide trace object IDs. IOGPUDeviceGetNextGlobalTraceID hands out IDs from ranges the kernel grants and, when
// the kernel grants none, counts from 1; the plugin has no kernel range and counts from 1.
std::atomic<uint64_t> gTraceIDs{1};

void setFrameworkTraceID(id object, const char *cls)
{
    Ivar ivar = class_getInstanceVariable(objc_getClass(cls), "_globalTraceObjectID");
    *(uint64_t *)((uint8_t *)(__bridge void *)object + ivar_getOffset(ivar)) = gTraceIDs.fetch_add(1);
}

// -[IOGPUMetalCommandBuffer validate], which -[AGXA10FamilyCommandBuffer commit] calls through super before anything
// else. The checks report through Metal.framework's failure channel, as the driver's do.
void validateForCommit(WebMetalCommandBuffer *buffer)
{
    if (buffer.status >= MTLCommandBufferStatusCommitted)
        MTLReportFailure(0, "-[WebMetalCommandBuffer commit]", __LINE__, @"commit an already committed command buffer");
    if (buffer->_currentEncoder)
        MTLReportFailure(0, "-[WebMetalCommandBuffer commit]", __LINE__,
                         @"commit command buffer with uncommitted encoder");
}

// The event a command-buffer command names. A nil event is LOUD: IOGPU sends the encode message to the event, so a nil
// event leaves the kernel command's arguments unset, which the host has no form for.
uint64_t eventReference(WebMetalCommandBuffer *buffer, SEL cmd, id<MTLEvent> event)
{
    if (!event)
        WMLoudMethod(buffer, cmd, "a nil event: 21H461's driver encodes a kernel event command without event arguments");
    uint64_t handle = HandleOf(event);
    if (!handle) {
        std::string problem;
        handle = SharedEventHandle(buffer->_state->connection, event, &problem);
        if (!handle)
            WMLoudMethod(buffer, cmd, problem.c_str());
    }
    buffer->_state->stream.use(event, handle);
    return handle;
}

}  // namespace

std::shared_ptr<EncoderState> NewEncoderState(WebMetalCommandBuffer *buffer, Stream *subStream)
{
    auto state = std::make_shared<EncoderState>();
    state->commandBuffer = buffer->_state;
    state->stream = subStream ? subStream : &buffer->_state->stream;
    state->releases = buffer->_state->releases;
    state->handle = buffer->_state->connection->NewHandle();
    state->stream->encoders().push_back(state->handle);
    state->stream->beginEncoder(state->handle, subStream == nullptr);
    return state;
}

}  // namespace webmetal

using namespace webmetal;

@implementation WebMetalCommandBuffer

- (instancetype)initWithQueue:(id<MTLCommandQueue>)queue retainedReferences:(BOOL)retainedReferences
{
    return [self initWithQueue:queue retainedReferences:retainedReferences synchronousDebugMode:NO];
}

// IOGPUMetalCommandBuffer requires an IOGPU queue, initializes the framework's command buffer (which waits for a free
// slot of the queue), counts the buffer on the device, takes command-buffer storage and a global trace ID.
- (instancetype)initWithQueue:(id<MTLCommandQueue>)queue
           retainedReferences:(BOOL)retainedReferences
          synchronousDebugMode:(BOOL)synchronousDebugMode
{
    if (![(id)queue isKindOfClass:[WebMetalCommandQueue class]])
        MTLReleaseAssertionFailure("-[WebMetalCommandBuffer initWithQueue:retainedReferences:synchronousDebugMode:]",
                                   __LINE__, "[queue isKindOfClass:[WebMetalCommandQueue class]]", nullptr);
    self = [super initWithQueue:queue retainedReferences:retainedReferences synchronousDebugMode:synchronousDebugMode];
    if (!self)
        return nil;
    WebMetalCommandQueue *owner = (WebMetalCommandQueue *)queue;
    _state = std::make_shared<CommandBufferState>(owner->_connection, owner->_connection->NewHandle(),
                                                  retainedReferences);
    setFrameworkTraceID(self, "_MTLCommandBuffer");
    return self;
}

- (void)dealloc
{
    if (_state && _state->bound)
        ReleaseHandle(_state->connection, _state->handle);
}

- (id<MTLDevice>)device
{
    return [(WebMetalCommandQueue *)self.commandQueue device];
}

- (void)validate
{
    // -[AGXA10FamilyCommandBuffer validate] is empty; the checks are IOGPU's, which AGX's -commit calls directly.
}

- (void)commit
{
    validateForCommit(self);
    [super commit];
}

// IOGPU requires the buffer not to be committed (a release assertion), then records the encoder with the framework.
- (void)setCurrentCommandEncoder:(id<MTLCommandEncoder>)encoder
{
    if (self.status >= MTLCommandBufferStatusCommitted)
        MTLReleaseAssertionFailure("-[WebMetalCommandBuffer setCurrentCommandEncoder:]", __LINE__,
                                   "_status < MTLCommandBufferStatusCommitted", nullptr);
    [super setCurrentCommandEncoder:encoder];
    _currentEncoder = encoder;
}

// IOGPU completes the framework's buffer, then returns its command-buffer storage.
- (void)didCompleteWithStartTime:(uint64_t)startTime endTime:(uint64_t)endTime error:(NSError *)error
{
    [super didCompleteWithStartTime:startTime endTime:endTime error:error];
    _state->stream.releaseObjects();
}

- (void)setLabel:(NSString *)label
{
    // The label reaches the host with Commit.
    [super setLabel:label];
}

// -[AGXA10FamilyCommandBuffer getSupportedAndEnabledErrorOptions]: the buffer's errorOptions within the device's
// supported options (-getSupportedCommandBufferErrorOptions).
- (NSUInteger)getSupportedAndEnabledErrorOptions
{
    return self.errorOptions & [(id<WebMetalDeviceErrorOptions>)self.device getSupportedCommandBufferErrorOptions];
}

- (NSUInteger)protectionOptions
{
    return _protectionOptions;
}

// -[IOGPUMetalCommandBuffer setProtectionOptions:] (0x1fcb2e0d0): report an open encoder; a value equal to the current
// one changes nothing; otherwise store it and encode, at this point of the command buffer, the kernel command that
// sets it (kIOGPUKernelCommandSetProtectionOptions).
- (void)setProtectionOptions:(NSUInteger)options
{
    if (_currentEncoder)
        MTLReportFailure(0, "-[WebMetalCommandBuffer setProtectionOptions:]", __LINE__,
                         @"setProtectionOptions: with uncommitted encoder");
    if (options == _protectionOptions)
        return;
    _protectionOptions = options;
    wire::CommandBufferSetProtectionOptions record;
    record.options = options;
    EmitCommandBufferRecord(self, _cmd, record);
}

- (void)pushDebugGroup:(NSString *)string
{
    [super pushDebugGroup:string];
    wire::CommandBufferPushDebugGroup record;
    if (!wire::strFromNSString(string, record.string))
        WMLoudMethod(self, _cmd, "the string has no UTF-8 form");
    EmitCommandBufferRecord(self, _cmd, record);
}

- (void)popDebugGroup
{
    [super popDebugGroup];
    EmitCommandBufferRecord(self, _cmd, wire::CommandBufferPopDebugGroup{});
}

// -[IOGPUMetalCommandBuffer encodeSignalEvent:value:] (AGX's calls it): report an open encoder, record the signal,
// and retain the event when the buffer retains its references.
- (void)encodeSignalEvent:(id<MTLEvent>)event value:(uint64_t)value
{
    if (_currentEncoder)
        MTLReportFailure(0, "-[WebMetalCommandBuffer encodeSignalEvent:value:]", __LINE__,
                         @"encodeSignalEvent:value: with uncommitted encoder");
    wire::EncodeSignalEvent record;
    record.event = eventReference(self, _cmd, event);
    record.value = value;
    EmitCommandBufferRecord(self, _cmd, record);
}

- (void)encodeWaitForEvent:(id<MTLEvent>)event value:(uint64_t)value
{
    if (_currentEncoder)
        MTLReportFailure(0, "-[WebMetalCommandBuffer encodeWaitForEvent:value:]", __LINE__,
                         @"encodeWaitForEvent:value: with uncommitted encoder");
    wire::EncodeWaitForEvent record;
    record.event = eventReference(self, _cmd, event);
    record.value = value;
    EmitCommandBufferRecord(self, _cmd, record);
    _state->stream.noteWait();
}

- (void)encodeSignalEvent:(id<MTLEvent>)event value:(uint64_t)value agentMask:(uint64_t)agentMask
{
    if (_currentEncoder)
        MTLReportFailure(0, "-[WebMetalCommandBuffer encodeSignalEvent:value:agentMask:]", __LINE__,
                         @"encodeSignalEvent:value: with uncommitted encoder");
    wire::EncodeSignalEventAgentMask record;
    record.event = eventReference(self, _cmd, event);
    record.value = value;
    record.agentMask = agentMask;
    EmitCommandBufferRecord(self, _cmd, record);
}

- (void)encodeWaitForEvent:(id<MTLEvent>)event value:(uint64_t)value timeout:(uint32_t)timeout
{
    if (_currentEncoder)
        MTLReportFailure(0, "-[WebMetalCommandBuffer encodeWaitForEvent:value:timeout:]", __LINE__,
                         @"encodeWaitForEvent:value: with uncommitted encoder");
    wire::EncodeWaitForEventTimeout record;
    record.event = eventReference(self, _cmd, event);
    record.value = value;
    // The event encodes the kernel command's arguments: -[_MTLSharedEvent(IOGPUFamilySupport)
    // _encodeIOGPUKernelWaitEventCommandArgs:value:timeout:] (0x1fcb36920) no more than 0xffff seconds,
    // -[IOGPUMTLEvent _encodeIOGPUKernelWaitEventCommandArgs:value:timeout:] (0x1fcb3bbe8) the timeout as given.
    static Class sharedEventClass = objc_getClass("_MTLSharedEvent");
    record.timeout = [(id)event isKindOfClass:sharedEventClass] ? std::min<uint32_t>(timeout, 0xffff) : timeout;
    EmitCommandBufferRecord(self, _cmd, record);
    _state->stream.noteWait();
}

- (void)encodeSignalEventScheduled:(id<MTLEvent>)event value:(uint64_t)value
{
    if (_currentEncoder)
        MTLReportFailure(0, "-[WebMetalCommandBuffer encodeSignalEventScheduled:value:]", __LINE__,
                         @"encodeSignalEventScheduled:value: with uncommitted encoder");
    wire::EncodeSignalEventScheduled record;
    record.event = eventReference(self, _cmd, event);
    record.value = value;
    EmitCommandBufferRecord(self, _cmd, record);
}

@end
