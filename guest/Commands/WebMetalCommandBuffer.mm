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
    if (topLevel) {
        segments_.push_back(handle);
        afterWait_.push_back(waited_);
    }
}

void Stream::splice(Stream &other, uint64_t parent)
{
    writer_.out.insert(writer_.out.end(), other.writer_.out.begin(), other.writer_.out.end());
    for (Use use : other.uses_) {
        use.encoder = parent;
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

// Whether the A10X driver is traced to list the resource that `cmd` of encoder `self` names in the encoder's IOGPU
// resource list (IOGPUResourceListAddResource) with flags whose per-flag sets the kernel stamps on submission
// (AGXAllocationList2, kernelcache 0xfffffff005bf1914; IOGPUResource's stamp 0xfffffff005bc4e68). Render
// -useResource:usage: (AGXMetalA10 0x21c960e94), -useResources:count:usage: (0x21c960ef0), -useHeap: (0x21c961088) and
// -useHeaps:count: (0x21c961190) list the resource, or the heap's, with flags 5; the indexed draws list the index buffer
// with 1 (0x21c963d78, 0x21c964048, 0x21c964330, 0x21c9647b4) and the indirect draws the indirect buffer with 0x41
// (0x21c964570, 0x21c9647f0). Every other use is untraced.
Stream::UseKind useKind(id self, SEL cmd)
{
    static const std::vector<SEL> render = {
        @selector(useResource:usage:),
        @selector(useResources:count:usage:),
        @selector(useHeap:),
        @selector(useHeaps:count:),
        @selector(drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:),
        @selector(drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:),
        @selector(drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:
                      baseInstance:),
        @selector(drawPrimitives:indirectBuffer:indirectBufferOffset:),
        @selector(drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:),
    };
    if ([self isKindOfClass:[WebMetalRenderCommandEncoder class]] &&
        std::find(render.begin(), render.end(), cmd) != render.end())
        return Stream::UseKind::Stamped;
    return Stream::UseKind::Untraced;
}

uint64_t Reference(Stream &stream, id self, SEL cmd, id object)
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
    if (const uint64_t key = KernelResourceKey(object))
        stream.noteUse(key, useKind(self, cmd));
    return handle;
}

// The stamps committed command buffers have pending, by kernel resource ----------------------------------------------

namespace {

std::mutex gStampsLock;
std::condition_variable gStampsChanged;

struct StampEntry {
    std::shared_ptr<StampCompletion> commandBuffer;
    // Index into the command buffer's segments; SIZE_MAX for a use outside any encoder.
    size_t encoder;
    Stream::UseKind kind;
};
std::unordered_map<uint64_t, std::vector<StampEntry>> gStamps;

bool pending(const StampEntry &entry)
{
    const StampCompletion &c = *entry.commandBuffer;
    return !c.done && (entry.encoder == SIZE_MAX || entry.encoder >= c.completed);
}

// Called with gStampsLock held: what `key` has pending, dropping the entries that have completed.
PendingStamps pendingLocked(uint64_t key)
{
    PendingStamps out;
    auto found = gStamps.find(key);
    if (found == gStamps.end())
        return out;
    std::vector<StampEntry> &entries = found->second;
    entries.erase(std::remove_if(entries.begin(), entries.end(), [](const StampEntry &e) { return !pending(e); }),
                  entries.end());
    for (const StampEntry &e : entries) {
        switch (e.kind) {
        case Stream::UseKind::Stamped:
            out.stamped = true;
            break;
        case Stream::UseKind::Deferred:
            // Once the command buffer is scheduled, its waits and the earlier ones of its queue are passed.
            (e.commandBuffer->scheduled ? out.stamped : out.deferred) = true;
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

// An encoder an event wait precedes is processed by the kernel once the wait is passed (measured on Apple M4, macOS
// 27.0.1: a tracked buffer only such an encoder uses reports complete while the command buffer waits) or, if the A10X
// kernel turns the wait into a fence later work depends on (IOGPUCommandQueue::processKernelCommand, kernelcache
// 0xfffffff005bd8a90 to 0xfffffff005bd8bb4, merges the wait's event into the queue's), already at submission; which
// the A10X does is not established, so its traced uses are Deferred until it completes.
std::shared_ptr<StampCompletion> RegisterStamps(const Stream &stream, bool behindWait)
{
    auto completion = std::make_shared<StampCompletion>();
    completion->segments = stream.segments();
    completion->waits = stream.waited();
    std::lock_guard<std::mutex> guard(gStampsLock);
    for (const Stream::Use &use : stream.uses()) {
        size_t index = SIZE_MAX;
        if (use.encoder) {
            auto at = std::find(completion->segments.begin(), completion->segments.end(), use.encoder);
            if (at != completion->segments.end())
                index = static_cast<size_t>(at - completion->segments.begin());
        }
        Stream::UseKind kind = use.kind;
        if (kind == Stream::UseKind::Stamped &&
            (behindWait || (index != SIZE_MAX && stream.segmentsAfterWait()[index])))
            kind = Stream::UseKind::Deferred;
        gStamps[use.key].push_back(StampEntry{completion, index, kind});
    }
    return completion;
}

bool StampsWaiting(const std::shared_ptr<StampCompletion> &completion)
{
    std::lock_guard<std::mutex> guard(gStampsLock);
    return completion && completion->waits && !completion->done;
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

void CommandBufferStampsScheduled(const std::shared_ptr<StampCompletion> &completion)
{
    if (!completion)
        return;
    std::lock_guard<std::mutex> guard(gStampsLock);
    completion->scheduled = true;
}

void CommandBufferStampsCompleted(const std::shared_ptr<StampCompletion> &completion)
{
    if (!completion)
        return;
    {
        std::lock_guard<std::mutex> guard(gStampsLock);
        completion->done = true;
    }
    gStampsChanged.notify_all();
}

PendingStamps QueryStamps(uint64_t key)
{
    std::lock_guard<std::mutex> guard(gStampsLock);
    return pendingLocked(key);
}

PendingStamps WaitStamps(uint64_t key)
{
    std::unique_lock<std::mutex> guard(gStampsLock);
    for (;;) {
        const PendingStamps p = pendingLocked(key);
        if (!p.stamped)
            return p;
        gStampsChanged.wait(guard);
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
