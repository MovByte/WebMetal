// The command queue (AGXA10FamilyCommandQueue on 21H461; WebMetalCommands.h).
//
// _MTLCommandQueue keeps the pending and submitted command buffers, blocks -commandBuffer at maxCommandBufferCount, and
// calls -submitCommandBuffers:count: on its command dispatch queue in enqueue order once each buffer is committed. The
// queue sends each one to the host with Commit and turns the host's events into the framework's completion calls on
// the queue's completion dispatch queue, where IOGPU's notification queue delivers them on the target
// (-[IOGPUMetalCommandQueue initWithDevice:descriptor:] targets it at _MTLCommandQueue._completionQueueDispatch).

#import "WebMetalCommandState.h"
#import "../Plugin/WebMetalReport.h"
#import "../SPI/WebMetalSPI.h"

#import <objc/runtime.h>

#include <unordered_map>

// The class adopts the driver's protocols and implements only what the 21H461 driver layers implement; the rest is
// inherited from _MTLCommandQueue or a LOUD stub.
#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

// The state of MTLCommandQueueDescriptorInternal and _MTLCommandBufferDescriptor the driver reads (21H461's getters;
// the wire schema's CommandQueueDescriptor lists the same ones).
@protocol WebMetalCommandQueueDescriptorState
- (BOOL)commitSynchronously;
- (NSUInteger)devicePartition;
- (BOOL)disableAsyncCompletionDispatch;
- (BOOL)disableCrossQueueHazardTracking;
- (BOOL)disableIOFencing;
- (BOOL)enableLowLatencySignalSharedEvent;
- (BOOL)enableLowLatencyWaitSharedEvent;
- (BOOL)isOpenGLQueue;
- (BOOL)lockParameterBufferSizeToMax;
- (NSUInteger)maxCommandBufferCount;
- (NSUInteger)qosLevel;
@end

@protocol WebMetalCommandBufferDescriptorState
- (id)deadlineProfile;
- (BOOL)captureProgramAddressTable;
- (BOOL)cloneIntersectionFunctionTablesPerDispatch;
- (BOOL)disableFineGrainedComputePreemption;
@end

namespace {

// IOGPUMetalCommandQueue's command buffers between Commit and completion, by handle, for the event handler.
struct QueueEvents {
    std::mutex lock;
    std::unordered_map<uint64_t, WebMetalCommandBuffer *> inFlight;  // retained through `retained`
    NSMutableArray *retained = [NSMutableArray array];
};

// The queue's events by queue handle. Entries live from queue creation to queue deallocation.
std::mutex gQueuesLock;
std::unordered_map<uint64_t, std::shared_ptr<QueueEvents>> gQueues;

std::shared_ptr<QueueEvents> eventsOf(uint64_t handle)
{
    std::lock_guard<std::mutex> guard(gQueuesLock);
    auto found = gQueues.find(handle);
    return found == gQueues.end() ? nullptr : found->second;
}

// _MTLCommandQueue._completionQueueDispatch: the queue completion notifications run on (IOGPU reads and writes the
// instance variable directly; the guard has checked its name, type and offset).
Ivar completionDispatchIvar()
{
    static Ivar ivar = class_getInstanceVariable(objc_getClass("_MTLCommandQueue"), "_completionQueueDispatch");
    return ivar;
}

dispatch_queue_t completionDispatch(id queue)
{
    return (dispatch_queue_t)object_getIvar(queue, completionDispatchIvar());
}

// Runs the events the host has for one queue (WebMetalCommandQueue's doorbell, and -dispatchAvailableCompletionNotifications).
void dispatchEvents(WebMetalCommandQueue *queue, webmetal::Connection *connection, uint64_t handle)
{
    std::vector<webmetal::wire::Event> events;
    std::string problem;
    if (!webmetal::FetchEvents(connection, handle, events, &problem)) {
        WMReport("-[WebMetalCommandQueue dispatchAvailableCompletionNotifications]", problem.c_str());
        return;
    }
    std::shared_ptr<QueueEvents> state = eventsOf(handle);
    if (!state)
        return;
    for (const webmetal::wire::Event &event : events) {
        if (auto *scheduled = std::get_if<webmetal::wire::CommandBufferScheduled>(&event)) {
            WebMetalCommandBuffer *buffer = nil;
            {
                std::lock_guard<std::mutex> guard(state->lock);
                auto found = state->inFlight.find(scheduled->commandBuffer);
                buffer = found == state->inFlight.end() ? nil : found->second;
            }
            if (!buffer) {
                WMReport("CommandBufferScheduled", "the host scheduled a command buffer this queue has not committed");
                continue;
            }
            // The kernel's scheduled notification carries the start and the end of its submission and the status of
            // processing the command buffer, and IOGPU passes the times and the error it makes of the status. The
            // failure is reported once, with the completion that carries it again.
            NSError *error = scheduled->failure.status == 0
                                 ? nil
                                 : webmetal::CommandBufferError(scheduled->failure, "-[MTLCommandBuffer commit]", false);
            [buffer didScheduleWithStartTime:webmetal::MachTicksFromNanoseconds(scheduled->kernelStartTime)
                                     endTime:webmetal::MachTicksFromNanoseconds(scheduled->kernelEndTime)
                                       error:error];
        } else if (auto *completed = std::get_if<webmetal::wire::CommandBufferCompleted>(&event)) {
            WebMetalCommandBuffer *buffer = nil;
            {
                std::lock_guard<std::mutex> guard(state->lock);
                auto found = state->inFlight.find(completed->commandBuffer);
                if (found != state->inFlight.end()) {
                    buffer = found->second;
                    state->inFlight.erase(found);
                }
            }
            if (!buffer) {
                WMReport("CommandBufferCompleted", "the host completed a command buffer this queue has not committed");
                continue;
            }
            NSError *error = nil;
            if (completed->failure.status != 0)
                error = webmetal::CommandBufferError(completed->failure, "-[MTLCommandBuffer commit]");
            [queue commandBufferDidComplete:buffer
                                  startTime:webmetal::MachTicksFromNanoseconds(completed->gpuStartTime)
                             completionTime:webmetal::MachTicksFromNanoseconds(completed->gpuEndTime)
                                      error:error];
            std::lock_guard<std::mutex> guard(state->lock);
            [state->retained removeObjectIdenticalTo:buffer];
        } else {
            WMReport("-[WebMetalCommandQueue dispatchAvailableCompletionNotifications]",
                     "not implemented: an event record other than CommandBufferScheduled or CommandBufferCompleted "
                     "arrived on a command queue");
        }
    }
}

// Rings on `queue` whenever the host has events for the command queue.
bool watchQueue(WebMetalCommandQueue *queue, dispatch_queue_t on)
{
    __weak WebMetalCommandQueue *weakQueue = queue;
    webmetal::Connection *connection = queue->_connection;
    const uint64_t handle = queue->_handle;
    std::string problem;
    if (!connection->Watch(handle, on,
                           ^{
                               WebMetalCommandQueue *strong = weakQueue;
                               if (strong)
                                   dispatchEvents(strong, connection, handle);
                           },
                           &problem)) {
        WMReport("-[WebMetalCommandQueue initWithDevice:descriptor:]", problem.c_str());
        return false;
    }
    return true;
}

// The error IOGPU makes of the kernel's command-buffer status: -[NSError(IOGPUMetalErrors) initWithIOGPUError:]
// (0x1fcb3661c). Statuses up to 18 index IOGPU's tables of MTLCommandBufferError codes (0x1fcb48638), status names
// (0x229637740) and descriptions (0x2296377d8); any other status is "Internal Error" with code 1. The description is
// "<description> (<status as %08x>:<name>)", and the underlying error is the status in IOGPUCommandQueueErrorDomain.
NSError *iogpuError(int64_t status)
{
    struct Entry {
        NSInteger code;
        const char *name;
        const char *description;
    };
    static const Entry kEntries[] = {
        {0, "kIOReturnSuccess", "No Error"},
        {1, "Internal Error", "Internal Error"},
        {2, "kIOGPUCommandBufferCallbackErrorTimeout", "Caused GPU Timeout Error"},
        {2, "kIOGPUCommandBufferCallbackErrorHang", "Caused GPU Hang Error"},
        {4, "kIOGPUCommandBufferCallbackErrorSubmissionsIgnored", "Ignored (for causing prior/excessive GPU errors)"},
        {1, "kIOGPUCommandBufferCallbackErrorInnocentVictim", "Discarded (victim of GPU error/recovery)"},
        {7, "kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted",
         "Insufficient Permission (to submit GPU work from background)"},
        {7, "kIOGPUCommandBufferCallbackErrorNotPermitted", "Insufficient Permission"},
        {8, "kIOGPUCommandBufferCallbackErrorOutOfMemory", "Insufficient Memory"},
        {9, "kIOGPUCommandBufferCallbackErrorInvalidResource", "Invalid Resource"},
        {1, "Internal Error", "Internal Error"},
        {3, "kIOGPUCommandBufferCallbackErrorPageFault", "Caused GPU Address Fault Error"},
        {10, "kIOGPUCommandBufferCallbackErrorExceededHardwareLimit",
         "Too many unique viewports, scissor rectangles or depth-bias values to support memoryless render pass "
         "attachments."},
        {10, "kIOGPUCommandBufferCallbackErrorOutOfMemoryForParameterBuffer",
         "Too much geometry to support memoryless render pass attachments."},
        {1, "Internal Error", "Internal Error"},
        {1, "kIOGPUCommandBufferCallbackErrorDeviceRemoved", "Device Removed"},
        {16, "kIOGPUCommandBufferCallbackErrorProtectionViolation", "Protection Violation"},
        {12, "kIOGPUCommandBufferCallbackErrorStackOverflow",
         "GPU Command Buffer execution stopped due to Stack Overflow Exception. Please check the "
         "[MTLComputePipelineDescriptor maxCallStackDepth] setting."},
        {17, "kIOGPUCommandBufferCallbackErrorConditionalEventAbort",
         "GPU Command Buffer execution stopped due to conditional event abort request."},
    };
    static const Entry kOther = {1, "Internal Error", "Internal Error"};
    const Entry &entry = (uint64_t)status < sizeof kEntries / sizeof kEntries[0] ? kEntries[status] : kOther;
    NSString *description =
        [NSString stringWithFormat:@"%s (%08x:%s)", entry.description, (unsigned)status, entry.name];
    NSError *underlying = [NSError errorWithDomain:@"IOGPUCommandQueueErrorDomain" code:status userInfo:nil];
    return [[NSError alloc] initWithDomain:MTLCommandBufferErrorDomain
                                      code:entry.code
                                  userInfo:@{NSLocalizedDescriptionKey : description, NSUnderlyingErrorKey : underlying}];
}

}  // namespace

namespace webmetal {

NSError *CommandBufferError(const wire::Failure &failure, const char *what, bool report)
{
    NSString *message = [[NSString alloc] initWithBytes:failure.message.data()
                                                 length:failure.message.size()
                                               encoding:NSUTF8StringEncoding];
    if (failure.status == 1 && failure.domain) {
        NSString *domain = [[NSString alloc] initWithBytes:failure.domain->data()
                                                    length:failure.domain->size()
                                                  encoding:NSUTF8StringEncoding];
        // The kernel's command-buffer status (wire schema, Failure): the error IOGPU makes of it.
        if ([domain isEqualToString:@"IOGPUCommandQueueErrorDomain"])
            return iogpuError(failure.code);
        return [NSError errorWithDomain:domain
                                   code:(NSInteger)failure.code
                               userInfo:@{NSLocalizedDescriptionKey : message ?: @""}];
    }
    // Statuses 2 (not implemented by the host), 3 (a capability the host cannot honor) and 4 (a protocol violation)
    // are no Metal results: they are reported, and the command buffer completes with an error that says so.
    if (report)
        WMReport(what, failure.message.c_str());
    return [NSError errorWithDomain:@"WebMetalErrorDomain"
                               code:(NSInteger)failure.status
                           userInfo:@{NSLocalizedDescriptionKey : message ?: @""}];
}

}  // namespace webmetal

@implementation WebMetalCommandQueue

// -[AGXA10FamilyCommandQueue initWithDevice:descriptor:] initializes the framework queue, applies the descriptor's
// configuration to the driver queue and makes a fence; IOGPUMetalCommandQueue creates the kernel's command queue with
// the queue's QoS and OpenGL flag and targets its notifications at the completion dispatch queue.
- (instancetype)initWithDevice:(id<MTLDevice>)device descriptor:(id)descriptor
{
    self = [super initWithDevice:device descriptor:descriptor];
    id<WebMetalCommandQueueDescriptorState> settings = descriptor;
    if (!self)
        return nil;
    _device = device;
    _connection = webmetal::DeviceConnection((WebMetalDevice *)device);
    if (!_connection) {
        WMReport("-[WebMetalCommandQueue initWithDevice:descriptor:]", "refused: the device has no WebMetal connection");
        return nil;
    }
    _handle = _connection->NewHandle();
    webmetal::wire::CreateCommandQueueWithDescriptor request;
    request.queue = _handle;
    auto &d = request.descriptor;
    d.commitSynchronously = [settings commitSynchronously];
    d.devicePartition = [settings devicePartition];
    d.disableAsyncCompletionDispatch = [settings disableAsyncCompletionDispatch];
    d.disableCrossQueueHazardTracking = [settings disableCrossQueueHazardTracking];
    d.disableIoFencing = [settings disableIOFencing];
    d.enableLowLatencySignalSharedEvent = [settings enableLowLatencySignalSharedEvent];
    d.enableLowLatencyWaitSharedEvent = [settings enableLowLatencyWaitSharedEvent];
    d.isOpenGlQueue = [settings isOpenGLQueue];
    d.lockParameterBufferSizeToMax = [settings lockParameterBufferSizeToMax];
    d.maxCommandBufferCount = [settings maxCommandBufferCount];
    d.qosLevel = [settings qosLevel];
    if (d.disableAsyncCompletionDispatch) {
        // IOGPU then leaves its notification queue without a dispatch queue: completions run only from
        // -dispatchAvailableCompletionNotifications.
        WMReport("-[WebMetalCommandQueue initWithDevice:descriptor:]",
                 "not implemented: disableAsyncCompletionDispatch (completions delivered only on request)");
        return nil;
    }
    bool created;
    std::string problem;
    webmetal::wire::Failure failure;
    if (webmetal::PlainQueueRequested()) {
        webmetal::wire::CreateCommandQueue plain;
        plain.queue = _handle;
        webmetal::wire::Reply<webmetal::wire::CreateCommandQueueReply> reply;
        created = _connection->TryCall(plain, &reply, &problem);
        if (created && !reply.ok) {
            created = false;
            failure = reply.failure;
        }
    } else {
        webmetal::wire::Reply<webmetal::wire::CreateCommandQueueWithDescriptorReply> reply;
        created = _connection->TryCall(request, &reply, &problem);
        if (created && !reply.ok) {
            created = false;
            failure = reply.failure;
        }
    }
    if (!created) {
        // IOGPUCommandQueueCreateWithQoS failing makes the driver's initializer return nil.
        if (failure.status != 1)
            WMReport("-[WebMetalCommandQueue initWithDevice:descriptor:]",
                     (problem.empty() ? failure.message : problem).c_str());
        _handle = 0;
        return nil;
    }
    webmetal::RegisterHandle(self, _handle);
    {
        std::lock_guard<std::mutex> guard(gQueuesLock);
        gQueues[_handle] = std::make_shared<QueueEvents>();
    }
    if (!watchQueue(self, completionDispatch(self)))
        return nil;
    // -[IOGPUMetalCommandQueue initWithDevice:descriptor:] (0x1fcb26578) starts with priority 1 and background priority
    // 3; -[AGXA10FamilyCommandQueue initWithDevice:descriptor:] (0x21c8ca504) then sets priority 1 on internal builds.
    _gpuPriority = 1;
    _backgroundGPUPriority = 3;
    if (webmetal::InternalBuild(self, _cmd))
        [(id<WebMetalQueuePriority>)self setGPUPriority:1];
    return self;
}

- (void)dealloc
{
    if (!_handle)
        return;
    _connection->Unwatch(_handle);
    {
        std::lock_guard<std::mutex> guard(gQueuesLock);
        gQueues.erase(_handle);
    }
    webmetal::UnregisterHandle(self);
    webmetal::ReleaseHandle(_connection, _handle);
}

// IOGPUMetalCommandQueue keeps the device it was created on.
- (id<MTLDevice>)device
{
    return _device;
}

- (id<MTLCommandBuffer>)commandBuffer
{
    return [[WebMetalCommandBuffer alloc] initWithQueue:self retainedReferences:YES];
}

- (id<MTLCommandBuffer>)commandBufferWithUnretainedReferences
{
    return [[WebMetalCommandBuffer alloc] initWithQueue:self retainedReferences:NO];
}

// -[AGXA10FamilyCommandQueue commandBufferWithDescriptor:] creates the buffer with the descriptor's retainedReferences,
// then sets its errorOptions and copies deadlineProfile, captureProgramAddressTable,
// cloneIntersectionFunctionTablesPerDispatch and disableFineGrainedComputePreemption into the driver's command buffer.
- (id<MTLCommandBuffer>)commandBufferWithDescriptor:(MTLCommandBufferDescriptor *)descriptor
{
    id<WebMetalCommandBufferDescriptorState> d = (id<WebMetalCommandBufferDescriptorState>)descriptor;
    if ([d deadlineProfile] || [d captureProgramAddressTable] || [d cloneIntersectionFunctionTablesPerDispatch] ||
        [d disableFineGrainedComputePreemption])
        WMLoudMethod(self, _cmd,
                     "deadlineProfile, captureProgramAddressTable, cloneIntersectionFunctionTablesPerDispatch and "
                     "disableFineGrainedComputePreemption configure AGX command-stream features the host has no "
                     "form for");
    WebMetalCommandBuffer *buffer = [[WebMetalCommandBuffer alloc] initWithQueue:self
                                                              retainedReferences:descriptor.retainedReferences];
    [buffer setErrorOptions:descriptor.errorOptions];
    return buffer;
}

- (void)setLabel:(NSString *)label
{
    [super setLabel:label];
    webmetal::wire::SetLabel request;
    request.object = _handle;
    if (!webmetal::wire::strFromNSString(label, request.label))
        WMLoudMethod(self, _cmd, "the label has no UTF-8 form");
    webmetal::wire::Reply<webmetal::wire::SetLabelReply> reply;
    std::string problem;
    if (!_connection->TryCall(request, &reply, &problem))
        WMLoudMethod(self, _cmd, problem.c_str());
    if (!reply.ok)
        WMLoudMethod(self, _cmd, reply.failure.message.c_str());
}

// IOGPU retargets the kernel queue's notifications at the new queue; _MTLCommandQueue's own setter is not called.
- (void)setCompletionQueue:(dispatch_queue_t)queue
{
    if (!queue)
        MTLReleaseAssertionFailure("-[WebMetalCommandQueue setCompletionQueue:]", __LINE__, "queue != nil", nullptr);
    Ivar ivar = completionDispatchIvar();
    void *previous = (__bridge void *)object_getIvar(self, ivar);
    object_setIvar(self, ivar, (__bridge id)CFBridgingRetain(queue));
    if (previous)
        CFRelease(previous);
    _connection->Unwatch(_handle);
    watchQueue(self, queue);
}

- (void)dispatchAvailableCompletionNotifications
{
    dispatchEvents(self, _connection, _handle);
}

// IOGPUMetalCommandQueue hands the kernel at most 32 command buffers per call.
- (void)submitCommandBuffers:(id<MTLCommandBuffer> __unsafe_unretained *)commandBuffers count:(NSUInteger)count
{
    for (NSUInteger at = 0; at < count; at += 32)
        [self _submitCommandBuffers:commandBuffers + at count:std::min<NSUInteger>(count - at, 32)];
}

- (void)_submitCommandBuffers:(id<MTLCommandBuffer> __unsafe_unretained *)commandBuffers count:(NSUInteger)count
{
    std::shared_ptr<QueueEvents> state = eventsOf(_handle);
    for (NSUInteger i = 0; i < count; i++) {
        WebMetalCommandBuffer *buffer = (WebMetalCommandBuffer *)commandBuffers[i];
        webmetal::CommandBufferState &cb = *buffer->_state;
        webmetal::wire::Commit request;
        request.commandBuffer = cb.handle;
        request.queue = _handle;
        if (!webmetal::wire::strFromNSString(buffer.label, request.label))
            WMLoudMethod(buffer, @selector(commit), "the command buffer's label has no UTF-8 form");
        request.retainedReferences = buffer.retainedReferences;
        request.errorOptions = buffer.errorOptions;
        request.stream = (uint64_t)(uintptr_t)cb.stream.bytes().data();
        request.streamLength = cb.stream.bytes().size();
        request.resources = {cb.stream.resources().data(), cb.stream.resources().size()};
        {
            std::lock_guard<std::mutex> guard(state->lock);
            state->inFlight[cb.handle] = buffer;
            [state->retained addObject:buffer];
        }
        webmetal::wire::Reply<webmetal::wire::CommitReply> reply;
        std::string problem;
        const bool delivered = _connection->TryCall(request, &reply, &problem);
        if (delivered && reply.ok) {
            cb.bound = true;
            cb.releases->bound(cb.stream.encoders(), _connection);
            continue;
        }
        // The host ran nothing of it: the command buffer completes here with the failure, as the kernel's
        // notifications would.
        webmetal::wire::Failure failure;
        if (delivered) {
            failure = reply.failure;
        } else {
            failure.status = 4;
            failure.message = "Commit: " + problem;
        }
        {
            std::lock_guard<std::mutex> guard(state->lock);
            state->inFlight.erase(cb.handle);
        }
        NSError *error = webmetal::CommandBufferError(failure, "-[MTLCommandBuffer commit]");
        dispatch_async(completionDispatch(self), ^{
            [buffer didScheduleWithStartTime:0 endTime:0 error:error];
            [self commandBufferDidComplete:buffer startTime:0 completionTime:0 error:error];
            std::lock_guard<std::mutex> guard(state->lock);
            [state->retained removeObjectIdenticalTo:buffer];
        });
    }
}

@end
