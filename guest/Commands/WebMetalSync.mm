// The device's command-path methods (queues, events, fences) and the event and fence classes (WebMetalCommands.h).

#import "WebMetalCommandState.h"
#import "../Plugin/WebMetalDevice.h"
#import "../Plugin/WebMetalReport.h"

#import <objc/runtime.h>

#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

@interface _MTLDevice (WebMetalFramework)
- (void)_incrementCommandQueueCount;
- (void)_decrementCommandQueueCount;
- (void)_purgeDevice;
@end

namespace webmetal {

namespace {

// Set while -newCommandQueue creates its queue, so the queue's initializer sends CreateCommandQueue (the record of
// -[MTLDevice newCommandQueue]) rather than CreateCommandQueueWithDescriptor.
thread_local bool tPlainQueue = false;

struct EntryPoint {
    const char *cls;
    const char *selector;
    const char *types;
};

// The Metal.framework methods WebMetalCommands.h declares, with their 21H461 type encodings.
const EntryPoint kEntryPoints[] = {
    {"_MTLCommandQueue", "initWithDevice:", "@24@0:8@16"},
    {"_MTLCommandQueue", "initWithDevice:descriptor:", "@32@0:8@16@24"},
    {"_MTLCommandQueue", "setLabel:", "v24@0:8@16"},
    {"_MTLCommandQueue", "commandBufferDidComplete:startTime:completionTime:error:", "v48@0:8@16Q24Q32@40"},
    {"_MTLCommandBuffer", "initWithQueue:retainedReferences:synchronousDebugMode:", "@32@0:8@16B24B28"},
    {"_MTLCommandBuffer", "didScheduleWithStartTime:endTime:error:", "v40@0:8Q16Q24@32"},
    {"_MTLCommandBuffer", "didCompleteWithStartTime:endTime:error:", "v40@0:8Q16Q24@32"},
    {"_MTLCommandBuffer", "setCurrentCommandEncoder:", "v24@0:8@16"},
    {"_MTLCommandBuffer", "setErrorOptions:", "v24@0:8Q16"},
    {"_MTLCommandBuffer", "commit", "v16@0:8"},
    {"_MTLCommandBuffer", "pushDebugGroup:", "v24@0:8@16"},
    {"_MTLCommandBuffer", "popDebugGroup", "v16@0:8"},
    {"_MTLCommandEncoder", "initWithCommandBuffer:", "@24@0:8@16"},
    {"_MTLCommandEncoder", "endEncoding", "v16@0:8"},
    {"_MTLCommandEncoder", "pushDebugGroup:", "v24@0:8@16"},
    {"_MTLCommandEncoder", "popDebugGroup", "v16@0:8"},
    {"_MTLParallelRenderCommandEncoder", "initWithCommandBuffer:renderPassDescriptor:", "@32@0:8@16@24"},
    {"_MTLParallelRenderCommandEncoder", "_renderCommandEncoderCommon", "@16@0:8"},
    {"_MTLParallelRenderCommandEncoder", "endEncoding", "v16@0:8"},
};

// Instance variables the driver layers read or write directly, as IOGPU does.
const EntryPoint kIvars[] = {
    {"_MTLCommandQueue", "_completionQueueDispatch", "@\"NSObject<OS_dispatch_queue>\""},
    {"_MTLCommandBuffer", "_globalTraceObjectID", "Q"},
    {"_MTLCommandEncoder", "_globalTraceObjectID", "Q"},
};

}  // namespace

bool CheckFrameworkEntryPoints(std::string *problem)
{
    static std::string missing;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        for (const EntryPoint &e : kEntryPoints) {
            Method method = class_getInstanceMethod(objc_getClass(e.cls), sel_registerName(e.selector));
            const char *types = method ? method_getTypeEncoding(method) : nullptr;
            if (!types || strcmp(types, e.types) != 0) {
                missing = std::string("-[") + e.cls + " " + e.selector + "] " +
                          (types ? std::string("has type encoding ") + types : "is missing") + ", expected " + e.types;
                return;
            }
        }
        for (const EntryPoint &e : kIvars) {
            Ivar ivar = class_getInstanceVariable(objc_getClass(e.cls), e.selector);
            const char *types = ivar ? ivar_getTypeEncoding(ivar) : nullptr;
            if (!types || strcmp(types, e.types) != 0) {
                missing = std::string(e.cls) + "." + e.selector + " " +
                          (types ? std::string("has type ") + types : "is missing") + ", expected " + e.types;
                return;
            }
        }
    });
    *problem = missing;
    return missing.empty();
}

bool PlainQueueRequested()
{
    return tPlainQueue;
}

}  // namespace webmetal

// Releases a shared event's handle when the event is deallocated (an associated object of the event).
@interface WebMetalSharedEventRegistration : NSObject
@end

@implementation WebMetalSharedEventRegistration {
@public
    webmetal::Connection *_connection;
    uint64_t _handle;
    const void *_event;
}

- (void)dealloc
{
    webmetal::ForgetHandle(_event);
    ReleaseHandle(_connection, _handle);
}

@end

@protocol WebMetalEventPort
- (uint32_t)eventPort;
@end

namespace webmetal {

uint64_t SharedEventHandle(Connection *connection, id event, std::string *problem)
{
    static std::mutex lock;
    static char key;
    std::lock_guard<std::mutex> guard(lock);
    if (const uint64_t known = HandleOf(event))
        return known;
    if (![event conformsToProtocol:@protocol(MTLSharedEvent)] || ![event respondsToSelector:@selector(eventPort)]) {
        *problem = std::string("an event of class ") + object_getClassName(event) +
                   " is neither a WebMetal event nor a shared event with an event port";
        return 0;
    }
    wire::RegisterSharedEvent request;
    request.event = connection->NewHandle();
    request.port = [(id<WebMetalEventPort>)event eventPort];
    wire::Reply<wire::RegisterSharedEventReply> reply;
    if (!connection->TryCall(request, &reply, problem)) {
        *problem = "RegisterSharedEvent: " + *problem;
        return 0;
    }
    if (!reply.ok) {
        *problem = "the host refused RegisterSharedEvent (status " + std::to_string(reply.failure.status) + "): " +
                   reply.failure.message;
        return 0;
    }
    RegisterHandle(event, request.event);
    WebMetalSharedEventRegistration *registration = [WebMetalSharedEventRegistration new];
    registration->_connection = connection;
    registration->_handle = request.event;
    registration->_event = (__bridge const void *)event;
    objc_setAssociatedObject(event, &key, registration, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    return request.event;
}

}  // namespace webmetal

using namespace webmetal;

@implementation WebMetalDevice (WebMetalCommands)

// -[AGXA10FamilyDevice newCommandQueue]: a driver queue through _MTLCommandQueue's -initWithDevice:, which configures
// it with the framework's default descriptor.
- (id<MTLCommandQueue>)newCommandQueue
{
    std::string problem;
    if (!CheckFrameworkEntryPoints(&problem))
        WMLoudMethod(self, _cmd, ("refused: the running Metal.framework lacks what the command path uses: " + problem).c_str());
    tPlainQueue = true;
    WebMetalCommandQueue *queue = [[WebMetalCommandQueue alloc] initWithDevice:self];
    tPlainQueue = false;
    return queue;
}

- (id<MTLCommandQueue>)newCommandQueueWithDescriptor:(id)descriptor
{
    std::string problem;
    if (!CheckFrameworkEntryPoints(&problem))
        WMLoudMethod(self, _cmd, ("refused: the running Metal.framework lacks what the command path uses: " + problem).c_str());
    return [[WebMetalCommandQueue alloc] initWithDevice:self descriptor:descriptor];
}

// AGX counts queues for its telemetry on top of _MTLDevice's count.
- (void)_incrementCommandQueueCount
{
    [super _incrementCommandQueueCount];
}

- (void)_decrementCommandQueueCount
{
    [super _decrementCommandQueueCount];
}

// _MTLDevice calls this when its last command queue goes away. -[AGXA10FamilyDevice _purgeDevice] purges the AGX
// device's own memory pools (AGX::Device::purgeDevice), which have no counterpart in the plugin, then calls _MTLDevice's.
- (void)_purgeDevice
{
    [super _purgeDevice];
}

// -[AGXA10FamilyDevice getSupportedCommandBufferErrorOptions] (0x21c8d4fb0) reads AGXA10FamilyDevice._impl+0x988, which
// the device's initializer sets to 1 unconditionally (-[AGXA10FamilyDevice initWithAcceleratorPort:
// simultaneousInstances:], str x8, [x28, #0x988] at 0x21c8d3354 with x8 = 1): encoder execution status.
- (NSUInteger)getSupportedCommandBufferErrorOptions
{
    return MTLCommandBufferErrorOptionEncoderExecutionStatus;
}

- (id<MTLEvent>)newEvent
{
    return [[WebMetalEvent alloc] initWithDevice:self];
}

- (id<MTLFence>)newFence
{
    return [[WebMetalFence alloc] initWithDevice:self];
}

@end

// Event ----------------------------------------------------------------------------------------------------------------

@implementation WebMetalEvent {
    __weak id<MTLDevice> _device;
    Connection *_connection;
    uint64_t _handle;
    NSString *_label;
    std::mutex _labelLock;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
{
    self = [super init];
    if (!self)
        return nil;
    _device = device;
    _connection = DeviceConnection((WebMetalDevice *)device);
    _handle = _connection->NewHandle();
    wire::CreateEvent request;
    request.event = _handle;
    wire::Reply<wire::CreateEventReply> reply;
    std::string problem;
    if (!_connection->TryCall(request, &reply, &problem) || !reply.ok) {
        WMReport("-[WebMetalDevice newEvent]", (problem.empty() ? reply.failure.message : problem).c_str());
        _handle = 0;
        return nil;
    }
    RegisterHandle(self, _handle);
    return self;
}

- (void)dealloc
{
    if (!_handle)
        return;
    UnregisterHandle(self);
    ReleaseHandle(_connection, _handle);
}

- (id<MTLDevice>)device
{
    return _device;
}

- (NSString *)label
{
    std::lock_guard<std::mutex> guard(_labelLock);
    return _label;
}

- (void)setLabel:(NSString *)label
{
    NSString *copy = [label copy];
    {
        std::lock_guard<std::mutex> guard(_labelLock);
        _label = copy;
    }
    wire::SetLabel request;
    request.object = _handle;
    if (!wire::strFromNSString(copy, request.label))
        WMLoudMethod(self, _cmd, "the label has no UTF-8 form");
    wire::Reply<wire::SetLabelReply> reply;
    std::string problem;
    if (!_connection->TryCall(request, &reply, &problem))
        WMLoudMethod(self, _cmd, problem.c_str());
    if (!reply.ok)
        WMLoudMethod(self, _cmd, reply.failure.message.c_str());
}

@end

// Fence ----------------------------------------------------------------------------------------------------------------

@implementation WebMetalFence {
    __weak id<MTLDevice> _device;
    Connection *_connection;
    uint64_t _handle;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
{
    self = [super init];
    if (!self)
        return nil;
    _device = device;
    _connection = DeviceConnection((WebMetalDevice *)device);
    _handle = _connection->NewHandle();
    wire::CreateFence request;
    request.fence = _handle;
    wire::Reply<wire::CreateFenceReply> reply;
    std::string problem;
    if (!_connection->TryCall(request, &reply, &problem) || !reply.ok) {
        WMReport("-[WebMetalDevice newFence]", (problem.empty() ? reply.failure.message : problem).c_str());
        _handle = 0;
        return nil;
    }
    RegisterHandle(self, _handle);
    return self;
}

- (void)dealloc
{
    if (!_handle)
        return;
    UnregisterHandle(self);
    ReleaseHandle(_connection, _handle);
}

- (id<MTLDevice>)device
{
    return _device;
}

@end
