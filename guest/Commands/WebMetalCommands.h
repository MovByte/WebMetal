// The command path: command queues, command buffers, encoders, events and fences.
//
// Each class stands in for the 21H461 driver class guest/Plugin/roles.toml names and subclasses the same
// Metal.framework base, so the framework's command-buffer status machine, handler lists, waits, timing fields and
// misuse assertions run as they do over the A10X driver. What the driver layers (AGXMetalA10 and IOGPU.framework) do
// is done here: encoders write wire records into their command buffer's stream (wire/README.md, "Stream"), the queue
// submits each committed command buffer with Commit, and the host's CommandBufferScheduled and CommandBufferCompleted
// events drive the framework's -didScheduleWithStartTime:endTime:error: and
// -commandBufferDidComplete:startTime:completionTime:error: on the queue's completion dispatch queue, as IOGPU's
// notification queue does.

#pragma once

#import "../Private/MetalPrivate.h"
#import "../Plugin/WebMetalConnection.h"
#import "../Plugin/WebMetalDevice.h"
#import "../Resources/WebMetalResources.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

// Private protocols the driver classes adopt on 21H461. The definitions the runtime uses are Metal.framework's; these
// only let the plugin's classes adopt them.
@protocol MTLCommandQueueSPI <MTLCommandQueue>
@end
@protocol MTLCommandBufferSPI <MTLCommandBuffer>
@end
@protocol MTLRenderCommandEncoderSPI <MTLRenderCommandEncoder>
@end
@protocol MTLComputeCommandEncoderSPI <MTLComputeCommandEncoder>
@end
@protocol MTLBlitCommandEncoderSPI <MTLBlitCommandEncoder>
@end
@protocol MTLParallelRenderCommandEncoderSPI <MTLParallelRenderCommandEncoder>
@end
@protocol MTLFenceSPI <MTLFence>
@end

// Metal.framework methods of the bases the command path calls. 21H461 implements each of them in the base class named;
// webmetal::CheckFrameworkEntryPoints verifies them in the running framework before the first queue is created.
@interface _MTLCommandQueue (WebMetalFramework)
- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (instancetype)initWithDevice:(id<MTLDevice>)device descriptor:(id)descriptor;
- (void)setLabel:(NSString *)label;
- (void)commandBufferDidComplete:(id<MTLCommandBuffer>)commandBuffer
                       startTime:(uint64_t)startTime
                  completionTime:(uint64_t)completionTime
                           error:(NSError *)error;
@end

@interface _MTLCommandBuffer (WebMetalFramework)
- (instancetype)initWithQueue:(id<MTLCommandQueue>)queue
           retainedReferences:(BOOL)retainedReferences
          synchronousDebugMode:(BOOL)synchronousDebugMode;
- (void)didScheduleWithStartTime:(uint64_t)startTime endTime:(uint64_t)endTime error:(NSError *)error;
- (void)didCompleteWithStartTime:(uint64_t)startTime endTime:(uint64_t)endTime error:(NSError *)error;
- (void)setCurrentCommandEncoder:(id<MTLCommandEncoder>)encoder;
- (void)commit;
- (void)setLabel:(NSString *)label;
- (void)pushDebugGroup:(NSString *)string;
- (void)popDebugGroup;
- (void)setErrorOptions:(MTLCommandBufferErrorOption)errorOptions;
- (BOOL)ownedByParallelEncoder;
@end

@interface _MTLCommandEncoder (WebMetalFramework)
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)endEncoding;
- (void)setLabel:(NSString *)label;
- (void)pushDebugGroup:(NSString *)string;
- (void)popDebugGroup;
@end

@interface _MTLParallelRenderCommandEncoder (WebMetalFramework)
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                 renderPassDescriptor:(MTLRenderPassDescriptor *)descriptor;
- (id<MTLCommandBuffer>)_renderCommandEncoderCommon;
- (void)endEncoding;
- (void)setLabel:(NSString *)label;
- (void)pushDebugGroup:(NSString *)string;
- (void)popDebugGroup;
@end

// Metal.framework's unconditional assertion failure (never returns).
extern "C" void MTLReleaseAssertionFailure(const char *function, uint32_t line, const char *condition, const char *message);

@interface WebMetalCommandQueue : _MTLCommandQueue <MTLCommandQueueSPI>
- (instancetype)initWithDevice:(id<MTLDevice>)device descriptor:(id)descriptor;
@end

@interface WebMetalCommandBuffer : _MTLCommandBuffer <MTLCommandBufferSPI>
- (instancetype)initWithQueue:(id<MTLCommandQueue>)queue retainedReferences:(BOOL)retainedReferences;
- (NSUInteger)getSupportedAndEnabledErrorOptions;
@end

// The encoder-creating methods (WebMetalEncoders.mm).
@interface WebMetalCommandBuffer (WebMetalEncoders)
- (id<MTLRenderCommandEncoder>)subRenderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)descriptor
                                                     subEncoderIndex:(NSUInteger)index
                                                         framebuffer:(void *)framebuffer;
@end

@interface WebMetalRenderCommandEncoder : _MTLCommandEncoder <MTLRenderCommandEncoderSPI>
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer descriptor:(MTLRenderPassDescriptor *)descriptor;
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           descriptor:(MTLRenderPassDescriptor *)descriptor
                      subEncoderIndex:(NSUInteger)index
                          framebuffer:(void *)framebuffer;
@end

@interface WebMetalComputeCommandEncoder : _MTLCommandEncoder <MTLComputeCommandEncoderSPI>
@end

@interface WebMetalBlitCommandEncoder : _MTLCommandEncoder <MTLBlitCommandEncoderSPI>
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer descriptor:(MTLBlitPassDescriptor *)descriptor;
@end

@interface WebMetalParallelRenderCommandEncoder : _MTLParallelRenderCommandEncoder <MTLParallelRenderCommandEncoderSPI>
@end

// The device's answer to -getSupportedCommandBufferErrorOptions (WebMetalSync.mm).
@protocol WebMetalDeviceErrorOptions
- (NSUInteger)getSupportedCommandBufferErrorOptions;
@end

// The event IOGPUMetalDevice -newEvent makes (_IOGPUMetalMTLEvent on 21H461 derives from NSObject through IOGPUMTLEvent).
@interface WebMetalEvent : NSObject <MTLEvent>
- (instancetype)initWithDevice:(id<MTLDevice>)device;
@end

// The fence IOGPUMetalDevice -newFence makes (IOGPUMetalFence : _MTLFence on 21H461).
@interface WebMetalFence : _MTLFence <MTLFenceSPI>
- (instancetype)initWithDevice:(id<MTLDevice>)device;
@end

namespace webmetal {

// Wire handles of the command path's objects (queues, events, fences, shared events). Each registers its handle when
// the host has created it and unregisters it when it is deallocated. HandleOf answers those, and the handle of a
// plugin buffer or texture the host holds (guest/Resources, StateOf); 0 for nil or for an object it has no handle for.
void RegisterHandle(id object, uint64_t handle);
void UnregisterHandle(id object);
// As UnregisterHandle, for an object that is being deallocated and is known only by its address.
void ForgetHandle(const void *object);
uint64_t HandleOf(id object);

// Sends Release for a handle the host holds; a failure is reported (the object is gone either way).
void ReleaseHandle(Connection *connection, uint64_t handle);

// Collects the pending event records of a command queue (0: device-level events). Returns false, with the reason, when
// the request fails or a record does not decode.
bool FetchEvents(Connection *connection, uint64_t queue, std::vector<wire::Event> &events, std::string *problem);

// The guest's clock in the units Metal.framework's command buffer times use (mach_absolute_time ticks), from the
// host's nanoseconds on the same clock (host/src/embed.rs, MonotonicNanos). IOGPU.framework converts the kernel's
// nanoseconds the same way (_IOGPUNotificationQueueDispatchAvailableCompletionNotifications: t * denom / numer).
uint64_t MachTicksFromNanoseconds(uint64_t nanoseconds);

// The records of one command buffer, or of one sub-encoder of a parallel render encoder, and the objects they name.
class Stream {
public:
    explicit Stream(bool retainReferences) : retain_(retainReferences) {}

    wire::Writer &writer() { return writer_; }
    const std::vector<uint8_t> &bytes() const { return writer_.out; }

    // Notes an existing object a record names: listed once in the Commit's resources and, when the command buffer
    // retains its references, kept alive until the command buffer completes.
    void use(id object, uint64_t handle);

    // Appends another stream's records and objects (a parallel encoder's sub-encoders), whose resource uses become the
    // uses of encoder `parent`.
    void splice(Stream &other, uint64_t parent);

    // How a record names a resource, for the stamps of its kernel resource's event 0: where the A10X driver is traced to
    // list it in the encoder's IOGPU resource list with flags the kernel stamps, or where that is not traced.
    enum class UseKind : uint8_t { Stamped, Untraced };
    struct Use {
        // The encoder whose work the host reports (EncoderCompleted): a top-level encoder; 0 outside any encoder.
        uint64_t encoder;
        // KernelResourceKey of the resource.
        uint64_t key;
        // The command-buffer-level event waits recorded before the use: the kernel processes the encoder, and stamps
        // what it lists, only once it has passed them.
        uint32_t waits;
        UseKind kind;
    };
    // The encoder whose records follow (`topLevel`: not a sub-encoder of a parallel render encoder), and its end.
    void beginEncoder(uint64_t handle, bool topLevel);
    void endEncoder() { current_ = 0; }
    void noteUse(uint64_t key, UseKind kind) { uses_.push_back(Use{current_, key, waits_, kind}); }
    const std::vector<Use> &uses() const { return uses_; }
    // The top-level encoders, in stream order: the host's encoder segments.
    const std::vector<uint64_t> &segments() const { return segments_; }
    // A command-buffer-level event wait was recorded; how many were.
    void noteWait() { waits_++; }
    uint32_t waits() const { return waits_; }

    const std::vector<uint64_t> &resources() const { return resources_; }

    // Handles of encoders this stream creates; the host binds them at Commit.
    std::vector<uint64_t> &encoders() { return encoders_; }

    // Drops the retained objects (the command buffer completed).
    void releaseObjects() { objects_ = nil; }

private:
    bool retain_;
    wire::Writer writer_;
    std::vector<uint64_t> resources_;
    std::unordered_set<uint64_t> named_;
    std::vector<uint64_t> encoders_;
    NSMutableArray *objects_ = nil;
    uint64_t current_ = 0;
    std::vector<Use> uses_;
    std::vector<uint64_t> segments_;
    uint32_t waits_ = 0;
};

// Handles whose Release must wait until the host has bound them (encoders, bound at their command buffer's Commit).
class PendingRelease {
public:
    // The encoder object is gone.
    void dead(uint64_t handle);
    // The host bound these handles; releases the dead ones now and the others when they die.
    void bound(const std::vector<uint64_t> &handles, Connection *connection);

private:
    std::mutex lock_;
    std::unordered_set<uint64_t> dead_;
    std::unordered_set<uint64_t> bound_;
    Connection *connection_ = nullptr;
};

// Verifies the Metal.framework methods the command path calls (the categories above) in the running framework, once.
// Returns false and describes the first one missing or with another type encoding.
bool CheckFrameworkEntryPoints(std::string *problem);

// The handle of a shared event (Metal.framework's _MTLSharedEvent, an IOSurface shared event on 21H461): registered
// with the host by its event port (RegisterSharedEvent) the first time a command names it, released when the event is
// deallocated. Returns 0 and the reason for an object that is no shared event or a registration the host refuses.
uint64_t SharedEventHandle(Connection *connection, id event, std::string *problem);

// Whether the queue being initialized on this thread comes from -newCommandQueue (WebMetalSync.mm).
bool PlainQueueRequested();

// The command buffer a command-buffer object belongs to, for encoders. Defined in WebMetalCommandBuffer.mm.
struct CommandBufferState;

}  // namespace webmetal
