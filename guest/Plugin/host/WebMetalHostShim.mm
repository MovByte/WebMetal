// macOS test shim: the plugin's classes in a library that registers a WebMetalDevice with this Mac's Metal.framework
// the way Metal.framework's discovery does on the target (_MTLRegisterDevices: -initWithAcceleratorPort:, then
// MTLAddDevice), for the host checks and for the reference harness (tests/reference, `--device provider:`).
//
// It is a test vehicle, not a port. Where it differs from the target:
//  - The service. macOS has no WebMetal accelerator service; the device is given the I/O Registry's root entry, so its
//    acceleratorPort and registryID are that entry's, and its control requests go to the host executor running in this
//    process (executor/lib.rs) instead of the service's user client.
//  - Device state the kernel supplies. On the target the device reads the GPU's configuration from the service's IOGPU
//    device client as IOGPU.framework does (WebMetalIOGPUDevice.h): maxBufferLength, recommendedMaxWorkingSetSize,
//    sharedMemorySize, supportPriorityBand and the event class come from it. The shim's device has no such client; the
//    shim stands in for the kernel and gives the device the configuration the A10X's client answers (kA10XConfig),
//    except its largest resource, maxBufferLength, which is a sentinel value the caller chooses, so the checks can show
//    that the limits Metal.framework's -initLimits copies during MTLAddDevice carry exactly that input. It serves no
//    trace object (selector 4), so the device's global trace object ID is 0, as where the selector fails.
//  - Shared events. The target's kernel resolves a shared event's port for the executor; here the shim does, with this
//    process's IOSurface shared events (Metal.framework's _MTLSharedEvent), and tells the executor about CPU-side
//    signals it observes.
//  - The layouts. This library is built against the layouts of the Mac's Metal.framework, recorded when it was built
//    (WebMetalLayoutHost.mm); the iOS bundle requires 21H461's.

#import "../WebMetalConnection.h"
#import "../WebMetalDevice.h"
#import "../WebMetalIOGPUDevice.h"
#import "../WebMetalReport.h"
#import "../WebMetalSurface.h"

#import <objc/message.h>
#import <objc/runtime.h>

#include <mach/mach.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

extern "C" void MTLAddDevice(id<MTLDevice> device);

// The host executor in this process (executor/lib.rs).
extern "C" bool webmetal_shim_executor_start(char *error, size_t capacity);
extern "C" bool webmetal_shim_executor_control(const uint8_t *request, size_t length, uint8_t **reply,
                                               size_t *replyLength, char *error, size_t capacity);
extern "C" void webmetal_shim_executor_free(uint8_t *reply, size_t length);
extern "C" void webmetal_shim_executor_set_doorbell(void (*ring)(void *context, uint64_t queue), void *context);
struct WebMetalShimSharedEvents {
    void *context;
    bool (*resolve)(void *context, uint32_t port, uint64_t *event);
    bool (*value)(void *context, uint64_t event, uint64_t *value);
    bool (*signal)(void *context, uint64_t event, uint64_t value);
    void (*release)(void *context, uint64_t event);
};
extern "C" void webmetal_shim_executor_set_shared_events(const WebMetalShimSharedEvents *callbacks);
extern "C" void webmetal_shim_executor_shared_event_signaled(uint64_t event);

// _MTLSharedEvent's initializer from a send right (Metal.framework on 21H461 and on this Mac).
@protocol WebMetalShimMachPortEvent
- (id)initWithMachPort:(uint32_t)port;
@end

// The executor's shared events: each port the executor resolves names one of this task's IOSurface shared events, held
// as a _MTLSharedEvent of its own. A listener re-armed one value above each value it observes reports CPU-side signals.
namespace {

std::mutex gEventsLock;
uint64_t gNextEvent = 1;
std::map<uint64_t, id<MTLSharedEvent>> gEvents;
MTLSharedEventListener *gListener;

bool sharedEventLive(uint64_t key)
{
    std::lock_guard<std::mutex> guard(gEventsLock);
    return gEvents.count(key) != 0;
}

id<MTLSharedEvent> sharedEvent(uint64_t key)
{
    std::lock_guard<std::mutex> guard(gEventsLock);
    auto found = gEvents.find(key);
    return found == gEvents.end() ? nil : found->second;
}

void watchSharedEvent(uint64_t key, id<MTLSharedEvent> event, uint64_t value)
{
    [event notifyListener:gListener
                  atValue:value
                    block:^(id<MTLSharedEvent> e, uint64_t reached) {
                        webmetal_shim_executor_shared_event_signaled(key);
                        if (sharedEventLive(key))
                            watchSharedEvent(key, e, reached + 1);
                    }];
}

bool resolveSharedEvent(void *, uint32_t port, uint64_t *out)
{
    // -initWithMachPort: takes over a send right: the event gets one of its own.
    if (mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_SEND, 1) != KERN_SUCCESS)
        return false;
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)[(id<WebMetalShimMachPortEvent>)[objc_getClass("_MTLSharedEvent") alloc]
        initWithMachPort:port];
    if (!event)
        return false;
    uint64_t key;
    {
        std::lock_guard<std::mutex> guard(gEventsLock);
        key = gNextEvent++;
        gEvents[key] = event;
    }
    watchSharedEvent(key, event, event.signaledValue + 1);
    *out = key;
    return true;
}

bool sharedEventValue(void *, uint64_t key, uint64_t *value)
{
    id<MTLSharedEvent> event = sharedEvent(key);
    if (!event)
        return false;
    *value = event.signaledValue;
    return true;
}

bool signalSharedEvent(void *, uint64_t key, uint64_t value)
{
    id<MTLSharedEvent> event = sharedEvent(key);
    if (!event)
        return false;
    event.signaledValue = value;
    return true;
}

void releaseSharedEvent(void *, uint64_t key)
{
    std::lock_guard<std::mutex> guard(gEventsLock);
    gEvents.erase(key);
}

const WebMetalShimSharedEvents kSharedEvents = {nullptr, &resolveSharedEvent, &sharedEventValue, &signalSharedEvent,
                                                &releaseSharedEvent};

}  // namespace

// The connection's transport in this process: one control request to the executor.
static bool executorTransport(void *, const uint8_t *request, size_t length, std::vector<uint8_t> *reply,
                              std::string *problem)
{
    uint8_t *bytes = nullptr;
    size_t count = 0;
    char error[512];
    if (!webmetal_shim_executor_control(request, length, &bytes, &count, error, sizeof error)) {
        *problem = std::string("the shim's executor refused the request: ") + error;
        return false;
    }
    reply->assign(bytes, bytes + count);
    webmetal_shim_executor_free(bytes, count);
    return true;
}

// The executor rings this on its thread when it posts events for a command queue (0: device-level); in place of the
// accelerator service's async completion, the connection's doorbell for that queue runs on its dispatch queue.
static void executorDoorbell(void *, uint64_t queue)
{
    webmetal::Connection::Ring(queue);
}

// The sentinel -WMRefProvideDevice gives the device for maxBufferLength (the first one the device profile's
// extraction uses for kernel inputs).
static const uint64_t kDefaultMaxBufferLength = 0x5e5e1000;

static id<MTLDevice> gDevice;
static uint64_t gMaxBufferLength;

// What the A10X's IOGPU device client answers for selector 0 in the 21H461 kernel, apart from the largest resource:
// the AGXG9G configuration's bits 0x404340 (0xfffffff00531a9c0), reported as bits 0-15 with bit 21 into bit 12; no
// device bits, texture or video memory; IOSurface accelerator 1; and the wired limit of the iPad's memory, 0xAAAAC000
// (the configuration gex's accelerator service serves, crates/gex-kernel/src/darwin/accel/iogpu.rs).
static webmetal::IOGPUConfig a10xConfig(uint64_t maxResourceSize)
{
    webmetal::IOGPUConfig config = {};
    config.configBits = (0x404340 & 0xffff) | ((0x404340 >> 9) & 0x1000);
    config.accelID = 1;
    config.sharedMemorySize = 0xAAAAC000;
    config.maxResourceSize = maxResourceSize;
    return config;
}

// Creates a WebMetalDevice, registers it with MTLAddDevice and returns it, or returns nil after a report. Once per
// process; a later call returns the same device when it asks for the same sentinel.
extern "C" __attribute__((visibility("default"))) id<MTLDevice> WebMetalHostShimRegister(uint64_t maxBufferLength)
{
    @synchronized([WebMetalDevice class]) {
        if (gMaxBufferLength) {
            if (maxBufferLength != gMaxBufferLength) {
                WMReport("WebMetalHostShimRegister", "refused: the device exists with another maxBufferLength sentinel");
                return nil;
            }
            return gDevice;
        }
        gMaxBufferLength = maxBufferLength;
        char error[512];
        if (!webmetal_shim_executor_start(error, sizeof error)) {
            WMReport("WebMetalHostShimRegister", (std::string("refused: the host executor did not start: ") + error).c_str());
            return nil;
        }
        webmetal::SetTransportOverride(executorTransport, nullptr);
        webmetal::SetShimIOGPUConfig(a10xConfig(maxBufferLength));
        webmetal_shim_executor_set_doorbell(executorDoorbell, nullptr);
        gListener = [[MTLSharedEventListener alloc] init];
        webmetal_shim_executor_set_shared_events(&kSharedEvents);
        // Metal.framework's device array, which MTLAddDevice requires, exists once the system's devices are listed.
        (void)MTLCopyAllDevices();
        io_registry_entry_t root = IORegistryGetRootEntry(kIOMainPortDefault);
        WebMetalDevice *device = [[WebMetalDevice alloc] initWithAcceleratorPort:root];
        IOObjectRelease(root);
        if (!device)
            return nil;
        MTLAddDevice(device);
        // With Metal's API validation enabled (MTL_DEBUG_LAYER), MTLAddDevice wraps the device and lists the wrapper,
        // which -_deviceWrapper returns; clients receive that object, as MTLCopyAllDevices hands it out.
        id<MTLDevice> listed = ((id(*)(id, SEL))objc_msgSend)(device, sel_registerName("_deviceWrapper"));
        if (![MTLCopyAllDevices() containsObject:listed]) {
            WMReport("WebMetalHostShimRegister", "refused: MTLAddDevice did not list the device");
            return nil;
        }
        gDevice = listed;
        return listed;
    }
}

// The reference harness's provider entry point (tests/reference/README.md, "Device specifications").
extern "C" __attribute__((visibility("default"))) id<MTLDevice> WMRefProvideDevice(void)
{
    return WebMetalHostShimRegister(kDefaultMaxBufferLength);
}

// The plugin's guard over any layout set, for the checks: returns 1 when the running classes match.
extern "C" __attribute__((visibility("default"))) int WebMetalHostShimCheckLayouts(const webmetal::LayoutSet *layouts,
                                                                                    char *difference, size_t size)
{
    std::string text;
    const bool same = webmetal::CheckLayouts(*layouts, &text);
    if (size)
        snprintf(difference, size, "%s", text.c_str());
    return same;
}

extern "C" __attribute__((visibility("default"))) const webmetal::LayoutSet *WebMetalHostShimExpectedLayouts(void)
{
    return &webmetal::kExpectedLayouts;
}
