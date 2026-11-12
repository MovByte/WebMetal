// The Metal device of the WebMetal accelerator service (WebMetalDevice.h).

#import "WebMetalDevice.h"
#import "WebMetalConnection.h"
#import "WebMetalReport.h"
#import "WebMetalSurface.h"

#import <objc/message.h>

#include <dispatch/dispatch.h>

#include <atomic>
#include <cstdio>
#include <memory>
#include <string>

// The class adopts MTLDevice and MTLDeviceSPI as AGXA10XDevice does and implements only what the 21H461 driver layers
// implement. The protocols' other methods are inherited from _MTLDevice or are LOUD stubs added at run time
// (WebMetalStubs.mm); the compiler sees neither.
#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"

// Metal.framework's overriding device creation flags: true, with the flags, when an override is in effect.
extern "C" bool MTLGetOverridenDeviceCreationFlags(uint64_t *flags);

namespace {

// The base-class guard, then the LOUD stubs: once per process, before the first device is created.
bool prepareClasses()
{
    static bool ready;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        std::string problem;
        if (!webmetal::CheckLayouts(webmetal::kExpectedLayouts, &problem)) {
            const std::string text = std::string("refused: the running Metal.framework is not the one the plugin was "
                                                 "built for (") + webmetal::kExpectedLayouts.source + "): " + problem;
            WMReport("WebMetalDevice", text.c_str());
            return;
        }
        if (!webmetal::InstallLoudStubs(&problem)) {
            WMReport("WebMetalDevice", ("refused: " + problem).c_str());
            return;
        }
        ready = true;
    });
    return ready;
}

// -[AGXA10FamilyDevice initWithAcceleratorPort:simultaneousInstances:] (AGXMetalA10 0x21c8d1c50) clears a process-wide
// word, runs gatherDeviceOptions's block once per process (0x21c8fd014: the word loses its low 25 bits, then takes
// Metal's overriding device creation flags when MTLGetOverridenDeviceCreationFlags supplies them), and keeps the word as
// the device's creation flags. A device created after the first one therefore keeps 0.
uint64_t gatherCreationFlags()
{
    static uint64_t word;
    static dispatch_once_t once;
    word = 0;
    dispatch_once(&once, ^{
        word &= ~uint64_t(0x1ffffff);
        uint64_t overridden = 0;
        if (MTLGetOverridenDeviceCreationFlags(&overridden))
            word = overridden;
    });
    return word;
}

}  // namespace

@interface WebMetalDevice () {
@public
    io_service_t _acceleratorPort;
    uint64_t _registryID;
    // The connection to the accelerator service (WebMetalConnection.h).
    std::unique_ptr<webmetal::Connection> _connection;
    // The allocated sizes of the device's live resources (currentAllocatedSize).
    std::atomic<int64_t> _allocated;
    // Live resources whose allocated size has no source yet (guest/Resources/WebMetalResources.h).
    std::atomic<int64_t> _unsized;
    // IOGPUMetalDevice._deviceWrapper: the object Metal.framework wraps the device in (its debug layer), held weakly.
    __weak id _deviceWrapper;
    // The AGX device's creation flags (AGX::Device +0x750, -deviceCreationFlags).
    uint64_t _creationFlags;
}
@end


namespace webmetal {

Connection *DeviceConnection(WebMetalDevice *device)
{
    return device->_connection.get();
}

void DeviceAddAllocation(WebMetalDevice *device, int64_t bytes)
{
    device->_allocated.fetch_add(bytes, std::memory_order_relaxed);
}

int64_t DeviceAllocatedSize(WebMetalDevice *device)
{
    return device->_allocated.load(std::memory_order_relaxed);
}

void DeviceAddUnsized(WebMetalDevice *device, int64_t count)
{
    device->_unsized.fetch_add(count, std::memory_order_relaxed);
}

int64_t DeviceUnsizedCount(WebMetalDevice *device)
{
    return device->_unsized.load(std::memory_order_relaxed);
}

uint64_t DeviceCreationFlags(WebMetalDevice *device)
{
    return device->_creationFlags;
}

}  // namespace webmetal

@implementation WebMetalDevice

// -[IOGPUMetalDevice initWithAcceleratorPort:options:] on 21H461 initializes _MTLDevice, opens the kernel's GPU
// device, then keeps the service's registry entry ID and a reference to the service.
- (instancetype)initWithAcceleratorPort:(io_service_t)service
{
    // The stubs are in place before _MTLDevice's initializer runs, as the driver's own methods are on the target.
    // A refused device is still initialized: ARC releases it, and _MTLDevice's -dealloc expects its initializer to
    // have run.
    const bool ready = prepareClasses();
    self = [super init];
    if (!self || !ready)
        return nil;
    uint64_t registryID = 0;
    const kern_return_t status = IORegistryEntryGetRegistryEntryID(service, &registryID);
    if (status != KERN_SUCCESS) {
        char text[96];
        snprintf(text, sizeof text, "refused: service 0x%x has no registry entry ID (status 0x%x)", service, status);
        WMReport("-[WebMetalDevice initWithAcceleratorPort:]", text);
        return nil;
    }
    std::string problem;
    std::unique_ptr<webmetal::Connection> connection = webmetal::Connection::Open(service, &problem);
    if (!connection) {
        WMReport("-[WebMetalDevice initWithAcceleratorPort:]", ("refused: " + problem).c_str());
        return nil;
    }
    IOObjectRetain(service);
    _acceleratorPort = service;
    _registryID = registryID;
    _connection = std::move(connection);
    _creationFlags = gatherCreationFlags();
    return self;
}

- (void)dealloc
{
    if (_acceleratorPort)
        IOObjectRelease(_acceleratorPort);
}

- (io_service_t)acceleratorPort
{
    return _acceleratorPort;
}

- (uint64_t)registryID
{
    return _registryID;
}

// -[AGXA10FamilyDevice deviceCreationFlags] (AGXMetalA10 0x21c8db548).
- (NSUInteger)deviceCreationFlags
{
    return _creationFlags;
}

// -[IOGPUMetalDevice _setDeviceWrapper:] (IOGPU 0x1fcb35bb0): stores the wrapper weakly.
- (void)_setDeviceWrapper:(id)wrapper
{
    _deviceWrapper = wrapper;
}

// -[IOGPUMetalDevice _deviceWrapper] (IOGPU 0x1fcb35bc4): the wrapper's own outermost wrapper, or the device itself when
// nothing wraps it.
- (id)_deviceWrapper
{
    id wrapper = _deviceWrapper;
    if (wrapper)
        return ((id(*)(id, SEL))objc_msgSend)(wrapper, _cmd);
    return self;
}

// -[AGXA10FamilyDevice minimumTextureBufferAlignmentForPixelFormat:] (0x21c8db1f4 on 21H461) is a tail call to
// minimumLinearTextureAlignmentForPixelFormat:, which _MTLDevice implements; its answers and its validation failures
// are Metal.framework's.
- (NSUInteger)minimumTextureBufferAlignmentForPixelFormat:(MTLPixelFormat)format
{
    return [self minimumLinearTextureAlignmentForPixelFormat:format];
}

@end
