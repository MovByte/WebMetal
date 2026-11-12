// The Metal device of the WebMetal accelerator service (WebMetalDevice.h).

#import "WebMetalDevice.h"
#import "WebMetalConnection.h"
#import "WebMetalIOGPUDevice.h"
#import "WebMetalReport.h"
#import "WebMetalSurface.h"
#import "../Resources/WebMetalDepthStencilCache.h"

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

struct DeviceOptions {
    uint64_t creationFlags;
    uint16_t iogpuOptions;
};

// -[AGXA10FamilyDevice initWithAcceleratorPort:simultaneousInstances:] (AGXMetalA10 0x21c8d1c50) clears a process-wide
// word, runs gatherDeviceOptions's block once per process (0x21c8fd014: the word loses its low 25 bits, then takes
// Metal's overriding device creation flags when MTLGetOverridenDeviceCreationFlags supplies them), and keeps the word as
// the device's creation flags. A device created after the first one therefore keeps 0. The block also ORs the word's
// bit 24 as bit 5 and its bits 0-1 as bits 3-4 into a process-wide halfword (0x22f963f10, 0 in the image), which the
// initializer passes, before anything else, as the options of -[IOGPUMetalDevice initWithAcceleratorPort:options:]
// (0x21c8d1cac).
DeviceOptions gatherDeviceOptions()
{
    static uint64_t word;
    static uint16_t options;
    static dispatch_once_t once;
    word = 0;
    dispatch_once(&once, ^{
        word &= ~uint64_t(0x1ffffff);
        uint64_t overridden = 0;
        if (MTLGetOverridenDeviceCreationFlags(&overridden))
            word = overridden;
        options |= uint16_t(((word >> 19) & 0x20) | ((word & 0x3) << 3));
    });
    return {word, options};
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
    // The AGX device's depth-stencil state cache (AGX::Device +0x800).
    webmetal::DepthStencilCache _depthStencilStates;
    // IOGPUMetalDevice._deviceRef: the IOGPU device (WebMetalIOGPUDevice.h). None in a process without an accelerator
    // service (the macOS test shim).
    std::unique_ptr<webmetal::IOGPUDevice> _iogpuDevice;
    // IOGPUMetalDevice._configBits, _videoRam and _sharedMemorySize, from the IOGPU device's configuration.
    uint32_t _configBits;
    uint64_t _videoRam;
    uint64_t _sharedMemorySize;
}
@end

namespace {

// The IOGPU device a getter reads. A device has none in a process without an accelerator service (the macOS test shim),
// where the getter is LOUD.
const webmetal::IOGPUDevice &iogpuDevice(WebMetalDevice *device, SEL cmd)
{
    if (!device->_iogpuDevice)
        WMLoudMethod(device, cmd, "the process has no IOGPU device client to supply it (the macOS test shim)");
    return *device->_iogpuDevice;
}

}  // namespace

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

DepthStencilCache &DeviceDepthStencilStates(WebMetalDevice *device)
{
    return device->_depthStencilStates;
}

}  // namespace webmetal

@implementation WebMetalDevice

// The AGX initializer gathers the device options, then -[IOGPUMetalDevice initWithAcceleratorPort:options:] on 21H461
// initializes _MTLDevice, creates the IOGPU device (a device whose IOGPU device cannot be created is released and nil
// returned, 0x1fcb2f420), keeps the service's registry entry ID and a reference to the service, and copies what it reads
// of the IOGPU device's configuration.
- (instancetype)initWithAcceleratorPort:(io_service_t)service
{
    // The stubs are in place before _MTLDevice's initializer runs, as the driver's own methods are on the target.
    // A refused device is still initialized: ARC releases it, and _MTLDevice's -dealloc expects its initializer to
    // have run.
    const bool ready = prepareClasses();
    const DeviceOptions options = gatherDeviceOptions();
    self = [super init];
    if (!self || !ready)
        return nil;
    std::string problem;
    if (webmetal::ProcessHasIOGPUDevice()) {
        _iogpuDevice = webmetal::IOGPUDevice::Create(service, options.iogpuOptions, &problem);
        if (!_iogpuDevice) {
            WMReport("-[WebMetalDevice initWithAcceleratorPort:]", ("refused: " + problem).c_str());
            return nil;
        }
        // IOGPUDeviceGetConfig64 (IOGPU 0x1fcb2f128) copies the configuration's bits, device bits, texture and video
        // memory sizes and accelerator ID, and IOGPUDeviceGetSharedMemorySize (0x1fcb2f13c) its shared memory size; the
        // initializer then clears the video and texture memory sizes (0x1fcb2f140). Nothing in IOGPU, AGXMetalA10 or
        // Metal.framework reads the device bits, texture memory size or accelerator ID, or writes the others again.
        const webmetal::IOGPUConfig &config = _iogpuDevice->Config();
        _configBits = config.configBits;
        _sharedMemorySize = config.sharedMemorySize;
        _videoRam = 0;
    }
    uint64_t registryID = 0;
    const kern_return_t status = IORegistryEntryGetRegistryEntryID(service, &registryID);
    if (status != KERN_SUCCESS) {
        char text[96];
        snprintf(text, sizeof text, "refused: service 0x%x has no registry entry ID (status 0x%x)", service, status);
        WMReport("-[WebMetalDevice initWithAcceleratorPort:]", text);
        return nil;
    }
    std::unique_ptr<webmetal::Connection> connection = webmetal::Connection::Open(service, &problem);
    if (!connection) {
        WMReport("-[WebMetalDevice initWithAcceleratorPort:]", ("refused: " + problem).c_str());
        return nil;
    }
    IOObjectRetain(service);
    _acceleratorPort = service;
    _registryID = registryID;
    _connection = std::move(connection);
    _creationFlags = options.creationFlags;
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

// -[IOGPUMetalDevice maxBufferLength] (IOGPU 0x1fcb302e8): IOGPUDeviceGetMaxResourceSize (0x1fcb3031c), the
// configuration's largest resource (IOGPUDevice +0xc8), read at each call. Metal.framework's -initLimits copies it into
// the device's limits during MTLAddDevice.
- (NSUInteger)maxBufferLength
{
    return iogpuDevice(self, _cmd).Config().maxResourceSize;
}

// -[IOGPUMetalDevice sharedMemorySize] (IOGPU 0x1fcb30c80).
- (uint64_t)sharedMemorySize
{
    iogpuDevice(self, _cmd);
    return _sharedMemorySize;
}

// -[IOGPUMetalDevice dedicatedMemorySize] (IOGPU 0x1fcb30c90): the video memory size, which the initializer clears.
- (uint64_t)dedicatedMemorySize
{
    return _videoRam;
}

// -[IOGPUMetalDevice recommendedMaxWorkingSetSize] (IOGPU 0x1fcb358d4): the video memory size, or the shared memory
// size when it is 0.
- (uint64_t)recommendedMaxWorkingSetSize
{
    if (_videoRam)
        return _videoRam;
    iogpuDevice(self, _cmd);
    return _sharedMemorySize;
}

// -[IOGPUMetalDevice supportPriorityBand] (IOGPU 0x1fcb359e4): bit 6 of the configuration's bits.
- (BOOL)supportPriorityBand
{
    iogpuDevice(self, _cmd);
    return (_configBits >> 6) & 1;
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
