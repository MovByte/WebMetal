// The Metal device of the WebMetal accelerator service (WebMetalDevice.h).

#import "WebMetalDevice.h"
#import "WebMetalConnection.h"
#import "WebMetalIOGPUDevice.h"
#import "WebMetalReport.h"
#import "WebMetalSurface.h"
#import "../Resources/WebMetalDepthStencilCache.h"

#import <objc/message.h>
#import <objc/runtime.h>

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
    // service (the macOS test shim), whose shim supplies the configuration instead.
    std::unique_ptr<webmetal::IOGPUDevice> _iogpuDevice;
    // The IOGPU device's configuration (IOGPUDevice +0xa0).
    webmetal::IOGPUConfig _iogpuConfig;
    // IOGPUMetalDevice._configBits, _videoRam and _sharedMemorySize, from the IOGPU device's configuration.
    uint32_t _configBits;
    uint64_t _videoRam;
    uint64_t _sharedMemorySize;
}
@end

// IOSurface.framework (exported, no SDK declaration).
extern "C" void *IOSurfaceGetGraphicsCommPageAddress(void);

namespace {

// IOGPU's kdebug trace gate: the word at the graphics commpage IOSurfaceGetGraphicsCommPageAddress gives, or a zeroed
// word where there is none (the once block of IOGPUDeviceCreateWithAPIProperty, 0x1fcb30374).
bool kdebugTracing()
{
    static const uint32_t *word = [] {
        const uint32_t *page = static_cast<const uint32_t *>(IOSurfaceGetGraphicsCommPageAddress());
        static const uint32_t zero = 0;
        return page ? page : &zero;
    }();
    return *word != 0;
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

uint32_t DeviceConfigBits(WebMetalDevice *device)
{
    return device->_configBits;
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
    if (const webmetal::IOGPUConfig *shim = webmetal::ShimIOGPUConfig()) {
        _iogpuConfig = *shim;
    } else {
        _iogpuDevice = webmetal::IOGPUDevice::Create(service, options.iogpuOptions, &problem);
        if (!_iogpuDevice) {
            WMReport("-[WebMetalDevice initWithAcceleratorPort:]", ("refused: " + problem).c_str());
            return nil;
        }
        _iogpuConfig = _iogpuDevice->Config();
    }
    // IOGPUDeviceGetConfig64 (IOGPU 0x1fcb2f128) copies the configuration's bits, device bits, texture and video memory
    // sizes and accelerator ID, and IOGPUDeviceGetSharedMemorySize (0x1fcb2f13c) its shared memory size; the initializer
    // then clears the video and texture memory sizes (0x1fcb2f140). Nothing in IOGPU, AGXMetalA10 or Metal.framework
    // reads the device bits, texture memory size or accelerator ID, or writes the others again.
    _configBits = _iogpuConfig.configBits;
    _sharedMemorySize = _iogpuConfig.sharedMemorySize;
    _videoRam = 0;
    // _MTLDevice._globalTraceObjectID takes IOGPUDeviceGetGlobalTraceObjectID (0x1fcb2f308): the trace object selector 4
    // answered, or 0 where it did not (the shim's kernel stand-in serves no selector 4).
    Ivar traceObject = class_getInstanceVariable([_MTLDevice class], "_globalTraceObjectID");
    if (!traceObject || strcmp(ivar_getTypeEncoding(traceObject), "Q") != 0) {
        WMReport("-[WebMetalDevice initWithAcceleratorPort:]", "refused: _MTLDevice has no _globalTraceObjectID");
        return nil;
    }
    *reinterpret_cast<uint64_t *>(reinterpret_cast<uint8_t *>((__bridge void *)self) + ivar_getOffset(traceObject)) =
        _iogpuDevice ? _iogpuDevice->GlobalTraceObjectID() : 0;
    // With kdebug tracing on, IOGPU records the device's creation as a trace event (IOGPUDeviceTraceEvent with class 8,
    // code 0xb, 0x1fcb2f31c-0x1fcb2f358), which the plugin does not implement.
    if (_iogpuDevice && kdebugTracing()) {
        WMReport("-[WebMetalDevice initWithAcceleratorPort:]",
                 "not implemented: IOGPU's kdebug trace events (IOGPUDeviceTraceEvent) while kdebug tracing is on");
        return nil;
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
    return _iogpuConfig.maxResourceSize;
}

// -[IOGPUMetalDevice sharedMemorySize] (IOGPU 0x1fcb30c80).
- (uint64_t)sharedMemorySize
{
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
    return _sharedMemorySize;
}

// -[IOGPUMetalDevice supportPriorityBand] (IOGPU 0x1fcb359e4): bit 6 of the configuration's bits.
- (BOOL)supportPriorityBand
{
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
