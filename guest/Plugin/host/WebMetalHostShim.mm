// macOS test shim: the plugin's classes in a library that registers a WebMetalDevice with this Mac's Metal.framework
// the way Metal.framework's discovery does on the target (_MTLRegisterDevices: -initWithAcceleratorPort:, then
// MTLAddDevice), for the host checks and for the reference harness (tests/reference, `--device provider:`).
//
// It is a test vehicle, not a port. Where it differs from the target:
//  - The service. macOS has no WebMetal accelerator service; the device is given the I/O Registry's root entry, so its
//    acceleratorPort and registryID are that entry's, and its control requests go to the host executor running in this
//    process (executor/lib.rs) instead of the service's user client.
//  - Device state the kernel supplies. The 21H461 driver reads maxBufferLength from the kernel
//    (IOGPUDeviceGetMaxResourceSize) and Metal.framework's -initLimits copies it into the device's limits during
//    MTLAddDevice. WebMetal has no source for it yet, and the iOS bundle answers it with a LOUD stub. The shim stands
//    in for the kernel: it adds -maxBufferLength with a sentinel value the caller chooses, so the checks can show that
//    the limits carry exactly that input.
//  - The layouts. This library is built against the layouts of the Mac's Metal.framework, recorded when it was built
//    (WebMetalLayoutHost.mm); the iOS bundle requires 21H461's.

#import "../WebMetalConnection.h"
#import "../WebMetalDevice.h"
#import "../WebMetalReport.h"
#import "../WebMetalSurface.h"

#import <objc/message.h>
#import <objc/runtime.h>

#include <cstdio>
#include <cstring>

extern "C" void MTLAddDevice(id<MTLDevice> device);

// The host executor in this process (executor/lib.rs).
extern "C" bool webmetal_shim_executor_start(char *error, size_t capacity);
extern "C" bool webmetal_shim_executor_control(const uint8_t *request, size_t length, uint8_t **reply,
                                               size_t *replyLength, char *error, size_t capacity);
extern "C" void webmetal_shim_executor_free(uint8_t *reply, size_t length);

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

// The sentinel -WMRefProvideDevice gives the device for maxBufferLength (the first one the device profile's
// extraction uses for kernel inputs).
static const uint64_t kDefaultMaxBufferLength = 0x5e5e1000;

static id<MTLDevice> gDevice;
static uint64_t gMaxBufferLength;

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
        IMP fixture = imp_implementationWithBlock(^NSUInteger(id) { return maxBufferLength; });
        if (!class_addMethod([WebMetalDevice class], sel_registerName("maxBufferLength"), fixture, "Q16@0:8")) {
            WMReport("WebMetalHostShimRegister", "refused: WebMetalDevice already implements maxBufferLength");
            return nil;
        }
        char error[512];
        if (!webmetal_shim_executor_start(error, sizeof error)) {
            WMReport("WebMetalHostShimRegister", (std::string("refused: the host executor did not start: ") + error).c_str());
            return nil;
        }
        webmetal::SetTransportOverride(executorTransport, nullptr);
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
