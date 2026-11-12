// macOS test shim: the plugin's classes in a library that registers a WebMetalDevice with this Mac's Metal.framework
// the way Metal.framework's discovery does on the target (_MTLRegisterDevices: -initWithAcceleratorPort:, then
// MTLAddDevice), for the host checks and for the reference harness (tests/reference, `--device provider:`).
//
// It is a test vehicle, not a port. Where it differs from the target:
//  - The service. macOS has no WebMetal accelerator service; the device is given the I/O Registry's root entry, so its
//    acceleratorPort and registryID are that entry's.
//  - Device state the kernel supplies. The 21H461 driver reads maxBufferLength from the kernel
//    (IOGPUDeviceGetMaxResourceSize) and Metal.framework's -initLimits copies it into the device's limits during
//    MTLAddDevice. WebMetal has no source for it yet, and the iOS bundle answers it with a LOUD stub. The shim stands
//    in for the kernel: it adds -maxBufferLength with a sentinel value the caller chooses, so the checks can show that
//    the limits carry exactly that input.
//  - The layouts. This library is built against the layouts of the Mac's Metal.framework, recorded when it was built
//    (WebMetalLayoutHost.mm); the iOS bundle requires 21H461's.

#import "../WebMetalDevice.h"
#import "../WebMetalReport.h"
#import "../WebMetalSurface.h"

#import <objc/runtime.h>

#include <cstdio>
#include <cstring>

extern "C" void MTLAddDevice(id<MTLDevice> device);

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
        // Metal.framework's device array, which MTLAddDevice requires, exists once the system's devices are listed.
        (void)MTLCopyAllDevices();
        io_registry_entry_t root = IORegistryGetRootEntry(kIOMainPortDefault);
        WebMetalDevice *device = [[WebMetalDevice alloc] initWithAcceleratorPort:root];
        IOObjectRelease(root);
        if (!device)
            return nil;
        MTLAddDevice(device);
        if (![MTLCopyAllDevices() containsObject:device]) {
            WMReport("WebMetalHostShimRegister", "refused: MTLAddDevice did not list the device");
            return nil;
        }
        gDevice = device;
        return device;
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
