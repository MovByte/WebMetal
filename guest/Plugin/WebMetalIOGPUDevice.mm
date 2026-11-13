// The device's IOGPU device (WebMetalIOGPUDevice.h).

#import "WebMetalIOGPUDevice.h"

#include <mach/mach.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace webmetal {

namespace {

// IOGPUDeviceUserClient's open type and the selectors IOGPUDeviceCreateWithAPIProperty calls, with the structure sizes
// of the client's method table (0xfffffff006d404d0 in the 21H461 kernel).
constexpr uint32_t kUserClientType = 1;
constexpr uint32_t kSelectorConfig = 0;
constexpr uint32_t kSelectorStamps = 2;
constexpr size_t kStampsSize = 0x218;
constexpr uint32_t kSelectorSetAPIProperty = 6;
constexpr size_t kAPIPropertySize = 0x10;
constexpr uint32_t kSelectorTraceObject = 4;
constexpr size_t kTraceObjectSize = 0x20;

// The API IOGPUMetalDevice names (IOGPU 0x1fcb2f054).
constexpr char kAPI[] = "Metal";

bool gShim;
IOGPUConfig gShimConfig;

uint64_t u64At(const uint8_t *bytes, size_t offset)
{
    uint64_t v;
    memcpy(&v, bytes + offset, sizeof v);
    return v;
}

}  // namespace

void SetShimIOGPUConfig(const IOGPUConfig &config)
{
    gShimConfig = config;
    gShim = true;
}

const IOGPUConfig *ShimIOGPUConfig()
{
    return gShim ? &gShimConfig : nullptr;
}

std::unique_ptr<IOGPUDevice> IOGPUDevice::Create(io_service_t service, uint16_t options, std::string *problem)
{
    // The client's type: 1, the options in bits 16-31 (0x1fcb2f8f8). Where it cannot be opened IOGPU logs a fault
    // (0x1fcb40b2c) and creates no device.
    const uint32_t type = kUserClientType | uint32_t(options) << 16;
    io_connect_t connect = IO_OBJECT_NULL;
    kern_return_t status = IOServiceOpen(service, mach_task_self(), type, &connect);
    if (status != KERN_SUCCESS) {
        char text[128];
        snprintf(text, sizeof text, "IOServiceOpen of the IOGPU device client (type 0x%x) returned 0x%x", type, status);
        *problem = text;
        return nullptr;
    }
    // From here a failure releases the device, whose finalizer (ioGPUDeviceFinalize, 0x1fcb3b8f0) releases the
    // connection.
    std::unique_ptr<IOGPUDevice> device(new IOGPUDevice(connect));
    // The API, strncpy'd into 16 bytes, as the client's API property (0x1fcb2f950); IOGPU ignores the result, which is
    // the kernel's setProperty answer (0xfffffff005bbb754).
    char api[kAPIPropertySize];
    strncpy(api, kAPI, sizeof api);
    (void)IOConnectCallStructMethod(connect, kSelectorSetAPIProperty, api, sizeof api, nullptr, nullptr);
    // The client's stamps (0x1fcb2fa50): the address of its stamp mapping, the stamp count and each stamp's offset, for
    // IOGPU's command queues to wait on. The plugin's command buffers complete through its connection, so the device
    // keeps none of it; a failure creates no device, as IOGPU's does.
    std::vector<uint8_t> stamps(kStampsSize);
    size_t size = stamps.size();
    status = IOConnectCallStructMethod(connect, kSelectorStamps, nullptr, 0, stamps.data(), &size);
    if (status != KERN_SUCCESS) {
        char text[96];
        snprintf(text, sizeof text, "the IOGPU device client's stamps (selector %u) returned 0x%x", kSelectorStamps,
                 status);
        *problem = text;
        return nullptr;
    }
    // The configuration, into IOGPUDevice +0xa0 (0x1fcb2fafc); a failure creates no device.
    size = sizeof device->config_;
    status = IOConnectCallStructMethod(connect, kSelectorConfig, nullptr, 0, &device->config_, &size);
    if (status != KERN_SUCCESS) {
        char text[96];
        snprintf(text, sizeof text, "the IOGPU device client's configuration (selector %u) returned 0x%x",
                 kSelectorConfig, status);
        *problem = text;
        return nullptr;
    }
    // Then the trace object (0x1fcb2fb68), 32 bytes, whose failure IOGPU ignores: on success the device's global trace
    // object ID (+0x58 from the answer's +0x0), the task's trace page (+0x60 from +0x8, only when the page's length at
    // +0x18 is at least 4) and the client-shared block (+0x68 from +0x10) (0x1fcb2fb70-0x1fcb2fb90).
    uint8_t trace[kTraceObjectSize] = {};
    size = sizeof trace;
    if (IOConnectCallStructMethod(connect, kSelectorTraceObject, nullptr, 0, trace, &size) == KERN_SUCCESS) {
        device->traceObjectID_ = u64At(trace, 0x0);
        uint32_t pageLength;
        memcpy(&pageLength, trace + 0x18, sizeof pageLength);
        if (pageLength >= 4)
            device->tracePage_ = u64At(trace, 0x8);
        device->traceSharedBlock_ = u64At(trace, 0x10);
    }
    return device;
}

IOGPUDevice::~IOGPUDevice()
{
    IOConnectRelease(connect_);
}

}  // namespace webmetal
