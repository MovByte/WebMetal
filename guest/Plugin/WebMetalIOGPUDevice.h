// The device's IOGPU device: what IOGPU.framework's IOGPUDevice is to IOGPUMetalDevice.
//
// -[IOGPUMetalDevice initWithAcceleratorPort:options:] (IOGPU 0x1fcb2efb0) creates it with
// IOGPUDeviceCreateWithAPIProperty(service, "Metal", options) (0x1fcb2f8c8), which opens the accelerator service's IOGPU
// device client (IOGPUFamily's IOGPUDeviceUserClient, type 1 with the options in the type's high 16 bits) and asks it
// for the client's stamps and the GPU's configuration; a device whose IOGPU device cannot be created is not created.
// The device's limits are the configuration's (WebMetalDevice.mm). The plugin's work reaches the host through its own
// connection (WebMetalConnection.h, type 0), not IOGPU's command queues, so of what the client answers the device keeps
// only the configuration.

#pragma once

#include <IOKit/IOKitLib.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace webmetal {

// What the IOGPU device client's selector 0 answers (0xfffffff005bbb490 in the 21H461 kernel), 64 bytes that
// IOGPUDeviceCreateWithAPIProperty keeps at IOGPUDevice +0xa0. The sources named are fields and virtual methods of the
// accelerator's IOGPU.
struct IOGPUConfig {
    uint32_t configBits;        // +0x00: the IOGPU's bits (+0x1f0), bits 0-15 with bit 21 ORed into bit 12
    uint32_t deviceBits;        // +0x04: +0x1f8
    uint64_t textureRam;        // +0x08: the virtual at vtable offset 0x540
    uint64_t videoRam;          // +0x10: +0x208
    uint32_t accelID;           // +0x18: the IOSurface accelerator ID (+0xf8)
    uint32_t unwritten;         // +0x1c: the kernel writes nothing here
    uint64_t sharedMemorySize;  // +0x20: the wired limit, IOGPU::getSharedMemorySize(0) (0xfffffff005bd60fc)
    uint64_t maxResourceSize;   // +0x28: the largest resource for the client, the virtual at vtable offset 0x678
    uint64_t field258;          // +0x30: +0x258
    uint32_t field260;          // +0x38: the 16 bits at +0x260
    uint32_t field262;          // +0x3c: the 16 bits at +0x262
};
static_assert(offsetof(IOGPUConfig, accelID) == 0x18 && offsetof(IOGPUConfig, sharedMemorySize) == 0x20 &&
                  offsetof(IOGPUConfig, maxResourceSize) == 0x28 && sizeof(IOGPUConfig) == 0x40,
              "selector 0's output layout");

class IOGPUDevice {
public:
    // Creates the IOGPU device of `service` as IOGPUDeviceCreateWithAPIProperty(service, "Metal", options) does.
    // Returns null and describes the problem where IOGPU creates no device.
    static std::unique_ptr<IOGPUDevice> Create(io_service_t service, uint16_t options, std::string *problem);
    ~IOGPUDevice();

    IOGPUDevice(const IOGPUDevice &) = delete;
    IOGPUDevice &operator=(const IOGPUDevice &) = delete;

    // The configuration the client answered at creation (IOGPUDeviceGetConfig64, IOGPUDeviceGetSharedMemorySize and
    // IOGPUDeviceGetMaxResourceSize read it from IOGPUDevice +0xa0).
    const IOGPUConfig &Config() const { return config_; }

private:
    explicit IOGPUDevice(io_connect_t connect) : connect_(connect) {}

    io_connect_t connect_;
    IOGPUConfig config_ = {};
};

// For a process with no accelerator service (the macOS test shim): devices created afterwards have no IOGPU device, and
// what its configuration feeds is LOUD. Set before the first device is created.
void SetNoIOGPUDevice();

// Whether devices created now open an IOGPU device.
bool ProcessHasIOGPUDevice();

}  // namespace webmetal
