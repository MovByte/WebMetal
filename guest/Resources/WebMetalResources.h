// The plugin's resource objects: buffers, textures, heaps, samplers, depth-stencil states, texture layouts and resource
// groups, standing in for the AGXMetalA10 classes of iPadOS 17.7.11 (21H461) (guest/Plugin/roles.toml).
//
// The 21H461 driver builds buffers and textures on IOGPU.framework's IOGPUMetalResource, whose state lives in one
// structure (_res) and whose answers come from it or from the kernel's resource. The plugin keeps the same state in
// webmetal::ResourceState and answers from it with the same rules; what the kernel's resource holds on the target lives
// in the host executor, reached through the device's connection (WebMetalConnection.h). Each class comments the 21H461
// method it reproduces, by symbol.

#pragma once

#import "../Plugin/WebMetalConnection.h"
#import "../Plugin/WebMetalDevice.h"
#import "../Plugin/WebMetalReport.h"

#import <IOSurface/IOSurfaceRef.h>
#import <Metal/Metal.h>
#import <os/lock.h>

#include <cstdint>
#include <string>

// Private protocols the driver classes adopt. Metal.framework defines them; these declarations only let the plugin's
// classes adopt them (as guest/Private/MetalPrivate.h does for MTLDeviceSPI).
@protocol MTLResourceSPI <MTLResource>
@end
@protocol MTLTextureSPI <MTLTexture>
@end
@protocol MTLTextureImplementation <MTLTexture>
@end
@protocol MTLHeapSPI <MTLHeap>
@end
@protocol MTLSamplerStateSPI <MTLSamplerState>
@end
@protocol MTLDepthStencilStateSPI <MTLDepthStencilState>
@end
@protocol MTLResourceGroupSPI <NSObject>
@end
@protocol MTLTextureLayout <NSObject>
@end

// Metal.framework's descriptor validation (-[MTLTextureDescriptorInternal validateWithDevice:] and the heap
// descriptor's), which the driver calls before it creates an object.
@protocol WMDescriptorValidation
- (BOOL)validateWithDevice:(id<MTLDevice>)device;
@end

@interface WebMetalBuffer : _MTLResource <MTLBuffer, MTLResourceSPI>
@end

@interface WebMetalHeap : _MTLHeap <MTLHeapSPI>
// IOGPUMetalHeap's sub-resource bookkeeping, which the heap's resources call.
- (void)unpinMemoryAtOffset:(uint64_t)offset withLength:(uint64_t)length;
- (void)deallocHeapSubResource;
@end

// Metal.framework's failure report (Metal`MTLReportFailure on 21H461), which the driver calls for misuse it detects.
// Its effect (a log line, or an assertion when Metal's assertions are enabled) is Metal.framework's.
extern "C" void MTLReportFailure(uint64_t type, const char *function, int line, NSString *format, ...);

namespace webmetal {

// MTLResourceOptions fields (MTLResource.h).
constexpr uint64_t kCPUCacheModeMask = 0xf;
constexpr unsigned kStorageModeShift = 4;
constexpr uint64_t kStorageModeMask = 0xf << kStorageModeShift;
constexpr unsigned kHazardTrackingModeShift = 8;
constexpr uint64_t kHazardTrackingModeMask = 0x3 << kHazardTrackingModeShift;

// The page size the driver rounds allocations to (vm_page_size, read by AGXBuffer's initializers).
uint64_t PageSize();

// The device's connection, and the device's live-allocation total (currentAllocatedSize).
Connection *DeviceConnection(WebMetalDevice *device);
void DeviceAddAllocation(WebMetalDevice *device, int64_t bytes);
int64_t DeviceAllocatedSize(WebMetalDevice *device);
// Live resources whose allocated size has no source yet (it depends on the AGX texture layout); while any lives, the
// device total is not answered.
void DeviceAddUnsized(WebMetalDevice *device, int64_t count);
int64_t DeviceUnsizedCount(WebMetalDevice *device);

// IOGPUMetalResource._res, as far as the plugin's answers depend on it. The comments name the field of
// _IOGPUMetalResource each member reproduces.
struct ResourceState {
    WebMetalDevice *__strong device = nil;  // device
    uint64_t handle = 0;                    // the wire handle (the kernel's resourceRef on the target)
    bool created = false;                   // the host holds the object (Release on deallocation)
    uint64_t options = 0;                   // options: the MTLResourceOptions the resource was created with
    uint64_t storageMode = 0;               // storageMode: options bits 4..7
    uint64_t cpuCacheMode = 0;              // cpuCacheMode: options bits 0..3
    uint64_t residentSize = 0;              // info.resourceSize: the kernel resource's resident size
    uint64_t protectionOptions = 0;         // the kernel resource's protection options
    int32_t responsibleProcess = 0;         // responsibleProcess
    uint64_t purgeableState = 2;            // purgeableState: the last state set (MTLPurgeableStateNonVolatile)
    bool purgeableAllowed = true;           // purgeableAllowed
    id<MTLHeap> __strong heap = nil;        // heap
    uint64_t heapOffset = 0;                // offset in the heap
    uint64_t heapLength = 0;                // length of the heap range
    bool pinned = true;                     // pinned: not aliasable
    IOSurfaceRef iosurface = nullptr;       // info.iosurface (retained)
    NSString *__strong label = nil;         // label
    os_unfair_lock labelLock = OS_UNFAIR_LOCK_INIT;

    ~ResourceState();
};

// Fills the state IOGPUMetalResource's initializers set (-[IOGPUMetalResource initWithDevice:options:args:argsSize:]):
// the options and the modes they pack, the responsible process, a non-volatile purgeable state, pinned.
void InitResourceState(ResourceState &state, WebMetalDevice *device, uint64_t options);

// IOGPUMetalResource's answers, for any resource class holding a ResourceState.
MTLHazardTrackingMode HazardTrackingMode(const ResourceState &state);
MTLResourceOptions ResourceOptions(const ResourceState &state);
NSUInteger HeapOffset(const ResourceState &state);
NSString *RetainedLabel(ResourceState &state);
void SetLabel(id self, SEL _cmd, ResourceState &state, NSString *label);
MTLPurgeableState SetPurgeableState(id self, SEL _cmd, ResourceState &state, MTLPurgeableState purgeableState);
void MakeAliasable(ResourceState &state);
BOOL DoesAlias(const ResourceState &a, const ResourceState &b);
int SetOwnerWithIdentity(id self, SEL _cmd, ResourceState &state, uint32_t identity);
// Sends Release for a created object; failures are reported only.
void ReleaseObject(ResourceState &state, const char *what);

// The ResourceState of any plugin buffer or texture, or null for another object.
ResourceState *StateOf(id resource);

}  // namespace webmetal
