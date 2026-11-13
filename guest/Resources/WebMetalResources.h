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

#import <objc/message.h>

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

// The device's sub-allocator (IOGPUMetalSuballocatorAllocate) and the com.apple.Metal DisableDriverSuballocation
// preference that turns driver sub-allocation off (WebMetalBuffer.mm).
bool SuballocatorServes(uint64_t length, uint64_t options);
bool SuballocationDisabled();

// _res.info.resourceSize of a buffer or texture the sub-allocator places. Each has a kernel resource of its own, made
// with the primary buffer's as its parent (flag 0x800 and the primary's resource id: -[IOGPUMetalBuffer
// initWithPrimaryBuffer:...] IOGPU 0x1fcb2a1a8, -[IOGPUMetalTexture initWithBuffer:...isStrideTexture:NO] 0x1fcb2e300).
// IOGPUResource::newResourceWithParentResourceID (kernelcache 0xfffffff005bc42d8) gives it the parent's resident size
// (+0x78, stored at 0xfffffff005bc4468), which IOGPUDevice::new_resource returns as its resident size
// (0xfffffff005bd07b8 through 0xfffffff005bc3a8c). The plugin's sub-allocations take the ordinary path, whose primaries
// are dedicated buffers of 0x20000 bytes (IOGPU 0x1fcb2876c), a page multiple and so their resident size
// (IOGPUResource::newResourceWithOptions rounds the system memory size up to the page, 0xfffffff005bc3f94). The pinned
// path, whose primary has the caller's parent length, is reachable only through initializers the plugin answers LOUD.
constexpr uint64_t kSuballocatedResourceSize = 0x20000;

// The device's connection, and the device's live-allocation total (currentAllocatedSize).
Connection *DeviceConnection(WebMetalDevice *device);
void DeviceAddAllocation(WebMetalDevice *device, int64_t bytes);
int64_t DeviceAllocatedSize(WebMetalDevice *device);
// Live resources whose allocated size has no source yet (it depends on the AGX texture layout); while any lives, the
// device total is not answered.
void DeviceAddUnsized(WebMetalDevice *device, int64_t count);
int64_t DeviceUnsizedCount(WebMetalDevice *device);
// The AGX device's creation flags (-deviceCreationFlags).
uint64_t DeviceCreationFlags(WebMetalDevice *device);
// IOGPUMetalDevice._configBits: the IOGPU device configuration's bits (WebMetalIOGPUDevice.h).
uint32_t DeviceConfigBits(WebMetalDevice *device);
// The AGX device's depth-stencil state cache (WebMetalDepthStencilCache.h).
struct DepthStencilCache;
DepthStencilCache &DeviceDepthStencilStates(WebMetalDevice *device);

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

// -[IOGPUMetalResource isComplete] and -waitUntilComplete for a resource whose kernel resource has the state `kernel`
// (the parent's, buffer's or heap's for a resource that shares one): returns for a kernel resource made untracked,
// LOUD otherwise (WebMetalResource.mm).
void CheckEventComplete(id self, SEL _cmd, const ResourceState &kernel, bool iosurface);
MTLResourceOptions ResourceOptions(const ResourceState &state);
NSUInteger HeapOffset(const ResourceState &state);
NSString *RetainedLabel(ResourceState &state);
void SetLabel(id self, SEL _cmd, ResourceState &state, NSString *label);
MTLPurgeableState SetPurgeableState(id self, SEL _cmd, ResourceState &state, MTLPurgeableState purgeableState);
// `reportOnly` for a call from -dealloc, where a host failure is reported without raising.
void MakeAliasable(id self, SEL _cmd, ResourceState &state, bool reportOnly);
BOOL DoesAlias(const ResourceState &a, const ResourceState &b);
int SetOwnerWithIdentity(id self, SEL _cmd, ResourceState &state, uint32_t identity);
// Sends Release for a created object; failures are reported only.
void ReleaseObject(ResourceState &state, const char *what);

// A getter of a Metal.framework descriptor, which may be SPI. The plugin reads every field the wire carries; a running
// framework that lacks the getter cannot be served.
template <typename T> T Get(id self, SEL _cmd, id object, const char *name)
{
    SEL selector = sel_registerName(name);
    if (![object respondsToSelector:selector])
        WMLoudMethod(self, _cmd, (std::string("the descriptor has no -") + name + ", which the wire format carries").c_str());
    return ((T(*)(id, SEL))objc_msgSend)(object, selector);
}

// The ResourceState of any plugin buffer or texture, or null for another object.
ResourceState *StateOf(id resource);

}  // namespace webmetal
