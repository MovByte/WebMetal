// IOGPUMetalResource's state and answers, shared by the plugin's buffers and textures (WebMetalResources.h).

#import "WebMetalResources.h"

#include <mach/mach.h>
#include <unistd.h>

namespace webmetal {

uint64_t PageSize()
{
    return vm_page_size;
}

ResourceState::~ResourceState()
{
    if (iosurface)
        CFRelease(iosurface);
}

// -[IOGPUMetalResource initWithDevice:options:args:argsSize:] (IOGPU 0x1fcb2bb20): options at _res.options, the storage
// mode (bits 4..7) and CPU cache mode (bits 0..3) beside it, getpid() as the responsible process, purgeable state 2
// (non-volatile) with purgeability allowed, no heap, pinned. A memoryless resource has no kernel resource and is not
// purgeable.
void InitResourceState(ResourceState &state, WebMetalDevice *device, uint64_t options)
{
    state.device = device;
    state.options = options;
    state.storageMode = (options >> kStorageModeShift) & 0xf;
    state.cpuCacheMode = options & kCPUCacheModeMask;
    state.responsibleProcess = getpid();
    state.purgeableState = MTLPurgeableStateNonVolatile;
    state.purgeableAllowed = state.storageMode != MTLStorageModeMemoryless;
    state.pinned = true;
}

// -[IOGPUMetalResource hazardTrackingMode] (0x1fcb26124): the options' mode; for the default mode, the heap's mode, and
// without a heap MTLHazardTrackingModeTracked.
MTLHazardTrackingMode HazardTrackingMode(const ResourceState &state)
{
    const uint64_t mode = (state.options & kHazardTrackingModeMask) >> kHazardTrackingModeShift;
    if (mode)
        return static_cast<MTLHazardTrackingMode>(mode);
    if (state.heap)
        return state.heap.hazardTrackingMode;
    return MTLHazardTrackingModeTracked;
}

// -[IOGPUMetalResource resourceOptions] (0x1fcb26168): the CPU cache mode, the storage mode and the resolved hazard
// tracking mode, packed.
MTLResourceOptions ResourceOptions(const ResourceState &state)
{
    return (state.cpuCacheMode & kCPUCacheModeMask) | ((state.storageMode << kStorageModeShift) & kStorageModeMask) |
           ((static_cast<uint64_t>(HazardTrackingMode(state)) << kHazardTrackingModeShift) & kHazardTrackingModeMask);
}

// -[IOGPUMetalResource heapOffset] (0x1fcb30d98): the offset in the heap for a placement heap's resource
// (MTLHeapTypePlacement, 1); 0 otherwise.
NSUInteger HeapOffset(const ResourceState &state)
{
    if (state.heap && state.heap.type == MTLHeapTypePlacement)
        return state.heapOffset;
    return 0;
}

// -[IOGPUMetalResource retainedLabel] (0x1fcb2e7d8): the label under the label lock.
NSString *RetainedLabel(ResourceState &state)
{
    os_unfair_lock_lock(&state.labelLock);
    NSString *label = state.label;
    os_unfair_lock_unlock(&state.labelLock);
    return label;
}

// -[IOGPUMetalResource setLabel:] (0x1fcb2bea4): nothing changes when the same object is set again; otherwise a copy
// replaces the label under the lock, and the kernel resource takes the new label (the host, through SetLabel).
void SetLabel(id self, SEL _cmd, ResourceState &state, NSString *label)
{
    if (state.label == label)
        return;
    NSString *copy = [label copy];
    os_unfair_lock_lock(&state.labelLock);
    state.label = copy;
    os_unfair_lock_unlock(&state.labelLock);
    if (!state.created)
        return;
    wire::SetLabel request;
    request.object = state.handle;
    if (!wire::strFromNSString(copy, request.label))
        WMLoudMethod(self, _cmd, "the label has no UTF-8 form, which the wire format carries");
    wire::Reply<wire::SetLabelReply> reply;
    DeviceConnection(state.device)->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, "the host answered SetLabel with a Metal error, which Metal does not define");
}

// -[IOGPUMetalResource setPurgeableState:] (0x1fcb29fac). A resource that may not be purgeable answers
// MTLPurgeableStateNonVolatile. Otherwise any state but keep-current (1) and 0x100 is remembered, and the kernel
// resource takes the state the driver maps it to: 1..4 unchanged, 0x100 to 5, anything else to 1 (keep current). The
// answer is the kernel's previous state when it is 2..4 (non-volatile, volatile, empty), keep-current otherwise.
// (The driver takes another path for an IOSurface-backed resource; the plugin's IOSurface textures refuse it below.)
MTLPurgeableState SetPurgeableState(id self, SEL _cmd, ResourceState &state, MTLPurgeableState purgeableState)
{
    if (!state.purgeableAllowed)
        return MTLPurgeableStateNonVolatile;
    const uint64_t value = purgeableState;
    if (value != MTLPurgeableStateKeepCurrent && value != 0x100)
        state.purgeableState = value;
    if (state.iosurface && value != 0x100)
        WMLoudMethod(self, _cmd, "21H461 IOGPU sets an IOSurface-backed resource's state with IOSurfaceSetPurgeable; "
                                 "the plugin has no IOSurface purgeability yet");
    uint64_t kernelState;
    switch (value) {
    case 1: case 2: case 3: case 4: kernelState = value; break;
    case 0x100: kernelState = 5; break;
    default: kernelState = 1; break;
    }
    wire::SetPurgeableState request;
    request.object = state.handle;
    request.state = kernelState;
    wire::Reply<wire::SetPurgeableStateReply> reply;
    DeviceConnection(state.device)->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, "the host answered SetPurgeableState with a Metal error, which Metal does not define");
    const uint64_t previous = reply.value.previousState;
    return previous >= 2 && previous <= 4 ? static_cast<MTLPurgeableState>(previous) : MTLPurgeableStateKeepCurrent;
}

// -[IOGPUMetalResource makeAliasable] (0x1fcb2cd24): a pinned heap resource unpins its heap range (the heap's
// unpinMemoryAtOffset:withLength:) and becomes aliasable; any other resource is unchanged.
void MakeAliasable(ResourceState &state)
{
    if (!state.heap || !state.pinned)
        return;
    [(id)state.heap unpinMemoryAtOffset:state.heapOffset withLength:state.heapLength];
    state.pinned = false;
}

// -[IOGPUMetalResource doesAliasResource:] (0x1fcb39f30): both resources come from the same heap, both ranges are
// non-empty, and the ranges overlap.
BOOL DoesAlias(const ResourceState &a, const ResourceState &b)
{
    if (!a.heap || a.heap != b.heap || !a.heapLength || !b.heapLength)
        return NO;
    return b.heapOffset + b.heapLength > a.heapOffset && a.heapOffset + a.heapLength > b.heapOffset;
}

// -[IOGPUMetalResource setOwnerWithIdentity:] (0x1fcb30d70): the kernel resource's IOGPUResourceSetOwnerIdentity, or
// kIOReturnBadArgument (0xe00002c2) without a kernel resource.
int SetOwnerWithIdentity(id self, SEL _cmd, ResourceState &state, uint32_t identity)
{
    if (!state.created)
        return static_cast<int>(0xe00002c2);
    wire::SetOwnerWithIdentity request;
    request.resource = state.handle;
    request.taskIdToken = identity;
    wire::Reply<wire::SetOwnerWithIdentityReply> reply;
    DeviceConnection(state.device)->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, "the host answered SetOwnerWithIdentity with a Metal error, which Metal does not define");
    return reply.value.result;
}

void ReleaseObject(ResourceState &state, const char *what)
{
    if (!state.created)
        return;
    state.created = false;
    wire::Release request;
    request.object = state.handle;
    DeviceConnection(state.device)->CallReportOnly<wire::Release, wire::ReleaseReply>(what, request);
}

}  // namespace webmetal
