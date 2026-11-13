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

// -[IOGPUMetalResource isComplete] (0x1fcb39934) and -waitUntilComplete (0x1fcb39964) test and wait on event 0 of the
// kernel resource (IOGPUResourceTestEvent and IOGPUResourceFinishEvent with _res.resourceRef and 0; the wait enters the
// kernel only while the test fails). The kernel clears both events of a resource when it makes it (IOGPUResource's
// initializer, kernelcache 0xfffffff005bc3624, through 0xfffffff005bc3694). A kernel resource made untracked (options
// & 0x300 == 0x100) carries creation flag 0x1000 (-[IOGPUMetalResource initWithDevice:options:args:argsSize:]
// 0x1fcb2bc1c), which IOGPUResource's argument initializer (0xfffffff005bc4b88) keeps as +0x10 bit 17 with a second,
// private event pair at +0x58; IOGPUResource's stamp (0xfffffff005bc4e68) then stamps the private pair instead of event
// 0 unless its flags carry bit 1, and every call of it in the A10X driver (AGXG9G) passes 0 or 1. Event 0 of an
// untracked kernel resource therefore stays clear: the test passes and the wait returns at once.
static const char kUntracedUse[] =
    "a command buffer that has not completed names the resource in a use whose listing in the A10X driver's IOGPU resource "
    "list, which decides whether the kernel stamps event 0 (AGXAllocationList2 0xfffffff005bf1914, IOGPUResource's stamp "
    "0xfffffff005bc4e68, kernelcache), is not traced: the draw-time vertex and fragment binds the driver lists by masks of "
    "the bound pipeline (0x21c96ad3c), which come from its compiler (packages G4 and G5), and the uses not read yet";
static const char kDeferredStamp[] =
    "an encoder that lists the resource follows an event wait that has not been passed (in its command buffer, or an "
    "earlier one of its queue): on the Apple M4 the kernel processes, and stamps, such an encoder only after the wait "
    "(spi.resource-complete-per-encoder), while the A10X's IOGPUCommandQueue::processKernelCommand (kernelcache "
    "0xfffffff005bd8a90 to 0xfffffff005bd8bb4) turns the wait into an event that later work depends on, and whether it "
    "stamps the later encoders at submission is not established";
bool CheckEventComplete(id self, SEL _cmd, uint64_t key, const ResourceState &kernel, bool iosurface, bool wait)
{
    if (kernel.storageMode == MTLStorageModeMemoryless)
        WMLoudMethod(self, _cmd, "a memoryless resource has no kernel resource, and IOGPU passes its null resourceRef to "
                                 "IOGPUResourceTestEvent and IOGPUResourceFinishEvent, which dereference it");
    if (iosurface)
        WMLoudMethod(self, _cmd, "the kernel resource of an IOSurface texture (type 0x82) has its submissions stamp the "
                                 "surface's shared resource (IOGPUResource's stamp, kernelcache 0xfffffff005bc4e94); "
                                 "what its own event 0 holds is not read yet");
    if ((kernel.options & kHazardTrackingModeMask) ==
        (uint64_t(MTLHazardTrackingModeUntracked) << kHazardTrackingModeShift))
        return true;
    // A tracked kernel resource: event 0 holds the stamps of the encoders that list it, each complete once the host
    // reports that encoder's work done (EncoderCompleted) or its command buffer completed. A pending traced stamp
    // decides the answer; without one, a pending use whose stamp is not established leaves it open.
    const PendingStamps pending = wait ? WaitStamps(key) : QueryStamps(key);
    if (pending.stamped)
        return false;
    if (pending.untraced)
        WMLoudMethod(self, _cmd, kUntracedUse);
    if (pending.deferred)
        WMLoudMethod(self, _cmd, kDeferredStamp);
    return true;
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
// unpinMemoryAtOffset:withLength:) and becomes aliasable; any other resource is unchanged. The host learns of it with
// MakeAliasable.
void MakeAliasable(id self, SEL _cmd, ResourceState &state, bool reportOnly)
{
    if (!state.heap || !state.pinned)
        return;
    [(WebMetalHeap *)state.heap unpinMemoryAtOffset:state.heapOffset withLength:state.heapLength];
    state.pinned = false;
    if (!state.created)
        return;
    wire::MakeAliasable request;
    request.resource = state.handle;
    if (reportOnly) {
        DeviceConnection(state.device)->CallReportOnly<wire::MakeAliasable, wire::MakeAliasableReply>(
            "-[IOGPUMetalResource makeAliasable] from -dealloc", request);
        return;
    }
    wire::Reply<wire::MakeAliasableReply> reply;
    DeviceConnection(state.device)->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, "the host answered MakeAliasable with a Metal error, which Metal does not define");
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
