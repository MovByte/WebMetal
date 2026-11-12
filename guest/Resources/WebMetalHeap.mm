// Heaps: WebMetalHeap stands in for AGXA10FamilyHeap (AGXA10FamilyHeap -> IOGPUMetalHeap -> _MTLHeap on 21H461).
//
// An IOGPUMetalHeap is one kernel resource (the backing resource, here the host's heap object) and, for an automatic
// heap, Metal.framework's range allocator over it; sub-allocation happens in the process, with the same allocator
// (MTLRangeAllocator*, exported by Metal.framework) called with the arguments IOGPU passes. A shared heap's bytes are
// guest memory the heap allocates; its buffers' contents lie inside them.

#import "WebMetalBuffer.h"
#import "WebMetalHeapPrivate.h"

#import <objc/runtime.h>

#include <mach/mach.h>

// The class adopts MTLHeap and MTLHeapSPI as AGXA10FamilyHeap does and implements only what the 21H461 driver layers
// implement; the protocols' other methods are Metal.framework's or LOUD stubs (guest/Plugin/WebMetalStubs.mm).
#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"

extern "C" {
NSString *MTLCPUCacheModeString(MTLCPUCacheMode mode);
NSString *MTLStorageModeString(MTLStorageMode mode);
}

@interface _MTLHeap (WebMetalFramework)
- (instancetype)initWithType:(MTLHeapType)type options:(MTLResourceOptions)options;
- (void)setLabel:(NSString *)label;
@end

@interface WebMetalHeap () {
@public
    webmetal::ResourceState _resource;  // IOGPUMetalHeap._resource: the backing kernel resource
    uint64_t _size;                     // IOGPUMetalHeap._size
    MTLRangeAllocator _allocator;       // IOGPUMetalHeap._allocator (automatic heaps)
    pthread_mutex_t _mutex;             // IOGPUMetalHeap._mutex
    void *_contents;                    // the shared heap's bytes
    vm_size_t _ownedSize;
}
@end

namespace webmetal {

namespace {

const char kSubResource[] = "-[IOGPUMetalHeap newSubResourceWithLength:alignment:options:offset:]";
const char kSubResourceAt[] = "-[IOGPUMetalHeap newSubResourceAtOffset:withLength:alignment:options:]";

// The mode checks IOGPU's sub-resource methods report with MTLReportFailure and then pass over: a CPU cache mode other
// than the heap's, and a storage mode other than the heap's (the automatic form lets memoryless through).
void reportModes(const char *function, int cacheLine, int storageLine, uint64_t options, uint64_t heapOptions,
                 bool memorylessAllowed)
{
    const uint64_t cache = options & kCPUCacheModeMask, heapCache = heapOptions & kCPUCacheModeMask;
    if (cache != heapCache)
        MTLReportFailure(0, function, cacheLine, @"The requested CPU cache mode (%@) does not match the heap's mode (%@)",
                         MTLCPUCacheModeString(static_cast<MTLCPUCacheMode>(cache)),
                         MTLCPUCacheModeString(static_cast<MTLCPUCacheMode>(heapCache)));
    const uint64_t storage = (options >> kStorageModeShift) & 0xf, heapStorage = (heapOptions >> kStorageModeShift) & 0xf;
    if (!(memorylessAllowed && storage == MTLStorageModeMemoryless) && storage != heapStorage)
        MTLReportFailure(0, function, storageLine,
                         @"The requested storage mode (%@) is not compatible with the heap's mode (%@)",
                         MTLStorageModeString(static_cast<MTLStorageMode>(storage)),
                         MTLStorageModeString(static_cast<MTLStorageMode>(heapStorage)));
}

}  // namespace

// -[AGXA10FamilyHeap initWithDevice:descriptor:] (AGXMetalA10 0x21c919e5c), then
// -[IOGPUMetalHeap initWithDevice:size:options:args:argsSize:desc:] (IOGPU 0x1fcb2fc30).
WebMetalHeap *NewHeap(WebMetalDevice *device, SEL _cmd, MTLHeapDescriptor *descriptor)
{
    if (![(id<WMDescriptorValidation>)descriptor validateWithDevice:device])
        return nil;
    // The size rounds up to the page size. The options: write-combined when the descriptor's CPU cache mode is not the
    // default, private when its storage mode is private (any other storage mode makes a shared heap), the
    // descriptor's hazard-tracking mode unless the device's creation flags carry bit 4 (then tracked), and the
    // resource-option bits 0x200000 and 0x400000 of the descriptor.
    const uint64_t page = PageSize();
    const uint64_t requested = descriptor.size;
    const uint64_t size = (requested + page - 1) / page * page;
    const uint64_t resourceOptions = descriptor.resourceOptions;
    uint64_t options = (descriptor.cpuCacheMode != 0 ? 1 : 0) |
                       (descriptor.storageMode == MTLStorageModePrivate ? 0x20 : 0) |
                       ((DeviceCreationFlags(device) & 0x10) ? 0x200 : (descriptor.hazardTrackingMode << 8) & 0x300) |
                       (resourceOptions & 0x200000) | (resourceOptions & 0x400000);
    // IOGPU: a heap without a hazard-tracking mode is untracked.
    if (!(options & kHazardTrackingModeMask))
        options = (options & ~kHazardTrackingModeMask) | 0x100;
    const MTLHeapType type = descriptor.type;
    WebMetalHeap *heap = [[WebMetalHeap alloc] initWithType:type options:options];
    if (!heap)
        return nil;
    InitResourceState(heap->_resource, device, options);
    heap->_resource.protectionOptions = Get<uint64_t>(device, _cmd, descriptor, "protectionOptions");
    heap->_resource.residentSize = size;
    heap->_size = size;
    pthread_mutex_init(&heap->_mutex, nullptr);
    if (type == MTLHeapTypeAutomatic)
        MTLRangeAllocatorInit(&heap->_allocator, size - 1, 0, 0x80);
    Connection *connection = DeviceConnection(device);
    heap->_resource.handle = connection->NewHandle();
    if (!(options & 0x20)) {
        vm_address_t address = 0;
        if (vm_allocate(mach_task_self(), &address, static_cast<vm_size_t>(size ? size : page), VM_FLAGS_ANYWHERE) !=
            KERN_SUCCESS)
            return nil;
        heap->_contents = reinterpret_cast<void *>(address);
        heap->_ownedSize = static_cast<vm_size_t>(size ? size : page);
    }
    wire::CreateHeap request;
    request.heap = heap->_resource.handle;
    request.descriptor.cpuCacheMode = descriptor.cpuCacheMode;
    request.descriptor.hazardTrackingMode = descriptor.hazardTrackingMode;
    request.descriptor.memoryPoolId = Get<uint64_t>(device, _cmd, descriptor, "memoryPoolId");
    request.descriptor.pinnedGpuAddress = Get<uint64_t>(device, _cmd, descriptor, "pinnedGPUAddress");
    request.descriptor.protectionOptions = Get<uint64_t>(device, _cmd, descriptor, "protectionOptions");
    request.descriptor.resourceOptions = resourceOptions;
    request.descriptor.size = requested;
    request.descriptor.sparsePageAccessCountersEnabled = Get<BOOL>(device, _cmd, descriptor, "sparsePageAccessCountersEnabled");
    request.descriptor.sparsePageSize = descriptor.sparsePageSize;
    request.descriptor.storageMode = descriptor.storageMode;
    request.descriptor.type = type;
    wire::Reply<wire::CreateHeapReply> reply;
    connection->Call(device, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(device, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
    heap->_resource.created = true;
    DeviceAddAllocation(device, static_cast<int64_t>(size));
    return heap;
}

uint64_t HeapHandle(WebMetalHeap *heap)
{
    return heap->_resource.handle;
}

}  // namespace webmetal

@implementation WebMetalHeap

- (void)dealloc
{
    using namespace webmetal;
    DeviceAddAllocation(_resource.device, -static_cast<int64_t>(_size));
    ReleaseObject(_resource, "-[WebMetalHeap dealloc]");
    if (self.type == MTLHeapTypeAutomatic)
        MTLRangeAllocatorDestroy(&_allocator);
    pthread_mutex_destroy(&_mutex);
    if (_ownedSize)
        vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(_contents), _ownedSize);
}

// -[IOGPUMetalHeap device] (0x1fcb2e0b0).
- (id<MTLDevice>)device
{
    return _resource.device;
}

// -[IOGPUMetalHeap size] (0x1fcb2e1a8).
- (NSUInteger)size
{
    return _size;
}

// -[IOGPUMetalHeap usedSize] (0x1fcb36fc8): 0 for a placement heap; otherwise the size less the allocator's free space.
- (NSUInteger)usedSize
{
    if (self.type != MTLHeapTypeAutomatic)
        return 0;
    pthread_mutex_lock(&_mutex);
    const uint64_t used = _size - MTLRangeAllocatorGetFreeSize(&_allocator);
    pthread_mutex_unlock(&_mutex);
    return used;
}

// -[IOGPUMetalHeap currentAllocatedSize] (0x1fcb37148): the backing resource's allocated size, its resident size.
- (NSUInteger)currentAllocatedSize
{
    return _resource.residentSize;
}

// -[IOGPUMetalHeap maxAvailableSizeWithAlignment:] (0x1fcb2638c): the alignment must be a power of two (the driver
// traps otherwise); 0 for a placement heap; otherwise the allocator's largest free range at that alignment.
- (NSUInteger)maxAvailableSizeWithAlignment:(NSUInteger)alignment
{
    if (alignment & (alignment - 1))
        WMLoudMethod(self, _cmd, "21H461 IOGPU traps on an alignment that is not a power of two");
    if (self.type != MTLHeapTypeAutomatic)
        return 0;
    pthread_mutex_lock(&_mutex);
    const uint64_t available = MTLRangeAllocatorGetMaxFreeSize(&_allocator, static_cast<uint32_t>(alignment));
    pthread_mutex_unlock(&_mutex);
    return available;
}

// -[IOGPUMetalHeap unpinMemoryAtOffset:withLength:] (0x1fcb2d570): an automatic heap's range goes back to the
// allocator (the driver traps for any other heap).
- (void)unpinMemoryAtOffset:(uint64_t)offset withLength:(uint64_t)length
{
    if (self.type != MTLHeapTypeAutomatic)
        WMLoudMethod(self, _cmd, "21H461 IOGPU traps when a resource of a heap that is not automatic is unpinned");
    pthread_mutex_lock(&_mutex);
    MTLRangeAllocatorDeallocate(&_allocator, offset, length);
    pthread_mutex_unlock(&_mutex);
}

// -[IOGPUMetalHeap deallocHeapSubResource] (0x1fcb2db08) releases, under the heap's lock, the reference a sub-resource
// took on the backing resource. A heap resource here holds the heap itself, which owns the backing.
- (void)deallocHeapSubResource
{
}

// -[IOGPUMetalHeap setPurgeableState:] (0x1fcb2632c): the backing resource's, under the heap's lock.
- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state
{
    pthread_mutex_lock(&_mutex);
    MTLPurgeableState previous;
    @try {
        previous = webmetal::SetPurgeableState(self, _cmd, _resource, state);
    } @finally {
        pthread_mutex_unlock(&_mutex);
    }
    return previous;
}

// -[IOGPUMetalHeap protectionOptions] (0x1fcb2e1c8): the backing resource's, the descriptor's.
- (uint64_t)protectionOptions
{
    return _resource.protectionOptions;
}

// -[IOGPUMetalHeap setLabel:] (0x1fcb37044): the backing resource is labelled (the host's heap object, through
// SetLabel with the label given), then _MTLHeap keeps the label.
- (void)setLabel:(NSString *)label
{
    if (_resource.created) {
        webmetal::wire::SetLabel request;
        request.object = _resource.handle;
        if (!webmetal::wire::strFromNSString(label, request.label))
            WMLoudMethod(self, _cmd, "the label has no UTF-8 form, which the wire format carries");
        webmetal::wire::Reply<webmetal::wire::SetLabelReply> reply;
        webmetal::DeviceConnection(_resource.device)->Call(self, _cmd, request, &reply);
        if (!reply.ok)
            WMLoudMethod(self, _cmd, "the host answered SetLabel with a Metal error, which Metal does not define");
    }
    [super setLabel:label];
}

// -[AGXA10FamilyHeap newBufferWithLength:options:] (AGXMetalA10 0x21c91a008): -[AGXBuffer initWithHeap:length:
// alignment:options:] with alignment 1, which needs an automatic heap and takes a range from
// -[IOGPUMetalHeap newSubResourceWithLength:alignment:options:offset:] at an alignment of at least 256.
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
    if (self.type != MTLHeapTypeAutomatic)
        return nil;
    if (length > _size)
        return nil;
    webmetal::reportModes(webmetal::kSubResource, 0xb7, 0xc0, options, _resource.options, true);
    uint64_t offset = 0;
    pthread_mutex_lock(&_mutex);
    const bool granted = MTLRangeAllocatorAllocate(&_allocator, length, &offset, 0x100);
    pthread_mutex_unlock(&_mutex);
    if (!granted)
        return nil;
    void *contents = _contents ? static_cast<uint8_t *>(_contents) + offset : nullptr;
    return webmetal::NewHeapBuffer(self, _cmd, length, options, _resource.options, offset, length, contents, false);
}

// -[AGXA10FamilyHeap newBufferWithLength:options:atOffset:]: -[AGXBuffer initWithHeap:length:alignment:options:atOffset:]
// (AGXMetalA10 0x21c8a47e8) with alignment 1, which needs a placement heap and the range
// -[IOGPUMetalHeap newSubResourceAtOffset:withLength:alignment:options:] (IOGPU 0x1fcb3099c) grants: inside the heap,
// with the offset a multiple of the alignment (at least 256).
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options atOffset:(NSUInteger)offset
{
    if (self.type != MTLHeapTypePlacement)
        return nil;
    webmetal::reportModes(webmetal::kSubResourceAt, 0x106, 0x10b, options, _resource.options, false);
    const uint64_t alignment = 0x100;
    uint64_t end = 0;
    if (__builtin_add_overflow(static_cast<uint64_t>(offset), static_cast<uint64_t>(length), &end) || end > _size ||
        ((alignment | offset) & (alignment - 1)))
        return nil;
    void *contents = _contents ? static_cast<uint8_t *>(_contents) + offset : nullptr;
    return webmetal::NewHeapBuffer(self, _cmd, length, options, _resource.options, offset, length, contents, true);
}

// -[IOGPUMetalHeap gpuAddress] (0x1fcb37158): the backing resource's GPU address.
- (uint64_t)gpuAddress
{
    WMLoudMethod(self, _cmd, "the wire protocol defines no GPU address space for heaps and buffers");
}

@end
