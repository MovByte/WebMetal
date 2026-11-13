// Buffers: WebMetalBuffer stands in for AGXA10FamilyBuffer (AGXA10FamilyBuffer -> AGXBuffer -> IOGPUMetalBuffer ->
// IOGPUMetalResource -> _MTLResource on 21H461).
//
// Storage. A shared buffer's bytes are guest memory the plugin allocates and the host executor keeps coherent with its
// GPU copy (the shadow/diff protocol); a private buffer has no CPU bytes; a no-copy buffer's bytes are the caller's.
// On the target the kernel maps a shared resource into the process and reports its resident size; here the WebMetal
// service holds what the driver asked the kernel for, so a dedicated allocation's resident size is the size the driver
// requests (sysMemSize: the length rounded up to the page size, or the length for caller memory).

#import "WebMetalBuffer.h"
#import "WebMetalTexture.h"
#import "WebMetalHeapPrivate.h"

#include <mach/mach.h>

#include <cstring>

// The class adopts MTLBuffer and MTLResourceSPI as AGXA10FamilyBuffer does and implements only what the 21H461 driver
// layers implement; the protocols' other methods are LOUD stubs added at run time (guest/Plugin/WebMetalStubs.mm).
#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"

extern "C" {
NSString *MTLCPUCacheModeString(MTLCPUCacheMode mode);
NSString *MTLStorageModeString(MTLStorageMode mode);
NSString *MTLHazardTrackingModeString(MTLHazardTrackingMode mode);
NSString *MTLResourceOptionsString(MTLResourceOptions options);
NSString *MTLPurgeableStateString(MTLPurgeableState state);
}

namespace webmetal {

// IOGPUMetalSuballocatorAllocate (IOGPU 0x1fcb2805c) serves lengths 1 to 0x20000 with storage mode at most private,
// CPU cache mode at most write-combined and hazard tracking untracked or default (tracked counts as default).
bool SuballocatorServes(uint64_t length, uint64_t options)
{
    uint64_t hazard = (options & webmetal::kHazardTrackingModeMask) >> webmetal::kHazardTrackingModeShift;
    if (hazard == MTLHazardTrackingModeTracked)
        hazard = 0;
    const uint64_t storage = (options & webmetal::kStorageModeMask) >> webmetal::kStorageModeShift;
    const uint64_t cache = options & webmetal::kCPUCacheModeMask;
    return length != 0 && length <= 0x20000 && storage <= 2 && cache <= 1 && hazard <= 1;
}

// -[AGXA10FamilyDevice initWithAcceleratorPort:simultaneousInstances:] (AGXMetalA10 0x21c8d22bc) disables driver
// sub-allocation when the com.apple.Metal preference DisableDriverSuballocation is set to true.
bool SuballocationDisabled()
{
    static const bool disabled = [] {
        Boolean valid = false;
        const Boolean value =
            CFPreferencesGetAppBooleanValue(CFSTR("DisableDriverSuballocation"), CFSTR("com.apple.Metal"), &valid);
        return valid && value;
    }();
    return disabled;
}

}  // namespace webmetal

namespace {

// The function names the driver's failure reports carry.
const char kIOGPUBufferInit[] = "-[IOGPUMetalBuffer initWithDevice:pointer:length:alignment:options:sysMemSize:"
                                "gpuAddress:args:argsSize:deallocator:]";

// The checks -[IOGPUMetalBuffer initWithDevice:pointer:...] (IOGPU 0x1fcb2b3c8) reports through MTLReportFailure and
// then continues past.
void reportInitFailures(MTLResourceOptions options, bool callerMemory)
{
    const uint64_t storageBits = options & webmetal::kStorageModeMask;
    if (callerMemory && storageBits == 0x20)
        MTLReportFailure(0, kIOGPUBufferInit, 0x82, @"storageModePrivate incompatible with ...WithBytes variant of newBuffer");
    if (callerMemory && storageBits == 0x30)
        MTLReportFailure(0, kIOGPUBufferInit, 0x84,
                         @"storageModeMemoryless incompatible with ...WithBytes variant of newBuffer");
    const uint64_t cache = options & webmetal::kCPUCacheModeMask;
    if (cache >= 2)
        MTLReportFailure(0, kIOGPUBufferInit, 0x9e, @"Invalid cacheMode %u", static_cast<unsigned>(cache));
    if (options & 0xd0)
        MTLReportFailure(0, kIOGPUBufferInit, 0xa8, @"Invalid storageMode %u",
                         static_cast<unsigned>((options >> webmetal::kStorageModeShift) & 0xf));
}

}  // namespace

@interface WebMetalBuffer () {
@public
    webmetal::ResourceState _state;
    NSUInteger _length;
    void *_contents;
    vm_size_t _ownedSize;  // guest memory the buffer allocated for its contents
    int64_t _accounted;    // what the buffer adds to the device's currentAllocatedSize
    bool _suballocated;    // the driver sub-allocates it: its allocated size is its length
    void (^_deallocator)(void *pointer, NSUInteger length);
}
@end

namespace webmetal {

ResourceState *TextureState(WebMetalTexture *texture);

ResourceState *StateOf(id resource)
{
    if ([resource isKindOfClass:[WebMetalBuffer class]])
        return &static_cast<WebMetalBuffer *>(resource)->_state;
    if ([resource isKindOfClass:[WebMetalTexture class]])
        return TextureState(static_cast<WebMetalTexture *>(resource));
    return nullptr;
}

uint64_t BufferHandle(WebMetalBuffer *buffer)
{
    return buffer->_state.handle;
}

void *BufferContents(WebMetalBuffer *buffer)
{
    return buffer->_contents;
}

// The steps the driver's dedicated and sub-allocated paths share once the kernel resource exists: the buffer joins the
// device's allocation total.
static WebMetalBuffer *finishCreation(WebMetalBuffer *buffer)
{
    buffer->_state.created = buffer->_state.storageMode != MTLStorageModeMemoryless;
    buffer->_accounted = static_cast<int64_t>(buffer.allocatedSize);
    DeviceAddAllocation(buffer->_state.device, buffer->_accounted);
    return buffer;
}

// AGXBuffer(Internal) -initWithDevice:length:alignment:options:isSuballocDisabled:resourceInArgs:pinnedGPULocation:
// (AGXMetalA10 0x21c8a4900) with alignment 1 and no pinned location, as -[AGXA10FamilyDevice newBufferWithLength:options:]
// (0x21c8d5e48) and -newBufferWithBytes:length:options: (0x21c8d5f0c) call it. The size requested from the kernel is the
// length rounded up to the page size; a length whose rounding overflows fails. A length below the page size is
// sub-allocated unless the preference disables it or the sub-allocator refuses it; otherwise the buffer is a dedicated
// kernel resource. The kernel resource cannot be made for a zero length or one above the device's maxBufferLength (the
// IOGPUMetalSuballocatorAllocate path refuses zero, the kernel refuses both): nil, as Metal answers.
WebMetalBuffer *NewDeviceBuffer(WebMetalDevice *device, SEL _cmd, NSUInteger length, MTLResourceOptions options,
                                const void *bytes)
{
    const uint64_t page = PageSize();
    const uint64_t sysMemSize = (static_cast<uint64_t>(length) + page - 1) & ~(page - 1);
    if (sysMemSize < length)
        return nil;
    const bool suballocated = length < page && !SuballocationDisabled() && SuballocatorServes(length, options);
    if (!suballocated)
        reportInitFailures(options, false);
    const uint64_t storage = (options & kStorageModeMask) >> kStorageModeShift;
    // A memoryless resource has no kernel resource (-[IOGPUMetalResource initWithDevice:options:args:argsSize:]).
    const bool memoryless = storage == MTLStorageModeMemoryless;
    if (!memoryless && (length == 0 || length > [device maxBufferLength]))
        return nil;
    WebMetalBuffer *buffer = [WebMetalBuffer new];
    InitResourceState(buffer->_state, device, options);
    buffer->_length = length;
    buffer->_suballocated = suballocated;
    buffer->_state.residentSize = suballocated ? kSuballocatedResourceSize : sysMemSize;
    Connection *connection = DeviceConnection(device);
    buffer->_state.handle = connection->NewHandle();
    if (memoryless)
        return finishCreation(buffer);
    if (storage != MTLStorageModePrivate) {
        vm_address_t address = 0;
        const vm_size_t size = static_cast<vm_size_t>(sysMemSize ? sysMemSize : page);
        if (vm_allocate(mach_task_self(), &address, size, VM_FLAGS_ANYWHERE) != KERN_SUCCESS)
            return nil;
        buffer->_contents = reinterpret_cast<void *>(address);
        buffer->_ownedSize = size;
    }
    wire::Reply<wire::CreateBufferReply> created;
    wire::Reply<wire::CreateBufferWithBytesReply> createdWithBytes;
    bool ok;
    if (bytes) {
        wire::CreateBufferWithBytes request;
        request.buffer = buffer->_state.handle;
        request.pointer = reinterpret_cast<uint64_t>(bytes);
        request.length = length;
        request.options = options;
        request.contents = reinterpret_cast<uint64_t>(buffer->_contents);
        connection->Call(device, _cmd, request, &createdWithBytes);
        ok = createdWithBytes.ok;
    } else {
        wire::CreateBuffer request;
        request.buffer = buffer->_state.handle;
        request.length = length;
        request.options = options;
        request.contents = reinterpret_cast<uint64_t>(buffer->_contents);
        connection->Call(device, _cmd, request, &created);
        ok = created.ok;
    }
    if (!ok)
        WMLoudMethod(device, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
    return finishCreation(buffer);
}

// -[AGXA10FamilyDevice newBufferWithBytesNoCopy:length:options:deallocator:] (AGXMetalA10 0x21c8d5ffc):
// -[AGXBuffer initWithDevice:bytes:length:alignment:options:deallocator:pinnedGPUAddress:] with alignment 1, which
// asks IOGPU for a kernel resource over the caller's memory of exactly `length` bytes and keeps the deallocator, run when
// the buffer is deallocated.
WebMetalBuffer *NewNoCopyBuffer(WebMetalDevice *device, SEL _cmd, void *pointer, NSUInteger length,
                                MTLResourceOptions options, void (^deallocator)(void *pointer, NSUInteger length))
{
    reportInitFailures(options, true);
    const uint64_t storage = (options & kStorageModeMask) >> kStorageModeShift;
    const bool memoryless = storage == MTLStorageModeMemoryless;
    if (!memoryless && (length == 0 || length > [device maxBufferLength]))
        return nil;
    WebMetalBuffer *buffer = [WebMetalBuffer new];
    InitResourceState(buffer->_state, device, options);
    buffer->_length = length;
    buffer->_contents = pointer;
    buffer->_state.residentSize = length;
    buffer->_deallocator = [deallocator copy];
    Connection *connection = DeviceConnection(device);
    buffer->_state.handle = connection->NewHandle();
    if (memoryless)
        return finishCreation(buffer);
    wire::CreateBufferNoCopy request;
    request.buffer = buffer->_state.handle;
    request.pointer = reinterpret_cast<uint64_t>(pointer);
    request.length = length;
    request.options = options;
    request.deallocator = deallocator != nil;
    wire::Reply<wire::CreateBufferNoCopyReply> reply;
    connection->Call(device, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(device, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
    return finishCreation(buffer);
}

// -[AGXBuffer initWithHeap:length:alignment:options:] and its atOffset: form, through -[AGXBuffer(Internal)
// initImplWithHeap:resource:length:atOffset:] (AGXMetalA10 0x21c8a4b04) and -[IOGPUMetalBuffer initWithHeap:resource:
// offset:length:] (IOGPU 0x1fcb2d810): the buffer takes the heap backing resource's options and modes and may not be
// purgeable (-[IOGPUMetalResource initWithResource:]), keeps the heap, the range and its length, and is pinned unless
// the heap is a placement heap. Its contents lie at the range's offset in the heap's.
WebMetalBuffer *NewHeapBuffer(WebMetalHeap *heap, SEL _cmd, NSUInteger length, MTLResourceOptions options,
                              uint64_t heapOptions, uint64_t offset, uint64_t rangeLength, void *contents,
                              bool placement)
{
    WebMetalDevice *device = static_cast<WebMetalDevice *>(heap.device);
    WebMetalBuffer *buffer = [WebMetalBuffer new];
    InitResourceState(buffer->_state, device, heapOptions);
    buffer->_state.purgeableAllowed = false;
    buffer->_state.heap = heap;
    buffer->_state.heapOffset = offset;
    buffer->_state.heapLength = rangeLength;
    buffer->_state.pinned = !placement;
    buffer->_length = length;
    buffer->_contents = contents;
    Connection *connection = DeviceConnection(device);
    buffer->_state.handle = connection->NewHandle();
    wire::HeapCreateBuffer request;
    request.heap = HeapHandle(heap);
    request.buffer = buffer->_state.handle;
    request.length = length;
    request.options = options;
    request.contents = reinterpret_cast<uint64_t>(contents);
    wire::Reply<wire::HeapCreateBufferReply> reply;
    connection->Call(heap, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(heap, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
    buffer->_state.created = true;
    return buffer;
}

}  // namespace webmetal

@implementation WebMetalBuffer

// -[IOGPUMetalBuffer dealloc] (IOGPU 0x1fcb2640c): a heap buffer becomes aliasable and returns its range to the heap;
// the deallocator of caller memory runs with the pointer and length; then the kernel resource goes (here: Release, sent
// before the deallocator runs so that the host has stopped using the caller's memory when it is handed back).
- (void)dealloc
{
    using namespace webmetal;
    DeviceAddAllocation(_state.device, -_accounted);
    if (_state.heap) {
        MakeAliasable(self, _cmd, _state, true);
        [(WebMetalHeap *)_state.heap deallocHeapSubResource];
    }
    ReleaseObject(_state, "-[WebMetalBuffer dealloc]");
    if (_deallocator) {
        _deallocator(_contents, _length);
        _deallocator = nil;
    }
    if (_ownedSize)
        vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(_contents), _ownedSize);
}

// MTLResource ----------------------------------------------------------------------------------------------------------

- (id<MTLDevice>)device
{
    return _state.device;
}

- (NSString *)label
{
    return webmetal::RetainedLabel(_state);
}

- (NSString *)retainedLabel
{
    return webmetal::RetainedLabel(_state);
}

- (void)setLabel:(NSString *)label
{
    webmetal::SetLabel(self, _cmd, _state, label);
}

- (MTLCPUCacheMode)cpuCacheMode
{
    return static_cast<MTLCPUCacheMode>(_state.cpuCacheMode);
}

- (MTLStorageMode)storageMode
{
    return static_cast<MTLStorageMode>(_state.storageMode);
}

- (MTLHazardTrackingMode)hazardTrackingMode
{
    return webmetal::HazardTrackingMode(_state);
}

- (MTLResourceOptions)resourceOptions
{
    return webmetal::ResourceOptions(_state);
}

// -[IOGPUMetalResource unfilteredResourceOptions] (0x1fcb398c4): the options as given.
- (MTLResourceOptions)unfilteredResourceOptions
{
    return _state.options;
}

- (id<MTLHeap>)heap
{
    return _state.heap;
}

- (NSUInteger)heapOffset
{
    return webmetal::HeapOffset(_state);
}

// -[IOGPUMetalResource isAliasable] (0x1fcb39f14).
- (BOOL)isAliasable
{
    return !_state.pinned;
}

// -[IOGPUMetalResource isComplete] (0x1fcb39934), on the buffer's kernel resource: its own (a sub-allocated buffer's is
// made with its primary's options, untracked exactly when the buffer's are: IOGPUMetalSuballocatorAllocate keys its
// pools by them, clearing only the tracked mode) or, for a heap buffer, the heap's.
- (BOOL)isComplete
{
    return webmetal::CheckEventComplete(self, _cmd, webmetal::KernelResourceKey(self), _state, false, false);
}

// -[IOGPUMetalResource waitUntilComplete] (0x1fcb39964).
- (void)waitUntilComplete
{
    webmetal::CheckEventComplete(self, _cmd, webmetal::KernelResourceKey(self), _state, false, true);
}

- (void)makeAliasable
{
    webmetal::MakeAliasable(self, _cmd, _state, false);
}

// -[IOGPUMetalBuffer allocatedSize] (0x1fcb31bfc): a sub-allocated buffer's length; otherwise
// -[IOGPUMetalResource allocatedSize] (0x1fcb3a10c): a heap buffer's range length, or the kernel resource's resident size.
- (NSUInteger)allocatedSize
{
    if (_suballocated)
        return _length;
    if (_state.heap)
        return _state.heapLength;
    return _state.residentSize;
}

// -[IOGPUMetalResource resourceSize] (0x1fcb2e758): _res.info.resourceSize, its low 56 bits: the kernel resource's
// resident size. A heap buffer takes the heap's backing resource's _res, info included (-[IOGPUMetalBuffer
// initWithHeap:resource:offset:length:] 0x1fcb2d810 through -[IOGPUMetalResource initWithResource:] 0x1fcb2acc8); a
// sub-allocated buffer's kernel resource has its primary buffer's resident size (kSuballocatedResourceSize).
- (NSUInteger)resourceSize
{
    if (_state.heap)
        return [_state.heap currentAllocatedSize] & 0x00ffffffffffffffull;
    return _state.residentSize & 0x00ffffffffffffffull;
}

- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state
{
    return webmetal::SetPurgeableState(self, _cmd, _state, state);
}

// -[IOGPUMetalResource isPurgeable] (0x1fcb39914): the keep-current answer is above non-volatile.
- (BOOL)isPurgeable
{
    return [self setPurgeableState:MTLPurgeableStateKeepCurrent] > MTLPurgeableStateNonVolatile;
}

// -[IOGPUMetalResource protectionOptions] (0x1fcb2df48): the kernel resource's protection options; a buffer's driver
// path requests none.
- (uint64_t)protectionOptions
{
    return _state.protectionOptions;
}

- (int)responsibleProcess
{
    return _state.responsibleProcess;
}

// -[IOGPUMetalResource setResponsibleProcess:] (0x1fcb398d8).
- (void)setResponsibleProcess:(int)pid
{
    _state.responsibleProcess = pid;
}

- (BOOL)doesAliasResource:(id<MTLResource>)resource
{
    webmetal::ResourceState *other = webmetal::StateOf(resource);
    return other && webmetal::DoesAlias(_state, *other);
}

// -[IOGPUMetalResource doesAliasAllResources:count:] (0x1fcb39fb0): YES for a non-empty list every member of which
// aliases the receiver.
- (BOOL)doesAliasAllResources:(const id<MTLResource> *)resources count:(NSUInteger)count
{
    if (!count)
        return NO;
    for (NSUInteger i = 0; i < count; i++) {
        webmetal::ResourceState *other = webmetal::StateOf(resources[i]);
        if (!other || !webmetal::DoesAlias(_state, *other))
            return NO;
    }
    return YES;
}

// -[IOGPUMetalResource doesAliasAnyResources:count:] (0x1fcb3a050).
- (BOOL)doesAliasAnyResources:(const id<MTLResource> *)resources count:(NSUInteger)count
{
    for (NSUInteger i = 0; i < count; i++) {
        webmetal::ResourceState *other = webmetal::StateOf(resources[i]);
        if (other && webmetal::DoesAlias(_state, *other))
            return YES;
    }
    return NO;
}

// -[IOGPUMetalBuffer setOwnerWithIdentity:] (0x1fcb32120) for a buffer without an IOSurface:
// -[IOGPUMetalResource setOwnerWithIdentity:].
- (int)setOwnerWithIdentity:(uint32_t)identity
{
    return webmetal::SetOwnerWithIdentity(self, _cmd, _state, identity);
}

// MTLBuffer ------------------------------------------------------------------------------------------------------------

// -[IOGPUMetalBuffer length] (0x1fcb26538).
- (NSUInteger)length
{
    return _length;
}

// -[IOGPUMetalBuffer contents] (0x1fcb2600c): the virtual address.
- (void *)contents
{
    return _contents;
}

// -[IOGPUMetalResource virtualAddress] (0x1fcb28048).
- (void *)virtualAddress
{
    return _contents;
}

// -[IOGPUMetalBuffer didModifyRange:] (0x1fcb31d8c): a failure report; the method is macOS-only.
- (void)didModifyRange:(NSRange)range
{
    MTLReportFailure(0, "-[IOGPUMetalBuffer didModifyRange:]", 0x1e2, @"didModifyRange: is not supported on this platform.");
}

// -[AGXA10FamilyBuffer addDebugMarker:range:] and -removeAllDebugMarkers (AGXMetalA10 0x21c8c643c, 0x21c8c6440) do
// nothing on the device; the markers are carried to the host as the wire protocol defines.
- (void)addDebugMarker:(NSString *)marker range:(NSRange)range
{
    using namespace webmetal;
    wire::BufferAddDebugMarker request;
    request.buffer = _state.handle;
    if (!wire::strFromNSString(marker, request.marker))
        WMLoudMethod(self, _cmd, "the marker has no UTF-8 form, which the wire format carries");
    request.range.location = range.location;
    request.range.length = range.length;
    wire::Reply<wire::BufferAddDebugMarkerReply> reply;
    DeviceConnection(_state.device)->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
}

- (void)removeAllDebugMarkers
{
    using namespace webmetal;
    wire::BufferRemoveAllDebugMarkers request;
    request.buffer = _state.handle;
    wire::Reply<wire::BufferRemoveAllDebugMarkersReply> reply;
    DeviceConnection(_state.device)->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
}

// -[AGXA10FamilyBuffer newTextureWithDescriptor:offset:bytesPerRow:] (AGXMetalA10 0x21c8c6374).
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                    offset:(NSUInteger)offset
                               bytesPerRow:(NSUInteger)bytesPerRow
{
    return webmetal::NewBufferTexture(self, _cmd, descriptor, offset, bytesPerRow, false, 0);
}

// -[IOGPUMetalBuffer newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:] (IOGPU 0x1fcb2dfd8).
- (id<MTLTexture>)newLinearTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                          offset:(NSUInteger)offset
                                     bytesPerRow:(NSUInteger)bytesPerRow
                                   bytesPerImage:(NSUInteger)bytesPerImage
{
    return webmetal::NewBufferTexture(self, _cmd, descriptor, offset, bytesPerRow, true, bytesPerImage);
}

// -[IOGPUMetalBuffer iosurface] (0x1fcb2df38).
- (IOSurfaceRef)iosurface
{
    return _state.iosurface;
}

// -[AGXBuffer backingResource] (0x21c8a488c) returns nothing (a bare return: nil).
- (id)backingResource
{
    return nil;
}

// -[AGXBuffer parentGPUAddress] and -parentGPUSize (0x21c8a4894, 0x21c8a48b0): the primary buffer's GPU address and
// length for a sub-allocated buffer, 0 otherwise. A sub-allocation has no primary buffer of its own here.
- (uint64_t)parentGPUAddress
{
    if (_suballocated)
        WMLoudMethod(self, _cmd, "the driver's primary buffers of sub-allocations have no WebMetal GPU address");
    return 0;
}

- (uint64_t)parentGPUSize
{
    if (_suballocated)
        WMLoudMethod(self, _cmd, "the driver's primary buffers of sub-allocations have no WebMetal counterpart");
    return 0;
}

// -[IOGPUMetalBuffer formattedDescription:] (0x1fcb31a20): the object's own description, then the label, length, modes,
// options and purgeable state, each on its own line indented by `indent` + 4.
- (NSString *)formattedDescription:(NSUInteger)indent
{
    NSString *pad = [@"\n" stringByPaddingToLength:indent + 4 withString:@" " startingAtIndex:0];
    NSString *label = webmetal::RetainedLabel(_state);
    NSArray *parts = @[
        pad, @"label =", label ?: @"<none>",
        pad, @"length =", @(_length),
        pad, @"cpuCacheMode =", MTLCPUCacheModeString(static_cast<MTLCPUCacheMode>(_state.cpuCacheMode)),
        pad, @"storageMode =", MTLStorageModeString(static_cast<MTLStorageMode>(_state.storageMode)),
        pad, @"hazardTrackingMode =", MTLHazardTrackingModeString(self.hazardTrackingMode),
        pad, @"resourceOptions =", MTLResourceOptionsString(self.resourceOptions),
        pad, @"purgeableState =", MTLPurgeableStateString(static_cast<MTLPurgeableState>(_state.purgeableState)),
    ];
    return [NSString stringWithFormat:@"%@%@", [super description], [parts componentsJoinedByString:@" "]];
}

// -[IOGPUMetalBuffer description] (0x1fcb31bf4).
- (NSString *)description
{
    return [self formattedDescription:0];
}

@end
