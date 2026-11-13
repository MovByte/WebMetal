// Textures: WebMetalTexture stands in for AGXA10FamilyTexture (AGXA10FamilyTexture -> AGXTexture -> IOGPUMetalTexture ->
// IOGPUMetalResource -> _MTLResource on 21H461).
//
// The driver checks a descriptor with Metal.framework's own -[MTLTextureDescriptorInternal validateWithDevice:], lays
// the texture out in its private memory format (AGX::Texture, the AGX layout), and creates the kernel resource. The
// plugin calls the same validation, and the host executor holds the texture's contents (CreateTexture and the other
// creation requests). The AGX layout's sizes, alignment and row pitch come from the port of AGX::TextureGen2
// (WebMetalTextureLayout.h) for device and heap textures, textures over caller memory and their views; textures over
// IOSurfaces and buffers have no layout here, and what depends on it is LOUD for them.

#import "WebMetalTexture.h"
#import "WebMetalBuffer.h"
#import "WebMetalHeapPrivate.h"
#import "WebMetalTextureLayout.h"

#import <objc/message.h>

#include <IOSurface/IOSurfaceRef.h>

#include <algorithm>
#include <cstring>
#include <unistd.h>

// The class adopts the protocols AGXA10FamilyTexture adopts and implements only what the 21H461 driver layers
// implement; the protocols' other methods are LOUD stubs added at run time (guest/Plugin/WebMetalStubs.mm).
#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"

extern "C" {
NSString *MTLCPUCacheModeString(MTLCPUCacheMode mode);
NSString *MTLStorageModeString(MTLStorageMode mode);
NSString *MTLHazardTrackingModeString(MTLHazardTrackingMode mode);
NSString *MTLResourceOptionsString(MTLResourceOptions options);
NSString *MTLPurgeableStateString(MTLPurgeableState state);
MTLTextureSwizzleChannels MTLTextureSwizzleKeyToChannels(uint32_t key);
uint32_t MTLTextureSwizzleChannelsToKey(MTLTextureSwizzleChannels channels);
uint32_t IOSurfaceGetCompressionTypeOfPlane(IOSurfaceRef surface, size_t plane);
Boolean IOSurfaceIsTiled(IOSurfaceRef surface);
size_t IOSurfaceGetSliceCount(IOSurfaceRef surface);
// Metal.framework's texture-view checks, which -[IOGPUMetalTexture initWithTextureInternal:...] calls. They report
// through MTLReportFailure.
void _mtlValidateMTLTextureSwizzleKey(uint32_t key);
void _mtlValidateArgumentsForTextureViewOnDevice(id<MTLDevice> device, id<MTLTexture> texture, MTLPixelFormat pixelFormat,
                                                 MTLTextureType textureType, NSUInteger levelLocation,
                                                 NSUInteger levelLength, NSUInteger sliceLocation,
                                                 NSUInteger sliceLength, BOOL compressedView);
MTLTextureUsage MTLGetDisallowedTextureUsagesWhenSwizzling(id<MTLDevice> device, BOOL writeSwizzleEnabled);
void _mtlValidateTextureUsage(MTLTextureUsage usage);
}

// -[_MTLDevice supportsTextureSwizzle] (Metal.framework), which the plugin's device inherits.
@protocol WMTextureSwizzleSupport
- (BOOL)supportsTextureSwizzle;
@end

namespace {

constexpr char kLayout[] = "the A10X texture layout (AGX::TextureGen2 in AGXMetalA10) of a texture over an IOSurface or "
                           "a buffer is not ported";
}  // namespace

@interface WebMetalTexture () {
@public
    webmetal::ResourceState _state;
    // IOGPUMetalTexture's instance variables (the comments name them).
    WebMetalTexture *_parentTexture;     // _parentTexture
    NSUInteger _parentRelativeLevel;     // _parentRelativeLevel
    NSUInteger _parentRelativeSlice;     // _parentRelativeSlice
    WebMetalBuffer *_buffer;             // _buffer
    uint32_t _swizzle;                   // _swizzle: MTLTextureSwizzleChannels as its key
    bool _writeSwizzleEnabled;           // _writeSwizzleEnabled
    bool _isCompressed;                  // _isCompressed
    NSUInteger _iosurfacePlane;          // _iosurfacePlane
    MTLTextureType _textureType;         // _textureType
    MTLPixelFormat _pixelFormat;         // _pixelFormat
    MTLTextureUsage _usage;              // _usage
    NSUInteger _rotation;                // _rotation
    NSUInteger _width, _height, _depth;  // _width, _height, _depth
    NSUInteger _mipmapLevelCount;        // _mipmapLevelCount
    NSUInteger _sampleCount;             // _sampleCount
    NSUInteger _arrayLength;             // _arrayLength
    NSUInteger _numFaces;                // _numFaces
    bool _shareable;                     // _shareable
    bool _framebufferOnly;               // _framebufferOnly
    bool _isDrawable;                    // _isDrawable
    NSUInteger _bufferOffset;            // _bufferOffset
    NSUInteger _bufferBytesPerRow;       // _bufferBytesPerRow
    bool _allowGPUOptimizedContents;     // _allowGPUOptimizedContents
    // AGXTexture's layout answers and AGXA10FamilyTexture's layout object (_impl), for device, heap and view textures
    // (_hasLayout).
    bool _hasLayout;
    uint64_t _alignment;     // _alignment
    uint64_t _cpuSizeBytes;  // _cpuSizeBytes
    uint64_t _bytesPerRow;   // _bytesPerRow
    webmetal::agx::TextureGen2 _layout;  // _impl
    // The driver sub-allocates the texture (-[AGXTexture initWithDevice:desc:isSuballocDisabled:]): its allocated size
    // is its CPU size, which it adds to the device total (_accounted).
    bool _suballocated;
    int64_t _accounted;
    // A texture over caller memory (newTextureWithBytesNoCopy:length:descriptor:deallocator:): IOGPUMetalTexture's
    // _pointer, _length and _deallocator.
    void *_pointer;
    NSUInteger _length;
    void (^_deallocator)(void *pointer, NSUInteger length);
}
@end

// IOGPUMetalResource's -resourceSize, which WebMetalBuffer answers.
@protocol WebMetalResourceSize
- (NSUInteger)resourceSize;
@end

namespace webmetal {

wire::TextureDescriptor WireTextureDescriptor(id self, SEL _cmd, MTLTextureDescriptor *d)
{
    wire::TextureDescriptor w;
    w.allowGpuOptimizedContents = Get<BOOL>(self, _cmd, d, "allowGPUOptimizedContents");
    w.arrayLength = Get<NSUInteger>(self, _cmd, d, "arrayLength");
    w.colorSpaceConversionMatrix = Get<NSUInteger>(self, _cmd, d, "colorSpaceConversionMatrix");
    w.compressionFootprint = Get<NSUInteger>(self, _cmd, d, "compressionFootprint");
    w.compressionMode = Get<NSUInteger>(self, _cmd, d, "compressionMode");
    w.compressionType = Get<NSInteger>(self, _cmd, d, "compressionType");
    w.cpuCacheMode = Get<NSUInteger>(self, _cmd, d, "cpuCacheMode");
    w.depth = Get<NSUInteger>(self, _cmd, d, "depth");
    w.forceResourceIndex = Get<BOOL>(self, _cmd, d, "forceResourceIndex");
    w.framebufferOnly = Get<BOOL>(self, _cmd, d, "framebufferOnly");
    w.hazardTrackingMode = Get<NSUInteger>(self, _cmd, d, "hazardTrackingMode");
    w.height = Get<NSUInteger>(self, _cmd, d, "height");
    w.isDrawable = Get<BOOL>(self, _cmd, d, "isDrawable");
    w.lossyCompressionMode = Get<NSInteger>(self, _cmd, d, "lossyCompressionMode");
    w.mipmapLevelCount = Get<NSUInteger>(self, _cmd, d, "mipmapLevelCount");
    w.pixelFormat = Get<NSUInteger>(self, _cmd, d, "pixelFormat");
    w.protectionOptions = Get<NSUInteger>(self, _cmd, d, "protectionOptions");
    w.resourceIndex = Get<NSUInteger>(self, _cmd, d, "resourceIndex");
    w.resourceOptions = Get<NSUInteger>(self, _cmd, d, "resourceOptions");
    w.rotation = Get<NSUInteger>(self, _cmd, d, "rotation");
    w.sampleCount = Get<NSUInteger>(self, _cmd, d, "sampleCount");
    w.sparseSurfaceDefaultValue = Get<NSUInteger>(self, _cmd, d, "sparseSurfaceDefaultValue");
    w.storageMode = Get<NSUInteger>(self, _cmd, d, "storageMode");
    const MTLTextureSwizzleChannels swizzle = Get<MTLTextureSwizzleChannels>(self, _cmd, d, "swizzle");
    w.swizzle.red = swizzle.red;
    w.swizzle.green = swizzle.green;
    w.swizzle.blue = swizzle.blue;
    w.swizzle.alpha = swizzle.alpha;
    w.swizzleKey = Get<uint32_t>(self, _cmd, d, "swizzleKey");
    w.textureType = Get<NSUInteger>(self, _cmd, d, "textureType");
    w.textureUsage = Get<NSUInteger>(self, _cmd, d, "textureUsage");
    w.usage = Get<NSUInteger>(self, _cmd, d, "usage");
    w.width = Get<NSUInteger>(self, _cmd, d, "width");
    w.writeSwizzleEnabled = Get<BOOL>(self, _cmd, d, "writeSwizzleEnabled");
    return w;
}

ResourceState *TextureState(WebMetalTexture *texture)
{
    return &texture->_state;
}

// What the AGX layout reads besides the descriptor: the page size, and the IOSurface offset alignment that
// AGX::Texture's constructor keeps in a static local (IOSurfaceAlignProperty(kIOSurfaceOffset, 1)).
agx::SystemValues LayoutSystemValues()
{
    static const agx::SystemValues values = [] {
        agx::SystemValues v;
        v.vmPageSize = PageSize();
        v.iosurfaceAlignment = IOSurfaceAlignProperty(kIOSurfaceOffset, 1);
        return v;
    }();
    return values;
}

// The MTLTextureDescriptorPrivate fields the driver's layout callers read.
agx::DescriptorFields LayoutDescriptorFields(const wire::TextureDescriptor &d)
{
    agx::DescriptorFields f;
    f.textureType = d.textureType;
    f.pixelFormat = d.pixelFormat;
    f.width = d.width;
    f.height = d.height;
    f.depth = d.depth;
    f.mipmapLevelCount = d.mipmapLevelCount;
    f.sampleCount = d.sampleCount;
    f.arrayLength = d.arrayLength;
    f.rotation = d.rotation;
    f.swizzle = d.swizzleKey;
    f.compressionMode = d.compressionMode;
    f.usage = d.usage;
    f.resourceOptions = d.resourceOptions;
    f.allowGPUOptimizedContents = d.allowGpuOptimizedContents;
    f.compressionFootprint = d.compressionFootprint;
    f.colorSpaceConversion = d.colorSpaceConversionMatrix;
    f.storageMode = d.storageMode;
    return f;
}

SizeAndAlign HeapTextureSizeAndAlign(WebMetalDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor)
{
    if (![(id<WMDescriptorValidation>)descriptor validateWithDevice:device])
        return SizeAndAlign{0, 0};
    const agx::DescriptorFields fields = LayoutDescriptorFields(WireTextureDescriptor(device, _cmd, descriptor));
    const agx::SizeAndAlign sa = agx::heapTextureSizeAndAlign(fields, LayoutSystemValues());
    return SizeAndAlign{sa.size, sa.align};
}

// The fields -[IOGPUMetalTexture initWithDevice:descriptor:sysMemSize:sysMemRowBytes:vidMemSize:vidMemRowBytes:args:
// argsSize:] (IOGPU 0x1fcb2b8a8) copies from the descriptor's private state: type, size, levels, samples, array length,
// pixel format, usage, swizzle, write swizzle, drawable and GPU-optimized contents; six faces for cube and cube-array
// textures; not compressed, not shareable. The driver traps on a rotation or a framebuffer-only descriptor (its
// .cold.1 and .cold.2), which only drawables use.
static WebMetalTexture *newTextureObject(WebMetalDevice *device, SEL _cmd, const wire::TextureDescriptor &d)
{
    if (d.rotation)
        WMLoudMethod(device, _cmd, "21H461 IOGPU traps on a descriptor with a rotation (the drawable path, not served yet)");
    if (d.framebufferOnly)
        WMLoudMethod(device, _cmd, "21H461 IOGPU traps on a framebuffer-only descriptor (the drawable path, not served yet)");
    WebMetalTexture *texture = [WebMetalTexture new];
    InitResourceState(texture->_state, device, d.resourceOptions);
    texture->_textureType = static_cast<MTLTextureType>(d.textureType);
    texture->_width = d.width;
    texture->_height = d.height;
    texture->_depth = d.depth;
    texture->_mipmapLevelCount = d.mipmapLevelCount;
    texture->_sampleCount = d.sampleCount;
    texture->_arrayLength = d.arrayLength;
    texture->_numFaces = d.textureType - MTLTextureTypeCube < 2 ? 6 : 1;
    texture->_pixelFormat = static_cast<MTLPixelFormat>(d.pixelFormat);
    texture->_usage = d.usage;
    texture->_swizzle = MTLTextureSwizzleChannelsToKey(MTLTextureSwizzleChannels{
        static_cast<MTLTextureSwizzle>(d.swizzle.red), static_cast<MTLTextureSwizzle>(d.swizzle.green),
        static_cast<MTLTextureSwizzle>(d.swizzle.blue), static_cast<MTLTextureSwizzle>(d.swizzle.alpha)});
    texture->_writeSwizzleEnabled = d.writeSwizzleEnabled;
    texture->_isDrawable = d.isDrawable;
    texture->_allowGPUOptimizedContents = d.allowGpuOptimizedContents;
    texture->_state.protectionOptions = d.protectionOptions;
    texture->_state.handle = DeviceConnection(device)->NewHandle();
    return texture;
}

template <class Request, class ReplyFields>
static void create(WebMetalDevice *device, SEL _cmd, WebMetalTexture *texture, const Request &request)
{
    wire::Reply<ReplyFields> reply;
    DeviceConnection(device)->Call(device, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(device, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
    texture->_state.created = true;
    if (texture->_hasLayout) {
        texture->_accounted = static_cast<int64_t>(texture->_suballocated ? texture->_cpuSizeBytes
                                                                          : texture->_state.residentSize);
        DeviceAddAllocation(device, texture->_accounted);
    } else {
        DeviceAddUnsized(device, 1);
    }
}

static void setLayout(WebMetalTexture *texture, const agx::TextureAllocation &layout)
{
    texture->_hasLayout = true;
    texture->_alignment = layout.alignment;
    texture->_cpuSizeBytes = layout.cpuSizeBytes;
    texture->_bytesPerRow = layout.bytesPerRow;
    texture->_layout = layout.layout;
}

// -[AGXTexture initWithDevice:desc:isSuballocDisabled:] (AGXMetalA10 0x21c9d48a8): Metal's descriptor validation, then
// the AGX layout (-[AGXA10FamilyTexture initImplWithDevice:...allowNPOT:YES ... isHeapBacked:NO], which fails for a
// layout that is not valid). A texture that is not memoryless, has no protection options, is smaller than a page and
// whose layout does not veto it (initNewTextureData: +0x58 bit 0) is sub-allocated from the device's buffers, unless
// DisableDriverSuballocation is set (AGX::Device +0x945) or the sub-allocator refuses it
// (-[IOGPUMetalDevice allocBufferSubDataWithLength:options:alignment:...]: a private texture's options gain 0x20000,
// which picks no other pool; alignments above 0x20000 are refused). Then the kernel resource (CreateTexture).
WebMetalTexture *NewDeviceTexture(WebMetalDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor)
{
    if (![(id<WMDescriptorValidation>)descriptor validateWithDevice:device])
        return nil;
    const wire::TextureDescriptor d = WireTextureDescriptor(device, _cmd, descriptor);
    const agx::TextureAllocation layout = agx::initImplDevice(LayoutDescriptorFields(d), LayoutSystemValues());
    if (!layout.initialized)
        return nil;
    WebMetalTexture *texture = newTextureObject(device, _cmd, d);
    setLayout(texture, layout);
    uint64_t options = d.resourceOptions;
    if ((options & kStorageModeMask) == (uint64_t(MTLStorageModePrivate) << kStorageModeShift))
        options |= 0x20000;
    texture->_suballocated = d.storageMode != MTLStorageModeMemoryless && d.protectionOptions == 0 &&
                             layout.cpuSizeBytes < PageSize() && !(layout.textureData58 & 1) &&
                             !SuballocationDisabled() && SuballocatorServes(layout.cpuSizeBytes, options) &&
                             layout.alignment <= 0x20000;
    // A sub-allocated texture is a texture over the primary buffer with a kernel resource of its own
    // (-[IOGPUMetalTexture initWithPrimaryBuffer:...] 0x1fcb2e244), whose resident size is the primary's. Otherwise
    // -[IOGPUMetalTexture initWithDevice:descriptor:sysMemSize:...] creates the kernel resource with the CPU size as its
    // system memory (IOGPU 0x1fcb2c870), whose resident size is that size rounded up to the kernel page; a memoryless
    // texture has no kernel resource and a resident size of 0 (IOGPU 0x1fcb2bbe8).
    if (texture->_suballocated)
        texture->_state.residentSize = kSuballocatedResourceSize;
    else
        texture->_state.residentSize = d.storageMode == MTLStorageModeMemoryless
                                           ? 0
                                           : (layout.cpuSizeBytes + kKernelPageSize - 1) & ~(kKernelPageSize - 1);
    wire::CreateTexture request;
    request.texture = texture->_state.handle;
    request.descriptor = d;
    create<wire::CreateTexture, wire::CreateTextureReply>(device, _cmd, texture, request);
    return texture;
}

// -[AGXA10FamilyDevice newTextureWithBytesNoCopy:length:descriptor:deallocator:] (AGXMetalA10 0x21c8d62c8):
// -[AGXTexture initWithDevice:bytes:length:desc:deallocator:] (0x21c9d54e0) validates the descriptor, lays the texture
// out as a device texture (initImplWithDevice:...allowNPOT:YES ... isHeapBacked:NO, nil when the layout is not valid)
// and has -[IOGPUMetalTexture initWithDevice:descriptor:sysMemPointer:sysMemSize:sysMemLength:sysMemRowBytes:args:
// argsSize:deallocator:] (IOGPU 0x1fcb30400) make the kernel resource over the caller's memory, the layout's size from
// bytes on, keeping the pointer, the length and a copy of the deallocator. -validateBufferTextureWithSize: (0x21c982338)
// checks the length and the watermark only for layouts with +0x17a set, which a texture without a buffer or an IOSurface
// does not have. The kernel resource's resident size is the caller's pages under the layout's size
// (ClientMemoryResidentSize). The memory holds the texels in the A10X layout; the host keeps it and the texture equal
// (CreateTextureNoCopy). A memoryless texture has no kernel resource (-[IOGPUMetalResource initWithDevice:options:args:
// argsSize:], IOGPU 0x1fcb2bbe8): the memory is never mapped and its resident size is 0.
WebMetalTexture *NewNoCopyTexture(WebMetalDevice *device, SEL _cmd, void *bytes, NSUInteger length,
                                  MTLTextureDescriptor *descriptor, void (^deallocator)(void *pointer, NSUInteger length))
{
    if (![(id<WMDescriptorValidation>)descriptor validateWithDevice:device])
        return nil;
    const wire::TextureDescriptor d = WireTextureDescriptor(device, _cmd, descriptor);
    const agx::TextureAllocation layout = agx::initImplDevice(LayoutDescriptorFields(d), LayoutSystemValues());
    if (!layout.initialized)
        return nil;
    WebMetalTexture *texture = newTextureObject(device, _cmd, d);
    setLayout(texture, layout);
    texture->_pointer = bytes;
    texture->_length = length;
    texture->_deallocator = [deallocator copy];
    texture->_state.residentSize = d.storageMode == MTLStorageModeMemoryless
                                       ? 0
                                       : ClientMemoryResidentSize(device, _cmd, bytes, layout.cpuSizeBytes);
    const agx::SystemValues system = LayoutSystemValues();
    wire::CreateTextureNoCopy request;
    request.texture = texture->_state.handle;
    request.bytes = reinterpret_cast<uint64_t>(bytes);
    request.length = length;
    request.descriptor = d;
    request.deallocator = deallocator != nil;
    request.pageSize = system.vmPageSize;
    request.iosurfaceAlignment = system.iosurfaceAlignment;
    create<wire::CreateTextureNoCopy, wire::CreateTextureNoCopyReply>(device, _cmd, texture, request);
    return texture;
}

// -[AGXA10FamilyDevice newTextureWithDescriptor:iosurface:plane:] (AGXMetalA10 0x21c8d64e8): a surface whose creation
// properties carry kAGXCompressionMode was made by the AGX driver for its compressed layout and takes the shared
// texture path; any other goes to -[AGXTexture initWithDevice:desc:iosurface:plane:slice:] (0x21c9d529c), which refuses a
// plane compressed with type 1 or above 4, validates the descriptor, refuses a tiled surface, and creates the kernel
// resource over the plane.
WebMetalTexture *NewSurfaceTexture(WebMetalDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor,
                                   IOSurfaceRef iosurface, NSUInteger plane, NSUInteger slice)
{
    CFTypeRef properties = IOSurfaceCopyValue(iosurface, CFSTR("CreationProperties"));
    if (properties) {
        id mode = ((__bridge NSDictionary *)properties)[@"kAGXCompressionMode"];
        CFRelease(properties);
        if (mode)
            WMLoudMethod(device, _cmd, "a surface the AGX driver created for its compressed layout "
                                       "(-[AGXTexture initWithSharedTextureHandle:device:desc:compressionMode:])");
    }
    const uint32_t compression = IOSurfaceGetCompressionTypeOfPlane(iosurface, plane);
    if (compression == 1 || compression > 4)
        return nil;
    if (![(id<WMDescriptorValidation>)descriptor validateWithDevice:device])
        return nil;
    if (IOSurfaceIsTiled(iosurface))
        return nil;
    if (slice)
        WMLoudMethod(device, _cmd, "a texture over one slice of a multi-slice surface: the wire carries no slice");
    const wire::TextureDescriptor d = WireTextureDescriptor(device, _cmd, descriptor);
    WebMetalTexture *texture = newTextureObject(device, _cmd, d);
    texture->_state.iosurface = (IOSurfaceRef)CFRetain(iosurface);
    texture->_iosurfacePlane = plane;
    wire::CreateTextureWithIOSurface request;
    request.texture = texture->_state.handle;
    request.descriptor = d;
    request.iosurface = IOSurfaceGetID(iosurface);
    request.plane = plane;
    create<wire::CreateTextureWithIOSurface, wire::CreateTextureWithIOSurfaceReply>(device, _cmd, texture, request);
    return texture;
}

// -[AGXTexture initWithBuffer:desc:offset:bytesPerRow:] (AGXMetalA10 0x21c9d4ff0): the descriptor is validated against the
// buffer's device, laid out over the buffer's bytes, and the texture keeps the buffer, the offset and the row pitch.
// -newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage: marks the same call as linear for the AGX layout
// (a thread-local flag around it) and does not pass the image pitch on; the host takes it as CreateLinearTexture.
WebMetalTexture *NewBufferTexture(WebMetalBuffer *buffer, SEL _cmd, MTLTextureDescriptor *descriptor, NSUInteger offset,
                                  NSUInteger bytesPerRow, bool linear, NSUInteger bytesPerImage)
{
    WebMetalDevice *device = static_cast<WebMetalDevice *>(buffer.device);
    if (![(id<WMDescriptorValidation>)descriptor validateWithDevice:device])
        return nil;
    const wire::TextureDescriptor d = WireTextureDescriptor(buffer, _cmd, descriptor);
    WebMetalTexture *texture = newTextureObject(device, _cmd, d);
    texture->_buffer = buffer;
    texture->_bufferOffset = offset;
    texture->_bufferBytesPerRow = bytesPerRow;
    if (linear) {
        wire::CreateLinearTexture request;
        request.buffer = BufferHandle(buffer);
        request.texture = texture->_state.handle;
        request.descriptor = d;
        request.offset = offset;
        request.bytesPerRow = bytesPerRow;
        request.bytesPerImage = bytesPerImage;
        create<wire::CreateLinearTexture, wire::CreateLinearTextureReply>(device, _cmd, texture, request);
    } else {
        wire::CreateBufferTexture request;
        request.buffer = BufferHandle(buffer);
        request.texture = texture->_state.handle;
        request.descriptor = d;
        request.offset = offset;
        request.bytesPerRow = bytesPerRow;
        create<wire::CreateBufferTexture, wire::CreateBufferTextureReply>(device, _cmd, texture, request);
    }
    return texture;
}

// -[AGXTexture initWithHeap:desc:] (AGXMetalA10 0x21c9d4d6c) and initWithHeap:desc:atOffset: (0x21c9d4ea4): an
// automatic heap (a placement heap for the offset form; nil for any other), Metal's descriptor validation against the
// heap's device, the AGX layout of a heap texture (initImpl with allowNPOT:NO isHeapBacked:YES), and the heap's range
// of the layout's CPU size at its alignment (-[IOGPUMetalHeap newSubResourceWithLength:alignment:options:offset:] or
// newSubResourceAtOffset:withLength:alignment:options:); then -[IOGPUMetalTexture initWithHeap:resource:offset:length:
// device:descriptor:] (IOGPU 0x1fcb2d5d8) takes the heap resource's state, pinned unless the heap is a placement heap.
WebMetalTexture *NewHeapTexture(WebMetalHeap *heap, SEL _cmd, MTLTextureDescriptor *descriptor, bool placement,
                                NSUInteger offset)
{
    if (heap.type != (placement ? MTLHeapTypePlacement : MTLHeapTypeAutomatic))
        return nil;
    WebMetalDevice *device = static_cast<WebMetalDevice *>(heap.device);
    if (![(id<WMDescriptorValidation>)descriptor validateWithDevice:device])
        return nil;
    const wire::TextureDescriptor d = WireTextureDescriptor(heap, _cmd, descriptor);
    const agx::TextureAllocation layout = agx::initImplHeap(LayoutDescriptorFields(d), LayoutSystemValues());
    if (!layout.initialized)
        return nil;
    if (d.storageMode == MTLStorageModeMemoryless)
        WMLoudMethod(heap, _cmd, "a memoryless heap texture (-[AGXTexture initWithImplHeapMemoryless:descriptor:length:])");
    uint64_t at = offset;
    if (!HeapSubResource(heap, _cmd, layout.cpuSizeBytes, layout.alignment, d.resourceOptions, placement, &at))
        return nil;
    WebMetalTexture *texture = newTextureObject(device, _cmd, d);
    InitResourceState(texture->_state, device, HeapOptions(heap));
    texture->_state.protectionOptions = d.protectionOptions;
    texture->_state.purgeableAllowed = false;
    texture->_state.heap = heap;
    texture->_state.heapOffset = at;
    texture->_state.heapLength = layout.cpuSizeBytes;
    texture->_state.pinned = !placement;
    setLayout(texture, layout);
    wire::HeapCreateTexture request;
    request.heap = HeapHandle(heap);
    request.texture = texture->_state.handle;
    request.descriptor = d;
    wire::Reply<wire::HeapCreateTextureReply> reply;
    DeviceConnection(device)->Call(heap, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(heap, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
    texture->_state.created = true;
    return texture;
}

// What a view is made of. The pixel-format forms take the parent's type, all its levels and all its slices
// (-[AGXTexture initWithTexture:pixelFormat:resourceIndex:], AGXMetalA10 0x21c9d56a4: levels (0, _mipmapLevelCount),
// slices (0, _arrayLength * _numFaces)); the forms without a swizzle keep the parent's. The resource index (0 where it
// is not given) reaches initImpl, which does not read it; the host receives it with the request.
struct ViewArguments {
    MTLPixelFormat pixelFormat;
    MTLTextureType textureType;
    NSRange levels;
    NSRange slices;
    bool hasSwizzle;
    MTLTextureSwizzleChannels swizzle;
};

static ViewArguments wholeView(WebMetalTexture *parent, MTLPixelFormat pixelFormat)
{
    return ViewArguments{pixelFormat,
                         parent->_textureType,
                         NSMakeRange(0, parent->_mipmapLevelCount),
                         NSMakeRange(0, parent->_arrayLength * parent->_numFaces),
                         false,
                         MTLTextureSwizzleChannels{}};
}

// -[AGXTexture initWithTexture:pixelFormat:textureType:levels:slices:swizzle:resourceIndex:] (AGXMetalA10 0x21c9d57f4)
// and its form without a swizzle (0x21c9d5714): the AGX layout of the view first, which is nil when the parent is
// memoryless or the view's layout is not valid; then -[IOGPUMetalTexture initWithTextureInternal:pixelFormat:
// textureType:levels:slices:swizzle:compressedView:NO] (IOGPU 0x1fcb2762c), reached through initWithTexture:pixelFormat:
// textureType:levels:slices:swizzle: (0x1fcb3aa04) or, without a swizzle, initWithTexture:pixelFormat:textureType:
// levels:slices: (0x1fcb27594), which passes the parent's -swizzle. (updateBindDataWithTexture:... and
// finalizeTextureCreation then fill the GPU descriptors, which nothing the plugin answers reads.)
static WebMetalTexture *newView(WebMetalTexture *parent, SEL _cmd, const ViewArguments &a)
{
    // -[AGXA10FamilyTexture initImplWithTexture:...] (0x21c98172c, 0x21c981074) builds the view's layout from the
    // parent's layout object, with the swizzle the method was given (its channels, as the u32 they pack into) or the
    // parent's layout swizzle (+0xbc).
    if (!parent->_hasLayout)
        WMLoudMethod(parent, _cmd, "a view of a texture over an IOSurface, a buffer or caller memory: the A10X layout "
                                   "(AGX::TextureGen2) of its parent is not ported");
    uint32_t layoutSwizzle = parent->_layout.swizzle;
    if (a.hasSwizzle)
        std::memcpy(&layoutSwizzle, &a.swizzle, sizeof layoutSwizzle);
    const agx::TextureAllocation layout =
        agx::initImplView(parent->_layout, a.pixelFormat, a.textureType, a.levels.location, a.levels.length,
                          a.slices.length, layoutSwizzle, LayoutSystemValues());
    if (!layout.initialized)
        return nil;

    // -[IOGPUMetalTexture initWithTextureInternal:...]: Metal's checks of the swizzle and of the view's arguments.
    WebMetalDevice *device = parent->_state.device;
    const uint32_t key = MTLTextureSwizzleChannelsToKey(a.hasSwizzle ? a.swizzle : parent.swizzle);
    _mtlValidateMTLTextureSwizzleKey(key);
    _mtlValidateArgumentsForTextureViewOnDevice(device, parent, a.pixelFormat, a.textureType, a.levels.location,
                                                a.levels.length, a.slices.location, a.slices.length, NO);
    WebMetalTexture *view = [WebMetalTexture new];
    setLayout(view, layout);

    // -[IOGPUMetalResource initWithResource:] (IOGPU 0x1fcb2acc8) takes the parent's device and shares its kernel
    // resource (_res.resourceRef, retained), whose info it copies (the IOSurface and the resource size; the kernel
    // resource's protection options are then the parent's); it copies the options, storage and CPU cache modes and the
    // purgeable state, takes getpid() as the responsible process, and makes the view not purgeable. The view has no
    // label of its own.
    ResourceState &state = view->_state;
    const ResourceState &from = parent->_state;
    state.device = device;
    state.iosurface = from.iosurface ? (IOSurfaceRef)CFRetain(from.iosurface) : nullptr;
    state.residentSize = from.residentSize;
    state.protectionOptions = from.protectionOptions;
    state.options = from.options;
    state.storageMode = from.storageMode;
    state.cpuCacheMode = from.cpuCacheMode;
    state.responsibleProcess = getpid();
    state.purgeableState = from.purgeableState;
    state.purgeableAllowed = false;

    view->_parentTexture = parent;
    view->_buffer = parent->_buffer;
    view->_parentRelativeLevel = a.levels.location;
    view->_parentRelativeSlice = a.slices.location;
    if (view->_buffer && a.levels.length != 1)
        WMLoudMethod(parent, _cmd, "21H461 IOGPU asserts levelRange.length == 1 for a view of a buffer's texture");
    if (view->_buffer && a.slices.length != 1)
        WMLoudMethod(parent, _cmd, "21H461 IOGPU asserts sliceRange.length == 1 for a view of a buffer's texture");
    view->_bufferOffset = parent->_bufferOffset;
    view->_bufferBytesPerRow = parent->_bufferBytesPerRow;
    view->_textureType = a.textureType;
    // The parent's size at the view's first level (a 64-bit shift, by the level modulo 64), at least 1.
    const unsigned shift = a.levels.location & 63;
    view->_width = std::max<NSUInteger>(parent->_width >> shift, 1);
    view->_height = std::max<NSUInteger>(parent->_height >> shift, 1);
    view->_depth = std::max<NSUInteger>(parent->_depth >> shift, 1);
    view->_pixelFormat = a.pixelFormat;
    view->_usage = parent->_usage;
    view->_numFaces = a.textureType - MTLTextureTypeCube < 2 ? 6 : 1;
    view->_arrayLength = a.slices.length / view->_numFaces;
    view->_mipmapLevelCount = a.levels.length;
    view->_sampleCount = parent->_sampleCount;
    view->_rotation = parent->_rotation;
    view->_swizzle = key;
    view->_writeSwizzleEnabled = parent->_writeSwizzleEnabled;
    // A swizzled view on a device that supports swizzling loses the usages swizzling disallows
    // (MTLGetDisallowedTextureUsagesWhenSwizzling: write and render target unless write swizzle is enabled).
    if ([(id<WMTextureSwizzleSupport>)device supportsTextureSwizzle] && key != 0x05040302) {
        if (view->_usage == MTLTextureUsageUnknown)
            WMLoudMethod(parent, _cmd, "21H461 IOGPU asserts _usage != MTLTextureUsageUnknown for a swizzled view");
        view->_usage &= ~MTLGetDisallowedTextureUsagesWhenSwizzling(device, view->_writeSwizzleEnabled);
        _mtlValidateTextureUsage(view->_usage);
    }
    view->_shareable = false;
    view->_framebufferOnly = parent->_framebufferOnly;
    view->_iosurfacePlane = parent->_iosurfacePlane;
    view->_isDrawable = parent->_isDrawable;
    view->_allowGPUOptimizedContents = parent->_allowGPUOptimizedContents;
    // The parent's heap (retained), its range in it and whether it is pinned. (IOGPU asserts that the view's
    // _iosurface, copied from the parent, is the surface of the shared kernel resource: the plugin keeps one field for
    // both.)
    state.heap = from.heap;
    state.heapOffset = from.heapOffset;
    state.heapLength = from.heapLength;
    state.pinned = from.pinned;
    // The host holds the view as its own object over the parent's contents.
    state.handle = DeviceConnection(device)->NewHandle();
    return view;
}

// -[AGXTexture newTextureViewWithPixelFormat:...] (AGXMetalA10 0x21c9d59a4 and the five that follow):
// [[[self class] alloc] initWithTexture:self ...]. The host creates the view (CreateTextureView and the other view
// requests).
template <class ReplyFields, class Request>
static WebMetalTexture *NewTextureView(WebMetalTexture *parent, SEL _cmd, const ViewArguments &a, Request request)
{
    WebMetalTexture *view = newView(parent, _cmd, a);
    if (!view)
        return nil;
    request.texture = parent->_state.handle;
    request.view = view->_state.handle;
    wire::Reply<ReplyFields> reply;
    DeviceConnection(view->_state.device)->Call(parent, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(parent, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
    view->_state.created = true;
    return view;
}

static wire::Range wireRange(NSRange range)
{
    wire::Range r;
    r.location = range.location;
    r.length = range.length;
    return r;
}

static wire::TextureSwizzleChannels wireSwizzle(MTLTextureSwizzleChannels swizzle)
{
    wire::TextureSwizzleChannels s;
    s.red = swizzle.red;
    s.green = swizzle.green;
    s.blue = swizzle.blue;
    s.alpha = swizzle.alpha;
    return s;
}

}  // namespace webmetal

namespace webmetal {

uint64_t KernelResourceKey(id resource)
{
    if ([resource isKindOfClass:[WebMetalHeap class]])
        return HeapHandle(static_cast<WebMetalHeap *>(resource));
    if ([resource isKindOfClass:[WebMetalTexture class]]) {
        WebMetalTexture *root = static_cast<WebMetalTexture *>(resource);
        while (root->_parentTexture)
            root = root->_parentTexture;
        if (root->_buffer)
            return KernelResourceKey(root->_buffer);
        resource = root;
    }
    const ResourceState *state = StateOf(resource);
    if (!state)
        return 0;
    return state->heap ? HeapHandle(static_cast<WebMetalHeap *>(state->heap)) : state->handle;
}

}  // namespace webmetal

@implementation WebMetalTexture

// -[IOGPUMetalTexture dealloc] (IOGPU 0x1fcb2b634) releases the parent and the buffer, runs a no-copy deallocator with
// the pointer and the length, releases the surface, and leaves the kernel resource to -[IOGPUMetalResource dealloc];
// here Release goes first so that the host has stopped using caller memory when it is handed back. A heap resource that is not memoryless makes itself aliasable, and gives its range back
// to the heap unless it is a view or a buffer's texture: a view's range is its parent's, which the view's
// makeAliasable unpins. A view the host never created was never handed out and leaves its parent's range alone. A view
// shares its parent's kernel resource and adds nothing to the device's total.
- (void)dealloc
{
    using namespace webmetal;
    if (_state.created && !_state.heap && !_parentTexture) {
        if (_hasLayout)
            DeviceAddAllocation(_state.device, -_accounted);
        else
            DeviceAddUnsized(_state.device, -1);
    }
    if (_state.heap && (_state.options & kStorageModeMask) != (uint64_t(MTLStorageModeMemoryless) << kStorageModeShift) &&
        (_state.created || !_parentTexture)) {
        MakeAliasable(self, _cmd, _state, true);
        if (!_buffer && !_parentTexture)
            [(WebMetalHeap *)_state.heap deallocHeapSubResource];
    }
    ReleaseObject(_state, "-[WebMetalTexture dealloc]");
    if (_deallocator) {
        _deallocator(_pointer, _length);
        _deallocator = nil;
    }
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

- (BOOL)isAliasable
{
    return !_state.pinned;
}

// -[IOGPUMetalResource isComplete] (0x1fcb39934) and -waitUntilComplete (0x1fcb39964), on the texture's kernel resource:
// a view shares its parent's and a texture over a buffer the buffer's (-[IOGPUMetalResource initWithResource:]); a heap
// texture's state carries the heap's options; a sub-allocated texture's own kernel resource takes the hazard-tracking
// mode of its primary buffer (-[IOGPUMetalTexture initWithBuffer:...isStrideTexture:] 0x1fcb2e508), untracked exactly
// when the texture's options are.
static BOOL checkTextureEvent(WebMetalTexture *self, SEL _cmd, bool wait)
{
    WebMetalTexture *root = self;
    while (root->_parentTexture)
        root = root->_parentTexture;
    const webmetal::ResourceState *kernel = root->_buffer ? webmetal::StateOf(root->_buffer) : &root->_state;
    return webmetal::CheckEventComplete(self, _cmd, webmetal::KernelResourceKey(self), *kernel,
                                        root->_state.iosurface != nullptr, wait);
}

- (BOOL)isComplete
{
    return checkTextureEvent(self, _cmd, false);
}

- (void)waitUntilComplete
{
    checkTextureEvent(self, _cmd, true);
}

- (void)makeAliasable
{
    webmetal::MakeAliasable(self, _cmd, _state, false);
}

// -[IOGPUMetalTexture allocatedSize] (0x1fcb3a7bc): a sub-allocated texture's length (its CPU size); otherwise
// -[IOGPUMetalResource allocatedSize] (0x1fcb3a10c): a heap texture's range length, or the kernel resource's size
// (-resourceSize, 0x1fcb2e758: _res.info.resourceSize). A view has no sub-allocation of its own (_primaryBuffer is not
// copied) and answers with its parent's heap range or the parent's kernel resource size, which it copied.
- (NSUInteger)allocatedSize
{
    if (_suballocated)
        return _cpuSizeBytes;
    if (_state.heap)
        return _state.heapLength;
    if (!_hasLayout)
        WMLoudMethod(self, _cmd, kLayout);
    return _state.residentSize;
}

// -[IOGPUMetalResource resourceSize] (0x1fcb2e758): _res.info.resourceSize, its low 56 bits. A view, a texture over a
// buffer and a heap texture take their parent's, their buffer's and the heap's backing resource's _res, info included
// (-[IOGPUMetalResource initWithResource:] 0x1fcb2acc8, from -[IOGPUMetalTexture initWithTextureInternal:...],
// initWithBuffer:descriptor:offset:bytesPerRow: and initWithHeap:...); a sub-allocated texture's own kernel resource
// has its primary buffer's resident size (kSuballocatedResourceSize).
- (NSUInteger)resourceSize
{
    if (_parentTexture)
        return [_parentTexture resourceSize];
    if (_buffer)
        return [(id<WebMetalResourceSize>)_buffer resourceSize];
    if (_state.heap)
        return [_state.heap currentAllocatedSize] & 0x00ffffffffffffffull;
    if (!_hasLayout)
        WMLoudMethod(self, _cmd, kLayout);
    return _state.residentSize & 0x00ffffffffffffffull;
}

// -[AGXTexture getAlignment], getCPUSizeBytes, getBytesPerRow (AGXMetalA10 0x21c9d5da0, 0x21c9d5dc0, 0x21c9d5de0): the
// layout's answers initImpl stored.
- (uint64_t)getAlignment
{
    if (!_hasLayout)
        WMLoudMethod(self, _cmd, kLayout);
    return _alignment;
}

- (uint64_t)getCPUSizeBytes
{
    if (!_hasLayout)
        WMLoudMethod(self, _cmd, kLayout);
    return _cpuSizeBytes;
}

- (uint64_t)getBytesPerRow
{
    if (!_hasLayout)
        WMLoudMethod(self, _cmd, kLayout);
    return _bytesPerRow;
}

- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state
{
    return webmetal::SetPurgeableState(self, _cmd, _state, state);
}

- (BOOL)isPurgeable
{
    return [self setPurgeableState:MTLPurgeableStateKeepCurrent] > MTLPurgeableStateNonVolatile;
}

// -[IOGPUMetalResource protectionOptions] (0x1fcb2df48): the kernel resource's, which the descriptor's are.
- (uint64_t)protectionOptions
{
    return _state.protectionOptions;
}

- (int)responsibleProcess
{
    return _state.responsibleProcess;
}

- (void)setResponsibleProcess:(int)pid
{
    _state.responsibleProcess = pid;
}

- (BOOL)doesAliasResource:(id<MTLResource>)resource
{
    webmetal::ResourceState *other = webmetal::StateOf(resource);
    return other && webmetal::DoesAlias(_state, *other);
}

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

- (BOOL)doesAliasAnyResources:(const id<MTLResource> *)resources count:(NSUInteger)count
{
    for (NSUInteger i = 0; i < count; i++) {
        webmetal::ResourceState *other = webmetal::StateOf(resources[i]);
        if (other && webmetal::DoesAlias(_state, *other))
            return YES;
    }
    return NO;
}

// -[IOGPUMetalResource setOwnerWithIdentity:] (0x1fcb30d70); IOGPUMetalTexture does not override it.
- (int)setOwnerWithIdentity:(uint32_t)identity
{
    return webmetal::SetOwnerWithIdentity(self, _cmd, _state, identity);
}

// MTLTexture -----------------------------------------------------------------------------------------------------------

- (MTLTextureType)textureType
{
    return _textureType;
}

- (MTLPixelFormat)pixelFormat
{
    return _pixelFormat;
}

- (NSUInteger)width
{
    return _width;
}

- (NSUInteger)height
{
    return _height;
}

- (NSUInteger)depth
{
    return _depth;
}

- (NSUInteger)mipmapLevelCount
{
    return _mipmapLevelCount;
}

- (NSUInteger)sampleCount
{
    return _sampleCount;
}

- (NSUInteger)arrayLength
{
    return _arrayLength;
}

- (NSUInteger)numFaces
{
    return _numFaces;
}

- (MTLTextureUsage)usage
{
    return _usage;
}

- (NSUInteger)rotation
{
    return _rotation;
}

// -[IOGPUMetalTexture swizzle] (0x1fcb2761c): the swizzle key's channels (MTLTextureSwizzleKeyToChannels).
- (MTLTextureSwizzleChannels)swizzle
{
    return MTLTextureSwizzleKeyToChannels(_swizzle);
}

// -[IOGPUMetalTexture swizzleKey] (0x1fcb3aa6c).
- (uint32_t)swizzleKey
{
    return _swizzle;
}

- (BOOL)isFramebufferOnly
{
    return _framebufferOnly;
}

- (BOOL)isShareable
{
    return _shareable;
}

- (BOOL)isDrawable
{
    return _isDrawable;
}

- (BOOL)isCompressed
{
    return _isCompressed;
}

// -[AGXA10FamilyTexture isMemoryless] (0x21c98231c): the storage mode is memoryless.
- (BOOL)isMemoryless
{
    return _state.storageMode == MTLStorageModeMemoryless;
}

- (BOOL)allowGPUOptimizedContents
{
    return _allowGPUOptimizedContents;
}

- (id<MTLTexture>)parentTexture
{
    return _parentTexture;
}

- (NSUInteger)parentRelativeLevel
{
    return _parentRelativeLevel;
}

- (NSUInteger)parentRelativeSlice
{
    return _parentRelativeSlice;
}

- (id<MTLBuffer>)buffer
{
    return _buffer;
}

- (NSUInteger)bufferOffset
{
    return _bufferOffset;
}

- (NSUInteger)bufferBytesPerRow
{
    return _bufferBytesPerRow;
}

// -[IOGPUMetalTexture rootResource] (0x1fcb3a798): the parent texture, else the buffer.
- (id<MTLResource>)rootResource
{
    if (_parentTexture)
        return _parentTexture;
    return _buffer;
}

// -[AGXTexture backingResource] (0x21c9d5d34): the buffer, else the parent's backing resource, else the texture itself.
- (id)backingResource
{
    if (_buffer)
        return _buffer;
    if (_parentTexture)
        return [_parentTexture backingResource];
    return self;
}

- (IOSurfaceRef)iosurface
{
    return _state.iosurface;
}

- (NSUInteger)iosurfacePlane
{
    return _iosurfacePlane;
}

// -[IOGPUMetalResource virtualAddress] (0x1fcb28048): the kernel resource's CPU mapping, in the AGX layout.
- (void *)virtualAddress
{
    WMLoudMethod(self, _cmd, "a texture's memory in the AGX layout has no guest mapping");
}

// Views ----------------------------------------------------------------------------------------------------------------

// -[AGXTexture newTextureViewWithPixelFormat:] (AGXMetalA10 0x21c9d59a4).
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat
{
    using namespace webmetal;
    wire::CreateTextureView request;
    request.pixelFormat = pixelFormat;
    return NewTextureView<wire::CreateTextureViewReply>(self, _cmd, wholeView(self, pixelFormat), request);
}

// -[AGXTexture newTextureViewWithPixelFormat:textureType:levels:slices:] (AGXMetalA10 0x21c9d59d4).
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat
                                    textureType:(MTLTextureType)textureType
                                         levels:(NSRange)levels
                                         slices:(NSRange)slices
{
    using namespace webmetal;
    wire::CreateTextureViewLevelsSlices request;
    request.pixelFormat = pixelFormat;
    request.textureType = textureType;
    request.levelRange = wireRange(levels);
    request.sliceRange = wireRange(slices);
    const ViewArguments a{pixelFormat, textureType, levels, slices, false, MTLTextureSwizzleChannels{}};
    return NewTextureView<wire::CreateTextureViewLevelsSlicesReply>(self, _cmd, a, request);
}

// -[AGXTexture newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:] (AGXMetalA10 0x21c9d5a4c).
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat
                                    textureType:(MTLTextureType)textureType
                                         levels:(NSRange)levels
                                         slices:(NSRange)slices
                                        swizzle:(MTLTextureSwizzleChannels)swizzle
{
    using namespace webmetal;
    wire::CreateTextureViewLevelsSlicesSwizzle request;
    request.pixelFormat = pixelFormat;
    request.textureType = textureType;
    request.levelRange = wireRange(levels);
    request.sliceRange = wireRange(slices);
    request.swizzle = wireSwizzle(swizzle);
    const ViewArguments a{pixelFormat, textureType, levels, slices, true, swizzle};
    return NewTextureView<wire::CreateTextureViewLevelsSlicesSwizzleReply>(self, _cmd, a, request);
}

// -[AGXTexture newTextureViewWithPixelFormat:resourceIndex:] (AGXMetalA10 0x21c9d5acc).
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat resourceIndex:(NSUInteger)resourceIndex
{
    using namespace webmetal;
    wire::CreateTextureViewResourceIndex request;
    request.pixelFormat = pixelFormat;
    request.resourceIndex = resourceIndex;
    return NewTextureView<wire::CreateTextureViewResourceIndexReply>(self, _cmd, wholeView(self, pixelFormat), request);
}

// -[AGXTexture newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:] (AGXMetalA10 0x21c9d5b0c).
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat
                                    textureType:(MTLTextureType)textureType
                                         levels:(NSRange)levels
                                         slices:(NSRange)slices
                                  resourceIndex:(NSUInteger)resourceIndex
{
    using namespace webmetal;
    wire::CreateTextureViewLevelsSlicesResourceIndex request;
    request.pixelFormat = pixelFormat;
    request.textureType = textureType;
    request.levels = wireRange(levels);
    request.slices = wireRange(slices);
    request.resourceIndex = resourceIndex;
    const ViewArguments a{pixelFormat, textureType, levels, slices, false, MTLTextureSwizzleChannels{}};
    return NewTextureView<wire::CreateTextureViewLevelsSlicesResourceIndexReply>(self, _cmd, a, request);
}

// -[AGXTexture newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:] (AGXMetalA10
// 0x21c9d5b8c).
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat
                                    textureType:(MTLTextureType)textureType
                                         levels:(NSRange)levels
                                         slices:(NSRange)slices
                                        swizzle:(MTLTextureSwizzleChannels)swizzle
                                  resourceIndex:(NSUInteger)resourceIndex
{
    using namespace webmetal;
    wire::CreateTextureViewLevelsSlicesSwizzleResourceIndex request;
    request.pixelFormat = pixelFormat;
    request.textureType = textureType;
    request.levels = wireRange(levels);
    request.slices = wireRange(slices);
    request.swizzle = wireSwizzle(swizzle);
    request.resourceIndex = resourceIndex;
    const ViewArguments a{pixelFormat, textureType, levels, slices, true, swizzle};
    return NewTextureView<wire::CreateTextureViewLevelsSlicesSwizzleResourceIndexReply>(self, _cmd, a, request);
}

// CPU access -----------------------------------------------------------------------------------------------------------

// -[IOGPUMetalTexture replaceRegion:mipmapLevel:withBytes:bytesPerRow:] (IOGPU 0x1fcb2e204): the slice form with slice 0
// and image pitch 0.
- (void)replaceRegion:(MTLRegion)region
          mipmapLevel:(NSUInteger)level
            withBytes:(const void *)bytes
          bytesPerRow:(NSUInteger)bytesPerRow
{
    [self replaceRegion:region mipmapLevel:level slice:0 withBytes:bytes bytesPerRow:bytesPerRow bytesPerImage:0];
}

// AGX::Texture::validateRegion (AGXMetalA10 0x21c983ef4), which writeRegion and readRegion start with, on the texture's
// layout object: a copy it refuses is logged (agx::logRegionRefusal) and copies nothing. The host copies the regions it
// accepts as the driver does (Memory::cpu_region). A texture over an IOSurface or a buffer has no ported layout; the
// host answers its copies by its own rules.
static bool refusedRegion(WebMetalTexture *texture, MTLRegion region, NSUInteger level, NSUInteger slice,
                          NSUInteger bytesPerRow)
{
    if (!texture->_hasLayout)
        return false;
    webmetal::agx::RegionCopy c;
    c.x = static_cast<uint32_t>(region.origin.x);
    c.y = static_cast<uint32_t>(region.origin.y);
    c.z = static_cast<uint32_t>(region.origin.z);
    c.width = static_cast<uint32_t>(region.size.width);
    c.height = static_cast<uint32_t>(region.size.height);
    c.depth = static_cast<uint32_t>(region.size.depth);
    c.slice = static_cast<uint32_t>(slice);
    c.level = static_cast<uint32_t>(level);
    c.bytesPerRow = bytesPerRow;
    const char *refusal = webmetal::agx::validateRegion(texture->_layout, c);
    if (!refusal)
        return false;
    webmetal::agx::logRegionRefusal(refusal);
    return true;
}

// -[AGXA10FamilyTexture replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:] (AGXMetalA10 0x21c9819fc)
// writes the region into the texture's memory on the CPU (AGX::Texture::writeRegion, 0x21c982fe0, with each region
// field, the slice and the level as unsigned int); the host holds that memory.
- (void)replaceRegion:(MTLRegion)region
          mipmapLevel:(NSUInteger)level
                slice:(NSUInteger)slice
            withBytes:(const void *)bytes
          bytesPerRow:(NSUInteger)bytesPerRow
        bytesPerImage:(NSUInteger)bytesPerImage
{
    using namespace webmetal;
    if (refusedRegion(self, region, level, slice, bytesPerRow))
        return;
    wire::ReplaceRegionSlice request;
    request.texture = _state.handle;
    request.region.origin.x = region.origin.x;
    request.region.origin.y = region.origin.y;
    request.region.origin.z = region.origin.z;
    request.region.size.width = region.size.width;
    request.region.size.height = region.size.height;
    request.region.size.depth = region.size.depth;
    request.level = level;
    request.slice = slice;
    request.pixelBytes = reinterpret_cast<uint64_t>(bytes);
    request.bytesPerRow = bytesPerRow;
    request.bytesPerImage = bytesPerImage;
    wire::Reply<wire::ReplaceRegionSliceReply> reply;
    DeviceConnection(_state.device)->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
}

// -[IOGPUMetalTexture getBytes:bytesPerRow:fromRegion:mipmapLevel:] (IOGPU 0x1fcb31158): the slice form with slice 0 and
// an image pitch of the region's height times the row pitch.
- (void)getBytes:(void *)bytes bytesPerRow:(NSUInteger)bytesPerRow fromRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level
{
    [self getBytes:bytes bytesPerRow:bytesPerRow bytesPerImage:region.size.height * bytesPerRow fromRegion:region
        mipmapLevel:level slice:0];
}

// -[AGXA10FamilyTexture getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:] (AGXMetalA10 0x21c98199c)
// reads the region from the texture's memory on the CPU (AGX::Texture::readRegion, 0x21c9a4e84, with each region field,
// the slice and the level as unsigned int); the host holds that memory.
- (void)getBytes:(void *)bytes
      bytesPerRow:(NSUInteger)bytesPerRow
    bytesPerImage:(NSUInteger)bytesPerImage
       fromRegion:(MTLRegion)region
      mipmapLevel:(NSUInteger)level
            slice:(NSUInteger)slice
{
    using namespace webmetal;
    if (refusedRegion(self, region, level, slice, bytesPerRow))
        return;
    wire::GetBytesSlice request;
    request.texture = _state.handle;
    request.pixelBytes = reinterpret_cast<uint64_t>(bytes);
    request.bytesPerRow = bytesPerRow;
    request.bytesPerImage = bytesPerImage;
    request.region.origin.x = region.origin.x;
    request.region.origin.y = region.origin.y;
    request.region.origin.z = region.origin.z;
    request.region.size.width = region.size.width;
    request.region.size.height = region.size.height;
    request.region.size.depth = region.size.depth;
    request.level = level;
    request.slice = slice;
    wire::Reply<wire::GetBytesSliceReply> reply;
    DeviceConnection(_state.device)->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
}

@end
