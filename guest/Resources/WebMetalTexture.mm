// Textures: WebMetalTexture stands in for AGXA10FamilyTexture (AGXA10FamilyTexture -> AGXTexture -> IOGPUMetalTexture ->
// IOGPUMetalResource -> _MTLResource on 21H461).
//
// The driver checks a descriptor with Metal.framework's own -[MTLTextureDescriptorInternal validateWithDevice:], lays
// the texture out in its private memory format (AGX::Texture, the AGX layout), and creates the kernel resource. The
// plugin calls the same validation, and the host executor holds the texture's contents (CreateTexture and the other
// creation requests). What depends on the AGX layout itself, the allocated size, its alignment and row pitch, has no
// source yet: those answers are LOUD.

#import "WebMetalTexture.h"
#import "WebMetalBuffer.h"

#import <objc/message.h>

#include <cstring>

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
}

namespace {

constexpr char kLayout[] = "the A10X texture layout (AGX::Texture in AGXMetalA10) that decides it is not established";

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
    void *_pointer;                      // _pointer: caller memory of a no-copy texture
    void (^_deallocator)(void *pointer, NSUInteger length);  // _deallocator
    NSUInteger _noCopyLength;
}
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
    DeviceAddUnsized(device, 1);
}

// -[AGXTexture initWithDevice:desc:isSuballocDisabled:] (AGXMetalA10 0x21c9d48a8): Metal's descriptor validation, then
// the AGX layout and the kernel resource (CreateTexture).
WebMetalTexture *NewDeviceTexture(WebMetalDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor)
{
    if (![(id<WMDescriptorValidation>)descriptor validateWithDevice:device])
        return nil;
    const wire::TextureDescriptor d = WireTextureDescriptor(device, _cmd, descriptor);
    WebMetalTexture *texture = newTextureObject(device, _cmd, d);
    wire::CreateTexture request;
    request.texture = texture->_state.handle;
    request.descriptor = d;
    create<wire::CreateTexture, wire::CreateTextureReply>(device, _cmd, texture, request);
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

// -[AGXA10FamilyDevice newTextureWithBytesNoCopy:length:descriptor:deallocator:] (AGXMetalA10 0x21c8d62cc) wraps
// caller memory laid out in the AGX format; the deallocator runs when the texture is deallocated.
WebMetalTexture *NewNoCopyTexture(WebMetalDevice *device, SEL _cmd, void *bytes, NSUInteger length,
                                  MTLTextureDescriptor *descriptor, void (^deallocator)(void *pointer, NSUInteger length))
{
    if (![(id<WMDescriptorValidation>)descriptor validateWithDevice:device])
        return nil;
    const wire::TextureDescriptor d = WireTextureDescriptor(device, _cmd, descriptor);
    WebMetalTexture *texture = newTextureObject(device, _cmd, d);
    texture->_pointer = bytes;
    texture->_noCopyLength = length;
    texture->_deallocator = [deallocator copy];
    wire::CreateTextureNoCopy request;
    request.texture = texture->_state.handle;
    request.bytes = reinterpret_cast<uint64_t>(bytes);
    request.length = length;
    request.descriptor = d;
    request.deallocator = deallocator != nil;
    create<wire::CreateTextureNoCopy, wire::CreateTextureNoCopyReply>(device, _cmd, texture, request);
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

}  // namespace webmetal

@implementation WebMetalTexture

// -[IOGPUMetalTexture dealloc] releases the parent, buffer and surface and runs a no-copy deallocator after the kernel
// resource is gone; here Release goes first so that the host has stopped using caller memory when it is handed back.
- (void)dealloc
{
    using namespace webmetal;
    if (_state.created)
        DeviceAddUnsized(_state.device, -1);
    ReleaseObject(_state, "-[WebMetalTexture dealloc]");
    if (_deallocator) {
        _deallocator(_pointer, _noCopyLength);
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

- (void)makeAliasable
{
    webmetal::MakeAliasable(self, _cmd, _state, false);
}

// -[IOGPUMetalResource allocatedSize] (0x1fcb3a10c): a heap texture's range length, or the kernel resource's resident
// size, which the AGX layout decides.
- (NSUInteger)allocatedSize
{
    if (_state.heap)
        return _state.heapLength;
    WMLoudMethod(self, _cmd, kLayout);
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

// -[AGXA10FamilyTexture replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:] (AGXMetalA10 0x21c9819fc)
// writes the region into the texture's memory on the CPU (AGX::Texture::writeRegion); the host holds that memory.
- (void)replaceRegion:(MTLRegion)region
          mipmapLevel:(NSUInteger)level
                slice:(NSUInteger)slice
            withBytes:(const void *)bytes
          bytesPerRow:(NSUInteger)bytesPerRow
        bytesPerImage:(NSUInteger)bytesPerImage
{
    using namespace webmetal;
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
// reads the region from the texture's memory on the CPU (AGX::Texture::readRegion); the host holds that memory.
- (void)getBytes:(void *)bytes
      bytesPerRow:(NSUInteger)bytesPerRow
    bytesPerImage:(NSUInteger)bytesPerImage
       fromRegion:(MTLRegion)region
      mipmapLevel:(NSUInteger)level
            slice:(NSUInteger)slice
{
    using namespace webmetal;
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
