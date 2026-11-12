// The device's resource methods: AGXA10FamilyDevice's (and IOGPUMetalDevice's) factories and accounting, as a category
// of WebMetalDevice.

#import "WebMetalBuffer.h"
#import "WebMetalTexture.h"
#import "WebMetalHeapPrivate.h"

namespace webmetal {
id<MTLSamplerState> NewSamplerState(WebMetalDevice *device, SEL _cmd, MTLSamplerDescriptor *descriptor);
}

// The class adopts the protocols that declare these methods; this category, not the class, implements them.
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

@implementation WebMetalDevice (WebMetalResources)

// -[AGXA10FamilyDevice newBufferWithLength:options:] (AGXMetalA10 0x21c8d5e48). The driver also counts the call in its
// telemetry, which has no observable effect.
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
    return webmetal::NewDeviceBuffer(self, _cmd, length, options, nullptr);
}

// -[AGXA10FamilyDevice newBufferWithBytes:length:options:] (AGXMetalA10 0x21c8d5f0c): the buffer of
// newBufferWithLength:options:, then the bytes copied in (CreateBufferWithBytes carries both steps).
- (id<MTLBuffer>)newBufferWithBytes:(const void *)pointer length:(NSUInteger)length options:(MTLResourceOptions)options
{
    return webmetal::NewDeviceBuffer(self, _cmd, length, options, pointer);
}

// -[AGXA10FamilyDevice newBufferWithBytesNoCopy:length:options:deallocator:] (AGXMetalA10 0x21c8d5ffc).
- (id<MTLBuffer>)newBufferWithBytesNoCopy:(void *)pointer
                                   length:(NSUInteger)length
                                  options:(MTLResourceOptions)options
                              deallocator:(void (^)(void *pointer, NSUInteger length))deallocator
{
    return webmetal::NewNoCopyBuffer(self, _cmd, pointer, length, options, deallocator);
}

// -[IOGPUMetalDevice currentAllocatedSize] (IOGPU 0x1fcb2be10) asks the kernel for the device's allocation total; the
// WebMetal service's total is the sum of the allocated sizes of the device's live resources (wire schema, guest-local
// methods). A live texture's size depends on the AGX layout, which has no source yet.
- (NSUInteger)currentAllocatedSize
{
    if (webmetal::DeviceUnsizedCount(self))
        WMLoudMethod(self, _cmd, "a live texture's allocated size depends on the A10X texture layout, not established");
    return static_cast<NSUInteger>(webmetal::DeviceAllocatedSize(self));
}

// -[AGXA10FamilyDevice newTextureWithDescriptor:] (AGXMetalA10 0x21c8d64a4).
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
{
    return webmetal::NewDeviceTexture(self, _cmd, descriptor);
}

// -[AGXA10FamilyDevice newTextureWithDescriptor:iosurface:plane:] (AGXMetalA10 0x21c8d64e8).
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                 iosurface:(IOSurfaceRef)iosurface
                                     plane:(NSUInteger)plane
{
    return webmetal::NewSurfaceTexture(self, _cmd, descriptor, iosurface, plane, 0);
}

// -[AGXA10FamilyDevice newTextureWithDescriptor:iosurface:plane:slice:] (AGXMetalA10 0x21c8d65bc).
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                 iosurface:(IOSurfaceRef)iosurface
                                     plane:(NSUInteger)plane
                                     slice:(NSUInteger)slice
{
    return webmetal::NewSurfaceTexture(self, _cmd, descriptor, iosurface, plane, slice);
}

// -[AGXA10FamilyDevice newTextureWithBytesNoCopy:length:descriptor:deallocator:] (AGXMetalA10 0x21c8d62cc).
- (id<MTLTexture>)newTextureWithBytesNoCopy:(void *)bytes
                                     length:(NSUInteger)length
                                 descriptor:(MTLTextureDescriptor *)descriptor
                                deallocator:(void (^)(void *pointer, NSUInteger length))deallocator
{
    return webmetal::NewNoCopyTexture(self, _cmd, bytes, length, descriptor, deallocator);
}

// -[AGXA10FamilyDevice newHeapWithDescriptor:] (AGXMetalA10 0x21c8d6918): an AGXA10FamilySparseHeap for
// MTLHeapTypeSparse (2), an AGXA10FamilyHeap otherwise. The driver also counts the call in its telemetry.
- (id<MTLHeap>)newHeapWithDescriptor:(MTLHeapDescriptor *)descriptor
{
    if (descriptor.type == 2)
        WMLoudMethod(self, _cmd, "a sparse heap (AGXA10FamilySparseHeap) is not implemented");
    return webmetal::NewHeap(self, _cmd, descriptor);
}

// -[AGXA10FamilyDevice newSamplerStateWithDescriptor:] (AGXMetalA10 0x21c8d7dc0).
- (id<MTLSamplerState>)newSamplerStateWithDescriptor:(MTLSamplerDescriptor *)descriptor
{
    return webmetal::NewSamplerState(self, _cmd, descriptor);
}

// -[AGXA10FamilyDevice newResourceGroupFromResources:count:] (AGXMetalA10 0x21c8db1f8) answers nil: with the
// disable_drop_hints boot argument set to 1 directly, and otherwise because -[AGXA10FamilyResourceGroup
// initWithDevice:resources:count:] (0x21c8dc920) fails for a count of 0 and, for any other count, its block
// (0x21c8dcbe4) clears _initedResourceGroup before testing it, so the initializer releases the group and returns nil.
- (id)newResourceGroupFromResources:(const id<MTLResource> *)resources count:(NSUInteger)count
{
    return nil;
}

@end
