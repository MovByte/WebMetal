// The device's resource methods: AGXA10FamilyDevice's (and IOGPUMetalDevice's) factories and accounting, as a category
// of WebMetalDevice.

#import "WebMetalBuffer.h"

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
// methods).
- (NSUInteger)currentAllocatedSize
{
    return static_cast<NSUInteger>(webmetal::DeviceAllocatedSize(self));
}

@end
