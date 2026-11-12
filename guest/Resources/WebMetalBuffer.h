// Buffers (WebMetalBuffer, standing in for AGXA10FamilyBuffer): how the device and heaps create them.

#pragma once

#import "WebMetalResources.h"

namespace webmetal {

// -[AGXA10FamilyDevice newBufferWithLength:options:] and -[AGXA10FamilyDevice newBufferWithBytes:length:options:]
// (bytes non-null): a buffer the device allocates.
WebMetalBuffer *NewDeviceBuffer(WebMetalDevice *device, SEL _cmd, NSUInteger length, MTLResourceOptions options,
                                const void *bytes);

// -[AGXA10FamilyDevice newBufferWithBytesNoCopy:length:options:deallocator:]: a buffer over the caller's memory.
WebMetalBuffer *NewNoCopyBuffer(WebMetalDevice *device, SEL _cmd, void *pointer, NSUInteger length,
                                MTLResourceOptions options, void (^deallocator)(void *pointer, NSUInteger length));

}  // namespace webmetal
