// Buffers (AGXA10FamilyBuffer over AGXBuffer): how the device and heaps create them.

#pragma once

#import "WebMetalResources.h"

namespace webmetal {

// -[AGXA10FamilyDevice newBufferWithLength:options:] and -[AGXA10FamilyDevice newBufferWithBytes:length:options:]
// (bytes non-null): a buffer the device allocates.
AGXA10FamilyBuffer *NewDeviceBuffer(AGXA10FamilyDevice *device, SEL _cmd, NSUInteger length, MTLResourceOptions options,
                                    const void *bytes);

// -[AGXA10FamilyDevice newBufferWithBytesNoCopy:length:options:deallocator:]: a buffer over the caller's memory.
AGXA10FamilyBuffer *NewNoCopyBuffer(AGXA10FamilyDevice *device, SEL _cmd, void *pointer, NSUInteger length,
                                    MTLResourceOptions options, void (^deallocator)(void *pointer, NSUInteger length));

}  // namespace webmetal
