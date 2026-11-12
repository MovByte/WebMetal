// Textures (WebMetalTexture, standing in for AGXA10FamilyTexture): how the device and buffers create them.

#pragma once

#import "WebMetalResources.h"

@interface WebMetalTexture : _MTLResource <MTLTexture, MTLTextureSPI, MTLTextureImplementation, MTLResourceSPI>
@end

namespace webmetal {

// -[AGXA10FamilyDevice newTextureWithDescriptor:] (AGXMetalA10 0x21c8d64a4).
WebMetalTexture *NewDeviceTexture(WebMetalDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor);

// -[AGXA10FamilyDevice newTextureWithDescriptor:iosurface:plane:] and its slice form.
WebMetalTexture *NewSurfaceTexture(WebMetalDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor,
                                   IOSurfaceRef iosurface, NSUInteger plane, NSUInteger slice);

// -[AGXA10FamilyDevice newTextureWithBytesNoCopy:length:descriptor:deallocator:] (AGXMetalA10 0x21c8d62cc).
WebMetalTexture *NewNoCopyTexture(WebMetalDevice *device, SEL _cmd, void *bytes, NSUInteger length,
                                  MTLTextureDescriptor *descriptor, void (^deallocator)(void *pointer, NSUInteger length));

// -[AGXA10FamilyBuffer newTextureWithDescriptor:offset:bytesPerRow:] (AGXMetalA10 0x21c8c6374) and
// -[IOGPUMetalBuffer newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:] (IOGPU 0x1fcb2dfd8, linear).
WebMetalTexture *NewBufferTexture(WebMetalBuffer *buffer, SEL _cmd, MTLTextureDescriptor *descriptor, NSUInteger offset,
                                  NSUInteger bytesPerRow, bool linear, NSUInteger bytesPerImage);

// The wire form of a texture descriptor: every field the schema carries, read with the descriptor's own getters.
wire::TextureDescriptor WireTextureDescriptor(id self, SEL _cmd, MTLTextureDescriptor *descriptor);

// A buffer's handle, CPU contents and resource state (WebMetalBuffer.mm).
uint64_t BufferHandle(WebMetalBuffer *buffer);
void *BufferContents(WebMetalBuffer *buffer);

}  // namespace webmetal
