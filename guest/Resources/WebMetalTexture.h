// Textures (AGXA10FamilyTexture over AGXTexture): how the device and buffers create them.

#pragma once

#import "WebMetalResources.h"
#import "WebMetalTextureLayout.h"

// AGXTexture adopts what 21H461's AGXTexture and IOGPUMetalResource do; AGXA10FamilyTexture is the class of every
// texture the plugin creates.
@interface AGXTexture : _MTLResource <MTLTexture, MTLTextureSPI, MTLTextureImplementation, MTLResourceSPI>
@end

@interface AGXA10FamilyTexture : AGXTexture
@end

namespace webmetal {

// -[AGXA10FamilyDevice newTextureWithDescriptor:] (AGXMetalA10 0x21c8d64a4).
AGXTexture *NewDeviceTexture(AGXA10FamilyDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor);

// -[AGXA10FamilyDevice newTextureWithBytesNoCopy:length:descriptor:deallocator:] (AGXMetalA10 0x21c8d62c8).
AGXTexture *NewNoCopyTexture(AGXA10FamilyDevice *device, SEL _cmd, void *bytes, NSUInteger length,
                             MTLTextureDescriptor *descriptor, void (^deallocator)(void *pointer, NSUInteger length));

// -[AGXA10FamilyDevice newTextureWithDescriptor:iosurface:plane:] and its slice form.
AGXTexture *NewSurfaceTexture(AGXA10FamilyDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor,
                              IOSurfaceRef iosurface, NSUInteger plane, NSUInteger slice);

// -[AGXA10FamilyBuffer newTextureWithDescriptor:offset:bytesPerRow:] (AGXMetalA10 0x21c8c6374) and
// -[IOGPUMetalBuffer newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:] (IOGPU 0x1fcb2dfd8, linear).
AGXTexture *NewBufferTexture(AGXA10FamilyBuffer *buffer, SEL _cmd, MTLTextureDescriptor *descriptor, NSUInteger offset,
                             NSUInteger bytesPerRow, bool linear, NSUInteger bytesPerImage);

// -[AGXA10FamilyHeap newTextureWithDescriptor:] (AGXMetalA10 0x21c91a050) and newTextureWithDescriptor:atOffset:
// (0x21c91a0d0, `placement`).
AGXTexture *NewHeapTexture(AGXA10FamilyHeap *heap, SEL _cmd, MTLTextureDescriptor *descriptor, bool placement,
                           NSUInteger offset);

// -[AGXA10FamilyDevice heapTextureSizeAndAlignWithDescriptor:] (AGXMetalA10 0x21c8d6694): {0, 0} for a descriptor
// Metal's validation refuses, else the AGX layout's size and alignment for a heap texture.
struct SizeAndAlign {
    uint64_t size;
    uint64_t align;
};
SizeAndAlign HeapTextureSizeAndAlign(AGXA10FamilyDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor);

// The inputs of the AGX layout port: the descriptor fields its callers read, and the running system's values.
agx::DescriptorFields LayoutDescriptorFields(const wire::TextureDescriptor &d);
agx::SystemValues LayoutSystemValues();

// -[AGXA10FamilyDevice newTextureLayoutWithDescriptor:isHeapOrBufferBacked:] (AGXMetalA10 0x21c8d8fd4)
// (WebMetalTextureLayoutObject.mm).
id NewTextureLayout(AGXA10FamilyDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor, bool heapOrBufferBacked);

// The wire form of a texture descriptor: every field the schema carries, read with the descriptor's own getters.
wire::TextureDescriptor WireTextureDescriptor(id self, SEL _cmd, MTLTextureDescriptor *descriptor);

// The A10X layout (AGX::TextureGen2, the texture's _impl) of a plugin texture: null for another object and for a texture
// whose layout is not ported (over an IOSurface, or a texture-buffer texture over a buffer).
const agx::TextureGen2 *TextureLayoutOf(id texture);

// A buffer's handle, CPU contents and resource state (WebMetalBuffer.mm).
uint64_t BufferHandle(AGXBuffer *buffer);
void *BufferContents(AGXBuffer *buffer);

}  // namespace webmetal
