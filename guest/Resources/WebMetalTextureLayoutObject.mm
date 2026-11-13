// Texture layout objects: WebMetalTextureLayout stands in for AGXTextureLayout (AGXTextureLayout ->
// IOGPUMetalTextureLayout -> NSObject on 21H461), which -[AGXA10FamilyDevice newTextureLayoutWithDescriptor:
// isHeapOrBufferBacked:] (AGXMetalA10 0x21c8d8fd4) makes. It answers the AGX layout of a texture whose memory the caller
// provides: its size (the layout and a 16-byte watermark after it), alignment and copy granularity; and it copies texels
// between linear bytes and such memory, in the layout, as the driver's writeRegion and readRegion do. The host executor
// makes those copies in the calling task's memory (TextureLayoutCopyFromLinearBytes and
// TextureLayoutCopyFromTextureMemory), with the same port of the layout it keeps textures over caller memory with.

#import "WebMetalResources.h"
#import "WebMetalTexture.h"
#import "WebMetalTextureLayout.h"

#include <cstring>

// The class adopts MTLTextureLayout as AGXTextureLayout does and implements only what the 21H461 driver implements; the
// protocol's other methods are LOUD stubs.
#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"

namespace {

// -[AGXTextureLayout watermark] (AGXMetalA10 0x21c9c048c) and the bytes -finalizeTextureMemory: stores (0x21c9e02f0).
constexpr uint64_t kWatermark[2] = {0xf4aa93d1f315e8aeull, 0x6b4dc612a90e09f1ull};

// The driver's return types are unnamed structs (type encodings {?=[2Q]} and {?=QQQ}).
typedef struct {
    uint64_t words[2];
} Watermark;

typedef struct {
    uint64_t width, height, depth;
} Granularity;

}  // namespace

@interface WebMetalTextureLayout : NSObject <MTLTextureLayout>
@end

@interface WebMetalTextureLayout () {
@public
    uint64_t _total;      // AGX::TextureGen2 +0x158: the layout's bytes, where the watermark goes
    uint64_t _alignment;  // the layout's alignment
    uint64_t _blockWidth, _blockHeight;  // AGX::TextureGen2 +0xd0, +0xd4
    // What the layout was made from, for the host's copies: the device, the descriptor and the memory's kind.
    WebMetalDevice *_device;
    webmetal::wire::TextureDescriptor _descriptor;
    bool _heapOrBufferBacked;
}
@end

namespace webmetal {

// -[AGXTextureLayout initWithDevice:descriptor:isHeapOrBufferBacked:] (AGXMetalA10 0x21c9c019c): the layout of the
// descriptor (no validation), twiddled planes allowed unless the memory is a heap's or a buffer's; nil when the layout
// is not valid.
id NewTextureLayout(WebMetalDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor, bool heapOrBufferBacked)
{
    const wire::TextureDescriptor d = WireTextureDescriptor(device, _cmd, descriptor);
    const agx::LayoutResult r = agx::textureLayout(LayoutDescriptorFields(d), heapOrBufferBacked, LayoutSystemValues());
    if (!r.initialized)
        return nil;
    WebMetalTextureLayout *layout = [WebMetalTextureLayout new];
    layout->_total = r.total;
    layout->_alignment = r.alignment;
    layout->_blockWidth = r.blockWidth;
    layout->_blockHeight = r.blockHeight;
    layout->_device = device;
    layout->_descriptor = d;
    layout->_heapOrBufferBacked = heapOrBufferBacked;
    return layout;
}

}  // namespace webmetal

namespace {

webmetal::wire::Region wireRegion(MTLRegion region)
{
    webmetal::wire::Region r;
    r.origin.x = region.origin.x;
    r.origin.y = region.origin.y;
    r.origin.z = region.origin.z;
    r.size.width = region.size.width;
    r.size.height = region.size.height;
    r.size.depth = region.size.depth;
    return r;
}

}  // namespace

@implementation WebMetalTextureLayout

// -[AGXTextureLayout size] (AGXMetalA10 0x21c9c0380): the layout and its watermark.
- (uint64_t)size
{
    return _total + sizeof kWatermark;
}

// -[AGXTextureLayout alignment] (AGXMetalA10 0x21c9c0398).
- (uint64_t)alignment
{
    return _alignment;
}

// -[AGXTextureLayout copyGranularity] (AGXMetalA10 0x21c9c0468): the format's block, one deep.
- (Granularity)copyGranularity
{
    return Granularity{_blockWidth, _blockHeight, 1};
}

- (Watermark)watermark
{
    return Watermark{{kWatermark[0], kWatermark[1]}};
}

// -[AGXTextureLayout initializeTextureMemory:] (AGXMetalA10 0x21c9c04b0) does nothing.
- (void)initializeTextureMemory:(void *)memory
{
}

// -[AGXTextureLayout finalizeTextureMemory:] (AGXMetalA10 0x21c9c0614): the watermark after the layout's bytes.
- (void)finalizeTextureMemory:(void *)memory
{
    std::memcpy(static_cast<uint8_t *>(memory) + _total, kWatermark, sizeof kWatermark);
}

// -[AGXTextureLayout copyFromLinearBytes:linearOffset:linearBytesPerRow:linearBytesPerImage:toTextureMemory:textureSlice:
// textureLevel:textureRegion:] (AGXMetalA10 0x21c9c04b4): AGX::Texture::writeRegion (0x21c982fe0) on the layout's
// object with the memory as its CPU address, from the bytes at bytes + offset.
- (void)copyFromLinearBytes:(const void *)bytes
               linearOffset:(NSUInteger)offset
          linearBytesPerRow:(NSUInteger)bytesPerRow
        linearBytesPerImage:(NSUInteger)bytesPerImage
            toTextureMemory:(void *)memory
               textureSlice:(NSUInteger)slice
               textureLevel:(NSUInteger)level
              textureRegion:(MTLRegion)region
{
    using namespace webmetal;
    const agx::SystemValues system = LayoutSystemValues();
    wire::TextureLayoutCopyFromLinearBytes request;
    request.descriptor = _descriptor;
    request.heapOrBufferBacked = _heapOrBufferBacked;
    request.pageSize = system.vmPageSize;
    request.iosurfaceAlignment = system.iosurfaceAlignment;
    request.bytes = reinterpret_cast<uint64_t>(bytes);
    request.offset = offset;
    request.bytesPerRow = bytesPerRow;
    request.bytesPerImage = bytesPerImage;
    request.memory = reinterpret_cast<uint64_t>(memory);
    request.slice = slice;
    request.level = level;
    request.region = wireRegion(region);
    wire::Reply<wire::TextureLayoutCopyFromLinearBytesReply> reply;
    DeviceConnection(_device)->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
    if (reply.value.refusal)
        webmetal::agx::logRegionRefusal(reply.value.refusal->c_str());
}

// -[AGXTextureLayout copyFromTextureMemory:textureSlice:textureLevel:textureRegion:toLinearBytes:linearOffset:
// linearBytesPerRow:linearBytesPerImage:] (AGXMetalA10 0x21c9c0564): AGX::Texture::readRegion (0x21c9a4e84) on the
// layout's object with the memory as its CPU address, into the bytes at bytes + offset.
- (void)copyFromTextureMemory:(const void *)memory
                 textureSlice:(NSUInteger)slice
                 textureLevel:(NSUInteger)level
                textureRegion:(MTLRegion)region
                toLinearBytes:(void *)bytes
                 linearOffset:(NSUInteger)offset
            linearBytesPerRow:(NSUInteger)bytesPerRow
          linearBytesPerImage:(NSUInteger)bytesPerImage
{
    using namespace webmetal;
    const agx::SystemValues system = LayoutSystemValues();
    wire::TextureLayoutCopyFromTextureMemory request;
    request.descriptor = _descriptor;
    request.heapOrBufferBacked = _heapOrBufferBacked;
    request.pageSize = system.vmPageSize;
    request.iosurfaceAlignment = system.iosurfaceAlignment;
    request.memory = reinterpret_cast<uint64_t>(memory);
    request.slice = slice;
    request.level = level;
    request.region = wireRegion(region);
    request.bytes = reinterpret_cast<uint64_t>(bytes);
    request.offset = offset;
    request.bytesPerRow = bytesPerRow;
    request.bytesPerImage = bytesPerImage;
    wire::Reply<wire::TextureLayoutCopyFromTextureMemoryReply> reply;
    DeviceConnection(_device)->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
    if (reply.value.refusal)
        webmetal::agx::logRegionRefusal(reply.value.refusal->c_str());
}

@end
