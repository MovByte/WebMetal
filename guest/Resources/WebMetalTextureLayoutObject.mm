// Texture layout objects: WebMetalTextureLayout stands in for AGXTextureLayout (AGXTextureLayout ->
// IOGPUMetalTextureLayout -> NSObject on 21H461), which -[AGXA10FamilyDevice newTextureLayoutWithDescriptor:
// isHeapOrBufferBacked:] (AGXMetalA10 0x21c8d8fd4) makes. It answers the AGX layout of a texture whose memory the caller
// provides: its size (the layout and a 16-byte watermark after it), alignment and copy granularity.

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
}
@end

namespace webmetal {

// -[AGXTextureLayout initWithDevice:descriptor:isHeapOrBufferBacked:] (AGXMetalA10 0x21c9c019c): the layout of the
// descriptor (no validation), twiddled planes allowed unless the memory is a heap's or a buffer's; nil when the layout
// is not valid.
id NewTextureLayout(WebMetalDevice *device, SEL _cmd, MTLTextureDescriptor *descriptor, bool heapOrBufferBacked)
{
    const agx::LayoutResult r =
        agx::textureLayout(LayoutDescriptorFields(WireTextureDescriptor(device, _cmd, descriptor)), heapOrBufferBacked,
                           LayoutSystemValues());
    if (!r.initialized)
        return nil;
    WebMetalTextureLayout *layout = [WebMetalTextureLayout new];
    layout->_total = r.total;
    layout->_alignment = r.alignment;
    layout->_blockWidth = r.blockWidth;
    layout->_blockHeight = r.blockHeight;
    return layout;
}

}  // namespace webmetal

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

@end
