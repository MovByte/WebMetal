// How the A10X driver lists the resources of a blit operation in the IOGPU resource lists of its segments, and so
// whether the kernel stamps their event 0 (WebMetalCommandBuffer.mm states the kernel's rule).
//
// AGXA10FamilyBlitContext runs each operation in segments of its own: it ends the open segment (0x21c8c10a0) and begins
// a kick whose type the kick header's word at +0x18 gives (0x21c8d02b0 writes the kernel command 0x10000; the kernel's
// AGX kick reads the type at 0xfffffff005bf7ce8: 1 render, 2 blit, 3 compute). The operations reach three kicks:
//  - a render kick of a vertex program over buffer memory (0x21c8c372c, header 0x21c9dba60: type 1), whose resource
//    list holds the destination with flags 3 and the source with 1 (0x21c8c3ccc): the render kick stamps both
//    (flag bits 1 and 0);
//  - a blit kick over a framebuffer (0x21c8c1bb8 and 0x21c8c2a6c, header 0x21c9dba30: type 2), whose list holds the
//    source with 4 and the destination with 12, or the framebuffer's attachments with 12 (0x21c8c191c, AGX::Framebuffer
//    0x21c919170): the blit kick stamps both (bits 2 and 3);
//  - copies of buffer memory, which use the two kicks above (0x21c8bb208).
// The source's flags are 0x103 instead only while -resolveCounters:inRange:destinationBuffer:destinationOffset: copies
// a counter sample buffer (ctx+0x730, 0x21c8bb150 to 0x21c8bb178). Each kick lists both resources once it begins, so an
// operation stamps its resources unless it begins no kick. The rules below say when it begins none; where a path that
// the arguments select is not read, the use stays untraced.

#import "WebMetalCommandState.h"
#import "../Resources/WebMetalTexture.h"

#include <algorithm>

namespace webmetal {

namespace {

enum class Decision { Stamped, Unlisted, Untraced };

id recordObject(EncoderState &encoder, uint64_t handle)
{
    if (!handle)
        return nil;
    for (auto at = encoder.byRecord.rbegin(); at != encoder.byRecord.rend(); ++at)
        if (at->handle == handle)
            return at->object;
    return nil;
}

// Notes the uses of the record's objects with the decision, and forgets the record's objects.
void apply(EncoderState &encoder, Decision decision, std::initializer_list<uint64_t> handles)
{
    if (decision != Decision::Unlisted) {
        const Stream::UseKind kind = decision == Decision::Stamped ? Stream::UseKind::Stamped : Stream::UseKind::Untraced;
        std::vector<uint64_t> keys;
        for (uint64_t handle : handles) {
            const uint64_t key = KernelResourceKey(recordObject(encoder, handle));
            if (key && std::find(keys.begin(), keys.end(), key) == keys.end()) {
                keys.push_back(key);
                encoder.stream->noteUse(key, kind);
            }
        }
    }
    encoder.byRecord.clear();
}

// The texture type AGX::Texture keeps at +0x18 (the texture's or the view's MTLTextureType).
MTLTextureType textureType(id texture)
{
    return [(id<MTLTexture>)texture textureType];
}

// The extent the copies and fills iterate over in depth. Each of them normalizes the size by the texture type first: a
// texture of type 1D or 1D array gets height and depth 1, of type 2D, 2D array, 2D multisample, cube, cube array or 2D
// multisample array depth 1, and a 3D texture keeps the depth (`mov w14, #0x17c` and `tst x14, #0x3` at 0x21c8be94c,
// 0x21c8be98c, 0x21c8bef1c, 0x21c8bf7d8 and 0x21c8c04bc). A texture-buffer texture takes the buffer paths instead.
uint64_t normalizedDepth(MTLTextureType type, uint64_t depth)
{
    return type == MTLTextureType3D ? depth : 1;
}

MTLOrigin origin(const wire::Origin &o)
{
    return MTLOriginMake(o.x, o.y, o.z);
}

MTLSize size(const wire::Size &s)
{
    return MTLSizeMake(s.width, s.height, s.depth);
}

// The address of a buffer's memory in its kernel resource: a heap buffer's lies at its offset in the heap's resource
// (-[IOGPUMetalBuffer initWithHeap:resource:offset:length:] adds it to _res's GPU address, IOGPU 0x1fcb2d8c4); every
// other buffer has a kernel resource of its own. A texture over a buffer lies at its offset in the buffer's memory.
uint64_t bufferBase(id object)
{
    if ([object isKindOfClass:[AGXTexture class]]) {
        id<MTLTexture> texture = (id<MTLTexture>)object;
        return texture.buffer ? bufferBase(texture.buffer) + texture.bufferOffset : 0;
    }
    const ResourceState *state = StateOf(object);
    return state && state->heap ? state->heapOffset : 0;
}

// The bytes of one element of a texture's plane in a copy: the plane's bytes per block (+0x58) times the sample count
// (+0x68), the plane being the one MTLBlitOptionStencilFromDepthStencil chooses when the entry has two planes and +0x50
// is set (0x21c8bee1c to 0x21c8bee5c, 0x21c8bf6c8 to 0x21c8bf708).
uint64_t elementBytes(const agx::TextureGen2 &layout, uint64_t options)
{
    const agx::FormatInfo *format = layout.format;
    const uint32_t plane = format->planes >= 2 && format->planesWithoutSurface ? (options >> 1) & 1 : 0;
    return uint64_t(format->planeBytes[plane]) * layout.samples;
}

// A copy of memory between two resources (0x21c8bb208): no kick for 0 bytes, nor when both GPU addresses are equal
// (0x21c8bb2ac, the ranges then overlapping); any other begins a kick.
Decision copyMemory(id destination, uint64_t destinationOffset, id source, uint64_t sourceOffset, uint64_t size)
{
    if (size == 0)
        return Decision::Unlisted;
    if (KernelResourceKey(source) == KernelResourceKey(destination) &&
        bufferBase(source) + sourceOffset == bufferBase(destination) + destinationOffset)
        return Decision::Unlisted;
    return Decision::Stamped;
}

// -[AGXA10FamilyBlitContext copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:] (0x21c8b9860, 0x21c8bb208)
// returns without a kick when the source and destination GPU addresses are equal and the ranges overlap
// (0x21c8bb2ac); a size of 0 ends there too, as the vertex copy (0x21c8c4ca8) then has no head, body or tail and
// returns before its kick, and the remainder it leaves the framebuffer copies is 0 (0x21c8bba60). Any other size
// begins a kick for the head or tail (0x21c8c4e98) or the framebuffer copies of the body (0x21c8bbe40), overlapping
// ranges through a scratch buffer in chunks that each list their source or destination (0x21c8bb3d8 to 0x21c8bb908).
Decision copyBuffers(id source, uint64_t sourceOffset, id destination, uint64_t destinationOffset, uint64_t size)
{
    if (!source || !destination)
        return Decision::Untraced;
    return copyMemory(destination, destinationOffset, source, sourceOffset, size);
}

// -fillBuffer:range:value: and -fillBuffer:range:pattern4: (0x21c8b9d78, 0x21c8b9f84, both 0x21c8c07b0): the vertex
// fill lists the buffer for an unaligned head or tail (0x21c8c4e98, with no source), and the framebuffer fills list it
// for the 16-byte body (0x21c8c0b50); a range of length 0 begins neither (0x21c8c0858).
Decision fillBuffer(id buffer, uint64_t length)
{
    return length ? Decision::Stamped : Decision::Unlisted;
}

// The copies between a buffer and a texture (0x21c8bedc4 from buffer to texture, 0x21c8bf66c from texture to buffer).
// A texture-buffer texture takes a copy of memory between the buffer at its offset and the texture's memory at the
// origin's x elements, the width's elements long (0x21c8bf50c to 0x21c8bf564, 0x21c8bfdbc to 0x21c8bfe0c). Any other
// texture first passes the level check (0x21c9c0a2c, true for every format of 1 by 1 blocks): where it fails, one
// element is copied per slice of the normalized depth as memory between distinct resources (0x21c8bf494,
// 0x21c8bfd10). Then elements of 32, 64 and 128 bytes (the element-size table 0x21c9dc5be leaves `w17` clear for
// them, 0x21c8bf638, 0x21c8bf650, 0x21c8beec0) take a path that begins one framebuffer kick whatever the depth
// (0x21c8bf274 to 0x21c8bf5fc, 0x21c8bfaa8), and the others one per slice of the normalized depth (0x21c8bf0f8,
// 0x21c8bf9c0). The kicks list both resources whether or not they share a kernel resource; a texture of 1 by 1 blocks
// over the buffer's memory takes those paths too.
Decision copyBufferTexture(id buffer, uint64_t bufferOffset, id texture, uint64_t originX, uint64_t width,
                           uint64_t depth, uint64_t options)
{
    if (!buffer || !texture)
        return Decision::Untraced;
    const agx::TextureGen2 *layout = TextureLayoutOf(texture);
    const MTLTextureType type = textureType(texture);
    if (type == MTLTextureTypeTextureBuffer) {
        if (!layout || !layout->format)
            return Decision::Untraced;
        const uint64_t element = elementBytes(*layout, options);
        return copyMemory(texture, originX * element, buffer, bufferOffset, width * element);
    }
    if (normalizedDepth(type, depth))
        return Decision::Stamped;
    if (!layout || !layout->format)
        return Decision::Untraced;
    const uint64_t element = elementBytes(*layout, options);
    const bool unitBlocks = layout->format->blockWidth == 1 && layout->format->blockHeight == 1;
    return unitBlocks && (element == 32 || element == 64 || element == 128) ? Decision::Stamped : Decision::Unlisted;
}

// -copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:
// destinationOrigin: (0x21c8b9648, and the options form through it, 0x21c8b9f1c) copies each plane both formats have
// (0x21c8b96e4 to 0x21c8b96f8) with 0x21c8be838. A texture-buffer source makes it the copy from a buffer to the
// destination, the source's memory at its origin's x elements standing for the buffer (0x21c8becb8 to 0x21c8bed1c); a
// texture-buffer destination the copy from the source to a buffer (0x21c8bed24 to 0x21c8bed84). Otherwise it
// normalizes the size by both types and begins a framebuffer copy per slice of the depth (0x21c8beb98), or, where the
// level check fails for either texture (0x21c9c0a2c, never for formats of 1 by 1 blocks), copies one element per slice
// as memory (0x21c8bec8c), whose addresses textures that share a kernel resource may make equal, which is not traced.
Decision copyTextures(id source, MTLOrigin sourceOrigin, id destination, MTLOrigin destinationOrigin, MTLSize size)
{
    if (!source || !destination)
        return Decision::Untraced;
    const MTLTextureType sourceType = textureType(source), destinationType = textureType(destination);
    const agx::TextureGen2 *sourceLayout = TextureLayoutOf(source), *destinationLayout = TextureLayoutOf(destination);
    if (!sourceLayout || !sourceLayout->format || !destinationLayout || !destinationLayout->format)
        return Decision::Untraced;
    if (sourceType == MTLTextureTypeTextureBuffer)
        return copyBufferTexture(source, sourceOrigin.x * elementBytes(*sourceLayout, 0), destination,
                                 destinationOrigin.x, size.width, size.depth, 0);
    if (destinationType == MTLTextureTypeTextureBuffer)
        return copyBufferTexture(destination, destinationOrigin.x * elementBytes(*destinationLayout, 0), source,
                                 sourceOrigin.x, size.width, size.depth, 0);
    const uint64_t slices = sourceType == MTLTextureType3D && destinationType == MTLTextureType3D ? size.depth : 1;
    if (!slices)
        return Decision::Unlisted;
    if (KernelResourceKey(source) != KernelResourceKey(destination))
        return Decision::Stamped;
    auto unitBlocks = [](const agx::TextureGen2 *layout) {
        return layout->format->blockWidth == 1 && layout->format->blockHeight == 1;
    };
    return unitBlocks(sourceLayout) && unitBlocks(destinationLayout) ? Decision::Stamped : Decision::Untraced;
}

// -copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:
// (0x21c8ba05c) copies, for each plane both formats have, each level and slice of the counts (the loops at 0x21c8ba1a8,
// 0x21c8ba2f0, 0x21c8ba3f8 and 0x21c8ba660). For a sparse texture or layouts of different kinds every copy is a region
// copy (0x21c8bbe98 to 0x21c8be838). Otherwise it collects the source's memory ranges, each depth slice of each level
// (0x21c8bc014, sorted by 0x21c8bc280 and joined), and copies each depth slice of each destination level as memory
// (0x21c8bcde8 to 0x21c8be14c) when the destination's range, the source level's size long, overlaps one of them (the
// binary search 0x21c8bfe4c), else as a region copy (0x21c8bd058 to 0x21c8be838). Between distinct kernel resources
// the ranges never overlap and every copy is a region copy, which kicks. Textures of one root overlap only where a
// copy names the same subresource on both sides, a copy of memory onto itself that begins no kick (0x21c8bb2ac), and
// are disjoint elsewhere. Textures over a buffer, the plane pairs of 260 and 261, and distinct textures of one heap,
// whose memory may alias, are not traced here.
Decision copySurfaces(id source, uint64_t sourceSlice, uint64_t sourceLevel, id destination, uint64_t destinationSlice,
                      uint64_t destinationLevel, uint64_t slices, uint64_t levels)
{
    if (!source || !destination)
        return Decision::Untraced;
    const agx::TextureGen2 *sourceLayout = TextureLayoutOf(source), *destinationLayout = TextureLayoutOf(destination);
    if (!sourceLayout || !sourceLayout->format || !destinationLayout || !destinationLayout->format)
        return Decision::Untraced;
    const uint32_t planes = std::min(sourceLayout->format->planes, destinationLayout->format->planes);
    if (!slices || !levels || !planes)
        return Decision::Unlisted;
    if (KernelResourceKey(source) != KernelResourceKey(destination))
        return Decision::Stamped;
    if (((id<MTLTexture>)source).buffer || ((id<MTLTexture>)destination).buffer ||
        sourceLayout->pixelFormat != destinationLayout->pixelFormat)
        return Decision::Untraced;
    // The root, and the root's level and slice of a view's first level and slice.
    auto root = [](id<MTLTexture> texture, uint64_t &level, uint64_t &slice) {
        for (; texture.parentTexture; texture = texture.parentTexture) {
            level += texture.parentRelativeLevel;
            slice += texture.parentRelativeSlice;
        }
        return texture;
    };
    // Distinct roots in one heap may alias each other's memory.
    if (root(source, sourceLevel, sourceSlice) != root(destination, destinationLevel, destinationSlice))
        return Decision::Untraced;
    return sourceLevel == destinationLevel && sourceSlice == destinationSlice ? Decision::Unlisted : Decision::Stamped;
}

// -fillTexture:level:slice:region:bytes:length: (0x21c8b9fe8, 0x21c8c036c), which IOGPU's color forms reach through
// MTLPackColor with a stack buffer (0x1fcb315a0): one framebuffer fill per slice of the normalized depth (0x21c8c06ec).
// Before it, a pattern shorter than an element is repeated modulo its length (0x21c8c05b4): with a length of 0 the
// driver reads the caller's bytes past their end, which is not traced. A texture-buffer texture takes the buffer fill of
// its memory from the origin's x elements, the width's elements long (0x21c8c074c to 0x21c8c07a8, plane 0's bytes per
// block times the sample count per element).
Decision fillTexture(id texture, uint64_t width, uint64_t depth, bool callerBytes, uint64_t length)
{
    if (!texture)
        return Decision::Untraced;
    const MTLTextureType type = textureType(texture);
    if (callerBytes && length == 0)
        return Decision::Untraced;
    if (type == MTLTextureTypeTextureBuffer) {
        const agx::TextureGen2 *layout = TextureLayoutOf(texture);
        if (!layout || !layout->format)
            return Decision::Untraced;
        return fillBuffer(texture, width * elementBytes(*layout, 0));
    }
    return normalizedDepth(type, depth) ? Decision::Stamped : Decision::Unlisted;
}

// -[AGXA10FamilyBlitContext generateMipmapsForTexture:] (0x21c8b996c) returns at once for a texture over an IOSurface
// (+0xa0, 0x21c8b99ec), the pixel formats 550 and 551 and 624 to 627 (0x21c8b99f4 to 0x21c8b9a08), an entry whose
// +0x3c & 0x3 is clear (0x21c8b9a14) and a single level (+0x64, 0x21c8b9a90); otherwise every level after the first
// begins a framebuffer kick per slice that lists the texture as the attachment (0x21c8b9ccc, 0x21c8b9d14).
Decision generateMipmaps(id texture)
{
    if (!texture)
        return Decision::Untraced;
    const ResourceState *state = StateOf(texture);
    if (state && state->iosurface)
        return Decision::Unlisted;
    const agx::TextureGen2 *layout = TextureLayoutOf(texture);
    if (!layout || !layout->format)
        return Decision::Untraced;
    const uint64_t format = layout->pixelFormat;
    if ((format >= 624 && format < 628) || (format >= 550 && format < 552) || !layout->format->mipmapGeneration ||
        layout->levelCount < 2)
        return Decision::Unlisted;
    return Decision::Stamped;
}

}  // namespace

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyBufferToBuffer &r)
{
    apply(encoder,
          copyBuffers(recordObject(encoder, r.sourceBuffer), r.sourceOffset,
                      recordObject(encoder, r.destinationBuffer), r.destinationOffset, r.size),
          {r.sourceBuffer, r.destinationBuffer});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyBufferToTexture &r)
{
    apply(encoder,
          copyBufferTexture(recordObject(encoder, r.sourceBuffer), r.sourceOffset, recordObject(encoder, r.destinationTexture),
                            r.destinationOrigin.x, r.sourceSize.width, r.sourceSize.depth, 0),
          {r.sourceBuffer, r.destinationTexture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyBufferToTextureOptions &r)
{
    apply(encoder,
          copyBufferTexture(recordObject(encoder, r.sourceBuffer), r.sourceOffset, recordObject(encoder, r.destinationTexture),
                            r.destinationOrigin.x, r.sourceSize.width, r.sourceSize.depth, r.options),
          {r.sourceBuffer, r.destinationTexture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyTextureToBuffer &r)
{
    apply(encoder,
          copyBufferTexture(recordObject(encoder, r.destinationBuffer), r.destinationOffset, recordObject(encoder, r.sourceTexture),
                            r.sourceOrigin.x, r.sourceSize.width, r.sourceSize.depth, 0),
          {r.sourceTexture, r.destinationBuffer});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyTextureToBufferOptions &r)
{
    apply(encoder,
          copyBufferTexture(recordObject(encoder, r.destinationBuffer), r.destinationOffset, recordObject(encoder, r.sourceTexture),
                            r.sourceOrigin.x, r.sourceSize.width, r.sourceSize.depth, r.options),
          {r.sourceTexture, r.destinationBuffer});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyTextureToTexture &r)
{
    apply(encoder,
          copyTextures(recordObject(encoder, r.sourceTexture), origin(r.sourceOrigin),
                       recordObject(encoder, r.destinationTexture), origin(r.destinationOrigin), size(r.sourceSize)),
          {r.sourceTexture, r.destinationTexture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyTextureToTextureOptions &r)
{
    apply(encoder,
          copyTextures(recordObject(encoder, r.sourceTexture), origin(r.sourceOrigin),
                       recordObject(encoder, r.destinationTexture), origin(r.destinationOrigin), size(r.sourceSize)),
          {r.sourceTexture, r.destinationTexture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyTextureToTextureSlicesLevels &r)
{
    apply(encoder,
          copySurfaces(recordObject(encoder, r.sourceTexture), r.sourceSlice, r.sourceLevel,
                       recordObject(encoder, r.destinationTexture), r.destinationSlice, r.destinationLevel,
                       r.sliceCount, r.levelCount),
          {r.sourceTexture, r.destinationTexture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitFillBuffer &r)
{
    apply(encoder, fillBuffer(recordObject(encoder, r.buffer), r.range.length), {r.buffer});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitFillBufferPattern4 &r)
{
    apply(encoder, fillBuffer(recordObject(encoder, r.buffer), r.range.length), {r.buffer});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitFillTextureBytes &r)
{
    apply(encoder, fillTexture(recordObject(encoder, r.texture), r.region.size.width, r.region.size.depth, true, r.length), {r.texture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitFillTextureColor &r)
{
    apply(encoder, fillTexture(recordObject(encoder, r.texture), r.region.size.width, r.region.size.depth, false, 0), {r.texture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitFillTextureColorPixelFormat &r)
{
    apply(encoder, fillTexture(recordObject(encoder, r.texture), r.region.size.width, r.region.size.depth, false, 0), {r.texture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitGenerateMipmapsForTexture &r)
{
    apply(encoder, generateMipmaps(recordObject(encoder, r.texture)), {r.texture});
}

}  // namespace webmetal
