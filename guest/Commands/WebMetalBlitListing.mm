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

// The address of a buffer's memory in its kernel resource: a heap buffer's lies at its offset in the heap's resource
// (-[IOGPUMetalBuffer initWithHeap:resource:offset:length:] adds it to _res's GPU address, IOGPU 0x1fcb2d8c4); every
// other buffer has a kernel resource of its own.
uint64_t bufferBase(id buffer)
{
    const ResourceState *state = StateOf(buffer);
    return state && state->heap ? state->heapOffset : 0;
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
    if (size == 0)
        return Decision::Unlisted;
    if (KernelResourceKey(source) == KernelResourceKey(destination) &&
        bufferBase(source) + sourceOffset == bufferBase(destination) + destinationOffset)
        return Decision::Unlisted;
    return Decision::Stamped;
}

// -fillBuffer:range:value: and -fillBuffer:range:pattern4: (0x21c8b9d78, 0x21c8b9f84, both 0x21c8c07b0): the vertex
// fill lists the buffer for an unaligned head or tail (0x21c8c4e98, with no source), and the framebuffer fills list it
// for the 16-byte body (0x21c8c0b50); a range of length 0 begins neither (0x21c8c0858).
Decision fillBuffer(id buffer, uint64_t length)
{
    return length ? Decision::Stamped : Decision::Unlisted;
}

// The bytes of one element of a buffer copy's texture: the plane's bytes per block (+0x58, the plane the
// MTLBlitOptionStencilFromDepthStencil option chooses when the entry has two planes and +0x50 is set) times the sample
// count (+0x68) (0x21c8bee1c to 0x21c8bee5c, 0x21c8bf6c8 to 0x21c8bf708). Elements of 32, 64 and 128 bytes take a path
// that begins one kick for all of the depth (`w17` clear: 0x21c8bf274, 0x21c8bfaa8); the others begin one per slice
// of the normalized depth (0x21c8bf0f8, 0x21c8bf9c0) or, for the levels the framebuffer cannot address
// (0x21c9c0a2c), copy one element per slice as buffer memory (0x21c8bf494, 0x21c8bfd10).
bool singleKickElement(const agx::TextureGen2 &layout, uint64_t options)
{
    const agx::FormatInfo *format = layout.format;
    if (!format)
        return true;
    const uint32_t plane = format->planes >= 2 && format->planesWithoutSurface ? (options >> 1) & 1 : 0;
    const uint64_t bytes = uint64_t(format->planeBytes[plane]) * layout.samples;
    return bytes == 32 || bytes == 64 || bytes == 128;
}

// The copies between a buffer and a texture (0x21c8bedc4 from buffer to texture, 0x21c8bf66c from texture to buffer):
// a texture-buffer texture takes a copy of buffer memory (0x21c8bf564, 0x21c8bfe0c); a texture whose kernel resource is
// the buffer's may make that copy's addresses equal, which is not traced.
Decision copyBufferTexture(id buffer, id texture, uint64_t depth, uint64_t options)
{
    if (!buffer || !texture)
        return Decision::Untraced;
    const MTLTextureType type = textureType(texture);
    if (type == MTLTextureTypeTextureBuffer || KernelResourceKey(buffer) == KernelResourceKey(texture))
        return Decision::Untraced;
    if (normalizedDepth(type, depth))
        return Decision::Stamped;
    const agx::TextureGen2 *layout = TextureLayoutOf(texture);
    if (!layout || singleKickElement(*layout, options))
        return Decision::Untraced;
    return Decision::Unlisted;
}

// -copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:
// destinationOrigin: (0x21c8b9648, and the options form through it, 0x21c8b9f1c) copies each plane both formats have
// (0x21c8b96e4 to 0x21c8b96f8) with 0x21c8be838, which normalizes the size by both types and begins a framebuffer copy
// per slice of the depth (0x21c8beb98), or copies one element per slice as texture memory where the framebuffer cannot
// address a level (0x21c8bec8c). Textures that share a kernel resource may make that copy's addresses equal, which is
// not traced; a texture-buffer texture takes the buffer copies.
Decision copyTextures(id source, id destination, uint64_t depth)
{
    if (!source || !destination)
        return Decision::Untraced;
    const MTLTextureType sourceType = textureType(source), destinationType = textureType(destination);
    if (sourceType == MTLTextureTypeTextureBuffer || destinationType == MTLTextureTypeTextureBuffer ||
        KernelResourceKey(source) == KernelResourceKey(destination))
        return Decision::Untraced;
    const uint64_t slices = sourceType == MTLTextureType3D && destinationType == MTLTextureType3D ? depth : 1;
    return slices ? Decision::Stamped : Decision::Unlisted;
}

// -fillTexture:level:slice:region:bytes:length: (0x21c8b9fe8, 0x21c8c036c), which IOGPU's color forms reach through
// MTLPackColor with a stack buffer (0x1fcb315a0): one framebuffer fill per slice of the normalized depth (0x21c8c06ec).
// Before it, a pattern shorter than an element is repeated modulo its length (0x21c8c05b4): with a length of 0 the
// driver reads the caller's bytes past their end, which is not traced. A texture-buffer texture takes the buffer fill
// (0x21c8c07a8).
Decision fillTexture(id texture, uint64_t depth, bool callerBytes, uint64_t length)
{
    if (!texture)
        return Decision::Untraced;
    const MTLTextureType type = textureType(texture);
    if (type == MTLTextureTypeTextureBuffer || (callerBytes && length == 0))
        return Decision::Untraced;
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
          copyBufferTexture(recordObject(encoder, r.sourceBuffer), recordObject(encoder, r.destinationTexture),
                            r.sourceSize.depth, 0),
          {r.sourceBuffer, r.destinationTexture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyBufferToTextureOptions &r)
{
    apply(encoder,
          copyBufferTexture(recordObject(encoder, r.sourceBuffer), recordObject(encoder, r.destinationTexture),
                            r.sourceSize.depth, r.options),
          {r.sourceBuffer, r.destinationTexture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyTextureToBuffer &r)
{
    apply(encoder,
          copyBufferTexture(recordObject(encoder, r.destinationBuffer), recordObject(encoder, r.sourceTexture),
                            r.sourceSize.depth, 0),
          {r.sourceTexture, r.destinationBuffer});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyTextureToBufferOptions &r)
{
    apply(encoder,
          copyBufferTexture(recordObject(encoder, r.destinationBuffer), recordObject(encoder, r.sourceTexture),
                            r.sourceSize.depth, r.options),
          {r.sourceTexture, r.destinationBuffer});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyTextureToTexture &r)
{
    apply(encoder,
          copyTextures(recordObject(encoder, r.sourceTexture), recordObject(encoder, r.destinationTexture),
                       r.sourceSize.depth),
          {r.sourceTexture, r.destinationTexture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitCopyTextureToTextureOptions &r)
{
    apply(encoder,
          copyTextures(recordObject(encoder, r.sourceTexture), recordObject(encoder, r.destinationTexture),
                       r.sourceSize.depth),
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
    apply(encoder, fillTexture(recordObject(encoder, r.texture), r.region.size.depth, true, r.length), {r.texture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitFillTextureColor &r)
{
    apply(encoder, fillTexture(recordObject(encoder, r.texture), r.region.size.depth, false, 0), {r.texture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitFillTextureColorPixelFormat &r)
{
    apply(encoder, fillTexture(recordObject(encoder, r.texture), r.region.size.depth, false, 0), {r.texture});
}

void NoteRecordListing(id, SEL, EncoderState &encoder, const wire::BlitGenerateMipmapsForTexture &r)
{
    apply(encoder, generateMipmaps(recordObject(encoder, r.texture)), {r.texture});
}

}  // namespace webmetal
