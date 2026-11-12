// Argument buffers on the A10X's tier 1: the layout (AGXA10FamilyIndirectArgumentBufferLayout) and the encoder
// (AGXA10FamilyIndirectArgumentEncoder) of 21H461, with the encoding the wire schema defines (wire/README.md, "Handles").
//
// The A10X driver lays an argument buffer out in its hardware's format and writes GPU addresses and descriptors into
// it. WebMetal's format is the front end's: every member sits at the offset the struct type (the AIR struct layout)
// gives it, constants hold their bytes, and a buffer, texture or sampler member holds a 64-bit handle word, the
// object's wire handle (0 for nil). The host's compiler reads argument buffers by the same definition.

#import "WebMetalShaders.h"
#import "../Commands/WebMetalCommands.h"

#import <objc/runtime.h>

#include <algorithm>
#include <cstring>
#include <map>

#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

// Metal.framework's methods of the bases (checked with the shader path's entry points, CheckShaderEntryPoints).
@interface _MTLIndirectArgumentEncoder (WebMetalFramework)
- (instancetype)initWithLayout:(_MTLIndirectArgumentBufferLayout *)layout device:(id<MTLDevice>)device;
@end

@interface _MTLIndirectArgumentBufferLayout (WebMetalFramework)
- (void)setStructType:(MTLStructType *)structType withDevice:(id<MTLDevice>)device;
@end

// The layout of an argument buffer (AGXA10FamilyIndirectArgumentBufferLayout on 21H461) and its encoder
// (AGXA10FamilyIndirectArgumentEncoder).
@interface WebMetalArgumentBufferLayout : _MTLIndirectArgumentBufferLayout
- (instancetype)initWithStructType:(MTLStructType *)structType device:(id<MTLDevice>)device;
@end

@interface WebMetalArgumentEncoder : _MTLIndirectArgumentEncoder <MTLArgumentEncoder>
- (instancetype)initWithLayout:(WebMetalArgumentBufferLayout *)layout device:(id<MTLDevice>)device;
@end

// Metal.framework's data type size and alignment tables (exported by Metal on 21H461: MTLDataTypeGetSize at
// 0x187abb144, MTLDataTypeGetAlignment at 0x187babbd4; 0 for struct and array types).
extern "C" uint64_t MTLDataTypeGetSize(uint64_t type);
extern "C" uint64_t MTLDataTypeGetAlignment(uint64_t type);

namespace {

// MTLDataType values (MTLArgument.h).
constexpr uint64_t kStruct = 1;
constexpr uint64_t kArray = 2;
constexpr uint64_t kTexture = 58;
constexpr uint64_t kSampler = 59;
constexpr uint64_t kPointer = 60;

enum class Slot { Buffer, Texture, Sampler, Constant, Other };

struct Entry {
    Slot slot;
    uint64_t offset;
    uint64_t dataType;
};

Slot slotOf(uint64_t dataType)
{
    switch (dataType) {
    case kPointer:
        return Slot::Buffer;
    case kTexture:
        return Slot::Texture;
    case kSampler:
        return Slot::Sampler;
    case 78:   // MTLDataTypeRenderPipeline
    case 79:   // MTLDataTypeComputePipeline
    case 80:   // MTLDataTypeIndirectCommandBuffer
    case 115:  // MTLDataTypeVisibleFunctionTable
    case 116:  // MTLDataTypeIntersectionFunctionTable
    case 117:  // MTLDataTypePrimitiveAccelerationStructure
    case 118:  // MTLDataTypeInstanceAccelerationStructure
        // Resource kinds that have no word in the wire schema's argument-buffer format.
        return Slot::Other;
    default:
        // Scalars, vectors, matrices and the pixel-format texel types are constants.
        return Slot::Constant;
    }
}

struct Extent {
    uint64_t size = 0;
    uint64_t alignment = 1;
};

Extent structExtent(MTLStructType *type, std::map<uint64_t, Entry> *entries, uint64_t base);

// The bytes and alignment of one member's storage, and its entries by argument index.
Extent memberExtent(uint64_t dataType, id details, uint64_t index, std::map<uint64_t, Entry> *entries, uint64_t offset)
{
    if (dataType == kArray) {
        MTLArrayType *array = details;
        uint64_t stride = array.stride, length = array.arrayLength, step = array.argumentIndexStride;
        Extent element{};
        for (uint64_t i = 0; i < length; i++) {
            id elementDetails = array.elementStructType ?: (id)array.elementArrayType;
            element = memberExtent(array.elementType, elementDetails, index + i * (step ? step : 1), entries,
                                   offset + i * stride);
        }
        return {stride * length, element.alignment};
    }
    if (dataType == kStruct)
        return structExtent((MTLStructType *)details, entries, offset);
    if (entries)
        (*entries)[index] = {slotOf(dataType), offset, dataType};
    return {MTLDataTypeGetSize(dataType), std::max<uint64_t>(MTLDataTypeGetAlignment(dataType), 1)};
}

// A struct's size (its members' extent rounded up to its alignment) and alignment, and its members' entries.
Extent structExtent(MTLStructType *type, std::map<uint64_t, Entry> *entries, uint64_t base)
{
    Extent out;
    uint64_t end = 0;
    for (MTLStructMember *member in type.members) {
        id details = member.structType ?: (id)member.arrayType;
        const Extent e = memberExtent(member.dataType, details, member.argumentIndex, entries, base + member.offset);
        end = std::max<uint64_t>(end, member.offset + e.size);
        out.alignment = std::max(out.alignment, e.alignment);
    }
    out.size = (end + out.alignment - 1) / out.alignment * out.alignment;
    return out;
}

}  // namespace

@implementation WebMetalArgumentBufferLayout {
@public
    std::map<uint64_t, Entry> _entries;
    uint64_t _encodedLength;
    uint64_t _alignment;
}

// -[AGXA10FamilyIndirectArgumentBufferLayout initWithStructType:] (AGXMetalA10 0x21c91b0ec) lays the struct out in the
// A10X's format. WebMetal's format is the struct's own layout: the members' offsets, the size of the struct.
- (instancetype)initWithStructType:(MTLStructType *)structType device:(id<MTLDevice>)device
{
    self = [super init];
    if (!self)
        return nil;
    const Extent extent = structExtent(structType, &_entries, 0);
    _encodedLength = extent.size;
    _alignment = extent.alignment;
    return self;
}

// -[AGXA10FamilyIndirectArgumentBufferLayout encodedLength] and -alignment (AGXMetalA10 0x21c91b7a4, 0x21c91b7b8).
- (NSUInteger)encodedLength
{
    return _encodedLength;
}

- (NSUInteger)alignment
{
    return _alignment;
}

// The layout is the front end's by definition (-bufferLayoutMatchesFrontEndLayout, AGXMetalA10 0x21c91b790, answers
// whether the driver's layout is).
- (BOOL)bufferLayoutMatchesFrontEndLayout
{
    return YES;
}

// -[AGXA10FamilyIndirectArgumentBufferLayout constantOffsetAtIndex:] (AGXMetalA10 0x21c91b804): the constant's offset;
// the driver aborts for an index that names no constant (std::map::at).
- (NSUInteger)constantOffsetAtIndex:(NSUInteger)index
{
    auto found = _entries.find(index);
    if (found == _entries.end() || found->second.slot != Slot::Constant)
        abort();
    return found->second.offset;
}

// The A10X's format carries a layout hash word that the WebMetal format does not have; and an argument buffer's
// objects are named by handle words, not by the driver's GPU addresses and unique identifiers.
- (uint32_t)hashValue
{
    WMLoudMethod(self, _cmd, "the WebMetal argument-buffer format has no layout hash");
}

- (uint32_t)hashOffset
{
    WMLoudMethod(self, _cmd, "the WebMetal argument-buffer format has no layout hash");
}

- (uint32_t)hashMask
{
    WMLoudMethod(self, _cmd, "the WebMetal argument-buffer format has no layout hash");
}

- (uint32_t)hashSignature
{
    WMLoudMethod(self, _cmd, "the WebMetal argument-buffer format has no layout hash");
}

- (void *)virtualAddressForBufferAtIndex:(NSUInteger)index inIndirectArgumentBuffer:(id)buffer atOffset:(NSUInteger)offset
{
    WMLoudMethod(self, _cmd, "argument buffers hold handle words, not GPU addresses");
}

- (uint64_t)uniqueIdentifierForTextureAtIndex:(NSUInteger)index inIndirectArgumentBuffer:(id)buffer atOffset:(NSUInteger)offset
{
    WMLoudMethod(self, _cmd, "argument buffers hold handle words, not the driver's unique identifiers");
}

- (uint64_t)uniqueIdentifierForSamplerAtIndex:(NSUInteger)index inIndirectArgumentBuffer:(id)buffer atOffset:(NSUInteger)offset
{
    WMLoudMethod(self, _cmd, "argument buffers hold handle words, not the driver's unique identifiers");
}

@end

namespace {

const Entry *entryOf(WebMetalArgumentBufferLayout *layout, uint64_t index)
{
    auto found = layout->_entries.find(index);
    return found == layout->_entries.end() ? nullptr : &found->second;
}

}  // namespace

@implementation WebMetalArgumentEncoder {
@public
    WebMetalArgumentBufferLayout *_webmetalLayout;
    id<MTLBuffer> _argumentBuffer;
    uint8_t *_bytes;
}

// -[AGXA10FamilyIndirectArgumentEncoder initWithLayout:device:] (AGXMetalA10 0x21c91d8dc): _MTLIndirectArgumentEncoder's
// initializer, then the layout and device kept for the setters (and a telemetry bit, which has no observable effect).
- (instancetype)initWithLayout:(WebMetalArgumentBufferLayout *)layout device:(id<MTLDevice>)device
{
    self = [super initWithLayout:layout device:device];
    if (self)
        _webmetalLayout = layout;
    return self;
}

// -[AGXA10FamilyIndirectArgumentEncoder setArgumentBuffer:offset:] (AGXMetalA10 0x21c91d960): the encoder writes at the
// buffer's CPU address plus the offset (nothing for a nil buffer).
- (void)setArgumentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset
{
    _argumentBuffer = buffer;
    _bytes = buffer ? static_cast<uint8_t *>(buffer.contents) + offset : nullptr;
}

// -[IOGPUMetalIndirectArgumentEncoder setIndirectArgumentBuffer:offset:] (IOGPU 0x1fcb37618) does not recognize itself.
- (void)setIndirectArgumentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset
{
    [self doesNotRecognizeSelector:_cmd];
}

// Writes a handle word at the member an index names. The driver aborts for an index its layout does not have
// (std::map::at in each setter).
static void writeWord(WebMetalArgumentEncoder *encoder, uint64_t word, Slot slot, NSUInteger index)
{
    const Entry *entry = entryOf(encoder->_webmetalLayout, index);
    if (!entry || entry->slot != slot)
        abort();
    std::memcpy(encoder->_bytes + entry->offset, &word, sizeof word);
}

// -[AGXA10FamilyIndirectArgumentEncoder setBuffer:offset:atIndex:] (AGXMetalA10 0x21c91d9d8) writes the buffer's GPU
// address plus the offset. The WebMetal word is the buffer's handle, which carries no offset.
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    if (buffer && offset)
        WMLoudMethod(self, _cmd, "a buffer at a non-zero offset: the argument buffer's handle word has no offset");
    const uint64_t handle = webmetal::HandleOf(buffer);
    if (buffer && !handle)
        WMLoudMethod(self, _cmd, "the buffer has no WebMetal handle");
    writeWord(self, handle, Slot::Buffer, index);
}

- (void)setBuffers:(const id<MTLBuffer> __unsafe_unretained *)buffers
           offsets:(const NSUInteger *)offsets
         withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setBuffer:buffers[i] offset:offsets[i] atIndex:range.location + i];
}

// -[AGXA10FamilyIndirectArgumentEncoder setTexture:atIndex:] (AGXMetalA10 0x21c91de7c).
- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
    const uint64_t handle = webmetal::HandleOf(texture);
    if (texture && !handle)
        WMLoudMethod(self, _cmd, "the texture has no WebMetal handle");
    writeWord(self, handle, Slot::Texture, index);
}

- (void)setTextures:(const id<MTLTexture> __unsafe_unretained *)textures withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setTexture:textures[i] atIndex:range.location + i];
}

// -[AGXA10FamilyIndirectArgumentEncoder setSamplerState:atIndex:] (AGXMetalA10 0x21c91deec).
- (void)setSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
    const uint64_t handle = webmetal::HandleOf(sampler);
    if (sampler && !handle)
        WMLoudMethod(self, _cmd, "the sampler has no WebMetal handle");
    writeWord(self, handle, Slot::Sampler, index);
}

- (void)setSamplerStates:(const id<MTLSamplerState> __unsafe_unretained *)samplers withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setSamplerState:samplers[i] atIndex:range.location + i];
}

// -[AGXA10FamilyIndirectArgumentEncoder constantDataAtIndex:] (AGXMetalA10 0x21c91e090): the constant's address in the
// argument buffer; the driver aborts for an index that names no constant.
- (void *)constantDataAtIndex:(NSUInteger)index
{
    const Entry *entry = entryOf(_webmetalLayout, index);
    if (!entry || entry->slot != Slot::Constant)
        abort();
    return _bytes + entry->offset;
}

// -[AGXA10FamilyIndirectArgumentEncoder encodedLength] and -alignment (AGXMetalA10 0x21c91e0ec, 0x21c91e104): the
// layout's.
- (NSUInteger)encodedLength
{
    return _webmetalLayout.encodedLength;
}

- (NSUInteger)alignment
{
    return _webmetalLayout.alignment;
}

// -[AGXA10FamilyIndirectArgumentEncoder newArgumentEncoderForBufferAtIndex:] (AGXMetalA10 0x21c91e11c) returns nil:
// argument buffers inside argument buffers are tier 2, which the A10X does not have.
- (id<MTLArgumentEncoder>)newArgumentEncoderForBufferAtIndex:(NSUInteger)index
{
    return nil;
}

- (void)setRenderPipelineState:(id<MTLRenderPipelineState>)pipeline atIndex:(NSUInteger)index
{
    WMLoudMethod(self, _cmd, "pipeline states in argument buffers have no word in the wire schema");
}

- (void)setRenderPipelineStates:(const id<MTLRenderPipelineState> __unsafe_unretained *)pipelines withRange:(NSRange)range
{
    WMLoudMethod(self, _cmd, "pipeline states in argument buffers have no word in the wire schema");
}

- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipeline atIndex:(NSUInteger)index
{
    WMLoudMethod(self, _cmd, "pipeline states in argument buffers have no word in the wire schema");
}

- (void)setComputePipelineStates:(const id<MTLComputePipelineState> __unsafe_unretained *)pipelines withRange:(NSRange)range
{
    WMLoudMethod(self, _cmd, "pipeline states in argument buffers have no word in the wire schema");
}

- (void)setIndirectCommandBuffer:(id<MTLIndirectCommandBuffer>)buffer atIndex:(NSUInteger)index
{
    WMLoudMethod(self, _cmd, "indirect command buffers need a command layout the wire schema does not define yet");
}

- (void)setIndirectCommandBuffers:(const id<MTLIndirectCommandBuffer> __unsafe_unretained *)buffers withRange:(NSRange)range
{
    WMLoudMethod(self, _cmd, "indirect command buffers need a command layout the wire schema does not define yet");
}

@end

@implementation WebMetalDevice (WebMetalArgumentBuffers)

// -[AGXA10FamilyDevice newIndirectArgumentBufferLayoutWithStructType:] (AGXMetalA10 0x21c8db140). Metal.framework
// passes the struct type of the argument buffer: from the argument descriptors (newArgumentEncoderWithArguments:), from
// a reflected binding, or from the function's reflection, which it asks the device's compiler for (-[_MTLFunction
// reflectionWithOptions:binaryArchives:], Metal 0x187a9fbbc); the plugin has no compiler, so that path brings no type.
- (id)newIndirectArgumentBufferLayoutWithStructType:(MTLStructType *)structType
{
    if (!structType)
        WMLoudMethod(self, _cmd, "no struct type: a function's reflection, which Metal.framework asks the device's "
                                 "compiler for, has no WebMetal source");
    return [[WebMetalArgumentBufferLayout alloc] initWithStructType:structType device:self];
}

// -[AGXA10FamilyDevice newArgumentEncoderWithLayout:] (AGXMetalA10 0x21c8db16c).
- (id<MTLArgumentEncoder>)newArgumentEncoderWithLayout:(id)layout
{
    if (![layout isKindOfClass:[WebMetalArgumentBufferLayout class]])
        WMLoudMethod(self, _cmd, "the layout was not made by this device");
    webmetal::RequireShaderEntryPoints(self, _cmd);
    return [[WebMetalArgumentEncoder alloc] initWithLayout:layout device:self];
}

// -[IOGPUMetalDevice newIndirectArgumentEncoderWithLayout:] (IOGPU 0x1fcb35974) aborts.
- (id<MTLArgumentEncoder>)newIndirectArgumentEncoderWithLayout:(id)layout
{
    abort();
}

@end
