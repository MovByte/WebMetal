// Texture memory layout of the A10X Metal driver (AGXMetalA10, iPadOS 17.7.11, 21H461).
//
// A port of AGX::TextureGen2<AGXTextureMemoryLayout1, G9::Encoders, G9::Classes>::TextureGen2 (0x21c9becd4), the
// AGX::Texture helpers it and its callers use, and the callers' derivations:
//   -[AGXA10FamilyDevice heapTextureSizeAndAlignWithDescriptor:]                          0x21c8d6694
//   -[AGXA10FamilyTexture initImplWithDevice:Descriptor:...isHeapBacked:]                  0x21c980388
//   -[AGXA10FamilyTexture initImplWithTexture:pixelFormat:textureType:levels:slices:...]   0x21c98172c, 0x21c981074
//   -[AGXA10FamilyTexture initNewTextureData:]                                            0x21c981d88
//   -[AGXTextureLayout initWithDevice:descriptor:isHeapOrBufferBacked:], size, alignment  0x21c9c019c, 0x21c9c0380,
//                                                                                          0x21c9c0398
// Arithmetic follows the AArch64 code: 32-bit operations wrap, register shifts take their amount modulo the
// register width and division by zero yields zero.
//
// Not ported, and reported through the plugin's not-implemented channel (the call fails): IOSurface-backed textures
// (and with them the IOSurface tiled layout, kind 2), the interchange-tiled layout (kind 3; without an IOSurface the
// driver itself reads through the null format pointer it stores for that kind), calcSparseParams (the ctor clears
// this+0xe8, which gates it) and formats with more than two planes (the 21H461 table has none).
#pragma once

#include <cstdint>

namespace webmetal::agx {

// The fields of a format-table entry (AGX::TextureFormat) that the layout reads.
struct FormatInfo {
    uint32_t pixelFormat;          // MTLPixelFormat the entry is chosen for (0 for _texFormatUnsupported)
    uint32_t planes;               // +0x18
    uint32_t cls;                  // +0x20
    uint8_t alignLog2;             // +0x30
    uint8_t bytesPerPixel;         // +0x31
    uint32_t blockWidth;           // 0x21c9e4c70[+0x38].first
    uint32_t blockHeight;          // 0x21c9e4c70[+0x38].second
    bool depthStencil;             // (+0x3c & 0xc) != 0
    uint32_t chroma;               // +0x48 (3: 4:2:0)
    uint32_t planesWithoutSurface; // +0x50
    uint8_t planeBytes[2];         // +0x58, +0x59
};

// AGX::G9::TextureFormatTable::chooseTextureFormatOBJC (0x21c9c1318). Never null; unsupported formats return the
// _texFormatUnsupported entry.
const FormatInfo *chooseTextureFormat(uint64_t pixelFormat);
bool isUnsupported(const FormatInfo *format);

// Values of the running system that the layout reads.
struct SystemValues {
    uint64_t vmPageSize = 16384;          // libsystem_kernel _vm_page_size (iPadOS arm64: 16 KiB pages)
    uint64_t iosurfaceAlignment = 16384;  // IOSurfaceAlignProperty(kIOSurfaceOffset, 1), the static local
                                          // `iosurface_alignment` of the ctor
};

// The ctor's arguments, in Darwin argument order.
struct TextureGen2Args {
    uint64_t type = 0;              // MTLTextureType
    uint64_t pixelFormat = 0;       // MTLPixelFormat
    uint64_t usage = 0;             // MTLTextureUsage
    uint64_t storageMode = 0;       // MTLStorageMode
    uint32_t compression = 0;       // AGXTextureCompressionSettings (three bytes)
    uint32_t colorConversion = 0;   // eAGXColorSpaceConversion
    uint32_t rotation = 0;          // eAGXTextureRotation
    uint32_t swizzle = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t mipLevels = 0;
    uint32_t samples = 0;
    uint32_t arrayLength = 0;
    uint64_t bytesPerRow = 0;
    uint32_t sparsePageSize = 0;
    uint64_t cpuCacheMode = 0;
    bool iosurface = false;         // an IOSurface is given (not ported: aborts)
    uint32_t plane = 0;
    uint32_t slice = 0;
    bool iosurface2 = false;        // stored only
    uint32_t baseLevel = 0;
    uint32_t levelCount = 0;
    bool allowTwiddledPlanes = false;  // stack +0x58: allowNPOT from initImpl, !isHeapOrBufferBacked from AGXTextureLayout
    bool compressedBacking = false;    // stack +0x59: initImpl sets it for an IOSurface or buffer texture whose pixel
                                       // format is in its case list at 0x21c9dd1fd (the block-compressed formats);
                                       // the layout kind is then not forced to linear by bytesPerRow
    bool heapOrBufferBacked = false;   // stack +0x5a
};

// agxsTwiddledBlockInfo: one per plane at TextureGen2 +0xea, 0x18 bytes each.
struct TwiddledBlockInfo {
    uint16_t width = 0;       // +0x0
    uint16_t height = 0;      // +0x2
    uint16_t layers = 0;      // +0x4
    uint16_t pageAlign = 0;   // +0x6
    uint8_t levels = 0;       // +0x8
    uint8_t pagesPerTile = 0; // +0x9
    uint8_t widthLog2 = 0;    // +0xa
    uint8_t heightLog2 = 0;   // +0xb
    uint8_t tileWidthLog2 = 0;   // +0xc
    uint8_t tileHeightLog2 = 0;  // +0xd
    uint8_t bytesPerBlock = 0;   // +0xe
    uint8_t alignLog2 = 0;       // +0xf
    uint8_t mode = 0;            // +0x10: 0, or 1 (2 for cubes) for formats with this+0x134 set
    uint8_t tileMode = 0;        // +0x11: the ctor stores 1
    uint8_t blockWidth = 0;      // +0x12
    uint8_t blockHeight = 0;     // +0x13
    uint8_t reserved = 0;        // +0x14
    uint8_t cube = 0;            // +0x15
    uint8_t volume = 0;          // +0x16
};

// The TextureGen2 object, the fields the layout writes and reads.
struct TextureGen2 {
    SystemValues system;
    bool resolved = false;              // +0x10 (never set by this ctor)
    uint64_t type = 0;                  // +0x18
    uint64_t pixelFormat = 0;           // +0x20
    uint64_t usage = 0;                 // +0x28
    uint64_t storageMode = 0;           // +0x30
    uint8_t compressionMode = 0;        // +0x38
    uint8_t compressionFootprint = 0;   // +0x39
    uint8_t compressionByte2 = 0;       // +0x3a
    uint32_t levels = 0;                // +0x3c: full mip chain length when mipLevels > 1, else 1
    uint32_t lastSlice = 0;             // +0x4c
    uint32_t baseWidth = 0;             // +0x50
    uint32_t baseHeight = 0;            // +0x54
    uint32_t baseDepth = 0;             // +0x58
    uint32_t arrayLengthCopy = 0;       // +0x5c
    uint32_t classWord = 0;             // +0x60
    uint32_t levelCount = 0;            // +0x64
    uint32_t samples = 0;               // +0x68: sample count rounded up to a power of two
    uint32_t baseLevel = 0;             // +0x6c
    uint32_t width = 0;                 // +0x78
    uint32_t height = 0;                // +0x7c
    uint32_t depth = 0;                 // +0x80
    uint32_t mipLevels = 0;             // +0x84
    uint64_t bytesPerRow = 0;           // +0x90
    uint64_t cpuCacheMode = 0;          // +0x98
    bool iosurface = false;             // +0xa0
    uint32_t plane = 0;                 // +0xa8
    uint32_t slice = 0;                 // +0xac
    bool iosurface2 = false;            // +0xb0
    uint32_t rotation = 0;              // +0xb8
    uint32_t swizzle = 0;               // +0xbc
    bool valid = false;                 // +0xc0
    const FormatInfo *format = nullptr; // +0xc8
    uint32_t blockWidth = 0;            // +0xd0
    uint32_t blockHeight = 0;           // +0xd4
    uint32_t sparseTileWidth = 0;       // +0xd8
    uint32_t sparseTileHeight = 0;      // +0xdc
    bool twiddledPlanes = false;        // +0xe0
    uint32_t sparsePageSize = 0;        // +0xe4
    bool sparseParams = false;          // +0xe8
    TwiddledBlockInfo blockInfo[2];     // +0xea
    bool flag128 = false;               // +0x128
    uint32_t colorSpace = 0;            // +0x12c
    uint32_t chromaSiting = 0;          // +0x130
    bool twiddleCompressed = false;     // +0x134
    bool pvrtc = false;                 // +0x135
    uint64_t pageAlignment = 0;         // +0x138
    uint64_t planeEnd0[2] = {0, 0};     // +0x148, +0x150 (cumulative per plane, both views)
    uint64_t size = 0;                  // +0x158: sum of getPlaneSize<View1>
    uint64_t size0 = 0;                 // +0x160: sum of getPlaneSize<View0>
    uint32_t faces = 0;                 // +0x168
    uint32_t arrayLength = 0;           // +0x170
    uint8_t kind = 0;                   // +0x174: 0 linear, 1 twiddled, 2 IOSurface tiled, 3 interchange tiled
    uint8_t dim = 0;                    // +0x175: 1 1D, 2 2D, 3 3D, 4 cube
    uint8_t arrayBits = 0;              // +0x176: bit 0 array, bit 1 multisample, bit 2 YUV
    uint8_t compressionCopy[3] = {0, 0, 0};  // +0x177..+0x179
    bool compressedBacking = false;     // +0x17a
    bool heapOrBufferBacked = false;    // +0x17b
};

// AGX::TextureGen2<...>::TextureGen2 (0x21c9becd4).
TextureGen2 construct(const TextureGen2Args &args, const SystemValues &system = SystemValues());

// AGX::Texture<...> helpers.
uint64_t getPlaneSize1(const TextureGen2 &t, uint32_t plane);     // getPlaneSize<View1> 0x21c985a50
uint64_t getPlaneSize0(const TextureGen2 &t, uint32_t plane);     // getPlaneSize<View0> 0x21c91598c
uint64_t getMipmapSize1(const TextureGen2 &t, uint32_t plane);    // getMipmapSize<View1> 0x21c913aa4
uint64_t getMipmapSize0(const TextureGen2 &t, uint32_t plane);    // getMipmapSize<View0> 0x21c9136d4
uint64_t getLevelOffset1(const TextureGen2 &t, uint32_t level, uint32_t plane);  // 0x21c913bc4
uint64_t getLevelOffset0(const TextureGen2 &t, uint32_t level, uint32_t plane);  // 0x21c9137f4
uint64_t getStride1(const TextureGen2 &t, uint32_t plane);        // getStride<View1, Units0, Op0> 0x21c9851c8
uint64_t getStride0(const TextureGen2 &t, uint32_t level, uint32_t plane);  // getStride<View0, Units0, Op0> 0x21c9134f8
bool isDepthStencilAllocTypePadded(const TextureGen2 &t);         // 0x21c9c0b68
bool isYUVTexture(uint64_t pixelFormat);                          // 0x21c9c0100

// The alignment the callers derive from the object: heapTextureSizeAndAlign, initImpl (set_alignment:) and
// -[AGXTextureLayout alignment] share this code.
uint64_t textureAlignment(const TextureGen2 &t);

// The MTLTextureDescriptorPrivate fields the callers read. Every field is taken from the descriptor as it is; the
// descriptor's own defaults belong to MTLTextureDescriptor.
struct DescriptorFields {
    uint64_t textureType = 0;               // +0x00
    uint64_t pixelFormat = 0;               // +0x08
    uint64_t width = 0;                     // +0x10
    uint64_t height = 0;                    // +0x18
    uint64_t depth = 0;                     // +0x20
    uint64_t mipmapLevelCount = 0;          // +0x28
    uint64_t sampleCount = 0;               // +0x30
    uint64_t arrayLength = 0;               // +0x38
    uint64_t rotation = 0;                  // +0x48
    uint32_t swizzle = 0;                   // +0x54
    uint64_t compressionMode = 0;           // +0x60
    uint64_t usage = 0;                     // +0x68
    uint64_t resourceOptions = 0;           // +0x70
    bool allowGPUOptimizedContents = false; // +0x80
    uint64_t compressionFootprint = 0;      // +0x98
    uint64_t colorSpaceConversion = 0;      // +0xa8
    uint64_t storageMode = 0;               // +0xc0
};

// The ctor arguments the callers derive from the descriptor (compression word, color conversion, rotation).
uint32_t compressionSettings(const DescriptorFields &d);
uint32_t colorConversion(const DescriptorFields &d);
uint32_t rotation(const DescriptorFields &d);

struct SizeAndAlign {
    uint64_t size;
    uint64_t align;
};

// -[AGXA10FamilyDevice heapTextureSizeAndAlignWithDescriptor:] for a descriptor that passed validateWithDevice:
// (the method returns {0, 0} without building a layout otherwise).
SizeAndAlign heapTextureSizeAndAlign(const DescriptorFields &d, const SystemValues &system = SystemValues());

struct TextureAllocation {
    bool initialized = false;   // initImpl returned the texture (the layout is valid)
    uint64_t cpuSizeBytes = 0;  // AGXTexture _cpuSizeBytes
    uint64_t alignment = 0;     // AGXTexture _alignment
    uint64_t bytesPerRow = 0;   // AGXTexture _bytesPerRow
    uint64_t textureData58 = 0;  // initNewTextureData: argument +0x58, from a zeroed argument
    TextureGen2 layout;          // the constructed object (AGXA10FamilyTexture._impl), which views are built from
};

// -[AGXA10FamilyTexture initImplWithDevice:Descriptor:iosurface:nil plane:0 slice:0 buffer:nil bytesPerRow:0
//   allowNPOT:1 sparsePageSize:0 isCompressedIOSurface:NO isHeapBacked:NO] and initNewTextureData:.
TextureAllocation initImplDevice(const DescriptorFields &d, const SystemValues &system = SystemValues());
// The same with allowNPOT:0 isHeapBacked:YES (heap textures).
TextureAllocation initImplHeap(const DescriptorFields &d, const SystemValues &system = SystemValues());
// The device texture with sparsePageSize:sparsePageSize.
TextureAllocation initImplDeviceSparse(const DescriptorFields &d, uint32_t sparsePageSize,
                                       const SystemValues &system = SystemValues());

// -[AGXA10FamilyTexture initImplWithTexture:pixelFormat:textureType:levels:slices:swizzle:resourceIndex:] (0x21c98172c):
// a view's layout, built from its parent's constructed object. Not initialized when the parent is memoryless (+0x30)
// or the view's layout is not valid. The form without a swizzle
// (initImplWithTexture:pixelFormat:textureType:levels:slices:resourceIndex:, 0x21c981074) passes the parent's swizzle
// (+0xbc). Neither reads the slice location or the resource index.
TextureAllocation initImplView(const TextureGen2 &parent, uint64_t pixelFormat, uint64_t textureType,
                               uint64_t levelLocation, uint64_t levelLength, uint64_t sliceLength, uint32_t swizzle,
                               const SystemValues &system = SystemValues());

// -[AGXA10FamilyTexture initNewTextureData:]: the u64 at argument +0x58, given its previous value.
uint64_t initNewTextureData58(const TextureGen2 &t, uint64_t previous);
// Bit 0 of that word (this+0xe0): AGXTexture init does not suballocate the texture when it is set.
bool suballocationVeto(const TextureGen2 &t);

struct LayoutResult {
    bool initialized;  // -[AGXTextureLayout initWithDevice:descriptor:isHeapOrBufferBacked:] returned an object
    uint64_t size;     // -[AGXTextureLayout size]
    uint64_t alignment;  // -[AGXTextureLayout alignment]
    uint64_t total;      // this+0x158, where -finalizeTextureMemory: puts the watermark
    uint32_t blockWidth;   // this+0xd0, -[AGXTextureLayout copyGranularity]
    uint32_t blockHeight;  // this+0xd4
};

LayoutResult textureLayout(const DescriptorFields &d, bool heapOrBufferBacked,
                           const SystemValues &system = SystemValues());

}  // namespace webmetal::agx
