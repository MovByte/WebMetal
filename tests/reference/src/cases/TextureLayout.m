// Texture layouts, the driver SPI QuartzCore prepares image textures with: a
// layout object (newTextureLayoutWithDescriptor:isHeapOrBufferBacked:)
// answers the size and alignment of the memory a texture needs and copies
// texels between linear bytes and that memory (copyFromLinearBytes:... and
// copyFromTextureMemory:...), which newTextureWithBytesNoCopy:length:
// descriptor:deallocator: then makes a texture over. The memory's layout is
// the GPU's own, so the case records only what passes through the API: the
// texels read back through the layout, and the layout's answers under the
// device rule.

#import "WMRefCase.h"

#include <mach/mach.h>
#include <string.h>

@protocol WMRefTextureLayout
- (NSUInteger)size;
- (NSUInteger)alignment;
- (void)initializeTextureMemory:(void *)memory;
- (void)finalizeTextureMemory:(void *)memory;
- (void)copyFromLinearBytes:(const void *)bytes
               linearOffset:(NSUInteger)offset
          linearBytesPerRow:(NSUInteger)bytesPerRow
        linearBytesPerImage:(NSUInteger)bytesPerImage
            toTextureMemory:(void *)memory
               textureSlice:(NSUInteger)slice
               textureLevel:(NSUInteger)level
              textureRegion:(MTLRegion)region;
- (void)copyFromTextureMemory:(const void *)memory
                 textureSlice:(NSUInteger)slice
                 textureLevel:(NSUInteger)level
                textureRegion:(MTLRegion)region
                toLinearBytes:(void *)bytes
                 linearOffset:(NSUInteger)offset
            linearBytesPerRow:(NSUInteger)bytesPerRow
          linearBytesPerImage:(NSUInteger)bytesPerImage;
@end

@protocol WMRefTextureLayoutDevice
- (id<WMRefTextureLayout>)newTextureLayoutWithDescriptor:(MTLTextureDescriptor *)descriptor
                                    isHeapOrBufferBacked:(BOOL)heapOrBufferBacked;
@end

static void LayoutPattern(uint8_t *bytes, size_t length, uint32_t seed)
{
    uint32_t x = seed * 2654435761u + 1;
    for (size_t i = 0; i < length; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        bytes[i] = (uint8_t)x;
    }
}

// One level of a slice as tight rows of blocks.
typedef struct {
    NSUInteger width, height, depth;
    NSUInteger bytesPerRow, rows, bytesPerImage, length;
} LayoutLevel;

static LayoutLevel LayoutLevelOf(MTLTextureDescriptor *d, NSUInteger level, NSUInteger blockWidth,
                                 NSUInteger blockHeight, NSUInteger blockBytes)
{
    LayoutLevel l;
    l.width = MAX(d.width >> level, 1u);
    l.height = d.textureType == MTLTextureType1D || d.textureType == MTLTextureType1DArray ? 1
                                                                                          : MAX(d.height >> level, 1u);
    l.depth = d.textureType == MTLTextureType3D ? MAX(d.depth >> level, 1u) : 1;
    l.bytesPerRow = (l.width + blockWidth - 1) / blockWidth * blockBytes;
    l.rows = (l.height + blockHeight - 1) / blockHeight;
    l.bytesPerImage = l.bytesPerRow * l.rows;
    l.length = l.bytesPerImage * l.depth;
    return l;
}

static NSUInteger LayoutSlices(MTLTextureDescriptor *d)
{
    switch (d.textureType) {
    case MTLTextureTypeCube: return 6;
    case MTLTextureTypeCubeArray: return 6 * d.arrayLength;
    case MTLTextureType1DArray:
    case MTLTextureType2DArray: return d.arrayLength;
    default: return 1;
    }
}

static void *LayoutMemory(WMCase *c, NSUInteger size)
{
    vm_address_t address = 0;
    if (vm_allocate(mach_task_self(), &address, MAX(size, 1u), VM_FLAGS_ANYWHERE) != KERN_SUCCESS)
        [c fail:@"vm_allocate of %lu bytes failed", (unsigned long)size];
    return (void *)address;
}

static void LayoutFree(void *memory, NSUInteger size)
{
    vm_deallocate(mach_task_self(), (vm_address_t)memory, MAX(size, 1u));
}

static id<WMRefTextureLayout> LayoutFor(WMCase *c, MTLTextureDescriptor *d)
{
    id device = c.device;
    id<WMRefTextureLayout> layout = [(id<WMRefTextureLayoutDevice>)device newTextureLayoutWithDescriptor:d
                                                                                  isHeapOrBufferBacked:NO];
    if (!layout) [c fail:@"newTextureLayoutWithDescriptor:isHeapOrBufferBacked: answered nil"];
    return layout;
}

// Fills every slice and level of the memory through the layout, from the pattern of (slice, level).
static void LayoutFill(id<WMRefTextureLayout> layout, MTLTextureDescriptor *d, void *memory, NSUInteger bw,
                       NSUInteger bh, NSUInteger bb, uint32_t seed)
{
    for (NSUInteger s = 0; s < LayoutSlices(d); s++)
        for (NSUInteger level = 0; level < d.mipmapLevelCount; level++) {
            LayoutLevel l = LayoutLevelOf(d, level, bw, bh, bb);
            NSMutableData *src = [NSMutableData dataWithLength:l.length];
            LayoutPattern(src.mutableBytes, l.length, seed + (uint32_t)(s * 64 + level));
            [layout copyFromLinearBytes:src.bytes linearOffset:0 linearBytesPerRow:l.bytesPerRow
                    linearBytesPerImage:l.bytesPerImage toTextureMemory:memory textureSlice:s textureLevel:level
                          textureRegion:MTLRegionMake3D(0, 0, 0, l.width, l.height, l.depth)];
        }
}

static NSData *LayoutRead(id<WMRefTextureLayout> layout, MTLTextureDescriptor *d, const void *memory, NSUInteger slice,
                          NSUInteger level, NSUInteger bw, NSUInteger bh, NSUInteger bb)
{
    LayoutLevel l = LayoutLevelOf(d, level, bw, bh, bb);
    NSMutableData *out = [NSMutableData dataWithLength:l.length];
    [layout copyFromTextureMemory:memory textureSlice:slice textureLevel:level
                    textureRegion:MTLRegionMake3D(0, 0, 0, l.width, l.height, l.depth) toLinearBytes:out.mutableBytes
                     linearOffset:0 linearBytesPerRow:l.bytesPerRow linearBytesPerImage:l.bytesPerImage];
    return out;
}

static MTLTextureDescriptor *LayoutDescriptor(MTLTextureType type, MTLPixelFormat format, NSUInteger w, NSUInteger h,
                                              NSUInteger depth, NSUInteger levels, NSUInteger arrayLength)
{
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = type;
    d.pixelFormat = format;
    d.width = w;
    d.height = h;
    d.depth = depth;
    d.mipmapLevelCount = levels;
    d.arrayLength = arrayLength;
    d.storageMode = MTLStorageModeShared;
    d.usage = MTLTextureUsageShaderRead;
    return d;
}

WMREF_CASE(textureLayoutCopies, "spi.texture-layout-copies", "spi.texture-layout",
           "A texture layout's copies into and out of texture memory: every slice and level, and a sub-region")
{
    [c require:@"spi.texture-layout"
         value:[(id)c.device respondsToSelector:@selector(newTextureLayoutWithDescriptor:isHeapOrBufferBacked:)]];
    struct {
        const char *name;
        MTLTextureType type;
        MTLPixelFormat format;
        NSUInteger w, h, depth, levels, array, bw, bh, bb;
    } cases[] = {
        {"2d-rgba8-37x21", MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 37, 21, 1, 6, 1, 1, 1, 4},
        {"2d-r8-64x64", MTLTextureType2D, MTLPixelFormatR8Unorm, 64, 64, 1, 2, 1, 1, 1, 1},
        {"2d-array-bgra8-16x16", MTLTextureType2DArray, MTLPixelFormatBGRA8Unorm, 16, 16, 1, 3, 2, 1, 1, 4},
        {"cube-rgba16f-8", MTLTextureTypeCube, MTLPixelFormatRGBA16Float, 8, 8, 1, 2, 1, 1, 1, 8},
        {"3d-rg8-9x7x5", MTLTextureType3D, MTLPixelFormatRG8Unorm, 9, 7, 5, 3, 1, 1, 1, 2},
        {"2d-rgba32f-300x200", MTLTextureType2D, MTLPixelFormatRGBA32Float, 300, 200, 1, 1, 1, 1, 1, 16},
        {"2d-astc4x4-40x24", MTLTextureType2D, MTLPixelFormatASTC_4x4_LDR, 40, 24, 1, 3, 1, 4, 4, 16},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        MTLTextureDescriptor *d = LayoutDescriptor(cases[i].type, cases[i].format, cases[i].w, cases[i].h,
                                                   cases[i].depth, cases[i].levels, cases[i].array);
        NSString *base = [NSString stringWithUTF8String:cases[i].name];
        [c step:base];
        id<WMRefTextureLayout> layout = LayoutFor(c, d);
        NSUInteger size = layout.size;
        [c value:[base stringByAppendingString:@".size"] u64:size rule:WMDevice()];
        [c value:[base stringByAppendingString:@".alignment"] u64:layout.alignment rule:WMDevice()];
        void *memory = LayoutMemory(c, size);
        [layout initializeTextureMemory:memory];
        NSUInteger bw = cases[i].bw, bh = cases[i].bh, bb = cases[i].bb;
        LayoutFill(layout, d, memory, bw, bh, bb, (uint32_t)i * 977);
        [layout finalizeTextureMemory:memory];
        for (NSUInteger s = 0; s < LayoutSlices(d); s++)
            for (NSUInteger level = 0; level < d.mipmapLevelCount; level++) {
                NSData *back = LayoutRead(layout, d, memory, s, level, bw, bh, bb);
                [c data:[NSString stringWithFormat:@"%@.slice%lu.level%lu", base, (unsigned long)s,
                                                   (unsigned long)level]
                   elem:WMElemU8 bytes:back.bytes length:back.length rule:WMExact()];
            }
        // A sub-region of the last slice's first level from rows with a gap and an offset, then the whole level.
        LayoutLevel l = LayoutLevelOf(d, 0, bw, bh, bb);
        NSUInteger x = bw, y = bh, w = MIN(3 * bw, l.width - x), h = MIN(2 * bh, l.height - y);
        NSUInteger pitch = (w + bw - 1) / bw * bb + 12, rows = (h + bh - 1) / bh;
        NSMutableData *sub = [NSMutableData dataWithLength:5 + pitch * rows];
        LayoutPattern(sub.mutableBytes, sub.length, 4099 + (uint32_t)i);
        [layout copyFromLinearBytes:sub.bytes linearOffset:5 linearBytesPerRow:pitch linearBytesPerImage:pitch * rows
                    toTextureMemory:memory textureSlice:LayoutSlices(d) - 1 textureLevel:0
                      textureRegion:MTLRegionMake3D(x, y, 0, w, h, 1)];
        NSData *after = LayoutRead(layout, d, memory, LayoutSlices(d) - 1, 0, bw, bh, bb);
        [c data:[base stringByAppendingString:@".sub-region-written"] elem:WMElemU8 bytes:after.bytes
          length:after.length rule:WMExact()];
        LayoutFree(memory, size);
    }
}
