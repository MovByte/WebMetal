// Texture memory layout of the A10X Metal driver (AGXMetalA10, iPadOS 17.7.11, 21H461). See WebMetalTextureLayout.h.
#import "WebMetalTextureLayout.h"
#import "../Plugin/WebMetalReport.h"

#include <algorithm>
#include <climits>
#include <initializer_list>

namespace webmetal::agx {
namespace {

#include "WebMetalTextureFormats.inc"

// A layout the port does not cover fails the call through the plugin's not-implemented channel.
[[noreturn]] void unported(const char *what) {
    WMReport("the A10X texture layout (AGX::TextureGen2)", what);
    [NSException raise:WMNotImplementedException format:@"the A10X texture layout: %s is not implemented", what];
    __builtin_unreachable();
}

// AArch64 semantics of the operations the driver code uses.
uint32_t clz32(uint32_t v) { return v ? static_cast<uint32_t>(__builtin_clz(v)) : 32; }
uint32_t lsl32(uint32_t v, uint32_t s) { return v << (s & 31); }
uint32_t lsr32(uint32_t v, uint32_t s) { return v >> (s & 31); }
uint64_t lsl64(uint64_t v, uint64_t s) { return v << (s & 63); }
uint32_t udiv32(uint32_t n, uint32_t d) { return d ? n / d : 0; }
uint64_t udiv64(uint64_t n, uint64_t d) { return d ? n / d : 0; }
int32_t sdiv32(int32_t n, int32_t d) {
    if (d == 0) return 0;
    if (n == INT32_MIN && d == -1) return INT32_MIN;
    return n / d;
}
uint64_t umulh(uint64_t a, uint64_t b) { return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64); }
// add x, v, a; sub x, x, #1; neg y, a; and x, x, y
uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) & (0 - a); }
// cmp w, #1; csinc w, w, wzr, hi
uint32_t atLeastOne(uint32_t v) { return v > 1 ? v : 1; }
// The power of two at or above v: subs w, v, #1; clz; neg; lsl w, 1, w; csinc w, w, wzr, hi
uint32_t ceilPow2(uint32_t v) { return v > 1 ? lsl32(1, 0 - clz32(v - 1)) : 1; }
// sub w, v, #1; clz; neg; lsl w, -1, w; mvn; clz; 32 - w: the bit length of v - 1 (0 when it is 32)
uint32_t bitLengthOfPredecessor(uint32_t v) {
    uint32_t c = clz32(v - 1);
    return 32 - clz32(~lsl32(0xffffffffu, 0 - c));
}
// Round v up to a multiple of 1 << log2: lsl w, -1, log2; mvn; add; and
uint32_t alignLog2Up(uint32_t v, uint32_t log2) {
    uint32_t mask = lsl32(0xffffffffu, log2);
    return (v + ~mask) & mask;
}
uint64_t bfi(uint64_t dst, uint64_t src, unsigned lsb, unsigned width) {
    uint64_t mask = ((uint64_t(1) << width) - 1) << lsb;
    return (dst & ~mask) | ((src << lsb) & mask);
}

bool inSet(uint64_t v, std::initializer_list<uint64_t> set) { return std::find(set.begin(), set.end(), v) != set.end(); }

// The depth/stencil formats the allocation padding applies to: pixelFormat - 0xfa <= 0xb and bit set in 0xc0d.
bool paddedDepthStencilFormat(uint64_t pixelFormat) {
    uint64_t i = pixelFormat - 0xfa;
    return i <= 0xb && ((uint64_t(1) << i) & 0xc0d) != 0;
}

// The allocation padding rule shared by the ctor, getLevelOffset and isDepthStencilAllocTypePadded: a padded
// depth/stencil format, smaller than 32 in width or height, render-target-only usage (usage & 5 == 4), one level.
bool depthStencilPadRule(const TextureGen2 &t) {
    return paddedDepthStencilFormat(t.pixelFormat) && (t.width < 0x20 || t.height <= 0x1f) && (t.usage & 5) == 4 &&
           t.levels <= 1;
}

const FormatInfo &formatOf(const TextureGen2 &t) {
    if (!t.format) unported("a layout whose format pointer is null (interchange-tiled kind 3)");
    return *t.format;
}

uint8_t planeBytes(const FormatInfo &f, uint32_t plane) {
    if (plane >= 2) unported("format plane index above 1");
    return f.planeBytes[plane];
}

// agxsTwiddledBlockInfo::getParamsForLevel (0x21c913f64).
uint64_t paramsForLevel(const TwiddledBlockInfo &in, uint32_t level, uint32_t &o2, uint32_t &o3, uint32_t &o4,
                        uint32_t &o5) {
    uint32_t w13 = in.widthLog2;
    uint32_t w8 = w13 < level ? 0 : w13 - level;
    uint32_t w14 = in.heightLog2;
    uint32_t w9 = w14 < level ? 0 : w14 - level;
    uint32_t w10 = in.mode;
    uint32_t w11 = lsr32(in.width, level);
    uint32_t w12 = lsr32(in.height, level);
    uint32_t w15, w16;
    if (w10 - 1 <= 1) {
        w11 = atLeastOne(udiv32(in.blockWidth + w11 - 1, in.blockWidth));
        w12 = atLeastOne(udiv32(in.blockHeight + w12 - 1, in.blockHeight));
        w15 = lsl32(1, w8);
        uint32_t w7 = lsl32(1, w9);
        uint32_t s1 = in.tileHeightLog2;
        uint32_t s0 = in.tileWidthLog2;
        if (lsl32(1, s1) < w7 && lsr32(w15, s0) == 0) {
            uint32_t bits = 32 - clz32(~lsl32(0xffffffffu, 0 - clz32(w15 - 1)));
            w15 = w13 <= level ? 0 : bits;
            w13 = s0 + s1 - w15;
        } else {
            // 0x21c914174
            uint32_t shifted = lsr32(w7, s1);
            uint32_t one = lsl32(1, s0);
            uint32_t negBits = clz32(~lsl32(0xffffffffu, 0 - clz32(w7 - 1))) - 32;
            w14 = w14 <= level ? 0 : negBits;
            uint32_t sum = s0 + s1 + w14;
            bool hs = one >= w15;
            w15 = hs ? s0 : sum;
            w14 = hs ? s1 : 0 - w14;
            if (shifted != 0) {
                w15 = s0;
                w13 = s1;
            } else {
                w13 = w14;
            }
        }
        // 0x21c9141c4
        w16 = w8;
        w14 = w9;
    } else {
        w11 = udiv32(in.blockWidth + atLeastOne(w11) - 1, in.blockWidth);
        w12 = udiv32(in.blockHeight + atLeastOne(w12) - 1, in.blockHeight);
        uint32_t a = in.alignLog2;
        uint32_t unit = lsl32(1, a);
        bool hi = unit <= w11 ? unit > w12 : true;
        uint32_t mask = lsl32(0xffffffffu, a);
        uint32_t aw = (w11 + ~mask) & mask;
        uint32_t ah = (w12 + ~mask) & mask;
        uint32_t w17 = hi ? w11 : aw;
        uint32_t w16a = hi ? w12 : ah;
        if (a != 0) {
            w11 = w17;
            w12 = w16a;
        }
        uint32_t s1 = in.tileHeightLog2;
        uint32_t s0 = in.tileWidthLog2;
        uint32_t tileH = lsl32(1, s1);
        uint32_t wTiles = lsr32(w11, s0);
        uint32_t hTiles = lsr32(w12, s1);
        uint32_t tileW = lsl32(1, s0);
        uint32_t c7 = clz32(w12 - 1);
        uint32_t w19 = clz32(~lsl32(0xffffffffu, 0 - c7)) - 32;
        if (w12 < 2) w19 = 0;
        uint32_t w20 = s0 + s1;
        uint32_t w21 = w20 + w19;
        bool hs = hTiles == 0 ? tileW >= w11 : true;
        w15 = hs ? s0 : w21;
        w13 = hs ? s1 : 0 - w19;
        uint32_t c16 = clz32(w11 - 1);
        uint32_t w6 = 32 - clz32(~lsl32(0xffffffffu, 0 - c16));
        if (w11 < 2) w6 = 0;
        w20 = w20 - w6;
        bool ne = tileH < w12 ? wTiles != 0 : true;
        w15 = ne ? w15 : w6;
        w13 = ne ? w13 : w20;
        w16 = 32 - clz32(~lsl32(0xffffffffu, 0 - c16));
        if (w11 < 2) w16 = 0;
        w14 = 32 - clz32(~lsl32(0xffffffffu, 0 - c7));
        if (w12 < 2) w14 = 0;
    }
    // 0x21c9141cc
    o2 = w16 < w15 ? 0 : w16 - w15;
    o3 = w14 < w13 ? 0 : w14 - w13;
    o4 = lsr32(w11 + ~lsl32(0xffffffffu, w15), w15);
    o5 = lsr32(w12 + ~lsl32(0xffffffffu, w13), w13);
    uint32_t k = in.tileMode;
    if (w10 == 2 && k == 0) return lsl64(lsl64(1, w9), w9) * in.bytesPerBlock;
    uint64_t blocks = lsl64(lsl64(1, w9), w8);
    if (k == 4 || k == 3 || k == 2) return (blocks * in.bytesPerBlock + 0x7f) & ~uint64_t(0x7f);
    return blocks * in.bytesPerBlock;
}

// agxsTwiddledBlockInfo::getGPUVirtualLengthWithinLayer (0x21c914288).
uint64_t gpuVirtualLengthWithinLayer(const TwiddledBlockInfo &in, uint32_t first, uint32_t count) {
    uint32_t a = in.widthLog2 < first ? 0 : in.widthLog2 - first;
    uint32_t b = in.heightLog2 < first ? 0 : in.heightLog2 - first;
    if (count == 0) return 0;
    uint64_t bpb = in.bytesPerBlock;
    uint64_t acc = 0;
    if (in.mode == 2 && in.tileMode == 0) {
        for (; count; count--) {
            acc += lsl64(lsl64(1, b), b) * bpb;
            b = b == 0 ? 0 : b - 1;
        }
        return acc;
    }
    bool aligned = in.tileMode == 4 || in.tileMode == 3 || in.tileMode == 2;
    for (; count; count--) {
        uint64_t v = lsl64(lsl64(1, b), a) * bpb;
        acc += aligned ? (v + 0x7f) & ~uint64_t(0x7f) : v;
        a = a == 0 ? 0 : a - 1;
        b = b == 0 ? 0 : b - 1;
    }
    return acc;
}

// getLevelOffset<View0> (0x21c9137f4) and the non-twiddled part of getLevelOffset<View1> (0x21c913cc8).
uint64_t levelOffsetLinear(const TextureGen2 &t, uint32_t level, uint32_t plane) {
    const FormatInfo &f = formatOf(t);
    bool pad = depthStencilPadRule(t);
    uint32_t w = pad ? alignLog2Up(t.width, f.alignLog2) : t.width;
    uint32_t w1 = udiv32(t.blockWidth + atLeastOne(w) - 1, t.blockWidth);
    uint32_t h = pad ? alignLog2Up(t.height, f.alignLog2) : t.height;
    uint32_t w2 = udiv32(t.blockHeight + atLeastOne(h) - 1, t.blockHeight);
    uint8_t kind = t.kind;
    if (kind != 0 && t.dim != 1 && kind != 3) {
        bool check = false;
        if (t.dim == 4) {
            check = true;
        } else if (t.dim == 2) {
            check = t.arrayBits == 0 || (t.arrayBits & 3) != 0;
        }
        // getLevelOffset<View1> calls isDepthStencilAllocTypePadded; getLevelOffset<View0> inlines the same test.
        if (check && f.alignLog2 != 0 && isDepthStencilAllocTypePadded(t)) {
            w1 = alignLog2Up(w1, f.alignLog2);
            w2 = alignLog2Up(w2, f.alignLog2);
        }
    }
    if (kind == 2) unported("IOSurface tiled layout (kind 2): sumLevelSizesTiled 0x21c9c0bcc");
    if (level == 0) return 0;
    uint64_t bpb = uint64_t(t.samples) * planeBytes(f, plane);
    uint32_t e4 = t.sparsePageSize;
    uint32_t hh = ceilPow2(w2);
    uint32_t ww = ceilPow2(w1);
    uint64_t acc = 0;
    for (uint32_t n = level; n; n--) {
        uint64_t s = bpb * uint64_t(uint32_t(hh * ww));
        if (e4 == 0) {
            acc += s;
        } else {
            uint64_t a = s >= e4 ? e4 : 1;
            acc += alignUp(s, a);
        }
        hh = hh > 1 ? hh >> 1 : hh;
        ww = ww > 1 ? ww >> 1 : ww;
    }
    return acc;
}

// The tail of getMipmapSize<View0>/<View1> after the level offset (0x21c913b00, 0x21c913730).
uint64_t mipmapAlignment(const TextureGen2 &t, uint64_t r) {
    uint8_t kind = t.kind;
    if (kind == 0) return r;
    uint32_t dim = t.dim;
    if (dim - 1 > 3) return r;
    uint8_t a = t.arrayBits;
    switch (dim) {
    case 3:
        if (a != 0) return r;
        break;
    case 4:
        if (a != 0 && !(a & 1)) return r;
        break;
    default:  // 1, 2
        if (!(a & 1)) return r;
        break;
    }
    if (t.resolved) return 0;
    uint32_t a0 = a & 1;
    uint32_t cond;
    if (dim - 3 < 2) {
        cond = 1;
    } else if ((dim == 1) & a0) {
        cond = 1;
    } else {
        cond = (dim == 2) & a0;
    }
    bool big = (r & ~uint64_t(0x3fff)) != 0 && kind == 1 && cond;
    return alignUp(r, big ? 0x4000 : 0x40);
}

enum class View { Zero, One };

uint64_t planeSize(const TextureGen2 &t, uint32_t plane, View view) {
    auto mip = [&](uint32_t p) { return view == View::One ? getMipmapSize1(t, p) : getMipmapSize0(t, p); };
    auto stride = [&](uint32_t p) { return view == View::One ? getStride1(t, p) : getStride0(t, 0, p); };
    const FormatInfo &f = formatOf(t);
    uint64_t r;
    uint64_t e4 = t.sparsePageSize;
    if (e4 != 0) {
        uint64_t m0 = getMipmapSize0(t, 0);
        uint32_t layers;
        if (t.dim == 3) {
            layers = t.depth;
        } else if ((t.arrayBits & 1) || t.dim == 4) {
            layers = t.dim == 4 ? t.arrayLength * 6 : t.arrayLength;
        } else {
            layers = 1;
        }
        uint64_t unit = uint32_t(uint32_t(e4) << 8);
        uint64_t pages = udiv64(e4 + alignUp(m0, unit) - 1, e4);
        r = ((pages << 3) + 0x7ff & ~uint64_t(0x7ff)) * layers;
        if (t.sparseParams) unported("calcSparseParams 0x21c985de0 (this+0xe8 set)");
    } else if (t.arrayBits & 1) {
        uint64_t mm = mip(plane);
        uint64_t layer = t.dim == 4 ? (e4 ? alignUp(mm, e4) : mm) * 6 : mm;
        r = (e4 ? alignUp(layer, e4) : layer) * t.arrayLength;
    } else if (t.dim == 4) {
        uint64_t mm = mip(plane);
        r = (e4 ? alignUp(mm, e4) : mm) * 6;
    } else if (t.kind == 3) {
        unported("interchange-tiled plane size (InterchangeTiledMemory::getPlaneSize 0x21c9c59e0)");
    } else if (t.kind == 0) {
        if (t.iosurface) unported("IOSurfaceGetHeightOfPlane");
        uint64_t rows = (plane == 1 && f.chroma == 3) ? uint64_t(t.height >> 1) : uint64_t(t.height);
        uint64_t s = stride(plane);
        r = s * rows;
        if (f.planes - 1 == plane) r = (r - s) + uint64_t(planeBytes(f, plane)) * t.width;
    } else {
        uint32_t dim = t.dim;
        if (dim - 1 < 2) {
            r = mip(plane);
        } else if (dim == 3) {
            uint64_t mm = mip(plane);
            r = (e4 ? alignUp(mm, e4) : mm) * t.depth;
        } else {
            r = 0;
        }
    }
    if (f.planes - 1 != plane) r = alignUp(r, t.pageAlignment);
    return r;
}

TextureAllocation initImpl(const DescriptorFields &d, bool allowNPOT, bool heapBacked, uint32_t sparsePageSize,
                           const SystemValues &system);
TextureGen2Args argsFromDescriptor(const DescriptorFields &d);

}  // namespace

const FormatInfo *chooseTextureFormat(uint64_t pixelFormat) {
    uint32_t i = uint32_t(pixelFormat) - 1;
    if (i > kChooseTextureFormatLimit) return &kUnsupportedFormat;
    uint32_t pf = uint32_t(pixelFormat);
    const FormatInfo *end = kFormats + sizeof(kFormats) / sizeof(kFormats[0]);
    const FormatInfo *e =
        std::lower_bound(kFormats, end, pf, [](const FormatInfo &f, uint32_t v) { return f.pixelFormat < v; });
    if (e == end || e->pixelFormat != pf) return &kUnsupportedFormat;
    return e;
}

bool isUnsupported(const FormatInfo *format) { return format == &kUnsupportedFormat; }

bool isYUVTexture(uint64_t pixelFormat) {
    uint64_t i = pixelFormat - 0xf0;
    if (i > 0x17c) return false;
    return inSet(pixelFormat, {0xf0, 0xf1, 0x1f4, 0x1f5, 0x1f6, 0x1f7, 0x1f8, 0x1f9, 0x1fa, 0x1fb, 0x1fc, 0x1fd, 0x1fe,
                               0x208, 0x209, 0x20a, 0x20b, 0x20c, 0x20d, 0x20e, 0x20f, 0x210, 0x211, 0x212, 0x21c,
                               0x21d, 0x21e, 0x21f, 0x220, 0x221, 0x222, 0x223, 0x224, 0x230, 0x232, 0x233, 0x234,
                               0x235, 0x236, 0x237, 0x238, 0x239, 0x23a, 0x23b, 0x23c, 0x23d, 0x23e, 0x23f, 0x244,
                               0x245, 0x246, 0x247, 0x248, 0x249, 0x24c, 0x24d, 0x24e, 0x24f, 0x250, 0x251, 0x252,
                               0x253, 0x254, 0x255, 0x256, 0x257, 0x258, 0x259, 0x25a, 0x25f, 0x260, 0x261, 0x262,
                               0x263, 0x264, 0x265, 0x267, 0x268, 0x269, 0x26a, 0x26b, 0x26c});
}

bool isDepthStencilAllocTypePadded(const TextureGen2 &t) { return depthStencilPadRule(t); }

uint64_t getLevelOffset0(const TextureGen2 &t, uint32_t level, uint32_t plane) {
    return levelOffsetLinear(t, level, plane);
}

uint64_t getLevelOffset1(const TextureGen2 &t, uint32_t level, uint32_t plane) {
    if (!t.twiddledPlanes && !t.sparseParams) return levelOffsetLinear(t, level, plane);
    if (plane >= 2) unported("twiddled block info of a plane above 1");
    const TwiddledBlockInfo &info = t.blockInfo[plane];
    uint32_t lvl = 0;
    uint64_t acc = 0;
    if (level != 0) {
        for (;;) {
            uint32_t o2 = 0, o3 = 0, o4 = 0, o5 = 0;
            uint64_t r = paramsForLevel(info, lvl, o2, o3, o4, o5);
            uint64_t limit = umulh(r << 2, 0xaaaaaaaaaaaaaaabULL) >> 1;
            uint32_t size = o5 * o4 * uint32_t(info.pagesPerTile) * uint32_t(info.pageAlign);
            if (uint64_t(size) >= limit) break;
            acc += size;
            lvl++;
            if (lvl == level) break;
        }
    }
    return gpuVirtualLengthWithinLayer(info, lvl, level - lvl) + acc;
}

uint64_t getMipmapSize1(const TextureGen2 &t, uint32_t plane) {
    if (t.kind == 0 || t.kind == 3) return 0;
    uint64_t r = getLevelOffset1(t, t.levels, plane);
    if (t.twiddledPlanes) return alignUp(r, t.pageAlignment);
    return mipmapAlignment(t, r);
}

uint64_t getMipmapSize0(const TextureGen2 &t, uint32_t plane) {
    if (t.kind == 0 || t.kind == 3) return 0;
    uint64_t r = getLevelOffset0(t, t.levels, plane);
    if (t.twiddledPlanes) r = alignUp(r, t.pageAlignment);
    return mipmapAlignment(t, r);
}

uint64_t getStride1(const TextureGen2 &t, uint32_t plane) {
    if (t.kind == 3) unported("interchange-tiled stride (InterchangeTiledMemory::getPlaneInfo 0x21c9c5494)");
    if (t.kind != 0) return 0;
    if (t.iosurface) unported("IOSurfaceGetBytesPerRowOfPlane");
    uint64_t pf = t.pixelFormat;
    if (pf - 0x270 < 4 || pf - 0x226 <= 1)
        return (uint64_t(t.width) * planeBytes(formatOf(t), plane) + 0xf) & 0x1fffffffff0ULL;
    return t.bytesPerRow;
}

uint64_t getStride0(const TextureGen2 &t, uint32_t, uint32_t plane) {
    if (t.kind == 3) unported("interchange-tiled stride (InterchangeTiledMemory::getPlaneInfo 0x21c9c5494)");
    if (t.kind != 0) return 0;
    if (t.iosurface) unported("IOSurfaceGetBytesPerRowOfPlane");
    uint64_t pf = t.pixelFormat;
    const FormatInfo &f = formatOf(t);
    if (pf - 0x270 < 4 || pf - 0x226 <= 1) return (uint64_t(t.width) * planeBytes(f, plane) + 0xf) & 0x1fffffffff0ULL;
    uint64_t r = t.bytesPerRow;
    if (t.type == 9) {
        uint64_t cap = uint64_t(f.bytesPerPixel) << 14;
        if (cap < r) r = cap;
    }
    return r;
}

uint64_t getPlaneSize1(const TextureGen2 &t, uint32_t plane) { return planeSize(t, plane, View::One); }
uint64_t getPlaneSize0(const TextureGen2 &t, uint32_t plane) { return planeSize(t, plane, View::Zero); }

TextureGen2 construct(const TextureGen2Args &a, const SystemValues &system) {
    if (a.iosurface) unported("IOSurface-backed textures");
    TextureGen2 t;
    t.system = system;
    const FormatInfo *fmt = chooseTextureFormat(a.pixelFormat);
    uint32_t cls = fmt->cls - 2;
    t.classWord = (cls <= 0x10 && ((kClassWordMask >> cls) & 1)) ? kClassWords[cls] : 0;
    t.resolved = false;
    t.type = a.type;
    t.pixelFormat = a.pixelFormat;
    t.usage = a.usage;
    t.storageMode = a.storageMode;
    t.compressionMode = uint8_t(a.compression);
    t.compressionFootprint = uint8_t(a.compression >> 8);
    t.compressionByte2 = uint8_t(a.compression >> 16);
    t.samples = a.samples > 1 ? lsl32(1, 0 - clz32(a.samples - 1)) : 1;
    t.lastSlice = a.arrayLength - 1;
    // ushl.2s by -baseLevel (the signed low byte of each lane), then umax with 1
    auto ushl = [](uint32_t v, uint32_t shift) -> uint32_t {
        int s = int8_t(uint8_t(shift));
        if (s >= 0) return s >= 32 ? 0 : v << s;
        return -s >= 32 ? 0 : v >> -s;
    };
    t.baseWidth = std::max<uint32_t>(ushl(a.width, 0 - a.baseLevel), 1);
    t.baseHeight = std::max<uint32_t>(ushl(a.height, 0 - a.baseLevel), 1);
    t.baseDepth = atLeastOne(lsr32(a.depth, a.baseLevel));
    t.arrayLengthCopy = a.arrayLength;
    t.levelCount = a.levelCount;
    t.baseLevel = a.baseLevel;
    t.width = a.width;
    t.height = a.height;
    t.depth = a.depth;
    t.mipLevels = a.mipLevels;
    t.bytesPerRow = a.bytesPerRow;
    t.cpuCacheMode = a.cpuCacheMode;
    t.iosurface = a.iosurface;
    t.plane = a.plane;
    t.slice = a.slice;
    t.iosurface2 = a.iosurface2;
    t.rotation = a.rotation;
    t.swizzle = a.swizzle;
    t.valid = false;
    t.format = fmt;
    t.twiddledPlanes = false;
    t.sparsePageSize = a.sparsePageSize;
    t.arrayLength = a.arrayLength;
    t.compressionCopy[0] = t.compressionMode;
    t.compressionCopy[1] = t.compressionFootprint;
    t.compressionCopy[2] = t.compressionByte2;
    t.compressedBacking = a.compressedBacking;
    t.heapOrBufferBacked = a.heapOrBufferBacked;
    t.pageAlignment = system.iosurfaceAlignment;

    // this+0x3c: levels of a full chain from the largest dimension when more than one level is requested
    if (a.mipLevels < 2) {
        t.levels = 1;
    } else {
        uint32_t m = a.width > a.height ? a.width : a.height;
        uint32_t m3 = m > a.depth ? m : a.depth;
        m = !t.resolved ? m3 : m;
        t.levels = m == 0 ? 0 : 32 - clz32(m);
    }

    // Layout kind, dimensionality and array bits (0x21c9bef90..0x21c9bf198)
    uint64_t pf = t.pixelFormat;
    bool lossyCandidate = false;
    if (t.compressionFootprint != 0) {
        uint8_t fp = t.compressionFootprint;
        uint64_t x9 = pf - 0x228;
        bool typeCheck;
        if (x9 > 0x59) {
            typeCheck = fp != 3;
        } else if (inSet(pf, {0x240, 0x241, 0x242, 0x243, 0x25b, 0x25c, 0x25d, 0x276, 0x277, 0x278, 0x279, 0x27a,
                              0x27b, 0x27c})) {
            typeCheck = true;
        } else if (inSet(pf, {0x228, 0x229, 0x27d, 0x27e, 0x27f, 0x280, 0x281})) {
            typeCheck = fp != 1;
        } else {
            typeCheck = fp != 3;
        }
        if (typeCheck && t.type == 2 && t.levels <= 1) {
            if (pf - 1 <= 0x72) {
                lossyCandidate = inSet(pf, {0x1, 0xa, 0xb, 0xd, 0x14, 0x16, 0x17, 0x18, 0x19, 0x1e, 0x1f, 0x20,
                                            0x21, 0x22, 0x3c, 0x3e, 0x3f, 0x40, 0x41, 0x46, 0x47, 0x48, 0x49, 0x4a,
                                            0x50, 0x51, 0x54, 0x55, 0x5a, 0x5b, 0x5e, 0x6e, 0x70, 0x71, 0x72, 0x73});
            } else if (x9 <= 0x59) {
                lossyCandidate = inSet(pf, {0x228, 0x229, 0x22a, 0x22b, 0x240, 0x241, 0x242, 0x243, 0x25b, 0x25c,
                                            0x25d, 0x276, 0x277, 0x278, 0x279, 0x27a, 0x27b, 0x27c, 0x27d, 0x27e,
                                            0x27f, 0x280, 0x281});
            }
        }
    }
    bool yuv = isYUVTexture(pf);
    uint8_t kind = 1, dim = 2, arrayBits = yuv ? 4 : 0;
    if (lossyCandidate) {
        // 0x21c9bf130
        kind = 1;
        dim = 2;
    } else if (t.type <= 9) {
        switch (t.type) {
        case 0: dim = 1; break;
        case 1: arrayBits = 1; dim = 1; break;
        case 2: break;
        case 3: dim = 2; arrayBits = 1; break;
        case 4: arrayBits = 2; dim = 2; break;
        case 5: dim = 4; break;
        case 6: arrayBits = 1; dim = 4; break;
        case 7: dim = 3; break;
        case 8: dim = 2; arrayBits = 3; break;
        case 9: kind = 0; dim = 2; break;
        }
    }
    if (!t.compressedBacking) {
        // 0x21c9bf0b4 without an IOSurface: rows given by bytesPerRow, and some formats, are linear
        if (t.bytesPerRow != 0 || pf - 0x270 < 4 || pf - 0x226 < 2) {
            kind = 0;
        } else {
            if (lossyCandidate) kind = 3;
            if (yuv) kind = 0;
        }
    }
    t.kind = kind;
    t.dim = dim;
    t.arrayBits = arrayBits;

    auto invalid = [&]() -> TextureGen2 {
        // 0x21c9bf378
        t.valid = false;
        t.format = &kUnsupportedFormat;
        return t;
    };

    const FormatInfo *f = fmt;
    if (kind == 3) {
        f = nullptr;
        t.format = nullptr;
    } else if (isUnsupported(fmt)) {
        return invalid();
    }
    // Pixel-format checks (0x21c9bf1ec)
    if (pf - 0xa0 <= 0x1cb) {
        if (inSet(pf, {0x1f4, 0x1f5, 0x1f6, 0x1f9, 0x1fa, 0x1fc, 0x1fd, 0x208, 0x209, 0x20a, 0x20d, 0x20e, 0x210,
                       0x211, 0x21c, 0x21d, 0x21f, 0x220, 0x222, 0x223, 0x232, 0x234, 0x235, 0x237, 0x238, 0x23a,
                       0x23b, 0x23d, 0x23e, 0x244, 0x245, 0x247, 0x248, 0x24f, 0x250, 0x252, 0x253, 0x255, 0x256,
                       0x258, 0x259, 0x260, 0x261, 0x263, 0x264, 0x267, 0x268, 0x26a, 0x26b})) {
            if (((t.width & 1) || (t.height & 1)) && (t.usage == 0 || (t.usage & 6) != 0)) return invalid();
        } else if (pf - 0xa0 <= 7) {
            if (__builtin_popcount(t.width) != 1 || __builtin_popcount(t.height) != 1) return invalid();
        }
    }
    if (!f) unported("interchange-tiled layout (kind 3) without an IOSurface: the driver reads format+0x18 through "
                     "the null format pointer it has just stored");
    if (f->planes >= 2 && !t.iosurface && f->planesWithoutSurface == 0 && pf - 0x270 >= 4 && pf - 0x226 >= 2 &&
        !t.resolved)
        return invalid();
    if (t.sparsePageSize != 0 && kind != 1) return invalid();
    if (t.compressionFootprint != 0 && kind == 0) return invalid();

    t.valid = true;
    uint64_t q = pf - 0xb9;
    t.twiddleCompressed = q < 0x34 && ((0x000ffff3fffcffffULL >> (q & 63)) & 1);
    t.pvrtc = (pf & ~uint64_t(7)) == 0xa0;
    t.blockWidth = f->blockWidth;
    t.blockHeight = f->blockHeight;
    t.flag128 = pf - 0x28 < 5;
    {
        uint64_t r = pf - 0x24a;
        if (!(arrayBits & 4)) {
            if (r <= 0x2b && ((uint64_t(1) << r) & 0xc0000000003ULL)) t.colorSpace = a.colorConversion ? a.colorConversion : 7;
        } else {
            if (r <= 0x2b && ((uint64_t(1) << r) & 0xc2000000003ULL)) {
                t.colorSpace = a.colorConversion ? a.colorConversion : 7;
            } else {
                t.colorSpace = a.colorConversion ? a.colorConversion : 0;
            }
            t.chromaSiting = 0;
        }
    }
    // this+0x90 for linear layouts (0x21c9bf460)
    if (kind == 0) {
        uint64_t bpr = t.bytesPerRow;
        if ((a.pixelFormat & ~uint64_t(1)) == 0xf0 && bpr == 0) {
            bpr = (uint64_t(f->bytesPerPixel) * t.width + 0xf) & 0x1fffffffff0ULL;
            t.bytesPerRow = bpr;
        }
        if (bpr == 0) t.bytesPerRow = uint64_t(f->bytesPerPixel) * t.width;
    }
    if (t.type == 9) t.bytesPerRow = (t.bytesPerRow + 0xf) & ~uint64_t(0xf);
    t.faces = dim == 4 ? 6 : 1;
    t.sparseParams = false;

    bool twiddled = false;
    if (t.sparsePageSize == 0 && a.allowTwiddledPlanes && kind == 1) {
        // 0x21c9bf508
        bool loop = false;
        if (uint32_t(dim - 2) <= 2) {
            bool pad = depthStencilPadRule(t);
            uint32_t bw = t.blockWidth;
            uint32_t w = atLeastOne(pad ? alignLog2Up(t.width, f->alignLog2) : t.width);
            if (bw > bw - 1 + w) {
                loop = true;
            } else {
                uint32_t nw = udiv32(bw - 1 + w, bw);
                if ((nw - 1) & nw) {
                    loop = true;
                } else {
                    uint32_t bh = t.blockHeight;
                    uint32_t h = atLeastOne(pad ? alignLog2Up(t.height, f->alignLog2) : t.height);
                    if (bh > bh - 1 + h) {
                        loop = true;
                    } else {
                        uint32_t nh = udiv32(bh - 1 + h, bh);
                        loop = ((nh - 1) & nh) != 0;
                    }
                }
            }
            if (loop && pf == 0x104 && (t.width < 0x40 || t.height < 0x40)) loop = false;
        }
        if (loop) {
            // 0x21c9bf960
            if (f->planes == 0) {
                twiddled = true;
            } else {
                bool w20 = true;
                for (uint32_t p = 0; p < f->planes; p++) {
                    if (!w20) continue;
                    if (p >= 2) unported("twiddled block info of a plane above 1");
                    TwiddledBlockInfo &info = t.blockInfo[p];
                    uint32_t layers;
                    if (dim == 3) {
                        layers = t.depth;
                    } else if ((arrayBits & 1) || dim == 4) {
                        layers = dim == 4 ? t.arrayLength * 6 : t.arrayLength;
                    } else {
                        layers = 1;
                    }
                    w20 = false;
                    uint32_t bpb32 = t.samples * f->planeBytes[p];
                    uint32_t bpb = bpb32 & 0xff;
                    uint64_t align = t.pageAlignment;
                    info.bytesPerBlock = uint8_t(bpb32);
                    info.width = uint16_t(t.width);
                    info.height = uint16_t(t.height);
                    info.layers = uint16_t(layers);
                    info.levels = uint8_t(t.levels);
                    info.pageAlign = uint16_t(align);
                    info.tileMode = 1;
                    info.alignLog2 = f->alignLog2;
                    info.blockWidth = uint8_t(t.blockWidth);
                    info.blockHeight = uint8_t(t.blockHeight);
                    info.reserved = 0;
                    info.cube = dim == 4;
                    info.volume = dim == 3;
                    uint32_t bw = t.blockWidth & 0xff;
                    uint32_t bh = t.blockHeight & 0xff;
                    uint32_t nw = udiv32(t.width + bw - 1, bw);
                    uint32_t nh = udiv32(t.height + bh - 1, bh);
                    uint32_t level0 = nw * bpb * nh;
                    uint64_t page = system.vmPageSize;
                    uint64_t align16 = align & 0xffff;
                    if (!(page <= level0 && page >= align16)) continue;
                    info.mode = t.twiddleCompressed ? (dim == 4 ? 2 : 1) : 0;
                    uint32_t p16 = uint32_t(align & 0xffff);
                    uint32_t span = (bpb * 0xaaaaaaabu < 0x55555556u) ? p16 + (p16 << 1) : p16;
                    span = udiv32(span, bpb);
                    uint32_t bits = span < 2 ? 0 : 32 - clz32(span - 1);
                    uint32_t s0 = bits >> 1;
                    uint32_t s1 = bits - s0;
                    info.tileWidthLog2 = uint8_t(s0);
                    info.tileHeightLog2 = uint8_t(s1);
                    if (t.twiddleCompressed) {
                        uint32_t s0b = (bits >> 1) & 0xff;
                        uint32_t s1b = s1 & 0xff;
                        s0 = s0b > 5 ? s0b : 5;
                        s1 = s1b > 5 ? s1b : 5;
                        info.tileWidthLog2 = uint8_t(s0);
                        info.tileHeightLog2 = uint8_t(s1);
                    }
                    uint32_t r21 = s0 & 0xff;
                    uint32_t r20 = s1 & 0xff;
                    info.pagesPerTile = uint8_t(sdiv32(int32_t(lsl32(bpb, r20 + r21)), int32_t(p16)));
                    info.widthLog2 = uint8_t(nw >= 2 ? bitLengthOfPredecessor(nw) : 0);
                    info.heightLog2 = uint8_t(nh >= 2 ? bitLengthOfPredecessor(nh) : 0);
                    if (lsr32(nw, r21) == 0) {
                        w20 = false;
                        continue;
                    }
                    w20 = lsr32(nh, r20) != 0;
                }
                twiddled = w20;
            }
        }
    }
    t.twiddledPlanes = twiddled;
    // The twiddle test above runs only without a sparse page size, so every path that reaches here with one
    // computes the sparse tile (0x21c9bfbc4).
    if (t.sparsePageSize != 0) {
        uint64_t denom = uint64_t(f->planeBytes[0]) * t.samples;
        uint64_t qn = udiv64(t.sparsePageSize, denom);
        uint32_t bits = qn < 2 ? 0 : 32 - clz32(uint32_t(qn) - 1);
        uint32_t hi = bits >> 1;
        uint32_t lo = bits - hi;
        t.sparseTileWidth = lsl32(f->blockWidth, hi);
        t.sparseTileHeight = lsl32(f->blockHeight, lo);
    }

    // Plane sizes (0x21c9bfc30)
    uint32_t planes = f->planes;
    if (planes == 0) {
        t.size = 0;
        t.size0 = 0;
        return t;
    }
    uint64_t s = 0;
    for (uint32_t p = 0; p < planes; p++) s += getPlaneSize1(t, p);
    t.size = s;
    s = 0;
    for (uint32_t p = 0; p < planes; p++) s += getPlaneSize0(t, p);
    t.size0 = s;
    if (t.resolved || planes < 2) return t;
    if (planes > 2) unported("per-plane offsets of a format with more than two planes (they overlap this+0x158)");
    t.planeEnd0[0] = getPlaneSize0(t, 0);
    t.planeEnd0[1] = getPlaneSize1(t, 0);
    return t;
}

uint64_t textureAlignment(const TextureGen2 &t) {
    if (t.resolved) return 0;
    if (t.twiddledPlanes) return t.pageAlignment;
    if (t.sparsePageSize != 0) return t.sparsePageSize;
    uint8_t kind = t.kind;
    uint32_t dim = t.dim;
    uint64_t mip0 = getMipmapSize0(t, 0);
    uint32_t a0 = t.arrayBits & 1;
    uint32_t cond;
    if (dim - 3 < 2) {
        cond = 1;
    } else if ((dim == 1) & a0) {
        cond = 1;
    } else {
        cond = (dim == 2) & a0;
    }
    if (kind == 0) return 0x10;
    return ((mip0 & ~uint64_t(0x3fff)) != 0 && kind == 1 && cond) ? 0x4000 : 0x40;
}

uint32_t compressionSettings(const DescriptorFields &d) {
    uint32_t mode;
    uint32_t footprint;
    if (d.compressionMode == 2) {
        mode = 2;
        footprint = 0;
    } else {
        mode = d.compressionMode == 1 ? 9 : 0;
        footprint = (d.compressionFootprint == 1 || d.compressionFootprint == 2 || d.compressionFootprint == 3)
                        ? uint32_t(d.compressionFootprint)
                        : 0;
    }
    uint32_t byte0 = d.allowGPUOptimizedContents ? mode : 2;
    byte0 |= chooseTextureFormat(d.pixelFormat)->depthStencil ? 1 : 0;
    return (0x1000000 | (footprint << 8) | byte0) & 0x3fb;
}

uint32_t colorConversion(const DescriptorFields &d) {
    uint64_t i = d.colorSpaceConversion - 1;
    return i < 10 ? kColorSpaceConversions[i] : 0;
}

uint32_t rotation(const DescriptorFields &d) { return d.rotation - 1 < 4 ? uint32_t(d.rotation) : 0; }

namespace {

TextureGen2Args argsFromDescriptor(const DescriptorFields &d) {
    TextureGen2Args a;
    a.type = d.textureType;
    a.pixelFormat = d.pixelFormat;
    a.usage = d.usage;
    a.storageMode = d.storageMode;
    a.compression = compressionSettings(d);
    a.colorConversion = colorConversion(d);
    a.rotation = rotation(d);
    a.swizzle = d.swizzle;
    a.width = uint32_t(d.width);
    a.height = uint32_t(d.height);
    a.depth = uint32_t(d.depth);
    a.mipLevels = uint32_t(d.mipmapLevelCount);
    a.samples = uint32_t(d.sampleCount);
    a.arrayLength = uint32_t(d.arrayLength);
    a.bytesPerRow = 0;
    a.sparsePageSize = 0;
    a.cpuCacheMode = d.resourceOptions & 0xf;
    a.iosurface = false;
    a.plane = 0;
    a.slice = 0;
    a.iosurface2 = false;
    a.baseLevel = 0;
    a.levelCount = uint32_t(d.mipmapLevelCount);
    return a;
}

TextureAllocation initImpl(const DescriptorFields &d, bool allowNPOT, bool heapBacked, uint32_t sparsePageSize,
                           const SystemValues &system) {
    TextureGen2Args a = argsFromDescriptor(d);
    a.sparsePageSize = sparsePageSize;
    a.allowTwiddledPlanes = allowNPOT;
    a.compressedBacking = false;   // no IOSurface and no buffer
    a.heapOrBufferBacked = heapBacked;
    TextureGen2 t = construct(a, system);
    if (!t.valid) return {false, 0, 0, 0, 0};
    return {true, t.size, textureAlignment(t), getStride1(t, 0), initNewTextureData58(t, 0)};
}

}  // namespace

SizeAndAlign heapTextureSizeAndAlign(const DescriptorFields &d, const SystemValues &system) {
    TextureGen2Args a = argsFromDescriptor(d);
    a.allowTwiddledPlanes = false;
    a.compressedBacking = false;
    a.heapOrBufferBacked = true;
    TextureGen2 t = construct(a, system);
    // The method reads this+0x158 and the alignment whether or not the layout is valid.
    return {t.size, textureAlignment(t)};
}

TextureAllocation initImplDevice(const DescriptorFields &d, const SystemValues &system) {
    return initImpl(d, true, false, 0, system);
}

TextureAllocation initImplHeap(const DescriptorFields &d, const SystemValues &system) {
    return initImpl(d, false, true, 0, system);
}

TextureAllocation initImplDeviceSparse(const DescriptorFields &d, uint32_t sparsePageSize, const SystemValues &system) {
    return initImpl(d, true, false, sparsePageSize, system);
}

uint64_t initNewTextureData58(const TextureGen2 &t, uint64_t previous) {
    const FormatInfo &f = formatOf(t);
    uint64_t x = (previous & ~uint64_t(1)) | uint64_t(t.twiddledPlanes);
    x = (x & ~uint64_t(0x1000)) | (uint64_t(t.twiddleCompressed) << 12);
    x = bfi(x, t.dim == 4, 13, 1);
    x = bfi(x, t.dim == 3, 14, 1);
    x = bfi(x, f.alignLog2, 4, 3);
    uint64_t samples = 0;
    if (t.arrayBits & 2) samples = t.samples < 2 ? 0 : ((0 - (clz32(t.samples - 1) << 1)) & 0xe);
    x = (x & ~uint64_t(0xe)) | samples;
    uint64_t planes = f.planes >= 2 ? uint64_t(f.planesWithoutSurface != 0) << 7 : 0;
    x = (x & ~uint64_t(0x80)) | planes;
    x = bfi(x, ((uint32_t(f.planeBytes[0]) << 8) + 0x300) >> 8, 8, 2);
    x = bfi(x, ((uint32_t(f.planeBytes[1]) << 10) + 0xc00) >> 10, 10, 2);
    x = bfi(x, uint32_t(t.blockWidth << 15) >> 15, 15, 4);
    x = bfi(x, uint32_t(t.blockHeight << 19) >> 19, 19, 4);
    uint64_t levels = uint64_t(uint32_t(t.levels << 23)) & 0x7800000;
    x &= 0xfffffffc787fffffULL;
    return levels | x | 0x80000000;
}

bool suballocationVeto(const TextureGen2 &t) { return t.twiddledPlanes; }

LayoutResult textureLayout(const DescriptorFields &d, bool heapOrBufferBacked, const SystemValues &system) {
    TextureGen2Args a = argsFromDescriptor(d);
    a.allowTwiddledPlanes = !heapOrBufferBacked;
    a.compressedBacking = false;
    a.heapOrBufferBacked = heapOrBufferBacked;
    TextureGen2 t = construct(a, system);
    if (!t.valid) return {false, 0, 0};
    return {true, t.size + 0x10, textureAlignment(t)};
}

}  // namespace webmetal::agx
