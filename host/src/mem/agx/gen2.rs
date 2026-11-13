//! `AGX::TextureGen2<AGXTextureMemoryLayout1, G9::Encoders, G9::Classes>` (the A10X driver's texture object): the
//! constructor (0x21c9becd4) and the `AGX::Texture` helpers that place mipmap levels, slices and planes in texture
//! memory, as the guest's port (`guest/Resources/WebMetalTextureLayout.mm`) has them, instruction for instruction.
//!
//! Arithmetic follows the AArch64 code: 32-bit operations wrap, register shifts take their amount modulo the register
//! width and division by zero yields zero.
//!
//! Not ported, and refused with [`Unported`]: IOSurface-backed textures (and with them the IOSurface tiled layout, kind
//! 2), the interchange-tiled layout (kind 3; without an IOSurface the driver itself reads through the null format
//! pointer it stores for that kind), `calcSparseParams` (the constructor clears this+0xe8, which gates it) and formats
//! with more than two planes (the 21H461 table has none).

use super::formats::{
    CHOOSE_TEXTURE_FORMAT_LIMIT, CLASS_WORD_MASK, CLASS_WORDS, COLOR_SPACE_CONVERSIONS, FORMATS, UNSUPPORTED_FORMAT,
};

/// A layout the port does not cover: what it is and where the driver handles it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unported(pub &'static str);

/// The fields of a format-table entry (`AGX::TextureFormat`) that the layout reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatInfo {
    /// `MTLPixelFormat` the entry is chosen for (0 for `_texFormatUnsupported`).
    pub pixel_format: u32,
    /// +0x18
    pub planes: u32,
    /// +0x20
    pub cls: u32,
    /// +0x30
    pub align_log2: u8,
    /// +0x31
    pub bytes_per_pixel: u8,
    /// 0x21c9e4c70[+0x38].first
    pub block_width: u32,
    /// 0x21c9e4c70[+0x38].second
    pub block_height: u32,
    /// (+0x3c & 0xc) != 0
    pub depth_stencil: bool,
    /// +0x48 (3: 4:2:0)
    pub chroma: u32,
    /// +0x50
    pub planes_without_surface: u32,
    /// +0x58, +0x59
    pub plane_bytes: [u8; 2],
}

/// `AGX::G9::TextureFormatTable::chooseTextureFormatOBJC` (0x21c9c1318). Never fails; unsupported formats return the
/// `_texFormatUnsupported` entry.
pub fn choose_texture_format(pixel_format: u64) -> &'static FormatInfo {
    let i = (pixel_format as u32).wrapping_sub(1);
    if i > CHOOSE_TEXTURE_FORMAT_LIMIT {
        return &UNSUPPORTED_FORMAT;
    }
    let pf = pixel_format as u32;
    match FORMATS.binary_search_by_key(&pf, |f| f.pixel_format) {
        Ok(k) => &FORMATS[k],
        Err(_) => &UNSUPPORTED_FORMAT,
    }
}

pub fn is_unsupported(format: &FormatInfo) -> bool {
    core::ptr::eq(format, &UNSUPPORTED_FORMAT)
}

/// Values of the running system that the layout reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemValues {
    /// libsystem_kernel `_vm_page_size` (iPadOS arm64: 16 KiB pages).
    pub vm_page_size: u64,
    /// `IOSurfaceAlignProperty(kIOSurfaceOffset, 1)`, the static local `iosurface_alignment` of the constructor.
    pub iosurface_alignment: u64,
}

impl Default for SystemValues {
    fn default() -> SystemValues {
        SystemValues { vm_page_size: 16384, iosurface_alignment: 16384 }
    }
}

/// The constructor's arguments, in Darwin argument order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureGen2Args {
    pub texture_type: u64,
    pub pixel_format: u64,
    pub usage: u64,
    pub storage_mode: u64,
    /// AGXTextureCompressionSettings (three bytes).
    pub compression: u32,
    /// eAGXColorSpaceConversion.
    pub color_conversion: u32,
    /// eAGXTextureRotation.
    pub rotation: u32,
    pub swizzle: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub samples: u32,
    pub array_length: u32,
    pub bytes_per_row: u64,
    pub sparse_page_size: u32,
    pub cpu_cache_mode: u64,
    /// An IOSurface is given (not ported).
    pub iosurface: bool,
    pub plane: u32,
    pub slice: u32,
    /// Stored only.
    pub iosurface2: bool,
    pub base_level: u32,
    pub level_count: u32,
    /// Stack +0x58: allowNPOT from initImpl, !isHeapOrBufferBacked from AGXTextureLayout.
    pub allow_twiddled_planes: bool,
    /// Stack +0x59: initImpl sets it for an IOSurface or buffer texture whose pixel format is in its case list at
    /// 0x21c9dd1fd (the block-compressed formats); the layout kind is then not forced to linear by bytesPerRow.
    pub compressed_backing: bool,
    /// Stack +0x5a.
    pub heap_or_buffer_backed: bool,
}

/// `agxsTwiddledBlockInfo`: one per plane at TextureGen2 +0xea, 0x18 bytes each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TwiddledBlockInfo {
    /// +0x0
    pub width: u16,
    /// +0x2
    pub height: u16,
    /// +0x4
    pub layers: u16,
    /// +0x6
    pub page_align: u16,
    /// +0x8
    pub levels: u8,
    /// +0x9
    pub pages_per_tile: u8,
    /// +0xa
    pub width_log2: u8,
    /// +0xb
    pub height_log2: u8,
    /// +0xc
    pub tile_width_log2: u8,
    /// +0xd
    pub tile_height_log2: u8,
    /// +0xe
    pub bytes_per_block: u8,
    /// +0xf
    pub align_log2: u8,
    /// +0x10: 0, or 1 (2 for cubes) for formats with this+0x134 set.
    pub mode: u8,
    /// +0x11: the constructor stores 1.
    pub tile_mode: u8,
    /// +0x12
    pub block_width: u8,
    /// +0x13
    pub block_height: u8,
    /// +0x14: the sparse flag the twiddle functions test (the constructor stores 0).
    pub sparse: u8,
    /// +0x15
    pub cube: u8,
    /// +0x16
    pub volume: u8,
}

/// The TextureGen2 object: the fields the layout writes and reads.
#[derive(Clone, Copy, Debug)]
pub struct TextureGen2 {
    pub system: SystemValues,
    /// +0x10 (stored 0 by this constructor).
    pub resolved: bool,
    /// +0x18
    pub texture_type: u64,
    /// +0x20
    pub pixel_format: u64,
    /// +0x28
    pub usage: u64,
    /// +0x30
    pub storage_mode: u64,
    /// +0x38
    pub compression_mode: u8,
    /// +0x39
    pub compression_footprint: u8,
    /// +0x3a
    pub compression_byte2: u8,
    /// +0x3c: the full mip chain length when mipLevels > 1, else 1.
    pub levels: u32,
    /// +0x4c
    pub last_slice: u32,
    /// +0x50
    pub base_width: u32,
    /// +0x54
    pub base_height: u32,
    /// +0x58
    pub base_depth: u32,
    /// +0x5c
    pub array_length_copy: u32,
    /// +0x60
    pub class_word: u32,
    /// +0x64
    pub level_count: u32,
    /// +0x68: the sample count rounded up to a power of two.
    pub samples: u32,
    /// +0x6c and +0x88 (the constructor stores the base level in both).
    pub base_level: u32,
    /// +0x78
    pub width: u32,
    /// +0x7c
    pub height: u32,
    /// +0x80
    pub depth: u32,
    /// +0x84
    pub mip_levels: u32,
    /// +0x90
    pub bytes_per_row: u64,
    /// +0x98
    pub cpu_cache_mode: u64,
    /// +0xa0
    pub iosurface: bool,
    /// +0xa8
    pub plane: u32,
    /// +0xac
    pub slice: u32,
    /// +0xb0
    pub iosurface2: bool,
    /// +0xb8
    pub rotation: u32,
    /// +0xbc
    pub swizzle: u32,
    /// +0xc0
    pub valid: bool,
    /// +0xc8 (null for the interchange-tiled kind).
    pub format: Option<&'static FormatInfo>,
    /// +0xd0
    pub block_width: u32,
    /// +0xd4
    pub block_height: u32,
    /// +0xd8
    pub sparse_tile_width: u32,
    /// +0xdc
    pub sparse_tile_height: u32,
    /// +0xe0
    pub twiddled_planes: bool,
    /// +0xe4
    pub sparse_page_size: u32,
    /// +0xe8
    pub sparse_params: bool,
    /// +0xea
    pub block_info: [TwiddledBlockInfo; 2],
    /// +0x128
    pub flag128: bool,
    /// +0x12c
    pub color_space: u32,
    /// +0x130
    pub chroma_siting: u32,
    /// +0x134
    pub twiddle_compressed: bool,
    /// +0x135
    pub pvrtc: bool,
    /// +0x138
    pub page_alignment: u64,
    /// +0x148, +0x150: getPlaneSize<View0>(0) and getPlaneSize<View1>(0) of a two-plane format.
    pub plane_end: [u64; 2],
    /// +0x158: the sum of getPlaneSize<View1>.
    pub size: u64,
    /// +0x160: the sum of getPlaneSize<View0>.
    pub size0: u64,
    /// +0x168
    pub faces: u32,
    /// +0x170
    pub array_length: u32,
    /// +0x174: 0 linear, 1 twiddled, 2 IOSurface tiled, 3 interchange tiled.
    pub kind: u8,
    /// +0x175: 1 1D, 2 2D, 3 3D, 4 cube.
    pub dim: u8,
    /// +0x176: bit 0 array, bit 1 multisample, bit 2 YUV.
    pub array_bits: u8,
    /// +0x177..+0x179
    pub compression_copy: [u8; 3],
    /// +0x17a
    pub compressed_backing: bool,
    /// +0x17b
    pub heap_or_buffer_backed: bool,
}

// AArch64 semantics of the operations the driver code uses.
pub(crate) fn clz32(v: u32) -> u32 {
    v.leading_zeros()
}
pub(crate) fn lsl32(v: u32, s: u32) -> u32 {
    v << (s & 31)
}
pub(crate) fn lsr32(v: u32, s: u32) -> u32 {
    v >> (s & 31)
}
fn lsl64(v: u64, s: u64) -> u64 {
    v << (s & 63)
}
pub(crate) fn udiv32(n: u32, d: u32) -> u32 {
    n.checked_div(d).unwrap_or(0)
}
fn udiv64(n: u64, d: u64) -> u64 {
    n.checked_div(d).unwrap_or(0)
}
fn sdiv32(n: i32, d: i32) -> i32 {
    if d == 0 { 0 } else { n.wrapping_div(d) }
}
fn umulh(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}
/// add x, v, a; sub x, x, #1; neg y, a; and x, x, y
pub(crate) fn align_up(v: u64, a: u64) -> u64 {
    v.wrapping_add(a).wrapping_sub(1) & 0u64.wrapping_sub(a)
}
/// cmp w, #1; csinc w, w, wzr, hi
pub(crate) fn at_least_one(v: u32) -> u32 {
    if v > 1 { v } else { 1 }
}
/// The power of two at or above v: subs w, v, #1; clz; neg; lsl w, 1, w; csinc w, w, wzr, hi
pub(crate) fn ceil_pow2(v: u32) -> u32 {
    if v > 1 { lsl32(1, 0u32.wrapping_sub(clz32(v - 1))) } else { 1 }
}
/// sub w, v, #1; clz; neg; lsl w, -1, w; mvn; clz; 32 - w: the bit length of v - 1 (0 when it is 32)
fn bit_length_of_predecessor(v: u32) -> u32 {
    let c = clz32(v.wrapping_sub(1));
    32 - clz32(!lsl32(0xffff_ffff, 0u32.wrapping_sub(c)))
}
/// Round v up to a multiple of 1 << log2: lsl w, -1, log2; mvn; add; and
pub(crate) fn align_log2_up(v: u32, log2: u32) -> u32 {
    let mask = lsl32(0xffff_ffff, log2);
    v.wrapping_add(!mask) & mask
}

/// The depth/stencil formats the allocation padding applies to: pixelFormat - 0xfa <= 0xb and bit set in 0xc0d.
pub(crate) fn padded_depth_stencil_format(pixel_format: u64) -> bool {
    let i = pixel_format.wrapping_sub(0xfa);
    i <= 0xb && ((1u64 << i) & 0xc0d) != 0
}

/// The allocation padding rule shared by the constructor, getLevelOffset, isDepthStencilAllocTypePadded and the
/// addressing widths: a padded depth/stencil format, smaller than 32 in width or height, render-target-only usage
/// (usage & 5 == 4), one level.
pub(crate) fn depth_stencil_pad_rule(t: &TextureGen2) -> bool {
    padded_depth_stencil_format(t.pixel_format)
        && (t.width < 0x20 || t.height <= 0x1f)
        && (t.usage & 5) == 4
        && t.levels <= 1
}

impl TextureGen2 {
    pub fn format(&self) -> Result<&'static FormatInfo, Unported> {
        self.format.ok_or(Unported("a layout whose format pointer is null (interchange-tiled kind 3)"))
    }
}

fn plane_bytes(f: &FormatInfo, plane: u32) -> Result<u8, Unported> {
    if plane >= 2 {
        return Err(Unported("format plane index above 1"));
    }
    Ok(f.plane_bytes[plane as usize])
}

/// `agxsTwiddledBlockInfo::getParamsForLevel` (0x21c913f64): the level's size, and (o2..o5) its tile geometry.
fn params_for_level(info: &TwiddledBlockInfo, level: u32) -> (u64, u32, u32, u32, u32) {
    let mut w13 = info.width_log2 as u32;
    let w8 = w13.saturating_sub(level);
    let mut w14 = info.height_log2 as u32;
    let w9 = w14.saturating_sub(level);
    let w10 = info.mode as u32;
    let mut w11 = lsr32(info.width as u32, level);
    let mut w12 = lsr32(info.height as u32, level);
    let w15;
    let w16;
    let bw = info.block_width as u32;
    let bh = info.block_height as u32;
    if w10.wrapping_sub(1) <= 1 {
        w11 = at_least_one(udiv32(bw.wrapping_add(w11).wrapping_sub(1), bw));
        w12 = at_least_one(udiv32(bh.wrapping_add(w12).wrapping_sub(1), bh));
        let mut w15a = lsl32(1, w8);
        let w7 = lsl32(1, w9);
        let s1 = info.tile_height_log2 as u32;
        let s0 = info.tile_width_log2 as u32;
        if lsl32(1, s1) < w7 && lsr32(w15a, s0) == 0 {
            let bits = 32 - clz32(!lsl32(0xffff_ffff, 0u32.wrapping_sub(clz32(w15a.wrapping_sub(1)))));
            w15a = if w13 <= level { 0 } else { bits };
            w13 = s0.wrapping_add(s1).wrapping_sub(w15a);
        } else {
            // 0x21c914174
            let shifted = lsr32(w7, s1);
            let one = lsl32(1, s0);
            let neg_bits = clz32(!lsl32(0xffff_ffff, 0u32.wrapping_sub(clz32(w7.wrapping_sub(1))))).wrapping_sub(32);
            w14 = if w14 <= level { 0 } else { neg_bits };
            let sum = s0.wrapping_add(s1).wrapping_add(w14);
            let hs = one >= w15a;
            w15a = if hs { s0 } else { sum };
            w14 = if hs { s1 } else { 0u32.wrapping_sub(w14) };
            if shifted != 0 {
                w15a = s0;
                w13 = s1;
            } else {
                w13 = w14;
            }
        }
        // 0x21c9141c4
        w15 = w15a;
        w16 = w8;
        w14 = w9;
    } else {
        w11 = udiv32(bw.wrapping_add(at_least_one(w11)).wrapping_sub(1), bw);
        w12 = udiv32(bh.wrapping_add(at_least_one(w12)).wrapping_sub(1), bh);
        let a = info.align_log2 as u32;
        let unit = lsl32(1, a);
        let hi = if unit <= w11 { unit > w12 } else { true };
        let mask = lsl32(0xffff_ffff, a);
        let aw = w11.wrapping_add(!mask) & mask;
        let ah = w12.wrapping_add(!mask) & mask;
        let w17 = if hi { w11 } else { aw };
        let w16a = if hi { w12 } else { ah };
        if a != 0 {
            w11 = w17;
            w12 = w16a;
        }
        let s1 = info.tile_height_log2 as u32;
        let s0 = info.tile_width_log2 as u32;
        let tile_h = lsl32(1, s1);
        let w_tiles = lsr32(w11, s0);
        let h_tiles = lsr32(w12, s1);
        let tile_w = lsl32(1, s0);
        let c7 = clz32(w12.wrapping_sub(1));
        let mut w19 = clz32(!lsl32(0xffff_ffff, 0u32.wrapping_sub(c7))).wrapping_sub(32);
        if w12 < 2 {
            w19 = 0;
        }
        let mut w20 = s0.wrapping_add(s1);
        let w21 = w20.wrapping_add(w19);
        let hs = if h_tiles == 0 { tile_w >= w11 } else { true };
        let mut w15a = if hs { s0 } else { w21 };
        w13 = if hs { s1 } else { 0u32.wrapping_sub(w19) };
        let c16 = clz32(w11.wrapping_sub(1));
        let mut w6 = 32 - clz32(!lsl32(0xffff_ffff, 0u32.wrapping_sub(c16)));
        if w11 < 2 {
            w6 = 0;
        }
        w20 = w20.wrapping_sub(w6);
        let ne = if tile_h < w12 { w_tiles != 0 } else { true };
        w15a = if ne { w15a } else { w6 };
        w13 = if ne { w13 } else { w20 };
        w15 = w15a;
        let mut w16b = 32 - clz32(!lsl32(0xffff_ffff, 0u32.wrapping_sub(c16)));
        if w11 < 2 {
            w16b = 0;
        }
        w16 = w16b;
        w14 = 32 - clz32(!lsl32(0xffff_ffff, 0u32.wrapping_sub(c7)));
        if w12 < 2 {
            w14 = 0;
        }
    }
    // 0x21c9141cc
    let o2 = w16.saturating_sub(w15);
    let o3 = w14.saturating_sub(w13);
    let o4 = lsr32(w11.wrapping_add(!lsl32(0xffff_ffff, w15)), w15);
    let o5 = lsr32(w12.wrapping_add(!lsl32(0xffff_ffff, w13)), w13);
    let k = info.tile_mode as u32;
    let bpb = info.bytes_per_block as u64;
    if w10 == 2 && k == 0 {
        return (lsl64(lsl64(1, w9 as u64), w9 as u64).wrapping_mul(bpb), o2, o3, o4, o5);
    }
    let blocks = lsl64(lsl64(1, w9 as u64), w8 as u64);
    let r = if k == 4 || k == 3 || k == 2 {
        blocks.wrapping_mul(bpb).wrapping_add(0x7f) & !0x7f
    } else {
        blocks.wrapping_mul(bpb)
    };
    (r, o2, o3, o4, o5)
}

/// `agxsTwiddledBlockInfo::getGPUVirtualLengthWithinLayer` (0x21c914288).
fn gpu_virtual_length_within_layer(info: &TwiddledBlockInfo, first: u32, mut count: u32) -> u64 {
    let mut a = (info.width_log2 as u32).saturating_sub(first);
    let mut b = (info.height_log2 as u32).saturating_sub(first);
    if count == 0 {
        return 0;
    }
    let bpb = info.bytes_per_block as u64;
    let mut acc = 0u64;
    if info.mode == 2 && info.tile_mode == 0 {
        while count != 0 {
            acc = acc.wrapping_add(lsl64(lsl64(1, b as u64), b as u64).wrapping_mul(bpb));
            b = b.saturating_sub(1);
            count -= 1;
        }
        return acc;
    }
    let aligned = info.tile_mode == 4 || info.tile_mode == 3 || info.tile_mode == 2;
    while count != 0 {
        let v = lsl64(lsl64(1, b as u64), a as u64).wrapping_mul(bpb);
        acc = acc.wrapping_add(if aligned { v.wrapping_add(0x7f) & !0x7f } else { v });
        a = a.saturating_sub(1);
        b = b.saturating_sub(1);
        count -= 1;
    }
    acc
}

/// `getLevelOffset<View0>` (0x21c9137f4) and the non-twiddled part of `getLevelOffset<View1>` (0x21c913cc8).
fn level_offset_linear(t: &TextureGen2, level: u32, plane: u32) -> Result<u64, Unported> {
    let f = t.format()?;
    let pad = depth_stencil_pad_rule(t);
    let w = if pad { align_log2_up(t.width, f.align_log2 as u32) } else { t.width };
    let mut w1 = udiv32(t.block_width.wrapping_add(at_least_one(w)).wrapping_sub(1), t.block_width);
    let h = if pad { align_log2_up(t.height, f.align_log2 as u32) } else { t.height };
    let mut w2 = udiv32(t.block_height.wrapping_add(at_least_one(h)).wrapping_sub(1), t.block_height);
    let kind = t.kind;
    if kind != 0 && t.dim != 1 && kind != 3 {
        let check = match t.dim {
            4 => true,
            2 => t.array_bits == 0 || (t.array_bits & 3) != 0,
            _ => false,
        };
        // getLevelOffset<View1> calls isDepthStencilAllocTypePadded; getLevelOffset<View0> inlines the same test.
        if check && f.align_log2 != 0 && is_depth_stencil_alloc_type_padded(t) {
            w1 = align_log2_up(w1, f.align_log2 as u32);
            w2 = align_log2_up(w2, f.align_log2 as u32);
        }
    }
    if kind == 2 {
        return Err(Unported("IOSurface tiled layout (kind 2): sumLevelSizesTiled 0x21c9c0bcc"));
    }
    if level == 0 {
        return Ok(0);
    }
    let bpb = (t.samples as u64).wrapping_mul(plane_bytes(f, plane)? as u64);
    let e4 = t.sparse_page_size as u64;
    let mut hh = ceil_pow2(w2);
    let mut ww = ceil_pow2(w1);
    let mut acc = 0u64;
    for _ in 0..level {
        let s = bpb.wrapping_mul(hh.wrapping_mul(ww) as u64);
        if e4 == 0 {
            acc = acc.wrapping_add(s);
        } else {
            let a = if s >= e4 { e4 } else { 1 };
            acc = acc.wrapping_add(align_up(s, a));
        }
        if hh > 1 {
            hh >>= 1;
        }
        if ww > 1 {
            ww >>= 1;
        }
    }
    Ok(acc)
}

/// The tail of `getMipmapSize<View0>`/`<View1>` after the level offset (0x21c913b00, 0x21c913730).
fn mipmap_alignment(t: &TextureGen2, r: u64) -> u64 {
    let kind = t.kind;
    if kind == 0 {
        return r;
    }
    let dim = t.dim as u32;
    if dim.wrapping_sub(1) > 3 {
        return r;
    }
    let a = t.array_bits;
    match dim {
        3 => {
            if a != 0 {
                return r;
            }
        }
        4 => {
            if a != 0 && (a & 1) == 0 {
                return r;
            }
        }
        _ => {
            if (a & 1) == 0 {
                return r;
            }
        }
    }
    if t.resolved {
        return 0;
    }
    let a0 = (a & 1) as u32;
    let cond = if dim.wrapping_sub(3) < 2 || (dim == 1) as u32 & a0 != 0 { 1 } else { (dim == 2) as u32 & a0 };
    let big = (r & !0x3fff) != 0 && kind == 1 && cond != 0;
    align_up(r, if big { 0x4000 } else { 0x40 })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Zero,
    One,
}

fn plane_size(t: &TextureGen2, plane: u32, view: View) -> Result<u64, Unported> {
    let mip = |p: u32| if view == View::One { get_mipmap_size1(t, p) } else { get_mipmap_size0(t, p) };
    let stride = |p: u32| if view == View::One { get_stride1(t, p) } else { get_stride0(t, 0, p) };
    let f = t.format()?;
    let e4 = t.sparse_page_size as u64;
    let mut r;
    if e4 != 0 {
        let m0 = get_mipmap_size0(t, 0)?;
        let layers = if t.dim == 3 {
            t.depth
        } else if (t.array_bits & 1) != 0 || t.dim == 4 {
            if t.dim == 4 { t.array_length.wrapping_mul(6) } else { t.array_length }
        } else {
            1
        };
        let unit = ((e4 as u32) << 8) as u64;
        let pages = udiv64(e4.wrapping_add(align_up(m0, unit)).wrapping_sub(1), e4);
        r = ((pages << 3).wrapping_add(0x7ff) & !0x7ff).wrapping_mul(layers as u64);
        if t.sparse_params {
            return Err(Unported("calcSparseParams 0x21c985de0 (this+0xe8 set)"));
        }
    } else if (t.array_bits & 1) != 0 {
        let mm = mip(plane)?;
        let layer = if t.dim == 4 { mm.wrapping_mul(6) } else { mm };
        r = layer.wrapping_mul(t.array_length as u64);
    } else if t.dim == 4 {
        r = mip(plane)?.wrapping_mul(6);
    } else if t.kind == 3 {
        return Err(Unported("interchange-tiled plane size (InterchangeTiledMemory::getPlaneSize 0x21c9c59e0)"));
    } else if t.kind == 0 {
        if t.iosurface {
            return Err(Unported("IOSurfaceGetHeightOfPlane"));
        }
        let rows = if plane == 1 && f.chroma == 3 { (t.height >> 1) as u64 } else { t.height as u64 };
        let s = stride(plane)?;
        r = s.wrapping_mul(rows);
        if f.planes.wrapping_sub(1) == plane {
            r = r.wrapping_sub(s).wrapping_add((plane_bytes(f, plane)? as u64).wrapping_mul(t.width as u64));
        }
    } else {
        let dim = t.dim as u32;
        if dim.wrapping_sub(1) < 2 {
            r = mip(plane)?;
        } else if dim == 3 {
            r = mip(plane)?.wrapping_mul(t.depth as u64);
        } else {
            r = 0;
        }
    }
    if f.planes.wrapping_sub(1) != plane {
        r = align_up(r, t.page_alignment);
    }
    Ok(r)
}

/// `AGX::Texture::isYUVTexture` (0x21c9c0100).
pub fn is_yuv_texture(pixel_format: u64) -> bool {
    const YUV: &[u64] = &[
        0xf0, 0xf1, 0x1f4, 0x1f5, 0x1f6, 0x1f7, 0x1f8, 0x1f9, 0x1fa, 0x1fb, 0x1fc, 0x1fd, 0x1fe, 0x208, 0x209, 0x20a,
        0x20b, 0x20c, 0x20d, 0x20e, 0x20f, 0x210, 0x211, 0x212, 0x21c, 0x21d, 0x21e, 0x21f, 0x220, 0x221, 0x222, 0x223,
        0x224, 0x230, 0x232, 0x233, 0x234, 0x235, 0x236, 0x237, 0x238, 0x239, 0x23a, 0x23b, 0x23c, 0x23d, 0x23e, 0x23f,
        0x244, 0x245, 0x246, 0x247, 0x248, 0x249, 0x24c, 0x24d, 0x24e, 0x24f, 0x250, 0x251, 0x252, 0x253, 0x254, 0x255,
        0x256, 0x257, 0x258, 0x259, 0x25a, 0x25f, 0x260, 0x261, 0x262, 0x263, 0x264, 0x265, 0x267, 0x268, 0x269, 0x26a,
        0x26b, 0x26c,
    ];
    let i = pixel_format.wrapping_sub(0xf0);
    if i > 0x17c {
        return false;
    }
    YUV.contains(&pixel_format)
}

/// `AGX::Texture::isDepthStencilAllocTypePadded` (0x21c9c0b68).
pub fn is_depth_stencil_alloc_type_padded(t: &TextureGen2) -> bool {
    depth_stencil_pad_rule(t)
}

/// `getLevelOffset<View0>` (0x21c9137f4).
pub fn get_level_offset0(t: &TextureGen2, level: u32, plane: u32) -> Result<u64, Unported> {
    level_offset_linear(t, level, plane)
}

/// `getLevelOffset<View1>` (0x21c913bc4).
pub fn get_level_offset1(t: &TextureGen2, level: u32, plane: u32) -> Result<u64, Unported> {
    if !t.twiddled_planes && !t.sparse_params {
        return level_offset_linear(t, level, plane);
    }
    if plane >= 2 {
        return Err(Unported("twiddled block info of a plane above 1"));
    }
    let info = &t.block_info[plane as usize];
    let mut lvl = 0u32;
    let mut acc = 0u64;
    if level != 0 {
        loop {
            let (r, _o2, _o3, o4, o5) = params_for_level(info, lvl);
            let limit = umulh(r << 2, 0xaaaa_aaaa_aaaa_aaab) >> 1;
            let size =
                o5.wrapping_mul(o4).wrapping_mul(info.pages_per_tile as u32).wrapping_mul(info.page_align as u32);
            if size as u64 >= limit {
                break;
            }
            acc = acc.wrapping_add(size as u64);
            lvl += 1;
            if lvl == level {
                break;
            }
        }
    }
    Ok(gpu_virtual_length_within_layer(info, lvl, level.wrapping_sub(lvl)).wrapping_add(acc))
}

/// `getMipmapSize<View1>` (0x21c913aa4).
pub fn get_mipmap_size1(t: &TextureGen2, plane: u32) -> Result<u64, Unported> {
    if t.kind == 0 || t.kind == 3 {
        return Ok(0);
    }
    let r = get_level_offset1(t, t.levels, plane)?;
    if t.twiddled_planes {
        return Ok(align_up(r, t.page_alignment));
    }
    Ok(mipmap_alignment(t, r))
}

/// `getMipmapSize<View0>` (0x21c9136d4).
pub fn get_mipmap_size0(t: &TextureGen2, plane: u32) -> Result<u64, Unported> {
    if t.kind == 0 || t.kind == 3 {
        return Ok(0);
    }
    let mut r = get_level_offset0(t, t.levels, plane)?;
    if t.twiddled_planes {
        r = align_up(r, t.page_alignment);
    }
    Ok(mipmap_alignment(t, r))
}

/// `getStride<View1, Units0, Op0>` (0x21c9851c8).
pub fn get_stride1(t: &TextureGen2, plane: u32) -> Result<u64, Unported> {
    if t.kind == 3 {
        return Err(Unported("interchange-tiled stride (InterchangeTiledMemory::getPlaneInfo 0x21c9c5494)"));
    }
    if t.kind != 0 {
        return Ok(0);
    }
    if t.iosurface {
        return Err(Unported("IOSurfaceGetBytesPerRowOfPlane"));
    }
    let pf = t.pixel_format;
    if pf.wrapping_sub(0x270) < 4 || pf.wrapping_sub(0x226) <= 1 {
        return Ok(
            (t.width as u64).wrapping_mul(plane_bytes(t.format()?, plane)? as u64).wrapping_add(0xf) & 0x1ff_ffff_fff0
        );
    }
    Ok(t.bytes_per_row)
}

/// `getStride<View0, Units0, Op0>` (0x21c9134f8).
pub fn get_stride0(t: &TextureGen2, _level: u32, plane: u32) -> Result<u64, Unported> {
    if t.kind == 3 {
        return Err(Unported("interchange-tiled stride (InterchangeTiledMemory::getPlaneInfo 0x21c9c5494)"));
    }
    if t.kind != 0 {
        return Ok(0);
    }
    if t.iosurface {
        return Err(Unported("IOSurfaceGetBytesPerRowOfPlane"));
    }
    let pf = t.pixel_format;
    let f = t.format()?;
    if pf.wrapping_sub(0x270) < 4 || pf.wrapping_sub(0x226) <= 1 {
        return Ok((t.width as u64).wrapping_mul(plane_bytes(f, plane)? as u64).wrapping_add(0xf) & 0x1ff_ffff_fff0);
    }
    let mut r = t.bytes_per_row;
    if t.texture_type == 9 {
        let cap = (f.bytes_per_pixel as u64) << 14;
        if cap < r {
            r = cap;
        }
    }
    Ok(r)
}

/// `getPlaneSize<View1>` (0x21c985a50).
pub fn get_plane_size1(t: &TextureGen2, plane: u32) -> Result<u64, Unported> {
    plane_size(t, plane, View::One)
}

/// `getPlaneSize<View0>` (0x21c91598c).
pub fn get_plane_size0(t: &TextureGen2, plane: u32) -> Result<u64, Unported> {
    plane_size(t, plane, View::Zero)
}

fn in_set(v: u64, set: &[u64]) -> bool {
    set.contains(&v)
}

/// `AGX::TextureGen2<...>::TextureGen2` (0x21c9becd4).
pub fn construct(a: &TextureGen2Args, system: SystemValues) -> Result<TextureGen2, Unported> {
    if a.iosurface {
        return Err(Unported("IOSurface-backed textures"));
    }
    let fmt = choose_texture_format(a.pixel_format);
    let cls = fmt.cls.wrapping_sub(2);
    let ushl = |v: u32, shift: u32| -> u32 {
        // ushl.2s by -baseLevel (the signed low byte of each lane)
        let s = shift as u8 as i8 as i32;
        if s >= 0 {
            if s >= 32 { 0 } else { v << s }
        } else if -s >= 32 {
            0
        } else {
            v >> -s
        }
    };
    let mut t = TextureGen2 {
        system,
        resolved: false,
        texture_type: a.texture_type,
        pixel_format: a.pixel_format,
        usage: a.usage,
        storage_mode: a.storage_mode,
        compression_mode: a.compression as u8,
        compression_footprint: (a.compression >> 8) as u8,
        compression_byte2: (a.compression >> 16) as u8,
        levels: 0,
        last_slice: a.array_length.wrapping_sub(1),
        base_width: ushl(a.width, 0u32.wrapping_sub(a.base_level)).max(1),
        base_height: ushl(a.height, 0u32.wrapping_sub(a.base_level)).max(1),
        base_depth: at_least_one(lsr32(a.depth, a.base_level)),
        array_length_copy: a.array_length,
        class_word: if cls <= 0x10 && ((CLASS_WORD_MASK >> cls) & 1) != 0 { CLASS_WORDS[cls as usize] } else { 0 },
        level_count: a.level_count,
        samples: if a.samples > 1 { lsl32(1, 0u32.wrapping_sub(clz32(a.samples - 1))) } else { 1 },
        base_level: a.base_level,
        width: a.width,
        height: a.height,
        depth: a.depth,
        mip_levels: a.mip_levels,
        bytes_per_row: a.bytes_per_row,
        cpu_cache_mode: a.cpu_cache_mode,
        iosurface: a.iosurface,
        plane: a.plane,
        slice: a.slice,
        iosurface2: a.iosurface2,
        rotation: a.rotation,
        swizzle: a.swizzle,
        valid: false,
        format: Some(fmt),
        block_width: 0,
        block_height: 0,
        sparse_tile_width: 0,
        sparse_tile_height: 0,
        twiddled_planes: false,
        sparse_page_size: a.sparse_page_size,
        sparse_params: false,
        block_info: [TwiddledBlockInfo::default(); 2],
        flag128: false,
        color_space: 0,
        chroma_siting: 0,
        twiddle_compressed: false,
        pvrtc: false,
        page_alignment: system.iosurface_alignment,
        plane_end: [0, 0],
        size: 0,
        size0: 0,
        faces: 0,
        array_length: a.array_length,
        kind: 0,
        dim: 0,
        array_bits: 0,
        compression_copy: [a.compression as u8, (a.compression >> 8) as u8, (a.compression >> 16) as u8],
        compressed_backing: a.compressed_backing,
        heap_or_buffer_backed: a.heap_or_buffer_backed,
    };

    // this+0x3c: levels of a full chain from the largest dimension when more than one level is requested
    if a.mip_levels < 2 {
        t.levels = 1;
    } else {
        let m = if a.width > a.height { a.width } else { a.height };
        let m3 = if m > a.depth { m } else { a.depth };
        let m = if !t.resolved { m3 } else { m };
        t.levels = if m == 0 { 0 } else { 32 - clz32(m) };
    }

    // Layout kind, dimensionality and array bits (0x21c9bef90..0x21c9bf198)
    let pf = t.pixel_format;
    let mut lossy_candidate = false;
    if t.compression_footprint != 0 {
        let fp = t.compression_footprint;
        let x9 = pf.wrapping_sub(0x228);
        let type_check = if x9 > 0x59 {
            fp != 3
        } else if in_set(
            pf,
            &[0x240, 0x241, 0x242, 0x243, 0x25b, 0x25c, 0x25d, 0x276, 0x277, 0x278, 0x279, 0x27a, 0x27b, 0x27c],
        ) {
            true
        } else if in_set(pf, &[0x228, 0x229, 0x27d, 0x27e, 0x27f, 0x280, 0x281]) {
            fp != 1
        } else {
            fp != 3
        };
        if type_check && t.texture_type == 2 && t.levels <= 1 {
            if pf.wrapping_sub(1) <= 0x72 {
                lossy_candidate = in_set(
                    pf,
                    &[
                        0x1, 0xa, 0xb, 0xd, 0x14, 0x16, 0x17, 0x18, 0x19, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x3c, 0x3e,
                        0x3f, 0x40, 0x41, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x50, 0x51, 0x54, 0x55, 0x5a, 0x5b, 0x5e, 0x6e,
                        0x70, 0x71, 0x72, 0x73,
                    ],
                );
            } else if x9 <= 0x59 {
                lossy_candidate = in_set(
                    pf,
                    &[
                        0x228, 0x229, 0x22a, 0x22b, 0x240, 0x241, 0x242, 0x243, 0x25b, 0x25c, 0x25d, 0x276, 0x277,
                        0x278, 0x279, 0x27a, 0x27b, 0x27c, 0x27d, 0x27e, 0x27f, 0x280, 0x281,
                    ],
                );
            }
        }
    }
    let yuv = is_yuv_texture(pf);
    let mut kind = 1u8;
    let mut dim = 2u8;
    let mut array_bits = if yuv { 4u8 } else { 0 };
    if lossy_candidate {
        // 0x21c9bf130
        kind = 1;
        dim = 2;
    } else if t.texture_type <= 9 {
        match t.texture_type {
            0 => dim = 1,
            1 => {
                array_bits = 1;
                dim = 1;
            }
            2 => {}
            3 => {
                dim = 2;
                array_bits = 1;
            }
            4 => {
                array_bits = 2;
                dim = 2;
            }
            5 => dim = 4,
            6 => {
                array_bits = 1;
                dim = 4;
            }
            7 => dim = 3,
            8 => {
                dim = 2;
                array_bits = 3;
            }
            _ => {
                kind = 0;
                dim = 2;
            }
        }
    }
    if !t.compressed_backing {
        // 0x21c9bf0b4 without an IOSurface: rows given by bytesPerRow, and some formats, are linear
        if t.bytes_per_row != 0 || pf.wrapping_sub(0x270) < 4 || pf.wrapping_sub(0x226) < 2 {
            kind = 0;
        } else {
            if lossy_candidate {
                kind = 3;
            }
            if yuv {
                kind = 0;
            }
        }
    }
    t.kind = kind;
    t.dim = dim;
    t.array_bits = array_bits;

    let invalid = |mut t: TextureGen2| -> Result<TextureGen2, Unported> {
        // 0x21c9bf378
        t.valid = false;
        t.format = Some(&UNSUPPORTED_FORMAT);
        Ok(t)
    };

    let f: Option<&'static FormatInfo> = if kind == 3 {
        t.format = None;
        None
    } else if is_unsupported(fmt) {
        return invalid(t);
    } else {
        Some(fmt)
    };
    // Pixel-format checks (0x21c9bf1ec)
    if pf.wrapping_sub(0xa0) <= 0x1cb {
        if in_set(
            pf,
            &[
                0x1f4, 0x1f5, 0x1f6, 0x1f9, 0x1fa, 0x1fc, 0x1fd, 0x208, 0x209, 0x20a, 0x20d, 0x20e, 0x210, 0x211,
                0x21c, 0x21d, 0x21f, 0x220, 0x222, 0x223, 0x232, 0x234, 0x235, 0x237, 0x238, 0x23a, 0x23b, 0x23d,
                0x23e, 0x244, 0x245, 0x247, 0x248, 0x24f, 0x250, 0x252, 0x253, 0x255, 0x256, 0x258, 0x259, 0x260,
                0x261, 0x263, 0x264, 0x267, 0x268, 0x26a, 0x26b,
            ],
        ) {
            if ((t.width & 1) != 0 || (t.height & 1) != 0) && (t.usage == 0 || (t.usage & 6) != 0) {
                return invalid(t);
            }
        } else if pf.wrapping_sub(0xa0) <= 7 && (t.width.count_ones() != 1 || t.height.count_ones() != 1) {
            return invalid(t);
        }
    }
    let f = f.ok_or(Unported(
        "interchange-tiled layout (kind 3) without an IOSurface: the driver reads format+0x18 through the null format \
         pointer it has just stored",
    ))?;
    if f.planes >= 2
        && !t.iosurface
        && f.planes_without_surface == 0
        && pf.wrapping_sub(0x270) >= 4
        && pf.wrapping_sub(0x226) >= 2
        && !t.resolved
    {
        return invalid(t);
    }
    if t.sparse_page_size != 0 && kind != 1 {
        return invalid(t);
    }
    if t.compression_footprint != 0 && kind == 0 {
        return invalid(t);
    }

    t.valid = true;
    let q = pf.wrapping_sub(0xb9);
    t.twiddle_compressed = q < 0x34 && ((0x000f_fff3_fffc_ffffu64 >> (q & 63)) & 1) != 0;
    t.pvrtc = (pf & !7) == 0xa0;
    t.block_width = f.block_width;
    t.block_height = f.block_height;
    t.flag128 = pf.wrapping_sub(0x28) < 5;
    {
        let r = pf.wrapping_sub(0x24a);
        let conversion = |default: u32| if a.color_conversion != 0 { a.color_conversion } else { default };
        if (array_bits & 4) == 0 {
            if r <= 0x2b && ((1u64 << r) & 0xc00_0000_0003) != 0 {
                t.color_space = conversion(7);
            }
        } else {
            t.color_space =
                if r <= 0x2b && ((1u64 << r) & 0xc20_0000_0003) != 0 { conversion(7) } else { conversion(0) };
            t.chroma_siting = 0;
        }
    }
    // this+0x90 for linear layouts (0x21c9bf460)
    if kind == 0 {
        let mut bpr = t.bytes_per_row;
        if (a.pixel_format & !1) == 0xf0 && bpr == 0 {
            bpr = ((f.bytes_per_pixel as u64).wrapping_mul(t.width as u64).wrapping_add(0xf)) & 0x1ff_ffff_fff0;
            t.bytes_per_row = bpr;
        }
        if bpr == 0 {
            t.bytes_per_row = (f.bytes_per_pixel as u64).wrapping_mul(t.width as u64);
        }
    }
    if t.texture_type == 9 {
        t.bytes_per_row = t.bytes_per_row.wrapping_add(0xf) & !0xf;
    }
    t.faces = if dim == 4 { 6 } else { 1 };
    t.sparse_params = false;

    let mut twiddled = false;
    if t.sparse_page_size == 0 && a.allow_twiddled_planes && kind == 1 {
        // 0x21c9bf508
        let mut looped = false;
        if (dim as u32).wrapping_sub(2) <= 2 {
            let pad = depth_stencil_pad_rule(&t);
            let bw = t.block_width;
            let w = at_least_one(if pad { align_log2_up(t.width, f.align_log2 as u32) } else { t.width });
            if bw > bw.wrapping_sub(1).wrapping_add(w) {
                looped = true;
            } else {
                let nw = udiv32(bw.wrapping_sub(1).wrapping_add(w), bw);
                if (nw.wrapping_sub(1) & nw) != 0 {
                    looped = true;
                } else {
                    let bh = t.block_height;
                    let h = at_least_one(if pad { align_log2_up(t.height, f.align_log2 as u32) } else { t.height });
                    if bh > bh.wrapping_sub(1).wrapping_add(h) {
                        looped = true;
                    } else {
                        let nh = udiv32(bh.wrapping_sub(1).wrapping_add(h), bh);
                        looped = (nh.wrapping_sub(1) & nh) != 0;
                    }
                }
            }
            if looped && pf == 0x104 && (t.width < 0x40 || t.height < 0x40) {
                looped = false;
            }
        }
        if looped {
            // 0x21c9bf960
            if f.planes == 0 {
                twiddled = true;
            } else {
                let mut w20 = true;
                for p in 0..f.planes {
                    if !w20 {
                        continue;
                    }
                    if p >= 2 {
                        return Err(Unported("twiddled block info of a plane above 1"));
                    }
                    let layers = if dim == 3 {
                        t.depth
                    } else if (array_bits & 1) != 0 || dim == 4 {
                        if dim == 4 { t.array_length.wrapping_mul(6) } else { t.array_length }
                    } else {
                        1
                    };
                    w20 = false;
                    let bpb32 = t.samples.wrapping_mul(f.plane_bytes[p as usize] as u32);
                    let bpb = bpb32 & 0xff;
                    let align = t.page_alignment;
                    let twiddle_compressed = t.twiddle_compressed;
                    let (tbw, tbh, tw, th, tlevels) = (t.block_width, t.block_height, t.width, t.height, t.levels);
                    let info = &mut t.block_info[p as usize];
                    info.bytes_per_block = bpb32 as u8;
                    info.width = tw as u16;
                    info.height = th as u16;
                    info.layers = layers as u16;
                    info.levels = tlevels as u8;
                    info.page_align = align as u16;
                    info.tile_mode = 1;
                    info.align_log2 = f.align_log2;
                    info.block_width = tbw as u8;
                    info.block_height = tbh as u8;
                    info.sparse = 0;
                    info.cube = (dim == 4) as u8;
                    info.volume = (dim == 3) as u8;
                    let bw = tbw & 0xff;
                    let bh = tbh & 0xff;
                    let nw = udiv32(tw.wrapping_add(bw).wrapping_sub(1), bw);
                    let nh = udiv32(th.wrapping_add(bh).wrapping_sub(1), bh);
                    let level0 = nw.wrapping_mul(bpb).wrapping_mul(nh);
                    let page = system.vm_page_size;
                    let align16 = align & 0xffff;
                    if !(page <= level0 as u64 && page >= align16) {
                        continue;
                    }
                    info.mode = if twiddle_compressed { if dim == 4 { 2 } else { 1 } } else { 0 };
                    let p16 = (align & 0xffff) as u32;
                    let mut span =
                        if bpb.wrapping_mul(0xaaaa_aaab) < 0x5555_5556 { p16.wrapping_add(p16 << 1) } else { p16 };
                    span = udiv32(span, bpb);
                    let bits = if span < 2 { 0 } else { 32 - clz32(span - 1) };
                    let mut s0 = bits >> 1;
                    let mut s1 = bits - s0;
                    info.tile_width_log2 = s0 as u8;
                    info.tile_height_log2 = s1 as u8;
                    if twiddle_compressed {
                        let s0b = (bits >> 1) & 0xff;
                        let s1b = s1 & 0xff;
                        s0 = if s0b > 5 { s0b } else { 5 };
                        s1 = if s1b > 5 { s1b } else { 5 };
                        info.tile_width_log2 = s0 as u8;
                        info.tile_height_log2 = s1 as u8;
                    }
                    let r21 = s0 & 0xff;
                    let r20 = s1 & 0xff;
                    info.pages_per_tile = sdiv32(lsl32(bpb, r20.wrapping_add(r21)) as i32, p16 as i32) as u8;
                    info.width_log2 = if nw >= 2 { bit_length_of_predecessor(nw) } else { 0 } as u8;
                    info.height_log2 = if nh >= 2 { bit_length_of_predecessor(nh) } else { 0 } as u8;
                    if lsr32(nw, r21) == 0 {
                        w20 = false;
                        continue;
                    }
                    w20 = lsr32(nh, r20) != 0;
                }
                twiddled = w20;
            }
        }
    }
    t.twiddled_planes = twiddled;
    // The twiddle test above runs only without a sparse page size, so every path that reaches here with one computes
    // the sparse tile (0x21c9bfbc4).
    if t.sparse_page_size != 0 {
        let denom = (f.plane_bytes[0] as u64).wrapping_mul(t.samples as u64);
        let qn = udiv64(t.sparse_page_size as u64, denom);
        let bits = if qn < 2 { 0 } else { 32 - clz32((qn as u32).wrapping_sub(1)) };
        let hi = bits >> 1;
        let lo = bits - hi;
        t.sparse_tile_width = lsl32(f.block_width, hi);
        t.sparse_tile_height = lsl32(f.block_height, lo);
    }

    // Plane sizes (0x21c9bfc30)
    let planes = f.planes;
    if planes == 0 {
        t.size = 0;
        t.size0 = 0;
        return Ok(t);
    }
    let mut s = 0u64;
    for p in 0..planes {
        s = s.wrapping_add(get_plane_size1(&t, p)?);
    }
    t.size = s;
    s = 0;
    for p in 0..planes {
        s = s.wrapping_add(get_plane_size0(&t, p)?);
    }
    t.size0 = s;
    if t.resolved || planes < 2 {
        return Ok(t);
    }
    if planes > 2 {
        return Err(Unported("per-plane offsets of a format with more than two planes (they overlap this+0x158)"));
    }
    t.plane_end[0] = get_plane_size0(&t, 0)?;
    t.plane_end[1] = get_plane_size1(&t, 0)?;
    Ok(t)
}

/// The alignment the callers derive from the object: heapTextureSizeAndAlign, initImpl (set_alignment:) and
/// `-[AGXTextureLayout alignment]` share this code.
pub fn texture_alignment(t: &TextureGen2) -> Result<u64, Unported> {
    if t.resolved {
        return Ok(0);
    }
    if t.twiddled_planes {
        return Ok(t.page_alignment);
    }
    if t.sparse_page_size != 0 {
        return Ok(t.sparse_page_size as u64);
    }
    let kind = t.kind;
    let dim = t.dim as u32;
    let mip0 = get_mipmap_size0(t, 0)?;
    let a0 = (t.array_bits & 1) as u32;
    let cond = if dim.wrapping_sub(3) < 2 || (dim == 1) as u32 & a0 != 0 { 1 } else { (dim == 2) as u32 & a0 };
    if kind == 0 {
        return Ok(0x10);
    }
    Ok(if (mip0 & !0x3fff) != 0 && kind == 1 && cond != 0 { 0x4000 } else { 0x40 })
}

/// The `MTLTextureDescriptorPrivate` fields the callers read. Every field is taken from the descriptor as it is; the
/// descriptor's own defaults belong to `MTLTextureDescriptor`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DescriptorFields {
    /// +0x00
    pub texture_type: u64,
    /// +0x08
    pub pixel_format: u64,
    /// +0x10
    pub width: u64,
    /// +0x18
    pub height: u64,
    /// +0x20
    pub depth: u64,
    /// +0x28
    pub mipmap_level_count: u64,
    /// +0x30
    pub sample_count: u64,
    /// +0x38
    pub array_length: u64,
    /// +0x48
    pub rotation: u64,
    /// +0x54
    pub swizzle: u32,
    /// +0x60
    pub compression_mode: u64,
    /// +0x68
    pub usage: u64,
    /// +0x70
    pub resource_options: u64,
    /// +0x80
    pub allow_gpu_optimized_contents: bool,
    /// +0x98
    pub compression_footprint: u64,
    /// +0xa8
    pub color_space_conversion: u64,
    /// +0xc0
    pub storage_mode: u64,
}

/// The constructor's compression word, as the callers derive it from the descriptor.
pub fn compression_settings(d: &DescriptorFields) -> u32 {
    let (mode, footprint) = if d.compression_mode == 2 {
        (2u32, 0u32)
    } else {
        (
            if d.compression_mode == 1 { 9 } else { 0 },
            if matches!(d.compression_footprint, 1..=3) { d.compression_footprint as u32 } else { 0 },
        )
    };
    let mut byte0 = if d.allow_gpu_optimized_contents { mode } else { 2 };
    byte0 |= choose_texture_format(d.pixel_format).depth_stencil as u32;
    (0x100_0000 | (footprint << 8) | byte0) & 0x3fb
}

/// The constructor's eAGXColorSpaceConversion, as the callers derive it from the descriptor.
pub fn color_conversion(d: &DescriptorFields) -> u32 {
    let i = d.color_space_conversion.wrapping_sub(1);
    if i < 10 { COLOR_SPACE_CONVERSIONS[i as usize] } else { 0 }
}

/// The constructor's eAGXTextureRotation, as the callers derive it from the descriptor.
pub fn rotation(d: &DescriptorFields) -> u32 {
    if d.rotation.wrapping_sub(1) < 4 { d.rotation as u32 } else { 0 }
}

fn args_from_descriptor(d: &DescriptorFields) -> TextureGen2Args {
    TextureGen2Args {
        texture_type: d.texture_type,
        pixel_format: d.pixel_format,
        usage: d.usage,
        storage_mode: d.storage_mode,
        compression: compression_settings(d),
        color_conversion: color_conversion(d),
        rotation: rotation(d),
        swizzle: d.swizzle,
        width: d.width as u32,
        height: d.height as u32,
        depth: d.depth as u32,
        mip_levels: d.mipmap_level_count as u32,
        samples: d.sample_count as u32,
        array_length: d.array_length as u32,
        bytes_per_row: 0,
        sparse_page_size: 0,
        cpu_cache_mode: d.resource_options & 0xf,
        iosurface: false,
        plane: 0,
        slice: 0,
        iosurface2: false,
        base_level: 0,
        level_count: d.mipmap_level_count as u32,
        allow_twiddled_planes: false,
        compressed_backing: false,
        heap_or_buffer_backed: false,
    }
}

/// The layout of `-[AGXTextureLayout initWithDevice:descriptor:isHeapOrBufferBacked:]` (0x21c9c019c): twiddled planes
/// allowed unless the memory is a heap's or a buffer's. The same object is what
/// `-[AGXA10FamilyTexture initImplWithDevice:Descriptor:iosurface:nil ... allowNPOT:1 ... isHeapBacked:NO]` builds for
/// a texture without a heap, buffer or IOSurface (`-[AGXTexture initWithDevice:bytes:length:desc:deallocator:]`,
/// 0x21c9d54e0, for `newTextureWithBytesNoCopy:length:descriptor:deallocator:`) when `heap_or_buffer_backed` is false.
pub fn texture_layout(
    d: &DescriptorFields,
    heap_or_buffer_backed: bool,
    system: SystemValues,
) -> Result<TextureGen2, Unported> {
    let mut a = args_from_descriptor(d);
    a.allow_twiddled_planes = !heap_or_buffer_backed;
    a.compressed_backing = false;
    a.heap_or_buffer_backed = heap_or_buffer_backed;
    construct(&a, system)
}

/// `-[AGXA10FamilyTexture initImplWithTexture:pixelFormat:textureType:levels:slices:swizzle:resourceIndex:]`
/// (0x21c98172c; the form without a swizzle, 0x21c981074, passes the parent's +0xbc): a view's layout, built from its
/// parent's object with the parent's extent, level count and samples, the view's format and type, the parent's base
/// level plus the view's first level, the view's level count, and the view's slice count over its faces as its array
/// length. `None` where the method fails: for a memoryless parent (+0x30 == 3) or a layout that is not valid. Neither
/// form reads the slice location; the view's memory starts at the parent's CPU address of that slice
/// (`updateBindDataWithTexture:...` 0x21c981c84).
#[allow(clippy::too_many_arguments)]
pub fn init_impl_view(
    parent: &TextureGen2,
    pixel_format: u64,
    texture_type: u64,
    level_location: u64,
    level_length: u64,
    slice_length: u64,
    swizzle: u32,
    system: SystemValues,
) -> Result<Option<TextureGen2>, Unported> {
    if parent.storage_mode == 3 {
        return Ok(None);
    }
    let faces = if texture_type.wrapping_sub(5) < 2 { 6 } else { 1 };
    let a = TextureGen2Args {
        texture_type,
        pixel_format,
        usage: parent.usage,
        storage_mode: parent.storage_mode,
        compression: (u32::from(parent.compression_mode)
            | u32::from(parent.compression_footprint) << 8
            | u32::from(parent.compression_byte2) << 16)
            & 0xff_ffff,
        color_conversion: parent.color_space,
        rotation: parent.rotation,
        swizzle,
        width: parent.width,
        height: parent.height,
        depth: parent.depth,
        mip_levels: parent.mip_levels,
        samples: parent.samples,
        array_length: udiv32(slice_length as u32, faces),
        bytes_per_row: parent.bytes_per_row,
        sparse_page_size: if parent.sparse_params { 0 } else { parent.sparse_page_size },
        cpu_cache_mode: parent.cpu_cache_mode,
        iosurface: parent.iosurface,
        plane: parent.plane,
        slice: parent.slice,
        iosurface2: parent.iosurface2,
        base_level: parent.base_level.wrapping_add(level_location as u32),
        level_count: level_length as u32,
        allow_twiddled_planes: parent.twiddled_planes,
        compressed_backing: parent.compressed_backing,
        heap_or_buffer_backed: parent.heap_or_buffer_backed,
    };
    let t = construct(&a, system)?;
    Ok(t.valid.then_some(t))
}
