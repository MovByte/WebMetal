//! The A10X driver's CPU copies between texture memory and linear bytes: `AGX::Texture::writeRegion` (0x21c982fe0),
//! which `-[AGXA10FamilyTexture replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:]` and
//! `-[AGXTextureLayout copyFromLinearBytes:...]` call, and `AGX::Texture::readRegion` (0x21c9a4e84), behind
//! `-[AGXA10FamilyTexture getBytes:...]` and `-[AGXTextureLayout copyFromTextureMemory:...]`.
//!
//! Both start with `validateRegion` (0x21c983ef4), which refuses the call (nothing is copied) for an empty region, a
//! region past the level's extent (rounded up to whole blocks), a slice or level the texture does not have, a format
//! without planes or a row pitch below one row of blocks, and otherwise reads one byte per page of every block row of
//! the linear bytes (`agxaAssertBufferIsValid`, 0x21c9a4aa4). Then:
//!
//! - a linear texture (kind 0) copies `planeBytes[0] * width` bytes per row for `height` rows (the depth is not
//!   read), with the texture's row pitch (`getStride<View1>`);
//! - a twiddled texture (kind 1) of type 1D, 2D or cube places each level of a slice as one twiddled block, or, with
//!   twiddled block info (TextureGen2 +0xe0, an NPOT texture whose first level fills a page), as rows of tiles of
//!   `pageAlign * pagesPerTile` bytes, each a twiddled block. Elements are `planeBytes[0] * samples` bytes, blocks of
//!   block-compressed formats, or, for PVRTC, the level's bytes as they are;
//! - a 3D texture places depth slice z of a level at the level's offset plus z times the size of the mip chain, each
//!   depth slice like a 2D level.
//!
//! Twiddled order (`twiddleBlock`): within a block of 2^a by 2^b elements, bit i of y goes to bit i + min(i, a) of the
//! element index and bit i of x to bit i + min(i + 1, b).

use super::gen2::{
    TextureGen2, TwiddledBlockInfo, Unported, align_log2_up, at_least_one, ceil_pow2, clz32, depth_stencil_pad_rule,
    get_level_offset1, get_mipmap_size1, get_stride1, lsl32, lsr32, udiv32,
};

/// An `MTLRegion`, each field truncated to 32 bits as the driver passes them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Region {
    /// The driver's view of an `MTLRegion`: `copyFromLinearBytes:` and its siblings pass each field as `unsigned int`.
    pub fn from_mtl(origin: [u64; 3], size: [u64; 3]) -> Region {
        Region {
            x: origin[0] as u32,
            y: origin[1] as u32,
            z: origin[2] as u32,
            width: size[0] as u32,
            height: size[1] as u32,
            depth: size[2] as u32,
        }
    }
}

/// The linear side of a copy: its row and image pitches. The bytes start at the caller's pointer (plus its offset).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Linear {
    pub bytes_per_row: u64,
    pub bytes_per_image: u64,
}

/// Why `validateRegion` refused a copy: the assertion the driver logs (`AGX: Texture read/write assertion failed:`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Width,
    Height,
    Depth,
    RegionWidth,
    RegionHeight,
    RegionDepth,
    Slice,
    Level,
    Plane,
    /// `agxaAssertBufferIsValid`: the row pitch is below one row of blocks.
    RowPitch,
}

impl Refusal {
    /// The assertion text the driver logs.
    pub fn assertion(self) -> &'static str {
        match self {
            Refusal::Width => "width > 0",
            Refusal::Height => "height > 0",
            Refusal::Depth => "depth > 0",
            Refusal::RegionWidth => {
                "(xoffset + width) <= ALIGNGRAN_NPOT(getViewLevelWidth(mipmapLevel), block_width) && \"Region width OOB\""
            }
            Refusal::RegionHeight => {
                "(yoffset + height) <= ALIGNGRAN_NPOT(getViewLevelHeight(mipmapLevel), block_height) && \"Region height \
                 OOB\""
            }
            Refusal::RegionDepth => "(zoffset + depth) <= getViewLevelDepth(mipmapLevel) && \"Region depth OOB\"",
            Refusal::Slice => "slice < getNumSlices() && \"Specified slice OOB\"",
            Refusal::Level => "mipmapLevel < getBaseNumMipmapLevels() && \"Specified mipmap level OOB\"",
            Refusal::Plane => "plane < getNumPlanes() && \"Specified plane OOB\"",
            Refusal::RowPitch => "bytes_per_row >= used_bytes_per_row",
        }
    }
}

/// A copy the port cannot do as the driver does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CopyError {
    /// A path of the driver the port does not cover.
    Unported(Unported),
    /// The driver calls a null twiddle function for this layout (the process crashes).
    NullFunction(&'static str),
    /// The driver would touch texture memory outside the layout's bytes.
    TextureBounds { offset: u64, length: u64, size: u64 },
    /// The driver would touch linear bytes outside the given range.
    LinearBounds { offset: u64, length: u64, size: u64 },
    /// A read whose destination bytes overlap (rows or images closer than they are long): the driver's result depends
    /// on the order of its copy loops, which is not ported.
    OverlappingRead,
    /// A region wider or taller than the twiddled block it is addressed in (the driver addresses the 3D levels of ETC2
    /// and EAC formats in texels within a block-sized twiddle): texels share elements, and which one an element holds
    /// depends on the order of the driver's copy loops, which is not ported.
    AliasedElements,
}

impl From<Unported> for CopyError {
    fn from(u: Unported) -> CopyError {
        CopyError::Unported(u)
    }
}

impl core::fmt::Display for CopyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CopyError::Unported(u) => write!(f, "the A10X texture layout: {} is not implemented", u.0),
            CopyError::NullFunction(what) => {
                write!(f, "the A10X driver calls a null twiddle function for {what} (the process faults)")
            }
            CopyError::TextureBounds { offset, length, size } => write!(
                f,
                "the A10X driver would copy {length} bytes at {offset:#x} of texture memory of {size:#x} bytes (out of \
                 bounds)"
            ),
            CopyError::LinearBounds { offset, length, size } => write!(
                f,
                "the A10X driver would copy {length} bytes at {offset:#x} of {size:#x} linear bytes (out of bounds)"
            ),
            CopyError::OverlappingRead => write!(
                f,
                "a texture read into linear rows or images that overlap: the A10X driver's result depends on the \
                 order of its copy loops, which is not ported"
            ),
            CopyError::AliasedElements => write!(
                f,
                "a region wider or taller than the twiddled block it is addressed in: texels share elements, and the \
                 A10X driver's result depends on the order of its copy loops, which is not ported"
            ),
        }
    }
}

/// What the copy did: refused by `validateRegion`, or copied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Refused(Refusal),
    Copied,
}

/// The arguments of `writeRegion`/`readRegion` after the receiver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegionCopy {
    pub region: Region,
    pub slice: u32,
    pub level: u32,
    pub linear: Linear,
}

/// `validateRegion` (0x21c983ef4) without its final call, `agxaAssertBufferIsValid` (0x21c9a4aa4), whose pitch test
/// it includes; [`probe_extent`] gives the bytes that call reads.
pub fn validate_region(t: &TextureGen2, c: &RegionCopy) -> Result<Option<Refusal>, CopyError> {
    let r = c.region;
    if r.width == 0 {
        return Ok(Some(Refusal::Width));
    }
    if r.height == 0 {
        return Ok(Some(Refusal::Height));
    }
    if r.depth == 0 {
        return Ok(Some(Refusal::Depth));
    }
    let shift = t.base_level.wrapping_add(c.level);
    // add w14, w14, w7 (32 bits); sub x15, x14, #1; udiv; msub; mvn; add (64 bits): the extent rounded up to blocks.
    let aligned = |extent: u32, block: u32| -> u64 {
        let s = extent.wrapping_add(block) as u64;
        let m = s.wrapping_sub(1);
        let rem = m.wrapping_sub(m.checked_div(block as u64).unwrap_or(0).wrapping_mul(block as u64));
        s.wrapping_add(!rem)
    };
    if aligned(at_least_one(lsr32(t.width, shift)), t.block_width) < r.x.wrapping_add(r.width) as u64 {
        return Ok(Some(Refusal::RegionWidth));
    }
    if aligned(at_least_one(lsr32(t.height, shift)), t.block_height) < r.y.wrapping_add(r.height) as u64 {
        return Ok(Some(Refusal::RegionHeight));
    }
    if r.z.wrapping_add(r.depth) > at_least_one(lsr32(t.depth, shift)) {
        return Ok(Some(Refusal::RegionDepth));
    }
    let slices = if (t.array_bits & 1) != 0 || t.dim == 4 {
        if t.dim == 4 { t.array_length.wrapping_mul(6) } else { t.array_length }
    } else {
        1
    };
    if c.slice >= slices {
        return Ok(Some(Refusal::Slice));
    }
    if c.level >= t.mip_levels {
        return Ok(Some(Refusal::Level));
    }
    let f = t.format()?;
    if f.planes == 0 {
        return Ok(Some(Refusal::Plane));
    }
    let (_, _, min_bpr) = buffer_geometry(t, r)?;
    let bpr = if c.linear.bytes_per_row != 0 { c.linear.bytes_per_row } else { min_bpr as u64 };
    if (bpr as u32) < min_bpr {
        return Ok(Some(Refusal::RowPitch));
    }
    Ok(None)
}

/// `agxaAssertBufferIsValid`'s block columns, block rows and minimum row pitch (32-bit, from `planeBytes[0]`).
fn buffer_geometry(t: &TextureGen2, r: Region) -> Result<(u32, u32, u32), CopyError> {
    let pb = t.format()?.plane_bytes[0] as u32;
    let rows = udiv32(r.height.wrapping_add(t.block_height).wrapping_sub(1), t.block_height);
    let cols = udiv32(r.width.wrapping_add(t.block_width).wrapping_sub(1), t.block_width);
    Ok((cols, rows, cols.wrapping_mul(pb)))
}

/// The linear bytes `agxaAssertBufferIsValid` reads (one byte every `vm_page_size` bytes of each block row, rows a
/// 32-bit pitch apart and images `bytesPerImage` apart): one past the last, from the linear pointer. Zero when it reads
/// nothing.
pub fn probe_extent(t: &TextureGen2, c: &RegionCopy) -> Result<u64, CopyError> {
    let (_, rows, min_bpr) = buffer_geometry(t, c.region)?;
    let bpr = if c.linear.bytes_per_row != 0 { c.linear.bytes_per_row as u32 } else { min_bpr };
    if rows == 0 || min_bpr == 0 || c.region.depth == 0 {
        return Ok(0);
    }
    let page = t.system.vm_page_size;
    let last_column = (min_bpr as u64 - 1) / page * page;
    Ok(((c.region.depth - 1) as u64)
        .wrapping_mul(c.linear.bytes_per_image)
        .wrapping_add((rows - 1) as u64 * bpr as u64)
        .wrapping_add(last_column)
        .wrapping_add(1))
}

/// `getCPUPtr` (0x21c982724) without an IOSurface: the offset of a slice's level of a plane in texture memory.
pub fn cpu_offset(t: &TextureGen2, slice: u32, level: u32, plane: u32) -> Result<u64, CopyError> {
    if t.iosurface {
        return Err(Unported("the CPU address of an IOSurface plane (IOSurfaceGetBaseAddressOfPlane)").into());
    }
    let faces = t.faces;
    let mut offset = if plane == 0 { 0 } else { plane_end(t, plane)? };
    let group = udiv32(slice, faces);
    let mut dim = t.dim;
    if (t.array_bits & 1) != 0 && faces <= slice {
        if t.sparse_page_size != 0 {
            return Err(Unported("the slice offset of a sparse texture (getCPUPtr with this+0xe4 set)").into());
        }
        let mut layer = get_mipmap_size1(t, plane)?;
        if t.dim == 4 {
            layer = layer.wrapping_mul(6);
        }
        offset = offset.wrapping_add(layer.wrapping_mul(group as u64));
        dim = t.dim;
    }
    if dim == 4 && group.wrapping_mul(faces) != slice {
        if t.sparse_page_size != 0 {
            return Err(Unported("the face offset of a sparse texture (getCPUPtr with this+0xe4 set)").into());
        }
        let face = slice.wrapping_sub(group.wrapping_mul(faces)) as u64;
        offset = offset.wrapping_add(get_mipmap_size1(t, plane)?.wrapping_mul(face));
    }
    if level != 0 {
        offset = offset.wrapping_add(get_level_offset1(t, level, plane)?);
    }
    Ok(offset)
}

/// this+0x150 + (plane - 1) * 8: getPlaneSize<View1>(0) for plane 1, this+0x158 (the size) for plane 2.
fn plane_end(t: &TextureGen2, plane: u32) -> Result<u64, CopyError> {
    match plane {
        1 => Ok(t.plane_end[1]),
        2 => Ok(t.size),
        _ => Err(Unported("the offset of a plane above 2").into()),
    }
}

/// The level's width (`w == true`) or height in blocks for twiddled addressing:
/// `getAddressingBaseLevelWidthInBlocks` (0x21c985388) and `getAddressingBaseLevelHeightInBlocks` (0x21c9855d0)
/// without an IOSurface.
fn addressing_extent_in_blocks(t: &TextureGen2, level: u32, width: bool) -> Result<u32, CopyError> {
    if t.kind == 2 {
        return Err(Unported("addressing of the IOSurface tiled layout (kind 2)").into());
    }
    let f = t.format()?;
    let (extent, block) = if width { (t.width, t.block_width) } else { (t.height, t.block_height) };
    let e = if depth_stencil_pad_rule(t) { align_log2_up(extent, f.align_log2 as u32) } else { extent };
    if t.twiddle_compressed && t.kind == 1 {
        // 0x21c985454: the first level's blocks, rounded up to a power of two, then shifted.
        let n = udiv32(block.wrapping_sub(1).wrapping_add(at_least_one(e)), block);
        let v = lsr32(ceil_pow2(n), level);
        return Ok(if v < 2 { 1 } else { v });
    }
    let e = at_least_one(lsr32(e, level));
    Ok(udiv32(block.wrapping_add(e).wrapping_sub(1), block))
}

/// The twiddle function `getTwiddleFunc` (0x21c985810, 2D) or writeRegion's 3D switch (0x21c983cec) selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Twiddle {
    /// `agxsTwiddleAddressCommon<..., AGXSConverterPassThru<dir, element, ...>>`: elements of this many bytes.
    Common {
        element: u32,
    },
    /// `agxsTwiddleAddressCompressed<dir, bw, bh, bytes>`: blocks of bw by bh texels.
    Compressed {
        bw: u32,
        bh: u32,
        bytes: u32,
    },
    /// `agxsTwiddleAddressPVRTC2` and `agxsTwiddleAddressPVRTC4`: the level's bytes as they are.
    Pvrtc2,
    Pvrtc4,
}

/// The block size of the `agxsTwiddleAddressCompressed` variant of jump table 0x21c9dd617 (2D) for MTLPixelFormat
/// 0x82..=0xec, or `None` for the formats it sends to the element-size tables.
fn compressed_block(pixel_format: u64) -> Option<(u32, u32, u32)> {
    match pixel_format {
        170 | 172 | 180..=183 => Some((4, 4, 8)),
        174 | 176 | 178 | 179 => Some((4, 4, 16)),
        _ => compressed_block_3d(pixel_format),
    }
}

/// The block size of the `agxsTwiddle3DAddressCompressed` variant of jump table 0x21c9dd59c (3D), which sends the
/// ETC2 and EAC formats (170..=184), like PVRTC, to the bytes-per-pixel table.
fn compressed_block_3d(pixel_format: u64) -> Option<(u32, u32, u32)> {
    Some(match pixel_format {
        130 | 131 | 140 | 141 => (4, 4, 8),
        132..=135 | 142 | 143 | 150..=153 | 186 | 204 | 222 => (4, 4, 16),
        185 | 203 | 221 => (4, 2, 16),
        187 | 205 | 223 => (5, 4, 16),
        188 | 206 | 224 => (5, 5, 16),
        189 | 207 | 225 => (6, 5, 16),
        190 | 208 | 226 => (6, 6, 16),
        191 | 209 | 227 => (8, 4, 16),
        192 | 210 | 228 => (8, 5, 16),
        193 | 211 | 229 => (8, 6, 16),
        194 | 212 | 230 => (8, 8, 16),
        195 | 213 | 231 => (10, 5, 16),
        196 | 214 | 232 => (10, 6, 16),
        197 | 215 | 233 => (10, 8, 16),
        198 | 216 | 234 => (10, 10, 16),
        199 | 217 | 235 => (12, 10, 16),
        200 | 218 | 236 => (12, 12, 16),
        _ => return None,
    })
}

/// `getTwiddleFunc` / `getDetwiddleFunc` (0x21c985810, 0x21c9a58f0): the 2D function.
fn twiddle_2d(t: &TextureGen2) -> Result<Twiddle, CopyError> {
    let pf = t.pixel_format;
    if (160..=167).contains(&pf) {
        return Ok(if pf == 162 || pf == 163 || pf == 166 || pf == 167 { Twiddle::Pvrtc4 } else { Twiddle::Pvrtc2 });
    }
    if let Some((bw, bh, bytes)) = compressed_block(pf) {
        return Ok(Twiddle::Compressed { bw, bh, bytes });
    }
    // Jump table 0x21c9dd682 on planeBytes[0] - 1, then the tables at 0x22b7be488.. on samples - 1.
    let pb = t.format()?.plane_bytes[0] as u32;
    let samples = t.samples;
    if !matches!(pb, 1 | 2 | 4 | 8 | 16) || !matches!(samples, 1 | 2 | 4 | 8) {
        return Err(CopyError::NullFunction("a twiddled level of this element size and sample count"));
    }
    if pb == 2 && t.resolved {
        return Err(Unported("the Z-order twiddle (agxsTwiddleAddressZ) of a texture with this+0x10 set").into());
    }
    Ok(Twiddle::Common { element: pb * samples })
}

/// writeRegion's 3D switch (0x21c983cec): jump table 0x21c9dd59c, then 0x21c9dd607 on bytesPerPixel - 1.
fn twiddle_3d(t: &TextureGen2) -> Result<Twiddle, CopyError> {
    let pf = t.pixel_format;
    if let Some((bw, bh, bytes)) = compressed_block_3d(pf) {
        return Ok(Twiddle::Compressed { bw, bh, bytes });
    }
    let bpp = t.format()?.bytes_per_pixel as u32;
    if !matches!(bpp, 1 | 2 | 4 | 8 | 16) {
        return Err(CopyError::NullFunction("a twiddled 3D level of this element size"));
    }
    Ok(Twiddle::Common { element: bpp })
}

/// One element copy: (texture offset, linear offset, length).
pub type Visit<'a> = dyn FnMut(u64, u64, u64) -> Result<(), CopyError> + 'a;

/// Bit lengths of a power-of-two extent: `32 - clz(v - 1)`, 0 for 0 and 1.
fn log2_ceil(v: u32) -> u32 {
    if v > 1 { 32 - clz32(v - 1) } else { 0 }
}

/// The element index of (x, y) in a twiddled block of 2^lw by 2^lh elements (`twiddleBlock`'s masks).
fn morton(x: u32, y: u32, lw: u32, lh: u32) -> u64 {
    let mut out = 0u64;
    for i in 0..lw {
        out |= (((x >> i) & 1) as u64) << (i + (i + 1).min(lh));
    }
    for i in 0..lh {
        out |= (((y >> i) & 1) as u64) << (i + i.min(lw));
    }
    out
}

/// `agxsBlockTwiddleAddress` (e.g. 0x21c98a8dc) and the `twiddleBlock` copies it splits the region into: the w by h
/// elements at (x, y) of a twiddled block of `bw` by `bh` elements, rows `bpr` apart in the linear bytes.
#[allow(clippy::too_many_arguments)]
fn block_twiddle(
    tex: u64,
    lin: u64,
    bw: u32,
    bh: u32,
    w: u32,
    h: u32,
    bpr: u64,
    x: u32,
    y: u32,
    element: u32,
    visit: &mut Visit,
) -> Result<(), CopyError> {
    let lw = log2_ceil(bw);
    let lh = log2_ceil(bh);
    if u64::from(x) + u64::from(w) > 1u64 << lw || u64::from(y) + u64::from(h) > 1u64 << lh {
        return Err(CopyError::AliasedElements);
    }
    let e = element as u64;
    for j in 0..h {
        for i in 0..w {
            let index = morton(x.wrapping_add(i), y.wrapping_add(j), lw, lh);
            visit(tex.wrapping_add(index.wrapping_mul(e)), lin.wrapping_add(j as u64 * bpr + i as u64 * e), e)?;
        }
    }
    Ok(())
}

/// `agxsTwiddleAddressCommon` (e.g. 0x21c98a030) without sparse information.
#[allow(clippy::too_many_arguments)]
fn twiddle_common(
    tex: u64,
    lin: u64,
    level_w: u32,
    level_h: u32,
    pot_w: u32,
    pot_h: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    bpr: u64,
    info: Option<&TwiddledBlockInfo>,
    element: u32,
    visit: &mut Visit,
) -> Result<(), CopyError> {
    let Some(info) = info else {
        return block_twiddle(tex, lin, pot_w, pot_h, w, h, bpr, x, y, element, visit);
    };
    if info.sparse != 0 {
        return Err(Unported("the sparse twiddle (agxCalcSparseInputs), block info +0x14 set").into());
    }
    // The tile, reshaped to the level when the level is narrower or shorter than it.
    let mut th = info.tile_height_log2 as u32;
    let mut tw = info.tile_width_log2 as u32;
    let lg_h = log2_ceil(pot_h);
    let lg_w = log2_ceil(pot_w);
    let (th0, tw0) = (th, tw);
    if lsr32(pot_h, th0) == 0 && lsl32(1, tw0) < pot_w {
        th = lg_h;
        tw = tw0.wrapping_add(th0).wrapping_sub(lg_h);
    }
    if lsl32(1, th0) < pot_h && lsr32(pot_w, tw0) == 0 {
        th = tw0.wrapping_add(th0).wrapping_sub(lg_w);
        tw = lg_w;
    }
    let tile_h = lsl32(1, th);
    let tile_w = lsl32(1, tw);
    let a = info.align_log2 as u32;
    let unit = lsl32(1, a);
    let level_w = if a != 0 && unit <= level_w && unit <= level_h { align_log2_up(level_w, a) } else { level_w };
    let tiles_per_row = lsr32(level_w.wrapping_add(tile_w - 1), tw);
    let block_w = pot_w.min(tile_w);
    let block_h = pot_h.min(tile_h);
    let row0 = lsr32(y, th);
    let row1 = lsr32(y.wrapping_add(h).wrapping_add(tile_h - 1), th);
    let col0 = lsr32(x, tw);
    let col1 = lsr32(x.wrapping_add(w).wrapping_add(tile_w - 1), tw);
    let e = element as u64;
    for r in row0..row1 {
        let top = (r << th).max(y);
        let bottom = ((r + 1) << th).min(y + h);
        for c in col0..col1 {
            let left = (c << tw).max(x);
            let right = ((c + 1) << tw).min(x + w);
            // (tile index * pageAlign * pagesPerTile), a 32-bit product
            let tile = r
                .wrapping_mul(tiles_per_row)
                .wrapping_add(c)
                .wrapping_mul(info.page_align as u32)
                .wrapping_mul(info.pages_per_tile as u32) as u64;
            let src = lin + (top - y) as u64 * bpr + (left - x) as u64 * e;
            block_twiddle(
                tex + tile,
                src,
                block_w,
                block_h,
                right - left,
                bottom - top,
                bpr,
                left - (c << tw),
                top - (r << th),
                element,
                visit,
            )?;
        }
    }
    Ok(())
}

/// A 2D twiddle function applied to one level (or one depth slice of a 3D level).
#[allow(clippy::too_many_arguments)]
fn twiddle_level(
    f: Twiddle,
    tex: u64,
    lin: u64,
    level_w: u32,
    level_h: u32,
    pot_w: u32,
    pot_h: u32,
    r: Region,
    bpr: u64,
    info: Option<&TwiddledBlockInfo>,
    visit: &mut Visit,
) -> Result<(), CopyError> {
    match f {
        Twiddle::Common { element } => twiddle_common(
            tex, lin, level_w, level_h, pot_w, pot_h, r.x, r.y, r.width, r.height, bpr, info, element, visit,
        ),
        Twiddle::Compressed { bw, bh, bytes } => {
            // agxsTwiddleAddressCompressed: the level, the origin and the size in blocks (32-bit sums).
            let blocks = |v: u32, b: u32| v.wrapping_add(b - 1) / b;
            twiddle_common(
                tex,
                lin,
                blocks(level_w, bw),
                blocks(level_h, bh),
                pot_w,
                pot_h,
                r.x / bw,
                r.y / bh,
                blocks(r.width, bw),
                blocks(r.height, bh),
                bpr,
                info,
                bytes,
                visit,
            )
        }
        // agxsTwiddleAddressPVRTC4 (0x21c98ebb8): ((w + 3) >> 2) * ((h + 3) >> 2) * 8 bytes as they are.
        Twiddle::Pvrtc4 => {
            let n = (r.width.wrapping_add(3) >> 2) as u64 * (r.height.wrapping_add(3) >> 2) as u64 * 8;
            visit(tex, lin, n)
        }
        // agxsTwiddleAddressPVRTC2 (0x21c995308): ((w + 7) & ~7) * ((h + 3) >> 2) bytes as they are.
        Twiddle::Pvrtc2 => {
            let n = (r.width.wrapping_add(7) & !7) as u64 * (r.height.wrapping_add(3) >> 2) as u64;
            visit(tex, lin, n)
        }
    }
}

/// The copies of `writeRegion` and `readRegion` after validation (which the caller has made), in the driver's order:
/// `visit(texture offset, linear offset, length)` for each element (or run).
pub fn visit(t: &TextureGen2, c: &RegionCopy, visit: &mut Visit) -> Result<(), CopyError> {
    plan(t, c, visit)
}

fn plan(t: &TextureGen2, c: &RegionCopy, visit: &mut Visit) -> Result<(), CopyError> {
    if t.sparse_params {
        return Err(Unported("sparse region copies (agxCalcSparseInputs), this+0xe8 set").into());
    }
    if t.iosurface || t.iosurface2 {
        return Err(Unported("region copies of an IOSurface-backed texture (IOSurfaceLock)").into());
    }
    let r = c.region;
    let level = t.base_level.wrapping_add(c.level);
    let f = t.format()?;
    let bpr = c.linear.bytes_per_row;
    if t.kind == 0 {
        // Rows of planeBytes[0] * width bytes, the texture's rows getStride<View1>(0) apart.
        let stride = get_stride1(t, 0)?;
        let base = cpu_offset(t, c.slice, level, 0)?;
        let pb = f.plane_bytes[0] as u64;
        let row = base.wrapping_add(stride.wrapping_mul(r.y as u64)).wrapping_add(pb.wrapping_mul(r.x as u64));
        for j in 0..r.height as u64 {
            visit(row.wrapping_add(stride.wrapping_mul(j)), bpr.wrapping_mul(j), pb.wrapping_mul(r.width as u64))?;
        }
        return Ok(());
    }
    if t.kind == 2 {
        return Err(
            Unported("region copies of the IOSurface tiled layout (kind 2, TiledMemory::convertTiles2D)").into()
        );
    }
    if t.kind == 3 {
        return Err(Unported("region copies of the interchange-tiled layout (kind 3)").into());
    }
    let info = if t.twiddled_planes { Some(&t.block_info[0]) } else { None };
    match t.dim {
        1 | 2 | 4 => {
            let level_w = at_least_one(lsr32(t.width, level));
            let level_h = at_least_one(lsr32(t.height, level));
            let pot_w = ceil_pow2(addressing_extent_in_blocks(t, level, true)?);
            let pot_h = ceil_pow2(addressing_extent_in_blocks(t, level, false)?);
            let func = twiddle_2d(t)?;
            let tex = cpu_offset(t, c.slice, level, 0)?;
            twiddle_level(func, tex, 0, level_w, level_h, pot_w, pot_h, r, bpr, info, visit)
        }
        3 => {
            let level_w = at_least_one(lsr32(t.width, level));
            let level_h = at_least_one(lsr32(t.height, level));
            let pot_w = ceil_pow2(addressing_extent_in_blocks(t, level, true)?);
            let pot_h = ceil_pow2(addressing_extent_in_blocks(t, level, false)?);
            let func = twiddle_3d(t)?;
            let tex = cpu_offset(t, 0, level, 0)?;
            let stride = get_mipmap_size1(t, 0)?;
            // agxsTwiddle3DAddressCommon / agxsTwiddle3DAddressCompressed: depth slice z at z * stride, the linear
            // images bytesPerImage apart.
            for k in 0..r.depth {
                let z = r.z.wrapping_add(k) as u64;
                let lin = c.linear.bytes_per_image.wrapping_mul(k as u64);
                twiddle_level(
                    func,
                    tex.wrapping_add(z.wrapping_mul(stride)),
                    lin,
                    level_w,
                    level_h,
                    pot_w,
                    pot_h,
                    r,
                    bpr,
                    info,
                    visit,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn check(offset: u64, length: u64, size: usize, linear: bool) -> Result<core::ops::Range<usize>, CopyError> {
    match offset.checked_add(length) {
        Some(end) if end <= size as u64 => Ok(offset as usize..end as usize),
        _ => Err(if linear {
            CopyError::LinearBounds { offset, length, size: size as u64 }
        } else {
            CopyError::TextureBounds { offset, length, size: size as u64 }
        }),
    }
}

/// `AGX::Texture::writeRegion` (0x21c982fe0): copies the region from `linear` (the bytes at the caller's pointer plus
/// its offset) into `memory`, which holds the texture memory from byte `memory_offset` on.
pub fn write_region(
    t: &TextureGen2,
    c: &RegionCopy,
    memory: &mut [u8],
    memory_offset: u64,
    linear: &[u8],
) -> Result<Outcome, CopyError> {
    if let Some(refusal) = validate_region(t, c)? {
        return Ok(Outcome::Refused(refusal));
    }
    check(0, probe_extent(t, c)?, linear.len(), true)?;
    let size = memory.len();
    plan(t, c, &mut |tex, lin, len| {
        let to = check(tex.wrapping_sub(memory_offset), len, size, false)?;
        let from = check(lin, len, linear.len(), true)?;
        memory[to].copy_from_slice(&linear[from]);
        Ok(())
    })?;
    Ok(Outcome::Copied)
}

/// `AGX::Texture::readRegion` (0x21c9a4e84): copies the region from `memory` (the texture memory from byte
/// `memory_offset` on) into `linear`. Reads whose destination rows or images overlap are refused with
/// [`CopyError::OverlappingRead`].
pub fn read_region(
    t: &TextureGen2,
    c: &RegionCopy,
    memory: &[u8],
    memory_offset: u64,
    linear: &mut [u8],
) -> Result<Outcome, CopyError> {
    if let Some(refusal) = validate_region(t, c)? {
        return Ok(Outcome::Refused(refusal));
    }
    check(0, probe_extent(t, c)?, linear.len(), true)?;
    // Every element's destination is written once only when rows and images do not overlap.
    let mut spans: Vec<(u64, u64)> = Vec::new();
    plan(t, c, &mut |_, lin, len| {
        spans.push((lin, len));
        Ok(())
    })?;
    spans.sort_unstable();
    if spans.windows(2).any(|p| p[0].0 + p[0].1 > p[1].0) {
        return Err(CopyError::OverlappingRead);
    }
    let size = linear.len();
    plan(t, c, &mut |tex, lin, len| {
        let from = check(tex.wrapping_sub(memory_offset), len, memory.len(), false)?;
        let to = check(lin, len, size, true)?;
        linear[to].copy_from_slice(&memory[from]);
        Ok(())
    })?;
    Ok(Outcome::Copied)
}

/// The bytes a copy touches, for a caller that moves them between guest memory and the slices [`write_region`] and
/// [`read_region`] take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    /// One past the last linear byte `agxaAssertBufferIsValid` reads or an element copy moves.
    pub linear: u64,
    /// The texture bytes the copy moves: [first, end), or `None` when it moves none.
    pub texture: Option<(u64, u64)>,
}

/// The bytes a copy touches; `None` when `validateRegion` refuses it.
pub fn extent(t: &TextureGen2, c: &RegionCopy) -> Result<Option<Extent>, CopyError> {
    if validate_region(t, c)?.is_some() {
        return Ok(None);
    }
    let mut linear = probe_extent(t, c)?;
    let mut texture: Option<(u64, u64)> = None;
    plan(t, c, &mut |tex, lin, len| {
        linear = linear.max(lin.saturating_add(len));
        let end = tex.saturating_add(len);
        texture = Some(match texture {
            Some((a, b)) => (a.min(tex), b.max(end)),
            None => (tex, end),
        });
        Ok(())
    })?;
    Ok(Some(Extent { linear, texture }))
}

/// The number of slices `validateRegion` accepts.
pub fn slice_count(t: &TextureGen2) -> u32 {
    if (t.array_bits & 1) != 0 || t.dim == 4 {
        if t.dim == 4 { t.array_length.wrapping_mul(6) } else { t.array_length }
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::super::gen2::{DescriptorFields, SystemValues, texture_layout};
    use super::*;

    fn descriptor(texture_type: u64, pixel_format: u64, size: (u64, u64, u64), levels: u64) -> DescriptorFields {
        DescriptorFields {
            texture_type,
            pixel_format,
            width: size.0,
            height: size.1,
            depth: size.2,
            mipmap_level_count: levels,
            sample_count: 1,
            array_length: 1,
            usage: 7,
            allow_gpu_optimized_contents: true,
            ..Default::default()
        }
    }

    fn layout(d: &DescriptorFields, hb: bool) -> TextureGen2 {
        let t = texture_layout(d, hb, SystemValues::default()).unwrap();
        assert!(t.valid);
        t
    }

    fn pattern(n: usize, seed: u8) -> Vec<u8> {
        (0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed ^ (i >> 8) as u8)).collect()
    }

    fn whole(t: &TextureGen2, slice: u32, level: u32, element: u64) -> RegionCopy {
        let w = (t.width >> level).max(1);
        let h = (t.height >> level).max(1);
        let d = if t.dim == 3 { (t.depth >> level).max(1) } else { 1 };
        let cols = w.div_ceil(t.block_width) as u64;
        let rows = h.div_ceil(t.block_height) as u64;
        RegionCopy {
            region: Region { x: 0, y: 0, z: 0, width: w, height: h, depth: d },
            slice,
            level,
            linear: Linear { bytes_per_row: cols * element, bytes_per_image: cols * element * rows },
        }
    }

    /// Every slice and level written from distinct bytes reads back unchanged, and the levels and slices occupy
    /// disjoint bytes of the layout.
    fn round_trip(t: &TextureGen2, element: u64) {
        let size = t.size as usize;
        let mut memory = vec![0u8; size];
        let mut touched = vec![false; size];
        let mut sources = Vec::new();
        for slice in 0..slice_count(t) {
            for level in 0..t.mip_levels {
                let c = whole(t, slice, level, element);
                let n = (c.linear.bytes_per_image * c.region.depth as u64) as usize;
                let src = pattern(n, (slice * 7 + level) as u8);
                assert_eq!(write_region(t, &c, &mut memory, 0, &src), Ok(Outcome::Copied));
                visit(t, &c, &mut |tex, _, len| {
                    for b in &mut touched[tex as usize..(tex + len) as usize] {
                        assert!(!*b, "byte {tex:#x} of two elements");
                        *b = true;
                    }
                    Ok(())
                })
                .unwrap();
                sources.push((c, src));
            }
        }
        for (c, src) in sources {
            let mut back = vec![0u8; src.len()];
            assert_eq!(read_region(t, &c, &memory, 0, &mut back), Ok(Outcome::Copied));
            assert_eq!(back, src, "slice {} level {}", c.slice, c.level);
        }
    }

    /// twiddleBlock's order: y in the low bit of each pair, the longer dimension's extra bits above.
    #[test]
    fn twiddled_order() {
        let square: Vec<u64> = (0..4).flat_map(|y| (0..4).map(move |x| morton(x, y, 2, 2))).collect();
        assert_eq!(square, [0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15]);
        // 4 wide, 2 high: index = y0 | x0 << 1 | x1 << 2.
        let wide: Vec<u64> = (0..2).flat_map(|y| (0..4).map(move |x| morton(x, y, 2, 1))).collect();
        assert_eq!(wide, [0, 2, 4, 6, 1, 3, 5, 7]);
        // 2 wide, 4 high: index = y0 | x0 << 1 | y1 << 2.
        let tall: Vec<u64> = (0..4).flat_map(|y| (0..2).map(move |x| morton(x, y, 1, 2))).collect();
        assert_eq!(tall, [0, 2, 1, 3, 4, 6, 5, 7]);
    }

    /// A power-of-two RGBA8 texture: each level one twiddled block, levels one after the other.
    #[test]
    fn pot_levels_are_twiddled_blocks() {
        let t = layout(&descriptor(2, 70, (8, 8, 1), 4), false);
        assert_eq!((t.kind, t.dim, t.twiddled_planes), (1, 2, false));
        let c = whole(&t, 0, 1, 4);
        let mut memory = vec![0u8; t.size as usize];
        let src: Vec<u8> = (0..64).collect();
        assert_eq!(write_region(&t, &c, &mut memory, 0, &src), Ok(Outcome::Copied));
        // Level 1 starts after the 8x8 level 0 (256 bytes); texel (1, 0) is element 2, (0, 1) element 1.
        assert_eq!(&memory[256..260], &src[0..4]);
        assert_eq!(&memory[256 + 8..256 + 12], &src[4..8]);
        assert_eq!(&memory[256 + 4..256 + 8], &src[16..20]);
        round_trip(&t, 4);
    }

    #[test]
    fn round_trips() {
        // NPOT, large enough for twiddled block information, and the same heap-backed (no block information).
        for hb in [false, true] {
            let t = layout(&descriptor(2, 70, (1000, 700, 1), 10), hb);
            assert_eq!(t.twiddled_planes, !hb);
            round_trip(&t, 4);
        }
        round_trip(&layout(&descriptor(2, 125, (100, 37, 1), 7), false), 16);
        round_trip(&layout(&descriptor(2, 10, (3, 5, 1), 3), false), 1);
        // 2D array, cube, cube array, 1D array and 3D.
        let mut array = descriptor(3, 115, (64, 48, 1), 3);
        array.array_length = 3;
        round_trip(&layout(&array, false), 8);
        round_trip(&layout(&descriptor(5, 70, (32, 32, 1), 6), false), 4);
        let mut cubes = descriptor(6, 70, (16, 16, 1), 1);
        cubes.array_length = 2;
        round_trip(&layout(&cubes, true), 4);
        let mut lines = descriptor(1, 30, (100, 1, 1), 1);
        lines.array_length = 4;
        round_trip(&layout(&lines, false), 2);
        round_trip(&layout(&descriptor(7, 70, (33, 17, 9), 6), false), 4);
        // Block-compressed: ETC2 RGB8 (4x4 blocks of 8 bytes) and ASTC 5x5 (16 bytes).
        round_trip(&layout(&descriptor(2, 180, (100, 37, 1), 7), false), 8);
        round_trip(&layout(&descriptor(2, 206, (64, 64, 1), 7), false), 16);
    }

    /// validateRegion's refusals copy nothing.
    #[test]
    fn refusals() {
        let t = layout(&descriptor(2, 70, (16, 16, 1), 2), false);
        let ok = whole(&t, 0, 0, 4);
        let cases = [
            (RegionCopy { region: Region { width: 0, ..ok.region }, ..ok }, Refusal::Width),
            (RegionCopy { region: Region { width: 17, ..ok.region }, ..ok }, Refusal::RegionWidth),
            (RegionCopy { region: Region { y: 1, ..ok.region }, ..ok }, Refusal::RegionHeight),
            (RegionCopy { region: Region { depth: 2, ..ok.region }, ..ok }, Refusal::RegionDepth),
            (RegionCopy { slice: 1, ..ok }, Refusal::Slice),
            (RegionCopy { level: 2, region: Region { width: 1, height: 1, ..ok.region }, ..ok }, Refusal::Level),
            (RegionCopy { linear: Linear { bytes_per_row: 63, bytes_per_image: 0 }, ..ok }, Refusal::RowPitch),
        ];
        let src = vec![1u8; 2048];
        for (c, refusal) in cases {
            let mut memory = vec![0u8; t.size as usize];
            assert_eq!(write_region(&t, &c, &mut memory, 0, &src), Ok(Outcome::Refused(refusal)));
            assert!(memory.iter().all(|&b| b == 0));
        }
        // A zero row pitch passes as the minimum, which agxaAssertBufferIsValid reads every row of; the copy reads
        // the first row for every row.
        let c = RegionCopy { linear: Linear { bytes_per_row: 0, bytes_per_image: 0 }, ..ok };
        let mut memory = vec![0u8; t.size as usize];
        assert_eq!(
            write_region(&t, &c, &mut memory, 0, &pattern(64, 3)),
            Err(CopyError::LinearBounds { offset: 0, length: 15 * 64 + 1, size: 64 })
        );
        let src = pattern(1024, 3);
        assert_eq!(write_region(&t, &c, &mut memory, 0, &src), Ok(Outcome::Copied));
        let mut back = vec![0u8; 1024];
        assert_eq!(read_region(&t, &ok, &memory, 0, &mut back), Ok(Outcome::Copied));
        assert!(back.chunks(64).all(|row| row == &src[..64]));
        // Reading into overlapping rows is not ported.
        let mut out = vec![0u8; 1024];
        assert_eq!(read_region(&t, &c, &memory, 0, &mut out), Err(CopyError::OverlappingRead));
    }

    /// A linear texture (a YCbCr 4:2:2 format, kind 0) copies rows with its row pitch; a PVRTC level is copied as
    /// it is.
    #[test]
    fn linear_and_pvrtc() {
        let t = layout(&descriptor(2, 240, (10, 4, 1), 1), false);
        assert_eq!(t.kind, 0);
        let pitch = get_stride1(&t, 0).unwrap();
        let c = RegionCopy {
            region: Region { x: 2, y: 1, z: 0, width: 4, height: 2, depth: 1 },
            slice: 0,
            level: 0,
            linear: Linear { bytes_per_row: 8, bytes_per_image: 0 },
        };
        let mut memory = vec![0u8; t.size as usize];
        let src = pattern(16, 9);
        assert_eq!(write_region(&t, &c, &mut memory, 0, &src), Ok(Outcome::Copied));
        let pb = t.format.unwrap().plane_bytes[0] as usize;
        let at = pitch as usize + 2 * pb;
        assert_eq!(&memory[at..at + 4 * pb], &src[..4 * pb]);

        let t = layout(&descriptor(2, 166, (16, 16, 1), 1), false);
        let c = RegionCopy {
            region: Region { x: 0, y: 0, z: 0, width: 16, height: 16, depth: 1 },
            slice: 0,
            level: 0,
            linear: Linear::default(),
        };
        let mut memory = vec![0u8; t.size as usize];
        let src = pattern(128, 4);
        assert_eq!(write_region(&t, &c, &mut memory, 0, &src), Ok(Outcome::Copied));
        assert_eq!(&memory[..128], &src[..]);
    }
}
