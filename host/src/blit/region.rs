//! Metal's rules for the regions and buffer layouts of blits, from the Apple documentation of
//! `copy(from:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:to:destinationSlice:destinationLevel:destinationOrigin:options:)`,
//! its texture-to-buffer counterpart and
//! `copy(from:sourceSlice:sourceLevel:sourceOrigin:sourceSize:to:destinationSlice:destinationLevel:destinationOrigin:)`:
//!
//! - the offset "needs to be a multiple of the destination texture's pixel size, in bytes";
//! - bytesPerRow is "A multiple of the source texture's pixel size, in bytes" and "Less than or equal to the product of
//!   the destination texture's pixel size, in bytes, and the largest pixel width the destination texture's type
//!   allows"; for compressed formats it is "the number of bytes between the starts of two row blocks";
//! - bytesPerImage "needs to be a multiple of the source texture's pixel size"; "Set this value to `0` for 2D
//!   textures, which means `sourceSize.depth` is equal to `1`" (Metal accepts other values there and does not read
//!   them: `format.Depth32Float.upload-read` passes one with and without the debug layer);
//! - "If `destinationTexture` uses a compressed pixel format, set `sourceSize` to a multiple of the pixel format's
//!   block size. If the block extends outside the bounds of the texture, clamp `sourceSize` to the edge of the
//!   texture";
//! - "For textures that use a PVRTC pixel format, you can use this method to copy the entire texture, but not a
//!   subregion of the texture", and copies between a PVRTC texture and a buffer take `rowLinearPVRTC`.
//!
//! Anything outside these rules is behavior Metal does not define (validation class 2 of plan section 6.10).

use webmetal_wire::{Origin, Size};

use crate::format::pvrtc;
use crate::mem::MemError;
use crate::mem::layout::{Blocks, Geometry, RegionLayout, TextureType};

/// `MTLBlitOption` bits (`MTLBlitCommandEncoder.h`).
pub mod option {
    pub const DEPTH_FROM_DEPTH_STENCIL: u64 = 1 << 0;
    pub const STENCIL_FROM_DEPTH_STENCIL: u64 = 1 << 1;
    pub const ROW_LINEAR_PVRTC: u64 = 1 << 2;
}

/// The A10X's largest texture width of each type (`maxTextureWidth1D`, `maxTextureWidth2D`, `maxTextureWidth3D` of the
/// device profile; cube and array types take the 2D limit, `initLimits` has no other).
pub fn max_width(kind: TextureType) -> u64 {
    match kind {
        TextureType::D1 | TextureType::D1Array => 16384,
        TextureType::D3 => 2048,
        _ => 16384,
    }
}

/// How the bytes of a region are grouped: blocks of `width` × `height` texels of `bytes` bytes, or the 64-bit words of a
/// PVRTC level in row-linear order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unit {
    pub width: u32,
    pub height: u32,
    pub bytes: u32,
    pub pvrtc: Option<pvrtc::Bpp>,
}

impl Unit {
    pub fn of(b: Blocks) -> Unit {
        Unit { width: b.width, height: b.height, bytes: b.bytes, pvrtc: b.pvrtc }
    }

    /// One aspect of a combined depth-stencil format: texels of `bytes` bytes.
    pub fn aspect(bytes: u32) -> Unit {
        Unit { width: 1, height: 1, bytes, pvrtc: None }
    }

    pub fn compressed(self) -> bool {
        self.width != 1 || self.height != 1
    }
}

/// A validated region of one subresource: texel origin and size, and its blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub level: u32,
    pub slice: u32,
    pub origin: (u32, u32, u32),
    pub size: (u32, u32, u32),
    /// Blocks (PVRTC: words) of the region in x and y.
    pub blocks: (u32, u32),
}

/// Validate a region of `g` for a blit: `slice` and `level` inside the texture, the region inside the level, compressed
/// regions on block boundaries (or reaching the level's edge), PVRTC regions the whole level.
pub fn region(
    call: &'static str,
    g: &Geometry,
    unit: Unit,
    (slice, level): (u64, u64),
    origin: &Origin,
    size: &Size,
) -> Result<Region, MemError> {
    let bad = |rule: String| Err(MemError::undefined(call, rule));
    if level >= u64::from(g.mip_levels) {
        return bad(format!("mipmap level {level} of a texture with {} levels", g.mip_levels));
    }
    if slice >= u64::from(g.slices()) {
        return bad(format!("slice {slice} of a texture with {} slices", g.slices()));
    }
    let (lw, lh, ld) = g.level_extent(level as u32);
    let (ox, oy, oz) = (origin.x, origin.y, origin.z);
    let (sw, sh, sd) = (size.width, size.height, size.depth);
    if sw == 0 || sh == 0 || sd == 0 {
        return bad(format!("an empty region {sw}x{sh}x{sd}"));
    }
    let fits = |o: u64, s: u64, l: u32| o.checked_add(s).is_some_and(|e| e <= u64::from(l));
    if !fits(ox, sw, lw) || !fits(oy, sh, lh) || !fits(oz, sd, ld) {
        return bad(format!(
            "region origin ({ox}, {oy}, {oz}) size ({sw}, {sh}, {sd}) outside level {level} of extent {lw}x{lh}x{ld}"
        ));
    }
    let (ox, oy, oz, sw, sh, sd) = (ox as u32, oy as u32, oz as u32, sw as u32, sh as u32, sd as u32);
    let blocks = match unit.pvrtc {
        Some(bpp) => {
            if (ox, oy, oz, sw, sh) != (0, 0, 0, lw, lh) {
                return bad("a PVRTC blit copies the entire level".into());
            }
            pvrtc::word_dims(bpp, lw, lh)
        }
        None => {
            if unit.compressed() {
                let edge_ok =
                    |o: u32, s: u32, b: u32, l: u32| o.is_multiple_of(b) && (s.is_multiple_of(b) || o + s == l);
                if !edge_ok(ox, sw, unit.width, lw) || !edge_ok(oy, sh, unit.height, lh) {
                    return bad(format!(
                        "region ({ox}, {oy}) {sw}x{sh} is not block aligned for {}x{} blocks",
                        unit.width, unit.height
                    ));
                }
            }
            (sw.div_ceil(unit.width), sh.div_ceil(unit.height))
        }
    };
    Ok(Region { level: level as u32, slice: slice as u32, origin: (ox, oy, oz), size: (sw, sh, sd), blocks })
}

/// The buffer side of a blit between a buffer and a texture: the bytes of `region` at `offset` with the given strides,
/// laid out as the memory module's transfers take them. `buffer_len` is the Metal buffer's length.
pub fn buffer_layout(
    call: &'static str,
    g: &Geometry,
    unit: Unit,
    r: &Region,
    (offset, bytes_per_row, bytes_per_image): (u64, u64, u64),
    buffer_len: u64,
) -> Result<RegionLayout, MemError> {
    let bad = |rule: String| Err(MemError::undefined(call, rule));
    let u = u64::from(unit.bytes);
    if !offset.is_multiple_of(u) {
        return bad(format!("buffer offset {offset} is not a multiple of the {u}-byte pixel size"));
    }
    let row_bytes = u64::from(r.blocks.0) * u;
    let rows = r.blocks.1;
    let images = r.size.2;
    let row_limit = u * max_width(g.kind).div_ceil(u64::from(unit.width));
    let bytes_per_row = match bytes_per_row {
        0 if rows == 1 => row_bytes,
        0 => return bad(format!("bytesPerRow 0 for {rows} rows")),
        b if !b.is_multiple_of(u) => return bad(format!("bytesPerRow {b} is not a multiple of {u}")),
        b if b > row_limit => return bad(format!("bytesPerRow {b} exceeds {row_limit}")),
        b if rows > 1 && b < row_bytes => return bad(format!("bytesPerRow {b} is less than a row of {row_bytes}")),
        b => b,
    };
    let image = (u64::from(rows) - 1) * bytes_per_row + row_bytes;
    let bytes_per_image = match bytes_per_image {
        i if !i.is_multiple_of(u) => return bad(format!("bytesPerImage {i} is not a multiple of {u}")),
        _ if images == 1 => image,
        i if i < image => return bad(format!("bytesPerImage {i} is less than an image of {image} bytes")),
        i => i,
    };
    let span = (u64::from(images) - 1) * bytes_per_image + image;
    if offset.checked_add(span).is_none_or(|end| end > buffer_len) {
        return bad(format!("{span} bytes at offset {offset} of a {buffer_len}-byte buffer"));
    }
    Ok(RegionLayout {
        origin: r.origin,
        size: r.size,
        blocks: r.blocks,
        images,
        row_bytes,
        bytes_per_row,
        bytes_per_image,
        span,
        pvrtc: false,
    })
}

/// The layout of a region with tight strides, for copies between textures, which carry no buffer layout.
pub fn tight_layout(unit: Unit, r: &Region) -> RegionLayout {
    let row_bytes = u64::from(r.blocks.0) * u64::from(unit.bytes);
    let image = row_bytes * u64::from(r.blocks.1);
    RegionLayout {
        origin: r.origin,
        size: r.size,
        blocks: r.blocks,
        images: r.size.2,
        row_bytes,
        bytes_per_row: row_bytes,
        bytes_per_image: image,
        span: image * u64::from(r.size.2),
        pvrtc: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g2d(w: u32, h: u32, levels: u32) -> Geometry {
        Geometry {
            kind: TextureType::D2,
            width: w,
            height: h,
            depth: 1,
            array_length: 1,
            mip_levels: levels,
            sample_count: 1,
        }
    }

    fn o(x: u64, y: u64, z: u64) -> Origin {
        Origin { x, y, z }
    }

    fn s(width: u64, height: u64, depth: u64) -> Size {
        Size { width, height, depth }
    }

    const RGBA8: Unit = Unit { width: 1, height: 1, bytes: 4, pvrtc: None };

    #[test]
    fn strides_follow_the_blit_rules() {
        let g = g2d(16, 12, 1);
        let r = region("t", &g, RGBA8, (0, 0), &o(3, 2, 0), &s(7, 5, 1)).unwrap();
        // blit.buffer-to-texture-2d: a row pitch of 10 pixels, an offset of 5 pixels, a 2D bytesPerImage of 0.
        let l = buffer_layout("t", &g, RGBA8, &r, (20, 40, 0), 20 + 4 * 40 + 28).unwrap();
        assert_eq!((l.row_bytes, l.bytes_per_row, l.span), (28, 40, 4 * 40 + 28));
        // A 2D copy may carry a bytesPerImage; it is not read.
        assert!(buffer_layout("t", &g, RGBA8, &r, (20, 40, 4096), 20 + 4 * 40 + 28).is_ok());
        // Offsets and strides in whole pixels; the range inside the buffer.
        assert!(buffer_layout("t", &g, RGBA8, &r, (2, 40, 0), 1000).is_err());
        assert!(buffer_layout("t", &g, RGBA8, &r, (20, 42, 0), 1000).is_err());
        assert!(buffer_layout("t", &g, RGBA8, &r, (20, 24, 0), 1000).is_err());
        assert!(buffer_layout("t", &g, RGBA8, &r, (20, 40, 0), 20 + 4 * 40 + 28).is_ok());
        assert!(buffer_layout("t", &g, RGBA8, &r, (20, 40, 0), 20 + 4 * 40 + 27).is_err());
        assert!(buffer_layout("t", &g, RGBA8, &r, (20, 0, 0), 1000).is_err());
        assert!(buffer_layout("t", &g, RGBA8, &r, (20, 4 * 16385, 0), 1 << 20).is_err());
    }

    #[test]
    fn regions_stay_inside_the_level() {
        let g = g2d(8, 8, 3);
        assert!(region("t", &g, RGBA8, (0, 2), &o(0, 0, 0), &s(2, 2, 1)).is_ok());
        assert!(region("t", &g, RGBA8, (0, 2), &o(1, 0, 0), &s(2, 2, 1)).is_err());
        assert!(region("t", &g, RGBA8, (0, 3), &o(0, 0, 0), &s(1, 1, 1)).is_err());
        assert!(region("t", &g, RGBA8, (1, 0), &o(0, 0, 0), &s(1, 1, 1)).is_err());
        assert!(region("t", &g, RGBA8, (0, 0), &o(0, 0, 0), &s(0, 1, 1)).is_err());
        let etc = Unit { width: 4, height: 4, bytes: 8, pvrtc: None };
        let g = g2d(6, 6, 1);
        assert_eq!(region("t", &g, etc, (0, 0), &o(4, 0, 0), &s(2, 4, 1)).unwrap().blocks, (1, 1));
        assert!(region("t", &g, etc, (0, 0), &o(2, 0, 0), &s(4, 4, 1)).is_err());
    }

    #[test]
    fn pvrtc_is_whole_levels_of_words() {
        let unit = Unit { width: 4, height: 4, bytes: 8, pvrtc: Some(pvrtc::Bpp::Four) };
        let g = g2d(16, 16, 1);
        let r = region("t", &g, unit, (0, 0), &o(0, 0, 0), &s(16, 16, 1)).unwrap();
        assert_eq!(r.blocks, (4, 4));
        let l = buffer_layout("t", &g, unit, &r, (0, 32, 128), 128).unwrap();
        assert_eq!((l.row_bytes, l.span), (32, 128));
        assert!(region("t", &g, unit, (0, 0), &o(0, 0, 0), &s(8, 8, 1)).is_err());
    }
}
