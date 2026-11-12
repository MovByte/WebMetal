//! Texture geometry and the byte layouts of Metal's CPU and blit transfers.
//!
//! Metal names texels of a mipmap level of a slice by an `MTLRegion` and lays the bytes of a transfer out with a row
//! stride (`bytesPerRow`) and an image stride (`bytesPerImage`). For uncompressed formats a row is a row of pixels;
//! for compressed formats it is a row of blocks; for PVRTC the whole level is one run of 64-bit words in reflected
//! Morton order with both strides 0 (`replaceRegion`/`getBytes` documentation, quoted in [`RegionLayout::new`]).
//!
//! WebGPU copies between buffers and textures need `bytesPerRow` to be a multiple of 256 (WebGPU section 13.3,
//! `tables/limits.toml` question `6.3-copy-alignment`); [`StagingPitch`] is the aligned layout the executor stages
//! through, so the layouts the guest sees are Metal's.

use crate::format::decode::Scheme;
use crate::format::handling::{Resolution, Upload};
use crate::format::pvrtc::{self, Bpp};
use crate::format::webgpu::Table;

use super::error::MemError;

/// WebGPU's `bytesPerRow` alignment for buffer-texture copies.
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u64 = 256;

/// `MTLTextureType`, from `MTLTexture.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureType {
    D1,
    D1Array,
    D2,
    D2Array,
    D2Multisample,
    Cube,
    CubeArray,
    D3,
    D2MultisampleArray,
    TextureBuffer,
}

impl TextureType {
    pub fn from_value(v: u64, call: &'static str) -> Result<TextureType, MemError> {
        Ok(match v {
            0 => TextureType::D1,
            1 => TextureType::D1Array,
            2 => TextureType::D2,
            3 => TextureType::D2Array,
            4 => TextureType::D2Multisample,
            5 => TextureType::Cube,
            6 => TextureType::CubeArray,
            7 => TextureType::D3,
            8 => TextureType::D2MultisampleArray,
            9 => TextureType::TextureBuffer,
            v => return Err(MemError::undefined(call, format!("{v} is not an MTLTextureType value (MTLTexture.h)"))),
        })
    }

    pub fn is_1d(self) -> bool {
        matches!(self, TextureType::D1 | TextureType::D1Array)
    }

    pub fn is_3d(self) -> bool {
        self == TextureType::D3
    }
}

/// How a Metal format's bytes are grouped: blocks of `width` × `height` texels of `bytes` bytes each (1 × 1 for
/// uncompressed formats), or the words of a PVRTC level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blocks {
    pub width: u32,
    pub height: u32,
    pub bytes: u32,
    pub pvrtc: Option<Bpp>,
}

impl Blocks {
    /// The block layout of a Metal format as the host provides it. A combined depth-stencil format has no single
    /// texel size: its bytes move one aspect at a time, which only blits do.
    pub fn of(resolution: Resolution) -> Option<Blocks> {
        match resolution {
            Resolution::Native(f) => {
                let caps = f.caps();
                let bytes = match (caps.table, caps.aspects) {
                    (Table::DepthStencil, [only]) => only.footprint?,
                    (Table::DepthStencil, _) => return None,
                    _ => caps.footprint,
                };
                Some(Blocks { width: caps.block.0, height: caps.block.1, bytes, pvrtc: None })
            }
            Resolution::StencilAspect => None,
            Resolution::Emulated(e) => Some(match e.upload {
                Upload::Bytes | Upload::SwapRedBlue10 => {
                    Blocks { width: 1, height: 1, bytes: e.storage.caps().footprint, pvrtc: None }
                }
                Upload::Decode(scheme) => {
                    let (width, height) = scheme.block_dims();
                    let pvrtc = match scheme {
                        Scheme::Pvrtc { bpp, .. } => Some(bpp),
                        _ => None,
                    };
                    Blocks { width, height, bytes: scheme.block_bytes() as u32, pvrtc }
                }
            }),
        }
    }

    pub fn compressed(self) -> bool {
        self.width != 1 || self.height != 1
    }

    /// Blocks covering `w` × `h` texels of a level: whole blocks, and for PVRTC at least two words each way.
    pub fn count(self, w: u32, h: u32) -> (u32, u32) {
        match self.pvrtc {
            Some(bpp) => pvrtc::word_dims(bpp, w, h),
            None => (w.div_ceil(self.width), h.div_ceil(self.height)),
        }
    }
}

/// The dimensions and subresources of a texture, as Metal defines them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub kind: TextureType,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_length: u32,
    pub mip_levels: u32,
    pub sample_count: u32,
}

impl Geometry {
    /// Slices of the texture: array elements, faces (six per cube) or one.
    pub fn slices(&self) -> u32 {
        match self.kind {
            TextureType::D1Array | TextureType::D2Array | TextureType::D2MultisampleArray => self.array_length,
            TextureType::Cube => 6,
            TextureType::CubeArray => 6 * self.array_length,
            _ => 1,
        }
    }

    /// The extent of `level`: each dimension halves per level, down to 1.
    pub fn level_extent(&self, level: u32) -> (u32, u32, u32) {
        let d = if self.kind.is_3d() { (self.depth >> level).max(1) } else { 1 };
        ((self.width >> level).max(1), (self.height >> level).max(1), d)
    }

    /// WebGPU's `depthOrArrayLayers`: the 3D depth, or the slices.
    pub fn layers(&self) -> u32 {
        if self.kind.is_3d() { self.depth } else { self.slices() }
    }
}

/// A region transfer between a texture subresource and bytes in Metal's layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionLayout {
    /// Texel origin and size of the region (size in texels, the last block may be partial at the level's edge).
    pub origin: (u32, u32, u32),
    pub size: (u32, u32, u32),
    /// Blocks of the region in x and y, and images (3D depth).
    pub blocks: (u32, u32),
    pub images: u32,
    /// Bytes of one row of blocks.
    pub row_bytes: u64,
    /// Row and image strides of the Metal bytes (0 strides resolved to the tight values).
    pub bytes_per_row: u64,
    pub bytes_per_image: u64,
    /// Bytes from the first to past the last byte of the region in Metal's layout.
    pub span: u64,
    /// The region is a whole PVRTC level, as Morton-ordered words.
    pub pvrtc: bool,
}

/// The region arguments of a Metal call, before validation.
#[derive(Clone, Copy, Debug)]
pub struct RegionArgs {
    pub origin: (u64, u64, u64),
    pub size: (u64, u64, u64),
    pub level: u64,
    pub slice: u64,
    pub bytes_per_row: u64,
    pub bytes_per_image: u64,
}

impl RegionLayout {
    /// Validate and lay out a region of `geometry`'s format `blocks`, with Metal's rules for `replaceRegion` and
    /// `getBytes` (their documentation, iPhoneOS SDK 27.2):
    ///
    /// - "The region needs to be within the dimensions of the slice"; `slice` "For all other texture types, use 0"
    ///   (cube faces 0-5, array index, cube array `cubeFace + arrayIndex * 6`).
    /// - bytesPerRow: "For type1D and type1DArray, use 0 ... When source data consists of only a single row, use 0 ...
    ///   For compressed pixel formats, the stride is the number of bytes from the beginning of one row of blocks to
    ///   the beginning of the next ... When working with PowerVR Texture Compression (PVRTC), use 0. Nonzero values
    ///   smaller than the texture width or not a multiple of the pixel size cause an error."
    /// - bytesPerImage: "Supply a nonzero value only when you copy data to a type3D type texture ... For data that
    ///   consists only of a single image, use 0 ... For ordinary or packed pixel formats, use a multiple of pixel
    ///   size. For compressed pixel data, use a multiple of the compression block size. When working with PVRTC, use
    ///   0."
    /// - "If the texture image has a compressed pixel format, only write to block-aligned regions. If the size of a
    ///   dimension of region isn't a multiple of the block size, include both the edge block and additional space up
    ///   to the texture dimensions"; for PVRTC "use the entire texture for the region".
    ///
    /// Anything outside these rules is behavior Metal does not define.
    pub fn new(call: &'static str, g: &Geometry, blocks: Blocks, a: &RegionArgs) -> Result<RegionLayout, MemError> {
        let bad = |rule: String| Err(MemError::undefined(call, rule));
        if a.level >= u64::from(g.mip_levels) {
            return bad(format!("mipmap level {} of a texture with {} levels", a.level, g.mip_levels));
        }
        if a.slice >= u64::from(g.slices()) {
            return bad(format!("slice {} of a texture with {} slices", a.slice, g.slices()));
        }
        let (lw, lh, ld) = g.level_extent(a.level as u32);
        let (ox, oy, oz) = a.origin;
        let (sw, sh, sd) = a.size;
        if sw == 0 || sh == 0 || sd == 0 {
            return bad(format!("an empty region {sw}x{sh}x{sd}"));
        }
        let fits = |o: u64, s: u64, l: u32| o.checked_add(s).is_some_and(|e| e <= u64::from(l));
        if !fits(ox, sw, lw) || !fits(oy, sh, lh) || !fits(oz, sd, ld) {
            return bad(format!(
                "region origin ({ox}, {oy}, {oz}) size ({sw}, {sh}, {sd}) outside level {} of extent {lw}x{lh}x{ld}",
                a.level
            ));
        }
        let (ox, oy, oz, sw, sh, sd) = (ox as u32, oy as u32, oz as u32, sw as u32, sh as u32, sd as u32);
        if let Some(bpp) = blocks.pvrtc {
            if (ox, oy, oz, sw, sh) != (0, 0, 0, lw, lh) {
                return bad("a PVRTC transfer must cover the entire level".into());
            }
            if a.bytes_per_row != 0 || a.bytes_per_image != 0 {
                return bad("PVRTC transfers take bytesPerRow and bytesPerImage 0".into());
            }
            let (wx, wy) = blocks.count(lw, lh);
            let len = pvrtc::data_len(bpp, lw, lh) as u64;
            return Ok(RegionLayout {
                origin: (0, 0, 0),
                size: (lw, lh, 1),
                blocks: (wx, wy),
                images: 1,
                row_bytes: u64::from(wx) * u64::from(blocks.bytes),
                bytes_per_row: u64::from(wx) * u64::from(blocks.bytes),
                bytes_per_image: len,
                span: len,
                pvrtc: true,
            });
        }
        if blocks.compressed() {
            let edge_ok = |o: u32, s: u32, b: u32, l: u32| o.is_multiple_of(b) && (s.is_multiple_of(b) || o + s == l);
            if !edge_ok(ox, sw, blocks.width, lw) || !edge_ok(oy, sh, blocks.height, lh) {
                return bad(format!(
                    "region ({ox}, {oy}) {sw}x{sh} is not block aligned for {}x{} blocks",
                    blocks.width, blocks.height
                ));
            }
        }
        let (bx, by) = (sw.div_ceil(blocks.width), sh.div_ceil(blocks.height));
        let unit = u64::from(blocks.bytes);
        let row_bytes = u64::from(bx) * unit;
        let bytes_per_row = match a.bytes_per_row {
            0 if by == 1 || g.kind.is_1d() => row_bytes,
            0 => return bad(format!("bytesPerRow 0 for {by} rows")),
            r if r < row_bytes => return bad(format!("bytesPerRow {r} is less than a row of {row_bytes} bytes")),
            r if !r.is_multiple_of(unit) => return bad(format!("bytesPerRow {r} is not a multiple of {unit}")),
            r => r,
        };
        let image = (u64::from(by) - 1) * bytes_per_row + row_bytes;
        let bytes_per_image = match (a.bytes_per_image, sd) {
            (0, 1) => image,
            (0, _) => return bad(format!("bytesPerImage 0 for a region {sd} images deep")),
            (_, _) if !g.kind.is_3d() => return bad("a nonzero bytesPerImage for a texture that is not 3D".into()),
            (i, _) if i < image => return bad(format!("bytesPerImage {i} is less than an image of {image} bytes")),
            (i, _) if !i.is_multiple_of(unit) => return bad(format!("bytesPerImage {i} is not a multiple of {unit}")),
            (i, _) => i,
        };
        let span = (u64::from(sd) - 1) * bytes_per_image + image;
        Ok(RegionLayout {
            origin: (ox, oy, oz),
            size: (sw, sh, sd),
            blocks: (bx, by),
            images: sd,
            row_bytes,
            bytes_per_row,
            bytes_per_image,
            span,
            pvrtc: false,
        })
    }

    /// The tight bytes of the region (rows of `row_bytes`, images of `blocks.1` rows) from Metal's layout.
    pub fn pack(&self, metal: &[u8]) -> Vec<u8> {
        if self.pvrtc {
            return metal[..self.span as usize].to_vec();
        }
        let mut out = Vec::with_capacity((self.row_bytes * u64::from(self.blocks.1) * u64::from(self.images)) as usize);
        for i in 0..u64::from(self.images) {
            for r in 0..u64::from(self.blocks.1) {
                let at = (i * self.bytes_per_image + r * self.bytes_per_row) as usize;
                out.extend_from_slice(&metal[at..at + self.row_bytes as usize]);
            }
        }
        out
    }

    /// Metal's layout from tight bytes, over `metal` (bytes between rows and images keep their values).
    pub fn unpack(&self, tight: &[u8], metal: &mut [u8]) {
        if self.pvrtc {
            metal[..self.span as usize].copy_from_slice(&tight[..self.span as usize]);
            return;
        }
        let row = self.row_bytes as usize;
        let mut k = 0;
        for i in 0..u64::from(self.images) {
            for r in 0..u64::from(self.blocks.1) {
                let at = (i * self.bytes_per_image + r * self.bytes_per_row) as usize;
                metal[at..at + row].copy_from_slice(&tight[k..k + row]);
                k += row;
            }
        }
    }

    /// Bytes of the tight layout.
    pub fn tight_len(&self) -> u64 {
        if self.pvrtc { self.span } else { self.row_bytes * u64::from(self.blocks.1) * u64::from(self.images) }
    }
}

/// A buffer layout WebGPU accepts for copies with a texture: rows padded to a multiple of 256 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagingPitch {
    pub bytes_per_row: u64,
    pub rows_per_image: u64,
    pub images: u64,
    pub row_bytes: u64,
}

impl StagingPitch {
    pub fn new(row_bytes: u64, rows: u64, images: u64) -> StagingPitch {
        StagingPitch {
            bytes_per_row: row_bytes.next_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT),
            rows_per_image: rows,
            images,
            row_bytes,
        }
    }

    pub fn size(&self) -> u64 {
        self.bytes_per_row * self.rows_per_image * self.images
    }

    /// The tight rows of staged bytes.
    pub fn tighten(&self, staged: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity((self.row_bytes * self.rows_per_image * self.images) as usize);
        for i in 0..self.images * self.rows_per_image {
            let at = (i * self.bytes_per_row) as usize;
            out.extend_from_slice(&staged[at..at + self.row_bytes as usize]);
        }
        out
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

    fn args(origin: (u64, u64, u64), size: (u64, u64, u64), bpr: u64, bpi: u64) -> RegionArgs {
        RegionArgs { origin, size, level: 0, slice: 0, bytes_per_row: bpr, bytes_per_image: bpi }
    }

    const RGBA8: Blocks = Blocks { width: 1, height: 1, bytes: 4, pvrtc: None };
    const ETC: Blocks = Blocks { width: 4, height: 4, bytes: 8, pvrtc: None };

    #[test]
    fn strides() {
        let l = RegionLayout::new("t", &g2d(8, 4, 1), RGBA8, &args((1, 1, 0), (3, 2, 1), 40, 0)).unwrap();
        assert_eq!((l.row_bytes, l.bytes_per_row, l.bytes_per_image, l.span), (12, 40, 52, 52));
        let metal: Vec<u8> = (0..52).collect();
        let tight = l.pack(&metal);
        assert_eq!(tight, [&metal[0..12], &metal[40..52]].concat());
        let mut back = vec![0xee; 52];
        l.unpack(&tight, &mut back);
        assert_eq!(back[..12], metal[..12]);
        assert_eq!(back[12..40], [0xee; 28]);
        // A single row may use bytesPerRow 0; several may not.
        assert!(RegionLayout::new("t", &g2d(8, 4, 1), RGBA8, &args((0, 0, 0), (8, 1, 1), 0, 0)).is_ok());
        assert!(RegionLayout::new("t", &g2d(8, 4, 1), RGBA8, &args((0, 0, 0), (8, 2, 1), 0, 0)).is_err());
        assert!(RegionLayout::new("t", &g2d(8, 4, 1), RGBA8, &args((0, 0, 0), (8, 2, 1), 30, 0)).is_err());
        assert!(RegionLayout::new("t", &g2d(8, 4, 1), RGBA8, &args((0, 0, 0), (8, 2, 1), 34, 0)).is_err());
        assert!(RegionLayout::new("t", &g2d(8, 4, 1), RGBA8, &args((5, 0, 0), (4, 1, 1), 0, 0)).is_err());
        // bytesPerImage only for 3D.
        assert!(RegionLayout::new("t", &g2d(8, 4, 1), RGBA8, &args((0, 0, 0), (8, 2, 1), 32, 64)).is_err());
    }

    #[test]
    fn compressed_edges() {
        // A 6x6 level of 4x4 blocks: the edge block at x = 4 may have width 2.
        let g = g2d(6, 6, 1);
        let l = RegionLayout::new("t", &g, ETC, &args((4, 0, 0), (2, 4, 1), 8, 0)).unwrap();
        assert_eq!((l.blocks, l.row_bytes), ((1, 1), 8));
        assert!(RegionLayout::new("t", &g, ETC, &args((2, 0, 0), (4, 4, 1), 8, 0)).is_err());
        assert!(RegionLayout::new("t", &g, ETC, &args((0, 0, 0), (3, 4, 1), 8, 0)).is_err());
        let all = RegionLayout::new("t", &g, ETC, &args((0, 0, 0), (6, 6, 1), 16, 0)).unwrap();
        assert_eq!((all.blocks, all.span), ((2, 2), 32));
    }

    #[test]
    fn three_d_and_arrays() {
        let g = Geometry {
            kind: TextureType::D3,
            width: 4,
            height: 4,
            depth: 8,
            array_length: 1,
            mip_levels: 3,
            sample_count: 1,
        };
        assert_eq!(g.level_extent(2), (1, 1, 2));
        let mut a = args((0, 0, 0), (1, 1, 2), 0, 0);
        a.level = 2;
        assert!(RegionLayout::new("t", &g, RGBA8, &a).is_err());
        a.bytes_per_image = 8;
        assert_eq!(RegionLayout::new("t", &g, RGBA8, &a).unwrap().span, 12);
        let cube = Geometry { kind: TextureType::CubeArray, array_length: 2, ..g2d(4, 4, 1) };
        assert_eq!(cube.slices(), 12);
        let mut a = args((0, 0, 0), (4, 4, 1), 16, 0);
        a.slice = 11;
        assert!(RegionLayout::new("t", &cube, RGBA8, &a).is_ok());
        a.slice = 12;
        assert!(RegionLayout::new("t", &cube, RGBA8, &a).is_err());
    }

    #[test]
    fn pvrtc_whole_level() {
        let b = Blocks { width: 4, height: 4, bytes: 8, pvrtc: Some(Bpp::Four) };
        let g = g2d(16, 16, 1);
        let l = RegionLayout::new("t", &g, b, &args((0, 0, 0), (16, 16, 1), 0, 0)).unwrap();
        assert_eq!((l.blocks, l.span), ((4, 4), 128));
        assert!(RegionLayout::new("t", &g, b, &args((0, 0, 0), (8, 8, 1), 0, 0)).is_err());
        assert!(RegionLayout::new("t", &g, b, &args((0, 0, 0), (16, 16, 1), 32, 0)).is_err());
        // Levels below two words keep two.
        let small = g2d(4, 4, 1);
        assert_eq!(RegionLayout::new("t", &small, b, &args((0, 0, 0), (4, 4, 1), 0, 0)).unwrap().span, 32);
    }

    #[test]
    fn staging_pitch() {
        let p = StagingPitch::new(12, 2, 1);
        assert_eq!((p.bytes_per_row, p.size()), (256, 512));
        let mut staged = vec![0u8; 512];
        staged[..12].fill(1);
        staged[256..268].fill(2);
        assert_eq!(p.tighten(&staged), [[1u8; 12], [2u8; 12]].concat());
    }
}
