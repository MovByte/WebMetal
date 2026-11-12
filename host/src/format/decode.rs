//! Software decoding of Metal's compressed pixel formats into the texels the emulated texture stores.
//!
//! [`scheme`] names the decoder of a Metal pixel format; [`decode`] decodes one 2D image (a mipmap level of one slice,
//! or one depth slice of a 3D texture) given as the bytes Metal accepts for it. Block formats take their blocks in
//! row-major order of blocks with `bytes_per_row` bytes between rows of blocks; PVRTC takes the whole level in
//! reflected Morton order, as `replaceRegion` does for PVRTC with a `bytesPerRow` of 0.

use super::astc::{self, OperationMode};
use super::etc2;
use super::pvrtc::{self, Bpp};

/// The decoder of a compressed Metal pixel format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    /// `ETC2_RGB8`, `ETC2_RGB8_sRGB`: 8-bit RGB, alpha 255.
    Etc2Rgb8 { srgb: bool },
    /// `ETC2_RGB8A1`, `ETC2_RGB8A1_sRGB`: 8-bit RGBA with punchthrough alpha.
    Etc2Rgb8A1 { srgb: bool },
    /// `EAC_RGBA8`, `EAC_RGBA8_sRGB`: 8-bit RGBA.
    EacRgba8 { srgb: bool },
    /// `EAC_R11Unorm`, `EAC_R11Snorm`, `EAC_RG11Unorm`, `EAC_RG11Snorm`: 11-bit channels.
    Eac11 { channels: u32, signed: bool },
    /// `ASTC_*_LDR` (linear LDR operation mode) and `ASTC_*_sRGB` (sRGB operation mode).
    Astc { width: u32, height: u32, srgb: bool },
    /// `PVRTC_RGB_*` (`alpha` false) and `PVRTC_RGBA_*`, linear or sRGB.
    Pvrtc { bpp: Bpp, alpha: bool, srgb: bool },
}

impl Scheme {
    /// Texel block width and height.
    pub fn block_dims(self) -> (u32, u32) {
        match self {
            Scheme::Astc { width, height, .. } => (width, height),
            Scheme::Pvrtc { bpp, .. } => bpp.block_dims(),
            _ => (etc2::BLOCK_DIM, etc2::BLOCK_DIM),
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Scheme::Etc2Rgb8 { .. } | Scheme::Etc2Rgb8A1 { .. } => 8,
            Scheme::EacRgba8 { .. } => 16,
            Scheme::Eac11 { channels, .. } => 8 * channels as usize,
            Scheme::Astc { .. } => astc::BLOCK_BYTES,
            Scheme::Pvrtc { .. } => pvrtc::WORD_BYTES,
        }
    }

    pub fn srgb(self) -> bool {
        match self {
            Scheme::Etc2Rgb8 { srgb } | Scheme::Etc2Rgb8A1 { srgb } | Scheme::EacRgba8 { srgb } => srgb,
            Scheme::Astc { srgb, .. } | Scheme::Pvrtc { srgb, .. } => srgb,
            Scheme::Eac11 { .. } => false,
        }
    }
}

/// The decoder of a Metal pixel format (`MTLPixelFormat` value), if it is a compressed format this module decodes.
///
/// The values come from the SDK header (`tables/formats.toml`). The ASTC footprints follow the format names; the HDR
/// ASTC formats, the BC formats and the private ASTC formats of the A10X profile have no scheme here (see
/// [`super::handling`]).
pub fn scheme(value: u32) -> Option<Scheme> {
    const ASTC: [(u32, u32); 14] = astc::FOOTPRINTS;
    Some(match value {
        160 => Scheme::Pvrtc { bpp: Bpp::Two, alpha: false, srgb: false },
        161 => Scheme::Pvrtc { bpp: Bpp::Two, alpha: false, srgb: true },
        162 => Scheme::Pvrtc { bpp: Bpp::Four, alpha: false, srgb: false },
        163 => Scheme::Pvrtc { bpp: Bpp::Four, alpha: false, srgb: true },
        164 => Scheme::Pvrtc { bpp: Bpp::Two, alpha: true, srgb: false },
        165 => Scheme::Pvrtc { bpp: Bpp::Two, alpha: true, srgb: true },
        166 => Scheme::Pvrtc { bpp: Bpp::Four, alpha: true, srgb: false },
        167 => Scheme::Pvrtc { bpp: Bpp::Four, alpha: true, srgb: true },
        170 => Scheme::Eac11 { channels: 1, signed: false },
        172 => Scheme::Eac11 { channels: 1, signed: true },
        174 => Scheme::Eac11 { channels: 2, signed: false },
        176 => Scheme::Eac11 { channels: 2, signed: true },
        178 => Scheme::EacRgba8 { srgb: false },
        179 => Scheme::EacRgba8 { srgb: true },
        180 => Scheme::Etc2Rgb8 { srgb: false },
        181 => Scheme::Etc2Rgb8 { srgb: true },
        182 => Scheme::Etc2Rgb8A1 { srgb: false },
        183 => Scheme::Etc2Rgb8A1 { srgb: true },
        // ASTC_4x4_sRGB = 186 ... ASTC_12x12_sRGB = 200 and ASTC_4x4_LDR = 204 ... ASTC_12x12_LDR = 218, in the order of
        // Table 155, skipping the values the SDK does not assign (191, 209).
        186..=200 | 204..=218 => {
            let srgb = value <= 200;
            let index = if srgb { value - 186 } else { value - 204 };
            let index = match index {
                0..=4 => index,
                6..=14 => index - 1,
                _ => return None,
            };
            let (width, height) = ASTC[index as usize];
            Scheme::Astc { width, height, srgb }
        }
        _ => return None,
    })
}

/// Decoded texels of one image, in raster order.
#[derive(Clone, Debug, PartialEq)]
pub enum Texels {
    /// 8-bit RGBA; R, G and B are sRGB-encoded for the sRGB formats.
    Rgba8(Vec<[u8; 4]>),
    /// The UNORM16 interpolation result `C` of ASTC linear LDR (section 23.18.1), the error color as
    /// [`astc::ERROR_UNORM16`]. [`astc::to_unorm12`] gives the value real Metal returns from it.
    Unorm16(Vec<[u16; 4]>),
    /// Unsigned 11-bit EAC values (0 = 0.0, 2047 = 1.0), one per channel.
    Unsigned11(Vec<[u16; 2]>),
    /// Signed 11-bit EAC values (-1023 = -1.0, 1023 = 1.0), one per channel.
    Signed11(Vec<[i16; 2]>),
}

/// Why an image cannot be decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The data is shorter than the image's blocks.
    Truncated {
        needed: usize,
        got: usize,
    },
    /// `bytes_per_row` is smaller than a row of blocks.
    RowTooShort {
        bytes_per_row: usize,
        needed: usize,
    },
    Pvrtc(pvrtc::PvrtcError),
}

/// An image of `width` × `height` texels made of `bw` × `bh` blocks.
struct Grid {
    width: u32,
    height: u32,
    bw: u32,
    bh: u32,
}

impl Grid {
    /// Copies decoded block `(bx, by)` (raster order, `bw` texels per row) into the image, dropping padding texels.
    fn place<T: Copy>(&self, out: &mut [T], bx: usize, by: usize, block: &[T]) {
        for y in 0..self.bh {
            for x in 0..self.bw {
                let (ix, iy) = (bx as u32 * self.bw + x, by as u32 * self.bh + y);
                if ix < self.width && iy < self.height {
                    out[(iy * self.width + ix) as usize] = block[(y * self.bw + x) as usize];
                }
            }
        }
    }
}

/// Decodes a `width` × `height` image of `scheme`.
pub fn decode(
    scheme: Scheme,
    width: u32,
    height: u32,
    data: &[u8],
    bytes_per_row: usize,
) -> Result<Texels, DecodeError> {
    if let Scheme::Pvrtc { bpp, alpha, .. } = scheme {
        let texels = if alpha {
            pvrtc::decode(bpp, width, height, data)
        } else {
            pvrtc::decode_opaque(bpp, width, height, data)
        }
        .map_err(DecodeError::Pvrtc)?;
        return Ok(Texels::Rgba8(texels));
    }
    let (bw, bh) = scheme.block_dims();
    let block_bytes = scheme.block_bytes();
    let (blocks_x, blocks_y) = (width.div_ceil(bw) as usize, height.div_ceil(bh) as usize);
    let row = blocks_x * block_bytes;
    if bytes_per_row < row {
        return Err(DecodeError::RowTooShort { bytes_per_row, needed: row });
    }
    let needed = if blocks_y == 0 { 0 } else { (blocks_y - 1) * bytes_per_row + row };
    if data.len() < needed {
        return Err(DecodeError::Truncated { needed, got: data.len() });
    }
    let texel_count = width as usize * height as usize;
    let block_at = |bx: usize, by: usize| &data[by * bytes_per_row + bx * block_bytes..][..block_bytes];
    let grid = Grid { width, height, bw, bh };
    Ok(match scheme {
        Scheme::Etc2Rgb8 { .. } | Scheme::Etc2Rgb8A1 { .. } | Scheme::EacRgba8 { .. } => {
            let mut out = vec![[0u8; 4]; texel_count];
            for by in 0..blocks_y {
                for bx in 0..blocks_x {
                    let b = block_at(bx, by);
                    let texels = match scheme {
                        Scheme::Etc2Rgb8 { .. } => etc2::decode_rgb8(b.try_into().unwrap()),
                        Scheme::Etc2Rgb8A1 { .. } => etc2::decode_rgb8a1(b.try_into().unwrap()),
                        _ => etc2::decode_rgba8(b.try_into().unwrap()),
                    };
                    grid.place(&mut out, bx, by, &texels);
                }
            }
            Texels::Rgba8(out)
        }
        Scheme::Eac11 { channels, signed } => {
            if signed {
                let mut out = vec![[0i16; 2]; texel_count];
                for by in 0..blocks_y {
                    for bx in 0..blocks_x {
                        let b = block_at(bx, by);
                        let mut texels = [[0i16; 2]; 16];
                        for c in 0..channels as usize {
                            let v = etc2::decode_r11_signed(b[8 * c..8 * c + 8].try_into().unwrap());
                            for i in 0..16 {
                                texels[i][c] = v[i];
                            }
                        }
                        grid.place(&mut out, bx, by, &texels);
                    }
                }
                Texels::Signed11(out)
            } else {
                let mut out = vec![[0u16; 2]; texel_count];
                for by in 0..blocks_y {
                    for bx in 0..blocks_x {
                        let b = block_at(bx, by);
                        let mut texels = [[0u16; 2]; 16];
                        for c in 0..channels as usize {
                            let v = etc2::decode_r11_unsigned(b[8 * c..8 * c + 8].try_into().unwrap());
                            for i in 0..16 {
                                texels[i][c] = v[i];
                            }
                        }
                        grid.place(&mut out, bx, by, &texels);
                    }
                }
                Texels::Unsigned11(out)
            }
        }
        Scheme::Astc { width: fw, height: fh, srgb } => {
            let mode = if srgb { OperationMode::Srgb } else { OperationMode::LinearLdr };
            if srgb {
                let mut out = vec![[0u8; 4]; texel_count];
                for by in 0..blocks_y {
                    for bx in 0..blocks_x {
                        let decoded = astc::decode_block(block_at(bx, by).try_into().unwrap(), fw, fh, mode)
                            .expect("scheme() yields Table 155 footprints only");
                        let texels: Vec<[u8; 4]> = decoded
                            .into_iter()
                            .map(|t| t.map_or(astc::ERROR_SRGB8, |c| c.map(astc::to_srgb8)))
                            .collect();
                        grid.place(&mut out, bx, by, &texels);
                    }
                }
                Texels::Rgba8(out)
            } else {
                let mut out = vec![[0u16; 4]; texel_count];
                for by in 0..blocks_y {
                    for bx in 0..blocks_x {
                        let decoded = astc::decode_block(block_at(bx, by).try_into().unwrap(), fw, fh, mode)
                            .expect("scheme() yields Table 155 footprints only");
                        let texels: Vec<[u16; 4]> =
                            decoded.into_iter().map(|t| t.unwrap_or(astc::ERROR_UNORM16)).collect();
                        grid.place(&mut out, bx, by, &texels);
                    }
                }
                Texels::Unorm16(out)
            }
        }
        Scheme::Pvrtc { .. } => unreachable!("handled above"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn astc_values_follow_the_header() {
        // MTLPixelFormat.h: ASTC_4x4_sRGB = 186 ... ASTC_6x6_sRGB = 190, ASTC_8x5_sRGB = 192 ... ASTC_12x12_sRGB = 200,
        // ASTC_4x4_LDR = 204 ... ASTC_6x6_LDR = 208, ASTC_8x5_LDR = 210 ... ASTC_12x12_LDR = 218.
        let table = &super::super::table::METAL_FORMATS;
        for row in table.iter().filter(|r| r.name.starts_with("MTLPixelFormatASTC_") && !r.name.ends_with("_HDR")) {
            let s = scheme(row.value).unwrap_or_else(|| panic!("{}", row.name));
            let Scheme::Astc { width, height, srgb } = s else { panic!() };
            let expect = format!("MTLPixelFormatASTC_{width}x{height}_{}", if srgb { "sRGB" } else { "LDR" });
            assert_eq!(row.name, expect);
        }
        assert_eq!(scheme(191), None);
        assert_eq!(scheme(209), None);
        assert_eq!(scheme(222), None);
    }
}
