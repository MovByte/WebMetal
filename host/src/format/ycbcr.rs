//! Y'CbCr data: the 4:2:2 formats `GBGR422` and `BGRG422`, and textures over the planes of multi-planar IOSurfaces.
//!
//! The 4:2:2 formats store two pixels in 32 bits, `GBGR422` as Y0, Cb, Y1, Cr from the lowest byte and `BGRG422` as
//! Cb, Y0, Cr, Y1, and a read returns (Cr, Y, Cb, 1) without colorspace conversion (`MTLPixelFormat.h`, quoted in
//! `tables/formats.toml` question `6.3-422-chroma`). The executor stores the bytes unchanged as `rg8uint`, one
//! (Y, C) or (C, Y) pair per pixel, so uploads, readbacks and copies are exact; reads are refused until the chroma
//! reconstruction is known (see [`super::handling`]).
//!
//! The A10X's private multi-planar Y'CbCr formats have no definition available here and are refused. A planar
//! IOSurface reaches Metal as one texture per plane in an ordinary format (one channel of luma, two of chroma);
//! [`plane_rows`] locates such a texture's rows in the surface.

use crate::embed::SurfacePlane;

/// The byte order of a 4:2:2 format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order422 {
    /// `GBGR422`: Y0, Cb, Y1, Cr.
    LumaFirst,
    /// `BGRG422`: Cb, Y0, Cr, Y1.
    ChromaFirst,
}

/// The order of a 4:2:2 Metal pixel format.
pub fn order_422(value: u32) -> Option<Order422> {
    match value {
        240 => Some(Order422::LumaFirst),
        241 => Some(Order422::ChromaFirst),
        _ => None,
    }
}

impl Order422 {
    /// The `rg8uint` channel (0 = red, 1 = green) that holds a pixel's luma.
    pub fn luma_channel(self) -> usize {
        match self {
            Order422::LumaFirst => 0,
            Order422::ChromaFirst => 1,
        }
    }

    /// The luma of each pixel of a row of pairs.
    pub fn luma(self, bytes: &[u8]) -> Vec<u8> {
        bytes.as_chunks::<2>().0.iter().map(|px| px[self.luma_channel()]).collect()
    }

    /// The (Cb, Cr) of each pair of pixels of a row.
    pub fn chroma(self, bytes: &[u8]) -> Vec<(u8, u8)> {
        let c = 1 - self.luma_channel();
        bytes.as_chunks::<4>().0.iter().map(|pair| (pair[c], pair[2 + c])).collect()
    }
}

/// Why a plane cannot hold a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneError {
    /// The texture's rows do not fit in the plane: `needed` bytes from the plane's start, `size` available.
    OutOfPlane { needed: u64, size: u64 },
    /// The texture is wider or taller than the plane.
    TooLarge,
    /// A row of the texture is longer than the plane's row pitch.
    RowTooLong,
}

/// The byte ranges, relative to the start of the surface's memory, of the `height` rows of a `width`-texel texture
/// with `bytes_per_texel` bytes per texel over `plane`. Every range lies inside the plane.
pub fn plane_rows(
    plane: &SurfacePlane,
    bytes_per_texel: u64,
    width: u64,
    height: u64,
) -> Result<Vec<std::ops::Range<u64>>, PlaneError> {
    if width > plane.width || height > plane.height {
        return Err(PlaneError::TooLarge);
    }
    let row = width.checked_mul(bytes_per_texel).ok_or(PlaneError::RowTooLong)?;
    if row > plane.bytes_per_row {
        return Err(PlaneError::RowTooLong);
    }
    if height == 0 {
        return Ok(Vec::new());
    }
    let needed = (height - 1) * plane.bytes_per_row + row;
    if needed > plane.size {
        return Err(PlaneError::OutOfPlane { needed, size: plane.size });
    }
    Ok((0..height)
        .map(|y| plane.offset + y * plane.bytes_per_row..plane.offset + y * plane.bytes_per_row + row)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_orders() {
        let row = [10u8, 20, 30, 40, 50, 60, 70, 80];
        assert_eq!(Order422::LumaFirst.luma(&row), vec![10, 30, 50, 70]);
        assert_eq!(Order422::LumaFirst.chroma(&row), vec![(20, 40), (60, 80)]);
        assert_eq!(Order422::ChromaFirst.luma(&row), vec![20, 40, 60, 80]);
        assert_eq!(Order422::ChromaFirst.chroma(&row), vec![(10, 30), (50, 70)]);
    }

    /// The luma of the reference cases: the harness's ideal values are `Y / 255` of the bytes in Metal's order, and the
    /// device's luma reads equal `Y / 255` correctly rounded.
    #[test]
    fn luma_matches_real_metal() {
        use crate::format::packed::unorm_to_f32;
        use crate::format::testdata::{Case, hex_bytes};
        for (name, value) in [("GBGR422", 240), ("BGRG422", 241)] {
            let case = Case::load(&format!("format.{name}.upload-read"));
            let input = case.bytes("input");
            assert_eq!(case.bytes("readback"), input);
            let luma = order_422(value).unwrap().luma(&input);
            let ideal: Vec<f64> = hex_bytes(
                case.record("read.luma")
                    .and_then(|r| r.get("rule"))
                    .and_then(|r| r.get("ideal"))
                    .and_then(|v| v.str())
                    .unwrap(),
            )
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
            assert_eq!(ideal, luma.iter().map(|&y| y as f64 / 255.0).collect::<Vec<_>>(), "{name}");
            let read: Vec<u32> = case.f32s("read.luma").iter().map(|v| v.to_bits()).collect();
            assert_eq!(read, luma.iter().map(|&y| unorm_to_f32(y as u32, 8).to_bits()).collect::<Vec<_>>(), "{name}");
        }
    }

    #[test]
    fn plane_bounds() {
        let plane = SurfacePlane {
            width: 4,
            height: 2,
            bytes_per_row: 16,
            bytes_per_element: 2,
            element_width: 1,
            element_height: 1,
            offset: 64,
            size: 32,
        };
        assert_eq!(plane_rows(&plane, 2, 4, 2), Ok(vec![64..72, 80..88]));
        assert_eq!(plane_rows(&plane, 2, 5, 2), Err(PlaneError::TooLarge));
        assert_eq!(plane_rows(&plane, 8, 4, 1), Err(PlaneError::RowTooLong));
        let short = SurfacePlane { size: 20, ..plane };
        assert_eq!(plane_rows(&short, 2, 4, 2), Err(PlaneError::OutOfPlane { needed: 24, size: 20 }));
    }
}
