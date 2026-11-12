//! The texels of an emulated texture: from the bytes Metal accepts for a region to the stored WebGPU format's texels,
//! and back for readback.
//!
//! Stored layouts, little-endian like every WebGPU texel copy: the bytes unchanged for [`Upload::Bytes`], swapped
//! 32-bit texels for [`Upload::SwapRedBlue10`], and for [`Upload::Decode`] the decoded texels: four bytes per texel for
//! `rgba8unorm` and `rgba8unorm-srgb`, four 16-bit values per texel for the `rgba16uint` of linear ASTC.

use super::decode::{self, DecodeError, Texels};
use super::handling::{Emulation, Upload};
use super::packed;

/// Why texels cannot be converted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageError {
    Decode(DecodeError),
    /// The region's bytes are shorter than its rows.
    Truncated {
        needed: usize,
        got: usize,
    },
    /// Decoded storage has no inverse: readback and copies of a compressed texture use the kept blocks.
    NotInvertible,
    /// The decoded representation does not match the emulation's stored format.
    Representation,
}

/// The stored texels of a `width` × `height` region of `emulation` from Metal's bytes (`bytes_per_row` between rows
/// for uncompressed formats, between rows of blocks for block formats, and ignored for PVRTC, whose level is given
/// whole in Morton order). The result is tightly packed: `width` texels per row.
pub fn to_storage(
    emulation: &Emulation,
    width: u32,
    height: u32,
    data: &[u8],
    bytes_per_row: usize,
) -> Result<Vec<u8>, StorageError> {
    let texel = emulation.storage.caps().footprint as usize;
    match emulation.upload {
        Upload::Bytes | Upload::SwapRedBlue10 => {
            let row = width as usize * texel;
            let needed = if height == 0 { 0 } else { (height as usize - 1) * bytes_per_row + row };
            if data.len() < needed || bytes_per_row < row {
                return Err(StorageError::Truncated { needed, got: data.len() });
            }
            let mut out = Vec::with_capacity(row * height as usize);
            for y in 0..height as usize {
                out.extend_from_slice(&data[y * bytes_per_row..y * bytes_per_row + row]);
            }
            if emulation.upload == Upload::SwapRedBlue10 {
                packed::swap_red_blue_10_in_place(&mut out);
            }
            Ok(out)
        }
        Upload::Decode(scheme) => {
            match (decode::decode(scheme, width, height, data, bytes_per_row).map_err(StorageError::Decode)?, texel) {
                (Texels::Rgba8(t), 4) => Ok(t.into_iter().flatten().collect()),
                (Texels::Unorm16(t), 8) => Ok(t.into_iter().flatten().flat_map(u16::to_le_bytes).collect()),
                _ => Err(StorageError::Representation),
            }
        }
    }
}

/// Metal's bytes of a region from stored texels (tightly packed), with `bytes_per_row` between rows of the result.
pub fn from_storage(
    emulation: &Emulation,
    width: u32,
    height: u32,
    texels: &[u8],
    bytes_per_row: usize,
) -> Result<Vec<u8>, StorageError> {
    let texel = emulation.storage.caps().footprint as usize;
    if let Upload::Decode(_) = emulation.upload {
        return Err(StorageError::NotInvertible);
    }
    let row = width as usize * texel;
    let needed = row * height as usize;
    if texels.len() < needed || bytes_per_row < row {
        return Err(StorageError::Truncated { needed, got: texels.len() });
    }
    let mut out = vec![0u8; if height == 0 { 0 } else { (height as usize - 1) * bytes_per_row + row }];
    for y in 0..height as usize {
        out[y * bytes_per_row..y * bytes_per_row + row].copy_from_slice(&texels[y * row..(y + 1) * row]);
        if emulation.upload == Upload::SwapRedBlue10 {
            packed::swap_red_blue_10_in_place(&mut out[y * bytes_per_row..y * bytes_per_row + row]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::handling::{Handling, Resolution, resolve};
    use crate::format::webgpu::HostFeatures;

    fn emulation(value: u32) -> Emulation {
        match resolve(value, HostFeatures::NONE) {
            Ok(Resolution::Emulated(e)) => e,
            other => panic!("{value}: {other:?}"),
        }
    }

    #[test]
    fn bytes_round_trip_with_row_pitch() {
        // B5G6R5Unorm, 3 × 2 texels with 8 bytes between rows.
        let e = emulation(40);
        let data: Vec<u8> = (0..14).collect();
        let stored = to_storage(&e, 3, 2, &data, 8).unwrap();
        assert_eq!(stored, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]);
        let back = from_storage(&e, 3, 2, &stored, 8).unwrap();
        assert_eq!(back[..6], data[..6]);
        assert_eq!(back[8..14], data[8..14]);
    }

    #[test]
    fn bgr10a2_round_trips_through_rgb10a2() {
        let e = emulation(94);
        let data = 0x8765_4321u32.to_le_bytes();
        let stored = to_storage(&e, 1, 1, &data, 4).unwrap();
        assert_eq!(u32::from_le_bytes(stored[..].try_into().unwrap()), packed::swap_red_blue_10(0x8765_4321));
        assert_eq!(from_storage(&e, 1, 1, &stored, 4).unwrap(), data);
    }

    #[test]
    fn decoded_storage() {
        // ASTC_4x4_LDR without texture-compression-astc: rgba16uint holding C; a void-extent block decodes to its
        // color everywhere.
        let e = emulation(204);
        let mut v: u128 = 0x1fc | (0b11 << 10);
        for shift in [12, 25, 38, 51] {
            v |= 0x1fff << shift;
        }
        v |= (0x1234u128 << 64) | (0x5678u128 << 80) | (0x9abcu128 << 96) | (0xffffu128 << 112);
        let stored = to_storage(&e, 4, 4, &v.to_le_bytes(), 16).unwrap();
        assert_eq!(stored.len(), 16 * 8);
        assert_eq!(stored[..8], [0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a, 0xff, 0xff]);
        assert_eq!(from_storage(&e, 4, 4, &stored, 16), Err(StorageError::NotInvertible));
        // PVRTC is always decoded.
        assert!(matches!(crate::format::handling::handling(166), Handling::Emulated(_)));
        let pv = emulation(166);
        let stored = to_storage(&pv, 8, 8, &[0u8; 32], 0).unwrap();
        assert_eq!(stored.len(), 64 * 4);
    }
}
