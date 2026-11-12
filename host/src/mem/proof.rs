//! Texel copy tests: does this host move a format's bits through texel copies unchanged?
//!
//! WebGPU guarantees only that "valid, finite, non-subnormal numeric values in the source have the same numeric
//! value in the destination"; subnormals may become zero, NaN and infinity "may be replaced by an indeterminate
//! value", and "packed formats and snorm formats may change bit-representation" (WebGPU section 13.3.1, quoted in
//! `tables/formats.toml` question `6.3-texel-copy-bit-exactness`). Plan section 9 (row 2) settles such latitude per
//! host with a test that proves each operation. This module is that test for texel copies: it writes every bit
//! pattern of a format whose preservation the specification leaves open into a texture (`writeTexture`), copies the
//! texture back to a buffer (`copyTextureToBuffer`), and compares. Only a host on which every such pattern survives
//! the round trip unchanged moves the format's bits through texel copies; elsewhere the transfers that need it are
//! refused.
//!
//! The patterns, by format:
//!
//! - binary16 channels (`r16float`, `rg16float`, `rgba16float`): all 65536 codes;
//! - binary32 channels (`r32float`, `rg32float`, `rgba32float`): every code with exponent 0 (zeros and subnormals) or
//!   255 (infinities and NaNs), 2^25 codes; all other codes are finite normal numbers with one representation each;
//! - snorm channels (`r8snorm`, `rg8snorm`, `rgba8snorm`, `r16snorm`, `rg16snorm`, `rgba16snorm`): all codes;
//! - `rg11b10ufloat`: every combination of the codes of each channel with exponent 0 or 31 (128 × 128 × 64 words), and
//!   every code of each channel with the others fixed;
//! - `rgb9e5ufloat`, whose values have several encodings each: the test compares values, not bits, over every
//!   (exponent, mantissa) of a channel, since what such a texture holds is read only as values (its bytes are kept in
//!   the texture's twin).
//!
//! Depth formats cannot be written by texel copies at all in WebGPU, so they are not testable here.

use crate::format::webgpu::WebGpuFormat;
use crate::gpu::idl::{
    CommandBufferDescriptor, CommandEncoderDescriptor, Extent3D, Extent3DDict, MapMode, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, TextureDescriptor, TextureUsage,
};
use crate::gpu::{Api, Buffer as _, CommandEncoder as _, Device as _, GpuFuture, MappedRange as _, Queue as _, Size64};

use super::error::MemError;
use super::memory::Memory;
use super::texture::idl_format;

/// Texels per row of a test texture: 2048 texels make rows of a multiple of 256 bytes for every footprint.
const ROW_TEXELS: u32 = 2048;
/// Bytes of texel data per round trip.
const CHUNK_BYTES: usize = 16 << 20;

/// How a format's round trip is judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    Bits,
    /// `rgb9e5ufloat` values.
    Rgb9e5Values,
}

/// The test of one format: its pattern bytes (whole texels) and how to compare them.
pub struct Patterns {
    pub format: WebGpuFormat,
    pub data: Vec<u8>,
    pub compare: Compare,
}

fn channel_codes_16(mut f: impl FnMut(u16) -> bool) -> Vec<u8> {
    (0..=u16::MAX).filter(|&c| f(c)).flat_map(u16::to_le_bytes).collect()
}

/// The patterns of `format`, or why it cannot be tested.
pub fn patterns(format: WebGpuFormat) -> Result<Patterns, String> {
    use WebGpuFormat as W;
    if format.texel_copy_exact(None) {
        // Bit-exact by the specification: nothing to test.
        return Ok(Patterns { format, data: Vec::new(), compare: Compare::Bits });
    }
    let caps = format.caps();
    let footprint = caps.footprint as usize;
    let bits = |data: Vec<u8>| Patterns { format, data, compare: Compare::Bits };
    let pad = |mut data: Vec<u8>| {
        // Whole texels: repeat the start until the last texel is complete.
        let rem = data.len() % footprint;
        if rem != 0 {
            let need = footprint - rem;
            data.extend_from_within(..need);
        }
        data
    };
    Ok(match format {
        W::R16float | W::Rg16float | W::Rgba16float | W::R16snorm | W::Rg16snorm | W::Rgba16snorm => {
            bits(pad(channel_codes_16(|_| true)))
        }
        W::R8snorm | W::Rg8snorm | W::Rgba8snorm => bits(pad((0..=255u8).collect())),
        W::R32float | W::Rg32float | W::Rgba32float => {
            // Index bits 0-22 are the mantissa, bit 23 picks exponent 0 or 255, bit 24 the sign.
            let mut v = vec![0u8; 4 << 25];
            for (i, texel) in v.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let i = i as u32;
                let code = ((i >> 24) << 31) | (((i >> 23) & 1) * (255 << 23)) | (i & 0x7f_ffff);
                *texel = code.to_le_bytes();
            }
            bits(pad(v))
        }
        W::Rg11b10ufloat => {
            let special = |mantissa_bits: u32| -> Vec<u32> {
                (0..1u32 << mantissa_bits).flat_map(|m| [m, (31 << mantissa_bits) | m]).collect()
            };
            let (s11, s10) = (special(6), special(5));
            let word = |r: u32, g: u32, b: u32| r | (g << 11) | (b << 22);
            let mut v = Vec::new();
            for &r in &s11 {
                for &g in &s11 {
                    for &b in &s10 {
                        v.extend_from_slice(&word(r, g, b).to_le_bytes());
                    }
                }
            }
            // Every code of one channel, the others at 1.0 (exponent 15, mantissa 0).
            for c in 0..2048 {
                v.extend_from_slice(&word(c, 15 << 6, 15 << 5).to_le_bytes());
                v.extend_from_slice(&word(15 << 6, c, 15 << 5).to_le_bytes());
            }
            for c in 0..1024 {
                v.extend_from_slice(&word(15 << 6, 15 << 6, c).to_le_bytes());
            }
            bits(v)
        }
        W::Rgb9e5ufloat => {
            let mut v = Vec::new();
            for e in 0..32u32 {
                for m in 0..512u32 {
                    v.extend_from_slice(&(m | (m << 9) | ((511 - m) << 18) | (e << 27)).to_le_bytes());
                }
            }
            Patterns { format, data: v, compare: Compare::Rgb9e5Values }
        }
        f if !f.caps().aspects.is_empty() => {
            return Err(format!("{}: WebGPU texel copies cannot write depth or stencil-and-depth formats", f.name()));
        }
        f if f.is_compressed() => {
            return Err(format!("{}: compressed contents are read from the texture's twin, never tested", f.name()));
        }
        f => return Err(format!("{}: no pattern set for this format", f.name())),
    })
}

/// The value of an `rgb9e5ufloat` word: three channels `mantissa * 2^(exponent - 24)` (WebGPU section 26.1.3,
/// `rgb9e5ufloat`, with N = 9 mantissa bits and B = 15).
pub fn rgb9e5_values(w: u32) -> [f64; 3] {
    let e = (w >> 27) as i32 - 15 - 9;
    let ch = |shift: u32| f64::from((w >> shift) & 511) * 2f64.powi(e);
    [ch(0), ch(9), ch(18)]
}

/// Compare a round trip; `Err` names the first pattern that changed.
pub fn judge(p: &Patterns, sent: &[u8], got: &[u8]) -> Result<(), String> {
    match p.compare {
        Compare::Bits => match sent.iter().zip(got).position(|(a, b)| a != b) {
            None => Ok(()),
            Some(i) => {
                let f = p.format.caps().footprint as usize;
                let t = i / f * f;
                Err(format!("texel {:02x?} came back as {:02x?}", &sent[t..t + f], &got[t..t + f]))
            }
        },
        Compare::Rgb9e5Values => {
            for (a, b) in sent.chunks(4).zip(got.chunks(4)) {
                let (wa, wb) = (u32::from_le_bytes(a.try_into().unwrap()), u32::from_le_bytes(b.try_into().unwrap()));
                if rgb9e5_values(wa) != rgb9e5_values(wb) {
                    return Err(format!("word {wa:#010x} came back as {wb:#010x} with other values"));
                }
            }
            Ok(())
        }
    }
}

impl<A: Api> Memory<A> {
    /// Run the texel copy test of `format` on this device. The caller records the outcome with
    /// [`Memory::record_proof`]; the future does not borrow the memory services.
    pub fn prove_texel_copies(&self, format: WebGpuFormat) -> GpuFuture<Result<Result<(), String>, MemError>> {
        let device = self.device.clone();
        let queue = self.queue.clone();
        Box::pin(async move {
            let p = match patterns(format) {
                Ok(p) => p,
                Err(why) => return Ok(Err(why)),
            };
            let footprint = p.format.caps().footprint as usize;
            let row = ROW_TEXELS as usize * footprint;
            let chunk = CHUNK_BYTES / row * row;
            for part in p.data.chunks(chunk) {
                let mut data = part.to_vec();
                data.resize(part.len().next_multiple_of(row), 0);
                let rows = (data.len() / row) as u32;
                let size =
                    Extent3D::Extent3DDict(Extent3DDict { width: ROW_TEXELS, height: rows, depth_or_array_layers: 1 });
                let mut td = TextureDescriptor::new(
                    size.clone(),
                    idl_format(format),
                    TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
                );
                td.label = format!("texel copy test {}", format.name());
                let texture = device.create_texture(&td).map_err(|e| MemError::gpu("GPUDevice.createTexture", e))?;
                let layout = TexelCopyBufferLayout {
                    offset: Size64::new_const(0),
                    bytes_per_row: Some(row as u32),
                    rows_per_image: Some(rows),
                };
                queue
                    .write_texture(&TexelCopyTextureInfo::<A>::new(&texture), &data, &layout, &size)
                    .map_err(|e| MemError::gpu("GPUQueue.writeTexture", e))?;
                let bytes = Size64::new(data.len() as u64).map_err(|e| MemError::gpu("createBuffer", e))?;
                let staging = device
                    .create_buffer(&crate::gpu::idl::BufferDescriptor::new(
                        bytes,
                        crate::gpu::idl::BufferUsage::MAP_READ | crate::gpu::idl::BufferUsage::COPY_DST,
                    ))
                    .map_err(|e| MemError::gpu("GPUDevice.createBuffer", e))?;
                let encoder = device.create_command_encoder(&CommandEncoderDescriptor::new());
                let mut dst = TexelCopyBufferInfo::<A>::new(&staging);
                dst.bytes_per_row = Some(row as u32);
                dst.rows_per_image = Some(rows);
                encoder
                    .copy_texture_to_buffer(&TexelCopyTextureInfo::<A>::new(&texture), &dst, &size)
                    .map_err(|e| MemError::gpu("GPUCommandEncoder.copyTextureToBuffer", e))?;
                queue.submit(&[&encoder.finish(&CommandBufferDescriptor::new())]);
                staging
                    .map_async(MapMode::READ, Size64::new_const(0), None)
                    .await
                    .map_err(|e| MemError::gpu("GPUBuffer.mapAsync", e))?;
                let range = staging
                    .get_mapped_range(Size64::new_const(0), None)
                    .map_err(|e| MemError::gpu("GPUBuffer.getMappedRange", e))?;
                let mut got = vec![0u8; data.len()];
                range.read(0, &mut got).map_err(|e| MemError::gpu("ArrayBuffer read", e))?;
                staging.unmap();
                crate::gpu::Texture::destroy(&texture);
                if let Err(why) = judge(&p, &data[..part.len()], &got[..part.len()]) {
                    return Ok(Err(why));
                }
            }
            Ok(Ok(()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_sets() {
        assert_eq!(patterns(WebGpuFormat::R16float).unwrap().data.len(), 65536 * 2);
        // rg16float: 65536 codes make 32768 whole texels.
        assert_eq!(patterns(WebGpuFormat::Rg16float).unwrap().data.len(), 65536 * 2);
        // rgba8snorm: 256 codes padded to 64 texels.
        assert_eq!(patterns(WebGpuFormat::Rgba8snorm).unwrap().data.len(), 256);
        let f32s = patterns(WebGpuFormat::R32float).unwrap().data;
        assert_eq!(f32s.len(), 4 << 25);
        let w = |i: usize| u32::from_le_bytes(f32s[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!((w(0), w(1 << 23), w((1 << 24) - 1), w(1 << 24)), (0, 0x7f80_0000, 0x7fff_ffff, 0x8000_0000));
        assert_eq!(patterns(WebGpuFormat::Rg11b10ufloat).unwrap().data.len(), 4 * (128 * 128 * 64 + 2048 * 2 + 1024));
        assert!(patterns(WebGpuFormat::Depth32float).is_err());
        assert!(patterns(WebGpuFormat::Astc4x4Unorm).is_err());
        assert!(patterns(WebGpuFormat::Rgba8uint).unwrap().data.is_empty());
    }

    #[test]
    fn rgb9e5_judged_by_value() {
        let p = patterns(WebGpuFormat::Rgb9e5ufloat).unwrap();
        // 1.0 as mantissa 256 at exponent 16, or mantissa 128 at exponent 17: the same value.
        let a = (256u32 | (256 << 9) | (256 << 18) | (16 << 27)).to_le_bytes();
        let b = (128u32 | (128 << 9) | (128 << 18) | (17 << 27)).to_le_bytes();
        assert_eq!(rgb9e5_values(u32::from_le_bytes(a)), [1.0, 1.0, 1.0]);
        assert!(judge(&p, &a, &b).is_ok());
        assert!(judge(&Patterns { compare: Compare::Bits, ..p }, &a, &b).is_err());
    }
}
