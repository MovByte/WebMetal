//! Bit layouts and exact conversions of the Metal formats the executor stores in another WebGPU format: the packed
//! 16-bit formats, `A8Unorm`, `BGR10A2Unorm` and the extended-range (XR) formats.
//!
//! The layouts are the ones the Metal reference harness measured on the device and records under exact rules
//! (`tests/reference/src/cases/Formats.m`, "Bit positions of the packed formats were confirmed on the device"): they
//! read the format name from the least significant bit. For the XR formats Apple's documentation states the same
//! (`tables/formats.toml`, question `6.3-xr-encoding`).
//!
//! The conversions are the Metal Shading Language rules (MSL 4.1 section 8.7.1): reading an n-bit unorm code `c`
//! gives `c / (2^n - 1)`, and writing a float `f` stores `intRTNE(clamp(f × (2^n - 1), 0, 2^n - 1))`, NaN storing 0.
//! [`unorm_to_f32`] returns the correctly rounded binary32 value, which is within the 1.5 ulp the specification allows
//! and exact at the corner cases.

/// A color component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    R,
    G,
    B,
    A,
}

/// One channel of a packed texel: `bits` bits starting at bit `offset` of the little-endian texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Channel {
    pub component: Component,
    pub offset: u32,
    pub bits: u32,
}

/// The layout of a packed texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Bytes per texel.
    pub bytes: usize,
    pub channels: &'static [Channel],
}

const fn ch(component: Component, offset: u32, bits: u32) -> Channel {
    Channel { component, offset, bits }
}

use Component::{A, B, G, R};

/// `A8Unorm` (value 1).
pub const A8_UNORM: Layout = Layout { bytes: 1, channels: &[ch(A, 0, 8)] };
/// `B5G6R5Unorm` (value 40). Alpha reads as 1.
pub const B5G6R5_UNORM: Layout = Layout { bytes: 2, channels: &[ch(B, 0, 5), ch(G, 5, 6), ch(R, 11, 5)] };
/// `A1BGR5Unorm` (value 41).
pub const A1BGR5_UNORM: Layout = Layout { bytes: 2, channels: &[ch(A, 0, 1), ch(B, 1, 5), ch(G, 6, 5), ch(R, 11, 5)] };
/// `ABGR4Unorm` (value 42).
pub const ABGR4_UNORM: Layout = Layout { bytes: 2, channels: &[ch(A, 0, 4), ch(B, 4, 4), ch(G, 8, 4), ch(R, 12, 4)] };
/// `BGR5A1Unorm` (value 43).
pub const BGR5A1_UNORM: Layout = Layout { bytes: 2, channels: &[ch(B, 0, 5), ch(G, 5, 5), ch(R, 10, 5), ch(A, 15, 1)] };
/// `BGR10A2Unorm` (value 94).
pub const BGR10A2_UNORM: Layout =
    Layout { bytes: 4, channels: &[ch(B, 0, 10), ch(G, 10, 10), ch(R, 20, 10), ch(A, 30, 2)] };
/// `BGR10_XR` and `BGR10_XR_sRGB` (values 554 and 555): bits 30 and 31 are padding. Alpha reads as 1.
pub const BGR10_XR: Layout = Layout { bytes: 4, channels: &[ch(B, 0, 10), ch(G, 10, 10), ch(R, 20, 10)] };
/// `BGRA10_XR` and `BGRA10_XR_sRGB` (values 552 and 553): each code in the 10 most significant bits of a 16-bit
/// chunk; the low 6 bits of each chunk do not contribute to the value.
pub const BGRA10_XR: Layout =
    Layout { bytes: 8, channels: &[ch(B, 6, 10), ch(G, 22, 10), ch(R, 38, 10), ch(A, 54, 10)] };

/// The layout of a Metal pixel format this module converts.
pub fn layout(value: u32) -> Option<&'static Layout> {
    Some(match value {
        1 => &A8_UNORM,
        40 => &B5G6R5_UNORM,
        41 => &A1BGR5_UNORM,
        42 => &ABGR4_UNORM,
        43 => &BGR5A1_UNORM,
        94 => &BGR10A2_UNORM,
        552 | 553 => &BGRA10_XR,
        554 | 555 => &BGR10_XR,
        _ => return None,
    })
}

fn texel_bits(texel: &[u8]) -> u64 {
    texel.iter().rev().fold(0u64, |acc, &b| (acc << 8) | b as u64)
}

/// The codes of a texel's channels, indexed R, G, B, A; `None` for a component the format does not store.
pub fn unpack(layout: &Layout, texel: &[u8]) -> [Option<u32>; 4] {
    let bits = texel_bits(&texel[..layout.bytes]);
    let mut out = [None; 4];
    for c in layout.channels {
        out[c.component as usize] = Some(((bits >> c.offset) & ((1u64 << c.bits) - 1)) as u32);
    }
    out
}

/// A texel from channel codes (indexed R, G, B, A); bits outside the channels are zero.
pub fn pack(layout: &Layout, codes: [u32; 4]) -> Vec<u8> {
    let mut bits = 0u64;
    for c in layout.channels {
        bits |= ((codes[c.component as usize] as u64) & ((1u64 << c.bits) - 1)) << c.offset;
    }
    bits.to_le_bytes()[..layout.bytes].to_vec()
}

/// The binary32 value nearest to `num / den` (ties to even), for `den` > 0 and |`num`| < 2^24 · `den`.
fn ratio_to_f32(num: i64, den: i64) -> f32 {
    if num == 0 {
        return 0.0;
    }
    let neg = num < 0;
    let (n, d) = (num.unsigned_abs() as u128, den as u128);
    // Find e with 2^23 <= n / d × 2^e < 2^24, then round the 24-bit significand.
    let mut e: i32 = 0;
    while (n << e.max(0)) >> (-e).max(0) < d << 23 {
        e += 1;
    }
    while (n << e.max(0)) >> (-e).max(0) >= d << 24 {
        e -= 1;
    }
    let (sn, sd) = if e >= 0 { (n << e, d) } else { (n, d << -e) };
    let q = sn / sd;
    let r = sn % sd;
    let q = if 2 * r > sd || (2 * r == sd && q & 1 == 1) { q + 1 } else { q };
    let v = (q as f64) * 2f64.powi(-e);
    if neg { -(v as f32) } else { v as f32 }
}

/// MSL 4.1 section 8.7.1.1: an n-bit unorm code as `c / (2^n - 1)`, correctly rounded.
pub fn unorm_to_f32(code: u32, bits: u32) -> f32 {
    ratio_to_f32(code as i64, (1i64 << bits) - 1)
}

/// MSL 4.1 section 8.7.1.2: the n-bit unorm code a float writes. NaN gives 0; the product `f × (2^n - 1)` is formed
/// exactly (binary64 holds it) and rounded to nearest even after clamping.
pub fn unorm_from_f32(f: f32, bits: u32) -> u32 {
    if f.is_nan() {
        return 0;
    }
    let max = ((1u64 << bits) - 1) as f64;
    let x = (f as f64 * max).clamp(0.0, max);
    x.round_ties_even() as u32
}

/// The value of an extended-range code: `(c - 384) / 510`, correctly rounded (Apple's documentation of
/// `MTLPixelFormatBGR10_XR`, quoted in `tables/formats.toml` question `6.3-xr-encoding`).
pub fn xr_to_f32(code: u32) -> f32 {
    ratio_to_f32(code as i64 - 384, 510)
}

/// Exchanges the 10-bit fields at bits 0 and 20 of a 32-bit texel: `BGR10A2Unorm` to the `rgb10a2unorm` layout and
/// back. Bits 10 to 19 (green) and 30 to 31 (alpha) are unchanged.
pub fn swap_red_blue_10(texel: u32) -> u32 {
    (texel & 0xc00f_fc00) | ((texel & 0x3ff) << 20) | ((texel >> 20) & 0x3ff)
}

/// Applies [`swap_red_blue_10`] to every 32-bit texel of `data` in place.
pub fn swap_red_blue_10_in_place(data: &mut [u8]) {
    let (texels, rest) = data.as_chunks_mut::<4>();
    debug_assert!(rest.is_empty(), "32-bit texels");
    for t in texels {
        *t = swap_red_blue_10(u32::from_le_bytes(*t)).to_le_bytes();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correctly_rounded_ratios() {
        // Exhaustive for the unorm widths of the formats here: the binary32 result is the nearest to c / max.
        for bits in [1u32, 4, 5, 6, 8, 10] {
            let max = (1u32 << bits) - 1;
            for c in 0..=max {
                let v = unorm_to_f32(c, bits);
                let exact = c as f64 / max as f64;
                let below = f32::from_bits(v.to_bits().wrapping_sub(1));
                let above = f32::from_bits(v.to_bits() + 1);
                if c != 0 {
                    assert!((v as f64 - exact).abs() <= (below as f64 - exact).abs(), "{c}/{max}");
                }
                assert!((v as f64 - exact).abs() <= (above as f64 - exact).abs(), "{c}/{max}");
            }
            assert_eq!(unorm_to_f32(max, bits), 1.0);
            assert_eq!(unorm_to_f32(0, bits).to_bits(), 0);
        }
        assert_eq!(xr_to_f32(384), 0.0);
        assert_eq!(xr_to_f32(894), 1.0);
    }

    #[test]
    fn write_rounding() {
        // MSL 8.7.1.2: x = min(max(f × 31, 0), 31), intRTNE(x).
        assert_eq!(unorm_from_f32(f32::NAN, 5), 0);
        assert_eq!(unorm_from_f32(-1.0, 5), 0);
        assert_eq!(unorm_from_f32(2.0, 5), 31);
        // 0.5 × 31 = 15.5 rounds to the even 16; 1.5 / 31 × 31 = 1.5 rounds to 2.
        assert_eq!(unorm_from_f32(0.5, 5), 16);
        for bits in [1u32, 4, 5, 6, 10] {
            for c in 0..(1u32 << bits) {
                assert_eq!(unorm_from_f32(unorm_to_f32(c, bits), bits), c);
            }
        }
    }

    #[test]
    fn swap_is_an_involution() {
        for v in [0u32, 0xffff_ffff, 0x3ff, 0x3ff0_0000, 0x000f_fc00, 0xc000_0000, 0x1234_5678, 0x8765_4321] {
            assert_eq!(swap_red_blue_10(swap_red_blue_10(v)), v);
            let bgr = unpack(&BGR10A2_UNORM, &v.to_le_bytes());
            let rgb = swap_red_blue_10(v);
            // In the rgb10a2unorm layout red is at bit 0 and blue at bit 20.
            assert_eq!(Some(rgb & 0x3ff), bgr[0]);
            assert_eq!(Some((rgb >> 20) & 0x3ff), bgr[2]);
            assert_eq!(Some((rgb >> 10) & 0x3ff), bgr[1]);
            assert_eq!(Some(rgb >> 30), bgr[3]);
        }
    }
}
