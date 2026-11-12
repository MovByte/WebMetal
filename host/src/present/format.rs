//! Framebuffer pixel formats: which IOSurface pixel formats a display frame can be shown from, and how a pixel's bytes
//! become a texel of the canvas texture.
//!
//! A frame's surface names its pixel format with the four-character code `IOSurfaceGetPixelFormat` returns, from the
//! CoreVideo list (`kCVPixelFormatType_*`, `CoreVideo.framework/Headers/CVPixelBuffer.h` of the iPhoneOS SDK). The
//! canvas formats are WebGPU's supported context formats: `bgra8unorm`, `rgba8unorm` and `rgba16float` (WebGPU §21.4).
//!
//! The byte order of a code is CoreVideo's description of it (`CVPixelFormatDescriptionCreateWithPixelFormatType`),
//! measured on macOS 27.0.1 (26A434) by `measure/pixel_formats.swift`, whose output is `measure/pixel_formats.txt`:
//!
//! - `kCVPixelFormatType_32BGRA` (`'BGRA'`): `CGBitmapInfo` 8196 (0x2004: `kCGBitmapByteOrder32Little | kCGImageAlphaFirst`),
//!   black block `00 00 00 ff`: the bytes of a pixel are blue, green, red, alpha, the order of `bgra8unorm`.
//! - `kCVPixelFormatType_32ARGB` (0x20): `CGBitmapInfo` 16388 (0x4004: `kCGBitmapByteOrder32Big | kCGImageAlphaFirst`), black
//!   block `ff 00 00 00`: alpha, red, green, blue, the reverse of `bgra8unorm`'s order.
//!
//! Both are shown exactly: the canvas texel holds the pixel's four 8-bit codes, unchanged, in the canvas format's
//! order, and WebGPU's texel copies keep every bit of the non-packed unorm formats (WebGPU #texel-copies). Every other
//! code is refused, with the reason below; none is approximated.

use crate::format::handling::Refusal;
use crate::gpu::idl::TextureFormat;

/// The four-character code of `chars`, as `IOSurfaceGetPixelFormat` returns it (the first character in the most
/// significant byte).
pub const fn fourcc(chars: &[u8; 4]) -> u32 {
    ((chars[0] as u32) << 24) | ((chars[1] as u32) << 16) | ((chars[2] as u32) << 8) | chars[3] as u32
}

/// `kCVPixelFormatType_*` values (`CVPixelBuffer.h`, iPhoneOS SDK).
pub mod code {
    use super::fourcc;

    /// `kCVPixelFormatType_32ARGB`: "32 bit ARGB".
    pub const ARGB_32: u32 = 0x0000_0020;
    /// `kCVPixelFormatType_32BGRA`: "32 bit BGRA".
    pub const BGRA_32: u32 = fourcc(b"BGRA");
    /// `kCVPixelFormatType_32ABGR`: "32 bit ABGR".
    pub const ABGR_32: u32 = fourcc(b"ABGR");
    /// `kCVPixelFormatType_32RGBA`: "32 bit RGBA".
    pub const RGBA_32: u32 = fourcc(b"RGBA");
    /// `kCVPixelFormatType_30RGB`: "30 bit RGB, 10-bit big-endian samples, 2 unused padding bits (at least
    /// significant end)".
    pub const RGB_30: u32 = fourcc(b"R10k");
    /// `kCVPixelFormatType_30RGB_r210`: "30 bit RGB, 10-bit big-endian samples, 2 unused padding bits (at most
    /// significant end), video-range (64-940)".
    pub const RGB_30_R210: u32 = fourcc(b"r210");
    /// `kCVPixelFormatType_ARGB2101010LEPacked`: "little-endian ARGB2101010 full-range ARGB".
    pub const ARGB_2101010_LE_PACKED: u32 = fourcc(b"l10r");
    /// `kCVPixelFormatType_30RGBLEPackedWideGamut`: "little-endian RGB101010, 2 MSB are ignored, wide-gamut
    /// (384-895)" (the layout of `MTLPixelFormatBGR10_XR`).
    pub const RGB_30_LE_PACKED_WIDE_GAMUT: u32 = fourcc(b"w30r");
    /// `kCVPixelFormatType_40ARGBLEWideGamut`: "little-endian ARGB10101010, each 10 bits in the MSBs of 16bits,
    /// wide-gamut (384-895, including alpha)" (the layout of `MTLPixelFormatBGRA10_XR`).
    pub const ARGB_40_LE_WIDE_GAMUT: u32 = fourcc(b"w40a");
    /// `kCVPixelFormatType_40ARGBLEWideGamutPremultiplied`: as [`ARGB_40_LE_WIDE_GAMUT`], alpha premultiplied.
    pub const ARGB_40_LE_WIDE_GAMUT_PREMULTIPLIED: u32 = fourcc(b"w40m");
    /// `kCVPixelFormatType_64RGBAHalf`: "64 bit RGBA IEEE half-precision float, 16-bit little-endian samples".
    pub const RGBA_64_HALF: u32 = fourcc(b"RGhA");
}

/// How the four bytes of a framebuffer pixel are arranged in the canvas texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Texels {
    /// In the same order.
    Same,
    /// In the reverse order.
    Reversed,
}

/// A framebuffer pixel format the sink shows exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FramebufferFormat {
    /// The four-character code.
    pub code: u32,
    /// The `kCVPixelFormatType_*` name.
    pub name: &'static str,
    /// Bytes of one pixel (`IOSurfaceGetBytesPerElement` of a surface in this format, whose elements are one pixel).
    pub bytes_per_pixel: u64,
    /// The canvas format that holds the pixel's codes.
    pub canvas: TextureFormat,
    pub texels: Texels,
}

impl FramebufferFormat {
    /// Arrange one row of framebuffer pixels (`row.len()` a multiple of the pixel size) as canvas texels, in place.
    pub fn to_texels(&self, row: &mut [u8]) {
        match self.texels {
            Texels::Same => {}
            Texels::Reversed => {
                let (pixels, rest) = row.as_chunks_mut::<4>();
                debug_assert!(rest.is_empty(), "4-byte pixels");
                for p in pixels {
                    p.reverse();
                }
            }
        }
    }
}

const BGRA_32: FramebufferFormat = FramebufferFormat {
    code: code::BGRA_32,
    name: "kCVPixelFormatType_32BGRA",
    bytes_per_pixel: 4,
    canvas: TextureFormat::Bgra8unorm,
    texels: Texels::Same,
};

const ARGB_32: FramebufferFormat = FramebufferFormat {
    code: code::ARGB_32,
    name: "kCVPixelFormatType_32ARGB",
    bytes_per_pixel: 4,
    canvas: TextureFormat::Bgra8unorm,
    texels: Texels::Reversed,
};

/// The formats the sink shows.
pub const SHOWN: [FramebufferFormat; 2] = [BGRA_32, ARGB_32];

/// A 10-bit unorm framebuffer.
pub const TEN_BIT_UNORM: Refusal = Refusal {
    id: "present-10-bit-unorm",
    reason: "A 10-bit unorm code c has the value c / 1023, which binary16 holds exactly only for c = 0 and c = 1023; \
             rgba16float is the only canvas format wider than 8 bits, so no canvas texture holds the frame's values.",
    source: "WebGPU §21.4 (supported context formats); IEEE 754 binary16",
};

/// An extended-range (XR) framebuffer.
pub const EXTENDED_RANGE: Refusal = Refusal {
    id: "present-extended-range",
    reason: "An XR code c has the value (c - 384) / 510, which binary16 holds exactly only for the four codes whose \
             value is -0.5, 0, 0.5 or 1; rgba16float is the only canvas format that holds values outside [0, 1].",
    source: "tables/formats.toml question 6.3-xr-encoding; WebGPU §21.4 (supported context formats)",
};

/// A half-float framebuffer.
pub const HALF_FLOAT: Refusal = Refusal {
    id: "present-half-float",
    reason: "The display model does not state how a half-float framebuffer's values are encoded (transfer function \
             and range), and WebGPU texel copies keep only finite, non-subnormal values of float formats.",
    source: "tables/limits.toml question system-behavior-items (the framebuffer color space, owner K3); WebGPU \
             #texel-copies",
};

/// A code CoreVideo describes no layout for.
pub const UNDESCRIBED_LAYOUT: Refusal = Refusal {
    id: "present-undescribed-layout",
    reason: "CoreVideo publishes no description of this code (CVPixelFormatDescriptionCreateWithPixelFormatType \
             returns none on the reference system), so its byte order is not established.",
    source: "host/src/present/measure/pixel_formats.txt",
};

/// Any other code.
pub const NOT_A_FRAMEBUFFER_FORMAT: Refusal = Refusal {
    id: "present-pixel-format",
    reason: "The pixel format is not one the presentation sink shows exactly: only 32-bit BGRA and ARGB framebuffers \
             are shown.",
    source: "host/src/present/format.rs",
};

/// The format of a framebuffer whose surface reports `code`, or why the sink does not show it.
pub fn framebuffer_format(code: u32) -> Result<FramebufferFormat, Refusal> {
    if let Some(f) = SHOWN.iter().find(|f| f.code == code) {
        return Ok(*f);
    }
    Err(match code {
        code::ARGB_2101010_LE_PACKED | code::RGB_30 | code::RGB_30_R210 => TEN_BIT_UNORM,
        code::RGB_30_LE_PACKED_WIDE_GAMUT | code::ARGB_40_LE_WIDE_GAMUT | code::ARGB_40_LE_WIDE_GAMUT_PREMULTIPLIED => {
            EXTENDED_RANGE
        }
        code::RGBA_64_HALF => HALF_FLOAT,
        code::ABGR_32 | code::RGBA_32 => UNDESCRIBED_LAYOUT,
        _ => NOT_A_FRAMEBUFFER_FORMAT,
    })
}

/// A four-character code for messages: its characters when they are printable, else its value.
pub fn code_name(code: u32) -> String {
    let b = code.to_be_bytes();
    if b.iter().all(|c| c.is_ascii_graphic() || *c == b' ') {
        format!("'{}'", b.iter().map(|&c| c as char).collect::<String>())
    } else {
        format!("{code:#010x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether `num / den` (den > 0) is a finite binary16 value: in lowest terms the denominator is a power of two, the
    /// odd part of the numerator fits the 11-bit significand and the exponent is in range (subnormals included).
    fn binary16_exact(num: i64, den: i64) -> bool {
        if num == 0 {
            return true;
        }
        let g = gcd(num.unsigned_abs(), den as u64);
        let (mut n, d) = (num.unsigned_abs() / g, den as u64 / g);
        if !d.is_power_of_two() {
            return false;
        }
        let mut e = -(d.trailing_zeros() as i32);
        while n % 2 == 0 {
            n /= 2;
            e += 1;
        }
        // value = n × 2^e with n odd; binary16 holds m × 2^q with m < 2^11 and -24 <= q, value < 65520.
        let bits = 64 - n.leading_zeros() as i32;
        bits <= 11 && e >= -24 && e + bits <= 16
    }

    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    #[test]
    fn the_shown_formats() {
        assert_eq!(framebuffer_format(fourcc(b"BGRA")), Ok(BGRA_32));
        assert_eq!(framebuffer_format(0x20), Ok(ARGB_32));
        assert_eq!(code::BGRA_32, 0x4247_5241);
    }

    #[test]
    fn argb_pixels_reverse_into_bgra_texels() {
        let f = framebuffer_format(code::ARGB_32).unwrap();
        // Alpha, red, green, blue.
        let mut row = vec![0xa0, 0x10, 0x20, 0x30, 0xa1, 0x11, 0x21, 0x31];
        f.to_texels(&mut row);
        assert_eq!(row, vec![0x30, 0x20, 0x10, 0xa0, 0x31, 0x21, 0x11, 0xa1]);
        let g = framebuffer_format(code::BGRA_32).unwrap();
        let mut same = vec![1, 2, 3, 4];
        g.to_texels(&mut same);
        assert_eq!(same, vec![1, 2, 3, 4]);
    }

    #[test]
    fn every_other_code_is_refused() {
        for (c, id) in [
            (code::ARGB_2101010_LE_PACKED, "present-10-bit-unorm"),
            (code::RGB_30, "present-10-bit-unorm"),
            (code::RGB_30_R210, "present-10-bit-unorm"),
            (code::RGB_30_LE_PACKED_WIDE_GAMUT, "present-extended-range"),
            (code::ARGB_40_LE_WIDE_GAMUT, "present-extended-range"),
            (code::ARGB_40_LE_WIDE_GAMUT_PREMULTIPLIED, "present-extended-range"),
            (code::RGBA_64_HALF, "present-half-float"),
            (code::RGBA_32, "present-undescribed-layout"),
            (code::ABGR_32, "present-undescribed-layout"),
            (fourcc(b"420v"), "present-pixel-format"),
            (fourcc(b"&BGA"), "present-pixel-format"),
            (0, "present-pixel-format"),
        ] {
            assert_eq!(framebuffer_format(c).map_err(|r| r.id), Err(id), "{}", code_name(c));
        }
    }

    #[test]
    fn ten_bit_codes_have_two_exact_binary16_values() {
        let exact: Vec<i64> = (0..1024).filter(|&c| binary16_exact(c, 1023)).collect();
        assert_eq!(exact, vec![0, 1023], "{}", TEN_BIT_UNORM.reason);
    }

    #[test]
    fn extended_range_codes_have_four_exact_binary16_values() {
        let exact: Vec<i64> = (0..1024).filter(|&c| binary16_exact(c - 384, 510)).collect();
        // (c - 384) / 510 = -0.5, 0, 0.5, 1.
        assert_eq!(exact, vec![129, 384, 639, 894], "{}", EXTENDED_RANGE.reason);
    }

    #[test]
    fn the_exactness_check_agrees_with_known_values() {
        assert!(binary16_exact(1, 2));
        assert!(binary16_exact(2047, 2048));
        assert!(!binary16_exact(4095, 4096), "12 significant bits");
        assert!(binary16_exact(1, 1 << 24), "the smallest subnormal");
        assert!(!binary16_exact(1, 1 << 25));
        assert!(binary16_exact(65504, 1));
        assert!(!binary16_exact(65536, 1));
        assert!(!binary16_exact(1, 3));
    }

    #[test]
    fn code_names() {
        assert_eq!(code_name(code::BGRA_32), "'BGRA'");
        assert_eq!(code_name(code::ARGB_32), "0x00000020");
    }
}
