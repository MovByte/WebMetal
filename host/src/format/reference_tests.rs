//! The decoders against real Metal: the Metal reference outputs recorded on an Apple M4 (`tests/reference/goldens`)
//! and the PVRTC measurement of `tables/measure/pvrtc_mac.toml`.
//!
//! The reference outputs record the input blocks, `getBytes` and blit readbacks under exact rules, and the decoded values a compute
//! `read()` returns under rule `none` (Metal does not state the precision of compressed-format decoding for shader
//! reads). The exact records are asserted as such. The decoded reads are compared with the specification decode under
//! the conversion each comparison names; they describe the M4, not the A10X.

use super::decode::{Scheme, Texels, decode, scheme};
use super::pvrtc::{self, Bpp};
use super::testdata::{Case, pvrtc_measurement};

/// The sRGB EOTF of MSL 4.1 section 8.7.7 applied to an 8-bit code, in binary64.
fn srgb_to_linear(code: u8) -> f64 {
    let c = code as f64 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// The 8-bit sRGB code whose exact EOTF is nearest to `v`, and whether that choice is unambiguous (no other code is
/// within the same distance).
fn nearest_srgb_code(v: f32) -> (u8, bool) {
    let mut best = (0u8, f64::INFINITY);
    let mut second = f64::INFINITY;
    for c in 0..=255u8 {
        let d = (srgb_to_linear(c) - v as f64).abs();
        if d < best.1 {
            second = best.1;
            best = (c, d);
        } else if d < second {
            second = d;
        }
    }
    (best.0, second > best.1)
}

fn unorm8(c: u8) -> f32 {
    c as f32 / 255.0
}

/// Compares 8-bit RGBA texels with a decoded read: alpha (and, when `srgb` is false, R, G and B) as `c / 255`
/// exactly, R, G and B of sRGB formats through the code whose EOTF is nearest to the read value.
fn compare_rgba8(name: &str, texels: &[[u8; 4]], read: &[f32], srgb: bool) {
    assert_eq!(texels.len() * 4, read.len(), "{name}: texel count");
    for (i, t) in texels.iter().enumerate() {
        for ch in 0..4 {
            let v = read[i * 4 + ch];
            if srgb && ch < 3 {
                let (code, unambiguous) = nearest_srgb_code(v);
                assert!(unambiguous, "{name}: texel {i} channel {ch}: {v} is midway between two sRGB codes");
                assert_eq!(code, t[ch], "{name}: texel {i} channel {ch}: read {v}");
            } else {
                assert_eq!(v.to_bits(), unorm8(t[ch]).to_bits(), "{name}: texel {i} channel {ch}: {v} vs {}", t[ch]);
            }
        }
    }
}

/// Runs one `format.<name>.blocks` case: asserts the exact records and returns the decode and the decoded read.
fn blocks_case(name: &str) -> (Scheme, Texels, Vec<f32>) {
    let case = Case::load(&format!("format.{name}.blocks"));
    assert_eq!(case.status(), "ok", "{name}");
    let input = case.bytes("input");
    assert_eq!(case.rule("input"), "exact");
    assert_eq!(case.bytes("getbytes"), input, "{name}: getBytes returns the uploaded blocks");
    let value = super::table::METAL_FORMATS
        .iter()
        .find(|r| r.name == format!("MTLPixelFormat{name}"))
        .unwrap_or_else(|| panic!("{name}"))
        .value;
    let scheme = scheme(value).unwrap_or_else(|| panic!("{name}: no scheme"));
    let (w, h) = (case.layout_u32("input", "width"), case.layout_u32("input", "height"));
    let bpr = case.layout_u32("input", "bytes_per_row") as usize;
    if let Scheme::Pvrtc { bpp, .. } = scheme {
        assert_eq!(bpr, 0);
        // The row-linear blit readback is the Morton-ordered input reordered word by word.
        let linear = pvrtc::reorder(bpp, w, h, &input, false).unwrap();
        assert_eq!(case.rule("readback-row-linear"), "exact");
        assert_eq!(case.bytes("readback-row-linear"), linear, "{name}: row-linear order");
        assert_eq!(pvrtc::reorder(bpp, w, h, &linear, true).unwrap(), input);
    } else {
        assert_eq!(case.bytes("readback"), input, "{name}: blit readback returns the blocks");
    }
    let texels = decode(scheme, w, h, &input, bpr).unwrap();
    assert_eq!(case.rule("read.decoded"), "none");
    (scheme, texels, case.f32s("read.decoded"))
}

const ETC: [&str; 6] =
    ["ETC2_RGB8", "ETC2_RGB8_sRGB", "ETC2_RGB8A1", "ETC2_RGB8A1_sRGB", "EAC_RGBA8", "EAC_RGBA8_sRGB"];

#[test]
fn etc2_matches_real_metal() {
    for name in ETC {
        let (scheme, texels, read) = blocks_case(name);
        let Texels::Rgba8(t) = texels else { panic!() };
        compare_rgba8(name, &t, &read, scheme.srgb());
    }
}

/// EAC R11 and RG11: the M4 returns the 11-bit value `x` as `x / 2047` (unsigned) and `x / 1023` (signed), correctly
/// rounded to binary32, not the 16-bit bit-replicated extension. Missing channels read as 0 and alpha as 1.
#[test]
fn eac_11_matches_real_metal_as_eleven_bit_values() {
    for (name, channels) in [("EAC_R11Unorm", 1), ("EAC_R11Snorm", 1), ("EAC_RG11Unorm", 2), ("EAC_RG11Snorm", 2)] {
        let (_, texels, read) = blocks_case(name);
        let values: Vec<[f32; 2]> = match texels {
            Texels::Unsigned11(t) => t.iter().map(|v| v.map(|x| x as f32 / 2047.0)).collect(),
            Texels::Signed11(t) => t.iter().map(|v| v.map(|x| x as f32 / 1023.0)).collect(),
            _ => panic!(),
        };
        for (i, v) in values.iter().enumerate() {
            for ch in 0..4 {
                let expect = match ch {
                    0 => v[0],
                    1 if channels == 2 => v[1],
                    3 => 1.0,
                    _ => 0.0,
                };
                assert_eq!(read[i * 4 + ch].to_bits(), expect.to_bits(), "{name}: texel {i} channel {ch}");
            }
        }
    }
}

fn astc_names() -> Vec<String> {
    super::astc::FOOTPRINTS
        .iter()
        .flat_map(|(w, h)| [format!("ASTC_{w}x{h}_LDR"), format!("ASTC_{w}x{h}_sRGB")])
        .collect()
}

/// `k / 4095` correctly rounded to binary32: the quotient is computed in binary64 and checked against its neighbors
/// exactly, since rounding twice could differ from rounding once.
fn unorm12(k: u16) -> f32 {
    let v = (k as f64 / 4095.0) as f32;
    let err = |c: f32| ((c as f64) * 4095.0 - k as f64).abs();
    let (down, up) = (f32::from_bits(v.to_bits().saturating_sub(1)), f32::from_bits(v.to_bits() + 1));
    assert!(err(v) <= err(down) && err(v) <= err(up));
    v
}

/// ASTC LDR and sRGB. The M4's linear reads are `(C >> 4) / 4095` of the specification's UNORM16 result `C` for every
/// texel, error blocks included (magenta); its sRGB reads equal the 8-bit sRGB-mode results through the EOTF.
#[test]
fn astc_matches_real_metal() {
    let mut valid_texels = 0;
    for name in astc_names() {
        let (scheme, texels, read) = blocks_case(&name);
        match texels {
            Texels::Unorm16(t) => {
                for (i, texel) in t.iter().enumerate() {
                    if *texel != super::astc::ERROR_UNORM16 {
                        valid_texels += 1;
                    }
                    for ch in 0..4 {
                        let expect = unorm12(super::astc::to_unorm12(texel[ch]));
                        assert_eq!(read[i * 4 + ch].to_bits(), expect.to_bits(), "{name}: texel {i} channel {ch}");
                    }
                }
            }
            Texels::Rgba8(t) => compare_rgba8(&name, &t, &read, scheme.srgb()),
            _ => panic!(),
        }
    }
    // The reference cases' pseudo-random blocks are mostly illegal; the comparison is only meaningful if some decode to colors.
    assert!(valid_texels > 0);
}

#[test]
fn pvrtc_matches_real_metal() {
    for name in [
        "PVRTC_RGB_2BPP",
        "PVRTC_RGB_2BPP_sRGB",
        "PVRTC_RGB_4BPP",
        "PVRTC_RGB_4BPP_sRGB",
        "PVRTC_RGBA_2BPP",
        "PVRTC_RGBA_2BPP_sRGB",
        "PVRTC_RGBA_4BPP",
        "PVRTC_RGBA_4BPP_sRGB",
    ] {
        let (scheme, texels, read) = blocks_case(name);
        let Texels::Rgba8(t) = texels else { panic!() };
        compare_rgba8(name, &t, &read, scheme.srgb());
    }
}

/// `tables/measure/pvrtc_mac.toml`: 8×8 images (one 2 bpp word wide) filled with block `i` byte `j` = `(i * 37 + j * 11) & 0xff`.
#[test]
fn pvrtc_matches_the_mac_measurement() {
    for (id, bpp, alpha) in [
        ("MTLPixelFormatPVRTC_RGB_2BPP", Bpp::Two, false),
        ("MTLPixelFormatPVRTC_RGB_4BPP", Bpp::Four, false),
        ("MTLPixelFormatPVRTC_RGBA_2BPP", Bpp::Two, true),
        ("MTLPixelFormatPVRTC_RGBA_4BPP", Bpp::Four, true),
    ] {
        let words = 4;
        let data: Vec<u8> = (0..words).flat_map(|i| (0..8).map(move |j| ((i * 37 + j * 11) & 0xff) as u8)).collect();
        let t = if alpha { pvrtc::decode(bpp, 8, 8, &data) } else { pvrtc::decode_opaque(bpp, 8, 8, &data) }.unwrap();
        let read = pvrtc_measurement(id);
        compare_rgba8(id, &t, &read, false);
    }
}

/// FNV-1a 64 of a byte string, the digest the reference vectors use.
pub(super) fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// The digests of one ASTC block's decode: UNORM16 linear results (little-endian) and sRGB-mode bytes.
pub(super) fn astc_digests(block: &[u8; 16], w: u32, h: u32) -> (u64, u64) {
    use super::astc::{ERROR_SRGB8, ERROR_UNORM16, OperationMode, decode_block};
    let lin = decode_block(block, w, h, OperationMode::LinearLdr).unwrap();
    let srgb = decode_block(block, w, h, OperationMode::Srgb).unwrap();
    let lin: Vec<u8> = lin.iter().flat_map(|t| t.unwrap_or(ERROR_UNORM16)).flat_map(|c| c.to_le_bytes()).collect();
    let srgb: Vec<u8> = srgb.iter().flat_map(|t| t.map_or(ERROR_SRGB8, |c| c.map(super::astc::to_srgb8))).collect();
    (fnv64(&lin), fnv64(&srgb))
}

/// Every line of a reference file `<w> <h> <block hex> <linear digest> <sRGB digest>` produced by the Khronos
/// VK-GL-CTS ASTC decoder (`tcuAstcUtil.cpp`, `decompressBlock`): `vectors/gen_vectors.py --full` writes the whole
/// generated corpus. Run with `WEBMETAL_ASTC_REFERENCE=<file>`.
#[test]
#[ignore = "needs a reference file generated from VK-GL-CTS"]
fn astc_matches_khronos_reference_file() {
    let path = std::env::var("WEBMETAL_ASTC_REFERENCE").expect("WEBMETAL_ASTC_REFERENCE");
    let text = std::fs::read_to_string(path).unwrap();
    let (mut n, mut bad) = (0, 0);
    for line in text.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let (w, h): (u32, u32) = (f[0].parse().unwrap(), f[1].parse().unwrap());
        let block: [u8; 16] = super::testdata::hex_bytes(f[2]).try_into().unwrap();
        let expect = (u64::from_str_radix(f[3], 16).unwrap(), u64::from_str_radix(f[4], 16).unwrap());
        n += 1;
        if astc_digests(&block, w, h) != expect {
            bad += 1;
            if bad <= 20 {
                eprintln!("mismatch: {line}");
            }
        }
    }
    eprintln!("{n} blocks, {bad} mismatches");
    assert_eq!(bad, 0);
}

/// The digest of one ETC2 or EAC block decode, as the reference files record it: RGBA bytes in raster order for
/// `rgb8`, `rgb8a1` and `rgba8`; 16-bit little-endian 11-bit values for `r11u` and `r11s`.
pub(super) fn etc_digest(kind: &str, block: &[u8]) -> u64 {
    use super::etc2;
    match kind {
        "rgb8" => fnv64(etc2::decode_rgb8(block.try_into().unwrap()).as_flattened()),
        "rgb8a1" => fnv64(etc2::decode_rgb8a1(block.try_into().unwrap()).as_flattened()),
        "rgba8" => fnv64(etc2::decode_rgba8(block.try_into().unwrap()).as_flattened()),
        "r11u" => {
            let v = etc2::decode_r11_unsigned(block.try_into().unwrap());
            fnv64(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
        }
        "r11s" => {
            let v = etc2::decode_r11_signed(block.try_into().unwrap());
            fnv64(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
        }
        _ => panic!("{kind}"),
    }
}

/// Every line of a reference file `<kind> <block hex> <digest>` produced by the Khronos VK-GL-CTS ETC2 and EAC
/// decoders (`tcuCompressedTexture.cpp`): `vectors/gen_vectors.py --full` writes one. Run with
/// `WEBMETAL_ETC_REFERENCE=<file>`.
#[test]
#[ignore = "needs a reference file generated from VK-GL-CTS"]
fn etc_matches_khronos_reference_file() {
    let path = std::env::var("WEBMETAL_ETC_REFERENCE").expect("WEBMETAL_ETC_REFERENCE");
    let text = std::fs::read_to_string(path).unwrap();
    let (mut n, mut bad) = (0, 0);
    for line in text.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        n += 1;
        if etc_digest(f[0], &super::testdata::hex_bytes(f[1])) != u64::from_str_radix(f[2], 16).unwrap() {
            bad += 1;
            if bad <= 20 {
                eprintln!("mismatch: {line}");
            }
        }
    }
    eprintln!("{n} blocks, {bad} mismatches");
    assert_eq!(bad, 0);
}

fn vector_lines(name: &str) -> Vec<String> {
    let path = super::testdata::repo_path(&format!("host/src/format/vectors/{name}"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).map(str::to_owned).collect()
}

/// The ASTC vectors of `vectors/astc_cts.txt`: CTS block test cases, random valid and random blocks of every LDR
/// footprint, decoded in both operation modes, equal the Khronos reference decoder's results.
#[test]
fn astc_matches_khronos_vectors() {
    let lines = vector_lines("astc_cts.txt");
    let mut footprints = std::collections::BTreeSet::new();
    for line in &lines {
        let f: Vec<&str> = line.split(' ').collect();
        let (w, h): (u32, u32) = (f[0].parse().unwrap(), f[1].parse().unwrap());
        footprints.insert((w, h));
        let block: [u8; 16] = super::testdata::hex_bytes(f[2]).try_into().unwrap();
        let expect = (u64::from_str_radix(f[3], 16).unwrap(), u64::from_str_radix(f[4], 16).unwrap());
        assert_eq!(astc_digests(&block, w, h), expect, "{line}");
    }
    assert_eq!(footprints.len(), super::astc::FOOTPRINTS.len());
}

/// The xorshift32 byte generator of tests/reference/FORMAT.md ("Generated inputs").
fn xorshift_bytes(seed: u32, n: usize) -> Vec<u8> {
    let mut s = if seed == 0 { 0x9e3779b9 } else { seed };
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as u8
        })
        .collect()
}

/// The ETC2 and EAC vectors of `vectors/etc_cts.txt`: pseudo-random blocks of each kind decode to the Khronos
/// reference decoder's results.
#[test]
fn etc_matches_khronos_vectors() {
    let lines = vector_lines("etc_cts.txt");
    assert_eq!(lines.len(), 5);
    for line in lines {
        let f: Vec<&str> = line.split(' ').collect();
        let kind = f[0];
        let seed = u32::from_str_radix(f[1].trim_start_matches("0x"), 16).unwrap();
        let count: usize = f[2].parse().unwrap();
        let size = if kind == "rgba8" { 16 } else { 8 };
        let data = xorshift_bytes(seed, count * size);
        let digests: Vec<u8> = data.chunks(size).flat_map(|b| etc_digest(kind, b).to_le_bytes()).collect();
        assert_eq!(fnv64(&digests), u64::from_str_radix(f[3], 16).unwrap(), "{kind}");
    }
}

/// The ideal values of a bounded record (binary64, little-endian hexadecimal).
fn ideal_values(case: &Case, name: &str) -> Vec<f64> {
    let hex = case.record(name).and_then(|r| r.get("rule")).and_then(|r| r.get("ideal")).and_then(|v| v.str()).unwrap();
    super::testdata::hex_bytes(hex).chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Packed and alpha-only formats: the layouts of `packed` give the codes the reference harness read (its ideal values
/// `c / (2^n - 1)` are equal), the device's reads equal [`super::packed::unorm_to_f32`] of those codes, and quantizing
/// the exactly representable render inputs reproduces the bytes the device stored.
#[test]
fn packed_layouts_and_conversions_match_real_metal() {
    use super::packed::{layout, pack, unorm_from_f32, unorm_to_f32, unpack};
    for (name, value) in [
        ("A8Unorm", 1),
        ("B5G6R5Unorm", 40),
        ("A1BGR5Unorm", 41),
        ("ABGR4Unorm", 42),
        ("BGR5A1Unorm", 43),
        ("BGR10A2Unorm", 94),
    ] {
        let l = layout(value).unwrap();
        let case = Case::load(&format!("format.{name}.upload-read"));
        assert_eq!(case.status(), "ok");
        let input = case.bytes("input");
        assert_eq!(case.bytes("readback"), input, "{name}");
        let (mut ideal, mut ours) = (Vec::new(), Vec::new());
        for texel in input.chunks(l.bytes) {
            let codes = unpack(l, texel);
            for (comp, code) in codes.iter().enumerate() {
                if let Some(c) = code {
                    let bits = l.channels.iter().find(|ch| ch.component as usize == comp).unwrap().bits;
                    ideal.push(*c as f64 / ((1u64 << bits) - 1) as f64);
                    ours.push(unorm_to_f32(*c, bits));
                }
            }
        }
        assert_eq!(ideal_values(&case, "read.normalized"), ideal, "{name}: codes");
        let read = case.f32s("read.normalized");
        assert_eq!(
            read.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ours.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name}: reads"
        );

        let render = Case::load(&format!("format.{name}.render"));
        assert_eq!(render.rule("render.representable"), "exact");
        let floats = render.f32s("render.input");
        let stored = render.bytes("render.representable");
        let texels = stored.len() / l.bytes;
        for t in 0..texels {
            let mut codes = [0u32; 4];
            for c in l.channels {
                codes[c.component as usize] = unorm_from_f32(floats[t * 4 + c.component as usize], c.bits);
            }
            assert_eq!(pack(l, codes), stored[t * l.bytes..(t + 1) * l.bytes], "{name}: render texel {t}");
        }
    }
}

/// The linear XR formats read as `(c - 384) / 510`; the alpha of `BGRA10_XR` is clamped to [0, 1] in sampling
/// (Apple's documentation, quoted in `tables/formats.toml` question `6.3-xr-encoding`).
#[test]
fn xr_reads_match_real_metal() {
    use super::packed::{layout, unpack, xr_to_f32};
    for (name, value) in [("BGR10_XR", 554), ("BGRA10_XR", 552)] {
        let l = layout(value).unwrap();
        let case = Case::load(&format!("format.{name}.upload-read"));
        let input = case.bytes("input");
        assert_eq!(case.bytes("readback"), input, "{name}");
        let mut ours = Vec::new();
        for texel in input.chunks(l.bytes) {
            for (comp, code) in unpack(l, texel).iter().enumerate() {
                if let Some(c) = code {
                    let v = xr_to_f32(*c);
                    ours.push(if comp == 3 { v.clamp(0.0, 1.0) } else { v });
                }
            }
        }
        let read = case.f32s("read.xr");
        assert_eq!(
            read.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ours.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name}"
        );
    }
}
