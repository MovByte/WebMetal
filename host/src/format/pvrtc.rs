//! PVRTC1 decoding (2 and 4 bits per pixel), as the Khronos Data Format Specification 1.4.0 describes it (section 24,
//! "PVRTC Compressed Texture Image Formats"; the pinned text is `kdfs` in `tables/sources.toml`), with the choices the
//! specification leaves to implementations fixed by real Metal's decode (`tables/measure/pvrtc_mac.toml` and the Metal
//! reference outputs `format.PVRTC_*.blocks`).
//!
//! PVRTC is not a block format in the usual sense: a texel depends on up to four adjacent 64-bit words, which wrap
//! around the image. Decoding therefore works on a whole image (one mipmap level of one slice). Metal stores the words
//! in the reflected Morton order of section 24.2.2: `replaceRegion` takes them in that order, the reference outputs
//! record the same blocks in row-linear order through `MTLBlitOptionRowLinearPVRTC`, and [`morton_offset`] maps one to
//! the other exactly.
//!
//! The result is 8-bit RGBA per texel in raster order. For the sRGB formats the R, G and B codes are sRGB-encoded
//! (section 24.1: the EOTF applies to `R'/255` at the end of the decode); alpha is linear.

/// Bits per pixel of a PVRTC1 format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bpp {
    Two,
    Four,
}

impl Bpp {
    /// Texels covered by one word's modulation data: 8×4 for 2 bpp, 4×4 for 4 bpp (sections 24.2 and 24.3).
    pub fn block_dims(self) -> (u32, u32) {
        match self {
            Bpp::Two => (8, 4),
            Bpp::Four => (4, 4),
        }
    }
}

/// Bytes in a PVRTC1 word.
pub const WORD_BYTES: usize = 8;

/// Why an image cannot be decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PvrtcError {
    /// PVRTC1 image dimensions must be powers of two (section 24.2).
    NotPowerOfTwo { width: u32, height: u32 },
    /// The data is shorter than the image's words.
    Truncated { needed: usize, got: usize },
}

/// The words of an image in each direction: at least two (sections 24.2.2 and 24.3.1).
pub fn word_dims(bpp: Bpp, width: u32, height: u32) -> (u32, u32) {
    let (bw, bh) = bpp.block_dims();
    (width.div_ceil(bw).max(2), height.div_ceil(bh).max(2))
}

/// Bytes of PVRTC1 data for a `width` × `height` image.
pub fn data_len(bpp: Bpp, width: u32, height: u32) -> usize {
    let (ww, wh) = word_dims(bpp, width, height);
    ww as usize * wh as usize * WORD_BYTES
}

/// The word index of word `(x, y)` in reflected Morton order (section 24.2.2, `reflectedMortonOffset`). `word_width`
/// and `word_height` are powers of two of at least 2.
pub fn morton_offset(x: u32, y: u32, word_width: u32, word_height: u32) -> u32 {
    debug_assert!(x < word_width && y < word_height);
    let min_dim = word_width.min(word_height);
    let mut offset = 0;
    let mut shift = 0;
    let mut mask = 1;
    while mask < min_dim {
        offset |= ((y & mask) | ((x & mask) << 1)) << shift;
        shift += 1;
        mask <<= 1;
    }
    offset | (((x | y) >> shift) << (shift * 2))
}

/// One decoded color of a word, as ARGB 4555 (section 24.2.4): `[r, g, b, a]` with 5-bit R, G, B and 4-bit A.
fn color_4555(c: u32, is_b: bool) -> [u32; 4] {
    let rep3 = |v: u32| (v << 2) | (v >> 1);
    let rep4 = |v: u32| (v << 1) | (v >> 3);
    if c & 0x8000 != 0 {
        // Opaque (Table 189): color B is RGB 555, color A is RGB 554 (the bit below it is the modulation flag).
        if is_b {
            [(c >> 10) & 31, (c >> 5) & 31, c & 31, 15]
        } else {
            [(c >> 10) & 31, (c >> 5) & 31, rep4((c >> 1) & 15), 15]
        }
    } else {
        // Translucent: A 3 bits (zero-padded to 4), then RGB 444 for color B and RGB 443 for color A.
        let a = ((c >> 12) & 7) << 1;
        if is_b {
            [rep4((c >> 8) & 15), rep4((c >> 4) & 15), rep4(c & 15), a]
        } else {
            [rep4((c >> 8) & 15), rep4((c >> 4) & 15), rep3((c >> 1) & 7), a]
        }
    }
}

struct Image<'a> {
    bpp: Bpp,
    data: &'a [u8],
    ww: u32,
    wh: u32,
    /// Words covering the image in each direction; color reconstruction samples wrap at these counts.
    cover_w: u32,
    cover_h: u32,
    width: u32,
    height: u32,
}

impl Image<'_> {
    fn word(&self, x: u32, y: u32) -> u64 {
        let i = morton_offset(x, y, self.ww, self.wh) as usize * WORD_BYTES;
        u64::from_le_bytes(self.data[i..i + WORD_BYTES].try_into().unwrap())
    }

    /// Colors A and B of word `(x, y)` as ARGB 4555.
    fn colors(&self, x: u32, y: u32) -> [[u32; 4]; 2] {
        let w = self.word(x, y);
        [color_4555(((w >> 32) & 0xffff) as u32, false), color_4555((w >> 48) as u32, true)]
    }

    /// The modulation weight of texel `(x, y)` (Tables 190 and 191) and whether it is a punch-through texel.
    // The interpolated weights keep the specification's form, floor((a + b + 1) / 2).
    #[allow(clippy::manual_div_ceil)]
    fn modulation(&self, x: u32, y: u32) -> (u32, bool) {
        const WEIGHTS: [u32; 4] = [0, 3, 5, 8];
        let (bw, bh) = self.bpp.block_dims();
        let w = self.word(x / bw, y / bh);
        let (xo, yo) = (x % bw, y % bh);
        let flag = (w >> 32) & 1 == 1;
        match self.bpp {
            Bpp::Four => {
                let bits = ((w >> (2 * (yo * 4 + xo))) & 3) as u32;
                if flag {
                    // Punch-through mode: 00 → 0, 01 → 4, 10 → 4 with alpha 0, 11 → 8.
                    ([0, 4, 4, 8][bits as usize], bits == 2)
                } else {
                    (WEIGHTS[bits as usize], false)
                }
            }
            Bpp::Two => {
                if let Some(code) = self.stored_2bpp(x, y) {
                    return (WEIGHTS[code as usize], false);
                }
                // Interpolated texel: the mode bits of its own word decide the neighbors.
                let at = |dx: i32, dy: i32| {
                    let nx = (x as i32 + dx).rem_euclid(self.width as i32) as u32;
                    let ny = (y as i32 + dy).rem_euclid(self.height as i32) as u32;
                    WEIGHTS[self.stored_2bpp(nx, ny).expect("a neighbor of an interpolated texel is stored") as usize]
                };
                let weight = if w & 1 == 0 {
                    (at(-1, 0) + at(0, -1) + at(1, 0) + at(0, 1) + 2) / 4
                } else if (w >> 20) & 1 == 1 {
                    (at(0, -1) + at(0, 1) + 1) / 2
                } else {
                    (at(-1, 0) + at(1, 0) + 1) / 2
                };
                (weight, false)
            }
        }
    }

    /// The stored 2-bit modulation code of a 2 bpp texel, or `None` for a texel the interpolated mode does not store.
    fn stored_2bpp(&self, x: u32, y: u32) -> Option<u32> {
        let w = self.word(x / 8, y / 4);
        let (xo, yo) = (x % 8, y % 4);
        if (w >> 32) & 1 == 0 {
            // Direct encoding: one bit per texel selecting image A (00) or image B (11).
            return Some(if (w >> (yo * 8 + xo)) & 1 == 1 { 3 } else { 0 });
        }
        if (xo + yo) % 2 == 1 {
            return None;
        }
        let k = yo * 4 + xo / 2;
        let bits = ((w >> (2 * k)) & 3) as u32;
        // The texel at (0, 0), and at (4, 2) when bit 0 is set, has a single bit (the higher one), duplicated.
        let single = k == 0 || (k == 10 && w & 1 == 1);
        Some(if single { if bits & 2 != 0 { 3 } else { 0 } } else { bits })
    }

    fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        let (bw, bh) = self.bpp.block_dims();
        let (hx, hy) = (bw / 2, bh / 2);
        // Color reconstruction (Equations 10 and 12).
        let xl = (x as i32 - hx as i32).div_euclid(bw as i32);
        let yl = (y as i32 - hy as i32).div_euclid(bh as i32);
        let xr = (x as i32 - hx as i32 - bw as i32 * xl) as u32;
        let yr = (y as i32 - hy as i32 - bh as i32 * yl) as u32;
        let wx = |n: i32| n.rem_euclid(self.cover_w as i32) as u32;
        let wy = |n: i32| n.rem_euclid(self.cover_h as i32) as u32;
        let c00 = self.colors(wx(xl), wy(yl));
        let c10 = self.colors(wx(xl + 1), wy(yl));
        let c01 = self.colors(wx(xl), wy(yl + 1));
        let c11 = self.colors(wx(xl + 1), wy(yl + 1));
        let mut images = [[0u32; 4]; 2];
        for (image, out) in images.iter_mut().enumerate() {
            for ch in 0..4 {
                let c = c00[image][ch] * (bw - xr) * (bh - yr)
                    + c10[image][ch] * xr * (bh - yr)
                    + c01[image][ch] * (bw - xr) * yr
                    + c11[image][ch] * xr * yr;
                out[ch] = match (self.bpp, ch) {
                    // 5.4 (4 bpp) or 5.5 (2 bpp) fixed point to 8 bits.
                    (Bpp::Four, 0..=2) => c / 2 + c / 64,
                    (Bpp::Four, _) => c + c / 16,
                    (Bpp::Two, 0..=2) => c / 4 + c / 128,
                    (Bpp::Two, _) => c / 2 + c / 32,
                };
            }
        }
        // Modulation (Equation 11).
        let (weight, punch) = self.modulation(x, y);
        let mut out = [0u8; 4];
        for ch in 0..4 {
            out[ch] = ((images[0][ch] * (8 - weight) + images[1][ch] * weight) / 8) as u8;
        }
        if punch {
            out[3] = 0;
        }
        out
    }
}

/// Decodes a `width` × `height` PVRTC1 image stored in reflected Morton order. Returns 8-bit RGBA per texel in raster
/// order (R, G and B sRGB-encoded for the sRGB formats).
///
/// For an image one word wide or high, the color reconstruction samples wrap within the words that cover the image,
/// so the padding word is not read; this is the "modern decoder" behavior of section 24.2.3 and what real Metal
/// returns for an 8×8 2 bpp image (`tables/measure/pvrtc_mac.toml`).
pub fn decode(bpp: Bpp, width: u32, height: u32, data: &[u8]) -> Result<Vec<[u8; 4]>, PvrtcError> {
    if !width.is_power_of_two() || !height.is_power_of_two() {
        return Err(PvrtcError::NotPowerOfTwo { width, height });
    }
    let needed = data_len(bpp, width, height);
    if data.len() < needed {
        return Err(PvrtcError::Truncated { needed, got: data.len() });
    }
    let (ww, wh) = word_dims(bpp, width, height);
    let (bw, bh) = bpp.block_dims();
    let image = Image { bpp, data, ww, wh, cover_w: width.div_ceil(bw), cover_h: height.div_ceil(bh), width, height };
    let mut out = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            out.push(image.texel(x, y));
        }
    }
    Ok(out)
}

/// Decodes as [`decode`] and sets alpha to 255: Metal's `PVRTC_RGB_*` formats return alpha 1.0 for the same words
/// that the `PVRTC_RGBA_*` formats decode with alpha, and the same R, G and B (`tables/measure/pvrtc_mac.toml`).
pub fn decode_opaque(bpp: Bpp, width: u32, height: u32, data: &[u8]) -> Result<Vec<[u8; 4]>, PvrtcError> {
    let mut out = decode(bpp, width, height, data)?;
    for t in &mut out {
        t[3] = 255;
    }
    Ok(out)
}

/// Converts words from reflected Morton order (as Metal stores them) to row-linear order (as
/// `MTLBlitOptionRowLinearPVRTC` copies them), or back with `to_morton`.
pub fn reorder(bpp: Bpp, width: u32, height: u32, data: &[u8], to_morton: bool) -> Result<Vec<u8>, PvrtcError> {
    if !width.is_power_of_two() || !height.is_power_of_two() {
        return Err(PvrtcError::NotPowerOfTwo { width, height });
    }
    let needed = data_len(bpp, width, height);
    if data.len() < needed {
        return Err(PvrtcError::Truncated { needed, got: data.len() });
    }
    let (ww, wh) = word_dims(bpp, width, height);
    let mut out = vec![0u8; needed];
    for y in 0..wh {
        for x in 0..ww {
            let linear = (y * ww + x) as usize * WORD_BYTES;
            let morton = morton_offset(x, y, ww, wh) as usize * WORD_BYTES;
            let (src, dst) = if to_morton { (linear, morton) } else { (morton, linear) };
            out[dst..dst + WORD_BYTES].copy_from_slice(&data[src..src + WORD_BYTES]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton_offset_example() {
        // Table 188: X = 13, Y = 2 in a 16×4-word image is word 54.
        assert_eq!(morton_offset(13, 2, 16, 4), 54);
        // Every word of a few shapes gets a distinct offset.
        for (ww, wh) in [(2, 2), (16, 4), (4, 16), (8, 8), (2, 32)] {
            let mut seen = vec![false; (ww * wh) as usize];
            for y in 0..wh {
                for x in 0..ww {
                    let o = morton_offset(x, y, ww, wh) as usize;
                    assert!(!seen[o]);
                    seen[o] = true;
                }
            }
        }
    }

    #[test]
    fn color_expansion() {
        // Opaque color B: 1 RRRRR GGGGG BBBBB.
        assert_eq!(color_4555(0x8000 | (31 << 10) | (1 << 5) | 16, true), [31, 1, 16, 15]);
        // Opaque color A: blue has 4 bits (above the modulation flag) replicated to 5.
        assert_eq!(color_4555(0x8000 | (0b1001 << 1), false)[2], 0b10011);
        // Translucent color A: alpha 3 bits zero-padded, blue 3 bits replicated to 5.
        assert_eq!(color_4555((0b101 << 12) | (0b011 << 1), false), [0, 0, 0b01101, 0b1010]);
    }

    #[test]
    fn single_word_colors_are_exact_at_their_centers() {
        // Section 24.2.4: at (4 X + 2, 4 Y + 2) image A and B are exactly the word's colors. Fill all words with the
        // same opaque colors and modulation 0 (image A).
        let color_a: u64 = 0x8000 | (20 << 10) | (10 << 5) | (6 << 1);
        let color_b: u64 = 0x8000 | (31 << 10) | (31 << 5) | 31;
        let word = (color_b << 48) | (color_a << 32);
        let data: Vec<u8> = (0..4).flat_map(|_| word.to_le_bytes()).collect();
        let out = decode(Bpp::Four, 8, 8, &data).unwrap();
        // 5-bit 20 → 8-bit via 5.4 fixed point: C = 20 × 16 = 320 → 160 + 5 = 165; 10 → 80 + 2 = 82; blue 4-bit 6 →
        // 5-bit 01100 = 12 → 96 + 3 = 99.
        assert_eq!(out[2 * 8 + 2], [165, 82, 99, 255]);
    }
}
