//! ETC2 and EAC block decoding, as the Khronos Data Format Specification 1.4.0 defines it (section 22, "ETC2
//! Compressed Texture Image Formats"; the pinned text is `kdfs` in `tables/sources.toml`).
//!
//! Every format decodes 4×4 texel blocks. Texels are returned in raster order: index `y * 4 + x`, with `x` across and
//! `y` down the block. The specification names the texels of a block `a` to `p` down the columns (Figure 53), so the
//! texel at `(x, y)` is letter number `x * 4 + y`; its color index bits are bit `x * 4 + y` (low bit) and bit
//! `16 + x * 4 + y` (high bit) of the low word (Table 141 f), and its alpha or EAC index occupies bits
//! `47 - 3 * (x * 4 + y)` down to `45 - 3 * (x * 4 + y)` (Table 145 b).
//!
//! The results are the specification's exact integer values: 8-bit color and alpha for the RGB, RGBA and punchthrough
//! formats ("these values all need to be divided by 255"), and the 11-bit fixed-point value for the EAC formats
//! ("This is the exact representation for the decoded value").

/// Block footprint of every ETC2 and EAC format, in texels.
pub const BLOCK_DIM: u32 = 4;

/// Table 142 (and Table 148): the intensity modifier pairs `(a, b)` of the individual and differential modes; the four
/// modifiers of a table are `-b, -a, +a, +b`.
const MODIFIERS: [[i32; 2]; 8] = [[2, 8], [5, 17], [9, 29], [13, 42], [18, 60], [24, 80], [33, 106], [47, 183]];

/// Table 144: the distances of the 'T' and 'H' modes.
const DISTANCES: [i32; 8] = [3, 6, 11, 16, 23, 32, 41, 64];

/// Table 146: the modifier tables of the RGBA ETC2 alpha part and of the EAC formats.
const ALPHA_MODIFIERS: [[i32; 8]; 16] = [
    [-3, -6, -9, -15, 2, 5, 8, 14],
    [-3, -7, -10, -13, 2, 6, 9, 12],
    [-2, -5, -8, -13, 1, 4, 7, 12],
    [-2, -4, -6, -13, 1, 3, 5, 12],
    [-3, -6, -8, -12, 2, 5, 7, 11],
    [-3, -7, -9, -11, 2, 6, 8, 10],
    [-4, -7, -8, -11, 3, 6, 7, 10],
    [-3, -5, -8, -11, 2, 4, 7, 10],
    [-2, -6, -8, -10, 1, 5, 7, 9],
    [-2, -5, -8, -10, 1, 4, 7, 9],
    [-2, -4, -8, -10, 1, 3, 7, 9],
    [-2, -5, -7, -10, 1, 4, 6, 9],
    [-3, -4, -7, -10, 2, 3, 6, 9],
    [-1, -2, -3, -10, 0, 1, 2, 9],
    [-4, -6, -8, -9, 3, 5, 7, 8],
    [-3, -5, -7, -9, 2, 4, 6, 8],
];

/// A decoded block of 8-bit RGBA texels in raster order.
pub type Rgba8Block = [[u8; 4]; 16];

/// The 64-bit word of a block half: bytes `q0..q7` with `q0` most significant (section 22, "int64bit").
fn word(bytes: &[u8]) -> u64 {
    u64::from_be_bytes(bytes[..8].try_into().expect("an ETC block half is 8 bytes"))
}

fn bits(w: u64, hi: u32, lo: u32) -> u32 {
    ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32
}

/// A 3-bit two's-complement field (R_d, G_d, B_d).
fn signed3(v: u32) -> i32 {
    if v & 4 != 0 { v as i32 - 8 } else { v as i32 }
}

fn extend4(v: u32) -> i32 {
    (v * 17) as i32
}

fn extend5(v: u32) -> i32 {
    ((v << 3) | (v >> 2)) as i32
}

fn extend6(v: u32) -> i32 {
    ((v << 2) | (v >> 4)) as i32
}

fn extend7(v: u32) -> i32 {
    ((v << 1) | (v >> 6)) as i32
}

fn clamp255(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// The 2-bit color index of texel `(x, y)`: (MSB, LSB) as (bit 16 + i, bit i), i = x * 4 + y.
fn color_index(w: u64, x: usize, y: usize) -> u32 {
    let i = (x * 4 + y) as u32;
    (bits(w, 16 + i, 16 + i) << 1) | bits(w, i, i)
}

/// The 3-bit alpha or EAC index of texel `(x, y)`.
fn alpha_index(w: u64, x: usize, y: usize) -> u32 {
    let i = (x * 4 + y) as u32;
    bits(w, 47 - 3 * i, 45 - 3 * i)
}

/// Table 143: modifier for a pixel index (MSB, LSB) = 00 → +a, 01 → +b, 10 → -a, 11 → -b.
fn modifier(table: u32, index: u32) -> i32 {
    let [a, b] = MODIFIERS[table as usize];
    match index {
        0 => a,
        1 => b,
        2 => -a,
        _ => -b,
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Individual,
    Differential,
    T,
    H,
    Planar,
}

/// Mode selection of Table 141 a (and Table 147 a). `bit33` is the differential bit D for RGB ETC2 and the opaque bit
/// for the punchthrough format, which has no individual mode.
fn mode(w: u64, punchthrough: bool) -> Mode {
    if !punchthrough && bits(w, 33, 33) == 0 {
        return Mode::Individual;
    }
    let r = bits(w, 63, 59) as i32 + signed3(bits(w, 58, 56));
    let g = bits(w, 55, 51) as i32 + signed3(bits(w, 50, 48));
    let b = bits(w, 47, 43) as i32 + signed3(bits(w, 42, 40));
    if !(0..=31).contains(&r) {
        Mode::T
    } else if !(0..=31).contains(&g) {
        Mode::H
    } else if !(0..=31).contains(&b) {
        Mode::Planar
    } else {
        Mode::Differential
    }
}

/// Decodes the color part of a block: RGB ETC2 when `punchthrough` is false (alpha 255), RGB ETC2 with punchthrough
/// alpha when it is true.
fn decode_color(w: u64, punchthrough: bool) -> Rgba8Block {
    let mut out = [[0u8; 4]; 16];
    // The punchthrough format's bit 33 is the opaque flag; RGB ETC2 blocks are always opaque.
    let opaque = !punchthrough || bits(w, 33, 33) == 1;
    match mode(w, punchthrough) {
        m @ (Mode::Individual | Mode::Differential) => {
            let (base1, base2) = if let Mode::Individual = m {
                // Table 141 b.
                (
                    [extend4(bits(w, 63, 60)), extend4(bits(w, 55, 52)), extend4(bits(w, 47, 44))],
                    [extend4(bits(w, 59, 56)), extend4(bits(w, 51, 48)), extend4(bits(w, 43, 40))],
                )
            } else {
                // Table 141 c: the second base color is the first plus the 3-bit offsets, in five-bit form.
                let (r, g, b) = (bits(w, 63, 59), bits(w, 55, 51), bits(w, 47, 43));
                let r2 = (r as i32 + signed3(bits(w, 58, 56))) as u32;
                let g2 = (g as i32 + signed3(bits(w, 50, 48))) as u32;
                let b2 = (b as i32 + signed3(bits(w, 42, 40))) as u32;
                ([extend5(r), extend5(g), extend5(b)], [extend5(r2), extend5(g2), extend5(b2)])
            };
            let tables = [bits(w, 39, 37), bits(w, 36, 34)];
            let flip = bits(w, 32, 32) == 1;
            for y in 0..4 {
                for x in 0..4 {
                    // Flip bit 0: two 2×4 subblocks side by side; 1: two 4×2 subblocks on top of each other.
                    let second = if flip { y >= 2 } else { x >= 2 };
                    let base = if second { base2 } else { base1 };
                    let index = color_index(w, x, y);
                    let texel = &mut out[y * 4 + x];
                    if !opaque && index == 2 {
                        // Table 151 and the punchthrough rule: MSB 1, LSB 0 is transparent black.
                        *texel = [0, 0, 0, 0];
                        continue;
                    }
                    // Table 151 replaces the small modifiers by zero when the opaque bit is unset.
                    let m = if !opaque && index == 0 { 0 } else { modifier(tables[second as usize], index) };
                    *texel = [clamp255(base[0] + m), clamp255(base[1] + m), clamp255(base[2] + m), 255];
                }
            }
        }
        m @ (Mode::T | Mode::H) => {
            let paint = if let Mode::T = m {
                // Table 141 d.
                let base1 = [
                    extend4((bits(w, 60, 59) << 2) | bits(w, 57, 56)),
                    extend4(bits(w, 55, 52)),
                    extend4(bits(w, 51, 48)),
                ];
                let base2 = [extend4(bits(w, 47, 44)), extend4(bits(w, 43, 40)), extend4(bits(w, 39, 36))];
                let d = DISTANCES[((bits(w, 35, 34) << 1) | bits(w, 32, 32)) as usize];
                [base1, base2.map(|c| c + d), base2, base2.map(|c| c - d)]
            } else {
                // Table 141 e.
                let base1 = [
                    extend4(bits(w, 62, 59)),
                    extend4((bits(w, 58, 56) << 1) | bits(w, 52, 52)),
                    extend4((bits(w, 51, 51) << 3) | bits(w, 49, 47)),
                ];
                let base2 = [extend4(bits(w, 46, 43)), extend4(bits(w, 42, 39)), extend4(bits(w, 38, 35))];
                let value = |c: [i32; 3]| (c[0] << 16) + (c[1] << 8) + c[2];
                let low = (value(base1) >= value(base2)) as u32;
                let d = DISTANCES[((bits(w, 34, 34) << 2) | (bits(w, 32, 32) << 1) | low) as usize];
                [base1.map(|c| c + d), base1.map(|c| c - d), base2.map(|c| c + d), base2.map(|c| c - d)]
            };
            for y in 0..4 {
                for x in 0..4 {
                    let index = color_index(w, x, y);
                    out[y * 4 + x] = if !opaque && index == 2 {
                        [0, 0, 0, 0]
                    } else {
                        let c = paint[index as usize];
                        [clamp255(c[0]), clamp255(c[1]), clamp255(c[2]), 255]
                    };
                }
            }
        }
        Mode::Planar => {
            // Table 141 g. The punchthrough format treats an unset opaque bit as set in this mode.
            let o = [
                extend6(bits(w, 62, 57)),
                extend7((bits(w, 56, 56) << 6) | bits(w, 54, 49)),
                extend6((bits(w, 48, 48) << 5) | (bits(w, 44, 43) << 3) | bits(w, 41, 39)),
            ];
            let h =
                [extend6((bits(w, 38, 34) << 1) | bits(w, 32, 32)), extend7(bits(w, 31, 25)), extend6(bits(w, 24, 19))];
            let v = [extend6(bits(w, 18, 13)), extend7(bits(w, 12, 6)), extend6(bits(w, 5, 0))];
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let c = |k: usize| clamp255((x * (h[k] - o[k]) + y * (v[k] - o[k]) + 4 * o[k] + 2) >> 2);
                    out[(y * 4 + x) as usize] = [c(0), c(1), c(2), 255];
                }
            }
        }
    }
    out
}

/// RGB ETC2 (section 22.1): 8-byte blocks. Alpha is 255.
pub fn decode_rgb8(block: &[u8; 8]) -> Rgba8Block {
    decode_color(word(block), false)
}

/// RGB ETC2 with punchthrough alpha (section 22.9): 8-byte blocks.
pub fn decode_rgb8a1(block: &[u8; 8]) -> Rgba8Block {
    decode_color(word(block), true)
}

/// RGBA ETC2 (section 22.3): 16-byte blocks, the alpha half first.
pub fn decode_rgba8(block: &[u8; 16]) -> Rgba8Block {
    let alpha = word(&block[..8]);
    let mut out = decode_color(word(&block[8..]), false);
    // Table 145 a and Equation 3; a zero multiplier gives zero.
    let base = bits(alpha, 63, 56) as i32;
    let multiplier = bits(alpha, 55, 52) as i32;
    let table = &ALPHA_MODIFIERS[bits(alpha, 51, 48) as usize];
    for y in 0..4 {
        for x in 0..4 {
            out[y * 4 + x][3] = clamp255(base + table[alpha_index(alpha, x, y) as usize] * multiplier);
        }
    }
    out
}

/// Unsigned R11 EAC (section 22.5) of one 8-byte channel block: 11-bit values, 0 representing 0.0 and 2047 representing
/// 1.0, in raster order.
pub fn decode_r11_unsigned(block: &[u8; 8]) -> [u16; 16] {
    let w = word(block);
    let base = bits(w, 63, 56) as i32;
    let multiplier = bits(w, 55, 52) as i32;
    let table = &ALPHA_MODIFIERS[bits(w, 51, 48) as usize];
    let mut out = [0u16; 16];
    for y in 0..4 {
        for x in 0..4 {
            let m = table[alpha_index(w, x, y) as usize];
            // Equation 5, or Equation 6 when the multiplier is zero (the multiplier is then 1/8).
            let v = if multiplier == 0 { base * 8 + 4 + m } else { base * 8 + 4 + m * multiplier * 8 };
            out[y * 4 + x] = v.clamp(0, 2047) as u16;
        }
    }
    out
}

/// Signed R11 EAC (section 22.7) of one 8-byte channel block: 11-bit two's-complement values, -1023 representing -1.0
/// and 1023 representing 1.0, in raster order.
pub fn decode_r11_signed(block: &[u8; 8]) -> [i16; 16] {
    let w = word(block);
    // The base codeword is two's complement; -128 is not allowed and is treated as -127.
    let base = (bits(w, 63, 56) as u8 as i8).max(-127) as i32;
    let multiplier = bits(w, 55, 52) as i32;
    let table = &ALPHA_MODIFIERS[bits(w, 51, 48) as usize];
    let mut out = [0i16; 16];
    for y in 0..4 {
        for x in 0..4 {
            let m = table[alpha_index(w, x, y) as usize];
            // Equation 8, or Equation 9 when the multiplier is zero.
            let v = if multiplier == 0 { base * 8 + m } else { base * 8 + m * multiplier * 8 };
            out[y * 4 + x] = v.clamp(-1023, 1023) as i16;
        }
    }
    out
}

/// The 16-bit extension of an unsigned 11-bit EAC value by bit replication, `(x << 5) + (x >> 6)` (section 22.5).
pub fn r11_unsigned_to_16(x: u16) -> u16 {
    (x << 5) + (x >> 6)
}

/// The 16-bit extension of a signed 11-bit EAC value (section 22.7): bit replication of the magnitude.
pub fn r11_signed_to_16(x: i16) -> i16 {
    if x >= 0 {
        (x << 5) + (x >> 5)
    } else {
        let m = -x;
        -((m << 5) + (m >> 5))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes the 2-bit color index of texel (x, y) into the low word.
    fn set_index(w: &mut u64, x: usize, y: usize, index: u32) {
        let i = x * 4 + y;
        *w &= !((1u64 << i) | (1u64 << (16 + i)));
        *w |= ((index as u64 & 1) << i) | ((index as u64 >> 1) << (16 + i));
    }

    fn put(w: &mut u64, hi: u32, lo: u32, v: u32) {
        let mask = ((1u64 << (hi - lo + 1)) - 1) << lo;
        *w = (*w & !mask) | (((v as u64) << lo) & mask);
    }

    #[test]
    fn extensions_match_the_worked_examples() {
        // Section 22.1: 1110b -> 238, 0011b -> 51, 1000b -> 136; 11100b -> 231, 00100b -> 33, 00011b -> 24,
        // 11000b -> 198.
        assert_eq!([extend4(14), extend4(3), extend4(8)], [238, 51, 136]);
        assert_eq!([extend5(28), extend5(4), extend5(3), extend5(24)], [231, 33, 24, 198]);
    }

    #[test]
    fn individual_mode_example() {
        // Section 22.1, Figure 56: base colors (4, 11, 9) with table 4 and (14, 3, 8) with table 0.
        let mut w = 0u64;
        put(&mut w, 63, 60, 4);
        put(&mut w, 59, 56, 14);
        put(&mut w, 55, 52, 11);
        put(&mut w, 51, 48, 3);
        put(&mut w, 47, 44, 9);
        put(&mut w, 43, 40, 8);
        put(&mut w, 39, 37, 4);
        put(&mut w, 36, 34, 0);
        // D = 0, flip = 0: subblock 1 is x < 2. The example's table prints 58 for the red channel of the -18 row; the
        // rule it illustrates gives 68 - 18 = 50, as the other rows (68 - 60 = 8, 68 + 18 = 86) confirm.
        let expect1 = [[8, 127, 93], [50, 169, 135], [86, 205, 171], [128, 247, 213]];
        let expect2 = [[230, 43, 128], [236, 49, 134], [240, 53, 138], [246, 59, 144]];
        // Modifier order -b, -a, +a, +b corresponds to pixel indices 3, 2, 0, 1.
        for (k, index) in [3u32, 2, 0, 1].into_iter().enumerate() {
            set_index(&mut w, 0, 0, index);
            set_index(&mut w, 3, 3, index);
            let out = decode_rgb8(&w.to_be_bytes());
            assert_eq!(out[0][..3], expect1[k]);
            assert_eq!(out[15][..3], expect2[k]);
            assert_eq!(out[0][3], 255);
        }
    }

    #[test]
    fn differential_mode_example() {
        // Figure 57: (29, 26, 8) with table 2, offset (-4, -3, +3) with table 3.
        let mut w = 0u64;
        put(&mut w, 63, 59, 29);
        put(&mut w, 58, 56, 0b100);
        put(&mut w, 55, 51, 26);
        put(&mut w, 50, 48, 0b101);
        put(&mut w, 47, 43, 8);
        put(&mut w, 42, 40, 3);
        put(&mut w, 39, 37, 2);
        put(&mut w, 36, 34, 3);
        put(&mut w, 33, 33, 1);
        put(&mut w, 32, 32, 1); // flip: subblock 2 is y >= 2
        let expect1 = [[210, 185, 37], [230, 205, 57], [248, 223, 75], [255, 243, 95]];
        let expect2 = [[164, 147, 48], [193, 176, 77], [219, 202, 103], [248, 231, 132]];
        for (k, index) in [3u32, 2, 0, 1].into_iter().enumerate() {
            set_index(&mut w, 3, 1, index);
            set_index(&mut w, 0, 2, index);
            let out = decode_rgb8(&w.to_be_bytes());
            assert_eq!(out[4 + 3][..3], expect1[k]);
            assert_eq!(out[8][..3], expect2[k]);
        }
    }

    #[test]
    fn t_mode_example() {
        // Figure 58: base 1 (13, 1, 8), base 2 (4, 12, 13), distance index 5 (d = 32). The 'T' mode is chosen by
        // R + R_d overflowing: R = bits 63..59 and R_d = bits 58..56 must sum outside 0..31.
        let mut w = 0u64;
        put(&mut w, 60, 59, 13 >> 2);
        put(&mut w, 57, 56, 13 & 3);
        put(&mut w, 55, 52, 1);
        put(&mut w, 51, 48, 8);
        put(&mut w, 47, 44, 4);
        put(&mut w, 43, 40, 12);
        put(&mut w, 39, 36, 13);
        put(&mut w, 35, 34, 0b10);
        put(&mut w, 33, 33, 1);
        put(&mut w, 32, 32, 1);
        // R = bits 63..59 = 11111 and R_d = bits 58..56 = 001, so R + R_d = 32 selects the 'T' mode.
        put(&mut w, 63, 61, 0b111);
        put(&mut w, 58, 58, 0);
        assert!(matches!(mode(w, false), Mode::T));
        let expect = [[221, 17, 136], [100, 236, 253], [68, 204, 221], [36, 172, 189]];
        for index in 0..4u32 {
            set_index(&mut w, 1, 2, index);
            assert_eq!(decode_rgb8(&w.to_be_bytes())[2 * 4 + 1][..3], expect[index as usize]);
        }
    }

    #[test]
    fn h_mode_example() {
        // Figure 59: base 1 (13, 1, 8), base 2 (4, 12, 13), distance index 5: d_a = 1, d_b = 0, and base 1 >= base 2
        // supplies the low bit.
        let mut w = 0u64;
        put(&mut w, 62, 59, 13);
        put(&mut w, 58, 56, 1 >> 1);
        put(&mut w, 52, 52, 1 & 1);
        put(&mut w, 51, 51, 8 >> 3);
        put(&mut w, 49, 47, 8 & 7);
        put(&mut w, 46, 43, 4);
        put(&mut w, 42, 39, 12);
        put(&mut w, 38, 35, 13);
        put(&mut w, 34, 34, 1);
        put(&mut w, 33, 33, 1);
        put(&mut w, 32, 32, 0);
        // G + G_d must leave 0..31 while R + R_d stays inside: R = bits 63..59 = 01101 and R_d = bits 58..56 = 000;
        // G = bits 55..51 = 00011 (bits 52 and 51 are G^0 and B^3) and G_d = bits 50..48 = 100 = -4.
        put(&mut w, 55, 53, 0);
        put(&mut w, 50, 50, 1);
        put(&mut w, 63, 63, 0);
        assert!(matches!(mode(w, false), Mode::H), "block does not select H");
        let expect = [[253, 49, 168], [189, 0, 104], [100, 236, 253], [36, 172, 189]];
        for index in 0..4u32 {
            set_index(&mut w, 2, 0, index);
            assert_eq!(decode_rgb8(&w.to_be_bytes())[2][..3], expect[index as usize]);
        }
    }

    #[test]
    fn planar_mode_example() {
        // Figure 60: origin (12, 64, 62), horizontal (50, 5, 37), vertical (40, 112, 45).
        let mut w = 0u64;
        put(&mut w, 62, 57, 12);
        put(&mut w, 56, 56, 64 >> 6);
        put(&mut w, 54, 49, 64 & 63);
        put(&mut w, 48, 48, 62 >> 5);
        put(&mut w, 44, 43, (62 >> 3) & 3);
        put(&mut w, 41, 39, 62 & 7);
        put(&mut w, 38, 34, 50 >> 1);
        put(&mut w, 32, 32, 50 & 1);
        put(&mut w, 31, 25, 5);
        put(&mut w, 24, 19, 37);
        put(&mut w, 18, 13, 40);
        put(&mut w, 12, 6, 112);
        put(&mut w, 5, 0, 45);
        put(&mut w, 33, 33, 1);
        // B + B_d must overflow while R + R_d and G + G_d stay in range: bits 47..43 = B, 42..40 = B_d.
        put(&mut w, 47, 45, 0b111);
        put(&mut w, 42, 42, 0);
        put(&mut w, 63, 63, 0);
        put(&mut w, 55, 55, 0);
        assert!(matches!(mode(w, false), Mode::Planar), "block does not select planar");
        let out = decode_rgb8(&w.to_be_bytes());
        // The table of Figure 60 lists (x, y) row by row with x fastest.
        let expect = [
            [48, 129, 251],
            [87, 99, 226],
            [126, 70, 201],
            [164, 40, 175],
            [77, 153, 234],
            [115, 123, 209],
            [154, 94, 183],
            [193, 64, 158],
            [105, 177, 217],
            [144, 147, 191],
            [183, 118, 166],
            [221, 88, 141],
            [134, 201, 199],
            [172, 171, 174],
            [211, 142, 149],
            [250, 112, 124],
        ];
        for (i, e) in expect.iter().enumerate() {
            assert_eq!(out[i][..3], *e, "texel {i}");
        }
    }

    #[test]
    fn alpha_and_eac_examples() {
        // Sections 22.3 and 22.5: base codeword 103, table 13, pixel index 3, multiplier 2.
        let mut w = 0u64;
        put(&mut w, 63, 56, 103);
        put(&mut w, 55, 52, 2);
        put(&mut w, 51, 48, 13);
        // Pixel b is (x = 0, y = 1): index bits 44..42.
        put(&mut w, 44, 42, 3);
        let mut block = [0u8; 16];
        block[..8].copy_from_slice(&w.to_be_bytes());
        assert_eq!(decode_rgba8(&block)[4][3], 83);
        assert_eq!(decode_r11_unsigned(&w.to_be_bytes())[4], 668);
        assert_eq!(r11_unsigned_to_16(668), 21386);
        put(&mut w, 55, 52, 0);
        assert_eq!(decode_r11_unsigned(&w.to_be_bytes())[4], 818);
        // Alpha with a zero multiplier is the base codeword.
        block[..8].copy_from_slice(&w.to_be_bytes());
        assert_eq!(decode_rgba8(&block)[4][3], 103);

        // Section 22.7: base 60, table 13, index 3, multiplier 2 -> 320; multiplier 0 -> 470.
        put(&mut w, 63, 56, 60);
        put(&mut w, 55, 52, 2);
        assert_eq!(decode_r11_signed(&w.to_be_bytes())[4], 320);
        put(&mut w, 55, 52, 0);
        assert_eq!(decode_r11_signed(&w.to_be_bytes())[4], 470);
        assert_eq!(r11_signed_to_16(470), 15054);
        // -128 is treated as -127.
        put(&mut w, 63, 56, 0x80);
        let a = decode_r11_signed(&w.to_be_bytes());
        put(&mut w, 63, 56, 0x81);
        assert_eq!(decode_r11_signed(&w.to_be_bytes()), a);
    }

    #[test]
    fn punchthrough_transparent_index() {
        // Differential mode with the opaque bit unset: index 2 is transparent black, index 0 uses modifier zero.
        let mut w = 0u64;
        put(&mut w, 63, 59, 28);
        put(&mut w, 55, 51, 4);
        put(&mut w, 47, 43, 3);
        put(&mut w, 39, 37, 2);
        set_index(&mut w, 0, 0, 2);
        set_index(&mut w, 1, 0, 0);
        set_index(&mut w, 0, 1, 1);
        let out = decode_rgb8a1(&w.to_be_bytes());
        assert_eq!(out[0], [0, 0, 0, 0]);
        assert_eq!(out[1], [231, 33, 24, 255]);
        assert_eq!(out[4], [255, 62, 53, 255]);
        // With the opaque bit set the same indices use Table 148.
        put(&mut w, 33, 33, 1);
        let out = decode_rgb8a1(&w.to_be_bytes());
        assert_eq!(out[0], [222, 24, 15, 255]);
        assert_eq!(out[1], [240, 42, 33, 255]);
    }
}
