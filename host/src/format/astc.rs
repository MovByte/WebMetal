//! ASTC block decoding for the LDR profile, as the Khronos Data Format Specification 1.4.0 defines it (section 23,
//! "ASTC Compressed Texture Image Formats"; the pinned text is `kdfs` in `tables/sources.toml`).
//!
//! The decoder implements the two operation modes of the LDR profile (section 23.24): sRGB and linear LDR. HDR
//! endpoint modes and HDR void-extent blocks decode to the error color, as the LDR profile requires. Every illegal
//! encoding of section 23.23 is recognized and decodes to the error color.
//!
//! [`decode_block`] returns, for each texel of the block in raster order (index `y * width + x`), the UNORM16
//! interpolation result `C` of section 23.18.1 for the four channels, or `None` for the error color. [`to_float16`]
//! and [`to_srgb8`] apply the final conversions of section 23.18.1 for the `decode_float16` decode mode and for the
//! sRGB operation mode.
//!
//! The textures this decodes are 2D (or 3D textures made of 2D slices, which section 23.8 decodes slice by slice);
//! the 3D block footprints of Table 156 are not part of the LDR profile and are not decoded.

/// The 2D block footprints of Table 155, the only ones the LDR profile supports.
pub const FOOTPRINTS: [(u32, u32); 14] = [
    (4, 4),
    (5, 4),
    (5, 5),
    (6, 5),
    (6, 6),
    (8, 5),
    (8, 6),
    (8, 8),
    (10, 5),
    (10, 6),
    (10, 8),
    (10, 10),
    (12, 10),
    (12, 12),
];

/// Bytes in every ASTC block (section 23.9).
pub const BLOCK_BYTES: usize = 16;

/// The operation mode (section 23.5). HDR operation mode is outside the LDR profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationMode {
    Srgb,
    LinearLdr,
}

/// The magenta error color (Table 154), (1.0, 0.0, 1.0, 1.0), as UNORM16 values: every conversion of section
/// 23.18.1 and [`to_unorm12`] maps 0 to 0.0 and 65535 to 1.0.
pub const ERROR_UNORM16: [u16; 4] = [0xffff, 0, 0xffff, 0xffff];
/// The magenta error color as IEEE binary16 values (`decode_float16`).
pub const ERROR_FLOAT16: [u16; 4] = [0x3c00, 0, 0x3c00, 0x3c00];
/// The magenta error color as the 8-bit unsigned normalized values of the sRGB operation mode.
pub const ERROR_SRGB8: [u8; 4] = [255, 0, 255, 255];

/// Why a footprint is not decodable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsupportedFootprint {
    pub width: u32,
    pub height: u32,
}

/// The value range of an integer sequence (Table 167): `0 .. (3^trits * 5^quints * 2^bits) - 1` with at most one
/// trit or one quint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Range {
    trits: u32,
    quints: u32,
    bits: u32,
}

impl Range {
    const fn new(trits: u32, quints: u32, bits: u32) -> Range {
        Range { trits, quints, bits }
    }

    /// Bits that `count` values occupy (section 23.21).
    fn stream_bits(self, count: u32) -> u32 {
        (count * 8 * self.trits).div_ceil(5) + (count * 7 * self.quints).div_ceil(3) + count * self.bits
    }
}

/// Weight ranges of Table 160, indexed by `[P][ρ]`; `None` is invalid.
const WEIGHT_RANGES: [[Option<Range>; 8]; 2] = [
    [
        None,
        None,
        Some(Range::new(0, 0, 1)), // 0..1
        Some(Range::new(1, 0, 0)), // 0..2
        Some(Range::new(0, 0, 2)), // 0..3
        Some(Range::new(0, 1, 0)), // 0..4
        Some(Range::new(1, 0, 1)), // 0..5
        Some(Range::new(0, 0, 3)), // 0..7
    ],
    [
        None,
        None,
        Some(Range::new(0, 1, 1)), // 0..9
        Some(Range::new(1, 0, 2)), // 0..11
        Some(Range::new(0, 0, 4)), // 0..15
        Some(Range::new(0, 1, 2)), // 0..19
        Some(Range::new(1, 0, 3)), // 0..23
        Some(Range::new(0, 0, 5)), // 0..31
    ],
];

/// The color endpoint ranges in descending order of size (section 23.21). The smallest is 0..5: a block that cannot
/// fit its endpoints at that range is illegal (section 23.23).
const COLOR_RANGES: [Range; 17] = [
    Range::new(0, 0, 8), // 0..255
    Range::new(1, 0, 6), // 0..191
    Range::new(0, 1, 5), // 0..159
    Range::new(0, 0, 7), // 0..127
    Range::new(1, 0, 5), // 0..95
    Range::new(0, 1, 4), // 0..79
    Range::new(0, 0, 6), // 0..63
    Range::new(1, 0, 4), // 0..47
    Range::new(0, 1, 3), // 0..39
    Range::new(0, 0, 5), // 0..31
    Range::new(1, 0, 3), // 0..23
    Range::new(0, 1, 2), // 0..19
    Range::new(0, 0, 4), // 0..15
    Range::new(1, 0, 2), // 0..11
    Range::new(0, 1, 1), // 0..9
    Range::new(0, 0, 3), // 0..7
    Range::new(1, 0, 1), // 0..5
];

/// Table 172: color unquantization parameters `(B bit layout, C)` for trit and quint ranges, keyed by the range.
/// The layout lists bits 8..0 of `B`; a letter is a bit of the value's low bits (`a` = bit 0, `b` = bit 1, ...).
const COLOR_UNQUANT: [(Range, &str, u32); 11] = [
    (Range::new(1, 0, 1), "000000000", 204),
    (Range::new(0, 1, 1), "000000000", 113),
    (Range::new(1, 0, 2), "b000b0bb0", 93),
    (Range::new(0, 1, 2), "b0000bb00", 54),
    (Range::new(1, 0, 3), "cb000cbcb", 44),
    (Range::new(0, 1, 3), "cb0000cbc", 26),
    (Range::new(1, 0, 4), "dcb000dcb", 22),
    (Range::new(0, 1, 4), "dcb0000dc", 13),
    (Range::new(1, 0, 5), "edcb000ed", 11),
    (Range::new(0, 1, 5), "edcb0000e", 6),
    (Range::new(1, 0, 6), "fedcb000f", 5),
];

/// Table 182: weight unquantization parameters, as [`COLOR_UNQUANT`] with 7-bit `B` layouts.
const WEIGHT_UNQUANT: [(Range, &str, u32); 5] = [
    (Range::new(1, 0, 1), "0000000", 50),
    (Range::new(0, 1, 1), "0000000", 28),
    (Range::new(1, 0, 2), "b000b0b", 23),
    (Range::new(0, 1, 2), "b0000b0", 13),
    (Range::new(1, 0, 3), "cb000cb", 11),
];

/// `B` from a Table 172 or 182 bit layout and the value's low bits `m`.
fn layout_value(layout: &str, m: u32) -> u32 {
    layout.bytes().fold(0, |acc, c| (acc << 1) | if c == b'0' { 0 } else { (m >> (c - b'a')) & 1 })
}

/// A 128-bit block with bit `n` the bit `n % 8` of byte `n / 8` (section 23.9).
#[derive(Clone, Copy)]
struct Bits(u128);

impl Bits {
    fn get(self, lo: u32, count: u32) -> u32 {
        if count == 0 {
            return 0;
        }
        ((self.0 >> lo) & ((1u128 << count) - 1)) as u32
    }
}

/// An integer-sequence-encoded stream (section 23.12): `length` bits starting at bit `start` of `bits`. Bits at or
/// beyond `length` read as 0 ("unused (higher-order) bits are assumed to be 0").
struct Stream {
    bits: Bits,
    start: u32,
    length: u32,
}

impl Stream {
    fn read(&self, offset: u32, count: u32) -> u32 {
        let mut v = 0;
        for i in 0..count {
            let pos = offset + i;
            if pos < self.length {
                v |= ((self.bits.0 >> (self.start + pos)) & 1) as u32 * (1 << i);
            }
        }
        v
    }
}

/// The five trits of an 8-bit packed value `T` (section 23.12).
fn trits(t: u32) -> [u32; 5] {
    let b = |hi: u32, lo: u32| (t >> lo) & ((1 << (hi - lo + 1)) - 1);
    let (c, t4, t3);
    if b(4, 2) == 0b111 {
        c = (b(7, 5) << 2) | b(1, 0);
        t4 = 2;
        t3 = 2;
    } else {
        c = b(4, 0);
        if b(6, 5) == 0b11 {
            t4 = 2;
            t3 = b(7, 7);
        } else {
            t4 = b(7, 7);
            t3 = b(6, 5);
        }
    }
    let cb = |hi: u32, lo: u32| (c >> lo) & ((1 << (hi - lo + 1)) - 1);
    let (t2, t1, t0);
    if cb(1, 0) == 0b11 {
        t2 = 2;
        t1 = cb(4, 4);
        t0 = (cb(3, 3) << 1) | (cb(2, 2) & !cb(3, 3) & 1);
    } else if cb(3, 2) == 0b11 {
        t2 = 2;
        t1 = 2;
        t0 = cb(1, 0);
    } else {
        t2 = cb(4, 4);
        t1 = cb(3, 2);
        t0 = (cb(1, 1) << 1) | (cb(0, 0) & !cb(1, 1) & 1);
    }
    [t0, t1, t2, t3, t4]
}

/// The three quints of a 7-bit packed value `Q` (section 23.12).
fn quints(q: u32) -> [u32; 3] {
    let b = |hi: u32, lo: u32| (q >> lo) & ((1 << (hi - lo + 1)) - 1);
    if b(2, 1) == 0b11 && b(6, 5) == 0b00 {
        let q2 = (b(0, 0) << 2) | ((b(4, 4) & !b(0, 0) & 1) << 1) | (b(3, 3) & !b(0, 0) & 1);
        return [4, 4, q2];
    }
    let (q2, c);
    if b(2, 1) == 0b11 {
        q2 = 4;
        c = (b(4, 3) << 3) | ((!b(6, 5) & 0b11) << 1) | b(0, 0);
    } else {
        q2 = b(6, 5);
        c = b(4, 0);
    }
    let cb = |hi: u32, lo: u32| (c >> lo) & ((1 << (hi - lo + 1)) - 1);
    if cb(2, 0) == 0b101 { [cb(4, 3), 4, q2] } else { [cb(2, 0), cb(4, 3), q2] }
}

/// Decodes `count` values of `range` from `stream`: `(trit or quint, low bits)` per value.
fn decode_ise(stream: &Stream, range: Range, count: u32) -> Vec<(u32, u32)> {
    let n = range.bits;
    let mut out = Vec::with_capacity(count as usize);
    if range.trits == 1 {
        // Table 169: m0, T1..0, m1, T3..2, m2, T4, m3, T6..5, m4, T7.
        let mut block = 0;
        while out.len() < count as usize {
            let base = block * (8 + 5 * n);
            let mut pos = base;
            let mut m = [0u32; 5];
            let mut t = 0u32;
            let fields: [(Option<usize>, u32, u32); 10] = [
                (Some(0), n, 0),
                (None, 2, 0),
                (Some(1), n, 0),
                (None, 2, 2),
                (Some(2), n, 0),
                (None, 1, 4),
                (Some(3), n, 0),
                (None, 2, 5),
                (Some(4), n, 0),
                (None, 1, 7),
            ];
            for (slot, width, shift) in fields {
                let v = stream.read(pos, width);
                match slot {
                    Some(i) => m[i] = v,
                    None => t |= v << shift,
                }
                pos += width;
            }
            let ts = trits(t);
            for i in 0..5 {
                if out.len() < count as usize {
                    out.push((ts[i], m[i]));
                }
            }
            block += 1;
        }
    } else if range.quints == 1 {
        // Tables 170 and 171: m0, Q2..0, m1, Q4..3, m2, Q6..5.
        let mut block = 0;
        while out.len() < count as usize {
            let mut pos = block * (7 + 3 * n);
            let mut m = [0u32; 3];
            let mut q = 0u32;
            let fields: [(Option<usize>, u32, u32); 6] =
                [(Some(0), n, 0), (None, 3, 0), (Some(1), n, 0), (None, 2, 3), (Some(2), n, 0), (None, 2, 5)];
            for (slot, width, shift) in fields {
                let v = stream.read(pos, width);
                match slot {
                    Some(i) => m[i] = v,
                    None => q |= v << shift,
                }
                pos += width;
            }
            let qs = quints(q);
            for i in 0..3 {
                if out.len() < count as usize {
                    out.push((qs[i], m[i]));
                }
            }
            block += 1;
        }
    } else {
        for i in 0..count {
            out.push((0, stream.read(i * n, n)));
        }
    }
    out
}

/// Bit replication of an `n`-bit value to `width` bits, from the most significant bit.
fn replicate(v: u32, n: u32, width: u32) -> u32 {
    let mut out = 0;
    let mut have = 0;
    while have < width {
        let take = n.min(width - have);
        out = (out << take) | (v >> (n - take));
        have += take;
    }
    out
}

/// Section 23.13: an encoded color value to 0..255.
fn unquantize_color(range: Range, (d, m): (u32, u32)) -> u32 {
    if range.trits == 0 && range.quints == 0 {
        return replicate(m, range.bits, 8);
    }
    let (_, layout, c) = COLOR_UNQUANT.iter().find(|(r, ..)| *r == range).expect("a color range of Table 172");
    let a = if m & 1 == 1 { 0x1ff } else { 0 };
    let b = layout_value(layout, m);
    let unq = (d * c + b) ^ a;
    (a & 0x80) | (unq >> 2)
}

/// Section 23.16: an encoded weight to 0..64.
fn unquantize_weight(range: Range, (d, m): (u32, u32)) -> u32 {
    let unq = if range.trits == 0 && range.quints == 0 {
        replicate(m, range.bits, 6)
    } else if range.bits == 0 {
        // Table 181.
        if range.trits == 1 { [0, 32, 63][d as usize] } else { [0, 16, 32, 47, 63][d as usize] }
    } else {
        let (_, layout, c) = WEIGHT_UNQUANT.iter().find(|(r, ..)| *r == range).expect("a weight range of Table 182");
        let a = if m & 1 == 1 { 0x7f } else { 0 };
        let b = layout_value(layout, m);
        let unq = (d * c + b) ^ a;
        (a & 0x20) | (unq >> 2)
    };
    if unq > 32 { unq + 1 } else { unq }
}

/// The block mode fields of Table 161.
struct BlockMode {
    width: u32,
    height: u32,
    dual_plane: bool,
    range: Range,
}

/// Table 161. `None` for the reserved rows and invalid weight ranges.
fn block_mode(mode: u32) -> Option<BlockMode> {
    let bit = |n: u32| (mode >> n) & 1;
    let field = |hi: u32, lo: u32| (mode >> lo) & ((1 << (hi - lo + 1)) - 1);
    let (width, height, rho, high, dual);
    if field(1, 0) != 0 {
        rho = (bit(1) << 2) | (bit(0) << 1) | bit(4);
        let (a, b) = (field(6, 5), field(8, 7));
        (width, height) = match field(3, 2) {
            0b00 => (b + 4, a + 2),
            0b01 => (b + 8, a + 2),
            0b10 => (a + 2, b + 8),
            _ if bit(8) == 0 => (a + 2, bit(7) + 6),
            _ => (bit(7) + 2, a + 2),
        };
        high = bit(9);
        dual = bit(10);
    } else {
        if field(3, 0) == 0 {
            return None;
        }
        rho = (bit(3) << 2) | (bit(2) << 1) | bit(4);
        let a = field(6, 5);
        match field(8, 7) {
            0b00 => {
                (width, height) = (12, a + 2);
                high = bit(9);
                dual = bit(10);
            }
            0b01 => {
                (width, height) = (a + 2, 12);
                high = bit(9);
                dual = bit(10);
            }
            0b10 => {
                (width, height) = (a + 6, field(10, 9) + 6);
                high = 0;
                dual = 0;
            }
            _ => {
                (width, height) = match field(6, 5) {
                    0b00 => (6, 10),
                    0b01 => (10, 6),
                    // Bits 8..6 all 1: reserved unless the block is a void-extent block, which is handled first.
                    _ => return None,
                };
                high = bit(9);
                dual = bit(10);
            }
        }
    }
    let range = WEIGHT_RANGES[high as usize][rho as usize]?;
    Some(BlockMode { width, height, dual_plane: dual == 1, range })
}

/// Section 23.20.
fn hash52(mut p: u32) -> u32 {
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    p
}

/// Section 23.20, `select_partition`.
fn select_partition(seed: u32, x: u32, y: u32, z: u32, partitions: u32, small_block: bool) -> u32 {
    let (x, y, z) = if small_block { (x << 1, y << 1, z << 1) } else { (x, y, z) };
    let seed = seed + (partitions - 1) * 1024;
    let rnum = hash52(seed);
    let mut s = [
        rnum & 0xf,
        (rnum >> 4) & 0xf,
        (rnum >> 8) & 0xf,
        (rnum >> 12) & 0xf,
        (rnum >> 16) & 0xf,
        (rnum >> 20) & 0xf,
        (rnum >> 24) & 0xf,
        (rnum >> 28) & 0xf,
        (rnum >> 18) & 0xf,
        (rnum >> 22) & 0xf,
        (rnum >> 26) & 0xf,
        // seed12 = ((rnum >> 30) | (rnum << 2)) & 0xF, a rotation.
        rnum.rotate_left(2) & 0xf,
    ];
    for v in &mut s {
        *v *= *v;
    }
    let (sh1, sh2) = if seed & 1 != 0 {
        (if seed & 2 != 0 { 4 } else { 5 }, if partitions == 3 { 6 } else { 5 })
    } else {
        (if partitions == 3 { 6 } else { 5 }, if seed & 2 != 0 { 4 } else { 5 })
    };
    let sh3 = if seed & 0x10 != 0 { sh1 } else { sh2 };
    for (i, v) in s.iter_mut().enumerate() {
        *v >>= match i {
            0 | 2 | 4 | 6 => sh1,
            1 | 3 | 5 | 7 => sh2,
            _ => sh3,
        };
    }
    let a = s[0].wrapping_mul(x).wrapping_add(s[1].wrapping_mul(y)).wrapping_add(s[10].wrapping_mul(z)) + (rnum >> 14);
    let b = s[2].wrapping_mul(x).wrapping_add(s[3].wrapping_mul(y)).wrapping_add(s[11].wrapping_mul(z)) + (rnum >> 10);
    let c = s[4].wrapping_mul(x).wrapping_add(s[5].wrapping_mul(y)).wrapping_add(s[8].wrapping_mul(z)) + (rnum >> 6);
    let d = s[6].wrapping_mul(x).wrapping_add(s[7].wrapping_mul(y)).wrapping_add(s[9].wrapping_mul(z)) + (rnum >> 2);
    let (a, b, mut c, mut d) = (a & 0x3f, b & 0x3f, c & 0x3f, d & 0x3f);
    if partitions < 4 {
        d = 0;
    }
    if partitions < 3 {
        c = 0;
    }
    if a >= b && a >= c && a >= d {
        0
    } else if b >= c && b >= d {
        1
    } else if c >= d {
        2
    } else {
        3
    }
}

/// `bit_transfer_signed` of section 23.14.1.
fn bit_transfer_signed(a: &mut i32, b: &mut i32) {
    *b >>= 1;
    *b |= *a & 0x80;
    *a >>= 1;
    *a &= 0x3f;
    if *a & 0x20 != 0 {
        *a -= 0x40;
    }
}

fn blue_contract(r: i32, g: i32, b: i32, a: i32) -> [i32; 4] {
    [(r + b) >> 1, (g + b) >> 1, b, a]
}

fn clamp_unorm8(c: [i32; 4]) -> [i32; 4] {
    c.map(|v| v.clamp(0, 255))
}

/// Section 23.14.1: the endpoints of an LDR endpoint mode, or `None` for an HDR endpoint mode.
fn ldr_endpoints(cem: u32, v: &[i32]) -> Option<([i32; 4], [i32; 4])> {
    let mut v: Vec<i32> = v.to_vec();
    Some(match cem {
        0 => ([v[0], v[0], v[0], 0xff], [v[1], v[1], v[1], 0xff]),
        1 => {
            let l0 = (v[0] >> 2) | (v[1] & 0xc0);
            let l1 = (l0 + (v[1] & 0x3f)).min(0xff);
            ([l0, l0, l0, 0xff], [l1, l1, l1, 0xff])
        }
        4 => ([v[0], v[0], v[0], v[2]], [v[1], v[1], v[1], v[3]]),
        5 => {
            let (mut v0, mut v1, mut v2, mut v3) = (v[0], v[1], v[2], v[3]);
            bit_transfer_signed(&mut v1, &mut v0);
            bit_transfer_signed(&mut v3, &mut v2);
            let e0 = [v0, v0, v0, v2];
            let e1 = [v0 + v1, v0 + v1, v0 + v1, v2 + v3];
            (clamp_unorm8(e0), clamp_unorm8(e1))
        }
        6 => ([(v[0] * v[3]) >> 8, (v[1] * v[3]) >> 8, (v[2] * v[3]) >> 8, 0xff], [v[0], v[1], v[2], 0xff]),
        8 => {
            let s0 = v[0] + v[2] + v[4];
            let s1 = v[1] + v[3] + v[5];
            if s1 >= s0 {
                ([v[0], v[2], v[4], 0xff], [v[1], v[3], v[5], 0xff])
            } else {
                (blue_contract(v[1], v[3], v[5], 0xff), blue_contract(v[0], v[2], v[4], 0xff))
            }
        }
        9 | 13 => {
            let pairs = if cem == 9 { 3 } else { 4 };
            for p in 0..pairs {
                let (lo, hi) = v.split_at_mut(2 * p + 1);
                bit_transfer_signed(&mut hi[0], &mut lo[2 * p]);
            }
            let a0 = if cem == 9 { 0xff } else { v[6] };
            let a1 = if cem == 9 { 0xff } else { v[6] + v[7] };
            let (e0, e1) = if v[1] + v[3] + v[5] >= 0 {
                ([v[0], v[2], v[4], a0], [v[0] + v[1], v[2] + v[3], v[4] + v[5], a1])
            } else {
                (blue_contract(v[0] + v[1], v[2] + v[3], v[4] + v[5], a1), blue_contract(v[0], v[2], v[4], a0))
            };
            (clamp_unorm8(e0), clamp_unorm8(e1))
        }
        10 => ([(v[0] * v[3]) >> 8, (v[1] * v[3]) >> 8, (v[2] * v[3]) >> 8, v[4]], [v[0], v[1], v[2], v[5]]),
        12 => {
            let s0 = v[0] + v[2] + v[4];
            let s1 = v[1] + v[3] + v[5];
            if s1 >= s0 {
                ([v[0], v[2], v[4], v[6]], [v[1], v[3], v[5], v[7]])
            } else {
                (blue_contract(v[1], v[3], v[5], v[7]), blue_contract(v[0], v[2], v[4], v[6]))
            }
        }
        // 2, 3, 7, 11, 14 and 15 are HDR endpoint modes (Table 163).
        _ => return None,
    })
}

/// A decoded block: per texel, the UNORM16 `C` of each channel, or `None` for the error color.
pub type DecodedBlock = Vec<Option<[u16; 4]>>;

fn error_block(texels: usize) -> DecodedBlock {
    vec![None; texels]
}

/// Decodes one 16-byte block of a `width` × `height` footprint in the given operation mode.
pub fn decode_block(
    block: &[u8; BLOCK_BYTES],
    width: u32,
    height: u32,
    mode: OperationMode,
) -> Result<DecodedBlock, UnsupportedFootprint> {
    if !FOOTPRINTS.contains(&(width, height)) {
        return Err(UnsupportedFootprint { width, height });
    }
    let texels = (width * height) as usize;
    let bits = Bits(u128::from_le_bytes(*block));
    let srgb = mode == OperationMode::Srgb;

    // Section 23.9.3 and 23.22: void-extent blocks.
    if bits.get(0, 9) == 0b1_1111_1100 {
        // Bits 10 and 11 are reserved and must be 1; the D bit (9) selects HDR, which decodes to the error color in
        // the LDR operation modes.
        if bits.get(10, 2) != 0b11 || bits.get(9, 1) == 1 {
            return Ok(error_block(texels));
        }
        let (min_s, max_s, min_t, max_t) = (bits.get(12, 13), bits.get(25, 13), bits.get(38, 13), bits.get(51, 13));
        let all_ones = [min_s, max_s, min_t, max_t].iter().all(|&c| c == 0x1fff);
        if !all_ones && (min_s >= max_s || min_t >= max_t) {
            return Ok(error_block(texels));
        }
        let c = [bits.get(64, 16), bits.get(80, 16), bits.get(96, 16), bits.get(112, 16)].map(|v| v as u16);
        return Ok(vec![Some(c); texels]);
    }

    let Some(bm) = block_mode(bits.get(0, 11)) else {
        return Ok(error_block(texels));
    };
    let partitions = bits.get(11, 2) + 1;
    let planes = if bm.dual_plane { 2 } else { 1 };
    let weight_count = bm.width * bm.height * planes;
    let weight_bits = bm.range.stream_bits(weight_count);
    // Section 23.23.
    if bm.width > width
        || bm.height > height
        || weight_count > 64
        || !(24..=96).contains(&weight_bits)
        || (bm.dual_plane && partitions == 4)
    {
        return Ok(error_block(texels));
    }

    // Color endpoint modes (section 23.11) and the configuration size (section 23.21).
    let mut cems = [0u32; 4];
    let mut config_bits;
    let mut extra_cem_bits = 0;
    if partitions == 1 {
        cems[0] = bits.get(13, 4);
        config_bits = 17;
    } else {
        let selector = bits.get(23, 2);
        if selector == 0 {
            let cem = bits.get(25, 4);
            cems[..partitions as usize].fill(cem);
            config_bits = 29;
        } else {
            // Tables 165 and 166: the CEM field continues in 3 * partitions - 4 bits directly below the weights.
            extra_cem_bits = 3 * partitions - 4;
            let extra = bits.get(128 - weight_bits - extra_cem_bits, extra_cem_bits);
            let field = bits.get(23, 6) | (extra << 6);
            let base_class = selector - 1;
            for (p, cem) in cems.iter_mut().enumerate().take(partitions as usize) {
                let class = base_class + ((field >> (2 + p)) & 1);
                let low = (field >> (2 + partitions as usize + 2 * p)) & 0b11;
                *cem = (class << 2) | low;
            }
            config_bits = 25 + 3 * partitions;
        }
    }
    let ccs_pos = 128 - weight_bits - extra_cem_bits - 2;
    if bm.dual_plane {
        config_bits += 2;
    }
    let color_values: u32 = cems[..partitions as usize].iter().map(|c| 2 * ((c >> 2) + 1)).sum();
    if color_values > 18 || config_bits + weight_bits > 128 {
        return Ok(error_block(texels));
    }
    let remaining = 128 - config_bits - weight_bits;
    if remaining < (13 * color_values).div_ceil(5) {
        return Ok(error_block(texels));
    }
    let color_range = *COLOR_RANGES
        .iter()
        .find(|r| r.stream_bits(color_values) <= remaining)
        .expect("range 0..5 fits once the 13 * C / 5 check has passed");

    // Color endpoints (sections 23.13 and 23.14).
    let color_start = if partitions == 1 { 17 } else { 29 };
    let color_stream = Stream { bits, start: color_start, length: color_range.stream_bits(color_values) };
    let colors: Vec<i32> = decode_ise(&color_stream, color_range, color_values)
        .into_iter()
        .map(|v| unquantize_color(color_range, v) as i32)
        .collect();
    let mut endpoints = [None; 4];
    let mut next = 0;
    for p in 0..partitions as usize {
        let n = 2 * ((cems[p] >> 2) + 1) as usize;
        endpoints[p] = ldr_endpoints(cems[p], &colors[next..next + n]);
        next += n;
    }

    // Weights (sections 23.15 and 23.16): the stream grows down from bit 127.
    let reversed = Bits(bits.0.reverse_bits());
    let weight_stream = Stream { bits: reversed, start: 0, length: weight_bits };
    let weights: Vec<u32> = decode_ise(&weight_stream, bm.range, weight_count)
        .into_iter()
        .map(|v| unquantize_weight(bm.range, v))
        .collect();
    let plane_weights: Vec<Vec<u32>> =
        (0..planes).map(|pl| weights.iter().skip(pl as usize).step_by(planes as usize).copied().collect()).collect();
    let ccs = if bm.dual_plane { Some(bits.get(ccs_pos, 2) as usize) } else { None };

    let seed = bits.get(13, 10);
    let small_block = texels < 31;
    // Section 23.17: scale factors for the infill.
    let ds = (1024 + width / 2) / (width - 1);
    let dt = (1024 + height / 2) / (height - 1);
    let mut out = Vec::with_capacity(texels);
    for y in 0..height {
        for x in 0..width {
            let p = if partitions > 1 { select_partition(seed, x, y, 0, partitions, small_block) } else { 0 };
            let Some((e0, e1)) = endpoints[p as usize] else {
                out.push(None);
                continue;
            };
            let gs = (ds * x * (bm.width - 1) + 32) >> 6;
            let gt = (dt * y * (bm.height - 1) + 32) >> 6;
            let (js, fs) = (gs >> 4, gs & 0xf);
            let (jt, ft) = (gt >> 4, gt & 0xf);
            let w11 = (fs * ft + 8) >> 4;
            let w10 = ft - w11;
            let w01 = fs - w11;
            // 16 - fs - ft + w11 is never negative; add first so the unsigned arithmetic does not wrap.
            let w00 = 16 + w11 - fs - ft;
            let v0 = (js + jt * bm.width) as usize;
            let infill = |w: &[u32]| {
                // Grid positions past the last weight are only reached with a zero factor.
                let at = |n: usize| w.get(n).copied().unwrap_or(0);
                (at(v0) * w00
                    + at(v0 + 1) * w01
                    + at(v0 + bm.width as usize) * w10
                    + at(v0 + bm.width as usize + 1) * w11
                    + 8)
                    >> 4
            };
            let i0 = infill(&plane_weights[0]);
            let i1 = if bm.dual_plane { infill(&plane_weights[1]) } else { i0 };
            let mut c = [0u16; 4];
            for ch in 0..4 {
                let i = if ccs == Some(ch) { i1 } else { i0 };
                let (c0, c1) = (e0[ch] as u32, e1[ch] as u32);
                let (c0, c1) =
                    if srgb { ((c0 << 8) | 0x80, (c1 << 8) | 0x80) } else { ((c0 << 8) | c0, (c1 << 8) | c1) };
                c[ch] = ((c0 * (64 - i) + c1 * i + 32) / 64) as u16;
            }
            out.push(Some(c));
        }
    }
    Ok(out)
}

/// The `decode_float16` result of a UNORM16 value in linear LDR operation mode (section 23.18.1): 1.0 for 65535,
/// otherwise `C / 65536` converted to FP16 with round-to-zero. Returns the IEEE binary16 bits.
pub fn to_float16(c: u16) -> u16 {
    if c == 0xffff {
        return 0x3c00;
    }
    if c == 0 {
        return 0;
    }
    // c / 2^16 = 1.m × 2^(top - 16) with `top` the index of the highest set bit.
    let top = 15 - c.leading_zeros();
    let exponent = top as i32 - 16 + 15;
    if exponent <= 0 {
        // Subnormal: value = mantissa × 2^-24, and c × 2^-16 = (c << 8) × 2^-24 exactly.
        return c << 8;
    }
    let mantissa = if top >= 10 { (c >> (top - 10)) & 0x3ff } else { (c << (10 - top)) & 0x3ff };
    ((exponent as u16) << 10) | mantissa
}

/// The 12-bit unsigned normalized value real Metal returns for a linear LDR texel: the top 12 bits of `C`, read as
/// `(C >> 4) / 4095`.
///
/// This is not one of the decode modes of section 23.5. It is what an Apple M4 under macOS 27.0.1 returns from
/// `read()` of `ASTC_*_LDR` textures: every decoded channel of the Metal reference outputs `format.ASTC_*_LDR.blocks`
/// equals `(C >> 4) / 4095` rounded to binary32, with `C` from [`decode_block`] (the error color's channels, 0 and
/// 65535, give 0.0 and 1.0). The A10X's conversion is not measured.
pub fn to_unorm12(c: u16) -> u16 {
    c >> 4
}

/// The sRGB operation mode's 8-bit result (section 23.18.1): the top 8 bits of `C`, to be converted with the sRGB EOTF
/// for R, G and B and used directly for A.
pub fn to_srgb8(c: u16) -> u8 {
    (c >> 8) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trit_and_quint_tables_are_bijective_over_valid_codes() {
        // Every trit tuple 0..3^5 and quint tuple 0..5^3 is produced by some packed value (section 23.12 states that
        // all 256 and 128 values decode; a few are duplicates).
        let mut seen = std::collections::HashSet::new();
        for t in 0..256 {
            let v = trits(t);
            assert!(v.iter().all(|&x| x < 3));
            seen.insert(v);
        }
        assert_eq!(seen.len(), 243);
        let mut seen = std::collections::HashSet::new();
        for q in 0..128 {
            let v = quints(q);
            assert!(v.iter().all(|&x| x < 5));
            seen.insert(v);
        }
        assert_eq!(seen.len(), 125);
    }

    #[test]
    fn weight_unquantization_is_monotonic_and_spans_0_to_64() {
        for row in WEIGHT_RANGES.iter() {
            for range in row.iter().flatten() {
                let levels = (3u32.pow(range.trits) * 5u32.pow(range.quints)) << range.bits;
                let mut values: Vec<u32> = (0..levels)
                    .map(|v| {
                        let (d, m) = (v >> range.bits, v & ((1 << range.bits) - 1));
                        unquantize_weight(*range, (d, m))
                    })
                    .collect();
                values.sort();
                assert_eq!(values[0], 0);
                assert_eq!(*values.last().unwrap(), 64);
                values.dedup();
                assert_eq!(values.len() as u32, levels, "{range:?}");
            }
        }
    }

    #[test]
    fn color_unquantization_spans_0_to_255() {
        for range in COLOR_RANGES {
            let levels = (3u32.pow(range.trits) * 5u32.pow(range.quints)) << range.bits;
            let mut values: Vec<u32> =
                (0..levels).map(|v| unquantize_color(range, (v >> range.bits, v & ((1 << range.bits) - 1)))).collect();
            values.sort();
            values.dedup();
            assert_eq!(values.len() as u32, levels, "{range:?}");
            assert_eq!((values[0], *values.last().unwrap()), (0, 255), "{range:?}");
        }
    }

    #[test]
    fn infill_never_reads_past_the_grid_with_a_nonzero_factor() {
        for (bw, bh) in FOOTPRINTS {
            for gw in 2..=bw {
                for gh in 2..=bh {
                    let ds = (1024 + bw / 2) / (bw - 1);
                    let dt = (1024 + bh / 2) / (bh - 1);
                    for y in 0..bh {
                        for x in 0..bw {
                            let gs = (ds * x * (gw - 1) + 32) >> 6;
                            let gt = (dt * y * (gh - 1) + 32) >> 6;
                            let (js, fs, jt, ft) = (gs >> 4, gs & 15, gt >> 4, gt & 15);
                            assert!(js < gw && jt < gh);
                            if js == gw - 1 {
                                assert_eq!(fs, 0);
                            }
                            if jt == gh - 1 {
                                assert_eq!(ft, 0);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn float16_conversion_rounds_toward_zero() {
        assert_eq!(to_float16(0xffff), 0x3c00);
        assert_eq!(to_float16(0x8000), 0x3800); // 0.5
        assert_eq!(to_float16(1), 1 << 8); // 2^-16 is subnormal: 256 × 2^-24
        assert_eq!(to_float16(4), 0x0400); // 2^-14, the smallest normal
        // 0xfffe / 65536 = 0.999969...: the FP16 values below 1.0 are spaced 2^-11; round toward zero gives 0x3bff.
        assert_eq!(to_float16(0xfffe), 0x3bff);
        for c in 1..=0xfffeu32 {
            let h = to_float16(c as u16);
            let v = f16_to_f64(h);
            let exact = c as f64 / 65536.0;
            assert!(v <= exact, "{c}");
            let next = f16_to_f64(h + 1);
            assert!(next > exact, "{c}");
        }
    }

    pub(crate) fn f16_to_f64(h: u16) -> f64 {
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f64;
        if e == 0 { m * 2f64.powi(-24) } else { (1.0 + m / 1024.0) * 2f64.powi(e - 15) }
    }

    #[test]
    fn void_extent_block() {
        // A constant-color LDR block with all-ones coordinates.
        let mut v: u128 = 0x1fc | (0b11 << 10);
        for (i, shift) in [12, 25, 38, 51].into_iter().enumerate() {
            let _ = i;
            v |= 0x1fff << shift;
        }
        v |= (0x1234u128 << 64) | (0x5678u128 << 80) | (0x9abcu128 << 96) | (0xffffu128 << 112);
        let out = decode_block(&v.to_le_bytes(), 4, 4, OperationMode::LinearLdr).unwrap();
        assert!(out.iter().all(|t| *t == Some([0x1234, 0x5678, 0x9abc, 0xffff])));
        // Reserved bit 10 cleared: illegal.
        let out = decode_block(&(v & !(1 << 10)).to_le_bytes(), 4, 4, OperationMode::LinearLdr).unwrap();
        assert!(out.iter().all(|t| t.is_none()));
        // HDR flag: error color in LDR operation modes.
        let out = decode_block(&(v | (1 << 9)).to_le_bytes(), 4, 4, OperationMode::Srgb).unwrap();
        assert!(out.iter().all(|t| t.is_none()));
        // Minimum s not below maximum s, coordinates not all ones: illegal.
        let w = v & !(0x1fffu128 << 25);
        let out = decode_block(&w.to_le_bytes(), 4, 4, OperationMode::LinearLdr).unwrap();
        assert!(out.iter().all(|t| t.is_none()));
    }

    #[test]
    fn unsupported_footprint_is_refused() {
        assert_eq!(
            decode_block(&[0; 16], 4, 2, OperationMode::LinearLdr),
            Err(UnsupportedFootprint { width: 4, height: 2 })
        );
    }
}
