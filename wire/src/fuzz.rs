//! Deterministic random values for round-trip tests.
//!
//! The generators draw from splitmix64 (Steele, Lea and Flood, "Fast splittable pseudorandom number generators",
//! OOPSLA 2014) in exactly the order the C++ generators in `generated/WebMetalWireFuzz.h` do, so the same seed produces
//! the same records in both languages. Every value produced is valid: it encodes without error.

use crate::codec::Handle;
use crate::generated::Failure;
use crate::value::Value;

/// splitmix64 state and the value draws shared with the C++ generator.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    pub fn u8(&mut self) -> u8 {
        self.next_u64() as u8
    }

    pub fn u16(&mut self) -> u16 {
        self.next_u64() as u16
    }

    pub fn u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    pub fn u64(&mut self) -> u64 {
        self.next_u64()
    }

    pub fn i32(&mut self) -> i32 {
        self.u32() as i32
    }

    pub fn i64(&mut self) -> i64 {
        self.next_u64() as i64
    }

    pub fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }

    pub fn f64(&mut self) -> f64 {
        f64::from_bits(self.next_u64())
    }

    pub fn boolean(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    pub fn handle(&mut self) -> Handle {
        let v = self.next_u64();
        Handle::new(if v == 0 { 1 } else { v }).unwrap()
    }

    pub fn opt_handle(&mut self) -> Option<Handle> {
        if self.below(4) == 0 { None } else { Some(self.handle()) }
    }

    pub fn string_len(&mut self) -> u64 {
        self.below(17)
    }

    pub fn bytes_len(&mut self) -> u64 {
        self.below(33)
    }

    pub fn array_len(&mut self) -> u64 {
        self.below(5)
    }

    fn code_point(&mut self) -> char {
        let cp = match self.below(4) {
            0 => 0x20 + self.below(0x5f) as u32,
            1 => 0x80 + self.below(0x780) as u32,
            2 => {
                let cp = 0x800 + self.below(0xf000) as u32;
                if cp >= 0xd800 { cp + 0x800 } else { cp }
            }
            _ => 0x10000 + self.below(0x10_0000) as u32,
        };
        char::from_u32(cp).expect("generated code point is a scalar value")
    }

    /// A string of `code_points` random scalar values.
    pub fn string(&mut self, code_points: u64) -> String {
        (0..code_points).map(|_| self.code_point()).collect()
    }

    pub fn opt_string(&mut self) -> Option<String> {
        if self.below(4) == 0 {
            return None;
        }
        let n = self.string_len();
        Some(self.string(n))
    }

    /// A C string of `n` non-NUL bytes.
    pub fn cstr(&mut self, n: u64) -> Vec<u8> {
        (0..n).map(|_| 1 + self.below(255) as u8).collect()
    }

    pub fn opt_cstr(&mut self) -> Option<Vec<u8>> {
        if self.below(4) == 0 {
            return None;
        }
        let n = self.string_len();
        Some(self.cstr(n))
    }

    pub fn bytes(&mut self, n: u64) -> Vec<u8> {
        (0..n).map(|_| self.u8()).collect()
    }

    pub fn fixed<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        for b in &mut out {
            *b = self.u8();
        }
        out
    }

    pub fn opt_value(&mut self) -> Option<Value> {
        if self.below(4) == 0 { None } else { Some(self.value(1)) }
    }

    fn value(&mut self, depth: usize) -> Value {
        match self.below(if depth < 3 { 9 } else { 7 }) {
            0 => Value::Bool(self.boolean()),
            1 => Value::Int(self.i64()),
            2 => Value::UInt(self.u64()),
            3 => Value::Float(self.f32()),
            4 => Value::Double(self.f64()),
            5 => {
                let n = self.string_len();
                Value::String(self.string(n))
            }
            6 => {
                let n = self.bytes_len();
                Value::Data(self.bytes(n))
            }
            7 => {
                let n = self.below(4);
                Value::Array((0..n).map(|_| self.value(depth + 1)).collect())
            }
            _ => {
                let n = self.below(4);
                let mut entries: Vec<(String, Value)> = Vec::new();
                for _ in 0..n {
                    let len = self.string_len();
                    let key = self.string(len);
                    let item = self.value(depth + 1);
                    entries.push((key, item));
                }
                entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
                entries.dedup_by(|later, earlier| later.0 == earlier.0);
                Value::Dictionary(entries)
            }
        }
    }
}

fn failure_with_status(rng: &mut Rng, status: u32) -> Failure {
    if status == 0 {
        return Failure { status, domain: None, code: 0, message: String::new() };
    }
    let (domain, code) = if status == 1 {
        let n = rng.string_len();
        let domain = rng.string(n);
        (Some(domain), rng.i64())
    } else {
        (None, 0)
    };
    let n = rng.string_len();
    Failure { status, domain, code, message: rng.string(n) }
}

/// A random valid [`Failure`], status 0 included.
pub fn random_failure(rng: &mut Rng) -> Failure {
    let status = rng.below(5) as u32;
    failure_with_status(rng, status)
}

/// A random valid [`Failure`] with a non-zero status.
pub fn random_error_failure(rng: &mut Rng) -> Failure {
    let status = 1 + rng.below(4) as u32;
    failure_with_status(rng, status)
}
