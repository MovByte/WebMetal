//! Primitive encoding and decoding shared by every generated record.
//!
//! The format is little-endian and canonical: every value has exactly one encoding. A record is an 8-byte header
//! (`opcode: u16`, `flags: u16`, `length: u32`, where `length` counts the whole record and is a multiple of 8) followed by
//! the record's fields in schema order. Each field starts at the next offset, measured from the start of the record, that
//! is a multiple of its type's alignment; skipped bytes and the bytes after the last field up to `length` are zero. The
//! decoder rejects anything else, so a record that decodes re-encodes to the same bytes.

use core::fmt;
use core::num::NonZeroU64;

/// Size of the record header.
pub const HEADER_LEN: usize = 8;
/// Records start and end on this alignment.
pub const RECORD_ALIGN: usize = 8;
/// Count field value that encodes a nil string.
pub const NIL_COUNT: u32 = 0xffff_ffff;
/// Bit set in a reply's opcode; the rest is the request's opcode.
pub const REPLY_BIT: u16 = 0x8000;

/// A guest-allocated object handle. Handles are unique for the life of a connection and never reused; 0 encodes nil.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Handle(pub NonZeroU64);

impl Handle {
    /// The handle with this raw value, or `None` for 0.
    pub fn new(raw: u64) -> Option<Handle> {
        NonZeroU64::new(raw).map(Handle)
    }

    /// The raw value.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

/// A virtual address in the task that sent the record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GuestAddr(pub u64);

/// An IOSurfaceID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceId(pub u32);

/// Why a record or stream did not decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeError {
    /// Byte offset, from the start of the record or stream being decoded, where the problem was found.
    pub offset: usize,
    /// What was wrong.
    pub kind: DecodeErrorKind,
}

/// The kinds of decode failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeErrorKind {
    /// The bytes end inside a header, field or record.
    Truncated,
    /// The header's length is shorter than a header, not a multiple of 8, longer than the bytes available, or not the
    /// canonical length of the decoded fields.
    BadLength(u32),
    /// The header's flags are not zero.
    BadFlags(u16),
    /// No record of the expected channel has this opcode.
    UnknownOpcode(u16),
    /// The opcode belongs to a different channel than the one being decoded.
    WrongChannel(u16),
    /// A padding byte is not zero.
    NonZeroPadding,
    /// A bool or presence byte is not 0 or 1.
    BadBool(u8),
    /// A handle that cannot be nil is 0.
    NullHandle(&'static str),
    /// A string is not valid UTF-8, or a C string contains a NUL byte.
    BadString(&'static str),
    /// An element count exceeds the bytes left in the record.
    BadCount(&'static str),
    /// A length constraint between two fields does not hold.
    Constraint(&'static str),
    /// A property-list value tag is unknown, nested too deeply, or a dictionary's keys are not strictly ascending.
    BadValue,
    /// A failure record breaks the rules of [`crate::Failure`].
    BadFailure,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "wire decode error at byte {}: {:?}", self.offset, self.kind)
    }
}

impl std::error::Error for DecodeError {}

/// Why a value could not be encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// A length constraint between two fields does not hold; names the constrained field.
    Constraint(&'static str),
    /// A string, byte string or array has more than `u32::MAX - 1` elements.
    TooLong(&'static str),
    /// The record would exceed the largest length a header can state.
    RecordTooLarge,
    /// A C string contains a NUL byte.
    BadString(&'static str),
    /// A property-list value is nested deeper than [`crate::VALUE_DEPTH_LIMIT`] or a dictionary repeats a key.
    BadValue,
    /// A failure value breaks the rules of [`crate::Failure`].
    BadFailure,
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "wire encode error: {:?}", self)
    }
}

impl std::error::Error for EncodeError {}

/// Appends one record to a byte vector.
pub struct Writer<'a> {
    out: &'a mut Vec<u8>,
    start: usize,
}

impl<'a> Writer<'a> {
    /// Start a record with this opcode at the end of `out`, which must already be 8-byte aligned in length.
    pub fn begin(out: &'a mut Vec<u8>, opcode: u16) -> Writer<'a> {
        debug_assert_eq!(out.len() % RECORD_ALIGN, 0);
        let start = out.len();
        out.extend_from_slice(&opcode.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        Writer { out, start }
    }

    fn align(&mut self, align: usize) {
        while !(self.out.len() - self.start).is_multiple_of(align) {
            self.out.push(0);
        }
    }

    pub fn u8(&mut self, v: u8) {
        self.out.push(v);
    }

    pub fn bool(&mut self, v: bool) {
        self.out.push(v as u8);
    }

    pub fn u16(&mut self, v: u16) {
        self.align(2);
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.align(4);
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn i32(&mut self, v: i32) {
        self.u32(v as u32);
    }

    pub fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }

    pub fn u64(&mut self, v: u64) {
        self.align(8);
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn i64(&mut self, v: i64) {
        self.u64(v as u64);
    }

    pub fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }

    pub fn handle(&mut self, v: Handle) {
        self.u64(v.get());
    }

    pub fn opt_handle(&mut self, v: Option<Handle>) {
        self.u64(v.map_or(0, Handle::get));
    }

    /// An element count. `u32::MAX` is reserved for nil, so the largest count is `u32::MAX - 1`.
    pub fn count(&mut self, n: usize, field: &'static str) -> Result<(), EncodeError> {
        if n >= NIL_COUNT as usize {
            return Err(EncodeError::TooLong(field));
        }
        self.u32(n as u32);
        Ok(())
    }

    pub fn raw(&mut self, bytes: &[u8]) {
        self.out.extend_from_slice(bytes);
    }

    pub fn str(&mut self, v: &str, field: &'static str) -> Result<(), EncodeError> {
        self.count(v.len(), field)?;
        self.raw(v.as_bytes());
        Ok(())
    }

    pub fn opt_str(&mut self, v: Option<&str>, field: &'static str) -> Result<(), EncodeError> {
        match v {
            None => {
                self.u32(NIL_COUNT);
                Ok(())
            }
            Some(s) => self.str(s, field),
        }
    }

    pub fn opt_cstr(&mut self, v: Option<&[u8]>, field: &'static str) -> Result<(), EncodeError> {
        match v {
            None => {
                self.u32(NIL_COUNT);
                Ok(())
            }
            Some(s) => {
                if s.contains(&0) {
                    return Err(EncodeError::BadString(field));
                }
                self.bytes(s, field)
            }
        }
    }

    pub fn bytes(&mut self, v: &[u8], field: &'static str) -> Result<(), EncodeError> {
        self.count(v.len(), field)?;
        self.raw(v);
        Ok(())
    }

    /// Pad to the record alignment and write the length into the header.
    pub fn finish(mut self) -> Result<(), EncodeError> {
        self.align(RECORD_ALIGN);
        let len = self.out.len() - self.start;
        if len > (u32::MAX as usize & !(RECORD_ALIGN - 1)) {
            self.out.truncate(self.start);
            return Err(EncodeError::RecordTooLarge);
        }
        self.out[self.start + 4..self.start + 8].copy_from_slice(&(len as u32).to_le_bytes());
        Ok(())
    }
}

/// Reads the fields of one record.
pub struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    base: usize,
}

impl<'a> Reader<'a> {
    /// Read the fields of the record `bytes` (header included). `base` is the record's offset in the enclosing stream,
    /// used only to report error offsets.
    pub fn new(bytes: &'a [u8], base: usize) -> Reader<'a> {
        Reader { bytes, at: HEADER_LEN, base }
    }

    pub fn error(&self, kind: DecodeErrorKind) -> DecodeError {
        DecodeError { offset: self.base + self.at, kind }
    }

    fn align(&mut self, align: usize) -> Result<(), DecodeError> {
        while !self.at.is_multiple_of(align) {
            let b = *self.bytes.get(self.at).ok_or_else(|| self.error(DecodeErrorKind::Truncated))?;
            if b != 0 {
                return Err(self.error(DecodeErrorKind::NonZeroPadding));
            }
            self.at += 1;
        }
        Ok(())
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len());
        match end {
            Some(end) => {
                let s = &self.bytes[self.at..end];
                self.at = end;
                Ok(s)
            }
            None => Err(self.error(DecodeErrorKind::Truncated)),
        }
    }

    /// Current offset within the record.
    pub fn position(&self) -> usize {
        self.at
    }

    /// Return to an offset previously returned by [`Reader::position`].
    pub fn set_position(&mut self, at: usize) {
        self.at = at;
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.at
    }

    pub fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    pub fn bool(&mut self) -> Result<bool, DecodeError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            b => {
                self.at -= 1;
                Err(self.error(DecodeErrorKind::BadBool(b)))
            }
        }
    }

    pub fn u16(&mut self) -> Result<u16, DecodeError> {
        self.align(2)?;
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32, DecodeError> {
        self.align(4)?;
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn i32(&mut self) -> Result<i32, DecodeError> {
        Ok(self.u32()? as i32)
    }

    pub fn f32(&mut self) -> Result<f32, DecodeError> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub fn u64(&mut self) -> Result<u64, DecodeError> {
        self.align(8)?;
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn i64(&mut self) -> Result<i64, DecodeError> {
        Ok(self.u64()? as i64)
    }

    pub fn f64(&mut self) -> Result<f64, DecodeError> {
        Ok(f64::from_bits(self.u64()?))
    }

    pub fn handle(&mut self, field: &'static str) -> Result<Handle, DecodeError> {
        let v = self.u64()?;
        Handle::new(v).ok_or_else(|| DecodeError { offset: self.base + self.at - 8, kind: DecodeErrorKind::NullHandle(field) })
    }

    pub fn opt_handle(&mut self) -> Result<Option<Handle>, DecodeError> {
        Ok(Handle::new(self.u64()?))
    }

    /// An element count, checked against the bytes left in the record (every element occupies at least one byte).
    pub fn count(&mut self, field: &'static str) -> Result<usize, DecodeError> {
        let n = self.u32()?;
        if n == NIL_COUNT || n as usize > self.remaining() {
            self.at -= 4;
            return Err(self.error(DecodeErrorKind::BadCount(field)));
        }
        Ok(n as usize)
    }

    pub fn raw(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        self.take(n)
    }

    pub fn str(&mut self, field: &'static str) -> Result<String, DecodeError> {
        let n = self.count(field)?;
        let at = self.at;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| DecodeError { offset: self.base + at, kind: DecodeErrorKind::BadString(field) })
    }

    pub fn opt_str(&mut self, field: &'static str) -> Result<Option<String>, DecodeError> {
        if self.peek_nil()? {
            return Ok(None);
        }
        self.str(field).map(Some)
    }

    pub fn opt_cstr(&mut self, field: &'static str) -> Result<Option<Vec<u8>>, DecodeError> {
        if self.peek_nil()? {
            return Ok(None);
        }
        let at = self.at;
        let bytes = self.bytes(field)?;
        if bytes.contains(&0) {
            return Err(DecodeError { offset: self.base + at, kind: DecodeErrorKind::BadString(field) });
        }
        Ok(Some(bytes))
    }

    /// Consume a nil count if one is next.
    fn peek_nil(&mut self) -> Result<bool, DecodeError> {
        let save = self.at;
        if self.u32()? == NIL_COUNT {
            return Ok(true);
        }
        self.at = save;
        Ok(false)
    }

    pub fn bytes(&mut self, field: &'static str) -> Result<Vec<u8>, DecodeError> {
        let n = self.count(field)?;
        Ok(self.take(n)?.to_vec())
    }

    /// Check the canonical tail: zero padding to the record alignment, and nothing after it.
    pub fn finish(mut self) -> Result<(), DecodeError> {
        self.align(RECORD_ALIGN)?;
        if self.at != self.bytes.len() {
            let len = self.bytes.len() as u32;
            return Err(DecodeError { offset: self.base, kind: DecodeErrorKind::BadLength(len) });
        }
        Ok(())
    }
}

/// Split the next record off `bytes` (which starts at stream offset `base`): checks the header and returns
/// (opcode, the whole record including its header).
pub fn split_record(bytes: &[u8], base: usize) -> Result<(u16, &[u8]), DecodeError> {
    if bytes.len() < HEADER_LEN {
        return Err(DecodeError { offset: base, kind: DecodeErrorKind::Truncated });
    }
    let opcode = u16::from_le_bytes([bytes[0], bytes[1]]);
    let flags = u16::from_le_bytes([bytes[2], bytes[3]]);
    let len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if flags != 0 {
        return Err(DecodeError { offset: base + 2, kind: DecodeErrorKind::BadFlags(flags) });
    }
    if (len as usize) < HEADER_LEN || !(len as usize).is_multiple_of(RECORD_ALIGN) || len as usize > bytes.len() {
        return Err(DecodeError { offset: base + 4, kind: DecodeErrorKind::BadLength(len) });
    }
    Ok((opcode, &bytes[..len as usize]))
}
