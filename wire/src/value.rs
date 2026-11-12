//! Property-list values: the wire form of Foundation dictionaries, arrays, strings, numbers and data that Metal
//! descriptors carry (driver and compiler option dictionaries, plugin data).
//!
//! Encoding: a one-byte tag, then the payload aligned like the scalar fields of a record.
//!
//! | tag | value                    | payload                                              |
//! | --- | ------------------------ | ---------------------------------------------------- |
//! | 0   | nil (top level only)     | none                                                 |
//! | 1   | boolean                  | one byte, 0 or 1                                     |
//! | 2   | signed integer NSNumber  | i64                                                  |
//! | 3   | unsigned integer NSNumber| u64                                                  |
//! | 4   | float NSNumber           | f32                                                  |
//! | 5   | double NSNumber          | f64                                                  |
//! | 6   | string                   | u32 byte count, UTF-8 bytes                          |
//! | 7   | data                     | u32 byte count, bytes                                |
//! | 8   | array                    | u32 element count, values                            |
//! | 9   | dictionary               | u32 entry count, (u32 key byte count, key, value)... |
//!
//! Dictionary keys are strings, strictly ascending by their UTF-8 bytes (so unique). Nesting is limited to
//! [`VALUE_DEPTH_LIMIT`] levels, counting the top-level value as 1.

use crate::codec::{DecodeError, DecodeErrorKind, EncodeError, Reader, Writer};

/// Deepest nesting a value may have; the top-level value is depth 1.
pub const VALUE_DEPTH_LIMIT: usize = 32;

const TAG_NIL: u8 = 0;
const TAG_BOOL: u8 = 1;
const TAG_INT: u8 = 2;
const TAG_UINT: u8 = 3;
const TAG_FLOAT: u8 = 4;
const TAG_DOUBLE: u8 = 5;
const TAG_STRING: u8 = 6;
const TAG_DATA: u8 = 7;
const TAG_ARRAY: u8 = 8;
const TAG_DICTIONARY: u8 = 9;

/// A property-list value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f32),
    Double(f64),
    String(String),
    Data(Vec<u8>),
    Array(Vec<Value>),
    /// Entries in strictly ascending key order.
    Dictionary(Vec<(String, Value)>),
}

/// Encode an optional value.
pub fn encode_opt_value(w: &mut Writer<'_>, v: Option<&Value>) -> Result<(), EncodeError> {
    match v {
        None => {
            w.u8(TAG_NIL);
            Ok(())
        }
        Some(v) => encode_value(w, v, 1),
    }
}

fn encode_value(w: &mut Writer<'_>, v: &Value, depth: usize) -> Result<(), EncodeError> {
    if depth > VALUE_DEPTH_LIMIT {
        return Err(EncodeError::BadValue);
    }
    match v {
        Value::Bool(b) => {
            w.u8(TAG_BOOL);
            w.bool(*b);
        }
        Value::Int(i) => {
            w.u8(TAG_INT);
            w.i64(*i);
        }
        Value::UInt(u) => {
            w.u8(TAG_UINT);
            w.u64(*u);
        }
        Value::Float(f) => {
            w.u8(TAG_FLOAT);
            w.f32(*f);
        }
        Value::Double(d) => {
            w.u8(TAG_DOUBLE);
            w.f64(*d);
        }
        Value::String(s) => {
            w.u8(TAG_STRING);
            w.str(s, "value")?;
        }
        Value::Data(d) => {
            w.u8(TAG_DATA);
            w.bytes(d, "value")?;
        }
        Value::Array(items) => {
            w.u8(TAG_ARRAY);
            w.count(items.len(), "value")?;
            for item in items {
                encode_value(w, item, depth + 1)?;
            }
        }
        Value::Dictionary(entries) => {
            for pair in entries.windows(2) {
                if pair[0].0.as_bytes() >= pair[1].0.as_bytes() {
                    return Err(EncodeError::BadValue);
                }
            }
            w.u8(TAG_DICTIONARY);
            w.count(entries.len(), "value")?;
            for (key, value) in entries {
                w.str(key, "value")?;
                encode_value(w, value, depth + 1)?;
            }
        }
    }
    Ok(())
}

/// Decode an optional value.
pub fn decode_opt_value(r: &mut Reader<'_>) -> Result<Option<Value>, DecodeError> {
    let tag = r.u8()?;
    if tag == TAG_NIL {
        return Ok(None);
    }
    decode_tagged(r, tag, 1).map(Some)
}

fn decode_tagged(r: &mut Reader<'_>, tag: u8, depth: usize) -> Result<Value, DecodeError> {
    if depth > VALUE_DEPTH_LIMIT {
        return Err(r.error(DecodeErrorKind::BadValue));
    }
    Ok(match tag {
        TAG_BOOL => Value::Bool(r.bool()?),
        TAG_INT => Value::Int(r.i64()?),
        TAG_UINT => Value::UInt(r.u64()?),
        TAG_FLOAT => Value::Float(r.f32()?),
        TAG_DOUBLE => Value::Double(r.f64()?),
        TAG_STRING => Value::String(r.str("value")?),
        TAG_DATA => Value::Data(r.bytes("value")?),
        TAG_ARRAY => {
            let n = r.count("value")?;
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                let tag = r.u8()?;
                items.push(decode_tagged(r, tag, depth + 1)?);
            }
            Value::Array(items)
        }
        TAG_DICTIONARY => {
            let n = r.count("value")?;
            let mut entries: Vec<(String, Value)> = Vec::with_capacity(n);
            for _ in 0..n {
                let key = r.str("value")?;
                if let Some((last, _)) = entries.last()
                    && last.as_bytes() >= key.as_bytes()
                {
                    return Err(r.error(DecodeErrorKind::BadValue));
                }
                let tag = r.u8()?;
                let value = decode_tagged(r, tag, depth + 1)?;
                entries.push((key, value));
            }
            Value::Dictionary(entries)
        }
        _ => return Err(r.error(DecodeErrorKind::BadValue)),
    })
}
