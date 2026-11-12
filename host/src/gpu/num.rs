//! The IDL numeric types whose domain no Rust primitive has.
//!
//! WebIDL converts a JavaScript value to an IDL type before a method runs, throwing a `TypeError` when the value is
//! outside the type's domain. These types perform that conversion when they are constructed, so a value of one is
//! always a value the IDL type accepts, and the methods that take them cannot fail for that reason.

use core::fmt;

use super::api::{Exception, ExceptionKind};

/// `[EnforceRange] unsigned long long` (`GPUSize64`): an integer in 0 ..= 2^53 − 1. WebIDL's EnforceRange conversion
/// limits `unsigned long long` to the integers a JavaScript number holds exactly.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Size64(u64);

impl Size64 {
    /// 2^53 − 1, the largest value EnforceRange accepts for `unsigned long long`.
    pub const MAX: u64 = (1 << 53) - 1;

    /// The value, or the `TypeError` WebIDL throws for it.
    pub fn new(v: u64) -> Result<Size64, Exception> {
        if v <= Self::MAX {
            Ok(Size64(v))
        } else {
            Err(Exception::new(
                ExceptionKind::TypeError,
                format!("{v} is outside the range of [EnforceRange] unsigned long long"),
            ))
        }
    }

    /// The value, for constants; panics (at compile time in a constant context) if it is out of range.
    pub const fn new_const(v: u64) -> Size64 {
        assert!(v <= Self::MAX, "Size64 out of range");
        Size64(v)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u32> for Size64 {
    fn from(v: u32) -> Size64 {
        Size64(u64::from(v))
    }
}

impl fmt::Debug for Size64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

/// The restricted IDL `float`: a finite single-precision value (WebIDL throws `TypeError` for NaN and infinities).
#[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Float32(f32);

impl Float32 {
    pub fn new(v: f32) -> Result<Float32, Exception> {
        if v.is_finite() {
            Ok(Float32(v))
        } else {
            Err(Exception::new(ExceptionKind::TypeError, format!("{v} is not a finite float")))
        }
    }

    pub const fn new_const(v: f32) -> Float32 {
        assert!(v.is_finite(), "Float32 must be finite");
        Float32(v)
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

impl fmt::Debug for Float32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

/// The restricted IDL `double`: a finite double-precision value (WebIDL throws `TypeError` for NaN and infinities).
#[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Float64(f64);

impl Float64 {
    pub fn new(v: f64) -> Result<Float64, Exception> {
        if v.is_finite() {
            Ok(Float64(v))
        } else {
            Err(Exception::new(ExceptionKind::TypeError, format!("{v} is not a finite double")))
        }
    }

    pub const fn new_const(v: f64) -> Float64 {
        assert!(v.is_finite(), "Float64 must be finite");
        Float64(v)
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Debug for Float64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforce_range_limit() {
        assert_eq!(Size64::new((1 << 53) - 1).unwrap().get(), (1 << 53) - 1);
        assert_eq!(Size64::new(1 << 53).unwrap_err().kind, ExceptionKind::TypeError);
        assert_eq!(Size64::new(u64::MAX).unwrap_err().kind, ExceptionKind::TypeError);
    }

    #[test]
    fn restricted_floats() {
        assert!(Float32::new(f32::NAN).is_err());
        assert!(Float32::new(f32::INFINITY).is_err());
        assert!(Float64::new(f64::NEG_INFINITY).is_err());
        assert_eq!(Float32::new(-0.0).unwrap().get().to_bits(), (-0.0f32).to_bits());
        assert_eq!(Float64::new(1.5).unwrap().get(), 1.5);
    }
}
