//! The WebGPU binding layer: the executor's only way to WebGPU.
//!
//! - [`api`]: the WebGPU IDL as Rust traits, with the types of [`idl`] (generated from the IDL index of the pinned
//!   specification by `host/gen/gpugen.py`) and the numeric domains of [`num`].
//! - [`negotiate`]: which features and limits to request from an adapter, from the tables of `tables/limits.toml`
//!   ([`tables`]), and the check that the device received them.

pub mod api;
#[rustfmt::skip]
pub mod idl;
pub mod negotiate;
pub mod num;
#[rustfmt::skip]
pub mod tables;

pub use api::*;
pub use num::{Float32, Float64, Size64};

/// HTML's `PredefinedColorSpace` enum, which WebGPU dictionaries use. It is defined by the HTML standard, not by
/// WebGPU; the binding carries the string and the implementation validates it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PredefinedColorSpace(String);

impl PredefinedColorSpace {
    pub fn new(value: impl Into<String>) -> PredefinedColorSpace {
        PredefinedColorSpace(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
