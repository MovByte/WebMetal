//! Pixel formats.
//!
//! - [`table`]: the Metal and WebGPU format lists, generated from `tables/formats.toml`, `tables/limits.toml` and the
//!   A10X device profile.
//! - [`webgpu`]: WebGPU format capabilities evaluated against a host device's features.
//! - [`etc2`], [`astc`], [`pvrtc`]: software decoders of the compressed formats, and [`decode`], which decodes a
//!   Metal format's image with them.

pub mod astc;
pub mod decode;
pub mod etc2;
pub mod packed;
pub mod pvrtc;
#[rustfmt::skip]
pub mod table;
pub mod webgpu;

#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod testdata;
