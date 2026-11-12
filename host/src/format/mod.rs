//! Pixel formats.
//!
//! - [`table`]: the Metal and WebGPU format lists, generated from `tables/formats.toml`, `tables/limits.toml` and the
//!   A10X device profile.
//! - [`webgpu`]: WebGPU format capabilities evaluated against a host device's features.

#[rustfmt::skip]
pub mod table;
pub mod webgpu;
