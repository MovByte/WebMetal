//! Pixel formats.
//!
//! - [`table`]: the Metal and WebGPU format lists, generated from `tables/formats.toml`, `tables/limits.toml` and the
//!   A10X device profile.
//! - [`webgpu`]: WebGPU format capabilities evaluated against a host device's features.
//! - [`handling`]: how each Metal format the A10X reports is provided (natively, emulated or refused), resolved
//!   against a host's features, and the capability checks of the creating and using calls.
//! - [`etc2`], [`astc`], [`pvrtc`]: software decoders of the compressed formats, and [`decode`], which decodes a
//!   Metal format's image with them.
//! - [`packed`]: bit layouts and exact conversions of the formats stored in another WebGPU format.
//! - [`storage`]: the stored texels of emulated textures, to and from Metal's bytes.
//! - [`reinterpret`]: views in another pixel format.
//! - [`ycbcr`]: the 4:2:2 formats and textures over IOSurface planes.

pub mod astc;
pub mod decode;
pub mod etc2;
pub mod handling;
pub mod packed;
pub mod pvrtc;
pub mod reinterpret;
pub mod storage;
#[rustfmt::skip]
pub mod table;
pub mod webgpu;
pub mod ycbcr;

#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod testdata;
