//! The WebMetal host executor: decodes the wire protocol (`webmetal-wire`), validates it against Metal's rules and
//! replays it on one WebGPU device.
//!
//! [`embed`] is the interface the embedding environment (the kernel side) provides to the executor.

pub mod embed;
