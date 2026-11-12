//! The WebMetal host executor: decodes the wire protocol (`webmetal-wire`), validates it against Metal's rules and
//! replays it on one WebGPU device.
//!
//! [`embed`] is the interface the embedding environment (the kernel side) provides to the executor; [`exec`] is the core
//! executor; [`gpu`] is the executor's binding to WebGPU; [`mem`] keeps buffers, textures and IOSurfaces coherent
//! between guest memory and WebGPU; [`blit`] and [`compute`] record blit and compute encoders, and [`render`] replays
//! render encoders; [`pipeline`] serves shader modules, functions, pipelines and binary archives.

pub mod blit;
pub mod compute;
pub mod embed;
pub mod exec;
pub mod format;
pub mod gpu;
pub mod mem;
pub mod pipeline;
pub mod render;
