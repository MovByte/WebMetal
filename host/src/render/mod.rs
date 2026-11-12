//! The render path: render and parallel render command encoders replayed as WebGPU render passes (plan sections 6.4,
//! 6.5 and 5.9, package H5).
//!
//! - [`metal`]: the Metal values the path reads, with their WebGPU counterparts.
//! - [`depth_stencil`]: `MTLDepthStencilState` objects (the control request `CreateDepthStencilState`).
//! - [`pipeline`]: the render pipeline payload the pipeline service builds, and its dynamic-state variants.
//!
//! **Payloads.** A `DepthStencilState` handle carries a [`depth_stencil::DepthStencilState`]; a `RenderPipelineState`
//! handle a [`pipeline::RenderPipeline`] (created by the pipeline service).
//!
//! Every failure is a [`RenderError`] naming what failed; nothing here substitutes a result.

pub mod depth_stencil;
pub mod error;
pub mod metal;
pub mod pipeline;

pub use error::RenderError;
