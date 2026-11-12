//! The render path: render and parallel render command encoders replayed as WebGPU render passes (plan sections 6.4,
//! 6.5 and 5.9, package H5).
//!
//! - [`metal`]: the Metal values the path reads, with their WebGPU counterparts.
//! - [`depth_stencil`]: `MTLDepthStencilState` objects (the control request `CreateDepthStencilState`).
//! - [`pipeline`]: the render pipeline payload the pipeline service builds, and its dynamic-state variants.
//! - [`attachments`]: the pass descriptor, load, store and resolve actions.
//! - [`encoder`]: the encoder service: state, draws, splits, separate stencil faces, parallel encoders.
//! - [`index`]: index staging for Metal's primitive restart in list topologies.
//! - [`visibility`]: visibility results over occlusion queries.
//! - [`resolve`]: depth and stencil multisample resolve passes.
//! - [`helpers`]: the executor's own GPU programs those use.
//!
//! **Payloads.** A `DepthStencilState` handle carries a [`depth_stencil::DepthStencilState`]; a `RenderPipelineState`
//! handle a [`pipeline::RenderPipeline`] (created by the pipeline service); a `SamplerState` handle the render path
//! binds carries an [`encoder::SamplerObject`].
//!
//! Every failure is a [`RenderError`] naming what failed; nothing here substitutes a result.

pub mod attachments;
pub mod depth_stencil;
pub mod encoder;
pub mod error;
pub mod helpers;
pub mod index;
pub mod metal;
pub mod pipeline;
pub mod resolve;
pub mod visibility;

use std::rc::Rc;

use crate::embed::Embedder;
use crate::exec::{ControlService, EncoderService};
use crate::gpu::Api;
use crate::mem::Memory;

pub use error::RenderError;

/// The render path's services.
pub struct RenderServices<A: Api> {
    memory: Rc<Memory<A>>,
}

impl<A: Api> RenderServices<A> {
    /// The services over the memory module that owns the buffers and textures passes use.
    pub fn new(memory: Rc<Memory<A>>) -> RenderServices<A> {
        RenderServices { memory }
    }

    /// The encoder service for render and parallel render encoders.
    pub fn encoder<E: Embedder>(&self) -> Box<dyn EncoderService<A, E>> {
        Box::new(encoder::RenderEncoder::new(self.memory.clone()))
    }

    /// The control service that creates depth-stencil state objects.
    pub fn control<E: Embedder>(&self) -> Box<dyn ControlService<A, E>> {
        Box::new(depth_stencil::DepthStencilService)
    }
}
