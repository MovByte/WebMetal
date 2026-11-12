//! Memory and coherence: the storage behind Metal buffers, textures and heaps, and how guest memory, IOSurfaces and
//! WebGPU objects are kept equal.
//!
//! - [`options`]: `MTLResourceOptions` and the storage, CPU cache and hazard-tracking modes.
//! - [`purgeable`]: the states of `setPurgeableState:`.
//! - [`heap`]: heaps and which sub-allocations they can be shown to hold.
//! - [`shadow`]: the shadow/diff protocol for CPU-visible memory (plan section 6.2).
//! - [`memory`]: [`Memory`], the services of one WebGPU device, and [`staging`], its readback buffers.
//! - [`buffer`]: Metal buffers over WebGPU buffers and guest memory.
//! - [`layout`]: texture geometry and the byte layouts of CPU and blit transfers.
//! - [`texture`]: Metal textures, their exact byte copies, `replaceRegion`, `getBytes` and the blit transfers.
//! - [`proof`]: the per-host test of texel copies of formats whose bits WebGPU does not promise to keep.
//! - [`linear`]: textures over a buffer's bytes; [`surface`]: textures over IOSurface planes and the plane table.
//! - [`calls`]: how the module reaches WebGPU (through the executor's error scopes, or directly).
//! - [`service`]: the module as the executor's control and coherence services, and the payloads it binds to handles.
//!
//! Every failure is a [`MemError`] that names what failed; nothing here substitutes a result.

pub mod buffer;
pub mod calls;
pub mod error;
pub mod heap;
pub mod layout;
pub mod linear;
pub mod memory;
pub mod options;
pub mod proof;
pub mod purgeable;
pub mod service;
pub mod shadow;
pub mod staging;
pub mod surface;
pub mod texture;

#[cfg(test)]
mod testutil;

pub use error::MemError;
pub use memory::Memory;
