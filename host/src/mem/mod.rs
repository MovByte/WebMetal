//! Memory and coherence: the storage behind Metal buffers, textures and heaps, and how guest memory, IOSurfaces and
//! WebGPU objects are kept equal.
//!
//! - [`options`]: `MTLResourceOptions` and the storage, CPU cache and hazard-tracking modes.
//! - [`purgeable`]: the states of `setPurgeableState:`.
//! - [`heap`]: heaps and which sub-allocations they can be shown to hold.
//! - [`shadow`]: the shadow/diff protocol for CPU-visible memory (plan section 6.2).
//! - [`memory`]: [`Memory`], the services of one WebGPU device, and [`staging`], its readback buffers.
//! - [`buffer`]: Metal buffers over WebGPU buffers and guest memory.
//!
//! Every failure is a [`MemError`] that names what failed; nothing here substitutes a result.

pub mod buffer;
pub mod error;
pub mod heap;
pub mod memory;
pub mod options;
pub mod purgeable;
pub mod shadow;
pub mod staging;

#[cfg(test)]
mod testutil;

pub use error::MemError;
pub use memory::Memory;
