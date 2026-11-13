//! The A10X's texture memory layout: where the A10X Metal driver (AGXMetalA10 of iPadOS 17.7.11, 21H461) puts each
//! texel of a texture whose memory the CPU sees, and the driver's CPU copies between that memory and linear bytes.
//!
//! A texture made over caller memory (`newTextureWithBytesNoCopy:length:descriptor:deallocator:`) and the memory a
//! texture layout object fills (`-[MTLTextureLayout copyFromLinearBytes:...]`, which QuartzCore uses to prepare it)
//! hold texels in this layout, not Metal's linear one. The executor reads and writes such memory with this module.
//!
//! - [`gen2`]: the texture object the driver builds from a descriptor (`AGX::TextureGen2`): kind (linear or
//!   twiddled), level offsets, slice sizes, twiddled block information and the layout's size and alignment.
//! - [`region`]: `writeRegion` and `readRegion`, with `validateRegion`'s refusals.
//! - [`formats`]: the fields of the driver's format table the layout reads (generated).

pub mod formats;
pub mod gen2;
pub mod region;
