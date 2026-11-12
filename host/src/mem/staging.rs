//! Staging arenas: the mappable buffers readbacks copy into.
//!
//! A readback copies GPU bytes into a `MAP_READ | COPY_DST` buffer, maps it once the work before the copy has run, and
//! reads it on the CPU. WebGPU forbids `MAP_READ` together with any usage other than `COPY_DST` (WebGPU section
//! 5.1.3), so readbacks always go through such a buffer. The arena keeps unmapped staging buffers for reuse, in
//! power-of-two size classes of at least [`MIN_CLASS`] bytes, and holds at most [`MAX_POOLED`] bytes of them.

use std::cell::RefCell;

use crate::gpu::idl::{BufferDescriptor, BufferUsage};
use crate::gpu::{Api, Buffer as _, Device as _, Object as _, Size64};

use super::error::MemError;

/// The smallest staging buffer.
pub const MIN_CLASS: u64 = 4096;
/// The most bytes of idle staging buffers the arena keeps.
pub const MAX_POOLED: u64 = 64 << 20;

/// A pool of `MAP_READ | COPY_DST` buffers.
pub struct StagingArena<A: Api> {
    free: RefCell<Vec<A::Buffer>>,
}

impl<A: Api> Default for StagingArena<A> {
    fn default() -> Self {
        StagingArena { free: RefCell::new(Vec::new()) }
    }
}

impl<A: Api> StagingArena<A> {
    /// A staging buffer of at least `size` bytes.
    pub fn take(&self, device: &A::Device, size: u64) -> Result<A::Buffer, MemError> {
        let class = size.max(MIN_CLASS).next_power_of_two();
        let mut free = self.free.borrow_mut();
        if let Some(i) = free.iter().position(|b| b.size() == class) {
            return Ok(free.swap_remove(i));
        }
        drop(free);
        let size = Size64::new(class).map_err(|e| MemError::gpu("GPUDevice.createBuffer", e))?;
        let b = device
            .create_buffer(&BufferDescriptor::new(size, BufferUsage::MAP_READ | BufferUsage::COPY_DST))
            .map_err(|e| MemError::gpu("GPUDevice.createBuffer", e))?;
        b.set_label("webmetal staging");
        Ok(b)
    }

    /// Return an unmapped staging buffer.
    pub fn give_back(&self, buffer: A::Buffer) {
        let mut free = self.free.borrow_mut();
        let pooled: u64 = free.iter().map(|b| b.size()).sum();
        if pooled + buffer.size() <= MAX_POOLED {
            free.push(buffer);
        } else {
            buffer.destroy();
        }
    }
}
