//! [`Memory`]: the executor's memory services on its one WebGPU device.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::format::webgpu::{HostFeatures, WebGpuFormat};
use crate::gpu::idl::SupportedLimits;
use crate::gpu::{Api, Device as _, Queue as _};

use super::staging::StagingArena;

/// Memory services of one WebGPU device: resource creation, coherence with guest memory, texture contents and the
/// IOSurface table. Objects it creates ([`super::buffer::Buffer`], [`super::texture::Texture`]) are owned by the
/// caller's object tables; their GPU operations go through this value.
pub struct Memory<A: Api> {
    pub(crate) device: A::Device,
    pub(crate) queue: A::Queue,
    pub(crate) features: HostFeatures,
    pub(crate) limits: SupportedLimits,
    pub(crate) staging: StagingArena<A>,
    /// The next queue position (see [`super::shadow`]).
    seq: Cell<u64>,
    /// Formats whose texel copies were tested on this device (see [`super::proof`]).
    proofs: RefCell<HashMap<WebGpuFormat, Result<(), String>>>,
}

impl<A: Api> Memory<A> {
    /// The services for `device`, created with `features` (the WebGPU features the device has, as the format module
    /// names them).
    pub fn new(device: A::Device, features: HostFeatures) -> Memory<A> {
        let queue = device.queue();
        let limits = device.limits();
        Memory {
            device,
            queue,
            features,
            limits,
            staging: StagingArena::default(),
            seq: Cell::new(1),
            proofs: RefCell::new(HashMap::new()),
        }
    }

    pub fn device(&self) -> &A::Device {
        &self.device
    }

    pub fn features(&self) -> HostFeatures {
        self.features
    }

    /// A new queue position: every queue operation that moves bytes of CPU-visible memory takes one, in the order
    /// the operations reach the queue.
    pub(crate) fn next_seq(&self) -> u64 {
        let s = self.seq.get();
        self.seq.set(s + 1);
        s
    }

    /// The outcome of the texel copy test of `format` on this device, if it ran.
    pub fn proof(&self, format: WebGpuFormat) -> Option<Result<(), String>> {
        self.proofs.borrow().get(&format).cloned()
    }

    /// Record the outcome of a texel copy test ([`super::proof::TexelCopyTest`]).
    pub fn record_proof(&self, format: WebGpuFormat, outcome: Result<(), String>) {
        self.proofs.borrow_mut().insert(format, outcome);
    }

    /// `queue.submit(command_buffers)` for work whose readbacks are in `readbacks`: the readbacks take their queue
    /// position here. Every command buffer that holds readback copies must be submitted through this method, and no
    /// other upload may be made between encoding the copies and this call.
    pub fn submit(&self, command_buffers: &[&A::CommandBuffer], readbacks: &mut super::buffer::Readbacks<A>) {
        self.queue.submit(command_buffers);
        let seq = self.next_seq();
        readbacks.submitted(seq);
    }
}
