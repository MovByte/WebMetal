//! [`Memory`]: the executor's memory services on its one WebGPU device.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::format::webgpu::{HostFeatures, WebGpuFormat};
use crate::gpu::idl::SupportedLimits;
use crate::gpu::{Api, Device as _};
use webmetal_wire::SurfaceId;

use super::staging::StagingArena;
use super::surface::SurfacePlane;

/// Memory services of one WebGPU device: resource creation, coherence with guest memory, texture contents and the
/// IOSurface table. Objects it creates ([`super::buffer::Buffer`], [`super::texture::Texture`]) live in the
/// executor's object tables as payloads; their GPU operations go through this value with the caller's
/// [`super::calls::Calls`].
pub struct Memory<A: Api> {
    pub(crate) features: HostFeatures,
    pub(crate) limits: SupportedLimits,
    pub(crate) staging: StagingArena<A>,
    /// The next queue position (see [`super::shadow`]).
    seq: Cell<u64>,
    /// Formats whose texel copies were tested on this device (see [`super::proof`]).
    proofs: RefCell<HashMap<WebGpuFormat, Result<(), String>>>,
    /// The masked-write pipeline and its bind group layout, made on first use.
    pub(crate) masked: RefCell<Option<(A::ComputePipeline, A::BindGroupLayout)>>,
    /// Words written under a mask so far (see [`super::shadow`]).
    pub(crate) masked_words: Cell<u64>,
    /// The IOSurface planes textures are made over, shared by every task.
    pub(crate) surfaces: RefCell<HashMap<(SurfaceId, u32), Rc<SurfacePlane>>>,
}

impl<A: Api> Memory<A> {
    /// The services for `device`, created with `features` (the WebGPU features the device has, as the format module
    /// names them).
    pub fn new(device: &A::Device, features: HostFeatures) -> Memory<A> {
        Memory {
            features,
            limits: device.limits(),
            staging: StagingArena::default(),
            seq: Cell::new(1),
            proofs: RefCell::new(HashMap::new()),
            masked: RefCell::new(None),
            masked_words: Cell::new(0),
            surfaces: RefCell::new(HashMap::new()),
        }
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

    /// Words written in part, under a mask, since the services started.
    pub fn masked_words(&self) -> u64 {
        self.masked_words.get()
    }

    /// The outcome of the texel copy test of `format` on this device, if it ran.
    pub fn proof(&self, format: WebGpuFormat) -> Option<Result<(), String>> {
        self.proofs.borrow().get(&format).cloned()
    }

    /// Record the outcome of a texel copy test ([`Memory::prove_texel_copies`]).
    pub fn record_proof(&self, format: WebGpuFormat, outcome: Result<(), String>) {
        self.proofs.borrow_mut().insert(format, outcome);
    }
}
