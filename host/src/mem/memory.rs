//! [`Memory`]: the executor's memory services on its one WebGPU device.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::rc::Rc;
use std::task::{Poll, Waker};

use crate::blit::kernels::Kernels;
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
    /// The readbacks encoded and not yet merged, in queue order.
    readbacks: Rc<ReadbackOrder>,
    /// Formats whose texel copies were tested on this device (see [`super::proof`]).
    proofs: RefCell<HashMap<WebGpuFormat, Result<(), String>>>,
    /// The masked-write pipeline and its bind group layout, made on first use.
    pub(crate) masked: RefCell<Option<(A::ComputePipeline, A::BindGroupLayout)>>,
    /// Words written under a mask so far (see [`super::shadow`]).
    pub(crate) masked_words: Cell<u64>,
    /// The IOSurface planes textures are made over, shared by every task.
    pub(crate) surfaces: RefCell<HashMap<(SurfaceId, u32), Rc<SurfacePlane>>>,
    /// The pipelines of the passes that move bytes WebGPU's copies cannot (`crate::blit::kernels`), made on first use.
    pub(crate) kernels: Kernels<A>,
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
            readbacks: Rc::default(),
            proofs: RefCell::new(HashMap::new()),
            masked: RefCell::new(None),
            masked_words: Cell::new(0),
            surfaces: RefCell::new(HashMap::new()),
            kernels: Kernels::default(),
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

    /// The queue position of a readback and its [`Turn`]: a new position, taken as the readback's copies are encoded.
    pub(crate) fn readback_turn(&self) -> Turn {
        let seq = self.next_seq();
        self.readbacks.open.borrow_mut().insert(seq);
        Turn { order: self.readbacks.clone(), seq }
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

/// The readbacks whose copies are encoded and whose bytes have not been merged into guest memory.
///
/// A readback's bytes are merged against the shadows as the last merged readback left them ([`super::shadow`]), so
/// readbacks merge in queue order. Their bytes do not arrive in that order: each is mapped with `mapAsync`, and WebGPU
/// orders the settlement of `mapAsync` promises of different buffers in no way (WebGPU section 3.2.2, "Promise
/// Ordering"; under load, Dawn and wgpu-native both settle a later readback's maps before an earlier one's). Merged in
/// arrival order, the bytes an earlier readback copied before a later command wrote the buffer would replace what that
/// command wrote.
#[derive(Default)]
struct ReadbackOrder {
    open: RefCell<BTreeSet<u64>>,
    /// The readbacks whose bytes arrived while an earlier one was open.
    waiting: RefCell<Vec<Waker>>,
}

/// A readback's place in queue order, from the encoding of its copies until it is dropped: once its bytes are merged,
/// or when the readback is abandoned (a failed map, a dropped command buffer). Later readbacks merge only after it.
pub struct Turn {
    order: Rc<ReadbackOrder>,
    seq: u64,
}

impl Turn {
    /// The readback's queue position.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Resolves once every readback encoded before this one has been dropped.
    pub(crate) fn ready(&self) -> impl Future<Output = ()> + '_ {
        std::future::poll_fn(move |cx| {
            if self.order.open.borrow().first() == Some(&self.seq) {
                Poll::Ready(())
            } else {
                self.order.waiting.borrow_mut().push(cx.waker().clone());
                Poll::Pending
            }
        })
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        self.order.open.borrow_mut().remove(&self.seq);
        let waiting = std::mem::take(&mut *self.order.waiting.borrow_mut());
        for w in waiting {
            w.wake();
        }
    }
}
