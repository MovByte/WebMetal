//! Holding an encoder until the earlier work it depends on has settled.
//!
//! **Why.** Metal leaves the encoders at and after a command buffer's first fault Faulted ("caused an error"),
//! Affected ("affected by an error ... and failed to execute in full") or Pending ("never started execution")
//! (`MTLCommandEncoderErrorState`, MTLCommandBuffer.h), so the executor keeps none of their writes: a write-back of that
//! work is dropped ([`crate::exec::Verdict::Drop`]) and what it could have written is made again from guest memory, or
//! poisoned where nothing CPU-visible backs it. But WebGPU cannot take back work already submitted: the queue timeline
//! runs submitted work in submission order (WebGPU section 3.4.1, "Timelines"), so a later command buffer's work runs
//! on the GPU copies the faulting work left. Whether that work faulted is known only once the promises that report it
//! have settled (`popErrorScope`, `onSubmittedWorkDone`, `mapAsync`), whose order WebGPU leaves undefined (section
//! 3.2.2, "Promise Ordering"), and by then a later command buffer may already be on the queue. Had it copied a buffer
//! the dropped work filled into another buffer, the dropped bytes would reach that output, where no write-back of the
//! dropped work can find them.
//!
//! **What waits.** Before an encoder is recorded, the coherence service admits it ([`crate::exec::Coherence::admit`]).
//! Every write-back records what the work it reads back could have written ([`Holds::record`]): the buffers it reads
//! back or would poison, by the rule that picks its readbacks ([`super::Memory::encode_readbacks`]), the textures it
//! reads back or would poison, and what the work declared it writes through argument buffers. The record is unsettled
//! until its future has finished: merged, or dropped with every resource it names made stale or poisoned. An encoder
//! is held while a record of another command buffer is unsettled, its verdict is not
//! [`crate::exec::Verdict::Merge`], and it names a resource the encoder uses: a resource a record of the encoder names,
//! a view's root texture, a texture over a buffer's buffer and texture, and every resource allocated from a heap the
//! encoder makes resident, as the heap holds them then. The encoder's command buffer and the rest of its queue wait
//! with it, so queue order is kept, and nothing else does: an encoder that uses nothing such a record names runs at
//! once, and work of the same command buffer never waits on its own write-backs, since work after its fault is dropped
//! with them.
//!
//! **What runs.** Once nothing holds it, the encoder runs on what Metal would have left: a buffer whose bytes dropped
//! work left stale is uploaded from guest memory, a texture over guest memory or an IOSurface plane whose write-back was
//! dropped is written whole from it, and a poisoned resource fails the command buffer at the encoder, before any of its
//! work reaches the queue. A merged write-back needs none of this: the work it read back is the work Metal ran.

use std::cell::Cell;
use std::rc::{Rc, Weak};

use webmetal_wire::Handle;

use super::buffer::Buffer;
use super::texture::Texture;
use crate::embed::TaskId;
use crate::exec::{Gate, Verdict};
use crate::gpu::Api;

/// The resources a piece of work names: what an encoder uses, or what a write-back's work could have written.
pub(crate) struct Named<A: Api> {
    pub(crate) buffers: Vec<Rc<Buffer<A>>>,
    pub(crate) textures: Vec<Rc<Texture<A>>>,
}

impl<A: Api> Default for Named<A> {
    fn default() -> Named<A> {
        Named { buffers: Vec::new(), textures: Vec::new() }
    }
}

impl<A: Api> Named<A> {
    pub(crate) fn buffer(&mut self, b: Rc<Buffer<A>>) {
        if !self.buffers.iter().any(|u| Rc::ptr_eq(u, &b)) {
            self.buffers.push(b);
        }
    }

    pub(crate) fn texture(&mut self, t: Rc<Texture<A>>) {
        if !self.textures.iter().any(|u| Rc::ptr_eq(u, &t)) {
            self.textures.push(t);
        }
    }
}

/// What one write-back's work could have written, until the write-back settles.
struct Unsettled<A: Api> {
    /// The command buffer (task and handle) the write-back belongs to.
    run: (TaskId, Handle),
    gate: Gate,
    buffers: Vec<Weak<Buffer<A>>>,
    textures: Vec<Weak<Texture<A>>>,
    settled: Cell<bool>,
}

impl<A: Api> Unsettled<A> {
    /// Whether the record may still make what it names stale or poisoned.
    fn open(&self) -> bool {
        !self.settled.get() && self.gate.decided() != Some(Verdict::Merge)
    }

    fn names(&self, n: &Named<A>) -> bool {
        self.buffers.iter().filter_map(Weak::upgrade).any(|b| n.buffers.iter().any(|u| Rc::ptr_eq(u, &b)))
            || self.textures.iter().filter_map(Weak::upgrade).any(|t| n.textures.iter().any(|u| Rc::ptr_eq(u, &t)))
    }
}

/// Settles its record when dropped: when the write-back's future finishes, or is dropped unfinished with its
/// command buffer (device loss, the task's exit).
pub(crate) struct Settles<A: Api>(Rc<Unsettled<A>>);

impl<A: Api> Drop for Settles<A> {
    fn drop(&mut self) {
        self.0.settled.set(true);
    }
}

/// The unsettled write-backs, in the order their readbacks were encoded (queue order).
pub(crate) struct Holds<A: Api> {
    unsettled: Vec<Rc<Unsettled<A>>>,
}

impl<A: Api> Default for Holds<A> {
    fn default() -> Holds<A> {
        Holds { unsettled: Vec::new() }
    }
}

impl<A: Api> Holds<A> {
    /// A write-back of command buffer `run`, decided through `gate`, whose work could have written `written`. The
    /// record settles when the returned guard is dropped.
    pub(crate) fn record(&mut self, run: (TaskId, Handle), gate: Gate, written: &Named<A>) -> Settles<A> {
        let u = Rc::new(Unsettled {
            run,
            gate,
            buffers: written.buffers.iter().map(Rc::downgrade).collect(),
            textures: written.textures.iter().map(Rc::downgrade).collect(),
            settled: Cell::new(false),
        });
        self.unsettled.push(u.clone());
        Settles(u)
    }

    /// Whether an encoder of command buffer `run` that uses `uses` must wait: an earlier write-back of another command
    /// buffer that could have written one of them is unsettled and not known to merge.
    pub(crate) fn holds(&mut self, run: (TaskId, Handle), uses: &Named<A>) -> bool {
        self.unsettled.retain(|u| !u.settled.get());
        self.unsettled.iter().any(|u| u.run != run && u.open() && u.names(uses))
    }

    /// The task exited: its write-backs are dropped with its command buffers.
    pub(crate) fn task_exited(&mut self, task: TaskId) {
        self.unsettled.retain(|u| u.run.0 != task && !u.settled.get());
    }
}
