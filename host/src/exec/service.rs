//! The seams the other executor modules plug into.
//!
//! The core owns the object tables, the command queues, command buffers, events and fences, and the order in which
//! work reaches WebGPU. Everything else is a service:
//!
//! - [`ControlService`] (memory `mem/`, pipelines `pipeline/`): control requests by opcode, such as resource, heap,
//!   sampler and pipeline creation, `replaceRegion:`, `getBytes:` and purgeable state.
//! - [`EncoderService`] (render `render/`, compute `compute/`, blit `blit/`): one encoder's records, recorded into the
//!   command buffer's WebGPU command encoder.
//! - [`Coherence`] (memory `mem/`): guest memory and the GPU copies of CPU-visible resources: uploads at commit,
//!   write-back before a shared event is signalled and before completion.
//! - [`super::Executor::committed_work_done`] (presentation `present/`): waiting for every command buffer committed
//!   before a frame.
//!
//! Services reach other services' objects through the object table: the service that creates an object attaches a
//! payload ([`ServiceCx::attach`]) whose type it documents, and any service reads it through [`Table::get`].
//!
//! Every WebGPU call that can raise a device-timeline error must go through [`Gpu::call`], from the synchronous part of
//! a service method: the core settles the error scopes when the method returns and attributes what they captured to the
//! request or command buffer the method served. A future a service returns may await WebGPU promises (`mapAsync`,
//! `onSubmittedWorkDone`) and touch guest memory, but makes no such calls.

use std::any::Any;
use std::cell::Cell;
use std::rc::Rc;

use webmetal_wire::{Command, Failure, Handle, Reply, Request};

use super::objects::{Entry, Table};
use super::stream::Segment;
use crate::embed::{Embedder, TaskId};
use crate::gpu::idl::{CommandBufferDescriptor, CommandEncoderDescriptor};
use crate::gpu::scope::ErrorScopes;
use crate::gpu::{Api, CommandEncoder, DebugCommands, Device, GpuFuture, Queue};

/// The executor's WebGPU device, for services.
pub struct Gpu<'a, A: Api> {
    pub(crate) device: &'a A::Device,
    pub(crate) queue: &'a A::Queue,
    pub(crate) scopes: &'a ErrorScopes<A>,
    pub(crate) calls: &'a Cell<u64>,
}

impl<A: Api> Gpu<'_, A> {
    /// Make a call that may raise a device-timeline error. `op` names the IDL operation (`"GPUDevice.createBuffer"`),
    /// `label` the object it makes or acts on.
    pub fn call<T>(&self, op: &'static str, label: &str, f: impl FnOnce(&A::Device) -> T) -> T {
        self.calls.set(self.calls.get() + 1);
        self.scopes.call(op, label, f)
    }

    /// The device, for reads that raise no device-timeline error (features, limits, labels).
    pub fn device(&self) -> &A::Device {
        self.device
    }

    /// The device's queue. Its operations go through [`Gpu::call`] like any other.
    pub fn queue(&self) -> &A::Queue {
        self.queue
    }
}

/// What a control service or the coherence service works with.
pub struct ServiceCx<'a, A: Api, E: Embedder> {
    pub(crate) task: TaskId,
    pub(crate) gpu: Gpu<'a, A>,
    pub(crate) embedder: &'a E,
    pub(crate) objects: &'a Table,
    pub(crate) attached: Option<Rc<dyn Any>>,
}

impl<'a, A: Api, E: Embedder> ServiceCx<'a, A, E> {
    /// The task that sent the request.
    pub fn task(&self) -> TaskId {
        self.task
    }

    pub fn gpu(&self) -> &Gpu<'a, A> {
        &self.gpu
    }

    pub fn embedder(&self) -> &'a E {
        self.embedder
    }

    /// The task's live objects.
    pub fn objects(&self) -> &'a Table {
        self.objects
    }

    /// Attach a payload to the object the request creates. It becomes the entry's payload when the request succeeds.
    pub fn attach(&mut self, payload: Rc<dyn Any>) {
        self.attached = Some(payload);
    }
}

/// The result of serving a control request.
pub enum Served {
    /// The reply.
    Ready(Reply),
    /// The reply once the future resolves (after a readback, for example).
    Later(GpuFuture<Reply>),
}

/// A service that serves a set of control requests.
pub trait ControlService<A: Api, E: Embedder> {
    /// The control opcodes this service serves. No two services, and not the core, may serve the same opcode.
    fn opcodes(&self) -> Vec<u16>;

    /// Serve one request. Before the call the core has checked every handle the request carries: the object it creates
    /// has a fresh handle, every other handle is live in the task and of a kind the field accepts. When the reply is a
    /// success and the error scopes captured nothing, the core binds the created handle to this service with the
    /// payload attached through `cx`; otherwise it replies with the failure and calls [`ControlService::release`] for
    /// the handle so the service drops what it made.
    fn serve(&mut self, cx: &mut ServiceCx<'_, A, E>, request: &Request) -> Served;

    /// SetLabel changed the label of an object this service created; `entry.label` is the new label.
    fn set_label(&mut self, cx: &mut ServiceCx<'_, A, E>, handle: Handle, entry: &Entry) {
        let _ = (cx, handle, entry);
    }

    /// The guest released an object this service created (or a request that would have created it failed).
    fn release(&mut self, task: TaskId, handle: Handle, entry: &Entry);

    /// The task exited; every object of the task is gone.
    fn task_exited(&mut self, task: TaskId);
}

/// One committed command buffer, as services see it.
#[derive(Clone, Debug)]
pub struct CommitInfo {
    pub task: TaskId,
    pub command_buffer: Handle,
    pub queue: Handle,
    pub label: Option<String>,
    pub retained_references: bool,
    /// Every existing object the stream names (the Commit's `resources`).
    pub resources: Vec<Handle>,
    /// The decoded stream: each record with its byte offset.
    pub records: Rc<[(usize, Command)]>,
}

/// A record that failed: its stream offset and the failure the command buffer completes with.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordFailure {
    pub offset: u64,
    pub failure: Failure,
}

/// The WebGPU command encoder a command buffer is being recorded into, with the command buffer's open debug groups.
pub(crate) struct Recording<A: Api> {
    pub(crate) label: String,
    pub(crate) encoder: Option<A::CommandEncoder>,
    /// `-[MTLCommandBuffer pushDebugGroup:]` groups open at this point of the stream.
    pub(crate) debug: Vec<String>,
    /// Submissions made so far.
    pub(crate) submits: u64,
}

impl<A: Api> Recording<A> {
    pub(crate) fn new(label: String) -> Recording<A> {
        Recording { label, encoder: None, debug: Vec::new(), submits: 0 }
    }

    /// The current command encoder, created with the command buffer's open debug groups pushed on it.
    pub(crate) fn encoder(&mut self, gpu: &Gpu<'_, A>) -> &A::CommandEncoder {
        if self.encoder.is_none() {
            let desc = CommandEncoderDescriptor { label: self.label.clone() };
            let e = gpu.call("GPUDevice.createCommandEncoder", &self.label, |d| d.create_command_encoder(&desc));
            for g in &self.debug {
                gpu.call("GPUCommandEncoder.pushDebugGroup", &self.label, |_| e.push_debug_group(g));
            }
            self.encoder = Some(e);
        }
        self.encoder.as_ref().expect("encoder")
    }

    /// Finish the current command encoder, closing the debug groups it holds, and submit it. Returns whether anything
    /// was submitted.
    pub(crate) fn submit(&mut self, gpu: &Gpu<'_, A>) -> bool {
        let Some(e) = self.encoder.take() else { return false };
        for _ in &self.debug {
            gpu.call("GPUCommandEncoder.popDebugGroup", &self.label, |_| e.pop_debug_group());
        }
        let desc = CommandBufferDescriptor { label: self.label.clone() };
        let cb = gpu.call("GPUCommandEncoder.finish", &self.label, |_| e.finish(&desc));
        gpu.call("GPUQueue.submit", &self.label, |_| gpu.queue.submit(&[&cb]));
        self.submits += 1;
        true
    }

    /// Drop the current command encoder without submitting it (its recording failed).
    pub(crate) fn discard(&mut self) {
        self.encoder = None;
    }

    pub(crate) fn push_debug_group(&mut self, gpu: &Gpu<'_, A>, label: &str) {
        if let Some(e) = &self.encoder {
            gpu.call("GPUCommandEncoder.pushDebugGroup", &self.label, |_| e.push_debug_group(label));
        }
        self.debug.push(label.to_string());
    }

    /// Pop the innermost group. Metal accepts a pop with no group open and the command buffer completes normally
    /// (measured on Apple M4, macOS 27.0.1); there is nothing to pop then.
    pub(crate) fn pop_debug_group(&mut self, gpu: &Gpu<'_, A>) {
        if self.debug.pop().is_some()
            && let Some(e) = &self.encoder
        {
            gpu.call("GPUCommandEncoder.popDebugGroup", &self.label, |_| e.pop_debug_group());
        }
    }
}

/// What an encoder service works with while recording one segment.
pub struct EncodeCx<'a, A: Api, E: Embedder> {
    pub(crate) task: TaskId,
    pub(crate) info: &'a CommitInfo,
    pub(crate) gpu: Gpu<'a, A>,
    pub(crate) embedder: &'a E,
    pub(crate) objects: &'a Table,
    pub(crate) recording: &'a mut Recording<A>,
}

impl<'a, A: Api, E: Embedder> EncodeCx<'a, A, E> {
    pub fn task(&self) -> TaskId {
        self.task
    }

    /// The command buffer being recorded.
    pub fn info(&self) -> &'a CommitInfo {
        self.info
    }

    /// The command buffer's stream: every record with its offset.
    pub fn records(&self) -> &'a [(usize, Command)] {
        &self.info.records
    }

    pub fn gpu(&self) -> &Gpu<'a, A> {
        &self.gpu
    }

    pub fn embedder(&self) -> &'a E {
        self.embedder
    }

    pub fn objects(&self) -> &'a Table {
        self.objects
    }

    /// The WebGPU command encoder to record into. Passes begun on it must end before the service returns.
    pub fn encoder(&mut self) -> &A::CommandEncoder {
        self.recording.encoder(&self.gpu)
    }

    /// Submit what has been recorded so far, so that a queue operation made next (`writeBuffer`, `writeTexture`)
    /// lands after it on the queue timeline. Later recording goes into a new command encoder.
    pub fn submit(&mut self) {
        self.recording.submit(&self.gpu);
    }
}

/// A service that records one kind of encoder.
pub trait EncoderService<A: Api, E: Embedder> {
    /// Record `segment` (its records are `cx.records()[i]` for each `i` in `segment.records`, from the `Begin*` record
    /// to the closing `EncoderEndEncoding`; fence records are the core's and are left out). A failure stops the
    /// command buffer: it completes with the failure, and nothing recorded since the last submission is submitted.
    fn encode(&mut self, cx: &mut EncodeCx<'_, A, E>, segment: &Segment) -> Result<(), RecordFailure>;

    /// The task exited.
    fn task_exited(&mut self, task: TaskId) {
        let _ = task;
    }
}

/// The service that keeps guest memory and the GPU copies of CPU-visible resources coherent: guest memory is what the
/// CPU sees, the GPU copy what the GPU sees, and each is brought up to date from the other at the points below.
pub trait Coherence<A: Api, E: Embedder> {
    /// A command buffer was accepted; called inside its Commit, before the reply, and before any of its work. Brings
    /// the CPU's writes to the resources it uses to their GPU copies. A failure makes the command buffer complete with
    /// it, without executing.
    fn commit(&mut self, cx: &mut ServiceCx<'_, A, E>, info: &CommitInfo) -> Result<(), Failure>;

    /// All of the command buffer's work up to this point has been recorded: record (on `cx`'s encoder) what brings the
    /// GPU's writes to CPU-visible resources back, and return the future that writes them to guest memory once the
    /// GPU has finished. Called before a shared event the command buffer signals becomes visible to the CPU, and
    /// before the command buffer completes.
    fn write_back(
        &mut self,
        cx: &mut EncodeCx<'_, A, E>,
        info: &CommitInfo,
    ) -> Result<Option<GpuFuture<Result<(), Failure>>>, Failure>;

    /// The task exited.
    fn task_exited(&mut self, task: TaskId);
}

/// The services an executor runs with. A missing service makes what it serves unimplemented (status 2).
pub struct Services<A: Api, E: Embedder> {
    pub control: Vec<Box<dyn ControlService<A, E>>>,
    /// Render encoders and parallel render encoders.
    pub render: Option<Box<dyn EncoderService<A, E>>>,
    pub compute: Option<Box<dyn EncoderService<A, E>>>,
    pub blit: Option<Box<dyn EncoderService<A, E>>>,
    pub coherence: Option<Box<dyn Coherence<A, E>>>,
}

impl<A: Api, E: Embedder> Default for Services<A, E> {
    fn default() -> Services<A, E> {
        Services { control: Vec::new(), render: None, compute: None, blit: None, coherence: None }
    }
}

/// An encoder's debug groups (`pushDebugGroup:`, `popDebugGroup`), with Metal's rules on WebGPU's: Metal accepts a pop
/// with no group open and an encoder that ends with groups open, and the command buffer completes normally (measured on
/// Apple M4, macOS 27.0.1); WebGPU rejects both. Encoder services keep one per pass and call [`DebugGroups::pop`] and
/// [`DebugGroups::close`] so the WebGPU calls stay balanced.
#[derive(Clone, Debug, Default)]
pub struct DebugGroups {
    open: usize,
}

impl DebugGroups {
    /// A group was pushed on the pass.
    pub fn push(&mut self) {
        self.open += 1;
    }

    /// Whether `popDebugGroup` should be forwarded to the pass: only when a group is open.
    pub fn pop(&mut self) -> bool {
        if self.open == 0 {
            return false;
        }
        self.open -= 1;
        true
    }

    /// The groups still open when the encoder ends, to pop before ending the pass.
    pub fn close(&mut self) -> usize {
        std::mem::take(&mut self.open)
    }
}
