//! The core executor: control requests, object tables, command queues, command buffers, events and fences.
//!
//! The embedder hands the executor each control request a guest task sends ([`Executor::control`]) and keeps it
//! running ([`Executor::pump`], [`Executor::idle`]); the executor answers with the reply record and reports command
//! buffer progress through [`Embedder::post`]. Everything runs on the thread that owns the WebGPU device. The embedder's
//! methods and the services are called from inside the executor and must not call back into it.
//!
//! # Requests
//!
//! Requests decode strictly with `webmetal-wire`. Before anything acts on one, every handle it carries is checked
//! against the task's object table by the rules the schema implies (`objects`, `schema`): the handle of the object a
//! request creates is fresh, every other handle is live and of a kind the field accepts. A malformed request, a handle
//! that breaks those rules or any other protocol violation is answered with status 4. Requests the core does not serve
//! go to the [`ControlService`] that claims their opcode; without one they are answered with status 2. `FetchEvents`
//! is answered by the embedder from the events posted to it ([`fetch_events_reply`]), not by the executor.
//!
//! # Command buffers
//!
//! `Commit` reads the stream from guest memory, decodes it and checks its structure (`stream`) before it replies; a
//! stream that breaks the protocol is refused with status 4 and nothing of it runs. An accepted command buffer joins
//! its queue. Each queue replays its command buffers in commit order, one at a time; queues interleave in commit order
//! of the command buffer each would replay next, which is one schedule Metal allows. Encoder segments go to the
//! [`EncoderService`] of their kind, which records them into the command buffer's WebGPU command encoder.
//!
//! Every command buffer produces exactly one CommandBufferScheduled event, once all its work is on the WebGPU queue,
//! and then one CommandBufferCompleted event, once the GPU has finished its work, its writes to CPU-visible memory are
//! in guest memory ([`Coherence`]) and the error scopes covering its WebGPU calls have reported nothing. Completed
//! events of one queue are posted in commit order. The framework's `_MTLCommandBuffer` runs the status machine and
//! the handlers from these events on the guest side (the schema's guest-local annotations).
//!
//! # Events
//!
//! A wait (`encodeWaitForEvent:value:`) holds its queue until the event's value is at least the value; the work before
//! it is submitted first. A signal splits the submission: an `MTLEvent`'s value (kept here, starting at 0, never
//! decreasing) is set as soon as the work before the signal is on the queue, since only GPU work submitted later can
//! observe it; an `MTLSharedEvent` (a kernel object reached through the embedder) is signalled once that work has
//! completed and been written back, because the CPU observes it. Each rule is measured on real Metal; the comments at
//! each site give the measurement.
//!
//! # Failures
//!
//! A record the host does not implement makes its command buffer complete with status 2 and the record's offset,
//! without executing. A service's failure stops the command buffer at the failing record. WebGPU errors the error
//! scopes capture, and device loss, map to Metal's results as `failure` describes; after the device is lost, every
//! command buffer in flight or committed later completes with the `device-removed` failure and every task receives a
//! DeviceLost event. Every failure with status 2 or 3 is also kept as a [`Report`] for the embedder.

pub mod failure;
pub mod objects;
mod runtime;
#[rustfmt::skip]
#[allow(clippy::all)]
pub mod schema;
pub mod service;
pub mod stream;

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

use webmetal_wire::{DecodeError, EncodeError, Event, Failure, FetchEventsReply, Handle, Reply};

use crate::embed::{Embedder, SharedEventId, TaskId};
use crate::gpu::scope::Policy;
use crate::gpu::{Api, GpuFuture};
use runtime::{Core, Pending};

pub use objects::{Entry, Owner, Table};
pub use schema::Kind;
pub use service::{
    Coherence, CommitInfo, ControlService, DebugGroups, EncodeCx, EncoderService, Gpu, RecordFailure, Served,
    ServiceCx, Services,
};
pub use stream::{EncoderKind, Segment};

/// How the executor runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// The error-scope plumbing: per-call attribution for tests, batches otherwise; under `strict`, an error that
    /// escapes every scope panics.
    pub scopes: Policy,
}

/// A failure with status 2 or 3 the executor answered with: something the host does not implement or cannot honor.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub task: TaskId,
    /// The command buffer that completed with the failure.
    pub command_buffer: Option<Handle>,
    /// The control request answered with the failure.
    pub opcode: Option<u16>,
    /// The stream offset of the record that failed.
    pub offset: Option<u64>,
    pub failure: Failure,
}

/// The executor of one WebGPU device. Cloning refers to the same executor.
pub struct Executor<A: Api, E: Embedder> {
    core: Rc<RefCell<Core<A, E>>>,
}

impl<A: Api, E: Embedder> Clone for Executor<A, E> {
    fn clone(&self) -> Executor<A, E> {
        Executor { core: self.core.clone() }
    }
}

impl<A: Api, E: Embedder + 'static> Executor<A, E> {
    /// An executor on `device`, which no one else uses. Installs the error-scope plumbing and watches for device loss.
    ///
    /// Panics if two services claim one opcode, or a service claims an opcode the core serves or that is not a control
    /// request: that is a wiring defect of the embedder, not a run-time condition.
    pub fn new(device: A::Device, embedder: Rc<E>, services: Services<A, E>, config: Config) -> Executor<A, E> {
        Executor { core: Rc::new(RefCell::new(Core::new(device, embedder, services, config))) }
    }

    /// Run one control request from `task` and produce its reply record. A request whose opcode cannot be read as a
    /// control request has no reply record; the decode error is returned for the transport to refuse the call.
    pub fn control(&self, task: TaskId, request: &[u8]) -> GpuFuture<Result<Vec<u8>, DecodeError>> {
        let first = self.core.borrow_mut().control(task, request);
        let exec = self.clone();
        Box::pin(async move {
            let reply = match first? {
                Pending::Reply(r) => r,
                Pending::Service(mut p) => {
                    let captured = match p.settle.take() {
                        Some(f) => exec.pumped(f).await,
                        None => Vec::new(),
                    };
                    let reply = match p.served.take().expect("served") {
                        Served::Ready(r) => r,
                        Served::Later(f) => exec.pumped(f).await,
                    };
                    exec.core.borrow_mut().finish_service(p, captured, reply)
                }
            };
            let mut core = exec.core.borrow_mut();
            core.report_reply(task, &reply);
            Ok(core.encode_reply(&reply))
        })
    }

    /// A CPU-side signal changed a shared event's value (the kernel's `setSignaledValue:`); waits on it are checked
    /// again.
    pub fn shared_event_signaled(&self, event: SharedEventId) {
        let _ = event;
        self.core.borrow_mut().settle_state();
    }

    /// The task exited. Its objects are gone, its command buffers that have not completed are dropped without events,
    /// and the services are told.
    pub fn task_exited(&self, task: TaskId) {
        self.core.borrow_mut().task_exited(task);
    }

    /// Make all the progress that is possible without waiting: collect finished GPU work, signal events, post
    /// completions and replay unblocked command buffers. Returns whether GPU work or write-backs are still
    /// outstanding (command buffers waiting on an event do not count: only a signal moves them).
    pub fn pump(&self, cx: &mut Context<'_>) -> bool {
        loop {
            let mut changed = false;
            let watch = self.core.borrow_mut().lost_watch.take();
            if let Some(mut w) = watch {
                match w.as_mut().poll(cx) {
                    Poll::Ready(info) => {
                        self.core.borrow_mut().on_lost(info);
                        changed = true;
                    }
                    Poll::Pending => self.core.borrow_mut().lost_watch = Some(w),
                }
            }
            let jobs = self.core.borrow_mut().take_jobs();
            let mut pending = Vec::new();
            let mut ready = Vec::new();
            for mut job in jobs {
                match job.fut.as_mut().poll(cx) {
                    Poll::Ready(out) => ready.push((job.owner, job.seq(), out)),
                    Poll::Pending => pending.push(job),
                }
            }
            let mut core = self.core.borrow_mut();
            core.return_jobs(pending);
            if !ready.is_empty() {
                changed = true;
                for (owner, seq, out) in ready {
                    core.job_done(owner, seq, out);
                }
            }
            changed |= core.settle_state();
            if !changed {
                return core.has_jobs();
            }
        }
    }

    /// Resolves once nothing is outstanding but command buffers waiting on events.
    pub fn idle(&self) -> GpuFuture<()> {
        let exec = self.clone();
        Box::pin(std::future::poll_fn(move |cx| if exec.pump(cx) { Poll::Pending } else { Poll::Ready(()) }))
    }

    /// Resolves once every command buffer committed so far (by `task`, or by any task) has completed: the point after
    /// which a frame that shows their results may be presented.
    pub fn committed_work_done(&self, task: Option<TaskId>) -> GpuFuture<()> {
        let mark = self.core.borrow().next_run();
        let exec = self.clone();
        Box::pin(std::future::poll_fn(move |cx| {
            exec.pump(cx);
            if exec.core.borrow().runs_before(mark, task) { Poll::Pending } else { Poll::Ready(()) }
        }))
    }

    /// The failures with status 2 or 3 since the last call.
    pub fn take_reports(&self) -> Vec<Report> {
        self.core.borrow_mut().take_reports()
    }

    /// A future that pumps the executor while it waits for `f`.
    fn pumped<T: 'static>(&self, f: GpuFuture<T>) -> impl Future<Output = T> + use<T, A, E> {
        Pumped { exec: self.clone(), f }
    }
}

struct Pumped<T, A: Api, E: Embedder> {
    exec: Executor<A, E>,
    f: GpuFuture<T>,
}

impl<T, A: Api, E: Embedder + 'static> Future for Pumped<T, A, E> {
    type Output = T;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        if let Poll::Ready(v) = self.f.as_mut().poll(cx) {
            return Poll::Ready(v);
        }
        self.exec.pump(cx);
        Poll::Pending
    }
}

/// The FetchEvents reply record an embedder sends for the events it queued for one command queue, in the order they
/// were posted.
pub fn fetch_events_reply(events: &[Event]) -> Result<Vec<u8>, EncodeError> {
    let mut records = Vec::new();
    for e in events {
        e.encode(&mut records)?;
    }
    let mut out = Vec::new();
    Reply::FetchEvents(Ok(FetchEventsReply { events: records })).encode(&mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::schema;

    #[test]
    fn schema_rules_are_generated_from_the_current_schema() {
        assert_eq!(schema::SCHEMA_HASH, webmetal_wire::SCHEMA_HASH, "run host/src/exec/gen/execgen.py");
    }

    #[test]
    fn generated_rules_are_current() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let status = std::process::Command::new("python3")
            .args(["-B", "-I"])
            .arg(root.join("src/exec/gen/execgen.py"))
            .arg("--check")
            .status()
            .expect("python3 is required to check the generated rules");
        assert!(status.success(), "host/src/exec/schema.rs is stale: run host/src/exec/gen/execgen.py");
    }
}
