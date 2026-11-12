//! The executor's state and its replay of command queues.

use std::any::Any;
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::rc::{Rc, Weak};

use webmetal_wire::{
    Command, CommandBufferCompleted, CommandBufferScheduled, Commit, CommitReply, CreateCommandQueueReply,
    CreateCommandQueueWithDescriptorReply, CreateEventReply, CreateFenceReply, DeviceLost, Event, Failure, Handle,
    OpenReply, RegisterSharedEventReply, ReleaseReply, Reply, Request, SetLabelReply, opcode, status,
};

use super::failure::{self, GpuCause, NO_RECORD, command_buffer_status};
use super::objects::{Entry, Owner, Table};
use super::schema::{self, HandleUse, Kind, Role};
use super::service::{CommitInfo, EncodeCx, Gpu, Recording, Served, ServiceCx, Services};
use super::stream::{self, EncoderKind, Step};
use super::{Config, Report};
use crate::embed::{Embedder, SharedEventId, TaskId};
use crate::gpu::scope::{Captured, ErrorScopes};
use crate::gpu::{Api, Device, DeviceLostInfo, GpuFuture, Queue};

/// Commit order: runs are numbered in the order their Commit arrived.
pub(crate) type RunId = u64;

/// The control opcodes the core serves itself.
pub(crate) const CORE_OPCODES: [u16; 14] = [
    opcode::OPEN,
    opcode::FETCH_EVENTS,
    opcode::RELEASE,
    opcode::SET_LABEL,
    opcode::CREATE_COMMAND_QUEUE,
    opcode::CREATE_COMMAND_QUEUE_WITH_DESCRIPTOR,
    opcode::SET_GPU_PRIORITY,
    opcode::SET_GPU_PRIORITY_OFFSET,
    opcode::SET_BACKGROUND_GPU_PRIORITY,
    opcode::SET_BACKGROUND_GPU_PRIORITY_OFFSET,
    opcode::COMMIT,
    opcode::REGISTER_SHARED_EVENT,
    opcode::CREATE_FENCE,
    opcode::CREATE_EVENT,
];

/// A shared event kernel object the executor refers to; released through the embedder when the last reference goes.
pub(crate) struct SharedEvent<E: Embedder> {
    pub(crate) id: SharedEventId,
    embedder: Rc<E>,
}

impl<E: Embedder> Drop for SharedEvent<E> {
    fn drop(&mut self) {
        self.embedder.shared_event_release(self.id);
    }
}

/// An `MTLEvent` (its value lives in the executor) or an `MTLSharedEvent` (a kernel object).
pub(crate) enum EventRef<E: Embedder> {
    Local(Rc<Cell<u64>>),
    Shared(Rc<SharedEvent<E>>),
}

impl<E: Embedder> Clone for EventRef<E> {
    fn clone(&self) -> EventRef<E> {
        match self {
            EventRef::Local(v) => EventRef::Local(v.clone()),
            EventRef::Shared(s) => EventRef::Shared(s.clone()),
        }
    }
}

#[derive(Default)]
struct QueueState {
    /// The guest released the queue; its state goes once its last command buffer completes.
    released: bool,
    /// Committed command buffers that have not started.
    backlog: VecDeque<RunId>,
    /// The command buffer whose records are being replayed.
    active: Option<RunId>,
    /// Command buffers whose work is all submitted (CommandBufferScheduled posted), in commit order, until their
    /// CommandBufferCompleted is posted.
    scheduled: VecDeque<RunId>,
}

impl QueueState {
    fn idle(&self) -> bool {
        self.backlog.is_empty() && self.active.is_none() && self.scheduled.is_empty()
    }
}

struct TaskState<E: Embedder> {
    table: Table,
    queues: HashMap<Handle, QueueState>,
    events: HashMap<Handle, EventRef<E>>,
}

enum ActionKind<E: Embedder> {
    /// Set a shared event's value once the work before the signal has completed and been written back.
    SignalShared(Rc<SharedEvent<E>>, u64, u64),
    /// Post CommandBufferCompleted (in queue order).
    Complete,
}

/// Something a run does once every job it started before the action has finished.
struct Action<E: Embedder> {
    after: u64,
    kind: ActionKind<E>,
}

/// One committed command buffer.
struct Run<A: Api, E: Embedder> {
    task: TaskId,
    queue: Handle,
    info: Rc<CommitInfo>,
    steps: Vec<Step>,
    cursor: usize,
    events: HashMap<Handle, EventRef<E>>,
    /// The failure the command buffer completes with, and the stream offset of the record that failed.
    failure: Option<(Failure, u64)>,
    recording: Recording<A>,
    started: bool,
    kernel_start: u64,
    kernel_end: u64,
    gpu_start: Option<u64>,
    gpu_end: u64,
    scheduled: bool,
    next_job: u64,
    outstanding: BTreeSet<u64>,
    actions: VecDeque<Action<E>>,
    /// The completion action ran; CommandBufferCompleted is posted once every earlier command buffer of the queue has
    /// completed.
    complete: bool,
}

impl<A: Api, E: Embedder> Run<A, E> {
    fn fail(&mut self, failure: Failure, offset: u64) {
        if self.failure.is_none() {
            self.failure = Some((failure, offset));
        }
    }
}

pub(crate) enum JobOut {
    Done,
    Settled(Vec<Captured>),
    WrittenBack(Result<(), Failure>),
}

/// Whose WebGPU work a job waits for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JobOwner {
    Run(RunId),
    /// Calls a service made outside any request or command buffer (a label, a release, a task's exit): what their
    /// error scopes capture is reported for the task.
    Detached(TaskId),
}

pub(crate) struct Job {
    pub(crate) owner: JobOwner,
    seq: u64,
    pub(crate) fut: GpuFuture<JobOut>,
}

impl Job {
    pub(crate) fn seq(&self) -> u64 {
        self.seq
    }
}

/// A control request that went to a service and waits for its error scopes and reply.
pub(crate) struct ServicePending {
    pub(crate) task: TaskId,
    pub(crate) opcode: u16,
    index: usize,
    result: Option<(Handle, Kind)>,
    attached: Option<Rc<dyn Any>>,
    pub(crate) settle: Option<GpuFuture<Vec<Captured>>>,
    pub(crate) served: Option<Served>,
}

pub(crate) enum Pending {
    Reply(Reply),
    Service(ServicePending),
}

macro_rules! gpu {
    ($s:ident) => {
        Gpu { device: &$s.device, queue: &$s.queue, scopes: &$s.scopes, calls: &$s.calls }
    };
}

pub(crate) struct Core<A: Api, E: Embedder> {
    device: A::Device,
    queue: A::Queue,
    scopes: ErrorScopes<A>,
    /// Calls made through [`Gpu::call`] since the error scopes were last settled.
    calls: Cell<u64>,
    embedder: Rc<E>,
    services: Services<A, E>,
    service_of: HashMap<u16, usize>,
    tasks: BTreeMap<TaskId, TaskState<E>>,
    shared: HashMap<SharedEventId, Weak<SharedEvent<E>>>,
    runs: BTreeMap<RunId, Run<A, E>>,
    next_run: RunId,
    jobs: Vec<Job>,
    lost: Option<DeviceLostInfo>,
    pub(crate) lost_watch: Option<GpuFuture<DeviceLostInfo>>,
    reports: Vec<Report>,
}

fn settle<A: Api>(scopes: &ErrorScopes<A>, calls: &Cell<u64>) -> Option<GpuFuture<Vec<Captured>>> {
    if calls.get() == 0 {
        return None;
    }
    calls.set(0);
    Some(scopes.settle())
}

fn add_job<A: Api, E: Embedder>(jobs: &mut Vec<Job>, run: &mut Run<A, E>, id: RunId, fut: GpuFuture<JobOut>) {
    let seq = run.next_job;
    run.next_job += 1;
    run.outstanding.insert(seq);
    jobs.push(Job { owner: JobOwner::Run(id), seq, fut });
}

fn settled(f: GpuFuture<Vec<Captured>>) -> GpuFuture<JobOut> {
    Box::pin(async move { JobOut::Settled(f.await) })
}

fn written_back(f: GpuFuture<Result<(), Failure>>) -> GpuFuture<JobOut> {
    Box::pin(async move { JobOut::WrittenBack(f.await) })
}

fn work_done(f: GpuFuture<()>) -> GpuFuture<JobOut> {
    Box::pin(async move {
        f.await;
        JobOut::Done
    })
}

impl<A: Api, E: Embedder> Core<A, E> {
    pub(crate) fn new(device: A::Device, embedder: Rc<E>, services: Services<A, E>, config: Config) -> Core<A, E> {
        let mut service_of = HashMap::new();
        for (i, s) in services.control.iter().enumerate() {
            for op in s.opcodes() {
                let info = webmetal_wire::op_info(op);
                assert!(
                    info.is_some_and(|i| i.channel == webmetal_wire::Channel::Control),
                    "a control service claims {op:#06x}, which is not a control request"
                );
                assert!(!CORE_OPCODES.contains(&op), "a control service claims {op:#06x}, which the core serves");
                assert!(service_of.insert(op, i).is_none(), "two control services claim {op:#06x}");
            }
        }
        let scopes = ErrorScopes::new(&device, config.scopes);
        let lost_watch = Some(device.lost());
        Core {
            queue: device.queue(),
            device,
            scopes,
            calls: Cell::new(0),
            embedder,
            services,
            service_of,
            tasks: BTreeMap::new(),
            shared: HashMap::new(),
            runs: BTreeMap::new(),
            next_run: 0,
            jobs: Vec::new(),
            lost: None,
            lost_watch,
            reports: Vec::new(),
        }
    }

    fn now(&self) -> u64 {
        self.embedder.now().0
    }

    pub(crate) fn take_reports(&mut self) -> Vec<Report> {
        std::mem::take(&mut self.reports)
    }

    pub(crate) fn has_jobs(&self) -> bool {
        !self.jobs.is_empty()
    }

    pub(crate) fn next_run(&self) -> RunId {
        self.next_run
    }

    /// Whether a command buffer committed before `mark` (in `task`, or in any task) has not completed.
    pub(crate) fn runs_before(&self, mark: RunId, task: Option<TaskId>) -> bool {
        self.runs.range(..mark).any(|(_, r)| task.is_none_or(|t| r.task == t))
    }

    fn report(
        &mut self,
        task: TaskId,
        command_buffer: Option<Handle>,
        opcode: Option<u16>,
        offset: Option<u64>,
        f: &Failure,
    ) {
        if matches!(f.status, status::UNIMPLEMENTED | status::HOST_UNSUPPORTED) {
            self.reports.push(Report { task, command_buffer, opcode, offset, failure: f.clone() });
        }
    }

    // Control requests ------------------------------------------------------------------------------------------

    pub(crate) fn control(&mut self, task: TaskId, bytes: &[u8]) -> Result<Pending, webmetal_wire::DecodeError> {
        let (request, len) = match Request::decode(bytes, 0) {
            Ok(v) => v,
            Err(e) => {
                // A request whose opcode is known is answered; anything else cannot be framed as a reply.
                if bytes.len() >= 2 {
                    let op = u16::from_le_bytes([bytes[0], bytes[1]]);
                    if let Some(r) = schema::failure_reply(op, failure::invalid(format!("malformed request: {e}"))) {
                        return Ok(Pending::Reply(r));
                    }
                }
                return Err(e);
            }
        };
        let op = request.opcode();
        let reply = |f: Failure| Ok(Pending::Reply(schema::failure_reply(op, f).expect("control opcode")));
        if len != bytes.len() {
            return reply(failure::invalid(format!("{} bytes follow the request record", bytes.len() - len)));
        }
        if let Request::Open(o) = &request {
            return Ok(Pending::Reply(self.open(task, o)));
        }
        let Some(state) = self.tasks.get(&task) else {
            return reply(failure::invalid("request before Open"));
        };
        let mut problem = None;
        schema::request_handles(&request, &mut |u: HandleUse| {
            if problem.is_some() {
                return;
            }
            // Commit names the command buffer the host has not seen yet: Metal refuses a second commit of one
            // (-[IOGPUMetalCommandBuffer validate]: "commit an already committed command buffer").
            let r = if op == opcode::COMMIT && u.field == "command_buffer" {
                if state.table.is_fresh(u.handle) {
                    Ok(())
                } else {
                    Err(format!("command_buffer: handle {:#x} was already committed or used", u.handle.get()))
                }
            } else {
                state.table.check(&u)
            };
            if let Err(m) = r {
                problem = Some(m);
            }
        });
        if let Some(m) = problem {
            return reply(failure::invalid(m));
        }
        let r = match &request {
            Request::FetchEvents(_) => {
                return reply(failure::invalid("FetchEvents is answered by the embedder from the events posted to it"));
            }
            Request::Release(r) => self.release(task, r.object),
            Request::SetLabel(r) => self.set_label(task, r.object, r.label.clone()),
            Request::CreateCommandQueue(r) => {
                self.bind_core(task, r.queue, Kind::CommandQueue);
                self.tasks.get_mut(&task).expect("task").queues.insert(r.queue, QueueState::default());
                Reply::CreateCommandQueue(Ok(CreateCommandQueueReply {}))
            }
            Request::CreateCommandQueueWithDescriptor(r) => {
                // MTLCommandQueueDescriptorInternal SPI state. maxCommandBufferCount is the guest's (the plugin blocks
                // commandBuffer at capacity); qosLevel, commitSynchronously and disableAsyncCompletionDispatch govern
                // how the guest submits and runs handlers; disabling cross-queue hazard tracking or IO fencing allows
                // less ordering than the executor's single in-order queue gives, and the low-latency shared-event
                // options change timing only. What the remaining three do is not established, so only their
                // default (zero) is accepted.
                let d = &r.descriptor;
                let mut unknown = Vec::new();
                if d.device_partition != 0 {
                    unknown.push(format!("devicePartition {}", d.device_partition));
                }
                if d.is_open_gl_queue {
                    unknown.push("isOpenGLQueue".to_string());
                }
                if d.lock_parameter_buffer_size_to_max {
                    unknown.push("lockParameterBufferSizeToMax".to_string());
                }
                if !unknown.is_empty() {
                    return reply(failure::unimplemented(format!(
                        "-[MTLDeviceSPI newCommandQueueWithDescriptor:] with {}: not implemented by the host",
                        unknown.join(", ")
                    )));
                }
                self.bind_core(task, r.queue, Kind::CommandQueue);
                self.tasks.get_mut(&task).expect("task").queues.insert(r.queue, QueueState::default());
                Reply::CreateCommandQueueWithDescriptor(Ok(CreateCommandQueueWithDescriptorReply {}))
            }
            Request::SetGPUPriority(_)
            | Request::SetGPUPriorityOffset(_)
            | Request::SetBackgroundGPUPriority(_)
            | Request::SetBackgroundGPUPriorityOffset(_) => {
                // The BOOL these return on 21H461 is not measured.
                return reply(failure::unimplemented(format!(
                    "{}: the value the A10X driver returns is not established",
                    failure::record_methods(op)
                )));
            }
            Request::Commit(c) => self.commit(task, c),
            Request::RegisterSharedEvent(r) => match self.embedder.shared_event_resolve(task, r.port) {
                Err(e) => return reply(failure::invalid(format!("RegisterSharedEvent: {e}"))),
                Ok(id) => {
                    let event = match self.shared.get(&id).and_then(Weak::upgrade) {
                        Some(e) => e,
                        None => {
                            let e = Rc::new(SharedEvent { id, embedder: self.embedder.clone() });
                            self.shared.insert(id, Rc::downgrade(&e));
                            e
                        }
                    };
                    self.bind_core(task, r.event, Kind::Event);
                    self.tasks.get_mut(&task).expect("task").events.insert(r.event, EventRef::Shared(event));
                    Reply::RegisterSharedEvent(Ok(RegisterSharedEventReply {}))
                }
            },
            Request::CreateEvent(r) => {
                // A new event's value is 0: a wait for 0 on a new event is already satisfied (measured on Apple M4,
                // macOS 27.0.1).
                self.bind_core(task, r.event, Kind::Event);
                self.tasks.get_mut(&task).expect("task").events.insert(r.event, EventRef::Local(Rc::new(Cell::new(0))));
                Reply::CreateEvent(Ok(CreateEventReply {}))
            }
            Request::CreateFence(r) => {
                self.bind_core(task, r.fence, Kind::Fence);
                Reply::CreateFence(Ok(CreateFenceReply {}))
            }
            other => return Ok(self.serve(task, other)),
        };
        Ok(Pending::Reply(r))
    }

    fn open(&mut self, task: TaskId, o: &webmetal_wire::Open) -> Reply {
        if self.tasks.contains_key(&task) {
            return Reply::Open(Err(failure::invalid("Open on a connection that is already open")));
        }
        if o.schema_hash.bytes != webmetal_wire::SCHEMA_HASH {
            return Reply::Open(Err(failure::invalid(format!(
                "schema hash {} differs from the host's {}",
                hex(&o.schema_hash.bytes),
                hex(&webmetal_wire::SCHEMA_HASH)
            ))));
        }
        self.tasks.insert(task, TaskState { table: Table::default(), queues: HashMap::new(), events: HashMap::new() });
        Reply::Open(Ok(OpenReply { schema_hash: webmetal_wire::Hash { bytes: webmetal_wire::SCHEMA_HASH } }))
    }

    fn bind_core(&mut self, task: TaskId, handle: Handle, kind: Kind) {
        let state = self.tasks.get_mut(&task).expect("task");
        state.table.bind(handle, Entry { kind, owner: Owner::Core, label: None, payload: None }).expect("fresh handle");
    }

    fn release(&mut self, task: TaskId, handle: Handle) -> Reply {
        let state = self.tasks.get_mut(&task).expect("task");
        let entry = state.table.unbind(handle).expect("live handle");
        match entry.owner {
            Owner::Core => match entry.kind {
                Kind::CommandQueue => {
                    let q = state.queues.get_mut(&handle).expect("queue state");
                    q.released = true;
                    if q.idle() {
                        state.queues.remove(&handle);
                    }
                }
                Kind::Event => {
                    // Command buffers that use the event keep their own reference.
                    state.events.remove(&handle);
                }
                // A command buffer keeps running after its handle is released; encoders and fences hold no state.
                _ => {}
            },
            Owner::Service(i) => {
                self.services.control[i].release(task, handle, &entry);
                self.detach_calls(task);
            }
        }
        Reply::Release(Ok(ReleaseReply {}))
    }

    fn set_label(&mut self, task: TaskId, handle: Handle, label: Option<String>) -> Reply {
        let state = self.tasks.get_mut(&task).expect("task");
        let entry = state.table.get_mut(handle).expect("live handle");
        entry.label = label;
        if let Owner::Service(i) = entry.owner {
            let entry = entry.clone();
            let state = self.tasks.get(&task).expect("task");
            let mut cx =
                ServiceCx { task, gpu: gpu!(self), embedder: &*self.embedder, objects: &state.table, attached: None };
            self.services.control[i].set_label(&mut cx, handle, &entry);
            self.detach_calls(task);
        }
        Reply::SetLabel(Ok(SetLabelReply {}))
    }

    fn serve(&mut self, task: TaskId, request: &Request) -> Pending {
        let op = request.opcode();
        let Some(&index) = self.service_of.get(&op) else {
            return Pending::Reply(
                schema::failure_reply(op, failure::unimplemented_record(op)).expect("control opcode"),
            );
        };
        if let Some(info) = &self.lost {
            return Pending::Reply(schema::failure_reply(op, failure::device_removed(info)).expect("control opcode"));
        }
        let mut result = None;
        schema::request_handles(request, &mut |u| {
            if u.role == Role::Result {
                result = Some((u.handle, u.kind));
            }
        });
        let state = self.tasks.get(&task).expect("task");
        let mut cx =
            ServiceCx { task, gpu: gpu!(self), embedder: &*self.embedder, objects: &state.table, attached: None };
        let served = self.services.control[index].serve(&mut cx, request);
        let attached = cx.attached.take();
        let settle = settle(&self.scopes, &self.calls);
        Pending::Service(ServicePending { task, opcode: op, index, result, attached, settle, served: Some(served) })
    }

    /// Complete a request a service served, once its error scopes settled and its reply is known.
    pub(crate) fn finish_service(&mut self, p: ServicePending, captured: Vec<Captured>, reply: Reply) -> Reply {
        let op = p.opcode;
        let fail = |f: Failure| schema::failure_reply(op, f).expect("control opcode");
        let made = |kind: Kind, payload: Option<Rc<dyn Any>>| Entry {
            kind,
            owner: Owner::Service(p.index),
            label: None,
            payload,
        };
        let succeeded = schema::reply_failure(&reply).is_none();
        if reply.request_opcode() != op {
            fail(failure::unimplemented(format!(
                "the service for {} replied to opcode {:#06x}",
                failure::record_methods(op),
                reply.request_opcode()
            )))
        } else if let Some(cause) = GpuCause::worst(&captured) {
            if let (true, Some((h, kind))) = (succeeded, p.result) {
                self.services.control[p.index].release(p.task, h, &made(kind, p.attached.clone()));
                self.detach_calls(p.task);
            }
            fail(cause.request_failure())
        } else {
            if let (true, Some((h, kind))) = (succeeded, p.result) {
                match self.tasks.get_mut(&p.task) {
                    Some(state) => state.table.bind(h, made(kind, p.attached.clone())).expect("fresh handle"),
                    // The task exited while the request was pending.
                    None => {
                        self.services.control[p.index].release(p.task, h, &made(kind, p.attached.clone()));
                        self.detach_calls(p.task);
                    }
                }
            }
            reply
        }
    }

    /// Keep a report of a reply that fails with status 2 or 3.
    pub(crate) fn report_reply(&mut self, task: TaskId, reply: &Reply) {
        if let Some(f) = schema::reply_failure(reply) {
            let f = f.clone();
            self.report(task, None, Some(reply.request_opcode()), None, &f);
        }
    }

    // Commit ---------------------------------------------------------------------------------------------------

    fn commit(&mut self, task: TaskId, c: &Commit) -> Reply {
        let fail = |f: Failure| Reply::Commit(Err(f));
        let Ok(len) = usize::try_from(c.stream_length) else {
            return fail(failure::host_unsupported(format!(
                "a {}-byte stream does not fit in host memory",
                c.stream_length
            )));
        };
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(len).is_err() {
            return fail(failure::host_unsupported(format!("a {len}-byte stream does not fit in host memory")));
        }
        bytes.resize(len, 0);
        if let Err(e) = self.embedder.read_guest(task, c.stream.0, &mut bytes) {
            return fail(failure::invalid(format!("stream: {e}")));
        }
        let records: Rc<[(usize, Command)]> = match Command::decode_all(&bytes) {
            Ok(r) => r.into(),
            Err(e) => return fail(failure::invalid(format!("stream: {e}"))),
        };
        let services = &self.services;
        let implemented = |k: EncoderKind| match k {
            EncoderKind::Render | EncoderKind::ParallelRender => services.render.is_some(),
            EncoderKind::Compute => services.compute.is_some(),
            EncoderKind::Blit => services.blit.is_some(),
        };
        let state = self.tasks.get_mut(&task).expect("task");
        let plan = match stream::plan(&records, &state.table, &c.resources, &implemented) {
            Ok(p) => p,
            Err(v) => return fail(failure::invalid(v.to_string())),
        };

        let core_entry = |kind, label| Entry { kind, owner: Owner::Core, label, payload: None };
        state.table.bind(c.command_buffer, core_entry(Kind::CommandBuffer, c.label.clone())).expect("fresh handle");
        for (h, _) in &plan.encoders {
            state.table.bind(*h, core_entry(Kind::Encoder, None)).expect("fresh handle");
        }
        let mut events = HashMap::new();
        for step in &plan.steps {
            if let Step::Wait { event, .. } | Step::Signal { event, .. } = step {
                events.insert(*event, state.events.get(event).expect("live event").clone());
            }
        }
        let info = Rc::new(CommitInfo {
            task,
            command_buffer: c.command_buffer,
            queue: c.queue,
            label: c.label.clone(),
            retained_references: c.retained_references,
            resources: c.resources.clone(),
            records,
        });
        let mut failure = plan.unimplemented.map(|(offset, f)| (f, offset));
        if failure.is_none()
            && self.lost.is_none()
            && let Some(co) = self.services.coherence.as_mut()
        {
            let state = self.tasks.get(&task).expect("task");
            let mut cx =
                ServiceCx { task, gpu: gpu!(self), embedder: &*self.embedder, objects: &state.table, attached: None };
            if let Err(f) = co.commit(&mut cx, &info) {
                failure = Some((f, NO_RECORD));
            }
        }

        let id = self.next_run;
        self.next_run += 1;
        let label = c.label.clone().unwrap_or_default();
        let mut run = Run {
            task,
            queue: c.queue,
            info,
            steps: plan.steps,
            cursor: 0,
            events,
            failure,
            recording: Recording::new(label),
            started: false,
            kernel_start: 0,
            kernel_end: 0,
            gpu_start: None,
            gpu_end: 0,
            scheduled: false,
            next_job: 0,
            outstanding: BTreeSet::new(),
            actions: VecDeque::new(),
            complete: false,
        };
        if let Some(f) = settle(&self.scopes, &self.calls) {
            add_job(&mut self.jobs, &mut run, id, settled(f));
        }
        self.runs.insert(id, run);
        let state = self.tasks.get_mut(&task).expect("task");
        state.queues.get_mut(&c.queue).expect("queue state").backlog.push_back(id);
        self.settle_state();
        Reply::Commit(Ok(CommitReply {}))
    }

    // Replay ---------------------------------------------------------------------------------------------------

    /// Fire ready actions, post completions in queue order, and replay runnable command buffers, until nothing
    /// changes. Returns whether anything changed.
    pub(crate) fn settle_state(&mut self) -> bool {
        let mut any = false;
        loop {
            let a = self.fire_actions();
            let b = self.post_completions();
            let c = self.advance();
            if !(a || b || c) {
                return any;
            }
            any = true;
        }
    }

    /// Replay every queue's current command buffer as far as it goes, taking queues in the commit order of the command
    /// buffer each would replay next: one valid Metal schedule, since Metal orders command buffers only within a
    /// queue and through events.
    fn advance(&mut self) -> bool {
        let mut any = false;
        loop {
            let mut next: Vec<(RunId, TaskId, Handle)> = Vec::new();
            for (t, state) in &self.tasks {
                for (q, qs) in &state.queues {
                    if let Some(id) = qs.active.or(qs.backlog.front().copied()) {
                        next.push((id, *t, *q));
                    }
                }
            }
            next.sort();
            let mut progress = false;
            for (id, t, q) in next {
                let qs = self.tasks.get_mut(&t).expect("task").queues.get_mut(&q).expect("queue");
                if qs.active.is_none() {
                    qs.backlog.pop_front();
                    qs.active = Some(id);
                }
                progress |= self.step(id);
            }
            if !progress {
                return any;
            }
            any = true;
        }
    }

    /// Replay one command buffer until it blocks on an event or all its work is submitted.
    fn step(&mut self, id: RunId) -> bool {
        let now = self.now();
        let run = self.runs.get_mut(&id).expect("run");
        let state = self.tasks.get(&run.task).expect("task");
        let gpu = gpu!(self);
        let mut progress = false;
        if !run.started {
            run.started = true;
            run.kernel_start = now;
            progress = true;
        }
        if let Some(info) = &self.lost {
            // Committed after the device was lost.
            run.failure = Some((failure::device_removed(info), NO_RECORD));
        }
        if run.failure.is_some() {
            run.cursor = run.steps.len();
        }
        while run.cursor < run.steps.len() {
            let step = run.steps[run.cursor].clone();
            match step {
                Step::Encoder(segment) => {
                    let service = match segment.kind {
                        EncoderKind::Render | EncoderKind::ParallelRender => self.services.render.as_mut(),
                        EncoderKind::Compute => self.services.compute.as_mut(),
                        EncoderKind::Blit => self.services.blit.as_mut(),
                    }
                    .expect("the plan checked the service exists");
                    let info = run.info.clone();
                    let mut cx = EncodeCx {
                        task: run.task,
                        info: &info,
                        gpu: gpu!(self),
                        embedder: &*self.embedder,
                        objects: &state.table,
                        recording: &mut run.recording,
                    };
                    if let Err(rf) = service.encode(&mut cx, &segment) {
                        run.recording.discard();
                        run.fail(rf.failure, rf.offset);
                        run.cursor = run.steps.len();
                        break;
                    }
                }
                Step::Wait { event, value, offset } => {
                    let reached = match run.events.get(&event).expect("event") {
                        EventRef::Local(v) => Ok(v.get() >= value),
                        EventRef::Shared(s) => self.embedder.shared_event_value(s.id).map(|v| v >= value),
                    };
                    match reached {
                        Ok(true) => {}
                        Ok(false) => {
                            // The work before the wait runs now; the queue waits here (measured on Apple M4, macOS
                            // 27.0.1: work encoded before a wait completes before the signal, and the command buffer
                            // and every later one of its queue stay unscheduled until the event reaches the value).
                            if run.recording.submit(&gpu) {
                                run.gpu_start.get_or_insert(now);
                            }
                            if let Some(f) = settle(&self.scopes, &self.calls) {
                                add_job(&mut self.jobs, run, id, settled(f));
                            }
                            return progress;
                        }
                        Err(e) => {
                            run.recording.discard();
                            run.fail(failure::invalid(format!("EncodeWaitForEvent: {e}")), offset);
                            run.cursor = run.steps.len();
                            break;
                        }
                    }
                }
                Step::Signal { event, value, offset } => {
                    // Split the submission: everything before the signal is on the queue before anything a waiter
                    // submits after seeing the value.
                    if run.recording.submit(&gpu) {
                        run.gpu_start.get_or_insert(now);
                    }
                    match run.events.get(&event).expect("event").clone() {
                        EventRef::Local(v) => {
                            // Only GPU work observes an MTLEvent, and work submitted after this point runs after the
                            // work before it on WebGPU's queue, so the value is set now. Values never decrease: after
                            // signals of 10 and then 5, a wait for 7 is satisfied (measured on Apple M4, macOS 27.0.1).
                            v.set(v.get().max(value));
                        }
                        EventRef::Shared(s) => {
                            // The CPU observes a shared event: the value is set once the work before the signal has
                            // completed and its writes to CPU-visible memory are in guest memory.
                            let mut wb = None;
                            if let Some(co) = self.services.coherence.as_mut() {
                                let info = run.info.clone();
                                let mut cx = EncodeCx {
                                    task: run.task,
                                    info: &info,
                                    gpu: gpu!(self),
                                    embedder: &*self.embedder,
                                    objects: &state.table,
                                    recording: &mut run.recording,
                                };
                                match co.write_back(&mut cx, &info) {
                                    Ok(f) => wb = f,
                                    Err(f) => {
                                        run.recording.discard();
                                        run.fail(f, offset);
                                        run.cursor = run.steps.len();
                                        break;
                                    }
                                }
                                if run.recording.submit(&gpu) {
                                    run.gpu_start.get_or_insert(now);
                                }
                            }
                            if let Some(f) = settle(&self.scopes, &self.calls) {
                                add_job(&mut self.jobs, run, id, settled(f));
                            }
                            add_job(&mut self.jobs, run, id, work_done(self.queue.on_submitted_work_done()));
                            if let Some(f) = wb {
                                add_job(&mut self.jobs, run, id, written_back(f));
                            }
                            let after = run.next_job;
                            run.actions.push_back(Action { after, kind: ActionKind::SignalShared(s, value, offset) });
                        }
                    }
                }
                Step::PushDebugGroup { label } => run.recording.push_debug_group(&gpu, &label),
                Step::PopDebugGroup => run.recording.pop_debug_group(&gpu),
            }
            run.cursor += 1;
            progress = true;
        }

        // Every step has run: submit the rest, then the command buffer is scheduled.
        let mut wb = None;
        if run.failure.is_none() && self.lost.is_none() {
            if let Some(co) = self.services.coherence.as_mut() {
                let info = run.info.clone();
                let mut cx = EncodeCx {
                    task: run.task,
                    info: &info,
                    gpu: gpu!(self),
                    embedder: &*self.embedder,
                    objects: &state.table,
                    recording: &mut run.recording,
                };
                match co.write_back(&mut cx, &info) {
                    Ok(f) => wb = f,
                    Err(f) => {
                        run.recording.discard();
                        run.fail(f, NO_RECORD);
                    }
                }
            }
            if run.recording.submit(&gpu) {
                run.gpu_start.get_or_insert(now);
            }
        } else {
            run.recording.discard();
        }
        run.kernel_end = self.embedder.now().0;
        run.scheduled = true;
        self.embedder.post(
            run.task,
            Some(run.queue),
            Event::CommandBufferScheduled(CommandBufferScheduled {
                command_buffer: run.info.command_buffer,
                kernel_start_time: run.kernel_start,
                kernel_end_time: run.kernel_end,
            }),
        );
        if self.lost.is_none() {
            if let Some(f) = settle(&self.scopes, &self.calls) {
                add_job(&mut self.jobs, run, id, settled(f));
            }
            add_job(&mut self.jobs, run, id, work_done(self.queue.on_submitted_work_done()));
            if let Some(f) = wb {
                add_job(&mut self.jobs, run, id, written_back(f));
            }
        }
        let after = run.next_job;
        run.actions.push_back(Action { after, kind: ActionKind::Complete });
        let qs = self.tasks.get_mut(&run.task).expect("task").queues.get_mut(&run.queue).expect("queue");
        qs.active = None;
        qs.scheduled.push_back(id);
        true
    }

    fn fire_actions(&mut self) -> bool {
        let mut changed = false;
        let now = self.now();
        for run in self.runs.values_mut() {
            let first_outstanding = run.outstanding.first().copied().unwrap_or(u64::MAX);
            while run.actions.front().is_some_and(|a| a.after <= first_outstanding) {
                let action = run.actions.pop_front().expect("action");
                changed = true;
                match action.kind {
                    ActionKind::SignalShared(event, value, offset) => {
                        // A command buffer that failed stops at the failure; its later signals do not happen.
                        if run.failure.is_none()
                            && self.lost.is_none()
                            && let Err(e) = self.embedder.shared_event_signal(event.id, value)
                        {
                            run.fail(failure::invalid(format!("EncodeSignalEvent: {e}")), offset);
                        }
                    }
                    ActionKind::Complete => {
                        run.complete = true;
                        run.gpu_end = now;
                    }
                }
            }
        }
        changed
    }

    fn post_completions(&mut self) -> bool {
        let mut done = Vec::new();
        for (t, state) in self.tasks.iter_mut() {
            for qs in state.queues.values_mut() {
                while let Some(&id) = qs.scheduled.front() {
                    if !self.runs.get(&id).expect("run").complete {
                        break;
                    }
                    qs.scheduled.pop_front();
                    done.push((*t, id));
                }
            }
            state.queues.retain(|_, q| !(q.released && q.idle()));
        }
        let changed = !done.is_empty();
        for (task, id) in done {
            let run = self.runs.remove(&id).expect("run");
            let gpu_start = run.gpu_start.unwrap_or(run.kernel_end);
            let (status, failure, offset) = match run.failure {
                None => (command_buffer_status::COMPLETED, failure::ok(), NO_RECORD),
                Some((f, o)) => (command_buffer_status::ERROR, f, o),
            };
            self.report(task, Some(run.info.command_buffer), None, (offset != NO_RECORD).then_some(offset), &failure);
            self.embedder.post(
                task,
                Some(run.queue),
                Event::CommandBufferCompleted(CommandBufferCompleted {
                    command_buffer: run.info.command_buffer,
                    status,
                    failure,
                    failed_offset: offset,
                    kernel_start_time: run.kernel_start,
                    kernel_end_time: run.kernel_end,
                    gpu_start_time: gpu_start,
                    gpu_end_time: run.gpu_end.max(gpu_start),
                }),
            );
        }
        changed
    }

    // Jobs and device loss ---------------------------------------------------------------------------------------

    pub(crate) fn take_jobs(&mut self) -> Vec<Job> {
        std::mem::take(&mut self.jobs)
    }

    pub(crate) fn return_jobs(&mut self, jobs: Vec<Job>) {
        // Jobs of runs dropped meanwhile (device loss, task exit) are dropped with them.
        for job in jobs {
            let live = match job.owner {
                JobOwner::Run(id) => self.runs.get(&id).is_some_and(|r| r.outstanding.contains(&job.seq)),
                JobOwner::Detached(task) => self.lost.is_none() && self.tasks.contains_key(&task),
            };
            if live {
                self.jobs.push(job);
            }
        }
    }

    /// Settle the error scopes over calls made outside a request or command buffer.
    fn detach_calls(&mut self, task: TaskId) {
        if let Some(f) = settle(&self.scopes, &self.calls) {
            self.jobs.push(Job { owner: JobOwner::Detached(task), seq: 0, fut: settled(f) });
        }
    }

    pub(crate) fn job_done(&mut self, owner: JobOwner, seq: u64, out: JobOut) {
        let id = match owner {
            JobOwner::Run(id) => id,
            JobOwner::Detached(task) => {
                if let JobOut::Settled(captured) = out {
                    for c in &captured {
                        let f = GpuCause::of(c).command_buffer_failure();
                        self.report(task, None, None, None, &f);
                    }
                }
                return;
            }
        };
        let lost = self.lost.is_some();
        let Some(run) = self.runs.get_mut(&id) else { return };
        if !run.outstanding.remove(&seq) || lost {
            return;
        }
        match out {
            JobOut::Done => {}
            JobOut::Settled(captured) => {
                if let Some(cause) = GpuCause::worst(&captured) {
                    run.fail(cause.command_buffer_failure(), NO_RECORD);
                }
            }
            JobOut::WrittenBack(Err(f)) => run.fail(f, NO_RECORD),
            JobOut::WrittenBack(Ok(())) => {}
        }
    }

    /// The WebGPU device was lost: every command buffer not yet completed, and every one committed later, completes
    /// with the device-removed failure; every task is told with a DeviceLost event.
    pub(crate) fn on_lost(&mut self, info: DeviceLostInfo) {
        let f = failure::device_removed(&info);
        self.lost = Some(info);
        self.jobs.clear();
        let now = self.now();
        for run in self.runs.values_mut() {
            run.outstanding.clear();
            run.actions.clear();
            run.recording.discard();
            run.failure = Some((f.clone(), NO_RECORD));
            if run.scheduled {
                run.complete = true;
                run.gpu_end = now;
            }
        }
        let tasks: Vec<TaskId> = self.tasks.keys().copied().collect();
        for t in tasks {
            self.embedder.post(t, None, Event::DeviceLost(DeviceLost { failure: f.clone() }));
        }
        self.settle_state();
    }

    pub(crate) fn task_exited(&mut self, task: TaskId) {
        self.tasks.remove(&task);
        self.runs.retain(|_, r| r.task != task);
        let runs = &self.runs;
        self.jobs.retain(|j| match j.owner {
            JobOwner::Run(id) => runs.contains_key(&id),
            JobOwner::Detached(t) => t != task,
        });
        for s in &mut self.services.control {
            s.task_exited(task);
        }
        for s in [&mut self.services.render, &mut self.services.compute, &mut self.services.blit].into_iter().flatten()
        {
            s.task_exited(task);
        }
        if let Some(c) = self.services.coherence.as_mut() {
            c.task_exited(task);
        }
        // What the services' teardown calls raise is reported for no task that still exists; settle it anyway so it
        // is not attributed to the next request.
        if let Some(f) = settle(&self.scopes, &self.calls) {
            self.jobs.push(Job { owner: JobOwner::Detached(task), seq: 0, fut: settled(f) });
        }
        self.settle_state();
    }

    pub(crate) fn encode_reply(&self, reply: &Reply) -> Vec<u8> {
        let mut out = Vec::new();
        if let Err(e) = reply.encode(&mut out) {
            out.clear();
            let f = failure::unimplemented(format!("the host's reply did not encode: {e}"));
            schema::failure_reply(reply.request_opcode(), f)
                .expect("control opcode")
                .encode(&mut out)
                .expect("a failure reply encodes");
        }
        out
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
