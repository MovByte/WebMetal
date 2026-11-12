//! The executor on a native WebGPU implementation (Dawn or wgpu-native, whichever the build links): command buffer
//! status and completion order, events, fences, debug groups, protocol violations, unimplemented records, device loss
//! and out-of-memory, each checked against the reference outputs real Metal recorded (`tests/reference/goldens`)
//! where the Metal reference harness has a case for it.
//!
//! The executor runs here with stand-ins for the modules that plug into it: a memory service that makes buffers and
//! keeps their guest bytes and GPU copies coherent (whole-buffer shadow comparison, exact for programs that do not
//! write a buffer from the CPU while the GPU may be writing it, which none of these do), and a blit service with
//! `fillBuffer:range:value:` and `copyFromBuffer:...` over WebGPU copies. They make the order of GPU work observable in
//! guest memory; they are not the memory and blit modules. Error scopes run per call and strict: an error that escapes
//! every scope fails the test.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use webmetal_wire::*;

use super::failure::{COMMAND_BUFFER_ERROR_DOMAIN, NO_RECORD, command_buffer_error, command_buffer_status};
use super::*;
use crate::embed::*;
use crate::gpu::idl::{
    BufferDescriptor, BufferUsage, DeviceDescriptor, Extent3D, Extent3DDict, MapMode, RequestAdapterOptions,
    TextureDescriptor, TextureFormat, TextureUsage,
};
use crate::gpu::native::{Native, NativeDevice, NativeGpu};
use crate::gpu::scope::{Granularity, Policy};
use crate::gpu::{Adapter, Buffer as _, CommandEncoder as _, DebugCommands, Device as _, Gpu as _, MappedRange as _};
use crate::gpu::{Queue as _, Size64, negotiate};

const WAIT: Duration = Duration::from_secs(30);

// The embedder --------------------------------------------------------------------------------------------------------

/// A task's mapped regions: base address and bytes.
type Regions = Vec<(u64, Vec<u8>)>;

/// Guest memory, shared events, posted events and the clock of a test.
struct TestEmbedder {
    start: Instant,
    memory: RefCell<HashMap<TaskId, Regions>>,
    next_addr: Cell<u64>,
    /// Shared event kernel objects: value by id.
    events: RefCell<HashMap<u64, u64>>,
    ports: RefCell<HashMap<(TaskId, u32), u64>>,
    /// GPU-side signals the executor made: (event, value).
    signals: RefCell<Vec<(u64, u64)>>,
    released: RefCell<Vec<u64>>,
    posted: RefCell<Vec<(TaskId, Option<Handle>, Event)>>,
}

impl TestEmbedder {
    fn new() -> TestEmbedder {
        TestEmbedder {
            start: Instant::now(),
            memory: RefCell::new(HashMap::new()),
            next_addr: Cell::new(0x1_0000_0000),
            events: RefCell::new(HashMap::new()),
            ports: RefCell::new(HashMap::new()),
            signals: RefCell::new(Vec::new()),
            released: RefCell::new(Vec::new()),
            posted: RefCell::new(Vec::new()),
        }
    }

    fn alloc(&self, task: TaskId, bytes: Vec<u8>) -> u64 {
        let addr = self.next_addr.get();
        self.next_addr.set(addr + (bytes.len() as u64).div_ceil(0x4000).max(1) * 0x4000);
        self.memory.borrow_mut().entry(task).or_default().push((addr, bytes));
        addr
    }

    fn region<T>(
        &self,
        task: TaskId,
        addr: u64,
        len: usize,
        f: impl FnOnce(&mut [u8]) -> T,
    ) -> Result<T, GuestMemoryError> {
        let mut memory = self.memory.borrow_mut();
        let fault = GuestMemoryError { task, addr, len: len as u64, fault: addr, kind: GuestMemoryErrorKind::Unmapped };
        let regions = memory.get_mut(&task).ok_or(GuestMemoryError { kind: GuestMemoryErrorKind::NoTask, ..fault })?;
        for (base, bytes) in regions.iter_mut() {
            if addr >= *base && addr + len as u64 <= *base + bytes.len() as u64 {
                let at = (addr - *base) as usize;
                return Ok(f(&mut bytes[at..at + len]));
            }
        }
        Err(fault)
    }

    fn guest(&self, task: TaskId, addr: u64, len: usize) -> Vec<u8> {
        self.region(task, addr, len, |b| b.to_vec()).expect("mapped")
    }

    /// A shared event kernel object with `value`, named by `port` in `task`.
    fn shared_event(&self, task: TaskId, port: u32, id: u64, value: u64) {
        self.events.borrow_mut().insert(id, value);
        self.ports.borrow_mut().insert((task, port), id);
    }

    /// The kernel object's `setSignaledValue:` from the CPU. Shared event values never decrease: after a CPU set of 10
    /// a GPU signal of 5 and a CPU set of 3 both leave 10 (measured on Apple M4, macOS 27.0.1).
    fn cpu_signal(&self, id: u64, value: u64) {
        let mut e = self.events.borrow_mut();
        let v = e.get_mut(&id).expect("event");
        *v = (*v).max(value);
    }

    fn value(&self, id: u64) -> u64 {
        self.events.borrow()[&id]
    }

    fn take_posted(&self) -> Vec<(TaskId, Option<Handle>, Event)> {
        std::mem::take(&mut *self.posted.borrow_mut())
    }
}

impl Embedder for TestEmbedder {
    fn read_guest(&self, task: TaskId, addr: u64, dst: &mut [u8]) -> Result<(), GuestMemoryError> {
        self.region(task, addr, dst.len(), |b| dst.copy_from_slice(b))
    }
    fn write_guest(&self, task: TaskId, addr: u64, src: &[u8]) -> Result<(), GuestMemoryError> {
        self.region(task, addr, src.len(), |b| b.copy_from_slice(src))
    }
    fn surface_lookup(&self, id: SurfaceId) -> Result<SurfaceDesc, SurfaceError> {
        Err(SurfaceError::Unknown(id))
    }
    fn read_surface(&self, id: SurfaceId, _: u64, _: &mut [u8]) -> Result<(), SurfaceError> {
        Err(SurfaceError::Unknown(id))
    }
    fn write_surface_changed(&self, id: SurfaceId, _: u64, _: &[u8], _: &[u8]) -> Result<(), SurfaceError> {
        Err(SurfaceError::Unknown(id))
    }
    fn surface_note_gpu_write(&self, _: SurfaceId, _: u32) {}
    fn shared_event_resolve(&self, task: TaskId, port: u32) -> Result<SharedEventId, SharedEventError> {
        self.ports
            .borrow()
            .get(&(task, port))
            .map(|&id| SharedEventId(id))
            .ok_or(SharedEventError::NotAnEvent { task, port })
    }
    fn shared_event_value(&self, event: SharedEventId) -> Result<u64, SharedEventError> {
        self.events.borrow().get(&event.0).copied().ok_or(SharedEventError::Gone(event))
    }
    fn shared_event_signal(&self, event: SharedEventId, value: u64) -> Result<(), SharedEventError> {
        let mut e = self.events.borrow_mut();
        let v = e.get_mut(&event.0).ok_or(SharedEventError::Gone(event))?;
        *v = (*v).max(value);
        self.signals.borrow_mut().push((event.0, value));
        Ok(())
    }
    fn shared_event_release(&self, event: SharedEventId) {
        self.released.borrow_mut().push(event.0);
    }
    fn post(&self, task: TaskId, queue: Option<Handle>, event: Event) {
        self.posted.borrow_mut().push((task, queue, event));
    }
    fn present(&self, frame: &PresentedFrame) -> Result<(), PresentError> {
        Err(PresentError::UnknownDisplay(frame.display))
    }
    fn now(&self) -> MonotonicNanos {
        // Positive from the first reading, as Metal's times are once set.
        MonotonicNanos(self.start.elapsed().as_nanos() as u64 + 1)
    }
}

// Stand-in services ---------------------------------------------------------------------------------------------------

/// The `length` of a CreateBuffer whose WebGPU allocation the memory stand-in makes fail with out-of-memory.
const FAULT_LENGTH: u64 = 0xdead_0000;

/// A buffer: its WebGPU copy, its guest bytes and the shadow of what they held when last synchronized.
struct TestBuffer {
    gpu: <Native as Api>::Buffer,
    length: u64,
    size: u64,
    contents: u64,
    shadow: RefCell<Vec<u8>>,
}

/// Allocate a texture no GPU can hold (`maxTextureDimension2D`² texels of 16 bytes in `maxTextureArrayLayers`
/// layers): WebGPU reports a `GPUOutOfMemoryError` (Metal returns nil for it, measured on Apple M4, macOS 27.0.1).
fn exhaust_memory(gpu: &super::Gpu<'_, Native>) {
    let limits = gpu.device().limits();
    let side = limits.max_texture_dimension_2d.expect("limit");
    let layers = limits.max_texture_array_layers.expect("limit");
    let mut size = Extent3DDict::new(side);
    size.height = side;
    size.depth_or_array_layers = layers;
    let desc = TextureDescriptor::new(Extent3D::Extent3DDict(size), TextureFormat::Rgba32float, TextureUsage::COPY_SRC);
    let t = gpu.call("GPUDevice.createTexture", "exhaust", |d| d.create_texture(&desc));
    assert!(t.is_ok(), "createTexture threw: {t:?}");
}

struct Memory {
    embedder: Rc<TestEmbedder>,
}

impl ControlService<Native, TestEmbedder> for Memory {
    fn opcodes(&self) -> Vec<u16> {
        vec![opcode::CREATE_BUFFER]
    }

    fn serve(&mut self, cx: &mut ServiceCx<'_, Native, TestEmbedder>, request: &Request) -> Served {
        let Request::CreateBuffer(r) = request else { unreachable!() };
        // The injected fault: the allocation fails; the buffer made after it is never used.
        let fault = r.length == FAULT_LENGTH;
        if fault {
            exhaust_memory(cx.gpu());
        }
        let size = if fault { 4 } else { r.length.div_ceil(4).max(1) * 4 };
        let mut desc = BufferDescriptor::new(Size64::new(size).unwrap(), BufferUsage::COPY_SRC | BufferUsage::COPY_DST);
        desc.label = format!("buffer {:#x}", r.buffer.get());
        let gpu = cx.gpu().call("GPUDevice.createBuffer", &desc.label.clone(), |d| d.create_buffer(&desc)).unwrap();
        let shadow = if r.contents.0 != 0 && !fault {
            let mut bytes = vec![0; r.length as usize];
            self.embedder.read_guest(cx.task(), r.contents.0, &mut bytes).unwrap();
            let mut padded = bytes.clone();
            padded.resize(size as usize, 0);
            cx.gpu()
                .call("GPUQueue.writeBuffer", "", |_| {
                    cx.gpu().queue().write_buffer(&gpu, Size64::new(0).unwrap(), &padded, Size64::new(0).unwrap(), None)
                })
                .unwrap();
            bytes
        } else {
            Vec::new()
        };
        let b = TestBuffer { gpu, length: r.length, size, contents: r.contents.0, shadow: RefCell::new(shadow) };
        cx.attach(Rc::new(b));
        Served::Ready(Reply::CreateBuffer(Ok(CreateBufferReply {})))
    }

    fn release(&mut self, _: TaskId, _: Handle, _: &Entry) {}

    fn task_exited(&mut self, _: TaskId) {}
}

fn buffers(objects: &Table, handles: &[Handle]) -> Vec<(Handle, Rc<dyn Any>)> {
    handles
        .iter()
        .filter_map(|h| objects.get(*h).and_then(|e| e.payload.clone()).map(|p| (*h, p)))
        .filter(|(_, p)| p.downcast_ref::<TestBuffer>().is_some())
        .collect()
}

struct Coherent {
    embedder: Rc<TestEmbedder>,
}

impl Coherence<Native, TestEmbedder> for Coherent {
    fn commit(&mut self, cx: &mut ServiceCx<'_, Native, TestEmbedder>, info: &CommitInfo) -> Result<(), Failure> {
        for (_, p) in buffers(cx.objects(), &info.resources) {
            let b = p.downcast_ref::<TestBuffer>().unwrap();
            if b.contents == 0 {
                continue;
            }
            let mut guest = vec![0; b.length as usize];
            self.embedder.read_guest(cx.task(), b.contents, &mut guest).map_err(|e| failure::invalid(e.to_string()))?;
            if guest != *b.shadow.borrow() {
                let mut padded = guest.clone();
                padded.resize(b.size as usize, 0);
                cx.gpu()
                    .call("GPUQueue.writeBuffer", "", |_| {
                        cx.gpu().queue().write_buffer(
                            &b.gpu,
                            Size64::new(0).unwrap(),
                            &padded,
                            Size64::new(0).unwrap(),
                            None,
                        )
                    })
                    .unwrap();
                *b.shadow.borrow_mut() = guest;
            }
        }
        Ok(())
    }

    fn write_back(
        &mut self,
        cx: &mut EncodeCx<'_, Native, TestEmbedder>,
        info: &CommitInfo,
    ) -> Result<Option<GpuFuture<Result<(), Failure>>>, Failure> {
        let mut reads = Vec::new();
        for (_, p) in buffers(cx.objects(), &info.resources) {
            let b = p.downcast_ref::<TestBuffer>().unwrap();
            if b.contents == 0 {
                continue;
            }
            let desc =
                BufferDescriptor::new(Size64::new(b.size).unwrap(), BufferUsage::MAP_READ | BufferUsage::COPY_DST);
            let staging = cx.gpu().call("GPUDevice.createBuffer", "staging", |d| d.create_buffer(&desc)).unwrap();
            let e = cx.encoder().clone();
            cx.gpu()
                .call("GPUCommandEncoder.copyBufferToBuffer", "", |_| e.copy_buffer_to_buffer(&b.gpu, &staging, None));
            reads.push((p.clone(), staging));
        }
        if reads.is_empty() {
            return Ok(None);
        }
        let embedder = self.embedder.clone();
        let task = cx.task();
        Ok(Some(Box::pin(async move {
            for (p, staging) in reads {
                let b = p.downcast_ref::<TestBuffer>().unwrap();
                staging
                    .map_async(MapMode::READ, Size64::new(0).unwrap(), None)
                    .await
                    .map_err(|e| failure::invalid(e.to_string()))?;
                let range = staging.get_mapped_range(Size64::new(0).unwrap(), None).unwrap();
                let mut new = vec![0; b.length as usize];
                range.read(0, &mut new).unwrap();
                staging.unmap();
                let mut shadow = b.shadow.borrow_mut();
                embedder
                    .write_guest_changed(task, b.contents, &new, &shadow)
                    .map_err(|e| failure::invalid(e.to_string()))?;
                *shadow = new;
            }
            Ok(())
        })))
    }

    fn task_exited(&mut self, _: TaskId) {}
}

struct Blit;

impl EncoderService<Native, TestEmbedder> for Blit {
    fn encode(&mut self, cx: &mut EncodeCx<'_, Native, TestEmbedder>, segment: &Segment) -> Result<(), RecordFailure> {
        let mut groups = DebugGroups::default();
        let records = cx.records();
        let gpu_buffer = |cx: &EncodeCx<'_, Native, TestEmbedder>, h: Option<Handle>| {
            let p = cx.objects().get(h.unwrap()).unwrap().payload.clone().unwrap();
            p.downcast_ref::<TestBuffer>().unwrap().gpu.clone()
        };
        for &i in &segment.records {
            let (offset, record) = &records[i];
            let fail =
                || RecordFailure { offset: *offset as u64, failure: failure::unimplemented_record(record.opcode()) };
            let e = cx.encoder().clone();
            match record {
                Command::BeginBlit(_) | Command::EncoderSetLabel(_) => {}
                Command::EncoderPushDebugGroup(r) => {
                    let g = r.string.clone().unwrap_or_default();
                    cx.gpu().call("GPUCommandEncoder.pushDebugGroup", "", |_| e.push_debug_group(&g));
                    groups.push();
                }
                Command::EncoderPopDebugGroup(_) => {
                    if groups.pop() {
                        cx.gpu().call("GPUCommandEncoder.popDebugGroup", "", |_| e.pop_debug_group());
                    }
                }
                Command::EncoderInsertDebugSignpost(r) => {
                    let g = r.string.clone().unwrap_or_default();
                    cx.gpu().call("GPUCommandEncoder.insertDebugMarker", "", |_| e.insert_debug_marker(&g));
                }
                Command::BlitFillBuffer(r) => {
                    if r.range.location % 4 != 0 || r.range.length % 4 != 0 || r.range.length == 0 {
                        return Err(fail());
                    }
                    let mut desc = BufferDescriptor::new(Size64::new(r.range.length).unwrap(), BufferUsage::COPY_SRC);
                    desc.mapped_at_creation = true;
                    let pattern =
                        cx.gpu().call("GPUDevice.createBuffer", "pattern", |d| d.create_buffer(&desc)).unwrap();
                    let m = pattern.get_mapped_range(Size64::new(0).unwrap(), None).unwrap();
                    m.write(0, &vec![r.value; r.range.length as usize]).unwrap();
                    pattern.unmap();
                    let dst = gpu_buffer(cx, r.buffer);
                    cx.gpu().call("GPUCommandEncoder.copyBufferToBuffer", "", |_| {
                        e.copy_buffer_to_buffer_with_offsets(
                            &pattern,
                            Size64::new(0).unwrap(),
                            &dst,
                            Size64::new(r.range.location).unwrap(),
                            Some(Size64::new(r.range.length).unwrap()),
                        )
                    });
                }
                Command::BlitCopyBufferToBuffer(r) => {
                    let (src, dst) = (gpu_buffer(cx, r.source_buffer), gpu_buffer(cx, r.destination_buffer));
                    cx.gpu().call("GPUCommandEncoder.copyBufferToBuffer", "", |_| {
                        e.copy_buffer_to_buffer_with_offsets(
                            &src,
                            Size64::new(r.source_offset).unwrap(),
                            &dst,
                            Size64::new(r.destination_offset).unwrap(),
                            Some(Size64::new(r.size).unwrap()),
                        )
                    });
                }
                Command::EncoderEndEncoding(_) => {
                    for _ in 0..groups.close() {
                        cx.gpu().call("GPUCommandEncoder.popDebugGroup", "", |_| e.pop_debug_group());
                    }
                }
                _ => return Err(fail()),
            }
        }
        Ok(())
    }
}

/// A compute stand-in whose every encoder hits out-of-memory: the fault injected for the out-of-memory mapping.
struct ExhaustingCompute;

impl EncoderService<Native, TestEmbedder> for ExhaustingCompute {
    fn encode(&mut self, cx: &mut EncodeCx<'_, Native, TestEmbedder>, _: &Segment) -> Result<(), RecordFailure> {
        exhaust_memory(cx.gpu());
        Ok(())
    }
}

// The harness ---------------------------------------------------------------------------------------------------------

struct Harness {
    gpu: NativeGpu,
    device: NativeDevice,
    exec: Executor<Native, TestEmbedder>,
    em: Rc<TestEmbedder>,
    next: Cell<u64>,
}

const T1: TaskId = TaskId(101);
const T2: TaskId = TaskId(202);

impl Harness {
    fn new() -> Harness {
        Harness::with(true)
    }

    fn with(blit: bool) -> Harness {
        let gpu = NativeGpu::new();
        let adapter = gpu
            .block_on(gpu.request_adapter(&RequestAdapterOptions::new()), WAIT)
            .unwrap()
            .unwrap()
            .expect("an adapter");
        let descriptor = match negotiate::plan(&adapter.features(), &adapter.limits()) {
            Ok(plan) => plan.descriptor,
            // wgpu-native's adapter does not report "core-features-and-limits" (tests/gpu_smoke_known_wgpu-native.txt);
            // as the smoke test does, the executor then runs on a device with the default features and limits.
            Err(negotiate::NegotiationError::NotCore) if cfg!(webgpu_native = "wgpu-native") => DeviceDescriptor::new(),
            Err(e) => panic!("negotiation: {e}"),
        };
        let device = gpu.block_on(adapter.request_device(&descriptor), WAIT).unwrap().expect("a device");
        let em = Rc::new(TestEmbedder::new());
        let services = Services {
            control: vec![Box::new(Memory { embedder: em.clone() }) as Box<dyn ControlService<_, _>>],
            render: None,
            compute: Some(Box::new(ExhaustingCompute)),
            blit: if blit { Some(Box::new(Blit)) } else { None },
            coherence: Some(Box::new(Coherent { embedder: em.clone() })),
        };
        let config = Config { scopes: Policy { granularity: Granularity::PerCall, strict: true } };
        let exec = Executor::new(device.clone(), em.clone(), services, config);
        Harness { gpu, device, exec, em, next: Cell::new(0x1000) }
    }

    fn handle(&self) -> Handle {
        self.next.set(self.next.get() + 1);
        Handle::new(self.next.get()).unwrap()
    }

    fn raw(&self, task: TaskId, bytes: &[u8]) -> Reply {
        let out = self.gpu.block_on(self.exec.control(task, bytes), WAIT).unwrap().expect("a reply record");
        let (reply, len) = Reply::decode(&out, 0).expect("the reply decodes");
        assert_eq!(len, out.len());
        reply
    }

    fn request(&self, task: TaskId, request: Request) -> Reply {
        let mut bytes = Vec::new();
        request.encode(&mut bytes).unwrap();
        let reply = self.raw(task, &bytes);
        assert_eq!(reply.request_opcode(), request.opcode());
        reply
    }

    /// The failure of a reply (status 0 for success).
    fn status(&self, task: TaskId, request: Request) -> Failure {
        schema::reply_failure(&self.request(task, request)).cloned().unwrap_or_else(failure::ok)
    }

    fn ok(&self, task: TaskId, request: Request) {
        let f = self.status(task, request);
        assert_eq!(f.status, status::OK, "{f:?}");
    }

    fn open(&self, task: TaskId) {
        self.ok(task, Request::Open(Open { schema_hash: Hash { bytes: SCHEMA_HASH }, pid: task.0 as i32 }));
    }

    fn queue(&self, task: TaskId) -> Handle {
        let q = self.handle();
        self.ok(task, Request::CreateCommandQueue(CreateCommandQueue { queue: q }));
        q
    }

    fn event(&self, task: TaskId) -> Handle {
        let e = self.handle();
        self.ok(task, Request::CreateEvent(CreateEvent { event: e }));
        e
    }

    fn buffer(&self, task: TaskId, len: usize) -> (Handle, u64) {
        let addr = self.em.alloc(task, vec![0; len]);
        let b = self.handle();
        self.ok(
            task,
            Request::CreateBuffer(CreateBuffer {
                buffer: b,
                length: len as u64,
                options: 0,
                contents: GuestAddr(addr),
            }),
        );
        (b, addr)
    }

    /// Commit a command buffer with this stream; returns its handle and the reply's failure.
    fn commit_with(
        &self,
        task: TaskId,
        queue: Handle,
        records: &[Command],
        resources: &[Handle],
        retained: bool,
    ) -> (Handle, Failure) {
        let mut bytes = Vec::new();
        for r in records {
            r.encode(&mut bytes).unwrap();
        }
        let len = bytes.len() as u64;
        let addr = self.em.alloc(task, bytes);
        let cb = self.handle();
        let f = self.status(
            task,
            Request::Commit(Commit {
                command_buffer: cb,
                queue,
                label: Some(format!("cb {:#x}", cb.get())),
                retained_references: retained,
                error_options: 0,
                protection_options: 0,
                stream: GuestAddr(addr),
                stream_length: len,
                resources: resources.to_vec(),
            }),
        );
        (cb, f)
    }

    fn commit(&self, task: TaskId, queue: Handle, records: &[Command], resources: &[Handle]) -> Handle {
        let (cb, f) = self.commit_with(task, queue, records, resources, true);
        assert_eq!(f.status, status::OK, "{f:?}");
        cb
    }

    fn idle(&self) {
        self.gpu.block_on(self.exec.idle(), WAIT).expect("the executor goes idle");
    }

    fn guest(&self, task: TaskId, addr: u64, len: usize) -> Vec<u8> {
        self.em.guest(task, addr, len)
    }
}

fn blit(encoder: Handle, body: Vec<Command>) -> Vec<Command> {
    let mut v = vec![Command::BeginBlit(BeginBlit { encoder })];
    v.extend(body);
    v.push(Command::EncoderEndEncoding(EncoderEndEncoding {}));
    v
}

fn fill(buffer: Handle, length: u64, value: u8) -> Command {
    Command::BlitFillBuffer(BlitFillBuffer { buffer: Some(buffer), range: Range { location: 0, length }, value })
}

fn copy(source: Handle, destination: Handle, size: u64) -> Command {
    Command::BlitCopyBufferToBuffer(BlitCopyBufferToBuffer {
        source_buffer: Some(source),
        source_offset: 0,
        destination_buffer: Some(destination),
        destination_offset: 0,
        size,
    })
}

fn signal(event: Handle, value: u64) -> Command {
    Command::EncodeSignalEvent(EncodeSignalEvent { event: Some(event), value })
}

fn wait(event: Handle, value: u64) -> Command {
    Command::EncodeWaitForEvent(EncodeWaitForEvent { event: Some(event), value })
}

/// The command-buffer events of one queue, as (command buffer, "scheduled" or "completed", the completion).
fn queue_events(
    posted: &[(TaskId, Option<Handle>, Event)],
    queue: Handle,
) -> Vec<(Handle, &'static str, Option<CommandBufferCompleted>)> {
    posted
        .iter()
        .filter(|(_, q, _)| *q == Some(queue))
        .map(|(_, _, e)| match e {
            Event::CommandBufferScheduled(s) => (s.command_buffer, "scheduled", None),
            Event::CommandBufferCompleted(c) => (c.command_buffer, "completed", Some(c.clone())),
            other => panic!("unexpected event on a queue: {other:?}"),
        })
        .collect()
}

fn completion(posted: &[(TaskId, Option<Handle>, Event)], cb: Handle) -> CommandBufferCompleted {
    let found: Vec<_> = posted
        .iter()
        .filter_map(|(_, _, e)| match e {
            Event::CommandBufferCompleted(c) if c.command_buffer == cb => Some(c.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(found.len(), 1, "one completion for {cb:?} in {posted:?}");
    found[0].clone()
}

fn scheduled_count(posted: &[(TaskId, Option<Handle>, Event)], cb: Handle) -> usize {
    posted.iter().filter(|(_, _, e)| matches!(e, Event::CommandBufferScheduled(s) if s.command_buffer == cb)).count()
}

fn assert_completed(c: &CommandBufferCompleted) {
    assert_eq!(c.status, command_buffer_status::COMPLETED, "{c:?}");
    assert_eq!(c.failure, failure::ok());
    assert_eq!(c.failed_offset, NO_RECORD);
}

// The reference outputs -----------------------------------------------------------------------------------------------

mod golden {
    //! The records of a reference case recorded on real Metal, read from `tests/reference/goldens` (format:
    //! `tests/reference/FORMAT.md`). Only the `release` variant applies: the debug layer runs in the guest, in
    //! Metal.framework, before anything reaches the executor.

    use std::collections::BTreeMap;

    #[derive(Clone, Debug, PartialEq)]
    pub enum Json {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Json>),
        Obj(BTreeMap<String, Json>),
    }

    impl Json {
        pub fn get(&self, key: &str) -> &Json {
            match self {
                Json::Obj(m) => m.get(key).unwrap_or(&Json::Null),
                _ => &Json::Null,
            }
        }
        pub fn str(&self) -> &str {
            match self {
                Json::Str(s) => s,
                other => panic!("not a string: {other:?}"),
            }
        }
        pub fn bool(&self) -> bool {
            match self {
                Json::Bool(b) => *b,
                other => panic!("not a boolean: {other:?}"),
            }
        }
        pub fn u64(&self) -> u64 {
            match self {
                Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => *n as u64,
                other => panic!("not an unsigned integer: {other:?}"),
            }
        }
        pub fn strings(&self) -> Vec<String> {
            match self {
                Json::Arr(v) => v.iter().map(|s| s.str().to_string()).collect(),
                other => panic!("not an array: {other:?}"),
            }
        }
    }

    struct Parser<'a> {
        s: &'a [u8],
        at: usize,
    }

    impl Parser<'_> {
        fn ws(&mut self) {
            while self.at < self.s.len() && self.s[self.at].is_ascii_whitespace() {
                self.at += 1;
            }
        }
        fn eat(&mut self, c: u8) {
            self.ws();
            assert_eq!(self.s[self.at], c, "JSON: expected {:?} at {}", c as char, self.at);
            self.at += 1;
        }
        fn value(&mut self) -> Json {
            self.ws();
            match self.s[self.at] {
                b'{' => {
                    self.at += 1;
                    let mut m = BTreeMap::new();
                    self.ws();
                    if self.s[self.at] == b'}' {
                        self.at += 1;
                        return Json::Obj(m);
                    }
                    loop {
                        self.ws();
                        let Json::Str(k) = self.value() else { panic!("JSON: key") };
                        self.eat(b':');
                        let v = self.value();
                        m.insert(k, v);
                        self.ws();
                        let c = self.s[self.at];
                        self.at += 1;
                        if c == b'}' {
                            return Json::Obj(m);
                        }
                        assert_eq!(c, b',');
                    }
                }
                b'[' => {
                    self.at += 1;
                    let mut v = Vec::new();
                    self.ws();
                    if self.s[self.at] == b']' {
                        self.at += 1;
                        return Json::Arr(v);
                    }
                    loop {
                        v.push(self.value());
                        self.ws();
                        let c = self.s[self.at];
                        self.at += 1;
                        if c == b']' {
                            return Json::Arr(v);
                        }
                        assert_eq!(c, b',');
                    }
                }
                b'"' => {
                    self.at += 1;
                    let mut out = String::new();
                    loop {
                        let c = self.s[self.at];
                        self.at += 1;
                        match c {
                            b'"' => return Json::Str(out),
                            b'\\' => {
                                let e = self.s[self.at];
                                self.at += 1;
                                match e {
                                    b'n' => out.push('\n'),
                                    b't' => out.push('\t'),
                                    b'r' => out.push('\r'),
                                    b'b' => out.push('\u{8}'),
                                    b'f' => out.push('\u{c}'),
                                    b'u' => {
                                        let hex = std::str::from_utf8(&self.s[self.at..self.at + 4]).unwrap();
                                        self.at += 4;
                                        out.push(
                                            char::from_u32(u32::from_str_radix(hex, 16).unwrap()).unwrap_or('\u{fffd}'),
                                        );
                                    }
                                    other => out.push(other as char),
                                }
                            }
                            _ => {
                                // Copy one UTF-8 sequence.
                                let start = self.at - 1;
                                let mut end = self.at;
                                while end < self.s.len() && (self.s[end] & 0xc0) == 0x80 {
                                    end += 1;
                                }
                                out.push_str(std::str::from_utf8(&self.s[start..end]).unwrap());
                                self.at = end;
                            }
                        }
                    }
                }
                b't' => {
                    self.at += 4;
                    Json::Bool(true)
                }
                b'f' => {
                    self.at += 5;
                    Json::Bool(false)
                }
                b'n' => {
                    self.at += 4;
                    Json::Null
                }
                _ => {
                    let start = self.at;
                    while self.at < self.s.len() && b"-+.eE0123456789".contains(&self.s[self.at]) {
                        self.at += 1;
                    }
                    Json::Num(std::str::from_utf8(&self.s[start..self.at]).unwrap().parse().unwrap())
                }
            }
        }
    }

    pub const DEVICE: &str = "apple-m4-macos-27.0.1-26a434";

    /// The release-variant case file of `case`.
    pub fn case(case: &str) -> Json {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/reference/goldens")
            .join(DEVICE)
            .join("release")
            .join(format!("{case}.json"));
        let text = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Parser { s: &text, at: 0 }.value()
    }

    /// The output record named `name` of a case.
    pub fn record(case: &Json, name: &str) -> Json {
        let Json::Arr(records) = case.get("records") else { panic!("records") };
        records
            .iter()
            .find(|r| r.get("name") == &Json::Str(name.to_string()) && r.get("kind") != &Json::Str("step".into()))
            .unwrap_or_else(|| panic!("no record {name}"))
            .clone()
    }

    /// The `value` of a value record.
    pub fn value(case: &Json, name: &str) -> Json {
        record(case, name).get("value").clone()
    }

    /// The bytes of a data record.
    pub fn data(case: &Json, name: &str) -> Vec<u8> {
        let r = record(case, name);
        let hex = r.get("hex").str();
        (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
    }

    /// Whether an error record is nil. FORMAT.md calls `nil` a boolean; the recorded files hold it as the number 0 or 1
    /// (`@(error == nil)` in Harness.m), so both are read.
    pub fn error_is_nil(case: &Json, name: &str) -> bool {
        match record(case, name).get("nil") {
            Json::Num(n) if *n == 0.0 || *n == 1.0 => *n == 1.0,
            other => other.bool(),
        }
    }

    /// The outcome of a case run that ended on an assertion in the driver or the framework.
    pub fn terminated(case: &Json) -> bool {
        case.get("status").str() == "terminated"
    }
}

// Command buffers -----------------------------------------------------------------------------------------------------

/// Reference case `command-buffer.status-sequence`, `command-buffer.handler-order`,
/// `command-buffer.success-error-and-logs` and `command-buffer.timing`: one CommandBufferScheduled, then one
/// CommandBufferCompleted with status completed and no error; the work is in guest memory when it completes; the times
/// are positive and ordered.
#[test]
fn status_handlers_and_timing() {
    let x = Harness::new();
    x.open(T1);
    let q = x.queue(T1);
    let (data, addr) = x.buffer(T1, 256);
    let cb = x.commit(T1, q, &blit(x.handle(), vec![fill(data, 256, 7)]), &[data]);
    x.idle();
    let posted = x.em.take_posted();
    let events = queue_events(&posted, q);
    assert_eq!(events.iter().map(|e| (e.0, e.1)).collect::<Vec<_>>(), vec![(cb, "scheduled"), (cb, "completed")]);
    let c = events[1].2.clone().unwrap();
    assert_completed(&c);

    // The guest framework runs the scheduled handlers on CommandBufferScheduled and the completed handlers on
    // CommandBufferCompleted, each in registration order.
    let order = golden::case("command-buffer.handler-order");
    let mut sequence = Vec::new();
    for (_, kind, _) in &events {
        for i in 1..=3 {
            sequence.push(format!("{kind}-{i}"));
        }
    }
    assert_eq!(sequence, golden::value(&order, "sequence").strings());

    let statuses = golden::case("command-buffer.status-sequence");
    assert_eq!(c.status, golden::value(&statuses, "after-wait-until-completed").u64());
    assert_eq!(c.status, golden::value(&statuses, "in-completed-handler").u64());
    assert_eq!(c.failure.status == status::OK, golden::error_is_nil(&statuses, "error"));
    assert_eq!(x.guest(T1, addr, 256) == vec![7; 256], golden::value(&statuses, "result-written").bool());

    let logs = golden::case("command-buffer.success-error-and-logs");
    assert_eq!(c.status, golden::value(&logs, "status").u64());
    assert_eq!(c.failure.status == status::OK, golden::error_is_nil(&logs, "error"));

    let timing = golden::case("command-buffer.timing");
    assert_eq!(c.gpu_start_time > 0, golden::value(&timing, "completed.gpu-start-positive").bool());
    assert_eq!(c.gpu_end_time >= c.gpu_start_time, golden::value(&timing, "completed.gpu-end-not-before-start").bool());
    assert_eq!(c.kernel_start_time > 0, golden::value(&timing, "completed.kernel-start-positive").bool());
    assert_eq!(
        c.kernel_end_time >= c.kernel_start_time,
        golden::value(&timing, "completed.kernel-end-not-before-start").bool()
    );
    // Recorded without a comparison rule on real Metal; holds here by construction.
    assert!(c.kernel_start_time <= c.gpu_start_time);
    let Event::CommandBufferScheduled(s) =
        &posted.iter().find(|p| matches!(p.2, Event::CommandBufferScheduled(_))).unwrap().2
    else {
        unreachable!()
    };
    assert_eq!(s.kernel_start_time, c.kernel_start_time);
    assert_eq!(s.kernel_end_time, c.kernel_end_time);
    assert!(x.exec.take_reports().is_empty());
}

/// Reference case `command-buffer.enqueue-order`: the framework hands command buffers to the driver's commit in enqueue
/// order (b, then a); they run and complete in that order.
#[test]
fn completion_follows_commit_order() {
    let x = Harness::new();
    x.open(T1);
    let q = x.queue(T1);
    let (target, addr) = x.buffer(T1, 256);
    let b = x.commit(T1, q, &blit(x.handle(), vec![fill(target, 256, 0xb2)]), &[target]);
    let a = x.commit(T1, q, &blit(x.handle(), vec![fill(target, 256, 0xa1)]), &[target]);
    x.idle();
    let posted = x.em.take_posted();
    let names: HashMap<Handle, &str> = [(a, "a"), (b, "b")].into_iter().collect();
    let order: Vec<String> =
        queue_events(&posted, q).iter().filter(|e| e.1 == "completed").map(|e| names[&e.0].to_string()).collect();
    let golden = golden::case("command-buffer.enqueue-order");
    assert_eq!(order, golden::value(&golden, "completed-handler-order").strings());
    assert_eq!(completion(&posted, a).status, golden::value(&golden, "a.status").u64());
    assert_eq!(completion(&posted, b).status, golden::value(&golden, "b.status").u64());
    assert_eq!(x.guest(T1, addr, 256), golden::data(&golden, "target"));
    // Scheduled events come in commit order too, each before its completion.
    let all: Vec<_> = queue_events(&posted, q).iter().map(|e| (e.0, e.1)).collect();
    assert_eq!(all, vec![(b, "scheduled"), (a, "scheduled"), (b, "completed"), (a, "completed")]);
}

/// Reference case `command-buffer.retained-references`: an unretained command buffer whose objects the caller keeps
/// alive runs normally.
#[test]
fn unretained_references_run_normally() {
    let x = Harness::new();
    x.open(T1);
    let q = x.queue(T1);
    let (data, addr) = x.buffer(T1, 256);
    let (cb, f) = x.commit_with(T1, q, &blit(x.handle(), vec![fill(data, 256, 9)]), &[data], false);
    assert_eq!(f.status, status::OK);
    x.idle();
    let golden = golden::case("command-buffer.retained-references");
    let c = completion(&x.em.take_posted(), cb);
    assert_eq!(c.status, golden::value(&golden, "unretained.status").u64());
    assert_eq!(x.guest(T1, addr, 256) == vec![9; 256], golden::value(&golden, "unretained.result-written").bool());
}

/// Reference case `debug.labels-and-groups`, and Metal's acceptance of a pop with no group open and of groups left open
/// (measured on Apple M4, macOS 27.0.1): every command buffer completes and no WebGPU error is raised.
#[test]
fn labels_and_debug_groups() {
    let x = Harness::new();
    x.open(T1);
    let q = x.queue(T1);
    x.ok(T1, Request::SetLabel(SetLabel { object: q, label: Some("queue-label".into()) }));
    let (data, addr) = x.buffer(T1, 1024);
    x.ok(T1, Request::SetLabel(SetLabel { object: data, label: Some("buffer-label".into()) }));
    let e = x.event(T1);
    x.ok(T1, Request::SetLabel(SetLabel { object: e, label: Some("event-label".into()) }));
    let push = |s: &str| Command::CommandBufferPushDebugGroup(CommandBufferPushDebugGroup { string: Some(s.into()) });
    let pop = || Command::CommandBufferPopDebugGroup(CommandBufferPopDebugGroup {});
    let streams = [
        // The reference case's shape.
        [
            vec![push("outer")],
            blit(
                x.handle(),
                vec![
                    Command::EncoderSetLabel(EncoderSetLabel { label: Some("encoder-label".into()) }),
                    Command::EncoderPushDebugGroup(EncoderPushDebugGroup { string: Some("inner".into()) }),
                    Command::EncoderInsertDebugSignpost(EncoderInsertDebugSignpost { string: Some("signpost".into()) }),
                    fill(data, 1024, 4),
                    Command::EncoderPopDebugGroup(EncoderPopDebugGroup {}),
                ],
            ),
            vec![pop()],
        ]
        .concat(),
        // A pop with nothing pushed, on the command buffer and in an encoder.
        [
            vec![pop()],
            blit(x.handle(), vec![Command::EncoderPopDebugGroup(EncoderPopDebugGroup {}), fill(data, 1024, 4)]),
        ]
        .concat(),
        // Groups left open; one spans a split of the submission at a signal.
        [
            vec![push("open")],
            blit(
                x.handle(),
                vec![Command::EncoderPushDebugGroup(EncoderPushDebugGroup { string: None }), fill(data, 1024, 4)],
            ),
            vec![signal(e, 1), push("second")],
            blit(x.handle(), vec![fill(data, 1024, 4)]),
        ]
        .concat(),
    ];
    let mut cbs = Vec::new();
    for s in &streams {
        let resources: Vec<Handle> =
            if s.iter().any(|c| matches!(c, Command::EncodeSignalEvent(_))) { vec![data, e] } else { vec![data] };
        cbs.push(x.commit(T1, q, s, &resources));
    }
    x.idle();
    let posted = x.em.take_posted();
    let golden = golden::case("debug.labels-and-groups");
    for cb in cbs {
        let c = completion(&posted, cb);
        assert_completed(&c);
        assert_eq!(c.status, golden::value(&golden, "status").u64());
    }
    assert_eq!(x.guest(T1, addr, 1024) == vec![4; 1024], golden::value(&golden, "result-written").bool());
    assert!(x.exec.take_reports().is_empty());
}

// Events --------------------------------------------------------------------------------------------------------------

/// Reference case `event.cross-queue-signal-wait`: a command buffer waiting on an MTLEvent does not complete before a
/// buffer on another queue signals it, then runs after the signalling work.
#[test]
fn cross_queue_signal_and_wait() {
    let x = Harness::new();
    x.open(T1);
    let (waiting_q, signalling_q) = (x.queue(T1), x.queue(T1));
    let event = x.event(T1);
    let (source, _) = x.buffer(T1, 1024);
    let (destination, addr) = x.buffer(T1, 1024);
    let waiting = x.commit(
        T1,
        waiting_q,
        &[vec![wait(event, 1)], blit(x.handle(), vec![copy(source, destination, 1024)])].concat(),
        &[event, source, destination],
    );
    x.idle();
    let golden = golden::case("event.cross-queue-signal-wait");
    let posted = x.em.take_posted();
    assert_eq!(
        posted.iter().any(|p| matches!(&p.2, Event::CommandBufferCompleted(c) if c.command_buffer == waiting)),
        golden::value(&golden, "waiting.completed-before-signal-commit").bool()
    );
    // Committed, not scheduled, while it waits (measured on Apple M4, macOS 27.0.1; the reference record allows
    // either).
    assert_eq!(scheduled_count(&posted, waiting), 0);
    assert!(golden::value(&golden, "waiting.committed-or-scheduled-while-waiting").bool());

    let signalling = x.commit(
        T1,
        signalling_q,
        &[blit(x.handle(), vec![fill(source, 1024, 0x5a)]), vec![signal(event, 1)]].concat(),
        &[source, event],
    );
    x.idle();
    let posted = x.em.take_posted();
    assert_eq!(completion(&posted, waiting).status, golden::value(&golden, "waiting.status").u64());
    assert_eq!(completion(&posted, signalling).status, golden::value(&golden, "signalling.status").u64());
    assert_eq!(x.guest(T1, addr, 1024), golden::data(&golden, "destination"));
}

/// A waiting command buffer holds back the later command buffers of its queue: neither is scheduled until the
/// signal, then both run in order (measured on Apple M4, macOS 27.0.1).
#[test]
fn a_wait_holds_its_queue() {
    let x = Harness::new();
    x.open(T1);
    let (q1, q2) = (x.queue(T1), x.queue(T1));
    let event = x.event(T1);
    let (buf, addr) = x.buffer(T1, 256);
    let w =
        x.commit(T1, q1, &[vec![wait(event, 1)], blit(x.handle(), vec![fill(buf, 256, 4)])].concat(), &[event, buf]);
    let later = x.commit(T1, q1, &blit(x.handle(), vec![fill(buf, 256, 5)]), &[buf]);
    x.idle();
    assert!(x.em.take_posted().is_empty());
    let s = x.commit(T1, q2, &[signal(event, 1)], &[event]);
    x.idle();
    let posted = x.em.take_posted();
    let all: Vec<_> = queue_events(&posted, q1).iter().map(|e| (e.0, e.1)).collect();
    assert_eq!(all, vec![(w, "scheduled"), (later, "scheduled"), (w, "completed"), (later, "completed")]);
    assert_completed(&completion(&posted, s));
    assert_eq!(x.guest(T1, addr, 256), vec![5; 256]);
}

/// MTLEvent values never decrease, waits are satisfied at or above their value, a wait for 0 on a new event passes,
/// and a command buffer can wait for its own earlier signal (all measured on Apple M4, macOS 27.0.1).
#[test]
fn event_values() {
    let x = Harness::new();
    x.open(T1);
    let (q1, q2) = (x.queue(T1), x.queue(T1));
    let e = x.event(T1);
    let (buf, _) = x.buffer(T1, 256);
    x.commit(T1, q1, &[signal(e, 10)], &[e]);
    x.commit(T1, q1, &[signal(e, 5)], &[e]);
    x.idle();
    let w = x.commit(T1, q2, &[vec![wait(e, 7)], blit(x.handle(), vec![fill(buf, 256, 1)])].concat(), &[e, buf]);
    let fresh = x.event(T1);
    let zero = x.commit(T1, q2, &[wait(fresh, 0)], &[fresh]);
    let own = x.event(T1);
    let same = x.commit(T1, q1, &[signal(own, 1), wait(own, 1)], &[own]);
    x.idle();
    let posted = x.em.take_posted();
    for cb in [w, zero, same] {
        assert_completed(&completion(&posted, cb));
    }
}

/// Reference case `event.shared-cpu-signal`: a CPU signal releases a GPU wait only at the awaited value; the GPU signal
/// sets the kernel object's value before the command buffer completes.
#[test]
fn shared_event_cpu_and_gpu_signals() {
    let x = Harness::new();
    x.open(T1);
    let q = x.queue(T1);
    let golden = golden::case("event.shared-cpu-signal");
    x.em.shared_event(T1, 77, 9001, golden::value(&golden, "initial-signaled-value").u64());
    let event = x.handle();
    x.ok(T1, Request::RegisterSharedEvent(RegisterSharedEvent { event, port: 77 }));
    let (target, addr) = x.buffer(T1, 512);
    let cb = x.commit(
        T1,
        q,
        &[vec![wait(event, 2)], blit(x.handle(), vec![fill(target, 512, 0x3c)]), vec![signal(event, 5)]].concat(),
        &[event, target],
    );
    x.idle();
    x.em.cpu_signal(9001, 1);
    x.exec.shared_event_signaled(SharedEventId(9001));
    x.idle();
    assert_eq!(x.em.value(9001), golden::value(&golden, "after-cpu-signal-1.signaled-value").u64());
    let early = x.em.take_posted().iter().any(|p| matches!(p.2, Event::CommandBufferCompleted(_)));
    assert_eq!(early, golden::value(&golden, "completed-with-value-1").bool());
    x.em.cpu_signal(9001, 2);
    x.exec.shared_event_signaled(SharedEventId(9001));
    x.idle();
    let posted = x.em.take_posted();
    assert_eq!(completion(&posted, cb).status, golden::value(&golden, "status").u64());
    assert_eq!(x.em.value(9001), golden::value(&golden, "after-gpu-signal.signaled-value").u64());
    assert_eq!(x.guest(T1, addr, 512), golden::data(&golden, "target"));
    assert_eq!(x.em.value(9001) >= 5, golden::value(&golden, "wait-until-5").bool());
    assert_eq!(x.em.value(9001) >= 100, golden::value(&golden, "wait-until-100-timeout-20ms").bool());
}

/// Reference case `event.shared-listener` and `event.shared-handle`: the kernel object receives exactly the GPU
/// signal's value (its listeners fire from it, in IOSurface.framework), handles registered from one kernel object share
/// its value, and the executor releases the object once no handle or command buffer refers to it.
#[test]
fn shared_event_handles_and_signal_values() {
    let x = Harness::new();
    x.open(T1);
    x.open(T2);
    let q = x.queue(T1);
    x.em.shared_event(T1, 5, 42, 3);
    x.em.ports.borrow_mut().insert((T2, 6), 42);
    let (first, second, other) = (x.handle(), x.handle(), x.handle());
    x.ok(T1, Request::RegisterSharedEvent(RegisterSharedEvent { event: first, port: 5 }));
    x.ok(T1, Request::RegisterSharedEvent(RegisterSharedEvent { event: second, port: 5 }));
    x.ok(T2, Request::RegisterSharedEvent(RegisterSharedEvent { event: other, port: 6 }));
    // A port that names no shared event is refused.
    let bad = x.status(T1, Request::RegisterSharedEvent(RegisterSharedEvent { event: x.handle(), port: 99 }));
    assert_eq!(bad.status, status::INVALID);

    let listener = golden::case("event.shared-listener");
    let cb = x.commit(T1, q, &[signal(first, 6)], &[first]);
    x.idle();
    assert_completed(&completion(&x.em.take_posted(), cb));
    assert_eq!(*x.em.signals.borrow(), vec![(42, 6)]);
    assert!(!golden::value(&listener, "fired-beyond-signaled-value").bool());

    let handle = golden::case("event.shared-handle");
    let cb = x.commit(T1, q, &[signal(second, 12)], &[second]);
    x.idle();
    assert_completed(&completion(&x.em.take_posted(), cb));
    assert_eq!(x.em.value(42), golden::value(&handle, "original.sees-gpu-12").u64());

    for (t, e) in [(T1, first), (T1, second)] {
        x.ok(t, Request::Release(Release { object: e }));
    }
    assert!(x.em.released.borrow().is_empty(), "task 2 still names the event");
    x.ok(T2, Request::Release(Release { object: other }));
    assert_eq!(*x.em.released.borrow(), vec![42]);
}

// Fences --------------------------------------------------------------------------------------------------------------

/// Reference case `fence.across-encoders`, `fence.wait-before-later-update` and `fence.update-then-wait-same-encoder`
/// (release variant): command buffers with fences complete, and passes run in encoding order.
#[test]
fn fences() {
    let x = Harness::new();
    x.open(T1);
    let q = x.queue(T1);
    let fence = x.handle();
    x.ok(T1, Request::CreateFence(CreateFence { fence }));
    let update = || Command::BlitUpdateFence(BlitUpdateFence { fence: Some(fence) });
    let wait_fence = || Command::BlitWaitForFence(BlitWaitForFence { fence: Some(fence) });
    let (first, _) = x.buffer(T1, 1024);
    let (second, addr) = x.buffer(T1, 1024);
    let across = x.commit(
        T1,
        q,
        &[
            blit(x.handle(), vec![fill(first, 1024, 0x11), update()]),
            blit(x.handle(), vec![wait_fence(), copy(first, second, 1024), update()]),
            blit(x.handle(), vec![wait_fence(), fill(first, 1024, 0x22)]),
        ]
        .concat(),
        &[first, fence, second],
    );
    let (a, a_addr) = x.buffer(T1, 1024);
    let (b, b_addr) = x.buffer(T1, 1024);
    let before_update = x.commit(
        T1,
        q,
        &[blit(x.handle(), vec![wait_fence(), fill(a, 1024, 1)]), blit(x.handle(), vec![fill(b, 1024, 2), update()])]
            .concat(),
        &[fence, a, b],
    );
    let same_encoder = x.commit(T1, q, &blit(x.handle(), vec![fill(a, 1024, 3), update(), wait_fence()]), &[a, fence]);
    x.idle();
    let posted = x.em.take_posted();
    let c = completion(&posted, across);
    assert_completed(&c);
    let wait_before = golden::case("fence.wait-before-later-update");
    let c = completion(&posted, before_update);
    assert_eq!(c.status, golden::value(&wait_before, "status").u64());
    assert!(golden::value(&wait_before, "completed-within-20s").bool());
    let same = golden::case("fence.update-then-wait-same-encoder");
    assert_eq!(completion(&posted, same_encoder).status, golden::value(&same, "status").u64());
    // The copy ran after the first fill and before the second.
    assert_eq!(x.guest(T1, addr, 1024), vec![0x11; 1024]);
    assert_eq!(x.guest(T1, a_addr, 1024), vec![3; 1024]);
    assert_eq!(x.guest(T1, b_addr, 1024), vec![2; 1024]);
    // A fence that is not a fence is refused.
    let bad = blit(x.handle(), vec![Command::BlitWaitForFence(BlitWaitForFence { fence: Some(first) })]);
    assert_eq!(x.commit_with(T1, q, &bad, &[first], true).1.status, status::INVALID);
}

// Protocol violations and unimplemented records -------------------------------------------------------------------

/// The misuse the reference harness records as a terminated process in the driver (`command-buffer.commit-twice`,
/// `command-buffer.commit-with-active-encoder`, `command-buffer.encoder-while-encoder-open`,
/// `event.signal-with-open-encoder`) never reaches the host from a guest that does what Metal does; if it does, it is
/// refused as a protocol violation and nothing of it runs.
#[test]
fn misuse_metal_terminates_on_is_refused() {
    let x = Harness::new();
    x.open(T1);
    let q = x.queue(T1);
    let event = x.event(T1);
    let (data, _) = x.buffer(T1, 256);
    for case in [
        "command-buffer.commit-twice",
        "command-buffer.commit-with-active-encoder",
        "command-buffer.encoder-while-encoder-open",
        "event.signal-with-open-encoder",
    ] {
        assert!(golden::terminated(&golden::case(case)), "{case}");
    }
    // Commit twice.
    let cb = x.commit(T1, q, &blit(x.handle(), vec![fill(data, 256, 1)]), &[data]);
    let mut bytes = Vec::new();
    Command::EncoderEndEncoding(EncoderEndEncoding {}).encode(&mut bytes).unwrap();
    let again = Request::Commit(Commit {
        command_buffer: cb,
        queue: q,
        label: None,
        retained_references: true,
        error_options: 0,
        protection_options: 0,
        stream: GuestAddr(x.em.alloc(T1, Vec::new())),
        stream_length: 0,
        resources: vec![],
    });
    assert_eq!(x.status(T1, again).status, status::INVALID);
    // An encoder left open; an encoder opened while one is open; a signal with an encoder open.
    let open = vec![Command::BeginBlit(BeginBlit { encoder: x.handle() }), fill(data, 256, 1)];
    assert_eq!(x.commit_with(T1, q, &open, &[data], true).1.status, status::INVALID);
    let nested = [
        vec![Command::BeginBlit(BeginBlit { encoder: x.handle() })],
        blit(x.handle(), vec![]),
        vec![Command::EncoderEndEncoding(EncoderEndEncoding {})],
    ]
    .concat();
    assert_eq!(x.commit_with(T1, q, &nested, &[], true).1.status, status::INVALID);
    let signalled = blit(x.handle(), vec![signal(event, 1)]);
    assert_eq!(x.commit_with(T1, q, &signalled, &[event], true).1.status, status::INVALID);
    // A nil event (Metal raises NSInvalidArgumentException at encode time).
    let nil = [Command::EncodeSignalEvent(EncodeSignalEvent { event: None, value: 1 })];
    assert_eq!(x.commit_with(T1, q, &nil, &[], true).1.status, status::INVALID);
    x.idle();
    let posted = x.em.take_posted();
    // Only the first command buffer ran.
    assert_eq!(queue_events(&posted, q).len(), 2);
    assert_completed(&completion(&posted, cb));
}

#[test]
fn requests_obey_the_protocol() {
    let x = Harness::new();
    // Before Open; a wrong schema hash; Open twice.
    assert_eq!(
        x.status(T1, Request::CreateCommandQueue(CreateCommandQueue { queue: x.handle() })).status,
        status::INVALID
    );
    let wrong = x.status(T1, Request::Open(Open { schema_hash: Hash { bytes: [0; 32] }, pid: 1 }));
    assert_eq!(wrong.status, status::INVALID);
    x.open(T1);
    assert_eq!(
        x.status(T1, Request::Open(Open { schema_hash: Hash { bytes: SCHEMA_HASH }, pid: 1 })).status,
        status::INVALID
    );
    let OpenReply { schema_hash } =
        match x.request(T2, Request::Open(Open { schema_hash: Hash { bytes: SCHEMA_HASH }, pid: 2 })) {
            Reply::Open(Ok(r)) => r,
            other => panic!("{other:?}"),
        };
    assert_eq!(schema_hash.bytes, SCHEMA_HASH);

    // Handles: reuse, unknown, released, wrong kind, another task's.
    let q = x.queue(T1);
    assert_eq!(x.status(T1, Request::CreateCommandQueue(CreateCommandQueue { queue: q })).status, status::INVALID);
    assert_eq!(x.status(T1, Request::Release(Release { object: x.handle() })).status, status::INVALID);
    let e = x.event(T1);
    x.ok(T1, Request::Release(Release { object: e }));
    assert_eq!(x.status(T1, Request::Release(Release { object: e })).status, status::INVALID);
    assert_eq!(x.status(T1, Request::CreateEvent(CreateEvent { event: e })).status, status::INVALID);
    let gpu_priority = Request::SetGPUPriority(SetGPUPriority { queue: q, priority: 1 });
    assert_eq!(x.status(T2, gpu_priority.clone()).status, status::INVALID);
    let f = x.status(T1, Request::FetchEvents(FetchEvents { queue: Some(q) }));
    assert_eq!(f.status, status::INVALID);
    let (b, _) = x.buffer(T1, 64);
    assert_eq!(x.status(T1, Request::SetGPUPriority(SetGPUPriority { queue: b, priority: 1 })).status, status::INVALID);

    // A malformed record: non-zero padding after the handle of CreateEvent is refused with the request's opcode.
    let mut bytes = Vec::new();
    Request::CreateEvent(CreateEvent { event: x.handle() }).encode(&mut bytes).unwrap();
    bytes[4] = 24;
    bytes.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
    let r = x.raw(T1, &bytes);
    assert_eq!(r.request_opcode(), opcode::CREATE_EVENT);
    assert_eq!(schema::reply_failure(&r).unwrap().status, status::INVALID);
    // Bytes that are not a control request have no reply record.
    let mut stream_record = Vec::new();
    Command::EncoderEndEncoding(EncoderEndEncoding {}).encode(&mut stream_record).unwrap();
    assert!(x.gpu.block_on(x.exec.control(T1, &stream_record), WAIT).unwrap().is_err());

    // A command queue descriptor with state whose effect is not established.
    let mut desc = CommandQueueDescriptor::random(&mut webmetal_wire::fuzz::Rng::new(3));
    desc.device_partition = 0;
    desc.is_open_gl_queue = true;
    desc.lock_parameter_buffer_size_to_max = false;
    let r = x.status(
        T1,
        Request::CreateCommandQueueWithDescriptor(CreateCommandQueueWithDescriptor {
            queue: x.handle(),
            descriptor: desc.clone(),
        }),
    );
    assert_eq!(r.status, status::UNIMPLEMENTED);
    desc.is_open_gl_queue = false;
    x.ok(
        T1,
        Request::CreateCommandQueueWithDescriptor(CreateCommandQueueWithDescriptor {
            queue: x.handle(),
            descriptor: desc,
        }),
    );
    // GPU priority: the value the A10X driver returns is not established.
    assert_eq!(x.status(T1, gpu_priority).status, status::UNIMPLEMENTED);
}

/// Records the host does not implement complete the command buffer with status 2 and the record's offset, after the
/// command buffers before it, without running anything of it; requests no service serves are answered with status 2.
#[test]
fn unimplemented_records_fail_loudly() {
    let x = Harness::with(false);
    x.open(T1);
    let q = x.queue(T1);
    let e = x.event(T1);
    let blit_cb = x.commit(T1, q, &blit(x.handle(), vec![]), &[]);
    let spi = [
        signal(e, 1),
        Command::EncodeSignalEventAgentMask(EncodeSignalEventAgentMask { event: Some(e), value: 2, agent_mask: 1 }),
    ];
    let spi_cb = x.commit(T1, q, &spi, &[e]);
    x.idle();
    let posted = x.em.take_posted();
    assert_eq!(
        queue_events(&posted, q).iter().map(|e| (e.0, e.1)).collect::<Vec<_>>(),
        vec![(blit_cb, "scheduled"), (spi_cb, "scheduled"), (blit_cb, "completed"), (spi_cb, "completed")]
    );
    let c = completion(&posted, blit_cb);
    assert_eq!(c.status, command_buffer_status::ERROR);
    assert_eq!(c.failure.status, status::UNIMPLEMENTED);
    assert_eq!(c.failed_offset, 0);
    assert!(c.failure.message.contains("blitCommandEncoder"), "{}", c.failure.message);
    let c = completion(&posted, spi_cb);
    assert_eq!(c.status, command_buffer_status::ERROR);
    assert!(c.failure.message.contains("encodeSignalEvent:value:agentMask:"), "{}", c.failure.message);
    let mut first = Vec::new();
    spi[0].encode(&mut first).unwrap();
    assert_eq!(c.failed_offset, first.len() as u64);
    // Nothing of it ran: the signal before the unimplemented record did not happen.
    let w = x.commit(T1, x.queue(T1), &[wait(e, 1)], &[e]);
    x.idle();
    assert!(
        x.em.take_posted().iter().all(|p| !matches!(&p.2, Event::CommandBufferCompleted(c) if c.command_buffer == w))
    );

    let r = x.status(T1, Request::CreateFence(CreateFence { fence: x.handle() }));
    assert_eq!(r.status, status::OK);
    let tex = Request::CreateTexture(CreateTexture {
        texture: x.handle(),
        descriptor: webmetal_wire::TextureDescriptor::random(&mut webmetal_wire::fuzz::Rng::new(1)),
    });
    let r = x.status(T1, tex);
    assert_eq!(r.status, status::UNIMPLEMENTED);
    assert!(r.message.contains("newTextureWithDescriptor:"), "{}", r.message);
    let reports = x.exec.take_reports();
    assert_eq!(reports.len(), 3, "{reports:?}");
}

// Device loss and out-of-memory --------------------------------------------------------------------------------------

/// Fault injection: the WebGPU device is destroyed with command buffers in flight. Every command buffer not yet
/// completed, and every one committed after, completes with an error carrying refusal `device-removed` (iOS defines
/// no code for it; `tables/limits.toml` question `6.1-device-removed-code`); every task receives DeviceLost; requests
/// that need the GPU fail the same way.
#[test]
fn device_loss() {
    let x = Harness::new();
    x.open(T1);
    x.open(T2);
    let (q1, q2) = (x.queue(T1), x.queue(T1));
    let e = x.event(T1);
    let (buf, _) = x.buffer(T1, 256);
    let done = x.commit(T1, q2, &blit(x.handle(), vec![fill(buf, 256, 1)]), &[buf]);
    x.idle();
    assert_completed(&completion(&x.em.take_posted(), done));
    let in_flight =
        x.commit(T1, q1, &[vec![wait(e, 1)], blit(x.handle(), vec![fill(buf, 256, 2)])].concat(), &[e, buf]);
    x.idle();

    x.device.destroy();
    x.idle();
    let after = x.commit(T1, q2, &blit(x.handle(), vec![fill(buf, 256, 3)]), &[buf]);
    x.idle();
    let posted = x.em.take_posted();
    for t in [T1, T2] {
        let lost: Vec<_> = posted.iter().filter(|p| p.0 == t && matches!(p.2, Event::DeviceLost(_))).collect();
        assert_eq!(lost.len(), 1, "{t:?}: {posted:?}");
        assert_eq!(lost[0].1, None);
        let Event::DeviceLost(d) = &lost[0].2 else { unreachable!() };
        assert_eq!(d.failure.status, status::UNIMPLEMENTED);
        assert!(d.failure.message.starts_with("device-removed:"), "{}", d.failure.message);
    }
    for cb in [in_flight, after] {
        assert_eq!(scheduled_count(&posted, cb), 1);
        let c = completion(&posted, cb);
        assert_eq!(c.status, command_buffer_status::ERROR);
        assert_eq!(c.failure.status, status::UNIMPLEMENTED);
        assert!(c.failure.message.starts_with("device-removed:"));
    }
    let r = x.status(
        T1,
        Request::CreateBuffer(CreateBuffer { buffer: x.handle(), length: 4, options: 0, contents: GuestAddr(0) }),
    );
    assert_eq!(r.status, status::UNIMPLEMENTED);
    assert!(r.message.starts_with("device-removed:"));
    // Requests that do not need the GPU still work.
    x.queue(T1);
}

/// wgpu-native v29.0.1.1 reports a failed texture allocation as a `GPUValidationError`, not a `GPUOutOfMemoryError`:
/// its `handle_error` (src/lib.rs) looks for wgpu-core's `DeviceError::OutOfMemory` along the error's `source()` chain,
/// and `CreateTextureError::Device` is `#[error(transparent)]`, which hides it. The executor cannot tell that report
/// from one of its own defects and reports it as one (status 2): loud, not a Metal error code.
const WGPU_OOM_AS_VALIDATION: bool = cfg!(webgpu_native = "wgpu-native");

/// Fault injection: WebGPU runs out of memory while a command buffer's encoder is recorded. The command buffer
/// completes with `MTLCommandBufferErrorDomain` / `MTLCommandBufferErrorOutOfMemory`; the device stays usable. A
/// creation request that runs out of memory fails with status 2 (Metal returns nil without an NSError, which the wire
/// cannot carry yet) and its handle stays unbound.
#[test]
fn out_of_memory() {
    let x = Harness::new();
    x.open(T1);
    let q = x.queue(T1);
    let (buf, addr) = x.buffer(T1, 256);
    let oom = x.commit(
        T1,
        q,
        &[
            Command::BeginCompute(BeginCompute { encoder: x.handle() }),
            Command::EncoderEndEncoding(EncoderEndEncoding {}),
        ],
        &[],
    );
    let next = x.commit(T1, q, &blit(x.handle(), vec![fill(buf, 256, 6)]), &[buf]);
    x.idle();
    let posted = x.em.take_posted();
    let c = completion(&posted, oom);
    assert_eq!(c.status, command_buffer_status::ERROR);
    assert_eq!(c.failed_offset, NO_RECORD);
    if WGPU_OOM_AS_VALIDATION {
        assert_eq!(c.failure.status, status::UNIMPLEMENTED, "{:?}", c.failure);
        assert!(c.failure.message.contains("Not enough memory left"), "{}", c.failure.message);
    } else {
        assert_eq!(c.failure.status, status::METAL_ERROR, "{:?}", c.failure);
        assert_eq!(c.failure.domain.as_deref(), Some(COMMAND_BUFFER_ERROR_DOMAIN));
        assert_eq!(c.failure.code, command_buffer_error::OUT_OF_MEMORY);
    }
    assert_completed(&completion(&posted, next));
    assert_eq!(x.guest(T1, addr, 256), vec![6; 256]);

    let b = x.handle();
    let r = x.status(
        T1,
        Request::CreateBuffer(CreateBuffer { buffer: b, length: FAULT_LENGTH, options: 0, contents: GuestAddr(0) }),
    );
    assert_eq!(r.status, status::UNIMPLEMENTED);
    let expected = if WGPU_OOM_AS_VALIDATION { "Not enough memory left" } else { "out of memory" };
    assert!(r.message.contains(expected), "{}", r.message);
    assert_eq!(x.status(T1, Request::Release(Release { object: b })).status, status::INVALID);
}

// Tasks ------------------------------------------------------------------------------------------------------------

/// A task that exits drops its command buffers without events and releases its shared events; a released queue keeps
/// running the command buffers committed to it; committed_work_done waits for every earlier command buffer.
#[test]
fn task_exit_and_release() {
    let x = Harness::new();
    x.open(T1);
    x.open(T2);
    let q = x.queue(T1);
    x.em.shared_event(T1, 1, 7, 0);
    let shared = x.handle();
    x.ok(T1, Request::RegisterSharedEvent(RegisterSharedEvent { event: shared, port: 1 }));
    x.commit(T1, q, &[wait(shared, 1)], &[shared]);
    x.idle();
    x.exec.task_exited(T1);
    x.em.cpu_signal(7, 1);
    x.exec.shared_event_signaled(SharedEventId(7));
    x.idle();
    assert!(x.em.take_posted().is_empty());
    assert_eq!(*x.em.released.borrow(), vec![7]);

    let q2 = x.queue(T2);
    let (buf, addr) = x.buffer(T2, 256);
    let cb = x.commit(T2, q2, &blit(x.handle(), vec![fill(buf, 256, 8)]), &[buf]);
    x.ok(T2, Request::Release(Release { object: q2 }));
    x.gpu.block_on(x.exec.committed_work_done(Some(T2)), WAIT).unwrap();
    assert_completed(&completion(&x.em.take_posted(), cb));
    assert_eq!(x.guest(T2, addr, 256), vec![8; 256]);
}
