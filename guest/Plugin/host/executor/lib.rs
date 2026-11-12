//! The WebMetal host executor inside the guest plugin's macOS test shim.
//!
//! On the target, the plugin's control requests reach the executor through the WebMetal accelerator service (wire
//! README, "Transport"). The macOS test shim has no such service: it runs the executor in the plugin's own process, on
//! a native WebGPU implementation (webgpu.h over Dawn or wgpu-native, `webmetal-host`'s `native` feature), and passes
//! each request to it through [`webmetal_shim_executor_control`]. The guest task is this process; its memory is read
//! and written with the Mach VM calls, so an unmapped or protected address fails the way the embedder interface
//! requires, and IOSurfaces are this Mac's own, found by their IDs.
//!
//! The executor and its WebGPU device belong to one thread, started by [`webmetal_shim_executor_start`]. Requests from
//! any thread queue to it and wait for their replies. Between requests the thread keeps the executor's GPU work moving.
//!
//! The executor serves the memory module's requests and coherence, the state service's sampler states, the render
//! path's depth-stencil states, and records render, compute and blit encoders with the render, compute and blit
//! services.
//!
//! Shared events are this process's IOSurface shared events, which the shim reaches through the callbacks it registers
//! with [`webmetal_shim_executor_set_shared_events`]; a CPU-side signal it observes reaches the executor through
//! [`webmetal_shim_executor_shared_event_signaled`]. What the shim's embedder does not provide fails as the interface
//! defines: the presentation sink (`UnknownDisplay`); the executor reports the failure through its reply.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Mutex, OnceLock};
use std::task::Context;
use std::time::Duration;

use webmetal_host::embed::{
    Embedder, GuestMemoryError, GuestMemoryErrorKind, MonotonicNanos, PresentError, PresentedFrame, SharedEventError,
    SharedEventId, SurfaceDesc, SurfaceError, SurfacePlane, TaskId,
};
use webmetal_host::exec::{Config, ControlService, Executor, Services, fetch_events_reply};
use webmetal_host::format::webgpu::{Feature, HostFeatures};
use webmetal_host::gpu::idl::{DeviceDescriptor, FeatureName, RequestAdapterOptions};
use webmetal_host::gpu::native::{Native, NativeGpu};
use webmetal_host::gpu::scope::{Granularity, Policy};
use webmetal_host::gpu::{Adapter as _, Device as _, Gpu as _, negotiate};
use webmetal_host::blit::BlitService;
use webmetal_host::compute::ComputeService;
use webmetal_host::mem::service::MemoryServices;
use webmetal_host::render::RenderServices;
use webmetal_host::state::StateServices;
use webmetal_wire::{Event, Handle, Request, SurfaceId};

/// How long a request may wait on the GPU without any progress before the shim stops the process.
const STALL: Duration = Duration::from_secs(120);

// Mach and IOSurface ---------------------------------------------------------------------------------------------------

type KernReturn = i32;
const KERN_SUCCESS: KernReturn = 0;
const KERN_PROTECTION_FAILURE: KernReturn = 2;

#[repr(C)]
#[derive(Default)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

type IOSurfaceRef = *mut c_void;
const IOSURFACE_LOCK_READ_ONLY: u32 = 1;

unsafe extern "C" {
    static mach_task_self_: u32;
    static vm_page_size: usize;
    fn mach_vm_read_overwrite(task: u32, address: u64, size: u64, data: u64, out_size: *mut u64) -> KernReturn;
    fn mach_vm_write(task: u32, address: u64, data: usize, count: u32) -> KernReturn;
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> KernReturn;
    fn getpid() -> i32;
    fn abort() -> !;
}

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceLookup(id: u32) -> IOSurfaceRef;
    fn IOSurfaceGetWidth(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetHeight(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetBytesPerRow(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetBytesPerElement(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetElementWidth(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetElementHeight(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetPixelFormat(s: IOSurfaceRef) -> u32;
    fn IOSurfaceGetAllocSize(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetSeed(s: IOSurfaceRef) -> u32;
    fn IOSurfaceGetPlaneCount(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetWidthOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
    fn IOSurfaceGetHeightOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
    fn IOSurfaceGetBytesPerRowOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
    fn IOSurfaceGetBytesPerElementOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
    fn IOSurfaceGetElementWidthOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
    fn IOSurfaceGetElementHeightOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
    fn IOSurfaceGetBaseAddress(s: IOSurfaceRef) -> *mut c_void;
    fn IOSurfaceGetBaseAddressOfPlane(s: IOSurfaceRef, plane: usize) -> *mut c_void;
    fn IOSurfaceLock(s: IOSurfaceRef, options: u32, seed: *mut u32) -> KernReturn;
    fn IOSurfaceUnlock(s: IOSurfaceRef, options: u32, seed: *mut u32) -> KernReturn;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(object: *const c_void);
}

/// A surface found by its ID, released when dropped.
struct Surface(IOSurfaceRef);

impl Surface {
    fn lookup(id: SurfaceId) -> Result<Surface, SurfaceError> {
        let s = unsafe { IOSurfaceLookup(id.0) };
        if s.is_null() { Err(SurfaceError::Unknown(id)) } else { Ok(Surface(s)) }
    }

    fn checked(&self, id: SurfaceId, offset: u64, len: usize) -> Result<usize, SurfaceError> {
        let size = unsafe { IOSurfaceGetAllocSize(self.0) } as u64;
        match offset.checked_add(len as u64) {
            Some(end) if end <= size => Ok(offset as usize),
            _ => Err(SurfaceError::OutOfRange { id, offset, len: len as u64 }),
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { CFRelease(self.0) }
    }
}

// Embedder -------------------------------------------------------------------------------------------------------------

/// The doorbell the guest registered: called on the executor's thread with the queue handle (0: device-level events)
/// whenever events for that queue are queued.
#[derive(Clone, Copy)]
struct Doorbell {
    ring: extern "C" fn(context: *mut c_void, queue: u64),
    context: *mut c_void,
}

// The context is the guest's; the shim only passes it back.
unsafe impl Send for Doorbell {}

static DOORBELL: Mutex<Option<Doorbell>> = Mutex::new(None);

/// The shim's shared events: resolve a send right of this task to an event id, read and set an event's value, release
/// an id. Each returns false for an event it does not know.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SharedEventCallbacks {
    pub context: *mut c_void,
    pub resolve: extern "C" fn(context: *mut c_void, port: u32, event: *mut u64) -> bool,
    pub value: extern "C" fn(context: *mut c_void, event: u64, value: *mut u64) -> bool,
    pub signal: extern "C" fn(context: *mut c_void, event: u64, value: u64) -> bool,
    pub release: extern "C" fn(context: *mut c_void, event: u64),
}

// The context is the shim's; the executor only passes it back.
unsafe impl Send for SharedEventCallbacks {}

static SHARED_EVENTS: Mutex<Option<SharedEventCallbacks>> = Mutex::new(None);

fn shared_events() -> Option<SharedEventCallbacks> {
    *SHARED_EVENTS.lock().unwrap_or_else(|e| e.into_inner())
}

/// This process as the executor's guest task.
struct InProcess {
    task: TaskId,
    events: RefCell<HashMap<Option<Handle>, Vec<Event>>>,
    timebase: MachTimebaseInfo,
}

impl InProcess {
    fn new() -> InProcess {
        let mut timebase = MachTimebaseInfo::default();
        if unsafe { mach_timebase_info(&mut timebase) } != KERN_SUCCESS || timebase.denom == 0 {
            fatal("mach_timebase_info failed");
        }
        InProcess { task: TaskId(unsafe { getpid() } as u64), events: RefCell::new(HashMap::new()), timebase }
    }

    fn check_task(&self, task: TaskId, addr: u64, len: usize) -> Result<(), GuestMemoryError> {
        if task == self.task {
            return Ok(());
        }
        Err(GuestMemoryError { task, addr, len: len as u64, fault: addr, kind: GuestMemoryErrorKind::NoTask })
    }

    /// The failure of an access to `[addr, addr + len)`, naming the first page the access cannot reach.
    fn fault(&self, task: TaskId, addr: u64, len: usize, write: bool) -> GuestMemoryError {
        let page = unsafe { vm_page_size } as u64;
        let mut at = addr;
        let end = addr + len as u64;
        let mut byte = 0u8;
        while at < end {
            let mut got = 0u64;
            let kr = unsafe { mach_vm_read_overwrite(mach_task_self_, at, 1, &mut byte as *mut u8 as u64, &mut got) };
            let kr = if kr == KERN_SUCCESS && write {
                unsafe { mach_vm_write(mach_task_self_, at, &byte as *const u8 as usize, 1) }
            } else {
                kr
            };
            if kr != KERN_SUCCESS {
                let kind = if kr == KERN_PROTECTION_FAILURE {
                    GuestMemoryErrorKind::Protection
                } else {
                    GuestMemoryErrorKind::Unmapped
                };
                return GuestMemoryError { task, addr, len: len as u64, fault: at, kind };
            }
            at = (at / page + 1) * page;
        }
        // The whole range is reachable now; the failed call raced a change of the mapping.
        GuestMemoryError { task, addr, len: len as u64, fault: addr, kind: GuestMemoryErrorKind::Unmapped }
    }
}

impl Embedder for InProcess {
    fn read_guest(&self, task: TaskId, addr: u64, dst: &mut [u8]) -> Result<(), GuestMemoryError> {
        self.check_task(task, addr, dst.len())?;
        if dst.is_empty() {
            return Ok(());
        }
        let mut got = 0u64;
        let kr = unsafe {
            mach_vm_read_overwrite(mach_task_self_, addr, dst.len() as u64, dst.as_mut_ptr() as u64, &mut got)
        };
        if kr != KERN_SUCCESS || got != dst.len() as u64 {
            return Err(self.fault(task, addr, dst.len(), false));
        }
        Ok(())
    }

    fn write_guest(&self, task: TaskId, addr: u64, src: &[u8]) -> Result<(), GuestMemoryError> {
        self.check_task(task, addr, src.len())?;
        for (i, chunk) in src.chunks(u32::MAX as usize).enumerate() {
            let at = addr + (i * u32::MAX as usize) as u64;
            let kr = unsafe { mach_vm_write(mach_task_self_, at, chunk.as_ptr() as usize, chunk.len() as u32) };
            if kr != KERN_SUCCESS {
                return Err(self.fault(task, at, chunk.len(), true));
            }
        }
        Ok(())
    }

    fn surface_lookup(&self, id: SurfaceId) -> Result<SurfaceDesc, SurfaceError> {
        let s = Surface::lookup(id)?;
        let r = s.0;
        unsafe {
            let base = IOSurfaceGetBaseAddress(r) as u64;
            let planes = (0..IOSurfaceGetPlaneCount(r))
                .map(|p| SurfacePlane {
                    width: IOSurfaceGetWidthOfPlane(r, p) as u64,
                    height: IOSurfaceGetHeightOfPlane(r, p) as u64,
                    bytes_per_row: IOSurfaceGetBytesPerRowOfPlane(r, p) as u64,
                    bytes_per_element: IOSurfaceGetBytesPerElementOfPlane(r, p) as u64,
                    element_width: IOSurfaceGetElementWidthOfPlane(r, p) as u64,
                    element_height: IOSurfaceGetElementHeightOfPlane(r, p) as u64,
                    offset: IOSurfaceGetBaseAddressOfPlane(r, p) as u64 - base,
                    size: IOSurfaceGetBytesPerRowOfPlane(r, p) as u64 * IOSurfaceGetHeightOfPlane(r, p) as u64,
                })
                .collect();
            Ok(SurfaceDesc {
                id,
                width: IOSurfaceGetWidth(r) as u64,
                height: IOSurfaceGetHeight(r) as u64,
                bytes_per_row: IOSurfaceGetBytesPerRow(r) as u64,
                bytes_per_element: IOSurfaceGetBytesPerElement(r) as u64,
                element_width: IOSurfaceGetElementWidth(r) as u64,
                element_height: IOSurfaceGetElementHeight(r) as u64,
                pixel_format: IOSurfaceGetPixelFormat(r),
                alloc_size: IOSurfaceGetAllocSize(r) as u64,
                planes,
                seed: IOSurfaceGetSeed(r),
            })
        }
    }

    fn read_surface(&self, id: SurfaceId, offset: u64, dst: &mut [u8]) -> Result<(), SurfaceError> {
        let s = Surface::lookup(id)?;
        let at = s.checked(id, offset, dst.len())?;
        unsafe {
            if IOSurfaceLock(s.0, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut()) != KERN_SUCCESS {
                fatal("IOSurfaceLock (read-only) failed");
            }
            let base = IOSurfaceGetBaseAddress(s.0) as *const u8;
            std::ptr::copy_nonoverlapping(base.add(at), dst.as_mut_ptr(), dst.len());
            IOSurfaceUnlock(s.0, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut());
        }
        Ok(())
    }

    fn write_surface_changed(&self, id: SurfaceId, offset: u64, new: &[u8], shadow: &[u8]) -> Result<(), SurfaceError> {
        assert_eq!(new.len(), shadow.len(), "write_surface_changed: new and shadow differ in length");
        let s = Surface::lookup(id)?;
        let at = s.checked(id, offset, new.len())?;
        unsafe {
            if IOSurfaceLock(s.0, 0, std::ptr::null_mut()) != KERN_SUCCESS {
                fatal("IOSurfaceLock failed");
            }
            let base = (IOSurfaceGetBaseAddress(s.0) as *mut u8).add(at);
            for (i, (&n, &o)) in new.iter().zip(shadow).enumerate() {
                if n != o {
                    *base.add(i) = n;
                }
            }
            IOSurfaceUnlock(s.0, 0, std::ptr::null_mut());
        }
        Ok(())
    }

    fn surface_note_gpu_write(&self, id: SurfaceId, _plane: u32) {
        // A write lock and unlock changes the surface's seed, as a GPU write does on the target.
        if let Ok(s) = Surface::lookup(id) {
            unsafe {
                IOSurfaceLock(s.0, 0, std::ptr::null_mut());
                IOSurfaceUnlock(s.0, 0, std::ptr::null_mut());
            }
        }
    }

    fn shared_event_resolve(&self, task: TaskId, port: u32) -> Result<SharedEventId, SharedEventError> {
        let mut id = 0u64;
        match shared_events() {
            Some(c) if task == self.task && (c.resolve)(c.context, port, &mut id) => Ok(SharedEventId(id)),
            _ => Err(SharedEventError::NotAnEvent { task, port }),
        }
    }

    fn shared_event_value(&self, event: SharedEventId) -> Result<u64, SharedEventError> {
        let mut value = 0u64;
        match shared_events() {
            Some(c) if (c.value)(c.context, event.0, &mut value) => Ok(value),
            _ => Err(SharedEventError::Gone(event)),
        }
    }

    fn shared_event_signal(&self, event: SharedEventId, value: u64) -> Result<(), SharedEventError> {
        match shared_events() {
            Some(c) if (c.signal)(c.context, event.0, value) => Ok(()),
            _ => Err(SharedEventError::Gone(event)),
        }
    }

    fn shared_event_release(&self, event: SharedEventId) {
        if let Some(c) = shared_events() {
            (c.release)(c.context, event.0);
        }
    }

    fn post(&self, task: TaskId, queue: Option<Handle>, event: Event) {
        if task != self.task {
            return;
        }
        self.events.borrow_mut().entry(queue).or_default().push(event);
        let doorbell = *DOORBELL.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(d) = doorbell {
            (d.ring)(d.context, queue.map_or(0, |h| h.0.get()));
        }
    }

    fn present(&self, frame: &PresentedFrame) -> Result<(), PresentError> {
        Err(PresentError::UnknownDisplay(frame.display))
    }

    fn now(&self) -> MonotonicNanos {
        let t = unsafe { mach_absolute_time() } as u128;
        MonotonicNanos((t * self.timebase.numer as u128 / self.timebase.denom as u128) as u64)
    }
}

// The executor's thread ------------------------------------------------------------------------------------------------

enum Job {
    Control { request: Vec<u8>, reply: SyncSender<Result<Vec<u8>, String>> },
    /// A CPU-side signal changed a shared event's value: waits on it are checked again.
    Signaled(u64),
}

static QUEUE: OnceLock<Mutex<Sender<Job>>> = OnceLock::new();

fn fatal(what: &str) -> ! {
    eprintln!("WebMetal: shim executor: {what}");
    unsafe { abort() }
}

type Exec = Executor<Native, InProcess>;

fn set_up() -> Result<(NativeGpu, Exec, Rc<InProcess>), String> {
    let gpu = NativeGpu::new();
    let device = gpu
        .block_on(
            async {
                let adapter = gpu
                    .request_adapter(&RequestAdapterOptions::new())
                    .await
                    .map_err(|e| format!("requestAdapter: {e:?}"))?
                    .ok_or("requestAdapter: no adapter")?;
                let mut descriptor = match negotiate::plan(&adapter.features(), &adapter.limits()) {
                    Ok(p) => p.descriptor,
                    // wgpu-native reports no "core-features-and-limits"; ask for every feature it offers with the
                    // default limits, as the executor's own native tests do.
                    Err(negotiate::NegotiationError::NotCore) => {
                        let offered = adapter.features();
                        let mut d = DeviceDescriptor::new();
                        d.required_features =
                            FeatureName::ALL.into_iter().filter(|f| offered.contains(f.as_str())).collect();
                        d
                    }
                    Err(e) => return Err(format!("negotiation: {e}")),
                };
                descriptor.label = "WebMetal shim".into();
                adapter.request_device(&descriptor).await.map_err(|e| format!("requestDevice: {e:?}"))
            },
            STALL,
        )
        .map_err(|s| format!("device request stalled: {s:?}"))??;
    let names = device.features();
    let features = HostFeatures::new(Feature::ALL.into_iter().filter(|f| names.contains(f.name())));
    let embedder = Rc::new(InProcess::new());
    let memory = gpu
        .block_on(MemoryServices::new(&device, features, embedder.clone()), STALL)
        .map_err(|s| format!("memory set-up stalled: {s:?}"))?
        .map_err(|e| format!("memory set-up: {e}"))?;
    // The memory module's buffers and textures are the ones the encoder services record their passes over.
    let render = RenderServices::new(memory.memory());
    let services = Services {
        control: vec![
            memory.control() as Box<dyn ControlService<Native, InProcess>>,
            Box::new(StateServices::new()),
            render.control(),
        ],
        render: Some(render.encoder()),
        compute: Some(Box::new(ComputeService::new(memory.memory()))),
        blit: Some(Box::new(BlitService::new(memory.memory()))),
        coherence: Some(memory.coherence()),
    };
    let config = Config { scopes: Policy { granularity: Granularity::PerCall, strict: true } };
    let exec = Executor::new(device, embedder.clone(), services, config);
    Ok((gpu, exec, embedder))
}

fn serve(gpu: &NativeGpu, exec: &Exec, embedder: &InProcess, request: &[u8]) -> Result<Vec<u8>, String> {
    // FetchEvents is answered from the events the executor posted (webmetal_host::exec, "Requests").
    if let Ok((Request::FetchEvents(f), _)) = Request::decode(request, 0) {
        let events = embedder.events.borrow_mut().remove(&f.queue).unwrap_or_default();
        return fetch_events_reply(&events).map_err(|e| format!("FetchEvents reply: {e:?}"));
    }
    match gpu.block_on(exec.control(embedder.task, request), STALL) {
        Ok(Ok(reply)) => Ok(reply),
        Ok(Err(e)) => Err(format!("not a control request: {e:?}")),
        Err(s) => fatal(&format!("a control request stalled: {s:?}")),
    }
}

fn run(gpu: NativeGpu, exec: Exec, embedder: Rc<InProcess>, jobs: Receiver<Job>) {
    let mut cx = Context::from_waker(std::task::Waker::noop());
    loop {
        let busy = exec.pump(&mut cx);
        let job = if busy {
            gpu.process_events();
            match jobs.recv_timeout(Duration::from_millis(1)) {
                Ok(job) => job,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        } else {
            match jobs.recv() {
                Ok(job) => job,
                Err(_) => return,
            }
        };
        match job {
            Job::Control { request, reply } => {
                let _ = reply.send(serve(&gpu, &exec, &embedder, &request));
            }
            Job::Signaled(event) => exec.shared_event_signaled(SharedEventId(event)),
        }
    }
}

// C interface ----------------------------------------------------------------------------------------------------------

fn write_text(text: &str, buffer: *mut c_char, capacity: usize) {
    if buffer.is_null() || capacity == 0 {
        return;
    }
    let n = text.len().min(capacity - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), buffer as *mut u8, n);
        *buffer.add(n) = 0;
    }
}

/// Starts the executor's thread with its WebGPU device, once per process. Returns false, with the reason in `error`,
/// when no device or executor could be set up.
#[unsafe(no_mangle)]
pub extern "C" fn webmetal_shim_executor_start(error: *mut c_char, capacity: usize) -> bool {
    static STARTED: Mutex<Option<Result<(), String>>> = Mutex::new(None);
    let mut started = STARTED.lock().unwrap_or_else(|e| e.into_inner());
    if started.is_none() {
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), String>>(1);
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let spawned = std::thread::Builder::new().name("WebMetal shim executor".into()).spawn(move || match set_up() {
            Ok((gpu, exec, embedder)) => {
                let _ = ready_tx.send(Ok(()));
                run(gpu, exec, embedder, job_rx);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        });
        let outcome = match spawned {
            Ok(_) => ready_rx.recv().unwrap_or_else(|_| Err("the executor thread ended during set-up".into())),
            Err(e) => Err(format!("cannot start the executor thread: {e}")),
        };
        if outcome.is_ok() {
            let _ = QUEUE.set(Mutex::new(job_tx));
        }
        *started = Some(outcome);
    }
    match started.as_ref().expect("set above") {
        Ok(()) => true,
        Err(e) => {
            write_text(e, error, capacity);
            false
        }
    }
}

/// Runs one control request and returns its reply record in a buffer the caller frees with
/// [`webmetal_shim_executor_free`]. Returns false, with the reason in `error`, when the executor is not running or the
/// bytes are not a control request (the transport refuses the call, as the service does on the target).
///
/// # Safety
///
/// `request` points to `length` readable bytes; `reply` and `reply_length` are writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn webmetal_shim_executor_control(
    request: *const u8,
    length: usize,
    reply: *mut *mut u8,
    reply_length: *mut usize,
    error: *mut c_char,
    capacity: usize,
) -> bool {
    let Some(queue) = QUEUE.get() else {
        write_text("the executor is not running", error, capacity);
        return false;
    };
    let bytes = unsafe { std::slice::from_raw_parts(request, length) }.to_vec();
    let (tx, rx) = mpsc::sync_channel(1);
    if queue.lock().unwrap_or_else(|e| e.into_inner()).send(Job::Control { request: bytes, reply: tx }).is_err() {
        write_text("the executor thread has ended", error, capacity);
        return false;
    }
    match rx.recv() {
        Ok(Ok(out)) => {
            let boxed = out.into_boxed_slice();
            unsafe {
                *reply_length = boxed.len();
                *reply = Box::into_raw(boxed) as *mut u8;
            }
            true
        }
        Ok(Err(e)) => {
            write_text(&e, error, capacity);
            false
        }
        Err(_) => {
            write_text("the executor thread ended during the request", error, capacity);
            false
        }
    }
}

/// Frees a reply [`webmetal_shim_executor_control`] returned.
///
/// # Safety
///
/// `reply` and `length` are exactly what one successful call returned, and are freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn webmetal_shim_executor_free(reply: *mut u8, length: usize) {
    if !reply.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(reply, length)) });
    }
}

/// Registers the event doorbell: `ring(context, queue)` runs on the executor's thread whenever events for `queue` (0:
/// device-level events) are queued; the guest then collects them with FetchEvents. A null `ring` removes it.
#[unsafe(no_mangle)]
pub extern "C" fn webmetal_shim_executor_set_doorbell(
    ring: Option<extern "C" fn(context: *mut c_void, queue: u64)>,
    context: *mut c_void,
) {
    *DOORBELL.lock().unwrap_or_else(|e| e.into_inner()) = ring.map(|ring| Doorbell { ring, context });
}

/// Registers the shim's shared events (see [`SharedEventCallbacks`]); before it, every shared event is refused.
///
/// # Safety
///
/// `callbacks` points to a readable value; its functions may be called from the executor's thread until the process
/// ends.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn webmetal_shim_executor_set_shared_events(callbacks: *const SharedEventCallbacks) {
    *SHARED_EVENTS.lock().unwrap_or_else(|e| e.into_inner()) = Some(unsafe { *callbacks });
}

/// A CPU-side signal changed the value of shared event `event` (an id the resolve callback gave).
#[unsafe(no_mangle)]
pub extern "C" fn webmetal_shim_executor_shared_event_signaled(event: u64) {
    if let Some(queue) = QUEUE.get() {
        let _ = queue.lock().unwrap_or_else(|e| e.into_inner()).send(Job::Signaled(event));
    }
}
