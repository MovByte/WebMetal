//! The binding over webgpu.h, linked against Dawn or wgpu-native (chosen at build time, see `host/build.rs`).
//!
//! This layer is what a JavaScript binding is to a browser's GPU process: it runs the specification's
//! content-timeline steps itself (IDL value domains, the exceptions those steps throw, the `mapState` and mapping
//! bookkeeping, object attributes) and hands device-timeline work to the implementation through webgpu.h. The
//! implementation's own validation produces the device-timeline errors, which reach the error scopes and the
//! `uncapturederror` handler exactly as in a browser.
//!
//! Asynchronous operations use `WGPUCallbackMode_AllowProcessEvents`; their callbacks run on the thread that drives
//! the instance ([`NativeGpu::block_on`], or any poll of a future the binding returned). The device-lost and
//! uncaptured-error callbacks may run on any thread; they only record into shared state, which is delivered to the
//! program the next time the instance processes events, as a browser queues the `uncapturederror` event as a task.
//!
//! Where an implementation does not provide a webgpu.h entry point (see [`quirks`]), the member that needs it returns
//! [`ExceptionKind::Unsupported`], unless the member's observable behavior can be produced exactly from entry points
//! that exist.

pub mod sys;

mod conv;
mod desc;
mod objects;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, VecDeque};
use std::ffi::c_void;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use crate::gpu::api::*;
use crate::gpu::idl::*;
use crate::gpu::tables::{LIMITS, LimitClass};
use conv::{c_named, from_c, from_sv, sv, to_c};

pub use objects::*;

/// The entry points an implementation leaves unimplemented (calling one aborts the process), read from the
/// implementation's source: wgpu-native v29.0.1.1 `src/unimplemented.rs` and `src/lib.rs` (`wgpuGetInstanceFeatures`,
/// `wgpuHasInstanceFeature`). Dawn implements all of them.
pub mod quirks {
    /// `wgpuInstanceWaitAny`, `wgpuGetInstanceFeatures`, `wgpuHasInstanceFeature`.
    pub const WAIT_ANY: bool = !cfg!(webgpu_native = "wgpu-native");
    /// `wgpuDeviceCreateComputePipelineAsync`, `wgpuDeviceCreateRenderPipelineAsync`.
    pub const PIPELINE_ASYNC: bool = !cfg!(webgpu_native = "wgpu-native");
    /// `wgpuShaderModuleGetCompilationInfo`.
    pub const COMPILATION_INFO: bool = !cfg!(webgpu_native = "wgpu-native");
    /// `wgpuInstanceGetWGSLLanguageFeatures`.
    pub const WGSL_LANGUAGE_FEATURES: bool = !cfg!(webgpu_native = "wgpu-native");
    /// `wgpu*SetLabel` on every object type. Without it the `label` attribute still reads back what was set; the
    /// implementation's own messages keep naming the object by its creation label.
    pub const SET_LABEL: bool = !cfg!(webgpu_native = "wgpu-native");
    /// `wgpuDevicePopErrorScope` on an empty stack reports `WGPUPopErrorScopeStatus_Error`. wgpu-native's unwraps the
    /// pop instead (`src/lib.rs`, `error_sink.scopes.pop().unwrap()`), which aborts; the binding then answers from its
    /// own count of the scopes it pushed.
    pub const POP_EMPTY_ERROR_SCOPE: bool = !cfg!(webgpu_native = "wgpu-native");
    /// `wgpuDevicePushErrorScope` accepts `WGPUErrorFilter_Internal`. wgpu-native's panics on it (`src/lib.rs`,
    /// `panic!("invalid error filter")`), and it never reports an internal error (its error type has only device-lost,
    /// out-of-memory and validation variants). The binding keeps such a scope itself: it captures nothing, which is
    /// exactly what an internal scope captures on that implementation, and the errors it does report skip it as they
    /// would skip any scope of another filter.
    pub const INTERNAL_ERROR_SCOPE: bool = !cfg!(webgpu_native = "wgpu-native");
    /// `wgpuDeviceDestroy` loses the device and fires the device-lost callback with reason Destroyed. wgpu-native's
    /// only destroys (`src/lib.rs`, `context.device_destroy`); its device-lost callback fires only when a later
    /// operation fails with a device-lost error (`ErrorSinkRaw::handle_error`). The binding then performs "lose the
    /// device" itself when `destroy()` is called, as `destroy()` requires.
    pub const LOST_ON_DESTROY: bool = !cfg!(webgpu_native = "wgpu-native");
    /// The implementation lists "core-features-and-limits" among the features of a Core-defaulting adapter and of the
    /// devices made from it (§4.2.2 "core"; §25.1: the feature is always available unless featureLevel is
    /// "compatibility"). wgpu-native lists it for none: its feature conversion has no entry for
    /// `WGPUFeatureName_CoreFeaturesAndLimits` (`src/conv.rs`, `features_to_native` and `map_feature`; wgpu-types has
    /// no such feature), and it never reads `WGPURequestAdapterOptions.featureLevel` (`src/lib.rs`,
    /// `wgpuInstanceRequestAdapter`). wgpu-core enforces no Compatibility Mode rule, so every adapter it returns is
    /// returned as for "core" (§4.2.2: an implementation that cannot enforce Compatibility Mode treats the request as
    /// one for "core"). What it does not check is the hardware: it opens an adapter that lacks a flag of
    /// `DownlevelFlags::compliant()` with only a log message (wgpu-core `instance.rs`, `create_device_and_queue`). On
    /// its Metal backend on macOS every such flag holds (wgpu-hal `metal/adapter.rs`: each needs at most
    /// `macOS_GPUFamily1_v1`), so there the binding lists the feature itself, for the adapter and its devices; on any
    /// other backend or system the adapter is reported as the implementation lists it, as Compatibility-defaulting.
    pub const LISTS_CORE_FEATURES_AND_LIMITS: bool = !cfg!(webgpu_native = "wgpu-native");
}

/// The [`Api`] of this binding.
#[derive(Clone, Copy, Debug)]
pub struct Native;

/// Platform image and video sources exist only in a browser, so no value of this type exists.
#[derive(Clone, Copy, Debug)]
pub enum NoExternalSource {}

impl Api for Native {
    type Gpu = NativeGpu;
    type Adapter = NativeAdapter;
    type Device = NativeDevice;
    type Queue = NativeQueue;
    type Buffer = NativeBuffer;
    type MappedRange = NativeMappedRange;
    type Texture = NativeTexture;
    type TextureView = NativeTextureView;
    type ExternalTexture = NativeExternalTexture;
    type Sampler = NativeSampler;
    type BindGroupLayout = NativeBindGroupLayout;
    type BindGroup = NativeBindGroup;
    type PipelineLayout = NativePipelineLayout;
    type ShaderModule = NativeShaderModule;
    type ComputePipeline = NativeComputePipeline;
    type RenderPipeline = NativeRenderPipeline;
    type CommandBuffer = NativeCommandBuffer;
    type CommandEncoder = NativeCommandEncoder;
    type ComputePassEncoder = NativeComputePassEncoder;
    type RenderPassEncoder = NativeRenderPassEncoder;
    type RenderBundle = NativeRenderBundle;
    type RenderBundleEncoder = NativeRenderBundleEncoder;
    type QuerySet = NativeQuerySet;
    type ExternalImageSource = NoExternalSource;
    type ExternalVideoSource = NoExternalSource;
}

// ---------------------------------------------------------------------------------------------------------------------
// Instance and event processing
// ---------------------------------------------------------------------------------------------------------------------

/// Which ordering an asynchronous operation belongs to: a timed `wgpuInstanceWaitAny` may only wait on futures of one
/// source (webgpu-headers, "Mixed Sources").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Timeline {
    Cpu,
    /// The queue, by its handle's address.
    Queue(usize),
}

struct Outstanding {
    future: sys::WGPUFuture,
    timeline: Timeline,
    /// The device the operation runs on, for implementations that wait by polling a device.
    device: Option<Weak<DeviceShared>>,
    done: Rc<Cell<bool>>,
}

pub(crate) struct Instance {
    raw: sys::WGPUInstance,
    timed_wait: bool,
    outstanding: RefCell<Vec<Outstanding>>,
    devices: RefCell<Vec<Weak<DeviceShared>>>,
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { sys::wgpuInstanceRelease(self.raw) };
    }
}

impl Instance {
    /// Run callbacks of completed operations and deliver queued uncaptured errors.
    pub(crate) fn process_events(&self) {
        unsafe { sys::wgpuInstanceProcessEvents(self.raw) };
        self.deliver();
    }

    fn deliver(&self) {
        let devices: Vec<Rc<DeviceShared>> = {
            let mut d = self.devices.borrow_mut();
            d.retain(|w| w.strong_count() > 0);
            d.iter().filter_map(|w| w.upgrade()).collect()
        };
        for dev in devices {
            dev.deliver_uncaptured();
        }
    }

    fn pending(&self) -> usize {
        self.outstanding.borrow_mut().retain(|o| !o.done.get());
        self.outstanding.borrow().len()
    }

    /// Block until at least one outstanding operation may have completed. Returns false when nothing is outstanding.
    fn wait(&self) -> bool {
        if self.pending() == 0 {
            return false;
        }
        let pending: Vec<(sys::WGPUFuture, Timeline, Option<Weak<DeviceShared>>)> =
            self.outstanding.borrow().iter().map(|o| (o.future, o.timeline, o.device.clone())).collect();
        if quirks::WAIT_ANY && self.timed_wait {
            let cpu: Vec<sys::WGPUFuture> = pending.iter().filter(|p| p.1 == Timeline::Cpu).map(|p| p.0).collect();
            let first_queue = pending.iter().find_map(|p| match p.1 {
                Timeline::Queue(q) => Some(q),
                Timeline::Cpu => None,
            });
            let queue_group: Vec<sys::WGPUFuture> = match first_queue {
                Some(q) => pending.iter().filter(|p| p.1 == Timeline::Queue(q)).map(|p| p.0).collect(),
                None => Vec::new(),
            };
            // Futures of different sources cannot share a timed wait: alternate between them, waiting briefly on one
            // source while the other also has work outstanding.
            if !cpu.is_empty() {
                self.wait_any(&cpu, if queue_group.is_empty() { 100_000_000 } else { 1_000_000 });
            }
            if !queue_group.is_empty() {
                self.wait_any(&queue_group, if cpu.is_empty() { 100_000_000 } else { 1_000_000 });
            }
        } else {
            self.wait_without_wait_any(&pending);
        }
        self.process_events();
        true
    }

    #[cfg(webgpu_native = "wgpu-native")]
    fn wait_without_wait_any(&self, pending: &[(sys::WGPUFuture, Timeline, Option<Weak<DeviceShared>>)]) {
        // wgpu-native's wgpuDevicePoll(wait = true) blocks until the device's queue is idle, after which every
        // queue-timeline callback can run. Its CPU-timeline callbacks run inside wgpuInstanceProcessEvents.
        let mut polled = Vec::new();
        for (_, t, dev) in pending {
            if let (Timeline::Queue(q), Some(dev)) = (t, dev.as_ref().and_then(|d| d.upgrade()))
                && !polled.contains(q)
            {
                polled.push(*q);
                unsafe { sys::wgpuDevicePoll(dev.raw, 1, std::ptr::null()) };
            }
        }
        if polled.is_empty() {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[cfg(not(webgpu_native = "wgpu-native"))]
    fn wait_without_wait_any(&self, _pending: &[(sys::WGPUFuture, Timeline, Option<Weak<DeviceShared>>)]) {
        std::thread::sleep(Duration::from_millis(1));
    }

    fn wait_any(&self, futures: &[sys::WGPUFuture], timeout_ns: u64) {
        let mut infos: Vec<sys::WGPUFutureWaitInfo> =
            futures.iter().map(|f| sys::WGPUFutureWaitInfo { future: *f, completed: 0 }).collect();
        unsafe { sys::wgpuInstanceWaitAny(self.raw, infos.len(), infos.as_mut_ptr(), timeout_ns) };
    }
}

/// A one-shot result written by a callback and read by a future.
pub(crate) struct Slot<T> {
    value: RefCell<Option<T>>,
    waker: RefCell<Option<Waker>>,
}

impl<T> Slot<T> {
    pub(crate) fn new() -> Rc<Slot<T>> {
        Rc::new(Slot { value: RefCell::new(None), waker: RefCell::new(None) })
    }

    pub(crate) fn fill(&self, v: T) {
        *self.value.borrow_mut() = Some(v);
        if let Some(w) = self.waker.borrow_mut().take() {
            w.wake();
        }
    }
}

/// A future over a [`Slot`]; polling processes the instance's events.
pub(crate) struct SlotFuture<T> {
    pub(crate) inst: Rc<Instance>,
    pub(crate) slot: Rc<Slot<T>>,
}

impl<T> Future for SlotFuture<T> {
    type Output = T;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        if let Some(v) = self.slot.value.borrow_mut().take() {
            return Poll::Ready(v);
        }
        self.inst.process_events();
        if let Some(v) = self.slot.value.borrow_mut().take() {
            return Poll::Ready(v);
        }
        *self.slot.waker.borrow_mut() = Some(cx.waker().clone());
        Poll::Pending
    }
}

/// What a callback receives through `userdata1`: the slot to fill, the flag that tells the event loop its operation is
/// complete, and per-operation data.
pub(crate) struct CallbackBox<T, X> {
    pub(crate) slot: Rc<Slot<T>>,
    done: Rc<Cell<bool>>,
    pub(crate) extra: X,
}

/// Box the callback state; the flag is shared with the operation's entry in the instance's outstanding list, which
/// [`Instance::track`] creates once webgpu.h has returned the operation's `WGPUFuture`.
pub(crate) fn callback_box<T, X>(slot: &Rc<Slot<T>>, extra: X) -> (*mut c_void, Rc<Cell<bool>>) {
    let done = Rc::new(Cell::new(false));
    let b = Box::new(CallbackBox { slot: slot.clone(), done: done.clone(), extra });
    (Box::into_raw(b).cast(), done)
}

/// Take back a callback's state and mark its operation complete.
pub(crate) unsafe fn take_box<T, X>(p: *mut c_void) -> Box<CallbackBox<T, X>> {
    let b = unsafe { Box::from_raw(p.cast::<CallbackBox<T, X>>()) };
    b.done.set(true);
    b
}

/// Record an operation started through webgpu.h, sharing its completion flag with its callback state.
pub(crate) fn track(
    inst: &Instance,
    future: sys::WGPUFuture,
    timeline: Timeline,
    device: Option<&Rc<DeviceShared>>,
    done: Rc<Cell<bool>>,
) {
    inst.outstanding.borrow_mut().push(Outstanding { future, timeline, device: device.map(Rc::downgrade), done });
}

pub(crate) fn callback_mode() -> sys::WGPUCallbackMode {
    c_named("WGPUCallbackMode", "AllowProcessEvents") as sys::WGPUCallbackMode
}

pub(crate) fn status_is(status: u32, ty: &'static str, name: &str) -> bool {
    status == c_named(ty, name)
}

/// A deadline passed without any WebGPU operation completing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stalled {
    pub waited: Duration,
    /// Operations started through webgpu.h that had not completed.
    pub outstanding: usize,
}

impl fmt::Display for Stalled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no progress after {:?} with {} WebGPU operations outstanding", self.waited, self.outstanding)
    }
}

impl std::error::Error for Stalled {}

// ---------------------------------------------------------------------------------------------------------------------
// GPU
// ---------------------------------------------------------------------------------------------------------------------

/// `GPU` over webgpu.h: one `WGPUInstance`.
#[derive(Clone)]
pub struct NativeGpu {
    inst: Rc<Instance>,
}

impl fmt::Debug for NativeGpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NativeGpu({})", sys::meta::IMPLEMENTATION)
    }
}

impl NativeGpu {
    /// Create the instance, with timed waits enabled where the implementation supports them.
    pub fn new() -> NativeGpu {
        let mut timed_wait = false;
        let mut raw = std::ptr::null_mut();
        if quirks::WAIT_ANY {
            let feat = c_named("WGPUInstanceFeatureName", "TimedWaitAny") as sys::WGPUInstanceFeatureName;
            if unsafe { sys::wgpuHasInstanceFeature(feat) } != 0 {
                let req = [feat];
                let desc = sys::WGPUInstanceDescriptor {
                    nextInChain: std::ptr::null_mut(),
                    requiredFeatureCount: 1,
                    requiredFeatures: req.as_ptr(),
                    requiredLimits: std::ptr::null(),
                };
                raw = unsafe { sys::wgpuCreateInstance(&desc) };
                timed_wait = !raw.is_null();
            }
        }
        if raw.is_null() {
            raw = unsafe { sys::wgpuCreateInstance(std::ptr::null()) };
        }
        assert!(!raw.is_null(), "wgpuCreateInstance failed ({})", sys::meta::IMPLEMENTATION);
        NativeGpu {
            inst: Rc::new(Instance {
                raw,
                timed_wait,
                outstanding: RefCell::new(Vec::new()),
                devices: RefCell::new(Vec::new()),
            }),
        }
    }

    /// Run a future to completion on this thread, processing the instance's events while it waits. Fails when no
    /// WebGPU operation completes for `deadline`.
    pub fn block_on<F: Future>(&self, fut: F, deadline: Duration) -> Result<F::Output, Stalled> {
        let mut fut = std::pin::pin!(fut);
        let mut cx = Context::from_waker(Waker::noop());
        let mut last_progress = Instant::now();
        loop {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return Ok(v);
            }
            let before = self.inst.pending();
            if !self.inst.wait() {
                // Nothing started through webgpu.h is outstanding: the future waits on a callback that has no
                // WGPUFuture (device loss). Give the implementation time to deliver it.
                std::thread::sleep(Duration::from_millis(1));
                self.inst.process_events();
            }
            let after = self.inst.pending();
            if after < before {
                last_progress = Instant::now();
            }
            if last_progress.elapsed() > deadline {
                return Err(Stalled { waited: last_progress.elapsed(), outstanding: after });
            }
        }
    }

    /// Process completed operations' callbacks and deliver queued uncaptured errors, without blocking.
    pub fn process_events(&self) {
        self.inst.process_events();
    }
}

impl Default for NativeGpu {
    fn default() -> NativeGpu {
        NativeGpu::new()
    }
}

unsafe extern "C" fn on_adapter(
    status: sys::WGPURequestAdapterStatus,
    adapter: sys::WGPUAdapter,
    message: sys::WGPUStringView,
    u1: *mut c_void,
    _u2: *mut c_void,
) {
    let b = unsafe { take_box::<Result<Option<sys::WGPUAdapter>, Exception>, ()>(u1) };
    let r = if status_is(status, "WGPURequestAdapterStatus", "Success") {
        Ok(Some(adapter))
    } else if status_is(status, "WGPURequestAdapterStatus", "Unavailable") {
        // No adapter: requestAdapter() resolves with null.
        Ok(None)
    } else {
        Err(Exception::unsupported(format!(
            "wgpuInstanceRequestAdapter failed with status {status}: {}",
            from_sv(message)
        )))
    };
    b.slot.fill(r);
}

impl Gpu<Native> for NativeGpu {
    fn request_adapter(&self, options: &RequestAdapterOptions) -> GpuFuture<Result<Option<NativeAdapter>, Exception>> {
        // featureLevel is a DOMString: "core" and "compatibility" are feature level strings; any other value resolves
        // with null (requestAdapter, device timeline steps).
        let level = match options.feature_level.as_str() {
            "core" => c_named("WGPUFeatureLevel", "Core"),
            "compatibility" => c_named("WGPUFeatureLevel", "Compatibility"),
            _ => return Box::pin(async { Ok(None) }),
        };
        let power = match options.power_preference {
            Some(p) => match to_c(p) {
                Ok(v) => v,
                Err(e) => return Box::pin(async move { Err(e) }),
            },
            None => c_named("WGPUPowerPreference", "Undefined"),
        };
        let mut xr = sys::WGPURequestAdapterWebXROptions {
            chain: conv::chain("RequestAdapterWebXROptions", std::ptr::null_mut()),
            xrCompatible: conv::wbool(options.xr_compatible),
        };
        let o = sys::WGPURequestAdapterOptions {
            nextInChain: if options.xr_compatible { &mut xr.chain } else { std::ptr::null_mut() },
            featureLevel: level as sys::WGPUFeatureLevel,
            powerPreference: power as sys::WGPUPowerPreference,
            forceFallbackAdapter: conv::wbool(options.force_fallback_adapter),
            backendType: c_named("WGPUBackendType", "Undefined") as sys::WGPUBackendType,
            compatibleSurface: std::ptr::null_mut(),
        };
        let slot = Slot::<Result<Option<sys::WGPUAdapter>, Exception>>::new();
        let (bx, done) = callback_box(&slot, ());
        let info = sys::WGPURequestAdapterCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: callback_mode(),
            callback: Some(on_adapter),
            userdata1: bx,
            userdata2: std::ptr::null_mut(),
        };
        let fut = unsafe { sys::wgpuInstanceRequestAdapter(self.inst.raw, &o, info) };
        track(&self.inst, fut, Timeline::Cpu, None, done);
        let inst = self.inst.clone();
        Box::pin(async move {
            let r = SlotFuture { inst: inst.clone(), slot }.await?;
            Ok(r.map(|raw| NativeAdapter::wrap(inst, raw)))
        })
    }

    fn wgsl_language_features(&self) -> Result<BTreeSet<String>, Exception> {
        if !quirks::WGSL_LANGUAGE_FEATURES {
            return Err(Exception::unsupported(format!(
                "wgslLanguageFeatures: {} does not implement wgpuInstanceGetWGSLLanguageFeatures",
                sys::meta::IMPLEMENTATION
            )));
        }
        let mut f = sys::WGPUSupportedWGSLLanguageFeatures { featureCount: 0, features: std::ptr::null() };
        unsafe { sys::wgpuInstanceGetWGSLLanguageFeatures(self.inst.raw, &mut f) };
        let vals = unsafe { slice_or_empty(f.features, f.featureCount) }.to_vec();
        unsafe { sys::wgpuSupportedWGSLLanguageFeaturesFreeMembers(f) };
        Ok(vals.into_iter().filter_map(wgsl_feature_name).map(str::to_string).collect())
    }

    fn implementation(&self) -> String {
        sys::meta::IMPLEMENTATION.to_string()
    }
}

/// The WGSL language extension a `WGPUWGSLLanguageFeatureName` enumerator names, matched by name against the
/// extensions of the WGSL specification; implementation-specific enumerators name none.
fn wgsl_feature_name(v: sys::WGPUWGSLLanguageFeatureName) -> Option<&'static str> {
    let key = |s: &str| -> String {
        s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
    };
    let c = sys::meta::C_ENUMS.iter().find(|(t, _, val)| *t == "WGPUWGSLLanguageFeatureName" && *val == v)?;
    crate::gpu::tables::WGSL_LANGUAGE_EXTENSIONS.iter().copied().find(|n| key(n) == key(c.1))
}

pub(crate) unsafe fn slice_or_empty<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if p.is_null() || n == 0 { &[] } else { unsafe { std::slice::from_raw_parts(p, n) } }
}

// ---------------------------------------------------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------------------------------------------------

struct AdapterShared {
    inst: Rc<Instance>,
    raw: sys::WGPUAdapter,
    /// The adapter's `[[state]]` is "expired" (a device was created from it).
    expired: Cell<bool>,
    features: BTreeSet<String>,
    /// The binding lists "core-features-and-limits" for this adapter and its devices (see
    /// [`quirks::LISTS_CORE_FEATURES_AND_LIMITS`]).
    core_unlisted: bool,
    limits: SupportedLimits,
    info: AdapterInfo,
}

impl Drop for AdapterShared {
    fn drop(&mut self) {
        unsafe { sys::wgpuAdapterRelease(self.raw) };
    }
}

/// `GPUAdapter` over a `WGPUAdapter`.
#[derive(Clone)]
pub struct NativeAdapter(Rc<AdapterShared>);

impl fmt::Debug for NativeAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NativeAdapter({:?})", self.0.info.description)
    }
}

fn read_features(get: impl FnOnce(&mut sys::WGPUSupportedFeatures)) -> BTreeSet<String> {
    let mut f = sys::WGPUSupportedFeatures { featureCount: 0, features: std::ptr::null() };
    get(&mut f);
    let vals = unsafe { slice_or_empty(f.features, f.featureCount) }.to_vec();
    unsafe { sys::wgpuSupportedFeaturesFreeMembers(f) };
    vals.into_iter().filter_map(from_c::<FeatureName>).map(|n| n.as_str().to_string()).collect()
}

/// Read `WGPULimits` with the Compatibility Mode chain. A field the implementation leaves at its undefined sentinel is
/// not exposed.
fn read_limits(get: impl FnOnce(&mut sys::WGPULimits) -> sys::WGPUStatus) -> Result<SupportedLimits, Exception> {
    let mut compat = desc::undefined_compat_limits();
    let mut l = desc::undefined_limits();
    l.nextInChain = &mut compat.chain;
    let st = get(&mut l);
    if !status_is(st, "WGPUStatus", "Success") {
        return Err(Exception::unsupported("getting limits failed"));
    }
    let mut out = SupportedLimits::default();
    for lim in LIMITS.iter() {
        let ok = out.set(lim.name, desc::get_limit(&l, &compat, lim.name));
        debug_assert!(ok);
    }
    Ok(out)
}

/// The adapter's info and its `WGPUBackendType` (`None` if the implementation cannot report them).
fn read_info(raw: sys::WGPUAdapter) -> (AdapterInfo, Option<sys::WGPUBackendType>) {
    let mut i: sys::WGPUAdapterInfo = unsafe { std::mem::zeroed() };
    let st = unsafe { sys::wgpuAdapterGetInfo(raw, &mut i) };
    if !status_is(st, "WGPUStatus", "Success") {
        return (AdapterInfo::default(), None);
    }
    let info = AdapterInfo {
        vendor: Some(from_sv(i.vendor)),
        architecture: Some(from_sv(i.architecture)),
        device: Some(from_sv(i.device)),
        description: Some(from_sv(i.description)),
        subgroup_min_size: Some(i.subgroupMinSize),
        subgroup_max_size: Some(i.subgroupMaxSize),
        // webgpu.h has no counterpart of isFallbackAdapter.
        is_fallback_adapter: None,
    };
    let backend = i.backendType;
    unsafe { sys::wgpuAdapterInfoFreeMembers(i) };
    (info, Some(backend))
}

impl NativeAdapter {
    fn wrap(inst: Rc<Instance>, raw: sys::WGPUAdapter) -> NativeAdapter {
        let mut features = read_features(|f| unsafe { sys::wgpuAdapterGetFeatures(raw, f) });
        let limits = read_limits(|l| unsafe { sys::wgpuAdapterGetLimits(raw, l) }).unwrap_or_default();
        let (info, backend) = read_info(raw);
        let core = FeatureName::CoreFeaturesAndLimits.as_str();
        let core_unlisted = !quirks::LISTS_CORE_FEATURES_AND_LIMITS
            && cfg!(target_os = "macos")
            && backend == Some(c_named("WGPUBackendType", "Metal") as sys::WGPUBackendType)
            && !features.contains(core);
        if core_unlisted {
            features.insert(core.to_string());
        }
        NativeAdapter(Rc::new(AdapterShared {
            inst,
            raw,
            expired: Cell::new(false),
            features,
            core_unlisted,
            limits,
            info,
        }))
    }
}

unsafe extern "C" fn on_device(
    status: sys::WGPURequestDeviceStatus,
    device: sys::WGPUDevice,
    message: sys::WGPUStringView,
    u1: *mut c_void,
    _u2: *mut c_void,
) {
    let b = unsafe { take_box::<Result<sys::WGPUDevice, String>, ()>(u1) };
    let r = if status_is(status, "WGPURequestDeviceStatus", "Success") {
        Ok(device)
    } else {
        Err(format!("status {status}: {}", from_sv(message)))
    };
    b.slot.fill(r);
}

impl Adapter<Native> for NativeAdapter {
    fn features(&self) -> BTreeSet<String> {
        self.0.features.clone()
    }

    fn limits(&self) -> SupportedLimits {
        self.0.limits
    }

    fn info(&self) -> AdapterInfo {
        self.0.info.clone()
    }

    fn request_device(&self, descriptor: &DeviceDescriptor) -> GpuFuture<Result<NativeDevice, Exception>> {
        let a = self.0.clone();
        // requestDevice(), device timeline initialization steps.
        for f in &descriptor.required_features {
            if !a.features.contains(f.as_str()) {
                let e = Exception::new(
                    ExceptionKind::TypeError,
                    format!("the adapter does not support feature \"{}\"", f.as_str()),
                );
                return Box::pin(async move { Err(e) });
            }
        }
        for (key, value) in &descriptor.required_limits {
            let Some(value) = value else { continue };
            let Some(lim) = LIMITS.iter().find(|l| l.name == key) else {
                let e = Exception::new(ExceptionKind::OperationError, format!("\"{key}\" is not the name of a limit"));
                return Box::pin(async move { Err(e) });
            };
            let adapter = a.limits.get(key);
            let no_better = match (adapter, lim.class) {
                (Some(av), LimitClass::Maximum) => value.get() <= av,
                (Some(av), LimitClass::Alignment) => value.get() >= av,
                (None, _) => false,
            };
            let power_of_two =
                lim.class != LimitClass::Alignment || (value.get().is_power_of_two() && value.get() < 1 << 32);
            if !no_better || !power_of_two {
                let e = Exception::new(
                    ExceptionKind::OperationError,
                    format!(
                        "requiredLimits.{key} = {} is not a value the adapter can provide (adapter: {adapter:?})",
                        value.get()
                    ),
                );
                return Box::pin(async move { Err(e) });
            }
        }
        if a.expired.get() {
            let e = Exception::unsupported(
                "requestDevice() on an expired adapter resolves with a lost device, which the native binding cannot create",
            );
            return Box::pin(async move { Err(e) });
        }
        let mut arena = conv::Arena::default();
        let mut d = match desc::device_descriptor(&mut arena, descriptor) {
            Ok(d) => d,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let callbacks = Arc::new(DeviceCallbacks::default());
        d.deviceLostCallbackInfo = sys::WGPUDeviceLostCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: c_named("WGPUCallbackMode", "AllowSpontaneous") as sys::WGPUCallbackMode,
            callback: Some(on_device_lost),
            userdata1: Arc::into_raw(callbacks.clone()) as *mut c_void,
            userdata2: std::ptr::null_mut(),
        };
        d.uncapturedErrorCallbackInfo = sys::WGPUUncapturedErrorCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            callback: Some(on_uncaptured),
            userdata1: Arc::into_raw(callbacks.clone()) as *mut c_void,
            userdata2: std::ptr::null_mut(),
        };
        let slot = Slot::<Result<sys::WGPUDevice, String>>::new();
        let (bx, done) = callback_box(&slot, ());
        let info = sys::WGPURequestDeviceCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: callback_mode(),
            callback: Some(on_device),
            userdata1: bx,
            userdata2: std::ptr::null_mut(),
        };
        let fut = unsafe { sys::wgpuAdapterRequestDevice(a.raw, &d, info) };
        drop(arena);
        track(&a.inst, fut, Timeline::Cpu, None, done);
        let label = descriptor.label.clone();
        let queue_label = descriptor.default_queue.label.clone();
        Box::pin(async move {
            match (SlotFuture { inst: a.inst.clone(), slot }).await {
                Ok(raw) => {
                    a.expired.set(true);
                    Ok(NativeDevice::wrap(&a, raw, callbacks, label, queue_label))
                }
                // The callbacks' references are released by the device-lost callback, which webgpu.h fires with
                // reason FailedCreation.
                Err(msg) => Err(Exception::unsupported(format!(
                    "the implementation could not create the device ({msg}); WebGPU resolves with a lost device here, \
                     which the native binding cannot create"
                ))),
            }
        })
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------------------------------------------

/// What the device-lost and uncaptured-error callbacks write; they may run on any thread.
#[derive(Default)]
pub(crate) struct DeviceCallbacks {
    uncaptured: Mutex<VecDeque<GpuError>>,
    lost: Mutex<(Option<DeviceLostInfo>, Vec<Waker>)>,
}

unsafe extern "C" fn on_uncaptured(
    _device: *const sys::WGPUDevice,
    ty: sys::WGPUErrorType,
    message: sys::WGPUStringView,
    u1: *mut c_void,
    _u2: *mut c_void,
) {
    let cb = unsafe { &*(u1 as *const DeviceCallbacks) };
    let error = gpu_error(ty, from_sv(message));
    cb.uncaptured.lock().unwrap().push_back(error);
}

unsafe extern "C" fn on_device_lost(
    _device: *const sys::WGPUDevice,
    reason: sys::WGPUDeviceLostReason,
    message: sys::WGPUStringView,
    u1: *mut c_void,
    _u2: *mut c_void,
) {
    // userdata1 of this callback and of the uncaptured-error callback each own one reference. The uncaptured-error
    // callback never fires after this one (webgpu-headers, "Device Events"), so both are released here.
    let cb: Arc<DeviceCallbacks> = unsafe { Arc::from_raw(u1 as *const DeviceCallbacks) };
    unsafe { Arc::decrement_strong_count(u1 as *const DeviceCallbacks) };
    let reason = if status_is(reason, "WGPUDeviceLostReason", "Destroyed") {
        DeviceLostReason::Destroyed
    } else {
        DeviceLostReason::Unknown
    };
    let mut l = cb.lost.lock().unwrap();
    l.0 = Some(DeviceLostInfo { reason, message: from_sv(message) });
    for w in l.1.drain(..) {
        w.wake();
    }
}

/// A `GPUError` from a webgpu.h error type. `WGPUErrorType_Unknown` is not a WebGPU error type; it is reported as an
/// internal error whose message says so.
fn gpu_error(ty: sys::WGPUErrorType, message: String) -> GpuError {
    let filter = if status_is(ty, "WGPUErrorType", "Validation") {
        ErrorFilter::Validation
    } else if status_is(ty, "WGPUErrorType", "OutOfMemory") {
        ErrorFilter::OutOfMemory
    } else if status_is(ty, "WGPUErrorType", "Internal") {
        ErrorFilter::Internal
    } else {
        return GpuError {
            filter: ErrorFilter::Internal,
            message: format!("[webgpu.h error type {ty}, not a WebGPU error type] {message}"),
        };
    };
    GpuError { filter, message }
}

pub(crate) struct DeviceShared {
    pub(crate) inst: Rc<Instance>,
    pub(crate) raw: sys::WGPUDevice,
    label: RefCell<String>,
    features: BTreeSet<String>,
    feature_set: BTreeSet<FeatureName>,
    limits: SupportedLimits,
    adapter_info: AdapterInfo,
    callbacks: Arc<DeviceCallbacks>,
    handler: RefCell<Option<UncapturedErrorHandler>>,
    pub(crate) queue: RefCell<Option<NativeQueue>>,
    /// The device's error scope stack as this binding pushed it (all on this thread): true for a scope pushed to the
    /// implementation, false for one the binding keeps itself (see [`quirks::INTERNAL_ERROR_SCOPE`]).
    scope_stack: RefCell<Vec<bool>>,
    /// Every buffer of the device, for `destroy()`, which unmaps them all.
    pub(crate) buffers: RefCell<Vec<Weak<objects::BufferInner>>>,
}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        unsafe { sys::wgpuDeviceRelease(self.raw) };
    }
}

impl DeviceShared {
    fn deliver_uncaptured(&self) {
        loop {
            let next = self.callbacks.uncaptured.lock().unwrap().pop_front();
            let Some(e) = next else { break };
            // The handler is taken out while it runs, so it may replace itself.
            let h = self.handler.borrow_mut().take();
            if let Some(mut h) = h {
                h(e);
                let mut slot = self.handler.borrow_mut();
                if slot.is_none() {
                    *slot = Some(h);
                }
            }
        }
    }

    /// The feature is enabled for the device.
    pub(crate) fn has(&self, f: FeatureName) -> bool {
        self.feature_set.contains(&f)
    }
}

/// `GPUDevice` over a `WGPUDevice`.
#[derive(Clone)]
pub struct NativeDevice(pub(crate) Rc<DeviceShared>);

impl fmt::Debug for NativeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NativeDevice({:?})", self.0.label.borrow())
    }
}

impl NativeDevice {
    fn wrap(
        a: &AdapterShared,
        raw: sys::WGPUDevice,
        callbacks: Arc<DeviceCallbacks>,
        label: String,
        queue_label: String,
    ) -> NativeDevice {
        let mut features = read_features(|f| unsafe { sys::wgpuDeviceGetFeatures(raw, f) });
        // "a new device" adds the default features of the adapter's default feature level.
        if a.core_unlisted {
            features.insert(FeatureName::CoreFeaturesAndLimits.as_str().to_string());
        }
        let feature_set = features.iter().filter_map(|n| FeatureName::from_idl(n)).collect();
        let limits = read_limits(|l| unsafe { sys::wgpuDeviceGetLimits(raw, l) }).unwrap_or_default();
        let shared = Rc::new(DeviceShared {
            inst: a.inst.clone(),
            raw,
            label: RefCell::new(label),
            features,
            feature_set,
            limits,
            // GPUDevice.adapterInfo is the info of the adapter the device was created from.
            adapter_info: a.info.clone(),
            callbacks,
            handler: RefCell::new(None),
            queue: RefCell::new(None),
            scope_stack: RefCell::new(Vec::new()),
            buffers: RefCell::new(Vec::new()),
        });
        a.inst.devices.borrow_mut().push(Rc::downgrade(&shared));
        let q = unsafe { sys::wgpuDeviceGetQueue(raw) };
        *shared.queue.borrow_mut() = Some(NativeQueue::wrap(&shared, q, &queue_label));
        NativeDevice(shared)
    }
}

/// Forward a label to the implementation where it can take one after creation.
pub(crate) fn set_backend_label(f: impl FnOnce(sys::WGPUStringView), label: &str) {
    if quirks::SET_LABEL {
        f(sv(label));
    }
}

impl Object for NativeDevice {
    fn label(&self) -> String {
        self.0.label.borrow().clone()
    }
    fn set_label(&self, label: &str) {
        *self.0.label.borrow_mut() = label.to_string();
        set_backend_label(|l| unsafe { sys::wgpuDeviceSetLabel(self.0.raw, l) }, label);
    }
}

unsafe extern "C" fn on_pop(
    status: sys::WGPUPopErrorScopeStatus,
    ty: sys::WGPUErrorType,
    message: sys::WGPUStringView,
    u1: *mut c_void,
    _u2: *mut c_void,
) {
    let b = unsafe { take_box::<Result<Option<GpuError>, Exception>, ()>(u1) };
    let r = if status_is(status, "WGPUPopErrorScopeStatus", "Success") {
        if status_is(ty, "WGPUErrorType", "NoError") { Ok(None) } else { Ok(Some(gpu_error(ty, from_sv(message)))) }
    } else if status_is(status, "WGPUPopErrorScopeStatus", "Error") {
        // The error scope stack was empty.
        Err(Exception::new(ExceptionKind::OperationError, from_sv(message)))
    } else {
        Err(Exception::unsupported(format!("wgpuDevicePopErrorScope callback cancelled: {}", from_sv(message))))
    };
    b.slot.fill(r);
}

struct LostFuture(Arc<DeviceCallbacks>, Rc<Instance>);

impl Future for LostFuture {
    type Output = DeviceLostInfo;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<DeviceLostInfo> {
        {
            let mut l = self.0.lost.lock().unwrap();
            if let Some(info) = &l.0 {
                return Poll::Ready(info.clone());
            }
            l.1.push(cx.waker().clone());
        }
        self.1.process_events();
        match &self.0.lost.lock().unwrap().0 {
            Some(info) => Poll::Ready(info.clone()),
            None => Poll::Pending,
        }
    }
}

impl Device<Native> for NativeDevice {
    fn features(&self) -> BTreeSet<String> {
        self.0.features.clone()
    }

    fn limits(&self) -> SupportedLimits {
        self.0.limits
    }

    fn adapter_info(&self) -> AdapterInfo {
        self.0.adapter_info.clone()
    }

    fn queue(&self) -> NativeQueue {
        self.0.queue.borrow().clone().expect("device queue")
    }

    fn destroy(&self) {
        // Content timeline: unmap() every buffer of the device; then the device is lost with reason "destroyed".
        objects::unmap_all(&self.0);
        unsafe { sys::wgpuDeviceDestroy(self.0.raw) };
        if !quirks::LOST_ON_DESTROY {
            let mut l = self.0.callbacks.lost.lock().unwrap();
            if l.0.is_none() {
                l.0 = Some(DeviceLostInfo { reason: DeviceLostReason::Destroyed, message: String::new() });
                for w in l.1.drain(..) {
                    w.wake();
                }
            }
        }
    }

    fn create_buffer(&self, d: &BufferDescriptor) -> Result<NativeBuffer, Exception> {
        objects::create_buffer(self, d)
    }

    fn create_texture(&self, d: &TextureDescriptor) -> Result<NativeTexture, Exception> {
        objects::create_texture(self, d)
    }

    fn create_sampler(&self, d: &SamplerDescriptor) -> NativeSampler {
        objects::create_sampler(self, d)
    }

    fn import_external_texture(
        &self,
        d: &ExternalTextureDescriptor<'_, Native>,
    ) -> Result<NativeExternalTexture, Exception> {
        match *d.source {}
    }

    fn create_bind_group_layout(&self, d: &BindGroupLayoutDescriptor) -> Result<NativeBindGroupLayout, Exception> {
        objects::create_bind_group_layout(self, d)
    }

    fn create_pipeline_layout(&self, d: &PipelineLayoutDescriptor<'_, Native>) -> NativePipelineLayout {
        objects::create_pipeline_layout(self, d)
    }

    fn create_bind_group(&self, d: &BindGroupDescriptor<'_, Native>) -> NativeBindGroup {
        objects::create_bind_group(self, d)
    }

    fn create_shader_module(&self, d: &ShaderModuleDescriptor<'_, Native>) -> NativeShaderModule {
        objects::create_shader_module(self, d)
    }

    fn create_compute_pipeline(&self, d: &ComputePipelineDescriptor<'_, Native>) -> NativeComputePipeline {
        objects::create_compute_pipeline(self, d)
    }

    fn create_render_pipeline(
        &self,
        d: &RenderPipelineDescriptor<'_, Native>,
    ) -> Result<NativeRenderPipeline, Exception> {
        objects::create_render_pipeline(self, d)
    }

    fn create_compute_pipeline_async(
        &self,
        d: &ComputePipelineDescriptor<'_, Native>,
    ) -> GpuFuture<Result<NativeComputePipeline, Exception>> {
        objects::create_compute_pipeline_async(self, d)
    }

    fn create_render_pipeline_async(
        &self,
        d: &RenderPipelineDescriptor<'_, Native>,
    ) -> GpuFuture<Result<NativeRenderPipeline, Exception>> {
        objects::create_render_pipeline_async(self, d)
    }

    fn create_command_encoder(&self, d: &CommandEncoderDescriptor) -> NativeCommandEncoder {
        objects::create_command_encoder(self, d)
    }

    fn create_render_bundle_encoder(
        &self,
        d: &RenderBundleEncoderDescriptor,
    ) -> Result<NativeRenderBundleEncoder, Exception> {
        objects::create_render_bundle_encoder(self, d)
    }

    fn create_query_set(&self, d: &QuerySetDescriptor) -> Result<NativeQuerySet, Exception> {
        objects::create_query_set(self, d)
    }

    fn lost(&self) -> GpuFuture<DeviceLostInfo> {
        Box::pin(LostFuture(self.0.callbacks.clone(), self.0.inst.clone()))
    }

    fn push_error_scope(&self, filter: ErrorFilter) {
        if filter == ErrorFilter::Internal && !quirks::INTERNAL_ERROR_SCOPE {
            self.0.scope_stack.borrow_mut().push(false);
            return;
        }
        let f = to_c(filter).expect("webgpu.h defines every GPUErrorFilter");
        unsafe { sys::wgpuDevicePushErrorScope(self.0.raw, f as sys::WGPUErrorFilter) };
        self.0.scope_stack.borrow_mut().push(true);
    }

    fn pop_error_scope(&self) -> GpuFuture<Result<Option<GpuError>, Exception>> {
        let top = self.0.scope_stack.borrow_mut().pop();
        match top {
            Some(false) => return Box::pin(async { Ok(None) }),
            None if !quirks::POP_EMPTY_ERROR_SCOPE => {
                // popErrorScope(), device timeline check steps: a lost device resolves with null, an empty stack
                // rejects with OperationError.
                let lost = self.0.callbacks.lost.lock().unwrap().0.is_some();
                return Box::pin(async move {
                    if lost {
                        Ok(None)
                    } else {
                        Err(Exception::new(ExceptionKind::OperationError, "the error scope stack is empty"))
                    }
                });
            }
            _ => {}
        }
        let slot = Slot::<Result<Option<GpuError>, Exception>>::new();
        let (bx, done) = callback_box(&slot, ());
        let info = sys::WGPUPopErrorScopeCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: callback_mode(),
            callback: Some(on_pop),
            userdata1: bx,
            userdata2: std::ptr::null_mut(),
        };
        let fut = unsafe { sys::wgpuDevicePopErrorScope(self.0.raw, info) };
        track(&self.0.inst, fut, Timeline::Cpu, Some(&self.0), done);
        Box::pin(SlotFuture { inst: self.0.inst.clone(), slot })
    }

    fn set_onuncapturederror(&self, handler: Option<UncapturedErrorHandler>) {
        *self.0.handler.borrow_mut() = handler;
    }
}
