//! The binding over the browser's WebGPU, for wasm32 (feature `web`).
//!
//! Every call goes to the JavaScript object by its IDL name, with dictionaries built by the generated [`dict`]
//! conversions, so the binding reaches whatever the browser implements of the specification and is not limited to a
//! wrapper crate's API. Exceptions are caught and returned as [`Exception`] values. A member the browser does not
//! have is [`ExceptionKind::Unsupported`]. An exception from a method the specification never lets throw (the browser
//! rejects a value it does not support, for instance) has no place in that method's result; it aborts with a message
//! naming the call, as an unimplemented path must.

#[rustfmt::skip]
mod dict;

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt;
use std::rc::Rc;

use js_sys::{Array, ArrayBuffer, Function, Promise, Reflect, Uint8Array, Uint32Array};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::gpu::api::*;
use crate::gpu::idl::*;
use crate::gpu::num::{Float32, Size64};

/// The [`Api`] of this binding.
#[derive(Clone, Copy, Debug)]
pub struct Web;

fn get(obj: &JsValue, key: &str) -> JsValue {
    Reflect::get(obj, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}

fn set(obj: &JsValue, key: &str, value: &JsValue) {
    // Assigning an attribute of a WebGPU object or a property of a plain object does not throw for these values.
    let _ = Reflect::set(obj, &JsValue::from_str(key), value);
}

/// A JavaScript exception as an [`Exception`].
fn exception(e: JsValue) -> Exception {
    let name = get(&e, "name").as_string().unwrap_or_default();
    let message = get(&e, "message").as_string().unwrap_or_else(|| format!("{e:?}"));
    let kind = match name.as_str() {
        "TypeError" => ExceptionKind::TypeError,
        "RangeError" => ExceptionKind::RangeError,
        "OperationError" => ExceptionKind::OperationError,
        "AbortError" => ExceptionKind::AbortError,
        "SecurityError" => ExceptionKind::SecurityError,
        "InvalidStateError" => ExceptionKind::InvalidStateError,
        "GPUPipelineError" => match get(&e, "reason").as_string().as_deref().and_then(PipelineErrorReason::from_idl) {
            Some(r) => ExceptionKind::PipelineError(r),
            None => ExceptionKind::Other(name),
        },
        _ => ExceptionKind::Other(name),
    };
    Exception::new(kind, message)
}

/// `obj.method(...args)`. Trailing `undefined` arguments are left off, as an omitted optional argument.
fn call(obj: &JsValue, iface: &str, method: &str, args: &[JsValue]) -> Result<JsValue, Exception> {
    let f = get(obj, method);
    let Some(f) = f.dyn_ref::<Function>() else {
        return Err(Exception::unsupported(format!("the browser has no {iface}.{method}()")));
    };
    let mut n = args.len();
    while n > 0 && args[n - 1].is_undefined() {
        n -= 1;
    }
    let arr = Array::new();
    for a in &args[..n] {
        arr.push(a);
    }
    Reflect::apply(f, obj, &arr).map_err(exception)
}

/// A call to a method the specification never lets throw.
fn call_infallible(obj: &JsValue, iface: &str, method: &str, args: &[JsValue]) -> JsValue {
    match call(obj, iface, method, args) {
        Ok(v) => v,
        Err(e) => panic!("{iface}.{method}() threw {e}, which the WebGPU specification does not allow"),
    }
}

/// The settlement of a promise.
async fn settle(p: Result<JsValue, Exception>) -> Result<JsValue, Exception> {
    let p = p?;
    let p: Promise = match p.dyn_into() {
        Ok(p) => p,
        Err(v) => return Ok(v),
    };
    JsFuture::from(p).await.map_err(exception)
}

fn num(v: impl Into<f64>) -> JsValue {
    JsValue::from_f64(v.into())
}

fn size(v: Size64) -> JsValue {
    // Size64 holds at most 2^53 − 1, which a JavaScript number represents exactly.
    JsValue::from_f64(v.get() as f64)
}

fn opt_size(v: Option<Size64>) -> JsValue {
    v.map_or(JsValue::UNDEFINED, size)
}

fn float(v: Float32) -> JsValue {
    JsValue::from_f64(f64::from(v.get()))
}

fn bytes(b: &[u8]) -> JsValue {
    JsValue::from(Uint8Array::from(b))
}

fn string_set(setlike: &JsValue) -> BTreeSet<String> {
    if setlike.is_undefined() || setlike.is_null() {
        return BTreeSet::new();
    }
    Array::from(setlike).iter().filter_map(|v| v.as_string()).collect()
}

macro_rules! js_object {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone)]
        pub struct $name(JsValue);

        impl $name {
            /// The JavaScript object.
            pub fn js(&self) -> &JsValue {
                &self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({:?})"), get(&self.0, "label").as_string())
            }
        }

        impl Object for $name {
            fn label(&self) -> String {
                get(&self.0, "label").as_string().unwrap_or_default()
            }
            fn set_label(&self, label: &str) {
                set(&self.0, "label", &JsValue::from_str(label));
            }
        }
    };
}

js_object!(
    /// `GPUTextureView`.
    WebTextureView
);
js_object!(
    /// `GPUExternalTexture`.
    WebExternalTexture
);
js_object!(
    /// `GPUSampler`.
    WebSampler
);
js_object!(
    /// `GPUBindGroupLayout`.
    WebBindGroupLayout
);
js_object!(
    /// `GPUBindGroup`.
    WebBindGroup
);
js_object!(
    /// `GPUPipelineLayout`.
    WebPipelineLayout
);
js_object!(
    /// `GPUShaderModule`.
    WebShaderModule
);
js_object!(
    /// `GPUComputePipeline`.
    WebComputePipeline
);
js_object!(
    /// `GPURenderPipeline`.
    WebRenderPipeline
);
js_object!(
    /// `GPUCommandBuffer`.
    WebCommandBuffer
);
js_object!(
    /// `GPUCommandEncoder`.
    WebCommandEncoder
);
js_object!(
    /// `GPUComputePassEncoder`.
    WebComputePassEncoder
);
js_object!(
    /// `GPURenderPassEncoder`.
    WebRenderPassEncoder
);
js_object!(
    /// `GPURenderBundle`.
    WebRenderBundle
);
js_object!(
    /// `GPURenderBundleEncoder`.
    WebRenderBundleEncoder
);
js_object!(
    /// `GPUQuerySet`.
    WebQuerySet
);
js_object!(
    /// `GPUQueue`.
    WebQueue
);
js_object!(
    /// `GPUBuffer`.
    WebBuffer
);
js_object!(
    /// `GPUTexture`.
    WebTexture
);

/// A platform image source (`GPUCopyExternalImageSource`): the JavaScript object.
#[derive(Clone, Debug)]
pub struct WebImageSource(pub JsValue);

impl WebImageSource {
    pub fn js(&self) -> &JsValue {
        &self.0
    }
}

/// An `HTMLVideoElement` or `VideoFrame`: the JavaScript object.
#[derive(Clone, Debug)]
pub struct WebVideoSource(pub JsValue);

impl WebVideoSource {
    pub fn js(&self) -> &JsValue {
        &self.0
    }
}

impl Api for Web {
    type Gpu = WebGpu;
    type Adapter = WebAdapter;
    type Device = WebDevice;
    type Queue = WebQueue;
    type Buffer = WebBuffer;
    type MappedRange = WebMappedRange;
    type Texture = WebTexture;
    type TextureView = WebTextureView;
    type ExternalTexture = WebExternalTexture;
    type Sampler = WebSampler;
    type BindGroupLayout = WebBindGroupLayout;
    type BindGroup = WebBindGroup;
    type PipelineLayout = WebPipelineLayout;
    type ShaderModule = WebShaderModule;
    type ComputePipeline = WebComputePipeline;
    type RenderPipeline = WebRenderPipeline;
    type CommandBuffer = WebCommandBuffer;
    type CommandEncoder = WebCommandEncoder;
    type ComputePassEncoder = WebComputePassEncoder;
    type RenderPassEncoder = WebRenderPassEncoder;
    type RenderBundle = WebRenderBundle;
    type RenderBundleEncoder = WebRenderBundleEncoder;
    type QuerySet = WebQuerySet;
    type ExternalImageSource = WebImageSource;
    type ExternalVideoSource = WebVideoSource;
}

// ---------------------------------------------------------------------------------------------------------------------
// GPU, adapter
// ---------------------------------------------------------------------------------------------------------------------

/// `GPU`: `navigator.gpu` of the window or worker.
#[derive(Clone)]
pub struct WebGpu(JsValue);

impl fmt::Debug for WebGpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WebGpu")
    }
}

impl WebGpu {
    /// `navigator.gpu` of the global object (a window or a worker).
    pub fn from_global() -> Result<WebGpu, Exception> {
        let nav = get(&js_sys::global(), "navigator");
        let gpu = get(&nav, "gpu");
        if gpu.is_undefined() || gpu.is_null() {
            return Err(Exception::unsupported("navigator.gpu is not available in this context"));
        }
        Ok(WebGpu(gpu))
    }

    pub fn js(&self) -> &JsValue {
        &self.0
    }
}

fn read_limits(limits: &JsValue) -> SupportedLimits {
    let mut out = SupportedLimits::default();
    for name in SUPPORTED_LIMITS {
        let v = get(limits, name).as_f64().map(|f| f as u64);
        let ok = out.set(name, v);
        debug_assert!(ok);
    }
    out
}

fn read_info(info: &JsValue) -> AdapterInfo {
    let s = |k: &str| get(info, k).as_string();
    let n = |k: &str| get(info, k).as_f64().map(|f| f as u32);
    AdapterInfo {
        vendor: s("vendor"),
        architecture: s("architecture"),
        device: s("device"),
        description: s("description"),
        subgroup_min_size: n("subgroupMinSize"),
        subgroup_max_size: n("subgroupMaxSize"),
        is_fallback_adapter: get(info, "isFallbackAdapter").as_bool(),
    }
}

impl Gpu<Web> for WebGpu {
    fn request_adapter(&self, options: &RequestAdapterOptions) -> GpuFuture<Result<Option<WebAdapter>, Exception>> {
        let p = call(&self.0, "GPU", "requestAdapter", &[dict::dict_request_adapter_options(options).into()]);
        Box::pin(async move {
            let a = settle(p).await?;
            Ok(if a.is_null() || a.is_undefined() { None } else { Some(WebAdapter(a)) })
        })
    }

    fn wgsl_language_features(&self) -> Result<BTreeSet<String>, Exception> {
        let f = get(&self.0, "wgslLanguageFeatures");
        if f.is_undefined() {
            return Err(Exception::unsupported("the browser has no GPU.wgslLanguageFeatures"));
        }
        Ok(string_set(&f))
    }

    fn implementation(&self) -> String {
        get(&get(&js_sys::global(), "navigator"), "userAgent").as_string().unwrap_or_default()
    }
}

/// `GPUAdapter`.
#[derive(Clone)]
pub struct WebAdapter(JsValue);

impl fmt::Debug for WebAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WebAdapter")
    }
}

impl Adapter<Web> for WebAdapter {
    fn features(&self) -> BTreeSet<String> {
        string_set(&get(&self.0, "features"))
    }

    fn limits(&self) -> SupportedLimits {
        read_limits(&get(&self.0, "limits"))
    }

    fn info(&self) -> AdapterInfo {
        read_info(&get(&self.0, "info"))
    }

    fn request_device(&self, descriptor: &DeviceDescriptor) -> GpuFuture<Result<WebDevice, Exception>> {
        let p = call(&self.0, "GPUAdapter", "requestDevice", &[dict::dict_device_descriptor(descriptor).into()]);
        Box::pin(async move {
            let d = settle(p).await?;
            Ok(WebDevice { js: d, handler: Rc::new(RefCell::new(None)) })
        })
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------------------------------------------

/// The closure installed as `onuncapturederror`.
type HandlerSlot = Rc<RefCell<Option<Closure<dyn FnMut(JsValue)>>>>;

/// `GPUDevice`. Holds the closure installed as `onuncapturederror`, which must live as long as it is installed.
#[derive(Clone)]
pub struct WebDevice {
    js: JsValue,
    handler: HandlerSlot,
}

impl fmt::Debug for WebDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WebDevice({:?})", get(&self.js, "label").as_string())
    }
}

impl WebDevice {
    pub fn js(&self) -> &JsValue {
        &self.js
    }
}

impl Object for WebDevice {
    fn label(&self) -> String {
        get(&self.js, "label").as_string().unwrap_or_default()
    }
    fn set_label(&self, label: &str) {
        set(&self.js, "label", &JsValue::from_str(label));
    }
}

/// A `GPUError` object as a [`GpuError`], by its interface. A subtype the specification does not define yet is
/// reported as an internal error whose message names it.
fn gpu_error(e: &JsValue) -> GpuError {
    let message = get(e, "message").as_string().unwrap_or_default();
    let ctor = get(&get(e, "constructor"), "name").as_string().unwrap_or_default();
    let filter = match ctor.as_str() {
        "GPUValidationError" => ErrorFilter::Validation,
        "GPUOutOfMemoryError" => ErrorFilter::OutOfMemory,
        "GPUInternalError" => ErrorFilter::Internal,
        _ => {
            return GpuError {
                filter: ErrorFilter::Internal,
                message: format!("[{ctor}, not a WebGPU error type] {message}"),
            };
        }
    };
    GpuError { filter, message }
}

macro_rules! fallible_create {
    ($self:ident, $iface:literal, $method:literal, $arg:expr, $wrap:ident) => {
        call(&$self.js, $iface, $method, &[$arg.into()]).map($wrap)
    };
}

impl Device<Web> for WebDevice {
    fn features(&self) -> BTreeSet<String> {
        string_set(&get(&self.js, "features"))
    }

    fn limits(&self) -> SupportedLimits {
        read_limits(&get(&self.js, "limits"))
    }

    fn adapter_info(&self) -> AdapterInfo {
        read_info(&get(&self.js, "adapterInfo"))
    }

    fn queue(&self) -> WebQueue {
        WebQueue(get(&self.js, "queue"))
    }

    fn destroy(&self) {
        call_infallible(&self.js, "GPUDevice", "destroy", &[]);
    }

    fn create_buffer(&self, d: &BufferDescriptor) -> Result<WebBuffer, Exception> {
        fallible_create!(self, "GPUDevice", "createBuffer", dict::dict_buffer_descriptor(d), WebBuffer)
    }

    fn create_texture(&self, d: &TextureDescriptor) -> Result<WebTexture, Exception> {
        fallible_create!(self, "GPUDevice", "createTexture", dict::dict_texture_descriptor(d), WebTexture)
    }

    fn create_sampler(&self, d: &SamplerDescriptor) -> WebSampler {
        WebSampler(call_infallible(&self.js, "GPUDevice", "createSampler", &[dict::dict_sampler_descriptor(d).into()]))
    }

    fn import_external_texture(&self, d: &ExternalTextureDescriptor<'_, Web>) -> Result<WebExternalTexture, Exception> {
        fallible_create!(
            self,
            "GPUDevice",
            "importExternalTexture",
            dict::dict_external_texture_descriptor(d),
            WebExternalTexture
        )
    }

    fn create_bind_group_layout(&self, d: &BindGroupLayoutDescriptor) -> Result<WebBindGroupLayout, Exception> {
        fallible_create!(
            self,
            "GPUDevice",
            "createBindGroupLayout",
            dict::dict_bind_group_layout_descriptor(d),
            WebBindGroupLayout
        )
    }

    fn create_pipeline_layout(&self, d: &PipelineLayoutDescriptor<'_, Web>) -> WebPipelineLayout {
        WebPipelineLayout(call_infallible(
            &self.js,
            "GPUDevice",
            "createPipelineLayout",
            &[dict::dict_pipeline_layout_descriptor(d).into()],
        ))
    }

    fn create_bind_group(&self, d: &BindGroupDescriptor<'_, Web>) -> WebBindGroup {
        WebBindGroup(call_infallible(
            &self.js,
            "GPUDevice",
            "createBindGroup",
            &[dict::dict_bind_group_descriptor(d).into()],
        ))
    }

    fn create_shader_module(&self, d: &ShaderModuleDescriptor<'_, Web>) -> WebShaderModule {
        WebShaderModule(call_infallible(
            &self.js,
            "GPUDevice",
            "createShaderModule",
            &[dict::dict_shader_module_descriptor(d).into()],
        ))
    }

    fn create_compute_pipeline(&self, d: &ComputePipelineDescriptor<'_, Web>) -> WebComputePipeline {
        WebComputePipeline(call_infallible(
            &self.js,
            "GPUDevice",
            "createComputePipeline",
            &[dict::dict_compute_pipeline_descriptor(d).into()],
        ))
    }

    fn create_render_pipeline(&self, d: &RenderPipelineDescriptor<'_, Web>) -> Result<WebRenderPipeline, Exception> {
        fallible_create!(
            self,
            "GPUDevice",
            "createRenderPipeline",
            dict::dict_render_pipeline_descriptor(d),
            WebRenderPipeline
        )
    }

    fn create_compute_pipeline_async(
        &self,
        d: &ComputePipelineDescriptor<'_, Web>,
    ) -> GpuFuture<Result<WebComputePipeline, Exception>> {
        let p = call(
            &self.js,
            "GPUDevice",
            "createComputePipelineAsync",
            &[dict::dict_compute_pipeline_descriptor(d).into()],
        );
        Box::pin(async move { settle(p).await.map(WebComputePipeline) })
    }

    fn create_render_pipeline_async(
        &self,
        d: &RenderPipelineDescriptor<'_, Web>,
    ) -> GpuFuture<Result<WebRenderPipeline, Exception>> {
        let p = call(
            &self.js,
            "GPUDevice",
            "createRenderPipelineAsync",
            &[dict::dict_render_pipeline_descriptor(d).into()],
        );
        Box::pin(async move { settle(p).await.map(WebRenderPipeline) })
    }

    fn create_command_encoder(&self, d: &CommandEncoderDescriptor) -> WebCommandEncoder {
        WebCommandEncoder(call_infallible(
            &self.js,
            "GPUDevice",
            "createCommandEncoder",
            &[dict::dict_command_encoder_descriptor(d).into()],
        ))
    }

    fn create_render_bundle_encoder(
        &self,
        d: &RenderBundleEncoderDescriptor,
    ) -> Result<WebRenderBundleEncoder, Exception> {
        fallible_create!(
            self,
            "GPUDevice",
            "createRenderBundleEncoder",
            dict::dict_render_bundle_encoder_descriptor(d),
            WebRenderBundleEncoder
        )
    }

    fn create_query_set(&self, d: &QuerySetDescriptor) -> Result<WebQuerySet, Exception> {
        fallible_create!(self, "GPUDevice", "createQuerySet", dict::dict_query_set_descriptor(d), WebQuerySet)
    }

    fn lost(&self) -> GpuFuture<DeviceLostInfo> {
        let p = get(&self.js, "lost");
        Box::pin(async move {
            // The lost promise never rejects.
            let info = settle(Ok(p)).await.unwrap_or(JsValue::UNDEFINED);
            let reason = get(&info, "reason").as_string().as_deref().and_then(DeviceLostReason::from_idl);
            DeviceLostInfo {
                reason: reason.unwrap_or(DeviceLostReason::Unknown),
                message: get(&info, "message").as_string().unwrap_or_default(),
            }
        })
    }

    fn push_error_scope(&self, filter: ErrorFilter) {
        call_infallible(&self.js, "GPUDevice", "pushErrorScope", &[JsValue::from_str(filter.as_str())]);
    }

    fn pop_error_scope(&self) -> GpuFuture<Result<Option<GpuError>, Exception>> {
        let p = call(&self.js, "GPUDevice", "popErrorScope", &[]);
        Box::pin(async move {
            let e = settle(p).await?;
            Ok(if e.is_null() || e.is_undefined() { None } else { Some(gpu_error(&e)) })
        })
    }

    fn set_onuncapturederror(&self, handler: Option<UncapturedErrorHandler>) {
        match handler {
            Some(mut h) => {
                let c = Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| h(gpu_error(&get(&event, "error"))));
                set(&self.js, "onuncapturederror", c.as_ref());
                *self.handler.borrow_mut() = Some(c);
            }
            None => {
                set(&self.js, "onuncapturederror", &JsValue::NULL);
                *self.handler.borrow_mut() = None;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Buffers and mapped ranges
// ---------------------------------------------------------------------------------------------------------------------

impl Buffer<Web> for WebBuffer {
    fn size(&self) -> u64 {
        get(&self.0, "size").as_f64().unwrap_or(0.0) as u64
    }

    fn usage(&self) -> u32 {
        get(&self.0, "usage").as_f64().unwrap_or(0.0) as u32
    }

    fn map_state(&self) -> BufferMapState {
        let s = get(&self.0, "mapState").as_string().unwrap_or_default();
        BufferMapState::from_idl(&s).unwrap_or_else(|| panic!("GPUBuffer.mapState is {s:?}, not a GPUBufferMapState"))
    }

    fn map_async(&self, mode: u32, offset: Size64, sz: Option<Size64>) -> GpuFuture<Result<(), Exception>> {
        let p = call(&self.0, "GPUBuffer", "mapAsync", &[num(mode), size(offset), opt_size(sz)]);
        Box::pin(async move { settle(p).await.map(|_| ()) })
    }

    fn get_mapped_range(&self, offset: Size64, sz: Option<Size64>) -> Result<WebMappedRange, Exception> {
        let ab = call(&self.0, "GPUBuffer", "getMappedRange", &[size(offset), opt_size(sz)])?;
        Ok(WebMappedRange(ab.unchecked_into()))
    }

    fn unmap(&self) {
        call_infallible(&self.0, "GPUBuffer", "unmap", &[]);
    }

    fn destroy(&self) {
        call_infallible(&self.0, "GPUBuffer", "destroy", &[]);
    }
}

/// The `ArrayBuffer` of a mapped range.
#[derive(Clone)]
pub struct WebMappedRange(ArrayBuffer);

impl fmt::Debug for WebMappedRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WebMappedRange({} bytes)", self.byte_length())
    }
}

impl WebMappedRange {
    /// `new Uint8Array(buffer, offset, n)`, with the exceptions it throws.
    fn view(&self, offset: u64, n: usize) -> Result<Uint8Array, Exception> {
        let ctor: Function = get(&js_sys::global(), "Uint8Array").unchecked_into();
        let args = Array::of3(&self.0, &JsValue::from_f64(offset as f64), &JsValue::from_f64(n as f64));
        Reflect::construct(&ctor, &args).map(|v| v.unchecked_into()).map_err(exception)
    }
}

impl MappedRange for WebMappedRange {
    fn byte_length(&self) -> u64 {
        get(&self.0, "byteLength").as_f64().unwrap_or(0.0) as u64
    }

    fn read(&self, offset: u64, dst: &mut [u8]) -> Result<(), Exception> {
        self.view(offset, dst.len())?.copy_to(dst);
        Ok(())
    }

    fn write(&self, offset: u64, src: &[u8]) -> Result<(), Exception> {
        self.view(offset, src.len())?.copy_from(src);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Textures, shaders, pipelines, query sets
// ---------------------------------------------------------------------------------------------------------------------

fn attr_u32(obj: &JsValue, key: &str) -> u32 {
    get(obj, key).as_f64().unwrap_or(0.0) as u32
}

impl Texture<Web> for WebTexture {
    fn create_view(&self, d: &TextureViewDescriptor) -> Result<WebTextureView, Exception> {
        call(&self.0, "GPUTexture", "createView", &[dict::dict_texture_view_descriptor(d).into()]).map(WebTextureView)
    }

    fn destroy(&self) {
        call_infallible(&self.0, "GPUTexture", "destroy", &[]);
    }

    fn width(&self) -> u32 {
        attr_u32(&self.0, "width")
    }
    fn height(&self) -> u32 {
        attr_u32(&self.0, "height")
    }
    fn depth_or_array_layers(&self) -> u32 {
        attr_u32(&self.0, "depthOrArrayLayers")
    }
    fn mip_level_count(&self) -> u32 {
        attr_u32(&self.0, "mipLevelCount")
    }
    fn sample_count(&self) -> u32 {
        attr_u32(&self.0, "sampleCount")
    }
    fn dimension(&self) -> TextureDimension {
        let s = get(&self.0, "dimension").as_string().unwrap_or_default();
        TextureDimension::from_idl(&s).unwrap_or_else(|| panic!("GPUTexture.dimension is {s:?}"))
    }
    fn format(&self) -> TextureFormat {
        let s = get(&self.0, "format").as_string().unwrap_or_default();
        TextureFormat::from_idl(&s).unwrap_or_else(|| panic!("GPUTexture.format is {s:?}, not a format of this IDL"))
    }
    fn usage(&self) -> u32 {
        attr_u32(&self.0, "usage")
    }
    fn texture_binding_view_dimension(&self) -> Option<TextureViewDimension> {
        get(&self.0, "textureBindingViewDimension").as_string().as_deref().and_then(TextureViewDimension::from_idl)
    }
}

impl ShaderModule<Web> for WebShaderModule {
    fn get_compilation_info(&self) -> GpuFuture<Result<CompilationInfo, Exception>> {
        let p = call(&self.0, "GPUShaderModule", "getCompilationInfo", &[]);
        Box::pin(async move {
            let info = settle(p).await?;
            let mut messages = Vec::new();
            for m in Array::from(&get(&info, "messages")).iter() {
                let ty = get(&m, "type").as_string().unwrap_or_default();
                let n = |k: &str| get(&m, k).as_f64().unwrap_or(0.0) as u64;
                messages.push(CompilationMessage {
                    message: get(&m, "message").as_string().unwrap_or_default(),
                    r#type: CompilationMessageType::from_idl(&ty)
                        .ok_or_else(|| Exception::unsupported(format!("compilation message type {ty:?}")))?,
                    line_num: n("lineNum"),
                    line_pos: n("linePos"),
                    offset: n("offset"),
                    length: n("length"),
                });
            }
            Ok(CompilationInfo { messages })
        })
    }
}

impl PipelineBase<Web> for WebComputePipeline {
    fn get_bind_group_layout(&self, index: u32) -> WebBindGroupLayout {
        WebBindGroupLayout(call_infallible(&self.0, "GPUComputePipeline", "getBindGroupLayout", &[num(index)]))
    }
}

impl PipelineBase<Web> for WebRenderPipeline {
    fn get_bind_group_layout(&self, index: u32) -> WebBindGroupLayout {
        WebBindGroupLayout(call_infallible(&self.0, "GPURenderPipeline", "getBindGroupLayout", &[num(index)]))
    }
}

impl QuerySet<Web> for WebQuerySet {
    fn destroy(&self) {
        call_infallible(&self.0, "GPUQuerySet", "destroy", &[]);
    }
    fn r#type(&self) -> QueryType {
        let s = get(&self.0, "type").as_string().unwrap_or_default();
        QueryType::from_idl(&s).unwrap_or_else(|| panic!("GPUQuerySet.type is {s:?}"))
    }
    fn count(&self) -> u32 {
        attr_u32(&self.0, "count")
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------------------------------------------------

impl Queue<Web> for WebQueue {
    fn submit(&self, command_buffers: &[&WebCommandBuffer]) {
        let arr = Array::new();
        for c in command_buffers {
            arr.push(c.js());
        }
        call_infallible(&self.0, "GPUQueue", "submit", &[arr.into()]);
    }

    fn on_submitted_work_done(&self) -> GpuFuture<()> {
        let p = call(&self.0, "GPUQueue", "onSubmittedWorkDone", &[]);
        Box::pin(async move {
            if let Err(e) = settle(p).await {
                panic!(
                    "GPUQueue.onSubmittedWorkDone() rejected with {e}, which the WebGPU specification does not allow"
                );
            }
        })
    }

    fn write_buffer(
        &self,
        buffer: &WebBuffer,
        buffer_offset: Size64,
        data: &[u8],
        data_offset: Size64,
        sz: Option<Size64>,
    ) -> Result<(), Exception> {
        call(
            &self.0,
            "GPUQueue",
            "writeBuffer",
            &[buffer.0.clone(), size(buffer_offset), bytes(data), size(data_offset), opt_size(sz)],
        )
        .map(|_| ())
    }

    fn write_texture(
        &self,
        destination: &TexelCopyTextureInfo<'_, Web>,
        data: &[u8],
        data_layout: &TexelCopyBufferLayout,
        sz: &Extent3D,
    ) -> Result<(), Exception> {
        call(
            &self.0,
            "GPUQueue",
            "writeTexture",
            &[
                dict::dict_texel_copy_texture_info(destination).into(),
                bytes(data),
                dict::dict_texel_copy_buffer_layout(data_layout).into(),
                dict::union_extent_3d(sz),
            ],
        )
        .map(|_| ())
    }

    fn copy_external_image_to_texture(
        &self,
        source: &CopyExternalImageSourceInfo<'_, Web>,
        destination: &CopyExternalImageDestInfo<'_, Web>,
        copy_size: &Extent3D,
    ) -> Result<(), Exception> {
        call(
            &self.0,
            "GPUQueue",
            "copyExternalImageToTexture",
            &[
                dict::dict_copy_external_image_source_info(source).into(),
                dict::dict_copy_external_image_dest_info(destination).into(),
                dict::union_extent_3d(copy_size),
            ],
        )
        .map(|_| ())
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------------------------------------------------

macro_rules! debug_commands {
    ($ty:ident, $iface:literal) => {
        impl DebugCommands for $ty {
            fn push_debug_group(&self, l: &str) {
                call_infallible(&self.0, $iface, "pushDebugGroup", &[JsValue::from_str(l)]);
            }
            fn pop_debug_group(&self) {
                call_infallible(&self.0, $iface, "popDebugGroup", &[]);
            }
            fn insert_debug_marker(&self, l: &str) {
                call_infallible(&self.0, $iface, "insertDebugMarker", &[JsValue::from_str(l)]);
            }
        }
    };
}

debug_commands!(WebCommandEncoder, "GPUCommandEncoder");
debug_commands!(WebComputePassEncoder, "GPUComputePassEncoder");
debug_commands!(WebRenderPassEncoder, "GPURenderPassEncoder");
debug_commands!(WebRenderBundleEncoder, "GPURenderBundleEncoder");

fn bind_group_or_null(bg: Option<&WebBindGroup>) -> JsValue {
    bg.map_or(JsValue::NULL, |b| b.0.clone())
}

macro_rules! binding_commands {
    ($ty:ident, $iface:literal) => {
        impl BindingCommands<Web> for $ty {
            fn set_bind_group(&self, index: u32, bg: Option<&WebBindGroup>, offsets: &[u32]) {
                let arr = Array::new();
                for o in offsets {
                    arr.push(&num(*o));
                }
                call_infallible(&self.0, $iface, "setBindGroup", &[num(index), bind_group_or_null(bg), arr.into()]);
            }
            fn set_bind_group_with_data(
                &self,
                index: u32,
                bg: Option<&WebBindGroup>,
                data: &[u32],
                start: Size64,
                length: u32,
            ) -> Result<(), Exception> {
                let d = Uint32Array::from(data);
                call(
                    &self.0,
                    $iface,
                    "setBindGroup",
                    &[num(index), bind_group_or_null(bg), d.into(), size(start), num(length)],
                )
                .map(|_| ())
            }
            fn set_immediates(
                &self,
                range_offset: u32,
                data: &[u8],
                data_offset: Size64,
                data_size: Option<Size64>,
            ) -> Result<(), Exception> {
                call(
                    &self.0,
                    $iface,
                    "setImmediates",
                    &[num(range_offset), bytes(data), size(data_offset), opt_size(data_size)],
                )
                .map(|_| ())
            }
        }
    };
}

binding_commands!(WebComputePassEncoder, "GPUComputePassEncoder");
binding_commands!(WebRenderPassEncoder, "GPURenderPassEncoder");
binding_commands!(WebRenderBundleEncoder, "GPURenderBundleEncoder");

macro_rules! render_commands {
    ($ty:ident, $iface:literal) => {
        impl RenderCommands<Web> for $ty {
            fn set_pipeline(&self, p: &WebRenderPipeline) {
                call_infallible(&self.0, $iface, "setPipeline", std::slice::from_ref(&p.0));
            }
            fn set_index_buffer(&self, b: &WebBuffer, f: IndexFormat, offset: Size64, sz: Option<Size64>) {
                call_infallible(
                    &self.0,
                    $iface,
                    "setIndexBuffer",
                    &[b.0.clone(), JsValue::from_str(f.as_str()), size(offset), opt_size(sz)],
                );
            }
            fn set_vertex_buffer(&self, slot: u32, b: Option<&WebBuffer>, offset: Size64, sz: Option<Size64>) {
                let b = b.map_or(JsValue::NULL, |b| b.0.clone());
                call_infallible(&self.0, $iface, "setVertexBuffer", &[num(slot), b, size(offset), opt_size(sz)]);
            }
            fn draw(&self, vc: u32, ic: u32, fv: u32, fi: u32) {
                call_infallible(&self.0, $iface, "draw", &[num(vc), num(ic), num(fv), num(fi)]);
            }
            fn draw_indexed(&self, ic: u32, inst: u32, fi: u32, bv: i32, finst: u32) {
                call_infallible(&self.0, $iface, "drawIndexed", &[num(ic), num(inst), num(fi), num(bv), num(finst)]);
            }
            fn draw_indirect(&self, b: &WebBuffer, off: Size64) {
                call_infallible(&self.0, $iface, "drawIndirect", &[b.0.clone(), size(off)]);
            }
            fn draw_indexed_indirect(&self, b: &WebBuffer, off: Size64) {
                call_infallible(&self.0, $iface, "drawIndexedIndirect", &[b.0.clone(), size(off)]);
            }
        }
    };
}

render_commands!(WebRenderPassEncoder, "GPURenderPassEncoder");
render_commands!(WebRenderBundleEncoder, "GPURenderBundleEncoder");

impl CommandEncoder<Web> for WebCommandEncoder {
    fn begin_render_pass(&self, d: &RenderPassDescriptor<'_, Web>) -> Result<WebRenderPassEncoder, Exception> {
        call(&self.0, "GPUCommandEncoder", "beginRenderPass", &[dict::dict_render_pass_descriptor(d).into()])
            .map(WebRenderPassEncoder)
    }

    fn begin_compute_pass(&self, d: &ComputePassDescriptor<'_, Web>) -> Result<WebComputePassEncoder, Exception> {
        call(&self.0, "GPUCommandEncoder", "beginComputePass", &[dict::dict_compute_pass_descriptor(d).into()])
            .map(WebComputePassEncoder)
    }

    fn copy_buffer_to_buffer(&self, source: &WebBuffer, destination: &WebBuffer, sz: Option<Size64>) {
        call_infallible(
            &self.0,
            "GPUCommandEncoder",
            "copyBufferToBuffer",
            &[source.0.clone(), destination.0.clone(), opt_size(sz)],
        );
    }

    fn copy_buffer_to_buffer_with_offsets(
        &self,
        source: &WebBuffer,
        source_offset: Size64,
        destination: &WebBuffer,
        destination_offset: Size64,
        sz: Option<Size64>,
    ) {
        call_infallible(
            &self.0,
            "GPUCommandEncoder",
            "copyBufferToBuffer",
            &[source.0.clone(), size(source_offset), destination.0.clone(), size(destination_offset), opt_size(sz)],
        );
    }

    fn copy_buffer_to_texture(
        &self,
        source: &TexelCopyBufferInfo<'_, Web>,
        destination: &TexelCopyTextureInfo<'_, Web>,
        copy_size: &Extent3D,
    ) -> Result<(), Exception> {
        call(
            &self.0,
            "GPUCommandEncoder",
            "copyBufferToTexture",
            &[
                dict::dict_texel_copy_buffer_info(source).into(),
                dict::dict_texel_copy_texture_info(destination).into(),
                dict::union_extent_3d(copy_size),
            ],
        )
        .map(|_| ())
    }

    fn copy_texture_to_buffer(
        &self,
        source: &TexelCopyTextureInfo<'_, Web>,
        destination: &TexelCopyBufferInfo<'_, Web>,
        copy_size: &Extent3D,
    ) -> Result<(), Exception> {
        call(
            &self.0,
            "GPUCommandEncoder",
            "copyTextureToBuffer",
            &[
                dict::dict_texel_copy_texture_info(source).into(),
                dict::dict_texel_copy_buffer_info(destination).into(),
                dict::union_extent_3d(copy_size),
            ],
        )
        .map(|_| ())
    }

    fn copy_texture_to_texture(
        &self,
        source: &TexelCopyTextureInfo<'_, Web>,
        destination: &TexelCopyTextureInfo<'_, Web>,
        copy_size: &Extent3D,
    ) -> Result<(), Exception> {
        call(
            &self.0,
            "GPUCommandEncoder",
            "copyTextureToTexture",
            &[
                dict::dict_texel_copy_texture_info(source).into(),
                dict::dict_texel_copy_texture_info(destination).into(),
                dict::union_extent_3d(copy_size),
            ],
        )
        .map(|_| ())
    }

    fn clear_buffer(&self, buffer: &WebBuffer, offset: Size64, sz: Option<Size64>) {
        call_infallible(&self.0, "GPUCommandEncoder", "clearBuffer", &[buffer.0.clone(), size(offset), opt_size(sz)]);
    }

    fn resolve_query_set(
        &self,
        q: &WebQuerySet,
        first: u32,
        count: u32,
        destination: &WebBuffer,
        destination_offset: Size64,
    ) {
        call_infallible(
            &self.0,
            "GPUCommandEncoder",
            "resolveQuerySet",
            &[q.0.clone(), num(first), num(count), destination.0.clone(), size(destination_offset)],
        );
    }

    fn finish(&self, d: &CommandBufferDescriptor) -> WebCommandBuffer {
        WebCommandBuffer(call_infallible(
            &self.0,
            "GPUCommandEncoder",
            "finish",
            &[dict::dict_command_buffer_descriptor(d).into()],
        ))
    }
}

impl ComputePassEncoder<Web> for WebComputePassEncoder {
    fn set_pipeline(&self, p: &WebComputePipeline) {
        call_infallible(&self.0, "GPUComputePassEncoder", "setPipeline", std::slice::from_ref(&p.0));
    }
    fn dispatch_workgroups(&self, x: u32, y: u32, z: u32) {
        call_infallible(&self.0, "GPUComputePassEncoder", "dispatchWorkgroups", &[num(x), num(y), num(z)]);
    }
    fn dispatch_workgroups_indirect(&self, b: &WebBuffer, off: Size64) {
        call_infallible(&self.0, "GPUComputePassEncoder", "dispatchWorkgroupsIndirect", &[b.0.clone(), size(off)]);
    }
    fn end(&self) {
        call_infallible(&self.0, "GPUComputePassEncoder", "end", &[]);
    }
}

impl RenderPassEncoder<Web> for WebRenderPassEncoder {
    fn set_viewport(&self, x: Float32, y: Float32, w: Float32, h: Float32, min_depth: Float32, max_depth: Float32) {
        call_infallible(
            &self.0,
            "GPURenderPassEncoder",
            "setViewport",
            &[float(x), float(y), float(w), float(h), float(min_depth), float(max_depth)],
        );
    }
    fn set_scissor_rect(&self, x: u32, y: u32, w: u32, h: u32) {
        call_infallible(&self.0, "GPURenderPassEncoder", "setScissorRect", &[num(x), num(y), num(w), num(h)]);
    }
    fn set_blend_constant(&self, color: &Color) -> Result<(), Exception> {
        call(&self.0, "GPURenderPassEncoder", "setBlendConstant", &[dict::union_color(color)]).map(|_| ())
    }
    fn set_stencil_reference(&self, r: u32) {
        call_infallible(&self.0, "GPURenderPassEncoder", "setStencilReference", &[num(r)]);
    }
    fn begin_occlusion_query(&self, i: u32) {
        call_infallible(&self.0, "GPURenderPassEncoder", "beginOcclusionQuery", &[num(i)]);
    }
    fn end_occlusion_query(&self) {
        call_infallible(&self.0, "GPURenderPassEncoder", "endOcclusionQuery", &[]);
    }
    fn execute_bundles(&self, bundles: &[&WebRenderBundle]) {
        let arr = Array::new();
        for b in bundles {
            arr.push(b.js());
        }
        call_infallible(&self.0, "GPURenderPassEncoder", "executeBundles", &[arr.into()]);
    }
    fn end(&self) {
        call_infallible(&self.0, "GPURenderPassEncoder", "end", &[]);
    }
}

impl RenderBundleEncoder<Web> for WebRenderBundleEncoder {
    fn finish(&self, d: &RenderBundleDescriptor) -> WebRenderBundle {
        WebRenderBundle(call_infallible(
            &self.0,
            "GPURenderBundleEncoder",
            "finish",
            &[dict::dict_render_bundle_descriptor(d).into()],
        ))
    }
}
