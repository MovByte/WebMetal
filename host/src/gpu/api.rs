//! The WebGPU interfaces as Rust traits.
//!
//! One trait per IDL interface or mixin, one method per member, with the IDL's argument order, optionality and
//! defaults (an optional argument without an IDL default is an `Option`; one with a default is passed explicitly).
//! An [`Api`] names the types an implementation uses for each interface; `native` (webgpu.h) and `web` (the browser)
//! are the two implementations. [`COVERAGE`] maps every member of every interface of the IDL index to its method here,
//! or states why it is not part of the binding; a test checks it against the generated inventory
//! [`super::idl::IDL_MEMBERS`].
//!
//! Errors follow the specification's three channels:
//!
//! - an operation that throws on the content timeline returns `Err(Exception)`;
//! - a promise that rejects resolves its [`GpuFuture`] to `Err(Exception)`;
//! - a device-timeline error is a [`GpuError`] dispatched to the device's error scopes or, if none captures it, to the
//!   `uncapturederror` handler. Methods return normally in that case, as in JavaScript.
//!
//! A method an implementation does not provide returns [`ExceptionKind::Unsupported`], which is not a JavaScript
//! exception: it is the binding's report that the call could not be made.

use core::fmt;
use core::future::Future;
use core::pin::Pin;
use std::collections::BTreeSet;

use super::idl::*;
use super::num::{Float32, Size64};

/// The result of an operation that returns a promise. Futures do not borrow the object they came from.
pub type GpuFuture<T> = Pin<Box<dyn Future<Output = T>>>;

/// What a JavaScript exception or promise rejection is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExceptionKind {
    /// ECMAScript `TypeError`.
    TypeError,
    /// ECMAScript `RangeError`.
    RangeError,
    /// `DOMException` "OperationError".
    OperationError,
    /// `DOMException` "AbortError".
    AbortError,
    /// `DOMException` "SecurityError".
    SecurityError,
    /// `DOMException` "InvalidStateError".
    InvalidStateError,
    /// `GPUPipelineError` with its reason.
    PipelineError(PipelineErrorReason),
    /// Any other exception, by its `name` (a browser may throw one the specification does not list, for example when a
    /// value is unusable for a reason outside WebGPU).
    Other(String),
    /// Not an exception: the implementation does not provide this member. The message names the member and why.
    Unsupported,
}

/// A JavaScript exception thrown by a WebGPU method, or a rejected promise's reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exception {
    pub kind: ExceptionKind,
    pub message: String,
}

impl Exception {
    pub fn new(kind: ExceptionKind, message: impl Into<String>) -> Exception {
        Exception { kind, message: message.into() }
    }

    pub fn unsupported(message: impl Into<String>) -> Exception {
        Exception::new(ExceptionKind::Unsupported, message)
    }
}

impl fmt::Display for Exception {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Exception {}

/// A `GPUError`: `GPUValidationError`, `GPUOutOfMemoryError` or `GPUInternalError`, named by the error filter that
/// captures it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuError {
    pub filter: ErrorFilter,
    pub message: String,
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} error: {}", self.filter.as_str(), self.message)
    }
}

/// `GPUDeviceLostInfo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceLostInfo {
    pub reason: DeviceLostReason,
    pub message: String,
}

/// `GPUCompilationMessage`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilationMessage {
    pub message: String,
    pub r#type: CompilationMessageType,
    pub line_num: u64,
    pub line_pos: u64,
    pub offset: u64,
    pub length: u64,
}

/// `GPUCompilationInfo`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct CompilationInfo {
    pub messages: Vec<CompilationMessage>,
}

/// The `onuncapturederror` event handler: receives the `error` of each `GPUUncapturedErrorEvent`.
pub type UncapturedErrorHandler = Box<dyn FnMut(GpuError)>;

/// The types of one WebGPU implementation.
pub trait Api: Sized + Clone + fmt::Debug + 'static {
    type Gpu: Gpu<Self>;
    type Adapter: Adapter<Self>;
    type Device: Device<Self>;
    type Queue: Queue<Self>;
    type Buffer: Buffer<Self>;
    /// The `ArrayBuffer` returned by `getMappedRange()`.
    type MappedRange: MappedRange;
    type Texture: Texture<Self>;
    type TextureView: Object;
    type ExternalTexture: Object;
    type Sampler: Object;
    type BindGroupLayout: Object;
    type BindGroup: Object;
    type PipelineLayout: Object;
    type ShaderModule: ShaderModule<Self>;
    type ComputePipeline: PipelineBase<Self>;
    type RenderPipeline: PipelineBase<Self>;
    type CommandBuffer: Object;
    type CommandEncoder: CommandEncoder<Self>;
    type ComputePassEncoder: ComputePassEncoder<Self>;
    type RenderPassEncoder: RenderPassEncoder<Self>;
    type RenderBundle: Object;
    type RenderBundleEncoder: RenderBundleEncoder<Self>;
    type QuerySet: QuerySet<Self>;
    /// A `GPUCopyExternalImageSource` (an image, canvas, bitmap, video or frame of the host platform).
    type ExternalImageSource: Clone + fmt::Debug;
    /// An `HTMLVideoElement` or `VideoFrame`.
    type ExternalVideoSource: Clone + fmt::Debug;
}

/// `GPUObjectBase`. Objects are references: cloning one refers to the same WebGPU object.
pub trait Object: Clone + fmt::Debug + 'static {
    /// The `label` attribute.
    fn label(&self) -> String;
    /// Set the `label` attribute.
    fn set_label(&self, label: &str);
}

/// `GPU`.
pub trait Gpu<A: Api> {
    /// `requestAdapter(options)`. `Ok(None)` when no adapter is returned.
    fn request_adapter(&self, options: &RequestAdapterOptions) -> GpuFuture<Result<Option<A::Adapter>, Exception>>;
    /// The `wgslLanguageFeatures` attribute.
    fn wgsl_language_features(&self) -> Result<BTreeSet<String>, Exception>;
    /// Not an IDL member: which implementation this is, for reports.
    fn implementation(&self) -> String;
}

/// `GPUAdapter`.
pub trait Adapter<A: Api>: Clone + fmt::Debug {
    /// The `features` attribute (a set of feature names; the implementation may know names newer than this IDL).
    fn features(&self) -> BTreeSet<String>;
    /// The `limits` attribute.
    fn limits(&self) -> SupportedLimits;
    /// The `info` attribute.
    fn info(&self) -> AdapterInfo;
    /// `requestDevice(descriptor)`.
    fn request_device(&self, descriptor: &DeviceDescriptor) -> GpuFuture<Result<A::Device, Exception>>;
}

/// `GPUDevice`.
pub trait Device<A: Api>: Object {
    fn features(&self) -> BTreeSet<String>;
    fn limits(&self) -> SupportedLimits;
    fn adapter_info(&self) -> AdapterInfo;
    /// The `queue` attribute.
    fn queue(&self) -> A::Queue;
    fn destroy(&self);
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<A::Buffer, Exception>;
    fn create_texture(&self, descriptor: &TextureDescriptor) -> Result<A::Texture, Exception>;
    fn create_sampler(&self, descriptor: &SamplerDescriptor) -> A::Sampler;
    fn import_external_texture(
        &self,
        descriptor: &ExternalTextureDescriptor<'_, A>,
    ) -> Result<A::ExternalTexture, Exception>;
    fn create_bind_group_layout(&self, descriptor: &BindGroupLayoutDescriptor)
    -> Result<A::BindGroupLayout, Exception>;
    fn create_pipeline_layout(&self, descriptor: &PipelineLayoutDescriptor<'_, A>) -> A::PipelineLayout;
    fn create_bind_group(&self, descriptor: &BindGroupDescriptor<'_, A>) -> A::BindGroup;
    fn create_shader_module(&self, descriptor: &ShaderModuleDescriptor<'_, A>) -> A::ShaderModule;
    fn create_compute_pipeline(&self, descriptor: &ComputePipelineDescriptor<'_, A>) -> A::ComputePipeline;
    fn create_render_pipeline(
        &self,
        descriptor: &RenderPipelineDescriptor<'_, A>,
    ) -> Result<A::RenderPipeline, Exception>;
    fn create_compute_pipeline_async(
        &self,
        descriptor: &ComputePipelineDescriptor<'_, A>,
    ) -> GpuFuture<Result<A::ComputePipeline, Exception>>;
    fn create_render_pipeline_async(
        &self,
        descriptor: &RenderPipelineDescriptor<'_, A>,
    ) -> GpuFuture<Result<A::RenderPipeline, Exception>>;
    fn create_command_encoder(&self, descriptor: &CommandEncoderDescriptor) -> A::CommandEncoder;
    fn create_render_bundle_encoder(
        &self,
        descriptor: &RenderBundleEncoderDescriptor,
    ) -> Result<A::RenderBundleEncoder, Exception>;
    fn create_query_set(&self, descriptor: &QuerySetDescriptor) -> Result<A::QuerySet, Exception>;
    /// The `lost` attribute: resolves once, when the device is lost; every call returns a future of the same promise.
    fn lost(&self) -> GpuFuture<DeviceLostInfo>;
    fn push_error_scope(&self, filter: ErrorFilter);
    fn pop_error_scope(&self) -> GpuFuture<Result<Option<GpuError>, Exception>>;
    /// Set the `onuncapturederror` attribute (`None`: no handler).
    fn set_onuncapturederror(&self, handler: Option<UncapturedErrorHandler>);
}

/// `GPUBuffer`.
pub trait Buffer<A: Api>: Object {
    fn size(&self) -> u64;
    fn usage(&self) -> u32;
    fn map_state(&self) -> BufferMapState;
    fn map_async(&self, mode: u32, offset: Size64, size: Option<Size64>) -> GpuFuture<Result<(), Exception>>;
    fn get_mapped_range(&self, offset: Size64, size: Option<Size64>) -> Result<A::MappedRange, Exception>;
    fn unmap(&self);
    fn destroy(&self);
}

/// The `ArrayBuffer` `getMappedRange()` returns. `unmap()` detaches it: its length becomes 0 and access fails with
/// `TypeError`, as a view over a detached `ArrayBuffer` does.
pub trait MappedRange: fmt::Debug {
    /// `byteLength`.
    fn byte_length(&self) -> u64;
    /// Copy `dst.len()` bytes at `offset` out (`new Uint8Array(buffer, offset, dst.len())`): `TypeError` when detached,
    /// `RangeError` when the range is outside the buffer.
    fn read(&self, offset: u64, dst: &mut [u8]) -> Result<(), Exception>;
    /// Copy `src` in at `offset`, with the same errors as [`MappedRange::read`].
    fn write(&self, offset: u64, src: &[u8]) -> Result<(), Exception>;
}

/// `GPUTexture`.
pub trait Texture<A: Api>: Object {
    fn create_view(&self, descriptor: &TextureViewDescriptor) -> Result<A::TextureView, Exception>;
    fn destroy(&self);
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn depth_or_array_layers(&self) -> u32;
    fn mip_level_count(&self) -> u32;
    fn sample_count(&self) -> u32;
    fn dimension(&self) -> TextureDimension;
    fn format(&self) -> TextureFormat;
    fn usage(&self) -> u32;
    /// `textureBindingViewDimension`; `None` is `undefined`.
    fn texture_binding_view_dimension(&self) -> Option<TextureViewDimension>;
}

/// `GPUShaderModule`.
pub trait ShaderModule<A: Api>: Object {
    fn get_compilation_info(&self) -> GpuFuture<Result<CompilationInfo, Exception>>;
}

/// `GPUPipelineBase` with `GPUObjectBase`.
pub trait PipelineBase<A: Api>: Object {
    fn get_bind_group_layout(&self, index: u32) -> A::BindGroupLayout;
}

/// `GPUDebugCommandsMixin`.
pub trait DebugCommands {
    fn push_debug_group(&self, group_label: &str);
    fn pop_debug_group(&self);
    fn insert_debug_marker(&self, marker_label: &str);
}

/// `GPUBindingCommandsMixin`.
pub trait BindingCommands<A: Api> {
    /// `setBindGroup(index, bindGroup, dynamicOffsets)`.
    fn set_bind_group(&self, index: u32, bind_group: Option<&A::BindGroup>, dynamic_offsets: &[u32]);
    /// `setBindGroup(index, bindGroup, dynamicOffsetsData, dynamicOffsetsDataStart, dynamicOffsetsDataLength)`, with
    /// `data` as the `Uint32Array`.
    fn set_bind_group_with_data(
        &self,
        index: u32,
        bind_group: Option<&A::BindGroup>,
        dynamic_offsets_data: &[u32],
        dynamic_offsets_data_start: Size64,
        dynamic_offsets_data_length: u32,
    ) -> Result<(), Exception>;
    /// `setImmediates(rangeOffset, data, dataOffset, dataSize)`, with `data` as an `ArrayBuffer` (element size 1).
    fn set_immediates(
        &self,
        range_offset: u32,
        data: &[u8],
        data_offset: Size64,
        data_size: Option<Size64>,
    ) -> Result<(), Exception>;
}

/// `GPURenderCommandsMixin`.
pub trait RenderCommands<A: Api> {
    fn set_pipeline(&self, pipeline: &A::RenderPipeline);
    fn set_index_buffer(&self, buffer: &A::Buffer, index_format: IndexFormat, offset: Size64, size: Option<Size64>);
    fn set_vertex_buffer(&self, slot: u32, buffer: Option<&A::Buffer>, offset: Size64, size: Option<Size64>);
    fn draw(&self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32);
    fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    );
    fn draw_indirect(&self, indirect_buffer: &A::Buffer, indirect_offset: Size64);
    fn draw_indexed_indirect(&self, indirect_buffer: &A::Buffer, indirect_offset: Size64);
}

/// `GPUCommandEncoder`.
pub trait CommandEncoder<A: Api>: Object + DebugCommands {
    fn begin_render_pass(&self, descriptor: &RenderPassDescriptor<'_, A>) -> Result<A::RenderPassEncoder, Exception>;
    /// Fails only with [`ExceptionKind::Unsupported`], for a value the implementation cannot carry.
    fn begin_compute_pass(&self, descriptor: &ComputePassDescriptor<'_, A>)
    -> Result<A::ComputePassEncoder, Exception>;
    /// `copyBufferToBuffer(source, destination, size)`.
    fn copy_buffer_to_buffer(&self, source: &A::Buffer, destination: &A::Buffer, size: Option<Size64>);
    /// `copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size)`.
    fn copy_buffer_to_buffer_with_offsets(
        &self,
        source: &A::Buffer,
        source_offset: Size64,
        destination: &A::Buffer,
        destination_offset: Size64,
        size: Option<Size64>,
    );
    fn copy_buffer_to_texture(
        &self,
        source: &TexelCopyBufferInfo<'_, A>,
        destination: &TexelCopyTextureInfo<'_, A>,
        copy_size: &Extent3D,
    ) -> Result<(), Exception>;
    fn copy_texture_to_buffer(
        &self,
        source: &TexelCopyTextureInfo<'_, A>,
        destination: &TexelCopyBufferInfo<'_, A>,
        copy_size: &Extent3D,
    ) -> Result<(), Exception>;
    fn copy_texture_to_texture(
        &self,
        source: &TexelCopyTextureInfo<'_, A>,
        destination: &TexelCopyTextureInfo<'_, A>,
        copy_size: &Extent3D,
    ) -> Result<(), Exception>;
    fn clear_buffer(&self, buffer: &A::Buffer, offset: Size64, size: Option<Size64>);
    fn resolve_query_set(
        &self,
        query_set: &A::QuerySet,
        first_query: u32,
        query_count: u32,
        destination: &A::Buffer,
        destination_offset: Size64,
    );
    fn finish(&self, descriptor: &CommandBufferDescriptor) -> A::CommandBuffer;
}

/// `GPUComputePassEncoder`.
pub trait ComputePassEncoder<A: Api>: Object + DebugCommands + BindingCommands<A> {
    fn set_pipeline(&self, pipeline: &A::ComputePipeline);
    fn dispatch_workgroups(&self, workgroup_count_x: u32, workgroup_count_y: u32, workgroup_count_z: u32);
    fn dispatch_workgroups_indirect(&self, indirect_buffer: &A::Buffer, indirect_offset: Size64);
    fn end(&self);
}

/// `GPURenderPassEncoder`.
pub trait RenderPassEncoder<A: Api>: Object + DebugCommands + BindingCommands<A> + RenderCommands<A> {
    fn set_viewport(
        &self,
        x: Float32,
        y: Float32,
        width: Float32,
        height: Float32,
        min_depth: Float32,
        max_depth: Float32,
    );
    fn set_scissor_rect(&self, x: u32, y: u32, width: u32, height: u32);
    fn set_blend_constant(&self, color: &Color) -> Result<(), Exception>;
    fn set_stencil_reference(&self, reference: u32);
    fn begin_occlusion_query(&self, query_index: u32);
    fn end_occlusion_query(&self);
    fn execute_bundles(&self, bundles: &[&A::RenderBundle]);
    fn end(&self);
}

/// `GPURenderBundleEncoder`.
pub trait RenderBundleEncoder<A: Api>: Object + DebugCommands + BindingCommands<A> + RenderCommands<A> {
    fn finish(&self, descriptor: &RenderBundleDescriptor) -> A::RenderBundle;
}

/// `GPUQueue`.
pub trait Queue<A: Api>: Object {
    fn submit(&self, command_buffers: &[&A::CommandBuffer]);
    fn on_submitted_work_done(&self) -> GpuFuture<()>;
    /// `writeBuffer(buffer, bufferOffset, data, dataOffset, size)`, with `data` as an `ArrayBuffer` (element size 1).
    fn write_buffer(
        &self,
        buffer: &A::Buffer,
        buffer_offset: Size64,
        data: &[u8],
        data_offset: Size64,
        size: Option<Size64>,
    ) -> Result<(), Exception>;
    fn write_texture(
        &self,
        destination: &TexelCopyTextureInfo<'_, A>,
        data: &[u8],
        data_layout: &TexelCopyBufferLayout,
        size: &Extent3D,
    ) -> Result<(), Exception>;
    fn copy_external_image_to_texture(
        &self,
        source: &CopyExternalImageSourceInfo<'_, A>,
        destination: &CopyExternalImageDestInfo<'_, A>,
        copy_size: &Extent3D,
    ) -> Result<(), Exception>;
}

/// `GPUQuerySet`.
pub trait QuerySet<A: Api>: Object {
    fn destroy(&self);
    fn r#type(&self) -> QueryType;
    fn count(&self) -> u32;
}

/// Where an IDL interface member lives in this binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// The trait methods that implement it (`Trait::method`; several for an attribute with a setter or an overloaded
    /// operation).
    Methods(&'static [&'static str]),
    /// Not part of the binding, and why.
    Excluded(&'static str),
}

const PRESENTATION: &str =
    "canvas presentation (GPUCanvasContext and getPreferredCanvasFormat) is the presentation layer's binding";
const EVENT_OBJECT: &str = "a DOM event object; the binding delivers its `error` to the onuncapturederror handler";
const ERROR_OBJECT: &str = "an error object; the binding reports it as GpuError or Exception values";
const LIMITS_OBJECT: &str = "read as one SupportedLimits value through Adapter::limits and Device::limits";
const INFO_OBJECT: &str = "read as one AdapterInfo value through Adapter::info and Device::adapter_info";
const FEATURES_OBJECT: &str =
    "read as one set through Adapter::features, Device::features and Gpu::wgsl_language_features";
const NAVIGATOR: &str =
    "the entry point: an implementation's Gpu value is created by the embedder (navigator.gpu in a page)";
const INFO_RECORDS: &str = "read as plain values in CompilationInfo, CompilationMessage and DeviceLostInfo";

macro_rules! m {
    ($($s:literal),+) => {
        Coverage::Methods(&[$($s),+])
    };
}

/// Every member of every interface in the IDL index (see [`super::idl::IDL_MEMBERS`]).
pub const COVERAGE: &[(&str, &str, Coverage)] = &[
    ("GPU", "requestAdapter", m!("Gpu::request_adapter")),
    ("GPU", "getPreferredCanvasFormat", Coverage::Excluded(PRESENTATION)),
    ("GPU", "wgslLanguageFeatures", m!("Gpu::wgsl_language_features")),
    ("GPUAdapter", "features", m!("Adapter::features")),
    ("GPUAdapter", "limits", m!("Adapter::limits")),
    ("GPUAdapter", "info", m!("Adapter::info")),
    ("GPUAdapter", "requestDevice", m!("Adapter::request_device")),
    ("GPUAdapterInfo", "vendor", Coverage::Excluded(INFO_OBJECT)),
    ("GPUAdapterInfo", "architecture", Coverage::Excluded(INFO_OBJECT)),
    ("GPUAdapterInfo", "device", Coverage::Excluded(INFO_OBJECT)),
    ("GPUAdapterInfo", "description", Coverage::Excluded(INFO_OBJECT)),
    ("GPUAdapterInfo", "subgroupMinSize", Coverage::Excluded(INFO_OBJECT)),
    ("GPUAdapterInfo", "subgroupMaxSize", Coverage::Excluded(INFO_OBJECT)),
    ("GPUAdapterInfo", "isFallbackAdapter", Coverage::Excluded(INFO_OBJECT)),
    ("GPUBindGroup", "label", m!("Object::label", "Object::set_label")),
    ("GPUBindGroupLayout", "label", m!("Object::label", "Object::set_label")),
    ("GPUBuffer", "size", m!("Buffer::size")),
    ("GPUBuffer", "usage", m!("Buffer::usage")),
    ("GPUBuffer", "mapState", m!("Buffer::map_state")),
    ("GPUBuffer", "mapAsync", m!("Buffer::map_async")),
    ("GPUBuffer", "getMappedRange", m!("Buffer::get_mapped_range")),
    ("GPUBuffer", "unmap", m!("Buffer::unmap")),
    ("GPUBuffer", "destroy", m!("Buffer::destroy")),
    ("GPUBuffer", "label", m!("Object::label", "Object::set_label")),
    ("GPUCanvasContext", "canvas", Coverage::Excluded(PRESENTATION)),
    ("GPUCanvasContext", "configure", Coverage::Excluded(PRESENTATION)),
    ("GPUCanvasContext", "unconfigure", Coverage::Excluded(PRESENTATION)),
    ("GPUCanvasContext", "getConfiguration", Coverage::Excluded(PRESENTATION)),
    ("GPUCanvasContext", "getCurrentTexture", Coverage::Excluded(PRESENTATION)),
    ("GPUCommandBuffer", "label", m!("Object::label", "Object::set_label")),
    ("GPUCommandEncoder", "beginRenderPass", m!("CommandEncoder::begin_render_pass")),
    ("GPUCommandEncoder", "beginComputePass", m!("CommandEncoder::begin_compute_pass")),
    (
        "GPUCommandEncoder",
        "copyBufferToBuffer",
        m!("CommandEncoder::copy_buffer_to_buffer", "CommandEncoder::copy_buffer_to_buffer_with_offsets"),
    ),
    ("GPUCommandEncoder", "copyBufferToTexture", m!("CommandEncoder::copy_buffer_to_texture")),
    ("GPUCommandEncoder", "copyTextureToBuffer", m!("CommandEncoder::copy_texture_to_buffer")),
    ("GPUCommandEncoder", "copyTextureToTexture", m!("CommandEncoder::copy_texture_to_texture")),
    ("GPUCommandEncoder", "clearBuffer", m!("CommandEncoder::clear_buffer")),
    ("GPUCommandEncoder", "resolveQuerySet", m!("CommandEncoder::resolve_query_set")),
    ("GPUCommandEncoder", "finish", m!("CommandEncoder::finish")),
    ("GPUCommandEncoder", "label", m!("Object::label", "Object::set_label")),
    ("GPUCommandEncoder", "pushDebugGroup", m!("DebugCommands::push_debug_group")),
    ("GPUCommandEncoder", "popDebugGroup", m!("DebugCommands::pop_debug_group")),
    ("GPUCommandEncoder", "insertDebugMarker", m!("DebugCommands::insert_debug_marker")),
    ("GPUCompilationInfo", "messages", Coverage::Excluded(INFO_RECORDS)),
    ("GPUCompilationMessage", "message", Coverage::Excluded(INFO_RECORDS)),
    ("GPUCompilationMessage", "type", Coverage::Excluded(INFO_RECORDS)),
    ("GPUCompilationMessage", "lineNum", Coverage::Excluded(INFO_RECORDS)),
    ("GPUCompilationMessage", "linePos", Coverage::Excluded(INFO_RECORDS)),
    ("GPUCompilationMessage", "offset", Coverage::Excluded(INFO_RECORDS)),
    ("GPUCompilationMessage", "length", Coverage::Excluded(INFO_RECORDS)),
    ("GPUComputePassEncoder", "setPipeline", m!("ComputePassEncoder::set_pipeline")),
    ("GPUComputePassEncoder", "dispatchWorkgroups", m!("ComputePassEncoder::dispatch_workgroups")),
    ("GPUComputePassEncoder", "dispatchWorkgroupsIndirect", m!("ComputePassEncoder::dispatch_workgroups_indirect")),
    ("GPUComputePassEncoder", "end", m!("ComputePassEncoder::end")),
    ("GPUComputePassEncoder", "label", m!("Object::label", "Object::set_label")),
    ("GPUComputePassEncoder", "pushDebugGroup", m!("DebugCommands::push_debug_group")),
    ("GPUComputePassEncoder", "popDebugGroup", m!("DebugCommands::pop_debug_group")),
    ("GPUComputePassEncoder", "insertDebugMarker", m!("DebugCommands::insert_debug_marker")),
    (
        "GPUComputePassEncoder",
        "setBindGroup",
        m!("BindingCommands::set_bind_group", "BindingCommands::set_bind_group_with_data"),
    ),
    ("GPUComputePassEncoder", "setImmediates", m!("BindingCommands::set_immediates")),
    ("GPUComputePipeline", "label", m!("Object::label", "Object::set_label")),
    ("GPUComputePipeline", "getBindGroupLayout", m!("PipelineBase::get_bind_group_layout")),
    ("GPUDevice", "features", m!("Device::features")),
    ("GPUDevice", "limits", m!("Device::limits")),
    ("GPUDevice", "adapterInfo", m!("Device::adapter_info")),
    ("GPUDevice", "queue", m!("Device::queue")),
    ("GPUDevice", "destroy", m!("Device::destroy")),
    ("GPUDevice", "createBuffer", m!("Device::create_buffer")),
    ("GPUDevice", "createTexture", m!("Device::create_texture")),
    ("GPUDevice", "createSampler", m!("Device::create_sampler")),
    ("GPUDevice", "importExternalTexture", m!("Device::import_external_texture")),
    ("GPUDevice", "createBindGroupLayout", m!("Device::create_bind_group_layout")),
    ("GPUDevice", "createPipelineLayout", m!("Device::create_pipeline_layout")),
    ("GPUDevice", "createBindGroup", m!("Device::create_bind_group")),
    ("GPUDevice", "createShaderModule", m!("Device::create_shader_module")),
    ("GPUDevice", "createComputePipeline", m!("Device::create_compute_pipeline")),
    ("GPUDevice", "createRenderPipeline", m!("Device::create_render_pipeline")),
    ("GPUDevice", "createComputePipelineAsync", m!("Device::create_compute_pipeline_async")),
    ("GPUDevice", "createRenderPipelineAsync", m!("Device::create_render_pipeline_async")),
    ("GPUDevice", "createCommandEncoder", m!("Device::create_command_encoder")),
    ("GPUDevice", "createRenderBundleEncoder", m!("Device::create_render_bundle_encoder")),
    ("GPUDevice", "createQuerySet", m!("Device::create_query_set")),
    ("GPUDevice", "label", m!("Object::label", "Object::set_label")),
    ("GPUDevice", "lost", m!("Device::lost")),
    ("GPUDevice", "pushErrorScope", m!("Device::push_error_scope")),
    ("GPUDevice", "popErrorScope", m!("Device::pop_error_scope")),
    ("GPUDevice", "onuncapturederror", m!("Device::set_onuncapturederror")),
    ("GPUDeviceLostInfo", "reason", Coverage::Excluded(INFO_RECORDS)),
    ("GPUDeviceLostInfo", "message", Coverage::Excluded(INFO_RECORDS)),
    ("GPUError", "message", Coverage::Excluded(ERROR_OBJECT)),
    ("GPUExternalTexture", "label", m!("Object::label", "Object::set_label")),
    ("GPUInternalError", "constructor", Coverage::Excluded(ERROR_OBJECT)),
    ("GPUOutOfMemoryError", "constructor", Coverage::Excluded(ERROR_OBJECT)),
    ("GPUPipelineError", "constructor", Coverage::Excluded(ERROR_OBJECT)),
    ("GPUPipelineError", "reason", Coverage::Excluded(ERROR_OBJECT)),
    ("GPUPipelineLayout", "label", m!("Object::label", "Object::set_label")),
    ("GPUQuerySet", "destroy", m!("QuerySet::destroy")),
    ("GPUQuerySet", "type", m!("QuerySet::type")),
    ("GPUQuerySet", "count", m!("QuerySet::count")),
    ("GPUQuerySet", "label", m!("Object::label", "Object::set_label")),
    ("GPUQueue", "submit", m!("Queue::submit")),
    ("GPUQueue", "onSubmittedWorkDone", m!("Queue::on_submitted_work_done")),
    ("GPUQueue", "writeBuffer", m!("Queue::write_buffer")),
    ("GPUQueue", "writeTexture", m!("Queue::write_texture")),
    ("GPUQueue", "copyExternalImageToTexture", m!("Queue::copy_external_image_to_texture")),
    ("GPUQueue", "label", m!("Object::label", "Object::set_label")),
    ("GPURenderBundle", "label", m!("Object::label", "Object::set_label")),
    ("GPURenderBundleEncoder", "finish", m!("RenderBundleEncoder::finish")),
    ("GPURenderBundleEncoder", "label", m!("Object::label", "Object::set_label")),
    ("GPURenderBundleEncoder", "pushDebugGroup", m!("DebugCommands::push_debug_group")),
    ("GPURenderBundleEncoder", "popDebugGroup", m!("DebugCommands::pop_debug_group")),
    ("GPURenderBundleEncoder", "insertDebugMarker", m!("DebugCommands::insert_debug_marker")),
    (
        "GPURenderBundleEncoder",
        "setBindGroup",
        m!("BindingCommands::set_bind_group", "BindingCommands::set_bind_group_with_data"),
    ),
    ("GPURenderBundleEncoder", "setImmediates", m!("BindingCommands::set_immediates")),
    ("GPURenderBundleEncoder", "setPipeline", m!("RenderCommands::set_pipeline")),
    ("GPURenderBundleEncoder", "setIndexBuffer", m!("RenderCommands::set_index_buffer")),
    ("GPURenderBundleEncoder", "setVertexBuffer", m!("RenderCommands::set_vertex_buffer")),
    ("GPURenderBundleEncoder", "draw", m!("RenderCommands::draw")),
    ("GPURenderBundleEncoder", "drawIndexed", m!("RenderCommands::draw_indexed")),
    ("GPURenderBundleEncoder", "drawIndirect", m!("RenderCommands::draw_indirect")),
    ("GPURenderBundleEncoder", "drawIndexedIndirect", m!("RenderCommands::draw_indexed_indirect")),
    ("GPURenderPassEncoder", "setViewport", m!("RenderPassEncoder::set_viewport")),
    ("GPURenderPassEncoder", "setScissorRect", m!("RenderPassEncoder::set_scissor_rect")),
    ("GPURenderPassEncoder", "setBlendConstant", m!("RenderPassEncoder::set_blend_constant")),
    ("GPURenderPassEncoder", "setStencilReference", m!("RenderPassEncoder::set_stencil_reference")),
    ("GPURenderPassEncoder", "beginOcclusionQuery", m!("RenderPassEncoder::begin_occlusion_query")),
    ("GPURenderPassEncoder", "endOcclusionQuery", m!("RenderPassEncoder::end_occlusion_query")),
    ("GPURenderPassEncoder", "executeBundles", m!("RenderPassEncoder::execute_bundles")),
    ("GPURenderPassEncoder", "end", m!("RenderPassEncoder::end")),
    ("GPURenderPassEncoder", "label", m!("Object::label", "Object::set_label")),
    ("GPURenderPassEncoder", "pushDebugGroup", m!("DebugCommands::push_debug_group")),
    ("GPURenderPassEncoder", "popDebugGroup", m!("DebugCommands::pop_debug_group")),
    ("GPURenderPassEncoder", "insertDebugMarker", m!("DebugCommands::insert_debug_marker")),
    (
        "GPURenderPassEncoder",
        "setBindGroup",
        m!("BindingCommands::set_bind_group", "BindingCommands::set_bind_group_with_data"),
    ),
    ("GPURenderPassEncoder", "setImmediates", m!("BindingCommands::set_immediates")),
    ("GPURenderPassEncoder", "setPipeline", m!("RenderCommands::set_pipeline")),
    ("GPURenderPassEncoder", "setIndexBuffer", m!("RenderCommands::set_index_buffer")),
    ("GPURenderPassEncoder", "setVertexBuffer", m!("RenderCommands::set_vertex_buffer")),
    ("GPURenderPassEncoder", "draw", m!("RenderCommands::draw")),
    ("GPURenderPassEncoder", "drawIndexed", m!("RenderCommands::draw_indexed")),
    ("GPURenderPassEncoder", "drawIndirect", m!("RenderCommands::draw_indirect")),
    ("GPURenderPassEncoder", "drawIndexedIndirect", m!("RenderCommands::draw_indexed_indirect")),
    ("GPURenderPipeline", "label", m!("Object::label", "Object::set_label")),
    ("GPURenderPipeline", "getBindGroupLayout", m!("PipelineBase::get_bind_group_layout")),
    ("GPUSampler", "label", m!("Object::label", "Object::set_label")),
    ("GPUShaderModule", "getCompilationInfo", m!("ShaderModule::get_compilation_info")),
    ("GPUShaderModule", "label", m!("Object::label", "Object::set_label")),
    ("GPUSupportedFeatures", "setlike", Coverage::Excluded(FEATURES_OBJECT)),
    ("GPUSupportedLimits", "maxTextureDimension1D", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxTextureDimension2D", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxTextureDimension3D", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxTextureArrayLayers", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxBindGroups", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxBindGroupsPlusVertexBuffers", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxImmediateSize", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxBindingsPerBindGroup", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxDynamicUniformBuffersPerPipelineLayout", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxDynamicStorageBuffersPerPipelineLayout", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxSampledTexturesPerShaderStage", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxSamplersPerShaderStage", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxStorageBuffersPerShaderStage", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxStorageBuffersInVertexStage", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxStorageBuffersInFragmentStage", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxStorageTexturesPerShaderStage", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxStorageTexturesInVertexStage", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxStorageTexturesInFragmentStage", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxUniformBuffersPerShaderStage", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxUniformBufferBindingSize", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxStorageBufferBindingSize", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "minUniformBufferOffsetAlignment", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "minStorageBufferOffsetAlignment", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxVertexBuffers", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxBufferSize", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxVertexAttributes", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxVertexBufferArrayStride", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxInterStageShaderVariables", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxColorAttachments", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxColorAttachmentBytesPerSample", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxComputeWorkgroupStorageSize", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxComputeInvocationsPerWorkgroup", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxComputeWorkgroupSizeX", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxComputeWorkgroupSizeY", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxComputeWorkgroupSizeZ", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUSupportedLimits", "maxComputeWorkgroupsPerDimension", Coverage::Excluded(LIMITS_OBJECT)),
    ("GPUTexture", "createView", m!("Texture::create_view")),
    ("GPUTexture", "destroy", m!("Texture::destroy")),
    ("GPUTexture", "width", m!("Texture::width")),
    ("GPUTexture", "height", m!("Texture::height")),
    ("GPUTexture", "depthOrArrayLayers", m!("Texture::depth_or_array_layers")),
    ("GPUTexture", "mipLevelCount", m!("Texture::mip_level_count")),
    ("GPUTexture", "sampleCount", m!("Texture::sample_count")),
    ("GPUTexture", "dimension", m!("Texture::dimension")),
    ("GPUTexture", "format", m!("Texture::format")),
    ("GPUTexture", "usage", m!("Texture::usage")),
    ("GPUTexture", "textureBindingViewDimension", m!("Texture::texture_binding_view_dimension")),
    ("GPUTexture", "label", m!("Object::label", "Object::set_label")),
    ("GPUTextureView", "label", m!("Object::label", "Object::set_label")),
    ("GPUUncapturedErrorEvent", "constructor", Coverage::Excluded(EVENT_OBJECT)),
    ("GPUUncapturedErrorEvent", "error", Coverage::Excluded(EVENT_OBJECT)),
    ("GPUValidationError", "constructor", Coverage::Excluded(ERROR_OBJECT)),
    ("Navigator", "gpu", Coverage::Excluded(NAVIGATOR)),
    ("WGSLLanguageFeatures", "setlike", Coverage::Excluded(FEATURES_OBJECT)),
    ("WorkerNavigator", "gpu", Coverage::Excluded(NAVIGATOR)),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_accounts_for_every_idl_member() {
        let idl: BTreeSet<(&str, &str)> = IDL_MEMBERS.iter().map(|(i, m, _)| (*i, *m)).collect();
        let cov: BTreeSet<(&str, &str)> = COVERAGE.iter().map(|(i, m, _)| (*i, *m)).collect();
        assert_eq!(COVERAGE.len(), cov.len(), "COVERAGE lists a member twice");
        let missing: Vec<_> = idl.difference(&cov).collect();
        let extra: Vec<_> = cov.difference(&idl).collect();
        assert!(missing.is_empty() && extra.is_empty(), "not covered: {missing:?}; not in the IDL: {extra:?}");
    }

    #[test]
    fn coverage_names_methods_that_exist() {
        let src = include_str!("api.rs");
        for (iface, member, cov) in COVERAGE {
            let Coverage::Methods(methods) = cov else { continue };
            for path in *methods {
                let (tr, method) = path.split_once("::").unwrap();
                let start =
                    src.find(&format!("pub trait {tr}")).unwrap_or_else(|| panic!("{iface}.{member}: no trait {tr}"));
                let body = &src[start..];
                let end = body.find("\n}\n").unwrap();
                let method = if method == "type" { "r#type" } else { method };
                assert!(
                    body[..end].contains(&format!("fn {method}(")),
                    "{iface}.{member}: trait {tr} has no method {method}"
                );
            }
        }
    }
}
