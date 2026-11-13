//! The seam between the compute encoder and the pipeline service (plan section 11, H7).
//!
//! The pipeline service serves `CreateComputePipeline` and `CreateComputePipelineAsync` and binds the handle with a
//! [`ComputePipelineState`] payload. Metal's threadgroup size and threadgroup memory lengths are dispatch state, while
//! WGSL fixes the workgroup size and the size of every `var<workgroup>` in the shader (plan section 6.6), and the
//! bound textures decide the WGSL texture types: so the encoder asks, at each dispatch, for the variant of the pipeline
//! that matches the dispatch ([`ComputeVariants::variant`], keyed by [`VariantKey`]), and binds what its
//! [`ComputeVariant`] layout names. The pipeline service makes and caches the variants (from AirWgsl's output: WGSL, a
//! binding layout whose entries name their Metal source, and the reflection); this module only consumes them.

use std::rc::Rc;

use webmetal_wire::Failure;

use crate::exec::Gpu;
use crate::gpu::Api;
use crate::gpu::idl::{
    BufferBindingType, SamplerBindingType, StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};

/// The payload of a `ComputePipelineState` handle.
pub struct ComputePipelineState<A: Api> {
    /// `maxTotalThreadsPerThreadgroup` of the pipeline (the device's limit, the descriptor's, and what the function's
    /// resources allow, as the pipeline service computes it for the A10X).
    pub max_total_threads_per_threadgroup: u64,
    /// `threadExecutionWidth` of the pipeline.
    pub thread_execution_width: u64,
    /// The descriptor's `threadGroupSizeIsMultipleOfThreadExecutionWidth`: dispatches promise threadgroup sizes that
    /// are multiples of `threadExecutionWidth`.
    pub threadgroup_size_is_multiple_of_execution_width: bool,
    /// `staticThreadgroupMemoryLength`: threadgroup memory the function declares itself.
    pub static_threadgroup_memory_length: u64,
    /// The descriptor has a `stageInputDescriptor`: dispatches take the stage-in region.
    pub stage_input: bool,
    /// The variants of the pipeline.
    pub variants: Rc<dyn ComputeVariants<A>>,
}

/// The source of a pipeline's variants (the pipeline service).
pub trait ComputeVariants<A: Api> {
    /// The pipeline for a dispatch with this state. Called from the synchronous part of the encoder service; WebGPU
    /// calls go through `gpu`. A variant the service cannot make is a failure the command buffer completes with
    /// (status 2 for what is not implemented, 3 for what the host lacks); it never returns a stand-in.
    fn variant(&self, gpu: &Gpu<'_, A>, key: &VariantKey) -> Result<Rc<ComputeVariant<A>>, Failure>;
}

/// What, besides the pipeline, decides the generated code of a dispatch (AirWgsl's `ComputeState` and the kernel
/// stage's `StageBindings`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VariantKey {
    /// `threadsPerThreadgroup` of the dispatch.
    pub threads_per_threadgroup: [u32; 3],
    /// `setThreadgroupMemoryLength:atIndex:` lengths, index `i` at position `i` (0 where nothing was set), up to the
    /// highest index set.
    pub threadgroup_memory_lengths: Vec<u64>,
    /// Buffer argument table indices that hold a buffer or bytes.
    pub buffers: Vec<u64>,
    /// The bound textures.
    pub textures: Vec<TextureSlot>,
    /// The bound sampler states.
    pub samplers: Vec<SamplerSlot>,
}

/// A bound sampler state: the fields of its `MTLSamplerDescriptor` the compiler takes (AirWgsl's `SamplerSlot`), the
/// LOD clamps as their bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerSlot {
    /// Sampler argument table index.
    pub index: u64,
    pub min_filter: u64,
    pub mag_filter: u64,
    pub mip_filter: u64,
    /// s, t and r.
    pub address_modes: [u64; 3],
    pub border_color: u64,
    pub normalized_coordinates: bool,
    pub compare_function: u64,
    pub max_anisotropy: u64,
    pub lod_min_clamp: u32,
    pub lod_max_clamp: u32,
    pub lod_average: bool,
}

/// A bound texture: what of it changes the generated code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureSlot {
    /// Texture argument table index.
    pub index: u64,
    /// `MTLTextureType`.
    pub texture_type: u64,
    /// `MTLPixelFormat`.
    pub pixel_format: u64,
    pub sample_count: u64,
    /// The `MTLTextureSwizzle` channels reads apply ([`crate::mem::view::read_swizzle`]).
    pub swizzle: [u8; 4],
}

/// A compiled variant: the WebGPU pipeline and its bind groups.
pub struct ComputeVariant<A: Api> {
    pub pipeline: A::ComputePipeline,
    pub groups: Vec<BindGroup<A>>,
}

/// One bind group of a variant.
pub struct BindGroup<A: Api> {
    /// The `@group` index.
    pub group: u32,
    pub layout: A::BindGroupLayout,
    pub entries: Vec<Binding>,
}

/// One binding: its layout and where the encoder gets its resource.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    /// The `@binding` index.
    pub binding: u32,
    pub resource: Resource,
    pub source: Source,
}

/// The binding's resource layout (one `GPUBindGroupLayoutEntry` member).
#[derive(Clone, Debug, PartialEq)]
pub enum Resource {
    /// `GPUBufferBindingLayout`. `min_binding_size` is the size the shader's type needs.
    Buffer { ty: BufferBindingType, min_binding_size: u64 },
    /// `GPUTextureBindingLayout`.
    Texture { sample_type: TextureSampleType, view_dimension: TextureViewDimension, multisampled: bool },
    /// `GPUStorageTextureBindingLayout`.
    StorageTexture { access: StorageTextureAccess, format: TextureFormat, view_dimension: TextureViewDimension },
    /// `GPUSamplerBindingLayout`.
    Sampler { ty: SamplerBindingType },
}

/// Where the encoder gets a binding's resource. A source the encoder does not know is refused, never left unbound.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Source {
    /// The buffer (or bytes) at a buffer argument table index, bound from the Metal offset rounded down to the
    /// binding alignment; the rest of the offset is the slot's [`UniformValue::ResidualOffset`].
    Buffer { index: u64 },
    /// The texture at a texture argument table index.
    Texture { index: u64 },
    /// The sampler at a sampler argument table index.
    Sampler { index: u64 },
    /// A uniform block the encoder fills for each dispatch.
    Uniforms(UniformBlock),
}

/// A uniform block: its size and what each field holds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UniformBlock {
    pub size: u64,
    /// (byte offset, value).
    pub fields: Vec<(u64, UniformValue)>,
}

/// What a uniform field holds. Each value is one `u32` unless stated otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UniformValue {
    /// Bytes from the buffer binding's aligned base to the Metal offset (AirWgsl `ResidualOffset`).
    ResidualOffset { index: u64 },
    /// Length in bytes of the buffer bound at the slot, from the Metal offset (AirWgsl `BufferLength`).
    BufferLength { index: u64 },
    /// The stage-in region's origin, three `u32` (`setStageInRegion:`).
    StageInOrigin,
    /// The stage-in region's size, three `u32`.
    StageInSize,
}
