//! Render pipelines as the render path uses them, and their dynamic-state variants.
//!
//! **The seam with the pipeline service.** The service that serves `CreateRenderPipeline*` (plan package H7) binds a
//! [`RenderPipeline`] to each `RenderPipelineState` handle. It holds what the compiler produced for the Metal
//! descriptor (plan section 4.3: WGSL stages, the binding layout, pass plans) in WebGPU objects, plus the Metal
//! descriptor fields the encoder checks a pass against. The pipeline service builds it with
//! [`RenderPipeline::create`] from a [`RenderPipelineDesc`]; tests build one from WGSL the same way. Its obligations:
//!
//! - `colors` has one entry per Metal color attachment index up to the last attachment with a pixel format, with
//!   Metal's pixel format and, for a valid format, the WebGPU target (the stored format of the format module's
//!   resolution, blend state and write mask). WebGPU requires the targets to match the pass exactly
//!   (`GPURenderPassEncoder.setPipeline` validation), so a Metal pipeline without a fragment function is given a
//!   fragment stage that writes nothing, with every target's write mask 0.
//! - `groups` describes every bind group of the pipeline layout: each binding's resource layout and where the encoder
//!   finds the resource ([`Source`]). A source the encoder does not know is reported, never left unbound.
//! - `vertex_buffers` lists the WebGPU vertex buffer slots of native vertex input, each with the Metal buffer index it
//!   reads; vertex pulling reads its buffers through `groups` instead.
//! - `passes` says how draws run: one WebGPU draw in the current pass, or each primitive in its own pass (framebuffer
//!   fetch and raster order groups, plan section 5.9).
//!
//! **Variants.** Metal sets primitive type, culling, winding, depth clipping, depth-stencil state and stencil masks,
//! depth bias and dynamic vertex strides on the encoder; WebGPU fixes them in the pipeline. The encoder computes the
//! [`VariantKey`] of each draw and [`RenderPipeline::variant`] creates the WebGPU pipeline for it once, keeping it for
//! later draws with the same key (plan section 6.5). Variants share the pipeline layout, so bind groups made for one
//! serve all.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::gpu::idl::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferBindingLayout, BufferBindingType, ColorTargetState,
    CompareFunction as GpuCompare, CullMode as GpuCull, DepthStencilState as GpuDepthStencil, FragmentState, FrontFace,
    IndexFormat, MultisampleState, PipelineLayoutDescriptor, PipelineLayoutOrAuto, PrimitiveState, PrimitiveTopology,
    RenderPipelineDescriptor, SamplerBindingLayout, SamplerBindingType, ShaderModuleDescriptor, StencilFaceState,
    StencilOperation as GpuStencilOp, StorageTextureAccess, StorageTextureBindingLayout, TextureBindingLayout,
    TextureFormat, TextureSampleType, TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexState,
    VertexStepMode,
};
use crate::gpu::{Api, Device as _, Float32, Float64, Size64};
use crate::mem::calls::Calls;

use super::error::{RenderError, Result};
use super::metal::TopologyClass;

/// A Metal shader stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetalStage {
    Vertex,
    Fragment,
}

impl MetalStage {
    pub fn name(self) -> &'static str {
        match self {
            MetalStage::Vertex => "vertex",
            MetalStage::Fragment => "fragment",
        }
    }
}

/// What a field of the per-draw uniform block holds. Each value is a `u32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniformValue {
    /// The draw's base vertex (two's complement for a negative value); 0 for non-indexed draws.
    BaseVertex,
    /// The draw's base instance.
    BaseInstance,
    /// Bytes from a buffer binding's aligned base to the Metal offset (WebGPU requires aligned binding offsets).
    ResidualOffset { stage: MetalStage, index: u64 },
    /// Length in bytes of the buffer bound at the slot, from the offset.
    BufferLength { stage: MetalStage, index: u64 },
}

/// One field of the per-draw uniform block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformField {
    /// Byte offset in the block, a multiple of 4.
    pub offset: u64,
    pub value: UniformValue,
}

/// The executor's per-draw uniform block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UniformBlock {
    pub fields: Vec<UniformField>,
    /// Size in bytes.
    pub size: u64,
}

/// Where the encoder finds a binding's resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A buffer argument table slot (`setVertexBuffer:`, `setFragmentBytes:`, ...).
    MetalBuffer { stage: MetalStage, index: u64 },
    /// A texture argument table slot.
    MetalTexture { stage: MetalStage, index: u64 },
    /// A sampler argument table slot.
    MetalSampler { stage: MetalStage, index: u64 },
    /// The executor's per-draw uniform block.
    Uniforms(UniformBlock),
    /// The snapshot of a color attachment taken before each primitive of a framebuffer-fetch draw (plan section
    /// 5.9): a texture of the attachment's format and sample count.
    FetchSnapshot { attachment: u32 },
}

/// A binding's resource layout: one of the `GPUBindGroupLayoutEntry` members.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resource {
    Buffer { ty: BufferBindingType, min_binding_size: u64 },
    Sampler { ty: SamplerBindingType },
    Texture { sample_type: TextureSampleType, view_dimension: TextureViewDimension, multisampled: bool },
    StorageTexture { access: StorageTextureAccess, format: TextureFormat, view_dimension: TextureViewDimension },
}

/// One binding of a bind group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub binding: u32,
    /// `GPUShaderStage` flags.
    pub visibility: u32,
    pub resource: Resource,
    pub source: Source,
}

impl Binding {
    /// The `GPUBindGroupLayoutEntry`.
    pub fn layout_entry(&self) -> BindGroupLayoutEntry {
        let mut e = BindGroupLayoutEntry::new(self.binding, self.visibility);
        match &self.resource {
            Resource::Buffer { ty, min_binding_size } => {
                let mut b = BufferBindingLayout::new();
                b.r#type = *ty;
                b.min_binding_size = Size64::new(*min_binding_size).expect("a binding size below 2^53");
                e.buffer = Some(b);
            }
            Resource::Sampler { ty } => {
                let mut s = SamplerBindingLayout::new();
                s.r#type = *ty;
                e.sampler = Some(s);
            }
            Resource::Texture { sample_type, view_dimension, multisampled } => {
                let mut t = TextureBindingLayout::new();
                t.sample_type = *sample_type;
                t.view_dimension = *view_dimension;
                t.multisampled = *multisampled;
                e.texture = Some(t);
            }
            Resource::StorageTexture { access, format, view_dimension } => {
                let mut t = StorageTextureBindingLayout::new(*format);
                t.access = *access;
                t.view_dimension = *view_dimension;
                e.storage_texture = Some(t);
            }
        }
        e
    }

    /// Can a draw write the resource through this binding?
    pub fn writable(&self) -> bool {
        match &self.resource {
            Resource::Buffer { ty, .. } => *ty == BufferBindingType::Storage,
            Resource::StorageTexture { access, .. } => *access != StorageTextureAccess::ReadOnly,
            _ => false,
        }
    }
}

/// The description of one bind group of the pipeline layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupDesc {
    /// The `@group` index.
    pub index: u32,
    pub bindings: Vec<Binding>,
}

/// One bind group of the pipeline layout.
pub struct Group<A: Api> {
    pub index: u32,
    pub layout: A::BindGroupLayout,
    pub bindings: Vec<Binding>,
}

/// A vertex buffer's stride: fixed in the pipeline, or `MTLBufferLayoutStrideDynamic`, given per binding by
/// `setVertexBuffer:offset:attributeStride:atIndex:`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stride {
    Fixed(u64),
    Dynamic,
}

/// One WebGPU vertex buffer slot of native vertex input.
#[derive(Clone, Debug)]
pub struct VertexBuffer {
    /// The Metal vertex buffer argument table index the slot reads.
    pub metal_index: u64,
    pub stride: Stride,
    pub step_mode: VertexStepMode,
    pub attributes: Vec<VertexAttribute>,
}

/// One Metal color attachment of the pipeline.
#[derive(Clone, Debug)]
pub struct ColorTarget {
    /// `MTLPixelFormat`; 0 (invalid) where the pipeline has no attachment at this index.
    pub pixel_format: u32,
    /// The WebGPU target; `None` exactly when `pixel_format` is invalid.
    pub state: Option<ColorTargetState>,
}

/// A WGSL stage.
#[derive(Clone, Debug)]
pub struct StageSource {
    pub wgsl: String,
    pub entry_point: String,
    /// Pipeline-overridable constants.
    pub constants: Vec<(String, f64)>,
}

/// A compiled stage.
pub struct Stage<A: Api> {
    pub module: A::ShaderModule,
    pub entry_point: String,
    pub constants: Vec<(String, Float64)>,
}

/// What the WebGPU blend constant holds. Metal's `MTLBlendFactorBlendAlpha` and `MTLBlendFactorOneMinusBlendAlpha`
/// weight every channel by the blend color's alpha; WebGPU's `"constant"` weights each channel by its own component of
/// the constant. A pipeline whose color factors name the blend alpha (and never the blend color) takes `"constant"`
/// for them with the constant set to the blend color's alpha in every component, which leaves the alpha channel's
/// `"constant"` factor the blend alpha as well.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendConstant {
    /// The blend color, `(red, green, blue, alpha)`.
    Color,
    /// `(alpha, alpha, alpha, alpha)`.
    AlphaSplat,
}

/// How a pipeline's draws run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PassPlan {
    /// Each draw is one WebGPU draw in the current pass.
    Direct,
    /// Each primitive is drawn in a pass of its own, in primitive order, so that it sees every earlier primitive's
    /// writes: framebuffer fetch reads `fetched` color attachments through [`Source::FetchSnapshot`] bindings, and
    /// raster order groups read and write memory in primitive order (plan section 5.9).
    PerPrimitive { fetched: Vec<u32> },
}

/// What [`RenderPipeline::create`] builds a pipeline from.
#[derive(Clone, Debug)]
pub struct RenderPipelineDesc {
    pub label: String,
    pub vertex: StageSource,
    pub fragment: Option<StageSource>,
    pub groups: Vec<GroupDesc>,
    pub vertex_buffers: Vec<VertexBuffer>,
    pub colors: Vec<ColorTarget>,
    /// `depthAttachmentPixelFormat` and `stencilAttachmentPixelFormat`.
    pub depth_format: u32,
    pub stencil_format: u32,
    /// The WebGPU format of the depth-stencil attachment the pipeline is used with, when either Metal format is set:
    /// the stored format of the format module's resolution of that Metal format.
    pub depth_stencil_format: Option<TextureFormat>,
    /// `rasterSampleCount`.
    pub sample_count: u32,
    pub alpha_to_coverage: bool,
    pub blend_constant: BlendConstant,
    /// `inputPrimitiveTopology`.
    pub topology_class: TopologyClass,
    /// `rasterizationEnabled`. A pipeline the pipeline service creates with it off has a vertex stage without side
    /// effects (vertex-stage writes are refused, `tables/refusals.md` row `vertex-stage-writes`), so its draws have
    /// no observable effect (plan section 6.7).
    pub rasterization: bool,
    pub passes: PassPlan,
    /// The fragment stage writes the depth (`[[depth(...)]]`, WGSL `frag_depth`). Metal stores the written depth
    /// clamped to [0, 1] only, while WebGPU clamps it to the viewport's depth range (WebGPU 23.2.6, "If frag_depth
    /// builtin is produced by the shader"), so its draws need the viewport depth range [0, 1] (AirWgsl
    /// `tables/intrinsics.toml`, `air.depth`, measured on Apple M4).
    pub writes_depth: bool,
}

/// The payload of a `RenderPipelineState` handle.
pub struct RenderPipeline<A: Api> {
    pub label: String,
    pub vertex: Stage<A>,
    pub fragment: Option<Stage<A>>,
    pub layout: A::PipelineLayout,
    pub groups: Vec<Group<A>>,
    pub vertex_buffers: Vec<VertexBuffer>,
    pub colors: Vec<ColorTarget>,
    pub depth_format: u32,
    pub stencil_format: u32,
    pub depth_stencil_format: Option<TextureFormat>,
    pub sample_count: u32,
    pub alpha_to_coverage: bool,
    pub blend_constant: BlendConstant,
    pub topology_class: TopologyClass,
    pub rasterization: bool,
    pub passes: PassPlan,
    /// See [`RenderPipelineDesc::writes_depth`].
    pub writes_depth: bool,
    /// The size of the largest uniform block of the pipeline's bindings (0 without one).
    pub uniform_size: u64,
    variants: RefCell<HashMap<VariantKey, A::RenderPipeline>>,
}

/// The stencil state of one face in a variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilFaceKey {
    pub compare: GpuCompare,
    pub fail: GpuStencilOp,
    pub depth_fail: GpuStencilOp,
    pub pass: GpuStencilOp,
}

impl StencilFaceKey {
    fn gpu(&self) -> StencilFaceState {
        let mut s = StencilFaceState::new();
        s.compare = self.compare;
        s.fail_op = self.fail;
        s.depth_fail_op = self.depth_fail;
        s.pass_op = self.pass;
        s
    }
}

/// The depth-stencil state of a variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DepthStencilKey {
    pub format: TextureFormat,
    /// For formats with a depth aspect.
    pub depth: Option<(bool, GpuCompare)>,
    pub front: StencilFaceKey,
    pub back: StencilFaceKey,
    pub read_mask: u32,
    pub write_mask: u32,
}

/// The encoder state a WebGPU render pipeline fixes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VariantKey {
    pub topology: PrimitiveTopology,
    pub strip_index_format: Option<IndexFormat>,
    pub front_face: FrontFace,
    pub cull_mode: GpuCull,
    pub unclipped_depth: bool,
    pub depth_stencil: Option<DepthStencilKey>,
    /// The array stride of each vertex buffer slot, in slot order.
    pub strides: Vec<u64>,
}

fn gpu_err(op: &'static str) -> impl Fn(crate::gpu::Exception) -> RenderError {
    move |e| RenderError::gpu(op, e)
}

impl<A: Api> RenderPipeline<A> {
    /// Create the shader modules, bind group layouts and pipeline layout of `desc`. Shader compilation errors are
    /// device-timeline errors the caller's error scopes capture.
    pub fn create(g: &impl Calls<A>, desc: RenderPipelineDesc) -> Result<RenderPipeline<A>> {
        let invalid = |what: String| RenderError::invalid(format!("render pipeline {}: {what}", desc.label));
        if !matches!(desc.sample_count, 1 | 4) {
            return Err(invalid(format!("sample count {} has no WebGPU form", desc.sample_count)));
        }
        for (i, c) in desc.colors.iter().enumerate() {
            if (c.pixel_format == 0) != c.state.is_none() {
                return Err(invalid(format!("color attachment {i}: pixel format and target disagree")));
            }
        }
        if (desc.depth_format != 0 || desc.stencil_format != 0) != desc.depth_stencil_format.is_some() {
            return Err(invalid("depth and stencil formats and the WebGPU depth-stencil format disagree".into()));
        }
        if desc.fragment.is_none() && desc.colors.iter().any(|c| c.state.is_some()) {
            return Err(invalid("color targets without a fragment stage".into()));
        }
        let mut uniform_size = 0;
        for group in &desc.groups {
            for b in &group.bindings {
                match (&b.source, &b.resource) {
                    (Source::Uniforms(u), Resource::Buffer { ty, .. }) => {
                        if u.fields.iter().any(|f| f.offset % 4 != 0 || f.offset + 4 > u.size) {
                            return Err(invalid(format!("binding {}: a uniform field outside the block", b.binding)));
                        }
                        if *ty != BufferBindingType::Uniform {
                            return Err(invalid(format!("binding {}: the uniform block is not a uniform", b.binding)));
                        }
                        uniform_size = uniform_size.max(u.size);
                    }
                    (Source::Uniforms(_), _) => return Err(invalid(format!("binding {}: uniforms", b.binding))),
                    (Source::MetalBuffer { .. }, Resource::Buffer { .. })
                    | (Source::MetalTexture { .. }, Resource::Texture { .. } | Resource::StorageTexture { .. })
                    | (Source::MetalSampler { .. }, Resource::Sampler { .. })
                    | (Source::FetchSnapshot { .. }, Resource::Texture { .. }) => {}
                    (s, r) => {
                        return Err(invalid(format!("binding {}: source {s:?} with resource {r:?}", b.binding)));
                    }
                }
                if let Source::FetchSnapshot { attachment } = b.source
                    && !matches!(&desc.passes, PassPlan::PerPrimitive { fetched } if fetched.contains(&attachment))
                {
                    return Err(invalid(format!("binding {}: a snapshot of an attachment not fetched", b.binding)));
                }
            }
        }

        let module = |s: &StageSource, what: &str| -> Stage<A> {
            let mut d = ShaderModuleDescriptor::new(s.wgsl.clone());
            d.label = format!("{} {what}", desc.label);
            Stage {
                module: g.call("GPUDevice.createShaderModule", &d.label, |dev| dev.create_shader_module(&d)),
                entry_point: s.entry_point.clone(),
                constants: s
                    .constants
                    .iter()
                    .map(|(k, v)| (k.clone(), Float64::new(*v).expect("finite pipeline constants")))
                    .collect(),
            }
        };
        let vertex = module(&desc.vertex, "vertex");
        let fragment = desc.fragment.as_ref().map(|f| module(f, "fragment"));

        let mut groups: Vec<Group<A>> = Vec::new();
        for gd in &desc.groups {
            let mut d =
                BindGroupLayoutDescriptor::new(gd.bindings.iter().map(Binding::layout_entry).collect::<Vec<_>>());
            d.label = format!("{} group {}", desc.label, gd.index);
            let layout = g
                .call("GPUDevice.createBindGroupLayout", &d.label, |dev| dev.create_bind_group_layout(&d))
                .map_err(gpu_err("GPUDevice.createBindGroupLayout"))?;
            groups.push(Group { index: gd.index, layout, bindings: gd.bindings.clone() });
        }
        let count = groups.iter().map(|g| g.index + 1).max().unwrap_or(0);
        for index in 0..count {
            match groups.iter().filter(|g| g.index == index).count() {
                0 => {
                    // A group index the shaders do not use gets an empty layout (and an empty bind group at each
                    // draw): wgpu-native does not take the null entries GPUPipelineLayoutDescriptor allows.
                    let mut d = BindGroupLayoutDescriptor::new(Vec::new());
                    d.label = format!("{} group {index} (empty)", desc.label);
                    let layout = g
                        .call("GPUDevice.createBindGroupLayout", &d.label, |dev| dev.create_bind_group_layout(&d))
                        .map_err(gpu_err("GPUDevice.createBindGroupLayout"))?;
                    groups.push(Group { index, layout, bindings: Vec::new() });
                }
                1 => {}
                _ => return Err(invalid(format!("bind group {index} described twice"))),
            }
        }
        groups.sort_by_key(|g| g.index);
        let slots: Vec<Option<&A::BindGroupLayout>> = groups.iter().map(|g| Some(&g.layout)).collect();
        let mut pd = PipelineLayoutDescriptor::new(slots);
        pd.label = desc.label.clone();
        let layout = g.call("GPUDevice.createPipelineLayout", &desc.label, |dev| dev.create_pipeline_layout(&pd));

        Ok(RenderPipeline {
            label: desc.label,
            vertex,
            fragment,
            layout,
            groups,
            vertex_buffers: desc.vertex_buffers,
            colors: desc.colors,
            depth_format: desc.depth_format,
            stencil_format: desc.stencil_format,
            depth_stencil_format: desc.depth_stencil_format,
            sample_count: desc.sample_count,
            alpha_to_coverage: desc.alpha_to_coverage,
            blend_constant: desc.blend_constant,
            topology_class: desc.topology_class,
            rasterization: desc.rasterization,
            passes: desc.passes,
            writes_depth: desc.writes_depth,
            uniform_size,
            variants: RefCell::new(HashMap::new()),
        })
    }

    /// The WebGPU pipeline for `key`, created on first use.
    pub fn variant(&self, g: &impl Calls<A>, key: &VariantKey) -> Result<A::RenderPipeline> {
        if let Some(p) = self.variants.borrow().get(key) {
            return Ok(p.clone());
        }
        let buffers = self
            .vertex_buffers
            .iter()
            .zip(&key.strides)
            .map(|(vb, &stride)| {
                let mut l = VertexBufferLayout::new(
                    Size64::new(stride).map_err(gpu_err("GPUDevice.createRenderPipeline"))?,
                    vb.attributes.clone(),
                );
                l.step_mode = vb.step_mode;
                Ok(Some(l))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut vs = VertexState::new(&self.vertex.module);
        vs.entry_point = Some(self.vertex.entry_point.clone());
        vs.constants = self.vertex.constants.clone();
        vs.buffers = buffers;
        let mut d = RenderPipelineDescriptor::new(PipelineLayoutOrAuto::PipelineLayout(&self.layout), vs);
        d.label = self.label.clone();
        let mut primitive = PrimitiveState::new();
        primitive.topology = key.topology;
        primitive.strip_index_format = key.strip_index_format;
        primitive.front_face = key.front_face;
        primitive.cull_mode = key.cull_mode;
        primitive.unclipped_depth = key.unclipped_depth;
        d.primitive = primitive;
        d.depth_stencil = key.depth_stencil.map(|k| {
            let mut s = GpuDepthStencil::new(k.format);
            if let Some((write, compare)) = k.depth {
                s.depth_write_enabled = Some(write);
                s.depth_compare = Some(compare);
            }
            s.stencil_front = k.front.gpu();
            s.stencil_back = k.back.gpu();
            s.stencil_read_mask = k.read_mask;
            s.stencil_write_mask = k.write_mask;
            s.depth_bias = 0;
            s.depth_bias_slope_scale = Float32::new(0.0).expect("finite");
            s.depth_bias_clamp = Float32::new(0.0).expect("finite");
            s
        });
        let mut ms = MultisampleState::new();
        ms.count = self.sample_count;
        ms.alpha_to_coverage_enabled = self.alpha_to_coverage;
        d.multisample = ms;
        d.fragment = self.fragment.as_ref().map(|f| {
            let mut fs = FragmentState::new(&f.module, self.colors.iter().map(|c| c.state.clone()).collect());
            fs.entry_point = Some(f.entry_point.clone());
            fs.constants = f.constants.clone();
            fs
        });
        let p = g
            .call("GPUDevice.createRenderPipeline", &self.label, |dev| dev.create_render_pipeline(&d))
            .map_err(gpu_err("GPUDevice.createRenderPipeline"))?;
        self.variants.borrow_mut().insert(key.clone(), p.clone());
        Ok(p)
    }

    /// The number of variants created so far.
    pub fn variant_count(&self) -> usize {
        self.variants.borrow().len()
    }
}
