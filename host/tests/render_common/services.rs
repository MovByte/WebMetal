//! Stand-ins for the modules the render path works with but does not own, so that whole command buffers run through
//! the executor as the guest sends them:
//!
//! - [`Pipelines`] serves `RegisterFunction`, `CreateRenderPipeline` and `CreateComputePipeline` from the WGSL
//!   translations of [`super::wgsl`], building the render pipeline payload the render path defines (the pipeline
//!   service's job, plan package H7). It maps only the descriptor fields the replayed cases use and refuses others.
//! - [`Blit`] records `copyFromTexture:...toBuffer:` (with the depth and stencil options) as WebGPU copies, to read
//!   results back (the blit module's job, package H6).
//! - [`Compute`] runs the harness's multisample readback kernels (the compute module's job, package H6).
//!
//! They measure; they are not the modules they stand in for.

use std::any::Any;
use std::collections::HashMap;
use std::rc::Rc;

use webmetal_host::embed::TaskId;
use webmetal_host::exec::{
    ControlService, EncodeCx, EncoderService, Entry, RecordFailure, Segment, Served, ServiceCx, failure,
};
use webmetal_host::format::handling::{self, Resolution};
use webmetal_host::format::webgpu::HostFeatures;
use webmetal_host::gpu::idl::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindingResource, BlendComponent, BlendFactor,
    BlendOperation, BlendState, BufferBinding, BufferDescriptor, BufferUsage, ColorTargetState, ComputePassDescriptor,
    ComputePipelineDescriptor as GpuComputePipelineDescriptor, Extent3D, Extent3DDict, Origin3D, Origin3DDict,
    PipelineLayoutDescriptor, PipelineLayoutOrAuto, ProgrammableStage, ShaderModuleDescriptor, TexelCopyBufferInfo,
    TexelCopyTextureInfo, TextureAspect, TextureViewDescriptor, VertexAttribute, VertexFormat, VertexStepMode,
};
use webmetal_host::gpu::{
    BindingCommands as _, CommandEncoder as _, ComputePassEncoder as _, Device as _, Size64, Texture as _,
};
use webmetal_host::mem::Memory;
use webmetal_host::mem::buffer::Buffer;
use webmetal_host::mem::texture::{Texture, idl_format};
use webmetal_host::render::metal::TopologyClass;
use webmetal_host::render::pipeline::{
    BlendConstant, ColorTarget, GroupDesc, MetalStage, PassPlan, RenderPipeline, RenderPipelineDesc, Source,
    StageSource, Stride, UniformBlock, UniformField, UniformValue, VertexBuffer,
};
use webmetal_wire::*;

use super::wgsl::{self, Func};
use crate::mem_common::{Api, Guest};

type A = Api;
type E = Guest;

/// A registered function: its name.
pub struct FunctionName(pub String);

/// A compute pipeline of the stand-in.
pub struct TestCompute {
    pub pipeline: <A as webmetal_host::gpu::Api>::ComputePipeline,
    pub layout: <A as webmetal_host::gpu::Api>::BindGroupLayout,
    pub func: Func,
}

pub struct Pipelines {
    pub features: HostFeatures,
}

fn payload<T: 'static>(objects: &webmetal_host::exec::Table, h: Handle) -> Option<Rc<T>> {
    objects.get(h)?.payload.clone()?.downcast::<T>().ok()
}

fn blend_factor(v: u64) -> Result<BlendFactor, String> {
    Ok(match v {
        0 => BlendFactor::Zero,
        1 => BlendFactor::One,
        2 => BlendFactor::Src,
        3 => BlendFactor::OneMinusSrc,
        4 => BlendFactor::SrcAlpha,
        5 => BlendFactor::OneMinusSrcAlpha,
        6 => BlendFactor::Dst,
        7 => BlendFactor::OneMinusDst,
        8 => BlendFactor::DstAlpha,
        9 => BlendFactor::OneMinusDstAlpha,
        10 => BlendFactor::SrcAlphaSaturated,
        11 | 13 => BlendFactor::Constant,
        12 | 14 => BlendFactor::OneMinusConstant,
        15 => BlendFactor::Src1,
        16 => BlendFactor::OneMinusSrc1,
        17 => BlendFactor::Src1Alpha,
        18 => BlendFactor::OneMinusSrc1Alpha,
        v => return Err(format!("blend factor {v}")),
    })
}

fn blend_op(v: u64) -> Result<BlendOperation, String> {
    Ok(match v {
        0 => BlendOperation::Add,
        1 => BlendOperation::Subtract,
        2 => BlendOperation::ReverseSubtract,
        3 => BlendOperation::Min,
        4 => BlendOperation::Max,
        v => return Err(format!("blend operation {v}")),
    })
}

/// `MTLColorWriteMask` (red 8, green 4, blue 2, alpha 1) as `GPUColorWrite` (red 1, green 2, blue 4, alpha 8).
fn write_mask(m: u64) -> u32 {
    ((m >> 3) & 1 | ((m >> 2) & 1) << 1 | ((m >> 1) & 1) << 2 | (m & 1) << 3) as u32
}

fn storage_format(value: u64, features: HostFeatures) -> Result<webmetal_host::gpu::idl::TextureFormat, String> {
    match handling::resolve(value as u32, features) {
        Ok(Resolution::Native(f)) => Ok(idl_format(f)),
        other => Err(format!("pixel format {value}: {other:?} (only natively stored formats are replayed)")),
    }
}

impl Pipelines {
    fn render(&self, cx: &mut ServiceCx<'_, A, E>, d: &RenderPipelineDescriptor) -> Result<RenderPipeline<A>, String> {
        let name = |h: Option<Handle>| -> Option<String> {
            h.and_then(|h| payload::<FunctionName>(cx.objects(), h)).map(|f| f.0.clone())
        };
        let vname = name(d.vertex_function).ok_or("no vertex function")?;
        let vfunc = wgsl::function(&vname).ok_or(format!("no translation of {vname}"))?;
        let ffunc = name(d.fragment_function).map(|n| wgsl::function(&n).ok_or(format!("no translation of {n}")));
        let ffunc = ffunc.transpose()?;
        if d.alpha_to_one_enabled {
            return Err("alphaToOne".into());
        }
        // Color attachments, trimmed after the last valid format.
        let last = d.color_attachments.iter().rposition(|c| c.pixel_format != 0).map_or(0, |i| i + 1);
        let mut colors = Vec::new();
        let mut alpha_splat = false;
        for c in &d.color_attachments[..last] {
            if c.pixel_format == 0 {
                colors.push(ColorTarget { pixel_format: 0, state: None });
                continue;
            }
            let mut t = ColorTargetState::new(storage_format(c.pixel_format, self.features)?);
            t.write_mask = write_mask(c.write_mask);
            if c.blending_enabled {
                for f in [c.source_rgb_blend_factor, c.destination_rgb_blend_factor] {
                    alpha_splat |= matches!(f, 13 | 14);
                    if matches!(f, 11 | 12) && alpha_splat {
                        return Err("blend color and blend alpha in the color factors".into());
                    }
                }
                let component = |op: u64, src: u64, dst: u64| -> Result<BlendComponent, String> {
                    let mut b = BlendComponent::new();
                    b.operation = blend_op(op)?;
                    // Minimum and maximum ignore the factors in both APIs (plan section 6.4); WebGPU requires them to
                    // be "one" (GPUBlendComponent validation).
                    if matches!(b.operation, BlendOperation::Min | BlendOperation::Max) {
                        b.src_factor = BlendFactor::One;
                        b.dst_factor = BlendFactor::One;
                    } else {
                        b.src_factor = blend_factor(src)?;
                        b.dst_factor = blend_factor(dst)?;
                    }
                    Ok(b)
                };
                t.blend = Some(BlendState::new(
                    component(c.rgb_blend_operation, c.source_rgb_blend_factor, c.destination_rgb_blend_factor)?,
                    component(c.alpha_blend_operation, c.source_alpha_blend_factor, c.destination_alpha_blend_factor)?,
                ));
            }
            colors.push(ColorTarget { pixel_format: c.pixel_format as u32, state: Some(t) });
        }
        let ds_format = [d.depth_attachment_pixel_format, d.stencil_attachment_pixel_format]
            .into_iter()
            .find(|&f| f != 0)
            .map(|f| storage_format(f, self.features))
            .transpose()?;

        let mut vertex_bindings = vfunc.bindings.clone();
        let mut vwgsl = vfunc.wgsl.clone();
        let mut vertex_buffers = Vec::new();
        if vfunc.stage_in {
            let vd = d.vertex_descriptor.as_ref().ok_or("a stage-in function without a vertex descriptor")?;
            let pulled = vd.layouts.iter().any(|l| l.stride != 0 && (l.step_function == 0 || l.step_rate > 1));
            if pulled {
                // Vertex pulling: rnd_vertex_fetch_two with buffer 1 per vertex and buffer 2 per instance or constant.
                if vname != "rnd_vertex_fetch_two" {
                    return Err(format!("vertex pulling for {vname}"));
                }
                let l2 = &vd.layouts[2];
                let index2 = match l2.step_function {
                    0 => "0u".to_string(),
                    2 => format!("(iid - u.base_instance) / {}u", l2.step_rate),
                    s => return Err(format!("step function {s}")),
                };
                vwgsl = format!(
                    "struct V {{ @builtin(position) position: vec4<f32>, @location(0) @interpolate(flat) data: vec4<u32> }}
                    struct U {{ base_instance: u32, residual1: u32, residual2: u32, pad: u32 }}
                    @group(0) @binding(0) var<storage, read> b1: array<u32>;
                    @group(0) @binding(1) var<storage, read> b2: array<u32>;
                    @group(0) @binding(2) var<uniform> u: U;
                    fn centre(x: u32, y: u32) -> vec4<f32> {{
                        return vec4<f32>((f32(x) + 0.5) / 8.0 - 1.0, 1.0 - (f32(y) + 0.5) / 8.0, 0.0, 1.0);
                    }}
                    @vertex fn main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> V {{
                        let a = (u.residual1 + vid * {s1}u) / 4u;
                        let b = (u.residual2 + ({index2}) * {s2}u) / 4u;
                        return V(centre(vid % 16u, iid % 16u), vec4<u32>(b1[a], b1[a + 1u], b2[b], b2[b + 1u]));
                    }}",
                    s1 = vd.layouts[1].stride,
                    s2 = l2.stride,
                );
                use webmetal_host::gpu::idl::{BufferBindingType, ShaderStage};
                use webmetal_host::render::pipeline::{Binding, Resource};
                let storage = |binding: u32, index: u64| Binding {
                    binding,
                    visibility: ShaderStage::VERTEX,
                    resource: Resource::Buffer { ty: BufferBindingType::ReadOnlyStorage, min_binding_size: 4 },
                    source: Source::MetalBuffer { stage: MetalStage::Vertex, index },
                };
                vertex_bindings = vec![
                    storage(0, 1),
                    storage(1, 2),
                    Binding {
                        binding: 2,
                        visibility: ShaderStage::VERTEX,
                        resource: Resource::Buffer { ty: BufferBindingType::Uniform, min_binding_size: 16 },
                        source: Source::Uniforms(UniformBlock {
                            fields: vec![
                                UniformField { offset: 0, value: UniformValue::BaseInstance },
                                UniformField {
                                    offset: 4,
                                    value: UniformValue::ResidualOffset { stage: MetalStage::Vertex, index: 1 },
                                },
                                UniformField {
                                    offset: 8,
                                    value: UniformValue::ResidualOffset { stage: MetalStage::Vertex, index: 2 },
                                },
                            ],
                            size: 16,
                        }),
                    },
                ];
            } else {
                // Native vertex input: one slot per layout an attribute uses, attributes at their locations.
                let mut slots: Vec<u64> =
                    vd.attributes.iter().filter(|a| a.format != 0).map(|a| a.buffer_index).collect();
                slots.sort_unstable();
                slots.dedup();
                for b in slots {
                    let l = vd.layouts.get(b as usize).ok_or("a layout index")?;
                    let attributes = vd
                        .attributes
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| a.format != 0 && a.buffer_index == b)
                        .map(|(loc, a)| {
                            let format = match a.format {
                                36 => VertexFormat::Uint32,
                                37 => VertexFormat::Uint32x2,
                                38 => VertexFormat::Uint32x3,
                                39 => VertexFormat::Uint32x4,
                                31 => VertexFormat::Float32x4,
                                f => return Err(format!("vertex format {f}")),
                            };
                            Ok(VertexAttribute::new(format, Size64::new(a.offset).unwrap(), loc as u32))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    vertex_buffers.push(VertexBuffer {
                        metal_index: b,
                        stride: if l.stride == u64::MAX { Stride::Dynamic } else { Stride::Fixed(l.stride) },
                        step_mode: if l.step_function == 2 { VertexStepMode::Instance } else { VertexStepMode::Vertex },
                        attributes,
                    });
                }
            }
        }
        let mut groups = Vec::new();
        if !vertex_bindings.is_empty() {
            groups.push(GroupDesc { index: 0, bindings: vertex_bindings });
        }
        let (fragment, passes) = match &ffunc {
            Some(f) => {
                if !f.bindings.is_empty() {
                    groups.push(GroupDesc { index: 1, bindings: f.bindings.clone() });
                }
                let passes = if f.fetch.is_empty() {
                    PassPlan::Direct
                } else {
                    PassPlan::PerPrimitive { fetched: f.fetch.clone() }
                };
                (Some(StageSource { wgsl: f.wgsl.clone(), entry_point: "main".into(), constants: vec![] }), passes)
            }
            None if colors.iter().any(|c| c.state.is_some()) => {
                // No fragment function: an empty fragment stage with every write masked.
                for c in &mut colors {
                    if let Some(s) = &mut c.state {
                        s.write_mask = 0;
                        s.blend = None;
                    }
                }
                (
                    Some(StageSource {
                        wgsl: "@fragment fn main() {}".into(),
                        entry_point: "main".into(),
                        constants: vec![],
                    }),
                    PassPlan::Direct,
                )
            }
            None => (None, PassPlan::Direct),
        };
        if let Some(f) = ffunc.as_ref().and_then(|f| f.feature)
            && !cx.gpu().device().features().contains(f)
        {
            return Err(format!("the host lacks {f}"));
        }
        let writes_depth = fragment.as_ref().is_some_and(|f| f.wgsl.contains("@builtin(frag_depth)"));
        let desc = RenderPipelineDesc {
            label: d.label.clone().unwrap_or_else(|| format!("{vname} pipeline")),
            vertex: StageSource { wgsl: vwgsl, entry_point: "main".into(), constants: vec![] },
            fragment,
            groups,
            vertex_buffers,
            colors,
            depth_format: d.depth_attachment_pixel_format as u32,
            stencil_format: d.stencil_attachment_pixel_format as u32,
            depth_stencil_format: ds_format,
            sample_count: d.raster_sample_count as u32,
            alpha_to_coverage: d.alpha_to_coverage_enabled,
            blend_constant: if alpha_splat { BlendConstant::AlphaSplat } else { BlendConstant::Color },
            topology_class: TopologyClass::from_value(d.input_primitive_topology, "test").map_err(|e| e.to_string())?,
            rasterization: d.rasterization_enabled,
            passes,
            writes_depth,
        };
        RenderPipeline::create(cx.gpu(), desc).map_err(|e| e.to_string())
    }

    fn compute(&self, cx: &mut ServiceCx<'_, A, E>, d: &ComputePipelineDescriptor) -> Result<TestCompute, String> {
        let name = d
            .compute_function
            .and_then(|h| payload::<FunctionName>(cx.objects(), h))
            .map(|f| f.0.clone())
            .ok_or("no compute function")?;
        let func = wgsl::function(&name).ok_or(format!("no translation of {name}"))?;
        let g = cx.gpu();
        let module = g.call("GPUDevice.createShaderModule", &name, |dev| {
            dev.create_shader_module(&ShaderModuleDescriptor::new(func.wgsl.clone()))
        });
        let entries = func.bindings.iter().map(|b| b.layout_entry()).collect::<Vec<_>>();
        let layout = g
            .call("GPUDevice.createBindGroupLayout", &name, |dev| {
                dev.create_bind_group_layout(&BindGroupLayoutDescriptor::new(entries))
            })
            .map_err(|e| e.to_string())?;
        let pl = g.call("GPUDevice.createPipelineLayout", &name, |dev| {
            dev.create_pipeline_layout(&PipelineLayoutDescriptor::new(vec![Some(&layout)]))
        });
        let mut stage = ProgrammableStage::new(&module);
        stage.entry_point = Some("main".into());
        let pipeline = g.call("GPUDevice.createComputePipeline", &name, |dev| {
            dev.create_compute_pipeline(&GpuComputePipelineDescriptor::new(
                PipelineLayoutOrAuto::PipelineLayout(&pl),
                stage,
            ))
        });
        Ok(TestCompute { pipeline, layout, func })
    }
}

impl ControlService<A, E> for Pipelines {
    fn opcodes(&self) -> Vec<u16> {
        vec![opcode::REGISTER_FUNCTION, opcode::CREATE_RENDER_PIPELINE, opcode::CREATE_COMPUTE_PIPELINE]
    }

    fn serve(&mut self, cx: &mut ServiceCx<'_, A, E>, request: &Request) -> Served {
        let fail = |m: String| {
            Served::Ready(
                webmetal_host::exec::schema::failure_reply(request.opcode(), failure::unimplemented(m)).unwrap(),
            )
        };
        match request {
            Request::RegisterFunction(r) => {
                if wgsl::function(&r.name).is_none() {
                    return fail(format!("no translation of {}", r.name));
                }
                cx.attach(Rc::new(FunctionName(r.name.clone())) as Rc<dyn Any>);
                Served::Ready(Reply::RegisterFunction(Ok(RegisterFunctionReply {})))
            }
            Request::CreateRenderPipeline(r) => match self.render(cx, &r.descriptor) {
                Ok(p) => {
                    cx.attach(Rc::new(p) as Rc<dyn Any>);
                    Served::Ready(Reply::CreateRenderPipeline(Ok(CreateRenderPipelineReply {})))
                }
                Err(e) => fail(e),
            },
            Request::CreateComputePipeline(r) => match self.compute(cx, &r.descriptor) {
                Ok(p) => {
                    cx.attach(Rc::new(p) as Rc<dyn Any>);
                    Served::Ready(Reply::CreateComputePipeline(Ok(CreateComputePipelineReply { reflection: None })))
                }
                Err(e) => fail(e),
            },
            _ => fail("not a pipeline request".into()),
        }
    }

    fn release(&mut self, _: TaskId, _: Handle, _: &Entry) {}

    fn task_exited(&mut self, _: TaskId) {}
}

/// `MTLBlitOption` bits used by the readbacks (`MTLBlitCommandEncoder.h`).
const DEPTH_FROM_DEPTH_STENCIL: u64 = 1 << 0;
const STENCIL_FROM_DEPTH_STENCIL: u64 = 1 << 1;

pub struct Blit {
    pub memory: Rc<Memory<A>>,
}

impl Blit {
    #[allow(clippy::too_many_arguments)]
    fn copy(
        &self,
        cx: &mut EncodeCx<'_, A, E>,
        texture: Handle,
        slice: u64,
        level: u64,
        size: &Size,
        buffer: Handle,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
        options: u64,
    ) -> Result<(), String> {
        let t = payload::<Texture<A>>(cx.objects(), texture).ok_or("not a texture")?;
        let b = payload::<Buffer<A>>(cx.objects(), buffer).ok_or("not a buffer")?;
        let encoder = cx.encoder().clone();
        let g = cx.gpu();
        let aspect = match options {
            0 => TextureAspect::All,
            DEPTH_FROM_DEPTH_STENCIL => TextureAspect::DepthOnly,
            STENCIL_FROM_DEPTH_STENCIL => TextureAspect::StencilOnly,
            o => return Err(format!("blit options {o}")),
        };
        let texel = bytes_per_row / size.width;
        let staged_row = (size.width * texel).next_multiple_of(256);
        let rows = size.height;
        let mut d = BufferDescriptor::new(
            Size64::new(staged_row * rows * size.depth).unwrap(),
            BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        );
        d.label = "readback staging".into();
        let staging =
            g.call("GPUDevice.createBuffer", "readback", |dev| dev.create_buffer(&d)).map_err(|e| e.to_string())?;
        let mut src = TexelCopyTextureInfo::<A>::new(t.gpu());
        src.mip_level = level as u32;
        src.origin = Origin3D::Origin3DDict(Origin3DDict { x: 0, y: 0, z: slice as u32 });
        src.aspect = aspect;
        let mut dst = TexelCopyBufferInfo::<A>::new(&staging);
        dst.bytes_per_row = Some(staged_row as u32);
        dst.rows_per_image = Some(rows as u32);
        let extent = Extent3D::Extent3DDict(Extent3DDict {
            width: size.width as u32,
            height: size.height as u32,
            depth_or_array_layers: size.depth as u32,
        });
        g.call("GPUCommandEncoder.copyTextureToBuffer", "readback", |_| {
            encoder.copy_texture_to_buffer(&src, &dst, &extent)
        })
        .map_err(|e| e.to_string())?;
        for image in 0..size.depth {
            for row in 0..rows {
                let from = (image * rows + row) * staged_row;
                let to = offset + image * bytes_per_image + row * bytes_per_row;
                let len = size.width * texel;
                // copyBufferToBuffer moves whole words; the replayed readbacks have rows of whole words.
                if !len.is_multiple_of(4) || !to.is_multiple_of(4) {
                    return Err(format!("row {len} bytes at {to}"));
                }
                g.call("GPUCommandEncoder.copyBufferToBuffer", "readback", |_| {
                    encoder.copy_buffer_to_buffer_with_offsets(
                        &staging,
                        Size64::new(from).unwrap(),
                        b.gpu(),
                        Size64::new(to).unwrap(),
                        Some(Size64::new(len).unwrap()),
                    )
                });
            }
        }
        self.memory.note_buffer_write(&b, Some((offset, offset + bytes_per_image * size.depth.max(1))));
        Ok(())
    }
}

impl EncoderService<A, E> for Blit {
    fn encode(&mut self, cx: &mut EncodeCx<'_, A, E>, segment: &Segment) -> Result<(), RecordFailure> {
        let records = cx.records();
        for &i in &segment.records {
            let (offset, record) = &records[i];
            let fail = |m: String| RecordFailure { offset: *offset as u64, failure: failure::unimplemented(m) };
            match record {
                Command::BeginBlit(_) | Command::EncoderEndEncoding(_) => {}
                Command::BlitCopyTextureToBuffer(r) => self
                    .copy(
                        cx,
                        r.source_texture.unwrap(),
                        r.source_slice,
                        r.source_level,
                        &r.source_size,
                        r.destination_buffer.unwrap(),
                        r.destination_offset,
                        r.destination_bytes_per_row,
                        r.destination_bytes_per_image,
                        0,
                    )
                    .map_err(fail)?,
                Command::BlitCopyTextureToBufferOptions(r) => self
                    .copy(
                        cx,
                        r.source_texture.unwrap(),
                        r.source_slice,
                        r.source_level,
                        &r.source_size,
                        r.destination_buffer.unwrap(),
                        r.destination_offset,
                        r.destination_bytes_per_row,
                        r.destination_bytes_per_image,
                        r.options,
                    )
                    .map_err(fail)?,
                other => return Err(fail(format!("blit record {:#06x}", other.opcode()))),
            }
        }
        Ok(())
    }
}

pub struct Compute {
    pub memory: Rc<Memory<A>>,
}

impl EncoderService<A, E> for Compute {
    fn encode(&mut self, cx: &mut EncodeCx<'_, A, E>, segment: &Segment) -> Result<(), RecordFailure> {
        let records = cx.records();
        let mut pipeline: Option<Rc<TestCompute>> = None;
        let mut textures: HashMap<u64, Rc<Texture<A>>> = HashMap::new();
        let mut buffers: HashMap<u64, Rc<Buffer<A>>> = HashMap::new();
        for &i in &segment.records {
            let (offset, record) = &records[i];
            let fail = |m: String| RecordFailure { offset: *offset as u64, failure: failure::unimplemented(m) };
            match record {
                Command::BeginCompute(_) | Command::EncoderEndEncoding(_) => {}
                Command::ComputeSetComputePipelineState(r) => {
                    pipeline = r.state.and_then(|h| payload::<TestCompute>(cx.objects(), h));
                }
                Command::ComputeSetTexture(r) => {
                    textures.insert(r.index, payload::<Texture<A>>(cx.objects(), r.texture.unwrap()).unwrap());
                }
                Command::ComputeSetBuffer(r) => {
                    assert_eq!(r.offset, 0);
                    buffers.insert(r.index, payload::<Buffer<A>>(cx.objects(), r.buffer.unwrap()).unwrap());
                }
                Command::ComputeDispatchThreadgroups(r) => {
                    let p = pipeline.clone().ok_or_else(|| fail("no pipeline".into()))?;
                    let views: Vec<_> = p
                        .func
                        .bindings
                        .iter()
                        .filter_map(|b| match &b.source {
                            Source::MetalTexture { index, .. } => {
                                let t = &textures[index];
                                let mut d = TextureViewDescriptor::new();
                                d.aspect = if t.gpu().format().as_str().contains("stencil") {
                                    TextureAspect::DepthOnly
                                } else {
                                    TextureAspect::All
                                };
                                let view =
                                    cx.gpu().call("GPUTexture.createView", "readback", |_| t.gpu().create_view(&d));
                                Some((b.binding, view.unwrap()))
                            }
                            _ => None,
                        })
                        .collect();
                    let mut entries: Vec<BindGroupEntry<'_, A>> = Vec::new();
                    for (binding, v) in &views {
                        entries.push(BindGroupEntry::new(*binding, BindingResource::TextureView(v)));
                    }
                    for b in &p.func.bindings {
                        if let Source::MetalBuffer { index, .. } = &b.source {
                            let buf = &buffers[index];
                            self.memory.note_buffer_write(buf, None);
                            entries.push(BindGroupEntry::new(
                                b.binding,
                                BindingResource::BufferBinding(BufferBinding::new(buf.gpu())),
                            ));
                        }
                    }
                    let encoder = cx.encoder().clone();
                    let g = cx.gpu();
                    let bg = g.call("GPUDevice.createBindGroup", "readback", |dev| {
                        dev.create_bind_group(&BindGroupDescriptor::new(&p.layout, entries))
                    });
                    let pass = g
                        .call("GPUCommandEncoder.beginComputePass", "readback", |_| {
                            encoder.begin_compute_pass(&ComputePassDescriptor::new())
                        })
                        .map_err(|e| fail(e.to_string()))?;
                    g.call("GPUComputePassEncoder.setPipeline", "readback", |_| pass.set_pipeline(&p.pipeline));
                    g.call("GPUComputePassEncoder.setBindGroup", "readback", |_| {
                        pass.set_bind_group(0, Some(&bg), &[])
                    });
                    let t = r.threadgroups_per_grid.clone();
                    g.call("GPUComputePassEncoder.dispatchWorkgroups", "readback", |_| {
                        pass.dispatch_workgroups(t.width as u32, t.height as u32, t.depth as u32)
                    });
                    g.call("GPUComputePassEncoder.end", "readback", |_| pass.end());
                }
                other => return Err(fail(format!("compute record {:#06x}", other.opcode()))),
            }
        }
        Ok(())
    }
}
