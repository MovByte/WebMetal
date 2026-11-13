//! The render encoder service: one render or parallel render encoder's records, recorded as WebGPU render passes.
//!
//! **Before the pass.** The segment is read once ahead ([`scan`]) for what must be known before the first WebGPU pass
//! begins: the final store actions (`set*StoreAction:` records finalize `MTLStoreActionUnknown`), the textures the
//! pass binds (brought up to date by the memory module, which copies and so cannot run inside a pass), the indexed
//! draws of list topologies (index staging, `index`), the size of the encoder's arena of inline bytes and per-draw
//! uniforms, and whether the Metal pass is split into several WebGPU passes.
//!
//! **State.** Metal's encoder state is replayed record by record. What WebGPU fixes in the pipeline (primitive
//! topology, cull mode, winding, depth clip mode, depth-stencil state, stencil masks, dynamic vertex strides) selects a
//! pipeline variant per draw (`pipeline`); viewport, scissor rectangle, blend color and stencil reference are WebGPU
//! pass state, applied before each draw that sees a change. A parallel render encoder's sub-encoders arrive spliced in
//! creation order, which is the order Metal executes them in ("The rendering commands ... are executed in the order in
//! which the MTLRenderCommandEncoder objects are created", reference case `pass.parallel-encoder-order`); each starts
//! from the default state, as a new render command encoder does, and all draw in one WebGPU pass.
//!
//! **Separate stencil faces.** Metal has a reference value and read and write masks per face; WebGPU has one of each
//! for both faces. When the faces a draw rasterizes need different values, the draw is made one primitive at a time,
//! each primitive twice: culling back faces with the front face's values, then culling front faces with the back
//! face's. A triangle is either front- or back-facing, so exactly one of the two draws rasterizes it and primitive
//! order is kept (plan section 6.4). WebGPU does not say which face a point or line uses (section 23.3.5), nor does
//! Metal, so points and lines with differing faces are not drawn.
//!
//! **Splits.** A WebGPU pass is ended and a new one begun (loading the attachments) at a memory barrier after the
//! vertex stage, and around each primitive of a pipeline whose pass plan is per-primitive (framebuffer fetch, plan
//! section 5.9); `attachments` describes how the store and resolve actions then apply.
//!
//! **Views.** A texture view binds as a WebGPU view of its root texture over the view's levels and slices, in its
//! format ([`crate::mem::view`]); its writes are noted on the root's subresources it covers. A view in a format the
//! root's stored texture cannot be viewed in binds through the root's alias, brought up to date before the pass and
//! noted written in its place; a pass that writes subresources through the one and uses them through the other is
//! refused.

use std::collections::HashMap;
use std::rc::Rc;

use webmetal_wire::{Command, Handle, RenderPassDescriptor, Viewport};

use crate::embed::Embedder;
use crate::exec::{EncodeCx, EncoderKind, EncoderService, RecordFailure, Segment, Table, failure};
use crate::format::handling::{Read, Resolution};
use crate::format::webgpu::Feature;
use crate::gpu::idl::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, BufferBinding, BufferBindingType, BufferDescriptor,
    BufferUsage, Color, ColorDict, Extent3D, Extent3DDict, IndexFormat, Origin3D, Origin3DDict, TexelCopyTextureInfo,
    TextureAspect, TextureDescriptor, TextureUsage, TextureViewDescriptor, TextureViewDimension,
};
use crate::gpu::{
    Api, BindingCommands as _, Buffer as _, CommandEncoder as _, DebugCommands as _, Device as _, Float32, Float64,
    Queue as _, RenderCommands as _, RenderPassEncoder as _, Size64, Texture as _,
};
use crate::mem::buffer::Buffer;
use crate::mem::texture::{Prepare, Texture, usage as texture_usage};
use crate::mem::view::{IDENTITY_SWIZZLE, PassCopies, read_swizzle, swizzle_string};
use crate::mem::{MemError, Memory};
use crate::state::SamplerObject;

use super::attachments::{FinalStores, Targets, Tex};
use super::depth_stencil::{DepthStencilState, Face};
use super::error::{RenderError, Result};
use super::helpers::{Helpers, size64};
use super::index::{self, StageSource, Staged, StagingLimits};
use super::metal::{
    CullMode, DepthClipMode, FillMode, IndexType, PrimitiveType, TopologyClass, VisibilityMode, Winding, barrier_scope,
    stages,
};
use super::pipeline::{
    BindingKey, BlendConstant, DepthStencilKey, MetalStage, PassPlan, RenderPipeline, RenderPipelineState, Resource,
    Source, StencilFaceKey, Stride, UniformValue, VariantKey,
};
use super::visibility::Visibility;
use crate::compute::pipeline::{SamplerSlot, TextureSlot};

/// The render encoder service.
pub struct RenderEncoder<A: Api> {
    memory: Rc<Memory<A>>,
    helpers: Rc<Helpers<A>>,
    tessellator: Rc<crate::tess::Tessellator<A>>,
}

impl<A: Api> RenderEncoder<A> {
    pub fn new(memory: Rc<Memory<A>>) -> RenderEncoder<A> {
        RenderEncoder { memory, helpers: Rc::new(Helpers::default()), tessellator: Rc::default() }
    }
}

type Fail = (u64, RenderError);

fn at(offset: u64) -> impl Fn(RenderError) -> Fail {
    move |e| (offset, e)
}

fn mem_at(offset: u64) -> impl Fn(MemError) -> Fail {
    move |e| (offset, RenderError::Mem(e))
}

fn payload<T: 'static>(objects: &Table, handle: Handle) -> Option<Rc<T>> {
    objects.get(handle)?.payload.clone()?.downcast::<T>().ok()
}

fn buffer<A: Api>(objects: &Table, handle: Handle, call: &str) -> Result<Rc<Buffer<A>>> {
    payload::<Buffer<A>>(objects, handle).ok_or_else(|| {
        RenderError::unimplemented(format!("{call}: buffer {:#x} has no storage the render path knows", handle.get()))
    })
}

fn pipeline<A: Api>(objects: &Table, handle: Handle) -> Result<Rc<RenderPipelineState<A>>> {
    payload::<RenderPipelineState<A>>(objects, handle).ok_or_else(|| {
        RenderError::unimplemented(format!(
            "render pipeline {:#x} has no WebGPU pipeline (the pipeline service did not create one)",
            handle.get()
        ))
    })
}

fn u32_of(v: u64, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| RenderError::unimplemented(format!("{what} {v} does not fit WebGPU's 32-bit count")))
}

/// The device limits and features the encoder checks against.
#[derive(Clone, Copy, Debug)]
struct Limits {
    storage_align: u64,
    uniform_align: u64,
    max_uniform_binding: u64,
    max_storage_binding: u64,
    max_texture_2d: f64,
    max_vertex_stride: u64,
    max_buffer: u64,
    depth_clip_control: bool,
    indirect_first_instance: bool,
}

/// One draw record.
#[derive(Clone, Debug)]
enum DrawKind {
    Direct {
        start: u64,
        count: u64,
        instances: u64,
        base_instance: u64,
    },
    Indexed {
        count: u64,
        index_type: u64,
        buffer: Option<Handle>,
        offset: u64,
        instances: u64,
        base_vertex: i64,
        base_instance: u64,
    },
    Indirect {
        buffer: Option<Handle>,
        offset: u64,
    },
    IndexedIndirect {
        index_type: u64,
        buffer: Option<Handle>,
        offset: u64,
        indirect: Option<Handle>,
        indirect_offset: u64,
    },
    /// `drawPatches` and `drawIndexedPatches`: tessellated before the pass (`crate::tess`) and drawn as the
    /// generated triangles.
    Patches(crate::tess::PatchDraw),
    /// The indirect patch draws, which the A10X does not support.
    PatchesIndirect,
}

fn draw_of(c: &Command) -> Option<(u64, DrawKind)> {
    Some(match c {
        Command::RenderDrawPrimitives(r) => (
            r.primitive_type,
            DrawKind::Direct { start: r.vertex_start, count: r.vertex_count, instances: 1, base_instance: 0 },
        ),
        Command::RenderDrawPrimitivesInstanced(r) => (
            r.primitive_type,
            DrawKind::Direct {
                start: r.vertex_start,
                count: r.vertex_count,
                instances: r.instance_count,
                base_instance: 0,
            },
        ),
        Command::RenderDrawPrimitivesInstancedBaseInstance(r) => (
            r.primitive_type,
            DrawKind::Direct {
                start: r.vertex_start,
                count: r.vertex_count,
                instances: r.instance_count,
                base_instance: r.base_instance,
            },
        ),
        Command::RenderDrawIndexedPrimitives(r) => (
            r.primitive_type,
            DrawKind::Indexed {
                count: r.index_count,
                index_type: r.index_type,
                buffer: r.index_buffer,
                offset: r.index_buffer_offset,
                instances: 1,
                base_vertex: 0,
                base_instance: 0,
            },
        ),
        Command::RenderDrawIndexedPrimitivesInstanced(r) => (
            r.primitive_type,
            DrawKind::Indexed {
                count: r.index_count,
                index_type: r.index_type,
                buffer: r.index_buffer,
                offset: r.index_buffer_offset,
                instances: r.instance_count,
                base_vertex: 0,
                base_instance: 0,
            },
        ),
        Command::RenderDrawIndexedPrimitivesInstancedBaseVertexBaseInstance(r) => (
            r.primitive_type,
            DrawKind::Indexed {
                count: r.index_count,
                index_type: r.index_type,
                buffer: r.index_buffer,
                offset: r.index_buffer_offset,
                instances: r.instance_count,
                base_vertex: r.base_vertex,
                base_instance: r.base_instance,
            },
        ),
        Command::RenderDrawPrimitivesIndirect(r) => {
            (r.primitive_type, DrawKind::Indirect { buffer: r.indirect_buffer, offset: r.indirect_buffer_offset })
        }
        // A patch draw rasterizes the tessellator's triangles (MTLPrimitiveTypeTriangle).
        Command::RenderDrawPatches(r) => (3, DrawKind::Patches(crate::tess::PatchDraw::of(r))),
        Command::RenderDrawIndexedPatches(r) => (3, DrawKind::Patches(crate::tess::PatchDraw::of_indexed(r))),
        Command::RenderDrawPatchesIndirect(_) | Command::RenderDrawIndexedPatchesIndirect(_) => {
            (3, DrawKind::PatchesIndirect)
        }
        Command::RenderDrawIndexedPrimitivesIndirect(r) => (
            r.primitive_type,
            DrawKind::IndexedIndirect {
                index_type: r.index_type,
                buffer: r.index_buffer,
                offset: r.index_buffer_offset,
                indirect: r.indirect_buffer,
                indirect_offset: r.indirect_buffer_offset,
            },
        ),
        _ => return None,
    })
}

const DRAW: &str = "-[MTLRenderCommandEncoder draw...]";

/// The tessellation bindings of a patch draw.
struct PatchBinding<A: Api> {
    /// The draw's point records.
    vertices: A::Buffer,
    /// Its control-point index buffer's binding (`drawIndexedPatches`).
    indices: Option<crate::tess::IndexWindow<A>>,
}

/// An indexed list draw to stage before the pass.
struct Job<A: Api> {
    draw: usize,
    offset: u64,
    index_buffer: Rc<Buffer<A>>,
    index_offset: u64,
    index_type: IndexType,
    per_primitive: u32,
    source: StageSource<A>,
}

/// What the segment needs before its first WebGPU pass.
struct Scan<A: Api> {
    stores: FinalStores,
    textures: Vec<(u64, Handle)>,
    draws: usize,
    bytes_size: u64,
    uniform_stride: u64,
    spans: u32,
    splits: bool,
    jobs: Vec<Job<A>>,
    /// Base vertex and base instance of indirect draws, copied from the indirect arguments into the draw's uniform
    /// block: (draw, source buffer, source offset, offset in the block).
    copies: Vec<(usize, Rc<Buffer<A>>, u64, u64)>,
    /// The patch draws to tessellate before the pass: (draw, record offset, job).
    tess: Vec<(usize, u64, crate::tess::Job<A>)>,
}

fn scan<A: Api>(items: &[(u64, &Command)], objects: &Table, uniform_align: u64) -> std::result::Result<Scan<A>, Fail> {
    let mut s = Scan {
        stores: FinalStores::default(),
        textures: Vec::new(),
        draws: 0,
        bytes_size: 0,
        uniform_stride: 0,
        spans: 0,
        splits: false,
        jobs: Vec::new(),
        copies: Vec::new(),
        tess: Vec::new(),
    };
    let mut current: Option<Rc<RenderPipeline<A>>> = None;
    let mut tess = crate::tess::EncoderState::<A>::default();
    let mut max_uniform = 0u64;
    let set_color = |s: &mut Scan<A>, index: u64, action: u64| {
        let i = index as usize;
        if s.stores.colors.len() <= i {
            s.stores.colors.resize(i + 1, None);
        }
        s.stores.colors[i] = Some(action);
    };
    for &(offset, c) in items {
        let bytes = |len: u64| len.max(1).next_multiple_of(256);
        match c {
            Command::BeginParallelSubEncoder(_) => {
                current = None;
                tess = crate::tess::EncoderState::default();
            }
            Command::RenderSetTessellationFactorBuffer(r) => {
                tess.factors = match r.buffer {
                    Some(h) => Some((
                        buffer::<A>(objects, h, "-[MTLRenderCommandEncoder setTessellationFactorBuffer:...]")
                            .map_err(at(offset))?,
                        r.offset,
                        r.instance_stride,
                    )),
                    None => None,
                };
            }
            Command::RenderSetTessellationFactorScale(r) => tess.scale = Some(r.scale),
            Command::RenderSetRenderPipelineState(r) => {
                current = r
                    .pipeline_state
                    .and_then(|h| payload::<RenderPipelineState<A>>(objects, h))
                    .map(|p| p.creation.clone());
                if let Some(p) = &current {
                    max_uniform = max_uniform.max(p.uniform_size);
                }
            }
            Command::RenderSetColorStoreAction(r) => set_color(&mut s, r.color_attachment_index, r.store_action),
            Command::ParallelSetColorStoreAction(r) => set_color(&mut s, r.color_attachment_index, r.store_action),
            Command::RenderSetDepthStoreAction(r) => s.stores.depth = Some(r.store_action),
            Command::ParallelSetDepthStoreAction(r) => s.stores.depth = Some(r.store_action),
            Command::RenderSetStencilStoreAction(r) => s.stores.stencil = Some(r.store_action),
            Command::ParallelSetStencilStoreAction(r) => s.stores.stencil = Some(r.store_action),
            Command::RenderSetVertexTexture(r) => s.textures.extend(r.texture.map(|h| (offset, h))),
            Command::RenderSetFragmentTexture(r) => s.textures.extend(r.texture.map(|h| (offset, h))),
            Command::RenderSetVertexTextures(r) => s.textures.extend(r.textures.iter().flatten().map(|&h| (offset, h))),
            Command::RenderSetFragmentTextures(r) => {
                s.textures.extend(r.textures.iter().flatten().map(|&h| (offset, h)))
            }
            Command::RenderSetFragmentTextureAndSamplerState(r) => s.textures.extend(r.texture.map(|h| (offset, h))),
            Command::RenderSetVertexBytes(r) => s.bytes_size += bytes(r.length),
            Command::RenderSetVertexBytesAttributeStride(r) => s.bytes_size += bytes(r.length),
            Command::RenderSetFragmentBytes(r) => s.bytes_size += bytes(r.length),
            Command::RenderSetVisibilityResultMode(_) => s.spans += 1,
            Command::RenderMemoryBarrierWithScope(r) if splits_pass(r.after, r.before) => s.splits = true,
            Command::RenderMemoryBarrierWithResources(r) if splits_pass(r.after, r.before) => s.splits = true,
            c => {
                if let Some((ty, kind)) = draw_of(c) {
                    let ordinal = s.draws;
                    s.draws += 1;
                    let prim = PrimitiveType::from_value(ty, DRAW).map_err(at(offset))?;
                    if let Some(p) = &current
                        && matches!(p.passes, PassPlan::PerPrimitive { .. })
                    {
                        s.splits = true;
                    }
                    scan_draw(objects, &mut s, current.as_deref(), ordinal, offset, prim, &kind).map_err(at(offset))?;
                    let plan = current.as_deref().map(|p| &p.passes);
                    let job = match &kind {
                        DrawKind::Patches(d) => crate::tess::job(objects, plan, &tess, d, ordinal),
                        DrawKind::PatchesIndirect => Err(crate::tess::indirect_patch_draw()),
                        _ => crate::tess::primitive_draw(plan).map(|()| None),
                    };
                    s.tess.extend(job.map_err(at(offset))?.map(|j| (ordinal, offset, j)));
                }
            }
        }
    }
    s.uniform_stride = if max_uniform == 0 { 0 } else { max_uniform.next_multiple_of(uniform_align) };
    Ok(s)
}

fn scan_draw<A: Api>(
    objects: &Table,
    s: &mut Scan<A>,
    current: Option<&RenderPipeline<A>>,
    ordinal: usize,
    offset: u64,
    prim: PrimitiveType,
    kind: &DrawKind,
) -> Result<()> {
    let uniform_field = |value: UniformValue| -> Option<u64> {
        current?.groups.iter().flat_map(|g| &g.bindings).find_map(|b| match &b.source {
            Source::Uniforms(u) => u.fields.iter().find(|f| f.value == value).map(|f| f.offset),
            _ => None,
        })
    };
    let index = |ty: u64, handle: Option<Handle>, offset: u64| -> Result<(IndexType, Rc<Buffer<A>>)> {
        let it = IndexType::from_value(ty, DRAW)?;
        let h = handle.ok_or_else(|| RenderError::undefined(DRAW, "a nil index buffer"))?;
        let b = buffer::<A>(objects, h, DRAW)?;
        if !offset.is_multiple_of(it.size()) {
            return Err(RenderError::undefined(
                DRAW,
                format!(
                    "index buffer offset {offset} is not a multiple of the index size {} (MTLRenderCommandEncoder.h: \
                     \"indexBufferOffset must be a multiple of the index size\"); WebGPU requires it too",
                    it.size()
                ),
            ));
        }
        Ok((it, b))
    };
    match kind {
        DrawKind::Indexed { count, index_type, buffer: h, offset: io, instances, base_vertex, base_instance } => {
            let (it, b) = index(*index_type, *h, *io)?;
            if !prim.is_strip() {
                s.jobs.push(Job {
                    draw: ordinal,
                    offset,
                    index_buffer: b,
                    index_offset: *io,
                    index_type: it,
                    per_primitive: prim.vertices_per_primitive(),
                    source: StageSource::Direct {
                        count: u32_of(*count, "index count")?,
                        instances: u32_of(*instances, "instance count")?,
                        base_vertex: i32::try_from(*base_vertex).map_err(|_| {
                            RenderError::unimplemented(format!("base vertex {base_vertex} outside WebGPU's i32"))
                        })?,
                        base_instance: u32_of(*base_instance, "base instance")?,
                    },
                });
            }
        }
        DrawKind::IndexedIndirect { index_type, buffer: h, offset: io, indirect, indirect_offset } => {
            let (it, b) = index(*index_type, *h, *io)?;
            let ib = buffer::<A>(
                objects,
                indirect.ok_or_else(|| RenderError::undefined(DRAW, "a nil indirect buffer"))?,
                DRAW,
            )?;
            check_indirect(&ib, *indirect_offset, 20)?;
            if !prim.is_strip() {
                s.jobs.push(Job {
                    draw: ordinal,
                    offset,
                    index_buffer: b,
                    index_offset: *io,
                    index_type: it,
                    per_primitive: prim.vertices_per_primitive(),
                    source: StageSource::Indirect { buffer: ib.clone(), offset: *indirect_offset },
                });
            }
            if let Some(f) = uniform_field(UniformValue::BaseVertex) {
                s.copies.push((ordinal, ib.clone(), indirect_offset + 12, f));
            }
            if let Some(f) = uniform_field(UniformValue::BaseInstance) {
                s.copies.push((ordinal, ib, indirect_offset + 16, f));
            }
        }
        DrawKind::Indirect { buffer: h, offset: io } => {
            let ib =
                buffer::<A>(objects, h.ok_or_else(|| RenderError::undefined(DRAW, "a nil indirect buffer"))?, DRAW)?;
            check_indirect(&ib, *io, 16)?;
            if let Some(f) = uniform_field(UniformValue::BaseInstance) {
                s.copies.push((ordinal, ib, io + 12, f));
            }
        }
        // Patch draws: crate::tess::job.
        DrawKind::Direct { .. } | DrawKind::Patches(_) | DrawKind::PatchesIndirect => {}
    }
    Ok(())
}

fn check_indirect<A: Api>(b: &Buffer<A>, offset: u64, size: u64) -> Result<()> {
    if !offset.is_multiple_of(4) {
        return Err(RenderError::undefined(
            DRAW,
            format!("indirect buffer offset {offset} is not a multiple of 4 (MTLRenderCommandEncoder.h)"),
        ));
    }
    if offset + size > b.length {
        return Err(RenderError::undefined(
            DRAW,
            format!("indirect arguments at {offset}..{} of a {}-byte buffer", offset + size, b.length),
        ));
    }
    Ok(())
}

/// A buffer argument table entry.
#[derive(Clone)]
enum Bound<A: Api> {
    Buffer {
        buffer: Rc<Buffer<A>>,
        offset: u64,
        stride: Option<u64>,
    },
    /// `set*Bytes:`: a range of the encoder's arena.
    Bytes {
        offset: u64,
        len: u64,
        stride: Option<u64>,
    },
}

/// The argument tables of one stage.
struct Slots<A: Api> {
    buffers: HashMap<u64, Bound<A>>,
    textures: HashMap<u64, Tex<A>>,
    samplers: HashMap<u64, Rc<SamplerObject<A>>>,
}

impl<A: Api> Default for Slots<A> {
    fn default() -> Self {
        Slots { buffers: HashMap::new(), textures: HashMap::new(), samplers: HashMap::new() }
    }
}

/// A render command encoder's state.
struct State<A: Api> {
    pipeline: Option<Rc<RenderPipelineState<A>>>,
    depth_stencil: DepthStencilState,
    cull: CullMode,
    winding: Winding,
    clip: DepthClipMode,
    fill: FillMode,
    bias: (f32, f32, f32),
    viewport: [f32; 6],
    scissor: [u32; 4],
    blend: [f32; 4],
    refs: (u32, u32),
    vertex: Slots<A>,
    fragment: Slots<A>,
}

impl<A: Api> State<A> {
    /// A new render command encoder's state: no pipeline, the default depth-stencil state, no culling, clockwise
    /// front faces, depth clipping, filled triangles, no depth bias, a viewport and scissor rectangle covering the
    /// render target, blend color and stencil references 0 (Metal's documented defaults for each setter).
    fn new(width: u32, height: u32) -> State<A> {
        State {
            pipeline: None,
            depth_stencil: DepthStencilState::DEFAULT,
            cull: CullMode::None,
            winding: Winding::Clockwise,
            clip: DepthClipMode::Clip,
            fill: FillMode::Fill,
            bias: (0.0, 0.0, 0.0),
            viewport: [0.0, 0.0, width as f32, height as f32, 0.0, 1.0],
            scissor: [0, 0, width, height],
            blend: [0.0; 4],
            refs: (0, 0),
            vertex: Slots::default(),
            fragment: Slots::default(),
        }
    }

    fn slots(&mut self, stage: MetalStage) -> &mut Slots<A> {
        match stage {
            MetalStage::Vertex => &mut self.vertex,
            MetalStage::Fragment => &mut self.fragment,
        }
    }

    fn slots_ref(&self, stage: MetalStage) -> &Slots<A> {
        match stage {
            MetalStage::Vertex => &self.vertex,
            MetalStage::Fragment => &self.fragment,
        }
    }
}

/// Pass state set on the current WebGPU pass.
#[derive(Default)]
struct Applied {
    pipeline: Option<(usize, VariantKey)>,
    viewport: Option<[u32; 6]>,
    scissor: Option<[u32; 4]>,
    blend: Option<[u32; 4]>,
    reference: Option<u32>,
}

/// The stencil values WebGPU takes once for both faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FaceValues {
    reference: u32,
    read_mask: u32,
    write_mask: u32,
}

#[derive(Clone, Copy, Debug)]
enum FacePlan {
    One(FaceValues),
    /// Draw front faces with the first values, back faces with the second.
    Split(FaceValues, FaceValues),
}

/// The stencil values of a draw. `cull`: the faces Metal discards; `triangles`: whether the primitives have a facing.
fn face_plan(ds: &DepthStencilState, refs: (u32, u32), cull: CullMode, triangles: bool) -> Result<FacePlan> {
    let (front, back) = (ds.front_face(), ds.back_face());
    let values = |f: &Face, reference: u32| FaceValues { reference, read_mask: f.read_mask, write_mask: f.write_mask };
    if !triangles {
        if front != back || (front.uses_reference() && refs.0 != refs.1) {
            return Err(RenderError::unimplemented(
                "a point or line draw whose front and back stencil state or references differ: neither Metal nor \
                 WebGPU (section 23.3.5) defines which face a point or line uses",
            ));
        }
        return Ok(FacePlan::One(values(&front, refs.0)));
    }
    let drawn: Vec<(Face, u32)> = match cull {
        CullMode::None => vec![(front, refs.0), (back, refs.1)],
        CullMode::Front => vec![(back, refs.1)],
        CullMode::Back => vec![(front, refs.0)],
    };
    let agree = |used: fn(&Face) -> bool, value: fn(&Face, u32) -> u32| -> Option<u32> {
        let mut v = None;
        for (f, r) in &drawn {
            if used(f) {
                let x = value(f, *r);
                if v.is_some_and(|y| y != x) {
                    return None;
                }
                v = Some(x);
            }
        }
        Some(v.unwrap_or_else(|| value(&drawn[0].0, drawn[0].1)))
    };
    let reference = agree(Face::uses_reference, |_, r| r);
    let read_mask = agree(Face::uses_read_mask, |f, _| f.read_mask);
    let write_mask = agree(Face::uses_write_mask, |f, _| f.write_mask);
    match (reference, read_mask, write_mask) {
        (Some(reference), Some(read_mask), Some(write_mask)) => {
            Ok(FacePlan::One(FaceValues { reference, read_mask, write_mask }))
        }
        _ => Ok(FacePlan::Split(values(&front, refs.0), values(&back, refs.1))),
    }
}

/// The encoder's arena: inline bytes, then one uniform block per draw.
struct Arena<A: Api> {
    buffer: Option<A::Buffer>,
    mirror: Vec<u8>,
    next_bytes: u64,
    uniform_base: u64,
    uniform_stride: u64,
}

/// A buffer and a byte range of it.
type BufferRange<A> = (Rc<Buffer<A>>, (u64, u64));

/// A WebGPU buffer binding: buffer, offset and size.
type GpuBinding<A> = (<A as Api>::Buffer, u64, u64);

/// What the encoder used, for the memory module.
struct Notes<A: Api> {
    read: Vec<Rc<Buffer<A>>>,
    written: Vec<BufferRange<A>>,
    textures: Vec<(Tex<A>, u32, u32)>,
}

impl<A: Api> Default for Notes<A> {
    fn default() -> Self {
        Notes { read: Vec::new(), written: Vec::new(), textures: Vec::new() }
    }
}

/// One segment's recording.
struct Run<'r, A: Api> {
    g: &'r crate::exec::Gpu<'r, A>,
    objects: &'r Table,
    encoder: A::CommandEncoder,
    memory: &'r Memory<A>,
    helpers: &'r Helpers<A>,
    label: String,
    limits: Limits,
    targets: Targets<A>,
    parallel: bool,
    in_sub: bool,
    splits: bool,
    pass: Option<A::RenderPassEncoder>,
    passes: u32,
    state: State<A>,
    applied: Applied,
    groups: Vec<String>,
    visibility: Visibility<A>,
    arena: Arena<A>,
    staged: HashMap<usize, Staged<A>>,
    draws: usize,
    snapshots: HashMap<u32, A::Texture>,
    notes: Notes<A>,
    /// The stored textures and aliases the pass's draws bind.
    copies: PassCopies,
    /// The generated geometry of each patch draw, by draw.
    tessellated: HashMap<usize, crate::tess::Geometry<A>>,
    /// The tessellation bindings of the patch draw being prepared.
    patch_binding: Option<PatchBinding<A>>,
    /// Checks of the tessellated draws' results, for the command buffer.
    checks: Vec<crate::gpu::GpuFuture<std::result::Result<(), webmetal_wire::Failure>>>,
}

impl<A: Api, E: Embedder> EncoderService<A, E> for RenderEncoder<A> {
    fn encode(&mut self, cx: &mut EncodeCx<'_, A, E>, segment: &Segment) -> std::result::Result<(), RecordFailure> {
        let encoder = cx.encoder().clone();
        let records = cx.records();
        let items: Vec<(u64, &Command)> =
            segment.records.iter().map(|&i| (records[i].0 as u64, &records[i].1)).collect();
        let label = format!("task {} render encoder {:#x}", cx.task().0, segment.encoder.get());
        let parallel = segment.kind == EncoderKind::ParallelRender;
        let mut checks = Vec::new();
        let r = run(self, cx.gpu(), cx.objects(), encoder, &items, label, parallel, &mut checks)
            .map_err(|(offset, e)| RecordFailure { offset, failure: e.failure() });
        for c in checks {
            cx.check(c);
        }
        r
    }
}

fn limits<A: Api>(g: &crate::exec::Gpu<'_, A>) -> Limits {
    use crate::gpu::Device as _;
    let l = g.device().limits();
    let f = g.device().features();
    Limits {
        storage_align: u64::from(l.min_storage_buffer_offset_alignment.unwrap_or(256)),
        uniform_align: u64::from(l.min_uniform_buffer_offset_alignment.unwrap_or(256)),
        max_uniform_binding: l.max_uniform_buffer_binding_size.unwrap_or(65536),
        max_storage_binding: l.max_storage_buffer_binding_size.unwrap_or(134_217_728),
        max_texture_2d: f64::from(l.max_texture_dimension_2d.unwrap_or(8192)),
        max_vertex_stride: u64::from(l.max_vertex_buffer_array_stride.unwrap_or(2048)),
        max_buffer: l.max_buffer_size.unwrap_or(268_435_456),
        depth_clip_control: f.contains("depth-clip-control"),
        indirect_first_instance: f.contains("indirect-first-instance"),
    }
}

#[allow(clippy::too_many_arguments)]
fn run<A: Api>(
    svc: &RenderEncoder<A>,
    g: &crate::exec::Gpu<'_, A>,
    objects: &Table,
    encoder: A::CommandEncoder,
    items: &[(u64, &Command)],
    label: String,
    parallel: bool,
    checks: &mut Vec<crate::gpu::GpuFuture<std::result::Result<(), webmetal_wire::Failure>>>,
) -> std::result::Result<(), Fail> {
    let (begin_offset, desc): (u64, &RenderPassDescriptor) = match items.first() {
        Some((o, Command::BeginRenderPass(r))) => (*o, &r.render_pass_descriptor),
        Some((o, Command::BeginParallelRenderPass(r))) => (*o, &r.render_pass_descriptor),
        _ => return Err((0, RenderError::invalid("a render segment that does not begin with a render pass"))),
    };
    let lim = limits(g);
    let s = scan::<A>(items, objects, lim.uniform_align)?;
    let end_offset = items.last().map_or(begin_offset, |i| i.0);
    let targets = Targets::<A>::new(g, objects, desc, &s.stores, svc.memory.features()).map_err(|e| {
        // A store action left unknown is reported at the encoder's end.
        let o = if matches!(&e, RenderError::Undefined { call, .. } if call.contains("endEncoding")) {
            end_offset
        } else {
            begin_offset
        };
        (o, e)
    })?;
    if let Some(ds) = &targets.depth_stencil {
        super::resolve::check(ds).map_err(at(begin_offset))?;
    }
    let visibility_buffer = match desc.visibility_result_buffer {
        Some(h) => Some(buffer::<A>(objects, h, super::attachments::BEGIN).map_err(at(begin_offset))?),
        None => None,
    };

    // Arena.
    let uniform_base = s.bytes_size;
    let size = uniform_base + s.uniform_stride * s.draws as u64;
    let arena_buffer = if size == 0 {
        None
    } else {
        let mut d = BufferDescriptor::new(
            size64(size, "GPUDevice.createBuffer").map_err(at(begin_offset))?,
            BufferUsage::UNIFORM | BufferUsage::STORAGE | BufferUsage::VERTEX | BufferUsage::COPY_DST,
        );
        d.label = format!("{label} arena");
        Some(
            g.call("GPUDevice.createBuffer", &d.label, |dev| dev.create_buffer(&d))
                .map_err(|e| (begin_offset, RenderError::gpu("GPUDevice.createBuffer", e)))?,
        )
    };

    let mut r = Run {
        g,
        objects,
        encoder,
        memory: &svc.memory,
        helpers: &svc.helpers,
        label,
        limits: lim,
        state: State::new(targets.width, targets.height),
        targets,
        parallel,
        in_sub: false,
        splits: s.splits,
        pass: None,
        passes: 0,
        applied: Applied::default(),
        groups: Vec::new(),
        visibility: Visibility::new(visibility_buffer, s.spans),
        arena: Arena {
            buffer: arena_buffer,
            mirror: vec![0; size as usize],
            next_bytes: 0,
            uniform_base,
            uniform_stride: s.uniform_stride,
        },
        staged: HashMap::new(),
        draws: 0,
        snapshots: HashMap::new(),
        notes: Notes::default(),
        copies: PassCopies::default(),
        tessellated: HashMap::new(),
        patch_binding: None,
        checks: Vec::new(),
    };
    let before = r.before(&s, begin_offset, &svc.tessellator);
    checks.append(&mut r.checks);
    before?;
    r.begin_pass(true).map_err(at(begin_offset))?;
    for &(offset, c) in &items[1..] {
        r.record(offset, c)?;
    }
    Ok(())
}

impl<A: Api> Run<'_, A> {
    fn call<T>(&self, op: &'static str, f: impl FnOnce(&A::Device) -> T) -> T {
        self.g.call(op, &self.label, f)
    }

    /// The device limits the tessellator's bindings respect.
    fn tess_limits(&self) -> crate::tess::Limits {
        crate::tess::Limits {
            storage_offset_alignment: self.limits.storage_align,
            max_storage_binding: self.limits.max_storage_binding,
            max_buffer: self.limits.max_buffer,
        }
    }

    /// Work before the first WebGPU pass: bring textures up to date, stage indices, copy indirect draw parameters,
    /// tessellate patch draws.
    fn before(
        &mut self,
        s: &Scan<A>,
        begin: u64,
        tessellator: &crate::tess::Tessellator<A>,
    ) -> std::result::Result<(), Fail> {
        let mut textures: Vec<(u64, Tex<A>)> = Vec::new();
        for c in self.targets.colors.iter().flatten() {
            textures.push((begin, c.texture.clone()));
        }
        if let Some(ds) = &self.targets.depth_stencil {
            textures.push((begin, ds.texture.clone()));
        }
        for &(o, h) in &s.textures {
            // A handle that is not a texture the render path knows is reported when the record binds it.
            if let Ok(t) = Tex::<A>::lookup(self.objects, h) {
                textures.push((o, t));
            }
        }
        for (o, t) in &textures {
            if let Tex::Linear(lt) = t {
                self.memory.prepare_linear(self.g, &self.encoder, lt).map_err(mem_at(*o))?;
            }
            if let Some(a) = t.alias() {
                self.memory
                    .prepare_alias(self.g, &self.encoder, t.texture(), a, t.root_window())
                    .map_err(mem_at(*o))?;
                continue;
            }
            match self.memory.prepare_texture(self.g, &self.encoder, t.texture()).map_err(mem_at(*o))? {
                Prepare::Ready => {}
                Prepare::Decode => {
                    return Err((
                        *o,
                        RenderError::unimplemented(format!(
                            "{}: a software-decoded texture filled by a blit earlier in the command buffer; its \
                             decoded texels reach the GPU at the next commit",
                            t.texture().label
                        )),
                    ));
                }
            }
        }
        let staging = StagingLimits {
            storage_offset_alignment: self.limits.storage_align,
            max_storage_binding: self.limits.max_storage_binding,
        };
        for job in &s.jobs {
            let staged = index::encode(
                self.g,
                &self.encoder,
                self.helpers,
                staging,
                &job.index_buffer,
                job.index_offset,
                job.index_type,
                job.per_primitive,
                &job.source,
            )
            .map_err(at(job.offset))?;
            self.notes.read.push(job.index_buffer.clone());
            if let StageSource::Indirect { buffer, .. } = &job.source {
                self.notes.read.push(buffer.clone());
            }
            self.staged.insert(job.draw, staged);
        }
        for (draw, src, src_offset, field) in &s.copies {
            let arena = self.arena.buffer.clone().expect("draws with uniform blocks have an arena");
            let dst = self.arena.uniform_base + self.arena.uniform_stride * *draw as u64 + field;
            self.call("GPUCommandEncoder.copyBufferToBuffer", |_| {
                self.encoder.copy_buffer_to_buffer_with_offsets(
                    src.gpu(),
                    Size64::new(*src_offset).expect("checked"),
                    &arena,
                    Size64::new(dst).expect("small"),
                    Some(Size64::from(4u32)),
                )
            });
            self.notes.read.push(src.clone());
        }
        for (draw, offset, job) in &s.tess {
            let generated = tessellator.encode(self.g, &self.encoder, job, self.tess_limits()).map_err(at(*offset))?;
            self.notes.read.push(job.factors.clone());
            if let Some((b, _)) = &job.patch_indices {
                self.notes.read.push(b.clone());
            }
            self.checks.push(generated.check);
            self.tessellated.insert(*draw, generated.geometry);
        }
        Ok(())
    }

    fn pass(&self) -> &A::RenderPassEncoder {
        self.pass.as_ref().expect("a WebGPU pass is open while records are replayed")
    }

    fn begin_pass(&mut self, first: bool) -> Result<()> {
        let set = self.visibility.pass_set(self.g)?;
        let last = !self.splits;
        let label = format!("{} pass {}", self.label, self.passes);
        let d = self.targets.descriptor(&label, first, last, set.as_ref());
        let pass = self
            .g
            .call("GPUCommandEncoder.beginRenderPass", &label, |_| self.encoder.begin_render_pass(&d))
            .map_err(|e| RenderError::gpu("GPUCommandEncoder.beginRenderPass", e))?;
        drop(d);
        for group in &self.groups {
            self.call("GPURenderPassEncoder.pushDebugGroup", |_| pass.push_debug_group(group));
        }
        self.passes += 1;
        self.applied = Applied::default();
        self.visibility.pass_begun(self.g, &pass)?;
        self.pass = Some(pass);
        Ok(())
    }

    fn end_pass(&mut self) {
        let pass = self.pass.take().expect("an open pass");
        self.visibility.pass_ending(self.g, &pass);
        for _ in &self.groups {
            self.call("GPURenderPassEncoder.popDebugGroup", |_| pass.pop_debug_group());
        }
        self.call("GPURenderPassEncoder.end", |_| pass.end());
    }

    /// End the WebGPU pass and begin the next one of the Metal pass.
    fn split(&mut self) -> Result<()> {
        if !self.splits {
            return Err(RenderError::invalid("a split the encoder's scan did not find"));
        }
        self.end_pass();
        self.begin_pass(false)
    }

    fn record(&mut self, offset: u64, c: &Command) -> std::result::Result<(), Fail> {
        self.record_inner(c).map_err(|e| match e {
            RenderError::Unimplemented { what } if what.is_empty() => {
                (offset, RenderError::Unimplemented { what: failure::unimplemented_record(c.opcode()).message })
            }
            e => (offset, e),
        })
    }

    fn record_inner(&mut self, c: &Command) -> Result<()> {
        use MetalStage::{Fragment, Vertex};
        match c {
            Command::BeginParallelSubEncoder(_) => {
                if !self.parallel {
                    return Err(RenderError::invalid("a sub-encoder outside a parallel render encoder"));
                }
                self.in_sub = true;
                self.state = State::new(self.targets.width, self.targets.height);
            }
            Command::EncoderEndEncoding(_) => {
                if self.parallel && self.in_sub {
                    // A sub-encoder ends: its debug groups and visibility span end with it, the next starts afresh.
                    self.close_groups();
                    let pass = self.pass.clone().expect("an open pass");
                    self.visibility.end_span(self.g, &pass);
                    self.in_sub = false;
                } else {
                    self.finish()?;
                }
            }
            Command::EncoderSetLabel(_) => {}
            Command::EncoderPushDebugGroup(r) => {
                let label = r.string.clone().unwrap_or_default();
                self.call("GPURenderPassEncoder.pushDebugGroup", |_| self.pass().push_debug_group(&label));
                self.groups.push(label);
            }
            Command::EncoderPopDebugGroup(_) => {
                // Metal accepts a pop with no group open (measured on Apple M4, macOS 27.0.1; `DebugGroups`).
                if self.groups.pop().is_some() {
                    self.call("GPURenderPassEncoder.popDebugGroup", |_| self.pass().pop_debug_group());
                }
            }
            Command::EncoderInsertDebugSignpost(r) => {
                let label = r.string.clone().unwrap_or_default();
                self.call("GPURenderPassEncoder.insertDebugMarker", |_| self.pass().insert_debug_marker(&label));
            }

            Command::RenderSetRenderPipelineState(r) => {
                self.state.pipeline = match r.pipeline_state {
                    Some(h) => Some(pipeline::<A>(self.objects, h)?),
                    None => None,
                };
            }
            Command::RenderSetDepthStencilState(r) => {
                self.state.depth_stencil = match r.depth_stencil_state {
                    Some(h) => payload::<DepthStencilState>(self.objects, h)
                        .ok_or_else(|| RenderError::invalid(format!("{:#x} is not a depth-stencil state", h.get())))?
                        .as_ref()
                        .clone(),
                    None => DepthStencilState::DEFAULT,
                };
            }
            Command::RenderSetCullMode(r) => {
                self.state.cull = CullMode::from_value(r.cull_mode, "-[MTLRenderCommandEncoder setCullMode:]")?
            }
            Command::RenderSetFrontFacingWinding(r) => {
                self.state.winding =
                    Winding::from_value(r.front_facing_winding, "-[MTLRenderCommandEncoder setFrontFacingWinding:]")?
            }
            Command::RenderSetDepthClipMode(r) => {
                self.state.clip =
                    DepthClipMode::from_value(r.depth_clip_mode, "-[MTLRenderCommandEncoder setDepthClipMode:]")?
            }
            Command::RenderSetTriangleFillMode(r) => {
                self.state.fill = FillMode::from_value(r.fill_mode, "-[MTLRenderCommandEncoder setTriangleFillMode:]")?
            }
            Command::RenderSetDepthBias(r) => self.state.bias = (r.depth_bias, r.slope_scale, r.clamp),
            Command::RenderSetViewport(r) => self.state.viewport = self.viewport(&r.viewport)?,
            Command::RenderSetViewports(r) => {
                if r.viewports.len() != 1 {
                    return Err(RenderError::undefined(
                        "-[MTLRenderCommandEncoder setViewports:count:]",
                        format!("{} viewports; the A10X's maxViewportCount is 1 (device profile)", r.viewports.len()),
                    ));
                }
                self.state.viewport = self.viewport(&r.viewports[0])?;
            }
            Command::RenderSetScissorRect(r) => self.state.scissor = self.scissor(&r.rect)?,
            Command::RenderSetScissorRects(r) => {
                if r.scissor_rects.len() != 1 {
                    return Err(RenderError::undefined(
                        "-[MTLRenderCommandEncoder setScissorRects:count:]",
                        format!(
                            "{} rectangles; the A10X's maxViewportCount is 1 (device profile)",
                            r.scissor_rects.len()
                        ),
                    ));
                }
                self.state.scissor = self.scissor(&r.scissor_rects[0])?;
            }
            Command::RenderSetBlendColor(r) => self.state.blend = [r.red, r.green, r.blue, r.alpha],
            Command::RenderSetStencilReferenceValue(r) => {
                self.state.refs = (reference(r.reference_value)?, reference(r.reference_value)?)
            }
            Command::RenderSetStencilFrontBackReferenceValue(r) => {
                self.state.refs = (reference(r.front_reference_value)?, reference(r.back_reference_value)?)
            }
            Command::RenderSetVisibilityResultMode(r) => {
                let mode =
                    VisibilityMode::from_value(r.mode, "-[MTLRenderCommandEncoder setVisibilityResultMode:offset:]")?;
                let pass = self.pass.clone().expect("an open pass");
                self.visibility.set_mode(self.g, &pass, mode, r.offset)?;
            }
            Command::RenderSetColorStoreAction(_)
            | Command::RenderSetDepthStoreAction(_)
            | Command::RenderSetStencilStoreAction(_)
            | Command::ParallelSetColorStoreAction(_)
            | Command::ParallelSetDepthStoreAction(_)
            | Command::ParallelSetStencilStoreAction(_) => {
                // Applied before the pass began (`scan`).
                if self.in_sub {
                    return Err(RenderError::unimplemented(
                        "a store action set on a parallel render encoder's sub-encoder: Metal's handling of it is \
                         not established",
                    ));
                }
            }
            Command::RenderSetColorStoreActionOptions(r) => store_options(r.store_action_options)?,
            Command::RenderSetDepthStoreActionOptions(r) => store_options(r.store_action_options)?,
            Command::RenderSetStencilStoreActionOptions(r) => store_options(r.store_action_options)?,
            Command::ParallelSetColorStoreActionOptions(r) => store_options(r.store_action_options)?,
            Command::ParallelSetDepthStoreActionOptions(r) => store_options(r.store_action_options)?,
            Command::ParallelSetStencilStoreActionOptions(r) => store_options(r.store_action_options)?,

            Command::RenderSetVertexBuffer(r) => self.set_buffer(Vertex, r.index, r.buffer, r.offset, None)?,
            Command::RenderSetFragmentBuffer(r) => self.set_buffer(Fragment, r.index, r.buffer, r.offset, None)?,
            Command::RenderSetVertexBufferAttributeStride(r) => {
                self.set_buffer(Vertex, r.index, r.buffer, r.offset, Some(r.stride))?
            }
            Command::RenderSetVertexBufferOffset(r) => self.set_offset(Vertex, r.index, r.offset, None)?,
            Command::RenderSetFragmentBufferOffset(r) => self.set_offset(Fragment, r.index, r.offset, None)?,
            Command::RenderSetVertexBufferOffsetAttributeStride(r) => {
                self.set_offset(Vertex, r.index, r.offset, Some(r.stride))?
            }
            Command::RenderSetVertexBuffers(r) => {
                for (i, (b, o)) in r.buffers.iter().zip(&r.offsets).enumerate() {
                    self.set_buffer(Vertex, r.range.location + i as u64, *b, *o, None)?;
                }
            }
            Command::RenderSetFragmentBuffers(r) => {
                for (i, (b, o)) in r.buffers.iter().zip(&r.offsets).enumerate() {
                    self.set_buffer(Fragment, r.range.location + i as u64, *b, *o, None)?;
                }
            }
            Command::RenderSetVertexBuffersAttributeStrides(r) => {
                for (i, ((b, o), s)) in r.buffers.iter().zip(&r.offsets).zip(&r.strides).enumerate() {
                    self.set_buffer(Vertex, r.range.location + i as u64, *b, *o, Some(*s))?;
                }
            }
            Command::RenderSetVertexBytes(r) => self.set_bytes(Vertex, r.index, &r.bytes, None)?,
            Command::RenderSetVertexBytesAttributeStride(r) => {
                self.set_bytes(Vertex, r.index, &r.bytes, Some(r.stride))?
            }
            Command::RenderSetFragmentBytes(r) => self.set_bytes(Fragment, r.index, &r.bytes, None)?,
            Command::RenderSetVertexTexture(r) => self.set_texture(Vertex, r.index, r.texture)?,
            Command::RenderSetFragmentTexture(r) => self.set_texture(Fragment, r.index, r.texture)?,
            Command::RenderSetVertexTextures(r) => {
                for (i, t) in r.textures.iter().enumerate() {
                    self.set_texture(Vertex, r.range.location + i as u64, *t)?;
                }
            }
            Command::RenderSetFragmentTextures(r) => {
                for (i, t) in r.textures.iter().enumerate() {
                    self.set_texture(Fragment, r.range.location + i as u64, *t)?;
                }
            }
            Command::RenderSetVertexSamplerState(r) => self.set_sampler(Vertex, r.index, r.sampler)?,
            Command::RenderSetFragmentSamplerState(r) => self.set_sampler(Fragment, r.index, r.sampler)?,
            Command::RenderSetVertexSamplerStates(r) => {
                for (i, s) in r.samplers.iter().enumerate() {
                    self.set_sampler(Vertex, r.range.location + i as u64, *s)?;
                }
            }
            Command::RenderSetFragmentSamplerStates(r) => {
                for (i, s) in r.samplers.iter().enumerate() {
                    self.set_sampler(Fragment, r.range.location + i as u64, *s)?;
                }
            }
            Command::RenderSetFragmentTextureAndSamplerState(r) => {
                self.set_texture(Fragment, r.texture_index, r.texture)?;
                self.set_sampler(Fragment, r.sampler_index, r.sampler)?;
            }

            // Residency and hazard-tracking declarations. WebGPU makes every bound resource resident and orders
            // accesses itself; the resources these name reach a draw only through argument buffers, which the render
            // path does not bind (an argument-buffer binding source is reported at the draw).
            Command::RenderUseResource(_)
            | Command::RenderUseResourceStages(_)
            | Command::RenderUseResources(_)
            | Command::RenderUseResourcesStages(_)
            | Command::RenderUseHeap(_)
            | Command::RenderUseHeapStages(_)
            | Command::RenderUseHeaps(_)
            | Command::RenderUseHeapsStages(_) => {}

            // The tessellation state is read by the scan, which makes each patch draw's job (crate::tess).
            Command::RenderSetTessellationFactorBuffer(_) | Command::RenderSetTessellationFactorScale(_) => {}
            Command::RenderMemoryBarrierWithScope(r) => self.barrier(r.scope, r.after, r.before)?,
            Command::RenderMemoryBarrierWithResources(r) => {
                self.barrier(barrier_scope::BUFFERS | barrier_scope::TEXTURES, r.after, r.before)?
            }
            Command::RenderTextureBarrier(_) => {
                return Err(RenderError::undefined(
                    "-[MTLRenderCommandEncoder textureBarrier]",
                    "API_UNAVAILABLE(ios) (MTLRenderCommandEncoder.h)",
                ));
            }

            c => match draw_of(c) {
                Some((ty, kind)) => {
                    let prim = PrimitiveType::from_value(ty, DRAW)?;
                    let ordinal = self.draws;
                    self.draws += 1;
                    self.draw(ordinal, prim, &kind)?;
                }
                None => return Err(RenderError::Unimplemented { what: String::new() }),
            },
        }
        Ok(())
    }

    fn close_groups(&mut self) {
        while self.groups.pop().is_some() {
            self.call("GPURenderPassEncoder.popDebugGroup", |_| self.pass().pop_debug_group());
        }
    }

    /// The encoder's `endEncoding`: end the pass, apply the store and resolve actions, write visibility results and
    /// tell the memory module what was used.
    fn finish(&mut self) -> Result<()> {
        // Metal accepts an encoder that ends with debug groups open (measured on Apple M4, macOS 27.0.1); WebGPU's
        // must be balanced within the pass.
        self.close_groups();
        self.end_pass();
        if self.splits {
            let label = format!("{} store", self.label);
            let d = self.targets.descriptor(&label, false, true, None);
            let pass = self
                .g
                .call("GPUCommandEncoder.beginRenderPass", &label, |_| self.encoder.begin_render_pass(&d))
                .map_err(|e| RenderError::gpu("GPUCommandEncoder.beginRenderPass", e))?;
            self.call("GPURenderPassEncoder.end", |_| pass.end());
        }
        if let Some(ds) = &self.targets.depth_stencil {
            super::resolve::encode(self.g, &self.encoder, self.helpers, ds)?;
        }
        if let Some((buffer, ranges)) =
            self.visibility.finish(self.g, &self.encoder, self.helpers, self.limits.storage_align)?
        {
            for r in ranges {
                self.notes.written.push((buffer.clone(), r));
            }
            self.notes.read.push(buffer);
        }
        if let Some(arena) = &self.arena.buffer {
            let mirror = &self.arena.mirror;
            self.call("GPUQueue.writeBuffer", |_| {
                self.g.queue().write_buffer(arena, Size64::from(0u32), mirror, Size64::from(0u32), None)
            })
            .map_err(|e| RenderError::gpu("GPUQueue.writeBuffer", e))?;
        }
        // The memory module's bookkeeping.
        for (t, level, layer) in self.targets.written() {
            self.notes.textures.push((t, level, layer));
        }
        for b in &self.notes.read {
            self.memory.note_buffer_read(b);
        }
        for (b, range) in &self.notes.written {
            self.memory.note_buffer_write(b, Some(*range));
        }
        for (t, level, layer) in &self.notes.textures {
            let tex = t.texture();
            let layers = if tex.geometry.kind.is_3d() { 0..1 } else { *layer..layer + 1 };
            match t.alias() {
                Some(a) => self.memory.note_alias_write(tex, a, *level..level + 1, layers),
                None => self.memory.note_texture_write(tex, *level..level + 1, layers),
            }
            if let Tex::Linear(lt) = t {
                self.memory.note_linear_write(lt);
            }
        }
        Ok(())
    }

    fn viewport(&self, v: &Viewport) -> Result<[f32; 6]> {
        const CALL: &str = "-[MTLRenderCommandEncoder setViewport:]";
        let [x, y, w, h, n, f] = [v.origin_x, v.origin_y, v.width, v.height, v.znear, v.zfar];
        if [x, y, w, h].iter().any(|c| c.fract() != 0.0 || !c.is_finite()) {
            return Err(RenderError::unimplemented(format!(
                "{CALL}: the fractional viewport ({x}, {y}, {w}, {h}): WebGPU may round viewport bounds to integer \
                 precision (tables/limits.toml question 6.4-viewport-bounds), and the shader-side transform is not \
                 implemented"
            )));
        }
        let max = self.limits.max_texture_2d;
        let range = max * 2.0;
        if w < 0.0
            || h < 0.0
            || w > max
            || h > max
            || x < -range
            || y < -range
            || x + w > range - 1.0
            || y + h > range - 1.0
        {
            return Err(RenderError::unimplemented(format!(
                "{CALL}: the viewport ({x}, {y}, {w}, {h}) is outside WebGPU's viewport range for maxTextureDimension2D \
                 {max} (tables/limits.toml question 6.4-viewport-bounds); the shader-side transform is not implemented"
            )));
        }
        if !(0.0..=1.0).contains(&n) || !(0.0..=1.0).contains(&f) || n > f {
            return Err(RenderError::unimplemented(format!(
                "{CALL}: the depth range [{n}, {f}]: WebGPU requires 0 <= minDepth <= maxDepth <= 1; the shader-side \
                 transform is not implemented"
            )));
        }
        let narrow = |c: f64| c as f32;
        if f64::from(narrow(n)) != n || f64::from(narrow(f)) != f {
            return Err(RenderError::unimplemented(format!(
                "{CALL}: the depth range [{n}, {f}] is not exact in binary32, WebGPU's viewport type"
            )));
        }
        Ok([x as f32, y as f32, w as f32, h as f32, n as f32, f as f32])
    }

    fn scissor(&self, r: &webmetal_wire::ScissorRect) -> Result<[u32; 4]> {
        let (w, h) = (u64::from(self.targets.width), u64::from(self.targets.height));
        if r.x.checked_add(r.width).is_none_or(|e| e > w) || r.y.checked_add(r.height).is_none_or(|e| e > h) {
            return Err(RenderError::undefined(
                "-[MTLRenderCommandEncoder setScissorRect:]",
                format!(
                    "the rectangle ({}, {}, {}, {}) extends past the {w}x{h} render target; WebGPU requires it within",
                    r.x, r.y, r.width, r.height
                ),
            ));
        }
        Ok([r.x as u32, r.y as u32, r.width as u32, r.height as u32])
    }

    fn set_buffer(
        &mut self,
        stage: MetalStage,
        index: u64,
        handle: Option<Handle>,
        offset: u64,
        stride: Option<u64>,
    ) -> Result<()> {
        check_index(stage, index, 31, "buffer")?;
        let slots = self.state.slots(stage);
        match handle {
            None => {
                slots.buffers.remove(&index);
            }
            Some(h) => {
                let b = buffer::<A>(self.objects, h, "-[MTLRenderCommandEncoder setVertexBuffer:offset:atIndex:]")?;
                self.state.slots(stage).buffers.insert(index, Bound::Buffer { buffer: b, offset, stride });
            }
        }
        Ok(())
    }

    fn set_offset(&mut self, stage: MetalStage, index: u64, offset: u64, stride: Option<u64>) -> Result<()> {
        let call = "-[MTLRenderCommandEncoder setVertexBufferOffset:atIndex:]";
        match self.state.slots(stage).buffers.get_mut(&index) {
            Some(Bound::Buffer { offset: o, stride: s, .. }) => {
                *o = offset;
                if stride.is_some() {
                    *s = stride;
                }
                Ok(())
            }
            Some(Bound::Bytes { .. }) => {
                Err(RenderError::undefined(call, format!("a new offset for the inline bytes at index {index}")))
            }
            None => Err(RenderError::undefined(call, format!("no buffer is bound at index {index}"))),
        }
    }

    fn set_bytes(&mut self, stage: MetalStage, index: u64, bytes: &[u8], stride: Option<u64>) -> Result<()> {
        check_index(stage, index, 31, "buffer")?;
        let offset = self.arena.next_bytes;
        let len = bytes.len() as u64;
        self.arena.mirror[offset as usize..(offset + len) as usize].copy_from_slice(bytes);
        self.arena.next_bytes += len.max(1).next_multiple_of(256);
        self.state.slots(stage).buffers.insert(index, Bound::Bytes { offset, len, stride });
        Ok(())
    }

    fn set_texture(&mut self, stage: MetalStage, index: u64, handle: Option<Handle>) -> Result<()> {
        check_index(stage, index, 31, "texture")?;
        match handle {
            None => {
                self.state.slots(stage).textures.remove(&index);
            }
            Some(h) => {
                let t = Tex::<A>::lookup(self.objects, h)?;
                self.state.slots(stage).textures.insert(index, t);
            }
        }
        Ok(())
    }

    fn set_sampler(&mut self, stage: MetalStage, index: u64, handle: Option<Handle>) -> Result<()> {
        check_index(stage, index, 16, "sampler")?;
        match handle {
            None => {
                self.state.slots(stage).samplers.remove(&index);
            }
            Some(h) => {
                let s = payload::<SamplerObject<A>>(self.objects, h)
                    .ok_or_else(|| RenderError::invalid(format!("handle {:#x} is not a sampler state", h.get())))?;
                self.state.slots(stage).samplers.insert(index, s);
            }
        }
        Ok(())
    }

    /// `memoryBarrierWithScope:afterStages:beforeStages:` and its resource form.
    fn barrier(&mut self, scope: u64, after: u64, before: u64) -> Result<()> {
        const CALL: &str = "-[MTLRenderCommandEncoder memoryBarrierWithScope:afterStages:beforeStages:]";
        if scope & barrier_scope::RENDER_TARGETS != 0 {
            return Err(RenderError::undefined(CALL, "MTLBarrierScopeRenderTargets is API_UNAVAILABLE(ios)"));
        }
        if (after | before) & !stages::ALL != 0 {
            return Err(RenderError::undefined(CALL, format!("stages {after:#x} {before:#x} are not MTLRenderStages")));
        }
        if after & (stages::FRAGMENT | stages::TILE) != 0 {
            // Apple-family GPUs reject it (debug-layer assertion of reference case
            // `render.error.memory-barrier-after-fragment`).
            return Err(RenderError::undefined(
                CALL,
                format!("afterStages {after:#x} contains the fragment or tile stage, which Apple-family GPUs reject"),
            ));
        }
        if after & (stages::OBJECT | stages::MESH) != 0 || before & (stages::OBJECT | stages::MESH | stages::TILE) != 0
        {
            return Err(RenderError::undefined(CALL, "object, mesh and tile stages are not part of the A10X's family"));
        }
        if splits_pass(after, before) {
            // Stores of the vertex stage of earlier draws become visible to the fragment stage of later draws: end the
            // WebGPU pass, whose end makes every store of its draws visible to the next pass.
            self.split()?;
        }
        Ok(())
    }

    /// One draw record.
    fn draw(&mut self, ordinal: usize, prim: PrimitiveType, kind: &DrawKind) -> Result<()> {
        let state = self.state.pipeline.clone().ok_or_else(|| {
            RenderError::undefined(DRAW, "renderPipelineState must be set (debug-layer assertion); Metal faults")
        })?;
        self.check_pipeline(&state.creation, prim)?;
        if !state.creation.rasterization {
            // The vertex stage has no side effects (the pipeline service refuses vertex-stage writes) and nothing is
            // rasterized: the draw has no observable effect.
            return Ok(());
        }
        if self.state.fill == FillMode::Lines && prim.is_triangle() {
            return Err(RenderError::host(
                "fill-mode-lines: setTriangleFillMode:MTLTriangleFillModeLines; GPUPrimitiveState has no fill mode \
                 (tables/refusals.md)",
            ));
        }
        let (bias, slope, _) = self.state.bias;
        if prim.is_triangle() && (bias != 0.0 || slope != 0.0) {
            return Err(RenderError::unimplemented(format!(
                "setDepthBias:{bias} slopeScale:{slope}: Metal does not define the depth bias formula or its units, \
                 and the A10X's is not measured (tables/limits.toml question 6.4-depth-bias-formula)"
            )));
        }
        let unclipped = match self.state.clip {
            DepthClipMode::Clip => false,
            DepthClipMode::Clamp if self.limits.depth_clip_control => true,
            DepthClipMode::Clamp => {
                return Err(RenderError::host(
                    "depth-clamp: setDepthClipMode:MTLDepthClipModeClamp; unclippedDepth requires the \
                     depth-clip-control feature (tables/refusals.md)",
                ));
            }
        };
        let ds = self.state.depth_stencil.clone();
        let plan = self.depth_stencil_plan(&ds, prim)?;
        let strip_index_format = match kind {
            DrawKind::Indexed { index_type, .. } | DrawKind::IndexedIndirect { index_type, .. } if prim.is_strip() => {
                Some(IndexType::from_value(*index_type, DRAW)?.format())
            }
            _ => None,
        };
        let p = self.binding_variant(&state)?;
        let strides = self.strides(&p)?;
        let key = VariantKey {
            topology: prim.topology(),
            strip_index_format,
            front_face: self.state.winding.gpu(),
            cull_mode: self.state.cull.gpu(),
            unclipped_depth: unclipped,
            depth_stencil: plan.as_ref().map(|(k, _)| *k),
            strides,
        };
        let faces = match &plan {
            Some((_, f)) => *f,
            None => FacePlan::One(FaceValues { reference: 0, read_mask: 0xff, write_mask: 0xff }),
        };
        self.patch_binding = None;
        if let DrawKind::Patches(d) = kind {
            let Some(geometry) = self.tessellated.get(&ordinal) else {
                // A draw of no patches or no instances draws nothing.
                return Ok(());
            };
            let vertices = geometry.vertices.clone();
            let indices = crate::tess::control_point_indices::<A>(self.objects, d, self.tess_limits())?;
            if let Some(w) = &indices {
                self.notes.read.push(w.buffer.clone());
            }
            self.patch_binding = Some(PatchBinding { vertices, indices });
        }
        let uniforms = self.uniforms(&p, ordinal, kind)?;
        let per_primitive = matches!(p.passes, PassPlan::PerPrimitive { .. });
        let split_faces = matches!(faces, FacePlan::Split(..));
        if per_primitive || split_faces {
            let DrawKind::Direct { start, count, instances, base_instance } = kind else {
                let (what, source) = match kind {
                    DrawKind::Patches(_) | DrawKind::PatchesIndirect => {
                        ("a patch draw", "the triangles the tessellator generates on the GPU are")
                    }
                    _ => ("an indexed or indirect draw", "the primitive list of index data is"),
                };
                return Err(RenderError::unimplemented(format!(
                    "{what} drawn one primitive at a time ({}): {source} not read on the CPU",
                    if per_primitive { "framebuffer fetch" } else { "separate front and back stencil values" }
                )));
            };
            return self.draw_primitives(&p, &key, faces, uniforms, prim, (*start, *count, *instances, *base_instance));
        }
        let FacePlan::One(values) = faces else { unreachable!("split handled above") };
        self.prepare_draw(&p, &key, values, &uniforms, None)?;
        self.issue(ordinal, kind, prim)
    }

    /// The pipeline compiled for the textures and sampler states the draw binds at its functions' texture and sampler
    /// arguments (the pipeline service's binding variant), or the pipeline itself when its code does not depend on
    /// them. A bound texture an argument cannot take is behavior Metal does not define (its validation layer rejects
    /// the draw), reported before the compiler sees it.
    fn binding_variant(&self, state: &RenderPipelineState<A>) -> Result<Rc<RenderPipeline<A>>> {
        let Some(variants) = &state.variants else { return Ok(state.creation.clone()) };
        let mut key = BindingKey::default();
        for stage in [MetalStage::Vertex, MetalStage::Fragment] {
            let arguments = variants.arguments(stage);
            let slots = self.state.slots_ref(stage);
            let k = match stage {
                MetalStage::Vertex => &mut key.vertex,
                MetalStage::Fragment => &mut key.fragment,
            };
            let mut textures: Vec<u64> =
                arguments.texture_indices().filter(|i| slots.textures.contains_key(i)).collect();
            textures.sort_unstable();
            textures.dedup();
            for index in textures {
                let tex = &slots.textures[&index];
                let g = tex.geometry();
                if let Some(a) = crate::pipeline::bound::argument_at(&arguments.textures, index)
                    && let Some(why) = crate::pipeline::bound::texture_mismatch(
                        a,
                        index,
                        crate::pipeline::bound::texture_type_value(g.kind),
                        u64::from(tex.format()),
                    )
                {
                    return Err(RenderError::undefined(DRAW, format!("the {} function's {why}", stage.name())));
                }
                k.textures.push(TextureSlot::of(index, g, tex.format(), tex.swizzle()));
            }
            let mut samplers: Vec<u64> =
                arguments.sampler_indices().filter(|i| slots.samplers.contains_key(i)).collect();
            samplers.sort_unstable();
            samplers.dedup();
            k.samplers = samplers.into_iter().map(|i| SamplerSlot::of(i, &slots.samplers[&i].descriptor)).collect();
        }
        let p = variants.variant(self.g, &key)?;
        // The encoder's scan placed the per-draw uniform blocks and split the passes by the pipeline as created.
        if p.uniform_blocks() != state.creation.uniform_blocks() || p.passes != state.creation.passes {
            return Err(RenderError::unimplemented(format!(
                "render pipeline {}: the variant for the bound textures has another per-draw uniform block or pass plan \
                 than the pipeline as created, which the encoder sized and split the pass by",
                p.label
            )));
        }
        Ok(p)
    }

    /// Check the pipeline against the pass (Metal requires them to agree; the debug layer asserts otherwise).
    fn check_pipeline(&self, p: &RenderPipeline<A>, prim: PrimitiveType) -> Result<()> {
        let formats = self.targets.color_formats();
        let n = formats.len().max(p.colors.len());
        for i in 0..n {
            let pass = formats.get(i).copied().unwrap_or(0);
            let pipe = p.colors.get(i).map_or(0, |c| c.pixel_format);
            if pass != pipe {
                return Err(RenderError::undefined(
                    DRAW,
                    format!(
                        "color attachment {i}: the pipeline's pixel format {pipe} differs from the attachment's {pass}"
                    ),
                ));
            }
        }
        let (depth, stencil) = match &self.targets.depth_stencil {
            Some(ds) => (
                if ds.depth.is_some() { ds.pixel_format } else { 0 },
                if ds.stencil.is_some() { ds.pixel_format } else { 0 },
            ),
            None => (0, 0),
        };
        if p.depth_format != depth || p.stencil_format != stencil {
            return Err(RenderError::undefined(
                DRAW,
                format!(
                    "the pipeline's depth and stencil pixel formats ({}, {}) differ from the pass's ({depth}, {stencil})",
                    p.depth_format, p.stencil_format
                ),
            ));
        }
        if p.sample_count != self.targets.sample_count {
            return Err(RenderError::undefined(
                DRAW,
                format!(
                    "the pipeline's rasterSampleCount {} differs from the attachments' sample count {}",
                    p.sample_count, self.targets.sample_count
                ),
            ));
        }
        let [.., near, far] = self.state.viewport;
        if p.writes_depth && (near, far) != (0.0, 1.0) {
            return Err(RenderError::unimplemented(format!(
                "{DRAW}: the fragment function writes the depth with the viewport depth range [{near}, {far}]: Metal \
                 stores the written depth clamped to [0, 1], WebGPU clamps it to the viewport's depth range (WebGPU \
                 23.2.6), and the two agree only for [0, 1]"
            )));
        }
        if p.topology_class != TopologyClass::Unspecified && p.topology_class != prim.class() {
            return Err(RenderError::undefined(
                DRAW,
                format!("a {prim:?} draw with a pipeline of inputPrimitiveTopology {:?}", p.topology_class),
            ));
        }
        Ok(())
    }

    /// The variant's depth-stencil state and the stencil values of a draw, for a pass with a depth-stencil attachment.
    fn depth_stencil_plan(
        &self,
        ds: &DepthStencilState,
        prim: PrimitiveType,
    ) -> Result<Option<(DepthStencilKey, FacePlan)>> {
        let Some(att) = &self.targets.depth_stencil else {
            if ds.uses_depth() || ds.uses_stencil() {
                return Err(RenderError::undefined(
                    DRAW,
                    "the depth-stencil state tests or writes depth or stencil but the pass has no such attachment",
                ));
            }
            return Ok(None);
        };
        if ds.uses_depth() && att.depth.is_none() {
            return Err(RenderError::undefined(
                DRAW,
                "the depth-stencil state uses depth but the pass has no depth attachment",
            ));
        }
        if ds.uses_stencil() && att.stencil.is_none() {
            return Err(RenderError::undefined(
                DRAW,
                "the depth-stencil state uses stencil but the pass has no stencil attachment",
            ));
        }
        let plan = face_plan(ds, self.state.refs, self.state.cull, prim.is_triangle())?;
        let face = |f: Face| StencilFaceKey {
            compare: f.compare.gpu(),
            fail: f.fail.gpu(),
            depth_fail: f.depth_fail.gpu(),
            pass: f.pass.gpu(),
        };
        let values = match plan {
            FacePlan::One(v) | FacePlan::Split(v, _) => v,
        };
        // An aspect the pass does not attach is neither tested nor written.
        let stencil = att.stencil.is_some();
        let key = DepthStencilKey {
            format: att.format,
            depth: att.has_depth.then(|| {
                if att.depth.is_some() {
                    (ds.depth_write, ds.depth_compare.gpu())
                } else {
                    (false, crate::gpu::idl::CompareFunction::Always)
                }
            }),
            front: if stencil { face(ds.front_face()) } else { face(Face::DISABLED) },
            back: if stencil { face(ds.back_face()) } else { face(Face::DISABLED) },
            read_mask: if stencil { values.read_mask } else { 0xff },
            write_mask: if stencil { values.write_mask } else { 0 },
        };
        Ok(Some((key, plan)))
    }

    fn strides(&self, p: &RenderPipeline<A>) -> Result<Vec<u64>> {
        p.vertex_buffers
            .iter()
            .map(|vb| {
                let s = match vb.stride {
                    Stride::Fixed(s) => s,
                    Stride::Dynamic => match self.state.vertex.buffers.get(&vb.metal_index) {
                        Some(Bound::Buffer { stride: Some(s), .. } | Bound::Bytes { stride: Some(s), .. }) => *s,
                        _ => {
                            return Err(RenderError::undefined(
                                DRAW,
                                format!(
                                    "vertex buffer {} has a dynamic stride and was bound without attributeStride",
                                    vb.metal_index
                                ),
                            ));
                        }
                    },
                };
                if s % 4 != 0 || s > self.limits.max_vertex_stride {
                    return Err(RenderError::unimplemented(format!(
                        "vertex buffer {} stride {s}: WebGPU requires a multiple of 4 up to maxVertexBufferArrayStride \
                         {} for native vertex input (tables/limits.toml question 5.10-vertex-buffer-alignment)",
                        vb.metal_index, self.limits.max_vertex_stride
                    )));
                }
                Ok(s)
            })
            .collect()
    }

    /// The per-draw uniform block values that are known on the CPU, and the block's arena offset.
    fn uniforms(&mut self, p: &RenderPipeline<A>, ordinal: usize, kind: &DrawKind) -> Result<Option<u64>> {
        if p.uniform_size == 0 {
            return Ok(None);
        }
        let base = self.arena.uniform_base + self.arena.uniform_stride * ordinal as u64;
        let (base_vertex, base_instance) = match kind {
            DrawKind::Direct { base_instance, .. } => (Some(0i64), Some(*base_instance)),
            DrawKind::Indexed { base_vertex, base_instance, .. } => (Some(*base_vertex), Some(*base_instance)),
            // Copied from the indirect arguments before the pass (`before`); non-indexed draws have base vertex 0.
            DrawKind::Indirect { .. } => (Some(0), None),
            DrawKind::IndexedIndirect { .. } => (None, None),
            DrawKind::Patches(d) => (Some(0), Some(d.base_instance)),
            DrawKind::PatchesIndirect => (None, None),
        };
        let mut writes: Vec<(u64, u32)> = Vec::new();
        for group in &p.groups {
            for b in &group.bindings {
                let Source::Uniforms(block) = &b.source else { continue };
                for f in &block.fields {
                    let v = match f.value {
                        UniformValue::BaseVertex => match base_vertex {
                            Some(v) => i32::try_from(v)
                                .map_err(|_| RenderError::unimplemented(format!("base vertex {v} outside i32")))?
                                as u32,
                            None => continue,
                        },
                        UniformValue::BaseInstance => match base_instance {
                            Some(v) => u32_of(v, "base instance")?,
                            None => continue,
                        },
                        UniformValue::ResidualOffset { stage, index } => {
                            let (_, residual, _) = self.buffer_binding(stage, index, BufferBindingType::Storage)?;
                            residual as u32
                        }
                        UniformValue::BufferLength { stage, index } => {
                            let (_, _, length) = self.buffer_binding(stage, index, BufferBindingType::Storage)?;
                            u32_of(length, "buffer length")?
                        }
                        UniformValue::PatchControlPoints => match kind {
                            DrawKind::Patches(d) => u32_of(d.control_points, "control points")?,
                            _ => return Err(RenderError::invalid("a control-point count for a draw of no patches")),
                        },
                        UniformValue::ControlPointIndexOffset => match &self.patch_binding {
                            Some(PatchBinding { indices: Some(w), .. }) => u32_of(w.residual, "index offset")?,
                            _ => {
                                return Err(RenderError::invalid(
                                    "a control-point index offset for a draw without control-point indices",
                                ));
                            }
                        },
                    };
                    writes.push((base + f.offset, v));
                }
            }
        }
        for (o, v) in writes {
            self.arena.mirror[o as usize..o as usize + 4].copy_from_slice(&v.to_le_bytes());
        }
        Ok(Some(base))
    }

    /// The WebGPU binding of a Metal buffer slot: (buffer, aligned offset), the residual offset and Metal's length
    /// from the offset.
    fn buffer_binding(
        &self,
        stage: MetalStage,
        index: u64,
        ty: BufferBindingType,
    ) -> Result<(GpuBinding<A>, u64, u64)> {
        let bound = self.state.slots_ref(stage).buffers.get(&index).ok_or_else(|| {
            RenderError::undefined(
                DRAW,
                format!("the pipeline's {} function reads buffer {index}, which is not bound", stage.name()),
            )
        })?;
        match bound {
            Bound::Buffer { buffer, offset, .. } => {
                if *offset > buffer.length {
                    return Err(RenderError::undefined(
                        DRAW,
                        format!(
                            "{} buffer {index}: offset {offset} past the {}-byte buffer",
                            stage.name(),
                            buffer.length
                        ),
                    ));
                }
                let align = if ty == BufferBindingType::Uniform {
                    self.limits.uniform_align
                } else {
                    self.limits.storage_align
                };
                let base = offset - offset % align;
                let size = buffer.gpu().size() - base;
                Ok(((buffer.gpu().clone(), base, size), offset - base, buffer.length - offset))
            }
            Bound::Bytes { offset, len, .. } => {
                let arena = self.arena.buffer.clone().expect("inline bytes have an arena");
                Ok(((arena, *offset, len.max(&1).next_multiple_of(4)), 0, *len))
            }
        }
    }

    /// Bind everything a draw needs on the current pass: the pipeline variant, bind groups, vertex buffers and pass
    /// state. `values` are the stencil values; `cull`/`flip` override culling and winding for a per-primitive draw.
    fn prepare_draw(
        &mut self,
        p: &Rc<RenderPipeline<A>>,
        key: &VariantKey,
        values: FaceValues,
        uniforms: &Option<u64>,
        snapshot_views: Option<&HashMap<u32, A::TextureView>>,
    ) -> Result<()> {
        let mut key = key.clone();
        if let Some(ds) = &mut key.depth_stencil
            && self.targets.depth_stencil.as_ref().is_some_and(|a| a.stencil.is_some())
        {
            ds.read_mask = values.read_mask;
            ds.write_mask = values.write_mask;
        }
        let id = Rc::as_ptr(p) as usize;
        if self.applied.pipeline.as_ref() != Some(&(id, key.clone())) {
            let variant = p.variant(self.g, &key)?;
            self.call("GPURenderPassEncoder.setPipeline", |_| self.pass().set_pipeline(&variant));
            self.applied.pipeline = Some((id, key));
        }
        // Bind groups.
        for group in &p.groups {
            let mut buffers: Vec<(u32, GpuBinding<A>)> = Vec::new();
            let mut views: Vec<(u32, A::TextureView)> = Vec::new();
            let mut samplers: Vec<(u32, A::Sampler)> = Vec::new();
            for b in &group.bindings {
                match (&b.source, &b.resource) {
                    (Source::MetalBuffer { stage, index }, Resource::Buffer { ty, min_binding_size }) => {
                        let (mut binding, _, _) = self.buffer_binding(*stage, *index, *ty)?;
                        if *ty == BufferBindingType::Uniform {
                            binding.2 = binding.2.min(self.limits.max_uniform_binding);
                        } else if binding.2 > self.limits.max_storage_binding {
                            return Err(RenderError::host(format!(
                                "{} buffer {index}: a {}-byte binding exceeds maxStorageBufferBindingSize {}",
                                stage.name(),
                                binding.2,
                                self.limits.max_storage_binding
                            )));
                        }
                        if binding.2 < *min_binding_size {
                            return Err(RenderError::undefined(
                                DRAW,
                                format!(
                                    "{} buffer {index}: {} bytes from the offset, the function reads {min_binding_size}",
                                    stage.name(),
                                    binding.2
                                ),
                            ));
                        }
                        if let Some(Bound::Buffer { buffer, offset, .. }) =
                            self.state.slots_ref(*stage).buffers.get(index).cloned()
                        {
                            if b.writable() {
                                self.notes.written.push((buffer.clone(), (offset, buffer.length)));
                            }
                            self.notes.read.push(buffer);
                        }
                        buffers.push((b.binding, binding));
                    }
                    (
                        source @ (Source::MetalTexture { stage, index }
                        | Source::MetalTextureLevel { stage, index, .. }),
                        resource,
                    ) => {
                        let level = match source {
                            Source::MetalTextureLevel { level, .. } => Some(*level),
                            _ => None,
                        };
                        let tex = self.state.slots_ref(*stage).textures.get(index).cloned().ok_or_else(|| {
                            RenderError::undefined(
                                DRAW,
                                format!(
                                    "the pipeline's {} function reads texture {index}, which is not bound",
                                    stage.name()
                                ),
                            )
                        })?;
                        if let Some(a) = &b.texture {
                            // The type and format the function sees: a view's own.
                            let kind = crate::pipeline::bound::texture_type_value(tex.geometry().kind);
                            if let Some(why) =
                                crate::pipeline::bound::texture_mismatch(a, *index, kind, u64::from(tex.format()))
                            {
                                return Err(RenderError::undefined(
                                    DRAW,
                                    format!("the {} function's {why}", stage.name()),
                                ));
                            }
                        }
                        self.copies.add(tex.texture(), tex.alias().map(|a| &**a), tex.root_window(), b.writable())?;
                        let view = self.texture_view(&tex, resource, *stage, *index, level, p.bound)?;
                        if b.writable() {
                            // A storage binding is one level of the texture or view.
                            let (levels, slices) = tex.root_window();
                            for layer in slices {
                                self.notes.textures.push((tex.clone(), levels.start + level.unwrap_or(0), layer));
                            }
                        }
                        views.push((b.binding, view));
                    }
                    (Source::MetalSampler { stage, index }, Resource::Sampler { ty }) => {
                        let s = self.state.slots_ref(*stage).samplers.get(index).cloned().ok_or_else(|| {
                            RenderError::undefined(
                                DRAW,
                                format!(
                                    "the pipeline's {} function uses sampler {index}, which is not bound",
                                    stage.name()
                                ),
                            )
                        })?;
                        let gpu = s
                            .sampler(*ty)
                            .map_err(|why| RenderError::host(format!("{} sampler {index}: {why}", stage.name())))?;
                        samplers.push((b.binding, gpu.clone()));
                    }
                    (Source::Uniforms(block), Resource::Buffer { .. }) => {
                        let base = uniforms.expect("a pipeline with a uniform block has a uniform slot");
                        let arena = self.arena.buffer.clone().expect("uniform blocks live in the arena");
                        buffers.push((b.binding, (arena, base, block.size.next_multiple_of(16))));
                    }
                    (Source::TessellationVertices, Resource::Buffer { .. }) => {
                        let PatchBinding { vertices, .. } = self.patch_binding.as_ref().ok_or_else(|| {
                            RenderError::invalid("tessellated points bound for a draw that is not a patch draw")
                        })?;
                        buffers.push((b.binding, (vertices.clone(), 0, vertices.size())));
                    }
                    (Source::ControlPointIndices, Resource::Buffer { .. }) => {
                        let Some(PatchBinding { indices: Some(w), .. }) = self.patch_binding.as_ref() else {
                            return Err(RenderError::invalid("control-point indices bound for a draw without them"));
                        };
                        buffers.push((b.binding, (w.buffer.gpu().clone(), w.base, w.size)));
                    }
                    (Source::FetchSnapshot { attachment }, Resource::Texture { .. }) => {
                        let view = snapshot_views.and_then(|v| v.get(attachment)).cloned().ok_or_else(|| {
                            RenderError::invalid(format!("no snapshot of color attachment {attachment} for this draw"))
                        })?;
                        views.push((b.binding, view));
                    }
                    (s, r) => {
                        return Err(RenderError::unimplemented(format!("binding source {s:?} for resource {r:?}")));
                    }
                }
            }
            let mut entries: Vec<BindGroupEntry<'_, A>> = Vec::new();
            for (binding, (buffer, offset, size)) in &buffers {
                let mut bb = BufferBinding::new(buffer);
                bb.offset = size64(*offset, "GPUDevice.createBindGroup")?;
                bb.size = Some(size64(*size, "GPUDevice.createBindGroup")?);
                entries.push(BindGroupEntry::new(*binding, BindingResource::BufferBinding(bb)));
            }
            for (binding, v) in &views {
                entries.push(BindGroupEntry::new(*binding, BindingResource::TextureView(v)));
            }
            for (binding, s) in &samplers {
                entries.push(BindGroupEntry::new(*binding, BindingResource::Sampler(s)));
            }
            let mut d = BindGroupDescriptor::new(&group.layout, entries);
            d.label = format!("{} group {}", self.label, group.index);
            let bg = self.call("GPUDevice.createBindGroup", |dev| dev.create_bind_group(&d));
            self.call("GPURenderPassEncoder.setBindGroup", |_| self.pass().set_bind_group(group.index, Some(&bg), &[]));
        }
        // Vertex buffers.
        for (slot, vb) in p.vertex_buffers.iter().enumerate() {
            let bound = self.state.vertex.buffers.get(&vb.metal_index).cloned().ok_or_else(|| {
                RenderError::undefined(
                    DRAW,
                    format!("vertex buffer {} of the vertex descriptor is not bound", vb.metal_index),
                )
            })?;
            let (buffer, offset) = match bound {
                Bound::Buffer { buffer, offset, .. } => {
                    self.notes.read.push(buffer.clone());
                    (buffer.gpu().clone(), offset)
                }
                Bound::Bytes { offset, .. } => (self.arena.buffer.clone().expect("arena"), offset),
            };
            if offset % 4 != 0 || offset > buffer.size() {
                return Err(RenderError::unimplemented(format!(
                    "vertex buffer {} offset {offset}: WebGPU requires a multiple of 4 within the buffer for native \
                     vertex input (tables/limits.toml question 5.10-vertex-buffer-alignment)",
                    vb.metal_index
                )));
            }
            let o = size64(offset, "GPURenderPassEncoder.setVertexBuffer")?;
            self.call("GPURenderPassEncoder.setVertexBuffer", |_| {
                self.pass().set_vertex_buffer(slot as u32, Some(&buffer), o, None)
            });
        }
        // Pass state.
        let vp = self.state.viewport;
        let vp_bits = vp.map(f32::to_bits);
        if self.applied.viewport != Some(vp_bits) {
            let f = |v: f32| Float32::new(v).expect("finite, checked");
            self.call("GPURenderPassEncoder.setViewport", |_| {
                self.pass().set_viewport(f(vp[0]), f(vp[1]), f(vp[2]), f(vp[3]), f(vp[4]), f(vp[5]))
            });
            self.applied.viewport = Some(vp_bits);
        }
        let sc = self.state.scissor;
        if self.applied.scissor != Some(sc) {
            self.call("GPURenderPassEncoder.setScissorRect", |_| {
                self.pass().set_scissor_rect(sc[0], sc[1], sc[2], sc[3])
            });
            self.applied.scissor = Some(sc);
        }
        let bc = match p.blend_constant {
            BlendConstant::Color => self.state.blend,
            BlendConstant::AlphaSplat => [self.state.blend[3]; 4],
        };
        let bc_bits = bc.map(f32::to_bits);
        if self.applied.blend != Some(bc_bits) {
            if bc.iter().any(|c| !c.is_finite()) {
                return Err(RenderError::unimplemented("a blend color that is not finite"));
            }
            let f = |v: f32| Float64::new(f64::from(v)).expect("finite");
            let color = Color::ColorDict(ColorDict { r: f(bc[0]), g: f(bc[1]), b: f(bc[2]), a: f(bc[3]) });
            self.call("GPURenderPassEncoder.setBlendConstant", |_| self.pass().set_blend_constant(&color))
                .map_err(|e| RenderError::gpu("GPURenderPassEncoder.setBlendConstant", e))?;
            self.applied.blend = Some(bc_bits);
        }
        if self.applied.reference != Some(values.reference) {
            self.call("GPURenderPassEncoder.setStencilReference", |_| {
                self.pass().set_stencil_reference(values.reference)
            });
            self.applied.reference = Some(values.reference);
        }
        Ok(())
    }

    /// The view of a bound texture for a binding: `level` is a storage binding's level (counted from a view's first
    /// level), `None` a binding of the texture itself. `bound` when the pipeline was compiled for the textures the draw
    /// binds ([`RenderPipeline::bound`]): its bindings then have the bound texture's storage format and levels, and
    /// its shader converts emulated formats and applies swizzles the views do not.
    fn texture_view(
        &self,
        tex: &Tex<A>,
        resource: &Resource,
        stage: MetalStage,
        index: u64,
        level: Option<u32>,
        bound: bool,
    ) -> Result<A::TextureView> {
        let t: &Texture<A> = tex.texture();
        // The root's subresources the binding sees: every level and slice of a texture, a view's own.
        let (levels, slices) = tex.root_window();
        let sees =
            |level: u32, slice: u32| levels.contains(&level) && (t.geometry.kind.is_3d() || slices.contains(&slice));
        for c in self.targets.colors.iter().flatten() {
            if Rc::ptr_eq(c.texture.texture(), tex.texture()) && sees(c.level, c.slice) {
                return Err(RenderError::unimplemented(format!(
                    "{} texture {index} is an attachment of the pass: WebGPU forbids a texture binding and an \
                     attachment of one subresource in a pass, and the binding sees the attachment's subresource",
                    stage.name()
                )));
            }
        }
        if let Some(ds) = &self.targets.depth_stencil
            && Rc::ptr_eq(ds.texture.texture(), tex.texture())
            && sees(ds.level, ds.slice)
        {
            return Err(RenderError::unimplemented(format!(
                "{} texture {index} is the depth-stencil attachment of the pass",
                stage.name()
            )));
        }
        match tex.resolution() {
            // An X32_Stencil8 view reads the stencil aspect, a WebGPU view of its own.
            Resolution::Native(_) | Resolution::StencilAspect => {}
            // A format read with its components rearranged: the view's swizzle below.
            Resolution::Emulated(e) if matches!(e.read, Read::Native | Read::Swizzle(_)) => {}
            // The variant's shader reads and writes the stored format as the bound pixel format.
            Resolution::Emulated(_) if bound => {}
            _ => {
                return Err(RenderError::unimplemented(format!(
                    "{} texture {index}: a texture whose format is emulated with a shader conversion, bound to a \
                     pipeline whose code was not compiled for the bound textures",
                    stage.name()
                )));
            }
        }
        let mut d = TextureViewDescriptor::new();
        d.label = format!("{} {} texture {index}", t.label, stage.name());
        match resource {
            Resource::Texture { sample_type, view_dimension, multisampled } => {
                if *multisampled != (t.geometry.sample_count > 1) {
                    return Err(RenderError::undefined(
                        DRAW,
                        format!(
                            "{} texture {index}: the function's texture type and the texture's sample count disagree",
                            stage.name()
                        ),
                    ));
                }
                d.dimension = Some(*view_dimension);
                d.aspect = super::attachments::binding_aspect(
                    crate::mem::texture::idl_format(t.resolution.storage()),
                    *sample_type == crate::gpu::idl::TextureSampleType::Depth
                        || *sample_type == crate::gpu::idl::TextureSampleType::UnfilterableFloat,
                );
            }
            Resource::StorageTexture { view_dimension, format, .. } if bound => {
                // A storage binding shows one level (WebGPU 8.2.1): a level binding's own, counted from a view's
                // first level. A binding of the texture itself shows its level 0, which is all of it only when it has
                // one level.
                let levels_of = tex.geometry().mip_levels;
                let level = match level {
                    Some(l) if l < levels_of => l,
                    Some(l) => {
                        return Err(RenderError::invalid(format!(
                            "the variant binds level {l} of {} texture {index}, which has {levels_of} levels; the \
                             variant key holds the level count",
                            stage.name()
                        )));
                    }
                    None if levels_of > 1 => {
                        return Err(RenderError::unimplemented(format!(
                            "{} texture {index} is a mipmapped texture bound for writing; the binding layout does not \
                             say which level the function writes",
                            stage.name()
                        )));
                    }
                    None => 0,
                };
                let writable = self.memory.features().allows(tex.resolution().storage().caps().storage_wo);
                if tex.gpu().usage() & TextureUsage::STORAGE_BINDING == 0 || !writable {
                    return Err(if t.usage != texture_usage::UNKNOWN && t.usage & texture_usage::SHADER_WRITE == 0 {
                        RenderError::undefined(
                            DRAW,
                            format!("{} texture {index} is written and lacks MTLTextureUsageShaderWrite", stage.name()),
                        )
                    } else {
                        RenderError::host(format!(
                            "{} texture {index}: this host cannot write {} textures from shaders",
                            stage.name(),
                            tex.resolution().storage().name()
                        ))
                    });
                }
                d.dimension = Some(*view_dimension);
                d.format = Some(*format);
                d.base_mip_level = level;
                d.mip_level_count = Some(1);
                if let Some(v) = tex.view() {
                    let stored = crate::mem::texture::idl_format(v.resolution.storage());
                    if *format != stored {
                        return Err(RenderError::unimplemented(format!(
                            "{} texture {index} is a view stored as {} that the function writes as {}",
                            stage.name(),
                            stored.as_str(),
                            format.as_str()
                        )));
                    }
                    v.describe(&mut d, level..level + 1, TextureUsage::STORAGE_BINDING);
                    d.format = Some(*format);
                }
            }
            Resource::StorageTexture { view_dimension, .. } => {
                // A pipeline whose code was not compiled for the bound textures has a written texture's binding of
                // one level: its write at a level computed at run time selects level 0 alone, and a level out of a
                // texture's range is not known to be out of range. Only a texture of one level, written at level 0,
                // is written as Metal writes it.
                if tex.geometry().mip_levels != 1 {
                    return Err(RenderError::unimplemented(format!(
                        "{} texture {index}: a read-write or write-only binding of a texture with {} levels, bound to \
                         a pipeline whose code was not compiled for the bound textures' level counts",
                        stage.name(),
                        tex.geometry().mip_levels
                    )));
                }
                if level.unwrap_or(0) != 0 {
                    return Err(RenderError::unimplemented(format!(
                        "{} texture {index}: the function writes level {} of a texture of one level; Metal ignores \
                         the write, and the pipeline's code was not compiled for the bound textures' level counts",
                        stage.name(),
                        level.unwrap_or(0)
                    )));
                }
                if tex.gpu().usage() & TextureUsage::STORAGE_BINDING == 0 {
                    return Err(RenderError::undefined(
                        DRAW,
                        format!("{} texture {index} was not created with MTLTextureUsageShaderWrite", stage.name()),
                    ));
                }
                if let Some(f) = tex.view().and_then(|v| v.view_format()) {
                    return Err(RenderError::host(format!(
                        "{} texture {index}: this host cannot write {} textures from shaders",
                        stage.name(),
                        f.as_str()
                    )));
                }
                d.dimension = Some(*view_dimension);
                d.mip_level_count = Some(1);
                if let Some(v) = tex.view() {
                    v.describe(&mut d, 0..1, TextureUsage::STORAGE_BINDING);
                }
            }
            _ => return Err(RenderError::invalid("a texture source for a non-texture binding")),
        }
        if let Some(v) = tex.view()
            && let Some(dimension) = d.dimension
            && !v.takes(dimension)
        {
            return Err(RenderError::undefined(
                DRAW,
                format!(
                    "{} texture {index} is a {:?} view of {} slices, and the function's texture type takes a {} view",
                    stage.name(),
                    v.geometry.kind,
                    v.geometry.slices(),
                    dimension.as_str()
                ),
            ));
        }
        if let (Some(v), Resource::Texture { .. }) = (tex.view(), resource) {
            v.describe(&mut d, 0..v.geometry.mip_levels, TextureUsage::TEXTURE_BINDING);
        }
        // The swizzle reads and samples apply (writes take none); depth reads take none, as in the compiler. The
        // view applies it when the host has texture-component-swizzle; otherwise a pipeline compiled for the bound
        // textures applies it in its shader, and one that was not cannot.
        let swizzle = read_swizzle(tex.resolution(), tex.swizzle());
        if let Resource::Texture { sample_type, .. } = resource
            && swizzle != IDENTITY_SWIZZLE
            && *sample_type != crate::gpu::idl::TextureSampleType::Depth
        {
            if self.memory.features().has(Feature::TextureComponentSwizzle) {
                d.swizzle = swizzle_string(swizzle);
            } else if !bound {
                return Err(RenderError::unimplemented(format!(
                    "{} texture {index}: swizzle {swizzle:?} needs texture-component-swizzle on this host, and the \
                     pipeline's code was not compiled for the bound textures to apply it",
                    stage.name()
                )));
            }
        }
        self.call("GPUTexture.createView", |_| tex.gpu().create_view(&d))
            .map_err(|e| RenderError::gpu("GPUTexture.createView", e))
    }

    /// Issue one draw record on the current pass.
    fn issue(&mut self, ordinal: usize, kind: &DrawKind, prim: PrimitiveType) -> Result<()> {
        match kind {
            DrawKind::Patches(_) => {
                let g = self
                    .tessellated
                    .get(&ordinal)
                    .cloned()
                    .ok_or_else(|| RenderError::invalid("a patch draw the scan did not tessellate"))?;
                self.call("GPURenderPassEncoder.setIndexBuffer", |_| {
                    self.pass().set_index_buffer(&g.indices, IndexFormat::Uint32, Size64::from(0u32), None)
                });
                self.call("GPURenderPassEncoder.drawIndexedIndirect", |_| {
                    self.pass().draw_indexed_indirect(&g.draw, Size64::from(0u32))
                });
            }
            DrawKind::PatchesIndirect => return Err(RenderError::invalid("an indirect patch draw past the scan")),
            DrawKind::Direct { start, count, instances, base_instance } => {
                let (start, count) = (u32_of(*start, "vertex start")?, u32_of(*count, "vertex count")?);
                let (instances, base) =
                    (u32_of(*instances, "instance count")?, u32_of(*base_instance, "base instance")?);
                start.checked_add(count).ok_or_else(|| RenderError::unimplemented("vertex range beyond 2^32"))?;
                self.call("GPURenderPassEncoder.draw", |_| self.pass().draw(count, instances, start, base));
            }
            DrawKind::Indexed { count, index_type, buffer: h, offset, instances, base_vertex, base_instance } => {
                if let Some(staged) = self.staged.get(&ordinal) {
                    if *base_instance != 0 && !self.limits.indirect_first_instance {
                        return Err(RenderError::host(
                            "an indexed point, line or triangle list draw with a base instance: primitive restart is \
                             applied by staging the indices and drawing indirectly, and a non-zero first instance in \
                             an indirect draw needs the indirect-first-instance feature",
                        ));
                    }
                    let (indices, args) = (staged.indices.clone(), staged.args.clone());
                    self.call("GPURenderPassEncoder.setIndexBuffer", |_| {
                        self.pass().set_index_buffer(&indices, IndexFormat::Uint32, Size64::from(0u32), None)
                    });
                    self.call("GPURenderPassEncoder.drawIndexedIndirect", |_| {
                        self.pass().draw_indexed_indirect(&args, Size64::from(0u32))
                    });
                    return Ok(());
                }
                let it = IndexType::from_value(*index_type, DRAW)?;
                let b = buffer::<A>(self.objects, h.expect("checked in scan"), DRAW)?;
                let end = offset + count * it.size();
                if end > b.length {
                    return Err(RenderError::undefined(
                        DRAW,
                        format!("indices {offset}..{end} of a {}-byte index buffer", b.length),
                    ));
                }
                let count = u32_of(*count, "index count")?;
                let instances = u32_of(*instances, "instance count")?;
                let base = u32_of(*base_instance, "base instance")?;
                let bv = i32::try_from(*base_vertex)
                    .map_err(|_| RenderError::unimplemented(format!("base vertex {base_vertex} outside i32")))?;
                let o = size64(*offset, "GPURenderPassEncoder.setIndexBuffer")?;
                self.call("GPURenderPassEncoder.setIndexBuffer", |_| {
                    self.pass().set_index_buffer(b.gpu(), it.format(), o, None)
                });
                self.call("GPURenderPassEncoder.drawIndexed", |_| {
                    self.pass().draw_indexed(count, instances, 0, bv, base)
                });
                self.notes.read.push(b);
            }
            DrawKind::Indirect { buffer: h, offset } => {
                self.require_first_instance()?;
                let b = buffer::<A>(self.objects, h.expect("checked in scan"), DRAW)?;
                let o = size64(*offset, "GPURenderPassEncoder.drawIndirect")?;
                self.call("GPURenderPassEncoder.drawIndirect", |_| self.pass().draw_indirect(b.gpu(), o));
                self.notes.read.push(b);
            }
            DrawKind::IndexedIndirect { index_type, buffer: h, offset, indirect, indirect_offset } => {
                self.require_first_instance()?;
                if let Some(staged) = self.staged.get(&ordinal) {
                    let (indices, args) = (staged.indices.clone(), staged.args.clone());
                    self.call("GPURenderPassEncoder.setIndexBuffer", |_| {
                        self.pass().set_index_buffer(&indices, IndexFormat::Uint32, Size64::from(0u32), None)
                    });
                    self.call("GPURenderPassEncoder.drawIndexedIndirect", |_| {
                        self.pass().draw_indexed_indirect(&args, Size64::from(0u32))
                    });
                    return Ok(());
                }
                debug_assert!(prim.is_strip());
                let it = IndexType::from_value(*index_type, DRAW)?;
                let b = buffer::<A>(self.objects, h.expect("checked in scan"), DRAW)?;
                let ib = buffer::<A>(self.objects, indirect.expect("checked in scan"), DRAW)?;
                let o = size64(*offset, "GPURenderPassEncoder.setIndexBuffer")?;
                self.call("GPURenderPassEncoder.setIndexBuffer", |_| {
                    self.pass().set_index_buffer(b.gpu(), it.format(), o, None)
                });
                let io = size64(*indirect_offset, "GPURenderPassEncoder.drawIndexedIndirect")?;
                self.call("GPURenderPassEncoder.drawIndexedIndirect", |_| {
                    self.pass().draw_indexed_indirect(ib.gpu(), io)
                });
                self.notes.read.push(b);
                self.notes.read.push(ib);
            }
        }
        Ok(())
    }

    fn require_first_instance(&self) -> Result<()> {
        if self.limits.indirect_first_instance {
            Ok(())
        } else {
            Err(RenderError::host(
                "an indirect draw on a host without indirect-first-instance: WebGPU treats a non-zero first instance \
                 as a no-op (tables/limits.toml question 6.5-indirect-draw-layout), and reading the arguments back \
                 mid-command-buffer is not implemented",
            ))
        }
    }

    /// A non-indexed direct draw made one primitive at a time: for framebuffer fetch each primitive is drawn in a
    /// WebGPU pass of its own after a snapshot of the fetched attachments; for separate stencil faces each primitive is
    /// drawn once per face with the other face culled.
    fn draw_primitives(
        &mut self,
        p: &Rc<RenderPipeline<A>>,
        key: &VariantKey,
        faces: FacePlan,
        uniforms: Option<u64>,
        prim: PrimitiveType,
        (start, count, instances, base_instance): (u64, u64, u64, u64),
    ) -> Result<()> {
        let start = u32_of(start, "vertex start")?;
        let count = u32_of(count, "vertex count")?;
        let instances = u32_of(instances, "instance count")?;
        let base = u32_of(base_instance, "base instance")?;
        let fetched = match &p.passes {
            PassPlan::PerPrimitive { fetched } => Some(fetched.clone()),
            PassPlan::Direct | PassPlan::Tessellation { .. } => None,
        };
        let n = prim.vertices_per_primitive();
        let primitives = prim.primitives(count);
        let step = if prim.is_strip() { 1 } else { n };
        for instance in 0..instances {
            for i in 0..primitives {
                let first = start + i * step;
                // A strip's odd triangles have the opposite vertex order of the same triangle drawn alone; flipping the
                // winding gives the lone triangle the strip's facing.
                let flip = prim == PrimitiveType::TriangleStrip && i % 2 == 1;
                let snapshots = match &fetched {
                    Some(attachments) => Some(self.snapshot(attachments)?),
                    None => None,
                };
                let mut k = key.clone();
                if flip {
                    k.front_face = if k.front_face == crate::gpu::idl::FrontFace::Cw {
                        crate::gpu::idl::FrontFace::Ccw
                    } else {
                        crate::gpu::idl::FrontFace::Cw
                    };
                }
                let draws: Vec<(VariantKey, FaceValues)> = match faces {
                    FacePlan::One(v) => vec![(k, v)],
                    FacePlan::Split(front, back) => {
                        let mut kf = k.clone();
                        kf.cull_mode = crate::gpu::idl::CullMode::Back;
                        let mut kb = k;
                        kb.cull_mode = crate::gpu::idl::CullMode::Front;
                        vec![(kf, front), (kb, back)]
                    }
                };
                for (k, v) in draws {
                    self.prepare_draw(p, &k, v, &uniforms, snapshots.as_ref())?;
                    self.call("GPURenderPassEncoder.draw", |_| self.pass().draw(n, 1, first, base + instance));
                }
            }
        }
        Ok(())
    }

    /// End the current pass, copy each fetched color attachment into its snapshot texture and begin the next pass.
    /// Returns the snapshots' views.
    fn snapshot(&mut self, attachments: &[u32]) -> Result<HashMap<u32, A::TextureView>> {
        if !self.splits {
            return Err(RenderError::invalid("a per-primitive draw the encoder's scan did not find"));
        }
        self.end_pass();
        let mut views = HashMap::new();
        for &a in attachments {
            let c = self.targets.colors.get(a as usize).and_then(|c| c.as_ref()).ok_or_else(|| {
                RenderError::undefined(DRAW, format!("the pipeline reads color attachment {a}, which the pass lacks"))
            })?;
            let t = c.texture.texture();
            let source = c.texture.gpu();
            if source.usage() & TextureUsage::COPY_SRC == 0 {
                return Err(RenderError::unimplemented(format!(
                    "framebuffer fetch of the memoryless color attachment {a}: the snapshot copy needs its contents"
                )));
            }
            let (w, h, _) = t.geometry.level_extent(c.level);
            // The snapshot reads as the attachment does: in a view's format (an sRGB sibling, copy-compatible).
            let format = c.texture.view().and_then(|v| v.view_format()).unwrap_or_else(|| source.format());
            let snapshot = match self.snapshots.get(&a) {
                Some(s) => s.clone(),
                None => {
                    let mut d = TextureDescriptor::new(
                        Extent3D::Extent3DDict(Extent3DDict { width: w, height: h, depth_or_array_layers: 1 }),
                        format,
                        // WebGPU requires RENDER_ATTACHMENT of a multisampled texture (createTexture validation).
                        TextureUsage::COPY_DST
                            | TextureUsage::TEXTURE_BINDING
                            | if t.geometry.sample_count > 1 { TextureUsage::RENDER_ATTACHMENT } else { 0 },
                    );
                    d.sample_count = t.geometry.sample_count;
                    d.label = format!("{} snapshot of color attachment {a}", self.label);
                    let s = self
                        .call("GPUDevice.createTexture", |dev| dev.create_texture(&d))
                        .map_err(|e| RenderError::gpu("GPUDevice.createTexture", e))?;
                    self.snapshots.insert(a, s.clone());
                    s
                }
            };
            let mut src = TexelCopyTextureInfo::<A>::new(source);
            src.mip_level = c.level;
            src.origin = Origin3D::Origin3DDict(Origin3DDict { x: 0, y: 0, z: c.depth_slice.unwrap_or(c.slice) });
            src.aspect = TextureAspect::All;
            let dst = TexelCopyTextureInfo::<A>::new(&snapshot);
            let size = Extent3D::Extent3DDict(Extent3DDict { width: w, height: h, depth_or_array_layers: 1 });
            self.call("GPUCommandEncoder.copyTextureToTexture", |_| {
                self.encoder.copy_texture_to_texture(&src, &dst, &size)
            })
            .map_err(|e| RenderError::gpu("GPUCommandEncoder.copyTextureToTexture", e))?;
            let mut vd = TextureViewDescriptor::new();
            vd.dimension = Some(TextureViewDimension::D2);
            let view = self
                .call("GPUTexture.createView", |_| snapshot.create_view(&vd))
                .map_err(|e| RenderError::gpu("GPUTexture.createView", e))?;
            views.insert(a, view);
        }
        self.begin_pass(false)?;
        Ok(views)
    }
}

/// Does a render-encoder barrier end the WebGPU pass? Apple-family GPUs accept barriers after the vertex stage only
/// (reference case `render.error.memory-barrier-after-fragment`). One that orders vertex-stage stores before later
/// fragment-stage loads ends the pass: a pass boundary is WebGPU's only ordering between draws. One before the vertex
/// stage alone orders vertex-stage stores against vertex-stage loads of later draws; a WebGPU vertex stage cannot
/// store (`tables/limits.toml` question `5.8-vertex-stage-writable-storage`, so the pipeline service refuses such
/// pipelines), and such a barrier has nothing to order.
fn splits_pass(after: u64, before: u64) -> bool {
    after & stages::VERTEX != 0 && before & stages::FRAGMENT != 0
}

fn reference(v: u32) -> Result<u32> {
    if v > 0xff {
        return Err(RenderError::unimplemented(format!(
            "stencil reference {v:#x} wider than the 8-bit stencil: Metal's handling of the high bits is not established"
        )));
    }
    Ok(v)
}

fn store_options(options: u64) -> Result<()> {
    match options {
        0 => Ok(()),
        _ => Err(RenderError::host(
            "custom-sample-positions: MTLStoreActionOptionCustomSamplePositions; WebGPU mandates the standard sample \
             pattern (tables/refusals.md)",
        )),
    }
}

fn check_index(stage: MetalStage, index: u64, limit: u64, what: &str) -> Result<()> {
    if index >= limit {
        return Err(RenderError::undefined(
            "-[MTLRenderCommandEncoder set...]",
            format!(
                "{} {what} index {index}: the A10X's argument table has {limit} entries (device profile)",
                stage.name()
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::metal::{CompareFunction, StencilOperation};

    fn face(compare: CompareFunction, pass: StencilOperation, read: u32, write: u32) -> Face {
        Face {
            compare,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass,
            read_mask: read,
            write_mask: write,
        }
    }

    #[test]
    fn faces_split_only_when_the_drawn_faces_disagree() {
        let equal = face(CompareFunction::Equal, StencilOperation::Keep, 0xff, 0xff);
        let ds = DepthStencilState { front: Some(equal), back: Some(equal), ..DepthStencilState::DEFAULT };
        // Different references used by both faces: split, unless culling leaves one face.
        assert!(matches!(face_plan(&ds, (1, 2), CullMode::None, true).unwrap(), FacePlan::Split(..)));
        match face_plan(&ds, (1, 2), CullMode::Back, true).unwrap() {
            FacePlan::One(v) => assert_eq!(v.reference, 1),
            p => panic!("{p:?}"),
        }
        match face_plan(&ds, (1, 2), CullMode::Front, true).unwrap() {
            FacePlan::One(v) => assert_eq!(v.reference, 2),
            p => panic!("{p:?}"),
        }
        // A face that does not read the reference does not force a split.
        let always = face(CompareFunction::Always, StencilOperation::Invert, 0xff, 0xff);
        let ds = DepthStencilState { front: Some(equal), back: Some(always), ..DepthStencilState::DEFAULT };
        match face_plan(&ds, (1, 2), CullMode::None, true).unwrap() {
            FacePlan::One(v) => assert_eq!(v.reference, 1),
            p => panic!("{p:?}"),
        }
        // Different write masks of two writing faces split.
        let a = face(CompareFunction::Always, StencilOperation::Invert, 0xff, 0x0f);
        let b = face(CompareFunction::Always, StencilOperation::Invert, 0xff, 0xf0);
        let ds = DepthStencilState { front: Some(a), back: Some(b), ..DepthStencilState::DEFAULT };
        assert!(matches!(face_plan(&ds, (0, 0), CullMode::None, true).unwrap(), FacePlan::Split(..)));
        // Points and lines cannot use differing faces.
        assert!(face_plan(&ds, (0, 0), CullMode::None, false).is_err());
        let ds = DepthStencilState { front: Some(equal), back: Some(equal), ..DepthStencilState::DEFAULT };
        assert!(face_plan(&ds, (3, 3), CullMode::None, false).is_ok());
        assert!(face_plan(&ds, (3, 4), CullMode::None, false).is_err());
    }
}
