//! The executor's own GPU programs for the render path, created once per device on first use:
//!
//! - index staging: re-indexes an indexed draw of a list topology with Metal's primitive restart (see `index`);
//! - visibility merge: combines the occlusion results of one Metal visibility offset that several WebGPU queries
//!   measured (see `visibility`);
//! - depth and stencil resolve: the manual resolve passes of `resolve`.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::gpu::idl::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferBindingLayout, BufferBindingType,
    CompareFunction as GpuCompare, ComputePipelineDescriptor, DepthStencilState, FragmentState,
    PipelineLayoutDescriptor, PipelineLayoutOrAuto, ProgrammableStage, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderStage, StencilFaceState, StencilOperation as GpuStencilOp, TextureBindingLayout,
    TextureFormat, TextureSampleType, TextureViewDimension, VertexState,
};
use crate::gpu::{Api, Device as _, Float64, Size64};
use crate::mem::calls::Calls;

use super::error::{RenderError, Result};
use super::metal::DepthResolveFilter;

/// Index staging: one invocation walks the index range in order, as Metal's primitive assembly does, and writes the
/// indices of every complete primitive followed by the `drawIndexedIndirect` arguments that draw them.
pub const INDEX_STAGING_WGSL: &str = r#"
struct Params {
    first_byte: u32,
    index_size: u32,
    per_primitive: u32,
    indirect: u32,
    count: u32,
    instance_count: u32,
    base_vertex: u32,
    base_instance: u32,
    restart: u32,
    args_word: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> args_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read_write> args_out: array<u32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(1)
fn main() {
    var count = p.count;
    var first = 0u;
    var instances = p.instance_count;
    var base_vertex = p.base_vertex;
    var base_instance = p.base_instance;
    if (p.indirect != 0u) {
        count = args_in[p.args_word];
        instances = args_in[p.args_word + 1u];
        first = args_in[p.args_word + 2u];
        base_vertex = args_in[p.args_word + 3u];
        base_instance = args_in[p.args_word + 4u];
    }
    var out = 0u;
    var run = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        let byte = p.first_byte + (first + i) * p.index_size;
        let word = src[byte / 4u];
        var v = word;
        if (p.index_size == 2u) {
            v = (word >> ((byte & 2u) * 8u)) & 0xffffu;
        }
        if (v == p.restart) {
            out = out - run;
            run = 0u;
            continue;
        }
        dst[out] = v;
        out = out + 1u;
        run = run + 1u;
        if (run == p.per_primitive) {
            run = 0u;
        }
    }
    out = out - run;
    args_out[0] = out;
    args_out[1] = instances;
    args_out[2] = 0u;
    args_out[3] = base_vertex;
    args_out[4] = base_instance;
}
"#;

/// Visibility merge: each target is one Metal visibility offset (a word index in the bound window of the visibility
/// buffer) with the list of WebGPU query results that measured it; the target gets their bitwise OR, which is nonzero
/// exactly when one of them is.
pub const VISIBILITY_MERGE_WGSL: &str = r#"
struct Target {
    word: u32,
    first: u32,
    count: u32,
    _pad: u32,
}
@group(0) @binding(0) var<storage, read> results: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> targets: array<Target>;
@group(0) @binding(2) var<storage, read> queries: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&targets)) {
        return;
    }
    let t = targets[id.x];
    var v = vec2<u32>(0u, 0u);
    for (var i = 0u; i < t.count; i = i + 1u) {
        v = v | results[queries[t.first + i]];
    }
    out[t.word] = v.x;
    out[t.word + 1u] = v.y;
}
"#;

/// Depth and stencil resolve: a triangle covering the target, and fragment stages that read the multisampled
/// attachment at the fragment's pixel. `fs_depth` writes the filtered depth (`filter`: 0 sample 0, 1 minimum,
/// 2 maximum, `MTLMultisampleDepthResolveFilter`); `fs_stencil` keeps the fragment only where bit `bit` of sample 0's
/// stencil value is set, so that eight draws with write mask `1 << bit` and the reference 0xff rebuild the value.
pub const RESOLVE_WGSL: &str = r#"
override resolve_filter: u32 = 0u;
override bit: u32 = 0u;
@group(0) @binding(0) var depth_src: texture_depth_multisampled_2d;
@group(0) @binding(1) var stencil_src: texture_multisampled_2d<u32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_depth(@builtin(position) p: vec4<f32>) -> @builtin(frag_depth) f32 {
    let c = vec2<i32>(p.xy);
    var d = textureLoad(depth_src, c, 0);
    let n = i32(textureNumSamples(depth_src));
    for (var s = 1; s < n; s = s + 1) {
        let v = textureLoad(depth_src, c, s);
        if (resolve_filter == 1u) {
            d = min(d, v);
        } else if (resolve_filter == 2u) {
            d = max(d, v);
        }
    }
    return d;
}

@fragment
fn fs_stencil(@builtin(position) p: vec4<f32>) {
    let v = textureLoad(stencil_src, vec2<i32>(p.xy), 0).x;
    if ((v & (1u << bit)) == 0u) {
        discard;
    }
}
"#;

fn gpu_err(op: &'static str) -> impl Fn(crate::gpu::Exception) -> RenderError {
    move |e| RenderError::gpu(op, e)
}

fn storage_entry(binding: u32, ty: BufferBindingType, visibility: u32) -> BindGroupLayoutEntry {
    let mut e = BindGroupLayoutEntry::new(binding, visibility);
    let mut b = BufferBindingLayout::new();
    b.r#type = ty;
    e.buffer = Some(b);
    e
}

/// A compute program: its bind group layout and pipeline.
pub struct Compute<A: Api> {
    pub layout: A::BindGroupLayout,
    pub pipeline: A::ComputePipeline,
}

/// The resolve programs: the bind group layouts and pipelines for one target format.
pub struct Resolve<A: Api> {
    pub depth_layout: A::BindGroupLayout,
    pub stencil_layout: A::BindGroupLayout,
    depth_pipeline_layout: A::PipelineLayout,
    stencil_pipeline_layout: A::PipelineLayout,
    pipelines: HashMap<(TextureFormat, ResolveKind), A::RenderPipeline>,
}

/// Which resolve pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResolveKind {
    Depth(u32),
    StencilBit(u32),
}

/// The programs, created on first use.
pub struct Helpers<A: Api> {
    index: RefCell<Option<std::rc::Rc<Compute<A>>>>,
    visibility: RefCell<Option<std::rc::Rc<Compute<A>>>>,
    resolve_module: RefCell<Option<A::ShaderModule>>,
    resolve: RefCell<Option<Resolve<A>>>,
}

impl<A: Api> Default for Helpers<A> {
    fn default() -> Self {
        Helpers {
            index: RefCell::new(None),
            visibility: RefCell::new(None),
            resolve_module: RefCell::new(None),
            resolve: RefCell::new(None),
        }
    }
}

fn compute<A: Api>(
    g: &impl Calls<A>,
    label: &str,
    wgsl: &str,
    entries: Vec<BindGroupLayoutEntry>,
) -> Result<Compute<A>> {
    let mut md = ShaderModuleDescriptor::new(wgsl.to_string());
    md.label = label.to_string();
    let module = g.call("GPUDevice.createShaderModule", label, |d| d.create_shader_module(&md));
    let mut ld = BindGroupLayoutDescriptor::new(entries);
    ld.label = label.to_string();
    let layout = g
        .call("GPUDevice.createBindGroupLayout", label, |d| d.create_bind_group_layout(&ld))
        .map_err(gpu_err("GPUDevice.createBindGroupLayout"))?;
    let mut pd = PipelineLayoutDescriptor::new(vec![Some(&layout)]);
    pd.label = label.to_string();
    let pl = g.call("GPUDevice.createPipelineLayout", label, |d| d.create_pipeline_layout(&pd));
    let mut stage = ProgrammableStage::new(&module);
    stage.entry_point = Some("main".into());
    let mut cd = ComputePipelineDescriptor::new(PipelineLayoutOrAuto::PipelineLayout(&pl), stage);
    cd.label = label.to_string();
    let pipeline = g.call("GPUDevice.createComputePipeline", label, |d| d.create_compute_pipeline(&cd));
    Ok(Compute { layout, pipeline })
}

impl<A: Api> Helpers<A> {
    pub fn index_staging(&self, g: &impl Calls<A>) -> Result<std::rc::Rc<Compute<A>>> {
        if let Some(c) = self.index.borrow().as_ref() {
            return Ok(c.clone());
        }
        let c = ShaderStage::COMPUTE;
        let entries = vec![
            storage_entry(0, BufferBindingType::ReadOnlyStorage, c),
            storage_entry(1, BufferBindingType::ReadOnlyStorage, c),
            storage_entry(2, BufferBindingType::Storage, c),
            storage_entry(3, BufferBindingType::Storage, c),
            storage_entry(4, BufferBindingType::Uniform, c),
        ];
        let p = std::rc::Rc::new(compute(g, "webmetal index staging", INDEX_STAGING_WGSL, entries)?);
        *self.index.borrow_mut() = Some(p.clone());
        Ok(p)
    }

    pub fn visibility_merge(&self, g: &impl Calls<A>) -> Result<std::rc::Rc<Compute<A>>> {
        if let Some(c) = self.visibility.borrow().as_ref() {
            return Ok(c.clone());
        }
        let c = ShaderStage::COMPUTE;
        let entries = vec![
            storage_entry(0, BufferBindingType::ReadOnlyStorage, c),
            storage_entry(1, BufferBindingType::ReadOnlyStorage, c),
            storage_entry(2, BufferBindingType::ReadOnlyStorage, c),
            storage_entry(3, BufferBindingType::Storage, c),
        ];
        let p = std::rc::Rc::new(compute(g, "webmetal visibility merge", VISIBILITY_MERGE_WGSL, entries)?);
        *self.visibility.borrow_mut() = Some(p.clone());
        Ok(p)
    }

    /// The resolve bind group layouts (depth: binding 0; stencil: binding 1).
    pub fn resolve_layouts(&self, g: &impl Calls<A>) -> Result<(A::BindGroupLayout, A::BindGroupLayout)> {
        self.ensure_resolve(g)?;
        let r = self.resolve.borrow();
        let r = r.as_ref().expect("created");
        Ok((r.depth_layout.clone(), r.stencil_layout.clone()))
    }

    fn ensure_resolve(&self, g: &impl Calls<A>) -> Result<()> {
        if self.resolve.borrow().is_some() {
            return Ok(());
        }
        let label = "webmetal depth-stencil resolve";
        let mut md = ShaderModuleDescriptor::new(RESOLVE_WGSL.to_string());
        md.label = label.to_string();
        let module = g.call("GPUDevice.createShaderModule", label, |d| d.create_shader_module(&md));
        let texture = |binding: u32, sample_type: TextureSampleType| {
            let mut e = BindGroupLayoutEntry::new(binding, ShaderStage::FRAGMENT);
            let mut t = TextureBindingLayout::new();
            t.sample_type = sample_type;
            t.view_dimension = TextureViewDimension::D2;
            t.multisampled = true;
            e.texture = Some(t);
            e
        };
        let mut layouts = Vec::new();
        for (entry, what) in
            [(texture(0, TextureSampleType::Depth), "depth"), (texture(1, TextureSampleType::Uint), "stencil")]
        {
            let mut ld = BindGroupLayoutDescriptor::new(vec![entry]);
            ld.label = format!("{label} {what}");
            layouts.push(
                g.call("GPUDevice.createBindGroupLayout", label, |d| d.create_bind_group_layout(&ld))
                    .map_err(gpu_err("GPUDevice.createBindGroupLayout"))?,
            );
        }
        let stencil_layout = layouts.pop().expect("two layouts");
        let depth_layout = layouts.pop().expect("two layouts");
        let pipeline_layout = |l: &A::BindGroupLayout| {
            let pd = PipelineLayoutDescriptor::new(vec![Some(l)]);
            g.call("GPUDevice.createPipelineLayout", label, |d| d.create_pipeline_layout(&pd))
        };
        let depth_pipeline_layout = pipeline_layout(&depth_layout);
        // The stencil pipelines' layout skips binding 0: the stencil group is bound at index 0 with its own layout.
        let stencil_pipeline_layout = pipeline_layout(&stencil_layout);
        *self.resolve_module.borrow_mut() = Some(module);
        *self.resolve.borrow_mut() = Some(Resolve {
            depth_layout,
            stencil_layout,
            depth_pipeline_layout,
            stencil_pipeline_layout,
            pipelines: HashMap::new(),
        });
        Ok(())
    }

    /// The resolve pipeline of `kind` into a target of `format` (`has_depth`/`has_stencil`: the format's aspects).
    pub fn resolve_pipeline(
        &self,
        g: &impl Calls<A>,
        format: TextureFormat,
        has_depth: bool,
        has_stencil: bool,
        kind: ResolveKind,
    ) -> Result<A::RenderPipeline> {
        self.ensure_resolve(g)?;
        if let Some(p) = self.resolve.borrow().as_ref().expect("created").pipelines.get(&(format, kind)) {
            return Ok(p.clone());
        }
        let module = self.resolve_module.borrow().clone().expect("created");
        let mut r = self.resolve.borrow_mut();
        let r = r.as_mut().expect("created");
        let (layout, entry, constants) = match kind {
            ResolveKind::Depth(f) => (
                &r.depth_pipeline_layout,
                "fs_depth",
                vec![("resolve_filter".to_string(), Float64::new(f64::from(f)).expect("finite"))],
            ),
            ResolveKind::StencilBit(b) => (
                &r.stencil_pipeline_layout,
                "fs_stencil",
                vec![("bit".to_string(), Float64::new(f64::from(b)).expect("finite"))],
            ),
        };
        let mut vs = VertexState::new(&module);
        vs.entry_point = Some("vs".into());
        let mut d = RenderPipelineDescriptor::new(PipelineLayoutOrAuto::PipelineLayout(layout), vs);
        d.label = format!("webmetal resolve {kind:?} {}", format.as_str());
        let mut fs = FragmentState::new(&module, Vec::new());
        fs.entry_point = Some(entry.into());
        fs.constants = constants;
        d.fragment = Some(fs);
        let mut ds = DepthStencilState::new(format);
        if has_depth {
            ds.depth_write_enabled = Some(matches!(kind, ResolveKind::Depth(_)));
            ds.depth_compare = Some(GpuCompare::Always);
        }
        if has_stencil {
            let mut face = StencilFaceState::new();
            face.compare = GpuCompare::Always;
            if let ResolveKind::StencilBit(b) = kind {
                face.pass_op = GpuStencilOp::Replace;
                ds.stencil_write_mask = 1 << b;
            } else {
                ds.stencil_write_mask = 0;
            }
            ds.stencil_front = face.clone();
            ds.stencil_back = face;
            ds.stencil_read_mask = 0xff;
        }
        d.depth_stencil = Some(ds);
        let label = d.label.clone();
        let p = g
            .call("GPUDevice.createRenderPipeline", &label, |dev| dev.create_render_pipeline(&d))
            .map_err(gpu_err("GPUDevice.createRenderPipeline"))?;
        r.pipelines.insert((format, kind), p.clone());
        Ok(p)
    }
}

/// The `filter` constant of a depth resolve filter.
pub fn depth_filter_constant(f: DepthResolveFilter) -> u32 {
    match f {
        DepthResolveFilter::Sample0 => 0,
        DepthResolveFilter::Min => 1,
        DepthResolveFilter::Max => 2,
    }
}

/// A size as a `GPUSize64`.
pub fn size64(v: u64, op: &'static str) -> Result<Size64> {
    Size64::new(v).map_err(gpu_err(op))
}
