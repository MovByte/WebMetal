//! The WebGPU objects of compiled pipelines: shader modules, bind group layouts, pipeline layouts and pipelines, and
//! their cache.
//!
//! A compiled pipeline's WebGPU objects depend only on its WGSL, its entry points and the resource layout of its
//! bindings (not on where the encoder finds each resource), so pipelines and variants whose compilation produced the
//! same of those share one set of objects: [`GpuCache`] keys them by the SHA-256 of that description.
//!
//! Before any WebGPU call the compiled pipeline's host requirements are checked against the device ([`requirements`]):
//! a shortfall is the host's (status 3), never a WebGPU error.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use airwgsl::{BindingLayout, BindingResource, Compiled, HostRequirements};
use webmetal_wire::Failure;

use super::compile::KeyWriter;
use super::sha256::sha256;
use crate::exec::failure;
use crate::format::webgpu::{Feature, HostFeatures, WebGpuFormat};
use crate::gpu::idl::{
    self, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferBindingLayout, ComputePipelineDescriptor,
    PipelineLayoutDescriptor, PipelineLayoutOrAuto, ProgrammableStage, SamplerBindingLayout, ShaderModuleDescriptor,
    StorageTextureBindingLayout, TextureBindingLayout,
};
use crate::gpu::{Api, Device as _, Exception, ExceptionKind, GpuFuture, Size64};
use crate::mem::calls::Calls;

/// Check what a compiled pipeline needs against the device: every feature, every limit at least the value it needs,
/// every WGSL language extension.
pub fn requirements<A: Api>(
    call: &str,
    device: &A::Device,
    wgsl_language_features: &std::collections::BTreeSet<String>,
    req: &HostRequirements,
) -> Result<(), Failure> {
    let HostRequirements { features, limits, wgsl_language_features: extensions } = req;
    let have = device.features();
    if let Some(f) = features.iter().find(|f| !have.contains(*f)) {
        return Err(failure::host_unsupported(format!("{call}: the pipeline needs the WebGPU feature {f}")));
    }
    let device_limits = device.limits();
    for (name, need) in limits {
        match device_limits.get(name) {
            Some(v) if v >= *need => {}
            Some(v) => {
                return Err(failure::host_unsupported(format!(
                    "{call}: the pipeline needs the WebGPU limit {name} of at least {need}; this device has {v}"
                )));
            }
            None => {
                return Err(failure::host_unsupported(format!(
                    "{call}: the pipeline needs the WebGPU limit {name}, which this implementation does not report"
                )));
            }
        }
    }
    if let Some(x) = extensions.iter().find(|x| !wgsl_language_features.contains(*x)) {
        return Err(failure::host_unsupported(format!("{call}: the pipeline needs the WGSL language extension {x}")));
    }
    Ok(())
}

fn buffer_type(t: airwgsl::BufferBindingType) -> idl::BufferBindingType {
    match t {
        airwgsl::BufferBindingType::Uniform => idl::BufferBindingType::Uniform,
        airwgsl::BufferBindingType::Storage => idl::BufferBindingType::Storage,
        airwgsl::BufferBindingType::ReadOnlyStorage => idl::BufferBindingType::ReadOnlyStorage,
    }
}

/// The WebGPU sampler binding type of a compiler sampler binding type.
pub fn sampler_type(t: airwgsl::SamplerBindingType) -> idl::SamplerBindingType {
    match t {
        airwgsl::SamplerBindingType::Filtering => idl::SamplerBindingType::Filtering,
        airwgsl::SamplerBindingType::NonFiltering => idl::SamplerBindingType::NonFiltering,
        airwgsl::SamplerBindingType::Comparison => idl::SamplerBindingType::Comparison,
    }
}

/// The WebGPU sample type of a compiler sample type.
pub fn sample_type(t: airwgsl::TextureSampleType) -> idl::TextureSampleType {
    match t {
        airwgsl::TextureSampleType::Float => idl::TextureSampleType::Float,
        airwgsl::TextureSampleType::UnfilterableFloat => idl::TextureSampleType::UnfilterableFloat,
        airwgsl::TextureSampleType::Depth => idl::TextureSampleType::Depth,
        airwgsl::TextureSampleType::Sint => idl::TextureSampleType::Sint,
        airwgsl::TextureSampleType::Uint => idl::TextureSampleType::Uint,
    }
}

/// The WebGPU view dimension of a compiler view dimension.
pub fn view_dimension(d: airwgsl::TextureViewDimension) -> idl::TextureViewDimension {
    match d {
        airwgsl::TextureViewDimension::D1 => idl::TextureViewDimension::D1,
        airwgsl::TextureViewDimension::D2 => idl::TextureViewDimension::D2,
        airwgsl::TextureViewDimension::D2Array => idl::TextureViewDimension::D2Array,
        airwgsl::TextureViewDimension::Cube => idl::TextureViewDimension::Cube,
        airwgsl::TextureViewDimension::CubeArray => idl::TextureViewDimension::CubeArray,
        airwgsl::TextureViewDimension::D3 => idl::TextureViewDimension::D3,
    }
}

/// The WebGPU storage access of a compiler storage access.
pub fn storage_access(a: airwgsl::StorageTextureAccess) -> idl::StorageTextureAccess {
    match a {
        airwgsl::StorageTextureAccess::WriteOnly => idl::StorageTextureAccess::WriteOnly,
        airwgsl::StorageTextureAccess::ReadOnly => idl::StorageTextureAccess::ReadOnly,
        airwgsl::StorageTextureAccess::ReadWrite => idl::StorageTextureAccess::ReadWrite,
    }
}

/// The `GPUTextureFormat` named by `name`.
pub fn texture_format(name: &str) -> Result<idl::TextureFormat, Failure> {
    idl::TextureFormat::ALL.into_iter().find(|f| f.as_str() == name).ok_or_else(|| {
        failure::unimplemented(format!("the compiler named the texture format {name:?}, not a GPUTextureFormat"))
    })
}

/// The `GPUBindGroupLayoutEntry` of one binding.
pub fn layout_entry(e: &airwgsl::BindingEntry) -> Result<BindGroupLayoutEntry, Failure> {
    let mut out = BindGroupLayoutEntry::new(e.binding, e.visibility.0);
    match &e.resource {
        BindingResource::Buffer { ty, has_dynamic_offset, min_binding_size } => {
            let mut b = BufferBindingLayout::new();
            b.r#type = buffer_type(*ty);
            b.has_dynamic_offset = *has_dynamic_offset;
            b.min_binding_size = Size64::new(*min_binding_size).map_err(|_| {
                failure::unimplemented(format!("binding {}: minBindingSize {min_binding_size}", e.binding))
            })?;
            out.buffer = Some(b);
        }
        BindingResource::Sampler { ty } => {
            let mut s = SamplerBindingLayout::new();
            s.r#type = sampler_type(*ty);
            out.sampler = Some(s);
        }
        BindingResource::Texture { sample_type: st, view_dimension: vd, multisampled } => {
            let mut t = TextureBindingLayout::new();
            t.sample_type = sample_type(*st);
            t.view_dimension = view_dimension(*vd);
            t.multisampled = *multisampled;
            out.texture = Some(t);
        }
        BindingResource::StorageTexture { access, format, view_dimension: vd } => {
            let mut t = StorageTextureBindingLayout::new(texture_format(format)?);
            t.access = storage_access(*access);
            t.view_dimension = view_dimension(*vd);
            out.storage_texture = Some(t);
        }
    }
    Ok(out)
}

/// Check a compiled binding layout against WebGPU's rules for `createBindGroupLayout` (WebGPU 8.1.1, "Bind Group
/// Layout Creation") before any WebGPU call, so that a layout WebGPU would reject is reported here and never raised as
/// a WebGPU error: an entry the rules forbid is a compiler defect (status 2), a layout over the device's binding limits
/// is the host's shortfall (status 3).
pub fn check_layout<A: Api>(call: &str, device: &A::Device, layout: &BindingLayout) -> Result<(), Failure> {
    let limits = device.limits();
    let names = device.features();
    let features = HostFeatures::new(Feature::ALL.into_iter().filter(|f| names.contains(f.name())));
    let defect = |group: u32, binding: u32, rule: &str| {
        failure::unimplemented(format!(
            "{call}: the compiler's bind group {group}, binding {binding}, breaks WebGPU's createBindGroupLayout rule \
             that {rule} (WebGPU 8.1.1)"
        ))
    };
    // The pipeline layout's group count (createPipelineLayout: at most maxBindGroups layouts).
    let groups = layout.groups.iter().map(|g| u64::from(g.group) + 1).max().unwrap_or(0);
    if groups > 0 && limits.get("maxBindGroups").is_none_or(|l| groups > l) {
        return Err(failure::host_unsupported(format!(
            "{call}: the pipeline uses {groups} bind groups; this device's maxBindGroups is {:?}",
            limits.get("maxBindGroups")
        )));
    }
    // Per stage (vertex, fragment, compute): sampled textures, samplers, storage buffers, storage textures, uniform
    // buffers, over every group: createPipelineLayout checks all the layouts' entries together against the binding slot
    // limits, which bounds each layout's own entries as well.
    let mut counts = [[0u64; 5]; 3];
    for g in &layout.groups {
        let mut seen = std::collections::BTreeSet::new();
        for e in &g.entries {
            if !seen.insert(e.binding) {
                return Err(defect(g.group, e.binding, "each binding of a layout is unique"));
            }
            if let Some(max) = limits.get("maxBindingsPerBindGroup")
                && u64::from(e.binding) >= max
            {
                return Err(failure::host_unsupported(format!(
                    "{call}: binding {} of bind group {} is not below this device's maxBindingsPerBindGroup {max}",
                    e.binding, g.group
                )));
            }
            if e.visibility.0 & !0x7 != 0 {
                return Err(defect(g.group, e.binding, "visibility contains only GPUShaderStage bits"));
            }
            let vertex = e.visibility.contains(airwgsl::ShaderStages::VERTEX);
            let category = match &e.resource {
                BindingResource::Buffer { ty, .. } => {
                    if vertex && *ty == airwgsl::BufferBindingType::Storage {
                        return Err(defect(
                            g.group,
                            e.binding,
                            "a buffer the vertex stage sees is uniform or read-only",
                        ));
                    }
                    if *ty == airwgsl::BufferBindingType::Uniform { 4 } else { 2 }
                }
                BindingResource::Sampler { .. } => 1,
                BindingResource::Texture { sample_type, view_dimension, multisampled } => {
                    if *multisampled
                        && (*view_dimension != airwgsl::TextureViewDimension::D2
                            || *sample_type == airwgsl::TextureSampleType::Float)
                    {
                        return Err(defect(
                            g.group,
                            e.binding,
                            "a multisampled texture has view dimension \"2d\" and a sample type other than \"float\"",
                        ));
                    }
                    0
                }
                BindingResource::StorageTexture { access, format, view_dimension } => {
                    if vertex && *access != airwgsl::StorageTextureAccess::ReadOnly {
                        return Err(defect(g.group, e.binding, "a storage texture the vertex stage sees is read-only"));
                    }
                    if matches!(
                        view_dimension,
                        airwgsl::TextureViewDimension::Cube | airwgsl::TextureViewDimension::CubeArray
                    ) {
                        return Err(defect(g.group, e.binding, "a storage texture's view dimension is not a cube"));
                    }
                    let f = WebGpuFormat::from_name(format).ok_or_else(|| {
                        defect(g.group, e.binding, "a storage texture's format is a GPUTextureFormat")
                    })?;
                    let caps = f.caps();
                    let cond = match access {
                        airwgsl::StorageTextureAccess::WriteOnly => caps.storage_wo,
                        airwgsl::StorageTextureAccess::ReadOnly => caps.storage_ro,
                        airwgsl::StorageTextureAccess::ReadWrite => caps.storage_rw,
                    };
                    if !features.allows(cond) {
                        return Err(failure::host_unsupported(format!(
                            "{call}: bind group {}, binding {}: this device cannot bind {format} for storage access \
                             {access:?} (WebGPU 26.1.1)",
                            g.group, e.binding
                        )));
                    }
                    3
                }
            };
            for (i, stage) in
                [airwgsl::ShaderStages::VERTEX, airwgsl::ShaderStages::FRAGMENT, airwgsl::ShaderStages::COMPUTE]
                    .into_iter()
                    .enumerate()
            {
                if e.visibility.contains(stage) {
                    counts[i][category] += 1;
                }
            }
        }
    }
    // The binding slot limits (WebGPU 8.1.1, "exceeds the binding slot limits"): storage buffers and storage
    // textures of the vertex and fragment stages count against their stage's own limit, the compute stage's
    // against the per-stage limit. An implementation that does not report a stage's own limit (a webgpu.h without
    // the field) predates it and counts those bindings against the per-stage limit.
    let slot = |stage: usize, category: usize| -> &'static str {
        match (stage, category) {
            (_, 0) => "maxSampledTexturesPerShaderStage",
            (_, 1) => "maxSamplersPerShaderStage",
            (0, 2) => "maxStorageBuffersInVertexStage",
            (1, 2) => "maxStorageBuffersInFragmentStage",
            (_, 2) => "maxStorageBuffersPerShaderStage",
            (0, 3) => "maxStorageTexturesInVertexStage",
            (1, 3) => "maxStorageTexturesInFragmentStage",
            (_, 3) => "maxStorageTexturesPerShaderStage",
            _ => "maxUniformBuffersPerShaderStage",
        }
    };
    for (i, stage) in ["vertex", "fragment", "compute"].iter().enumerate() {
        for (k, &used) in counts[i].iter().enumerate() {
            if used == 0 {
                continue;
            }
            let mut name = slot(i, k);
            let mut limit = limits.get(name);
            if limit.is_none() && matches!(k, 2 | 3) {
                name = slot(2, k);
                limit = limits.get(name);
            }
            match limit {
                Some(l) if used <= l => {}
                Some(l) => {
                    return Err(failure::host_unsupported(format!(
                        "{call}: the pipeline uses {used} {name} slots in the {stage} stage; this device allows {l}"
                    )));
                }
                None => {
                    return Err(failure::host_unsupported(format!(
                        "{call}: this implementation does not report {name}, which the pipeline uses"
                    )));
                }
            }
        }
    }
    for (ty, name) in [
        (airwgsl::BufferBindingType::Uniform, "maxDynamicUniformBuffersPerPipelineLayout"),
        (airwgsl::BufferBindingType::Storage, "maxDynamicStorageBuffersPerPipelineLayout"),
    ] {
        let used = layout
            .groups
            .iter()
            .flat_map(|g| &g.entries)
            .filter(
                |e| matches!(&e.resource, BindingResource::Buffer { ty: t, has_dynamic_offset: true, .. } if *t == ty),
            )
            .count() as u64;
        if used > 0 && limits.get(name).is_none_or(|l| used > l) {
            return Err(failure::host_unsupported(format!(
                "{call}: the pipeline uses {used} {name} slots; this device allows {:?}",
                limits.get(name)
            )));
        }
    }
    Ok(())
}

/// The bind group layouts of a pipeline (each with its `@group` index) and its pipeline layout.
pub struct Layouts<A: Api> {
    pub groups: Vec<(u32, A::BindGroupLayout)>,
    pub pipeline: A::PipelineLayout,
}

/// Create the bind group layouts and the pipeline layout of `layout`. A group index the layout skips gets an empty
/// layout, which `createPipelineLayout` treats as null (it keeps only layouts with entries, WebGPU 8.3.1); wgpu-native
/// does not take the null entries `GPUPipelineLayoutDescriptor` allows (its createPipelineLayout aborts on one). The
/// encoder binds an empty bind group there.
pub fn create_layouts<A: Api>(g: &impl Calls<A>, label: &str, layout: &BindingLayout) -> Result<Layouts<A>, Failure> {
    let mut groups: Vec<(u32, A::BindGroupLayout)> = Vec::new();
    for group in &layout.groups {
        if groups.iter().any(|(i, _)| *i == group.group) {
            return Err(failure::unimplemented(format!(
                "{label}: the compiler described bind group {} twice",
                group.group
            )));
        }
        let entries = group.entries.iter().map(layout_entry).collect::<Result<Vec<_>, _>>()?;
        let mut d = BindGroupLayoutDescriptor::new(entries);
        d.label = format!("{label} group {}", group.group);
        let bgl = g
            .call("GPUDevice.createBindGroupLayout", &d.label, |dev| dev.create_bind_group_layout(&d))
            .map_err(|e| gpu_failure("GPUDevice.createBindGroupLayout", label, &e))?;
        groups.push((group.group, bgl));
    }
    let count = groups.iter().map(|(i, _)| *i + 1).max().unwrap_or(0);
    for index in 0..count {
        if groups.iter().any(|(i, _)| *i == index) {
            continue;
        }
        let mut d = BindGroupLayoutDescriptor::new(Vec::new());
        d.label = format!("{label} group {index} (empty)");
        let bgl = g
            .call("GPUDevice.createBindGroupLayout", &d.label, |dev| dev.create_bind_group_layout(&d))
            .map_err(|e| gpu_failure("GPUDevice.createBindGroupLayout", label, &e))?;
        groups.push((index, bgl));
    }
    groups.sort_by_key(|(i, _)| *i);
    let slots: Vec<Option<&A::BindGroupLayout>> = groups.iter().map(|(_, l)| Some(l)).collect();
    let mut pd = PipelineLayoutDescriptor::new(slots);
    pd.label = label.to_string();
    let pipeline = g.call("GPUDevice.createPipelineLayout", label, |dev| dev.create_pipeline_layout(&pd));
    Ok(Layouts { groups, pipeline })
}

/// A WebGPU exception or promise rejection from creating a pipeline's objects. The executor checks what it gives
/// WebGPU first, so this is an executor or compiler defect, reported with status 2.
pub fn gpu_failure(op: &str, label: &str, e: &Exception) -> Failure {
    let what = match &e.kind {
        ExceptionKind::PipelineError(r) => format!("GPUPipelineError ({})", r.as_str()),
        k => format!("{k:?}"),
    };
    failure::unimplemented(format!(
        "{op} for {label} failed with {what}: {}; the pipeline the compiler produced is not valid on this device",
        e.message
    ))
}

/// The WebGPU objects of a compiled compute pipeline.
pub struct ComputeObjects<A: Api> {
    pub module: A::ShaderModule,
    pub layouts: Layouts<A>,
    pub pipeline: A::ComputePipeline,
}

/// The objects of a compute pipeline once its `createComputePipelineAsync` promise resolves.
pub type PendingCompute<A> = GpuFuture<Result<Rc<ComputeObjects<A>>, Failure>>;

/// Counters of the cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuStats {
    pub hits: u64,
    pub misses: u64,
}

/// WebGPU objects by the SHA-256 of the WGSL, entry points and resource layout they were made from.
pub struct GpuCache<A: Api> {
    compute: RefCell<HashMap<[u8; 32], Rc<ComputeObjects<A>>>>,
    stats: Cell<GpuStats>,
}

impl<A: Api> Default for GpuCache<A> {
    fn default() -> Self {
        GpuCache { compute: RefCell::new(HashMap::new()), stats: Cell::new(GpuStats::default()) }
    }
}

/// The cache key of a compiled pipeline's WebGPU objects.
pub fn objects_key(c: &Compiled) -> [u8; 32] {
    let mut w = KeyWriter::default();
    w.seq(&c.wgsl, |w, s| {
        let airwgsl::Stage { stage, entry_point, source } = s;
        w.tag(*stage as u8);
        w.str(entry_point);
        w.str(source);
    });
    w.seq(&c.layout.groups, |w, g| {
        w.u32(g.group);
        w.seq(&g.entries, |w, e| {
            w.u32(e.binding);
            w.u32(e.visibility.0);
            match &e.resource {
                BindingResource::Buffer { ty, has_dynamic_offset, min_binding_size } => {
                    w.tag(0);
                    w.tag(*ty as u8);
                    w.bool(*has_dynamic_offset);
                    w.u64(*min_binding_size);
                }
                BindingResource::Sampler { ty } => {
                    w.tag(1);
                    w.tag(*ty as u8);
                }
                BindingResource::Texture { sample_type, view_dimension, multisampled } => {
                    w.tag(2);
                    w.tag(*sample_type as u8);
                    w.tag(*view_dimension as u8);
                    w.bool(*multisampled);
                }
                BindingResource::StorageTexture { access, format, view_dimension } => {
                    w.tag(3);
                    w.tag(*access as u8);
                    w.str(format);
                    w.tag(*view_dimension as u8);
                }
            }
        });
    });
    sha256(&w.0)
}

impl<A: Api> GpuCache<A> {
    pub fn stats(&self) -> GpuStats {
        self.stats.get()
    }

    fn count(&self, hit: bool) {
        let mut s = self.stats.get();
        if hit {
            s.hits += 1;
        } else {
            s.misses += 1;
        }
        self.stats.set(s);
    }

    /// The cached objects of `c`.
    pub fn compute(&self, c: &Compiled) -> Option<Rc<ComputeObjects<A>>> {
        let hit = self.compute.borrow().get(&objects_key(c)).cloned();
        if hit.is_some() {
            self.count(true);
        }
        hit
    }

    /// The objects of the compute pipeline `c`, created now with `createComputePipeline` unless cached. Errors the
    /// creation raises on the device timeline reach the caller's error scopes.
    pub fn compute_now(&self, g: &impl Calls<A>, label: &str, c: &Compiled) -> Result<Rc<ComputeObjects<A>>, Failure> {
        if let Some(o) = self.compute(c) {
            return Ok(o);
        }
        self.count(false);
        let (module, layouts, stage) = compute_parts(g, label, c)?;
        let mut ps = ProgrammableStage::new(&module);
        ps.entry_point = Some(stage.entry_point.clone());
        let mut d = ComputePipelineDescriptor::new(PipelineLayoutOrAuto::PipelineLayout(&layouts.pipeline), ps);
        d.label = label.to_string();
        let pipeline = g.call("GPUDevice.createComputePipeline", label, |dev| dev.create_compute_pipeline(&d));
        let o = Rc::new(ComputeObjects { module, layouts, pipeline });
        self.compute.borrow_mut().insert(objects_key(c), o.clone());
        Ok(o)
    }

    /// The objects of the compute pipeline `c`, the pipeline created with `createComputePipelineAsync` unless cached:
    /// the module and layouts now, the pipeline when the returned future resolves. A rejected promise is a failure.
    pub fn compute_async(
        self: &Rc<Self>,
        g: &impl Calls<A>,
        label: &str,
        c: &Compiled,
    ) -> Result<PendingCompute<A>, Failure> {
        if let Some(o) = self.compute(c) {
            return Ok(Box::pin(async move { Ok(o) }));
        }
        self.count(false);
        let (module, layouts, stage) = compute_parts(g, label, c)?;
        let mut ps = ProgrammableStage::new(&module);
        ps.entry_point = Some(stage.entry_point.clone());
        let mut d = ComputePipelineDescriptor::new(PipelineLayoutOrAuto::PipelineLayout(&layouts.pipeline), ps);
        d.label = label.to_string();
        let pending =
            g.call("GPUDevice.createComputePipelineAsync", label, |dev| dev.create_compute_pipeline_async(&d));
        let key = objects_key(c);
        let cache = self.clone();
        let label = label.to_string();
        Ok(Box::pin(async move {
            let pipeline =
                pending.await.map_err(|e| gpu_failure("GPUDevice.createComputePipelineAsync", &label, &e))?;
            let o = Rc::new(ComputeObjects { module, layouts, pipeline });
            cache.compute.borrow_mut().entry(key).or_insert_with(|| o.clone());
            Ok(o)
        }))
    }
}

fn compute_parts<'c, A: Api>(
    g: &impl Calls<A>,
    label: &str,
    c: &'c Compiled,
) -> Result<(A::ShaderModule, Layouts<A>, &'c airwgsl::Stage), Failure> {
    let [stage] = &c.wgsl[..] else {
        return Err(failure::unimplemented(format!(
            "{label}: a compute pipeline compiled to {} WGSL modules",
            c.wgsl.len()
        )));
    };
    if stage.stage != airwgsl::ShaderStage::Compute {
        return Err(failure::unimplemented(format!(
            "{label}: a compute pipeline compiled to a {:?} stage",
            stage.stage
        )));
    }
    check_layout::<A>(label, g.device(), &c.layout)?;
    let mut md = ShaderModuleDescriptor::new(stage.source.clone());
    md.label = label.to_string();
    let module = g.call("GPUDevice.createShaderModule", label, |dev| dev.create_shader_module(&md));
    let layouts = create_layouts(g, label, &c.layout)?;
    Ok((module, layouts, stage))
}
