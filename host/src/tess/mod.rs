//! Tessellated draws (plan 5.7, package S6e): the fixed-function tessellator as compute work before a render pass.
//!
//! WebGPU has no tessellation stage. For each patch draw of a render encoder, the tessellator's kernels
//! (`airwgsl::tess::kernel`, the same computation as AirWgsl's CPU tessellator, which equals real Metal's output on an
//! Apple M4 over the V1 `tessellation.*` sweeps) run on the draw's factor buffer before the encoder's first WebGPU pass:
//! they write the draw's points (a record per point: u and v in 16.16 fixed point, the patch identifier, the instance
//! identifier) and the indices of its triangles in Metal's order, instance by instance and patch by patch, with the
//! `drawIndexedIndirect` arguments that draw them. The draw is then that indirect draw with the pipeline's
//! post-tessellation vertex function reading the point records.
//!
//! The factors are read on the GPU because a compute pass of the command buffer usually writes them; a render pass
//! cannot write them for a later draw of the same pass (no barrier orders fragment-stage stores before vertex-stage
//! reads on Apple-family GPUs, reference case `render.error.memory-barrier-after-fragment`; vertex-stage stores are
//! refused), so running every patch draw's kernels before the pass reads the factors the draws read.
//!
//! The M4 emits an inside blade wider than 36 grid points in tiles whose sizes depend on what the draw emitted before
//! (`airwgsl::tess::hw::Counts::batched`), an order the kernels do not reproduce. The count kernel flags such patches;
//! after the command buffer runs, a flagged draw fails the command buffer as not implemented ([`Generated::check`]),
//! naming the draw.

use std::cell::RefCell;
use std::rc::Rc;

use airwgsl::tess::{PartitionMode, PatchType, Settings, kernel};
use webmetal_wire::{Failure, Handle, RenderDrawIndexedPatches, RenderDrawPatches};

use crate::exec::{Table, failure};
use crate::gpu::idl::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
    BufferBinding, BufferBindingLayout, BufferBindingType, BufferDescriptor, BufferUsage, ComputePassDescriptor,
    ComputePipelineDescriptor, MapMode, PipelineLayoutDescriptor, PipelineLayoutOrAuto, ProgrammableStage,
    ShaderModuleDescriptor, ShaderStage,
};
use crate::gpu::{
    Api, BindingCommands as _, Buffer as _, CommandEncoder as _, ComputePassEncoder as _, Device as _, GpuFuture,
    MappedRange as _, Queue as _, Size64,
};
use crate::mem::buffer::Buffer;
use crate::mem::calls::Calls;
use crate::render::RenderError;
use crate::render::pipeline::PassPlan;

type Result<T> = std::result::Result<T, RenderError>;

const LABEL: &str = "webmetal tessellator";

/// The kernels' programs, created on first use.
pub struct Kernels<A: Api> {
    layout: A::BindGroupLayout,
    count: A::ComputePipeline,
    scan: A::ComputePipeline,
    generate: A::ComputePipeline,
}

/// The tessellator's programs of one device.
pub struct Tessellator<A: Api> {
    kernels: RefCell<Option<Rc<Kernels<A>>>>,
}

impl<A: Api> Default for Tessellator<A> {
    fn default() -> Self {
        Tessellator { kernels: RefCell::new(None) }
    }
}

fn size64(v: u64, op: &str) -> Result<Size64> {
    Size64::new(v).map_err(|_| RenderError::unimplemented(format!("{op}: {v} bytes beyond WebGPU's sizes")))
}

/// The A10X's `maxTessellationFactor` (device profile `ipados-17.7.11-21H461-a10x.toml`).
pub const A10X_MAX_FACTOR: u64 = 16;

/// The render path's pass plan for the compiler's `PassPlan::Tessellation`, from the pipeline's tessellation state:
/// the checks Metal.framework makes on it (measured on Apple M4, macOS 27.0.1, with and without the validation layer)
/// and the tessellator's settings.
pub fn pass_plan(call: &str, t: Option<&airwgsl::Tessellation>, quad: bool) -> std::result::Result<PassPlan, Failure> {
    use crate::pipeline::errors::{field, framework_asserts};
    let t = t.ok_or_else(|| {
        failure::invalid(format!("{call}: a tessellation pass plan for a pipeline without tessellation state"))
    })?;
    let partition = PartitionMode::from_metal(t.partition_mode).ok_or_else(|| {
        field(call, "tessellationPartitionMode", t.partition_mode, "not an MTLTessellationPartitionMode")
    })?;
    let checks = [
        (t.factor_format == 0, "tessellationFactorFormat", t.factor_format, "not MTLTessellationFactorFormatHalf"),
        (
            t.factor_step_function <= 3,
            "tessellationFactorStepFunction",
            t.factor_step_function,
            "not an MTLTessellationFactorStepFunction",
        ),
        (t.output_winding_order <= 1, "tessellationOutputWindingOrder", t.output_winding_order, "not an MTLWinding"),
        (
            t.control_point_index_type <= 2,
            "tessellationControlPointIndexType",
            t.control_point_index_type,
            "not an MTLTessellationControlPointIndexType",
        ),
    ];
    if let Some((_, name, value, why)) = checks.into_iter().find(|c| !c.0) {
        return Err(field(call, name, value, why));
    }
    let max = t.max_tessellation_factor;
    if !(1..=64).contains(&max) {
        return Err(failure::unimplemented(format!(
            "{call}: maxTessellationFactor {max}: Metal.framework asserts that it is at least 1 and at most 64 \
             (validateMaxTessellationFactor, measured on Apple M4, macOS 27.0.1), so no driver result is defined"
        )));
    }
    if partition == PartitionMode::Pow2 && !max.is_power_of_two() {
        return Err(framework_asserts(
            call,
            format!("maxTessellationFactor {max} is not a power of 2 with MTLTessellationPartitionModePow2"),
        ));
    }
    if matches!(partition, PartitionMode::FractionalOdd | PartitionMode::FractionalEven) && max % 2 != 0 {
        return Err(framework_asserts(
            call,
            format!("maxTessellationFactor {max} is odd with a fractional partition mode"),
        ));
    }
    if max > A10X_MAX_FACTOR {
        return Err(failure::unimplemented(format!(
            "{call}: maxTessellationFactor {max} is above the A10X's maxTessellationFactor of {A10X_MAX_FACTOR}; what \
             its Metal.framework and driver do with it is not measured"
        )));
    }
    Ok(PassPlan::Tessellation {
        patch: if quad { PatchType::Quad } else { PatchType::Triangle },
        settings: Settings {
            partition,
            max_factor: max as u32,
            scale_enabled: t.factor_scale_enabled,
            clockwise: t.output_winding_order == 0,
        },
        control_point_index_type: t.control_point_index_type,
        factor_step_function: t.factor_step_function,
    })
}

/// The call patch draws are reported under.
pub const PATCHES: &str = "-[MTLRenderCommandEncoder drawPatches:...]";

/// The arguments of a `drawPatches` or `drawIndexedPatches` record.
#[derive(Clone, Copy, Debug)]
pub struct PatchDraw {
    pub control_points: u64,
    pub start: u64,
    pub count: u64,
    /// The patch index buffer and its offset.
    pub patch_indices: Option<(Handle, u64)>,
    /// `drawIndexedPatches`: the control-point index buffer (nil allowed on the wire) and its offset.
    pub control_point_indices: Option<(Option<Handle>, u64)>,
    pub instances: u64,
    pub base_instance: u64,
}

impl PatchDraw {
    pub fn of(r: &RenderDrawPatches) -> PatchDraw {
        PatchDraw {
            control_points: r.number_of_patch_control_points,
            start: r.patch_start,
            count: r.patch_count,
            patch_indices: r.patch_index_buffer.map(|h| (h, r.patch_index_buffer_offset)),
            control_point_indices: None,
            instances: r.instance_count,
            base_instance: r.base_instance,
        }
    }

    pub fn of_indexed(r: &RenderDrawIndexedPatches) -> PatchDraw {
        PatchDraw {
            control_points: r.number_of_patch_control_points,
            start: r.patch_start,
            count: r.patch_count,
            patch_indices: r.patch_index_buffer.map(|h| (h, r.patch_index_buffer_offset)),
            control_point_indices: Some((r.control_point_index_buffer, r.control_point_index_buffer_offset)),
            instances: r.instance_count,
            base_instance: r.base_instance,
        }
    }
}

/// The tessellation state of a render encoder (`setTessellationFactorBuffer:offset:instanceStride:`,
/// `setTessellationFactorScale:`), reset for each parallel sub-encoder.
pub struct EncoderState<A: Api> {
    /// The factor buffer, its offset and its instance stride.
    pub factors: Option<(Rc<Buffer<A>>, u64, u64)>,
    pub scale: Option<f32>,
}

impl<A: Api> Default for EncoderState<A> {
    fn default() -> Self {
        EncoderState { factors: None, scale: None }
    }
}

fn buffer<A: Api>(objects: &Table, handle: Handle) -> Result<Rc<Buffer<A>>> {
    objects.get(handle).and_then(|o| o.payload.clone()).and_then(|p| p.downcast::<Buffer<A>>().ok()).ok_or_else(|| {
        RenderError::unimplemented(format!(
            "{PATCHES}: buffer {:#x} has no storage the render path knows",
            handle.get()
        ))
    })
}

/// An indirect patch draw: the A10X has none.
pub fn indirect_patch_draw() -> RenderError {
    RenderError::undefined(
        "-[MTLRenderCommandEncoder drawPatches:...indirectBuffer:...]",
        "the A10X reports supportsIndirectTessellation = false (device profile); its driver has no indirect patch draw",
    )
}

/// A primitive draw with the pipeline of pass plan `plan`.
pub fn primitive_draw(plan: Option<&PassPlan>) -> Result<()> {
    if matches!(plan, Some(PassPlan::Tessellation { .. })) {
        return Err(RenderError::undefined(
            "-[MTLRenderCommandEncoder draw...]",
            "a primitive draw with a pipeline whose vertex function is a post-tessellation function (the validation \
             layer asserts)",
        ));
    }
    Ok(())
}

/// The job of patch draw `d`, the encoder's `ordinal`th draw, with the pipeline of pass plan `plan` and the encoder's
/// tessellation state, checked as the validation layer checks them (measured on Apple M4, macOS 27.0.1); None for a
/// draw of no patches or no instances.
pub fn job<A: Api>(
    objects: &Table,
    plan: Option<&PassPlan>,
    state: &EncoderState<A>,
    d: &PatchDraw,
    ordinal: usize,
) -> Result<Option<Job<A>>> {
    let Some(&PassPlan::Tessellation { patch, settings, control_point_index_type, factor_step_function }) = plan else {
        return Err(RenderError::undefined(
            PATCHES,
            "a patch draw with a pipeline whose vertex function is not a post-tessellation function (the validation \
             layer asserts)",
        ));
    };
    // A pipeline with a control-point index type draws with drawIndexedPatches only, one without with drawPatches
    // only (the validation layer asserts "tessellationControlPointIndexType of MTLTessellationControlPointIndexType
    // UInt16 not valid for drawPatches").
    if (control_point_index_type != 0) != d.control_point_indices.is_some() {
        return Err(RenderError::undefined(
            PATCHES,
            format!(
                "tessellationControlPointIndexType {control_point_index_type} with a draw that is not its kind of \
                 patch draw"
            ),
        ));
    }
    // "controlPointIndexBufferOffset(1) must be a multiple of 2 bytes", "controlPointIndexBufferOffset(220) +
    // (indexCount(8) * 2) must be <= [controlPointIndexBuffer length](128)", the index count being (patchStart +
    // patchCount) * numberOfPatchControlPoints whether or not a patch index buffer is set.
    if let Some((h, o)) = d.control_point_indices {
        let size = if control_point_index_type == 1 { 2 } else { 4 };
        let b = buffer::<A>(
            objects,
            h.ok_or_else(|| RenderError::undefined(PATCHES, "a nil control-point index buffer"))?,
        )?;
        let end = o + (d.start + d.count) * d.control_points * size;
        if !o.is_multiple_of(size) || end > b.length {
            return Err(RenderError::undefined(
                PATCHES,
                format!("control-point indices at offset {o} up to byte {end} of a {}-byte buffer", b.length),
            ));
        }
    }
    let Some((factors, factor_offset, instance_stride)) = state.factors.clone() else {
        return Err(RenderError::undefined(
            PATCHES,
            "no tessellation factor buffer is set (the validation layer asserts)",
        ));
    };
    // An encoder that never sets a scale culls every patch of a scale-enabled pipeline, as a scale of 0 does (with and
    // without the validation layer; tessellation.factors scale-unset).
    let scale = match (settings.scale_enabled, state.scale) {
        (false, _) => 1.0,
        (true, v) => v.unwrap_or(0.0),
    };
    let patch_indices = match d.patch_indices {
        Some((h, o)) => Some((buffer::<A>(objects, h)?, o)),
        None => None,
    };
    if d.count == 0 || d.instances == 0 {
        return Ok(None);
    }
    Ok(Some(Job {
        settings,
        patch,
        scale,
        factors,
        factor_offset,
        instance_stride,
        step: factor_step_function,
        patch_start: d.start,
        patch_count: d.count,
        instance_count: d.instances,
        base_instance: d.base_instance,
        patch_indices,
        what: format!("{PATCHES} (draw {ordinal} of the encoder)"),
    }))
}

/// The binding of a `drawIndexedPatches` draw's control-point index buffer: from an aligned base at or before the
/// draw's offset to the buffer's end.
pub struct IndexWindow<A: Api> {
    pub buffer: Rc<Buffer<A>>,
    pub base: u64,
    pub size: u64,
    /// Bytes from `base` to the draw's `controlPointIndexBufferOffset`.
    pub residual: u64,
}

/// The control-point index binding of patch draw `d` (None for `drawPatches`).
pub fn control_point_indices<A: Api>(objects: &Table, d: &PatchDraw, limits: Limits) -> Result<Option<IndexWindow<A>>> {
    let Some((Some(h), offset)) = d.control_point_indices else { return Ok(None) };
    let b = buffer::<A>(objects, h)?;
    let (base, size) = window(&b, offset, limits, PATCHES)?;
    Ok(Some(IndexWindow { buffer: b, base, size, residual: offset - base }))
}

impl<A: Api> Tessellator<A> {
    fn kernels(&self, g: &impl Calls<A>) -> Result<Rc<Kernels<A>>> {
        if let Some(k) = self.kernels.borrow().as_ref() {
            return Ok(k.clone());
        }
        let c = ShaderStage::COMPUTE;
        let entry = |binding: kernel::Binding, ty| {
            let mut e = BindGroupLayoutEntry::new(binding as u32, c);
            let mut b = BufferBindingLayout::new();
            b.r#type = ty;
            e.buffer = Some(b);
            e
        };
        let entries = vec![
            entry(kernel::Binding::Params, BufferBindingType::Uniform),
            entry(kernel::Binding::Factors, BufferBindingType::ReadOnlyStorage),
            entry(kernel::Binding::PatchIndices, BufferBindingType::ReadOnlyStorage),
            entry(kernel::Binding::Slots, BufferBindingType::Storage),
            entry(kernel::Binding::Vertices, BufferBindingType::Storage),
            entry(kernel::Binding::Indices, BufferBindingType::Storage),
            entry(kernel::Binding::Draw, BufferBindingType::Storage),
        ];
        let mut md = ShaderModuleDescriptor::new(kernel::WGSL.to_string());
        md.label = LABEL.to_string();
        let module = g.call("GPUDevice.createShaderModule", LABEL, |d| d.create_shader_module(&md));
        let mut ld = BindGroupLayoutDescriptor::new(entries);
        ld.label = LABEL.to_string();
        let layout = g
            .call("GPUDevice.createBindGroupLayout", LABEL, |d| d.create_bind_group_layout(&ld))
            .map_err(|e| RenderError::gpu("GPUDevice.createBindGroupLayout", e))?;
        let mut pd = PipelineLayoutDescriptor::new(vec![Some(&layout)]);
        pd.label = LABEL.to_string();
        let pl = g.call("GPUDevice.createPipelineLayout", LABEL, |d| d.create_pipeline_layout(&pd));
        let pipeline = |entry: &str| {
            let mut stage = ProgrammableStage::new(&module);
            stage.entry_point = Some(entry.into());
            let mut cd = ComputePipelineDescriptor::new(PipelineLayoutOrAuto::PipelineLayout(&pl), stage);
            cd.label = format!("{LABEL} {entry}");
            g.call("GPUDevice.createComputePipeline", LABEL, |d| d.create_compute_pipeline(&cd))
        };
        let k = Rc::new(Kernels {
            count: pipeline(kernel::ENTRY_COUNT),
            scan: pipeline(kernel::ENTRY_SCAN),
            generate: pipeline(kernel::ENTRY_GENERATE),
            layout,
        });
        *self.kernels.borrow_mut() = Some(k.clone());
        Ok(k)
    }
}

/// One patch draw to tessellate.
pub struct Job<A: Api> {
    /// The pipeline's tessellation state and patch type.
    pub settings: Settings,
    pub patch: PatchType,
    /// The factor scale applied: 1 for a pipeline without `tessellationFactorScaleEnabled`, else the encoder's
    /// `setTessellationFactorScale:` value, 0 when it never set one (measured, tessellation.factors scale-unset).
    pub scale: f32,
    /// `setTessellationFactorBuffer:offset:instanceStride:`.
    pub factors: Rc<Buffer<A>>,
    pub factor_offset: u64,
    pub instance_stride: u64,
    /// `MTLTessellationFactorStepFunction`.
    pub step: u64,
    pub patch_start: u64,
    pub patch_count: u64,
    pub instance_count: u64,
    pub base_instance: u64,
    /// The patch index buffer and its offset.
    pub patch_indices: Option<(Rc<Buffer<A>>, u64)>,
    /// What the draw is called in reports.
    pub what: String,
}

/// A patch draw's generated geometry.
pub struct Geometry<A: Api> {
    /// The point records ([`airwgsl::BindingSource::TessellationVertices`]).
    pub vertices: A::Buffer,
    /// The `uint32` index buffer.
    pub indices: A::Buffer,
    /// The `drawIndexedIndirect` arguments at offset 0.
    pub draw: A::Buffer,
}

impl<A: Api> Clone for Geometry<A> {
    fn clone(&self) -> Self {
        Geometry { vertices: self.vertices.clone(), indices: self.indices.clone(), draw: self.draw.clone() }
    }
}

/// What [`Tessellator::encode`] recorded for a draw.
pub struct Generated<A: Api> {
    pub geometry: Geometry<A>,
    /// After the command buffer has run: whether a patch had a blade the M4 batches, as a failure.
    pub check: GpuFuture<std::result::Result<(), Failure>>,
}

/// The device limits the kernels' bindings respect.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub storage_offset_alignment: u64,
    pub max_storage_binding: u64,
    pub max_buffer: u64,
}

/// The binding of `buffer` from an aligned base at or before `offset` to its end: (base, size).
fn window<A: Api>(b: &Buffer<A>, offset: u64, limits: Limits, what: &str) -> Result<(u64, u64)> {
    let base = offset - offset % limits.storage_offset_alignment;
    let size = b.gpu().size() - base;
    if size > limits.max_storage_binding {
        return Err(RenderError::host(format!(
            "{what}: a {size}-byte storage binding exceeds the device's maxStorageBufferBindingSize {}",
            limits.max_storage_binding
        )));
    }
    Ok((base, size))
}

impl<A: Api> Tessellator<A> {
    /// Encode the kernels of `job` on `encoder` (outside any pass).
    pub fn encode(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        job: &Job<A>,
        limits: Limits,
    ) -> Result<Generated<A>> {
        let k = self.kernels(g)?;
        let what = &job.what;
        let u32_of = |v: u64, name: &str| {
            u32::try_from(v).map_err(|_| RenderError::unimplemented(format!("{what}: {name} {v} beyond 32 bits")))
        };
        let size = job.patch.factors_size();
        if job.factor_offset > job.factors.length {
            return Err(RenderError::undefined(
                what.clone(),
                format!(
                    "tessellation factor buffer offset {} past its {} bytes",
                    job.factor_offset, job.factors.length
                ),
            ));
        }
        // The last factors the draw reads must lie in the buffer (Metal reads past it with an undefined result).
        let last_patch = job.patch_start + job.patch_count.max(1) - 1;
        let last_instance = job.base_instance + job.instance_count.max(1) - 1;
        let mut end = job.factor_offset + size;
        if job.step == 1 || job.step == 3 {
            end += last_patch * size;
        }
        if job.step == 2 || job.step == 3 {
            end += last_instance * job.instance_stride;
        }
        if end > job.factors.length {
            return Err(RenderError::undefined(
                what.clone(),
                format!(
                    "the draw reads tessellation factors up to byte {end} of a {}-byte factor buffer",
                    job.factors.length
                ),
            ));
        }
        let (factor_base, factor_size) = window(&job.factors, job.factor_offset, limits, what)?;
        let mut p = kernel::Params::new(&job.settings, job.patch).with_scale(job.scale);
        p.step = job.step as u32;
        p.factor_offset = u32_of(job.factor_offset - factor_base, "factor offset")?;
        p.instance_stride = u32_of(job.instance_stride, "instance stride")?;
        p.patch_start = u32_of(job.patch_start, "patch start")?;
        p.patch_count = u32_of(job.patch_count, "patch count")?;
        p.instance_count = u32_of(job.instance_count, "instance count")?;
        p.base_instance = u32_of(job.base_instance, "base instance")?;
        let patch_window = match &job.patch_indices {
            Some((b, offset)) => {
                if !offset.is_multiple_of(4) || offset + 4 * (job.patch_start + job.patch_count) > b.length {
                    return Err(RenderError::undefined(
                        what.clone(),
                        format!(
                            "patch indices {}..{} at offset {offset} of a {}-byte patch index buffer",
                            job.patch_start,
                            job.patch_start + job.patch_count,
                            b.length
                        ),
                    ));
                }
                let (base, wsize) = window(b, *offset, limits, what)?;
                p.patch_index_offset = Some(u32_of(offset - base, "patch index offset")?);
                Some((b.clone(), base, wsize))
            }
            None => None,
        };
        let slots = u64::from(p.slots());
        let (max_points, max_indices) = kernel::bounds(&job.settings, job.patch);
        let vertex_bytes = slots * u64::from(max_points) * 4 * u64::from(kernel::VERTEX_WORDS);
        let index_bytes = slots * u64::from(max_indices) * 4;
        for (bytes, name) in [(vertex_bytes, "point records"), (index_bytes, "indices")] {
            if bytes > limits.max_storage_binding || bytes > limits.max_buffer {
                return Err(RenderError::host(format!(
                    "{what}: {slots} patches need up to {bytes} bytes of {name} at the pipeline's maximum factor; the \
                     device's maxStorageBufferBindingSize is {} and maxBufferSize {}",
                    limits.max_storage_binding, limits.max_buffer
                )));
            }
        }
        let make = |len: u64, usage: u32, name: &str| -> Result<A::Buffer> {
            let mut d = BufferDescriptor::new(size64(len.max(16), "GPUDevice.createBuffer")?, usage);
            d.label = format!("{LABEL} {name}");
            g.call("GPUDevice.createBuffer", &d.label, |dev| dev.create_buffer(&d))
                .map_err(|e| RenderError::gpu("GPUDevice.createBuffer", e))
        };
        let params = make(64, BufferUsage::UNIFORM | BufferUsage::COPY_DST, "parameters")?;
        let slot_buffer = make(slots * 4 * u64::from(kernel::SLOT_WORDS), BufferUsage::STORAGE, "slots")?;
        let vertices = make(vertex_bytes, BufferUsage::STORAGE, "point records")?;
        let indices = make(index_bytes, BufferUsage::STORAGE | BufferUsage::INDEX, "indices")?;
        let draw = make(
            4 * u64::from(kernel::DRAW_WORDS),
            BufferUsage::STORAGE | BufferUsage::INDIRECT | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            "draw",
        )?;
        let flag = make(16, BufferUsage::MAP_READ | BufferUsage::COPY_DST, "batched flag")?;
        // Buffers are created zeroed (WebGPU 5.1.4): the batched flag starts clear.
        let bytes = p.bytes();
        g.call("GPUQueue.writeBuffer", LABEL, |_| {
            g.queue().write_buffer(&params, Size64::from(0u32), &bytes, Size64::from(0u32), None)
        })
        .map_err(|e| RenderError::gpu("GPUQueue.writeBuffer", e))?;
        fn binding<A: Api>(buffer: &A::Buffer, offset: u64, size: u64) -> Result<BindingResource<'_, A>> {
            let mut b = BufferBinding::new(buffer);
            b.offset = size64(offset, "GPUDevice.createBindGroup")?;
            b.size = Some(size64(size, "GPUDevice.createBindGroup")?);
            Ok(BindingResource::BufferBinding(b))
        }
        let (pi_buffer, pi_base, pi_size) = match &patch_window {
            Some((b, base, size)) => (b.gpu(), *base, *size),
            // Bound and never read: a draw without a patch index buffer leaves the parameter unset.
            None => (job.factors.gpu(), factor_base, factor_size),
        };
        let entries = vec![
            BindGroupEntry::new(kernel::Binding::Params as u32, BindingResource::Buffer(&params)),
            BindGroupEntry::new(
                kernel::Binding::Factors as u32,
                binding::<A>(job.factors.gpu(), factor_base, factor_size)?,
            ),
            BindGroupEntry::new(kernel::Binding::PatchIndices as u32, binding::<A>(pi_buffer, pi_base, pi_size)?),
            BindGroupEntry::new(kernel::Binding::Slots as u32, BindingResource::Buffer(&slot_buffer)),
            BindGroupEntry::new(kernel::Binding::Vertices as u32, BindingResource::Buffer(&vertices)),
            BindGroupEntry::new(kernel::Binding::Indices as u32, BindingResource::Buffer(&indices)),
            BindGroupEntry::new(kernel::Binding::Draw as u32, BindingResource::Buffer(&draw)),
        ];
        let mut bd = BindGroupDescriptor::new(&k.layout, entries);
        bd.label = LABEL.into();
        let group = g.call("GPUDevice.createBindGroup", LABEL, |dev| dev.create_bind_group(&bd));
        let groups = u32_of(slots.div_ceil(u64::from(kernel::WORKGROUP)), "workgroup count")?;
        for (pipeline, count) in [(&k.count, groups), (&k.scan, 1), (&k.generate, groups)] {
            let mut pd = ComputePassDescriptor::new();
            pd.label = LABEL.into();
            let pass = g
                .call("GPUCommandEncoder.beginComputePass", LABEL, |_| encoder.begin_compute_pass(&pd))
                .map_err(|e| RenderError::gpu("GPUCommandEncoder.beginComputePass", e))?;
            g.call("GPUComputePassEncoder.setPipeline", LABEL, |_| pass.set_pipeline(pipeline));
            g.call("GPUComputePassEncoder.setBindGroup", LABEL, |_| pass.set_bind_group(0, Some(&group), &[]));
            if count > 0 {
                g.call("GPUComputePassEncoder.dispatchWorkgroups", LABEL, |_| pass.dispatch_workgroups(count, 1, 1));
            }
            g.call("GPUComputePassEncoder.end", LABEL, |_| pass.end());
        }
        let flag_at = Size64::from(4 * kernel::DRAW_BATCHED);
        g.call("GPUCommandEncoder.copyBufferToBuffer", LABEL, |_| {
            encoder.copy_buffer_to_buffer_with_offsets(
                &draw,
                flag_at,
                &flag,
                Size64::from(0u32),
                Some(Size64::from(4u32)),
            )
        });
        let what = what.clone();
        let check: GpuFuture<std::result::Result<(), Failure>> = Box::pin(async move {
            let size = Size64::from(4u32);
            let unreadable = |e: &dyn std::fmt::Debug| {
                failure::host_unsupported(format!("{what}: the tessellator's batched flag could not be read: {e:?}"))
            };
            flag.map_async(MapMode::READ, Size64::new_const(0), Some(size)).await.map_err(|e| unreadable(&e))?;
            let range = flag.get_mapped_range(Size64::new_const(0), Some(size)).map_err(|e| unreadable(&e))?;
            let mut word = [0u8; 4];
            range.read(0, &mut word).map_err(|e| unreadable(&e))?;
            drop(range);
            flag.unmap();
            if u32::from_le_bytes(word) != 0 {
                return Err(failure::unimplemented(format!(
                    "{what}: a patch's inside has a blade wider than 36 grid points, which the Apple M4 emits in \
                     tiles whose sizes depend on the patches before it; that triangle order is not reproduced \
                     (airwgsl::tess::hw::Counts::batched)"
                )));
            }
            Ok(())
        });
        Ok(Generated { geometry: Geometry { vertices, indices, draw }, check })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use webmetal_wire::status;

    fn state(partition_mode: u64, max_tessellation_factor: u64) -> airwgsl::Tessellation {
        airwgsl::Tessellation {
            partition_mode,
            max_tessellation_factor,
            factor_scale_enabled: false,
            factor_format: 0,
            control_point_index_type: 0,
            factor_step_function: 0,
            output_winding_order: 0,
        }
    }

    /// The maximum factors Metal.framework accepts for each partition mode (measured on Apple M4, macOS 27.0.1:
    /// "maxTessellationFactor must be >= 1 and <= 64", "must be a power of 2 when using
    /// MTLTessellationPartitionModePow2", "must be even when using MTLTessellationPartitionModeFractionalOdd or
    /// MTLTessellationPartitionModeFractionalEven"), and the A10X's maximum of 16.
    #[test]
    fn maximum_factors_follow_metal_s_checks() {
        let plan = |mode, max| pass_plan("call", Some(&state(mode, max)), true);
        for (mode, max) in [(0, 16), (0, 8), (1, 13), (1, 1), (2, 16), (3, 2), (1, 16)] {
            assert!(plan(mode, max).is_ok(), "{mode} {max}");
        }
        for (mode, max, needle) in [
            (1, 0, "at least 1 and at most 64"),
            (2, 65, "at least 1 and at most 64"),
            (0, 12, "not a power of 2"),
            (2, 13, "is odd"),
            (3, 15, "is odd"),
            (1, 32, "above the A10X's maxTessellationFactor of 16"),
            (0, 64, "above the A10X's maxTessellationFactor of 16"),
        ] {
            let f = plan(mode, max).unwrap_err();
            assert_eq!(f.status, status::UNIMPLEMENTED, "{mode} {max}: {f:?}");
            assert!(f.message.contains(needle), "{mode} {max}: {f:?}");
        }
        assert_eq!(pass_plan("call", None, true).unwrap_err().status, status::INVALID);
    }
}
