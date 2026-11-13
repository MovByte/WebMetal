//! Compute pipelines: `CreateComputePipeline` and `CreateComputePipelineAsync`, and the variants the compute encoder
//! asks for at each dispatch (the seam of `crate::compute::pipeline`).
//!
//! **Creation.** The descriptor is checked field by field ([`check_descriptor`]): every field either changes what is
//! compiled, is kept for the encoder, cannot change a result, or has a value whose behavior the host does not
//! implement, which fails with status 2 naming the field. The compute function must be a kernel. The pipeline is then
//! compiled once at creation, for the *creation variant*: one thread per threadgroup, every `[[threadgroup(n)]]`
//! argument given the length of its type (rounded up to 16 bytes, as `setThreadgroupMemoryLength:atIndex:` requires),
//! no bound resources. Nothing the compiler refuses depends on those choices except the dispatch state itself, so a
//! refusal there is the pipeline's and fails the creation as [`super::errors::refusal`] maps it; the creation variant's
//! WebGPU pipeline is made with `createComputePipelineAsync`, so a compiler output WebGPU rejects fails the creation
//! too (status 2); its WebGPU objects stay in the object cache, where a dispatch variant that compiles to the same
//! WGSL finds them. The reply carries the compiler's reflection when the caller asked for it.
//!
//! A dispatch variant is compiled for what the encoder bound and set. A bound texture the kernel's argument cannot take
//! fails the dispatch as behavior Metal does not define, naming the index and both types ([`super::bound`]), before
//! the compiler sees it; any other refusal of a dispatch variant is the bound or set state's
//! ([`super::errors::variant_refusal`]), never the pipeline's creation.
//!
//! **Pipeline properties.** The A10X driver takes `maxTotalThreadsPerThreadgroup` and `staticThreadgroupMemoryLength`
//! from its back-end compiler's reply (`AGX::ComputeProgramVariant` constructor, AGXMetalA10 21H461 0x21c92f9b4: the
//! limit at +0xb38 is a reply field when the compiler sets one, otherwise the smaller of two register-file terms,
//! each at most 512, rounded down to a multiple of 32 above 32 (0x21c934520-0x21c9345b0); the threadgroup memory at
//! +0xb34 is a reply field rounded up to 16 (0x21c934458-0x21c934488); `-[AGXA10FamilyComputePipeline
//! maxTotalThreadsPerThreadgroup]` and `staticThreadgroupMemoryLength` (0x21c8cc39c, 0x21c8cc3b0) read them back).
//! Those register counts exist only in the output of Apple's back-end compiler, so the host cannot reproduce either
//! value; the guest reports them as not established (the schema's guest-local note on `MTLComputePipelineState`). What
//! the encoder checks dispatches against is therefore a bound that is exact in the direction it is used:
//!
//! - `max_total_threads_per_threadgroup` is at least the A10X's value: 512 (the bound of the driver's computation and
//!   the device's `maxThreadsPerThreadgroup`), the descriptor's `maxTotalThreadsPerThreadgroup` when set, and the
//!   function's `[[max_total_threads_per_threadgroup]]` when declared, whichever is least. Metal never reports a
//!   pipeline limit above any of them, so a dispatch above the bound is above the A10X's limit as well.
//! - `static_threadgroup_memory_length` is 0, which is at most the A10X's value: the encoder's check against
//!   `maxThreadgroupMemoryLength` then counts the lengths the dispatch sets and never refuses a dispatch the A10X
//!   accepts. The memory the compiled variant declares is checked against the host's limit when the variant is made.
//! - `thread_execution_width` is 32: `-[AGXA10FamilyComputePipeline threadExecutionWidth]` returns the constant 32
//!   (0x21c8cc394).
//!
//! **Variants.** WGSL fixes the workgroup size and the size of every `var<workgroup>`, and the bound textures decide
//! WGSL texture types, so each dispatch state the encoder names ([`VariantKey`]) is compiled once per pipeline: the
//! compiler cache and the WebGPU object cache make a variant another pipeline already needed free.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use airwgsl::{
    BindingSource, BufferSlot, ComputeState, MetalStage, PipelineDesc, PipelineKind, StageBindings,
    StageInputDescriptor, TextureSlot,
};
use webmetal_wire::{ComputePipelineDescriptor, Failure, Handle};

use super::Shared;
use super::gpu::{self, ComputeObjects};
use super::registry::{Function, function_type};
use super::{bound, errors};
use crate::compute::pipeline::{
    BindGroup, Binding, ComputePipelineState, ComputeVariant, ComputeVariants, Resource, Source, UniformBlock,
    UniformValue, VariantKey,
};
use crate::exec::{Gpu, Table, failure};
use crate::gpu::Api;
use crate::mem::MemError;

/// `threadExecutionWidth` of every A10X compute pipeline (`-[AGXA10FamilyComputePipeline threadExecutionWidth]`,
/// AGXMetalA10 21H461 0x21c8cc394: `mov w0, #0x20; ret`).
pub const THREAD_EXECUTION_WIDTH: u64 = 32;

/// The largest `maxTotalThreadsPerThreadgroup` the A10X driver computes from its compiler's register counts (both
/// terms start at and are capped to 0x200 before the minimum is taken, AGXMetalA10 21H461 0x21c934520-0x21c93459c);
/// the device's `maxThreadsPerThreadgroup` is 512 in each dimension (device profile).
pub const MAX_THREADS_BOUND: u64 = 512;

/// `MTLPipelineOption` (iPhoneOS SDK, `MTLDevice.h`), and the bits beyond it that the 21H461 frameworks pass.
pub mod option {
    use webmetal_wire::Failure;

    use crate::exec::failure;

    pub const BINDING_INFO: u64 = 1 << 0;
    pub const BUFFER_TYPE_INFO: u64 = 1 << 1;
    pub const FAIL_ON_BINARY_ARCHIVE_MISS: u64 = 1 << 2;
    /// The options this service implements.
    pub const ALL: u64 = BINDING_INFO | BUFFER_TYPE_INFO | FAIL_ON_BINARY_ARCHIVE_MISS;

    /// The debug layer's options: `-[MTLToolsDevice debugPipelineOptions:includePipelinePerfStats:]` (MetalTools
    /// 21H461, 0x216e6644c) ORs 0x810003 (bits 0, 1, 16 and 23) into the caller's options, and 0x40000 as well when
    /// the caller did not pass FailOnBinaryArchiveMiss and asked for performance statistics.
    pub const DEBUG_16: u64 = 1 << 16;
    pub const DEBUG_23: u64 = 1 << 23;
    /// Set by `-[MTLDebugDevice newRenderPipelineStateWithDescriptor:error:]` (MetalTools 21H461, 0x216ec2c60) and
    /// cleared by the debug layer's forwarding methods (0x216ec2c58 and 0x216ec3148: `and x3, x3, #~0x20000`) before
    /// the options reach the driver.
    pub const DEBUG_17: u64 = 1 << 17;
    /// The pipeline's performance statistics: the A10X driver reads the bit at creation (AGXMetalA10 21H461
    /// 0x21c8df82c), asks its compiler for the pipeline's statistics (`getComputePipelinePerformanceStatistics`,
    /// 0x21c8df834) and keeps them with `setPerformanceStatistics:`; `createComputeProgramVariant` masks the options
    /// with 0xc0040000 (0x21c8a6508).
    pub const PERFORMANCE_STATISTICS: u64 = 1 << 18;

    /// Check `options` bit by bit: every bit outside [`ALL`] is a loud failure naming the bit and what is known of it.
    pub fn check(call: &str, options: u64) -> Result<(), Failure> {
        let Some(bit) = (0..64).map(|b| 1u64 << b).find(|b| options & b != 0 && ALL & b == 0) else {
            return Ok(());
        };
        let why = match bit {
            PERFORMANCE_STATISTICS => {
                "the A10X driver then gathers its GPU compiler's performance statistics for the pipeline (AGXMetalA10 \
                 21H461 0x21c8df82c-0x21c8df834), which this host's compiler, AirWgsl over WebGPU, does not produce"
            }
            DEBUG_16 | DEBUG_23 | DEBUG_17 => {
                "the A10X driver hands the options to its back-end compile request (createComputeProgramVariant, \
                 AGXMetalA10 21H461: setPipelineOptions: at 0x21c8a66b4 with options | 3), and bits 16 and 17 are \
                 among the bits (0x330003, tested at 0x21c8a66c0) that take creation down its reflection path; what \
                 the back end (AGXCompilerCore, in MTLCompilerService) does with the bit is not established. \
                 MetalTools' debug layer sets bits 16 and 23 (debugPipelineOptions:includePipelinePerfStats:, \
                 0x216e6644c) and bit 17 (MTLDebugDevice, 0x216ec2c60)"
            }
            _ => "it is not an MTLPipelineOption of the iPhoneOS SDK, nor a bit the 21H461 frameworks pass",
        };
        Err(failure::unimplemented(format!("{call}: pipeline option {bit:#x} (of options {options:#x}): {why}")))
    }
}

/// `MTLMutability` values (iPhoneOS SDK, `MTLPipeline.h`): Default, Mutable, Immutable.
pub const MUTABILITY_VALUES: [u64; 3] = [0, 1, 2];

/// The function a handle names, as this service registered it.
pub fn function(objects: &Table, handle: Handle) -> Result<Rc<Function>, Failure> {
    let entry =
        objects.get(handle).ok_or_else(|| failure::invalid(format!("function {:#x} is not live", handle.get())))?;
    entry
        .payload
        .clone()
        .and_then(|p| p.downcast::<Function>().ok())
        .ok_or_else(|| failure::invalid(format!("function {:#x} was not made by RegisterFunction", handle.get())))
}

/// What creation keeps of a compute descriptor besides the function.
#[derive(Clone, Debug)]
pub struct ComputeSettings {
    pub label: String,
    /// `maxTotalThreadsPerThreadgroup` (0: not set).
    pub max_total_threads_per_threadgroup: u64,
    pub threadgroup_size_is_multiple_of_thread_execution_width: bool,
    pub support_indirect_command_buffers: bool,
    pub stage_input: Option<StageInputDescriptor>,
}

/// Check every field of a compute pipeline descriptor (see the module documentation) and keep what creation uses.
/// `call` names the Metal method for messages.
pub fn check_descriptor(call: &str, d: &ComputePipelineDescriptor) -> Result<ComputeSettings, Failure> {
    let ComputePipelineDescriptor {
        // Looked up by the caller under MTLPipelineOptionFailOnBinaryArchiveMiss; otherwise an archive only saves
        // compilation time.
        binary_archives: _,
        binary_libraries,
        buffers,
        compute_function: _,
        driver_compiler_options,
        dynamic_libraries,
        force_resource_index,
        function_pointers,
        gpu_compiler_spi_options,
        insert_libraries,
        label,
        linked_functions,
        // Bounds of ray traversal and of call stacks through function pointers; with no intersection functions and no
        // function pointers (both refused) they cannot change a result.
        max_acceleration_structure_traversal_depth: _,
        max_call_stack_depth: _,
        max_total_threads_per_threadgroup,
        needs_custom_border_color_samplers,
        open_cl_mode_enabled,
        open_gl_mode_enabled,
        pipeline_library,
        plugin_data,
        preloaded_libraries,
        profile_control,
        resource_index,
        stage_input_descriptor,
        support_adding_binary_functions,
        support_indirect_command_buffers,
        texture_write_fp_rounding_mode,
        texture_write_rounding_mode,
        thread_group_size_is_multiple_of_thread_execution_width,
    } = d;
    // MTLDynamicLibrary is unsupported on the A10X; no record creates a DynamicLibrary handle, so the executor core has
    // already refused any handle in these lists.
    for (name, list) in [
        ("binaryLibraries", binary_libraries),
        ("dynamicLibraries", dynamic_libraries),
        ("insertLibraries", insert_libraries),
        ("preloadedLibraries", preloaded_libraries),
    ] {
        if !list.is_empty() {
            return Err(failure::invalid(format!("{call}: {name} names dynamic libraries, which no record creates")));
        }
    }
    for (i, b) in buffers.iter().enumerate() {
        if !MUTABILITY_VALUES.contains(&b.mutability) {
            return Err(failure::invalid(format!(
                "{call}: buffers[{i}].mutability {} is not an MTLMutability",
                b.mutability
            )));
        }
    }
    let unknown = "the behavior of a value other than the default (measured on macOS 27.0.1) is not established";
    if driver_compiler_options.is_some() {
        return Err(errors::field(call, "driverCompilerOptions", "a dictionary", unknown));
    }
    if gpu_compiler_spi_options.is_some() {
        return Err(errors::field(call, "gpuCompilerSPIOptions", "a dictionary", unknown));
    }
    if plugin_data.is_some() {
        return Err(errors::field(call, "pluginData", "a dictionary", unknown));
    }
    if *force_resource_index {
        return Err(errors::field(
            call,
            "forceResourceIndex",
            format!("YES (resourceIndex {resource_index})"),
            unknown,
        ));
    }
    if *needs_custom_border_color_samplers {
        return Err(errors::field(call, "needsCustomBorderColorSamplers", "YES", unknown));
    }
    if *open_cl_mode_enabled {
        return Err(errors::field(call, "openCLModeEnabled", "YES", unknown));
    }
    if *open_gl_mode_enabled {
        return Err(errors::field(call, "openGLModeEnabled", "YES", unknown));
    }
    if *pipeline_library {
        return Err(errors::field(
            call,
            "pipelineLibrary",
            "an MTLPipelineLibrary",
            "pipeline libraries are not implemented",
        ));
    }
    if profile_control.is_some() {
        return Err(errors::field(call, "profileControl", "an MTLProfileControl", unknown));
    }
    if *texture_write_fp_rounding_mode != 0 {
        return Err(errors::field(call, "textureWriteFPRoundingMode", texture_write_fp_rounding_mode, unknown));
    }
    if *texture_write_rounding_mode != 0 {
        return Err(errors::field(call, "textureWriteRoundingMode", texture_write_rounding_mode, unknown));
    }
    if !function_pointers.is_empty() {
        return Err(errors::field(
            call,
            "functionPointers",
            format!("{} functions", function_pointers.len()),
            "function pointers are not implemented",
        ));
    }
    if *support_adding_binary_functions {
        return Err(errors::field(call, "supportAddingBinaryFunctions", "YES", "binary functions are not implemented"));
    }
    if let Some(l) = linked_functions {
        let n = l.binary_functions.len() + l.functions.len() + l.groups.len() + l.private_functions.len();
        if n > 0 {
            return Err(errors::field(
                call,
                "linkedFunctions",
                format!("{n} entries"),
                "linked functions are not implemented",
            ));
        }
    }
    Ok(ComputeSettings {
        label: label.clone().unwrap_or_default(),
        max_total_threads_per_threadgroup: *max_total_threads_per_threadgroup,
        threadgroup_size_is_multiple_of_thread_execution_width:
            *thread_group_size_is_multiple_of_thread_execution_width,
        support_indirect_command_buffers: *support_indirect_command_buffers,
        stage_input: stage_input_descriptor.as_ref().map(|s| StageInputDescriptor {
            layouts: s.layouts.iter().map(|l| (l.stride, l.step_function, l.step_rate)).collect(),
            attributes: s.attributes.iter().map(|a| (a.format, a.offset, a.buffer_index)).collect(),
            index_type: s.index_type,
            index_buffer_index: s.index_buffer_index,
        }),
    })
}

/// What the creation variant needs to know of the kernel: its threadgroup arguments, whether it has a stage input, and
/// its `[[max_total_threads_per_threadgroup]]`.
pub struct KernelInfo {
    /// `[[threadgroup(n)]]` arguments: (index, length of the argument's type).
    pub threadgroup_arguments: Vec<(u64, u64)>,
    pub stage_in: bool,
    pub max_total_threads: Option<u64>,
}

/// Decode a function's module as the compiler decodes it (AirWgsl `wgsl::input`), with the refusals it names.
// The refusal is the compiler's failure value and is returned by value, as airwgsl::compile returns it.
#[allow(clippy::result_large_err)]
pub fn decode_module(f: &Function) -> Result<airwgsl::air::AirModule, airwgsl::Refusal> {
    let ir = airwgsl::parse_bitcode(&f.air.bitcode)
        .map_err(|e| airwgsl::wgsl::tracked::refuse("module-unreadable", "module", None, e.to_string()))?;
    airwgsl::air::AirModule::decode(ir).map_err(airwgsl::wgsl::tracked::name_item)
}

/// Read what [`KernelInfo`] holds from the function's module. A module or entry the AIR layer cannot read is the
/// compiler's refusal, which compiling it would return as well.
// The refusal is the compiler's failure value and is returned by value, as airwgsl::compile returns it.
#[allow(clippy::result_large_err)]
pub fn kernel_info(f: &Function) -> Result<KernelInfo, airwgsl::Refusal> {
    let module = decode_module(f)?;
    let Some(entry) = module.entry_point(&f.name) else {
        // The compiler names this refusal for the same input.
        return Err(airwgsl::wgsl::tracked::refuse(
            "entry-metadata",
            f.name.clone(),
            None,
            "the function is not an entry point of the module",
        ));
    };
    let mut threadgroup_arguments = Vec::new();
    let mut stage_in = false;
    for a in &entry.arguments {
        match &a.role {
            airwgsl::air::Role::Buffer(b) if b.address_space == Some(3) => {
                if let airwgsl::air::Slot::Value(i) = b.location.index {
                    threadgroup_arguments.push((u64::from(i), u64::from(a.type_size.unwrap_or(0))));
                }
            }
            airwgsl::air::Role::StageIn(_) => stage_in = true,
            _ => {}
        }
    }
    Ok(KernelInfo { threadgroup_arguments, stage_in, max_total_threads: entry.max_work_group_size.map(u64::from) })
}

/// The pipeline description of a dispatch state.
fn desc(
    stage_input: &Option<StageInputDescriptor>,
    threads: [u64; 3],
    lengths: Vec<u64>,
    bindings: Vec<StageBindings>,
) -> PipelineDesc {
    PipelineDesc {
        kind: PipelineKind::Compute(ComputeState {
            threads_per_threadgroup: threads,
            non_uniform_grid: false,
            threadgroup_memory_lengths: lengths,
            stage_input: stage_input.clone(),
        }),
        bindings,
    }
}

/// The creation variant's description (module documentation).
pub fn creation_desc(settings: &ComputeSettings, info: &KernelInfo) -> PipelineDesc {
    let mut lengths: Vec<u64> = Vec::new();
    for &(index, size) in &info.threadgroup_arguments {
        let i = index as usize;
        if lengths.len() <= i {
            lengths.resize(i + 1, 0);
        }
        lengths[i] = size.max(1).div_ceil(16) * 16;
    }
    desc(&settings.stage_input, [1, 1, 1], lengths, Vec::new())
}

/// The variant description of a dispatch the encoder names.
pub fn variant_desc(stage_input: &Option<StageInputDescriptor>, key: &VariantKey) -> Result<PipelineDesc, Failure> {
    let bindings = StageBindings {
        stage: MetalStage::Kernel,
        // The encoder's key does not say which bound ranges overlap; slots are compiled as distinct buffers.
        buffers: key
            .buffers
            .iter()
            .map(|&index| BufferSlot { index, alias_group: None, also_vertex_or_index_buffer: false })
            .collect(),
        textures: key
            .textures
            .iter()
            .map(|t| TextureSlot {
                index: t.index,
                texture_type: t.texture_type,
                pixel_format: t.pixel_format,
                sample_count: t.sample_count,
                swizzle: t.swizzle,
            })
            .collect(),
        samplers: key
            .samplers
            .iter()
            .map(|s| airwgsl::SamplerSlot {
                index: s.index,
                min_filter: s.min_filter,
                mag_filter: s.mag_filter,
                mip_filter: s.mip_filter,
                address_modes: s.address_modes,
                border_color: s.border_color,
                normalized_coordinates: s.normalized_coordinates,
                compare_function: s.compare_function,
                max_anisotropy: s.max_anisotropy,
                lod_min_clamp: f32::from_bits(s.lod_min_clamp),
                lod_max_clamp: f32::from_bits(s.lod_max_clamp),
                lod_average: s.lod_average,
            })
            .collect(),
        argument_buffers: Vec::new(),
    };
    let t = key.threads_per_threadgroup;
    Ok(desc(
        stage_input,
        [u64::from(t[0]), u64::from(t[1]), u64::from(t[2])],
        key.threadgroup_memory_lengths.clone(),
        vec![bindings],
    ))
}

/// The encoder's form of a compiled binding layout.
fn bind_groups<A: Api>(
    label: &str,
    compiled: &airwgsl::Compiled,
    objects: &ComputeObjects<A>,
) -> Result<Vec<BindGroup<A>>, Failure> {
    let mut out = Vec::new();
    for g in &compiled.layout.groups {
        let layout = objects
            .layouts
            .groups
            .iter()
            .find(|(i, _)| *i == g.group)
            .map(|(_, l)| l.clone())
            .ok_or_else(|| failure::unimplemented(format!("{label}: no layout for bind group {}", g.group)))?;
        let mut entries = Vec::new();
        for e in &g.entries {
            let what =
                |s: &str| failure::unimplemented(format!("{label}: binding {} of group {}: {s}", e.binding, g.group));
            let resource = match &e.resource {
                airwgsl::BindingResource::Buffer { ty, has_dynamic_offset, min_binding_size } => {
                    if *has_dynamic_offset {
                        return Err(what("a buffer with a dynamic offset, which the compute encoder does not bind"));
                    }
                    let ty = match ty {
                        airwgsl::BufferBindingType::Uniform => crate::gpu::idl::BufferBindingType::Uniform,
                        airwgsl::BufferBindingType::Storage => crate::gpu::idl::BufferBindingType::Storage,
                        airwgsl::BufferBindingType::ReadOnlyStorage => {
                            crate::gpu::idl::BufferBindingType::ReadOnlyStorage
                        }
                    };
                    Resource::Buffer { ty, min_binding_size: *min_binding_size }
                }
                airwgsl::BindingResource::Sampler { ty } => Resource::Sampler { ty: gpu::sampler_type(*ty) },
                airwgsl::BindingResource::Texture { sample_type, view_dimension, multisampled } => Resource::Texture {
                    sample_type: gpu::sample_type(*sample_type),
                    view_dimension: gpu::view_dimension(*view_dimension),
                    multisampled: *multisampled,
                },
                airwgsl::BindingResource::StorageTexture { access, format, view_dimension } => {
                    Resource::StorageTexture {
                        access: gpu::storage_access(*access),
                        format: gpu::texture_format(format)?,
                        view_dimension: gpu::view_dimension(*view_dimension),
                    }
                }
            };
            let kernel = |stage: &MetalStage| *stage == MetalStage::Kernel;
            let source = match &e.source {
                BindingSource::MetalBuffer { stage, index } if kernel(stage) => Source::Buffer { index: *index },
                BindingSource::MetalTexture { stage, index } if kernel(stage) => Source::Texture { index: *index },
                BindingSource::MetalSampler { stage, index } if kernel(stage) => Source::Sampler { index: *index },
                BindingSource::Uniforms(block) => {
                    let mut fields = Vec::new();
                    for f in &block.fields {
                        let value = match &f.value {
                            airwgsl::UniformValue::ResidualOffset { stage, index } if kernel(stage) => {
                                UniformValue::ResidualOffset { index: *index }
                            }
                            airwgsl::UniformValue::BufferLength { stage, index } if kernel(stage) => {
                                UniformValue::BufferLength { index: *index }
                            }
                            // Three words each, at consecutive offsets, in both.
                            airwgsl::UniformValue::StageInOrigin => UniformValue::StageInOrigin,
                            airwgsl::UniformValue::StageInSize => UniformValue::StageInSize,
                            other => return Err(what(&format!("a uniform field holding {other:?}"))),
                        };
                        fields.push((f.offset, value));
                    }
                    Source::Uniforms(UniformBlock { size: block.size, fields })
                }
                other => return Err(what(&format!("the source {other:?}"))),
            };
            entries.push(Binding { binding: e.binding, resource, source });
        }
        out.push(BindGroup { group: g.group, layout, entries });
    }
    // The groups the compiled layout skips have empty layouts (see `gpu::create_layouts`), bound empty.
    for (index, layout) in &objects.layouts.groups {
        if !out.iter().any(|g| g.group == *index) {
            out.push(BindGroup { group: *index, layout: layout.clone(), entries: Vec::new() });
        }
    }
    out.sort_by_key(|g| g.group);
    Ok(out)
}

/// The variants of one compute pipeline.
pub struct Variants<A: Api> {
    shared: Rc<Shared<A>>,
    function: Rc<Function>,
    stage_input: Option<StageInputDescriptor>,
    label: String,
    /// The kernel's texture arguments, from the creation variant's reflection.
    textures: Vec<bound::TextureArgument>,
    made: RefCell<HashMap<VariantKey, Rc<ComputeVariant<A>>>>,
}

impl<A: Api> Variants<A> {
    pub fn new(
        shared: Rc<Shared<A>>,
        function: Rc<Function>,
        stage_input: Option<StageInputDescriptor>,
        label: String,
        textures: Vec<bound::TextureArgument>,
    ) -> Variants<A> {
        Variants { shared, function, stage_input, label, textures, made: RefCell::new(HashMap::new()) }
    }

    /// The number of variants made so far.
    pub fn count(&self) -> usize {
        self.made.borrow().len()
    }

    /// The compiled description of a variant (the compiler cache's entry), for comparison with a fresh compilation.
    pub fn compiled(&self, key: &VariantKey) -> Result<super::compile::Outcome, Failure> {
        let d = variant_desc(&self.stage_input, key)?;
        Ok(self.shared.compiler.compile(&[self.function.stage_module()], &d))
    }
}

/// The dispatch call variants are made for, for messages.
const DISPATCH: &str = "-[MTLComputeCommandEncoder dispatch...]";

impl<A: Api> ComputeVariants<A> for Variants<A> {
    fn variant(&self, gpu: &Gpu<'_, A>, key: &VariantKey) -> Result<Rc<ComputeVariant<A>>, Failure> {
        if let Some(v) = self.made.borrow().get(key) {
            return Ok(v.clone());
        }
        for t in &key.textures {
            if let Some(a) = bound::argument_at(&self.textures, t.index)
                && let Some(why) = bound::texture_mismatch(a, t.index, t.texture_type, t.pixel_format)
            {
                return Err(MemError::undefined(DISPATCH, why).failure());
            }
        }
        let outcome = self.compiled(key)?;
        let compiled = match &*outcome {
            Ok(c) => c,
            Err(r) => return Err(errors::variant_refusal(DISPATCH, r)),
        };
        gpu::requirements::<A>(DISPATCH, gpu.device(), &self.shared.wgsl_language_features, &compiled.requirements)?;
        let objects = self.shared.gpu.compute_now(gpu, &self.label, compiled)?;
        let v = Rc::new(ComputeVariant {
            pipeline: objects.pipeline.clone(),
            groups: bind_groups(&self.label, compiled, &objects)?,
        });
        self.made.borrow_mut().insert(key.clone(), v.clone());
        Ok(v)
    }
}

/// The payload of a compute pipeline handle for `function` with `settings`, and the kernel's own bound.
pub fn payload<A: Api>(
    shared: &Rc<Shared<A>>,
    function: Rc<Function>,
    settings: &ComputeSettings,
    info: &KernelInfo,
    reflection: &airwgsl::Reflection,
) -> (ComputePipelineState<A>, Rc<Variants<A>>) {
    let mut max = MAX_THREADS_BOUND;
    if settings.max_total_threads_per_threadgroup != 0 {
        max = max.min(settings.max_total_threads_per_threadgroup);
    }
    if let Some(m) = info.max_total_threads {
        max = max.min(m);
    }
    let textures = reflection.stages.first().map(bound::texture_arguments).unwrap_or_default();
    let variants = Rc::new(Variants::new(
        shared.clone(),
        function,
        settings.stage_input.clone(),
        settings.label.clone(),
        textures,
    ));
    let state = ComputePipelineState {
        max_total_threads_per_threadgroup: max,
        thread_execution_width: THREAD_EXECUTION_WIDTH,
        threadgroup_size_is_multiple_of_execution_width: settings
            .threadgroup_size_is_multiple_of_thread_execution_width,
        static_threadgroup_memory_length: 0,
        stage_input: info.stage_in,
        variants: variants.clone(),
    };
    (state, variants)
}

/// Check that a function given as a compute function is a kernel.
pub fn check_kernel(call: &str, f: &Function) -> Result<(), Failure> {
    if f.function_type != function_type::KERNEL {
        return Err(errors::driver_error(call, errors::WRONG_FUNCTION_TYPE, f.describe()));
    }
    Ok(())
}
