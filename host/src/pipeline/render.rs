//! Render pipelines: `CreateRenderPipeline` and its reflection and asynchronous forms.
//!
//! **Descriptor.** Every field of `MTLRenderPipelineDescriptor` the wire carries is classified ([`check_descriptor`]):
//! it changes what is compiled (the functions, vertex descriptor, attachment formats, blend state, sample count,
//! alpha-to-coverage, rasterization, primitive topology class, tessellation and amplification state), it is kept for
//! the render encoder (the WebGPU color targets, the depth-stencil format, the sample mask), it cannot change a result
//! for the pipeline's functions (tessellation fields of a pipeline whose vertex function has no patch, call-stack
//! depths without function pointers, mesh and object limits without mesh functions), or its value has a behavior the
//! host does not implement, which fails with status 2 naming the field. The SPI fields are accepted at the value a new
//! descriptor holds on macOS 27.0.1 (Apple M4) when that value means the feature is off; their 21H461 defaults are not
//! measured, so a different default there fails loudly naming the field.
//!
//! **Metal's errors.** The checks Metal.framework makes before the driver (a nil vertex function, blending into an
//! integer format: `validateWithDevice:` asserts, measured on Apple M4, macOS 27.0.1) cannot reach the host from a
//! Metal program; a request that has them fails with status 2 saying so. The driver's errors the reference harness
//! measures (unmatched varyings, a vertex function with stage input and no vertex descriptor, a vertex result with
//! rasterization disabled) are detected from the functions' AIR records before compiling and fail as
//! [`super::errors`] describes, since their A10X domain and code are not established.
//!
//! **Compilation.** The pipeline compiles at creation (the compiler takes no draw state for render pipelines yet:
//! no bound resources, no primitive type).
//!
//! **Payload.** The handle's payload is the render path's [`RenderPipeline`] (`crate::render::pipeline`), built by
//! [`RenderPipeline::create`] from the [`RenderPipelineDesc`] [`plan`] makes of the compiled stages: the binding
//! layout with each binding's source, the WebGPU color targets (Metal's blend state as WebGPU's, with the blend
//! constant holding the blend color or its alpha), the depth-stencil format, the sample count and the pass plan. A
//! pipeline without a fragment function but with color attachments gets an empty fragment stage whose targets all have
//! write mask 0, so that its WebGPU pipelines match passes with those attachments; Metal writes no color attachment
//! then. When rasterization is enabled the pipeline's variant for a new render encoder's state ([`creation_key`]) is
//! made at creation, through the render path's variant cache, so a compiler output WebGPU rejects fails the creation;
//! with rasterization disabled the vertex function has no side effects (vertex-stage writes are refused) and draws
//! make no WebGPU pipeline.

use std::rc::Rc;

use airwgsl::{
    BindingResource, BindingSource, ColorAttachment, Compiled, PipelineDesc, PipelineKind, RenderState, ShaderStage,
    ShaderStages, StorageTextureAccess, Tessellation, VertexAttribute, VertexBufferLayout, VertexDescriptor,
};
use webmetal_wire::{Failure, Handle, RenderPipelineDescriptor};

use super::compute::{MUTABILITY_VALUES, function};
use super::errors;
use super::gpu;
use super::registry::{Function, function_type};
use crate::exec::{Table, failure};
use crate::format::handling::{self, Resolution, Unavailable, Usage};
use crate::format::webgpu::{Aspect, HostFeatures, SampleType, WebGpuFormat};
use crate::gpu::Api;
use crate::gpu::idl::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState, ColorWrite, CompareFunction, CullMode,
    FrontFace, PrimitiveTopology, StencilOperation, TextureFormat,
};
use crate::mem::texture::idl_format;
use crate::render::metal::TopologyClass;
use crate::render::pipeline::{
    Binding, BlendConstant, ColorTarget, DepthStencilKey, GroupDesc, MetalStage, PassPlan, RenderPipeline,
    RenderPipelineDesc, Resource, Source, StageSource, StencilFaceKey, UniformBlock, UniformField, UniformValue,
    VariantKey,
};

/// `maxColorAttachments` of the A10X (device profile; Metal Feature Set Tables, Apple3: 8).
pub const MAX_COLOR_ATTACHMENTS: usize = 8;

/// The empty fragment stage of a pipeline without a fragment function.
pub const EMPTY_FRAGMENT: &str = "@fragment\nfn aw_main() {\n}\n";

/// What creation keeps of a render descriptor besides the functions.
#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub label: String,
    pub vertex: Handle,
    pub fragment: Option<Handle>,
    pub state: RenderState,
    pub colors: Vec<ColorTarget>,
    pub depth_format: u32,
    pub stencil_format: u32,
    pub depth_stencil_format: Option<TextureFormat>,
    pub sample_count: u32,
    pub sample_mask: u32,
    pub blend_constant: BlendConstant,
    pub support_indirect_command_buffers: bool,
    /// The fragment function writes the depth (`[[depth(...)]]`).
    pub writes_depth: bool,
}

/// `MTLPrimitiveTopologyClass` values: Unspecified, Point, Line, Triangle.
pub const TOPOLOGY_CLASSES: [u64; 4] = [0, 1, 2, 3];

/// The WebGPU blend factor of an `MTLBlendFactor` (iPhoneOS SDK `MTLRenderPipeline.h`). The blend color and its
/// alpha are both WebGPU's `"constant"`: the pipeline's [`BlendConstant`] says which of them the constant holds, and
/// in an alpha slot either one is the constant's alpha.
fn blend_factor(f: u64) -> Option<BlendFactor> {
    Some(match f {
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
        _ => return None,
    })
}

/// The WebGPU blend operation of an `MTLBlendOperation`.
fn blend_operation(o: u64) -> Option<BlendOperation> {
    Some(match o {
        0 => BlendOperation::Add,
        1 => BlendOperation::Subtract,
        2 => BlendOperation::ReverseSubtract,
        3 => BlendOperation::Min,
        4 => BlendOperation::Max,
        _ => return None,
    })
}

/// The WebGPU write mask of an `MTLColorWriteMask` (Red 0x8, Green 0x4, Blue 0x2, Alpha 0x1).
pub fn write_mask(m: u64) -> Option<u32> {
    if m > 0xf {
        return None;
    }
    let mut out = 0;
    for (metal, webgpu) in
        [(0x8, ColorWrite::RED), (0x4, ColorWrite::GREEN), (0x2, ColorWrite::BLUE), (0x1, ColorWrite::ALPHA)]
    {
        if m & metal != 0 {
            out |= webgpu;
        }
    }
    Some(out)
}

/// The blend component of one Metal operation and factor pair. `MTLBlendOperationMin` and `Max` ignore the factors
/// (the reference harness's blend model, checked against real Metal: `tests/reference/src/cases/Blend.m`), and WebGPU
/// requires both factors to be `"one"` for them.
fn component(call: &str, i: usize, op: u64, src: u64, dst: u64) -> Result<BlendComponent, Failure> {
    let bad =
        |what: &str, v: u64| failure::invalid(format!("{call}: colorAttachments[{i}] {what} {v} is not a Metal value"));
    let operation = blend_operation(op).ok_or_else(|| bad("blend operation", op))?;
    let mut c = BlendComponent::new();
    c.operation = operation;
    if matches!(operation, BlendOperation::Min | BlendOperation::Max) {
        c.src_factor = BlendFactor::One;
        c.dst_factor = BlendFactor::One;
    } else {
        c.src_factor = blend_factor(src).ok_or_else(|| bad("blend factor", src))?;
        c.dst_factor = blend_factor(dst).ok_or_else(|| bad("blend factor", dst))?;
    }
    Ok(c)
}

/// The format of an attachment as this host provides it, for a render target of `sample_count` samples.
fn attachment_format(
    call: &str,
    what: &str,
    value: u64,
    sample_count: u32,
    features: HostFeatures,
) -> Result<Resolution, Failure> {
    let v = u32::try_from(value)
        .map_err(|_| failure::invalid(format!("{call}: {what} {value} is not an MTLPixelFormat")))?;
    let r = match handling::resolve(v, features) {
        Ok(r) => r,
        Err(Unavailable::NotOnDevice(why)) => {
            return Err(failure::unimplemented(format!(
                "{call}: {what} {value} is not a pixel format of the A10X ({why}); Metal's result is not established"
            )));
        }
        Err(Unavailable::Refused(r)) => {
            return Err(failure::host_unsupported(format!(
                "{call}: {what} {value}: {}: {} ({})",
                r.id, r.reason, r.source
            )));
        }
    };
    let usage = Usage { shader_read: false, shader_write: false, render_target: true, sample_count };
    handling::check_usage(r, usage, features).map_err(|x| {
        failure::host_unsupported(format!("{call}: {what} {value}: {}: {} ({})", x.id, x.reason, x.source))
    })?;
    Ok(r)
}

/// Check every field of a render pipeline descriptor (module documentation) and keep what creation uses.
pub fn check_descriptor(
    call: &str,
    d: &RenderPipelineDescriptor,
    features: HostFeatures,
) -> Result<RenderSettings, Failure> {
    let RenderPipelineDescriptor {
        alpha_test_enabled,
        // The comparison of the alpha test, which is off.
        alpha_test_function: _,
        alpha_to_coverage_enabled,
        alpha_to_one_enabled,
        // Looked up by the caller under MTLPipelineOptionFailOnBinaryArchiveMiss.
        binary_archives: _,
        // Which clip distances the vertex function's [[clip_distance]] outputs enable; checked against the function
        // (check_functions).
        clip_distance_enable_mask: _,
        color_attachments,
        color_sample_count,
        depth_attachment_pixel_format,
        depth_stencil_write_disabled,
        driver_compiler_options,
        explicit_visibility_group_id,
        force_resource_index,
        force_software_vertex_fetch,
        fragment_buffers,
        fragment_depth_compare_clamp_mask,
        fragment_function,
        fragment_linked_functions,
        fragment_preloaded_libraries,
        gpu_compiler_spi_options,
        input_primitive_topology,
        insert_fragment_libraries,
        insert_mesh_libraries,
        insert_object_libraries,
        insert_vertex_libraries,
        label,
        logic_operation: _,
        logic_operation_enabled,
        // Bounds of ray traversal and of call stacks through function pointers; with neither (both refused by the
        // compiler) they cannot change a result.
        max_acceleration_structure_traversal_depth: _,
        max_fragment_call_stack_depth: _,
        max_mesh_call_stack_depth: _,
        max_object_call_stack_depth: _,
        max_pipeline_children,
        // The tessellation state, read when the vertex function is a post-tessellation function (check_functions).
        max_tessellation_factor: _,
        // Mesh and object state, used with mesh functions only (refused below).
        max_total_threads_per_mesh_threadgroup: _,
        max_total_threads_per_object_threadgroup: _,
        max_vertex_amplification_count,
        max_vertex_call_stack_depth: _,
        mesh_function,
        mesh_linked_functions,
        mesh_preloaded_libraries,
        mesh_threadgroup_size_is_multiple_of_thread_execution_width: _,
        mesh_threads_per_threadgroup: _,
        needs_custom_border_color_samplers,
        object_function,
        object_linked_functions,
        object_preloaded_libraries,
        object_threadgroup_size_is_multiple_of_thread_execution_width: _,
        object_threads_per_threadgroup: _,
        open_gl_mode_enabled,
        pipeline_library,
        pipeline_memory_length,
        plugin_data,
        point_coord_lower_left,
        point_size_output_vs,
        point_smooth_enabled,
        post_vertex_dump_buffer_index,
        profile_control,
        raster_sample_count,
        rasterization_enabled,
        resource_index,
        sample_count,
        sample_coverage,
        sample_coverage_invert,
        sample_mask,
        stencil_attachment_pixel_format,
        support_adding_fragment_binary_functions,
        support_adding_mesh_binary_functions,
        support_adding_object_binary_functions,
        support_adding_vertex_binary_functions,
        support_indirect_command_buffers,
        tessellation_control_point_index_type: _,
        tessellation_factor_format: _,
        tessellation_factor_scale_enabled: _,
        tessellation_factor_step_function: _,
        tessellation_output_winding_order: _,
        tessellation_partition_mode: _,
        texture_write_rounding_mode,
        two_side_enabled,
        vertex_amplification_mode,
        vertex_buffers,
        vertex_depth_compare_clamp_mask,
        vertex_descriptor,
        vertex_function,
        vertex_linked_functions,
        vertex_preloaded_libraries,
        mesh_buffers,
        object_buffers,
    } = d;
    for (name, list) in [
        ("fragmentPreloadedLibraries", fragment_preloaded_libraries),
        ("insertFragmentLibraries", insert_fragment_libraries),
        ("insertMeshLibraries", insert_mesh_libraries),
        ("insertObjectLibraries", insert_object_libraries),
        ("insertVertexLibraries", insert_vertex_libraries),
        ("meshPreloadedLibraries", mesh_preloaded_libraries),
        ("objectPreloadedLibraries", object_preloaded_libraries),
        ("vertexPreloadedLibraries", vertex_preloaded_libraries),
    ] {
        if !list.is_empty() {
            return Err(failure::invalid(format!("{call}: {name} names dynamic libraries, which no record creates")));
        }
    }
    for (name, list) in [
        ("vertexBuffers", vertex_buffers),
        ("fragmentBuffers", fragment_buffers),
        ("meshBuffers", mesh_buffers),
        ("objectBuffers", object_buffers),
    ] {
        for (i, b) in list.iter().enumerate() {
            if !MUTABILITY_VALUES.contains(&b.mutability) {
                return Err(failure::invalid(format!(
                    "{call}: {name}[{i}].mutability {} is not an MTLMutability",
                    b.mutability
                )));
            }
        }
    }
    let unknown = "the behavior of a value other than a new descriptor's (measured on macOS 27.0.1) is not established";
    let flags: [(&str, bool); 17] = [
        ("alphaTestEnabled", *alpha_test_enabled),
        ("depthStencilWriteDisabled", *depth_stencil_write_disabled),
        ("forceResourceIndex", *force_resource_index),
        ("forceSoftwareVertexFetch", *force_software_vertex_fetch),
        ("logicOperationEnabled", *logic_operation_enabled),
        ("needsCustomBorderColorSamplers", *needs_custom_border_color_samplers),
        ("openGLModeEnabled", *open_gl_mode_enabled),
        ("pipelineLibrary", *pipeline_library),
        ("pointCoordLowerLeft", *point_coord_lower_left),
        ("pointSizeOutputVS", *point_size_output_vs),
        ("pointSmoothEnabled", *point_smooth_enabled),
        ("sampleCoverageInvert", *sample_coverage_invert),
        ("supportAddingFragmentBinaryFunctions", *support_adding_fragment_binary_functions),
        ("supportAddingMeshBinaryFunctions", *support_adding_mesh_binary_functions),
        ("supportAddingObjectBinaryFunctions", *support_adding_object_binary_functions),
        ("supportAddingVertexBinaryFunctions", *support_adding_vertex_binary_functions),
        ("twoSideEnabled", *two_side_enabled),
    ];
    if let Some((name, _)) = flags.iter().find(|(_, on)| *on) {
        return Err(errors::field(call, name, "YES", unknown));
    }
    if driver_compiler_options.is_some() {
        return Err(errors::field(call, "driverCompilerOptions", "a dictionary", unknown));
    }
    if gpu_compiler_spi_options.is_some() {
        return Err(errors::field(call, "gpuCompilerSPIOptions", "a dictionary", unknown));
    }
    if plugin_data.is_some() {
        return Err(errors::field(call, "pluginData", "a dictionary", unknown));
    }
    if profile_control.is_some() {
        return Err(errors::field(call, "profileControl", "an MTLProfileControl", unknown));
    }
    let numbers: [(&str, u64, u64); 9] = [
        ("explicitVisibilityGroupID", u64::from(*explicit_visibility_group_id), 0xffff_ffff),
        ("fragmentDepthCompareClampMask", u64::from(*fragment_depth_compare_clamp_mask), 0),
        ("vertexDepthCompareClampMask", u64::from(*vertex_depth_compare_clamp_mask), 0),
        ("pipelineMemoryLength", *pipeline_memory_length, 0),
        ("postVertexDumpBufferIndex", *post_vertex_dump_buffer_index, 0),
        ("textureWriteRoundingMode", *texture_write_rounding_mode as u64, 0),
        ("vertexAmplificationMode", *vertex_amplification_mode, 0),
        ("colorSampleCount", *color_sample_count, 0),
        ("resourceIndex", if *force_resource_index { *resource_index } else { 0 }, 0),
    ];
    if let Some((name, v, _)) = numbers.iter().find(|(_, v, default)| v != default) {
        return Err(errors::field(call, name, v, unknown));
    }
    if (max_pipeline_children.width, max_pipeline_children.height, max_pipeline_children.depth) != (0, 0, 0) {
        return Err(errors::field(call, "maxPipelineChildren", format!("{max_pipeline_children:?}"), unknown));
    }
    if sample_coverage.to_bits() != 1.0f32.to_bits() {
        return Err(errors::field(call, "sampleCoverage", sample_coverage, unknown));
    }
    if mesh_function.is_some() || object_function.is_some() {
        return Err(failure::unimplemented(format!(
            "{call}: a mesh or object function: mesh pipelines are not a feature of the A10X's family and the driver's \
             result is not established"
        )));
    }
    for (name, l) in [
        ("vertexLinkedFunctions", vertex_linked_functions),
        ("fragmentLinkedFunctions", fragment_linked_functions),
        ("meshLinkedFunctions", mesh_linked_functions),
        ("objectLinkedFunctions", object_linked_functions),
    ] {
        if let Some(l) = l {
            let n = l.binary_functions.len() + l.functions.len() + l.groups.len() + l.private_functions.len();
            if n > 0 {
                return Err(errors::field(call, name, format!("{n} entries"), "linked functions are not implemented"));
            }
        }
    }
    let vertex = vertex_function.ok_or_else(|| errors::framework_asserts(call, "vertexFunction is nil"))?;
    // rasterSampleCount and sampleCount are one property (sampleCount is its older name); both are carried.
    if raster_sample_count != sample_count {
        return Err(failure::invalid(format!(
            "{call}: rasterSampleCount {raster_sample_count} and sampleCount {sample_count} differ; they are one property"
        )));
    }
    let samples = match *raster_sample_count {
        1 => 1u32,
        4 => 4,
        2 => {
            return Err(failure::host_unsupported(format!(
                "{call}: rasterSampleCount 2: msaa-2x: WebGPU multisample count must be 1 or 4 (tables/refusals.md)"
            )));
        }
        n => {
            return Err(failure::unimplemented(format!(
                "{call}: rasterSampleCount {n}: the A10X supports 1, 2 and 4 samples; Metal's result is not established"
            )));
        }
    };
    // The sampleMask SPI: a sample whose bit is clear is not written. The render path's pipelines keep WebGPU's
    // default mask (every sample), which is this mask exactly when it has every sample's bit.
    let all_samples = (1u64 << samples) - 1;
    if sample_mask & all_samples != all_samples {
        return Err(errors::field(
            call,
            "sampleMask",
            format!("{sample_mask:#x}"),
            "a mask that clears samples of the raster sample count is not applied by the render path",
        ));
    }
    if *alpha_to_one_enabled {
        return Err(failure::host_unsupported(format!(
            "{call}: alphaToOneEnabled: alpha-to-one: GPUMultisampleState has count, mask and alphaToCoverageEnabled \
             only (tables/refusals.md)"
        )));
    }
    if !TOPOLOGY_CLASSES.contains(input_primitive_topology) {
        return Err(failure::invalid(format!(
            "{call}: inputPrimitiveTopology {input_primitive_topology} is not an MTLPrimitiveTopologyClass"
        )));
    }

    // Color attachments: the blend constant first, as every attachment's factors depend on it.
    if color_attachments.len() > MAX_COLOR_ATTACHMENTS {
        return Err(failure::invalid(format!(
            "{call}: {} color attachments; the A10X has {MAX_COLOR_ATTACHMENTS}",
            color_attachments.len()
        )));
    }
    // The blend constant holds the blend color unless the RGB factors name only its alpha (BlendAlpha 13,
    // OneMinusBlendAlpha 14), when it holds the alpha in every component; RGB factors naming both the color (11, 12) and
    // the alpha have no WebGPU form. In the alpha slots both are the constant's alpha. Min and Max (3, 4) ignore their
    // factors.
    let rgb_factors: Vec<u64> = color_attachments
        .iter()
        .filter(|c| c.pixel_format != 0 && c.blending_enabled && !matches!(c.rgb_blend_operation, 3 | 4))
        .flat_map(|c| [c.source_rgb_blend_factor, c.destination_rgb_blend_factor])
        .collect();
    let names_color = rgb_factors.iter().any(|f| matches!(f, 11 | 12));
    let names_alpha = rgb_factors.iter().any(|f| matches!(f, 13 | 14));
    let blend_constant = match (names_color, names_alpha) {
        (true, true) => {
            return Err(failure::unimplemented(format!(
                "{call}: RGB blend factors name both the blend color and the blend alpha; WebGPU has one blend \
                 constant, which cannot hold both"
            )));
        }
        (false, true) => BlendConstant::AlphaSplat,
        _ => BlendConstant::Color,
    };
    let mut colors = Vec::new();
    let last = color_attachments.iter().rposition(|c| c.pixel_format != 0).map_or(0, |i| i + 1);
    for (i, c) in color_attachments.iter().take(last).enumerate() {
        if c.pixel_format == 0 {
            colors.push(ColorTarget { pixel_format: 0, state: None });
            continue;
        }
        let what = format!("colorAttachments[{i}].pixelFormat");
        let r = attachment_format(call, &what, c.pixel_format, samples, features)?;
        if let Resolution::Emulated(e) = r
            && !matches!(e.write, handling::Write::Native)
        {
            return Err(failure::unimplemented(format!(
                "{call}: {what} {}: rendering into it needs a write conversion in the fragment shader ({:?}), which the \
                 compiler does not emit",
                c.pixel_format, e.write
            )));
        }
        let storage = r.storage();
        let mut t = ColorTargetState::new(idl_format(storage));
        t.write_mask = write_mask(c.write_mask).ok_or_else(|| {
            failure::invalid(format!(
                "{call}: colorAttachments[{i}].writeMask {} is not an MTLColorWriteMask",
                c.write_mask
            ))
        })?;
        if c.blending_enabled {
            if matches!(storage.caps().sample, SampleType::Uint | SampleType::Sint) {
                return Err(errors::framework_asserts(
                    call,
                    format!(
                        "blending is enabled for color attachment {i}, whose pixel format {} is an integer format",
                        c.pixel_format
                    ),
                ));
            }
            handling::check_blending(r, features).map_err(|x| {
                failure::host_unsupported(format!(
                    "{call}: blending into {what} {}: {}: {} ({})",
                    c.pixel_format, x.id, x.reason, x.source
                ))
            })?;
            t.blend = Some(BlendState::new(
                component(call, i, c.rgb_blend_operation, c.source_rgb_blend_factor, c.destination_rgb_blend_factor)?,
                component(
                    call,
                    i,
                    c.alpha_blend_operation,
                    c.source_alpha_blend_factor,
                    c.destination_alpha_blend_factor,
                )?,
            ));
        }
        colors.push(ColorTarget { pixel_format: c.pixel_format as u32, state: Some(t) });
    }

    // Depth and stencil: one WebGPU attachment holds both; when both are set Metal requires one combined format.
    let (depth, stencil) = (*depth_attachment_pixel_format, *stencil_attachment_pixel_format);
    let depth_stencil_format = match (depth, stencil) {
        (0, 0) => None,
        (d, s) if d != 0 && s != 0 && d != s => {
            return Err(errors::framework_asserts(
                call,
                format!("depthAttachmentPixelFormat {d} and stencilAttachmentPixelFormat {s} differ"),
            ));
        }
        (d, s) => {
            let v = if d != 0 { d } else { s };
            let r = attachment_format(call, "the depth-stencil pixel format", v, samples, features)?;
            let f = r.storage();
            let aspects = f.caps().aspects;
            if aspects.is_empty() {
                return Err(errors::framework_asserts(
                    call,
                    format!("pixel format {v} is not a depth or stencil format"),
                ));
            }
            Some(idl_format(f))
        }
    };

    let state = RenderState {
        vertex_descriptor: vertex_descriptor.as_ref().map(|v| VertexDescriptor {
            layouts: v
                .layouts
                .iter()
                .map(|l| VertexBufferLayout {
                    stride: l.stride,
                    step_function: l.step_function,
                    step_rate: l.step_rate,
                })
                .collect(),
            attributes: v
                .attributes
                .iter()
                .map(|a| VertexAttribute { format: a.format, offset: a.offset, buffer_index: a.buffer_index })
                .collect(),
        }),
        color_attachments: color_attachments
            .iter()
            .map(|c| ColorAttachment {
                pixel_format: c.pixel_format,
                blending_enabled: c.blending_enabled,
                rgb_blend_operation: c.rgb_blend_operation,
                alpha_blend_operation: c.alpha_blend_operation,
                source_rgb_blend_factor: c.source_rgb_blend_factor,
                destination_rgb_blend_factor: c.destination_rgb_blend_factor,
                source_alpha_blend_factor: c.source_alpha_blend_factor,
                destination_alpha_blend_factor: c.destination_alpha_blend_factor,
                write_mask: c.write_mask,
            })
            .collect(),
        depth_attachment_pixel_format: depth,
        stencil_attachment_pixel_format: stencil,
        raster_sample_count: *raster_sample_count,
        alpha_to_coverage_enabled: *alpha_to_coverage_enabled,
        alpha_to_one_enabled: false,
        rasterization_enabled: *rasterization_enabled,
        input_primitive_topology: *input_primitive_topology,
        primitive_type: None,
        // Set when the vertex function is a post-tessellation function (check_functions reads its patch record).
        tessellation: None,
        max_vertex_amplification_count: *max_vertex_amplification_count,
    };
    Ok(RenderSettings {
        label: label.clone().unwrap_or_default(),
        vertex,
        fragment: *fragment_function,
        state,
        colors,
        depth_format: depth as u32,
        stencil_format: stencil as u32,
        depth_stencil_format,
        sample_count: samples,
        sample_mask: (*sample_mask & 0xffff_ffff) as u32,
        blend_constant,
        support_indirect_command_buffers: *support_indirect_command_buffers,
        // Set from the fragment function (check_functions).
        writes_depth: false,
    })
}

/// The tessellation state of a descriptor, for a pipeline whose vertex function is a post-tessellation function.
pub fn tessellation_of(d: &RenderPipelineDescriptor) -> Tessellation {
    Tessellation {
        partition_mode: d.tessellation_partition_mode,
        max_tessellation_factor: d.max_tessellation_factor,
        factor_scale_enabled: d.tessellation_factor_scale_enabled,
        factor_format: d.tessellation_factor_format,
        control_point_index_type: d.tessellation_control_point_index_type,
        factor_step_function: d.tessellation_factor_step_function,
        output_winding_order: d.tessellation_output_winding_order,
    }
}

/// The checks Metal's driver makes on the functions of a render pipeline (as measured; [`super::errors`]), and the
/// color attachments the fragment function leaves unwritten (write mask 0).
pub fn check_functions(
    call: &str,
    settings: &mut RenderSettings,
    d: &RenderPipelineDescriptor,
    vertex: &Function,
    fragment: Option<&Function>,
) -> Result<(), Failure> {
    if vertex.function_type != function_type::VERTEX {
        return Err(failure::unimplemented(format!(
            "{call}: vertexFunction is a {}; Metal's result is not established",
            vertex.describe()
        )));
    }
    if let Some(f) = fragment
        && f.function_type != function_type::FRAGMENT
    {
        return Err(failure::unimplemented(format!(
            "{call}: fragmentFunction is a {}; Metal's result is not established",
            f.describe()
        )));
    }
    let parse = |f: &Function| -> Result<airwgsl::air::AirModule, Failure> {
        super::compute::decode_module(f).map_err(|r| errors::refusal(call, &r))
    };
    let vm = parse(vertex)?;
    let ve = vm.entry_point(&vertex.name).ok_or_else(|| {
        errors::refusal(
            call,
            &airwgsl::wgsl::tracked::refuse(
                "entry-metadata",
                vertex.name.clone(),
                None,
                "the function is not an entry point of the module",
            ),
        )
    })?;
    if ve.patch.is_some() {
        settings.state.tessellation = Some(tessellation_of(d));
    }
    use airwgsl::air::{OutputRole, Role};
    if ve.arguments.iter().any(|a| matches!(a.role, Role::VertexInput(_))) && settings.state.vertex_descriptor.is_none()
    {
        return Err(errors::driver_error(call, errors::STAGE_IN_WITHOUT_DESCRIPTOR, vertex.describe()));
    }
    if !settings.state.rasterization_enabled && !ve.outputs.is_empty() {
        return Err(errors::driver_error(call, errors::VERTEX_RESULT_WITHOUT_RASTERIZATION, vertex.describe()));
    }
    if ve.outputs.iter().any(|o| matches!(o.role, OutputRole::ClipDistance(_))) && d.clip_distance_enable_mask != 0xff {
        return Err(errors::field(
            call,
            "clipDistanceEnableMask",
            d.clip_distance_enable_mask,
            "a vertex function with clip distances and a mask other than a new descriptor's (0xff on macOS 27.0.1): \
             which distances the A10X then clips against is not established",
        ));
    }
    let Some(fragment) = fragment else { return Ok(()) };
    let fm = parse(fragment)?;
    let fe = fm.entry_point(&fragment.name).ok_or_else(|| {
        errors::refusal(
            call,
            &airwgsl::wgsl::tracked::refuse(
                "entry-metadata",
                fragment.name.clone(),
                None,
                "the function is not an entry point of the module",
            ),
        )
    })?;
    // Varyings: every fragment input the vertex function must write, with the same type.
    let outputs: Vec<(&str, Option<&str>)> = ve
        .outputs
        .iter()
        .filter_map(|o| match &o.role {
            OutputRole::VertexOutput(n) => Some((n.as_str(), o.type_name.as_deref())),
            _ => None,
        })
        .collect();
    let mut unmatched = Vec::new();
    for a in &fe.arguments {
        if let Role::FragmentInput { name, .. } = &a.role
            && !outputs.iter().any(|(n, t)| *n == name.as_str() && *t == a.type_name.as_deref())
        {
            unmatched.push(name.clone());
        }
    }
    if !unmatched.is_empty() {
        return Err(errors::driver_error(
            call,
            errors::VARYING_MISMATCH,
            format!("fragment inputs {}", unmatched.join(", ")),
        ));
    }
    settings.writes_depth = fe.outputs.iter().any(|o| matches!(o.role, OutputRole::Depth(_)));
    // The color attachments the fragment function writes. Whether each output suits its attachment's format (class,
    // components, an alpha the blend factors read) is the compiler's check, with Metal's measured rule.
    let mut written = vec![false; settings.colors.len()];
    for o in &fe.outputs {
        let OutputRole::RenderTarget { index, blend_index: 0 } = &o.role else { continue };
        let airwgsl::air::Slot::Value(index) = index else {
            return Err(failure::unimplemented(format!("{call}: a color output whose index is a function constant")));
        };
        if let Some(w) = written.get_mut(*index as usize) {
            *w = true;
        }
    }
    // An attachment the function does not write keeps its contents: Metal leaves it undefined, and WebGPU requires a
    // shader output for a target with a non-zero write mask.
    for (i, w) in written.iter().enumerate() {
        if !w && let Some(t) = settings.colors[i].state.as_mut() {
            t.write_mask = 0;
        }
    }
    Ok(())
}

/// Every attachment target with write mask 0, for a pipeline without a fragment function.
pub fn mask_all(settings: &mut RenderSettings) {
    for c in &mut settings.colors {
        if let Some(t) = c.state.as_mut() {
            t.write_mask = 0;
        }
    }
}

/// The pipeline description the compiler takes.
pub fn desc(settings: &RenderSettings) -> PipelineDesc {
    PipelineDesc { kind: PipelineKind::Render(settings.state.clone()), bindings: Vec::new() }
}

/// Refuse writable bindings the vertex stage sees: WebGPU allows only uniform and read-only bindings there.
pub fn check_vertex_writes(call: &str, c: &Compiled) -> Result<(), Failure> {
    for g in &c.layout.groups {
        for e in &g.entries {
            let writable = match &e.resource {
                BindingResource::Buffer { ty, .. } => *ty == airwgsl::BufferBindingType::Storage,
                BindingResource::StorageTexture { access, .. } => *access != StorageTextureAccess::ReadOnly,
                _ => false,
            };
            if writable && e.visibility.contains(ShaderStages::VERTEX) {
                return Err(failure::host_unsupported(format!(
                    "{call}: the vertex function writes {:?}: vertex-stage-writes: bindings visible to the vertex stage \
                     must be uniform or read-only (tables/refusals.md)",
                    e.source
                )));
            }
        }
    }
    Ok(())
}

/// The WGSL stages of a compiled render pipeline: the vertex stage, and the fragment stage when there is one.
pub fn stages(call: &str, c: &Compiled) -> Result<(StageSource, Option<StageSource>), Failure> {
    let mut vertex = None;
    let mut fragment = None;
    for s in &c.wgsl {
        let src = StageSource { wgsl: s.source.clone(), entry_point: s.entry_point.clone(), constants: Vec::new() };
        match s.stage {
            ShaderStage::Vertex if vertex.is_none() => vertex = Some(src),
            ShaderStage::Fragment if fragment.is_none() => fragment = Some(src),
            other => {
                return Err(failure::unimplemented(format!(
                    "{call}: the compiler produced an unexpected {other:?} stage"
                )));
            }
        }
    }
    let vertex =
        vertex.ok_or_else(|| failure::unimplemented(format!("{call}: the compiler produced no vertex stage")))?;
    Ok((vertex, fragment))
}

/// The render path's stage of a compiler stage.
fn stage(call: &str, s: airwgsl::MetalStage) -> Result<MetalStage, Failure> {
    match s {
        airwgsl::MetalStage::Vertex => Ok(MetalStage::Vertex),
        airwgsl::MetalStage::Fragment => Ok(MetalStage::Fragment),
        other => Err(failure::unimplemented(format!("{call}: a render pipeline binding of the {other:?} stage"))),
    }
}

/// The render path's form of one compiled binding.
fn binding(call: &str, e: &airwgsl::BindingEntry, reflection: &airwgsl::Reflection) -> Result<Binding, Failure> {
    let what = |x: String| failure::unimplemented(format!("{call}: binding {}: {x}", e.binding));
    let resource = match &e.resource {
        BindingResource::Buffer { ty, has_dynamic_offset, min_binding_size } => {
            if *has_dynamic_offset {
                return Err(what("a buffer with a dynamic offset, which the render encoder does not bind".into()));
            }
            let ty = match ty {
                airwgsl::BufferBindingType::Uniform => crate::gpu::idl::BufferBindingType::Uniform,
                airwgsl::BufferBindingType::Storage => crate::gpu::idl::BufferBindingType::Storage,
                airwgsl::BufferBindingType::ReadOnlyStorage => crate::gpu::idl::BufferBindingType::ReadOnlyStorage,
            };
            Resource::Buffer { ty, min_binding_size: *min_binding_size }
        }
        BindingResource::Sampler { ty } => Resource::Sampler {
            ty: match ty {
                airwgsl::SamplerBindingType::Filtering => crate::gpu::idl::SamplerBindingType::Filtering,
                airwgsl::SamplerBindingType::NonFiltering => crate::gpu::idl::SamplerBindingType::NonFiltering,
                airwgsl::SamplerBindingType::Comparison => crate::gpu::idl::SamplerBindingType::Comparison,
            },
        },
        BindingResource::Texture { sample_type, view_dimension, multisampled } => Resource::Texture {
            sample_type: gpu::sample_type(*sample_type),
            view_dimension: gpu::view_dimension(*view_dimension),
            multisampled: *multisampled,
        },
        BindingResource::StorageTexture { access, format, view_dimension } => Resource::StorageTexture {
            access: gpu::storage_access(*access),
            format: gpu::texture_format(format)?,
            view_dimension: gpu::view_dimension(*view_dimension),
        },
    };
    let source = match &e.source {
        BindingSource::MetalBuffer { stage: s, index } => {
            Source::MetalBuffer { stage: stage(call, *s)?, index: *index }
        }
        BindingSource::MetalTexture { stage: s, index } => {
            Source::MetalTexture { stage: stage(call, *s)?, index: *index }
        }
        BindingSource::MetalTextureLevel { stage: s, index, level } => {
            Source::MetalTextureLevel { stage: stage(call, *s)?, index: *index, level: *level }
        }
        BindingSource::MetalSampler { stage: s, index } => {
            Source::MetalSampler { stage: stage(call, *s)?, index: *index }
        }
        // Framebuffer fetch: the snapshot the render path takes before each group of primitives (plan section 5.9).
        BindingSource::FetchSnapshot { attachment } => Source::FetchSnapshot { attachment: *attachment },
        BindingSource::Uniforms(block) => {
            let mut fields = Vec::new();
            for f in &block.fields {
                let value = match &f.value {
                    airwgsl::UniformValue::BaseVertex => UniformValue::BaseVertex,
                    airwgsl::UniformValue::BaseInstance => UniformValue::BaseInstance,
                    airwgsl::UniformValue::ResidualOffset { stage: s, index } => {
                        UniformValue::ResidualOffset { stage: stage(call, *s)?, index: *index }
                    }
                    airwgsl::UniformValue::BufferLength { stage: s, index } => {
                        UniformValue::BufferLength { stage: stage(call, *s)?, index: *index }
                    }
                    other => return Err(what(format!("a uniform field holding {other:?}"))),
                };
                fields.push(UniformField { offset: f.offset, value });
            }
            Source::Uniforms(UniformBlock { fields, size: block.size })
        }
        other => return Err(what(format!("the source {other:?}"))),
    };
    // The function's texture argument the binding stands for (vertex functions reflect as type 1, fragment as 2).
    let texture = match &source {
        Source::MetalTexture { stage, index } | Source::MetalTextureLevel { stage, index, .. } => {
            let function_type = if *stage == MetalStage::Vertex { 1 } else { 2 };
            reflection
                .stages
                .iter()
                .find(|s| s.function_type == function_type)
                .map(super::bound::texture_arguments)
                .and_then(|args| super::bound::argument_at(&args, *index).cloned())
        }
        _ => None,
    };
    Ok(Binding { binding: e.binding, visibility: e.visibility.0, resource, source, texture })
}

/// The render path's description of a compiled render pipeline with `settings`.
pub fn plan(call: &str, label: &str, settings: &RenderSettings, c: &Compiled) -> Result<RenderPipelineDesc, Failure> {
    let (vertex, fragment) = stages(call, c)?;
    let has_targets = settings.colors.iter().any(|t| t.state.is_some());
    let mut colors = settings.colors.clone();
    let fragment = match fragment {
        Some(f) => Some(f),
        None if has_targets => {
            // No fragment function: nothing is written, so the targets are not blended either.
            for t in colors.iter_mut().filter_map(|t| t.state.as_mut()) {
                t.write_mask = 0;
                t.blend = None;
            }
            Some(StageSource { wgsl: EMPTY_FRAGMENT.to_string(), entry_point: "aw_main".into(), constants: Vec::new() })
        }
        None => None,
    };
    let mut groups = Vec::new();
    for g in &c.layout.groups {
        groups.push(GroupDesc {
            index: g.group,
            bindings: g.entries.iter().map(|e| binding(call, e, &c.reflection)).collect::<Result<Vec<_>, _>>()?,
        });
    }
    let passes = match &c.passes[..] {
        [] | [airwgsl::PassPlan::Direct] => PassPlan::Direct,
        // Each primitive in a pass of its own is exact for any geometry; the compiler's overlap prover only allows
        // fewer passes, which the render path does not take yet.
        [airwgsl::PassPlan::PerPrimitive { fetched, .. }] => PassPlan::PerPrimitive { fetched: fetched.clone() },
        other => return Err(failure::unimplemented(format!("{call}: the pass plans {other:?}"))),
    };
    Ok(RenderPipelineDesc {
        label: label.to_string(),
        vertex,
        fragment,
        groups,
        // The compiler pulls vertices through bindings; it describes no native vertex input.
        vertex_buffers: Vec::new(),
        colors,
        depth_format: settings.depth_format,
        stencil_format: settings.stencil_format,
        depth_stencil_format: settings.depth_stencil_format,
        sample_count: settings.sample_count,
        alpha_to_coverage: settings.state.alpha_to_coverage_enabled,
        blend_constant: settings.blend_constant,
        topology_class: TopologyClass::from_value(settings.state.input_primitive_topology, call)
            .map_err(|e| e.failure())?,
        rasterization: settings.state.rasterization_enabled,
        passes,
        writes_depth: settings.writes_depth,
    })
}

/// The variant of a new render encoder's state: primitives of the topology class (triangles when unspecified),
/// clockwise front faces, no culling, depth clipping, and a new `MTLDepthStencilDescriptor` (depth compare Always
/// without writes; stencil compare Always, every operation Keep, masks 0xffffffff).
pub fn creation_key(p: &RenderPipeline<impl Api>) -> VariantKey {
    let topology = match p.topology_class {
        TopologyClass::Point => PrimitiveTopology::PointList,
        TopologyClass::Line => PrimitiveTopology::LineList,
        TopologyClass::Triangle | TopologyClass::Unspecified => PrimitiveTopology::TriangleList,
    };
    let face = StencilFaceKey {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let depth_stencil = p.depth_stencil_format.map(|format| {
        let f = WebGpuFormat::ALL.into_iter().find(|f| idl_format(*f) == format).expect("a stored format");
        let depth = f.caps().aspects.iter().any(|a| a.aspect == Aspect::Depth);
        DepthStencilKey {
            format,
            depth: depth.then_some((false, CompareFunction::Always)),
            front: face,
            back: face,
            read_mask: 0xffff_ffff,
            write_mask: 0xffff_ffff,
        }
    });
    VariantKey {
        topology,
        strip_index_format: None,
        front_face: FrontFace::Cw,
        cull_mode: CullMode::None,
        unclipped_depth: false,
        depth_stencil,
        strides: Vec::new(),
    }
}

/// The functions of a render descriptor.
pub fn functions(objects: &Table, s: &RenderSettings) -> Result<(Rc<Function>, Option<Rc<Function>>), Failure> {
    Ok((function(objects, s.vertex)?, s.fragment.map(|f| function(objects, f)).transpose()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_masks_reorder_the_channels() {
        assert_eq!(write_mask(0xf), Some(ColorWrite::ALL));
        assert_eq!(write_mask(0x8), Some(ColorWrite::RED));
        assert_eq!(write_mask(0x1), Some(ColorWrite::ALPHA));
        assert_eq!(write_mask(0x6), Some(ColorWrite::GREEN | ColorWrite::BLUE));
        assert_eq!(write_mask(0x10), None);
    }

    #[test]
    fn blend_color_and_alpha_are_the_constant() {
        for (f, g) in [(11, BlendFactor::Constant), (13, BlendFactor::Constant), (12, BlendFactor::OneMinusConstant)] {
            assert_eq!(blend_factor(f), Some(g));
        }
        assert_eq!(blend_factor(14), Some(BlendFactor::OneMinusConstant));
        assert_eq!(blend_factor(19), None);
        let c = component("t", 0, 3, 4, 5).unwrap();
        assert_eq!(
            (c.operation, c.src_factor, c.dst_factor),
            (BlendOperation::Min, BlendFactor::One, BlendFactor::One)
        );
    }
}
