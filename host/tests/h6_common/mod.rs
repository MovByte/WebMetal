//! The executor with the memory, blit and compute services on a native WebGPU implementation (`WEBGPU_NATIVE`, see
//! `host/build.rs`), and the requests a guest sends for the Metal reference programs.
//!
//! Pipelines come from the pipeline service (`webmetal_host::pipeline`), which compiles the reference kernels of
//! `tests/reference/shaders` from their AIR through AirWgsl; [`Pipelines`] forwards every shader record to it. For a
//! kernel the compiler refuses so far (texture and stage-input arguments), the creation falls back to a hand
//! translation to WGSL (each next to its kernel below), so that the encoder's dispatch, binding and coherence paths
//! still run on those kernels; [`Harness::compiled`] says which kernels came from the compiler.
//!
//! [`ClearPass`] stands in for the render encoder (package H5): it records render passes that only clear their color
//! attachments (the one render pass the blit cases use, in `blit.texture-to-texture-multisample`).

#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;
use std::time::Duration;

use super::mem_common::{Api, Gpu, Guest, TASK};
use super::pipeline_common::{LibraryFunction, reference_library};
use webmetal_host::blit::BlitService;
use webmetal_host::compute::ComputeService;
use webmetal_host::compute::pipeline::{
    BindGroup, Binding, ComputePipelineState, ComputeVariant, ComputeVariants, Resource, Source, UniformBlock,
    UniformValue, VariantKey,
};
use webmetal_host::embed::TaskId;
use webmetal_host::exec::failure;
use webmetal_host::exec::schema::{Role, command_handles};
use webmetal_host::exec::{
    Config, ControlService, EncodeCx, EncoderService, Entry, Executor, Gpu as ExecGpu, RecordFailure, Segment, Served,
    ServiceCx, Services,
};
use webmetal_host::gpu::idl::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferBindingLayout, BufferBindingType, Color, ColorDict,
    ComputePipelineDescriptor as GpuComputePipelineDescriptor, LoadOp, PipelineLayoutDescriptor, PipelineLayoutOrAuto,
    ProgrammableStage, RenderPassColorAttachment as GpuColorAttachment, RenderPassDescriptor as GpuRenderPass,
    SamplerBindingLayout, SamplerBindingType, ShaderModuleDescriptor, ShaderStage, StorageTextureBindingLayout,
    StoreOp, TextureBindingLayout, TextureOrView, TextureSampleType, TextureViewDescriptor, TextureViewDimension,
};
use webmetal_host::gpu::num::Float64;
use webmetal_host::gpu::scope::{Granularity, Policy};
use webmetal_host::gpu::{CommandEncoder as _, Device as _, Gpu as _, RenderPassEncoder as _, Size64, Texture as _};
use webmetal_host::mem::Memory;
use webmetal_host::mem::service::MemoryServices;
use webmetal_host::mem::texture::Texture;
use webmetal_host::pipeline::sha256::sha256;
use webmetal_host::pipeline::{self, PipelineServices};
use webmetal_wire::*;

pub type A = Api;
pub type E = Guest;

/// `MTLStorageMode`.
pub const SHARED: u64 = 0;
pub const PRIVATE: u64 = 2;
/// `MTLTextureType`.
pub const T1D: u64 = 0;
pub const T1D_ARRAY: u64 = 1;
pub const T2D: u64 = 2;
pub const T2D_ARRAY: u64 = 3;
pub const T2D_MS: u64 = 4;
pub const TCUBE: u64 = 5;
pub const T3D: u64 = 7;
/// `MTLTextureUsage`.
pub const SHADER_READ: u64 = 1;
pub const SHADER_WRITE: u64 = 2;
pub const RENDER_TARGET: u64 = 4;

/// The `MTLPixelFormat` value of a format name without its prefix.
pub fn fmt(name: &str) -> u64 {
    let full = format!("MTLPixelFormat{name}");
    u64::from(
        webmetal_host::format::table::METAL_FORMATS
            .iter()
            .find(|r| r.name == full)
            .unwrap_or_else(|| panic!("{name}"))
            .value,
    )
}

/// `WMPattern` of the reference harness (tests/reference/FORMAT.md, "Generated inputs").
pub fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut state = if seed == 0 { 0x9e37_79b9 } else { seed };
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect()
}

pub fn words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn origin(x: u64, y: u64, z: u64) -> Origin {
    Origin { x, y, z }
}

pub fn size(width: u64, height: u64, depth: u64) -> Size {
    Size { width, height, depth }
}

/// An `MTLTextureDescriptor` of the reference programs: storage, usage and the dimensions.
#[allow(clippy::too_many_arguments)]
pub fn texture_descriptor(
    kind: u64,
    format: u64,
    (w, h, d): (u64, u64, u64),
    array_length: u64,
    levels: u64,
    storage: u64,
    usage: u64,
    samples: u64,
) -> TextureDescriptor {
    TextureDescriptor {
        allow_gpu_optimized_contents: true,
        array_length,
        color_space_conversion_matrix: 0,
        compression_footprint: 0,
        compression_mode: 0,
        compression_type: 0,
        cpu_cache_mode: 0,
        depth: d,
        force_resource_index: false,
        framebuffer_only: false,
        hazard_tracking_mode: 0,
        height: h,
        is_drawable: false,
        lossy_compression_mode: 0,
        mipmap_level_count: levels,
        pixel_format: format,
        protection_options: 0,
        resource_index: 0,
        resource_options: storage << 4,
        rotation: 0,
        sample_count: samples,
        sparse_surface_default_value: 0,
        storage_mode: storage,
        swizzle: TextureSwizzleChannels { red: 2, green: 3, blue: 4, alpha: 5 },
        swizzle_key: 0,
        texture_type: kind,
        texture_usage: usage,
        usage,
        width: w,
        write_swizzle_enabled: false,
    }
}

/// `MTLComputePipelineDescriptor` with its defaults and a compute function.
pub fn compute_descriptor(function: Handle) -> ComputePipelineDescriptor {
    ComputePipelineDescriptor {
        binary_archives: Vec::new(),
        binary_libraries: Vec::new(),
        buffers: Vec::new(),
        compute_function: Some(function),
        driver_compiler_options: None,
        dynamic_libraries: Vec::new(),
        force_resource_index: false,
        function_pointers: Vec::new(),
        gpu_compiler_spi_options: None,
        insert_libraries: Vec::new(),
        label: None,
        linked_functions: None,
        max_acceleration_structure_traversal_depth: 0,
        max_call_stack_depth: 1,
        max_total_threads_per_threadgroup: 0,
        needs_custom_border_color_samplers: false,
        open_cl_mode_enabled: false,
        open_gl_mode_enabled: false,
        pipeline_library: false,
        plugin_data: None,
        preloaded_libraries: Vec::new(),
        profile_control: None,
        resource_index: 0,
        stage_input_descriptor: None,
        support_adding_binary_functions: false,
        support_indirect_command_buffers: false,
        texture_write_fp_rounding_mode: 0,
        texture_write_rounding_mode: 0,
        thread_group_size_is_multiple_of_thread_execution_width: false,
    }
}

// The pipeline stand-in ------------------------------------------------------------------------------------------------

/// A reference kernel: its bindings (group 0) and its WGSL for a dispatch.
struct Kernel {
    bindings: Vec<Binding>,
    /// Bytes of `var<workgroup>` the WGSL declares itself.
    static_memory: u64,
    wgsl: fn(&VariantKey) -> Result<String, String>,
}

fn storage(binding: u32, index: u64, write: bool) -> Binding {
    storage_of(binding, index, write, 4)
}

/// A storage buffer of elements of `element` bytes.
fn storage_of(binding: u32, index: u64, write: bool, element: u64) -> Binding {
    let ty = if write { BufferBindingType::Storage } else { BufferBindingType::ReadOnlyStorage };
    Binding { binding, resource: Resource::Buffer { ty, min_binding_size: element }, source: Source::Buffer { index } }
}

fn uniform(binding: u32, index: u64, size: u64) -> Binding {
    Binding {
        binding,
        resource: Resource::Buffer { ty: BufferBindingType::Uniform, min_binding_size: size },
        source: Source::Buffer { index },
    }
}

fn uniforms(binding: u32, fields: Vec<(u64, UniformValue)>, size: u64) -> Binding {
    Binding {
        binding,
        resource: Resource::Buffer { ty: BufferBindingType::Uniform, min_binding_size: size },
        source: Source::Uniforms(UniformBlock { size, fields }),
    }
}

fn sampled(binding: u32, index: u64, sample_type: TextureSampleType, multisampled: bool) -> Binding {
    Binding {
        binding,
        resource: Resource::Texture { sample_type, view_dimension: TextureViewDimension::D2, multisampled },
        source: Source::Texture { index },
    }
}

/// A texture read without a sampler, in a view of `dimension`.
fn read_texture(binding: u32, index: u64, sample_type: TextureSampleType, dimension: TextureViewDimension) -> Binding {
    Binding {
        binding,
        resource: Resource::Texture { sample_type, view_dimension: dimension, multisampled: false },
        source: Source::Texture { index },
    }
}

/// The bindings of the read kernels of textures.metal: the texture, the queries (`device const uint4 *q
/// [[buffer(0)]]`) and the results (`device uint4 *out [[buffer(1)]]`).
fn read_bindings(sample_type: TextureSampleType, dimension: TextureViewDimension) -> Vec<Binding> {
    vec![read_texture(0, 0, sample_type, dimension), storage_of(1, 0, false, 16), storage_of(2, 1, true, 16)]
}

fn sampler(binding: u32, index: u64, ty: SamplerBindingType) -> Binding {
    Binding { binding, resource: Resource::Sampler { ty }, source: Source::Sampler { index } }
}

fn workgroup(key: &VariantKey) -> String {
    let [x, y, z] = key.threads_per_threadgroup;
    format!("@compute @workgroup_size({x}, {y}, {z})")
}

/// The kernels of tests/reference/shaders that the replayed cases dispatch, translated to WGSL.
fn kernel(name: &str) -> Option<Kernel> {
    Some(match name {
        // compute.metal: compute_positions. `constant uint3 &grid` is 16 bytes; the out pointer's offset is applied
        // from the residual-offset uniform.
        "compute_positions" => Kernel {
            bindings: vec![
                storage(0, 0, true),
                uniform(1, 1, 16),
                uniforms(2, vec![(0, UniformValue::ResidualOffset { index: 0 })], 16),
            ],
            static_memory: 0,
            wgsl: |k| {
                let [x, y, z] = k.threads_per_threadgroup;
                Ok(format!(
                    "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> grid: vec4u;
@group(0) @binding(2) var<uniform> u: vec4u;
{wg}
fn main(@builtin(global_invocation_id) tpig: vec3u, @builtin(workgroup_id) tgpig: vec3u,
        @builtin(local_invocation_id) tpitg: vec3u, @builtin(local_invocation_index) tiitg: u32,
        @builtin(num_workgroups) tgpg: vec3u) {{
    let tptg = vec3u({x}u, {y}u, {z}u);
    let tpg = tgpg * tptg;
    let index = (tpig.z * grid.y + tpig.y) * grid.x + tpig.x;
    let r = u.x / 4u + index * 16u;
    out[r + 0u] = tpig.x; out[r + 1u] = tpig.y; out[r + 2u] = tpig.z;
    out[r + 3u] = tgpig.x; out[r + 4u] = tgpig.y; out[r + 5u] = tgpig.z;
    out[r + 6u] = tpitg.x; out[r + 7u] = tpitg.y; out[r + 8u] = tpitg.z;
    out[r + 9u] = tiitg;
    out[r + 10u] = tptg.x; out[r + 11u] = tptg.y; out[r + 12u] = tptg.z;
    out[r + 13u] = tgpg.x * 0x10000u + tgpg.y * 0x100u + tgpg.z;
    out[r + 14u] = tpg.x * 0x10000u + tpg.y * 0x100u + tpg.z;
    out[r + 15u] = 0xc0ffee00u;
}}",
                    wg = workgroup(k)
                ))
            },
        },
        // compute_threadgroup_mirror: `threadgroup uint *scratch [[threadgroup(0)]]` sized by the encoder; the scalar
        // `uint` position attributes are the x components.
        "compute_threadgroup_mirror" => Kernel {
            bindings: vec![storage(0, 0, true)],
            static_memory: 0,
            wgsl: |k| {
                let len = *k.threadgroup_memory_lengths.first().ok_or("threadgroup memory 0 is not set")?;
                let x = k.threads_per_threadgroup[0];
                Ok(format!(
                    "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
var<workgroup> scratch: array<u32, {n}>;
{wg}
fn main(@builtin(local_invocation_index) tiitg: u32, @builtin(workgroup_id) tgpig: vec3u) {{
    scratch[tiitg] = tgpig.x * 1000u + tiitg;
    workgroupBarrier();
    out[tgpig.x * {x}u + tiitg] = scratch[{x}u - 1u - tiitg];
}}",
                    n = len / 4,
                    wg = workgroup(k)
                ))
            },
        },
        // compute_copy_constant: `constant uint *in` (4096 bytes in the case) as a uniform array of 16-byte elements.
        "compute_copy_constant" => Kernel {
            bindings: vec![storage(0, 0, true), uniform(1, 1, 4096)],
            static_memory: 0,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> input: array<vec4u, 256>;
{wg}
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let i = id.x;
    out[i] = input[i / 4u][i % 4u];
}}",
                    wg = workgroup(k)
                ))
            },
        },
        // samplers.metal: smp_sample_2d, over the RGBA32Float texture of the sampler cases, which WebGPU does not filter.
        "smp_sample_2d" => Kernel {
            bindings: vec![
                sampled(0, 0, TextureSampleType::UnfilterableFloat, false),
                sampler(1, 0, SamplerBindingType::NonFiltering),
                storage_of(2, 0, false, 8),
                storage_of(3, 1, true, 16),
            ],
            static_memory: 0,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var<storage, read> coords: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4f>;
{wg}
fn main(@builtin(global_invocation_id) id: vec3u) {{
    out[id.x] = textureSampleLevel(t, s, coords[id.x], 0.0);
}}",
                    wg = workgroup(k)
                ))
            },
        },
        "compute_add_one" => Kernel {
            bindings: vec![storage(0, 0, true)],
            static_memory: 0,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var<storage, read_write> data: array<u32>;
{wg}
fn main(@builtin(global_invocation_id) id: vec3u) {{
    data[id.x] = data[id.x] + 1u;
}}",
                    wg = workgroup(k)
                ))
            },
        },
        "compute_scale" => Kernel {
            bindings: vec![storage(0, 0, true), storage(1, 1, false), uniform(2, 2, 4)],
            static_memory: 0,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<uniform> factor: u32;
{wg}
fn main(@builtin(global_invocation_id) id: vec3u) {{
    out[id.x] = input[id.x] * factor;
}}",
                    wg = workgroup(k)
                ))
            },
        },
        // compute_threadgroup_atomics: two `threadgroup atomic_uint` (8 bytes the function declares).
        "compute_threadgroup_atomics" => Kernel {
            bindings: vec![storage(0, 0, true)],
            static_memory: 8,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
var<workgroup> sum: atomic<u32>;
var<workgroup> maximum: atomic<u32>;
{wg}
fn main(@builtin(local_invocation_index) tiitg: u32, @builtin(workgroup_id) tgpig: vec3u) {{
    if (tiitg == 0u) {{
        atomicStore(&sum, 0u);
        atomicStore(&maximum, 0u);
    }}
    workgroupBarrier();
    atomicAdd(&sum, tiitg + tgpig.x);
    atomicMax(&maximum, tiitg * 3u + tgpig.x);
    workgroupBarrier();
    if (tiitg == 0u) {{
        out[tgpig.x * 2u] = atomicLoad(&sum);
        out[tgpig.x * 2u + 1u] = atomicLoad(&maximum);
    }}
}}",
                    wg = workgroup(k)
                ))
            },
        },
        // compute_stage_in with the case's stage input descriptor (checked when the pipeline is made): attribute 0
        // ushort2 at offset 0 and attribute 1 float at offset 4 of buffer 1, stride 8, step function
        // MTLStepFunctionThreadPositionInGridX with rate 1, so the record of a thread is the region's x origin plus
        // its x position in the grid.
        "compute_stage_in" => Kernel {
            bindings: vec![
                storage(0, 0, true),
                storage(1, 1, false),
                uniforms(2, vec![(0, UniformValue::StageInOrigin), (16, UniformValue::StageInSize)], 32),
            ],
            static_memory: 0,
            wgsl: |k| {
                let [x, y, _] = k.threads_per_threadgroup;
                Ok(format!(
                    "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<storage, read> records: array<u32>;
struct Stage {{ origin: vec4u, size: vec4u }}
@group(0) @binding(2) var<uniform> stage: Stage;
{wg}
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {{
    let grid = groups.xy * vec2u({x}u, {y}u);
    let index = stage.origin.x + gid.x;
    let pair = records[index * 2u];
    let scale = records[index * 2u + 1u];
    let r = (gid.y * grid.x + gid.x) * 4u;
    out[r] = pair & 0xffffu;
    out[r + 1u] = pair >> 16u;
    out[r + 2u] = scale;
    out[r + 3u] = gid.y * 0x100u + gid.x;
}}",
                    wg = workgroup(k)
                ))
            },
        },
        // blit.metal: blit_read_ms.
        "blit_read_ms" => Kernel {
            bindings: vec![sampled(0, 0, TextureSampleType::Uint, true), storage(1, 0, true)],
            static_memory: 0,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var source: texture_multisampled_2d<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
{wg}
fn main(@builtin(global_invocation_id) gid: vec3u) {{
    for (var s = 0u; s < 4u; s++) {{
        out[(gid.y * 4u + gid.x) * 4u + s] = textureLoad(source, gid.xy, s).r;
    }}
}}",
                    wg = workgroup(k)
                ))
            },
        },
        // formats.metal: fmt_read_depth (`depth2d<float, access::read>`).
        "fmt_read_depth" => Kernel {
            bindings: vec![sampled(0, 0, TextureSampleType::Depth, false), storage(1, 0, true)],
            static_memory: 0,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var t: texture_depth_2d;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
{wg}
fn main(@builtin(global_invocation_id) gid: vec3u) {{
    let size = textureDimensions(t);
    if (gid.x < size.x && gid.y < size.y) {{
        out[gid.y * size.x + gid.x] = textureLoad(t, gid.xy, 0);
    }}
}}",
                    wg = workgroup(k)
                ))
            },
        },
        // formats.metal: fmt_read_uint (`texture2d<uint, access::read>`).
        "fmt_read_uint" => Kernel {
            bindings: vec![sampled(0, 0, TextureSampleType::Uint, false), storage_of(1, 0, true, 16)],
            static_memory: 0,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var t: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4u>;
{wg}
fn main(@builtin(global_invocation_id) gid: vec3u) {{
    let size = textureDimensions(t);
    if (gid.x < size.x && gid.y < size.y) {{
        out[gid.y * size.x + gid.x] = textureLoad(t, gid.xy, 0);
    }}
}}",
                    wg = workgroup(k)
                ))
            },
        },
        // textures.metal: tex_read_2d, `out[i] = t.read(q[2 * i].xy, q[2 * i + 1].x)` of a `texture2d<uint,
        // access::read>`.
        "tex_read_2d" => Kernel {
            bindings: read_bindings(TextureSampleType::Uint, TextureViewDimension::D2),
            static_memory: 0,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var t: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> q: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4u>;
{wg}
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let i = id.x;
    out[i] = textureLoad(t, q[2u * i].xy, q[2u * i + 1u].x);
}}",
                    wg = workgroup(k)
                ))
            },
        },
        // tex_read_2d_float: the same of a `texture2d<float, access::read>`.
        "tex_read_2d_float" => Kernel {
            bindings: read_bindings(TextureSampleType::UnfilterableFloat, TextureViewDimension::D2),
            static_memory: 0,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> q: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4f>;
{wg}
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let i = id.x;
    out[i] = textureLoad(t, q[2u * i].xy, q[2u * i + 1u].x);
}}",
                    wg = workgroup(k)
                ))
            },
        },
        // tex_read_2d_array: `t.read(q[2 * i].xy, q[2 * i].z, q[2 * i + 1].x)` of a `texture2d_array<uint,
        // access::read>`.
        "tex_read_2d_array" => Kernel {
            bindings: read_bindings(TextureSampleType::Uint, TextureViewDimension::D2Array),
            static_memory: 0,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var t: texture_2d_array<u32>;
@group(0) @binding(1) var<storage, read> q: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4u>;
{wg}
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let i = id.x;
    out[i] = textureLoad(t, q[2u * i].xy, q[2u * i].z, q[2u * i + 1u].x);
}}",
                    wg = workgroup(k)
                ))
            },
        },
        // tex_read_cube: `t.read(q[2 * i].xy, q[2 * i].z, q[2 * i + 1].x)` of a `texturecube<uint, access::read>`,
        // which reads texel (x, y) of a face. WGSL has no load from a cube texture, so the cube is bound as the 2D
        // array of its six faces.
        "tex_read_cube" => Kernel {
            bindings: read_bindings(TextureSampleType::Uint, TextureViewDimension::D2Array),
            static_memory: 0,
            wgsl: |k| {
                Ok(format!(
                    "@group(0) @binding(0) var t: texture_2d_array<u32>;
@group(0) @binding(1) var<storage, read> q: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4u>;
{wg}
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let i = id.x;
    out[i] = textureLoad(t, q[2u * i].xy, q[2u * i].z, q[2u * i + 1u].x);
}}",
                    wg = workgroup(k)
                ))
            },
        },
        _ => return None,
    })
}

/// The variants of one stand-in pipeline, made on first use for each key.
struct StandIn {
    name: String,
    cache: RefCell<HashMap<VariantKey, Rc<ComputeVariant<A>>>>,
}

impl ComputeVariants<A> for StandIn {
    fn variant(&self, gpu: &ExecGpu<'_, A>, key: &VariantKey) -> Result<Rc<ComputeVariant<A>>, Failure> {
        if let Some(v) = self.cache.borrow().get(key) {
            return Ok(v.clone());
        }
        let k = kernel(&self.name).expect("made only for known kernels");
        let code = (k.wgsl)(key).map_err(|why| failure::unimplemented(format!("{}: {why}", self.name)))?;
        let module = gpu.call("GPUDevice.createShaderModule", &self.name, |d| {
            let mut m = ShaderModuleDescriptor::new(code.clone());
            m.label = self.name.clone();
            d.create_shader_module(&m)
        });
        let entries: Vec<BindGroupLayoutEntry> = k
            .bindings
            .iter()
            .map(|b| {
                let mut e = BindGroupLayoutEntry::new(b.binding, ShaderStage::COMPUTE);
                match &b.resource {
                    Resource::Buffer { ty, min_binding_size } => {
                        let mut l = BufferBindingLayout::new();
                        l.r#type = *ty;
                        l.min_binding_size = Size64::new(*min_binding_size).unwrap();
                        e.buffer = Some(l);
                    }
                    Resource::Texture { sample_type, view_dimension, multisampled } => {
                        let mut l = TextureBindingLayout::new();
                        l.sample_type = *sample_type;
                        l.view_dimension = *view_dimension;
                        l.multisampled = *multisampled;
                        e.texture = Some(l);
                    }
                    Resource::StorageTexture { access, format, view_dimension } => {
                        let mut l = StorageTextureBindingLayout::new(*format);
                        l.access = *access;
                        l.view_dimension = *view_dimension;
                        e.storage_texture = Some(l);
                    }
                    Resource::Sampler { ty } => {
                        let mut l = SamplerBindingLayout::new();
                        l.r#type = *ty;
                        e.sampler = Some(l);
                    }
                }
                e
            })
            .collect();
        let layout = gpu
            .call("GPUDevice.createBindGroupLayout", &self.name, |d| {
                d.create_bind_group_layout(&BindGroupLayoutDescriptor::new(entries))
            })
            .map_err(|e| failure::unimplemented(format!("{e}")))?;
        let pl = gpu.call("GPUDevice.createPipelineLayout", &self.name, |d| {
            d.create_pipeline_layout(&PipelineLayoutDescriptor::new(vec![Some(&layout)]))
        });
        let mut stage = ProgrammableStage::new(&module);
        stage.entry_point = Some("main".into());
        let mut desc = GpuComputePipelineDescriptor::new(PipelineLayoutOrAuto::PipelineLayout(&pl), stage);
        desc.label = self.name.clone();
        let pipeline = gpu.call("GPUDevice.createComputePipeline", &self.name, |d| d.create_compute_pipeline(&desc));
        let v = Rc::new(ComputeVariant { pipeline, groups: vec![BindGroup { group: 0, layout, entries: k.bindings }] });
        self.cache.borrow_mut().insert(key.clone(), v.clone());
        Ok(v)
    }
}

/// The pipeline service, with the hand translations for the kernels its compiler refuses.
pub struct Pipelines {
    service: Box<dyn ControlService<A, E>>,
    functions: HashMap<(TaskId, Handle), String>,
    compiled: Rc<RefCell<BTreeSet<String>>>,
}

impl ControlService<A, E> for Pipelines {
    fn opcodes(&self) -> Vec<u16> {
        self.service.opcodes()
    }

    fn serve(&mut self, cx: &mut ServiceCx<'_, A, E>, request: &Request) -> Served {
        let Request::CreateComputePipeline(r) = request else {
            if let Request::RegisterFunction(r) = request {
                self.functions.insert((cx.task(), r.function), r.name.clone());
            }
            return self.service.serve(cx, request);
        };
        let name = r.descriptor.compute_function.and_then(|f| self.functions.get(&(cx.task(), f)).cloned());
        let served = self.service.serve(cx, request);
        let refused = match &served {
            Served::Ready(reply) => webmetal_host::exec::schema::reply_failure(reply)
                .is_some_and(|f| f.status == status::UNIMPLEMENTED && f.message.contains("shader compiler refuses")),
            Served::Later(_) => false,
        };
        let name = name.unwrap_or_default();
        if !refused {
            let failed = matches!(&served, Served::Ready(reply)
                if webmetal_host::exec::schema::reply_failure(reply).is_some());
            if !failed {
                self.compiled.borrow_mut().insert(name);
            }
            return served;
        }
        let Some(k) = kernel(&name) else { return served };
        let stage_input = r.descriptor.stage_input_descriptor.is_some();
        if name == "compute_stage_in" {
            // The fetch the WGSL hard-codes: the case's descriptor.
            let s = r.descriptor.stage_input_descriptor.as_ref().expect("a stage input descriptor");
            let attrs: Vec<(u64, u64, u64)> =
                s.attributes.iter().map(|a| (a.format, a.offset, a.buffer_index)).collect();
            // MTLAttributeFormatUShort2 13, MTLAttributeFormatFloat 28.
            assert_eq!(&attrs[..2], &[(13, 0, 1), (28, 4, 1)]);
            // MTLStepFunctionThreadPositionInGridX 5.
            let l = &s.layouts[1];
            assert_eq!((l.stride, l.step_function, l.step_rate), (8, 5, 1));
        }
        let max = match r.descriptor.max_total_threads_per_threadgroup {
            0 => 512,
            m => m.min(512),
        };
        cx.attach(Rc::new(ComputePipelineState::<A> {
            max_total_threads_per_threadgroup: max,
            thread_execution_width: 32,
            threadgroup_size_is_multiple_of_execution_width: r
                .descriptor
                .thread_group_size_is_multiple_of_thread_execution_width,
            static_threadgroup_memory_length: k.static_memory,
            stage_input,
            variants: Rc::new(StandIn { name, cache: RefCell::new(HashMap::new()) }),
        }));
        Served::Ready(Reply::CreateComputePipeline(Ok(CreateComputePipelineReply { reflection: None })))
    }

    fn release(&mut self, task: TaskId, handle: Handle, entry: &Entry) {
        self.functions.remove(&(task, handle));
        self.service.release(task, handle, entry);
    }

    fn task_exited(&mut self, task: TaskId) {
        self.functions.retain(|(t, _), _| *t != task);
        self.service.task_exited(task);
    }
}

// The render stand-in ---------------------------------------------------------------------------------------------------

/// Render passes that clear color attachment 0 and store it, with no draws.
pub struct ClearPass {
    memory: Rc<Memory<A>>,
}

impl EncoderService<A, E> for ClearPass {
    fn encode(&mut self, cx: &mut EncodeCx<'_, A, E>, segment: &Segment) -> Result<(), RecordFailure> {
        let records = cx.records();
        let (offset, first) = &records[segment.records[0]];
        let refuse = || RecordFailure {
            offset: *offset as u64,
            failure: failure::unimplemented("the render stand-in records only clearing passes"),
        };
        let Command::BeginRenderPass(r) = first else { return Err(refuse()) };
        if segment.records.len() != 2 || r.render_pass_descriptor.color_attachments.len() != 1 {
            return Err(refuse());
        }
        let a = &r.render_pass_descriptor.color_attachments[0];
        // MTLLoadActionClear 2, MTLStoreActionStore 1.
        if a.load_action != 2 || a.store_action != 1 {
            return Err(refuse());
        }
        let t =
            cx.objects().get(a.texture.unwrap()).unwrap().payload.clone().unwrap().downcast::<Texture<A>>().unwrap();
        let encoder = cx.encoder().clone();
        let view = t.gpu().create_view(&TextureViewDescriptor::new()).unwrap();
        let c = &a.clear_color;
        let f = |v: f64| Float64::new(v).unwrap();
        let mut ca = GpuColorAttachment::new(TextureOrView::TextureView(&view), LoadOp::Clear, StoreOp::Store);
        ca.clear_value = Some(Color::ColorDict(ColorDict { r: f(c.red), g: f(c.green), b: f(c.blue), a: f(c.alpha) }));
        let rp = GpuRenderPass::new(vec![Some(ca)]);
        let pass = cx.gpu().call("GPUCommandEncoder.beginRenderPass", "", |_| encoder.begin_render_pass(&rp)).unwrap();
        cx.gpu().call("GPURenderPassEncoder.end", "", |_| pass.end());
        self.memory.note_texture_write(&t, 0..1, 0..1);
        Ok(())
    }
}

// The harness -----------------------------------------------------------------------------------------------------------

pub struct Harness {
    /// The kernels the pipeline service compiled (the others came from the hand translations).
    pub compiled: Rc<RefCell<BTreeSet<String>>>,
    library: HashMap<String, LibraryFunction>,
    pub g: Gpu,
    pub guest: Rc<Guest>,
    pub exec: Executor<A, E>,
    pub memory: Rc<Memory<A>>,
    next: Cell<u64>,
    alloc: Cell<u64>,
    queue: Cell<Option<Handle>>,
}

impl Harness {
    pub fn new() -> Harness {
        let g = Gpu::new(true);
        let guest = Rc::new(Guest::new(96 << 20, None));
        let svc = g
            .gpu
            .block_on(MemoryServices::new(&g.device, g.features, guest.clone()), Duration::from_secs(120))
            .expect("setup stalled")
            .expect("memory services");
        let memory = svc.memory();
        // wgpu-native does not report its WGSL language extensions; it is taken to offer none.
        let wgsl = g.gpu.wgsl_language_features().unwrap_or_default();
        let compiled = Rc::new(RefCell::new(BTreeSet::new()));
        let pipelines = Pipelines {
            service: PipelineServices::<A>::new(&g.device, wgsl, pipeline::Config::default()).control(),
            functions: HashMap::new(),
            compiled: compiled.clone(),
        };
        let services = Services {
            control: vec![
                svc.control() as Box<dyn ControlService<A, E>>,
                Box::new(pipelines),
                Box::new(webmetal_host::state::StateServices::new()),
            ],
            render: Some(Box::new(ClearPass { memory: memory.clone() })),
            compute: Some(Box::new(ComputeService::new(memory.clone()))),
            blit: Some(Box::new(BlitService::new(memory.clone()))),
            coherence: Some(svc.coherence()),
        };
        let config = Config { scopes: Policy { granularity: Granularity::PerCall, strict: true } };
        let exec = Executor::new(g.device.clone(), guest.clone(), services, config);
        let base = guest.base;
        let h = Harness {
            compiled,
            library: reference_library(),
            g,
            guest,
            exec,
            memory,
            next: Cell::new(0x100),
            alloc: Cell::new(base + 0x1000),
            queue: Cell::new(None),
        };
        h.ok(Request::Open(Open { schema_hash: Hash { bytes: SCHEMA_HASH }, pid: 1 }));
        let q = h.handle();
        h.ok(Request::CreateCommandQueue(CreateCommandQueue { queue: q }));
        h.queue.set(Some(q));
        h
    }

    pub fn name(&self) -> String {
        self.g.name()
    }

    pub fn handle(&self) -> Handle {
        self.next.set(self.next.get() + 1);
        Handle::new(self.next.get()).unwrap()
    }

    pub fn alloc(&self, len: u64) -> u64 {
        let a = self.alloc.get();
        self.alloc.set((a + len.max(1)).next_multiple_of(0x4000) + 0x4000);
        a
    }

    pub fn request(&self, request: Request) -> Reply {
        let mut bytes = Vec::new();
        request.encode(&mut bytes).unwrap();
        let out = self.g.block_on(self.exec.control(TASK, &bytes)).unwrap();
        Reply::decode(&out, 0).unwrap().0
    }

    pub fn status(&self, request: Request) -> Failure {
        webmetal_host::exec::schema::reply_failure(&self.request(request)).cloned().unwrap_or_else(failure::ok)
    }

    pub fn ok(&self, request: Request) {
        let f = self.status(request);
        assert_eq!(f.status, status::OK, "{f:?}");
    }

    /// `newBufferWithBytes:length:options:` with shared storage: the handle and the guest address of its contents.
    pub fn buffer_with(&self, bytes: &[u8]) -> (Handle, u64) {
        let addr = self.alloc(bytes.len() as u64);
        self.guest.cpu_write(addr, bytes);
        let b = self.handle();
        self.ok(Request::CreateBuffer(CreateBuffer {
            buffer: b,
            length: bytes.len() as u64,
            options: 0,
            contents: GuestAddr(addr),
        }));
        (b, addr)
    }

    /// `newBufferWithLength:` (shared, zero-filled as the harness's `buffer:` does).
    pub fn buffer(&self, len: usize) -> (Handle, u64) {
        self.buffer_with(&vec![0; len])
    }

    pub fn texture(&self, d: TextureDescriptor) -> Handle {
        let t = self.handle();
        self.ok(Request::CreateTexture(CreateTexture { texture: t, descriptor: d }));
        t
    }

    pub fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        self.guest.cpu_read(addr, len)
    }

    /// A compute pipeline of a reference kernel (`[c computePipeline:]`), with an optional stage input descriptor.
    pub fn pipeline(&self, name: &str, stage: Option<StageInputOutputDescriptor>, max_threads: u64) -> Handle {
        let bitcode =
            &self.library.get(name).unwrap_or_else(|| panic!("{name} is not in the reference library")).bitcode;
        let hash = sha256(bitcode);
        // RegisterAir is sent once per digest on a connection.
        let s = self.status(Request::RegisterAir(RegisterAir { hash: Hash { bytes: hash }, bitcode: bitcode.clone() }));
        assert!(s.status == status::OK || s.message.contains("already registered"), "{s:?}");
        let f = self.handle();
        self.ok(Request::RegisterFunction(RegisterFunction {
            function: f,
            air: Hash { bytes: hash },
            name: name.into(),
            function_type: 3,
            constants: Vec::new(),
        }));
        let p = self.handle();
        let mut d = compute_descriptor(f);
        d.stage_input_descriptor = stage;
        d.max_total_threads_per_threadgroup = max_threads;
        self.ok(Request::CreateComputePipeline(CreateComputePipeline {
            pipeline: p,
            descriptor: d,
            options: 0,
            reflection: false,
            error: false,
        }));
        p
    }

    /// Commit `records` as one command buffer on the harness's queue and wait until it completes; returns its
    /// completion failure (status 0 when it completed normally) and the failed record's offset.
    pub fn run(&self, records: &[Command]) -> (Failure, u64) {
        let mut bytes = Vec::new();
        for r in records {
            r.encode(&mut bytes).unwrap();
        }
        let mut created = BTreeSet::new();
        let mut named = Vec::new();
        for r in records {
            command_handles(r, &mut |u| {
                if u.role == Role::Result {
                    created.insert(u.handle);
                } else if !named.contains(&u.handle) {
                    named.push(u.handle);
                }
            });
        }
        named.retain(|h| !created.contains(h));
        let addr = self.alloc(bytes.len() as u64);
        self.guest.cpu_write(addr, &bytes);
        let cb = self.handle();
        self.ok(Request::Commit(Commit {
            command_buffer: cb,
            queue: self.queue.get().unwrap(),
            label: None,
            retained_references: true,
            error_options: 0,
            stream: GuestAddr(addr),
            stream_length: bytes.len() as u64,
            resources: named,
        }));
        self.g.block_on(self.exec.idle());
        let posted = self.guest.posted.borrow();
        let c = posted
            .iter()
            .find_map(|e| match e {
                Event::CommandBufferCompleted(c) if c.command_buffer == cb => Some(c.clone()),
                _ => None,
            })
            .expect("a completion");
        (c.failure, c.failed_offset)
    }

    /// Run one encoder and require that its command buffer completes normally.
    pub fn encode(&self, begin: Command, body: Vec<Command>) {
        let mut v = vec![begin];
        v.extend(body);
        v.push(Command::EncoderEndEncoding(EncoderEndEncoding {}));
        let (f, at) = self.run(&v);
        assert_eq!(f.status, status::OK, "{}: offset {at}: {f:?}", self.name());
    }

    pub fn blit(&self, body: Vec<Command>) {
        let e = self.handle();
        self.encode(Command::BeginBlit(BeginBlit { encoder: e }), body);
    }

    pub fn compute(&self, body: Vec<Command>) {
        let e = self.handle();
        self.encode(Command::BeginCompute(BeginCompute { encoder: e }), body);
    }

    /// Run a blit encoder expected to fail: its completion failure.
    pub fn blit_failure(&self, body: Vec<Command>) -> Failure {
        let e = self.handle();
        let mut v = vec![Command::BeginBlit(BeginBlit { encoder: e })];
        v.extend(body);
        v.push(Command::EncoderEndEncoding(EncoderEndEncoding {}));
        self.run(&v).0
    }

    pub fn compute_failure(&self, body: Vec<Command>) -> Failure {
        let e = self.handle();
        let mut v = vec![Command::BeginCompute(BeginCompute { encoder: e })];
        v.extend(body);
        v.push(Command::EncoderEndEncoding(EncoderEndEncoding {}));
        self.run(&v).0
    }

    /// `WMFillTexture` of Blit.m: every level and slice filled through blits from buffers of generated bytes.
    pub fn fill_texture(&self, t: Handle, d: &TextureDescriptor, bytes_per_pixel: u64, seed: u32) {
        let slices = d.array_length * if d.texture_type == TCUBE || d.texture_type == 6 { 6 } else { 1 };
        for level in 0..d.mipmap_level_count {
            let w = (d.width >> level).max(1);
            let h = (d.height >> level).max(1);
            let depth = if d.texture_type == T3D { (d.depth >> level).max(1) } else { 1 };
            for slice in 0..slices {
                let len = w * h * depth * bytes_per_pixel;
                let bytes = pattern(len as usize, seed + (level * 64 + slice) as u32);
                let (src, _) = self.buffer_with(&bytes);
                let image = if d.texture_type == T3D { w * h * bytes_per_pixel } else { 0 };
                self.blit(vec![Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
                    source_buffer: Some(src),
                    source_offset: 0,
                    source_bytes_per_row: w * bytes_per_pixel,
                    source_bytes_per_image: image,
                    source_size: size(w, h, depth),
                    destination_texture: Some(t),
                    destination_slice: slice,
                    destination_level: level,
                    destination_origin: origin(0, 0, 0),
                })]);
            }
        }
    }

    /// `readTexture:level:slice:bytesPerRow:` of the harness: the level through a blit into a buffer.
    pub fn read_texture(&self, t: Handle, d: &TextureDescriptor, level: u64, slice: u64, bpr: u64) -> Vec<u8> {
        let w = (d.width >> level).max(1);
        let h = (d.height >> level).max(1);
        let depth = if d.texture_type == T3D { (d.depth >> level).max(1) } else { 1 };
        let bpi = bpr * h;
        let (b, addr) = self.buffer((bpi * depth) as usize);
        self.blit(vec![Command::BlitCopyTextureToBuffer(BlitCopyTextureToBuffer {
            source_texture: Some(t),
            source_slice: slice,
            source_level: level,
            source_origin: origin(0, 0, 0),
            source_size: size(w, h, depth),
            destination_buffer: Some(b),
            destination_offset: 0,
            destination_bytes_per_row: bpr,
            destination_bytes_per_image: bpi,
        })]);
        self.read(addr, (bpi * depth) as usize)
    }

    /// Fails if WebGPU reported an error outside every scope.
    pub fn assert_no_errors(&self) {
        eprintln!("{}: kernels compiled by the pipeline service: {:?}", self.name(), self.compiled.borrow());
        self.g.assert_no_errors();
    }
}
