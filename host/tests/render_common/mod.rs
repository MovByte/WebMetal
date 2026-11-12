//! Shared pieces of the render path's tests on a native WebGPU implementation (`WEBGPU_NATIVE`, see `host/build.rs`):
//! an executor with the memory module, the render path and stand-ins for the services the render path does not own
//! (`services`), the Metal reference outputs (`golden`), and builders for the requests and streams the reference
//! harness's render cases make (`tests/reference/src/cases`).

#![allow(dead_code)]

pub mod golden;
pub mod json;
pub mod services;
pub mod wgsl;

use std::cell::Cell;
use std::collections::BTreeSet;
use std::rc::Rc;
use std::time::Duration;

use webmetal_host::exec::schema::{Role, command_handles};
use webmetal_host::exec::{Config, ControlService, Executor, Services, failure};
use webmetal_host::gpu::scope::{Granularity, Policy};
use webmetal_host::mem::Memory;
use webmetal_host::mem::service::MemoryServices;
use webmetal_host::render::RenderServices;
use webmetal_wire::*;

use crate::mem_common::{Api, Gpu, Guest, TASK};

pub type A = Api;
pub type E = Guest;

/// `MTLPixelFormat` values used by the cases (`MTLPixelFormat.h`).
pub mod pf {
    pub const R32_UINT: u64 = 53;
    pub const RGBA16_FLOAT: u64 = 115;
    pub const RGBA32_UINT: u64 = 123;
    pub const RGBA32_FLOAT: u64 = 125;
    pub const RGBA8_UNORM: u64 = 70;
    pub const RGBA16_SINT: u64 = 114;
    pub const DEPTH16_UNORM: u64 = 250;
    pub const DEPTH32_FLOAT: u64 = 252;
    pub const STENCIL8: u64 = 253;
    pub const DEPTH32_FLOAT_STENCIL8: u64 = 260;
}

/// `MTLTextureUsage` and `MTLStorageMode` values.
pub mod usage {
    pub const SHADER_READ: u64 = 0x1;
    pub const SHADER_WRITE: u64 = 0x2;
    pub const RENDER_TARGET: u64 = 0x4;
}
pub const STORAGE_SHARED: u64 = 0;
pub const STORAGE_PRIVATE: u64 = 2;
pub const STORAGE_MEMORYLESS: u64 = 3;

pub struct Harness {
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
        let guest = Rc::new(Guest::new(64 << 20, None));
        let svc = g
            .gpu
            .block_on(MemoryServices::new(&g.device, g.features, guest.clone()), Duration::from_secs(120))
            .expect("setup stalled")
            .expect("memory services");
        let memory = svc.memory();
        let render = RenderServices::new(memory.clone());
        let services = Services {
            control: vec![
                svc.control() as Box<dyn ControlService<A, E>>,
                render.control(),
                Box::new(services::Pipelines { features: g.features }),
            ],
            render: Some(render.encoder()),
            compute: Some(Box::new(services::Compute { memory: memory.clone() })),
            blit: Some(Box::new(services::Blit { memory: memory.clone() })),
            coherence: Some(svc.coherence()),
        };
        let config = Config { scopes: Policy { granularity: Granularity::PerCall, strict: true } };
        let exec = Executor::new(g.device.clone(), guest.clone(), services, config);
        let base = guest.base;
        let h = Harness {
            g,
            guest,
            exec,
            memory,
            next: Cell::new(0x100),
            alloc: Cell::new(base + 0x1000),
            queue: Cell::new(None),
        };
        h.ok(Request::Open(Open { schema_hash: Hash { bytes: SCHEMA_HASH }, pid: 1 }));
        h
    }

    pub fn has(&self, feature: &str) -> bool {
        use webmetal_host::gpu::Device as _;
        self.g.device.features().contains(feature)
    }

    pub fn handle(&self) -> Handle {
        self.next.set(self.next.get() + 1);
        Handle::new(self.next.get()).unwrap()
    }

    pub fn alloc(&self, len: u64) -> u64 {
        let a = self.alloc.get();
        self.alloc.set((a + len).next_multiple_of(0x4000) + 0x4000);
        a
    }

    pub fn request(&self, request: Request) -> Reply {
        let mut bytes = Vec::new();
        request.encode(&mut bytes).unwrap();
        let out = self.g.block_on(self.exec.control(TASK, &bytes)).unwrap();
        let (reply, _) = Reply::decode(&out, 0).unwrap();
        reply
    }

    pub fn status(&self, request: Request) -> Failure {
        webmetal_host::exec::schema::reply_failure(&self.request(request)).cloned().unwrap_or_else(failure::ok)
    }

    pub fn ok(&self, request: Request) {
        let f = self.status(request);
        assert_eq!(f.status, status::OK, "{f:?}");
    }

    /// A shared buffer holding `bytes` (`newBufferWithBytes:length:options:` copies them).
    pub fn buffer(&self, bytes: &[u8]) -> (Handle, u64) {
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

    pub fn zeros(&self, len: usize) -> (Handle, u64) {
        self.buffer(&vec![0; len])
    }

    pub fn texture(&self, d: TextureDescriptor) -> Result<Handle, Failure> {
        let t = self.handle();
        let f = self.status(Request::CreateTexture(CreateTexture { texture: t, descriptor: d }));
        if f.status == status::OK { Ok(t) } else { Err(f) }
    }

    /// `RTargetSampled` of `RenderSupport.m`: a private render target.
    pub fn target(&self, format: u64, w: u64, h: u64, samples: u64) -> Handle {
        let mut d = texture_descriptor(format, w, h, STORAGE_PRIVATE, usage::RENDER_TARGET | usage::SHADER_READ);
        if samples > 1 {
            d.texture_type = 4;
            d.sample_count = samples;
        }
        self.texture(d).unwrap_or_else(|f| panic!("texture of format {format}: {f:?}"))
    }

    pub fn queue(&self) -> Handle {
        if let Some(q) = self.queue.get() {
            return q;
        }
        let q = self.handle();
        self.ok(Request::CreateCommandQueue(CreateCommandQueue { queue: q }));
        self.queue.set(Some(q));
        q
    }

    pub fn function(&self, name: &str) -> Handle {
        let f = self.handle();
        self.ok(Request::RegisterFunction(RegisterFunction {
            function: f,
            air: Hash { bytes: [0; 32] },
            name: name.into(),
            function_type: 0,
            constants: vec![],
        }));
        f
    }

    pub fn pipeline(&self, d: RenderPipelineDescriptor) -> Result<Handle, Failure> {
        let p = self.handle();
        let f = self.status(Request::CreateRenderPipeline(CreateRenderPipeline {
            pipeline: p,
            descriptor: d,
            error: true,
        }));
        if f.status == status::OK { Ok(p) } else { Err(f) }
    }

    /// `RPipelineDescriptor` of `RenderSupport.m`.
    pub fn pipeline_of(&self, vertex: &str, fragment: Option<&str>, color: u64) -> RenderPipelineDescriptor {
        let mut d = render_pipeline_descriptor();
        d.vertex_function = Some(self.function(vertex));
        d.fragment_function = fragment.map(|f| self.function(f));
        d.color_attachments[0].pixel_format = color;
        d
    }

    pub fn simple_pipeline(&self, vertex: &str, fragment: &str, color: u64) -> Handle {
        self.pipeline(self.pipeline_of(vertex, Some(fragment), color)).expect("pipeline")
    }

    pub fn depth_stencil(&self, d: DepthStencilDescriptor) -> Handle {
        let s = self.handle();
        self.ok(Request::CreateDepthStencilState(CreateDepthStencilState { state: s, descriptor: d }));
        s
    }

    pub fn compute_pipeline(&self, name: &str) -> Handle {
        let mut d = compute_pipeline_descriptor();
        d.compute_function = Some(self.function(name));
        let p = self.handle();
        self.ok(Request::CreateComputePipeline(CreateComputePipeline {
            pipeline: p,
            descriptor: d,
            options: 0,
            reflection: false,
            error: true,
        }));
        p
    }

    /// Commit a command buffer with `records` (its resources are the objects they name) and wait for it. Returns the
    /// failure it completed with.
    pub fn run(&self, records: &[Command]) -> Failure {
        let mut bytes = Vec::new();
        let mut created = BTreeSet::new();
        let mut named = Vec::new();
        for r in records {
            r.encode(&mut bytes).unwrap();
            command_handles(r, &mut |u| {
                if u.role == Role::Result {
                    created.insert(u.handle);
                } else if !created.contains(&u.handle) && !named.contains(&u.handle) {
                    named.push(u.handle);
                }
            });
        }
        let addr = self.alloc(bytes.len() as u64);
        self.guest.cpu_write(addr, &bytes);
        let cb = self.handle();
        self.guest.posted.borrow_mut().clear();
        self.ok(Request::Commit(Commit {
            command_buffer: cb,
            queue: self.queue(),
            label: None,
            retained_references: true,
            error_options: 0,
            protection_options: 0,
            stream: GuestAddr(addr),
            stream_length: bytes.len() as u64,
            resources: named,
        }));
        self.g.block_on(self.exec.idle());
        self.g.assert_no_errors();
        let posted = self.guest.posted.borrow();
        posted
            .iter()
            .find_map(|e| match e {
                Event::CommandBufferCompleted(c) if c.command_buffer == cb => Some(c.failure.clone()),
                _ => None,
            })
            .expect("the command buffer completed")
    }

    /// As [`Harness::run`], requiring success.
    pub fn run_ok(&self, records: &[Command]) {
        let f = self.run(records);
        assert_eq!(f.status, status::OK, "{f:?}");
    }

    pub fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        self.guest.cpu_read(addr, len)
    }

    /// `readTexture:level:slice:bytesPerRow:` of the harness: a blit into a shared buffer.
    pub fn read_texture(&self, t: Handle, w: u64, h: u64, bytes_per_row: u64) -> Vec<u8> {
        self.read_texture_options(t, 0, w, h, bytes_per_row, 0)
    }

    pub fn read_texture_options(
        &self,
        t: Handle,
        slice: u64,
        w: u64,
        h: u64,
        bytes_per_row: u64,
        options: u64,
    ) -> Vec<u8> {
        let len = (bytes_per_row * h) as usize;
        let (b, addr) = self.zeros(len);
        let e = self.handle();
        self.run_ok(&[
            Command::BeginBlit(BeginBlit { encoder: e }),
            Command::BlitCopyTextureToBufferOptions(BlitCopyTextureToBufferOptions {
                source_texture: Some(t),
                source_slice: slice,
                source_level: 0,
                source_origin: Origin { x: 0, y: 0, z: 0 },
                source_size: Size { width: w, height: h, depth: 1 },
                destination_buffer: Some(b),
                destination_offset: 0,
                destination_bytes_per_row: bytes_per_row,
                destination_bytes_per_image: bytes_per_row * h,
                options,
            }),
            end(),
        ]);
        self.read(addr, len)
    }

    /// `BReadSamples` of `Blend.m`: every sample of a multisample texture through one of the `rnd_read_ms_*` kernels.
    pub fn read_samples(&self, t: Handle, kernel: &str, element: u64, w: u64, h: u64, samples: u64) -> Vec<u8> {
        let p = self.compute_pipeline(kernel);
        let len = (w * h * samples * element) as usize;
        let (b, addr) = self.zeros(len);
        let e = self.handle();
        self.run_ok(&[
            Command::BeginCompute(BeginCompute { encoder: e }),
            Command::ComputeSetComputePipelineState(ComputeSetComputePipelineState { state: Some(p) }),
            Command::ComputeSetTexture(ComputeSetTexture { texture: Some(t), index: 0 }),
            Command::ComputeSetBuffer(ComputeSetBuffer { buffer: Some(b), offset: 0, index: 0 }),
            Command::ComputeDispatchThreadgroups(ComputeDispatchThreadgroups {
                threadgroups_per_grid: Size { width: w, height: h, depth: 1 },
                threads_per_threadgroup: Size { width: 1, height: 1, depth: 1 },
            }),
            end(),
        ]);
        self.read(addr, len)
    }
}

pub fn end() -> Command {
    Command::EncoderEndEncoding(EncoderEndEncoding {})
}

/// A render encoder: `BeginRenderPass`, `body`, `EndEncoding`.
pub fn render(encoder: Handle, pass: RenderPassDescriptor, body: Vec<Command>) -> Vec<Command> {
    let mut v = vec![Command::BeginRenderPass(BeginRenderPass { encoder, render_pass_descriptor: pass })];
    v.extend(body);
    v.push(end());
    v
}

/// A texture descriptor with Metal's defaults (`texture2DDescriptorWithPixelFormat:width:height:mipmapped:NO`).
pub fn texture_descriptor(format: u64, w: u64, h: u64, storage: u64, usage: u64) -> TextureDescriptor {
    TextureDescriptor {
        allow_gpu_optimized_contents: true,
        array_length: 1,
        color_space_conversion_matrix: 0,
        compression_footprint: 0,
        compression_mode: 0,
        compression_type: 0,
        cpu_cache_mode: 0,
        depth: 1,
        force_resource_index: false,
        framebuffer_only: false,
        hazard_tracking_mode: 0,
        height: h,
        is_drawable: false,
        lossy_compression_mode: 0,
        mipmap_level_count: 1,
        pixel_format: format,
        protection_options: 0,
        resource_index: 0,
        resource_options: storage << 4,
        rotation: 0,
        sample_count: 1,
        sparse_surface_default_value: 0,
        storage_mode: storage,
        swizzle: TextureSwizzleChannels { red: 2, green: 3, blue: 4, alpha: 5 },
        swizzle_key: 0,
        texture_type: 2,
        texture_usage: usage,
        usage,
        width: w,
        write_swizzle_enabled: false,
    }
}

pub fn clear(r: f64, g: f64, b: f64, a: f64) -> ClearColor {
    ClearColor { red: r, green: g, blue: b, alpha: a }
}

/// A color attachment descriptor with Metal's defaults (load dontCare, store dontCare), for `texture`.
pub fn color_attachment(texture: Option<Handle>) -> RenderPassColorAttachment {
    RenderPassColorAttachment {
        clear_color: clear(0.0, 0.0, 0.0, 1.0),
        depth_plane: 0,
        level: 0,
        load_action: 0,
        resolve_depth_plane: 0,
        resolve_level: 0,
        resolve_slice: 0,
        resolve_texture: None,
        slice: 0,
        store_action: 0,
        store_action_options: 0,
        texture,
        y_invert: false,
    }
}

pub fn depth_attachment(texture: Handle) -> RenderPassDepthAttachment {
    RenderPassDepthAttachment {
        clear_depth: 1.0,
        depth_plane: 0,
        depth_resolve_filter: 0,
        level: 0,
        load_action: 0,
        resolve_depth_plane: 0,
        resolve_level: 0,
        resolve_slice: 0,
        resolve_texture: None,
        slice: 0,
        store_action: 0,
        store_action_options: 0,
        texture: Some(texture),
        y_invert: false,
    }
}

pub fn stencil_attachment(texture: Handle) -> RenderPassStencilAttachment {
    RenderPassStencilAttachment {
        clear_stencil: 0,
        depth_plane: 0,
        level: 0,
        load_action: 0,
        resolve_depth_plane: 0,
        resolve_level: 0,
        resolve_slice: 0,
        resolve_texture: None,
        slice: 0,
        stencil_resolve_filter: 0,
        store_action: 0,
        store_action_options: 0,
        texture: Some(texture),
        y_invert: false,
    }
}

/// `[MTLRenderPassDescriptor renderPassDescriptor]`: eight color attachments without textures.
pub fn pass_descriptor() -> RenderPassDescriptor {
    RenderPassDescriptor {
        binary_archives: vec![],
        color_attachments: (0..8).map(|_| color_attachment(None)).collect(),
        default_color_sample_count: 0,
        default_raster_sample_count: 0,
        depth_attachment: None,
        dither_enabled: false,
        fine_grained_background_visibility_enabled: false,
        imageblock_sample_length: 0,
        open_gl_mode_enabled: false,
        point_coord_y_flip_enabled: false,
        rasterization_rate_map: None,
        render_target_array_length: 0,
        render_target_height: 0,
        render_target_width: 0,
        sample_buffer_attachments: vec![],
        skip_empty_tiles_on_clear_enabled: false,
        stencil_attachment: None,
        threadgroup_memory_length: 0,
        tile_height: 0,
        tile_width: 0,
        visibility_result_buffer: None,
        sample_positions: vec![],
    }
}

/// `RPassColor` of `RenderSupport.m`: color attachment 0 cleared to `clear` and stored.
pub fn pass_color(texture: Handle, c: ClearColor) -> RenderPassDescriptor {
    let mut p = pass_descriptor();
    p.color_attachments[0].texture = Some(texture);
    p.color_attachments[0].load_action = 2;
    p.color_attachments[0].store_action = 1;
    p.color_attachments[0].clear_color = c;
    p
}

/// A stencil descriptor with Metal's defaults (compare always, keep, masks all ones).
pub fn stencil(compare: u64, fail: u64, depth_fail: u64, pass: u64, read: u32, write: u32) -> StencilDescriptor {
    StencilDescriptor {
        depth_failure_operation: depth_fail,
        depth_stencil_pass_operation: pass,
        read_mask: read,
        stencil_compare_function: compare,
        stencil_failure_operation: fail,
        write_mask: write,
    }
}

pub fn depth_stencil_descriptor(compare: u64, write: bool) -> DepthStencilDescriptor {
    DepthStencilDescriptor {
        back_face_stencil: None,
        depth_compare_function: compare,
        depth_write_enabled: write,
        front_face_stencil: None,
        label: None,
    }
}

fn size0() -> Size {
    Size { width: 0, height: 0, depth: 0 }
}

/// `[MTLRenderPipelineDescriptor new]`: Metal's defaults, eight color attachments of the invalid format.
pub fn render_pipeline_descriptor() -> RenderPipelineDescriptor {
    let color = RenderPipelineColorAttachment {
        alpha_blend_operation: 0,
        blending_enabled: false,
        destination_alpha_blend_factor: 0,
        destination_rgb_blend_factor: 0,
        pixel_format: 0,
        rgb_blend_operation: 0,
        source_alpha_blend_factor: 1,
        source_rgb_blend_factor: 1,
        write_mask: 0xf,
    };
    RenderPipelineDescriptor {
        alpha_test_enabled: false,
        alpha_test_function: 0,
        alpha_to_coverage_enabled: false,
        alpha_to_one_enabled: false,
        binary_archives: vec![],
        clip_distance_enable_mask: 0,
        color_attachments: vec![color; 8],
        color_sample_count: 1,
        depth_attachment_pixel_format: 0,
        depth_stencil_write_disabled: false,
        driver_compiler_options: None,
        explicit_visibility_group_id: 0,
        force_resource_index: false,
        force_software_vertex_fetch: false,
        fragment_buffers: vec![],
        fragment_depth_compare_clamp_mask: 0,
        fragment_function: None,
        fragment_linked_functions: None,
        fragment_preloaded_libraries: vec![],
        gpu_compiler_spi_options: None,
        input_primitive_topology: 0,
        insert_fragment_libraries: vec![],
        insert_mesh_libraries: vec![],
        insert_object_libraries: vec![],
        insert_vertex_libraries: vec![],
        label: None,
        logic_operation: 0,
        logic_operation_enabled: false,
        max_acceleration_structure_traversal_depth: 0,
        max_fragment_call_stack_depth: 1,
        max_mesh_call_stack_depth: 1,
        max_object_call_stack_depth: 1,
        max_pipeline_children: size0(),
        max_tessellation_factor: 16,
        max_total_threads_per_mesh_threadgroup: 0,
        max_total_threads_per_object_threadgroup: 0,
        max_vertex_amplification_count: 1,
        max_vertex_call_stack_depth: 1,
        mesh_function: None,
        mesh_linked_functions: None,
        mesh_preloaded_libraries: vec![],
        mesh_threadgroup_size_is_multiple_of_thread_execution_width: false,
        mesh_threads_per_threadgroup: size0(),
        needs_custom_border_color_samplers: false,
        object_function: None,
        object_linked_functions: None,
        object_preloaded_libraries: vec![],
        object_threadgroup_size_is_multiple_of_thread_execution_width: false,
        object_threads_per_threadgroup: size0(),
        open_gl_mode_enabled: false,
        pipeline_library: false,
        pipeline_memory_length: 0,
        plugin_data: None,
        point_coord_lower_left: false,
        point_size_output_vs: false,
        point_smooth_enabled: false,
        post_vertex_dump_buffer_index: 0,
        profile_control: None,
        raster_sample_count: 1,
        rasterization_enabled: true,
        resource_index: 0,
        sample_count: 1,
        sample_coverage: 1.0,
        sample_coverage_invert: false,
        sample_mask: 0xffff_ffff,
        stencil_attachment_pixel_format: 0,
        support_adding_fragment_binary_functions: false,
        support_adding_mesh_binary_functions: false,
        support_adding_object_binary_functions: false,
        support_adding_vertex_binary_functions: false,
        support_indirect_command_buffers: false,
        tessellation_control_point_index_type: 0,
        tessellation_factor_format: 0,
        tessellation_factor_scale_enabled: false,
        tessellation_factor_step_function: 0,
        tessellation_output_winding_order: 0,
        tessellation_partition_mode: 0,
        texture_write_rounding_mode: 0,
        two_side_enabled: false,
        vertex_amplification_mode: 0,
        vertex_buffers: vec![],
        vertex_depth_compare_clamp_mask: 0,
        vertex_descriptor: None,
        vertex_function: None,
        vertex_linked_functions: None,
        vertex_preloaded_libraries: vec![],
        mesh_buffers: vec![],
        object_buffers: vec![],
    }
}

pub fn compute_pipeline_descriptor() -> ComputePipelineDescriptor {
    ComputePipelineDescriptor {
        binary_archives: vec![],
        binary_libraries: vec![],
        buffers: vec![],
        compute_function: None,
        driver_compiler_options: None,
        dynamic_libraries: vec![],
        force_resource_index: false,
        function_pointers: vec![],
        gpu_compiler_spi_options: None,
        insert_libraries: vec![],
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
        preloaded_libraries: vec![],
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

/// `RVertex` of `RenderSupport.m` (48 bytes).
#[derive(Clone, Copy, Debug, Default)]
pub struct RVertex {
    pub position: [f32; 4],
    pub data: [u32; 4],
    pub extra: [f32; 4],
}

impl RVertex {
    /// `RVertexAt`: a vertex at pixel (px, py) of a width x height target, depth z, carrying id; point size 1.
    pub fn at(px: f64, py: f64, z: f64, width: u64, height: u64, id: u32) -> RVertex {
        RVertex {
            position: [(px / width as f64 * 2.0 - 1.0) as f32, (1.0 - py / height as f64 * 2.0) as f32, z as f32, 1.0],
            data: [id, 0, 0, 0],
            extra: [1.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn bytes(vs: &[RVertex]) -> Vec<u8> {
        let mut out = Vec::new();
        for v in vs {
            for f in v.position {
                out.extend(f.to_le_bytes());
            }
            for d in v.data {
                out.extend(d.to_le_bytes());
            }
            for f in v.extra {
                out.extend(f.to_le_bytes());
            }
        }
        out
    }
}

/// `RVertexBuffer`.
pub fn vertex_buffer(h: &Harness, vs: &[RVertex]) -> Handle {
    h.buffer(&RVertex::bytes(vs)).0
}

/// `RPaintTriangles` of `RenderSupport.m`: paint triangles by pixel-centre containment; `None` if a centre lies within
/// `margin` of an edge of a triangle whose bounding box holds it.
pub fn paint_triangles(triangles: &[[f64; 6]], ids: &[u32], w: usize, h: usize, margin: f64) -> Option<Vec<u32>> {
    let edge = |a: [f64; 2], b: [f64; 2], x: f64, y: f64| {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        ((x - a[0]) * dy - (y - a[1]) * dx) / (dx * dx + dy * dy).sqrt()
    };
    let mut map = vec![0u32; w * h];
    for (t, v) in triangles.iter().enumerate() {
        let (a, b, c) = ([v[0], v[1]], [v[2], v[3]], [v[4], v[5]]);
        for y in 0..h {
            for x in 0..w {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let (min_x, max_x) = (a[0].min(b[0].min(c[0])), a[0].max(b[0].max(c[0])));
                let (min_y, max_y) = (a[1].min(b[1].min(c[1])), a[1].max(b[1].max(c[1])));
                if px < min_x - margin || px > max_x + margin || py < min_y - margin || py > max_y + margin {
                    continue;
                }
                let (e0, e1, e2) = (edge(a, b, px, py), edge(b, c, px, py), edge(c, a, px, py));
                if e0.abs() < margin || e1.abs() < margin || e2.abs() < margin {
                    let outside =
                        (e0 < -margin || e1 < -margin || e2 < -margin) && (e0 > margin || e1 > margin || e2 > margin);
                    if !outside {
                        return None;
                    }
                    continue;
                }
                if (e0 > 0.0 && e1 > 0.0 && e2 > 0.0) || (e0 < 0.0 && e1 < 0.0 && e2 < 0.0) {
                    map[y * w + x] = ids[t];
                }
            }
        }
    }
    Some(map)
}

/// `RTrianglesToVertices`.
pub fn triangle_vertices(triangles: &[[f64; 6]], ids: &[u32], w: u64, h: u64) -> Vec<RVertex> {
    let mut out = Vec::new();
    for (t, v) in triangles.iter().enumerate() {
        for k in 0..3 {
            out.push(RVertex::at(v[2 * k], v[2 * k + 1], 0.0, w, h, ids[t]));
        }
    }
    out
}

pub fn words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

pub fn from_words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn set_pipeline(p: Handle) -> Command {
    Command::RenderSetRenderPipelineState(RenderSetRenderPipelineState { pipeline_state: Some(p) })
}

pub fn set_vertex_buffer(b: Handle, offset: u64, index: u64) -> Command {
    Command::RenderSetVertexBuffer(RenderSetVertexBuffer { buffer: Some(b), offset, index })
}

pub fn draw(primitive: u64, start: u64, count: u64) -> Command {
    Command::RenderDrawPrimitives(RenderDrawPrimitives {
        primitive_type: primitive,
        vertex_start: start,
        vertex_count: count,
    })
}

pub fn scissor(x: u64, y: u64, w: u64, h: u64) -> Command {
    Command::RenderSetScissorRect(RenderSetScissorRect { rect: ScissorRect { x, y, width: w, height: h } })
}

pub fn set_depth_stencil(s: Handle) -> Command {
    Command::RenderSetDepthStencilState(RenderSetDepthStencilState { depth_stencil_state: Some(s) })
}

/// `MTLPrimitiveType` values.
pub mod prim {
    pub const POINT: u64 = 0;
    pub const LINE: u64 = 1;
    pub const LINE_STRIP: u64 = 2;
    pub const TRIANGLE: u64 = 3;
    pub const TRIANGLE_STRIP: u64 = 4;
}
