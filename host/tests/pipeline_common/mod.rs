//! The executor with the memory, pipeline, render, compute and blit services on a native WebGPU implementation
//! (`WEBGPU_NATIVE`, see `host/build.rs`), the reference shader library compiled from `tests/reference/shaders`, and
//! the requests a guest sends to register its functions and create pipelines.
//!
//! The reference library is the one `tests/reference/build.py` builds for iOS (`air64-apple-ios17.0`, the AIR the
//! A10X's Metal.framework hands its driver): `WEBMETAL_REFERENCE_BUILD` names a build directory that holds
//! `wmref.metallib`, or else the tests build one under Cargo's target directory (macOS with Xcode's Metal toolchain).

#![allow(dead_code)]

use std::cell::Cell;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use super::mem_common::{Api, Gpu, Guest, TASK};
use webmetal_host::blit::BlitService;
use webmetal_host::compute::ComputeService;
use webmetal_host::exec::failure;
use webmetal_host::exec::schema::{Role, command_handles};
use webmetal_host::exec::{Config, ControlService, Executor, Services};
use webmetal_host::gpu::Gpu as _;
use webmetal_host::gpu::scope::{Granularity, Policy};
use webmetal_host::mem::Memory;
use webmetal_host::mem::service::MemoryServices;
use webmetal_host::pipeline::sha256::sha256;
use webmetal_host::pipeline::{self, PipelineServices, Shared};
use webmetal_host::render::RenderServices;
use webmetal_wire::*;

pub type A = Api;
pub type E = Guest;

/// One function of the reference library: its module's bytes as the metallib stores them, and its type.
#[derive(Clone)]
pub struct LibraryFunction {
    pub bitcode: Vec<u8>,
    /// `MTLFunctionType`.
    pub function_type: u64,
}

/// The reference library's build directory: `WEBMETAL_REFERENCE_BUILD`, or an iOS build made once per test process
/// (the tests of one binary run on parallel threads and must not build into the same directory at the same time).
fn reference_build() -> &'static PathBuf {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| match std::env::var_os("WEBMETAL_REFERENCE_BUILD") {
        Some(d) => PathBuf::from(d),
        None => {
            let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("reference-ios");
            if !d.join("wmref.metallib").is_file() {
                let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/reference/build.py");
                let ok = std::process::Command::new("python3")
                    .args(["-B", "-I"])
                    .arg(&script)
                    .args(["--platform", "ios", "--out"])
                    .arg(&d)
                    .status()
                    .expect("python3 runs tests/reference/build.py");
                assert!(
                    ok.success(),
                    "tests/reference/build.py --platform ios failed; set WEBMETAL_REFERENCE_BUILD to a build directory"
                );
            }
            d
        }
    })
}

/// The functions of the reference library by name.
pub fn reference_library() -> HashMap<String, LibraryFunction> {
    let dir = reference_build();
    let data = std::fs::read(dir.join("wmref.metallib")).expect("wmref.metallib");
    let lib = airwgsl::parse_metallib(&data).expect("a metallib");
    let mut out = HashMap::new();
    for f in &lib.functions {
        let Some(module) = f.module else { continue };
        // METALLIB function types: vertex 0, fragment 1, compute 2 (airwgsl::mtlb); MTLFunctionType: 1, 2, 3.
        let function_type = match f.function_type {
            0 => 1,
            1 => 2,
            2 => 3,
            _ => continue,
        };
        out.insert(f.name.clone(), LibraryFunction { bitcode: module.to_vec(), function_type });
    }
    out
}

/// `MTLComputePipelineDescriptor` as a new descriptor reads on macOS 27.0.1 (the defaults `MTLComputePipeline.h`
/// states, and the SPI state of a new `MTLComputePipelineDescriptorInternal`), with `function`.
pub fn compute_descriptor(function: Handle) -> ComputePipelineDescriptor {
    ComputePipelineDescriptor {
        binary_archives: Vec::new(),
        binary_libraries: Vec::new(),
        buffers: (0..31).map(|_| PipelineBufferDescriptor { mutability: 0 }).collect(),
        compute_function: Some(function),
        driver_compiler_options: None,
        dynamic_libraries: Vec::new(),
        force_resource_index: false,
        function_pointers: Vec::new(),
        gpu_compiler_spi_options: None,
        insert_libraries: Vec::new(),
        label: None,
        linked_functions: empty_linked(),
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
        stage_input_descriptor: Some(StageInputOutputDescriptor {
            attributes: Vec::new(),
            index_buffer_index: 0,
            // MTLIndexTypeUInt16.
            index_type: 0,
            layouts: Vec::new(),
        }),
        support_adding_binary_functions: false,
        support_indirect_command_buffers: false,
        texture_write_fp_rounding_mode: 0,
        texture_write_rounding_mode: 0,
        thread_group_size_is_multiple_of_thread_execution_width: false,
    }
}

fn empty_linked() -> Option<LinkedFunctions> {
    Some(LinkedFunctions {
        binary_functions: Vec::new(),
        functions: Vec::new(),
        groups: Vec::new(),
        private_functions: Vec::new(),
    })
}

/// A new `MTLRenderPipelineColorAttachmentDescriptor` (the defaults `MTLRenderPipeline.h` states).
pub fn color_attachment(pixel_format: u64) -> RenderPipelineColorAttachment {
    RenderPipelineColorAttachment {
        alpha_blend_operation: 0,
        blending_enabled: false,
        destination_alpha_blend_factor: 0,
        destination_rgb_blend_factor: 0,
        pixel_format,
        rgb_blend_operation: 0,
        source_alpha_blend_factor: 1,
        source_rgb_blend_factor: 1,
        write_mask: 0xf,
    }
}

/// `MTLRenderPipelineDescriptor` as a new descriptor reads on macOS 27.0.1 (`pipeline.descriptor-defaults` and the SPI
/// state of a new `MTLRenderPipelineDescriptorInternal`), with the functions, no vertex descriptor set, and color
/// attachment 0 of `format` (0: none).
pub fn render_descriptor(vertex: Option<Handle>, fragment: Option<Handle>, format: u64) -> RenderPipelineDescriptor {
    let buffers = || (0..31).map(|_| PipelineBufferDescriptor { mutability: 0 }).collect::<Vec<_>>();
    let zero = Size { width: 0, height: 0, depth: 0 };
    let mut colors: Vec<RenderPipelineColorAttachment> = (0..8).map(|_| color_attachment(0)).collect();
    colors[0].pixel_format = format;
    RenderPipelineDescriptor {
        alpha_test_enabled: false,
        alpha_test_function: 0,
        alpha_to_coverage_enabled: false,
        alpha_to_one_enabled: false,
        binary_archives: Vec::new(),
        clip_distance_enable_mask: 0xff,
        color_attachments: colors,
        color_sample_count: 0,
        depth_attachment_pixel_format: 0,
        depth_stencil_write_disabled: false,
        driver_compiler_options: None,
        explicit_visibility_group_id: 0xffff_ffff,
        force_resource_index: false,
        force_software_vertex_fetch: false,
        fragment_buffers: buffers(),
        fragment_depth_compare_clamp_mask: 0,
        fragment_function: fragment,
        fragment_linked_functions: empty_linked(),
        fragment_preloaded_libraries: Vec::new(),
        gpu_compiler_spi_options: None,
        input_primitive_topology: 0,
        insert_fragment_libraries: Vec::new(),
        insert_mesh_libraries: Vec::new(),
        insert_object_libraries: Vec::new(),
        insert_vertex_libraries: Vec::new(),
        label: None,
        logic_operation: 0,
        logic_operation_enabled: false,
        max_acceleration_structure_traversal_depth: 0,
        max_fragment_call_stack_depth: 1,
        max_mesh_call_stack_depth: 1,
        max_object_call_stack_depth: 1,
        max_pipeline_children: zero.clone(),
        max_tessellation_factor: 16,
        max_total_threads_per_mesh_threadgroup: 0,
        max_total_threads_per_object_threadgroup: 0,
        max_vertex_amplification_count: 1,
        max_vertex_call_stack_depth: 1,
        mesh_function: None,
        mesh_linked_functions: empty_linked(),
        mesh_preloaded_libraries: Vec::new(),
        mesh_threadgroup_size_is_multiple_of_thread_execution_width: false,
        mesh_threads_per_threadgroup: zero.clone(),
        needs_custom_border_color_samplers: false,
        object_function: None,
        object_linked_functions: empty_linked(),
        object_preloaded_libraries: Vec::new(),
        object_threadgroup_size_is_multiple_of_thread_execution_width: false,
        object_threads_per_threadgroup: zero,
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
        sample_mask: u64::MAX,
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
        vertex_buffers: buffers(),
        vertex_depth_compare_clamp_mask: 0,
        vertex_descriptor: None,
        vertex_function: vertex,
        vertex_linked_functions: empty_linked(),
        vertex_preloaded_libraries: Vec::new(),
        mesh_buffers: buffers(),
        object_buffers: buffers(),
    }
}

pub struct Harness {
    pub g: Gpu,
    pub guest: Rc<Guest>,
    pub exec: Executor<A, E>,
    pub memory: Rc<Memory<A>>,
    pub shared: Rc<Shared<A>>,
    next: Cell<u64>,
    alloc: Cell<u64>,
    queue: Cell<Option<Handle>>,
    registered: std::cell::RefCell<BTreeSet<[u8; 32]>>,
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
        // wgpu-native does not implement wgslLanguageFeatures; an implementation that does not report its extensions
        // is taken to offer none, so a pipeline that needs one is refused rather than assumed to work.
        let wgsl = match g.gpu.wgsl_language_features() {
            Ok(f) => f,
            Err(e) if e.kind == webmetal_host::gpu::ExceptionKind::Unsupported => Default::default(),
            Err(e) => panic!("wgslLanguageFeatures: {e}"),
        };
        let pipelines = PipelineServices::<A>::new(&g.device, wgsl, pipeline::Config::default());
        let shared = pipelines.shared();
        let render = RenderServices::new(memory.clone());
        let services = Services {
            control: vec![svc.control() as Box<dyn ControlService<A, E>>, pipelines.control(), render.control()],
            render: Some(render.encoder()),
            compute: Some(Box::new(ComputeService::new(memory.clone()))),
            blit: Some(Box::new(BlitService::new(memory.clone()))),
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
            shared,
            next: Cell::new(0x100),
            alloc: Cell::new(base + 0x1000),
            queue: Cell::new(None),
            registered: Default::default(),
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
        assert_eq!(f.status, status::OK, "{}: {f:?}", self.name());
    }

    /// `RegisterAir` once per digest, as the guest sends it.
    pub fn register(&self, bitcode: &[u8]) -> [u8; 32] {
        let hash = sha256(bitcode);
        if self.registered.borrow_mut().insert(hash) {
            self.ok(Request::RegisterAir(RegisterAir { hash: Hash { bytes: hash }, bitcode: bitcode.to_vec() }));
        }
        hash
    }

    /// `RegisterFunction` for `name` of the module `bitcode`.
    pub fn function(&self, bitcode: &[u8], name: &str, function_type: u64) -> Handle {
        let hash = self.register(bitcode);
        let f = self.handle();
        self.ok(Request::RegisterFunction(RegisterFunction {
            function: f,
            air: Hash { bytes: hash },
            name: name.into(),
            function_type,
            constants: Vec::new(),
        }));
        f
    }

    /// `CreateComputePipeline` with `descriptor`: the reply.
    pub fn compute_pipeline(&self, pipeline: Handle, descriptor: ComputePipelineDescriptor, options: u64) -> Reply {
        self.request(Request::CreateComputePipeline(CreateComputePipeline {
            pipeline,
            descriptor,
            options,
            reflection: true,
            error: true,
        }))
    }

    /// `CreateRenderPipelineReflection` with `descriptor` and reflection asked for: the reply.
    pub fn render_pipeline(&self, pipeline: Handle, descriptor: RenderPipelineDescriptor, options: u64) -> Reply {
        self.request(Request::CreateRenderPipelineReflection(CreateRenderPipelineReflection {
            pipeline,
            descriptor,
            options,
            reflection: true,
            error: true,
        }))
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

    pub fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        self.guest.cpu_read(addr, len)
    }

    /// Commit `records` as one command buffer and wait until it completes: its completion failure (status 0 when it
    /// completed normally) and the failed record's offset.
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
            protection_options: 0,
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

    /// One compute encoder with `body`: its command buffer's completion.
    pub fn compute(&self, body: Vec<Command>) -> (Failure, u64) {
        let e = self.handle();
        let mut v = vec![Command::BeginCompute(BeginCompute { encoder: e })];
        v.extend(body);
        v.push(Command::EncoderEndEncoding(EncoderEndEncoding {}));
        self.run(&v)
    }

    /// Fails if WebGPU reported an error outside every scope.
    pub fn assert_no_errors(&self) {
        self.g.assert_no_errors();
        let reports = self.exec.take_reports();
        // Reports are failures the executor answered with; uncaptured errors would have panicked under strict mode.
        let _ = reports;
    }
}
