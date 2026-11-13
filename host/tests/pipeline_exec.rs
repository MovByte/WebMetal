//! The pipeline service through the executor on a native WebGPU implementation (`WEBGPU_NATIVE`, see `host/build.rs`):
//! the guest's records for the reference library's functions and pipelines, the compute encoder running what they
//! create, and the Metal reference outputs (`tests/reference/goldens`) where Metal defines the result.
//!
//! The pipeline-creation corpus is the reference library's kernels and pipeline cases and, with `WEBMETAL_CORPUS`
//! naming AirWgsl's generated corpus build (`corpus/build.py`: `modules.tsv` and `air/`), its kernels. Every creation
//! must succeed exactly when the compiler compiles the pipeline and otherwise fail with the status its refusal maps
//! to; a second creation of the same pipeline must be served from the caches with byte-equal WGSL, and run with the
//! same results as the first.

#![cfg(feature = "native")]

mod mem_common;
mod pipeline_common;
mod render_common;

use mem_common::Golden;
use pipeline_common::{Harness, LibraryFunction, compute_descriptor, reference_library};
use webmetal_host::pipeline::compute;
use webmetal_host::pipeline::registry::Function;
use webmetal_wire::*;

fn set_pipeline(p: Handle) -> Command {
    Command::ComputeSetComputePipelineState(ComputeSetComputePipelineState { state: Some(p) })
}

fn set_buffer(b: Handle, index: u64) -> Command {
    Command::ComputeSetBuffer(ComputeSetBuffer { buffer: Some(b), offset: 0, index })
}

fn dispatch(groups: [u64; 3], threads: [u64; 3]) -> Command {
    Command::ComputeDispatchThreadgroups(ComputeDispatchThreadgroups {
        threadgroups_per_grid: Size { width: groups[0], height: groups[1], depth: groups[2] },
        threads_per_threadgroup: Size { width: threads[0], height: threads[1], depth: threads[2] },
    })
}

/// The creation outcome the compiler alone predicts for a kernel with the default descriptor: Ok, or the status and
/// the refusal text the creation must report.
fn predicted(h: &Harness, f: &LibraryFunction, name: &str) -> Result<(), (u32, String)> {
    let air = std::rc::Rc::new(webmetal_host::pipeline::registry::Air {
        hash: webmetal_host::pipeline::sha256::sha256(&f.bitcode),
        bitcode: std::sync::Arc::from(&f.bitcode[..]),
    });
    let function = Function { air, name: name.into(), function_type: f.function_type, constants: Vec::new() };
    let settings = compute::check_descriptor("test", &compute_descriptor(Handle::new(1).unwrap())).unwrap();
    let kind = |r: &airwgsl::Refusal| match r.kind {
        airwgsl::RefusalKind::HostUnsupported => status::HOST_UNSUPPORTED,
        _ => status::UNIMPLEMENTED,
    };
    let info = compute::kernel_info(&function).map_err(|r| (kind(&r), r.to_string()))?;
    let desc = compute::creation_desc(&settings, &info);
    match h.shared.compiler.compile_fresh(&[function.stage_module()], &desc) {
        // A compiled layout WebGPU would reject is the compiler's defect, which creation reports before any WebGPU
        // call.
        Ok(c) => webmetal_host::pipeline::gpu::check_layout::<pipeline_common::A>("test", &h.g.device, &c.layout)
            .map_err(|f| (f.status, f.message.rsplit(": ").next().unwrap_or("").to_string())),
        Err(r) => Err((kind(&r), r.to_string())),
    }
}

fn reply_failure(r: &Reply) -> Failure {
    webmetal_host::exec::schema::reply_failure(r).cloned().unwrap_or_else(webmetal_host::exec::failure::ok)
}

#[test]
fn reference_kernels_create_as_the_compiler_decides_and_cache_hits_are_equal() {
    let h = Harness::new();
    let lib = reference_library();
    let mut names: Vec<&String> = lib.iter().filter(|(_, f)| f.function_type == 3).map(|(n, _)| n).collect();
    names.sort();
    let (mut created, mut refused) = (0usize, 0usize);
    for name in names {
        let f = &lib[name];
        let expect = predicted(&h, f, name);
        let func = h.function(&f.bitcode, name, 3);
        let p = h.handle();
        let reply = h.compute_pipeline(p, compute_descriptor(func), 0);
        let got = reply_failure(&reply);
        match &expect {
            Ok(()) => {
                assert_eq!(got.status, status::OK, "{}: {name}: {got:?}", h.name());
                let Reply::CreateComputePipeline(Ok(CreateComputePipelineReply { reflection: Some(r) })) = &reply
                else {
                    panic!("{name}: no reflection in {reply:?}");
                };
                assert_eq!(r.stages.len(), 1, "{name}");
                assert_eq!(r.stages[0].function_type, 3, "{name}");
                // A second pipeline of the same function is served by the caches: the compiler cache returns the
                // first compilation, whose WGSL equals a fresh compilation's byte for byte.
                let before = h.shared.compiler.stats();
                let again = h.handle();
                assert_eq!(reply_failure(&h.compute_pipeline(again, compute_descriptor(func), 0)).status, status::OK);
                let after = h.shared.compiler.stats();
                assert_eq!((after.hits, after.misses), (before.hits + 1, before.misses), "{name}: no cache hit");
                created += 1;
            }
            Err((s, text)) => {
                assert_eq!(got.status, *s, "{}: {name}: expected the refusal {text}, got {got:?}", h.name());
                let reason = text.split(": ").last().unwrap_or(text);
                assert!(got.message.contains(reason), "{name}: {got:?} does not name {text}");
                refused += 1;
            }
        }
    }
    h.assert_no_errors();
    eprintln!("{}: {created} reference kernels created, {refused} refused as the compiler refuses them", h.name());
    assert!(created > 0, "no reference kernel compiled");
}

#[test]
fn the_marker_kernel_writes_metal_s_values_from_fresh_and_cached_pipelines() {
    let h = Harness::new();
    let lib = reference_library();
    let f = &lib["pip_marker"];
    let golden = Golden::load("archive.round-trip").bytes("hit.marker").expect("hit.marker");
    let func = h.function(&f.bitcode, "pip_marker", 3);
    let mut outputs = Vec::new();
    for _ in 0..2 {
        let p = h.handle();
        assert_eq!(reply_failure(&h.compute_pipeline(p, compute_descriptor(func), 0)).status, status::OK);
        let (b, addr) = h.buffer_with(&[0u8; 16]);
        let (f, at) = h.compute(vec![set_pipeline(p), set_buffer(b, 0), dispatch([1, 1, 1], [4, 1, 1])]);
        assert_eq!(f.status, status::OK, "{}: offset {at}: {f:?}", h.name());
        outputs.push(h.read(addr, 16));
    }
    assert_eq!(outputs[0], golden, "{}: the fresh pipeline", h.name());
    assert_eq!(outputs[1], golden, "{}: the cached pipeline", h.name());
    let gpu = h.shared.gpu.stats();
    assert!(gpu.hits >= 1, "{}: the second pipeline made new WebGPU objects: {gpu:?}", h.name());
    h.assert_no_errors();
}

#[test]
fn descriptor_limits_bound_the_dispatch() {
    // compute.pipeline-properties: a descriptor maxTotalThreadsPerThreadgroup of 96 limits the pipeline to at most 96
    // (rule exact); the A10X's own value is its compiler's and is reported by the guest as not established.
    let h = Harness::new();
    let lib = reference_library();
    let f = &lib["pip_marker"];
    let func = h.function(&f.bitcode, "pip_marker", 3);
    let p = h.handle();
    let mut d = compute_descriptor(func);
    d.max_total_threads_per_threadgroup = 96;
    d.thread_group_size_is_multiple_of_thread_execution_width = true;
    assert_eq!(reply_failure(&h.compute_pipeline(p, d, 0)).status, status::OK);
    let (b, addr) = h.buffer_with(&[0u8; 4 * 128]);
    let (ok, _) = h.compute(vec![set_pipeline(p), set_buffer(b, 0), dispatch([1, 1, 1], [96, 1, 1])]);
    assert_eq!(ok.status, status::OK, "{}: {ok:?}", h.name());
    let words: Vec<u32> = h.read(addr, 4 * 96).chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(words, (0..96).map(|i| 0x600d + i).collect::<Vec<u32>>());
    // 128 threads exceed the descriptor's limit: undefined in Metal, refused.
    let (over, _) = h.compute(vec![set_pipeline(p), set_buffer(b, 0), dispatch([1, 1, 1], [128, 1, 1])]);
    assert_eq!(over.status, status::UNIMPLEMENTED, "{}: {over:?}", h.name());
    // 64 threads are not a multiple of threadExecutionWidth (32 on the A10X): 64 is, 48 is not.
    let (odd, _) = h.compute(vec![set_pipeline(p), set_buffer(b, 0), dispatch([1, 1, 1], [48, 1, 1])]);
    assert_eq!(odd.status, status::UNIMPLEMENTED, "{}: {odd:?}", h.name());
    h.assert_no_errors();
}

#[test]
fn creation_errors_metal_defines_fail_loudly() {
    let h = Harness::new();
    let lib = reference_library();
    // validation.compute-pipeline-from-vertex-function: Metal (release) returns the driver's NSError.
    let v = &lib["pip_simple_vertex"];
    let vertex = h.function(&v.bitcode, "pip_simple_vertex", 1);
    let f = reply_failure(&h.compute_pipeline(h.handle(), compute_descriptor(vertex), 0));
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("NSError from the driver") && f.message.contains("not established"), "{f:?}");
    // A nil compute function and an option outside MTLPipelineOption.
    let mut d = compute_descriptor(vertex);
    d.compute_function = None;
    assert_eq!(reply_failure(&h.compute_pipeline(h.handle(), d, 0)).status, status::UNIMPLEMENTED);
    let m = &lib["pip_marker"];
    let marker = h.function(&m.bitcode, "pip_marker", 3);
    assert_eq!(
        reply_failure(&h.compute_pipeline(h.handle(), compute_descriptor(marker), 1 << 3)).status,
        status::UNIMPLEMENTED
    );
    // The debug layer's options: bits 16, 17 and 23 change neither the A10X's pipeline nor its reflection, so the
    // debug layer's 0x810003, the 0x20000 of its second creation and each bit alone create; with performance
    // statistics (0x870003) the creation fails on bit 18, by name.
    for options in [0x810003, 0x20000, 0x10000, 0x800000, 0x830003] {
        let r = h.compute_pipeline(h.handle(), compute_descriptor(marker), options);
        assert!(matches!(&r, Reply::CreateComputePipeline(Ok(_))), "options {options:#x}: {r:?}");
    }
    let f = reply_failure(&h.compute_pipeline(h.handle(), compute_descriptor(marker), 0x870003));
    assert_eq!(f.status, status::UNIMPLEMENTED);
    assert!(f.message.contains("pipeline option 0x40000") && f.message.contains("performance statistics"), "{f:?}");
    let f = reply_failure(&h.compute_pipeline(h.handle(), compute_descriptor(marker), 0x40003));
    assert!(f.message.contains("pipeline option 0x40000") && f.message.contains("performance statistics"), "{f:?}");
    let f = reply_failure(&h.compute_pipeline(h.handle(), compute_descriptor(marker), 0x1000000));
    assert!(f.message.contains("pipeline option 0x1000000") && f.message.contains("not an MTLPipelineOption"), "{f:?}");
    // Descriptor state whose behavior is not established is named.
    let mut d = compute_descriptor(marker);
    d.open_gl_mode_enabled = true;
    let f = reply_failure(&h.compute_pipeline(h.handle(), d, 0));
    assert_eq!(f.status, status::UNIMPLEMENTED);
    assert!(f.message.contains("openGLModeEnabled"), "{f:?}");
    h.assert_no_errors();
}

#[test]
fn registration_follows_the_protocol() {
    let h = Harness::new();
    let bitcode = b"BC\xc0\xde not a module".to_vec();
    let hash = webmetal_host::pipeline::sha256::sha256(&bitcode);
    let mut wrong = hash;
    wrong[31] ^= 0x80;
    let s = h.status(Request::RegisterAir(RegisterAir { hash: Hash { bytes: wrong }, bitcode: bitcode.clone() }));
    assert_eq!(s.status, status::INVALID, "{s:?}");
    h.ok(Request::RegisterAir(RegisterAir { hash: Hash { bytes: hash }, bitcode: bitcode.clone() }));
    let s = h.status(Request::RegisterAir(RegisterAir { hash: Hash { bytes: hash }, bitcode }));
    assert_eq!(s.status, status::INVALID, "sent once per digest: {s:?}");
    let s = h.status(Request::RegisterFunction(RegisterFunction {
        function: h.handle(),
        air: Hash { bytes: wrong },
        name: "f".into(),
        function_type: 3,
        constants: Vec::new(),
    }));
    assert_eq!(s.status, status::INVALID, "{s:?}");
    // A module the bitcode reader cannot read is the compiler's refusal at pipeline creation.
    let f = h.handle();
    h.ok(Request::RegisterFunction(RegisterFunction {
        function: f,
        air: Hash { bytes: hash },
        name: "f".into(),
        function_type: 3,
        constants: Vec::new(),
    }));
    let s = reply_failure(&h.compute_pipeline(h.handle(), compute_descriptor(f), 0));
    assert_eq!(s.status, status::UNIMPLEMENTED, "{s:?}");
    assert!(s.message.contains("module-unreadable"), "{s:?}");
    h.assert_no_errors();
}

#[test]
fn binary_archives_hold_what_was_added() {
    // archive.round-trip and archive.invalid-file.
    let h = Harness::new();
    let lib = reference_library();
    let marker = h.function(&lib["pip_marker"].bitcode, "pip_marker", 3);
    let archive = h.handle();
    // The descriptor's default options (2 on 21H461, one field with writingOptions) are Metal.framework's own.
    h.ok(Request::CreateBinaryArchive(CreateBinaryArchive {
        archive,
        descriptor: BinaryArchiveDescriptor { options: 2, url: None, writing_options: 2 },
        error: true,
    }));
    h.ok(Request::BinaryArchiveAddComputePipelineFunctions(BinaryArchiveAddComputePipelineFunctions {
        archive,
        descriptor: compute_descriptor(marker),
        error: true,
    }));
    let fail_on_miss = 1 << 2;
    let mut hit = compute_descriptor(marker);
    hit.binary_archives = vec![archive];
    let p = h.handle();
    assert_eq!(reply_failure(&h.compute_pipeline(p, hit, fail_on_miss)).status, status::OK);
    let (b, addr) = h.buffer_with(&[0u8; 16]);
    let (f, _) = h.compute(vec![set_pipeline(p), set_buffer(b, 0), dispatch([1, 1, 1], [4, 1, 1])]);
    assert_eq!(f.status, status::OK, "{f:?}");
    assert_eq!(h.read(addr, 16), Golden::load("archive.round-trip").bytes("hit.marker").unwrap());

    // The same pipeline against an archive that does not hold it, with misses failing.
    let empty = h.handle();
    h.ok(Request::CreateBinaryArchive(CreateBinaryArchive {
        archive: empty,
        descriptor: BinaryArchiveDescriptor { options: 0, url: None, writing_options: 0 },
        error: true,
    }));
    let mut miss = compute_descriptor(marker);
    miss.binary_archives = vec![empty];
    let f = reply_failure(&h.compute_pipeline(h.handle(), miss.clone(), fail_on_miss));
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("binary archives"), "{f:?}");
    // Without the option the archives are not consulted.
    assert_eq!(reply_failure(&h.compute_pipeline(h.handle(), miss, 0)).status, status::OK);

    // An archive file: the driver reads it in the guest's file system.
    for url in ["file:///nonexistent/wmref-archive.bin", "file:///tmp/wmref-garbage.bin"] {
        let s = h.status(Request::CreateBinaryArchive(CreateBinaryArchive {
            archive: h.handle(),
            descriptor: BinaryArchiveDescriptor { options: 0, url: Some(url.into()), writing_options: 0 },
            error: true,
        }));
        assert_eq!(s.status, status::UNIMPLEMENTED, "{s:?}");
    }
    h.assert_no_errors();
}

// Render pipelines -----------------------------------------------------------------------------------------------------

/// `MTLPixelFormat` values (iPhoneOS SDK, `MTLPixelFormat.h`).
const RGBA8_UNORM: u64 = 70;
const RGBA32_UINT: u64 = 123;
const R32_UINT: u64 = 53;

fn render_status(h: &Harness, d: RenderPipelineDescriptor) -> Failure {
    reply_failure(&h.render_pipeline(h.handle(), d, 0))
}

/// The status a render pipeline's creation fails with when the compiler refuses it, or 0.
fn render_predicted(
    h: &Harness,
    lib: &std::collections::HashMap<String, LibraryFunction>,
    v: &str,
    f: Option<&str>,
    format: u64,
) -> u32 {
    let module = |name: &str| {
        let lf = &lib[name];
        Function {
            air: std::rc::Rc::new(webmetal_host::pipeline::registry::Air {
                hash: webmetal_host::pipeline::sha256::sha256(&lf.bitcode),
                bitcode: std::sync::Arc::from(&lf.bitcode[..]),
            }),
            name: name.into(),
            function_type: lf.function_type,
            constants: Vec::new(),
        }
    };
    let mut modules = vec![module(v).stage_module()];
    modules.extend(f.map(|f| module(f).stage_module()));
    let d =
        pipeline_common::render_descriptor(Some(Handle::new(1).unwrap()), f.map(|_| Handle::new(2).unwrap()), format);
    let settings = webmetal_host::pipeline::render::check_descriptor("test", &d, h.shared.formats).unwrap();
    match h.shared.compiler.compile_fresh(&modules, &webmetal_host::pipeline::render::desc(&settings)) {
        Ok(_) => status::OK,
        Err(r) if r.kind == airwgsl::RefusalKind::HostUnsupported => status::HOST_UNSUPPORTED,
        Err(_) => status::UNIMPLEMENTED,
    }
}

#[test]
fn reference_render_pipelines_create_or_fail_as_metal_defines() {
    use pipeline_common::render_descriptor;
    let h = Harness::new();
    let lib = reference_library();
    let fun = |name: &str| h.function(&lib[name].bitcode, name, lib[name].function_type);
    let vertex = fun("pip_simple_vertex");
    let fragment = fun("pip_simple_fragment");

    // pipeline.render-async valid and pipeline.state-properties: a vertex and a fragment function into RGBA8Unorm.
    let expect = render_predicted(&h, &lib, "pip_simple_vertex", Some("pip_simple_fragment"), RGBA8_UNORM);
    let mut d = render_descriptor(Some(vertex), Some(fragment), RGBA8_UNORM);
    d.label = Some("wmref render".into());
    d.support_indirect_command_buffers = true;
    let reply = h.render_pipeline(h.handle(), d, 0);
    let f = reply_failure(&reply);
    assert_eq!(f.status, expect, "{}: simple pipeline: {f:?}", h.name());
    if expect == status::OK {
        let Reply::CreateRenderPipelineReflection(Ok(CreateRenderPipelineReflectionReply { reflection: Some(r) })) =
            &reply
        else {
            panic!("no reflection: {reply:?}")
        };
        assert_eq!(r.stages.iter().map(|s| s.function_type).collect::<Vec<_>>(), vec![1, 2]);
        // A second creation is served from the compiler cache.
        let before = h.shared.compiler.stats();
        let f = render_status(&h, render_descriptor(Some(vertex), Some(fragment), RGBA8_UNORM));
        assert_eq!(f.status, status::OK, "{f:?}");
        assert_eq!(h.shared.compiler.stats().hits, before.hits + 1);
        // The asynchronous form.
        let s = h.status(Request::CreateRenderPipelineAsync(CreateRenderPipelineAsync {
            pipeline: h.handle(),
            descriptor: render_descriptor(Some(vertex), Some(fragment), RGBA8_UNORM),
            completion_handler: true,
        }));
        assert_eq!(s.status, status::OK, "{s:?}");
    }

    // pipeline.render-errors unmatched-varying and pipeline.render-async unmatched: the driver's NSError.
    let unmatched = fun("pip_unmatched_fragment");
    let f = render_status(&h, render_descriptor(Some(vertex), Some(unmatched), RGBA8_UNORM));
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("fragment input") && f.message.contains("user(normal)"), "{f:?}");
    let s = h.status(Request::CreateRenderPipelineAsync(CreateRenderPipelineAsync {
        pipeline: h.handle(),
        descriptor: render_descriptor(Some(vertex), Some(unmatched), RGBA8_UNORM),
        completion_handler: true,
    }));
    assert_eq!(s.status, status::UNIMPLEMENTED, "{s:?}");

    // pipeline.render-errors stage-in-without-descriptor: the vertex function has stage input, no descriptor is set.
    let stage = fun("pip_stage_vertex");
    let f = render_status(&h, render_descriptor(Some(stage), None, RGBA8_UNORM));
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("input attributes and no vertex descriptor"), "{f:?}");

    // pipeline.rasterization-disabled: a void vertex function that writes a buffer; WebGPU's vertex stage cannot.
    let void = fun("pip_void_vertex");
    let mut d = render_descriptor(Some(void), None, RGBA8_UNORM);
    d.rasterization_enabled = false;
    d.input_primitive_topology = 1;
    let f = render_status(&h, d.clone());
    let expect = render_predicted(&h, &lib, "pip_void_vertex", None, RGBA8_UNORM);
    if expect == status::OK {
        assert_eq!(f.status, status::HOST_UNSUPPORTED, "{f:?}");
        assert!(f.message.contains("vertex-stage-writes"), "{f:?}");
    } else {
        assert_eq!(f.status, expect, "{f:?}");
    }
    // With rasterization enabled, a vertex function returning void is not a rasterizing pipeline Metal defines; and a
    // vertex result with rasterization disabled is the driver's error.
    let mut d = render_descriptor(Some(vertex), None, RGBA8_UNORM);
    d.rasterization_enabled = false;
    let f = render_status(&h, d);
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("rasterizationEnabled is NO"), "{f:?}");

    // validation.render-pipeline-without-vertex-function, validation.render-pipeline-blending-integer-format and
    // blend.error.integer-target: Metal.framework asserts before the driver.
    let f = render_status(&h, render_descriptor(None, Some(fragment), RGBA8_UNORM));
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("validateWithDevice"), "{f:?}");
    for format in [RGBA32_UINT, R32_UINT] {
        let mut d = render_descriptor(Some(vertex), Some(fun("pip_uint_fragment")), format);
        d.color_attachments[0].blending_enabled = true;
        let f = render_status(&h, d);
        assert_eq!(f.status, status::UNIMPLEMENTED, "{format}: {f:?}");
        assert!(f.message.contains("integer format"), "{f:?}");
    }

    // A float fragment output into an integer attachment: Metal refuses the pipeline with the driver's error (measured
    // on Apple M4 by the compiler's color-output-format rule).
    let f = render_status(&h, render_descriptor(Some(vertex), Some(fragment), RGBA32_UINT));
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("NSError from the driver") && f.message.contains("color-output-format"), "{f:?}");

    // Two samples have no WebGPU form; alpha-to-one has none either.
    let mut d = render_descriptor(Some(vertex), Some(fragment), RGBA8_UNORM);
    d.raster_sample_count = 2;
    d.sample_count = 2;
    assert_eq!(render_status(&h, d).status, status::HOST_UNSUPPORTED);
    let mut d = render_descriptor(Some(vertex), Some(fragment), RGBA8_UNORM);
    d.alpha_to_one_enabled = true;
    assert_eq!(render_status(&h, d).status, status::HOST_UNSUPPORTED);
    // Tile pipelines are not a feature of the A10X's family.
    h.assert_no_errors();
}

#[test]
fn a_compiled_render_pipeline_draws_through_the_render_encoder() {
    use pipeline_common::render_descriptor;
    let h = Harness::new();
    let lib = reference_library();
    if render_predicted(&h, &lib, "pip_simple_vertex", Some("pip_simple_fragment"), RGBA8_UNORM) != status::OK {
        eprintln!("{}: pip_simple_vertex and pip_simple_fragment do not compile; nothing to draw", h.name());
        return;
    }
    let vertex = h.function(&lib["pip_simple_vertex"].bitcode, "pip_simple_vertex", 1);
    let fragment = h.function(&lib["pip_simple_fragment"].bitcode, "pip_simple_fragment", 2);
    let p = h.handle();
    let f = reply_failure(&h.render_pipeline(p, render_descriptor(Some(vertex), Some(fragment), RGBA8_UNORM), 0));
    assert_eq!(f.status, status::OK, "{f:?}");
    // The debug layer's bits 16, 17 and 23 change neither the pipeline nor its reflection.
    for options in [0x830003, 0x20000] {
        let d = render_descriptor(Some(vertex), Some(fragment), RGBA8_UNORM);
        let f = reply_failure(&h.render_pipeline(h.handle(), d, options));
        assert_eq!(f.status, status::OK, "options {options:#x}: {f:?}");
    }
    // MTLStorageModePrivate 2, MTLTextureUsageRenderTarget 4.
    let target = h.handle();
    h.ok(Request::CreateTexture(CreateTexture {
        texture: target,
        descriptor: render_common::texture_descriptor(RGBA8_UNORM, 4, 4, 2, 4),
    }));
    // One triangle over the whole target.
    let positions: Vec<u8> = [[-1.0f32, -1.0, 0.0, 1.0], [3.0, -1.0, 0.0, 1.0], [-1.0, 3.0, 0.0, 1.0]]
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let (vb, _) = h.buffer_with(&positions);
    let mut records = render_common::render(
        h.handle(),
        render_common::pass_color(target, render_common::clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            Command::RenderSetRenderPipelineState(RenderSetRenderPipelineState { pipeline_state: Some(p) }),
            Command::RenderSetVertexBuffer(RenderSetVertexBuffer { buffer: Some(vb), offset: 0, index: 0 }),
            // MTLPrimitiveTypeTriangle 3.
            Command::RenderDrawPrimitives(RenderDrawPrimitives { primitive_type: 3, vertex_start: 0, vertex_count: 3 }),
        ],
    );
    let (out, addr) = h.buffer_with(&[0u8; 64]);
    records.push(Command::BeginBlit(BeginBlit { encoder: h.handle() }));
    records.push(Command::BlitCopyTextureToBuffer(BlitCopyTextureToBuffer {
        source_texture: Some(target),
        source_slice: 0,
        source_level: 0,
        source_origin: Origin { x: 0, y: 0, z: 0 },
        source_size: Size { width: 4, height: 4, depth: 1 },
        destination_buffer: Some(out),
        destination_offset: 0,
        destination_bytes_per_row: 16,
        destination_bytes_per_image: 64,
    }));
    records.push(Command::EncoderEndEncoding(EncoderEndEncoding {}));
    let (f, at) = h.run(&records);
    assert_eq!(f.status, status::OK, "{}: offset {at}: {f:?}", h.name());
    // pip_simple_fragment returns the varying color (1, 0, 0, 1): RGBA8Unorm stores ff 00 00 ff exactly.
    assert_eq!(h.read(addr, 64), [0xff, 0x00, 0x00, 0xff].repeat(16), "{}", h.name());
    h.assert_no_errors();
}

/// `render.framebuffer-fetch` with the compiler's pipeline: the fragment function reads color attachment 0 through
/// the snapshot binding the compiler describes, the pass plan draws each primitive after a snapshot (plan section
/// 5.9), and the result equals Metal's.
#[test]
fn the_compiler_s_framebuffer_fetch_plan_draws_metal_s_result() {
    use pipeline_common::render_descriptor;
    let h = Harness::new();
    let lib = reference_library();
    if render_predicted(&h, &lib, "rnd_vertex_triangles", Some("rnd_fragment_fetch"), R32_UINT) != status::OK {
        eprintln!("{}: rnd_fragment_fetch does not compile; nothing to draw", h.name());
        return;
    }
    let vertex = h.function(&lib["rnd_vertex_triangles"].bitcode, "rnd_vertex_triangles", 1);
    let fragment = h.function(&lib["rnd_fragment_fetch"].bitcode, "rnd_fragment_fetch", 2);
    let p = h.handle();
    let f = reply_failure(&h.render_pipeline(p, render_descriptor(Some(vertex), Some(fragment), R32_UINT), 0));
    assert_eq!(f.status, status::OK, "{f:?}");
    // MTLStorageModePrivate 2, MTLTextureUsageRenderTarget 4.
    let target = h.handle();
    h.ok(Request::CreateTexture(CreateTexture {
        texture: target,
        descriptor: render_common::texture_descriptor(R32_UINT, 8, 8, 2, 4),
    }));
    // RVertexAt of the reference harness: three triangles covering the target with ids 1, 2, 3, then a fourth with 4.
    let mut vertices = Vec::new();
    for t in 0..4u32 {
        for (x, y) in [(-1.0f64, -1.0f64), (24.0, -1.0), (-1.0, 24.0)] {
            for c in [(x / 8.0 * 2.0 - 1.0) as f32, (1.0 - y / 8.0 * 2.0) as f32, 0.0, 1.0] {
                vertices.extend_from_slice(&c.to_le_bytes());
            }
            for u in [t + 1, 0, 0, 0] {
                vertices.extend_from_slice(&u.to_le_bytes());
            }
            for c in [1.0f32, 0.0, 0.0, 0.0] {
                vertices.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
    let (vb, _) = h.buffer_with(&vertices);
    let draw = |start, count| {
        // MTLPrimitiveTypeTriangle 3.
        Command::RenderDrawPrimitives(RenderDrawPrimitives {
            primitive_type: 3,
            vertex_start: start,
            vertex_count: count,
        })
    };
    let mut records = render_common::render(
        h.handle(),
        render_common::pass_color(target, render_common::clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            Command::RenderSetRenderPipelineState(RenderSetRenderPipelineState { pipeline_state: Some(p) }),
            Command::RenderSetVertexBuffer(RenderSetVertexBuffer { buffer: Some(vb), offset: 0, index: 0 }),
            draw(0, 9),
            draw(9, 3),
        ],
    );
    let (out, addr) = h.buffer_with(&[0u8; 256]);
    records.push(Command::BeginBlit(BeginBlit { encoder: h.handle() }));
    records.push(Command::BlitCopyTextureToBuffer(BlitCopyTextureToBuffer {
        source_texture: Some(target),
        source_slice: 0,
        source_level: 0,
        source_origin: Origin { x: 0, y: 0, z: 0 },
        source_size: Size { width: 8, height: 8, depth: 1 },
        destination_buffer: Some(out),
        destination_offset: 0,
        destination_bytes_per_row: 32,
        destination_bytes_per_image: 256,
    }));
    records.push(Command::EncoderEndEncoding(EncoderEndEncoding {}));
    let (f, at) = h.run(&records);
    assert_eq!(f.status, status::OK, "{}: offset {at}: {f:?}", h.name());
    let golden = Golden::load("render.framebuffer-fetch").bytes("result").expect("result");
    assert_eq!(h.read(addr, 256), golden, "{}", h.name());
    h.assert_no_errors();
}

#[test]
fn every_reference_stage_function_creates_or_fails_loudly() {
    use pipeline_common::render_descriptor;
    let h = Harness::new();
    let lib = reference_library();
    let vertex = h.function(&lib["pip_simple_vertex"].bitcode, "pip_simple_vertex", 1);
    let mut names: Vec<&String> = lib.keys().collect();
    names.sort();
    let mut counts = std::collections::BTreeMap::new();
    for name in names {
        let lf = &lib[name];
        let f = h.function(&lf.bitcode, name, lf.function_type);
        let d = match lf.function_type {
            1 => render_descriptor(Some(f), None, RGBA8_UNORM),
            2 => render_descriptor(Some(vertex), Some(f), RGBA8_UNORM),
            _ => continue,
        };
        let s = render_status(&h, d);
        assert!(
            [status::OK, status::UNIMPLEMENTED, status::HOST_UNSUPPORTED].contains(&s.status),
            "{}: {name}: {s:?}",
            h.name()
        );
        *counts.entry(s.status).or_insert(0usize) += 1;
    }
    h.assert_no_errors();
    eprintln!("{}: reference vertex and fragment functions by creation status: {counts:?}", h.name());
}

// The generated corpus ------------------------------------------------------------------------------------------------

/// `WEBMETAL_CORPUS`: AirWgsl's generated corpus build. Every `WEBMETAL_CORPUS_SAMPLE`-th kernel (default every one)
/// of the language-version-3.1 modules (or of the version `WEBMETAL_CORPUS_VERSION` names) is created twice through the
/// executor; every `WEBMETAL_CORPUS_STRIDE`-th of those that compile (default 16) is also run, once through each
/// pipeline, with equal inputs. Progress goes to standard error every 100 kernels.
#[test]
fn generated_corpus_kernels_create_as_the_compiler_decides_and_cached_runs_equal_fresh_ones() {
    use airwgsl::air::{Role, Slot};
    let Some(root) = std::env::var_os("WEBMETAL_CORPUS").map(std::path::PathBuf::from) else {
        eprintln!("generated corpus not run: WEBMETAL_CORPUS is unset");
        return;
    };
    let version = std::env::var("WEBMETAL_CORPUS_VERSION").unwrap_or_else(|_| "3.1".into());
    let stride: usize = std::env::var("WEBMETAL_CORPUS_STRIDE").ok().and_then(|s| s.parse().ok()).unwrap_or(16).max(1);
    let sample: usize = std::env::var("WEBMETAL_CORPUS_SAMPLE").ok().and_then(|s| s.parse().ok()).unwrap_or(1).max(1);
    let start = std::time::Instant::now();
    let (mut kernels, mut sampled_out) = (0usize, 0usize);
    let tsv = std::fs::read_to_string(root.join("modules.tsv")).expect("modules.tsv");
    let mut modules: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for line in tsv.lines().filter(|l| !l.starts_with('#')) {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() >= 3 && c[1] == version {
            modules.entry(c[0].to_string()).or_default().push(c[2].rsplit('/').next().unwrap().to_string());
        }
    }
    let h = Harness::new();
    let (mut created, mut refused, mut ran, mut compiled) = (0usize, 0usize, 0usize, 0usize);
    for (path, functions) in &modules {
        let bitcode = std::fs::read(root.join(path)).expect("a corpus module");
        let Ok(decoded) = airwgsl::air::AirModule::parse(&bitcode) else { continue };
        for name in functions {
            let Some(entry) = decoded.entry_point(name) else { continue };
            if entry.stage != airwgsl::air::Stage::Kernel {
                continue;
            }
            kernels += 1;
            if !(kernels - 1).is_multiple_of(sample) {
                sampled_out += 1;
                continue;
            }
            if (kernels - 1).is_multiple_of(100 * sample) {
                eprintln!("{:?}: {path} {name}: {created} created, {refused} refused", start.elapsed());
            }
            let lf = LibraryFunction { bitcode: bitcode.clone(), function_type: 3 };
            let expect = predicted(&h, &lf, name);
            let func = h.function(&bitcode, name, 3);
            let (p1, p2) = (h.handle(), h.handle());
            let s1 = reply_failure(&h.compute_pipeline(p1, compute_descriptor(func), 0));
            let before = h.shared.compiler.stats();
            let s2 = reply_failure(&h.compute_pipeline(p2, compute_descriptor(func), 0));
            assert_eq!((s1.status, &s1.message), (s2.status, &s2.message), "{path} {name}: a cached creation differs");
            match &expect {
                Err((s, _)) => {
                    assert_eq!(s1.status, *s, "{path} {name}: {s1:?} instead of {expect:?}");
                    refused += 1;
                    continue;
                }
                Ok(()) => assert_eq!(s1.status, status::OK, "{path} {name}: {s1:?}"),
            }
            created += 1;
            assert_eq!(h.shared.compiler.stats().hits, before.hits + 1, "{path} {name}: no compiler cache hit");
            // The cached compilation equals a fresh one, WGSL byte for byte.
            let air = std::rc::Rc::new(webmetal_host::pipeline::registry::Air {
                hash: webmetal_host::pipeline::sha256::sha256(&bitcode),
                bitcode: std::sync::Arc::from(&bitcode[..]),
            });
            let function = Function { air, name: name.clone(), function_type: 3, constants: Vec::new() };
            let settings = compute::check_descriptor("test", &compute_descriptor(func)).unwrap();
            let info = compute::kernel_info(&function).unwrap();
            let desc = compute::creation_desc(&settings, &info);
            let cached = h.shared.compiler.compile(&[function.stage_module()], &desc);
            let fresh = h.shared.compiler.compile_fresh(&[function.stage_module()], &desc);
            assert_eq!(&*cached, &fresh, "{path} {name}: the cached compilation differs from a fresh one");
            compiled += 1;
            if !(compiled - 1).is_multiple_of(stride) || info.stage_in {
                continue;
            }
            // One thread, every buffer argument a buffer of generated bytes, every threadgroup argument its length.
            let mut setup = Vec::new();
            for a in &entry.arguments {
                let Role::Buffer(b) = &a.role else { continue };
                let Slot::Value(index) = b.location.index else { continue };
                if b.address_space == Some(3) {
                    let len = u64::from(a.type_size.unwrap_or(0)).max(1).div_ceil(16) * 16;
                    setup.push((u64::from(index), None, len));
                } else if !a.unused {
                    setup.push((u64::from(index), Some(mem_common_pattern(4096, index * 31 + a.position)), 0));
                }
            }
            let mut outputs = Vec::new();
            for p in [p1, p2] {
                let mut body = vec![set_pipeline(p)];
                let mut addrs = Vec::new();
                for (index, bytes, len) in &setup {
                    match bytes {
                        Some(bytes) => {
                            let (b, addr) = h.buffer_with(bytes);
                            body.push(set_buffer(b, *index));
                            addrs.push(addr);
                        }
                        None => {
                            body.push(Command::ComputeSetThreadgroupMemoryLength(ComputeSetThreadgroupMemoryLength {
                                length: *len,
                                index: *index,
                            }))
                        }
                    }
                }
                body.push(dispatch([1, 1, 1], [1, 1, 1]));
                let (f, at) = h.compute(body);
                outputs.push((f.status, addrs.iter().map(|&a| h.read(a, 4096)).collect::<Vec<_>>()));
                if f.status != status::OK {
                    // A dispatch the host refuses is refused the same way through both pipelines.
                    eprintln!("{path} {name}: dispatch refused at {at}: {f:?}");
                }
            }
            assert_eq!(outputs[0], outputs[1], "{path} {name}: the cached pipeline ran differently");
            ran += 1;
        }
    }
    h.assert_no_errors();
    eprintln!(
        "{}: corpus {version}: {created} kernels created and recreated from the caches with equal WGSL, {refused} \
         refused as the compiler refuses them; {ran} run through both pipelines with equal results (stride {stride}); \
         {sampled_out} of {kernels} kernels not sampled (every {sample}th created); {:?}",
        h.name(),
        start.elapsed()
    );
}

/// `WMPattern` of the reference harness (`tests/reference/FORMAT.md`, "Generated inputs").
fn mem_common_pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut s = if seed == 0 { 0x9e37_79b9u32 } else { seed };
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as u8
        })
        .collect()
}

/// `texture.fragment-access` with the compiler's pipelines: fragment functions write an R32Uint texture at levels
/// computed at run time, directly and through a level-range view, write an RGBA8Unorm texture of two levels with float
/// values, and read through a swizzled view, each access one fragment of a draw of its own. The render encoder draws
/// with each pipeline compiled for the textures the draw binds (their storage formats, level counts and swizzles), and
/// every `exact` record equals Metal's.
#[test]
fn fragment_functions_access_textures_as_metal_does() {
    use pipeline_common::render_descriptor;
    const RGBA8_UINT: u64 = 73;
    const RGBA32_UINT: u64 = 123;
    let h = Harness::new();
    let lib = reference_library();
    let g = Golden::load("texture.fragment-access");
    let quads = |name: &str| -> Vec<[u32; 8]> {
        let b = g.bytes(name).unwrap_or_else(|| panic!("{name} is recorded"));
        b.chunks(32)
            .map(|c| std::array::from_fn(|i| u32::from_le_bytes(c[4 * i..4 * i + 4].try_into().unwrap())))
            .collect()
    };
    let bytes_of = |q: &[u32]| -> Vec<u8> { q.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let fun = |name: &str| h.function(&lib[name].bitcode, name, lib[name].function_type);
    let vertex = fun("tex_cover_vertex");
    let pipeline = |fragment: &str, format: u64| -> Handle {
        let p = h.handle();
        let f = reply_failure(&h.render_pipeline(p, render_descriptor(Some(vertex), Some(fun(fragment)), format), 0));
        assert_eq!(f.status, status::OK, "{}: {fragment}: {f:?}", h.name());
        p
    };
    let texture = |d: TextureDescriptor| -> Handle {
        let t = h.handle();
        h.ok(Request::CreateTexture(CreateTexture { texture: t, descriptor: d }));
        t
    };
    // MTLStorageModePrivate 2; MTLTextureUsageShaderRead 1, ShaderWrite 2, RenderTarget 4, PixelFormatView 0x10.
    let target = |format: u64| texture(render_common::texture_descriptor(format, 1, 1, 2, 4));
    let run = |records: Vec<Command>| {
        let (f, at) = h.run(&records);
        assert_eq!(f.status, status::OK, "{}: offset {at}: {f:?}", h.name());
    };
    let draw = |p: Handle, target: Handle, t: Handle, q: &[u32; 8], value: Option<Vec<u8>>| {
        let (qb, _) = h.buffer_with(&bytes_of(q));
        let mut body = vec![
            Command::RenderSetRenderPipelineState(RenderSetRenderPipelineState { pipeline_state: Some(p) }),
            Command::RenderSetFragmentTexture(RenderSetFragmentTexture { texture: Some(t), index: 0 }),
            Command::RenderSetFragmentBuffer(RenderSetFragmentBuffer { buffer: Some(qb), offset: 0, index: 0 }),
        ];
        if let Some(v) = value {
            let (vb, _) = h.buffer_with(&v);
            body.push(Command::RenderSetFragmentBuffer(RenderSetFragmentBuffer {
                buffer: Some(vb),
                offset: 0,
                index: 1,
            }));
        }
        // MTLPrimitiveTypeTriangle 3.
        body.push(Command::RenderDrawPrimitives(RenderDrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
        }));
        run(render_common::render(
            h.handle(),
            render_common::pass_color(target, render_common::clear(0.0, 0.0, 0.0, 0.0)),
            body,
        ));
    };
    let upload = |t: Handle, (w, hh): (u64, u64), level: u64, bytes: &[u8]| {
        let (src, _) = h.buffer_with(bytes);
        run(vec![
            Command::BeginBlit(BeginBlit { encoder: h.handle() }),
            Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
                source_buffer: Some(src),
                source_offset: 0,
                source_bytes_per_row: bytes.len() as u64 / hh,
                source_bytes_per_image: 0,
                source_size: Size { width: w, height: hh, depth: 1 },
                destination_texture: Some(t),
                destination_slice: 0,
                destination_level: level,
                destination_origin: Origin { x: 0, y: 0, z: 0 },
            }),
            Command::EncoderEndEncoding(EncoderEndEncoding {}),
        ]);
    };
    let read = |t: Handle, (w, hh): (u64, u64), level: u64, texel: u64| -> Vec<u8> {
        let len = (w * hh * texel) as usize;
        let (out, addr) = h.buffer_with(&vec![0u8; len]);
        run(vec![
            Command::BeginBlit(BeginBlit { encoder: h.handle() }),
            Command::BlitCopyTextureToBuffer(BlitCopyTextureToBuffer {
                source_texture: Some(t),
                source_slice: 0,
                source_level: level,
                source_origin: Origin { x: 0, y: 0, z: 0 },
                source_size: Size { width: w, height: hh, depth: 1 },
                destination_buffer: Some(out),
                destination_offset: 0,
                destination_bytes_per_row: w * texel,
                destination_bytes_per_image: w * hh * texel,
            }),
            Command::EncoderEndEncoding(EncoderEndEncoding {}),
        ]);
        h.read(addr, len)
    };
    let mip = |size: u64, level: u64| (size >> level).max(1);
    let mut exact = 0;
    let mut compare = |name: &str, got: &[u8]| {
        assert_eq!(g.rule(name), Some("exact"), "{name}");
        assert_eq!(Some(got.to_vec()), g.bytes(name), "{}: {name}", h.name());
        exact += 1;
    };

    // R32Uint, 4 levels, written at levels computed at run time: the texture, then a view of its levels 1 and 2.
    let uint_target = target(R32_UINT);
    let write_uint = pipeline("tex_fragment_write_2d", R32_UINT);
    for (i, name) in ["levels", "level-view"].into_iter().enumerate() {
        let mut d = render_common::texture_descriptor(R32_UINT, 8, 8, 2, 0x13);
        d.mipmap_level_count = 4;
        let root = texture(d);
        for level in 0..4 {
            let w = mip(8, level);
            let texels: Vec<u32> =
                (0..w).flat_map(|y| (0..w).map(move |x| 0xa000_0000 | (level << 24 | y << 8 | x) as u32)).collect();
            upload(root, (w, w), level, &bytes_of(&texels));
        }
        let t = if name == "level-view" {
            let v = h.handle();
            h.ok(Request::CreateTextureViewLevelsSlices(CreateTextureViewLevelsSlices {
                texture: root,
                view: v,
                pixel_format: R32_UINT,
                texture_type: 2,
                level_range: Range { location: 1, length: 2 },
                slice_range: Range { location: 0, length: 1 },
            }));
            v
        } else {
            root
        };
        let queries = quads(&format!("{name}.writes"));
        for (j, q) in queries.iter().enumerate() {
            let value = [0xf7a0_0000 | (i as u32) << 8 | j as u32, 0, 0, 0];
            draw(write_uint, uint_target, t, q, Some(bytes_of(&value)));
        }
        let contents: Vec<u8> = (0..4).flat_map(|level| read(root, (mip(8, level), mip(8, level)), level, 4)).collect();
        compare(&format!("{name}.contents"), &contents);
    }

    // RGBA8Unorm, 2 levels, written with float values.
    let mut ud = render_common::texture_descriptor(RGBA8_UNORM, 4, 4, 2, 0x2);
    ud.mipmap_level_count = 2;
    let unorm = texture(ud);
    let initial = mem_common_pattern(64 + 16, 73);
    upload(unorm, (4, 4), 0, &initial[..64]);
    upload(unorm, (2, 2), 1, &initial[64..]);
    let write_float = pipeline("tex_fragment_write_2d_float", R32_UINT);
    let values = g.bytes("unorm.values").expect("unorm.values is recorded");
    for (j, q) in quads("unorm.writes").iter().enumerate() {
        draw(write_float, uint_target, unorm, q, Some(values[16 * j..16 * j + 16].to_vec()));
    }
    let contents: Vec<u8> = [read(unorm, (4, 4), 0, 4), read(unorm, (2, 2), 1, 4)].concat();
    compare("unorm.contents", &contents);

    // An RGBA8Uint view swizzled (blue, zero, one, red), read texel by texel.
    let mut pattern = mem_common_pattern(64, 71);
    pattern[..6].copy_from_slice(&[0, 255, 0, 255, 255, 0]);
    let base = texture(render_common::texture_descriptor(RGBA8_UINT, 4, 4, 2, 0x11));
    upload(base, (4, 4), 0, &pattern);
    let view = h.handle();
    // MTLTextureSwizzle: zero 0, one 1, red 2, green 3, blue 4, alpha 5.
    h.ok(Request::CreateTextureViewLevelsSlicesSwizzle(CreateTextureViewLevelsSlicesSwizzle {
        texture: base,
        view,
        pixel_format: RGBA8_UINT,
        texture_type: 2,
        level_range: Range { location: 0, length: 1 },
        slice_range: Range { location: 0, length: 1 },
        swizzle: TextureSwizzleChannels { red: 4, green: 0, blue: 1, alpha: 2 },
    }));
    let read_target = target(RGBA32_UINT);
    let read_uint = pipeline("tex_fragment_read_2d", RGBA32_UINT);
    let mut reads = Vec::new();
    for q in quads("swizzled-view.reads.queries") {
        draw(read_uint, read_target, view, &q, None);
        reads.extend(read(read_target, (1, 1), 0, 16));
    }
    compare("swizzled-view.reads", &reads);
    eprintln!("{}: {}: {exact} exact records equal", h.name(), g.case);
    h.assert_no_errors();
}
