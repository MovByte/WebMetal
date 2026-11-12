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
        Ok(_) => Ok(()),
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
    h.ok(Request::CreateBinaryArchive(CreateBinaryArchive {
        archive,
        descriptor: BinaryArchiveDescriptor { options: 0, url: None, writing_options: 0 },
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

    // A fragment output whose type does not match the attachment's format: undefined in Metal, rejected by WebGPU.
    let f = render_status(&h, render_descriptor(Some(vertex), Some(fragment), RGBA32_UINT));
    if render_predicted(&h, &lib, "pip_simple_vertex", Some("pip_simple_fragment"), RGBA32_UINT) == status::OK {
        assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
        assert!(f.message.contains("sample type"), "{f:?}");
    }

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
