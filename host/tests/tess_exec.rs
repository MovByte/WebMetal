//! Patch draws through the executor on a native WebGPU implementation (`WEBGPU_NATIVE`, see `host/build.rs`): the
//! pipeline service compiling the reference library's post-tessellation vertex functions, the render path
//! tessellating each draw before its pass (`webmetal_host::tess`), and the targets compared with what Metal drew
//! (`tests/reference/goldens`, `tessellation.render`). Patch draws Metal leaves undefined, and the triangle order the
//! executor does not reproduce, fail their command buffers.

#![cfg(feature = "native")]

mod mem_common;
mod pipeline_common;
mod render_common;

use std::collections::HashMap;

use mem_common::Golden;
use pipeline_common::{Harness, LibraryFunction, reference_library, render_descriptor};
use webmetal_wire::*;

/// `MTLPixelFormatR32Uint`.
const R32_UINT: u64 = 53;
/// The side of the targets of `tessellation.render`.
const SIZE: u64 = 32;
/// Control points and per-patch attributes of `tessellation.render`.
const POINTS: usize = 64;
const TAGS: u32 = 16;
const SLOTS: usize = 24;

fn reply_failure(r: &Reply) -> Failure {
    webmetal_host::exec::schema::reply_failure(r).cloned().unwrap_or_else(webmetal_host::exec::failure::ok)
}

/// `WMHalfFromFloat`: binary16, rounded to nearest even.
fn half(v: f32) -> u16 {
    airwgsl::tess::factors::f32_to_half(v)
}

/// One draw of `tessellation.render` (`TRender` of `Tessellation.m`).
#[derive(Clone, Copy)]
struct Draw {
    name: &'static str,
    quad: bool,
    /// `MTLTessellationPartitionMode`.
    mode: u64,
    /// `MTLWinding`.
    winding: u64,
    /// `MTLTessellationFactorStepFunction`.
    step: u64,
    /// `MTLTessellationControlPointIndexType`.
    index_type: u64,
    /// 0: scaling disabled.
    scale: f32,
    offset: u64,
    stride: u64,
    start: u64,
    count: u64,
    instances: u64,
    base: u64,
    patch_indices: bool,
    index_offset: u64,
    /// The fragment function reads the swizzled texture at index 0 (`tess_draw_fragment_texture`).
    texture: bool,
}

const POW2: u64 = 0;
const INTEGER: u64 = 1;
const ODD: u64 = 2;
const EVEN: u64 = 3;
const CW: u64 = 0;
const CCW: u64 = 1;
const CONSTANT: u64 = 0;
const PER_PATCH: u64 = 1;
const PER_INSTANCE: u64 = 2;
const PER_PATCH_AND_INSTANCE: u64 = 3;

#[allow(clippy::too_many_arguments)]
const fn draw(
    name: &'static str,
    quad: bool,
    mode: u64,
    winding: u64,
    step: u64,
    index_type: u64,
    scale: f32,
    [offset, stride, start, count, instances, base]: [u64; 6],
    patch_indices: bool,
    index_offset: u64,
) -> Draw {
    Draw {
        name,
        quad,
        mode,
        winding,
        step,
        index_type,
        scale,
        offset,
        stride,
        start,
        count,
        instances,
        base,
        patch_indices,
        index_offset,
        texture: false,
    }
}

const fn with_texture(d: Draw) -> Draw {
    Draw { texture: true, ..d }
}

/// The draws of `tessellation.render`, in its order.
const DRAWS: [Draw; 14] = [
    draw("quad-integer", true, INTEGER, CW, PER_PATCH, 0, 0.0, [0, 0, 0, 3, 1, 0], false, 0),
    draw("quad-fractional-odd-instances", true, ODD, CW, PER_PATCH, 0, 0.0, [48, 0, 1, 2, 2, 1], false, 0),
    draw("quad-culled-patch", true, INTEGER, CW, PER_PATCH, 0, 0.0, [0, 0, 9, 3, 1, 0], false, 0),
    draw("triangle-fractional-even", false, EVEN, CW, PER_PATCH, 0, 0.0, [0, 0, 0, 3, 1, 0], false, 0),
    draw("triangle-pow2-counter-clockwise", false, POW2, CCW, PER_PATCH, 0, 0.0, [0, 0, 1, 2, 1, 0], false, 0),
    draw("quad-constant", true, EVEN, CW, CONSTANT, 0, 0.0, [24, 0, 0, 3, 2, 3], false, 0),
    draw("quad-per-instance", true, INTEGER, CW, PER_INSTANCE, 0, 0.0, [0, 24, 0, 2, 3, 2], false, 0),
    draw(
        "triangle-per-patch-and-instance",
        false,
        ODD,
        CW,
        PER_PATCH_AND_INSTANCE,
        0,
        0.0,
        [8, 40, 1, 2, 2, 1],
        false,
        0,
    ),
    draw("quad-patch-indices", true, INTEGER, CW, PER_PATCH, 0, 0.0, [0, 0, 1, 3, 1, 0], true, 0),
    draw("quad-indexed-uint16", true, EVEN, CW, PER_PATCH, 1, 0.0, [0, 0, 1, 2, 2, 0], false, 6),
    draw("triangle-indexed-uint32-patch-indices", false, INTEGER, CW, PER_PATCH, 2, 0.0, [0, 0, 0, 4, 1, 0], true, 8),
    draw("quad-scale", true, ODD, CW, PER_PATCH, 0, 1.5, [0, 0, 0, 2, 1, 0], false, 0),
    draw("quad-tiled-inside", true, INTEGER, CW, PER_PATCH, 0, 0.0, [276, 0, 0, 1, 1, 0], false, 0),
    with_texture(draw("quad-fragment-texture", true, INTEGER, CW, PER_PATCH, 0, 0.0, [0, 0, 0, 2, 1, 0], false, 0)),
];

fn le32(words: impl IntoIterator<Item = u32>) -> Vec<u8> {
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

/// The buffers of `tessellation.render`, created through the executor.
struct Buffers {
    points: Handle,
    tags: Handle,
    factors: Handle,
    patch_indices: Handle,
    indices16: Handle,
    indices32: Handle,
    /// RGBA8Uint, 4 x 4, with the descriptor swizzle (alpha, red, green, blue).
    texture: Handle,
}

impl Buffers {
    fn new(h: &Harness) -> Buffers {
        let mut points = Vec::new();
        for i in 0..POINTS {
            points.extend([(i % 7) as f32 - 3.0, (i % 5) as f32 - 2.0].map(|v| v / 64.0));
            points.extend([0.0, (i + 1) as f32 * 1.25]);
        }
        let mut factors: Vec<u16> = (0..SLOTS * 6).map(|i| half(1.0 + ((i * 7) % 11) as f32 * 0.5)).collect();
        factors[10 * 6 + 1] = half(0.0);
        for (k, v) in [4.0, 4.0, 4.0, 4.0, 16.0, 16.0].into_iter().enumerate() {
            factors[23 * 6 + k] = half(v);
        }
        let indices16: Vec<u8> = (0..64u16).flat_map(|i| ((i * 7 + 3) % POINTS as u16).to_le_bytes()).collect();
        Buffers {
            points: h.buffer_with(&le32(points.into_iter().map(f32::to_bits))).0,
            tags: h.buffer_with(&le32((0..TAGS).map(|j| (j + 1).wrapping_mul(0x9e37_79b9)))).0,
            factors: h.buffer_with(&factors.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>()).0,
            patch_indices: h.buffer_with(&le32([5, 2, 9, 0])).0,
            indices16: h.buffer_with(&indices16).0,
            indices32: h.buffer_with(&le32((0..64u32).map(|i| (i * 11 + 5) % POINTS as u32))).0,
            texture: swizzled_texture(h),
        }
    }
}

/// The texture of `quad-fragment-texture`: bytes `i * 37 + 11`, copied in by a blit.
fn swizzled_texture(h: &Harness) -> Handle {
    // MTLPixelFormatRGBA8Uint 73; MTLStorageModePrivate 2; MTLTextureUsageShaderRead 1; MTLTextureSwizzleRed 2, Green
    // 3, Blue 4, Alpha 5.
    let mut d = render_common::texture_descriptor(73, 4, 4, 2, 1);
    d.swizzle = TextureSwizzleChannels { red: 5, green: 2, blue: 3, alpha: 4 };
    let t = h.handle();
    h.ok(Request::CreateTexture(CreateTexture { texture: t, descriptor: d }));
    let bytes: Vec<u8> = (0..64u32).map(|i| (i * 37 + 11) as u8).collect();
    let (src, _) = h.buffer_with(&bytes);
    let (f, at) = h.run(&[
        Command::BeginBlit(BeginBlit { encoder: h.handle() }),
        Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
            source_buffer: Some(src),
            source_offset: 0,
            source_bytes_per_row: 16,
            source_bytes_per_image: 64,
            source_size: Size { width: 4, height: 4, depth: 1 },
            destination_texture: Some(t),
            destination_slice: 0,
            destination_level: 0,
            destination_origin: Origin { x: 0, y: 0, z: 0 },
        }),
        Command::EncoderEndEncoding(EncoderEndEncoding {}),
    ]);
    assert_eq!(f.status, status::OK, "{}: offset {at}: {f:?}", h.name());
    t
}

/// Attribute 0 a float4 of buffer 1 per control point, attribute 1 a uint of buffer 2 per patch.
fn vertex_descriptor() -> VertexDescriptor {
    // MTLVertexFormatFloat4 31, MTLVertexFormatUInt 36; MTLVertexStepFunctionPerPatch 3, PerPatchControlPoint 4, and
    // the new descriptor's PerVertex 1.
    VertexDescriptor {
        attributes: (0..31)
            .map(|i| match i {
                0 => VertexAttribute { buffer_index: 1, format: 31, offset: 0 },
                1 => VertexAttribute { buffer_index: 2, format: 36, offset: 0 },
                _ => VertexAttribute { buffer_index: 0, format: 0, offset: 0 },
            })
            .collect(),
        layouts: (0..31)
            .map(|i| match i {
                1 => VertexBufferLayout { step_function: 4, step_rate: 1, stride: 16 },
                2 => VertexBufferLayout { step_function: 3, step_rate: 1, stride: 4 },
                _ => VertexBufferLayout { step_function: 1, step_rate: 1, stride: 0 },
            })
            .collect(),
    }
}

fn descriptor(h: &Harness, lib: &HashMap<String, LibraryFunction>, d: &Draw) -> RenderPipelineDescriptor {
    let fun = |name: &str| h.function(&lib[name].bitcode, name, lib[name].function_type);
    let vertex = fun(if d.quad { "tess_draw_quad" } else { "tess_draw_triangle" });
    let fragment = fun(if d.texture { "tess_draw_fragment_texture" } else { "tess_draw_fragment" });
    let mut desc = render_descriptor(Some(vertex), Some(fragment), R32_UINT);
    desc.tessellation_partition_mode = d.mode;
    desc.max_tessellation_factor = 16;
    desc.tessellation_factor_scale_enabled = d.scale != 0.0;
    desc.tessellation_factor_step_function = d.step;
    desc.tessellation_output_winding_order = d.winding;
    desc.tessellation_control_point_index_type = d.index_type;
    desc.vertex_descriptor = Some(vertex_descriptor());
    desc
}

fn pipeline(h: &Harness, desc: RenderPipelineDescriptor) -> Handle {
    let p = h.handle();
    let f = reply_failure(&h.render_pipeline(p, desc, 0));
    assert_eq!(f.status, status::OK, "{}: {f:?}", h.name());
    p
}

fn target(h: &Harness) -> Handle {
    // MTLStorageModePrivate 2, MTLTextureUsageRenderTarget 4.
    let t = h.handle();
    h.ok(Request::CreateTexture(CreateTexture {
        texture: t,
        descriptor: render_common::texture_descriptor(R32_UINT, SIZE, SIZE, 2, 4),
    }));
    t
}

/// The draw's patch draw record.
fn patch_draw(d: &Draw, b: &Buffers) -> Command {
    let patches = d.patch_indices.then_some(b.patch_indices);
    let control_points = if d.quad { 4 } else { 3 };
    match d.index_type {
        0 => Command::RenderDrawPatches(RenderDrawPatches {
            number_of_patch_control_points: control_points,
            patch_start: d.start,
            patch_count: d.count,
            patch_index_buffer: patches,
            patch_index_buffer_offset: 0,
            instance_count: d.instances,
            base_instance: d.base,
        }),
        t => Command::RenderDrawIndexedPatches(RenderDrawIndexedPatches {
            number_of_patch_control_points: control_points,
            patch_start: d.start,
            patch_count: d.count,
            patch_index_buffer: patches,
            patch_index_buffer_offset: 0,
            control_point_index_buffer: Some(if t == 1 { b.indices16 } else { b.indices32 }),
            control_point_index_buffer_offset: d.index_offset,
            instance_count: d.instances,
            base_instance: d.base,
        }),
    }
}

/// One render pass into a cleared target with `body` after the pipeline, buffers and factor buffer of `d`, then a
/// copy of the target into a shared buffer: the command buffer's completion and the copied target.
fn render(h: &Harness, p: Handle, d: &Draw, b: &Buffers, body: Vec<Command>) -> ((Failure, u64), Vec<u8>) {
    let t = target(h);
    let mut commands = vec![
        Command::RenderSetRenderPipelineState(RenderSetRenderPipelineState { pipeline_state: Some(p) }),
        Command::RenderSetVertexBuffer(RenderSetVertexBuffer { buffer: Some(b.points), offset: 0, index: 1 }),
        Command::RenderSetVertexBuffer(RenderSetVertexBuffer { buffer: Some(b.tags), offset: 0, index: 2 }),
        Command::RenderSetTessellationFactorBuffer(RenderSetTessellationFactorBuffer {
            buffer: Some(b.factors),
            offset: d.offset,
            instance_stride: d.stride,
        }),
    ];
    if d.texture {
        commands
            .push(Command::RenderSetFragmentTexture(RenderSetFragmentTexture { texture: Some(b.texture), index: 0 }));
    }
    if d.scale != 0.0 {
        commands.push(Command::RenderSetTessellationFactorScale(RenderSetTessellationFactorScale { scale: d.scale }));
    }
    commands.extend(body);
    let mut records = render_common::render(
        h.handle(),
        render_common::pass_color(t, render_common::clear(0.0, 0.0, 0.0, 0.0)),
        commands,
    );
    let len = SIZE * SIZE * 4;
    let (out, addr) = h.buffer_with(&vec![0u8; len as usize]);
    records.push(Command::BeginBlit(BeginBlit { encoder: h.handle() }));
    records.push(Command::BlitCopyTextureToBuffer(BlitCopyTextureToBuffer {
        source_texture: Some(t),
        source_slice: 0,
        source_level: 0,
        source_origin: Origin { x: 0, y: 0, z: 0 },
        source_size: Size { width: SIZE, height: SIZE, depth: 1 },
        destination_buffer: Some(out),
        destination_offset: 0,
        destination_bytes_per_row: SIZE * 4,
        destination_bytes_per_image: len,
    }));
    records.push(Command::EncoderEndEncoding(EncoderEndEncoding {}));
    let done = h.run(&records);
    (done, h.read(addr, len as usize))
}

/// `tessellation.render`: every draw's target equals Metal's, except the draw whose inside the M4 emits in tiles,
/// whose command buffer fails naming the order it does not reproduce. The draw whose fragment function reads a swizzled
/// texture is drawn with the pipeline compiled for the texture it binds, as an ordinary draw is.
#[test]
fn rendered_patch_draws_equal_metal_s_targets() {
    let h = Harness::new();
    let lib = reference_library();
    let golden = Golden::load("tessellation.render");
    assert_eq!(golden.status(), "ok");
    let b = Buffers::new(&h);
    let mut equal = 0;
    for d in &DRAWS {
        let p = pipeline(&h, descriptor(&h, &lib, d));
        let ((f, at), got) = render(&h, p, d, &b, vec![patch_draw(d, &b)]);
        if d.name == "quad-tiled-inside" {
            assert_eq!(f.status, status::UNIMPLEMENTED, "{}: {}: {f:?}", h.name(), d.name);
            assert!(f.message.contains("blade wider than 36 grid points"), "{}: {f:?}", d.name);
            continue;
        }
        assert_eq!(f.status, status::OK, "{}: {}: offset {at}: {f:?}", h.name(), d.name);
        let want = golden.bytes(d.name).expect("the draw's target");
        if got != want {
            let differ = got.chunks(4).zip(want.chunks(4)).filter(|(a, b)| a != b).count();
            panic!("{}: {}: {differ} of {} pixels differ from Metal's", h.name(), d.name, SIZE * SIZE);
        }
        equal += 1;
    }
    println!("{}: {equal} patch draws equal Metal's targets on every pixel", h.name());
    h.assert_no_errors();
}

/// Patch draws the validation layer rejects (measured on Apple M4, macOS 27.0.1) and the ones the A10X cannot make
/// fail their command buffers naming the call and the rule.
#[test]
fn patch_draws_metal_leaves_undefined_fail_loudly() {
    let h = Harness::new();
    let lib = reference_library();
    let b = Buffers::new(&h);
    let quad = DRAWS[0];
    let tessellated = pipeline(&h, descriptor(&h, &lib, &quad));
    let fails = |p: Handle, d: &Draw, body: Vec<Command>, needle: &str| {
        let ((f, _), _) = render(&h, p, d, &b, body);
        assert_eq!(f.status, status::UNIMPLEMENTED, "{}: {needle}: {f:?}", h.name());
        assert!(f.message.contains(needle), "{}: {:?} does not mention {needle:?}", h.name(), f.message);
    };

    // A patch draw needs a post-tessellation vertex function, and such a pipeline draws patches only.
    let plain = {
        let fun = |name: &str| h.function(&lib[name].bitcode, name, lib[name].function_type);
        pipeline(&h, render_descriptor(Some(fun("rnd_vertex_triangles")), Some(fun("rnd_fragment_discard")), R32_UINT))
    };
    fails(plain, &quad, vec![patch_draw(&quad, &b)], "not a post-tessellation function");
    let primitives =
        Command::RenderDrawPrimitives(RenderDrawPrimitives { primitive_type: 3, vertex_start: 0, vertex_count: 3 });
    fails(tessellated, &quad, vec![primitives], "a primitive draw with a pipeline whose vertex function is");

    // drawIndexedPatches needs a control-point index type, drawPatches none.
    let mut indexed = quad;
    indexed.index_type = 1;
    let indexed_pipeline = pipeline(&h, descriptor(&h, &lib, &indexed));
    fails(indexed_pipeline, &quad, vec![patch_draw(&quad, &b)], "tessellationControlPointIndexType 1");
    fails(tessellated, &quad, vec![patch_draw(&indexed, &b)], "tessellationControlPointIndexType 0");

    // controlPointIndexBufferOffset(1) must be a multiple of 2 bytes; offset + (patchStart + patchCount) *
    // controlPoints * 2 must be within the buffer.
    let mut odd = indexed;
    odd.index_offset = 1;
    fails(indexed_pipeline, &odd, vec![patch_draw(&odd, &b)], "control-point indices at offset 1");
    let mut past = indexed;
    past.index_offset = 112;
    fails(indexed_pipeline, &past, vec![patch_draw(&past, &b)], "up to byte 136 of a 128-byte buffer");

    // patchIndexBufferOffset must be a multiple of 4 bytes, and offset + (patchStart + patchCount) * 4 within the
    // buffer.
    let mut patches = quad;
    patches.patch_indices = true;
    patches.start = 2;
    fails(tessellated, &patches, vec![patch_draw(&patches, &b)], "patch indices 2..5 at offset 0 of a 16-byte");

    // The factors the draw reads must lie in the factor buffer.
    let mut factors = quad;
    factors.offset = 276;
    fails(tessellated, &factors, vec![patch_draw(&factors, &b)], "up to byte 312 of a 288-byte factor buffer");

    // No factor buffer set ("missing tessellationFactorBuffer binding").
    let records = render_common::render(
        h.handle(),
        render_common::pass_color(target(&h), render_common::clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            Command::RenderSetRenderPipelineState(RenderSetRenderPipelineState { pipeline_state: Some(tessellated) }),
            patch_draw(&quad, &b),
        ],
    );
    let (f, _) = h.run(&records);
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("no tessellation factor buffer is set"), "{f:?}");

    // The A10X has no indirect patch draws (supportsIndirectTessellation is NO in its device profile).
    let (args, _) = h.buffer_with(&le32([1, 1, 0, 0]));
    let indirect = Command::RenderDrawPatchesIndirect(RenderDrawPatchesIndirect {
        number_of_patch_control_points: 4,
        patch_index_buffer: None,
        patch_index_buffer_offset: 0,
        indirect_buffer: Some(args),
        indirect_buffer_offset: 0,
    });
    fails(tessellated, &quad, vec![indirect], "supportsIndirectTessellation = false");

    h.assert_no_errors();
}

/// Scaling enabled on an encoder that never sets a scale culls every patch (tessellation.factors scale-unset): the
/// draw succeeds and draws nothing; the same draw after setTessellationFactorScale: draws.
#[test]
fn an_unset_factor_scale_culls_every_patch() {
    let h = Harness::new();
    let lib = reference_library();
    let golden = Golden::load("tessellation.factors");
    assert_eq!(golden.bytes("scale-unset.points").expect("scale-unset.points"), Vec::<u8>::new());
    let b = Buffers::new(&h);
    let mut scaled = DRAWS[0];
    scaled.scale = 1.5;
    let p = pipeline(&h, descriptor(&h, &lib, &scaled));
    let mut unset = scaled;
    unset.scale = 0.0;
    let ((f, at), got) = render(&h, p, &unset, &b, vec![patch_draw(&unset, &b)]);
    assert_eq!(f.status, status::OK, "{}: offset {at}: {f:?}", h.name());
    assert!(got.iter().all(|&x| x == 0), "{}: an unset scale drew", h.name());
    let ((f, at), got) = render(&h, p, &scaled, &b, vec![patch_draw(&scaled, &b)]);
    assert_eq!(f.status, status::OK, "{}: offset {at}: {f:?}", h.name());
    assert!(got.iter().any(|&x| x != 0), "{}: a scale of 1.5 drew nothing", h.name());
    h.assert_no_errors();
}
