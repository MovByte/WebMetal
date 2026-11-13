//! The render rows of the Metal reference harness (`tests/reference/src/cases`: Raster.m, Draw.m, DepthStencil.m,
//! Blend.m, RenderPass.m, RenderFragment.m, Formats.m), replayed through the executor on a native WebGPU
//! implementation and compared with the outputs real Metal recorded (`tests/reference/goldens`, release variant) under
//! the rules of `tests/reference/FORMAT.md`.
//!
//! Each test is one reference case: the same resources, pipelines, passes and draws, sent as the guest sends them
//! (requests and command-buffer streams), read back as the case reads them, and compared record by record. Cases that
//! need what the render path refuses assert the refusal instead (see the test's comment), so a refusal that goes
//! away is noticed.

#![cfg(feature = "native")]

mod mem_common;
mod render_common;

use render_common::golden::{Candidate, Golden, layout};
use render_common::*;
use webmetal_wire::*;

/// Compare a replay with the reference and print the counts.
fn check(case: &str, c: &Candidate) {
    let g = Golden::load(case);
    assert_eq!(g.status(), "ok", "{case}: the reference case did not complete");
    match c.compare(&g) {
        Ok(o) => println!("{case}: {o}"),
        Err(e) => panic!("{case}: the replay differs from Metal's reference outputs:\n{e}"),
    }
}

/// A replay that the render path refuses: the failure's status and a word of its message.
fn refused(case: &str, f: &Failure, status: u32, needle: &str) {
    assert_eq!(f.status, status, "{case}: {f:?}");
    assert!(f.message.contains(needle), "{case}: {:?} does not mention {needle:?}", f.message);
    println!("{case}: refused as expected (status {status}): {}", f.message);
}

fn tex_layout(format: &str, w: u64, h: u64, bpr: u64) -> render_common::json::Json {
    layout(format, &[("width", w), ("height", h), ("bytes_per_row", bpr)])
}

/// `RRecordTexture`.
#[allow(clippy::too_many_arguments)]
fn record_texture(
    h: &Harness,
    c: &mut Candidate,
    name: &str,
    t: Handle,
    elem: &'static str,
    format: &str,
    w: u64,
    hh: u64,
    bpp: u64,
) -> Vec<u8> {
    let data = h.read_texture(t, w, hh, w * bpp);
    c.data_layout(name, elem, &data, tex_layout(format, w, hh, w * bpp));
    data
}

/// `RRecordRegion`.
#[allow(clippy::too_many_arguments)]
fn record_region(
    c: &mut Candidate,
    name: &str,
    image: &[u8],
    image_width: u64,
    bpp: u64,
    format: &str,
    x: u64,
    y: u64,
    w: u64,
    hh: u64,
    elem: &'static str,
) {
    let mut region = Vec::new();
    for row in 0..hh {
        let start = (((y + row) * image_width + x) * bpp) as usize;
        region.extend_from_slice(&image[start..start + (w * bpp) as usize]);
    }
    c.data_layout(
        name,
        elem,
        &region,
        layout(format, &[("x", x), ("y", y), ("width", w), ("height", hh), ("bytes_per_row", w * bpp)]),
    );
}

/// `RDraw`: one pass with one pipeline drawing `vs` from vertex buffer 0.
fn rdraw(h: &Harness, pass: RenderPassDescriptor, pipeline: Handle, primitive: u64, vs: &[RVertex]) {
    let b = vertex_buffer(h, vs);
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass,
        vec![set_pipeline(pipeline), set_vertex_buffer(b, 0, 0), draw(primitive, 0, vs.len() as u64)],
    ));
}

const MARGIN: f64 = 1.0 / 16.0;

// Raster.m ------------------------------------------------------------------------------------------------------------

#[test]
fn raster_triangle_coverage() {
    let h = Harness::new();
    let tris = [
        [1.125, 1.5625, 9.375, 2.0625, 3.4375, 10.9375],
        [14.875, 1.125, 8.75, 13.3125, 15.375, 13.0],
        [1.25, 13.125, 32.0, 13.4375, 1.25, 13.9375],
        [10.0625, 5.4375, 12.25, 7.5625, 11.15625, 6.5],
        [6.25, 5.9375, 6.875, 6.625, 6.25, 6.8125],
        [12.5625, -2.8125, 20.125, 4.3125, 11.5, 6.375],
    ];
    let ids = [1, 2, 3, 4, 5, 6];
    let expected = paint_triangles(&tris, &ids, 16, 16, MARGIN).expect("unambiguous");
    let target = h.target(pf::R32_UINT, 16, 16, 1);
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    rdraw(
        &h,
        pass_color(target, clear(0.0, 0.0, 0.0, 0.0)),
        p,
        prim::TRIANGLE,
        &triangle_vertices(&tris, &ids, 16, 16),
    );
    let mut c = Candidate::default();
    let data = record_texture(&h, &mut c, "coverage", target, "u32", "R32Uint", 16, 16, 4);
    c.bool("coverage-equals-centre-containment", words(&data) == expected);
    check("raster.triangle-coverage", &c);
}

#[test]
fn raster_edge_ties() {
    let h = Harness::new();
    let tris = [[4.5, 4.5, 11.5, 4.5, 11.5, 11.5], [4.5, 4.5, 11.5, 11.5, 4.5, 11.5]];
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    let mut c = Candidate::default();
    let mut maps = Vec::new();
    for (t, tri) in tris.iter().enumerate() {
        let target = h.target(pf::R32_UINT, 16, 16, 1);
        rdraw(
            &h,
            pass_color(target, clear(0.0, 0.0, 0.0, 0.0)),
            p,
            prim::TRIANGLE,
            &triangle_vertices(&[*tri], &[t as u32 + 1], 16, 16),
        );
        let name = if t == 1 { "lower-left" } else { "upper-right" };
        maps.push(words(&record_texture(&h, &mut c, name, target, "u32", "R32Uint", 16, 16, 4)));
    }
    let (mut both, mut neither) = (0, 0);
    for y in 4..12 {
        for x in 4..12 {
            let (a, b) = (maps[0][y * 16 + x] != 0, maps[1][y * 16 + x] != 0);
            both += (a && b) as u64;
            neither += (!a && !b) as u64;
        }
    }
    c.u64("square-centres-covered-twice", both);
    c.u64("square-centres-covered-by-neither", neither);
    check("raster.edge-ties", &c);
}

#[test]
fn raster_lines() {
    let h = Harness::new();
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    let v = |x, y, id| RVertex::at(x, y, 0.0, 16, 16, id);
    let lines = [v(1.2, 1.7, 1), v(13.8, 4.3, 1), v(2.3, 14.1, 2), v(5.1, 6.2, 2), v(8.5, 8.5, 3), v(14.5, 8.5, 3)];
    let mut c = Candidate::default();
    let t = h.target(pf::R32_UINT, 16, 16, 1);
    rdraw(&h, pass_color(t, clear(0.0, 0.0, 0.0, 0.0)), p, prim::LINE, &lines);
    record_texture(&h, &mut c, "line-list", t, "u32", "R32Uint", 16, 16, 4);
    let strip = [v(1.5, 1.5, 4), v(14.2, 2.6, 4), v(3.7, 12.9, 4), v(13.1, 14.4, 4)];
    let t2 = h.target(pf::R32_UINT, 16, 16, 1);
    rdraw(&h, pass_color(t2, clear(0.0, 0.0, 0.0, 0.0)), p, prim::LINE_STRIP, &strip);
    record_texture(&h, &mut c, "line-strip", t2, "u32", "R32Uint", 16, 16, 4);
    check("raster.lines", &c);
}

#[test]
fn raster_cull_winding() {
    let h = Harness::new();
    let tris = [
        [1.1875, 1.25, 6.4375, 1.375, 1.0, 6.5],
        [9.0625, 1.375, 9.125, 6.75, 14.875, 0.875],
        [0.8125, 9.3125, 6.6875, 8.75, 1.3125, 14.375],
        [9.3125, 9.375, 9.25, 14.375, 14.5, 8.8125],
    ];
    let ids = [1, 2, 3, 4];
    paint_triangles(&tris, &ids, 16, 16, MARGIN).expect("unambiguous");
    let b = vertex_buffer(&h, &triangle_vertices(&tris, &ids, 16, 16));
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_facing", pf::R32_UINT);
    let mut c = Candidate::default();
    for (w, wname) in ["clockwise", "counter-clockwise"].iter().enumerate() {
        for (k, cname) in ["none", "front", "back"].iter().enumerate() {
            let t = h.target(pf::R32_UINT, 16, 16, 1);
            let e = h.handle();
            h.run_ok(&render(
                e,
                pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
                vec![
                    set_pipeline(p),
                    Command::RenderSetFrontFacingWinding(RenderSetFrontFacingWinding {
                        front_facing_winding: w as u64,
                    }),
                    Command::RenderSetCullMode(RenderSetCullMode { cull_mode: k as u64 }),
                    set_vertex_buffer(b, 0, 0),
                    draw(prim::TRIANGLE, 0, 12),
                ],
            ));
            record_texture(&h, &mut c, &format!("front-{wname}.cull-{cname}"), t, "u32", "R32Uint", 16, 16, 4);
        }
    }
    check("raster.cull-winding", &c);
}

#[test]
fn raster_viewport() {
    let h = Harness::new();
    let corners = [[-1.0f32, -1.0], [3.0, -1.0], [-1.0, 3.0]];
    let quad: Vec<RVertex> = corners
        .iter()
        .map(|c| RVertex { position: [c[0], c[1], 0.5, 1.0], data: [9, 0, 0, 0], extra: [1.0, 0.0, 0.0, 0.0] })
        .collect();
    let b = vertex_buffer(&h, &quad);
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_position_w", pf::RGBA32_UINT);
    let vp = |x: f64, y: f64, w: f64, hh: f64, n: f64, f: f64| {
        Command::RenderSetViewport(RenderSetViewport {
            viewport: Viewport { origin_x: x, origin_y: y, width: w, height: hh, znear: n, zfar: f },
        })
    };
    let mut c = Candidate::default();
    let t = h.target(pf::RGBA32_UINT, 16, 16, 1);
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            set_pipeline(p),
            vp(-8.0, -8.0, 32.0, 32.0, 0.25, 0.75),
            set_vertex_buffer(b, 0, 0),
            draw(prim::TRIANGLE, 0, 3),
        ],
    ));
    let image = words(&record_texture(&h, &mut c, "position-xyzw-bits", t, "u32", "RGBA32Uint", 16, 16, 16));
    let mut centres = true;
    for y in 0..16 {
        for x in 0..16 {
            let p: Vec<f32> = (0..4).map(|k| f32::from_bits(image[(y * 16 + x) * 4 + k])).collect();
            centres &= p[0] == x as f32 + 0.5 && p[1] == y as f32 + 0.5 && p[2] == 0.5 && p[3] == 1.0;
        }
    }
    c.bool("positions-are-centres-z-half-w-one", centres);
    let t2 = h.target(pf::RGBA32_UINT, 16, 16, 1);
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass_color(t2, clear(0.0, 0.0, 0.0, 0.0)),
        vec![set_pipeline(p), vp(3.0, 5.0, 8.0, 4.0, 0.0, 1.0), set_vertex_buffer(b, 0, 0), draw(prim::TRIANGLE, 0, 3)],
    ));
    record_texture(&h, &mut c, "inner-viewport-position-xyzw-bits", t2, "u32", "RGBA32Uint", 16, 16, 16);
    check("raster.viewport", &c);
}

#[test]
fn raster_scissor() {
    let h = Harness::new();
    let b = vertex_buffer(&h, &triangle_vertices(&[[-1.0, -1.0, 40.0, -1.0, -1.0, 40.0]], &[5], 16, 16));
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    let t = h.target(pf::R32_UINT, 16, 16, 1);
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![set_pipeline(p), scissor(3, 5, 7, 4), set_vertex_buffer(b, 0, 0), draw(prim::TRIANGLE, 0, 3)],
    ));
    let mut c = Candidate::default();
    record_texture(&h, &mut c, "coverage", t, "u32", "R32Uint", 16, 16, 4);
    check("raster.scissor", &c);
}

#[test]
fn raster_depth_clip_clamp() {
    let h = Harness::new();
    let mut quads = Vec::new();
    for (q, (xs, z)) in [([-1.0f32, 0.0], 1.5f32), ([0.0, 1.0], -0.5)].iter().enumerate() {
        let corners = [[xs[0], -1.0], [xs[1], -1.0], [xs[1], 1.0], [xs[0], -1.0], [xs[1], 1.0], [xs[0], 1.0]];
        for c in corners {
            quads.push(RVertex {
                position: [c[0], c[1], *z, 1.0],
                data: [q as u32 + 1, 0, 0, 0],
                extra: [1.0, 0.0, 0.0, 0.0],
            });
        }
    }
    let b = vertex_buffer(&h, &quads);
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some("rnd_fragment_id"), pf::R32_UINT);
    d.depth_attachment_pixel_format = pf::DEPTH32_FLOAT;
    let p = h.pipeline(d).unwrap();
    let state = h.depth_stencil(depth_stencil_descriptor(7, true));
    let mut c = Candidate::default();
    for (m, name) in ["clip", "clamp"].iter().enumerate() {
        let color = h.target(pf::R32_UINT, 8, 8, 1);
        let depth = h.target(pf::DEPTH32_FLOAT, 8, 8, 1);
        let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
        let mut da = depth_attachment(depth);
        da.load_action = 2;
        da.store_action = 1;
        da.clear_depth = 0.5;
        pass.depth_attachment = Some(da);
        let e = h.handle();
        let f = h.run(&render(
            e,
            pass,
            vec![
                set_pipeline(p),
                set_depth_stencil(state),
                Command::RenderSetDepthClipMode(RenderSetDepthClipMode { depth_clip_mode: m as u64 }),
                set_vertex_buffer(b, 0, 0),
                draw(prim::TRIANGLE, 0, 12),
            ],
        ));
        if m == 1 && !h.has("depth-clip-control") {
            refused("raster.depth-clip-clamp", &f, status::HOST_UNSUPPORTED, "depth-clamp");
            return;
        }
        assert_eq!(f.status, status::OK, "{f:?}");
        record_texture(&h, &mut c, &format!("{name}.color"), color, "u32", "R32Uint", 8, 8, 4);
        record_texture(&h, &mut c, &format!("{name}.depth"), depth, "f32", "Depth32Float", 8, 8, 4);
    }
    check("raster.depth-clip-clamp", &c);
}

/// `raster.fill-mode-lines`: refused (`tables/refusals.md` row `fill-mode-lines`).
#[test]
fn raster_fill_mode_lines() {
    let h = Harness::new();
    let v = |x, y| RVertex::at(x, y, 0.0, 16, 16, 7);
    let b = vertex_buffer(&h, &[v(1.3, 1.2), v(14.2, 3.1), v(4.4, 13.6)]);
    let t = h.target(pf::R32_UINT, 16, 16, 1);
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    let e = h.handle();
    let f = h.run(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            set_pipeline(p),
            Command::RenderSetTriangleFillMode(RenderSetTriangleFillMode { fill_mode: 1 }),
            set_vertex_buffer(b, 0, 0),
            draw(prim::TRIANGLE, 0, 3),
        ],
    ));
    refused("raster.fill-mode-lines", &f, status::HOST_UNSUPPORTED, "fill-mode-lines");
}

/// `raster.error.scissor-outside`: a misuse Metal leaves undefined and WebGPU rejects (plan section 6.10 class 2).
#[test]
fn raster_error_scissor_outside() {
    let h = Harness::new();
    let b = vertex_buffer(&h, &triangle_vertices(&[[-1.0, -1.0, 40.0, -1.0, -1.0, 40.0]], &[3], 16, 16));
    let t = h.target(pf::R32_UINT, 16, 16, 1);
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    let e = h.handle();
    let f = h.run(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![set_pipeline(p), scissor(10, 10, 10, 10), set_vertex_buffer(b, 0, 0), draw(prim::TRIANGLE, 0, 3)],
    ));
    refused("raster.error.scissor-outside", &f, status::UNIMPLEMENTED, "Metal does not define");
}

// Draw.m --------------------------------------------------------------------------------------------------------------

fn ids_pipeline(h: &Harness) -> Handle {
    h.simple_pipeline("rnd_vertex_ids", "rnd_fragment_id_data", pf::RGBA32_UINT)
}

#[test]
fn draw_vertex_instance_ids() {
    let h = Harness::new();
    let p = ids_pipeline(&h);
    let t = h.target(pf::RGBA32_UINT, 16, 16, 1);
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            set_pipeline(p),
            Command::RenderDrawPrimitivesInstanced(RenderDrawPrimitivesInstanced {
                primitive_type: prim::POINT,
                vertex_start: 0,
                vertex_count: 4,
                instance_count: 2,
            }),
            Command::RenderDrawPrimitivesInstancedBaseInstance(RenderDrawPrimitivesInstancedBaseInstance {
                primitive_type: prim::POINT,
                vertex_start: 5,
                vertex_count: 6,
                instance_count: 3,
                base_instance: 9,
            }),
        ],
    ));
    let mut c = Candidate::default();
    record_texture(&h, &mut c, "ids", t, "u32", "RGBA32Uint", 16, 16, 16);
    check("draw.vertex-instance-ids", &c);
}

#[allow(clippy::too_many_arguments)]
fn indexed(
    primitive: u64,
    count: u64,
    ty: u64,
    buffer: Handle,
    offset: u64,
    instances: u64,
    base_vertex: i64,
    base_instance: u64,
) -> Command {
    Command::RenderDrawIndexedPrimitivesInstancedBaseVertexBaseInstance(
        RenderDrawIndexedPrimitivesInstancedBaseVertexBaseInstance {
            primitive_type: primitive,
            index_count: count,
            index_type: ty,
            index_buffer: Some(buffer),
            index_buffer_offset: offset,
            instance_count: instances,
            base_vertex,
            base_instance,
        },
    )
}

#[test]
fn draw_indexed() {
    let h = Harness::new();
    let p = ids_pipeline(&h);
    let i16s: Vec<u8> = [0xdeadu16, 0xbeef, 0xf00d, 0, 1, 2, 5, 9, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
    let i32s = from_words(&[0xdeadbeef, 0xfeedface, 10, 12, 14, 7]);
    let (b16, _) = h.buffer(&i16s);
    let (b32, _) = h.buffer(&i32s);
    let t = h.target(pf::RGBA32_UINT, 16, 16, 1);
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            set_pipeline(p),
            indexed(prim::POINT, 6, 0, b16, 6, 2, 4, 7),
            indexed(prim::POINT, 4, 1, b32, 8, 1, -3, 12),
            Command::RenderDrawIndexedPrimitives(RenderDrawIndexedPrimitives {
                primitive_type: prim::POINT,
                index_count: 3,
                index_type: 0,
                index_buffer: Some(b16),
                index_buffer_offset: 8,
            }),
        ],
    ));
    let mut c = Candidate::default();
    record_texture(&h, &mut c, "ids", t, "u32", "RGBA32Uint", 16, 16, 16);
    check("draw.indexed", &c);
}

#[test]
fn draw_primitive_restart() {
    let h = Harness::new();
    let corners = [[1.0, 1.0], [1.0, 5.0], [7.0, 1.0], [7.0, 5.0], [9.0, 8.0], [9.0, 14.0], [14.0, 8.0], [14.0, 14.0]];
    let vs: Vec<RVertex> =
        corners.iter().enumerate().map(|(i, c)| RVertex::at(c[0], c[1], 0.0, 16, 16, i as u32 / 4 + 1)).collect();
    let vb = vertex_buffer(&h, &vs);
    let s16: Vec<u8> = [0u16, 1, 2, 3, 0xffff, 4, 5, 6, 7].iter().flat_map(|v| v.to_le_bytes()).collect();
    let s32 = from_words(&[0, 1, 2, 3, 0xffff_ffff, 4, 5, 6, 7]);
    let (b16, _) = h.buffer(&s16);
    let (b32, _) = h.buffer(&s32);
    let tris = [
        [1.0, 1.0, 1.0, 5.0, 7.0, 1.0],
        [1.0, 5.0, 7.0, 1.0, 7.0, 5.0],
        [9.0, 8.0, 9.0, 14.0, 14.0, 8.0],
        [9.0, 14.0, 14.0, 8.0, 14.0, 14.0],
    ];
    let expected = paint_triangles(&tris, &[1, 1, 2, 2], 16, 16, MARGIN).expect("unambiguous");
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    let mut c = Candidate::default();
    for (ty, it, name) in [
        (prim::TRIANGLE_STRIP, 0, "triangle-strip.uint16"),
        (prim::TRIANGLE_STRIP, 1, "triangle-strip.uint32"),
        (prim::LINE_STRIP, 0, "line-strip.uint16"),
        (prim::LINE_STRIP, 1, "line-strip.uint32"),
    ] {
        let t = h.target(pf::R32_UINT, 16, 16, 1);
        let e = h.handle();
        h.run_ok(&render(
            e,
            pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
            vec![
                set_pipeline(p),
                set_vertex_buffer(vb, 0, 0),
                Command::RenderDrawIndexedPrimitives(RenderDrawIndexedPrimitives {
                    primitive_type: ty,
                    index_count: 9,
                    index_type: it,
                    index_buffer: Some(if it == 0 { b16 } else { b32 }),
                    index_buffer_offset: 0,
                }),
            ],
        ));
        let image = record_texture(&h, &mut c, name, t, "u32", "R32Uint", 16, 16, 4);
        if ty == prim::TRIANGLE_STRIP {
            c.bool(&format!("{name}.equals-two-separate-quads"), words(&image) == expected);
        }
    }
    check("draw.primitive-restart", &c);
}

#[test]
fn draw_indirect() {
    let h = Harness::new();
    let p = ids_pipeline(&h);
    let mut args = [0xcccc_ccccu32; 24];
    args[4..8].copy_from_slice(&[3, 2, 4, 6]);
    args[12..17].copy_from_slice(&[4, 2, 1, 5, 11]);
    let (indirect, _) = h.buffer(&from_words(&args));
    let indices: Vec<u8> = [0xffffu16, 0, 2, 4, 6, 8].iter().flat_map(|v| v.to_le_bytes()).collect();
    let (ib, _) = h.buffer(&indices);
    let t = h.target(pf::RGBA32_UINT, 16, 16, 1);
    let e = h.handle();
    let f = h.run(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            set_pipeline(p),
            Command::RenderDrawPrimitivesIndirect(RenderDrawPrimitivesIndirect {
                primitive_type: prim::POINT,
                indirect_buffer: Some(indirect),
                indirect_buffer_offset: 16,
            }),
            Command::RenderDrawIndexedPrimitivesIndirect(RenderDrawIndexedPrimitivesIndirect {
                primitive_type: prim::POINT,
                index_type: 0,
                index_buffer: Some(ib),
                index_buffer_offset: 0,
                indirect_buffer: Some(indirect),
                indirect_buffer_offset: 48,
            }),
        ],
    ));
    if !h.has("indirect-first-instance") {
        refused("draw.indirect", &f, status::HOST_UNSUPPORTED, "indirect-first-instance");
        return;
    }
    assert_eq!(f.status, status::OK, "{f:?}");
    let mut c = Candidate::default();
    c.data("arguments", "u32", &from_words(&args));
    record_texture(&h, &mut c, "ids", t, "u32", "RGBA32Uint", 16, 16, 16);
    check("draw.indirect", &c);
}

/// `draw.error.index-offset-unaligned`: Metal requires `indexBufferOffset` to be a multiple of the index size
/// (`MTLRenderCommandEncoder.h`); a misuse with an undefined result, which WebGPU rejects too.
#[test]
fn draw_error_index_offset_unaligned() {
    let h = Harness::new();
    let p = ids_pipeline(&h);
    let (ib, _) = h.buffer(&[0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0]);
    let t = h.target(pf::RGBA32_UINT, 16, 16, 1);
    let e = h.handle();
    let f = h.run(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            set_pipeline(p),
            Command::RenderDrawIndexedPrimitives(RenderDrawIndexedPrimitives {
                primitive_type: prim::POINT,
                index_count: 3,
                index_type: 1,
                index_buffer: Some(ib),
                index_buffer_offset: 2,
            }),
        ],
    ));
    refused("draw.error.index-offset-unaligned", &f, status::UNIMPLEMENTED, "multiple of the index size");
}

/// A pipeline whose vertex function fetches attribute 0 (UInt2) from buffer 1.
fn fetch_uint_pipeline(h: &Harness, stride: u64) -> Handle {
    let mut d = h.pipeline_of("rnd_vertex_fetch_uint", Some("rnd_fragment_id_data"), pf::RGBA32_UINT);
    d.vertex_descriptor = Some(VertexDescriptor {
        attributes: (0..31)
            .map(|i| VertexAttribute {
                buffer_index: if i == 0 { 1 } else { 0 },
                format: if i == 0 { 37 } else { 0 },
                offset: 0,
            })
            .collect(),
        layouts: (0..31)
            .map(|i| VertexBufferLayout { step_function: 1, step_rate: 1, stride: if i == 1 { stride } else { 0 } })
            .collect(),
    });
    h.pipeline(d).unwrap()
}

#[test]
fn vertex_bytes_and_buffer_offset() {
    let h = Harness::new();
    let p = fetch_uint_pipeline(&h, 8);
    let bytes = from_words(&(0..8).map(|i| 0x1000 + i).collect::<Vec<_>>());
    let (buffer, _) = h.buffer(&from_words(&(0..32).map(|i| 0x2000 + i).collect::<Vec<_>>()));
    let t = h.target(pf::RGBA32_UINT, 16, 16, 1);
    let inst = |base: u64| {
        Command::RenderDrawPrimitivesInstancedBaseInstance(RenderDrawPrimitivesInstancedBaseInstance {
            primitive_type: prim::POINT,
            vertex_start: 0,
            vertex_count: 4,
            instance_count: 1,
            base_instance: base,
        })
    };
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            set_pipeline(p),
            Command::RenderSetVertexBytes(RenderSetVertexBytes { bytes: bytes.clone(), length: 32, index: 1 }),
            draw(prim::POINT, 0, 4),
            set_vertex_buffer(buffer, 0, 1),
            inst(1),
            Command::RenderSetVertexBufferOffset(RenderSetVertexBufferOffset { offset: 40, index: 1 }),
            inst(2),
            Command::RenderSetVertexBufferOffset(RenderSetVertexBufferOffset { offset: 64, index: 1 }),
            inst(3),
        ],
    ));
    let image = h.read_texture(t, 16, 16, 256);
    let mut c = Candidate::default();
    record_region(&mut c, "fetched", &image, 16, 16, "RGBA32Uint", 0, 0, 4, 4, "u32");
    check("vertex.bytes-and-buffer-offset", &c);
}

#[test]
fn vertex_attribute_stride() {
    let h = Harness::new();
    let p = fetch_uint_pipeline(&h, u64::MAX);
    let (buffer, _) = h.buffer(&from_words(&(0..64).map(|i| 0x3000 + i).collect::<Vec<_>>()));
    let t = h.target(pf::RGBA32_UINT, 16, 16, 1);
    let inst = |base: u64| {
        Command::RenderDrawPrimitivesInstancedBaseInstance(RenderDrawPrimitivesInstancedBaseInstance {
            primitive_type: prim::POINT,
            vertex_start: 0,
            vertex_count: 4,
            instance_count: 1,
            base_instance: base,
        })
    };
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            set_pipeline(p),
            Command::RenderSetVertexBufferAttributeStride(RenderSetVertexBufferAttributeStride {
                buffer: Some(buffer),
                offset: 0,
                stride: 8,
                index: 1,
            }),
            draw(prim::POINT, 0, 4),
            Command::RenderSetVertexBufferOffsetAttributeStride(RenderSetVertexBufferOffsetAttributeStride {
                offset: 4,
                stride: 16,
                index: 1,
            }),
            inst(1),
            Command::RenderSetVertexBufferAttributeStride(RenderSetVertexBufferAttributeStride {
                buffer: Some(buffer),
                offset: 8,
                stride: 12,
                index: 1,
            }),
            inst(2),
        ],
    ));
    let image = h.read_texture(t, 16, 16, 256);
    let mut c = Candidate::default();
    record_region(&mut c, "fetched", &image, 16, 16, "RGBA32Uint", 0, 0, 4, 3, "u32");
    check("vertex.attribute-stride", &c);
}

fn fetch_two_pipeline(h: &Harness, step2: u64, rate2: u64) -> Handle {
    let mut d = h.pipeline_of("rnd_vertex_fetch_two", Some("rnd_fragment_id_data"), pf::RGBA32_UINT);
    d.vertex_descriptor = Some(VertexDescriptor {
        attributes: (0..31)
            .map(|i| VertexAttribute { buffer_index: i + 1, format: if i < 2 { 37 } else { 0 }, offset: 0 })
            .collect(),
        layouts: (0..31)
            .map(|i| match i {
                1 => VertexBufferLayout { step_function: 1, step_rate: 1, stride: 8 },
                2 => VertexBufferLayout { step_function: step2, step_rate: rate2, stride: 8 },
                _ => VertexBufferLayout { step_function: 1, step_rate: 1, stride: 0 },
            })
            .collect(),
    });
    h.pipeline(d).unwrap()
}

#[test]
fn vertex_step_functions() {
    let h = Harness::new();
    let pv: Vec<u32> = (0..16).flat_map(|i| [100 + i, 200 + i]).collect();
    let pi: Vec<u32> = (0..16).flat_map(|i| [300 + i, 400 + i]).collect();
    let (vd, _) = h.buffer(&from_words(&pv));
    let (id, _) = h.buffer(&from_words(&pi));
    let mut c = Candidate::default();
    for (step, rate, name) in [(2, 2, "per-instance-rate-2"), (0, 0, "constant")] {
        let p = fetch_two_pipeline(&h, step, rate);
        let t = h.target(pf::RGBA32_UINT, 16, 16, 1);
        let e = h.handle();
        h.run_ok(&render(
            e,
            pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
            vec![
                set_pipeline(p),
                set_vertex_buffer(vd, 0, 1),
                set_vertex_buffer(id, 0, 2),
                Command::RenderDrawPrimitivesInstanced(RenderDrawPrimitivesInstanced {
                    primitive_type: prim::POINT,
                    vertex_start: 0,
                    vertex_count: 4,
                    instance_count: 5,
                }),
            ],
        ));
        let image = h.read_texture(t, 16, 16, 256);
        record_region(&mut c, name, &image, 16, 16, "RGBA32Uint", 0, 0, 4, 5, "u32");
    }
    check("vertex.step-functions", &c);
}

#[test]
fn vertex_step_base_instance() {
    let h = Harness::new();
    let pv: Vec<u32> = (0..4).flat_map(|i| [i, i]).collect();
    let pi: Vec<u32> = (0..16).flat_map(|i| [300 + i, 300 + i]).collect();
    let (vd, _) = h.buffer(&from_words(&pv));
    let (id, _) = h.buffer(&from_words(&pi));
    let p = fetch_two_pipeline(&h, 2, 2);
    let t = h.target(pf::RGBA32_UINT, 16, 16, 1);
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            set_pipeline(p),
            set_vertex_buffer(vd, 0, 1),
            set_vertex_buffer(id, 0, 2),
            Command::RenderDrawPrimitivesInstancedBaseInstance(RenderDrawPrimitivesInstancedBaseInstance {
                primitive_type: prim::POINT,
                vertex_start: 0,
                vertex_count: 2,
                instance_count: 4,
                base_instance: 3,
            }),
        ],
    ));
    let image = h.read_texture(t, 16, 16, 256);
    let mut c = Candidate::default();
    record_region(&mut c, "fetched", &image, 16, 16, "RGBA32Uint", 0, 3, 2, 4, "u32");
    check("vertex.step-base-instance", &c);
}

// DepthStencil.m ------------------------------------------------------------------------------------------------------

const DSW: u64 = 16;
const DSH: u64 = 4;

fn ds_cover(h: &Harness, z: f64, id: u32) -> Handle {
    vertex_buffer(
        h,
        &[
            RVertex::at(-1.0, -1.0, z, DSW, DSH, id),
            RVertex::at(3.0 * DSW as f64, -1.0, z, DSW, DSH, id),
            RVertex::at(-1.0, 3.0 * DSH as f64, z, DSW, DSH, id),
        ],
    )
}

fn ds_aspect(h: &Harness, c: &mut Candidate, name: &str, t: Handle, stencil: bool) {
    let bpp = if stencil { 1 } else { 4 };
    let data = h.read_texture_options(t, 0, DSW, DSH, DSW * bpp, if stencil { 2 } else { 1 });
    let format = if stencil { "Depth32Float_Stencil8.stencil" } else { "Depth32Float_Stencil8.depth" };
    c.data_layout(name, if stencil { "u8" } else { "f32" }, &data, tex_layout(format, DSW, DSH, DSW * bpp));
}

#[test]
fn depth_stencil_compare_functions() {
    let h = Harness::new();
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some("rnd_fragment_id"), pf::R32_UINT);
    d.depth_attachment_pixel_format = pf::DEPTH32_FLOAT;
    let p = h.pipeline(d).unwrap();
    let covers = [ds_cover(&h, 0.25, 1), ds_cover(&h, 0.5, 2), ds_cover(&h, 0.75, 3)];
    let names = ["never", "less", "equal", "less-equal", "greater", "not-equal", "greater-equal", "always"];
    let mut c = Candidate::default();
    for write in [true, false] {
        for (f, name) in names.iter().enumerate() {
            let state = h.depth_stencil(depth_stencil_descriptor(f as u64, write));
            let color = h.target(pf::R32_UINT, DSW, DSH, 1);
            let depth = h.target(pf::DEPTH32_FLOAT, DSW, DSH, 1);
            let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
            let mut da = depth_attachment(depth);
            da.load_action = 2;
            da.store_action = 1;
            da.clear_depth = 0.5;
            pass.depth_attachment = Some(da);
            let mut body = vec![set_pipeline(p), set_depth_stencil(state)];
            for (k, cover) in covers.iter().enumerate() {
                body.push(scissor(k as u64 * 5, 0, if k == 2 { 6 } else { 5 }, DSH));
                body.push(set_vertex_buffer(*cover, 0, 0));
                body.push(draw(prim::TRIANGLE, 0, 3));
            }
            let e = h.handle();
            h.run_ok(&render(e, pass, body));
            let prefix = format!("{name}.write-{}", if write { "on" } else { "off" });
            record_texture(&h, &mut c, &format!("{prefix}.color"), color, "u32", "R32Uint", DSW, DSH, 4);
            record_texture(&h, &mut c, &format!("{prefix}.depth"), depth, "f32", "Depth32Float", DSW, DSH, 4);
        }
    }
    check("depth-stencil.compare-functions", &c);
}

#[test]
fn depth_stencil_stencil_operations() {
    let h = Harness::new();
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some("rnd_fragment_id"), pf::R32_UINT);
    d.depth_attachment_pixel_format = pf::DEPTH32_FLOAT_STENCIL8;
    d.stencil_attachment_pixel_format = pf::DEPTH32_FLOAT_STENCIL8;
    let p = h.pipeline(d).unwrap();
    let cover = ds_cover(&h, 0.5, 1);
    let names =
        ["keep", "zero", "replace", "increment-clamp", "decrement-clamp", "invert", "increment-wrap", "decrement-wrap"];
    let mut c = Candidate::default();
    for clear_value in [0xfeu32, 0x01] {
        for (op, name) in names.iter().enumerate() {
            let op = op as u64;
            let states: Vec<Handle> = (0..3)
                .map(|k| {
                    let s = stencil(
                        if k == 0 { 0 } else { 7 },
                        if k == 0 { op } else { 0 },
                        if k == 1 { op } else { 0 },
                        if k == 2 { op } else { 0 },
                        0xff,
                        0xf7,
                    );
                    let mut ds = depth_stencil_descriptor(if k == 1 { 0 } else { 7 }, false);
                    ds.front_face_stencil = Some(s.clone());
                    ds.back_face_stencil = Some(s);
                    h.depth_stencil(ds)
                })
                .collect();
            let color = h.target(pf::R32_UINT, DSW, DSH, 1);
            let dst = h.target(pf::DEPTH32_FLOAT_STENCIL8, DSW, DSH, 1);
            let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
            let mut da = depth_attachment(dst);
            da.load_action = 2;
            da.store_action = 1;
            da.clear_depth = 0.5;
            pass.depth_attachment = Some(da);
            let mut sa = stencil_attachment(dst);
            sa.load_action = 2;
            sa.store_action = 1;
            sa.clear_stencil = clear_value;
            pass.stencil_attachment = Some(sa);
            let mut body = vec![
                set_pipeline(p),
                set_vertex_buffer(cover, 0, 0),
                Command::RenderSetStencilReferenceValue(RenderSetStencilReferenceValue { reference_value: 0x3c }),
            ];
            for k in 0..3u64 {
                body.push(set_depth_stencil(states[k as usize]));
                body.push(scissor(k * 5, 0, if k == 2 { 6 } else { 5 }, DSH));
                body.push(draw(prim::TRIANGLE, 0, 3));
            }
            let e = h.handle();
            h.run_ok(&render(e, pass, body));
            let prefix = format!("{name}.clear-{clear_value:02x}");
            ds_aspect(&h, &mut c, &format!("{prefix}.stencil"), dst, true);
            record_texture(&h, &mut c, &format!("{prefix}.color"), color, "u32", "R32Uint", DSW, DSH, 4);
        }
    }
    check("depth-stencil.stencil-operations", &c);
}

#[test]
fn depth_stencil_stencil_read_mask() {
    let h = Harness::new();
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some("rnd_fragment_id"), pf::R32_UINT);
    d.stencil_attachment_pixel_format = pf::STENCIL8;
    let p = h.pipeline(d).unwrap();
    let cover = ds_cover(&h, 0.5, 1);
    let tests = [(2u64, 0xefu32), (2, 0xff), (1, 0xff), (4, 0xff)];
    let states: Vec<Handle> = tests
        .iter()
        .map(|&(f, mask)| {
            let s = stencil(f, 0, 0, 0, mask, 0xff);
            let mut ds = depth_stencil_descriptor(7, false);
            ds.front_face_stencil = Some(s.clone());
            ds.back_face_stencil = Some(s);
            h.depth_stencil(ds)
        })
        .collect();
    let color = h.target(pf::R32_UINT, DSW, DSH, 1);
    let st = h.target(pf::STENCIL8, DSW, DSH, 1);
    let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
    let mut sa = stencil_attachment(st);
    sa.load_action = 2;
    sa.store_action = 1;
    sa.clear_stencil = 0x5a;
    pass.stencil_attachment = Some(sa);
    let mut body = vec![
        set_pipeline(p),
        set_vertex_buffer(cover, 0, 0),
        Command::RenderSetStencilReferenceValue(RenderSetStencilReferenceValue { reference_value: 0x4a }),
    ];
    for (k, s) in states.iter().enumerate() {
        body.push(set_depth_stencil(*s));
        body.push(scissor(k as u64 * 4, 0, 4, DSH));
        body.push(draw(prim::TRIANGLE, 0, 3));
    }
    let e = h.handle();
    h.run_ok(&render(e, pass, body));
    let mut c = Candidate::default();
    record_texture(&h, &mut c, "color", color, "u32", "R32Uint", DSW, DSH, 4);
    record_texture(&h, &mut c, "stencil", st, "u8", "Stencil8", DSW, DSH, 1);
    check("depth-stencil.stencil-read-mask", &c);
}

#[test]
fn depth_stencil_separate_front_back() {
    let h = Harness::new();
    let (w, hh) = (16u64, 8u64);
    let tris = [[0.25, 0.25, 7.5, 0.25, 0.25, 7.5], [8.25, 0.25, 8.25, 7.5, 15.5, 0.25]];
    let mut vs = Vec::new();
    for (t, tri) in tris.iter().enumerate() {
        for k in 0..3 {
            vs.push(RVertex::at(tri[2 * k], tri[2 * k + 1], 0.5, w, hh, t as u32 + 1));
        }
    }
    let b = vertex_buffer(&h, &vs);
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some("rnd_fragment_facing"), pf::R32_UINT);
    d.stencil_attachment_pixel_format = pf::STENCIL8;
    let p = h.pipeline(d).unwrap();
    let equal = {
        let s = stencil(2, 0, 0, 0, 0xff, 0xff);
        let mut ds = depth_stencil_descriptor(7, false);
        ds.front_face_stencil = Some(s.clone());
        ds.back_face_stencil = Some(s);
        ds
    };
    let split = {
        let mut ds = depth_stencil_descriptor(7, false);
        ds.front_face_stencil = Some(stencil(7, 0, 0, 2, 0xff, 0xff));
        ds.back_face_stencil = Some(stencil(7, 0, 0, 5, 0xff, 0xff));
        ds
    };
    let mut c = Candidate::default();
    for (state, clear_value, front, back, name) in [
        (equal.clone(), 1u32, 1u32, 2u32, "equal-clear-1"),
        (equal, 2, 1, 2, "equal-clear-2"),
        (split, 0x0f, 0x21, 0x42, "replace-front-invert-back"),
    ] {
        let s = h.depth_stencil(state);
        let color = h.target(pf::R32_UINT, w, hh, 1);
        let st = h.target(pf::STENCIL8, w, hh, 1);
        let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
        let mut sa = stencil_attachment(st);
        sa.load_action = 2;
        sa.store_action = 1;
        sa.clear_stencil = clear_value;
        pass.stencil_attachment = Some(sa);
        let e = h.handle();
        h.run_ok(&render(
            e,
            pass,
            vec![
                set_pipeline(p),
                set_depth_stencil(s),
                Command::RenderSetFrontFacingWinding(RenderSetFrontFacingWinding { front_facing_winding: 0 }),
                Command::RenderSetStencilFrontBackReferenceValue(RenderSetStencilFrontBackReferenceValue {
                    front_reference_value: front,
                    back_reference_value: back,
                }),
                set_vertex_buffer(b, 0, 0),
                draw(prim::TRIANGLE, 0, 6),
            ],
        ));
        record_texture(&h, &mut c, &format!("{name}.color"), color, "u32", "R32Uint", w, hh, 4);
        record_texture(&h, &mut c, &format!("{name}.stencil"), st, "u8", "Stencil8", w, hh, 1);
    }
    check("depth-stencil.separate-front-back", &c);
}

#[test]
fn depth_stencil_fragment_depth() {
    let h = Harness::new();
    let state = h.depth_stencil(depth_stencil_descriptor(7, true));
    let mut c = Candidate::default();
    for (fragment, z, name) in [
        ("rnd_fragment_depth_any", 0.5, "any"),
        ("rnd_fragment_depth_greater", 0.25, "greater"),
        ("rnd_fragment_depth_less", 0.75, "less"),
    ] {
        let mut d = h.pipeline_of("rnd_vertex_triangles", Some(fragment), pf::R32_UINT);
        d.depth_attachment_pixel_format = pf::DEPTH32_FLOAT;
        let p = h.pipeline(d).unwrap();
        let cover = ds_cover(&h, z, 1);
        let color = h.target(pf::R32_UINT, DSW, DSH, 1);
        let depth = h.target(pf::DEPTH32_FLOAT, DSW, DSH, 1);
        let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
        let mut da = depth_attachment(depth);
        da.load_action = 2;
        da.store_action = 1;
        da.clear_depth = 1.0;
        pass.depth_attachment = Some(da);
        let e = h.handle();
        h.run_ok(&render(
            e,
            pass,
            vec![set_pipeline(p), set_depth_stencil(state), set_vertex_buffer(cover, 0, 0), draw(prim::TRIANGLE, 0, 3)],
        ));
        record_texture(&h, &mut c, &format!("{name}.depth"), depth, "f32", "Depth32Float", DSW, DSH, 4);
    }
    check("depth-stencil.fragment-depth", &c);
}

/// `depth-stencil.depth-bias`: the unbiased draws run; the first biased draw is refused, since Metal does not define
/// the bias formula and the A10X's is not measured (`tables/limits.toml` question `6.4-depth-bias-formula`).
#[test]
fn depth_stencil_depth_bias() {
    let h = Harness::new();
    let state = h.depth_stencil(depth_stencil_descriptor(7, true));
    let flat = vertex_buffer(
        &h,
        &[
            RVertex::at(-1.0, -1.0, 0.5, 16, 16, 1),
            RVertex::at(48.0, -1.0, 0.5, 16, 16, 1),
            RVertex::at(-1.0, 48.0, 0.5, 16, 16, 1),
        ],
    );
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some("rnd_fragment_id"), pf::R32_UINT);
    d.depth_attachment_pixel_format = pf::DEPTH32_FLOAT;
    let p = h.pipeline(d).unwrap();
    let run = |bias: f32| {
        let color = h.target(pf::R32_UINT, 16, 16, 1);
        let depth = h.target(pf::DEPTH32_FLOAT, 16, 16, 1);
        let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
        let mut da = depth_attachment(depth);
        da.load_action = 2;
        da.store_action = 1;
        pass.depth_attachment = Some(da);
        let e = h.handle();
        let f = h.run(&render(
            e,
            pass,
            vec![
                set_pipeline(p),
                set_depth_stencil(state),
                Command::RenderSetDepthBias(RenderSetDepthBias { depth_bias: bias, slope_scale: 0.0, clamp: 0.0 }),
                set_vertex_buffer(flat, 0, 0),
                draw(prim::TRIANGLE, 0, 3),
            ],
        ));
        (f, depth)
    };
    let (f, depth) = run(0.0);
    assert_eq!(f.status, status::OK, "{f:?}");
    let g = Golden::load("depth-stencil.depth-bias");
    assert_eq!(h.read_texture(depth, 16, 16, 64), g.data("depth32float.flat.none"), "the exact unbiased record");
    let (f, _) = run(64.0);
    refused("depth-stencil.depth-bias", &f, status::UNIMPLEMENTED, "6.4-depth-bias-formula");
}

/// `depth-stencil.error.test-without-attachment`: a misuse whose result Metal does not define.
#[test]
fn depth_stencil_error_test_without_attachment() {
    let h = Harness::new();
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    let state = h.depth_stencil(depth_stencil_descriptor(1, true));
    let cover = ds_cover(&h, 0.5, 1);
    let color = h.target(pf::R32_UINT, DSW, DSH, 1);
    let e = h.handle();
    let f = h.run(&render(
        e,
        pass_color(color, clear(0.0, 0.0, 0.0, 0.0)),
        vec![set_pipeline(p), set_depth_stencil(state), set_vertex_buffer(cover, 0, 0), draw(prim::TRIANGLE, 0, 3)],
    ));
    refused("depth-stencil.error.test-without-attachment", &f, status::UNIMPLEMENTED, "no such attachment");
}

// Blend.m -------------------------------------------------------------------------------------------------------------

const SOURCE: [f64; 4] = [0.75, 0.5, 0.25, 0.5];
const SOURCE1: [f64; 4] = [0.125, 0.375, 0.625, 0.875];
const DESTINATION: [f64; 4] = [0.25, 0.5, 0.75, 0.25];
const BLEND_COLOR: [f64; 4] = [0.5, 0.25, 0.125, 0.75];

/// (srcRGB, dstRGB, srcA, dstA, opRGB, opA)
type BState = (u64, u64, u64, u64, u64, u64);

fn blend_factor(f: u64, i: usize) -> f64 {
    let (s, d, b, s1) = (SOURCE, DESTINATION, BLEND_COLOR, SOURCE1);
    match f {
        0 => 0.0,
        1 => 1.0,
        2 => s[i],
        3 => 1.0 - s[i],
        4 => s[3],
        5 => 1.0 - s[3],
        6 => d[i],
        7 => 1.0 - d[i],
        8 => d[3],
        9 => 1.0 - d[3],
        10 => {
            if i < 3 {
                s[3].min(1.0 - d[3])
            } else {
                1.0
            }
        }
        11 => b[i],
        12 => 1.0 - b[i],
        13 => b[3],
        14 => 1.0 - b[3],
        15 => s1[i],
        16 => 1.0 - s1[i],
        17 => s1[3],
        18 => 1.0 - s1[3],
        _ => f64::NAN,
    }
}

fn blend_equation(op: u64, s: f64, fs: f64, d: f64, fd: f64) -> f64 {
    match op {
        0 => s * fs + d * fd,
        1 => s * fs - d * fd,
        2 => d * fd - s * fs,
        3 => s.min(d),
        4 => s.max(d),
        _ => f64::NAN,
    }
}

fn half_to_f64(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    match e {
        0 => s * m * 2f64.powi(-24),
        31 => {
            if m == 0.0 {
                s * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => s * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

fn blend_states(h: &Harness, states: &[BState], dual: bool) -> Option<Vec<u8>> {
    let t = h.target(pf::RGBA16_FLOAT, 16, 4, 1);
    let pass = pass_color(t, clear(DESTINATION[0], DESTINATION[1], DESTINATION[2], DESTINATION[3]));
    let mut pipelines = Vec::new();
    for s in states {
        let mut d = h.pipeline_of(
            "rnd_vertex",
            Some(if dual { "rnd_fragment_dual_source" } else { "rnd_fragment_float_point" }),
            pf::RGBA16_FLOAT,
        );
        let a = &mut d.color_attachments[0];
        a.blending_enabled = true;
        (a.source_rgb_blend_factor, a.destination_rgb_blend_factor) = (s.0, s.1);
        (a.source_alpha_blend_factor, a.destination_alpha_blend_factor) = (s.2, s.3);
        (a.rgb_blend_operation, a.alpha_blend_operation) = (s.4, s.5);
        match h.pipeline(d) {
            Ok(p) => pipelines.push(p),
            Err(f) if dual => {
                println!("blend.dual-source: the pipeline is refused on this host: {}", f.message);
                return None;
            }
            Err(f) => panic!("{f:?}"),
        }
    }
    let source: [u32; 4] = SOURCE.map(|v| (v as f32).to_bits());
    let points: Vec<RVertex> = (0..states.len())
        .map(|k| {
            let mut v = RVertex::at((k % 16) as f64 + 0.5, (k / 16) as f64 + 0.5, 0.0, 16, 4, 0);
            v.data = source;
            v
        })
        .collect();
    let b = vertex_buffer(h, &points);
    let second: Vec<u8> = SOURCE1.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect();
    let mut body = vec![
        Command::RenderSetBlendColor(RenderSetBlendColor {
            red: BLEND_COLOR[0] as f32,
            green: BLEND_COLOR[1] as f32,
            blue: BLEND_COLOR[2] as f32,
            alpha: BLEND_COLOR[3] as f32,
        }),
        set_vertex_buffer(b, 0, 0),
    ];
    if dual {
        body.push(Command::RenderSetFragmentBytes(RenderSetFragmentBytes { bytes: second, length: 16, index: 0 }));
    }
    for (k, p) in pipelines.iter().enumerate() {
        body.push(set_pipeline(*p));
        body.push(draw(prim::POINT, k as u64, 1));
    }
    let e = h.handle();
    h.run_ok(&render(e, pass, body));
    Some(h.read_texture(t, 16, 4, 128))
}

fn record_states(h: &Harness, c: &mut Candidate, name: &str, states: &[BState], dual: bool) -> bool {
    let Some(image) = blend_states(h, states, dual) else { return false };
    let halves: Vec<u16> = image.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut equal = true;
    for (k, s) in states.iter().enumerate() {
        for i in 0..4 {
            let rgb = i < 3;
            let expected = blend_equation(
                if rgb { s.4 } else { s.5 },
                SOURCE[i],
                blend_factor(if rgb { s.0 } else { s.2 }, i),
                DESTINATION[i],
                blend_factor(if rgb { s.1 } else { s.3 }, i),
            );
            equal &= half_to_f64(halves[k * 4 + i]) == expected;
        }
    }
    c.data_layout(
        name,
        "f16",
        &image[..states.len() * 8],
        layout("RGBA16Float", &[("width", 16), ("height", 4), ("bytes_per_row", 128)]),
    );
    c.bool(&format!("{name}.equals-documented-equations"), equal);
    true
}

#[test]
fn blend_factors_operations() {
    let h = Harness::new();
    let mut states = Vec::new();
    for f in 0..=14 {
        states.push((f, 0, f, 0, 0, 0));
        states.push((0, f, 0, f, 0, 0));
    }
    for op in 0..=4 {
        states.push((4, 5, 1, 1, op, op));
    }
    states.push((4, 5, 1, 0, 0, 0));
    states.push((6, 11, 13, 8, 2, 4));
    let mut c = Candidate::default();
    assert!(record_states(&h, &mut c, "results", &states, false));
    check("blend.factors-operations", &c);
}

#[test]
fn blend_dual_source() {
    let h = Harness::new();
    let mut states = Vec::new();
    for f in 15..=18 {
        states.push((f, 0, f, 0, 0, 0));
        states.push((1, f, 1, f, 0, 0));
    }
    let mut c = Candidate::default();
    if record_states(&h, &mut c, "results", &states, true) {
        check("blend.dual-source", &c);
    } else {
        assert!(!h.has("dual-source-blending"), "the pipeline was refused although the host has dual-source-blending");
    }
}

#[test]
fn blend_write_mask() {
    let h = Harness::new();
    let masks = [0u64, 8 | 1, 4 | 2, 0xf];
    let t = h.target(pf::RGBA16_FLOAT, 8, 1, 1);
    let pass = pass_color(t, clear(DESTINATION[0], DESTINATION[1], DESTINATION[2], DESTINATION[3]));
    let mut pipelines = Vec::new();
    for blend in [false, true] {
        for m in masks {
            let mut d = h.pipeline_of("rnd_vertex", Some("rnd_fragment_float_point"), pf::RGBA16_FLOAT);
            let a = &mut d.color_attachments[0];
            a.write_mask = m;
            a.blending_enabled = blend;
            (
                a.source_rgb_blend_factor,
                a.destination_rgb_blend_factor,
                a.source_alpha_blend_factor,
                a.destination_alpha_blend_factor,
            ) = (1, 1, 1, 1);
            pipelines.push(h.pipeline(d).unwrap());
        }
    }
    let source: [u32; 4] = SOURCE.map(|v| (v as f32).to_bits());
    let points: Vec<RVertex> = (0..8)
        .map(|k| {
            let mut v = RVertex::at(k as f64 + 0.5, 0.5, 0.0, 8, 1, 0);
            v.data = source;
            v
        })
        .collect();
    let b = vertex_buffer(&h, &points);
    let mut body = vec![set_vertex_buffer(b, 0, 0)];
    for (k, p) in pipelines.iter().enumerate() {
        body.push(set_pipeline(*p));
        body.push(draw(prim::POINT, k as u64, 1));
    }
    let e = h.handle();
    h.run_ok(&render(e, pass, body));
    let mut c = Candidate::default();
    record_texture(&h, &mut c, "results", t, "f16", "RGBA16Float", 8, 1, 8);
    check("blend.write-mask", &c);
}

// Multisampling (Blend.m) ---------------------------------------------------------------------------------------------

/// The color format of the multisampled attachments of the multisample cases. The cases use `R32Uint`, which WebGPU
/// cannot multisample (`tables/formats.toml`: r32uint has no multisampling), so the format module refuses such a
/// texture ([`msaa_r32uint_attachment_refused`]); the replays use `R16Uint`, which holds every value the cases write
/// (ids, sample indices and masks below 2^16) and which the multisample readback kernel reads the same way.
const MS_COLOR: u64 = 23;

/// Can the host multisample `format`? WebGPU offers multisampled r16uint and rgba16float only with
/// "core-features-and-limits" (`tables/formats.toml`), which wgpu-native does not report
/// (`host/tests/gpu_smoke_known_wgpu-native.txt`); there the format module refuses the texture, which is asserted.
fn multisamples(h: &Harness, case: &str, format: u64) -> bool {
    let mut d = texture_descriptor(format, 4, 4, STORAGE_PRIVATE, usage::RENDER_TARGET | usage::SHADER_READ);
    d.texture_type = 4;
    d.sample_count = 4;
    match h.texture(d) {
        Ok(_) => true,
        Err(f) => {
            assert!(!h.has("core-features-and-limits"), "{case}: {f:?}");
            refused(case, &f, status::HOST_UNSUPPORTED, "multisampling");
            false
        }
    }
}

const BMW: u64 = 8;
const BMH: u64 = 8;
const AMBIGUOUS: u32 = 0xffff_ffff;
const MS_TRIANGLE: [f64; 6] = [1.0625, 0.8125, 7.4375, 2.3125, 2.1875, 7.6875];
/// The 4-sample positions, WebGPU's standard pattern (`tables/limits.toml` question `6.4-sample-positions-4x`), which
/// the reference records as the device's default positions.
const POSITIONS_4: [[f64; 2]; 4] = [[0.375, 0.125], [0.875, 0.375], [0.125, 0.625], [0.625, 0.875]];

fn classify() -> Vec<u32> {
    let edge = |a: [f64; 2], b: [f64; 2], x: f64, y: f64| {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        ((x - a[0]) * dy - (y - a[1]) * dx) / (dx * dx + dy * dy).sqrt()
    };
    let t = MS_TRIANGLE;
    let (a, b, c) = ([t[0], t[1]], [t[2], t[3]], [t[4], t[5]]);
    let margin = 1.0 / 16.0;
    let mut out = Vec::new();
    for y in 0..BMH {
        for x in 0..BMW {
            for p in POSITIONS_4 {
                let (px, py) = (x as f64 + p[0], y as f64 + p[1]);
                let e = [edge(a, b, px, py), edge(b, c, px, py), edge(c, a, px, py)];
                let strong_out = e.iter().any(|v| *v < -margin) && e.iter().any(|v| *v > margin);
                let near = e.iter().any(|v| v.abs() < margin);
                let value = (e.iter().all(|v| *v > 0.0) || e.iter().all(|v| *v < 0.0)) as u32;
                out.push(if near && !strong_out { AMBIGUOUS } else { value });
            }
        }
    }
    out
}

fn mask(measured: &[u32], classes: &[u32], whole_pixel: bool) -> Vec<u32> {
    let mut m = measured.to_vec();
    for p in 0..(BMW * BMH) as usize {
        let any = (0..4).any(|s| classes[p * 4 + s] == AMBIGUOUS);
        for s in 0..4 {
            if if whole_pixel { any } else { classes[p * 4 + s] == AMBIGUOUS } {
                m[p * 4 + s] = AMBIGUOUS;
            }
        }
    }
    m
}

fn ms_pipeline(h: &Harness, fragment: &str, format: u64, samples: u64) -> Handle {
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some(fragment), format);
    d.raster_sample_count = samples;
    h.pipeline(d).unwrap()
}

fn draw_ms(h: &Harness, pipeline: Handle, sample_mask: u32) -> Vec<u32> {
    let vs: Vec<RVertex> = (0..3)
        .map(|k| {
            let mut v = RVertex::at(MS_TRIANGLE[2 * k], MS_TRIANGLE[2 * k + 1], 0.0, BMW, BMH, 1);
            v.data[1] = sample_mask;
            v
        })
        .collect();
    let t = h.target(MS_COLOR, BMW, BMH, 4);
    rdraw(h, pass_color(t, clear(0.0, 0.0, 0.0, 0.0)), pipeline, prim::TRIANGLE, &vs);
    words(&h.read_samples(t, "rnd_read_ms_uint", 4, BMW, BMH, 4))
}

#[test]
fn msaa_coverage_4x() {
    let h = Harness::new();
    if !multisamples(&h, "msaa.coverage-4x", MS_COLOR) {
        return;
    }
    let classes = classify();
    let mut c = Candidate::default();
    let raw: Vec<u8> =
        POSITIONS_4.iter().flat_map(|p| [(p[0] as f32).to_le_bytes(), (p[1] as f32).to_le_bytes()]).flatten().collect();
    c.data("default-sample-positions", "f32", &raw);
    c.data("classification", "u32", &from_words(&classes));
    let coverage = draw_ms(&h, ms_pipeline(&h, "rnd_fragment_id", MS_COLOR, 4), 0);
    c.data("coverage", "u32", &from_words(&coverage));
    let masked = mask(&coverage, &classes, false);
    c.data("coverage.unambiguous", "u32", &from_words(&masked));
    c.bool("coverage.unambiguous-equals-sample-containment", masked == mask(&classes, &classes, false));
    let mask_in = draw_ms(&h, ms_pipeline(&h, "rnd_fragment_mask_in", MS_COLOR, 4), 0);
    c.data("sample-mask-input", "u32", &from_words(&mask_in));
    c.data("sample-mask-input.unambiguous-pixels", "u32", &from_words(&mask(&mask_in, &classes, true)));
    let sample_id = draw_ms(&h, ms_pipeline(&h, "rnd_fragment_sample_id", MS_COLOR, 4), 0);
    c.data("sample-id.unambiguous", "u32", &from_words(&mask(&sample_id, &classes, false)));
    let mask_out = draw_ms(&h, ms_pipeline(&h, "rnd_fragment_mask_out", MS_COLOR, 4), 0x5);
    c.data("sample-mask-output-0x5.unambiguous", "u32", &from_words(&mask(&mask_out, &classes, false)));
    check("msaa.coverage-4x", &c);
}

#[test]
fn msaa_color_resolve() {
    let h = Harness::new();
    if !multisamples(&h, "msaa.color-resolve", pf::RGBA16_FLOAT) {
        return;
    }
    let classes = classify();
    let color = [1.0f32, 0.5, 0.25, 1.0].map(f32::to_bits);
    let vs: Vec<RVertex> = (0..3)
        .map(|k| {
            let mut v = RVertex::at(MS_TRIANGLE[2 * k], MS_TRIANGLE[2 * k + 1], 0.0, BMW, BMH, 0);
            v.data = color;
            v
        })
        .collect();
    let b = vertex_buffer(&h, &vs);
    let p = ms_pipeline(&h, "rnd_fragment_float", pf::RGBA16_FLOAT, 4);
    let target = h.target(pf::RGBA16_FLOAT, BMW, BMH, 4);
    let resolved = h.target(pf::RGBA16_FLOAT, BMW, BMH, 1);
    let mut pass = pass_color(target, clear(0.0, 0.0, 0.0, 0.0));
    pass.color_attachments[0].store_action = 3;
    pass.color_attachments[0].resolve_texture = Some(resolved);
    let e = h.handle();
    h.run_ok(&render(e, pass, vec![set_pipeline(p), set_vertex_buffer(b, 0, 0), draw(prim::TRIANGLE, 0, 3)]));
    let samples = words(&h.read_samples(target, "rnd_read_ms_float", 16, BMW, BMH, 4));
    let mut masked = samples.clone();
    for (i, class) in classes.iter().enumerate() {
        if *class == AMBIGUOUS {
            masked[i * 4..i * 4 + 4].fill(AMBIGUOUS);
        }
    }
    let mut c = Candidate::default();
    c.data("samples.unambiguous", "u32", &from_words(&masked));
    record_texture(&h, &mut c, "resolved", resolved, "f16", "RGBA16Float", BMW, BMH, 8);
    check("msaa.color-resolve", &c);
}

fn depth_resolve(h: &Harness, stencil_too: bool, filters: &[(u64, u64, &str)], c: &mut Candidate) {
    let b = vertex_buffer(
        h,
        &[
            RVertex::at(-1.0, -1.0, 0.5, 4, 4, 1),
            RVertex::at(12.0, -1.0, 0.5, 4, 4, 1),
            RVertex::at(-1.0, 12.0, 0.5, 4, 4, 1),
        ],
    );
    let format = if stencil_too { pf::DEPTH32_FLOAT_STENCIL8 } else { pf::DEPTH32_FLOAT };
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some("rnd_fragment_sample_depth"), MS_COLOR);
    d.raster_sample_count = 4;
    d.depth_attachment_pixel_format = format;
    if stencil_too {
        d.stencil_attachment_pixel_format = format;
    }
    let p = h.pipeline(d).unwrap();
    let mut ds = depth_stencil_descriptor(7, true);
    if stencil_too {
        let s = stencil(7, 0, 0, 2, 0xffff_ffff, 0xffff_ffff);
        ds.front_face_stencil = Some(s.clone());
        ds.back_face_stencil = Some(s);
    }
    let state = h.depth_stencil(ds);
    for (f, &(depth_filter, stencil_filter, name)) in filters.iter().enumerate() {
        let color = h.target(MS_COLOR, 4, 4, 4);
        let dst = h.target(format, 4, 4, 4);
        let resolved = h.target(format, 4, 4, 1);
        let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
        pass.color_attachments[0].store_action = 0;
        let mut da = depth_attachment(dst);
        da.load_action = 2;
        da.clear_depth = 1.0;
        da.store_action = 3;
        da.resolve_texture = Some(resolved);
        da.depth_resolve_filter = depth_filter;
        pass.depth_attachment = Some(da);
        if stencil_too {
            let mut sa = stencil_attachment(dst);
            sa.load_action = 2;
            sa.clear_stencil = 3;
            sa.store_action = 3;
            sa.resolve_texture = Some(resolved);
            sa.stencil_resolve_filter = stencil_filter;
            pass.stencil_attachment = Some(sa);
        }
        let e = h.handle();
        h.run_ok(&render(
            e,
            pass,
            vec![
                set_pipeline(p),
                set_depth_stencil(state),
                Command::RenderSetStencilReferenceValue(RenderSetStencilReferenceValue { reference_value: 9 }),
                set_vertex_buffer(b, 0, 0),
                draw(prim::TRIANGLE, 0, 3),
            ],
        ));
        if f == 0 {
            c.data("depth-samples", "f32", &h.read_samples(dst, "rnd_read_ms_depth", 4, 4, 4, 4));
        }
        let depth = h.read_texture_options(resolved, 0, 4, 4, 16, if stencil_too { 1 } else { 0 });
        c.data(&format!("{name}.resolved-depth"), "f32", &depth);
        if stencil_too {
            c.data(&format!("{name}.resolved-stencil"), "u8", &h.read_texture_options(resolved, 0, 4, 4, 4, 2));
        }
    }
}

#[test]
fn msaa_depth_resolve() {
    let h = Harness::new();
    if !multisamples(&h, "msaa.depth-resolve", MS_COLOR) {
        return;
    }
    let mut c = Candidate::default();
    depth_resolve(&h, false, &[(0, 0, "sample0"), (1, 0, "min"), (2, 0, "max")], &mut c);
    check("msaa.depth-resolve", &c);
}

/// `msaa.depth-stencil-resolve` requires family Apple5, so on the A10X profile it is not comparable; the stencil
/// filter it uses for `min` and `max` (`depthResolvedSample`) is not implemented (the A10X reports no stencil resolve
/// filters). The `sample0` resolve, which the A10X supports (`supportsMSAAStencilResolve`), is checked against the
/// reference's records.
#[test]
fn msaa_depth_stencil_resolve_sample0() {
    let h = Harness::new();
    if !multisamples(&h, "msaa.depth-stencil-resolve", MS_COLOR) {
        return;
    }
    let mut c = Candidate::default();
    depth_resolve(&h, true, &[(0, 0, "sample0")], &mut c);
    let g = Golden::load("msaa.depth-stencil-resolve");
    for (name, m) in &c.records {
        if let render_common::golden::Measured::Data { bytes, .. } = m {
            assert_eq!(bytes, &g.data(name), "{name}");
        }
    }
    println!("msaa.depth-stencil-resolve (sample0 filter): {} records equal the reference", c.records.len());
}

#[test]
fn msaa_alpha_to_coverage() {
    let h = Harness::new();
    if !multisamples(&h, "msaa.alpha-to-coverage", pf::RGBA16_FLOAT) {
        return;
    }
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some("rnd_fragment_alpha"), pf::RGBA16_FLOAT);
    d.raster_sample_count = 4;
    d.alpha_to_coverage_enabled = true;
    let p = h.pipeline(d).unwrap();
    let t = h.target(pf::RGBA16_FLOAT, 10, 2, 4);
    let mut body = vec![set_pipeline(p)];
    for k in 0..5u64 {
        let alpha = (k as f32 * 0.25).to_bits();
        let vs: Vec<RVertex> = [(-1.0, -1.0), (40.0, -1.0), (-1.0, 40.0)]
            .iter()
            .map(|(x, y)| {
                let mut v = RVertex::at(*x, *y, 0.0, 10, 2, 0);
                v.data[0] = alpha;
                v
            })
            .collect();
        let b = vertex_buffer(&h, &vs);
        body.extend([scissor(k * 2, 0, 2, 2), set_vertex_buffer(b, 0, 0), draw(prim::TRIANGLE, 0, 3)]);
    }
    let e = h.handle();
    h.run_ok(&render(e, pass_color(t, clear(0.0, 0.0, 0.0, 0.0)), body));
    let mut c = Candidate::default();
    c.data("samples", "f32", &h.read_samples(t, "rnd_read_ms_float", 16, 10, 2, 4));
    check("msaa.alpha-to-coverage", &c);
}

/// The multisampled `R32Uint` attachments of the multisample cases: refused by the format module at texture creation,
/// as WebGPU has no multisampled r32uint.
#[test]
fn msaa_r32uint_attachment_refused() {
    let h = Harness::new();
    let mut d = texture_descriptor(pf::R32_UINT, 8, 8, STORAGE_PRIVATE, usage::RENDER_TARGET | usage::SHADER_READ);
    d.texture_type = 4;
    d.sample_count = 4;
    let f = h.texture(d).unwrap_err();
    refused("msaa (R32Uint, 4 samples)", &f, status::HOST_UNSUPPORTED, "multisampling");
}

/// `msaa.coverage-2x`: refused (`tables/refusals.md` row `msaa-2x`) at texture creation by the format module.
#[test]
fn msaa_coverage_2x() {
    let h = Harness::new();
    let mut d = texture_descriptor(pf::R32_UINT, 8, 8, STORAGE_PRIVATE, usage::RENDER_TARGET | usage::SHADER_READ);
    d.texture_type = 4;
    d.sample_count = 2;
    let f = h.texture(d).unwrap_err();
    refused("msaa.coverage-2x", &f, status::HOST_UNSUPPORTED, "msaa-2x");
}

/// `msaa.error.sample-count-mismatch`: a misuse whose result Metal does not define.
#[test]
fn msaa_error_sample_count_mismatch() {
    let h = Harness::new();
    if !multisamples(&h, "msaa.error.sample-count-mismatch", MS_COLOR) {
        return;
    }
    let p = ms_pipeline(&h, "rnd_fragment_id", MS_COLOR, 1);
    let b = vertex_buffer(
        &h,
        &[
            RVertex::at(-1.0, -1.0, 0.0, 4, 4, 1),
            RVertex::at(12.0, -1.0, 0.0, 4, 4, 1),
            RVertex::at(-1.0, 12.0, 0.0, 4, 4, 1),
        ],
    );
    let t = h.target(MS_COLOR, 4, 4, 4);
    let e = h.handle();
    let f = h.run(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![set_pipeline(p), set_vertex_buffer(b, 0, 0), draw(prim::TRIANGLE, 0, 3)],
    ));
    refused("msaa.error.sample-count-mismatch", &f, status::UNIMPLEMENTED, "rasterSampleCount");
}

// RenderPass.m --------------------------------------------------------------------------------------------------------

fn cover(h: &Harness, w: u64, hh: u64, z: f64, id: u32) -> Handle {
    vertex_buffer(
        h,
        &[
            RVertex::at(-1.0, -1.0, z, w, hh, id),
            RVertex::at(3.0 * w as f64, -1.0, z, w, hh, id),
            RVertex::at(-1.0, 3.0 * hh as f64, z, w, hh, id),
        ],
    )
}

#[test]
fn pass_load_store_actions() {
    let h = Harness::new();
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    let b = cover(&h, 8, 8, 0.0, 2);
    let t = h.target(pf::R32_UINT, 8, 8, 1);
    let e = h.handle();
    h.run_ok(&render(e, pass_color(t, clear(5.0, 0.0, 0.0, 0.0)), vec![]));
    let mut load = pass_color(t, clear(0.0, 0.0, 0.0, 0.0));
    load.color_attachments[0].load_action = 1;
    let e = h.handle();
    h.run_ok(&render(
        e,
        load,
        vec![set_pipeline(p), scissor(2, 3, 3, 2), set_vertex_buffer(b, 0, 0), draw(prim::TRIANGLE, 0, 3)],
    ));
    let mut c = Candidate::default();
    record_texture(&h, &mut c, "load", t, "u32", "R32Uint", 8, 8, 4);
    let mut dont = pass_color(t, clear(0.0, 0.0, 0.0, 0.0));
    dont.color_attachments[0].load_action = 0;
    let e = h.handle();
    h.run_ok(&render(
        e,
        dont,
        vec![set_pipeline(p), scissor(0, 0, 4, 4), set_vertex_buffer(b, 0, 0), draw(prim::TRIANGLE, 0, 3)],
    ));
    let image = words(&h.read_texture(t, 8, 8, 32));
    let drawn: Vec<u32> = (0..4).flat_map(|y| (0..4).map(move |x| (y, x))).map(|(y, x)| image[y * 8 + x]).collect();
    c.data("dont-care-load.drawn-rectangle", "u32", &from_words(&drawn));
    c.data("dont-care-load.all", "u32", &from_words(&image));
    let mut discard = pass_color(t, clear(7.0, 0.0, 0.0, 0.0));
    discard.color_attachments[0].store_action = 0;
    let e = h.handle();
    h.run_ok(&render(e, discard, vec![]));
    record_texture(&h, &mut c, "dont-care-store", t, "u32", "R32Uint", 8, 8, 4);
    check("pass.load-store-actions", &c);
}

#[test]
fn pass_clear_values() {
    let h = Harness::new();
    let mut c = Candidate::default();
    for (format, cl, bpp, elem, name, metal) in [
        (pf::RGBA8_UNORM, clear(0.0, 1.0, 1.0, 0.0), 4u64, "u8", "rgba8unorm", "RGBA8Unorm"),
        (pf::RGBA16_FLOAT, clear(0.375, -2.5, 65504.0, 0.0009765625), 8, "f16", "rgba16float", "RGBA16Float"),
        (pf::RGBA32_FLOAT, clear(0.375, -2.5, 1048576.25, 1.0), 16, "f32", "rgba32float", "RGBA32Float"),
        (pf::RGBA32_UINT, clear(0.0, 1.0, 4294967295.0, 123456789.0), 16, "u32", "rgba32uint", "RGBA32Uint"),
        (pf::RGBA16_SINT, clear(-32768.0, 32767.0, -1.0, 0.0), 8, "i16", "rgba16sint", "RGBA16Sint"),
    ] {
        let t = h.target(format, 4, 4, 1);
        let e = h.handle();
        h.run_ok(&render(e, pass_color(t, cl), vec![]));
        record_texture(&h, &mut c, name, t, elem, metal, 4, 4, bpp);
    }
    let ds = h.target(pf::DEPTH32_FLOAT_STENCIL8, 4, 4, 1);
    let mut pass = pass_descriptor();
    let mut da = depth_attachment(ds);
    da.load_action = 2;
    da.store_action = 1;
    da.clear_depth = 0.375;
    pass.depth_attachment = Some(da);
    let mut sa = stencil_attachment(ds);
    sa.load_action = 2;
    sa.store_action = 1;
    sa.clear_stencil = 0xa5;
    pass.stencil_attachment = Some(sa);
    let e = h.handle();
    h.run_ok(&render(e, pass, vec![]));
    c.data("depth", "f32", &h.read_texture_options(ds, 0, 4, 4, 16, 1));
    c.data("stencil", "u8", &h.read_texture_options(ds, 0, 4, 4, 4, 2));
    check("pass.clear-values", &c);
}

#[test]
fn pass_store_action_deferred() {
    let h = Harness::new();
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    let b = cover(&h, 4, 4, 0.0, 3);
    let t = h.target(pf::R32_UINT, 4, 4, 1);
    let mut pass = pass_color(t, clear(0.0, 0.0, 0.0, 0.0));
    pass.color_attachments[0].store_action = 4;
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass,
        vec![
            set_pipeline(p),
            set_vertex_buffer(b, 0, 0),
            draw(prim::TRIANGLE, 0, 3),
            Command::RenderSetColorStoreAction(RenderSetColorStoreAction {
                store_action: 1,
                color_attachment_index: 0,
            }),
        ],
    ));
    let mut c = Candidate::default();
    record_texture(&h, &mut c, "result", t, "u32", "R32Uint", 4, 4, 4);
    check("pass.store-action-deferred", &c);
}

/// `pass.error.store-action-unknown-unset`: a misuse whose result Metal does not define.
#[test]
fn pass_error_store_action_unknown_unset() {
    let h = Harness::new();
    let t = h.target(pf::R32_UINT, 4, 4, 1);
    let mut pass = pass_color(t, clear(6.0, 0.0, 0.0, 0.0));
    pass.color_attachments[0].store_action = 4;
    let e = h.handle();
    let f = h.run(&render(e, pass, vec![]));
    refused("pass.error.store-action-unknown-unset", &f, status::UNIMPLEMENTED, "MTLStoreActionUnknown");
}

#[test]
fn pass_parallel_encoder_order() {
    let h = Harness::new();
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    let (first, second) = (cover(&h, 8, 8, 0.0, 1), cover(&h, 8, 8, 0.0, 2));
    let t = h.target(pf::R32_UINT, 8, 8, 1);
    let (parallel, a, b) = (h.handle(), h.handle(), h.handle());
    // The guest splices sub-encoders in creation order: a (created first, encoded second), then b.
    h.run_ok(&[
        Command::BeginParallelRenderPass(BeginParallelRenderPass {
            encoder: parallel,
            render_pass_descriptor: pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        }),
        Command::BeginParallelSubEncoder(BeginParallelSubEncoder { parent: parallel, encoder: a }),
        set_pipeline(p),
        set_vertex_buffer(first, 0, 0),
        draw(prim::TRIANGLE, 0, 3),
        end(),
        Command::BeginParallelSubEncoder(BeginParallelSubEncoder { parent: parallel, encoder: b }),
        set_pipeline(p),
        scissor(0, 0, 4, 8),
        set_vertex_buffer(second, 0, 0),
        draw(prim::TRIANGLE, 0, 3),
        end(),
        end(),
    ]);
    // The sub-encoders are render encoders the guest holds: their handles are released like the parallel encoder's.
    for encoder in [parallel, a, b] {
        h.ok(Request::Release(Release { object: encoder }));
    }
    let mut c = Candidate::default();
    record_texture(&h, &mut c, "result", t, "u32", "R32Uint", 8, 8, 4);
    check("pass.parallel-encoder-order", &c);
}

fn memoryless_draw(h: &Harness, depth_store: u64) -> (Failure, Handle) {
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some("rnd_fragment_id"), pf::R32_UINT);
    d.depth_attachment_pixel_format = pf::DEPTH32_FLOAT;
    let p = h.pipeline(d).unwrap();
    let state = h.depth_stencil(depth_stencil_descriptor(1, true));
    let depth =
        h.texture(texture_descriptor(pf::DEPTH32_FLOAT, 8, 8, STORAGE_MEMORYLESS, usage::RENDER_TARGET)).unwrap();
    let color = h.target(pf::R32_UINT, 8, 8, 1);
    let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
    let mut da = depth_attachment(depth);
    da.load_action = 2;
    da.clear_depth = 1.0;
    da.store_action = depth_store;
    pass.depth_attachment = Some(da);
    let (near, far) = (cover(h, 8, 8, 0.25, 1), cover(h, 8, 8, 0.5, 2));
    let e = h.handle();
    let f = h.run(&render(
        e,
        pass,
        vec![
            set_pipeline(p),
            set_depth_stencil(state),
            scissor(0, 0, 5, 8),
            set_vertex_buffer(near, 0, 0),
            draw(prim::TRIANGLE, 0, 3),
            scissor(0, 0, 8, 8),
            set_vertex_buffer(far, 0, 0),
            draw(prim::TRIANGLE, 0, 3),
        ],
    ));
    (f, color)
}

#[test]
fn pass_memoryless_depth() {
    let h = Harness::new();
    let (f, color) = memoryless_draw(&h, 0);
    assert_eq!(f.status, status::OK, "{f:?}");
    let mut c = Candidate::default();
    record_texture(&h, &mut c, "color", color, "u32", "R32Uint", 8, 8, 4);
    check("pass.memoryless-depth", &c);
}

/// `pass.error.memoryless-store`: "Memoryless attachment content cannot be stored in memory" (debug layer).
#[test]
fn pass_error_memoryless_store() {
    let h = Harness::new();
    let (f, _) = memoryless_draw(&h, 1);
    refused("pass.error.memoryless-store", &f, status::UNIMPLEMENTED, "memoryless");
}

/// `pass.visibility-results`: the boolean results are measured; the counting mode that follows in the same encoder is
/// refused (`tables/refusals.md` row `counting-occlusion`).
#[test]
fn pass_visibility_results() {
    let h = Harness::new();
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some("rnd_fragment_id"), pf::R32_UINT);
    d.depth_attachment_pixel_format = pf::DEPTH32_FLOAT;
    let p = h.pipeline(d).unwrap();
    let passing = h.depth_stencil(depth_stencil_descriptor(7, false));
    let failing = h.depth_stencil(depth_stencil_descriptor(0, false));
    let b = cover(&h, 8, 8, 0.5, 1);
    let make_pass = |results: Handle| {
        let color = h.target(pf::R32_UINT, 8, 8, 1);
        let depth = h.target(pf::DEPTH32_FLOAT, 8, 8, 1);
        let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
        let mut da = depth_attachment(depth);
        da.load_action = 2;
        da.store_action = 0;
        pass.depth_attachment = Some(da);
        pass.visibility_result_buffer = Some(results);
        pass
    };
    let mode =
        |m: u64, offset: u64| Command::RenderSetVisibilityResultMode(RenderSetVisibilityResultMode { mode: m, offset });
    // The boolean spans of the case, alone.
    let (results, addr) = h.zeros(48);
    let e = h.handle();
    h.run_ok(&render(
        e,
        make_pass(results),
        vec![
            set_pipeline(p),
            set_vertex_buffer(b, 0, 0),
            mode(1, 0),
            set_depth_stencil(passing),
            scissor(0, 0, 4, 3),
            draw(prim::TRIANGLE, 0, 3),
            mode(1, 8),
            set_depth_stencil(failing),
            scissor(0, 0, 8, 8),
            draw(prim::TRIANGLE, 0, 3),
            mode(0, 0),
        ],
    ));
    let r = words(&h.read(addr, 16));
    let passing_value = u64::from(r[0]) | u64::from(r[1]) << 32;
    let failing_value = u64::from(r[2]) | u64::from(r[3]) << 32;
    let g = Golden::load("pass.visibility-results");
    let want = |name: &str| g.record(name).and_then(|r| r.get("value")).cloned().unwrap();
    assert_eq!(want("boolean.passing.nonzero").bool(), Some(passing_value != 0));
    assert_eq!(want("boolean.failing").int(), Some(failing_value as i128));
    println!(
        "pass.visibility-results: boolean.passing.nonzero and boolean.failing equal the reference ({passing_value}, {failing_value})"
    );
    // The counting mode.
    let (results, _) = h.zeros(48);
    let e = h.handle();
    let f = h.run(&render(
        e,
        make_pass(results),
        vec![
            set_pipeline(p),
            set_vertex_buffer(b, 0, 0),
            mode(2, 16),
            set_depth_stencil(passing),
            draw(prim::TRIANGLE, 0, 3),
        ],
    ));
    refused("pass.visibility-results", &f, status::HOST_UNSUPPORTED, "counting-occlusion");
}

/// `pass.error.visibility-offset-unaligned` and `pass.error.visibility-offset-reused` (with boolean mode, since
/// counting is refused first): misuses whose results Metal does not define.
#[test]
fn pass_error_visibility_offsets() {
    let h = Harness::new();
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    let b = cover(&h, 4, 4, 0.0, 1);
    let mode =
        |m: u64, offset: u64| Command::RenderSetVisibilityResultMode(RenderSetVisibilityResultMode { mode: m, offset });
    for (body_modes, needle, case) in [
        (vec![mode(1, 4)], "multiple of 8", "pass.error.visibility-offset-unaligned"),
        (vec![mode(1, 0), mode(1, 8), mode(1, 0)], "used twice", "pass.error.visibility-offset-reused"),
    ] {
        let (results, _) = h.zeros(32);
        let t = h.target(pf::R32_UINT, 4, 4, 1);
        let mut pass = pass_color(t, clear(0.0, 0.0, 0.0, 0.0));
        pass.visibility_result_buffer = Some(results);
        let mut body = vec![set_pipeline(p), set_vertex_buffer(b, 0, 0)];
        for m in body_modes {
            body.push(m);
            body.push(draw(prim::TRIANGLE, 0, 3));
        }
        let e = h.handle();
        let f = h.run(&render(e, pass, body));
        refused(case, &f, status::UNIMPLEMENTED, needle);
    }
}

/// `storage.memoryless-depth` (Storage.m): a memoryless depth attachment cleared, tested and discarded in one pass.
#[test]
fn storage_memoryless_depth() {
    let h = Harness::new();
    let rgba8_uint = 73;
    let color = h.texture(texture_descriptor(rgba8_uint, 8, 8, STORAGE_PRIVATE, usage::RENDER_TARGET)).unwrap();
    let depth =
        h.texture(texture_descriptor(pf::DEPTH32_FLOAT, 8, 8, STORAGE_MEMORYLESS, usage::RENDER_TARGET)).unwrap();
    let mut c = Candidate::default();
    // The texture's storageMode, a property of the guest's object.
    c.u64("depth.storage-mode", STORAGE_MEMORYLESS);
    let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
    let mut da = depth_attachment(depth);
    da.load_action = 2;
    da.clear_depth = 0.5;
    da.store_action = 0;
    pass.depth_attachment = Some(da);
    let mut d = h.pipeline_of("sto_fullscreen_vertex", Some("sto_color_fragment"), rgba8_uint);
    d.depth_attachment_pixel_format = pf::DEPTH32_FLOAT;
    let p = h.pipeline(d).unwrap();
    let state = h.depth_stencil(depth_stencil_descriptor(1, true));
    let mut body = vec![set_pipeline(p), set_depth_stencil(state)];
    for (i, depth) in [0.25f32, 0.75, 0.125].iter().enumerate() {
        let i = i as u32;
        body.push(Command::RenderSetVertexBytes(RenderSetVertexBytes {
            bytes: depth.to_le_bytes().to_vec(),
            length: 4,
            index: 0,
        }));
        body.push(Command::RenderSetFragmentBytes(RenderSetFragmentBytes {
            bytes: from_words(&[i + 1, 10 * (i + 1), 100 * (i + 1), 255]),
            length: 16,
            index: 0,
        }));
        body.push(draw(prim::TRIANGLE, 0, 3));
    }
    let e = h.handle();
    h.run_ok(&render(e, pass, body));
    record_texture(&h, &mut c, "color", color, "u8", "RGBA8Uint", 8, 8, 4);
    check("storage.memoryless-depth", &c);
}

// Render path behavior without a reference case -----------------------------------------------------------------------

/// Primitive restart in indexed list topologies, which no reference case covers: Metal "complete[s] a primitive and
/// start[s] a new one" at the largest index value in every indexed draw; WebGPU splits only strips. The staged
/// indices follow `render::index::assemble`.
#[test]
fn index_staging_restarts_lists() {
    use webmetal_host::render::index::assemble;
    let h = Harness::new();
    // Points: the sentinel is skipped, so no point lands at vertex id 0xffff (x = 15).
    let p = ids_pipeline(&h);
    let i16s: Vec<u8> = [0u16, 0xffff, 2, 0xffff, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
    let (ib, _) = h.buffer(&i16s);
    let t = h.target(pf::RGBA32_UINT, 16, 16, 1);
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![set_pipeline(p), indexed(prim::POINT, 5, 0, ib, 0, 1, 0, 0)],
    ));
    let w = words(&h.read_texture(t, 16, 16, 256));
    let written: Vec<usize> = (0..16).filter(|x| w[x * 4] != 0).collect();
    assert_eq!(written, vec![0, 2, 4], "points drawn at vertex ids");
    // Triangles: (0,1,2), 3 dropped at the restart, (4,5,6), trailing 7 dropped; indirect arguments too.
    let corners =
        [[1.0, 1.0], [7.25, 1.0], [1.0, 7.25], [7.0, 7.0], [9.0, 9.0], [15.25, 9.0], [9.0, 15.25], [15.0, 15.0]];
    let vs: Vec<RVertex> = corners.iter().map(|c| RVertex::at(c[0], c[1], 0.0, 16, 16, 1)).collect();
    let vb = vertex_buffer(&h, &vs);
    let list = [0u32, 1, 2, 3, 0xffff_ffff, 4, 5, 6, 7];
    let kept = assemble(&list, 3, 0xffff_ffff);
    assert_eq!(kept, vec![0, 1, 2, 4, 5, 6]);
    let tris: Vec<[f64; 6]> = kept
        .chunks(3)
        .map(|t| {
            let p = |i: u32| corners[i as usize];
            [p(t[0])[0], p(t[0])[1], p(t[1])[0], p(t[1])[1], p(t[2])[0], p(t[2])[1]]
        })
        .collect();
    let expected = paint_triangles(&tris, &[1, 1], 16, 16, MARGIN).expect("unambiguous");
    let (lb, _) = h.buffer(&from_words(&list));
    let (args, _) = h.buffer(&from_words(&[9, 1, 0, 0, 0]));
    let tri = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_id", pf::R32_UINT);
    for indirect in [false, true] {
        let t = h.target(pf::R32_UINT, 16, 16, 1);
        let draw = if indirect {
            Command::RenderDrawIndexedPrimitivesIndirect(RenderDrawIndexedPrimitivesIndirect {
                primitive_type: prim::TRIANGLE,
                index_type: 1,
                index_buffer: Some(lb),
                index_buffer_offset: 0,
                indirect_buffer: Some(args),
                indirect_buffer_offset: 0,
            })
        } else {
            indexed(prim::TRIANGLE, 9, 1, lb, 0, 1, 0, 0)
        };
        let e = h.handle();
        let f = h.run(&render(
            e,
            pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
            vec![set_pipeline(tri), set_vertex_buffer(vb, 0, 0), draw],
        ));
        if indirect && !h.has("indirect-first-instance") {
            refused("index staging (indirect)", &f, status::HOST_UNSUPPORTED, "indirect-first-instance");
            continue;
        }
        assert_eq!(f.status, status::OK, "{f:?}");
        assert_eq!(words(&h.read_texture(t, 16, 16, 64)), expected, "indirect: {indirect}");
    }
    println!("index staging: point and triangle lists restart as Metal assembles them");
}

/// A barrier after the vertex stage and before the fragment stage splits the WebGPU pass. The result equals the
/// unsplit draws', a visibility span that crosses the split gets the OR of both passes' queries, and debug groups
/// stay balanced.
#[test]
fn barrier_split_keeps_state() {
    let h = Harness::new();
    let mut d = h.pipeline_of("rnd_vertex_triangles", Some("rnd_fragment_id"), pf::R32_UINT);
    d.depth_attachment_pixel_format = pf::DEPTH32_FLOAT;
    let p = h.pipeline(d).unwrap();
    let passing = h.depth_stencil(depth_stencil_descriptor(7, false));
    let failing = h.depth_stencil(depth_stencil_descriptor(0, false));
    let (a, b) = (cover(&h, 8, 8, 0.5, 1), cover(&h, 8, 8, 0.5, 2));
    let mode =
        |m: u64, offset: u64| Command::RenderSetVisibilityResultMode(RenderSetVisibilityResultMode { mode: m, offset });
    let run = |barrier: bool, first_passes: bool| {
        let (results, addr) = h.zeros(16);
        let color = h.target(pf::R32_UINT, 8, 8, 1);
        let depth = h.target(pf::DEPTH32_FLOAT, 8, 8, 1);
        let mut pass = pass_color(color, clear(0.0, 0.0, 0.0, 0.0));
        let mut da = depth_attachment(depth);
        da.load_action = 2;
        da.store_action = 0;
        pass.depth_attachment = Some(da);
        pass.visibility_result_buffer = Some(results);
        let mut body = vec![
            Command::EncoderPushDebugGroup(EncoderPushDebugGroup { string: Some("outer".into()) }),
            set_pipeline(p),
            Command::RenderSetViewport(RenderSetViewport {
                viewport: Viewport { origin_x: 0.0, origin_y: 0.0, width: 8.0, height: 8.0, znear: 0.0, zfar: 1.0 },
            }),
            scissor(1, 1, 5, 5),
            mode(1, 8),
            set_depth_stencil(if first_passes { passing } else { failing }),
            set_vertex_buffer(a, 0, 0),
            draw(prim::TRIANGLE, 0, 3),
        ];
        if barrier {
            body.push(Command::RenderMemoryBarrierWithScope(RenderMemoryBarrierWithScope {
                scope: 1,
                after: 1,
                before: 2,
            }));
        }
        body.extend([
            set_depth_stencil(failing),
            scissor(2, 2, 2, 2),
            set_vertex_buffer(b, 0, 0),
            draw(prim::TRIANGLE, 0, 3),
            mode(0, 0),
        ]);
        let e = h.handle();
        h.run_ok(&render(e, pass, body));
        (h.read_texture(color, 8, 8, 32), words(&h.read(addr, 16)))
    };
    for first_passes in [true, false] {
        let (plain, plain_results) = run(false, first_passes);
        let (split, split_results) = run(true, first_passes);
        assert_eq!(plain, split);
        assert_eq!(plain_results[..2], [0, 0], "offset 0 is not written");
        assert_eq!(split_results[..2], [0, 0], "offset 0 is not written");
        assert_eq!(plain_results[2] != 0, first_passes);
        assert_eq!(split_results[2] != 0, first_passes, "the span's value is the OR of both passes' queries");
    }
    println!("barrier split: results equal the unsplit pass; visibility spans merge across the split");
}

/// Records the render path does not implement complete the command buffer with status 2 and name the Metal method.
#[test]
fn unimplemented_records_are_reported() {
    let h = Harness::new();
    let t = h.target(pf::R32_UINT, 4, 4, 1);
    let e = h.handle();
    let f = h.run(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![Command::RenderExecuteCommandsInBuffer(RenderExecuteCommandsInBuffer {
            indirect_command_buffer: None,
            execution_range: Range { location: 0, length: 1 },
        })],
    ));
    refused("executeCommandsInBuffer", &f, status::UNIMPLEMENTED, "executeCommandsInBuffer");
}

// RenderFragment.m ----------------------------------------------------------------------------------------------------

#[test]
fn render_discard() {
    let h = Harness::new();
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_discard", pf::R32_UINT);
    let b = cover(&h, 8, 8, 0.0, 4);
    let t = h.target(pf::R32_UINT, 8, 8, 1);
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass_color(t, clear(9.0, 0.0, 0.0, 0.0)),
        vec![set_pipeline(p), set_vertex_buffer(b, 0, 0), draw(prim::TRIANGLE, 0, 3)],
    ));
    let mut c = Candidate::default();
    record_texture(&h, &mut c, "result", t, "u32", "R32Uint", 8, 8, 4);
    check("render.discard", &c);
}

#[test]
fn render_interpolation() {
    let h = Harness::new();
    let p = h.simple_pipeline("rnd_vertex_interpolated", "rnd_fragment_interpolated", pf::RGBA32_UINT);
    let mut v = [
        RVertex::at(-1.0, -1.0, 0.0, 8, 8, 1),
        RVertex::at(24.0, -1.0, 0.0, 8, 8, 2),
        RVertex::at(-1.0, 24.0, 0.0, 8, 8, 3),
    ];
    let values = [0.0f32, 1.0, 0.5];
    let ws = [1.0f32, 2.0, 4.0];
    for k in 0..3 {
        v[k].extra[1] = values[k];
        for i in 0..4 {
            v[k].position[i] *= ws[k];
        }
    }
    let mut c = Candidate::default();
    let t = h.target(pf::RGBA32_UINT, 8, 8, 1);
    rdraw(&h, pass_color(t, clear(0.0, 0.0, 0.0, 0.0)), p, prim::TRIANGLE, &v);
    record_texture(&h, &mut c, "distinct-vertex-values", t, "u32", "RGBA32Uint", 8, 8, 16);
    for x in &mut v {
        x.extra[1] = 0.75;
        x.data[0] = 6;
    }
    let t2 = h.target(pf::RGBA32_UINT, 8, 8, 1);
    rdraw(&h, pass_color(t2, clear(0.0, 0.0, 0.0, 0.0)), p, prim::TRIANGLE, &v);
    let uniform = h.read_texture(t2, 8, 8, 128);
    let w = words(&uniform);
    let tq = 0.75f32.to_bits();
    let flat: Vec<u32> = (0..64).map(|i| w[i * 4 + 3]).collect();
    let constant = (0..64).all(|i| w[i * 4] == tq && w[i * 4 + 1] == tq && w[i * 4 + 2] == tq);
    c.data("equal-vertex-values.flat", "u32", &from_words(&flat));
    c.data("equal-vertex-values.smooth", "u32", &uniform);
    c.bool("equal-vertex-values.smooth-all-0.75", constant);
    check("render.interpolation", &c);
}

/// `render.framebuffer-fetch`: the per-primitive pass plan of plan section 5.9 (the fetching shader is the compiler's;
/// here a translation that reads the snapshot binding the plan defines).
#[test]
fn render_framebuffer_fetch() {
    let h = Harness::new();
    let p = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_fetch", pf::R32_UINT);
    let mut vs = Vec::new();
    for t in 0..4u32 {
        vs.extend([
            RVertex::at(-1.0, -1.0, 0.0, 8, 8, t + 1),
            RVertex::at(24.0, -1.0, 0.0, 8, 8, t + 1),
            RVertex::at(-1.0, 24.0, 0.0, 8, 8, t + 1),
        ]);
    }
    let b = vertex_buffer(&h, &vs);
    let t = h.target(pf::R32_UINT, 8, 8, 1);
    let e = h.handle();
    h.run_ok(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![set_pipeline(p), set_vertex_buffer(b, 0, 0), draw(prim::TRIANGLE, 0, 9), draw(prim::TRIANGLE, 9, 3)],
    ));
    let mut c = Candidate::default();
    record_texture(&h, &mut c, "result", t, "u32", "R32Uint", 8, 8, 4);
    check("render.framebuffer-fetch", &c);
}

/// `render.error.memory-barrier-after-fragment`: Apple-family GPUs reject a barrier after the fragment stage.
#[test]
fn render_error_memory_barrier_after_fragment() {
    let h = Harness::new();
    let store = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_store", pf::R32_UINT);
    let load = h.simple_pipeline("rnd_vertex_triangles", "rnd_fragment_load", pf::R32_UINT);
    let b = cover(&h, 8, 8, 0.0, 5);
    let (shared, _) = h.zeros(256);
    let t = h.target(pf::R32_UINT, 8, 8, 1);
    let e = h.handle();
    let f = h.run(&render(
        e,
        pass_color(t, clear(0.0, 0.0, 0.0, 0.0)),
        vec![
            set_vertex_buffer(b, 0, 0),
            Command::RenderSetFragmentBuffer(RenderSetFragmentBuffer { buffer: Some(shared), offset: 0, index: 0 }),
            set_pipeline(store),
            draw(prim::TRIANGLE, 0, 3),
            Command::RenderMemoryBarrierWithScope(RenderMemoryBarrierWithScope { scope: 1, after: 2, before: 2 }),
            set_pipeline(load),
            draw(prim::TRIANGLE, 0, 3),
        ],
    ));
    refused("render.error.memory-barrier-after-fragment", &f, status::UNIMPLEMENTED, "fragment or tile stage");
}

// Formats.m -----------------------------------------------------------------------------------------------------------

/// Every `format.<format>.render` case whose format the host stores natively and renders: the reference's recorded
/// input values are drawn through the case's pipeline and the stored texels compared.
#[test]
fn format_render() {
    let h = Harness::new();
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/reference/goldens")
        .join(mem_common::GOLDEN_DEVICE)
        .join("release");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with("format.") && n.ends_with(".render.json"))
        .map(|n| n.trim_end_matches(".json").to_string())
        .collect();
    names.sort();
    let (mut compared, mut skipped) = (0, Vec::new());
    for case in names {
        let metal = case.trim_start_matches("format.").trim_end_matches(".render");
        let Some(value) = webmetal_host::format::table::METAL_FORMATS
            .iter()
            .find(|r| r.name.strip_prefix("MTLPixelFormat") == Some(metal))
            .map(|r| r.value)
        else {
            skipped.push(format!("{metal}: not in the format table"));
            continue;
        };
        let g = Golden::load(&case);
        if g.status() != "ok" {
            skipped.push(format!("{metal}: reference status {}", g.status()));
            continue;
        }
        let resolution = webmetal_host::format::handling::resolve(value, h.g.features);
        if !matches!(resolution, Ok(webmetal_host::format::handling::Resolution::Native(_))) {
            skipped.push(format!(
                "{metal}: {resolution:?} (an emulated format's fragment output conversion is the pipeline variant's)"
            ));
            continue;
        }
        let input = g.data("render.input");
        let elem = g.record("render.input").and_then(|r| r.get("elem")).and_then(|e| e.str()).unwrap().to_string();
        let fragment = match elem.as_str() {
            "f32" => "fmt_fragment_float",
            "u32" => "fmt_fragment_uint",
            _ => "fmt_fragment_int",
        };
        let target = match h.texture(texture_descriptor(value as u64, 8, 4, STORAGE_PRIVATE, usage::RENDER_TARGET)) {
            Ok(t) => t,
            Err(f) => {
                skipped.push(format!("{metal}: the texture is refused on this host: {}", f.message));
                continue;
            }
        };
        let p = h.pipeline(h.pipeline_of("fmt_fullscreen_vertex", Some(fragment), value as u64)).unwrap();
        let (values, _) = h.buffer(&input);
        let e = h.handle();
        h.run_ok(&render(
            e,
            pass_color(target, clear(0.0, 0.0, 0.0, 0.0)),
            vec![
                set_pipeline(p),
                Command::RenderSetFragmentBuffer(RenderSetFragmentBuffer { buffer: Some(values), offset: 0, index: 0 }),
                Command::RenderSetFragmentBytes(RenderSetFragmentBytes {
                    bytes: 8u32.to_le_bytes().to_vec(),
                    length: 4,
                    index: 1,
                }),
                draw(prim::TRIANGLE, 0, 3),
            ],
        ));
        let half = g.data("render.representable").len();
        let bytes_per_texel = (half * 2 / 32) as u64;
        let stored = h.read_texture(target, 8, 4, 8 * bytes_per_texel);
        let mut c = Candidate::default();
        c.data(
            "render.input",
            if elem == "f32" {
                "f32"
            } else if elem == "u32" {
                "u32"
            } else {
                "i32"
            },
            &input,
        );
        c.data("render.representable", "u8", &stored[..half]);
        if g.record("render.representable-equals-specification").is_some() {
            // The reference records true: its stored bytes are the specification's expected bytes.
            c.bool("render.representable-equals-specification", stored[..half] == g.data("render.representable")[..]);
        }
        c.data("render.converted", "u8", &stored[half..]);
        check(&case, &c);
        compared += 1;
    }
    println!("format.*.render: {compared} formats compared; skipped: {skipped:#?}");
    assert!(compared > 0);
}

/// Texture views as color attachments: a slice of a 2D array as a 2D view, a level of a mipmapped texture, and an sRGB
/// view of a linear texture, which encodes the clear color. Not a reference case: the views are checked against their
/// parent's stored bytes.
#[test]
fn view_attachments() {
    let h = Harness::new();
    const RGBA8_UNORM_SRGB: u64 = 71;
    const T2D: u64 = 2;
    let usage = usage::RENDER_TARGET | usage::SHADER_READ;
    let view = |parent: Handle, format: u64, levels: (u64, u64), slices: (u64, u64)| {
        let v = h.handle();
        h.ok(Request::CreateTextureViewLevelsSlices(CreateTextureViewLevelsSlices {
            texture: parent,
            view: v,
            pixel_format: format,
            texture_type: T2D,
            level_range: Range { location: levels.0, length: levels.1 },
            slice_range: Range { location: slices.0, length: slices.1 },
        }));
        v
    };
    let clear_pass = |t: Handle, level: u64, slice: u64, c: ClearColor| {
        let mut p = pass_color(t, c);
        p.color_attachments[0].level = level;
        p.color_attachments[0].slice = slice;
        h.run_ok(&render(h.handle(), p, vec![]));
    };
    let texels = |rgba: [u8; 4], n: usize| rgba.repeat(n);

    // Slice 2 of a three-slice 2D array, as a 2D view.
    let mut d = texture_descriptor(pf::RGBA8_UNORM, 4, 4, STORAGE_PRIVATE, usage);
    d.texture_type = 3;
    d.array_length = 3;
    let array = h.texture(d).unwrap();
    for slice in 0..3 {
        clear_pass(array, 0, slice, clear(0.0, 0.0, 0.0, 1.0));
    }
    let slice2 = view(array, pf::RGBA8_UNORM, (0, 1), (2, 1));
    clear_pass(slice2, 0, 0, clear(1.0, 0.0, 1.0, 1.0));
    assert_eq!(h.read_texture_options(array, 2, 4, 4, 16, 0), texels([255, 0, 255, 255], 16));
    assert_eq!(h.read_texture_options(array, 1, 4, 4, 16, 0), texels([0, 0, 0, 255], 16));

    // Level 1 of an 8x8 texture with two levels: a 4x4 attachment, which leaves level 0 as it was.
    let mut d = texture_descriptor(pf::RGBA8_UNORM, 8, 8, STORAGE_PRIVATE, usage);
    d.mipmap_level_count = 2;
    let mipped = h.texture(d).unwrap();
    clear_pass(mipped, 0, 0, clear(1.0, 0.0, 0.0, 1.0));
    let level1 = view(mipped, pf::RGBA8_UNORM, (1, 1), (0, 1));
    clear_pass(level1, 0, 0, clear(0.0, 1.0, 0.0, 1.0));
    // The parent's level 1, read back by a blit of that level.
    let (b, addr) = h.zeros(64);
    h.run_ok(&[
        Command::BeginBlit(BeginBlit { encoder: h.handle() }),
        Command::BlitCopyTextureToBufferOptions(BlitCopyTextureToBufferOptions {
            source_texture: Some(mipped),
            source_slice: 0,
            source_level: 1,
            source_origin: Origin { x: 0, y: 0, z: 0 },
            source_size: Size { width: 4, height: 4, depth: 1 },
            destination_buffer: Some(b),
            destination_offset: 0,
            destination_bytes_per_row: 16,
            destination_bytes_per_image: 64,
            options: 0,
        }),
        end(),
    ]);
    assert_eq!(h.read(addr, 64), texels([0, 255, 0, 255], 16));
    assert_eq!(h.read_texture(mipped, 8, 8, 32), texels([255, 0, 0, 255], 64));
    // The view's level 1 is not the texture's: an attachment there is outside the view.
    let mut p = pass_color(level1, clear(0.0, 0.0, 1.0, 1.0));
    p.color_attachments[0].level = 1;
    let f = h.run(&render(h.handle(), p, vec![]));
    refused("view-attachments", &f, status::UNIMPLEMENTED, "level 1 of a texture with 1 levels");

    // An sRGB view of a linear texture: the clear color is encoded (0.5 is code 188 within the conversion's rounding)
    // and alpha is stored linearly.
    let linear = h.texture(texture_descriptor(pf::RGBA8_UNORM, 4, 4, STORAGE_PRIVATE, usage)).unwrap();
    let srgb = h.handle();
    h.ok(Request::CreateTextureView(CreateTextureView { texture: linear, view: srgb, pixel_format: RGBA8_UNORM_SRGB }));
    clear_pass(srgb, 0, 0, clear(0.5, 0.5, 0.5, 1.0));
    let stored = h.read_texture(linear, 4, 4, 16);
    for px in stored.chunks(4) {
        assert!(px[..3].iter().all(|&c| c.abs_diff(188) <= 1) && px[3] == 255, "{px:?}");
    }

    // A BGRA8Unorm view of an RGBA8Unorm texture, which the texture's alias holds: the clear color lands with red and
    // blue in the view's order. The texture's next pass, which loads it, takes the alias's texels into its stored
    // texture first (the harness's readback blit reads the stored texture).
    const BGRA8_UNORM: u64 = 80;
    let rgba = h
        .texture(texture_descriptor(pf::RGBA8_UNORM, 4, 4, STORAGE_PRIVATE, usage | usage::PIXEL_FORMAT_VIEW))
        .unwrap();
    clear_pass(rgba, 0, 0, clear(0.0, 0.0, 0.0, 1.0));
    let bgra = h.handle();
    h.ok(Request::CreateTextureView(CreateTextureView { texture: rgba, view: bgra, pixel_format: BGRA8_UNORM }));
    clear_pass(bgra, 0, 0, clear(1.0, 0.0, 0.0, 1.0));
    let mut p = pass_color(rgba, clear(0.0, 0.0, 0.0, 0.0));
    // MTLLoadActionLoad.
    p.color_attachments[0].load_action = 1;
    h.run_ok(&render(h.handle(), p, vec![]));
    assert_eq!(h.read_texture(rgba, 4, 4, 16), texels([0, 0, 255, 255], 16));
    h.g.assert_no_errors();
}
