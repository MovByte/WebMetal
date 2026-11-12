//! The blit rows of the Metal reference harness (tests/reference/COVERAGE.md, "Compute and blit encoders", and the
//! blit parts of the format cases) replayed through the executor on a native WebGPU implementation, step by step as
//! the case sources do them (tests/reference/src/cases/Blit.m, Formats.m, FormatsCompressed.m, Textures.m), with every
//! `exact` record of the reference output compared byte for byte. Error scopes are per call and strict, so a WebGPU
//! error fails the run.

#![cfg(feature = "native")]

mod h6_common;
mod mem_common;
mod pipeline_common;

use h6_common::*;
use mem_common::Golden;
use webmetal_wire::*;

/// The records of one case compared with the reference.
struct Tally {
    host: String,
    rows: Vec<(String, usize, String)>,
}

impl Tally {
    fn new(h: &Harness) -> Tally {
        Tally { host: h.name(), rows: Vec::new() }
    }

    /// Compare a `data` record with the reference: its rule must be `exact` and the bytes equal.
    fn data(&mut self, g: &Golden, name: &str, got: &[u8]) {
        assert_eq!(g.rule(name), Some("exact"), "{}: {name} is not an exact record", g.case);
        assert_eq!(got, g.bytes(name).unwrap().as_slice(), "{}: {}: {name}", self.host, g.case);
        self.count(g);
    }

    fn value_bool(&mut self, g: &Golden, name: &str, got: bool) {
        assert_eq!(g.rule(name), Some("exact"), "{}: {name}", g.case);
        assert_eq!(Some(got), g.value_bool(name), "{}: {}: {name}", self.host, g.case);
        self.count(g);
    }

    fn count(&mut self, g: &Golden) {
        match self.rows.iter_mut().find(|r| r.0 == g.case) {
            Some(r) => r.1 += 1,
            None => self.rows.push((g.case.clone(), 1, String::new())),
        }
    }

    fn note(&mut self, case: &str, note: impl Into<String>) {
        match self.rows.iter_mut().find(|r| r.0 == case) {
            Some(r) => r.2 = note.into(),
            None => self.rows.push((case.into(), 0, note.into())),
        }
    }

    fn print(&self) {
        for (case, n, note) in &self.rows {
            eprintln!(
                "{}: {case}: {n} exact records equal{}",
                self.host,
                if note.is_empty() { "" } else { "; " }.to_string() + note
            );
        }
    }
}

fn copy_b2b(src: Handle, so: u64, dst: Handle, d: u64, n: u64) -> Command {
    Command::BlitCopyBufferToBuffer(BlitCopyBufferToBuffer {
        source_buffer: Some(src),
        source_offset: so,
        destination_buffer: Some(dst),
        destination_offset: d,
        size: n,
    })
}

#[allow(clippy::too_many_arguments)]
fn copy_b2t(
    src: Handle,
    offset: u64,
    bpr: u64,
    bpi: u64,
    sz: Size,
    dst: Handle,
    slice: u64,
    level: u64,
    o: Origin,
) -> Command {
    Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
        source_buffer: Some(src),
        source_offset: offset,
        source_bytes_per_row: bpr,
        source_bytes_per_image: bpi,
        source_size: sz,
        destination_texture: Some(dst),
        destination_slice: slice,
        destination_level: level,
        destination_origin: o,
    })
}

#[allow(clippy::too_many_arguments)]
fn copy_t2b(
    src: Handle,
    slice: u64,
    level: u64,
    o: Origin,
    sz: Size,
    dst: Handle,
    offset: u64,
    bpr: u64,
    bpi: u64,
) -> Command {
    Command::BlitCopyTextureToBuffer(BlitCopyTextureToBuffer {
        source_texture: Some(src),
        source_slice: slice,
        source_level: level,
        source_origin: o,
        source_size: sz,
        destination_buffer: Some(dst),
        destination_offset: offset,
        destination_bytes_per_row: bpr,
        destination_bytes_per_image: bpi,
    })
}

fn fill(b: Handle, location: u64, length: u64, value: u8) -> Command {
    Command::BlitFillBuffer(BlitFillBuffer { buffer: Some(b), range: Range { location, length }, value })
}

/// blit.copy-buffer, blit.copy-buffer-unaligned (iOS only), blit.fill-buffer.
#[test]
fn buffer_copies_and_fills() {
    let h = Harness::new();
    let mut t = Tally::new(&h);

    let g = Golden::load("blit.copy-buffer");
    let source = pattern(512, 101);
    let (a, a_addr) = h.buffer_with(&source);
    let (b, b_addr) = h.buffer(512);
    h.guest.cpu_write(b_addr, &[0x5a; 512]);
    h.blit(vec![copy_b2b(a, 12, b, 100, 200), copy_b2b(a, 0, a, 256, 64)]);
    t.data(&g, "destination", &h.read(b_addr, 512));
    t.data(&g, "source-after-self-copy", &h.read(a_addr, 512));

    // blit.copy-buffer-unaligned requires iOS ("In macOS, sourceOffset needs to be a multiple of 4, but can be any value
    // in iOS and tvOS"): the macOS reference is `unsupported`, so the result is checked against the copy's definition.
    let g = Golden::load("blit.copy-buffer-unaligned");
    assert_eq!(g.status(), "unsupported");
    let source = pattern(128, 102);
    let (a, _) = h.buffer_with(&source);
    let (b, b_addr) = h.buffer(128);
    h.blit(vec![copy_b2b(a, 3, b, 7, 61)]);
    let mut want = vec![0u8; 128];
    want[7..68].copy_from_slice(&source[3..64]);
    assert_eq!(h.read(b_addr, 128), want, "{}: unaligned copy", h.name());
    t.note("blit.copy-buffer-unaligned", "macOS reference unsupported; 128 bytes equal the copy's definition");

    let g = Golden::load("blit.fill-buffer");
    let (b, b_addr) = h.buffer(256);
    h.guest.cpu_write(b_addr, &[0x11; 256]);
    h.blit(vec![fill(b, 16, 64, 0xa5), fill(b, 131, 29, 0x3c)]);
    t.data(&g, "buffer", &h.read(b_addr, 256));

    // Unaligned copies within one buffer, and a large unaligned fill.
    let source = pattern(4096, 7);
    let (a, a_addr) = h.buffer_with(&source);
    h.blit(vec![copy_b2b(a, 1, a, 2051, 2001), fill(a, 5, 1000, 0)]);
    let mut want = source.clone();
    want.copy_within(1..2002, 2051);
    want[5..1005].fill(0);
    assert_eq!(h.read(a_addr, 4096), want, "{}: self copy", h.name());
    t.note("blit.copy-buffer", "plus unaligned self-copy and fill checked against their definitions");

    t.print();
    h.assert_no_errors();
}

/// blit.buffer-to-texture-2d, -3d, blit.slices-and-levels, blit.texture-to-buffer, blit.texture-to-texture-region,
/// blit.texture-to-texture-whole, blit.optimize-contents.
#[test]
fn texture_blits() {
    let h = Harness::new();
    let mut t = Tally::new(&h);

    let g = Golden::load("blit.buffer-to-texture-2d");
    for (f, (name, bpp)) in [("RGBA8Unorm", 4u64), ("R16Uint", 2)].into_iter().enumerate() {
        let d = texture_descriptor(T2D, fmt(name), (16, 12, 1), 1, 1, PRIVATE, SHADER_READ, 1);
        let tex = h.texture(d.clone());
        h.fill_texture(tex, &d, bpp, 200 + f as u32);
        let (pitch, offset) = (7 * bpp + bpp * 3, bpp * 5);
        let len = offset + pitch * 5;
        let (src, _) = h.buffer_with(&pattern(len as usize, 210 + f as u32));
        h.blit(vec![copy_b2t(src, offset, pitch, 0, size(7, 5, 1), tex, 0, 0, origin(3, 2, 0))]);
        t.data(&g, &format!("{name}.texture"), &h.read_texture(tex, &d, 0, 0, 16 * bpp));
    }

    let g = Golden::load("blit.buffer-to-texture-3d");
    let d = texture_descriptor(T3D, fmt("RGBA8Uint"), (8, 6, 5), 1, 1, PRIVATE, SHADER_READ, 1);
    let tex = h.texture(d.clone());
    h.fill_texture(tex, &d, 4, 300);
    let (row, image, offset) = (24, 24 * 4 + 8, 16);
    let (src, _) = h.buffer_with(&pattern((offset + image * 3) as usize, 301));
    let (region, region_addr) = h.buffer(5 * 4 * 2 * 4 + 64);
    h.guest.cpu_write(region_addr, &[0x77; 5 * 4 * 2 * 4 + 64]);
    h.blit(vec![
        copy_b2t(src, offset, row, image, size(5, 3, 3), tex, 0, 0, origin(2, 1, 1)),
        copy_t2b(tex, 0, 0, origin(1, 1, 2), size(5, 4, 2), region, 64, 20, 80),
    ]);
    t.data(&g, "texture", &h.read_texture(tex, &d, 0, 0, 32));
    t.data(&g, "region-readback", &h.read(region_addr, 224));

    let g = Golden::load("blit.slices-and-levels");
    let da = texture_descriptor(T2D_ARRAY, fmt("R32Uint"), (8, 8, 1), 3, 3, PRIVATE, SHADER_READ, 1);
    let dc = texture_descriptor(TCUBE, fmt("RG16Uint"), (4, 4, 1), 1, 2, PRIVATE, SHADER_READ, 1);
    let (array, cube) = (h.texture(da.clone()), h.texture(dc.clone()));
    h.fill_texture(array, &da, 4, 400);
    h.fill_texture(cube, &dc, 4, 500);
    let (src, _) = h.buffer_with(&pattern(64, 401));
    h.blit(vec![
        copy_b2t(src, 4, 16, 0, size(3, 2, 1), array, 2, 1, origin(1, 2, 0)),
        copy_b2t(src, 32, 8, 0, size(2, 2, 1), cube, 4, 1, origin(0, 0, 0)),
    ]);
    for slice in 0..3 {
        for level in 0..3 {
            let w = 8 >> level;
            t.data(&g, &format!("array.slice{slice}.level{level}"), &h.read_texture(array, &da, level, slice, w * 4));
        }
    }
    for face in 0..6 {
        for level in 0..2 {
            let w = 4 >> level;
            t.data(&g, &format!("cube.face{face}.level{level}"), &h.read_texture(cube, &dc, level, face, w * 4));
        }
    }

    let g = Golden::load("blit.texture-to-buffer");
    let d1 = texture_descriptor(T1D, fmt("RGBA16Uint"), (32, 1, 1), 1, 1, PRIVATE, SHADER_READ, 1);
    let d2 = texture_descriptor(T2D, fmt("R8Uint"), (13, 9, 1), 1, 1, PRIVATE, SHADER_READ, 1);
    let d3 = texture_descriptor(T1D_ARRAY, fmt("R32Float"), (16, 1, 1), 4, 1, PRIVATE, SHADER_READ, 1);
    let (t1, t2, t3) = (h.texture(d1.clone()), h.texture(d2.clone()), h.texture(d3.clone()));
    h.fill_texture(t1, &d1, 8, 600);
    h.fill_texture(t2, &d2, 1, 601);
    h.fill_texture(t3, &d3, 4, 602);
    let (out, out_addr) = h.buffer(1024);
    h.guest.cpu_write(out_addr, &[0xcd; 1024]);
    h.blit(vec![
        copy_t2b(t1, 0, 0, origin(5, 0, 0), size(20, 1, 1), out, 8, 0, 0),
        copy_t2b(t2, 0, 0, origin(2, 3, 0), size(9, 5, 1), out, 200, 11, 0),
        copy_t2b(t3, 3, 0, origin(4, 0, 0), size(10, 1, 1), out, 512, 0, 0),
    ]);
    t.data(&g, "buffer", &h.read(out_addr, 1024));

    let g = Golden::load("blit.texture-to-texture-region");
    let da = texture_descriptor(T2D_ARRAY, fmt("RGBA8Unorm"), (16, 16, 1), 2, 2, PRIVATE, SHADER_READ, 1);
    let db = texture_descriptor(T2D_ARRAY, fmt("RGBA8Unorm"), (8, 8, 1), 3, 1, PRIVATE, SHADER_READ, 1);
    let dv = texture_descriptor(T3D, fmt("R16Float"), (6, 6, 6), 1, 1, PRIVATE, SHADER_READ, 1);
    let dw = texture_descriptor(T3D, fmt("R16Float"), (4, 4, 4), 1, 1, PRIVATE, SHADER_READ, 1);
    let (a, b, v, w) = (h.texture(da.clone()), h.texture(db.clone()), h.texture(dv.clone()), h.texture(dw.clone()));
    h.fill_texture(a, &da, 4, 700);
    h.fill_texture(b, &db, 4, 701);
    h.fill_texture(v, &dv, 2, 702);
    h.fill_texture(w, &dw, 2, 703);
    let t2t = |s, ss, sl, so, sz, d, ds, dl, dor| {
        Command::BlitCopyTextureToTexture(BlitCopyTextureToTexture {
            source_texture: Some(s),
            source_slice: ss,
            source_level: sl,
            source_origin: so,
            source_size: sz,
            destination_texture: Some(d),
            destination_slice: ds,
            destination_level: dl,
            destination_origin: dor,
        })
    };
    h.blit(vec![
        t2t(a, 1, 1, origin(1, 1, 0), size(5, 4, 1), b, 2, 0, origin(2, 3, 0)),
        t2t(v, 0, 0, origin(1, 2, 3), size(3, 2, 2), w, 0, 0, origin(0, 1, 1)),
        t2t(b, 0, 0, origin(0, 0, 0), size(4, 4, 1), b, 1, 0, origin(4, 4, 0)),
    ]);
    for slice in 0..3 {
        t.data(&g, &format!("b.slice{slice}"), &h.read_texture(b, &db, 0, slice, 32));
    }
    t.data(&g, "w", &h.read_texture(w, &dw, 0, 0, 8));

    let g = Golden::load("blit.texture-to-texture-whole");
    let dab = texture_descriptor(T2D_ARRAY, fmt("RG8Uint"), (8, 4, 1), 3, 3, PRIVATE, SHADER_READ, 1);
    let dd = texture_descriptor(T2D_ARRAY, fmt("RG8Uint"), (4, 2, 1), 4, 2, PRIVATE, SHADER_READ, 1);
    let (a, b, d) = (h.texture(dab.clone()), h.texture(dab.clone()), h.texture(dd.clone()));
    h.fill_texture(a, &dab, 2, 800);
    h.fill_texture(b, &dab, 2, 801);
    h.fill_texture(d, &dd, 2, 802);
    h.blit(vec![
        Command::BlitCopyTextureToTextureAll(BlitCopyTextureToTextureAll {
            source_texture: Some(a),
            destination_texture: Some(b),
        }),
        Command::BlitCopyTextureToTextureSlicesLevels(BlitCopyTextureToTextureSlicesLevels {
            source_texture: Some(a),
            source_slice: 1,
            source_level: 1,
            destination_texture: Some(d),
            destination_slice: 2,
            destination_level: 0,
            slice_count: 2,
            level_count: 2,
        }),
    ]);
    for slice in 0..3 {
        for level in 0..3 {
            let width = 8 >> level;
            t.data(&g, &format!("b.slice{slice}.level{level}"), &h.read_texture(b, &dab, level, slice, width * 2));
        }
    }
    for slice in 0..4 {
        for level in 0..2 {
            let width = 4 >> level;
            t.data(&g, &format!("d.slice{slice}.level{level}"), &h.read_texture(d, &dd, level, slice, width * 2));
        }
    }

    let g = Golden::load("blit.optimize-contents");
    let d = texture_descriptor(T2D, fmt("RGBA8Uint"), (8, 8, 1), 1, 4, SHARED, SHADER_READ, 1);
    let tex = h.texture(d.clone());
    let region = |w: u64| Region { origin: origin(0, 0, 0), size: size(w, w, 1) };
    for level in 0..4 {
        let w = 8 >> level;
        let bytes = pattern((w * w * 4) as usize, 1000 + level as u32);
        let addr = h.alloc(bytes.len() as u64);
        h.guest.cpu_write(addr, &bytes);
        h.ok(Request::ReplaceRegion(ReplaceRegion {
            texture: tex,
            region: region(w),
            level,
            pixel_bytes: GuestAddr(addr),
            bytes_per_row: w * 4,
        }));
    }
    let before = h.read_texture(tex, &d, 0, 0, 32);
    h.blit(vec![
        Command::BlitOptimizeContentsForGPUAccess(BlitOptimizeContentsForGPUAccess { texture: Some(tex) }),
        Command::BlitOptimizeContentsForCPUAccessSliceLevel(BlitOptimizeContentsForCPUAccessSliceLevel {
            texture: Some(tex),
            slice: 0,
            level: 1,
        }),
        Command::BlitOptimizeContentsForGPUAccessSliceLevel(BlitOptimizeContentsForGPUAccessSliceLevel {
            texture: Some(tex),
            slice: 0,
            level: 2,
        }),
        Command::BlitOptimizeContentsForCPUAccess(BlitOptimizeContentsForCPUAccess { texture: Some(tex) }),
    ]);
    for level in 0..4 {
        let w = 8 >> level;
        let addr = h.alloc(w * w * 4);
        h.ok(Request::GetBytes(GetBytes {
            texture: tex,
            pixel_bytes: GuestAddr(addr),
            bytes_per_row: w * 4,
            region: region(w),
            level,
        }));
        t.data(&g, &format!("level{level}"), &h.read(addr, (w * w * 4) as usize));
    }
    t.value_bool(&g, "level0-unchanged", before == h.read_texture(tex, &d, 0, 0, 32));

    t.print();
    h.assert_no_errors();
}

/// blit.texture-to-texture-multisample: a cleared 4-sample texture copied whole and read per sample by a kernel.
#[test]
fn multisample_copy() {
    let h = Harness::new();
    let mut t = Tally::new(&h);
    let g = Golden::load("blit.texture-to-texture-multisample");
    let d = texture_descriptor(T2D_MS, fmt("R32Uint"), (4, 4, 1), 1, 1, PRIVATE, RENDER_TARGET | SHADER_READ, 4);
    let a = h.handle();
    let f = h.status(Request::CreateTexture(CreateTexture { texture: a, descriptor: d.clone() }));
    if f.status != status::OK {
        // WebGPU has no multisampled r32uint (WebGPU section 26.1.1): the texture is refused with status 3.
        assert_eq!(f.status, status::HOST_UNSUPPORTED, "{f:?}");
        t.note(&g.case, format!("refused on this host: {}", f.message));
        t.print();
        return;
    }
    let b = h.texture(d.clone());
    let e = h.handle();
    h.encode(
        Command::BeginRenderPass(BeginRenderPass { encoder: e, render_pass_descriptor: clear_pass(a, 305419896.0) }),
        vec![],
    );
    h.blit(vec![Command::BlitCopyTextureToTextureAll(BlitCopyTextureToTextureAll {
        source_texture: Some(a),
        destination_texture: Some(b),
    })]);
    let p = h.pipeline("blit_read_ms", None, 0);
    let (out, out_addr) = h.buffer(4 * 4 * 4 * 4);
    h.compute(vec![
        Command::ComputeSetComputePipelineState(ComputeSetComputePipelineState { state: Some(p) }),
        Command::ComputeSetTexture(ComputeSetTexture { texture: Some(b), index: 0 }),
        Command::ComputeSetBuffer(ComputeSetBuffer { buffer: Some(out), offset: 0, index: 0 }),
        Command::ComputeDispatchThreadgroups(ComputeDispatchThreadgroups {
            threadgroups_per_grid: size(1, 1, 1),
            threads_per_threadgroup: size(4, 4, 1),
        }),
    ]);
    t.data(&g, "samples", &h.read(out_addr, 256));
    t.print();
    h.assert_no_errors();
}

/// A render pass descriptor that clears color attachment 0 to `red` and stores it.
fn clear_pass(texture: Handle, red: f64) -> RenderPassDescriptor {
    RenderPassDescriptor {
        binary_archives: Vec::new(),
        color_attachments: vec![RenderPassColorAttachment {
            clear_color: ClearColor { red, green: 0.0, blue: 0.0, alpha: 0.0 },
            depth_plane: 0,
            level: 0,
            load_action: 2,
            resolve_depth_plane: 0,
            resolve_level: 0,
            resolve_slice: 0,
            resolve_texture: None,
            slice: 0,
            store_action: 1,
            store_action_options: 0,
            texture: Some(texture),
            y_invert: false,
        }],
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
        sample_buffer_attachments: Vec::new(),
        skip_empty_tiles_on_clear_enabled: false,
        stencil_attachment: None,
        threadgroup_memory_length: 0,
        tile_height: 0,
        tile_width: 0,
        visibility_result_buffer: None,
        sample_positions: Vec::new(),
    }
}

/// The depth values of `FmtDepthValues` (Formats.m).
fn depth_values(seed: u32) -> Vec<u8> {
    let random = words(&pattern(128, seed));
    (0..32)
        .map(|t| match t {
            0 => 0.0f32,
            1 => 1.0,
            2 => 0.5,
            _ => ((random[t] >> 8) & 0xff_ffff) as f32 / 16_777_216.0,
        })
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// `FmtRunRead`: a kernel that reads every texel of `texture` into a buffer of `element` bytes per texel.
fn run_read(h: &Harness, texture: Handle, kernel: &str, element: usize) -> Vec<u8> {
    let p = h.pipeline(kernel, None, 0);
    let (out, addr) = h.buffer(8 * 4 * element);
    h.compute(vec![
        Command::ComputeSetComputePipelineState(ComputeSetComputePipelineState { state: Some(p) }),
        Command::ComputeSetTexture(ComputeSetTexture { texture: Some(texture), index: 0 }),
        Command::ComputeSetBuffer(ComputeSetBuffer { buffer: Some(out), offset: 0, index: 0 }),
        Command::ComputeDispatchThreadgroups(ComputeDispatchThreadgroups {
            threadgroups_per_grid: size(1, 1, 1),
            threads_per_threadgroup: size(8, 8, 1),
        }),
    ]);
    h.read(addr, 8 * 4 * element)
}

/// `FmtBlitUpload` and `FmtBlitReadback` of Formats.m: 8 x 4 texels of one aspect through blits.
fn upload(h: &Harness, texture: Handle, bytes: &[u8], bpp: u64, options: u64) {
    let (src, _) = h.buffer_with(bytes);
    h.blit(vec![Command::BlitCopyBufferToTextureOptions(BlitCopyBufferToTextureOptions {
        source_buffer: Some(src),
        source_offset: 0,
        source_bytes_per_row: 8 * bpp,
        source_bytes_per_image: 32 * bpp,
        source_size: size(8, 4, 1),
        destination_texture: Some(texture),
        destination_slice: 0,
        destination_level: 0,
        destination_origin: origin(0, 0, 0),
        options,
    })]);
}

fn readback(h: &Harness, texture: Handle, bpp: u64, options: u64) -> Vec<u8> {
    let (dst, addr) = h.buffer((32 * bpp) as usize);
    h.blit(vec![Command::BlitCopyTextureToBufferOptions(BlitCopyTextureToBufferOptions {
        source_texture: Some(texture),
        source_slice: 0,
        source_level: 0,
        source_origin: origin(0, 0, 0),
        source_size: size(8, 4, 1),
        destination_buffer: Some(dst),
        destination_offset: 0,
        destination_bytes_per_row: 8 * bpp,
        destination_bytes_per_image: 32 * bpp,
        options,
    })]);
    h.read(addr, (32 * bpp) as usize)
}

/// The depth and stencil format cases: per-aspect blits and reads (Formats.m).
#[test]
fn depth_stencil_blits() {
    let h = Harness::new();
    let mut t = Tally::new(&h);
    let depth_tex = |name: &str, usage: u64| {
        let d = texture_descriptor(T2D, fmt(name), (8, 4, 1), 1, 1, PRIVATE, usage, 1);
        let tex = h.handle();
        let f = h.status(Request::CreateTexture(CreateTexture { texture: tex, descriptor: d }));
        (f.status == status::OK).then_some(tex).ok_or(f)
    };

    let g = Golden::load("format.Depth32Float.upload-read");
    let values = depth_values(0x4002);
    let tex = depth_tex("Depth32Float", SHADER_READ).unwrap();
    upload(&h, tex, &values, 4, 0);
    t.data(&g, "input", &values);
    t.data(&g, "readback", &readback(&h, tex, 4, 0));
    let read = run_read(&h, tex, "fmt_read_depth", 4);
    t.data(&g, "read", &read);
    t.value_bool(&g, "read-equals-stored", read == values);

    let g = Golden::load("format.Depth16Unorm.upload-read");
    let mut codes = pattern(64, 0x4001);
    codes[0..2].copy_from_slice(&0u16.to_le_bytes());
    codes[2..4].copy_from_slice(&0xffffu16.to_le_bytes());
    let tex = depth_tex("Depth16Unorm", SHADER_READ).unwrap();
    upload(&h, tex, &codes, 2, 0);
    t.data(&g, "input", &codes);
    t.data(&g, "readback", &readback(&h, tex, 2, 0));
    let read = run_read(&h, tex, "fmt_read_depth", 4);
    t.data(&g, "read.exact", &read[..8]);

    let g = Golden::load("format.Stencil8.upload-read");
    let mut stencil = pattern(32, 0x4003);
    stencil[0] = 0;
    stencil[1] = 255;
    let tex = depth_tex("Stencil8", SHADER_READ).unwrap();
    upload(&h, tex, &stencil, 1, 0);
    t.data(&g, "input", &stencil);
    t.data(&g, "readback", &readback(&h, tex, 1, 0));
    let read = words(&run_read(&h, tex, "fmt_read_uint", 16));
    let first: Vec<u8> = (0..32).flat_map(|i| read[i * 4].to_le_bytes()).collect();
    t.data(&g, "read.stencil", &first);
    t.value_bool(&g, "read.stencil-equals-stored", (0..32).all(|i| read[i * 4] == u32::from(stencil[i])));

    let g = Golden::load("format.Depth32Float_Stencil8.upload-read");
    match depth_tex("Depth32Float_Stencil8", SHADER_READ | 0x10) {
        Err(f) => {
            assert_eq!(f.status, status::HOST_UNSUPPORTED, "{f:?}");
            t.note(&g.case, format!("refused on this host: {}", f.message));
        }
        Ok(tex) => {
            let depth = depth_values(0x4004);
            let stencil = pattern(32, 0x4005);
            upload(&h, tex, &depth, 4, 1);
            upload(&h, tex, &stencil, 1, 2);
            t.data(&g, "input.depth", &depth);
            t.data(&g, "input.stencil", &stencil);
            t.data(&g, "readback.depth", &readback(&h, tex, 4, 1));
            t.data(&g, "readback.stencil", &readback(&h, tex, 1, 2));
            let read = run_read(&h, tex, "fmt_read_depth", 4);
            t.data(&g, "read.depth", &read);
            t.value_bool(&g, "read.depth-equals-stored", read == depth);
            t.note(&g.case, "the X32_Stencil8 view records need texture views, which no service creates yet");
        }
    }

    // A depth value the render-pass write does not keep is refused, as is a source whose bytes are only on the GPU.
    let tex = depth_tex("Depth32Float", SHADER_READ).unwrap();
    let mut bad = depth_values(9);
    bad[4..8].copy_from_slice(&1.5f32.to_le_bytes());
    let (src, _) = h.buffer_with(&bad);
    let f = h.blit_failure(vec![Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
        source_buffer: Some(src),
        source_offset: 0,
        source_bytes_per_row: 32,
        source_bytes_per_image: 0,
        source_size: size(8, 4, 1),
        destination_texture: Some(tex),
        destination_slice: 0,
        destination_level: 0,
        destination_origin: origin(0, 0, 0),
    })]);
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("0x3fc00000"), "{f:?}");

    t.print();
    h.assert_no_errors();
}

/// format.<format>.blocks: blit readback and texture copy of compressed blocks (FormatsCompressed.m), for every
/// compressed format with a reference case that this host stores (natively or decoded).
#[test]
fn compressed_block_blits() {
    let h = Harness::new();
    let mut t = Tally::new(&h);
    let (mut exact, mut refused) = (0, Vec::new());
    for row in webmetal_host::format::table::METAL_FORMATS.iter() {
        let name = row.name.trim_start_matches("MTLPixelFormat");
        let case = format!("format.{name}.blocks");
        if !Golden::exists(&case) {
            continue;
        }
        let g = Golden::load(&case);
        if g.status() != "ok" {
            continue;
        }
        let Some(input) = g.bytes("input") else { continue };
        let (w, hh) = (g.layout("input", "width").unwrap(), g.layout("input", "height").unwrap());
        let bpr_cpu = g.layout("input", "bytes_per_row").unwrap();
        let (bw, bb) = (g.layout("input", "block_width").unwrap(), g.layout("input", "block_bytes").unwrap());
        let pvrtc = name.starts_with("PVRTC");
        let bpr = (w / bw) * bb;
        let d = texture_descriptor(T2D, u64::from(row.value), (w, hh, 1), 1, 1, SHARED, SHADER_READ, 1);
        let tex = h.handle();
        let f = h.status(Request::CreateTexture(CreateTexture { texture: tex, descriptor: d.clone() }));
        if f.status != status::OK {
            // Not a format of the A10X (BC: MTLPixelFormatGetInfoForDevice's invalid record), or one this host lacks.
            if f.status == status::UNIMPLEMENTED && f.message.contains("invalid record") {
                continue;
            }
            assert_eq!(f.status, status::HOST_UNSUPPORTED, "{name}: {f:?}");
            refused.push(name.to_string());
            continue;
        }
        let addr = h.alloc(input.len() as u64);
        h.guest.cpu_write(addr, &input);
        h.ok(Request::ReplaceRegionSlice(ReplaceRegionSlice {
            texture: tex,
            region: Region { origin: origin(0, 0, 0), size: size(w, hh, 1) },
            level: 0,
            slice: 0,
            pixel_bytes: GuestAddr(addr),
            bytes_per_row: bpr_cpu,
            bytes_per_image: 0,
        }));
        let options = if pvrtc { 4 } else { 0 };
        let len = input.len();
        let (rb, rb_addr) = h.buffer(len);
        let to_buffer = |src: Handle, dst: Handle| {
            Command::BlitCopyTextureToBufferOptions(BlitCopyTextureToBufferOptions {
                source_texture: Some(src),
                source_slice: 0,
                source_level: 0,
                source_origin: origin(0, 0, 0),
                source_size: size(w, hh, 1),
                destination_buffer: Some(dst),
                destination_offset: 0,
                destination_bytes_per_row: bpr,
                destination_bytes_per_image: len as u64,
                options,
            })
        };
        h.blit(vec![to_buffer(tex, rb)]);
        let readback = h.read(rb_addr, len);
        t.data(&g, if pvrtc { "readback-row-linear" } else { "readback" }, &readback);
        let mut cd = d.clone();
        cd.storage_mode = PRIVATE;
        cd.resource_options = PRIVATE << 4;
        let copy = h.texture(cd);
        let (cb, cb_addr) = h.buffer(len);
        h.blit(vec![
            Command::BlitCopyTextureToTextureAll(BlitCopyTextureToTextureAll {
                source_texture: Some(tex),
                destination_texture: Some(copy),
            }),
            to_buffer(copy, cb),
        ]);
        t.value_bool(&g, "copy-equals-source", h.read(cb_addr, len) == readback);
        exact += 1;
    }
    eprintln!("{}: compressed blit cases: {exact} exact, refused on this host: {refused:?}", h.name());
    assert!(exact >= 10, "{exact}");
    t.print();
    h.assert_no_errors();
}

/// texture.generate-mipmaps: level 0 unchanged (exact); the other levels are the implementation's filter (`none` in
/// the reference), checked here to be the average of the four texels each covers, to within one code of rounding.
#[test]
fn mipmap_generation() {
    let h = Harness::new();
    let mut t = Tally::new(&h);
    let g = Golden::load("texture.generate-mipmaps");
    let d = texture_descriptor(T2D, fmt("RGBA8Unorm"), (8, 8, 1), 1, 4, PRIVATE, SHADER_READ | RENDER_TARGET, 1);
    let tex = h.texture(d.clone());
    assert_eq!(g.value_u64("mipmap-level-count"), Some(4));
    let base = pattern(256, 73);
    let (src, _) = h.buffer_with(&base);
    h.blit(vec![copy_b2t(src, 0, 32, 0, size(8, 8, 1), tex, 0, 0, origin(0, 0, 0))]);
    h.blit(vec![Command::BlitGenerateMipmapsForTexture(BlitGenerateMipmapsForTexture { texture: Some(tex) })]);
    let mut above = h.read_texture(tex, &d, 0, 0, 32);
    t.data(&g, "level0", &above);
    for level in 1..4u64 {
        let w = 8usize >> level;
        let got = h.read_texture(tex, &d, level, 0, w as u64 * 4);
        let wa = w * 2;
        for y in 0..w {
            for x in 0..w {
                for c in 0..4 {
                    let at = |xx: usize, yy: usize| f64::from(above[(yy * wa + xx) * 4 + c]);
                    let avg =
                        (at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1))
                            / 4.0;
                    let v = f64::from(got[(y * w + x) * 4 + c]);
                    assert!((v - avg).abs() <= 1.0, "{}: level {level} ({x}, {y}) channel {c}: {v} vs {avg}", h.name());
                }
            }
        }
        above = got;
    }
    t.note(&g.case, "levels 1-3 are within one code of the 2x2 average");
    t.print();
    h.assert_no_errors();
}

/// Blits Metal does not define, or that the host cannot serve yet, complete with status 2 and the reason.
#[test]
fn blit_refusals() {
    let h = Harness::new();
    let (a, _) = h.buffer(256);
    let expect = |f: Failure, what: &str| {
        assert_eq!(f.status, status::UNIMPLEMENTED, "{what}: {f:?}");
        assert!(f.message.contains(what), "{what}: {f:?}");
        eprintln!("{}: refused: {}", h.name(), f.message);
    };
    // blit.texture-to-texture-format-mismatch: the debug layer asserts; undefined without it.
    let d1 = texture_descriptor(T2D, fmt("RGBA8Unorm"), (4, 4, 1), 1, 1, PRIVATE, SHADER_READ, 1);
    let d2 = texture_descriptor(T2D, fmt("RGBA8Uint"), (4, 4, 1), 1, 1, PRIVATE, SHADER_READ, 1);
    let (x, y) = (h.texture(d1.clone()), h.texture(d2.clone()));
    h.fill_texture(x, &d1, 4, 900);
    h.fill_texture(y, &d2, 4, 901);
    expect(
        h.blit_failure(vec![Command::BlitCopyTextureToTexture(BlitCopyTextureToTexture {
            source_texture: Some(x),
            source_slice: 0,
            source_level: 0,
            source_origin: origin(0, 0, 0),
            source_size: size(4, 4, 1),
            destination_texture: Some(y),
            destination_slice: 0,
            destination_level: 0,
            destination_origin: origin(0, 0, 0),
        })]),
        "differs from destination pixel format",
    );
    expect(h.blit_failure(vec![copy_b2b(a, 0, a, 16, 64)]), "overlapping regions within the same buffer");
    expect(h.blit_failure(vec![fill(a, 0, 0, 1)]), "range 0+0");
    expect(
        h.blit_failure(vec![Command::BlitResetCommandsInBuffer(BlitResetCommandsInBuffer {
            buffer: None,
            range: Range { location: 0, length: 1 },
        })]),
        "indirect command buffer",
    );
    expect(
        h.blit_failure(vec![Command::BlitResolveCounters(BlitResolveCounters {
            sample_buffer: None,
            range: Range { location: 0, length: 1 },
            destination_buffer: Some(a),
            destination_offset: 0,
        })]),
        "counter sample buffer",
    );
    expect(
        h.blit_failure(vec![Command::BlitSampleCountersInBuffer(BlitSampleCountersInBuffer {
            sample_buffer: None,
            sample_index: 0,
            barrier: true,
        })]),
        "MTLCounterSamplingPointAtBlitBoundary",
    );
    expect(
        h.blit_failure(vec![Command::BlitSynchronizeResource(BlitSynchronizeResource { resource: Some(a) })]),
        "API_UNAVAILABLE(ios)",
    );
    h.assert_no_errors();
}
