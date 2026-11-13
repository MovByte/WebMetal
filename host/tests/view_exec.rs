//! Texture views (`newTextureViewWithPixelFormat:` and its five variants) through the executor on a native WebGPU
//! implementation: a view of each plan the format module gives, CPU transfers and blits through views, compute bindings
//! of views, and the views and uses the host refuses. The reference cases `texture.view-levels-slices` and
//! `texture.view-format-reinterpret` (tests/reference/src/cases/Textures.m) are replayed where the host provides them,
//! with every `exact` record compared byte for byte.

#![cfg(feature = "native")]

mod h6_common;
mod mem_common;
mod pipeline_common;

use h6_common::*;
use mem_common::Golden;
use webmetal_host::format::webgpu::Feature;
use webmetal_wire::*;

const PIXEL_FORMAT_VIEW: u64 = 0x10;
const TCUBE_ARRAY: u64 = 6;

fn range(location: u64, length: u64) -> Range {
    Range { location, length }
}

fn region(o: (u64, u64, u64), s: (u64, u64, u64)) -> Region {
    Region { origin: origin(o.0, o.1, o.2), size: size(s.0, s.1, s.2) }
}

/// `TexValue` of Textures.m: every texel of every level and slice distinct.
fn tex_value(level: u64, slice: u64, y: u64, x: u64) -> u32 {
    0xa000_0000 | (level << 24 | slice << 16 | y << 8 | x) as u32
}

fn mip(size: u64, level: u64) -> u64 {
    (size >> level).max(1)
}

/// `newTextureViewWithPixelFormat:textureType:levels:slices:`: the view's handle, or the failure.
fn view(
    h: &Harness,
    parent: Handle,
    format: u64,
    kind: u64,
    levels: (u64, u64),
    slices: (u64, u64),
) -> Result<Handle, Failure> {
    let v = h.handle();
    let f = h.status(Request::CreateTextureViewLevelsSlices(CreateTextureViewLevelsSlices {
        texture: parent,
        view: v,
        pixel_format: format,
        texture_type: kind,
        level_range: range(levels.0, levels.1),
        slice_range: range(slices.0, slices.1),
    }));
    if f.status == status::OK { Ok(v) } else { Err(f) }
}

/// `newTextureViewWithPixelFormat:`.
fn format_view(h: &Harness, parent: Handle, format: u64) -> Result<Handle, Failure> {
    let v = h.handle();
    let f = h.status(Request::CreateTextureView(CreateTextureView { texture: parent, view: v, pixel_format: format }));
    if f.status == status::OK { Ok(v) } else { Err(f) }
}

/// `TexUploadAll` of Textures.m: every level and slice of an R32Uint texture through blits, with [`tex_value`].
fn upload_all(h: &Harness, t: Handle, d: &TextureDescriptor) {
    let slices = d.array_length * if d.texture_type == TCUBE || d.texture_type == TCUBE_ARRAY { 6 } else { 1 };
    for level in 0..d.mipmap_level_count {
        let (w, hh) = (mip(d.width, level), mip(d.height, level));
        for slice in 0..slices {
            let bytes: Vec<u8> = (0..hh)
                .flat_map(|y| (0..w).map(move |x| (y, x)))
                .flat_map(|(y, x)| tex_value(level, slice, y, x).to_le_bytes())
                .collect();
            let (src, _) = h.buffer_with(&bytes);
            h.blit(vec![copy_to_texture(src, w * 4, (w, hh), t, slice, level)]);
        }
    }
}

fn copy_to_texture(src: Handle, bpr: u64, (w, hh): (u64, u64), t: Handle, slice: u64, level: u64) -> Command {
    Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
        source_buffer: Some(src),
        source_offset: 0,
        source_bytes_per_row: bpr,
        source_bytes_per_image: 0,
        source_size: size(w, hh, 1),
        destination_texture: Some(t),
        destination_slice: slice,
        destination_level: level,
        destination_origin: origin(0, 0, 0),
    })
}

/// One read or write query of shaders/textures.metal: two uint4, (x, y, z, w) and the level.
fn query(x: u32, y: u32, z: u32, lod: u32) -> [u32; 8] {
    [x, y, z, 0, lod, 0, 0, 0]
}

fn query_bytes(q: &[[u32; 8]]) -> Vec<u8> {
    q.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
}

fn set_texture(t: Handle) -> Command {
    Command::ComputeSetTexture(ComputeSetTexture { texture: Some(t), index: 0 })
}

fn set_buffer(b: Handle, index: u64) -> Command {
    Command::ComputeSetBuffer(ComputeSetBuffer { buffer: Some(b), offset: 0, index })
}

/// `TexRunReads`: one threadgroup of one thread per query; the uint4 (or float4) results.
fn run_reads(h: &Harness, kernel: &str, t: Handle, q: &[[u32; 8]]) -> Vec<u8> {
    let (records, addr) = read_records(h, kernel, t, q);
    h.compute(records);
    h.read(addr, q.len() * 16)
}

/// The records of [`run_reads`] and the guest address of the results.
fn read_records(h: &Harness, kernel: &str, t: Handle, q: &[[u32; 8]]) -> (Vec<Command>, u64) {
    let p = h.pipeline(kernel, None, 0);
    let (qb, _) = h.buffer_with(&query_bytes(q));
    let (out, addr) = h.buffer(q.len() * 16);
    let records = vec![
        Command::ComputeSetComputePipelineState(ComputeSetComputePipelineState { state: Some(p) }),
        set_texture(t),
        set_buffer(qb, 0),
        set_buffer(out, 1),
        Command::ComputeDispatchThreadgroups(ComputeDispatchThreadgroups {
            threadgroups_per_grid: size(q.len() as u64, 1, 1),
            threads_per_threadgroup: size(1, 1, 1),
        }),
    ];
    (records, addr)
}

/// `TexRunWrites`: the dispatch's records, for a run expected to succeed or fail.
fn write_records(h: &Harness, kernel: &str, t: Handle, q: &[[u32; 8]], values: &[[u32; 4]]) -> Vec<Command> {
    let p = h.pipeline(kernel, None, 0);
    let (qb, _) = h.buffer_with(&query_bytes(q));
    let bytes: Vec<u8> = values.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    let (vb, _) = h.buffer_with(&bytes);
    vec![
        Command::ComputeSetComputePipelineState(ComputeSetComputePipelineState { state: Some(p) }),
        set_texture(t),
        set_buffer(qb, 0),
        set_buffer(vb, 1),
        Command::ComputeDispatchThreadgroups(ComputeDispatchThreadgroups {
            threadgroups_per_grid: size(q.len() as u64, 1, 1),
            threads_per_threadgroup: size(1, 1, 1),
        }),
    ]
}

/// The records of `fmt_write_uint` (formats.metal) writing every texel of a `w` × `hh` texture, row-major: a write
/// at the constant level 0, which the compiler lowers (it does not lower `tex_write_2d`'s level taken from the query
/// yet, `[compiler:texture-write-level]`).
fn write_all_records(h: &Harness, t: Handle, (w, hh): (u64, u64), values: &[[u32; 4]]) -> Vec<Command> {
    assert_eq!(values.len() as u64, w * hh);
    let p = h.pipeline("fmt_write_uint", None, 0);
    let bytes: Vec<u8> = values.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    let (vb, _) = h.buffer_with(&bytes);
    vec![
        Command::ComputeSetComputePipelineState(ComputeSetComputePipelineState { state: Some(p) }),
        set_texture(t),
        set_buffer(vb, 0),
        Command::ComputeDispatchThreadgroups(ComputeDispatchThreadgroups {
            threadgroups_per_grid: size(w, hh, 1),
            threads_per_threadgroup: size(1, 1, 1),
        }),
    ]
}

/// The uint4 texels of RGBA8 bytes.
fn rgba8_texels(bytes: &[u8]) -> Vec<[u32; 4]> {
    bytes.chunks(4).map(|c| [u32::from(c[0]), u32::from(c[1]), u32::from(c[2]), u32::from(c[3])]).collect()
}

/// Reads of an R32Uint texture: (value, 0, 0, 1) (Textures.m, MSL 4.1 section 6.13).
fn texel(value: u32) -> [u32; 4] {
    [value, 0, 0, 1]
}

fn texel_bytes(t: &[[u32; 4]]) -> Vec<u8> {
    t.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
}

struct Tally {
    host: String,
    count: usize,
}

impl Tally {
    fn exact(&mut self, g: &Golden, name: &str, got: &[u8]) {
        assert_eq!(g.rule(name), Some("exact"), "{}: {name} is not an exact record", g.case);
        assert_eq!(got, g.bytes(name).unwrap().as_slice(), "{}: {}: {name}", self.host, g.case);
        self.count += 1;
    }

    fn exact_bool(&mut self, g: &Golden, name: &str, got: bool) {
        assert_eq!(g.rule(name), Some("exact"), "{}: {name}", g.case);
        assert_eq!(Some(got), g.value_bool(name), "{}: {}: {name}", self.host, g.case);
        self.count += 1;
    }
}

fn r32_descriptor(kind: u64, (w, hh): (u64, u64), levels: u64, array_length: u64, storage: u64) -> TextureDescriptor {
    texture_descriptor(
        kind,
        fmt("R32Uint"),
        (w, hh, 1),
        array_length,
        levels,
        storage,
        SHADER_READ | SHADER_WRITE | PIXEL_FORMAT_VIEW,
        1,
    )
}

/// texture.view-levels-slices: level and slice ranges, type-changing views and a view of a view, read by kernels.
#[test]
fn levels_and_slices() {
    let h = Harness::new();
    let g = Golden::load("texture.view-levels-slices");
    let mut t = Tally { host: h.name(), count: 0 };

    let d = r32_descriptor(T2D_ARRAY, (8, 8), 3, 4, PRIVATE);
    let parent = h.texture(d.clone());
    upload_all(&h, parent, &d);

    // Levels [1, 3) and slices [1, 3) as a 2D array.
    let range_view = view(&h, parent, fmt("R32Uint"), T2D_ARRAY, (1, 2), (1, 2)).unwrap();
    let mut q = Vec::new();
    let mut expected = Vec::new();
    for s in 0..2u32 {
        for l in 0..2u32 {
            for y in 0..mip(4, l as u64) as u32 {
                for x in 0..mip(4, l as u64) as u32 {
                    q.push(query(x, y, s, l));
                    expected.push(texel(tex_value(1 + l as u64, 1 + s as u64, y as u64, x as u64)));
                }
            }
        }
    }
    let inside = q.len();
    // Slice 2 and level 2 of the view are outside it, as is x = 4 of its level 0.
    for (x, y, s, l) in [(0, 0, 2, 0), (0, 0, 0, 2), (4, 0, 0, 0)] {
        q.push(query(x, y, s, l));
        expected.push([0, 0, 0, 1]);
    }
    t.exact(&g, "range-view.reads.queries", &query_bytes(&q));
    let reads = run_reads(&h, "tex_read_2d_array", range_view, &q);
    assert_eq!(&reads[..inside * 16], &texel_bytes(&expected[..inside])[..], "{}: in-range reads", h.name());
    let golden = g.bytes("range-view.reads").unwrap();
    if h.compiled.borrow().contains("tex_read_2d_array") {
        // The compiler bounds reads by the bound view's own dimensions, levels and layers (MSL 4.1 section 6.13), so the
        // reads outside the view are Metal's (0, 0, 0, 1).
        t.exact(&g, "range-view.reads", &reads);
        t.exact_bool(&g, "range-view.reads.match-specification", reads == texel_bytes(&expected));
    } else {
        // The hand translation reads with textureLoad, whose out-of-range result WebGPU leaves to the implementation:
        // only the in-range reads are Metal's.
        assert_eq!(&reads[..inside * 16], &golden[..inside * 16], "{}: range-view.reads", h.name());
        eprintln!(
            "{}: range-view.reads: the hand translation's out-of-range reads: {:x?}",
            h.name(),
            &reads[inside * 16..]
        );
    }

    // A write through a view of two levels: a WebGPU storage binding is one level, so it is refused. The case's write
    // (texel (3, 2) of the view's level 0 and slice 1: the parent's level 1 and slice 2) is made through a view of that
    // one level and slice, and the parent's level reads back as Metal's.
    let wq = [query(3, 2, 1, 0)];
    let f = {
        let e = h.handle();
        let mut v = vec![Command::BeginCompute(BeginCompute { encoder: e })];
        v.extend(write_records(&h, "tex_write_2d_array", range_view, &wq, &[[0xfeed_f00d, 0, 0, 0]]));
        v.push(Command::EncoderEndEncoding(EncoderEndEncoding {}));
        h.run(&v).0
    };
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("mipmapped texture bound for writing"), "{f:?}");
    let one = view(&h, parent, fmt("R32Uint"), T2D, (1, 1), (2, 1)).unwrap();
    h.compute(write_records(&h, "tex_write_2d", one, &[query(3, 2, 0, 0)], &[[0xfeed_f00d, 0, 0, 0]]));
    t.exact(&g, "parent.level1.slice2.after-view-write", &h.read_texture(parent, &d, 1, 2, 16));

    // Slice 3 as a 2D texture with all levels.
    let slice_view = view(&h, parent, fmt("R32Uint"), T2D, (0, 3), (3, 1)).unwrap();
    let q: Vec<_> = (0..3).map(|l| query(1, 1, 0, l)).collect();
    let expected: Vec<_> = (0..3).map(|l| texel(tex_value(l, 3, 1, 1))).collect();
    t.exact(&g, "slice-view.reads.queries", &query_bytes(&q));
    let reads = run_reads(&h, "tex_read_2d", slice_view, &q);
    t.exact(&g, "slice-view.reads", &reads);
    t.exact_bool(&g, "slice-view.reads.match-specification", reads == texel_bytes(&expected));

    // A view of the range view: its level 1 and slice 1, the parent's level 2 and slice 2.
    let nested = view(&h, range_view, fmt("R32Uint"), T2D, (1, 1), (1, 1)).unwrap();
    let q = [query(1, 0, 0, 0)];
    t.exact(&g, "nested-view.reads.queries", &query_bytes(&q));
    let reads = run_reads(&h, "tex_read_2d", nested, &q);
    t.exact(&g, "nested-view.reads", &reads);
    t.exact_bool(&g, "nested-view.reads.match-specification", reads == texel_bytes(&[texel(tex_value(2, 2, 0, 1))]));

    // A cube as a six-slice 2D array and one face as 2D.
    let cd = r32_descriptor(TCUBE, (4, 4), 2, 1, PRIVATE);
    let cube = h.texture(cd.clone());
    upload_all(&h, cube, &cd);
    let faces = view(&h, cube, fmt("R32Uint"), T2D_ARRAY, (0, 2), (0, 6)).unwrap();
    let face = view(&h, cube, fmt("R32Uint"), T2D, (0, 1), (3, 1)).unwrap();
    let mut q = Vec::new();
    let mut expected = Vec::new();
    for f in 0..6u32 {
        for l in 0..2u32 {
            q.push(query(1, 0, f, l));
            expected.push(texel(tex_value(l as u64, f as u64, 0, 1)));
        }
    }
    t.exact(&g, "cube-as-array.reads.queries", &query_bytes(&q));
    let reads = run_reads(&h, "tex_read_2d_array", faces, &q);
    t.exact(&g, "cube-as-array.reads", &reads);
    t.exact_bool(&g, "cube-as-array.reads.match-specification", reads == texel_bytes(&expected));
    let q = [query(2, 3, 0, 0)];
    t.exact(&g, "face-as-2d.reads.queries", &query_bytes(&q));
    let reads = run_reads(&h, "tex_read_2d", face, &q);
    t.exact(&g, "face-as-2d.reads", &reads);
    t.exact_bool(&g, "face-as-2d.reads.match-specification", reads == texel_bytes(&[texel(tex_value(0, 3, 3, 2))]));

    // A six-slice 2D array as a cube.
    let ad = r32_descriptor(T2D_ARRAY, (2, 2), 1, 6, PRIVATE);
    let array = h.texture(ad.clone());
    upload_all(&h, array, &ad);
    let as_cube = view(&h, array, fmt("R32Uint"), TCUBE, (0, 1), (0, 6)).unwrap();
    let q: Vec<_> = (0..6).map(|f| query(f % 2, f / 4, f, 0)).collect();
    let expected: Vec<_> = (0..6u64).map(|f| texel(tex_value(0, f, f / 4, f % 2))).collect();
    t.exact(&g, "array-as-cube.reads.queries", &query_bytes(&q));
    let reads = run_reads(&h, "tex_read_cube", as_cube, &q);
    t.exact(&g, "array-as-cube.reads", &reads);
    t.exact_bool(&g, "array-as-cube.reads.match-specification", reads == texel_bytes(&expected));

    eprintln!(
        "{}: {}: {} exact records equal; the view properties are the guest's; the write is made through a one-level \
         view of the subresource the case writes",
        h.name(),
        g.case,
        t.count
    );
    h.assert_no_errors();
}

/// The `TexRGBA8Pattern` bytes of Textures.m: `WMPattern(64, 71)` with the corner codes 0 and 255 in texel 0.
fn rgba8_pattern() -> Vec<u8> {
    let mut p = pattern(64, 71);
    p[..6].copy_from_slice(&[0, 255, 0, 255, 255, 0]);
    p
}

/// `TexRecordUnormReads` of Textures.m: every value within 1.5 ulp of code / 255, and the (index, bits) pairs of the
/// corner codes 0 and 255, which are exact.
fn unorm_corner_codes(h: &Harness, codes: &[u8], reads: &[u8]) -> Vec<u8> {
    let values: Vec<f32> = reads.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut corners = Vec::new();
    for (i, (&v, &c)) in values.iter().zip(codes).enumerate() {
        let ideal = f64::from(c) / 255.0;
        let ulp = f64::from(f32::from_bits((ideal as f32).to_bits() + 1) - ideal as f32);
        assert!((f64::from(v) - ideal).abs() <= 1.5 * ulp, "{}: texel component {i}: {v} for code {c}", h.name());
        if c == 0 || c == 255 {
            corners.extend((i as u32).to_le_bytes());
            corners.extend(v.to_bits().to_le_bytes());
        }
    }
    corners
}

/// texture.view-format-reinterpret: RGBA8Unorm read and written through an RGBA8Uint view, R32Float written through an
/// R32Uint view, and an sRGB texture through its linear and RGBA8Uint views, with every `exact` record of the case's
/// reads and writes compared byte for byte. The uint views keep their texels in the texture's aliases.
#[test]
fn format_views() {
    let h = Harness::new();
    let g = Golden::load("texture.view-format-reinterpret");
    let mut t = Tally { host: h.name(), count: 0 };
    let codes = rgba8_pattern();
    let all: Vec<_> = (0..16).map(|i| query(i % 4, i / 4, 0, 0)).collect();
    let usage = SHADER_READ | SHADER_WRITE | PIXEL_FORMAT_VIEW;

    // RGBA8Unorm viewed as RGBA8Uint.
    let ud = texture_descriptor(T2D, fmt("RGBA8Unorm"), (4, 4, 1), 1, 1, PRIVATE, usage, 1);
    let unorm = h.texture(ud.clone());
    let (src, _) = h.buffer_with(&codes);
    h.blit(vec![copy_to_texture(src, 16, (4, 4), unorm, 0, 0)]);
    let uint_view = format_view(&h, unorm, fmt("RGBA8Uint")).unwrap();
    t.exact(&g, "uint-view.reads.queries", &query_bytes(&all));
    let reads = run_reads(&h, "tex_read_2d", uint_view, &all);
    t.exact(&g, "uint-view.reads", &reads);
    let expected = rgba8_texels(&codes);
    t.exact_bool(&g, "uint-view.reads.match-specification", reads == texel_bytes(&expected));
    let floats = run_reads(&h, "tex_read_2d_float", unorm, &all);
    t.exact(&g, "unorm.reads.corner-codes", &unorm_corner_codes(&h, &codes, &floats));
    // Writes through the uint view saturate to 8 bits and are visible through the parent. The case writes texels (1, 0)
    // and (2, 3) with tex_write_2d; here every texel is written, the others with the codes they hold.
    let mut values = rgba8_texels(&codes);
    values[1] = [1, 128, 255, 256];
    values[14] = [1000, 0, 77, 0xffff_ffff];
    h.compute(write_all_records(&h, uint_view, (4, 4), &values));
    t.exact(&g, "unorm.after-uint-writes", &h.read_texture(unorm, &ud, 0, 0, 16));

    // R32Float viewed as R32Uint: bit patterns written through the uint view are stored unchanged, denormals and NaNs
    // included; float reads keep normal values.
    let fd = texture_descriptor(T2D, fmt("R32Float"), (4, 2, 1), 1, 1, PRIVATE, usage, 1);
    let floats = h.texture(fd.clone());
    let bits_view = format_view(&h, floats, fmt("R32Uint")).unwrap();
    let bits = [0x3f80_0000u32, 0xc2f6_e979, 0, 0x8000_0000, 0x7f7f_ffff, 0x7f80_0000, 1, 0x7fc0_0001];
    let fq: Vec<_> = (0..8).map(|i| query(i % 4, i / 4, 0, 0)).collect();
    let fv: Vec<[u32; 4]> = bits.iter().map(|&b| [b, 0, 0, 0]).collect();
    h.compute(write_all_records(&h, bits_view, (4, 2), &fv));
    t.exact(&g, "r32float.stored-bits", &h.read_texture(floats, &fd, 0, 0, 16));
    let reads = run_reads(&h, "tex_read_2d_float", floats, &fq);
    let preserved: Vec<u8> = reads.chunks(16).take(6).flat_map(|c| c[..4].to_vec()).collect();
    t.exact(&g, "r32float.reads.preserved", &preserved);

    // An sRGB texture with a linear view (a WebGPU view in the sibling format) and an RGBA8Uint view.
    let srgb = h.texture(texture_descriptor(
        T2D,
        fmt("RGBA8Unorm_sRGB"),
        (4, 4, 1),
        1,
        1,
        PRIVATE,
        SHADER_READ | PIXEL_FORMAT_VIEW,
        1,
    ));
    let (src, _) = h.buffer_with(&codes);
    h.blit(vec![copy_to_texture(src, 16, (4, 4), srgb, 0, 0)]);
    let linear = format_view(&h, srgb, fmt("RGBA8Unorm")).unwrap();
    let raw = format_view(&h, srgb, fmt("RGBA8Uint")).unwrap();
    let reads = run_reads(&h, "tex_read_2d_float", linear, &all);
    t.exact(&g, "srgb.linear-view.reads.corner-codes", &unorm_corner_codes(&h, &codes, &reads));
    t.exact(&g, "srgb.uint-view.reads.queries", &query_bytes(&all));
    let reads = run_reads(&h, "tex_read_2d", raw, &all);
    t.exact(&g, "srgb.uint-view.reads", &reads);
    t.exact_bool(&g, "srgb.uint-view.reads.match-specification", reads == texel_bytes(&expected));

    for k in ["tex_read_2d", "tex_read_2d_float", "fmt_write_uint"] {
        assert!(h.compiled.borrow().contains(k), "{}: {k} is not a compiled kernel", h.name());
    }
    eprintln!(
        "{}: {}: {} exact records equal; the writes through the uint views are made by fmt_write_uint",
        h.name(),
        g.case,
        t.count
    );
    h.assert_no_errors();
}

/// A texture and its view in another format kept equal through each kind of write: a CPU upload, a blit of part of a
/// subresource, a blit between textures and mipmap generation through the view; and the pass that would use both in
/// one WebGPU pass with a write, refused.
#[test]
fn alias_coherence() {
    let h = Harness::new();
    let usage = SHADER_READ | SHADER_WRITE | PIXEL_FORMAT_VIEW;
    let d = texture_descriptor(T2D, fmt("RGBA8Unorm"), (4, 4, 1), 1, 1, SHARED, usage, 1);
    let unorm = h.texture(d.clone());
    let uint_view = format_view(&h, unorm, fmt("RGBA8Uint")).unwrap();
    let all: Vec<_> = (0..16).map(|i| query(i % 4, i / 4, 0, 0)).collect();
    let texels = |bytes: &[u8]| texel_bytes(&rgba8_texels(bytes));

    // Written through the view, then replaceRegion of part of the texture: the CPU upload writes the alias the view
    // left current, and getBytes reads it.
    let first = pattern(64, 5);
    let (src, _) = h.buffer_with(&first);
    h.blit(vec![copy_to_texture(src, 16, (4, 4), unorm, 0, 0)]);
    let mut expect = first.clone();
    expect[..4].copy_from_slice(&[9, 8, 7, 6]);
    h.compute(write_all_records(&h, uint_view, (4, 4), &rgba8_texels(&expect)));
    let row = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let addr = h.alloc(8);
    h.guest.cpu_write(addr, &row);
    h.ok(Request::ReplaceRegion(ReplaceRegion {
        texture: unorm,
        region: region((2, 1, 0), (2, 1, 1)),
        level: 0,
        pixel_bytes: GuestAddr(addr),
        bytes_per_row: 8,
    }));
    expect[24..32].copy_from_slice(&row);
    let dst = h.alloc(64);
    h.ok(Request::GetBytes(GetBytes {
        texture: unorm,
        pixel_bytes: GuestAddr(dst),
        bytes_per_row: 16,
        region: region((0, 0, 0), (4, 4, 1)),
        level: 0,
    }));
    assert_eq!(h.read(dst, 64), expect, "{}: getBytes after a view write and replaceRegion", h.name());
    // The texture's own texels after the alias's: a float read sees both writes.
    let floats = run_reads(&h, "tex_read_2d_float", unorm, &all);
    let codes: Vec<u8> =
        floats.chunks(4).map(|c| (f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * 255.0).round() as u8).collect();
    assert_eq!(codes, expect, "{}: float reads of the texture after the view's write", h.name());

    // A blit of part of the texture after a write through the view: the stored texture takes the alias's texels
    // first, so the texels around the region keep the view's write.
    expect[60..64].copy_from_slice(&[40, 41, 42, 43]);
    h.compute(write_all_records(&h, uint_view, (4, 4), &rgba8_texels(&expect)));
    let (b, _) = h.buffer_with(&[50, 51, 52, 53]);
    h.blit(vec![Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
        source_buffer: Some(b),
        source_offset: 0,
        source_bytes_per_row: 4,
        source_bytes_per_image: 0,
        source_size: size(1, 1, 1),
        destination_texture: Some(unorm),
        destination_slice: 0,
        destination_level: 0,
        destination_origin: origin(1, 1, 0),
    })]);
    expect[20..24].copy_from_slice(&[50, 51, 52, 53]);
    assert_eq!(h.read_texture(unorm, &d, 0, 0, 16), expect, "{}: a partial blit after a view write", h.name());
    assert_eq!(run_reads(&h, "tex_read_2d", uint_view, &all), texels(&expect), "{}: the view after the blit", h.name());

    // A blit from another texture into the texture leaves the alias stale; the view reads the copy.
    let other = h.texture(d.clone());
    let second = pattern(64, 6);
    let (src, _) = h.buffer_with(&second);
    h.blit(vec![copy_to_texture(src, 16, (4, 4), other, 0, 0)]);
    h.blit(vec![Command::BlitCopyTextureToTexture(BlitCopyTextureToTexture {
        source_texture: Some(other),
        source_slice: 0,
        source_level: 0,
        source_origin: origin(0, 0, 0),
        source_size: size(4, 4, 1),
        destination_texture: Some(unorm),
        destination_slice: 0,
        destination_level: 0,
        destination_origin: origin(0, 0, 0),
    })]);
    assert_eq!(run_reads(&h, "tex_read_2d", uint_view, &all), texels(&second), "{}: the view after a copy", h.name());

    // One compute pass that writes through the view and reads through the texture is refused.
    let mut records = write_all_records(&h, uint_view, (4, 4), &[[1, 1, 1, 1]; 16]);
    let (read, _) = read_records(&h, "tex_read_2d_float", unorm, &all);
    records.extend(read);
    let f = h.compute_failure(records);
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("through two pixel formats"), "{f:?}");

    // Mipmaps generated through a BGRA8Unorm view of an RGBA8Unorm texture: the levels are made in the alias and read
    // through the texture.
    let md = texture_descriptor(
        T2D,
        fmt("RGBA8Unorm"),
        (4, 4, 1),
        1,
        3,
        PRIVATE,
        SHADER_READ | RENDER_TARGET | PIXEL_FORMAT_VIEW,
        1,
    );
    let mipped = h.texture(md.clone());
    let (src, _) = h.buffer_with(&[0x10, 0x80, 0xf0, 0xff].repeat(16));
    h.blit(vec![copy_to_texture(src, 16, (4, 4), mipped, 0, 0)]);
    let bgra = view(&h, mipped, fmt("BGRA8Unorm"), T2D, (0, 3), (0, 1)).unwrap();
    h.blit(vec![Command::BlitGenerateMipmapsForTexture(BlitGenerateMipmapsForTexture { texture: Some(bgra) })]);
    for level in 1..3 {
        let w = mip(4, level);
        let got = h.read_texture(mipped, &md, level, 0, w * 4);
        assert_eq!(got, [0x10, 0x80, 0xf0, 0xff].repeat((w * w) as usize), "{}: level {level}", h.name());
    }
    h.assert_no_errors();
}

/// MTLSamplerDescriptor's defaults (MTLSampler.h): nearest filters, not mipmapped, clamp to edge, normalized
/// coordinates, LOD clamps [0, FLT_MAX], anisotropy 1, compare never.
fn default_sampler(h: &Harness) -> Handle {
    let sampler = h.handle();
    h.ok(Request::CreateSamplerState(CreateSamplerState {
        sampler,
        descriptor: SamplerDescriptor {
            border_color: 0,
            border_color_spi: 0,
            compare_function: 0,
            custom_border_color_value_0: 0,
            custom_border_color_value_1: 0,
            custom_border_color_value_2: 0,
            custom_border_color_value_3: 0,
            force_resource_index: false,
            force_seams_on_cubemap_filtering: false,
            label: None,
            lod_average: false,
            lod_bias: 0.0,
            lod_max_clamp: f32::MAX,
            lod_min_clamp: 0.0,
            mag_filter: 0,
            max_anisotropy: 1,
            min_filter: 0,
            mip_filter: 0,
            normalized_coordinates: true,
            pixel_format: 0,
            r_address_mode: 0,
            resource_index: 0,
            s_address_mode: 0,
            support_argument_buffers: false,
            t_address_mode: 0,
        },
    }));
    sampler
}

/// texture.view-swizzle: a swizzled view, a descriptor swizzle read and sampled, and a swizzled one-component view,
/// with every `exact` record of the reads compared byte for byte. The swizzle is the WebGPU view's where the host has
/// texture-component-swizzle, else the compiled kernel's.
#[test]
fn swizzles() {
    let h = Harness::new();
    let g = Golden::load("texture.view-swizzle");
    let mut t = Tally { host: h.name(), count: 0 };
    let codes = rgba8_pattern();
    let all: Vec<_> = (0..16).map(|i| query(i % 4, i / 4, 0, 0)).collect();
    let usage = SHADER_READ | PIXEL_FORMAT_VIEW;

    // A view with swizzle (blue, zero, one, red) of an RGBA8Uint texture.
    let mut desc = texture_descriptor(T2D, fmt("RGBA8Uint"), (4, 4, 1), 1, 1, PRIVATE, usage, 1);
    let base = h.texture(desc.clone());
    let (src, _) = h.buffer_with(&codes);
    h.blit(vec![copy_to_texture(src, 16, (4, 4), base, 0, 0)]);
    let v = h.handle();
    h.ok(Request::CreateTextureViewLevelsSlicesSwizzle(CreateTextureViewLevelsSlicesSwizzle {
        texture: base,
        view: v,
        pixel_format: fmt("RGBA8Uint"),
        texture_type: T2D,
        level_range: range(0, 1),
        slice_range: range(0, 1),
        swizzle: TextureSwizzleChannels { red: 4, green: 0, blue: 1, alpha: 2 },
    }));
    t.exact(&g, "view.reads.queries", &query_bytes(&all));
    t.exact(&g, "view.reads", &run_reads(&h, "tex_read_2d", v, &all));

    // A descriptor swizzle (alpha, red, green, blue), read and nearest-sampled at texel centres.
    desc.swizzle = TextureSwizzleChannels { red: 5, green: 2, blue: 3, alpha: 4 };
    desc.usage = SHADER_READ;
    desc.texture_usage = SHADER_READ;
    let swizzled = h.texture(desc);
    let (src, _) = h.buffer_with(&codes);
    h.blit(vec![copy_to_texture(src, 16, (4, 4), swizzled, 0, 0)]);
    t.exact(&g, "descriptor.reads", &run_reads(&h, "tex_read_2d", swizzled, &all));
    let centres: Vec<u8> = (0..16)
        .flat_map(|i| [((i % 4) as f32 + 0.5) / 4.0, ((i / 4) as f32 + 0.5) / 4.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    let sampler = default_sampler(&h);
    let p = h.pipeline("tex_sample_2d_uint", None, 0);
    let (coords, _) = h.buffer_with(&centres);
    let (out, addr) = h.buffer(16 * 16);
    h.compute(vec![
        Command::ComputeSetComputePipelineState(ComputeSetComputePipelineState { state: Some(p) }),
        set_texture(swizzled),
        Command::ComputeSetSamplerState(ComputeSetSamplerState { sampler: Some(sampler), index: 0 }),
        set_buffer(coords, 0),
        set_buffer(out, 1),
        Command::ComputeDispatchThreadgroups(ComputeDispatchThreadgroups {
            threadgroups_per_grid: size(16, 1, 1),
            threads_per_threadgroup: size(1, 1, 1),
        }),
    ]);
    t.exact(&g, "descriptor.samples", &h.read(addr, 16 * 16));

    // An R8Uint view with swizzle (alpha, green, red, blue): the missing components read as 0 and alpha as 1.
    let single = h.texture(texture_descriptor(T2D, fmt("R8Uint"), (4, 4, 1), 1, 1, PRIVATE, usage, 1));
    let (src, _) = h.buffer_with(&pattern(16, 72));
    h.blit(vec![copy_to_texture(src, 4, (4, 4), single, 0, 0)]);
    let red_view = h.handle();
    h.ok(Request::CreateTextureViewLevelsSlicesSwizzle(CreateTextureViewLevelsSlicesSwizzle {
        texture: single,
        view: red_view,
        pixel_format: fmt("R8Uint"),
        texture_type: T2D,
        level_range: range(0, 1),
        slice_range: range(0, 1),
        swizzle: TextureSwizzleChannels { red: 5, green: 3, blue: 2, alpha: 4 },
    }));
    t.exact(&g, "r8uint-view.reads", &run_reads(&h, "tex_read_2d", red_view, &all));

    // A hand translation would not apply the swizzle: the reads come from compiled kernels.
    for k in ["tex_read_2d", "tex_sample_2d_uint"] {
        assert!(h.compiled.borrow().contains(k), "{}: {k} is not a compiled kernel", h.name());
    }
    eprintln!(
        "{}: {}: {} exact records equal; the swizzle is applied by the {}",
        h.name(),
        g.case,
        t.count,
        if h.memory.features().has(Feature::TextureComponentSwizzle) { "WebGPU view" } else { "compiled kernel" }
    );
    h.assert_no_errors();
}

/// A view of each plan, each request form, a view of a view, and a view that outlives its parent's handle.
#[test]
fn views_of_each_plan() {
    let h = Harness::new();
    let d = r32_descriptor(T2D_ARRAY, (8, 8), 3, 4, SHARED);
    let parent = h.texture(d.clone());

    // Same: each of the six requests.
    let v = h.handle();
    h.ok(Request::CreateTextureView(CreateTextureView { texture: parent, view: v, pixel_format: fmt("R32Uint") }));
    h.ok(Request::CreateTextureViewResourceIndex(CreateTextureViewResourceIndex {
        texture: parent,
        view: h.handle(),
        pixel_format: fmt("R32Uint"),
        resource_index: 5,
    }));
    h.ok(Request::CreateTextureViewLevelsSlices(CreateTextureViewLevelsSlices {
        texture: parent,
        view: h.handle(),
        pixel_format: fmt("R32Uint"),
        texture_type: T2D,
        level_range: range(2, 1),
        slice_range: range(3, 1),
    }));
    let identity = TextureSwizzleChannels { red: 2, green: 3, blue: 4, alpha: 5 };
    h.ok(Request::CreateTextureViewLevelsSlicesSwizzle(CreateTextureViewLevelsSlicesSwizzle {
        texture: parent,
        view: h.handle(),
        pixel_format: fmt("R32Uint"),
        texture_type: T2D_ARRAY,
        level_range: range(0, 3),
        slice_range: range(0, 4),
        swizzle: identity.clone(),
    }));
    h.ok(Request::CreateTextureViewLevelsSlicesResourceIndex(CreateTextureViewLevelsSlicesResourceIndex {
        texture: parent,
        view: h.handle(),
        pixel_format: fmt("R32Uint"),
        texture_type: T2D_ARRAY,
        levels: range(1, 2),
        slices: range(1, 3),
        resource_index: 0,
    }));
    h.ok(Request::CreateTextureViewLevelsSlicesSwizzleResourceIndex(
        CreateTextureViewLevelsSlicesSwizzleResourceIndex {
            texture: parent,
            view: h.handle(),
            pixel_format: fmt("R32Uint"),
            texture_type: T2D,
            levels: range(0, 1),
            slices: range(0, 1),
            swizzle: identity,
            resource_index: 9,
        },
    ));
    // A view of a view.
    view(&h, v, fmt("R32Uint"), T2D, (1, 1), (2, 1)).unwrap();

    // ViewFormat: an sRGB view of a texture without MTLTextureUsagePixelFormatView, which Metal makes
    // (texture.view-srgb-without-usage).
    let rgba = h.texture(texture_descriptor(T2D, fmt("RGBA8Unorm"), (4, 4, 1), 1, 1, SHARED, SHADER_READ, 1));
    format_view(&h, rgba, fmt("RGBA8Unorm_sRGB")).unwrap();
    // Copy: views in other formats of a texture without the usage, which the A10X driver makes as well and Metal
    // returns without its debug layer (texture.error-view-without-usage and
    // validation.texture-view-without-format-view-usage, release).
    let uint = format_view(&h, rgba, fmt("RGBA8Uint")).unwrap();
    format_view(&h, rgba, fmt("R32Float")).unwrap();
    let codes = pattern(64, 9);
    let addr = h.alloc(64);
    h.guest.cpu_write(addr, &codes);
    h.ok(Request::ReplaceRegion(ReplaceRegion {
        texture: rgba,
        region: region((0, 0, 0), (4, 4, 1)),
        level: 0,
        pixel_bytes: GuestAddr(addr),
        bytes_per_row: 16,
    }));
    let all: Vec<_> = (0..16).map(|i| query(i % 4, i / 4, 0, 0)).collect();
    assert_eq!(run_reads(&h, "tex_read_2d", uint, &all), texel_bytes(&rgba8_texels(&codes)), "{}", h.name());
    // SharedStorage: R8Unorm as R8Unorm_sRGB (both r8unorm), B5G6R5Unorm as R16Uint (both r16uint).
    let r8 = h.texture(texture_descriptor(T2D, fmt("R8Unorm"), (4, 4, 1), 1, 1, SHARED, SHADER_READ, 1));
    format_view(&h, r8, fmt("R8Unorm_sRGB")).unwrap();
    let packed = h.texture(texture_descriptor(
        T2D,
        fmt("B5G6R5Unorm"),
        (4, 4, 1),
        1,
        1,
        SHARED,
        SHADER_READ | PIXEL_FORMAT_VIEW,
        1,
    ));
    format_view(&h, packed, fmt("R16Uint")).unwrap();
    // StencilAspect: X32_Stencil8 of Depth32Float_Stencil8, where the host has the format.
    if h.memory.features().has(Feature::Depth32floatStencil8) {
        let ds = h.texture(texture_descriptor(
            T2D,
            fmt("Depth32Float_Stencil8"),
            (8, 4, 1),
            1,
            1,
            PRIVATE,
            SHADER_READ | PIXEL_FORMAT_VIEW,
            1,
        ));
        let stencil = format_view(&h, ds, fmt("X32_Stencil8")).unwrap();
        // Its bytes are one aspect of the texture's: CPU transfers through it are refused.
        let addr = h.alloc(64);
        let f = h.status(Request::GetBytes(GetBytes {
            texture: stencil,
            pixel_bytes: GuestAddr(addr),
            bytes_per_row: 8,
            region: region((0, 0, 0), (8, 4, 1)),
            level: 0,
        }));
        assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    }

    // The view keeps its parent's storage when the parent's handle is released.
    let bytes: Vec<u8> = (0..16u32).flat_map(|i| (0xc000_0000 | i).to_le_bytes()).collect();
    let src = h.alloc(64);
    h.guest.cpu_write(src, &bytes);
    let level2 = view(&h, parent, fmt("R32Uint"), T2D, (2, 1), (1, 1)).unwrap();
    h.ok(Request::ReplaceRegion(ReplaceRegion {
        texture: level2,
        region: region((0, 0, 0), (2, 2, 1)),
        level: 0,
        pixel_bytes: GuestAddr(src),
        bytes_per_row: 8,
    }));
    h.ok(Request::Release(Release { object: parent }));
    let dst = h.alloc(64);
    h.ok(Request::GetBytes(GetBytes {
        texture: level2,
        pixel_bytes: GuestAddr(dst),
        bytes_per_row: 8,
        region: region((0, 0, 0), (2, 2, 1)),
        level: 0,
    }));
    assert_eq!(h.read(dst, 16), bytes[..16]);
    h.assert_no_errors();
}

/// `replaceRegion:` and `getBytes:` through views: the parent's subresources at the view's bases.
#[test]
fn cpu_transfers_through_views() {
    let h = Harness::new();
    let d = r32_descriptor(T2D_ARRAY, (8, 8), 3, 4, SHARED);
    let parent = h.texture(d.clone());
    // Every level and slice of the parent through replaceRegion of the parent.
    for level in 0..3 {
        let w = mip(8, level);
        for slice in 0..4 {
            let bytes: Vec<u8> = (0..w)
                .flat_map(|y| (0..w).map(move |x| (y, x)))
                .flat_map(|(y, x)| tex_value(level, slice, y, x).to_le_bytes())
                .collect();
            let addr = h.alloc(bytes.len() as u64);
            h.guest.cpu_write(addr, &bytes);
            h.ok(Request::ReplaceRegionSlice(ReplaceRegionSlice {
                texture: parent,
                region: region((0, 0, 0), (w, w, 1)),
                level,
                slice,
                pixel_bytes: GuestAddr(addr),
                bytes_per_row: w * 4,
                bytes_per_image: 0,
            }));
        }
    }
    let get = |t: Handle, level: u64, slice: u64, w: u64| {
        let addr = h.alloc(w * w * 4);
        h.ok(Request::GetBytesSlice(GetBytesSlice {
            texture: t,
            pixel_bytes: GuestAddr(addr),
            bytes_per_row: w * 4,
            bytes_per_image: 0,
            region: region((0, 0, 0), (w, w, 1)),
            level,
            slice,
        }));
        h.read(addr, (w * w * 4) as usize)
    };
    let range_view = view(&h, parent, fmt("R32Uint"), T2D_ARRAY, (1, 2), (1, 2)).unwrap();
    // The view's level 1 and slice 1 are the parent's level 2 and slice 2.
    let expect: Vec<u8> = (0..2u64)
        .flat_map(|y| (0..2u64).map(move |x| (y, x)))
        .flat_map(|(y, x)| tex_value(2, 2, y, x).to_le_bytes())
        .collect();
    assert_eq!(get(range_view, 1, 1, 2), expect);
    // A write of the view's level 0 and slice 0 lands in the parent's level 1 and slice 1.
    let bytes: Vec<u8> = (0..16u32).flat_map(|i| (0xb000_0000 | i).to_le_bytes()).collect();
    let addr = h.alloc(64);
    h.guest.cpu_write(addr, &bytes);
    h.ok(Request::ReplaceRegionSlice(ReplaceRegionSlice {
        texture: range_view,
        region: region((0, 0, 0), (2, 2, 1)),
        level: 0,
        slice: 0,
        pixel_bytes: GuestAddr(addr),
        bytes_per_row: 8,
        bytes_per_image: 0,
    }));
    let level1 = get(parent, 1, 1, 4);
    for y in 0..4usize {
        for x in 0..4usize {
            let got = u32::from_le_bytes(level1[(y * 4 + x) * 4..][..4].try_into().unwrap());
            let want =
                if x < 2 && y < 2 { 0xb000_0000 | (y * 2 + x) as u32 } else { tex_value(1, 1, y as u64, x as u64) };
            assert_eq!(got, want, "{}: parent level 1 slice 1 texel ({x}, {y})", h.name());
        }
    }
    // Metal's rules apply to the view: its level 2 and its slice 2 do not exist, though the parent's do.
    for (level, slice) in [(2, 0), (0, 2)] {
        let f = h.status(Request::GetBytesSlice(GetBytesSlice {
            texture: range_view,
            pixel_bytes: GuestAddr(addr),
            bytes_per_row: 8,
            bytes_per_image: 0,
            region: region((0, 0, 0), (1, 1, 1)),
            level,
            slice,
        }));
        assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
        assert!(f.message.contains("behavior Metal does not define"), "{f:?}");
    }

    // An sRGB view's bytes are the texture's.
    let rgba = h.texture(texture_descriptor(T2D, fmt("RGBA8Unorm"), (4, 4, 1), 1, 1, SHARED, SHADER_READ, 1));
    let srgb = format_view(&h, rgba, fmt("RGBA8Unorm_sRGB")).unwrap();
    let codes = rgba8_pattern();
    let src = h.alloc(64);
    h.guest.cpu_write(src, &codes);
    h.ok(Request::ReplaceRegion(ReplaceRegion {
        texture: srgb,
        region: region((0, 0, 0), (4, 4, 1)),
        level: 0,
        pixel_bytes: GuestAddr(src),
        bytes_per_row: 16,
    }));
    let dst = h.alloc(64);
    h.ok(Request::GetBytes(GetBytes {
        texture: rgba,
        pixel_bytes: GuestAddr(dst),
        bytes_per_row: 16,
        region: region((0, 0, 0), (4, 4, 1)),
        level: 0,
    }));
    assert_eq!(h.read(dst, 64), codes);
    h.assert_no_errors();
}

/// Blits through views: buffer to view, view to texture, mipmaps of a view's levels, and Metal's rules on the view.
#[test]
fn blits_through_views() {
    let h = Harness::new();
    let d = r32_descriptor(T2D_ARRAY, (8, 8), 3, 4, PRIVATE);
    let parent = h.texture(d.clone());
    upload_all(&h, parent, &d);

    // A buffer into the 2D view of the parent's level 1 and slice 2.
    let slice = view(&h, parent, fmt("R32Uint"), T2D, (1, 1), (2, 1)).unwrap();
    let bytes: Vec<u8> = (0..16u32).flat_map(|i| (0xd000_0000 | i).to_le_bytes()).collect();
    let (src, _) = h.buffer_with(&bytes);
    h.blit(vec![copy_to_texture(src, 16, (4, 4), slice, 0, 0)]);
    assert_eq!(h.read_texture(parent, &d, 1, 2, 16), bytes);
    // The other slices of that level are untouched.
    let other: Vec<u8> = (0..4u64)
        .flat_map(|y| (0..4u64).map(move |x| (y, x)))
        .flat_map(|(y, x)| tex_value(1, 3, y, x).to_le_bytes())
        .collect();
    assert_eq!(h.read_texture(parent, &d, 1, 3, 16), other);

    // The view into a texture, and back out into a buffer.
    let flat_d = r32_descriptor(T2D, (4, 4), 1, 1, PRIVATE);
    let flat = h.texture(flat_d.clone());
    h.blit(vec![Command::BlitCopyTextureToTexture(BlitCopyTextureToTexture {
        source_texture: Some(slice),
        source_slice: 0,
        source_level: 0,
        source_origin: origin(0, 0, 0),
        source_size: size(4, 4, 1),
        destination_texture: Some(flat),
        destination_slice: 0,
        destination_level: 0,
        destination_origin: origin(0, 0, 0),
    })]);
    assert_eq!(h.read_texture(flat, &flat_d, 0, 0, 16), bytes);
    let view_d = TextureDescriptor {
        texture_type: T2D,
        width: 4,
        height: 4,
        array_length: 1,
        mipmap_level_count: 1,
        ..d.clone()
    };
    assert_eq!(h.read_texture(slice, &view_d, 0, 0, 16), bytes);

    // The view's levels and slices are Metal's bounds: level 1 and slice 1 of the one-level, one-slice view are not.
    for (s, l) in [(0, 1), (1, 0)] {
        let f = h.blit_failure(vec![copy_to_texture(src, 16, (1, 1), slice, s, l)]);
        assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
        assert!(f.message.contains("behavior Metal does not define"), "{f:?}");
    }

    // Mipmaps of a view's levels [1, 4) of an RGBA8Unorm texture: levels 2 and 3 are made from level 1; levels 0 and 1
    // keep their contents. Each level is one color, which the filter keeps.
    let md = texture_descriptor(T2D, fmt("RGBA8Unorm"), (8, 8, 1), 1, 4, PRIVATE, SHADER_READ, 1);
    let mipped = h.texture(md.clone());
    for (level, value) in [(0u64, 0x10u8), (1, 0x80), (2, 0x33), (3, 0x33)] {
        let w = mip(8, level);
        let (src, _) = h.buffer_with(&vec![value; (w * w * 4) as usize]);
        h.blit(vec![copy_to_texture(src, w * 4, (w, w), mipped, 0, level)]);
    }
    let lower = view(&h, mipped, fmt("RGBA8Unorm"), T2D, (1, 3), (0, 1)).unwrap();
    h.blit(vec![Command::BlitGenerateMipmapsForTexture(BlitGenerateMipmapsForTexture { texture: Some(lower) })]);
    for (level, value) in [(0u64, 0x10u8), (1, 0x80), (2, 0x80), (3, 0x80)] {
        let w = mip(8, level);
        assert_eq!(h.read_texture(mipped, &md, level, 0, w * 4), vec![value; (w * w * 4) as usize], "level {level}");
    }
    // Through an sRGB view the levels are filtered in linear space, which keeps a single color's code to within the
    // conversions' rounding.
    let srgb = view(&h, mipped, fmt("RGBA8Unorm_sRGB"), T2D, (0, 4), (0, 1)).unwrap();
    h.blit(vec![Command::BlitGenerateMipmapsForTexture(BlitGenerateMipmapsForTexture { texture: Some(srgb) })]);
    for level in 1..4 {
        let w = mip(8, level);
        let got = h.read_texture(mipped, &md, level, 0, w * 4);
        assert!(got.iter().all(|&c| c.abs_diff(0x10) <= 1), "level {level}: {got:x?}");
    }

    // A copy between an sRGB view and a texture of the linear format: Metal requires the same pixel format.
    let rgba = h.texture(texture_descriptor(T2D, fmt("RGBA8Unorm"), (8, 8, 1), 1, 1, PRIVATE, SHADER_READ, 1));
    let f = h.blit_failure(vec![Command::BlitCopyTextureToTexture(BlitCopyTextureToTexture {
        source_texture: Some(srgb),
        source_slice: 0,
        source_level: 0,
        source_origin: origin(0, 0, 0),
        source_size: size(8, 8, 1),
        destination_texture: Some(rgba),
        destination_slice: 0,
        destination_level: 0,
        destination_origin: origin(0, 0, 0),
    })]);
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("differs from destination pixel format"), "{f:?}");
    h.assert_no_errors();
}

/// The views the host refuses, and the uses of views it refuses, each with the reason.
#[test]
fn view_refusals() {
    let h = Harness::new();
    let d = r32_descriptor(T2D_ARRAY, (8, 8), 3, 4, PRIVATE);
    let parent = h.texture(d);
    let undefined = |f: Failure| {
        assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
        assert!(f.message.contains("behavior Metal does not define"), "{f:?}");
    };
    let r32 = fmt("R32Uint");
    // Levels and slices outside the parent; empty ranges.
    undefined(view(&h, parent, r32, T2D_ARRAY, (2, 2), (0, 1)).unwrap_err());
    undefined(view(&h, parent, r32, T2D_ARRAY, (0, 1), (3, 2)).unwrap_err());
    undefined(view(&h, parent, r32, T2D_ARRAY, (0, 0), (0, 1)).unwrap_err());
    // A cast the documentation does not list, and slice counts the view's type does not take.
    undefined(view(&h, parent, r32, T3D, (0, 1), (0, 1)).unwrap_err());
    undefined(view(&h, parent, r32, TCUBE, (0, 1), (0, 4)).unwrap_err());
    undefined(view(&h, parent, r32, T2D, (0, 1), (0, 2)).unwrap_err());
    let flat = h.texture(r32_descriptor(T2D, (8, 8), 1, 1, PRIVATE));
    undefined(view(&h, flat, r32, TCUBE, (0, 1), (0, 1)).unwrap_err());
    // A cube view of slices that are not square.
    let wide = h.texture(r32_descriptor(T2D_ARRAY, (8, 4), 1, 6, PRIVATE));
    let f = view(&h, wide, r32, TCUBE, (0, 1), (0, 6)).unwrap_err();
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("square"), "{f:?}");
    // A channel that is not an MTLTextureSwizzle value.
    let f = h.status(Request::CreateTextureViewLevelsSlicesSwizzle(CreateTextureViewLevelsSlicesSwizzle {
        texture: parent,
        view: h.handle(),
        pixel_format: r32,
        texture_type: T2D,
        level_range: range(0, 1),
        slice_range: range(0, 1),
        swizzle: TextureSwizzleChannels { red: 4, green: 0, blue: 6, alpha: 2 },
    }));
    undefined(f.clone());
    assert!(f.message.contains("MTLTextureSwizzle"), "{f:?}");
    // A view in another format of a multisample texture.
    let ms = h.texture(texture_descriptor(
        T2D_MS,
        fmt("RGBA8Unorm"),
        (4, 4, 1),
        1,
        1,
        PRIVATE,
        SHADER_READ | PIXEL_FORMAT_VIEW,
        4,
    ));
    let f = format_view(&h, ms, fmt("RGBA8Uint")).unwrap_err();
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("multisample"), "{f:?}");
    // Texel sizes that differ (texture.error-view-size-mismatch).
    let rgba_view = h.texture(texture_descriptor(
        T2D,
        fmt("RGBA8Unorm"),
        (4, 4, 1),
        1,
        1,
        PRIVATE,
        SHADER_READ | PIXEL_FORMAT_VIEW,
        1,
    ));
    let f = format_view(&h, rgba_view, fmt("R8Unorm")).unwrap_err();
    assert_eq!(f.status, status::HOST_UNSUPPORTED, "{f:?}");
    assert!(f.message.contains("view-size-mismatch"), "{f:?}");
    // A view of a texture over a buffer.
    let (b, _) = h.buffer(4096);
    let lt = h.handle();
    h.ok(Request::CreateBufferTexture(CreateBufferTexture {
        buffer: b,
        texture: lt,
        descriptor: texture_descriptor(T2D, fmt("RGBA8Unorm"), (16, 4, 1), 1, 1, SHARED, SHADER_READ, 1),
        offset: 0,
        bytes_per_row: 64,
    }));
    let f = format_view(&h, lt, fmt("RGBA8Unorm")).unwrap_err();
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");

    // A view of two slices where the function reads a 2D texture.
    let two = view(&h, parent, r32, T2D_ARRAY, (0, 1), (0, 2)).unwrap();
    let f = h.compute_failure(read_records(&h, "tex_read_2d", two, &[query(0, 0, 0, 0)]).0);
    undefined(f.clone());
    assert!(
        f.message.contains(
            "texture index 0 (argument t) is declared MTLTextureType2D, and the bound texture is MTLTextureType2DArray"
        ),
        "{f:?}"
    );

    // A view's purgeable state is the guest's to answer; makeAliasable is its parent's heap's.
    let v = view(&h, parent, r32, T2D, (0, 1), (0, 1)).unwrap();
    let f = h.status(Request::SetPurgeableState(SetPurgeableState { object: v, state: 1 }));
    assert_eq!(f.status, status::INVALID, "{f:?}");
    undefined(h.status(Request::MakeAliasable(MakeAliasable { resource: v })));
    let heap = h.handle();
    h.ok(Request::CreateHeap(CreateHeap {
        heap,
        descriptor: HeapDescriptor {
            cpu_cache_mode: 0,
            hazard_tracking_mode: 0,
            memory_pool_id: 0,
            pinned_gpu_address: 0,
            protection_options: 0,
            resource_options: 0x20,
            size: 64 * 1024,
            sparse_page_access_counters_enabled: false,
            sparse_page_size: 0,
            storage_mode: 2,
            r#type: 0,
        },
    }));
    let ht = h.handle();
    h.ok(Request::HeapCreateTexture(HeapCreateTexture {
        heap,
        texture: ht,
        descriptor: r32_descriptor(T2D, (8, 8), 1, 1, PRIVATE),
    }));
    let hv = format_view(&h, ht, r32).unwrap();
    h.ok(Request::MakeAliasable(MakeAliasable { resource: hv }));
    h.assert_no_errors();
}
