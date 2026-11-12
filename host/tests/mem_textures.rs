//! Texture contents on a native WebGPU implementation (`WEBGPU_NATIVE`, see `host/build.rs`): the byte-exact upload
//! and readback matrix over every pixel format the format module handles and every storage mode, compared with the
//! Metal reference outputs, and the CPU and blit transfer forms over every texture type.
//!
//! `WEBMETAL_MEM_REPORT=<file>` keeps the matrix as a table.

#![cfg(feature = "native")]

mod mem_common;

use std::fmt::Write as _;
use std::rc::Rc;

use mem_common::{Api, Golden, Gpu, Guest, TASK};
use webmetal_host::format::handling::{self, Dimension, Resolution, Unavailable};
use webmetal_host::format::pvrtc;
use webmetal_host::format::table::METAL_FORMATS;
use webmetal_host::gpu::idl::{CommandBufferDescriptor, CommandEncoderDescriptor};
use webmetal_host::gpu::{CommandEncoder as _, Device as _};
use webmetal_host::mem::buffer::{Buffer, Readbacks};
use webmetal_host::mem::layout::{Blocks, RegionArgs, RegionLayout};
use webmetal_host::mem::texture::{Prepare, Texture, TextureRegion};
use webmetal_host::mem::{MemError, Memory};
use webmetal_wire::{
    CreateBuffer, GetBytesSlice, GuestAddr, Handle, Origin, Region, ReplaceRegionSlice, Size, TextureDescriptor,
    TextureSwizzleChannels,
};

type A = Api;

const SHARED: u64 = 0;
const PRIVATE: u64 = 2;
const MEMORYLESS: u64 = 3;

fn descriptor(
    kind: u64,
    format: u32,
    (w, h, d): (u64, u64, u64),
    array: u64,
    mips: u64,
    storage: u64,
    usage: u64,
) -> TextureDescriptor {
    TextureDescriptor {
        allow_gpu_optimized_contents: true,
        array_length: array,
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
        mipmap_level_count: mips,
        pixel_format: u64::from(format),
        protection_options: 0,
        resource_index: 0,
        resource_options: storage << 4,
        rotation: 0,
        sample_count: 1,
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

fn handle(n: u64) -> Handle {
    Handle::new(n).unwrap()
}

fn region(o: (u64, u64, u64), s: (u64, u64, u64)) -> Region {
    Region { origin: Origin { x: o.0, y: o.1, z: o.2 }, size: Size { width: s.0, height: s.1, depth: s.2 } }
}

/// The executor side of the tests: memory services, guest memory and a bump allocator in it.
struct Ctx<'a> {
    g: &'a Gpu,
    mem: Memory<A>,
    guest: &'a Guest,
    next: std::cell::Cell<u64>,
    handles: std::cell::Cell<u64>,
}

impl<'a> Ctx<'a> {
    fn new(g: &'a Gpu, guest: &'a Guest) -> Ctx<'a> {
        Ctx {
            g,
            mem: Memory::new(g.device.clone(), g.features),
            guest,
            next: std::cell::Cell::new(guest.base + 0x1000),
            handles: std::cell::Cell::new(1),
        }
    }

    fn alloc(&self, len: u64) -> u64 {
        let a = self.next.get();
        self.next.set((a + len).next_multiple_of(4096) + 4096);
        assert!(self.next.get() < self.guest.base + self.guest.memory.borrow().len() as u64, "guest memory exhausted");
        a
    }

    fn handle(&self) -> Handle {
        let h = self.handles.get();
        self.handles.set(h + 1);
        handle(h)
    }

    /// Run the texel copy test of every format a texture of `t` needs, recording the outcome.
    fn prove(&self, t: &Texture<A>) {
        if let Some(f) = self.mem.texture_needs_proof(t) {
            let outcome = self.g.block_on(self.mem.prove_texel_copies(f)).unwrap();
            self.mem.record_proof(f, outcome);
        }
    }

    fn replace(
        &self,
        t: &Texture<A>,
        (level, slice): (u64, u64),
        r: Region,
        bytes: &[u8],
        (bpr, bpi): (u64, u64),
    ) -> Result<(), MemError> {
        let addr = self.alloc(bytes.len() as u64);
        self.guest.cpu_write(addr, bytes);
        let req = ReplaceRegionSlice {
            texture: handle(1),
            region: r,
            level,
            slice,
            pixel_bytes: GuestAddr(addr),
            bytes_per_row: bpr,
            bytes_per_image: bpi,
        };
        self.mem.replace_region_slice(self.guest, TASK, t, &req)
    }

    /// `getBytes` into guest memory that holds `fill` bytes first; returns the guest's `span` bytes afterwards.
    fn get(
        &self,
        t: &Rc<Texture<A>>,
        (level, slice): (u64, u64),
        r: Region,
        (bpr, bpi): (u64, u64),
        span: usize,
    ) -> Result<Vec<u8>, MemError> {
        let addr = self.alloc(span as u64);
        self.guest.cpu_write(addr, &vec![0xa5; span]);
        let req = GetBytesSlice {
            texture: handle(1),
            pixel_bytes: GuestAddr(addr),
            bytes_per_row: bpr,
            bytes_per_image: bpi,
            region: r,
            level,
            slice,
        };
        let read = self.mem.get_bytes_slice(t, &req)?;
        let bytes = self.g.block_on(read.wait())?;
        bytes.write_guest(&self.mem, self.guest, TASK, addr)?;
        Ok(self.guest.cpu_read(addr, span))
    }

    fn shared_buffer(&self, bytes: &[u8]) -> Rc<Buffer<A>> {
        let contents = self.alloc(bytes.len() as u64);
        self.guest.cpu_write(contents, bytes);
        let r = CreateBuffer {
            buffer: self.handle(),
            length: bytes.len() as u64,
            options: 0,
            contents: GuestAddr(contents),
        };
        self.mem.create_buffer(self.guest, TASK, &r).unwrap()
    }

    fn contents(&self, b: &Buffer<A>) -> Vec<u8> {
        let webmetal_host::mem::buffer::Backing::Guest { addr, .. } = b.backing else { panic!("private") };
        self.guest.cpu_read(addr, b.length as usize)
    }

    /// Submit an encoder with the readbacks of `buffers`, wait and merge.
    fn finish(&self, encoder: <A as webmetal_host::gpu::Api>::CommandEncoder, buffers: &[Rc<Buffer<A>>]) {
        let mut r: Readbacks<A> = self.mem.encode_readbacks(&encoder, buffers).unwrap();
        self.mem.submit(&[&encoder.finish(&CommandBufferDescriptor::new())], &mut r);
        let arrived = self.g.block_on(r.wait()).unwrap();
        arrived.merge(&self.mem, self.guest).unwrap();
    }

    /// Blit `bytes` (laid out as `layout`) into a region of `t` from a buffer, then blit the region into another
    /// buffer laid out the same way; returns that buffer's contents.
    fn blit_round_trip(&self, t: &Rc<Texture<A>>, r: &TextureRegion, bytes: &[u8]) -> Result<Vec<u8>, MemError> {
        let src = self.shared_buffer(bytes);
        let dst = self.shared_buffer(&vec![0u8; bytes.len()]);
        let mut encoder = self.g.device.create_command_encoder(&CommandEncoderDescriptor::new());
        self.mem.encode_buffer_to_texture(&encoder, (&src, 0), t, r)?;
        if self.mem.prepare_texture(&encoder, t)? == Prepare::Decode {
            // A decoded format rebuilds its stored texels from the blocks on the CPU before the next GPU use.
            self.finish(encoder, &[]);
            for read in self.mem.refresh_decoded(t)? {
                let blocks = self.g.block_on(read.wait())?;
                self.mem.finish_refresh(t, blocks)?;
            }
            encoder = self.g.device.create_command_encoder(&CommandEncoderDescriptor::new());
        }
        self.mem.encode_texture_to_buffer(&encoder, t, r, (&dst, 0))?;
        self.finish(encoder, std::slice::from_ref(&dst));
        Ok(self.contents(&dst))
    }
}

/// One cell of the matrix.
#[derive(Clone, Debug, PartialEq)]
enum Cell {
    Exact,
    /// Refused loudly: the reason, as the executor reports it.
    Refused(String),
    /// Metal itself does not define the transfer (memoryless contents).
    Undefined,
    /// The reference has no such transfer to compare with.
    NotRun(&'static str),
}

/// The refusals the matrix may show, each an open item of the memory module or a format the host cannot provide.
const ALLOWED_TRANSFER_REFUSALS: [&str; 3] = [
    "BGR10A2 blits need a channel swap pass",
    "WebGPU cannot copy into depth32float textures",
    "per-aspect blits of a combined depth-stencil texture",
];
const ALLOWED_RENDER_TARGET_REFUSALS: [&str; 5] =
    ["render-attachment", "compressed-render-target", "422-chroma-reconstruction", "xr-srgb-transfer", "msaa"];

impl Cell {
    fn of(r: Result<(), MemError>) -> Cell {
        match r {
            Ok(()) => Cell::Exact,
            Err(MemError::Undefined { .. }) => Cell::Undefined,
            Err(e) => Cell::Refused(e.to_string()),
        }
    }

    fn short(&self) -> String {
        match self {
            Cell::Exact => "exact".into(),
            Cell::Undefined => "undefined in Metal".into(),
            Cell::Refused(why) => format!("refused: {}", why.chars().take(110).collect::<String>()),
            Cell::NotRun(why) => format!("not run: {why}"),
        }
    }
}

/// The golden input of a format and its layout: (bytes, width, height, bytes per row, block width, block height).
fn golden_input(g: &Golden) -> Option<(Vec<u8>, u64, u64, u64)> {
    let bytes = g.bytes("input")?;
    Some((bytes, g.layout("input", "width")?, g.layout("input", "height")?, g.layout("input", "bytes_per_row")?))
}

/// The byte-exact upload/readback matrix: every format the format module handles, through replaceRegion and getBytes
/// of a shared texture and through blits to and from a private texture, on a device with every feature the adapter
/// offers and on one with none, compared with the Metal reference outputs.
#[test]
fn upload_readback_matrix() {
    let mut report = String::new();
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for all in [true, false] {
        let g = Gpu::new(all);
        let guest = Guest::new(64 << 20, None);
        let ctx = Ctx::new(&g, &guest);
        let host = format!("{} ({})", g.name(), if all { "all features" } else { "no features" });
        writeln!(
            report,
            "\n## {host}\n\n| Format | Stored as | Shared: replaceRegion/getBytes | Private: blits | Memoryless |"
        )
        .unwrap();
        writeln!(report, "| --- | --- | --- | --- | --- |").unwrap();
        for value in handling::all_values() {
            let Some(row) = METAL_FORMATS.iter().find(|r| r.value == value) else { continue };
            let name = row.name.trim_start_matches("MTLPixelFormat");
            let resolution = match handling::resolve_texture(value, ctx.mem.features(), Dimension::Planar) {
                Ok(r) => r,
                Err(Unavailable::NotOnDevice(_)) => continue,
                Err(Unavailable::Refused(r)) => {
                    writeln!(report, "| {name} | | refused: {} | refused: {} | |", r.id, r.id).unwrap();
                    *counts.entry("format refused by the format table").or_default() += 1;
                    continue;
                }
            };
            let case =
                ["upload-read", "blocks"].iter().map(|k| format!("format.{name}.{k}")).find(|c| Golden::exists(c));
            let Some(case) = case else {
                writeln!(report, "| {name} | {resolution:?} | no reference output | | |").unwrap();
                *counts.entry("no reference output").or_default() += 1;
                continue;
            };
            let golden = Golden::load(&case);
            if golden.status() != "ok" {
                writeln!(report, "| {name} | | reference: {} | | |", golden.status()).unwrap();
                *counts.entry("reference case did not run on the reference device").or_default() += 1;
                continue;
            }
            let blocks = Blocks::of(resolution);
            let Some((input, w, h, bpr)) = golden_input(&golden) else {
                // Depth32Float_Stencil8: the reference blits each aspect separately.
                assert!(blocks.is_none(), "{case} has no input record");
                let why = "refused: per-aspect blits of a combined depth-stencil texture are not implemented";
                writeln!(report, "| {name} | {} (aspects) | {why} | {why} | |", resolution.storage().name()).unwrap();
                *counts.entry("refused").or_default() += 2;
                continue;
            };
            let stored = format!("{}{}", resolution.storage().name(), if blocks.is_some() { "" } else { " (aspects)" });
            let depth_like = !resolution.storage().caps().aspects.is_empty();

            // Shared storage: replaceRegion, then getBytes (not for depth formats: the reference uses blits).
            let shared = if depth_like {
                Cell::NotRun("the reference transfers depth and stencil textures with blits only")
            } else {
                let t = ctx
                    .mem
                    .create_texture(TASK, ctx.handle(), &descriptor(2, value, (w, h, 1), 1, 1, SHARED, 0))
                    .unwrap();
                ctx.prove(&t);
                let expected = golden.bytes("getbytes").or_else(|| golden.bytes("readback")).unwrap();
                Cell::of(ctx.replace(&t, (0, 0), region((0, 0, 0), (w, h, 1)), &input, (bpr, 0)).and_then(|()| {
                    let got = ctx.get(&t, (0, 0), region((0, 0, 0), (w, h, 1)), (bpr, 0), input.len())?;
                    assert_eq!(got, expected, "{host}: {name} getBytes");
                    Ok(())
                }))
            };

            // Private storage: a blit from a buffer and a blit back to a buffer.
            let private = {
                let t = ctx
                    .mem
                    .create_texture(TASK, ctx.handle(), &descriptor(2, value, (w, h, 1), 1, 1, PRIVATE, 0))
                    .unwrap();
                ctx.prove(&t);
                let pv = blocks.and_then(|b| b.pvrtc);
                let (bytes, expected, layout) = match (pv, blocks) {
                    (Some(bpp), Some(b)) => {
                        // PVRTC blits take row-linear words (MTLBlitOptionRowLinearPVRTC).
                        let rows = pvrtc::reorder(bpp, w as u32, h as u32, &input, false).unwrap();
                        let (wx, wy) = b.count(w as u32, h as u32);
                        let l = RegionLayout {
                            origin: (0, 0, 0),
                            size: (w as u32, h as u32, 1),
                            blocks: (wx, wy),
                            images: 1,
                            row_bytes: u64::from(wx) * 8,
                            bytes_per_row: u64::from(wx) * 8,
                            bytes_per_image: rows.len() as u64,
                            span: rows.len() as u64,
                            pvrtc: false,
                        };
                        (rows, golden.bytes("readback-row-linear").unwrap(), Some(l))
                    }
                    (_, Some(b)) => {
                        let a = RegionArgs {
                            origin: (0, 0, 0),
                            size: (w, h, 1),
                            level: 0,
                            slice: 0,
                            bytes_per_row: bpr,
                            bytes_per_image: 0,
                        };
                        let l = RegionLayout::new("blit", &t.geometry, b, &a).unwrap();
                        (input.clone(), golden.bytes("readback").unwrap(), Some(l))
                    }
                    (_, None) => (input.clone(), vec![], None),
                };
                match layout {
                    None => {
                        Cell::of(Err(MemError::unimplemented("per-aspect blits of a combined depth-stencil texture")))
                    }
                    Some(layout) => {
                        let r = TextureRegion { level: 0, slice: 0, layout };
                        Cell::of(ctx.blit_round_trip(&t, &r, &bytes).map(|got| {
                            assert_eq!(got, expected, "{host}: {name} blits");
                        }))
                    }
                }
            };

            // Memoryless: only a render target; it has no contents to transfer.
            let memoryless = {
                let d = descriptor(2, value, (w, h, 1), 1, 1, MEMORYLESS, 0x4);
                match ctx.mem.create_texture(TASK, ctx.handle(), &d) {
                    Ok(t) => {
                        let replace = ctx.replace(&t, (0, 0), region((0, 0, 0), (w, h, 1)), &input, (bpr, 0));
                        assert!(matches!(replace, Err(MemError::Undefined { .. })), "{name}: {replace:?}");
                        Cell::Undefined
                    }
                    Err(e) => Cell::of(Err(e)),
                }
            };
            for (what, c) in [("shared", &shared), ("private", &private), ("memoryless", &memoryless)] {
                let key = match c {
                    Cell::Exact => "exact",
                    Cell::Undefined => "undefined in Metal",
                    Cell::Refused(_) => "refused",
                    Cell::NotRun(_) => "not run",
                };
                *counts.entry(key).or_default() += 1;
                if let Cell::Refused(why) = c {
                    let allowed = if what == "memoryless" {
                        ALLOWED_RENDER_TARGET_REFUSALS.iter().any(|a| why.contains(a))
                    } else {
                        ALLOWED_TRANSFER_REFUSALS.iter().any(|a| why.contains(a))
                    };
                    assert!(allowed, "{host}: {name} {what}: unexpected refusal: {why}");
                }
            }
            writeln!(
                report,
                "| {name} | {stored} | {} | {} | {} |",
                shared.short(),
                private.short(),
                memoryless.short()
            )
            .unwrap();
        }
        for (f, outcome) in
            webmetal_host::format::webgpu::WebGpuFormat::ALL.iter().filter_map(|f| ctx.mem.proof(*f).map(|o| (f, o)))
        {
            writeln!(
                report,
                "\ntexel copy test of {}: {}",
                f.name(),
                match outcome {
                    Ok(()) => "every pattern survived".to_string(),
                    Err(why) => format!("failed: {why}"),
                }
            )
            .unwrap();
        }
        g.assert_no_errors();
    }
    writeln!(report, "\ncells: {counts:?}").unwrap();
    eprintln!("{report}");
    if let Ok(path) = std::env::var("WEBMETAL_MEM_REPORT") {
        std::fs::write(path, &report).unwrap();
    }
}

/// `WMPattern` of the Metal reference harness (tests/reference/src/Harness.m): xorshift32 bytes.
fn wm_pattern(len: usize, seed: u32) -> Vec<u8> {
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

/// A CPU model of a texture's bytes: tight rows of blocks per subresource (`level * layers + layer`, a 3D level's
/// images one after another).
struct Model {
    subs: Vec<Vec<u8>>,
}

/// Every texture type: random regions written with padded strides and read back whole, against a CPU model, for
/// formats stored natively, with a twin (float, compressed, decoded) and converted (BGR10A2, packed 16-bit).
#[test]
fn texture_types_and_regions() {
    for all in [true, false] {
        let g = Gpu::new(all);
        let guest = Guest::new(256 << 20, None);
        let ctx = Ctx::new(&g, &guest);
        let mut rng = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        // (format, can be 1D, can be 3D)
        let formats: [(u32, bool, bool); 8] = [
            (70, true, true),
            (115, true, true),
            (10, true, true),
            (94, true, true),
            (40, true, true),
            (180, false, true),
            (204, false, true),
            (166, false, false),
        ];
        // (type, width, height, depth, array length, mip levels)
        let types: [(u64, u64, u64, u64, u64, u64); 7] = [
            (0, 37, 1, 1, 1, 1),
            (1, 21, 1, 1, 3, 1),
            (2, 24, 20, 1, 1, 3),
            (3, 16, 12, 1, 3, 2),
            (5, 16, 16, 1, 1, 2),
            (6, 8, 8, 1, 2, 1),
            (7, 12, 8, 6, 1, 2),
        ];
        let mut runs = 0;
        for &(format, one_d, three_d) in &formats {
            let resolution = match handling::resolve_texture(format, ctx.mem.features(), Dimension::Planar) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let b = Blocks::of(resolution).unwrap();
            for &(kind, w, h, d, array, mips) in &types {
                if (kind <= 1 && !one_d) || (kind == 7 && !three_d) {
                    continue;
                }
                let pvrtc = b.pvrtc.is_some();
                let (w, h, mips) = if pvrtc { (16, 16, 3) } else { (w, h, mips) };
                if pvrtc && kind != 2 {
                    continue;
                }
                let dim = if kind == 7 { Dimension::ThreeD } else { Dimension::Planar };
                if handling::resolve_texture(format, ctx.mem.features(), dim).ok() != Some(resolution) {
                    continue;
                }
                let t = ctx
                    .mem
                    .create_texture(TASK, ctx.handle(), &descriptor(kind, format, (w, h, d), array, mips, SHARED, 1))
                    .unwrap();
                ctx.prove(&t);
                let geo = t.geometry;
                let per_level = if kind == 7 { 1 } else { geo.slices() };
                let mut model = Model { subs: Vec::new() };
                // Fill every subresource whole.
                for level in 0..geo.mip_levels {
                    let (lw, lh, ld) = geo.level_extent(level);
                    let (bx, by) = b.count(lw, lh);
                    for layer in 0..per_level {
                        let images = if kind == 7 { ld } else { 1 };
                        let row = u64::from(bx) * u64::from(b.bytes);
                        let data: Vec<u8> =
                            (0..row * u64::from(by) * u64::from(images)).map(|_| next() as u8).collect();
                        let (bpr, bpi) =
                            if pvrtc { (0, 0) } else { (row, if images > 1 { row * u64::from(by) } else { 0 }) };
                        ctx.replace(
                            &t,
                            (u64::from(level), u64::from(layer)),
                            region((0, 0, 0), (u64::from(lw), u64::from(lh), u64::from(images))),
                            &data,
                            (bpr, bpi),
                        )
                        .unwrap_or_else(|e| panic!("{}: format {format} type {kind} level {level}: {e}", g.name()));
                        model.subs.push(data);
                    }
                }
                // Overwrite random block-aligned regions with padded strides.
                if !pvrtc {
                    for _ in 0..6 {
                        let level = (next() % u64::from(geo.mip_levels)) as u32;
                        let layer = (next() % u64::from(per_level)) as u32;
                        let (lw, lh, ld) = geo.level_extent(level);
                        let (bx, by) = b.count(lw, lh);
                        let x0 = (next() % u64::from(bx)) as u32;
                        let y0 = (next() % u64::from(by)) as u32;
                        let nx = 1 + (next() % u64::from(bx - x0)) as u32;
                        let ny = 1 + (next() % u64::from(by - y0)) as u32;
                        let (z0, nz) = if kind == 7 {
                            let z0 = (next() % u64::from(ld)) as u32;
                            (z0, 1 + (next() % u64::from(ld - z0)) as u32)
                        } else {
                            (0, 1)
                        };
                        let row = u64::from(nx) * u64::from(b.bytes);
                        let bpr = row + u64::from(b.bytes) * (next() % 5);
                        let bpi = if nz > 1 { bpr * u64::from(ny) + u64::from(b.bytes) * (next() % 3) } else { 0 };
                        let span = if nz > 1 { bpi * u64::from(nz - 1) } else { 0 } + bpr * u64::from(ny - 1) + row;
                        let data: Vec<u8> = (0..span).map(|_| next() as u8).collect();
                        // Texel region: whole blocks, the last ones clipped to the level.
                        let tx = x0 * b.width;
                        let ty = y0 * b.height;
                        let tw = (nx * b.width).min(lw - tx);
                        let th = (ny * b.height).min(lh - ty);
                        ctx.replace(
                            &t,
                            (u64::from(level), u64::from(layer)),
                            region(
                                (u64::from(tx), u64::from(ty), u64::from(z0)),
                                (u64::from(tw), u64::from(th), u64::from(nz)),
                            ),
                            &data,
                            (bpr, bpi),
                        )
                        .unwrap_or_else(|e| panic!("{}: format {format} type {kind}: {e}", g.name()));
                        let sub = &mut model.subs[(level * per_level + layer) as usize];
                        let full_row = (u64::from(bx) * u64::from(b.bytes)) as usize;
                        for z in 0..nz as usize {
                            for y in 0..ny as usize {
                                let src = z * bpi as usize + y * bpr as usize;
                                let dst = ((z0 as usize + z) * by as usize + y0 as usize + y) * full_row
                                    + x0 as usize * b.bytes as usize;
                                sub[dst..dst + row as usize].copy_from_slice(&data[src..src + row as usize]);
                            }
                        }
                    }
                }
                // Read every subresource back whole and compare with the model.
                for level in 0..geo.mip_levels {
                    let (lw, lh, ld) = geo.level_extent(level);
                    let (bx, by) = b.count(lw, lh);
                    for layer in 0..per_level {
                        let images = if kind == 7 { ld } else { 1 };
                        let row = u64::from(bx) * u64::from(b.bytes);
                        let (bpr, bpi) =
                            if pvrtc { (0, 0) } else { (row, if images > 1 { row * u64::from(by) } else { 0 }) };
                        let want = &model.subs[(level * per_level + layer) as usize];
                        let got = ctx
                            .get(
                                &t,
                                (u64::from(level), u64::from(layer)),
                                region((0, 0, 0), (u64::from(lw), u64::from(lh), u64::from(images))),
                                (bpr, bpi),
                                want.len(),
                            )
                            .unwrap();
                        assert_eq!(&got, want, "{}: format {format} type {kind} level {level} layer {layer}", g.name());
                    }
                }
                runs += 1;
            }
        }
        eprintln!("{}: {runs} textures written and read back", g.name());
        assert!(runs >= 30, "{}: only {runs} texture runs", g.name());
        g.assert_no_errors();
    }
}

/// `storage.texture-replace-region-ordering`: a copy encoded before a replaceRegion but committed after it reads the
/// later bytes, as on real Metal.
#[test]
fn replace_region_ordering_matches_metal() {
    let golden = Golden::load("storage.texture-replace-region-ordering");
    let g = Gpu::new(true);
    let guest = Guest::new(16 << 20, None);
    let ctx = Ctx::new(&g, &guest);
    let t = ctx.mem.create_texture(TASK, ctx.handle(), &descriptor(2, 73, (8, 8, 1), 1, 1, SHARED, 1)).unwrap();
    let (first, second) = (wm_pattern(256, 101), wm_pattern(256, 102));
    ctx.replace(&t, (0, 0), region((0, 0, 0), (8, 8, 1)), &first, (32, 0)).unwrap();
    let out = ctx.shared_buffer(&[0u8; 256]);
    let encoder = g.device.create_command_encoder(&CommandEncoderDescriptor::new());
    let a =
        RegionArgs { origin: (0, 0, 0), size: (8, 8, 1), level: 0, slice: 0, bytes_per_row: 32, bytes_per_image: 0 };
    let r = TextureRegion {
        level: 0,
        slice: 0,
        layout: RegionLayout::new("blit", &t.geometry, t.blocks.unwrap(), &a).unwrap(),
    };
    ctx.mem.encode_texture_to_buffer(&encoder, &t, &r, (&out, 0)).unwrap();
    ctx.replace(&t, (0, 0), region((0, 0, 0), (8, 8, 1)), &second, (32, 0)).unwrap();
    ctx.finish(encoder, std::slice::from_ref(&out));
    assert_eq!(ctx.contents(&out), golden.bytes("copied").unwrap());
    assert_eq!(ctx.contents(&out), second);
    g.assert_no_errors();
}

const WRITE_TEXTURE: &str = "
@group(0) @binding(0) var texels: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(1) var<storage, read> values: array<vec4u>;
@compute @workgroup_size(8, 4) fn main(@builtin(local_invocation_id) id: vec3u) {
    textureStore(texels, id.xy, values[id.y * 8u + id.x]);
}
";

/// `storage.texture-get-bytes-after-gpu-write`: getBytes of a shared texture after a compute pass wrote it.
#[test]
fn get_bytes_after_gpu_write_matches_metal() {
    use webmetal_host::gpu::idl::{
        AutoLayoutMode, BindGroupDescriptor, BindGroupEntry, BindingResource, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineLayoutOrAuto, ProgrammableStage, ShaderModuleDescriptor,
        TextureViewDescriptor,
    };
    use webmetal_host::gpu::{BindingCommands as _, ComputePassEncoder as _, PipelineBase as _, Texture as _};
    let golden = Golden::load("storage.texture-get-bytes-after-gpu-write");
    let values = golden.bytes("texels").unwrap();
    let g = Gpu::new(true);
    let guest = Guest::new(16 << 20, None);
    let ctx = Ctx::new(&g, &guest);
    // RGBA32Uint, shader read and write.
    let t = ctx.mem.create_texture(TASK, ctx.handle(), &descriptor(2, 123, (8, 4, 1), 1, 1, SHARED, 3)).unwrap();
    let input = ctx.shared_buffer(&values);
    let module = g.device.create_shader_module(&ShaderModuleDescriptor::new(WRITE_TEXTURE.to_string()));
    let pipeline = g.device.create_compute_pipeline(&ComputePipelineDescriptor::new(
        PipelineLayoutOrAuto::AutoLayoutMode(AutoLayoutMode::Auto),
        ProgrammableStage::new(&module),
    ));
    let view = t.gpu().create_view(&TextureViewDescriptor::new()).unwrap();
    let layout = pipeline.get_bind_group_layout(0);
    let group = g.device.create_bind_group(&BindGroupDescriptor::new(
        &layout,
        vec![
            BindGroupEntry::new(0, BindingResource::TextureView(&view)),
            BindGroupEntry::new(1, BindingResource::Buffer(input.gpu())),
        ],
    ));
    ctx.mem.upload_buffer(&guest, &input).unwrap();
    let encoder = g.device.create_command_encoder(&CommandEncoderDescriptor::new());
    let pass =
        encoder.begin_compute_pass(&ComputePassDescriptor { label: String::new(), timestamp_writes: None }).unwrap();
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, Some(&group), &[]);
    pass.dispatch_workgroups(1, 1, 1);
    pass.end();
    ctx.mem.note_texture_write(&t, 0..1, 0..1);
    ctx.finish(encoder, &[]);
    let got = ctx.get(&t, (0, 0), region((0, 0, 0), (8, 4, 1)), (128, 0), values.len()).unwrap();
    assert_eq!(got, values);
    g.assert_no_errors();
}

/// A float texture whose twin went stale because the GPU wrote it is read from the stored texture, which needs the
/// host's texel copies of the format proven; the specials the reference inputs hold come back unchanged.
#[test]
fn stored_reads_after_gpu_writes() {
    let golden = Golden::load("format.RGBA16Float.upload-read");
    let input = golden.bytes("input").unwrap();
    let g = Gpu::new(true);
    let guest = Guest::new(16 << 20, None);
    let ctx = Ctx::new(&g, &guest);
    let t = ctx.mem.create_texture(TASK, ctx.handle(), &descriptor(2, 115, (8, 4, 1), 1, 1, SHARED, 0)).unwrap();
    assert!(t.has_twin());
    // Before the proof: the transfer is refused loudly.
    let early = ctx.replace(&t, (0, 0), region((0, 0, 0), (8, 4, 1)), &input, (64, 0));
    assert!(matches!(early, Err(MemError::Unimplemented { .. })), "{early:?}");
    ctx.prove(&t);
    ctx.replace(&t, (0, 0), region((0, 0, 0), (8, 4, 1)), &input, (64, 0)).unwrap();
    ctx.mem.note_texture_write(&t, 0..1, 0..1);
    assert!(!t.twin_current(0, 0));
    let got = ctx.get(&t, (0, 0), region((0, 0, 0), (8, 4, 1)), (64, 0), input.len()).unwrap();
    assert_eq!(got, input);
    g.assert_no_errors();
}

/// Software-decoded compressed textures filled by blits rebuild their stored texels from the blocks; the stored
/// texels equal the format module's decode of the blocks.
#[test]
fn decoded_textures_refresh_after_blits() {
    use webmetal_host::format::handling::Emulation;
    use webmetal_host::format::storage;
    use webmetal_host::gpu::idl::{MapMode, TexelCopyBufferInfo, TexelCopyTextureInfo};
    use webmetal_host::gpu::{Buffer as _, MappedRange as _, Size64};
    let g = Gpu::new(false);
    let guest = Guest::new(16 << 20, None);
    let ctx = Ctx::new(&g, &guest);
    for (format, case) in
        [(180, "format.ETC2_RGB8.blocks"), (166, "format.PVRTC_RGBA_4BPP.blocks"), (204, "format.ASTC_4x4_LDR.blocks")]
    {
        let golden = Golden::load(case);
        let (input, w, h, bpr) = golden_input(&golden).unwrap();
        let t =
            ctx.mem.create_texture(TASK, ctx.handle(), &descriptor(2, format, (w, h, 1), 1, 1, PRIVATE, 1)).unwrap();
        let Resolution::Emulated(e @ Emulation { .. }) = t.resolution else {
            panic!("{case} is decoded on a host without the feature")
        };
        let b = t.blocks.unwrap();
        let (bytes, layout) = match b.pvrtc {
            Some(bpp) => {
                let rows = pvrtc::reorder(bpp, w as u32, h as u32, &input, false).unwrap();
                let (wx, wy) = b.count(w as u32, h as u32);
                let n = rows.len() as u64;
                (
                    rows,
                    RegionLayout {
                        origin: (0, 0, 0),
                        size: (w as u32, h as u32, 1),
                        blocks: (wx, wy),
                        images: 1,
                        row_bytes: u64::from(wx) * 8,
                        bytes_per_row: u64::from(wx) * 8,
                        bytes_per_image: n,
                        span: n,
                        pvrtc: false,
                    },
                )
            }
            None => {
                let a = RegionArgs {
                    origin: (0, 0, 0),
                    size: (w, h, 1),
                    level: 0,
                    slice: 0,
                    bytes_per_row: bpr,
                    bytes_per_image: 0,
                };
                (input.clone(), RegionLayout::new("blit", &t.geometry, b, &a).unwrap())
            }
        };
        let src = ctx.shared_buffer(&bytes);
        let encoder = g.device.create_command_encoder(&CommandEncoderDescriptor::new());
        let r = TextureRegion { level: 0, slice: 0, layout };
        ctx.mem.encode_buffer_to_texture(&encoder, (&src, 0), &t, &r).unwrap();
        assert!(!t.stored_current(0, 0));
        assert_eq!(ctx.mem.prepare_texture(&encoder, &t).unwrap(), Prepare::Decode);
        ctx.finish(encoder, &[]);
        for read in ctx.mem.refresh_decoded(&t).unwrap() {
            let blocks = g.block_on(read.wait()).unwrap();
            ctx.mem.finish_refresh(&t, blocks).unwrap();
        }
        assert!(t.stored_current(0, 0));
        // Read the stored texels directly and compare with the decode of the blocks.
        let footprint = u64::from(e.storage.caps().footprint);
        let row = (w * footprint).next_multiple_of(256);
        let staging = g
            .device
            .create_buffer(&webmetal_host::gpu::idl::BufferDescriptor::new(
                Size64::new(row * h).unwrap(),
                webmetal_host::gpu::idl::BufferUsage::MAP_READ | webmetal_host::gpu::idl::BufferUsage::COPY_DST,
            ))
            .unwrap();
        let encoder = g.device.create_command_encoder(&CommandEncoderDescriptor::new());
        let mut dst = TexelCopyBufferInfo::<A>::new(&staging);
        dst.bytes_per_row = Some(row as u32);
        encoder
            .copy_texture_to_buffer(
                &TexelCopyTextureInfo::<A>::new(t.gpu()),
                &dst,
                &webmetal_host::gpu::idl::Extent3D::Sequence(vec![w as u32, h as u32, 1]),
            )
            .unwrap();
        use webmetal_host::gpu::Queue as _;
        g.device.queue().submit(&[&encoder.finish(&CommandBufferDescriptor::new())]);
        g.block_on(staging.map_async(MapMode::READ, Size64::new_const(0), None)).unwrap();
        let mut got = vec![0u8; (row * h) as usize];
        staging.get_mapped_range(Size64::new_const(0), None).unwrap().read(0, &mut got).unwrap();
        staging.unmap();
        let want = storage::to_storage(&e, w as u32, h as u32, &input, bpr as usize).unwrap();
        let tight: Vec<u8> =
            (0..h as usize).flat_map(|y| got[y * row as usize..][..(w * footprint) as usize].to_vec()).collect();
        assert_eq!(tight, want, "{case}");
    }
    g.assert_no_errors();
}

/// Error paths: calls Metal does not define are refused as such, and transfers WebGPU cannot do are refused loudly.
#[test]
fn texture_error_paths() {
    let g = Gpu::new(true);
    let guest = Guest::new(16 << 20, None);
    let ctx = Ctx::new(&g, &guest);
    let data = vec![1u8; 4096];
    let private = ctx.mem.create_texture(TASK, ctx.handle(), &descriptor(2, 70, (8, 8, 1), 1, 1, PRIVATE, 1)).unwrap();
    assert!(matches!(
        ctx.replace(&private, (0, 0), region((0, 0, 0), (8, 8, 1)), &data, (32, 0)),
        Err(MemError::Undefined { .. })
    ));
    let shared = ctx.mem.create_texture(TASK, ctx.handle(), &descriptor(2, 70, (8, 8, 1), 1, 2, SHARED, 1)).unwrap();
    for (o, s, bpr, why) in [
        ((4, 0, 0), (8, 8, 1), 32, "outside the level"),
        ((0, 0, 0), (8, 8, 1), 16, "short rows"),
        ((0, 0, 0), (8, 8, 1), 0, "bytesPerRow 0 for several rows"),
        ((0, 0, 0), (8, 8, 1), 34, "rows not a multiple of the texel"),
        ((0, 0, 0), (0, 8, 1), 32, "empty"),
    ] {
        assert!(
            matches!(ctx.replace(&shared, (0, 0), region(o, s), &data, (bpr, 0)), Err(MemError::Undefined { .. })),
            "{why}"
        );
    }
    assert!(
        matches!(
            ctx.replace(&shared, (2, 0), region((0, 0, 0), (1, 1, 1)), &data, (4, 0)),
            Err(MemError::Undefined { .. })
        ),
        "level"
    );
    assert!(
        matches!(
            ctx.replace(&shared, (0, 1), region((0, 0, 0), (1, 1, 1)), &data, (4, 0)),
            Err(MemError::Undefined { .. })
        ),
        "slice"
    );
    let mut msaa = descriptor(4, 70, (8, 8, 1), 1, 1, PRIVATE, 4);
    msaa.sample_count = 4;
    assert!(ctx.mem.create_texture(TASK, ctx.handle(), &msaa).is_ok());
    msaa.sample_count = 2;
    assert!(matches!(ctx.mem.create_texture(TASK, ctx.handle(), &msaa), Err(MemError::HostUnsupported { .. })));
    // A texture buffer comes from a buffer; 2D multisample arrays are not on the A10X.
    assert!(matches!(
        ctx.mem.create_texture(TASK, ctx.handle(), &descriptor(9, 70, (8, 1, 1), 1, 1, SHARED, 1)),
        Err(MemError::Undefined { .. })
    ));
    assert!(matches!(
        ctx.mem.create_texture(TASK, ctx.handle(), &descriptor(8, 70, (8, 8, 1), 2, 1, PRIVATE, 4)),
        Err(MemError::Undefined { .. })
    ));
    // Too many levels.
    assert!(matches!(
        ctx.mem.create_texture(TASK, ctx.handle(), &descriptor(2, 70, (8, 8, 1), 1, 5, SHARED, 1)),
        Err(MemError::Undefined { .. })
    ));
    // An R8 blit whose rows start off a 4-byte boundary needs a shader re-pitch.
    let r8 = ctx.mem.create_texture(TASK, ctx.handle(), &descriptor(2, 10, (3, 3, 1), 1, 1, PRIVATE, 1)).unwrap();
    let a = RegionArgs { origin: (0, 0, 0), size: (3, 3, 1), level: 0, slice: 0, bytes_per_row: 3, bytes_per_image: 0 };
    let r = TextureRegion {
        level: 0,
        slice: 0,
        layout: RegionLayout::new("blit", &r8.geometry, r8.blocks.unwrap(), &a).unwrap(),
    };
    let src = ctx.shared_buffer(&[7u8; 12]);
    let encoder = g.device.create_command_encoder(&CommandEncoderDescriptor::new());
    assert!(matches!(
        ctx.mem.encode_buffer_to_texture(&encoder, (&src, 0), &r8, &r),
        Err(MemError::Unimplemented { .. })
    ));
    g.assert_no_errors();
}
