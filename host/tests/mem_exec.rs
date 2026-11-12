//! The memory module as the executor's services, on a native WebGPU implementation (`WEBGPU_NATIVE`, see
//! `host/build.rs`): control requests and command buffers go through `webmetal_host::exec::Executor` as the guest
//! sends them, with error scopes per call and strict, and the results are compared with the Metal reference outputs.
//!
//! The executor needs an encoder service to run blits; the blit module is another package, so a stand-in here records
//! the blit records these programs use with the memory module's transfer primitives (`fillBuffer:range:value:`,
//! `copyFromBuffer:...`, `copyFromTexture:...toBuffer:`).

#![cfg(feature = "native")]

mod mem_common;

use std::rc::Rc;
use std::time::Duration;

use mem_common::{Api, Golden, Gpu, Guest, TASK};
use std::collections::BTreeSet;
use webmetal_host::embed::{SurfaceDesc, SurfacePlane as PlaneDesc};
use webmetal_host::exec::failure;
use webmetal_host::exec::{
    Config, ControlService, DebugGroups, EncodeCx, EncoderService, Executor, RecordFailure, Segment, Services,
};
use webmetal_host::gpu::idl::{BufferDescriptor, BufferUsage};
use webmetal_host::gpu::scope::{Granularity, Policy};
use webmetal_host::gpu::{Buffer as _, CommandEncoder as _, Device as _, MappedRange as _, Size64};
use webmetal_host::mem::buffer::Buffer;
use webmetal_host::mem::layout::{RegionArgs, RegionLayout};
use webmetal_host::mem::linear::LinearTexture;
use webmetal_host::mem::service::MemoryServices;
use webmetal_host::mem::texture::{Texture, TextureRegion};
use webmetal_host::mem::{MemError, Memory};
use webmetal_wire::*;

type A = Api;
type E = Guest;

/// The stand-in blit service.
struct Blit {
    memory: Rc<Memory<A>>,
}

fn payload<T: 'static>(cx: &EncodeCx<'_, A, E>, h: Option<Handle>) -> Option<Rc<T>> {
    let p = cx.objects().get(h?)?.payload.clone()?;
    p.downcast::<T>().ok()
}

fn region_args(origin: &Origin, size: &Size, level: u64, slice: u64, bpr: u64, bpi: u64) -> RegionArgs {
    RegionArgs {
        origin: (origin.x, origin.y, origin.z),
        size: (size.width, size.height, size.depth),
        level,
        slice,
        bytes_per_row: bpr,
        bytes_per_image: bpi,
    }
}

impl EncoderService<A, E> for Blit {
    fn encode(&mut self, cx: &mut EncodeCx<'_, A, E>, segment: &Segment) -> Result<(), RecordFailure> {
        let mut groups = DebugGroups::default();
        let records = cx.records();
        for &i in &segment.records {
            let (offset, record) = &records[i];
            let fail = |e: MemError| RecordFailure { offset: *offset as u64, failure: e.failure() };
            let unimplemented =
                || RecordFailure { offset: *offset as u64, failure: failure::unimplemented_record(record.opcode()) };
            let encoder = cx.encoder().clone();
            let mem = &*self.memory;
            match record {
                Command::BeginBlit(_) | Command::EncoderSetLabel(_) => {}
                Command::EncoderPushDebugGroup(_) => groups.push(),
                Command::EncoderPopDebugGroup(_) => {
                    groups.pop();
                }
                Command::BlitFillBuffer(r) => {
                    let b = payload::<Buffer<A>>(cx, r.buffer).ok_or_else(unimplemented)?;
                    if r.range.location % 4 != 0 || r.range.length % 4 != 0 || r.range.length == 0 {
                        return Err(unimplemented());
                    }
                    let mut desc = BufferDescriptor::new(Size64::new(r.range.length).unwrap(), BufferUsage::COPY_SRC);
                    desc.mapped_at_creation = true;
                    let pattern = cx.gpu().call("GPUDevice.createBuffer", "fill", |d| d.create_buffer(&desc)).unwrap();
                    let m = pattern.get_mapped_range(Size64::new(0).unwrap(), None).unwrap();
                    m.write(0, &vec![r.value; r.range.length as usize]).unwrap();
                    pattern.unmap();
                    cx.gpu().call("GPUCommandEncoder.copyBufferToBuffer", "fill", |_| {
                        encoder.copy_buffer_to_buffer_with_offsets(
                            &pattern,
                            Size64::new(0).unwrap(),
                            b.gpu(),
                            Size64::new(r.range.location).unwrap(),
                            Some(Size64::new(r.range.length).unwrap()),
                        )
                    });
                    mem.note_buffer_write(&b, Some((r.range.location, r.range.location + r.range.length)));
                }
                Command::BlitCopyBufferToBuffer(r) => {
                    let src = payload::<Buffer<A>>(cx, r.source_buffer).ok_or_else(unimplemented)?;
                    let dst = payload::<Buffer<A>>(cx, r.destination_buffer).ok_or_else(unimplemented)?;
                    cx.gpu().call("GPUCommandEncoder.copyBufferToBuffer", "copy", |_| {
                        encoder.copy_buffer_to_buffer_with_offsets(
                            src.gpu(),
                            Size64::new(r.source_offset).unwrap(),
                            dst.gpu(),
                            Size64::new(r.destination_offset).unwrap(),
                            Some(Size64::new(r.size).unwrap()),
                        )
                    });
                    mem.note_buffer_read(&src);
                    mem.note_buffer_write(&dst, Some((r.destination_offset, r.destination_offset + r.size)));
                }
                Command::BlitCopyBufferToTexture(r) => {
                    let src = payload::<Buffer<A>>(cx, r.source_buffer).ok_or_else(unimplemented)?;
                    let (t, lt) = match payload::<Texture<A>>(cx, r.destination_texture) {
                        Some(t) => (t, None),
                        None => {
                            let lt =
                                payload::<LinearTexture<A>>(cx, r.destination_texture).ok_or_else(unimplemented)?;
                            (lt.texture.clone().ok_or_else(unimplemented)?, Some(lt))
                        }
                    };
                    let a = region_args(
                        &r.destination_origin,
                        &r.source_size,
                        r.destination_level,
                        r.destination_slice,
                        r.source_bytes_per_row,
                        if r.source_size.depth > 1 { r.source_bytes_per_image } else { 0 },
                    );
                    let layout = RegionLayout::new("blit", &t.geometry, t.blocks.unwrap(), &a).map_err(fail)?;
                    let region = TextureRegion { level: a.level as u32, slice: a.slice as u32, layout };
                    mem.encode_buffer_to_texture(cx.gpu(), &encoder, (&src, r.source_offset), &t, &region)
                        .map_err(fail)?;
                    mem.note_buffer_read(&src);
                    if let Some(lt) = lt {
                        mem.note_linear_write(&lt);
                    }
                }
                Command::BlitCopyTextureToBuffer(r) => {
                    let dst = payload::<Buffer<A>>(cx, r.destination_buffer).ok_or_else(unimplemented)?;
                    let t = match payload::<Texture<A>>(cx, r.source_texture) {
                        Some(t) => t,
                        None => {
                            let lt = payload::<LinearTexture<A>>(cx, r.source_texture).ok_or_else(unimplemented)?;
                            lt.texture.clone().ok_or_else(unimplemented)?
                        }
                    };
                    let a = region_args(
                        &r.source_origin,
                        &r.source_size,
                        r.source_level,
                        r.source_slice,
                        r.destination_bytes_per_row,
                        if r.source_size.depth > 1 { r.destination_bytes_per_image } else { 0 },
                    );
                    let layout = RegionLayout::new("blit", &t.geometry, t.blocks.unwrap(), &a).map_err(fail)?;
                    let region = TextureRegion { level: a.level as u32, slice: a.slice as u32, layout };
                    mem.encode_texture_to_buffer(cx.gpu(), &encoder, &t, &region, (&dst, r.destination_offset))
                        .map_err(fail)?;
                }
                Command::EncoderEndEncoding(_) => {}
                _ => return Err(unimplemented()),
            }
        }
        Ok(())
    }
}

struct Harness {
    g: Gpu,
    guest: Rc<Guest>,
    exec: Executor<A, E>,
    next: std::cell::Cell<u64>,
    alloc: std::cell::Cell<u64>,
}

impl Harness {
    fn new() -> Harness {
        let g = Gpu::new(true);
        let guest = Rc::new(Guest::new(64 << 20, None));
        let svc = g
            .gpu
            .block_on(MemoryServices::new(&g.device, g.features, guest.clone()), Duration::from_secs(120))
            .expect("setup stalled")
            .expect("memory services");
        let services = Services {
            control: vec![svc.control() as Box<dyn ControlService<A, E>>],
            render: None,
            compute: None,
            blit: Some(Box::new(Blit { memory: svc.memory() })),
            coherence: Some(svc.coherence()),
        };
        let config = Config { scopes: Policy { granularity: Granularity::PerCall, strict: true } };
        let exec = Executor::new(g.device.clone(), guest.clone(), services, config);
        let base = guest.base;
        let h =
            Harness { g, guest, exec, next: std::cell::Cell::new(0x100), alloc: std::cell::Cell::new(base + 0x1000) };
        h.ok(Request::Open(Open { schema_hash: Hash { bytes: SCHEMA_HASH }, pid: 1 }));
        h
    }

    fn handle(&self) -> Handle {
        self.next.set(self.next.get() + 1);
        Handle::new(self.next.get()).unwrap()
    }

    fn alloc(&self, len: u64) -> u64 {
        let a = self.alloc.get();
        self.alloc.set((a + len).next_multiple_of(0x4000) + 0x4000);
        a
    }

    fn request(&self, request: Request) -> Reply {
        let mut bytes = Vec::new();
        request.encode(&mut bytes).unwrap();
        let out = self.g.block_on(self.exec.control(TASK, &bytes)).unwrap();
        let (reply, _) = Reply::decode(&out, 0).unwrap();
        reply
    }

    fn status(&self, request: Request) -> Failure {
        webmetal_host::exec::schema::reply_failure(&self.request(request)).cloned().unwrap_or_else(failure::ok)
    }

    fn ok(&self, request: Request) {
        let f = self.status(request);
        assert_eq!(f.status, status::OK, "{f:?}");
    }

    fn buffer(&self, bytes: &[u8], options: u64) -> (Handle, u64) {
        let addr = if options == 0x20 { 0 } else { self.alloc(bytes.len() as u64) };
        if addr != 0 {
            self.guest.cpu_write(addr, bytes);
        }
        let b = self.handle();
        self.ok(Request::CreateBuffer(CreateBuffer {
            buffer: b,
            length: bytes.len() as u64,
            options,
            contents: GuestAddr(addr),
        }));
        (b, addr)
    }

    fn queue(&self) -> Handle {
        let q = self.handle();
        self.ok(Request::CreateCommandQueue(CreateCommandQueue { queue: q }));
        q
    }

    fn commit(&self, queue: Handle, records: &[Command], resources: &[Handle]) -> Handle {
        let mut bytes = Vec::new();
        for r in records {
            r.encode(&mut bytes).unwrap();
        }
        let addr = self.alloc(bytes.len() as u64);
        self.guest.cpu_write(addr, &bytes);
        let cb = self.handle();
        self.ok(Request::Commit(Commit {
            command_buffer: cb,
            queue,
            label: None,
            retained_references: true,
            error_options: 0,
            stream: GuestAddr(addr),
            stream_length: bytes.len() as u64,
            resources: resources.to_vec(),
        }));
        cb
    }

    fn idle(&self) {
        self.g.block_on(self.exec.idle());
        for e in self.guest.posted.borrow().iter() {
            if let Event::CommandBufferCompleted(c) = e {
                assert_eq!(c.failure, failure::ok(), "{c:?}");
            }
        }
    }
}

fn blit(encoder: Handle, body: Vec<Command>) -> Vec<Command> {
    let mut v = vec![Command::BeginBlit(BeginBlit { encoder })];
    v.extend(body);
    v.push(Command::EncoderEndEncoding(EncoderEndEncoding {}));
    v
}

fn texture_descriptor(format: u64, w: u64, h: u64, storage: u64, usage: u64) -> TextureDescriptor {
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

fn region(w: u64, h: u64) -> Region {
    Region { origin: Origin { x: 0, y: 0, z: 0 }, size: Size { width: w, height: h, depth: 1 } }
}

/// replaceRegion and getBytes of shared textures through the executor, for every format with a reference
/// `upload-read` case, compared with the reference readback.
#[test]
fn texture_cpu_access_through_the_executor() {
    let h = Harness::new();
    let mut exact = 0;
    for value in webmetal_host::format::handling::all_values() {
        let Some(row) = webmetal_host::format::table::METAL_FORMATS.iter().find(|r| r.value == value) else { continue };
        let name = row.name.trim_start_matches("MTLPixelFormat");
        let case = format!("format.{name}.upload-read");
        if !Golden::exists(&case) {
            continue;
        }
        let golden = Golden::load(&case);
        let Some(input) = golden.bytes("input") else { continue };
        let (w, hh, bpr) = (
            golden.layout("input", "width").unwrap(),
            golden.layout("input", "height").unwrap(),
            golden.layout("input", "bytes_per_row").unwrap(),
        );
        let t = h.handle();
        let f = h.status(Request::CreateTexture(CreateTexture {
            texture: t,
            descriptor: texture_descriptor(u64::from(value), w, hh, 0, 0),
        }));
        if f.status != status::OK {
            // A format this host cannot provide: refused loudly with status 3.
            assert_eq!(f.status, status::HOST_UNSUPPORTED, "{name}: {f:?}");
            continue;
        }
        let src = h.alloc(input.len() as u64);
        h.guest.cpu_write(src, &input);
        let f = h.status(Request::ReplaceRegion(ReplaceRegion {
            texture: t,
            region: region(w, hh),
            level: 0,
            pixel_bytes: GuestAddr(src),
            bytes_per_row: bpr,
        }));
        // Depth32Float is written through a render pass, which keeps the reference's values (+0 and normal values of
        // [0, 1]); every format the host stores takes its bytes.
        assert_eq!(f.status, status::OK, "{name}: {f:?}");
        let dst = h.alloc(input.len() as u64);
        h.ok(Request::GetBytes(GetBytes {
            texture: t,
            pixel_bytes: GuestAddr(dst),
            bytes_per_row: bpr,
            region: region(w, hh),
            level: 0,
        }));
        assert_eq!(h.guest.cpu_read(dst, input.len()), golden.bytes("readback").unwrap(), "{name}");
        exact += 1;
    }
    eprintln!("{}: {exact} formats exact through the executor", h.g.name());
    assert!(exact >= 40, "{exact}");
    // A private texture has no CPU access: Metal does not define it, reported as status 2.
    let p = h.handle();
    h.ok(Request::CreateTexture(CreateTexture { texture: p, descriptor: texture_descriptor(70, 4, 4, 2, 1) }));
    let src = h.alloc(64);
    let f = h.status(Request::ReplaceRegion(ReplaceRegion {
        texture: p,
        region: region(4, 4),
        level: 0,
        pixel_bytes: GuestAddr(src),
        bytes_per_row: 16,
    }));
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    h.g.assert_no_errors();
}

/// `resource.purgeable-state` and the heap and aliasing rules through the executor.
#[test]
fn purgeable_state_and_heaps_through_the_executor() {
    let golden = Golden::load("resource.purgeable-state");
    let h = Harness::new();
    let set =
        |object: Handle, state: u64| match h.request(Request::SetPurgeableState(SetPurgeableState { object, state })) {
            Reply::SetPurgeableState(Ok(r)) => r.previous_state,
            other => panic!("{other:?}"),
        };
    let steps = [
        ("query-initial", 1),
        ("set-volatile", 3),
        ("query-volatile.volatile-or-empty", 1),
        ("set-nonvolatile.volatile-or-empty", 2),
        ("query-nonvolatile", 1),
        ("set-empty", 4),
        ("query-empty", 1),
        ("set-nonvolatile-after-empty", 2),
        ("query-final", 1),
    ];
    let (buffer, contents) = h.buffer(&[7u8; 64 * 1024], 0);
    let texture = h.handle();
    h.ok(Request::CreateTexture(CreateTexture { texture, descriptor: texture_descriptor(70, 64, 64, 2, 0) }));
    for (prefix, object) in [("shared-buffer", buffer), ("private-texture", texture)] {
        for (step, state) in steps {
            let got = set(object, state);
            let name = format!("{prefix}.{step}");
            match step.strip_suffix(".volatile-or-empty") {
                Some(_) => assert!(got == 3 || got == 4, "{name}: {got}"),
                None => assert_eq!(Some(got), golden.value_u64(&name), "{name}"),
            }
        }
    }
    // Emptying the shared buffer zeroed its contents.
    assert_eq!(h.guest.cpu_read(contents, 64 * 1024), vec![0u8; 64 * 1024]);
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
    assert_eq!(set(heap, 1), golden.value_u64("heap.query-initial").unwrap());
    assert_eq!(set(heap, 3), golden.value_u64("heap.set-volatile").unwrap());
    let v = set(heap, 2);
    assert!(v == 3 || v == 4);
    assert_eq!(set(heap, 1), golden.value_u64("heap.query-final").unwrap());
    let hb = h.handle();
    h.ok(Request::HeapCreateBuffer(HeapCreateBuffer {
        heap,
        buffer: hb,
        length: 4096,
        options: 0x20,
        contents: GuestAddr(0),
    }));
    h.ok(Request::MakeAliasable(MakeAliasable { resource: hb }));
    // makeAliasable of a buffer not in a heap is illegal (MTLResource.h): status 2.
    assert_eq!(h.status(Request::MakeAliasable(MakeAliasable { resource: buffer })).status, status::UNIMPLEMENTED);
    // Heap textures: the A10X's sizes are not in the device profile, refused loudly.
    let ht = h.handle();
    let f = h.status(Request::HeapCreateTexture(HeapCreateTexture {
        heap,
        texture: ht,
        descriptor: texture_descriptor(70, 16, 16, 2, 1),
    }));
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    h.g.assert_no_errors();
}

/// Random programs through the executor: CPU writes to bytes no committed command buffer uses, blit fills and copies
/// of shared buffers, compared with one shared copy of memory once the executor is idle.
#[test]
fn shared_buffers_through_the_executor() {
    let h = Harness::new();
    let q = h.queue();
    const LEN: u64 = 2048;
    let (a, a_addr) = h.buffer(&[0u8; LEN as usize], 0);
    let (b, b_addr) = h.buffer(&[0u8; LEN as usize], 0);
    let mut model = [vec![0u8; LEN as usize], vec![0u8; LEN as usize]];
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for round in 0..40 {
        // CPU writes at any alignment, then one command buffer, then wait: the CPU never races the GPU.
        for _ in 0..(next() % 6) {
            let which = (next() % 2) as usize;
            let at = next() % LEN;
            let end = (at + 1 + next() % 9).min(LEN);
            let bytes: Vec<u8> = (at..end).map(|_| next() as u8).collect();
            h.guest.cpu_write([a_addr, b_addr][which] + at, &bytes);
            model[which][at as usize..end as usize].copy_from_slice(&bytes);
        }
        let mut body = Vec::new();
        for _ in 0..(1 + next() % 3) {
            let lo = (next() % (LEN / 4)) * 4;
            let len = ((1 + next() % 64) * 4).min(LEN - lo);
            if next() % 2 == 0 {
                let value = next() as u8;
                let which = (next() % 2) as usize;
                body.push(Command::BlitFillBuffer(BlitFillBuffer {
                    buffer: Some([a, b][which]),
                    range: Range { location: lo, length: len },
                    value,
                }));
                model[which][lo as usize..(lo + len) as usize].fill(value);
            } else {
                let to = (next() % ((LEN - len) / 4 + 1)) * 4;
                body.push(Command::BlitCopyBufferToBuffer(BlitCopyBufferToBuffer {
                    source_buffer: Some(a),
                    source_offset: lo,
                    destination_buffer: Some(b),
                    destination_offset: to,
                    size: len,
                }));
                let src = model[0][lo as usize..(lo + len) as usize].to_vec();
                model[1][to as usize..(to + len) as usize].copy_from_slice(&src);
            }
        }
        // The resources are the objects the stream names, once each.
        let named: BTreeSet<Handle> = body
            .iter()
            .flat_map(|c| match c {
                Command::BlitFillBuffer(f) => vec![f.buffer.unwrap()],
                Command::BlitCopyBufferToBuffer(c) => vec![c.source_buffer.unwrap(), c.destination_buffer.unwrap()],
                _ => vec![],
            })
            .collect();
        h.commit(q, &blit(h.handle(), body), &named.into_iter().collect::<Vec<_>>());
        h.idle();
        assert_eq!(h.guest.cpu_read(a_addr, LEN as usize), model[0], "round {round}: buffer a");
        assert_eq!(h.guest.cpu_read(b_addr, LEN as usize), model[1], "round {round}: buffer b");
    }
    h.g.assert_no_errors();
}

/// `storage.iosurface-texture`: CPU writes to a surface reach a blit that reads its texture; a blit that writes the
/// texture reaches the CPU's view of the surface, as on real Metal.
#[test]
fn iosurface_texture_matches_metal() {
    let golden = Golden::load("storage.iosurface-texture");
    let h = Harness::new();
    let (w, hh) = (8u64, 4u64);
    let bpr = golden.value_u64("surface.bytes-per-row").unwrap();
    let id = SurfaceId(7);
    let plane = PlaneDesc {
        width: w,
        height: hh,
        bytes_per_row: bpr,
        bytes_per_element: 4,
        element_width: 1,
        element_height: 1,
        offset: 0,
        size: bpr * hh,
    };
    h.guest.surfaces.borrow_mut().push((
        SurfaceDesc {
            id,
            width: w,
            height: hh,
            bytes_per_row: bpr,
            bytes_per_element: 4,
            element_width: 1,
            element_height: 1,
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: bpr * hh,
            planes: vec![],
            seed: 1,
        },
        vec![0u8; (bpr * hh) as usize],
    ));
    let _ = plane;
    let t = h.handle();
    h.ok(Request::CreateTextureWithIOSurface(CreateTextureWithIOSurface {
        texture: t,
        descriptor: texture_descriptor(80, w, hh, 0, 1),
        iosurface: id,
        plane: 0,
    }));
    let (first, second) = (wm_pattern((w * hh * 4) as usize, 141), wm_pattern((w * hh * 4) as usize, 142));
    for y in 0..hh as usize {
        h.guest.surface_cpu_write(id, y * bpr as usize, &first[y * 32..(y + 1) * 32]);
    }
    let q = h.queue();
    let (out, out_addr) = h.buffer(&vec![0u8; first.len()], 0);
    let copy_out = Command::BlitCopyTextureToBuffer(BlitCopyTextureToBuffer {
        source_texture: Some(t),
        source_slice: 0,
        source_level: 0,
        source_origin: Origin { x: 0, y: 0, z: 0 },
        source_size: Size { width: w, height: hh, depth: 1 },
        destination_buffer: Some(out),
        destination_offset: 0,
        destination_bytes_per_row: w * 4,
        destination_bytes_per_image: w * hh * 4,
    });
    h.commit(q, &blit(h.handle(), vec![copy_out]), &[t, out]);
    h.idle();
    assert_eq!(h.guest.cpu_read(out_addr, first.len()), first, "gpu-reads-cpu-bytes");
    assert_eq!(golden.value_bool("gpu-reads-cpu-bytes"), Some(true));
    let (input, _) = h.buffer(&second, 0);
    let copy_in = Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
        source_buffer: Some(input),
        source_offset: 0,
        source_bytes_per_row: w * 4,
        source_bytes_per_image: w * hh * 4,
        source_size: Size { width: w, height: hh, depth: 1 },
        destination_texture: Some(t),
        destination_slice: 0,
        destination_level: 0,
        destination_origin: Origin { x: 0, y: 0, z: 0 },
    });
    h.commit(q, &blit(h.handle(), vec![copy_in]), &[t, input]);
    h.idle();
    let surface = h.guest.surface_bytes(id);
    let seen: Vec<u8> = (0..hh as usize).flat_map(|y| surface[y * bpr as usize..][..32].to_vec()).collect();
    assert_eq!(seen, golden.bytes("cpu-view-after-gpu-write").unwrap(), "cpu-view-after-gpu-write");
    assert_eq!(seen, second);
    assert_eq!(*h.guest.surface_writes.borrow(), vec![(id, 0)]);
    h.g.assert_no_errors();
}

/// A texture over a shared buffer (`storage.buffer-backed-texture`'s layout, offset and row stride at the A10X's
/// 64-byte linear alignment): the CPU's bytes in the buffer are what a blit reads from the texture, and a blit that
/// writes the texture lands in the buffer's bytes, with the bytes between rows untouched.
#[test]
fn buffer_backed_texture_through_the_executor() {
    let golden = Golden::load("storage.buffer-backed-texture");
    let h = Harness::new();
    let (w, hh) = (8u64, 4u64);
    let (offset, bpr) = (64u64, 64u64);
    let texels = wm_pattern((w * hh * 4) as usize, 131);
    // The reference read each texel's components as uints.
    let read: Vec<u8> = golden.bytes("read").unwrap().chunks(4).map(|c| c[0]).collect();
    assert_eq!(read, texels);
    let mut bytes = vec![0xeeu8; (offset + bpr * hh) as usize];
    for y in 0..hh as usize {
        bytes[offset as usize + y * bpr as usize..][..32].copy_from_slice(&texels[y * 32..(y + 1) * 32]);
    }
    let (buffer, addr) = h.buffer(&bytes, 0);
    let t = h.handle();
    h.ok(Request::CreateBufferTexture(CreateBufferTexture {
        buffer,
        texture: t,
        descriptor: texture_descriptor(73, w, hh, 0, 3),
        offset,
        bytes_per_row: bpr,
    }));
    let q = h.queue();
    let (out, out_addr) = h.buffer(&vec![0u8; texels.len()], 0);
    let copy_out = Command::BlitCopyTextureToBuffer(BlitCopyTextureToBuffer {
        source_texture: Some(t),
        source_slice: 0,
        source_level: 0,
        source_origin: Origin { x: 0, y: 0, z: 0 },
        source_size: Size { width: w, height: hh, depth: 1 },
        destination_buffer: Some(out),
        destination_offset: 0,
        destination_bytes_per_row: w * 4,
        destination_bytes_per_image: w * hh * 4,
    });
    h.commit(q, &blit(h.handle(), vec![copy_out]), &[t, out]);
    h.idle();
    assert_eq!(h.guest.cpu_read(out_addr, texels.len()), texels, "the texture reads the buffer's bytes");
    let written = golden.bytes("written-rows").unwrap();
    let (input, _) = h.buffer(&written, 0);
    let copy_in = Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
        source_buffer: Some(input),
        source_offset: 0,
        source_bytes_per_row: w * 4,
        source_bytes_per_image: w * hh * 4,
        source_size: Size { width: w, height: hh, depth: 1 },
        destination_texture: Some(t),
        destination_slice: 0,
        destination_level: 0,
        destination_origin: Origin { x: 0, y: 0, z: 0 },
    });
    h.commit(q, &blit(h.handle(), vec![copy_in]), &[t, input]);
    h.idle();
    let now = h.guest.cpu_read(addr, bytes.len());
    for y in 0..hh as usize {
        let row = offset as usize + y * bpr as usize;
        assert_eq!(now[row..row + 32], written[y * 32..(y + 1) * 32], "row {y}");
        assert!(now[row + 32..row + bpr as usize].iter().all(|&b| b == 0xee), "bytes after row {y}");
    }
    assert!(now[..offset as usize].iter().all(|&b| b == 0xee));
    h.g.assert_no_errors();
}
