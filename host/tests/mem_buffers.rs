//! Coherent buffers on a native WebGPU implementation (`WEBGPU_NATIVE`, see `host/build.rs`): the Metal reference
//! program `storage.coherence-interleaving` replayed through the shadow/diff protocol and compared with real Metal's
//! outputs, random programs checked against a shared-memory model, and the buffer creation forms.

#![cfg(feature = "native")]

mod mem_common;

use std::rc::Rc;

use mem_common::{Api, Golden, Gpu, Guest, TASK};
use webmetal_host::gpu::idl::{
    AutoLayoutMode, BindGroupDescriptor, BindGroupEntry, BindingResource, BufferBinding, CommandBufferDescriptor,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, PipelineLayoutOrAuto,
    ProgrammableStage, ShaderModuleDescriptor,
};
use webmetal_host::gpu::{
    BindingCommands as _, CommandEncoder as _, ComputePassEncoder as _, Device as _, PipelineBase as _, Size64,
};
use webmetal_host::mem::buffer::{Buffer, Readbacks, Upload};
use webmetal_host::mem::{MemError, Memory};
use webmetal_wire::{CreateBuffer, CreateBufferNoCopy, CreateBufferWithBytes, GuestAddr, Handle};

type A = Api;

fn create(handle: u64, length: u64, options: u64, contents: u64) -> CreateBuffer {
    CreateBuffer { buffer: Handle::new(handle).unwrap(), length, options, contents: GuestAddr(contents) }
}

fn with_bytes(handle: u64, pointer: u64, length: u64, options: u64, contents: u64) -> CreateBufferWithBytes {
    CreateBufferWithBytes {
        buffer: Handle::new(handle).unwrap(),
        pointer: GuestAddr(pointer),
        length,
        options,
        contents: GuestAddr(contents),
    }
}

fn no_copy(handle: u64, pointer: u64, length: u64) -> CreateBufferNoCopy {
    CreateBufferNoCopy {
        buffer: Handle::new(handle).unwrap(),
        pointer: GuestAddr(pointer),
        length,
        options: 0,
        deallocator: true,
    }
}

/// Compute kernels standing in for the reference program's GPU work: add one to each word, and fill each word.
const KERNELS: &str = "
@group(0) @binding(0) var<storage, read_write> words: array<u32>;
@group(0) @binding(1) var<storage, read> value: array<u32>;
@compute @workgroup_size(64) fn add_one(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < arrayLength(&words)) { words[id.x] = words[id.x] + 1u; }
}
@compute @workgroup_size(64) fn fill(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < arrayLength(&words)) { words[id.x] = value[0]; }
}
";

struct Kernels {
    add_one: <A as webmetal_host::gpu::Api>::ComputePipeline,
    fill: <A as webmetal_host::gpu::Api>::ComputePipeline,
}

impl Kernels {
    fn new(g: &Gpu) -> Kernels {
        let module = g.device.create_shader_module(&ShaderModuleDescriptor::new(KERNELS.to_string()));
        let pipeline = |entry: &str| {
            let mut stage = ProgrammableStage::new(&module);
            stage.entry_point = Some(entry.to_string());
            g.device.create_compute_pipeline(&ComputePipelineDescriptor::new(
                PipelineLayoutOrAuto::AutoLayoutMode(AutoLayoutMode::Auto),
                stage,
            ))
        };
        Kernels { add_one: pipeline("add_one"), fill: pipeline("fill") }
    }
}

/// The executor's side of one command buffer that runs a kernel over words `first..first + count` of `b`, the way
/// the executor drives the memory module.
struct Executor<'a> {
    g: &'a Gpu,
    mem: Memory<A>,
    kernels: Kernels,
    guest: &'a Guest,
    /// Submitted command buffers with the byte range their work writes.
    in_flight: Vec<(Readbacks<A>, (u64, u64))>,
}

impl<'a> Executor<'a> {
    fn new(g: &'a Gpu, guest: &'a Guest) -> Executor<'a> {
        Executor {
            g,
            mem: Memory::new(g.device.clone(), g.features),
            kernels: Kernels::new(g),
            guest,
            in_flight: Vec::new(),
        }
    }

    /// Upload, completing the command buffers the protocol says it must wait for.
    fn upload(&mut self, b: &Buffer<A>) {
        loop {
            match self.mem.upload_buffer(self.guest, b).unwrap() {
                Upload::Done => return,
                Upload::Wait(seqs) => {
                    while self.in_flight.iter().any(|(r, _)| r.seq().is_some_and(|s| seqs.contains(&s))) {
                        self.complete_oldest();
                    }
                }
            }
        }
    }

    fn commit(&mut self, b: &Rc<Buffer<A>>, first: u64, count: u64, fill: Option<u32>) {
        self.upload(b);
        let encoder = self.g.device.create_command_encoder(&CommandEncoderDescriptor::new());
        let pass = encoder
            .begin_compute_pass(&ComputePassDescriptor { label: String::new(), timestamp_writes: None })
            .unwrap();
        let pipeline = if fill.is_some() { &self.kernels.fill } else { &self.kernels.add_one };
        pass.set_pipeline(pipeline);
        let value = self.g.device.create_buffer(&webmetal_host::gpu::idl::BufferDescriptor::new(
            Size64::new_const(4),
            webmetal_host::gpu::idl::BufferUsage::STORAGE | webmetal_host::gpu::idl::BufferUsage::COPY_DST,
        ));
        let value = value.unwrap();
        use webmetal_host::gpu::Queue as _;
        self.g
            .device
            .queue()
            .write_buffer(&value, Size64::new_const(0), &fill.unwrap_or(0).to_le_bytes(), Size64::new_const(0), None)
            .unwrap();
        let mut entries = vec![BindGroupEntry::new(
            0,
            BindingResource::BufferBinding(BufferBinding {
                buffer: b.gpu(),
                offset: Size64::new(first * 4).unwrap(),
                size: Some(Size64::new(count * 4).unwrap()),
            }),
        )];
        if fill.is_some() {
            entries.push(BindGroupEntry::new(1, BindingResource::Buffer(&value)));
        }
        let layout = pipeline.get_bind_group_layout(0);
        let group = self.g.device.create_bind_group(&BindGroupDescriptor::new(&layout, entries));
        pass.set_bind_group(0, Some(&group), &[]);
        pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
        pass.end();
        self.mem.note_buffer_write(b, Some((first * 4, (first + count) * 4)));
        let mut readbacks = self.mem.encode_readbacks(&encoder, std::slice::from_ref(b)).unwrap();
        let cb = encoder.finish(&CommandBufferDescriptor::new());
        self.mem.submit(&[&cb], &mut readbacks);
        self.in_flight.push((readbacks, (first * 4, (first + count) * 4)));
    }

    /// Complete the oldest command buffer; returns the range its work wrote.
    fn complete_oldest(&mut self) -> (u64, u64) {
        let (r, range) = self.in_flight.remove(0);
        let arrived = self.g.block_on(r.wait()).unwrap();
        arrived.merge(&self.mem, self.guest).unwrap();
        range
    }

    fn submit_readback(&mut self, b: &Rc<Buffer<A>>) {
        self.mem.note_buffer_write(b, None);
        let encoder = self.g.device.create_command_encoder(&CommandEncoderDescriptor::new());
        let mut r = self.mem.encode_readbacks(&encoder, std::slice::from_ref(b)).unwrap();
        self.mem.submit(&[&encoder.finish(&CommandBufferDescriptor::new())], &mut r);
        self.in_flight.push((r, (0, b.length)));
    }

    fn wait_all(&mut self) {
        while !self.in_flight.is_empty() {
            self.complete_oldest();
        }
    }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// `storage.coherence-interleaving` of the Metal reference harness (tests/reference/src/cases/Storage.m), step for
/// step: CPU writes to blocks no in-flight command buffer touches, GPU increments and fills, commits and waits on one
/// shared buffer. Every snapshot, the trace and the final contents must equal real Metal's.
fn coherence_interleaving(g: &Gpu, page: Option<u64>) {
    const BLOCKS: u32 = 8;
    const BLOCK_WORDS: u32 = 64;
    const WORDS: u32 = BLOCKS * BLOCK_WORDS;
    let golden = Golden::load("storage.coherence-interleaving");
    assert_eq!(golden.status(), "ok");
    let guest = Guest::new(1 << 16, page);
    let mut ex = Executor::new(g, &guest);
    let seeds = [101u32, 202];
    for (program, &seed) in seeds.iter().enumerate() {
        let contents = guest.base + 0x1000 * program as u64;
        let b = ex.mem.create_buffer(&guest, TASK, &create(1, (WORDS * 4) as u64, 0, contents)).unwrap();
        let mut busy = 0u32;
        let mut state = seed;
        let mut snapshot = 0;
        let mut trace = Vec::new();
        for step in 0..24u32 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let r = state;
            let mut op = r % 4;
            let first = (r >> 4) % BLOCKS;
            let mut length = 1 + (r >> 8) % 3;
            if first + length > BLOCKS {
                length = BLOCKS - first;
            }
            let mask = ((1u32 << length) - 1) << first;
            if op == 0 && mask & busy != 0 {
                op = 3;
            }
            let (w0, wn) = ((first * BLOCK_WORDS) as u64, (length * BLOCK_WORDS) as u64);
            match op {
                0 => {
                    for i in w0..w0 + wn {
                        let value = (step << 24) | ((program as u32) << 16) | i as u32;
                        guest.cpu_write(contents + i * 4, &value.to_le_bytes());
                    }
                    trace.push(format!("cpu-write {first}+{length}"));
                }
                1 => {
                    ex.commit(&b, w0, wn, None);
                    busy |= mask;
                    trace.push(format!("gpu-add {first}+{length}"));
                }
                2 => {
                    let value = (r >> 16) as u8;
                    ex.commit(&b, w0, wn, Some(u32::from(value) * 0x0101_0101));
                    busy |= mask;
                    trace.push(format!("gpu-fill {first}+{length} 0x{value:02x}"));
                }
                _ => {
                    ex.wait_all();
                    busy = 0;
                    trace.push("wait".to_string());
                    let name = format!("program-{program}.snapshot-{snapshot}");
                    snapshot += 1;
                    let ours = guest.cpu_read(contents, (WORDS * 4) as usize);
                    let theirs = golden.bytes(&name).unwrap();
                    assert_eq!(words(&ours), words(&theirs), "{}: {name}", g.name());
                }
            }
        }
        ex.wait_all();
        trace.push("wait".to_string());
        assert_eq!(Some(trace), golden.strings(&format!("program-{program}.trace")), "{}: trace", g.name());
        let ours = guest.cpu_read(contents, (WORDS * 4) as usize);
        let theirs = golden.bytes(&format!("program-{program}.final")).unwrap();
        assert_eq!(words(&ours), words(&theirs), "{}: program-{program}.final", g.name());
    }
    g.assert_no_errors();
}

#[test]
fn coherence_interleaving_matches_metal() {
    let g = Gpu::new(true);
    coherence_interleaving(&g, None);
    coherence_interleaving(&g, Some(256));
}

/// Random legal programs on the real GPU: CPU writes of any length at any alignment to bytes no in-flight command
/// buffer uses, GPU increments and fills of word ranges, and completions in order, checked against one shared copy of
/// memory after every completion and at the end. With and without the embedder's write tracking.
#[test]
fn random_programs_match_shared_memory() {
    let g = Gpu::new(true);
    for (round, page) in [(0u64, None), (1, Some(64)), (2, Some(4096))] {
        let guest = Guest::new(1 << 16, page);
        let mut ex = Executor::new(&g, &guest);
        let mut rng = 0x2545_f491_4f6c_dd1du64 ^ round;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        const LEN: u64 = 4096;
        let contents = guest.base + 0x2000;
        let b = ex.mem.create_buffer(&guest, TASK, &create(1, LEN, 0, contents)).unwrap();
        let mut model = vec![0u8; LEN as usize];
        for _ in 0..300 {
            match next() % 6 {
                0 | 1 => {
                    let start = next() % LEN;
                    let end = (start + 1 + next() % 23).min(LEN);
                    if ex.in_flight.iter().any(|&(_, (a, z))| start < z && a < end) {
                        continue;
                    }
                    let bytes: Vec<u8> = (start..end).map(|_| next() as u8).collect();
                    guest.cpu_write(contents + start, &bytes);
                    model[start as usize..end as usize].copy_from_slice(&bytes);
                }
                2 | 3 => {
                    // Word ranges: the binding offset must be a multiple of minStorageBufferOffsetAlignment (256).
                    let first = (next() % (LEN / 256)) * 64;
                    let count = (1 + next() % 96).min(LEN / 4 - first);
                    let (lo, hi) = (first * 4, (first + count) * 4);
                    if ex.in_flight.iter().any(|&(_, (a, z))| lo < z && a < hi) {
                        continue;
                    }
                    let fill = (next() % 2 == 0).then(|| next() as u32);
                    ex.commit(&b, first, count, fill);
                    for w in first..first + count {
                        let o = (w * 4) as usize;
                        let v = match fill {
                            Some(f) => f,
                            None => u32::from_le_bytes(model[o..o + 4].try_into().unwrap()).wrapping_add(1),
                        };
                        model[o..o + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
                _ => {
                    if !ex.in_flight.is_empty() {
                        let (lo, hi) = ex.complete_oldest();
                        assert_eq!(
                            guest.cpu_read(contents + lo, (hi - lo) as usize),
                            model[lo as usize..hi as usize],
                            "{}: completed range {lo}..{hi}",
                            g.name()
                        );
                    }
                }
            }
        }
        ex.wait_all();
        assert_eq!(guest.cpu_read(contents, LEN as usize), model, "{}: round {round}", g.name());
    }
    g.assert_no_errors();
}

#[test]
fn creation_forms() {
    let g = Gpu::new(true);
    let guest = Guest::new(1 << 16, None);
    let mut ex = Executor::new(&g, &guest);
    // newBufferWithBytes: the initial bytes reach the GPU copy and the contents.
    let source = guest.base + 0x100;
    let pattern: Vec<u8> = (0..37u8).map(|i| i.wrapping_mul(29)).collect();
    guest.cpu_write(source, &pattern);
    let contents = guest.base + 0x1000;
    let b = ex.mem.create_buffer_with_bytes(&guest, TASK, &with_bytes(2, source, 37, 0, contents)).unwrap();
    assert_eq!(guest.cpu_read(contents, 37), pattern);
    ex.submit_readback(&b);
    ex.wait_all();
    assert_eq!(b.shadow_bytes().unwrap(), pattern);
    // Private storage with bytes: no CPU copy.
    let p = ex.mem.create_buffer_with_bytes(&guest, TASK, &with_bytes(3, source, 37, 0x20, 0)).unwrap();
    assert!(p.shadow_bytes().is_none());
    // A buffer whose contents already hold bytes uploads them at creation.
    let pre = guest.base + 0x3000;
    guest.cpu_write(pre, &[9; 64]);
    let c = ex.mem.create_buffer(&guest, TASK, &create(4, 64, 0, pre)).unwrap();
    assert_eq!(c.shadow_bytes().unwrap(), vec![9; 64]);
    // No-copy buffers need minBufferNoCopyAlignmentBytes alignment.
    let aligned = guest.base + 0x4000;
    assert!(ex.mem.create_buffer_no_copy(&guest, TASK, &no_copy(5, aligned, 4096)).is_ok());
    assert!(matches!(
        ex.mem.create_buffer_no_copy(&guest, TASK, &no_copy(6, aligned + 16, 4096)),
        Err(MemError::Undefined { .. })
    ));
    assert!(matches!(ex.mem.create_buffer(&guest, TASK, &create(7, 0, 0, contents)), Err(MemError::Undefined { .. })));
    assert!(matches!(ex.mem.create_buffer(&guest, TASK, &create(8, 16, 0x30, 0)), Err(MemError::Undefined { .. })));
    assert!(matches!(
        ex.mem.create_buffer(&guest, TASK, &create(9, 16, 0x10, contents)),
        Err(MemError::Undefined { .. })
    ));
    assert!(matches!(ex.mem.create_buffer(&guest, TASK, &create(10, 16, 0, 0)), Err(MemError::Invalid { .. })));
    g.assert_no_errors();
}

#[test]
fn purgeable_empty_discards() {
    let g = Gpu::new(true);
    let guest = Guest::new(1 << 16, None);
    let mut ex = Executor::new(&g, &guest);
    let contents = guest.base + 0x1000;
    let b = ex.mem.create_buffer(&guest, TASK, &create(1, 256, 0, contents)).unwrap();
    guest.cpu_write(contents, &[7; 256]);
    ex.commit(&b, 0, 64, None);
    // Empty while the readback is pending: zeroing waits for it.
    assert_eq!(ex.mem.set_buffer_purgeable(&guest, &b, 4).unwrap(), 2);
    ex.wait_all();
    assert_eq!(guest.cpu_read(contents, 256), vec![0; 256]);
    assert_eq!(b.shadow_bytes().unwrap(), vec![0; 256]);
    assert_eq!(ex.mem.set_buffer_purgeable(&guest, &b, 1).unwrap(), 4);
    assert_eq!(ex.mem.set_buffer_purgeable(&guest, &b, 2).unwrap(), 4);
    // The GPU copy is zero too: a readback of the whole buffer changes nothing.
    ex.submit_readback(&b);
    ex.wait_all();
    assert_eq!(guest.cpu_read(contents, 256), vec![0; 256]);
    g.assert_no_errors();
}
