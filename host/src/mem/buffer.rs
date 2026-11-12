//! Metal buffers: a WebGPU buffer, and for CPU-visible storage the guest pages and the [`Shadow`] between them.
//!
//! - `storageModePrivate`: GPU only; no shadow.
//! - `storageModeShared` (`newBufferWithLength:options:`, `newBufferWithBytes:length:options:`, heap buffers): the
//!   guest plugin allocates the CPU copy (`contents`) in the task; the shadow/diff protocol keeps it and the WebGPU
//!   buffer equal.
//! - `newBufferWithBytesNoCopy:length:options:deallocator:`: the same protocol over the caller's pages. The
//!   deallocator stays in the guest, which runs it once the buffer is released and no committed command buffer that
//!   uses it is outstanding; the executor never touches the pages after the last readback that the guest's
//!   completion waits for.
//!
//! The WebGPU buffer is the Metal length rounded up to whole words (WebGPU copies and writes move whole words) and
//! carries every usage a Metal buffer can be put to, so no later legal Metal call is refused for usage (plan section
//! 6.10). Its label names the Metal object, so an error a WebGPU call raises names it too.
//!
//! Coherence runs in queue order:
//!
//! 1. [`Memory::upload_buffer`] when a command buffer that uses the buffer is committed (the executor's
//!    [`crate::exec::Coherence::commit`]): the guest's changes reach the WebGPU buffer ahead of the work. Words the
//!    guest changed only in part go through a masked shader write while readbacks are pending (see
//!    [`super::shadow`]), so an upload never waits for the GPU.
//! 2. [`Memory::note_buffer_write`] for commands that may write the buffer (writable bindings per reflection, blit
//!    destinations, indirect command buffers) and [`Memory::note_buffer_read`] for buffers a command buffer only
//!    reads. A buffer a command buffer uses without either is read back whole: the safe answer when nothing says what
//!    its commands write.
//! 3. [`Memory::encode_readbacks`] at the end of the command buffer's encoder (or before a shared event signal): the
//!    copies take their queue position here, so the encoder must reach the queue before any other upload.
//! 4. [`Readbacks::wait`] and [`Arrived::merge`] once the GPU has run them, before the command buffer is reported
//!    completed.

use std::cell::RefCell;
use std::rc::Rc;

use crate::embed::{Embedder, TaskId};
use crate::gpu::idl::{
    AutoLayoutMode, BindGroupDescriptor, BindGroupEntry, BindingResource, BufferBinding, BufferDescriptor, BufferUsage,
    CommandBufferDescriptor, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
    PipelineLayoutOrAuto, ProgrammableStage, ShaderModuleDescriptor,
};
use crate::gpu::{
    Api, BindingCommands as _, Buffer as _, CommandEncoder as _, ComputePassEncoder as _, Device as _, GpuFuture,
    MappedRange as _, PipelineBase as _, Queue as _, Size64,
};
use webmetal_wire::{CreateBuffer, CreateBufferNoCopy, CreateBufferWithBytes, GuestAddr, Handle, HeapCreateBuffer};

use super::calls::Calls;
use super::error::MemError;
use super::heap::{Heap, HeapAllocation};
use super::memory::Memory;
use super::options::{ResourceOptions, StorageMode};
use super::purgeable::Purgeable;
use super::shadow::{MaskedWord, Shadow, UploadPlan, UploadWrite};

/// `minBufferNoCopyAlignmentBytes` of the A10X (`[getters]` of the device profile, both methods agreeing): the
/// alignment of a no-copy buffer's pointer and length.
pub const MIN_BUFFER_NO_COPY_ALIGNMENT: u64 = 4096;

/// Every usage a Metal buffer can need: copies, vertex, index, uniform, storage and indirect use, and counter
/// resolution.
pub const BUFFER_USAGE: u32 = BufferUsage::COPY_SRC
    | BufferUsage::COPY_DST
    | BufferUsage::INDEX
    | BufferUsage::VERTEX
    | BufferUsage::UNIFORM
    | BufferUsage::STORAGE
    | BufferUsage::INDIRECT
    | BufferUsage::QUERY_RESOLVE;

/// The masked write: each entry `(word, value, mask, 0)` replaces the bits of `mask` in `words[word]` with `value`'s.
const MASKED_WRITE: &str = "
@group(0) @binding(0) var<storage, read_write> words: array<u32>;
@group(0) @binding(1) var<storage, read> writes: array<vec4u>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&writes)) { return; }
    let w = writes[id.x];
    words[w.x] = (words[w.x] & ~w.z) | (w.y & w.z);
}
";

/// Where the CPU's copy of a buffer lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// No CPU copy.
    Private,
    /// Guest memory of `task` at `addr`.
    Guest { task: TaskId, addr: u64 },
}

/// What the command buffer being recorded is known to write of a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Writes {
    /// Nothing was said: the whole buffer is read back.
    Unknown,
    /// The hull of the bytes it may write (`None`: none).
    Known(Option<(u64, u64)>),
}

#[derive(Debug)]
struct State {
    shadow: Option<Shadow>,
    purgeable: Purgeable,
    writes: Writes,
    /// Contents emptied while readbacks were pending: zero them once the last one is merged.
    discard_when_idle: bool,
    /// The guest's bytes and the shadow were zeroed by a discard; the GPU copy is zeroed before its next use.
    zero_gpu: bool,
    /// Advances whenever the GPU copy may change: an upload that wrote bytes, GPU work that may write, a discard.
    generation: u64,
}

/// A Metal buffer: the payload of a buffer handle in the executor's object table.
pub struct Buffer<A: Api> {
    pub length: u64,
    pub options: ResourceOptions,
    /// The heap sub-allocation, for a heap buffer.
    pub heap: Option<HeapAllocation>,
    /// The heap it was allocated from: its purgeable state is the buffer's.
    pub heap_object: Option<Rc<RefCell<Heap>>>,
    pub backing: Backing,
    pub label: String,
    gpu: A::Buffer,
    state: RefCell<State>,
}

impl<A: Api> Buffer<A> {
    /// The WebGPU buffer (`length` rounded up to whole words).
    pub fn gpu(&self) -> &A::Buffer {
        &self.gpu
    }

    /// The purgeable state.
    pub fn purgeable(&self) -> Purgeable {
        self.state.borrow().purgeable
    }

    /// A counter that advances whenever the GPU copy may change (textures over the buffer compare it).
    pub fn generation(&self) -> u64 {
        self.state.borrow().generation
    }

    /// The bytes the guest and the GPU last agreed on (CPU-visible buffers).
    pub fn shadow_bytes(&self) -> Option<Vec<u8>> {
        self.state.borrow().shadow.as_ref().map(|s| s.bytes().to_vec())
    }
}

/// Readback copies encoded into a command encoder, at one queue position.
pub struct Readbacks<A: Api> {
    entries: Vec<Entry<A>>,
    seq: u64,
}

struct Entry<A: Api> {
    buffer: Rc<Buffer<A>>,
    staging: A::Buffer,
    offset: u64,
    size: u64,
}

/// Readback data, mapped and copied out, to merge into the guest.
pub struct Arrived<A: Api> {
    entries: Vec<(Entry<A>, Vec<u8>)>,
    seq: u64,
}

/// The label of a WebGPU object made for a Metal object, so that a WebGPU error names it.
pub(crate) fn label(task: TaskId, kind: &str, handle: Handle) -> String {
    format!("task {} {kind} {:#x}", task.0, handle.get())
}

pub(crate) fn size64(v: u64, op: &'static str) -> Result<Size64, MemError> {
    Size64::new(v).map_err(|e| MemError::gpu(op, e))
}

/// The guest's bytes of `len` bytes at `addr` that may have changed since the last call, as (offset, bytes) chunks:
/// the pages the embedder reports written, or everything when it does not track writes. The tracking is re-armed
/// before the bytes are read, so a write that lands after the read is reported next time.
fn changed_guest_chunks(e: &dyn Embedder, task: TaskId, addr: u64, len: u64) -> Result<Vec<(u64, Vec<u8>)>, MemError> {
    let mut out: Vec<(u64, Vec<u8>)> = Vec::new();
    match e.dirty_pages(task, addr, len)? {
        None => {
            let mut all = vec![0u8; len as usize];
            e.read_guest(task, addr, &mut all)?;
            out.push((0, all));
        }
        Some(d) => {
            let end = addr + len;
            let mut page = addr & !(d.page_size - 1);
            let mut run: Option<(u64, u64)> = None;
            while page < end {
                let (lo, hi) = (page.max(addr), (page + d.page_size).min(end));
                if d.is_dirty(page) {
                    run = Some(match run {
                        Some((s, _)) => (s, hi),
                        None => (lo, hi),
                    });
                } else if let Some((s, t)) = run.take() {
                    let mut b = vec![0u8; (t - s) as usize];
                    e.read_guest(task, s, &mut b)?;
                    out.push((s - addr, b));
                }
                page += d.page_size;
            }
            if let Some((s, t)) = run {
                let mut b = vec![0u8; (t - s) as usize];
                e.read_guest(task, s, &mut b)?;
                out.push((s - addr, b));
            }
        }
    }
    Ok(out)
}

impl<A: Api> Memory<A> {
    fn gpu_buffer(&self, g: &impl Calls<A>, length: u64, label: &str) -> Result<A::Buffer, MemError> {
        let padded = length.next_multiple_of(4);
        if let Some(max) = self.limits.max_buffer_size
            && padded > max
        {
            return Err(MemError::host(format!(
                "a {length}-byte buffer exceeds the device's maxBufferSize of {max} bytes"
            )));
        }
        let mut d = BufferDescriptor::new(size64(padded, "createBuffer")?, BUFFER_USAGE);
        d.label = label.to_string();
        g.call("GPUDevice.createBuffer", label, |dev| dev.create_buffer(&d))
            .map_err(|e| MemError::gpu("GPUDevice.createBuffer", e))
    }

    fn write_gpu(&self, g: &impl Calls<A>, b: &Buffer<A>, writes: &[UploadWrite]) -> Result<(), MemError> {
        for w in writes {
            let offset = size64(w.offset, "writeBuffer")?;
            g.call("GPUQueue.writeBuffer", &b.label, |_| {
                g.queue().write_buffer(&b.gpu, offset, &w.bytes, Size64::new_const(0), None)
            })
            .map_err(|e| MemError::gpu("GPUQueue.writeBuffer", e))?;
        }
        Ok(())
    }

    /// Write words of `b` in part: a compute pass that merges each word's masked bytes into the word as the GPU holds
    /// it at this point of the queue. Storage bindings start at a multiple of `minStorageBufferOffsetAlignment` and
    /// are at most `maxStorageBufferBindingSize` long, so the words go in windows the limits allow.
    fn write_masked(&self, g: &impl Calls<A>, b: &Buffer<A>, words: &[MaskedWord]) -> Result<(), MemError> {
        if words.is_empty() {
            return Ok(());
        }
        self.masked_words.set(self.masked_words.get() + words.len() as u64);
        let align = u64::from(self.limits.min_storage_buffer_offset_alignment.unwrap_or(256));
        let max_binding = self.limits.max_storage_buffer_binding_size.unwrap_or(128 << 20) & !3;
        let padded = b.length.next_multiple_of(4);
        if self.masked.borrow().is_none() {
            let module = g.call("GPUDevice.createShaderModule", "webmetal masked write", |d| {
                d.create_shader_module(&ShaderModuleDescriptor::new(MASKED_WRITE.to_string()))
            });
            let pipeline = g.call("GPUDevice.createComputePipeline", "webmetal masked write", |d| {
                d.create_compute_pipeline(&ComputePipelineDescriptor::new(
                    PipelineLayoutOrAuto::AutoLayoutMode(AutoLayoutMode::Auto),
                    ProgrammableStage::new(&module),
                ))
            });
            let layout = g.call("GPUPipelineBase.getBindGroupLayout", "webmetal masked write", |_| {
                pipeline.get_bind_group_layout(0)
            });
            *self.masked.borrow_mut() = Some((pipeline, layout));
        }
        let masked = self.masked.borrow();
        let (pipeline, layout) = masked.as_ref().expect("made above");
        let encoder = g.call("GPUDevice.createCommandEncoder", &b.label, |d| {
            d.create_command_encoder(&CommandEncoderDescriptor::new())
        });
        let mut at = 0;
        while at < words.len() {
            let start = words[at].offset / align * align;
            let size = max_binding.min(padded - start);
            let group: Vec<&MaskedWord> = words[at..].iter().take_while(|w| w.offset + 4 <= start + size).collect();
            at += group.len();
            let mut data = Vec::with_capacity(group.len() * 16);
            for w in &group {
                let (value, mask) = w.words();
                for v in [((w.offset - start) / 4) as u32, value, mask, 0] {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
            let d = BufferDescriptor::new(
                size64(data.len() as u64, "createBuffer")?,
                BufferUsage::STORAGE | BufferUsage::COPY_DST,
            );
            let list = g
                .call("GPUDevice.createBuffer", &b.label, |dev| dev.create_buffer(&d))
                .map_err(|e| MemError::gpu("GPUDevice.createBuffer", e))?;
            g.call("GPUQueue.writeBuffer", &b.label, |_| {
                g.queue().write_buffer(&list, Size64::new_const(0), &data, Size64::new_const(0), None)
            })
            .map_err(|e| MemError::gpu("GPUQueue.writeBuffer", e))?;
            let entries = vec![
                BindGroupEntry::new(
                    0,
                    BindingResource::BufferBinding(BufferBinding {
                        buffer: &b.gpu,
                        offset: size64(start, "createBindGroup")?,
                        size: Some(size64(size, "createBindGroup")?),
                    }),
                ),
                BindGroupEntry::new(1, BindingResource::Buffer(&list)),
            ];
            let bind = g.call("GPUDevice.createBindGroup", &b.label, |d| {
                d.create_bind_group(&BindGroupDescriptor::new(layout, entries))
            });
            let pass = g
                .call("GPUCommandEncoder.beginComputePass", &b.label, |_| {
                    encoder.begin_compute_pass(&ComputePassDescriptor { label: String::new(), timestamp_writes: None })
                })
                .map_err(|e| MemError::gpu("GPUCommandEncoder.beginComputePass", e))?;
            g.call("GPUComputePassEncoder.setPipeline", &b.label, |_| pass.set_pipeline(pipeline));
            g.call("GPUComputePassEncoder.setBindGroup", &b.label, |_| pass.set_bind_group(0, Some(&bind), &[]));
            let groups = (group.len() as u32).div_ceil(64);
            g.call("GPUComputePassEncoder.dispatchWorkgroups", &b.label, |_| pass.dispatch_workgroups(groups, 1, 1));
            g.call("GPUComputePassEncoder.end", &b.label, |_| pass.end());
        }
        let cb = g.call("GPUCommandEncoder.finish", &b.label, |_| encoder.finish(&CommandBufferDescriptor::new()));
        g.call("GPUQueue.submit", &b.label, |_| g.queue().submit(&[&cb]));
        Ok(())
    }

    /// Carry out an upload plan on the GPU and record it in the shadow.
    fn carry_out(&self, g: &impl Calls<A>, b: &Buffer<A>, st: &mut State, plan: &UploadPlan) -> Result<(), MemError> {
        if plan.is_empty() {
            return Ok(());
        }
        self.write_gpu(g, b, &plan.writes)?;
        self.write_masked(g, b, &plan.masked)?;
        let seq = self.next_seq();
        st.shadow.as_mut().expect("guest-backed buffers have a shadow").apply_upload(plan, seq);
        st.generation += 1;
        Ok(())
    }

    fn new_buffer(
        &self,
        g: &impl Calls<A>,
        (length, options): (u64, ResourceOptions),
        heap: Option<(HeapAllocation, Rc<RefCell<Heap>>)>,
        backing: Backing,
        shadow: Option<Shadow>,
        label: String,
    ) -> Result<Rc<Buffer<A>>, MemError> {
        let gpu = self.gpu_buffer(g, length, &label)?;
        let (heap, heap_object) = match heap {
            Some((a, h)) => (Some(a), Some(h)),
            None => (None, None),
        };
        Ok(Rc::new(Buffer {
            length,
            options,
            heap,
            heap_object,
            backing,
            label,
            gpu,
            state: RefCell::new(State {
                shadow,
                purgeable: Purgeable::NonVolatile,
                writes: Writes::Unknown,
                discard_when_idle: false,
                zero_gpu: false,
                generation: 0,
            }),
        }))
    }

    /// The first upload of a CPU-visible buffer: arm the write tracking and send whatever the guest holds.
    fn first_upload(&self, g: &impl Calls<A>, e: &dyn Embedder, b: &Buffer<A>) -> Result<(), MemError> {
        let Backing::Guest { task, addr } = b.backing else { return Ok(()) };
        let _ = e.dirty_pages(task, addr, b.length)?;
        let mut all = vec![0u8; b.length as usize];
        e.read_guest(task, addr, &mut all)?;
        let mut st = b.state.borrow_mut();
        let plan = st.shadow.as_ref().expect("guest-backed buffers have a shadow").plan_upload(&[(0, &all)]);
        self.carry_out(g, b, &mut st, &plan)
    }

    #[allow(clippy::too_many_arguments)]
    fn create_shared_or_private(
        &self,
        g: &impl Calls<A>,
        e: &dyn Embedder,
        task: TaskId,
        call: &'static str,
        (handle, length, options, contents): (Handle, u64, u64, GuestAddr),
        heap: Option<&Rc<RefCell<Heap>>>,
    ) -> Result<Rc<Buffer<A>>, MemError> {
        let options = ResourceOptions::decode(options, call)?;
        if length == 0 {
            // MTLDevice.h does not define a zero length; the debug layer rejects it ("Cannot create buffer of zero
            // length", storage.buffer-length-zero in the Metal reference outputs).
            return Err(MemError::undefined(call, "a buffer of length 0"));
        }
        let backing = self.backing_for(call, task, options.storage, contents.0)?;
        let allocation = match heap {
            Some(h) => Some((h.borrow_mut().allocate_buffer(length, &options)?, h.clone())),
            None => None,
        };
        let shadow = matches!(backing, Backing::Guest { .. }).then(|| Shadow::new(length, None));
        let b = self.new_buffer(g, (length, options), allocation, backing, shadow, label(task, "buffer", handle))?;
        self.first_upload(g, e, &b)?;
        Ok(b)
    }

    /// `newBufferWithLength:options:` (CreateBuffer). `contents` is the guest allocation backing a CPU-visible buffer,
    /// 0 for private storage.
    pub fn create_buffer(
        &self,
        g: &impl Calls<A>,
        e: &dyn Embedder,
        task: TaskId,
        r: &CreateBuffer,
    ) -> Result<Rc<Buffer<A>>, MemError> {
        let call = "-[MTLDevice newBufferWithLength:options:]";
        self.create_shared_or_private(g, e, task, call, (r.buffer, r.length, r.options, r.contents), None)
    }

    /// `-[MTLHeap newBufferWithLength:options:]` (HeapCreateBuffer), sub-allocated from `heap`.
    pub fn create_heap_buffer(
        &self,
        g: &impl Calls<A>,
        e: &dyn Embedder,
        task: TaskId,
        heap: &Rc<RefCell<Heap>>,
        r: &HeapCreateBuffer,
    ) -> Result<Rc<Buffer<A>>, MemError> {
        let call = "-[MTLHeap newBufferWithLength:options:]";
        self.create_shared_or_private(g, e, task, call, (r.buffer, r.length, r.options, r.contents), Some(heap))
    }

    fn backing_for(
        &self,
        call: &'static str,
        task: TaskId,
        storage: StorageMode,
        contents: u64,
    ) -> Result<Backing, MemError> {
        match (storage, contents) {
            (StorageMode::Shared, 0) => Err(MemError::invalid(format!("{call}: a shared buffer without contents"))),
            (StorageMode::Shared, addr) => Ok(Backing::Guest { task, addr }),
            (StorageMode::Private, 0) => Ok(Backing::Private),
            (StorageMode::Private, _) => Err(MemError::invalid(format!("{call}: a private buffer with contents"))),
            (StorageMode::Memoryless, _) => Err(MemError::undefined(call, "a buffer with MTLStorageModeMemoryless")),
        }
    }

    /// `newBufferWithBytes:length:options:` (CreateBufferWithBytes): the initial bytes are read from `pointer`
    /// during the call; for shared storage the executor also fills `contents` with them, writing only the bytes that
    /// differ.
    pub fn create_buffer_with_bytes(
        &self,
        g: &impl Calls<A>,
        e: &dyn Embedder,
        task: TaskId,
        r: &CreateBufferWithBytes,
    ) -> Result<Rc<Buffer<A>>, MemError> {
        const CALL: &str = "-[MTLDevice newBufferWithBytes:length:options:]";
        let (length, pointer, contents) = (r.length, r.pointer.0, r.contents.0);
        let options = ResourceOptions::decode(r.options, CALL)?;
        if length == 0 {
            return Err(MemError::undefined(CALL, "a buffer of length 0"));
        }
        let backing = self.backing_for(CALL, task, options.storage, contents)?;
        let mut initial = vec![0u8; length as usize];
        e.read_guest(task, pointer, &mut initial)?;
        let shadow = matches!(backing, Backing::Guest { .. }).then(|| Shadow::new(length, Some(&initial)));
        let b = self.new_buffer(g, (length, options), None, backing, shadow, label(task, "buffer", r.buffer))?;
        let mut padded = initial.clone();
        padded.resize(length.next_multiple_of(4) as usize, 0);
        self.write_gpu(g, &b, &[UploadWrite { offset: 0, bytes: padded }])?;
        self.next_seq();
        if let Backing::Guest { task, addr } = backing {
            let _ = e.dirty_pages(task, addr, length)?;
            let mut held = vec![0u8; length as usize];
            e.read_guest(task, addr, &mut held)?;
            e.write_guest_changed(task, addr, &initial, &held)?;
        }
        Ok(b)
    }

    /// `newBufferWithBytesNoCopy:length:options:deallocator:` (CreateBufferNoCopy). The documentation requires a
    /// page-aligned pointer and a length that makes a page-aligned region; the A10X states its page alignment as
    /// `minBufferNoCopyAlignmentBytes` (4096).
    pub fn create_buffer_no_copy(
        &self,
        g: &impl Calls<A>,
        e: &dyn Embedder,
        task: TaskId,
        r: &CreateBufferNoCopy,
    ) -> Result<Rc<Buffer<A>>, MemError> {
        const CALL: &str = "-[MTLDevice newBufferWithBytesNoCopy:length:options:deallocator:]";
        let (pointer, length) = (r.pointer.0, r.length);
        let options = ResourceOptions::decode(r.options, CALL)?;
        if length == 0 {
            return Err(MemError::undefined(CALL, "a buffer of length 0"));
        }
        if !pointer.is_multiple_of(MIN_BUFFER_NO_COPY_ALIGNMENT) || !length.is_multiple_of(MIN_BUFFER_NO_COPY_ALIGNMENT)
        {
            return Err(MemError::undefined(
                CALL,
                format!(
                    "pointer {pointer:#x} and length {length:#x} must be multiples of minBufferNoCopyAlignmentBytes \
                     ({MIN_BUFFER_NO_COPY_ALIGNMENT}); the documentation does not define other input"
                ),
            ));
        }
        if options.storage != StorageMode::Shared {
            return Err(MemError::undefined(CALL, "a no-copy buffer whose storage mode is not shared"));
        }
        let backing = Backing::Guest { task, addr: pointer };
        let shadow = Some(Shadow::new(length, None));
        let b = self.new_buffer(g, (length, options), None, backing, shadow, label(task, "buffer", r.buffer))?;
        self.first_upload(g, e, &b)?;
        Ok(b)
    }

    /// Send the guest's changes to the GPU copy, ahead of the GPU work submitted next.
    pub fn upload_buffer(&self, g: &impl Calls<A>, e: &dyn Embedder, b: &Buffer<A>) -> Result<(), MemError> {
        if b.state.borrow().zero_gpu {
            self.clear_gpu(g, b)?;
            b.state.borrow_mut().zero_gpu = false;
        }
        let Backing::Guest { task, addr } = b.backing else { return Ok(()) };
        let chunks = changed_guest_chunks(e, task, addr, b.length)?;
        let views: Vec<(u64, &[u8])> = chunks.iter().map(|(o, v)| (*o, v.as_slice())).collect();
        let mut st = b.state.borrow_mut();
        let plan = st.shadow.as_ref().expect("guest-backed buffers have a shadow").plan_upload(&views);
        self.carry_out(g, b, &mut st, &plan)
    }

    /// GPU work recorded next may write bytes `range` (`None`: anywhere) of the buffer.
    pub fn note_buffer_write(&self, b: &Buffer<A>, range: Option<(u64, u64)>) {
        let mut st = b.state.borrow_mut();
        st.generation += 1;
        if b.backing == Backing::Private {
            return;
        }
        let (lo, hi) = range.unwrap_or((0, b.length));
        let (lo, hi) = (lo.min(b.length), hi.min(b.length));
        let hull = match st.writes {
            Writes::Known(Some((a, z))) if lo < hi => Some((a.min(lo), z.max(hi))),
            Writes::Known(h) if lo >= hi => h,
            _ if lo < hi => Some((lo, hi)),
            _ => None,
        };
        st.writes = Writes::Known(hull);
    }

    /// The command buffer being recorded uses the buffer and its commands that write it are all noted with
    /// [`Memory::note_buffer_write`] (a buffer it only reads needs no readback).
    pub fn note_buffer_read(&self, b: &Buffer<A>) {
        let mut st = b.state.borrow_mut();
        if st.writes == Writes::Unknown {
            st.writes = Writes::Known(None);
        }
    }

    /// Encode, at the end of `encoder`, the readback of every CPU-visible buffer of `buffers` that the recorded work
    /// may have written. The readbacks take their queue position now: `encoder` must be submitted before any other
    /// upload of this module reaches the queue.
    pub fn encode_readbacks(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        buffers: &[Rc<Buffer<A>>],
    ) -> Result<Readbacks<A>, MemError> {
        let seq = self.next_seq();
        let mut entries = Vec::new();
        for b in buffers {
            let window = {
                let mut st = b.state.borrow_mut();
                if b.backing == Backing::Private {
                    continue;
                }
                match std::mem::replace(&mut st.writes, Writes::Unknown) {
                    Writes::Unknown => Some((0, b.length)),
                    Writes::Known(h) => h,
                }
            };
            let Some((lo, hi)) = window else { continue };
            let (offset, end) = (lo & !3, hi.next_multiple_of(4));
            let size = end - offset;
            let staging = self.staging.take(g, size)?;
            let (from, to, n) =
                (size64(offset, "copyBufferToBuffer")?, Size64::new_const(0), size64(size, "copyBufferToBuffer")?);
            g.call("GPUCommandEncoder.copyBufferToBuffer", &b.label, |_| {
                encoder.copy_buffer_to_buffer_with_offsets(&b.gpu, from, &staging, to, Some(n))
            });
            b.state.borrow_mut().shadow.as_mut().expect("guest-backed buffers have a shadow").begin_readback(seq);
            entries.push(Entry { buffer: b.clone(), staging, offset, size });
        }
        Ok(Readbacks { entries, seq })
    }

    /// `setPurgeableState:` of a buffer that is not in a heap (a heap buffer reflects its heap: see
    /// [`Purgeable::set_on_heap_resource`] and [`Memory::discard_buffer`]).
    pub fn set_buffer_purgeable(
        &self,
        g: &impl Calls<A>,
        e: &dyn Embedder,
        b: &Buffer<A>,
        state: u64,
    ) -> Result<u64, MemError> {
        if b.heap.is_some() {
            return Err(MemError::invalid("setPurgeableState: of a heap buffer goes through its heap"));
        }
        let t = b.state.borrow_mut().purgeable.set(state, "-[MTLResource setPurgeableState:]")?;
        if t.discard {
            self.discard_buffer(g, e, b)?;
        }
        Ok(t.previous)
    }

    /// Discard the buffer's contents (purgeable state empty): zero the GPU copy and the guest's bytes, so that nothing
    /// stale stays observable. With readbacks pending the zeroing waits until the last one is merged.
    pub fn discard_buffer(&self, g: &impl Calls<A>, e: &dyn Embedder, b: &Buffer<A>) -> Result<(), MemError> {
        let mut st = b.state.borrow_mut();
        if st.shadow.as_ref().is_some_and(|s| s.pending().next().is_some()) {
            st.discard_when_idle = true;
            return Ok(());
        }
        drop(st);
        self.clear_gpu(g, b)?;
        zero_cpu(e, b)
    }

    fn clear_gpu(&self, g: &impl Calls<A>, b: &Buffer<A>) -> Result<(), MemError> {
        let encoder = g.call("GPUDevice.createCommandEncoder", &b.label, |d| {
            d.create_command_encoder(&CommandEncoderDescriptor::new())
        });
        g.call("GPUCommandEncoder.clearBuffer", &b.label, |_| encoder.clear_buffer(&b.gpu, Size64::new_const(0), None));
        let cb = g.call("GPUCommandEncoder.finish", &b.label, |_| encoder.finish(&CommandBufferDescriptor::new()));
        g.call("GPUQueue.submit", &b.label, |_| g.queue().submit(&[&cb]));
        self.next_seq();
        b.state.borrow_mut().generation += 1;
        Ok(())
    }
}

/// Zero the guest's bytes and the shadow of a discarded buffer.
fn zero_cpu<A: Api>(e: &dyn Embedder, b: &Buffer<A>) -> Result<(), MemError> {
    let mut st = b.state.borrow_mut();
    st.discard_when_idle = false;
    if let (Backing::Guest { task, addr }, Some(shadow)) = (b.backing, st.shadow.as_mut()) {
        shadow.reset_zero();
        e.write_guest(task, addr, &vec![0u8; b.length as usize])?;
    }
    Ok(())
}

impl<A: Api> Readbacks<A> {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The queue position of the copies.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Map the staging buffers once the GPU has run the copies and take their bytes. The future makes no WebGPU call
    /// that can raise a device-timeline error.
    pub fn wait(self) -> GpuFuture<Result<Arrived<A>, MemError>> {
        Box::pin(async move {
            let mut out = Vec::with_capacity(self.entries.len());
            for entry in self.entries {
                let size = size64(entry.size, "mapAsync")?;
                entry
                    .staging
                    .map_async(MapMode::READ, Size64::new_const(0), Some(size))
                    .await
                    .map_err(|e| MemError::gpu("GPUBuffer.mapAsync", e))?;
                let range = entry
                    .staging
                    .get_mapped_range(Size64::new_const(0), Some(size))
                    .map_err(|e| MemError::gpu("GPUBuffer.getMappedRange", e))?;
                let mut data = vec![0u8; entry.size as usize];
                range.read(0, &mut data).map_err(|e| MemError::gpu("ArrayBuffer read", e))?;
                entry.staging.unmap();
                out.push((entry, data));
            }
            Ok(Arrived { entries: out, seq: self.seq })
        })
    }
}

impl<A: Api> Arrived<A> {
    /// The queue position of the readback.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Merge into the shadows and write the bytes the GPU changed into the guest. Makes no WebGPU call: a discard
    /// that waited for this readback zeroes the guest's bytes now and the GPU copy before its next use.
    pub fn merge(self, mem: &Memory<A>, e: &dyn Embedder) -> Result<(), MemError> {
        let mut first_error = None;
        for (entry, data) in self.entries {
            let b = &entry.buffer;
            let (writes, discard) = {
                let mut st = b.state.borrow_mut();
                let shadow = st.shadow.as_mut().expect("readbacks are of guest-backed buffers");
                let writes = shadow.merge_readback(self.seq, entry.offset, &data);
                let idle = shadow.pending().next().is_none();
                (writes, st.discard_when_idle && idle)
            };
            mem.staging.give_back(entry.staging);
            if let Backing::Guest { task, addr } = b.backing {
                for w in &writes {
                    if let Err(err) = e.write_guest_changed(task, addr + w.offset, &w.new, &w.old) {
                        first_error.get_or_insert(MemError::Guest(err));
                    }
                }
            }
            if discard {
                b.state.borrow_mut().zero_gpu = true;
                if let Err(err) = zero_cpu(e, b) {
                    first_error.get_or_insert(err);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
