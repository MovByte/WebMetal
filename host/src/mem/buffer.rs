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
//! 6.10). Its label is the caller's, so an error a WebGPU call raises names the Metal object.
//!
//! The executor drives coherence through four calls, in queue order:
//!
//! 1. [`Memory::upload_buffer`] before the first GPU use of the buffer in a command buffer (or in a segment of one
//!    that resumes after a wait): the guest's changes reach the WebGPU buffer ahead of the work.
//! 2. [`Memory::note_buffer_write`] for every command that may write the buffer (writable bindings per reflection,
//!    blit destinations, indirect command buffers).
//! 3. [`Memory::encode_readbacks`] at the end of the command encoder (or before a signal the CPU may wait on), then
//!    [`Memory::submit`].
//! 4. [`Readbacks::wait`], then [`Arrived::merge`] before the command buffer is reported completed.

use std::cell::RefCell;
use std::rc::Rc;

use crate::embed::{Embedder, TaskId};
use crate::gpu::idl::{BufferDescriptor, BufferUsage, CommandBufferDescriptor, CommandEncoderDescriptor, MapMode};
use crate::gpu::{Api, Buffer as _, CommandEncoder as _, Device as _, GpuFuture, MappedRange as _, Queue as _, Size64};

use webmetal_wire::{CreateBuffer, CreateBufferNoCopy, CreateBufferWithBytes, GuestAddr, Handle, HeapCreateBuffer};

use super::error::MemError;
use super::heap::{Heap, HeapAllocation};
use super::memory::Memory;
use super::options::{ResourceOptions, StorageMode};
use super::purgeable::Purgeable;
use super::shadow::{Shadow, UploadPlan, UploadWrite};

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

/// Where the CPU's copy of a buffer lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// No CPU copy.
    Private,
    /// Guest memory of `task` at `addr`.
    Guest { task: TaskId, addr: u64 },
}

#[derive(Debug)]
struct State {
    shadow: Option<Shadow>,
    purgeable: Purgeable,
    /// The hull of the bytes GPU work since the last readback may have written.
    written: Option<(u64, u64)>,
    /// Contents emptied while readbacks were pending: zero them once the last one is merged.
    discard_when_idle: bool,
    /// Ranges the write tracking reported (and re-armed) for an upload that then had to wait: the next upload reads
    /// them again.
    unconsumed: Vec<(u64, u64)>,
}

/// A Metal buffer.
pub struct Buffer<A: Api> {
    pub length: u64,
    pub options: ResourceOptions,
    /// The heap sub-allocation, for a heap buffer.
    pub heap: Option<HeapAllocation>,
    pub backing: Backing,
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

    /// The bytes the guest and the GPU last agreed on (CPU-visible buffers).
    pub fn shadow_bytes(&self) -> Option<Vec<u8>> {
        self.state.borrow().shadow.as_ref().map(|s| s.bytes().to_vec())
    }
}

/// What [`Memory::upload_buffer`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Upload {
    /// The GPU copy holds the guest's bytes (or the buffer has no CPU copy).
    Done,
    /// The upload needs the readbacks with these queue positions merged first (see [`super::shadow`]); merge them
    /// and call again.
    Wait(Vec<u64>),
}

/// Readback copies encoded into a command encoder.
pub struct Readbacks<A: Api> {
    entries: Vec<Entry<A>>,
    seq: Option<u64>,
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

fn size64(v: u64, op: &'static str) -> Result<Size64, MemError> {
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
    fn gpu_buffer(&self, length: u64, label: &str) -> Result<A::Buffer, MemError> {
        let padded = length.next_multiple_of(4);
        if let Some(max) = self.limits.max_buffer_size
            && padded > max
        {
            return Err(MemError::host(format!(
                "a {length}-byte buffer exceeds the device's maxBufferSize of {max} bytes"
            )));
        }
        let d = BufferDescriptor {
            label: label.to_string(),
            ..BufferDescriptor::new(size64(padded, "createBuffer")?, BUFFER_USAGE)
        };
        self.device.create_buffer(&d).map_err(|e| MemError::gpu("GPUDevice.createBuffer", e))
    }

    fn write_gpu(&self, buffer: &A::Buffer, writes: &[UploadWrite]) -> Result<(), MemError> {
        for w in writes {
            self.queue
                .write_buffer(buffer, size64(w.offset, "writeBuffer")?, &w.bytes, Size64::new_const(0), None)
                .map_err(|e| MemError::gpu("GPUQueue.writeBuffer", e))?;
        }
        Ok(())
    }

    fn new_buffer(
        &self,
        length: u64,
        options: ResourceOptions,
        heap: Option<HeapAllocation>,
        backing: Backing,
        shadow: Option<Shadow>,
        label: &str,
    ) -> Result<Rc<Buffer<A>>, MemError> {
        let gpu = self.gpu_buffer(length, label)?;
        Ok(Rc::new(Buffer {
            length,
            options,
            heap,
            backing,
            gpu,
            state: RefCell::new(State {
                shadow,
                purgeable: Purgeable::NonVolatile,
                written: None,
                discard_when_idle: false,
                unconsumed: Vec::new(),
            }),
        }))
    }

    /// The first upload of a CPU-visible buffer: arm the write tracking and send whatever the guest holds.
    fn first_upload(&self, e: &dyn Embedder, b: &Buffer<A>) -> Result<(), MemError> {
        let Backing::Guest { task, addr } = b.backing else { return Ok(()) };
        let _ = e.dirty_pages(task, addr, b.length)?;
        let mut all = vec![0u8; b.length as usize];
        e.read_guest(task, addr, &mut all)?;
        let mut st = b.state.borrow_mut();
        let shadow = st.shadow.as_mut().expect("guest-backed buffers have a shadow");
        let UploadPlan::Writes(writes) = shadow.plan_upload(&[(0, &all)]) else {
            unreachable!("a new buffer has no pending readback")
        };
        self.write_gpu(&b.gpu, &writes)?;
        let seq = self.next_seq();
        shadow.apply_upload(&writes, seq);
        Ok(())
    }

    fn create_shared_or_private(
        &self,
        e: &dyn Embedder,
        task: TaskId,
        call: &'static str,
        (handle, length, options, contents): (Handle, u64, u64, GuestAddr),
        heap: Option<&mut Heap>,
    ) -> Result<Rc<Buffer<A>>, MemError> {
        let options = ResourceOptions::decode(options, call)?;
        if length == 0 {
            // MTLDevice.h does not define a zero length; the debug layer rejects it ("Cannot create buffer of zero
            // length", storage.buffer-length-zero in the Metal reference outputs).
            return Err(MemError::undefined(call, "a buffer of length 0"));
        }
        let backing = self.backing_for(call, task, options.storage, contents.0)?;
        let allocation = match heap {
            Some(h) => Some(h.allocate_buffer(length, &options)?),
            None => None,
        };
        let shadow = matches!(backing, Backing::Guest { .. }).then(|| Shadow::new(length, None));
        let b = self.new_buffer(length, options, allocation, backing, shadow, &label(task, "buffer", handle))?;
        self.first_upload(e, &b)?;
        Ok(b)
    }

    /// `newBufferWithLength:options:` (CreateBuffer). `contents` is the guest allocation backing a CPU-visible buffer,
    /// 0 for private storage.
    pub fn create_buffer(&self, e: &dyn Embedder, task: TaskId, r: &CreateBuffer) -> Result<Rc<Buffer<A>>, MemError> {
        let call = "-[MTLDevice newBufferWithLength:options:]";
        self.create_shared_or_private(e, task, call, (r.buffer, r.length, r.options, r.contents), None)
    }

    /// `-[MTLHeap newBufferWithLength:options:]` (HeapCreateBuffer), sub-allocated from `heap`.
    pub fn create_heap_buffer(
        &self,
        e: &dyn Embedder,
        task: TaskId,
        heap: &mut Heap,
        r: &HeapCreateBuffer,
    ) -> Result<Rc<Buffer<A>>, MemError> {
        let call = "-[MTLHeap newBufferWithLength:options:]";
        self.create_shared_or_private(e, task, call, (r.buffer, r.length, r.options, r.contents), Some(heap))
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
        let b = self.new_buffer(length, options, None, backing, shadow, &label(task, "buffer", r.buffer))?;
        let mut padded = initial.clone();
        padded.resize(length.next_multiple_of(4) as usize, 0);
        self.write_gpu(&b.gpu, &[UploadWrite { offset: 0, bytes: padded }])?;
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
        let label = label(task, "buffer", r.buffer);
        let b = self.new_buffer(length, options, None, backing, Some(Shadow::new(length, None)), &label)?;
        self.first_upload(e, &b)?;
        Ok(b)
    }

    /// Send the guest's changes to the GPU copy, ahead of the GPU work that is encoded next.
    pub fn upload_buffer(&self, e: &dyn Embedder, b: &Buffer<A>) -> Result<Upload, MemError> {
        let Backing::Guest { task, addr } = b.backing else { return Ok(Upload::Done) };
        let mut chunks = changed_guest_chunks(e, task, addr, b.length)?;
        let mut st = b.state.borrow_mut();
        for (offset, len) in std::mem::take(&mut st.unconsumed) {
            let mut bytes = vec![0u8; len as usize];
            e.read_guest(task, addr + offset, &mut bytes)?;
            chunks.push((offset, bytes));
        }
        let views: Vec<(u64, &[u8])> = chunks.iter().map(|(o, v)| (*o, v.as_slice())).collect();
        let shadow = st.shadow.as_mut().expect("guest-backed buffers have a shadow");
        match shadow.plan_upload(&views) {
            UploadPlan::Blocked(seqs) => {
                // The tracking was re-armed for these ranges without their changes being sent.
                st.unconsumed = chunks.iter().map(|(o, v)| (*o, v.len() as u64)).collect();
                Ok(Upload::Wait(seqs))
            }
            UploadPlan::Writes(writes) => {
                if !writes.is_empty() {
                    self.write_gpu(&b.gpu, &writes)?;
                    let seq = self.next_seq();
                    shadow.apply_upload(&writes, seq);
                }
                Ok(Upload::Done)
            }
        }
    }

    /// GPU work encoded next may write bytes `range` (`None`: anywhere) of the buffer.
    pub fn note_buffer_write(&self, b: &Buffer<A>, range: Option<(u64, u64)>) {
        if b.backing == Backing::Private {
            return;
        }
        let (lo, hi) = range.unwrap_or((0, b.length));
        let (lo, hi) = (lo.min(b.length), hi.min(b.length));
        if lo >= hi {
            return;
        }
        let mut st = b.state.borrow_mut();
        st.written = Some(match st.written {
            Some((a, z)) => (a.min(lo), z.max(hi)),
            None => (lo, hi),
        });
    }

    /// Encode the readback of every buffer of `buffers` that GPU work may have written since its last readback.
    pub fn encode_readbacks(
        &self,
        encoder: &A::CommandEncoder,
        buffers: &[Rc<Buffer<A>>],
    ) -> Result<Readbacks<A>, MemError> {
        let mut entries = Vec::new();
        for b in buffers {
            let Some((lo, hi)) = b.state.borrow_mut().written.take() else { continue };
            let (offset, end) = (lo & !3, hi.next_multiple_of(4));
            let size = end - offset;
            let staging = self.staging.take(&self.device, size)?;
            encoder.copy_buffer_to_buffer_with_offsets(
                &b.gpu,
                size64(offset, "copyBufferToBuffer")?,
                &staging,
                Size64::new_const(0),
                Some(size64(size, "copyBufferToBuffer")?),
            );
            entries.push(Entry { buffer: b.clone(), staging, offset, size });
        }
        Ok(Readbacks { entries, seq: None })
    }

    /// `setPurgeableState:` of a buffer that is not in a heap (a heap buffer reflects its heap: see
    /// [`Purgeable::set_on_heap_resource`] and [`Memory::discard_buffer`]).
    pub fn set_buffer_purgeable(&self, e: &dyn Embedder, b: &Buffer<A>, state: u64) -> Result<u64, MemError> {
        if b.heap.is_some() {
            return Err(MemError::invalid("setPurgeableState: of a heap buffer goes through its heap"));
        }
        let t = b.state.borrow_mut().purgeable.set(state, "-[MTLResource setPurgeableState:]")?;
        if t.discard {
            self.discard_buffer(e, b)?;
        }
        Ok(t.previous)
    }

    /// Discard the buffer's contents (purgeable state empty): zero the GPU copy and the guest's bytes, so that nothing
    /// stale stays observable. With readbacks pending the zeroing waits until the last one is merged.
    pub fn discard_buffer(&self, e: &dyn Embedder, b: &Buffer<A>) -> Result<(), MemError> {
        let mut st = b.state.borrow_mut();
        if st.shadow.as_ref().is_some_and(|s| s.pending().next().is_some()) {
            st.discard_when_idle = true;
            return Ok(());
        }
        drop(st);
        self.zero_buffer(e, b)
    }

    fn zero_buffer(&self, e: &dyn Embedder, b: &Buffer<A>) -> Result<(), MemError> {
        let encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::new());
        encoder.clear_buffer(&b.gpu, Size64::new_const(0), None);
        self.queue.submit(&[&encoder.finish(&CommandBufferDescriptor::new())]);
        self.next_seq();
        let mut st = b.state.borrow_mut();
        st.discard_when_idle = false;
        if let (Backing::Guest { task, addr }, Some(shadow)) = (b.backing, st.shadow.as_mut()) {
            shadow.reset_zero();
            e.write_guest(task, addr, &vec![0u8; b.length as usize])?;
        }
        Ok(())
    }
}

impl<A: Api> Readbacks<A> {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The queue position the copies were submitted at.
    pub fn seq(&self) -> Option<u64> {
        self.seq
    }

    pub(crate) fn submitted(&mut self, seq: u64) {
        assert!(self.seq.is_none(), "readbacks submitted twice");
        self.seq = Some(seq);
        for e in &self.entries {
            if let Some(s) = e.buffer.state.borrow_mut().shadow.as_mut() {
                s.begin_readback(seq);
            }
        }
    }

    /// Map the staging buffers once the GPU has run the copies and take their bytes.
    pub fn wait(self) -> GpuFuture<Result<Arrived<A>, MemError>> {
        Box::pin(async move {
            let seq = self.seq.expect("readbacks are waited for after Memory::submit");
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
            Ok(Arrived { entries: out, seq })
        })
    }
}

impl<A: Api> Arrived<A> {
    /// The queue position of the readback.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Merge into the shadows and write the bytes the GPU changed into the guest.
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
            if discard && let Err(err) = mem.zero_buffer(e, b) {
                first_error.get_or_insert(err);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
