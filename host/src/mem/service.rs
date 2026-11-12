//! The memory module as the executor's services: a [`ControlService`] for resource creation, CPU access and
//! purgeable state, and the [`Coherence`] service.
//!
//! **Payloads.** The object a request creates is bound in the task's object table with one of these payloads, which
//! other services read with [`Table::get`] and `downcast_ref`:
//!
//! | Handle kind | Payload |
//! | --- | --- |
//! | buffer (`CreateBuffer`, `CreateBufferWithBytes`, `CreateBufferNoCopy`, `HeapCreateBuffer`) | [`Buffer<A>`] |
//! | texture (`CreateTexture`, `CreateTextureWithIOSurface`) | [`Texture<A>`] |
//! | texture over a buffer (`CreateBufferTexture`, `CreateLinearTexture`) | [`LinearTexture<A>`] |
//! | heap (`CreateHeap`) | `RefCell<`[`Heap`]`>` |
//!
//! **What encoder services owe this module** (they record the passes; this module cannot see inside them):
//!
//! - every command that may write a buffer: [`Memory::note_buffer_write`] with the range when it is known; a buffer a
//!   command buffer only reads may be declared with [`Memory::note_buffer_read`] (a buffer with neither is read back
//!   whole, which is exact but costs a copy);
//! - every pass that writes a texture: [`Memory::note_texture_write`];
//! - before a pass uses a texture: [`Memory::prepare_texture`] (refreshes the stored texture after a blit into its
//!   twin; a decoded format refreshed in the same command buffer is refused, see below) and, for a texture over a
//!   buffer, [`Memory::prepare_linear`]; after a pass that writes a texture over a buffer: [`Memory::sync_linear`];
//! - blits between buffers and textures: [`Memory::encode_buffer_to_texture`] and
//!   [`Memory::encode_texture_to_buffer`], which keep the twins and the buffers' readback windows right.
//!
//! **Coherence.** At commit the guest's writes to every CPU-visible buffer the command buffer names reach its WebGPU
//! buffer ([`Memory::upload_buffer`]), IOSurface textures take the rows the CPU changed, and textures over buffers take
//! the buffer's bytes. Before completion (and before a shared event signal) the readbacks of written buffers and
//! IOSurface textures are recorded; their future writes the GPU's bytes into guest memory and surfaces. A
//! software-decoded texture filled by a blit gets its blocks read back too, and the decoded texels reach its stored
//! texture at the next commit.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use crate::embed::{Embedder, TaskId};
use crate::exec::{Coherence, CommitInfo, ControlService, EncodeCx, Entry, Served, ServiceCx, Table};
use crate::format::handling::{self, Resolution, Upload};
use crate::format::webgpu::{HostFeatures, WebGpuFormat};
use crate::gpu::{Api, GpuFuture};
use webmetal_wire::{
    BufferDidModifyRangeReply, CreateBufferNoCopyReply, CreateBufferReply, CreateBufferTextureReply,
    CreateBufferWithBytesReply, CreateHeapReply, CreateLinearTextureReply, CreateTextureReply,
    CreateTextureWithIOSurfaceReply, Failure, GetBytesReply, GetBytesSliceReply, Handle, HeapCreateBufferReply,
    MakeAliasableReply, Region, ReplaceRegionReply, ReplaceRegionSliceReply, Reply, Request, SetPurgeableState,
    SetPurgeableStateReply, opcode,
};

use super::buffer::{Buffer, Readbacks};
use super::error::MemError;
use super::heap::Heap;
use super::layout::RegionArgs;
use super::linear::LinearTexture;
use super::memory::Memory;
use super::purgeable::Purgeable;
use super::surface::SurfaceReadback;
use super::texture::{Texture, TextureBytes, TextureRead};

/// The buffers of each heap, for a heap's purgeable state.
type HeapMembers<A> = Rc<RefCell<HashMap<(TaskId, Handle), Vec<Weak<Buffer<A>>>>>>;

/// Decoded blocks read back after a blit, waiting for the next commit to reach their stored texture.
type Decoded<A> = Rc<RefCell<Vec<(Rc<Texture<A>>, TextureBytes<A>)>>>;

/// The memory module's services, sharing one [`Memory`].
pub struct MemoryServices<A: Api, E: Embedder> {
    memory: Rc<Memory<A>>,
    embedder: Rc<E>,
    decoded: Decoded<A>,
    /// The buffers of each heap, for a heap's purgeable state.
    heaps: HeapMembers<A>,
}

impl<A: Api, E: Embedder + 'static> MemoryServices<A, E> {
    /// Set up the services on `device`: run the texel copy test ([`super::proof`]) of every format this host stores
    /// a Metal format in whose texel copies WebGPU does not promise to keep bits, so that transfers through those
    /// formats can be checked synchronously later. The tests make WebGPU calls, so they run here, before the executor
    /// takes the device.
    pub fn new(device: &A::Device, features: HostFeatures, embedder: Rc<E>) -> GpuFuture<Result<Self, MemError>> {
        let memory = Rc::new(Memory::new(device, features));
        let device = device.clone();
        Box::pin(async move {
            for f in formats_to_prove(features) {
                let outcome = memory.prove_texel_copies(&device, f).await?;
                memory.record_proof(f, outcome);
            }
            Ok(MemoryServices {
                memory,
                embedder,
                decoded: Rc::new(RefCell::new(Vec::new())),
                heaps: Rc::new(RefCell::new(HashMap::new())),
            })
        })
    }

    pub fn memory(&self) -> Rc<Memory<A>> {
        self.memory.clone()
    }

    /// The control service.
    pub fn control(&self) -> Box<dyn ControlService<A, E>> {
        Box::new(Control { memory: self.memory.clone(), embedder: self.embedder.clone(), heaps: self.heaps.clone() })
    }

    /// The coherence service.
    pub fn coherence(&self) -> Box<dyn Coherence<A, E>> {
        Box::new(Coherent {
            memory: self.memory.clone(),
            embedder: self.embedder.clone(),
            decoded: self.decoded.clone(),
        })
    }
}

/// The WebGPU formats a Metal format is stored in on a host with `features` whose texel copies are not bit-exact by
/// the specification and that a texel copy test can check.
pub fn formats_to_prove(features: HostFeatures) -> Vec<WebGpuFormat> {
    let mut out: Vec<WebGpuFormat> = Vec::new();
    for value in handling::all_values() {
        let Ok(r) = handling::resolve(value, features) else { continue };
        let f = r.storage();
        let decoded = matches!(r, Resolution::Emulated(e) if matches!(e.upload, Upload::Decode(_)));
        if !decoded
            && !f.texel_copy_exact(None)
            && !f.is_compressed()
            && f.caps().aspects.is_empty()
            && !out.contains(&f)
        {
            out.push(f);
        }
    }
    out
}

fn payload<T: 'static>(objects: &Table, handle: Handle) -> Option<Rc<T>> {
    let p: Rc<dyn Any> = objects.get(handle)?.payload.clone()?;
    p.downcast::<T>().ok()
}

fn fail(opcode: u16, e: MemError) -> Reply {
    crate::exec::schema::failure_reply(opcode, e.failure()).expect("a control request")
}

struct Control<A: Api, E: Embedder> {
    memory: Rc<Memory<A>>,
    embedder: Rc<E>,
    heaps: HeapMembers<A>,
}

macro_rules! ok {
    ($variant:ident, $reply:ident) => {
        Reply::$variant(Ok($reply {}))
    };
}

impl<A: Api, E: Embedder + 'static> Control<A, E> {
    fn serve_now(&mut self, cx: &mut ServiceCx<'_, A, E>, request: &Request) -> Result<Served, MemError> {
        let mem = &*self.memory;
        let (g, e, task) = (cx.gpu(), self.embedder.as_ref() as &dyn Embedder, cx.task());
        let objects = cx.objects();
        let missing = |h: Handle| MemError::invalid(format!("handle {:#x} has no memory object", h.get()));
        let attach: Rc<dyn Any>;
        let reply = match request {
            Request::CreateBuffer(r) => {
                attach = mem.create_buffer(g, e, task, r)?;
                ok!(CreateBuffer, CreateBufferReply)
            }
            Request::CreateBufferWithBytes(r) => {
                attach = mem.create_buffer_with_bytes(g, e, task, r)?;
                ok!(CreateBufferWithBytes, CreateBufferWithBytesReply)
            }
            Request::CreateBufferNoCopy(r) => {
                attach = mem.create_buffer_no_copy(g, e, task, r)?;
                ok!(CreateBufferNoCopy, CreateBufferNoCopyReply)
            }
            Request::CreateHeap(r) => {
                attach = Rc::new(RefCell::new(Heap::new(&r.descriptor)?));
                self.heaps.borrow_mut().insert((task, r.heap), Vec::new());
                ok!(CreateHeap, CreateHeapReply)
            }
            Request::HeapCreateBuffer(r) => {
                let heap = payload::<RefCell<Heap>>(objects, r.heap).ok_or_else(|| missing(r.heap))?;
                let b = mem.create_heap_buffer(g, e, task, &heap, r)?;
                if let Some(list) = self.heaps.borrow_mut().get_mut(&(task, r.heap)) {
                    list.push(Rc::downgrade(&b));
                }
                attach = b;
                ok!(HeapCreateBuffer, HeapCreateBufferReply)
            }
            Request::HeapCreateTexture(r) => {
                let heap = payload::<RefCell<Heap>>(objects, r.heap).ok_or_else(|| missing(r.heap))?;
                heap.borrow_mut().allocate_texture()?;
                return Err(MemError::unimplemented("-[MTLHeap newTextureWithDescriptor:]"));
            }
            Request::CreateTexture(r) => {
                attach = mem.create_texture(g, task, r.texture, &r.descriptor)?;
                ok!(CreateTexture, CreateTextureReply)
            }
            Request::CreateTextureWithIOSurface(r) => {
                attach = mem.create_texture_iosurface(g, e, task, r)?;
                ok!(CreateTextureWithIOSurface, CreateTextureWithIOSurfaceReply)
            }
            Request::CreateTextureNoCopy(_) => {
                return Err(MemError::unimplemented(
                    "-[MTLDeviceSPI newTextureWithBytesNoCopy:length:descriptor:deallocator:]: the texel layout the \
                     A10X driver gives such a texture's bytes is not in the device profile",
                ));
            }
            Request::CreateBufferTexture(r) => {
                let b = payload::<Buffer<A>>(objects, r.buffer).ok_or_else(|| missing(r.buffer))?;
                attach = Rc::new(mem.create_buffer_texture(g, task, &b, r)?);
                ok!(CreateBufferTexture, CreateBufferTextureReply)
            }
            Request::CreateLinearTexture(r) => {
                let b = payload::<Buffer<A>>(objects, r.buffer).ok_or_else(|| missing(r.buffer))?;
                attach = Rc::new(mem.create_linear_texture(g, task, &b, r)?);
                ok!(CreateLinearTexture, CreateLinearTextureReply)
            }
            Request::ReplaceRegion(r) => {
                if let Some(t) = payload::<Texture<A>>(objects, r.texture) {
                    mem.replace_region(g, e, task, &t, r)?;
                } else {
                    let lt = payload::<LinearTexture<A>>(objects, r.texture).ok_or_else(|| missing(r.texture))?;
                    mem.replace_region_linear(
                        e,
                        task,
                        &lt,
                        &args(&r.region, r.level, 0, r.bytes_per_row, 0),
                        r.pixel_bytes.0,
                    )?;
                }
                return Ok(Served::Ready(ok!(ReplaceRegion, ReplaceRegionReply)));
            }
            Request::ReplaceRegionSlice(r) => {
                if let Some(t) = payload::<Texture<A>>(objects, r.texture) {
                    mem.replace_region_slice(g, e, task, &t, r)?;
                } else {
                    let lt = payload::<LinearTexture<A>>(objects, r.texture).ok_or_else(|| missing(r.texture))?;
                    let a = args(&r.region, r.level, r.slice, r.bytes_per_row, r.bytes_per_image);
                    mem.replace_region_linear(e, task, &lt, &a, r.pixel_bytes.0)?;
                }
                return Ok(Served::Ready(ok!(ReplaceRegionSlice, ReplaceRegionSliceReply)));
            }
            Request::GetBytes(r) => {
                if let Some(t) = payload::<Texture<A>>(objects, r.texture) {
                    let read = mem.get_bytes(g, &t, r)?;
                    return Ok(self.later(task, read, r.pixel_bytes.0, |res| match res {
                        Ok(()) => ok!(GetBytes, GetBytesReply),
                        Err(e) => fail(opcode::GET_BYTES, e),
                    }));
                }
                let lt = payload::<LinearTexture<A>>(objects, r.texture).ok_or_else(|| missing(r.texture))?;
                mem.get_bytes_linear(e, task, &lt, &args(&r.region, r.level, 0, r.bytes_per_row, 0), r.pixel_bytes.0)?;
                return Ok(Served::Ready(ok!(GetBytes, GetBytesReply)));
            }
            Request::GetBytesSlice(r) => {
                if let Some(t) = payload::<Texture<A>>(objects, r.texture) {
                    let read = mem.get_bytes_slice(g, &t, r)?;
                    return Ok(self.later(task, read, r.pixel_bytes.0, |res| match res {
                        Ok(()) => ok!(GetBytesSlice, GetBytesSliceReply),
                        Err(e) => fail(opcode::GET_BYTES_SLICE, e),
                    }));
                }
                let lt = payload::<LinearTexture<A>>(objects, r.texture).ok_or_else(|| missing(r.texture))?;
                let a = args(&r.region, r.level, r.slice, r.bytes_per_row, r.bytes_per_image);
                mem.get_bytes_linear(e, task, &lt, &a, r.pixel_bytes.0)?;
                return Ok(Served::Ready(ok!(GetBytesSlice, GetBytesSliceReply)));
            }
            Request::SetPurgeableState(r) => {
                let previous = self.set_purgeable(cx, r)?;
                return Ok(Served::Ready(Reply::SetPurgeableState(Ok(SetPurgeableStateReply {
                    previous_state: previous,
                }))));
            }
            Request::MakeAliasable(r) => {
                let b = payload::<Buffer<A>>(objects, r.resource);
                match b.as_ref().and_then(|b| b.heap_object.clone()) {
                    Some(heap) => heap.borrow().make_aliasable()?,
                    None => {
                        return Err(MemError::undefined(
                            "-[MTLResource makeAliasable]",
                            "It is illegal to call this method on a non heap-based resource (MTLResource.h)",
                        ));
                    }
                }
                return Ok(Served::Ready(ok!(MakeAliasable, MakeAliasableReply)));
            }
            Request::BufferDidModifyRange(r) => {
                // didModifyRange: tells a managed buffer's GPU copy which bytes the CPU changed; the shadow comparison
                // finds them on its own, so the call changes nothing (plan section 6.2).
                let b = payload::<Buffer<A>>(objects, r.buffer).ok_or_else(|| missing(r.buffer))?;
                if r.range.location.checked_add(r.range.length).is_none_or(|end| end > b.length) {
                    return Err(MemError::undefined("-[MTLBuffer didModifyRange:]", "a range outside the buffer"));
                }
                return Ok(Served::Ready(ok!(BufferDidModifyRange, BufferDidModifyRangeReply)));
            }
            other => {
                return Err(MemError::invalid(format!("opcode {:#06x} is not a memory request", other.opcode())));
            }
        };
        cx.attach(attach);
        Ok(Served::Ready(reply))
    }

    /// Finish a texture read once the GPU has run it.
    fn later(
        &self,
        task: TaskId,
        read: TextureRead<A>,
        addr: u64,
        reply: impl FnOnce(Result<(), MemError>) -> Reply + 'static,
    ) -> Served {
        let (memory, embedder) = (self.memory.clone(), self.embedder.clone());
        Served::Later(Box::pin(async move {
            let res = match read.wait().await {
                Ok(bytes) => bytes.write_guest(&memory, embedder.as_ref(), task, addr),
                Err(e) => Err(e),
            };
            reply(res)
        }))
    }

    fn set_purgeable(&mut self, cx: &mut ServiceCx<'_, A, E>, r: &SetPurgeableState) -> Result<u64, MemError> {
        const CALL: &str = "-[MTLResource setPurgeableState:]";
        let (g, e) = (cx.gpu(), self.embedder.as_ref() as &dyn Embedder);
        let objects = cx.objects();
        if let Some(b) = payload::<Buffer<A>>(objects, r.object) {
            return match &b.heap_object {
                Some(heap) => Purgeable::set_on_heap_resource(heap.borrow().purgeable, r.state, CALL),
                None => self.memory.set_buffer_purgeable(g, e, &b, r.state),
            };
        }
        if let Some(t) = payload::<Texture<A>>(objects, r.object) {
            return self.memory.set_texture_purgeable(g, &t, r.state);
        }
        if let Some(heap) = payload::<RefCell<Heap>>(objects, r.object) {
            let t = heap.borrow_mut().purgeable.set(r.state, "-[MTLHeap setPurgeableState:]")?;
            if t.discard {
                let members = self.heaps.borrow().get(&(cx.task(), r.object)).cloned().unwrap_or_default();
                for b in members.iter().filter_map(Weak::upgrade) {
                    self.memory.discard_buffer(g, e, &b)?;
                }
            }
            return Ok(t.previous);
        }
        Err(MemError::unimplemented(format!("{CALL} of a texture over a buffer")))
    }
}

fn args(region: &Region, level: u64, slice: u64, bytes_per_row: u64, bytes_per_image: u64) -> RegionArgs {
    RegionArgs {
        origin: (region.origin.x, region.origin.y, region.origin.z),
        size: (region.size.width, region.size.height, region.size.depth),
        level,
        slice,
        bytes_per_row,
        bytes_per_image,
    }
}

impl<A: Api, E: Embedder + 'static> ControlService<A, E> for Control<A, E> {
    fn opcodes(&self) -> Vec<u16> {
        vec![
            opcode::CREATE_BUFFER,
            opcode::CREATE_BUFFER_WITH_BYTES,
            opcode::CREATE_BUFFER_NO_COPY,
            opcode::CREATE_TEXTURE,
            opcode::CREATE_TEXTURE_WITH_IO_SURFACE,
            opcode::CREATE_TEXTURE_NO_COPY,
            opcode::CREATE_BUFFER_TEXTURE,
            opcode::CREATE_LINEAR_TEXTURE,
            opcode::REPLACE_REGION,
            opcode::REPLACE_REGION_SLICE,
            opcode::GET_BYTES,
            opcode::GET_BYTES_SLICE,
            opcode::SET_PURGEABLE_STATE,
            opcode::MAKE_ALIASABLE,
            opcode::CREATE_HEAP,
            opcode::HEAP_CREATE_BUFFER,
            opcode::HEAP_CREATE_TEXTURE,
            opcode::BUFFER_DID_MODIFY_RANGE,
        ]
    }

    fn serve(&mut self, cx: &mut ServiceCx<'_, A, E>, request: &Request) -> Served {
        match self.serve_now(cx, request) {
            Ok(s) => s,
            Err(e) => Served::Ready(fail(request.opcode(), e)),
        }
    }

    fn release(&mut self, task: TaskId, handle: Handle, _: &Entry) {
        self.heaps.borrow_mut().remove(&(task, handle));
    }

    fn task_exited(&mut self, task: TaskId) {
        self.heaps.borrow_mut().retain(|(t, _), _| *t != task);
    }
}

struct Coherent<A: Api, E: Embedder> {
    memory: Rc<Memory<A>>,
    embedder: Rc<E>,
    decoded: Decoded<A>,
}

impl<A: Api, E: Embedder + 'static> Coherence<A, E> for Coherent<A, E> {
    fn commit(&mut self, cx: &mut ServiceCx<'_, A, E>, info: &CommitInfo) -> Result<(), Failure> {
        let (mem, g, e) = (&*self.memory, cx.gpu(), self.embedder.as_ref() as &dyn Embedder);
        for (t, blocks) in self.decoded.borrow_mut().drain(..) {
            mem.finish_refresh(g, &t, blocks).map_err(|e| e.failure())?;
        }
        let objects = cx.objects();
        let mut linear = Vec::new();
        for &h in &info.resources {
            if let Some(b) = payload::<Buffer<A>>(objects, h) {
                mem.upload_buffer(g, e, &b).map_err(|e| e.failure())?;
            } else if let Some(t) = payload::<Texture<A>>(objects, h) {
                if t.surface.is_some() {
                    mem.prepare_surface_texture(g, e, &t).map_err(|e| e.failure())?;
                }
            } else if let Some(lt) = payload::<LinearTexture<A>>(objects, h) {
                mem.upload_buffer(g, e, &lt.buffer).map_err(|e| e.failure())?;
                linear.push(lt);
            }
        }
        if !linear.is_empty() {
            // The bytes of textures over buffers, as the buffers hold them now, ahead of the command buffer's work.
            let encoder = g.call("GPUDevice.createCommandEncoder", "webmetal linear textures", |d| {
                crate::gpu::Device::create_command_encoder(d, &crate::gpu::idl::CommandEncoderDescriptor::new())
            });
            for lt in &linear {
                mem.prepare_linear(g, &encoder, lt).map_err(|e| e.failure())?;
            }
            let cb = g.call("GPUCommandEncoder.finish", "webmetal linear textures", |_| {
                crate::gpu::CommandEncoder::finish(&encoder, &crate::gpu::idl::CommandBufferDescriptor::new())
            });
            g.call("GPUQueue.submit", "webmetal linear textures", |_| crate::gpu::Queue::submit(g.queue(), &[&cb]));
        }
        Ok(())
    }

    fn write_back(
        &mut self,
        cx: &mut EncodeCx<'_, A, E>,
        info: &CommitInfo,
    ) -> Result<Option<GpuFuture<Result<(), Failure>>>, Failure> {
        let mem = self.memory.clone();
        let encoder = cx.encoder().clone();
        let g = cx.gpu();
        let objects = cx.objects();
        let mut buffers = Vec::new();
        let mut surfaces: Vec<SurfaceReadback<A>> = Vec::new();
        let mut decodes: Vec<(Rc<Texture<A>>, TextureRead<A>)> = Vec::new();
        for &h in &info.resources {
            if let Some(b) = payload::<Buffer<A>>(objects, h) {
                buffers.push(b);
            } else if let Some(t) = payload::<Texture<A>>(objects, h) {
                if t.surface.is_some() {
                    surfaces.extend(mem.read_back_surface_texture(g, &encoder, &t).map_err(|e| e.failure())?);
                }
                let stale = (0..t.geometry.mip_levels).any(|l| !t.stored_current(l, 0));
                if stale && matches!(t.resolution, Resolution::Emulated(e) if matches!(e.upload, Upload::Decode(_))) {
                    for read in mem.refresh_decoded(g, &encoder, &t).map_err(|e| e.failure())? {
                        decodes.push((t.clone(), read));
                    }
                }
            } else if let Some(lt) = payload::<LinearTexture<A>>(objects, h) {
                mem.sync_linear(g, &encoder, &lt).map_err(|e| e.failure())?;
                buffers.push(lt.buffer.clone());
            }
        }
        // A buffer named directly and through a texture over it is read back once.
        let mut unique: Vec<Rc<Buffer<A>>> = Vec::with_capacity(buffers.len());
        for b in buffers {
            if !unique.iter().any(|u| Rc::ptr_eq(u, &b)) {
                unique.push(b);
            }
        }
        let readbacks: Readbacks<A> = mem.encode_readbacks(g, &encoder, &unique).map_err(|e| e.failure())?;
        if readbacks.is_empty() && surfaces.is_empty() && decodes.is_empty() {
            return Ok(None);
        }
        let (embedder, decoded) = (self.embedder.clone(), self.decoded.clone());
        Ok(Some(Box::pin(async move {
            let arrived = readbacks.wait().await.map_err(|e| e.failure())?;
            arrived.merge(&mem, embedder.as_ref()).map_err(|e| e.failure())?;
            for s in surfaces {
                let a = s.wait().await.map_err(|e| e.failure())?;
                a.merge(&mem, embedder.as_ref()).map_err(|e| e.failure())?;
            }
            for (t, read) in decodes {
                let blocks = read.wait().await.map_err(|e| e.failure())?;
                decoded.borrow_mut().push((t, blocks));
            }
            Ok(())
        })))
    }

    fn task_exited(&mut self, task: TaskId) {
        let _ = task;
    }
}
