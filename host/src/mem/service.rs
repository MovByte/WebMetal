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
//! | texture view (`CreateTextureView` and the five other view requests) | [`TextureView<A>`] |
//! | heap (`CreateHeap`) | `RefCell<`[`Heap`]`>` |
//!
//! A view's payload holds its root texture ([`super::view`]); a service that acts on a view's storage acts on the root
//! at the view's levels and slices, and notes writes there.
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
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

use crate::embed::{Embedder, TaskId};
use crate::exec::schema::command_handles;
use crate::exec::{
    Admit, Coherence, CommitInfo, ControlService, EncodeCx, Entry, Gate, Segment, Served, ServiceCx, Table, Verdict,
};
use crate::format::handling::{self, Resolution, Upload};
use crate::format::webgpu::{HostFeatures, WebGpuFormat};
use crate::gpu::{Api, GpuFuture};
use webmetal_wire::{
    BufferDidModifyRangeReply, Command, CountCounterSampleBufferReply, CreateBufferNoCopyReply, CreateBufferReply,
    CreateBufferTextureReply, CreateBufferWithBytesReply, CreateHeapReply, CreateLinearTextureReply,
    CreateTextureNoCopyReply, CreateTextureReply, CreateTextureViewLevelsSlicesReply,
    CreateTextureViewLevelsSlicesResourceIndexReply, CreateTextureViewLevelsSlicesSwizzleReply,
    CreateTextureViewLevelsSlicesSwizzleResourceIndexReply, CreateTextureViewReply,
    CreateTextureViewResourceIndexReply, CreateTextureWithIOSurfaceReply, Failure, GetBytesReply, GetBytesSliceReply,
    Handle, HeapCreateBufferReply, HeapCreateTextureReply, MakeAliasableReply, Region, ReplaceRegionReply,
    ReplaceRegionSliceReply, Reply, Request, SetPurgeableState, SetPurgeableStateReply,
    TextureLayoutCopyFromLinearBytesReply, TextureLayoutCopyFromTextureMemoryReply, opcode,
};

use super::agx::guest as agx_guest;
use super::buffer::{Buffer, Readbacks};
use super::calls::Calls;
use super::error::MemError;
use super::heap::Heap;
use super::hold::{Holds, Named};
use super::layout::RegionArgs;
use super::linear::LinearTexture;
use super::memory::Memory;
use super::nocopy::{self, GuestReadback};
use super::purgeable::Purgeable;
use super::surface::SurfaceReadback;
use super::texture::{
    GET_BYTES_CALL, GET_BYTES_SLICE_CALL, PvrtcReadback, REPLACE_REGION_CALL, REPLACE_REGION_SLICE_CALL, Texture,
    TextureBytes, TextureRead, get_bytes_image_pitch,
};
use super::view::{Parent, TextureView, ViewArgs};

/// A resource of a heap.
enum Member<A: Api> {
    Buffer(Weak<Buffer<A>>),
    Texture(Weak<Texture<A>>),
}

impl<A: Api> Clone for Member<A> {
    fn clone(&self) -> Member<A> {
        match self {
            Member::Buffer(b) => Member::Buffer(b.clone()),
            Member::Texture(t) => Member::Texture(t.clone()),
        }
    }
}

/// The resources of each heap, for a heap's purgeable state.
type HeapMembers<A> = Rc<RefCell<HashMap<(TaskId, Handle), Vec<Member<A>>>>>;

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
        Box::new(Control {
            memory: self.memory.clone(),
            embedder: self.embedder.clone(),
            heaps: self.heaps.clone(),
            counted: HashSet::new(),
        })
    }

    /// The coherence service.
    pub fn coherence(&self) -> Box<dyn Coherence<A, E>> {
        Box::new(Coherent {
            memory: self.memory.clone(),
            embedder: self.embedder.clone(),
            decoded: self.decoded.clone(),
            heaps: self.heaps.clone(),
            holds: Holds::default(),
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

/// The counter sample buffers the AGX kernel extension counts per system (AGXMTLCounterSampler, kernelcache
/// 0xfffffff005c31c2c: it refuses a 1025th).
const COUNTER_SAMPLE_BUFFERS: usize = 1024;

struct Control<A: Api, E: Embedder> {
    memory: Rc<Memory<A>>,
    embedder: Rc<E>,
    heaps: HeapMembers<A>,
    /// The buffers counted as counter sample storage (`CountCounterSampleBuffer`), of every task.
    counted: HashSet<(TaskId, Handle)>,
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
                    list.push(Member::Buffer(Rc::downgrade(&b)));
                }
                attach = b;
                ok!(HeapCreateBuffer, HeapCreateBufferReply)
            }
            Request::HeapCreateTexture(r) => {
                let heap = payload::<RefCell<Heap>>(objects, r.heap).ok_or_else(|| missing(r.heap))?;
                let t = mem.create_heap_texture(g, task, r.texture, &r.descriptor, &heap)?;
                if let Some(list) = self.heaps.borrow_mut().get_mut(&(task, r.heap)) {
                    list.push(Member::Texture(Rc::downgrade(&t)));
                }
                attach = t;
                ok!(HeapCreateTexture, HeapCreateTextureReply)
            }
            Request::CreateTexture(r) => {
                attach = mem.create_texture(g, task, r.texture, &r.descriptor)?;
                ok!(CreateTexture, CreateTextureReply)
            }
            Request::CreateTextureWithIOSurface(r) => {
                attach = mem.create_texture_iosurface(g, e, task, r)?;
                ok!(CreateTextureWithIOSurface, CreateTextureWithIOSurfaceReply)
            }
            Request::CreateTextureNoCopy(r) => {
                attach = mem.create_texture_no_copy(g, e, task, r)?;
                ok!(CreateTextureNoCopy, CreateTextureNoCopyReply)
            }
            Request::CreateTextureView(r) => {
                let a = ViewArgs {
                    call: "-[MTLTexture newTextureViewWithPixelFormat:]",
                    pixel_format: r.pixel_format,
                    texture_type: None,
                    levels: None,
                    slices: None,
                    swizzle: None,
                };
                attach = self.view(g, objects, task, r.texture, r.view, &a)?;
                ok!(CreateTextureView, CreateTextureViewReply)
            }
            Request::CreateTextureViewLevelsSlices(r) => {
                let a = ViewArgs {
                    call: "-[MTLTexture newTextureViewWithPixelFormat:textureType:levels:slices:]",
                    pixel_format: r.pixel_format,
                    texture_type: Some(r.texture_type),
                    levels: Some(&r.level_range),
                    slices: Some(&r.slice_range),
                    swizzle: None,
                };
                attach = self.view(g, objects, task, r.texture, r.view, &a)?;
                ok!(CreateTextureViewLevelsSlices, CreateTextureViewLevelsSlicesReply)
            }
            Request::CreateTextureViewLevelsSlicesSwizzle(r) => {
                let a = ViewArgs {
                    call: "-[MTLTexture newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:]",
                    pixel_format: r.pixel_format,
                    texture_type: Some(r.texture_type),
                    levels: Some(&r.level_range),
                    slices: Some(&r.slice_range),
                    swizzle: Some(&r.swizzle),
                };
                attach = self.view(g, objects, task, r.texture, r.view, &a)?;
                ok!(CreateTextureViewLevelsSlicesSwizzle, CreateTextureViewLevelsSlicesSwizzleReply)
            }
            Request::CreateTextureViewResourceIndex(r) => {
                let a = ViewArgs {
                    call: "-[MTLTextureSPI newTextureViewWithPixelFormat:resourceIndex:]",
                    pixel_format: r.pixel_format,
                    texture_type: None,
                    levels: None,
                    slices: None,
                    swizzle: None,
                };
                attach = self.view(g, objects, task, r.texture, r.view, &a)?;
                ok!(CreateTextureViewResourceIndex, CreateTextureViewResourceIndexReply)
            }
            Request::CreateTextureViewLevelsSlicesResourceIndex(r) => {
                let a = ViewArgs {
                    call: "-[MTLTextureSPI newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:]",
                    pixel_format: r.pixel_format,
                    texture_type: Some(r.texture_type),
                    levels: Some(&r.levels),
                    slices: Some(&r.slices),
                    swizzle: None,
                };
                attach = self.view(g, objects, task, r.texture, r.view, &a)?;
                ok!(CreateTextureViewLevelsSlicesResourceIndex, CreateTextureViewLevelsSlicesResourceIndexReply)
            }
            Request::CreateTextureViewLevelsSlicesSwizzleResourceIndex(r) => {
                let a = ViewArgs {
                    call: "-[MTLTextureSPI newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:]",
                    pixel_format: r.pixel_format,
                    texture_type: Some(r.texture_type),
                    levels: Some(&r.levels),
                    slices: Some(&r.slices),
                    swizzle: Some(&r.swizzle),
                };
                attach = self.view(g, objects, task, r.texture, r.view, &a)?;
                ok!(
                    CreateTextureViewLevelsSlicesSwizzleResourceIndex,
                    CreateTextureViewLevelsSlicesSwizzleResourceIndexReply
                )
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
                if let Some(t) = payload::<Texture<A>>(objects, r.texture).filter(|t| t.guest.is_some()) {
                    let (c, call) = nocopy::replace_region_copy(r);
                    mem.replace_region_guest(e, call, &t, &c, r.pixel_bytes.0)?;
                } else if let Some(t) = payload::<Texture<A>>(objects, r.texture) {
                    mem.replace_region(g, e, task, &t, r)?;
                } else if let Some(v) = payload::<TextureView<A>>(objects, r.texture).filter(|v| v.guest.is_some()) {
                    let (c, call) = nocopy::replace_region_copy(r);
                    let gv = v.guest.as_ref().expect("a view of a texture over caller memory");
                    mem.replace_region_guest_view(e, call, &v.root, gv, &c, r.pixel_bytes.0)?;
                } else if let Some(v) = payload::<TextureView<A>>(objects, r.texture) {
                    let a = args(&r.region, r.level, 0, r.bytes_per_row, 0);
                    mem.replace_region_view(g, e, (task, REPLACE_REGION_CALL), &v, &a, r.pixel_bytes.0)?;
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
                if let Some(t) = payload::<Texture<A>>(objects, r.texture).filter(|t| t.guest.is_some()) {
                    let (c, call) = nocopy::replace_region_slice_copy(r);
                    mem.replace_region_guest(e, call, &t, &c, r.pixel_bytes.0)?;
                } else if let Some(t) = payload::<Texture<A>>(objects, r.texture) {
                    mem.replace_region_slice(g, e, task, &t, r)?;
                } else if let Some(v) = payload::<TextureView<A>>(objects, r.texture).filter(|v| v.guest.is_some()) {
                    let (c, call) = nocopy::replace_region_slice_copy(r);
                    let gv = v.guest.as_ref().expect("a view of a texture over caller memory");
                    mem.replace_region_guest_view(e, call, &v.root, gv, &c, r.pixel_bytes.0)?;
                } else if let Some(v) = payload::<TextureView<A>>(objects, r.texture) {
                    let a = args(&r.region, r.level, r.slice, r.bytes_per_row, r.bytes_per_image);
                    mem.replace_region_view(g, e, (task, REPLACE_REGION_SLICE_CALL), &v, &a, r.pixel_bytes.0)?;
                } else {
                    let lt = payload::<LinearTexture<A>>(objects, r.texture).ok_or_else(|| missing(r.texture))?;
                    let a = args(&r.region, r.level, r.slice, r.bytes_per_row, r.bytes_per_image);
                    mem.replace_region_linear(e, task, &lt, &a, r.pixel_bytes.0)?;
                }
                return Ok(Served::Ready(ok!(ReplaceRegionSlice, ReplaceRegionSliceReply)));
            }
            Request::GetBytes(r) => {
                if let Some(t) = payload::<Texture<A>>(objects, r.texture).filter(|t| t.guest.is_some()) {
                    let (c, call) = nocopy::get_bytes_copy(r);
                    mem.get_bytes_guest(e, call, &t, &c, r.pixel_bytes.0)?;
                    return Ok(Served::Ready(ok!(GetBytes, GetBytesReply)));
                }
                if let Some(v) = payload::<TextureView<A>>(objects, r.texture).filter(|v| v.guest.is_some()) {
                    let (c, call) = nocopy::get_bytes_copy(r);
                    let gv = v.guest.as_ref().expect("a view of a texture over caller memory");
                    mem.get_bytes_guest_view(e, call, &v.root, gv, &c, r.pixel_bytes.0)?;
                    return Ok(Served::Ready(ok!(GetBytes, GetBytesReply)));
                }
                if let Some(t) = payload::<Texture<A>>(objects, r.texture) {
                    let read = mem.get_bytes(g, &t, r)?;
                    return Ok(self.later(task, read, r.pixel_bytes.0, |res| match res {
                        Ok(()) => ok!(GetBytes, GetBytesReply),
                        Err(e) => fail(opcode::GET_BYTES, e),
                    }));
                }
                let image_pitch = get_bytes_image_pitch(r);
                if let Some(v) = payload::<TextureView<A>>(objects, r.texture) {
                    let a = args(&r.region, r.level, 0, r.bytes_per_row, image_pitch);
                    let read = mem.get_bytes_view(g, GET_BYTES_CALL, &v, &a)?;
                    return Ok(self.later(task, read, r.pixel_bytes.0, |res| match res {
                        Ok(()) => ok!(GetBytes, GetBytesReply),
                        Err(e) => fail(opcode::GET_BYTES, e),
                    }));
                }
                let lt = payload::<LinearTexture<A>>(objects, r.texture).ok_or_else(|| missing(r.texture))?;
                let a = args(&r.region, r.level, 0, r.bytes_per_row, image_pitch);
                mem.get_bytes_linear(e, task, &lt, &a, r.pixel_bytes.0)?;
                return Ok(Served::Ready(ok!(GetBytes, GetBytesReply)));
            }
            Request::GetBytesSlice(r) => {
                if let Some(t) = payload::<Texture<A>>(objects, r.texture).filter(|t| t.guest.is_some()) {
                    let (c, call) = nocopy::get_bytes_slice_copy(r);
                    mem.get_bytes_guest(e, call, &t, &c, r.pixel_bytes.0)?;
                    return Ok(Served::Ready(ok!(GetBytesSlice, GetBytesSliceReply)));
                }
                if let Some(v) = payload::<TextureView<A>>(objects, r.texture).filter(|v| v.guest.is_some()) {
                    let (c, call) = nocopy::get_bytes_slice_copy(r);
                    let gv = v.guest.as_ref().expect("a view of a texture over caller memory");
                    mem.get_bytes_guest_view(e, call, &v.root, gv, &c, r.pixel_bytes.0)?;
                    return Ok(Served::Ready(ok!(GetBytesSlice, GetBytesSliceReply)));
                }
                if let Some(t) = payload::<Texture<A>>(objects, r.texture) {
                    let read = mem.get_bytes_slice(g, &t, r)?;
                    return Ok(self.later(task, read, r.pixel_bytes.0, |res| match res {
                        Ok(()) => ok!(GetBytesSlice, GetBytesSliceReply),
                        Err(e) => fail(opcode::GET_BYTES_SLICE, e),
                    }));
                }
                if let Some(v) = payload::<TextureView<A>>(objects, r.texture) {
                    let a = args(&r.region, r.level, r.slice, r.bytes_per_row, r.bytes_per_image);
                    let read = mem.get_bytes_view(g, GET_BYTES_SLICE_CALL, &v, &a)?;
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
            Request::TextureLayoutCopyFromLinearBytes(r) => {
                let refusal = agx_guest::copy_from_linear_bytes(e, task, r)?.map(str::to_owned);
                return Ok(Served::Ready(Reply::TextureLayoutCopyFromLinearBytes(Ok(
                    TextureLayoutCopyFromLinearBytesReply { refusal },
                ))));
            }
            Request::TextureLayoutCopyFromTextureMemory(r) => {
                let refusal = agx_guest::copy_from_texture_memory(e, task, r)?.map(str::to_owned);
                return Ok(Served::Ready(Reply::TextureLayoutCopyFromTextureMemory(Ok(
                    TextureLayoutCopyFromTextureMemoryReply { refusal },
                ))));
            }
            Request::CountCounterSampleBuffer(r) => {
                if payload::<Buffer<A>>(objects, r.buffer).is_none() {
                    return Err(missing(r.buffer));
                }
                if self.counted.contains(&(task, r.buffer)) {
                    return Err(MemError::invalid(format!("buffer {:#x} is counted already", r.buffer.get())));
                }
                let counted = self.counted.len() < COUNTER_SAMPLE_BUFFERS;
                if counted {
                    self.counted.insert((task, r.buffer));
                }
                return Ok(Served::Ready(Reply::CountCounterSampleBuffer(Ok(CountCounterSampleBufferReply {
                    counted,
                }))));
            }
            Request::SetPurgeableState(r) => {
                let previous = self.set_purgeable(cx, r)?;
                return Ok(Served::Ready(Reply::SetPurgeableState(Ok(SetPurgeableStateReply {
                    previous_state: previous,
                }))));
            }
            Request::MakeAliasable(r) => {
                // A view's dealloc makes its heap range aliasable, which is its root's range of the root's heap.
                let heap = match payload::<Buffer<A>>(objects, r.resource) {
                    Some(b) => b.heap_object.clone(),
                    None => match payload::<Texture<A>>(objects, r.resource) {
                        Some(t) => t.heap_object.clone(),
                        None => payload::<TextureView<A>>(objects, r.resource).and_then(|v| v.root.heap_object.clone()),
                    },
                };
                match heap {
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
            return match &t.heap_object {
                Some(heap) => Purgeable::set_on_heap_resource(heap.borrow().purgeable, r.state, CALL),
                None => self.memory.set_texture_purgeable(g, &t, r.state),
            };
        }
        if let Some(heap) = payload::<RefCell<Heap>>(objects, r.object) {
            let t = heap.borrow_mut().purgeable.set(r.state, "-[MTLHeap setPurgeableState:]")?;
            if t.discard {
                let members = self.heaps.borrow().get(&(cx.task(), r.object)).cloned().unwrap_or_default();
                for member in members.iter() {
                    match member {
                        Member::Buffer(b) => {
                            if let Some(b) = b.upgrade() {
                                self.memory.discard_buffer(g, e, &b)?;
                            }
                        }
                        Member::Texture(t) => {
                            if let Some(t) = t.upgrade() {
                                self.memory.discard_texture(g, &t)?;
                            }
                        }
                    }
                }
            }
            return Ok(t.previous);
        }
        if payload::<TextureView<A>>(objects, r.object).is_some() {
            return Err(MemError::invalid(format!(
                "{CALL} of a texture view: IOGPU makes a view not purgeable (purgeableAllowed NO), so the guest answers \
                 the call itself"
            )));
        }
        Err(MemError::unimplemented(format!("{CALL} of a texture over a buffer")))
    }

    /// A view of the texture or view `texture`.
    fn view(
        &self,
        g: &impl Calls<A>,
        objects: &Table,
        task: TaskId,
        texture: Handle,
        view: Handle,
        a: &ViewArgs<'_>,
    ) -> Result<Rc<dyn Any>, MemError> {
        let mem = &*self.memory;
        if let Some(t) = payload::<Texture<A>>(objects, texture) {
            return Ok(mem.create_texture_view(g, (task, view), Parent::Texture(&t), a)?);
        }
        if let Some(v) = payload::<TextureView<A>>(objects, texture) {
            return Ok(mem.create_texture_view(g, (task, view), Parent::View(&v), a)?);
        }
        if payload::<LinearTexture<A>>(objects, texture).is_some() {
            return Err(MemError::unimplemented(format!(
                "{} of a texture over a buffer: the A10X layout of its parent is not ported to the guest, which does \
                 not make the view",
                a.call
            )));
        }
        Err(MemError::invalid(format!("{}: handle {:#x} has no texture storage", a.call, texture.get())))
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
            opcode::CREATE_TEXTURE_VIEW,
            opcode::CREATE_TEXTURE_VIEW_LEVELS_SLICES,
            opcode::CREATE_TEXTURE_VIEW_LEVELS_SLICES_SWIZZLE,
            opcode::CREATE_TEXTURE_VIEW_RESOURCE_INDEX,
            opcode::CREATE_TEXTURE_VIEW_LEVELS_SLICES_RESOURCE_INDEX,
            opcode::CREATE_TEXTURE_VIEW_LEVELS_SLICES_SWIZZLE_RESOURCE_INDEX,
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
            opcode::TEXTURE_LAYOUT_COPY_FROM_LINEAR_BYTES,
            opcode::TEXTURE_LAYOUT_COPY_FROM_TEXTURE_MEMORY,
            opcode::COUNT_COUNTER_SAMPLE_BUFFER,
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
        self.counted.remove(&(task, handle));
    }

    fn task_exited(&mut self, task: TaskId) {
        self.heaps.borrow_mut().retain(|(t, _), _| *t != task);
        self.counted.retain(|(t, _)| *t != task);
    }
}

/// What a command buffer's use of a resource is called in failures.
const USE: &str = "a command buffer's use";

struct Coherent<A: Api, E: Embedder> {
    memory: Rc<Memory<A>>,
    embedder: Rc<E>,
    decoded: Decoded<A>,
    /// The resources of each heap: a heap an encoder makes resident stands for them.
    heaps: HeapMembers<A>,
    /// The write-backs whose dropped writes a later encoder could still read ([`super::hold`]).
    holds: Holds<A>,
}

impl<A: Api, E: Embedder> Coherent<A, E> {
    /// The resources the records `indices` of `records` name, with a view's root texture, a texture over a buffer's
    /// buffer and texture, and the resources allocated from a heap as it holds them now.
    fn named(&self, task: TaskId, objects: &Table, records: &[(usize, Command)], indices: &[usize]) -> Named<A> {
        let mut n = Named::default();
        for &i in indices {
            command_handles(&records[i].1, &mut |u| {
                let h = u.handle;
                if let Some(b) = payload::<Buffer<A>>(objects, h) {
                    n.buffer(b);
                } else if let Some(t) = payload::<Texture<A>>(objects, h) {
                    n.texture(t);
                } else if let Some(v) = payload::<TextureView<A>>(objects, h) {
                    n.texture(v.root.clone());
                } else if let Some(lt) = payload::<LinearTexture<A>>(objects, h) {
                    n.buffer(lt.buffer.clone());
                    if let Some(t) = &lt.texture {
                        n.texture(t.clone());
                    }
                } else if let Some(members) = self.heaps.borrow().get(&(task, h)) {
                    for m in members {
                        match m {
                            Member::Buffer(b) => {
                                if let Some(b) = b.upgrade() {
                                    n.buffer(b);
                                }
                            }
                            Member::Texture(t) => {
                                if let Some(t) = t.upgrade() {
                                    n.texture(t);
                                }
                            }
                        }
                    }
                }
            });
        }
        n
    }
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
                t.check_usable(USE).map_err(|e| e.failure())?;
                if t.surface.is_some() {
                    mem.prepare_surface_texture(g, e, &t).map_err(|e| e.failure())?;
                }
                if t.guest.is_some() {
                    mem.prepare_guest_texture(g, e, &t).map_err(|e| e.failure())?;
                }
            } else if let Some(v) = payload::<TextureView<A>>(objects, h) {
                v.root.check_usable(USE).map_err(|e| e.failure())?;
                if v.root.guest.is_some() {
                    mem.prepare_guest_texture(g, e, &v.root).map_err(|e| e.failure())?;
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

    fn admit(&mut self, cx: &mut EncodeCx<'_, A, E>, info: &CommitInfo, segment: &Segment) -> Result<Admit, Failure> {
        let uses = self.named(info.task, cx.objects(), cx.records(), &segment.records);
        if self.holds.holds((info.task, info.command_buffer), &uses) {
            return Ok(Admit::Wait);
        }
        // What dropped work left behind is made again from guest memory before the encoder's work; a poisoned
        // resource fails here.
        for t in &uses.textures {
            t.check_usable(USE).map_err(|e| e.failure())?;
        }
        let stale_buffers: Vec<&Rc<Buffer<A>>> = uses.buffers.iter().filter(|b| b.dropped()).collect();
        let stale_textures: Vec<&Rc<Texture<A>>> = uses.textures.iter().filter(|t| t.was_dropped()).collect();
        if stale_buffers.is_empty() && stale_textures.is_empty() {
            return Ok(Admit::Ready);
        }
        // The uploads land after everything the command buffer recorded so far.
        cx.submit();
        let (mem, g, e) = (&*self.memory, cx.gpu(), self.embedder.as_ref() as &dyn Embedder);
        for b in stale_buffers {
            mem.upload_buffer(g, e, b).map_err(|e| e.failure())?;
        }
        for t in stale_textures {
            if t.surface.is_some() {
                mem.prepare_surface_texture(g, e, t).map_err(|e| e.failure())?;
            }
            if t.guest.is_some() {
                mem.prepare_guest_texture(g, e, t).map_err(|e| e.failure())?;
            }
        }
        Ok(Admit::Ready)
    }

    fn write_back(
        &mut self,
        cx: &mut EncodeCx<'_, A, E>,
        info: &CommitInfo,
        gate: Gate,
    ) -> Result<Option<GpuFuture<Result<(), Failure>>>, Failure> {
        let mem = self.memory.clone();
        let encoder = cx.encoder().clone();
        let g = cx.gpu();
        let objects = cx.objects();
        let mut buffers = Vec::new();
        let mut surfaces: Vec<SurfaceReadback<A>> = Vec::new();
        let mut guests: Vec<GuestReadback<A>> = Vec::new();
        let mut decodes: Vec<(Rc<Texture<A>>, TextureRead<A>)> = Vec::new();
        let mut words: Vec<PvrtcReadback<A>> = Vec::new();
        // A texture named directly and through views of it is written back once.
        let mut textures: Vec<Rc<Texture<A>>> = Vec::new();
        for &h in &info.resources {
            let t = payload::<Texture<A>>(objects, h)
                .or_else(|| payload::<TextureView<A>>(objects, h).map(|v| v.root.clone()));
            if let Some(t) = t
                && !textures.iter().any(|u| Rc::ptr_eq(u, &t))
            {
                textures.push(t);
            }
        }
        // The textures without CPU-visible backing the work since the last write-back may have written: poisoned if
        // that work is dropped.
        let mut gpu_only: Vec<Rc<Texture<A>>> = Vec::new();
        // What the work could have written, for the encoders that use it later ([`super::hold`]).
        let mut written: Named<A> = Named::default();
        for t in &textures {
            let reads = (surfaces.len(), guests.len(), decodes.len(), words.len());
            if t.take_gpu_written() && t.surface.is_none() && t.guest.is_none() {
                gpu_only.push(t.clone());
                written.texture(t.clone());
            }
            if t.surface.is_some() {
                surfaces.extend(mem.read_back_surface_texture(g, &encoder, t).map_err(|e| e.failure())?);
            }
            if t.guest.is_some() {
                guests.extend(mem.read_back_guest_texture(g, &encoder, t).map_err(|e| e.failure())?);
            }
            // Any subresource a blit left stale: every level, every slice.
            let stale = (0..t.geometry.mip_levels)
                .any(|l| (0..t.geometry.slices()).any(|s| !t.stored_current(l, t.layer_of(s))));
            if stale && matches!(t.resolution, Resolution::Emulated(e) if matches!(e.upload, Upload::Decode(_))) {
                for read in mem.refresh_decoded(g, &encoder, t).map_err(|e| e.failure())? {
                    decodes.push((t.clone(), read));
                }
            }
            words.extend(mem.read_back_pvrtc_words(g, &encoder, t).map_err(|e| e.failure())?);
            if reads != (surfaces.len(), guests.len(), decodes.len(), words.len()) {
                written.texture(t.clone());
            }
        }
        for &h in &info.resources {
            if let Some(b) = payload::<Buffer<A>>(objects, h) {
                buffers.push(b);
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
        if readbacks.is_empty()
            && readbacks.private().is_empty()
            && gpu_only.is_empty()
            && surfaces.is_empty()
            && guests.is_empty()
            && decodes.is_empty()
            && words.is_empty()
        {
            return Ok(None);
        }
        for b in readbacks.written() {
            written.buffer(b.clone());
        }
        let settles = self.holds.record((info.task, info.command_buffer), gate.clone(), &written);
        let (embedder, decoded) = (self.embedder.clone(), self.decoded.clone());
        Ok(Some(Box::pin(async move {
            // The record settles when the future finishes or is dropped.
            let _settles = settles;
            // The buffers' readback resolves once every write-back encoded earlier has merged; its turn is held until
            // this write-back's other readbacks are merged too, so write-backs reach guest memory in queue order.
            let fail = |e: MemError| e.failure();
            let arrived = readbacks.wait().await.map_err(fail)?;
            let seq = arrived.seq();
            let mut surfaced = Vec::with_capacity(surfaces.len());
            for s in surfaces {
                surfaced.push(s.wait().await.map_err(fail)?);
            }
            let mut guested = Vec::with_capacity(guests.len());
            for r in guests {
                guested.push(r.wait(mem.clone()).await.map_err(fail)?);
            }
            let mut blocks = Vec::with_capacity(decodes.len());
            for (t, read) in decodes {
                blocks.push((t, read.wait().await.map_err(fail)?));
            }
            let mut kept = Vec::with_capacity(words.len());
            for r in words {
                kept.push(r.wait().await.map_err(fail)?);
            }
            let e = embedder.as_ref();
            match gate.verdict().await {
                Verdict::Merge => {
                    let _turn = arrived.merge(&mem, e).map_err(fail)?;
                    for a in surfaced {
                        a.merge(&mem, e, seq).map_err(fail)?;
                    }
                    for a in guested {
                        a.merge(e, seq).map_err(fail)?;
                    }
                    decoded.borrow_mut().extend(blocks);
                    for w in kept {
                        w.keep(&mem);
                    }
                }
                Verdict::Drop => {
                    // The work this write-back reads back followed the command buffer's first fault: none of its
                    // writes reach guest memory, and the GPU copies it could have written are made again from guest
                    // memory before their next use. What has no CPU-visible backing (private buffers, the other
                    // textures, and with them a decoded texture's blocks and a PVRTC subresource's words) keeps no
                    // copy of its contents from before that work: it is poisoned, and its next use fails.
                    let _turn = arrived.discard(&mem, e).map_err(fail)?;
                    for t in gpu_only {
                        t.poison();
                    }
                    for a in surfaced {
                        a.discard(&mem, seq);
                    }
                    for a in guested {
                        a.discard(seq);
                    }
                    for (_, b) in blocks {
                        b.release(&mem);
                    }
                    for w in kept {
                        w.discard(&mem);
                    }
                }
            }
            Ok(())
        })))
    }

    fn task_exited(&mut self, task: TaskId) {
        self.holds.task_exited(task);
    }
}
