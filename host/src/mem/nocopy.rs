//! Textures over caller memory: `-[MTLDeviceSPI newTextureWithBytesNoCopy:length:descriptor:deallocator:]`.
//!
//! On the A10X the texture's memory is the caller's, from `bytes` on, in the driver's layout for the descriptor
//! ([`super::agx`]): `-[AGXTexture initWithDevice:bytes:length:desc:deallocator:]` (0x21c9d54e0) builds the layout of
//! a device texture (`initImplWithDevice:...allowNPOT:YES ... isHeapBacked:NO`) and has IOGPU make the kernel resource
//! over the caller's pages (`-[IOGPUMetalTexture initWithDevice:descriptor:sysMemPointer:sysMemSize:sysMemLength:
//! sysMemRowBytes:args:argsSize:deallocator:]`, IOGPU 0x1fcb30400, with the layout's size as the system memory size).
//! The CPU reaches the texels through that memory, and `replaceRegion`/`getBytes` are the driver's `writeRegion` and
//! `readRegion` on it.
//!
//! Here the guest's memory stays what the CPU sees and a WebGPU texture holds the GPU's copy. They are kept equal at the
//! coherence points of buffers (plan section 6.2), one subresource (a level of a slice, or a level of a 3D texture) at
//! a time:
//!
//! - at the commit of a command buffer that names the texture ([`Memory::prepare_guest_texture`]), the memory is read
//!   and every subresource whose bytes differ from what the executor last saw is put in Metal's layout and written into
//!   the texture;
//! - after GPU work that wrote the texture ([`Memory::read_back_guest_texture`]), its subresources are read back, put
//!   in the driver's layout over the bytes last seen, and only the bytes that changed are written into the memory
//!   ([`Embedder::write_guest_changed`]), so CPU writes to other bytes meanwhile are kept.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::embed::{Embedder, TaskId};
use crate::gpu::{Api, GpuFuture};
use webmetal_wire::{CreateTextureNoCopy, GetBytes, GetBytesSlice, ReplaceRegion, ReplaceRegionSlice};

use super::agx::gen2::{self, TextureGen2};
use super::agx::guest::{copy_error, layout_fields, system_values};
use super::agx::region::{self, Linear, Outcome, Region, RegionCopy};
use super::calls::Calls;
use super::error::MemError;
use super::layout::{RegionArgs, RegionLayout};
use super::memory::Memory;
use super::options::StorageMode;
use super::texture::{
    GET_BYTES_CALL, GET_BYTES_SLICE_CALL, REPLACE_REGION_CALL, REPLACE_REGION_SLICE_CALL, Texture, TextureRead,
    get_bytes_image_pitch,
};

const CALL: &str = "-[MTLDeviceSPI newTextureWithBytesNoCopy:length:descriptor:deallocator:]";
const USE: &str = "a texture over caller memory";

/// One subresource: where its texels are in the driver's layout and in Metal's.
struct Subresource {
    level: u32,
    slice: u32,
    /// The driver's copy of the whole subresource to and from tight rows.
    copy: RegionCopy,
    /// The same bytes as a Metal transfer lays them out (tight).
    metal: RegionLayout,
    /// The bytes of the memory the subresource occupies: [first, end).
    texture: (u64, u64),
}

/// The caller memory a texture is, as the textures over it see it.
pub struct GuestTextureMemory {
    pub task: TaskId,
    pub addr: u64,
    /// The length the caller gave, which the deallocator receives; the driver does not read it for this layout.
    pub length: u64,
    /// The driver's layout of the texture.
    pub layout: TextureGen2,
    subresources: Vec<Subresource>,
    /// The memory's bytes (the layout's size) as the executor last saw them.
    shadow: RefCell<Vec<u8>>,
    /// The texture holds the shadow's contents.
    loaded: Cell<bool>,
}

/// A texture's subresources read back after GPU work wrote it, to put into its memory.
pub struct GuestReadback<A: Api> {
    texture: Rc<Texture<A>>,
    reads: Vec<TextureRead<A>>,
}

impl<A: Api> Memory<A> {
    /// `newTextureWithBytesNoCopy:length:descriptor:deallocator:` (CreateTextureNoCopy).
    pub fn create_texture_no_copy(
        &self,
        g: &impl Calls<A>,
        e: &dyn Embedder,
        task: TaskId,
        r: &CreateTextureNoCopy,
    ) -> Result<Rc<Texture<A>>, MemError> {
        let d = &r.descriptor;
        let system = system_values(r.page_size, r.iosurface_alignment);
        let layout = gen2::texture_layout(&layout_fields(d), false, system)
            .map_err(|u| copy_error(CALL, region::CopyError::Unported(u)))?;
        if !layout.valid {
            return Err(MemError::invalid(format!(
                "{CALL}: the descriptor has no A10X layout (initImplWithDevice: fails and the method answers nil)"
            )));
        }
        let storage = StorageMode::from_value(d.storage_mode, CALL)?;
        if storage == StorageMode::Memoryless {
            return Err(MemError::unimplemented(format!(
                "{CALL}: a memoryless texture over caller memory: what IOGPU makes of it is not traced"
            )));
        }
        if d.sample_count > 1 {
            return Err(MemError::unimplemented(format!(
                "{CALL}: a multisampled texture over caller memory: WebGPU copies multisampled textures only to \
                 textures, so the memory and the texture cannot be kept equal"
            )));
        }
        let alignment =
            gen2::texture_alignment(&layout).map_err(|u| copy_error(CALL, region::CopyError::Unported(u)))?;
        if alignment == 0 || !r.bytes.0.is_multiple_of(alignment) {
            return Err(MemError::unimplemented(format!(
                "{CALL}: memory at {:#x}, not a multiple of the layout's alignment {alignment}: how the kernel places a \
                 texture over such memory is not traced",
                r.bytes.0
            )));
        }
        let t = self.build_texture(g, CALL, (task, r.texture), d, None)?;
        let Some(blocks) = t.blocks else {
            return Err(MemError::unimplemented(format!(
                "{CALL}: a combined depth-stencil texture over caller memory: its bytes have no single texel size"
            )));
        };
        let mut subresources = Vec::new();
        let slices = if t.geometry.kind.is_3d() { 1 } else { t.geometry.slices() };
        for level in 0..t.geometry.mip_levels {
            let (w, h, depth) = t.geometry.level_extent(level);
            let depth = if t.geometry.kind.is_3d() { depth } else { 1 };
            // Tight rows of blocks, and images for 3D levels; PVRTC levels take both pitches 0.
            let (bx, by) = blocks.count(w, h);
            let row = u64::from(bx) * u64::from(blocks.bytes);
            let (bytes_per_row, bytes_per_image) = match (blocks.pvrtc, t.geometry.kind.is_3d()) {
                (Some(_), _) => (0, 0),
                (None, true) => (row, row * u64::from(by)),
                (None, false) => (row, 0),
            };
            for slice in 0..slices {
                let a = RegionArgs {
                    origin: (0, 0, 0),
                    size: (u64::from(w), u64::from(h), u64::from(depth)),
                    level: u64::from(level),
                    slice: u64::from(slice),
                    bytes_per_row,
                    bytes_per_image,
                };
                let metal = RegionLayout::new(CALL, &t.geometry, blocks, &a)?;
                let copy = RegionCopy {
                    region: Region { x: 0, y: 0, z: 0, width: w, height: h, depth },
                    slice,
                    level,
                    linear: if metal.pvrtc {
                        Linear::default()
                    } else {
                        Linear { bytes_per_row: metal.bytes_per_row, bytes_per_image: metal.bytes_per_image }
                    },
                };
                let extent = region::extent(&layout, &copy).map_err(|e| copy_error(CALL, e))?.ok_or_else(|| {
                    MemError::unimplemented(format!(
                        "{CALL}: the driver refuses to copy level {level} of slice {slice} of its own layout"
                    ))
                })?;
                // The driver's element copies must fill Metal's tight bytes exactly, once each.
                let mut spans = Vec::new();
                region::visit(&layout, &copy, &mut |_, lin, len| {
                    spans.push((lin, len));
                    Ok(())
                })
                .map_err(|e| copy_error(CALL, e))?;
                spans.sort_unstable();
                let mut at = 0;
                for (lin, len) in spans {
                    if lin != at {
                        at = u64::MAX;
                        break;
                    }
                    at += len;
                }
                if at != metal.tight_len() {
                    return Err(MemError::unimplemented(format!(
                        "{CALL}: the A10X driver's copy of level {level} of pixel format {} does not lay its bytes out as \
                         Metal's transfers do (their correspondence is not traced)",
                        d.pixel_format
                    )));
                }
                let texture = extent.texture.unwrap_or((0, 0));
                if texture.1 > layout.size {
                    return Err(MemError::unimplemented(format!(
                        "{CALL}: level {level} of slice {slice} lies past the layout's {} bytes",
                        layout.size
                    )));
                }
                subresources.push(Subresource { level, slice, copy, metal, texture });
            }
        }
        let size = usize::try_from(layout.size).map_err(|_| MemError::invalid(format!("{CALL}: layout size")))?;
        // The kernel resource is the caller's pages for the layout's size; memory that is not there makes it fail.
        let mut bytes = vec![0u8; size];
        e.read_guest(task, r.bytes.0, &mut bytes)?;
        let memory = GuestTextureMemory {
            task,
            addr: r.bytes.0,
            length: r.length,
            layout,
            subresources,
            shadow: RefCell::new(bytes),
            loaded: Cell::new(false),
        };
        let mut t = t;
        Rc::get_mut(&mut t).expect("a texture just made has no other owner").guest = Some(Rc::new(memory));
        Ok(t)
    }

    /// Before GPU work uses a texture over caller memory (at the commit of a command buffer that names it): write into
    /// the texture every subresource whose bytes changed since the executor last saw them.
    pub fn prepare_guest_texture(&self, g: &impl Calls<A>, e: &dyn Embedder, t: &Texture<A>) -> Result<(), MemError> {
        let m = t.guest.as_ref().expect("a texture over caller memory");
        let mut bytes = vec![0u8; m.layout.size as usize];
        e.read_guest(m.task, m.addr, &mut bytes)?;
        let loaded = m.loaded.get();
        let state = t.surface_state();
        {
            let shadow = m.shadow.borrow();
            for s in &m.subresources {
                let (first, end) = (s.texture.0 as usize, s.texture.1 as usize);
                if loaded && bytes[first..end] == shadow[first..end] {
                    continue;
                }
                let mut tight = vec![0u8; s.metal.tight_len() as usize];
                match region::read_region(&m.layout, &s.copy, &bytes, 0, &mut tight).map_err(|e| copy_error(USE, e))? {
                    Outcome::Copied => {}
                    Outcome::Refused(r) => unreachable!("checked at creation, refused with {r:?}"),
                }
                self.upload_region(g, USE, t, &s.metal, (s.level, s.slice), &tight)?;
            }
        }
        *m.shadow.borrow_mut() = bytes;
        m.loaded.set(true);
        t.set_surface_state(state.0, state.1);
        Ok(())
    }

    /// After GPU work that wrote a texture over caller memory: encode the reads of its subresources into `encoder`.
    pub fn read_back_guest_texture(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        t: &Rc<Texture<A>>,
    ) -> Result<Option<GuestReadback<A>>, MemError> {
        let (version, written) = t.surface_state();
        if !written {
            return Ok(None);
        }
        let m = t.guest.as_ref().expect("a texture over caller memory");
        let mut reads = Vec::with_capacity(m.subresources.len());
        for s in &m.subresources {
            reads.push(self.encode_read(g, USE, encoder, t, &s.metal, (s.level, s.slice))?);
        }
        t.set_surface_state(version, false);
        Ok(Some(GuestReadback { texture: t.clone(), reads }))
    }

    /// `replaceRegion:...` of a texture over caller memory: the driver's `writeRegion` on the memory.
    pub fn replace_region_guest(
        &self,
        e: &dyn Embedder,
        call: &'static str,
        t: &Texture<A>,
        c: &RegionCopy,
        pixel_bytes: u64,
    ) -> Result<(), MemError> {
        let m = t.guest.as_ref().expect("a texture over caller memory");
        if let Some(refusal) = region::validate_region(&m.layout, c).map_err(|e| copy_error(call, e))? {
            return Err(MemError::invalid(format!(
                "{call}: a region the A10X driver's validateRegion refuses ({}): the guest logs it and copies nothing, \
                 so it does not reach the host",
                refusal.assertion()
            )));
        }
        if pixel_bytes == 0 {
            return Err(MemError::undefined(call, "the bytes pointer is null (the driver reads through it)"));
        }
        let extent = region::extent(&m.layout, c).map_err(|e| copy_error(call, e))?.expect("validated");
        let mut linear = vec![0u8; extent.linear as usize];
        e.read_guest(m.task, pixel_bytes, &mut linear)?;
        let Some((first, end)) = extent.texture else { return Ok(()) };
        let mut held = vec![0u8; (end - first) as usize];
        e.read_guest(m.task, m.addr + first, &mut held)?;
        let mut new = held.clone();
        region::write_region(&m.layout, c, &mut new, first, &linear).map_err(|e| copy_error(call, e))?;
        e.write_guest_changed(m.task, m.addr + first, &new, &held)?;
        Ok(())
    }

    /// `getBytes:...` of a texture over caller memory: the driver's `readRegion` from the memory.
    pub fn get_bytes_guest(
        &self,
        e: &dyn Embedder,
        call: &'static str,
        t: &Texture<A>,
        c: &RegionCopy,
        pixel_bytes: u64,
    ) -> Result<(), MemError> {
        let m = t.guest.as_ref().expect("a texture over caller memory");
        if let Some(refusal) = region::validate_region(&m.layout, c).map_err(|e| copy_error(call, e))? {
            return Err(MemError::invalid(format!(
                "{call}: a region the A10X driver's validateRegion refuses ({}): the guest logs it and copies nothing, \
                 so it does not reach the host",
                refusal.assertion()
            )));
        }
        if pixel_bytes == 0 {
            return Err(MemError::undefined(call, "the bytes pointer is null (the driver reads through it)"));
        }
        let extent = region::extent(&m.layout, c).map_err(|e| copy_error(call, e))?.expect("validated");
        let mut held = vec![0u8; extent.linear as usize];
        e.read_guest(m.task, pixel_bytes, &mut held)?;
        let Some((first, end)) = extent.texture else { return Ok(()) };
        let mut memory = vec![0u8; (end - first) as usize];
        e.read_guest(m.task, m.addr + first, &mut memory)?;
        let mut new = held.clone();
        region::read_region(&m.layout, c, &memory, first, &mut new).map_err(|e| copy_error(call, e))?;
        e.write_guest_changed(m.task, pixel_bytes, &new, &held)?;
        Ok(())
    }
}

fn wire_region(r: &webmetal_wire::Region) -> Region {
    Region::from_mtl([r.origin.x, r.origin.y, r.origin.z], [r.size.width, r.size.height, r.size.depth])
}

/// `replaceRegion:mipmapLevel:withBytes:bytesPerRow:` is `-[IOGPUMetalTexture replaceRegion:...]` (IOGPU 0x1fcb2e204):
/// the slice form with slice 0 and bytesPerImage 0.
pub fn replace_region_copy(r: &ReplaceRegion) -> (RegionCopy, &'static str) {
    let c = RegionCopy {
        region: wire_region(&r.region),
        slice: 0,
        level: r.level as u32,
        linear: Linear { bytes_per_row: r.bytes_per_row, bytes_per_image: 0 },
    };
    (c, REPLACE_REGION_CALL)
}

/// `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:` (AGXMetalA10 0x21c9819fc): `writeRegion`.
pub fn replace_region_slice_copy(r: &ReplaceRegionSlice) -> (RegionCopy, &'static str) {
    let c = RegionCopy {
        region: wire_region(&r.region),
        slice: r.slice as u32,
        level: r.level as u32,
        linear: Linear { bytes_per_row: r.bytes_per_row, bytes_per_image: r.bytes_per_image },
    };
    (c, REPLACE_REGION_SLICE_CALL)
}

/// `getBytes:bytesPerRow:fromRegion:mipmapLevel:` is `-[IOGPUMetalTexture getBytes:...]` (IOGPU 0x1fcb31158): the slice
/// form with slice 0 and bytesPerImage the region's height times bytesPerRow.
pub fn get_bytes_copy(r: &GetBytes) -> (RegionCopy, &'static str) {
    let c = RegionCopy {
        region: wire_region(&r.region),
        slice: 0,
        level: r.level as u32,
        linear: Linear { bytes_per_row: r.bytes_per_row, bytes_per_image: get_bytes_image_pitch(r) },
    };
    (c, GET_BYTES_CALL)
}

/// `getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:` (AGXMetalA10 0x21c98199c): `readRegion`.
pub fn get_bytes_slice_copy(r: &GetBytesSlice) -> (RegionCopy, &'static str) {
    let c = RegionCopy {
        region: wire_region(&r.region),
        slice: r.slice as u32,
        level: r.level as u32,
        linear: Linear { bytes_per_row: r.bytes_per_row, bytes_per_image: r.bytes_per_image },
    };
    (c, GET_BYTES_SLICE_CALL)
}

/// A texture over caller memory's subresources, read back.
pub struct GuestArrived<A: Api> {
    texture: Rc<Texture<A>>,
    tights: Vec<Vec<u8>>,
}

impl<A: Api> GuestReadback<A> {
    /// Map the reads once the GPU has run them. Makes no WebGPU call that can raise a device-timeline error.
    pub fn wait(self, mem: Rc<Memory<A>>) -> GpuFuture<Result<GuestArrived<A>, MemError>> {
        Box::pin(async move {
            let mut tights = Vec::with_capacity(self.reads.len());
            for read in self.reads {
                tights.push(read.wait().await?.release(&mem));
            }
            Ok(GuestArrived { texture: self.texture, tights })
        })
    }
}

impl<A: Api> GuestArrived<A> {
    /// Put the subresources into the driver's layout over the bytes last seen and write the bytes that changed into
    /// the memory.
    pub fn merge(self, e: &dyn Embedder) -> Result<(), MemError> {
        let m = self.texture.guest.as_ref().expect("a texture over caller memory").clone();
        let mut shadow = m.shadow.borrow_mut();
        let mut new = shadow.clone();
        for (s, tight) in m.subresources.iter().zip(&self.tights) {
            match region::write_region(&m.layout, &s.copy, &mut new, 0, tight).map_err(|e| copy_error(USE, e))? {
                Outcome::Copied => {}
                Outcome::Refused(r) => unreachable!("checked at creation, refused with {r:?}"),
            }
        }
        e.write_guest_changed(m.task, m.addr, &new, &shadow)?;
        *shadow = new;
        Ok(())
    }
}
