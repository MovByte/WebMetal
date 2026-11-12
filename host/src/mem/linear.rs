//! Textures that share a buffer's storage: `-[MTLBuffer newTextureWithDescriptor:offset:bytesPerRow:]` and the SPI
//! `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:`.
//!
//! In Metal the texture and the buffer are one memory: "Modifying the buffer also modifies the new texture because
//! they share the same underlying memory", coherent between passes but not necessarily within one ("The texture's
//! resource data is coherent between multiple render passes", `makeTexture(descriptor:offset:bytesPerRow:)`
//! documentation). Here the buffer stays the storage the CPU sees (through the shadow/diff protocol) and the texture
//! is a WebGPU texture kept equal to the buffer's bytes at pass boundaries:
//!
//! - before a pass uses the texture, [`Memory::prepare_linear`] copies the buffer's bytes into it if the buffer may
//!   have changed since they last agreed (the buffer's generation);
//! - after a pass that writes the texture, [`Memory::sync_linear`] copies the texture into the buffer's bytes, and the
//!   buffer's readback carries them to the guest.
//!
//! A texture of type `MTLTextureTypeTextureBuffer` has no WebGPU texture: shaders read and write it as the buffer
//! (plan section 5.7), so it only records its place in the buffer.
//!
//! The A10X requires the offset and the row stride to be multiples of `minimumLinearTextureAlignmentForPixelFormat:`
//! (texture buffers: `minimumTextureBufferAlignmentForPixelFormat:`), which is 64 for each of the 55 formats it accepts;
//! for every other format the query reports a validation failure (`[queries]` and `[query_failures]` of the device
//! profile). It has neither `supportsLinearTexture2DArray` nor `supports2DLinearTexArraySPI`.

use std::cell::Cell;
use std::rc::Rc;

use crate::embed::{Embedder, TaskId};
use crate::format::handling::{Resolution, Upload};
use crate::gpu::Api;
use webmetal_wire::{CreateBufferTexture, CreateLinearTexture, Handle, TextureDescriptor};

use super::buffer::{Backing, Buffer};
use super::calls::Calls;
use super::error::MemError;
use super::layout::{RegionArgs, RegionLayout, TextureType};
use super::memory::Memory;
use super::options::StorageMode;
use super::texture::{Texture, TextureRegion};

/// `minimumLinearTextureAlignmentForPixelFormat:` and `minimumTextureBufferAlignmentForPixelFormat:` of the A10X for
/// every format of [`LINEAR_FORMATS`].
pub const LINEAR_ALIGNMENT: u64 = 64;

/// The formats the A10X's linear-alignment queries accept (`[queries."minimumLinearTextureAlignmentForPixelFormat:"]`
/// of the device profile; the texture-buffer query accepts the same list).
pub const LINEAR_FORMATS: [u32; 55] = [
    1, 10, 11, 12, 13, 14, 20, 22, 23, 24, 25, 30, 31, 32, 33, 34, 40, 41, 42, 43, 53, 54, 55, 60, 62, 63, 64, 65, 70,
    71, 72, 73, 74, 80, 81, 90, 91, 92, 93, 94, 103, 104, 105, 110, 112, 113, 114, 115, 123, 124, 125, 552, 553, 554,
    555,
];

/// A texture over bytes of a buffer.
pub struct LinearTexture<A: Api> {
    /// The WebGPU-backed texture; `None` for a texture buffer.
    pub texture: Option<Rc<Texture<A>>>,
    pub buffer: Rc<Buffer<A>>,
    pub offset: u64,
    pub bytes_per_row: u64,
    /// The texture's bytes in the buffer, as a transfer of the whole texture lays them out.
    pub layout: RegionLayout,
    /// The buffer generation the texture last agreed with.
    synced: Cell<Option<u64>>,
    /// GPU work since the last [`Memory::sync_linear`] wrote the texture.
    written: Cell<bool>,
}

impl<A: Api> Memory<A> {
    /// `-[MTLBuffer newTextureWithDescriptor:offset:bytesPerRow:]` (CreateBufferTexture) over `buffer`.
    pub fn create_buffer_texture(
        &self,
        g: &impl Calls<A>,
        task: TaskId,
        buffer: &Rc<Buffer<A>>,
        r: &CreateBufferTexture,
    ) -> Result<LinearTexture<A>, MemError> {
        const CALL: &str = "-[MTLBuffer newTextureWithDescriptor:offset:bytesPerRow:]";
        self.linear(g, CALL, task, buffer, r.texture, &r.descriptor, r.offset, r.bytes_per_row)
    }

    /// `-[MTLBufferSPI newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:]` (CreateLinearTexture). The
    /// image stride matters only for 2D-array linear textures, which the A10X does not support.
    pub fn create_linear_texture(
        &self,
        g: &impl Calls<A>,
        task: TaskId,
        buffer: &Rc<Buffer<A>>,
        r: &CreateLinearTexture,
    ) -> Result<LinearTexture<A>, MemError> {
        const CALL: &str = "-[MTLBufferSPI newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:]";
        if r.descriptor.texture_type == 3 {
            return Err(MemError::undefined(CALL, "the A10X supports no 2D-array linear textures"));
        }
        self.linear(g, CALL, task, buffer, r.texture, &r.descriptor, r.offset, r.bytes_per_row)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        g: &impl Calls<A>,
        call: &'static str,
        task: TaskId,
        buffer: &Rc<Buffer<A>>,
        handle: Handle,
        d: &TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
    ) -> Result<LinearTexture<A>, MemError> {
        let format = u32::try_from(d.pixel_format).map_err(|_| MemError::undefined(call, "pixel format"))?;
        if !LINEAR_FORMATS.contains(&format) {
            return Err(MemError::undefined(
                call,
                format!("the A10X's linear texture alignment query reports a failure for pixel format {format}"),
            ));
        }
        if !offset.is_multiple_of(LINEAR_ALIGNMENT) || !bytes_per_row.is_multiple_of(LINEAR_ALIGNMENT) {
            return Err(MemError::undefined(
                call,
                format!("offset {offset} and bytesPerRow {bytes_per_row} must be multiples of {LINEAR_ALIGNMENT}"),
            ));
        }
        let storage = StorageMode::from_value(d.storage_mode, call)?;
        if storage != buffer.options.storage {
            return Err(MemError::undefined(call, "the texture's storage mode differs from the buffer's"));
        }
        if d.mipmap_level_count != 1 || d.sample_count != 1 {
            return Err(MemError::undefined(call, "a linear texture has one level and one sample"));
        }
        let kind = TextureType::from_value(d.texture_type, call)?;
        let texture = match kind {
            TextureType::TextureBuffer => None,
            TextureType::D2 | TextureType::D1 => {
                // The WebGPU texture is private to the executor: shared storage here is the buffer's.
                let mut own = d.clone();
                own.texture_type = 2;
                own.storage_mode = 2;
                own.resource_options = (d.resource_options & !0xf0) | (2 << 4);
                let t = self.texture_from_descriptor(g, call, task, handle, &own)?;
                match t.resolution {
                    Resolution::Emulated(e) if e.upload == Upload::SwapRedBlue10 => {
                        return Err(MemError::unimplemented(format!(
                            "{call}: a BGR10A2 texture over a buffer needs a channel swap pass between the two"
                        )));
                    }
                    _ => {}
                }
                if t.has_twin() {
                    return Err(MemError::unimplemented(format!(
                        "{call}: a texture over a buffer in a format kept with a twin"
                    )));
                }
                Some(t)
            }
            _ => return Err(MemError::undefined(call, "a linear texture is 2D or a texture buffer")),
        };
        let blocks = match &texture {
            Some(t) => t.blocks.expect("linear formats are uncompressed"),
            None => {
                let r = crate::format::handling::resolve(format, self.features).map_err(|e| match e {
                    crate::format::handling::Unavailable::NotOnDevice(w) => MemError::undefined(call, w.to_string()),
                    crate::format::handling::Unavailable::Refused(r) => MemError::refused(r),
                })?;
                super::layout::Blocks::of(r).expect("linear formats are uncompressed")
            }
        };
        let geometry = super::layout::Geometry {
            kind: TextureType::D2,
            width: u32::try_from(d.width).map_err(|_| MemError::undefined(call, "width"))?,
            height: u32::try_from(d.height.max(1)).map_err(|_| MemError::undefined(call, "height"))?,
            depth: 1,
            array_length: 1,
            mip_levels: 1,
            sample_count: 1,
        };
        let a = RegionArgs {
            origin: (0, 0, 0),
            size: (u64::from(geometry.width), u64::from(geometry.height), 1),
            level: 0,
            slice: 0,
            bytes_per_row,
            bytes_per_image: 0,
        };
        let layout = RegionLayout::new(call, &geometry, blocks, &a)?;
        if offset.checked_add(layout.span).is_none_or(|end| end > buffer.length) {
            return Err(MemError::undefined(call, "the texture's rows extend past the buffer"));
        }
        Ok(LinearTexture {
            texture,
            buffer: buffer.clone(),
            offset,
            bytes_per_row,
            layout,
            synced: Cell::new(None),
            written: Cell::new(false),
        })
    }

    /// Before a pass uses the texture: copy the buffer's bytes into it if the buffer may have changed since they last
    /// agreed. The buffer's own upload ([`Memory::upload_buffer`]) comes first, as for any use of the buffer.
    pub fn prepare_linear(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        lt: &LinearTexture<A>,
    ) -> Result<(), MemError> {
        let Some(t) = &lt.texture else { return Ok(()) };
        let generation = lt.buffer.generation();
        if lt.synced.get() == Some(generation) {
            return Ok(());
        }
        let r = TextureRegion { level: 0, slice: 0, layout: lt.layout };
        self.encode_buffer_to_texture(g, encoder, (&lt.buffer, lt.offset), t, &r)?;
        lt.synced.set(Some(generation));
        Ok(())
    }

    /// GPU work encoded next writes the texture.
    pub fn note_linear_write(&self, lt: &LinearTexture<A>) {
        if let Some(t) = &lt.texture {
            self.note_texture_write(t, 0..1, 0..1);
            lt.written.set(true);
        }
    }

    /// After the passes that wrote the texture: copy it into the buffer's bytes, which then reach the guest with the
    /// buffer's readback.
    pub fn sync_linear(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        lt: &LinearTexture<A>,
    ) -> Result<(), MemError> {
        let Some(t) = &lt.texture else { return Ok(()) };
        if !lt.written.replace(false) {
            return Ok(());
        }
        let r = TextureRegion { level: 0, slice: 0, layout: lt.layout };
        self.encode_texture_to_buffer(g, encoder, t, &r, (&lt.buffer, lt.offset))?;
        lt.synced.set(Some(lt.buffer.generation()));
        Ok(())
    }

    /// `replaceRegion` of a texture over a shared buffer writes the buffer's memory: the guest's bytes, which the
    /// buffer's next upload carries to the GPU.
    pub fn replace_region_linear(
        &self,
        e: &dyn Embedder,
        task: TaskId,
        lt: &LinearTexture<A>,
        a: &RegionArgs,
        pixel_bytes: u64,
    ) -> Result<(), MemError> {
        const CALL: &str = "-[MTLTexture replaceRegion:...]";
        let (addr, l, at) = self.linear_cpu_region(CALL, lt, a)?;
        let mut src = vec![0u8; l.span as usize];
        e.read_guest(task, pixel_bytes, &mut src)?;
        let tight = l.pack(&src);
        for r in 0..u64::from(l.blocks.1) {
            let row = &tight[(r * l.row_bytes) as usize..((r + 1) * l.row_bytes) as usize];
            e.write_guest(task, addr + at + r * lt.bytes_per_row, row)?;
        }
        Ok(())
    }

    /// `getBytes` of a texture over a shared buffer reads the buffer's memory.
    pub fn get_bytes_linear(
        &self,
        e: &dyn Embedder,
        task: TaskId,
        lt: &LinearTexture<A>,
        a: &RegionArgs,
        pixel_bytes: u64,
    ) -> Result<(), MemError> {
        const CALL: &str = "-[MTLTexture getBytes:...]";
        let (addr, l, at) = self.linear_cpu_region(CALL, lt, a)?;
        let mut tight = vec![0u8; l.tight_len() as usize];
        for r in 0..u64::from(l.blocks.1) {
            let row = &mut tight[(r * l.row_bytes) as usize..((r + 1) * l.row_bytes) as usize];
            e.read_guest(task, addr + at + r * lt.bytes_per_row, row)?;
        }
        let mut held = vec![0u8; l.span as usize];
        e.read_guest(task, pixel_bytes, &mut held)?;
        let mut new = held.clone();
        l.unpack(&tight, &mut new);
        e.write_guest_changed(task, pixel_bytes, &new, &held)?;
        Ok(())
    }

    /// The buffer's guest address, the region's layout and its first byte's offset from the address.
    fn linear_cpu_region(
        &self,
        call: &'static str,
        lt: &LinearTexture<A>,
        a: &RegionArgs,
    ) -> Result<(u64, RegionLayout, u64), MemError> {
        let Backing::Guest { addr, .. } = lt.buffer.backing else {
            return Err(MemError::undefined(call, "CPU access to a texture over a private buffer"));
        };
        let geometry = super::layout::Geometry {
            kind: TextureType::D2,
            width: lt.layout.size.0,
            height: lt.layout.size.1,
            depth: 1,
            array_length: 1,
            mip_levels: 1,
            sample_count: 1,
        };
        let unit = lt.layout.row_bytes / u64::from(lt.layout.blocks.0);
        let blocks = super::layout::Blocks { width: 1, height: 1, bytes: unit as u32, pvrtc: None };
        let l = RegionLayout::new(call, &geometry, blocks, a)?;
        let at = lt.offset + u64::from(l.origin.1) * lt.bytes_per_row + u64::from(l.origin.0) * unit;
        Ok((addr, l, at))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::testutil::repo_path;

    /// The alignment and the format list against the device profile.
    #[test]
    fn profile_values() {
        let profile = std::fs::read_to_string(repo_path("profile/ipados-17.7.11-21H461-a10x.toml")).unwrap();
        for query in ["minimumLinearTextureAlignmentForPixelFormat:", "minimumTextureBufferAlignmentForPixelFormat:"] {
            let header = format!("[queries.\"{query}\"]\n");
            let section = profile.split(&header).nth(1).unwrap().split("\n[").next().unwrap();
            let mut formats = Vec::new();
            for line in section.lines().filter(|l| l.starts_with(|c: char| c.is_ascii_digit())) {
                let (k, v) = line.split_once(" = ").unwrap();
                formats.push(k.parse::<u32>().unwrap());
                assert_eq!(
                    v.split_whitespace().next().unwrap().parse::<u64>().unwrap(),
                    LINEAR_ALIGNMENT,
                    "{query} {k}"
                );
            }
            assert_eq!(formats, LINEAR_FORMATS, "{query}");
        }
    }
}
