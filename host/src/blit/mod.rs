//! The blit encoder: [`BlitService`] records a `MTLBlitCommandEncoder`'s commands into the command buffer's WebGPU
//! command encoder.
//!
//! | Metal | Here |
//! | --- | --- |
//! | `copyFromBuffer:...toBuffer:` | `copyBufferToBuffer` when the offsets and size are whole words; else the GPU's byte moves (`kernels::encode_rows`); within one buffer through a staging copy. Overlapping ranges of one buffer are undefined in Metal ("Copying data to overlapping regions within the same buffer may result in unexpected behavior") and refused. |
//! | `fillBuffer:range:value:` | `clearBuffer` for zero over whole words; else the GPU's byte fill (`clearBuffer` writes zeros only, `tables/limits.toml` question `6.6-clear-buffer`). |
//! | `copyFromBuffer:...toTexture:`, `copyFromTexture:...toBuffer:`, with and without `options:` | The memory module's transfers ([`Memory::encode_buffer_to_texture`], [`Memory::encode_texture_to_buffer`]), which keep the twins and the readback windows right; depth and stencil aspects through [`depth`]. Regions and layouts follow [`region`]. |
//! | `copyFromTexture:...toTexture:` (region, slices and levels, whole texture) | [`texture::copy_region`]. |
//! | `generateMipmapsForTexture:` | [`texture::generate_mipmaps`]. |
//! | `optimizeContentsForGPUAccess:` / `ForCPUAccess:` (and the slice and level forms) | Hints with no effect on contents (`blit.optimize-contents`); the arguments are checked. |
//! | `synchronizeResource:`, `synchronizeTexture:slice:level:` | `API_UNAVAILABLE(ios)`: undefined. |
//! | `sampleCountersInBuffer:atSampleIndex:withBarrier:` | The A10X does not support `MTLCounterSamplingPointAtBlitBoundary` (device profile): "this method is not available and will generate an error if called" (`MTLBlitCommandEncoder.h`): undefined. |
//! | `resolveCounters:inRange:destinationBuffer:destinationOffset:`, a pass descriptor's sample buffer attachments | Refused: the wire protocol has no request that creates a counter sample buffer. |
//! | Indirect command buffer copies, resets and optimizations | Refused: the wire protocol does not define the indirect command buffer's memory yet (`CreateIndirectCommandBuffer`). |
//! | Texture access counters, the SPI fills and compressed-texture invalidations, `copyFromTexture:...options:` with options | Refused: their semantics are not established. |
//!
//! Every Metal object a command names must be non-nil; Metal does not define a blit of nil. Fences are the core's.

pub mod depth;
pub mod kernels;
pub mod region;
pub mod texture;

use std::rc::Rc;

use webmetal_wire::{Command, Failure, Handle, Origin, Size};

use crate::embed::Embedder;
use crate::exec::failure;
use crate::exec::{DebugGroups, EncodeCx, EncoderService, RecordFailure, Segment, Table};
use crate::format::handling::Resolution;
use crate::format::webgpu::Aspect;
use crate::gpu::{Api, CommandEncoder as _, DebugCommands as _, Size64};
use crate::mem::buffer::Buffer;
use crate::mem::calls::Calls;
use crate::mem::layout::TextureType;
use crate::mem::linear::LinearTexture;
use crate::mem::texture::{Texture, TextureRegion, depth_float_only};
use crate::mem::{MemError, Memory};

use kernels::{Rows, Source, encode_rows};
use region::{Unit, buffer_layout, option};

/// The blit encoder service.
pub struct BlitService<A: Api> {
    memory: Rc<Memory<A>>,
}

impl<A: Api> BlitService<A> {
    /// The service over the memory module's [`Memory`] (`MemoryServices::memory`).
    pub fn new(memory: Rc<Memory<A>>) -> BlitService<A> {
        BlitService { memory }
    }
}

fn payload<T: 'static>(objects: &Table, h: Handle) -> Option<Rc<T>> {
    objects.get(h)?.payload.clone()?.downcast::<T>().ok()
}

/// A texture a blit names: a texture, or a texture over a buffer's bytes.
pub(crate) enum Tex<A: Api> {
    Plain(Rc<Texture<A>>),
    Linear(Rc<LinearTexture<A>>, Rc<Texture<A>>),
}

impl<A: Api> Tex<A> {
    pub(crate) fn texture(&self) -> &Rc<Texture<A>> {
        match self {
            Tex::Plain(t) | Tex::Linear(_, t) => t,
        }
    }

    /// Before the blit reads or writes it: a texture over a buffer takes the buffer's bytes.
    fn prepare(&self, mem: &Memory<A>, g: &impl Calls<A>, encoder: &A::CommandEncoder) -> Result<(), MemError> {
        match self {
            Tex::Plain(_) => Ok(()),
            Tex::Linear(lt, _) => mem.prepare_linear(g, encoder, lt),
        }
    }

    /// After the blit wrote it: a texture over a buffer gives its bytes back to the buffer.
    fn written(&self, mem: &Memory<A>, g: &impl Calls<A>, encoder: &A::CommandEncoder) -> Result<(), MemError> {
        match self {
            Tex::Plain(_) => Ok(()),
            Tex::Linear(lt, _) => {
                mem.note_linear_write(lt);
                mem.sync_linear(g, encoder, lt)
            }
        }
    }
}

fn nil(call: &'static str, what: &str) -> MemError {
    MemError::undefined(call, format!("{what} is nil"))
}

pub(crate) fn buffer<A: Api>(
    objects: &Table,
    h: Option<Handle>,
    call: &'static str,
    what: &str,
) -> Result<Rc<Buffer<A>>, MemError> {
    let h = h.ok_or_else(|| nil(call, what))?;
    payload::<Buffer<A>>(objects, h)
        .ok_or_else(|| MemError::invalid(format!("{call}: {what} {:#x} has no buffer storage", h.get())))
}

pub(crate) fn texture<A: Api>(
    objects: &Table,
    h: Option<Handle>,
    call: &'static str,
    what: &str,
) -> Result<Tex<A>, MemError> {
    let h = h.ok_or_else(|| nil(call, what))?;
    if let Some(t) = payload::<Texture<A>>(objects, h) {
        return Ok(Tex::Plain(t));
    }
    if let Some(lt) = payload::<LinearTexture<A>>(objects, h) {
        return match lt.texture.clone() {
            Some(t) => Ok(Tex::Linear(lt, t)),
            None => Err(MemError::unimplemented(format!(
                "{call}: {what} is an MTLTextureTypeTextureBuffer texture, which shaders read as its buffer"
            ))),
        };
    }
    Err(MemError::invalid(format!("{call}: {what} {:#x} has no texture storage", h.get())))
}

/// What a blit between a buffer and a texture moves: the texture's own texels, or one aspect of a depth or stencil
/// texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    Texels,
    Aspect(Aspect),
}

/// Check `options` against the texture's format ("If the texture's pixel format is a combined depth/stencil format,
/// set `options` to either `depthFromDepthStencil` or `stencilFromDepthStencil`, but not both. If the texture's pixel
/// format is a PVRTC format, set `options` to `rowLinearPVRTC`") and say what the blit moves, in which units.
fn part<A: Api>(call: &'static str, t: &Texture<A>, options: u64) -> Result<(Part, Unit), MemError> {
    let known = option::DEPTH_FROM_DEPTH_STENCIL | option::STENCIL_FROM_DEPTH_STENCIL | option::ROW_LINEAR_PVRTC;
    if options & !known != 0 {
        return Err(MemError::undefined(call, format!("options {options:#x} has bits MTLBlitOption does not define")));
    }
    let caps = t.resolution.storage().caps();
    if t.blocks.is_none() {
        let aspect = match options {
            option::DEPTH_FROM_DEPTH_STENCIL => Aspect::Depth,
            option::STENCIL_FROM_DEPTH_STENCIL => Aspect::Stencil,
            _ => {
                return Err(MemError::undefined(
                    call,
                    format!(
                        "options {options:#x} for a combined depth/stencil format: one of the depth or stencil options"
                    ),
                ));
            }
        };
        let a = caps.aspects.iter().find(|a| a.aspect == aspect).expect("combined formats have both aspects");
        let bytes = a
            .footprint
            .ok_or_else(|| MemError::host(format!("the {aspect:?} aspect of {} has no copy footprint", caps.name)))?;
        return Ok((Part::Aspect(aspect), Unit::aspect(bytes)));
    }
    let unit = Unit::of(t.blocks.expect("checked"));
    let pvrtc = unit.pvrtc.is_some();
    match (pvrtc, options) {
        (true, option::ROW_LINEAR_PVRTC) | (false, 0) => {}
        (true, _) => {
            return Err(MemError::undefined(call, "PVRTC blits with a buffer take MTLBlitOptionRowLinearPVRTC"));
        }
        (false, o) => {
            return Err(MemError::undefined(call, format!("options {o:#x} for a format they do not apply to")));
        }
    }
    if depth_float_only(t.resolution.storage()) {
        return Ok((Part::Aspect(Aspect::Depth), unit));
    }
    Ok((Part::Texels, unit))
}

impl<A: Api> BlitService<A> {
    #[allow(clippy::too_many_arguments)]
    fn buffer_to_texture<E: Embedder>(
        &self,
        cx: &mut EncodeCx<'_, A, E>,
        call: &'static str,
        (src, offset, bytes_per_row, bytes_per_image): (Option<Handle>, u64, u64, u64),
        size: &Size,
        (dst, slice, level, origin): (Option<Handle>, u64, u64, &Origin),
        options: u64,
    ) -> Result<(), MemError> {
        let mem = &*self.memory;
        let encoder = cx.encoder().clone();
        let (g, objects) = (cx.gpu(), cx.objects());
        let src = buffer(objects, src, call, "the source buffer")?;
        let tex = texture(objects, dst, call, "the destination texture")?;
        let t = tex.texture();
        check_buffer_blit_texture(call, t)?;
        let (part, unit) = part(call, t, options)?;
        let r = region::region(call, &t.geometry, unit, (slice, level), origin, size)?;
        let l = buffer_layout(call, &t.geometry, unit, &r, (offset, bytes_per_row, bytes_per_image), src.length)?;
        tex.prepare(mem, g, &encoder)?;
        match part {
            Part::Texels => {
                let tr = TextureRegion { level: r.level, slice: r.slice, layout: l };
                mem.encode_buffer_to_texture(g, &encoder, (&src, offset), t, &tr)?;
                mem.note_buffer_read(&src);
            }
            Part::Aspect(a) => depth::buffer_to_aspect(mem, g, &encoder, (&src, offset), (t, a), &r, &l)?,
        }
        tex.written(mem, g, &encoder)
    }

    #[allow(clippy::too_many_arguments)]
    fn texture_to_buffer<E: Embedder>(
        &self,
        cx: &mut EncodeCx<'_, A, E>,
        call: &'static str,
        (src, slice, level, origin): (Option<Handle>, u64, u64, &Origin),
        size: &Size,
        (dst, offset, bytes_per_row, bytes_per_image): (Option<Handle>, u64, u64, u64),
        options: u64,
    ) -> Result<(), MemError> {
        let mem = &*self.memory;
        let encoder = cx.encoder().clone();
        let (g, objects) = (cx.gpu(), cx.objects());
        let tex = texture(objects, src, call, "the source texture")?;
        let dst = buffer(objects, dst, call, "the destination buffer")?;
        let t = tex.texture();
        check_buffer_blit_texture(call, t)?;
        let (part, unit) = part(call, t, options)?;
        let r = region::region(call, &t.geometry, unit, (slice, level), origin, size)?;
        let l = buffer_layout(call, &t.geometry, unit, &r, (offset, bytes_per_row, bytes_per_image), dst.length)?;
        tex.prepare(mem, g, &encoder)?;
        match part {
            Part::Aspect(a) if t.blocks.is_none() => {
                depth::aspect_to_buffer(mem, g, &encoder, (t, a), &r, &l, (&dst, offset))
            }
            _ => {
                let tr = TextureRegion { level: r.level, slice: r.slice, layout: l };
                mem.encode_texture_to_buffer(g, &encoder, t, &tr, (&dst, offset))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn texture_to_texture<E: Embedder>(
        &self,
        cx: &mut EncodeCx<'_, A, E>,
        call: &'static str,
        (src, src_slice, src_level, src_origin): (Option<Handle>, u64, u64, &Origin),
        size: &Size,
        (dst, dst_slice, dst_level, dst_origin): (Option<Handle>, u64, u64, &Origin),
    ) -> Result<(), MemError> {
        let mem = &*self.memory;
        let encoder = cx.encoder().clone();
        let (g, objects) = (cx.gpu(), cx.objects());
        let s = texture(objects, src, call, "the source texture")?;
        let d = texture(objects, dst, call, "the destination texture")?;
        let (st, dt) = (s.texture(), d.texture());
        check_pair(call, st, dt)?;
        let unit = texture::unit_of(st).unwrap_or(Unit::aspect(1));
        let sr = region::region(call, &st.geometry, unit, (src_slice, src_level), src_origin, size)?;
        // The destination region has the source region's size, except that a compressed source region clamped to the
        // source level's edge covers whole blocks at the destination, clamped to the destination level's edge.
        let dsize = if unit.compressed() && unit.pvrtc.is_none() && dst_level < u64::from(dt.geometry.mip_levels) {
            let (dw, dh, _) = dt.geometry.level_extent(dst_level as u32);
            let span = |blocks: u32, b: u32, o: u64, l: u32| {
                (u64::from(blocks) * u64::from(b)).min(u64::from(l).saturating_sub(o))
            };
            Size {
                width: span(sr.blocks.0, unit.width, dst_origin.x, dw),
                height: span(sr.blocks.1, unit.height, dst_origin.y, dh),
                depth: size.depth,
            }
        } else {
            size.clone()
        };
        let dr = region::region(call, &dt.geometry, unit, (dst_slice, dst_level), dst_origin, &dsize)?;
        s.prepare(mem, g, &encoder)?;
        d.prepare(mem, g, &encoder)?;
        texture::copy_region(mem, g, &encoder, (st, &sr), (dt, &dr))?;
        d.written(mem, g, &encoder)
    }

    /// `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:`.
    #[allow(clippy::too_many_arguments)]
    fn surfaces<E: Embedder>(
        &self,
        cx: &mut EncodeCx<'_, A, E>,
        call: &'static str,
        (src, src_slice, src_level): (Option<Handle>, u64, u64),
        (dst, dst_slice, dst_level): (Option<Handle>, u64, u64),
        (slice_count, level_count): (u64, u64),
    ) -> Result<(), MemError> {
        let mem = &*self.memory;
        let encoder = cx.encoder().clone();
        let (g, objects) = (cx.gpu(), cx.objects());
        let s = texture(objects, src, call, "the source texture")?;
        let d = texture(objects, dst, call, "the destination texture")?;
        let (st, dt) = (s.texture(), d.texture());
        check_pair(call, st, dt)?;
        let bad = |rule: String| Err(MemError::undefined(call, rule));
        // "The sourceTexture must have at least sourceLevel + levelCount mips ... at least sourceSlice + sliceCount
        // array slices", and the same for the destination.
        if src_level.checked_add(level_count).is_none_or(|e| e > u64::from(st.geometry.mip_levels))
            || dst_level.checked_add(level_count).is_none_or(|e| e > u64::from(dt.geometry.mip_levels))
            || src_slice.checked_add(slice_count).is_none_or(|e| e > u64::from(st.geometry.slices()))
            || dst_slice.checked_add(slice_count).is_none_or(|e| e > u64::from(dt.geometry.slices()))
        {
            return bad(format!(
                "slices {src_slice}+{slice_count} levels {src_level}+{level_count} to slices {dst_slice}+{slice_count} \
                 levels {dst_level}+{level_count} outside the textures"
            ));
        }
        let unit = texture::unit_of(st).unwrap_or(Unit::aspect(1));
        s.prepare(mem, g, &encoder)?;
        d.prepare(mem, g, &encoder)?;
        for k in 0..level_count {
            let (sl, dl) = (src_level + k, dst_level + k);
            let (w, h, depth) = st.geometry.level_extent(sl as u32);
            if (w, h, depth) != dt.geometry.level_extent(dl as u32) {
                return bad(format!("source level {sl} and destination level {dl} differ in size"));
            }
            let whole = Size { width: u64::from(w), height: u64::from(h), depth: u64::from(depth) };
            let zero = Origin { x: 0, y: 0, z: 0 };
            for i in 0..slice_count {
                let sr = region::region(call, &st.geometry, unit, (src_slice + i, sl), &zero, &whole)?;
                let dr = region::region(call, &dt.geometry, unit, (dst_slice + i, dl), &zero, &whole)?;
                texture::copy_region(mem, g, &encoder, (st, &sr), (dt, &dr))?;
            }
        }
        d.written(mem, g, &encoder)
    }

    /// `copyFromTexture:toTexture:`: `MTLBlitCommandEncoder.h` defines it as the slices-and-levels form with the
    /// levels and slices it computes.
    fn whole_texture<E: Embedder>(
        &self,
        cx: &mut EncodeCx<'_, A, E>,
        call: &'static str,
        src: Option<Handle>,
        dst: Option<Handle>,
    ) -> Result<(), MemError> {
        let objects = cx.objects();
        let (st, dt) = {
            let s = texture::<A>(objects, src, call, "the source texture")?;
            let d = texture::<A>(objects, dst, call, "the destination texture")?;
            (s.texture().clone(), d.texture().clone())
        };
        check_pair(call, &st, &dt)?;
        if matches!(st.geometry.kind, TextureType::Cube | TextureType::CubeArray)
            || matches!(dt.geometry.kind, TextureType::Cube | TextureType::CubeArray)
        {
            // The header computes sliceCount = min(arrayLength, arrayLength), which for a cube does not count its
            // faces; what the driver copies for cubes is not measured.
            return Err(MemError::unimplemented(format!(
                "{call} of cube textures: whether the slice count covers the faces is not established"
            )));
        }
        // "sourceTexture must have a mip M with identical dimensions as the first mip of destinationTexture:
        // sourceLevel = M, destinationLevel = 0", or the other way around.
        let find = |a: &Texture<A>, b: &Texture<A>| {
            (0..a.geometry.mip_levels).find(|&m| a.geometry.level_extent(m) == b.geometry.level_extent(0))
        };
        let (sl, dl) = match (find(&st, &dt), find(&dt, &st)) {
            (Some(m), _) => (m, 0),
            (None, Some(m)) => (0, m),
            (None, None) => {
                return Err(MemError::undefined(
                    call,
                    "neither texture has a level the size of the other's first level",
                ));
            }
        };
        let levels = (st.geometry.mip_levels - sl).min(dt.geometry.mip_levels - dl);
        let slices = st.geometry.array_length.min(dt.geometry.array_length);
        self.surfaces(
            cx,
            call,
            (src, 0, u64::from(sl)),
            (dst, 0, u64::from(dl)),
            (u64::from(slices), u64::from(levels)),
        )
    }

    fn copy_buffer<E: Embedder>(
        &self,
        cx: &mut EncodeCx<'_, A, E>,
        (src, src_offset): (Option<Handle>, u64),
        (dst, dst_offset): (Option<Handle>, u64),
        size: u64,
    ) -> Result<(), MemError> {
        const CALL: &str = "-[MTLBlitCommandEncoder copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:]";
        let mem = &*self.memory;
        let encoder = cx.encoder().clone();
        let (g, objects) = (cx.gpu(), cx.objects());
        let src = buffer(objects, src, CALL, "the source buffer")?;
        let dst = buffer(objects, dst, CALL, "the destination buffer")?;
        let inside = |o: u64, b: &Buffer<A>| o.checked_add(size).is_some_and(|e| e <= b.length);
        if !inside(src_offset, &src) || !inside(dst_offset, &dst) {
            return Err(MemError::undefined(
                CALL,
                format!("{size} bytes at {src_offset} to {dst_offset} outside a buffer"),
            ));
        }
        let same = Rc::ptr_eq(&src, &dst);
        if same && src_offset < dst_offset + size && dst_offset < src_offset + size {
            return Err(MemError::undefined(
                CALL,
                "copying data to overlapping regions within the same buffer may result in unexpected behavior",
            ));
        }
        mem.note_buffer_read(&src);
        mem.note_buffer_write(&dst, Some((dst_offset, dst_offset + size)));
        if size == 0 {
            return Ok(());
        }
        let words = src_offset.is_multiple_of(4) && dst_offset.is_multiple_of(4) && size.is_multiple_of(4);
        let s64 = |v: u64| Size64::new(v).map_err(|e| MemError::gpu("copyBufferToBuffer", e));
        let copy = |from: &A::Buffer, a: u64, to: &A::Buffer, b: u64, n: u64| -> Result<(), MemError> {
            let (a, b, n) = (s64(a)?, s64(b)?, s64(n)?);
            g.call("GPUCommandEncoder.copyBufferToBuffer", CALL, |_| {
                encoder.copy_buffer_to_buffer_with_offsets(from, a, to, b, Some(n))
            });
            Ok(())
        };
        match (words, same) {
            (true, false) => copy(src.gpu(), src_offset, dst.gpu(), dst_offset, size),
            (false, false) => encode_rows(
                mem,
                g,
                &encoder,
                Source::Buffer(src.gpu(), src_offset),
                (dst.gpu(), dst_offset),
                Rows::run(size),
            ),
            (_, true) => {
                // WebGPU copies need two buffers (WebGPU section 13.2): the source words go through a staging buffer.
                let (lo, hi) = (src_offset & !3, (src_offset + size).next_multiple_of(4));
                let staging = kernels::scratch(
                    g,
                    "webmetal buffer self-copy",
                    hi - lo,
                    crate::gpu::idl::BufferUsage::COPY_SRC
                        | crate::gpu::idl::BufferUsage::COPY_DST
                        | crate::gpu::idl::BufferUsage::STORAGE,
                )?;
                copy(src.gpu(), lo, &staging, 0, hi - lo)?;
                if words {
                    copy(&staging, 0, dst.gpu(), dst_offset, size)
                } else {
                    let s = Source::Buffer(&staging, src_offset - lo);
                    encode_rows(mem, g, &encoder, s, (dst.gpu(), dst_offset), Rows::run(size))
                }
            }
        }
    }

    fn fill_buffer<E: Embedder>(
        &self,
        cx: &mut EncodeCx<'_, A, E>,
        call: &'static str,
        h: Option<Handle>,
        (location, length): (u64, u64),
        value: Fill,
    ) -> Result<(), MemError> {
        let mem = &*self.memory;
        let encoder = cx.encoder().clone();
        let (g, objects) = (cx.gpu(), cx.objects());
        let b = buffer(objects, h, call, "the buffer")?;
        // "The range's count property needs to be greater than 0".
        if length == 0 || location.checked_add(length).is_none_or(|e| e > b.length) {
            return Err(MemError::undefined(call, format!("range {location}+{length} of a {}-byte buffer", b.length)));
        }
        mem.note_buffer_write(&b, Some((location, location + length)));
        let words = location.is_multiple_of(4) && length.is_multiple_of(4);
        let zero = match value {
            Fill::Byte(v) => v == 0,
            Fill::Word(w) => w == 0,
        };
        if zero && words {
            let (o, n) = (
                Size64::new(location).map_err(|e| MemError::gpu("clearBuffer", e))?,
                Size64::new(length).map_err(|e| MemError::gpu("clearBuffer", e))?,
            );
            g.call("GPUCommandEncoder.clearBuffer", call, |_| encoder.clear_buffer(b.gpu(), o, Some(n)));
            return Ok(());
        }
        match value {
            Fill::Byte(v) => encode_rows(mem, g, &encoder, Source::Fill(v), (b.gpu(), location), Rows::run(length)),
            Fill::Word(w) => {
                if !words {
                    return Err(MemError::unimplemented(format!(
                        "{call} over a range that is not whole words: where the pattern starts is not established"
                    )));
                }
                let pattern = kernels::buffer_with(g, call, &w.to_le_bytes(), crate::gpu::idl::BufferUsage::STORAGE)?;
                let rows = Rows {
                    row_bytes: 4,
                    rows: length / 4,
                    images: 1,
                    src_row_pitch: 0,
                    src_image_pitch: 0,
                    dst_row_pitch: 4,
                    dst_image_pitch: 0,
                };
                encode_rows(mem, g, &encoder, Source::Buffer(&pattern, 0), (b.gpu(), location), rows)
            }
        }
    }

    fn generate_mipmaps<E: Embedder>(&self, cx: &mut EncodeCx<'_, A, E>, h: Option<Handle>) -> Result<(), MemError> {
        const CALL: &str = "-[MTLBlitCommandEncoder generateMipmapsForTexture:]";
        let mem = &*self.memory;
        let encoder = cx.encoder().clone();
        let (g, objects) = (cx.gpu(), cx.objects());
        let tex = texture(objects, h, CALL, "the texture")?;
        check_blit_texture(CALL, tex.texture())?;
        tex.prepare(mem, g, &encoder)?;
        texture::generate_mipmaps(mem, g, &encoder, tex.texture())?;
        tex.written(mem, g, &encoder)
    }

    /// `optimizeContentsForGPUAccess:` and `optimizeContentsForCPUAccess:`, whole or one slice and level: performance
    /// hints that leave the contents as they are (`blit.optimize-contents`).
    fn optimize<E: Embedder>(
        &self,
        cx: &mut EncodeCx<'_, A, E>,
        call: &'static str,
        h: Option<Handle>,
        at: Option<(u64, u64)>,
    ) -> Result<(), MemError> {
        let tex = texture::<A>(cx.objects(), h, call, "the texture")?;
        let g = &tex.texture().geometry;
        if let Some((slice, level)) = at
            && (slice >= u64::from(g.slices()) || level >= u64::from(g.mip_levels))
        {
            return Err(MemError::undefined(call, format!("slice {slice} level {level} outside the texture")));
        }
        Ok(())
    }

    fn record<E: Embedder>(
        &self,
        cx: &mut EncodeCx<'_, A, E>,
        record: &Command,
        groups: &mut DebugGroups,
    ) -> Result<(), Failure> {
        let m = |e: MemError| e.failure();
        match record {
            Command::BeginBlit(_) | Command::EncoderSetLabel(_) | Command::EncoderEndEncoding(_) => Ok(()),
            Command::BeginBlitWithDescriptor(r) => {
                if r.blit_pass_descriptor.sample_buffer_attachments.iter().any(|a| a.sample_buffer.is_some()) {
                    return Err(failure::unimplemented(
                        "-[MTLCommandBuffer blitCommandEncoderWithDescriptor:] with counter sample buffer attachments: \
                         the wire protocol has no request that creates a counter sample buffer",
                    ));
                }
                Ok(())
            }
            Command::EncoderPushDebugGroup(r) => {
                let e = cx.encoder().clone();
                let label = r.string.clone().unwrap_or_default();
                cx.gpu().call("GPUCommandEncoder.pushDebugGroup", "", |_| e.push_debug_group(&label));
                groups.push();
                Ok(())
            }
            Command::EncoderPopDebugGroup(_) => {
                if groups.pop() {
                    let e = cx.encoder().clone();
                    cx.gpu().call("GPUCommandEncoder.popDebugGroup", "", |_| e.pop_debug_group());
                }
                Ok(())
            }
            Command::EncoderInsertDebugSignpost(r) => {
                let e = cx.encoder().clone();
                let label = r.string.clone().unwrap_or_default();
                cx.gpu().call("GPUCommandEncoder.insertDebugMarker", "", |_| e.insert_debug_marker(&label));
                Ok(())
            }
            Command::BlitCopyBufferToBuffer(r) => self
                .copy_buffer(cx, (r.source_buffer, r.source_offset), (r.destination_buffer, r.destination_offset), r.size)
                .map_err(m),
            Command::BlitCopyBufferToTexture(r) => self
                .buffer_to_texture(
                    cx,
                    "-[MTLBlitCommandEncoder copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:]",
                    (r.source_buffer, r.source_offset, r.source_bytes_per_row, r.source_bytes_per_image),
                    &r.source_size,
                    (r.destination_texture, r.destination_slice, r.destination_level, &r.destination_origin),
                    0,
                )
                .map_err(m),
            Command::BlitCopyBufferToTextureOptions(r) => self
                .buffer_to_texture(
                    cx,
                    "-[MTLBlitCommandEncoder copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:]",
                    (r.source_buffer, r.source_offset, r.source_bytes_per_row, r.source_bytes_per_image),
                    &r.source_size,
                    (r.destination_texture, r.destination_slice, r.destination_level, &r.destination_origin),
                    r.options,
                )
                .map_err(m),
            Command::BlitCopyTextureToBuffer(r) => self
                .texture_to_buffer(
                    cx,
                    "-[MTLBlitCommandEncoder copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:]",
                    (r.source_texture, r.source_slice, r.source_level, &r.source_origin),
                    &r.source_size,
                    (r.destination_buffer, r.destination_offset, r.destination_bytes_per_row, r.destination_bytes_per_image),
                    0,
                )
                .map_err(m),
            Command::BlitCopyTextureToBufferOptions(r) => self
                .texture_to_buffer(
                    cx,
                    "-[MTLBlitCommandEncoder copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:]",
                    (r.source_texture, r.source_slice, r.source_level, &r.source_origin),
                    &r.source_size,
                    (r.destination_buffer, r.destination_offset, r.destination_bytes_per_row, r.destination_bytes_per_image),
                    r.options,
                )
                .map_err(m),
            Command::BlitCopyTextureToTexture(r) => self
                .texture_to_texture(
                    cx,
                    "-[MTLBlitCommandEncoder copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:]",
                    (r.source_texture, r.source_slice, r.source_level, &r.source_origin),
                    &r.source_size,
                    (r.destination_texture, r.destination_slice, r.destination_level, &r.destination_origin),
                )
                .map_err(m),
            Command::BlitCopyTextureToTextureOptions(r) => {
                const CALL: &str = "-[MTLBlitCommandEncoderSPI copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:]";
                if r.options != 0 {
                    return Err(failure::unimplemented(format!(
                        "{CALL} with options {:#x}: what they do is not established",
                        r.options
                    )));
                }
                self.texture_to_texture(
                    cx,
                    CALL,
                    (r.source_texture, r.source_slice, r.source_level, &r.source_origin),
                    &r.source_size,
                    (r.destination_texture, r.destination_slice, r.destination_level, &r.destination_origin),
                )
                .map_err(m)
            }
            Command::BlitCopyTextureToTextureSlicesLevels(r) => self
                .surfaces(
                    cx,
                    "-[MTLBlitCommandEncoder copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:]",
                    (r.source_texture, r.source_slice, r.source_level),
                    (r.destination_texture, r.destination_slice, r.destination_level),
                    (r.slice_count, r.level_count),
                )
                .map_err(m),
            Command::BlitCopyTextureToTextureAll(r) => self
                .whole_texture(cx, "-[MTLBlitCommandEncoder copyFromTexture:toTexture:]", r.source_texture, r.destination_texture)
                .map_err(m),
            Command::BlitFillBuffer(r) => self
                .fill_buffer(
                    cx,
                    "-[MTLBlitCommandEncoder fillBuffer:range:value:]",
                    r.buffer,
                    (r.range.location, r.range.length),
                    Fill::Byte(r.value),
                )
                .map_err(m),
            Command::BlitFillBufferPattern4(r) => self
                .fill_buffer(
                    cx,
                    "-[MTLBlitCommandEncoderSPI fillBuffer:range:pattern4:]",
                    r.buffer,
                    (r.range.location, r.range.length),
                    Fill::Word(r.pattern),
                )
                .map_err(m),
            Command::BlitGenerateMipmapsForTexture(r) => self.generate_mipmaps(cx, r.texture).map_err(m),
            Command::BlitOptimizeContentsForCPUAccess(r) => {
                self.optimize(cx, "-[MTLBlitCommandEncoder optimizeContentsForCPUAccess:]", r.texture, None).map_err(m)
            }
            Command::BlitOptimizeContentsForGPUAccess(r) => {
                self.optimize(cx, "-[MTLBlitCommandEncoder optimizeContentsForGPUAccess:]", r.texture, None).map_err(m)
            }
            Command::BlitOptimizeContentsForCPUAccessSliceLevel(r) => self
                .optimize(cx, "-[MTLBlitCommandEncoder optimizeContentsForCPUAccess:slice:level:]", r.texture, Some((r.slice, r.level)))
                .map_err(m),
            Command::BlitOptimizeContentsForGPUAccessSliceLevel(r) => self
                .optimize(cx, "-[MTLBlitCommandEncoder optimizeContentsForGPUAccess:slice:level:]", r.texture, Some((r.slice, r.level)))
                .map_err(m),
            Command::BlitSynchronizeResource(_) | Command::BlitSynchronizeTexture(_) => Err(failure::unimplemented(format!(
                "{}: behavior Metal does not define: API_UNAVAILABLE(ios) in MTLBlitCommandEncoder.h, the method does \
                 not exist on the device",
                failure::record_methods(record.opcode())
            ))),
            Command::BlitSampleCountersInBuffer(_) => Err(failure::unimplemented(format!(
                "{}: the A10X does not support MTLCounterSamplingPointAtBlitBoundary (device profile, \
                 supportsCounterSampling:), so \"this method is not available and will generate an error if called\" \
                 (MTLBlitCommandEncoder.h)",
                failure::record_methods(record.opcode())
            ))),
            Command::BlitResolveCounters(_) => Err(failure::unimplemented(format!(
                "{}: the wire protocol has no request that creates a counter sample buffer",
                failure::record_methods(record.opcode())
            ))),
            Command::BlitCopyIndirectCommandBuffer(_)
            | Command::BlitResetCommandsInBuffer(_)
            | Command::BlitOptimizeIndirectCommandBuffer(_) => Err(failure::unimplemented(format!(
                "{}: the wire protocol does not define the indirect command buffer's memory yet \
                 (CreateIndirectCommandBuffer)",
                failure::record_methods(record.opcode())
            ))),
            other => Err(failure::unimplemented_record(other.opcode())),
        }
    }
}

/// The value a buffer fill writes.
#[derive(Clone, Copy, Debug)]
enum Fill {
    Byte(u8),
    /// `fillBuffer:range:pattern4:`: a 32-bit pattern, little-endian in memory.
    Word(u32),
}

/// A texture a blit reads or writes must have contents: not memoryless; and blits move single-sample textures between
/// buffers.
fn check_blit_texture<A: Api>(call: &'static str, t: &Texture<A>) -> Result<(), MemError> {
    if t.storage == crate::mem::options::StorageMode::Memoryless {
        return Err(MemError::undefined(call, "a memoryless texture has no contents"));
    }
    if matches!(t.resolution, Resolution::StencilAspect) {
        return Err(MemError::undefined(call, "a stencil view format"));
    }
    Ok(())
}

/// A texture a blit moves to or from a buffer: single-sample. Metal defines buffer blits for texels, and a multisample
/// texture has none to lay out in a buffer.
fn check_buffer_blit_texture<A: Api>(call: &'static str, t: &Texture<A>) -> Result<(), MemError> {
    check_blit_texture(call, t)?;
    if t.geometry.sample_count > 1 {
        return Err(MemError::undefined(call, "a blit between a buffer and a multisample texture"));
    }
    Ok(())
}

/// The source and destination of a texture copy: "The pixel format is the same as `sourceTexture`. The sample count
/// is the same as `sourceTexture`." (debug layer: `blit.texture-to-texture-format-mismatch`).
fn check_pair<A: Api>(call: &'static str, s: &Texture<A>, d: &Texture<A>) -> Result<(), MemError> {
    check_blit_texture(call, s)?;
    check_blit_texture(call, d)?;
    if s.format != d.format {
        return Err(MemError::undefined(
            call,
            format!("source pixel format {} differs from destination pixel format {}", s.format, d.format),
        ));
    }
    if s.geometry.sample_count != d.geometry.sample_count {
        return Err(MemError::undefined(call, "source and destination sample counts differ"));
    }
    Ok(())
}

impl<A: Api, E: Embedder> EncoderService<A, E> for BlitService<A> {
    fn encode(&mut self, cx: &mut EncodeCx<'_, A, E>, segment: &Segment) -> Result<(), RecordFailure> {
        let mut groups = DebugGroups::default();
        let records = cx.records();
        for &i in &segment.records {
            let (offset, record) = &records[i];
            self.record(cx, record, &mut groups)
                .map_err(|failure| RecordFailure { offset: *offset as u64, failure })?;
        }
        let open = groups.close();
        if open > 0 {
            let e = cx.encoder().clone();
            for _ in 0..open {
                cx.gpu().call("GPUCommandEncoder.popDebugGroup", "", |_| e.pop_debug_group());
            }
        }
        Ok(())
    }
}
