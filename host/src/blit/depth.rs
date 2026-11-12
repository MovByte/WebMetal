//! Blits between buffers and the depth or stencil aspect of a texture where WebGPU's copies do not reach.
//!
//! WebGPU copies the stencil aspect of `depth32float-stencil8` both ways and its depth aspect, like that of
//! `depth32float`, only out of the texture (WebGPU section 26.1.2); copies with an aspect need 4-byte buffer offsets and
//! 256-byte rows (section 13.3). So:
//!
//! - stencil: through a buffer whose rows are re-pitched on the GPU ([`encode_rows`]);
//! - depth out of the texture: a copy of the aspect, re-pitched into the destination;
//! - depth into the texture: a render pass that writes the values ([`encode_depth_write`]). It stores +0 and the normal
//!   values of [0, 1] unchanged and nothing else, so the values are checked first, which needs them on the CPU: the
//!   source must be a CPU-visible buffer whose GPU copy equals what the CPU last saw ([`Memory::gpu_bytes_known`]).
//!   Anything else is refused.
//!
//! A combined format takes `MTLBlitOptionDepthFromDepthStencil` or `MTLBlitOptionStencilFromDepthStencil`, "but not
//! both" (Apple documentation of the blit methods with `options:`).

use crate::format::webgpu::Aspect;
use crate::gpu::idl::{
    BufferDescriptor, BufferUsage, Extent3D, Extent3DDict, Origin3D, Origin3DDict, TexelCopyBufferInfo,
    TexelCopyTextureInfo, TextureAspect,
};
use crate::gpu::{Api, CommandEncoder as _, Device as _, Size64};
use crate::mem::buffer::Buffer;
use crate::mem::calls::Calls;
use crate::mem::layout::{RegionLayout, StagingPitch};
use crate::mem::texture::Texture;
use crate::mem::{MemError, Memory};

use super::kernels::{DepthWrite, Rows, Source, depth_bits, encode_depth_write, encode_rows};
use super::region::Region;

fn gpu_err(op: &'static str) -> impl Fn(crate::gpu::Exception) -> MemError {
    move |e| MemError::gpu(op, e)
}

fn idl_aspect(a: Aspect) -> TextureAspect {
    match a {
        Aspect::Depth => TextureAspect::DepthOnly,
        Aspect::Stencil => TextureAspect::StencilOnly,
    }
}

fn staging<A: Api>(g: &impl Calls<A>, size: u64) -> Result<A::Buffer, MemError> {
    let d = BufferDescriptor::new(
        Size64::new(size.max(4)).map_err(gpu_err("createBuffer"))?,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST | BufferUsage::STORAGE,
    );
    g.call("GPUDevice.createBuffer", "webmetal aspect staging", |dev| dev.create_buffer(&d))
        .map_err(gpu_err("GPUDevice.createBuffer"))
}

/// The texel copy of `r` in `t`'s stored texture.
fn texel<'a, A: Api>(t: &'a Texture<A>, r: &Region, aspect: Aspect) -> TexelCopyTextureInfo<'a, A> {
    let mut c = TexelCopyTextureInfo::<A>::new(t.gpu());
    c.mip_level = r.level;
    c.origin = Origin3D::Origin3DDict(Origin3DDict { x: r.origin.0, y: r.origin.1, z: r.slice });
    c.aspect = idl_aspect(aspect);
    c
}

fn extent(r: &Region) -> Extent3D {
    Extent3D::Extent3DDict(Extent3DDict { width: r.size.0, height: r.size.1, depth_or_array_layers: 1 })
}

/// A blit from a buffer into the `aspect` of a depth or stencil texture, laid out as `l` at `offset` of `src`.
pub fn buffer_to_aspect<A: Api>(
    mem: &Memory<A>,
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    (src, offset): (&Buffer<A>, u64),
    (t, aspect): (&Texture<A>, Aspect),
    r: &Region,
    l: &RegionLayout,
) -> Result<(), MemError> {
    const CALL: &str = "-[MTLBlitCommandEncoder copyFromBuffer:...toTexture:...]";
    let layer = t.layer_of(r.slice);
    match aspect {
        Aspect::Depth => {
            let bytes = mem.gpu_bytes_known(src, offset..offset + l.span).ok_or_else(|| {
                MemError::unimplemented(format!(
                    "{CALL}: WebGPU writes a depth32float aspect only through a render pass, whose values are \
                     checked on the CPU first; the source buffer's bytes are on the GPU only (private storage, or \
                     written by GPU work not read back yet)"
                ))
            })?;
            let bits = depth_bits(CALL, &l.pack(&bytes))?;
            let w = DepthWrite {
                texture: t.gpu(),
                level: r.level,
                layer: r.slice,
                origin: (r.origin.0, r.origin.1),
                size: (r.size.0, r.size.1),
                bits: &bits,
            };
            encode_depth_write(mem, g, encoder, &w)?;
        }
        Aspect::Stencil => {
            let rows = u64::from(l.blocks.1);
            let direct = offset.is_multiple_of(4) && (rows == 1 || l.bytes_per_row.is_multiple_of(256));
            let (buffer, base, bytes_per_row) = if direct {
                (src.gpu().clone(), offset, l.bytes_per_row.next_multiple_of(256).max(256))
            } else {
                let pitch = StagingPitch::new(l.row_bytes, rows, 1);
                let s = staging(g, pitch.size())?;
                let r = Rows {
                    row_bytes: l.row_bytes,
                    rows,
                    images: 1,
                    src_row_pitch: l.bytes_per_row,
                    src_image_pitch: 0,
                    dst_row_pitch: pitch.bytes_per_row,
                    dst_image_pitch: 0,
                };
                encode_rows(mem, g, encoder, Source::Buffer(src.gpu(), offset), (&s, 0), r)?;
                (s, 0, pitch.bytes_per_row)
            };
            let mut b = TexelCopyBufferInfo::<A>::new(&buffer);
            b.offset = Size64::new(base).map_err(gpu_err("copyBufferToTexture"))?;
            b.bytes_per_row = Some(bytes_per_row as u32);
            b.rows_per_image = Some(l.blocks.1);
            g.call("GPUCommandEncoder.copyBufferToTexture", "", |_| {
                encoder.copy_buffer_to_texture(&b, &texel(t, r, aspect), &extent(r))
            })
            .map_err(gpu_err("GPUCommandEncoder.copyBufferToTexture"))?;
        }
    }
    mem.note_buffer_read(src);
    mem.note_texture_write(t, r.level..r.level + 1, layer..layer + 1);
    Ok(())
}

/// A blit from the `aspect` of a depth or stencil texture into a buffer, laid out as `l` at `offset` of `dst`.
///
/// The depth aspect holds only what a depth output clamped to [0, 1] or [`encode_depth_write`] stored, finite values
/// WebGPU's copies keep (WebGPU section 11.2), so the copy is the stored values.
pub fn aspect_to_buffer<A: Api>(
    mem: &Memory<A>,
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    (t, aspect): (&Texture<A>, Aspect),
    r: &Region,
    l: &RegionLayout,
    (dst, offset): (&Buffer<A>, u64),
) -> Result<(), MemError> {
    let rows = u64::from(l.blocks.1);
    let pitch = StagingPitch::new(l.row_bytes, rows, 1);
    let s = staging(g, pitch.size())?;
    let mut b = TexelCopyBufferInfo::<A>::new(&s);
    b.bytes_per_row = Some(pitch.bytes_per_row as u32);
    b.rows_per_image = Some(l.blocks.1);
    g.call("GPUCommandEncoder.copyTextureToBuffer", "", |_| {
        encoder.copy_texture_to_buffer(&texel(t, r, aspect), &b, &extent(r))
    })
    .map_err(gpu_err("GPUCommandEncoder.copyTextureToBuffer"))?;
    mem.encode_staged_rows(g, encoder, &s, (pitch, rows), l, (dst, offset))
}
