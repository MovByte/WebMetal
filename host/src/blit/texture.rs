//! Copies between textures and mipmap generation.
//!
//! A texture's bytes may live in two copies (the stored texture and its twin, see [`crate::mem::texture`]). A copy
//! between textures brings both textures' stored copies up to date ([`Memory::prepare_texture`]) and copies the stored
//! texels; when both twins are current it copies the twins' blocks too, so that the destination keeps exact bytes
//! without a later round trip, and otherwise leaves the destination's twin stale. A software-decoded format's twin
//! holds its blocks, which are always current: they are copied, and the decoded texels follow when both are current,
//! else at the next decode.
//!
//! WebGPU copies between textures need disjoint subresources (WebGPU section 13.3, `copyTextureToTexture`); a copy
//! within one subresource goes through a buffer. Copies between overlapping regions are undefined in Metal
//! (`tables/limits.toml` question `6.6-overlapping-texture-copy`) and refused. Depth and multisample textures copy
//! whole subresources only (WebGPU section 13.3: "the copy size must equal the subresource size" for those formats).

use std::rc::Rc;

use crate::format::handling::{Emulation, Resolution, Upload};
use crate::format::webgpu::{Cond, SampleType};
use crate::gpu::idl::{Extent3D, Extent3DDict, Origin3D, Origin3DDict, TexelCopyTextureInfo, TextureAspect};
use crate::gpu::{Api, CommandEncoder as _, Texture as _};
use crate::mem::calls::Calls;
use crate::mem::layout::TextureType;
use crate::mem::texture::{Alias, Prepare, Texture, idl_format};
use crate::mem::{MemError, Memory};

use super::kernels::{MipLevel, encode_mip_level, encode_texture_copy_staged};
use super::region::{Region, Unit, tight_layout};

fn gpu_err(op: &'static str) -> impl Fn(crate::gpu::Exception) -> MemError {
    move |e| MemError::gpu(op, e)
}

fn origin(o: (u32, u32, u32)) -> Origin3D {
    Origin3D::Origin3DDict(Origin3DDict { x: o.0, y: o.1, z: o.2 })
}

fn extent(e: (u32, u32, u32)) -> Extent3D {
    Extent3D::Extent3DDict(Extent3DDict { width: e.0, height: e.1, depth_or_array_layers: e.2 })
}

/// The unit a texture's regions are counted in, `None` for a combined depth-stencil format (one unit per aspect).
pub fn unit_of<A: Api>(t: &Texture<A>) -> Option<Unit> {
    t.blocks.map(Unit::of)
}

fn decoded<A: Api>(t: &Texture<A>) -> bool {
    matches!(t.resolution, Resolution::Emulated(Emulation { upload: Upload::Decode(_), .. }))
}

/// Do the two regions of one subresource overlap?
fn overlap(a: &Region, b: &Region) -> bool {
    let axis = |o1: u32, s1: u32, o2: u32, s2: u32| o1 < o2 + s2 && o2 < o1 + s1;
    axis(a.origin.0, a.size.0, b.origin.0, b.size.0)
        && axis(a.origin.1, a.size.1, b.origin.1, b.size.1)
        && axis(a.origin.2, a.size.2, b.origin.2, b.size.2)
}

/// One texture copy WebGPU makes: a texture-to-texture copy, or through a buffer when the two are one subresource.
#[allow(clippy::too_many_arguments)]
fn copy<A: Api>(
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    (src, src_level, src_origin): (&A::Texture, u32, (u32, u32, u32)),
    (dst, dst_level, dst_origin): (&A::Texture, u32, (u32, u32, u32)),
    size: (u32, u32, u32),
    one_subresource: bool,
    (bytes, block): (u32, (u32, u32)),
) -> Result<(), MemError> {
    if one_subresource {
        return encode_texture_copy_staged(
            g,
            encoder,
            (src, src_level, src_origin),
            (dst, dst_level, dst_origin),
            size,
            TextureAspect::All,
            (bytes, block),
        );
    }
    let mut s = TexelCopyTextureInfo::<A>::new(src);
    s.mip_level = src_level;
    s.origin = origin(src_origin);
    let mut d = TexelCopyTextureInfo::<A>::new(dst);
    d.mip_level = dst_level;
    d.origin = origin(dst_origin);
    g.call("GPUCommandEncoder.copyTextureToTexture", "", |_| encoder.copy_texture_to_texture(&s, &d, &extent(size)))
        .map_err(gpu_err("GPUCommandEncoder.copyTextureToTexture"))
}

/// Copy region `sr` of `src` to region `dr` of `dst` (the same size; both validated with [`super::region::region`];
/// formats and sample counts already checked equal).
pub fn copy_region<A: Api>(
    mem: &Memory<A>,
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    (src, sr): (&Rc<Texture<A>>, &Region),
    (dst, dr): (&Rc<Texture<A>>, &Region),
) -> Result<(), MemError> {
    const CALL: &str = "-[MTLBlitCommandEncoder copyFromTexture:...toTexture:...]";
    let same = Rc::ptr_eq(src, dst);
    let three_d = src.geometry.kind.is_3d();
    let one_subresource = same && sr.level == dr.level && (three_d || sr.slice == dr.slice);
    if one_subresource && overlap(sr, dr) {
        return Err(MemError::undefined(
            CALL,
            "copying data to overlapping regions within the same texture may result in unexpected behavior",
        ));
    }
    let caps = src.resolution.storage().caps();
    let whole = |t: &Texture<A>, r: &Region| {
        let (w, h, d) = t.geometry.level_extent(r.level);
        r.origin == (0, 0, 0) && r.size == (w, h, d)
    };
    let (src_layer, dst_layer) = (src.layer_of(sr.slice), dst.layer_of(dr.slice));
    let z = |t: &Texture<A>, r: &Region| if t.geometry.kind.is_3d() { r.origin.2 } else { r.slice };
    let src_origin = (sr.origin.0, sr.origin.1, z(src, sr));
    let dst_origin = (dr.origin.0, dr.origin.1, z(dst, dr));
    let images = sr.size.2;

    // Depth, stencil and multisample textures: whole subresources, stored texture to stored texture.
    if !caps.aspects.is_empty() || src.geometry.sample_count > 1 {
        if !(whole(src, sr) && whole(dst, dr)) {
            return Err(MemError::unimplemented(format!(
                "{CALL}: a part of a {} subresource (WebGPU copies {} textures only whole)",
                src.resolution.storage().name(),
                if caps.aspects.is_empty() { "multisample" } else { "depth and stencil" }
            )));
        }
        if one_subresource {
            unreachable!("whole regions of one subresource overlap");
        }
        if caps.aspects.is_empty() {
            mem.check_stored_transfer(src, CALL, false)?;
        }
        copy(
            g,
            encoder,
            (src.gpu(), sr.level, src_origin),
            (dst.gpu(), dr.level, dst_origin),
            (sr.size.0, sr.size.1, images),
            false,
            (caps.footprint, caps.block),
        )?;
        mem.note_texture_write(dst, dr.level..dr.level + 1, dst_layer..dst_layer + 1);
        return Ok(());
    }

    let unit = unit_of(src).expect("color formats have blocks");
    let sl = tight_layout(unit, sr);
    let dl = tight_layout(unit, dr);
    let (s_stored, s_twin) = mem.placements(src, &sl, sr.level, sr.slice);
    let (d_stored, d_twin) = mem.placements(dst, &dl, dr.level, dr.slice);
    let twins_current = src.twin_current(sr.level, src_layer) && dst.twin_current(dr.level, dst_layer);
    let stored_block = (caps.footprint, caps.block);
    let twin_block = (unit.bytes, (1, 1));

    if decoded(src) {
        // The blocks are the authoritative copy and always current; the decoded texels follow when both are.
        if !src.twin_current(sr.level, src_layer) || !dst.twin_current(dr.level, dst_layer) {
            return Err(MemError::unimplemented(format!("{CALL}: a decoded texture whose blocks are not current")));
        }
        copy(
            g,
            encoder,
            (src.twin(sr.level).expect("twin"), 0, s_twin.origin),
            (dst.twin(dr.level).expect("twin"), 0, d_twin.origin),
            s_twin.extent,
            one_subresource,
            twin_block,
        )?;
        let stored = src.stored_current(sr.level, src_layer)
            && dst.stored_current(dr.level, dst_layer)
            && mem.check_stored_transfer(src, CALL, false).is_ok();
        if stored {
            copy(
                g,
                encoder,
                (src.gpu(), sr.level, s_stored.origin),
                (dst.gpu(), dr.level, d_stored.origin),
                s_stored.extent,
                one_subresource,
                (src.resolution.storage().caps().footprint, (1, 1)),
            )?;
        }
        dst.set_current(dr.level, dst_layer, stored, true);
        return Ok(());
    }

    let stored_exact =
        mem.check_stored_transfer(src, CALL, false).and_then(|()| mem.check_stored_transfer(dst, CALL, true));
    if let Err(e) = stored_exact {
        // A format whose texel copies keep values but not bits (compressed blocks, without a proof on this host):
        // the twins' blocks are copied, and the destination's stored texture is rebuilt from its twin before its
        // next use.
        if !twins_current {
            return Err(e);
        }
        copy(
            g,
            encoder,
            (src.twin(sr.level).expect("twin"), 0, s_twin.origin),
            (dst.twin(dr.level).expect("twin"), 0, d_twin.origin),
            s_twin.extent,
            one_subresource,
            twin_block,
        )?;
        dst.set_current(dr.level, dst_layer, false, true);
        return Ok(());
    }
    // Stored texels: both stored textures current, then the copy.
    for t in [src, dst] {
        if mem.prepare_texture(g, encoder, t)? == Prepare::Decode {
            unreachable!("only decoded formats refresh on the CPU");
        }
    }
    copy(
        g,
        encoder,
        (src.gpu(), sr.level, s_stored.origin),
        (dst.gpu(), dr.level, d_stored.origin),
        s_stored.extent,
        one_subresource,
        stored_block,
    )?;
    if twins_current {
        copy(
            g,
            encoder,
            (src.twin(sr.level).expect("twin"), 0, s_twin.origin),
            (dst.twin(dr.level).expect("twin"), 0, d_twin.origin),
            s_twin.extent,
            one_subresource,
            twin_block,
        )?;
    }
    dst.set_current(dr.level, dst_layer, true, twins_current);
    Ok(())
}

/// The part of a texture `generateMipmapsForTexture:` makes: a texture's, or a view's levels and slices of its root,
/// in the view's format.
pub struct MipWindow<A: Api> {
    /// `MTLPixelFormat` of the texture or view.
    pub format: u32,
    pub resolution: Resolution,
    /// The root's levels: the first is the source of the rest.
    pub levels: std::ops::Range<u32>,
    /// The root's array slices (ignored for a 3D texture, whose levels are made whole).
    pub slices: std::ops::Range<u32>,
    /// The root's alias that holds a `Copy` view's texels, which the levels are then made in.
    pub alias: Option<Rc<Alias<A>>>,
}

/// `generateMipmapsForTexture:`: every level from the one above it. Metal requires "A `mipmapLevelCount` property
/// that's greater than `1`" and "A `pixelFormat` that's color-renderable and color-filterable"
/// (`generateMipmaps(for:)`), and states that "The image filtering that GPU drivers use to generate the mipmaps may
/// vary by the feature families": the filter is the implementation's (plan section 9, row 10). Here each texel of a
/// level is the source level sampled with a linear filter at the texel's centre: the average of the 2×2 (3D: 2×2×2)
/// texels it covers when the source dimensions are even, a weighted average otherwise; in linear space for sRGB
/// formats, whose views decode and encode. Through a view, the levels and slices are the view's, and the format.
pub fn generate_mipmaps<A: Api>(
    mem: &Memory<A>,
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    t: &Texture<A>,
    w: &MipWindow<A>,
) -> Result<(), MemError> {
    const CALL: &str = "-[MTLBlitCommandEncoder generateMipmapsForTexture:]";
    if w.levels.end - w.levels.start < 2 {
        return Err(MemError::undefined(CALL, "a texture with one mipmap level"));
    }
    let f = match w.resolution {
        Resolution::Native(f) => f,
        Resolution::Emulated(_) => {
            return Err(MemError::unimplemented(format!(
                "{CALL}: a texture of pixel format {} stored through an emulation",
                w.format
            )));
        }
        Resolution::StencilAspect => return Err(MemError::undefined(CALL, "a stencil format")),
    };
    let caps = f.caps();
    let features = mem.features();
    let filterable = match caps.sample {
        SampleType::Float => true,
        SampleType::FloatIf(feature) => features.has(feature),
        _ => false,
    };
    if !caps.aspects.is_empty() || f.is_compressed() || matches!(caps.sample, SampleType::Uint | SampleType::Sint) {
        return Err(MemError::undefined(
            CALL,
            format!("pixel format {} is not color-renderable and filterable", w.format),
        ));
    }
    let renders = match caps.render {
        Cond::Always => true,
        Cond::Never => false,
        Cond::Feature(feature) => features.has(feature),
    };
    let gpu = w.alias.as_ref().map_or(t.gpu(), |a| a.gpu());
    if !filterable || !renders || gpu.usage() & crate::gpu::idl::TextureUsage::RENDER_ATTACHMENT == 0 {
        return Err(MemError::host(format!(
            "{CALL}: {} textures cannot be filtered and rendered on this host",
            caps.name
        )));
    }
    match &w.alias {
        Some(a) => mem.prepare_alias(g, encoder, t, a, (w.levels.clone(), w.slices.clone()))?,
        None => {
            if mem.prepare_texture(g, encoder, t)? == Prepare::Decode {
                unreachable!("native formats are not decoded");
            }
        }
    }
    let format = idl_format(f);
    let three_d = t.geometry.kind == TextureType::D3;
    for level in w.levels.start + 1..w.levels.end {
        let size = t.geometry.level_extent(level);
        let layers = if three_d { 0..size.2 } else { w.slices.clone() };
        for layer in layers {
            let m = MipLevel { texture: gpu, format, level, layer, three_d, size };
            encode_mip_level(mem, g, encoder, &m)?;
        }
    }
    let (levels, slices) = (w.levels.start + 1..w.levels.end, w.slices.clone());
    match &w.alias {
        Some(a) => mem.note_alias_write(t, a, levels, slices),
        None => mem.note_texture_write(t, levels, slices),
    }
    Ok(())
}
