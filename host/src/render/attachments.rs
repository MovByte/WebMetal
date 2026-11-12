//! A render pass's attachments: the `MTLRenderPassDescriptor` checked against Metal's rules and the A10X, and the
//! WebGPU render pass descriptors that carry out its load, store and resolve actions.
//!
//! **Load actions.** `load` and `clear` are WebGPU's. `dontCare` leaves "each pixel in the attachment ... allowed to
//! take on any value at the start of the rendering pass" (Apple, "Setting load and store actions"); the pass loads
//! the attachment, whose contents are one such value.
//!
//! **Store actions.** `store` is WebGPU's store; `dontCare` leaves the contents undefined, which WebGPU's `discard`
//! produces (the texture reads as zeros afterwards). `multisampleResolve` and `storeAndMultisampleResolve` resolve color
//! through WebGPU's resolve target; Metal leaves the resolve filter to the implementation, as WebGPU does
//! (`tables/limits.toml` question `6.4-resolve-filter`). Depth and stencil have no WebGPU resolve: the multisampled
//! attachment is stored and resolved by the passes of [`super::resolve`] (plan section 6.4). `unknown` is finalized by
//! `set*StoreAction:` records before `endEncoding`; one left unknown is a misuse whose result Metal does not define.
//!
//! **Split passes.** When the encoder must end the WebGPU pass in the middle of the Metal pass (barriers, per-primitive
//! passes), every WebGPU pass but the first loads the attachments and every pass stores them; the Metal store and
//! resolve actions are then applied by a final pass that loads and draws nothing ([`Targets::finish_descriptor`]).
//!
//! **Views.** A texture view ([`crate::mem::view`]) is an attachment through its root texture: the attachment's level
//! and slice are the view's, checked against the view and moved to the root's, and its WebGPU view (one level and one
//! slice, as WebGPU's render attachments are) is in the view's format. Attachments record the root's level and slice.

use std::rc::Rc;

use webmetal_wire::{Handle, RenderPassDescriptor};

use crate::exec::Table;
use crate::format::handling::{self, Emulation, Resolution, Write};
use crate::format::webgpu::{Aspect, HostFeatures, Table as FormatTable};
use crate::gpu::idl::{
    Color, ColorDict, LoadOp, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor as GpuPassDescriptor, StoreOp, TextureAspect, TextureFormat, TextureOrView, TextureUsage,
    TextureViewDescriptor, TextureViewDimension,
};
use crate::gpu::{Api, Float32, Float64, Texture as _};
use crate::mem::calls::Calls;
use crate::mem::layout::Geometry;
use crate::mem::linear::LinearTexture;
use crate::mem::options::StorageMode;
use crate::mem::texture::{Texture, idl_format};
use crate::mem::view::TextureView;

use super::error::{RenderError, Result};
use super::metal::{DepthResolveFilter, LoadAction, StencilResolveFilter, StoreAction};

pub(crate) const BEGIN: &str = "-[MTLCommandBuffer renderCommandEncoderWithDescriptor:]";

/// A texture a pass uses: a texture, a texture over a buffer's bytes, or a view of a texture.
#[derive(Clone)]
pub enum Tex<A: Api> {
    Plain(Rc<Texture<A>>),
    Linear(Rc<LinearTexture<A>>),
    View(Rc<TextureView<A>>),
}

impl<A: Api> Tex<A> {
    /// The texture behind a handle.
    pub fn lookup(objects: &Table, handle: Handle) -> Result<Tex<A>> {
        let entry = objects
            .get(handle)
            .ok_or_else(|| RenderError::invalid(format!("handle {:#x} is not live", handle.get())))?;
        let p = entry.payload.clone().ok_or_else(|| {
            RenderError::unimplemented(format!("texture {:#x} has no storage the render path knows", handle.get()))
        })?;
        let p = match p.downcast::<Texture<A>>() {
            Ok(t) => return Ok(Tex::Plain(t)),
            Err(p) => p,
        };
        let p = match p.downcast::<LinearTexture<A>>() {
            Ok(lt) if lt.texture.is_some() => return Ok(Tex::Linear(lt)),
            Ok(_) => {
                return Err(RenderError::unimplemented(format!(
                    "texture {:#x} is of type MTLTextureTypeTextureBuffer, which shaders read through its buffer (plan \
                     section 5.7); the render path binds it neither as an attachment nor as a texture",
                    handle.get()
                )));
            }
            Err(p) => p,
        };
        match p.downcast::<TextureView<A>>() {
            Ok(v) => Ok(Tex::View(v)),
            Err(_) => Err(RenderError::invalid(format!("handle {:#x} is not a texture", handle.get()))),
        }
    }

    /// The texture whose storage the pass uses: a view's root.
    pub fn texture(&self) -> &Rc<Texture<A>> {
        match self {
            Tex::Plain(t) => t,
            Tex::Linear(lt) => lt.texture.as_ref().expect("checked at lookup"),
            Tex::View(v) => &v.root,
        }
    }

    /// The view, for a view.
    pub fn view(&self) -> Option<&Rc<TextureView<A>>> {
        match self {
            Tex::View(v) => Some(v),
            _ => None,
        }
    }

    /// The geometry Metal's rules apply to: a view's own.
    pub fn geometry(&self) -> &Geometry {
        match self {
            Tex::View(v) => &v.geometry,
            _ => &self.texture().geometry,
        }
    }

    /// `MTLPixelFormat`: a view's own.
    pub fn format(&self) -> u32 {
        match self {
            Tex::View(v) => v.format,
            _ => self.texture().format,
        }
    }

    /// How the host provides the pixel format: a view's own.
    pub fn resolution(&self) -> Resolution {
        match self {
            Tex::View(v) => v.resolution,
            _ => self.texture().resolution,
        }
    }

    /// The root's levels and array slices the texture covers (a 3D texture's depth images are one slice).
    pub fn root_window(&self) -> (std::ops::Range<u32>, std::ops::Range<u32>) {
        match self {
            Tex::View(v) => (v.root_levels(), v.root_slices()),
            _ => {
                let g = &self.texture().geometry;
                (0..g.mip_levels, 0..g.slices())
            }
        }
    }
}

/// Where a resolve goes.
pub struct ResolveTarget<A: Api> {
    pub texture: Tex<A>,
    pub level: u32,
    pub slice: u32,
}

/// One color attachment.
pub struct ColorAttachment<A: Api> {
    pub index: u32,
    pub texture: Tex<A>,
    pub level: u32,
    pub slice: u32,
    pub view: A::TextureView,
    /// The 3D depth plane, for an attachment that is a slice of a 3D texture.
    pub depth_slice: Option<u32>,
    pub pixel_format: u32,
    pub load: LoadAction,
    pub clear: Color,
    pub store: StoreAction,
    pub resolve: Option<(ResolveTarget<A>, A::TextureView)>,
}

/// One aspect (depth or stencil) of the depth-stencil attachment.
pub struct AspectAttachment<A: Api> {
    pub load: LoadAction,
    pub store: StoreAction,
    pub resolve: Option<ResolveTarget<A>>,
}

/// The depth-stencil attachment: Metal's depth and stencil attachments, which WebGPU takes as one.
pub struct DepthStencilAttachment<A: Api> {
    pub texture: Tex<A>,
    pub level: u32,
    pub slice: u32,
    pub view: A::TextureView,
    pub pixel_format: u32,
    pub format: TextureFormat,
    pub has_depth: bool,
    pub has_stencil: bool,
    /// Metal's depth attachment, with its clear depth.
    pub depth: Option<(AspectAttachment<A>, f32, DepthResolveFilter)>,
    /// Metal's stencil attachment, with its clear value.
    pub stencil: Option<(AspectAttachment<A>, u32, StencilResolveFilter)>,
}

/// The attachments of a pass.
pub struct Targets<A: Api> {
    pub colors: Vec<Option<ColorAttachment<A>>>,
    pub depth_stencil: Option<DepthStencilAttachment<A>>,
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
}

/// The final store actions, after the `set*StoreAction:` records of the encoder.
#[derive(Clone, Debug, Default)]
pub struct FinalStores {
    pub colors: Vec<Option<u64>>,
    pub depth: Option<u64>,
    pub stencil: Option<u64>,
}

fn f32_exact(v: f64, what: &str) -> Result<f32> {
    let f = v as f32;
    if !v.is_finite() || !f.is_finite() {
        return Err(RenderError::unimplemented(format!(
            "{what} {v} is outside binary32; Metal's conversion of it to the attachment is not established"
        )));
    }
    Ok(f)
}

fn f64v(v: f64) -> Float64 {
    Float64::new(v).expect("finite, checked")
}

/// The WebGPU clear value of a color attachment of `resolution`.
fn clear_color(resolution: Resolution, c: &webmetal_wire::ClearColor, index: u32) -> Result<Color> {
    let rgba = [c.red, c.green, c.blue, c.alpha];
    let what = format!("clear color of color attachment {index}");
    let storage = resolution.storage();
    let name = storage.caps().name;
    let integer = name.ends_with("uint") || name.ends_with("sint");
    let check = |v: f64, channel: usize| -> Result<f64> {
        if integer {
            let (lo, hi) = integer_range(name, channel);
            if v.fract() != 0.0 || v < lo || v > hi {
                return Err(RenderError::unimplemented(format!(
                    "{what}: {v} is not an integer of the attachment's range; Metal's conversion of it is not \
                     established"
                )));
            }
            Ok(v)
        } else {
            f32_exact(v, &what)?;
            Ok(v)
        }
    };
    let rgba = match resolution {
        Resolution::Native(_) | Resolution::Emulated(Emulation { write: Write::Native, .. }) => rgba,
        // A8Unorm: the stored red channel holds Metal's alpha (format module, Write::AlphaToRed).
        Resolution::Emulated(Emulation { write: Write::AlphaToRed, .. }) => [rgba[3], 0.0, 0.0, 0.0],
        Resolution::Emulated(e) => {
            return Err(RenderError::unimplemented(format!(
                "{what}: clearing an attachment stored as {} with write handling {:?} needs the format's encoding of \
                 the clear value, which the render path does not apply yet",
                e.storage.caps().name,
                e.write
            )));
        }
        Resolution::StencilAspect => {
            return Err(RenderError::undefined(BEGIN, "an X32_Stencil8 texture as a color attachment"));
        }
    };
    let rgba = [check(rgba[0], 0)?, check(rgba[1], 1)?, check(rgba[2], 2)?, check(rgba[3], 3)?];
    Ok(Color::ColorDict(ColorDict { r: f64v(rgba[0]), g: f64v(rgba[1]), b: f64v(rgba[2]), a: f64v(rgba[3]) }))
}

/// The range of channel `channel` of an integer format, by its WebGPU name.
fn integer_range(name: &str, channel: usize) -> (f64, f64) {
    let bits: i32 = if name == "rgb10a2uint" {
        if channel == 3 { 2 } else { 10 }
    } else if name.starts_with("r8") || name.starts_with("rg8") || name.starts_with("rgba8") {
        8
    } else if name.contains("16") {
        16
    } else {
        32
    };
    if name.ends_with("sint") {
        (-(2f64.powi(bits - 1)), 2f64.powi(bits - 1) - 1.0)
    } else {
        (0.0, 2f64.powi(bits) - 1.0)
    }
}

/// Check that a texture can be an attachment of a pass and return the WebGPU view of `(level, slice, depth_plane)`, with
/// the root's level and slice.
fn attachment_view<A: Api>(
    g: &impl Calls<A>,
    tex: &Tex<A>,
    level: u64,
    slice: u64,
    depth_plane: u64,
    what: &str,
    aspect: TextureAspect,
) -> Result<(A::TextureView, u32, u32, Option<u32>)> {
    let t = tex.texture();
    let geo = tex.geometry();
    if t.gpu().usage() & TextureUsage::RENDER_ATTACHMENT == 0 {
        return Err(RenderError::undefined(
            BEGIN,
            format!("{what}: the texture was not created with MTLTextureUsageRenderTarget"),
        ));
    }
    let level = u32::try_from(level).ok().filter(|&l| l < geo.mip_levels).ok_or_else(|| {
        RenderError::undefined(BEGIN, format!("{what}: level {level} of a texture with {} levels", geo.mip_levels))
    })?;
    let (_, _, depth) = geo.level_extent(level);
    let (slice, depth_slice) = if geo.kind.is_3d() {
        if slice != 0 {
            return Err(RenderError::undefined(BEGIN, format!("{what}: slice {slice} of a 3D texture")));
        }
        let plane = u32::try_from(depth_plane).ok().filter(|&p| p < depth).ok_or_else(|| {
            RenderError::undefined(BEGIN, format!("{what}: depth plane {depth_plane} of a level {depth} deep"))
        })?;
        (0, Some(plane))
    } else {
        if depth_plane != 0 {
            return Err(RenderError::undefined(BEGIN, format!("{what}: depth plane {depth_plane} of a 2D texture")));
        }
        let s = u32::try_from(slice).ok().filter(|&s| s < geo.slices()).ok_or_else(|| {
            RenderError::undefined(BEGIN, format!("{what}: slice {slice} of a texture with {} slices", geo.slices()))
        })?;
        (s, None)
    };
    let (level, slice) = match tex.view() {
        Some(v) => (v.root_level(level), if geo.kind.is_3d() { slice } else { v.root_slice(slice) }),
        None => (level, slice),
    };
    let mut d = TextureViewDescriptor::new();
    d.label = format!("{} {what}", t.label);
    d.base_mip_level = level;
    d.mip_level_count = Some(1);
    d.aspect = aspect;
    if let Some(f) = tex.view().and_then(|v| v.view_format()) {
        d.format = Some(f);
        d.usage = TextureUsage::RENDER_ATTACHMENT;
    }
    if geo.kind.is_3d() {
        d.dimension = Some(TextureViewDimension::D3);
    } else {
        d.dimension = Some(TextureViewDimension::D2);
        d.base_array_layer = slice;
        d.array_layer_count = Some(1);
    }
    let view = g
        .call("GPUTexture.createView", &d.label, |_| t.gpu().create_view(&d))
        .map_err(|e| RenderError::gpu("GPUTexture.createView", e))?;
    Ok((view, level, slice, depth_slice))
}

fn resolve_target<A: Api>(
    g: &impl Calls<A>,
    objects: &Table,
    handle: Option<Handle>,
    level: u64,
    slice: u64,
    depth_plane: u64,
    what: &str,
) -> Result<Option<(ResolveTarget<A>, A::TextureView)>> {
    let Some(h) = handle else { return Ok(None) };
    let tex = Tex::<A>::lookup(objects, h)?;
    if tex.geometry().kind.is_3d() {
        return Err(RenderError::unimplemented(format!(
            "{what}: a resolve into a 3D texture (WebGPU resolve targets are 2D views)"
        )));
    }
    let (view, level, slice, _) =
        attachment_view(g, &tex, level, slice, depth_plane, &format!("{what} resolve texture"), TextureAspect::All)?;
    Ok(Some((ResolveTarget { texture: tex, level, slice }, view)))
}

fn store_action(initial: u64, finalized: Option<u64>, what: &str) -> Result<StoreAction> {
    let s = StoreAction::from_value(finalized.unwrap_or(initial), BEGIN)?;
    match s {
        StoreAction::Unknown => Err(RenderError::undefined(
            "-[MTLRenderCommandEncoder endEncoding]",
            format!(
                "the store action of {what} is still MTLStoreActionUnknown (MTLRenderCommandEncoder.h: it must be finalized before endEncoding)"
            ),
        )),
        StoreAction::CustomSampleDepthStore => Err(RenderError::host(
            "custom-sample-positions: MTLStoreActionCustomSampleDepthStore stores depth at programmable sample positions; WebGPU mandates the standard sample pattern (tables/refusals.md)",
        )),
        s => Ok(s),
    }
}

fn store_options(options: u64, what: &str) -> Result<()> {
    match options {
        0 => Ok(()),
        1 => Err(RenderError::host(format!(
            "custom-sample-positions: {what} has MTLStoreActionOptionCustomSamplePositions; WebGPU mandates the \
             standard sample pattern (tables/refusals.md)"
        ))),
        v => Err(RenderError::undefined(BEGIN, format!("{what}: {v:#x} is not an MTLStoreActionOptions value"))),
    }
}

impl<A: Api> Targets<A> {
    /// Check the descriptor and make the attachments' views. `stores` are the final store actions; `features` the
    /// host's.
    pub fn new(
        g: &impl Calls<A>,
        objects: &Table,
        d: &RenderPassDescriptor,
        stores: &FinalStores,
        features: HostFeatures,
    ) -> Result<Targets<A>> {
        // Descriptor fields the A10X does not have or the render path does not act on.
        if !d.sample_positions.is_empty() {
            return Err(RenderError::host(
                "custom-sample-positions: setSamplePositions:count: on the pass; WebGPU mandates the standard sample \
                 pattern (tables/refusals.md)",
            ));
        }
        if d.rasterization_rate_map.is_some() {
            return Err(RenderError::undefined(
                BEGIN,
                "a rasterization rate map: the A10X has none (maxRasterizationRateLayerCount = 0 in the device profile)",
            ));
        }
        if d.render_target_array_length != 0 {
            return Err(RenderError::unimplemented(
                "renderTargetArrayLength: the A10X reports supportsLayeredRendering = false (device profile) and its \
                 behavior for a layered pass is not established",
            ));
        }
        if d.tile_width != 0
            || d.tile_height != 0
            || d.threadgroup_memory_length != 0
            || d.imageblock_sample_length != 0
        {
            return Err(RenderError::unimplemented(
                "tile dimensions, threadgroup memory or imageblock sample length on the pass: tile shading is not \
                 part of the A10X's family",
            ));
        }
        // The descriptor always has its four attachment slots; one without a sample buffer samples nothing.
        if d.sample_buffer_attachments.iter().any(|a| a.sample_buffer.is_some()) {
            return Err(RenderError::unimplemented("counter sample buffer attachments of a render pass"));
        }
        for (set, what) in [
            (d.dither_enabled, "ditherEnabled"),
            (d.open_gl_mode_enabled, "openGLModeEnabled"),
            (d.point_coord_y_flip_enabled, "pointCoordYFlipEnabled"),
            (d.skip_empty_tiles_on_clear_enabled, "skipEmptyTilesOnClearEnabled"),
            (d.fine_grained_background_visibility_enabled, "fineGrainedBackgroundVisibilityEnabled"),
        ] {
            if set {
                return Err(RenderError::unimplemented(format!("MTLRenderPassDescriptor SPI {what}")));
            }
        }

        let mut size: Option<(u32, u32)> = None;
        let mut samples: Option<u32> = None;
        let mut agree = |t: &Texture<A>, level: u32, what: &str| -> Result<()> {
            let (w, h, _) = t.geometry.level_extent(level);
            if *size.get_or_insert((w, h)) != (w, h) {
                return Err(RenderError::undefined(BEGIN, format!("{what}: attachments of different sizes")));
            }
            if *samples.get_or_insert(t.geometry.sample_count) != t.geometry.sample_count {
                return Err(RenderError::undefined(BEGIN, format!("{what}: attachments of different sample counts")));
            }
            Ok(())
        };

        let mut colors: Vec<Option<ColorAttachment<A>>> = Vec::new();
        for (i, a) in d.color_attachments.iter().enumerate() {
            let what = format!("color attachment {i}");
            let Some(handle) = a.texture else {
                colors.push(None);
                continue;
            };
            if a.y_invert {
                return Err(RenderError::unimplemented(format!("{what}: yInvert (SPI)")));
            }
            let tex = Tex::<A>::lookup(objects, handle)?;
            let t = tex.texture().clone();
            if tex.resolution().storage().caps().table == FormatTable::DepthStencil
                || tex.resolution() == Resolution::StencilAspect
            {
                return Err(RenderError::undefined(BEGIN, format!("{what}: a depth or stencil format")));
            }
            let load = LoadAction::from_value(a.load_action, BEGIN)?;
            let store = store_action(a.store_action, stores.colors.get(i).copied().flatten(), &what)?;
            store_options(a.store_action_options, &what)?;
            if load == LoadAction::Load && t.storage == StorageMode::Memoryless {
                return Err(RenderError::undefined(BEGIN, format!("{what}: a memoryless attachment cannot be loaded")));
            }
            if store.stores() && t.storage == StorageMode::Memoryless {
                return Err(RenderError::undefined(
                    BEGIN,
                    format!("{what}: memoryless attachment content cannot be stored in memory"),
                ));
            }
            let (view, level, slice, depth_slice) =
                attachment_view(g, &tex, a.level, a.slice, a.depth_plane, &what, TextureAspect::All)?;
            agree(&t, level, &what)?;
            // The clear value matters only to a clear load action.
            let clear = if load == LoadAction::Clear {
                clear_color(tex.resolution(), &a.clear_color, i as u32)?
            } else {
                Color::ColorDict(ColorDict { r: f64v(0.0), g: f64v(0.0), b: f64v(0.0), a: f64v(0.0) })
            };
            let resolve = if store.resolves() {
                if t.geometry.sample_count < 2 {
                    return Err(RenderError::undefined(BEGIN, format!("{what}: a resolve of a single-sample texture")));
                }
                let r = resolve_target::<A>(
                    g,
                    objects,
                    a.resolve_texture,
                    a.resolve_level,
                    a.resolve_slice,
                    a.resolve_depth_plane,
                    &what,
                )?
                .ok_or_else(|| {
                    RenderError::undefined(BEGIN, format!("{what}: a resolve store action without a resolve texture"))
                })?;
                let rt = r.0.texture.texture();
                if r.0.texture.format() != tex.format() || rt.geometry.sample_count != 1 {
                    return Err(RenderError::undefined(
                        BEGIN,
                        format!("{what}: the resolve texture differs in pixel format or is multisampled"),
                    ));
                }
                let (rw, rh, _) = rt.geometry.level_extent(r.0.level);
                let (aw, ah, _) = t.geometry.level_extent(level);
                if (rw, rh) != (aw, ah) {
                    return Err(RenderError::undefined(BEGIN, format!("{what}: the resolve texture's size differs")));
                }
                handling::check_resolve(tex.resolution(), features).map_err(RenderError::refused)?;
                Some(r)
            } else {
                None
            };
            while colors.len() < i {
                colors.push(None);
            }
            let pixel_format = tex.format();
            colors.push(Some(ColorAttachment {
                index: i as u32,
                texture: tex,
                level,
                slice,
                view,
                depth_slice,
                pixel_format,
                load,
                clear,
                store,
                resolve,
            }));
        }

        // Trailing attachments without a texture are dropped: the pass is the same, as GPURenderPassLayout equality
        // compares color formats "ignoring any trailing nulls", and wgpu-native compares the lists with them.
        while colors.last().is_some_and(Option::is_none) {
            colors.pop();
        }
        let depth_stencil = Self::depth_stencil(g, objects, d, stores, &mut agree)?;
        let (width, height) = size.ok_or_else(|| {
            RenderError::unimplemented(
                "a render pass without attachments: WebGPU requires at least one attachment (renderTargetWidth and \
                 renderTargetHeight passes are not emulated)",
            )
        })?;
        if (d.render_target_width != 0 && d.render_target_width != u64::from(width))
            || (d.render_target_height != 0 && d.render_target_height != u64::from(height))
        {
            return Err(RenderError::unimplemented(format!(
                "renderTargetWidth/Height {}x{} smaller than the {width}x{height} attachments",
                d.render_target_width, d.render_target_height
            )));
        }
        Ok(Targets { colors, depth_stencil, width, height, sample_count: samples.unwrap_or(1) })
    }

    fn depth_stencil(
        g: &impl Calls<A>,
        objects: &Table,
        d: &RenderPassDescriptor,
        stores: &FinalStores,
        agree: &mut impl FnMut(&Texture<A>, u32, &str) -> Result<()>,
    ) -> Result<Option<DepthStencilAttachment<A>>> {
        let depth = d.depth_attachment.as_ref().filter(|a| a.texture.is_some());
        let stencil = d.stencil_attachment.as_ref().filter(|a| a.texture.is_some());
        let (handle, level, slice, depth_plane) = match (depth, stencil) {
            (None, None) => return Ok(None),
            (Some(a), None) => (a.texture.unwrap(), a.level, a.slice, a.depth_plane),
            (None, Some(s)) => (s.texture.unwrap(), s.level, s.slice, s.depth_plane),
            (Some(a), Some(s)) => {
                if a.texture != s.texture || a.level != s.level || a.slice != s.slice || a.depth_plane != s.depth_plane
                {
                    return Err(RenderError::unimplemented(
                        "separate depth and stencil textures (or subresources): WebGPU has one depth-stencil \
                         attachment",
                    ));
                }
                (a.texture.unwrap(), a.level, a.slice, a.depth_plane)
            }
        };
        if depth.is_some_and(|a| a.y_invert) || stencil.is_some_and(|s| s.y_invert) {
            return Err(RenderError::unimplemented("depth or stencil attachment yInvert (SPI)"));
        }
        let tex = Tex::<A>::lookup(objects, handle)?;
        let t = tex.texture().clone();
        let storage = match tex.resolution() {
            Resolution::Native(f) if f.caps().table == FormatTable::DepthStencil => f,
            Resolution::StencilAspect => {
                return Err(RenderError::unimplemented("an X32_Stencil8 view as the stencil attachment"));
            }
            _ => return Err(RenderError::undefined(BEGIN, "a color format as the depth or stencil attachment")),
        };
        let aspects = storage.caps().aspects;
        let has_depth = aspects.iter().any(|a| a.aspect == Aspect::Depth);
        let has_stencil = aspects.iter().any(|a| a.aspect == Aspect::Stencil);
        if depth.is_some() && !has_depth {
            return Err(RenderError::undefined(BEGIN, "a stencil-only format as the depth attachment"));
        }
        if stencil.is_some() && !has_stencil {
            return Err(RenderError::undefined(BEGIN, "a depth-only format as the stencil attachment"));
        }
        let (view, level, slice, depth_slice) =
            attachment_view(g, &tex, level, slice, depth_plane, "depth-stencil attachment", TextureAspect::All)?;
        if depth_slice.is_some() {
            return Err(RenderError::undefined(BEGIN, "a 3D texture as the depth or stencil attachment"));
        }
        agree(&t, level, "depth-stencil attachment")?;
        let memoryless = t.storage == StorageMode::Memoryless;

        let aspect = |load: u64,
                      store: u64,
                      finalized: Option<u64>,
                      options: u64,
                      resolve: (Option<Handle>, u64, u64, u64),
                      what: &str|
         -> Result<AspectAttachment<A>> {
            let load = LoadAction::from_value(load, BEGIN)?;
            let store = store_action(store, finalized, what)?;
            store_options(options, what)?;
            if load == LoadAction::Load && memoryless {
                return Err(RenderError::undefined(BEGIN, format!("{what}: a memoryless attachment cannot be loaded")));
            }
            if store.stores() && memoryless {
                return Err(RenderError::undefined(
                    BEGIN,
                    format!("{what}: memoryless attachment content cannot be stored in memory"),
                ));
            }
            let resolve = if store.resolves() {
                if t.geometry.sample_count < 2 {
                    return Err(RenderError::undefined(BEGIN, format!("{what}: a resolve of a single-sample texture")));
                }
                let (target, _) = resolve_target::<A>(g, objects, resolve.0, resolve.1, resolve.2, resolve.3, what)?
                    .ok_or_else(|| {
                        RenderError::undefined(
                            BEGIN,
                            format!("{what}: a resolve store action without a resolve texture"),
                        )
                    })?;
                let rt = target.texture.texture();
                if target.texture.format() != tex.format() || rt.geometry.sample_count != 1 {
                    return Err(RenderError::undefined(
                        BEGIN,
                        format!("{what}: the resolve texture differs in pixel format or is multisampled"),
                    ));
                }
                let (w, h, _) = rt.geometry.level_extent(target.level);
                let (aw, ah, _) = t.geometry.level_extent(level);
                if (w, h) != (aw, ah) {
                    return Err(RenderError::undefined(BEGIN, format!("{what}: the resolve texture's size differs")));
                }
                Some(target)
            } else {
                None
            };
            Ok(AspectAttachment { load, store, resolve })
        };

        let depth_part = match depth {
            Some(a) => {
                let att = aspect(
                    a.load_action,
                    a.store_action,
                    stores.depth,
                    a.store_action_options,
                    (a.resolve_texture, a.resolve_level, a.resolve_slice, a.resolve_depth_plane),
                    "depth attachment",
                )?;
                if att.load == LoadAction::Clear && !(0.0..=1.0).contains(&a.clear_depth) {
                    return Err(RenderError::unimplemented(format!(
                        "clearDepth {} outside [0, 1]: Metal's handling of it is not established, WebGPU requires \
                         [0, 1]",
                        a.clear_depth
                    )));
                }
                let filter = DepthResolveFilter::from_value(a.depth_resolve_filter, BEGIN)?;
                Some((att, a.clear_depth as f32, filter))
            }
            None => None,
        };
        let stencil_part = match stencil {
            Some(s) => {
                let att = aspect(
                    s.load_action,
                    s.store_action,
                    stores.stencil,
                    s.store_action_options,
                    (s.resolve_texture, s.resolve_level, s.resolve_slice, s.resolve_depth_plane),
                    "stencil attachment",
                )?;
                if att.load == LoadAction::Clear && s.clear_stencil > 0xff {
                    return Err(RenderError::unimplemented(format!(
                        "clearStencil {:#x} wider than the 8-bit stencil: Metal's handling of the high bits is not \
                         established",
                        s.clear_stencil
                    )));
                }
                let filter = StencilResolveFilter::from_value(s.stencil_resolve_filter, BEGIN)?;
                Some((att, s.clear_stencil, filter))
            }
            None => None,
        };
        let pixel_format = tex.format();
        Ok(Some(DepthStencilAttachment {
            texture: tex,
            level,
            slice,
            view,
            pixel_format,
            format: idl_format(storage),
            has_depth,
            has_stencil,
            depth: depth_part,
            stencil: stencil_part,
        }))
    }

    /// The Metal pixel formats of the color attachments, by index (0 where there is none).
    pub fn color_formats(&self) -> Vec<u32> {
        self.colors.iter().map(|c| c.as_ref().map_or(0, |c| c.pixel_format)).collect()
    }

    /// The textures and subresources the pass writes: attachments and resolve targets.
    pub fn written(&self) -> Vec<(Tex<A>, u32, u32)> {
        let mut out = Vec::new();
        for c in self.colors.iter().flatten() {
            out.push((c.texture.clone(), c.level, c.depth_slice.unwrap_or(c.slice)));
            if let Some((r, _)) = &c.resolve {
                out.push((r.texture.clone(), r.level, r.slice));
            }
        }
        if let Some(ds) = &self.depth_stencil {
            out.push((ds.texture.clone(), ds.level, ds.slice));
            for r in [
                ds.depth.as_ref().and_then(|d| d.0.resolve.as_ref()),
                ds.stencil.as_ref().and_then(|s| s.0.resolve.as_ref()),
            ]
            .into_iter()
            .flatten()
            {
                out.push((r.texture.clone(), r.level, r.slice));
            }
        }
        out
    }

    /// The descriptor of a WebGPU pass of the Metal pass. `first`: the pass begins the Metal pass and applies its
    /// load actions (else it loads). `last`: the pass ends the Metal pass and applies its store and color resolve
    /// actions (else it stores).
    pub fn descriptor<'a>(
        &'a self,
        label: &str,
        first: bool,
        last: bool,
        occlusion: Option<&'a A::QuerySet>,
    ) -> GpuPassDescriptor<'a, A> {
        let colors = self
            .colors
            .iter()
            .map(|c| {
                c.as_ref().map(|c| {
                    let load = if first && c.load == LoadAction::Clear { LoadOp::Clear } else { LoadOp::Load };
                    let store = if !last || c.store.stores() { StoreOp::Store } else { StoreOp::Discard };
                    let mut a = RenderPassColorAttachment::new(TextureOrView::TextureView(&c.view), load, store);
                    a.depth_slice = c.depth_slice;
                    if load == LoadOp::Clear {
                        a.clear_value = Some(c.clear.clone());
                    }
                    if last && let Some((_, v)) = &c.resolve {
                        a.resolve_target = Some(TextureOrView::TextureView(v));
                    }
                    a
                })
            })
            .collect();
        let mut d = GpuPassDescriptor::new(colors);
        d.label = label.to_string();
        d.occlusion_query_set = occlusion;
        d.depth_stencil_attachment = self.depth_stencil.as_ref().map(|ds| self.depth_stencil_ops(ds, first, last));
        d
    }

    fn depth_stencil_ops<'a>(
        &self,
        ds: &'a DepthStencilAttachment<A>,
        first: bool,
        last: bool,
    ) -> RenderPassDepthStencilAttachment<'a, A> {
        let mut a = RenderPassDepthStencilAttachment::new(TextureOrView::TextureView(&ds.view));
        // An aspect Metal does not attach keeps its contents: load and store. An aspect with a depth or stencil
        // resolve is stored so the resolve pass can read it; its contents after the pass are Metal's to leave
        // undefined (multisampleResolve), and the stored samples are one such value.
        let ops = |att: Option<&AspectAttachment<A>>| -> (LoadOp, StoreOp, bool) {
            match att {
                None => (LoadOp::Load, StoreOp::Store, false),
                Some(at) => {
                    let clear = first && at.load == LoadAction::Clear;
                    let load = if clear { LoadOp::Clear } else { LoadOp::Load };
                    let store = if !last || at.store.stores() || at.resolve.is_some() {
                        StoreOp::Store
                    } else {
                        StoreOp::Discard
                    };
                    (load, store, clear)
                }
            }
        };
        if ds.has_depth {
            let (load, store, clear) = ops(ds.depth.as_ref().map(|d| &d.0));
            a.depth_load_op = Some(load);
            a.depth_store_op = Some(store);
            if clear {
                a.depth_clear_value = Some(Float32::new(ds.depth.as_ref().unwrap().1).expect("checked in [0, 1]"));
            }
        }
        if ds.has_stencil {
            let (load, store, clear) = ops(ds.stencil.as_ref().map(|s| &s.0));
            a.stencil_load_op = Some(load);
            a.stencil_store_op = Some(store);
            if clear {
                a.stencil_clear_value = ds.stencil.as_ref().unwrap().1;
            }
        }
        a
    }
}

/// The aspect of a view of `format` for a shader binding of `sample_type`.
pub fn binding_aspect(format: TextureFormat, depth: bool) -> TextureAspect {
    let f = crate::format::webgpu::WebGpuFormat::from_name(format.as_str()).expect("the format tables share names");
    let aspects = f.caps().aspects;
    if aspects.len() < 2 {
        TextureAspect::All
    } else if depth {
        TextureAspect::DepthOnly
    } else {
        TextureAspect::StencilOnly
    }
}
