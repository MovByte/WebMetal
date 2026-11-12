//! Texture views (`newTextureViewWithPixelFormat:` and its five variants): the payload of a view handle and how a view
//! reaches the storage of the texture it was made from.
//!
//! **What a view is.** On the A10X a view is its own object over its parent's storage: `-[IOGPUMetalResource
//! initWithResource:]` shares the parent's kernel resource, and the view keeps its parent alive. Its label is its own,
//! its purgeable state is not settable (IOGPU makes it not purgeable, so `setPurgeableState:` never reaches the host
//! for it), and releasing it releases only the view. Here a [`TextureView`] holds the root texture, the [`Texture`]
//! whose WebGPU texture is the storage: the parent, or for a view of a view the parent's root, with the level and slice
//! bases composed. The view's pixel format, type, levels and slices are its own.
//!
//! **Formats.** How the view's format is provided over the root's is the [`ViewPlan`] of the two formats
//! ([`crate::format::reinterpret::view_plan`]):
//!
//! - `Same`, `SharedStorage`, `ViewFormat` and `StencilAspect` are WebGPU views of the root ([`TextureView::describe`]):
//!   the root's format or its sRGB sibling (which every texture lists in `viewFormats`), or the stencil aspect of a
//!   `depth32float-stencil8` texture.
//! - `Copy` and `FromBlocks` need a second texture kept equal to the root through every CPU, blit and GPU write of
//!   either; the memory module does not keep such a texture yet, so those views are refused when they are created
//!   (status 2), and nothing ever reads a stale copy.
//!
//! **Bytes.** Under `Same`, `SharedStorage` and `ViewFormat` the view's Metal bytes are the root's bytes: the formats
//! store the same bytes with the same transforms, or are sRGB siblings that differ only in how a shader converts them.
//! `replaceRegion:`, `getBytes:` and blits through such a view are the root's transfers of the root's subresources at
//! the view's bases ([`TextureView::root_level`], [`TextureView::root_slice`]), after Metal's rules are checked
//! against the view. The stencil view's bytes are one aspect of the root's; its transfers are refused with the reason.
//!
//! **What is checked when a view is made.** Metal's own validation of the arguments
//! (`_mtlValidateArgumentsForTextureViewOnDevice`, which asserts in release builds as well) runs in the guest. The host
//! checks what it relies on and refuses the rest with the rule it breaks: levels and slices inside the parent; the
//! type casts the documentation lists; the slice count of the view's type; the `pixelFormatView` usage for a view that
//! does more than convert between sRGB and linear (the debug layer's `_validateTextureView`); and that the A10X makes
//! the view at all (no view of a memoryless texture: its layout is nil).

use std::ops::Range;
use std::rc::Rc;

use crate::format::handling::{self, Dimension, Resolution, Unavailable};
use crate::format::reinterpret::{ViewPlan, view_plan};
use crate::format::table::METAL_FORMATS;
use crate::gpu::Api;
use crate::gpu::idl::{TextureAspect, TextureFormat, TextureViewDescriptor, TextureViewDimension};
use webmetal_wire::{Handle, Range as WireRange, TextureSwizzleChannels};

use super::buffer::label;
use super::error::MemError;
use super::layout::{Geometry, TextureType};
use super::memory::Memory;
use super::options::StorageMode;
use super::texture::{Texture, idl_format, usage};
use crate::embed::TaskId;

/// `MTLTextureSwizzleChannelsDefault`: red 2, green 3, blue 4, alpha 5 (`MTLTextureSwizzle` of `MTLTexture.h`).
pub const IDENTITY_SWIZZLE: [u8; 4] = [2, 3, 4, 5];

/// A texture view: the payload of a view handle in the executor's object table.
pub struct TextureView<A: Api> {
    /// The texture whose storage the view shares.
    pub root: Rc<Texture<A>>,
    /// `MTLPixelFormat` of the view.
    pub format: u32,
    /// How the host provides the view's format.
    pub resolution: Resolution,
    /// The view's own geometry: its type, the root's extent at its first level, its slices and its levels.
    pub geometry: Geometry,
    /// The root's level of the view's level 0.
    pub base_level: u32,
    /// The root's slice of the view's slice 0 (0 for a view of a 3D texture).
    pub base_slice: u32,
    /// The view's `MTLTextureSwizzle` channels (red, green, blue, alpha).
    pub swizzle: [u8; 4],
    pub plan: ViewPlan,
    pub label: String,
}

impl<A: Api> TextureView<A> {
    /// The root's level of the view's `level`.
    pub fn root_level(&self, level: u32) -> u32 {
        self.base_level + level
    }

    /// The root's slice of the view's `slice`.
    pub fn root_slice(&self, slice: u32) -> u32 {
        self.base_slice + slice
    }

    /// The root's levels the view covers.
    pub fn root_levels(&self) -> Range<u32> {
        self.base_level..self.base_level + self.geometry.mip_levels
    }

    /// The root's array slices the view covers: for a 3D view, the one subresource layer of each level.
    pub fn root_slices(&self) -> Range<u32> {
        self.base_slice..self.base_slice + self.geometry.slices()
    }

    /// Are the view's bytes the root's (plans `Same`, `SharedStorage` and `ViewFormat`)? Otherwise `call` is refused
    /// with the reason.
    pub fn bytes_are_root(&self, call: &'static str) -> Result<(), MemError> {
        match self.plan {
            ViewPlan::Same | ViewPlan::SharedStorage | ViewPlan::ViewFormat(_) => Ok(()),
            ViewPlan::StencilAspect => Err(MemError::unimplemented(format!(
                "{call} through an X32_Stencil8 view: its bytes are the stencil aspect of the Depth32Float_Stencil8 \
                 texture, which the host moves only one aspect at a time through blits of the texture itself"
            ))),
            ViewPlan::Copy { .. } | ViewPlan::FromBlocks(_) => {
                unreachable!("views needing a second texture are refused when they are created")
            }
        }
    }

    /// The format of WebGPU views that show the view, when it is not the root's own: the sRGB sibling.
    pub fn view_format(&self) -> Option<TextureFormat> {
        match self.plan {
            ViewPlan::ViewFormat(f) => Some(idl_format(f)),
            _ => None,
        }
    }

    /// Can a WebGPU view of `dimension` show the view's slices (WebGPU section 6.2.1: one layer for `2d`, six for
    /// `cube`, a multiple of six for `cube-array`; `3d` of a 3D texture only)? A shader whose texture type does not
    /// take the view's type and slices is a binding Metal does not define.
    pub fn takes(&self, dimension: TextureViewDimension) -> bool {
        let (three_d, slices) = (self.root.geometry.kind.is_3d(), self.geometry.slices());
        match dimension {
            TextureViewDimension::D3 => three_d,
            TextureViewDimension::D1 | TextureViewDimension::D2 => !three_d && slices == 1,
            TextureViewDimension::D2Array => !three_d,
            TextureViewDimension::Cube => !three_d && slices == 6,
            TextureViewDimension::CubeArray => !three_d && slices.is_multiple_of(6),
        }
    }

    /// Set on `d` what makes a WebGPU view of the root show levels `levels` (of the view) and all of the view's slices
    /// in the view's format, for a binding of `usage` (a single `GPUTextureUsage` bit the root has): the sRGB sibling
    /// format with a usage of its own (a view's usage defaults to the texture's, and a sibling format may lack a
    /// capability of the root's, WebGPU section 6.2.1), or the stencil aspect.
    pub fn describe(&self, d: &mut TextureViewDescriptor, levels: Range<u32>, usage: u32) {
        if let Some(f) = self.view_format() {
            d.format = Some(f);
            d.usage = usage;
        }
        if self.plan == ViewPlan::StencilAspect {
            d.aspect = TextureAspect::StencilOnly;
        }
        d.base_mip_level = self.root_level(levels.start);
        d.mip_level_count = Some(levels.end - levels.start);
        if !self.root.geometry.kind.is_3d() {
            d.base_array_layer = self.base_slice;
            d.array_layer_count = Some(self.geometry.slices());
        }
    }
}

/// The parent a view request names.
pub enum Parent<'a, A: Api> {
    Texture(&'a Rc<Texture<A>>),
    View(&'a TextureView<A>),
}

/// The arguments of one of the view requests: the forms without a type, levels and slices take the parent's type, all
/// of its levels and all of its slices; the forms without a swizzle take the parent's swizzle (`-[AGXTexture
/// initWithTexture:pixelFormat:resourceIndex:]` and `-[IOGPUMetalTexture initWithTexture:pixelFormat:textureType:
/// levels:slices:]`, as `guest/Resources/WebMetalTexture.mm` records them). The SPI forms' resource index reaches the
/// A10X driver's `initImplWithTexture:...`, which does not read it, so it changes nothing here.
pub struct ViewArgs<'a> {
    pub call: &'static str,
    pub pixel_format: u64,
    pub texture_type: Option<u64>,
    pub levels: Option<&'a WireRange>,
    pub slices: Option<&'a WireRange>,
    pub swizzle: Option<&'a TextureSwizzleChannels>,
}

/// Metal's type casts of a view (`makeTextureView(pixelFormat:textureType:levels:slices:)` documentation: "1D to 1D;
/// 2D to 2D or 2D array; 2D array, cube or cube array to 2D, 2D array, cube or cube array; 3D to 3D"), and a view of
/// the parent's own type, which `newTextureViewWithPixelFormat:` makes of a texture of any type.
fn castable(parent: TextureType, view: TextureType) -> bool {
    use TextureType::*;
    parent == view
        || matches!((parent, view), (D2, D2Array) | (D2Array | Cube | CubeArray, D2 | D2Array | Cube | CubeArray))
}

/// The name of a Metal pixel format without its `MTLPixelFormat` prefix.
fn format_name(value: u32) -> Option<&'static str> {
    METAL_FORMATS.iter().find(|r| r.value == value).and_then(|r| r.name.strip_prefix("MTLPixelFormat"))
}

/// Do two Metal formats differ only in converting between sRGB and linear? `MTLPixelFormat.h` names the sRGB form of a
/// format with the suffix `_sRGB`, and that of an ASTC LDR format by replacing `_LDR` with `_sRGB`.
fn srgb_pair(a: u32, b: u32) -> bool {
    let linear = |v: u32| {
        format_name(v).map(|n| match n.strip_suffix("_sRGB") {
            Some(base) if base.starts_with("ASTC_") => format!("{base}_LDR"),
            Some(base) => base.to_string(),
            None => n.to_string(),
        })
    };
    a != b && linear(a).is_some_and(|l| Some(l) == linear(b))
}

fn range_u32(call: &'static str, what: &str, r: &WireRange, within: u32) -> Result<Range<u32>, MemError> {
    let end = r.location.checked_add(r.length);
    if r.length == 0 || end.is_none_or(|e| e > u64::from(within)) {
        return Err(MemError::undefined(
            call,
            format!("{what} ({}, {}) outside the parent's {within} {what}", r.location, r.length),
        ));
    }
    Ok(r.location as u32..(r.location + r.length) as u32)
}

impl<A: Api> Memory<A> {
    /// A view of `parent` (CreateTextureView and the other view requests).
    pub fn create_texture_view(
        &self,
        (task, handle): (TaskId, Handle),
        parent: Parent<'_, A>,
        a: &ViewArgs<'_>,
    ) -> Result<Rc<TextureView<A>>, MemError> {
        let call = a.call;
        // The parent's view of the root: its geometry, bases, format and swizzle.
        let (root, pg, (pbase_level, pbase_slice), pformat, pswizzle) = match parent {
            Parent::Texture(t) => {
                // The memory module refuses a descriptor swizzle that is not the identity (`build_texture`).
                (t.clone(), t.geometry, (0, 0), t.format, IDENTITY_SWIZZLE)
            }
            Parent::View(v) => (v.root.clone(), v.geometry, (v.base_level, v.base_slice), v.format, v.swizzle),
        };
        if root.surface.is_some() {
            return Err(MemError::unimplemented(format!(
                "{call} of a texture over an IOSurface plane: the A10X layout of its parent is not ported to the guest, \
                 which does not make the view"
            )));
        }
        if root.storage == StorageMode::Memoryless {
            return Err(MemError::invalid(format!(
                "{call} of a memoryless texture: the A10X driver makes no view of it (its view layout is nil), so the \
                 guest does not send the request"
            )));
        }

        // Type, levels and slices.
        let kind = match a.texture_type {
            Some(v) => TextureType::from_value(v, call)?,
            None => pg.kind,
        };
        if !castable(pg.kind, kind) {
            return Err(MemError::undefined(
                call,
                format!(
                    "a view of type {kind:?} of a texture of type {:?} is not a cast the texture view documentation \
                     lists",
                    pg.kind
                ),
            ));
        }
        let levels = match a.levels {
            Some(r) => range_u32(call, "mipmap levels", r, pg.mip_levels)?,
            None => 0..pg.mip_levels,
        };
        let slices = match a.slices {
            Some(r) => range_u32(call, "slices", r, pg.slices())?,
            None => 0..pg.slices(),
        };
        let count = slices.end - slices.start;
        let array_length = match kind {
            TextureType::Cube if count == 6 => 1,
            TextureType::CubeArray if count.is_multiple_of(6) => count / 6,
            TextureType::Cube | TextureType::CubeArray => {
                return Err(MemError::undefined(
                    call,
                    format!(
                        "{count} slices for a {kind:?} view: the slice range's length needs to be 6 for a cube and a \
                         multiple of 6 for a cube array (texture view documentation)"
                    ),
                ));
            }
            TextureType::D1Array | TextureType::D2Array | TextureType::D2MultisampleArray => count,
            _ if count == 1 => 1,
            _ => return Err(MemError::undefined(call, format!("{count} slices for a {kind:?} view, which has one"))),
        };
        let (width, height, depth) = pg.level_extent(levels.start);
        if matches!(kind, TextureType::Cube | TextureType::CubeArray) && width != height {
            return Err(MemError::unimplemented(format!(
                "{call}: a {kind:?} view of {width}x{height} slices: a WebGPU cube view needs square slices \
                 (GPUTexture.createView), and whether Metal makes one of slices that are not square is not established"
            )));
        }
        let geometry = Geometry {
            kind,
            width,
            height,
            depth,
            array_length,
            mip_levels: levels.end - levels.start,
            sample_count: pg.sample_count,
        };

        // Swizzle.
        let swizzle = match a.swizzle {
            Some(s) => [s.red, s.green, s.blue, s.alpha],
            None => pswizzle,
        };
        if let Some(c) = swizzle.iter().find(|&&c| c > 5) {
            return Err(MemError::undefined(call, format!("{c} is not an MTLTextureSwizzle value (MTLTexture.h)")));
        }
        if swizzle != IDENTITY_SWIZZLE {
            return Err(MemError::unimplemented(format!(
                "{call}: texture swizzle {swizzle:?} is not applied by the host yet"
            )));
        }

        // Pixel format.
        let format = u32::try_from(a.pixel_format)
            .map_err(|_| MemError::undefined(call, format!("pixel format {}", a.pixel_format)))?;
        if format != pformat
            && !srgb_pair(format, pformat)
            && root.usage != usage::UNKNOWN
            && root.usage & usage::PIXEL_FORMAT_VIEW == 0
        {
            // The debug layer: "texture usage (0x01) doesn't specify MTLTextureUsagePixelFormatView (0x10)"
            // (`texture.error-view-without-usage`); a view between sRGB and linear passes it
            // (`texture.view-srgb-without-usage`).
            return Err(MemError::undefined(
                call,
                format!(
                    "a view in pixel format {format} of a texture in pixel format {pformat} whose usage {:#x} lacks \
                     MTLTextureUsagePixelFormatView (debug layer, _validateTextureView)",
                    root.usage
                ),
            ));
        }
        let unavailable = |u: Unavailable| match u {
            Unavailable::NotOnDevice(why) => MemError::undefined(call, why.to_string()),
            Unavailable::Refused(r) => MemError::refused(r),
        };
        let dimension = if root.geometry.kind.is_3d() { Dimension::ThreeD } else { Dimension::Planar };
        if format != root.format && handling::resolve(root.format, self.features) != Ok(root.resolution) {
            return Err(MemError::unimplemented(format!(
                "{call}: a view in another pixel format of a texture stored in the form the host gives its format in 3D \
                 textures; the view plans compare the formats' 2D forms"
            )));
        }
        let plan = view_plan(root.format, format, self.features).map_err(unavailable)?;
        let resolution = match plan {
            ViewPlan::Same => root.resolution,
            _ => handling::resolve_texture(format, self.features, dimension).map_err(unavailable)?,
        };
        match plan {
            ViewPlan::Copy { storage, .. } => {
                return Err(MemError::unimplemented(format!(
                    "{call}: a view in pixel format {format} of a texture in pixel format {} needs a second texture in \
                     {} kept equal to the texture through every write of either, which the memory module does not \
                     keep yet",
                    root.format,
                    storage.name()
                )));
            }
            ViewPlan::FromBlocks(_) => {
                return Err(MemError::unimplemented(format!(
                    "{call}: a view in pixel format {format} of a compressed texture in pixel format {} needs a second \
                     texture filled from the texture's blocks and kept equal to them, which the memory module does not \
                     keep yet",
                    root.format
                )));
            }
            ViewPlan::ViewFormat(f) if !root.views_in(f) => {
                return Err(MemError::host(format!(
                    "{call}: the texture's WebGPU texture cannot be viewed in {}: it is not in its viewFormats",
                    f.name()
                )));
            }
            _ => {}
        }
        Ok(Rc::new(TextureView {
            root,
            format,
            resolution,
            geometry,
            base_level: pbase_level + levels.start,
            base_slice: pbase_slice + slices.start,
            swizzle,
            plan,
            label: label(task, "texture view", handle),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn casts_follow_the_documentation() {
        use TextureType::*;
        assert!(castable(D2, D2Array) && castable(D2, D2) && !castable(D2, Cube));
        assert!(castable(D2Array, Cube) && castable(Cube, D2) && castable(CubeArray, D2Array));
        assert!(castable(D1, D1) && !castable(D1, D1Array) && castable(D1Array, D1Array));
        assert!(castable(D3, D3) && !castable(D3, D2) && !castable(D2, D3));
        assert!(castable(D2Multisample, D2Multisample) && !castable(D2Multisample, D2));
    }

    #[test]
    fn srgb_pairs() {
        // RGBA8Unorm 70 and RGBA8Unorm_sRGB 71; RGBA8Uint 73; ASTC_4x4_sRGB 186 and ASTC_4x4_LDR 204; R8Unorm 10 and
        // R8Unorm_sRGB 11; BGRA8Unorm 80.
        assert!(srgb_pair(70, 71) && srgb_pair(71, 70));
        assert!(srgb_pair(186, 204) && srgb_pair(10, 11));
        assert!(!srgb_pair(70, 73) && !srgb_pair(70, 70) && !srgb_pair(80, 71));
    }

    #[test]
    fn ranges_inside_the_parent() {
        let r = |location, length| WireRange { location, length };
        assert_eq!(range_u32("t", "slices", &r(1, 2), 4).unwrap(), 1..3);
        assert!(range_u32("t", "slices", &r(3, 2), 4).is_err());
        assert!(range_u32("t", "slices", &r(0, 0), 4).is_err());
        assert!(range_u32("t", "slices", &r(u64::MAX, 2), 4).is_err());
    }
}
