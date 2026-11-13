//! Texture views that reinterpret a texture's pixel format (`newTextureViewWithPixelFormat:` and its variants).
//!
//! WebGPU views change only the `-srgb` suffix, and only to formats listed in the texture's `viewFormats`. Metal lets a
//! `pixelFormatView` texture be viewed in any format of the same texel size. A view the WebGPU texture cannot express
//! becomes a second texture kept equal to the first. Its contents move by texel copies through a buffer, which are
//! bit-preserving only when both formats are: WebGPU texel copies keep only finite, non-subnormal values of float
//! formats and may change the representation of snorm and packed formats (WebGPU section 11.2, quoted in
//! `tables/formats.toml` question `6.3-texel-copy-bit-exactness`), unless the host's copies of the format are proven
//! to keep every bit pattern (`crate::mem::proof`). Views between formats whose copies are not so proven are refused
//! (row `float-view-reinterpretation` of `tables/refusals.md`).
//!
//! Views of compressed textures are filled from the blocks the executor keeps for every compressed texture: decoded
//! with the view's scheme on hosts without the format, uploaded as blocks on hosts with it.

use super::decode::Scheme;
use super::handling::{Refusal, Resolution, Unavailable, Upload, resolve};
use super::webgpu::{HostFeatures, WebGpuFormat};

/// A byte transform between Metal's texel bytes and a stored texture's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    None,
    /// [`super::packed::swap_red_blue_10`] on each 32-bit texel (its own inverse).
    SwapRedBlue10,
}

/// How a view of a texture in another Metal format is provided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewPlan {
    /// The view's format is the texture's.
    Same,
    /// A WebGPU view of the texture in this format; the texture is created with it in `viewFormats`.
    ViewFormat(WebGpuFormat),
    /// The view reads the texture's own storage; only the shader's conversion differs.
    SharedStorage,
    /// The stencil aspect of the depth-stencil texture.
    StencilAspect,
    /// A second texture in `storage`, kept equal to the texture by bit-preserving texel copies through a buffer.
    /// `to_metal` turns the texture's stored bytes into Metal's bytes and `from_metal` Metal's bytes into the view's.
    Copy { storage: WebGpuFormat, to_metal: Transform, from_metal: Transform },
    /// A second texture filled from the texture's blocks: decoded with `Some(scheme)`, uploaded as blocks with
    /// `None`.
    FromBlocks(Option<Scheme>),
}

fn transform(upload: Upload) -> Option<Transform> {
    match upload {
        Upload::Bytes => Some(Transform::None),
        Upload::SwapRedBlue10 => Some(Transform::SwapRedBlue10),
        Upload::Decode(_) => None,
    }
}

/// Are two WebGPU formats the same but for the `-srgb` suffix (WebGPU section 6.1.2, view format compatibility)?
fn srgb_siblings(a: WebGpuFormat, b: WebGpuFormat) -> bool {
    a != b && a.name().trim_end_matches("-srgb") == b.name().trim_end_matches("-srgb")
}

fn is_compressed(r: Resolution) -> bool {
    match r {
        Resolution::Native(f) => f.is_compressed(),
        Resolution::Emulated(e) => matches!(e.upload, Upload::Decode(_)),
        Resolution::StencilAspect => false,
    }
}

const FLOAT_VIEW: Refusal = Refusal {
    id: "float-view-reinterpretation",
    reason: "WebGPU texel copies preserve only finite, non-subnormal values of float formats and may change snorm and \
             packed representations, and this host's copies of the format are not proven to keep its bits, so no \
             bit-exact copy connects the texture and the view.",
    source: "tables/refusals.md, float-view-reinterpretation",
};

/// How a view in Metal format `view` of a texture in Metal format `texture` is provided on a host with `features`,
/// whose texel copies keep every bit of a WebGPU format `f` when `proven(f)` (a bit-exact format, or one whose copies
/// the host proved). Metal's own rules for which views are valid (the `pixelFormatView` usage, equal texel sizes) are
/// the guest's to enforce; a pair whose stored texel sizes differ is refused here as well.
pub fn view_plan(
    texture: u32,
    view: u32,
    features: HostFeatures,
    proven: &dyn Fn(WebGpuFormat) -> bool,
) -> Result<ViewPlan, Unavailable> {
    let t = resolve(texture, features)?;
    if texture == view {
        return Ok(ViewPlan::Same);
    }
    let v = resolve(view, features)?;
    if let (Resolution::Native(WebGpuFormat::Depth32floatStencil8), Resolution::StencilAspect) = (t, v) {
        return Ok(ViewPlan::StencilAspect);
    }
    if is_compressed(t) || is_compressed(v) {
        if !(is_compressed(t) && is_compressed(v)) {
            return Err(Unavailable::Refused(Refusal {
                id: "compressed-uncompressed-view",
                reason: "A view between a compressed and an uncompressed format reinterprets blocks as texels; the \
                         emulated textures do not hold blocks.",
                source: "WebGPU section 6.1.2 (view formats differ only in -srgb)",
            }));
        }
        let (ts, vs) = (t.storage(), v.storage());
        // sRGB siblings that decode to the same codes: ETC2, EAC RGBA8 and PVRTC store 8-bit codes for both; native
        // compressed sRGB siblings are WebGPU view formats.
        let same_codes = match (t, v) {
            (Resolution::Emulated(a), Resolution::Emulated(b)) => !matches!(
                (a.upload, b.upload),
                (Upload::Decode(Scheme::Astc { .. }), _) | (_, Upload::Decode(Scheme::Astc { .. }))
            ),
            (Resolution::Native(_), Resolution::Native(_)) => true,
            _ => false,
        };
        if same_codes && srgb_siblings(ts, vs) {
            return Ok(ViewPlan::ViewFormat(vs));
        }
        return Ok(ViewPlan::FromBlocks(match v {
            Resolution::Emulated(e) => match e.upload {
                Upload::Decode(s) => Some(s),
                _ => unreachable!("compressed emulations decode"),
            },
            _ => None,
        }));
    }
    let (ts, vs) = (t.storage(), v.storage());
    let same_bytes = |r: Resolution| match r {
        Resolution::Native(_) => Some(Transform::None),
        Resolution::Emulated(e) => transform(e.upload),
        Resolution::StencilAspect => None,
    };
    let (to_metal, from_metal) = (same_bytes(t), same_bytes(v));
    if ts == vs && to_metal == from_metal {
        // The same stored bytes; only the shader conversion differs (for example R8Unorm and R8Unorm_sRGB).
        return Ok(ViewPlan::SharedStorage);
    }
    if srgb_siblings(ts, vs) && to_metal == Some(Transform::None) && from_metal == Some(Transform::None) {
        return Ok(ViewPlan::ViewFormat(vs));
    }
    if ts.caps().footprint != vs.caps().footprint || ts.caps().footprint == 0 {
        return Err(Unavailable::Refused(Refusal {
            id: "view-size-mismatch",
            reason: "The texture and the view store texels of different sizes.",
            source: "tables/formats.toml, Texel block copy footprint of the stored formats",
        }));
    }
    if !proven(ts) || !proven(vs) {
        return Err(Unavailable::Refused(FLOAT_VIEW));
    }
    let (Some(to_metal), Some(from_metal)) = (to_metal, from_metal) else {
        unreachable!("uncompressed resolutions store bytes")
    };
    Ok(ViewPlan::Copy { storage: vs, to_metal, from_metal })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::webgpu::Feature;

    /// The plan on a host whose texel copies keep the bits of the bit-exact formats only.
    fn view_plan(texture: u32, view: u32, features: HostFeatures) -> Result<ViewPlan, Unavailable> {
        super::view_plan(texture, view, features, &|f| f.texel_copy_exact(None))
    }

    #[test]
    fn plans() {
        let none = HostFeatures::NONE;
        let all = HostFeatures::new(Feature::ALL);
        // RGBA8Unorm and RGBA8Unorm_sRGB: a WebGPU view format.
        assert_eq!(view_plan(70, 71, none), Ok(ViewPlan::ViewFormat(WebGpuFormat::Rgba8unormSrgb)));
        // RGBA8Unorm as RGBA8Uint: both bit-exact under texel copies.
        assert_eq!(
            view_plan(70, 73, none),
            Ok(ViewPlan::Copy {
                storage: WebGpuFormat::Rgba8uint,
                to_metal: Transform::None,
                from_metal: Transform::None
            })
        );
        // R32Float as R32Uint: the float side is not bit-exact, unless the host's copies of r32float are proven.
        assert!(matches!(
            view_plan(55, 53, none),
            Err(Unavailable::Refused(Refusal { id: "float-view-reinterpretation", .. }))
        ));
        assert_eq!(
            super::view_plan(55, 53, none, &|_| true),
            Ok(ViewPlan::Copy {
                storage: WebGpuFormat::R32uint,
                to_metal: Transform::None,
                from_metal: Transform::None
            })
        );
        // R8Unorm as R8Unorm_sRGB: the same r8unorm storage.
        assert_eq!(view_plan(10, 11, none), Ok(ViewPlan::SharedStorage));
        // BGR10A2Unorm (stored swapped) as RGB10A2Uint.
        assert_eq!(
            view_plan(94, 91, none),
            Ok(ViewPlan::Copy {
                storage: WebGpuFormat::Rgb10a2uint,
                to_metal: Transform::SwapRedBlue10,
                from_metal: Transform::None
            })
        );
        // B5G6R5Unorm as R16Uint: r16uint both ways, the packed bytes unchanged.
        assert_eq!(view_plan(40, 23, none), Ok(ViewPlan::SharedStorage));
        // BGR10A2Unorm as RGB10A2Unorm: the same WebGPU format, but the texture holds the swapped texels.
        assert_eq!(
            view_plan(94, 90, none),
            Ok(ViewPlan::Copy {
                storage: WebGpuFormat::Rgb10a2unorm,
                to_metal: Transform::SwapRedBlue10,
                from_metal: Transform::None
            })
        );
        // ETC2_RGB8 and its sRGB sibling decode to the same codes; ASTC sRGB and LDR do not.
        assert_eq!(view_plan(180, 181, none), Ok(ViewPlan::ViewFormat(WebGpuFormat::Rgba8unormSrgb)));
        assert!(matches!(view_plan(186, 204, none), Ok(ViewPlan::FromBlocks(Some(Scheme::Astc { srgb: false, .. })))));
        assert_eq!(view_plan(186, 204, all), Ok(ViewPlan::ViewFormat(WebGpuFormat::Astc4x4Unorm)));
        // X32_Stencil8 of Depth32Float_Stencil8.
        assert_eq!(view_plan(260, 261, all), Ok(ViewPlan::StencilAspect));
    }
}
