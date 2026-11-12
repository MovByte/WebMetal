//! The display model the embedder states, and the canvas configuration a display needs for a frame (plan section 6.8).
//!
//! A display shows its framebuffer at the panel's mode, in the panel's color space, with nothing behind it. The canvas
//! that stands for the display is configured so that its drawing buffer holds the framebuffer's pixel codes unchanged
//! and is shown with the same meaning:
//!
//! - **size**: the panel mode (`canvas.width` × `canvas.height`). A frame covers the whole display at the panel's
//!   resolution; scaling and partial composition are refused by the sink.
//! - **format**: the canvas format of the framebuffer's pixel format ([`super::format`]).
//! - **colorSpace**: the display's color space, which must be one of HTML's `PredefinedColorSpace` values ("srgb",
//!   "srgb-linear", "display-p3", "display-p3-linear"); the canvas tags its drawing buffer with it and the browser
//!   converts from it to the screen's (WebGPU §21.4.1). A display whose color space is none of these is refused.
//! - **toneMapping**: "standard". Every 8-bit unorm code is in [0, 1], which "standard" leaves unchanged (HTML
//!   §4.12.5.4, "standard": the HDR headroom of the bitmap is 0 and tone mapping does not change its color values).
//! - **alphaMode**: "opaque". A display's framebuffer has nothing behind it to blend with; "opaque" shows the canvas
//!   the same way: the alpha channel is cleared to 1.0 in "get a copy of the image contents of a context", while the
//!   drawing buffer keeps the frame's alpha codes (WebGPU §21.2, §21.6). How a display with several hardware layers
//!   blends them is the display model's (K3); a frame here is one layer that covers the display.
//! - **usage**: `COPY_DST`, for `GPUQueue.writeTexture()` of the frame; **viewFormats**: none.
//!
//! The supported context formats are valid with every device (WebGPU §21.4), so a plan never needs a feature.

use crate::format::handling::Refusal;
use crate::gpu::idl::{
    CanvasAlphaMode, CanvasConfiguration, CanvasToneMapping, CanvasToneMappingMode, TextureFormat, TextureUsage,
};
use crate::gpu::{Api, PredefinedColorSpace};

use super::format::FramebufferFormat;

/// The color space a display shows its framebuffer's codes in, as the embedder's display model reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DisplayColorSpace {
    /// CSS 'srgb'.
    Srgb,
    /// CSS 'srgb-linear'.
    SrgbLinear,
    /// CSS 'display-p3'.
    DisplayP3,
    /// CSS 'display-p3-linear'.
    DisplayP3Linear,
    /// Any other color space, by the display model's description.
    Other(String),
}

impl DisplayColorSpace {
    /// The `PredefinedColorSpace` value that denotes it (HTML §4.12.5.4), if one does.
    pub fn predefined(&self) -> Option<&'static str> {
        Some(match self {
            DisplayColorSpace::Srgb => "srgb",
            DisplayColorSpace::SrgbLinear => "srgb-linear",
            DisplayColorSpace::DisplayP3 => "display-p3",
            DisplayColorSpace::DisplayP3Linear => "display-p3-linear",
            DisplayColorSpace::Other(_) => return None,
        })
    }
}

/// A display as the embedder's display model describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplayMode {
    /// The panel mode, in pixels.
    pub width: u32,
    pub height: u32,
    pub color_space: DisplayColorSpace,
    /// The value of [`crate::embed::PresentedFrame::transform`] that means "no transform" in the embedder's encoding.
    pub identity_transform: u32,
}

/// A display whose color space no `PredefinedColorSpace` value denotes.
pub const COLOR_SPACE: Refusal = Refusal {
    id: "present-color-space",
    reason: "A canvas tags its drawing buffer with a PredefinedColorSpace value; none denotes the display's color \
             space, so the browser would show the frame's codes with another meaning.",
    source: "WebGPU §21.4 (GPUCanvasConfiguration.colorSpace); HTML §4.12.5.4 (PredefinedColorSpace)",
};

/// A display larger than the device's textures can be.
pub const DISPLAY_SIZE: Refusal = Refusal {
    id: "present-display-size",
    reason: "The canvas texture of the display would exceed the device's maxTextureDimension2D, which fails \
             getCurrentTexture() (WebGPU §21.4.2).",
    source: "WebGPU §21.4.2 (Canvas Context sizing)",
};

/// The canvas configuration a display needs for frames of one framebuffer format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanvasPlan {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub usage: u32,
    pub view_formats: Vec<TextureFormat>,
    pub color_space: &'static str,
    pub tone_mapping: CanvasToneMappingMode,
    pub alpha_mode: CanvasAlphaMode,
}

impl CanvasPlan {
    /// The `GPUCanvasConfiguration` of the plan for `device`.
    pub fn configuration<'a, A: Api>(&self, device: &'a A::Device) -> CanvasConfiguration<'a, A> {
        let mut c = CanvasConfiguration::new(device, self.format);
        c.usage = self.usage;
        c.view_formats = self.view_formats.clone();
        c.color_space = PredefinedColorSpace::new(self.color_space);
        c.tone_mapping = CanvasToneMapping { mode: self.tone_mapping };
        c.alpha_mode = self.alpha_mode;
        c
    }
}

/// Why a display cannot be planned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// The display has no pixels.
    Empty,
    Refused(Refusal),
}

/// Check a display against the device: its size within `max_texture_dimension_2d` (the device's limit).
pub fn check_display(mode: &DisplayMode, max_texture_dimension_2d: u32) -> Result<(), PlanError> {
    if mode.width == 0 || mode.height == 0 {
        return Err(PlanError::Empty);
    }
    if mode.width > max_texture_dimension_2d || mode.height > max_texture_dimension_2d {
        return Err(PlanError::Refused(DISPLAY_SIZE));
    }
    Ok(())
}

/// The canvas configuration of `mode` for frames in `format`.
pub fn plan(mode: &DisplayMode, format: &FramebufferFormat) -> Result<CanvasPlan, PlanError> {
    if mode.width == 0 || mode.height == 0 {
        return Err(PlanError::Empty);
    }
    let color_space = mode.color_space.predefined().ok_or(PlanError::Refused(COLOR_SPACE))?;
    Ok(CanvasPlan {
        width: mode.width,
        height: mode.height,
        format: format.canvas,
        usage: TextureUsage::COPY_DST,
        view_formats: Vec::new(),
        color_space,
        tone_mapping: CanvasToneMappingMode::Standard,
        alpha_mode: CanvasAlphaMode::Opaque,
    })
}

/// WebGPU's supported context formats (WebGPU §21.4).
pub const SUPPORTED_CONTEXT_FORMATS: [TextureFormat; 3] =
    [TextureFormat::Bgra8unorm, TextureFormat::Rgba8unorm, TextureFormat::Rgba16float];

#[cfg(test)]
mod tests {
    use super::super::format::{SHOWN, code, framebuffer_format};
    use super::*;

    fn mode(color_space: DisplayColorSpace) -> DisplayMode {
        DisplayMode { width: 2224, height: 1668, color_space, identity_transform: 0 }
    }

    #[test]
    fn every_shown_format_has_a_supported_context_format() {
        for f in SHOWN {
            assert!(SUPPORTED_CONTEXT_FORMATS.contains(&f.canvas), "{}", f.name);
        }
    }

    #[test]
    fn the_plan_of_an_srgb_display() {
        let f = framebuffer_format(code::BGRA_32).unwrap();
        let p = plan(&mode(DisplayColorSpace::Srgb), &f).unwrap();
        assert_eq!(
            p,
            CanvasPlan {
                width: 2224,
                height: 1668,
                format: TextureFormat::Bgra8unorm,
                usage: TextureUsage::COPY_DST,
                view_formats: vec![],
                color_space: "srgb",
                tone_mapping: CanvasToneMappingMode::Standard,
                alpha_mode: CanvasAlphaMode::Opaque,
            }
        );
    }

    #[test]
    fn each_predefined_color_space_is_the_canvas_color_space() {
        let f = framebuffer_format(code::ARGB_32).unwrap();
        for (c, name) in [
            (DisplayColorSpace::Srgb, "srgb"),
            (DisplayColorSpace::SrgbLinear, "srgb-linear"),
            (DisplayColorSpace::DisplayP3, "display-p3"),
            (DisplayColorSpace::DisplayP3Linear, "display-p3-linear"),
        ] {
            assert_eq!(plan(&mode(c), &f).unwrap().color_space, name);
        }
    }

    #[test]
    fn other_color_spaces_are_refused() {
        let f = framebuffer_format(code::BGRA_32).unwrap();
        for other in ["ITU-R BT.2020 PQ", "Adobe RGB (1998)", ""] {
            let e = plan(&mode(DisplayColorSpace::Other(other.into())), &f).unwrap_err();
            assert_eq!(e, PlanError::Refused(COLOR_SPACE));
        }
    }

    #[test]
    fn display_size_against_the_device() {
        let m = mode(DisplayColorSpace::Srgb);
        assert_eq!(check_display(&m, 8192), Ok(()));
        assert_eq!(check_display(&m, 2224), Ok(()));
        assert_eq!(check_display(&m, 2223), Err(PlanError::Refused(DISPLAY_SIZE)));
        let empty = DisplayMode { width: 0, ..m };
        assert_eq!(check_display(&empty, 8192), Err(PlanError::Empty));
    }
}
