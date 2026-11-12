//! Presentation: the frames a display shows (plan section 6.8).
//!
//! `CAMetalLayer` and the render server are Apple's code in the guest: a drawable is an IOSurface the layer allocates,
//! `presentDrawable:` and its timed variants reach QuartzCore, and the render server composites into the display's
//! framebuffer and swaps it through `IOMobileFramebuffer`. The wire protocol therefore has no presentation records
//! (the `MTLDrawable` selectors are guest-local). The kernel's framebuffer service resolves each swap to a frame and
//! hands it to the [`Presenter`], which shows it on the display's target: a canvas in a page, an [`Offscreen`] target
//! elsewhere. The embedder learns that a frame was shown through [`crate::embed::Embedder::present`] and from the
//! result of [`Presenter::present`], and drives swap completion from it.
//!
//! - [`format`]: the framebuffer pixel formats shown, and how a pixel becomes a canvas texel.
//! - [`canvas`]: the display model and the canvas configuration a display needs.
//! - [`target`]: presentation targets, and [`Offscreen`].
//! - [`sink`]: the [`Presenter`].
//! - `web` (feature `web`, wasm32): the target over a page's `GPUCanvasContext`.
//!
//! For the formats shown, the canvas texture holds the surface's pixel codes exactly. Every frame that cannot be shown
//! exactly is refused with a named reason, never approximated.

pub mod canvas;
pub mod format;
pub mod sink;
pub mod target;
#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub mod web;

pub use canvas::{CanvasPlan, DisplayColorSpace, DisplayMode};
pub use sink::{PresentFailure, Presented, Presenter, SwapRequest};
pub use target::{Offscreen, Snapshot, Target};
