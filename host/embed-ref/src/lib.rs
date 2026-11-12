//! The WebMetal reference embedder: runs the host executor without a kernel.
//!
//! The executor reaches the guest system only through the host interface (`webmetal_host::embed`). This crate
//! implements that interface over in-process state ([`embedder`]), sets up the executor with the services this
//! repository provides and a presentation sink whose displays are offscreen targets ([`host`]), and replays
//! recordings on it ([`recording`]). It stands where a kernel stands when the guest runs, so that the executor and the
//! presentation path can be tested against WebGPU implementations and compared with real Metal without a guest.
//!
//! The `webmetal-embed-ref` program (feature `native`) replays a recording file on Dawn or wgpu-native. The
//! `present_web` example (feature `web`, wasm32), run by `tools/present_web.py`, shows frames on a page's canvas and
//! reads back what the browser shows.

pub mod embedder;
pub mod host;
pub mod recording;

pub use embedder::{Clock, RefEmbedder};
pub use host::RefHost;
pub use recording::{Step, StepOutcome};
