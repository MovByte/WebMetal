//! The pipeline service: shader modules, functions, pipelines and binary archives.
//!
//! The guest's Metal.framework turns MSL into AIR (its compiler service) and specializes functions; the guest sends
//! each AIR module once by its SHA-256 (`RegisterAir`), names the functions it uses (`RegisterFunction`), and sends
//! every pipeline descriptor whole. This module serves those records ([`PipelineService`], a
//! [`crate::exec::ControlService`]):
//!
//! - [`registry`]: the AIR registry by digest and the functions of each task;
//! - [`compile`]: the calls into the compiler (AirWgsl `compile`) and the cache of their results, keyed by everything
//!   that changes the generated code;
//! - [`gpu`]: the WebGPU objects of compiled pipelines and their cache;
//! - [`compute`]: compute pipeline creation and the variants the compute encoder asks for per dispatch;
//! - [`render`]: render pipeline creation: the descriptor's fields, Metal's checks, the WebGPU color targets;
//! - [`archive`]: binary archive records;
//! - [`reflection`]: the compiler's reflection as the wire protocol's, for the replies that carry it;
//! - [`errors`]: how creation fails, as Metal defines it and as far as the A10X's errors are established.
//!
//! **Payloads.** The objects these records create are bound with these payloads:
//!
//! | Handle kind | Payload |
//! | --- | --- |
//! | function (`RegisterFunction`) | [`registry::Function`] |
//! | compute pipeline (`CreateComputePipeline`, `CreateComputePipelineAsync`) | [`crate::compute::pipeline::ComputePipelineState`] |
//! | render pipeline (`CreateRenderPipeline` and its reflection and asynchronous forms) | [`crate::render::pipeline::RenderPipeline`] |
//! | binary archive (`CreateBinaryArchive`) | [`archive::Archive`] |
//!
//! Tile pipelines are refused: tile functions are not a feature of the A10X's family.
//!
//! **The compiler.** The `airwgsl` crate is a git dependency of this crate pinned to a commit of the AirWgsl
//! repository (`host/Cargo.toml`); it has no dependencies of its own and builds for the native targets and for
//! wasm32 alike. A local AirWgsl checkout is used instead by patching the source in an untracked
//! `.cargo/config.toml` (`[patch."https://github.com/MovByte/AirWgsl"] airwgsl = { path = "..." }`), which also
//! rewrites `Cargo.lock` for that build.
//!
//! **Asynchronous creation.** The asynchronous Metal methods are asynchronous in the guest, which runs the completion
//! handler on Metal's queue when the reply arrives. On the host every creation, synchronous or not, compiles in the
//! request. A compute pipeline's WebGPU pipeline is created with `createComputePipelineAsync`, so the executor keeps
//! serving other requests and command buffers while WebGPU builds it, and the reply waits for the promise. A render
//! pipeline's WebGPU pipelines belong to the render path's variant cache, which creates them with
//! `createRenderPipeline`; the creation makes the variant of a new encoder's state through it.

pub mod archive;
pub mod compile;
pub mod compute;
pub mod errors;
pub mod gpu;
pub mod reflection;
pub mod registry;
pub mod render;
mod service;
pub mod sha256;

use std::collections::BTreeSet;
use std::rc::Rc;

pub use service::PipelineService;

use crate::embed::Embedder;
use crate::exec::ControlService;
use crate::format::webgpu::HostFeatures;
use crate::gpu::{Api, Device as _};

/// How the pipeline service runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// The bytes of WGSL (and refusal text) the compiler cache keeps before it drops its oldest entries.
    pub compile_cache_bytes: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config { compile_cache_bytes: 64 << 20 }
    }
}

/// What the service and the pipelines it creates share: the compiler and its cache, the WebGPU object cache and the
/// host's capabilities.
pub struct Shared<A: Api> {
    pub compiler: compile::Compiler,
    pub gpu: Rc<gpu::GpuCache<A>>,
    /// `navigator.gpu.wgslLanguageFeatures` of the `GPU` the device came from.
    pub wgsl_language_features: BTreeSet<String>,
    /// The device's features as the format tables take them.
    pub formats: HostFeatures,
}

/// The pipeline services of one executor.
pub struct PipelineServices<A: Api> {
    shared: Rc<Shared<A>>,
}

impl<A: Api> PipelineServices<A> {
    /// The services for `device`, created from a `GPU` whose `wgslLanguageFeatures` are `wgsl_language_features`. An
    /// implementation that does not report them (wgpu-native) is described by the empty set: a pipeline that needs an
    /// extension is then refused for this host (status 3), never assumed to work.
    pub fn new(device: &A::Device, wgsl_language_features: BTreeSet<String>, config: Config) -> PipelineServices<A> {
        let host = compile::host_caps::<A>(device, wgsl_language_features.iter().cloned());
        let names = device.features();
        let formats =
            HostFeatures::new(crate::format::webgpu::Feature::ALL.into_iter().filter(|f| names.contains(f.name())));
        PipelineServices {
            shared: Rc::new(Shared {
                compiler: compile::Compiler::new(host, config.compile_cache_bytes),
                gpu: Rc::new(gpu::GpuCache::default()),
                wgsl_language_features,
                formats,
            }),
        }
    }

    pub fn shared(&self) -> Rc<Shared<A>> {
        self.shared.clone()
    }

    /// The control service.
    pub fn control<E: Embedder + 'static>(&self) -> Box<dyn ControlService<A, E>> {
        Box::new(PipelineService::new(self.shared.clone()))
    }
}
