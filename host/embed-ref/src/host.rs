//! [`RefHost`]: the executor with the services this repository provides, its presentation sink and offscreen displays,
//! over a [`RefEmbedder`].

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use webmetal_host::blit::BlitService;
use webmetal_host::compute::ComputeService;
use webmetal_host::exec::{Config, ControlService, Executor, Services};
use webmetal_host::format::webgpu::{Feature, HostFeatures};
use webmetal_host::gpu::{Api, Device as _, ExceptionKind, Gpu as _, GpuFuture};
use webmetal_host::mem::Memory;
use webmetal_host::mem::service::MemoryServices;
use webmetal_host::pipeline::{self, PipelineServices};
use webmetal_host::present::{DisplayMode, Offscreen, PresentFailure, Presenter};
use webmetal_host::render::RenderServices;
use webmetal_wire::{DecodeError, Reply, Request};

use crate::embedder::RefEmbedder;

/// The executor of one device with the memory, pipeline, render, compute and blit services, the presentation sink, and
/// an offscreen target per display.
pub struct RefHost<A: Api> {
    pub embedder: Rc<RefEmbedder>,
    pub exec: Executor<A, RefEmbedder>,
    pub presenter: Presenter<A, RefEmbedder>,
    pub memory: Rc<Memory<A>>,
    pub device: A::Device,
    targets: RefCell<HashMap<u32, Offscreen<A>>>,
}

/// The WebGPU features of `device`, as the format module names them.
pub fn host_features<A: Api>(device: &A::Device) -> HostFeatures {
    let names = device.features();
    HostFeatures::new(Feature::ALL.into_iter().filter(|f| names.contains(f.name())))
}

/// `wgslLanguageFeatures` of `gpu`. An implementation that does not provide the attribute (wgpu-native) is taken to
/// offer no extension, as the pipeline service requires: a pipeline that needs one is refused for the host, never
/// assumed to work (`webmetal_host::pipeline::PipelineServices::new`).
pub fn wgsl_language_features<A: Api>(gpu: &A::Gpu) -> Result<BTreeSet<String>, String> {
    match gpu.wgsl_language_features() {
        Ok(f) => Ok(f),
        Err(e) if e.kind == ExceptionKind::Unsupported => Ok(BTreeSet::new()),
        Err(e) => Err(format!("wgslLanguageFeatures: {e}")),
    }
}

impl<A: Api> RefHost<A> {
    /// Set up the services on `device` (which no one else uses) and the executor. `wgsl_language_features` are those of
    /// the `GPU` the device came from ([`wgsl_language_features`]).
    pub fn new(
        device: A::Device,
        wgsl_language_features: BTreeSet<String>,
        embedder: Rc<RefEmbedder>,
        config: Config,
    ) -> GpuFuture<Result<RefHost<A>, String>> {
        let services = MemoryServices::new(&device, host_features::<A>(&device), embedder.clone());
        Box::pin(async move {
            let svc = services.await.map_err(|e| format!("memory services: {e}"))?;
            let memory = svc.memory();
            let pipelines = PipelineServices::<A>::new(&device, wgsl_language_features, pipeline::Config::default());
            let render = RenderServices::new(memory.clone());
            let services = Services {
                control: vec![
                    svc.control() as Box<dyn ControlService<A, RefEmbedder>>,
                    pipelines.control(),
                    render.control(),
                ],
                render: Some(render.encoder()),
                compute: Some(Box::new(ComputeService::new(memory.clone()))),
                blit: Some(Box::new(BlitService::new(memory.clone()))),
                coherence: Some(svc.coherence()),
            };
            let exec = Executor::new(device.clone(), embedder.clone(), services, config);
            let presenter = Presenter::new(exec.clone(), device.clone(), embedder.clone());
            Ok(RefHost { embedder, exec, presenter, memory, device, targets: RefCell::new(HashMap::new()) })
        })
    }

    /// Add a display to the display model, shown on a new offscreen target.
    pub fn attach_offscreen(&self, display: u32, mode: DisplayMode) -> Result<Offscreen<A>, PresentFailure> {
        let target = Offscreen::new(mode.width, mode.height);
        self.presenter.attach(display, mode, Box::new(target.clone()))?;
        self.embedder.add_display(display);
        self.targets.borrow_mut().insert(display, target.clone());
        Ok(target)
    }

    /// The offscreen target of a display.
    pub fn target(&self, display: u32) -> Option<Offscreen<A>> {
        self.targets.borrow().get(&display).cloned()
    }

    /// Run a control request of `task`, given as the bytes the guest sent; resolves to the reply record.
    pub fn control_bytes(
        &self,
        task: webmetal_host::embed::TaskId,
        request: &[u8],
    ) -> GpuFuture<Result<Vec<u8>, DecodeError>> {
        self.exec.control(task, request)
    }

    /// Run a control request of `task`.
    pub fn control(&self, task: webmetal_host::embed::TaskId, request: &Request) -> GpuFuture<Result<Reply, String>> {
        let mut bytes = Vec::new();
        if let Err(e) = request.encode(&mut bytes) {
            return Box::pin(async move { Err(format!("encoding the request: {e:?}")) });
        }
        let reply = self.exec.control(task, &bytes);
        Box::pin(async move {
            let out = reply.await.map_err(|e| format!("the executor refused the request: {e:?}"))?;
            Reply::decode(&out, 0).map(|(r, _)| r).map_err(|e| format!("decoding the reply: {e:?}"))
        })
    }
}

/// A native WebGPU device (feature `native`): Dawn or wgpu-native, as `host/build.rs` selects.
#[cfg(feature = "native")]
pub mod native {
    use std::time::Duration;

    use webmetal_host::gpu::idl::{DeviceDescriptor, FeatureName, RequestAdapterOptions};
    use webmetal_host::gpu::native::{NativeDevice, NativeGpu};
    use webmetal_host::gpu::{Adapter as _, Gpu as _, negotiate};

    /// The instance and a device with the features and limits the negotiation asks for. An adapter without
    /// "core-features-and-limits" (wgpu-native reports none) gets every feature it offers with the default limits.
    pub fn device(label: &str) -> Result<(NativeGpu, NativeDevice), String> {
        let gpu = NativeGpu::new();
        let device = gpu
            .block_on(
                async {
                    let adapter = gpu
                        .request_adapter(&RequestAdapterOptions::new())
                        .await
                        .map_err(|e| format!("requestAdapter: {e}"))?
                        .ok_or("no adapter")?;
                    let mut d = match negotiate::plan(&adapter.features(), &adapter.limits()) {
                        Ok(p) => p.descriptor,
                        Err(negotiate::NegotiationError::NotCore) => {
                            let offered = adapter.features();
                            let mut d = DeviceDescriptor::new();
                            d.required_features =
                                FeatureName::ALL.into_iter().filter(|f| offered.contains(f.as_str())).collect();
                            d
                        }
                        Err(e) => return Err(format!("negotiation: {e}")),
                    };
                    d.label = label.into();
                    adapter.request_device(&d).await.map_err(|e| format!("requestDevice: {e}"))
                },
                Duration::from_secs(60),
            )
            .map_err(|e| format!("the device request stalled: {e}"))??;
        Ok((gpu, device))
    }
}
