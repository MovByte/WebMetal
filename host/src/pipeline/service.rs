//! The pipeline service as the executor's [`ControlService`]: the shader opcodes of the wire protocol.

use std::marker::PhantomData;
use std::rc::Rc;

use webmetal_wire::{
    BinaryArchiveAddComputePipelineFunctionsReply, BinaryArchiveAddRenderPipelineFunctionsReply,
    ComputePipelineDescriptor, CreateBinaryArchiveReply, CreateComputePipelineAsyncReply, CreateComputePipelineReply,
    CreateRenderPipelineAsyncReply, CreateRenderPipelineReflectionAsyncReply, CreateRenderPipelineReflectionReply,
    CreateRenderPipelineReply, Failure, Handle, RegisterAirReply, RegisterFunctionReply, RenderPipelineDescriptor,
    Reply, Request, opcode,
};

use super::archive::{self, Archive};
use super::compute::{self, option};
use super::registry::AirRegistry;
use super::render;
use super::{Shared, errors, gpu, reflection};
use crate::embed::{Embedder, TaskId};
use crate::exec::{ControlService, Entry, Served, ServiceCx, failure};
use crate::gpu::{Api, GpuFuture};
use crate::render::pipeline::RenderPipeline;

/// The control service of the pipeline module.
pub struct PipelineService<A: Api, E: Embedder> {
    shared: Rc<Shared<A>>,
    registry: AirRegistry,
    _embedder: PhantomData<E>,
}

impl<A: Api, E: Embedder> PipelineService<A, E> {
    pub fn new(shared: Rc<Shared<A>>) -> PipelineService<A, E> {
        PipelineService { shared, registry: AirRegistry::default(), _embedder: PhantomData }
    }
}

/// The Metal methods of the compute creation records, for messages.
const NEW_COMPUTE: &str = "-[MTLDevice newComputePipelineStateWithDescriptor:options:reflection:error:]";
const NEW_COMPUTE_ASYNC: &str = "-[MTLDevice newComputePipelineStateWithDescriptor:options:completionHandler:]";
const ADD_COMPUTE: &str = "-[MTLBinaryArchive addComputePipelineFunctionsWithDescriptor:error:]";
const NEW_ARCHIVE: &str = "-[MTLDevice newBinaryArchiveWithDescriptor:error:]";
const NEW_RENDER: &str = "-[MTLDevice newRenderPipelineStateWithDescriptor:error:]";
const NEW_RENDER_REFLECTION: &str = "-[MTLDevice newRenderPipelineStateWithDescriptor:options:reflection:error:]";
const NEW_RENDER_ASYNC: &str = "-[MTLDevice newRenderPipelineStateWithDescriptor:completionHandler:]";
const NEW_RENDER_REFLECTION_ASYNC: &str =
    "-[MTLDevice newRenderPipelineStateWithDescriptor:options:completionHandler:]";
const ADD_RENDER: &str = "-[MTLBinaryArchive addRenderPipelineFunctionsWithDescriptor:error:]";

/// A compute pipeline checked and compiled for its creation variant.
struct ComputeCreation<A: Api> {
    function: Rc<super::registry::Function>,
    settings: compute::ComputeSettings,
    info: compute::KernelInfo,
    outcome: super::compile::Outcome,
    record: Vec<u8>,
    _api: PhantomData<A>,
}

impl<A: Api, E: Embedder + 'static> PipelineService<A, E> {
    /// Check a compute descriptor and compile its creation variant; the failure is the creation's.
    fn compute_creation(
        &self,
        cx: &ServiceCx<'_, A, E>,
        call: &str,
        d: &ComputePipelineDescriptor,
    ) -> Result<ComputeCreation<A>, Failure> {
        let settings = compute::check_descriptor(call, d)?;
        let handle = d.compute_function.ok_or_else(|| {
            failure::unimplemented(format!("{call}: computeFunction is nil; Metal's result is not established"))
        })?;
        let function = compute::function(cx.objects(), handle)?;
        compute::check_kernel(call, &function)?;
        let info = compute::kernel_info(&function).map_err(|r| errors::refusal(call, &r))?;
        let desc = compute::creation_desc(&settings, &info);
        let modules = [function.stage_module()];
        let outcome = self.shared.compiler.compile(&modules, &desc);
        if let Err(r) = &*outcome {
            return Err(errors::refusal(call, r));
        }
        let key = super::compile::key(&modules, &desc, self.shared.compiler.host());
        let record = archive::record(archive::COMPUTE, &key, |w| {
            w.u64(settings.max_total_threads_per_threadgroup);
            w.bool(settings.threadgroup_size_is_multiple_of_thread_execution_width);
            w.bool(settings.support_indirect_command_buffers);
        });
        Ok(ComputeCreation { function, settings, info, outcome, record, _api: PhantomData })
    }

    /// `CreateComputePipeline` and `CreateComputePipelineAsync`: the reflection to reply with once the creation
    /// variant's WebGPU pipeline exists.
    fn create_compute(
        &self,
        cx: &mut ServiceCx<'_, A, E>,
        call: &str,
        handle: Handle,
        d: &ComputePipelineDescriptor,
        options: u64,
        reflect: bool,
    ) -> Result<GpuFuture<Result<Option<webmetal_wire::Reflection>, Failure>>, Failure> {
        option::check(call, options)?;
        let c = self.compute_creation(cx, call, d)?;
        if options & option::FAIL_ON_BINARY_ARCHIVE_MISS != 0 {
            archive::lookup(call, cx.objects(), &d.binary_archives, &c.record)?;
        }
        let compiled = (*c.outcome).as_ref().expect("checked");
        gpu::requirements::<A>(call, cx.gpu().device(), &self.shared.wgsl_language_features, &compiled.requirements)?;
        let label = if c.settings.label.is_empty() {
            format!("compute pipeline {:#x}", handle.get())
        } else {
            c.settings.label.clone()
        };
        let created = self.shared.gpu.compute_async(cx.gpu(), &label, compiled)?;
        let mut settings = c.settings.clone();
        settings.label = label;
        let (state, _) = compute::payload(&self.shared, c.function.clone(), &settings, &c.info, &compiled.reflection);
        cx.attach(Rc::new(state));
        let reflection = reflect.then(|| reflection::to_wire(&compiled.reflection));
        Ok(Box::pin(async move {
            created.await?;
            Ok(reflection)
        }))
    }
}

/// A render pipeline checked and compiled.
struct RenderCreation {
    settings: render::RenderSettings,
    outcome: super::compile::Outcome,
    record: Vec<u8>,
}

impl<A: Api, E: Embedder + 'static> PipelineService<A, E> {
    /// Check a render descriptor and compile its pipeline; the failure is the creation's.
    fn render_creation(
        &self,
        cx: &ServiceCx<'_, A, E>,
        call: &str,
        d: &RenderPipelineDescriptor,
    ) -> Result<RenderCreation, Failure> {
        let mut settings = render::check_descriptor(call, d, self.shared.formats)?;
        let (vertex, fragment) = render::functions(cx.objects(), &settings)?;
        render::check_functions(call, &mut settings, d, &vertex, fragment.as_deref())?;
        if fragment.is_none() {
            render::mask_all(&mut settings);
        }
        let desc = render::desc(&settings);
        let mut modules = vec![vertex.stage_module()];
        modules.extend(fragment.as_ref().map(|f| f.stage_module()));
        let outcome = self.shared.compiler.compile(&modules, &desc);
        match &*outcome {
            Err(r) => return Err(errors::refusal(call, r)),
            Ok(c) => render::check_vertex_writes(call, c)?,
        }
        let key = super::compile::key(&modules, &desc, self.shared.compiler.host());
        let record = archive::record(archive::RENDER, &key, |w| {
            for c in &settings.colors {
                w.u32(c.pixel_format);
            }
            w.u32(settings.sample_mask);
            w.bool(settings.support_indirect_command_buffers);
        });
        Ok(RenderCreation { settings, outcome, record })
    }

    /// The `CreateRenderPipeline` records: the reflection to reply with. The render path creates its WebGPU pipelines
    /// synchronously, through its own variant cache.
    fn create_render(
        &self,
        cx: &mut ServiceCx<'_, A, E>,
        call: &str,
        handle: Handle,
        d: &RenderPipelineDescriptor,
        options: u64,
        reflect: bool,
    ) -> Result<Option<webmetal_wire::Reflection>, Failure> {
        option::check(call, options)?;
        let c = self.render_creation(cx, call, d)?;
        if options & option::FAIL_ON_BINARY_ARCHIVE_MISS != 0 {
            archive::lookup(call, cx.objects(), &d.binary_archives, &c.record)?;
        }
        let compiled = c.outcome.clone();
        let compiled_ref = (*compiled).as_ref().expect("checked");
        gpu::requirements::<A>(
            call,
            cx.gpu().device(),
            &self.shared.wgsl_language_features,
            &compiled_ref.requirements,
        )?;
        let label = if c.settings.label.is_empty() {
            format!("render pipeline {:#x}", handle.get())
        } else {
            c.settings.label.clone()
        };
        gpu::check_layout::<A>(call, cx.gpu().device(), &compiled_ref.layout)?;
        let desc = render::plan(call, &label, &c.settings, compiled_ref)?;
        let pipeline = RenderPipeline::create(cx.gpu(), desc).map_err(|e| e.failure())?;
        if pipeline.rasterization {
            pipeline.variant(cx.gpu(), &render::creation_key(&pipeline)).map_err(|e| e.failure())?;
        }
        cx.attach(Rc::new(pipeline));
        Ok(reflect.then(|| reflection::to_wire(&compiled_ref.reflection)))
    }
}

/// The archive a record names as its receiver.
fn archive_of<A: Api, E: Embedder>(cx: &ServiceCx<'_, A, E>, h: Handle, call: &str) -> Result<Rc<Archive>, Failure> {
    cx.objects()
        .get(h)
        .and_then(|e| e.payload.clone())
        .and_then(|p| p.downcast::<Archive>().ok())
        .ok_or_else(|| failure::invalid(format!("{call}: the receiver {:#x} is not a binary archive", h.get())))
}

fn ready(r: Reply) -> Served {
    Served::Ready(r)
}

/// A reply that waits for `f` and wraps its result.
fn later<T: 'static>(
    f: Result<GpuFuture<Result<T, Failure>>, Failure>,
    wrap: impl FnOnce(Result<T, Failure>) -> Reply + 'static,
) -> Served {
    match f {
        Err(e) => Served::Ready(wrap(Err(e))),
        Ok(f) => Served::Later(Box::pin(async move { wrap(f.await) })),
    }
}

impl<A: Api, E: Embedder + 'static> ControlService<A, E> for PipelineService<A, E> {
    fn opcodes(&self) -> Vec<u16> {
        vec![
            opcode::REGISTER_AIR,
            opcode::REGISTER_FUNCTION,
            opcode::CREATE_RENDER_PIPELINE,
            opcode::CREATE_RENDER_PIPELINE_REFLECTION,
            opcode::CREATE_RENDER_PIPELINE_ASYNC,
            opcode::CREATE_RENDER_PIPELINE_REFLECTION_ASYNC,
            opcode::CREATE_TILE_RENDER_PIPELINE,
            opcode::CREATE_TILE_RENDER_PIPELINE_ASYNC,
            opcode::CREATE_COMPUTE_PIPELINE,
            opcode::CREATE_COMPUTE_PIPELINE_ASYNC,
            opcode::CREATE_BINARY_ARCHIVE,
            opcode::BINARY_ARCHIVE_ADD_RENDER_PIPELINE_FUNCTIONS,
            opcode::BINARY_ARCHIVE_ADD_COMPUTE_PIPELINE_FUNCTIONS,
            opcode::BINARY_ARCHIVE_ADD_TILE_RENDER_PIPELINE_FUNCTIONS,
        ]
    }

    fn serve(&mut self, cx: &mut ServiceCx<'_, A, E>, request: &Request) -> Served {
        match request {
            Request::RegisterAir(r) => ready(Reply::RegisterAir(
                self.registry.register(cx.task(), r.hash.bytes, &r.bitcode).map(|()| RegisterAirReply {}),
            )),
            Request::RegisterFunction(r) => {
                let f = self.registry.function(cx.task(), &r.air.bytes, &r.name, r.function_type, &r.constants);
                ready(Reply::RegisterFunction(f.map(|f| {
                    cx.attach(Rc::new(f));
                    RegisterFunctionReply {}
                })))
            }
            Request::CreateRenderPipeline(r) => {
                let res = self.create_render(cx, NEW_RENDER, r.pipeline, &r.descriptor, 0, false);
                ready(Reply::CreateRenderPipeline(res.map(|_| CreateRenderPipelineReply {})))
            }
            Request::CreateRenderPipelineReflection(r) => {
                let res =
                    self.create_render(cx, NEW_RENDER_REFLECTION, r.pipeline, &r.descriptor, r.options, r.reflection);
                ready(Reply::CreateRenderPipelineReflection(
                    res.map(|reflection| CreateRenderPipelineReflectionReply { reflection }),
                ))
            }
            Request::CreateRenderPipelineAsync(r) => {
                // MTLNewRenderPipelineStateCompletionHandler receives no reflection.
                let res = self.create_render(cx, NEW_RENDER_ASYNC, r.pipeline, &r.descriptor, 0, false);
                ready(Reply::CreateRenderPipelineAsync(res.map(|_| CreateRenderPipelineAsyncReply {})))
            }
            Request::CreateRenderPipelineReflectionAsync(r) => {
                let res = self.create_render(
                    cx,
                    NEW_RENDER_REFLECTION_ASYNC,
                    r.pipeline,
                    &r.descriptor,
                    r.options,
                    r.completion_handler,
                );
                ready(Reply::CreateRenderPipelineReflectionAsync(
                    res.map(|reflection| CreateRenderPipelineReflectionAsyncReply { reflection }),
                ))
            }
            Request::CreateTileRenderPipeline(_)
            | Request::CreateTileRenderPipelineAsync(_)
            | Request::BinaryArchiveAddTileRenderPipelineFunctions(_) => {
                let op = request.opcode();
                ready(
                    crate::exec::schema::failure_reply(
                        op,
                        failure::unimplemented(format!(
                            "{}: tile functions are not a feature of the A10X's GPU family (Apple3; tile shading is Apple4 \
                             and later in the Metal Feature Set Tables) and the driver's result for a tile pipeline \
                             is not established",
                            failure::record_methods(op)
                        )),
                    )
                    .expect("a control opcode"),
                )
            }
            Request::BinaryArchiveAddRenderPipelineFunctions(r) => {
                let res = self.render_creation(cx, ADD_RENDER, &r.descriptor).and_then(|c| {
                    archive_of(cx, r.archive, ADD_RENDER)?.add(c.record);
                    Ok(BinaryArchiveAddRenderPipelineFunctionsReply {})
                });
                ready(Reply::BinaryArchiveAddRenderPipelineFunctions(res))
            }
            Request::CreateComputePipeline(r) => {
                let f = self.create_compute(cx, NEW_COMPUTE, r.pipeline, &r.descriptor, r.options, r.reflection);
                later(f, |res| {
                    Reply::CreateComputePipeline(res.map(|reflection| CreateComputePipelineReply { reflection }))
                })
            }
            Request::CreateComputePipelineAsync(r) => {
                // The completion handler receives the reflection (MTLNewComputePipelineStateWithReflectionCompletion-
                // Handler); without a handler the caller has nowhere to receive it.
                let f = self.create_compute(
                    cx,
                    NEW_COMPUTE_ASYNC,
                    r.pipeline,
                    &r.descriptor,
                    r.options,
                    r.completion_handler,
                );
                later(f, |res| {
                    Reply::CreateComputePipelineAsync(
                        res.map(|reflection| CreateComputePipelineAsyncReply { reflection }),
                    )
                })
            }
            Request::CreateBinaryArchive(r) => {
                ready(Reply::CreateBinaryArchive(archive::create(NEW_ARCHIVE, &r.descriptor).map(|a| {
                    cx.attach(Rc::new(a));
                    CreateBinaryArchiveReply {}
                })))
            }
            Request::BinaryArchiveAddComputePipelineFunctions(r) => {
                let res = self.compute_creation(cx, ADD_COMPUTE, &r.descriptor).and_then(|c| {
                    archive_of(cx, r.archive, ADD_COMPUTE)?.add(c.record);
                    Ok(BinaryArchiveAddComputePipelineFunctionsReply {})
                });
                ready(Reply::BinaryArchiveAddComputePipelineFunctions(res))
            }
            other => ready(
                crate::exec::schema::failure_reply(other.opcode(), failure::unimplemented_record(other.opcode()))
                    .expect("a control opcode"),
            ),
        }
    }

    fn release(&mut self, _task: TaskId, _handle: Handle, _entry: &Entry) {
        // Functions, pipelines and archives keep their state in their payloads, which the object table drops.
    }

    fn task_exited(&mut self, task: TaskId) {
        self.registry.task_exited(task);
    }
}
