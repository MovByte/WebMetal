//! The compute encoder: [`ComputeService`] records a `MTLComputeCommandEncoder`'s commands as one WebGPU compute pass.
//!
//! | Metal | Here |
//! | --- | --- |
//! | `dispatchThreadgroups:threadsPerThreadgroup:` | `dispatchWorkgroups` with the pipeline variant whose workgroup size is the threadgroup size ([`pipeline`]). |
//! | `dispatchThreads:threadsPerThreadgroup:` | The A10X has no non-uniform threadgroups (`supportsNonUniformThreadgroupSize` is false in the device profile; Metal Feature Set Tables: "Nonuniform threadgroup size" from Apple4). A grid that is a whole number of threadgroups is that many threadgroups; any other grid is refused. |
//! | `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:` | `dispatchWorkgroupsIndirect`: `MTLDispatchThreadgroupsIndirectArguments` is three `uint32_t` threadgroup counts, WebGPU's three workgroup counts (`tables/limits.toml` question `6.6-indirect-dispatch-layout`). |
//! | `setBuffer:offset:atIndex:`, `setBuffers:offsets:withRange:`, `setBufferOffset:atIndex:`, `setBytes:length:atIndex:` | Buffer bindings from the offset rounded down to WebGPU's binding alignment, the rest of the offset passed to the shader; bytes in a buffer of their own. |
//! | `setTexture:atIndex:`, `setTextures:withRange:` | Texture views as the variant's layout describes. |
//! | `setThreadgroupMemoryLength:atIndex:` | Part of the variant key: WGSL fixes workgroup memory sizes. |
//! | `setStageInRegion:` | Uniform values of the variant. |
//! | `memoryBarrierWithScope:`, `memoryBarrierWithResources:count:`, the concurrent dispatch type | Dispatches of a WebGPU pass run in order and each sees the writes of those before it, which is one schedule a concurrent encoder allows and what every barrier asks for. |
//! | `useResource:usage:`, `useResources:count:usage:` | Resources used through argument buffers: a write usage makes the memory module read the resource back. |
//! | `setSamplerState:atIndex:`, `setSamplerStates:withRange:` | The state's WebGPU sampler for the binding's type ([`crate::state::SamplerObject`]); a state with no WebGPU form is refused when a dispatch uses it. |
//! | Per-binding sampler LOD clamps, dynamic attribute strides, the indirect stage-in region, imageblocks, function tables, acceleration structures, indirect command buffers, counter sampling, the SPI substreams and GPU control flow | Refused (see each record). |
//!
//! The memory module's duties (its `service` module documentation) are kept: textures the encoder binds are brought up
//! to date before the pass ([`Memory::prepare_texture`], [`Memory::prepare_linear`]); every writable binding is noted
//! ([`Memory::note_buffer_write`], [`Memory::note_texture_write`]) and every read-only one declared
//! ([`Memory::note_buffer_read`]); textures over buffers that a dispatch writes are copied back after the pass
//! ([`Memory::sync_linear`]).
//!
//! A texture view ([`crate::mem::view`]) binds as a WebGPU view of its root texture: its format (the root's, the sRGB
//! sibling, or the stencil aspect), its levels and its slices; the pipeline variant takes the view's pixel format and
//! type. Its writes are noted on the root's subresources it covers, and the root is brought up to date before the pass.

pub mod pipeline;

use std::collections::BTreeMap;
use std::rc::Rc;

use webmetal_wire::{Command, Failure, Handle, Size};

use crate::blit::{Tex, buffer as buffer_payload, texture as texture_payload};
use crate::embed::Embedder;
use crate::exec::failure;
use crate::exec::{DebugGroups, EncodeCx, EncoderService, Gpu, RecordFailure, Segment, Table};
use crate::format::webgpu::{Aspect, Feature};
use crate::gpu::idl::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, BufferBinding, BufferBindingType, BufferUsage,
    ComputePassDescriptor, StorageTextureAccess, TextureAspect, TextureSampleType, TextureUsage, TextureViewDescriptor,
    TextureViewDimension,
};
use crate::gpu::{
    Api, BindingCommands as _, Buffer as _, CommandEncoder as _, ComputePassEncoder as _, DebugCommands as _,
    Device as _, Size64, Texture as _,
};
use crate::mem::buffer::Buffer;
use crate::mem::layout::TextureType;
use crate::mem::texture::{Prepare, Texture, idl_format, usage};
use crate::mem::view::{IDENTITY_SWIZZLE, TextureView, read_swizzle, swizzle_string};
use crate::mem::{MemError, Memory};
use crate::state::SamplerObject;

use pipeline::{
    ComputePipelineState, ComputeVariant, Resource, SamplerSlot, Source, TextureSlot, UniformValue, VariantKey,
};

/// Buffer argument table entries of a compute encoder on the A10X (`maxComputeBuffers` of the device profile).
pub const MAX_BUFFERS: u64 = 31;
/// Texture argument table entries (`maxComputeTextures`).
pub const MAX_TEXTURES: u64 = 31;
/// Sampler argument table entries (`maxComputeSamplers`).
pub const MAX_SAMPLERS: u64 = 16;
/// `maxThreadsPerThreadgroup` of the A10X.
pub const MAX_THREADS: [u64; 3] = [512, 512, 512];
/// `maxThreadgroupMemoryLength` of the A10X: threadgroup memory of one threadgroup, static and set lengths together.
pub const MAX_THREADGROUP_MEMORY: u64 = 16384;
/// Threadgroup memory argument table entries (`setThreadgroupMemoryLength:atIndex:` takes the indices of the
/// function's `[[threadgroup(n)]]` arguments, 0 to 30 like the other tables).
pub const MAX_THREADGROUP_MEMORY_INDEX: u64 = 31;
/// The largest `setBytes:length:atIndex:` (the debug layer rejects more: `validation.set-bytes-over-4k`; the method
/// documentation: "Use this method for single-use data smaller than 4 KB").
pub const MAX_BYTES: u64 = 4096;

/// `MTLDispatchType` (`MTLComputeCommandEncoder.h`): serial 0, concurrent 1.
const DISPATCH_TYPES: [u64; 2] = [0, 1];
/// `MTLBarrierScope` bits: buffers, textures, render targets.
const BARRIER_SCOPES: u64 = 0x7;
/// `MTLResourceUsage` bits: read, write, sample (`MTLCommandEncoder.h`).
const USAGE_WRITE: u64 = 0x2;
const USAGES: u64 = 0x7;

/// The compute encoder service.
pub struct ComputeService<A: Api> {
    memory: Rc<Memory<A>>,
}

impl<A: Api> ComputeService<A> {
    /// The service over the memory module's [`Memory`] (`MemoryServices::memory`).
    pub fn new(memory: Rc<Memory<A>>) -> ComputeService<A> {
        ComputeService { memory }
    }
}

/// What a buffer argument table entry holds.
enum BufferSlot<A: Api> {
    Buffer {
        buffer: Rc<Buffer<A>>,
        offset: u64,
    },
    /// `setBytes:length:atIndex:`: the bytes in a buffer of their own.
    Bytes {
        gpu: A::Buffer,
        length: u64,
    },
}

/// The encoder's state.
struct State<A: Api> {
    pipeline: Option<Rc<ComputePipelineState<A>>>,
    buffers: BTreeMap<u64, BufferSlot<A>>,
    textures: BTreeMap<u64, Rc<TexEntry<A>>>,
    samplers: BTreeMap<u64, Rc<SamplerObject<A>>>,
    threadgroup_memory: BTreeMap<u64, u64>,
    stage_in: Option<([u32; 3], [u32; 3])>,
    /// Textures over buffers a dispatch wrote, to copy back after the pass.
    linear_written: Vec<Rc<TexEntry<A>>>,
}

impl<A: Api> Default for State<A> {
    fn default() -> State<A> {
        State {
            pipeline: None,
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            samplers: BTreeMap::new(),
            threadgroup_memory: BTreeMap::new(),
            stage_in: None,
            linear_written: Vec::new(),
        }
    }
}

/// A bound texture.
struct TexEntry<A: Api> {
    handle: Handle,
    tex: Tex<A>,
}

fn payload<T: 'static>(objects: &Table, h: Handle) -> Option<Rc<T>> {
    objects.get(h)?.payload.clone()?.downcast::<T>().ok()
}

fn undefined(call: &'static str, rule: impl Into<String>) -> Failure {
    MemError::undefined(call, rule).failure()
}

fn size64(v: u64, op: &'static str) -> Result<Size64, Failure> {
    Size64::new(v).map_err(|e| MemError::gpu(op, e).failure())
}

/// The textures the encoder's records bind, in order, with the offset of the first record that binds each.
fn bound_textures(records: &[(usize, Command)], segment: &Segment) -> Vec<(Handle, u64)> {
    let mut out: Vec<(Handle, u64)> = Vec::new();
    let mut add = |h: Option<Handle>, at: usize| {
        if let Some(h) = h
            && !out.iter().any(|(x, _)| *x == h)
        {
            out.push((h, at as u64));
        }
    };
    for &i in &segment.records {
        let (at, record) = &records[i];
        match record {
            Command::ComputeSetTexture(r) => add(r.texture, *at),
            Command::ComputeSetTextures(r) => r.textures.iter().for_each(|t| add(*t, *at)),
            _ => {}
        }
    }
    out
}

impl<A: Api> ComputeService<A> {
    fn record<E: Embedder>(
        &self,
        cx: &EncodeCx<'_, A, E>,
        pass: &A::ComputePassEncoder,
        st: &mut State<A>,
        record: &Command,
        groups: &mut DebugGroups,
    ) -> Result<(), Failure> {
        let g = cx.gpu();
        let objects = cx.objects();
        let call = || failure::record_methods(record.opcode());
        let refused = |why: &str| Err(failure::unimplemented(format!("{}: {why}", call())));
        match record {
            Command::BeginCompute(_) | Command::EncoderSetLabel(_) | Command::EncoderEndEncoding(_) => Ok(()),
            Command::BeginComputeWithDispatchType(r) => {
                if !DISPATCH_TYPES.contains(&r.dispatch_type) {
                    return Err(undefined(
                        "-[MTLCommandBuffer computeCommandEncoderWithDispatchType:]",
                        format!("{} is not an MTLDispatchType", r.dispatch_type),
                    ));
                }
                Ok(())
            }
            Command::BeginComputeWithDescriptor(r) => {
                let d = &r.compute_pass_descriptor;
                if !DISPATCH_TYPES.contains(&d.dispatch_type) {
                    return Err(undefined(
                        "-[MTLCommandBuffer computeCommandEncoderWithDescriptor:]",
                        format!("{} is not an MTLDispatchType", d.dispatch_type),
                    ));
                }
                if d.sample_buffer_attachments.iter().any(|a| a.sample_buffer.is_some()) {
                    return refused(
                        "counter sample buffer attachments: the wire protocol has no request that creates a counter sample buffer",
                    );
                }
                if d.substream_count != 0 {
                    return refused("compute substreams: the A10X reports maximumComputeSubstreams 0 (device profile)");
                }
                Ok(())
            }
            Command::BeginComputeWithDispatchTypeSubstreams(_) => {
                refused("compute substreams: the A10X reports maximumComputeSubstreams 0 (device profile)")
            }
            Command::EncoderPushDebugGroup(r) => {
                let label = r.string.clone().unwrap_or_default();
                g.call("GPUComputePassEncoder.pushDebugGroup", "", |_| pass.push_debug_group(&label));
                groups.push();
                Ok(())
            }
            Command::EncoderPopDebugGroup(_) => {
                if groups.pop() {
                    g.call("GPUComputePassEncoder.popDebugGroup", "", |_| pass.pop_debug_group());
                }
                Ok(())
            }
            Command::EncoderInsertDebugSignpost(r) => {
                let label = r.string.clone().unwrap_or_default();
                g.call("GPUComputePassEncoder.insertDebugMarker", "", |_| pass.insert_debug_marker(&label));
                Ok(())
            }
            Command::ComputeSetComputePipelineState(r) => {
                st.pipeline = match r.state {
                    None => None,
                    Some(h) => Some(payload::<ComputePipelineState<A>>(objects, h).ok_or_else(|| {
                        failure::invalid(format!("compute pipeline state {:#x} has no pipeline", h.get()))
                    })?),
                };
                Ok(())
            }
            Command::ComputeSetBuffer(r) => self.set_buffer(objects, st, r.buffer, r.offset, r.index),
            Command::ComputeSetBuffers(r) => {
                for (i, (b, o)) in r.buffers.iter().zip(&r.offsets).enumerate() {
                    self.set_buffer(objects, st, *b, *o, r.range.location + i as u64)?;
                }
                Ok(())
            }
            Command::ComputeSetBufferOffset(r) => {
                const CALL: &str = "-[MTLComputeCommandEncoder setBufferOffset:atIndex:]";
                match st.buffers.get_mut(&r.index) {
                    Some(BufferSlot::Buffer { buffer, offset }) => {
                        if r.offset > buffer.length {
                            return Err(undefined(
                                CALL,
                                format!("offset {} past a {}-byte buffer", r.offset, buffer.length),
                            ));
                        }
                        *offset = r.offset;
                        Ok(())
                    }
                    _ => Err(undefined(CALL, format!("no buffer is bound at index {}", r.index))),
                }
            }
            Command::ComputeSetBytes(r) => {
                const CALL: &str = "-[MTLComputeCommandEncoder setBytes:length:atIndex:]";
                if r.index >= MAX_BUFFERS {
                    return Err(undefined(CALL, format!("index {} of {MAX_BUFFERS} buffer entries", r.index)));
                }
                if r.length > MAX_BYTES {
                    return Err(undefined(CALL, format!("{} bytes; the limit is {MAX_BYTES}", r.length)));
                }
                let gpu = crate::blit::kernels::buffer_with(
                    g,
                    "webmetal setBytes",
                    &r.bytes,
                    BufferUsage::UNIFORM | BufferUsage::STORAGE,
                )
                .map_err(|e| e.failure())?;
                st.buffers.insert(r.index, BufferSlot::Bytes { gpu, length: r.length });
                Ok(())
            }
            Command::ComputeSetTexture(r) => self.set_texture(objects, st, r.texture, r.index),
            Command::ComputeSetTextures(r) => {
                for (i, t) in r.textures.iter().enumerate() {
                    self.set_texture(objects, st, *t, r.range.location + i as u64)?;
                }
                Ok(())
            }
            Command::ComputeSetSamplerState(r) => set_sampler(objects, st, r.sampler, r.index),
            Command::ComputeSetSamplerStates(r) => {
                for (i, s) in r.samplers.iter().enumerate() {
                    set_sampler(objects, st, *s, r.range.location + i as u64)?;
                }
                Ok(())
            }
            Command::ComputeSetSamplerStateLodMinClampLodMaxClamp(_)
            | Command::ComputeSetSamplerStatesLodMinClampsLodMaxClamps(_) => {
                refused("per-binding LOD clamps: the encoder binds each sampler state's own WebGPU sampler")
            }
            Command::ComputeSetThreadgroupMemoryLength(r) => {
                const CALL: &str = "-[MTLComputeCommandEncoder setThreadgroupMemoryLength:atIndex:]";
                if r.index >= MAX_THREADGROUP_MEMORY_INDEX {
                    return Err(undefined(CALL, format!("index {}", r.index)));
                }
                // "The length must be a multiple of 16 bytes" (setThreadgroupMemoryLength(_:index:) documentation).
                if !r.length.is_multiple_of(16) {
                    return Err(undefined(CALL, format!("length {} is not a multiple of 16", r.length)));
                }
                st.threadgroup_memory.insert(r.index, r.length);
                Ok(())
            }
            Command::ComputeSetStageInRegion(r) => {
                const CALL: &str = "-[MTLComputeCommandEncoder setStageInRegion:]";
                let v = |x: u64| u32::try_from(x).map_err(|_| undefined(CALL, format!("region value {x}")));
                let o = &r.region.origin;
                let s = &r.region.size;
                st.stage_in = Some(([v(o.x)?, v(o.y)?, v(o.z)?], [v(s.width)?, v(s.height)?, v(s.depth)?]));
                Ok(())
            }
            Command::ComputeSetStageInRegionWithIndirectBuffer(_) => refused(
                "a stage-in region read from a buffer at execution time: the uniform values of the variant are \
                 written when the dispatch is encoded",
            ),
            Command::ComputeSetBufferAttributeStride(_)
            | Command::ComputeSetBufferOffsetAttributeStride(_)
            | Command::ComputeSetBuffersAttributeStrides(_)
            | Command::ComputeSetBytesAttributeStride(_) => {
                refused("dynamic attribute strides for stage-in fetch are not implemented")
            }
            Command::ComputeSetImageblockWidth(_) => refused("imageblocks are not implemented"),
            Command::ComputeDispatchThreadgroups(r) => {
                let groups = dims(
                    "-[MTLComputeCommandEncoder dispatchThreadgroups:threadsPerThreadgroup:]",
                    &r.threadgroups_per_grid,
                )?;
                self.dispatch(cx, pass, st, Grid::Groups(groups), &r.threads_per_threadgroup)
            }
            Command::ComputeDispatchThreads(r) => {
                const CALL: &str = "-[MTLComputeCommandEncoder dispatchThreads:threadsPerThreadgroup:]";
                let grid = dims(CALL, &r.threads_per_grid)?;
                let tg = dims(CALL, &r.threads_per_threadgroup)?;
                if tg.contains(&0) {
                    return Err(undefined(CALL, "a threadgroup size of 0"));
                }
                if grid.iter().zip(&tg).any(|(g, t)| g % t != 0) {
                    return Err(undefined(
                        CALL,
                        format!(
                            "a grid of {grid:?} threads is not a whole number of {tg:?} threadgroups, and the A10X has \
                             no non-uniform threadgroups (supportsNonUniformThreadgroupSize is false in the device \
                             profile; Metal Feature Set Tables: \"Nonuniform threadgroup size\" from Apple4)"
                        ),
                    ));
                }
                let groups = [grid[0] / tg[0], grid[1] / tg[1], grid[2] / tg[2]];
                self.dispatch(cx, pass, st, Grid::Groups(groups), &r.threads_per_threadgroup)
            }
            Command::ComputeDispatchThreadgroupsWithIndirectBuffer(r) => {
                const CALL: &str = "-[MTLComputeCommandEncoder dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:]";
                let b = buffer_payload::<A>(objects, r.indirect_buffer, CALL, "the indirect buffer")
                    .map_err(|e| e.failure())?;
                // "indirectBufferOffset must be a multiple of 4" (MTLComputeCommandEncoder.h).
                if !r.indirect_buffer_offset.is_multiple_of(4) {
                    return Err(undefined(
                        CALL,
                        format!("indirect buffer offset {} is not a multiple of 4", r.indirect_buffer_offset),
                    ));
                }
                if r.indirect_buffer_offset.checked_add(12).is_none_or(|e| e > b.length) {
                    return Err(undefined(CALL, "the 12-byte arguments extend past the buffer"));
                }
                let limit = u64::from(g.device().limits().max_compute_workgroups_per_dimension.unwrap_or(65535));
                if let Some(bytes) =
                    self.memory.gpu_bytes_known(&b, r.indirect_buffer_offset..r.indirect_buffer_offset + 12)
                {
                    let counts: Vec<u64> =
                        bytes.chunks(4).map(|c| u64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
                    if counts.iter().any(|&c| c > limit) {
                        return Err(failure::host_unsupported(format!(
                            "{CALL}: threadgroup counts {counts:?} exceed this host's maxComputeWorkgroupsPerDimension \
                             {limit}"
                        )));
                    }
                }
                self.memory.note_buffer_read(&b);
                self.dispatch(cx, pass, st, Grid::Indirect(b, r.indirect_buffer_offset), &r.threads_per_threadgroup)
            }
            Command::ComputeDispatchThreadsWithIndirectBuffer(_) => refused(
                "the grid is read on the GPU, so whether it is a whole number of threadgroups (the A10X has no \
                 non-uniform threadgroups) cannot be checked",
            ),
            Command::ComputeMemoryBarrierWithScope(r) => {
                if r.scope & !BARRIER_SCOPES != 0 {
                    return Err(undefined(
                        "-[MTLComputeCommandEncoder memoryBarrierWithScope:]",
                        format!("scope {:#x}", r.scope),
                    ));
                }
                Ok(())
            }
            Command::ComputeMemoryBarrierWithResources(_) => Ok(()),
            Command::ComputeUseResource(r) => self.use_resources(objects, std::slice::from_ref(&r.resource), r.usage),
            Command::ComputeUseResources(r) => self.use_resources(objects, &r.resources, r.usage),
            Command::ComputeUseHeap(_) | Command::ComputeUseHeaps(_) => Ok(()),
            Command::ComputeExecuteCommandsInBuffer(_) | Command::ComputeExecuteCommandsInBufferIndirect(_) => refused(
                "the wire protocol does not define the indirect command buffer's memory yet (CreateIndirectCommandBuffer)",
            ),
            Command::ComputeSampleCountersInBuffer(_) => refused(
                "the A10X does not support MTLCounterSamplingPointAtDispatchBoundary (device profile, \
                 supportsCounterSampling:), and the wire protocol has no request that creates a counter sample buffer",
            ),
            Command::ComputeSetAccelerationStructure(_)
            | Command::ComputeSetIntersectionFunctionTable(_)
            | Command::ComputeSetIntersectionFunctionTables(_)
            | Command::ComputeSetVisibleFunctionTable(_)
            | Command::ComputeSetVisibleFunctionTables(_) => {
                refused("function tables and acceleration structures are not implemented")
            }
            other => Err(failure::unimplemented_record(other.opcode())),
        }
    }

    fn set_buffer(
        &self,
        objects: &Table,
        st: &mut State<A>,
        buffer: Option<Handle>,
        offset: u64,
        index: u64,
    ) -> Result<(), Failure> {
        const CALL: &str = "-[MTLComputeCommandEncoder setBuffer:offset:atIndex:]";
        if index >= MAX_BUFFERS {
            return Err(undefined(CALL, format!("index {index} of {MAX_BUFFERS} buffer entries")));
        }
        match buffer {
            None => {
                st.buffers.remove(&index);
            }
            Some(h) => {
                let b = buffer_payload::<A>(objects, Some(h), CALL, "the buffer").map_err(|e| e.failure())?;
                if offset > b.length {
                    return Err(undefined(CALL, format!("offset {offset} past a {}-byte buffer", b.length)));
                }
                st.buffers.insert(index, BufferSlot::Buffer { buffer: b, offset });
            }
        }
        Ok(())
    }

    fn set_texture(
        &self,
        objects: &Table,
        st: &mut State<A>,
        texture: Option<Handle>,
        index: u64,
    ) -> Result<(), Failure> {
        const CALL: &str = "-[MTLComputeCommandEncoder setTexture:atIndex:]";
        if index >= MAX_TEXTURES {
            return Err(undefined(CALL, format!("index {index} of {MAX_TEXTURES} texture entries")));
        }
        match texture {
            None => {
                st.textures.remove(&index);
            }
            Some(h) => {
                let tex = texture_payload::<A>(objects, Some(h), CALL, "the texture").map_err(|e| e.failure())?;
                st.textures.insert(index, Rc::new(TexEntry { handle: h, tex }));
            }
        }
        Ok(())
    }

    /// `useResource:usage:` and `useResources:count:usage:`: resources a function reaches through argument buffers. A
    /// resource it may write is read back whole.
    fn use_resources(&self, objects: &Table, resources: &[Option<Handle>], usage: u64) -> Result<(), Failure> {
        if usage & !USAGES != 0 {
            return Err(undefined("-[MTLComputeCommandEncoder useResource:usage:]", format!("usage {usage:#x}")));
        }
        if usage & USAGE_WRITE == 0 {
            return Ok(());
        }
        for h in resources.iter().flatten() {
            if let Some(b) = payload::<Buffer<A>>(objects, *h) {
                self.memory.note_buffer_write(&b, None);
            } else if let Some(t) = payload::<Texture<A>>(objects, *h) {
                self.memory.note_texture_write(&t, 0..t.geometry.mip_levels, 0..t.geometry.slices());
            } else if let Some(v) = payload::<TextureView<A>>(objects, *h) {
                self.memory.note_texture_write(&v.root, v.root_levels(), v.root_slices());
            } else if payload::<crate::mem::linear::LinearTexture<A>>(objects, *h).is_some() {
                return Err(failure::unimplemented(
                    "-[MTLComputeCommandEncoder useResource:usage:] with a write usage of a texture over a buffer: \
                     writes through argument buffers are not copied back to the buffer",
                ));
            }
        }
        Ok(())
    }

    fn dispatch<E: Embedder>(
        &self,
        cx: &EncodeCx<'_, A, E>,
        pass: &A::ComputePassEncoder,
        st: &mut State<A>,
        grid: Grid<A>,
        threads: &Size,
    ) -> Result<(), Failure> {
        const CALL: &str = "-[MTLComputeCommandEncoder dispatch...]";
        let g = cx.gpu();
        let pipeline = st.pipeline.clone().ok_or_else(|| undefined(CALL, "no compute pipeline state is set"))?;
        let tg = dims(CALL, threads)?;
        if tg.contains(&0) {
            return Err(undefined(CALL, "a threadgroup size of 0"));
        }
        if tg.iter().zip(MAX_THREADS).any(|(t, m)| u64::from(*t) > m) {
            return Err(undefined(
                CALL,
                format!("threadsPerThreadgroup {tg:?} exceeds maxThreadsPerThreadgroup {MAX_THREADS:?}"),
            ));
        }
        let total = tg.iter().map(|&t| u64::from(t)).product::<u64>();
        if total > pipeline.max_total_threads_per_threadgroup {
            return Err(undefined(
                CALL,
                format!(
                    "{total} threads per threadgroup exceed the pipeline's maxTotalThreadsPerThreadgroup {}",
                    pipeline.max_total_threads_per_threadgroup
                ),
            ));
        }
        if pipeline.threadgroup_size_is_multiple_of_execution_width
            && !total.is_multiple_of(pipeline.thread_execution_width.max(1))
        {
            return Err(undefined(
                CALL,
                format!(
                    "{total} threads per threadgroup are not a multiple of threadExecutionWidth {} that the pipeline \
                     descriptor promised",
                    pipeline.thread_execution_width
                ),
            ));
        }
        let memory = st.threadgroup_memory.values().sum::<u64>() + pipeline.static_threadgroup_memory_length;
        if memory > MAX_THREADGROUP_MEMORY {
            return Err(undefined(
                CALL,
                format!(
                    "{memory} bytes of threadgroup memory exceed maxThreadgroupMemoryLength {MAX_THREADGROUP_MEMORY}"
                ),
            ));
        }
        let limits = g.device().limits();
        let host_sizes = [
            limits.max_compute_workgroup_size_x.unwrap_or(256),
            limits.max_compute_workgroup_size_y.unwrap_or(256),
            limits.max_compute_workgroup_size_z.unwrap_or(64),
        ];
        if tg.iter().zip(host_sizes).any(|(t, m)| *t > m)
            || total > u64::from(limits.max_compute_invocations_per_workgroup.unwrap_or(256))
        {
            return Err(failure::host_unsupported(format!(
                "a threadgroup of {tg:?} threads exceeds this host's maxComputeWorkgroupSize {host_sizes:?} or \
                 maxComputeInvocationsPerWorkgroup {}",
                limits.max_compute_invocations_per_workgroup.unwrap_or(256)
            )));
        }
        if memory > u64::from(limits.max_compute_workgroup_storage_size.unwrap_or(16384)) {
            return Err(failure::host_unsupported(format!(
                "{memory} bytes of threadgroup memory exceed this host's maxComputeWorkgroupStorageSize"
            )));
        }
        if let Grid::Groups(groups) = &grid {
            let limit = limits.max_compute_workgroups_per_dimension.unwrap_or(65535);
            if groups.iter().any(|&n| n > limit) {
                return Err(failure::host_unsupported(format!(
                    "{groups:?} threadgroups exceed this host's maxComputeWorkgroupsPerDimension {limit}"
                )));
            }
        }
        let key = self.key(st, tg)?;
        let variant = pipeline.variants.variant(g, &key)?;
        let bind_groups = self.bind(g, st, &variant)?;
        g.call("GPUComputePassEncoder.setPipeline", "", |_| pass.set_pipeline(&variant.pipeline));
        for (index, group) in &bind_groups {
            g.call("GPUComputePassEncoder.setBindGroup", "", |_| pass.set_bind_group(*index, Some(group), &[]));
        }
        match grid {
            Grid::Groups([x, y, z]) => {
                if x != 0 && y != 0 && z != 0 {
                    g.call("GPUComputePassEncoder.dispatchWorkgroups", "", |_| pass.dispatch_workgroups(x, y, z));
                }
            }
            Grid::Indirect(b, offset) => {
                let o = size64(offset, "dispatchWorkgroupsIndirect")?;
                g.call("GPUComputePassEncoder.dispatchWorkgroupsIndirect", "", |_| {
                    pass.dispatch_workgroups_indirect(b.gpu(), o)
                });
            }
        }
        Ok(())
    }

    fn key(&self, st: &State<A>, tg: [u32; 3]) -> Result<VariantKey, Failure> {
        let lengths = match st.threadgroup_memory.keys().next_back() {
            None => Vec::new(),
            Some(&last) => (0..=last).map(|i| st.threadgroup_memory.get(&i).copied().unwrap_or(0)).collect(),
        };
        let textures = st
            .textures
            .iter()
            .map(|(&index, e)| TextureSlot {
                index,
                texture_type: texture_type_value(e.tex.geometry().kind),
                pixel_format: u64::from(e.tex.format()),
                sample_count: u64::from(e.tex.geometry().sample_count),
                swizzle: e.tex.swizzle(),
            })
            .collect();
        Ok(VariantKey {
            threads_per_threadgroup: tg,
            threadgroup_memory_lengths: lengths,
            buffers: st.buffers.keys().copied().collect(),
            textures,
            samplers: st
                .samplers
                .iter()
                .map(|(&index, s)| {
                    let d = &s.descriptor;
                    SamplerSlot {
                        index,
                        min_filter: d.min_filter,
                        mag_filter: d.mag_filter,
                        mip_filter: d.mip_filter,
                        address_modes: [d.s_address_mode, d.t_address_mode, d.r_address_mode],
                        border_color: d.border_color,
                        normalized_coordinates: d.normalized_coordinates,
                        compare_function: d.compare_function,
                        max_anisotropy: d.max_anisotropy,
                        lod_min_clamp: d.lod_min_clamp.to_bits(),
                        lod_max_clamp: d.lod_max_clamp.to_bits(),
                        lod_average: d.lod_average,
                    }
                })
                .collect(),
        })
    }

    /// Make the bind groups of `variant` from the encoder's state, and note what the dispatch writes and reads.
    fn bind(
        &self,
        g: &Gpu<'_, A>,
        st: &mut State<A>,
        variant: &ComputeVariant<A>,
    ) -> Result<Vec<(u32, A::BindGroup)>, Failure> {
        const CALL: &str = "-[MTLComputeCommandEncoder dispatch...]";
        let mem = &*self.memory;
        let limits = g.device().limits();
        let storage_align = u64::from(limits.min_storage_buffer_offset_alignment.unwrap_or(256));
        let uniform_align = u64::from(limits.min_uniform_buffer_offset_alignment.unwrap_or(256));
        // The binding alignment of each buffer index, from the kind of binding the variant gives it.
        let mut align: BTreeMap<u64, u64> = BTreeMap::new();
        for group in &variant.groups {
            for b in &group.entries {
                if let (Source::Buffer { index }, Resource::Buffer { ty, .. }) = (&b.source, &b.resource) {
                    let a = if *ty == BufferBindingType::Uniform { uniform_align } else { storage_align };
                    align.insert(*index, a);
                }
            }
        }
        let residual = |index: u64| -> u64 {
            match st.buffers.get(&index) {
                Some(BufferSlot::Buffer { offset, .. }) => offset % align.get(&index).copied().unwrap_or(storage_align),
                _ => 0,
            }
        };
        let mut out = Vec::new();
        for group in &variant.groups {
            // Owned resources first; the entries borrow them.
            enum Owned<A: Api> {
                Buffer(A::Buffer, u64, u64),
                View(A::TextureView),
                Sampler(A::Sampler),
            }
            let mut owned: Vec<(u32, Owned<A>)> = Vec::new();
            for b in &group.entries {
                match (&b.source, &b.resource) {
                    (Source::Buffer { index }, Resource::Buffer { ty, min_binding_size }) => {
                        let slot = st.buffers.get(index).ok_or_else(|| {
                            undefined(CALL, format!("the function uses buffer index {index} and none is bound"))
                        })?;
                        let (gpu, base, size, length) = match slot {
                            BufferSlot::Buffer { buffer, offset } => {
                                let a = align.get(index).copied().unwrap_or(storage_align);
                                let base = offset / a * a;
                                match ty {
                                    BufferBindingType::Storage => {
                                        mem.note_buffer_write(buffer, Some((*offset, buffer.length)))
                                    }
                                    _ => mem.note_buffer_read(buffer),
                                }
                                (buffer.gpu().clone(), base, buffer.gpu().size() - base, buffer.length - offset)
                            }
                            BufferSlot::Bytes { gpu, length } => (gpu.clone(), 0, gpu.size(), *length),
                        };
                        if length < *min_binding_size {
                            return Err(undefined(
                                CALL,
                                format!(
                                    "buffer index {index} holds {length} bytes from its offset; the function's argument \
                                     is {min_binding_size} bytes"
                                ),
                            ));
                        }
                        let limit = match ty {
                            BufferBindingType::Uniform => limits.max_uniform_buffer_binding_size.unwrap_or(65536),
                            _ => limits.max_storage_buffer_binding_size.unwrap_or(128 << 20),
                        };
                        let size = match ty {
                            // A uniform block has the fixed size of its type, within the limit.
                            BufferBindingType::Uniform => size.min(limit & !3),
                            _ if size > limit => {
                                return Err(failure::host_unsupported(format!(
                                    "buffer index {index}: {size} bytes exceed this host's maxStorageBufferBindingSize \
                                     {limit}"
                                )));
                            }
                            _ => size,
                        };
                        owned.push((b.binding, Owned::Buffer(gpu, base, size)));
                    }
                    (Source::Texture { index }, Resource::Texture { sample_type, view_dimension, .. }) => {
                        let e = st.textures.get(index).ok_or_else(|| {
                            undefined(CALL, format!("the function uses texture index {index} and none is bound"))
                        })?;
                        let t = e.tex.texture();
                        let mut d = TextureViewDescriptor::new();
                        d.dimension = Some(*view_dimension);
                        d.aspect = view_aspect(t, *sample_type);
                        if let Some(v) = e.tex.view() {
                            if !v.takes(*view_dimension) {
                                return Err(undefined(CALL, view_mismatch(*index, v, *view_dimension)));
                            }
                            v.describe(&mut d, 0..v.geometry.mip_levels, TextureUsage::TEXTURE_BINDING);
                        }
                        // The swizzle reads apply, which the variant applies in its shader unless the host has
                        // texture-component-swizzle (the variant key holds it); depth reads take none.
                        let swizzle = read_swizzle(e.tex.resolution(), e.tex.swizzle());
                        if swizzle != IDENTITY_SWIZZLE
                            && *sample_type != TextureSampleType::Depth
                            && mem.features().has(Feature::TextureComponentSwizzle)
                        {
                            d.swizzle = swizzle_string(swizzle);
                        }
                        let view = g
                            .call("GPUTexture.createView", &t.label, |_| t.gpu().create_view(&d))
                            .map_err(|ex| MemError::gpu("GPUTexture.createView", ex).failure())?;
                        owned.push((b.binding, Owned::View(view)));
                    }
                    (Source::Texture { index }, Resource::StorageTexture { access, format, view_dimension }) => {
                        let e = st.textures.get(index).cloned().ok_or_else(|| {
                            undefined(CALL, format!("the function uses texture index {index} and none is bound"))
                        })?;
                        let t = e.tex.texture();
                        let writable_format = mem.features().allows(e.tex.resolution().storage().caps().storage_wo);
                        if t.gpu().usage() & TextureUsage::STORAGE_BINDING == 0 || !writable_format {
                            return Err(if t.usage != usage::UNKNOWN && t.usage & usage::SHADER_WRITE == 0 {
                                undefined(
                                    CALL,
                                    format!("texture index {index} is written and lacks MTLTextureUsageShaderWrite"),
                                )
                            } else {
                                failure::host_unsupported(format!(
                                    "texture index {index}: this host cannot write {} textures from shaders",
                                    e.tex.resolution().storage().name()
                                ))
                            });
                        }
                        if e.tex.geometry().mip_levels > 1 {
                            return Err(failure::unimplemented(format!(
                                "{CALL}: texture index {index} is a mipmapped texture bound for writing; the binding \
                                 layout does not say which level the function writes"
                            )));
                        }
                        let mut d = TextureViewDescriptor::new();
                        d.dimension = Some(*view_dimension);
                        d.format = Some(*format);
                        d.mip_level_count = Some(1);
                        let (levels, slices) = e.tex.root_window();
                        if let Some(v) = e.tex.view() {
                            if !v.takes(*view_dimension) {
                                return Err(undefined(CALL, view_mismatch(*index, v, *view_dimension)));
                            }
                            let stored = idl_format(v.resolution.storage());
                            if *format != stored {
                                return Err(failure::unimplemented(format!(
                                    "{CALL}: texture index {index} is a view stored as {} that the function writes as \
                                     {}",
                                    stored.as_str(),
                                    format.as_str()
                                )));
                            }
                            v.describe(&mut d, 0..1, TextureUsage::STORAGE_BINDING);
                            d.format = Some(*format);
                        }
                        let view = g
                            .call("GPUTexture.createView", &t.label, |_| t.gpu().create_view(&d))
                            .map_err(|ex| MemError::gpu("GPUTexture.createView", ex).failure())?;
                        if *access != StorageTextureAccess::ReadOnly {
                            mem.note_texture_write(t, levels.start..levels.start + 1, slices);
                            if let Tex::Linear(lt, _) = &e.tex {
                                mem.note_linear_write(lt);
                                if !st.linear_written.iter().any(|x| x.handle == e.handle) {
                                    st.linear_written.push(e.clone());
                                }
                            }
                        }
                        owned.push((b.binding, Owned::View(view)));
                    }
                    (Source::Sampler { index }, Resource::Sampler { ty }) => {
                        let s = st.samplers.get(index).ok_or_else(|| {
                            undefined(CALL, format!("the function uses sampler index {index} and none is bound"))
                        })?;
                        let gpu = s.sampler(*ty).map_err(|why| {
                            failure::host_unsupported(format!("{CALL}: sampler index {index}: {why}"))
                        })?;
                        owned.push((b.binding, Owned::Sampler(gpu.clone())));
                    }
                    (Source::Uniforms(block), Resource::Buffer { ty, .. }) => {
                        let mut bytes = vec![0u8; block.size as usize];
                        for &(at, value) in &block.fields {
                            let words: Vec<u32> = match value {
                                UniformValue::ResidualOffset { index } => vec![residual(index) as u32],
                                UniformValue::BufferLength { index } => {
                                    let len = match st.buffers.get(&index) {
                                        Some(BufferSlot::Buffer { buffer, offset }) => buffer.length - offset,
                                        Some(BufferSlot::Bytes { length, .. }) => *length,
                                        None => {
                                            return Err(undefined(
                                                CALL,
                                                format!(
                                                    "the function asks the length of buffer index {index}, which is not bound"
                                                ),
                                            ));
                                        }
                                    };
                                    vec![u32::try_from(len).map_err(|_| {
                                        failure::host_unsupported(format!("buffer index {index}: {len} bytes"))
                                    })?]
                                }
                                UniformValue::StageInOrigin | UniformValue::StageInSize => {
                                    let (origin, size) = st.stage_in.ok_or_else(|| {
                                        undefined(CALL, "the pipeline has a stage input and no stage-in region is set")
                                    })?;
                                    (if value == UniformValue::StageInOrigin { origin } else { size }).to_vec()
                                }
                            };
                            for (k, w) in words.iter().enumerate() {
                                let o = at as usize + 4 * k;
                                bytes
                                    .get_mut(o..o + 4)
                                    .ok_or_else(|| {
                                        failure::invalid(format!("uniform field at {at} outside the block"))
                                    })?
                                    .copy_from_slice(&w.to_le_bytes());
                            }
                        }
                        let usage =
                            if *ty == BufferBindingType::Uniform { BufferUsage::UNIFORM } else { BufferUsage::STORAGE };
                        let gpu = crate::blit::kernels::buffer_with(g, "webmetal dispatch uniforms", &bytes, usage)
                            .map_err(|e| e.failure())?;
                        let size = gpu.size();
                        owned.push((b.binding, Owned::Buffer(gpu, 0, size)));
                    }
                    (source, resource) => {
                        return Err(failure::unimplemented(format!(
                            "{CALL}: a binding of {resource:?} from {source:?} is not one the encoder makes"
                        )));
                    }
                }
            }
            let mut entries = Vec::new();
            for (binding, o) in &owned {
                let resource = match o {
                    Owned::Buffer(buffer, base, size) => BindingResource::BufferBinding(BufferBinding {
                        buffer,
                        offset: size64(*base, "createBindGroup")?,
                        size: Some(size64(*size, "createBindGroup")?),
                    }),
                    Owned::View(v) => BindingResource::TextureView(v),
                    Owned::Sampler(s) => BindingResource::Sampler(s),
                };
                entries.push(BindGroupEntry::new(*binding, resource));
            }
            let bg = g.call("GPUDevice.createBindGroup", "", |d| {
                d.create_bind_group(&BindGroupDescriptor::new(&group.layout, entries))
            });
            out.push((group.group, bg));
        }
        Ok(out)
    }
}

/// How many threadgroups a dispatch runs.
enum Grid<A: Api> {
    Groups([u32; 3]),
    /// Read from a buffer at a byte offset when the dispatch runs.
    Indirect(Rc<Buffer<A>>, u64),
}

fn dims(call: &'static str, s: &Size) -> Result<[u32; 3], Failure> {
    let v = |x: u64| u32::try_from(x).map_err(|_| undefined(call, format!("a size component of {x}")));
    Ok([v(s.width)?, v(s.height)?, v(s.depth)?])
}

fn set_sampler<A: Api>(objects: &Table, st: &mut State<A>, sampler: Option<Handle>, index: u64) -> Result<(), Failure> {
    const CALL: &str = "-[MTLComputeCommandEncoder setSamplerState:atIndex:]";
    if index >= MAX_SAMPLERS {
        return Err(undefined(CALL, format!("index {index} of {MAX_SAMPLERS} sampler entries")));
    }
    match sampler {
        Some(h) => {
            let s = payload::<SamplerObject<A>>(objects, h)
                .ok_or_else(|| failure::invalid(format!("{CALL}: handle {:#x} is not a sampler state", h.get())))?;
            st.samplers.insert(index, s);
        }
        None => {
            st.samplers.remove(&index);
        }
    }
    Ok(())
}

/// Why a view cannot be bound where the function's texture type needs a view of `dimension`.
fn view_mismatch<A: Api>(index: u64, v: &TextureView<A>, dimension: TextureViewDimension) -> String {
    format!(
        "texture index {index} is a {:?} view of {} slices, and the function's texture type takes a {} view",
        v.geometry.kind,
        v.geometry.slices(),
        dimension.as_str()
    )
}

/// The `MTLTextureType` value of a texture type (`MTLTexture.h`).
fn texture_type_value(kind: TextureType) -> u64 {
    use TextureType as T;
    match kind {
        T::D1 => 0,
        T::D1Array => 1,
        T::D2 => 2,
        T::D2Array => 3,
        T::D2Multisample => 4,
        T::Cube => 5,
        T::CubeArray => 6,
        T::D3 => 7,
        T::D2MultisampleArray => 8,
        T::TextureBuffer => 9,
    }
}

/// The aspect a sampled view of `t` shows: the depth aspect of a combined depth-stencil texture read as depth or
/// float, its stencil aspect read as uint; every aspect of any other texture.
fn view_aspect<A: Api>(t: &Texture<A>, sample: TextureSampleType) -> TextureAspect {
    let aspects = t.resolution.storage().caps().aspects;
    if aspects.len() < 2 {
        return TextureAspect::All;
    }
    let want = if sample == TextureSampleType::Uint { Aspect::Stencil } else { Aspect::Depth };
    match want {
        Aspect::Depth => TextureAspect::DepthOnly,
        Aspect::Stencil => TextureAspect::StencilOnly,
    }
}

impl<A: Api, E: Embedder> EncoderService<A, E> for ComputeService<A> {
    fn encode(&mut self, cx: &mut EncodeCx<'_, A, E>, segment: &Segment) -> Result<(), RecordFailure> {
        let mem = self.memory.clone();
        let records = cx.records();
        let fail = |offset: u64, failure: Failure| RecordFailure { offset, failure };
        let mut label = String::new();
        for &i in &segment.records {
            if let Command::EncoderSetLabel(r) = &records[i].1 {
                label = r.label.clone().unwrap_or_default();
            }
        }
        // The textures the encoder binds take their current contents before the pass begins: copies cannot be made
        // inside a pass.
        let encoder = cx.encoder().clone();
        for (h, at) in bound_textures(records, segment) {
            let objects = cx.objects();
            let t = payload::<Texture<A>>(objects, h)
                .or_else(|| payload::<TextureView<A>>(objects, h).map(|v| v.root.clone()));
            if let Some(t) = t {
                if mem.prepare_texture(cx.gpu(), &encoder, &t).map_err(|e| fail(at, e.failure()))? == Prepare::Decode {
                    return Err(fail(
                        at,
                        failure::unimplemented(
                            "a software-decoded texture written by a blit in the same command buffer: its texels are \
                             decoded on the CPU once the blit has run",
                        ),
                    ));
                }
            } else if let Some(lt) = payload::<crate::mem::linear::LinearTexture<A>>(objects, h) {
                mem.prepare_linear(cx.gpu(), &encoder, &lt).map_err(|e| fail(at, e.failure()))?;
            }
        }
        let mut d = ComputePassDescriptor::new();
        d.label = label;
        let first = segment.records.first().map_or(0, |&i| records[i].0 as u64);
        let pass = cx
            .gpu()
            .call("GPUCommandEncoder.beginComputePass", "", |_| encoder.begin_compute_pass(&d))
            .map_err(|e| fail(first, MemError::gpu("GPUCommandEncoder.beginComputePass", e).failure()))?;
        let mut st = State::default();
        let mut groups = DebugGroups::default();
        for &i in &segment.records {
            let (offset, record) = &records[i];
            self.record(cx, &pass, &mut st, record, &mut groups).map_err(|f| fail(*offset as u64, f))?;
        }
        for _ in 0..groups.close() {
            cx.gpu().call("GPUComputePassEncoder.popDebugGroup", "", |_| pass.pop_debug_group());
        }
        cx.gpu().call("GPUComputePassEncoder.end", "", |_| pass.end());
        let last = segment.records.last().map_or(0, |&i| records[i].0 as u64);
        for e in &st.linear_written {
            if let Tex::Linear(lt, _) = &e.tex {
                mem.sync_linear(cx.gpu(), &encoder, lt).map_err(|err| fail(last, err.failure()))?;
            }
        }
        Ok(())
    }
}
