//! Index staging: Metal's primitive restart for indexed draws of list topologies.
//!
//! Metal treats the largest index value as a restart sentinel in every indexed draw: it "complete[s] a primitive and
//! start[s] a new one" (Apple's documentation of `drawIndexedPrimitives`). WebGPU splits the index list only for strip
//! topologies with `stripIndexFormat`; for a list topology the sentinel is an ordinary index (`tables/limits.toml`
//! question `6.5-primitive-restart`). Strips therefore map directly. For an indexed point, line or triangle list the
//! executor stages the indices before the pass: a compute pass ([`super::helpers::INDEX_STAGING_WGSL`]) assembles them
//! as Metal does, writing the indices of every complete primitive into a staging index buffer and the
//! `drawIndexedIndirect` arguments that draw them. The index data may have been written by earlier GPU work of the
//! command buffer, so the staging runs on the GPU, in stream order. A render pass cannot write its own index buffers
//! for a later draw (Metal orders no fragment stores before later vertex fetch inside a pass: a barrier after the
//! fragment stage is rejected on Apple-family GPUs, reference case `render.error.memory-barrier-after-fragment`), so
//! staging every draw of a pass before the pass begins sees the same index data the draws see.
//!
//! [`assemble`] is the same rule on the CPU, for the tests.

use std::rc::Rc;

use crate::gpu::idl::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, BufferBinding, BufferDescriptor, BufferUsage,
    ComputePassDescriptor,
};
use crate::gpu::{Api, Buffer as _, CommandEncoder as _, ComputePassEncoder as _, Device as _, Queue as _, Size64};
use crate::mem::buffer::Buffer;
use crate::mem::calls::Calls;

use super::error::{RenderError, Result};
use super::helpers::{Helpers, size64};
use super::metal::IndexType;

/// The indices of the complete primitives of `indices`, assembled with Metal's restart rule: a sentinel ends the
/// primitive being assembled, which is dropped if incomplete, and so is an incomplete primitive at the end.
pub fn assemble(indices: &[u32], per_primitive: u32, restart: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut run = 0usize;
    for &v in indices {
        if v == restart {
            out.truncate(out.len() - run);
            run = 0;
            continue;
        }
        out.push(v);
        run += 1;
        if run == per_primitive as usize {
            run = 0;
        }
    }
    out.truncate(out.len() - run);
    out
}

/// Where a staged draw's index range and draw parameters come from.
pub enum StageSource<A: Api> {
    /// A direct draw.
    Direct { count: u32, instances: u32, base_vertex: i32, base_instance: u32 },
    /// An indirect draw: `MTLDrawIndexedPrimitivesIndirectArguments` at `offset` of `buffer`.
    Indirect { buffer: Rc<Buffer<A>>, offset: u64 },
}

/// The result of staging: an index buffer of `uint32` indices and the indirect arguments that draw them.
pub struct Staged<A: Api> {
    pub indices: A::Buffer,
    pub args: A::Buffer,
}

/// The device limits staging needs.
#[derive(Clone, Copy, Debug)]
pub struct StagingLimits {
    pub storage_offset_alignment: u64,
    pub max_storage_binding: u64,
}

const CALL: &str = "-[MTLRenderCommandEncoder drawIndexedPrimitives:...]";

fn window<A: Api>(b: &Buffer<A>, offset: u64, end: u64, limits: StagingLimits) -> Result<(u64, u64)> {
    let base = offset - offset % limits.storage_offset_alignment;
    let size = end.next_multiple_of(4).min(b.gpu().size()) - base;
    if size > limits.max_storage_binding {
        return Err(RenderError::host(format!(
            "index staging needs a {size}-byte storage binding; the device's maxStorageBufferBindingSize is {}",
            limits.max_storage_binding
        )));
    }
    Ok((base, size))
}

/// Encode the staging of one draw on `encoder` (outside any pass). `per_primitive` is 1, 2 or 3.
#[allow(clippy::too_many_arguments)]
pub fn encode<A: Api>(
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    helpers: &Helpers<A>,
    limits: StagingLimits,
    index_buffer: &Rc<Buffer<A>>,
    index_offset: u64,
    index_type: IndexType,
    per_primitive: u32,
    source: &StageSource<A>,
) -> Result<Staged<A>> {
    let program = helpers.index_staging(g)?;
    let size = index_type.size();
    let available = index_buffer.length.saturating_sub(index_offset) / size;
    // The index window: the draw's range for a direct draw, the rest of the buffer for an indirect one.
    let (count_max, end) = match source {
        StageSource::Direct { count, .. } => {
            let end = index_offset + u64::from(*count) * size;
            if end > index_buffer.length {
                return Err(RenderError::undefined(
                    CALL,
                    format!("indices {index_offset}..{end} of a {}-byte index buffer", index_buffer.length),
                ));
            }
            (u64::from(*count), end)
        }
        StageSource::Indirect { .. } => (available, index_buffer.length),
    };
    let (src_base, src_size) = window(index_buffer, index_offset, end.max(index_offset + 4), limits)?;

    let make = |len: u64, usage: u32, what: &str| -> Result<A::Buffer> {
        let mut d = BufferDescriptor::new(size64(len, "GPUDevice.createBuffer")?, usage);
        d.label = format!("webmetal index staging {what}");
        g.call("GPUDevice.createBuffer", &d.label, |dev| dev.create_buffer(&d))
            .map_err(|e| RenderError::gpu("GPUDevice.createBuffer", e))
    };
    let indices = make((count_max * 4).max(4), BufferUsage::STORAGE | BufferUsage::INDEX, "indices")?;
    let args = make(20, BufferUsage::STORAGE | BufferUsage::INDIRECT, "arguments")?;
    let params = make(48, BufferUsage::UNIFORM | BufferUsage::COPY_DST, "parameters")?;

    let (indirect, count, instances, base_vertex, base_instance, args_window) = match source {
        StageSource::Direct { count, instances, base_vertex, base_instance } => {
            (0u32, *count, *instances, *base_vertex as u32, *base_instance, None)
        }
        StageSource::Indirect { buffer, offset } => {
            if offset % 4 != 0 || offset + 20 > buffer.length {
                return Err(RenderError::undefined(
                    CALL,
                    format!("indirect arguments at offset {offset} of a {}-byte buffer", buffer.length),
                ));
            }
            let (base, wsize) = window(buffer, *offset, offset + 20, limits)?;
            (1, 0, 0, 0, 0, Some((buffer.clone(), base, wsize, ((offset - base) / 4) as u32)))
        }
    };
    let args_word = args_window.as_ref().map_or(0, |w| w.3);
    let words: [u32; 12] = [
        (index_offset - src_base) as u32,
        size as u32,
        per_primitive,
        indirect,
        count,
        instances,
        base_vertex,
        base_instance,
        index_type.restart(),
        args_word,
        0,
        0,
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    g.call("GPUQueue.writeBuffer", "webmetal index staging", |_| {
        g.queue().write_buffer(&params, Size64::from(0u32), &bytes, Size64::from(0u32), None)
    })
    .map_err(|e| RenderError::gpu("GPUQueue.writeBuffer", e))?;

    fn binding<A: Api>(buffer: &A::Buffer, offset: u64, size: u64) -> Result<BufferBinding<'_, A>> {
        let mut b = BufferBinding::new(buffer);
        b.offset = size64(offset, "GPUDevice.createBindGroup")?;
        b.size = Some(size64(size, "GPUDevice.createBindGroup")?);
        Ok(b)
    }
    let (args_buffer, args_base, args_size) = match &args_window {
        Some((b, base, size, _)) => (b.gpu(), *base, *size),
        None => (index_buffer.gpu(), src_base, src_size),
    };
    let entries = vec![
        BindGroupEntry::new(0, BindingResource::BufferBinding(binding::<A>(index_buffer.gpu(), src_base, src_size)?)),
        BindGroupEntry::new(1, BindingResource::BufferBinding(binding::<A>(args_buffer, args_base, args_size)?)),
        BindGroupEntry::new(2, BindingResource::Buffer(&indices)),
        BindGroupEntry::new(3, BindingResource::Buffer(&args)),
        BindGroupEntry::new(4, BindingResource::Buffer(&params)),
    ];
    let mut bd = BindGroupDescriptor::new(&program.layout, entries);
    bd.label = "webmetal index staging".into();
    let group = g.call("GPUDevice.createBindGroup", &bd.label, |dev| dev.create_bind_group(&bd));
    let mut pd = ComputePassDescriptor::new();
    pd.label = "webmetal index staging".into();
    let pass = g
        .call("GPUCommandEncoder.beginComputePass", "webmetal index staging", |_| encoder.begin_compute_pass(&pd))
        .map_err(|e| RenderError::gpu("GPUCommandEncoder.beginComputePass", e))?;
    g.call("GPUComputePassEncoder.setPipeline", "webmetal index staging", |_| pass.set_pipeline(&program.pipeline));
    g.call("GPUComputePassEncoder.setBindGroup", "webmetal index staging", |_| {
        crate::gpu::BindingCommands::set_bind_group(&pass, 0, Some(&group), &[])
    });
    g.call("GPUComputePassEncoder.dispatchWorkgroups", "webmetal index staging", |_| pass.dispatch_workgroups(1, 1, 1));
    g.call("GPUComputePassEncoder.end", "webmetal index staging", |_| pass.end());
    Ok(Staged { indices, args })
}

#[cfg(test)]
mod tests {
    use super::assemble;

    #[test]
    fn restart_drops_incomplete_primitives() {
        // Triangles: (0,1,2), then 3 is dropped at the restart, then (4,5,6); the trailing 7 is incomplete.
        assert_eq!(assemble(&[0, 1, 2, 3, 0xffff, 4, 5, 6, 7], 3, 0xffff), vec![0, 1, 2, 4, 5, 6]);
        // Lines: (0,1), 2 dropped, (3,4).
        assert_eq!(assemble(&[0, 1, 2, 0xffff_ffff, 3, 4], 2, 0xffff_ffff), vec![0, 1, 3, 4]);
        // Points: the sentinel is skipped.
        assert_eq!(assemble(&[0xffff, 0, 2, 0xffff, 4], 1, 0xffff), vec![0, 2, 4]);
        // Without a sentinel only the trailing incomplete primitive goes.
        assert_eq!(assemble(&[5, 6, 7, 8], 3, 0xffff), vec![5, 6, 7]);
        // A 16-bit sentinel is an ordinary index of a 32-bit draw.
        assert_eq!(assemble(&[0xffff, 1, 2], 3, 0xffff_ffff), vec![0xffff, 1, 2]);
    }
}
