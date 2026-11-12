//! Visibility results: `setVisibilityResultMode:offset:` over WebGPU occlusion queries.
//!
//! `MTLVisibilityResultModeBoolean` writes "a nonzero value if at least one fragment passed depth and stencil tests,
//! and zero otherwise" at the offset; a WebGPU occlusion query result is "zero if no samples passed, non-zero
//! otherwise" (WebGPU section 20.4), so a query over the draws of each span of boolean mode measures it exactly.
//! `MTLVisibilityResultModeCounting` asks for the number of samples, which WebGPU does not define: it is refused
//! (`tables/refusals.md` row `counting-occlusion`).
//!
//! Each WebGPU pass of the Metal pass has its own query set; a span that continues across a split of the pass is
//! measured by one query in each, and the offset receives their OR, nonzero exactly when one of them is. After the
//! last pass the queries are resolved and each offset the encoder used is written: one query by a copy, several by
//! the merge program of [`super::helpers`]. Offsets the encoder did not use are not written.
//!
//! Metal requires the offset to be a multiple of 8 (`MTLRenderCommandEncoder.h`) and below the device's
//! `maxVisibilityQueryOffset` (65528 on the A10X, device profile), and rejects reusing an offset within one encoder
//! (debug-layer assertion of reference case `pass.error.visibility-offset-reused`); each of those is a misuse whose
//! result Metal does not define.

use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;

use crate::gpu::idl::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, BufferBinding, BufferDescriptor, BufferUsage,
    ComputePassDescriptor, QuerySetDescriptor, QueryType,
};
use crate::gpu::{Api, CommandEncoder as _, ComputePassEncoder as _, Device as _, Queue as _, RenderPassEncoder as _};
use crate::mem::buffer::Buffer;
use crate::mem::calls::Calls;

use super::error::{RenderError, Result};
use super::helpers::{Helpers, size64};
use super::metal::VisibilityMode;

const CALL: &str = "-[MTLRenderCommandEncoder setVisibilityResultMode:offset:]";

/// `maxVisibilityQueryOffset` of the A10X (device profile).
pub const MAX_VISIBILITY_QUERY_OFFSET: u64 = 65528;

/// The visibility buffer and the byte ranges written to it.
pub type Written<A> = (Rc<Buffer<A>>, Vec<(u64, u64)>);

/// The visibility state of one Metal pass.
pub struct Visibility<A: Api> {
    buffer: Option<Rc<Buffer<A>>>,
    /// Queries a WebGPU pass may need: one per span record, plus a span carried over from the previous pass.
    capacity: u32,
    sets: Vec<(A::QuerySet, u32)>,
    /// Each query: the Metal offset, its set and its index in the set.
    queries: Vec<(u64, usize, u32)>,
    used: HashSet<u64>,
    /// The offset of the boolean span in effect.
    active: Option<u64>,
    /// A query of the current WebGPU pass is open.
    open: bool,
}

impl<A: Api> Visibility<A> {
    /// `buffer`: the pass descriptor's `visibilityResultBuffer`; `spans`: the encoder's `setVisibilityResultMode:`
    /// records.
    pub fn new(buffer: Option<Rc<Buffer<A>>>, spans: u32) -> Visibility<A> {
        Visibility {
            buffer,
            capacity: if spans == 0 { 0 } else { spans + 1 },
            sets: Vec::new(),
            queries: Vec::new(),
            used: HashSet::new(),
            active: None,
            open: false,
        }
    }

    /// The query set of the next WebGPU pass, if the encoder measures visibility.
    pub fn pass_set(&mut self, g: &impl Calls<A>) -> Result<Option<A::QuerySet>> {
        if self.capacity == 0 {
            return Ok(None);
        }
        let mut d = QuerySetDescriptor::new(QueryType::Occlusion, self.capacity);
        d.label = "webmetal visibility results".into();
        let set = g
            .call("GPUDevice.createQuerySet", &d.label, |dev| dev.create_query_set(&d))
            .map_err(|e| RenderError::gpu("GPUDevice.createQuerySet", e))?;
        self.sets.push((set.clone(), 0));
        Ok(Some(set))
    }

    fn begin(&mut self, g: &impl Calls<A>, pass: &A::RenderPassEncoder) -> Result<()> {
        let Some(offset) = self.active else { return Ok(()) };
        let set = self.sets.len().checked_sub(1).expect("a WebGPU pass with a query set is open");
        let index = self.sets[set].1;
        if index >= self.capacity {
            return Err(RenderError::invalid("more visibility spans than the encoder's records"));
        }
        self.sets[set].1 += 1;
        self.queries.push((offset, set, index));
        g.call("GPURenderPassEncoder.beginOcclusionQuery", "", |_| pass.begin_occlusion_query(index));
        self.open = true;
        Ok(())
    }

    fn end(&mut self, g: &impl Calls<A>, pass: &A::RenderPassEncoder) {
        if self.open {
            g.call("GPURenderPassEncoder.endOcclusionQuery", "", |_| pass.end_occlusion_query());
            self.open = false;
        }
    }

    /// A WebGPU pass began: a span in effect continues in it.
    pub fn pass_begun(&mut self, g: &impl Calls<A>, pass: &A::RenderPassEncoder) -> Result<()> {
        self.begin(g, pass)
    }

    /// The WebGPU pass is about to end.
    pub fn pass_ending(&mut self, g: &impl Calls<A>, pass: &A::RenderPassEncoder) {
        self.end(g, pass);
    }

    /// `setVisibilityResultMode:offset:`.
    pub fn set_mode(
        &mut self,
        g: &impl Calls<A>,
        pass: &A::RenderPassEncoder,
        mode: VisibilityMode,
        offset: u64,
    ) -> Result<()> {
        self.end(g, pass);
        self.active = None;
        match mode {
            VisibilityMode::Disabled => Ok(()),
            VisibilityMode::Counting => Err(RenderError::host(
                "counting-occlusion: MTLVisibilityResultModeCounting; a WebGPU occlusion result is only defined as \
                 zero or non-zero (tables/refusals.md)",
            )),
            VisibilityMode::Boolean => {
                let Some(buffer) = &self.buffer else {
                    return Err(RenderError::undefined(CALL, "the render pass has no visibilityResultBuffer"));
                };
                if !offset.is_multiple_of(8) {
                    return Err(RenderError::undefined(CALL, format!("offset {offset} is not a multiple of 8")));
                }
                if offset > MAX_VISIBILITY_QUERY_OFFSET || offset + 8 > buffer.length {
                    return Err(RenderError::undefined(
                        CALL,
                        format!(
                            "offset {offset} beyond the {}-byte visibility buffer or maxVisibilityQueryOffset {}",
                            buffer.length, MAX_VISIBILITY_QUERY_OFFSET
                        ),
                    ));
                }
                if !self.used.insert(offset) {
                    return Err(RenderError::undefined(CALL, format!("offset {offset} used twice in one encoder")));
                }
                self.active = Some(offset);
                self.begin(g, pass)
            }
        }
    }

    /// A parallel sub-encoder ended: its span ends with it.
    pub fn end_span(&mut self, g: &impl Calls<A>, pass: &A::RenderPassEncoder) {
        self.end(g, pass);
        self.active = None;
    }

    /// After the last WebGPU pass: resolve the queries and write each used offset. Returns the visibility buffer and
    /// the byte ranges written.
    pub fn finish(
        &mut self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        helpers: &Helpers<A>,
        storage_offset_alignment: u64,
    ) -> Result<Option<Written<A>>> {
        let Some(buffer) = self.buffer.clone() else { return Ok(None) };
        if self.queries.is_empty() {
            return Ok(Some((buffer, Vec::new())));
        }
        // Resolve every set into one staging buffer, 256 bytes apart (resolveQuerySet's destination alignment).
        let stride = (u64::from(self.capacity) * 8).next_multiple_of(256);
        let mut d = BufferDescriptor::new(
            size64(stride * self.sets.len() as u64, "GPUDevice.createBuffer")?,
            BufferUsage::QUERY_RESOLVE | BufferUsage::COPY_SRC | BufferUsage::STORAGE,
        );
        d.label = "webmetal visibility resolve".into();
        let staging = g
            .call("GPUDevice.createBuffer", &d.label, |dev| dev.create_buffer(&d))
            .map_err(|e| RenderError::gpu("GPUDevice.createBuffer", e))?;
        for (i, (set, used)) in self.sets.iter().enumerate() {
            if *used > 0 {
                let at = size64(stride * i as u64, "GPUCommandEncoder.resolveQuerySet")?;
                g.call("GPUCommandEncoder.resolveQuerySet", &d.label, |_| {
                    encoder.resolve_query_set(set, 0, *used, &staging, at)
                });
            }
        }
        let mut by_offset: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for &(offset, set, index) in &self.queries {
            by_offset.entry(offset).or_default().push(stride * set as u64 + u64::from(index) * 8);
        }
        let mut merged = Vec::new();
        for (&offset, positions) in &by_offset {
            if let [single] = positions.as_slice() {
                g.call("GPUCommandEncoder.copyBufferToBuffer", &d.label, |_| {
                    encoder.copy_buffer_to_buffer_with_offsets(
                        &staging,
                        size64(*single, "GPUCommandEncoder.copyBufferToBuffer").expect("small"),
                        buffer.gpu(),
                        size64(offset, "GPUCommandEncoder.copyBufferToBuffer").expect("checked"),
                        Some(size64(8, "GPUCommandEncoder.copyBufferToBuffer").expect("small")),
                    )
                });
            } else {
                merged.push((offset, positions.clone()));
            }
        }
        if !merged.is_empty() {
            self.merge(g, encoder, helpers, &staging, &buffer, &merged, storage_offset_alignment)?;
        }
        Ok(Some((buffer, by_offset.keys().map(|&o| (o, o + 8)).collect())))
    }

    #[allow(clippy::too_many_arguments)]
    fn merge(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        helpers: &Helpers<A>,
        staging: &A::Buffer,
        buffer: &Buffer<A>,
        merged: &[(u64, Vec<u64>)],
        alignment: u64,
    ) -> Result<()> {
        let program = helpers.visibility_merge(g)?;
        let lo = merged.iter().map(|m| m.0).min().expect("non-empty");
        let hi = merged.iter().map(|m| m.0 + 8).max().expect("non-empty");
        let base = lo - lo % alignment;
        let mut targets: Vec<u32> = Vec::new();
        let mut list: Vec<u32> = Vec::new();
        for (offset, positions) in merged {
            targets.extend([((offset - base) / 4) as u32, list.len() as u32, positions.len() as u32, 0]);
            list.extend(positions.iter().map(|p| (p / 8) as u32));
        }
        let upload = |words: &[u32], what: &str| -> Result<A::Buffer> {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let mut d = BufferDescriptor::new(
                size64(bytes.len().max(4) as u64, "GPUDevice.createBuffer")?,
                BufferUsage::STORAGE | BufferUsage::COPY_DST,
            );
            d.label = format!("webmetal visibility merge {what}");
            let b = g
                .call("GPUDevice.createBuffer", &d.label, |dev| dev.create_buffer(&d))
                .map_err(|e| RenderError::gpu("GPUDevice.createBuffer", e))?;
            g.call("GPUQueue.writeBuffer", &d.label, |_| {
                g.queue().write_buffer(&b, crate::gpu::Size64::from(0u32), &bytes, crate::gpu::Size64::from(0u32), None)
            })
            .map_err(|e| RenderError::gpu("GPUQueue.writeBuffer", e))?;
            Ok(b)
        };
        let target_buffer = upload(&targets, "targets")?;
        let list_buffer = upload(&list, "queries")?;
        let mut window = BufferBinding::new(buffer.gpu());
        window.offset = size64(base, "GPUDevice.createBindGroup")?;
        window.size = Some(size64(hi.next_multiple_of(4) - base, "GPUDevice.createBindGroup")?);
        let entries = vec![
            BindGroupEntry::new(0, BindingResource::Buffer(staging)),
            BindGroupEntry::new(1, BindingResource::Buffer(&target_buffer)),
            BindGroupEntry::new(2, BindingResource::Buffer(&list_buffer)),
            BindGroupEntry::new(3, BindingResource::BufferBinding(window)),
        ];
        let mut bd = BindGroupDescriptor::new(&program.layout, entries);
        bd.label = "webmetal visibility merge".into();
        let group = g.call("GPUDevice.createBindGroup", &bd.label, |dev| dev.create_bind_group(&bd));
        let mut pd = ComputePassDescriptor::new();
        pd.label = "webmetal visibility merge".into();
        let pass = g
            .call("GPUCommandEncoder.beginComputePass", &pd.label, |_| encoder.begin_compute_pass(&pd))
            .map_err(|e| RenderError::gpu("GPUCommandEncoder.beginComputePass", e))?;
        g.call("GPUComputePassEncoder.setPipeline", "", |_| pass.set_pipeline(&program.pipeline));
        g.call("GPUComputePassEncoder.setBindGroup", "", |_| {
            crate::gpu::BindingCommands::set_bind_group(&pass, 0, Some(&group), &[])
        });
        let groups = (merged.len() as u32).div_ceil(64);
        g.call("GPUComputePassEncoder.dispatchWorkgroups", "", |_| pass.dispatch_workgroups(groups, 1, 1));
        g.call("GPUComputePassEncoder.end", "", |_| pass.end());
        Ok(())
    }
}
