//! Depth and stencil multisample resolve: WebGPU resolves color attachments only, so the executor resolves depth and
//! stencil with render passes of its own after the Metal pass (plan section 6.4).
//!
//! Depth: one pass into the resolve texture draws a triangle covering it with depth compare `always` and depth writes
//! on; the fragment stage reads every sample of the multisampled attachment at its pixel and writes the one the filter
//! selects (`MTLMultisampleDepthResolveFilter`: sample 0, the minimum or the maximum) as its depth. A `Depth32Float`
//! texture stores the value written, and every value a multisampled depth attachment holds lies in [0, 1] (depth is
//! clamped to the viewport depth range, which lies in [0, 1]), which the fragment depth clamp keeps, so the resolve
//! copies the selected sample exactly. Other depth formats round on the way back in and are not resolved.
//!
//! Stencil (`MTLMultisampleStencilResolveFilterSample0`, "the stencil sample corresponding to sample 0"): WGSL cannot
//! write a stencil value, so the pass clears the target's stencil to 0 and draws eight times, draw `b` with stencil
//! write mask `1 << b`, reference 0xff and pass operation `replace`, keeping only fragments whose sample-0 stencil has
//! bit `b` set: each bit of the result is the source's. `MTLMultisampleStencilResolveFilterDepthResolvedSample`
//! needs a filter the A10X does not report (`supportsMSAAStencilResolveFilter = false`, device profile); its result
//! there is not established, so it is not implemented.

use crate::gpu::idl::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, LoadOp, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, StoreOp, TextureAspect, TextureFormat, TextureOrView, TextureViewDescriptor,
    TextureViewDimension,
};
use crate::gpu::{
    Api, BindingCommands as _, CommandEncoder as _, Device as _, RenderCommands as _, RenderPassEncoder as _,
    Texture as _,
};
use crate::mem::calls::Calls;

use super::attachments::{DepthStencilAttachment, ResolveTarget};
use super::error::{RenderError, Result};
use super::helpers::{Helpers, ResolveKind, depth_filter_constant};
use super::metal::StencilResolveFilter;

fn view<A: Api>(g: &impl Calls<A>, t: &A::Texture, d: &TextureViewDescriptor) -> Result<A::TextureView> {
    g.call("GPUTexture.createView", &d.label, |_| t.create_view(d))
        .map_err(|e| RenderError::gpu("GPUTexture.createView", e))
}

fn target_view<A: Api>(g: &impl Calls<A>, r: &ResolveTarget<A>) -> Result<A::TextureView> {
    let t = r.texture.texture();
    let mut d = TextureViewDescriptor::new();
    d.label = format!("{} resolve target", t.label);
    d.dimension = Some(TextureViewDimension::D2);
    d.base_mip_level = r.level;
    d.mip_level_count = Some(1);
    d.base_array_layer = r.slice;
    d.array_layer_count = Some(1);
    view::<A>(g, t.gpu(), &d)
}

fn source_view<A: Api>(
    g: &impl Calls<A>,
    ds: &DepthStencilAttachment<A>,
    aspect: TextureAspect,
) -> Result<A::TextureView> {
    let t = ds.texture.texture();
    let mut d = TextureViewDescriptor::new();
    d.label = format!("{} resolve source", t.label);
    d.dimension = Some(TextureViewDimension::D2);
    d.aspect = aspect;
    d.base_mip_level = ds.level;
    d.mip_level_count = Some(1);
    d.base_array_layer = ds.slice;
    d.array_layer_count = Some(1);
    view::<A>(g, t.gpu(), &d)
}

/// Check, before the pass begins, that the depth-stencil attachment's resolves can be done exactly.
pub fn check<A: Api>(ds: &DepthStencilAttachment<A>) -> Result<()> {
    if let Some((att, _, _)) = &ds.depth
        && att.resolve.is_some()
        && !matches!(ds.format, TextureFormat::Depth32float | TextureFormat::Depth32floatStencil8)
    {
        return Err(RenderError::unimplemented(format!(
            "a depth resolve of {}: the resolve pass writes depth as a float, which is exact only for 32-bit float \
             depth",
            ds.format.as_str()
        )));
    }
    if let Some((att, _, filter)) = &ds.stencil
        && att.resolve.is_some()
        && *filter == StencilResolveFilter::DepthResolvedSample
    {
        return Err(RenderError::unimplemented(
            "MTLMultisampleStencilResolveFilterDepthResolvedSample: the A10X reports supportsMSAAStencilResolveFilter \
             = false (device profile) and its result for this filter is not established",
        ));
    }
    Ok(())
}

/// Encode the depth and stencil resolves of the pass's depth-stencil attachment, after its last WebGPU pass.
pub fn encode<A: Api>(
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    helpers: &Helpers<A>,
    ds: &DepthStencilAttachment<A>,
) -> Result<()> {
    check(ds)?;
    let resolves = ds.depth.as_ref().is_some_and(|d| d.0.resolve.is_some())
        || ds.stencil.as_ref().is_some_and(|s| s.0.resolve.is_some());
    if !resolves {
        return Ok(());
    }
    let (depth_layout, stencil_layout) = helpers.resolve_layouts(g)?;
    if let Some((att, _, filter)) = &ds.depth
        && let Some(target) = &att.resolve
    {
        let src = source_view(g, ds, if ds.has_stencil { TextureAspect::DepthOnly } else { TextureAspect::All })?;
        let dst = target_view(g, target)?;
        let pipeline = helpers.resolve_pipeline(
            g,
            ds.format,
            ds.has_depth,
            ds.has_stencil,
            ResolveKind::Depth(depth_filter_constant(*filter)),
        )?;
        let mut bd =
            BindGroupDescriptor::new(&depth_layout, vec![BindGroupEntry::new(0, BindingResource::TextureView(&src))]);
        bd.label = "webmetal depth resolve".into();
        let group = g.call("GPUDevice.createBindGroup", &bd.label, |d| d.create_bind_group(&bd));
        let mut a = RenderPassDepthStencilAttachment::new(TextureOrView::TextureView(&dst));
        a.depth_load_op = Some(LoadOp::Load);
        a.depth_store_op = Some(StoreOp::Store);
        if ds.has_stencil {
            a.stencil_load_op = Some(LoadOp::Load);
            a.stencil_store_op = Some(StoreOp::Store);
        }
        let mut pd = RenderPassDescriptor::new(Vec::new());
        pd.label = "webmetal depth resolve".into();
        pd.depth_stencil_attachment = Some(a);
        let pass = g
            .call("GPUCommandEncoder.beginRenderPass", &pd.label, |_| encoder.begin_render_pass(&pd))
            .map_err(|e| RenderError::gpu("GPUCommandEncoder.beginRenderPass", e))?;
        g.call("GPURenderPassEncoder.setPipeline", "", |_| pass.set_pipeline(&pipeline));
        g.call("GPURenderPassEncoder.setBindGroup", "", |_| pass.set_bind_group(0, Some(&group), &[]));
        g.call("GPURenderPassEncoder.draw", "", |_| pass.draw(3, 1, 0, 0));
        g.call("GPURenderPassEncoder.end", "", |_| pass.end());
    }
    if let Some((att, _, filter)) = &ds.stencil
        && let Some(target) = &att.resolve
    {
        debug_assert_eq!(*filter, StencilResolveFilter::Sample0);
        let src = source_view(g, ds, if ds.has_depth { TextureAspect::StencilOnly } else { TextureAspect::All })?;
        let dst = target_view(g, target)?;
        let mut bd =
            BindGroupDescriptor::new(&stencil_layout, vec![BindGroupEntry::new(1, BindingResource::TextureView(&src))]);
        bd.label = "webmetal stencil resolve".into();
        let group = g.call("GPUDevice.createBindGroup", &bd.label, |d| d.create_bind_group(&bd));
        let mut a = RenderPassDepthStencilAttachment::new(TextureOrView::TextureView(&dst));
        if ds.has_depth {
            a.depth_load_op = Some(LoadOp::Load);
            a.depth_store_op = Some(StoreOp::Store);
        }
        a.stencil_load_op = Some(LoadOp::Clear);
        a.stencil_store_op = Some(StoreOp::Store);
        a.stencil_clear_value = 0;
        let mut pd = RenderPassDescriptor::new(Vec::new());
        pd.label = "webmetal stencil resolve".into();
        pd.depth_stencil_attachment = Some(a);
        let pass = g
            .call("GPUCommandEncoder.beginRenderPass", &pd.label, |_| encoder.begin_render_pass(&pd))
            .map_err(|e| RenderError::gpu("GPUCommandEncoder.beginRenderPass", e))?;
        g.call("GPURenderPassEncoder.setBindGroup", "", |_| pass.set_bind_group(0, Some(&group), &[]));
        g.call("GPURenderPassEncoder.setStencilReference", "", |_| pass.set_stencil_reference(0xff));
        for bit in 0..8 {
            let pipeline =
                helpers.resolve_pipeline(g, ds.format, ds.has_depth, ds.has_stencil, ResolveKind::StencilBit(bit))?;
            g.call("GPURenderPassEncoder.setPipeline", "", |_| pass.set_pipeline(&pipeline));
            g.call("GPURenderPassEncoder.draw", "", |_| pass.draw(3, 1, 0, 0));
        }
        g.call("GPURenderPassEncoder.end", "", |_| pass.end());
    }
    Ok(())
}
