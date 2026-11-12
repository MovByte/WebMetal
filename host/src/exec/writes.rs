//! Which textures a stream record can write.
//!
//! A command buffer changes a texture's contents only through the records below; every other record that names a
//! texture reads it, or changes nothing a later read can observe (`optimizeContentsFor*`, `synchronize*`, the access
//! counters). [`texture_writes`] reports each texture a record names in a writing position, with how it is written,
//! so that a caller can decide whether a committed command buffer may write a given texture without replaying it.
//!
//! A texture reached only through an argument buffer is written by a shader only if the encoder declared it with
//! `useResource:usage:` and `MTLResourceUsageWrite` (`MTLCommandEncoder.h`: resources an argument buffer refers to
//! must be made resident with `useResource`), so the declaration is what names it here.

use webmetal_wire::{Command, Handle, RenderPassDescriptor};

/// `MTLResourceUsageWrite` (`MTLCommandEncoder.h`).
pub const RESOURCE_USAGE_WRITE: u64 = 1 << 1;

/// How a record can write a texture it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureWrite {
    /// An attachment of a render pass, or a resolve texture set for one: written by the pass's load, store and
    /// resolve actions and by its draws.
    Attachment,
    /// The destination of a blit (copies, fills, mipmap generation, invalidation of compressed contents).
    Blit,
    /// Bound to a shader stage. A shader writes a texture only when its `MTLTextureUsage` allows it: `ShaderWrite` or
    /// `ShaderAtomic`, or `Unknown`, which allows every use (`MTLTexture.h`).
    ShaderBinding,
    /// Declared with `useResource:usage:` and `MTLResourceUsageWrite`; written by a shader through an argument buffer
    /// under the same usage rule as [`TextureWrite::ShaderBinding`].
    UseResource,
}

fn pass(d: &RenderPassDescriptor, f: &mut dyn FnMut(Handle, TextureWrite)) {
    let mut each = |t: Option<Handle>| {
        if let Some(t) = t {
            f(t, TextureWrite::Attachment);
        }
    };
    for c in &d.color_attachments {
        each(c.texture);
        each(c.resolve_texture);
    }
    if let Some(a) = &d.depth_attachment {
        each(a.texture);
        each(a.resolve_texture);
    }
    if let Some(a) = &d.stencil_attachment {
        each(a.texture);
        each(a.resolve_texture);
    }
}

/// Report each texture `record` can write, with how. A resource a `useResource` record declares is reported whatever
/// its kind; the caller tells textures from buffers by the object.
pub fn texture_writes(record: &Command, f: &mut dyn FnMut(Handle, TextureWrite)) {
    use TextureWrite::*;
    let mut one = |t: Option<Handle>, w: TextureWrite| {
        if let Some(t) = t {
            f(t, w);
        }
    };
    match record {
        Command::BeginRenderPass(r) => pass(&r.render_pass_descriptor, f),
        Command::BeginParallelRenderPass(r) => pass(&r.render_pass_descriptor, f),
        Command::RenderSetColorResolveTexture(r) => one(r.texture, Attachment),
        Command::RenderSetColorResolveTextureYInvert(r) => one(r.texture, Attachment),
        Command::RenderSetDepthResolveTexture(r) => one(r.texture, Attachment),
        Command::RenderSetDepthResolveTextureYInvert(r) => one(r.texture, Attachment),
        Command::RenderSetStencilResolveTexture(r) => one(r.texture, Attachment),
        Command::RenderSetStencilResolveTextureYInvert(r) => one(r.texture, Attachment),

        Command::RenderSetVertexTexture(r) => one(r.texture, ShaderBinding),
        Command::RenderSetFragmentTexture(r) => one(r.texture, ShaderBinding),
        Command::RenderSetFragmentTextureAndSamplerState(r) => one(r.texture, ShaderBinding),
        Command::RenderSetTileTexture(r) => one(r.texture, ShaderBinding),
        Command::RenderSetObjectTexture(r) => one(r.texture, ShaderBinding),
        Command::RenderSetMeshTexture(r) => one(r.texture, ShaderBinding),
        Command::ComputeSetTexture(r) => one(r.texture, ShaderBinding),
        Command::RenderSetVertexTextures(r) => r.textures.iter().for_each(|t| one(*t, ShaderBinding)),
        Command::RenderSetFragmentTextures(r) => r.textures.iter().for_each(|t| one(*t, ShaderBinding)),
        Command::RenderSetTileTextures(r) => r.textures.iter().for_each(|t| one(*t, ShaderBinding)),
        Command::RenderSetObjectTextures(r) => r.textures.iter().for_each(|t| one(*t, ShaderBinding)),
        Command::RenderSetMeshTextures(r) => r.textures.iter().for_each(|t| one(*t, ShaderBinding)),
        Command::ComputeSetTextures(r) => r.textures.iter().for_each(|t| one(*t, ShaderBinding)),

        Command::RenderUseResource(r) if r.usage & RESOURCE_USAGE_WRITE != 0 => one(r.resource, UseResource),
        Command::RenderUseResourceStages(r) if r.usage & RESOURCE_USAGE_WRITE != 0 => one(r.resource, UseResource),
        Command::ComputeUseResource(r) if r.usage & RESOURCE_USAGE_WRITE != 0 => one(r.resource, UseResource),
        Command::RenderUseResources(r) if r.usage & RESOURCE_USAGE_WRITE != 0 => {
            r.resources.iter().for_each(|t| one(*t, UseResource))
        }
        Command::RenderUseResourcesStages(r) if r.usage & RESOURCE_USAGE_WRITE != 0 => {
            r.resources.iter().for_each(|t| one(*t, UseResource))
        }
        Command::ComputeUseResources(r) if r.usage & RESOURCE_USAGE_WRITE != 0 => {
            r.resources.iter().for_each(|t| one(*t, UseResource))
        }

        Command::BlitCopyBufferToTexture(r) => one(r.destination_texture, Blit),
        Command::BlitCopyBufferToTextureOptions(r) => one(r.destination_texture, Blit),
        Command::BlitCopyTextureToTexture(r) => one(r.destination_texture, Blit),
        Command::BlitCopyTextureToTextureSlicesLevels(r) => one(r.destination_texture, Blit),
        Command::BlitCopyTextureToTextureAll(r) => one(r.destination_texture, Blit),
        Command::BlitCopyTextureToTextureOptions(r) => one(r.destination_texture, Blit),
        Command::BlitGenerateMipmapsForTexture(r) => one(r.texture, Blit),
        Command::BlitFillTextureBytes(r) => one(r.texture, Blit),
        Command::BlitFillTextureColor(r) => one(r.texture, Blit),
        Command::BlitFillTextureColorPixelFormat(r) => one(r.texture, Blit),
        Command::BlitInvalidateCompressedTexture(r) => one(r.texture, Blit),
        Command::BlitInvalidateCompressedTextureSliceLevel(r) => one(r.texture, Blit),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::schema::{Kind, command_handles};
    use std::collections::{BTreeMap, BTreeSet};
    use webmetal_wire::fuzz::Rng;

    /// The stream records that name a texture (or any resource) without writing it.
    const READ_ONLY: [&str; 16] = [
        "BlitCopyTextureToBuffer",
        "BlitCopyTextureToBufferOptions",
        "BlitGetTextureAccessCounters",
        "BlitResetTextureAccessCounters",
        "BlitOptimizeContentsForCPUAccess",
        "BlitOptimizeContentsForCPUAccessSliceLevel",
        "BlitOptimizeContentsForGPUAccess",
        "BlitOptimizeContentsForGPUAccessSliceLevel",
        "BlitSynchronizeResource",
        "BlitSynchronizeTexture",
        "RenderMemoryBarrierWithResources",
        "ComputeMemoryBarrierWithResources",
        "EncodeCacheHintTag",
        "EncodeCacheHintFinalize",
        "SetResourceGroups",
        "DropResourceGroups",
    ];

    /// Every stream record that names a texture or a resource is classified as writing it, or listed as reading it,
    /// so a record added to the schema has to be classified.
    #[test]
    fn every_record_naming_a_texture_is_classified() {
        let streams: BTreeSet<u16> = webmetal_wire::OPS
            .iter()
            .filter(|op| op.channel == webmetal_wire::Channel::Stream)
            .map(|op| op.opcode)
            .collect();
        let mut seen: BTreeMap<u16, (bool, bool)> = BTreeMap::new();
        // Whether a record names or writes a texture depends on its values (optional handles, usage bits), so each op
        // is sampled many times.
        let mut rng = Rng::new(7);
        for _ in 0..200_000 {
            let record = Command::random(&mut rng);
            let mut names = false;
            command_handles(&record, &mut |u| names |= matches!(u.kind, Kind::Texture | Kind::Resource));
            let mut writes = false;
            texture_writes(&record, &mut |_, _| writes = true);
            let e = seen.entry(record.opcode()).or_default();
            e.0 |= names;
            e.1 |= writes;
        }
        assert_eq!(seen.keys().copied().collect::<BTreeSet<_>>(), streams, "every stream op was generated");
        let name = |op: u16| webmetal_wire::op_info(op).expect("op").name;
        let unclassified: Vec<&str> = seen
            .iter()
            .filter(|&(&op, &(names, writes))| names && !writes && !READ_ONLY.contains(&name(op)))
            .map(|(&op, _)| name(op))
            .collect();
        assert!(unclassified.is_empty(), "records naming textures with no classification: {unclassified:?}");
        let misfiled: Vec<&str> = seen
            .iter()
            .filter(|&(&op, &(_, writes))| writes && READ_ONLY.contains(&name(op)))
            .map(|(&op, _)| name(op))
            .collect();
        assert!(misfiled.is_empty(), "read-only records classified as writing: {misfiled:?}");
    }

    #[test]
    fn use_resource_writes_only_with_the_write_usage() {
        let h = Handle::new(9).unwrap();
        let mut got = Vec::new();
        let read = Command::ComputeUseResource(webmetal_wire::ComputeUseResource { resource: Some(h), usage: 1 });
        texture_writes(&read, &mut |t, w| got.push((t, w)));
        assert!(got.is_empty());
        let write = Command::ComputeUseResource(webmetal_wire::ComputeUseResource { resource: Some(h), usage: 3 });
        texture_writes(&write, &mut |t, w| got.push((t, w)));
        assert_eq!(got, vec![(h, TextureWrite::UseResource)]);
    }
}
