//! Metal textures: the WebGPU texture that holds them, the exact copy of their bytes some formats need, and the CPU
//! (`replaceRegion`, `getBytes`) and blit transfers of their contents.
//!
//! **Stored texture.** Every texture is a WebGPU texture in the format the format module resolves for the host
//! ([`crate::format::handling::resolve_texture`]): 1D and 1D-array textures are 2D textures one row high, cubes and
//! cube arrays are 2D arrays of six layers per cube, 3D textures are 3D (plan section 6.3). It carries every usage
//! the Metal descriptor permits and the host's format allows (plan section 6.10), and for `pixelFormatView` usage the
//! sRGB sibling as a view format.
//!
//! **Twin.** WebGPU texel copies preserve bits only for the uint, sint and non-packed unorm formats; for float, snorm,
//! packed and compressed formats they guarantee only values (WebGPU section 13.3.1, `tables/formats.toml` question
//! `6.3-texel-copy-bit-exactness`), and a software-decoded compressed texture holds decoded texels, not its blocks.
//! A texture in such a format therefore also keeps its bytes in a twin: one uint texture per mipmap level whose texels
//! are the format's blocks (1, 2, 4, 8 or 16 bytes), which copies move exactly. The PVRTC twin holds a level's words
//! in row-linear order, the order Metal's blits use (`MTLBlitOptionRowLinearPVRTC`, which PVRTC blits require); the
//! CPU calls take them in Metal's reflected Morton order and reorder on the CPU. Each subresource records which of the
//! two copies is current:
//!
//! - a CPU upload (`replaceRegion`) writes every current copy;
//! - a blit from a buffer writes the twin and leaves the stored texture stale until [`Memory::prepare_texture`]
//!   brings it up to date before its next GPU use (on the GPU, or by decoding on the CPU for decoded formats);
//! - GPU writes of the stored texture ([`Memory::note_texture_write`]) leave the twin stale;
//! - reads (`getBytes`, blits to a buffer) use the twin when it is current, else the stored texture.
//!
//! Moving bytes between the CPU or a buffer and a stored texture of a format whose copies are not bit-exact needs the
//! host's texel copies of that format to be proven exact first ([`super::proof`]); without that the transfer is
//! refused, never approximated.

use std::cell::RefCell;
use std::rc::Rc;

use crate::embed::{Embedder, TaskId};
use crate::format::handling::{self, Dimension, Emulation, Resolution, Unavailable, Upload, Usage};
use crate::format::packed;
use crate::format::pvrtc;
use crate::format::storage;
use crate::format::webgpu::{Aspect, Feature, WebGpuFormat};
use crate::gpu::idl::{
    BufferDescriptor, BufferUsage, CommandBufferDescriptor, CommandEncoderDescriptor, Extent3D, Extent3DDict, MapMode,
    Origin3D, Origin3DDict, TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDescriptor as GpuTextureDescriptor, TextureDimension, TextureFormat, TextureUsage, TextureViewDimension,
};
use crate::gpu::{
    Api, Buffer as _, CommandEncoder as _, Device as _, GpuFuture, MappedRange as _, Queue as _, Size64, Texture as _,
};
use webmetal_wire::{GetBytes, GetBytesSlice, Handle, ReplaceRegion, ReplaceRegionSlice, TextureDescriptor};

use crate::blit::kernels::{DepthWrite, Rows, Source, depth_bits, encode_depth_write, encode_rows, encode_swap10};

use super::buffer::{Buffer, label};
use super::calls::Calls;
use super::error::MemError;
use super::layout::{Blocks, Geometry, RegionArgs, RegionLayout, StagingPitch, TextureType};
use super::memory::Memory;
use super::options::{ResourceOptions, StorageMode};
use super::purgeable::Purgeable;
use super::surface::SurfacePlane;

/// `MTLTextureUsage` bits (`MTLTexture.h`).
pub mod usage {
    pub const UNKNOWN: u64 = 0x0000;
    pub const SHADER_READ: u64 = 0x0001;
    pub const SHADER_WRITE: u64 = 0x0002;
    pub const RENDER_TARGET: u64 = 0x0004;
    pub const PIXEL_FORMAT_VIEW: u64 = 0x0010;
}

/// The WebGPU texture format of a format module format.
pub fn idl_format(f: WebGpuFormat) -> TextureFormat {
    *TextureFormat::ALL.iter().find(|t| t.as_str() == f.name()).expect("the format tables share the IDL's names")
}

/// The uint format whose texels are `bytes` bytes: the twin of a format with blocks of that size.
fn twin_format(bytes: u32) -> Option<WebGpuFormat> {
    Some(match bytes {
        1 => WebGpuFormat::R8uint,
        2 => WebGpuFormat::R16uint,
        4 => WebGpuFormat::R32uint,
        8 => WebGpuFormat::Rg32uint,
        16 => WebGpuFormat::Rgba32uint,
        _ => return None,
    })
}

/// Does a texture of this resolution need a twin for its bytes to be exact?
fn needs_twin(r: Resolution) -> bool {
    match r {
        Resolution::Native(f) => f.is_compressed() || (!f.texel_copy_exact(None) && f.caps().aspects.is_empty()),
        Resolution::Emulated(e) => matches!(e.upload, Upload::Decode(_)) || !e.storage.texel_copy_exact(None),
        Resolution::StencilAspect => false,
    }
}

#[derive(Debug)]
struct State {
    /// Per subresource (`level * per_level + layer`): the stored texture holds the current contents.
    stored: Vec<bool>,
    /// Per subresource: the twin holds the current contents.
    twin: Vec<bool>,
    purgeable: Purgeable,
    /// For an IOSurface texture: the plane version the texture holds, and whether GPU work wrote it since its last
    /// write-back.
    surface_version: u64,
    surface_written: bool,
}

/// A Metal texture: the payload of a texture handle in the executor's object table.
pub struct Texture<A: Api> {
    /// `MTLPixelFormat`.
    pub format: u32,
    pub resolution: Resolution,
    pub geometry: Geometry,
    pub storage: StorageMode,
    /// `MTLTextureUsage`.
    pub usage: u64,
    /// The format's blocks, `None` for formats whose bytes move one aspect at a time (combined depth-stencil).
    pub blocks: Option<Blocks>,
    /// The IOSurface plane whose memory the texture is, for `newTextureWithDescriptor:iosurface:plane:`.
    pub surface: Option<Rc<SurfacePlane>>,
    pub label: String,
    stored: A::Texture,
    twins: Vec<A::Texture>,
    state: RefCell<State>,
}

impl<A: Api> Texture<A> {
    /// The WebGPU texture the GPU uses.
    pub fn gpu(&self) -> &A::Texture {
        &self.stored
    }

    pub fn has_twin(&self) -> bool {
        !self.twins.is_empty()
    }

    fn per_level(&self) -> u32 {
        if self.geometry.kind.is_3d() { 1 } else { self.geometry.slices() }
    }

    fn sub(&self, level: u32, layer: u32) -> usize {
        (level * self.per_level() + layer) as usize
    }

    /// Does the stored texture hold the current contents of `(level, layer)`?
    pub fn stored_current(&self, level: u32, layer: u32) -> bool {
        self.state.borrow().stored[self.sub(level, layer)]
    }

    /// Does the twin hold the current contents of `(level, layer)`?
    pub fn twin_current(&self, level: u32, layer: u32) -> bool {
        self.has_twin() && self.state.borrow().twin[self.sub(level, layer)]
    }

    pub fn purgeable(&self) -> Purgeable {
        self.state.borrow().purgeable
    }

    /// The twin texture of `level`, if the texture keeps one.
    pub(crate) fn twin(&self, level: u32) -> Option<&A::Texture> {
        self.twins.get(level as usize)
    }

    /// The layer of `slice` in the per-subresource state: the slice, or 0 for a 3D texture.
    pub(crate) fn layer_of(&self, slice: u32) -> u32 {
        if self.geometry.kind.is_3d() { 0 } else { slice }
    }

    /// Record which copies of `(level, layer)` hold the current contents after a blit wrote them: the stored texture,
    /// the twin, or both. A texture over an IOSurface plane is marked written.
    pub(crate) fn set_current(&self, level: u32, layer: u32, stored: bool, twin: bool) {
        debug_assert!(stored || (twin && self.has_twin()), "a subresource keeps a current copy");
        let i = self.sub(level, layer);
        let mut st = self.state.borrow_mut();
        st.stored[i] = stored;
        st.twin[i] = twin && self.has_twin();
        if self.surface.is_some() {
            st.surface_written = true;
        }
    }

    pub(crate) fn surface_state(&self) -> (u64, bool) {
        let st = self.state.borrow();
        (st.surface_version, st.surface_written)
    }

    pub(crate) fn set_surface_state(&self, version: u64, written: bool) {
        let mut st = self.state.borrow_mut();
        st.surface_version = version;
        st.surface_written = written;
    }
}

fn origin(x: u32, y: u32, z: u32) -> Origin3D {
    Origin3D::Origin3DDict(Origin3DDict { x, y, z })
}

fn extent(w: u32, h: u32, d: u32) -> Extent3D {
    Extent3D::Extent3DDict(Extent3DDict { width: w, height: h, depth_or_array_layers: d })
}

fn size64(v: u64, op: &'static str) -> Result<Size64, MemError> {
    Size64::new(v).map_err(|e| MemError::gpu(op, e))
}

fn gpu_err(op: &'static str) -> impl Fn(crate::gpu::Exception) -> MemError {
    move |e| MemError::gpu(op, e)
}

/// One copy of a region of a texture: which WebGPU texture, where, and the bytes of a row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Placement {
    pub(crate) level: u32,
    pub(crate) origin: (u32, u32, u32),
    pub(crate) extent: (u32, u32, u32),
    pub(crate) row_bytes: u64,
    pub(crate) rows: u32,
}

/// How the bytes read from a copy become Metal's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Convert {
    None,
    SwapRedBlue10,
    /// Row-linear PVRTC words of a `width` × `height` level to Morton order.
    PvrtcToMorton {
        bpp: pvrtc::Bpp,
        width: u32,
        height: u32,
    },
}

impl<A: Api> Memory<A> {
    /// `newTextureWithDescriptor:` (CreateTexture).
    pub fn create_texture(
        &self,
        g: &impl Calls<A>,
        task: TaskId,
        handle: Handle,
        d: &TextureDescriptor,
    ) -> Result<Rc<Texture<A>>, MemError> {
        self.texture_from_descriptor(g, "-[MTLDevice newTextureWithDescriptor:]", task, handle, d)
    }

    pub(crate) fn texture_from_descriptor(
        &self,
        g: &impl Calls<A>,
        call: &'static str,
        task: TaskId,
        handle: Handle,
        d: &TextureDescriptor,
    ) -> Result<Rc<Texture<A>>, MemError> {
        self.build_texture(g, call, (task, handle), d, None)
    }

    /// Make a texture. A texture over an IOSurface plane (`surface`) keeps no twin: its bytes are the plane's, held in
    /// the plane's shadow, and move between it and the stored texture on the CPU.
    pub(crate) fn build_texture(
        &self,
        g: &impl Calls<A>,
        call: &'static str,
        (task, handle): (TaskId, Handle),
        d: &TextureDescriptor,
        surface: Option<Rc<SurfacePlane>>,
    ) -> Result<Rc<Texture<A>>, MemError> {
        let kind = TextureType::from_value(d.texture_type, call)?;
        let storage = StorageMode::from_value(d.storage_mode, call)?;
        let options = ResourceOptions::decode(d.resource_options, call)?;
        if options.storage != storage || options.cpu_cache.value() != d.cpu_cache_mode {
            return Err(MemError::invalid(format!(
                "{call}: texture descriptor resourceOptions {:#x} disagrees with its storage and CPU cache modes",
                d.resource_options
            )));
        }
        match kind {
            TextureType::TextureBuffer => {
                return Err(MemError::undefined(call, "MTLTextureTypeTextureBuffer textures are made from a buffer"));
            }
            TextureType::D2MultisampleArray => {
                return Err(MemError::undefined(call, "the A10X has no MTLTextureType2DMultisampleArray (plan E 2.7)"));
            }
            _ => {}
        }
        let dims = |v: u64, what: &str| {
            u32::try_from(v).ok().filter(|&v| v > 0).ok_or_else(|| MemError::undefined(call, format!("{what} {v}")))
        };
        let geometry = Geometry {
            kind,
            width: dims(d.width, "width")?,
            height: if kind.is_1d() { 1 } else { dims(d.height, "height")? },
            depth: if kind.is_3d() { dims(d.depth, "depth")? } else { 1 },
            array_length: dims(d.array_length, "arrayLength")?,
            mip_levels: dims(d.mipmap_level_count, "mipmapLevelCount")?,
            sample_count: dims(d.sample_count, "sampleCount")?,
        };
        let full_chain = 32 - geometry.width.max(geometry.height).max(geometry.depth).leading_zeros();
        if geometry.mip_levels > full_chain {
            return Err(MemError::undefined(
                call,
                format!("{} mipmap levels exceed the full chain", geometry.mip_levels),
            ));
        }
        let multisampled = kind == TextureType::D2Multisample;
        if multisampled != (geometry.sample_count > 1) || (multisampled && geometry.mip_levels != 1) {
            return Err(MemError::undefined(call, "sample count and texture type disagree"));
        }
        // MTLTextureSwizzle red 2, green 3, blue 4, alpha 5: the identity. A descriptor swizzle changes what every read
        // of the texture returns; nothing here applies one yet.
        let sw = &d.swizzle;
        if (sw.red, sw.green, sw.blue, sw.alpha) != (2, 3, 4, 5) {
            return Err(MemError::unimplemented(format!(
                "{call}: texture swizzle ({}, {}, {}, {}) is not applied by the host yet",
                sw.red, sw.green, sw.blue, sw.alpha
            )));
        }
        let format = u32::try_from(d.pixel_format).map_err(|_| MemError::undefined(call, "pixel format"))?;
        let dimension = if kind.is_3d() { Dimension::ThreeD } else { Dimension::Planar };
        let resolution = match handling::resolve_texture(format, self.features, dimension) {
            Ok(r) => r,
            Err(Unavailable::NotOnDevice(why)) => return Err(MemError::undefined(call, why.to_string())),
            Err(Unavailable::Refused(r)) => return Err(MemError::refused(r)),
        };
        if resolution == Resolution::StencilAspect {
            return Err(MemError::undefined(call, "X32_Stencil8 is a view format of Depth32Float_Stencil8"));
        }
        let unknown = d.usage == usage::UNKNOWN;
        let wants = |bit: u64| unknown || d.usage & bit != 0;
        let metal_usage = Usage {
            shader_read: d.usage & usage::SHADER_READ != 0,
            shader_write: d.usage & usage::SHADER_WRITE != 0,
            render_target: d.usage & usage::RENDER_TARGET != 0 || storage == StorageMode::Memoryless,
            sample_count: geometry.sample_count,
        };
        handling::check_usage(resolution, metal_usage, self.features).map_err(MemError::refused)?;
        let stored_format = resolution.storage();
        let caps = stored_format.caps();
        let can_render = self.features.allows(caps.render)
            && !matches!(
                resolution,
                Resolution::Emulated(Emulation {
                    write: handling::Write::NotWritable | handling::Write::Refused(_),
                    ..
                })
            );
        let can_store = self.features.allows(caps.storage_wo) && !multisampled;
        let mut gpu_usage = TextureUsage::TEXTURE_BINDING;
        if storage != StorageMode::Memoryless {
            gpu_usage |= TextureUsage::COPY_SRC | TextureUsage::COPY_DST;
        }
        // Metal blits write every texture that is not memoryless and generate the mipmaps of any mipmapped
        // color-renderable texture, whatever its usage (`generateMipmaps(for:)`); WebGPU writes a depth aspect that is
        // not a copy destination, and makes mipmap levels, only through render passes.
        let blit_renders = storage != StorageMode::Memoryless
            && (caps.aspects.iter().any(|a| !a.copy_dst) || (geometry.mip_levels > 1 && caps.aspects.is_empty()));
        if (wants(usage::RENDER_TARGET) || storage == StorageMode::Memoryless || blit_renders) && can_render {
            gpu_usage |= TextureUsage::RENDER_ATTACHMENT;
        }
        if wants(usage::SHADER_WRITE) && can_store {
            gpu_usage |= TextureUsage::STORAGE_BINDING;
        }
        if multisampled {
            gpu_usage |= TextureUsage::RENDER_ATTACHMENT;
        }
        let mut desc = GpuTextureDescriptor::new(
            extent(geometry.width, geometry.height, geometry.layers()),
            idl_format(stored_format),
            gpu_usage,
        );
        desc.label = label(task, "texture", handle);
        desc.mip_level_count = geometry.mip_levels;
        desc.sample_count = geometry.sample_count;
        desc.dimension = if kind.is_3d() { TextureDimension::D3 } else { TextureDimension::D2 };
        // A view in the sibling format needs the sibling on this host; a view the host cannot give is refused when it
        // is created (format::reinterpret).
        if wants(usage::PIXEL_FORMAT_VIEW)
            && let Some(sibling) = srgb_sibling(stored_format)
            && self.features.supports(sibling)
        {
            desc.view_formats.push(idl_format(sibling));
        }
        if !self.features.has(Feature::CoreFeaturesAndLimits) {
            // A device without core-features-and-limits binds a texture only in the view dimension fixed at creation
            // (WebGPU section 6.1.3).
            desc.texture_binding_view_dimension = Some(match kind {
                TextureType::Cube => TextureViewDimension::Cube,
                TextureType::CubeArray => TextureViewDimension::CubeArray,
                TextureType::D3 => TextureViewDimension::D3,
                TextureType::D1Array | TextureType::D2Array => TextureViewDimension::D2Array,
                _ => TextureViewDimension::D2,
            });
        }
        let stored = g
            .call("GPUDevice.createTexture", &desc.label, |d| d.create_texture(&desc))
            .map_err(gpu_err("GPUDevice.createTexture"))?;
        let blocks = Blocks::of(resolution);
        let mut twins = Vec::new();
        if storage != StorageMode::Memoryless && needs_twin(resolution) && !multisampled && surface.is_none() {
            let b = blocks.expect("formats with twins have blocks");
            let tf = twin_format(b.bytes).ok_or_else(|| MemError::unimplemented(format!("{}-byte blocks", b.bytes)))?;
            for level in 0..geometry.mip_levels {
                let (w, h, depth) = geometry.level_extent(level);
                let (bx, by) = b.count(w, h);
                let layers = if kind.is_3d() { depth } else { geometry.slices() };
                let mut td = GpuTextureDescriptor::new(
                    extent(bx, by, layers),
                    idl_format(tf),
                    TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
                );
                td.label = format!("{} bytes of level {level}", desc.label);
                twins.push(
                    g.call("GPUDevice.createTexture", &td.label, |d| d.create_texture(&td))
                        .map_err(gpu_err("GPUDevice.createTexture"))?,
                );
            }
        }
        let subs = (geometry.mip_levels * if kind.is_3d() { 1 } else { geometry.slices() }) as usize;
        let state = State {
            stored: vec![true; subs],
            twin: vec![!twins.is_empty(); subs],
            purgeable: Purgeable::NonVolatile,
            surface_version: 0,
            surface_written: false,
        };
        let label = desc.label.clone();
        Ok(Rc::new(Texture {
            format,
            resolution,
            geometry,
            storage,
            usage: d.usage,
            blocks,
            surface,
            label,
            stored,
            twins,
            state: RefCell::new(state),
        }))
    }

    /// The layout of a `replaceRegion`/`getBytes` region, with the rules both calls share.
    fn cpu_region(
        &self,
        call: &'static str,
        t: &Texture<A>,
        a: &RegionArgs,
    ) -> Result<(RegionLayout, Blocks), MemError> {
        if t.surface.is_some() {
            return Err(MemError::unimplemented(format!(
                "{call} of an IOSurface texture: the CPU's bytes are the surface's, which the kernel's IOSurface \
                 implementation serves"
            )));
        }
        match t.storage {
            StorageMode::Shared => {}
            StorageMode::Private => {
                return Err(MemError::undefined(
                    call,
                    "CPU access for textures with MTLResourceStorageModePrivate storage mode is disallowed (debug \
                     layer, storage.private-texture-replace-region)",
                ));
            }
            StorageMode::Memoryless => return Err(MemError::undefined(call, "a memoryless texture has no contents")),
        }
        if t.geometry.sample_count > 1 {
            return Err(MemError::unimplemented(format!(
                "{call} of a multisampled texture: WebGPU copies multisampled textures only to textures"
            )));
        }
        let blocks = t.blocks.ok_or_else(|| {
            MemError::undefined(call, "a combined depth-stencil texture has no single texel size for CPU access")
        })?;
        Ok((RegionLayout::new(call, &t.geometry, blocks, a)?, blocks))
    }

    /// Where a region of `(level, slice)` lies in the stored texture (texel coordinates; compressed regions rounded
    /// out to whole blocks, as WebGPU copies them) and in the twin (block coordinates).
    pub(crate) fn placements(
        &self,
        t: &Texture<A>,
        l: &RegionLayout,
        level: u32,
        slice: u32,
    ) -> (Placement, Placement) {
        let b = t.blocks.expect("regions have blocks");
        let (ox, oy, oz) = l.origin;
        let z = if t.geometry.kind.is_3d() { oz } else { slice };
        let twin = Placement {
            level: 0,
            origin: (ox / b.width, oy / b.height, z),
            extent: (l.blocks.0, l.blocks.1, l.images),
            row_bytes: l.row_bytes,
            rows: l.blocks.1,
        };
        let stored = match t.resolution {
            Resolution::Native(f) if f.is_compressed() => Placement {
                level,
                origin: (ox, oy, z),
                extent: (l.blocks.0 * b.width, l.blocks.1 * b.height, l.images),
                row_bytes: l.row_bytes,
                rows: l.blocks.1,
            },
            r => {
                // Uncompressed storage: the Metal format's own texels, or the emulation's stored texels (decoded
                // formats store one texel per Metal texel).
                let footprint = match r {
                    Resolution::Emulated(e) => u64::from(e.storage.caps().footprint),
                    _ => u64::from(b.bytes),
                };
                Placement {
                    level,
                    origin: (ox, oy, z),
                    extent: (l.size.0, l.size.1, l.images),
                    row_bytes: u64::from(l.size.0) * footprint,
                    rows: l.size.1,
                }
            }
        };
        (stored, twin)
    }

    fn write_texture(
        &self,
        g: &impl Calls<A>,
        texture: &A::Texture,
        p: &Placement,
        aspect: TextureAspect,
        data: &[u8],
    ) -> Result<(), MemError> {
        let mut dst = TexelCopyTextureInfo::<A>::new(texture);
        dst.mip_level = p.level;
        dst.origin = origin(p.origin.0, p.origin.1, p.origin.2);
        dst.aspect = aspect;
        let layout = TexelCopyBufferLayout {
            offset: Size64::new_const(0),
            bytes_per_row: Some(u32::try_from(p.row_bytes).map_err(|_| MemError::host("row too long"))?),
            rows_per_image: Some(p.rows),
        };
        g.call("GPUQueue.writeTexture", "", |_| {
            g.queue().write_texture(&dst, data, &layout, &extent(p.extent.0, p.extent.1, p.extent.2))
        })
        .map_err(gpu_err("GPUQueue.writeTexture"))
    }

    /// The stored texture's bytes of a region from Metal's tight bytes: the same bytes, red and blue exchanged
    /// (BGR10A2), or decoded.
    fn stored_bytes(&self, t: &Texture<A>, l: &RegionLayout, tight: &[u8], level: u32) -> Result<Vec<u8>, MemError> {
        match t.resolution {
            Resolution::Native(_) | Resolution::StencilAspect => Ok(tight.to_vec()),
            Resolution::Emulated(e) => match e.upload {
                Upload::Bytes => Ok(tight.to_vec()),
                Upload::SwapRedBlue10 => {
                    let mut v = tight.to_vec();
                    packed::swap_red_blue_10_in_place(&mut v);
                    Ok(v)
                }
                Upload::Decode(_) if l.pvrtc => {
                    let (w, h, _) = t.geometry.level_extent(level);
                    storage::to_storage(&e, w, h, tight, 0)
                        .map_err(|err| MemError::invalid(format!("PVRTC level does not decode: {err:?}")))
                }
                Upload::Decode(_) => {
                    let image = (l.row_bytes * u64::from(l.blocks.1)) as usize;
                    let mut out = Vec::new();
                    for i in 0..l.images as usize {
                        let part = &tight[i * image..(i + 1) * image];
                        out.extend(
                            storage::to_storage(&e, l.size.0, l.size.1, part, l.row_bytes as usize)
                                .map_err(|err| MemError::invalid(format!("blocks do not decode: {err:?}")))?,
                        );
                    }
                    Ok(out)
                }
            },
        }
    }

    /// Is a transfer between bytes and the stored texture of `t` exact on this host? Bit-exact formats always are.
    /// An upload of compressed blocks is exact in what it is used for: the decoded values, which WebGPU preserves
    /// (every value a compressed block decodes to is finite and normal), while the bits stay in the twin. Every other
    /// transfer needs the format's texel copies proven on this host ([`super::proof`]).
    pub(crate) fn check_stored_transfer(
        &self,
        t: &Texture<A>,
        call: &'static str,
        upload: bool,
    ) -> Result<(), MemError> {
        let f = t.resolution.storage();
        if f.texel_copy_exact(None) || (upload && f.is_compressed()) {
            return Ok(());
        }
        if !upload && depth_float_only(f) {
            // A depth32float texture holds only what a depth output (clamped to [0, 1]) or the checked depth write
            // (`crate::blit::kernels::encode_depth_write`: +0 and normal values of [0, 1]) put there: finite values
            // whose copies keep them (WebGPU section 11.2: "valid, finite, non-subnormal numeric values in the source
            // have the same numeric value in the destination").
            return Ok(());
        }
        match self.proof(f) {
            Some(Ok(())) => Ok(()),
            Some(Err(why)) => {
                Err(MemError::host(format!("{call}: texel copies of {} are not exact on this host: {why}", f.name())))
            }
            None => Err(MemError::unimplemented(format!(
                "{call}: texel copies of {} must be proven exact on this host first (Memory::prove_texel_copies)",
                f.name()
            ))),
        }
    }

    /// The format whose texel copies a transfer of `t` needs proven before it can run, if any.
    pub fn texture_needs_proof(&self, t: &Texture<A>) -> Option<WebGpuFormat> {
        let f = t.resolution.storage();
        (!f.texel_copy_exact(None) && !f.is_compressed() && !depth_float_only(f) && self.proof(f).is_none())
            .then_some(f)
    }

    /// `replaceRegion:mipmapLevel:withBytes:bytesPerRow:` (ReplaceRegion).
    pub fn replace_region(
        &self,
        g: &impl Calls<A>,
        e: &dyn Embedder,
        task: TaskId,
        t: &Texture<A>,
        r: &ReplaceRegion,
    ) -> Result<(), MemError> {
        let a = RegionArgs {
            origin: (r.region.origin.x, r.region.origin.y, r.region.origin.z),
            size: (r.region.size.width, r.region.size.height, r.region.size.depth),
            level: r.level,
            slice: 0,
            bytes_per_row: r.bytes_per_row,
            bytes_per_image: 0,
        };
        let call = "-[MTLTexture replaceRegion:mipmapLevel:withBytes:bytesPerRow:]";
        self.replace(g, call, e, t, &a, (task, r.pixel_bytes.0))
    }

    /// `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:` (ReplaceRegionSlice).
    pub fn replace_region_slice(
        &self,
        g: &impl Calls<A>,
        e: &dyn Embedder,
        task: TaskId,
        t: &Texture<A>,
        r: &ReplaceRegionSlice,
    ) -> Result<(), MemError> {
        let a = RegionArgs {
            origin: (r.region.origin.x, r.region.origin.y, r.region.origin.z),
            size: (r.region.size.width, r.region.size.height, r.region.size.depth),
            level: r.level,
            slice: r.slice,
            bytes_per_row: r.bytes_per_row,
            bytes_per_image: r.bytes_per_image,
        };
        let call = "-[MTLTexture replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:]";
        self.replace(g, call, e, t, &a, (task, r.pixel_bytes.0))
    }

    /// Metal applies `replaceRegion` immediately, even before uncommitted command buffers that use the texture run;
    /// here the writes are queue operations, ordered before every later submission (plan section 6.2).
    fn replace(
        &self,
        g: &impl Calls<A>,
        call: &'static str,
        e: &dyn Embedder,
        t: &Texture<A>,
        a: &RegionArgs,
        (task, pixel_bytes): (TaskId, u64),
    ) -> Result<(), MemError> {
        let (l, _) = self.cpu_region(call, t, a)?;
        let mut metal = vec![0u8; l.span as usize];
        e.read_guest(task, pixel_bytes, &mut metal)?;
        self.upload_region(g, call, t, &l, (a.level as u32, a.slice as u32), &l.pack(&metal))
    }

    /// Write a region of a depth32float texture from bytes the CPU holds, through a render pass (`encode_depth_write`):
    /// WebGPU copies cannot write its depth aspect. Values the pass would not store unchanged are refused.
    fn upload_depth(
        &self,
        g: &impl Calls<A>,
        call: &'static str,
        t: &Texture<A>,
        l: &RegionLayout,
        (level, slice): (u32, u32),
        tight: &[u8],
    ) -> Result<(), MemError> {
        let bits = depth_bits(call, tight)?;
        let encoder = g.call("GPUDevice.createCommandEncoder", call, |d| {
            d.create_command_encoder(&CommandEncoderDescriptor::new())
        });
        let w = DepthWrite {
            texture: &t.stored,
            level,
            layer: slice,
            origin: (l.origin.0, l.origin.1),
            size: (l.size.0, l.size.1),
            bits: &bits,
        };
        encode_depth_write(self, g, &encoder, &w)?;
        let cb = g.call("GPUCommandEncoder.finish", call, |_| encoder.finish(&CommandBufferDescriptor::new()));
        g.call("GPUQueue.submit", call, |_| g.queue().submit(&[&cb]));
        Ok(())
    }

    /// Write a region's tight bytes into every current copy of the subresource.
    pub(crate) fn upload_region(
        &self,
        g: &impl Calls<A>,
        call: &'static str,
        t: &Texture<A>,
        l: &RegionLayout,
        (level, slice): (u32, u32),
        tight: &[u8],
    ) -> Result<(), MemError> {
        let layer = if t.geometry.kind.is_3d() { 0 } else { slice };
        let (stored_now, twin_now) = (t.stored_current(level, layer), t.twin_current(level, layer));
        let (sp, tp) = self.placements(t, l, level, slice);
        if stored_now {
            if depth_float_only(t.resolution.storage()) {
                return self.upload_depth(g, call, t, l, (level, slice), tight);
            }
            self.check_stored_transfer(t, call, true)?;
            let data = self.stored_bytes(t, l, tight, level)?;
            self.write_texture(g, &t.stored, &sp, TextureAspect::All, &data)?;
        }
        if twin_now {
            match t.blocks.and_then(|b| b.pvrtc) {
                Some(bpp) if l.pvrtc => {
                    let (w, h, _) = t.geometry.level_extent(level);
                    let rows = pvrtc_row_linear(bpp, w, h, tight)?;
                    self.write_texture(g, &t.twins[level as usize], &tp, TextureAspect::All, &rows)?;
                }
                _ => self.write_texture(g, &t.twins[level as usize], &tp, TextureAspect::All, tight)?,
            }
        }
        Ok(())
    }

    /// `getBytes:bytesPerRow:fromRegion:mipmapLevel:` (GetBytes): encode and submit the read of the region. The read
    /// is ordered after every command buffer already submitted, so it sees their writes.
    pub fn get_bytes(&self, g: &impl Calls<A>, t: &Rc<Texture<A>>, r: &GetBytes) -> Result<TextureRead<A>, MemError> {
        let a = RegionArgs {
            origin: (r.region.origin.x, r.region.origin.y, r.region.origin.z),
            size: (r.region.size.width, r.region.size.height, r.region.size.depth),
            level: r.level,
            slice: 0,
            bytes_per_row: r.bytes_per_row,
            bytes_per_image: 0,
        };
        self.read_region(g, "-[MTLTexture getBytes:bytesPerRow:fromRegion:mipmapLevel:]", t, &a)
    }

    /// `getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:` (GetBytesSlice).
    pub fn get_bytes_slice(
        &self,
        g: &impl Calls<A>,
        t: &Rc<Texture<A>>,
        r: &GetBytesSlice,
    ) -> Result<TextureRead<A>, MemError> {
        let a = RegionArgs {
            origin: (r.region.origin.x, r.region.origin.y, r.region.origin.z),
            size: (r.region.size.width, r.region.size.height, r.region.size.depth),
            level: r.level,
            slice: r.slice,
            bytes_per_row: r.bytes_per_row,
            bytes_per_image: r.bytes_per_image,
        };
        self.read_region(g, "-[MTLTexture getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:]", t, &a)
    }

    fn read_region(
        &self,
        g: &impl Calls<A>,
        call: &'static str,
        t: &Rc<Texture<A>>,
        a: &RegionArgs,
    ) -> Result<TextureRead<A>, MemError> {
        let (l, _) = self.cpu_region(call, t, a)?;
        let encoder = g.call("GPUDevice.createCommandEncoder", call, |d| {
            d.create_command_encoder(&CommandEncoderDescriptor::new())
        });
        let read = self.encode_read(g, call, &encoder, t, &l, (a.level as u32, a.slice as u32))?;
        let cb = g.call("GPUCommandEncoder.finish", call, |_| encoder.finish(&CommandBufferDescriptor::new()));
        g.call("GPUQueue.submit", call, |_| g.queue().submit(&[&cb]));
        Ok(read)
    }

    /// Encode the copy of a region into a staging buffer: from the twin when it is current, else from the stored
    /// texture.
    /// Which copy a read of a region comes from: the twin when it is current, else the stored texture, which must
    /// then be exact to read.
    fn read_source<'t>(
        &self,
        call: &'static str,
        t: &'t Texture<A>,
        l: &RegionLayout,
        level: u32,
        slice: u32,
    ) -> Result<(&'t A::Texture, Placement, Convert), MemError> {
        let layer = if t.geometry.kind.is_3d() { 0 } else { slice };
        let (sp, tp) = self.placements(t, l, level, slice);
        if t.twin_current(level, layer) {
            let convert = match (l.pvrtc, t.blocks.and_then(|b| b.pvrtc)) {
                (true, Some(bpp)) => {
                    let (width, height, _) = t.geometry.level_extent(level);
                    Convert::PvrtcToMorton { bpp, width, height }
                }
                _ => Convert::None,
            };
            return Ok((&t.twins[level as usize], tp, convert));
        }
        let convert = match t.resolution {
            Resolution::Emulated(e) => match e.upload {
                Upload::Bytes => Convert::None,
                Upload::SwapRedBlue10 => Convert::SwapRedBlue10,
                Upload::Decode(_) => unreachable!("decoded textures keep their blocks current in the twin"),
            },
            _ => Convert::None,
        };
        self.check_stored_transfer(t, call, false)?;
        Ok((&t.stored, sp, convert))
    }

    pub(crate) fn encode_read(
        &self,
        g: &impl Calls<A>,
        call: &'static str,
        encoder: &A::CommandEncoder,
        t: &Texture<A>,
        l: &RegionLayout,
        (level, slice): (u32, u32),
    ) -> Result<TextureRead<A>, MemError> {
        let (texture, p, convert) = self.read_source(call, t, l, level, slice)?;
        let pitch = StagingPitch::new(p.row_bytes, u64::from(p.rows), u64::from(p.extent.2));
        let staging = self.staging.take(g, pitch.size())?;
        let mut src = TexelCopyTextureInfo::<A>::new(texture);
        src.mip_level = p.level;
        src.origin = origin(p.origin.0, p.origin.1, p.origin.2);
        let mut dst = TexelCopyBufferInfo::<A>::new(&staging);
        dst.bytes_per_row = Some(pitch.bytes_per_row as u32);
        dst.rows_per_image = Some(p.rows);
        g.call("GPUCommandEncoder.copyTextureToBuffer", "", |_| {
            encoder.copy_texture_to_buffer(&src, &dst, &extent(p.extent.0, p.extent.1, p.extent.2))
        })
        .map_err(gpu_err("GPUCommandEncoder.copyTextureToBuffer"))?;
        Ok(TextureRead { staging, pitch, convert, layout: *l, refresh: None })
    }

    /// GPU work encoded next writes the stored texture's levels `levels` and layers `layers` (array slices, or the
    /// whole of a 3D level): the twin of those subresources becomes stale.
    pub fn note_texture_write(&self, t: &Texture<A>, levels: std::ops::Range<u32>, layers: std::ops::Range<u32>) {
        let mut st = t.state.borrow_mut();
        let per_level = t.per_level();
        for level in levels {
            for layer in layers.clone() {
                let i = (level * per_level + if t.geometry.kind.is_3d() { 0 } else { layer }) as usize;
                st.stored[i] = true;
                if !t.twins.is_empty() {
                    st.twin[i] = false;
                }
            }
        }
        if t.surface.is_some() {
            st.surface_written = true;
        }
    }

    /// `setPurgeableState:` of a texture that is not in a heap. Emptying zero-fills the contents (plan section 6.2):
    /// a write of zero blocks to every subresource of every copy.
    pub fn set_texture_purgeable(&self, g: &impl Calls<A>, t: &Texture<A>, state: u64) -> Result<u64, MemError> {
        let tr = t.state.borrow_mut().purgeable.set(state, "-[MTLResource setPurgeableState:]")?;
        if tr.discard && t.storage != StorageMode::Memoryless {
            self.zero_texture(g, t)?;
        }
        Ok(tr.previous)
    }

    fn zero_texture(&self, g: &impl Calls<A>, t: &Texture<A>) -> Result<(), MemError> {
        let Some(b) = t.blocks else {
            return Err(MemError::unimplemented("discarding the contents of a combined depth-stencil texture"));
        };
        let caps = t.resolution.storage().caps();
        if caps.aspects.len() == 1 && !caps.aspects[0].copy_dst {
            return Err(MemError::unimplemented(format!(
                "discarding the contents of a {} texture needs a render pass",
                t.resolution.storage().name()
            )));
        }
        for level in 0..t.geometry.mip_levels {
            let l = full_level(t, b, level)?;
            let (mut sp, mut tp) = self.placements(t, &l, level, 0);
            if !t.geometry.kind.is_3d() {
                sp.extent.2 = t.geometry.slices();
                tp.extent.2 = t.geometry.slices();
            }
            let bytes = |p: &Placement| vec![0u8; (p.row_bytes * u64::from(p.rows) * u64::from(p.extent.2)) as usize];
            self.write_texture(g, &t.stored, &sp, TextureAspect::All, &bytes(&sp))?;
            if let Some(twin) = t.twins.get(level as usize) {
                self.write_texture(g, twin, &tp, TextureAspect::All, &bytes(&tp))?;
            }
        }
        let mut st = t.state.borrow_mut();
        st.stored.fill(true);
        let has_twin = !t.twins.is_empty();
        st.twin.fill(has_twin);
        Ok(())
    }

    /// A buffer the executor uses as a GPU-side staging area for re-pitched copies; the blit passes
    /// ([`crate::blit::kernels`]) read and write it as a storage buffer.
    fn copy_staging(&self, g: &impl Calls<A>, size: u64) -> Result<A::Buffer, MemError> {
        let d = BufferDescriptor::new(
            size64(size.max(4), "createBuffer")?,
            BufferUsage::COPY_SRC | BufferUsage::COPY_DST | BufferUsage::STORAGE,
        );
        g.call("GPUDevice.createBuffer", "webmetal copy staging", |dev| dev.create_buffer(&d))
            .map_err(gpu_err("GPUDevice.createBuffer"))
    }

    /// Encode a blit of a region from a buffer into a texture (`copyFromBuffer:...toTexture:`): `l` lays the bytes out
    /// at `offset` of `src` as Metal does. The bytes land in the twin when the texture keeps one (the stored texture is
    /// then stale until [`Memory::prepare_texture`]), else in the stored texture. Rows whose Metal stride is not a
    /// multiple of 256 are re-pitched through a staging buffer: with buffer copies when every row starts on a 4-byte
    /// boundary, else with the GPU's byte moves (`encode_rows`). A depth aspect that WebGPU copies cannot write
    /// (`depth32float`) and the aspects of combined depth-stencil formats are the blit service's
    /// ([`crate::blit::depth`]).
    pub fn encode_buffer_to_texture(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        (src, offset): (&Buffer<A>, u64),
        t: &Texture<A>,
        r: &TextureRegion,
    ) -> Result<(), MemError> {
        let (l, level, slice) = (&r.layout, r.level, r.slice);
        const CALL: &str = "-[MTLBlitCommandEncoder copyFromBuffer:...toTexture:...]";
        if t.blocks.is_none() {
            return Err(MemError::unimplemented(format!(
                "{CALL}: per-aspect blits of a combined depth-stencil texture"
            )));
        }
        if t.storage == StorageMode::Memoryless {
            return Err(MemError::undefined(CALL, "a memoryless texture has no contents"));
        }
        let layer = if t.geometry.kind.is_3d() { 0 } else { slice };
        let (sp, tp) = self.placements(t, l, level, slice);
        let (texture, p) = if t.has_twin() {
            (&t.twins[level as usize], tp)
        } else {
            match t.resolution {
                Resolution::Emulated(e) if e.upload == Upload::SwapRedBlue10 => {
                    return Err(MemError::unimplemented(format!("{CALL}: BGR10A2 blits need a channel swap pass")));
                }
                _ => {}
            }
            let caps = t.resolution.storage().caps();
            if !caps.aspects.is_empty() && caps.aspects.len() == 1 && !caps.aspects[0].copy_dst {
                return Err(MemError::unimplemented(format!(
                    "{CALL}: WebGPU cannot copy into {} textures",
                    t.resolution.storage().name()
                )));
            }
            (&t.stored, sp)
        };
        self.encode_bytes_to_texture(g, encoder, (src.gpu(), offset), l, texture, &p)?;
        let mut st = t.state.borrow_mut();
        let i = t.sub(level, layer);
        if t.has_twin() {
            st.twin[i] = true;
            st.stored[i] = false;
        } else {
            st.stored[i] = true;
        }
        if t.surface.is_some() {
            st.surface_written = true;
        }
        Ok(())
    }

    fn encode_bytes_to_texture(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        (src, offset): (&A::Buffer, u64),
        l: &RegionLayout,
        texture: &A::Texture,
        p: &Placement,
    ) -> Result<(), MemError> {
        let mut dst = TexelCopyTextureInfo::<A>::new(texture);
        dst.mip_level = p.level;
        dst.origin = origin(p.origin.0, p.origin.1, p.origin.2);
        let size = extent(p.extent.0, p.extent.1, p.extent.2);
        let rows = u64::from(l.blocks.1);
        let direct = l.bytes_per_row.is_multiple_of(256)
            && offset.is_multiple_of(copy_offset_alignment(texture.format()))
            && (l.images == 1 || l.bytes_per_image.is_multiple_of(l.bytes_per_row));
        if direct {
            let mut s = TexelCopyBufferInfo::<A>::new(src);
            s.offset = size64(offset, "copyBufferToTexture")?;
            s.bytes_per_row = Some(l.bytes_per_row as u32);
            s.rows_per_image =
                Some(if l.images == 1 { l.blocks.1 } else { (l.bytes_per_image / l.bytes_per_row) as u32 });
            return g
                .call("GPUCommandEncoder.copyBufferToTexture", "", |_| encoder.copy_buffer_to_texture(&s, &dst, &size))
                .map_err(gpu_err("GPUCommandEncoder.copyBufferToTexture"));
        }
        let pitch = StagingPitch::new(l.row_bytes, rows, u64::from(l.images));
        let staging = self.copy_staging(g, pitch.size())?;
        let chunk = l.row_bytes.next_multiple_of(4);
        let starts = || {
            (0..u64::from(l.images))
                .flat_map(move |i| (0..rows).map(move |r| (i, r, offset + i * l.bytes_per_image + r * l.bytes_per_row)))
        };
        if starts().all(|(_, _, from)| from.is_multiple_of(4) && from + chunk <= src.size()) {
            // Whole words: one buffer copy per row.
            for (i, r, from) in starts() {
                let to = (i * rows + r) * pitch.bytes_per_row;
                let (a, b, n) = (
                    size64(from, "copyBufferToBuffer")?,
                    size64(to, "copyBufferToBuffer")?,
                    size64(chunk, "copyBufferToBuffer")?,
                );
                g.call("GPUCommandEncoder.copyBufferToBuffer", "", |_| {
                    encoder.copy_buffer_to_buffer_with_offsets(src, a, &staging, b, Some(n))
                });
            }
        } else {
            // Rows that start inside a word: moved byte by byte on the GPU.
            let rows_layout = Rows {
                row_bytes: l.row_bytes,
                rows,
                images: u64::from(l.images),
                src_row_pitch: l.bytes_per_row,
                src_image_pitch: l.bytes_per_image,
                dst_row_pitch: pitch.bytes_per_row,
                dst_image_pitch: pitch.bytes_per_row * rows,
            };
            encode_rows(self, g, encoder, Source::Buffer(src, offset), (&staging, 0), rows_layout)?;
        }
        let mut s = TexelCopyBufferInfo::<A>::new(&staging);
        s.bytes_per_row = Some(pitch.bytes_per_row as u32);
        s.rows_per_image = Some(l.blocks.1);
        g.call("GPUCommandEncoder.copyBufferToTexture", "", |_| encoder.copy_buffer_to_texture(&s, &dst, &size))
            .map_err(gpu_err("GPUCommandEncoder.copyBufferToTexture"))
    }

    /// Encode a blit of a region from a texture into a buffer (`copyFromTexture:...toBuffer:`), laid out at `offset`
    /// of `dst` as `l` describes; the buffer's readback window grows to cover it. Rows that are not whole 4-byte words
    /// are written with the GPU's byte moves (`encode_rows`), which leave the bytes beside them as they are; a
    /// BGR10A2 texture read from its stored `rgb10a2unorm` texels has its red and blue fields exchanged back first.
    pub fn encode_texture_to_buffer(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        t: &Texture<A>,
        r: &TextureRegion,
        (dst, offset): (&Buffer<A>, u64),
    ) -> Result<(), MemError> {
        let (l, level, slice) = (&r.layout, r.level, r.slice);
        const CALL: &str = "-[MTLBlitCommandEncoder copyFromTexture:...toBuffer:...]";
        if t.blocks.is_none() {
            return Err(MemError::unimplemented(format!(
                "{CALL}: per-aspect blits of a combined depth-stencil texture"
            )));
        }
        if t.storage == StorageMode::Memoryless {
            return Err(MemError::undefined(CALL, "a memoryless texture has no contents"));
        }
        let (texture, p, convert) = self.read_source(CALL, t, l, level, slice)?;
        if let Convert::PvrtcToMorton { .. } = convert {
            return Err(MemError::undefined(CALL, "PVRTC blits to a buffer take MTLBlitOptionRowLinearPVRTC"));
        }
        let rows = u64::from(l.blocks.1);
        let pitch = StagingPitch::new(l.row_bytes, rows, u64::from(l.images));
        let staging = self.copy_staging(g, pitch.size())?;
        let mut src = TexelCopyTextureInfo::<A>::new(texture);
        src.mip_level = p.level;
        src.origin = origin(p.origin.0, p.origin.1, p.origin.2);
        let mut sd = TexelCopyBufferInfo::<A>::new(&staging);
        sd.bytes_per_row = Some(pitch.bytes_per_row as u32);
        sd.rows_per_image = Some(p.rows);
        g.call("GPUCommandEncoder.copyTextureToBuffer", "", |_| {
            encoder.copy_texture_to_buffer(&src, &sd, &extent(p.extent.0, p.extent.1, p.extent.2))
        })
        .map_err(gpu_err("GPUCommandEncoder.copyTextureToBuffer"))?;
        if convert == Convert::SwapRedBlue10 {
            // The stored rgb10a2unorm texels back to Metal's BGR10A2 bytes.
            encode_swap10(self, g, encoder, &staging, 0, pitch.size() / 4)?;
        }
        self.encode_staged_rows(g, encoder, &staging, (pitch, rows), l, (dst, offset))
    }

    /// Write rows staged at `pitch` into `dst` at `offset` laid out as `l`: one buffer copy per row when every row is
    /// whole words, else the GPU's byte moves, which leave the bytes beside the rows as they are.
    pub(crate) fn encode_staged_rows(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        staging: &A::Buffer,
        (pitch, rows): (StagingPitch, u64),
        l: &RegionLayout,
        (dst, offset): (&Buffer<A>, u64),
    ) -> Result<(), MemError> {
        let starts = || {
            (0..u64::from(l.images))
                .flat_map(move |i| (0..rows).map(move |r| (i, r, offset + i * l.bytes_per_image + r * l.bytes_per_row)))
        };
        if l.row_bytes.is_multiple_of(4) && starts().all(|(_, _, to)| to.is_multiple_of(4)) {
            for (i, r, to) in starts() {
                let (a, b, n) = (
                    size64((i * rows + r) * pitch.bytes_per_row, "copyBufferToBuffer")?,
                    size64(to, "copyBufferToBuffer")?,
                    size64(l.row_bytes, "copyBufferToBuffer")?,
                );
                g.call("GPUCommandEncoder.copyBufferToBuffer", "", |_| {
                    encoder.copy_buffer_to_buffer_with_offsets(staging, a, dst.gpu(), b, Some(n))
                });
            }
        } else {
            let rows_layout = Rows {
                row_bytes: l.row_bytes,
                rows,
                images: u64::from(l.images),
                src_row_pitch: pitch.bytes_per_row,
                src_image_pitch: pitch.bytes_per_row * rows,
                dst_row_pitch: l.bytes_per_row,
                dst_image_pitch: l.bytes_per_image,
            };
            encode_rows(self, g, encoder, Source::Buffer(staging, 0), (dst.gpu(), offset), rows_layout)?;
        }
        self.note_buffer_write(dst, Some((offset, offset + l.span)));
        Ok(())
    }

    /// Bring the stored texture up to date with the twin where a blit left it stale, before the next GPU use of the
    /// texture. Formats WebGPU holds natively are copied on the GPU (encoded into `encoder`); decoded formats need the
    /// blocks on the CPU, which [`Memory::refresh_decoded`] does: this returns [`Prepare::Decode`] for them.
    pub fn prepare_texture(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        t: &Texture<A>,
    ) -> Result<Prepare, MemError> {
        let stale: Vec<usize> =
            t.state.borrow().stored.iter().enumerate().filter(|(_, c)| !**c).map(|(i, _)| i).collect();
        if stale.is_empty() {
            return Ok(Prepare::Ready);
        }
        if let Resolution::Emulated(Emulation { upload: Upload::Decode(_), .. }) = t.resolution {
            return Ok(Prepare::Decode);
        }
        self.check_stored_transfer(t, "texture use after a blit", true)?;
        let b = t.blocks.expect("twins have blocks");
        for i in stale {
            let level = i as u32 / t.per_level();
            let layer = i as u32 % t.per_level();
            let (w, h, d) = t.geometry.level_extent(level);
            let (bx, by) = b.count(w, h);
            let images = if t.geometry.kind.is_3d() { d } else { 1 };
            let pitch = StagingPitch::new(u64::from(bx) * u64::from(b.bytes), u64::from(by), u64::from(images));
            let staging = self.copy_staging(g, pitch.size())?;
            let mut src = TexelCopyTextureInfo::<A>::new(&t.twins[level as usize]);
            src.origin = origin(0, 0, layer);
            let mut sd = TexelCopyBufferInfo::<A>::new(&staging);
            sd.bytes_per_row = Some(pitch.bytes_per_row as u32);
            sd.rows_per_image = Some(by);
            g.call("GPUCommandEncoder.copyTextureToBuffer", "", |_| {
                encoder.copy_texture_to_buffer(&src, &sd, &extent(bx, by, images))
            })
            .map_err(gpu_err("GPUCommandEncoder.copyTextureToBuffer"))?;
            if let Resolution::Emulated(Emulation { upload: Upload::SwapRedBlue10, .. }) = t.resolution {
                // Metal's BGR10A2 bytes to the stored rgb10a2unorm texels.
                encode_swap10(self, g, encoder, &staging, 0, pitch.size() / 4)?;
            }
            let mut dst = TexelCopyTextureInfo::<A>::new(&t.stored);
            dst.mip_level = level;
            dst.origin = origin(0, 0, layer);
            let size = if b.compressed() { extent(bx * b.width, by * b.height, images) } else { extent(w, h, images) };
            g.call("GPUCommandEncoder.copyBufferToTexture", "", |_| encoder.copy_buffer_to_texture(&sd, &dst, &size))
                .map_err(gpu_err("GPUCommandEncoder.copyBufferToTexture"))?;
            t.state.borrow_mut().stored[i] = true;
        }
        Ok(Prepare::Ready)
    }

    /// Encode into `encoder` the read of the twin's blocks of every stale subresource of a decoded texture. Once the
    /// GPU has run it ([`TextureRead::wait`]), [`Memory::finish_refresh`] decodes the blocks into the stored texture.
    pub fn refresh_decoded(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        t: &Rc<Texture<A>>,
    ) -> Result<Vec<TextureRead<A>>, MemError> {
        let b = t.blocks.expect("decoded textures have blocks");
        let stale: Vec<usize> =
            t.state.borrow().stored.iter().enumerate().filter(|(_, c)| !**c).map(|(i, _)| i).collect();
        let mut reads = Vec::new();
        for i in stale {
            let level = i as u32 / t.per_level();
            let layer = i as u32 % t.per_level();
            let (w, h, d) = t.geometry.level_extent(level);
            let args = RegionArgs {
                origin: (0, 0, 0),
                size: (u64::from(w), u64::from(h), u64::from(if t.geometry.kind.is_3d() { d } else { 1 })),
                level: u64::from(level),
                slice: u64::from(layer),
                bytes_per_row: 0,
                bytes_per_image: 0,
            };
            let (bx, _) = b.count(w, h);
            let mut a = args;
            if b.pvrtc.is_none() {
                a.bytes_per_row = u64::from(bx) * u64::from(b.bytes);
                if t.geometry.kind.is_3d() && d > 1 {
                    a.bytes_per_image = a.bytes_per_row * u64::from(b.count(w, h).1);
                }
            }
            let l = RegionLayout::new("refresh", &t.geometry, b, &a)?;
            let mut read = self.encode_read(g, "refresh", encoder, t, &l, (level, layer))?;
            read.refresh = Some((level, layer));
            reads.push(read);
        }
        Ok(reads)
    }

    /// Finish [`Memory::refresh_decoded`] with the blocks read back: decode and upload them.
    pub fn finish_refresh(&self, g: &impl Calls<A>, t: &Texture<A>, blocks: TextureBytes<A>) -> Result<(), MemError> {
        let (level, layer) = blocks.refresh.expect("a refresh read");
        let l = blocks.layout;
        let data = self.stored_bytes(t, &l, &blocks.tight, level)?;
        let (sp, _) = self.placements(t, &l, level, layer);
        self.write_texture(g, &t.stored, &sp, TextureAspect::All, &data)?;
        self.staging.give_back(blocks.staging);
        t.state.borrow_mut().stored[t.sub(level, layer)] = true;
        Ok(())
    }
}

/// The layout of a whole level (all of a 3D level's images), as tight rows.
fn full_level<A: Api>(t: &Texture<A>, b: Blocks, level: u32) -> Result<RegionLayout, MemError> {
    let (w, h, d) = t.geometry.level_extent(level);
    let (bx, by) = b.count(w, h);
    let row = u64::from(bx) * u64::from(b.bytes);
    let images = if t.geometry.kind.is_3d() { d } else { 1 };
    let a = RegionArgs {
        origin: (0, 0, 0),
        size: (u64::from(w), u64::from(h), u64::from(images)),
        level: u64::from(level),
        slice: 0,
        bytes_per_row: if b.pvrtc.is_some() { 0 } else { row },
        bytes_per_image: if b.pvrtc.is_none() && images > 1 { row * u64::from(by) } else { 0 },
    };
    RegionLayout::new("whole level", &t.geometry, b, &a)
}

/// A region of one subresource of a texture, with its bytes laid out as a Metal transfer lays them out.
#[derive(Clone, Copy, Debug)]
pub struct TextureRegion {
    pub level: u32,
    /// The array slice or cube face (`cubeFace + arrayIndex * 6`); 0 for other types.
    pub slice: u32,
    pub layout: RegionLayout,
}

/// What [`Memory::prepare_texture`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prepare {
    Ready,
    /// The stored texture of a decoded format is stale: run [`Memory::refresh_decoded`] first.
    Decode,
}

/// Is `f` a depth format whose depth aspect WebGPU copies cannot write (`depth32float`, WebGPU section 26.1.2)?
pub(crate) fn depth_float_only(f: WebGpuFormat) -> bool {
    matches!(f.caps().aspects, [a] if a.aspect == Aspect::Depth && !a.copy_dst)
}

/// WebGPU's alignment of a buffer offset in a copy with a texture of `format`: the texel block copy footprint, or 4
/// for depth and stencil (WebGPU section 13.3, `tables/limits.toml` question `6.3-copy-alignment`).
fn copy_offset_alignment(format: TextureFormat) -> u64 {
    let f = WebGpuFormat::from_name(format.as_str()).expect("the format tables share the IDL's names");
    let caps = f.caps();
    if caps.aspects.is_empty() { u64::from(caps.footprint) } else { 4 }
}

/// The sRGB sibling of a format, for `viewFormats`.
fn srgb_sibling(f: WebGpuFormat) -> Option<WebGpuFormat> {
    let name = f.name();
    let other = match name.strip_suffix("-srgb") {
        Some(base) => base.to_string(),
        None => format!("{name}-srgb"),
    };
    WebGpuFormat::from_name(&other)
}

/// A texture region copied into a staging buffer, waiting for the GPU.
pub struct TextureRead<A: Api> {
    staging: A::Buffer,
    pitch: StagingPitch,
    convert: Convert,
    layout: RegionLayout,
    refresh: Option<(u32, u32)>,
}

/// The bytes of a texture region, in tight rows.
pub struct TextureBytes<A: Api> {
    pub tight: Vec<u8>,
    pub layout: RegionLayout,
    staging: A::Buffer,
    refresh: Option<(u32, u32)>,
}

impl<A: Api> TextureRead<A> {
    pub fn wait(self) -> GpuFuture<Result<TextureBytes<A>, MemError>> {
        Box::pin(async move {
            let size = size64(self.pitch.size(), "mapAsync")?;
            self.staging
                .map_async(MapMode::READ, Size64::new_const(0), Some(size))
                .await
                .map_err(gpu_err("GPUBuffer.mapAsync"))?;
            let range = self
                .staging
                .get_mapped_range(Size64::new_const(0), Some(size))
                .map_err(gpu_err("GPUBuffer.getMappedRange"))?;
            let mut staged = vec![0u8; self.pitch.size() as usize];
            range.read(0, &mut staged).map_err(gpu_err("ArrayBuffer read"))?;
            self.staging.unmap();
            let mut tight = self.pitch.tighten(&staged);
            match self.convert {
                Convert::None => {}
                Convert::SwapRedBlue10 => packed::swap_red_blue_10_in_place(&mut tight),
                Convert::PvrtcToMorton { bpp, width, height } => {
                    tight = pvrtc::reorder(bpp, width, height, &tight, true)
                        .map_err(|e| MemError::invalid(format!("PVRTC level: {e:?}")))?;
                }
            }
            tight.truncate(self.layout.tight_len() as usize);
            Ok(TextureBytes { tight, layout: self.layout, staging: self.staging, refresh: self.refresh })
        })
    }
}

impl<A: Api> TextureBytes<A> {
    /// Write the region into the guest at `addr` in Metal's layout, leaving the bytes between rows and images as they
    /// are (`getBytes` copies pixel data only).
    pub fn write_guest(self, mem: &Memory<A>, e: &dyn Embedder, task: TaskId, addr: u64) -> Result<(), MemError> {
        let mut held = vec![0u8; self.layout.span as usize];
        e.read_guest(task, addr, &mut held)?;
        let mut new = held.clone();
        self.layout.unpack(&self.tight, &mut new);
        mem.staging.give_back(self.staging);
        e.write_guest_changed(task, addr, &new, &held)?;
        Ok(())
    }

    /// Return the staging buffer without writing anywhere.
    pub fn release(self, mem: &Memory<A>) -> Vec<u8> {
        mem.staging.give_back(self.staging);
        self.tight
    }
}

/// Row-linear words of a PVRTC level from Morton-ordered ones ([`pvrtc::reorder`]).
pub fn pvrtc_row_linear(bpp: pvrtc::Bpp, width: u32, height: u32, morton: &[u8]) -> Result<Vec<u8>, MemError> {
    pvrtc::reorder(bpp, width, height, morton, false).map_err(|e| MemError::invalid(format!("PVRTC level: {e:?}")))
}
