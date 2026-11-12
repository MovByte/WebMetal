//! IDL dictionaries as webgpu.h structures, with the content-timeline checks the specification makes on them.
//!
//! An optional IDL member that is not present becomes the header's "undefined" sentinel for that field
//! (webgpu-headers, "Sentinel Values"). Some sentinels are also values of the member's IDL type (`0xFFFFFFFF` is
//! `WGPU_COPY_STRIDE_UNDEFINED` and a legal `GPUSize32`): the header cannot carry such a value, and the conversion
//! reports it as unsupported instead of letting it mean "not present".

use std::ptr::null_mut;

use super::conv::{self, Arena, c_named, sv, to_c, wbool};
use super::objects::*;
use super::sys;
use super::{DeviceShared, Native};
use crate::gpu::api::{Buffer, Exception, ExceptionKind, Texture};
use crate::gpu::idl::*;
use crate::gpu::num::Size64;
use crate::gpu::tables::{blend_factor_required_feature, format_required_feature};

/// `WGPU_LIMIT_U64_UNDEFINED`: bindgen evaluates the header's `UINT64_MAX` as `-1`; the cast restores the all-ones
/// value.
pub(crate) const LIMIT_U64_UNDEFINED: u64 = sys::WGPU_LIMIT_U64_UNDEFINED as u64;
/// `WGPU_DEPTH_CLEAR_VALUE_UNDEFINED`: the header defines it as `NAN`, and any NaN means undefined.
pub(crate) const DEPTH_CLEAR_VALUE_UNDEFINED: f32 = f32::NAN;

fn unrepresentable(what: &str, sentinel: &str) -> Exception {
    Exception::unsupported(format!(
        "{what} = 0xFFFFFFFF cannot be passed through webgpu.h, which reserves it as {sentinel}"
    ))
}

/// An optional `u32` member whose undefined sentinel is `u32::MAX`.
fn opt_u32(v: Option<u32>, what: &str, sentinel: &'static str) -> Result<u32, Exception> {
    match v {
        None => Ok(u32::MAX),
        Some(u32::MAX) => Err(unrepresentable(what, sentinel)),
        Some(x) => Ok(x),
    }
}

/// An optional size of a buffer range, with the specification's default when absent: `max(0, buffer size − offset)`
/// (clearBuffer, setIndexBuffer, setVertexBuffer, mapAsync, the effective buffer binding size). The binding computes it
/// rather than passing `WGPU_WHOLE_SIZE`: the buffer's size is a content-timeline attribute, the result is the same, and
/// not every implementation accepts the sentinel everywhere (wgpu-native v29.0.1.1's mapAsync rejects it).
pub(crate) fn size_or_rest(size: Option<Size64>, buffer_size: u64, offset: Size64) -> u64 {
    size.map_or(buffer_size.saturating_sub(offset.get()), Size64::get)
}

/// IDL flags as the header's flag type. The IDL and webgpu.h give each named flag the same bit (a test checks it for
/// the linked header). Bits no flag names are passed on unchanged, for the implementation's validation to reject as
/// the specification requires.
pub(crate) fn flags(v: u32) -> sys::WGPUFlags {
    sys::WGPUFlags::from(v)
}

fn e<E: conv::CEnum>(v: E) -> Result<u32, Exception> {
    to_c(v)
}

fn opt_e<E: conv::CEnum>(v: Option<E>, ty: &'static str) -> Result<u32, Exception> {
    match v {
        Some(v) => to_c(v),
        None => Ok(c_named(ty, "Undefined")),
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Content-timeline validation shared by several methods
// ---------------------------------------------------------------------------------------------------------------------

/// "Validate texture format required features".
pub(crate) fn check_format(dev: &DeviceShared, format: TextureFormat) -> Result<(), Exception> {
    if let Some(f) = format_required_feature(format)
        && !dev.has(f)
    {
        return Err(Exception::new(
            ExceptionKind::TypeError,
            format!("texture format \"{}\" requires feature \"{}\"", format.as_str(), f.as_str()),
        ));
    }
    Ok(())
}

/// "validate GPUExtent3D shape", then the accessors' defaults.
pub(crate) fn extent(v: &Extent3D) -> Result<sys::WGPUExtent3D, Exception> {
    match v {
        Extent3D::Sequence(s) => {
            if s.is_empty() || s.len() > 3 {
                return Err(Exception::new(
                    ExceptionKind::TypeError,
                    format!("a GPUExtent3D sequence has {} elements", s.len()),
                ));
            }
            Ok(sys::WGPUExtent3D {
                width: s[0],
                height: s.get(1).copied().unwrap_or(1),
                depthOrArrayLayers: s.get(2).copied().unwrap_or(1),
            })
        }
        Extent3D::Extent3DDict(d) => {
            Ok(sys::WGPUExtent3D { width: d.width, height: d.height, depthOrArrayLayers: d.depth_or_array_layers })
        }
    }
}

/// "validate GPUOrigin3D shape", then the accessors' defaults.
pub(crate) fn origin3d(v: &Origin3D) -> Result<sys::WGPUOrigin3D, Exception> {
    match v {
        Origin3D::Sequence(s) => {
            if s.len() > 3 {
                return Err(Exception::new(
                    ExceptionKind::TypeError,
                    format!("a GPUOrigin3D sequence has {} elements", s.len()),
                ));
            }
            Ok(sys::WGPUOrigin3D {
                x: s.first().copied().unwrap_or(0),
                y: s.get(1).copied().unwrap_or(0),
                z: s.get(2).copied().unwrap_or(0),
            })
        }
        Origin3D::Origin3DDict(d) => Ok(sys::WGPUOrigin3D { x: d.x, y: d.y, z: d.z }),
    }
}

/// "validate GPUColor shape", then the accessors.
pub(crate) fn color(v: &Color) -> Result<sys::WGPUColor, Exception> {
    match v {
        Color::Sequence(s) => {
            if s.len() != 4 {
                return Err(Exception::new(
                    ExceptionKind::TypeError,
                    format!("a GPUColor sequence has {} elements", s.len()),
                ));
            }
            Ok(sys::WGPUColor { r: s[0].get(), g: s[1].get(), b: s[2].get(), a: s[3].get() })
        }
        Color::ColorDict(d) => Ok(sys::WGPUColor { r: d.r.get(), g: d.g.get(), b: d.b.get(), a: d.a.get() }),
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------------------------------------------------

macro_rules! limits_impl {
    (u32: $($a:ident),*; u64: $($b:ident),*; compat: $($c:ident),*) => {
        /// `WGPULimits` with every field at its undefined sentinel.
        pub(crate) fn undefined_limits() -> sys::WGPULimits {
            let mut l: sys::WGPULimits = unsafe { std::mem::zeroed() };
            l.nextInChain = null_mut();
            $(l.$a = sys::WGPU_LIMIT_U32_UNDEFINED;)*
            $(l.$b = LIMIT_U64_UNDEFINED;)*
            l
        }

        /// `WGPUCompatibilityModeLimits` with every field at its undefined sentinel.
        pub(crate) fn undefined_compat_limits() -> sys::WGPUCompatibilityModeLimits {
            let mut c: sys::WGPUCompatibilityModeLimits = unsafe { std::mem::zeroed() };
            c.chain = conv::chain("CompatibilityModeLimits", null_mut());
            $(c.$c = sys::WGPU_LIMIT_U32_UNDEFINED;)*
            c
        }

        /// A limit by IDL name; `None` when it holds the undefined sentinel.
        pub(crate) fn get_limit(l: &sys::WGPULimits, c: &sys::WGPUCompatibilityModeLimits, name: &str) -> Option<u64> {
            match name {
                $(stringify!($a) => (l.$a != sys::WGPU_LIMIT_U32_UNDEFINED).then_some(u64::from(l.$a)),)*
                $(stringify!($b) => (l.$b != LIMIT_U64_UNDEFINED).then_some(l.$b),)*
                $(stringify!($c) => (c.$c != sys::WGPU_LIMIT_U32_UNDEFINED).then_some(u64::from(c.$c)),)*
                _ => None,
            }
        }

        /// Set a limit by IDL name. Returns false for a name that is not a limit, and for a value the header cannot
        /// carry (one that does not fit the field, or equals the undefined sentinel).
        pub(crate) fn set_limit(l: &mut sys::WGPULimits, c: &mut sys::WGPUCompatibilityModeLimits, name: &str, v: u64) -> bool {
            match name {
                $(stringify!($a) => match u32::try_from(v) {
                    Ok(x) if x != sys::WGPU_LIMIT_U32_UNDEFINED => { l.$a = x; true }
                    _ => false,
                },)*
                $(stringify!($b) => if v != LIMIT_U64_UNDEFINED { l.$b = v; true } else { false },)*
                $(stringify!($c) => match u32::try_from(v) {
                    Ok(x) if x != sys::WGPU_LIMIT_U32_UNDEFINED => { c.$c = x; true }
                    _ => false,
                },)*
                _ => false,
            }
        }
    };
}

limits_impl! {
    u32: maxTextureDimension1D, maxTextureDimension2D, maxTextureDimension3D, maxTextureArrayLayers, maxBindGroups,
    maxBindGroupsPlusVertexBuffers, maxBindingsPerBindGroup, maxDynamicUniformBuffersPerPipelineLayout,
    maxDynamicStorageBuffersPerPipelineLayout, maxSampledTexturesPerShaderStage, maxSamplersPerShaderStage,
    maxStorageBuffersPerShaderStage, maxStorageTexturesPerShaderStage, maxUniformBuffersPerShaderStage,
    minUniformBufferOffsetAlignment, minStorageBufferOffsetAlignment, maxVertexBuffers, maxVertexAttributes,
    maxVertexBufferArrayStride, maxInterStageShaderVariables, maxColorAttachments, maxColorAttachmentBytesPerSample,
    maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY, maxComputeWorkgroupSizeZ, maxComputeWorkgroupsPerDimension, maxImmediateSize;
    u64: maxUniformBufferBindingSize, maxStorageBufferBindingSize, maxBufferSize;
    compat: maxStorageBuffersInVertexStage, maxStorageTexturesInVertexStage, maxStorageBuffersInFragmentStage,
    maxStorageTexturesInFragmentStage
}

// ---------------------------------------------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------------------------------------------

/// A `WGPUDeviceDescriptor` for the request (the caller fills in the callbacks). Every limit and feature has already
/// been validated against the adapter.
pub(crate) fn device_descriptor(a: &mut Arena, d: &DeviceDescriptor) -> Result<sys::WGPUDeviceDescriptor, Exception> {
    let feats: Vec<sys::WGPUFeatureName> =
        d.required_features.iter().map(|f| to_c(*f).map(|v| v as sys::WGPUFeatureName)).collect::<Result<_, _>>()?;
    let (fp, fnum) = a.slice(feats);
    let compat = a.one_mut(undefined_compat_limits());
    let limits = a.one_mut(undefined_limits());
    unsafe {
        (*limits).nextInChain = &mut (*compat).chain;
        for (k, v) in &d.required_limits {
            if let Some(v) = v
                && !set_limit(&mut *limits, &mut *compat, k, v.get())
            {
                return Err(Exception::unsupported(format!(
                    "requiredLimits.{k} = {} cannot be passed through webgpu.h",
                    v.get()
                )));
            }
        }
    }
    let mut out: sys::WGPUDeviceDescriptor = unsafe { std::mem::zeroed() };
    out.nextInChain = null_mut();
    out.label = a.str(&d.label);
    out.requiredFeatureCount = fnum;
    out.requiredFeatures = fp;
    out.requiredLimits = limits;
    out.defaultQueue = sys::WGPUQueueDescriptor { nextInChain: null_mut(), label: a.str(&d.default_queue.label) };
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------------------------------------------------

pub(crate) fn buffer_descriptor(d: &BufferDescriptor) -> sys::WGPUBufferDescriptor {
    sys::WGPUBufferDescriptor {
        nextInChain: null_mut(),
        label: sv(&d.label),
        usage: flags(d.usage),
        size: d.size.get(),
        mappedAtCreation: wbool(d.mapped_at_creation),
    }
}

pub(crate) fn texture_descriptor(
    a: &mut Arena,
    d: &TextureDescriptor,
) -> Result<sys::WGPUTextureDescriptor, Exception> {
    let size = extent(&d.size)?;
    let views: Vec<sys::WGPUTextureFormat> =
        d.view_formats.iter().map(|f| e(*f).map(|v| v as sys::WGPUTextureFormat)).collect::<Result<_, _>>()?;
    let (vp, vn) = a.slice(views);
    let mut next: *mut sys::WGPUChainedStruct = null_mut();
    if let Some(dim) = d.texture_binding_view_dimension {
        let c = a.one_mut(sys::WGPUTextureBindingViewDimension {
            chain: conv::chain("TextureBindingViewDimension", null_mut()),
            textureBindingViewDimension: e(dim)? as sys::WGPUTextureViewDimension,
        });
        next = unsafe { &mut (*c).chain };
    }
    Ok(sys::WGPUTextureDescriptor {
        nextInChain: next,
        label: a.str(&d.label),
        usage: flags(d.usage),
        dimension: e(d.dimension)? as sys::WGPUTextureDimension,
        size,
        format: e(d.format)? as sys::WGPUTextureFormat,
        mipLevelCount: d.mip_level_count,
        sampleCount: d.sample_count,
        viewFormatCount: vn,
        viewFormats: vp,
    })
}

/// "Validate swizzle string" (`^[rgba01]{4}$`); `None` for the identity "rgba".
pub(crate) fn swizzle(s: &str) -> Result<Option<sys::WGPUTextureComponentSwizzle>, Exception> {
    let bytes = s.as_bytes();
    if bytes.len() != 4 || !bytes.iter().all(|b| b"rgba01".contains(b)) {
        return Err(Exception::new(
            ExceptionKind::TypeError,
            format!("swizzle \"{s}\" does not match ^[rgba01]{{4}}$"),
        ));
    }
    if s == "rgba" {
        return Ok(None);
    }
    let comp = |b: u8| -> sys::WGPUComponentSwizzle {
        let name = match b {
            b'0' => "Zero",
            b'1' => "One",
            b'r' => "R",
            b'g' => "G",
            b'b' => "B",
            _ => "A",
        };
        c_named("WGPUComponentSwizzle", name) as sys::WGPUComponentSwizzle
    };
    Ok(Some(sys::WGPUTextureComponentSwizzle {
        r: comp(bytes[0]),
        g: comp(bytes[1]),
        b: comp(bytes[2]),
        a: comp(bytes[3]),
    }))
}

pub(crate) fn texture_view_descriptor(
    a: &mut Arena,
    d: &TextureViewDescriptor,
) -> Result<sys::WGPUTextureViewDescriptor, Exception> {
    let mut next: *mut sys::WGPUChainedStruct = null_mut();
    if let Some(sw) = swizzle(&d.swizzle)? {
        // Any swizzle other than the identity travels in the chained structure.
        let c = a.one_mut(sys::WGPUTextureComponentSwizzleDescriptor {
            chain: conv::chain("TextureComponentSwizzleDescriptor", null_mut()),
            swizzle: sw,
        });
        next = unsafe { &mut (*c).chain };
    }
    Ok(sys::WGPUTextureViewDescriptor {
        nextInChain: next,
        label: a.str(&d.label),
        format: opt_e(d.format, "WGPUTextureFormat")? as sys::WGPUTextureFormat,
        dimension: opt_e(d.dimension, "WGPUTextureViewDimension")? as sys::WGPUTextureViewDimension,
        baseMipLevel: d.base_mip_level,
        mipLevelCount: opt_u32(d.mip_level_count, "mipLevelCount", "WGPU_MIP_LEVEL_COUNT_UNDEFINED")?,
        baseArrayLayer: d.base_array_layer,
        arrayLayerCount: opt_u32(d.array_layer_count, "arrayLayerCount", "WGPU_ARRAY_LAYER_COUNT_UNDEFINED")?,
        aspect: e(d.aspect)? as sys::WGPUTextureAspect,
        usage: flags(d.usage),
    })
}

pub(crate) fn sampler_descriptor(d: &SamplerDescriptor) -> Result<sys::WGPUSamplerDescriptor, Exception> {
    Ok(sys::WGPUSamplerDescriptor {
        nextInChain: null_mut(),
        label: sv(&d.label),
        addressModeU: e(d.address_mode_u)? as sys::WGPUAddressMode,
        addressModeV: e(d.address_mode_v)? as sys::WGPUAddressMode,
        addressModeW: e(d.address_mode_w)? as sys::WGPUAddressMode,
        magFilter: e(d.mag_filter)? as sys::WGPUFilterMode,
        minFilter: e(d.min_filter)? as sys::WGPUFilterMode,
        mipmapFilter: e(d.mipmap_filter)? as sys::WGPUMipmapFilterMode,
        lodMinClamp: d.lod_min_clamp.get(),
        lodMaxClamp: d.lod_max_clamp.get(),
        compare: opt_e(d.compare, "WGPUCompareFunction")? as sys::WGPUCompareFunction,
        maxAnisotropy: d.max_anisotropy,
    })
}

pub(crate) fn bind_group_layout_descriptor(
    a: &mut Arena,
    dev: &DeviceShared,
    d: &BindGroupLayoutDescriptor,
) -> Result<sys::WGPUBindGroupLayoutDescriptor, Exception> {
    let mut entries = Vec::with_capacity(d.entries.len());
    for en in &d.entries {
        let mut x: sys::WGPUBindGroupLayoutEntry = unsafe { std::mem::zeroed() };
        x.binding = en.binding;
        x.visibility = flags(en.visibility);
        x.bindingArraySize = 0;
        x.buffer.type_ = match &en.buffer {
            Some(b) => {
                x.buffer.hasDynamicOffset = wbool(b.has_dynamic_offset);
                x.buffer.minBindingSize = b.min_binding_size.get();
                e(b.r#type)?
            }
            None => c_named("WGPUBufferBindingType", "BindingNotUsed"),
        } as sys::WGPUBufferBindingType;
        x.sampler.type_ = match &en.sampler {
            Some(s) => e(s.r#type)?,
            None => c_named("WGPUSamplerBindingType", "BindingNotUsed"),
        } as sys::WGPUSamplerBindingType;
        x.texture.sampleType = match &en.texture {
            Some(t) => {
                x.texture.viewDimension = e(t.view_dimension)? as sys::WGPUTextureViewDimension;
                x.texture.multisampled = wbool(t.multisampled);
                e(t.sample_type)?
            }
            None => c_named("WGPUTextureSampleType", "BindingNotUsed"),
        } as sys::WGPUTextureSampleType;
        x.storageTexture.access = match &en.storage_texture {
            Some(s) => {
                check_format(dev, s.format)?;
                x.storageTexture.format = e(s.format)? as sys::WGPUTextureFormat;
                x.storageTexture.viewDimension = e(s.view_dimension)? as sys::WGPUTextureViewDimension;
                e(s.access)?
            }
            None => c_named("WGPUStorageTextureAccess", "BindingNotUsed"),
        } as sys::WGPUStorageTextureAccess;
        if en.external_texture.is_some() {
            let c = a.one_mut(sys::WGPUExternalTextureBindingLayout {
                chain: conv::chain("ExternalTextureBindingLayout", null_mut()),
            });
            x.nextInChain = unsafe { &mut (*c).chain };
        }
        entries.push(x);
    }
    let (p, n) = a.slice(entries);
    Ok(sys::WGPUBindGroupLayoutDescriptor {
        nextInChain: null_mut(),
        label: a.str(&d.label),
        entryCount: n,
        entries: p,
    })
}

pub(crate) fn pipeline_layout_descriptor(
    a: &mut Arena,
    d: &PipelineLayoutDescriptor<'_, Native>,
) -> sys::WGPUPipelineLayoutDescriptor {
    // A null element of bindGroupLayouts is a null handle.
    let layouts: Vec<sys::WGPUBindGroupLayout> =
        d.bind_group_layouts.iter().map(|l| l.map_or(null_mut(), |l| l.raw())).collect();
    let (p, n) = a.slice(layouts);
    sys::WGPUPipelineLayoutDescriptor {
        nextInChain: null_mut(),
        label: a.str(&d.label),
        bindGroupLayoutCount: n,
        bindGroupLayouts: p,
        immediateSize: d.immediate_size,
    }
}

/// `GPUBindGroupDescriptor`. A `GPUTexture` resource is "get as texture view": its `createView()`, kept alive in
/// `views` until the bind group exists. A `GPUBuffer` resource binds the whole buffer.
pub(crate) fn bind_group_descriptor(
    a: &mut Arena,
    d: &BindGroupDescriptor<'_, Native>,
    views: &mut Vec<NativeTextureView>,
) -> Result<sys::WGPUBindGroupDescriptor, Exception> {
    let mut entries = Vec::with_capacity(d.entries.len());
    for en in &d.entries {
        let mut x: sys::WGPUBindGroupEntry = unsafe { std::mem::zeroed() };
        x.binding = en.binding;
        match &en.resource {
            BindingResource::Sampler(s) => x.sampler = s.raw(),
            BindingResource::TextureView(v) => x.textureView = v.raw(),
            BindingResource::Texture(t) => {
                let v = t.create_view(&TextureViewDescriptor::new())?;
                x.textureView = v.raw();
                views.push(v);
            }
            // A GPUBuffer resource is the binding {buffer, offset 0, size undefined} ("get as buffer binding").
            BindingResource::Buffer(b) => {
                x.buffer = b.raw();
                x.offset = 0;
                x.size = b.size();
            }
            BindingResource::BufferBinding(bb) => {
                x.buffer = bb.buffer.raw();
                x.offset = bb.offset.get();
                x.size = size_or_rest(bb.size, bb.buffer.size(), bb.offset);
            }
            BindingResource::ExternalTexture(t) => t.absurd(),
        }
        entries.push(x);
    }
    let (p, n) = a.slice(entries);
    Ok(sys::WGPUBindGroupDescriptor {
        nextInChain: null_mut(),
        label: a.str(&d.label),
        layout: d.layout.raw(),
        entryCount: n,
        entries: p,
    })
}

pub(crate) fn shader_module_descriptor(
    a: &mut Arena,
    d: &ShaderModuleDescriptor<'_, Native>,
) -> sys::WGPUShaderModuleDescriptor {
    // compilationHints have no observable effect other than performance (§9.1.1.1); webgpu.h has no field for them.
    let code = a.str(&d.code);
    let wgsl = a.one_mut(sys::WGPUShaderSourceWGSL { chain: conv::chain("ShaderSourceWGSL", null_mut()), code });
    sys::WGPUShaderModuleDescriptor { nextInChain: unsafe { &mut (*wgsl).chain }, label: a.str(&d.label) }
}

fn layout(l: &PipelineLayoutOrAuto<'_, Native>) -> sys::WGPUPipelineLayout {
    match l {
        PipelineLayoutOrAuto::PipelineLayout(p) => p.raw(),
        // "auto" is a null layout in webgpu.h.
        PipelineLayoutOrAuto::AutoLayoutMode(AutoLayoutMode::Auto) => null_mut(),
    }
}

fn entry_point(a: &mut Arena, e: &Option<String>) -> sys::WGPUStringView {
    match e {
        Some(s) => a.str(s),
        None => conv::sv_null(),
    }
}

fn constants(a: &mut Arena, c: &[(String, crate::gpu::num::Float64)]) -> (*const sys::WGPUConstantEntry, usize) {
    let v: Vec<sys::WGPUConstantEntry> = c
        .iter()
        .map(|(k, v)| sys::WGPUConstantEntry { nextInChain: null_mut(), key: a.str(k), value: v.get() })
        .collect();
    a.slice(v)
}

pub(crate) fn compute_pipeline_descriptor(
    a: &mut Arena,
    d: &ComputePipelineDescriptor<'_, Native>,
) -> sys::WGPUComputePipelineDescriptor {
    let (cp, cn) = constants(a, &d.compute.constants);
    let entry = entry_point(a, &d.compute.entry_point);
    sys::WGPUComputePipelineDescriptor {
        nextInChain: null_mut(),
        label: a.str(&d.label),
        layout: layout(&d.layout),
        compute: sys::WGPUComputeState {
            nextInChain: null_mut(),
            module: d.compute.module.raw(),
            entryPoint: entry,
            constantCount: cn,
            constants: cp,
        },
    }
}

/// The content-timeline steps of `createRenderPipeline()`: the formats' required features, and the features the blend
/// factors require (a value of a non-enabled enum always throws `TypeError`, §3.6.1).
pub(crate) fn check_render_pipeline(
    dev: &DeviceShared,
    d: &RenderPipelineDescriptor<'_, Native>,
) -> Result<(), Exception> {
    if let Some(f) = &d.fragment {
        for t in f.targets.iter().flatten() {
            check_format(dev, t.format)?;
            if let Some(b) = &t.blend {
                for comp in [&b.color, &b.alpha] {
                    for factor in [comp.src_factor, comp.dst_factor] {
                        if let Some(feat) = blend_factor_required_feature(factor)
                            && !dev.has(feat)
                        {
                            return Err(Exception::new(
                                ExceptionKind::TypeError,
                                format!("blend factor \"{}\" requires feature \"{}\"", factor.as_str(), feat.as_str()),
                            ));
                        }
                    }
                }
            }
        }
    }
    if let Some(ds) = &d.depth_stencil {
        check_format(dev, ds.format)?;
    }
    Ok(())
}

fn stencil_face(s: &StencilFaceState) -> Result<sys::WGPUStencilFaceState, Exception> {
    Ok(sys::WGPUStencilFaceState {
        compare: e(s.compare)? as sys::WGPUCompareFunction,
        failOp: e(s.fail_op)? as sys::WGPUStencilOperation,
        depthFailOp: e(s.depth_fail_op)? as sys::WGPUStencilOperation,
        passOp: e(s.pass_op)? as sys::WGPUStencilOperation,
    })
}

fn blend_component(b: &BlendComponent) -> Result<sys::WGPUBlendComponent, Exception> {
    Ok(sys::WGPUBlendComponent {
        operation: e(b.operation)? as sys::WGPUBlendOperation,
        srcFactor: e(b.src_factor)? as sys::WGPUBlendFactor,
        dstFactor: e(b.dst_factor)? as sys::WGPUBlendFactor,
    })
}

pub(crate) fn render_pipeline_descriptor(
    a: &mut Arena,
    d: &RenderPipelineDescriptor<'_, Native>,
) -> Result<sys::WGPURenderPipelineDescriptor, Exception> {
    // A null buffer layout is a hole: stepMode Undefined with no attributes.
    let mut buffers = Vec::with_capacity(d.vertex.buffers.len());
    for b in &d.vertex.buffers {
        buffers.push(match b {
            Some(b) => {
                let attrs: Vec<sys::WGPUVertexAttribute> = b
                    .attributes
                    .iter()
                    .map(|at| {
                        Ok(sys::WGPUVertexAttribute {
                            nextInChain: null_mut(),
                            format: e(at.format)? as sys::WGPUVertexFormat,
                            offset: at.offset.get(),
                            shaderLocation: at.shader_location,
                        })
                    })
                    .collect::<Result<_, Exception>>()?;
                let (ap, an) = a.slice(attrs);
                sys::WGPUVertexBufferLayout {
                    nextInChain: null_mut(),
                    stepMode: e(b.step_mode)? as sys::WGPUVertexStepMode,
                    arrayStride: b.array_stride.get(),
                    attributeCount: an,
                    attributes: ap,
                }
            }
            None => sys::WGPUVertexBufferLayout {
                nextInChain: null_mut(),
                stepMode: c_named("WGPUVertexStepMode", "Undefined") as sys::WGPUVertexStepMode,
                arrayStride: 0,
                attributeCount: 0,
                attributes: std::ptr::null(),
            },
        });
    }
    let (bp, bn) = a.slice(buffers);
    let (vcp, vcn) = constants(a, &d.vertex.constants);
    let vertex = sys::WGPUVertexState {
        nextInChain: null_mut(),
        module: d.vertex.module.raw(),
        entryPoint: entry_point(a, &d.vertex.entry_point),
        constantCount: vcn,
        constants: vcp,
        bufferCount: bn,
        buffers: bp,
    };
    let p = &d.primitive;
    let primitive = sys::WGPUPrimitiveState {
        nextInChain: null_mut(),
        topology: e(p.topology)? as sys::WGPUPrimitiveTopology,
        stripIndexFormat: opt_e(p.strip_index_format, "WGPUIndexFormat")? as sys::WGPUIndexFormat,
        frontFace: e(p.front_face)? as sys::WGPUFrontFace,
        cullMode: e(p.cull_mode)? as sys::WGPUCullMode,
        unclippedDepth: wbool(p.unclipped_depth),
    };
    let depth_stencil = match &d.depth_stencil {
        Some(ds) => a.one(sys::WGPUDepthStencilState {
            nextInChain: null_mut(),
            format: e(ds.format)? as sys::WGPUTextureFormat,
            depthWriteEnabled: match ds.depth_write_enabled {
                None => c_named("WGPUOptionalBool", "Undefined"),
                Some(true) => c_named("WGPUOptionalBool", "True"),
                Some(false) => c_named("WGPUOptionalBool", "False"),
            } as sys::WGPUOptionalBool,
            depthCompare: opt_e(ds.depth_compare, "WGPUCompareFunction")? as sys::WGPUCompareFunction,
            stencilFront: stencil_face(&ds.stencil_front)?,
            stencilBack: stencil_face(&ds.stencil_back)?,
            stencilReadMask: ds.stencil_read_mask,
            stencilWriteMask: ds.stencil_write_mask,
            depthBias: ds.depth_bias,
            depthBiasSlopeScale: ds.depth_bias_slope_scale.get(),
            depthBiasClamp: ds.depth_bias_clamp.get(),
        }),
        None => std::ptr::null(),
    };
    let m = &d.multisample;
    let multisample = sys::WGPUMultisampleState {
        nextInChain: null_mut(),
        count: m.count,
        mask: m.mask,
        alphaToCoverageEnabled: wbool(m.alpha_to_coverage_enabled),
    };
    let fragment = match &d.fragment {
        Some(f) => {
            let mut targets = Vec::with_capacity(f.targets.len());
            for t in &f.targets {
                targets.push(match t {
                    Some(t) => {
                        let blend = match &t.blend {
                            Some(b) => a.one(sys::WGPUBlendState {
                                color: blend_component(&b.color)?,
                                alpha: blend_component(&b.alpha)?,
                            }),
                            None => std::ptr::null(),
                        };
                        sys::WGPUColorTargetState {
                            nextInChain: null_mut(),
                            format: e(t.format)? as sys::WGPUTextureFormat,
                            blend,
                            writeMask: flags(t.write_mask),
                        }
                    }
                    // A null target is a hole: format Undefined.
                    None => sys::WGPUColorTargetState {
                        nextInChain: null_mut(),
                        format: c_named("WGPUTextureFormat", "Undefined") as sys::WGPUTextureFormat,
                        blend: std::ptr::null(),
                        writeMask: 0,
                    },
                });
            }
            let (tp, tn) = a.slice(targets);
            let (fcp, fcn) = constants(a, &f.constants);
            let fs = sys::WGPUFragmentState {
                nextInChain: null_mut(),
                module: f.module.raw(),
                entryPoint: entry_point(a, &f.entry_point),
                constantCount: fcn,
                constants: fcp,
                targetCount: tn,
                targets: tp,
            };
            a.one(fs)
        }
        None => std::ptr::null(),
    };
    Ok(sys::WGPURenderPipelineDescriptor {
        nextInChain: null_mut(),
        label: a.str(&d.label),
        layout: layout(&d.layout),
        vertex,
        primitive,
        depthStencil: depth_stencil,
        multisample,
        fragment,
    })
}

pub(crate) fn query_set_descriptor(d: &QuerySetDescriptor) -> Result<sys::WGPUQuerySetDescriptor, Exception> {
    Ok(sys::WGPUQuerySetDescriptor {
        nextInChain: null_mut(),
        label: sv(&d.label),
        type_: e(d.r#type)? as sys::WGPUQueryType,
        count: d.count,
    })
}

pub(crate) fn render_bundle_encoder_descriptor(
    a: &mut Arena,
    d: &RenderBundleEncoderDescriptor,
) -> Result<sys::WGPURenderBundleEncoderDescriptor, Exception> {
    let formats: Vec<sys::WGPUTextureFormat> = d
        .color_formats
        .iter()
        .map(|f| opt_e(*f, "WGPUTextureFormat").map(|v| v as sys::WGPUTextureFormat))
        .collect::<Result<_, _>>()?;
    let (fp, fnum) = a.slice(formats);
    Ok(sys::WGPURenderBundleEncoderDescriptor {
        nextInChain: null_mut(),
        label: a.str(&d.label),
        colorFormatCount: fnum,
        colorFormats: fp,
        depthStencilFormat: opt_e(d.depth_stencil_format, "WGPUTextureFormat")? as sys::WGPUTextureFormat,
        sampleCount: d.sample_count,
        depthReadOnly: wbool(d.depth_read_only),
        stencilReadOnly: wbool(d.stencil_read_only),
    })
}

// ---------------------------------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------------------------------

/// "get as texture view": a texture's `createView()`, kept alive in `views` until the call that uses it.
fn texture_or_view(
    v: &TextureOrView<'_, Native>,
    views: &mut Vec<NativeTextureView>,
) -> Result<sys::WGPUTextureView, Exception> {
    Ok(match v {
        TextureOrView::TextureView(v) => v.raw(),
        TextureOrView::Texture(t) => {
            let v = t.create_view(&TextureViewDescriptor::new())?;
            let raw = v.raw();
            views.push(v);
            raw
        }
    })
}

fn timestamp_writes(
    a: &mut Arena,
    q: &NativeQuerySet,
    begin: Option<u32>,
    end: Option<u32>,
) -> Result<*const sys::WGPUPassTimestampWrites, Exception> {
    Ok(a.one(sys::WGPUPassTimestampWrites {
        nextInChain: null_mut(),
        querySet: q.raw(),
        beginningOfPassWriteIndex: opt_u32(begin, "beginningOfPassWriteIndex", "WGPU_QUERY_SET_INDEX_UNDEFINED")?,
        endOfPassWriteIndex: opt_u32(end, "endOfPassWriteIndex", "WGPU_QUERY_SET_INDEX_UNDEFINED")?,
    }))
}

pub(crate) fn compute_pass_descriptor(
    a: &mut Arena,
    d: &ComputePassDescriptor<'_, Native>,
) -> Result<sys::WGPUComputePassDescriptor, Exception> {
    let tw = match &d.timestamp_writes {
        Some(t) => timestamp_writes(a, t.query_set, t.beginning_of_pass_write_index, t.end_of_pass_write_index)?,
        None => std::ptr::null(),
    };
    Ok(sys::WGPUComputePassDescriptor { nextInChain: null_mut(), label: a.str(&d.label), timestampWrites: tw })
}

pub(crate) fn render_pass_descriptor(
    a: &mut Arena,
    d: &RenderPassDescriptor<'_, Native>,
    views: &mut Vec<NativeTextureView>,
) -> Result<sys::WGPURenderPassDescriptor, Exception> {
    let mut colors = Vec::with_capacity(d.color_attachments.len());
    for c in &d.color_attachments {
        colors.push(match c {
            Some(c) => {
                // The shape of clearValue is validated on the content timeline; a missing clearValue is {0, 0, 0, 0}.
                let clear = match &c.clear_value {
                    Some(v) => color(v)?,
                    None => sys::WGPUColor { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                };
                sys::WGPURenderPassColorAttachment {
                    nextInChain: null_mut(),
                    view: texture_or_view(&c.view, views)?,
                    depthSlice: opt_u32(c.depth_slice, "depthSlice", "WGPU_DEPTH_SLICE_UNDEFINED")?,
                    resolveTarget: match &c.resolve_target {
                        Some(r) => texture_or_view(r, views)?,
                        None => null_mut(),
                    },
                    loadOp: e(c.load_op)? as sys::WGPULoadOp,
                    storeOp: e(c.store_op)? as sys::WGPUStoreOp,
                    clearValue: clear,
                }
            }
            // A null attachment is a null view.
            None => {
                let mut x: sys::WGPURenderPassColorAttachment = unsafe { std::mem::zeroed() };
                x.depthSlice = sys::WGPU_DEPTH_SLICE_UNDEFINED;
                x
            }
        });
    }
    let (cp, cn) = a.slice(colors);
    let ds = match &d.depth_stencil_attachment {
        Some(ds) => {
            let view = texture_or_view(&ds.view, views)?;
            a.one(sys::WGPURenderPassDepthStencilAttachment {
                nextInChain: null_mut(),
                view,
                depthLoadOp: opt_e(ds.depth_load_op, "WGPULoadOp")? as sys::WGPULoadOp,
                depthStoreOp: opt_e(ds.depth_store_op, "WGPUStoreOp")? as sys::WGPUStoreOp,
                depthClearValue: ds.depth_clear_value.map_or(DEPTH_CLEAR_VALUE_UNDEFINED, |v| v.get()),
                depthReadOnly: wbool(ds.depth_read_only),
                stencilLoadOp: opt_e(ds.stencil_load_op, "WGPULoadOp")? as sys::WGPULoadOp,
                stencilStoreOp: opt_e(ds.stencil_store_op, "WGPUStoreOp")? as sys::WGPUStoreOp,
                stencilClearValue: ds.stencil_clear_value,
                stencilReadOnly: wbool(ds.stencil_read_only),
            })
        }
        None => std::ptr::null(),
    };
    let tw = match &d.timestamp_writes {
        Some(t) => timestamp_writes(a, t.query_set, t.beginning_of_pass_write_index, t.end_of_pass_write_index)?,
        None => std::ptr::null(),
    };
    // maxDrawCount always travels, so the implementation applies the IDL value, default included.
    let mdc = a.one_mut(sys::WGPURenderPassMaxDrawCount {
        chain: conv::chain("RenderPassMaxDrawCount", null_mut()),
        maxDrawCount: d.max_draw_count.get(),
    });
    Ok(sys::WGPURenderPassDescriptor {
        nextInChain: unsafe { &mut (*mdc).chain },
        label: a.str(&d.label),
        colorAttachmentCount: cn,
        colorAttachments: cp,
        depthStencilAttachment: ds,
        occlusionQuerySet: d.occlusion_query_set.map_or(null_mut(), |q| q.raw()),
        timestampWrites: tw,
    })
}

pub(crate) fn texel_copy_buffer_layout(l: &TexelCopyBufferLayout) -> Result<sys::WGPUTexelCopyBufferLayout, Exception> {
    Ok(sys::WGPUTexelCopyBufferLayout {
        offset: l.offset.get(),
        bytesPerRow: opt_u32(l.bytes_per_row, "bytesPerRow", "WGPU_COPY_STRIDE_UNDEFINED")?,
        rowsPerImage: opt_u32(l.rows_per_image, "rowsPerImage", "WGPU_COPY_STRIDE_UNDEFINED")?,
    })
}

pub(crate) fn texel_copy_buffer_info(
    i: &TexelCopyBufferInfo<'_, Native>,
) -> Result<sys::WGPUTexelCopyBufferInfo, Exception> {
    Ok(sys::WGPUTexelCopyBufferInfo {
        layout: sys::WGPUTexelCopyBufferLayout {
            offset: i.offset.get(),
            bytesPerRow: opt_u32(i.bytes_per_row, "bytesPerRow", "WGPU_COPY_STRIDE_UNDEFINED")?,
            rowsPerImage: opt_u32(i.rows_per_image, "rowsPerImage", "WGPU_COPY_STRIDE_UNDEFINED")?,
        },
        buffer: i.buffer.raw(),
    })
}

pub(crate) fn texel_copy_texture_info(
    i: &TexelCopyTextureInfo<'_, Native>,
) -> Result<sys::WGPUTexelCopyTextureInfo, Exception> {
    Ok(sys::WGPUTexelCopyTextureInfo {
        texture: i.texture.raw(),
        mipLevel: i.mip_level,
        origin: origin3d(&i.origin)?,
        aspect: e(i.aspect)? as sys::WGPUTextureAspect,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinels_are_the_documented_maxima() {
        assert_eq!(LIMIT_U64_UNDEFINED, u64::MAX);
        assert_eq!(sys::WGPU_STRLEN as usize, usize::MAX);
        assert_eq!(sys::WGPU_LIMIT_U32_UNDEFINED, u32::MAX);
    }

    #[test]
    fn idl_flags_have_the_header_bits() {
        let check = |ty: &'static str, flags: &[(&str, u32)]| {
            for (name, v) in flags {
                assert_eq!(c_named(ty, name), *v, "{ty} {name}");
            }
        };
        check("WGPUBufferUsage", &BufferUsage::FLAGS);
        check("WGPUMapMode", &MapMode::FLAGS);
        check("WGPUTextureUsage", &TextureUsage::FLAGS);
        check("WGPUShaderStage", &ShaderStage::FLAGS);
        check("WGPUColorWriteMask", &ColorWrite::FLAGS);
    }

    #[test]
    fn shapes() {
        assert!(extent(&Extent3D::Sequence(vec![])).is_err());
        assert!(extent(&Extent3D::Sequence(vec![1, 2, 3, 4])).is_err());
        let x = extent(&Extent3D::Sequence(vec![7])).unwrap();
        assert_eq!((x.width, x.height, x.depthOrArrayLayers), (7, 1, 1));
        assert!(origin3d(&Origin3D::Sequence(vec![0, 0, 0, 0])).is_err());
        let one = crate::gpu::num::Float64::new(1.0).unwrap();
        assert!(color(&Color::Sequence(vec![one; 3])).is_err());
        assert!(color(&Color::Sequence(vec![one; 4])).is_ok());
        assert_eq!(swizzle("rgbx").unwrap_err().kind, ExceptionKind::TypeError);
        assert!(swizzle("rgba").unwrap().is_none());
        assert!(swizzle("1g0r").unwrap().is_some());
    }

    #[test]
    fn every_idl_enum_value_has_an_enumerator() {
        fn missing<E: conv::CEnum>() -> Vec<String> {
            E::values()
                .iter()
                .filter(|v| to_c(**v).is_err())
                .map(|v| format!("{} {}", E::IDL_NAME, v.idl_str()))
                .collect()
        }
        let mut m = Vec::new();
        m.extend(missing::<TextureFormat>());
        m.extend(missing::<VertexFormat>());
        m.extend(missing::<BlendFactor>());
        m.extend(missing::<TextureViewDimension>());
        m.extend(missing::<TextureSampleType>());
        m.extend(missing::<StorageTextureAccess>());
        m.extend(missing::<ErrorFilter>());
        m.extend(missing::<QueryType>());
        // Values the linked header does not define (checked by name in the header itself). Using one is reported as
        // unsupported; any other gap is a naming mismatch in the binding.
        let absent: &[&str] = if cfg!(webgpu_native = "wgpu-native") {
            // wgpu-native v29.0.1.1 include/webgpu/webgpu.h has no WGPUVertexFormat_Snorm10_10_10_2.
            &["GPUVertexFormat snorm10-10-10-2"]
        } else {
            &[]
        };
        m.retain(|x| !absent.contains(&x.as_str()));
        assert!(m.is_empty(), "{} lacks {m:?}", sys::meta::IMPLEMENTATION);
        // A feature newer than the linked header is not an error: the binding cannot expose it.
        eprintln!("{} has no enumerator for {:?}", sys::meta::IMPLEMENTATION, missing::<FeatureName>());
    }
}
