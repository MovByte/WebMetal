//! How the executor provides each Metal pixel format the A10X reports, and what it refuses.
//!
//! [`handling`] classifies a format statically: the WebGPU format of the same definition, an emulation that stores it
//! in another WebGPU format with exact conversions, a refusal with the reason and its source, or not a format of the
//! device at all. [`resolve`] applies a host device's features. The capability checks ([`check_usage`],
//! [`check_filtering`], [`check_blending`], [`check_resolve`]) compare what a Metal call asks for with what the
//! resolved WebGPU format can do on the host; a capability the host lacks is a [`Refusal`], never a substitute.
//!
//! The format list is the union of `tables/formats.toml` and the formats `MTLPixelFormatGetInfoForDevice` knows for the
//! A10X (`pixel_format_list.known` of the device profile).

use super::decode::{self, Scheme};
use super::packed;
use super::table::{A10X_FORMATS, A10X_GETTERS, A10xFormatRecord, FormatClass, METAL_FORMATS, MetalFormatRow};
use super::webgpu::{Cond, Feature, HostFeatures, SampleType, WebGpuFormat};

/// Why the executor cannot provide a format or a capability exactly. Reported loudly, never substituted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// A row of `tables/refusals.md`, or a name for a refusal specific to this module.
    pub id: &'static str,
    pub reason: &'static str,
    /// Where the blocking rule or the missing definition is stated.
    pub source: &'static str,
}

/// A component a view or a shim reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Swizzle {
    R,
    G,
    B,
    A,
    Zero,
    One,
}

/// A conversion the shader performs between the stored texel and the value Metal defines. The exact rule of each is
/// named; the shader lowering implements it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shim {
    /// sRGB decode of `channels` stored unorm8 channels on read and encode on write (MSL 4.1 section 8.7.7); alpha is
    /// linear.
    Srgb8 { channels: u32 },
    /// The stored `r16uint` word is a texel of `layout`: channels read as [`packed::unorm_to_f32`] of their codes,
    /// missing color channels as 0 and missing alpha as 1; writes store [`packed::unorm_from_f32`] codes.
    Packed(&'static packed::Layout),
    /// The stored words are a texel of an XR `layout`: channels read as [`packed::xr_to_f32`] of their codes, the alpha
    /// of `BGRA10_XR` clamped to [0, 1].
    ExtendedRange(&'static packed::Layout),
    /// The stored `rgba16uint` values are the ASTC UNORM16 results `C`: channels read as [`super::astc::to_unorm12`]
    /// divided by 4095.
    AstcUnorm12,
}

/// How guest bytes become the stored texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Upload {
    /// The bytes are the texels.
    Bytes,
    /// [`packed::swap_red_blue_10`] on each 32-bit texel, both ways.
    SwapRedBlue10,
    /// Decoded from the blocks, which the executor keeps as the authoritative copy (WebGPU texture formats cannot
    /// hold them bit-exactly); readback and copies of the texture use the blocks.
    Decode(Scheme),
}

/// How a shader read or sample of the stored texture gives Metal's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read {
    /// The stored format's own read is Metal's value.
    Native,
    /// The stored format's read with components rearranged (a view swizzle where the host has
    /// `texture-component-swizzle`, else in the shader).
    Swizzle([Swizzle; 4]),
    Shim(Shim),
    Refused(Refusal),
}

/// How render-target and shader writes produce the stored texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Write {
    Native,
    /// The fragment output's alpha is written to the stored red channel. A blend state's alpha equation then applies
    /// to the red channel, with factors that name the destination alpha naming the stored red.
    AlphaToRed,
    Shim(Shim),
    /// Metal itself does not write the format (compressed formats are not render targets or writable).
    NotWritable,
    Refused(Refusal),
}

/// A Metal format stored in another WebGPU format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Emulation {
    pub storage: WebGpuFormat,
    pub upload: Upload,
    pub read: Read,
    pub write: Write,
}

/// The static handling of a Metal pixel format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handling {
    /// Metal rejects the format on the A10X itself: it is the invalid value, the iOS 17.7 SDK does not offer it, or
    /// `MTLPixelFormatGetInfoForDevice` returns the invalid record for it on this device.
    NotOnDevice(&'static str),
    /// The WebGPU format of the same definition (`same-format` in `tables/formats.toml`). `without_feature` applies on
    /// a host that lacks the format's required feature.
    Native {
        format: WebGpuFormat,
        without_feature: Option<Fallback>,
    },
    /// The stencil aspect of a `depth32float-stencil8` texture (`X32_Stencil8`).
    StencilAspect,
    Emulated(Emulation),
    Refused(Refusal),
}

/// What a same-format Metal format becomes on a host without the WebGPU format's feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    Emulated(Emulation),
    Refused(Refusal),
}

/// A format as a given host provides it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    Native(WebGpuFormat),
    StencilAspect,
    Emulated(Emulation),
}

impl Resolution {
    /// The WebGPU format of the texture the executor creates.
    pub fn storage(self) -> WebGpuFormat {
        match self {
            Resolution::Native(f) => f,
            Resolution::StencilAspect => WebGpuFormat::Depth32floatStencil8,
            Resolution::Emulated(e) => e.storage,
        }
    }
}

/// Why a format cannot be provided on a host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unavailable {
    NotOnDevice(&'static str),
    Refused(Refusal),
}

const PRIVATE_FORMAT: Refusal = Refusal {
    id: "private-pixel-format",
    reason: "The format is not in the SDK's MTLPixelFormat enum; the A10X profile records only its flag word, and no \
             source available here defines its texel layout or conversions.",
    source: "profile/ipados-17.7.11-21H461-a10x.toml, pixel_format_list.private",
};

const ASTC_HDR: Refusal = Refusal {
    id: "astc-hdr-on-a10x",
    reason: "The A10X answers supportsASTCHDRTextureCompression NO while MTLPixelFormatGetInfoForDevice marks the HDR \
             formats supported. Whether its texture unit returns the LDR profile's error color for HDR endpoint modes \
             or an HDR decode is hardware behavior with no reference here.",
    source: "profile/ipados-17.7.11-21H461-a10x.toml, getters.supportsASTCHDRTextureCompression and \
             pixel_formats.222-236",
};

const ASTC_PRIVATE_FOOTPRINT: Refusal = Refusal {
    id: "astc-private-footprint",
    reason: "A private ASTC format whose footprint (4x2 or 8x4 in the profile's block fields) is not one of the 2D \
             footprints of Table 155; the ASTC profiles support only those.",
    source: "Khronos Data Format Specification 1.4.0, section 23.8 and Table 155; profile pixel_format_list.private",
};

const EAC_R11: Refusal = Refusal {
    id: "eac-r11-software-path",
    reason: "Sampling EAC R11/RG11 textures without texture-compression-etc2 needs the A10X's decoded value, which is \
             not measured (Khronos allows the 11-bit value or its bit-replicated extension).",
    source: "tables/refusals.md, eac-r11-software-path; tables/formats.toml question 6.3-eac-r11-decode",
};

const UNORM16_WITHOUT_TIER1: Refusal = Refusal {
    id: "unorm16-without-texture-formats-tier1",
    reason: "R16, RG16 and RGBA16 Unorm and Snorm exist in WebGPU only with texture-formats-tier1, and no other \
             WebGPU format holds them exactly with the same filtering and rendering.",
    source: "tables/formats.toml question 6.3-unorm16-snorm16",
};

const DEPTH32_STENCIL8_WITHOUT_FEATURE: Refusal = Refusal {
    id: "depth32float-stencil8-without-feature",
    reason: "Depth32Float_Stencil8 needs the depth32float-stencil8 feature; depth24plus-stencil8 does not hold a \
             32-bit float depth and a render pass takes one depth-stencil texture.",
    source: "tables/formats.toml, webgpu_texture_format depth32float-stencil8 and depth24plus-stencil8",
};

const COMPATIBILITY_DEVICE: Refusal = Refusal {
    id: "compatibility-mode-device",
    reason: "The format exists in WebGPU only with core-features-and-limits, which a compatibility-mode device lacks.",
    source: "tables/formats.toml, the Required Feature cell of the format",
};

const CHROMA_422: Refusal = Refusal {
    id: "422-chroma-reconstruction",
    reason: "MTLPixelFormat.h states the byte order and that a read returns (Cr, Y, Cb, 1) without conversion, but not \
             how the shared Cb and Cr are reconstructed for each pixel of a pair or under filtering.",
    source: "tables/formats.toml question 6.3-422-chroma",
};

const XR_SRGB: Refusal = Refusal {
    id: "xr-srgb-transfer",
    reason: "The sRGB variants of the XR formats apply a transfer function to values outside [0, 1]; neither the SDK \
             header nor the pinned documentation defines it.",
    source: "tables/formats.toml question 6.3-xr-encoding (linear encoding only)",
};

const PACKED_BLEND: Refusal = Refusal {
    id: "blending-on-emulated-formats",
    reason: "The emulated storage is an integer format, which WebGPU does not blend; Metal blends the packed and XR \
             formats at their own precision.",
    source: "tables/formats.toml, the empty blendable cells of r16uint, r32uint and rgba16uint (WebGPU 26.1.1)",
};

const SRGB_SHIM_FILTER: Refusal = Refusal {
    id: "filtering-of-shader-srgb-formats",
    reason: "MSL converts sRGB to linear before the sampler's filter; the stored unorm codes would be filtered before \
             the shader converts them. Filtering needs the shader's own four loads and weights.",
    source: "MSL 4.1 section 8.7.7, quoted in tables/formats.toml question 6.3-srgb-filter-order",
};

const SRGB_SHIM_BLEND: Refusal = Refusal {
    id: "blending-on-shader-srgb-formats",
    reason: "Metal blends sRGB render targets in linear space; the stored unorm codes would be blended encoded.",
    source: "MSL 4.1 section 8.7.7",
};

const FILTER_UINT_STORAGE: Refusal = Refusal {
    id: "filtering-of-uint-stored-formats",
    reason: "The emulated storage is an integer format, which WebGPU cannot filter; filtering needs the shader's own \
             four loads and weights.",
    source: "WebGPU 26.1.1 (uint formats have sample type \"uint\" only)",
};

fn row(value: u32) -> Option<&'static MetalFormatRow> {
    METAL_FORMATS.iter().find(|r| r.value == value)
}

/// The A10X profile's record of a format.
pub fn a10x_record(value: u32) -> Option<&'static A10xFormatRecord> {
    A10X_FORMATS.iter().find(|r| r.value == value)
}

fn decoded(scheme: Scheme, storage: WebGpuFormat, read: Read) -> Emulation {
    Emulation { storage, upload: Upload::Decode(scheme), read, write: Write::NotWritable }
}

/// The software path of a compressed format: decoded on upload from the blocks.
fn compressed_emulation(value: u32) -> Option<Fallback> {
    let scheme = decode::scheme(value)?;
    Some(match scheme {
        Scheme::Etc2Rgb8 { srgb } | Scheme::Etc2Rgb8A1 { srgb } | Scheme::EacRgba8 { srgb } => Fallback::Emulated(
            decoded(scheme, if srgb { WebGpuFormat::Rgba8unormSrgb } else { WebGpuFormat::Rgba8unorm }, Read::Native),
        ),
        Scheme::Eac11 { .. } => Fallback::Refused(EAC_R11),
        Scheme::Astc { srgb: true, .. } => {
            Fallback::Emulated(decoded(scheme, WebGpuFormat::Rgba8unormSrgb, Read::Native))
        }
        Scheme::Astc { srgb: false, .. } => {
            Fallback::Emulated(decoded(scheme, WebGpuFormat::Rgba16uint, Read::Shim(Shim::AstcUnorm12)))
        }
        Scheme::Pvrtc { srgb, .. } => Fallback::Emulated(decoded(
            scheme,
            if srgb { WebGpuFormat::Rgba8unormSrgb } else { WebGpuFormat::Rgba8unorm },
            Read::Native,
        )),
    })
}

/// The static handling of `MTLPixelFormat` `value`.
pub fn handling(value: u32) -> Handling {
    let known = a10x_record(value).is_some_and(|r| r.supported);
    let Some(row) = row(value) else {
        return if let Some(r) = a10x_record(value) {
            match value {
                185 | 191 | 203 | 209 | 221 | 227 if r.compressed => Handling::Refused(ASTC_PRIVATE_FOOTPRINT),
                _ => Handling::Refused(PRIVATE_FORMAT),
            }
        } else {
            Handling::NotOnDevice(
                "MTLPixelFormatGetInfoForDevice returns the invalid record for this value on the A10X and the SDK \
                 does not name it",
            )
        };
    };
    match row.class {
        FormatClass::Sentinel => return Handling::NotOnDevice("MTLPixelFormatInvalid is not a texture format"),
        FormatClass::NotInIos17_7 => {
            return Handling::NotOnDevice("the iOS 17.7 SDK does not offer this format (tables/formats.toml)");
        }
        _ => {}
    }
    if !known {
        // In the SDK for iOS 17.7 but unknown to the A10X: the BC formats (isBCTextureCompressionSupported NO).
        return Handling::NotOnDevice(
            "MTLPixelFormatGetInfoForDevice returns the invalid record for this format on the A10X",
        );
    }
    match row.class {
        FormatClass::SameFormat => {
            let format = row.webgpu.expect("same-format rows name their WebGPU format");
            let without_feature = match format.caps().feature {
                None => None,
                Some(Feature::TextureCompressionEtc2 | Feature::TextureCompressionAstc) => compressed_emulation(value),
                Some(Feature::TextureFormatsTier1) => Some(Fallback::Refused(UNORM16_WITHOUT_TIER1)),
                Some(Feature::Depth32floatStencil8) => Some(Fallback::Refused(DEPTH32_STENCIL8_WITHOUT_FEATURE)),
                Some(Feature::CoreFeaturesAndLimits) => Some(Fallback::Refused(COMPATIBILITY_DEVICE)),
                Some(f) => unreachable!("no A10X format needs {f:?}"),
            };
            Handling::Native { format, without_feature }
        }
        FormatClass::StencilAspectView => Handling::StencilAspect,
        FormatClass::NoCounterpart => no_counterpart(value),
        FormatClass::Sentinel | FormatClass::NotInIos17_7 => unreachable!(),
    }
}

/// The formats `tables/formats.toml` classifies as having no WebGPU counterpart.
fn no_counterpart(value: u32) -> Handling {
    use WebGpuFormat as W;
    let emulated = |storage, upload, read, write| Handling::Emulated(Emulation { storage, upload, read, write });
    match value {
        // A8Unorm: the alpha byte in r8unorm, read as (0, 0, 0, r).
        1 => emulated(
            W::R8unorm,
            Upload::Bytes,
            Read::Swizzle([Swizzle::Zero, Swizzle::Zero, Swizzle::Zero, Swizzle::R]),
            Write::AlphaToRed,
        ),
        // R8Unorm_sRGB, RG8Unorm_sRGB: the codes in r8unorm / rg8unorm with the sRGB conversion in the shader.
        11 => emulated(
            W::R8unorm,
            Upload::Bytes,
            Read::Shim(Shim::Srgb8 { channels: 1 }),
            Write::Shim(Shim::Srgb8 { channels: 1 }),
        ),
        31 => emulated(
            W::Rg8unorm,
            Upload::Bytes,
            Read::Shim(Shim::Srgb8 { channels: 2 }),
            Write::Shim(Shim::Srgb8 { channels: 2 }),
        ),
        // The packed 16-bit formats: the word in r16uint. No WebGPU unorm format holds c / 31, c / 63 or c / 15
        // within the 1.5 ulp of MSL 4.1 section 8.7.1.1.
        40..=43 => {
            let layout = packed::layout(value).unwrap();
            emulated(W::R16uint, Upload::Bytes, Read::Shim(Shim::Packed(layout)), Write::Shim(Shim::Packed(layout)))
        }
        // BGR10A2Unorm: rgb10a2unorm with red and blue exchanged at the boundary; reads, writes and blending are the
        // format's own.
        94 => emulated(W::Rgb10a2unorm, Upload::SwapRedBlue10, Read::Native, Write::Native),
        // BGRA10_XR, BGR10_XR: the words in uint storage with the decode in the shader. The sRGB variants have no
        // stated transfer function.
        552 | 554 => {
            let layout = packed::layout(value).unwrap();
            let storage = if value == 552 { W::Rgba16uint } else { W::R32uint };
            emulated(
                storage,
                Upload::Bytes,
                Read::Shim(Shim::ExtendedRange(layout)),
                Write::Shim(Shim::ExtendedRange(layout)),
            )
        }
        553 | 555 => {
            let storage = if value == 553 { W::Rgba16uint } else { W::R32uint };
            emulated(storage, Upload::Bytes, Read::Refused(XR_SRGB), Write::Refused(XR_SRGB))
        }
        // PVRTC: decoded from the Morton-ordered words.
        160..=167 => match compressed_emulation(value) {
            Some(Fallback::Emulated(e)) => Handling::Emulated(e),
            _ => unreachable!("PVRTC always decodes"),
        },
        // ASTC HDR.
        222..=236 => Handling::Refused(ASTC_HDR),
        // GBGR422, BGRG422: (Y, C) byte pairs in rg8uint; reads need the chroma rule.
        240 | 241 => emulated(W::Rg8uint, Upload::Bytes, Read::Refused(CHROMA_422), Write::Refused(CHROMA_422)),
        _ => unreachable!("tables/formats.toml lists no other no-counterpart format"),
    }
}

/// The format as a host with `features` provides it.
pub fn resolve(value: u32, features: HostFeatures) -> Result<Resolution, Unavailable> {
    match handling(value) {
        Handling::NotOnDevice(why) => Err(Unavailable::NotOnDevice(why)),
        Handling::Refused(r) => Err(Unavailable::Refused(r)),
        Handling::StencilAspect => {
            if features.has(Feature::Depth32floatStencil8) {
                Ok(Resolution::StencilAspect)
            } else {
                Err(Unavailable::Refused(DEPTH32_STENCIL8_WITHOUT_FEATURE))
            }
        }
        Handling::Emulated(e) => Ok(Resolution::Emulated(e)),
        Handling::Native { format, without_feature } => {
            if features.supports(format) {
                Ok(Resolution::Native(format))
            } else {
                match without_feature.expect("a native format without a required feature is always supported") {
                    Fallback::Emulated(e) => Ok(Resolution::Emulated(e)),
                    Fallback::Refused(r) => Err(Unavailable::Refused(r)),
                }
            }
        }
    }
}

/// The dimension of a texture, for the `"3d"` capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dimension {
    /// 1D (as 2D with height 1), 2D, 2D array, cube, cube array.
    Planar,
    ThreeD,
}

/// [`resolve`] for a texture of `dimension`: a native compressed format whose `"3d"` capability the host lacks takes
/// its software path, which stores an uncompressed format.
pub fn resolve_texture(value: u32, features: HostFeatures, dimension: Dimension) -> Result<Resolution, Unavailable> {
    let r = resolve(value, features)?;
    if let (Resolution::Native(f), Dimension::ThreeD) = (r, dimension)
        && !features.allows(f.caps().three_d)
    {
        return match compressed_emulation(value) {
            Some(Fallback::Emulated(e)) => Ok(Resolution::Emulated(e)),
            Some(Fallback::Refused(r)) => Err(Unavailable::Refused(r)),
            None => Err(Unavailable::Refused(Refusal {
                id: "3d-texture-format",
                reason: "WebGPU does not allow this format in a \"3d\" texture.",
                source: "tables/formats.toml, the \"3d\" column of the format",
            })),
        };
    }
    Ok(r)
}

/// The Metal texture usages (`MTLTextureUsage`) a texture is created with, and its sample count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub shader_read: bool,
    pub shader_write: bool,
    pub render_target: bool,
    pub sample_count: u32,
}

fn webgpu_refusal(cell: Cond, what: &'static str) -> Refusal {
    Refusal {
        id: what,
        reason: match cell {
            Cond::Never => "WebGPU does not offer this capability for the texture's format.",
            _ => "WebGPU offers this capability for the texture's format only with a feature the host lacks.",
        },
        source: "tables/formats.toml, the format's capability cells (WebGPU section 26.1)",
    }
}

/// Can the host create the texture with these usages? A capability Metal grants (the guest's Metal validated the
/// request against the A10X) and the host lacks is refused here, at the creating call.
pub fn check_usage(r: Resolution, usage: Usage, features: HostFeatures) -> Result<(), Refusal> {
    let caps = r.storage().caps();
    match usage.sample_count {
        0 | 1 => {}
        4 => {
            if !features.allows(caps.msaa) {
                return Err(webgpu_refusal(caps.msaa, "multisampling"));
            }
        }
        2 => {
            return Err(Refusal {
                id: "msaa-2x",
                reason: "WebGPU multisample count must be 1 or 4.",
                source: "tables/refusals.md, msaa-2x",
            });
        }
        _ => {
            return Err(Refusal {
                id: "sample-count",
                reason: "The A10X supports sample counts 1, 2 and 4 only.",
                source: "profile/ipados-17.7.11-21H461-a10x.toml, queries.supportsTextureSampleCount:",
            });
        }
    }
    if usage.render_target {
        if let Resolution::Emulated(e) = r {
            match e.write {
                Write::Refused(x) => return Err(x),
                Write::NotWritable => {
                    return Err(Refusal {
                        id: "compressed-render-target",
                        reason: "Compressed formats are not render targets.",
                        source: "WebGPU 26.1.3; Metal decodes these formats only for reads",
                    });
                }
                _ => {}
            }
        }
        if !features.allows(caps.render) {
            return Err(webgpu_refusal(caps.render, "render-attachment"));
        }
    }
    if usage.shader_write {
        if let Resolution::Emulated(e) = r
            && let Write::Refused(x) = e.write
        {
            return Err(x);
        }
        if !features.allows(caps.storage_wo) {
            return Err(webgpu_refusal(caps.storage_wo, "storage-binding"));
        }
    }
    if usage.shader_read
        && let Resolution::Emulated(Emulation { read: Read::Refused(x), .. }) = r
    {
        return Err(x);
    }
    Ok(())
}

/// Can a sampler with a linear filter (min, mag or mip) sample the texture on this host?
pub fn check_filtering(r: Resolution, features: HostFeatures) -> Result<(), Refusal> {
    match r {
        Resolution::Emulated(Emulation { read: Read::Refused(x), .. }) => Err(x),
        Resolution::Emulated(Emulation { read: Read::Shim(Shim::Srgb8 { .. }), .. }) => Err(SRGB_SHIM_FILTER),
        Resolution::Emulated(Emulation { read: Read::Shim(_), storage, .. })
            if matches!(storage.caps().sample, SampleType::Uint | SampleType::Sint) =>
        {
            Err(FILTER_UINT_STORAGE)
        }
        _ => {
            let f = r.storage();
            if features.filterable(f) {
                Ok(())
            } else if matches!(
                f,
                WebGpuFormat::R16unorm
                    | WebGpuFormat::Rg16unorm
                    | WebGpuFormat::Rgba16unorm
                    | WebGpuFormat::R16snorm
                    | WebGpuFormat::Rg16snorm
                    | WebGpuFormat::Rgba16snorm
            ) {
                Err(Refusal {
                    id: "unorm16-filtering",
                    reason: "These formats need texture-formats-tier1 and are only unfilterable-float.",
                    source: "tables/refusals.md, unorm16-filtering",
                })
            } else {
                Err(Refusal {
                    id: "filtering",
                    reason: "The texture's WebGPU format is not filterable on this host.",
                    source: "tables/formats.toml, the format's GPUTextureSampleType cell (WebGPU section 26.1)",
                })
            }
        }
    }
}

/// Can a render pipeline blend into the texture's format on this host?
pub fn check_blending(r: Resolution, features: HostFeatures) -> Result<(), Refusal> {
    if let Resolution::Emulated(e) = r {
        match e.write {
            Write::Native => {}
            Write::Refused(x) => return Err(x),
            Write::NotWritable => return Err(webgpu_refusal(Cond::Never, "blending")),
            Write::AlphaToRed => {}
            Write::Shim(Shim::Srgb8 { .. }) => return Err(SRGB_SHIM_BLEND),
            Write::Shim(_) => return Err(PACKED_BLEND),
        }
    }
    let caps = r.storage().caps();
    if features.allows(caps.blend) { Ok(()) } else { Err(webgpu_refusal(caps.blend, "blending")) }
}

/// Can a multisampled texture of the format be resolved into on this host?
pub fn check_resolve(r: Resolution, features: HostFeatures) -> Result<(), Refusal> {
    if let Resolution::Emulated(e) = r
        && !matches!(e.write, Write::Native)
    {
        return Err(Refusal {
            id: "resolve-of-emulated-format",
            reason: "The emulated storage does not hold Metal's values in a resolvable form.",
            source: "tables/formats.toml",
        });
    }
    let caps = r.storage().caps();
    if features.allows(caps.resolve) { Ok(()) } else { Err(webgpu_refusal(caps.resolve, "resolve")) }
}

/// The formats this module answers for: every row of `tables/formats.toml` and every format the A10X profile knows.
pub fn all_values() -> Vec<u32> {
    let mut v: Vec<u32> = METAL_FORMATS.iter().map(|r| r.value).chain(A10X_FORMATS.iter().map(|r| r.value)).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Whether the profile's capability getters agree with the classification (checked by the tests): BC formats unknown
/// and unsupported, ASTC LDR supported, ASTC HDR not advertised.
pub fn getters_consistent() -> bool {
    !A10X_GETTERS.supports_bc_texture_compression
        && !A10X_GETTERS.is_bc_texture_compression_supported
        && A10X_GETTERS.supports_astc_texture_compression
        && !A10X_GETTERS.supports_astc_hdr_texture_compression
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPAT_FLOAT: u32 = 1 << 16;
    const COMPAT_HALF: u32 = 1 << 17;
    const COMPAT_INT: u32 = 1 << 18;
    const COMPAT_UINT: u32 = 1 << 19;

    #[test]
    fn every_reported_format_is_mapped_emulated_or_refused() {
        assert!(getters_consistent());
        let (mut native, mut emulated, mut refused, mut stencil) = (0, 0, 0, 0);
        for r in A10X_FORMATS.iter() {
            assert!(r.supported, "the profile lists only formats with the supported bit");
            match handling(r.value) {
                Handling::Native { .. } => native += 1,
                Handling::Emulated(_) => emulated += 1,
                Handling::StencilAspect => stencil += 1,
                Handling::Refused(x) => {
                    assert!(!x.reason.is_empty() && !x.source.is_empty());
                    refused += 1;
                }
                Handling::NotOnDevice(why) => panic!("{}: reported by the A10X but {why}", r.value),
            }
        }
        assert_eq!(native + emulated + refused + stencil, A10X_FORMATS.len());
        // Formats of tables/formats.toml the A10X does not know are the device's own rejections.
        for row in METAL_FORMATS.iter() {
            if a10x_record(row.value).is_none() {
                assert!(matches!(handling(row.value), Handling::NotOnDevice(_)), "{}", row.name);
            }
        }
    }

    /// The refusals that name a row of tables/refusals.md name an existing row.
    #[test]
    fn cited_refusal_rows_exist() {
        let md = std::fs::read_to_string(super::super::testdata::repo_path("tables/refusals.md")).unwrap();
        for v in all_values() {
            let r = match handling(v) {
                Handling::Native { without_feature: Some(Fallback::Refused(r)), .. } | Handling::Refused(r) => r,
                _ => continue,
            };
            if r.source.starts_with("tables/refusals.md") {
                assert!(md.contains(&format!("| {} |", r.id)), "{}", r.id);
            }
        }
        assert!(md.contains("| msaa-2x |") && md.contains("| unorm16-filtering |"));
    }

    /// The profile's flag words against the handling: compressed formats decode or are refused, and the shader types
    /// Metal allows for a format match how the host reads the stored texture.
    #[test]
    fn profile_flags_agree_with_the_handling() {
        let all = HostFeatures::new(Feature::ALL);
        for r in A10X_FORMATS.iter() {
            let h = handling(r.value);
            let compressed_handling = match h {
                Handling::Native { format, .. } => format.is_compressed(),
                Handling::Emulated(Emulation { upload: Upload::Decode(_), .. }) => true,
                Handling::Emulated(Emulation { storage: WebGpuFormat::Rg8uint, .. }) => true, // the 422 formats
                Handling::Refused(x) => x.id.starts_with("astc") || r.compressed,
                _ => false,
            };
            assert_eq!(r.compressed, compressed_handling, "format {}", r.value);
            let Ok(res) = resolve(r.value, all) else { continue };
            let sample = match res {
                Resolution::StencilAspect => SampleType::Uint,
                _ => match res.storage().caps().aspects {
                    [] => res.storage().caps().sample,
                    [a, ..] => a.sample,
                },
            };
            let shim = matches!(res, Resolution::Emulated(Emulation { read: Read::Shim(_) | Read::Refused(_), .. }));
            let float = r.word_08 & (COMPAT_FLOAT | COMPAT_HALF) != 0;
            let int = r.word_08 & COMPAT_INT != 0;
            let uint = r.word_08 & COMPAT_UINT != 0;
            if shim {
                // The shader converts integer storage to the float values Metal reads.
                assert!(float && !int && !uint, "format {}", r.value);
                continue;
            }
            match sample {
                SampleType::Float | SampleType::FloatIf(_) | SampleType::UnfilterableFloat | SampleType::Depth => {
                    assert!(float && !int && !uint, "format {}: {sample:?}", r.value)
                }
                SampleType::Uint => assert!(uint && !float && !int, "format {}", r.value),
                SampleType::Sint => assert!(int && !float && !uint, "format {}", r.value),
            }
        }
    }

    #[test]
    fn host_features_select_the_path() {
        let none = HostFeatures::NONE;
        let all = HostFeatures::new(Feature::ALL);
        // ETC2_RGB8: native with the feature, decoded without.
        assert_eq!(resolve(180, all), Ok(Resolution::Native(WebGpuFormat::Etc2Rgb8unorm)));
        assert!(matches!(
            resolve(180, none),
            Ok(Resolution::Emulated(Emulation { storage: WebGpuFormat::Rgba8unorm, .. }))
        ));
        // EAC_R11Unorm without the feature: the refusal of tables/refusals.md.
        assert!(matches!(resolve(170, none), Err(Unavailable::Refused(Refusal { id: "eac-r11-software-path", .. }))));
        // ASTC_4x4_LDR in a 3D texture without the sliced-3D feature takes the software path.
        let astc = HostFeatures::new([Feature::TextureCompressionAstc]);
        assert!(matches!(
            resolve_texture(204, astc, Dimension::ThreeD),
            Ok(Resolution::Emulated(Emulation { storage: WebGpuFormat::Rgba16uint, .. }))
        ));
        assert_eq!(resolve_texture(204, astc, Dimension::Planar), Ok(Resolution::Native(WebGpuFormat::Astc4x4Unorm)));
        // R16Unorm without texture-formats-tier1.
        assert!(matches!(resolve(20, none), Err(Unavailable::Refused(_))));
        // BC is not a format of the A10X.
        assert!(matches!(resolve(130, all), Err(Unavailable::NotOnDevice(_))));
        // X32_Stencil8.
        assert_eq!(resolve(261, all), Ok(Resolution::StencilAspect));
    }

    #[test]
    fn usage_checks() {
        let none = HostFeatures::NONE;
        let rgba8 = resolve(70, none).unwrap();
        assert!(check_usage(rgba8, Usage { render_target: true, sample_count: 4, ..Usage::default() }, none).is_ok());
        assert_eq!(check_usage(rgba8, Usage { sample_count: 2, ..Usage::default() }, none).unwrap_err().id, "msaa-2x");
        // RG11B10Float renders only with rg11b10ufloat-renderable.
        let rg11 = resolve(92, none).unwrap();
        let rt = Usage { render_target: true, ..Usage::default() };
        assert!(check_usage(rg11, rt, none).is_err());
        assert!(check_usage(rg11, rt, HostFeatures::new([Feature::Rg11b10ufloatRenderable])).is_ok());
        // R32Float filters only with float32-filterable.
        let r32f = resolve(55, none).unwrap();
        assert!(check_filtering(r32f, none).is_err());
        assert!(check_filtering(r32f, HostFeatures::new([Feature::Float32Filterable])).is_ok());
        // B5G6R5 is stored as r16uint: no native filtering or blending.
        let b5 = resolve(40, none).unwrap();
        assert_eq!(check_filtering(b5, none).unwrap_err().id, "filtering-of-uint-stored-formats");
        assert_eq!(check_blending(b5, none).unwrap_err().id, "blending-on-emulated-formats");
        // BGR10A2Unorm blends natively as rgb10a2unorm.
        assert!(check_blending(resolve(94, none).unwrap(), none).is_ok());
        // Reading a 422 texture is refused; creating it for upload and readback is not.
        let gbgr = resolve(240, none).unwrap();
        assert!(check_usage(gbgr, Usage::default(), none).is_ok());
        assert_eq!(
            check_usage(gbgr, Usage { shader_read: true, ..Usage::default() }, none).unwrap_err().id,
            "422-chroma-reconstruction"
        );
    }
}
