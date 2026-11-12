//! WebGPU texture format capabilities (WebGPU section 26.1, as `tables/formats.toml` records it) evaluated against the
//! features a host device has.

pub use super::table::{Feature, WebGpuFormat};

/// A capability cell: always available, never, or available with one optional feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cond {
    Always,
    Never,
    Feature(Feature),
}

/// The `GPUTextureSampleType` cell of a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleType {
    /// `"float"` and `"unfilterable-float"`: filterable.
    Float,
    /// `"float"` only with a feature (`float32-filterable`), else `"unfilterable-float"`.
    FloatIf(Feature),
    UnfilterableFloat,
    Uint,
    Sint,
    /// `"depth"` and `"unfilterable-float"`.
    Depth,
}

/// An aspect of a depth-or-stencil format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aspect {
    Depth,
    Stencil,
}

/// The table of section 26.1 a format is listed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    /// 26.1.1 Plain color formats.
    Plain,
    /// 26.1.3 Packed formats (rgb9e5ufloat and the compressed formats).
    Packed,
    /// 26.1.2 Depth-stencil formats.
    DepthStencil,
}

/// One aspect of a depth-or-stencil format (the depth-stencil table of section 26.1.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AspectCaps {
    pub aspect: Aspect,
    pub sample: SampleType,
    pub copy_src: bool,
    pub copy_dst: bool,
    /// Texel block copy footprint in bytes; `None` where the aspect cannot be copied.
    pub footprint: Option<u32>,
    /// The aspect-specific format.
    pub view: WebGpuFormat,
}

/// The capabilities of one WebGPU texture format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatCaps {
    pub name: &'static str,
    pub table: Table,
    /// The feature without which the format cannot be used at all.
    pub feature: Option<Feature>,
    pub sample: SampleType,
    pub render: Cond,
    pub blend: Cond,
    pub msaa: Cond,
    pub resolve: Cond,
    pub storage_wo: Cond,
    pub storage_ro: Cond,
    pub storage_rw: Cond,
    /// `"3d"` texture dimension.
    pub three_d: Cond,
    /// Texel block copy footprint in bytes (0 for depth-or-stencil formats: see `aspects`).
    pub footprint: u32,
    /// Texel block width and height.
    pub block: (u32, u32),
    pub aspects: &'static [AspectCaps],
}

impl WebGpuFormat {
    pub fn caps(self) -> &'static FormatCaps {
        &super::table::WEBGPU_FORMATS[self as usize]
    }

    pub fn from_name(name: &str) -> Option<WebGpuFormat> {
        WebGpuFormat::ALL.into_iter().find(|f| f.name() == name)
    }

    /// Is the format compressed (block size greater than 1×1, section 26.1.3)?
    pub fn is_compressed(self) -> bool {
        self.caps().block != (1, 1)
    }

    /// Do texel copies of this format (or of `aspect` of a depth-or-stencil format) preserve every bit pattern?
    ///
    /// WebGPU section 11.2 (#texel-copies) guarantees only that "valid, finite, non-subnormal numeric values in the
    /// source have the same numeric value in the destination": zero signs, subnormals, NaN and infinity of float
    /// formats may change, and "Packed formats and snorm formats may change bit-representation". The formats whose
    /// every bit pattern is a distinct valid value with one representation, so that value preservation is bit
    /// preservation, are the uint, sint and unorm (including -srgb) formats outside the packed table.
    pub fn texel_copy_exact(self, aspect: Option<Aspect>) -> bool {
        let caps = self.caps();
        match caps.table {
            Table::Packed => false,
            Table::DepthStencil => {
                let a = match (aspect, caps.aspects) {
                    (Some(a), aspects) => aspects.iter().find(|c| c.aspect == a),
                    (None, [only]) => Some(only),
                    (None, _) => None,
                };
                // The stencil aspect is 8-bit uint; the depth aspect is exact when it is a unorm format.
                match a {
                    Some(a) => {
                        a.footprint.is_some() && (a.sample == SampleType::Uint || a.view.caps().name.ends_with("unorm"))
                    }
                    None => false,
                }
            }
            Table::Plain => {
                let n = caps.name;
                n.ends_with("uint") || n.ends_with("sint") || n.ends_with("unorm") || n.ends_with("unorm-srgb")
            }
        }
    }
}

/// The features a WebGPU device was created with, closed over the feature dependencies of section 3.6.1.2.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostFeatures(u64);

impl HostFeatures {
    pub const NONE: HostFeatures = HostFeatures(0);

    /// The features of a device created with `enabled`, plus the features they require.
    pub fn new(enabled: impl IntoIterator<Item = Feature>) -> HostFeatures {
        let mut set = HostFeatures(0);
        let mut stack: Vec<Feature> = enabled.into_iter().collect();
        while let Some(f) = stack.pop() {
            if !set.has(f) {
                set.0 |= 1 << f as u32;
                stack.extend_from_slice(f.requires());
            }
        }
        set
    }

    /// From `GPUFeatureName` strings; an unknown name is returned as the error.
    pub fn from_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<HostFeatures, &'a str> {
        let mut features = Vec::new();
        for n in names {
            features.push(Feature::ALL.into_iter().find(|f| f.name() == n).ok_or(n)?);
        }
        Ok(HostFeatures::new(features))
    }

    pub fn has(self, f: Feature) -> bool {
        self.0 & (1 << f as u32) != 0
    }

    pub fn allows(self, c: Cond) -> bool {
        match c {
            Cond::Always => true,
            Cond::Never => false,
            Cond::Feature(f) => self.has(f),
        }
    }

    /// Can a device with these features create textures of `format`?
    pub fn supports(self, format: WebGpuFormat) -> bool {
        format.caps().feature.is_none_or(|f| self.has(f))
    }

    /// Is `format` filterable (sample type `"float"`) on this device?
    pub fn filterable(self, format: WebGpuFormat) -> bool {
        match format.caps().sample {
            SampleType::Float => true,
            SampleType::FloatIf(f) => self.has(f),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_spot_checks() {
        // tables/formats.toml rows quoted from WebGPU 26.1.1.
        let r32f = WebGpuFormat::R32float.caps();
        assert_eq!(r32f.sample, SampleType::FloatIf(Feature::Float32Filterable));
        assert_eq!(r32f.blend, Cond::Feature(Feature::Float32Blendable));
        assert_eq!(r32f.storage_rw, Cond::Always);
        assert_eq!(WebGpuFormat::R16unorm.caps().feature, Some(Feature::TextureFormatsTier1));
        assert_eq!(WebGpuFormat::Etc2Rgb8unorm.caps().feature, Some(Feature::TextureCompressionEtc2));
        assert_eq!(WebGpuFormat::Astc12x10Unorm.caps().block, (12, 10));
        assert_eq!(WebGpuFormat::Depth32floatStencil8.caps().aspects.len(), 2);
    }

    #[test]
    fn feature_dependencies_are_closed() {
        let f = HostFeatures::new([Feature::TextureFormatsTier2]);
        assert!(f.has(Feature::TextureFormatsTier1));
        assert!(f.has(Feature::Rg11b10ufloatRenderable));
        assert!(!f.has(Feature::Float32Filterable));
        assert_eq!(HostFeatures::from_names(["float32-filterable", "bogus"]), Err("bogus"));
    }

    #[test]
    fn texel_copy_exactness() {
        use WebGpuFormat::*;
        for f in [R8unorm, Rgba8unormSrgb, Bgra8unorm, Rgb10a2unorm, Rgb10a2uint, R32uint, Rgba16sint, Stencil8] {
            assert!(f.texel_copy_exact(None), "{f:?}");
        }
        for f in [R8snorm, R16float, R32float, Rg11b10ufloat, Rgb9e5ufloat, Etc2Rgb8unorm, Astc4x4Unorm, Depth32float] {
            assert!(!f.texel_copy_exact(None), "{f:?}");
        }
        assert!(Depth16unorm.texel_copy_exact(None));
        assert!(Depth32floatStencil8.texel_copy_exact(Some(Aspect::Stencil)));
        assert!(!Depth32floatStencil8.texel_copy_exact(Some(Aspect::Depth)));
        assert!(!Depth24plus.texel_copy_exact(None));
    }
}
