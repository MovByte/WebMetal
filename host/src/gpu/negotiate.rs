//! Feature and limit negotiation.
//!
//! The executor asks for everything the adapter offers: every feature of [`FEATURES`] the adapter supports and, for
//! every limit of [`LIMITS`], the adapter's own value (the best one it can request). [`plan`] builds that request
//! and, by running the specification's "a new device" steps (dependencies, default feature level, limit
//! normalization), the features and limits the device must then have; [`verify`] checks the device against them.
//!
//! A limit the implementation does not expose (an older browser, or a header without the field) is not requested and
//! is recorded as not exposed; it is never given a substitute value. An adapter that breaks the specification's
//! capability guarantees (a limit worse than the default, an alignment that is not a power of two, a feature without
//! the features it requires) is refused, because nothing the executor derived from it could be trusted.

use std::collections::BTreeSet;
use std::fmt;

use super::idl::{DeviceDescriptor, FeatureName, SupportedLimits};
use super::num::Size64;
use super::tables::{FEATURES, LIMITS, Limit, LimitClass};

/// One limit's negotiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LimitRecord {
    pub limit: &'static Limit,
    /// The adapter's value; `None` if the implementation does not expose the limit.
    pub adapter: Option<u64>,
    /// The value placed in `requiredLimits`; `None` if the limit is not requested.
    pub requested: Option<u64>,
    /// The device's value per "a new device"; `None` if the implementation does not expose the limit.
    pub expected: Option<u64>,
}

/// One feature's negotiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureRecord {
    pub feature: FeatureName,
    /// The adapter supports it.
    pub offered: bool,
    /// It is in `requiredFeatures`.
    pub requested: bool,
    /// The device will have it (requested, a dependency of a requested feature, or a default of the feature level).
    pub expected: bool,
}

/// The device request and the device it must produce.
#[derive(Clone, Debug)]
pub struct Plan {
    pub descriptor: DeviceDescriptor,
    pub limits: Vec<LimitRecord>,
    pub features: Vec<FeatureRecord>,
    /// Feature names the adapter reports that this IDL does not define. They cannot be requested by name through
    /// `GPUFeatureName`; they are listed so a report shows them.
    pub unknown_adapter_features: Vec<String>,
}

impl Plan {
    pub fn expected_features(&self) -> BTreeSet<String> {
        self.features.iter().filter(|f| f.expected).map(|f| f.feature.as_str().to_string()).collect()
    }

    pub fn expected_limits(&self) -> SupportedLimits {
        let mut l = SupportedLimits::default();
        for r in &self.limits {
            let ok = l.set(r.limit.name, r.expected);
            debug_assert!(ok);
        }
        l
    }

    pub fn limit(&self, name: &str) -> Option<&LimitRecord> {
        self.limits.iter().find(|r| r.limit.name == name)
    }
}

/// Why an adapter cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NegotiationError {
    /// A limit is worse than the default (Adapter Capability Guarantees, §4.2.1).
    LimitBelowDefault { name: &'static str, adapter: u64, default: u64 },
    /// An alignment limit is not a power of two below 2^32.
    AlignmentNotPowerOfTwo { name: &'static str, adapter: u64 },
    /// The adapter offers a feature without a feature it requires (§3.6.1.2, §4.2.1).
    MissingDependency { feature: FeatureName, requires: FeatureName },
    /// A core adapter (feature level "core", which the executor requests) must offer "core-features-and-limits".
    NotCore,
}

impl fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NegotiationError::LimitBelowDefault { name, adapter, default } => {
                write!(f, "adapter limit {name} = {adapter} is worse than the default {default}")
            }
            NegotiationError::AlignmentNotPowerOfTwo { name, adapter } => {
                write!(f, "adapter limit {name} = {adapter} is not a power of two below 2^32")
            }
            NegotiationError::MissingDependency { feature, requires } => {
                write!(f, "adapter offers {} without {}", feature.as_str(), requires.as_str())
            }
            NegotiationError::NotCore => write!(f, "adapter does not offer core-features-and-limits"),
        }
    }
}

impl std::error::Error for NegotiationError {}

fn better(class: LimitClass, a: u64, b: u64) -> bool {
    match class {
        LimitClass::Maximum => a > b,
        LimitClass::Alignment => a < b,
    }
}

/// Build the device request for an adapter returned for feature level "core".
pub fn plan(adapter_features: &BTreeSet<String>, adapter_limits: &SupportedLimits) -> Result<Plan, NegotiationError> {
    // Adapter capability guarantees.
    for l in LIMITS.iter() {
        if let Some(v) = adapter_limits.get(l.name) {
            if better(l.class, l.default, v) {
                return Err(NegotiationError::LimitBelowDefault { name: l.name, adapter: v, default: l.default });
            }
            if l.class == LimitClass::Alignment && (!v.is_power_of_two() || v >= 1 << 32) {
                return Err(NegotiationError::AlignmentNotPowerOfTwo { name: l.name, adapter: v });
            }
        }
    }
    for f in FEATURES.iter() {
        if adapter_features.contains(f.name.as_str()) {
            for r in f.requires {
                if !adapter_features.contains(r.as_str()) {
                    return Err(NegotiationError::MissingDependency { feature: f.name, requires: *r });
                }
            }
        }
    }
    if !adapter_features.contains(FeatureName::CoreFeaturesAndLimits.as_str()) {
        return Err(NegotiationError::NotCore);
    }

    // requiredFeatures: everything offered, in IDL order.
    let mut descriptor = DeviceDescriptor::new();
    for f in FEATURES.iter() {
        if adapter_features.contains(f.name.as_str()) {
            descriptor.required_features.push(f.name);
        }
    }
    // requiredLimits: the adapter's value of every exposed limit, in table order.
    for l in LIMITS.iter() {
        if let Some(v) = adapter_limits.get(l.name) {
            descriptor.required_limits.push((l.name.to_string(), Some(Size64::new(v).expect("limits fit GPUSize64"))));
        }
    }

    // "a new device": features.
    let mut features: BTreeSet<FeatureName> = descriptor.required_features.iter().copied().collect();
    // The default features of feature level "core".
    features.insert(FeatureName::CoreFeaturesAndLimits);
    loop {
        let before = features.len();
        for f in FEATURES.iter() {
            if features.contains(&f.name) {
                features.extend(f.requires.iter().copied());
            }
        }
        if features.len() == before {
            break;
        }
    }

    // "a new device": limits, starting from the core defaults.
    let mut expected: Vec<(&'static Limit, Option<u64>)> = LIMITS.iter().map(|l| (l, Some(l.default))).collect();
    for (name, value) in &descriptor.required_limits {
        if let Some(v) = value {
            let e = expected.iter_mut().find(|(l, _)| l.name == name).unwrap();
            if better(e.0.class, v.get(), e.1.unwrap()) {
                e.1 = Some(v.get());
            }
        }
    }
    let get = |e: &Vec<(&'static Limit, Option<u64>)>, n: &str| e.iter().find(|(l, _)| l.name == n).unwrap().1.unwrap();
    let set = |e: &mut Vec<(&'static Limit, Option<u64>)>, n: &str, v: u64| {
        e.iter_mut().find(|(l, _)| l.name == n).unwrap().1 = Some(v)
    };
    let sb = get(&expected, "maxStorageBuffersPerShaderStage")
        .max(get(&expected, "maxStorageBuffersInVertexStage"))
        .max(get(&expected, "maxStorageBuffersInFragmentStage"));
    set(&mut expected, "maxStorageBuffersPerShaderStage", sb);
    let st = get(&expected, "maxStorageTexturesPerShaderStage")
        .max(get(&expected, "maxStorageTexturesInVertexStage"))
        .max(get(&expected, "maxStorageTexturesInFragmentStage"));
    set(&mut expected, "maxStorageTexturesPerShaderStage", st);
    if features.contains(&FeatureName::CoreFeaturesAndLimits) {
        set(&mut expected, "maxStorageBuffersInVertexStage", sb);
        set(&mut expected, "maxStorageBuffersInFragmentStage", sb);
        set(&mut expected, "maxStorageTexturesInVertexStage", st);
        set(&mut expected, "maxStorageTexturesInFragmentStage", st);
    }

    let limits = LIMITS
        .iter()
        .zip(expected)
        .map(|(l, (_, e))| {
            let adapter = adapter_limits.get(l.name);
            LimitRecord { limit: l, adapter, requested: adapter, expected: if adapter.is_some() { e } else { None } }
        })
        .collect();
    let feature_records = FEATURES
        .iter()
        .map(|f| FeatureRecord {
            feature: f.name,
            offered: adapter_features.contains(f.name.as_str()),
            requested: descriptor.required_features.contains(&f.name),
            expected: features.contains(&f.name),
        })
        .collect();
    let unknown_adapter_features =
        adapter_features.iter().filter(|n| FeatureName::from_idl(n).is_none()).cloned().collect();
    Ok(Plan { descriptor, limits, features: feature_records, unknown_adapter_features })
}

/// A difference between the device and the plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mismatch {
    Feature { name: String, expected: bool, device: bool },
    Limit { name: &'static str, expected: Option<u64>, device: Option<u64> },
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mismatch::Feature { name, expected, device } => {
                write!(f, "feature {name}: device has it = {device}, \"a new device\" gives {expected}")
            }
            Mismatch::Limit { name, expected, device } => {
                write!(f, "limit {name}: device {device:?}, \"a new device\" gives {expected:?}")
            }
        }
    }
}

/// Compare a device's features and limits with the plan it was requested by.
pub fn verify(plan: &Plan, device_features: &BTreeSet<String>, device_limits: &SupportedLimits) -> Vec<Mismatch> {
    let mut out = Vec::new();
    let expected = plan.expected_features();
    for name in expected.union(device_features) {
        let e = expected.contains(name);
        let d = device_features.contains(name);
        if e != d {
            out.push(Mismatch::Feature { name: name.clone(), expected: e, device: d });
        }
    }
    for r in &plan.limits {
        let d = device_limits.get(r.limit.name);
        if d != r.expected {
            out.push(Mismatch::Limit { name: r.limit.name, expected: r.expected, device: d });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> SupportedLimits {
        let mut l = SupportedLimits::default();
        for d in LIMITS.iter() {
            assert!(l.set(d.name, Some(d.default)));
        }
        l
    }

    fn core() -> BTreeSet<String> {
        ["core-features-and-limits".to_string()].into()
    }

    #[test]
    fn default_adapter_gives_default_device() {
        let p = plan(&core(), &defaults()).unwrap();
        assert_eq!(p.descriptor.required_features, vec![FeatureName::CoreFeaturesAndLimits]);
        assert_eq!(p.expected_limits(), defaults());
        assert!(verify(&p, &core(), &defaults()).is_empty());
    }

    #[test]
    fn requests_adapter_maxima_and_alignments() {
        let mut a = defaults();
        a.max_texture_dimension_2d = Some(16384);
        a.min_uniform_buffer_offset_alignment = Some(32);
        a.max_buffer_size = Some(1 << 32);
        let p = plan(&core(), &a).unwrap();
        assert_eq!(p.limit("maxTextureDimension2D").unwrap().expected, Some(16384));
        assert_eq!(p.limit("minUniformBufferOffsetAlignment").unwrap().expected, Some(32));
        assert_eq!(p.limit("maxBufferSize").unwrap().expected, Some(1 << 32));
        let req: Vec<_> = p.descriptor.required_limits.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(req.len(), LIMITS.len());
    }

    #[test]
    fn storage_limits_are_normalized_on_core_devices() {
        let mut a = defaults();
        a.max_storage_buffers_per_shader_stage = Some(8);
        a.max_storage_buffers_in_vertex_stage = Some(10);
        a.max_storage_buffers_in_fragment_stage = Some(9);
        a.max_storage_textures_per_shader_stage = Some(8);
        let p = plan(&core(), &a).unwrap();
        for n in
            ["maxStorageBuffersPerShaderStage", "maxStorageBuffersInVertexStage", "maxStorageBuffersInFragmentStage"]
        {
            assert_eq!(p.limit(n).unwrap().expected, Some(10), "{n}");
        }
        for n in
            ["maxStorageTexturesPerShaderStage", "maxStorageTexturesInVertexStage", "maxStorageTexturesInFragmentStage"]
        {
            assert_eq!(p.limit(n).unwrap().expected, Some(8), "{n}");
        }
    }

    #[test]
    fn dependencies_are_enabled_and_checked() {
        let mut f = core();
        f.insert("texture-formats-tier2".into());
        assert_eq!(
            plan(&f, &defaults()).unwrap_err(),
            NegotiationError::MissingDependency {
                feature: FeatureName::TextureFormatsTier2,
                requires: FeatureName::TextureFormatsTier1
            }
        );
        f.insert("texture-formats-tier1".into());
        f.insert("rg11b10ufloat-renderable".into());
        let p = plan(&f, &defaults()).unwrap();
        assert!(p.expected_features().contains("rg11b10ufloat-renderable"));
    }

    #[test]
    fn unexposed_limits_are_not_requested() {
        let mut a = defaults();
        a.max_immediate_size = None;
        let p = plan(&core(), &a).unwrap();
        let r = p.limit("maxImmediateSize").unwrap();
        assert_eq!((r.adapter, r.requested, r.expected), (None, None, None));
        assert!(!p.descriptor.required_limits.iter().any(|(k, _)| k == "maxImmediateSize"));
    }

    #[test]
    fn broken_adapters_are_refused() {
        let mut a = defaults();
        a.max_bind_groups = Some(3);
        assert!(matches!(plan(&core(), &a), Err(NegotiationError::LimitBelowDefault { name: "maxBindGroups", .. })));
        let mut a = defaults();
        a.min_storage_buffer_offset_alignment = Some(48);
        assert!(matches!(plan(&core(), &a), Err(NegotiationError::AlignmentNotPowerOfTwo { .. })));
        assert_eq!(plan(&BTreeSet::new(), &defaults()).unwrap_err(), NegotiationError::NotCore);
    }

    #[test]
    fn verify_reports_differences() {
        let p = plan(&core(), &defaults()).unwrap();
        let mut d = defaults();
        d.max_bind_groups = Some(8);
        let mut f = core();
        f.insert("shader-f16".into());
        let m = verify(&p, &f, &d);
        assert_eq!(m.len(), 2);
    }
}
