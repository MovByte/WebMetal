//! Metal's resource options (`MTLResourceOptions`) and the storage, CPU cache and hazard-tracking modes they pack.
//!
//! Values and bit positions are those of the iPhoneOS SDK's `MTLResource.h`: `MTLResourceCPUCacheModeShift` 0 (mask
//! 0xf), `MTLResourceStorageModeShift` 4 (mask 0xf), `MTLResourceHazardTrackingModeShift` 8 (mask 0x3);
//! `MTLStorageModeShared` 0, `MTLStorageModeManaged` 1 (`API_UNAVAILABLE(ios)`), `MTLStorageModePrivate` 2,
//! `MTLStorageModeMemoryless` 3; `MTLCPUCacheModeDefaultCache` 0, `MTLCPUCacheModeWriteCombined` 1;
//! `MTLHazardTrackingModeDefault` 0, `MTLHazardTrackingModeUntracked` 1, `MTLHazardTrackingModeTracked` 2.

use super::error::MemError;

pub const CPU_CACHE_MODE_SHIFT: u32 = 0;
pub const CPU_CACHE_MODE_MASK: u64 = 0xf << CPU_CACHE_MODE_SHIFT;
pub const STORAGE_MODE_SHIFT: u32 = 4;
pub const STORAGE_MODE_MASK: u64 = 0xf << STORAGE_MODE_SHIFT;
pub const HAZARD_TRACKING_MODE_SHIFT: u32 = 8;
pub const HAZARD_TRACKING_MODE_MASK: u64 = 0x3 << HAZARD_TRACKING_MODE_SHIFT;

/// `MTLStorageMode`, as iOS offers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageMode {
    /// CPU and GPU share the memory; on this executor guest memory is authoritative for the CPU and a WebGPU object
    /// for the GPU, kept equal by the shadow/diff protocol.
    Shared,
    /// GPU only.
    Private,
    /// Render-pass-only contents (textures).
    Memoryless,
}

impl StorageMode {
    /// The `MTLStorageMode` value.
    pub fn value(self) -> u64 {
        match self {
            StorageMode::Shared => 0,
            StorageMode::Private => 2,
            StorageMode::Memoryless => 3,
        }
    }

    /// The mode of an `MTLStorageMode` value. `MTLStorageModeManaged` is unavailable on iOS (`API_UNAVAILABLE(ios)`)
    /// and other values are not modes; Metal does not define either.
    pub fn from_value(value: u64, call: &'static str) -> Result<StorageMode, MemError> {
        match value {
            0 => Ok(StorageMode::Shared),
            2 => Ok(StorageMode::Private),
            3 => Ok(StorageMode::Memoryless),
            1 => Err(MemError::undefined(call, "MTLStorageModeManaged is API_UNAVAILABLE(ios) (MTLResource.h)")),
            v => Err(MemError::undefined(call, format!("{v} is not an MTLStorageMode value (MTLResource.h)"))),
        }
    }
}

/// `MTLCPUCacheMode`. It changes how the CPU caches the memory, never what the memory holds, so it does not change
/// what the executor does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuCacheMode {
    DefaultCache,
    WriteCombined,
}

impl CpuCacheMode {
    pub fn value(self) -> u64 {
        match self {
            CpuCacheMode::DefaultCache => 0,
            CpuCacheMode::WriteCombined => 1,
        }
    }

    pub fn from_value(value: u64, call: &'static str) -> Result<CpuCacheMode, MemError> {
        match value {
            0 => Ok(CpuCacheMode::DefaultCache),
            1 => Ok(CpuCacheMode::WriteCombined),
            v => Err(MemError::undefined(call, format!("{v} is not an MTLCPUCacheMode value (MTLResource.h)"))),
        }
    }
}

/// `MTLHazardTrackingMode` as requested. WebGPU orders every command, which is stronger than either mode (plan
/// section 6.1), so the mode does not change what the executor does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HazardTrackingMode {
    Default,
    Untracked,
    Tracked,
}

impl HazardTrackingMode {
    pub fn value(self) -> u64 {
        match self {
            HazardTrackingMode::Default => 0,
            HazardTrackingMode::Untracked => 1,
            HazardTrackingMode::Tracked => 2,
        }
    }

    pub fn from_value(value: u64, call: &'static str) -> Result<HazardTrackingMode, MemError> {
        match value {
            0 => Ok(HazardTrackingMode::Default),
            1 => Ok(HazardTrackingMode::Untracked),
            2 => Ok(HazardTrackingMode::Tracked),
            v => Err(MemError::undefined(call, format!("{v} is not an MTLHazardTrackingMode value (MTLResource.h)"))),
        }
    }

    /// The mode a resource reports: `MTLResource.h`, hazardTrackingMode: "Resources created from heaps are by default
    /// untracked, whereas resources created from the device are by default tracked."
    pub fn resolved(self, from_heap: bool) -> HazardTrackingMode {
        match (self, from_heap) {
            (HazardTrackingMode::Default, true) => HazardTrackingMode::Untracked,
            (HazardTrackingMode::Default, false) => HazardTrackingMode::Tracked,
            (m, _) => m,
        }
    }
}

/// A decoded `MTLResourceOptions` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceOptions {
    pub storage: StorageMode,
    pub cpu_cache: CpuCacheMode,
    pub hazard_tracking: HazardTrackingMode,
}

impl ResourceOptions {
    /// Decode `options`. Bits outside the three fields are not defined by `MTLResource.h`.
    pub fn decode(options: u64, call: &'static str) -> Result<ResourceOptions, MemError> {
        let known = CPU_CACHE_MODE_MASK | STORAGE_MODE_MASK | HAZARD_TRACKING_MODE_MASK;
        if options & !known != 0 {
            return Err(MemError::undefined(
                call,
                format!("MTLResourceOptions {options:#x} has bits outside the fields MTLResource.h defines"),
            ));
        }
        Ok(ResourceOptions {
            storage: StorageMode::from_value((options & STORAGE_MODE_MASK) >> STORAGE_MODE_SHIFT, call)?,
            cpu_cache: CpuCacheMode::from_value((options & CPU_CACHE_MODE_MASK) >> CPU_CACHE_MODE_SHIFT, call)?,
            hazard_tracking: HazardTrackingMode::from_value(
                (options & HAZARD_TRACKING_MODE_MASK) >> HAZARD_TRACKING_MODE_SHIFT,
                call,
            )?,
        })
    }

    /// The `MTLResourceOptions` word.
    pub fn encode(self) -> u64 {
        (self.cpu_cache.value() << CPU_CACHE_MODE_SHIFT)
            | (self.storage.value() << STORAGE_MODE_SHIFT)
            | (self.hazard_tracking.value() << HAZARD_TRACKING_MODE_SHIFT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdk_values() {
        // MTLResource.h: MTLResourceStorageModePrivate = MTLStorageModePrivate << 4,
        // MTLResourceCPUCacheModeWriteCombined = 1, MTLResourceHazardTrackingModeTracked = 2 << 8.
        let o = ResourceOptions::decode(0x20 | 0x1 | 0x200, "test").unwrap();
        assert_eq!(o.storage, StorageMode::Private);
        assert_eq!(o.cpu_cache, CpuCacheMode::WriteCombined);
        assert_eq!(o.hazard_tracking, HazardTrackingMode::Tracked);
        assert_eq!(o.encode(), 0x221);
        assert_eq!(ResourceOptions::decode(0, "test").unwrap().storage, StorageMode::Shared);
        assert_eq!(ResourceOptions::decode(0x30, "test").unwrap().storage, StorageMode::Memoryless);
    }

    #[test]
    fn undefined_options_are_refused() {
        for bad in [0x10, 0x40, 0x2, 0x300, 0x400, 1 << 40] {
            assert!(matches!(ResourceOptions::decode(bad, "test"), Err(MemError::Undefined { .. })), "{bad:#x}");
        }
    }

    #[test]
    fn default_hazard_tracking() {
        assert_eq!(HazardTrackingMode::Default.resolved(true), HazardTrackingMode::Untracked);
        assert_eq!(HazardTrackingMode::Default.resolved(false), HazardTrackingMode::Tracked);
        assert_eq!(HazardTrackingMode::Untracked.resolved(false), HazardTrackingMode::Untracked);
    }
}
