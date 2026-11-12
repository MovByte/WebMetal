//! Heaps (`MTLHeap`): what a heap is, which allocations it can be shown to hold, and the rules of its resources.
//!
//! WebGPU has no heaps, so every heap resource is its own WebGPU object. Aliasing (two resources over the same heap
//! range) is defined by Metal only as "contents undefined after the other is used", so independent objects are one
//! of the outcomes Metal allows (plan section 6.2). What the executor must still reproduce is which allocations
//! succeed: `newBufferWithLength:options:` returns "nil if the heap memory is full" (`MTLHeap` documentation).
//!
//! The A10X's sizes come from the device profile: `heapBufferSizeAndAlignWithLength:options:` returns the length as
//! the size and 256 as the alignment (`[heap]` of `profile/ipados-17.7.11-21H461-a10x.toml`, both extraction methods
//! agreeing, over 504 lengths and every storage, cache and hazard-tracking mode). `heapTextureSizeAndAlignWithDescriptor:`
//! depends on the texture layout the AGX driver builds from the kernel's device object and is not in the profile, and
//! neither are the heap's size rounding nor the driver's free-space policy. An allocation is therefore granted only
//! when it is certain: its documented size, at its documented alignment, fits in the size the descriptor asked for
//! after every allocation granted so far, none of whose space is reused (a heap's `size` is at least the requested
//! size, as `heap.size-at-least-requested` records on real Metal). Anything else is refused loudly with the reason;
//! it is never answered with a guessed `nil` or a guessed success.
//!
//! Values are the SDK's `MTLHeap.h`: `MTLHeapTypeAutomatic` 0, `MTLHeapTypePlacement` 1, `MTLHeapTypeSparse` 2
//! (macOS, Mac Catalyst and tvOS only).

use super::error::MemError;
use super::options::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use super::purgeable::Purgeable;
use webmetal_wire::HeapDescriptor;

/// `heapBufferSizeAndAlignWithLength:options:` size: the length itself (profile `[heap] buffer_size_is_length`).
pub fn heap_buffer_size(length: u64) -> u64 {
    length
}

/// `heapBufferSizeAndAlignWithLength:options:` alignment (profile `[heap] buffer_alignment`).
pub const HEAP_BUFFER_ALIGNMENT: u64 = 256;

/// `MTLHeapType` as iOS offers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeapType {
    Automatic,
    Placement,
}

/// A heap: its descriptor's modes and the allocations granted from it.
#[derive(Clone, Debug)]
pub struct Heap {
    pub kind: HeapType,
    pub storage: StorageMode,
    pub cpu_cache: CpuCacheMode,
    /// The resolved mode (`MTLHazardTrackingModeDefault` is untracked for heaps, `MTLHeap.h`).
    pub hazard_tracking: HazardTrackingMode,
    /// The size the descriptor requested.
    pub requested_size: u64,
    pub purgeable: Purgeable,
    /// End of the highest allocation granted; space below it is never handed out twice.
    end: u64,
}

/// A granted sub-allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapAllocation {
    pub offset: u64,
    pub size: u64,
}

impl Heap {
    /// A heap for `newHeapWithDescriptor:`.
    pub fn new(d: &HeapDescriptor) -> Result<Heap, MemError> {
        const CALL: &str = "-[MTLDevice newHeapWithDescriptor:]";
        let kind = match d.r#type {
            0 => HeapType::Automatic,
            1 => HeapType::Placement,
            2 => return Err(MemError::undefined(CALL, "MTLHeapTypeSparse is not available on iOS (MTLHeap.h)")),
            t => return Err(MemError::undefined(CALL, format!("{t} is not an MTLHeapType value"))),
        };
        let storage = StorageMode::from_value(d.storage_mode, CALL)?;
        let cpu_cache = CpuCacheMode::from_value(d.cpu_cache_mode, CALL)?;
        let hazard = HazardTrackingMode::from_value(d.hazard_tracking_mode, CALL)?;
        // MTLHeapDescriptor.resourceOptions packs the same three modes; the descriptor carries both verbatim.
        let packed = ResourceOptions::decode(d.resource_options, CALL)?;
        if packed != (ResourceOptions { storage, cpu_cache, hazard_tracking: hazard }) {
            return Err(MemError::invalid(format!(
                "heap descriptor resourceOptions {:#x} disagrees with its storage, CPU cache and hazard tracking modes",
                d.resource_options
            )));
        }
        if storage == StorageMode::Memoryless {
            return Err(MemError::undefined(CALL, "a heap with MTLStorageModeMemoryless"));
        }
        if d.size == 0 {
            return Err(MemError::undefined(CALL, "a heap of size 0"));
        }
        Ok(Heap {
            kind,
            storage,
            cpu_cache,
            hazard_tracking: hazard.resolved(true),
            requested_size: d.size,
            purgeable: Purgeable::NonVolatile,
            end: 0,
        })
    }

    /// `-[MTLHeap newBufferWithLength:options:]`: the allocation, or why it cannot be shown to succeed.
    pub fn allocate_buffer(&mut self, length: u64, options: &ResourceOptions) -> Result<HeapAllocation, MemError> {
        const CALL: &str = "-[MTLHeap newBufferWithLength:options:]";
        // MTLHeap documentation: "The heap's type needs to be automatic; the buffer's storage mode option needs to
        // match the heap's storageMode property; the buffer's CPU cache mode option needs to match the heap's
        // cpuCacheMode property".
        if self.kind != HeapType::Automatic {
            return Err(MemError::undefined(CALL, "the heap's type is not MTLHeapTypeAutomatic"));
        }
        if options.storage != self.storage || options.cpu_cache != self.cpu_cache {
            return Err(MemError::undefined(CALL, "the buffer's storage or CPU cache mode differs from the heap's"));
        }
        if length == 0 {
            return Err(MemError::undefined(CALL, "a buffer of length 0"));
        }
        let size = heap_buffer_size(length);
        let offset = self.end.next_multiple_of(HEAP_BUFFER_ALIGNMENT);
        match offset.checked_add(size) {
            Some(end) if end <= self.requested_size => {
                self.end = end;
                Ok(HeapAllocation { offset, size })
            }
            _ => Err(MemError::unimplemented(format!(
                "{CALL}: whether a {size}-byte buffer fits in this {}-byte heap after {} bytes of earlier \
                 allocations depends on the A10X heap's size rounding and free-space policy, which the device \
                 profile does not record",
                self.requested_size, self.end
            ))),
        }
    }

    /// `-[MTLHeap newTextureWithDescriptor:]`.
    pub fn allocate_texture(&mut self) -> Result<HeapAllocation, MemError> {
        Err(MemError::unimplemented(
            "-[MTLHeap newTextureWithDescriptor:]: the A10X's heapTextureSizeAndAlignWithDescriptor: depends on the \
             AGX texture layout built from the kernel's device object and is not in the device profile",
        ))
    }

    /// `-[MTLResource makeAliasable]` on a resource of this heap. Its space is not handed out again by this account
    /// (see the module documentation), so only the rule is checked.
    pub fn make_aliasable(&self) -> Result<(), MemError> {
        if self.kind != HeapType::Automatic {
            return Err(MemError::undefined(
                "-[MTLResource makeAliasable]",
                "makeAliasable is only valid for resources of an MTLHeapTypeAutomatic heap (makeAliasable() documentation)",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::testutil::repo_path;

    fn descriptor(kind: i64, size: u64, storage: u64) -> HeapDescriptor {
        HeapDescriptor {
            cpu_cache_mode: 0,
            hazard_tracking_mode: 0,
            memory_pool_id: 0,
            pinned_gpu_address: 0,
            protection_options: 0,
            resource_options: storage << 4,
            size,
            sparse_page_access_counters_enabled: false,
            sparse_page_size: 0,
            storage_mode: storage,
            r#type: kind,
        }
    }

    #[test]
    fn profile_values() {
        let profile = std::fs::read_to_string(repo_path("profile/ipados-17.7.11-21H461-a10x.toml")).unwrap();
        let heap = profile.split("\n[heap]\n").nth(1).unwrap();
        let value = |key: &str| {
            heap.lines().find_map(|l| l.strip_prefix(&format!("{key} = "))).unwrap().split_whitespace().next().unwrap()
        };
        assert_eq!(value("buffer_size_is_length"), "true");
        assert_eq!(value("buffer_alignment").parse::<u64>().unwrap(), HEAP_BUFFER_ALIGNMENT);
    }

    #[test]
    fn grants_only_certain_allocations() {
        let mut h = Heap::new(&descriptor(0, 64 * 1024, 2)).unwrap();
        assert_eq!(h.hazard_tracking, HazardTrackingMode::Untracked);
        let private = ResourceOptions::decode(0x20, "t").unwrap();
        assert_eq!(h.allocate_buffer(100, &private).unwrap(), HeapAllocation { offset: 0, size: 100 });
        assert_eq!(h.allocate_buffer(4096, &private).unwrap(), HeapAllocation { offset: 256, size: 4096 });
        // 4352 bytes are taken; 60 KiB more does not fit in the requested size.
        assert!(matches!(h.allocate_buffer(60 * 1024, &private), Err(MemError::Unimplemented { .. })));
        assert_eq!(h.allocate_buffer(59 * 1024, &private).unwrap().offset, 4352);
        let shared = ResourceOptions::decode(0, "t").unwrap();
        assert!(matches!(h.allocate_buffer(16, &shared), Err(MemError::Undefined { .. })));
        assert!(matches!(h.allocate_buffer(0, &private), Err(MemError::Undefined { .. })));
        assert!(matches!(h.allocate_texture(), Err(MemError::Unimplemented { .. })));
    }

    #[test]
    fn descriptor_rules() {
        assert!(matches!(Heap::new(&descriptor(2, 4096, 2)), Err(MemError::Undefined { .. })));
        assert!(matches!(Heap::new(&descriptor(0, 4096, 3)), Err(MemError::Undefined { .. })));
        assert!(matches!(Heap::new(&descriptor(0, 0, 2)), Err(MemError::Undefined { .. })));
        let mut d = descriptor(0, 4096, 2);
        d.resource_options = 0;
        assert!(matches!(Heap::new(&d), Err(MemError::Invalid { .. })));
        let mut placement = Heap::new(&descriptor(1, 4096, 2)).unwrap();
        let private = ResourceOptions::decode(0x20, "t").unwrap();
        assert!(matches!(placement.allocate_buffer(16, &private), Err(MemError::Undefined { .. })));
        assert!(placement.make_aliasable().is_err());
    }
}
