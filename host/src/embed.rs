//! The host interface: what the embedding environment provides to the executor.
//!
//! The embedder is whatever runs the guest system: a kernel that serves the WebMetal accelerator service, or the
//! reference embedder that replays recorded wire streams. The executor reaches guest memory, IOSurfaces, shared events,
//! the guest's event queues, the display and the clock only through [`Embedder`]. Calls in the other direction (a new
//! task, a task's exit, a control request or a committed stream to execute, a CPU write to an IOSurface, a framebuffer
//! swap to show) are the executor's own API and are not part of this trait.
//!
//! Every method that can fail returns the failure. The executor never treats a failed access as success and never
//! skips the work that needed it: it completes the affected command buffer or request with a failure and reports it.

use core::fmt;

use webmetal_wire::{Event, Handle, SurfaceId};

/// A guest task (process), as the embedder names it. Object handles are per task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TaskId(pub u64);

/// A point on the embedder's monotonic clock, in nanoseconds.
///
/// This is the clock behind the timing fields Metal exposes (`GPUStartTime`, `kernelStartTime` and the rest, carried in
/// the CommandBufferScheduled and CommandBufferCompleted events). The guest reads the same clock, so the values are
/// ordered and comparable with the guest's own time; they are not GPU hardware times.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MonotonicNanos(pub u64);

/// Why a guest memory access failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuestMemoryErrorKind {
    /// Part of the range is not mapped in the task.
    Unmapped,
    /// The range is mapped but the task's mapping does not permit the access.
    Protection,
    /// The task does not exist (it exited).
    NoTask,
}

/// A failed guest memory access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestMemoryError {
    pub task: TaskId,
    /// First address of the requested range.
    pub addr: u64,
    /// Length of the requested range.
    pub len: u64,
    /// First address that could not be accessed.
    pub fault: u64,
    pub kind: GuestMemoryErrorKind,
}

impl fmt::Display for GuestMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "guest memory {:?} at {:#x} (access {:#x}+{:#x}) in task {}", self.kind, self.fault, self.addr, self.len, self.task.0)
    }
}

impl std::error::Error for GuestMemoryError {}

/// Write tracking for a range of guest memory, from the embedder's page-protection table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirtyPages {
    /// Size of a tracking page in bytes; a power of two.
    pub page_size: u64,
    /// Address of the first page reported (the page containing the start of the requested range).
    pub first_page: u64,
    /// Bit `i` (bit `i % 64` of word `i / 64`) is set when page `first_page + i * page_size` may have been written.
    pub bits: Vec<u64>,
}

impl DirtyPages {
    /// May the page containing `addr` have been written? Pages outside the report count as written.
    pub fn is_dirty(&self, addr: u64) -> bool {
        if addr < self.first_page {
            return true;
        }
        let page = (addr - self.first_page) / self.page_size;
        self.bits.get((page / 64) as usize).is_none_or(|w| w & (1 << (page % 64)) != 0)
    }
}

/// One plane of an IOSurface. Fields mirror the per-plane IOSurface properties of `IOSurfaceRef.h`
/// (`IOSurfaceGetWidthOfPlane`, `IOSurfaceGetBytesPerRowOfPlane` and the others).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfacePlane {
    pub width: u64,
    pub height: u64,
    pub bytes_per_row: u64,
    pub bytes_per_element: u64,
    pub element_width: u64,
    pub element_height: u64,
    /// Offset of the plane's first byte from the start of the surface's memory (`IOSurfaceGetBaseAddressOfPlane` minus
    /// `IOSurfaceGetBaseAddress`).
    pub offset: u64,
    /// Bytes the plane occupies.
    pub size: u64,
}

/// An IOSurface as the kernel's IOSurface implementation describes it. Fields mirror the whole-surface properties of
/// `IOSurfaceRef.h`; `planes` is empty for a surface without planes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceDesc {
    pub id: SurfaceId,
    pub width: u64,
    pub height: u64,
    pub bytes_per_row: u64,
    pub bytes_per_element: u64,
    pub element_width: u64,
    pub element_height: u64,
    /// `IOSurfaceGetPixelFormat`: a four-character code.
    pub pixel_format: u32,
    /// `IOSurfaceGetAllocSize`.
    pub alloc_size: u64,
    pub planes: Vec<SurfacePlane>,
    /// `IOSurfaceGetSeed`: changes whenever the surface's contents may have changed.
    pub seed: u32,
}

/// Why a surface operation failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceError {
    /// No surface has this id.
    Unknown(SurfaceId),
    /// The range lies outside the surface's allocation.
    OutOfRange { id: SurfaceId, offset: u64, len: u64 },
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "surface error: {:?}", self)
    }
}

impl std::error::Error for SurfaceError {}

/// A shared event kernel object, as the embedder names it.
///
/// On iPadOS 17.7.11 (21H461) Metal's `_MTLSharedEvent` derives from `IOSurfaceSharedEvent`: a shared event is an
/// IOSurface kernel object reached through a mach port (`-eventPort`), and IOSurface.framework, not the Metal driver,
/// serves `signaledValue`, `setSignaledValue:`, `notifyListener:atValue:block:` and `waitUntilSignaledValue:timeoutMS:`.
/// Command streams refer to it by a handle bound with the RegisterSharedEvent request; the executor reads and signals
/// the kernel object through this interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SharedEventId(pub u64);

/// Why a shared-event operation failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedEventError {
    /// The port name does not denote a shared event in the task.
    NotAnEvent { task: TaskId, port: u32 },
    /// The event no longer exists.
    Gone(SharedEventId),
}

impl fmt::Display for SharedEventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shared event error: {:?}", self)
    }
}

impl std::error::Error for SharedEventError {}

/// A rectangle in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: i64,
    pub y: i64,
    pub width: u64,
    pub height: u64,
}

/// What a presented frame shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentSource {
    /// A plane of an IOSurface (the display path: a framebuffer swap names surfaces).
    Surface { id: SurfaceId, plane: u32 },
    /// A texture of a task.
    Texture { task: TaskId, texture: Handle },
}

/// A frame for the presentation sink.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentedFrame {
    /// The display, as the embedder numbers them.
    pub display: u32,
    pub source: PresentSource,
    /// The part of the source shown.
    pub source_rect: Rect,
    /// Where it lands on the display.
    pub target_rect: Rect,
    /// The orientation transform the guest requested for the layer, as the guest's framebuffer call encoded it. The
    /// executor passes it through; its value set is the embedder's display model's to define.
    pub transform: u32,
    /// When the guest asked the frame to appear; `None` for as soon as possible.
    pub time: Option<MonotonicNanos>,
}

/// Why the presentation sink refused a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentError {
    /// No such display.
    UnknownDisplay(u32),
    /// The display cannot show this frame (format, size or transform it does not support); the embedder reports the
    /// detail through its own channel.
    Unsupported,
}

impl fmt::Display for PresentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "present error: {:?}", self)
    }
}

impl std::error::Error for PresentError {}

/// What the embedder provides to the executor.
///
/// Methods take `&self`: the executor runs on one thread (the one that owns the WebGPU device) and an embedder that
/// keeps mutable state uses interior mutability.
pub trait Embedder {
    // Guest memory --------------------------------------------------------------------------------------------------

    /// Copy `dst.len()` bytes at `addr` in `task` into `dst`. Fails if any byte is unreadable; on failure `dst` holds
    /// no data the executor may use.
    fn read_guest(&self, task: TaskId, addr: u64, dst: &mut [u8]) -> Result<(), GuestMemoryError>;

    /// Copy `src` to `addr` in `task`. Fails if any byte is unwritable; bytes before the fault may have been written.
    fn write_guest(&self, task: TaskId, addr: u64, src: &[u8]) -> Result<(), GuestMemoryError>;

    /// Write to `addr` in `task` exactly the bytes where `new` differs from `shadow` (the executor's copy of what the
    /// guest held when they were last synchronized), leaving every other byte of the guest's memory untouched, so that
    /// concurrent CPU writes to other bytes of the same buffer are preserved. `new` and `shadow` have the same
    /// length. The default writes each run of differing bytes with [`Embedder::write_guest`].
    fn write_guest_changed(&self, task: TaskId, addr: u64, new: &[u8], shadow: &[u8]) -> Result<(), GuestMemoryError> {
        assert_eq!(new.len(), shadow.len(), "write_guest_changed: new and shadow differ in length");
        let mut at = 0;
        while at < new.len() {
            if new[at] == shadow[at] {
                at += 1;
                continue;
            }
            let start = at;
            while at < new.len() && new[at] != shadow[at] {
                at += 1;
            }
            self.write_guest(task, addr + start as u64, &new[start..at])?;
        }
        Ok(())
    }

    /// Pages of `[addr, addr + len)` in `task` the guest may have written since the previous call that covered them,
    /// from the embedder's page-protection table; the call re-arms tracking for the pages it reports. A page reported
    /// clean must be unchanged since that previous report. `Ok(None)` means the embedder does not track writes: the
    /// executor then compares the guest's bytes with its shadow copy, which is always correct and slower.
    fn dirty_pages(&self, task: TaskId, addr: u64, len: u64) -> Result<Option<DirtyPages>, GuestMemoryError> {
        let _ = (task, addr, len);
        Ok(None)
    }

    // IOSurfaces ----------------------------------------------------------------------------------------------------

    /// Describe the surface with this id. Surfaces are global objects, shared by every task.
    fn surface_lookup(&self, id: SurfaceId) -> Result<SurfaceDesc, SurfaceError>;

    /// Copy `dst.len()` bytes at `offset` from the start of the surface's memory into `dst`.
    fn read_surface(&self, id: SurfaceId, offset: u64, dst: &mut [u8]) -> Result<(), SurfaceError>;

    /// Write into the surface's memory at `offset` exactly the bytes where `new` differs from `shadow`, as
    /// [`Embedder::write_guest_changed`] does for guest memory.
    fn write_surface_changed(&self, id: SurfaceId, offset: u64, new: &[u8], shadow: &[u8]) -> Result<(), SurfaceError>;

    /// The GPU wrote `plane` of the surface: its seed must change and CPU readers must see the new contents once they
    /// lock the surface. Called after the bytes have been written back with [`Embedder::write_surface_changed`].
    fn surface_note_gpu_write(&self, id: SurfaceId, plane: u32);

    // Shared events -------------------------------------------------------------------------------------------------

    /// Resolve the shared-event send right that `port` names in `task` (from a RegisterSharedEvent request).
    fn shared_event_resolve(&self, task: TaskId, port: u32) -> Result<SharedEventId, SharedEventError>;

    /// The event's current signaled value.
    fn shared_event_value(&self, event: SharedEventId) -> Result<u64, SharedEventError>;

    /// Set the event's signaled value, as a GPU-side signal does; the kernel object wakes its waiters and listeners.
    fn shared_event_signal(&self, event: SharedEventId, value: u64) -> Result<(), SharedEventError>;

    /// The executor no longer refers to the event.
    fn shared_event_release(&self, event: SharedEventId);

    // Completion sink -----------------------------------------------------------------------------------------------

    /// Queue an event record for `task` and ring the doorbell its guest registered for `queue` (`None`: device-level
    /// events). The guest collects queued records with the FetchEvents request, in the order they were posted.
    fn post(&self, task: TaskId, queue: Option<Handle>, event: Event);

    // Presentation sink ---------------------------------------------------------------------------------------------

    /// Show a frame. The executor calls this once every command buffer that writes the source and was committed
    /// before the frame was requested has completed, so the source's contents are final.
    fn present(&self, frame: &PresentedFrame) -> Result<(), PresentError>;

    // Clock ---------------------------------------------------------------------------------------------------------

    /// The current time on the embedder's monotonic clock (see [`MonotonicNanos`]).
    fn now(&self) -> MonotonicNanos;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Guest memory as one flat array, enough to exercise the provided methods.
    struct Flat {
        memory: RefCell<Vec<u8>>,
        writes: RefCell<Vec<(u64, usize)>>,
    }

    fn unmapped(task: TaskId, addr: u64, len: usize, size: usize) -> GuestMemoryError {
        GuestMemoryError { task, addr, len: len as u64, fault: size as u64, kind: GuestMemoryErrorKind::Unmapped }
    }

    impl Embedder for Flat {
        fn read_guest(&self, task: TaskId, addr: u64, dst: &mut [u8]) -> Result<(), GuestMemoryError> {
            let m = self.memory.borrow();
            let end = addr as usize + dst.len();
            if end > m.len() {
                return Err(unmapped(task, addr, dst.len(), m.len()));
            }
            dst.copy_from_slice(&m[addr as usize..end]);
            Ok(())
        }
        fn write_guest(&self, task: TaskId, addr: u64, src: &[u8]) -> Result<(), GuestMemoryError> {
            let mut m = self.memory.borrow_mut();
            let end = addr as usize + src.len();
            if end > m.len() {
                return Err(unmapped(task, addr, src.len(), m.len()));
            }
            m[addr as usize..end].copy_from_slice(src);
            self.writes.borrow_mut().push((addr, src.len()));
            Ok(())
        }
        fn surface_lookup(&self, id: SurfaceId) -> Result<SurfaceDesc, SurfaceError> {
            Err(SurfaceError::Unknown(id))
        }
        fn read_surface(&self, id: SurfaceId, _: u64, _: &mut [u8]) -> Result<(), SurfaceError> {
            Err(SurfaceError::Unknown(id))
        }
        fn write_surface_changed(&self, id: SurfaceId, _: u64, _: &[u8], _: &[u8]) -> Result<(), SurfaceError> {
            Err(SurfaceError::Unknown(id))
        }
        fn surface_note_gpu_write(&self, _: SurfaceId, _: u32) {}
        fn shared_event_resolve(&self, task: TaskId, port: u32) -> Result<SharedEventId, SharedEventError> {
            Err(SharedEventError::NotAnEvent { task, port })
        }
        fn shared_event_value(&self, event: SharedEventId) -> Result<u64, SharedEventError> {
            Err(SharedEventError::Gone(event))
        }
        fn shared_event_signal(&self, event: SharedEventId, _: u64) -> Result<(), SharedEventError> {
            Err(SharedEventError::Gone(event))
        }
        fn shared_event_release(&self, _: SharedEventId) {}
        fn post(&self, _: TaskId, _: Option<Handle>, _: Event) {}
        fn present(&self, frame: &PresentedFrame) -> Result<(), PresentError> {
            Err(PresentError::UnknownDisplay(frame.display))
        }
        fn now(&self) -> MonotonicNanos {
            MonotonicNanos(0)
        }
    }

    #[test]
    fn write_guest_changed_writes_only_differing_runs() {
        let e = Flat { memory: RefCell::new(vec![9; 16]), writes: RefCell::new(Vec::new()) };
        let shadow = [0, 0, 0, 0, 0, 0, 0, 0];
        let new = [0, 1, 1, 0, 0, 2, 0, 3];
        e.write_guest_changed(TaskId(1), 4, &new, &shadow).unwrap();
        assert_eq!(*e.writes.borrow(), vec![(5, 2), (9, 1), (11, 1)]);
        // Bytes equal to the shadow keep whatever the guest holds now.
        assert_eq!(*e.memory.borrow(), vec![9, 9, 9, 9, 9, 1, 1, 9, 9, 2, 9, 3, 9, 9, 9, 9]);
    }

    #[test]
    fn write_guest_changed_reports_the_fault() {
        let e = Flat { memory: RefCell::new(vec![0; 8]), writes: RefCell::new(Vec::new()) };
        let err = e.write_guest_changed(TaskId(2), 6, &[1, 1, 1, 1], &[0, 0, 0, 0]).unwrap_err();
        assert_eq!(err.kind, GuestMemoryErrorKind::Unmapped);
        assert_eq!(err.task, TaskId(2));
    }

    #[test]
    fn untracked_embedders_report_no_dirty_pages() {
        let e = Flat { memory: RefCell::new(vec![0; 8]), writes: RefCell::new(Vec::new()) };
        assert_eq!(e.dirty_pages(TaskId(1), 0, 8), Ok(None));
    }

    #[test]
    fn dirty_page_bits() {
        let d = DirtyPages { page_size: 0x4000, first_page: 0x10000, bits: vec![0b101] };
        assert!(d.is_dirty(0x10000));
        assert!(!d.is_dirty(0x14000));
        assert!(d.is_dirty(0x18fff));
        assert!(!d.is_dirty(0x1c000));
        assert!(d.is_dirty(0x10000 + 64 * 0x4000), "pages beyond the report count as written");
        assert!(d.is_dirty(0xc000), "pages before the report count as written");
    }
}
