//! [`RefEmbedder`]: the host interface over in-process state.
//!
//! Guest memory is a set of mapped ranges per task, IOSurfaces are byte arrays with their descriptions, shared events
//! are 64-bit values, posted events are kept in order, and the display model is the set of displays attached with
//! the frames shown on each. The clock is the process's monotonic clock, or a manual one that only moves when told,
//! for runs whose timing must be reproducible.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use webmetal_host::embed::{
    Embedder, GuestMemoryError, GuestMemoryErrorKind, MonotonicNanos, PresentError, PresentedFrame, SharedEventError,
    SharedEventId, SurfaceDesc, SurfaceError, TaskId,
};
use webmetal_wire::{Event, Handle, SurfaceId};

/// The embedder's clock.
pub enum Clock {
    /// The process's monotonic clock, from the embedder's creation.
    Monotonic(Instant),
    /// A clock that reads what it was last set to.
    Manual(Cell<u64>),
}

/// An event posted for a task, with the queue it was posted for.
#[derive(Clone, Debug, PartialEq)]
pub struct Posted {
    pub task: TaskId,
    pub queue: Option<Handle>,
    pub event: Event,
}

/// A frame the embedder was told was shown, and when.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shown {
    pub frame: PresentedFrame,
    pub at: MonotonicNanos,
}

struct Surface {
    desc: SurfaceDesc,
    bytes: Vec<u8>,
}

/// The reference embedder's state.
pub struct RefEmbedder {
    clock: Clock,
    /// Per task: mapped ranges by start address.
    memory: RefCell<HashMap<TaskId, BTreeMap<u64, Vec<u8>>>>,
    surfaces: RefCell<HashMap<SurfaceId, Surface>>,
    /// Shared events by (task, port name), and their values.
    event_ports: RefCell<HashMap<(TaskId, u32), SharedEventId>>,
    event_values: RefCell<HashMap<SharedEventId, u64>>,
    next_event: Cell<u64>,
    posted: RefCell<Vec<Posted>>,
    displays: RefCell<BTreeSet<u32>>,
    shown: RefCell<Vec<Shown>>,
    /// Surfaces the GPU wrote, in order.
    gpu_writes: RefCell<Vec<(SurfaceId, u32)>>,
}

impl RefEmbedder {
    pub fn new(clock: Clock) -> RefEmbedder {
        RefEmbedder {
            clock,
            memory: RefCell::new(HashMap::new()),
            surfaces: RefCell::new(HashMap::new()),
            event_ports: RefCell::new(HashMap::new()),
            event_values: RefCell::new(HashMap::new()),
            next_event: Cell::new(1),
            posted: RefCell::new(Vec::new()),
            displays: RefCell::new(BTreeSet::new()),
            shown: RefCell::new(Vec::new()),
            gpu_writes: RefCell::new(Vec::new()),
        }
    }

    /// Set the manual clock. Panics on the monotonic clock, or when the time would go backwards.
    pub fn set_time(&self, t: MonotonicNanos) {
        match &self.clock {
            Clock::Manual(c) => {
                assert!(t.0 >= c.get(), "the clock is monotonic");
                c.set(t.0);
            }
            Clock::Monotonic(_) => panic!("the monotonic clock cannot be set"),
        }
    }

    // Guest memory ----------------------------------------------------------------------------------------------

    /// Map `len` zeroed bytes at `addr` in `task`. The range must not overlap a mapped one.
    pub fn map(&self, task: TaskId, addr: u64, len: u64) -> Result<(), String> {
        let mut memory = self.memory.borrow_mut();
        let ranges = memory.entry(task).or_default();
        let end = addr.checked_add(len).ok_or("the range wraps")?;
        if let Some((&a, r)) = ranges.range(..end).next_back()
            && a + r.len() as u64 > addr
        {
            return Err(format!("{addr:#x}+{len:#x} overlaps the mapping at {a:#x}+{:#x}", r.len()));
        }
        ranges.insert(addr, vec![0; len as usize]);
        Ok(())
    }

    /// Visit the mapped pieces of `[addr, addr + len)` in `task` in order, failing at the first unmapped byte.
    fn each_piece(
        &self,
        task: TaskId,
        addr: u64,
        len: usize,
        mut f: impl FnMut(&mut Vec<u8>, usize, usize, usize),
    ) -> Result<(), GuestMemoryError> {
        let mut memory = self.memory.borrow_mut();
        let fail = |fault, kind| GuestMemoryError { task, addr, len: len as u64, fault, kind };
        let ranges = memory.get_mut(&task).ok_or_else(|| fail(addr, GuestMemoryErrorKind::NoTask))?;
        let mut done = 0usize;
        while done < len {
            let at = addr + done as u64;
            let Some((&base, r)) = ranges.range_mut(..=at).next_back() else {
                return Err(fail(at, GuestMemoryErrorKind::Unmapped));
            };
            let off = (at - base) as usize;
            if off >= r.len() {
                return Err(fail(at, GuestMemoryErrorKind::Unmapped));
            }
            let n = (r.len() - off).min(len - done);
            f(r, off, done, n);
            done += n;
        }
        Ok(())
    }

    /// A CPU write of the guest.
    pub fn cpu_write(&self, task: TaskId, addr: u64, bytes: &[u8]) -> Result<(), GuestMemoryError> {
        self.each_piece(task, addr, bytes.len(), |r, off, done, n| {
            r[off..off + n].copy_from_slice(&bytes[done..done + n])
        })
    }

    /// A CPU read of the guest.
    pub fn cpu_read(&self, task: TaskId, addr: u64, len: usize) -> Result<Vec<u8>, GuestMemoryError> {
        let mut out = vec![0; len];
        self.each_piece(task, addr, len, |r, off, done, n| out[done..done + n].copy_from_slice(&r[off..off + n]))?;
        Ok(out)
    }

    /// The task exited: its memory is gone.
    pub fn remove_task(&self, task: TaskId) {
        self.memory.borrow_mut().remove(&task);
    }

    // IOSurfaces ------------------------------------------------------------------------------------------------

    /// Create a surface with zeroed memory. Its id must be new.
    pub fn create_surface(&self, desc: SurfaceDesc) -> Result<(), String> {
        let mut s = self.surfaces.borrow_mut();
        if s.contains_key(&desc.id) {
            return Err(format!("surface {} exists", desc.id.0));
        }
        let bytes = vec![0; desc.alloc_size as usize];
        s.insert(desc.id, Surface { desc, bytes });
        Ok(())
    }

    /// A CPU write into a surface, between `IOSurfaceLock` and `IOSurfaceUnlock`: the seed changes.
    pub fn surface_cpu_write(&self, id: SurfaceId, offset: u64, bytes: &[u8]) -> Result<(), SurfaceError> {
        let mut s = self.surfaces.borrow_mut();
        let surface = s.get_mut(&id).ok_or(SurfaceError::Unknown(id))?;
        let end = offset as usize + bytes.len();
        if end > surface.bytes.len() {
            return Err(SurfaceError::OutOfRange { id, offset, len: bytes.len() as u64 });
        }
        surface.bytes[offset as usize..end].copy_from_slice(bytes);
        surface.desc.seed = surface.desc.seed.wrapping_add(1);
        Ok(())
    }

    /// The surface's memory.
    pub fn surface_bytes(&self, id: SurfaceId) -> Result<Vec<u8>, SurfaceError> {
        self.surfaces.borrow().get(&id).map(|s| s.bytes.clone()).ok_or(SurfaceError::Unknown(id))
    }

    /// The surface's last use is gone.
    pub fn remove_surface(&self, id: SurfaceId) {
        self.surfaces.borrow_mut().remove(&id);
    }

    /// The surfaces the GPU wrote, in order.
    pub fn gpu_writes(&self) -> Vec<(SurfaceId, u32)> {
        self.gpu_writes.borrow().clone()
    }

    // Shared events ---------------------------------------------------------------------------------------------

    /// A shared event whose send right `port` names in `task`, with `value`.
    pub fn create_shared_event(&self, task: TaskId, port: u32, value: u64) -> SharedEventId {
        let id = SharedEventId(self.next_event.get());
        self.next_event.set(id.0 + 1);
        self.event_ports.borrow_mut().insert((task, port), id);
        self.event_values.borrow_mut().insert(id, value);
        id
    }

    /// A CPU-side `setSignaledValue:`.
    pub fn set_shared_event(&self, event: SharedEventId, value: u64) -> Result<(), SharedEventError> {
        let mut v = self.event_values.borrow_mut();
        *v.get_mut(&event).ok_or(SharedEventError::Gone(event))? = value;
        Ok(())
    }

    // Events and displays ---------------------------------------------------------------------------------------

    /// The events posted since the last call, in order.
    pub fn take_posted(&self) -> Vec<Posted> {
        std::mem::take(&mut *self.posted.borrow_mut())
    }

    /// The events posted and not yet taken.
    pub fn posted(&self) -> Vec<Posted> {
        self.posted.borrow().clone()
    }

    /// A display of the display model.
    pub fn add_display(&self, display: u32) {
        self.displays.borrow_mut().insert(display);
    }

    pub fn remove_display(&self, display: u32) {
        self.displays.borrow_mut().remove(&display);
    }

    /// The frames shown so far, in order.
    pub fn shown(&self) -> Vec<Shown> {
        self.shown.borrow().clone()
    }
}

impl Embedder for RefEmbedder {
    fn read_guest(&self, task: TaskId, addr: u64, dst: &mut [u8]) -> Result<(), GuestMemoryError> {
        let len = dst.len();
        self.each_piece(task, addr, len, |r, off, done, n| dst[done..done + n].copy_from_slice(&r[off..off + n]))
    }

    fn write_guest(&self, task: TaskId, addr: u64, src: &[u8]) -> Result<(), GuestMemoryError> {
        self.cpu_write(task, addr, src)
    }

    fn surface_lookup(&self, id: SurfaceId) -> Result<SurfaceDesc, SurfaceError> {
        self.surfaces.borrow().get(&id).map(|s| s.desc.clone()).ok_or(SurfaceError::Unknown(id))
    }

    fn read_surface(&self, id: SurfaceId, offset: u64, dst: &mut [u8]) -> Result<(), SurfaceError> {
        let s = self.surfaces.borrow();
        let surface = s.get(&id).ok_or(SurfaceError::Unknown(id))?;
        let end = (offset as usize).checked_add(dst.len()).filter(|&e| e <= surface.bytes.len());
        let Some(end) = end else { return Err(SurfaceError::OutOfRange { id, offset, len: dst.len() as u64 }) };
        dst.copy_from_slice(&surface.bytes[offset as usize..end]);
        Ok(())
    }

    fn write_surface_changed(&self, id: SurfaceId, offset: u64, new: &[u8], shadow: &[u8]) -> Result<(), SurfaceError> {
        assert_eq!(new.len(), shadow.len(), "write_surface_changed: new and shadow differ in length");
        let mut s = self.surfaces.borrow_mut();
        let surface = s.get_mut(&id).ok_or(SurfaceError::Unknown(id))?;
        if offset as usize + new.len() > surface.bytes.len() {
            return Err(SurfaceError::OutOfRange { id, offset, len: new.len() as u64 });
        }
        for (i, (&n, &o)) in new.iter().zip(shadow).enumerate() {
            if n != o {
                surface.bytes[offset as usize + i] = n;
            }
        }
        Ok(())
    }

    fn surface_note_gpu_write(&self, id: SurfaceId, plane: u32) {
        self.gpu_writes.borrow_mut().push((id, plane));
        if let Some(s) = self.surfaces.borrow_mut().get_mut(&id) {
            s.desc.seed = s.desc.seed.wrapping_add(1);
        }
    }

    fn shared_event_resolve(&self, task: TaskId, port: u32) -> Result<SharedEventId, SharedEventError> {
        self.event_ports.borrow().get(&(task, port)).copied().ok_or(SharedEventError::NotAnEvent { task, port })
    }

    fn shared_event_value(&self, event: SharedEventId) -> Result<u64, SharedEventError> {
        self.event_values.borrow().get(&event).copied().ok_or(SharedEventError::Gone(event))
    }

    fn shared_event_signal(&self, event: SharedEventId, value: u64) -> Result<(), SharedEventError> {
        self.set_shared_event(event, value)
    }

    fn shared_event_release(&self, _: SharedEventId) {}

    fn post(&self, task: TaskId, queue: Option<Handle>, event: Event) {
        self.posted.borrow_mut().push(Posted { task, queue, event });
    }

    fn present(&self, frame: &PresentedFrame) -> Result<(), PresentError> {
        if !self.displays.borrow().contains(&frame.display) {
            return Err(PresentError::UnknownDisplay(frame.display));
        }
        let at = self.now();
        self.shown.borrow_mut().push(Shown { frame: *frame, at });
        Ok(())
    }

    fn now(&self) -> MonotonicNanos {
        match &self.clock {
            Clock::Monotonic(start) => MonotonicNanos(start.elapsed().as_nanos() as u64),
            Clock::Manual(c) => MonotonicNanos(c.get()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: TaskId = TaskId(7);

    #[test]
    fn guest_memory_spans_adjacent_mappings_and_reports_the_fault() {
        let e = RefEmbedder::new(Clock::Manual(Cell::new(0)));
        e.map(T, 0x1000, 0x10).unwrap();
        e.map(T, 0x1010, 0x10).unwrap();
        assert!(e.map(T, 0x1008, 4).is_err(), "overlap");
        e.cpu_write(T, 0x100c, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        assert_eq!(e.cpu_read(T, 0x100c, 8).unwrap(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let err = e.cpu_read(T, 0x101c, 8).unwrap_err();
        assert_eq!((err.fault, err.kind), (0x1020, GuestMemoryErrorKind::Unmapped));
        let err = e.cpu_read(TaskId(8), 0x1000, 1).unwrap_err();
        assert_eq!(err.kind, GuestMemoryErrorKind::NoTask);
    }

    #[test]
    fn surface_writes_change_the_seed() {
        let e = RefEmbedder::new(Clock::Manual(Cell::new(0)));
        let desc = SurfaceDesc {
            id: SurfaceId(3),
            width: 2,
            height: 2,
            bytes_per_row: 8,
            bytes_per_element: 4,
            element_width: 1,
            element_height: 1,
            pixel_format: 0,
            alloc_size: 16,
            planes: vec![],
            seed: 5,
        };
        e.create_surface(desc).unwrap();
        e.surface_cpu_write(SurfaceId(3), 4, &[9, 9]).unwrap();
        assert_eq!(e.surface_lookup(SurfaceId(3)).unwrap().seed, 6);
        assert!(e.surface_cpu_write(SurfaceId(3), 15, &[1, 2]).is_err());
        e.surface_note_gpu_write(SurfaceId(3), 0);
        assert_eq!(e.surface_lookup(SurfaceId(3)).unwrap().seed, 7);
        assert_eq!(&e.surface_bytes(SurfaceId(3)).unwrap()[4..6], &[9, 9]);
    }

    #[test]
    fn the_manual_clock() {
        let e = RefEmbedder::new(Clock::Manual(Cell::new(10)));
        assert_eq!(e.now(), MonotonicNanos(10));
        e.set_time(MonotonicNanos(25));
        assert_eq!(e.now(), MonotonicNanos(25));
    }
}
