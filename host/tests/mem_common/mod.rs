//! Shared pieces of the memory tests on a native WebGPU implementation: the device, a guest-memory embedder and the
//! Metal reference outputs.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use webmetal_host::embed::{
    DirtyPages, Embedder, GuestMemoryError, GuestMemoryErrorKind, MonotonicNanos, PresentError, PresentedFrame,
    SharedEventError, SharedEventId, SurfaceDesc, SurfaceError, TaskId,
};
use webmetal_host::format::webgpu::{Feature, HostFeatures};
use webmetal_host::gpu::idl::{DeviceDescriptor, FeatureName, RequestAdapterOptions};
use webmetal_host::gpu::native::{Native, NativeDevice, NativeGpu};
use webmetal_host::gpu::{Adapter as _, Device as _, Gpu as _, GpuError, negotiate};
use webmetal_wire::{Event, Handle, SurfaceId};

pub const TASK: TaskId = TaskId(1);

/// A WebGPU device with every feature the adapter offers (`all`), or with none.
pub struct Gpu {
    pub gpu: NativeGpu,
    pub device: NativeDevice,
    pub features: HostFeatures,
    pub errors: Rc<RefCell<Vec<GpuError>>>,
}

impl Gpu {
    pub fn new(all: bool) -> Gpu {
        let gpu = NativeGpu::new();
        let device = gpu
            .block_on(
                async {
                    let adapter =
                        gpu.request_adapter(&RequestAdapterOptions::new()).await.unwrap().expect("an adapter");
                    let mut d = if !all {
                        DeviceDescriptor::new()
                    } else {
                        match negotiate::plan(&adapter.features(), &adapter.limits()) {
                            Ok(p) => p.descriptor,
                            // wgpu-native reports no "core-features-and-limits" (host/tests/gpu_smoke_known_wgpu-native.txt),
                            // so the negotiation refuses its adapter; the memory tests then ask for every feature it
                            // offers with the default limits.
                            Err(negotiate::NegotiationError::NotCore) => {
                                let offered = adapter.features();
                                let mut d = DeviceDescriptor::new();
                                d.required_features =
                                    FeatureName::ALL.into_iter().filter(|f| offered.contains(f.as_str())).collect();
                                d
                            }
                            Err(e) => panic!("negotiation: {e}"),
                        }
                    };
                    d.label = "memory test device".into();
                    adapter.request_device(&d).await.expect("a device")
                },
                Duration::from_secs(30),
            )
            .expect("device request stalled");
        let names = device.features();
        let features = HostFeatures::new(Feature::ALL.into_iter().filter(|f| names.contains(f.name())));
        let errors = Rc::new(RefCell::new(Vec::new()));
        let sink = errors.clone();
        device.set_onuncapturederror(Some(Box::new(move |e| sink.borrow_mut().push(e))));
        Gpu { gpu, device, features, errors }
    }

    pub fn name(&self) -> String {
        self.gpu.implementation()
    }

    pub fn block_on<F: std::future::Future>(&self, f: F) -> F::Output {
        self.gpu.block_on(f, Duration::from_secs(120)).expect("GPU work stalled")
    }

    /// Fails if WebGPU reported any error.
    pub fn assert_no_errors(&self) {
        self.gpu.process_events();
        let errors = self.errors.borrow();
        assert!(errors.is_empty(), "{}: uncaptured WebGPU errors: {errors:?}", self.name());
    }
}

pub type Api = Native;

/// Guest memory of one task as a flat range, with optional write tracking in pages.
pub struct Guest {
    pub base: u64,
    pub memory: RefCell<Vec<u8>>,
    /// Page size of the write tracking; `None`: the embedder does not track writes.
    pub page: Option<u64>,
    dirty: RefCell<BTreeSet<u64>>,
    pub surfaces: RefCell<Vec<(SurfaceDesc, Vec<u8>)>>,
    pub surface_writes: RefCell<Vec<(SurfaceId, u32)>>,
}

impl Guest {
    pub fn new(size: usize, page: Option<u64>) -> Guest {
        Guest {
            base: 0x10_0000,
            memory: RefCell::new(vec![0; size]),
            page,
            dirty: RefCell::new(BTreeSet::new()),
            surfaces: RefCell::new(Vec::new()),
            surface_writes: RefCell::new(Vec::new()),
        }
    }

    /// A write by the guest's CPU.
    pub fn cpu_write(&self, addr: u64, bytes: &[u8]) {
        let o = (addr - self.base) as usize;
        self.memory.borrow_mut()[o..o + bytes.len()].copy_from_slice(bytes);
        if let Some(p) = self.page {
            let mut d = self.dirty.borrow_mut();
            let mut a = addr & !(p - 1);
            while a < addr + bytes.len() as u64 {
                d.insert(a);
                a += p;
            }
        }
    }

    pub fn cpu_read(&self, addr: u64, len: usize) -> Vec<u8> {
        let o = (addr - self.base) as usize;
        self.memory.borrow()[o..o + len].to_vec()
    }

    fn check(&self, task: TaskId, addr: u64, len: usize) -> Result<usize, GuestMemoryError> {
        let size = self.memory.borrow().len() as u64;
        if task != TASK || addr < self.base || addr + len as u64 > self.base + size {
            let kind = if task != TASK { GuestMemoryErrorKind::NoTask } else { GuestMemoryErrorKind::Unmapped };
            return Err(GuestMemoryError { task, addr, len: len as u64, fault: addr, kind });
        }
        Ok((addr - self.base) as usize)
    }
}

impl Embedder for Guest {
    fn read_guest(&self, task: TaskId, addr: u64, dst: &mut [u8]) -> Result<(), GuestMemoryError> {
        let o = self.check(task, addr, dst.len())?;
        dst.copy_from_slice(&self.memory.borrow()[o..o + dst.len()]);
        Ok(())
    }
    fn write_guest(&self, task: TaskId, addr: u64, src: &[u8]) -> Result<(), GuestMemoryError> {
        let o = self.check(task, addr, src.len())?;
        self.memory.borrow_mut()[o..o + src.len()].copy_from_slice(src);
        Ok(())
    }
    fn dirty_pages(&self, task: TaskId, addr: u64, len: u64) -> Result<Option<DirtyPages>, GuestMemoryError> {
        self.check(task, addr, len as usize)?;
        let Some(p) = self.page else { return Ok(None) };
        let first = addr & !(p - 1);
        let count = (addr + len - first).div_ceil(p);
        let mut bits = vec![0u64; count.div_ceil(64) as usize];
        let mut d = self.dirty.borrow_mut();
        for i in 0..count {
            if d.remove(&(first + i * p)) {
                bits[(i / 64) as usize] |= 1 << (i % 64);
            }
        }
        Ok(Some(DirtyPages { page_size: p, first_page: first, bits }))
    }
    fn surface_lookup(&self, id: SurfaceId) -> Result<SurfaceDesc, SurfaceError> {
        self.surfaces.borrow().iter().find(|(d, _)| d.id == id).map(|(d, _)| d.clone()).ok_or(SurfaceError::Unknown(id))
    }
    fn read_surface(&self, id: SurfaceId, offset: u64, dst: &mut [u8]) -> Result<(), SurfaceError> {
        let s = self.surfaces.borrow();
        let (_, mem) = s.iter().find(|(d, _)| d.id == id).ok_or(SurfaceError::Unknown(id))?;
        let end = offset as usize + dst.len();
        if end > mem.len() {
            return Err(SurfaceError::OutOfRange { id, offset, len: dst.len() as u64 });
        }
        dst.copy_from_slice(&mem[offset as usize..end]);
        Ok(())
    }
    fn write_surface_changed(&self, id: SurfaceId, offset: u64, new: &[u8], shadow: &[u8]) -> Result<(), SurfaceError> {
        let mut s = self.surfaces.borrow_mut();
        let (_, mem) = s.iter_mut().find(|(d, _)| d.id == id).ok_or(SurfaceError::Unknown(id))?;
        if offset as usize + new.len() > mem.len() {
            return Err(SurfaceError::OutOfRange { id, offset, len: new.len() as u64 });
        }
        for (i, (&n, &o)) in new.iter().zip(shadow).enumerate() {
            if n != o {
                mem[offset as usize + i] = n;
            }
        }
        Ok(())
    }
    fn surface_note_gpu_write(&self, id: SurfaceId, plane: u32) {
        self.surface_writes.borrow_mut().push((id, plane));
    }
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

/// The reference device whose outputs the tests compare with.
pub const GOLDEN_DEVICE: &str = "apple-m4-macos-27.0.1-26a434";

/// A case file of the Metal reference outputs (release variant). Records are one per line.
pub struct Golden {
    pub case: String,
    text: String,
}

impl Golden {
    pub fn load(case: &str) -> Golden {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/reference/goldens")
            .join(GOLDEN_DEVICE)
            .join("release")
            .join(format!("{case}.json"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Golden { case: case.to_string(), text }
    }

    pub fn exists(case: &str) -> bool {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/reference/goldens")
            .join(GOLDEN_DEVICE)
            .join("release")
            .join(format!("{case}.json"))
            .is_file()
    }

    pub fn status(&self) -> &str {
        let at = self.text.find("\"status\": \"").expect("status") + 11;
        &self.text[at..at + self.text[at..].find('"').unwrap()]
    }

    fn record(&self, name: &str) -> Option<&str> {
        let key = format!("\"name\":\"{name}\"");
        self.text.lines().find(|l| l.contains(&key))
    }

    fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        let k = format!("\"{key}\":");
        let at = line.find(&k)? + k.len();
        let rest = &line[at..];
        if let Some(s) = rest.strip_prefix('"') {
            Some(&s[..s.find('"')?])
        } else {
            let end = rest.find([',', '}']).unwrap_or(rest.len());
            Some(&rest[..end])
        }
    }

    /// The bytes of a `data` record, if the case recorded it.
    pub fn bytes(&self, name: &str) -> Option<Vec<u8>> {
        let hex = Self::field(self.record(name)?, "hex")?;
        Some((0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect())
    }

    /// The comparison rule of a record.
    pub fn rule(&self, name: &str) -> Option<&str> {
        let line = self.record(name)?;
        let at = line.find("\"compare\":\"")? + 11;
        Some(&line[at..at + line[at..].find('"')?])
    }

    /// A layout field of a `data` record.
    pub fn layout(&self, name: &str, key: &str) -> Option<u64> {
        let line = self.record(name)?;
        let layout = &line[line.find("\"layout\":{")?..];
        Self::field(layout, key)?.parse().ok()
    }

    /// The strings of a `value` record of type `strings`.
    pub fn strings(&self, name: &str) -> Option<Vec<String>> {
        let line = self.record(name)?;
        let at = line.find("\"value\":[")? + 9;
        let list = &line[at..at + line[at..].find(']')?];
        Some(list.split(',').map(|s| s.trim_matches('"').to_string()).collect())
    }
}
