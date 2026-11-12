//! The WebGPU objects over webgpu.h handles, with the content-timeline state the specification keeps on each.

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::fmt;
use std::rc::{Rc, Weak};

use super::conv::{self, Arena, from_c, from_sv, sv};
use super::desc;
use super::sys;
use super::{
    DeviceShared, Native, NativeDevice, NoExternalSource, Slot, SlotFuture, Timeline, callback_box, callback_mode,
    quirks, set_backend_label, status_is, take_box, track,
};
use crate::gpu::api::*;
use crate::gpu::idl::*;
use crate::gpu::num::{Float32, Size64};

/// The common part of every object: its handle, its `label` and its device.
struct Base<R: Copy> {
    raw: R,
    label: RefCell<String>,
    dev: Rc<DeviceShared>,
    release: unsafe extern "C" fn(R),
}

impl<R: Copy> Drop for Base<R> {
    fn drop(&mut self) {
        unsafe { (self.release)(self.raw) };
    }
}

macro_rules! object {
    ($(#[$m:meta])* $name:ident, $raw:ty, $release:path, $set_label:path) => {
        $(#[$m])*
        #[derive(Clone)]
        pub struct $name(Rc<Base<$raw>>);

        impl $name {
            pub(crate) fn wrap(dev: &Rc<DeviceShared>, raw: $raw, label: &str) -> $name {
                $name(Rc::new(Base { raw, label: RefCell::new(label.to_string()), dev: dev.clone(), release: $release }))
            }

            #[allow(dead_code)]
            pub(crate) fn raw(&self) -> $raw {
                self.0.raw
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({:?})"), self.0.label.borrow())
            }
        }

        impl Object for $name {
            fn label(&self) -> String {
                self.0.label.borrow().clone()
            }
            fn set_label(&self, label: &str) {
                *self.0.label.borrow_mut() = label.to_string();
                set_backend_label(|l| unsafe { $set_label(self.0.raw, l) }, label);
            }
        }
    };
}

object!(
    /// `GPUTextureView`.
    NativeTextureView, sys::WGPUTextureView, sys::wgpuTextureViewRelease, sys::wgpuTextureViewSetLabel
);
object!(
    /// `GPUSampler`.
    NativeSampler, sys::WGPUSampler, sys::wgpuSamplerRelease, sys::wgpuSamplerSetLabel
);
object!(
    /// `GPUBindGroupLayout`.
    NativeBindGroupLayout, sys::WGPUBindGroupLayout, sys::wgpuBindGroupLayoutRelease, sys::wgpuBindGroupLayoutSetLabel
);
object!(
    /// `GPUBindGroup`.
    NativeBindGroup, sys::WGPUBindGroup, sys::wgpuBindGroupRelease, sys::wgpuBindGroupSetLabel
);
object!(
    /// `GPUPipelineLayout`.
    NativePipelineLayout, sys::WGPUPipelineLayout, sys::wgpuPipelineLayoutRelease, sys::wgpuPipelineLayoutSetLabel
);
object!(
    /// `GPUCommandBuffer`.
    NativeCommandBuffer, sys::WGPUCommandBuffer, sys::wgpuCommandBufferRelease, sys::wgpuCommandBufferSetLabel
);
object!(
    /// `GPURenderBundle`.
    NativeRenderBundle, sys::WGPURenderBundle, sys::wgpuRenderBundleRelease, sys::wgpuRenderBundleSetLabel
);
object!(
    /// `GPUComputePipeline`.
    NativeComputePipeline, sys::WGPUComputePipeline, sys::wgpuComputePipelineRelease, sys::wgpuComputePipelineSetLabel
);
object!(
    /// `GPURenderPipeline`.
    NativeRenderPipeline, sys::WGPURenderPipeline, sys::wgpuRenderPipelineRelease, sys::wgpuRenderPipelineSetLabel
);
object!(
    /// `GPUCommandEncoder`.
    NativeCommandEncoder, sys::WGPUCommandEncoder, sys::wgpuCommandEncoderRelease, sys::wgpuCommandEncoderSetLabel
);
object!(
    /// `GPUComputePassEncoder`.
    NativeComputePassEncoder, sys::WGPUComputePassEncoder, sys::wgpuComputePassEncoderRelease, sys::wgpuComputePassEncoderSetLabel
);
object!(
    /// `GPURenderPassEncoder`.
    NativeRenderPassEncoder, sys::WGPURenderPassEncoder, sys::wgpuRenderPassEncoderRelease, sys::wgpuRenderPassEncoderSetLabel
);
object!(
    /// `GPURenderBundleEncoder`.
    NativeRenderBundleEncoder, sys::WGPURenderBundleEncoder, sys::wgpuRenderBundleEncoderRelease, sys::wgpuRenderBundleEncoderSetLabel
);

/// `GPUExternalTexture`: made only by `importExternalTexture()` from a platform video source, which exists only in a
/// browser; no value of this type exists.
#[derive(Clone, Debug)]
pub struct NativeExternalTexture(NoExternalSource);

impl NativeExternalTexture {
    /// No value of this type exists, so any result type can be produced from one.
    pub(crate) fn absurd<T>(&self) -> T {
        match self.0 {}
    }
}

impl Object for NativeExternalTexture {
    fn label(&self) -> String {
        match self.0 {}
    }
    fn set_label(&self, _: &str) {
        match self.0 {}
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------------------------------------------------

object!(
    /// `GPUQueue`.
    NativeQueue, sys::WGPUQueue, sys::wgpuQueueRelease, sys::wgpuQueueSetLabel
);

impl NativeQueue {
    fn timeline(&self) -> Timeline {
        Timeline::Queue(self.0.raw as usize)
    }
}

unsafe extern "C" fn on_work_done(
    _status: sys::WGPUQueueWorkDoneStatus,
    _message: sys::WGPUStringView,
    u1: *mut c_void,
    _u2: *mut c_void,
) {
    // onSubmittedWorkDone() never rejects: on device loss the work is complete too.
    let b = unsafe { take_box::<(), ()>(u1) };
    b.slot.fill(());
}

impl Queue<Native> for NativeQueue {
    fn submit(&self, command_buffers: &[&NativeCommandBuffer]) {
        let raws: Vec<sys::WGPUCommandBuffer> = command_buffers.iter().map(|c| c.raw()).collect();
        unsafe { sys::wgpuQueueSubmit(self.0.raw, raws.len(), raws.as_ptr()) };
    }

    fn on_submitted_work_done(&self) -> GpuFuture<()> {
        let slot = Slot::new();
        let (bx, done) = callback_box(&slot, ());
        let info = sys::WGPUQueueWorkDoneCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: callback_mode(),
            callback: Some(on_work_done),
            userdata1: bx,
            userdata2: std::ptr::null_mut(),
        };
        let fut = unsafe { sys::wgpuQueueOnSubmittedWorkDone(self.0.raw, info) };
        track(&self.0.dev.inst, fut, self.timeline(), Some(&self.0.dev), done);
        Box::pin(SlotFuture { inst: self.0.dev.inst.clone(), slot })
    }

    fn write_buffer(
        &self,
        buffer: &NativeBuffer,
        buffer_offset: Size64,
        data: &[u8],
        data_offset: Size64,
        size: Option<Size64>,
    ) -> Result<(), Exception> {
        // Content timeline: the data is an ArrayBuffer, so elements are bytes.
        let data_size = data.len() as u64;
        let contents_size = match size {
            Some(s) => s.get() as i128,
            None => data_size as i128 - data_offset.get() as i128,
        };
        if contents_size < 0 || data_offset.get() as i128 + contents_size > data_size as i128 || contents_size % 4 != 0
        {
            return Err(Exception::new(
                ExceptionKind::OperationError,
                format!(
                    "writeBuffer: dataOffset {} and size {size:?} do not select a multiple of 4 bytes within {data_size} bytes",
                    data_offset.get()
                ),
            ));
        }
        let start = data_offset.get() as usize;
        let bytes = &data[start..start + contents_size as usize];
        unsafe {
            sys::wgpuQueueWriteBuffer(self.0.raw, buffer.raw(), buffer_offset.get(), bytes.as_ptr().cast(), bytes.len())
        };
        Ok(())
    }

    fn write_texture(
        &self,
        destination: &TexelCopyTextureInfo<'_, Native>,
        data: &[u8],
        data_layout: &TexelCopyBufferLayout,
        size: &Extent3D,
    ) -> Result<(), Exception> {
        let dst = desc::texel_copy_texture_info(destination)?;
        let size = desc::extent(size)?;
        let layout = desc::texel_copy_buffer_layout(data_layout)?;
        unsafe { sys::wgpuQueueWriteTexture(self.0.raw, &dst, data.as_ptr().cast(), data.len(), &layout, &size) };
        Ok(())
    }

    fn copy_external_image_to_texture(
        &self,
        source: &CopyExternalImageSourceInfo<'_, Native>,
        _destination: &CopyExternalImageDestInfo<'_, Native>,
        _copy_size: &Extent3D,
    ) -> Result<(), Exception> {
        match *source.source {}
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------------------------------------------------

/// An active buffer mapping (`[[mapping]]`).
struct Mapping {
    write: bool,
    range: (u64, u64),
    views: Vec<(u64, u64)>,
}

/// A pending map's promise.
type MapSlot = Rc<Slot<Result<(), Exception>>>;

pub(crate) struct BufferInner {
    raw: sys::WGPUBuffer,
    dev: Rc<DeviceShared>,
    label: RefCell<String>,
    size: u64,
    usage: u32,
    pending: RefCell<Option<MapSlot>>,
    mapping: RefCell<Option<Mapping>>,
    /// Incremented by every unmap; a mapped range from an earlier generation is detached.
    generation: Cell<u64>,
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        unsafe { sys::wgpuBufferRelease(self.raw) };
    }
}

/// `GPUBuffer`.
#[derive(Clone)]
pub struct NativeBuffer(Rc<BufferInner>);

impl fmt::Debug for NativeBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NativeBuffer({:?})", self.0.label.borrow())
    }
}

impl NativeBuffer {
    pub(crate) fn raw(&self) -> sys::WGPUBuffer {
        self.0.raw
    }

    /// The content-timeline steps of `unmap()`.
    fn unmap_content(&self) {
        if let Some(p) = self.0.pending.borrow_mut().take() {
            p.fill(Err(Exception::new(
                ExceptionKind::AbortError,
                "the buffer was unmapped before the mapping completed",
            )));
        }
        if self.0.mapping.borrow_mut().take().is_some() {
            self.0.generation.set(self.0.generation.get() + 1);
        }
    }
}

pub(crate) fn create_buffer(dev: &NativeDevice, d: &BufferDescriptor) -> Result<NativeBuffer, Exception> {
    if d.mapped_at_creation && !d.size.get().is_multiple_of(4) {
        return Err(Exception::new(
            ExceptionKind::RangeError,
            format!("mappedAtCreation with size {} (not a multiple of 4)", d.size.get()),
        ));
    }
    let desc = desc::buffer_descriptor(d);
    let raw = unsafe { sys::wgpuDeviceCreateBuffer(dev.0.raw, &desc) };
    if raw.is_null() {
        // webgpu.h returns null only when the mapping of a mappedAtCreation buffer cannot be allocated.
        return Err(Exception::new(
            ExceptionKind::RangeError,
            "the mapping for mappedAtCreation could not be allocated",
        ));
    }
    let mapping = d.mapped_at_creation.then(|| Mapping { write: true, range: (0, d.size.get()), views: Vec::new() });
    let inner = Rc::new(BufferInner {
        raw,
        dev: dev.0.clone(),
        label: RefCell::new(d.label.clone()),
        size: d.size.get(),
        usage: d.usage,
        pending: RefCell::new(None),
        mapping: RefCell::new(mapping),
        generation: Cell::new(0),
    });
    dev.0.buffers.borrow_mut().push(Rc::downgrade(&inner));
    Ok(NativeBuffer(inner))
}

struct MapExtra {
    buffer: Weak<BufferInner>,
    write: bool,
    range: (u64, u64),
}

unsafe extern "C" fn on_map(
    status: sys::WGPUMapAsyncStatus,
    message: sys::WGPUStringView,
    u1: *mut c_void,
    _u2: *mut c_void,
) {
    let b = unsafe { take_box::<Result<(), Exception>, MapExtra>(u1) };
    let Some(buf) = b.extra.buffer.upgrade() else { return };
    // Content timeline map success and failure steps: a map that unmap() cancelled was already rejected.
    let current = buf.pending.borrow().as_ref().is_some_and(|p| Rc::ptr_eq(p, &b.slot));
    if !current {
        return;
    }
    *buf.pending.borrow_mut() = None;
    let r = if status_is(status, "WGPUMapAsyncStatus", "Success") {
        *buf.mapping.borrow_mut() = Some(Mapping { write: b.extra.write, range: b.extra.range, views: Vec::new() });
        Ok(())
    } else if status_is(status, "WGPUMapAsyncStatus", "Error") {
        Err(Exception::new(ExceptionKind::OperationError, from_sv(message)))
    } else {
        // Aborted (device lost, or the buffer destroyed or unmapped on the device) and CallbackCancelled.
        Err(Exception::new(ExceptionKind::AbortError, from_sv(message)))
    };
    b.slot.fill(r);
}

unsafe extern "C" fn on_map_ignored(
    _status: sys::WGPUMapAsyncStatus,
    _message: sys::WGPUStringView,
    u1: *mut c_void,
    _u2: *mut c_void,
) {
    drop(unsafe { take_box::<Result<(), Exception>, ()>(u1) });
}

impl Object for NativeBuffer {
    fn label(&self) -> String {
        self.0.label.borrow().clone()
    }
    fn set_label(&self, label: &str) {
        *self.0.label.borrow_mut() = label.to_string();
        set_backend_label(|l| unsafe { sys::wgpuBufferSetLabel(self.0.raw, l) }, label);
    }
}

impl Buffer<Native> for NativeBuffer {
    fn size(&self) -> u64 {
        self.0.size
    }

    fn usage(&self) -> u32 {
        self.0.usage
    }

    fn map_state(&self) -> BufferMapState {
        if self.0.mapping.borrow().is_some() {
            BufferMapState::Mapped
        } else if self.0.pending.borrow().is_some() {
            BufferMapState::Pending
        } else {
            BufferMapState::Unmapped
        }
    }

    fn map_async(&self, mode: u32, offset: Size64, size: Option<Size64>) -> GpuFuture<Result<(), Exception>> {
        let dev = &self.0.dev;
        let queue = dev.queue.borrow().clone().expect("device queue");
        let range_size = desc::size_or_rest(size, self.0.size, offset);
        let (Ok(off), Ok(sz)) = (usize::try_from(offset.get()), usize::try_from(range_size)) else {
            return Box::pin(async { Err(Exception::unsupported("mapAsync range does not fit size_t")) });
        };
        if self.map_state() != BufferMapState::Unmapped {
            // Early reject: the device timeline generates a validation error, which the implementation produces for a
            // map of a buffer that is already mapped or pending; its own callback result is not observable.
            let slot = Slot::<Result<(), Exception>>::new();
            let (bx, done) = callback_box(&slot, ());
            let info = sys::WGPUBufferMapCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: callback_mode(),
                callback: Some(on_map_ignored),
                userdata1: bx,
                userdata2: std::ptr::null_mut(),
            };
            let fut = unsafe { sys::wgpuBufferMapAsync(self.0.raw, desc::flags(mode), off, sz, info) };
            track(&dev.inst, fut, queue.timeline(), Some(dev), done);
            return Box::pin(async {
                Err(Exception::new(
                    ExceptionKind::OperationError,
                    "the buffer is already mapped or has a mapping pending",
                ))
            });
        }
        let slot = Slot::new();
        *self.0.pending.borrow_mut() = Some(slot.clone());
        let extra = MapExtra {
            buffer: Rc::downgrade(&self.0),
            write: mode & MapMode::WRITE != 0,
            range: (offset.get(), offset.get().saturating_add(range_size)),
        };
        let (bx, done) = callback_box(&slot, extra);
        let info = sys::WGPUBufferMapCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: callback_mode(),
            callback: Some(on_map),
            userdata1: bx,
            userdata2: std::ptr::null_mut(),
        };
        let fut = unsafe { sys::wgpuBufferMapAsync(self.0.raw, desc::flags(mode), off, sz, info) };
        track(&dev.inst, fut, queue.timeline(), Some(dev), done);
        Box::pin(SlotFuture { inst: dev.inst.clone(), slot })
    }

    fn get_mapped_range(&self, offset: Size64, size: Option<Size64>) -> Result<NativeMappedRange, Exception> {
        let offset = offset.get();
        let range_size = match size {
            Some(s) => s.get(),
            None => self.0.size.saturating_sub(offset),
        };
        let write = {
            let mut m = self.0.mapping.borrow_mut();
            let fail = |why: &str| {
                Err(Exception::new(
                    ExceptionKind::OperationError,
                    format!("getMappedRange({offset}, {range_size}): {why}"),
                ))
            };
            let Some(m) = m.as_mut() else { return fail("the buffer is not mapped") };
            if !offset.is_multiple_of(8) {
                return fail("offset is not a multiple of 8");
            }
            if range_size % 4 != 0 {
                return fail("size is not a multiple of 4");
            }
            if offset < m.range.0 || offset + range_size > m.range.1 {
                return fail("the range is outside the mapped range");
            }
            let end = offset + range_size;
            if m.views.iter().any(|(s, e)| offset < *e && *s < end) {
                return fail("the range overlaps a range already returned");
            }
            m.views.push((offset, end));
            m.write
        };
        let len = range_size as usize;
        let generation = self.0.generation.get();
        let kind = if len == 0 {
            RangeData::Copy(RefCell::new(Vec::new()))
        } else if write {
            let p = unsafe { sys::wgpuBufferGetMappedRange(self.0.raw, offset as usize, len) };
            if p.is_null() {
                return Err(Exception::unsupported(
                    "the implementation returned no memory for a mapped range the specification guarantees",
                ));
            }
            RangeData::Direct(p.cast())
        } else {
            // A read mapping's ArrayBuffer is the mapping's data: writes to it are allowed and discarded at unmap.
            let p = unsafe { sys::wgpuBufferGetConstMappedRange(self.0.raw, offset as usize, len) };
            if p.is_null() {
                return Err(Exception::unsupported(
                    "the implementation returned no memory for a mapped range the specification guarantees",
                ));
            }
            RangeData::Copy(RefCell::new(unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) }.to_vec()))
        };
        Ok(NativeMappedRange { buf: self.0.clone(), generation, len: range_size, data: kind })
    }

    fn unmap(&self) {
        // unmap() of a buffer with no mapping and no pending map does nothing (its content-timeline steps return
        // early); the implementation is told only when there is one to end, since wgpu-native reports an unmap of
        // an unmapped buffer as a validation error.
        let mapped = self.0.pending.borrow().is_some() || self.0.mapping.borrow().is_some();
        self.unmap_content();
        if mapped {
            unsafe { sys::wgpuBufferUnmap(self.0.raw) };
        }
    }

    fn destroy(&self) {
        self.unmap();
        unsafe { sys::wgpuBufferDestroy(self.0.raw) };
    }
}

enum RangeData {
    /// The implementation's mapped memory (write mappings).
    Direct(*mut u8),
    /// A copy (read mappings, and empty ranges).
    Copy(RefCell<Vec<u8>>),
}

/// The `ArrayBuffer` of a mapped range.
pub struct NativeMappedRange {
    buf: Rc<BufferInner>,
    generation: u64,
    len: u64,
    data: RangeData,
}

impl fmt::Debug for NativeMappedRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NativeMappedRange({} bytes{})", self.len, if self.detached() { ", detached" } else { "" })
    }
}

impl NativeMappedRange {
    fn detached(&self) -> bool {
        self.buf.generation.get() != self.generation
    }

    fn check(&self, offset: u64, n: usize) -> Result<usize, Exception> {
        if self.detached() {
            return Err(Exception::new(ExceptionKind::TypeError, "the ArrayBuffer is detached"));
        }
        match offset.checked_add(n as u64) {
            Some(end) if end <= self.len => Ok(offset as usize),
            _ => Err(Exception::new(
                ExceptionKind::RangeError,
                format!("{n} bytes at {offset} are outside an ArrayBuffer of {} bytes", self.len),
            )),
        }
    }
}

impl MappedRange for NativeMappedRange {
    fn byte_length(&self) -> u64 {
        if self.detached() { 0 } else { self.len }
    }

    fn read(&self, offset: u64, dst: &mut [u8]) -> Result<(), Exception> {
        let at = self.check(offset, dst.len())?;
        match &self.data {
            RangeData::Direct(p) => dst.copy_from_slice(unsafe { std::slice::from_raw_parts(p.add(at), dst.len()) }),
            RangeData::Copy(v) => dst.copy_from_slice(&v.borrow()[at..at + dst.len()]),
        }
        Ok(())
    }

    fn write(&self, offset: u64, src: &[u8]) -> Result<(), Exception> {
        let at = self.check(offset, src.len())?;
        match &self.data {
            RangeData::Direct(p) => {
                unsafe { std::slice::from_raw_parts_mut(p.add(at), src.len()) }.copy_from_slice(src)
            }
            RangeData::Copy(v) => v.borrow_mut()[at..at + src.len()].copy_from_slice(src),
        }
        Ok(())
    }
}

/// `destroy()` on the device: unmap every buffer of the device first (content timeline).
pub(crate) fn unmap_all(dev: &DeviceShared) {
    let bufs: Vec<Rc<BufferInner>> = {
        let mut b = dev.buffers.borrow_mut();
        b.retain(|w| w.strong_count() > 0);
        b.iter().filter_map(|w| w.upgrade()).collect()
    };
    for b in bufs {
        let nb = NativeBuffer(b);
        nb.unmap();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Texture
// ---------------------------------------------------------------------------------------------------------------------

pub(crate) struct TextureInner {
    raw: sys::WGPUTexture,
    dev: Rc<DeviceShared>,
    label: RefCell<String>,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    mip_level_count: u32,
    sample_count: u32,
    dimension: TextureDimension,
    format: TextureFormat,
    usage: u32,
    binding_view_dimension: Option<TextureViewDimension>,
}

impl Drop for TextureInner {
    fn drop(&mut self) {
        unsafe { sys::wgpuTextureRelease(self.raw) };
    }
}

/// `GPUTexture`.
#[derive(Clone)]
pub struct NativeTexture(Rc<TextureInner>);

impl fmt::Debug for NativeTexture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NativeTexture({:?})", self.0.label.borrow())
    }
}

impl NativeTexture {
    pub(crate) fn raw(&self) -> sys::WGPUTexture {
        self.0.raw
    }
}

pub(crate) fn create_texture(dev: &NativeDevice, d: &TextureDescriptor) -> Result<NativeTexture, Exception> {
    let (width, height, depth) = {
        let x = desc::extent(&d.size)?;
        (x.width, x.height, x.depthOrArrayLayers)
    };
    desc::check_format(&dev.0, d.format)?;
    for f in &d.view_formats {
        desc::check_format(&dev.0, *f)?;
    }
    let mut a = Arena::default();
    let cdesc = desc::texture_descriptor(&mut a, d)?;
    let raw = unsafe { sys::wgpuDeviceCreateTexture(dev.0.raw, &cdesc) };
    // textureBindingViewDimension is defined only on devices without "core-features-and-limits".
    let binding_view_dimension = if dev.0.has(FeatureName::CoreFeaturesAndLimits) {
        None
    } else if let Some(v) = d.texture_binding_view_dimension {
        Some(v)
    } else {
        Some(match d.dimension {
            TextureDimension::D1 => TextureViewDimension::D1,
            TextureDimension::D2 if depth == 1 => TextureViewDimension::D2,
            TextureDimension::D2 => TextureViewDimension::D2Array,
            TextureDimension::D3 => TextureViewDimension::D3,
        })
    };
    Ok(NativeTexture(Rc::new(TextureInner {
        raw,
        dev: dev.0.clone(),
        label: RefCell::new(d.label.clone()),
        width,
        height,
        depth_or_array_layers: depth,
        mip_level_count: d.mip_level_count,
        sample_count: d.sample_count,
        dimension: d.dimension,
        format: d.format,
        usage: d.usage,
        binding_view_dimension,
    })))
}

impl Object for NativeTexture {
    fn label(&self) -> String {
        self.0.label.borrow().clone()
    }
    fn set_label(&self, label: &str) {
        *self.0.label.borrow_mut() = label.to_string();
        set_backend_label(|l| unsafe { sys::wgpuTextureSetLabel(self.0.raw, l) }, label);
    }
}

impl Texture<Native> for NativeTexture {
    fn create_view(&self, d: &TextureViewDescriptor) -> Result<NativeTextureView, Exception> {
        if let Some(f) = d.format {
            desc::check_format(&self.0.dev, f)?;
        }
        let mut a = Arena::default();
        let cdesc = desc::texture_view_descriptor(&mut a, d)?;
        let raw = unsafe { sys::wgpuTextureCreateView(self.0.raw, &cdesc) };
        Ok(NativeTextureView::wrap(&self.0.dev, raw, &d.label))
    }

    fn destroy(&self) {
        unsafe { sys::wgpuTextureDestroy(self.0.raw) };
    }

    fn width(&self) -> u32 {
        self.0.width
    }
    fn height(&self) -> u32 {
        self.0.height
    }
    fn depth_or_array_layers(&self) -> u32 {
        self.0.depth_or_array_layers
    }
    fn mip_level_count(&self) -> u32 {
        self.0.mip_level_count
    }
    fn sample_count(&self) -> u32 {
        self.0.sample_count
    }
    fn dimension(&self) -> TextureDimension {
        self.0.dimension
    }
    fn format(&self) -> TextureFormat {
        self.0.format
    }
    fn usage(&self) -> u32 {
        self.0.usage
    }
    fn texture_binding_view_dimension(&self) -> Option<TextureViewDimension> {
        self.0.binding_view_dimension
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Samplers, layouts, bind groups
// ---------------------------------------------------------------------------------------------------------------------

pub(crate) fn create_sampler(dev: &NativeDevice, d: &SamplerDescriptor) -> NativeSampler {
    // Every member of GPUSamplerDescriptor has a webgpu.h enumerator (checked by the enum test).
    let cdesc = desc::sampler_descriptor(d).expect("sampler enums");
    let raw = unsafe { sys::wgpuDeviceCreateSampler(dev.0.raw, &cdesc) };
    NativeSampler::wrap(&dev.0, raw, &d.label)
}

pub(crate) fn create_bind_group_layout(
    dev: &NativeDevice,
    d: &BindGroupLayoutDescriptor,
) -> Result<NativeBindGroupLayout, Exception> {
    let mut a = Arena::default();
    let cdesc = desc::bind_group_layout_descriptor(&mut a, &dev.0, d)?;
    let raw = unsafe { sys::wgpuDeviceCreateBindGroupLayout(dev.0.raw, &cdesc) };
    Ok(NativeBindGroupLayout::wrap(&dev.0, raw, &d.label))
}

pub(crate) fn create_pipeline_layout(
    dev: &NativeDevice,
    d: &PipelineLayoutDescriptor<'_, Native>,
) -> NativePipelineLayout {
    let mut a = Arena::default();
    let cdesc = desc::pipeline_layout_descriptor(&mut a, d);
    let raw = unsafe { sys::wgpuDeviceCreatePipelineLayout(dev.0.raw, &cdesc) };
    NativePipelineLayout::wrap(&dev.0, raw, &d.label)
}

pub(crate) fn create_bind_group(dev: &NativeDevice, d: &BindGroupDescriptor<'_, Native>) -> NativeBindGroup {
    let mut a = Arena::default();
    let mut views = Vec::new();
    // A GPUTexture resource's implicit view can only fail for a reason createView() throws on, and createView() with
    // the default descriptor throws for none.
    let cdesc = desc::bind_group_descriptor(&mut a, d, &mut views).expect("default texture view");
    let raw = unsafe { sys::wgpuDeviceCreateBindGroup(dev.0.raw, &cdesc) };
    drop(views);
    NativeBindGroup::wrap(&dev.0, raw, &d.label)
}

// ---------------------------------------------------------------------------------------------------------------------
// Shaders and pipelines
// ---------------------------------------------------------------------------------------------------------------------

pub(crate) struct ShaderInner {
    base: Base<sys::WGPUShaderModule>,
    code: String,
}

/// `GPUShaderModule`.
#[derive(Clone)]
pub struct NativeShaderModule(Rc<ShaderInner>);

impl fmt::Debug for NativeShaderModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NativeShaderModule({:?})", self.0.base.label.borrow())
    }
}

impl NativeShaderModule {
    pub(crate) fn raw(&self) -> sys::WGPUShaderModule {
        self.0.base.raw
    }
}

pub(crate) fn create_shader_module(dev: &NativeDevice, d: &ShaderModuleDescriptor<'_, Native>) -> NativeShaderModule {
    let mut a = Arena::default();
    let cdesc = desc::shader_module_descriptor(&mut a, d);
    let raw = unsafe { sys::wgpuDeviceCreateShaderModule(dev.0.raw, &cdesc) };
    NativeShaderModule(Rc::new(ShaderInner {
        base: Base {
            raw,
            label: RefCell::new(d.label.clone()),
            dev: dev.0.clone(),
            release: sys::wgpuShaderModuleRelease,
        },
        code: d.code.clone(),
    }))
}

impl Object for NativeShaderModule {
    fn label(&self) -> String {
        self.0.base.label.borrow().clone()
    }
    fn set_label(&self, label: &str) {
        *self.0.base.label.borrow_mut() = label.to_string();
        set_backend_label(|l| unsafe { sys::wgpuShaderModuleSetLabel(self.0.base.raw, l) }, label);
    }
}

/// Number of UTF-16 code units in `s[a..b]` (byte offsets); `None` if either is not a character boundary.
fn utf16_len(s: &str, a: usize, b: usize) -> Option<u64> {
    s.get(a..b).map(|x| x.encode_utf16().count() as u64)
}

struct InfoExtra {
    code: String,
}

unsafe extern "C" fn on_compilation_info(
    status: sys::WGPUCompilationInfoRequestStatus,
    info: *const sys::WGPUCompilationInfo,
    u1: *mut c_void,
    _u2: *mut c_void,
) {
    let b = unsafe { take_box::<Result<CompilationInfo, Exception>, InfoExtra>(u1) };
    if !status_is(status, "WGPUCompilationInfoRequestStatus", "Success") || info.is_null() {
        b.slot.fill(Err(Exception::unsupported(format!(
            "wgpuShaderModuleGetCompilationInfo failed with status {status}"
        ))));
        return;
    }
    let info = unsafe { &*info };
    let code = &b.extra.code;
    let mut messages = Vec::new();
    for m in unsafe { super::slice_or_empty(info.messages, info.messageCount) } {
        // webgpu.h positions are UTF-8 code units; GPUCompilationMessage positions are UTF-16 code units.
        let ty = match from_c::<CompilationMessageType>(m.type_) {
            Some(t) => t,
            None => {
                b.slot.fill(Err(Exception::unsupported(format!(
                    "compilation message type {} is not a WebGPU type",
                    m.type_
                ))));
                return;
            }
        };
        let (line_pos, offset, length) = if m.lineNum == 0 {
            (0, 0, 0)
        } else {
            let off = m.offset as usize;
            let line_start = off.saturating_sub((m.linePos as usize).saturating_sub(1));
            match (
                utf16_len(code, line_start, off),
                utf16_len(code, 0, off),
                utf16_len(code, off, off + m.length as usize),
            ) {
                (Some(lp), Some(o), Some(l)) => (lp + 1, o, l),
                _ => {
                    b.slot.fill(Err(Exception::unsupported(
                        "a compilation message position is not on a character boundary",
                    )));
                    return;
                }
            }
        };
        messages.push(CompilationMessage {
            message: from_sv(m.message),
            r#type: ty,
            line_num: m.lineNum,
            line_pos,
            offset,
            length,
        });
    }
    b.slot.fill(Ok(CompilationInfo { messages }));
}

impl ShaderModule<Native> for NativeShaderModule {
    fn get_compilation_info(&self) -> GpuFuture<Result<CompilationInfo, Exception>> {
        if !quirks::COMPILATION_INFO {
            return Box::pin(async {
                Err(Exception::unsupported(format!(
                    "getCompilationInfo: {} does not implement wgpuShaderModuleGetCompilationInfo",
                    sys::meta::IMPLEMENTATION
                )))
            });
        }
        let slot = Slot::new();
        let (bx, done) = callback_box(&slot, InfoExtra { code: self.0.code.clone() });
        let info = sys::WGPUCompilationInfoCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: callback_mode(),
            callback: Some(on_compilation_info),
            userdata1: bx,
            userdata2: std::ptr::null_mut(),
        };
        let dev = &self.0.base.dev;
        let fut = unsafe { sys::wgpuShaderModuleGetCompilationInfo(self.0.base.raw, info) };
        track(&dev.inst, fut, Timeline::Cpu, Some(dev), done);
        Box::pin(SlotFuture { inst: dev.inst.clone(), slot })
    }
}

impl PipelineBase<Native> for NativeComputePipeline {
    fn get_bind_group_layout(&self, index: u32) -> NativeBindGroupLayout {
        let raw = unsafe { sys::wgpuComputePipelineGetBindGroupLayout(self.0.raw, index) };
        NativeBindGroupLayout::wrap(&self.0.dev, raw, "")
    }
}

impl PipelineBase<Native> for NativeRenderPipeline {
    fn get_bind_group_layout(&self, index: u32) -> NativeBindGroupLayout {
        let raw = unsafe { sys::wgpuRenderPipelineGetBindGroupLayout(self.0.raw, index) };
        NativeBindGroupLayout::wrap(&self.0.dev, raw, "")
    }
}

pub(crate) fn create_compute_pipeline(
    dev: &NativeDevice,
    d: &ComputePipelineDescriptor<'_, Native>,
) -> NativeComputePipeline {
    let mut a = Arena::default();
    let cdesc = desc::compute_pipeline_descriptor(&mut a, d);
    let raw = unsafe { sys::wgpuDeviceCreateComputePipeline(dev.0.raw, &cdesc) };
    NativeComputePipeline::wrap(&dev.0, raw, &d.label)
}

pub(crate) fn create_render_pipeline(
    dev: &NativeDevice,
    d: &RenderPipelineDescriptor<'_, Native>,
) -> Result<NativeRenderPipeline, Exception> {
    desc::check_render_pipeline(&dev.0, d)?;
    let mut a = Arena::default();
    let cdesc = desc::render_pipeline_descriptor(&mut a, d)?;
    let raw = unsafe { sys::wgpuDeviceCreateRenderPipeline(dev.0.raw, &cdesc) };
    Ok(NativeRenderPipeline::wrap(&dev.0, raw, &d.label))
}

fn pipeline_error(status: u32, message: String) -> Exception {
    let reason = if status_is(status, "WGPUCreatePipelineAsyncStatus", "ValidationError") {
        PipelineErrorReason::Validation
    } else if status_is(status, "WGPUCreatePipelineAsyncStatus", "InternalError") {
        PipelineErrorReason::Internal
    } else {
        return Exception::unsupported(format!("pipeline creation callback cancelled: {message}"));
    };
    Exception::new(ExceptionKind::PipelineError(reason), message)
}

unsafe extern "C" fn on_compute_pipeline(
    status: sys::WGPUCreatePipelineAsyncStatus,
    pipeline: sys::WGPUComputePipeline,
    message: sys::WGPUStringView,
    u1: *mut c_void,
    _u2: *mut c_void,
) {
    let b = unsafe { take_box::<Result<sys::WGPUComputePipeline, Exception>, ()>(u1) };
    let r = if status_is(status, "WGPUCreatePipelineAsyncStatus", "Success") {
        Ok(pipeline)
    } else {
        Err(pipeline_error(status, from_sv(message)))
    };
    b.slot.fill(r);
}

unsafe extern "C" fn on_render_pipeline(
    status: sys::WGPUCreatePipelineAsyncStatus,
    pipeline: sys::WGPURenderPipeline,
    message: sys::WGPUStringView,
    u1: *mut c_void,
    _u2: *mut c_void,
) {
    let b = unsafe { take_box::<Result<sys::WGPURenderPipeline, Exception>, ()>(u1) };
    let r = if status_is(status, "WGPUCreatePipelineAsyncStatus", "Success") {
        Ok(pipeline)
    } else {
        Err(pipeline_error(status, from_sv(message)))
    };
    b.slot.fill(r);
}

/// `createXPipelineAsync()` where the implementation lacks the asynchronous entry point: "a new pipeline created as if
/// createXPipeline() was called, except capturing any errors as error, rather than dispatching them to the device"
/// (§10.2.1). The synchronous call runs inside its own validation and internal error scopes, so its errors are
/// captured, never dispatched; the promise rejects with a `GPUPipelineError` of the captured error's kind, and resolves
/// when there is none (including on a lost device).
fn emulated_async<T: 'static>(dev: &NativeDevice, create: impl FnOnce() -> T) -> GpuFuture<Result<T, Exception>> {
    dev.push_error_scope_raw(ErrorFilter::Internal);
    dev.push_error_scope_raw(ErrorFilter::Validation);
    let p = create();
    let validation = dev.pop_error_scope();
    let internal = dev.pop_error_scope();
    Box::pin(async move {
        let v = validation.await;
        let i = internal.await;
        for r in [v, i] {
            match r {
                Ok(None) => {}
                Ok(Some(e)) => {
                    let reason = if e.filter == ErrorFilter::Internal {
                        PipelineErrorReason::Internal
                    } else {
                        PipelineErrorReason::Validation
                    };
                    return Err(Exception::new(ExceptionKind::PipelineError(reason), e.message));
                }
                Err(e) => return Err(e),
            }
        }
        Ok(p)
    })
}

impl NativeDevice {
    fn push_error_scope_raw(&self, f: ErrorFilter) {
        <NativeDevice as Device<Native>>::push_error_scope(self, f)
    }
}

pub(crate) fn create_compute_pipeline_async(
    dev: &NativeDevice,
    d: &ComputePipelineDescriptor<'_, Native>,
) -> GpuFuture<Result<NativeComputePipeline, Exception>> {
    if !quirks::PIPELINE_ASYNC {
        let dv = dev.clone();
        let label = d.label.clone();
        let mut a = Arena::default();
        let cdesc = desc::compute_pipeline_descriptor(&mut a, d);
        return emulated_async(dev, move || {
            let raw = unsafe { sys::wgpuDeviceCreateComputePipeline(dv.0.raw, &cdesc) };
            drop(a);
            NativeComputePipeline::wrap(&dv.0, raw, &label)
        });
    }
    let mut a = Arena::default();
    let cdesc = desc::compute_pipeline_descriptor(&mut a, d);
    let slot = Slot::<Result<sys::WGPUComputePipeline, Exception>>::new();
    let (bx, done) = callback_box(&slot, ());
    let info = sys::WGPUCreateComputePipelineAsyncCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: callback_mode(),
        callback: Some(on_compute_pipeline),
        userdata1: bx,
        userdata2: std::ptr::null_mut(),
    };
    let fut = unsafe { sys::wgpuDeviceCreateComputePipelineAsync(dev.0.raw, &cdesc, info) };
    drop(a);
    track(&dev.0.inst, fut, Timeline::Cpu, Some(&dev.0), done);
    let shared = dev.0.clone();
    let label = d.label.clone();
    Box::pin(async move {
        let raw = SlotFuture { inst: shared.inst.clone(), slot }.await?;
        Ok(NativeComputePipeline::wrap(&shared, raw, &label))
    })
}

pub(crate) fn create_render_pipeline_async(
    dev: &NativeDevice,
    d: &RenderPipelineDescriptor<'_, Native>,
) -> GpuFuture<Result<NativeRenderPipeline, Exception>> {
    // Exceptions thrown on the content timeline of a promise-returning operation reject its promise.
    if let Err(e) = desc::check_render_pipeline(&dev.0, d) {
        return Box::pin(async move { Err(e) });
    }
    let mut a = Arena::default();
    let cdesc = match desc::render_pipeline_descriptor(&mut a, d) {
        Ok(c) => c,
        Err(e) => return Box::pin(async move { Err(e) }),
    };
    if !quirks::PIPELINE_ASYNC {
        let dv = dev.clone();
        let label = d.label.clone();
        return emulated_async(dev, move || {
            let raw = unsafe { sys::wgpuDeviceCreateRenderPipeline(dv.0.raw, &cdesc) };
            drop(a);
            NativeRenderPipeline::wrap(&dv.0, raw, &label)
        });
    }
    let slot = Slot::<Result<sys::WGPURenderPipeline, Exception>>::new();
    let (bx, done) = callback_box(&slot, ());
    let info = sys::WGPUCreateRenderPipelineAsyncCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: callback_mode(),
        callback: Some(on_render_pipeline),
        userdata1: bx,
        userdata2: std::ptr::null_mut(),
    };
    let fut = unsafe { sys::wgpuDeviceCreateRenderPipelineAsync(dev.0.raw, &cdesc, info) };
    drop(a);
    track(&dev.0.inst, fut, Timeline::Cpu, Some(&dev.0), done);
    let shared = dev.0.clone();
    let label = d.label.clone();
    Box::pin(async move {
        let raw = SlotFuture { inst: shared.inst.clone(), slot }.await?;
        Ok(NativeRenderPipeline::wrap(&shared, raw, &label))
    })
}

// ---------------------------------------------------------------------------------------------------------------------
// Query sets
// ---------------------------------------------------------------------------------------------------------------------

pub(crate) struct QuerySetInner {
    base: Base<sys::WGPUQuerySet>,
    ty: QueryType,
    count: u32,
}

/// `GPUQuerySet`.
#[derive(Clone)]
pub struct NativeQuerySet(Rc<QuerySetInner>);

impl fmt::Debug for NativeQuerySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NativeQuerySet({:?})", self.0.base.label.borrow())
    }
}

impl NativeQuerySet {
    pub(crate) fn raw(&self) -> sys::WGPUQuerySet {
        self.0.base.raw
    }
}

pub(crate) fn create_query_set(dev: &NativeDevice, d: &QuerySetDescriptor) -> Result<NativeQuerySet, Exception> {
    if d.r#type == QueryType::Timestamp && !dev.0.has(FeatureName::TimestampQuery) {
        return Err(Exception::new(
            ExceptionKind::TypeError,
            "query type \"timestamp\" requires feature \"timestamp-query\"",
        ));
    }
    let cdesc = desc::query_set_descriptor(d)?;
    let raw = unsafe { sys::wgpuDeviceCreateQuerySet(dev.0.raw, &cdesc) };
    Ok(NativeQuerySet(Rc::new(QuerySetInner {
        base: Base { raw, label: RefCell::new(d.label.clone()), dev: dev.0.clone(), release: sys::wgpuQuerySetRelease },
        ty: d.r#type,
        count: d.count,
    })))
}

impl Object for NativeQuerySet {
    fn label(&self) -> String {
        self.0.base.label.borrow().clone()
    }
    fn set_label(&self, label: &str) {
        *self.0.base.label.borrow_mut() = label.to_string();
        set_backend_label(|l| unsafe { sys::wgpuQuerySetSetLabel(self.0.base.raw, l) }, label);
    }
}

impl QuerySet<Native> for NativeQuerySet {
    fn destroy(&self) {
        unsafe { sys::wgpuQuerySetDestroy(self.0.base.raw) };
    }
    fn r#type(&self) -> QueryType {
        self.0.ty
    }
    fn count(&self) -> u32 {
        self.0.count
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------------------------------------------------

pub(crate) fn create_command_encoder(dev: &NativeDevice, d: &CommandEncoderDescriptor) -> NativeCommandEncoder {
    let cdesc = sys::WGPUCommandEncoderDescriptor { nextInChain: std::ptr::null_mut(), label: sv(&d.label) };
    let raw = unsafe { sys::wgpuDeviceCreateCommandEncoder(dev.0.raw, &cdesc) };
    NativeCommandEncoder::wrap(&dev.0, raw, &d.label)
}

pub(crate) fn create_render_bundle_encoder(
    dev: &NativeDevice,
    d: &RenderBundleEncoderDescriptor,
) -> Result<NativeRenderBundleEncoder, Exception> {
    for f in d.color_formats.iter().flatten() {
        desc::check_format(&dev.0, *f)?;
    }
    if let Some(f) = d.depth_stencil_format {
        desc::check_format(&dev.0, f)?;
    }
    let mut a = Arena::default();
    let cdesc = desc::render_bundle_encoder_descriptor(&mut a, d)?;
    let raw = unsafe { sys::wgpuDeviceCreateRenderBundleEncoder(dev.0.raw, &cdesc) };
    Ok(NativeRenderBundleEncoder::wrap(&dev.0, raw, &d.label))
}

impl DebugCommands for NativeCommandEncoder {
    fn push_debug_group(&self, l: &str) {
        unsafe { sys::wgpuCommandEncoderPushDebugGroup(self.0.raw, sv(l)) };
    }
    fn pop_debug_group(&self) {
        unsafe { sys::wgpuCommandEncoderPopDebugGroup(self.0.raw) };
    }
    fn insert_debug_marker(&self, l: &str) {
        unsafe { sys::wgpuCommandEncoderInsertDebugMarker(self.0.raw, sv(l)) };
    }
}

impl CommandEncoder<Native> for NativeCommandEncoder {
    fn begin_render_pass(&self, d: &RenderPassDescriptor<'_, Native>) -> Result<NativeRenderPassEncoder, Exception> {
        let mut a = Arena::default();
        let mut views = Vec::new();
        let cdesc = desc::render_pass_descriptor(&mut a, d, &mut views)?;
        let raw = unsafe { sys::wgpuCommandEncoderBeginRenderPass(self.0.raw, &cdesc) };
        drop(views);
        Ok(NativeRenderPassEncoder::wrap(&self.0.dev, raw, &d.label))
    }

    fn begin_compute_pass(&self, d: &ComputePassDescriptor<'_, Native>) -> Result<NativeComputePassEncoder, Exception> {
        let mut a = Arena::default();
        let cdesc = desc::compute_pass_descriptor(&mut a, d)?;
        let raw = unsafe { sys::wgpuCommandEncoderBeginComputePass(self.0.raw, &cdesc) };
        Ok(NativeComputePassEncoder::wrap(&self.0.dev, raw, &d.label))
    }

    fn copy_buffer_to_buffer(&self, source: &NativeBuffer, destination: &NativeBuffer, size: Option<Size64>) {
        self.copy_buffer_to_buffer_with_offsets(source, Size64::new_const(0), destination, Size64::new_const(0), size)
    }

    fn copy_buffer_to_buffer_with_offsets(
        &self,
        source: &NativeBuffer,
        source_offset: Size64,
        destination: &NativeBuffer,
        destination_offset: Size64,
        size: Option<Size64>,
    ) {
        // Device timeline: an undefined size is source.size − sourceOffset. An offset past the end makes that size
        // negative and the copy invalid; zero bytes at that offset is invalid for the same reason
        // (sourceOffset + size ≤ source.size fails).
        let size = match size {
            Some(s) => s.get(),
            None => source.0.size.saturating_sub(source_offset.get()),
        };
        unsafe {
            sys::wgpuCommandEncoderCopyBufferToBuffer(
                self.0.raw,
                source.raw(),
                source_offset.get(),
                destination.raw(),
                destination_offset.get(),
                size,
            )
        };
    }

    fn copy_buffer_to_texture(
        &self,
        source: &TexelCopyBufferInfo<'_, Native>,
        destination: &TexelCopyTextureInfo<'_, Native>,
        copy_size: &Extent3D,
    ) -> Result<(), Exception> {
        let dst = desc::texel_copy_texture_info(destination)?;
        let size = desc::extent(copy_size)?;
        let src = desc::texel_copy_buffer_info(source)?;
        unsafe { sys::wgpuCommandEncoderCopyBufferToTexture(self.0.raw, &src, &dst, &size) };
        Ok(())
    }

    fn copy_texture_to_buffer(
        &self,
        source: &TexelCopyTextureInfo<'_, Native>,
        destination: &TexelCopyBufferInfo<'_, Native>,
        copy_size: &Extent3D,
    ) -> Result<(), Exception> {
        let src = desc::texel_copy_texture_info(source)?;
        let size = desc::extent(copy_size)?;
        let dst = desc::texel_copy_buffer_info(destination)?;
        unsafe { sys::wgpuCommandEncoderCopyTextureToBuffer(self.0.raw, &src, &dst, &size) };
        Ok(())
    }

    fn copy_texture_to_texture(
        &self,
        source: &TexelCopyTextureInfo<'_, Native>,
        destination: &TexelCopyTextureInfo<'_, Native>,
        copy_size: &Extent3D,
    ) -> Result<(), Exception> {
        let src = desc::texel_copy_texture_info(source)?;
        let dst = desc::texel_copy_texture_info(destination)?;
        let size = desc::extent(copy_size)?;
        unsafe { sys::wgpuCommandEncoderCopyTextureToTexture(self.0.raw, &src, &dst, &size) };
        Ok(())
    }

    fn clear_buffer(&self, buffer: &NativeBuffer, offset: Size64, size: Option<Size64>) {
        unsafe {
            sys::wgpuCommandEncoderClearBuffer(
                self.0.raw,
                buffer.raw(),
                offset.get(),
                desc::size_or_rest(size, buffer.0.size, offset),
            )
        };
    }

    fn resolve_query_set(
        &self,
        q: &NativeQuerySet,
        first: u32,
        count: u32,
        destination: &NativeBuffer,
        destination_offset: Size64,
    ) {
        unsafe {
            sys::wgpuCommandEncoderResolveQuerySet(
                self.0.raw,
                q.raw(),
                first,
                count,
                destination.raw(),
                destination_offset.get(),
            )
        };
    }

    fn finish(&self, d: &CommandBufferDescriptor) -> NativeCommandBuffer {
        let cdesc = sys::WGPUCommandBufferDescriptor { nextInChain: std::ptr::null_mut(), label: sv(&d.label) };
        let raw = unsafe { sys::wgpuCommandEncoderFinish(self.0.raw, &cdesc) };
        NativeCommandBuffer::wrap(&self.0.dev, raw, &d.label)
    }
}

/// `setBindGroup` with a `Uint32Array`: the content-timeline range check, then the selected offsets.
fn offsets_from_data(data: &[u32], start: Size64, length: u32) -> Result<&[u32], Exception> {
    let s = start.get();
    match s.checked_add(u64::from(length)) {
        Some(end) if end <= data.len() as u64 => Ok(&data[s as usize..end as usize]),
        _ => Err(Exception::new(
            ExceptionKind::RangeError,
            format!(
                "dynamicOffsetsDataStart {s} + dynamicOffsetsDataLength {length} exceeds the array length {}",
                data.len()
            ),
        )),
    }
}

/// `setImmediates`: the content-timeline checks, then the selected bytes (the data is an `ArrayBuffer`).
fn immediates(data: &[u8], data_offset: Size64, data_size: Option<Size64>) -> Result<&[u8], Exception> {
    let n = data.len() as i128;
    let off = data_offset.get() as i128;
    let size = match data_size {
        Some(s) => s.get() as i128,
        None => n - off,
    };
    if size < 0 || off + size > n || size % 4 != 0 {
        return Err(Exception::new(
            ExceptionKind::OperationError,
            format!(
                "setImmediates: dataOffset {off} and size {data_size:?} do not select a multiple of 4 bytes within {n} bytes"
            ),
        ));
    }
    Ok(&data[off as usize..(off + size) as usize])
}

macro_rules! debug_and_binding {
    ($ty:ident, $push:path, $pop:path, $marker:path, $set_bg:path, $set_imm:path) => {
        impl DebugCommands for $ty {
            fn push_debug_group(&self, l: &str) {
                unsafe { $push(self.0.raw, sv(l)) };
            }
            fn pop_debug_group(&self) {
                unsafe { $pop(self.0.raw) };
            }
            fn insert_debug_marker(&self, l: &str) {
                unsafe { $marker(self.0.raw, sv(l)) };
            }
        }

        impl BindingCommands<Native> for $ty {
            fn set_bind_group(&self, index: u32, bg: Option<&NativeBindGroup>, offsets: &[u32]) {
                let raw = bg.map_or(std::ptr::null_mut(), |b| b.raw());
                unsafe { $set_bg(self.0.raw, index, raw, offsets.len(), offsets.as_ptr()) };
            }
            fn set_bind_group_with_data(
                &self,
                index: u32,
                bg: Option<&NativeBindGroup>,
                data: &[u32],
                start: Size64,
                length: u32,
            ) -> Result<(), Exception> {
                let offsets = offsets_from_data(data, start, length)?;
                self.set_bind_group(index, bg, offsets);
                Ok(())
            }
            fn set_immediates(
                &self,
                range_offset: u32,
                data: &[u8],
                data_offset: Size64,
                data_size: Option<Size64>,
            ) -> Result<(), Exception> {
                let bytes = immediates(data, data_offset, data_size)?;
                unsafe { $set_imm(self.0.raw, range_offset, bytes.as_ptr().cast(), bytes.len()) };
                Ok(())
            }
        }
    };
}

debug_and_binding!(
    NativeComputePassEncoder,
    sys::wgpuComputePassEncoderPushDebugGroup,
    sys::wgpuComputePassEncoderPopDebugGroup,
    sys::wgpuComputePassEncoderInsertDebugMarker,
    sys::wgpuComputePassEncoderSetBindGroup,
    sys::wgpuComputePassEncoderSetImmediates
);
debug_and_binding!(
    NativeRenderPassEncoder,
    sys::wgpuRenderPassEncoderPushDebugGroup,
    sys::wgpuRenderPassEncoderPopDebugGroup,
    sys::wgpuRenderPassEncoderInsertDebugMarker,
    sys::wgpuRenderPassEncoderSetBindGroup,
    sys::wgpuRenderPassEncoderSetImmediates
);
debug_and_binding!(
    NativeRenderBundleEncoder,
    sys::wgpuRenderBundleEncoderPushDebugGroup,
    sys::wgpuRenderBundleEncoderPopDebugGroup,
    sys::wgpuRenderBundleEncoderInsertDebugMarker,
    sys::wgpuRenderBundleEncoderSetBindGroup,
    sys::wgpuRenderBundleEncoderSetImmediates
);

macro_rules! render_commands {
    ($ty:ident, $pipe:path, $index:path, $vertex:path, $draw:path, $draw_indexed:path, $indirect:path, $indexed_indirect:path) => {
        impl RenderCommands<Native> for $ty {
            fn set_pipeline(&self, p: &NativeRenderPipeline) {
                unsafe { $pipe(self.0.raw, p.raw()) };
            }
            fn set_index_buffer(&self, b: &NativeBuffer, f: IndexFormat, offset: Size64, size: Option<Size64>) {
                let f = conv::to_c(f).expect("webgpu.h defines every GPUIndexFormat");
                unsafe {
                    $index(
                        self.0.raw,
                        b.raw(),
                        f as sys::WGPUIndexFormat,
                        offset.get(),
                        desc::size_or_rest(size, b.0.size, offset),
                    )
                };
            }
            fn set_vertex_buffer(&self, slot: u32, b: Option<&NativeBuffer>, offset: Size64, size: Option<Size64>) {
                // A null buffer has size 0 (setVertexBuffer, device timeline steps).
                let raw = b.map_or(std::ptr::null_mut(), |b| b.raw());
                let buffer_size = b.map_or(0, |b| b.0.size);
                unsafe { $vertex(self.0.raw, slot, raw, offset.get(), desc::size_or_rest(size, buffer_size, offset)) };
            }
            fn draw(&self, vc: u32, ic: u32, fv: u32, fi: u32) {
                unsafe { $draw(self.0.raw, vc, ic, fv, fi) };
            }
            fn draw_indexed(&self, ic: u32, inst: u32, fi: u32, bv: i32, finst: u32) {
                unsafe { $draw_indexed(self.0.raw, ic, inst, fi, bv, finst) };
            }
            fn draw_indirect(&self, b: &NativeBuffer, off: Size64) {
                unsafe { $indirect(self.0.raw, b.raw(), off.get()) };
            }
            fn draw_indexed_indirect(&self, b: &NativeBuffer, off: Size64) {
                unsafe { $indexed_indirect(self.0.raw, b.raw(), off.get()) };
            }
        }
    };
}

render_commands!(
    NativeRenderPassEncoder,
    sys::wgpuRenderPassEncoderSetPipeline,
    sys::wgpuRenderPassEncoderSetIndexBuffer,
    sys::wgpuRenderPassEncoderSetVertexBuffer,
    sys::wgpuRenderPassEncoderDraw,
    sys::wgpuRenderPassEncoderDrawIndexed,
    sys::wgpuRenderPassEncoderDrawIndirect,
    sys::wgpuRenderPassEncoderDrawIndexedIndirect
);
render_commands!(
    NativeRenderBundleEncoder,
    sys::wgpuRenderBundleEncoderSetPipeline,
    sys::wgpuRenderBundleEncoderSetIndexBuffer,
    sys::wgpuRenderBundleEncoderSetVertexBuffer,
    sys::wgpuRenderBundleEncoderDraw,
    sys::wgpuRenderBundleEncoderDrawIndexed,
    sys::wgpuRenderBundleEncoderDrawIndirect,
    sys::wgpuRenderBundleEncoderDrawIndexedIndirect
);

impl ComputePassEncoder<Native> for NativeComputePassEncoder {
    fn set_pipeline(&self, p: &NativeComputePipeline) {
        unsafe { sys::wgpuComputePassEncoderSetPipeline(self.0.raw, p.raw()) };
    }
    fn dispatch_workgroups(&self, x: u32, y: u32, z: u32) {
        unsafe { sys::wgpuComputePassEncoderDispatchWorkgroups(self.0.raw, x, y, z) };
    }
    fn dispatch_workgroups_indirect(&self, b: &NativeBuffer, off: Size64) {
        unsafe { sys::wgpuComputePassEncoderDispatchWorkgroupsIndirect(self.0.raw, b.raw(), off.get()) };
    }
    fn end(&self) {
        unsafe { sys::wgpuComputePassEncoderEnd(self.0.raw) };
    }
}

impl RenderPassEncoder<Native> for NativeRenderPassEncoder {
    fn set_viewport(&self, x: Float32, y: Float32, w: Float32, h: Float32, min_depth: Float32, max_depth: Float32) {
        unsafe {
            sys::wgpuRenderPassEncoderSetViewport(
                self.0.raw,
                x.get(),
                y.get(),
                w.get(),
                h.get(),
                min_depth.get(),
                max_depth.get(),
            )
        };
    }
    fn set_scissor_rect(&self, x: u32, y: u32, w: u32, h: u32) {
        unsafe { sys::wgpuRenderPassEncoderSetScissorRect(self.0.raw, x, y, w, h) };
    }
    fn set_blend_constant(&self, color: &Color) -> Result<(), Exception> {
        let c = desc::color(color)?;
        unsafe { sys::wgpuRenderPassEncoderSetBlendConstant(self.0.raw, &c) };
        Ok(())
    }
    fn set_stencil_reference(&self, r: u32) {
        unsafe { sys::wgpuRenderPassEncoderSetStencilReference(self.0.raw, r) };
    }
    fn begin_occlusion_query(&self, i: u32) {
        unsafe { sys::wgpuRenderPassEncoderBeginOcclusionQuery(self.0.raw, i) };
    }
    fn end_occlusion_query(&self) {
        unsafe { sys::wgpuRenderPassEncoderEndOcclusionQuery(self.0.raw) };
    }
    fn execute_bundles(&self, bundles: &[&NativeRenderBundle]) {
        let raws: Vec<sys::WGPURenderBundle> = bundles.iter().map(|b| b.raw()).collect();
        unsafe { sys::wgpuRenderPassEncoderExecuteBundles(self.0.raw, raws.len(), raws.as_ptr()) };
    }
    fn end(&self) {
        unsafe { sys::wgpuRenderPassEncoderEnd(self.0.raw) };
    }
}

impl RenderBundleEncoder<Native> for NativeRenderBundleEncoder {
    fn finish(&self, d: &RenderBundleDescriptor) -> NativeRenderBundle {
        let cdesc = sys::WGPURenderBundleDescriptor { nextInChain: std::ptr::null_mut(), label: sv(&d.label) };
        let raw = unsafe { sys::wgpuRenderBundleEncoderFinish(self.0.raw, &cdesc) };
        NativeRenderBundle::wrap(&self.0.dev, raw, &d.label)
    }
}
