//! The presentation sink: frames the embedder's display model swaps, shown on the display's target.
//!
//! A frame names a plane of an IOSurface or a task's texture, the part of it shown and where it lands, an orientation
//! transform and the earliest time it may appear. [`Presenter::present`] checks it when it is requested and then, in
//! order per display:
//!
//! 1. waits until the command buffers committed before the request that can write the frame's source have completed
//!    ([`Executor::committed_work_done_where`], with the records that write textures from [`crate::exec::writes`]),
//!    so that their writes are in the surface's memory (the coherence protocol writes them back before completion,
//!    plan section 6.2) or in the texture. Other command buffers, of this task or any other, are not waited for: a
//!    queue held by an event no one signals holds a frame only when it can write what the frame shows;
//! 2. waits until its time: the frame's own time (`presentAtTime:`), and the previous frame's presentation plus the
//!    request's minimum duration (`presentAfterMinimumDuration:`); a frame is never shown earlier;
//! 3. reads the shown rows from the surface's memory (a texture over an IOSurface plane is that memory) or reads them
//!    back from the texture, arranges them as canvas texels ([`super::format`]), configures the target for the display
//!    and the format when the configuration changed ([`super::canvas`]), writes the frame into the target's current
//!    texture with `GPUQueue.writeTexture()` and presents the target;
//! 4. tells the embedder ([`Embedder::present`]) and reports when the frame reached the target.
//!
//! The canvas texture then holds the surface's pixel codes exactly. "At" a time cannot be honored exactly: a canvas is
//! shown at the browser's next rendering update after the frame is written, and the presentation time reported is the
//! embedder's clock when the frame reached the target, not a scan-out time (plan section 9, row 9). Everything the
//! sink cannot show exactly is refused with a [`PresentFailure`] naming the reason; nothing is approximated.

use std::cell::{Cell, RefCell, RefMut};
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

use webmetal_wire::{Failure, status};

use webmetal_wire::{Handle, SurfaceId};

use crate::embed::{Embedder, MonotonicNanos, PresentError, PresentSource, PresentedFrame, Rect, TaskId};
use crate::exec::writes::{TextureWrite, texture_writes};
use crate::exec::{CommitInfo, Executor, Kind, Report, Table, failure};
use crate::format::handling::{Refusal, Resolution};
use crate::gpu::idl::{ErrorFilter, TextureFormat};
use crate::gpu::{Api, Device as _, DeviceLostInfo, GpuFuture};
use crate::mem::layout::TextureType;
use crate::mem::options::StorageMode;
use crate::mem::texture::{Texture, idl_format, usage};

use super::canvas::{self, CanvasPlan, DisplayMode, PlanError};
use super::format::{FramebufferFormat, code_name, framebuffer_format, texture_format};
use super::target::{Target, read_texture};

/// `MTLTextureUsageShaderAtomic` (`MTLTexture.h`, iOS 17): atomic operations in a shader, which write.
const SHADER_ATOMIC: u64 = 0x0020;

/// A frame that asks for an orientation transform.
pub const TRANSFORM: Refusal = Refusal {
    id: "present-transform",
    reason: "The frame asks for an orientation transform. Its encoding is the embedder's display model's and the sink \
             shows frames untransformed only.",
    source: "host/src/embed.rs (PresentedFrame::transform); tables/limits.toml question system-behavior-items (K3)",
};

/// A frame whose shown part is scaled.
pub const SCALING: Refusal = Refusal {
    id: "present-scaling",
    reason: "The frame's source and target rectangles differ in size; the filter the display hardware scales with is \
             not established.",
    source: "tables/limits.toml question system-behavior-items (K3, K5)",
};

/// A frame that does not cover the display.
pub const PARTIAL_FRAME: Refusal = Refusal {
    id: "present-partial-frame",
    reason: "The frame's target rectangle is not the whole display; what the display shows outside it is not \
             established.",
    source: "tables/limits.toml question system-behavior-items (K3)",
};

/// The host's canvas refused the configuration.
pub const HOST_CANVAS: Refusal = Refusal {
    id: "present-host-canvas",
    reason: "The host's canvas refused the configuration the display needs (a value the browser does not implement).",
    source: "WebGPU §21.4 (GPUCanvasConfiguration)",
};

/// Why the sink did not show a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct PresentFailure {
    /// The refusal's id, when the sink refused the frame for a listed reason.
    pub refusal: Option<&'static str>,
    /// Status 2 (not implemented), 3 (the host cannot) or 4 (the request is malformed), as the executor reports them.
    pub failure: Failure,
}

impl PresentFailure {
    fn refused(r: Refusal, detail: impl AsRef<str>) -> PresentFailure {
        PresentFailure {
            refusal: Some(r.id),
            failure: failure::unimplemented(format!("{}: {} ({}); {}", r.id, r.reason, r.source, detail.as_ref())),
        }
    }

    fn host(r: Refusal, detail: impl AsRef<str>) -> PresentFailure {
        PresentFailure {
            refusal: Some(r.id),
            failure: failure::host_unsupported(format!("{}: {} ({}); {}", r.id, r.reason, r.source, detail.as_ref())),
        }
    }

    fn invalid(message: impl Into<String>) -> PresentFailure {
        PresentFailure { refusal: None, failure: failure::invalid(message) }
    }

    fn defect(message: impl Into<String>) -> PresentFailure {
        PresentFailure { refusal: None, failure: failure::unimplemented(message) }
    }

    fn lost(info: &DeviceLostInfo) -> PresentFailure {
        PresentFailure { refusal: Some("device-removed"), failure: failure::device_removed(info) }
    }

    /// The failure as an executor report for `task`, for the embedder's not-implemented channel (status 2 and 3).
    pub fn report(&self, task: crate::embed::TaskId) -> Option<Report> {
        matches!(self.failure.status, status::UNIMPLEMENTED | status::HOST_UNSUPPORTED).then(|| Report {
            task,
            command_buffer: None,
            opcode: None,
            offset: None,
            failure: self.failure.clone(),
        })
    }
}

/// A framebuffer swap: the frame, and the minimum time since the display's previous frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapRequest {
    /// The frame. Its `time` is the earliest time it may appear (`presentAtTime:`).
    pub frame: PresentedFrame,
    /// The frame may appear no earlier than this many nanoseconds after the display's previous frame was presented
    /// (`presentAfterMinimumDuration:`).
    pub minimum_duration: Option<u64>,
}

/// A frame that reached its display's target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Presented {
    pub display: u32,
    /// The frame's position among the frames requested for the display, from 0.
    pub sequence: u64,
    /// When the frame reached the target, on the embedder's clock.
    pub presented_at: MonotonicNanos,
    /// The earliest time the frame could appear, when its request bounded it.
    pub not_before: Option<MonotonicNanos>,
    /// The canvas configuration the frame was shown with.
    pub plan: CanvasPlan,
}

struct Display<A: Api> {
    /// Distinguishes this attachment from earlier ones of the same display number.
    generation: u64,
    mode: DisplayMode,
    target: Box<dyn Target<A>>,
    /// The configuration the target has.
    configured: Option<CanvasPlan>,
    /// Frames requested, and frames finished (shown or failed): the frame with sequence `finished` is the one whose
    /// turn it is.
    requested: u64,
    finished: u64,
    /// Frames whose futures were dropped before their turn: their turn passes when it comes.
    abandoned: BTreeSet<u64>,
    last_presented: Option<MonotonicNanos>,
    turn_wakers: Vec<Waker>,
}

struct Shared<A: Api, E: Embedder> {
    exec: Executor<A, E>,
    device: A::Device,
    queue: A::Queue,
    embedder: Rc<E>,
    max_texture_dimension_2d: u32,
    displays: RefCell<HashMap<u32, Display<A>>>,
    timers: RefCell<Vec<(MonotonicNanos, Waker)>>,
    lost_watch: RefCell<Option<GpuFuture<DeviceLostInfo>>>,
    lost: RefCell<Option<DeviceLostInfo>>,
    generations: Cell<u64>,
}

/// The presentation sink of one executor. Cloning refers to the same sink.
pub struct Presenter<A: Api, E: Embedder> {
    shared: Rc<Shared<A, E>>,
}

impl<A: Api, E: Embedder> Clone for Presenter<A, E> {
    fn clone(&self) -> Presenter<A, E> {
        Presenter { shared: self.shared.clone() }
    }
}

/// Where a frame's pixels are.
enum Source<A: Api> {
    /// Rows of an IOSurface plane's memory: the plane's offset in the surface and its bytes per row.
    Plane { id: SurfaceId, offset: u64, bytes_per_row: u64 },
    /// Level 0, slice 0 of a texture the GPU holds, stored in `format`.
    Texture { texture: Rc<Texture<A>>, format: TextureFormat },
}

/// The objects whose writes a frame shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shown {
    /// Every texture over this plane of this surface, in any task.
    Plane(SurfaceId, u32),
    /// This texture of this task.
    Texture(TaskId, Handle),
}

/// A frame checked at its request: what to read and how to show it.
struct Checked<A: Api> {
    display: u32,
    generation: u64,
    source: Source<A>,
    shown: Shown,
    format: FramebufferFormat,
    rect: Rect,
    plan: CanvasPlan,
}

/// Whether a write of `how` to `t` can change its contents: attachments and blits write; a shader writes only a
/// texture whose usage allows it (`ShaderWrite`, `ShaderAtomic`, or `Unknown`, which allows every use; `MTLTexture.h`).
fn can_write<A: Api>(t: &Texture<A>, how: TextureWrite) -> bool {
    match how {
        TextureWrite::Attachment | TextureWrite::Blit => true,
        TextureWrite::ShaderBinding | TextureWrite::UseResource => {
            t.usage == usage::UNKNOWN || t.usage & (usage::SHADER_WRITE | SHADER_ATOMIC) != 0
        }
    }
}

/// Whether the committed command buffer `info` can write what a frame shows. A handle its records name in a writing
/// position that no longer resolves (the guest released it after the commit) may have been the source, so it counts.
fn writes_shown<A: Api>(info: &CommitInfo, table: &Table, shown: Shown) -> bool {
    if let Shown::Texture(task, _) = shown
        && info.task != task
    {
        return false;
    }
    let mut hit = false;
    for (_, record) in info.records.iter() {
        texture_writes(record, &mut |h, how| {
            if hit {
                return;
            }
            let Some(entry) = table.get(h) else {
                hit = true;
                return;
            };
            let Some(t) = entry.payload.clone().and_then(|p| p.downcast::<Texture<A>>().ok()) else { return };
            let target = match shown {
                Shown::Plane(id, plane) => t.surface.as_ref().is_some_and(|p| p.id == id && p.plane == plane),
                Shown::Texture(_, handle) => h == handle,
            };
            hit = target && can_write(&t, how);
        });
        if hit {
            return true;
        }
    }
    false
}

impl<A: Api, E: Embedder + 'static> Presenter<A, E> {
    /// The sink of `exec`, whose device is `device`.
    pub fn new(exec: Executor<A, E>, device: A::Device, embedder: Rc<E>) -> Presenter<A, E> {
        let queue = device.queue();
        let max = device.limits().max_texture_dimension_2d.unwrap_or(0);
        let lost_watch = device.lost();
        Presenter {
            shared: Rc::new(Shared {
                exec,
                device,
                queue,
                embedder,
                max_texture_dimension_2d: max,
                displays: RefCell::new(HashMap::new()),
                timers: RefCell::new(Vec::new()),
                lost_watch: RefCell::new(Some(lost_watch)),
                lost: RefCell::new(None),
                generations: Cell::new(0),
            }),
        }
    }

    /// Show `display`'s frames on `target`, sized to the display's mode. The target is configured by the first frame.
    pub fn attach(
        &self,
        display: u32,
        mode: DisplayMode,
        mut target: Box<dyn Target<A>>,
    ) -> Result<(), PresentFailure> {
        match canvas::check_display(&mode, self.shared.max_texture_dimension_2d) {
            Ok(()) => {}
            Err(PlanError::Empty) => return Err(PresentFailure::invalid(format!("display {display} has no pixels"))),
            Err(PlanError::Refused(r)) => {
                return Err(PresentFailure::host(
                    r,
                    format!(
                        "display {display} is {}x{}, the device's maxTextureDimension2D is {}",
                        mode.width, mode.height, self.shared.max_texture_dimension_2d
                    ),
                ));
            }
        }
        let mut displays = self.shared.displays.borrow_mut();
        if displays.contains_key(&display) {
            return Err(PresentFailure::invalid(format!("display {display} is attached already")));
        }
        target.set_size(mode.width, mode.height);
        let generation = self.shared.generations.get() + 1;
        self.shared.generations.set(generation);
        displays.insert(
            display,
            Display {
                generation,
                mode,
                target,
                configured: None,
                requested: 0,
                finished: 0,
                abandoned: BTreeSet::new(),
                last_presented: None,
                turn_wakers: Vec::new(),
            },
        );
        Ok(())
    }

    /// Stop showing `display`: its target is unconfigured and handed back. Frames already requested for it fail.
    pub fn detach(&self, display: u32) -> Option<Box<dyn Target<A>>> {
        let mut d = self.shared.displays.borrow_mut().remove(&display)?;
        d.target.unconfigure();
        for w in d.turn_wakers.drain(..) {
            w.wake();
        }
        Some(d.target)
    }

    /// The earliest time a frame waits for, for the embedder to call [`Presenter::wake_due`] then.
    pub fn next_deadline(&self) -> Option<MonotonicNanos> {
        self.shared.timers.borrow().iter().map(|(t, _)| *t).min()
    }

    /// Wake the frames whose time has come.
    pub fn wake_due(&self) {
        let now = self.shared.embedder.now();
        let due: Vec<Waker> = {
            let mut timers = self.shared.timers.borrow_mut();
            let (due, rest): (Vec<_>, Vec<_>) = timers.drain(..).partition(|(t, _)| *t <= now);
            *timers = rest;
            due.into_iter().map(|(_, w)| w).collect()
        };
        for w in due {
            w.wake();
        }
    }

    /// Show a frame. The request is checked now; the returned future shows it once its turn, its source's work and
    /// its time have come.
    pub fn present(&self, request: SwapRequest) -> GpuFuture<Result<Presented, PresentFailure>> {
        let checked = match self.check(&request.frame) {
            Ok(c) => c,
            Err(f) => return Box::pin(async move { Err(f) }),
        };
        let display = checked.display;
        let sequence = {
            let mut displays = self.shared.displays.borrow_mut();
            let d = displays.get_mut(&display).expect("checked");
            d.requested += 1;
            d.requested - 1
        };
        // The command buffers committed before this point that can write the source are the ones whose writes the
        // frame shows.
        let shown = checked.shown;
        let work = self.shared.exec.committed_work_done_where(|info, table| writes_shown::<A>(info, table, shown));
        let shared = self.shared.clone();
        // The turn exists from the request on, so that dropping the future, even unpolled, passes it on.
        let generation = checked.generation;
        let mut turn = Turn { shared: shared.clone(), display, generation, sequence, done: false };
        Box::pin(async move {
            std::future::poll_fn(|cx| turn.poll(cx)).await?;
            work.await;
            let not_before = shared.not_before(display, generation, &request)?;
            if let Some(t) = not_before {
                std::future::poll_fn(|cx| shared.reached(t, cx)).await;
            }
            let shown = shared.show(&checked, &request.frame).await;
            turn.finish();
            let presented_at = shown?;
            Ok(Presented { display, sequence, presented_at, not_before, plan: checked.plan })
        })
    }

    /// Check a frame at its request, against the display and its source as they are now.
    fn check(&self, frame: &PresentedFrame) -> Result<Checked<A>, PresentFailure> {
        let displays = self.shared.displays.borrow();
        let d = displays
            .get(&frame.display)
            .ok_or_else(|| PresentFailure::invalid(format!("display {} is not attached", frame.display)))?;
        let mode = &d.mode;
        if frame.transform != mode.identity_transform {
            return Err(PresentFailure::refused(TRANSFORM, format!("transform {:#x}", frame.transform)));
        }
        let s = frame.source_rect;
        let t = frame.target_rect;
        if s.x < 0 || s.y < 0 || s.width == 0 || s.height == 0 {
            return Err(PresentFailure::invalid(format!("source rectangle {s:?}")));
        }
        if (s.width, s.height) != (t.width, t.height) {
            return Err(PresentFailure::refused(SCALING, format!("source {s:?}, target {t:?}")));
        }
        if (t.x, t.y, t.width, t.height) != (0, 0, u64::from(mode.width), u64::from(mode.height)) {
            return Err(PresentFailure::refused(
                PARTIAL_FRAME,
                format!("target {t:?} on a {}x{} display", mode.width, mode.height),
            ));
        }
        let (source, shown, format) = match frame.source {
            PresentSource::Surface { id, plane } => self.check_surface(id, plane, s)?,
            PresentSource::Texture { task, texture } => self.check_texture(task, texture, s)?,
        };
        let plan = canvas::plan(mode, &format).map_err(|e| match e {
            PlanError::Empty => PresentFailure::invalid(format!("display {} has no pixels", frame.display)),
            PlanError::Refused(r) => PresentFailure::refused(r, format!("display {}: {:?}", frame.display, mode)),
        })?;
        Ok(Checked { display: frame.display, generation: d.generation, source, shown, format, rect: s, plan })
    }

    /// A frame of plane `plane` of surface `id`.
    fn check_surface(
        &self,
        id: SurfaceId,
        plane: u32,
        s: Rect,
    ) -> Result<(Source<A>, Shown, FramebufferFormat), PresentFailure> {
        let surface = self.shared.embedder.surface_lookup(id).map_err(|e| PresentFailure::invalid(e.to_string()))?;
        let format = framebuffer_format(surface.pixel_format).map_err(|r| {
            PresentFailure::refused(r, format!("surface {} has pixel format {}", id.0, code_name(surface.pixel_format)))
        })?;
        let geometry = if surface.planes.is_empty() {
            if plane != 0 {
                return Err(PresentFailure::invalid(format!("plane {plane} of surface {}, which has no planes", id.0)));
            }
            crate::embed::SurfacePlane {
                width: surface.width,
                height: surface.height,
                bytes_per_row: surface.bytes_per_row,
                bytes_per_element: surface.bytes_per_element,
                element_width: surface.element_width,
                element_height: surface.element_height,
                offset: 0,
                size: surface.alloc_size,
            }
        } else {
            *surface.planes.get(plane as usize).ok_or_else(|| {
                PresentFailure::invalid(format!(
                    "plane {plane} of surface {}, which has {}",
                    id.0,
                    surface.planes.len()
                ))
            })?
        };
        if geometry.bytes_per_element != format.bytes_per_pixel
            || geometry.element_width != 1
            || geometry.element_height != 1
        {
            return Err(PresentFailure::invalid(format!(
                "surface {} in {} has {}-byte elements of {}x{} pixels",
                id.0, format.name, geometry.bytes_per_element, geometry.element_width, geometry.element_height
            )));
        }
        let (x, y) = (s.x as u64, s.y as u64);
        let row = s.width * format.bytes_per_pixel;
        if x + s.width > geometry.width
            || y + s.height > geometry.height
            || geometry.width * format.bytes_per_pixel > geometry.bytes_per_row
            || (y + s.height - 1) * geometry.bytes_per_row + x * format.bytes_per_pixel + row > geometry.size
        {
            return Err(PresentFailure::invalid(format!(
                "source {s:?} outside plane {plane} of surface {} ({}x{}, {} bytes per row, {} bytes)",
                id.0, geometry.width, geometry.height, geometry.bytes_per_row, geometry.size
            )));
        }
        let source = Source::Plane { id, offset: geometry.offset, bytes_per_row: geometry.bytes_per_row };
        Ok((source, Shown::Plane(id, plane), format))
    }

    /// A frame of level 0 of texture `handle` of `task`. A texture over an IOSurface plane is that plane's memory; any
    /// other is read back from the GPU.
    fn check_texture(
        &self,
        task: TaskId,
        handle: Handle,
        s: Rect,
    ) -> Result<(Source<A>, Shown, FramebufferFormat), PresentFailure> {
        let what = format!("texture {:#x} of task {}", handle.get(), task.0);
        let entry = self
            .shared
            .exec
            .object(task, handle)
            .ok_or_else(|| PresentFailure::invalid(format!("{what} is not a live object")))?;
        let t = (entry.kind == Kind::Texture)
            .then(|| entry.payload.clone().and_then(|p| p.downcast::<Texture<A>>().ok()))
            .flatten()
            .ok_or_else(|| PresentFailure::invalid(format!("{what} is not a texture")))?;
        let format = texture_format(t.format)
            .map_err(|r| PresentFailure::refused(r, format!("{what} has MTLPixelFormat {}", t.format)))?;
        let g = &t.geometry;
        if g.kind != TextureType::D2 || g.sample_count != 1 {
            return Err(PresentFailure::invalid(format!("{what} is not a 2D texture of one sample")));
        }
        if t.storage == StorageMode::Memoryless {
            return Err(PresentFailure::invalid(format!("{what} is memoryless: it keeps no contents")));
        }
        let (x, y) = (s.x as u64, s.y as u64);
        if x + s.width > u64::from(g.width) || y + s.height > u64::from(g.height) {
            return Err(PresentFailure::invalid(format!("source {s:?} outside {what} ({}x{})", g.width, g.height)));
        }
        if let Some(p) = &t.surface {
            let source = Source::Plane { id: p.id, offset: p.geometry.offset, bytes_per_row: p.geometry.bytes_per_row };
            return Ok((source, Shown::Plane(p.id, p.plane), format));
        }
        // The stored texture must hold the texel codes bit for bit: a non-packed unorm format, whose texel copies keep
        // every bit (WebGPU #texel-copies).
        let stored = match t.resolution {
            Resolution::Native(f) if f.texel_copy_exact(None) => idl_format(f),
            r => {
                return Err(PresentFailure::defect(format!(
                    "{what} in {} is stored as {:?}, whose texel copies do not keep its bits",
                    format.name, r
                )));
            }
        };
        Ok((Source::Texture { texture: t.clone(), format: stored }, Shown::Texture(task, handle), format))
    }
}

/// A frame's turn on its display. Finishing it, or dropping it unfinished, passes the turn to the next frame.
struct Turn<A: Api, E: Embedder> {
    shared: Rc<Shared<A, E>>,
    display: u32,
    generation: u64,
    sequence: u64,
    done: bool,
}

impl<A: Api, E: Embedder> Turn<A, E> {
    fn poll(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), PresentFailure>> {
        let Some(mut d) = self.shared.display(self.display, self.generation) else {
            self.done = true;
            return Poll::Ready(Err(PresentFailure::invalid(format!("display {} was detached", self.display))));
        };
        if d.finished == self.sequence {
            return Poll::Ready(Ok(()));
        }
        d.turn_wakers.push(cx.waker().clone());
        Poll::Pending
    }

    fn finish(&mut self) {
        if self.done {
            return;
        }
        self.done = true;
        let wakers = {
            let Some(mut d) = self.shared.display(self.display, self.generation) else { return };
            debug_assert_eq!(d.finished, self.sequence, "frames finish in order");
            d.finished += 1;
            let d = &mut *d;
            while d.abandoned.remove(&d.finished) {
                d.finished += 1;
            }
            std::mem::take(&mut d.turn_wakers)
        };
        for w in wakers {
            w.wake();
        }
    }
}

impl<A: Api, E: Embedder> Drop for Turn<A, E> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        // A frame dropped before it finished gives up its turn: now if the turn is its, else when the turn comes.
        let has_turn = {
            match self.shared.display(self.display, self.generation) {
                Some(d) if d.finished == self.sequence => true,
                Some(mut d) => {
                    d.abandoned.insert(self.sequence);
                    false
                }
                None => false,
            }
        };
        if has_turn {
            self.finish();
        }
        self.done = true;
    }
}

impl<A: Api, E: Embedder> Shared<A, E> {
    /// The display, if it is still the attachment `generation`.
    fn display(&self, display: u32, generation: u64) -> Option<RefMut<'_, Display<A>>> {
        RefMut::filter_map(self.displays.borrow_mut(), |m| m.get_mut(&display).filter(|d| d.generation == generation))
            .ok()
    }
}

impl<A: Api, E: Embedder + 'static> Shared<A, E> {
    /// The earliest time the frame may appear.
    fn not_before(
        &self,
        display: u32,
        generation: u64,
        request: &SwapRequest,
    ) -> Result<Option<MonotonicNanos>, PresentFailure> {
        let d = self
            .display(display, generation)
            .ok_or_else(|| PresentFailure::invalid(format!("display {display} was detached")))?;
        let after_previous = match (request.minimum_duration, d.last_presented) {
            (Some(n), Some(prev)) => Some(MonotonicNanos(prev.0.saturating_add(n))),
            _ => None,
        };
        Ok(match (request.frame.time, after_previous) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        })
    }

    fn reached(&self, t: MonotonicNanos, cx: &mut Context<'_>) -> Poll<()> {
        if self.embedder.now() >= t {
            return Poll::Ready(());
        }
        self.timers.borrow_mut().push((t, cx.waker().clone()));
        Poll::Pending
    }

    /// The device's loss, if it was lost.
    fn lost(&self) -> Option<DeviceLostInfo> {
        if self.lost.borrow().is_none()
            && let Some(w) = self.lost_watch.borrow_mut().as_mut()
            && let Poll::Ready(info) = w.as_mut().poll(&mut Context::from_waker(Waker::noop()))
        {
            *self.lost.borrow_mut() = Some(info);
        }
        self.lost.borrow().clone()
    }

    /// Read the frame's rows from the surface's memory.
    fn read_plane(
        &self,
        c: &Checked<A>,
        id: SurfaceId,
        offset: u64,
        bytes_per_row: u64,
    ) -> Result<Vec<u8>, PresentFailure> {
        let row = (c.rect.width * c.format.bytes_per_pixel) as usize;
        let mut out = vec![0u8; row * c.rect.height as usize];
        let (x, y) = (c.rect.x as u64, c.rect.y as u64);
        for (i, dst) in out.chunks_exact_mut(row).enumerate() {
            let at = offset + (y + i as u64) * bytes_per_row + x * c.format.bytes_per_pixel;
            self.embedder
                .read_surface(id, at, dst)
                .map_err(|e| PresentFailure::invalid(format!("reading the frame: {e}")))?;
        }
        Ok(out)
    }

    /// The frame's rows as canvas texels.
    fn frame_texels(&self, c: &Checked<A>) -> GpuFuture<Result<Vec<u8>, PresentFailure>> {
        let read = match &c.source {
            Source::Plane { id, offset, bytes_per_row } => {
                let r = self.read_plane(c, *id, *offset, *bytes_per_row);
                Box::pin(async move { r }) as GpuFuture<_>
            }
            Source::Texture { texture, format } => {
                // Level 0 of a texture whose stored copy is not current holds its contents only in the twin, which
                // these formats never need (`crate::mem::texture`); refuse rather than show stale texels.
                if !texture.stored_current(0, 0) {
                    let f = PresentFailure::defect(format!(
                        "the stored copy of texture \"{}\" is not current",
                        texture.label
                    ));
                    return Box::pin(async move { Err(f) });
                }
                let (x, y) = (c.rect.x as u32, c.rect.y as u32);
                let read = read_texture::<A>(
                    &self.device,
                    texture.gpu(),
                    *format,
                    (x, y),
                    c.rect.width as u32,
                    c.rect.height as u32,
                );
                Box::pin(async move {
                    read.await.map_err(|e| PresentFailure::defect(format!("reading the frame back: {e}")))
                })
            }
        };
        let format = c.format;
        Box::pin(async move {
            let mut data = read.await?;
            format.to_texels(&mut data);
            Ok(data)
        })
    }

    /// Write the frame into its display's target and present it. Resolves to when it reached the target.
    fn show(
        self: &Rc<Self>,
        c: &Checked<A>,
        frame: &PresentedFrame,
    ) -> GpuFuture<Result<MonotonicNanos, PresentFailure>> {
        if let Some(info) = self.lost() {
            return Box::pin(async move { Err(PresentFailure::lost(&info)) });
        }
        let texels = self.frame_texels(c);
        let shared = self.clone();
        let frame = *frame;
        let (display, generation) = (c.display, c.generation);
        let plan = c.plan.clone();
        let (width, height, row) =
            (c.rect.width as u32, c.rect.height as u32, (c.rect.width * c.format.bytes_per_pixel) as u32);
        Box::pin(async move {
            let data = texels.await?;
            if let Some(info) = shared.lost() {
                return Err(PresentFailure::lost(&info));
            }
            let filters = [ErrorFilter::Validation, ErrorFilter::OutOfMemory, ErrorFilter::Internal];
            for f in filters {
                shared.device.push_error_scope(f);
            }
            let written = shared.write(display, generation, &plan, &data, row, width, height);
            let pops: Vec<_> = filters.iter().map(|_| shared.device.pop_error_scope()).collect();
            let mut errors = Vec::new();
            for p in pops {
                match p.await {
                    Ok(Some(e)) => errors.push(e),
                    Ok(None) => {}
                    Err(e) => {
                        return Err(PresentFailure::defect(format!("popErrorScope rejected while presenting: {e}")));
                    }
                }
            }
            written?;
            // Errors are no longer reported once the device is lost (WebGPU §22), so a clean pop proves nothing then.
            if let Some(info) = shared.lost() {
                return Err(PresentFailure::lost(&info));
            }
            if let Some(e) = errors.iter().find(|e| e.filter == ErrorFilter::OutOfMemory) {
                return Err(PresentFailure::defect(format!("out of memory presenting display {display}: {e}")));
            }
            if !errors.is_empty() {
                let list: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
                return Err(PresentFailure::defect(format!(
                    "WebGPU reported an error the presentation sink should have prevented: {}",
                    list.join("; ")
                )));
            }
            let at = shared.embedder.now();
            if let Some(mut d) = shared.display(display, generation) {
                d.last_presented = Some(at);
            }
            shared.embedder.present(&frame).map_err(|e| match e {
                PresentError::UnknownDisplay(d) => {
                    PresentFailure::invalid(format!("the embedder has no display {d} (frame of display {display})"))
                }
                PresentError::Unsupported => PresentFailure {
                    refusal: None,
                    failure: failure::host_unsupported(format!(
                        "the embedder's display {display} cannot show the frame; it reports the detail itself"
                    )),
                },
            })?;
            Ok(at)
        })
    }

    /// Configure the display's target when the plan changed, write the frame (rows of `row` bytes) into it and present
    /// it.
    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        display: u32,
        generation: u64,
        plan: &CanvasPlan,
        data: &[u8],
        row: u32,
        width: u32,
        height: u32,
    ) -> Result<(), PresentFailure> {
        let mut d = self
            .display(display, generation)
            .ok_or_else(|| PresentFailure::invalid(format!("display {display} was detached")))?;
        if d.configured.as_ref() != Some(plan) {
            d.configured = None;
            d.target.configure(&plan.configuration::<A>(&self.device)).map_err(|e| {
                PresentFailure::host(HOST_CANVAS, format!("configure() of display {display} threw {e}"))
            })?;
            d.configured = Some(plan.clone());
        }
        d.target
            .write_frame(&self.queue, data, row, width, height)
            .map_err(|e| PresentFailure::defect(format!("writing the frame of display {display} threw {e}")))?;
        d.target.present().map_err(|e| PresentFailure::defect(format!("presenting display {display} threw {e}")))
    }
}
