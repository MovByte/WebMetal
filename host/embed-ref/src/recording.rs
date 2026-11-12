//! Recordings and the runner that replays them.
//!
//! A recording is what the executor and the display model saw of a guest system, in order: guest memory the tasks
//! mapped and wrote, IOSurfaces and their CPU writes, the control requests each task sent (with the replies they got,
//! when recorded), the points where the guest waited for its GPU work, displays and the framebuffer swaps shown on
//! them, and the clock. [`run`] replays one on a [`RefHost`]: replies are compared with the recorded ones and every
//! swap's outcome is reported, with the presented image when asked for.
//!
//! # File format
//!
//! Little-endian. A header of 16 bytes: the magic `WMRECRD\0`, the format version (u32, 1) and a reserved u32 (0).
//! Then records, each a header `{u32 kind, u32 reserved (0), u64 length}`, `length` payload bytes, and zero bytes to
//! the next multiple of 8. Payload fields are packed in the order [`Step`] lists them; byte strings are a u64 count
//! and the bytes; an optional value is a u8 (0 or 1) and the value when 1. A reader refuses any other magic, version,
//! kind, reserved value, tag, padding, or a payload with bytes left over or missing; it never guesses.

use std::fmt;

use webmetal_host::embed::{MonotonicNanos, PresentSource, PresentedFrame, Rect, SurfaceDesc, SurfacePlane, TaskId};
use webmetal_host::present::{DisplayColorSpace, DisplayMode, PresentFailure, Presented, SwapRequest};
use webmetal_wire::{Handle, SurfaceId};

use crate::host::RefHost;

/// One step of a recording.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// The task mapped `len` zeroed bytes at `addr`.
    Map { task: TaskId, addr: u64, len: u64 },
    /// The task's CPU wrote `bytes` at `addr`.
    Write { task: TaskId, addr: u64, bytes: Vec<u8> },
    /// The kernel created an IOSurface; its memory is zeroed.
    CreateSurface { desc: SurfaceDesc },
    /// A CPU wrote `bytes` into the surface's memory at `offset`, between lock and unlock.
    WriteSurface { id: SurfaceId, offset: u64, bytes: Vec<u8> },
    /// The task sent a control request (the record as the guest wrote it); `reply` is the reply it got, if recorded.
    Control { task: TaskId, request: Vec<u8>, reply: Option<Vec<u8>> },
    /// The guest waited until every command buffer committed so far had completed.
    Wait,
    /// A display joined the display model.
    Attach { display: u32, mode: DisplayMode },
    /// A framebuffer swap.
    Present { request: SwapRequest },
    /// The embedder's clock reads `now` from here on.
    SetTime { now: MonotonicNanos },
    /// The task exited.
    ExitTask { task: TaskId },
}

/// What one step did.
#[derive(Clone, Debug, PartialEq)]
pub enum StepOutcome {
    Done,
    /// The reply to a control request, and whether it equals the recorded one (`None`: none was recorded).
    Reply {
        reply: Vec<u8>,
        matches: Option<bool>,
    },
    /// A swap's outcome, with the bytes of the display's drawing buffer after it when the run captures frames.
    Presented {
        result: Result<Presented, PresentFailure>,
        drawing_buffer: Option<Vec<u8>>,
    },
    /// The step could not be performed (the recording is inconsistent with itself, or the executor refused a request
    /// record as undecodable).
    Failed(String),
}

/// Replay `steps` on `host`. With `capture`, each shown frame's drawing buffer is read back.
pub async fn run<A: webmetal_host::gpu::Api>(host: &RefHost<A>, steps: &[Step], capture: bool) -> Vec<StepOutcome> {
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        let o = match step {
            Step::Map { task, addr, len } => done(host.embedder.map(*task, *addr, *len)),
            Step::Write { task, addr, bytes } => {
                done(host.embedder.cpu_write(*task, *addr, bytes).map_err(|e| e.to_string()))
            }
            Step::CreateSurface { desc } => done(host.embedder.create_surface(desc.clone())),
            Step::WriteSurface { id, offset, bytes } => {
                done(host.embedder.surface_cpu_write(*id, *offset, bytes).map_err(|e| e.to_string()))
            }
            Step::Control { task, request, reply } => match host.control_bytes(*task, request).await {
                Ok(got) => {
                    let matches = reply.as_ref().map(|r| *r == got);
                    StepOutcome::Reply { reply: got, matches }
                }
                Err(e) => StepOutcome::Failed(format!("the request record does not decode: {e:?}")),
            },
            Step::Wait => {
                host.exec.committed_work_done(None).await;
                StepOutcome::Done
            }
            Step::Attach { display, mode } => match host.attach_offscreen(*display, mode.clone()) {
                Ok(_) => StepOutcome::Done,
                Err(f) => StepOutcome::Failed(format!("attaching display {display}: {}", f.failure.message)),
            },
            Step::Present { request } => {
                let result = host.presenter.present(*request).await;
                let mut drawing_buffer = None;
                if capture
                    && result.is_ok()
                    && let Some(t) = host.target(request.frame.display)
                {
                    match t.drawing_buffer().await {
                        Ok(b) => drawing_buffer = b,
                        Err(e) => {
                            out.push(StepOutcome::Failed(format!("reading the drawing buffer back: {e}")));
                            continue;
                        }
                    }
                }
                StepOutcome::Presented { result, drawing_buffer }
            }
            Step::SetTime { now } => {
                host.embedder.set_time(*now);
                host.presenter.wake_due();
                StepOutcome::Done
            }
            Step::ExitTask { task } => {
                host.exec.task_exited(*task);
                host.embedder.remove_task(*task);
                StepOutcome::Done
            }
        };
        out.push(o);
    }
    out
}

fn done(r: Result<(), String>) -> StepOutcome {
    match r {
        Ok(()) => StepOutcome::Done,
        Err(e) => StepOutcome::Failed(e),
    }
}

// File format ------------------------------------------------------------------------------------------------------

/// The magic of a recording file.
pub const MAGIC: [u8; 8] = *b"WMRECRD\0";
/// The format version this code reads and writes.
pub const VERSION: u32 = 1;

mod kind {
    pub const MAP: u32 = 1;
    pub const WRITE: u32 = 2;
    pub const CREATE_SURFACE: u32 = 3;
    pub const WRITE_SURFACE: u32 = 4;
    pub const CONTROL: u32 = 5;
    pub const WAIT: u32 = 6;
    pub const ATTACH: u32 = 7;
    pub const PRESENT: u32 = 8;
    pub const SET_TIME: u32 = 9;
    pub const EXIT_TASK: u32 = 10;
}

/// Why a recording file was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatError {
    /// Byte offset in the file.
    pub offset: usize,
    pub what: String,
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "recording file, byte {}: {}", self.offset, self.what)
    }
}

impl std::error::Error for FormatError {}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.u64(b.len() as u64);
        self.0.extend_from_slice(b);
    }
    fn opt_u64(&mut self, v: Option<u64>) {
        match v {
            Some(v) => {
                self.u8(1);
                self.u64(v);
            }
            None => self.u8(0),
        }
    }
    fn rect(&mut self, r: &Rect) {
        self.i64(r.x);
        self.i64(r.y);
        self.u64(r.width);
        self.u64(r.height);
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
    /// Offset of `b` in the file, for errors.
    base: usize,
}

impl<'a> Reader<'a> {
    fn err(&self, what: impl Into<String>) -> FormatError {
        FormatError { offset: self.base + self.at, what: what.into() }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], FormatError> {
        if self.b.len() - self.at < n {
            return Err(self.err(format!("{n} bytes needed, {} left", self.b.len() - self.at)));
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, FormatError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, FormatError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, FormatError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn bytes(&mut self) -> Result<Vec<u8>, FormatError> {
        let n = self.u64()?;
        let n = usize::try_from(n).map_err(|_| self.err("byte count too large"))?;
        Ok(self.take(n)?.to_vec())
    }
    fn flag(&mut self) -> Result<bool, FormatError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(self.err(format!("presence tag {v}"))),
        }
    }
    fn opt_u64(&mut self) -> Result<Option<u64>, FormatError> {
        Ok(if self.flag()? { Some(self.u64()?) } else { None })
    }
    fn rect(&mut self) -> Result<Rect, FormatError> {
        Ok(Rect { x: self.i64()?, y: self.i64()?, width: self.u64()?, height: self.u64()? })
    }
    fn task(&mut self) -> Result<TaskId, FormatError> {
        Ok(TaskId(self.u64()?))
    }
}

fn write_surface_desc(w: &mut Writer, d: &SurfaceDesc) {
    w.u32(d.id.0);
    for v in [d.width, d.height, d.bytes_per_row, d.bytes_per_element, d.element_width, d.element_height] {
        w.u64(v);
    }
    w.u32(d.pixel_format);
    w.u64(d.alloc_size);
    w.u32(d.seed);
    w.u64(d.planes.len() as u64);
    for p in &d.planes {
        for v in [p.width, p.height, p.bytes_per_row, p.bytes_per_element, p.element_width, p.element_height] {
            w.u64(v);
        }
        w.u64(p.offset);
        w.u64(p.size);
    }
}

fn read_surface_desc(r: &mut Reader<'_>) -> Result<SurfaceDesc, FormatError> {
    let id = SurfaceId(r.u32()?);
    let mut v = [0u64; 6];
    for x in &mut v {
        *x = r.u64()?;
    }
    let pixel_format = r.u32()?;
    let alloc_size = r.u64()?;
    let seed = r.u32()?;
    let count = r.u64()?;
    // Each plane is 64 bytes; a count the payload cannot hold is refused before anything is allocated for it.
    if count > ((r.b.len() - r.at) / 64) as u64 {
        return Err(r.err(format!("{count} planes")));
    }
    let mut planes = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let mut p = [0u64; 8];
        for x in &mut p {
            *x = r.u64()?;
        }
        planes.push(SurfacePlane {
            width: p[0],
            height: p[1],
            bytes_per_row: p[2],
            bytes_per_element: p[3],
            element_width: p[4],
            element_height: p[5],
            offset: p[6],
            size: p[7],
        });
    }
    Ok(SurfaceDesc {
        id,
        width: v[0],
        height: v[1],
        bytes_per_row: v[2],
        bytes_per_element: v[3],
        element_width: v[4],
        element_height: v[5],
        pixel_format,
        alloc_size,
        planes,
        seed,
    })
}

fn write_mode(w: &mut Writer, m: &DisplayMode) {
    w.u32(m.width);
    w.u32(m.height);
    w.u32(m.identity_transform);
    match &m.color_space {
        DisplayColorSpace::Srgb => w.u32(0),
        DisplayColorSpace::SrgbLinear => w.u32(1),
        DisplayColorSpace::DisplayP3 => w.u32(2),
        DisplayColorSpace::DisplayP3Linear => w.u32(3),
        DisplayColorSpace::Other(s) => {
            w.u32(4);
            w.bytes(s.as_bytes());
        }
    }
}

fn read_mode(r: &mut Reader<'_>) -> Result<DisplayMode, FormatError> {
    let width = r.u32()?;
    let height = r.u32()?;
    let identity_transform = r.u32()?;
    let color_space = match r.u32()? {
        0 => DisplayColorSpace::Srgb,
        1 => DisplayColorSpace::SrgbLinear,
        2 => DisplayColorSpace::DisplayP3,
        3 => DisplayColorSpace::DisplayP3Linear,
        4 => {
            let b = r.bytes()?;
            DisplayColorSpace::Other(String::from_utf8(b).map_err(|_| r.err("color space name is not UTF-8"))?)
        }
        t => return Err(r.err(format!("color space tag {t}"))),
    };
    Ok(DisplayMode { width, height, color_space, identity_transform })
}

fn write_request(w: &mut Writer, q: &SwapRequest) {
    let f = &q.frame;
    w.u32(f.display);
    match f.source {
        PresentSource::Surface { id, plane } => {
            w.u32(0);
            w.u32(id.0);
            w.u32(plane);
        }
        PresentSource::Texture { task, texture } => {
            w.u32(1);
            w.u64(task.0);
            w.u64(texture.get());
        }
    }
    w.rect(&f.source_rect);
    w.rect(&f.target_rect);
    w.u32(f.transform);
    w.opt_u64(f.time.map(|t| t.0));
    w.opt_u64(q.minimum_duration);
}

fn read_request(r: &mut Reader<'_>) -> Result<SwapRequest, FormatError> {
    let display = r.u32()?;
    let source = match r.u32()? {
        0 => PresentSource::Surface { id: SurfaceId(r.u32()?), plane: r.u32()? },
        1 => {
            let task = r.task()?;
            let texture = Handle::new(r.u64()?).ok_or_else(|| r.err("texture handle 0"))?;
            PresentSource::Texture { task, texture }
        }
        t => return Err(r.err(format!("source tag {t}"))),
    };
    let source_rect = r.rect()?;
    let target_rect = r.rect()?;
    let transform = r.u32()?;
    let time = r.opt_u64()?.map(MonotonicNanos);
    let minimum_duration = r.opt_u64()?;
    Ok(SwapRequest {
        frame: PresentedFrame { display, source, source_rect, target_rect, transform, time },
        minimum_duration,
    })
}

/// Encode a recording.
pub fn encode(steps: &[Step]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for s in steps {
        let mut w = Writer::default();
        let k = match s {
            Step::Map { task, addr, len } => {
                w.u64(task.0);
                w.u64(*addr);
                w.u64(*len);
                kind::MAP
            }
            Step::Write { task, addr, bytes } => {
                w.u64(task.0);
                w.u64(*addr);
                w.bytes(bytes);
                kind::WRITE
            }
            Step::CreateSurface { desc } => {
                write_surface_desc(&mut w, desc);
                kind::CREATE_SURFACE
            }
            Step::WriteSurface { id, offset, bytes } => {
                w.u32(id.0);
                w.u64(*offset);
                w.bytes(bytes);
                kind::WRITE_SURFACE
            }
            Step::Control { task, request, reply } => {
                w.u64(task.0);
                w.bytes(request);
                match reply {
                    Some(r) => {
                        w.u8(1);
                        w.bytes(r);
                    }
                    None => w.u8(0),
                }
                kind::CONTROL
            }
            Step::Wait => kind::WAIT,
            Step::Attach { display, mode } => {
                w.u32(*display);
                write_mode(&mut w, mode);
                kind::ATTACH
            }
            Step::Present { request } => {
                write_request(&mut w, request);
                kind::PRESENT
            }
            Step::SetTime { now } => {
                w.u64(now.0);
                kind::SET_TIME
            }
            Step::ExitTask { task } => {
                w.u64(task.0);
                kind::EXIT_TASK
            }
        };
        out.extend_from_slice(&k.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(w.0.len() as u64).to_le_bytes());
        out.extend_from_slice(&w.0);
        out.resize(out.len().next_multiple_of(8), 0);
    }
    out
}

/// Decode a recording file.
pub fn decode(file: &[u8]) -> Result<Vec<Step>, FormatError> {
    let mut r = Reader { b: file, at: 0, base: 0 };
    if r.take(8)? != MAGIC {
        return Err(FormatError { offset: 0, what: "not a recording file".into() });
    }
    let v = r.u32()?;
    if v != VERSION {
        return Err(r.err(format!("format version {v}; this reader reads {VERSION}")));
    }
    if r.u32()? != 0 {
        return Err(r.err("reserved header field is not 0"));
    }
    let mut steps = Vec::new();
    while r.at < file.len() {
        let k = r.u32()?;
        if r.u32()? != 0 {
            return Err(r.err("reserved record field is not 0"));
        }
        let len = usize::try_from(r.u64()?).map_err(|_| r.err("record length too large"))?;
        let start = r.at;
        let payload = r.take(len)?;
        let mut p = Reader { b: payload, at: 0, base: start };
        let step = match k {
            kind::MAP => Step::Map { task: p.task()?, addr: p.u64()?, len: p.u64()? },
            kind::WRITE => Step::Write { task: p.task()?, addr: p.u64()?, bytes: p.bytes()? },
            kind::CREATE_SURFACE => Step::CreateSurface { desc: read_surface_desc(&mut p)? },
            kind::WRITE_SURFACE => Step::WriteSurface { id: SurfaceId(p.u32()?), offset: p.u64()?, bytes: p.bytes()? },
            kind::CONTROL => {
                let task = p.task()?;
                let request = p.bytes()?;
                let reply = if p.flag()? { Some(p.bytes()?) } else { None };
                Step::Control { task, request, reply }
            }
            kind::WAIT => Step::Wait,
            kind::ATTACH => Step::Attach { display: p.u32()?, mode: read_mode(&mut p)? },
            kind::PRESENT => Step::Present { request: read_request(&mut p)? },
            kind::SET_TIME => Step::SetTime { now: MonotonicNanos(p.u64()?) },
            kind::EXIT_TASK => Step::ExitTask { task: p.task()? },
            _ => return Err(FormatError { offset: start - 16, what: format!("record kind {k}") }),
        };
        if p.at != len {
            return Err(p.err(format!("{} bytes left in the record", len - p.at)));
        }
        steps.push(step);
        let pad = len.next_multiple_of(8) - len;
        if r.take(pad)?.iter().any(|&b| b != 0) {
            return Err(r.err("padding is not zero"));
        }
    }
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steps() -> Vec<Step> {
        let desc = SurfaceDesc {
            id: SurfaceId(9),
            width: 4,
            height: 2,
            bytes_per_row: 16,
            bytes_per_element: 4,
            element_width: 1,
            element_height: 1,
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: 32,
            planes: vec![SurfacePlane {
                width: 4,
                height: 2,
                bytes_per_row: 16,
                bytes_per_element: 4,
                element_width: 1,
                element_height: 1,
                offset: 0,
                size: 32,
            }],
            seed: 3,
        };
        let rect = Rect { x: 0, y: 0, width: 4, height: 2 };
        vec![
            Step::Map { task: TaskId(1), addr: 0x1000, len: 0x4000 },
            Step::Write { task: TaskId(1), addr: 0x1001, bytes: vec![1, 2, 3] },
            Step::CreateSurface { desc },
            Step::WriteSurface { id: SurfaceId(9), offset: 4, bytes: vec![7; 5] },
            Step::Control { task: TaskId(1), request: vec![1, 0, 0, 0, 8, 0, 0, 0], reply: None },
            Step::Control { task: TaskId(1), request: vec![2; 9], reply: Some(vec![3; 11]) },
            Step::Wait,
            Step::Attach {
                display: 0,
                mode: DisplayMode {
                    width: 4,
                    height: 2,
                    color_space: DisplayColorSpace::DisplayP3,
                    identity_transform: 0,
                },
            },
            Step::Attach {
                display: 1,
                mode: DisplayMode {
                    width: 4,
                    height: 2,
                    color_space: DisplayColorSpace::Other("BT.2020".into()),
                    identity_transform: 5,
                },
            },
            Step::Present {
                request: SwapRequest {
                    frame: PresentedFrame {
                        display: 0,
                        source: PresentSource::Surface { id: SurfaceId(9), plane: 0 },
                        source_rect: rect,
                        target_rect: rect,
                        transform: 0,
                        time: Some(MonotonicNanos(500)),
                    },
                    minimum_duration: None,
                },
            },
            Step::Present {
                request: SwapRequest {
                    frame: PresentedFrame {
                        display: 1,
                        source: PresentSource::Texture { task: TaskId(2), texture: Handle::new(77).unwrap() },
                        source_rect: Rect { x: -1, y: 2, width: 3, height: 4 },
                        target_rect: rect,
                        transform: 1,
                        time: None,
                    },
                    minimum_duration: Some(16_666_667),
                },
            },
            Step::SetTime { now: MonotonicNanos(1_000) },
            Step::ExitTask { task: TaskId(1) },
        ]
    }

    #[test]
    fn recordings_round_trip() {
        let s = steps();
        let file = encode(&s);
        assert_eq!(file.len() % 8, 0);
        assert_eq!(decode(&file).unwrap(), s);
    }

    #[test]
    fn malformed_files_are_refused() {
        let file = encode(&steps());
        assert!(decode(&file[..7]).is_err(), "short magic");
        let mut bad = file.clone();
        bad[0] = b'X';
        assert!(decode(&bad).is_err(), "magic");
        let mut bad = file.clone();
        bad[8] = 2;
        assert!(decode(&bad).unwrap_err().what.contains("version"));
        // Every truncation fails, never yields fewer steps silently, except at a record boundary.
        let boundaries: Vec<usize> = {
            let mut v = vec![16];
            let mut at = 16;
            while at < file.len() {
                let len = u64::from_le_bytes(file[at + 8..at + 16].try_into().unwrap()) as usize;
                at += 16 + len.next_multiple_of(8);
                v.push(at);
            }
            v
        };
        for n in 0..file.len() {
            let r = decode(&file[..n]);
            assert_eq!(r.is_ok(), boundaries.contains(&n), "truncated to {n}: {r:?}");
        }
        // An unknown kind.
        let mut bad = file.clone();
        bad[16] = 0xee;
        assert!(decode(&bad).unwrap_err().what.contains("kind"));
        // A nonzero padding byte (the Write record's payload is 27 bytes, padded to 32).
        let mut bad = file.clone();
        let at = boundaries[2] - 1;
        bad[at] = 1;
        assert!(decode(&bad).unwrap_err().what.contains("padding"), "{:?}", decode(&bad));
        // A record with bytes left over: grow the Wait record's payload to 8 zero bytes.
        let mut s = encode(&[Step::Wait]);
        s[24..32].copy_from_slice(&8u64.to_le_bytes());
        s.extend_from_slice(&[0; 8]);
        assert!(decode(&s).unwrap_err().what.contains("left"));
    }
}
