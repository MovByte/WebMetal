//! Shared pieces of the reference embedder's tests on a native WebGPU implementation: a host on a fresh device, the
//! wire records the guest sends, surfaces and frames, and the Metal reference outputs.

#![allow(dead_code)]

use std::cell::Cell;
use std::collections::BTreeSet;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use webmetal_embed_ref::host::native;
use webmetal_embed_ref::{Clock, RefEmbedder, RefHost};
use webmetal_host::embed::{PresentSource, PresentedFrame, Rect, SurfaceDesc, TaskId};
use webmetal_host::exec::Config;
use webmetal_host::exec::schema::{Role, command_handles};
use webmetal_host::gpu::native::{Native, NativeGpu};
use webmetal_host::gpu::scope::{Granularity, Policy};
use webmetal_host::present::{DisplayColorSpace, DisplayMode, SwapRequest};
use webmetal_wire::*;

pub const TASK: TaskId = TaskId(1);
/// Where the tests map the task's memory.
pub const BASE: u64 = 0x10_0000;
pub const WAIT: Duration = Duration::from_secs(120);

/// A host on its own device, with a manual clock at 0.
pub struct H {
    pub gpu: NativeGpu,
    pub host: RefHost<Native>,
    next: Cell<u64>,
    alloc: Cell<u64>,
}

impl H {
    pub fn new() -> H {
        let (gpu, device) = native::device("present test device").expect("a device");
        let embedder = Rc::new(RefEmbedder::new(Clock::Manual(Cell::new(0))));
        let config = Config { scopes: Policy { granularity: Granularity::PerCall, strict: true } };
        let wgsl = webmetal_embed_ref::host::wgsl_language_features::<Native>(&gpu).expect("wgslLanguageFeatures");
        let host =
            gpu.block_on(RefHost::new(device, wgsl, embedder, config), WAIT).expect("setup stalled").expect("setup");
        host.embedder.map(TASK, BASE, 64 << 20).unwrap();
        H { gpu, host, next: Cell::new(0x100), alloc: Cell::new(BASE + 0x1000) }
    }

    pub fn block_on<F: Future>(&self, f: F) -> F::Output {
        self.gpu.block_on(f, WAIT).expect("GPU work stalled")
    }

    pub fn handle(&self) -> Handle {
        self.next.set(self.next.get() + 1);
        Handle::new(self.next.get()).unwrap()
    }

    pub fn alloc(&self, len: u64) -> u64 {
        let a = self.alloc.get();
        self.alloc.set((a + len).next_multiple_of(0x4000) + 0x4000);
        a
    }

    pub fn request(&self, r: &Request) -> Reply {
        self.block_on(self.host.control(TASK, r)).expect("a reply")
    }

    pub fn ok(&self, r: &Request) {
        let reply = self.request(r);
        let f = webmetal_host::exec::schema::reply_failure(&reply).cloned();
        assert!(f.is_none_or(|f| f.status == status::OK), "{r:?}: {reply:?}");
    }

    pub fn open(&self) {
        self.ok(&open());
    }

    pub fn queue(&self) -> Handle {
        let q = self.handle();
        self.ok(&Request::CreateCommandQueue(CreateCommandQueue { queue: q }));
        q
    }
}

pub fn open() -> Request {
    Request::Open(Open { schema_hash: Hash { bytes: SCHEMA_HASH }, pid: 1 })
}

/// Poll a future once, without blocking.
pub fn poll_once<F: Future + ?Sized>(f: Pin<&mut F>) -> Poll<F::Output> {
    f.poll(&mut Context::from_waker(Waker::noop()))
}

/// A one-plane 32-bit surface: `width` × `height` pixels in `format`, rows of `bytes_per_row` bytes.
pub fn surface(id: u32, format: u32, width: u64, height: u64, bytes_per_row: u64) -> SurfaceDesc {
    SurfaceDesc {
        id: SurfaceId(id),
        width,
        height,
        bytes_per_row,
        bytes_per_element: 4,
        element_width: 1,
        element_height: 1,
        pixel_format: format,
        alloc_size: bytes_per_row * height,
        planes: vec![],
        seed: 1,
    }
}

pub fn srgb(width: u32, height: u32) -> DisplayMode {
    DisplayMode { width, height, color_space: DisplayColorSpace::Srgb, identity_transform: 0 }
}

pub fn rect(x: i64, y: i64, width: u64, height: u64) -> Rect {
    Rect { x, y, width, height }
}

/// A frame of the whole of surface `id` (from `(x, y)`) on a `width` × `height` display.
pub fn frame(display: u32, id: u32, x: i64, y: i64, width: u64, height: u64) -> SwapRequest {
    SwapRequest {
        frame: PresentedFrame {
            display,
            source: PresentSource::Surface { id: SurfaceId(id), plane: 0 },
            source_rect: rect(x, y, width, height),
            target_rect: rect(0, 0, width, height),
            transform: 0,
            time: None,
        },
        minimum_duration: None,
    }
}

/// A deterministic byte pattern.
pub fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut state = if seed == 0 { 0x9e37_79b9 } else { seed };
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect()
}

/// Rows `x..x + width` of `height` rows from `y` of a surface's memory, tight.
pub fn rows(bytes: &[u8], bytes_per_row: usize, x: usize, y: usize, width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for r in y..y + height {
        out.extend_from_slice(&bytes[r * bytes_per_row + x * 4..r * bytes_per_row + (x + width) * 4]);
    }
    out
}

// Wire records ---------------------------------------------------------------------------------------------------

/// `MTLPixelFormatBGRA8Unorm` (`MTLPixelFormat.h`).
pub const BGRA8_UNORM: u64 = 80;

/// A 2D texture descriptor with Metal's defaults and `usage`, `storage`.
pub fn texture_descriptor(format: u64, w: u64, h: u64, storage: u64, usage: u64) -> TextureDescriptor {
    TextureDescriptor {
        allow_gpu_optimized_contents: true,
        array_length: 1,
        color_space_conversion_matrix: 0,
        compression_footprint: 0,
        compression_mode: 0,
        compression_type: 0,
        cpu_cache_mode: 0,
        depth: 1,
        force_resource_index: false,
        framebuffer_only: false,
        hazard_tracking_mode: 0,
        height: h,
        is_drawable: false,
        lossy_compression_mode: 0,
        mipmap_level_count: 1,
        pixel_format: format,
        protection_options: 0,
        resource_index: 0,
        resource_options: storage << 4,
        rotation: 0,
        sample_count: 1,
        sparse_surface_default_value: 0,
        storage_mode: storage,
        swizzle: TextureSwizzleChannels { red: 2, green: 3, blue: 4, alpha: 5 },
        swizzle_key: 0,
        texture_type: 2,
        texture_usage: usage,
        usage,
        width: w,
        write_swizzle_enabled: false,
    }
}

pub fn color_attachment(texture: Option<Handle>) -> RenderPassColorAttachment {
    RenderPassColorAttachment {
        clear_color: ClearColor { red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0 },
        depth_plane: 0,
        level: 0,
        load_action: 0,
        resolve_depth_plane: 0,
        resolve_level: 0,
        resolve_slice: 0,
        resolve_texture: None,
        slice: 0,
        store_action: 0,
        store_action_options: 0,
        texture,
        y_invert: false,
    }
}

/// A render pass whose color attachment 0 is `texture`, cleared to `clear` and stored.
pub fn pass_clear(texture: Handle, clear: ClearColor) -> RenderPassDescriptor {
    let mut color_attachments: Vec<RenderPassColorAttachment> = (0..8).map(|_| color_attachment(None)).collect();
    color_attachments[0].texture = Some(texture);
    color_attachments[0].load_action = 2;
    color_attachments[0].store_action = 1;
    color_attachments[0].clear_color = clear;
    RenderPassDescriptor {
        binary_archives: vec![],
        color_attachments,
        default_color_sample_count: 0,
        default_raster_sample_count: 0,
        depth_attachment: None,
        dither_enabled: false,
        fine_grained_background_visibility_enabled: false,
        imageblock_sample_length: 0,
        open_gl_mode_enabled: false,
        point_coord_y_flip_enabled: false,
        rasterization_rate_map: None,
        render_target_array_length: 0,
        render_target_height: 0,
        render_target_width: 0,
        sample_buffer_attachments: vec![],
        skip_empty_tiles_on_clear_enabled: false,
        stencil_attachment: None,
        threadgroup_memory_length: 0,
        tile_height: 0,
        tile_width: 0,
        visibility_result_buffer: None,
        sample_positions: vec![],
    }
}

pub fn end() -> Command {
    Command::EncoderEndEncoding(EncoderEndEncoding {})
}

/// A command stream, and the existing objects it names (the Commit's `resources`).
pub fn stream(records: &[Command]) -> (Vec<u8>, Vec<Handle>) {
    let mut bytes = Vec::new();
    let mut created = BTreeSet::new();
    let mut named = Vec::new();
    for r in records {
        r.encode(&mut bytes).unwrap();
        command_handles(r, &mut |u| {
            if u.role == Role::Result {
                created.insert(u.handle);
            } else if !created.contains(&u.handle) && !named.contains(&u.handle) {
                named.push(u.handle);
            }
        });
    }
    (bytes, named)
}

pub fn commit(command_buffer: Handle, queue: Handle, stream_addr: u64, len: usize, resources: Vec<Handle>) -> Request {
    Request::Commit(Commit {
        command_buffer,
        queue,
        label: None,
        retained_references: true,
        error_options: 0,
        stream: GuestAddr(stream_addr),
        stream_length: len as u64,
        resources,
    })
}

pub fn encode(r: &Request) -> Vec<u8> {
    let mut b = Vec::new();
    r.encode(&mut b).unwrap();
    b
}

// Metal reference outputs ------------------------------------------------------------------------------------------

/// The reference device whose outputs the tests compare with.
pub const GOLDEN_DEVICE: &str = "apple-m4-macos-27.0.1-26a434";

/// A case of the Metal reference outputs (release variant): one record per line.
pub struct Golden {
    text: String,
}

impl Golden {
    pub fn load(case: &str) -> Golden {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/reference/goldens")
            .join(GOLDEN_DEVICE)
            .join("release")
            .join(format!("{case}.json"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Golden { text }
    }

    fn record(&self, name: &str) -> &str {
        let key = format!("\"name\":\"{name}\"");
        self.text.lines().find(|l| l.contains(&key)).unwrap_or_else(|| panic!("no record {name}"))
    }

    fn field<'a>(line: &'a str, key: &str) -> &'a str {
        let k = format!("\"{key}\":");
        let at = line.find(&k).unwrap_or_else(|| panic!("no {key} in {line}")) + k.len();
        let rest = &line[at..];
        if let Some(s) = rest.strip_prefix('"') {
            &s[..s.find('"').unwrap()]
        } else {
            &rest[..rest.find([',', '}']).unwrap_or(rest.len())]
        }
    }

    /// The bytes of a `data` record.
    pub fn bytes(&self, name: &str) -> Vec<u8> {
        let hex = Self::field(self.record(name), "hex");
        (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
    }

    /// The value of a record.
    pub fn value(&self, name: &str) -> String {
        Self::field(self.record(name), "value").to_string()
    }

    /// The comparison rule of a record.
    pub fn rule(&self, name: &str) -> String {
        Self::field(self.record(name), "compare").to_string()
    }

    /// Whether an `error` record is nil.
    pub fn error_nil(&self, name: &str) -> bool {
        Self::field(self.record(name), "nil") == "1"
    }
}
