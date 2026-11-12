//! The presentation sink on a native WebGPU implementation (`WEBGPU_NATIVE`, see `host/build.rs`): frames from CPU- and
//! GPU-written IOSurfaces read back from the display's offscreen target equal the surface's bytes exactly; frames wait
//! for their source's work and their time and keep their order; every frame the sink cannot show exactly is refused,
//! with its reason. Run with `--release`.

#![cfg(feature = "native")]

mod common;

use std::task::Poll;

use common::*;
use webmetal_host::embed::{MonotonicNanos, PresentSource, TaskId};
use webmetal_host::gpu::idl::{
    CanvasAlphaMode, CanvasConfiguration, CanvasToneMappingMode, TextureFormat, TextureUsage,
};
use webmetal_host::gpu::native::Native;
use webmetal_host::gpu::{Device as _, ExceptionKind, Texture as _};
use webmetal_host::present::format::{code, fourcc};
use webmetal_host::present::{DisplayColorSpace, DisplayMode, Offscreen, Target};
use webmetal_wire::*;

/// A display of J120's panel size. J120 (iPad7,1) is the 12.9-inch iPad Pro, whose DeviceTree product node reports
/// `artwork-device-subtype` 2732, the panel's long side; the mode itself is the display model's (K3).
const PANEL: (u32, u32) = (2732, 2048);

fn drawing_buffer(h: &H, display: u32) -> Vec<u8> {
    h.block_on(h.host.target(display).unwrap().drawing_buffer()).unwrap().expect("configured")
}

#[test]
fn a_cpu_written_bgra_frame_is_shown_exactly() {
    let h = H::new();
    let (w, ht) = PANEL;
    let bpr = u64::from(w) * 4;
    h.host.embedder.create_surface(surface(1, code::BGRA_32, w.into(), ht.into(), bpr)).unwrap();
    let pixels = pattern((bpr * u64::from(ht)) as usize, 7);
    h.host.embedder.surface_cpu_write(SurfaceId(1), 0, &pixels).unwrap();
    h.host.attach_offscreen(0, srgb(w, ht)).unwrap();
    let p = h.block_on(h.host.presenter.present(frame(0, 1, 0, 0, w.into(), ht.into()))).unwrap();
    assert_eq!((p.display, p.sequence, p.plan.format), (0, 0, TextureFormat::Bgra8unorm));
    let shown = drawing_buffer(&h, 0);
    assert!(shown == pixels, "the drawing buffer differs from the surface");
    // What the browser shows: the same codes with alpha cleared to 1.0, tagged with the display's color space.
    let snap = h.block_on(h.host.target(0).unwrap().snapshot()).unwrap();
    assert_eq!(snap.format, Some(TextureFormat::Bgra8unorm));
    assert_eq!((snap.color_space.as_str(), snap.opaque), ("srgb", true));
    assert_eq!(snap.tone_mapping, Some(CanvasToneMappingMode::Standard));
    let mut opaque = pixels.clone();
    for t in opaque.as_chunks_mut::<4>().0 {
        t[3] = 0xff;
    }
    assert!(snap.bytes == opaque);
    let told = h.host.embedder.shown();
    assert_eq!(told.len(), 1);
    assert_eq!(told[0].at, p.presented_at);
}

/// The surface of a 64 x 32 drawable whose rows are padded, and a BGRA8Unorm texture over it.
fn drawable(h: &H, id: u32) -> (Handle, Handle) {
    let (w, ht, bpr) = (64u64, 32u64, 320u64);
    h.host.embedder.create_surface(surface(id, code::BGRA_32, w, ht, bpr)).unwrap();
    let q = h.queue();
    let t = h.handle();
    h.ok(&Request::CreateTextureWithIOSurface(CreateTextureWithIOSurface {
        texture: t,
        descriptor: texture_descriptor(BGRA8_UNORM, w, ht, 0, 0x17),
        iosurface: SurfaceId(id),
        plane: 0,
    }));
    (q, t)
}

fn submit(h: &H, q: Handle, records: &[Command]) -> Handle {
    let (bytes, resources) = stream(records);
    let addr = h.alloc(bytes.len() as u64);
    h.host.embedder.cpu_write(TASK, addr, &bytes).unwrap();
    let cb = h.handle();
    h.ok(&commit(cb, q, addr, bytes.len(), resources));
    cb
}

#[test]
fn a_frame_shows_the_gpu_writes_committed_before_it() {
    let h = H::new();
    h.open();
    let (q, t) = drawable(&h, 1);
    h.host.attach_offscreen(0, srgb(64, 32)).unwrap();
    // A blit writes a pattern into the drawable; the frame is requested right after the commit, before the GPU ran.
    let pixels = pattern(64 * 32 * 4, 11);
    let addr = h.alloc(pixels.len() as u64);
    h.host.embedder.cpu_write(TASK, addr, &pixels).unwrap();
    let b = h.handle();
    h.ok(&Request::CreateBuffer(CreateBuffer {
        buffer: b,
        length: pixels.len() as u64,
        options: 0,
        contents: GuestAddr(addr),
    }));
    let e = h.handle();
    submit(
        &h,
        q,
        &[
            Command::BeginBlit(BeginBlit { encoder: e }),
            Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
                source_buffer: Some(b),
                source_offset: 0,
                source_bytes_per_row: 256,
                source_bytes_per_image: 256 * 32,
                source_size: Size { width: 64, height: 32, depth: 1 },
                destination_texture: Some(t),
                destination_slice: 0,
                destination_level: 0,
                destination_origin: Origin { x: 0, y: 0, z: 0 },
            }),
            end(),
        ],
    );
    h.block_on(h.host.presenter.present(frame(0, 1, 0, 0, 64, 32))).unwrap();
    let surface = h.host.embedder.surface_bytes(SurfaceId(1)).unwrap();
    assert_eq!(rows(&surface, 320, 0, 0, 64, 32), pixels, "the surface holds the GPU's writes");
    assert!(drawing_buffer(&h, 0) == pixels);

    // A render pass clears the drawable; the next frame shows the cleared codes.
    let e = h.handle();
    submit(
        &h,
        q,
        &[
            Command::BeginRenderPass(BeginRenderPass {
                encoder: e,
                render_pass_descriptor: pass_clear(t, ClearColor { red: 1.0, green: 0.0, blue: 1.0, alpha: 1.0 }),
            }),
            end(),
        ],
    );
    h.block_on(h.host.presenter.present(frame(0, 1, 0, 0, 64, 32))).unwrap();
    assert!(drawing_buffer(&h, 0).as_chunks::<4>().0.iter().all(|t| *t == [0xff, 0x00, 0xff, 0xff]));
    assert_eq!(h.host.embedder.shown().len(), 2);
}

#[test]
fn argb_frames_show_their_codes_in_bgra_order() {
    let h = H::new();
    h.host.embedder.create_surface(surface(4, code::ARGB_32, 33, 17, 33 * 4 + 12)).unwrap();
    let pixels = pattern((33 * 4 + 12) * 17, 5);
    h.host.embedder.surface_cpu_write(SurfaceId(4), 0, &pixels).unwrap();
    h.host.attach_offscreen(2, srgb(33, 17)).unwrap();
    let p = h.block_on(h.host.presenter.present(frame(2, 4, 0, 0, 33, 17))).unwrap();
    assert_eq!(p.plan.format, TextureFormat::Bgra8unorm);
    let shown = drawing_buffer(&h, 2);
    let argb = rows(&pixels, 33 * 4 + 12, 0, 0, 33, 17);
    for (i, (s, a)) in shown.as_chunks::<4>().0.iter().zip(argb.as_chunks::<4>().0).enumerate() {
        // Alpha, red, green, blue in the surface; blue, green, red, alpha in the canvas texel.
        assert_eq!([s[3], s[2], s[1], s[0]], [a[0], a[1], a[2], a[3]], "pixel {i}");
    }
}

#[test]
fn a_frame_shows_the_part_of_the_surface_its_source_names() {
    let h = H::new();
    h.host.embedder.create_surface(surface(3, code::BGRA_32, 40, 30, 160)).unwrap();
    let pixels = pattern(160 * 30, 3);
    h.host.embedder.surface_cpu_write(SurfaceId(3), 0, &pixels).unwrap();
    h.host.attach_offscreen(1, srgb(16, 8)).unwrap();
    h.block_on(h.host.presenter.present(frame(1, 3, 5, 7, 16, 8))).unwrap();
    assert!(drawing_buffer(&h, 1) == rows(&pixels, 160, 5, 7, 16, 8));
    // The last row and column of the plane.
    h.block_on(h.host.presenter.present(frame(1, 3, 24, 22, 16, 8))).unwrap();
    assert!(drawing_buffer(&h, 1) == rows(&pixels, 160, 24, 22, 16, 8));
}

#[test]
fn display_color_spaces_tag_the_canvas() {
    let h = H::new();
    h.host.embedder.create_surface(surface(1, code::BGRA_32, 8, 8, 32)).unwrap();
    for (i, (c, name)) in [
        (DisplayColorSpace::DisplayP3, "display-p3"),
        (DisplayColorSpace::Srgb, "srgb"),
        (DisplayColorSpace::SrgbLinear, "srgb-linear"),
        (DisplayColorSpace::DisplayP3Linear, "display-p3-linear"),
    ]
    .into_iter()
    .enumerate()
    {
        let d = i as u32;
        h.host.attach_offscreen(d, DisplayMode { width: 8, height: 8, color_space: c, identity_transform: 0 }).unwrap();
        let p = h.block_on(h.host.presenter.present(frame(d, 1, 0, 0, 8, 8))).unwrap();
        assert_eq!(p.plan.color_space, name);
        let config = h.host.target(d).unwrap().configuration().unwrap();
        assert_eq!(config.color_space, name);
        assert_eq!(config.alpha_mode, CanvasAlphaMode::Opaque);
        assert_eq!(config.usage, TextureUsage::COPY_DST);
    }
}

#[test]
fn frames_wait_for_their_time_and_keep_their_order() {
    let h = H::new();
    h.host.embedder.create_surface(surface(1, code::BGRA_32, 4, 4, 16)).unwrap();
    h.host.embedder.create_surface(surface(2, code::BGRA_32, 4, 4, 16)).unwrap();
    h.host.embedder.surface_cpu_write(SurfaceId(1), 0, &[0x11; 64]).unwrap();
    h.host.embedder.surface_cpu_write(SurfaceId(2), 0, &[0x22; 64]).unwrap();
    h.host.attach_offscreen(0, srgb(4, 4)).unwrap();
    let target = h.host.target(0).unwrap();

    // presentAtTime: frame 0 may not appear before 1000; frame 1, untimed, waits behind it.
    let mut timed = frame(0, 1, 0, 0, 4, 4);
    timed.frame.time = Some(MonotonicNanos(1000));
    let mut f0 = h.host.presenter.present(timed);
    let mut f1 = h.host.presenter.present(frame(0, 2, 0, 0, 4, 4));
    for _ in 0..20 {
        assert!(poll_once(f0.as_mut()).is_pending());
        assert!(poll_once(f1.as_mut()).is_pending());
        h.gpu.process_events();
    }
    assert_eq!(target.presented(), 0);
    assert!(h.host.embedder.shown().is_empty());
    assert_eq!(h.host.presenter.next_deadline(), Some(MonotonicNanos(1000)));
    h.host.embedder.set_time(MonotonicNanos(999));
    h.host.presenter.wake_due();
    assert!(poll_once(f0.as_mut()).is_pending(), "one nanosecond early");
    h.host.embedder.set_time(MonotonicNanos(1000));
    h.host.presenter.wake_due();
    let p0 = h.block_on(f0).unwrap();
    assert_eq!((p0.sequence, p0.presented_at, p0.not_before), (0, MonotonicNanos(1000), Some(MonotonicNanos(1000))));
    let p1 = h.block_on(f1).unwrap();
    assert_eq!((p1.sequence, p1.not_before), (1, None));
    assert!(drawing_buffer(&h, 0) == vec![0x22; 64]);

    // presentAfterMinimumDuration: no earlier than the previous frame's presentation plus the duration.
    let mut after = frame(0, 1, 0, 0, 4, 4);
    after.minimum_duration = Some(500);
    let mut f2 = h.host.presenter.present(after);
    h.host.embedder.set_time(MonotonicNanos(1499));
    for _ in 0..5 {
        assert!(poll_once(f2.as_mut()).is_pending());
    }
    h.host.embedder.set_time(MonotonicNanos(1500));
    let p2 = h.block_on(f2).unwrap();
    assert_eq!((p2.sequence, p2.not_before, p2.presented_at), (2, Some(MonotonicNanos(1500)), MonotonicNanos(1500)));
    assert!(drawing_buffer(&h, 0) == vec![0x11; 64]);
    // Both bounds: the later one wins.
    let mut both = frame(0, 2, 0, 0, 4, 4);
    both.minimum_duration = Some(100);
    both.frame.time = Some(MonotonicNanos(1550));
    let mut f3 = h.host.presenter.present(both);
    h.host.embedder.set_time(MonotonicNanos(1599));
    assert!(poll_once(f3.as_mut()).is_pending());
    h.host.embedder.set_time(MonotonicNanos(1600));
    assert_eq!(h.block_on(f3).unwrap().not_before, Some(MonotonicNanos(1600)));
    let order: Vec<u32> = h
        .host
        .embedder
        .shown()
        .iter()
        .map(|s| match s.frame.source {
            PresentSource::Surface { id, .. } => id.0,
            PresentSource::Texture { .. } => unreachable!(),
        })
        .collect();
    assert_eq!(order, vec![1, 2, 1, 2]);
}

#[test]
fn a_frame_dropped_before_its_turn_does_not_hold_the_display() {
    let h = H::new();
    h.host.embedder.create_surface(surface(1, code::BGRA_32, 4, 4, 16)).unwrap();
    h.host.attach_offscreen(0, srgb(4, 4)).unwrap();
    let mut timed = frame(0, 1, 0, 0, 4, 4);
    timed.frame.time = Some(MonotonicNanos(10));
    let mut f0 = h.host.presenter.present(timed);
    assert!(poll_once(f0.as_mut()).is_pending());
    let f1 = h.host.presenter.present(frame(0, 1, 0, 0, 4, 4));
    drop(f1);
    drop(f0);
    let p = h.block_on(h.host.presenter.present(frame(0, 1, 0, 0, 4, 4))).unwrap();
    assert_eq!(p.sequence, 2);
}

/// The refusal id and status a request ends with.
fn refused(h: &H, r: webmetal_host::present::SwapRequest) -> (Option<&'static str>, u32) {
    let f = h.block_on(h.host.presenter.present(r)).unwrap_err();
    (f.refusal, f.failure.status)
}

#[test]
fn every_frame_that_cannot_be_shown_exactly_is_refused() {
    let h = H::new();
    h.host.embedder.create_surface(surface(1, code::BGRA_32, 16, 8, 64)).unwrap();
    h.host.attach_offscreen(0, srgb(16, 8)).unwrap();

    // Pixel formats.
    for (i, (c, id)) in [
        (code::ARGB_2101010_LE_PACKED, "present-10-bit-unorm"),
        (code::RGB_30, "present-10-bit-unorm"),
        (code::RGB_30_R210, "present-10-bit-unorm"),
        (code::RGB_30_LE_PACKED_WIDE_GAMUT, "present-extended-range"),
        (code::ARGB_40_LE_WIDE_GAMUT, "present-extended-range"),
        (code::ARGB_40_LE_WIDE_GAMUT_PREMULTIPLIED, "present-extended-range"),
        (code::RGBA_64_HALF, "present-half-float"),
        (code::RGBA_32, "present-undescribed-layout"),
        (code::ABGR_32, "present-undescribed-layout"),
        (fourcc(b"420v"), "present-pixel-format"),
        (fourcc(b"&BGA"), "present-pixel-format"),
    ]
    .into_iter()
    .enumerate()
    {
        let sid = 100 + i as u32;
        let bytes = if c == code::ARGB_40_LE_WIDE_GAMUT || c == code::RGBA_64_HALF { 8 } else { 4 };
        let mut s = surface(sid, c, 16, 8, 16 * bytes);
        s.bytes_per_element = bytes;
        h.host.embedder.create_surface(s).unwrap();
        assert_eq!(refused(&h, frame(0, sid, 0, 0, 16, 8)), (Some(id), status::UNIMPLEMENTED), "{c:#x}");
    }

    // Transforms, scaling, partial frames, texture sources.
    let mut r = frame(0, 1, 0, 0, 16, 8);
    r.frame.transform = 1;
    assert_eq!(refused(&h, r), (Some("present-transform"), status::UNIMPLEMENTED));
    let mut r = frame(0, 1, 0, 0, 16, 8);
    r.frame.source_rect = rect(0, 0, 8, 4);
    assert_eq!(refused(&h, r), (Some("present-scaling"), status::UNIMPLEMENTED));
    let mut r = frame(0, 1, 0, 0, 8, 8);
    r.frame.target_rect = rect(0, 0, 8, 8);
    assert_eq!(refused(&h, r), (Some("present-partial-frame"), status::UNIMPLEMENTED));
    let mut r = frame(0, 1, 0, 0, 16, 8);
    r.frame.target_rect = rect(1, 0, 16, 8);
    assert_eq!(refused(&h, r), (Some("present-partial-frame"), status::UNIMPLEMENTED));

    // A display color space no PredefinedColorSpace value denotes.
    let other = DisplayMode {
        width: 16,
        height: 8,
        color_space: DisplayColorSpace::Other("ITU-R BT.2100 PQ".into()),
        identity_transform: 0,
    };
    h.host.attach_offscreen(1, other).unwrap();
    assert_eq!(refused(&h, frame(1, 1, 0, 0, 16, 8)), (Some("present-color-space"), status::UNIMPLEMENTED));

    // A display larger than the device's textures.
    let max = h.host.device.limits().max_texture_dimension_2d.unwrap();
    let f = h.host.attach_offscreen(2, srgb(max + 1, 8)).unwrap_err();
    assert_eq!((f.refusal, f.failure.status), (Some("present-display-size"), status::HOST_UNSUPPORTED));

    // Malformed frames.
    assert_eq!(refused(&h, frame(9, 1, 0, 0, 16, 8)), (None, status::INVALID), "unknown display");
    assert_eq!(refused(&h, frame(0, 77, 0, 0, 16, 8)), (None, status::INVALID), "unknown surface");
    assert_eq!(refused(&h, frame(0, 1, 1, 0, 16, 8)), (None, status::INVALID), "outside the plane");
    assert_eq!(refused(&h, frame(0, 1, -1, 0, 16, 8)), (None, status::INVALID), "negative origin");
    let mut r = frame(0, 1, 0, 0, 16, 8);
    r.frame.source = PresentSource::Texture { task: TaskId(1), texture: Handle::new(5).unwrap() };
    assert_eq!(refused(&h, r), (None, status::INVALID), "a handle that names no object");
    let mut r = frame(0, 1, 0, 0, 16, 8);
    r.frame.source = PresentSource::Surface { id: SurfaceId(1), plane: 1 };
    assert_eq!(refused(&h, r), (None, status::INVALID), "plane of a surface without planes");
    let mut odd = surface(50, code::BGRA_32, 16, 8, 64);
    odd.bytes_per_element = 2;
    h.host.embedder.create_surface(odd).unwrap();
    assert_eq!(refused(&h, frame(0, 50, 0, 0, 16, 8)), (None, status::INVALID), "element size");

    // Nothing reached the display.
    assert!(h.host.embedder.shown().is_empty());
    assert_eq!(h.host.target(0).unwrap().presented(), 0);
}

/// A canvas that refuses every configuration, as a browser does for a value it does not implement.
struct Refusing(Offscreen<Native>);

impl Target<Native> for Refusing {
    fn size(&self) -> (u32, u32) {
        self.0.size()
    }
    fn set_size(&mut self, w: u32, h: u32) {
        self.0.set_size(w, h)
    }
    fn configure(&mut self, _: &CanvasConfiguration<'_, Native>) -> Result<(), webmetal_host::gpu::Exception> {
        Err(webmetal_host::gpu::Exception::new(ExceptionKind::TypeError, "colorSpace: invalid enum value"))
    }
    fn unconfigure(&mut self) {
        self.0.unconfigure()
    }
    fn write_frame(
        &mut self,
        q: &<Native as webmetal_host::gpu::Api>::Queue,
        d: &[u8],
        bpr: u32,
        w: u32,
        h: u32,
    ) -> Result<(), webmetal_host::gpu::Exception> {
        self.0.write_frame(q, d, bpr, w, h)
    }
    fn present(&mut self) -> Result<(), webmetal_host::gpu::Exception> {
        self.0.present()
    }
}

#[test]
fn a_canvas_that_refuses_the_configuration_is_reported() {
    let h = H::new();
    h.host.embedder.create_surface(surface(1, code::BGRA_32, 8, 8, 32)).unwrap();
    h.host.presenter.attach(0, srgb(8, 8), Box::new(Refusing(Offscreen::new(8, 8)))).unwrap();
    h.host.embedder.add_display(0);
    assert_eq!(refused(&h, frame(0, 1, 0, 0, 8, 8)), (Some("present-host-canvas"), status::HOST_UNSUPPORTED));
    assert!(h.host.embedder.shown().is_empty());
}

#[test]
fn presenting_on_a_lost_device_is_refused() {
    let h = H::new();
    h.host.embedder.create_surface(surface(1, code::BGRA_32, 8, 8, 32)).unwrap();
    h.host.attach_offscreen(0, srgb(8, 8)).unwrap();
    h.block_on(h.host.presenter.present(frame(0, 1, 0, 0, 8, 8))).unwrap();
    h.host.device.destroy();
    h.gpu.process_events();
    let f = h.block_on(h.host.presenter.present(frame(0, 1, 0, 0, 8, 8))).unwrap_err();
    assert_eq!((f.refusal, f.failure.status), (Some("device-removed"), status::UNIMPLEMENTED), "{f:?}");
    assert_eq!(h.host.embedder.shown().len(), 1);
}

#[test]
fn the_offscreen_target_follows_the_canvas_steps() {
    let h = H::new();
    let device = h.host.device.clone();
    let mut t = Offscreen::<Native>::new(4, 2);
    let e = t.get_current_texture().unwrap_err();
    assert_eq!(e.kind, ExceptionKind::InvalidStateError);
    let e = t.configure(&CanvasConfiguration::new(&device, TextureFormat::Bgra8unormSrgb)).unwrap_err();
    assert_eq!(e.kind, ExceptionKind::TypeError, "srgb formats are not context formats");
    let mut c = CanvasConfiguration::new(&device, TextureFormat::Rgba8unorm);
    c.usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::TRANSIENT_ATTACHMENT;
    assert_eq!(t.configure(&c).unwrap_err().kind, ExceptionKind::TypeError);
    if !device.features().contains("texture-compression-astc") {
        let mut c = CanvasConfiguration::new(&device, TextureFormat::Rgba8unorm);
        c.view_formats = vec![TextureFormat::Astc4x4Unorm];
        assert_eq!(t.configure(&c).unwrap_err().kind, ExceptionKind::TypeError, "a view format's feature");
    }
    assert!(t.configuration().is_none());

    for format in [TextureFormat::Bgra8unorm, TextureFormat::Rgba8unorm, TextureFormat::Rgba16float] {
        let mut c = CanvasConfiguration::new(&device, format);
        c.usage = TextureUsage::COPY_DST | TextureUsage::RENDER_ATTACHMENT;
        t.configure(&c).unwrap();
        let a = t.get_current_texture().unwrap();
        assert_eq!((a.width(), a.height(), a.format()), (4, 2, format));
        let b = t.get_current_texture().unwrap();
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        let n = 4 * 2 * webmetal_host::present::target::texel_bytes(format) as usize;
        let bytes: Vec<u8> = if format == TextureFormat::Rgba16float {
            // Finite, normal halves (0x3c00 = 1.0, 0x3800 = 0.5), which texel copies keep.
            (0..n / 2).flat_map(|i| if i % 2 == 0 { 0x3c00u16 } else { 0x3800 }.to_le_bytes()).collect()
        } else {
            pattern(n, 2)
        };
        let row = n as u32 / 2;
        t.write_frame(&device.queue(), &bytes, row, 4, 2).unwrap();
        t.present().unwrap();
        assert!(h.block_on(t.drawing_buffer()).unwrap().unwrap() == bytes, "{format:?}");
        let snap = h.block_on(t.snapshot()).unwrap();
        assert!(snap.opaque);
        // After presentation the next getCurrentTexture() replaces the drawing buffer with transparent black.
        let _ = t.get_current_texture().unwrap();
        assert!(h.block_on(t.drawing_buffer()).unwrap().unwrap().iter().all(|&b| b == 0), "{format:?}");
    }
    t.unconfigure();
    let snap = h.block_on(t.snapshot()).unwrap();
    assert_eq!((snap.format, snap.color_space.as_str(), snap.opaque), (None, "srgb", false));
    assert!(snap.bytes.iter().all(|&b| b == 0));
    h.gpu.process_events();
}

#[test]
fn every_poll_of_a_pending_frame_registers_its_deadline() {
    let h = H::new();
    h.host.embedder.create_surface(surface(1, code::BGRA_32, 4, 4, 16)).unwrap();
    h.host.attach_offscreen(0, srgb(4, 4)).unwrap();
    let mut r = frame(0, 1, 0, 0, 4, 4);
    r.frame.time = Some(MonotonicNanos(50));
    let mut f = h.host.presenter.present(r);
    assert!(matches!(poll_once(f.as_mut()), Poll::Pending));
    assert_eq!(h.host.presenter.next_deadline(), Some(MonotonicNanos(50)));
    h.host.embedder.set_time(MonotonicNanos(50));
    h.host.presenter.wake_due();
    assert_eq!(h.host.presenter.next_deadline(), None);
    h.block_on(f).unwrap();
}

#[test]
fn frames_of_a_detached_display_fail_and_a_new_attachment_starts_afresh() {
    let h = H::new();
    h.host.embedder.create_surface(surface(1, code::BGRA_32, 4, 4, 16)).unwrap();
    h.host.attach_offscreen(0, srgb(4, 4)).unwrap();
    let mut timed = frame(0, 1, 0, 0, 4, 4);
    timed.frame.time = Some(MonotonicNanos(100));
    let mut old = h.host.presenter.present(timed);
    assert!(poll_once(old.as_mut()).is_pending());
    assert!(h.host.presenter.detach(0).is_some());
    h.host.attach_offscreen(0, srgb(4, 4)).unwrap();
    let p = h.block_on(h.host.presenter.present(frame(0, 1, 0, 0, 4, 4))).unwrap();
    assert_eq!(p.sequence, 0, "the new attachment numbers its own frames");
    h.host.embedder.set_time(MonotonicNanos(100));
    let f = h.block_on(old).unwrap_err();
    assert_eq!((f.refusal, f.failure.status), (None, status::INVALID), "{f:?}");
    assert_eq!(h.host.embedder.shown().len(), 1);
}

/// A texture of the task, filled from a buffer by a committed blit.
fn filled_texture(h: &H, q: Handle, d: TextureDescriptor, bytes: &[u8], bytes_per_row: u64) -> Handle {
    let t = h.handle();
    h.ok(&Request::CreateTexture(CreateTexture { texture: t, descriptor: d.clone() }));
    let addr = h.alloc(bytes.len() as u64);
    h.host.embedder.cpu_write(TASK, addr, bytes).unwrap();
    let b = h.handle();
    h.ok(&Request::CreateBuffer(CreateBuffer {
        buffer: b,
        length: bytes.len() as u64,
        options: 0,
        contents: GuestAddr(addr),
    }));
    let e = h.handle();
    submit(
        h,
        q,
        &[
            Command::BeginBlit(BeginBlit { encoder: e }),
            Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
                source_buffer: Some(b),
                source_offset: 0,
                source_bytes_per_row: bytes_per_row,
                source_bytes_per_image: bytes_per_row * d.height,
                source_size: Size { width: d.width, height: d.height, depth: 1 },
                destination_texture: Some(t),
                destination_slice: 0,
                destination_level: 0,
                destination_origin: Origin { x: 0, y: 0, z: 0 },
            }),
            end(),
        ],
    );
    t
}

fn texture_frame(
    display: u32,
    texture: Handle,
    x: i64,
    y: i64,
    w: u64,
    ht: u64,
) -> webmetal_host::present::SwapRequest {
    let mut r = frame(display, 0, x, y, w, ht);
    r.frame.source = PresentSource::Texture { task: TASK, texture };
    r
}

#[test]
fn frames_of_a_task_texture_show_its_texels_exactly() {
    let h = H::new();
    h.open();
    let q = h.queue();
    // Private BGRA8Unorm and RGBA8Unorm_sRGB textures written by blits that are still in flight when the frames are
    // requested: each frame waits for the blit that writes its texture.
    let bgra = pattern(40 * 24 * 4, 21);
    let t = filled_texture(&h, q, texture_descriptor(BGRA8_UNORM, 40, 24, 2, 0x5), &bgra, 160);
    h.host.attach_offscreen(0, srgb(40, 24)).unwrap();
    let p = h.block_on(h.host.presenter.present(texture_frame(0, t, 0, 0, 40, 24))).unwrap();
    assert_eq!(p.plan.format, TextureFormat::Bgra8unorm);
    assert!(drawing_buffer(&h, 0) == bgra);
    h.host.attach_offscreen(1, srgb(16, 8)).unwrap();
    h.block_on(h.host.presenter.present(texture_frame(1, t, 3, 5, 16, 8))).unwrap();
    assert!(drawing_buffer(&h, 1) == rows(&bgra, 160, 3, 5, 16, 8));

    let rgba = pattern(40 * 24 * 4, 22);
    let u = filled_texture(&h, q, texture_descriptor(71, 40, 24, 2, 0x5), &rgba, 160);
    let p = h.block_on(h.host.presenter.present(texture_frame(0, u, 0, 0, 40, 24))).unwrap();
    assert_eq!(p.plan.format, TextureFormat::Rgba8unorm);
    assert!(drawing_buffer(&h, 0) == rgba);

    // A texture over an IOSurface plane is shown from the plane's memory.
    let (_, d) = drawable(&h, 9);
    let cpu = pattern(320 * 32, 23);
    h.host.embedder.surface_cpu_write(SurfaceId(9), 0, &cpu).unwrap();
    h.host.attach_offscreen(2, srgb(64, 32)).unwrap();
    h.block_on(h.host.presenter.present(texture_frame(2, d, 0, 0, 64, 32))).unwrap();
    assert!(drawing_buffer(&h, 2) == rows(&cpu, 320, 0, 0, 64, 32));

    // Textures the sink does not show exactly, and objects that are no frame.
    for (format, id) in
        [(94, "present-10-bit-unorm"), (115, "present-half-float"), (10, "present-texture-pixel-format")]
    {
        let v = h.handle();
        h.ok(&Request::CreateTexture(CreateTexture {
            texture: v,
            descriptor: texture_descriptor(format, 16, 8, 2, 0x5),
        }));
        assert_eq!(refused(&h, texture_frame(1, v, 0, 0, 16, 8)), (Some(id), status::UNIMPLEMENTED), "{format}");
    }
    let mut array = texture_descriptor(BGRA8_UNORM, 16, 8, 2, 0x5);
    array.texture_type = 3;
    array.array_length = 2;
    let a = h.handle();
    h.ok(&Request::CreateTexture(CreateTexture { texture: a, descriptor: array }));
    assert_eq!(refused(&h, texture_frame(1, a, 0, 0, 16, 8)), (None, status::INVALID), "a 2D array texture");
    let b = h.handle();
    let addr = h.alloc(64);
    h.ok(&Request::CreateBuffer(CreateBuffer { buffer: b, length: 64, options: 0, contents: GuestAddr(addr) }));
    assert_eq!(refused(&h, texture_frame(1, b, 0, 0, 16, 8)), (None, status::INVALID), "a buffer");
    assert_eq!(refused(&h, texture_frame(1, t, 30, 20, 16, 8)), (None, status::INVALID), "outside the texture");
}

#[test]
fn a_frame_waits_only_for_the_work_that_can_write_its_source() {
    let h = H::new();
    h.open();
    let (_, drawable_texture) = drawable(&h, 1);
    h.host.attach_offscreen(0, srgb(64, 32)).unwrap();
    let event = h.handle();
    h.ok(&Request::CreateEvent(CreateEvent { event }));
    let wait = || Command::EncodeWaitForEvent(EncodeWaitForEvent { event: Some(event), value: 1 });
    let blit_into = |t: Handle, bytes: &[u8]| {
        let addr = h.alloc(bytes.len() as u64);
        h.host.embedder.cpu_write(TASK, addr, bytes).unwrap();
        let b = h.handle();
        h.ok(&Request::CreateBuffer(CreateBuffer {
            buffer: b,
            length: bytes.len() as u64,
            options: 0,
            contents: GuestAddr(addr),
        }));
        let e = h.handle();
        vec![
            Command::BeginBlit(BeginBlit { encoder: e }),
            Command::BlitCopyBufferToTexture(BlitCopyBufferToTexture {
                source_buffer: Some(b),
                source_offset: 0,
                source_bytes_per_row: 256,
                source_bytes_per_image: 256 * 32,
                source_size: Size { width: 64, height: 32, depth: 1 },
                destination_texture: Some(t),
                destination_slice: 0,
                destination_level: 0,
                destination_origin: Origin { x: 0, y: 0, z: 0 },
            }),
            end(),
        ]
    };

    // A queue held by an event no one has signalled, about to write another texture and to bind a read-only texture
    // over the drawable's surface (a shader cannot write it: its usage lacks ShaderWrite), holds no frame of the
    // drawable.
    let other = h.handle();
    h.ok(&Request::CreateTexture(CreateTexture {
        texture: other,
        descriptor: texture_descriptor(BGRA8_UNORM, 64, 32, 2, 0x5),
    }));
    let read_only = h.handle();
    h.ok(&Request::CreateTextureWithIOSurface(CreateTextureWithIOSurface {
        texture: read_only,
        descriptor: texture_descriptor(BGRA8_UNORM, 64, 32, 0, 0x1),
        iosurface: SurfaceId(1),
        plane: 0,
    }));
    let held = h.queue();
    let mut records = vec![wait()];
    records.extend(blit_into(other, &pattern(64 * 32 * 4, 31)));
    let e = h.handle();
    records.extend([
        Command::BeginCompute(BeginCompute { encoder: e }),
        Command::ComputeSetTexture(ComputeSetTexture { texture: Some(read_only), index: 0 }),
        end(),
    ]);
    submit(&h, held, &records);
    let first = pattern(320 * 32, 32);
    h.host.embedder.surface_cpu_write(SurfaceId(1), 0, &first).unwrap();
    h.block_on(h.host.presenter.present(frame(0, 1, 0, 0, 64, 32))).unwrap();
    assert!(drawing_buffer(&h, 0) == rows(&first, 320, 0, 0, 64, 32));

    // A command buffer held by the same event that writes the drawable holds the next frame until it runs.
    let writer = h.queue();
    let second = pattern(64 * 32 * 4, 33);
    let mut records = vec![wait()];
    records.extend(blit_into(drawable_texture, &second));
    submit(&h, writer, &records);
    let mut f = h.host.presenter.present(frame(0, 1, 0, 0, 64, 32));
    for _ in 0..20 {
        assert!(poll_once(f.as_mut()).is_pending());
        h.gpu.process_events();
    }
    assert_eq!(h.host.embedder.shown().len(), 1);
    let signaller = h.queue();
    submit(&h, signaller, &[Command::EncodeSignalEvent(EncodeSignalEvent { event: Some(event), value: 1 })]);
    h.block_on(f).unwrap();
    assert!(drawing_buffer(&h, 0) == second);
}
