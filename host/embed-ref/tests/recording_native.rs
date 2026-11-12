//! Recordings replayed on a native WebGPU implementation (`WEBGPU_NATIVE`, see `host/build.rs`), against the
//! Metal reference outputs. Run with `--release`.
//!
//! `drawable.render-present` of the reference harness (`tests/reference/src/cases/Drawables.m`) renders into a
//! 16 x 8 BGRA8Unorm drawable of a `CAMetalLayer`, copies it to a buffer and presents it. On the host side that is an
//! IOSurface-backed texture, a render pass that clears it, a blit to a shared buffer, a commit, and the surface shown
//! as a frame. The rows compared are the ones the host produces: the command buffer's status and error, the buffer's
//! bytes, and, beyond the case, the frame. The layer and drawable rows (`texture.*`, `drawable.*`, the presented
//! handler) belong to QuartzCore and the guest plugin.

#![cfg(feature = "native")]

mod common;

use std::cell::Cell;
use std::rc::Rc;

use common::*;
use webmetal_embed_ref::recording::{self, Step, StepOutcome};
use webmetal_embed_ref::{Clock, RefEmbedder, RefHost};
use webmetal_host::exec::Config;
use webmetal_host::exec::failure::command_buffer_status;
use webmetal_host::gpu::scope::{Granularity, Policy};
use webmetal_host::present::format::code;
use webmetal_wire::*;

fn handle(v: u64) -> Handle {
    Handle::new(v).unwrap()
}

/// The steps of `drawable.render-present`, and where its pixel buffer lives.
fn render_present() -> (Vec<Step>, u64) {
    let (queue, texture, buffer, render, blit, cb) = (handle(1), handle(2), handle(3), handle(4), handle(5), handle(6));
    let contents = BASE + 0x1000;
    let stream_addr = BASE + 0x2000;
    // CAMetalLayer's drawable for a 16 x 8 BGRA8Unorm layer: an IOSurface of 'BGRA' pixels, 64 bytes per row.
    let desc = surface(1, code::BGRA_32, 16, 8, 64);
    // The texture over it: usage ShaderRead | ShaderWrite | RenderTarget | PixelFormatView (0x17, as Core Animation
    // chose on the reference system), shared storage (iOS has no managed storage).
    let tex = texture_descriptor(BGRA8_UNORM, 16, 8, 0, 0x17);
    let (bytes, resources) = stream(&[
        Command::BeginRenderPass(BeginRenderPass {
            encoder: render,
            render_pass_descriptor: pass_clear(texture, ClearColor { red: 1.0, green: 0.0, blue: 1.0, alpha: 1.0 }),
        }),
        end(),
        Command::BeginBlit(BeginBlit { encoder: blit }),
        Command::BlitCopyTextureToBuffer(BlitCopyTextureToBuffer {
            source_texture: Some(texture),
            source_slice: 0,
            source_level: 0,
            source_origin: Origin { x: 0, y: 0, z: 0 },
            source_size: Size { width: 16, height: 8, depth: 1 },
            destination_buffer: Some(buffer),
            destination_offset: 0,
            destination_bytes_per_row: 64,
            destination_bytes_per_image: 512,
        }),
        end(),
    ]);
    let control = |r: Request| Step::Control { task: TASK, request: encode(&r), reply: None };
    let steps = vec![
        Step::Map { task: TASK, addr: BASE, len: 0x10000 },
        Step::CreateSurface { desc },
        Step::Attach { display: 0, mode: srgb(16, 8) },
        control(open()),
        control(Request::CreateCommandQueue(CreateCommandQueue { queue })),
        control(Request::CreateTextureWithIOSurface(CreateTextureWithIOSurface {
            texture,
            descriptor: tex,
            iosurface: SurfaceId(1),
            plane: 0,
        })),
        control(Request::CreateBuffer(CreateBuffer { buffer, length: 512, options: 0, contents: GuestAddr(contents) })),
        Step::Write { task: TASK, addr: stream_addr, bytes: bytes.clone() },
        control(commit(cb, queue, stream_addr, bytes.len(), resources)),
        // presentDrawable: right after the commit; the sink waits for the command buffer.
        Step::Present { request: frame(0, 1, 0, 0, 16, 8) },
        Step::Wait,
    ];
    (steps, contents)
}

fn new_host() -> (webmetal_host::gpu::native::NativeGpu, RefHost<webmetal_host::gpu::native::Native>) {
    let (gpu, device) = webmetal_embed_ref::host::native::device("recording test device").unwrap();
    let embedder = Rc::new(RefEmbedder::new(Clock::Manual(Cell::new(0))));
    let config = Config { scopes: Policy { granularity: Granularity::PerCall, strict: true } };
    let wgsl = webmetal_embed_ref::host::wgsl_language_features::<webmetal_host::gpu::native::Native>(&gpu).unwrap();
    let host = gpu.block_on(RefHost::new(device, wgsl, embedder, config), WAIT).unwrap().unwrap();
    (gpu, host)
}

#[test]
fn drawable_render_present_matches_metal_through_a_recording_file() {
    let golden = Golden::load("drawable.render-present");
    let pixels = golden.bytes("pixels");
    assert_eq!(golden.rule("pixels"), "exact");
    assert_eq!(pixels.len(), 512);

    // Record the recording to a file and replay the file.
    let (steps, contents) = render_present();
    let file = recording::encode(&steps);
    // Left for the webmetal-embed-ref program: `webmetal-embed-ref <tmpdir>/drawable.render-present.wmrec`.
    let kept = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("drawable.render-present.wmrec");
    std::fs::write(&kept, &file).unwrap();
    let steps = recording::decode(&file).unwrap();
    let (gpu, host) = new_host();
    let outcomes = gpu.block_on(recording::run(&host, &steps, true), WAIT).unwrap();
    for (i, o) in outcomes.iter().enumerate() {
        match o {
            StepOutcome::Failed(e) => panic!("step {i}: {e}"),
            StepOutcome::Reply { reply, .. } => {
                let (r, _) = Reply::decode(reply, 0).unwrap();
                let f = webmetal_host::exec::schema::reply_failure(&r);
                assert!(f.is_none_or(|f| f.status == status::OK), "step {i}: {r:?}");
            }
            _ => {}
        }
    }

    // command-buffer.status and command-buffer.error.
    let completed: Vec<_> = host
        .embedder
        .posted()
        .into_iter()
        .filter_map(|p| match p.event {
            Event::CommandBufferCompleted(c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].status, command_buffer_status::COMPLETED);
    assert_eq!(golden.value("command-buffer.status"), command_buffer_status::COMPLETED.to_string());
    assert_eq!(completed[0].failure.status, status::OK);
    assert!(golden.error_nil("command-buffer.error"));

    // pixels: the buffer the blit wrote.
    assert_eq!(host.embedder.cpu_read(TASK, contents, 512).unwrap(), pixels);

    // The frame: the surface's bytes, which are Metal's pixels, are what the display's canvas holds.
    assert_eq!(host.embedder.surface_bytes(SurfaceId(1)).unwrap(), pixels);
    let StepOutcome::Presented { result: Ok(p), drawing_buffer: Some(shown) } = &outcomes[9] else {
        panic!("the frame was not shown: {:?}", outcomes[9]);
    };
    assert_eq!(p.sequence, 0);
    assert_eq!(*shown, pixels);
    let snap = gpu.block_on(host.target(0).unwrap().snapshot(), WAIT).unwrap().unwrap();
    assert_eq!(snap.bytes, pixels, "the cleared alpha is already 1.0");
}

#[test]
fn a_recording_with_recorded_replies_replays_identically() {
    let (steps, _) = render_present();
    let (gpu, host) = new_host();
    let first = gpu.block_on(recording::run(&host, &steps, false), WAIT).unwrap();
    // Record the replies the first run got, then replay on a new device.
    let recorded: Vec<Step> = steps
        .iter()
        .zip(&first)
        .map(|(s, o)| match (s, o) {
            (Step::Control { task, request, .. }, StepOutcome::Reply { reply, .. }) => {
                Step::Control { task: *task, request: request.clone(), reply: Some(reply.clone()) }
            }
            _ => s.clone(),
        })
        .collect();
    let file = recording::encode(&recorded);
    let (gpu2, host2) = new_host();
    let second = gpu2.block_on(recording::run(&host2, &recording::decode(&file).unwrap(), false), WAIT).unwrap();
    let mut replies = 0;
    for (i, o) in second.iter().enumerate() {
        match o {
            StepOutcome::Reply { matches, .. } => {
                replies += 1;
                assert_eq!(*matches, Some(true), "step {i}");
            }
            StepOutcome::Presented { result, .. } => assert!(result.is_ok(), "step {i}: {result:?}"),
            StepOutcome::Failed(e) => panic!("step {i}: {e}"),
            StepOutcome::Done => {}
        }
    }
    assert_eq!(replies, 5);
}
