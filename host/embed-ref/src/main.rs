//! `webmetal-embed-ref`: replay a recording on a native WebGPU implementation.
//!
//! ```text
//! webmetal-embed-ref RECORDING [--frames DIR] [--monotonic-clock]
//! ```
//!
//! Prints one line per step that is not plain bookkeeping: each control reply (and whether it equals the recorded
//! one) and each swap's outcome. With `--frames`, each shown frame's drawing buffer is written to DIR as a binary PPM
//! (`display<D>-frame<N>.ppm`, the 8-bit red, green and blue codes; the alpha codes are not shown by an opaque display).
//! The clock is manual (it moves only at the recording's SetTime steps) unless `--monotonic-clock` is given. Exits with
//! status 1 when a reply differs from the recorded one, a step failed or a swap was not shown, 2 on a usage or setup
//! error.

use std::cell::Cell;
use std::path::PathBuf;
use std::process::ExitCode;
use std::rc::Rc;
use std::time::{Duration, Instant};

use webmetal_embed_ref::host::native;
use webmetal_embed_ref::recording::{self, Step, StepOutcome};
use webmetal_embed_ref::{Clock, RefEmbedder, RefHost};
use webmetal_host::exec::Config;
use webmetal_host::gpu::Gpu as _;
use webmetal_host::gpu::idl::TextureFormat;
use webmetal_host::gpu::native::Native;
use webmetal_host::gpu::scope::{Granularity, Policy};

fn usage() -> ExitCode {
    eprintln!("usage: webmetal-embed-ref RECORDING [--frames DIR] [--monotonic-clock]");
    ExitCode::from(2)
}

/// A binary PPM of `bytes` (rows of `width` texels of `format`), or `None` for a format with no 8-bit form.
fn ppm(format: TextureFormat, width: u32, height: u32, bytes: &[u8]) -> Option<Vec<u8>> {
    let (r, g, b) = match format {
        TextureFormat::Bgra8unorm => (2, 1, 0),
        TextureFormat::Rgba8unorm => (0, 1, 2),
        _ => return None,
    };
    let mut out = format!("P6\n{width} {height}\n255\n").into_bytes();
    for t in bytes.as_chunks::<4>().0 {
        out.extend_from_slice(&[t[r], t[g], t[b]]);
    }
    Some(out)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut frames: Option<PathBuf> = None;
    let mut monotonic = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--frames" => match args.next() {
                Some(d) => frames = Some(PathBuf::from(d)),
                None => return usage(),
            },
            "--monotonic-clock" => monotonic = true,
            _ if path.is_none() && !a.starts_with('-') => path = Some(PathBuf::from(a)),
            _ => return usage(),
        }
    }
    let Some(path) = path else { return usage() };
    let file = match std::fs::read(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    let steps = match recording::decode(&file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    if let Some(d) = &frames
        && let Err(e) = std::fs::create_dir_all(d)
    {
        eprintln!("{}: {e}", d.display());
        return ExitCode::from(2);
    }
    let (gpu, device) = match native::device("webmetal-embed-ref") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("WebGPU: {e}");
            return ExitCode::from(2);
        }
    };
    let clock = if monotonic { Clock::Monotonic(Instant::now()) } else { Clock::Manual(Cell::new(0)) };
    let embedder = Rc::new(RefEmbedder::new(clock));
    let config = Config { scopes: Policy { granularity: Granularity::PerCall, strict: true } };
    let wait = Duration::from_secs(120);
    let host = match gpu.block_on(RefHost::<Native>::new(device, embedder, config), wait) {
        Ok(Ok(h)) => h,
        Ok(Err(e)) => {
            eprintln!("setup: {e}");
            return ExitCode::from(2);
        }
        Err(e) => {
            eprintln!("setup: {e}");
            return ExitCode::from(2);
        }
    };
    println!("{}: {} steps on {}", path.display(), steps.len(), gpu.implementation());
    let outcomes = match gpu.block_on(recording::run(&host, &steps, frames.is_some()), wait) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("the recording stalled: {e}");
            return ExitCode::from(1);
        }
    };
    let mut bad = 0;
    for (i, (step, o)) in steps.iter().zip(&outcomes).enumerate() {
        match o {
            StepOutcome::Done => {}
            StepOutcome::Failed(e) => {
                bad += 1;
                println!("step {i}: failed: {e}");
            }
            StepOutcome::Reply { reply, matches } => {
                let op = reply.get(..2).map_or(0, |b| u16::from_le_bytes([b[0], b[1]]));
                let verdict = match matches {
                    Some(true) => "matches the recording",
                    Some(false) => {
                        bad += 1;
                        "DIFFERS from the recording"
                    }
                    None => "not recorded",
                };
                println!("step {i}: reply {op:#06x}, {} bytes, {verdict}", reply.len());
            }
            StepOutcome::Presented { result: Ok(p), drawing_buffer } => {
                println!(
                    "step {i}: display {} frame {} shown at {} ns ({} {}x{}, {})",
                    p.display,
                    p.sequence,
                    p.presented_at.0,
                    p.plan.format.as_str(),
                    p.plan.width,
                    p.plan.height,
                    p.plan.color_space
                );
                if let (Some(dir), Some(bytes)) = (&frames, drawing_buffer) {
                    let name = dir.join(format!("display{}-frame{}.ppm", p.display, p.sequence));
                    match ppm(p.plan.format, p.plan.width, p.plan.height, bytes) {
                        Some(img) => {
                            if let Err(e) = std::fs::write(&name, img) {
                                eprintln!("{}: {e}", name.display());
                                return ExitCode::from(2);
                            }
                        }
                        None => println!("  (no 8-bit image of {})", p.plan.format.as_str()),
                    }
                }
            }
            StepOutcome::Presented { result: Err(f), .. } => {
                bad += 1;
                let what = if let Step::Present { request } = step { request.frame.display } else { u32::MAX };
                println!(
                    "step {i}: display {what}: frame NOT shown (status {}): {}",
                    f.failure.status, f.failure.message
                );
            }
        }
    }
    for r in host.exec.take_reports() {
        println!("report: task {} status {}: {}", r.task.0, r.failure.status, r.failure.message);
    }
    if bad == 0 { ExitCode::SUCCESS } else { ExitCode::from(1) }
}
