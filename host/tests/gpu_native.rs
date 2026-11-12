//! The conformance smoke test (`webmetal_host::gpu::smoke`) through the webgpu.h binding, against the implementation
//! the build links (see `host/build.rs`). The `check` lines must equal `tests/gpu_smoke_expected.txt`, the values the
//! WebGPU specification defines for each step, except for the differences recorded for the implementation
//! (`tests/gpu_smoke_known_<implementation>.txt`). Set `WEBMETAL_SMOKE_REPORT` to a path to keep the whole report.

#![cfg(feature = "native")]

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use webmetal_host::gpu::native::{Native, NativeGpu};
use webmetal_host::gpu::smoke;

#[test]
fn smoke_matches_the_specification() {
    let gpu = NativeGpu::new();
    let so_far = Rc::new(RefCell::new(String::new()));
    let sink = so_far.clone();
    let observer: smoke::Observer = Rc::new(move |line: &str| {
        let mut s = sink.borrow_mut();
        s.push_str(line);
        s.push('\n');
    });
    let report = match gpu.block_on(smoke::run_observed::<Native>(&gpu, Some(observer)), Duration::from_secs(60)) {
        Ok(r) => r,
        Err(stall) => panic!("{}: the smoke test stalled ({stall}) after:\n{}", gpu_name(&gpu), so_far.borrow()),
    };
    let text = report.text();
    if let Ok(path) = std::env::var("WEBMETAL_SMOKE_REPORT") {
        std::fs::write(&path, &text).expect("write report");
    }
    let expected = include_str!("gpu_smoke_expected.txt");
    let diff = smoke::conforms(expected, &text);
    let known: Vec<&str> = known_deviations().lines().filter(|l| !l.is_empty() && !l.starts_with('#')).collect();
    assert!(
        diff == known,
        "{} differs from the specification other than as recorded:\n{}\n\nrecorded:\n{}\n\nreport:\n{text}",
        gpu_name(&gpu),
        diff.join("\n"),
        known.join("\n")
    );
}

/// The recorded differences of the linked implementation (none for Dawn).
fn known_deviations() -> &'static str {
    if cfg!(webgpu_native = "wgpu-native") { include_str!("gpu_smoke_known_wgpu-native.txt") } else { "" }
}

fn gpu_name(gpu: &NativeGpu) -> String {
    use webmetal_host::gpu::Gpu;
    gpu.implementation()
}
