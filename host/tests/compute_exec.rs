//! The compute rows of the Metal reference harness (tests/reference/COVERAGE.md, "Compute and blit encoders") replayed
//! through the executor on a native WebGPU implementation, as tests/reference/src/cases/Compute.m dispatches them, with
//! every `exact` record of the reference output compared byte for byte. The kernels come from the pipeline stand-in of
//! `h6_common` (hand-translated WGSL of tests/reference/shaders/compute.metal) until the AIR to WGSL compiler produces
//! them; what these tests check is the encoder: dispatch forms, bindings and offsets, inline bytes, threadgroup
//! memory, stage-in regions, indirect arguments, barriers and coherence with guest memory.

#![cfg(feature = "native")]

mod h6_common;
mod mem_common;

use h6_common::*;
use mem_common::Golden;
use webmetal_wire::*;

fn set_pipeline(p: Handle) -> Command {
    Command::ComputeSetComputePipelineState(ComputeSetComputePipelineState { state: Some(p) })
}

fn set_buffer(b: Handle, offset: u64, index: u64) -> Command {
    Command::ComputeSetBuffer(ComputeSetBuffer { buffer: Some(b), offset, index })
}

fn set_bytes(bytes: &[u8], index: u64) -> Command {
    Command::ComputeSetBytes(ComputeSetBytes { bytes: bytes.to_vec(), length: bytes.len() as u64, index })
}

fn dispatch(groups: Size, threads: Size) -> Command {
    Command::ComputeDispatchThreadgroups(ComputeDispatchThreadgroups {
        threadgroups_per_grid: groups,
        threads_per_threadgroup: threads,
    })
}

fn grid(x: u32, y: u32, z: u32) -> Vec<u8> {
    [x, y, z, 0].iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn exact(h: &Harness, g: &Golden, name: &str, got: &[u8], count: &mut usize) {
    assert_eq!(g.rule(name), Some("exact"), "{}: {name}", g.case);
    assert_eq!(got, g.bytes(name).unwrap().as_slice(), "{}: {}: {name}", h.name(), g.case);
    *count += 1;
}

fn report(h: &Harness, case: &str, count: usize, note: &str) {
    eprintln!("{}: {case}: {count} exact records equal{}{note}", h.name(), if note.is_empty() { "" } else { "; " });
}

/// compute.thread-positions, compute.indirect-dispatch, and `dispatchThreads:` on the A10X.
#[test]
fn dispatch_forms() {
    let h = Harness::new();
    let p = h.pipeline("compute_positions", None, 0);

    let g = Golden::load("compute.thread-positions");
    let (out, out_addr) = h.buffer(12 * 4 * 2 * 16 * 4);
    h.compute(vec![
        set_pipeline(p),
        set_buffer(out, 0, 0),
        set_bytes(&grid(12, 4, 2), 1),
        dispatch(size(3, 2, 2), size(4, 2, 1)),
    ]);
    let mut n = 0;
    let records = h.read(out_addr, 12 * 4 * 2 * 64);
    exact(&h, &g, "records", &records, &mut n);
    report(&h, &g.case, n, "");

    // dispatchThreads: with a grid that is a whole number of threadgroups is the same dispatch.
    let (out2, out2_addr) = h.buffer(12 * 4 * 2 * 64);
    h.compute(vec![
        set_pipeline(p),
        set_buffer(out2, 0, 0),
        set_bytes(&grid(12, 4, 2), 1),
        Command::ComputeDispatchThreads(ComputeDispatchThreads {
            threads_per_grid: size(12, 4, 2),
            threads_per_threadgroup: size(4, 2, 1),
        }),
    ]);
    assert_eq!(h.read(out2_addr, records.len()), records, "{}: uniform dispatchThreads", h.name());
    eprintln!("{}: dispatchThreads with a uniform grid: equal to compute.thread-positions", h.name());

    // compute.dispatch-threads-nonuniform requires family.apple4; the A10X (Apple3) has no non-uniform threadgroups,
    // so the reference program's dispatch is refused.
    let g = Golden::load("compute.dispatch-threads-nonuniform");
    let (out3, _) = h.buffer(8 * 6 * 4 * 64);
    let f = h.compute_failure(vec![
        set_pipeline(p),
        set_buffer(out3, 0, 0),
        set_bytes(&grid(7, 5, 3), 1),
        Command::ComputeDispatchThreads(ComputeDispatchThreads {
            threads_per_grid: size(7, 5, 3),
            threads_per_threadgroup: size(4, 2, 2),
        }),
    ]);
    assert_eq!(f.status, status::UNIMPLEMENTED, "{f:?}");
    assert!(f.message.contains("non-uniform"), "{f:?}");
    report(&h, &g.case, 0, "the A10X profile has no non-uniform threadgroups: refused with status 2");

    let g = Golden::load("compute.indirect-dispatch");
    let (out, out_addr) = h.buffer(6 * 3 * 64);
    let args: Vec<u8> = [0xdead, 0xbeef, 0xfeed, 0xface, 3, 3, 1, 0u32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let (indirect, _) = h.buffer_with(&args);
    h.compute(vec![
        set_pipeline(p),
        set_buffer(out, 0, 0),
        set_bytes(&grid(6, 3, 1), 1),
        Command::ComputeDispatchThreadgroupsWithIndirectBuffer(ComputeDispatchThreadgroupsWithIndirectBuffer {
            indirect_buffer: Some(indirect),
            indirect_buffer_offset: 16,
            threads_per_threadgroup: size(2, 1, 1),
        }),
    ]);
    let mut n = 0;
    exact(&h, &g, "records", &h.read(out_addr, 6 * 3 * 64), &mut n);
    report(&h, &g.case, n, "");

    // A buffer bound at an offset that is not a multiple of the binding alignment: the residual reaches the kernel.
    let (big, big_addr) = h.buffer(12 * 4 * 2 * 64 + 4096);
    h.compute(vec![
        set_pipeline(p),
        set_buffer(big, 4 + 256, 0),
        set_bytes(&grid(12, 4, 2), 1),
        dispatch(size(3, 2, 2), size(4, 2, 1)),
    ]);
    assert_eq!(h.read(big_addr + 260, records.len()), records, "{}: offset binding", h.name());
    assert_eq!(h.read(big_addr, 260), vec![0; 260], "{}: bytes before the offset", h.name());
    eprintln!("{}: a binding at offset 260: equal to compute.thread-positions", h.name());
    h.assert_no_errors();
}

/// compute.threadgroup-memory-length, compute.threadgroup-atomics, compute.set-bytes, compute.dispatch-type-concurrent,
/// compute.stage-in-region.
#[test]
fn encoder_state() {
    let h = Harness::new();

    let g = Golden::load("compute.threadgroup-memory-length");
    let p = h.pipeline("compute_threadgroup_mirror", None, 0);
    let (out, addr) = h.buffer(64 * 3 * 4);
    h.compute(vec![
        set_pipeline(p),
        set_buffer(out, 0, 0),
        Command::ComputeSetThreadgroupMemoryLength(ComputeSetThreadgroupMemoryLength { length: 64 * 4, index: 0 }),
        dispatch(size(3, 1, 1), size(64, 1, 1)),
    ]);
    let mut n = 0;
    exact(&h, &g, "mirrored", &h.read(addr, 64 * 3 * 4), &mut n);
    report(&h, &g.case, n, "");

    let g = Golden::load("compute.threadgroup-atomics");
    let p = h.pipeline("compute_threadgroup_atomics", None, 0);
    let (out, addr) = h.buffer(4 * 2 * 4);
    h.compute(vec![set_pipeline(p), set_buffer(out, 0, 0), dispatch(size(4, 1, 1), size(32, 1, 1))]);
    let mut n = 0;
    exact(&h, &g, "sum-and-max", &h.read(addr, 32), &mut n);
    report(&h, &g.case, n, "");

    let g = Golden::load("compute.set-bytes");
    let p = h.pipeline("compute_copy_constant", None, 0);
    let words = pattern(4096, 11);
    let (out, addr) = h.buffer(4096);
    h.compute(vec![
        set_pipeline(p),
        set_buffer(out, 0, 0),
        set_bytes(&words, 1),
        dispatch(size(16, 1, 1), size(64, 1, 1)),
    ]);
    let copied = h.read(addr, 4096);
    let mut n = 0;
    assert_eq!(g.value_bool("equal-to-input"), Some(copied == words));
    n += 1;
    exact(&h, &g, "copied", &copied, &mut n);
    report(&h, &g.case, n, "");

    let g = Golden::load("compute.dispatch-type-concurrent");
    let add = h.pipeline("compute_add_one", None, 0);
    let scale = h.pipeline("compute_scale", None, 0);
    let input: Vec<u8> = (0..256u32).flat_map(|i| (i * 3).to_le_bytes()).collect();
    let (data, _) = h.buffer_with(&input);
    let (out, addr) = h.buffer(1024);
    let e = h.handle();
    h.encode(
        Command::BeginComputeWithDispatchType(BeginComputeWithDispatchType { encoder: e, dispatch_type: 1 }),
        vec![
            set_pipeline(add),
            set_buffer(data, 0, 0),
            dispatch(size(4, 1, 1), size(64, 1, 1)),
            Command::ComputeMemoryBarrierWithScope(ComputeMemoryBarrierWithScope { scope: 1 }),
            set_pipeline(scale),
            set_buffer(out, 0, 0),
            set_buffer(data, 0, 1),
            set_bytes(&5u32.to_le_bytes(), 2),
            dispatch(size(4, 1, 1), size(64, 1, 1)),
        ],
    );
    let mut n = 0;
    assert_eq!(g.value_u64("dispatch-type").map(|v| v as i64), Some(1));
    exact(&h, &g, "result", &h.read(addr, 1024), &mut n);
    report(&h, &g.case, n, "");

    let g = Golden::load("compute.stage-in-region");
    let stage = StageInputOutputDescriptor {
        attributes: vec![
            Attribute { buffer_index: 1, format: 13, offset: 0 },
            Attribute { buffer_index: 1, format: 28, offset: 4 },
        ],
        index_buffer_index: 0,
        index_type: 0,
        layouts: vec![
            BufferLayout { step_function: 1, step_rate: 1, stride: 0 },
            BufferLayout { step_function: 5, step_rate: 1, stride: 8 },
        ],
    };
    let p = h.pipeline("compute_stage_in", Some(stage), 0);
    let mut records = words_of(&pattern(128, 41));
    for i in 0..16 {
        records[i * 2 + 1] = (records[i * 2 + 1] & 0x807f_ffff) | (0x7e << 23);
    }
    let bytes: Vec<u8> = records.iter().flat_map(|w| w.to_le_bytes()).collect();
    let (input, _) = h.buffer_with(&bytes);
    let (out, addr) = h.buffer(6 * 2 * 4 * 4);
    h.compute(vec![
        set_pipeline(p),
        set_buffer(out, 0, 0),
        set_buffer(input, 0, 1),
        Command::ComputeSetStageInRegion(ComputeSetStageInRegion {
            region: Region { origin: origin(5, 0, 0), size: size(6, 2, 1) },
        }),
        dispatch(size(1, 1, 1), size(6, 2, 1)),
    ]);
    let mut n = 0;
    exact(&h, &g, "input", &bytes, &mut n);
    exact(&h, &g, "fetched", &h.read(addr, 6 * 2 * 16), &mut n);
    report(&h, &g.case, n, "");
    h.assert_no_errors();
}

fn words_of(bytes: &[u8]) -> Vec<u32> {
    h6_common::words(bytes)
}

/// Dispatches Metal does not define, or that the host cannot serve yet, complete with status 2 and the reason.
#[test]
fn compute_refusals() {
    let h = Harness::new();
    let p = h.pipeline("compute_positions", None, 96);
    let (out, _) = h.buffer(4096);
    let expect = |f: Failure, status: u32, what: &str| {
        assert_eq!(f.status, status, "{what}: {f:?}");
        assert!(f.message.contains(what), "{what}: {f:?}");
        eprintln!("{}: refused: {}", h.name(), f.message);
    };
    // validation.dispatch-without-pipeline.
    expect(
        h.compute_failure(vec![dispatch(size(1, 1, 1), size(1, 1, 1))]),
        status::UNIMPLEMENTED,
        "no compute pipeline",
    );
    // validation.threadgroup-above-pipeline-limit: 128 threads over a pipeline limited to 96.
    expect(
        h.compute_failure(vec![
            set_pipeline(p),
            set_buffer(out, 0, 0),
            set_bytes(&grid(1, 1, 1), 1),
            dispatch(size(1, 1, 1), size(128, 1, 1)),
        ]),
        status::UNIMPLEMENTED,
        "maxTotalThreadsPerThreadgroup",
    );
    // validation.set-bytes-over-4k.
    expect(h.compute_failure(vec![set_bytes(&vec![0; 4100], 0)]), status::UNIMPLEMENTED, "the limit is 4096");
    // validation.set-buffer-index-over-limit.
    expect(h.compute_failure(vec![set_buffer(out, 0, 31)]), status::UNIMPLEMENTED, "index 31");
    // An unbound buffer the kernel uses.
    expect(
        h.compute_failure(vec![set_pipeline(p), set_bytes(&grid(1, 1, 1), 1), dispatch(size(1, 1, 1), size(1, 1, 1))]),
        status::UNIMPLEMENTED,
        "buffer index 0",
    );
    expect(
        h.compute_failure(vec![Command::ComputeExecuteCommandsInBuffer(ComputeExecuteCommandsInBuffer {
            indirect_command_buffer: None,
            execution_range: Range { location: 0, length: 1 },
        })]),
        status::UNIMPLEMENTED,
        "indirect command buffer",
    );
    expect(
        h.compute_failure(vec![Command::ComputeSampleCountersInBuffer(ComputeSampleCountersInBuffer {
            sample_buffer: None,
            sample_index: 0,
            barrier: false,
        })]),
        status::UNIMPLEMENTED,
        "MTLCounterSamplingPointAtDispatchBoundary",
    );
    h.assert_no_errors();
}
