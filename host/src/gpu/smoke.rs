//! The conformance smoke test: one sequence of WebGPU calls, written once against [`Api`], run through every
//! implementation (`native` over Dawn and over wgpu-native, `web` in a browser), and compared.
//!
//! Each step records a line. `check` lines hold only what the specification defines (exception kinds, error filters,
//! promise outcomes, attribute values, bytes read back from exact operations), so two conforming implementations
//! produce the same `check` lines; [`compare`] lists the ones that differ. `info` lines describe the implementation
//! (adapter, limits, messages) and are not compared.
//!
//! The steps cover: adapter and device requests with feature and limit negotiation and the requestDevice() rejections;
//! error scopes (capture, nesting, filters, the empty-stack rejection, uncaptured delivery) and the error-scope
//! plumbing of [`super::scope`]; buffer creation, mapping, mapped ranges and their detachment, `writeBuffer`; a compute
//! dispatch; texture upload and readback; a render pass with a scissor and a render bundle; asynchronous pipeline
//! creation; the content-timeline exceptions (`TypeError`, `RangeError`, `OperationError`, `AbortError`); labels; and
//! device destruction.

use std::cell::RefCell;
use std::fmt::Display;
use std::rc::Rc;

use super::api::*;
use super::idl::*;
use super::negotiate;
use super::num::{Float32, Float64, Size64};
use super::scope::{Attribution, Captured, ErrorScopes, Granularity, Policy};

/// Receives each line of a report as it is recorded (so a run that never finishes still shows how far it got).
pub type Observer = Rc<dyn Fn(&str)>;

/// The report of one run.
#[derive(Clone, Default)]
pub struct Report {
    /// (compared, key, value), in run order.
    pub lines: Vec<(bool, String, String)>,
    observer: Option<Observer>,
}

impl Report {
    fn push(&mut self, check: bool, key: &str, value: String) {
        if let Some(o) = &self.observer {
            o(&format!("{} {key} = {}", if check { "check" } else { "info" }, value.replace('\n', "\\n")));
        }
        self.lines.push((check, key.to_string(), value));
    }

    fn check(&mut self, key: &str, value: impl Display) {
        self.push(true, key, value.to_string());
    }

    fn info(&mut self, key: &str, value: impl Display) {
        self.push(false, key, value.to_string());
    }

    /// One line per entry: `check <key> = <value>` or `info <key> = <value>`.
    pub fn text(&self) -> String {
        let mut s = String::new();
        for (c, k, v) in &self.lines {
            s.push_str(if *c { "check " } else { "info " });
            s.push_str(k);
            s.push_str(" = ");
            s.push_str(&v.replace('\n', "\\n"));
            s.push('\n');
        }
        s
    }
}

/// The `check` lines of a report that differ from an expectation file: `key: expected | got`. An expected value may
/// list alternatives separated by ` || ` (for a step the hardware decides whether it can run).
pub fn conforms(expected: &str, report: &str) -> Vec<String> {
    let got = checks(report);
    let mut out = Vec::new();
    for (k, v) in checks(expected) {
        let g = got.iter().find(|(x, _)| *x == k).map(|(_, v)| v.clone());
        match g {
            Some(g) if v.split(" || ").any(|alt| alt == g) => {}
            Some(g) => out.push(format!("{k}: {v} | {g}")),
            None => out.push(format!("{k}: {v} | <absent>")),
        }
    }
    for (k, g) in &got {
        if !expected
            .lines()
            .any(|l| l.strip_prefix("check ").and_then(|l| l.split_once(" = ")).is_some_and(|(x, _)| x == k))
        {
            out.push(format!("{k}: <not expected> | {g}"));
        }
    }
    out
}

fn checks(t: &str) -> Vec<(String, String)> {
    t.lines()
        .filter_map(|l| l.strip_prefix("check "))
        .filter_map(|l| l.split_once(" = "))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// The `check` lines of two reports that differ, as `key: a | b` (a missing line is `<absent>`).
pub fn compare(a: &str, b: &str) -> Vec<String> {
    let (ca, cb) = (checks(a), checks(b));
    let mut keys: Vec<&String> = ca.iter().map(|(k, _)| k).collect();
    for (k, _) in &cb {
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    let get = |c: &[(String, String)], k: &str| {
        c.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone()).unwrap_or_else(|| "<absent>".into())
    };
    keys.into_iter()
        .filter_map(|k| {
            let (va, vb) = (get(&ca, k), get(&cb, k));
            (va != vb).then(|| format!("{k}: {va} | {vb}"))
        })
        .collect()
}

fn kind(e: &Exception) -> String {
    match &e.kind {
        ExceptionKind::PipelineError(r) => format!("GPUPipelineError({})", r.as_str()),
        ExceptionKind::Other(n) => n.clone(),
        k => format!("{k:?}"),
    }
}

/// `ok` or the exception kind.
fn outcome<T>(r: &Result<T, Exception>) -> String {
    match r {
        Ok(_) => "ok".into(),
        Err(e) => kind(e),
    }
}

fn error_filter(e: &Option<GpuError>) -> String {
    match e {
        None => "none".into(),
        Some(e) => e.filter.as_str().into(),
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sz(v: u64) -> Size64 {
    Size64::new(v).expect("test sizes are small")
}

fn f64v(v: f64) -> Float64 {
    Float64::new(v).expect("finite")
}

fn f32v(v: f32) -> Float32 {
    Float32::new(v).expect("finite")
}

fn rgba(r: f64, g: f64, b: f64, a: f64) -> Color {
    Color::ColorDict(ColorDict { r: f64v(r), g: f64v(g), b: f64v(b), a: f64v(a) })
}

/// The value of a step the rest of the run depends on; when the implementation fails it, the failure is recorded and
/// the run ends with the report so far.
macro_rules! must {
    ($r:ident, $what:literal, $e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => {
                $r.check(concat!("required step: ", $what), format!("failed: {}", e));
                return $r;
            }
        }
    };
}

fn ext(w: u32, h: u32) -> Extent3D {
    Extent3D::Extent3DDict(Extent3DDict { width: w, height: h, depth_or_array_layers: 1 })
}

const COMPUTE_WGSL: &str = "
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < arrayLength(&data)) {
        data[id.x] = data[id.x] * 2u + id.x;
    }
}
";

const RENDER_WGSL: &str = "
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    // One triangle that covers the whole target.
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(p[i], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 1.0, 1.0);
}
";

const BROKEN_WGSL: &str = "@compute @workgroup_size(1) fn main() { let x: u32 = undefined_name; }";

/// Read `size` bytes of a buffer through a MAP_READ staging buffer.
async fn read_buffer<A: Api>(device: &A::Device, src: &A::Buffer, size: u64) -> Result<Vec<u8>, String> {
    let staging = device
        .create_buffer(&BufferDescriptor::new(sz(size), BufferUsage::MAP_READ | BufferUsage::COPY_DST))
        .map_err(|e| e.to_string())?;
    let enc = device.create_command_encoder(&CommandEncoderDescriptor::new());
    enc.copy_buffer_to_buffer(src, &staging, Some(sz(size)));
    device.queue().submit(&[&enc.finish(&CommandBufferDescriptor::new())]);
    staging.map_async(MapMode::READ, sz(0), None).await.map_err(|e| e.to_string())?;
    let r = staging.get_mapped_range(sz(0), None).map_err(|e| e.to_string())?;
    let mut out = vec![0; size as usize];
    r.read(0, &mut out).map_err(|e| e.to_string())?;
    staging.unmap();
    Ok(out)
}

fn render_descriptor<'a, A: Api>(
    module: &'a A::ShaderModule,
    blend: Option<BlendState>,
) -> RenderPipelineDescriptor<'a, A> {
    let mut vs = VertexState::new(module);
    vs.entry_point = Some("vs".into());
    let mut target = ColorTargetState::new(TextureFormat::Rgba8unorm);
    target.blend = blend;
    let mut fs = FragmentState::new(module, vec![Some(target)]);
    fs.entry_point = Some("fs".into());
    let mut d = RenderPipelineDescriptor::new(PipelineLayoutOrAuto::AutoLayoutMode(AutoLayoutMode::Auto), vs);
    d.fragment = Some(fs);
    d
}

/// Run the smoke test on an implementation.
pub async fn run<A: Api>(gpu: &A::Gpu) -> Report {
    run_observed::<A>(gpu, None).await
}

/// Run the smoke test, handing each line to `observer` as it is recorded.
pub async fn run_observed<A: Api>(gpu: &A::Gpu, observer: Option<Observer>) -> Report {
    let mut r = Report { lines: Vec::new(), observer };
    r.info("implementation", gpu.implementation());
    match gpu.wgsl_language_features() {
        Ok(f) => r.info("gpu.wgslLanguageFeatures", f.into_iter().collect::<Vec<_>>().join(",")),
        Err(e) => r.info("gpu.wgslLanguageFeatures", kind(&e)),
    }

    // requestAdapter() with an unknown feature level resolves with null.
    let mut opts = RequestAdapterOptions::new();
    opts.feature_level = "no-such-level".into();
    let none = gpu.request_adapter(&opts).await;
    r.check(
        "requestAdapter(featureLevel unknown)",
        match &none {
            Ok(None) => "null".to_string(),
            Ok(Some(_)) => "adapter".to_string(),
            Err(e) => kind(e),
        },
    );

    let adapter = match gpu.request_adapter(&RequestAdapterOptions::new()).await {
        Ok(Some(a)) => a,
        other => {
            r.check(
                "requestAdapter",
                match other {
                    Ok(_) => "null".to_string(),
                    Err(e) => kind(&e),
                },
            );
            return r;
        }
    };
    r.check("requestAdapter", "adapter");
    let info = adapter.info();
    r.info("adapter.info", format!("{info:?}"));
    let afeatures = adapter.features();
    let alimits = adapter.limits();
    r.info("adapter.features", afeatures.iter().cloned().collect::<Vec<_>>().join(","));
    r.info("adapter.limits", format!("{alimits:?}"));
    r.check("adapter.features has core-features-and-limits", afeatures.contains("core-features-and-limits"));

    // Negotiation from the tables. An adapter the negotiation refuses is reported; the binding steps below still run,
    // on a device with the default features and limits, so that the report covers the binding itself.
    let plan = negotiate::plan(&afeatures, &alimits);
    let mut desc = match &plan {
        Ok(p) => {
            r.check("negotiate.plan", "ok");
            let unexposed: Vec<&str> = p.limits.iter().filter(|l| l.adapter.is_none()).map(|l| l.limit.name).collect();
            r.info("negotiate.unexposed_limits", unexposed.join(","));
            r.info("negotiate.unknown_adapter_features", p.unknown_adapter_features.join(","));
            p.descriptor.clone()
        }
        Err(e) => {
            r.check("negotiate.plan", format!("refused: {e}"));
            DeviceDescriptor::new()
        }
    };
    desc.label = "smoke device".into();
    desc.default_queue.label = "smoke queue".into();
    let device = match adapter.request_device(&desc).await {
        Ok(d) => d,
        Err(e) => {
            r.check("requestDevice(plan)", kind(&e));
            return r;
        }
    };
    r.check("requestDevice(plan)", "ok");
    match &plan {
        Ok(p) => {
            let mismatches = negotiate::verify(p, &device.features(), &device.limits());
            r.check(
                "negotiate.verify",
                if mismatches.is_empty() {
                    "none".to_string()
                } else {
                    mismatches.iter().map(|m| m.to_string()).collect::<Vec<_>>().join("; ")
                },
            );
        }
        Err(_) => r.check("negotiate.verify", "not run: the adapter was refused"),
    }
    r.check("device.label", device.label());
    // The specification does not say that defaultQueue.label becomes the queue's label.
    r.info("device.queue.label", device.queue().label());
    r.check("device.adapterInfo == adapter.info", device.adapter_info() == info);

    // A second adapter, for the requestDevice() rejections and a device with the default features.
    let plain = match gpu.request_adapter(&RequestAdapterOptions::new()).await {
        Ok(Some(a)) => Some(a),
        _ => None,
    };
    r.check("requestAdapter (second)", if plain.is_some() { "adapter" } else { "null" });
    let mut plain_device = None;
    if let Some(a2) = &plain {
        let mut bad = DeviceDescriptor::new();
        bad.required_limits.push(("maxNoSuchLimit".into(), Some(sz(1))));
        r.check("requestDevice(unknown limit)", outcome(&a2.request_device(&bad).await));
        let mut bad = DeviceDescriptor::new();
        let better = u64::from(a2.limits().max_bind_groups.unwrap_or(4)) + 1;
        bad.required_limits.push(("maxBindGroups".into(), Some(sz(better))));
        r.check("requestDevice(limit better than adapter)", outcome(&a2.request_device(&bad).await));
        let mut bad = DeviceDescriptor::new();
        bad.required_limits.push(("minUniformBufferOffsetAlignment".into(), Some(sz(384))));
        r.check("requestDevice(alignment not a power of two)", outcome(&a2.request_device(&bad).await));
        match FeatureName::ALL.iter().find(|f| !a2.features().contains(f.as_str())) {
            Some(f) => {
                let mut bad = DeviceDescriptor::new();
                bad.required_features.push(*f);
                r.check("requestDevice(unsupported feature)", outcome(&a2.request_device(&bad).await));
            }
            None => r.check("requestDevice(unsupported feature)", "untestable: the adapter supports every feature"),
        }
        let mut undefined_unknown = DeviceDescriptor::new();
        undefined_unknown.required_limits.push(("maxNoSuchLimit".into(), None));
        match a2.request_device(&undefined_unknown).await {
            Ok(d) => {
                r.check("requestDevice(unknown limit, undefined value)", "ok");
                plain_device = Some(d);
            }
            Err(e) => r.check("requestDevice(unknown limit, undefined value)", kind(&e)),
        }
    }

    // Error scopes on the device itself.
    let invalid_usage = BufferDescriptor::new(sz(16), BufferUsage::MAP_READ | BufferUsage::STORAGE);
    r.check(
        "popErrorScope(empty stack)",
        match device.pop_error_scope().await {
            Ok(e) => format!("resolved {}", error_filter(&e)),
            Err(e) => kind(&e),
        },
    );
    device.push_error_scope(ErrorFilter::Validation);
    let _ = device.create_buffer(&invalid_usage);
    r.check(
        "scope(validation) catches invalid createBuffer",
        error_filter(&device.pop_error_scope().await.ok().flatten()),
    );
    device.push_error_scope(ErrorFilter::Validation);
    device.push_error_scope(ErrorFilter::Validation);
    let _ = device.create_buffer(&invalid_usage);
    let inner = device.pop_error_scope().await.ok().flatten();
    let outer = device.pop_error_scope().await.ok().flatten();
    r.check("nested scopes: inner, outer", format!("{}, {}", error_filter(&inner), error_filter(&outer)));
    device.push_error_scope(ErrorFilter::Validation);
    device.push_error_scope(ErrorFilter::OutOfMemory);
    let _ = device.create_buffer(&invalid_usage);
    let oom = device.pop_error_scope().await.ok().flatten();
    let val = device.pop_error_scope().await.ok().flatten();
    r.check("validation error skips an out-of-memory scope", format!("{}, {}", error_filter(&oom), error_filter(&val)));

    // An error outside any scope reaches onuncapturederror (as a task, so give the event loop turns).
    let seen: Rc<RefCell<Vec<GpuError>>> = Rc::new(RefCell::new(Vec::new()));
    let s2 = seen.clone();
    device.set_onuncapturederror(Some(Box::new(move |e| s2.borrow_mut().push(e))));
    let _ = device.create_buffer(&invalid_usage);
    for _ in 0..50 {
        if !seen.borrow().is_empty() {
            break;
        }
        device.queue().on_submitted_work_done().await;
    }
    // A user agent may choose not to fire uncapturederror (§22.3, "dispatch an error"), so this is not compared.
    r.info("uncaptured error delivered", seen.borrow().iter().map(|e| e.filter.as_str()).collect::<Vec<_>>().join(","));
    device.set_onuncapturederror(None);

    // The plumbing, per call.
    let scopes: ErrorScopes<A> = ErrorScopes::new(&device, Policy { granularity: Granularity::PerCall, strict: false });
    let _ = scopes.call("GPUDevice.createBuffer", "bad", |d| d.create_buffer(&invalid_usage));
    let ok_buf = scopes.call("GPUDevice.createBuffer", "good", |d| {
        d.create_buffer(&BufferDescriptor::new(sz(16), BufferUsage::COPY_DST))
    });
    let got = scopes.settle().await;
    r.check(
        "plumbing per-call",
        got.iter()
            .map(|c| match c {
                Captured::Error { error, at: Attribution::Call(site) } => {
                    format!("{} from #{} {} {:?}", error.filter.as_str(), site.seq, site.op, site.label)
                }
                other => format!("unexpected {other}"),
            })
            .collect::<Vec<_>>()
            .join("; "),
    );
    r.check("plumbing per-call valid buffer", outcome(&ok_buf));
    r.check("plumbing per-call close", scopes.close().await.len());

    // The plumbing, batched: an error names the calls since the previous settle.
    let scopes: ErrorScopes<A> = ErrorScopes::new(&device, Policy { granularity: Granularity::Batch, strict: true });
    let _ = scopes.call("GPUDevice.createBuffer", "first", |d| {
        d.create_buffer(&BufferDescriptor::new(sz(16), BufferUsage::COPY_DST))
    });
    let _ = scopes.call("GPUDevice.createBuffer", "second", |d| d.create_buffer(&invalid_usage));
    let got = scopes.settle().await;
    r.check(
        "plumbing batch",
        got.iter()
            .map(|c| match c {
                Captured::Error { error, at: Attribution::Batch(calls) } => {
                    format!(
                        "{} from [{}]",
                        error.filter.as_str(),
                        calls.iter().map(|c| c.label.clone()).collect::<Vec<_>>().join(",")
                    )
                }
                other => format!("unexpected {other}"),
            })
            .collect::<Vec<_>>()
            .join("; "),
    );
    r.check("plumbing batch, nothing after", scopes.settle().await.len());

    // From here on the batch plumbing's standing frame is open: every device-timeline error the remaining steps raise
    // (the invalid mapAsync calls among them) lands in it, and strict mode panics if one escapes.
    let q = device.queue();

    // Buffers: mappedAtCreation, mapped ranges, detachment.
    let mut d = BufferDescriptor::new(sz(6), BufferUsage::COPY_SRC);
    d.mapped_at_creation = true;
    r.check("createBuffer(mappedAtCreation, size 6)", outcome(&device.create_buffer(&d)));
    let mut d = BufferDescriptor::new(sz(16), BufferUsage::COPY_SRC);
    d.mapped_at_creation = true;
    d.label = "src".into();
    let src = must!(r, "mapped buffer", device.create_buffer(&d));
    r.check("src.mapState", src.map_state().as_str());
    r.check("src.size, usage, label", format!("{}, {}, {}", src.size(), src.usage(), src.label()));
    src.set_label("renamed");
    r.check("src.label after set", src.label());
    r.check("getMappedRange(offset 4)", outcome(&src.get_mapped_range(sz(4), None)));
    let range = must!(r, "first half", src.get_mapped_range(sz(0), Some(sz(8))));
    r.check("getMappedRange overlapping", outcome(&src.get_mapped_range(sz(0), Some(sz(8)))));
    let range2 = must!(r, "second half", src.get_mapped_range(sz(8), None));
    must!(r, "write", range.write(0, &[1, 2, 3, 4, 5, 6, 7, 8]));
    must!(r, "write", range2.write(0, &[9, 10, 11, 12, 13, 14, 15, 16]));
    r.check("range write past end", outcome(&range.write(4, &[0; 8])));
    src.unmap();
    r.check("range after unmap: byteLength", range.byte_length());
    let mut b = [0u8; 4];
    r.check("range after unmap: read", outcome(&range.read(0, &mut b)));
    r.check("getMappedRange unmapped", outcome(&src.get_mapped_range(sz(0), None)));
    r.check("src.mapState after unmap", src.map_state().as_str());
    match read_buffer::<A>(&device, &src, 16).await {
        Ok(v) => r.check("mappedAtCreation contents", hex(&v)),
        Err(e) => r.check("mappedAtCreation contents", format!("failed: {e}")),
    }

    // mapAsync outcomes.
    let rb = must!(
        r,
        "buffer",
        device.create_buffer(&BufferDescriptor::new(sz(16), BufferUsage::MAP_READ | BufferUsage::COPY_DST))
    );
    let first = rb.map_async(MapMode::READ, sz(0), None);
    r.check("mapState after mapAsync", rb.map_state().as_str());
    let second = rb.map_async(MapMode::READ, sz(0), None).await;
    r.check("mapAsync while pending", outcome(&second));
    r.check("mapAsync first", outcome(&first.await));
    let mapped = must!(r, "mapped", rb.get_mapped_range(sz(0), None));
    r.check("mapped read buffer byteLength", mapped.byte_length());
    rb.unmap();
    let pending = rb.map_async(MapMode::READ, sz(0), None);
    rb.unmap();
    r.check("mapAsync then unmap", outcome(&pending.await));
    r.check("mapAsync(WRITE) on a MAP_READ buffer", outcome(&rb.map_async(MapMode::WRITE, sz(0), None).await));
    r.check("mapAsync offset not a multiple of 8", outcome(&rb.map_async(MapMode::READ, sz(4), None).await));

    // writeBuffer.
    let wb = must!(
        r,
        "buffer",
        device.create_buffer(&BufferDescriptor::new(sz(16), BufferUsage::COPY_DST | BufferUsage::COPY_SRC))
    );
    let data: Vec<u8> = (0..20).collect();
    r.check("writeBuffer(dataOffset past end)", outcome(&q.write_buffer(&wb, sz(0), &data, sz(24), None)));
    r.check("writeBuffer(size not multiple of 4)", outcome(&q.write_buffer(&wb, sz(0), &data, sz(0), Some(sz(6)))));
    r.check("writeBuffer(valid)", outcome(&q.write_buffer(&wb, sz(4), &data, sz(4), Some(sz(8)))));
    match read_buffer::<A>(&device, &wb, 16).await {
        Ok(v) => r.check("writeBuffer contents", hex(&v)),
        Err(e) => r.check("writeBuffer contents", format!("failed: {e}")),
    }

    // Compute.
    let module = device.create_shader_module(&ShaderModuleDescriptor::new(COMPUTE_WGSL.into()));
    let mut cdesc = ComputePipelineDescriptor::new(
        PipelineLayoutOrAuto::AutoLayoutMode(AutoLayoutMode::Auto),
        ProgrammableStage::new(&module),
    );
    cdesc.compute.entry_point = Some("main".into());
    let pipeline = device.create_compute_pipeline(&cdesc);
    let n = 256u32;
    let input: Vec<u8> = (0..n).flat_map(|i| (i * 3 + 1).to_le_bytes()).collect();
    let storage = must!(
        r,
        "storage",
        device.create_buffer(&BufferDescriptor::new(
            sz(input.len() as u64),
            BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST
        ))
    );
    must!(r, "upload", q.write_buffer(&storage, sz(0), &input, sz(0), None));
    let bgl = pipeline.get_bind_group_layout(0);
    r.check("getBindGroupLayout(0).label", format!("{:?}", bgl.label()));
    let bg = device.create_bind_group(&BindGroupDescriptor::new(
        &bgl,
        vec![BindGroupEntry::new(0, BindingResource::Buffer(&storage))],
    ));
    let enc = device.create_command_encoder(&CommandEncoderDescriptor::new());
    let pass = must!(r, "compute pass", enc.begin_compute_pass(&ComputePassDescriptor::new()));
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, Some(&bg), &[]);
    r.check(
        "setBindGroup(data range past end)",
        outcome(&pass.set_bind_group_with_data(0, Some(&bg), &[0, 0], sz(1), 2)),
    );
    r.check("setImmediates(size not multiple of 4)", outcome(&pass.set_immediates(0, &[0; 8], sz(0), Some(sz(6)))));
    pass.dispatch_workgroups(n / 64, 1, 1);
    pass.end();
    q.submit(&[&enc.finish(&CommandBufferDescriptor::new())]);
    match read_buffer::<A>(&device, &storage, input.len() as u64).await {
        Ok(v) => {
            let expect: Vec<u8> = (0..n).flat_map(|i| ((i * 3 + 1) * 2 + i).to_le_bytes()).collect();
            r.check(
                "compute result",
                if v == expect { "exact".to_string() } else { format!("differs: {}", hex(&v[..32])) },
            );
        }
        Err(e) => r.check("compute result", format!("failed: {e}")),
    }
    r.check("createComputePipelineAsync(valid)", outcome(&device.create_compute_pipeline_async(&cdesc).await));
    let mut bad = cdesc.clone();
    bad.compute.entry_point = Some("no_such_entry_point".into());
    r.check("createComputePipelineAsync(bad entry point)", outcome(&device.create_compute_pipeline_async(&bad).await));
    let broken = device.create_shader_module(&ShaderModuleDescriptor::new(BROKEN_WGSL.into()));
    match broken.get_compilation_info().await {
        Ok(info) => {
            let errors = info.messages.iter().filter(|m| m.r#type == CompilationMessageType::Error).count();
            r.check("getCompilationInfo(broken) has errors", errors > 0);
            if let Some(m) = info.messages.first() {
                r.info(
                    "getCompilationInfo(broken) first",
                    format!("{}:{} +{} len {} {}", m.line_num, m.line_pos, m.offset, m.length, m.message),
                );
            }
        }
        Err(e) => r.check("getCompilationInfo(broken) has errors", kind(&e)),
    }

    // Textures.
    let mut tex_desc = TextureDescriptor::new(
        ext(4, 4),
        TextureFormat::Rgba8unorm,
        TextureUsage::COPY_DST
            | TextureUsage::COPY_SRC
            | TextureUsage::TEXTURE_BINDING
            | TextureUsage::RENDER_ATTACHMENT,
    );
    tex_desc.label = "tex".into();
    let tex = must!(r, "texture", device.create_texture(&tex_desc));
    r.check(
        "texture attributes",
        format!(
            "{}x{}x{} mips {} samples {} {} {} usage {} textureBindingViewDimension {:?}",
            tex.width(),
            tex.height(),
            tex.depth_or_array_layers(),
            tex.mip_level_count(),
            tex.sample_count(),
            tex.dimension().as_str(),
            tex.format().as_str(),
            tex.usage(),
            tex.texture_binding_view_dimension().map(|d| d.as_str())
        ),
    );
    let mut bad_tex = tex_desc.clone();
    bad_tex.size = Extent3D::Sequence(vec![4, 4, 1, 1]);
    r.check("createTexture(extent sequence of 4)", outcome(&device.create_texture(&bad_tex)));
    let mut sw = TextureViewDescriptor::new();
    sw.swizzle = "rgbx".into();
    r.check("createView(swizzle rgbx)", outcome(&tex.create_view(&sw)));
    let pixels: Vec<u8> = (0..64).collect();
    let mut layout = TexelCopyBufferLayout::new();
    layout.bytes_per_row = Some(16);
    let dst = TexelCopyTextureInfo::new(&tex);
    r.check("writeTexture", outcome(&q.write_texture(&dst, &pixels, &layout, &ext(4, 4))));
    r.check(
        "writeTexture(extent sequence of 0)",
        outcome(&q.write_texture(&dst, &pixels, &layout, &Extent3D::Sequence(vec![]))),
    );
    let readback = must!(
        r,
        "buffer",
        device.create_buffer(&BufferDescriptor::new(sz(256 * 4), BufferUsage::COPY_DST | BufferUsage::COPY_SRC))
    );
    let enc = device.create_command_encoder(&CommandEncoderDescriptor::new());
    let mut rbinfo = TexelCopyBufferInfo::new(&readback);
    rbinfo.bytes_per_row = Some(256);
    r.check(
        "copyTextureToBuffer",
        outcome(&enc.copy_texture_to_buffer(&TexelCopyTextureInfo::new(&tex), &rbinfo, &ext(4, 4))),
    );
    q.submit(&[&enc.finish(&CommandBufferDescriptor::new())]);
    match read_buffer::<A>(&device, &readback, 256 * 4).await {
        Ok(v) => {
            let rows: Vec<u8> = (0..4).flat_map(|row| v[row * 256..row * 256 + 16].to_vec()).collect();
            r.check("texture round trip", if rows == pixels { "exact".to_string() } else { hex(&rows) });
        }
        Err(e) => r.check("texture round trip", format!("failed: {e}")),
    }

    // Rendering: clear to blue and draw magenta into the left two columns through a scissor; draw the same through a
    // render bundle into a second target cleared to green. Every value is exact in rgba8unorm.
    let rmod = device.create_shader_module(&ShaderModuleDescriptor::new(RENDER_WGSL.into()));
    let rdesc = render_descriptor::<A>(&rmod, None);
    let rpipe = device.create_render_pipeline(&rdesc);
    r.check("createRenderPipeline", outcome(&rpipe));
    r.check("createRenderPipelineAsync", outcome(&device.create_render_pipeline_async(&rdesc).await));
    if let Some(pd) = &plain_device {
        // The plain device lacks dual-source-blending, so the src1 blend factor is not an enabled enum value.
        let pmod = pd.create_shader_module(&ShaderModuleDescriptor::new(RENDER_WGSL.into()));
        let src1 = BlendState::new(
            BlendComponent {
                operation: BlendOperation::Add,
                src_factor: BlendFactor::Src1,
                dst_factor: BlendFactor::Zero,
            },
            BlendComponent::new(),
        );
        let d = render_descriptor::<A>(&pmod, Some(src1));
        r.check("createRenderPipeline(src1 without dual-source-blending)", outcome(&pd.create_render_pipeline(&d)));
        r.check(
            "createRenderPipelineAsync(src1 without dual-source-blending)",
            outcome(&pd.create_render_pipeline_async(&d).await),
        );
    }
    if let Ok(rpipe) = &rpipe {
        let target = must!(r, "target", device.create_texture(&tex_desc));
        let enc = device.create_command_encoder(&CommandEncoderDescriptor::new());
        let mut ca = RenderPassColorAttachment::new(TextureOrView::Texture(&target), LoadOp::Clear, StoreOp::Store);
        ca.clear_value = Some(Color::Sequence(vec![f64v(0.0), f64v(0.0), f64v(1.0), f64v(1.0)]));
        let mut bad_pass = RenderPassDescriptor::new(vec![Some(ca.clone())]);
        if let Some(Some(c)) = bad_pass.color_attachments.first_mut() {
            c.clear_value = Some(Color::Sequence(vec![f64v(0.0); 3]));
        }
        r.check("beginRenderPass(clearValue of 3)", outcome(&enc.begin_render_pass(&bad_pass)));
        let pass = must!(r, "render pass", enc.begin_render_pass(&RenderPassDescriptor::new(vec![Some(ca)])));
        pass.set_pipeline(rpipe);
        pass.set_scissor_rect(0, 0, 2, 4);
        r.check(
            "setBlendConstant(sequence of 3)",
            outcome(&pass.set_blend_constant(&Color::Sequence(vec![f64v(0.0); 3]))),
        );
        r.check("setBlendConstant(dict)", outcome(&pass.set_blend_constant(&rgba(0.0, 0.0, 0.0, 0.0))));
        pass.set_viewport(f32v(0.0), f32v(0.0), f32v(4.0), f32v(4.0), f32v(0.0), f32v(1.0));
        pass.draw(3, 1, 0, 0);
        pass.end();
        let target2 = must!(r, "target", device.create_texture(&tex_desc));
        let be = must!(
            r,
            "bundle encoder",
            device.create_render_bundle_encoder(&RenderBundleEncoderDescriptor::new(vec![Some(
                TextureFormat::Rgba8unorm
            )]))
        );
        be.set_pipeline(rpipe);
        be.draw(3, 1, 0, 0);
        let bundle = be.finish(&RenderBundleDescriptor::new());
        let mut ca2 = RenderPassColorAttachment::new(TextureOrView::Texture(&target2), LoadOp::Clear, StoreOp::Store);
        ca2.clear_value = Some(rgba(0.0, 1.0, 0.0, 1.0));
        let pass2 = must!(r, "render pass", enc.begin_render_pass(&RenderPassDescriptor::new(vec![Some(ca2)])));
        pass2.execute_bundles(&[&bundle]);
        pass2.end();
        let mut info1 = TexelCopyBufferInfo::new(&readback);
        info1.bytes_per_row = Some(256);
        must!(r, "copy", enc.copy_texture_to_buffer(&TexelCopyTextureInfo::new(&target), &info1, &ext(4, 4)));
        let readback2 = must!(
            r,
            "buffer",
            device.create_buffer(&BufferDescriptor::new(sz(256 * 4), BufferUsage::COPY_DST | BufferUsage::COPY_SRC))
        );
        let mut info2 = TexelCopyBufferInfo::new(&readback2);
        info2.bytes_per_row = Some(256);
        must!(r, "copy", enc.copy_texture_to_buffer(&TexelCopyTextureInfo::new(&target2), &info2, &ext(4, 4)));
        q.submit(&[&enc.finish(&CommandBufferDescriptor::new())]);
        for (name, buf) in [("render pass with scissor", &readback), ("render bundle", &readback2)] {
            match read_buffer::<A>(&device, buf, 256 * 4).await {
                Ok(v) => {
                    let rows: Vec<String> = (0..4).map(|row| hex(&v[row * 256..row * 256 + 16])).collect();
                    r.check(name, rows.join(" "));
                }
                Err(e) => r.check(name, format!("failed: {e}")),
            }
        }
    }

    // Features enforced on the content timeline of a device that lacks them.
    if let Some(pd) = &plain_device {
        let mut bc = tex_desc.clone();
        bc.format = TextureFormat::Bc1RgbaUnorm;
        bc.usage = TextureUsage::TEXTURE_BINDING;
        r.check("createTexture(bc1 without texture-compression-bc)", outcome(&pd.create_texture(&bc)));
        r.check(
            "createQuerySet(timestamp without timestamp-query)",
            outcome(&pd.create_query_set(&QuerySetDescriptor::new(QueryType::Timestamp, 2))),
        );
        match pd.create_query_set(&QuerySetDescriptor::new(QueryType::Occlusion, 2)) {
            Ok(qs) => {
                r.check("createQuerySet(occlusion): type, count", format!("{}, {}", qs.r#type().as_str(), qs.count()))
            }
            Err(e) => r.check("createQuerySet(occlusion): type, count", kind(&e)),
        }
    }

    q.on_submitted_work_done().await;
    r.check("onSubmittedWorkDone", "resolved");
    let rest = scopes.close().await;
    r.check(
        "plumbing: errors the remaining steps raised",
        rest.iter()
            .map(|c| match c {
                Captured::Error { error, .. } => error.filter.as_str().to_string(),
                other => format!("unexpected {other}"),
            })
            .collect::<Vec<_>>()
            .join(","),
    );

    // Device destruction: the device's buffers are unmapped and the lost promise resolves with "destroyed".
    if let Some(pd) = plain_device {
        let mut d = BufferDescriptor::new(sz(16), BufferUsage::MAP_WRITE);
        d.mapped_at_creation = true;
        let b = must!(r, "buffer", pd.create_buffer(&d));
        let lost = pd.lost();
        pd.destroy();
        r.check("destroy: buffer mapState", b.map_state().as_str());
        let info = lost.await;
        r.check("destroy: lost.reason", info.reason.as_str());
        r.check("destroy: mapAsync after", outcome(&b.map_async(MapMode::WRITE, sz(0), None).await));
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_lists_differing_checks_only() {
        let a = "info implementation = x\ncheck a = 1\ncheck b = 2\n";
        let b = "info implementation = y\ncheck a = 1\ncheck b = 3\ncheck c = 4\n";
        assert_eq!(compare(a, b), vec!["b: 2 | 3".to_string(), "c: <absent> | 4".to_string()]);
    }

    #[test]
    fn conforms_accepts_alternatives() {
        let e = "check a = 1 || 2\ncheck b = x\n";
        assert!(conforms(e, "check a = 2\ncheck b = x\n").is_empty());
        assert_eq!(conforms(e, "check a = 3\ncheck c = y\n").len(), 3);
    }
}
