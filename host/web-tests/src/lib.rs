//! WebMetal's executor tests in a browser: the test files of `host/tests` that replay the Metal reference cases,
//! built as browser modules (`build.rs`) and run on the browser's WebGPU by `host/tools/web_corpus.py`.
//!
//! The tests are written against a blocking executor: each request, each wait for the GPU and each read of guest
//! memory returns its result. A browser's WebGPU settles its promises from the event loop, which a blocked thread
//! never returns to, so [`BlockingGpu::block_on`] suspends the test's WebAssembly stack instead, through JavaScript
//! Promise Integration (`WebAssembly.Suspending` and `WebAssembly.promising`): the event loop runs while the test
//! waits, and the test resumes where it stopped when its future is woken. The test's code, the executor and the
//! comparison with the reference outputs are the ones the native runs use.
//!
//! What the native harness takes from the process is taken from the page here:
//!
//! - files of the repository (the reference outputs, the reference shader library) are fetched from the page's
//!   server, whose root is the repository's ([`fs`]);
//! - a test's `println!` output goes to the page's log ([`log`]); a panic's message is reported before the module
//!   aborts ([`run_case`]).
//!
//! Each case runs in a module instance of its own, in a Worker of its own (`page/worker.js`): a failed case aborts
//! only its instance, as a panicking test ends only its own thread natively.

#![cfg(all(target_arch = "wasm32", feature = "web"))]

use std::collections::BTreeSet;
use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use wasm_bindgen::prelude::*;
use webmetal_host::gpu::idl::RequestAdapterOptions;
use webmetal_host::gpu::{Exception, Gpu, GpuFuture};

pub use webmetal_host::gpu::web::{Web, WebAdapter, WebDevice, WebGpu};

#[wasm_bindgen(module = "/js/runner.js")]
extern "C" {
    /// A `WebAssembly.Suspending` function: settles when [`wm_wake`] is called (1) or after `ms` milliseconds (0).
    /// Imported as it is, without a JavaScript wrapper (its argument and result need no conversion), as
    /// `WebAssembly.Suspending` requires.
    fn wm_suspend(ms: f64) -> f64;
    /// Settle the pending (or the next) [`wm_suspend`].
    fn wm_wake();
    /// A `WebAssembly.Suspending` function: settles from a task of the event loop (a zero-delay timer), after the
    /// tasks already queued.
    fn wm_yield();
    /// A line of the case's output.
    fn wm_log(line: &str);
    /// The case's panic message, before the module aborts.
    fn wm_panic(message: &str);
    /// The name of case `index` of the module.
    fn wm_case(index: u32, name: &str);
    /// The bytes of the file at `path` (relative to the server's root), or `undefined` when it is not there.
    fn wm_read(path: &str) -> Option<Vec<u8>>;
    /// The names of the files in directory `path`, separated by newlines, or `undefined`.
    fn wm_list(path: &str) -> Option<String>;
}

/// A line of the case's output.
pub fn log(line: &str) {
    wm_log(line);
}

/// Report the name of case `index` of `cases` to the page; the number of cases.
pub fn case_name(cases: &[(&str, fn())], index: u32) -> u32 {
    if let Some((name, _)) = cases.get(index as usize) {
        wm_case(index, name);
    }
    cases.len() as u32
}

/// Run case `index` of `cases`: 1 when it returns. A failing case panics; its message is reported and the module
/// aborts (the module is built with `panic = "abort"`, the only strategy of its target), so the page's call
/// rejects.
pub fn run_case(cases: &[(&str, fn())], index: u32) -> u32 {
    std::panic::set_hook(Box::new(|info| {
        let message = match info.payload().downcast_ref::<&str>() {
            Some(s) => s.to_string(),
            None => info.payload().downcast_ref::<String>().cloned().unwrap_or_default(),
        };
        let at = info.location().map(|l| format!(" at {}:{}", l.file(), l.line())).unwrap_or_default();
        wm_panic(&format!("{message}{at}"));
    }));
    let (_, case) = cases.get(index as usize).unwrap_or_else(|| panic!("no case {index}"));
    case();
    1
}

/// No future made progress for the deadline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stalled {
    pub waited: Duration,
}

impl fmt::Display for Stalled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not woken for {:?}", self.waited)
    }
}

impl std::error::Error for Stalled {}

struct Wakeup;

impl Wake for Wakeup {
    fn wake(self: Arc<Self>) {
        wm_wake();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        wm_wake();
    }
}

/// The browser's `GPU` with the blocking interface the tests use (`NativeGpu`'s, natively).
#[derive(Clone, Debug)]
pub struct BlockingGpu {
    gpu: WebGpu,
}

impl BlockingGpu {
    /// `navigator.gpu` of the Worker. A browser without WebGPU fails the case.
    pub fn new() -> BlockingGpu {
        BlockingGpu { gpu: WebGpu::from_global().unwrap_or_else(|e| panic!("{e}")) }
    }

    /// Run `fut` to completion, suspending the case while it waits so that the event loop settles the browser's
    /// promises. Fails when the future is not woken for `deadline`.
    pub fn block_on<F: Future>(&self, fut: F, deadline: Duration) -> Result<F::Output, Stalled> {
        let waker = Waker::from(Arc::new(Wakeup));
        let mut cx = Context::from_waker(&waker);
        let mut fut = std::pin::pin!(fut);
        loop {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return Ok(v);
            }
            if wm_suspend(deadline.as_secs_f64() * 1000.0) == 0.0 {
                return Err(Stalled { waited: deadline });
            }
        }
    }

    /// Let the event loop run once, so that events the browser has queued (uncaptured errors among them) are
    /// delivered: the browser's counterpart of processing a native instance's events.
    pub fn process_events(&self) {
        wm_yield();
    }
}

impl Default for BlockingGpu {
    fn default() -> BlockingGpu {
        BlockingGpu::new()
    }
}

impl Gpu<Web> for BlockingGpu {
    fn request_adapter(&self, options: &RequestAdapterOptions) -> GpuFuture<Result<Option<WebAdapter>, Exception>> {
        self.gpu.request_adapter(options)
    }

    fn wgsl_language_features(&self) -> Result<BTreeSet<String>, Exception> {
        self.gpu.wgsl_language_features()
    }

    fn implementation(&self) -> String {
        self.gpu.implementation()
    }
}

/// The repository's files, from the page's server (whose root is the repository's root).
pub mod fs {
    use std::io;

    fn missing(path: &str) -> io::Error {
        io::Error::new(io::ErrorKind::NotFound, format!("{path}: not served"))
    }

    pub fn read(path: &str) -> io::Result<Vec<u8>> {
        super::wm_read(path).ok_or_else(|| missing(path))
    }

    pub fn read_to_string(path: &str) -> io::Result<String> {
        String::from_utf8(read(path)?).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{path}: {e}")))
    }

    /// The names of the files in directory `path`.
    pub fn list(path: &str) -> io::Result<Vec<String>> {
        let text = super::wm_list(path).ok_or_else(|| missing(path))?;
        Ok(text.lines().filter(|l| !l.is_empty()).map(str::to_string).collect())
    }
}
