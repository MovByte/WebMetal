//! Error-scope plumbing: every WebGPU error is captured and attributed.
//!
//! A device-timeline error goes to the innermost error scope whose filter matches it, or, when none does, becomes an
//! `uncapturederror` event that the specification allows a user agent to throttle or drop (§22.4). Relying on that
//! event would lose errors, so [`ErrorScopes`] keeps one frame of three scopes (validation, out-of-memory, internal)
//! open on the device at all times. Every error therefore lands in a scope, and every scope is popped with the calls
//! it covered recorded:
//!
//! - [`Granularity::PerCall`] (test builds) wraps each call in its own frame, so an error names the call that raised it.
//!   A frame reports one error per filter, which is what one call can raise.
//! - [`Granularity::Batch`] wraps only the standing frame; [`ErrorScopes::settle`] rotates it, so an error names the
//!   calls since the previous settle. A scope reports one of the errors it observed (§22.3, `popErrorScope`).
//!
//! Errors reach the scopes in device-timeline order, the same order as the pushes and pops, so rotating the frame
//! between calls cannot move an error to the wrong frame. The `uncapturederror` handler is installed as well: an error
//! that reaches it escaped every scope, which is a defect of the executor or of the implementation, and is reported as
//! [`Captured::Uncaptured`]; under [`Policy::strict`] [`ErrorScopes::settle`] panics on it. Device loss stops error
//! reporting (§22), so it is reported too.
//!
//! Exceptions and promise rejections need no plumbing: they come back to the caller as `Err` values of the call.

use core::task::{Context, Poll, Waker};
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use super::api::{Api, Device, DeviceLostInfo, Exception, GpuError, GpuFuture};
use super::idl::ErrorFilter;

/// One call made under the plumbing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallSite {
    /// Position in the sequence of calls made through this plumbing.
    pub seq: u64,
    /// The IDL operation, as `Interface.member`.
    pub op: &'static str,
    /// The label of the object the call made or acted on.
    pub label: String,
}

impl fmt::Display for CallSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{} {}", self.seq, self.op)?;
        if !self.label.is_empty() {
            write!(f, " \"{}\"", self.label)?;
        }
        Ok(())
    }
}

/// Which calls an error belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Attribution {
    /// Raised by exactly this call.
    Call(CallSite),
    /// Raised by one of these calls (the standing frame between two settles).
    Batch(Vec<CallSite>),
}

impl fmt::Display for Attribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attribution::Call(c) => write!(f, "{c}"),
            Attribution::Batch(calls) => {
                write!(f, "one of [")?;
                for (i, c) in calls.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{c}")?;
                }
                write!(f, "]")
            }
        }
    }
}

/// An error the plumbing observed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Captured {
    /// A device-timeline error, captured by a scope.
    Error { error: GpuError, at: Attribution },
    /// An error that reached `uncapturederror`: it escaped every scope. `after` is the last call made before it was
    /// delivered.
    Uncaptured { error: GpuError, after: Option<CallSite> },
    /// `popErrorScope()` rejected: the plumbing's scope stack was disturbed by a pop it did not make.
    PopRejected { exception: Exception, at: Attribution },
    /// The device was lost; errors after this point are not reported (§22).
    DeviceLost(DeviceLostInfo),
}

impl fmt::Display for Captured {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Captured::Error { error, at } => write!(f, "{error} (from {at})"),
            Captured::Uncaptured { error, after } => match after {
                Some(c) => write!(f, "uncaptured {error} (delivered after {c})"),
                None => write!(f, "uncaptured {error}"),
            },
            Captured::PopRejected { exception, at } => {
                write!(f, "popErrorScope rejected with {exception} (frame of {at})")
            }
            Captured::DeviceLost(info) => write!(f, "device lost ({}): {}", info.reason.as_str(), info.message),
        }
    }
}

/// How finely errors are attributed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Granularity {
    PerCall,
    Batch,
}

/// The plumbing's configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    pub granularity: Granularity,
    /// Panic in [`ErrorScopes::settle`] when an error escaped every scope.
    pub strict: bool,
}

/// The three filters of a frame, in push order; pops go in reverse.
const FRAME: [ErrorFilter; 3] = [ErrorFilter::Internal, ErrorFilter::OutOfMemory, ErrorFilter::Validation];

struct Pending {
    at: Attribution,
    pop: GpuFuture<Result<Option<GpuError>, Exception>>,
}

#[derive(Default)]
struct State {
    seq: u64,
    last: Option<CallSite>,
    batch: Vec<CallSite>,
    pending: Vec<Pending>,
    uncaptured: Vec<(GpuError, Option<CallSite>)>,
    /// The device's `lost` promise until it resolves.
    lost: Option<GpuFuture<DeviceLostInfo>>,
}

/// The error-scope plumbing of one device.
pub struct ErrorScopes<A: Api> {
    device: A::Device,
    policy: Policy,
    state: Rc<RefCell<State>>,
    open: bool,
}

impl<A: Api> ErrorScopes<A> {
    /// Install the plumbing: the `uncapturederror` handler, the device-loss watch and the standing frame. The device
    /// must have no scopes of the caller's open across calls made through the plumbing.
    pub fn new(device: &A::Device, policy: Policy) -> ErrorScopes<A> {
        let state = Rc::new(RefCell::new(State::default()));
        let s = state.clone();
        device.set_onuncapturederror(Some(Box::new(move |error| {
            let mut st = s.borrow_mut();
            let after = st.last.clone();
            st.uncaptured.push((error, after));
        })));
        state.borrow_mut().lost = Some(device.lost());
        let scopes = ErrorScopes { device: device.clone(), policy, state, open: true };
        scopes.push_frame();
        scopes
    }

    pub fn device(&self) -> &A::Device {
        &self.device
    }

    fn push_frame(&self) {
        for f in FRAME {
            self.device.push_error_scope(f);
        }
    }

    fn pop_frame(&self, at: Attribution) {
        let mut st = self.state.borrow_mut();
        for _ in FRAME {
            let pop = self.device.pop_error_scope();
            st.pending.push(Pending { at: at.clone(), pop });
        }
    }

    fn site(&self, op: &'static str, label: &str) -> CallSite {
        let mut st = self.state.borrow_mut();
        st.seq += 1;
        let c = CallSite { seq: st.seq, op, label: label.to_string() };
        st.last = Some(c.clone());
        c
    }

    /// Make a call that may raise a device-timeline error. `op` names the IDL operation (`"GPUDevice.createBuffer"`)
    /// and `label` the object's label.
    pub fn call<T>(&self, op: &'static str, label: &str, f: impl FnOnce(&A::Device) -> T) -> T {
        assert!(self.open, "ErrorScopes used after close");
        let site = self.site(op, label);
        match self.policy.granularity {
            Granularity::PerCall => {
                self.push_frame();
                let r = f(&self.device);
                self.pop_frame(Attribution::Call(site));
                r
            }
            Granularity::Batch => {
                self.state.borrow_mut().batch.push(site);
                f(&self.device)
            }
        }
    }

    /// Wait for every scope popped so far and return what they captured, with any uncaptured error and device loss.
    /// In batch mode this first rotates the standing frame. Returns an empty list when nothing went wrong.
    pub fn settle(&self) -> GpuFuture<Vec<Captured>> {
        assert!(self.open, "ErrorScopes used after close");
        if self.policy.granularity == Granularity::Batch {
            let calls = std::mem::take(&mut self.state.borrow_mut().batch);
            self.pop_frame(Attribution::Batch(calls));
            self.push_frame();
        }
        self.drain()
    }

    /// Pop the standing frame, wait for every scope, and remove the `uncapturederror` handler.
    pub fn close(mut self) -> GpuFuture<Vec<Captured>> {
        let calls = std::mem::take(&mut self.state.borrow_mut().batch);
        self.pop_frame(Attribution::Batch(calls));
        self.open = false;
        let fut = self.drain();
        let device = self.device.clone();
        Box::pin(async move {
            let out = fut.await;
            device.set_onuncapturederror(None);
            out
        })
    }

    fn drain(&self) -> GpuFuture<Vec<Captured>> {
        let pending = std::mem::take(&mut self.state.borrow_mut().pending);
        let state = self.state.clone();
        let strict = self.policy.strict;
        Box::pin(async move {
            let mut out = Vec::new();
            for p in pending {
                match p.pop.await {
                    Ok(None) => {}
                    Ok(Some(error)) => out.push(Captured::Error { error, at: p.at }),
                    Err(exception) => out.push(Captured::PopRejected { exception, at: p.at }),
                }
            }
            // The lost promise is checked without waiting: it resolves only if the device is lost. It is taken out of
            // the state while polled, because polling may deliver uncaptured errors to the handler, which records them
            // in the state.
            let lost = state.borrow_mut().lost.take();
            if let Some(mut lost) = lost {
                let mut cx = Context::from_waker(Waker::noop());
                match lost.as_mut().poll(&mut cx) {
                    Poll::Ready(info) => out.push(Captured::DeviceLost(info)),
                    Poll::Pending => state.borrow_mut().lost = Some(lost),
                }
            }
            for (error, after) in state.borrow_mut().uncaptured.drain(..) {
                out.push(Captured::Uncaptured { error, after });
            }
            if strict {
                let escaped: Vec<_> = out
                    .iter()
                    .filter(|c| matches!(c, Captured::Uncaptured { .. } | Captured::PopRejected { .. }))
                    .collect();
                if !escaped.is_empty() {
                    let lines: Vec<String> = escaped.iter().map(|c| c.to_string()).collect();
                    panic!("WebGPU errors escaped the error scopes:\n{}", lines.join("\n"));
                }
            }
            out
        })
    }
}
