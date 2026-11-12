//! Why a memory operation failed, and the wire status each failure is reported with.

use core::fmt;

use webmetal_wire::status;

use crate::embed::{GuestMemoryError, SurfaceError};
use crate::format::handling::Refusal;
use crate::gpu::Exception;

/// A failed memory operation. Every variant names what failed; none is ever turned into a success.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemError {
    /// Metal returns `nil` for this call (for example a heap that is full). The wire's control channel carries the
    /// outcome; the message states the rule.
    Nil { reason: String },
    /// Metal leaves the call's behavior undefined (validation class 2 of plan section 6.10): reported through the
    /// not-implemented channel with the rule it breaks, never given a substitute behavior.
    Undefined { call: &'static str, rule: String },
    /// The executor has no exact implementation of this yet (LOUD).
    Unimplemented { what: String },
    /// The host's WebGPU cannot do this exactly: a missing feature or limit, or a refusal of the format table.
    HostUnsupported { what: String },
    /// The request is malformed or breaks the protocol (an argument outside the object it names).
    Invalid { what: String },
    /// Guest memory could not be read or written.
    Guest(GuestMemoryError),
    /// An IOSurface could not be described, read or written.
    Surface(SurfaceError),
    /// A WebGPU call threw or a promise rejected. The executor validates before it calls WebGPU, so this is an
    /// executor or implementation defect.
    Gpu { op: &'static str, exception: Exception },
}

impl MemError {
    pub fn undefined(call: &'static str, rule: impl Into<String>) -> MemError {
        MemError::Undefined { call, rule: rule.into() }
    }

    pub fn unimplemented(what: impl Into<String>) -> MemError {
        MemError::Unimplemented { what: what.into() }
    }

    pub fn host(what: impl Into<String>) -> MemError {
        MemError::HostUnsupported { what: what.into() }
    }

    pub fn invalid(what: impl Into<String>) -> MemError {
        MemError::Invalid { what: what.into() }
    }

    pub fn refused(r: Refusal) -> MemError {
        MemError::HostUnsupported { what: format!("{}: {} ({})", r.id, r.reason, r.source) }
    }

    pub fn gpu(op: &'static str, exception: Exception) -> MemError {
        MemError::Gpu { op, exception }
    }

    /// The wire status (`webmetal_wire::status`) a reply or a command-buffer completion reports this failure with.
    /// [`MemError::Nil`] has no NSError of its own: the caller completes the request with the `nil` result the
    /// request's Metal method returns.
    pub fn status(&self) -> u32 {
        match self {
            MemError::Nil { .. } => status::METAL_ERROR,
            MemError::Undefined { .. } | MemError::Unimplemented { .. } | MemError::Gpu { .. } => status::UNIMPLEMENTED,
            MemError::HostUnsupported { .. } => status::HOST_UNSUPPORTED,
            MemError::Invalid { .. } | MemError::Guest(_) | MemError::Surface(_) => status::INVALID,
        }
    }
}

impl fmt::Display for MemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemError::Nil { reason } => write!(f, "nil result: {reason}"),
            MemError::Undefined { call, rule } => write!(f, "{call}: behavior Metal does not define: {rule}"),
            MemError::Unimplemented { what } => write!(f, "not implemented: {what}"),
            MemError::HostUnsupported { what } => write!(f, "host cannot do this exactly: {what}"),
            MemError::Invalid { what } => write!(f, "invalid request: {what}"),
            MemError::Guest(e) => write!(f, "{e}"),
            MemError::Surface(e) => write!(f, "{e}"),
            MemError::Gpu { op, exception } => write!(f, "{op}: {exception}"),
        }
    }
}

impl std::error::Error for MemError {}

impl From<GuestMemoryError> for MemError {
    fn from(e: GuestMemoryError) -> MemError {
        MemError::Guest(e)
    }
}

impl From<SurfaceError> for MemError {
    fn from(e: SurfaceError) -> MemError {
        MemError::Surface(e)
    }
}
