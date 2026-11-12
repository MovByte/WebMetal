//! Why a render operation failed, and the wire status each failure is reported with.

use core::fmt;

use webmetal_wire::{Failure, status};

use crate::format::handling::Refusal;
use crate::gpu::Exception;
use crate::mem::MemError;

/// A failed render operation. Every variant names what failed; none is ever turned into a success.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderError {
    /// Metal leaves the use undefined (validation class 2 of plan section 6.10): reported through the not-implemented
    /// channel with the Metal call and the rule it breaks, never given a substitute behavior.
    Undefined { call: String, rule: String },
    /// The executor has no exact implementation of this yet (LOUD).
    Unimplemented { what: String },
    /// The host's WebGPU cannot do this exactly: a missing feature or limit, or a row of `tables/refusals.md`.
    HostUnsupported { what: String },
    /// The stream breaks the protocol (a handle of the wrong kind of object).
    Invalid { what: String },
    /// The memory module refused an operation the pass needs.
    Mem(MemError),
    /// A WebGPU call threw. The executor validates before it calls WebGPU, so this is an executor or implementation
    /// defect.
    Gpu { op: &'static str, exception: Exception },
}

impl RenderError {
    pub fn undefined(call: impl Into<String>, rule: impl Into<String>) -> RenderError {
        RenderError::Undefined { call: call.into(), rule: rule.into() }
    }

    pub fn unimplemented(what: impl Into<String>) -> RenderError {
        RenderError::Unimplemented { what: what.into() }
    }

    pub fn host(what: impl Into<String>) -> RenderError {
        RenderError::HostUnsupported { what: what.into() }
    }

    pub fn invalid(what: impl Into<String>) -> RenderError {
        RenderError::Invalid { what: what.into() }
    }

    /// A row of `tables/refusals.md` or of the format module's refusals.
    pub fn refused(r: Refusal) -> RenderError {
        RenderError::HostUnsupported { what: format!("{}: {} ({})", r.id, r.reason, r.source) }
    }

    pub fn gpu(op: &'static str, exception: Exception) -> RenderError {
        RenderError::Gpu { op, exception }
    }

    /// The wire status (`webmetal_wire::status`) this failure is reported with.
    pub fn status(&self) -> u32 {
        match self {
            RenderError::Undefined { .. } | RenderError::Unimplemented { .. } | RenderError::Gpu { .. } => {
                status::UNIMPLEMENTED
            }
            RenderError::HostUnsupported { .. } => status::HOST_UNSUPPORTED,
            RenderError::Invalid { .. } => status::INVALID,
            RenderError::Mem(e) => e.status(),
        }
    }

    /// The failure a command buffer or a reply carries.
    pub fn failure(&self) -> Failure {
        match self {
            RenderError::Mem(e) => e.failure(),
            other => Failure { status: other.status(), domain: None, code: 0, message: other.to_string() },
        }
    }
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::Undefined { call, rule } => write!(f, "{call}: behavior Metal does not define: {rule}"),
            RenderError::Unimplemented { what } => write!(f, "not implemented: {what}"),
            RenderError::HostUnsupported { what } => write!(f, "host cannot do this exactly: {what}"),
            RenderError::Invalid { what } => write!(f, "invalid stream: {what}"),
            RenderError::Mem(e) => write!(f, "{e}"),
            RenderError::Gpu { op, exception } => write!(f, "{op}: {exception}"),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<MemError> for RenderError {
    fn from(e: MemError) -> RenderError {
        RenderError::Mem(e)
    }
}

/// `Result` of render operations.
pub type Result<T> = core::result::Result<T, RenderError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_follow_the_wire_rules() {
        assert_eq!(RenderError::undefined("-[X y]", "r").failure().status, status::UNIMPLEMENTED);
        assert!(RenderError::undefined("-[X y]", "r").failure().message.contains("behavior Metal does not define"));
        assert_eq!(RenderError::unimplemented("u").failure().status, status::UNIMPLEMENTED);
        assert_eq!(RenderError::host("h").failure().status, status::HOST_UNSUPPORTED);
        assert_eq!(RenderError::invalid("i").failure().status, status::INVALID);
        for e in [RenderError::unimplemented("u"), RenderError::host("h"), RenderError::invalid("i")] {
            let f = e.failure();
            assert!(f.domain.is_none() && f.code == 0, "{f:?}");
        }
    }
}
