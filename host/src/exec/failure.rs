//! Failures the executor reports, and how host-side causes map onto Metal's results.
//!
//! A [`Failure`] travels in a control reply or in a CommandBufferCompleted event. Its status says who defines it
//! (`webmetal_wire::status`): 1 is an `NSError` Metal itself returns; 2 and 3 are reported by the guest through the
//! not-implemented channel; 4 is a protocol violation. This module holds the Metal values the mapping uses and the
//! mapping itself:
//!
//! - WebGPU device loss: status 2 with refusal `device-removed` (`tables/refusals.md`). iOS defines no command-buffer
//!   error code for a lost GPU (`tables/limits.toml`, question `6.1-device-removed-code`), so none is chosen until one
//!   is measured.
//! - A command buffer the kernel would fail with one of its command-buffer statuses (`kIOGPUCommandBufferCallbackError*`,
//!   such as a wait that times out): `IOGPUCommandQueueErrorDomain` with the status as the code ([`kernel`]). The
//!   guest makes of it the `NSError` IOGPU makes (`-[NSError(IOGPUMetalErrors) initWithIOGPUError:]`), in
//!   `MTLCommandBufferErrorDomain` with this one as its underlying error.
//! - A `GPUOutOfMemoryError` while a command buffer executes: the kernel status
//!   `kIOGPUCommandBufferCallbackErrorOutOfMemory` (8), of which IOGPU makes `MTLCommandBufferErrorOutOfMemory` (8,
//!   "Insufficient Memory"; it is the only status IOGPU maps to that code); `MTLCommandBuffer.h`: "Insufficient memory
//!   was available to execute this command buffer."
//! - A `GPUValidationError`, a `GPUInternalError` or an error that escaped every scope: status 2. The executor checks
//!   Metal's rules and WebGPU's before it calls WebGPU, so such an error is an executor defect; it is reported, never
//!   passed off as a Metal result.
//!
//! A control request that hits the same causes fails the same way, except that out-of-memory while creating an object
//! is status 2: Metal's result there is a nil object without an `NSError`, and the wire protocol has no encoding for a
//! nil result yet.

use core::fmt;

use webmetal_wire::{Failure, status};

use crate::gpu::idl::ErrorFilter;
use crate::gpu::scope::Captured;
use crate::gpu::{DeviceLostInfo, GpuError};

/// `MTLCommandBufferErrorDomain`: the value of the exported `NSErrorDomain` constant (read on macOS 27.0.1, where the
/// constant's string is its own name).
pub const COMMAND_BUFFER_ERROR_DOMAIN: &str = "MTLCommandBufferErrorDomain";

/// `IOGPUCommandQueueErrorDomain`: the domain IOGPU gives the kernel's command-buffer status (21H461's IOGPU,
/// `-[NSError(IOGPUMetalErrors) initWithIOGPUError:]`).
pub const IOGPU_COMMAND_QUEUE_ERROR_DOMAIN: &str = "IOGPUCommandQueueErrorDomain";

/// The kernel's command-buffer statuses (IOGPUFamily's `kIOGPUCommandBufferCallbackError*`, as IOGPU's error tables
/// name them).
pub mod kernel_status {
    /// `kIOGPUCommandBufferCallbackErrorTimeout`.
    pub const TIMEOUT: i64 = 2;
    /// `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
    pub const OUT_OF_MEMORY: i64 = 8;
}

/// `MTLCommandBufferStatus` (iPhoneOS SDK, `MTLCommandBuffer.h`).
pub mod command_buffer_status {
    pub const NOT_ENQUEUED: u64 = 0;
    pub const ENQUEUED: u64 = 1;
    pub const COMMITTED: u64 = 2;
    pub const SCHEDULED: u64 = 3;
    pub const COMPLETED: u64 = 4;
    pub const ERROR: u64 = 5;
}

/// `MTLCommandBufferError` codes available on iOS (iPhoneOS SDK, `MTLCommandBuffer.h`). `DeviceRemoved` (11) is
/// `API_UNAVAILABLE(ios)` and is not listed.
pub mod command_buffer_error {
    pub const NONE: i64 = 0;
    pub const INTERNAL: i64 = 1;
    pub const TIMEOUT: i64 = 2;
    pub const PAGE_FAULT: i64 = 3;
    pub const ACCESS_REVOKED: i64 = 4;
    pub const NOT_PERMITTED: i64 = 7;
    pub const OUT_OF_MEMORY: i64 = 8;
    pub const INVALID_RESOURCE: i64 = 9;
    pub const MEMORYLESS: i64 = 10;
    pub const STACK_OVERFLOW: i64 = 12;
}

/// `failed_offset` of a CommandBufferCompleted event when no single record failed.
pub const NO_RECORD: u64 = u64::MAX;

/// The success value of a [`Failure`].
pub fn ok() -> Failure {
    Failure { status: status::OK, domain: None, code: 0, message: String::new() }
}

/// An `NSError` Metal returns.
pub fn metal(domain: &str, code: i64, message: impl Into<String>) -> Failure {
    Failure { status: status::METAL_ERROR, domain: Some(domain.to_string()), code, message: message.into() }
}

/// A command buffer the kernel fails with its command-buffer status `status`.
pub fn kernel(status: i64, message: impl Into<String>) -> Failure {
    metal(IOGPU_COMMAND_QUEUE_ERROR_DOMAIN, status, message)
}

/// Not implemented by the host; the guest reports it through the not-implemented channel.
pub fn unimplemented(message: impl Into<String>) -> Failure {
    Failure { status: status::UNIMPLEMENTED, domain: None, code: 0, message: message.into() }
}

/// The host cannot honor a capability the device advertises.
pub fn host_unsupported(message: impl Into<String>) -> Failure {
    Failure { status: status::HOST_UNSUPPORTED, domain: None, code: 0, message: message.into() }
}

/// A malformed request, an unknown handle or another protocol violation.
pub fn invalid(message: impl Into<String>) -> Failure {
    Failure { status: status::INVALID, domain: None, code: 0, message: message.into() }
}

/// Metal leaves the use undefined and WebGPU rejects it. Reported through the not-implemented
/// channel with the Metal call and the WebGPU rule; the command buffer completes with an error.
pub fn undefined_in_metal(metal_call: &str, webgpu_rule: &str) -> Failure {
    unimplemented(format!("{metal_call}: undefined in Metal and rejected by WebGPU ({webgpu_rule})"))
}

/// The record's Metal methods, for messages: `-[MTLBlitCommandEncoder fillBuffer:range:value:]`, or the record's name
/// for a plugin record.
pub fn record_methods(opcode: u16) -> String {
    match webmetal_wire::op_info(opcode) {
        Some(op) if !op.methods.is_empty() => {
            op.methods.iter().map(|m| format!("-[{m}]")).collect::<Vec<_>>().join(", ")
        }
        Some(op) => op.name.to_string(),
        None => format!("opcode {opcode:#06x}"),
    }
}

/// A record the host does not implement.
pub fn unimplemented_record(opcode: u16) -> Failure {
    let name = webmetal_wire::op_info(opcode).map_or("?", |op| op.name);
    unimplemented(format!("{name} ({}) is not implemented by the host", record_methods(opcode)))
}

/// What went wrong on the WebGPU side of a command buffer or request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuCause {
    /// The device was lost.
    DeviceLost(DeviceLostInfo),
    /// A `GPUOutOfMemoryError`.
    OutOfMemory(String),
    /// A `GPUValidationError` or `GPUInternalError` an error scope captured, or an error that escaped every scope:
    /// an executor defect.
    Defect(String),
}

impl fmt::Display for GpuCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuCause::DeviceLost(info) => write!(f, "device lost ({}): {}", info.reason.as_str(), info.message),
            GpuCause::OutOfMemory(m) => write!(f, "out of memory: {m}"),
            GpuCause::Defect(m) => write!(f, "{m}"),
        }
    }
}

impl GpuCause {
    /// The cause of one captured error.
    pub fn of(captured: &Captured) -> GpuCause {
        match captured {
            Captured::DeviceLost(info) => GpuCause::DeviceLost(info.clone()),
            Captured::Error { error: GpuError { filter: ErrorFilter::OutOfMemory, message }, at } => {
                GpuCause::OutOfMemory(format!("{message} (from {at})"))
            }
            other => GpuCause::Defect(format!("WebGPU reported an error the executor should have prevented: {other}")),
        }
    }

    /// The most significant cause among several: device loss, then a defect, then out-of-memory. A defect outranks
    /// out-of-memory because it means the executor's own state can no longer be trusted.
    pub fn worst(captured: &[Captured]) -> Option<GpuCause> {
        let causes: Vec<GpuCause> = captured.iter().map(GpuCause::of).collect();
        let rank = |c: &GpuCause| match c {
            GpuCause::DeviceLost(_) => 0,
            GpuCause::Defect(_) => 1,
            GpuCause::OutOfMemory(_) => 2,
        };
        causes.into_iter().min_by_key(rank)
    }

    /// The failure a command buffer completes with.
    pub fn command_buffer_failure(&self) -> Failure {
        match self {
            GpuCause::DeviceLost(info) => device_removed(info),
            GpuCause::OutOfMemory(m) => kernel(kernel_status::OUT_OF_MEMORY, format!("out of memory: {m}")),
            GpuCause::Defect(m) => unimplemented(m.clone()),
        }
    }

    /// The failure a control request that creates an object replies with.
    pub fn request_failure(&self) -> Failure {
        match self {
            GpuCause::DeviceLost(info) => device_removed(info),
            GpuCause::OutOfMemory(m) => unimplemented(format!(
                "out of memory: {m}; Metal returns nil without an NSError, which the wire protocol cannot encode yet"
            )),
            GpuCause::Defect(m) => unimplemented(m.clone()),
        }
    }
}

/// The failure of every command buffer in flight or committed after the WebGPU device was lost, and of the DeviceLost
/// event: refusal `device-removed` of `tables/refusals.md`.
pub fn device_removed(info: &DeviceLostInfo) -> Failure {
    unimplemented(format!(
        "device-removed: the WebGPU device was lost ({}: {}); iOS defines no command-buffer error code for a lost GPU \
         (MTLCommandBufferErrorDeviceRemoved is API_UNAVAILABLE(ios)), see tables/refusals.md",
        info.reason.as_str(),
        info.message
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::idl::DeviceLostReason;
    use crate::gpu::scope::Attribution;

    fn wire_valid(f: &Failure) -> bool {
        // The rules webmetal_wire enforces when it encodes a failure.
        match f.status {
            status::OK => f.domain.is_none() && f.code == 0 && f.message.is_empty(),
            status::METAL_ERROR => f.domain.is_some(),
            status::UNIMPLEMENTED | status::HOST_UNSUPPORTED | status::INVALID => f.domain.is_none() && f.code == 0,
            _ => false,
        }
    }

    #[test]
    fn constructors_obey_the_failure_rules() {
        for f in [ok(), metal("D", 3, "m"), unimplemented("u"), host_unsupported("h"), invalid("i")] {
            assert!(wire_valid(&f), "{f:?}");
        }
    }

    #[test]
    fn out_of_memory_maps_to_the_kernel_status() {
        let c = Captured::Error {
            error: GpuError { filter: ErrorFilter::OutOfMemory, message: "no".into() },
            at: Attribution::Batch(Vec::new()),
        };
        let f = GpuCause::of(&c).command_buffer_failure();
        assert_eq!(f.status, status::METAL_ERROR);
        assert_eq!(f.domain.as_deref(), Some("IOGPUCommandQueueErrorDomain"));
        assert_eq!(f.code, 8);
        assert_eq!(GpuCause::of(&c).request_failure().status, status::UNIMPLEMENTED);
    }

    #[test]
    fn device_loss_outranks_the_rest() {
        let oom = Captured::Error {
            error: GpuError { filter: ErrorFilter::OutOfMemory, message: "a".into() },
            at: Attribution::Batch(Vec::new()),
        };
        let validation = Captured::Error {
            error: GpuError { filter: ErrorFilter::Validation, message: "b".into() },
            at: Attribution::Batch(Vec::new()),
        };
        let lost = Captured::DeviceLost(DeviceLostInfo { reason: DeviceLostReason::Destroyed, message: "c".into() });
        assert!(matches!(GpuCause::worst(&[oom.clone(), validation.clone()]), Some(GpuCause::Defect(_))));
        assert!(matches!(GpuCause::worst(&[oom.clone(), lost.clone(), validation]), Some(GpuCause::DeviceLost(_))));
        assert!(matches!(GpuCause::worst(&[oom]), Some(GpuCause::OutOfMemory(_))));
        let f = GpuCause::of(&lost).command_buffer_failure();
        assert_eq!(f.status, status::UNIMPLEMENTED);
        assert!(f.message.starts_with("device-removed:"));
    }
}
