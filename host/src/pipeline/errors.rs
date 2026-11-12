//! How pipeline creation fails, as Metal defines it and as far as the A10X's answer is established.
//!
//! Metal reports a pipeline that cannot be built with a nil pipeline and an `NSError`. The errors of the cases the
//! reference harness measures (`tests/reference`, cases `pipeline.render-errors`, `pipeline.render-async`,
//! `validation.compute-pipeline-from-vertex-function` and `archive.round-trip`) come from the device's driver and its
//! back-end compiler: on Apple M4, macOS 27.0.1 their domain is the driver bundle's name (`AGXMetalG16G_B0`) or
//! `CompilerError`, and the harness compares them only between runs on the same device (rule `device`). Which domain
//! and code the A10X driver (`AGXMetalA10`, iPadOS 17.7.11) returns for each case has not been extracted from the
//! binary or measured on a device, so the host cannot give the guest the `NSError` Metal would: each such case fails
//! with status 2, naming the Metal error it stands for ([`DriverError`]). Status 1 is used only where the domain and
//! code are Metal's own constants (`MTLBinaryArchiveDomain`, `MTLLibraryErrorDomain`) and the case is measured.
//!
//! A compiler [`Refusal`] maps to a failure by its kind ([`refusal`]): a construct the device rejects too is a driver
//! error of the same kind as above; a construct the compiler does not lower yet, or that has no exact WGSL form, is
//! status 2; one that needs a WebGPU feature or limit the host lacks is status 3.

use airwgsl::{Refusal, RefusalKind};
use webmetal_wire::Failure;

use crate::exec::failure;

/// A pipeline-creation error Metal returns from the driver, as measured on the reference device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DriverError {
    /// What Metal rejects.
    pub case: &'static str,
    /// The reference case and the error it recorded.
    pub measured: &'static str,
}

/// A fragment input that the vertex function does not write, or writes with another type.
pub const VARYING_MISMATCH: DriverError = DriverError {
    case: "a fragment input that the vertex function does not write or writes with another type",
    measured: "pipeline.render-errors unmatched-varying: AGXMetalG16G_B0 code 3",
};

/// A vertex function with `[[stage_in]]` attributes and no vertex descriptor.
pub const STAGE_IN_WITHOUT_DESCRIPTOR: DriverError = DriverError {
    case: "a vertex function with input attributes and no vertex descriptor",
    measured: "pipeline.render-errors stage-in-without-descriptor: CompilerError code 1",
};

/// A pipeline whose stage function is not of the stage's type: Metal compiles a vertex function given as a compute
/// function as a vertex-only render pipeline and rejects its non-void return value.
pub const WRONG_FUNCTION_TYPE: DriverError = DriverError {
    case: "a function of another type in the pipeline's stage",
    measured: "validation.compute-pipeline-from-vertex-function (release): AGXMetalG16G_B0 code 3, \"RasterizationEnabled \
               is false but the vertex shader's return type is not void\"",
};

/// A vertex function that returns a value in a pipeline with rasterization disabled.
pub const VERTEX_RESULT_WITHOUT_RASTERIZATION: DriverError = DriverError {
    case: "rasterizationEnabled is NO and the vertex function's return type is not void",
    measured: "validation.compute-pipeline-from-vertex-function (release): AGXMetalG16G_B0 code 3",
};

/// A pipeline missing from the descriptor's binary archives under `MTLPipelineOptionFailOnBinaryArchiveMiss`.
pub const ARCHIVE_MISS: DriverError = DriverError {
    case: "the pipeline is in none of the descriptor's binary archives and MTLPipelineOptionFailOnBinaryArchiveMiss \
           is set",
    measured: "archive.round-trip miss: AGXMetalG16G_B0 code 3",
};

/// A pipeline the compiler refuses as the device would (`RefusalKind::RejectedByDevice`).
pub const COMPILER_REJECTS: DriverError = DriverError {
    case: "the device's compiler rejects the pipeline's functions",
    measured: "no reference case; the error comes from the device's back-end compiler",
};

/// The failure of `call` for a Metal driver error: status 2, naming the error Metal returns.
pub fn driver_error(call: &str, e: DriverError, detail: impl AsRef<str>) -> Failure {
    let detail = detail.as_ref();
    let detail = if detail.is_empty() { String::new() } else { format!(" ({detail})") };
    failure::unimplemented(format!(
        "{call}: Metal fails with an NSError from the driver: {}{detail}; measured on Apple M4, macOS 27.0.1: {}. The \
         error domain and code the A10X driver returns are not established",
        e.case, e.measured
    ))
}

/// The failure of `call` for a compiler refusal.
pub fn refusal(call: &str, r: &Refusal) -> Failure {
    match r.kind {
        RefusalKind::RejectedByDevice => driver_error(call, COMPILER_REJECTS, r.to_string()),
        RefusalKind::NotImplemented | RefusalKind::NoExactLowering => {
            failure::unimplemented(format!("{call}: the shader compiler refuses the pipeline: {r}"))
        }
        RefusalKind::HostUnsupported => {
            failure::host_unsupported(format!("{call}: the pipeline needs what this host lacks: {r}"))
        }
    }
}

/// The failure of `call` for a descriptor field whose value the host does not implement.
pub fn field(call: &str, field: &str, value: impl std::fmt::Display, why: &str) -> Failure {
    failure::unimplemented(format!("{call}: descriptor {field} = {value}: {why}"))
}

/// The failure of `call` for a descriptor the guest's Metal.framework refuses before it reaches the driver: the
/// framework asserts (measured on Apple M4, macOS 27.0.1, with and without the debug layer), so the driver never
/// defines a result.
pub fn framework_asserts(call: &str, what: impl std::fmt::Display) -> Failure {
    failure::unimplemented(format!(
        "{call}: {what}; Metal.framework's descriptor validation asserts on this before the driver is called \
         (validateWithDevice:, measured on Apple M4, macOS 27.0.1), so no driver result is defined"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use webmetal_wire::status;

    fn r(kind: RefusalKind) -> Refusal {
        Refusal { kind, construct: "c".into(), location: None, reason: "why".into(), reference: None }
    }

    #[test]
    fn refusals_map_by_kind() {
        let call = "-[MTLDevice newComputePipelineStateWithDescriptor:options:reflection:error:]";
        let f = refusal(call, &r(RefusalKind::RejectedByDevice));
        assert_eq!(f.status, status::UNIMPLEMENTED);
        assert!(f.message.contains("not established"), "{}", f.message);
        assert_eq!(refusal(call, &r(RefusalKind::NotImplemented)).status, status::UNIMPLEMENTED);
        assert_eq!(refusal(call, &r(RefusalKind::NoExactLowering)).status, status::UNIMPLEMENTED);
        assert_eq!(refusal(call, &r(RefusalKind::HostUnsupported)).status, status::HOST_UNSUPPORTED);
    }
}
