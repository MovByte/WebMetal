//! The WebMetal wire protocol: the records the guest Metal plugin and the host executor exchange.
//!
//! Everything protocol-specific here is generated from `wire/schema/webmetal.toml` by `wire/gen/wiregen.py`
//! ([`generated`]); this crate adds the primitive codec ([`codec`]), property-list values ([`value`]) and deterministic
//! random values for tests ([`fuzz`]). The format and the transport are described in `wire/README.md`.
//!
//! Three channels:
//!
//! - **control**: guest-to-host requests ([`Request`]), each answered by one [`Reply`];
//! - **stream**: the command-buffer stream ([`Command`]), submitted by [`Commit`];
//! - **event**: host-to-guest records ([`Event`]), collected with [`FetchEvents`].
//!
//! Decoding is strict: a record that decodes re-encodes to the same bytes, and anything else is a [`DecodeError`]. A
//! record whose opcode the receiver does not implement still decodes; refusing it is the receiver's decision, reported
//! as a failure with status 2 (unimplemented), never skipped.

pub mod codec;
pub mod fuzz;
#[rustfmt::skip]
pub mod generated;
pub mod value;

pub use codec::{DecodeError, DecodeErrorKind, EncodeError, GuestAddr, Handle, SurfaceId};
pub use generated::*;
pub use value::{VALUE_DEPTH_LIMIT, Value};

/// Failure statuses (the `status` field of [`Failure`] and of a reply).
pub mod status {
    /// Success.
    pub const OK: u32 = 0;
    /// Metal defines this failure: `domain` and `code` are the NSError Metal returns for it.
    pub const METAL_ERROR: u32 = 1;
    /// The host does not implement what the record asks for. The guest reports it through the not-implemented channel.
    pub const UNIMPLEMENTED: u32 = 2;
    /// The host cannot honor a capability the device advertises (a missing WebGPU feature or a limit); the message
    /// names it. Reported like [`UNIMPLEMENTED`].
    pub const HOST_UNSUPPORTED: u32 = 3;
    /// The record is malformed, names an unknown handle, or breaks the protocol.
    pub const INVALID: u32 = 4;
}

/// The rules a [`Failure`] obeys: status 0 has no domain, code 0 and an empty message; status 1 has a domain;
/// statuses 2 to 4 have no domain and code 0; no other status exists.
pub(crate) fn failure_valid(f: &Failure) -> bool {
    match f.status {
        status::OK => f.domain.is_none() && f.code == 0 && f.message.is_empty(),
        status::METAL_ERROR => f.domain.is_some(),
        status::UNIMPLEMENTED | status::HOST_UNSUPPORTED | status::INVALID => f.domain.is_none() && f.code == 0,
        _ => false,
    }
}
