//! The reference host's services on a native WebGPU implementation (`WEBGPU_NATIVE`, see `host/build.rs`). Run with
//! `--release`.

#![cfg(feature = "native")]

mod common;

use common::*;
use webmetal_host::exec::schema::reply_failure;
use webmetal_host::pipeline::sha256::sha256;
use webmetal_wire::*;

/// The status a reply carries (0 for success).
fn reply_status(h: &H, r: &Request) -> u32 {
    reply_failure(&h.request(r)).map_or(status::OK, |f| f.status)
}

#[test]
fn pipeline_requests_reach_the_pipeline_service() {
    let h = H::new();
    h.open();
    let bitcode = b"not a module the compiler is asked about here".to_vec();
    let air = Request::RegisterAir(RegisterAir { hash: Hash { bytes: sha256(&bitcode) }, bitcode: bitcode.clone() });
    // Served by the pipeline service: registered once per digest and connection, refused (malformed) the second
    // time and under a digest the bytes do not have. Without the service each would be "not implemented" (2).
    assert_eq!(reply_status(&h, &air), status::OK);
    assert_eq!(reply_status(&h, &air), status::INVALID);
    let wrong = Request::RegisterAir(RegisterAir { hash: Hash { bytes: [7; 32] }, bitcode });
    assert_eq!(reply_status(&h, &wrong), status::INVALID);
}
