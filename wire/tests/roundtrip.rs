//! Rust round trips, golden encodings and length constraints.

use std::collections::BTreeSet;

use webmetal_wire::fuzz::Rng;
use webmetal_wire::{
    opcode, Channel, Command, EncodeError, EncoderPushDebugGroup, Event, Failure, Handle, Hash, RegisterFunction, Reply,
    Request, RenderSampleCountersInBuffer, RenderSetStencilReferenceValue, RenderSetVertexBuffer, RenderSetVertexBuffers,
    CreateBufferReply, Range, OPS,
};

const SEEDS: [u64; 4] = [1, 2, 0x5eed, 0xdead_beef_cafe_f00d];

/// Encode, decode, re-encode: the bytes must be identical and the decoded record must equal the original.
macro_rules! round_trip {
    ($ty:ty, $count:expr) => {{
        let mut seen = BTreeSet::new();
        for seed in SEEDS {
            let mut rng = Rng::new(seed);
            let mut bytes = Vec::new();
            let mut records = Vec::new();
            for _ in 0..$count {
                let v = <$ty>::random(&mut rng);
                v.encode(&mut bytes).expect("random records are valid");
                records.push(v);
            }
            let mut at = 0;
            let mut again = Vec::new();
            for expected in &records {
                let (decoded, len) = <$ty>::decode(&bytes[at..], at).expect("encoded record decodes");
                decoded.encode(&mut again).unwrap();
                assert_eq!(&again[at..at + len], &bytes[at..at + len], "re-encoding differs at byte {}", at);
                // Floats may be NaN; compare through the canonical encoding rather than PartialEq.
                let mut one = Vec::new();
                expected.encode(&mut one).unwrap();
                assert_eq!(one.as_slice(), &bytes[at..at + len]);
                seen.insert(decoded_opcode(&bytes[at..]));
                at += len;
            }
            assert_eq!(at, bytes.len());
        }
        seen
    }};
}

fn decoded_opcode(record: &[u8]) -> u16 {
    u16::from_le_bytes([record[0], record[1]])
}

fn opcodes(channel: Channel) -> BTreeSet<u16> {
    OPS.iter().filter(|op| op.channel == channel).map(|op| op.opcode).collect()
}

#[test]
fn every_command_round_trips() {
    let seen = round_trip!(Command, 8000);
    assert_eq!(seen, opcodes(Channel::Stream), "some stream records were never generated");
}

#[test]
fn every_request_round_trips() {
    let seen = round_trip!(Request, 3000);
    assert_eq!(seen, opcodes(Channel::Control), "some requests were never generated");
}

#[test]
fn every_event_round_trips() {
    let seen = round_trip!(Event, 500);
    assert_eq!(seen, opcodes(Channel::Event));
}

#[test]
fn every_reply_round_trips() {
    let seen = round_trip!(Reply, 3000);
    let expected: BTreeSet<u16> = opcodes(Channel::Control).into_iter().map(|op| op | 0x8000).collect();
    assert_eq!(seen, expected);
}

fn header(opcode: u16, len: u32) -> Vec<u8> {
    let mut v = opcode.to_le_bytes().to_vec();
    v.extend_from_slice(&[0, 0]);
    v.extend_from_slice(&len.to_le_bytes());
    v
}

fn encode(c: Command) -> Vec<u8> {
    let mut out = Vec::new();
    c.encode(&mut out).unwrap();
    out
}

#[test]
fn golden_u64_fields() {
    let bytes = encode(Command::RenderSetVertexBuffer(RenderSetVertexBuffer {
        buffer: Handle::new(0x1122_3344_5566_7788),
        offset: 0x10,
        index: 3,
    }));
    let mut expected = header(opcode::RENDER_SET_VERTEX_BUFFER, 32);
    expected.extend_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
    expected.extend_from_slice(&0x10u64.to_le_bytes());
    expected.extend_from_slice(&3u64.to_le_bytes());
    assert_eq!(bytes, expected);

    let nil = encode(Command::RenderSetVertexBuffer(RenderSetVertexBuffer { buffer: None, offset: 0, index: 0 }));
    assert_eq!(&nil[8..16], &[0; 8], "a nil handle is 0");
}

#[test]
fn golden_padding() {
    // u32 at offset 8, then zero padding to the 8-byte record alignment.
    let bytes = encode(Command::RenderSetStencilReferenceValue(RenderSetStencilReferenceValue { reference_value: 0xa1b2_c3d4 }));
    let mut expected = header(opcode::RENDER_SET_STENCIL_REFERENCE_VALUE, 16);
    expected.extend_from_slice(&[0xd4, 0xc3, 0xb2, 0xa1, 0, 0, 0, 0]);
    assert_eq!(bytes, expected);

    // A bool after two u64 fields is one byte; the record is padded to 32.
    let bytes = encode(Command::RenderSampleCountersInBuffer(RenderSampleCountersInBuffer {
        sample_buffer: Handle::new(5),
        sample_index: 6,
        barrier: true,
    }));
    let mut expected = header(opcode::RENDER_SAMPLE_COUNTERS_IN_BUFFER, 32);
    expected.extend_from_slice(&5u64.to_le_bytes());
    expected.extend_from_slice(&6u64.to_le_bytes());
    expected.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(bytes, expected);
}

#[test]
fn golden_strings() {
    let bytes = encode(Command::EncoderPushDebugGroup(EncoderPushDebugGroup { string: Some("ab".into()) }));
    let mut expected = header(opcode::ENCODER_PUSH_DEBUG_GROUP, 16);
    expected.extend_from_slice(&[2, 0, 0, 0, b'a', b'b', 0, 0]);
    assert_eq!(bytes, expected);

    let bytes = encode(Command::EncoderPushDebugGroup(EncoderPushDebugGroup { string: None }));
    let mut expected = header(opcode::ENCODER_PUSH_DEBUG_GROUP, 16);
    expected.extend_from_slice(&[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]);
    assert_eq!(bytes, expected);
}

#[test]
fn golden_replies() {
    let mut ok = Vec::new();
    Reply::CreateBuffer(Ok(CreateBufferReply {})).encode(&mut ok).unwrap();
    let mut expected = header(opcode::CREATE_BUFFER | 0x8000, 16);
    expected.extend_from_slice(&[0; 8]);
    assert_eq!(ok, expected);

    let mut failed = Vec::new();
    let failure = Failure { status: 1, domain: Some("D".into()), code: -2, message: String::new() };
    Reply::CreateBuffer(Err(failure)).encode(&mut failed).unwrap();
    let mut expected = header(opcode::CREATE_BUFFER | 0x8000, 40);
    expected.extend_from_slice(&1u32.to_le_bytes()); // status
    expected.extend_from_slice(&1u32.to_le_bytes()); // domain length
    expected.extend_from_slice(&[b'D', 0, 0, 0, 0, 0, 0, 0]); // domain, padding to 8
    expected.extend_from_slice(&(-2i64).to_le_bytes()); // code
    expected.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // empty message, padding
    assert_eq!(failed, expected);
}

#[test]
fn failure_rules_are_enforced() {
    let bad = [
        Failure { status: 0, domain: None, code: 0, message: "x".into() },
        Failure { status: 1, domain: None, code: 3, message: String::new() },
        Failure { status: 2, domain: Some("d".into()), code: 0, message: String::new() },
        Failure { status: 3, domain: None, code: 1, message: String::new() },
        Failure { status: 5, domain: None, code: 0, message: String::new() },
    ];
    for f in bad {
        let mut out = Vec::new();
        assert_eq!(Reply::CreateBuffer(Err(f)).encode(&mut out), Err(EncodeError::BadFailure));
        assert!(out.is_empty(), "a failed encode leaves the output unchanged");
    }
}

#[test]
fn length_constraints_are_enforced() {
    let mut out = vec![0u8; 8];
    let mismatched = Command::RenderSetVertexBuffers(RenderSetVertexBuffers {
        buffers: vec![None, Handle::new(1)],
        offsets: vec![0],
        range: Range { location: 0, length: 2 },
    });
    assert_eq!(mismatched.encode(&mut out), Err(EncodeError::Constraint("offsets")));
    assert_eq!(out, vec![0u8; 8], "a failed encode leaves the output unchanged");

    let matched = Command::RenderSetVertexBuffers(RenderSetVertexBuffers {
        buffers: vec![None, Handle::new(1)],
        offsets: vec![0, 16],
        range: Range { location: 4, length: 2 },
    });
    let mut bytes = Vec::new();
    matched.encode(&mut bytes).unwrap();
    assert_eq!(Command::decode(&bytes, 0).unwrap().0, matched);
}

#[test]
fn null_handles_are_refused_where_metal_has_an_object() {
    let mut bytes = Vec::new();
    Request::RegisterFunction(RegisterFunction {
        function: Handle::new(1).unwrap(),
        air: Hash { bytes: [7; 32] },
        name: "f".into(),
        function_type: 1,
        constants: Vec::new(),
    })
    .encode(&mut bytes)
    .unwrap();
    bytes[8..16].copy_from_slice(&[0; 8]);
    let err = Request::decode(&bytes, 0).unwrap_err();
    assert_eq!(err.kind, webmetal_wire::DecodeErrorKind::NullHandle("function"));
}

#[test]
fn op_table_is_consistent() {
    let mut codes = BTreeSet::new();
    for op in OPS {
        assert!(codes.insert(op.opcode), "duplicate opcode {:#06x}", op.opcode);
        assert_eq!(op.opcode & 0x8000, 0);
        assert_eq!(webmetal_wire::op_info(op.opcode).unwrap().name, op.name);
    }
}
