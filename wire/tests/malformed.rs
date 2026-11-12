//! Decoders against malformed input: every mutation of a valid stream either fails to decode or decodes to records
//! that re-encode to exactly the mutated bytes. A decoder may reject; it may never misinterpret, and never panic.

use webmetal_wire::fuzz::Rng;
use webmetal_wire::{Command, DecodeErrorKind, Event, Reply, Request};

fn mutate(rng: &mut Rng, bytes: &[u8]) -> Vec<u8> {
    let mut m = bytes.to_vec();
    match rng.below(6) {
        0 => {
            // flip one bit
            let at = rng.below(m.len() as u64) as usize;
            m[at] ^= 1 << rng.below(8);
        }
        1 => {
            // overwrite one byte
            let at = rng.below(m.len() as u64) as usize;
            m[at] = rng.u8();
        }
        2 => {
            // truncate
            let len = rng.below(m.len() as u64) as usize;
            m.truncate(len);
        }
        3 => {
            // append bytes
            let n = 1 + rng.below(16);
            for _ in 0..n {
                m.push(rng.u8());
            }
        }
        4 => {
            // rewrite the first header's length
            let len = (rng.u32() % (m.len() as u32 + 64)) & !3;
            m[4..8].copy_from_slice(&len.to_le_bytes());
        }
        _ => {
            // overwrite an aligned u32 (counts, tags, small fields)
            if m.len() >= 12 {
                let at = 8 + 4 * rng.below(((m.len() - 8) / 4) as u64) as usize;
                let v = match rng.below(4) {
                    0 => 0,
                    1 => 0xffff_ffff,
                    2 => rng.below(64) as u32,
                    _ => rng.u32(),
                };
                m[at..at + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    m
}

macro_rules! check_channel {
    ($ty:ty, $iterations:expr) => {{
        let mut rng = Rng::new(0x6d61_6c66_6f72_6d65);
        let mut rejected = 0usize;
        for _ in 0..$iterations {
            let mut bytes = Vec::new();
            let n = 1 + rng.below(3);
            for _ in 0..n {
                <$ty>::random(&mut rng).encode(&mut bytes).unwrap();
            }
            let mutated = mutate(&mut rng, &bytes);
            let mut at = 0;
            let mut again = Vec::new();
            loop {
                if at == mutated.len() {
                    // Everything decoded: it must be the canonical encoding of what was decoded.
                    assert_eq!(again, mutated, "decoded records do not re-encode to the input");
                    break;
                }
                match <$ty>::decode(&mutated[at..], at) {
                    Ok((v, len)) => {
                        v.encode(&mut again).unwrap();
                        assert_eq!(&again[at..], &mutated[at..at + len], "record at {} re-encodes differently", at);
                        at += len;
                    }
                    Err(e) => {
                        assert!(e.offset <= mutated.len());
                        rejected += 1;
                        break;
                    }
                }
            }
        }
        assert!(rejected > 0);
    }};
}

#[test]
fn malformed_commands_are_rejected_or_canonical() {
    check_channel!(Command, 20000);
}

#[test]
fn malformed_requests_are_rejected_or_canonical() {
    check_channel!(Request, 20000);
}

#[test]
fn malformed_replies_are_rejected_or_canonical() {
    check_channel!(Reply, 20000);
}

#[test]
fn malformed_events_are_rejected_or_canonical() {
    check_channel!(Event, 5000);
}

#[test]
fn records_of_other_channels_are_refused() {
    let mut rng = Rng::new(9);
    let mut bytes = Vec::new();
    Request::random(&mut rng).encode(&mut bytes).unwrap();
    let opcode = u16::from_le_bytes([bytes[0], bytes[1]]);
    assert_eq!(Command::decode(&bytes, 0).unwrap_err().kind, DecodeErrorKind::WrongChannel(opcode));
    let unknown = [0x34, 0x7f, 0, 0, 8, 0, 0, 0];
    assert_eq!(Command::decode(&unknown, 0).unwrap_err().kind, DecodeErrorKind::UnknownOpcode(0x7f34));
}

#[test]
fn nonzero_flags_and_padding_are_refused() {
    let mut bytes = Vec::new();
    Command::RenderSetStencilReferenceValue(webmetal_wire::RenderSetStencilReferenceValue { reference_value: 1 })
        .encode(&mut bytes)
        .unwrap();
    let mut flags = bytes.clone();
    flags[2] = 1;
    assert_eq!(Command::decode(&flags, 0).unwrap_err().kind, DecodeErrorKind::BadFlags(1));
    let mut padding = bytes.clone();
    padding[15] = 1;
    assert_eq!(Command::decode(&padding, 0).unwrap_err().kind, DecodeErrorKind::NonZeroPadding);
}
