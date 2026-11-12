//! Cross-language round trip: the generated C++ encoder against the generated Rust decoder, and the Rust encoder
//! against the C++ decoder, on random records from the same seeds.
//!
//! Builds wire/tests/cpp/crosslang.cpp with the C++ compiler named by $CXX (default: `xcrun clang++` on macOS, `c++`
//! elsewhere) as C++17 and, on macOS, as Objective-C++. A missing compiler fails the test.

use std::path::{Path, PathBuf};
use std::process::Command as Process;
use std::sync::OnceLock;

use webmetal_wire::fuzz::Rng;
use webmetal_wire::{Channel, Command, Event, OPS, Reply, Request, Value};

const SEEDS: [u64; 3] = [11, 0x0123_4567_89ab_cdef, 0xffff_ffff_ffff_fffe];

fn wire_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn compiler() -> Vec<String> {
    if let Ok(cxx) = std::env::var("CXX") {
        return cxx.split_whitespace().map(String::from).collect();
    }
    if cfg!(target_os = "macos") { vec!["xcrun".into(), "clang++".into()] } else { vec!["c++".into()] }
}

fn build(objc: bool) -> PathBuf {
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(if objc { "crosslang-objc" } else { "crosslang-cxx" });
    let cxx = compiler();
    let mut cmd = Process::new(&cxx[0]);
    cmd.args(&cxx[1..]);
    cmd.args(["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-I"]);
    cmd.arg(wire_dir().join("generated"));
    if objc {
        cmd.args(["-x", "objective-c++", "-fobjc-arc"]);
    }
    cmd.arg(wire_dir().join("tests/cpp/crosslang.cpp"));
    if objc {
        cmd.args(["-framework", "Foundation", "-framework", "CoreFoundation"]);
    }
    cmd.arg("-o").arg(&out);
    let status = cmd.status().unwrap_or_else(|e| panic!("cannot run the C++ compiler {:?}: {}", cxx, e));
    assert!(status.success(), "building {} failed", out.display());
    out
}

fn driver(objc: bool) -> &'static Path {
    static CXX: OnceLock<PathBuf> = OnceLock::new();
    static OBJC: OnceLock<PathBuf> = OnceLock::new();
    if objc { OBJC.get_or_init(|| build(true)) } else { CXX.get_or_init(|| build(false)) }
}

fn run(objc: bool, args: &[&str]) {
    let status = Process::new(driver(objc)).args(args).status().expect("run the C++ driver");
    assert!(status.success(), "C++ driver {:?} failed: {}", args, status);
}

fn scratch(name: &str, seed: u64, objc: bool) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{}-{}-{}.bin", name, seed, if objc { "objc" } else { "cxx" }))
}

/// C++ encodes `count` random records; Rust encodes the same records from the same seed. The bytes must be equal,
/// and the Rust decoder must reproduce them exactly.
macro_rules! cpp_to_rust {
    ($ty:ty, $mode:expr, $count:expr, $channel:expr, $objc:expr) => {{
        let mut seen = std::collections::BTreeSet::new();
        for seed in SEEDS {
            let path = scratch($mode, seed, $objc);
            run($objc, &[$mode, &seed.to_string(), &$count.to_string(), path.to_str().unwrap()]);
            let cpp = std::fs::read(&path).unwrap();
            let mut rng = Rng::new(seed);
            let mut rust = Vec::new();
            for _ in 0..$count {
                <$ty>::random(&mut rng).encode(&mut rust).unwrap();
            }
            assert_eq!(cpp.len(), rust.len(), "C++ and Rust encodings differ in length (seed {})", seed);
            assert!(cpp == rust, "C++ and Rust encodings differ (seed {})", seed);
            let decoded = <$ty>::decode_all(&cpp).expect("Rust decodes the C++ encoding");
            assert_eq!(decoded.len(), $count);
            let mut again = Vec::new();
            for (_, v) in &decoded {
                v.encode(&mut again).unwrap();
                seen.insert(v.opcode());
            }
            assert!(again == cpp, "re-encoding the decoded records differs (seed {})", seed);
        }
        let all: std::collections::BTreeSet<u16> =
            OPS.iter().filter(|op| op.channel == $channel).map(|op| op.opcode).collect();
        assert_eq!(seen, all, "some records were never exercised");
    }};
}

/// Rust encodes `count` random records; C++ decodes them and compares with its own generator's values.
macro_rules! rust_to_cpp {
    ($ty:ty, $mode:expr, $count:expr, $objc:expr) => {{
        for seed in SEEDS {
            let mut rng = Rng::new(seed);
            let mut bytes = Vec::new();
            for _ in 0..$count {
                <$ty>::random(&mut rng).encode(&mut bytes).unwrap();
            }
            let path = scratch($mode, seed, $objc);
            std::fs::write(&path, &bytes).unwrap();
            run($objc, &[$mode, &seed.to_string(), &$count.to_string(), path.to_str().unwrap()]);
        }
    }};
}

fn all_directions(objc: bool) {
    cpp_to_rust!(Request, "requests", 2500, Channel::Control, objc);
    cpp_to_rust!(Command, "commands", 6000, Channel::Stream, objc);
    rust_to_cpp!(Reply, "replies", 2500, objc);
    rust_to_cpp!(Event, "events", 400, objc);
}

#[test]
fn cxx17_encoder_and_decoder_agree_with_rust() {
    all_directions(false);
}

#[cfg(target_os = "macos")]
#[test]
fn objcxx_encoder_and_decoder_agree_with_rust() {
    all_directions(true);
}

#[cfg(target_os = "macos")]
#[test]
fn foundation_objects_encode_as_property_lists() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("foundation.bin");
    run(true, &["foundation", path.to_str().unwrap()]);
    let bytes = std::fs::read(&path).unwrap();
    let records = Request::decode_all(&bytes).unwrap();
    assert_eq!(records.len(), 2);
    match &records[0].1 {
        Request::RegisterFunction(f) => {
            assert_eq!(f.function.get(), 7);
            assert_eq!(f.name, "vertex_main\u{e9}");
        }
        other => panic!("unexpected {:?}", other),
    }
    match &records[1].1 {
        Request::CreateRenderPipeline(p) => {
            assert_eq!(p.pipeline.get(), 9);
            let expected = Value::Dictionary(vec![
                (
                    "a".into(),
                    Value::Array(vec![
                        Value::Int(-3),
                        Value::UInt(4_000_000_000),
                        Value::Float(1.5),
                        Value::Double(2.25),
                        Value::String("text".into()),
                        Value::Data(vec![1, 2]),
                    ]),
                ),
                ("b".into(), Value::Bool(true)),
            ]);
            assert_eq!(p.descriptor.driver_compiler_options.as_ref(), Some(&expected));
        }
        other => panic!("unexpected {:?}", other),
    }
}
