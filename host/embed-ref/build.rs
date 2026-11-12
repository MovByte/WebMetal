//! With the `native` feature on wgpu-native, which `webmetal-host` links dynamically, the programs and tests of this
//! crate find the library where `webmetal-host`'s own targets do: `WEBGPU_NATIVE_DIR/lib`, added to their run-time
//! search path (`host/build.rs` describes the variables). A build script's link arguments apply only to its own
//! package's targets, so this one repeats them for this package's.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=WEBGPU_NATIVE");
    println!("cargo:rerun-if-env-changed=WEBGPU_NATIVE_DIR");
    if std::env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        return;
    }
    if std::env::var("WEBGPU_NATIVE").as_deref() == Ok("wgpu-native")
        && let Ok(root) = std::env::var("WEBGPU_NATIVE_DIR")
    {
        let lib = std::path::Path::new(&root).join("lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }
}
