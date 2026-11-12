//! The core executor: control requests, object tables, command queues, command buffers, events and fences.
//!
//! [`schema`] holds the rules the executor derives from the wire schema: the object kinds and which kinds a field
//! accepts, the encoder scope of every stream record, and every handle each record carries with its role.

#[rustfmt::skip]
#[allow(clippy::all)]
pub mod schema;

pub use schema::Kind;

#[cfg(test)]
mod tests {
    use super::schema;

    #[test]
    fn schema_rules_are_generated_from_the_current_schema() {
        assert_eq!(schema::SCHEMA_HASH, webmetal_wire::SCHEMA_HASH, "run host/src/exec/gen/execgen.py");
    }

    #[test]
    fn generated_rules_are_current() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let status = std::process::Command::new("python3")
            .args(["-B", "-I"])
            .arg(root.join("src/exec/gen/execgen.py"))
            .arg("--check")
            .status()
            .expect("python3 is required to check the generated rules");
        assert!(status.success(), "host/src/exec/schema.rs is stale: run host/src/exec/gen/execgen.py");
    }
}
