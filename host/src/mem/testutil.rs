//! Test helpers: paths into the repository.

use std::path::PathBuf;

pub fn repo_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}
