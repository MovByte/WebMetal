//! Binary archives: `CreateBinaryArchive` and the `BinaryArchiveAdd*PipelineFunctions` records.
//!
//! An archive holds what its device compiled, so that a later pipeline can be built without compiling. The A10X
//! driver's archives hold AGX machine code; this host's hold *records*: for each pipeline added, the key that names
//! what the host compiles for it ([`record`]: the compiler input of the pipeline's creation variant and the
//! descriptor state the pipeline keeps). A pipeline created with `MTLPipelineOptionFailOnBinaryArchiveMiss` is built
//! when one of its descriptor's archives holds its record and fails as Metal fails on a miss otherwise; without the
//! option the archives are not consulted, as a hit only saves compilation time. An archive the A10X driver wrote
//! holds no record of this host and would be a miss, as an archive for another GPU is on a real device.
//!
//! An archive loaded from a URL is read by the driver from the file system of the task that names it; the wire
//! protocol carries the URL, not the file's bytes, so loading one is not implemented (status 2). `serializeToURL:`
//! is Metal.framework's (guest-local in the schema).

use std::cell::RefCell;
use std::collections::BTreeSet;

use webmetal_wire::{BinaryArchiveDescriptor, Failure, Handle};

use super::compile::KeyWriter;
use super::errors;
use crate::exec::{Table, failure};

/// The payload of a `BinaryArchive` handle.
#[derive(Debug, Default)]
pub struct Archive {
    records: RefCell<BTreeSet<Vec<u8>>>,
}

impl Archive {
    pub fn add(&self, record: Vec<u8>) {
        self.records.borrow_mut().insert(record);
    }

    pub fn contains(&self, record: &[u8]) -> bool {
        self.records.borrow().contains(record)
    }

    pub fn len(&self) -> usize {
        self.records.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.borrow().is_empty()
    }
}

/// The record of a pipeline: its kind, the compiler key of its creation variant and the descriptor state it keeps.
pub fn record(kind: u8, compile_key: &[u8], kept: impl FnOnce(&mut KeyWriter)) -> Vec<u8> {
    let mut w = KeyWriter::default();
    w.tag(kind);
    w.bytes(compile_key);
    kept(&mut w);
    w.0
}

/// Record kinds.
pub const RENDER: u8 = 0;
pub const COMPUTE: u8 = 1;

/// `CreateBinaryArchive`: an empty archive for a descriptor without a URL.
pub fn create(call: &str, d: &BinaryArchiveDescriptor) -> Result<Archive, Failure> {
    let BinaryArchiveDescriptor { options, url, writing_options } = d;
    if let Some(u) = url {
        return Err(failure::unimplemented(format!(
            "{call}: loading the binary archive at {u}: the driver reads the file in the calling task, and the wire \
             protocol carries the URL, not the archive's bytes"
        )));
    }
    let unknown = "the behavior of a value other than 0 (measured on macOS 27.0.1) is not established";
    if *options != 0 {
        return Err(errors::field(call, "options", options, unknown));
    }
    if *writing_options != 0 {
        return Err(errors::field(call, "writingOptions", writing_options, unknown));
    }
    Ok(Archive::default())
}

/// Whether one of `archives` holds `record`, under `MTLPipelineOptionFailOnBinaryArchiveMiss`.
pub fn lookup(call: &str, objects: &Table, archives: &[Handle], record: &[u8]) -> Result<(), Failure> {
    for &h in archives {
        let a = objects.get(h).and_then(|e| e.payload.clone()).and_then(|p| p.downcast::<Archive>().ok()).ok_or_else(
            || failure::invalid(format!("{call}: binary archive {:#x} was not made by CreateBinaryArchive", h.get())),
        )?;
        if a.contains(record) {
            return Ok(());
        }
    }
    Err(errors::driver_error(call, errors::ARCHIVE_MISS, format!("{} archives searched", archives.len())))
}
