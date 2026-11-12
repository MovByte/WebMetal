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
    // The options belong to Metal.framework, which runs in the guest. On 21H461 options and writingOptions are one
    // field (both accessors read offset 0x10: 0x187ad1f24, 0x187b0fb20), which `-[MTLBinaryArchiveDescriptor init]`
    // sets to 2 (0x187ad2408); `-[_MTLBinaryArchive initWithDevice:descriptor:error:]` (0x187ad1e8c) hands it to
    // `initWithOptions:device:url:error:` (0x187ad2940). The A10X driver's `newBinaryArchiveWithDescriptor:error:`
    // (AGXMetalA10 21H461 0x21c8d8f90) reads neither, so they change nothing the driver does.
    let BinaryArchiveDescriptor { options: _, url, writing_options: _ } = d;
    if let Some(u) = url {
        return Err(failure::unimplemented(format!(
            "{call}: loading the binary archive at {u}: the driver reads the file in the calling task, and the wire \
             protocol carries the URL, not the archive's bytes"
        )));
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
