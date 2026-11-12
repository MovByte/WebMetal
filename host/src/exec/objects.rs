//! Object tables: the handles of one task (one connection) and what each names.
//!
//! The guest allocates every handle; within a connection a handle is unique and never reused, and 0 is nil
//! (`wire/README.md`, "Handles"). A handle becomes live when the record that creates it succeeds (its `Result` field,
//! or the command buffer a Commit names) and stays live until the guest releases it. Every other handle a record
//! carries must be live and of a kind the schema's field accepts ([`Kind::satisfies`]); anything else is a protocol
//! violation. A handle is never valid in another task: shared events and IOSurfaces cross tasks as kernel objects,
//! through RegisterSharedEvent and surface ids, not through handles.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use webmetal_wire::Handle;

use super::schema::{HandleUse, Kind, Role};

/// Which part of the executor created an object and is told when it is released.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Owner {
    /// The executor core (command queues, command buffers, encoders, events, fences).
    Core,
    /// The control service with this index in [`super::Services::control`].
    Service(usize),
}

/// One live object.
#[derive(Clone)]
pub struct Entry {
    pub kind: Kind,
    pub owner: Owner,
    /// The label the guest last set with SetLabel (or carried in the record that created the object).
    pub label: Option<String>,
    /// What the owning service attached to the object, for the services that use it (a buffer's GPU storage, a
    /// pipeline's WebGPU pipelines). Its type is the owning service's to document.
    pub payload: Option<Rc<dyn Any>>,
}

impl std::fmt::Debug for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry")
            .field("kind", &self.kind)
            .field("owner", &self.owner)
            .field("label", &self.label)
            .field("payload", &self.payload.as_ref().map(|_| "..."))
            .finish()
    }
}

/// The handles of one task.
#[derive(Debug, Default)]
pub struct Table {
    live: HashMap<Handle, Entry>,
    /// Every handle ever bound in this task, live or released.
    seen: HashSet<Handle>,
}

impl Table {
    /// The live object with this handle.
    pub fn get(&self, handle: Handle) -> Option<&Entry> {
        self.live.get(&handle)
    }

    pub(crate) fn get_mut(&mut self, handle: Handle) -> Option<&mut Entry> {
        self.live.get_mut(&handle)
    }

    /// Whether the handle was never bound in this task.
    pub fn is_fresh(&self, handle: Handle) -> bool {
        !self.seen.contains(&handle)
    }

    /// Check a handle a record carries against the table. `Result` handles must be fresh and of a kind records
    /// create; every other role must name a live object the field's kind accepts.
    pub fn check(&self, u: &HandleUse) -> Result<(), String> {
        match u.role {
            Role::Result => {
                if !self.is_fresh(u.handle) {
                    return Err(format!(
                        "{}: handle {:#x} was already used in this connection",
                        u.field,
                        u.handle.get()
                    ));
                }
                if !u.kind.is_created() {
                    return Err(format!("{}: no record creates a {}", u.field, u.kind.name()));
                }
                Ok(())
            }
            Role::Receiver | Role::Argument | Role::Released => match self.live.get(&u.handle) {
                None if self.seen.contains(&u.handle) => {
                    Err(format!("{}: handle {:#x} was released", u.field, u.handle.get()))
                }
                None => Err(format!("{}: unknown handle {:#x}", u.field, u.handle.get())),
                Some(e) if !e.kind.satisfies(u.kind) => Err(format!(
                    "{}: handle {:#x} is a {}, the field takes a {}",
                    u.field,
                    u.handle.get(),
                    e.kind.name(),
                    u.kind.name()
                )),
                Some(_) => Ok(()),
            },
        }
    }

    /// Bind a fresh handle.
    pub(crate) fn bind(&mut self, handle: Handle, entry: Entry) -> Result<(), String> {
        if !self.seen.insert(handle) {
            return Err(format!("handle {:#x} was already used in this connection", handle.get()));
        }
        self.live.insert(handle, entry);
        Ok(())
    }

    /// Release a live handle; it can never be bound again.
    pub(crate) fn unbind(&mut self, handle: Handle) -> Option<Entry> {
        self.live.remove(&handle)
    }

    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(v: u64) -> Handle {
        Handle::new(v).unwrap()
    }

    fn entry(kind: Kind) -> Entry {
        Entry { kind, owner: Owner::Core, label: None, payload: None }
    }

    fn use_(handle: u64, kind: Kind, role: Role) -> HandleUse {
        HandleUse { field: "f", kind, role, handle: h(handle) }
    }

    #[test]
    fn results_must_be_fresh_and_creatable() {
        let mut t = Table::default();
        assert!(t.check(&use_(1, Kind::Buffer, Role::Result)).is_ok());
        t.bind(h(1), entry(Kind::Buffer)).unwrap();
        assert!(t.check(&use_(1, Kind::Buffer, Role::Result)).is_err());
        t.unbind(h(1));
        assert!(t.check(&use_(1, Kind::Buffer, Role::Result)).unwrap_err().contains("already used"));
        assert!(t.bind(h(1), entry(Kind::Buffer)).is_err(), "handles are never reused");
        assert!(t.check(&use_(2, Kind::Resource, Role::Result)).is_err(), "nothing creates a Resource");
    }

    #[test]
    fn arguments_must_be_live_and_of_an_accepted_kind() {
        let mut t = Table::default();
        t.bind(h(5), entry(Kind::Texture)).unwrap();
        assert!(t.check(&use_(5, Kind::Texture, Role::Argument)).is_ok());
        assert!(t.check(&use_(5, Kind::Resource, Role::Argument)).is_ok());
        assert!(t.check(&use_(5, Kind::Object, Role::Released)).is_ok());
        assert!(t.check(&use_(5, Kind::Buffer, Role::Receiver)).unwrap_err().contains("is a Texture"));
        assert!(t.check(&use_(6, Kind::Buffer, Role::Argument)).unwrap_err().contains("unknown"));
        t.unbind(h(5));
        assert!(t.check(&use_(5, Kind::Texture, Role::Argument)).unwrap_err().contains("released"));
    }
}
