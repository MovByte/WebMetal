//! The AIR registry: the modules each task registered by digest (`RegisterAir`) and the functions it named in them
//! (`RegisterFunction`).
//!
//! Modules are content-addressed: the host checks that a module's bytes have the SHA-256 the guest registers them
//! under, so two tasks that register the same digest share one copy, and a cache keyed by digests (the compiler
//! cache) can never be filled from one task's bytes for another task's different module. `RegisterAir` is sent once
//! per digest per connection (`wire/schema/webmetal.toml`); a second registration of a digest, or bytes that do not
//! hash to the digest, break the protocol (status 4).

use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use webmetal_wire::Failure;

use super::sha256::{hex, sha256};
use crate::embed::TaskId;
use crate::exec::failure;

/// One AIR module: an LLVM bitcode module as a metallib stores it for one function.
#[derive(Debug)]
pub struct Air {
    /// SHA-256 of `bitcode`.
    pub hash: [u8; 32],
    pub bitcode: Arc<[u8]>,
}

/// The payload of a `Function` handle (`RegisterFunction`): a function of a registered module.
#[derive(Debug)]
pub struct Function {
    pub air: Rc<Air>,
    pub name: String,
    /// `MTLFunctionType`.
    pub function_type: u64,
    /// Function constant values the bitcode was not already specialized with; empty when it was.
    pub constants: Vec<airwgsl::FunctionConstant>,
}

/// `MTLFunctionType` values (iPhoneOS SDK, `MTLLibrary.h`).
pub mod function_type {
    pub const VERTEX: u64 = 1;
    pub const FRAGMENT: u64 = 2;
    pub const KERNEL: u64 = 3;
    pub const VISIBLE: u64 = 5;
    pub const INTERSECTION: u64 = 6;
    pub const MESH: u64 = 7;
    pub const OBJECT: u64 = 8;

    pub const ALL: [u64; 7] = [VERTEX, FRAGMENT, KERNEL, VISIBLE, INTERSECTION, MESH, OBJECT];

    /// The enumerator's name without its prefix, for messages.
    pub fn name(t: u64) -> &'static str {
        match t {
            VERTEX => "Vertex",
            FRAGMENT => "Fragment",
            KERNEL => "Kernel",
            VISIBLE => "Visible",
            INTERSECTION => "Intersection",
            MESH => "Mesh",
            OBJECT => "Object",
            _ => "?",
        }
    }
}

impl Function {
    /// The compiler's input for this function used as a pipeline stage.
    pub fn stage_module(&self) -> airwgsl::AirModule {
        airwgsl::AirModule {
            hash: self.air.hash,
            bitcode: self.air.bitcode.clone(),
            entry: Some(airwgsl::EntryPoint {
                function: self.name.clone(),
                function_type: self.function_type,
                constants: self.constants.clone(),
            }),
        }
    }

    /// `name` and the module digest, for messages.
    pub fn describe(&self) -> String {
        format!("{} function {} (AIR {})", function_type::name(self.function_type), self.name, hex(&self.air.hash[..8]))
    }
}

/// The modules every task registered.
#[derive(Debug, Default)]
pub struct AirRegistry {
    /// Every module some task holds, by digest, so another task registering the same digest shares the copy.
    shared: HashMap<[u8; 32], Weak<Air>>,
    /// The modules each task registered.
    tasks: HashMap<TaskId, HashMap<[u8; 32], Rc<Air>>>,
}

impl AirRegistry {
    /// `RegisterAir`: make `bitcode` known to `task` under `hash`.
    pub fn register(&mut self, task: TaskId, hash: [u8; 32], bitcode: &[u8]) -> Result<(), Failure> {
        let mine = self.tasks.entry(task).or_default();
        if mine.contains_key(&hash) {
            return Err(failure::invalid(format!(
                "RegisterAir: AIR {} is already registered on this connection (it is sent once per digest)",
                hex(&hash)
            )));
        }
        let actual = sha256(bitcode);
        if actual != hash {
            return Err(failure::invalid(format!(
                "RegisterAir: the {} bytes registered under {} have the SHA-256 {}",
                bitcode.len(),
                hex(&hash),
                hex(&actual)
            )));
        }
        let air = match self.shared.get(&hash).and_then(Weak::upgrade) {
            Some(a) => a,
            None => {
                let a = Rc::new(Air { hash, bitcode: Arc::from(bitcode) });
                self.shared.insert(hash, Rc::downgrade(&a));
                a
            }
        };
        mine.insert(hash, air);
        Ok(())
    }

    /// The module `task` registered under `hash`.
    pub fn get(&self, task: TaskId, hash: &[u8; 32]) -> Option<Rc<Air>> {
        self.tasks.get(&task)?.get(hash).cloned()
    }

    /// `RegisterFunction`: the function `name` of type `function_type` in the module `task` registered under `air`.
    pub fn function(
        &self,
        task: TaskId,
        air: &[u8; 32],
        name: &str,
        function_type: u64,
        constants: &[webmetal_wire::FunctionConstant],
    ) -> Result<Function, Failure> {
        let module = self.get(task, air).ok_or_else(|| {
            failure::invalid(format!("RegisterFunction: AIR {} is not registered on this connection", hex(air)))
        })?;
        if !function_type::ALL.contains(&function_type) {
            return Err(failure::invalid(format!("RegisterFunction: {function_type} is not an MTLFunctionType")));
        }
        let constants = constants
            .iter()
            .map(|c| airwgsl::FunctionConstant {
                index: c.index,
                name: c.name.clone(),
                data_type: c.data_type,
                value: c.value.clone(),
            })
            .collect();
        Ok(Function { air: module, name: name.to_string(), function_type, constants })
    }

    /// The number of modules `task` registered.
    pub fn count(&self, task: TaskId) -> usize {
        self.tasks.get(&task).map_or(0, HashMap::len)
    }

    /// The number of distinct modules held by all tasks.
    pub fn distinct(&self) -> usize {
        self.shared.values().filter(|w| w.strong_count() > 0).count()
    }

    /// The task exited: its registrations end. Functions still referred to by live objects keep their module.
    pub fn task_exited(&mut self, task: TaskId) {
        self.tasks.remove(&task);
        self.shared.retain(|_, w| w.strong_count() > 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use webmetal_wire::status;

    const T1: TaskId = TaskId(1);
    const T2: TaskId = TaskId(2);

    #[test]
    fn modules_are_registered_once_under_their_digest() {
        let mut r = AirRegistry::default();
        let bytes = b"BC\xc0\xde module".to_vec();
        let hash = sha256(&bytes);
        r.register(T1, hash, &bytes).unwrap();
        let again = r.register(T1, hash, &bytes).unwrap_err();
        assert_eq!(again.status, status::INVALID);
        assert!(again.message.contains("already registered"), "{}", again.message);
        let mut wrong = hash;
        wrong[0] ^= 1;
        let bad = r.register(T1, wrong, &bytes).unwrap_err();
        assert_eq!(bad.status, status::INVALID);
        assert!(bad.message.contains("have the SHA-256"), "{}", bad.message);
        assert!(r.get(T1, &wrong).is_none());
        // Another task shares the copy.
        r.register(T2, hash, &bytes).unwrap();
        assert!(Rc::ptr_eq(&r.get(T1, &hash).unwrap(), &r.get(T2, &hash).unwrap()));
        assert_eq!(r.distinct(), 1);
        assert!(r.get(T2, &[0; 32]).is_none());
    }

    #[test]
    fn functions_name_a_registered_module() {
        let mut r = AirRegistry::default();
        let bytes = vec![7u8; 100];
        let hash = sha256(&bytes);
        assert_eq!(r.function(T1, &hash, "f", 3, &[]).unwrap_err().status, status::INVALID);
        r.register(T1, hash, &bytes).unwrap();
        let f = r.function(T1, &hash, "f", 3, &[]).unwrap();
        assert_eq!(f.stage_module().hash, hash);
        assert_eq!(&*f.stage_module().bitcode, &bytes[..]);
        assert_eq!(r.function(T1, &hash, "f", 4, &[]).unwrap_err().status, status::INVALID);
        assert_eq!(
            r.function(T2, &hash, "f", 3, &[]).unwrap_err().status,
            status::INVALID,
            "registrations are per task"
        );
        // A function keeps its module after the task's registrations end.
        r.task_exited(T1);
        assert_eq!(r.count(T1), 0);
        assert_eq!(&f.air.bitcode[..], &bytes[..]);
        assert_eq!(r.distinct(), 1);
        drop(f);
        r.task_exited(T1);
        assert_eq!(r.distinct(), 0);
    }
}
