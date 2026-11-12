//! `setPurgeableState:` of resources and heaps.
//!
//! `MTLResource.h` and the `setPurgeableState(_:)` documentation (`tables/limits.toml` question
//! `6.2-purgeable-state-return`): the call returns the prior state, and with `keepCurrent` the current state without
//! changing it. Volatile contents "can be discarded ... at any time without informing the app"; this executor never
//! discards them, which is one of the outcomes Metal allows (plan section 6.2), so a volatile resource keeps
//! answering volatile. `empty` discards the contents: the caller discards them (and zero-fills, so that nothing stale
//! is observable) when [`Purgeable::set`] says so. A heap's state covers all of its resources ("Heaps can be marked
//! purgeable but its resources cannot; the heap's resources always reflect the heap's purgeability state",
//! `MTLHeap.setPurgeableState(_:)` documentation).
//!
//! Values are the SDK's: `MTLPurgeableStateKeepCurrent` 1, `NonVolatile` 2, `Volatile` 3, `Empty` 4.

use super::error::MemError;

pub const KEEP_CURRENT: u64 = 1;
pub const NON_VOLATILE: u64 = 2;
pub const VOLATILE: u64 = 3;
pub const EMPTY: u64 = 4;

/// A resource's or heap's purgeable state. New objects are non-volatile (`resource.purgeable-state` of the Metal
/// reference outputs: the initial query answers 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Purgeable {
    NonVolatile,
    Volatile,
    Empty,
}

/// What a `setPurgeableState:` call does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    /// The value the call returns.
    pub previous: u64,
    /// The contents must be discarded now.
    pub discard: bool,
}

impl Purgeable {
    pub fn value(self) -> u64 {
        match self {
            Purgeable::NonVolatile => NON_VOLATILE,
            Purgeable::Volatile => VOLATILE,
            Purgeable::Empty => EMPTY,
        }
    }

    /// Apply `setPurgeableState:state`.
    pub fn set(&mut self, state: u64, call: &'static str) -> Result<Transition, MemError> {
        let previous = self.value();
        let next = match state {
            KEEP_CURRENT => return Ok(Transition { previous, discard: false }),
            NON_VOLATILE => Purgeable::NonVolatile,
            VOLATILE => Purgeable::Volatile,
            EMPTY => Purgeable::Empty,
            v => return Err(MemError::undefined(call, format!("{v} is not an MTLPurgeableState value"))),
        };
        // Emptying discards; a later transition out of empty does not bring the contents back.
        let discard = next == Purgeable::Empty && *self != Purgeable::Empty;
        *self = next;
        Ok(Transition { previous, discard })
    }

    /// `setPurgeableState:` on a resource allocated from a heap whose state is `heap`: only `keepCurrent` is defined
    /// (it reports the heap's state, which the resource reflects); changing a heap resource's own state is not.
    pub fn set_on_heap_resource(heap: Purgeable, state: u64, call: &'static str) -> Result<u64, MemError> {
        match state {
            KEEP_CURRENT => Ok(heap.value()),
            _ => Err(MemError::undefined(
                call,
                "a resource allocated from a heap cannot be marked purgeable; it reflects the heap's state \
                 (MTLHeap.setPurgeableState(_:) documentation)",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sequence of `resource.purgeable-state` in the Metal reference outputs (Apple M4, macOS 27.0.1), with the
    /// "volatile or empty" answers as this executor gives them (it never discards volatile contents).
    #[test]
    fn reference_sequence() {
        let mut p = Purgeable::NonVolatile;
        let steps = [
            (KEEP_CURRENT, NON_VOLATILE, false),
            (VOLATILE, NON_VOLATILE, false),
            (KEEP_CURRENT, VOLATILE, false),
            (NON_VOLATILE, VOLATILE, false),
            (KEEP_CURRENT, NON_VOLATILE, false),
            (EMPTY, NON_VOLATILE, true),
            (KEEP_CURRENT, EMPTY, false),
            (NON_VOLATILE, EMPTY, false),
            (KEEP_CURRENT, NON_VOLATILE, false),
        ];
        for (set, previous, discard) in steps {
            assert_eq!(p.set(set, "test").unwrap(), Transition { previous, discard }, "set {set}");
        }
        assert!(matches!(p.set(0, "test"), Err(MemError::Undefined { .. })));
        assert!(matches!(p.set(5, "test"), Err(MemError::Undefined { .. })));
    }

    #[test]
    fn heap_resources_reflect_the_heap() {
        assert_eq!(Purgeable::set_on_heap_resource(Purgeable::Volatile, KEEP_CURRENT, "t").unwrap(), VOLATILE);
        assert!(Purgeable::set_on_heap_resource(Purgeable::Volatile, EMPTY, "t").is_err());
    }
}
