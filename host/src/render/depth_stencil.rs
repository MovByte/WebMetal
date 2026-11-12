//! `MTLDepthStencilState`: the state object `-[MTLDevice newDepthStencilStateWithDescriptor:]` makes, and how it meets
//! WebGPU, where depth and stencil tests are render pipeline state.
//!
//! The control request `CreateDepthStencilState` is served here; the handle's payload is a [`DepthStencilState`]. The
//! encoder folds the state bound with `setDepthStencilState:` into the pipeline variant of each draw
//! ([`super::pipeline::VariantKey`]).
//!
//! A nil `frontFaceStencil` or `backFaceStencil` disables the stencil test for that face (`MTLDepthStencil.h`: the
//! properties default to nil); a disabled test is one that always passes and never writes, which is
//! [`Face::DISABLED`]. Metal keeps a read and a write mask per face, WebGPU one pair for both faces
//! (`GPUDepthStencilState.stencilReadMask`/`stencilWriteMask`); a draw whose faces need different masks or reference
//! values is drawn one face at a time (see `encoder`).

use std::any::Any;
use std::rc::Rc;

use webmetal_wire::{
    CreateDepthStencilStateReply, DepthStencilDescriptor, Handle, Reply, Request, StencilDescriptor, opcode,
};

use crate::embed::{Embedder, TaskId};
use crate::exec::{ControlService, Entry, Served, ServiceCx};
use crate::gpu::Api;
use crate::gpu::idl::StencilFaceState;

use super::error::{RenderError, Result};
use super::metal::{CompareFunction, StencilOperation};

const CALL: &str = "-[MTLDevice newDepthStencilStateWithDescriptor:]";

/// The stencil state of one face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Face {
    pub compare: CompareFunction,
    pub fail: StencilOperation,
    pub depth_fail: StencilOperation,
    pub pass: StencilOperation,
    pub read_mask: u32,
    pub write_mask: u32,
}

impl Face {
    /// A face whose stencil test is disabled: it always passes and never changes the stored value.
    pub const DISABLED: Face = Face {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
        read_mask: 0xffff_ffff,
        write_mask: 0xffff_ffff,
    };

    fn from_descriptor(d: &StencilDescriptor) -> Result<Face> {
        Ok(Face {
            compare: CompareFunction::from_value(d.stencil_compare_function, CALL)?,
            fail: StencilOperation::from_value(d.stencil_failure_operation, CALL)?,
            depth_fail: StencilOperation::from_value(d.depth_failure_operation, CALL)?,
            pass: StencilOperation::from_value(d.depth_stencil_pass_operation, CALL)?,
            read_mask: d.read_mask,
            write_mask: d.write_mask,
        })
    }

    fn ops(&self) -> [StencilOperation; 3] {
        [self.fail, self.depth_fail, self.pass]
    }

    /// Can the face change the stored stencil value?
    pub fn writes(&self) -> bool {
        self.write_mask != 0 && self.ops().iter().any(|&o| o != StencilOperation::Keep)
    }

    /// Does the face have any effect: a test that can fail or an operation that can write?
    pub fn active(&self) -> bool {
        self.compare != CompareFunction::Always || self.writes()
    }

    /// Does the outcome depend on the reference value? The comparison reads it unless it is `Always` or `Never`;
    /// `Replace` writes it.
    pub fn uses_reference(&self) -> bool {
        self.compare.reads_values() || (self.writes() && self.ops().contains(&StencilOperation::Replace))
    }

    /// Does the outcome depend on the read mask?
    pub fn uses_read_mask(&self) -> bool {
        self.compare.reads_values()
    }

    /// Does the outcome depend on the write mask?
    pub fn uses_write_mask(&self) -> bool {
        self.ops().iter().any(|&o| o != StencilOperation::Keep)
    }

    /// The WebGPU face state (operations and comparison; the masks are the pipeline's).
    pub fn gpu(&self) -> StencilFaceState {
        let mut s = StencilFaceState::new();
        s.compare = self.compare.gpu();
        s.fail_op = self.fail.gpu();
        s.depth_fail_op = self.depth_fail.gpu();
        s.pass_op = self.pass.gpu();
        s
    }
}

/// A depth-stencil state object: the payload of a `DepthStencilState` handle.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DepthStencilState {
    pub depth_compare: CompareFunction,
    pub depth_write: bool,
    /// `frontFaceStencil` (`None`: nil, the test is disabled).
    pub front: Option<Face>,
    /// `backFaceStencil` (`None`: nil, the test is disabled).
    pub back: Option<Face>,
}

impl DepthStencilState {
    /// The state a render command encoder starts with, before any `setDepthStencilState:`: the defaults of a new
    /// `MTLDepthStencilDescriptor` (compare `Always`, writes off, no stencil descriptors).
    pub const DEFAULT: DepthStencilState =
        DepthStencilState { depth_compare: CompareFunction::Always, depth_write: false, front: None, back: None };

    pub fn from_descriptor(d: &DepthStencilDescriptor) -> Result<DepthStencilState> {
        Ok(DepthStencilState {
            depth_compare: CompareFunction::from_value(d.depth_compare_function, CALL)?,
            depth_write: d.depth_write_enabled,
            front: d.front_face_stencil.as_ref().map(Face::from_descriptor).transpose()?,
            back: d.back_face_stencil.as_ref().map(Face::from_descriptor).transpose()?,
        })
    }

    /// The state of the front face, a disabled face for nil.
    pub fn front_face(&self) -> Face {
        self.front.unwrap_or(Face::DISABLED)
    }

    /// The state of the back face, a disabled face for nil.
    pub fn back_face(&self) -> Face {
        self.back.unwrap_or(Face::DISABLED)
    }

    /// Does the state test or write depth?
    pub fn uses_depth(&self) -> bool {
        self.depth_compare != CompareFunction::Always || self.depth_write
    }

    /// Does the state test or write stencil on either face?
    pub fn uses_stencil(&self) -> bool {
        self.front_face().active() || self.back_face().active()
    }
}

/// The control service that creates depth-stencil state objects.
pub struct DepthStencilService;

impl DepthStencilService {
    fn create(request: &Request) -> Result<Rc<dyn Any>> {
        match request {
            Request::CreateDepthStencilState(r) => Ok(Rc::new(DepthStencilState::from_descriptor(&r.descriptor)?)),
            other => Err(RenderError::invalid(format!("opcode {:#06x} is not a render request", other.opcode()))),
        }
    }
}

impl<A: Api, E: Embedder> ControlService<A, E> for DepthStencilService {
    fn opcodes(&self) -> Vec<u16> {
        vec![opcode::CREATE_DEPTH_STENCIL_STATE]
    }

    fn serve(&mut self, cx: &mut ServiceCx<'_, A, E>, request: &Request) -> Served {
        match Self::create(request) {
            Ok(payload) => {
                cx.attach(payload);
                Served::Ready(Reply::CreateDepthStencilState(Ok(CreateDepthStencilStateReply {})))
            }
            Err(e) => Served::Ready(
                crate::exec::schema::failure_reply(request.opcode(), e.failure()).expect("a control request"),
            ),
        }
    }

    fn release(&mut self, _: TaskId, _: Handle, _: &Entry) {}

    fn task_exited(&mut self, _: TaskId) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stencil(compare: u64, fail: u64, depth_fail: u64, pass: u64, read: u32, write: u32) -> StencilDescriptor {
        StencilDescriptor {
            stencil_compare_function: compare,
            stencil_failure_operation: fail,
            depth_failure_operation: depth_fail,
            depth_stencil_pass_operation: pass,
            read_mask: read,
            write_mask: write,
        }
    }

    #[test]
    fn nil_faces_are_disabled() {
        let d = DepthStencilDescriptor {
            back_face_stencil: None,
            depth_compare_function: 7,
            depth_write_enabled: false,
            front_face_stencil: None,
            label: None,
        };
        let s = DepthStencilState::from_descriptor(&d).unwrap();
        assert_eq!(s, DepthStencilState::DEFAULT);
        assert!(!s.uses_depth() && !s.uses_stencil());
    }

    #[test]
    fn reference_and_mask_use() {
        // Compare equal reads the reference and the read mask.
        let f = Face::from_descriptor(&stencil(2, 0, 0, 0, 0xef, 0xff)).unwrap();
        assert!(f.uses_reference() && f.uses_read_mask() && !f.uses_write_mask() && f.active());
        // Always with replace writes the reference; a zero write mask writes nothing.
        let f = Face::from_descriptor(&stencil(7, 0, 0, 2, 0xff, 0xff)).unwrap();
        assert!(f.uses_reference() && !f.uses_read_mask() && f.uses_write_mask());
        let f = Face::from_descriptor(&stencil(7, 0, 0, 2, 0xff, 0)).unwrap();
        assert!(!f.uses_reference() && !f.active());
        // Invert writes without the reference.
        let f = Face::from_descriptor(&stencil(7, 0, 0, 5, 0xff, 0xff)).unwrap();
        assert!(!f.uses_reference() && f.writes());
        assert!(Face::from_descriptor(&stencil(8, 0, 0, 0, 0, 0)).is_err());
    }
}
