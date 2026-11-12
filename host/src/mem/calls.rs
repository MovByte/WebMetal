//! How the memory module reaches WebGPU: through the executor's error-scope plumbing ([`crate::exec::Gpu`]) inside
//! service methods, or straight to a device ([`Direct`]) outside the executor (setup and tools).
//!
//! Every WebGPU call that can raise a device-timeline error goes through [`Calls::call`], so the executor can attribute
//! what its error scopes capture to the request or command buffer that caused it.

use crate::exec::Gpu;
use crate::gpu::{Api, Device as _};

/// A way to make WebGPU calls.
pub trait Calls<A: Api> {
    /// Make a call that may raise a device-timeline error. `op` names the IDL operation, `label` the object.
    fn call<T>(&self, op: &'static str, label: &str, f: impl FnOnce(&A::Device) -> T) -> T;

    /// The device, for reads that raise no device-timeline error.
    fn device(&self) -> &A::Device;

    /// The device's queue; its operations go through [`Calls::call`].
    fn queue(&self) -> &A::Queue;
}

impl<A: Api> Calls<A> for Gpu<'_, A> {
    fn call<T>(&self, op: &'static str, label: &str, f: impl FnOnce(&A::Device) -> T) -> T {
        Gpu::call(self, op, label, f)
    }

    fn device(&self) -> &A::Device {
        Gpu::device(self)
    }

    fn queue(&self) -> &A::Queue {
        Gpu::queue(self)
    }
}

/// Calls made straight to a device, outside the executor's error scopes: for setup before the executor runs and for
/// tools and tests that drive the memory module alone. Errors then reach the device's own scopes or its
/// `uncapturederror` handler.
pub struct Direct<A: Api> {
    device: A::Device,
    queue: A::Queue,
}

impl<A: Api> Direct<A> {
    pub fn new(device: &A::Device) -> Direct<A> {
        Direct { device: device.clone(), queue: device.queue() }
    }
}

impl<A: Api> Calls<A> for Direct<A> {
    fn call<T>(&self, _op: &'static str, _label: &str, f: impl FnOnce(&A::Device) -> T) -> T {
        f(&self.device)
    }

    fn device(&self) -> &A::Device {
        &self.device
    }

    fn queue(&self) -> &A::Queue {
        &self.queue
    }
}
