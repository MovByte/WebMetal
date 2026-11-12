//! Device state objects encoders bind: sampler states (`CreateSamplerState`), served as a [`ControlService`].
//! (`CreateResourceGroup` has no service: the A10X driver's `-[AGXA10FamilyDevice newResourceGroupFromResources:count:]`
//! never makes a group, so the guest never sends it.)
//!
//! **Payloads.** The object a request creates is bound in the task's object table with one of these payloads:
//!
//! | Handle kind | Payload |
//! | --- | --- |
//! | sampler state (`CreateSamplerState`) | [`SamplerState<A>`] |
//!
//! **Samplers.** Metal decides whether a sampler compares by how a shader uses it; WebGPU fixes it in the sampler. The
//! payload therefore keeps the WebGPU sampler for ordinary sampling, made at creation so that WebGPU checks it then,
//! and the dictionary with the compare function an encoder needs for a comparison variant. A Metal state with no exact
//! WebGPU form (plan section 6.4: clamp-to-zero, border colors, mirror-clamp-to-edge, unnormalized coordinates, LOD
//! averaging or bias, anisotropy over non-linear filters, a minimum LOD clamp above the maximum, an unmipmapped sampler
//! whose minification and magnification filters differ) is still created, as Metal creates it; the payload records why
//! it has no WebGPU sampler, and a use of it is refused loudly by the encoder that binds it.
//!
//! Values are the iPhoneOS SDK's (`MTLSampler.h`, `MTLDepthStencil.h`): address modes `ClampToEdge` 0,
//! `MirrorClampToEdge` 1, `Repeat` 2, `MirrorRepeat` 3, `ClampToZero` 4, `ClampToBorderColor` 5; min/mag filters
//! `Nearest` 0, `Linear` 1; mip filters `NotMipmapped` 0, `Nearest` 1, `Linear` 2; compare functions `Never` 0,
//! `Less` 1, `Equal` 2, `LessEqual` 3, `Greater` 4, `NotEqual` 5, `GreaterEqual` 6, `Always` 7.

use std::rc::Rc;

use webmetal_wire::{CreateSamplerStateReply, Failure, Handle, Reply, Request, SamplerDescriptor, opcode};

use crate::embed::{Embedder, TaskId};
use crate::exec::schema::failure_reply;
use crate::exec::{ControlService, Entry, Served, ServiceCx};
use crate::gpu::idl::{self, AddressMode, CompareFunction, FilterMode, MipmapFilterMode};
use crate::gpu::num::Float32;
use crate::gpu::{Api, Device};

const SAMPLER_CALL: &str = "-[MTLDevice newSamplerStateWithDescriptor:]";

/// A sampler state.
pub struct SamplerState<A: Api> {
    /// The Metal descriptor as the guest sent it.
    pub descriptor: SamplerDescriptor,
    /// The WebGPU form of the state for sampling without comparison, or why the state has none.
    pub gpu: Result<Gpu<A>, String>,
}

/// The WebGPU form of a sampler state.
pub struct Gpu<A: Api> {
    /// The sampler without a compare function.
    pub sampler: A::Sampler,
    /// Its dictionary; a comparison variant is this with [`Gpu::compare`] set.
    pub dictionary: idl::SamplerDescriptor,
    /// The state's compare function, for a comparison variant.
    pub compare: CompareFunction,
}

/// The service.
#[derive(Default)]
pub struct StateServices;

impl StateServices {
    pub fn new() -> StateServices {
        StateServices
    }
}

fn address_mode(value: u64) -> Result<AddressMode, String> {
    match value {
        0 => Ok(AddressMode::ClampToEdge),
        2 => Ok(AddressMode::Repeat),
        3 => Ok(AddressMode::MirrorRepeat),
        1 => Err("MTLSamplerAddressModeMirrorClampToEdge has no WebGPU address mode".into()),
        4 => Err("MTLSamplerAddressModeClampToZero has no WebGPU address mode".into()),
        5 => Err("MTLSamplerAddressModeClampToBorderColor has no WebGPU address mode".into()),
        v => Err(format!("{v} is not an MTLSamplerAddressMode value")),
    }
}

fn filter(value: u64) -> Result<FilterMode, String> {
    match value {
        0 => Ok(FilterMode::Nearest),
        1 => Ok(FilterMode::Linear),
        v => Err(format!("{v} is not an MTLSamplerMinMagFilter value")),
    }
}

/// Metal's compare function: the same set as WebGPU's.
fn compare(value: u64) -> Result<CompareFunction, String> {
    Ok(match value {
        0 => CompareFunction::Never,
        1 => CompareFunction::Less,
        2 => CompareFunction::Equal,
        3 => CompareFunction::LessEqual,
        4 => CompareFunction::Greater,
        5 => CompareFunction::NotEqual,
        6 => CompareFunction::GreaterEqual,
        7 => CompareFunction::Always,
        v => return Err(format!("{v} is not an MTLCompareFunction value")),
    })
}

/// The WebGPU dictionary of a Metal sampler descriptor, or why it has none.
pub fn dictionary(d: &SamplerDescriptor, label: &str) -> Result<(idl::SamplerDescriptor, CompareFunction), String> {
    if !d.normalized_coordinates {
        return Err("unnormalized coordinates have no WebGPU sampler".into());
    }
    if d.lod_average {
        return Err("LOD averaging has no WebGPU sampler".into());
    }
    if d.lod_bias != 0.0 {
        return Err("an LOD bias has no WebGPU sampler".into());
    }
    let mut w = idl::SamplerDescriptor::new();
    w.label = label.into();
    w.address_mode_u = address_mode(d.s_address_mode)?;
    w.address_mode_v = address_mode(d.t_address_mode)?;
    w.address_mode_w = address_mode(d.r_address_mode)?;
    w.min_filter = filter(d.min_filter)?;
    w.mag_filter = filter(d.mag_filter)?;
    let (mut min_lod, mut max_lod) = (d.lod_min_clamp, d.lod_max_clamp);
    w.mipmap_filter = match d.mip_filter {
        1 => MipmapFilterMode::Nearest,
        2 => MipmapFilterMode::Linear,
        // Sampling level 0 only: exact as a [0, 0] LOD clamp when the filter does not depend on whether the
        // texture is minified, which WebGPU decides from the clamped LOD.
        0 if w.min_filter == w.mag_filter => {
            (min_lod, max_lod) = (0.0, 0.0);
            MipmapFilterMode::Nearest
        }
        0 => return Err("an unmipmapped sampler with different minification and magnification filters".into()),
        v => return Err(format!("{v} is not an MTLSamplerMipFilter value")),
    };
    if !(min_lod >= 0.0 && max_lod >= min_lod) {
        return Err(format!("LOD clamps [{min_lod}, {max_lod}] have no WebGPU form"));
    }
    w.lod_min_clamp = Float32::new(min_lod).map_err(|e| format!("lodMinClamp: {e:?}"))?;
    w.lod_max_clamp = Float32::new(max_lod).map_err(|e| format!("lodMaxClamp: {e:?}"))?;
    if !(1..=16).contains(&d.max_anisotropy) {
        return Err(format!("a maximum anisotropy of {} has no WebGPU form", d.max_anisotropy));
    }
    if d.max_anisotropy > 1
        && (w.min_filter != FilterMode::Linear
            || w.mag_filter != FilterMode::Linear
            || w.mipmap_filter != MipmapFilterMode::Linear)
    {
        return Err("anisotropic filtering over a filter that is not linear has no WebGPU sampler".into());
    }
    w.max_anisotropy = d.max_anisotropy as u16;
    Ok((w, compare(d.compare_function)?))
}

impl<A: Api, E: Embedder + 'static> ControlService<A, E> for StateServices {
    fn opcodes(&self) -> Vec<u16> {
        vec![opcode::CREATE_SAMPLER_STATE]
    }

    fn serve(&mut self, cx: &mut ServiceCx<'_, A, E>, request: &Request) -> Served {
        match request {
            Request::CreateSamplerState(r) => {
                let label = format!("sampler {:#x}", r.sampler.0.get());
                let gpu = dictionary(&r.descriptor, &label).map(|(dictionary, compare)| {
                    let sampler = cx.gpu().call("GPUDevice.createSampler", &label, |d| d.create_sampler(&dictionary));
                    Gpu { sampler, dictionary, compare }
                });
                cx.attach(Rc::new(SamplerState::<A> { descriptor: r.descriptor.clone(), gpu }));
                Served::Ready(Reply::CreateSamplerState(Ok(CreateSamplerStateReply {})))
            }
            other => {
                let failure = Failure {
                    status: webmetal_wire::status::INVALID,
                    domain: None,
                    code: 0,
                    message: format!("opcode {:#06x} is not a state request ({SAMPLER_CALL})", other.opcode()),
                };
                Served::Ready(failure_reply(other.opcode(), failure).expect("a control request"))
            }
        }
    }

    /// The payload, dropped with the entry, is all a state object holds.
    fn release(&mut self, task: TaskId, handle: Handle, entry: &Entry) {
        let _ = (task, handle, entry);
    }

    fn task_exited(&mut self, task: TaskId) {
        let _ = task;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metal_default() -> SamplerDescriptor {
        // MTLSamplerDescriptor's defaults (V1 sampler.properties): nearest filters, not mipmapped, anisotropy 1,
        // clamp to edge, normalized coordinates, LOD clamps [0, FLT_MAX], compare never.
        SamplerDescriptor {
            border_color: 0,
            border_color_spi: 0,
            compare_function: 0,
            custom_border_color_value_0: 0,
            custom_border_color_value_1: 0,
            custom_border_color_value_2: 0,
            custom_border_color_value_3: 0,
            force_resource_index: false,
            force_seams_on_cubemap_filtering: false,
            label: None,
            lod_average: false,
            lod_bias: 0.0,
            lod_max_clamp: f32::MAX,
            lod_min_clamp: 0.0,
            mag_filter: 0,
            max_anisotropy: 1,
            min_filter: 0,
            mip_filter: 0,
            normalized_coordinates: true,
            pixel_format: 0,
            r_address_mode: 0,
            resource_index: 0,
            s_address_mode: 0,
            support_argument_buffers: false,
            t_address_mode: 0,
        }
    }

    #[test]
    fn default_sampler_is_level_zero_nearest() {
        let (w, c) = dictionary(&metal_default(), "s").unwrap();
        assert_eq!((w.lod_min_clamp.get(), w.lod_max_clamp.get()), (0.0, 0.0));
        assert_eq!(w.mipmap_filter, MipmapFilterMode::Nearest);
        assert_eq!(c, CompareFunction::Never);
    }

    #[test]
    fn modes_without_a_webgpu_form_are_refused() {
        for (field, value) in [("s", 1), ("t", 4), ("r", 5)] {
            let mut d = metal_default();
            match field {
                "s" => d.s_address_mode = value,
                "t" => d.t_address_mode = value,
                _ => d.r_address_mode = value,
            }
            assert!(dictionary(&d, "s").is_err(), "{field} {value}");
        }
        let mut d = metal_default();
        d.normalized_coordinates = false;
        assert!(dictionary(&d, "s").is_err());
        let mut d = metal_default();
        d.max_anisotropy = 4;
        assert!(dictionary(&d, "s").is_err(), "anisotropy over nearest filters");
        let mut d = metal_default();
        (d.min_filter, d.mag_filter) = (1, 0);
        assert!(dictionary(&d, "s").is_err(), "unmipmapped with different filters");
        let mut d = metal_default();
        (d.mip_filter, d.lod_min_clamp, d.lod_max_clamp) = (1, 4.0, 2.0);
        assert!(dictionary(&d, "s").is_err(), "minimum LOD clamp above the maximum");
    }
}
