//! Conversions between the IDL types and webgpu.h's.
//!
//! IDL enum values map to the header's enumerators by name: webgpu.h names each enumerator after the IDL value
//! (`"rgba8unorm-srgb"` is `WGPUTextureFormat_RGBA8UnormSrgb`), so both reduce to the same key once case and
//! punctuation are removed. The tables are built from the enumerators bindgen found in the linked header
//! (`sys::meta::C_ENUMS`), so a value the header does not define is reported, never guessed.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::sys;
use crate::gpu::api::Exception;
use crate::gpu::idl::*;

fn key(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

/// For each enum type of the header: normalized enumerator name to value, and value to normalized name.
struct CEnums {
    by_name: HashMap<(&'static str, String), u32>,
}

fn c_enums() -> &'static CEnums {
    static T: OnceLock<CEnums> = OnceLock::new();
    T.get_or_init(|| {
        let mut by_name = HashMap::new();
        for (ty, name, value) in sys::meta::C_ENUMS {
            by_name.insert((*ty, key(name)), *value);
        }
        CEnums { by_name }
    })
}

/// The webgpu.h enum type an IDL enum maps to.
pub(crate) trait CEnum: IdlEnum {
    const C_TYPE: &'static str;
    fn table() -> &'static OnceLock<Vec<Option<u32>>>;
}

macro_rules! c_enum {
    ($($idl:ty => $c:literal),* $(,)?) => {
        $(
            impl CEnum for $idl {
                const C_TYPE: &'static str = $c;
                fn table() -> &'static OnceLock<Vec<Option<u32>>> {
                    static T: OnceLock<Vec<Option<u32>>> = OnceLock::new();
                    &T
                }
            }
        )*
    };
}

c_enum! {
    FeatureName => "WGPUFeatureName",
    PowerPreference => "WGPUPowerPreference",
    BufferMapState => "WGPUBufferMapState",
    TextureDimension => "WGPUTextureDimension",
    TextureViewDimension => "WGPUTextureViewDimension",
    TextureAspect => "WGPUTextureAspect",
    TextureFormat => "WGPUTextureFormat",
    AddressMode => "WGPUAddressMode",
    FilterMode => "WGPUFilterMode",
    MipmapFilterMode => "WGPUMipmapFilterMode",
    CompareFunction => "WGPUCompareFunction",
    BufferBindingType => "WGPUBufferBindingType",
    SamplerBindingType => "WGPUSamplerBindingType",
    TextureSampleType => "WGPUTextureSampleType",
    StorageTextureAccess => "WGPUStorageTextureAccess",
    CompilationMessageType => "WGPUCompilationMessageType",
    PrimitiveTopology => "WGPUPrimitiveTopology",
    FrontFace => "WGPUFrontFace",
    CullMode => "WGPUCullMode",
    BlendFactor => "WGPUBlendFactor",
    BlendOperation => "WGPUBlendOperation",
    StencilOperation => "WGPUStencilOperation",
    IndexFormat => "WGPUIndexFormat",
    VertexFormat => "WGPUVertexFormat",
    VertexStepMode => "WGPUVertexStepMode",
    LoadOp => "WGPULoadOp",
    StoreOp => "WGPUStoreOp",
    QueryType => "WGPUQueryType",
    DeviceLostReason => "WGPUDeviceLostReason",
    ErrorFilter => "WGPUErrorFilter",
}

fn table<E: CEnum>() -> &'static [Option<u32>] {
    E::table().get_or_init(|| {
        let t = c_enums();
        E::values().iter().map(|v| t.by_name.get(&(E::C_TYPE, key(v.idl_str()))).copied()).collect()
    })
}

/// The header's enumerator for an IDL enum value, or `Unsupported` when the header does not define it.
pub(crate) fn to_c<E: CEnum>(v: E) -> Result<u32, Exception> {
    table::<E>()[v.index()].ok_or_else(|| {
        Exception::unsupported(format!(
            "{} \"{}\" has no {} enumerator in {}",
            E::IDL_NAME,
            v.idl_str(),
            E::C_TYPE,
            sys::meta::IMPLEMENTATION
        ))
    })
}

/// The IDL enum value of a header enumerator, if it is one.
pub(crate) fn from_c<E: CEnum>(c: u32) -> Option<E> {
    let t = table::<E>();
    E::values().iter().copied().find(|v| t[v.index()] == Some(c))
}

/// The value of a header enumerator by name (`c_named("WGPUCallbackMode", "AllowProcessEvents")`). The name must
/// exist; every name the binding uses is part of webgpu.h.
pub(crate) fn c_named(ty: &'static str, name: &str) -> u32 {
    match c_enums().by_name.get(&(ty, key(name))) {
        Some(v) => *v,
        None => panic!("{ty}_{name} is not defined by {}", sys::meta::HEADER),
    }
}

/// A `WGPUStringView` over a Rust string. The view borrows the string.
pub(crate) fn sv(s: &str) -> sys::WGPUStringView {
    sys::WGPUStringView { data: s.as_ptr().cast(), length: s.len() }
}

/// The null `WGPUStringView` (an absent nullable string).
pub(crate) fn sv_null() -> sys::WGPUStringView {
    sys::WGPUStringView { data: std::ptr::null(), length: sys::WGPU_STRLEN as usize }
}

/// A Rust string from an output `WGPUStringView` (explicitly sized, never null-terminated-only).
pub(crate) fn from_sv(v: sys::WGPUStringView) -> String {
    if v.data.is_null() || v.length == 0 {
        return String::new();
    }
    let len = if v.length == sys::WGPU_STRLEN as usize {
        // A null-terminated input string echoed back; webgpu.h output strings are always sized, but accept both.
        unsafe { std::ffi::CStr::from_ptr(v.data) }.to_bytes().len()
    } else {
        v.length
    };
    let bytes = unsafe { std::slice::from_raw_parts(v.data.cast::<u8>(), len) };
    String::from_utf8_lossy(bytes).into_owned()
}

pub(crate) fn wbool(b: bool) -> sys::WGPUBool {
    if b { 1 } else { 0 }
}

/// Storage that keeps the nested arrays and strings of a C descriptor alive until the call that reads it returns.
#[derive(Default)]
pub(crate) struct Arena {
    blocks: Vec<Box<dyn std::any::Any>>,
}

impl Arena {
    /// Move a vector into the arena and return a pointer to its elements (null for an empty vector).
    pub(crate) fn slice<T: 'static>(&mut self, v: Vec<T>) -> (*const T, usize) {
        if v.is_empty() {
            return (std::ptr::null(), 0);
        }
        let b = v.into_boxed_slice();
        let p = b.as_ptr();
        let n = b.len();
        self.blocks.push(Box::new(b));
        (p, n)
    }

    /// Move a value into the arena and return a pointer to it.
    pub(crate) fn one<T: 'static>(&mut self, v: T) -> *const T {
        let b = Box::new(v);
        let p: *const T = &*b;
        self.blocks.push(b);
        p
    }

    /// Move a value into the arena and return a mutable pointer to it (for chained structures).
    pub(crate) fn one_mut<T: 'static>(&mut self, v: T) -> *mut T {
        let mut b = Box::new(v);
        let p: *mut T = &mut *b;
        self.blocks.push(b);
        p
    }

    /// Keep a string alive and return a view of it.
    pub(crate) fn str(&mut self, s: &str) -> sys::WGPUStringView {
        let b: Box<str> = s.into();
        let v = sys::WGPUStringView { data: b.as_ptr().cast(), length: b.len() };
        self.blocks.push(Box::new(b));
        v
    }
}

/// A chained-structure header.
pub(crate) fn chain(stype: &str, next: *mut sys::WGPUChainedStruct) -> sys::WGPUChainedStruct {
    sys::WGPUChainedStruct { next, sType: c_named("WGPUSType", stype) as sys::WGPUSType }
}
