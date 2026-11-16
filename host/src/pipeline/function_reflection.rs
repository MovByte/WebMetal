//! Function reflection: the reply of the A10X's compiler service to a reflection-only request (`ReflectFunction`).
//!
//! **The request.** `-[_MTLFunction reflectionWithOptions:binaryArchives:]` (Metal 21H461 0x187a9fbbc), which
//! `newArgumentEncoderWithBufferIndex:` uses with options 3, sends `-[MTLCompiler reflectionWithFunction:options:sync:
//! binaryArchives:completionHandler:]` (0x187a9ff5c) to the device's compiler. The compiler builds a function request
//! with pipeline options `(options & 0x20200000) | 0x10003` (0x187a9fcb0) and sends it to MTLCompilerService with
//! `reflectionOnly` set (`compileFunctionRequestInternal:frameworkLinking:linkDataSize:reflectionOnly:
//! completionHandler:`; visible and intersection functions take the stateless request instead). The request's flags
//! (`MTLGetCompilerOptions`, 0x187a9acdc) carry bit 0 (reflection) and bit 1 (reflection only); bit 30, which defers
//! reflection after the framework passes, is never set from these options.
//!
//! **The service.** MTLCompilerService runs MTLCompiler.framework (343.21). The reflection-only request reads the
//! module (`MTLCompilerObject::readModuleFromBinaryRequest`, MTLCompiler 0x1c3e0938c) and, before any framework pass,
//! generates the reflection (`MTLCompilerObject::runFrameworkPasses` 0x1c3e08bb4 calls `generateReflection`
//! 0x1c3e0b1dc), which replies with a `CompilerOutputReflectionBlock` (0x1c3e01698) and nothing else. The block holds
//! the legacy reflection `MTLGenerateReflectionDataPass::generateReflectionData` (0x1c3dfd044) writes; the AIR
//! (flatbuffer) reflection is only written after `loadModule`, which a reflection-only request of an AIR module never
//! reaches, and plugin data is empty. Metal hands the block to `-[MTLFunctionReflectionInternal initWithDevice:
//! reflectionData:functionType:options:]` (0x187abe160), which extracts the legacy section (`_MTLNewReflectionData`
//! 0x187abe3d0) and parses it with the legacy deserializers (`ShouldCreateAIRVersion` 0x187abe8f8 picks them for data
//! that starts with `MTLPSBIN`).
//!
//! **What is produced.** This module writes the block from the function's AIR as the service does: the argument records
//! of `parseArgumentMetadata` (0x1c3dfeadc) in the module's argument order, built-ins after them, the struct layouts of
//! `parseStructTypeMetadata` (0x1c3df9d50), and the sections of `generateReflectionData` for kernel and fragment
//! functions. Every value is taken from the AIR metadata by the rules of those functions; the type-name tables are the
//! service's own (`bufferTypeMap` and `textureTypeMap`, below).
//!
//! What it does not produce is refused as not implemented, naming the reason:
//!
//! - vertex functions: the service adds post-vertex-dump instrumentation to a vertex function before reflecting it
//!   (`MTLAddPostVertexDumpingToModulePass::addPostVertexDumpingToModule`, 0x1c3e0c30c) and writes its outputs and the
//!   inputs `MTLGenerateReflectionDataPass::inferInputsFromModule` finds; neither is reproduced;
//! - visible, intersection, mesh and object functions (other requests, other readers);
//! - argument kinds whose records carry data the host does not model (imageblocks, payloads, meshes, `air.constant`
//!   and `air.byval` arguments, function pointers), buffers of pixel-format types, `air.global_bindings`, and an
//!   argument named `trace.buffer` (which the service lists or drops by a request flag);
//! - modules whose AIR upgrade would change what the reflection reads (below).
//!
//! **The AIR upgrade.** The service upgrades every module it reads from AIR bitcode before it reflects it:
//! `readModuleFromBinaryRequest` calls `MTLCompilerObject::upgradeAIRModule` (0x1c3e09578, 0x1c3e09964) with the
//! request's AIR version (request bytes 0x30-0x37, `getReadParametersFromRequest` 0x1c3e09340), and (0, 0) there means
//! AIR 2.6 (0x1c3e099fc). Metal sends (0, 0) for this request unless the app sets `MTL_FORCE_AIR_VERSION` or
//! `MTL_FORCE_LANGUAGE_VERSION` (`_getForcedAIRVersion`'s block 0x187b84e7c): `MTLCompilerFunctionRequest::
//! serializedRequest` (Metal 0x187ac3a58-0x187ac3aa0) sends (0, 0) when the request's LLVM version equals its argument,
//! and here both are 0 (`compileFunctionRequestInternal` stores 0 in the request, 0x187a99450, which
//! `_MTLGetDefaultLLVMVersion` 0x187ac506c also answers; `XPCCompilerConnection::BuildRequestInternal` passes 0,
//! 0x187ac26ec). `MTLUpgradeAIRModule` (libGPUCompiler 0x1c4094888) calls
//! `llvm::air::upgradeAIRModule` (libGPUCompilerImpl 0x1c43a9b08) with that version. It runs one step for each of AIR
//! 1.8, 1.11 and 2.0-2.6 that is above the module's `air.version` and not above the target, then, for a module at 2.6
//! or later, rewrites the calls to AIR built-ins declared with fewer operands than the 2.6 tables give them
//! (0x1c43b1ca4-0x1c43b21d4), and for every module adjusts LLVM attributes and `air.compile_options`
//! (0x1c43b21d8-0x1c43b2cf8). Its built-in rewrites keep every operand of the old call, bitcast where the new
//! parameter's type differs (0x1c43b3150 and its call rewrite 0x1c43b333c, with the operand lists of 0x1c43b4b20 and
//! 0x1c43b608c). The reflection reads the entry point's argument metadata, `air.struct_type_info`, whether each argument
//! has uses, and the module's data layout (`parseStructTypeMetadata` places members in their LLVM struct by it,
//! 0x1c3df9df8); it reads no attribute, no compile option, no triple and no built-in declaration. So:
//!
//! - an AIR 2.6 module runs no step, and the rest changes nothing the reflection reads: its own metadata is reflected;
//! - an AIR 2.5 module runs the 2.6 step (0x1c43b17fc-0x1c43b1ca0): the triple's OS version, the data layout
//!   (0x1c43b3034, the AIR layout of the triple's architecture), operand-keeping built-in rewrites and `air.version`.
//!   Its own metadata is reflected when the module is air64 with the air64 layout that step writes, which then leaves
//!   the layout as it was; any other 2.5 module is refused;
//! - a module below AIR 2.5 runs the steps of 1.8-2.5, which rewrite entry-point argument metadata (argument type
//!   names, built-in roles, `air.arg_unused`, `air.thread_execution_width` and `air.threads_per_simdgroup`). They are
//!   not reproduced, and such modules are refused.
//!
//! A forced version only decides which of the steps up to 2.6 run, and for the modules reflected here those steps
//! change nothing the reflection reads, so the reply does not depend on it.

use std::collections::HashMap;
use std::rc::Rc;

use airwgsl::air::{Access, AirModule, Argument, EntryPoint, IndirectArgument, Location, Role, Slot, StructInfo};
use airwgsl::ir::Value;
use webmetal_wire::Failure;

use super::registry::{Function, function_type};
use crate::exec::failure;

/// The block header's first two words (MTLCompiler 0x1c3eb1990), which `_MTLNewReflectionData` checks the first of.
const BLOCK_MAGIC: u64 = 0x8_ef13_c710;
const BLOCK_VERSION: u64 = 0x134_b112;
/// The block header's size: the legacy reflection starts after it.
const BLOCK_HEADER: usize = 0x50;

/// The data layout the AIR upgrade gives a module whose triple's architecture is not air32 (libGPUCompilerImpl
/// 0x1c43b3034: `setDataLayout` of the 0xd7-byte string at 0x1c48256f2).
const AIR64_DATA_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-\
v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-\
v1024:1024:1024-n8:16:32";

/// The legacy reflection's magic and the version `generateReflectionData` writes (0x1c3dfde60-0x1c3dfde8c).
const LEGACY_MAGIC: &[u8; 8] = b"MTLPSBIN";
const LEGACY_VERSION: u32 = 0x20000;

/// `MTLArgumentData` word 0 (`parseArgumentMetadata`; `serializeArguments` 0x1c3dfbd78 writes bits 4-11 and bit 3).
mod flag {
    /// An argument-table argument.
    pub const ARGUMENT: u32 = 1 << 0;
    /// A stage input (vertex, stage-in, patch or fragment input).
    pub const INPUT: u32 = 1 << 1;
    /// A built-in in the service's built-in set; its record has no index.
    pub const BUILTIN: u32 = 1 << 2;
    pub const ACTIVE: u32 = 1 << 3;
    pub const KIND_SHIFT: u32 = 4;
    pub const ACCESS_SHIFT: u32 = 9;
    /// A struct whose first non-struct member is an argument-buffer resource.
    pub const ARGUMENT_BUFFER: u32 = 1 << 12;
}

/// The record kinds (word 0 bits 4-8) this module writes, and the readers' kinds for them
/// (`deserializeArguments`, Metal 0x187a94cec).
mod kind {
    pub const CONSTANT_BUFFER: u32 = 1;
    pub const DEVICE_BUFFER: u32 = 2;
    pub const THREADGROUP: u32 = 3;
    pub const TEXTURE: u32 = 4;
    pub const SAMPLER: u32 = 5;
    pub const BUILTIN: u32 = 6;
    pub const INPUT: u32 = 7;
    pub const INDIRECT_CONSTANT: u32 = 8;
    pub const COMPUTE_PIPELINE_STATE: u32 = 11;
    pub const RENDER_PIPELINE_STATE: u32 = 12;
    pub const COMMAND_BUFFER: u32 = 13;
    pub const VISIBLE_FUNCTION_TABLE: u32 = 16;
    pub const INTERSECTION_FUNCTION_TABLE: u32 = 17;
    pub const PRIMITIVE_ACCELERATION_STRUCTURE: u32 = 18;
    pub const INSTANCE_ACCELERATION_STRUCTURE: u32 = 19;
}

/// `MTLBindingAccess` as word 0 bits 9-11.
mod access {
    pub const READ_ONLY: u32 = 0;
    pub const READ_WRITE: u32 = 1;
    pub const WRITE_ONLY: u32 = 2;
}

/// The struct member data types `parseStructTypeMetadata`'s member visitor (0x1c3dfa2c0) gives a member that is not a
/// plain value: an array, a struct, an argument-buffer resource.
const ARRAY_TYPE: u32 = 2;
const STRUCT_TYPE: u32 = 1;
const INDIRECT_ARGUMENT_TYPE: u32 = 0x39;

/// `bufferTypeMap()` (MTLCompiler, built by `$_0` at 0x1c3e23964): a type-name token's (qualifier, `MTLDataType`,
/// alignment, size). A qualifier token is skipped. The 172 entries in the order the initializer lists them.
#[rustfmt::skip]
const BUFFER_TYPES: [(&str, u32, u32, u32, u32); 172] = [
    ("const", 1, 0, 0, 0), ("global", 1, 0, 0, 0), ("local", 1, 0, 0, 0), ("constant", 1, 0, 0, 0),
    ("__global", 1, 0, 0, 0), ("__constant", 1, 0, 0, 0), ("float", 0, 3, 4, 4), ("float2", 0, 4, 8, 8),
    ("float3", 0, 5, 16, 16), ("float4", 0, 6, 16, 16), ("packed_float", 0, 3, 4, 4), ("packed_float2", 0, 4, 4, 8),
    ("packed_float3", 0, 5, 4, 12), ("packed_float4", 0, 6, 4, 16), ("half", 0, 16, 2, 2), ("half2", 0, 17, 4, 4),
    ("half3", 0, 18, 8, 8), ("half4", 0, 19, 8, 8), ("packed_half", 0, 16, 2, 2), ("packed_half2", 0, 17, 2, 4),
    ("packed_half3", 0, 18, 2, 6), ("packed_half4", 0, 19, 2, 8), ("bfloat", 0, 121, 2, 2), ("bfloat2", 0, 122, 4, 4),
    ("bfloat3", 0, 123, 8, 8), ("bfloat4", 0, 124, 8, 8), ("packed_bfloat", 0, 121, 2, 2),
    ("packed_bfloat2", 0, 122, 2, 4), ("packed_bfloat3", 0, 123, 2, 6), ("packed_bfloat4", 0, 124, 2, 8),
    ("int", 0, 29, 4, 4), ("int2", 0, 30, 8, 8), ("int3", 0, 31, 16, 16), ("int4", 0, 32, 16, 16),
    ("packed_int", 0, 29, 4, 4), ("packed_int2", 0, 30, 4, 8), ("packed_int3", 0, 31, 4, 12),
    ("packed_int4", 0, 32, 4, 16), ("uint", 0, 33, 4, 4), ("uint2", 0, 34, 8, 8), ("uint3", 0, 35, 16, 16),
    ("uint4", 0, 36, 16, 16), ("packed_uint", 0, 33, 4, 4), ("packed_uint2", 0, 34, 4, 8),
    ("packed_uint3", 0, 35, 4, 12), ("packed_uint4", 0, 36, 4, 16), ("short", 0, 37, 2, 2), ("short2", 0, 38, 4, 4),
    ("short3", 0, 39, 8, 8), ("short4", 0, 40, 8, 8), ("packed_short", 0, 37, 2, 2), ("packed_short2", 0, 38, 2, 4),
    ("packed_short3", 0, 39, 2, 6), ("packed_short4", 0, 40, 2, 8), ("ushort", 0, 41, 2, 2), ("ushort2", 0, 42, 4, 4),
    ("ushort3", 0, 43, 8, 8), ("ushort4", 0, 44, 8, 8), ("packed_ushort", 0, 41, 2, 2),
    ("packed_ushort2", 0, 42, 2, 4), ("packed_ushort3", 0, 43, 2, 6), ("packed_ushort4", 0, 44, 2, 8),
    ("char", 0, 45, 1, 1), ("char2", 0, 46, 2, 2), ("char3", 0, 47, 4, 4), ("char4", 0, 48, 4, 4),
    ("packed_char", 0, 45, 1, 1), ("packed_char2", 0, 46, 1, 2), ("packed_char3", 0, 47, 1, 3),
    ("packed_char4", 0, 48, 1, 4), ("uchar", 0, 49, 1, 1), ("uchar2", 0, 50, 2, 2), ("uchar3", 0, 51, 4, 4),
    ("uchar4", 0, 52, 4, 4), ("packed_uchar", 0, 49, 1, 1), ("packed_uchar2", 0, 50, 1, 2),
    ("packed_uchar3", 0, 51, 1, 3), ("packed_uchar4", 0, 52, 1, 4), ("long", 0, 81, 8, 8), ("long2", 0, 82, 16, 16),
    ("long3", 0, 83, 32, 32), ("long4", 0, 84, 32, 32), ("packed_long", 0, 81, 8, 8), ("packed_long2", 0, 82, 8, 16),
    ("packed_long3", 0, 83, 8, 24), ("packed_long4", 0, 84, 8, 32), ("ulong", 0, 85, 8, 8),
    ("ulong2", 0, 86, 16, 16), ("ulong3", 0, 87, 32, 32), ("ulong4", 0, 88, 32, 32), ("packed_ulong", 0, 85, 8, 8),
    ("packed_ulong2", 0, 86, 8, 16), ("packed_ulong3", 0, 87, 8, 24), ("packed_ulong4", 0, 88, 8, 32),
    ("double", 0, 89, 8, 8), ("double2", 0, 90, 16, 16), ("double3", 0, 91, 32, 32), ("double4", 0, 92, 32, 32),
    ("packed_double", 0, 89, 8, 8), ("packed_double2", 0, 90, 8, 16), ("packed_double3", 0, 91, 8, 24),
    ("packed_double4", 0, 92, 8, 32), ("float8", 0, 93, 32, 32), ("float16", 0, 94, 64, 64),
    ("half8", 0, 95, 16, 16), ("half16", 0, 96, 32, 32), ("bfloat8", 0, 125, 16, 16), ("bfloat16", 0, 126, 32, 32),
    ("int8", 0, 97, 32, 32), ("int16", 0, 98, 64, 64), ("uint8", 0, 99, 32, 32), ("uint16", 0, 100, 64, 64),
    ("short8", 0, 101, 16, 16), ("short16", 0, 102, 32, 32), ("ushort8", 0, 103, 16, 16),
    ("ushort16", 0, 104, 32, 32), ("char8", 0, 105, 8, 8), ("char16", 0, 106, 16, 16), ("uchar8", 0, 107, 8, 8),
    ("uchar16", 0, 108, 16, 16), ("long8", 0, 109, 64, 64), ("long16", 0, 110, 128, 128),
    ("ulong8", 0, 111, 64, 64), ("ulong16", 0, 112, 128, 128), ("double8", 0, 113, 64, 64),
    ("double16", 0, 114, 128, 128), ("bool8", 0, 119, 8, 8), ("bool16", 0, 120, 16, 16), ("unknown", 0, 0, 0, 0),
    ("bool", 0, 53, 1, 1), ("bool2", 0, 54, 2, 2), ("bool3", 0, 55, 4, 4), ("bool4", 0, 56, 4, 4),
    ("packed_bool", 0, 53, 1, 1), ("packed_bool2", 0, 54, 1, 2), ("packed_bool3", 0, 55, 1, 3),
    ("packed_bool4", 0, 56, 1, 4), ("float2x2", 0, 7, 8, 16), ("float2x3", 0, 8, 16, 32), ("float2x4", 0, 9, 16, 32),
    ("float3x2", 0, 10, 8, 24), ("float3x3", 0, 11, 16, 48), ("float3x4", 0, 12, 16, 48), ("float4x2", 0, 13, 8, 32),
    ("float4x3", 0, 14, 16, 64), ("float4x4", 0, 15, 16, 64), ("half2x2", 0, 20, 4, 8), ("half2x3", 0, 21, 8, 16),
    ("half2x4", 0, 22, 8, 16), ("half3x2", 0, 23, 4, 12), ("half3x3", 0, 24, 8, 24), ("half3x4", 0, 25, 8, 24),
    ("half4x2", 0, 26, 4, 16), ("half4x3", 0, 27, 8, 32), ("half4x4", 0, 28, 8, 32), ("bfloat2x2", 0, 127, 4, 8),
    ("bfloat2x3", 0, 128, 8, 16), ("bfloat2x4", 0, 129, 8, 16), ("bfloat3x2", 0, 130, 4, 12),
    ("bfloat3x3", 0, 131, 8, 24), ("bfloat3x4", 0, 132, 8, 24), ("bfloat4x2", 0, 133, 4, 16),
    ("bfloat4x3", 0, 134, 8, 32), ("bfloat4x4", 0, 135, 8, 32), ("void", 0, 0, 0, 0),
    ("indirect_command_buffer", 0, 80, 0, 0), ("visible_function_table", 0, 115, 0, 0),
    ("intersection_function_table", 0, 116, 0, 0), ("primitive_acceleration_structure", 0, 117, 0, 0),
    ("instance_acceleration_structure", 0, 118, 0, 0), ("mesh_grid_properties", 0, 139, 0, 0), ("mesh", 0, 140, 0, 0),
];

/// `pixelFormatTypeMap()` (built by `$_2` at 0x1c3e29aa4): the pixel-format type names. A type name that reaches one
/// is resolved through a second table this module does not carry, so it is refused.
const PIXEL_FORMAT_TYPES: [&str; 16] = [
    "r8unorm",
    "r8snorm",
    "r16unorm",
    "r16snorm",
    "rg8unorm",
    "rg8snorm",
    "rg16unorm",
    "rg16snorm",
    "rgba8unorm",
    "srgba8unorm",
    "rgba8snorm",
    "rgba16unorm",
    "rgba16snorm",
    "rgb10a2",
    "rg11b10f",
    "rgb9e5",
];

/// `textureTypeMap()` (built by `$_1` at 0x1c3e22bdc): a texture type token's `MTLTextureType` and whether it is a
/// depth texture.
const TEXTURE_TYPES: [(&str, u32, bool); 16] = [
    ("texture1d", 0, false),
    ("texture1d_array", 1, false),
    ("texture2d", 2, false),
    ("texture2d_array", 3, false),
    ("texture3d", 7, false),
    ("texturecube", 5, false),
    ("texturecube_array", 6, false),
    ("depthcube_array", 6, true),
    ("texture2d_ms", 4, false),
    ("texture2d_ms_array", 8, false),
    ("depth2d_ms_array", 8, true),
    ("depth2d", 2, true),
    ("depth2d_array", 3, true),
    ("depthcube", 5, true),
    ("depth2d_ms", 4, true),
    ("texture_buffer", 9, false),
];

/// The service's built-in set (`buildInMetadataSet`, filled at 0x1c3dfeb18): an argument whose role key is one of these
/// is a built-in record with that `builtInType`; any other role the service does not know is dropped.
const BUILTINS: [(&str, u32); 24] = [
    ("air.vertex_id", 1),
    ("air.instance_id", 2),
    ("air.base_vertex", 3),
    ("air.base_instance", 4),
    ("air.thread_position_in_grid", 5),
    ("air.threads_per_grid", 6),
    ("air.thread_position_in_threadgroup", 7),
    ("air.thread_index_in_threadgroup", 8),
    ("air.threads_per_threadgroup", 9),
    ("air.threadgroup_position_in_grid", 10),
    ("air.threadgroups_per_grid", 11),
    ("air.thread_execution_width", 12),
    ("air.patch_id", 13),
    ("air.position_in_patch", 14),
    ("air.stage_in_grid_origin", 15),
    ("air.stage_in_grid_size", 16),
    ("air.thread_index_in_quadgroup", 17),
    ("air.thread_index_in_simdgroup", 18),
    ("air.threads_per_quadgroup", 19),
    ("air.threads_per_simdgroup", 20),
    ("air.amplification_id", 21),
    ("air.amplification_count", 22),
    ("air.mesh_grid_properties", 23),
    ("air.mesh", 24),
];

/// The tokens of a type name as `BasicParser::getNextToken` (0x1c3dfb7cc) splits it: runs of characters other than
/// NUL, space, `&`, `*`, `,`, `<` and `>`, each cut to 63 bytes.
fn tokens(name: &str) -> impl Iterator<Item = &[u8]> {
    name.as_bytes()
        .split(|c| matches!(c, 0 | b' ' | b'&' | b'*' | b',' | b'<' | b'>'))
        .filter(|t| !t.is_empty())
        .map(|t| &t[..t.len().min(63)])
}

fn buffer_type(token: &[u8]) -> Option<(u32, u32, u32, u32)> {
    BUFFER_TYPES.iter().find(|e| e.0.as_bytes() == token).map(|e| (e.1, e.2, e.3, e.4))
}

/// `dataTypeInfoFromString` (0x1c3dfb414): the (`MTLDataType`, alignment, size, pixel format, ALU type) of a type
/// name. The first token that is not a qualifier decides; a name with none, or whose deciding token is not a type, is
/// all zeros.
fn data_type_info(name: &str) -> Result<(u32, u32, u32, u32, u32), String> {
    for token in tokens(name) {
        match buffer_type(token) {
            Some((1, ..)) => continue,
            Some((_, data_type, align, size)) => return Ok((data_type, align, size, 0, 0)),
            None if PIXEL_FORMAT_TYPES.iter().any(|p| p.as_bytes() == token) => {
                return Err(format!("the pixel-format type {name}"));
            }
            None => return Ok((0, 0, 0, 0, 0)),
        }
    }
    Ok((0, 0, 0, 0, 0))
}

/// `textureTypeFromString` (0x1c3dfca6c): a texture type name's (`MTLDataType` of its component type, `MTLTextureType`
/// | depth << 16), as `serializeArguments` writes them. The service looks both tokens up without checking that they are
/// in its tables, so a name outside them is refused.
fn texture_type(name: &str) -> Result<(u32, u32), String> {
    let mut t = tokens(name);
    let mut first = t.next();
    if first == Some(b"array") {
        first = t.next();
    }
    let Some((_, texture_type, depth)) = first.and_then(|f| TEXTURE_TYPES.iter().find(|e| e.0.as_bytes() == f)) else {
        return Err(format!("the texture type name {name}"));
    };
    let Some((_, data_type, _, _)) = t.next().and_then(buffer_type) else {
        return Err(format!("the texture type name {name}"));
    };
    Ok((data_type, texture_type | (u32::from(*depth) << 16)))
}

/// A struct layout as the service holds it (`std::vector<MTLStructMemberInfo>`), with the identity the serializer keys
/// it by: a layout reached twice through the same object is written once and referred to after.
#[derive(Debug)]
struct Layout {
    id: u64,
    members: Vec<StructMember>,
}

/// `MTLStructMemberInfo`.
#[derive(Debug)]
struct StructMember {
    name: String,
    offset: u32,
    /// +0x2c: `ARRAY_TYPE` for an array member.
    data_type: u32,
    array_length: u32,
    /// +0x34 and +0x38: the element's data type and the member size, for an array member.
    element_type: u32,
    stride: u32,
    /// +0x40: the argument-buffer resource.
    argument: Option<Record>,
    /// +0x50.
    argument_index: u32,
    render_target: u32,
    raster_order_group: u32,
    pixel_format: u32,
    alu_type: u32,
    /// +0x68 and +0x70: the nested layout.
    nested: Option<Rc<Layout>>,
}

/// `MTLArgumentData`, as far as the serializer reads it.
#[derive(Debug)]
struct Record {
    flags: u32,
    index: u32,
    array_length: u32,
    builtin_type: u32,
    name: String,
    type_name: Option<String>,
    data_type: u32,
    alignment: u32,
    data_size: u32,
    pixel_format: u32,
    alu_type: u32,
    /// +0x48 and +0xa8: the struct layout of `air.struct_type_info`.
    layout: Option<Rc<Layout>>,
}

impl Record {
    fn kind(&self) -> u32 {
        (self.flags >> flag::KIND_SHIFT) & 0x1f
    }
}

/// The state of one function's reflection: the layout objects made so far and, per top-level argument, the layouts of
/// `air.struct_type_info` nodes already decoded (`parseArgumentMetadata` caches them in a map keyed by the node that
/// `generateReflectionData` makes for each argument; uniqued metadata tuples are the same node exactly when their
/// contents are equal).
struct Parser<'a> {
    function: &'a str,
    /// The module's data layout, by which `parseStructTypeMetadata` lays out each struct it parses.
    data_layout: &'a str,
    next_id: u64,
    cache: Vec<(StructInfo, Rc<Layout>)>,
}

/// The value of a `Slot` as `parseArgumentMetadata` reads `air.location_index` operands: a constant integer, or
/// 0xffffffff when the operand is a global the function constants give.
fn slot(s: Slot) -> u32 {
    match s {
        Slot::Value(v) => v,
        Slot::Constant(_) => 0xffff_ffff,
    }
}

/// What the role key of an argument record makes of it (`parseArgumentMetadata`, 0x1c3dff3a0-0x1c3dff8ec and
/// 0x1c3dfff44-0x1c3e00128).
enum RoleClass<'a> {
    /// A buffer of address space `space`, with its access; `air.buffer` and `air.indirect_buffer`.
    Buffer {
        location: Location,
        access: Access,
        space: u32,
        layout: Option<&'a StructInfo>,
    },
    Texture {
        location: Location,
        access: Access,
    },
    Sampler(Location),
    IndirectConstant(Location),
    /// A resource whose record carries nothing but its index (pipeline states, indirect command buffers, function
    /// tables and acceleration structures).
    Plain {
        location: Location,
        kind: u32,
    },
    Builtin(u32),
    Input(Option<Location>),
    /// A role the service's built-in set does not list: no record.
    Dropped,
}

impl<'a> Parser<'a> {
    fn refuse(&self, what: impl std::fmt::Display) -> Failure {
        failure::unimplemented(format!(
            "function reflection of {}: {what}, which the host does not reproduce from the AIR as the A10X's compiler \
             service writes it",
            self.function
        ))
    }

    fn classify(&self, a: &'a Argument) -> Result<RoleClass<'a>, Failure> {
        Ok(match &a.role {
            Role::Buffer(b) | Role::IndirectBuffer(b) => {
                // The address space is the parameter's pointer type's (0x1c3dff2d0); the front end writes the same in
                // `air.address_space`.
                let Some(space) = b.address_space else {
                    return Err(self.refuse("a buffer record without air.address_space"));
                };
                RoleClass::Buffer { location: b.location, access: b.access, space, layout: b.struct_info.as_ref() }
            }
            Role::Texture { location, access, .. } => RoleClass::Texture { location: *location, access: *access },
            Role::Sampler(l) => RoleClass::Sampler(*l),
            Role::IndirectConstant(l) => RoleClass::IndirectConstant(*l),
            Role::ComputePipelineState(l) => RoleClass::Plain { location: *l, kind: kind::COMPUTE_PIPELINE_STATE },
            Role::RenderPipelineState(l) => RoleClass::Plain { location: *l, kind: kind::RENDER_PIPELINE_STATE },
            Role::CommandBuffer(l) => RoleClass::Plain { location: *l, kind: kind::COMMAND_BUFFER },
            Role::VisibleFunctionTable(l, _) => RoleClass::Plain { location: *l, kind: kind::VISIBLE_FUNCTION_TABLE },
            Role::IntersectionFunctionTable(l, _) => {
                RoleClass::Plain { location: *l, kind: kind::INTERSECTION_FUNCTION_TABLE }
            }
            Role::PrimitiveAccelerationStructure(l, _) => {
                RoleClass::Plain { location: *l, kind: kind::PRIMITIVE_ACCELERATION_STRUCTURE }
            }
            Role::InstanceAccelerationStructure(l, _) => {
                RoleClass::Plain { location: *l, kind: kind::INSTANCE_ACCELERATION_STRUCTURE }
            }
            Role::VertexInput(l) | Role::PatchInput(l) => RoleClass::Input(Some(*l)),
            Role::StageIn(l) => RoleClass::Input(*l),
            Role::FragmentInput { .. } => RoleClass::Input(None),
            Role::PatchControlPointInput(_) => {
                return Err(self.refuse("a patch control point input (air.patch_control_point_input)"));
            }
            Role::Imageblock(_) => return Err(self.refuse("an imageblock argument")),
            Role::Payload(_) => return Err(self.refuse("an object payload argument")),
            Role::Mesh(_) | Role::MeshGridProperties => return Err(self.refuse("a mesh argument")),
            Role::VisibleInput => return Err(self.refuse("a stitched visible function input")),
            Role::Builtin { key, .. } => match BUILTINS.iter().find(|b| b.0 == *key) {
                Some((_, t)) => RoleClass::Builtin(*t),
                None => RoleClass::Dropped,
            },
            // `air.render_target` arguments are not in the service's built-in set.
            Role::RenderTarget { .. } => RoleClass::Dropped,
        })
    }

    /// `parseArgumentMetadata` for one record. `used` is whether the LLVM argument has uses (`None` for an
    /// argument-buffer member, which the service parses without a value and reports active).
    fn record(&mut self, a: &'a Argument, used: Option<bool>) -> Result<Option<Record>, Failure> {
        let class = self.classify(a)?;
        if matches!(class, RoleClass::Dropped) {
            return Ok(None);
        }
        let Some(name) = a.name.clone() else {
            return Err(self.refuse("an argument without air.arg_name"));
        };
        // The service reads the type name of textures and built-ins without checking that it is present.
        if matches!(class, RoleClass::Texture { .. } | RoleClass::Builtin(_)) && a.type_name.is_none() {
            return Err(self.refuse("a texture or built-in argument without air.arg_type_name"));
        }
        if name.as_bytes().contains(&0) {
            return Err(self.refuse("an argument name with a NUL byte"));
        }
        let mut r = Record {
            flags: if used.unwrap_or(true) && !a.unused { flag::ACTIVE } else { 0 },
            index: 0,
            array_length: 0,
            builtin_type: 0,
            name,
            type_name: a.type_name.clone(),
            data_type: 0,
            alignment: a.type_align.unwrap_or(0),
            data_size: a.type_size.unwrap_or(0),
            pixel_format: 0,
            alu_type: 0,
            layout: None,
        };
        let location = match &class {
            RoleClass::Buffer { location, .. }
            | RoleClass::Texture { location, .. }
            | RoleClass::Sampler(location)
            | RoleClass::IndirectConstant(location)
            | RoleClass::Plain { location, .. } => Some(*location),
            RoleClass::Input(l) => *l,
            RoleClass::Builtin(_) | RoleClass::Dropped => None,
        };
        if let Some(l) = location {
            r.index = slot(l.index);
            r.array_length = slot(l.count);
        }
        // The value fields of the buffer classes (0x1c3dffe98): a struct's data type is 1 and its size and alignment
        // are the metadata's; otherwise the type name's, where the metadata gives none.
        let has_value = matches!(class, RoleClass::Buffer { .. } | RoleClass::IndirectConstant(_));
        if let RoleClass::Buffer { layout: Some(info), .. } = &class {
            let layout = self.layout_of_argument(info)?;
            if first_non_struct(&layout).is_some_and(|m| m.argument.is_some()) {
                r.flags |= flag::ARGUMENT_BUFFER;
            }
            r.layout = Some(layout);
            r.data_type = STRUCT_TYPE;
        } else if has_value {
            let (data_type, align, size, pixel_format, alu_type) =
                data_type_info(r.type_name.as_deref().unwrap_or("")).map_err(|e| self.refuse(e))?;
            r.data_type = data_type;
            if r.alignment == 0 {
                r.alignment = align;
            }
            if r.data_size == 0 {
                r.data_size = size;
            }
            r.pixel_format = pixel_format;
            r.alu_type = alu_type;
        }
        // Word 0 (0x1c3dfff44): the argument bit, the kind and the access.
        let base = r.flags & !0xff1;
        let access_bits = |a: u32| a << flag::ACCESS_SHIFT;
        let with_kind = |k: u32| (k << flag::KIND_SHIFT) | flag::ARGUMENT;
        r.flags = match class {
            RoleClass::Buffer { access, space, .. } => {
                if space == 3 {
                    base | with_kind(kind::THREADGROUP) | access_bits(access::READ_WRITE)
                } else {
                    let k = if space == 2 { kind::CONSTANT_BUFFER } else { kind::DEVICE_BUFFER };
                    base | with_kind(k) | access_bits(buffer_access(access))
                }
            }
            RoleClass::Texture { access, .. } => base | with_kind(kind::TEXTURE) | access_bits(texture_access(access)),
            RoleClass::Sampler(_) => base | with_kind(kind::SAMPLER),
            // The access of an indirect constant is the parse's access parameter, 0 from every caller.
            RoleClass::IndirectConstant(_) => base | with_kind(kind::INDIRECT_CONSTANT),
            RoleClass::Plain { kind, .. } => base | with_kind(kind),
            RoleClass::Builtin(t) => {
                r.builtin_type = t;
                base | flag::BUILTIN | (kind::BUILTIN << flag::KIND_SHIFT)
            }
            RoleClass::Input(_) => base | flag::INPUT | (kind::INPUT << flag::KIND_SHIFT),
            RoleClass::Dropped => unreachable!("returned above"),
        };
        Ok(Some(r))
    }

    /// The layout of an argument's `air.struct_type_info`: the one already decoded for an equal node within the same
    /// top-level argument, else a new one (0x1c3dffdd8-0x1c3dffe3c). The AIR layer decodes layouts as trees, so a
    /// layout never contains its own node.
    fn layout_of_argument(&mut self, info: &'a StructInfo) -> Result<Rc<Layout>, Failure> {
        if let Some((_, l)) = self.cache.iter().find(|(i, _)| i == info) {
            return Ok(l.clone());
        }
        let id = self.new_id();
        let layout = Rc::new(Layout { id, members: self.members(info)? });
        self.cache.push((info.clone(), layout.clone()));
        Ok(layout)
    }

    fn new_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// `parseStructTypeMetadata`'s member visitor (0x1c3dfa2c0) for each member.
    fn members(&mut self, info: &'a StructInfo) -> Result<Vec<StructMember>, Failure> {
        // parseStructTypeMetadata lays the struct's LLVM type out by the module's data layout (getStructLayout,
        // 0x1c3df9df8) and finds the element at each member's offset, whose type its nested and indirect members are
        // parsed from. Under the air64 layout the frontend writes, that element is the member the metadata describes,
        // as the members are read here; under another the search is not reproduced.
        if self.data_layout != AIR64_DATA_LAYOUT {
            return Err(self.refuse(format_args!(
                "a struct laid out by the data layout {:?} rather than the air64 one",
                self.data_layout
            )));
        }
        let mut out = Vec::with_capacity(info.members.len());
        for m in &info.members {
            if m.name.as_bytes().contains(&0) {
                return Err(self.refuse("a struct member name with a NUL byte"));
            }
            let nested = match &m.nested {
                Some(n) => {
                    let id = self.new_id();
                    Some(Rc::new(Layout { id, members: self.members(n)? }))
                }
                None => None,
            };
            let (value_type, pixel_format, alu_type) = if nested.is_some() {
                (STRUCT_TYPE, 0xffff_ffff, 0xffff_ffff)
            } else if m.indirect.is_some() {
                (INDIRECT_ARGUMENT_TYPE, 0xffff_ffff, 0xffff_ffff)
            } else {
                let (data_type, _, _, pixel_format, alu_type) =
                    data_type_info(&m.type_name).map_err(|e| self.refuse(e))?;
                (data_type, pixel_format, alu_type)
            };
            let (argument, argument_index) = match &m.indirect {
                Some(IndirectArgument::Resource(a)) => {
                    let Some(r) = self.record(a, None)? else {
                        return Err(self.refuse("an argument-buffer member whose role the service drops"));
                    };
                    let index = r.index;
                    (Some(r), index)
                }
                Some(IndirectArgument::Aggregate(index)) => {
                    if nested.is_none() {
                        // The member's type would be 0x39 with no record after it, which the reader cannot parse.
                        return Err(self.refuse("an argument-buffer member with an [[id]] and no struct layout"));
                    }
                    (None, *index)
                }
                None => (None, 0),
            };
            let array = m.array_length != 0;
            out.push(StructMember {
                name: m.name.clone(),
                offset: m.offset,
                data_type: if array { ARRAY_TYPE } else { value_type },
                array_length: m.array_length,
                element_type: if array { value_type } else { 0 },
                stride: if array { m.size } else { 0 },
                argument,
                argument_index,
                render_target: m.render_target.unwrap_or(0xffff_ffff),
                raster_order_group: 0xffff_ffff,
                pixel_format,
                alu_type,
                nested,
            });
        }
        Ok(out)
    }
}

/// The access bits of a buffer: `air.read_write` read-write, `air.write` write-only, otherwise read-only
/// (0x1c3e000d8-0x1c3e000f4).
fn buffer_access(a: Access) -> u32 {
    match a {
        Access::ReadWrite => access::READ_WRITE,
        Access::Write => access::WRITE_ONLY,
        Access::Read | Access::Sample => access::READ_ONLY,
    }
}

/// The access bits of a texture: `air.read_write` read-write, `air.read` or `air.sample` read-only, otherwise
/// write-only (0x1c3dfff98-0x1c3dfffac, 0x1c3e00090).
fn texture_access(a: Access) -> u32 {
    match a {
        Access::ReadWrite => access::READ_WRITE,
        Access::Read | Access::Sample => access::READ_ONLY,
        Access::Write => access::WRITE_ONLY,
    }
}

/// `findNonStructMember` (0x1c3e1a0d0): the first member, depth first, that has no nested members.
fn first_non_struct(l: &Layout) -> Option<&StructMember> {
    for m in &l.members {
        match &m.nested {
            Some(n) if !n.members.is_empty() => {
                if let Some(found) = first_non_struct(n) {
                    return Some(found);
                }
            }
            _ => return Some(m),
        }
    }
    None
}

/// `getStructIndirectArgumentIndexStride` (0x1c3e20238): the argument indices a layout spans.
fn index_stride(l: &Layout) -> u32 {
    let Some(last) = l.members.last() else { return 0 };
    let step = match (&last.nested, &last.argument) {
        (Some(n), _) => index_stride(n),
        (None, Some(a)) => a.array_length.max(1),
        (None, None) => 1,
    };
    last.argument_index.wrapping_add(last.array_length.max(1).wrapping_mul(step))
}

/// The writer: little-endian `u32`s and NUL-terminated strings (the blocks of `generateReflectionData`, 0x1c3dfc6ec
/// and 0x1c3dfc758), and the layouts written so far (`serializeStructInfo`'s map, shared by every section).
struct Writer {
    out: Vec<u8>,
    written: HashMap<u64, u32>,
}

impl Writer {
    fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn string(&mut self, s: &str) {
        self.out.extend_from_slice(s.as_bytes());
        self.out.push(0);
    }

    /// Reserves a `u32` and returns its offset, for an end offset written once the section is.
    fn reserve(&mut self) -> usize {
        self.u32(0);
        self.out.len() - 4
    }

    fn patch(&mut self, at: usize) {
        let end = self.out.len() as u32;
        self.out[at..at + 4].copy_from_slice(&end.to_le_bytes());
    }

    /// `serializeArguments` (0x1c3dfbd78): the count, every record, then every record's layout or 0.
    fn arguments(&mut self, records: &[Record]) -> Result<(), String> {
        self.u32(records.len() as u32);
        for r in records {
            self.string(&r.name);
            self.u32(((r.flags << 6) & 0x200) | ((r.flags >> 4) & 0xff));
            if r.flags & flag::BUILTIN == 0 {
                self.u32(r.index);
                self.u32(r.array_length);
            }
            match r.kind() {
                kind::CONSTANT_BUFFER | kind::DEVICE_BUFFER | kind::THREADGROUP | kind::INDIRECT_CONSTANT => {
                    self.u32(r.data_type);
                    self.u32(r.pixel_format);
                    self.u32(r.alu_type);
                    self.u32(r.alignment);
                    self.u32(r.data_size);
                }
                kind::TEXTURE => {
                    let (data_type, texture) = texture_type(r.type_name.as_deref().unwrap_or(""))?;
                    self.u32(data_type);
                    self.u32(texture);
                }
                kind::BUILTIN => {
                    let (data_type, ..) = data_type_info(r.type_name.as_deref().unwrap_or(""))?;
                    self.u32(data_type);
                    self.u32(r.builtin_type);
                }
                kind::SAMPLER
                | kind::COMPUTE_PIPELINE_STATE
                | kind::RENDER_PIPELINE_STATE
                | kind::COMMAND_BUFFER
                | kind::VISIBLE_FUNCTION_TABLE
                | kind::INTERSECTION_FUNCTION_TABLE
                | kind::PRIMITIVE_ACCELERATION_STRUCTURE
                | kind::INSTANCE_ACCELERATION_STRUCTURE => {}
                k => unreachable!("record kind {k} is refused before it is written"),
            }
        }
        for r in records {
            match &r.layout {
                Some(l) => self.layout(l, r.flags & flag::ARGUMENT_BUFFER != 0)?,
                None => self.u32(0),
            }
        }
        Ok(())
    }

    /// `serializeStructInfo` (0x1c3dfc154).
    fn layout(&mut self, l: &Layout, argument_buffer: bool) -> Result<(), String> {
        if let Some(id) = self.written.get(&l.id) {
            let id = *id;
            self.u32(id);
            return Ok(());
        }
        let id = self.written.len() as u32 + 1;
        self.written.insert(l.id, id);
        self.u32(id);
        self.u32(l.members.len() as u32);
        for m in &l.members {
            self.string(&m.name);
            self.u32(m.offset);
            self.u32(m.argument_index);
            if m.array_length != 0 {
                self.u32(ARRAY_TYPE);
                self.u32(0);
                self.u32(0);
                self.u32(m.array_length);
                self.u32(m.element_type);
                self.u32(m.stride);
                self.u32(match (argument_buffer, &m.nested) {
                    (false, _) => 0,
                    (true, Some(n)) => index_stride(n),
                    (true, None) => 1,
                });
            } else {
                self.u32(m.data_type);
            }
            self.u32(m.pixel_format);
            self.u32(m.alu_type);
            if let Some(n) = &m.nested {
                self.layout(n, argument_buffer)?;
            }
            if let Some(a) = &m.argument {
                self.arguments(std::slice::from_ref(a))?;
            }
            self.u32(m.render_target);
            self.u32(m.raster_order_group);
        }
        Ok(())
    }
}

/// Whether LLVM argument `i` of the entry's function has uses, as `parseArgumentMetadata` asks the argument
/// (0x1c3dff344): an operand of an instruction. Debug intrinsics refer to arguments through metadata, which is no use.
fn argument_used(module: &AirModule, entry: &EntryPoint, i: u32) -> bool {
    module.ir.functions[entry.function as usize].insts.iter().any(|inst| inst.op.operands().contains(&Value::Arg(i)))
}

/// The legacy reflection of a kernel or fragment entry point (`generateReflectionData`, 0x1c3dfd044).
fn legacy(module: &AirModule, entry: &EntryPoint, function_type: u64, describe: &str) -> Result<Vec<u8>, Failure> {
    let refuse = |what: &str| {
        failure::unimplemented(format!(
            "function reflection of {describe}: {what}, which the host does not reproduce from the AIR as the A10X's \
             compiler service writes it"
        ))
    };
    if module.ir.named_metadata.iter().any(|n| n.name == "air.global_bindings") {
        return Err(refuse("air.global_bindings"));
    }
    // A fragment function's output imageblock data (`extractOutputImageBlockDataFromMetadata`, 0x1c3e00dc4: an
    // output record of `air.imageblock_data`) has no output role in the AIR layer, which refuses such modules.
    let staged = function_type == function_type::KERNEL;
    let mut main = Vec::new();
    let mut builtins = Vec::new();
    let mut attributes = Vec::new();
    let mut writer = Writer { out: Vec::new(), written: HashMap::new() };
    let mut next_id = 0;
    for (i, a) in entry.arguments.iter().enumerate() {
        if a.position as usize != i {
            return Err(refuse("argument records out of the function's parameter order"));
        }
        let mut p = Parser { function: describe, data_layout: &module.ir.datalayout, next_id, cache: Vec::new() };
        let record = p.record(a, Some(argument_used(module, entry, i as u32)))?;
        next_id = p.next_id;
        let Some(mut r) = record else { continue };
        if r.flags & flag::ARGUMENT != 0 {
            if r.name == "trace.buffer" {
                return Err(refuse("an argument named trace.buffer"));
            }
            main.push(r);
        } else if r.flags & flag::BUILTIN != 0 {
            r.flags |= flag::ACTIVE;
            builtins.push(r);
        } else if r.flags & flag::INPUT != 0 && staged && r.flags & flag::ACTIVE != 0 {
            attributes.push(r.index);
        }
    }
    let builtin_count = builtins.len() as u32;
    main.extend(builtins);
    let w = &mut writer;
    w.out.extend_from_slice(LEGACY_MAGIC);
    w.u32(LEGACY_VERSION);
    if staged {
        w.u32(attributes.len() as u32);
        for a in &attributes {
            w.u32(*a);
        }
    }
    w.u32(builtin_count);
    let arguments_end = w.reserve();
    w.arguments(&main).map_err(|e| refuse(&e))?;
    w.patch(arguments_end);
    if function_type == function_type::FRAGMENT {
        // No output imageblock data argument.
        w.u32(0);
    }
    if staged {
        // The post-vertex-dump section: its index (0 outside vertex functions) and no outputs.
        w.u32(0);
        let end = w.reserve();
        w.u32(0);
        w.patch(end);
        // Inferred inputs: only a vertex function's are inferred.
        let end = w.reserve();
        w.u32(0);
        w.patch(end);
    }
    // No trace buffer, no plugin data.
    w.u32(0xffff_ffff);
    w.u32(0);
    // Global bindings: none.
    let end = w.reserve();
    w.u32(0);
    w.patch(end);
    Ok(writer.out)
}

/// The compiler service's reply to a reflection-only request for `f`: the `CompilerOutputReflectionBlock` with the
/// legacy reflection and empty AIR reflection and plugin sections. The bytes after the legacy reflection up to the
/// next multiple of 8 are allocated with `malloc` and never written by the service; they are zero here.
pub fn reply(f: &Function) -> Result<Vec<u8>, Failure> {
    let describe = f.describe();
    match f.function_type {
        function_type::KERNEL | function_type::FRAGMENT => {}
        function_type::VERTEX => {
            return Err(failure::unimplemented(format!(
                "function reflection of {describe}: the A10X's compiler service adds post-vertex-dump instrumentation \
                 to a vertex function before reflecting it (MTLAddPostVertexDumpingToModulePass::\
                 addPostVertexDumpingToModule, MTLCompiler 21H461 0x1c3e0c30c) and writes the inputs \
                 inferInputsFromModule finds, which the host does not reproduce"
            )));
        }
        t => {
            return Err(failure::unimplemented(format!(
                "function reflection of {describe}: {} functions are reflected by another request of the compiler \
                 service, which the host does not implement",
                function_type::name(t)
            )));
        }
    }
    let module = super::compute::decode_module(f).map_err(|r| {
        failure::unimplemented(format!("function reflection of {describe}: the AIR layer refuses the module: {r}"))
    })?;
    if let Some(problem) = upgrade_problem(module.air_version, &module.ir.triple, &module.ir.datalayout) {
        return Err(failure::unimplemented(format!("function reflection of {describe}: {problem}")));
    }
    let Some(entry) = module.entry_point(&f.name) else {
        return Err(failure::unimplemented(format!(
            "function reflection of {describe}: the function is not an entry point of its module"
        )));
    };
    if entry.stage.function_type() != f.function_type {
        return Err(failure::invalid(format!(
            "function reflection of {describe}: the module's entry point is a {} function",
            function_type::name(entry.stage.function_type())
        )));
    }
    let legacy = legacy(&module, entry, f.function_type, &describe)?;
    Ok(block(&legacy))
}

/// Why the AIR upgrade the service runs before reflecting (the module comment's **The AIR upgrade**) would change what
/// the reflection reads in a module of this `air.version`, triple and data layout, or `None` when it changes nothing.
fn upgrade_problem(version: Option<(u32, u32, u32)>, triple: &str, data_layout: &str) -> Option<String> {
    const STEPS: &str = "llvm::air::upgradeAIRModule, libGPUCompilerImpl 21H461 0x1c43a9b08";
    match version {
        None => Some(format!("the module has no air.version, which the service's AIR upgrade reads first ({STEPS})")),
        Some((2, 6, _)) => None,
        Some((2, 5, _)) if triple.split('-').next() == Some("air64") && data_layout == AIR64_DATA_LAYOUT => None,
        Some((2, 5, patch)) => Some(format!(
            "the service upgrades an AIR 2.5.{patch} module to 2.6 before reflecting it, and the 2.6 step sets the \
             data layout by which the service places struct members (0x1c43b3034); this module (triple {triple:?}, \
             data layout {data_layout:?}) is not an air64 module that already has the air64 layout, and the host does \
             not reproduce the reflection under the layout the step writes ({STEPS})"
        )),
        Some((major, minor, patch)) if (major, minor) < (2, 5) => Some(format!(
            "the service upgrades an AIR {major}.{minor}.{patch} module to 2.6 before reflecting it, and the upgrade \
             steps of AIR 1.8 to 2.5 rewrite entry-point argument metadata, which the host does not reproduce \
             ({STEPS}, 0x1c43a9b74-0x1c43b17bc)"
        )),
        Some((major, minor, patch)) => Some(format!(
            "an AIR {major}.{minor}.{patch} module is newer than the AIR 2.6 the service upgrades to, and what its \
             upgrade does to such a module is not established ({STEPS})"
        )),
    }
}

/// `CompilerOutputReflectionBlock`'s layout (0x1c3e01698) with only the legacy section: a 0x50-byte header of the
/// magic, the version and the (offset, size) of the legacy reflection, the second and first AIR reflections and the
/// plugin data, each section 8-byte aligned.
fn block(legacy: &[u8]) -> Vec<u8> {
    let end = (BLOCK_HEADER + legacy.len() + 7) & !7;
    let mut out = Vec::with_capacity(end);
    for v in [
        BLOCK_MAGIC,
        BLOCK_VERSION,
        BLOCK_HEADER as u64,
        legacy.len() as u64,
        end as u64,
        0,
        end as u64,
        0,
        end as u64,
        0,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(legacy);
    out.resize(end, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_names_tokenize_as_the_service_splits_them() {
        let t: Vec<&[u8]> = tokens("array<texture2d<float, read>, 4>").collect();
        assert_eq!(t, [&b"array"[..], b"texture2d", b"float", b"read", b"4"]);
        assert_eq!(tokens(&"x".repeat(70)).next().unwrap().len(), 63);
    }

    #[test]
    fn the_buffer_type_table_has_unique_names() {
        for (i, a) in BUFFER_TYPES.iter().enumerate() {
            assert!(BUFFER_TYPES[i + 1..].iter().all(|b| b.0 != a.0), "{} twice", a.0);
        }
    }

    #[test]
    fn data_types_skip_qualifiers_and_stop_at_the_first_type() {
        assert_eq!(data_type_info("float4"), Ok((6, 16, 16, 0, 0)));
        assert_eq!(data_type_info("const constant packed_half3"), Ok((18, 2, 6, 0, 0)));
        // A token that is no type ends the search with nothing (a struct's name, `device`).
        assert_eq!(data_type_info("device float"), Ok((0, 0, 0, 0, 0)));
        assert_eq!(data_type_info("ArgResources"), Ok((0, 0, 0, 0, 0)));
        assert_eq!(data_type_info(""), Ok((0, 0, 0, 0, 0)));
        assert!(data_type_info("rgba8unorm<float4>").is_err());
    }

    #[test]
    fn texture_types_carry_the_component_type_and_depth() {
        assert_eq!(texture_type("texture2d<float, sample>"), Ok((3, 2)));
        assert_eq!(texture_type("depth2d<float, sample>"), Ok((3, 2 | 1 << 16)));
        assert_eq!(texture_type("array<texturecube<half, read>, 2>"), Ok((16, 5)));
        assert_eq!(texture_type("texture2d_array<uint, write>"), Ok((33, 3)));
        assert!(texture_type("texture7d<float>").is_err());
        assert!(texture_type("texture2d<Foo>").is_err());
    }

    #[test]
    fn the_block_header_places_the_legacy_section_after_it() {
        // The header of a 0x217-byte legacy reflection as MTLCompilerService writes it (this Mac's service, macOS
        // 27, uses the same header and layout).
        let b = block(&[0xab; 0x217]);
        assert_eq!(b.len(), 0x268);
        let words: Vec<u64> = b[..0x50].chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(words, [0x8_ef13_c710, 0x134_b112, 0x50, 0x217, 0x268, 0, 0x268, 0, 0x268, 0]);
        assert!(b[0x50..0x267].iter().all(|&x| x == 0xab) && b[0x267] == 0);
    }

    #[test]
    fn the_air64_layout_is_the_one_the_upgrade_writes() {
        // `setDataLayout`'s length operand at 0x1c43b3084.
        assert_eq!(AIR64_DATA_LAYOUT.len(), 0xd7);
        assert!(AIR64_DATA_LAYOUT.starts_with("e-p:64:64:64-") && AIR64_DATA_LAYOUT.ends_with("-n8:16:32"));
    }

    #[test]
    fn modules_the_upgrade_leaves_unchanged_for_reflection_are_reflected() {
        let air64 = "air64-apple-ios17.0.0";
        // AIR 2.6: no upgrade step, whatever the triple or layout.
        assert_eq!(upgrade_problem(Some((2, 6, 0)), air64, AIR64_DATA_LAYOUT), None);
        assert_eq!(upgrade_problem(Some((2, 6, 1)), "air32-apple-ios17.0.0", ""), None);
        // AIR 2.5 with the layout the 2.6 step writes.
        assert_eq!(upgrade_problem(Some((2, 5, 0)), "air64-apple-ios16.0.0", AIR64_DATA_LAYOUT), None);
    }

    #[test]
    fn structs_are_laid_out_only_under_the_air64_layout() {
        let info = StructInfo { members: Vec::new() };
        let mut p = Parser { function: "f", data_layout: AIR64_DATA_LAYOUT, next_id: 0, cache: Vec::new() };
        assert!(p.members(&info).is_ok());
        let other = AIR64_DATA_LAYOUT.replace("-i64:64:64-", "-i64:32:64-");
        let mut p = Parser { function: "f", data_layout: &other, next_id: 0, cache: Vec::new() };
        assert!(p.members(&info).is_err());
    }

    #[test]
    fn modules_the_upgrade_changes_are_refused() {
        let air64 = "air64-apple-ios16.0.0";
        // AIR 2.5 whose layout the 2.6 step would replace.
        assert!(upgrade_problem(Some((2, 5, 0)), air64, "").is_some());
        assert!(upgrade_problem(Some((2, 5, 0)), air64, &AIR64_DATA_LAYOUT.replace("-n8:16:32", "")).is_some());
        // An air32 module gets the air32 layout instead.
        assert!(upgrade_problem(Some((2, 5, 0)), "air32-apple-ios16.0.0", AIR64_DATA_LAYOUT).is_some());
        // The steps of 1.8 to 2.5 run below 2.5.
        for v in [(1, 6, 0), (1, 8, 0), (1, 11, 0), (2, 0, 0), (2, 3, 0), (2, 4, 0), (2, 4, 9)] {
            assert!(upgrade_problem(Some(v), air64, AIR64_DATA_LAYOUT).is_some(), "{v:?}");
        }
        // Newer than the target, and no version at all.
        assert!(upgrade_problem(Some((2, 7, 0)), air64, AIR64_DATA_LAYOUT).is_some());
        assert!(upgrade_problem(Some((3, 0, 0)), air64, AIR64_DATA_LAYOUT).is_some());
        assert!(upgrade_problem(None, air64, AIR64_DATA_LAYOUT).is_some());
    }

    fn record(name: &str, flags: u32) -> Record {
        Record {
            flags,
            index: 2,
            array_length: 1,
            builtin_type: 5,
            name: name.into(),
            type_name: Some("uint".into()),
            data_type: 33,
            alignment: 4,
            data_size: 4,
            pixel_format: 0,
            alu_type: 0,
            layout: None,
        }
    }

    #[test]
    fn records_are_written_as_serialize_arguments_writes_them() {
        let mut w = Writer { out: Vec::new(), written: HashMap::new() };
        let buffer = record(
            "b",
            flag::ARGUMENT | flag::ACTIVE | (kind::DEVICE_BUFFER << flag::KIND_SHIFT) | (1 << flag::ACCESS_SHIFT),
        );
        let builtin = record("i", flag::BUILTIN | flag::ACTIVE | (kind::BUILTIN << flag::KIND_SHIFT));
        w.arguments(&[buffer, builtin]).unwrap();
        let mut want = Vec::new();
        let u = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
        u(&mut want, 2);
        want.extend_from_slice(b"b\0");
        // Active (bit 9), read-write (bits 5-7), kind 2 (bits 0-4).
        u(&mut want, 0x200 | 0x20 | 2);
        for x in [2, 1, 33, 0, 0, 4, 4] {
            u(&mut want, x);
        }
        want.extend_from_slice(b"i\0");
        u(&mut want, 0x200 | 6);
        u(&mut want, 33);
        u(&mut want, 5);
        u(&mut want, 0);
        u(&mut want, 0);
        assert_eq!(w.out, want);
    }
}
