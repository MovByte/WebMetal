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
//! `parseStructTypeMetadata` (0x1c3df9d50), and the sections of `generateReflectionData` for vertex, kernel and
//! fragment functions. Every value is taken from the AIR metadata by the rules of those functions; the type-name tables
//! are the service's own (`bufferTypeMap` and `textureTypeMap`, below).
//!
//! **Vertex functions.** For a vertex function (`BinaryRequestData` +0x28 == 1) `generateReflection` first runs
//! `MTLAddPostVertexDumpingToModulePass::addPostVertexDumpingToModule` (0x1c3e0c30c, called at 0x1c3e0b24c) with its
//! reflection-only argument set, which skips every change to the module (0x1c3e0c610, 0x1c3e0cbe4, 0x1c3e0cd10,
//! 0x1c3e0ce70, 0x1c3e0d6b0) and only lists what the dump would record: the built-ins the dump needs, then each value
//! of the function's return value. The list and its stride are the post-vertex-dump section of the reflection
//! ([`post_vertex_dump`]). `generateReflectionData` then runs `inferInputsFromModule` (0x1c3e184f8), which finds the
//! loads of read-only buffers the function indexes by its vertex or instance id, for the inferred-input section
//! ([`infer_inputs`]). The vertex inputs' attribute indices are listed as a kernel's stage-in attributes are.
//!
//! What it does not produce is refused as not implemented, naming the reason:
//!
//! - visible, intersection, mesh and object functions (other requests, other readers);
//! - argument kinds whose records carry data the host does not model (imageblocks, payloads, meshes, `air.constant`
//!   and `air.byval` arguments, function pointers), buffers of pixel-format types, `air.global_bindings`, and an
//!   argument named `trace.buffer` (which the service lists or drops by a request flag);
//! - modules whose AIR upgrade would change what the reflection reads (below);
//! - the vertex metadata and IR shapes on which the passes above read memory they do not own (a missing or non-tuple
//!   `air.vertex` operand, a key with no string after it), and the inferred inputs whose sizes come from a data layout
//!   other than the air64 one or from the pointee of a loaded pointer, which the AIR layer does not keep.
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
use airwgsl::ir::layout::DataLayout;
use airwgsl::ir::{BinOp, CastOp, ConstKind, GlobalId, MdId, Metadata, Module, Op, Type, TypeId, Value};
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

    /// A string of the post-vertex-dump section: its bytes, then a NUL unless it already ends in one (the writer
    /// 0x1c3dfcfdc, whose sizing counterpart 0x1c3dfcfa8 counts the same).
    fn dump_string(&mut self, s: &[u8]) {
        self.out.extend_from_slice(s);
        if s.last() != Some(&0) {
            self.out.push(0);
        }
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

// Vertex functions ----------------------------------------------------------------------------------------------------

/// `MTLFrameworkPass::vertexFetchbuiltInAIRString` (MTLCompiler 0x1ef046030): the built-ins the post-vertex dump and
/// the inferred inputs know, by their id.
const VERTEX_FETCH_BUILTINS: [&str; 10] = [
    "air.vertex_id",
    "air.instance_id",
    "air.base_instance",
    "air.patch_id",
    "air.position_in_patch",
    "air.thread_position_in_grid",
    "air.stage_in_grid_origin",
    "air.stage_in_grid_size",
    "air.amplification_id",
    "air.amplification_count",
];

/// The data type `addRequiredBuiltInsToBuiltInList` (0x1c3e19ea8) gives each built-in it requires, by id (the `u32`
/// table at 0x1c3eb12e4): `float` for `air.position_in_patch`, `uint` for the others.
const VERTEX_FETCH_BUILTIN_TYPES: [u32; 10] = [33, 33, 33, 33, 3, 33, 33, 33, 33, 33];

/// Whether the device supports vertex amplification. `MTLGetCompilerOptions` (Metal 0x187a9adf4) sets request flag
/// bit 10 from `-[MTLDevice supportsVertexAmplification]`, and `generateReflection` passes that bit to the post-vertex
/// dump (0x1c3e0b234), which then also requires `air.amplification_id`. The A10X answers no (`-[AGXA10FamilyDevice
/// supportsVertexAmplification]` 0x21c8d4f44, the device profile).
const SUPPORTS_VERTEX_AMPLIFICATION: bool = false;

/// `vertexFetchBuiltInInfo` (0x1c3e33344): the id of a built-in record's `builtInType` (the byte table at 0x1c3eb14d3,
/// for the types of mask 0x30f01b), or `None` for a built-in neither pass knows.
fn vertex_fetch_id(builtin_type: u32) -> Option<usize> {
    Some(match builtin_type {
        1 => 0,
        2 => 1,
        4 => 2,
        5 => 5,
        13 => 3,
        14 => 4,
        15 => 6,
        16 => 7,
        21 => 8,
        22 => 9,
        _ => return None,
    })
}

/// `MTLDataTypeGetComponentType` (0x1c3e1a6dc, table 0x1c3eb14ec) and `MTLDataTypeGetComponentCount` (0x1c3e333cc,
/// table 0x1c3eb1644): a scalar or vector data type's scalar type and length; (0, 0) for any other type.
fn component(data_type: u32) -> (u32, u32) {
    let base = match data_type {
        3..=6 => 3,
        16..=19 => 16,
        29..=56 => 29 + (data_type - 29) / 4 * 4,
        81..=88 => 81 + (data_type - 81) / 4 * 4,
        _ => return (0, 0),
    };
    (base, data_type - base + 1)
}

/// `MTLDataTypeGetVectorDataType` (MTLCompiler 0x1c3e0e4d0): the vector of `count` (1 to 4) elements of a scalar
/// data type, or 0. Metal's own (0x187bab674), which `inferInputsFromModule` calls, also knows `bfloat` (121); the
/// scalar types that pass gives it are among the ones both know.
fn vector_data_type(scalar: u32, count: u32) -> u32 {
    match scalar {
        3 | 16 | 29 | 33 | 37 | 41 | 45 | 49 | 53 | 81 | 85 if (1..=4).contains(&count) => scalar + count - 1,
        _ => 0,
    }
}

/// The size in bits of `MTLFrameworkPass::getLLVMVectorType(dataType, count)` (0x1c3e0e428): a `count`-element vector
/// of the data type's LLVM scalar, or the scalar itself for a count below 2. The service aborts for another type.
fn builtin_vector_bits(data_type: u32, count: u32) -> Result<u64, String> {
    let scalar = match data_type {
        3 | 29 | 33 => 32,
        16 | 37 | 41 => 16,
        53 => 1,
        t => return Err(format!("a post-vertex-dump built-in of data type {t}, for which the service aborts")),
    };
    Ok(scalar * if count >= 2 { count as u64 } else { 1 })
}

/// `llvm::Type::getPrimitiveSizeInBits`: a scalar's or fixed vector's size, 0 for any other type.
fn primitive_bits(ir: &Module, ty: TypeId) -> Result<u64, String> {
    Ok(match ir.types.get(ty) {
        Type::Half | Type::BFloat => 16,
        Type::Float => 32,
        Type::Double => 64,
        Type::X86Fp80 => 80,
        Type::Fp128 | Type::PpcFp128 => 128,
        Type::X86Amx => 8192,
        Type::Int(bits) => *bits as u64,
        Type::Vector { len, elem, scalable: false } => *len as u64 * primitive_bits(ir, *elem)?,
        Type::Vector { scalable: true, .. } => return Err("a scalable vector in a vertex function's output".into()),
        _ => 0,
    })
}

/// The bytes a post-vertex dump gives a value of `bits` bits (0x1c3e0d484, 0x1c3dfadb0): its bytes, at least 4,
/// rounded up to a multiple of 4.
fn dump_size(bits: u64) -> u32 {
    ((bits >> 3) as u32).max(4).wrapping_add(3) & !3
}

/// What the post-vertex dump records of one value (`PostVertexDumpOutput`, 0x1c3dfaf6c), in the order
/// `serializePostVertexDumpOutputs` (0x1c3dfcd64) writes it.
#[derive(Debug, PartialEq)]
struct DumpOutput {
    /// +0x10: the value's AIR role (`air.position`), or the built-in's name.
    md_type: Vec<u8>,
    data_type: u32,
    pixel_format: u32,
    alu_type: u32,
    /// +0x40: the value's path in the return value.
    name: Vec<u8>,
    /// +0x58: where the dump writes it.
    offset: u32,
}

/// The `air.vertex` node of the function: its outputs operand and, for a post-tessellation vertex function, whether its
/// patches are triangles. `eliminateOtherEntryPoints` (0x1c3e09a44) leaves the module this one node, which both passes
/// read as `air.vertex`'s operand 0.
struct VertexNode {
    outputs: Option<MdId>,
    triangle_patch: Option<bool>,
}

fn md_string<'a>(ir: &'a Module, op: Option<&Option<MdId>>) -> Option<&'a [u8]> {
    match op.copied().flatten().map(|id| ir.md(id)) {
        Some(Metadata::String(s)) => Some(s),
        _ => None,
    }
}

fn vertex_node(ir: &Module, function: u32) -> Result<VertexNode, String> {
    let nodes: Vec<&[Option<MdId>]> = ir
        .named_metadata
        .iter()
        .filter(|n| n.name == "air.vertex")
        .flat_map(|n| n.ops.iter())
        .filter_map(|&id| match ir.md(id) {
            Metadata::Tuple { ops, .. } => Some(ops.as_slice()),
            _ => None,
        })
        .filter(|ops| {
            matches!(
                ops.first().copied().flatten().map(|id| ir.md(id)),
                Some(Metadata::Value { value: Value::Global(GlobalId::Function(f)), .. }) if *f == function
            )
        })
        .collect();
    let [ops] = nodes.as_slice() else {
        return Err(format!("{} air.vertex nodes of the function rather than one", nodes.len()));
    };
    // The pass reads operand 1 as the outputs and operand 2 as the argument list (getArgumentMetadataIndices,
    // 0x1c3e1a04c) without checking that they are there.
    if ops.len() < 3 {
        return Err("an air.vertex node without its outputs and arguments operands".into());
    }
    for (i, what) in [(1, "outputs"), (2, "arguments")] {
        if let Some(id) = ops[i]
            && !matches!(ir.md(id), Metadata::Tuple { .. })
        {
            return Err(format!("an air.vertex {what} operand that is not a node"));
        }
    }
    if ops[2].is_none() {
        return Err("an air.vertex node whose arguments operand is null".into());
    }
    // The patch operand (0x1c3e0c3ec-0x1c3e0c448, 0x1c3e0dc00-0x1c3e0dc58): a node of at least two operands whose first
    // is "air.patch"; its second "triangle" makes triangle patches, any other string quad ones.
    let mut triangle_patch = None;
    if let Some(&patch) = ops.get(3) {
        let Some(Metadata::Tuple { ops: p, .. }) = patch.map(|id| ir.md(id)) else {
            return Err("an air.vertex operand 3 that is not a node".into());
        };
        if p.len() >= 2 {
            let Some(key) = md_string(ir, p.first()) else {
                return Err("an air.vertex operand 3 whose first operand is not a string".into());
            };
            if key == b"air.patch" {
                let Some(kind) = md_string(ir, p.get(1)) else {
                    return Err("an air.patch node whose patch type is not a string".into());
                };
                triangle_patch = Some(kind == b"triangle");
            }
        }
    }
    Ok(VertexNode { outputs: ops[1], triangle_patch })
}

/// The post-vertex dump's walk over the function's return value (`iterateAllContainedValues`, 0x1c3df9314, with the
/// pass's block 0x1c3dfad5c), appending to `outputs` and `stride`.
struct Dump<'a> {
    ir: &'a Module,
    outputs: Vec<DumpOutput>,
    stride: u32,
}

impl Dump<'_> {
    /// One value of type `ty` described by `md`, inside a value named `name` at `path` whose metadata gave
    /// (`data_type`, `pixel_format`, `alu_type`).
    #[allow(clippy::too_many_arguments)]
    fn value(
        &mut self,
        ty: TypeId,
        md: Option<MdId>,
        name: &[u8],
        data_type: u32,
        pixel_format: u32,
        alu_type: u32,
        path: &[u8],
    ) -> Result<(), String> {
        let ops: &[Option<MdId>] = match md.map(|id| self.ir.md(id)) {
            None => &[],
            Some(Metadata::Tuple { ops, .. }) => ops,
            Some(_) => return Err("a vertex output whose metadata is not a node".into()),
        };
        // The node's strings: the first is the role; "air.arg_type_name" and "air.arg_name" name the string after them.
        let mut role: &[u8] = b"";
        let mut arg_name: &[u8] = b"";
        let (mut md_type, mut md_format, mut md_alu) = (0, 0, 0);
        for (i, op) in ops.iter().enumerate() {
            let Some(s) = md_string(self.ir, Some(op)) else { continue };
            if s == b"air.arg_type_name" {
                let Some(t) = md_string(self.ir, ops.get(i + 1)) else {
                    return Err("an air.arg_type_name with no string after it in a vertex output".into());
                };
                // Read through `strlen` (0x1c3df9484).
                let t = &t[..t.iter().position(|&c| c == 0).unwrap_or(t.len())];
                let (dt, _, _, pf, alu) = data_type_info(&String::from_utf8_lossy(t))?;
                (md_type, md_format, md_alu) = (dt, pf, alu);
            } else if s == b"air.arg_name" {
                let Some(n) = md_string(self.ir, ops.get(i + 1)) else {
                    return Err("an air.arg_name with no string after it in a vertex output".into());
                };
                arg_name = n;
            } else if i == 0 {
                role = s;
            }
        }
        let name = if name.is_empty() { role.to_vec() } else { [name, b" ", role].concat() };
        let data_type = if data_type != 0 { data_type } else { md_type };
        let pixel_format = if pixel_format != 0 { pixel_format } else { md_format };
        let alu_type = if alu_type != 0 { alu_type } else { md_alu };
        let path = [path, arg_name].concat();
        match self.ir.types.get(ty) {
            Type::Void => {}
            Type::Struct { .. } | Type::LiteralStruct { .. } => {
                let (struct_name, elems): (&[u8], &[TypeId]) = match self.ir.types.get(ty) {
                    Type::Struct { name, body, .. } => (name.as_bytes(), body.as_deref().unwrap_or(&[])),
                    Type::LiteralStruct { elems, .. } => (b"", elems),
                    _ => unreachable!("matched above"),
                };
                let base = [if path.is_empty() { struct_name } else { &path }, b"."].concat();
                for (i, &elem) in elems.iter().enumerate() {
                    let member_md = match md {
                        None => None,
                        Some(_) => match ops.get(i) {
                            Some(op) => *op,
                            None => return Err("a vertex output struct with more members than its metadata".into()),
                        },
                    };
                    self.value(elem, member_md, &name, data_type, pixel_format, alu_type, &base)?;
                }
            }
            Type::Array { len, elem } => {
                for i in 0..*len {
                    let element_path = [&path, format!("[{i}]").as_bytes()].concat();
                    self.value(*elem, None, &name, data_type, pixel_format, alu_type, &element_path)?;
                }
            }
            _ => {
                let offset = self.stride;
                self.stride = self.stride.wrapping_add(dump_size(primitive_bits(self.ir, ty)?));
                self.outputs.push(DumpOutput { md_type: name, data_type, pixel_format, alu_type, name: path, offset });
            }
        }
        Ok(())
    }
}

/// What `addPostVertexDumpingToModule` reports in a reflection-only request: the dump's stride and its outputs, the
/// built-ins it requires first (0x1c3e0cb5c-0x1c3e0d5d4), then each value of the return value (0x1c3e0d67c).
fn post_vertex_dump(
    ir: &Module,
    entry: &EntryPoint,
    node: &VertexNode,
    records: &[Option<Record>],
) -> Result<(u32, Vec<DumpOutput>), String> {
    let mut dump = Dump { ir, outputs: Vec::new(), stride: 0 };
    dump.builtins(records, node.triangle_patch)?;
    let Type::Function { ret, .. } = ir.types.get(ir.functions[entry.function as usize].ty) else {
        return Err("an entry point whose type is not a function type".into());
    };
    dump.value(*ret, node.outputs, b"", 0, 0, 0, b"")?;
    Ok((dump.stride, dump.outputs))
}

impl Dump<'_> {
    /// The built-ins the dump requires, for a post-tessellation vertex function of triangle (`Some(true)`) or quad
    /// patches or for another vertex function (`None`).
    fn builtins(&mut self, records: &[Option<Record>], triangle_patch: Option<bool>) -> Result<(), String> {
        // The built-ins of the function's arguments (0x1c3e0c5b4-0x1c3e0c95c): (id, count, data type) in argument order,
        // and each id's place in that list.
        let mut list: Vec<(usize, u32, u32)> = Vec::new();
        let mut place: [Option<usize>; 10] = [None; 10];
        for r in records.iter().flatten() {
            if r.flags & flag::INPUT != 0 || r.flags & flag::BUILTIN == 0 {
                continue;
            }
            let Some(id) = vertex_fetch_id(r.builtin_type) else { continue };
            let type_name = r.type_name.as_deref().unwrap_or("");
            let type_name = &type_name[..type_name.find('\0').unwrap_or(type_name.len())];
            let (scalar, count) = component(data_type_info(type_name)?.0);
            place[id] = Some(list.len());
            list.push((id, count, scalar));
        }
        // The built-ins the dump requires, with their component counts (0x1c3e0cb5c-0x1c3e0cbdc).
        let mut required = [0u32; 10];
        required[1] = 1;
        match triangle_patch {
            None => required[0] = 1,
            Some(triangle) => {
                required[3] = 1;
                required[4] = if triangle { 3 } else { 2 };
            }
        }
        if SUPPORTS_VERTEX_AMPLIFICATION {
            required[8] = 1;
        }
        // addRequiredBuiltInsToBuiltInList: a required built-in the arguments have takes the larger count and the
        // required type; one they lack is added.
        for (id, &count) in required.iter().enumerate() {
            if count == 0 {
                continue;
            }
            match place[id] {
                Some(p) => list[p] = (list[p].0, list[p].1.max(count), VERTEX_FETCH_BUILTIN_TYPES[id]),
                None => {
                    place[id] = Some(list.len());
                    list.push((id, count, VERTEX_FETCH_BUILTIN_TYPES[id]));
                }
            }
        }
        for (id, _) in required.iter().enumerate().filter(|(_, c)| **c != 0) {
            let (name_id, count, data_type) = list[place[id].expect("added above")];
            let vector = vector_data_type(data_type, count);
            let offset = self.stride;
            self.stride = self.stride.wrapping_add(dump_size(builtin_vector_bits(data_type, count)?));
            self.outputs.push(DumpOutput {
                md_type: VERTEX_FETCH_BUILTINS[name_id].as_bytes().to_vec(),
                data_type: vector,
                pixel_format: 0,
                alu_type: vector,
                name: Vec::new(),
                offset,
            });
        }
        Ok(())
    }
}

/// An input `inferInputsFromModule` finds (`MTLGenerateReflectionDataPass::InferredInput`), in the order
/// `serializeInferredInputs` (0x1c3e0114c) writes it; the pixel format and ALU type it writes are always 0 and the last
/// word always 1 (0x1c3e18f88-0x1c3e18fa4).
#[derive(Debug, PartialEq)]
struct InferredInput {
    data_type: u32,
    offset: u32,
    buffer_index: u32,
    stride: u32,
    /// `MTLVertexStepFunction`: constant 0, per vertex 1, per instance 2.
    step_function: u32,
    step_rate: u32,
}

/// The value of an integer constant (`ConstantInt`, value id 0x10): its low 64 bits, as the pass reads them.
fn const_int(ir: &Module, v: Value) -> Option<u64> {
    let Value::Const(c) = v else { return None };
    let c = ir.constant(c);
    match (&c.kind, ir.types.get(c.ty)) {
        (ConstKind::Int(words), Type::Int(_)) => words.first().copied(),
        _ => None,
    }
}

/// `inferInputsFromModule` (0x1c3e184f8): the loads in the function's entry block of a read-only buffer the function
/// only indexes (a buffer argument with an array length of 1 whose every use is a GEP, load or bitcast in the entry
/// block), at an index of the form `(id + c) * m / d` (or with `<<` and `>>` for `*` and `/`, `or` for `+`, `zext`s
/// around the id) of the function's vertex or instance id or of a constant.
fn infer_inputs(ir: &Module, entry: &EntryPoint, records: &[Option<Record>]) -> Result<Vec<InferredInput>, String> {
    let f = &ir.functions[entry.function as usize];
    if f.blocks.is_empty() {
        return Err("an entry point without a body".into());
    }
    let mut block_of = vec![0usize; f.insts.len()];
    for (b, block) in f.blocks.iter().enumerate() {
        for i in &block.insts {
            block_of[i.0 as usize] = b;
        }
    }
    // The arguments (0x1c3e186bc-0x1c3e18994): active vertex-fetch built-ins, by id, and read-only buffers only indexed
    // in the entry block, by argument, with their buffer index.
    let mut builtins: Vec<u32> = Vec::new();
    let mut place: [Option<usize>; 10] = [None; 10];
    let mut buffers: HashMap<u32, u32> = HashMap::new();
    for (i, r) in records.iter().enumerate() {
        let Some(r) = r else { continue };
        if r.flags & flag::ACTIVE == 0 || r.flags & flag::INPUT != 0 {
            continue;
        }
        if r.flags & flag::BUILTIN != 0 {
            if let Some(id) = vertex_fetch_id(r.builtin_type) {
                place[id] = Some(builtins.len());
                builtins.push(i as u32);
            }
            continue;
        }
        if !matches!(r.kind(), kind::CONSTANT_BUFFER | kind::DEVICE_BUFFER)
            || r.flags & (0x7 << flag::ACCESS_SHIFT) != 0
            || r.array_length != 1
        {
            continue;
        }
        let only_indexed = f.insts.iter().enumerate().all(|(n, inst)| {
            !inst.op.operands().contains(&Value::Arg(i as u32))
                || (block_of[n] == 0
                    && matches!(inst.op, Op::Gep { .. } | Op::Load { .. } | Op::Cast { op: CastOp::BitCast, .. }))
        });
        if only_indexed {
            buffers.insert(i as u32, r.index);
        }
    }
    let mut out = Vec::new();
    if builtins.is_empty() || buffers.is_empty() {
        return Ok(out);
    }
    let op_of = |v: Value| match v {
        Value::Inst(i) => Some(&f.inst(i).op),
        _ => None,
    };
    let strip_zext = |v: Value| match op_of(v) {
        Some(Op::Cast { op: CastOp::ZExt, value, .. }) => *value,
        _ => v,
    };
    // An operand pair with a constant: (the other operand, the constant), the left operand tried first.
    let with_constant = |lhs: Value, rhs: Value| match (const_int(ir, lhs), const_int(ir, rhs)) {
        (Some(c), _) => Some((rhs, c)),
        (None, Some(c)) => Some((lhs, c)),
        (None, None) => None,
    };
    let air64 = ir.datalayout == AIR64_DATA_LAYOUT;
    let size_bytes = |t: TypeId| -> Result<u64, String> {
        if !air64 {
            return Err(format!(
                "an inferred vertex input sized by the data layout {:?} rather than the air64 one",
                ir.datalayout
            ));
        }
        let dl = DataLayout::parse(&ir.datalayout)?;
        dl.size_bits(&ir.types, t)
            .map(|b| (b + 7) >> 3)
            .ok_or_else(|| "an inferred vertex input of an unsized type".into())
    };
    // The loads of the entry block (0x1c3e189cc-0x1c3e190d4).
    for &id in &f.blocks[0].insts {
        let load = f.inst(id);
        let Op::Load { ptr, .. } = load.op else { continue };
        let ptr = match op_of(ptr) {
            Some(Op::Cast { op: CastOp::BitCast, value, .. }) => *value,
            _ => ptr,
        };
        let buffer_of = |v: Value| match v {
            Value::Arg(a) => buffers.get(&a).map(|&index| (a, index)),
            _ => None,
        };
        // (the buffer's index, the index value, the struct and member a three-operand GEP selects, the type the index
        // steps over: the load's type for a load of the buffer itself, else the buffer's pointee). The pass reads the
        // pointee from the GEP's base pointer type (0x1c3e18b2c-0x1c3e18b98, 0x1c3e18e10-0x1c3e18e28); the GEP's source
        // element type is that pointee in every module the service reads, whose LLVM rejects a typed-pointer GEP
        // whose explicit type is not its base's pointee.
        let (buffer_index, index, member, element) = if let Some((_, index)) = buffer_of(ptr) {
            (index, None, None, load.ty)
        } else {
            let Some(Op::Gep { source: pointee, ptr: base, indices, .. }) = op_of(ptr) else { continue };
            let pointee = *pointee;
            let Some((_, buffer_index)) = buffer_of(*base) else { continue };
            match indices.as_slice() {
                [index] => (buffer_index, Some(*index), None, pointee),
                [index, field] => {
                    let Some(field) = const_int(ir, *field) else { continue };
                    // An opaque struct has no members, so no member index is in range.
                    let members = match ir.types.get(pointee) {
                        Type::Struct { body, .. } => body.as_ref().map_or(0, Vec::len),
                        Type::LiteralStruct { elems, .. } => elems.len(),
                        _ => continue,
                    };
                    if field >= members as u64 {
                        continue;
                    }
                    (buffer_index, Some(*index), Some(field as usize), pointee)
                }
                _ => continue,
            }
        };
        // The index (0x1c3e18a64-0x1c3e18dfc): constant 0 for a load of the buffer itself.
        let (mut offset, mut multiplier, mut step_function, mut step_rate) = (0u64, 1u64, 0u32, 0u64);
        match index {
            None => {}
            Some(v) => {
                let v = strip_zext(v);
                let (v, addend) = match op_of(v) {
                    Some(Op::Binary { op: BinOp::Add | BinOp::Or, lhs, rhs, .. }) => {
                        let Some(pair) = with_constant(*lhs, *rhs) else { continue };
                        pair
                    }
                    _ => (v, 0),
                };
                let (v, scale, shift) = match op_of(v) {
                    Some(Op::Binary { op: BinOp::Shl, lhs, rhs, .. }) => {
                        let Some(s) = const_int(ir, *rhs) else { continue };
                        (*lhs, 1, s)
                    }
                    Some(Op::Binary { op: BinOp::Mul, lhs, rhs, .. }) => {
                        let Some((v, m)) = with_constant(*lhs, *rhs) else { continue };
                        (v, m, 0)
                    }
                    _ => (v, 1, 0),
                };
                let v = strip_zext(v);
                let (v, divisor, right) = match op_of(v) {
                    Some(Op::Binary { op: BinOp::LShr, lhs, rhs, .. }) => {
                        let Some(s) = const_int(ir, *rhs) else { continue };
                        (*lhs, 1, s)
                    }
                    Some(Op::Binary { op: BinOp::UDiv, lhs, rhs, .. }) => {
                        let Some(d) = const_int(ir, *rhs) else { continue };
                        (*lhs, d, 0)
                    }
                    _ => (v, 1, 0),
                };
                // AArch64 `lsl` by a register shifts by its low six bits; `udiv` by 0 is 0.
                multiplier = scale.wrapping_shl(shift as u32);
                let divisor = divisor.wrapping_shl(right as u32);
                if let Some(c) = const_int(ir, v) {
                    offset = c.checked_div(divisor).unwrap_or(0).wrapping_mul(multiplier).wrapping_add(addend);
                } else {
                    let is = |id: usize| place[id].is_some_and(|p| v == Value::Arg(builtins[p]));
                    step_function = if is(0) {
                        1
                    } else if is(1) {
                        2
                    } else {
                        continue;
                    };
                    offset = addend;
                    step_rate = divisor;
                }
            }
        }
        // The loaded type (0x1c3e18e58-0x1c3e18f70): a float, half, int, short or bool scalar or vector of 1 to 4.
        let (count, scalar) = match ir.types.get(load.ty) {
            Type::Ptr(_) => return Err("an inferred vertex input load of a pointer, whose pointee is not known".into()),
            Type::Vector { len, elem, scalable: false } => (*len, *elem),
            _ => (1, load.ty),
        };
        let scalar = match ir.types.get(scalar) {
            Type::Float => 3,
            Type::Half => 16,
            Type::Int(32) => 29,
            Type::Int(16) => 37,
            Type::Int(1) => 53,
            _ => continue,
        };
        let data_type = vector_data_type(scalar, count);
        if data_type == 0 {
            continue;
        }
        // The offset and stride (0x1c3e18e08-0x1c3e18ed0): a struct member read at the index itself (no constant, no
        // scale) is the member's offset in steps of the struct's size; otherwise the element's size scales the index.
        let (offset, stride) = match member {
            Some(m) => {
                if offset != 0 || multiplier != 1 {
                    continue;
                }
                let size = size_bytes(element)?;
                let layout = DataLayout::parse(&ir.datalayout)?
                    .struct_layout(&ir.types, element)
                    .ok_or_else(|| "an inferred vertex input in an unsized struct".to_string())?;
                (layout.offsets[m], size)
            }
            None => {
                let size = size_bytes(element)?;
                (size.wrapping_mul(offset), size.wrapping_mul(multiplier))
            }
        };
        out.push(InferredInput {
            data_type,
            offset: offset as u32,
            buffer_index,
            stride: stride as u32,
            step_function,
            step_rate: step_rate as u32,
        });
    }
    Ok(out)
}

/// The legacy reflection of a vertex, kernel or fragment entry point (`generateReflectionData`, 0x1c3dfd044).
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
    // Vertex functions and kernels have the stage-in attributes, the post-vertex-dump and inferred-input sections
    // (`(MTLFunctionType | 2) == 3`, 0x1c3dfde9c, 0x1c3dfe318, 0x1c3dfe3a8).
    let staged = matches!(function_type, function_type::VERTEX | function_type::KERNEL);
    let mut records = Vec::with_capacity(entry.arguments.len());
    let mut next_id = 0;
    for (i, a) in entry.arguments.iter().enumerate() {
        if a.position as usize != i {
            return Err(refuse("argument records out of the function's parameter order"));
        }
        let mut p = Parser { function: describe, data_layout: &module.ir.datalayout, next_id, cache: Vec::new() };
        records.push(p.record(a, Some(argument_used(module, entry, i as u32)))?);
        next_id = p.next_id;
    }
    let (dump_stride, dump, inferred) = if function_type == function_type::VERTEX {
        let node = vertex_node(&module.ir, entry.function).map_err(|e| refuse(&e))?;
        let (stride, dump) = post_vertex_dump(&module.ir, entry, &node, &records).map_err(|e| refuse(&e))?;
        (stride, dump, infer_inputs(&module.ir, entry, &records).map_err(|e| refuse(&e))?)
    } else {
        (0, Vec::new(), Vec::new())
    };
    let mut main = Vec::new();
    let mut builtins = Vec::new();
    let mut attributes = Vec::new();
    let mut writer = Writer { out: Vec::new(), written: HashMap::new() };
    for record in records {
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
        // The post-vertex-dump section (0x1c3dfe328-0x1c3dfe3a4): the dump's stride, the section's end, the outputs.
        // A kernel's has stride 0 and no outputs.
        w.u32(dump_stride);
        let end = w.reserve();
        w.u32(dump.len() as u32);
        for o in &dump {
            w.dump_string(&o.md_type);
            w.u32(o.data_type);
            w.u32(o.pixel_format);
            w.u32(o.alu_type);
            w.dump_string(&o.name);
            w.u32(o.offset);
        }
        w.patch(end);
        // The inferred inputs (0x1c3dfe3b4-0x1c3dfe3fc): only a vertex function's are inferred.
        let end = w.reserve();
        w.u32(inferred.len() as u32);
        for i in &inferred {
            for v in [i.data_type, 0, 0, i.offset, i.buffer_index, i.stride, i.step_function, i.step_rate, 1] {
                w.u32(v);
            }
        }
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
        function_type::VERTEX | function_type::KERNEL | function_type::FRAGMENT => {}
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

    #[test]
    fn data_types_split_into_component_type_and_count() {
        assert_eq!(component(3), (3, 1));
        assert_eq!(component(5), (3, 3));
        assert_eq!(component(19), (16, 4));
        assert_eq!(component(33), (33, 1));
        assert_eq!(component(55), (53, 3));
        assert_eq!(component(88), (85, 4));
        // Matrices and the types past the tables have none.
        assert_eq!(component(15), (0, 0));
        assert_eq!(component(57), (0, 0));
        assert_eq!(component(0), (0, 0));
        for t in 3..=88 {
            let (scalar, count) = component(t);
            if count != 0 {
                assert_eq!(vector_data_type(scalar, count), t, "{t}");
            }
        }
        assert_eq!(vector_data_type(3, 5), 0);
        assert_eq!(vector_data_type(3, 0), 0);
        assert_eq!(vector_data_type(121, 1), 0);
    }

    #[test]
    fn dumped_values_take_whole_words() {
        assert_eq!(dump_size(0), 4);
        assert_eq!(dump_size(1), 4);
        assert_eq!(dump_size(32), 4);
        assert_eq!(dump_size(48), 8);
        assert_eq!(dump_size(96), 12);
        assert_eq!(dump_size(128), 16);
    }

    #[test]
    fn dump_strings_end_in_one_nul() {
        let mut w = Writer { out: Vec::new(), written: HashMap::new() };
        w.dump_string(b"");
        w.dump_string(b"ab");
        w.dump_string(b"c\0");
        assert_eq!(w.out, b"\0ab\0c\0");
    }

    #[test]
    fn vertex_fetch_ids_follow_the_service_table() {
        let ids: Vec<Option<usize>> = (0..=23).map(vertex_fetch_id).collect();
        let want = [
            None,
            Some(0),
            Some(1),
            None,
            Some(2),
            Some(5),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(3),
            Some(4),
            Some(6),
            Some(7),
            None,
            None,
            None,
            None,
            Some(8),
            Some(9),
            None,
        ];
        assert_eq!(ids, want);
    }

    fn builtin(builtin_type: u32, type_name: &str) -> Option<Record> {
        let mut r = record("b", flag::BUILTIN | flag::ACTIVE | (kind::BUILTIN << flag::KIND_SHIFT));
        r.builtin_type = builtin_type;
        r.type_name = Some(type_name.into());
        Some(r)
    }

    fn dumped(name: &str, data_type: u32, offset: u32) -> DumpOutput {
        DumpOutput {
            md_type: name.as_bytes().to_vec(),
            data_type,
            pixel_format: 0,
            alu_type: data_type,
            name: Vec::new(),
            offset,
        }
    }

    #[test]
    fn a_vertex_function_dumps_its_vertex_and_instance_ids() {
        let ir = Module::new();
        let mut dump = Dump { ir: &ir, outputs: Vec::new(), stride: 0 };
        // A `ushort` vertex id is dumped as the `uint` the dump requires; a base instance is not required.
        dump.builtins(&[builtin(4, "uint"), builtin(1, "ushort")], None).unwrap();
        assert_eq!(dump.outputs, [dumped("air.vertex_id", 33, 0), dumped("air.instance_id", 33, 4)]);
        assert_eq!(dump.stride, 8);
    }

    #[test]
    fn a_post_tessellation_function_dumps_its_patch_and_position() {
        let ir = Module::new();
        let mut dump = Dump { ir: &ir, outputs: Vec::new(), stride: 0 };
        dump.builtins(&[builtin(14, "float3"), builtin(13, "uint")], Some(true)).unwrap();
        assert_eq!(
            dump.outputs,
            [dumped("air.instance_id", 33, 0), dumped("air.patch_id", 33, 4), dumped("air.position_in_patch", 5, 8)]
        );
        assert_eq!(dump.stride, 20);
        let mut dump = Dump { ir: &ir, outputs: Vec::new(), stride: 0 };
        dump.builtins(&[], Some(false)).unwrap();
        assert_eq!(dump.outputs[2], dumped("air.position_in_patch", 4, 8));
        assert_eq!(dump.stride, 16);
    }

    fn strings(m: &mut Module, s: &[&str]) -> MdId {
        let ops = s.iter().map(|s| Some(m.intern_md(Metadata::String(s.as_bytes().to_vec())))).collect();
        m.intern_md(Metadata::Tuple { distinct: false, ops })
    }

    #[test]
    fn the_dump_walks_the_return_value_by_its_metadata() {
        let mut m = Module::new();
        let float = m.types.intern(Type::Float);
        let half = m.types.intern(Type::Half);
        let float4 = m.types.vector(4, float);
        let half3 = m.types.vector(3, half);
        let clip = m.types.intern(Type::Array { len: 2, elem: float });
        let ret = m.types.intern(Type::LiteralStruct { packed: true, elems: vec![float4, clip, half3] });
        let position = strings(&mut m, &["air.position", "air.arg_type_name", "float4", "air.arg_name", "position"]);
        let distance = strings(&mut m, &["air.clip_distance", "air.arg_type_name", "float", "air.arg_name", "clip"]);
        let color =
            strings(&mut m, &["air.vertex_output", "generated(1c)", "air.arg_type_name", "half3", "air.arg_name", "c"]);
        let outputs =
            m.intern_md(Metadata::Tuple { distinct: false, ops: vec![Some(position), Some(distance), Some(color)] });
        let mut dump = Dump { ir: &m, outputs: Vec::new(), stride: 8 };
        dump.value(ret, Some(outputs), b"", 0, 0, 0, b"").unwrap();
        let got: Vec<(String, u32, String, u32)> = dump
            .outputs
            .iter()
            .map(|o| {
                (
                    String::from_utf8(o.md_type.clone()).unwrap(),
                    o.data_type,
                    String::from_utf8(o.name.clone()).unwrap(),
                    o.offset,
                )
            })
            .collect();
        // A literal struct has no name, so its members' paths start at the separator; an array's elements have no
        // metadata of their own, which leaves a space after their role.
        let want = [
            ("air.position", 6, ".position", 8),
            ("air.clip_distance ", 3, ".clip[0]", 24),
            ("air.clip_distance ", 3, ".clip[1]", 28),
            ("air.vertex_output", 18, ".c", 32),
        ];
        let want: Vec<(String, u32, String, u32)> =
            want.iter().map(|(a, b, c, d)| (a.to_string(), *b, c.to_string(), *d)).collect();
        assert_eq!(got, want);
        assert_eq!(dump.stride, 40);
        // A value that is not a struct is one output, named by nothing: the outputs node holds nodes, not strings.
        let mut dump = Dump { ir: &m, outputs: Vec::new(), stride: 0 };
        dump.value(float4, Some(outputs), b"", 0, 0, 0, b"").unwrap();
        assert_eq!(
            dump.outputs,
            [DumpOutput {
                md_type: Vec::new(),
                data_type: 0,
                pixel_format: 0,
                alu_type: 0,
                name: Vec::new(),
                offset: 0
            }]
        );
        // A named struct names the path; a member without its node is refused.
        let named = m.types.new_struct("struct.V".into());
        m.types.set_struct_body(named, false, vec![float4]);
        let one = m.intern_md(Metadata::Tuple { distinct: false, ops: vec![Some(position)] });
        let mut dump = Dump { ir: &m, outputs: Vec::new(), stride: 0 };
        dump.value(named, Some(one), b"", 0, 0, 0, b"").unwrap();
        assert_eq!(dump.outputs[0].name, b"struct.V.position");
        let two = m.intern_md(Metadata::Tuple { distinct: false, ops: vec![] });
        let mut dump = Dump { ir: &m, outputs: Vec::new(), stride: 0 };
        assert!(dump.value(named, Some(two), b"", 0, 0, 0, b"").is_err());
        // A key with nothing after it is refused.
        let bad = strings(&mut m, &["air.position", "air.arg_name"]);
        let list = m.intern_md(Metadata::Tuple { distinct: false, ops: vec![Some(bad)] });
        let mut dump = Dump { ir: &m, outputs: Vec::new(), stride: 0 };
        assert!(dump.value(named, Some(list), b"", 0, 0, 0, b"").is_err());
    }
}
