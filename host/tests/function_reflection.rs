//! `ReflectFunction` through the executor for functions of the reference library, with each reply read the way
//! Metal.framework 343.21 (iPadOS 17.7.11, 21H461) reads it: [`read_reply`] follows `_MTLNewReflectionData` and the
//! legacy deserializers step by step, with their checks, so a reply that 21H461's Metal.framework would reject or read
//! differently fails here. The values read back are then compared with the shaders' declarations
//! (tests/reference/shaders) and with the AIR metadata the host read them from.

#![cfg(feature = "native")]

mod mem_common;
mod pipeline_common;

use std::collections::HashMap;

use pipeline_common::{Harness, reference_library};
use webmetal_wire::*;

/// A read of the reply that fails as the reader aborts (`DeserialContext`, Metal 0x187a7bf98 and 0x187a7c200: `abort`
/// on a read past the end; `moveToOffset` 0x187abf4fc).
struct Context<'a> {
    data: &'a [u8],
    offset: usize,
}

impl Context<'_> {
    fn u32(&mut self) -> u32 {
        assert!(self.data.len() - self.offset >= 4, "a u32 read past the end at {:#x}", self.offset);
        let v = u32::from_le_bytes(self.data[self.offset..self.offset + 4].try_into().unwrap());
        self.offset += 4;
        v
    }

    fn string(&mut self) -> String {
        let rest = &self.data[self.offset..];
        let len = rest.iter().position(|&c| c == 0).expect("a string without its NUL");
        self.offset += len + 1;
        String::from_utf8(rest[..len].to_vec()).unwrap()
    }

    fn move_to(&mut self, offset: usize, strict: bool) {
        assert!(!strict || self.offset == offset, "a section ends at {:#x}, not at {offset:#x}", self.offset);
        assert!(offset <= self.data.len());
        self.offset = offset;
    }
}

/// A binding as `deserializeArguments` (Metal 0x187a94cec) builds it.
#[derive(Clone, Debug, PartialEq)]
struct Binding {
    name: String,
    /// `MTLArgumentType` the reader gives the object (15 for a built-in).
    arg_type: u64,
    access: u32,
    active: bool,
    index: u32,
    array_length: u32,
    detail: Detail,
    struct_type: Option<StructType>,
}

#[derive(Clone, Debug, PartialEq)]
enum Detail {
    Buffer { data_type: u32, pixel_format: u32, alu_type: u32, alignment: u32, data_size: u32, constant: bool },
    Threadgroup { data_type: u32, alignment: u32, data_size: u32 },
    Texture { data_type: u32, texture_type: u32, depth: bool },
    Builtin { data_type: u32, builtin_type: u32 },
    IndirectConstant { data_type: u32, pixel_format: u32, alu_type: u32, alignment: u32, data_size: u32 },
    None,
}

#[derive(Clone, Debug, PartialEq)]
struct StructType {
    members: Vec<StructMember>,
}

#[derive(Clone, Debug, PartialEq)]
struct StructMember {
    name: String,
    offset: u32,
    argument_index: u32,
    data_type: u32,
    pixel_format: u32,
    alu_type: u32,
    details: Option<Box<MemberDetails>>,
    render_target: u32,
    raster_order_group: u32,
}

#[derive(Clone, Debug, PartialEq)]
enum MemberDetails {
    Struct(StructType),
    Array {
        length: u32,
        element_type: u32,
        stride: u32,
        index_stride: u32,
        pixel_format: u32,
        alu_type: u32,
        element: Option<Box<MemberDetails>>,
    },
    Argument(Binding),
}

struct Reader<'a> {
    c: Context<'a>,
    structs: HashMap<u32, StructType>,
    version: u32,
}

impl Reader<'_> {
    /// `structTypeFromSerializedContext` (Metal 0x187a7bc70) with a nonzero version: an id, 0 for none, a known id
    /// for a struct read before.
    fn struct_type(&mut self) -> Option<StructType> {
        assert_ne!(self.version, 0);
        let id = self.c.u32();
        if id == 0 {
            return None;
        }
        if let Some(s) = self.structs.get(&id) {
            return Some(s.clone());
        }
        let count = self.c.u32();
        let mut members = Vec::new();
        for _ in 0..count {
            let name = self.c.string();
            let offset = self.c.u32();
            let argument_index = self.c.u32();
            let data_type = self.c.u32();
            let pixel_format = self.c.u32();
            let alu_type = self.c.u32();
            let details = self.details(data_type).map(Box::new);
            let render_target = self.c.u32();
            let raster_order_group = self.c.u32();
            members.push(StructMember {
                name,
                offset,
                argument_index,
                data_type,
                pixel_format,
                alu_type,
                details,
                render_target,
                raster_order_group,
            });
        }
        let s = StructType { members };
        self.structs.insert(id, s.clone());
        Some(s)
    }

    fn details(&mut self, data_type: u32) -> Option<MemberDetails> {
        match data_type {
            1 => self.struct_type().map(MemberDetails::Struct),
            2 => Some(self.array()),
            0x39 => {
                // `newArgumentTypeFromSerializedContext` (0x187a7e86c): one argument with its struct trailer.
                let mut list = self.arguments(true);
                assert_eq!(list.len(), 1, "an argument member with {} records", list.len());
                Some(MemberDetails::Argument(list.remove(0)))
            }
            _ => None,
        }
    }

    /// `newArrayTypeFromSerializedContext` (0x187a7e424).
    fn array(&mut self) -> MemberDetails {
        let length = self.c.u32();
        let element_type = self.c.u32();
        let stride = self.c.u32();
        let index_stride = self.c.u32();
        let pixel_format = self.c.u32();
        let alu_type = self.c.u32();
        let element = self.details(element_type).map(Box::new);
        MemberDetails::Array { length, element_type, stride, index_stride, pixel_format, alu_type, element }
    }

    /// `deserializeArguments` (0x187a94cec) with struct types read (MTLPipelineOptionBufferTypeInfo).
    fn arguments(&mut self, structs: bool) -> Vec<Binding> {
        let count = self.c.u32();
        let mut out = Vec::new();
        for _ in 0..count {
            let name = self.c.string();
            let flags = self.c.u32();
            let (index, array_length) = if flags & 0xf == 6 { (0, 1) } else { (self.c.u32(), self.c.u32()) };
            let kind = flags & 0x1f;
            let access = (flags >> 5) & 0xf;
            let active = flags & 0x200 != 0;
            let (arg_type, detail) = match kind {
                1 | 2 | 14 => {
                    let data_type = self.c.u32();
                    let pixel_format = self.c.u32();
                    let alu_type = self.c.u32();
                    let alignment = self.c.u32() & 0xffff;
                    let data_size = self.c.u32();
                    (
                        if kind == 14 { 0x16 } else { 0 },
                        Detail::Buffer { data_type, pixel_format, alu_type, alignment, data_size, constant: kind == 1 },
                    )
                }
                3 => {
                    let data_type = self.c.u32();
                    self.c.u32();
                    self.c.u32();
                    let alignment = self.c.u32() & 0xffff;
                    let data_size = self.c.u32();
                    (1, Detail::Threadgroup { data_type, alignment, data_size })
                }
                4 => {
                    let data_type = self.c.u32();
                    let t = self.c.u32();
                    (2, Detail::Texture { data_type, texture_type: t & 0xffff, depth: t & 0x10000 != 0 })
                }
                5 => (3, Detail::None),
                6 => {
                    let data_type = self.c.u32();
                    let builtin_type = self.c.u32();
                    (15, Detail::Builtin { data_type, builtin_type })
                }
                8 => {
                    let data_type = self.c.u32();
                    let pixel_format = self.c.u32();
                    let alu_type = self.c.u32();
                    let alignment = self.c.u32() & 0xffff;
                    let data_size = self.c.u32();
                    (4, Detail::IndirectConstant { data_type, pixel_format, alu_type, alignment, data_size })
                }
                11 => (0x14, Detail::None),
                12 => (0x13, Detail::None),
                13 => (0x15, Detail::None),
                16 => (0x18, Detail::None),
                17 => (0x1b, Detail::None),
                18 => (0x19, Detail::None),
                19 => (0x1a, Detail::None),
                k => panic!("record kind {k}, which the reader aborts on or which the host does not write"),
            };
            out.push(Binding { name, arg_type, access, active, index, array_length, detail, struct_type: None });
        }
        if structs {
            for b in &mut out {
                let s = self.struct_type();
                // The struct type is kept for buffers, threadgroup memory, imageblocks, 0x16 and payloads.
                if matches!(b.arg_type, 0 | 1 | 16 | 17 | 0x16 | 0x22) {
                    b.struct_type = s;
                }
            }
        }
        out
    }
}

/// The reflection of one function as Metal.framework reads it.
#[derive(Debug)]
struct FunctionReflection {
    stage_attributes: Vec<u32>,
    builtin_count: u32,
    arguments: Vec<Binding>,
    trace_buffer_index: u32,
}

/// `_MTLNewReflectionData` (0x187abe3d0) for retrieval types 1 and 4, then the legacy deserializer of the function type
/// (`MTLInputStageReflectionDeserializer::deserialize` 0x187abf0c4 for kernels, `MTLFragmentReflectionDeserializerLegacy::
/// deserialize` 0x187abef00 for fragment functions), reading struct types.
fn read_reply(reply: &[u8], function_type: u64) -> FunctionReflection {
    let word = |i: usize| u64::from_le_bytes(reply[i * 8..i * 8 + 8].try_into().unwrap());
    assert_eq!(word(0), 0x8_ef13_c710, "the block's magic");
    // Type 1 reads the AIR reflection at [0x30]/[0x38] when it is there, else the legacy one at [0x10]/[0x18].
    assert_eq!(word(7), 0, "an AIR reflection, which the service does not write for this request");
    assert_eq!(word(9), 0, "plugin data, which the service does not write for this request");
    let (start, size) = (word(2) as usize, word(3) as usize);
    let legacy = &reply[start..start + size];
    assert_eq!(&legacy[..8], b"MTLPSBIN", "ShouldCreateAIRVersion picks the legacy reader only for MTLPSBIN data");
    let mut r = Reader { c: Context { data: legacy, offset: 8 }, structs: HashMap::new(), version: 0 };
    r.version = r.c.u32();
    let mut stage_attributes = Vec::new();
    if function_type == 3 {
        let n = r.c.u32();
        for _ in 0..n {
            stage_attributes.push(r.c.u32());
        }
    }
    // deserializeArgumentsWithDeserializer (0x187abf530).
    let builtin_count = r.c.u32();
    let end = r.c.u32() as usize;
    let arguments = r.arguments(true);
    r.c.move_to(end, true);
    if function_type == 2 {
        if r.c.u32() != 0 {
            panic!("an output imageblock data argument");
        }
    } else {
        // The post-vertex-dump and inferred-input sections, read without MTLPipelineOption bits 19 and 20.
        r.c.u32();
        let end = r.c.u32() as usize;
        assert_eq!(r.c.u32(), 0, "post-vertex-dump outputs");
        r.c.move_to(end, false);
        let end = r.c.u32() as usize;
        assert_eq!(r.c.u32(), 0, "inferred inputs");
        r.c.move_to(end, false);
    }
    let trace_buffer_index = r.c.u32();
    if r.version >> 16 != 0 {
        assert_eq!(r.c.u32(), 0, "plugin return data");
    }
    if r.version >= 0x20000 {
        // deserializeGlobalBindingsWithDeserializer (0x187abf5b0).
        let end = r.c.u32() as usize;
        assert!(r.arguments(false).is_empty(), "global bindings");
        r.c.move_to(end, true);
    }
    assert_eq!(r.c.offset, legacy.len(), "the reflection ends where its data does");
    FunctionReflection { stage_attributes, builtin_count, arguments, trace_buffer_index }
}

fn reflect(h: &Harness, name: &str) -> (Vec<u8>, FunctionReflection) {
    let lib = reference_library();
    let f = &lib[name];
    let handle = h.function(&f.bitcode, name, f.function_type);
    match h.request(Request::ReflectFunction(ReflectFunction { function: handle })) {
        Reply::ReflectFunction(Ok(r)) => {
            let read = read_reply(&r.reflection, f.function_type);
            (r.reflection, read)
        }
        other => panic!("{name}: {other:?}"),
    }
}

fn member<'a>(s: &'a StructType, name: &str) -> &'a StructMember {
    s.members.iter().find(|m| m.name == name).unwrap_or_else(|| panic!("no member {name} in {s:?}"))
}

fn argument_of(m: &StructMember) -> &Binding {
    match m.details.as_deref() {
        Some(MemberDetails::Argument(b)) => b,
        other => panic!("{}: {other:?}", m.name),
    }
}

/// `arg_read(constant ArgResources &args [[buffer(0)]], uint i [[thread_position_in_grid]])`: an argument buffer of a
/// buffer, a texture, a sampler, a constant and a writable buffer.
#[test]
fn an_argument_buffer_kernel_reflects_its_struct() {
    let h = Harness::new();
    let (bytes, r) = reflect(&h, "arg_read");
    assert_eq!(bytes.len() % 8, 0);
    assert_eq!(r.trace_buffer_index, 0xffff_ffff);
    assert!(r.stage_attributes.is_empty());
    assert_eq!(r.builtin_count, 1);
    assert_eq!(r.arguments.len(), 2);
    let args = &r.arguments[0];
    assert_eq!((args.name.as_str(), args.arg_type, args.index, args.array_length), ("args", 0, 0, 1));
    assert!(args.active);
    // `constant` address space: the reader's isConstantBuffer; a struct (MTLDataTypeStruct).
    assert!(matches!(args.detail, Detail::Buffer { data_type: 1, constant: true, .. }), "{:?}", args.detail);
    let s = args.struct_type.as_ref().expect("the argument buffer's struct");
    let names: Vec<&str> = s.members.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["values", "image", "smp", "scale", "out"]);
    for (m, id) in s.members.iter().zip(0..) {
        assert_eq!(m.argument_index, id, "{}'s [[id]]", m.name);
        assert_eq!(m.data_type, 0x39, "{} is an argument-buffer resource", m.name);
        assert_eq!((m.pixel_format, m.alu_type), (0xffff_ffff, 0xffff_ffff));
        assert_eq!((m.render_target, m.raster_order_group), (0xffff_ffff, 0xffff_ffff));
    }
    let values = argument_of(member(s, "values"));
    assert_eq!((values.arg_type, values.index, values.access), (0, 0, 0));
    assert!(matches!(values.detail, Detail::Buffer { data_type: 3, constant: false, alignment: 4, data_size: 4, .. }));
    let image = argument_of(member(s, "image"));
    assert_eq!((image.arg_type, image.index, image.access), (2, 1, 0));
    assert_eq!(image.detail, Detail::Texture { data_type: 3, texture_type: 2, depth: false });
    let smp = argument_of(member(s, "smp"));
    assert_eq!((smp.arg_type, smp.index), (3, 2));
    let scale = argument_of(member(s, "scale"));
    assert_eq!((scale.arg_type, scale.index), (4, 3));
    assert!(matches!(scale.detail, Detail::IndirectConstant { data_type: 33, .. }), "{:?}", scale.detail);
    let out = argument_of(member(s, "out"));
    assert_eq!((out.arg_type, out.index, out.access), (0, 4, 1));
    assert!(matches!(out.detail, Detail::Buffer { data_type: 33, .. }));
    // Offsets increase in declaration order and each resource takes 8 bytes or more.
    assert!(s.members.windows(2).all(|w| w[1].offset >= w[0].offset + 8));
    let i = &r.arguments[1];
    assert_eq!((i.name.as_str(), i.arg_type), ("i", 15));
    assert!(i.active);
    assert_eq!(i.detail, Detail::Builtin { data_type: 33, builtin_type: 5 });
}

/// `arg_arrays`: arrays of textures, buffers and constants in an argument buffer. An `array<texture2d<float>, 4>` is one
/// resource of four elements; a C array of buffers or constants is an array member whose element is the resource, with
/// the argument indices each element takes.
#[test]
fn argument_buffer_arrays_carry_their_element_arguments() {
    let h = Harness::new();
    let (_, r) = reflect(&h, "arg_arrays");
    let s = r.arguments[0].struct_type.as_ref().expect("the argument buffer's struct");
    let names: Vec<&str> = s.members.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["textures", "buffers", "constants"]);
    let textures = member(s, "textures");
    assert_eq!((textures.argument_index, textures.data_type), (0, 0x39));
    let t = argument_of(textures);
    assert_eq!((t.arg_type, t.index, t.array_length), (2, 0, 4));
    for (name, id, length, arg_type) in [("buffers", 4, 3, 0), ("constants", 7, 2, 4)] {
        let m = member(s, name);
        assert_eq!((m.argument_index, m.data_type), (id, 2), "{name}");
        let Some(MemberDetails::Array { length: l, element_type, index_stride, element, .. }) = m.details.as_deref()
        else {
            panic!("{name}: {:?}", m.details)
        };
        assert_eq!((*l, *element_type), (length, 0x39), "{name}");
        // An argument buffer's array of resources steps one index per element.
        assert_eq!(*index_stride, 1, "{name}");
        let Some(MemberDetails::Argument(b)) = element.as_deref() else { panic!("{name}: {element:?}") };
        assert_eq!((b.arg_type, b.index), (arg_type, id), "{name}");
    }
}

/// `arg_nested`: an argument buffer that points to another; the inner struct is the pointee of a member buffer.
#[test]
fn nested_argument_buffers_reflect_their_inner_struct() {
    let h = Harness::new();
    let (_, r) = reflect(&h, "arg_nested");
    let outer = r.arguments[0].struct_type.as_ref().expect("the outer struct");
    let inner = argument_of(member(outer, "inner"));
    assert_eq!((inner.arg_type, inner.index), (0, 0));
    assert!(matches!(inner.detail, Detail::Buffer { data_type: 1, constant: true, .. }), "{:?}", inner.detail);
    let s = inner.struct_type.as_ref().expect("the inner struct");
    let data = argument_of(member(s, "data"));
    assert_eq!((data.arg_type, data.index, data.access), (0, 0, 0));
    let bias = argument_of(member(outer, "bias"));
    assert_eq!((bias.arg_type, bias.index), (4, 1));
    assert_eq!(r.arguments[1].name, "out");
}

/// `pip_reflect_compute`: buffers of each address space and access, textures of each access and kind, a depth texture,
/// a sampler and threadgroup memory, in the order the kernel declares them, then its built-ins.
#[test]
fn a_kernel_reflects_every_argument_kind() {
    let h = Harness::new();
    let (_, r) = reflect(&h, "pip_reflect_compute");
    let by_name: HashMap<&str, &Binding> = r.arguments.iter().map(|b| (b.name.as_str(), b)).collect();
    let records = by_name["records"];
    assert_eq!((records.arg_type, records.index, records.access), (0, 0, 1));
    assert!(records.struct_type.is_some());
    let matrix = by_name["matrix"];
    assert_eq!((matrix.arg_type, matrix.index), (0, 1));
    assert!(matches!(
        matrix.detail,
        Detail::Buffer { data_type: 15, constant: true, alignment: 16, data_size: 64, .. }
    ));
    assert_eq!(by_name["unused"].index, 3);
    // An argument the kernel does not use has no uses in its module and is inactive.
    assert!(!by_name["unused"].active);
    assert_eq!(by_name["input"].detail, Detail::Texture { data_type: 3, texture_type: 2, depth: false });
    assert_eq!(by_name["input"].access, 0);
    assert_eq!(by_name["output"].detail, Detail::Texture { data_type: 33, texture_type: 2, depth: false });
    assert_eq!(by_name["output"].access, 2);
    assert_eq!(by_name["sampled"].detail, Detail::Texture { data_type: 16, texture_type: 3, depth: false });
    assert_eq!(by_name["depth"].detail, Detail::Texture { data_type: 3, texture_type: 2, depth: true });
    assert_eq!((by_name["s"].arg_type, by_name["s"].index), (3, 0));
    let builtins = &r.arguments[r.arguments.len() - r.builtin_count as usize..];
    assert!(builtins.iter().all(|b| b.arg_type == 15 && b.active));
    assert!(r.arguments[..r.arguments.len() - r.builtin_count as usize].iter().all(|b| b.arg_type != 15));
}

/// `pip_reflect_fragment`: a stage-in struct (not reflected), a texture, a sampler and a constant buffer.
#[test]
fn a_fragment_function_reflects_its_resources() {
    let h = Harness::new();
    let (_, r) = reflect(&h, "pip_reflect_fragment");
    let names: Vec<&str> = r.arguments.iter().map(|b| b.name.as_str()).collect();
    assert_eq!(names, ["t", "s", "tint"]);
    assert_eq!(r.builtin_count, 0);
    assert!(matches!(r.arguments[2].detail, Detail::Buffer { data_type: 6, constant: true, .. }));
}

/// The reflection requests the host does not answer fail as not implemented, naming why.
#[test]
fn vertex_functions_are_refused() {
    let h = Harness::new();
    let lib = reference_library();
    let f = &lib["pip_reflect_vertex"];
    let handle = h.function(&f.bitcode, "pip_reflect_vertex", f.function_type);
    let failure = h.status(Request::ReflectFunction(ReflectFunction { function: handle }));
    assert_eq!(failure.status, status::UNIMPLEMENTED, "{failure:?}");
    assert!(failure.message.contains("post-vertex-dump"), "{}", failure.message);
}
