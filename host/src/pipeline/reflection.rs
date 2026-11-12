//! Reflection transport: the compiler's reflection (AirWgsl `Reflection`, built from the AIR metadata by its
//! reflection layer) as the wire protocol's `Reflection`, which the guest builds Metal's reflection objects from.
//!
//! The two have the same members with the same meaning (the compiler's type is defined field for field after the wire
//! struct); every struct is destructured here, so a member either side gains breaks the build until it is carried.

use webmetal_wire as wire;

/// The wire form of the compiler's reflection.
pub fn to_wire(r: &airwgsl::Reflection) -> wire::Reflection {
    let airwgsl::Reflection { stages } = r;
    wire::Reflection { stages: stages.iter().map(stage).collect() }
}

fn stage(s: &airwgsl::StageReflection) -> wire::StageReflection {
    let airwgsl::StageReflection { function_type, arguments, types } = s;
    wire::StageReflection {
        function_type: *function_type,
        arguments: arguments.iter().map(argument).collect(),
        types: types.iter().map(ty).collect(),
    }
}

fn argument(a: &airwgsl::ArgumentReflection) -> wire::ArgumentReflection {
    let airwgsl::ArgumentReflection {
        name,
        r#type,
        access,
        index,
        active,
        buffer_alignment,
        buffer_data_size,
        buffer_data_type,
        buffer_struct_type,
        buffer_pointer_type,
        buffer_indirect_argument_type,
        threadgroup_memory_alignment,
        threadgroup_memory_data_size,
        texture_type,
        texture_data_type,
        is_depth_texture,
        array_length,
        indirect_constant_alignment,
        indirect_constant_data_size,
        indirect_constant_data_type,
    } = a;
    wire::ArgumentReflection {
        name: name.clone(),
        r#type: *r#type,
        access: *access,
        index: *index,
        active: *active,
        buffer_alignment: *buffer_alignment,
        buffer_data_size: *buffer_data_size,
        buffer_data_type: *buffer_data_type,
        buffer_struct_type: *buffer_struct_type,
        buffer_pointer_type: *buffer_pointer_type,
        buffer_indirect_argument_type: *buffer_indirect_argument_type,
        threadgroup_memory_alignment: *threadgroup_memory_alignment,
        threadgroup_memory_data_size: *threadgroup_memory_data_size,
        texture_type: *texture_type,
        texture_data_type: *texture_data_type,
        is_depth_texture: *is_depth_texture,
        array_length: *array_length,
        indirect_constant_alignment: *indirect_constant_alignment,
        indirect_constant_data_size: *indirect_constant_data_size,
        indirect_constant_data_type: *indirect_constant_data_type,
    }
}

fn ty(t: &airwgsl::TypeReflection) -> wire::TypeReflection {
    let airwgsl::TypeReflection {
        data_type,
        members,
        element_data_type,
        element_type,
        array_length,
        stride,
        argument_index_stride,
        access,
        alignment,
        data_size,
        element_is_argument_buffer,
        texture_data_type,
        texture_type,
        is_depth_texture,
    } = t;
    wire::TypeReflection {
        data_type: *data_type,
        members: members
            .iter()
            .map(|m| {
                let airwgsl::StructMemberReflection { name, offset, data_type, r#type, argument_index } = m;
                wire::StructMemberReflection {
                    name: name.clone(),
                    offset: *offset,
                    data_type: *data_type,
                    r#type: *r#type,
                    argument_index: *argument_index,
                }
            })
            .collect(),
        element_data_type: *element_data_type,
        element_type: *element_type,
        array_length: *array_length,
        stride: *stride,
        argument_index_stride: *argument_index_stride,
        access: *access,
        alignment: *alignment,
        data_size: *data_size,
        element_is_argument_buffer: *element_is_argument_buffer,
        texture_data_type: *texture_data_type,
        texture_type: *texture_type,
        is_depth_texture: *is_depth_texture,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_member_is_carried() {
        let r = airwgsl::Reflection {
            stages: vec![airwgsl::StageReflection {
                function_type: 3,
                arguments: vec![airwgsl::ArgumentReflection {
                    name: "out".into(),
                    r#type: 0,
                    access: 1,
                    index: 2,
                    active: true,
                    buffer_alignment: 4,
                    buffer_data_size: 16,
                    buffer_data_type: 1,
                    buffer_struct_type: 0,
                    buffer_pointer_type: 1,
                    buffer_indirect_argument_type: airwgsl::NO_TYPE,
                    threadgroup_memory_alignment: 5,
                    threadgroup_memory_data_size: 6,
                    texture_type: 7,
                    texture_data_type: 8,
                    is_depth_texture: true,
                    array_length: 9,
                    indirect_constant_alignment: 10,
                    indirect_constant_data_size: 11,
                    indirect_constant_data_type: 12,
                }],
                types: vec![airwgsl::TypeReflection {
                    data_type: 1,
                    members: vec![airwgsl::StructMemberReflection {
                        name: "a".into(),
                        offset: 4,
                        data_type: 33,
                        r#type: airwgsl::NO_TYPE,
                        argument_index: 3,
                    }],
                    element_data_type: 2,
                    element_type: 1,
                    array_length: 3,
                    stride: 4,
                    argument_index_stride: 5,
                    access: 2,
                    alignment: 16,
                    data_size: 32,
                    element_is_argument_buffer: true,
                    texture_data_type: 3,
                    texture_type: 2,
                    is_depth_texture: true,
                }],
            }],
        };
        let w = to_wire(&r);
        // The wire record round-trips, and its debug form names every value of the compiler's.
        let reply = wire::Reply::CreateComputePipeline(Ok(wire::CreateComputePipelineReply { reflection: Some(w) }));
        let mut bytes = Vec::new();
        reply.encode(&mut bytes).unwrap();
        let (back, _) = wire::Reply::decode(&bytes, 0).unwrap();
        assert_eq!(back, reply);
        let wire::Reply::CreateComputePipeline(Ok(wire::CreateComputePipelineReply { reflection: Some(w) })) = back
        else {
            panic!("reply")
        };
        let a = &w.stages[0].arguments[0];
        assert_eq!(
            (a.index, a.threadgroup_memory_data_size, a.indirect_constant_data_type, a.buffer_indirect_argument_type),
            (2, 6, 12, airwgsl::NO_TYPE)
        );
        let t = &w.stages[0].types[0];
        assert_eq!((t.members[0].argument_index, t.argument_index_stride, t.is_depth_texture), (3, 5, true));
    }
}
