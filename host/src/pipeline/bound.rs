//! The textures an encoder binds, against the texture arguments of the pipeline's functions.
//!
//! A pipeline is created from its functions alone; what a draw or dispatch binds is checked by Metal's validation
//! layer, and without it a texture the argument cannot take is undefined. The shader compiler compiles a variant for the
//! bound textures and refuses one whose texture type or format class the argument does not take (AirWgsl's
//! `texture-slot-mismatch`, `tex::check_slot`). The encoders check the same rule first, from the pipeline's
//! reflection, so that the failure names the slot and both types and is never taken for the pipeline's creation
//! failing.

use airwgsl::tex::format::{Class, format};

/// `MTLArgumentType` of a texture argument.
const TEXTURE: u64 = 2;

/// `MTLDataType` values of a texture's texel type (iPhoneOS SDK, `MTLArgument.h`).
mod data_type {
    pub const FLOAT: u64 = 3;
    pub const HALF: u64 = 16;
    pub const INT: u64 = 29;
    pub const UINT: u64 = 33;
    pub const SHORT: u64 = 37;
    pub const USHORT: u64 = 41;
}

/// The `MTLTextureType` names, by value (iPhoneOS SDK, `MTLTexture.h`).
const TEXTURE_TYPES: [&str; 10] = [
    "MTLTextureType1D",
    "MTLTextureType1DArray",
    "MTLTextureType2D",
    "MTLTextureType2DArray",
    "MTLTextureType2DMultisample",
    "MTLTextureTypeCube",
    "MTLTextureTypeCubeArray",
    "MTLTextureType3D",
    "MTLTextureType2DMultisampleArray",
    "MTLTextureTypeTextureBuffer",
];

fn texture_type_name(value: u64) -> String {
    TEXTURE_TYPES.get(value as usize).map_or_else(|| format!("MTLTextureType {value}"), |n| n.to_string())
}

/// A texture argument of a function, as its reflection gives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureArgument {
    pub name: String,
    /// The first texture index and the number of indices (an array of textures takes several).
    pub index: u64,
    pub count: u64,
    /// `MTLTextureType`.
    pub texture_type: u64,
    /// `MTLDataType` of the texel type.
    pub data_type: u64,
    /// A `depth*` texture argument.
    pub is_depth: bool,
}

/// The `MTLTextureType` value of a texture's (or view's) type.
pub fn texture_type_value(kind: crate::mem::layout::TextureType) -> u64 {
    use crate::mem::layout::TextureType as T;
    match kind {
        T::D1 => 0,
        T::D1Array => 1,
        T::D2 => 2,
        T::D2Array => 3,
        T::D2Multisample => 4,
        T::Cube => 5,
        T::CubeArray => 6,
        T::D3 => 7,
        T::D2MultisampleArray => 8,
        T::TextureBuffer => 9,
    }
}

/// The texture arguments of one stage's reflection.
pub fn texture_arguments(stage: &airwgsl::StageReflection) -> Vec<TextureArgument> {
    stage
        .arguments
        .iter()
        .filter(|a| a.r#type == TEXTURE)
        .map(|a| TextureArgument {
            name: a.name.clone(),
            index: a.index,
            count: a.array_length.max(1),
            texture_type: a.texture_type,
            data_type: a.texture_data_type,
            is_depth: a.is_depth_texture,
        })
        .collect()
}

/// The argument among `arguments` that takes texture index `slot`.
pub fn argument_at(arguments: &[TextureArgument], slot: u64) -> Option<&TextureArgument> {
    arguments.iter().find(|a| slot >= a.index && slot - a.index < a.count)
}

/// Why `argument` cannot take a texture of `texture_type` and `pixel_format` bound at index `slot`, or `None` when it
/// takes it. The rule is the compiler's (`tex::check_slot`): the texture type is the argument's, a `depth*` argument
/// takes a depth format, a float or half argument a format of the float or depth class (MSL 4.1 section 2.9), an
/// integer argument a format of its signedness (a stencil format is unsigned). A format the compiler does not know is
/// left to it.
pub fn texture_mismatch(argument: &TextureArgument, slot: u64, texture_type: u64, pixel_format: u64) -> Option<String> {
    let at = || format!("texture index {slot} (argument {})", argument.name);
    if texture_type != argument.texture_type {
        return Some(format!(
            "{} is declared {}, and the bound texture is {}",
            at(),
            texture_type_name(argument.texture_type),
            texture_type_name(texture_type)
        ));
    }
    let f = format(pixel_format)?;
    let ok = match (f.class, argument.is_depth, argument.data_type) {
        (Class::Depth, true, _) => true,
        (_, true, _) => false,
        (Class::Float | Class::Depth, false, data_type::FLOAT | data_type::HALF) => true,
        (Class::Sint, false, data_type::INT | data_type::SHORT) => true,
        (Class::Uint | Class::Stencil, false, data_type::UINT | data_type::USHORT) => true,
        (_, false, data_type::FLOAT | data_type::HALF | data_type::INT | data_type::SHORT)
        | (_, false, data_type::UINT | data_type::USHORT) => false,
        // A texel type outside the texture element types: the compiler decides.
        _ => true,
    };
    (!ok).then(|| {
        let declared = if argument.is_depth {
            "a depth texture".to_string()
        } else {
            format!("with texel MTLDataType {}", argument.data_type)
        };
        format!("{} is declared {declared}, and the bound texture's format is {}", at(), f.name)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arg(texture_type: u64, data_type: u64, is_depth: bool) -> TextureArgument {
        TextureArgument { name: "t".into(), index: 2, count: 1, texture_type, data_type, is_depth }
    }

    #[test]
    fn the_compilers_slot_rule() {
        // MTLPixelFormatRGBA8Unorm 70, RGBA8Uint 73, RGBA8Sint 74, Depth32Float 252, Stencil8 253.
        let float2d = arg(2, data_type::FLOAT, false);
        assert_eq!(texture_mismatch(&float2d, 2, 2, 70), None);
        assert_eq!(texture_mismatch(&float2d, 2, 2, 252), None, "a depth format reads as float");
        let m = texture_mismatch(&float2d, 2, 3, 70).unwrap();
        assert!(m.contains("MTLTextureType2D,") && m.contains("MTLTextureType2DArray"), "{m}");
        assert!(texture_mismatch(&float2d, 2, 2, 73).unwrap().contains("RGBA8Uint"));
        assert_eq!(texture_mismatch(&arg(2, data_type::UINT, false), 2, 2, 73), None);
        assert_eq!(texture_mismatch(&arg(2, data_type::USHORT, false), 2, 2, 253), None);
        assert!(texture_mismatch(&arg(2, data_type::INT, false), 2, 2, 73).is_some());
        assert_eq!(texture_mismatch(&arg(2, data_type::SHORT, false), 2, 2, 74), None);
        let depth = arg(2, data_type::FLOAT, true);
        assert_eq!(texture_mismatch(&depth, 2, 2, 252), None);
        assert!(texture_mismatch(&depth, 2, 2, 70).unwrap().contains("depth texture"));
        assert!(texture_mismatch(&arg(4, data_type::FLOAT, false), 2, 2, 70).unwrap().contains("2DMultisample"));
    }

    #[test]
    fn arrays_of_textures_take_their_indices() {
        let a = TextureArgument { count: 3, ..arg(2, data_type::FLOAT, false) };
        let args = [a];
        assert!(argument_at(&args, 1).is_none());
        assert!(argument_at(&args, 4).is_some());
        assert!(argument_at(&args, 5).is_none());
    }
}
