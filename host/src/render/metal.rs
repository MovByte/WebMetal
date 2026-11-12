//! The Metal values the render path reads from the stream, with their WebGPU counterparts.
//!
//! Every enumerator value is from the iPhoneOS SDK headers (`MTLRenderCommandEncoder.h`, `MTLRenderPass.h`,
//! `MTLDepthStencil.h`, `MTLArgument.h`, `MTLCommandEncoder.h`, `MTLRenderPipeline.h`). A value outside an enumeration
//! is not something Metal defines: it is reported as undefined, with the call that carried it.

use crate::gpu::idl::StencilOperation as GpuStencilOp;
use crate::gpu::idl::{CompareFunction as GpuCompare, CullMode as GpuCull, FrontFace, IndexFormat, PrimitiveTopology};

use super::error::{RenderError, Result};

fn bad(call: &str, what: &str, v: u64, header: &str) -> RenderError {
    RenderError::undefined(call, format!("{v} is not an {what} value ({header})"))
}

/// `MTLPrimitiveType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveType {
    Point,
    Line,
    LineStrip,
    Triangle,
    TriangleStrip,
}

impl PrimitiveType {
    pub fn from_value(v: u64, call: &str) -> Result<PrimitiveType> {
        Ok(match v {
            0 => PrimitiveType::Point,
            1 => PrimitiveType::Line,
            2 => PrimitiveType::LineStrip,
            3 => PrimitiveType::Triangle,
            4 => PrimitiveType::TriangleStrip,
            v => return Err(bad(call, "MTLPrimitiveType", v, "MTLRenderCommandEncoder.h")),
        })
    }

    pub fn topology(self) -> PrimitiveTopology {
        match self {
            PrimitiveType::Point => PrimitiveTopology::PointList,
            PrimitiveType::Line => PrimitiveTopology::LineList,
            PrimitiveType::LineStrip => PrimitiveTopology::LineStrip,
            PrimitiveType::Triangle => PrimitiveTopology::TriangleList,
            PrimitiveType::TriangleStrip => PrimitiveTopology::TriangleStrip,
        }
    }

    pub fn is_strip(self) -> bool {
        matches!(self, PrimitiveType::LineStrip | PrimitiveType::TriangleStrip)
    }

    pub fn is_triangle(self) -> bool {
        matches!(self, PrimitiveType::Triangle | PrimitiveType::TriangleStrip)
    }

    /// Vertices per primitive of a list topology, or the vertices of the first primitive of a strip.
    pub fn vertices_per_primitive(self) -> u32 {
        match self {
            PrimitiveType::Point => 1,
            PrimitiveType::Line | PrimitiveType::LineStrip => 2,
            PrimitiveType::Triangle | PrimitiveType::TriangleStrip => 3,
        }
    }

    /// The number of primitives `count` vertices assemble into, without primitive restart.
    pub fn primitives(self, count: u32) -> u32 {
        let n = self.vertices_per_primitive();
        match self {
            PrimitiveType::Point | PrimitiveType::Line | PrimitiveType::Triangle => count / n,
            PrimitiveType::LineStrip | PrimitiveType::TriangleStrip => (count + 1).saturating_sub(n),
        }
    }

    /// The topology class a render pipeline's `inputPrimitiveTopology` names for this type.
    pub fn class(self) -> TopologyClass {
        match self {
            PrimitiveType::Point => TopologyClass::Point,
            PrimitiveType::Line | PrimitiveType::LineStrip => TopologyClass::Line,
            PrimitiveType::Triangle | PrimitiveType::TriangleStrip => TopologyClass::Triangle,
        }
    }
}

/// `MTLPrimitiveTopologyClass` (`MTLRenderPipeline.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TopologyClass {
    Unspecified,
    Point,
    Line,
    Triangle,
}

impl TopologyClass {
    pub fn from_value(v: u64, call: &str) -> Result<TopologyClass> {
        Ok(match v {
            0 => TopologyClass::Unspecified,
            1 => TopologyClass::Point,
            2 => TopologyClass::Line,
            3 => TopologyClass::Triangle,
            v => return Err(bad(call, "MTLPrimitiveTopologyClass", v, "MTLRenderPipeline.h")),
        })
    }
}

/// `MTLIndexType` (`MTLArgument.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexType {
    UInt16,
    UInt32,
}

impl IndexType {
    pub fn from_value(v: u64, call: &str) -> Result<IndexType> {
        Ok(match v {
            0 => IndexType::UInt16,
            1 => IndexType::UInt32,
            v => return Err(bad(call, "MTLIndexType", v, "MTLArgument.h")),
        })
    }

    pub fn size(self) -> u64 {
        match self {
            IndexType::UInt16 => 2,
            IndexType::UInt32 => 4,
        }
    }

    /// The primitive restart sentinel: "the largest unsigned integer possible for indexType" (Apple's documentation
    /// of `drawIndexedPrimitives`, `tables/limits.toml` question `6.5-primitive-restart`).
    pub fn restart(self) -> u32 {
        match self {
            IndexType::UInt16 => 0xffff,
            IndexType::UInt32 => 0xffff_ffff,
        }
    }

    pub fn format(self) -> IndexFormat {
        match self {
            IndexType::UInt16 => IndexFormat::Uint16,
            IndexType::UInt32 => IndexFormat::Uint32,
        }
    }
}

/// `MTLLoadAction` (`MTLRenderPass.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadAction {
    DontCare,
    Load,
    Clear,
}

impl LoadAction {
    pub fn from_value(v: u64, call: &str) -> Result<LoadAction> {
        Ok(match v {
            0 => LoadAction::DontCare,
            1 => LoadAction::Load,
            2 => LoadAction::Clear,
            v => return Err(bad(call, "MTLLoadAction", v, "MTLRenderPass.h")),
        })
    }
}

/// `MTLStoreAction` (`MTLRenderPass.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreAction {
    DontCare,
    Store,
    MultisampleResolve,
    StoreAndMultisampleResolve,
    Unknown,
    CustomSampleDepthStore,
}

impl StoreAction {
    pub fn from_value(v: u64, call: &str) -> Result<StoreAction> {
        Ok(match v {
            0 => StoreAction::DontCare,
            1 => StoreAction::Store,
            2 => StoreAction::MultisampleResolve,
            3 => StoreAction::StoreAndMultisampleResolve,
            4 => StoreAction::Unknown,
            5 => StoreAction::CustomSampleDepthStore,
            v => return Err(bad(call, "MTLStoreAction", v, "MTLRenderPass.h")),
        })
    }

    /// Does the attachment keep its contents after the pass?
    pub fn stores(self) -> bool {
        matches!(self, StoreAction::Store | StoreAction::StoreAndMultisampleResolve)
    }

    /// Does the pass resolve the attachment?
    pub fn resolves(self) -> bool {
        matches!(self, StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve)
    }
}

/// `MTLMultisampleDepthResolveFilter` (`MTLRenderPass.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthResolveFilter {
    Sample0,
    Min,
    Max,
}

impl DepthResolveFilter {
    pub fn from_value(v: u64, call: &str) -> Result<DepthResolveFilter> {
        Ok(match v {
            0 => DepthResolveFilter::Sample0,
            1 => DepthResolveFilter::Min,
            2 => DepthResolveFilter::Max,
            v => return Err(bad(call, "MTLMultisampleDepthResolveFilter", v, "MTLRenderPass.h")),
        })
    }
}

/// `MTLMultisampleStencilResolveFilter` (`MTLRenderPass.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilResolveFilter {
    Sample0,
    DepthResolvedSample,
}

impl StencilResolveFilter {
    pub fn from_value(v: u64, call: &str) -> Result<StencilResolveFilter> {
        Ok(match v {
            0 => StencilResolveFilter::Sample0,
            1 => StencilResolveFilter::DepthResolvedSample,
            v => return Err(bad(call, "MTLMultisampleStencilResolveFilter", v, "MTLRenderPass.h")),
        })
    }
}

/// `MTLCompareFunction` (`MTLDepthStencil.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl CompareFunction {
    pub fn from_value(v: u64, call: &str) -> Result<CompareFunction> {
        Ok(match v {
            0 => CompareFunction::Never,
            1 => CompareFunction::Less,
            2 => CompareFunction::Equal,
            3 => CompareFunction::LessEqual,
            4 => CompareFunction::Greater,
            5 => CompareFunction::NotEqual,
            6 => CompareFunction::GreaterEqual,
            7 => CompareFunction::Always,
            v => return Err(bad(call, "MTLCompareFunction", v, "MTLDepthStencil.h")),
        })
    }

    /// The WebGPU function with the same definition: both APIs compare the reference (or fragment) value on the left
    /// with the stored value on the right.
    pub fn gpu(self) -> GpuCompare {
        match self {
            CompareFunction::Never => GpuCompare::Never,
            CompareFunction::Less => GpuCompare::Less,
            CompareFunction::Equal => GpuCompare::Equal,
            CompareFunction::LessEqual => GpuCompare::LessEqual,
            CompareFunction::Greater => GpuCompare::Greater,
            CompareFunction::NotEqual => GpuCompare::NotEqual,
            CompareFunction::GreaterEqual => GpuCompare::GreaterEqual,
            CompareFunction::Always => GpuCompare::Always,
        }
    }

    /// Does the outcome depend on the values compared? `Always` and `Never` do not.
    pub fn reads_values(self) -> bool {
        !matches!(self, CompareFunction::Always | CompareFunction::Never)
    }
}

/// `MTLStencilOperation` (`MTLDepthStencil.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StencilOperation {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

impl StencilOperation {
    pub fn from_value(v: u64, call: &str) -> Result<StencilOperation> {
        Ok(match v {
            0 => StencilOperation::Keep,
            1 => StencilOperation::Zero,
            2 => StencilOperation::Replace,
            3 => StencilOperation::IncrementClamp,
            4 => StencilOperation::DecrementClamp,
            5 => StencilOperation::Invert,
            6 => StencilOperation::IncrementWrap,
            7 => StencilOperation::DecrementWrap,
            v => return Err(bad(call, "MTLStencilOperation", v, "MTLDepthStencil.h")),
        })
    }

    pub fn gpu(self) -> GpuStencilOp {
        match self {
            StencilOperation::Keep => GpuStencilOp::Keep,
            StencilOperation::Zero => GpuStencilOp::Zero,
            StencilOperation::Replace => GpuStencilOp::Replace,
            StencilOperation::IncrementClamp => GpuStencilOp::IncrementClamp,
            StencilOperation::DecrementClamp => GpuStencilOp::DecrementClamp,
            StencilOperation::Invert => GpuStencilOp::Invert,
            StencilOperation::IncrementWrap => GpuStencilOp::IncrementWrap,
            StencilOperation::DecrementWrap => GpuStencilOp::DecrementWrap,
        }
    }
}

/// `MTLCullMode` (`MTLRenderCommandEncoder.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CullMode {
    None,
    Front,
    Back,
}

impl CullMode {
    pub fn from_value(v: u64, call: &str) -> Result<CullMode> {
        Ok(match v {
            0 => CullMode::None,
            1 => CullMode::Front,
            2 => CullMode::Back,
            v => return Err(bad(call, "MTLCullMode", v, "MTLRenderCommandEncoder.h")),
        })
    }

    pub fn gpu(self) -> GpuCull {
        match self {
            CullMode::None => GpuCull::None,
            CullMode::Front => GpuCull::Front,
            CullMode::Back => GpuCull::Back,
        }
    }
}

/// `MTLWinding` (`MTLRenderCommandEncoder.h`): which screen-space orientation of a triangle is front-facing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Winding {
    Clockwise,
    CounterClockwise,
}

impl Winding {
    pub fn from_value(v: u64, call: &str) -> Result<Winding> {
        Ok(match v {
            0 => Winding::Clockwise,
            1 => Winding::CounterClockwise,
            v => return Err(bad(call, "MTLWinding", v, "MTLRenderCommandEncoder.h")),
        })
    }

    /// The WebGPU front face. Both APIs judge orientation in framebuffer coordinates with the origin at the top left
    /// and y down (`tables/limits.toml` question `5.10-clip-space-and-framebuffer-origin`). WebGPU section 23.2.5.4
    /// computes `area = 0.5 × ((v1.x × vn.y − vn.x × v1.y) + Σ (vi+1.x × vi.y − vi.x × vi+1.y))`, the negated shoelace
    /// sum, and "cw" takes `area < 0` as front-facing: a triangle whose vertices run clockwise on screen. That is
    /// Metal's `MTLWindingClockwise` (reference case `raster.cull-winding`).
    pub fn gpu(self) -> FrontFace {
        match self {
            Winding::Clockwise => FrontFace::Cw,
            Winding::CounterClockwise => FrontFace::Ccw,
        }
    }

    pub fn flipped(self) -> Winding {
        match self {
            Winding::Clockwise => Winding::CounterClockwise,
            Winding::CounterClockwise => Winding::Clockwise,
        }
    }
}

/// `MTLDepthClipMode` (`MTLRenderCommandEncoder.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DepthClipMode {
    Clip,
    Clamp,
}

impl DepthClipMode {
    pub fn from_value(v: u64, call: &str) -> Result<DepthClipMode> {
        Ok(match v {
            0 => DepthClipMode::Clip,
            1 => DepthClipMode::Clamp,
            v => return Err(bad(call, "MTLDepthClipMode", v, "MTLRenderCommandEncoder.h")),
        })
    }
}

/// `MTLTriangleFillMode` (`MTLRenderCommandEncoder.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillMode {
    Fill,
    Lines,
}

impl FillMode {
    pub fn from_value(v: u64, call: &str) -> Result<FillMode> {
        Ok(match v {
            0 => FillMode::Fill,
            1 => FillMode::Lines,
            v => return Err(bad(call, "MTLTriangleFillMode", v, "MTLRenderCommandEncoder.h")),
        })
    }
}

/// `MTLVisibilityResultMode` (`MTLRenderCommandEncoder.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisibilityMode {
    Disabled,
    Boolean,
    Counting,
}

impl VisibilityMode {
    pub fn from_value(v: u64, call: &str) -> Result<VisibilityMode> {
        Ok(match v {
            0 => VisibilityMode::Disabled,
            1 => VisibilityMode::Boolean,
            2 => VisibilityMode::Counting,
            v => return Err(bad(call, "MTLVisibilityResultMode", v, "MTLRenderCommandEncoder.h")),
        })
    }
}

/// `MTLRenderStages` bits (`MTLRenderCommandEncoder.h`).
pub mod stages {
    pub const VERTEX: u64 = 1 << 0;
    pub const FRAGMENT: u64 = 1 << 1;
    pub const TILE: u64 = 1 << 2;
    pub const OBJECT: u64 = 1 << 3;
    pub const MESH: u64 = 1 << 4;
    pub const ALL: u64 = VERTEX | FRAGMENT | TILE | OBJECT | MESH;
}

/// `MTLBarrierScope` bits (`MTLCommandEncoder.h`). `MTLBarrierScopeRenderTargets` is `API_UNAVAILABLE(ios)`.
pub mod barrier_scope {
    pub const BUFFERS: u64 = 1 << 0;
    pub const TEXTURES: u64 = 1 << 1;
    pub const RENDER_TARGETS: u64 = 1 << 2;
}

/// `MTLPixelFormatInvalid`.
pub const PIXEL_FORMAT_INVALID: u32 = 0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_follow_the_sdk() {
        assert_eq!(PrimitiveType::from_value(4, "t").unwrap(), PrimitiveType::TriangleStrip);
        assert!(PrimitiveType::from_value(5, "t").is_err());
        assert_eq!(StoreAction::from_value(3, "t").unwrap(), StoreAction::StoreAndMultisampleResolve);
        assert_eq!(StoreAction::from_value(4, "t").unwrap(), StoreAction::Unknown);
        assert_eq!(CompareFunction::from_value(7, "t").unwrap().gpu(), GpuCompare::Always);
        assert_eq!(StencilOperation::from_value(5, "t").unwrap().gpu(), GpuStencilOp::Invert);
        assert_eq!(StencilOperation::from_value(6, "t").unwrap().gpu(), GpuStencilOp::IncrementWrap);
        assert_eq!(Winding::from_value(0, "t").unwrap().gpu(), FrontFace::Cw);
        assert_eq!(IndexType::from_value(1, "t").unwrap().restart(), 0xffff_ffff);
        assert!(LoadAction::from_value(3, "t").is_err());
    }

    #[test]
    fn primitive_counts() {
        assert_eq!(PrimitiveType::Triangle.primitives(7), 2);
        assert_eq!(PrimitiveType::TriangleStrip.primitives(5), 3);
        assert_eq!(PrimitiveType::TriangleStrip.primitives(2), 0);
        assert_eq!(PrimitiveType::LineStrip.primitives(4), 3);
        assert_eq!(PrimitiveType::Line.primitives(5), 2);
        assert_eq!(PrimitiveType::Point.primitives(3), 3);
    }
}
