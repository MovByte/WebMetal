//! WGSL translations of the reference harness's shaders (`tests/reference/shaders/render.metal`,
//! `render_pass.metal`, `formats.metal`), standing in for the shader compiler (plan section 4.3) so the render path
//! can be checked before it exists. Each translation keeps the MSL function's interface (argument table slots,
//! varyings, outputs) and arithmetic.
//!
//! Binding plan: vertex-stage resources are in group 0, fragment-stage resources in group 1, compute in group 0;
//! within a group, bindings follow the list given here.
//!
//! `rnd_vertex` writes `[[point_size]]` from `extra.x`; WGSL has no point size output and WebGPU rasterizes points one
//! pixel wide, so this translation is used only by cases whose points have size 1 (every case but `raster.points`).

use webmetal_host::gpu::idl::{BufferBindingType, ShaderStage, TextureSampleType, TextureViewDimension};
use webmetal_host::render::pipeline::{Binding, MetalStage, Resource, Source};

/// A function's stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Vertex,
    Fragment,
    Compute,
}

/// A translated function.
pub struct Func {
    pub kind: Kind,
    pub wgsl: String,
    pub bindings: Vec<Binding>,
    /// Vertex functions with `[[stage_in]]`: native vertex input from the pipeline's vertex descriptor.
    pub stage_in: bool,
    /// Fragment functions that read `[[color(n)]]`.
    pub fetch: Vec<u32>,
    /// The WebGPU feature the translation needs.
    pub feature: Option<&'static str>,
}

const RVERTEX: &str = r#"
struct RVertex { position: vec4<f32>, data: vec4<u32>, extra: vec4<f32> }
@group(0) @binding(0) var<storage, read> vertices: array<RVertex>;
"#;

const VARYINGS: &str = r#"
struct V { @builtin(position) position: vec4<f32>, @location(0) @interpolate(flat) data: vec4<u32> }
"#;

fn buffer(stage: MetalStage, index: u64, binding: u32, ty: BufferBindingType, min: u64) -> Binding {
    let visibility = match stage {
        MetalStage::Vertex => ShaderStage::VERTEX,
        MetalStage::Fragment => ShaderStage::FRAGMENT,
    };
    Binding {
        binding,
        visibility,
        texture: None,
        resource: Resource::Buffer { ty, min_binding_size: min },
        source: Source::MetalBuffer { stage, index },
    }
}

fn vertices() -> Vec<Binding> {
    vec![buffer(MetalStage::Vertex, 0, 0, BufferBindingType::ReadOnlyStorage, 48)]
}

fn vertex(wgsl: &str) -> Func {
    Func {
        kind: Kind::Vertex,
        wgsl: format!("{RVERTEX}{VARYINGS}{wgsl}"),
        bindings: vertices(),
        stage_in: false,
        fetch: vec![],
        feature: None,
    }
}

fn fragment(wgsl: &str) -> Func {
    Func {
        kind: Kind::Fragment,
        wgsl: format!("{VARYINGS}{wgsl}"),
        bindings: vec![],
        stage_in: false,
        fetch: vec![],
        feature: None,
    }
}

/// The translation of `name`, if there is one.
pub fn function(name: &str) -> Option<Func> {
    Some(match name {
        "rnd_vertex_triangles" | "rnd_vertex" => vertex(
            "@vertex fn main(@builtin(vertex_index) vid: u32) -> V {
                let v = vertices[vid];
                return V(v.position, v.data);
            }",
        ),
        "rnd_fragment_id" | "rnd_fragment_point_id" => {
            fragment("@fragment fn main(i: V) -> @location(0) u32 { return i.data.x; }")
        }
        "rnd_fragment_data" | "rnd_fragment_id_data" => {
            fragment("@fragment fn main(i: V) -> @location(0) vec4<u32> { return i.data; }")
        }
        "rnd_fragment_float" | "rnd_fragment_float_point" => {
            fragment("@fragment fn main(i: V) -> @location(0) vec4<f32> { return bitcast<vec4<f32>>(i.data); }")
        }
        "rnd_fragment_facing" => fragment(
            "@fragment fn main(i: V, @builtin(front_facing) front: bool) -> @location(0) u32 {
                return select(0u, 0x100u, front) | i.data.x;
            }",
        ),
        "rnd_fragment_builtins" => fragment(
            "@fragment fn main(i: V, @builtin(front_facing) front: bool) -> @location(0) vec4<u32> {
                return vec4<u32>(bitcast<u32>(i.position.x), bitcast<u32>(i.position.y), bitcast<u32>(i.position.z),
                                 select(0u, 0x100u, front) | i.data.x);
            }",
        ),
        "rnd_fragment_position_w" => fragment(
            "@fragment fn main(i: V) -> @location(0) vec4<u32> { return bitcast<vec4<u32>>(i.position); }",
        ),
        "rnd_fragment_discard" => fragment(
            "@fragment fn main(i: V) -> @location(0) u32 {
                let p = vec2<u32>(i.position.xy);
                if ((p.x + p.y) % 3u == 0u) { discard; }
                return i.data.x;
            }",
        ),
        "rnd_fragment_depth_any" => fragment(
            "struct O { @location(0) id: u32, @builtin(frag_depth) depth: f32 }
            @fragment fn main(i: V) -> O {
                let p = vec2<u32>(i.position.xy);
                return O(i.data.x, f32(p.y * 16u + p.x) / 256.0);
            }",
        ),
        "rnd_fragment_depth_greater" => fragment(
            "struct O { @location(0) id: u32, @builtin(frag_depth) depth: f32 }
            @fragment fn main(i: V) -> O {
                let p = vec2<u32>(i.position.xy);
                return O(i.data.x, i.position.z + f32(p.x) / 64.0);
            }",
        ),
        "rnd_fragment_depth_less" => fragment(
            "struct O { @location(0) id: u32, @builtin(frag_depth) depth: f32 }
            @fragment fn main(i: V) -> O {
                let p = vec2<u32>(i.position.xy);
                return O(i.data.x, i.position.z - f32(p.x) / 64.0);
            }",
        ),
        "rnd_fragment_alpha" => fragment(
            "@fragment fn main(i: V) -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 1.0, 1.0, bitcast<f32>(i.data.x));
            }",
        ),
        "rnd_fragment_sample_id" => fragment(
            "@fragment fn main(i: V, @builtin(sample_index) s: u32) -> @location(0) u32 { return i.data.x * 16u + s; }",
        ),
        "rnd_fragment_mask_out" => fragment(
            "struct O { @location(0) color: u32, @builtin(sample_mask) mask: u32 }
            @fragment fn main(i: V) -> O { return O(i.data.x, i.data.y); }",
        ),
        "rnd_fragment_mask_in" => fragment(
            "@fragment fn main(i: V, @builtin(sample_mask) mask: u32) -> @location(0) u32 {
                return 0x100u * i.data.x + mask;
            }",
        ),
        "rnd_fragment_sample_depth" => fragment(
            "struct O { @location(0) color: u32, @builtin(frag_depth) depth: f32 }
            @fragment fn main(i: V, @builtin(sample_index) s: u32) -> O {
                var table = array<f32, 4>(0.25, 0.125, 0.5, 0.375);
                if (s == 0u) { discard; }
                return O(i.data.x, table[s & 3u]);
            }",
        ),
        "rnd_fragment_dual_source" => {
            let mut f = fragment(
                "enable dual_source_blending;
                @group(1) @binding(0) var<storage, read> second: vec4<f32>;
                struct O { @location(0) @blend_src(0) color: vec4<f32>, @location(0) @blend_src(1) second: vec4<f32> }
                @fragment fn main(i: V) -> O { return O(bitcast<vec4<f32>>(i.data), second); }",
            );
            // `enable` must come first in a module.
            f.wgsl = format!("enable dual_source_blending;\n{}", f.wgsl.replace("enable dual_source_blending;", ""));
            f.bindings = vec![buffer(MetalStage::Fragment, 0, 0, BufferBindingType::ReadOnlyStorage, 16)];
            f.feature = Some("dual-source-blending");
            f
        }
        "rnd_fragment_fetch" => {
            let mut f = fragment(
                "@group(1) @binding(0) var previous: texture_2d<u32>;
                @fragment fn main(i: V) -> @location(0) u32 {
                    return textureLoad(previous, vec2<i32>(i.position.xy), 0).x * 3u + i.data.x;
                }",
            );
            f.bindings = vec![Binding {
                binding: 0,
                visibility: ShaderStage::FRAGMENT,
                texture: None,
                resource: Resource::Texture {
                    sample_type: TextureSampleType::Uint,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                source: Source::FetchSnapshot { attachment: 0 },
            }];
            f.fetch = vec![0];
            f
        }
        "rnd_fragment_load" => {
            let mut f = fragment(
                "@group(1) @binding(0) var<storage, read> values: array<u32>;
                @fragment fn main(i: V) -> @location(0) u32 {
                    let p = vec2<u32>(i.position.xy);
                    return values[p.y * 8u + p.x] + i.data.x;
                }",
            );
            f.bindings = vec![buffer(MetalStage::Fragment, 0, 0, BufferBindingType::ReadOnlyStorage, 4)];
            f
        }
        "rnd_fragment_store" => {
            let mut f = fragment(
                "@group(1) @binding(0) var<storage, read_write> out: array<u32>;
                @fragment fn main(i: V) -> @location(0) u32 {
                    let p = vec2<u32>(i.position.xy);
                    out[p.y * 8u + p.x] = i.data.x * 100u + p.y * 8u + p.x;
                    return 0u;
                }",
            );
            f.bindings = vec![buffer(MetalStage::Fragment, 0, 0, BufferBindingType::Storage, 4)];
            f
        }
        "rnd_vertex_ids" => Func {
            kind: Kind::Vertex,
            wgsl: format!(
                "{VARYINGS}
                fn centre(x: u32, y: u32) -> vec4<f32> {{
                    return vec4<f32>((f32(x) + 0.5) / 8.0 - 1.0, 1.0 - (f32(y) + 0.5) / 8.0, 0.0, 1.0);
                }}
                @vertex fn main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> V {{
                    return V(centre(vid % 16u, iid % 16u), vec4<u32>(vid + 1u, iid + 1u, 0xa5u, 0u));
                }}"
            ),
            bindings: vec![],
            stage_in: false,
            fetch: vec![],
            feature: None,
        },
        "rnd_vertex_fetch_uint" => Func {
            kind: Kind::Vertex,
            wgsl: format!(
                "{VARYINGS}
                fn centre(x: u32, y: u32) -> vec4<f32> {{
                    return vec4<f32>((f32(x) + 0.5) / 8.0 - 1.0, 1.0 - (f32(y) + 0.5) / 8.0, 0.0, 1.0);
                }}
                @vertex fn main(@location(0) value: vec4<u32>, @builtin(vertex_index) vid: u32,
                                @builtin(instance_index) iid: u32) -> V {{
                    return V(centre(vid % 16u, iid % 16u), value);
                }}"
            ),
            bindings: vec![],
            stage_in: true,
            fetch: vec![],
            feature: None,
        },
        "rnd_vertex_fetch_two" => Func {
            kind: Kind::Vertex,
            wgsl: format!(
                "{VARYINGS}
                fn centre(x: u32, y: u32) -> vec4<f32> {{
                    return vec4<f32>((f32(x) + 0.5) / 8.0 - 1.0, 1.0 - (f32(y) + 0.5) / 8.0, 0.0, 1.0);
                }}
                @vertex fn main(@location(0) a: vec4<u32>, @location(1) b: vec4<u32>, @builtin(vertex_index) vid: u32,
                                @builtin(instance_index) iid: u32) -> V {{
                    return V(centre(vid % 16u, iid % 16u), vec4<u32>(a.x, a.y, b.x, b.y));
                }}"
            ),
            bindings: vec![],
            stage_in: true,
            fetch: vec![],
            feature: None,
        },
        "rnd_vertex_interpolated" => Func {
            kind: Kind::Vertex,
            wgsl: format!(
                "{RVERTEX}
                struct I {{
                    @builtin(position) position: vec4<f32>,
                    @location(0) @interpolate(perspective, center) perspective: f32,
                    @location(1) @interpolate(linear, center) linear: f32,
                    @location(2) @interpolate(perspective, centroid) centroid: f32,
                    @location(3) @interpolate(flat) id: u32,
                }}
                @vertex fn main(@builtin(vertex_index) vid: u32) -> I {{
                    let v = vertices[vid];
                    return I(v.position, v.extra.y, v.extra.y, v.extra.y, v.data.x);
                }}"
            ),
            bindings: vertices(),
            stage_in: false,
            fetch: vec![],
            feature: None,
        },
        "rnd_fragment_interpolated" => Func {
            kind: Kind::Fragment,
            wgsl: "struct I {
                    @builtin(position) position: vec4<f32>,
                    @location(0) @interpolate(perspective, center) perspective: f32,
                    @location(1) @interpolate(linear, center) linear: f32,
                    @location(2) @interpolate(perspective, centroid) centroid: f32,
                    @location(3) @interpolate(flat) id: u32,
                }
                @fragment fn main(i: I) -> @location(0) vec4<u32> {
                    return vec4<u32>(bitcast<u32>(i.perspective), bitcast<u32>(i.linear), bitcast<u32>(i.centroid), i.id);
                }"
            .into(),
            bindings: vec![],
            stage_in: false,
            fetch: vec![],
            feature: None,
        },
        "sto_fullscreen_vertex" => Func {
            kind: Kind::Vertex,
            wgsl: "@group(0) @binding(0) var<storage, read> depth: f32;
                @vertex fn main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
                    var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
                    return vec4<f32>(corners[vid], depth, 1.0);
                }"
            .into(),
            bindings: vec![buffer(MetalStage::Vertex, 0, 0, BufferBindingType::ReadOnlyStorage, 4)],
            stage_in: false,
            fetch: vec![],
            feature: None,
        },
        "sto_color_fragment" => Func {
            kind: Kind::Fragment,
            wgsl: "@group(1) @binding(0) var<storage, read> color: vec4<u32>;
                @fragment fn main() -> @location(0) vec4<u32> { return color; }"
                .into(),
            bindings: vec![buffer(MetalStage::Fragment, 0, 0, BufferBindingType::ReadOnlyStorage, 16)],
            stage_in: false,
            fetch: vec![],
            feature: None,
        },
        "fmt_fullscreen_vertex" => Func {
            kind: Kind::Vertex,
            wgsl: "@vertex fn main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
                    let p = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
                    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
                }"
            .into(),
            bindings: vec![],
            stage_in: false,
            fetch: vec![],
            feature: None,
        },
        "fmt_fragment_float" | "fmt_fragment_uint" | "fmt_fragment_int" => {
            let t = match name {
                "fmt_fragment_float" => "vec4<f32>",
                "fmt_fragment_uint" => "vec4<u32>",
                _ => "vec4<i32>",
            };
            Func {
                kind: Kind::Fragment,
                wgsl: format!(
                    "@group(1) @binding(0) var<storage, read> values: array<{t}>;
                    @group(1) @binding(1) var<storage, read> width: u32;
                    @fragment fn main(@builtin(position) position: vec4<f32>) -> @location(0) {t} {{
                        return values[u32(position.y) * width + u32(position.x)];
                    }}"
                ),
                bindings: vec![
                    buffer(MetalStage::Fragment, 0, 0, BufferBindingType::ReadOnlyStorage, 16),
                    buffer(MetalStage::Fragment, 1, 1, BufferBindingType::ReadOnlyStorage, 4),
                ],
                stage_in: false,
                fetch: vec![],
                feature: None,
            }
        }
        "rnd_read_ms_uint" | "rnd_read_ms_float" | "rnd_read_ms_depth" => {
            let (texture, element, load) = match name {
                "rnd_read_ms_uint" => ("texture_multisampled_2d<u32>", "u32", "textureLoad(t, p, i32(s)).x"),
                "rnd_read_ms_float" => ("texture_multisampled_2d<f32>", "vec4<f32>", "textureLoad(t, p, i32(s))"),
                _ => ("texture_depth_multisampled_2d", "f32", "textureLoad(t, p, i32(s))"),
            };
            let size = if name == "rnd_read_ms_float" { 16 } else { 4 };
            let sample_type = match name {
                "rnd_read_ms_uint" => TextureSampleType::Uint,
                "rnd_read_ms_float" => TextureSampleType::UnfilterableFloat,
                _ => TextureSampleType::Depth,
            };
            Func {
                kind: Kind::Compute,
                wgsl: format!(
                    "@group(0) @binding(0) var t: {texture};
                    @group(0) @binding(1) var<storage, read_write> out: array<{element}>;
                    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                        let n = textureNumSamples(t);
                        let w = textureDimensions(t).x;
                        let p = vec2<i32>(gid.xy);
                        for (var s = 0u; s < n; s = s + 1u) {{
                            out[(gid.y * w + gid.x) * n + s] = {load};
                        }}
                    }}"
                ),
                bindings: vec![
                    Binding {
                        binding: 0,
                        visibility: ShaderStage::COMPUTE,
                        texture: None,
                        resource: Resource::Texture {
                            sample_type,
                            view_dimension: TextureViewDimension::D2,
                            multisampled: true,
                        },
                        source: Source::MetalTexture { stage: MetalStage::Fragment, index: 0 },
                    },
                    Binding {
                        binding: 1,
                        visibility: ShaderStage::COMPUTE,
                        texture: None,
                        resource: Resource::Buffer { ty: BufferBindingType::Storage, min_binding_size: size },
                        source: Source::MetalBuffer { stage: MetalStage::Fragment, index: 0 },
                    },
                ],
                stage_in: false,
                fetch: vec![],
                feature: None,
            }
        }
        _ => return None,
    })
}
