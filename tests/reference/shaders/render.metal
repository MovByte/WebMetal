// Shaders for the render cases (src/cases/Render*.m, Raster*.m, Draw*.m,
// DepthStencil*.m, Blend*.m).

#include <metal_stdlib>
using namespace metal;

// Vertex record shared with RenderSupport.m (48 bytes).
struct RVertex {
    float4 position;   // clip-space position
    uint4 data;        // passed to the fragment stage with flat interpolation
    float4 extra;      // x: point size, yzw: smooth varyings
};

struct RVaryings {
    float4 position [[position]];
    uint4 data [[flat]];
    float point_size [[point_size]];
    float3 smooth_value;
};

vertex RVaryings rnd_vertex(const device RVertex *vertices [[buffer(0)]], uint vid [[vertex_id]])
{
    RVertex v = vertices[vid];
    RVaryings out;
    out.position = v.position;
    out.data = v.data;
    out.point_size = v.extra.x;
    out.smooth_value = v.extra.yzw;
    return out;
}

// Same as rnd_vertex without [[point_size]], for pipelines that draw
// triangles and lines only.
struct RVaryingsNoPoint {
    float4 position [[position]];
    uint4 data [[flat]];
};

vertex RVaryingsNoPoint rnd_vertex_triangles(const device RVertex *vertices [[buffer(0)]],
                                             uint vid [[vertex_id]])
{
    RVaryingsNoPoint out;
    out.position = vertices[vid].position;
    out.data = vertices[vid].data;
    return out;
}

fragment uint rnd_fragment_id(RVaryingsNoPoint in [[stage_in]])
{
    return in.data.x;
}

fragment uint4 rnd_fragment_data(RVaryings in [[stage_in]])
{
    return in.data;
}

fragment uint rnd_fragment_point_id(RVaryings in [[stage_in]])
{
    return in.data.x;
}

fragment float4 rnd_fragment_float(RVaryingsNoPoint in [[stage_in]])
{
    return as_type<float4>(in.data);
}

// Fragment builtins: position (bits), front-facing flag and the flat id.
fragment uint4 rnd_fragment_builtins(RVaryingsNoPoint in [[stage_in]], bool front [[front_facing]])
{
    return uint4(as_type<uint>(in.position.x), as_type<uint>(in.position.y),
                 as_type<uint>(in.position.z), (front ? 0x100u : 0u) | in.data.x);
}

fragment uint4 rnd_fragment_position_w(RVaryingsNoPoint in [[stage_in]])
{
    return uint4(as_type<uint>(in.position.x), as_type<uint>(in.position.y),
                 as_type<uint>(in.position.z), as_type<uint>(in.position.w));
}

// Discards fragments whose pixel coordinates satisfy (x + y) % 3 == 0.
fragment uint rnd_fragment_discard(RVaryingsNoPoint in [[stage_in]])
{
    uint2 p = uint2(in.position.xy);
    if ((p.x + p.y) % 3u == 0u)
        discard_fragment();
    return in.data.x;
}

// Smooth varyings with the three interpolation qualifiers; written as bits.
struct RInterpolated {
    float4 position [[position]];
    float perspective [[center_perspective]];
    float linear [[center_no_perspective]];
    float centroid [[centroid_perspective]];
    uint id [[flat]];
};

vertex RInterpolated rnd_vertex_interpolated(const device RVertex *vertices [[buffer(0)]],
                                             uint vid [[vertex_id]])
{
    RInterpolated out;
    out.position = vertices[vid].position;
    out.perspective = vertices[vid].extra.y;
    out.linear = vertices[vid].extra.y;
    out.centroid = vertices[vid].extra.y;
    out.id = vertices[vid].data.x;
    return out;
}

fragment uint4 rnd_fragment_interpolated(RInterpolated in [[stage_in]])
{
    return uint4(as_type<uint>(in.perspective), as_type<uint>(in.linear),
                 as_type<uint>(in.centroid), in.id);
}

// Points placed by vertex and instance id on a 16x16 target: x = vertex id
// modulo 16, y = instance id modulo 16. The colour records both ids plus one.
struct RIdVaryings {
    float4 position [[position]];
    uint4 data [[flat]];
    float point_size [[point_size]];
};

static float4 rnd_pixel_centre(uint x, uint y)
{
    return float4((float(x) + 0.5f) / 8.0f - 1.0f, 1.0f - (float(y) + 0.5f) / 8.0f, 0.0f, 1.0f);
}

vertex RIdVaryings rnd_vertex_ids(uint vid [[vertex_id]], uint iid [[instance_id]])
{
    RIdVaryings out;
    out.position = rnd_pixel_centre(vid % 16u, iid % 16u);
    out.data = uint4(vid + 1u, iid + 1u, 0xa5u, 0u);
    out.point_size = 1.0f;
    return out;
}

// Vertex attribute fetch: the fetched value is passed through as bits and
// the point is placed at the pixel for its vertex id (x) and instance id (y).
struct RFetchUint { uint4 value [[attribute(0)]]; };
struct RFetchInt { int4 value [[attribute(0)]]; };
struct RFetchFloat { float4 value [[attribute(0)]]; };

vertex RIdVaryings rnd_vertex_fetch_uint(RFetchUint in [[stage_in]], uint vid [[vertex_id]],
                                         uint iid [[instance_id]])
{
    RIdVaryings out;
    out.position = rnd_pixel_centre(vid % 16u, iid % 16u);
    out.data = in.value;
    out.point_size = 1.0f;
    return out;
}

vertex RIdVaryings rnd_vertex_fetch_int(RFetchInt in [[stage_in]], uint vid [[vertex_id]],
                                        uint iid [[instance_id]])
{
    RIdVaryings out;
    out.position = rnd_pixel_centre(vid % 16u, iid % 16u);
    out.data = as_type<uint4>(in.value);
    out.point_size = 1.0f;
    return out;
}

vertex RIdVaryings rnd_vertex_fetch_float(RFetchFloat in [[stage_in]], uint vid [[vertex_id]],
                                          uint iid [[instance_id]])
{
    RIdVaryings out;
    out.position = rnd_pixel_centre(vid % 16u, iid % 16u);
    out.data = as_type<uint4>(in.value);
    out.point_size = 1.0f;
    return out;
}

// Two attributes from two buffers with different step functions.
struct RFetchTwo {
    uint4 a [[attribute(0)]];
    uint4 b [[attribute(1)]];
};

vertex RIdVaryings rnd_vertex_fetch_two(RFetchTwo in [[stage_in]], uint vid [[vertex_id]],
                                        uint iid [[instance_id]])
{
    RIdVaryings out;
    out.position = rnd_pixel_centre(vid % 16u, iid % 16u);
    out.data = uint4(in.a.x, in.a.y, in.b.x, in.b.y);
    out.point_size = 1.0f;
    return out;
}

// Vertex stage that writes to a buffer and is used with rasterization
// disabled.
vertex void rnd_vertex_store(device uint *out [[buffer(0)]], uint vid [[vertex_id]],
                             uint iid [[instance_id]])
{
    out[iid * 8u + vid] = (iid << 16) | (vid * 3u + 1u);
}

fragment uint rnd_fragment_facing(RVaryingsNoPoint in [[stage_in]], bool front [[front_facing]])
{
    return (front ? 0x100u : 0u) | in.data.x;
}

fragment uint4 rnd_fragment_id_data(RIdVaryings in [[stage_in]])
{
    return in.data;
}

// Fragment depth outputs. The written depth is a function of the pixel
// position with values exactly representable in binary32.
struct RDepthOut {
    uint id [[color(0)]];
    float depth [[depth(any)]];
};

fragment RDepthOut rnd_fragment_depth_any(RVaryingsNoPoint in [[stage_in]])
{
    uint2 p = uint2(in.position.xy);
    RDepthOut out;
    out.id = in.data.x;
    out.depth = float(p.y * 16u + p.x) / 256.0f;
    return out;
}

struct RDepthGreaterOut {
    uint id [[color(0)]];
    float depth [[depth(greater)]];
};

fragment RDepthGreaterOut rnd_fragment_depth_greater(RVaryingsNoPoint in [[stage_in]])
{
    uint2 p = uint2(in.position.xy);
    RDepthGreaterOut out;
    out.id = in.data.x;
    out.depth = in.position.z + float(p.x) / 64.0f;
    return out;
}

struct RDepthLessOut {
    uint id [[color(0)]];
    float depth [[depth(less)]];
};

fragment RDepthLessOut rnd_fragment_depth_less(RVaryingsNoPoint in [[stage_in]])
{
    uint2 p = uint2(in.position.xy);
    RDepthLessOut out;
    out.id = in.data.x;
    out.depth = in.position.z - float(p.x) / 64.0f;
    return out;
}

// Blending: the source colour is the vertex data reinterpreted as float4.
fragment float4 rnd_fragment_float_point(RVaryings in [[stage_in]])
{
    return as_type<float4>(in.data);
}

struct RDualSource {
    float4 color [[color(0), index(0)]];
    float4 second [[color(0), index(1)]];
};

fragment RDualSource rnd_fragment_dual_source(RVaryings in [[stage_in]], constant float4 &second [[buffer(0)]])
{
    RDualSource out;
    out.color = as_type<float4>(in.data);
    out.second = second;
    return out;
}

// Alpha from the vertex data for alpha-to-coverage.
fragment float4 rnd_fragment_alpha(RVaryingsNoPoint in [[stage_in]])
{
    return float4(1.0f, 1.0f, 1.0f, as_type<float>(in.data.x));
}

// Per-sample shading: the colour records the sample index.
fragment uint rnd_fragment_sample_id(RVaryingsNoPoint in [[stage_in]], uint sample [[sample_id]])
{
    return in.data.x * 16u + sample;
}

// Coverage mask written by the fragment function.
struct RMaskOut {
    uint color [[color(0)]];
    uint mask [[sample_mask]];
};

fragment RMaskOut rnd_fragment_mask_out(RVaryingsNoPoint in [[stage_in]])
{
    RMaskOut out;
    out.color = in.data.x;
    out.mask = in.data.y;
    return out;
}

// Coverage mask received by the fragment function.
fragment uint rnd_fragment_mask_in(RVaryingsNoPoint in [[stage_in]], uint mask [[sample_mask]])
{
    return 0x100u * in.data.x + mask;
}

// Per-sample depth from a table; sample 0 is discarded.
struct RSampleDepthOut {
    uint color [[color(0)]];
    float depth [[depth(any)]];
};

fragment RSampleDepthOut rnd_fragment_sample_depth(RVaryingsNoPoint in [[stage_in]], uint sample [[sample_id]])
{
    const float table[4] = {0.25f, 0.125f, 0.5f, 0.375f};
    if (sample == 0u)
        discard_fragment();
    RSampleDepthOut out;
    out.color = in.data.x;
    out.depth = table[sample & 3u];
    return out;
}

// Reads every sample of a multisample texture: out[(y * width + x) * n + s].
kernel void rnd_read_ms_uint(texture2d_ms<uint, access::read> t [[texture(0)]],
                             device uint *out [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]])
{
    uint n = t.get_num_samples();
    for (uint s = 0; s < n; s++)
        out[(gid.y * t.get_width() + gid.x) * n + s] = t.read(gid, s).x;
}

kernel void rnd_read_ms_float(texture2d_ms<float, access::read> t [[texture(0)]],
                              device float4 *out [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]])
{
    uint n = t.get_num_samples();
    for (uint s = 0; s < n; s++)
        out[(gid.y * t.get_width() + gid.x) * n + s] = t.read(gid, s);
}

kernel void rnd_read_ms_depth(depth2d_ms<float, access::read> t [[texture(0)]],
                              device float *out [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]])
{
    uint n = t.get_num_samples();
    for (uint s = 0; s < n; s++)
        out[(gid.y * t.get_width() + gid.x) * n + s] = t.read(gid, s);
}
