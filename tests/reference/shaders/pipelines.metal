// Functions for the library, function and pipeline cases
// (src/cases/Libraries.m, src/cases/Pipelines.m, src/cases/Validation.m).

#include <metal_stdlib>
using namespace metal;

// Function constants. Constants 0, 1, 2, 4 and 5 are used without an
// is_function_constant_defined guard, so specialization must supply them;
// constant 3 is optional; constant 6 is set by name.
constant uint pip_fc_a [[function_constant(0)]];
constant bool pip_fc_flag [[function_constant(1)]];
constant float pip_fc_scale [[function_constant(2)]];
constant uint pip_fc_optional [[function_constant(3)]];
constant bool pip_fc_optional_defined = is_function_constant_defined(pip_fc_optional);
constant half pip_fc_half [[function_constant(4)]];
constant int2 pip_fc_vector [[function_constant(5)]];
constant ushort pip_fc_named [[function_constant(6)]];

kernel void pip_constants(device uint *out [[buffer(0)]],
                          device uint *extra [[buffer(1), function_constant(pip_fc_flag)]])
{
    out[0] = pip_fc_a;
    out[1] = pip_fc_flag ? 1u : 0u;
    out[2] = as_type<uint>(pip_fc_scale);
    out[3] = pip_fc_optional_defined ? pip_fc_optional : 0xffffu;
    out[4] = as_type<ushort>(pip_fc_half);
    out[5] = uint(pip_fc_vector.x);
    out[6] = uint(pip_fc_vector.y);
    out[7] = pip_fc_named;
    if (pip_fc_flag) {
        extra[0] = 0xf1a9u;
    }
}

// Argument kinds for compute reflection: a buffer of structures with nested
// and packed members, a constant reference, an atomic buffer, an unused
// buffer, read and write textures, a sampled array texture, a depth texture,
// a sampler and threadgroup memory.
struct PipInner {
    float2 a;
    uint b;
};

struct PipRecord {
    float4 color;
    packed_float3 position;
    uint flags;
    PipInner inner[2];
    half h;
    char c;
};

kernel void pip_reflect_compute(device PipRecord *records [[buffer(0)]],
                                constant float4x4 &matrix [[buffer(1)]],
                                device atomic_uint *counter [[buffer(2)]],
                                device uint *unused [[buffer(3)]],
                                texture2d<float, access::read> input [[texture(0)]],
                                texture2d<uint, access::write> output [[texture(1)]],
                                texture2d_array<half, access::sample> sampled [[texture(2)]],
                                depth2d<float> depth [[texture(3)]],
                                sampler s [[sampler(0)]],
                                threadgroup float *scratch [[threadgroup(0)]],
                                uint2 gid [[thread_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]])
{
    float4 value = input.read(gid) + float4(sampled.sample(s, float2(0.5f), 0)) + depth.sample(s, float2(0.5f));
    scratch[tid] = value.x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    records[gid.x].color = matrix * value + scratch[0];
    output.write(uint4(records[gid.x].flags), gid);
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
}

// Render reflection: vertex stage input with attribute formats, vertex and
// instance ids, buffers and textures in both stages.
struct PipVertexIn {
    float4 position [[attribute(0)]];
    half2 uv [[attribute(1)]];
    uchar4 color [[attribute(3)]];
    float unused [[attribute(4)]];
};

struct PipVarying {
    float4 position [[position]];
    float2 uv;
    half4 color [[flat]];
    float point [[point_size]];
};

vertex PipVarying pip_reflect_vertex(PipVertexIn in [[stage_in]],
                                     constant float4x4 &mvp [[buffer(1)]],
                                     const device float2 *offsets [[buffer(2)]],
                                     uint vid [[vertex_id]],
                                     uint iid [[instance_id]])
{
    PipVarying out;
    out.position = mvp * in.position + float4(offsets[iid], 0.0f, float(vid & 1u));
    out.uv = float2(in.uv);
    out.color = half4(in.color);
    out.point = 1.0f;
    return out;
}

struct PipFragmentOut {
    float4 color [[color(0)]];
    uint4 id [[color(2)]];
};

fragment PipFragmentOut pip_reflect_fragment(PipVarying in [[stage_in]],
                                             texture2d<float> t [[texture(0)]],
                                             sampler s [[sampler(0)]],
                                             constant float4 &tint [[buffer(0)]])
{
    PipFragmentOut out;
    out.color = t.sample(s, in.uv) * tint + float4(in.color);
    out.id = uint4(1u, 2u, 3u, 4u);
    return out;
}

// Vertex and fragment functions used to build simple pipelines.
struct PipSimpleVarying {
    float4 position [[position]];
    float4 color;
};

vertex PipSimpleVarying pip_simple_vertex(const device float4 *positions [[buffer(0)]],
                                          uint vid [[vertex_id]])
{
    PipSimpleVarying out;
    out.position = positions[vid];
    out.color = float4(1.0f, 0.0f, 0.0f, 1.0f);
    return out;
}

fragment float4 pip_simple_fragment(PipSimpleVarying in [[stage_in]])
{
    return in.color;
}

fragment uint4 pip_uint_fragment(PipSimpleVarying in [[stage_in]])
{
    return uint4(in.color * 255.0f);
}

// A fragment input that pip_simple_vertex does not produce.
struct PipOtherVarying {
    float4 position [[position]];
    float3 normal [[user(normal)]];
};

fragment float4 pip_unmatched_fragment(PipOtherVarying in [[stage_in]])
{
    return float4(in.normal, 1.0f);
}

// A vertex function with stage input, which needs a vertex descriptor.
struct PipStageVertexIn {
    float4 position [[attribute(0)]];
};

vertex float4 pip_stage_vertex(PipStageVertexIn in [[stage_in]])
{
    return in.position;
}

// A vertex function without rasterization output, for pipelines with
// rasterization disabled.
vertex void pip_void_vertex(device uint *out [[buffer(0)]], uint vid [[vertex_id]])
{
    out[vid] = vid * 2u + 1u;
}

// Post-tessellation vertex function for patch reflection.
struct PipControlPoint {
    float4 position [[attribute(0)]];
};

struct PipPatchIn {
    patch_control_point<PipControlPoint> points;
    float4 color [[attribute(1)]];
};

[[patch(triangle, 3)]]
vertex float4 pip_post_tess(PipPatchIn patch [[stage_in]],
                            float3 coord [[position_in_patch]])
{
    return patch.points[0].position * coord.x + patch.points[1].position * coord.y +
           patch.points[2].position * coord.z + patch.color * 0.0f;
}

// Compute stage input for stageInputAttributes reflection.
struct PipStageKernelIn {
    ushort2 pair [[attribute(0)]];
    float value [[attribute(2)]];
};

kernel void pip_stage_kernel(PipStageKernelIn in [[stage_in]],
                             device float *out [[buffer(0)]],
                             uint gid [[thread_position_in_grid]])
{
    out[gid] = float(in.pair.x + in.pair.y) + in.value;
}

// A kernel that records a marker, for pipeline and binary archive cases.
kernel void pip_marker(device uint *out [[buffer(0)]],
                       uint gid [[thread_position_in_grid]])
{
    out[gid] = 0x600du + gid;
}

// A kernel that writes a texture, for usage validation cases.
kernel void pip_write_texture(texture2d<float, access::write> t [[texture(0)]],
                              uint2 gid [[thread_position_in_grid]])
{
    t.write(float4(1.0f), gid);
}

// A kernel with a constant float4 argument, for offset validation cases.
kernel void pip_constant_vector(constant float4 &v [[buffer(0)]],
                                device float4 *out [[buffer(1)]])
{
    out[0] = v;
}
