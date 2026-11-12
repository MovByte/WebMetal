// Shaders for the render pass cases (src/cases/RenderPass.m,
// src/cases/RenderFragment.m).

#include <metal_stdlib>
using namespace metal;

struct RPVertex {
    float4 position;
    uint4 data;
    float4 extra;
};

struct RPVaryings {
    float4 position [[position]];
    uint4 data [[flat]];
};

// Framebuffer fetch: the colour attachment's current value is an input.
fragment uint rnd_fragment_fetch(RPVaryings in [[stage_in]], uint previous [[color(0)]])
{
    return previous * 3u + in.data.x;
}

// Read-modify-write of a texture inside a raster order group.
fragment void rnd_fragment_raster_order(RPVaryings in [[stage_in]],
                                        texture2d<uint, access::read_write> t [[texture(0), raster_order_group(0)]])
{
    uint2 p = uint2(in.position.xy);
    uint v = t.read(p).x;
    t.write(uint4(v * 3u + in.data.x), p);
}

// Fragment stores into a buffer (8 pixels per row) and a later draw reads it.
fragment uint rnd_fragment_store(RPVaryings in [[stage_in]], device uint *out [[buffer(0)]])
{
    uint2 p = uint2(in.position.xy);
    out[p.y * 8u + p.x] = in.data.x * 100u + p.y * 8u + p.x;
    return 0u;
}

fragment uint rnd_fragment_load(RPVaryings in [[stage_in]], device const uint *src [[buffer(0)]])
{
    uint2 p = uint2(in.position.xy);
    return src[p.y * 8u + p.x] + 1u;
}

// Layered rendering: instance i renders to layer i, except instance 3,
// which selects layer 7 (out of bounds for an array length of 3).
struct RPLayered {
    float4 position [[position]];
    uint4 data [[flat]];
    uint layer [[render_target_array_index]];
};

vertex RPLayered rnd_vertex_layered(const device RPVertex *vertices [[buffer(0)]], uint vid [[vertex_id]],
                                    uint iid [[instance_id]])
{
    RPLayered out;
    out.position = vertices[vid].position;
    out.data = uint4(iid + 1u, 0u, 0u, 0u);
    out.layer = iid == 3u ? 7u : iid;
    return out;
}

fragment uint rnd_fragment_layered(RPLayered in [[stage_in]])
{
    return in.data.x;
}

// Post-tessellation vertex functions that record every invocation's
// position in the patch; used with rasterization disabled.
[[patch(triangle, 3)]]
vertex void rnd_tess_triangle(device atomic_uint *count [[buffer(0)]], device float4 *points [[buffer(1)]],
                              float3 p [[position_in_patch]], uint patch [[patch_id]])
{
    uint i = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    if (i < 1024u)
        points[i] = float4(p, float(patch));
}

[[patch(quad, 4)]]
vertex void rnd_tess_quad(device atomic_uint *count [[buffer(0)]], device float4 *points [[buffer(1)]],
                          float2 p [[position_in_patch]], uint patch [[patch_id]])
{
    uint i = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    if (i < 1024u)
        points[i] = float4(p, 0.0f, float(patch));
}
