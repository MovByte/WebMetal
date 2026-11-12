// Kernels for the storage and heap cases (src/cases/Storage.m,
// src/cases/Heaps.m).

#include <metal_stdlib>
using namespace metal;

kernel void sto_add_one(device uint *data [[buffer(0)]],
                        uint i [[thread_position_in_grid]])
{
    data[i] = data[i] + 1u;
}

// Reads every texel of a 2D unsigned-integer texture into a buffer of
// uint4, row-major.
kernel void sto_read_texture(texture2d<uint, access::read> texture [[texture(0)]],
                             device uint4 *out [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texture.get_width() || gid.y >= texture.get_height()) return;
    out[gid.y * texture.get_width() + gid.x] = texture.read(gid);
}

// Writes every texel of a 2D unsigned-integer texture from a buffer of
// uint4, row-major.
kernel void sto_write_texture(texture2d<uint, access::write> texture [[texture(0)]],
                              device const uint4 *in [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texture.get_width() || gid.y >= texture.get_height()) return;
    texture.write(in[gid.y * texture.get_width() + gid.x], gid);
}

// A triangle covering the whole viewport at the depth given in buffer 0.
struct StoragePosition {
    float4 position [[position]];
};

vertex StoragePosition sto_fullscreen_vertex(uint vid [[vertex_id]],
                                             constant float &depth [[buffer(0)]])
{
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    StoragePosition out;
    out.position = float4(corners[vid], depth, 1.0f);
    return out;
}

fragment uint4 sto_color_fragment(constant uint4 &color [[buffer(0)]])
{
    return color;
}
