// Kernels and render functions for the pixel-format cases
// (src/cases/Formats.m, src/cases/FormatsCompressed.m).
//
// Texel results are stored row-major: index = y * width + x.

#include <metal_stdlib>
using namespace metal;

kernel void fmt_read_float(texture2d<float, access::read> t [[texture(0)]],
                           device float4 *out [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < t.get_width() && gid.y < t.get_height())
        out[gid.y * t.get_width() + gid.x] = t.read(gid);
}

kernel void fmt_read_uint(texture2d<uint, access::read> t [[texture(0)]],
                          device uint4 *out [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < t.get_width() && gid.y < t.get_height())
        out[gid.y * t.get_width() + gid.x] = t.read(gid);
}

kernel void fmt_read_int(texture2d<int, access::read> t [[texture(0)]],
                         device int4 *out [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < t.get_width() && gid.y < t.get_height())
        out[gid.y * t.get_width() + gid.x] = t.read(gid);
}

kernel void fmt_read_depth(depth2d<float, access::read> t [[texture(0)]],
                           device float *out [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < t.get_width() && gid.y < t.get_height())
        out[gid.y * t.get_width() + gid.x] = t.read(gid);
}

kernel void fmt_read_buffer_float(texture_buffer<float, access::read> t [[texture(0)]],
                                  device float4 *out [[buffer(0)]],
                                  uint gid [[thread_position_in_grid]])
{
    if (gid < t.get_width())
        out[gid] = t.read(gid);
}

kernel void fmt_read_buffer_uint(texture_buffer<uint, access::read> t [[texture(0)]],
                                 device uint4 *out [[buffer(0)]],
                                 uint gid [[thread_position_in_grid]])
{
    if (gid < t.get_width())
        out[gid] = t.read(gid);
}

kernel void fmt_read_buffer_int(texture_buffer<int, access::read> t [[texture(0)]],
                                device int4 *out [[buffer(0)]],
                                uint gid [[thread_position_in_grid]])
{
    if (gid < t.get_width())
        out[gid] = t.read(gid);
}

kernel void fmt_write_float(texture2d<float, access::write> t [[texture(0)]],
                            device const float4 *values [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < t.get_width() && gid.y < t.get_height())
        t.write(values[gid.y * t.get_width() + gid.x], gid);
}

kernel void fmt_write_uint(texture2d<uint, access::write> t [[texture(0)]],
                           device const uint4 *values [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < t.get_width() && gid.y < t.get_height())
        t.write(values[gid.y * t.get_width() + gid.x], gid);
}

kernel void fmt_write_int(texture2d<int, access::write> t [[texture(0)]],
                          device const int4 *values [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < t.get_width() && gid.y < t.get_height())
        t.write(values[gid.y * t.get_width() + gid.x], gid);
}

// A triangle that covers the whole viewport.
vertex float4 fmt_fullscreen_vertex(uint vid [[vertex_id]])
{
    float2 p = float2(float((vid << 1) & 2u), float(vid & 2u));
    return float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
}

// Each fragment outputs the value stored for its pixel. [[position]] holds
// the pixel center, so its integer part is the pixel coordinate.
fragment float4 fmt_fragment_float(float4 position [[position]],
                                   device const float4 *values [[buffer(0)]],
                                   constant uint &width [[buffer(1)]])
{
    return values[uint(position.y) * width + uint(position.x)];
}

fragment uint4 fmt_fragment_uint(float4 position [[position]],
                                 device const uint4 *values [[buffer(0)]],
                                 constant uint &width [[buffer(1)]])
{
    return values[uint(position.y) * width + uint(position.x)];
}

fragment int4 fmt_fragment_int(float4 position [[position]],
                               device const int4 *values [[buffer(0)]],
                               constant uint &width [[buffer(1)]])
{
    return values[uint(position.y) * width + uint(position.x)];
}
