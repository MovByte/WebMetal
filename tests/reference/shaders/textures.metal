// Kernels and render functions for the texture cases (src/cases/Textures.m).
//
// Read kernels take a list of queries, two uint4 per query:
//   q[2i]     = (x, y, z, w)   coordinates; the meaning of z and w depends
//                              on the texture type (slice, face, depth, lod,
//                              sample), as documented on each kernel
//   q[2i + 1] = (lod, 0, 0, 0)
// and write the texel read for query i to out[i]. Queries outside the
// texture are part of the list: MSL 4.1 section 6.13 defines their results.

#include <metal_stdlib>
using namespace metal;

// (x)
kernel void tex_read_1d(texture1d<uint, access::read> t [[texture(0)]],
                        device const uint4 *q [[buffer(0)]],
                        device uint4 *out [[buffer(1)]],
                        uint i [[thread_position_in_grid]])
{
    out[i] = t.read(q[2 * i].x);
}

// (x, slice)
kernel void tex_read_1d_array(texture1d_array<uint, access::read> t [[texture(0)]],
                              device const uint4 *q [[buffer(0)]],
                              device uint4 *out [[buffer(1)]],
                              uint i [[thread_position_in_grid]])
{
    out[i] = t.read(q[2 * i].x, q[2 * i].y);
}

// (x, y), lod
kernel void tex_read_2d(texture2d<uint, access::read> t [[texture(0)]],
                        device const uint4 *q [[buffer(0)]],
                        device uint4 *out [[buffer(1)]],
                        uint i [[thread_position_in_grid]])
{
    out[i] = t.read(q[2 * i].xy, q[2 * i + 1].x);
}

// (x, y, slice), lod
kernel void tex_read_2d_array(texture2d_array<uint, access::read> t [[texture(0)]],
                              device const uint4 *q [[buffer(0)]],
                              device uint4 *out [[buffer(1)]],
                              uint i [[thread_position_in_grid]])
{
    out[i] = t.read(q[2 * i].xy, q[2 * i].z, q[2 * i + 1].x);
}

// (x, y, z), lod
kernel void tex_read_3d(texture3d<uint, access::read> t [[texture(0)]],
                        device const uint4 *q [[buffer(0)]],
                        device uint4 *out [[buffer(1)]],
                        uint i [[thread_position_in_grid]])
{
    out[i] = t.read(q[2 * i].xyz, q[2 * i + 1].x);
}

// (x, y, face), lod
kernel void tex_read_cube(texturecube<uint, access::read> t [[texture(0)]],
                          device const uint4 *q [[buffer(0)]],
                          device uint4 *out [[buffer(1)]],
                          uint i [[thread_position_in_grid]])
{
    out[i] = t.read(q[2 * i].xy, q[2 * i].z, q[2 * i + 1].x);
}

// (x, y, face, slice), lod
kernel void tex_read_cube_array(texturecube_array<uint, access::read> t [[texture(0)]],
                                device const uint4 *q [[buffer(0)]],
                                device uint4 *out [[buffer(1)]],
                                uint i [[thread_position_in_grid]])
{
    out[i] = t.read(q[2 * i].xy, q[2 * i].z, q[2 * i].w, q[2 * i + 1].x);
}

// (x, y, sample)
kernel void tex_read_2d_ms(texture2d_ms<uint, access::read> t [[texture(0)]],
                           device const uint4 *q [[buffer(0)]],
                           device uint4 *out [[buffer(1)]],
                           uint i [[thread_position_in_grid]])
{
    out[i] = t.read(q[2 * i].xy, q[2 * i].z);
}

// (x, y, slice, sample)
kernel void tex_read_2d_ms_array(texture2d_ms_array<uint, access::read> t [[texture(0)]],
                                 device const uint4 *q [[buffer(0)]],
                                 device uint4 *out [[buffer(1)]],
                                 uint i [[thread_position_in_grid]])
{
    out[i] = t.read(q[2 * i].xy, q[2 * i].z, q[2 * i].w);
}

// (x)
kernel void tex_read_buffer(texture_buffer<uint, access::read> t [[texture(0)]],
                            device const uint4 *q [[buffer(0)]],
                            device uint4 *out [[buffer(1)]],
                            uint i [[thread_position_in_grid]])
{
    out[i] = t.read(q[2 * i].x);
}

// Size queries. Each kernel runs on one thread and writes the values in
// the order listed.

// width, get_num_mip_levels
kernel void tex_query_1d(texture1d<uint, access::read> t [[texture(0)]],
                         device uint *out [[buffer(1)]])
{
    out[0] = t.get_width();
    out[1] = t.get_num_mip_levels();
}

// width, array size
kernel void tex_query_1d_array(texture1d_array<uint, access::read> t [[texture(0)]],
                               device uint *out [[buffer(1)]])
{
    out[0] = t.get_width();
    out[1] = t.get_array_size();
}

// get_num_mip_levels, then width and height of each level
kernel void tex_query_2d(texture2d<uint, access::read> t [[texture(0)]],
                         device uint *out [[buffer(1)]])
{
    uint levels = t.get_num_mip_levels();
    out[0] = levels;
    for (uint l = 0; l < levels; l++) {
        out[1 + 2 * l] = t.get_width(l);
        out[2 + 2 * l] = t.get_height(l);
    }
}

// get_num_mip_levels, array size, then width and height of each level
kernel void tex_query_2d_array(texture2d_array<uint, access::read> t [[texture(0)]],
                               device uint *out [[buffer(1)]])
{
    uint levels = t.get_num_mip_levels();
    out[0] = levels;
    out[1] = t.get_array_size();
    for (uint l = 0; l < levels; l++) {
        out[2 + 2 * l] = t.get_width(l);
        out[3 + 2 * l] = t.get_height(l);
    }
}

// get_num_mip_levels, then width, height and depth of each level
kernel void tex_query_3d(texture3d<uint, access::read> t [[texture(0)]],
                         device uint *out [[buffer(1)]])
{
    uint levels = t.get_num_mip_levels();
    out[0] = levels;
    for (uint l = 0; l < levels; l++) {
        out[1 + 3 * l] = t.get_width(l);
        out[2 + 3 * l] = t.get_height(l);
        out[3 + 3 * l] = t.get_depth(l);
    }
}

// get_num_mip_levels, then width and height of each level
kernel void tex_query_cube(texturecube<uint, access::read> t [[texture(0)]],
                           device uint *out [[buffer(1)]])
{
    uint levels = t.get_num_mip_levels();
    out[0] = levels;
    for (uint l = 0; l < levels; l++) {
        out[1 + 2 * l] = t.get_width(l);
        out[2 + 2 * l] = t.get_height(l);
    }
}

// get_num_mip_levels, array size, then width and height of each level
kernel void tex_query_cube_array(texturecube_array<uint, access::read> t [[texture(0)]],
                                 device uint *out [[buffer(1)]])
{
    uint levels = t.get_num_mip_levels();
    out[0] = levels;
    out[1] = t.get_array_size();
    for (uint l = 0; l < levels; l++) {
        out[2 + 2 * l] = t.get_width(l);
        out[3 + 2 * l] = t.get_height(l);
    }
}

// width, height, sample count
kernel void tex_query_2d_ms(texture2d_ms<uint, access::read> t [[texture(0)]],
                            device uint *out [[buffer(1)]])
{
    out[0] = t.get_width();
    out[1] = t.get_height();
    out[2] = t.get_num_samples();
}

// width, height, sample count, array size
kernel void tex_query_2d_ms_array(texture2d_ms_array<uint, access::read> t [[texture(0)]],
                                  device uint *out [[buffer(1)]])
{
    out[0] = t.get_width();
    out[1] = t.get_height();
    out[2] = t.get_num_samples();
    out[3] = t.get_array_size();
}

// width
kernel void tex_query_buffer(texture_buffer<uint, access::read> t [[texture(0)]],
                             device uint *out [[buffer(1)]])
{
    out[0] = t.get_width();
}

// Write kernels: query i writes values[i] (the queries use the same layout
// as the read kernels). Writes outside the texture are ignored (MSL 4.1
// section 6.13).

kernel void tex_write_1d(texture1d<uint, access::write> t [[texture(0)]],
                         device const uint4 *q [[buffer(0)]],
                         device const uint4 *values [[buffer(1)]],
                         uint i [[thread_position_in_grid]])
{
    t.write(values[i], q[2 * i].x);
}

kernel void tex_write_2d(texture2d<uint, access::write> t [[texture(0)]],
                         device const uint4 *q [[buffer(0)]],
                         device const uint4 *values [[buffer(1)]],
                         uint i [[thread_position_in_grid]])
{
    t.write(values[i], q[2 * i].xy, q[2 * i + 1].x);
}

kernel void tex_write_2d_array(texture2d_array<uint, access::write> t [[texture(0)]],
                               device const uint4 *q [[buffer(0)]],
                               device const uint4 *values [[buffer(1)]],
                               uint i [[thread_position_in_grid]])
{
    t.write(values[i], q[2 * i].xy, q[2 * i].z, q[2 * i + 1].x);
}

kernel void tex_write_3d(texture3d<uint, access::write> t [[texture(0)]],
                         device const uint4 *q [[buffer(0)]],
                         device const uint4 *values [[buffer(1)]],
                         uint i [[thread_position_in_grid]])
{
    t.write(values[i], q[2 * i].xyz, q[2 * i + 1].x);
}

kernel void tex_write_cube(texturecube<uint, access::write> t [[texture(0)]],
                           device const uint4 *q [[buffer(0)]],
                           device const uint4 *values [[buffer(1)]],
                           uint i [[thread_position_in_grid]])
{
    t.write(values[i], q[2 * i].xy, q[2 * i].z, q[2 * i + 1].x);
}

kernel void tex_write_buffer(texture_buffer<uint, access::write> t [[texture(0)]],
                             device const uint4 *q [[buffer(0)]],
                             device const uint4 *values [[buffer(1)]],
                             uint i [[thread_position_in_grid]])
{
    t.write(values[i], q[2 * i].x);
}

// Float reads of a 2D texture, used for format-view and swizzle cases.
kernel void tex_read_2d_float(texture2d<float, access::read> t [[texture(0)]],
                              device const uint4 *q [[buffer(0)]],
                              device float4 *out [[buffer(1)]],
                              uint i [[thread_position_in_grid]])
{
    out[i] = t.read(q[2 * i].xy, q[2 * i + 1].x);
}

// Nearest sampling of a 2D texture through a uint view at texel centers.
kernel void tex_sample_2d_uint(texture2d<uint, access::sample> t [[texture(0)]],
                               sampler s [[sampler(0)]],
                               device const float2 *coords [[buffer(0)]],
                               device uint4 *out [[buffer(1)]],
                               uint i [[thread_position_in_grid]])
{
    out[i] = t.sample(s, coords[i]);
}

// Sampling a cube texture along a direction (level 0; compute functions
// have a default LOD of 0, MSL 4.1 section 6.13).
kernel void tex_sample_cube(texturecube<float, access::sample> t [[texture(0)]],
                            sampler s [[sampler(0)]],
                            device const float4 *directions [[buffer(0)]],
                            device float4 *out [[buffer(1)]],
                            uint i [[thread_position_in_grid]])
{
    out[i] = t.sample(s, directions[i].xyz);
}

// Rectangle covering pixel columns [0, 2) of a 4x4 target in two triangles.
// Every sample position of a pixel lies inside the pixel, so the coverage of
// each sample does not depend on the sample positions.
constant float2 tex_rect_corners[6] = {float2(-1, -1), float2(0, -1), float2(-1, 1),
                                       float2(0, -1), float2(0, 1), float2(-1, 1)};

struct TexRectOut {
    float4 position [[position]];
};

vertex TexRectOut tex_rect_vertex(uint vid [[vertex_id]])
{
    TexRectOut out;
    out.position = float4(tex_rect_corners[vid], 0, 1);
    return out;
}

// Per-sample shading (a sample_id input runs the function once per sample):
// value = 0xab000000 | sample << 16 | y << 8 | x.
fragment uint4 tex_ms_fragment(TexRectOut in [[stage_in]], uint sample [[sample_id]])
{
    uint x = uint(floor(in.position.x));
    uint y = uint(floor(in.position.y));
    return uint4(0xab000000u | (sample << 16) | (y << 8) | x, 0, 0, 0);
}

// Layered variant: one instance per layer.
struct TexLayeredOut {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
};

vertex TexLayeredOut tex_layered_vertex(uint vid [[vertex_id]], uint iid [[instance_id]])
{
    TexLayeredOut out;
    out.position = float4(tex_rect_corners[vid], 0, 1);
    out.layer = iid;
    return out;
}

// value = 0xab000000 | layer << 20 | sample << 16 | y << 8 | x.
fragment uint4 tex_ms_layered_fragment(TexLayeredOut in [[stage_in]], uint sample [[sample_id]])
{
    uint x = uint(floor(in.position.x));
    uint y = uint(floor(in.position.y));
    return uint4(0xab000000u | (in.layer << 20) | (sample << 16) | (y << 8) | x, 0, 0, 0);
}

// A triangle over the whole target: over a target of one pixel, one
// fragment per draw.
constant float2 tex_cover_corners[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};

vertex TexRectOut tex_cover_vertex(uint vid [[vertex_id]])
{
    TexRectOut out;
    out.position = float4(tex_cover_corners[vid], 0, 1);
    return out;
}

// Fragment writes: the query (two uint4, as for the kernels) and the value
// come from buffers 0 and 1; the fragment's own output is 1.
fragment uint tex_fragment_write_2d(TexRectOut in [[stage_in]],
                                    texture2d<uint, access::write> t [[texture(0)]],
                                    constant uint4 *q [[buffer(0)]],
                                    constant uint4 *value [[buffer(1)]])
{
    t.write(value[0], q[0].xy, q[1].x);
    return 1;
}

fragment uint tex_fragment_write_2d_float(TexRectOut in [[stage_in]],
                                          texture2d<float, access::write> t [[texture(0)]],
                                          constant uint4 *q [[buffer(0)]],
                                          constant float4 *value [[buffer(1)]])
{
    t.write(value[0], q[0].xy, q[1].x);
    return 1;
}

// A fragment read: the texel at the query is the fragment's output.
fragment uint4 tex_fragment_read_2d(TexRectOut in [[stage_in]],
                                    texture2d<uint, access::read> t [[texture(0)]],
                                    constant uint4 *q [[buffer(0)]])
{
    return t.read(q[0].xy, q[1].x);
}
