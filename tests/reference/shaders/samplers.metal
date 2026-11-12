// Kernels and render functions for the sampler cases (src/cases/Samplers.m).
// Compute functions that sample without an explicit LOD use a default LOD
// of 0 (MSL 4.1 section 6.13).

#include <metal_stdlib>
using namespace metal;

kernel void smp_sample_2d(texture2d<float, access::sample> t [[texture(0)]],
                          sampler s [[sampler(0)]],
                          device const float2 *coords [[buffer(0)]],
                          device float4 *out [[buffer(1)]],
                          uint i [[thread_position_in_grid]])
{
    out[i] = t.sample(s, coords[i]);
}

kernel void smp_sample_3d(texture3d<float, access::sample> t [[texture(0)]],
                          sampler s [[sampler(0)]],
                          device const float4 *coords [[buffer(0)]],
                          device float4 *out [[buffer(1)]],
                          uint i [[thread_position_in_grid]])
{
    out[i] = t.sample(s, coords[i].xyz);
}

// coords[i] = (u, v, level, unused)
kernel void smp_sample_2d_level(texture2d<float, access::sample> t [[texture(0)]],
                                sampler s [[sampler(0)]],
                                device const float4 *coords [[buffer(0)]],
                                device float4 *out [[buffer(1)]],
                                uint i [[thread_position_in_grid]])
{
    out[i] = t.sample(s, coords[i].xy, level(coords[i].z));
}

// coords[i] = (u, v, bias, unused)
kernel void smp_sample_2d_bias(texture2d<float, access::sample> t [[texture(0)]],
                               sampler s [[sampler(0)]],
                               device const float4 *coords [[buffer(0)]],
                               device float4 *out [[buffer(1)]],
                               uint i [[thread_position_in_grid]])
{
    out[i] = t.sample(s, coords[i].xy, bias(coords[i].z));
}

// coords[2i] = (u, v, 0, 0), coords[2i + 1] = (dPdx, dPdy)
kernel void smp_sample_2d_gradient(texture2d<float, access::sample> t [[texture(0)]],
                                   sampler s [[sampler(0)]],
                                   device const float4 *coords [[buffer(0)]],
                                   device float4 *out [[buffer(1)]],
                                   uint i [[thread_position_in_grid]])
{
    float4 g = coords[2 * i + 1];
    out[i] = t.sample(s, coords[2 * i].xy, gradient2d(g.xy, g.zw));
}

// Four gathers, one per component, and one gather of component x with a
// texel offset of (1, 0): out[5i .. 5i + 4].
kernel void smp_gather(texture2d<float, access::sample> t [[texture(0)]],
                       sampler s [[sampler(0)]],
                       device const float2 *coords [[buffer(0)]],
                       device float4 *out [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    float2 c = coords[i];
    out[5 * i + 0] = t.gather(s, c, int2(0), component::x);
    out[5 * i + 1] = t.gather(s, c, int2(0), component::y);
    out[5 * i + 2] = t.gather(s, c, int2(0), component::z);
    out[5 * i + 3] = t.gather(s, c, int2(0), component::w);
    out[5 * i + 4] = t.gather(s, c, int2(1, 0), component::x);
}

// q[i] = (u, v, reference, unused)
kernel void smp_compare(depth2d<float, access::sample> t [[texture(0)]],
                        sampler s [[sampler(0)]],
                        device const float4 *q [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        uint i [[thread_position_in_grid]])
{
    out[i] = t.sample_compare(s, q[i].xy, q[i].z);
}

// q[i] = (u, v, reference, unused)
kernel void smp_gather_compare(depth2d<float, access::sample> t [[texture(0)]],
                               sampler s [[sampler(0)]],
                               device const float4 *q [[buffer(0)]],
                               device float4 *out [[buffer(1)]],
                               uint i [[thread_position_in_grid]])
{
    out[i] = t.gather_compare(s, q[i].xy, q[i].z);
}

// Samplers declared in the program source (MSL 4.1 section 2.10). Each
// coordinate is sampled with five samplers: out[5i .. 5i + 4].
kernel void smp_constexpr(texture2d<float, access::sample> t [[texture(0)]],
                          device const float2 *coords [[buffer(0)]],
                          device float4 *out [[buffer(1)]],
                          uint i [[thread_position_in_grid]])
{
    constexpr sampler defaults;
    constexpr sampler repeat(address::repeat);
    constexpr sampler mirrored(address::mirrored_repeat, filter::nearest);
    constexpr sampler zero(address::clamp_to_zero);
    constexpr sampler mixed(s_address::repeat, t_address::mirrored_repeat);
    float2 c = coords[i];
    out[5 * i + 0] = t.sample(defaults, c);
    out[5 * i + 1] = t.sample(repeat, c);
    out[5 * i + 2] = t.sample(mirrored, c);
    out[5 * i + 3] = t.sample(zero, c);
    out[5 * i + 4] = t.sample(mixed, c);
}

// coord::pixel samplers: coordinates in texels. MSL 4.1 section 2.10 allows
// clamp_to_zero, clamp_to_border and clamp_to_edge with them.
kernel void smp_constexpr_pixel(texture2d<float, access::sample> t [[texture(0)]],
                                device const float2 *coords [[buffer(0)]],
                                device float4 *out [[buffer(1)]],
                                uint i [[thread_position_in_grid]])
{
    constexpr sampler edge(coord::pixel, address::clamp_to_edge);
    constexpr sampler zero(coord::pixel, address::clamp_to_zero);
    out[2 * i + 0] = t.sample(edge, coords[i]);
    out[2 * i + 1] = t.sample(zero, coords[i]);
}

// q[i] = (u, v, reference, unused)
kernel void smp_constexpr_compare(depth2d<float, access::sample> t [[texture(0)]],
                                  device const float4 *q [[buffer(0)]],
                                  device float *out [[buffer(1)]],
                                  uint i [[thread_position_in_grid]])
{
    constexpr sampler less(compare_func::less);
    constexpr sampler greater_equal(compare_func::greater_equal);
    out[2 * i + 0] = t.sample_compare(less, q[i].xy, q[i].z);
    out[2 * i + 1] = t.sample_compare(greater_equal, q[i].xy, q[i].z);
}

// clamp_to_border with each border color, declared in the program source.
kernel void smp_constexpr_border(texture2d<float, access::sample> t [[texture(0)]],
                                 device const float2 *coords [[buffer(0)]],
                                 device float4 *out [[buffer(1)]],
                                 uint i [[thread_position_in_grid]])
{
    constexpr sampler transparent(address::clamp_to_border, border_color::transparent_black);
    constexpr sampler black(address::clamp_to_border, border_color::opaque_black);
    constexpr sampler white(address::clamp_to_border, border_color::opaque_white);
    out[3 * i + 0] = t.sample(transparent, coords[i]);
    out[3 * i + 1] = t.sample(black, coords[i]);
    out[3 * i + 2] = t.sample(white, coords[i]);
}

// Full-screen triangle over an 8x8 target; the fragment samples with a
// texture coordinate of position / 8, one texel per pixel at level 0, with
// an implicit LOD computed from derivatives.
struct SmpOut {
    float4 position [[position]];
};

vertex SmpOut smp_fullscreen_vertex(uint vid [[vertex_id]])
{
    const float2 corners[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    SmpOut out;
    out.position = float4(corners[vid], 0, 1);
    return out;
}

fragment float4 smp_implicit_fragment(SmpOut in [[stage_in]],
                                      texture2d<float, access::sample> t [[texture(0)]],
                                      sampler s [[sampler(0)]])
{
    return t.sample(s, in.position.xy / 8.0f);
}

// Same, with a LOD bias of 3.
fragment float4 smp_bias_fragment(SmpOut in [[stage_in]],
                                  texture2d<float, access::sample> t [[texture(0)]],
                                  sampler s [[sampler(0)]])
{
    return t.sample(s, in.position.xy / 8.0f, bias(3.0f));
}

// Same, with min_lod_clamp(minimum) (MSL 4.1 section 6.13).
fragment float4 smp_min_lod_clamp_fragment(SmpOut in [[stage_in]],
                                           texture2d<float, access::sample> t [[texture(0)]],
                                           sampler s [[sampler(0)]],
                                           constant float &minimum [[buffer(0)]])
{
    return t.sample(s, in.position.xy / 8.0f, min_lod_clamp(minimum));
}
