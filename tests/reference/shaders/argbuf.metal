// Kernels for the argument buffer cases (src/cases/ArgumentBuffers.m).

#include <metal_stdlib>
using namespace metal;

// A buffer, a texture, a sampler, an inline constant and a writable buffer
// in one argument buffer.
struct ArgResources {
    device const float *values [[id(0)]];
    texture2d<float> image [[id(1)]];
    sampler smp [[id(2)]];
    uint scale [[id(3)]];
    device uint *out [[id(4)]];
};

// The 4x4 R32Float image is read and sampled (nearest, at texel centers);
// both return the stored value.
kernel void arg_read(constant ArgResources &args [[buffer(0)]],
                     uint i [[thread_position_in_grid]])
{
    uint2 coord = uint2(i % 4u, i / 4u);
    float2 uv = (float2(coord) + 0.5f) / 4.0f;
    args.out[i * 3u + 0u] = as_type<uint>(args.values[i] * float(args.scale));
    args.out[i * 3u + 1u] = as_type<uint>(args.image.read(coord).x);
    args.out[i * 3u + 2u] = as_type<uint>(args.image.sample(args.smp, uv, level(0.0f)).x);
}

// Arrays of textures, buffers and constants in an argument buffer.
struct ArgArrays {
    array<texture2d<float>, 4> textures [[id(0)]];
    device const uint *buffers [[id(4)]] [3];
    uint constants [[id(7)]] [2];
};

kernel void arg_arrays(constant ArgArrays &args [[buffer(0)]],
                       device uint *out [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    uint t = i % 4u;
    uint b = i % 3u;
    out[i * 3u + 0u] = as_type<uint>(args.textures[t].read(uint2(i % 2u, 0u)).x);
    out[i * 3u + 1u] = args.buffers[b][i];
    out[i * 3u + 2u] = args.constants[i % 2u];
}

// An argument buffer that points to another argument buffer.
struct ArgInner {
    device const uint *data [[id(0)]];
};

struct ArgOuter {
    constant ArgInner *inner [[id(0)]];
    uint bias [[id(1)]];
};

kernel void arg_nested(constant ArgOuter &outer [[buffer(0)]],
                       device uint *out [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    out[i] = outer.inner->data[i] + outer.bias;
}
