// Functions for the indirect command buffer cases
// (src/cases/IndirectCommands.m).

#include <metal_stdlib>
using namespace metal;

kernel void icb_fill(device uint *out [[buffer(0)]],
                     constant uint &value [[buffer(1)]],
                     uint i [[thread_position_in_grid]])
{
    out[i] = value + i;
}

vertex float4 icb_vertex(const device float2 *positions [[buffer(0)]],
                         uint vid [[vertex_id]])
{
    return float4(positions[vid], 0.0f, 1.0f);
}

fragment float4 icb_fragment(constant float4 &color [[buffer(0)]])
{
    return color;
}
