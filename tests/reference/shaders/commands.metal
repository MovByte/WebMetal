// Kernels for the command, event and fence cases (src/cases/Commands.m,
// src/cases/Events.m).

#include <metal_stdlib>
using namespace metal;

kernel void cmd_add_value(device uint *data [[buffer(0)]],
                          constant uint &value [[buffer(1)]],
                          uint i [[thread_position_in_grid]])
{
    data[i] = data[i] + value;
}

kernel void cmd_fill(device uint *data [[buffer(0)]],
                     constant uint &value [[buffer(1)]],
                     uint i [[thread_position_in_grid]])
{
    data[i] = value;
}

kernel void cmd_copy(device uint *destination [[buffer(0)]],
                     device const uint *source [[buffer(1)]],
                     uint i [[thread_position_in_grid]])
{
    destination[i] = source[i];
}
