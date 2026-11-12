// Kernels for the blit cases (src/cases/Blit.m).

#include <metal_stdlib>
using namespace metal;

// Every sample of a 4x4, 4-sample R32Uint texture, sample-major per pixel.
kernel void blit_read_ms(texture2d_ms<uint, access::read> source [[texture(0)]],
                         device uint *out [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]])
{
    for (uint s = 0; s < 4; s++)
        out[(gid.y * 4 + gid.x) * 4 + s] = source.read(gid, s).r;
}
