// Shaders for the system-client SPI cases (src/cases/SPI.m).

#include <metal_stdlib>
using namespace metal;

struct SPIVarying {
    float4 position [[position]];
};

// One triangle that covers the whole target, at depth 0.5.
vertex SPIVarying spi_cover_vertex(uint vid [[vertex_id]])
{
    const float2 corners[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    SPIVarying out;
    out.position = float4(corners[vid], 0.5, 1.0);
    return out;
}

// Values whose RGBA8Unorm codes do not depend on rounding: 64, 128, 192, 255.
fragment float4 spi_color_fragment(SPIVarying in [[stage_in]])
{
    return float4(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0);
}

// A fragment function whose color output is in raster order group 0.
struct SPIOrderedColor {
    float4 color [[color(0), raster_order_group(0)]];
};

fragment SPIOrderedColor spi_ordered_fragment(SPIVarying in [[stage_in]])
{
    SPIOrderedColor out;
    out.color = float4(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0);
    return out;
}
