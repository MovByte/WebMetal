// Shaders for the tessellation cases (src/cases/Tessellation.m).
//
// A tessellated draw is observed twice. The enumeration draw records every
// post-tessellation invocation's patch, instance and position in the patch
// (rasterization disabled). The order draws rasterize each generated triangle
// at pseudo-random window positions chosen per point and per draw, and every
// fragment appends the triangle it belongs to, as point indices, to a list
// kept per pixel: a fragment reads its pixel's list length from the color
// attachment (programmable blending, which Metal orders by primitive at each
// pixel) and returns it plus one. The case orders the triangles by those
// per-pixel lists.
//
// A fragment learns its triangle's points through its varyings: the first
// point's index is a flat varying (the provoking vertex is the triangle's
// first), and each point's index is also encoded as eight one-hot base-4
// digits in linearly interpolated varyings, from which the fragment, given its
// barycentric coordinates, recovers the second and third points' digits.

#include <metal_stdlib>
using namespace metal;

struct TessPoint {
    uint instance;
    uint patch;
    uint u, v, w; // position in the patch, binary32 bit patterns (w = 0 on a quad)
};

static void tess_record(device atomic_uint *count, device TessPoint *points, uint capacity, TessPoint p)
{
    uint i = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    if (i < capacity)
        points[i] = p;
}

[[patch(quad, 4)]]
vertex void tess_enum_quad(device atomic_uint *count [[buffer(0)]], device TessPoint *points [[buffer(1)]],
                           constant uint &capacity [[buffer(2)]], float2 p [[position_in_patch]],
                           uint patch [[patch_id]], uint instance [[instance_id]])
{
    tess_record(count, points, capacity, TessPoint{instance, patch, as_type<uint>(p.x), as_type<uint>(p.y), 0u});
}

[[patch(triangle, 3)]]
vertex void tess_enum_triangle(device atomic_uint *count [[buffer(0)]], device TessPoint *points [[buffer(1)]],
                               constant uint &capacity [[buffer(2)]], float3 p [[position_in_patch]],
                               uint patch [[patch_id]], uint instance [[instance_id]])
{
    tess_record(count, points, capacity,
                TessPoint{instance, patch, as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z)});
}

struct TessOrderOut {
    float4 position [[position]];
    uint first [[flat]];
    float4 d0 [[center_no_perspective]];
    float4 d1 [[center_no_perspective]];
    float4 d2 [[center_no_perspective]];
    float4 d3 [[center_no_perspective]];
    float4 d4 [[center_no_perspective]];
    float4 d5 [[center_no_perspective]];
    float4 d6 [[center_no_perspective]];
    float4 d7 [[center_no_perspective]];
};

struct TessOrderArgs {
    uint count;   // points in the sorted table
    uint seed;    // order draw
    uint width;   // target width in pixels
    uint capacity; // list entries per pixel
};

static uint tess_hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static bool tess_less(TessPoint a, TessPoint b)
{
    if (a.instance != b.instance) return a.instance < b.instance;
    if (a.patch != b.patch) return a.patch < b.patch;
    if (a.u != b.u) return a.u < b.u;
    if (a.v != b.v) return a.v < b.v;
    return a.w < b.w;
}

static float4 tess_digit(uint id, uint k)
{
    uint d = (id >> (2u * k)) & 3u;
    return float4(d == 0u, d == 1u, d == 2u, d == 3u);
}

static TessOrderOut tess_order(TessPoint key, constant TessPoint *table, constant TessOrderArgs &args)
{
    // The point's index in the table, which is sorted by tess_less.
    uint lo = 0, hi = args.count;
    while (lo < hi) {
        uint m = (lo + hi) / 2;
        if (tess_less(table[m], key)) lo = m + 1; else hi = m;
    }
    uint id = lo;
    uint h = tess_hash(id * 2654435761u + args.seed * 97u + 13u);
    uint h2 = tess_hash(h ^ 0x5bd1e995u);
    TessOrderOut o;
    o.position = float4(float(h & 0xffffu) / 32768.0f - 1.0f, float(h2 & 0xffffu) / 32768.0f - 1.0f, 0.5f, 1.0f);
    o.first = id;
    o.d0 = tess_digit(id, 0); o.d1 = tess_digit(id, 1); o.d2 = tess_digit(id, 2); o.d3 = tess_digit(id, 3);
    o.d4 = tess_digit(id, 4); o.d5 = tess_digit(id, 5); o.d6 = tess_digit(id, 6); o.d7 = tess_digit(id, 7);
    return o;
}

[[patch(quad, 4)]]
vertex TessOrderOut tess_order_quad(constant TessPoint *table [[buffer(0)]], constant TessOrderArgs &args [[buffer(1)]],
                                    float2 p [[position_in_patch]], uint patch [[patch_id]],
                                    uint instance [[instance_id]])
{
    return tess_order(TessPoint{instance, patch, as_type<uint>(p.x), as_type<uint>(p.y), 0u}, table, args);
}

[[patch(triangle, 3)]]
vertex TessOrderOut tess_order_triangle(constant TessPoint *table [[buffer(0)]],
                                        constant TessOrderArgs &args [[buffer(1)]], float3 p [[position_in_patch]],
                                        uint patch [[patch_id]], uint instance [[instance_id]])
{
    return tess_order(TessPoint{instance, patch, as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z)}, table,
                      args);
}

// Digit k of the second and third points: the interpolated one-hot vector less the first point's share is
// b.y * e(second) + b.z * e(third); the best of the 16 candidate pairs, and the margin to the next best.
static void tess_decode(float4 h, uint k, uint first, float3 b, thread uint &second, thread uint &third,
                        thread float &worst, thread float &margin)
{
    float4 r = h - b.x * tess_digit(first, k);
    float best = INFINITY, next = INFINITY;
    uint bx = 0, by = 0;
    for (uint x = 0; x < 4u; x++) {
        for (uint y = 0; y < 4u; y++) {
            float4 e = r - b.y * tess_digit(x, 0) - b.z * tess_digit(y, 0);
            float s = dot(e, e);
            if (s < best) { next = best; best = s; bx = x; by = y; } else if (s < next) { next = s; }
        }
    }
    second |= bx << (2u * k);
    third |= by << (2u * k);
    worst = max(worst, best);
    margin = min(margin, next);
}

fragment uint tess_order_fragment(TessOrderOut in [[stage_in]], float3 b [[barycentric_coord]],
                                  uint length [[color(0)]], device uint4 *lists [[buffer(0)]],
                                  constant TessOrderArgs &args [[buffer(1)]])
{
    uint second = 0, third = 0;
    float worst = 0, margin = INFINITY;
    tess_decode(in.d0, 0, in.first, b, second, third, worst, margin);
    tess_decode(in.d1, 1, in.first, b, second, third, worst, margin);
    tess_decode(in.d2, 2, in.first, b, second, third, worst, margin);
    tess_decode(in.d3, 3, in.first, b, second, third, worst, margin);
    tess_decode(in.d4, 4, in.first, b, second, third, worst, margin);
    tess_decode(in.d5, 5, in.first, b, second, third, worst, margin);
    tess_decode(in.d6, 6, in.first, b, second, third, worst, margin);
    tess_decode(in.d7, 7, in.first, b, second, third, worst, margin);
    // Decoded when every digit fits one pair closely and no other pair comes near.
    uint decoded = (worst < 1e-6f && margin > 1e-4f) ? 1u : 0u;
    uint pixel = uint(in.position.y) * args.width + uint(in.position.x);
    if (length < args.capacity)
        lists[pixel * args.capacity + length] = uint4(in.first, second, third, decoded);
    return length + 1u;
}

// Rendered patch draws (tessellation.render). Each point is placed at a window
// position hashed from its position in the patch, its patch identifier and its
// instance, so that the generated triangles overlap, and carries a flat code
// that mixes the hash with two of its patch's control points and the patch's
// per-patch attribute. A pixel of the target holds the code of the last
// triangle drawn over it: the result depends on every triangle, on its first
// point (the provoking vertex) and on the order of the triangles.

struct TessDrawPoint {
    float4 p [[attribute(0)]];
};

struct TessDrawPatch {
    patch_control_point<TessDrawPoint> cp;
    uint tag [[attribute(1)]];
};

struct TessDrawOut {
    float4 position [[position]];
    uint code [[flat]];
};

static TessDrawOut tess_place(uint a, uint b, uint c, uint patch, uint instance, uint tag, float4 cp0, float4 cp1)
{
    uint h = tess_hash(a ^ tess_hash(b ^ tess_hash(c ^ tess_hash(patch * 31u + instance))));
    // Clip-space positions on a grid of 1/64, exact in binary32 (the control points' offsets are on the grid too).
    float2 xy = float2(float(h & 0x7fu), float((h >> 7) & 0x7fu)) / 64.0f - 1.0f + cp0.xy;
    TessDrawOut o;
    o.position = float4(xy, 0.5f, 1.0f);
    o.code = (h | 1u) ^ tag ^ as_type<uint>(cp1.w);
    return o;
}

[[patch(quad, 4)]]
vertex TessDrawOut tess_draw_quad(TessDrawPatch in [[stage_in]], float2 p [[position_in_patch]],
                                  uint patch [[patch_id]], uint instance [[instance_id]])
{
    return tess_place(as_type<uint>(p.x), as_type<uint>(p.y), 0u, patch, instance, in.tag, in.cp[0].p, in.cp[3].p);
}

[[patch(triangle, 3)]]
vertex TessDrawOut tess_draw_triangle(TessDrawPatch in [[stage_in]], float3 p [[position_in_patch]],
                                      uint patch [[patch_id]], uint instance [[instance_id]])
{
    return tess_place(as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z), patch, instance, in.tag,
                      in.cp[1].p, in.cp[2].p);
}

fragment uint tess_draw_fragment(TessDrawOut in [[stage_in]])
{
    return in.code;
}
