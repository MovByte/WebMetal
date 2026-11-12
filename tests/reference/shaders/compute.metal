// Kernels for the compute cases (src/cases/Compute.m).

#include <metal_stdlib>
using namespace metal;

// One record of 16 words per thread, at the thread's linear grid index.
kernel void compute_positions(device uint *out [[buffer(0)]],
                              constant uint3 &grid [[buffer(1)]],
                              uint3 tpig [[thread_position_in_grid]],
                              uint3 tgpig [[threadgroup_position_in_grid]],
                              uint3 tpitg [[thread_position_in_threadgroup]],
                              uint tiitg [[thread_index_in_threadgroup]],
                              uint3 tptg [[threads_per_threadgroup]],
                              uint3 tgpg [[threadgroups_per_grid]],
                              uint3 tpg [[threads_per_grid]])
{
    uint index = (tpig.z * grid.y + tpig.y) * grid.x + tpig.x;
    device uint *r = out + index * 16;
    r[0] = tpig.x; r[1] = tpig.y; r[2] = tpig.z;
    r[3] = tgpig.x; r[4] = tgpig.y; r[5] = tgpig.z;
    r[6] = tpitg.x; r[7] = tpitg.y; r[8] = tpitg.z;
    r[9] = tiitg;
    r[10] = tptg.x; r[11] = tptg.y; r[12] = tptg.z;
    r[13] = tgpg.x * 0x10000u + tgpg.y * 0x100u + tgpg.z;
    r[14] = tpg.x * 0x10000u + tpg.y * 0x100u + tpg.z;
    r[15] = 0xc0ffee00u;
}

// Each thread stores its index in threadgroup memory, waits at a barrier and
// reads the value of the thread that mirrors it in the group.
kernel void compute_threadgroup_mirror(device uint *out [[buffer(0)]],
                                       threadgroup uint *scratch [[threadgroup(0)]],
                                       uint tiitg [[thread_index_in_threadgroup]],
                                       uint tptg [[threads_per_threadgroup]],
                                       uint tgpig [[threadgroup_position_in_grid]])
{
    scratch[tiitg] = tgpig * 1000u + tiitg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[tgpig * tptg + tiitg] = scratch[tptg - 1u - tiitg];
}

kernel void compute_copy_constant(device uint *out [[buffer(0)]],
                                  constant uint *in [[buffer(1)]],
                                  uint i [[thread_position_in_grid]])
{
    out[i] = in[i];
}

kernel void compute_add_one(device uint *data [[buffer(0)]],
                            uint i [[thread_position_in_grid]])
{
    data[i] = data[i] + 1u;
}

kernel void compute_scale(device uint *out [[buffer(0)]],
                          device const uint *in [[buffer(1)]],
                          constant uint &factor [[buffer(2)]],
                          uint i [[thread_position_in_grid]])
{
    out[i] = in[i] * factor;
}

// Integer operations whose results MSL defines exactly. Inputs avoid the
// cases the specification leaves unspecified (division by zero, signed
// overflow in division, shift counts at or above the width).
kernel void compute_integer_ops(device uint *out [[buffer(0)]],
                                device const uint *a [[buffer(1)]],
                                device const uint *b [[buffer(2)]],
                                uint i [[thread_position_in_grid]])
{
    uint x = a[i], y = b[i];
    int sx = int(x), sy = int(y);
    uint ny = y | 1u;                       // nonzero divisor
    int sny = sy | 1;                       // nonzero, odd, so never -1 with INT_MIN
    if (sny == -1) sny = 3;
    uint s = y & 31u;
    device uint *r = out + i * 32;
    r[0] = x + y;
    r[1] = x - y;
    r[2] = x * y;
    r[3] = mulhi(x, y);
    r[4] = uint(mulhi(sx, sy));
    r[5] = x / ny;
    r[6] = x % ny;
    r[7] = uint(sx / sny);
    r[8] = uint(sx % sny);
    r[9] = x << s;
    r[10] = x >> s;
    r[11] = uint(sx >> int(s));
    r[12] = clz(x);
    r[13] = ctz(x);
    r[14] = popcount(x);
    r[15] = rotate(x, s);
    r[16] = uint(abs(sx == INT_MIN ? 0 : sx));
    r[17] = absdiff(x, y);
    r[18] = uint(absdiff(sx, sy));
    r[19] = min(x, y);
    r[20] = uint(max(sx, sy));
    r[21] = addsat(x, y);
    r[22] = subsat(x, y);
    r[23] = uint(addsat(sx, sy));
    r[24] = uint(subsat(sx, sy));
    r[25] = hadd(x, y);
    r[26] = rhadd(x, y);
    r[27] = uint(hadd(sx, sy));
    r[28] = extract_bits(x, s & 15u, 8u);
    r[29] = insert_bits(x, y, s & 15u, 8u);
    r[30] = reverse_bits(x);
    r[31] = uint(clamp(sx, -1000, 1000));
}

// Conversions that section 8.6 of the MSL specification defines exactly for
// these inputs: float to integer rounds toward zero; float to half rounds to
// nearest even; half to float is lossless.
kernel void compute_conversions(device uint *out [[buffer(0)]],
                                device const float *f [[buffer(1)]],
                                device const ushort *h [[buffer(2)]],
                                uint i [[thread_position_in_grid]])
{
    device uint *r = out + i * 4;
    float x = f[i];
    r[0] = uint(int(x));
    r[1] = as_type<ushort>(half(x));
    r[2] = as_type<uint>(float(as_type<half>(h[i])));
    r[3] = x >= 0.0f ? uint(x) : 0u;
}

// Device-memory atomics under contention from every thread of the grid.
kernel void compute_atomics(device atomic_uint *counters [[buffer(0)]],
                            device uint *fetched [[buffer(1)]],
                            uint i [[thread_position_in_grid]])
{
    fetched[i] = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
    atomic_fetch_max_explicit(&counters[1], i * 7u, memory_order_relaxed);
    atomic_fetch_min_explicit(&counters[2], 1000u + i, memory_order_relaxed);
    atomic_fetch_or_explicit(&counters[3], 1u << (i & 31u), memory_order_relaxed);
    atomic_fetch_and_explicit(&counters[4], ~(1u << (i & 31u)), memory_order_relaxed);
    atomic_fetch_xor_explicit(&counters[5], i, memory_order_relaxed);
    atomic_fetch_sub_explicit(&counters[6], 2u, memory_order_relaxed);
    uint expected = atomic_load_explicit(&counters[7], memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(&counters[7], &expected, expected + 3u,
                                                  memory_order_relaxed, memory_order_relaxed)) {
    }
}

// Threadgroup-memory atomics; one result per threadgroup.
kernel void compute_threadgroup_atomics(device uint *out [[buffer(0)]],
                                        uint tiitg [[thread_index_in_threadgroup]],
                                        uint tgpig [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint sum;
    threadgroup atomic_uint maximum;
    if (tiitg == 0) {
        atomic_store_explicit(&sum, 0u, memory_order_relaxed);
        atomic_store_explicit(&maximum, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_fetch_add_explicit(&sum, tiitg + tgpig, memory_order_relaxed);
    atomic_fetch_max_explicit(&maximum, tiitg * 3u + tgpig, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiitg == 0) {
        out[tgpig * 2] = atomic_load_explicit(&sum, memory_order_relaxed);
        out[tgpig * 2 + 1] = atomic_load_explicit(&maximum, memory_order_relaxed);
    }
}

// Compute stage-in: one attribute record per thread, fetched by the
// pipeline's stage input descriptor from the region set by the encoder.
struct StageInRecord {
    uint2 pair [[attribute(0)]];
    float scale [[attribute(1)]];
};

kernel void compute_stage_in(StageInRecord in [[stage_in]],
                             device uint *out [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]],
                             uint2 grid [[threads_per_grid]])
{
    device uint *r = out + (gid.y * grid.x + gid.x) * 4;
    r[0] = in.pair.x;
    r[1] = in.pair.y;
    r[2] = as_type<uint>(in.scale);
    r[3] = gid.y * 0x100u + gid.x;
}
