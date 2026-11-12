// Measurement vertex-fetch-mac: how this Mac's Metal device fetches vertex and stage-in attributes.
//
// Conversions: every code of the 8- and 16-bit normalized and integer vertex formats, every 10-bit and 2-bit code of
// the 10-10-10-2 formats, the bytes of UChar4Normalized_BGRA, every binary16 pattern and 65536 binary32 patterns are
// fetched as points (one element per vertex, read back through a flat varying) and compared with exact references:
// the correctly rounded c / (2^n - 1) and max(-1, c / (2^(n-1) - 1)), the binary32 product of c and the binary32
// nearest to 1 / (2^(n-1) - 1), and IEEE round-to-nearest-even conversions. Each case records how many results equal
// each reference, and the distinct results for NaN inputs.
//
// Pipeline acceptance: for each vertex format, the attribute types (float4, half4, int4, uint4, short4, ushort4,
// char4, uchar4) a render pipeline is created for.
//
// Fetch index: the per-instance and constant step functions with a nonzero base instance; compute stage-in with
// step rates 1 to 3, origins in x and y, threads beyond the region, and both index types; unaligned strides and
// offsets.
//
// The output is TOML on stdout. The results describe this Mac's GPU only.
//
// Build and run: xcrun swiftc -O vertex_fetch_mac.swift -o vertex_fetch_mac && ./vertex_fetch_mac > vertex_fetch_mac.toml

import Foundation
import Metal

func q(_ s: String) -> String { "\"" + s.replacingOccurrences(of: "\"", with: "\\\"") + "\"" }
func fail(_ s: String) -> Never {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
    exit(1)
}

guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let queue = device.makeCommandQueue()!

let source = """
#include <metal_stdlib>
using namespace metal;
struct P { float4 pos [[position]]; uint4 data [[flat]]; float ps [[point_size]]; };
static float4 at(uint v) {
    return float4((float(v % 256u) + 0.5f) / 128.0f - 1.0f, 1.0f - (float(v / 256u) + 0.5f) / 128.0f, 0.0f, 1.0f);
}
#define FETCH(T, WIDEN) \\
    struct I_##T { T v [[attribute(0)]]; }; \\
    vertex P v_##T(I_##T in [[stage_in]], uint vid [[vertex_id]], uint iid [[instance_id]]) { \\
        P o; o.pos = at(vid + 256u * iid); o.data = WIDEN; o.ps = 1.0f; return o; \\
    }
FETCH(float4, as_type<uint4>(in.v))
FETCH(half4, uint4(as_type<ushort4>(in.v)))
FETCH(uint4, in.v)
FETCH(int4, as_type<uint4>(in.v))
FETCH(ushort4, uint4(in.v))
FETCH(short4, uint4(as_type<ushort4>(in.v)))
FETCH(uchar4, uint4(in.v))
FETCH(char4, uint4(as_type<uchar4>(in.v)))
fragment uint4 f_data(P in [[stage_in]]) { return in.data; }
struct S { uint v [[attribute(0)]]; };
kernel void k_stage_in(S in [[stage_in]], device uint *o [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    o[g.y * 8u + g.x] = in.v;
}
"""
let library: MTLLibrary
do { library = try device.makeLibrary(source: source, options: MTLCompileOptions()) } catch { fail("compile: \(error)") }

let formats: [(String, UInt)] = [
    ("UChar2", 1), ("UChar3", 2), ("UChar4", 3), ("Char2", 4), ("Char3", 5), ("Char4", 6), ("UChar2Normalized", 7),
    ("UChar3Normalized", 8), ("UChar4Normalized", 9), ("Char2Normalized", 10), ("Char3Normalized", 11),
    ("Char4Normalized", 12), ("UShort2", 13), ("UShort3", 14), ("UShort4", 15), ("Short2", 16), ("Short3", 17),
    ("Short4", 18), ("UShort2Normalized", 19), ("UShort3Normalized", 20), ("UShort4Normalized", 21),
    ("Short2Normalized", 22), ("Short3Normalized", 23), ("Short4Normalized", 24), ("Half2", 25), ("Half3", 26),
    ("Half4", 27), ("Float", 28), ("Float2", 29), ("Float3", 30), ("Float4", 31), ("Int", 32), ("Int2", 33),
    ("Int3", 34), ("Int4", 35), ("UInt", 36), ("UInt2", 37), ("UInt3", 38), ("UInt4", 39),
    ("Int1010102Normalized", 40), ("UInt1010102Normalized", 41), ("UChar4Normalized_BGRA", 42), ("UChar", 45),
    ("Char", 46), ("UCharNormalized", 47), ("CharNormalized", 48), ("UShort", 49), ("Short", 50),
    ("UShortNormalized", 51), ("ShortNormalized", 52), ("Half", 53),
]
let types = ["float4", "half4", "int4", "uint4", "short4", "ushort4", "char4", "uchar4"]

/// A render pipeline reading attribute 0 (format, offset) from buffer 1 (stride, step function, step rate).
func pipeline(_ type: String, _ format: UInt, stride: Int, offset: Int = 0, step: MTLVertexStepFunction = .perVertex,
              rate: Int = 1) -> MTLRenderPipelineState? {
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = library.makeFunction(name: "v_" + type)
    d.fragmentFunction = library.makeFunction(name: "f_data")
    d.colorAttachments[0].pixelFormat = .rgba32Uint
    let vd = MTLVertexDescriptor()
    vd.attributes[0].format = MTLVertexFormat(rawValue: format)!
    vd.attributes[0].offset = offset
    vd.attributes[0].bufferIndex = 1
    vd.layouts[1].stride = stride
    vd.layouts[1].stepFunction = step
    vd.layouts[1].stepRate = rate
    d.vertexDescriptor = vd
    return try? device.makeRenderPipelineState(descriptor: d)
}

/// Draw `count` points (instances of 256 vertices) and return each point's four fetched words.
func fetch(_ p: MTLRenderPipelineState, _ data: [UInt8], count: Int, instances: Int = 1, baseInstance: Int = 0)
    -> [UInt32]
{
    let buffer = data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: data.count)! }
    // The vertex function places point v at pixel (v % 256, v / 256) of a 256 x 256 target.
    let rows = 256
    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Uint, width: 256, height: rows,
                                                      mipmapped: false)
    td.usage = [.renderTarget]
    td.storageMode = .shared
    let t = device.makeTexture(descriptor: td)!
    let rp = MTLRenderPassDescriptor()
    rp.colorAttachments[0].texture = t
    rp.colorAttachments[0].loadAction = .clear
    rp.colorAttachments[0].storeAction = .store
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeRenderCommandEncoder(descriptor: rp)!
    e.setRenderPipelineState(p)
    e.setVertexBuffer(buffer, offset: 0, index: 1)
    e.drawPrimitives(type: .point, vertexStart: 0, vertexCount: count, instanceCount: instances,
                     baseInstance: baseInstance)
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    var words = [UInt32](repeating: 0, count: 256 * rows * 4)
    t.getBytes(&words, bytesPerRow: 256 * 16, from: MTLRegionMake2D(0, 0, 256, rows), mipmapLevel: 0)
    return words
}

// Exact references -----------------------------------------------------------------------------------------------

/// The correctly rounded binary32 bits of c / m, 0 < c < m < 2^31.
func divF32(_ c: UInt64, _ m: UInt64) -> UInt32 {
    var r = c, e: UInt64 = 127
    repeat { r <<= 1; e -= 1 } while r < m
    r -= m
    var q: UInt64 = 1
    for _ in 0..<24 { r <<= 1; q <<= 1; if r >= m { r -= m; q |= 1 } }
    let mant = q >> 1, up: UInt64 = (q & 1 == 1 && (r != 0 || mant & 1 == 1)) ? 1 : 0
    return UInt32(((e << 23) | (mant & 0x7fffff)) + up)
}

/// The correctly rounded binary16 bits of c / m, 0 < c < m < 2^31, subnormals included.
func divF16(_ c: UInt64, _ m: UInt64) -> UInt32 {
    var r = c, q: UInt64 = 0, k = 0, lead = 0, last = 25
    while true {
        k += 1; r <<= 1; q <<= 1
        if r >= m { r -= m; q |= 1; if lead == 0 { lead = k; if k <= 14 { last = k + 11 } } }
        if k == last { break }
    }
    let mant = q >> 1, up: UInt64 = (q & 1 == 1 && (r != 0 || mant & 1 == 1)) ? 1 : 0
    if lead != 0 && lead <= 14 { return UInt32((UInt64(15 - lead) << 10) + mant + up - 1024) }
    return UInt32(mant + up)
}

func unorm(_ c: UInt64, _ n: Int, half: Bool) -> UInt32 {
    let m = (UInt64(1) << n) - 1
    if c == 0 { return 0 }
    if c == m { return half ? 0x3c00 : 0x3f80_0000 }
    return half ? divF16(c, m) : divF32(c, m)
}

func snorm(_ c: UInt64, _ n: Int, half: Bool) -> UInt32 {
    let m = (UInt64(1) << (n - 1)) - 1
    let neg = (c >> (n - 1)) != 0
    let mag = neg ? (UInt64(1) << n) - c : c
    if mag == 0 { return 0 }
    let sign: UInt32 = neg ? (half ? 0x8000 : 0x8000_0000) : 0
    if mag >= m { return sign | (half ? 0x3c00 : 0x3f80_0000) }
    return sign | (half ? divF16(mag, m) : divF32(mag, m))
}

/// max(-1, Float(c) * Float(1/m)) with the binary32 nearest to 1/m (snorm), or Float(c) * Float(1/m) (unorm).
func reciprocal(_ c: UInt64, _ n: Int, signed: Bool) -> UInt32 {
    if signed {
        let m = (UInt64(1) << (n - 1)) - 1
        let v = Int64(bitPattern: c << (64 - UInt64(n))) >> (64 - Int64(n))
        if m == 1 { return Float(max(-1, v)).bitPattern }
        let r = Float(bitPattern: divF32(1, m))
        return max(Float(-1), Float(v) * r).bitPattern
    }
    let m = (UInt64(1) << n) - 1
    return (Float(c) * Float(bitPattern: divF32(1, m))).bitPattern
}

// Output ---------------------------------------------------------------------------------------------------------

var out = "# Generated by tables/measure/vertex_fetch_mac.swift. Do not edit by hand.\n\n[meta]\n"
out += "id = \"vertex-fetch-mac\"\nscript = \"tables/measure/vertex_fetch_mac.swift\"\n"
out += "device = \(q(device.name))\nhost = \(q("macOS " + ProcessInfo.processInfo.operatingSystemVersionString))\n"
func caseOut(_ id: String, _ fields: [(String, String)]) {
    out += "\n[[case]]\nid = \(q(id))\n"
    for (k, v) in fields { out += "\(k) = \(v)\n" }
}

/// Fetch every element of `data` (one element per `stride` bytes) as `type`; lane 0 of each result.
func lanes(_ type: String, _ format: UInt, _ data: [UInt8], stride: Int, lanes: Int = 1) -> [[UInt32]] {
    guard let p = pipeline(type, format, stride: stride) else { fail("pipeline \(format) \(type)") }
    let n = data.count / stride
    let w = fetch(p, data, count: n)
    return (0..<n).map { i in (0..<lanes).map { w[4 * i + $0] } }
}

// Normalized formats, every code.
let codes8 = (0..<256).map { UInt8($0) }
let codes16 = (0..<65536).flatMap { [UInt8($0 & 0xff), UInt8($0 >> 8)] }
for (name, format, n, signed, data, stride) in [
    ("UCharNormalized", UInt(47), 8, false, codes8, 1), ("CharNormalized", 48, 8, true, codes8, 1),
    ("UShortNormalized", 51, 16, false, codes16, 2), ("ShortNormalized", 52, 16, true, codes16, 2),
] {
    for type in ["float4", "half4"] {
        let half = type == "half4"
        let got = lanes(type, format, data, stride: stride)
        var cr = 0, rp = 0
        for (c, g) in got.enumerated() {
            let code = UInt64(c)
            if g[0] == (signed ? snorm(code, n, half: half) : unorm(code, n, half: half)) { cr += 1 }
            if !half && g[0] == reciprocal(code, n, signed: signed) { rp += 1 }
        }
        var fields = [("codes", "\(got.count)"), ("equal_to_correctly_rounded", "\(cr)")]
        if !half { fields.append(("equal_to_reciprocal_product", "\(rp)")) }
        caseOut("\(name)-as-\(type)", fields)
    }
}

// The 10-10-10-2 formats: every 10-bit code in x, other codes in y and z, every 2-bit code in w.
var packed: [UInt8] = []
for c in 0..<1024 as Range<UInt32> {
    let w = c | (((c * 7) & 1023) << 10) | (((c * 13) & 1023) << 20) | ((c & 3) << 30)
    packed += [UInt8(w & 0xff), UInt8((w >> 8) & 0xff), UInt8((w >> 16) & 0xff), UInt8(w >> 24)]
}
for (name, format, signed) in [("Int1010102Normalized", UInt(40), true), ("UInt1010102Normalized", 41, false)] {
    for type in ["float4", "half4"] {
        let half = type == "half4"
        let got = lanes(type, format, packed, stride: 4, lanes: 4)
        var cr10 = 0, rp10 = 0, cr2 = 0
        for (i, g) in got.enumerated() {
            let w = UInt32(packed[4 * i]) | UInt32(packed[4 * i + 1]) << 8 | UInt32(packed[4 * i + 2]) << 16
                | UInt32(packed[4 * i + 3]) << 24
            for (k, (o, b)) in [(0, 10), (10, 10), (20, 10), (30, 2)].enumerated() {
                let c = UInt64((w >> UInt32(o)) & ((1 << UInt32(b)) - 1))
                let exact = signed ? snorm(c, b, half: half) : unorm(c, b, half: half)
                if b == 10 {
                    if g[k] == exact { cr10 += 1 }
                    if !half && g[k] == reciprocal(c, b, signed: signed) { rp10 += 1 }
                } else if g[k] == exact {
                    cr2 += 1
                }
            }
        }
        var fields = [("codes_10_bit", "\(3 * got.count)"), ("equal_to_correctly_rounded_10_bit", "\(cr10)")]
        if !half { fields.append(("equal_to_reciprocal_product_10_bit", "\(rp10)")) }
        fields += [("codes_2_bit", "\(got.count)"), ("equal_to_correctly_rounded_2_bit", "\(cr2)")]
        caseOut("\(name)-as-\(type)", fields)
    }
}
for type in ["float4", "half4"] {
    let half = type == "half4"
    let got = lanes(type, 42, packed, stride: 4, lanes: 4)
    var cr = 0, order = 0
    for (i, g) in got.enumerated() {
        let bytes = (0..<4).map { UInt64(packed[4 * i + $0]) }
        let expect = [bytes[2], bytes[1], bytes[0], bytes[3]].map { unorm($0, 8, half: half) }
        cr += (0..<4).filter { g[$0] == expect[$0] }.count
        if (0..<4).allSatisfy({ g[$0] == expect[$0] }) { order += 1 }
    }
    caseOut("UChar4Normalized_BGRA-as-\(type)", [("components", "\(4 * got.count)"),
        ("equal_to_correctly_rounded_bytes_2_1_0_3", "\(cr)")])
}

// Integers to half.
for (name, format, signed) in [("UShort", UInt(49), false), ("Short", 50, true)] {
    let got = lanes("half4", format, codes16, stride: 2)
    var rne = 0
    for (c, g) in got.enumerated() {
        let v = signed ? Int(Int16(bitPattern: UInt16(c))) : c
        if g[0] == UInt32(Float16(v).bitPattern) { rne += 1 }
    }
    caseOut("\(name)-as-half4", [("codes", "\(got.count)"), ("equal_to_round_to_nearest_even", "\(rne)")])
}

// binary16 to binary32 and binary32 to binary16.
do {
    let got = lanes("float4", 53, codes16, stride: 2)
    var exact = 0, nans = Set<UInt32>(), nanCount = 0
    for (h, g) in got.enumerated() {
        let x = Float16(bitPattern: UInt16(h))
        if x.isNaN { nanCount += 1; nans.insert(g[0]) } else if g[0] == Float(x).bitPattern { exact += 1 }
    }
    caseOut("Half-as-float4", [("patterns", "\(got.count)"), ("non_nan_equal_to_exact", "\(exact)"),
        ("nan_patterns", "\(nanCount)"), ("nan_results", "[\(nans.sorted().map { q(String(format: "0x%08x", $0)) }.joined(separator: ", "))]")])
}
do {
    var words: [UInt32] = [0x7fc0_0000, 0xffc0_0000, 0x7f80_0001, 0xff80_0001, 0x7fa0_0000, 0x7fff_ffff, 0x7f80_0000,
                           0xff80_0000, 0, 0x8000_0000, 0x477f_e000, 0x477f_f000, 0x477f_efff, 0x477f_f001, 0x3300_0000,
                           0x3300_0001, 0x32ff_ffff, 0x387f_c000, 0x387f_e000, 0x3880_0000, 0x3f80_1000, 0x3f80_3000]
    var s: UInt32 = 0x1234_5677
    while words.count < 65536 { s ^= s << 13; s ^= s >> 17; s ^= s << 5; words.append(s) }
    let data = words.flatMap { w in (0..<4).map { UInt8((w >> (8 * $0)) & 0xff) } }
    let got = lanes("half4", 28, data, stride: 4)
    var rne = 0, nans = Set<UInt32>(), nanCount = 0
    for (i, g) in got.enumerated() {
        let x = Float(bitPattern: words[i])
        if x.isNaN { nanCount += 1; nans.insert(g[0]) } else if g[0] == UInt32(Float16(x).bitPattern) { rne += 1 }
    }
    caseOut("Float-as-half4", [("patterns", "\(got.count)"), ("non_nan_equal_to_round_to_nearest_even", "\(rne)"),
        ("nan_patterns", "\(nanCount)"), ("nan_results", "[\(nans.sorted().map { q(String(format: "0x%04x", $0)) }.joined(separator: ", "))]")])
}

// Pipeline acceptance.
for (name, value) in formats {
    let accepted = types.filter { pipeline($0, value, stride: 16) != nil }
    caseOut("accepts-\(name)", [("attribute_types", "[\(accepted.map(q).joined(separator: ", "))]")])
}

// Step functions with a base instance: buffer element i holds 1000 + i; four instances from base instance 3.
let elements = (0..<64).flatMap { i -> [UInt8] in let w = UInt32(1000 + i); return (0..<4).map { UInt8((w >> (8 * $0)) & 0xff) } }
for (id, step, rate) in [("per-instance-rate-2-base-3", MTLVertexStepFunction.perInstance, 2), ("constant-base-3", .constant, 0)] {
    guard let p = pipeline("uint4", 36, stride: 4, step: step, rate: rate) else { fail(id) }
    let w = fetch(p, elements, count: 1, instances: 4, baseInstance: 3)
    let fetched = (3..<7).map { w[4 * (256 * $0)] - 1000 }
    caseOut(id, [("instances", "[3, 4, 5, 6]"), ("fetched_elements", "[\(fetched.map(String.init).joined(separator: ", "))]")])
}

// Unaligned strides and offsets.
do {
    guard let p = pipeline("uint4", 14, stride: 6) else { fail("UShort3 stride 6") }
    let got = fetch(p, Array(codes16[0..<(6 * 64)]), count: 64)
    var ok = 0
    for i in 0..<64 { for k in 0..<3 { let b = 6 * i + 2 * k; if got[4 * i + k] == UInt32(codes16[b]) | UInt32(codes16[b + 1]) << 8 { ok += 1 } } }
    caseOut("UShort3-stride-6", [("pipeline", "\"created\""), ("components", "192"), ("equal_to_the_bytes_read", "\(ok)")])
    guard let p2 = pipeline("uint4", 36, stride: 8, offset: 2) else { fail("UInt offset 2") }
    let got2 = fetch(p2, Array(codes16[0..<(8 * 64)]), count: 63)
    var ok2 = 0
    for i in 0..<63 {
        let b = 8 * i + 2
        let word = (0..<4).reduce(UInt32(0), { $0 | UInt32(codes16[b + $1]) << (8 * UInt32($1)) })
        if got2[4 * i] == word { ok2 += 1 }
    }
    caseOut("UInt-offset-2", [("pipeline", "\"created\""), ("components", "63"), ("equal_to_the_bytes_read", "\(ok2)")])
}

// Compute stage-in: buffer element i holds 1000 + i (index buffers hold 100 + 3i); an 8 x 4 grid of threads.
func stageIn(_ step: MTLStepFunction, rate: Int, origin: (Int, Int), size: (Int, Int), indexType: MTLIndexType? = nil)
    -> [[Int]]
{
    let sd = MTLStageInputOutputDescriptor()
    sd.attributes[0].format = .uint
    sd.attributes[0].bufferIndex = 1
    sd.layouts[1].stride = 4
    sd.layouts[1].stepFunction = step
    sd.layouts[1].stepRate = rate
    if let t = indexType { sd.indexType = t; sd.indexBufferIndex = 2 }
    let cd = MTLComputePipelineDescriptor()
    cd.computeFunction = library.makeFunction(name: "k_stage_in")
    cd.stageInputDescriptor = sd
    let p = try! device.makeComputePipelineState(descriptor: cd, options: [], reflection: nil)
    let vals = (0..<256).map { UInt32(1000 + $0) }
    let vb = device.makeBuffer(bytes: vals, length: 1024)!
    let ib: MTLBuffer
    if indexType == .uint16 {
        let i16 = (0..<64).map { UInt16(100 + 3 * $0) }
        ib = device.makeBuffer(bytes: i16, length: 128)!
    } else {
        let i32 = (0..<64).map { UInt32(100 + 3 * $0) }
        ib = device.makeBuffer(bytes: i32, length: 256)!
    }
    let ob = device.makeBuffer(length: 32 * 4, options: .storageModeShared)!
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p)
    e.setBuffer(ob, offset: 0, index: 0)
    e.setBuffer(vb, offset: 0, index: 1)
    e.setBuffer(ib, offset: 0, index: 2)
    e.setStageInRegion(MTLRegionMake2D(origin.0, origin.1, size.0, size.1))
    e.dispatchThreadgroups(MTLSizeMake(2, 2, 1), threadsPerThreadgroup: MTLSizeMake(4, 2, 1))
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let w = ob.contents().bindMemory(to: UInt32.self, capacity: 32)
    return (0..<4).map { y in (0..<8).map { x in Int(w[y * 8 + x]) - 1000 } }
}
func rows(_ r: [[Int]]) -> String { "[" + r.map { "\"" + $0.map(String.init).joined(separator: " ") + "\"" }.joined(separator: ", ") + "]" }
for (id, step, rate, origin, size, it) in [
    ("x-rate-1-origin-5-size-6", MTLStepFunction.threadPositionInGridX, 1, (5, 0), (6, 2), nil as MTLIndexType?),
    ("x-rate-2-origin-1-size-6", .threadPositionInGridX, 2, (1, 1), (6, 2), nil),
    ("x-rate-3-origin-2-size-3", .threadPositionInGridX, 3, (2, 0), (3, 4), nil),
    ("y-rate-1-origin-1-size-2", .threadPositionInGridY, 1, (0, 1), (3, 2), nil),
    ("y-rate-3-origin-0-size-4", .threadPositionInGridY, 3, (0, 0), (3, 4), nil),
    ("x-indexed-uint32-rate-1-origin-1-size-5", .threadPositionInGridXIndexed, 1, (1, 0), (5, 2), .uint32),
    ("x-indexed-uint16-rate-2-origin-1-size-5", .threadPositionInGridXIndexed, 2, (1, 0), (5, 2), .uint16),
    ("y-indexed-uint32-rate-1-origin-1-size-2", .threadPositionInGridYIndexed, 1, (0, 1), (5, 2), .uint32),
    ("constant-origin-2-1", .constant, 0, (2, 1), (3, 2), nil),
] {
    caseOut("stage-in-" + id, [("fetched_elements_by_row", rows(stageIn(step, rate: rate, origin: origin, size: size,
                                                                       indexType: it)))])
}

print(out, terminator: "")
