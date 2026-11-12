// Prints CoreVideo's description of the pixel formats the presentation sink classifies
// (host/src/present/format.rs), from CVPixelFormatDescriptionCreateWithPixelFormatType.
// The output on the reference system is pixel_formats.txt.
//
//   swiftc -O pixel_formats.swift -o pixel_formats && ./pixel_formats > pixel_formats.txt

import CoreVideo
import Foundation

func fourcc(_ s: String) -> OSType { s.utf8.reduce(0) { ($0 << 8) | OSType($1) } }

let codes: [(String, OSType)] = [
    ("kCVPixelFormatType_32ARGB", 0x20),
    ("kCVPixelFormatType_32BGRA", fourcc("BGRA")),
    ("kCVPixelFormatType_32ABGR", fourcc("ABGR")),
    ("kCVPixelFormatType_32RGBA", fourcc("RGBA")),
    ("kCVPixelFormatType_ARGB2101010LEPacked", fourcc("l10r")),
    ("kCVPixelFormatType_30RGBLEPackedWideGamut", fourcc("w30r")),
    ("kCVPixelFormatType_40ARGBLEWideGamut", fourcc("w40a")),
    ("kCVPixelFormatType_64RGBAHalf", fourcc("RGhA")),
]
// Keys whose values are addresses in this process.
let skipped: Set<String> = ["FillExtendedPixelsCallback"]

let v = ProcessInfo.processInfo.operatingSystemVersionString
print("# \(v)")
for (name, code) in codes {
    print("\(name) 0x\(String(code, radix: 16))")
    guard let d = CVPixelFormatDescriptionCreateWithPixelFormatType(kCFAllocatorDefault, code) as? [String: Any] else {
        print("  (no description)")
        continue
    }
    for k in d.keys.sorted() where !skipped.contains(k) {
        print("  \(k) = \(d[k]!)")
    }
}
