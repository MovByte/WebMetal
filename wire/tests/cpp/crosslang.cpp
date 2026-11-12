// Cross-language driver for wire/tests/crosslang.rs. Built twice by that test: as C++17 and as Objective-C++.
//
//   crosslang requests <seed> <count> <out>   encode <count> random control requests with the C++ encoder
//   crosslang commands <seed> <count> <out>   encode <count> random stream records with the C++ encoder
//   crosslang replies <seed> <count> <in>     decode Rust-encoded replies and compare with the C++ generator's values
//   crosslang events <seed> <count> <in>      decode Rust-encoded events and compare with the C++ generator's values
//   crosslang foundation <out>                (Objective-C++ only) encode a record built from Foundation objects
//
// Exit status 0 on success; anything else names the failing record on stderr.

#include "WebMetalWireFuzz.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace webmetal::wire;

static bool readFile(const char* path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    uint8_t buffer[65536];
    size_t n;
    while ((n = std::fread(buffer, 1, sizeof buffer, f)) > 0) out.insert(out.end(), buffer, buffer + n);
    std::fclose(f);
    return true;
}

static bool writeFile(const char* path, const std::vector<uint8_t>& bytes) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    return std::fclose(f) == 0 && ok;
}

static int encodeMode(bool requests, uint64_t seed, uint64_t count, const char* out) {
    fuzz::Gen g(seed);
    Writer w;
    for (uint64_t i = 0; i < count; ++i) {
        bool ok = requests ? fuzz::encodeRandomRequest(g, w) : fuzz::encodeRandomCommand(g, w);
        if (!ok) {
            std::fprintf(stderr, "encoding random record %" PRIu64 " failed\n", i);
            return 2;
        }
    }
    return writeFile(out, w.out) ? 0 : 5;
}

static int checkMode(bool replies, uint64_t seed, uint64_t count, const char* in) {
    std::vector<uint8_t> bytes;
    if (!readFile(in, bytes)) return 5;
    fuzz::Gen g(seed);
    size_t at = 0;
    for (uint64_t i = 0; i < count; ++i) {
        size_t used = 0;
        bool ok = replies ? fuzz::checkRandomReply(g, bytes.data() + at, bytes.size() - at, &used)
                          : fuzz::checkRandomEvent(g, bytes.data() + at, bytes.size() - at, &used);
        if (!ok) {
            std::fprintf(stderr, "record %" PRIu64 " at byte %zu does not decode to the expected value\n", i, at);
            return 3;
        }
        at += used;
    }
    if (at != bytes.size()) {
        std::fprintf(stderr, "%zu trailing bytes\n", bytes.size() - at);
        return 4;
    }
    return 0;
}

#ifdef __OBJC__
// A RegisterFunction request whose name comes from an NSString, and a CreateRenderPipeline request whose
// driverCompilerOptions dictionary comes from Foundation objects. crosslang.rs decodes both and checks the values.
static int foundationMode(const char* out) {
    @autoreleasepool {
        Writer w;
        RegisterFunction f;
        f.function = 7;
        f.functionType = 1;
        if (!strFromNSString(@"vertex_mainé", f.name)) return 2;
        if (!encode(w, f)) return 2;

        NSDictionary* options = @{
            @"b": @YES,
            @"a": @[ @(-3), @(4000000000ull), @1.5f, @2.25, @"text", [NSData dataWithBytes:"\x01\x02" length:2] ],
        };
        Value value;
        if (!valueFromObject(options, value)) return 2;
        CreateRenderPipeline p;
        p.pipeline = 9;
        p.descriptor.driverCompilerOptions = &value;
        if (!encode(w, p)) return 2;

        // Objects outside the property-list classes are refused, not dropped.
        Value refused;
        if (valueFromObject(@{@"x": [NSDate date]}, refused)) return 6;
        if (valueFromObject(@{@1: @"non-string key"}, refused)) return 6;
        return writeFile(out, w.out) ? 0 : 5;
    }
}
#endif

int main(int argc, char** argv) {
    if (argc < 2) return 64;
    std::string mode = argv[1];
#ifdef __OBJC__
    if (mode == "foundation" && argc == 3) return foundationMode(argv[2]);
#endif
    if (argc != 5) return 64;
    uint64_t seed = std::strtoull(argv[2], nullptr, 10);
    uint64_t count = std::strtoull(argv[3], nullptr, 10);
    if (mode == "requests") return encodeMode(true, seed, count, argv[4]);
    if (mode == "commands") return encodeMode(false, seed, count, argv[4]);
    if (mode == "replies") return checkMode(true, seed, count, argv[4]);
    if (mode == "events") return checkMode(false, seed, count, argv[4]);
    return 64;
}
