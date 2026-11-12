// Reference decodes for the WebMetal format tests, from the Khronos VK-GL-CTS framework decoders
// (framework/common/tcuCompressedTexture.cpp and tcuAstcUtil.cpp), compiled into this translation unit so that their
// internal block decoders are reachable. gen_vectors.py builds and runs it; see that file for the CTS revision.
//
// Modes (input on stdin, one record per line):
//   astc-gen    no input; prints "<w> <h> <block hex>" for the CTS ASTC block test cases (every non-HDR
//               BlockTestType), 256 random valid LDR blocks and 256 random blocks of each LDR footprint
//   astc-hash   "<w> <h> <block hex>" -> "<w> <h> <block hex> <linear digest> <sRGB digest>"
//   etc-hash    "<kind> <block hex>"  -> "<kind> <block hex> <digest>", kind rgb8, rgb8a1, rgba8, r11u or r11s
//
// Digests are FNV-1a 64 of the decoded block in raster order: ASTC linear LDR results as little-endian UNORM16 (the
// reference returns C / 65536, or 1.0 for C = 65535), ASTC sRGB-mode results as bytes, ETC2 RGBA bytes, EAC 11-bit
// values as little-endian 16-bit integers.
#include "tcuCompressedTexture.cpp"
#include "tcuAstcUtil.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace tcu
{
// Framework functions the included files reference from paths this tool never runs.
PixelBufferAccess getSubregion(const PixelBufferAccess &, int, int, int, int, int, int) { abort(); }
void copy(const PixelBufferAccess &, const ConstPixelBufferAccess &, const bool) { abort(); }
int TextureFormat::getPixelSize() const { abort(); }
ConstPixelBufferAccess::ConstPixelBufferAccess() { abort(); }
PixelBufferAccess::PixelBufferAccess(const TextureFormat &, int, int, int, void *) { abort(); }
void PixelBufferAccess::setPixel(const Vec4 &, int, int, int) const { abort(); }
void PixelBufferAccess::setPixel(const IVec4 &, int, int, int) const { abort(); }
InternalError::InternalError(const std::string &m) : TestException(m.c_str(), nullptr, nullptr, 0, QP_TEST_RESULT_FAIL)
{
    abort();
}
TestException::TestException(const char *m, const char *e, const char *f, int l, qpTestResult r)
    : Exception(m, e, f, l), m_result(r)
{
    abort();
}
Exception::Exception(const char *m, const char *, const char *, int) : std::runtime_error(m), m_message(m) { abort(); }
} // namespace tcu

static std::vector<uint8_t> unhex(const char *s)
{
    std::vector<uint8_t> out;
    for (size_t i = 0; s[i] && s[i + 1]; i += 2)
    {
        char b[3] = {s[i], s[i + 1], 0};
        out.push_back((uint8_t)strtoul(b, nullptr, 16));
    }
    return out;
}

static void hex(const uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        printf("%02x", p[i]);
}

struct Fmt
{
    int w, h;
    tcu::CompressedTexFormat linear, srgb;
};
static const Fmt kAstc[] = {
    {4, 4, tcu::COMPRESSEDTEXFORMAT_ASTC_4x4_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_4x4_SRGB8_ALPHA8},
    {5, 4, tcu::COMPRESSEDTEXFORMAT_ASTC_5x4_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_5x4_SRGB8_ALPHA8},
    {5, 5, tcu::COMPRESSEDTEXFORMAT_ASTC_5x5_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_5x5_SRGB8_ALPHA8},
    {6, 5, tcu::COMPRESSEDTEXFORMAT_ASTC_6x5_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_6x5_SRGB8_ALPHA8},
    {6, 6, tcu::COMPRESSEDTEXFORMAT_ASTC_6x6_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_6x6_SRGB8_ALPHA8},
    {8, 5, tcu::COMPRESSEDTEXFORMAT_ASTC_8x5_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_8x5_SRGB8_ALPHA8},
    {8, 6, tcu::COMPRESSEDTEXFORMAT_ASTC_8x6_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_8x6_SRGB8_ALPHA8},
    {8, 8, tcu::COMPRESSEDTEXFORMAT_ASTC_8x8_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_8x8_SRGB8_ALPHA8},
    {10, 5, tcu::COMPRESSEDTEXFORMAT_ASTC_10x5_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_10x5_SRGB8_ALPHA8},
    {10, 6, tcu::COMPRESSEDTEXFORMAT_ASTC_10x6_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_10x6_SRGB8_ALPHA8},
    {10, 8, tcu::COMPRESSEDTEXFORMAT_ASTC_10x8_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_10x8_SRGB8_ALPHA8},
    {10, 10, tcu::COMPRESSEDTEXFORMAT_ASTC_10x10_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_10x10_SRGB8_ALPHA8},
    {12, 10, tcu::COMPRESSEDTEXFORMAT_ASTC_12x10_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_12x10_SRGB8_ALPHA8},
    {12, 12, tcu::COMPRESSEDTEXFORMAT_ASTC_12x12_RGBA, tcu::COMPRESSEDTEXFORMAT_ASTC_12x12_SRGB8_ALPHA8},
};

static void astcGen()
{
    using namespace tcu::astc;
    for (const Fmt &f : kAstc)
    {
        std::vector<uint8_t> blocks;
        for (int t = 0; t < BLOCK_TEST_TYPE_LAST; t++)
        {
            if (isBlockTestTypeHDROnly((BlockTestType)t))
                continue;
            std::vector<uint8_t> d;
            generateBlockCaseTestData(d, f.linear, (BlockTestType)t);
            blocks.insert(blocks.end(), d.begin(), d.end());
        }
        std::vector<uint8_t> v(16 * 256);
        generateRandomValidBlocks(v.data(), 256, f.linear, tcu::TexDecompressionParams::ASTCMODE_LDR, 0x1234 + f.w * 16 + f.h);
        blocks.insert(blocks.end(), v.begin(), v.end());
        std::vector<uint8_t> r(16 * 256);
        generateRandomBlocks(r.data(), 256, f.linear, 0x5678 + f.w * 16 + f.h);
        blocks.insert(blocks.end(), r.begin(), r.end());
        for (size_t i = 0; i < blocks.size(); i += 16)
        {
            printf("%d %d ", f.w, f.h);
            hex(&blocks[i], 16);
            printf("\n");
        }
    }
}

static uint64_t fnv(const uint8_t *p, size_t n, uint64_t h = 0xcbf29ce484222325ull)
{
    for (size_t i = 0; i < n; i++)
    {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// For each "<w> <h> <hex>" line on stdin: "<w> <h> <hex> <fnv64 of linear UNORM16 LE> <fnv64 of sRGB bytes>".
static void astcHash()
{
    char line[256];
    while (fgets(line, sizeof line, stdin))
    {
        int w, h;
        char hx[64];
        if (sscanf(line, "%d %d %63s", &w, &h, hx) != 3)
            continue;
        std::vector<uint8_t> b = unhex(hx);
        const tcu::astc::Block128 block(b.data());
        float lin[144 * 4];
        uint8_t srgb[144 * 4];
        tcu::astc::decompressBlock(lin, block, w, h, 1, false, true);
        tcu::astc::decompressBlock(srgb, block, w, h, 1, true, true);
        uint8_t c16[144 * 4 * 2];
        for (int i = 0; i < w * h * 4; i++)
        {
            float v = lin[i];
            unsigned c = v == 1.0f ? 65535u : (unsigned)(v * 65536.0f);
            if (v != 1.0f && (float)c / 65536.0f != v)
            {
                fprintf(stderr, "non-UNORM16 value %a\n", v);
                exit(1);
            }
            c16[i * 2] = (uint8_t)c;
            c16[i * 2 + 1] = (uint8_t)(c >> 8);
        }
        printf("%d %d %s %016llx %016llx\n", w, h, hx, (unsigned long long)fnv(c16, w * h * 8),
               (unsigned long long)fnv(srgb, w * h * 4));
    }
}

// For each "<kind> <hex>" line on stdin: "<kind> <hex> <fnv64>" of the decoded block in raster order: RGBA bytes for
// rgb8, rgb8a1 and rgba8; 16-bit little-endian values for r11u and r11s.
static void etcHash()
{
    using namespace tcu::EtcDecompressInternal;
    char line[256];
    while (fgets(line, sizeof line, stdin))
    {
        char kind[16], hx[64];
        if (sscanf(line, "%15s %63s", kind, hx) != 2)
            continue;
        std::vector<uint8_t> b = unhex(hx);
        std::string k = kind;
        uint8_t out[64];
        size_t n;
        if (k == "rgb8" || k == "rgb8a1" || k == "rgba8")
        {
            uint8_t rgb[16 * 3], a[16];
            memset(a, 255, sizeof a);
            if (k == "rgba8")
            {
                decompressEAC8Block(a, get128BitBlockStart(b.data(), 0));
                decompressETC2Block(rgb, get128BitBlockEnd(b.data(), 0), nullptr, false);
            }
            else
                decompressETC2Block(rgb, get64BitBlock(b.data(), 0), k == "rgb8a1" ? a : nullptr, k == "rgb8a1");
            for (int i = 0; i < 16; i++)
            {
                out[i * 4] = rgb[i * 3];
                out[i * 4 + 1] = rgb[i * 3 + 1];
                out[i * 4 + 2] = rgb[i * 3 + 2];
                out[i * 4 + 3] = a[i];
            }
            n = 64;
        }
        else
        {
            uint8_t v[16 * 2];
            decompressEAC11Block(v, get64BitBlock(b.data(), 0), k == "r11s");
            memcpy(out, v, 32);
            n = 32;
        }
        printf("%s %s %016llx\n", kind, hx, (unsigned long long)fnv(out, n));
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
        return 2;
    std::string mode = argv[1];
    if (mode == "astc-gen")
        astcGen();
    else if (mode == "astc-hash")
        astcHash();
    else if (mode == "etc-hash")
        etcHash();
    else
        return 2;
    return 0;
}
