"""Generate guest/Resources/WebMetalTextureFormats.inc: the fields of the A10X format table that the TextureGen2 layout port reads.

The table is AGX::G9::TextureFormatTable::chooseTextureFormatOBJC (0x21c9c1318) in AGXMetalA10 of the iPadOS 17.7.11
(21H461) dyld shared cache. The function's jump table is decoded statically: every case is `adrp x0; add x0; ret`.
Only named fields are emitted, never raw entry bytes.

usage (from profile/): python3 -B -m webmetal_profile.texture_formats <dyld_shared_cache_arm64> <out.inc>
"""
import struct
import sys

from .dyld import Cache

CHOOSE = 0x21c9c1318          # AGX::G9::TextureFormatTable::chooseTextureFormatOBJC(MTLPixelFormat)
BLOCK_DIMS = 0x21c9e4c70      # u32 (width, height) pairs indexed by entry +0x38
CLASS_WORDS = 0x21c9e4668     # u32 indexed by entry +0x20 - 2 (TextureGen2 ctor, this+0x60)
CLASS_WORDS_MASK = 0x187ff    # valid indices of CLASS_WORDS (ctor 0x21c9beda8)
CLASS_WORDS_COUNT = 17        # ctor: `cmp w8, #0x10; b.hi`
COLOR_CONVERSIONS = 0x21c9e03e0   # u32 indexed by MTLTextureDescriptor colorSpaceConversion - 1 (callers)
COLOR_CONVERSIONS_COUNT = 10      # callers: `cmp x8, #0x9; b.hi`
UNSUPPORTED_NAME = '_texFormatUnsupported'


def fail(msg):
    raise SystemExit('texture_formats: ' + msg)


def insn(c, a):
    return struct.unpack('<I', c.read(a, 4))[0]


def adrp(c, a, rd):
    i = insn(c, a)
    if (i & 0x9f000000) != 0x90000000 or (i & 0x1f) != rd:
        fail(f'expected adrp x{rd} at {a:#x}, found {i:#010x}')
    imm = (((i >> 5) & 0x7ffff) << 2) | ((i >> 29) & 3)
    if imm & (1 << 20):
        imm -= 1 << 21
    return (a & ~0xfff) + (imm << 12)


def add_imm(c, a, rd, rn):
    i = insn(c, a)
    if (i & 0xff800000) != 0x91000000 or (i & 0x1f) != rd or ((i >> 5) & 0x1f) != rn:
        fail(f'expected add x{rd}, x{rn}, #imm at {a:#x}, found {i:#010x}')
    imm = (i >> 10) & 0xfff
    return imm << 12 if (i >> 22) & 1 else imm


def adr(c, a, rd):
    i = insn(c, a)
    if (i & 0x9f000000) != 0x10000000 or (i & 0x1f) != rd:
        fail(f'expected adr x{rd} at {a:#x}, found {i:#010x}')
    imm = (((i >> 5) & 0x7ffff) << 2) | ((i >> 29) & 3)
    if imm & (1 << 20):
        imm -= 1 << 21
    return a + imm


def load_x0(c, a, x0=None):
    """A jump-table case returns x0: `adrp x0; add x0, x0, #imm; ret`, or a bare `ret` that returns the value the
    function loaded into x0 before the branch."""
    if insn(c, a) == 0xd65f03c0:
        if x0 is None:
            fail(f'bare ret at {a:#x} without a preset x0')
        return x0
    page = adrp(c, a, 0)
    target = page + add_imm(c, a + 4, 0, 0)
    if insn(c, a + 8) != 0xd65f03c0:
        fail(f'expected ret at {a + 8:#x}')
    return target


def decode_choose(c):
    # sub w8, w0, #0x1 ; cmp w8, #limit ; b.hi default
    if insn(c, CHOOSE) != 0x51000408:
        fail('chooseTextureFormatOBJC does not start with sub w8, w0, #1')
    cmp = insn(c, CHOOSE + 4)
    if (cmp & 0xffc003ff) != 0x7100011f:
        fail('expected cmp w8, #imm')
    limit = (cmp >> 10) & 0xfff
    bhi = insn(c, CHOOSE + 8)
    if (bhi & 0xff00001f) != 0x54000008:
        fail('expected b.hi')
    off = (bhi >> 5) & 0x7ffff
    if off & (1 << 18):
        off -= 1 << 19
    default = load_x0(c, CHOOSE + 8 + off * 4)
    # adrp x0; add x0 (the value a bare `ret` case returns); adrp x9; add x9 (table); adr x10 (base);
    # ldrh w11, [x9, x8, lsl #1]
    preset = adrp(c, CHOOSE + 0xc, 0) + add_imm(c, CHOOSE + 0x10, 0, 0)
    table = adrp(c, CHOOSE + 0x14, 9) + add_imm(c, CHOOSE + 0x18, 9, 9)
    base = adr(c, CHOOSE + 0x1c, 10)
    if insn(c, CHOOSE + 0x20) != 0x7868792b:
        fail('expected ldrh w11, [x9, x8, lsl #1]')
    return limit, default, table, base, preset


def main():
    cache_path, out_path = sys.argv[1], sys.argv[2]
    c = Cache(cache_path)
    limit, default, table, base, preset = decode_choose(c)
    sym = c.symbolize(default)
    if not sym or sym[1] != UNSUPPORTED_NAME or sym[2] != 0:
        fail(f'default case {default:#x} is not {UNSUPPORTED_NAME}')
    entries = {}
    for pf in range(1, limit + 2):           # w0 - 1 <= limit
        v = struct.unpack('<H', c.read(table + 2 * (pf - 1), 2))[0]
        entries[pf] = load_x0(c, base + 4 * v, preset)
    block_dims = []
    rows = []

    def fields(addr):
        e = c.read(addr, 0x60)
        planes = struct.unpack_from('<I', e, 0x18)[0]
        index = struct.unpack_from('<I', e, 0x38)[0]
        return dict(planes=planes,
                    cls=struct.unpack_from('<I', e, 0x20)[0],
                    alignLog2=e[0x30],
                    bytesPerPixel=e[0x31],
                    blockIndex=index,
                    depthStencil=1 if struct.unpack_from('<I', e, 0x3c)[0] & 0xc else 0,
                    chroma=struct.unpack_from('<I', e, 0x48)[0],
                    planesWithoutSurface=struct.unpack_from('<I', e, 0x50)[0],
                    planeBytes=(e[0x58], e[0x59]))

    unsupported = fields(default)
    supported = {}
    for pf, addr in sorted(entries.items()):
        if addr == default:
            continue
        f = fields(addr)
        if f['planes'] > 2:
            fail(f'pixel format {pf}: {f["planes"]} planes; the port models at most 2 twiddled block infos')
        s = c.symbolize(addr)
        f['entry'] = f'{s[1]}+{s[2]:#x}' if s and s[1] else f'{addr:#x}'
        supported[pf] = f
    max_index = max(f['blockIndex'] for f in list(supported.values()) + [unsupported])
    dims = [struct.unpack('<2I', c.read(BLOCK_DIMS + 8 * i, 8)) for i in range(max_index + 1)]
    class_words = struct.unpack(f'<{CLASS_WORDS_COUNT}I', c.read(CLASS_WORDS, 4 * CLASS_WORDS_COUNT))
    conversions = struct.unpack(f'<{COLOR_CONVERSIONS_COUNT}I', c.read(COLOR_CONVERSIONS, 4 * COLOR_CONVERSIONS_COUNT))

    def row(pf, f):
        bw, bh = dims[f['blockIndex']]
        return (f'    {{{pf}, {f["planes"]}, {f["cls"]}, {f["alignLog2"]}, {f["bytesPerPixel"]}, {bw}, {bh}, '
                f'{"true" if f["depthStencil"] else "false"}, {f["chroma"]}, {f["planesWithoutSurface"]}, '
                f'{{{f["planeBytes"][0]}, {f["planeBytes"][1]}}}}},')

    out = []
    out.append('// Generated by profile/webmetal_profile/texture_formats.py from AGXMetalA10 (iPadOS 17.7.11, 21H461). Do not edit.')
    out.append('// Source: AGX::G9::TextureFormatTable::chooseTextureFormatOBJC (0x21c9c1318); entry fields +0x18 planes,')
    out.append('// +0x20 class, +0x30 alignment log2, +0x31 bytes per pixel, +0x38 block-dimension index (resolved through')
    out.append('// 0x21c9e4c70), +0x3c & 0xc depth/stencil, +0x48 chroma subsampling, +0x50 planes usable without an')
    out.append('// IOSurface, +0x58/+0x59 bytes per block of planes 0 and 1.')
    out.append(f'// chooseTextureFormatOBJC maps pixel formats 1..{limit + 1} through its table; every other value, and every')
    out.append(f'// format not listed here, maps to {UNSUPPORTED_NAME}.')
    out.append('')
    out.append(f'constexpr uint32_t kChooseTextureFormatLimit = {limit};  // `cmp w8, #{limit:#x}` on pixelFormat - 1')
    out.append('')
    out.append('// _texFormatUnsupported (0x21c9e4c10)')
    out.append('constexpr FormatInfo kUnsupportedFormat = ' + row(0, unsupported)[4:-1] + ';')
    out.append('')
    out.append('constexpr FormatInfo kFormats[] = {')
    for pf, f in sorted(supported.items()):
        out.append(row(pf, f) + f'  // {f["entry"]}')
    out.append('};')
    out.append('')
    out.append('// 0x21c9e4668: TextureGen2 this+0x60, indexed by entry class - 2 where bit (class - 2) of 0x187ff is set')
    out.append(f'constexpr uint32_t kClassWordMask = {CLASS_WORDS_MASK:#x};')
    out.append('constexpr uint32_t kClassWords[' + str(CLASS_WORDS_COUNT) + '] = {' + ', '.join(map(str, class_words)) + '};')
    out.append('')
    out.append('// 0x21c9e03e0: eAGXColorSpaceConversion for MTLTextureDescriptor colorSpaceConversion 1..10 (callers of the ctor)')
    out.append('constexpr uint32_t kColorSpaceConversions[' + str(COLOR_CONVERSIONS_COUNT) + '] = {' + ', '.join(map(str, conversions)) + '};')
    out.append('')
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(out))
    print(f'{len(supported)} supported formats, block-dimension indices 0..{max_index}, written to {out_path}')


if __name__ == '__main__':
    main()
